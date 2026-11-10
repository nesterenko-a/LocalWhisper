"""Server-owned jobs with durable metadata, independent of browser sessions."""
from datetime import datetime
import json
from pathlib import Path
import queue
import shutil
import threading
import uuid
import multiprocessing
import time
import os
import subprocess
import signal
from storage import atomic_json


class JobCancelled(Exception):
    pass


def partial_changes(value):
    if isinstance(value, dict):
        return {"text": value["text"], "preview_data": value["preview_data"]}
    return {"text": value}


def _transcribe_process(source, settings, messages):
    """Heavy native inference lives in a killable child, not the web server."""
    from app import transcribe
    if os.name != "nt":
        os.setsid()
    try:
        result = transcribe(source, *settings,
                            progress=lambda value, desc="": messages.put(("progress", value, desc)),
                            on_partial=lambda text: messages.put(("text", text)))
        messages.put(("result", result))
    except Exception as exc:
        messages.put(("error", str(exc)))


class JobStore:
    def __init__(self, root, transcribe, process_jobs=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.transcribe = transcribe
        self.process_jobs = process_jobs
        self.lock = threading.RLock()
        self.records = {}
        self.dirty = set()
        self.save_event = threading.Event()
        self.queue = queue.Queue()
        for path in sorted(self.root.glob("*/job.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                self.records[record["id"]] = record
                if record["state"] in ("queued", "running", "cancelling"):
                    record.update(state="interrupted", status="Прервано остановкой сервера. Запустите задачу заново.")
                    self._save(record)
            except (OSError, ValueError, KeyError):
                import logging
                logging.exception("Cannot load job %s", path)
        threading.Thread(target=self._worker, daemon=True, name="whisper-jobs").start()
        threading.Thread(target=self._retry_saves, daemon=True, name="whisper-save-retry").start()

    def _save(self, record, strict=False):
        with self.lock:
            record.pop("persistence_error", None)
            try:
                atomic_json(self.root / record["id"] / "job.json", record)
            except OSError as exc:
                if strict:
                    raise
                record["persistence_error"] = str(exc)
                self.dirty.add(record["id"])
                self.save_event.set()
                import logging
                logging.warning("Job %s remains in memory; saving will be retried: %s", record["id"], exc)
                return False
            self.dirty.discard(record["id"])
            return True

    def _retry_saves(self):
        while True:
            self.save_event.wait()
            time.sleep(1)
            with self.lock:
                for identifier in list(self.dirty):
                    self._save(self.records[identifier])
                if not self.dirty:
                    self.save_event.clear()

    def submit(self, source, settings):
        if not source or not Path(source).is_file():
            raise ValueError("Загрузите файл или запишите голос перед распознаванием.")
        identifier = uuid.uuid4().hex
        folder = self.root / identifier
        folder.mkdir()
        original = Path(source)
        destination = folder / ("input" + original.suffix)
        shutil.copy2(original, destination)
        record = dict(id=identifier, name=original.name, created=datetime.now().isoformat(timespec="seconds"),
                      source=str(destination.resolve()), settings=list(settings), state="queued",
                      status="В очереди", progress=0, text="", files=[])
        with self.lock:
            self.records[identifier] = record
            try:
                self._save(record, strict=True)
            except OSError:
                del self.records[identifier]
                raise
            self.queue.put(identifier)
        return identifier

    def snapshot(self, identifier=None):
        with self.lock:
            records = sorted(self.records.values(), key=lambda record: record.get("created", ""))
            selected = self.records.get(identifier) or (records[-1] if records else None)
            # Copies prevent callers from modifying the worker's live state.
            return json.loads(json.dumps(records)), json.loads(json.dumps(selected))

    def cancel(self, identifier):
        with self.lock:
            record = self.records.get(identifier)
            if not record:
                raise ValueError("Выберите задачу, которую нужно остановить.")
            if record["state"] == "queued":
                record.update(state="cancelled", status="Отменено до начала распознавания")
            elif record["state"] == "running":
                record.update(state="cancelling", status="Останавливается…")
            else:
                return
            self._save(record)

    def _check_cancelled(self, record):
        with self.lock:
            if record["state"] in ("cancelling", "cancelled"):
                raise JobCancelled()

    def _execute_process(self, record, update):
        context = multiprocessing.get_context("spawn")
        messages = context.Queue()
        process = context.Process(target=_transcribe_process,
                                  args=(record["source"], record["settings"], messages), daemon=False)
        process.start()
        try:
            while True:
                self._check_cancelled(record)
                try:
                    message = messages.get(timeout=0.2)
                except queue.Empty:
                    if not process.is_alive():
                        raise RuntimeError(f"Процесс распознавания завершился без результата (код {process.exitcode}).")
                    continue
                kind, *values = message
                if kind == "progress":
                    update(progress=float(values[0]), status=values[1])
                elif kind == "text":
                    update(**partial_changes(values[0]))
                elif kind == "result":
                    return values[0]
                elif kind == "error":
                    raise RuntimeError(values[0])
        finally:
            if process.is_alive():
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                else:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                if process.is_alive():
                    process.terminate()
            process.join()
            messages.close()

    def _save_cancelled(self, record):
        with self.lock:
            files = record.get("files", [])
            if record["text"]:
                partial = self.root / record["id"] / "partial.txt"
                partial.write_text(record["text"], encoding="utf-8")
                files = [str(partial.resolve())]
            status = "Остановлено пользователем."
            status += " Сохранён частичный результат." if record["text"] else " Текст ещё не был получен."
            record.update(state="cancelled", status=status, files=files)
            self._save(record)

    def _metadata(self, identifier):
        record = self.records.get(identifier)
        if not record or record["state"] != "done":
            raise ValueError("Выберите завершённую задачу с разделением спикеров.")
        paths = [Path(p) for p in record["files"] if Path(p).suffix == ".json"]
        if not paths:
            raise ValueError("У этой задачи нет данных спикеров. Запустите её с разделением по спикерам.")
        return record, paths[0], json.loads(paths[0].read_text(encoding="utf-8"))

    def speaker_details(self, identifier):
        with self.lock:
            try:
                _, _, data = self._metadata(identifier)
            except (ValueError, OSError):
                return "У этой задачи нет данных спикеров."
            known = list(dict.fromkeys(row["speaker"] for row in data["rows"] if row.get("speaker")))
            if not known:
                return "Задача обработана без разделения по спикерам."
            lines = [f"{label} = {data['aliases'].get(label, '')}" for label in known]
            return "\n".join(lines + [""] + data.get("hints", []))

    def formatted_text(self, identifier):
        from transcript_view import render_transcript
        with self.lock:
            record = self.records.get(identifier)
            data = record.get("preview_data") if record else None
            if record and record["state"] == "done":
                try:
                    _, _, data = self._metadata(identifier)
                except (OSError, ValueError):
                    pass
            return render_transcript(record, data)

    def rename_speakers(self, identifier, mappings):
        from speakers import parse_aliases, export_transcript
        with self.lock:
            record, path, data = self._metadata(identifier)
            known = {row["speaker"] for row in data["rows"] if row.get("speaker")}
            changes = parse_aliases(mappings, known)
            if not changes:
                raise ValueError("Введите хотя бы одну строку, например: Спикер 1 = Алексей")
            data["aliases"].update(changes)
            text, files = export_transcript(path.parent, data["rows"], data["aliases"], data.get("hints", []))
            record.update(text=text, files=files)
            self._save(record)

    def _worker(self):
        while True:
            identifier = self.queue.get()
            try:
                self._run(identifier)
            finally:
                self.queue.task_done()

    def _run(self, identifier):
        last_progress_save = 0.0

        def update(**changes):
            nonlocal last_progress_save
            with self.lock:
                self._check_cancelled(record)
                record.update(changes)
                now = time.monotonic()
                if "state" in changes or "text" in changes or now - last_progress_save >= 1:
                    self._save(record)
                    last_progress_save = now

        def progress(value, desc=""):
            update(progress=float(value), status=desc)

        with self.lock:
            record = self.records[identifier]
            if record["state"] == "cancelled":
                return
        try:
            update(state="running", status="Запуск распознавания…")
            if self.process_jobs:
                text, files, status = self._execute_process(record, update)
            else:
                text, files, status = self.transcribe(
                    record["source"], *record["settings"], progress=progress,
                    on_partial=lambda value: update(**partial_changes(value)),
                )
            if not isinstance(text, str) or not isinstance(files, list):
                raise RuntimeError(status)
            update(state="done", text=text, files=files, status=status, progress=1)
        except JobCancelled:
            self._save_cancelled(record)
        except Exception as exc:
            import logging
            logging.exception("Job %s failed", identifier)
            with self.lock:
                if record["state"] == "cancelling":
                    self._save_cancelled(record)
                else:
                    record.update(state="error", status=str(exc))
                    self._save(record)
