"""Server-owned jobs with durable metadata, independent of browser sessions."""
from datetime import datetime
import json
from pathlib import Path
import queue
import shutil
import threading
import uuid


class JobStore:
    def __init__(self, root, transcribe):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.transcribe = transcribe
        self.lock = threading.RLock()
        self.records = {}
        self.queue = queue.Queue()
        for path in sorted(self.root.glob("*/job.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                self.records[record["id"]] = record
                if record["state"] in ("queued", "running"):
                    record.update(state="interrupted", status="Прервано остановкой сервера. Запустите задачу заново.")
                    self._save(record)
            except (OSError, ValueError, KeyError):
                import logging
                logging.exception("Cannot load job %s", path)
        threading.Thread(target=self._worker, daemon=True, name="whisper-jobs").start()

    def _save(self, record):
        path = self.root / record["id"] / "job.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

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
            self._save(record)
            self.queue.put(identifier)
        return identifier

    def snapshot(self, identifier=None):
        with self.lock:
            records = sorted(self.records.values(), key=lambda record: record.get("created", ""))
            selected = self.records.get(identifier) or (records[-1] if records else None)
            # Copies prevent callers from modifying the worker's live state.
            return json.loads(json.dumps(records)), json.loads(json.dumps(selected))

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
        def update(**changes):
            with self.lock:
                record.update(changes)
                self._save(record)

        def progress(value, desc=""):
            update(progress=float(value), status=desc)

        with self.lock:
            record = self.records[identifier]
        try:
            update(state="running", status="Запуск распознавания…")
            text, files, status = self.transcribe(
                record["source"], *record["settings"], progress=progress,
                on_partial=lambda text: update(text=text),
            )
            if not isinstance(text, str) or not isinstance(files, list):
                raise RuntimeError(status)
            update(state="done", text=text, files=files, status=status, progress=1)
        except Exception as exc:
            import logging
            logging.exception("Job %s failed", identifier)
            update(state="error", status=str(exc))
