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
