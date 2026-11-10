import tempfile
import threading
import unittest
import time
from unittest.mock import patch
from pathlib import Path

from jobs import JobStore


def process_engine(source, settings, messages):
    if settings[0] == "slow":
        messages.put(("text", "saved partial text"))
        time.sleep(30)
    messages.put(("result", ("complete", [], "done")))


class JobStoreTests(unittest.TestCase):
    def wait_state(self, store, identifier, states):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            record = store.snapshot(identifier)[1]
            if record["state"] in states:
                return record
            time.sleep(.05)
        self.fail(f"Timed out waiting for {states}: {record}")

    def test_process_cancel_preserves_text_skips_queued_job_and_runs_next(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "voice.wav"
            source.touch()
            with patch("jobs._transcribe_process", process_engine):
                store = JobStore(Path(folder) / "jobs", None, process_jobs=True)
                running = store.submit(source, ["slow"])
                queued = store.submit(source, ["fast"])
                store.cancel(queued)
                deadline = time.monotonic() + 15
                while not store.snapshot(running)[1]["text"] and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertEqual(store.snapshot(running)[1]["text"], "saved partial text")
                store.cancel(running)
                cancelled = self.wait_state(store, running, {"cancelled"})
                self.assertEqual(cancelled["text"], "saved partial text")
                self.assertEqual(Path(cancelled["files"][0]).read_text(encoding="utf-8"), "saved partial text")
                self.assertEqual(store.snapshot(queued)[1]["state"], "cancelled")
                next_job = store.submit(source, ["fast"])
                self.wait_state(store, next_job, {"done"})
                store.cancel(next_job)  # Completed results must remain intact.
                self.assertEqual(store.snapshot(next_job)[1]["state"], "done")
                store.queue.join()

    def test_reconnect_progress_completion_and_restart_history(self):
        entered, release = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "voice.wav"
            source.write_bytes(b"recording")

            def engine(path, *settings, progress, on_partial):
                self.assertEqual(Path(path).read_bytes(), b"recording")
                progress(0.5, desc="halfway")
                on_partial(dict(text="partial text", preview_data=dict(rows=[dict(start=0, end=1, speaker="Спикер 1", text="partial text")], aliases={}, hints=[])))
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test timeout")
                return "complete text", [], "done"

            root = Path(folder) / "jobs"
            store = JobStore(root, engine)
            identifier = store.submit(source, ["tiny", "Авто", "CPU", True, None])
            try:
                self.assertTrue(entered.wait(5))
                # A new browser has no selected job or previous session state.
                records, reconnected = store.snapshot()
                self.assertEqual(reconnected["id"], identifier)
                self.assertEqual(reconnected["state"], "running")
                self.assertEqual(reconnected["text"], "partial text")
                self.assertEqual(reconnected["progress"], 0.5)
                self.assertIn("Спикер 1", store.formatted_text(identifier))
                source.unlink()  # Browser upload may disappear; server copy remains.
            finally:
                release.set()
                store.queue.join()
            self.assertEqual(store.snapshot(identifier)[1]["state"], "done")
            restarted = JobStore(root, engine)
            self.assertEqual(restarted.snapshot()[1]["text"], "complete text")

    def test_failed_job_does_not_block_next_job(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "voice.wav"
            source.touch()

            def engine(path, fail, progress, on_partial):
                if fail:
                    raise RuntimeError("bad audio")
                return "ok", [], "done"

            store = JobStore(Path(folder) / "jobs", engine)
            with self.assertLogs(level="ERROR"):
                failed = store.submit(source, [True])
                success = store.submit(source, [False])
                store.queue.join()
            self.assertEqual(store.snapshot(failed)[1]["state"], "error")
            self.assertEqual(store.snapshot(success)[1]["state"], "done")

    def test_server_restart_marks_unfinished_job_interrupted(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / "example"
            task.mkdir()
            (task / "job.json").write_text(json.dumps(dict(id="example", state="running", text="partial")), encoding="utf-8")
            store = JobStore(root, None)
            record = store.snapshot()[1]
            self.assertEqual(record["state"], "interrupted")
            self.assertEqual(record["text"], "partial")
