import tempfile
import threading
import unittest
from pathlib import Path

from jobs import JobStore


class JobStoreTests(unittest.TestCase):
    def test_reconnect_progress_completion_and_restart_history(self):
        entered, release = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "voice.wav"
            source.write_bytes(b"recording")

            def engine(path, *settings, progress, on_partial):
                self.assertEqual(Path(path).read_bytes(), b"recording")
                progress(0.5, desc="halfway")
                on_partial("partial text")
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
