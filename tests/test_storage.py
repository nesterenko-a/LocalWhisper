import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from storage import atomic_json
from jobs import JobStore


class StorageTests(unittest.TestCase):
    def test_transient_replace_failures_retry_without_damaging_existing_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "job.json"
            atomic_json(path, {"version": 1})
            original_replace = Path.replace
            calls = []

            def replace(source, target):
                calls.append(source)
                if len(calls) < 3:
                    self.assertEqual(json.loads(path.read_text()), {"version": 1})
                    raise PermissionError("temporarily locked")
                return original_replace(source, target)

            with patch.object(Path, "replace", replace), patch("storage.time.sleep"):
                atomic_json(path, {"version": 2})
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads(path.read_text()), {"version": 2})
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_persistent_save_error_does_not_abort_job_and_background_retry_recovers(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.wav"
            source.touch()
            entered, release = threading.Event(), threading.Event()

            def engine(*args, progress, on_partial):
                entered.set()
                release.wait(5)
                on_partial("recognized text")
                return "recognized text", [], "done"

            store = JobStore(Path(folder) / "jobs", engine)
            task = store.submit(source, [])
            self.assertTrue(entered.wait(5))
            with patch("jobs.atomic_json", side_effect=PermissionError("locked")), self.assertLogs(level="WARNING"):
                release.set()
                store.queue.join()
                record = store.snapshot(task)[1]
                self.assertEqual(record["state"], "done")
                self.assertEqual(record["text"], "recognized text")
                self.assertIn("persistence_error", record)
            deadline = time.monotonic() + 5
            while store.snapshot(task)[1].get("persistence_error") and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertNotIn("persistence_error", store.snapshot(task)[1])
            saved = json.loads((store.root / task / "job.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["state"], "done")
            self.assertEqual(saved["text"], "recognized text")

    @unittest.skipUnless(os.name == "nt", "Windows sharing locks")
    def test_real_windows_file_lock(self):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "job.json"
            atomic_json(path, {"version": 1})
            handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            errors = []

            def save():
                try:
                    atomic_json(path, {"version": 2})
                except Exception as exc:
                    errors.append(exc)

            worker = threading.Thread(target=save)
            worker.start()
            try:
                time.sleep(.15)
                self.assertTrue(worker.is_alive())
            finally:
                kernel.CloseHandle(handle)
                worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(json.loads(path.read_text()), {"version": 2})
