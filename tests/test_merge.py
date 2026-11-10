import json
from pathlib import Path
import tempfile
import unittest

from jobs import JobStore
from speakers import export_transcript


class MergeTests(unittest.TestCase):
    def make_store(self, root):
        output = root / "output"
        output.mkdir()
        rows = [dict(start=0, end=1, speaker="Спикер 3", text="Привет."),
                dict(start=1, end=2, speaker="Спикер 10", text="Продолжим."),
                dict(start=2, end=3, speaker="Спикер 4", text="Хорошо."),
                dict(start=3, end=4, speaker="Спикер 10", text="До встречи.")]
        text, files = export_transcript(output, rows, {"Спикер 3": "Алексей", "Спикер 10": "Ошибочное имя"})
        jobs = root / "jobs"
        (jobs / "task").mkdir(parents=True)
        (jobs / "task/job.json").write_text(json.dumps(dict(id="task", state="done", text=text, files=files)), encoding="utf-8")
        return JobStore(jobs, None), files

    def test_merge_updates_every_export_preserves_target_name_times_and_history(self):
        with tempfile.TemporaryDirectory() as folder:
            store, files = self.make_store(Path(folder))
            original = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
            store.merge_speakers("task", ["Спикер 10"], "Спикер 3")
            for file in files:
                text = Path(file).read_text(encoding="utf-8")
                self.assertNotIn("Спикер 10", text)
                self.assertNotIn("Ошибочное имя", text)
                self.assertIn("Алексей", text)
            data = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
            self.assertEqual([r["speaker"] for r in data["rows"]], ["Спикер 3", "Спикер 3", "Спикер 4", "Спикер 3"])
            self.assertEqual([(r["start"], r["end"], r["text"]) for r in data["rows"]],
                             [(r["start"], r["end"], r["text"]) for r in original["rows"]])
            self.assertIn("Спикер 3", store.formatted_text("task"))
            self.assertNotIn("Спикер 10", store.formatted_text("task"))
            reloaded = JobStore(store.root, None)
            self.assertNotIn("Спикер 10", reloaded.snapshot()[1]["text"])
            reloaded.undo_speaker_merge("task")
            restored = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
            self.assertEqual(restored, original)
            self.assertIn("Спикер 10", reloaded.snapshot()[1]["text"])

    def test_merge_multiple_sources_and_reject_invalid_input_without_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            store, files = self.make_store(Path(folder))
            before = Path(files[-1]).read_bytes()
            for sources, target in [([], "Спикер 3"), (["Спикер 3"], "Спикер 3"), (["Спикер 99"], "Спикер 3"), (["Спикер 10"], None)]:
                with self.assertRaises(ValueError):
                    store.merge_speakers("task", sources, target)
                self.assertEqual(Path(files[-1]).read_bytes(), before)
            store.merge_speakers("task", ["Спикер 10", "Спикер 4"], "Спикер 3")
            self.assertEqual([label for _, label in store.speaker_choices("task")], ["Спикер 3"])

    def test_merging_unfinished_task_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            store, _ = self.make_store(Path(folder))
            store.records["task"]["state"] = "running"
            with self.assertRaises(ValueError):
                store.merge_speakers("task", ["Спикер 10"], "Спикер 3")
