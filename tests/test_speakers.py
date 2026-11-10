import json
from pathlib import Path
import tempfile
import unittest

from speakers import align_words, infer_names, export_transcript, parse_aliases
from jobs import JobStore


class SpeakerTests(unittest.TestCase):
    def test_word_level_changes_and_unknown_gaps(self):
        rows = align_words(
            [(0, 0.5, " Привет."), (0.6, 1, " Да?"), (1.2, 1.5, " Слушаю."), (2, 2.2, " Шум."), (3, 3.2, " Хорошо.")],
            [(0, 0.55, "Спикер 1"), (0.55, 1.8, "Спикер 2"), (3, 4, "Спикер 1")],
        )
        self.assertEqual([r["speaker"] for r in rows], ["Спикер 1", "Спикер 2", "Спикер неизвестен", "Спикер 1"])
        self.assertEqual(rows[1]["text"], "Да? Слушаю.")

    def test_largest_overlap_wins(self):
        rows = align_words([(0, 1, " слово")], [(0, .3, "Спикер 1"), (.2, 1, "Спикер 2")])
        self.assertEqual(rows[0]["speaker"], "Спикер 2")

    def test_zero_duration_word_uses_speaker_at_timestamp(self):
        rows = align_words([(1, 1, " We"), (1, 2, " agree.")], [(0.5, 3, "Спикер 1")])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["speaker"], "Спикер 1")
        self.assertEqual(rows[0]["text"], "We agree.")

    def test_self_introduction_but_not_address_assigns_name(self):
        aliases, hints = infer_names([
            dict(speaker="Спикер 1", text="Меня зовут Алексей."),
            dict(speaker="Спикер 2", text="Иван, что думаешь?"),
            dict(speaker="Спикер 1", text="Руководитель проекта, ответьте."),
        ])
        self.assertEqual(aliases, {"Спикер 1": "Алексей"})
        self.assertTrue(any("Иван" in hint for hint in hints))
        self.assertTrue(any("должности" in hint for hint in hints))

    def test_conflicting_introductions_do_not_assign_name(self):
        aliases, _ = infer_names([dict(speaker="Спикер 1", text="Меня зовут Иван."), dict(speaker="Спикер 1", text="Меня зовут Пётр.")])
        self.assertEqual(aliases, {})

    def test_rename_updates_all_exports_and_survives_reconnect(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "result"
            output.mkdir()
            rows = [dict(start=0, end=1, speaker="Спикер 1", text="Привет."), dict(start=1, end=2, speaker="Спикер 2", text="Здравствуйте.")]
            text, files = export_transcript(output, rows)
            jobs = root / "jobs"
            (jobs / "example").mkdir(parents=True)
            record = dict(id="example", state="done", text=text, files=files)
            (jobs / "example" / "job.json").write_text(json.dumps(record), encoding="utf-8")
            store = JobStore(jobs, None)
            store.rename_speakers("example", "Спикер 1 = Алексей\nСпикер 2 = Директор")
            for path in files:
                self.assertIn("Алексей", Path(path).read_text(encoding="utf-8"))
            self.assertIn("Директор", store.snapshot("example")[1]["text"])
            reloaded = JobStore(jobs, None)
            self.assertIn("Алексей", reloaded.snapshot()[1]["text"])
            store.rename_speakers("example", "Спикер 1 =")
            self.assertIn("Спикер 1: Привет.", store.snapshot()[1]["text"])

    def test_unknown_speaker_mapping_rejected(self):
        with self.assertRaises(ValueError):
            parse_aliases("Спикер 99 = Иван", {"Спикер 1"})
