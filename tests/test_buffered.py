import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from buffered import buffer_seconds, prepared_buffers, recognize, preserve_labels
from speakers import SpeakerTracker


class BufferTests(unittest.TestCase):
    def test_adaptive_buffer_bounds(self):
        self.assertEqual(buffer_seconds(60), 30)
        self.assertEqual(buffer_seconds(1200), 60)
        self.assertEqual(buffer_seconds(10000), 120)

    def test_next_diarization_overlaps_consumption_of_previous_buffer(self):
        second_started, allow_second = threading.Event(), threading.Event()
        calls = []

        def analyze(chunk):
            calls.append(len(chunk))
            if len(calls) == 2:
                second_started.set()
                if not allow_second.wait(5):
                    raise RuntimeError("test timeout")
            return []

        generator = prepared_buffers(np.zeros(60 * 16000), analyze, 30)
        try:
            first = next(generator)
            self.assertEqual(first[:3], (0, 30, 0))
            self.assertTrue(second_started.wait(3))
        finally:
            allow_second.set()
        second = next(generator)
        self.assertEqual(second[:3], (30, 60, 28))
        with self.assertRaises(StopIteration):
            next(generator)
        self.assertEqual(len(calls), 2)

    def test_global_speaker_identity_survives_local_label_permutation(self):
        tracker = SpeakerTracker(maximum=2)
        first = tracker.match({0: [1, 0], 1: [0, 1]})
        second = tracker.match({0: [0, .99], 1: [.99, .01]})
        self.assertEqual(first[0], second[1])
        self.assertEqual(first[1], second[0])
        self.assertEqual(tracker.match({4: [-1, 0]}), {})

    def test_boundary_word_is_not_duplicated_and_timestamps_are_global(self):
        model = Mock()

        def segment(words):
            return SimpleNamespace(text="text", start=0, end=32, words=[SimpleNamespace(start=a, end=b, word=w) for a, b, w in words])

        info = SimpleNamespace(language="ru")
        model.transcribe.side_effect = [
            (iter([segment([(0, 1, " Начало."), (29.8, 30.2, " Граница.")])]), info),
            (iter([segment([(1.8, 2.2, " Граница."), (3, 4, " Конец.")])]), info),
        ]
        analyzer = Mock()
        analyzer.analyze.return_value = [(0, 32, "Спикер 1")]
        analyzer.refine.return_value = [(0, 60, "Спикер 1")]
        partials = []
        rows, _, _, _ = recognize(model, np.zeros(60 * 16000), None, 0, None, None, lambda *a, **k: None, partials.append, analyzer=analyzer)
        text = " ".join(row["text"] for row in rows)
        self.assertEqual(text.count("Граница."), 1)
        self.assertIn("Начало.", text)
        self.assertIn("Конец.", text)
        self.assertAlmostEqual(rows[-1]["end"], 32)
        self.assertNotIn("Граница.", partials[0]["text"])
        self.assertEqual(model.transcribe.call_args_list[1].kwargs["language"], "ru")

    def test_refinement_keeps_ids_by_overlap_and_merges_false_splits(self):
        rows = [dict(start=0, end=10, speaker="Спикер 1"), dict(start=10, end=12, speaker="Спикер 3"),
                dict(start=12, end=20, speaker="Спикер 2")]
        turns = preserve_labels([(0, 12, "Спикер 2"), (12, 20, "Спикер 1")], rows)
        self.assertEqual(turns, [(0, 12, "Спикер 1"), (12, 20, "Спикер 2")])

    def test_final_pass_corrects_provisional_speakers_without_retranscription(self):
        model, analyzer = Mock(), Mock()
        words = [SimpleNamespace(start=0, end=1, word=" Привет."), SimpleNamespace(start=1.5, end=2, word=" Да.")]
        model.transcribe.return_value = (iter([SimpleNamespace(text="Привет. Да.", end=2, words=words)]), SimpleNamespace(language="ru"))
        analyzer.analyze.return_value = [(0, 3, "Спикер 1")]
        analyzer.refine.return_value = [(0, 1.1, "Спикер 1"), (1.1, 3, "Спикер 2")]
        partials, values = [], []
        rows, _, _, _ = recognize(model, np.zeros(3 * 16000), None, 0, None, None,
                                   lambda value, **kwargs: values.append(value), partials.append, analyzer=analyzer)
        self.assertNotIn("Спикер 2", partials[0]["text"])
        self.assertIn("Спикер 2", partials[-1]["text"])
        self.assertEqual([r["speaker"] for r in rows], ["Спикер 1", "Спикер 2"])
        self.assertEqual(model.transcribe.call_count, 1)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
