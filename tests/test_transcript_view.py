import unittest

from transcript_view import render_transcript, voice_style


class TranscriptViewTests(unittest.TestCase):
    def test_speakers_have_stable_colors_and_separate_timestamped_cards(self):
        data = dict(rows=[
            dict(start=0, end=1, speaker="Спикер 1", text="Привет."),
            dict(start=1, end=3, speaker="Спикер 2", text="Здравствуйте."),
            dict(start=3, end=4, speaker="Спикер 1", text="Продолжим."),
        ], aliases={"Спикер 1": "Алексей"})
        rendered = render_transcript(dict(state="done", text="text"), data)
        self.assertEqual(rendered.count('<article role="listitem"'), 3)
        self.assertIn("00:00:01.000 — 00:00:03.000", rendered)
        self.assertIn("Алексей (Спикер 1)", rendered)
        self.assertNotEqual(voice_style("Спикер 1"), voice_style("Спикер 2"))
        self.assertEqual(rendered.count(voice_style("Спикер 1")), 3)  # Legend and two turns.
        self.assertEqual(len({voice_style(f"Спикер {i}") for i in range(1, 31)}), 30)

    def test_transcript_and_names_are_escaped(self):
        rendered = render_transcript(dict(state="done", text="text"), dict(
            rows=[dict(start=0, end=1, speaker="Спикер 1", text='<script>alert(1)</script> & text')],
            aliases={"Спикер 1": '<img src=x onerror="alert(1)">'},
        ))
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("&lt;img", rendered)

    def test_partial_result_without_metadata_is_readable(self):
        rendered = render_transcript(dict(state="cancelled", text="Первая строка.\nВторая строка."))
        self.assertEqual(rendered.count('<article role="listitem"'), 2)
        self.assertIn("Частичный результат", rendered)
