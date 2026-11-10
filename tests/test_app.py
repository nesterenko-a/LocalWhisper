import unittest
import tempfile
import wave
from types import SimpleNamespace
from unittest.mock import patch

import app


class AppTests(unittest.TestCase):
    def test_rename_controls_disabled_until_final_result(self):
        store = unittest.mock.Mock()
        store.snapshot.return_value = ([], dict(id="job", state="running"))
        store.can_rename_speakers.return_value = False
        with patch.object(app, "get_jobs", return_value=store):
            message, button = app.speaker_controls("job")
            self.assertFalse(button["interactive"])
            self.assertIn("финальное", message)
            store.snapshot.return_value = ([], dict(id="job", state="done"))
            store.can_rename_speakers.return_value = True
            store.speaker_details.return_value = "Спикер 1"
            _, button = app.speaker_controls("job")
            self.assertTrue(button["interactive"])

    def test_rename_validation_keeps_text_and_reports_without_exception(self):
        store = unittest.mock.Mock()
        store.rename_speakers.side_effect = ValueError("Выберите завершённую задачу")
        with patch.object(app, "get_jobs", return_value=store):
            result = app.rename_speakers(None, "Спикер 1 = Алексей")
        self.assertEqual(len(result), 7)
        self.assertTrue(all(item == app.gr.skip() for item in result[:6]))
        self.assertIn("Имена не применены", result[-1])

    def test_speakers_arrive_before_transcription_finishes(self):
        with tempfile.TemporaryDirectory() as folder:
            source = app.Path(folder) / "recording.wav"
            source.touch()
            partials, events = [], []

            def segments():
                yield SimpleNamespace(start=0, end=1, text="Привет.", words=[SimpleNamespace(start=0, end=1, word=" Привет.")])
                self.assertEqual(partials[0]["preview_data"]["rows"][0]["speaker"], "Спикер 1")
                self.assertNotIn("Спикер 2", partials[0]["text"])
                yield SimpleNamespace(start=2, end=3, text="Здравствуйте.", words=[SimpleNamespace(start=2, end=3, word=" Здравствуйте.")])

            model = unittest.mock.Mock()
            model.transcribe.side_effect = lambda *args, **kwargs: (events.append("transcribe") or segments(), SimpleNamespace(language="ru", duration=3))

            def voices(*args):
                events.append("diarize")
                return [(0, 1, "Спикер 1"), (2, 3, "Спикер 2")]

            import numpy as np
            analyzer = unittest.mock.Mock()
            analyzer.analyze.side_effect = voices
            analyzer.refine.side_effect = voices
            from buffered import recognize

            with patch.object(app, "WhisperModel", return_value=model), \
                 patch.object(app, "cached_key", None), patch.object(app, "cached_model", None), \
                 patch.object(app, "ensure_models", return_value=("seg", "emb")), \
                 patch("faster_whisper.audio.decode_audio", return_value=np.zeros(48000, dtype=np.float32)), \
                 patch.object(app, "recognize_buffers", side_effect=lambda *args: recognize(*args, analyzer=analyzer)), \
                 patch.object(app, "OUTPUT", app.Path(folder) / "results"):
                text, files, _ = app.transcribe(str(source), "tiny", "Русский", "CPU", True, None, True, 2, on_partial=partials.append)
            self.assertEqual(events, ["diarize", "transcribe", "diarize"])
            self.assertEqual(len(partials[0]["preview_data"]["rows"]), 1)
            self.assertIn("Спикер 2", partials[-1]["text"])
            self.assertEqual(text, partials[-1]["text"])

    def test_subtitle_timestamps_round_and_carry(self):
        self.assertEqual(app.timestamp(59.9996), "00:01:00,000")
        self.assertEqual(app.timestamp(3661.234, "."), "01:01:01.234")
        self.assertEqual(app.timestamp(-1), "00:00:00,000")

    def test_missing_input_does_not_load_model(self):
        with patch.object(app, "WhisperModel") as model:
            with self.assertRaises(app.gr.Error):
                app.transcribe(None, "tiny", "Русский", "CPU", True, "")
            model.assert_not_called()

    def test_interface_builds(self):
        self.assertIsInstance(app.build_ui(), app.gr.Blocks)

    def test_audio_decoder_accepts_wav(self):
        from faster_whisper.audio import decode_audio
        with tempfile.TemporaryDirectory() as folder:
            path = str(app.Path(folder) / "silence.wav")
            with wave.open(path, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\0\0" * 16000)
            self.assertEqual(len(decode_audio(path)), 16000)

    def test_empty_browser_prompt_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            source = app.Path(folder) / "recording.wav"
            source.touch()
            model = unittest.mock.Mock()
            model.transcribe.return_value = (
                iter([SimpleNamespace(start=0, end=1, text=" Привет. ")]),
                SimpleNamespace(language="ru", duration=1),
            )
            with patch.object(app, "WhisperModel", return_value=model), \
                 patch.object(app, "cached_key", None), \
                 patch.object(app, "cached_model", None), \
                 patch.object(app, "OUTPUT", app.Path(folder) / "results"):
                text, files, status = app.transcribe(str(source), "tiny", "Русский", "CPU", True, None)
                self.assertEqual(text, "Привет.")
                self.assertIsNone(model.transcribe.call_args.kwargs["initial_prompt"])
                self.assertEqual(len(files), 4)
                self.assertTrue(all(app.Path(path).is_file() for path in files))

    def test_failure_preserves_previous_result(self):
        with tempfile.TemporaryDirectory() as folder:
            source = app.Path(folder) / "recording.wav"
            source.touch()
            with patch.object(app, "WhisperModel", side_effect=RuntimeError("test failure")), \
                 patch.object(app, "cached_key", None), \
                 patch.object(app, "cached_model", None), \
                 self.assertLogs(level="ERROR"):
                text, files, status = app.transcribe(str(source), "tiny", "Русский", "CPU", True, None)
            self.assertEqual(text, app.gr.skip())
            self.assertEqual(files, app.gr.skip())
            self.assertIn("test failure", status)


if __name__ == "__main__":
    unittest.main()
