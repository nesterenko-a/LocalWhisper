import unittest
import tempfile
import wave
from unittest.mock import patch

import app


class AppTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
