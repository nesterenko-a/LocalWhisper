"""Local Whisper transcription UI. Run with start.bat."""
import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from datetime import datetime
from pathlib import Path
import threading
import uuid

import gradio as gr
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUT = ROOT / "output"
LOCK = threading.Lock()
cached_key = None
cached_model = None


def timestamp(seconds, separator=","):
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{ms:03}"


def transcribe(source, size, language, device, offline, prompt, progress=gr.Progress()):
    global cached_key, cached_model
    if not source or not Path(source).is_file():
        raise gr.Error("Загрузите файл или запишите голос перед распознаванием.")
    target = "cuda" if device == "NVIDIA GPU (CUDA)" else "cpu"
    key = (size, target, offline)
    with LOCK:
        try:
            if cached_key != key:
                progress(0, desc="Загрузка модели… При первом использовании требуется интернет.")
                cached_model = None
                cached_key = None
                cached_model = WhisperModel(
                    size, device=target,
                    compute_type="float16" if target == "cuda" else "int8",
                    download_root=str(MODELS), local_files_only=offline,
                )
                cached_key = key
            progress(0, desc="Распознавание речи…")
            segments, info = cached_model.transcribe(
                source, language=None if language == "Авто" else {"Русский": "ru", "English": "en"}[language],
                beam_size=5, vad_filter=True, initial_prompt=prompt.strip() or None,
            )
            rows = []
            for segment in segments:
                rows.append((segment.start, segment.end, segment.text.strip()))
                progress(min(segment.end / max(info.duration, 1), 1), desc=f"Обработано {segment.end:.0f} из {info.duration:.0f} с")
            text = "\n".join(row[2] for row in rows)
            OUTPUT.mkdir(exist_ok=True)
            folder = OUTPUT / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + uuid.uuid4().hex[:8])
            folder.mkdir()
            txt, srt, vtt = [folder / f"transcript.{ext}" for ext in ("txt", "srt", "vtt")]
            txt.write_text(text, encoding="utf-8")
            srt.write_text("\n\n".join(f"{i}\n{timestamp(a)} --> {timestamp(b)}\n{t}" for i, (a, b, t) in enumerate(rows, 1)) + "\n", encoding="utf-8")
            vtt.write_text("WEBVTT\n\n" + "\n\n".join(f"{timestamp(a, '.')} --> {timestamp(b, '.')}\n{t}" for a, b, t in rows) + "\n", encoding="utf-8")
            status = f"Готово • язык: {info.language} • длительность: {info.duration:.1f} с • {size} / {target}"
            if not rows:
                status += " • Речь не обнаружена"
            return text, [str(txt), str(srt), str(vtt)], status
        except Exception as exc:
            import logging
            logging.exception("Transcription failed")
            hint = ""
            if target == "cuda":
                hint = " Проверьте совместимость NVIDIA GPU, драйвер, CUDA 12 и cuDNN 9 либо выберите CPU."
            elif offline:
                hint = " Для первой загрузки этой модели снимите флажок «Только офлайн»."
            raise gr.Error(f"Не удалось распознать запись: {exc}.{hint}") from exc


def build_ui():
    with gr.Blocks(title="Локальный Whisper") as demo:
        gr.Markdown("# Локальный Whisper\nАудио и видео в текст. Распознавание выполняется на этом компьютере.")
        with gr.Row():
            size = gr.Dropdown(["tiny", "base", "small", "medium", "large-v3", "turbo"], value="small", label="Модель", info="tiny/base — быстрее; small — баланс; large-v3 — точнее и требовательнее")
            language = gr.Dropdown(["Авто", "Русский", "English"], value="Русский", label="Язык речи")
            device = gr.Dropdown(["CPU", "NVIDIA GPU (CUDA)"], value="CPU", label="Устройство", info="Для GPU нужны совместимая NVIDIA, CUDA 12 и cuDNN 9")
        with gr.Accordion("Дополнительные настройки", open=False):
            offline = gr.Checkbox(False, label="Только офлайн — использовать уже скачанные модели")
            prompt = gr.Textbox(label="Термины и имена для подсказки", placeholder="Например: Kubernetes, Алексей, название компании")
            gr.Markdown("Модели хранятся в `models`, результаты — в `output` рядом с приложением. Выбранное устройство применяется к следующему распознаванию.")
        with gr.Tabs():
            with gr.Tab("Аудио / видео файл"):
                source = gr.File(label="Перетащите аудио или видео", type="filepath", file_count="single")
                file_button = gr.Button("Распознать файл", variant="primary")
            with gr.Tab("Микрофон"):
                microphone = gr.Audio(sources=["microphone"], type="filepath", label="Запишите речь и остановите запись")
                mic_button = gr.Button("Распознать запись", variant="primary")
        status = gr.Textbox(label="Статус", value="Выберите файл или запишите речь", interactive=False)
        result = gr.Textbox(label="Текст", lines=14, buttons=["copy"])
        downloads = gr.File(label="Скачать TXT / SRT / VTT", file_count="multiple", interactive=False)
        for button, audio in [(file_button, source), (mic_button, microphone)]:
            button.click(transcribe, [audio, size, language, device, offline, prompt], [result, downloads, status], concurrency_limit=1, concurrency_id="whisper")
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
