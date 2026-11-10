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
from jobs import JobStore
from speakers import timestamp, ensure_models, diarize, align_words, infer_names, export_transcript

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUT = ROOT / "output"
LOCK = threading.Lock()
cached_key = None
cached_model = None
job_store = None
STORE_LOCK = threading.Lock()


def transcribe(source, size, language, device, offline, prompt, separate_speakers=False, speaker_count=0, progress=gr.Progress(), on_partial=None):
    global cached_key, cached_model
    if not source or not Path(source).is_file():
        raise gr.Error("Загрузите файл или запишите голос перед распознаванием.")
    target = "cuda" if device == "NVIDIA GPU (CUDA)" else "cpu"
    key = (size, target, offline)
    with LOCK:
        try:
            speaker_models = None
            if separate_speakers:
                progress(0, desc="Подготовка моделей спикеров… Первая загрузка требует интернет.")
                speaker_models = ensure_models(MODELS, offline)
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
                beam_size=5, vad_filter=True, initial_prompt=(prompt or "").strip() or None,
                word_timestamps=separate_speakers,
            )
            rows = []
            words = []
            for segment in segments:
                rows.append(dict(start=segment.start, end=segment.end, text=segment.text.strip()))
                if separate_speakers:
                    words.extend((word.start, word.end, word.word) for word in (segment.words or []))
                if on_partial:
                    on_partial("\n".join(row["text"] for row in rows))
                progress(min(segment.end / max(info.duration, 1), 1) * (0.6 if separate_speakers else 1), desc=f"Обработано {segment.end:.0f} из {info.duration:.0f} с")
            aliases, hints = {}, []
            if separate_speakers and rows:
                from faster_whisper.audio import decode_audio
                progress(0.6, desc="Разделение по голосам на CPU…")
                turns = diarize(decode_audio(source, sampling_rate=16000), speaker_models, speaker_count, progress)
                if not words:
                    raise RuntimeError("Whisper не вернул временные отметки слов для разделения спикеров.")
                rows = align_words(words, turns)
                aliases, hints = infer_names(rows)
            OUTPUT.mkdir(exist_ok=True)
            folder = OUTPUT / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + uuid.uuid4().hex[:8])
            folder.mkdir()
            text, files = export_transcript(folder, rows, aliases, hints)
            status = f"Готово • язык: {info.language} • длительность: {info.duration:.1f} с • {size} / {target}"
            if not rows:
                status += " • Речь не обнаружена"
            elif separate_speakers:
                status += " • Спикеры разделены; проверьте имена и границы реплик"
            return text, files, status
        except Exception as exc:
            import logging
            logging.exception("Transcription failed")
            hint = ""
            if target == "cuda":
                hint = " Проверьте совместимость NVIDIA GPU, драйвер, CUDA 12 и cuDNN 9 либо выберите CPU."
            elif offline:
                hint = " Для первой загрузки этой модели снимите флажок «Только офлайн»."
            return gr.skip(), gr.skip(), f"Не удалось распознать запись: {exc}.{hint}"


def get_jobs():
    global job_store
    with STORE_LOCK:
        if job_store is None:
            job_store = JobStore(ROOT / "jobs", transcribe, process_jobs=True)
        return job_store


def refresh_job(selected=None):
    records, record = get_jobs().snapshot(selected)
    labels = {"queued": "В очереди", "running": "В работе", "cancelling": "Останавливается", "cancelled": "Остановлено", "done": "Готово", "error": "Ошибка", "interrupted": "Прервано"}
    choices = [(f"{r['created'].replace('T', ' ')} · {r['name']} · {labels[r['state']]}", r["id"]) for r in reversed(records)]
    if record is None:
        from transcript_view import render_transcript
        return gr.update(choices=choices, value=None), "Задач пока нет", "", [], render_transcript()
    status = f"{labels[record['state']]} • {record['progress']:.0%} • {record['status']}"
    return gr.update(choices=choices, value=record["id"]), status, record["text"], record["files"], get_jobs().formatted_text(record["id"])


def speaker_details(selected=None):
    _, record = get_jobs().snapshot(selected)
    if not record or record["state"] != "done":
        return "Имена и подсказки появятся после завершения задачи."
    details = get_jobs().speaker_details(record["id"])
    return details


def rename_speakers(selected, mappings):
    try:
        get_jobs().rename_speakers(selected, mappings)
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc)) from exc
    return (*refresh_job(selected), speaker_details(selected))


def submit_job(source, size, language, device, offline, prompt, separate_speakers=False, speaker_count=0):
    try:
        identifier = get_jobs().submit(source, (size, language, device, offline, prompt, separate_speakers, speaker_count))
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    return refresh_job(identifier)


def cancel_job(selected):
    try:
        get_jobs().cancel(selected)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return refresh_job(selected)


def build_ui():
    with gr.Blocks(title="Локальный Whisper") as demo:
        gr.Markdown("# Локальный Whisper\nАудио и видео в текст. Распознавание выполняется на этом компьютере.")
        with gr.Row(equal_height=True):
            with gr.Column(min_width=200):
                size = gr.Dropdown(["tiny", "base", "small", "medium", "large-v3", "turbo"], value="small", label="Модель")
                gr.Markdown("tiny/base — быстрее; small — баланс; large-v3 — точнее и требовательнее")
            with gr.Column(min_width=200):
                language = gr.Dropdown(["Авто", "Русский", "English"], value="Русский", label="Язык речи")
                gr.Markdown("Выберите язык записи или автоопределение.")
            with gr.Column(min_width=200):
                device = gr.Dropdown(["CPU", "NVIDIA GPU (CUDA)"], value="CPU", label="Устройство")
                gr.Markdown("Для GPU нужны совместимая NVIDIA, CUDA 12 и cuDNN 9")
        with gr.Accordion("Дополнительные настройки", open=False):
            offline = gr.Checkbox(False, label="Только офлайн — использовать уже скачанные модели")
            prompt = gr.Textbox(value="", label="Термины и имена для подсказки", placeholder="Например: Kubernetes, Алексей, название компании")
            gr.Markdown("Модели хранятся в `models`, результаты — в `output` рядом с приложением. Выбранное устройство применяется к следующему распознаванию.")
        with gr.Accordion("Спикеры", open=False):
            separate_speakers = gr.Checkbox(False, label="Разделять по спикерам")
            speaker_count = gr.Number(value=0, precision=0, minimum=0, maximum=30, label="Число участников (0 — определить автоматически)")
            gr.Markdown("Разделение голосов работает локально на CPU и увеличивает время обработки. При первом использовании скачаются дополнительные модели. Одновременная речь и похожие голоса могут разделяться неточно.")
        with gr.Tabs():
            with gr.Tab("Аудио / видео файл"):
                source = gr.File(label="Перетащите аудио или видео", type="filepath", file_count="single")
                file_button = gr.Button("Распознать файл", variant="primary")
                file_stop = gr.Button("Остановить распознавание", variant="stop")
            with gr.Tab("Микрофон"):
                microphone = gr.Audio(sources=["microphone"], type="filepath", label="Запишите речь и остановите запись")
                mic_button = gr.Button("Распознать запись", variant="primary")
                mic_stop = gr.Button("Остановить распознавание", variant="stop")
        gr.Markdown("### Задачи\nМожно закрыть вкладку и вернуться позже: задача продолжит работать, пока запущен сервер.")
        gr.Markdown("Кнопка «Остановить распознавание» отменяет задачу, выбранную в истории ниже. Уже полученный текст сохраняется; задача из очереди отменяется до запуска.")
        selected_job = gr.Dropdown(label="Текущая задача и история", choices=[], interactive=True)
        status = gr.Textbox(label="Статус", value="Загрузка задач…", interactive=False)
        gr.Markdown("### Текст")
        formatted = gr.HTML()
        with gr.Accordion("Обычный текст — копирование", open=False):
            result = gr.Textbox(label="Текст без оформления", lines=14, buttons=["copy"], interactive=False)
        downloads = gr.File(label="Скачать TXT / SRT / VTT / JSON", file_count="multiple", interactive=False)
        with gr.Accordion("Имена и должности спикеров", open=False):
            details = gr.Textbox(label="Спикеры и подсказки из разговора", lines=5, interactive=False)
            mappings = gr.Textbox(value="", label="Переименовать спикеров (по одному на строку)", placeholder="Спикер 1 = Алексей\nСпикер 2 = Руководитель проекта", lines=4)
            rename_button = gr.Button("Применить имена к выбранной задаче")
            gr.Markdown("Явное «меня зовут…» используется для имени автоматически. Обращения к собеседнику — только подсказки. Проверьте имена по записи. Ручные правки обновят текст и все файлы; пустое имя после = вернёт номер спикера.")
        for button, audio in [(file_button, source), (mic_button, microphone)]:
            button.click(submit_job, [audio, size, language, device, offline, prompt, separate_speakers, speaker_count], [selected_job, status, result, downloads, formatted], concurrency_limit=1, concurrency_id="submit-job")
        outputs = [selected_job, status, result, downloads, formatted]
        for stop_button in (file_stop, mic_stop):
            stop_button.click(cancel_job, [selected_job], outputs, queue=False)
        demo.load(refresh_job, [selected_job], outputs, queue=False)
        selected_job.input(refresh_job, [selected_job], outputs, queue=False)
        gr.Timer(2).tick(refresh_job, [selected_job], outputs, queue=False, show_progress="hidden")
        demo.load(speaker_details, [selected_job], [details], queue=False)
        gr.Timer(2).tick(speaker_details, [selected_job], [details], queue=False, show_progress="hidden")
        selected_job.input(speaker_details, [selected_job], [details], queue=False)
        rename_button.click(rename_speakers, [selected_job, mappings], [*outputs, details])
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
