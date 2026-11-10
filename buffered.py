"""Bounded one-buffer lookahead: diarization N+1 runs alongside Whisper N."""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from types import SimpleNamespace

from speakers import BufferedDiarizer, align_words, infer_names, speaker_text

SAMPLE_RATE = 16000
_analyzer = None


def _initialize_analyzer(models, count):
    global _analyzer
    _analyzer = BufferedDiarizer(models, count)


def _analyze(audio):
    return _analyzer.analyze(audio)


def _refine(audio):
    return _analyzer.refine(audio)


def preserve_labels(final_turns, provisional_rows):
    """Prefer recording-wide IDs with the largest overlap to keep colors stable."""
    scores = {}
    for a, b, local in final_turns:
        if local == "Спикер неизвестен":
            continue
        for row in provisional_rows:
            previous = row.get("speaker")
            if not previous or previous == "Спикер неизвестен":
                continue
            overlap = max(0, min(b, row["end"]) - max(a, row["start"]))
            if overlap:
                key = (local, previous)
                scores[key] = scores.get(key, 0) + overlap
    mapping, used = {}, set()
    for (local, previous), score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        if local not in mapping and previous not in used:
            mapping[local] = previous
            used.add(previous)
    # Reserve matched IDs before allocating labels to newly distinguished voices.
    for _, _, local in final_turns:
        if local != "Спикер неизвестен" and local not in mapping:
            index = 1
            while f"Спикер {index}" in used:
                index += 1
            mapping[local] = f"Спикер {index}"
            used.add(mapping[local])
    return [(a, b, mapping.get(local, local)) for a, b, local in final_turns]


def buffer_seconds(duration):
    return max(30, min(120, duration * .05))


def prepared_buffers(audio, analyze, seconds, executor=None):
    size = max(1, round(seconds * SAMPLE_RATE))
    context = 2 * SAMPLE_RATE
    owns_executor = executor is None
    executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="speaker-buffer")
    try:
        def submit(start):
            end = min(start + size, len(audio))
            left, right = max(0, start - context), min(len(audio), end + context)
            chunk = audio[left:right]
            return (start / SAMPLE_RATE, end / SAMPLE_RATE, left / SAMPLE_RATE, chunk,
                    executor.submit(analyze, chunk))

        pending = submit(0) if len(audio) else None
        while pending:
            start, end, offset, chunk, future = pending
            turns = future.result()
            next_start = round(end * SAMPLE_RATE)
            pending = submit(next_start) if next_start < len(audio) else None
            yield start, end, offset, chunk, turns
    finally:
        if owns_executor:
            executor.shutdown(wait=True, cancel_futures=True)


def recognize(model, audio, models, count, language, prompt, progress, on_partial, analyzer=None):
    # Native sherpa calls hold Python's GIL; a process is necessary for actual overlap.
    if analyzer is None:
        pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_initialize_analyzer, initargs=(models, count))
        analyze = _analyze
        refine = _refine
    else:
        pool = ThreadPoolExecutor(max_workers=1)
        analyze = analyzer.analyze
        refine = analyzer.refine
    try:
        return _recognize(model, audio, analyze, refine, pool, language, prompt, progress, on_partial)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _recognize(model, audio, analyze, refine, pool, language, prompt, progress, on_partial):
    duration = len(audio) / SAMPLE_RATE
    seconds = buffer_seconds(duration)
    progress(0, desc=f"Определение спикеров первого буфера (~{seconds:.0f} с)…")
    rows, aliases, hints = [], {}, []
    all_words = []
    detected_language = language
    for start, end, offset, chunk, turns in prepared_buffers(audio, analyze, seconds, pool):
        detail = "спикеры следующего буфера — параллельно" if end < duration else "последний буфер"
        progress(.9 * start / max(duration, 1), desc=f"Распознавание {start:.0f}–{end:.0f} с; {detail}…")
        segments, info = model.transcribe(chunk, language=detected_language, beam_size=5,
                                          vad_filter=True, initial_prompt=prompt, word_timestamps=True)
        detected_language = detected_language or info.language
        for segment in segments:
            if segment.text.strip() and not segment.words:
                raise RuntimeError("Whisper не вернул временные отметки слов для разделения спикеров.")
            # Context overlaps aid recognition; midpoint ownership prevents duplicate words.
            words = [(word.start, word.end, word.word) for word in (segment.words or [])
                     if start <= offset + (word.start + word.end) / 2 < end
                     or (end == duration and offset + word.start == offset + word.end == end)]
            all_words.extend((a + offset, b + offset, text) for a, b, text in words)
            additions = align_words(words, turns)
            for row in additions:
                row["start"] += offset
                row["end"] += offset
            if rows and additions and rows[-1]["speaker"] == additions[0]["speaker"] and additions[0]["start"] - rows[-1]["end"] < 1.5:
                first = additions.pop(0)
                rows[-1]["text"] += " " + first["text"]
                rows[-1]["end"] = first["end"]
            rows.extend(additions)
            aliases, hints = infer_names(rows)
            if on_partial:
                on_partial(dict(text="\n".join(speaker_text(row, aliases) for row in rows),
                                preview_data=dict(rows=[dict(row) for row in rows], aliases=dict(aliases), hints=list(hints))))
            processed = max(start, min(end, offset + segment.end))
            progress(.9 * processed / max(duration, 1), desc=f"Распознано {processed:.0f} из {duration:.0f} с • буферный режим")
        progress(.9 * end / max(duration, 1), desc=f"Обработано {end:.0f} из {duration:.0f} с • буферный режим")
    if all_words:
        progress(.9, desc="Финальное уточнение спикеров по всей записи… Текст уже доступен.")
        final_turns = preserve_labels(pool.submit(refine, audio).result(), rows)
        refined = align_words(all_words, final_turns)
        # An empty refinement must not replace usable buffered diarization.
        if any(row["speaker"] != "Спикер неизвестен" for row in refined):
            rows = refined
            aliases, hints = infer_names(rows)
            if on_partial:
                on_partial(dict(text="\n".join(speaker_text(row, aliases) for row in rows),
                                preview_data=dict(rows=rows, aliases=aliases, hints=hints)))
    progress(1, desc="Уточнение спикеров завершено")
    return rows, aliases, hints, SimpleNamespace(language=detected_language or "—", duration=duration)
