"""Local voice diarization, word alignment and reviewable speaker names."""
import json
from pathlib import Path
import re
import shutil
import tarfile
import urllib.request

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION = "sherpa-onnx-pyannote-segmentation-3-0"
EMBEDDING = "wespeaker_en_voxceleb_resnet34.onnx"


def ensure_models(root, offline=False):
    root = Path(root) / "speakers"
    root.mkdir(parents=True, exist_ok=True)
    segmentation, embedding = root / "segmentation.onnx", root / EMBEDDING
    if offline and (not segmentation.is_file() or not embedding.is_file()):
        raise RuntimeError("Модели спикеров ещё не скачаны. Отключите «Только офлайн» для первой загрузки.")
    if not segmentation.is_file():
        archive = root / "segmentation.tar.bz2.part"
        _download(f"{RELEASES}/speaker-segmentation-models/{SEGMENTATION}.tar.bz2", archive)
        temporary = segmentation.with_suffix(".part")
        try:
            with tarfile.open(archive, "r:bz2") as bundle:
                member = bundle.getmember(f"{SEGMENTATION}/model.onnx")
                with bundle.extractfile(member) as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            temporary.replace(segmentation)
        finally:
            archive.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
    if not embedding.is_file():
        temporary = embedding.with_suffix(".part")
        _download(f"{RELEASES}/speaker-recongition-models/{EMBEDDING}", temporary)
        temporary.replace(embedding)
    return segmentation, embedding


def _download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "local-whisper"})
    with urllib.request.urlopen(request, timeout=60) as response, Path(destination).open("wb") as output:
        shutil.copyfileobj(response, output)


def diarize(audio, models, count, progress):
    import sherpa_onnx as sherpa
    segmentation, embedding = models
    config = sherpa.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation)),
            num_threads=2, provider="cpu",
        ),
        embedding=sherpa.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=2, provider="cpu"),
        clustering=sherpa.FastClusteringConfig(num_clusters=int(count) if count else -1, threshold=0.5),
        min_duration_on=0.3, min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("Не удалось настроить модели разделения спикеров.")
    engine = sherpa.OfflineSpeakerDiarization(config)

    def callback(done, total):
        progress(done / max(total, 1), desc="Определение спикеров по всей записи…")
        return 0

    turns = engine.process(audio, callback=callback).sort_by_start_time()
    # Stable, human-readable IDs in order of first appearance, not cluster IDs.
    labels = {}
    result = []
    for turn in turns:
        label = labels.setdefault(turn.speaker, f"Спикер {len(labels) + 1}")
        result.append((turn.start, turn.end, label))
    return result


def align_words(words, turns):
    """Assign words by maximum time overlap, leaving unsupported speech unknown."""
    rows = []
    first = 0
    for start, end, word in words:
        scores = {}
        while first < len(turns) and turns[first][1] <= start:
            first += 1
        for index in range(first, len(turns)):
            a, b, label = turns[index]
            if a > end or (a == end and end > start):
                break
            overlap = max(0, min(end, b) - max(start, a))
            if end == start and a <= start < b:
                overlap = 1  # Whisper sometimes emits zero-duration words.
            if overlap:
                scores[label] = scores.get(label, 0) + overlap
        speaker = max(scores, key=scores.get) if scores else "Спикер неизвестен"
        if rows and rows[-1]["speaker"] == speaker and start - rows[-1]["end"] < 1.5:
            rows[-1]["end"] = end
            rows[-1]["text"] += word
        else:
            rows.append(dict(start=start, end=end, speaker=speaker, text=word))
    for row in rows:
        row["text"] = row["text"].strip()
    return rows


def infer_names(rows):
    aliases, candidates, hints = {}, {}, []
    name = r"[А-ЯЁA-Z][а-яёa-z]+(?:[ -][А-ЯЁA-Z][а-яёa-z]+)?"
    introduction = re.compile(r"(?:[Мм]еня зовут|[Мм]о[её] имя)\s+(" + name + r")(?=[,.!?;:]|$)")
    address = re.compile(r"(?:^|[.!?]\s+)(" + name + r"),\s")
    role_address = re.compile(r"(?:^|[.!?]\s+)((?:[Дд]иректор|[Рр]уководитель|[Нн]ачальник|[Мм]енеджер|[Кк]оллега)(?:\s+[а-яё]+){0,2}),\s")
    for row in rows:
        label, text = row["speaker"], row["text"]
        match = introduction.search(text)
        if match and label != "Спикер неизвестен":
            candidates.setdefault(label, set()).add(match[1])
            hints.append(f'{label}: самопредставление «{match[0]}». Проверьте имя по записи.')
        for match in address.finditer(text):
            hints.append(f'Обращение «{match[1]}» в реплике {label}. К кому обращались — проверьте по контексту.')
        for match in role_address.finditer(text):
            hints.append(f'Обращение по должности «{match[1]}» в реплике {label}. Проверьте, к какому спикеру оно относится.')
    for label, names in candidates.items():
        if len(names) == 1:
            value = next(iter(names))
            if sum(value in other for other in candidates.values()) == 1:
                aliases[label] = value
    return aliases, list(dict.fromkeys(hints))


def speaker_text(row, aliases):
    label = row.get("speaker")
    if not label:
        return row["text"]
    display = f"{aliases[label]} ({label})" if aliases.get(label) else label
    return f"{display}: {row['text']}"


def timestamp(seconds, separator=","):
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{ms:03}"


def export_transcript(folder, rows, aliases=None, hints=None):
    folder = Path(folder)
    aliases, hints = aliases or {}, hints or []
    text = "\n".join(speaker_text(row, aliases) for row in rows)
    txt, srt, vtt, metadata = [folder / f"transcript.{ext}" for ext in ("txt", "srt", "vtt", "json")]
    txt.write_text(text, encoding="utf-8")
    srt.write_text("\n\n".join(f"{i}\n{timestamp(r['start'])} --> {timestamp(r['end'])}\n{speaker_text(r, aliases)}" for i, r in enumerate(rows, 1)) + "\n", encoding="utf-8")
    vtt.write_text("WEBVTT\n\n" + "\n\n".join(f"{timestamp(r['start'], '.')} --> {timestamp(r['end'], '.')}\n{speaker_text(r, aliases)}" for r in rows) + "\n", encoding="utf-8")
    metadata.write_text(json.dumps(dict(rows=rows, aliases=aliases, hints=hints), ensure_ascii=False, indent=2), encoding="utf-8")
    return text, [str(p) for p in (txt, srt, vtt, metadata)]


def parse_aliases(value, known):
    aliases = {}
    for line in (value or "").splitlines():
        if not line.strip():
            continue
        label, separator, name = line.partition("=")
        label, name = label.strip(), name.strip()
        if not separator or label not in known:
            raise ValueError(f"Используйте «Спикер 1 = Имя или должность». Неизвестная строка: {line}")
        if len(name) > 100:
            raise ValueError("Имя или должность должны быть короче 100 символов.")
        aliases[label] = name
    return aliases
