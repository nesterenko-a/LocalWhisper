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


def create_diarizer(models, count=0):
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
    return sherpa.OfflineSpeakerDiarization(config)


def diarize(audio, models, count, progress):
    engine = create_diarizer(models, count)

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


class SpeakerTracker:
    """Match local clusters to recording-wide voices using normalized embeddings."""
    def __init__(self, maximum=0, threshold=0.55):
        self.maximum = int(maximum or 0)
        self.threshold = threshold
        self.voices = []

    def match(self, embeddings):
        import numpy as np
        vectors = {}
        for label, value in embeddings.items():
            vector = np.asarray(value, dtype=np.float32)
            norm = np.linalg.norm(vector)
            if norm > 1e-8 and np.isfinite(vector).all():
                vectors[label] = vector / norm
        matches, used = {}, set()
        pairs = sorted(((float(vector @ reference), label, index)
                        for label, vector in vectors.items() for index, reference in enumerate(self.voices)), reverse=True)
        for score, label, index in pairs:
            if score >= self.threshold and label not in matches and index not in used:
                matches[label] = index
                used.add(index)
        for label, vector in vectors.items():
            if label not in matches:
                if self.maximum and len(self.voices) >= self.maximum:
                    continue
                matches[label] = len(self.voices)
                self.voices.append(vector)
            else:
                index = matches[label]
                average = self.voices[index] * .9 + vector * .1
                self.voices[index] = average / np.linalg.norm(average)
        return {label: f"Спикер {index + 1}" for label, index in matches.items()}


class BufferedDiarizer:
    def __init__(self, models, count=0):
        import sherpa_onnx as sherpa
        self.engine = create_diarizer(models)  # Local speaker count may differ from the whole recording.
        self.extractor = sherpa.SpeakerEmbeddingExtractor(sherpa.SpeakerEmbeddingExtractorConfig(
            model=str(models[1]), num_threads=2, provider="cpu"))
        self.tracker = SpeakerTracker(count)

    def refine(self, audio):
        turns = self.engine.process(audio).sort_by_start_time()
        durations = {}
        for turn in turns:
            durations[turn.speaker] = durations.get(turn.speaker, 0) + turn.end - turn.start
        allowed = set(sorted(durations, key=durations.get, reverse=True)[:self.tracker.maximum]) if self.tracker.maximum else set(durations)
        labels = {}
        result = []
        for turn in turns:
            if turn.speaker not in labels:
                labels[turn.speaker] = (f"Спикер {sum(value != 'Спикер неизвестен' for value in labels.values()) + 1}"
                                        if turn.speaker in allowed
                                        else "Спикер неизвестен")
            result.append((turn.start, turn.end, labels[turn.speaker]))
        return result

    def analyze(self, audio):
        import numpy as np
        turns = self.engine.process(audio).sort_by_start_time()
        clips = {}
        for turn in turns:
            # Avoid contaminated speaker samples where local voices overlap.
            if turn.end - turn.start < .6 or any(
                other.speaker != turn.speaker and min(turn.end, other.end) > max(turn.start, other.start)
                for other in turns
            ):
                continue
            clips.setdefault(turn.speaker, []).append(audio[int(turn.start * 16000):int(turn.end * 16000)])
        embeddings = {}
        for label, pieces in clips.items():
            stream = self.extractor.create_stream()
            stream.accept_waveform(sample_rate=16000, waveform=np.concatenate(pieces)[:240000])
            stream.input_finished()
            if self.extractor.is_ready(stream):
                embeddings[label] = self.extractor.compute(stream)
        labels = self.tracker.match(embeddings)
        return [(turn.start, turn.end, labels.get(turn.speaker, "Спикер неизвестен")) for turn in turns]


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
