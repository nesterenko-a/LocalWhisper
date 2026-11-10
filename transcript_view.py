"""Escaped, theme-aware transcript cards; no transcript content is executable HTML."""
from html import escape
import re

from speakers import timestamp

PALETTE = [
    ("#2457a7", "#9dc2ff"), ("#8b3282", "#f0acf0"),
    ("#11654e", "#8edbc0"), ("#90551a", "#f8cb91"),
    ("#5e42a6", "#cdb8ff"), ("#9a3748", "#ffafbf"),
    ("#136570", "#8bdae6"), ("#605a14", "#e0da8c"),
]

STYLE = """<style>
.transcript-view { color: var(--body-text-color, #222); font-size: 15px; line-height: 1.65; }
.transcript-view .transcript-legend { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }
.transcript-view .speaker-chip { border:1px solid currentColor; border-radius:16px; padding:2px 10px; font-size:13px; }
.transcript-view .utterance { border:1px solid var(--border-color-primary, #8885); border-left:4px solid var(--voice-color); border-radius:8px; padding:12px 16px; margin:0 0 12px; background:var(--block-background-fill, transparent); }
.transcript-view .utterance-header { display:flex; flex-wrap:wrap; align-items:baseline; gap:8px 14px; margin-bottom:6px; }
.transcript-view .speaker-name { color:var(--voice-color); font-weight:650; }
.transcript-view .timecode { font-variant-numeric:tabular-nums; font-size:12px; color:var(--body-text-color-subdued, #777); }
.transcript-view .utterance-text { white-space:pre-wrap; overflow-wrap:anywhere; }
.transcript-view .view-note { color:var(--body-text-color-subdued, #777); margin:0 0 12px; font-size:13px; }
.transcript-view .voice { --voice-color:var(--voice-light); color:var(--voice-color); }
.dark .transcript-view .voice { --voice-color:var(--voice-dark); }
</style>"""


def voice_style(speaker):
    match = re.fullmatch(r"Спикер (\d+)", speaker or "")
    index = int(match[1]) - 1 if match else -1
    if 0 <= index < len(PALETTE):
        light, dark = PALETTE[index]
    elif index >= len(PALETTE):
        hue = (index * 137.508) % 360
        light, dark = f"hsl({hue:.2f} 62% 32%)", f"hsl({hue:.2f} 72% 78%)"
    else:
        light, dark = "#596574", "#b5c1d1"
    return f"--voice-light:{light};--voice-dark:{dark}"


def render_transcript(record=None, data=None):
    if not record or not record.get("text"):
        return STYLE + '<section class="transcript-view"><p class="view-note">Здесь появится текст распознавания.</p></section>'
    data = data or {}
    rows = data.get("rows") or [dict(text=line) for line in record["text"].splitlines() if line.strip()]
    aliases = data.get("aliases", {})
    parts = [STYLE, '<section class="transcript-view" aria-label="Текст распознавания">']
    if record.get("state") != "done":
        parts.append('<p class="view-note">Частичный результат. Новые реплики появляются по мере распознавания; имена ещё могут уточняться.</p>')
    speakers = list(dict.fromkeys(row["speaker"] for row in rows if row.get("speaker")))
    if speakers:
        parts.append('<div class="transcript-legend" aria-label="Участники разговора">')
        for speaker in speakers:
            name = aliases.get(speaker) or speaker
            label = f"{name} · {speaker}" if name != speaker else speaker
            parts.append(f'<span class="speaker-chip voice" style="{voice_style(speaker)}">{escape(label)}</span>')
        parts.append('</div>')
    parts.append('<div role="list">')
    for row in rows:
        speaker = row.get("speaker", "")
        name = aliases.get(speaker) or speaker
        parts.append(f'<article role="listitem" class="utterance voice" style="{voice_style(speaker)}">')
        parts.append('<div class="utterance-header">')
        if name:
            suffix = f" ({speaker})" if name != speaker else ""
            parts.append(f'<span class="speaker-name">{escape(name + suffix)}</span>')
        if "start" in row and "end" in row:
            a, b = timestamp(row["start"], "."), timestamp(row["end"], ".")
            parts.append(f'<span class="timecode">{a} — {b}</span>')
        parts.append(f'</div><div class="utterance-text">{escape(row["text"])}</div></article>')
    parts.append('</div></section>')
    return "".join(parts)
