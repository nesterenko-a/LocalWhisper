"""Atomic JSON replacement with retries for transient Windows sharing locks."""
import json
from pathlib import Path
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        for attempt in range(7):
            try:
                temporary.replace(path)
                return
            except OSError as exc:
                transient = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32, 33)
                if not transient or attempt == 6:
                    raise
                time.sleep(0.02 * 2 ** attempt)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass  # Another Windows reader may still hold this unique temporary file.
