from __future__ import annotations

from .config import SECRETS_DIR


def load_secret(name: str) -> str | None:
    path = SECRETS_DIR / name
    if not path.exists():
        return None

    value = path.read_text(encoding="utf-8").strip()
    return value or None
