from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens for lexical retrieval."""
    return _TOKEN.findall(text.lower())
