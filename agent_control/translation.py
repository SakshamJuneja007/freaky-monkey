"""Deterministic Hinglish normalisation for planner input.

This is deliberately not a second LLM. It handles common computer-control
phrases while preserving names, paths and URLs. The planner remains responsible
for final semantic interpretation.
"""
from __future__ import annotations

import re

_PHRASES = {
    "khol do": "open", "kholo": "open", "khol": "open",
    "chala do": "start", "chalado": "start", "chalao": "start",
    "dikha do": "show", "dikhao": "show",
    "bana do": "create", "banao": "create",
    "likh do": "write", "likho": "write",
    "hata do": "delete", "mita do": "delete",
    "band kar do": "close", "band karo": "close",
    "dhoondo": "search", "dhundo": "search",
    "baja do": "play", "bajao": "play",
    "kholna hai": "open",
    "youtube pe jao": "open youtube",
    "youtube par jao": "open youtube",
    "youtube kholo na": "open youtube",
}

_REPLACEMENTS = sorted(_PHRASES.items(), key=lambda item: len(item[0]), reverse=True)

def translate_for_planner(text: str) -> str:
    """Normalise common Hinglish control phrases without changing targets."""
    value = " ".join((text or "").strip().split())
    if not value:
        return value
    for source, target in _REPLACEMENTS:
        value = re.sub(rf"(?<![\w]){re.escape(source)}(?![\w])", target, value, flags=re.IGNORECASE)
    value = re.sub(r"\b(please|pls|plz|yaar|bhai|bro|zara)\b", " ", value, flags=re.IGNORECASE)
    return " ".join(value.split())
