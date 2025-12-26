"""Pydantic models for API request/response."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Supported translation tasks."""

    SPEECH2TEXT = "speech2text"
    SPEECH2SPEECH = "speech2speech"
    TEXT2TEXT = "text2text"
    TEXT2SPEECH = "text2speech"
    ASR = "auto_speech_recognition"


# Language metadata and per-direction capabilities derived from languages.md
@lru_cache(maxsize=1)
def _load_language_matrix() -> dict[str, dict[str, object]]:
    lang_path = Path(__file__).with_name("languages.json")
    raw = lang_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    matrix: dict[str, dict[str, object]] = {}
    for code, meta in data.items():
        matrix[code] = {
            "code": code,
            "name": meta.get("name"),
            "script": meta.get("script"),
            "source_caps": set(meta.get("source_caps") or []),
            "target_caps": set(meta.get("target_caps") or []),
        }

    return matrix


LANGUAGE_MATRIX = _load_language_matrix()

# Language codes supported by SeamlessM4T
SUPPORTED_LANGUAGES = sorted(LANGUAGE_MATRIX.keys())

# Map common language names to SeamlessM4T codes
LANGUAGE_NAME_TO_CODE = {v["name"]: k for k, v in LANGUAGE_MATRIX.items()}
_LANGUAGE_NAME_TO_CODE_LOWER = {k.lower(): v for k, v in LANGUAGE_NAME_TO_CODE.items()}


def supported_languages_for_task(task: TaskType, direction: str) -> list[str]:
    if direction not in ("source", "target"):
        raise ValueError("direction must be 'source' or 'target'")

    if task == TaskType.TEXT2TEXT:
        needed = "Tx"
        caps_key = "source_caps" if direction == "source" else "target_caps"
    elif task == TaskType.SPEECH2TEXT:
        needed = "Sp" if direction == "source" else "Tx"
        caps_key = "source_caps" if direction == "source" else "target_caps"
    elif task == TaskType.ASR:
        if direction == "target":
            return []
        needed = "Sp"
        caps_key = "source_caps"
    elif task == TaskType.TEXT2SPEECH:
        needed = "Tx" if direction == "source" else "Sp"
        caps_key = "source_caps" if direction == "source" else "target_caps"
    elif task == TaskType.SPEECH2SPEECH:
        needed = "Sp"
        caps_key = "source_caps" if direction == "source" else "target_caps"
    else:
        return []

    return sorted(
        [
            code
            for code, meta in LANGUAGE_MATRIX.items()
            if needed in (meta.get(caps_key) or set())
        ]
    )


def normalize_language(lang: str) -> str:
    """Convert language name to SeamlessM4T language code."""
    lang_lower = lang.lower().strip()

    # Already a valid code
    if lang_lower in SUPPORTED_LANGUAGES:
        return lang_lower

    # Try name mapping
    if lang_lower in _LANGUAGE_NAME_TO_CODE_LOWER:
        return _LANGUAGE_NAME_TO_CODE_LOWER[lang_lower]

    # Try partial match
    for name, code in _LANGUAGE_NAME_TO_CODE_LOWER.items():
        if name in lang_lower or lang_lower in name:
            return code

    # Default to English if unknown
    return "eng"


class TranslationData(BaseModel):
    """Inner data payload for translation request."""

    input: str = Field(..., description="Text or base64-encoded audio")
    task_string: TaskType
    source_language: str
    target_language: str


class TranslationRequest(BaseModel):
    """Translation request body."""

    data: TranslationData


class TranslationResponse(BaseModel):
    """Translation response."""

    output: str = Field(..., description="Translated text or base64-encoded audio")
    source_language: str
    target_language: str
    task: TaskType


class HealthResponse(BaseModel):
    """Health check response."""
