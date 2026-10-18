"""Emirati / Gulf Arabic STT multi-backend wrappers."""

from arabic_voice.stt.base import STTBackend
from arabic_voice.stt.registry import (
    create_stt_backend,
    get_stt_backend,
    list_stt_backend_ids,
    list_stt_backends,
)
from arabic_voice.stt.types import STTModelInfo, TranscriptionRequest, TranscriptionResult

__all__ = [
    "STTBackend",
    "STTModelInfo",
    "TranscriptionRequest",
    "TranscriptionResult",
    "get_stt_backend",
    "create_stt_backend",
    "list_stt_backend_ids",
    "list_stt_backends",
]
