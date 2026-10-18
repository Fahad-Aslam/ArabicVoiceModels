"""Emirati Arabic TTS + STT multi-backend wrappers."""

from arabic_voice.base import BackendUnavailableError, MissingDependencyError, TTSBackend
from arabic_voice.registry import create_backend, get_backend, list_backend_ids, list_backends
from arabic_voice.stt import (
    STTBackend,
    TranscriptionRequest,
    TranscriptionResult,
    create_stt_backend,
    get_stt_backend,
    list_stt_backend_ids,
    list_stt_backends,
)
from arabic_voice.stt.types import STTModelInfo
from arabic_voice.types import ModelInfo, SynthesisRequest, SynthesisResult

__all__ = [
    "TTSBackend",
    "STTBackend",
    "BackendUnavailableError",
    "MissingDependencyError",
    "ModelInfo",
    "STTModelInfo",
    "SynthesisRequest",
    "SynthesisResult",
    "TranscriptionRequest",
    "TranscriptionResult",
    "get_backend",
    "create_backend",
    "list_backend_ids",
    "list_backends",
    "get_stt_backend",
    "create_stt_backend",
    "list_stt_backend_ids",
    "list_stt_backends",
]

__version__ = "0.1.0"
