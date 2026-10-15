"""Emirati Arabic TTS multi-backend wrappers."""

from arabic_voice.base import BackendUnavailableError, MissingDependencyError, TTSBackend
from arabic_voice.registry import create_backend, get_backend, list_backend_ids, list_backends
from arabic_voice.types import ModelInfo, SynthesisRequest, SynthesisResult

__all__ = [
    "TTSBackend",
    "BackendUnavailableError",
    "MissingDependencyError",
    "ModelInfo",
    "SynthesisRequest",
    "SynthesisResult",
    "get_backend",
    "create_backend",
    "list_backend_ids",
    "list_backends",
]

__version__ = "0.1.0"
