from __future__ import annotations

from abc import ABC, abstractmethod

from arabic_voice.base import BackendUnavailableError, MissingDependencyError
from arabic_voice.stt.types import STTModelInfo, TranscriptionRequest, TranscriptionResult

__all__ = [
    "STTBackend",
    "BackendUnavailableError",
    "MissingDependencyError",
]


class STTBackend(ABC):
    """Shared interface for Emirati/Gulf ASR backends."""

    id: str

    def __init__(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def info(self) -> STTModelInfo:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, req: TranscriptionRequest) -> TranscriptionResult:
        raise NotImplementedError

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True
