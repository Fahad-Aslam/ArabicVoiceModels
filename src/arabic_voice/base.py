from __future__ import annotations

from abc import ABC, abstractmethod

from arabic_voice.types import ModelInfo, SynthesisRequest, SynthesisResult


class TTSBackend(ABC):
    """Shared interface for all Emirati/Gulf TTS backends."""

    id: str

    def __init__(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def info(self) -> ModelInfo:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        raise NotImplementedError

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True


class BackendUnavailableError(RuntimeError):
    """Raised when optional deps are missing or the backend cannot run on this Mac."""


class MissingDependencyError(BackendUnavailableError):
    def __init__(self, backend_id: str, extra: str, detail: str | None = None) -> None:
        msg = (
            f"Backend '{backend_id}' requires optional deps. "
            f'Install with: pip install -e ".[{extra}]"'
        )
        if detail:
            msg = f"{msg}\n{detail}"
        super().__init__(msg)
        self.backend_id = backend_id
        self.extra = extra
