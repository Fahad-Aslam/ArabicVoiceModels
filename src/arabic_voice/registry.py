from __future__ import annotations

from collections.abc import Callable

from arabic_voice.base import TTSBackend
from arabic_voice.backends.audar_flash import AudarFlashBackend
from arabic_voice.backends.emirati_vits import EmiratiVITSBackend
from arabic_voice.backends.habibi_uae import HabibiUAEBackend
from arabic_voice.backends.piper_emirati import PiperEmiratiBackend
from arabic_voice.backends.qwen_emirati import QwenEmiratiBackend

_FACTORIES: dict[str, Callable[[], TTSBackend]] = {
    PiperEmiratiBackend.id: PiperEmiratiBackend,
    AudarFlashBackend.id: AudarFlashBackend,
    HabibiUAEBackend.id: HabibiUAEBackend,
    EmiratiVITSBackend.id: EmiratiVITSBackend,
    QwenEmiratiBackend.id: QwenEmiratiBackend,
}

_INSTANCES: dict[str, TTSBackend] = {}


def list_backend_ids() -> list[str]:
    return list(_FACTORIES.keys())


def create_backend(backend_id: str) -> TTSBackend:
    if backend_id not in _FACTORIES:
        known = ", ".join(list_backend_ids())
        raise KeyError(f"Unknown backend '{backend_id}'. Known: {known}")
    return _FACTORIES[backend_id]()


def get_backend(backend_id: str, *, reload: bool = False) -> TTSBackend:
    """Return a cached backend instance (not yet loaded until synthesize/load)."""
    if reload or backend_id not in _INSTANCES:
        _INSTANCES[backend_id] = create_backend(backend_id)
    return _INSTANCES[backend_id]


def list_backends() -> list[TTSBackend]:
    return [get_backend(bid) for bid in list_backend_ids()]
