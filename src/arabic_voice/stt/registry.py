from __future__ import annotations

from collections.abc import Callable

from arabic_voice.stt.backends.audar_asr_flash import AudarAsrFlashBackend
from arabic_voice.stt.backends.jais_asr import JaisAsrBackend
from arabic_voice.stt.backends.qwen_asr_uae import QwenAsrUAEBackend
from arabic_voice.stt.backends.wav2vec_emirati import Wav2VecEmiratiBackend
from arabic_voice.stt.backends.whisper_arabic_ft import WhisperArabicFTBackend
from arabic_voice.stt.backends.whisper_baseline import WhisperBaselineBackend
from arabic_voice.stt.backends.whisper_gulf_small import WhisperGulfSmallBackend
from arabic_voice.stt.base import STTBackend

_FACTORIES: dict[str, Callable[[], STTBackend]] = {
    JaisAsrBackend.id: JaisAsrBackend,
    AudarAsrFlashBackend.id: AudarAsrFlashBackend,
    QwenAsrUAEBackend.id: QwenAsrUAEBackend,
    WhisperBaselineBackend.id: WhisperBaselineBackend,
    WhisperArabicFTBackend.id: WhisperArabicFTBackend,
    WhisperGulfSmallBackend.id: WhisperGulfSmallBackend,
    Wav2VecEmiratiBackend.id: Wav2VecEmiratiBackend,
}

_INSTANCES: dict[str, STTBackend] = {}


def list_stt_backend_ids() -> list[str]:
    return list(_FACTORIES.keys())


def create_stt_backend(backend_id: str) -> STTBackend:
    if backend_id not in _FACTORIES:
        known = ", ".join(list_stt_backend_ids())
        raise KeyError(f"Unknown STT backend '{backend_id}'. Known: {known}")
    return _FACTORIES[backend_id]()


def get_stt_backend(backend_id: str, *, reload: bool = False) -> STTBackend:
    if reload or backend_id not in _INSTANCES:
        _INSTANCES[backend_id] = create_stt_backend(backend_id)
    return _INSTANCES[backend_id]


def list_stt_backends() -> list[STTBackend]:
    return [get_stt_backend(bid) for bid in list_stt_backend_ids()]
