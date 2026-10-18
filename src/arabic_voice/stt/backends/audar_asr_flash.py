from __future__ import annotations

from arabic_voice.base import BackendUnavailableError, MissingDependencyError
from arabic_voice.device import pick_torch_device
from arabic_voice.stt.audio_io import audio_path
from arabic_voice.stt.base import STTBackend
from arabic_voice.stt.types import STTModelInfo, TranscriptionRequest, TranscriptionResult

HF_REPO = "audarai/Audar-ASR-V1-Flash"


class AudarAsrFlashBackend(STTBackend):
    id = "audar_asr_flash"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._device = None

    def info(self) -> STTModelInfo:
        return STTModelInfo(
            id=self.id,
            name="Audar-ASR V1 Flash",
            description="Arabic-first ASR claiming MSA + Gulf/Emirati; streaming-capable family.",
            license="AudarAI Open License v1.0",
            dialect="MSA + Gulf/Emirati (claimed)",
            checkpoint=HF_REPO,
            mac_status="good (Transformers; GGUF path optional)",
            commercial_ok=True,
            extra="asr-qwen",
            streaming=True,
        )

    def load(self) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise MissingDependencyError(
                self.id,
                "asr-qwen",
                f"{exc}\nInstall: pip install -e '.[asr-qwen]'",
            ) from exc

        self._device = pick_torch_device(prefer_mps=True)
        dtype = torch.float16 if self._device == "mps" else torch.bfloat16
        try:
            self._model = Qwen3ASRModel.from_pretrained(
                HF_REPO,
                dtype=dtype,
                device_map=self._device,
            )
        except Exception as exc:
            raise BackendUnavailableError(f"Failed to load Audar ASR Flash: {exc}") from exc
        self._loaded = True

    def transcribe(self, req: TranscriptionRequest) -> TranscriptionResult:
        self.ensure_loaded()
        assert self._model is not None
        path = audio_path(req)
        try:
            results = self._model.transcribe(audio=[str(path)], language=["Arabic"])
            item = results[0] if isinstance(results, list) else results
            text = getattr(item, "text", None) or str(item)
        except TypeError:
            results = self._model.transcribe(str(path), language="Arabic")
            text = getattr(results, "text", None) or str(results)

        return TranscriptionResult(
            text=str(text).strip(),
            backend_id=self.id,
            language="ar",
            meta={"checkpoint": HF_REPO, "device": self._device},
        )
