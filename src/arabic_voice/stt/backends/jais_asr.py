from __future__ import annotations

from arabic_voice.base import BackendUnavailableError, MissingDependencyError
from arabic_voice.device import empty_cache, pick_torch_device
from arabic_voice.stt.audio_io import audio_path
from arabic_voice.stt.base import STTBackend
from arabic_voice.stt.types import STTModelInfo, TranscriptionRequest, TranscriptionResult

HF_REPO = "IFM/Jais-ASR"


class JaisAsrBackend(STTBackend):
    id = "jais_asr"

    def __init__(self) -> None:
        super().__init__()
        self._pipe = None
        self._device = None

    def info(self) -> STTModelInfo:
        return STTModelInfo(
            id=self.id,
            name="Jais-ASR",
            description="IFM Jais-ASR with explicit Emirati (AE) dialect; strong Mixat CS numbers.",
            license="Apache-2.0",
            dialect="19 dialects incl. Emirati (AE) + code-switch",
            checkpoint=HF_REPO,
            mac_status="heavy (~12 GB; often OOM on laptop unified memory)",
            requires_gpu_preferred=True,
            commercial_ok=True,
            extra="asr-jais",
            streaming=False,
        )

    def load(self) -> None:
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise MissingDependencyError(
                self.id,
                "asr-jais",
                f"{exc}\nInstall: pip install -e '.[asr-jais]'",
            ) from exc

        self._device = pick_torch_device(prefer_mps=True)
        # Jais is large — prefer CPU with warnings if MPS memory is tight
        try:
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=HF_REPO,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16 if self._device != "mps" else torch.float16,
            )
        except Exception as exc:
            empty_cache(self._device or "cpu")
            raise BackendUnavailableError(
                f"Jais-ASR failed to load (needs ~12GB).\n{exc}\n"
                "Skip on this Mac and use audar_asr_flash / qwen_asr_uae / whisper_baseline."
            ) from exc
        self._loaded = True

    def transcribe(self, req: TranscriptionRequest) -> TranscriptionResult:
        self.ensure_loaded()
        assert self._pipe is not None
        path = audio_path(req)
        try:
            out = self._pipe(str(path), max_new_tokens=256)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
                empty_cache(self._device or "cpu")
                raise BackendUnavailableError("Jais-ASR OOM during decode on this Mac.") from exc
            raise
        text = out["text"] if isinstance(out, dict) else str(out)
        return TranscriptionResult(
            text=text.strip(),
            backend_id=self.id,
            language="AE",
            meta={"checkpoint": HF_REPO, "device": self._device},
        )
