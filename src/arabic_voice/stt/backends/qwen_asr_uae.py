from __future__ import annotations

from arabic_voice.base import BackendUnavailableError, MissingDependencyError
from arabic_voice.device import empty_cache, pick_torch_device
from arabic_voice.stt.audio_io import audio_path
from arabic_voice.stt.base import STTBackend
from arabic_voice.stt.types import STTModelInfo, TranscriptionRequest, TranscriptionResult

HF_REPO = "vadimbelsky/qwen3-asr-arabic-uae"


class QwenAsrUAEBackend(STTBackend):
    id = "qwen_asr_uae"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._device = None

    def info(self) -> STTModelInfo:
        return STTModelInfo(
            id=self.id,
            name="Qwen3-ASR Arabic UAE",
            description="Emirati fine-tune of Qwen3-ASR-1.7B (validate on real audio; synthetic train).",
            license="Apache-2.0",
            dialect="Emirati (ar-AE)",
            checkpoint=HF_REPO,
            mac_status="heavy (MPS; may OOM)",
            requires_gpu_preferred=True,
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
            empty_cache(self._device or "cpu")
            if self._device != "cpu":
                try:
                    self._device = "cpu"
                    self._model = Qwen3ASRModel.from_pretrained(
                        HF_REPO,
                        dtype=torch.float32,
                        device_map="cpu",
                    )
                except Exception as cpu_exc:
                    raise BackendUnavailableError(
                        f"Qwen ASR UAE failed on MPS and CPU.\n{exc}\n{cpu_exc}"
                    ) from cpu_exc
            else:
                raise BackendUnavailableError(f"Failed to load {HF_REPO}: {exc}") from exc
        self._loaded = True

    def transcribe(self, req: TranscriptionRequest) -> TranscriptionResult:
        self.ensure_loaded()
        assert self._model is not None
        path = audio_path(req)
        try:
            results = self._model.transcribe(
                audio=[str(path)],
                language=["Arabic"],
            )
            if isinstance(results, list) and results:
                item = results[0]
                text = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else str(item))
            else:
                text = str(results)
        except TypeError:
            # Older API: transcribe(path, language=...)
            results = self._model.transcribe(str(path), language="Arabic")
            text = getattr(results, "text", None) or str(results)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                empty_cache(self._device or "cpu")
                raise BackendUnavailableError(
                    "Qwen ASR UAE OOM on this Mac — try whisper_baseline or audar_asr_flash."
                ) from exc
            raise

        return TranscriptionResult(
            text=str(text).strip(),
            backend_id=self.id,
            language="ar",
            meta={"checkpoint": HF_REPO, "device": self._device},
        )
