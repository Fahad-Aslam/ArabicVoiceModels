from __future__ import annotations

import numpy as np

from arabic_voice.base import BackendUnavailableError, MissingDependencyError, TTSBackend
from arabic_voice.device import empty_cache, pick_torch_device
from arabic_voice.types import ModelInfo, SynthesisRequest, SynthesisResult

HF_REPO = "vadimbelsky/qwen3.5-TTS-Emirati"
DEFAULT_SPEAKER = "emirati_speaker"


class QwenEmiratiBackend(TTSBackend):
    id = "qwen_emirati"

    def __init__(
        self,
        checkpoint: str = HF_REPO,
        device: str | None = None,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        self._checkpoint = checkpoint
        self._device = device
        self._dtype = dtype
        self._tts = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            id=self.id,
            name="Qwen3-TTS Emirati",
            description=(
                "Community fine-tune of Qwen3-TTS-12Hz-1.7B-Base for Emirati Arabic. "
                "Heavy on Mac; may OOM — we fail gracefully."
            ),
            license="Apache-2.0",
            dialect="Emirati (ar-AE)",
            checkpoint=HF_REPO,
            mac_status="heavy (prefer MPS; may OOM on limited RAM)",
            requires_ref_audio=False,
            requires_gpu_preferred=True,
            commercial_ok=True,
            extra="qwen",
        )

    def load(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise MissingDependencyError(self.id, "qwen", str(exc)) from exc

        # Compat: some huggingface_hub 0.3x builds omit top-level is_offline_mode,
        # which transformers/qwen-tts may import during from_pretrained.
        try:
            import huggingface_hub as _hub

            if not hasattr(_hub, "is_offline_mode"):
                import os

                def _is_offline_mode() -> bool:
                    return bool(
                        os.environ.get("HF_HUB_OFFLINE")
                        or os.environ.get("TRANSFORMERS_OFFLINE")
                    )

                _hub.is_offline_mode = _is_offline_mode  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise MissingDependencyError(self.id, "qwen", str(exc)) from exc

        self._device = self._device or pick_torch_device(prefer_mps=True)
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        # MPS: float16 usually safest; bfloat16 support varies
        if self._device == "mps" and self._dtype == "bfloat16":
            self._dtype = "float16"
        torch_dtype = dtype_map.get(self._dtype, torch.float16)

        try:
            self._tts = Qwen3TTSModel.from_pretrained(
                self._checkpoint,
                device_map=self._device,
                dtype=torch_dtype,
                attn_implementation="sdpa",
            )
        except Exception as exc:
            # Retry on CPU float32 if MPS OOM / unsupported
            msg = str(exc).lower()
            if self._device != "cpu" and any(
                k in msg for k in ("out of memory", "oom", "mps", "not supported")
            ):
                empty_cache(self._device)
                try:
                    self._device = "cpu"
                    self._tts = Qwen3TTSModel.from_pretrained(
                        self._checkpoint,
                        device_map="cpu",
                        dtype=torch.float32,
                        attn_implementation="sdpa",
                    )
                except Exception as cpu_exc:
                    raise BackendUnavailableError(
                        f"Qwen Emirati failed on MPS and CPU.\nMPS/first: {exc}\nCPU: {cpu_exc}"
                    ) from cpu_exc
            else:
                raise BackendUnavailableError(
                    f"Failed to load Qwen Emirati ({self._checkpoint}): {exc}"
                ) from exc
        self._loaded = True

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        self.ensure_loaded()
        assert self._tts is not None

        text = req.text.strip()
        if not text:
            raise ValueError("text must be non-empty")

        speaker = req.speaker or DEFAULT_SPEAKER

        try:
            # Prefer custom-voice API used by fine-tunes
            if hasattr(self._tts, "generate_custom_voice"):
                wavs, sr = self._tts.generate_custom_voice(text=text, speaker=speaker)
            elif hasattr(self._tts, "generate"):
                wavs, sr = self._tts.generate(
                    text=text,
                    language=req.language if req.language else "arabic",
                    speaker=speaker,
                )
            else:
                raise BackendUnavailableError(
                    "qwen_tts model has neither generate_custom_voice nor generate"
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
                empty_cache(self._device or "cpu")
                raise BackendUnavailableError(
                    "Qwen Emirati ran out of memory on this Mac. "
                    "Try a shorter utterance, close other apps, or use piper_emirati / emirati_vits."
                ) from exc
            raise

        wave = np.asarray(wavs[0] if isinstance(wavs, (list, tuple)) else wavs, dtype=np.float32)
        wave = wave.squeeze()
        return SynthesisResult(
            audio=wave,
            sample_rate=int(sr),
            backend_id=self.id,
            meta={
                "checkpoint": self._checkpoint,
                "speaker": speaker,
                "device": self._device,
            },
        )
