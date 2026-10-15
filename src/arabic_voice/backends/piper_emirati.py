from __future__ import annotations

from pathlib import Path

import numpy as np

from arabic_voice.base import MissingDependencyError, TTSBackend
from arabic_voice.paths import CACHE_DIR, ensure_dirs
from arabic_voice.types import ModelInfo, SynthesisRequest, SynthesisResult

HF_REPO = "vadimbelsky/arabic-emirati-female-piper"
ONNX_NAME = "arabic-emirati-female-model.onnx"
JSON_NAME = "arabic-emirati-female-model.onnx.json"


class PiperEmiratiBackend(TTSBackend):
    id = "piper_emirati"

    def __init__(self, model_dir: Path | None = None) -> None:
        super().__init__()
        self._model_dir = model_dir
        self._voice = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            id=self.id,
            name="Piper Emirati Female",
            description="ONNX Piper voice trained for Emirati Arabic (female).",
            license="MIT",
            dialect="Emirati (ar-AE)",
            checkpoint=HF_REPO,
            mac_status="excellent (CPU/ONNX)",
            requires_ref_audio=False,
            requires_gpu_preferred=False,
            commercial_ok=True,
            extra="piper",
        )

    def _resolve_dir(self) -> Path:
        ensure_dirs()
        return self._model_dir or (CACHE_DIR / "piper_emirati")

    def load(self) -> None:
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise MissingDependencyError(self.id, "piper", str(exc)) from exc

        from huggingface_hub import hf_hub_download

        model_dir = self._resolve_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        onnx = hf_hub_download(
            repo_id=HF_REPO,
            filename=ONNX_NAME,
            local_dir=str(model_dir),
        )
        hf_hub_download(
            repo_id=HF_REPO,
            filename=JSON_NAME,
            local_dir=str(model_dir),
        )
        self._voice = PiperVoice.load(onnx)
        self._loaded = True

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        self.ensure_loaded()
        assert self._voice is not None

        text = req.text.strip()
        if not text:
            raise ValueError("text must be non-empty")

        chunks: list[np.ndarray] = []
        sample_rate = getattr(self._voice.config, "sample_rate", 22050)

        # piper-tts API variants across versions
        if hasattr(self._voice, "synthesize"):
            for chunk in self._voice.synthesize(text):
                audio_bytes = getattr(chunk, "audio_int16_bytes", None)
                if audio_bytes is None and hasattr(chunk, "audio_int16"):
                    arr = np.asarray(chunk.audio_int16, dtype=np.int16)
                elif audio_bytes is not None:
                    arr = np.frombuffer(audio_bytes, dtype=np.int16)
                else:
                    raise RuntimeError("Unexpected Piper chunk format")
                chunks.append(arr.astype(np.float32) / 32768.0)
                if hasattr(chunk, "sample_rate"):
                    sample_rate = chunk.sample_rate
        else:
            import io
            import wave

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)
            buf.seek(0)
            with wave.open(buf, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                chunks.append(arr)

        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        if req.speed and abs(req.speed - 1.0) > 1e-3:
            # Simple resample-by-index speed change (quality is secondary for A/B)
            indices = np.round(np.arange(0, len(audio), req.speed)).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]

        return SynthesisResult(
            audio=audio,
            sample_rate=int(sample_rate),
            backend_id=self.id,
            meta={"speaker": "emirati-female", "checkpoint": HF_REPO},
        )
