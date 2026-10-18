from __future__ import annotations

from arabic_voice.base import MissingDependencyError
from arabic_voice.stt.audio_io import load_audio_mono
from arabic_voice.stt.base import STTBackend
from arabic_voice.stt.types import STTModelInfo, TranscriptionRequest, TranscriptionResult

HF_REPO = "dev-ahmedhany/whisper-large-v3-arabic-ft-v3"
CT2_REPO = "dev-ahmedhany/whisper-large-v3-arabic-ft-v3-ct2-int8"


class WhisperArabicFTBackend(STTBackend):
    id = "whisper_arabic_ft"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None
        self._fw = None
        self._mode: str | None = None
        self._device = "cpu"

    def info(self) -> STTModelInfo:
        return STTModelInfo(
            id=self.id,
            name="Whisper-large-v3 Arabic 4-dialect FT",
            description="QLoRA-merged Whisper with measured Gulf/Casablanca UAE WER lift.",
            license="Apache-2.0",
            dialect="MSA + Egyptian + Levantine + Gulf/UAE",
            checkpoint=HF_REPO,
            mac_status="good (prefer CT2 int8 on CPU)",
            requires_gpu_preferred=True,
            commercial_ok=True,
            extra="asr-whisper",
        )

    def load(self) -> None:
        # Prefer faster-whisper CT2 on CPU (lighter). Fall back to transformers generate.
        try:
            from faster_whisper import WhisperModel

            try:
                self._fw = WhisperModel(CT2_REPO, device="cpu", compute_type="int8")
                self._mode = "ct2"
                self._loaded = True
                return
            except Exception:
                self._fw = None
        except ImportError:
            pass

        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor
        except ImportError as exc:
            raise MissingDependencyError(self.id, "asr-whisper", str(exc)) from exc

        self._device = (
            "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        dtype = torch.float16 if self._device == "mps" else torch.float32
        self._processor = WhisperProcessor.from_pretrained(HF_REPO)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(HF_REPO, torch_dtype=dtype)
        self._model.to(self._device)
        self._model.eval()
        self._mode = "hf"
        self._loaded = True

    def transcribe(self, req: TranscriptionRequest) -> TranscriptionResult:
        self.ensure_loaded()
        wav, sr = load_audio_mono(req, target_sr=16000)

        if self._mode == "ct2":
            assert self._fw is not None
            segments, info = self._fw.transcribe(wav, language="ar")
            text = "".join(s.text for s in segments).strip()
            return TranscriptionResult(
                text=text,
                backend_id=self.id,
                language=getattr(info, "language", "ar"),
                meta={"checkpoint": CT2_REPO, "engine": "faster-whisper-ct2"},
            )

        assert self._model is not None and self._processor is not None
        import torch

        inputs = self._processor(wav, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(self._device)
        if self._device == "mps":
            input_features = input_features.half()

        forced = None
        try:
            forced = self._processor.get_decoder_prompt_ids(language="arabic", task="transcribe")
        except Exception:
            forced = None

        with torch.inference_mode():
            predicted_ids = self._model.generate(
                input_features,
                forced_decoder_ids=forced,
                max_new_tokens=256,
            )
        text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return TranscriptionResult(
            text=text.strip(),
            backend_id=self.id,
            language=req.language or "ar",
            meta={"checkpoint": HF_REPO, "engine": "transformers", "device": self._device},
        )
