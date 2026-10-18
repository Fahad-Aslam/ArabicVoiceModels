from __future__ import annotations

from arabic_voice.base import MissingDependencyError
from arabic_voice.stt.audio_io import load_audio_mono
from arabic_voice.stt.base import STTBackend
from arabic_voice.stt.types import STTModelInfo, TranscriptionRequest, TranscriptionResult

HF_REPO = "eabayed/wav2vec2emiratidialict_1"


class Wav2VecEmiratiBackend(STTBackend):
    id = "wav2vec_emirati"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None

    def info(self) -> STTModelInfo:
        return STTModelInfo(
            id=self.id,
            name="Wav2Vec2 Emirati (legacy)",
            description="Early Emirati TV/shows CTC model (GPL-3.0).",
            license="GPL-3.0",
            dialect="Emirati (legacy)",
            checkpoint=HF_REPO,
            mac_status="good (Transformers CPU/MPS)",
            commercial_ok=False,
            extra="asr-wav2vec",
        )

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError as exc:
            raise MissingDependencyError(self.id, "asr-wav2vec", str(exc)) from exc

        try:
            self._processor = Wav2Vec2Processor.from_pretrained(HF_REPO)
            self._model = Wav2Vec2ForCTC.from_pretrained(HF_REPO)
        except Exception:
            self._processor = AutoProcessor.from_pretrained(HF_REPO)
            self._model = AutoModelForCTC.from_pretrained(HF_REPO)

        self._model.eval()
        device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        self._model.to(device)
        self._device = device
        self._loaded = True

    def transcribe(self, req: TranscriptionRequest) -> TranscriptionResult:
        self.ensure_loaded()
        assert self._model is not None and self._processor is not None
        import torch

        wav, sr = load_audio_mono(req, target_sr=16000)
        inputs = self._processor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self._device)
        with torch.inference_mode():
            logits = self._model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text = self._processor.batch_decode(predicted_ids)[0]
        # Some CTC checkpoints emit literal [PAD] tokens in decode
        text = text.replace("[PAD]", "").replace("|", " ")
        text = " ".join(text.split())
        return TranscriptionResult(
            text=text.strip(),
            backend_id=self.id,
            language="ar",
            meta={"checkpoint": HF_REPO, "device": self._device},
        )
