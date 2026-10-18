from __future__ import annotations

from arabic_voice.base import MissingDependencyError
from arabic_voice.stt.audio_io import load_audio_mono
from arabic_voice.stt.base import STTBackend
from arabic_voice.stt.types import STTModelInfo, TranscriptionRequest, TranscriptionResult

HF_REPO = "otozz/whisper-small-dialect_gulf"
# Checkpoint ships weights only — tokenizer/feature extractor from base Whisper-small
BASE_PROCESSOR = "openai/whisper-small"


class WhisperGulfSmallBackend(STTBackend):
    id = "whisper_gulf_small"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None
        self._device = "cpu"

    def info(self) -> STTModelInfo:
        return STTModelInfo(
            id=self.id,
            name="Whisper-small Gulf FT",
            description="otozz Whisper-small fine-tuned on MASC Gulf dialect data.",
            license="Whisper MIT lineage (check model card)",
            dialect="Gulf Arabic (MASC)",
            checkpoint=HF_REPO,
            mac_status="excellent (Transformers CPU; weights-only checkpoint)",
            commercial_ok=True,
            extra="asr-whisper",
        )

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor
        except ImportError as exc:
            raise MissingDependencyError(self.id, "asr-whisper", str(exc)) from exc

        # CPU float32 is reliable; MPS fp16 produced garbage (!!!) with this checkpoint
        self._device = "cpu"
        self._processor = WhisperProcessor.from_pretrained(BASE_PROCESSOR)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            HF_REPO, torch_dtype=torch.float32
        )
        # Repo generation_config has forced_decoder_ids with None language id — clear it
        if getattr(self._model, "generation_config", None) is not None:
            self._model.generation_config.forced_decoder_ids = None
        self._model.to(self._device)
        self._model.eval()
        self._loaded = True

    def _decode_chunk(self, wav, sr: int) -> str:
        import torch

        assert self._model is not None and self._processor is not None
        inputs = self._processor(wav, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(self._device)
        forced = self._processor.get_decoder_prompt_ids(language="ar", task="transcribe")
        with torch.inference_mode():
            predicted_ids = self._model.generate(
                input_features,
                forced_decoder_ids=forced,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=4,
            )
        return self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    def transcribe(self, req: TranscriptionRequest) -> TranscriptionResult:
        self.ensure_loaded()
        wav, sr = load_audio_mono(req, target_sr=16000)
        # Chunk ~10s — this FT can degenerate on longer single-pass decode
        chunk = int(10 * sr)
        pieces: list[str] = []
        for start in range(0, len(wav), chunk):
            piece = self._decode_chunk(wav[start : start + chunk], sr)
            if piece:
                pieces.append(piece)
        text = " ".join(pieces)
        return TranscriptionResult(
            text=text.strip(),
            backend_id=self.id,
            language=req.language or "ar",
            meta={"checkpoint": HF_REPO, "device": self._device, "processor": BASE_PROCESSOR},
        )
