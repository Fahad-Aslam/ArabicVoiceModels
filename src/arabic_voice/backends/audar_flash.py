from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from arabic_voice.base import BackendUnavailableError, MissingDependencyError, TTSBackend
from arabic_voice.paths import CACHE_DIR, REFS_DIR, ensure_dirs
from arabic_voice.types import ModelInfo, SynthesisRequest, SynthesisResult

HF_REPO = "audarai/Audar-TTS-V1-Flash"
GGUF_FILE = "Audar-TTS-V1-Flash-Q4_K_M.gguf"
CODEC_REPO = "neuphonic/neucodec"
DEFAULT_REF_TEXT = "مرحبا، كيف حالك اليوم؟"

_GATED_HELP = (
    f"Audar needs the gated Hugging Face codec `{CODEC_REPO}`.\n"
    "1) Create a free HF account and accept the model terms at:\n"
    f"   https://huggingface.co/{CODEC_REPO}\n"
    "2) Authenticate locally:\n"
    "   huggingface-cli login\n"
    "   # or: export HF_TOKEN=hf_...\n"
    "Then re-run synthesis."
)


class AudarFlashBackend(TTSBackend):
    id = "audar_flash"

    def __init__(
        self,
        gguf_path: Path | None = None,
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
    ) -> None:
        super().__init__()
        self._gguf_path = gguf_path
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._llm = None
        self._codec = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            id=self.id,
            name="Audar-TTS V1 Flash",
            description=(
                "Arabic-first zero-shot TTS (GGUF). Claims MSA + Gulf/Emirati + English. "
                f"Requires HF access to gated `{CODEC_REPO}`."
            ),
            license="AudarAI Open License v1.0",
            dialect="Arabic-first (Gulf/Emirati claimed)",
            checkpoint=HF_REPO,
            mac_status="good (CPU GGUF via llama.cpp; needs HF login for codec)",
            requires_ref_audio=True,
            requires_gpu_preferred=False,
            commercial_ok=True,
            extra="audar",
        )

    def load(self) -> None:
        try:
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama
            from neucodec import NeuCodec
        except ImportError as exc:
            raise MissingDependencyError(
                self.id,
                "audar",
                f"{exc}\nTip: use Python 3.13 and pin torchao<0.15 (see README).",
            ) from exc

        ensure_dirs()
        model_dir = CACHE_DIR / "audar_flash"
        model_dir.mkdir(parents=True, exist_ok=True)

        gguf = self._gguf_path
        if gguf is None:
            gguf = Path(
                hf_hub_download(
                    repo_id=HF_REPO,
                    filename=GGUF_FILE,
                    local_dir=str(model_dir),
                    token=os.environ.get("HF_TOKEN"),
                )
            )

        self._llm = Llama(
            model_path=str(gguf),
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=False,
        )

        try:
            self._codec = NeuCodec.from_pretrained(
                CODEC_REPO,
                token=os.environ.get("HF_TOKEN"),
            ).eval()
        except Exception as exc:
            msg = str(exc).lower()
            if "gated" in msg or "401" in msg or "unauthorized" in msg or "restricted" in msg:
                raise BackendUnavailableError(_GATED_HELP) from exc
            raise BackendUnavailableError(
                f"Failed to load NeuCodec ({CODEC_REPO}): {exc}\n{_GATED_HELP}"
            ) from exc

        self._loaded = True

    def _default_ref(self) -> tuple[Path, str]:
        candidates = [
            REFS_DIR / "audar_ref.wav",
            REFS_DIR / "uae_ref.wav",
        ]
        for path in candidates:
            if path.exists():
                txt_path = path.with_suffix(".txt")
                ref_text = (
                    txt_path.read_text(encoding="utf-8").strip()
                    if txt_path.exists()
                    else DEFAULT_REF_TEXT
                )
                return path, ref_text
        raise FileNotFoundError(
            f"Audar requires a 5–15s reference WAV. Place one at {REFS_DIR / 'audar_ref.wav'} "
            f"(and optional {REFS_DIR / 'audar_ref.txt'} transcript), or pass --ref-audio / --ref-text."
        )

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        self.ensure_loaded()
        assert self._llm is not None and self._codec is not None

        import torch

        text = req.text.strip()
        if not text:
            raise ValueError("text must be non-empty")

        if req.ref_audio is not None:
            ref_path = Path(req.ref_audio)
            ref_text = (req.ref_text or DEFAULT_REF_TEXT).strip()
        else:
            ref_path, ref_text = self._default_ref()
            if req.ref_text:
                ref_text = req.ref_text.strip()

        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")

        # NeuCodec expects 16 kHz mono; encode_code accepts path or tensor depending on version
        try:
            ref_codes = self._codec.encode_code(str(ref_path)).squeeze().tolist()
        except Exception:
            import librosa

            wav, _ = librosa.load(str(ref_path), sr=16000, mono=True)
            ref_codes = (
                self._codec.encode_code(torch.from_numpy(wav)[None, None, :])
                .squeeze()
                .tolist()
            )

        ref = "".join(f"<|speech_{c}|>" for c in ref_codes)

        prompt = (
            "user: Convert the text to speech:"
            f"<|REF_TEXT_START|>{ref_text}<|REF_TEXT_END|>"
            f"<|REF_SPEECH_START|>{ref}<|REF_SPEECH_END|>"
            f"<|TARGET_TEXT_START|>{text}<|TARGET_TEXT_END|>"
            "\nassistant:<|TARGET_CODES_START|>"
        )

        tce = self._llm.tokenize(b"<|TARGET_CODES_END|>", add_bos=False, special=True)[0]
        toks = self._llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        ids: list[int] = []
        for tid in self._llm.generate(
            toks,
            temp=1.0,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
        ):
            if tid == tce or len(ids) >= 2048:
                break
            ids.append(tid)

        decoded = "".join(
            self._llm.detokenize([t], special=True).decode("utf-8", "ignore") for t in ids
        )
        codes = [int(x) for x in re.findall(r"<\|speech_(\d+)\|>", decoded)]
        if not codes:
            raise RuntimeError(
                "Audar produced no speech codes; try a clearer ref clip or shorter text."
            )

        wav = self._codec.decode_code(torch.tensor(codes)[None, None, :]).cpu().numpy()[0, 0, :]
        audio = np.asarray(wav, dtype=np.float32)

        return SynthesisResult(
            audio=audio,
            sample_rate=24000,
            backend_id=self.id,
            meta={
                "checkpoint": HF_REPO,
                "ref_audio": str(ref_path),
                "gguf": GGUF_FILE,
                "codec": CODEC_REPO,
            },
        )
