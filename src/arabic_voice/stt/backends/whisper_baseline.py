from __future__ import annotations

from arabic_voice.base import MissingDependencyError
from arabic_voice.stt.audio_io import audio_path, load_audio_mono
from arabic_voice.stt.base import STTBackend
from arabic_voice.stt.types import STTModelInfo, TranscriptionRequest, TranscriptionResult

DEFAULT_MODEL = "large-v3-turbo"


class WhisperBaselineBackend(STTBackend):
    id = "whisper_baseline"

    def __init__(self, model_size: str = DEFAULT_MODEL) -> None:
        super().__init__()
        self._model_size = model_size
        self._engine = None
        self._mode: str | None = None  # mlx | faster

    def info(self) -> STTModelInfo:
        return STTModelInfo(
            id=self.id,
            name="Whisper large-v3-turbo (baseline)",
            description="Mac-first multilingual Whisper via mlx-whisper or faster-whisper.",
            license="MIT",
            dialect="Multilingual Arabic (MSA-biased) + EN code-switch",
            checkpoint=f"openai/whisper-{self._model_size}",
            mac_status="excellent (mlx Metal or faster-whisper CPU)",
            requires_gpu_preferred=False,
            commercial_ok=True,
            extra="asr-whisper",
            streaming=False,
        )

    def _load_faster(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MissingDependencyError(
                self.id,
                "asr-whisper",
                f"{exc}\nInstall: pip install -e '.[asr-whisper]'",
            ) from exc

        self._engine = WhisperModel(self._model_size, device="cpu", compute_type="int8")
        self._mode = "faster"
        self._loaded = True

    def load(self) -> None:
        # Prefer MLX on Apple Silicon; fall back to faster-whisper on any failure
        try:
            import mlx_whisper  # noqa: F401

            self._engine = mlx_whisper
            self._mode = "mlx"
            self._mlx_repo = f"mlx-community/whisper-{self._model_size}"
            self._loaded = True
            return
        except ImportError:
            pass

        self._load_faster()

    def transcribe(self, req: TranscriptionRequest) -> TranscriptionResult:
        self.ensure_loaded()
        lang = req.language or "ar"

        if self._mode == "mlx":
            path = audio_path(req)
            try:
                result = self._engine.transcribe(
                    str(path),
                    path_or_hf_repo=self._mlx_repo,
                    language=None if lang == "auto" else lang,
                    word_timestamps=False,
                )
            except Exception:
                # Repo missing / auth / Metal issue → CPU faster-whisper
                self._engine = None
                self._mode = None
                self._loaded = False
                self._load_faster()
            else:
                text = (result.get("text") or "").strip()
                return TranscriptionResult(
                    text=text,
                    backend_id=self.id,
                    language=lang,
                    meta={"engine": "mlx-whisper", "model": self._mlx_repo},
                )

        try:
            path = audio_path(req)
            segments, info = self._engine.transcribe(
                str(path), language=None if lang == "auto" else lang
            )
        except TypeError:
            wav, _sr = load_audio_mono(req, target_sr=16000)
            segments, info = self._engine.transcribe(
                wav, language=None if lang == "auto" else lang
            )
        text = "".join(seg.text for seg in segments).strip()
        return TranscriptionResult(
            text=text,
            backend_id=self.id,
            language=getattr(info, "language", lang),
            meta={"engine": "faster-whisper", "model": self._model_size},
        )
