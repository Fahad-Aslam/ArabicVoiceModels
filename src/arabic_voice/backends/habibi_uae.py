from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from arabic_voice.base import BackendUnavailableError, MissingDependencyError, TTSBackend
from arabic_voice.device import pick_torch_device
from arabic_voice.paths import CACHE_DIR, REFS_DIR, ensure_dirs
from arabic_voice.types import ModelInfo, SynthesisRequest, SynthesisResult

HF_REPO = "SWivid/Habibi-TTS"
DEFAULT_DIALECT = "UAE"
DEFAULT_REF_TEXT = (
    "قمنا نشتريها بشكل متكرر أو لما نلقى ستايل يعجبنا وحياناً هذا الستايل ما نحبه."
)


class HabibiUAEBackend(TTSBackend):
    id = "habibi_uae"

    def __init__(
        self,
        dialect: str = DEFAULT_DIALECT,
        model_type: str = "Specialized",
        device: str | None = None,
        prefer_cli: bool = False,
    ) -> None:
        super().__init__()
        self._dialect = dialect
        self._model_type = model_type
        self._device = device
        self._prefer_cli = prefer_cli
        self._mode: str | None = None  # "cli" | "api"
        self._ema_model = None
        self._vocoder = None
        self._infer_fn = None
        self._cli_bin: str | None = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            id=self.id,
            name="Habibi-TTS UAE",
            description=(
                "F5-TTS dialectal Arabic. Specialized UAE or Unified with dialect=UAE. "
                "Best documented Emirati open model (non-commercial weights)."
            ),
            license="CC-BY-NC-SA-4.0 (UAE/Unified weights); code MIT",
            dialect="Emirati (UAE)",
            checkpoint=HF_REPO,
            mac_status="fair (PyTorch MPS/CPU; slow on CPU)",
            requires_ref_audio=True,
            requires_gpu_preferred=True,
            commercial_ok=False,
            extra="habibi",
        )

    def load(self) -> None:
        ensure_dirs()
        self._device = self._device or pick_torch_device(prefer_mps=True)

        cli = shutil.which("habibi-tts_infer-cli")
        if self._prefer_cli and cli:
            self._cli_bin = cli
            self._mode = "cli"
            self._loaded = True
            return

        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise MissingDependencyError(self.id, "habibi", str(exc)) from exc

        try:
            self._load_via_api()
            self._mode = "api"
            self._loaded = True
        except Exception as api_err:
            if cli:
                self._cli_bin = cli
                self._mode = "cli"
                self._loaded = True
                return
            raise BackendUnavailableError(
                f"Could not load Habibi UAE.\n{api_err}\n"
                f'Install: pip install -e ".[habibi]"'
            ) from api_err

    def _load_via_api(self) -> None:
        try:
            from habibi_tts.infer.utils_infer import (  # type: ignore
                infer_process,
                load_model,
                load_vocoder,
            )
        except ImportError:
            from f5_tts.infer.utils_infer import (  # type: ignore
                infer_process,
                load_model,
                load_vocoder,
            )

        from cached_path import cached_path

        if self._model_type.lower() == "unified":
            ckpt = str(cached_path(f"hf://{HF_REPO}/Unified/model_200000.safetensors"))
            vocab = str(cached_path(f"hf://{HF_REPO}/Unified/vocab.txt"))
        else:
            ckpt = str(
                cached_path(f"hf://{HF_REPO}/Specialized/UAE/model_100000.safetensors")
            )
            vocab = str(cached_path(f"hf://{HF_REPO}/Specialized/UAE/vocab.txt"))

        from f5_tts.model import DiT  # type: ignore

        try:
            from hydra.utils import get_class
            from omegaconf import OmegaConf
            from importlib.resources import files

            cfg_path = str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml"))
            model_cfg = OmegaConf.load(cfg_path)
            model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
            arch = model_cfg.model.arch
        except Exception:
            model_cls = DiT
            arch = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

        self._vocoder = load_vocoder(device=self._device)
        self._ema_model = load_model(
            model_cls,
            arch,
            ckpt,
            mel_spec_type="vocos",
            vocab_file=vocab,
            device=self._device,
        )
        self._infer_fn = infer_process

    def _resolve_ref(self, req: SynthesisRequest) -> tuple[str, str]:
        if req.ref_audio is not None:
            path = Path(req.ref_audio)
            if not path.exists():
                raise FileNotFoundError(f"Reference audio not found: {path}")
            text = (req.ref_text or DEFAULT_REF_TEXT).strip()
            return str(path), text

        for name in ("uae_ref.wav", "UAE.wav", "habibi_uae_ref.wav"):
            candidate = REFS_DIR / name
            if candidate.exists():
                txt = candidate.with_suffix(".txt")
                ref_text = (
                    txt.read_text(encoding="utf-8").strip()
                    if txt.exists()
                    else DEFAULT_REF_TEXT
                )
                if req.ref_text:
                    ref_text = req.ref_text.strip()
                return str(candidate), ref_text

        # Try package assets after pip install habibi-tts
        try:
            from importlib.resources import files

            assets = files("habibi_tts").joinpath("assets")
            for fname in ("UAE.wav", "UAE.mp3"):
                asset = assets.joinpath(fname)
                if asset.is_file():
                    cached = CACHE_DIR / "habibi_refs"
                    cached.mkdir(parents=True, exist_ok=True)
                    dest = cached / fname
                    dest.write_bytes(asset.read_bytes())
                    return str(dest), DEFAULT_REF_TEXT
        except Exception:
            pass

        raise FileNotFoundError(
            f"Habibi requires a UAE reference WAV + transcript. "
            f"Place {REFS_DIR / 'uae_ref.wav'} and {REFS_DIR / 'uae_ref.txt'}, "
            f"or pass --ref-audio / --ref-text."
        )

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        self.ensure_loaded()
        text = req.text.strip()
        if not text:
            raise ValueError("text must be non-empty")

        ref_audio, ref_text = self._resolve_ref(req)

        if self._mode == "cli":
            try:
                return self._synthesize_cli(text, ref_audio, ref_text)
            except Exception as cli_err:
                # CLI often breaks on torchaudio ABI mismatches; fall back to in-process API.
                if self._ema_model is None:
                    try:
                        self._load_via_api()
                    except Exception as api_load_err:
                        raise BackendUnavailableError(
                            f"Habibi CLI failed and API load failed.\nCLI: {cli_err}\nAPI: {api_load_err}"
                        ) from api_load_err
                self._mode = "api"
                return self._synthesize_api(text, ref_audio, ref_text)
        return self._synthesize_api(text, ref_audio, ref_text)

    def _synthesize_cli(self, text: str, ref_audio: str, ref_text: str) -> SynthesisResult:
        assert self._cli_bin
        with tempfile.TemporaryDirectory(prefix="habibi_") as tmp:
            out = Path(tmp) / "out.wav"
            cmd = [
                self._cli_bin,
                "--ref_audio",
                ref_audio,
                "--ref_text",
                ref_text,
                "--gen_text",
                text,
                "--dialect",
                self._dialect,
                "-o",
                str(out),
            ]
            # Output flag names differ across versions; try common ones
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError:
                cmd = [
                    self._cli_bin,
                    "--ref_audio",
                    ref_audio,
                    "--ref_text",
                    ref_text,
                    "--gen_text",
                    text,
                    "--dialect",
                    self._dialect,
                    "--output_dir",
                    tmp,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode != 0:
                    raise BackendUnavailableError(
                        f"habibi-tts_infer-cli failed:\n{proc.stderr or proc.stdout}"
                    )

            wav_path = out if out.exists() else None
            if wav_path is None:
                wavs = sorted(Path(tmp).rglob("*.wav"))
                if not wavs:
                    raise BackendUnavailableError(
                        f"Habibi CLI produced no WAV.\n{proc.stderr if 'proc' in dir() else ''}"
                    )
                wav_path = wavs[0]

            audio, sr = sf.read(str(wav_path), dtype="float32")
            return SynthesisResult(
                audio=np.asarray(audio, dtype=np.float32).squeeze(),
                sample_rate=int(sr),
                backend_id=self.id,
                meta={
                    "checkpoint": HF_REPO,
                    "dialect": self._dialect,
                    "mode": "cli",
                    "ref_audio": ref_audio,
                    "license_note": "UAE weights are CC-BY-NC-SA-4.0",
                    "device": self._device,
                },
            )

    def _synthesize_api(self, text: str, ref_audio: str, ref_text: str) -> SynthesisResult:
        assert self._ema_model is not None and self._infer_fn is not None
        try:
            audio, sample_rate, _ = self._infer_fn(
                ref_audio,
                ref_text,
                text,
                self._ema_model,
                self._vocoder,
                device=self._device,
            )
        except TypeError:
            audio, sample_rate, _ = self._infer_fn(
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=text,
                model_obj=self._ema_model,
                vocoder=self._vocoder,
                device=self._device,
            )

        wave = np.asarray(audio, dtype=np.float32).squeeze()
        return SynthesisResult(
            audio=wave,
            sample_rate=int(sample_rate),
            backend_id=self.id,
            meta={
                "checkpoint": HF_REPO,
                "dialect": self._dialect,
                "model_type": self._model_type,
                "mode": "api",
                "device": self._device,
                "ref_audio": ref_audio,
                "license_note": "UAE weights are CC-BY-NC-SA-4.0",
            },
        )
