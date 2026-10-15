from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from arabic_voice.base import BackendUnavailableError, MissingDependencyError, TTSBackend
from arabic_voice.device import pick_torch_device
from arabic_voice.paths import CACHE_DIR, ensure_dirs
from arabic_voice.types import ModelInfo, SynthesisRequest, SynthesisResult

HF_REPO = "vadimbelsky/emirati-vits-male-1.0"
CKPT_NAME = "VITS_emirati_v3--loss_gen_all=20.4726-epoch=179-last.ckpt"
HPARAMS_NAME = "hparams.yaml"


def _import_vits_model():
    """Needs ScienceSoft/NeMo 2.x fork with VitsModel (not stock NeMo 3)."""
    try:
        from nemo.collections.tts.models import VitsModel

        return VitsModel
    except ImportError as exc:
        raise BackendUnavailableError(
            "Emirati VITS needs the ScienceSoft NeMo fork (VitsModel + Emirati G2P).\n"
            "Use a dedicated venv — it conflicts with Habibi/Qwen deps. Follow:\n"
            "  https://huggingface.co/vadimbelsky/emirati-vits-male-1.0/blob/main/INSTALL_MAC.md\n"
            "Then: pip install -e '.[vits]' and\n"
            "  pip install --no-build-isolation "
            "'git+https://github.com/VadzimBelski-ScienceSoft/NeMo.git'\n"
            f"Detail: {exc}"
        ) from exc


class EmiratiVITSBackend(TTSBackend):
    id = "emirati_vits"

    def __init__(self, device: str | None = None, require_normalize: bool = True) -> None:
        super().__init__()
        self._device = device
        self._require_normalize = require_normalize
        self._model = None
        self._normalizer = None
        self._tn_warning: str | None = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            id=self.id,
            name="Emirati VITS Male 1.0",
            description=(
                "NeMo VITS bilingual Emirati Arabic + English with Emirati G2P and "
                "ar_ae text normalization (numbers, dates, currency). Requires NeMo 2.x."
            ),
            license="Apache-2.0",
            dialect="Emirati (ar-AE) + English code-switch",
            checkpoint=HF_REPO,
            mac_status="challenging (NeMo 2.x + Pynini/OpenFst on macOS)",
            requires_ref_audio=False,
            requires_gpu_preferred=False,
            commercial_ok=True,
            extra="vits",
        )

    def load(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise MissingDependencyError(
                self.id,
                "vits",
                f"{exc}\nOn macOS, Pynini/OpenFst often fails — see README.",
            ) from exc

        VitsModel = _import_vits_model()

        ensure_dirs()
        self._device = self._device or pick_torch_device(prefer_mps=False)
        if self._device == "mps":
            self._device = "cpu"

        from huggingface_hub import snapshot_download

        model_dir = CACHE_DIR / "emirati_vits"
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(repo_id=HF_REPO, local_dir=str(model_dir))
        except Exception as exc:
            raise BackendUnavailableError(
                f"Failed to download {HF_REPO}: {exc}"
            ) from exc

        ckpt = model_dir / CKPT_NAME
        if not ckpt.exists():
            ckpts = list(model_dir.rglob("*.ckpt")) + list(model_dir.rglob("*.nemo"))
            if not ckpts:
                raise BackendUnavailableError(f"No checkpoint found under {model_dir}")
            ckpt = ckpts[0]

        map_location = torch.device(self._device)
        try:
            self._model = VitsModel.restore_from(str(ckpt), map_location=map_location)
        except Exception:
            self._model = VitsModel.load_from_checkpoint(str(ckpt), map_location=map_location)
        self._model.eval()

        self._normalizer = None
        self._tn_warning = None
        if self._require_normalize:
            self._normalizer = self._load_normalizer(model_dir)

        self._loaded = True

    def _load_normalizer(self, model_dir: Path):
        try:
            from bilingual_text_normalizer import BilingualTextNormalizer  # type: ignore

            return BilingualTextNormalizer(ar_lang="ar_ae", en_lang="en")
        except ImportError:
            pass

        try:
            norm_path = model_dir / "bilingual_text_normalizer.py"
            if not norm_path.exists():
                found = list(model_dir.rglob("bilingual_text_normalizer.py"))
                if not found:
                    self._tn_warning = "bilingual_text_normalizer.py not found in model repo"
                    return None
                norm_path = found[0]
            spec = importlib.util.spec_from_file_location(
                "bilingual_text_normalizer", norm_path
            )
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            sys.modules["bilingual_text_normalizer"] = mod
            spec.loader.exec_module(mod)
            return mod.BilingualTextNormalizer(ar_lang="ar_ae", en_lang="en")
        except Exception as norm_err:
            self._tn_warning = str(norm_err)
            return None

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        self.ensure_loaded()
        assert self._model is not None

        import torch

        text = req.text.strip()
        if not text:
            raise ValueError("text must be non-empty")

        if self._normalizer is not None:
            text = self._normalizer.normalize(text)

        with torch.inference_mode():
            if hasattr(self._model, "nemo_infer"):
                with self._model.nemo_infer():
                    tokens = self._model.parse(text)
                    audio = self._model.convert_text_to_waveform(tokens=tokens)
            else:
                tokens = self._model.parse(text)
                audio = self._model.convert_text_to_waveform(tokens=tokens)

        wave = audio.squeeze().detach().cpu().numpy().astype(np.float32)
        meta = {
            "checkpoint": HF_REPO,
            "device": self._device,
            "speaker": "emirati-male",
            "normalized": self._normalizer is not None,
        }
        if self._tn_warning:
            meta["tn_warning"] = self._tn_warning

        return SynthesisResult(
            audio=wave,
            sample_rate=22050,
            backend_id=self.id,
            meta=meta,
        )
