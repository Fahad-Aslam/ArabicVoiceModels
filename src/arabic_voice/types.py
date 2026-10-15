from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModelInfo:
    id: str
    name: str
    description: str
    license: str
    dialect: str
    checkpoint: str
    mac_status: str
    requires_ref_audio: bool = False
    requires_gpu_preferred: bool = False
    commercial_ok: bool = True
    extra: str | None = None


@dataclass
class SynthesisRequest:
    text: str
    language: str = "ar-AE"
    speaker: str | None = None
    ref_audio: Path | None = None
    ref_text: str | None = None
    speed: float = 1.0
    seed: int | None = None


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    backend_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    def write_wav(self, path: str | Path) -> Path:
        import soundfile as sf

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        audio = np.asarray(self.audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        sf.write(str(out), audio, self.sample_rate)
        return out
