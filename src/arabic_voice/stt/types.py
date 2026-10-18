from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class STTModelInfo:
    id: str
    name: str
    description: str
    license: str
    dialect: str
    checkpoint: str
    mac_status: str
    requires_gpu_preferred: bool = False
    commercial_ok: bool = True
    extra: str | None = None
    streaming: bool = False


@dataclass
class TranscriptionRequest:
    audio: Path | str | np.ndarray
    language: str | None = "ar"
    sample_rate: int | None = None


@dataclass
class TranscriptionResult:
    text: str
    backend_id: str
    language: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def write_text(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.text.strip() + "\n", encoding="utf-8")
        return out
