from __future__ import annotations

from pathlib import Path

# src/arabic_voice/paths.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
REFS_DIR = ASSETS_DIR / "refs"
OUTPUTS_DIR = REPO_ROOT / "outputs"
SCRIPTS_DIR = REPO_ROOT / "scripts"
CACHE_DIR = Path.home() / ".cache" / "arabic_voice"


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REFS_DIR.mkdir(parents=True, exist_ok=True)
