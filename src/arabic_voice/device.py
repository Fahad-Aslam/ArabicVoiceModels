from __future__ import annotations


def pick_torch_device(prefer_mps: bool = True) -> str:
    """Return best available torch device for this Mac (mps > cpu)."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for this backend") from exc

    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache(device: str) -> None:
    try:
        import torch
    except ImportError:
        return
    if device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
