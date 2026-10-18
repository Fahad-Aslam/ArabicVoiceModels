from __future__ import annotations

from pathlib import Path

import numpy as np

from arabic_voice.stt.types import TranscriptionRequest


def load_audio_mono(req: TranscriptionRequest, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Return float32 mono waveform and sample rate."""
    if isinstance(req.audio, np.ndarray):
        wav = np.asarray(req.audio, dtype=np.float32).squeeze()
        sr = req.sample_rate or target_sr
        if sr != target_sr:
            try:
                import librosa

                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except ImportError:
                pass
        return wav, sr

    path = Path(req.audio)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")

    try:
        import librosa

        wav, sr = librosa.load(str(path), sr=target_sr, mono=True)
        return np.asarray(wav, dtype=np.float32), int(sr)
    except ImportError:
        import soundfile as sf

        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        return wav, int(sr)


def audio_path(req: TranscriptionRequest) -> Path:
    if isinstance(req.audio, (str, Path)):
        path = Path(req.audio)
        if not path.exists():
            raise FileNotFoundError(f"Audio not found: {path}")
        return path
    raise TypeError("This backend requires a filesystem path, not an in-memory array")
