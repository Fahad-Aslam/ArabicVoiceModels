#!/usr/bin/env python3
"""Streamlit latency lab: Habibi TTS + Whisper Gulf STT.

Run:
  pip install -e ".[habibi,asr-whisper,ui]"
  streamlit run scripts/streamlit_latency.py
"""

from __future__ import annotations

import io
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arabic_voice.paths import REFS_DIR, SCRIPTS_DIR, ensure_dirs  # noqa: E402
from arabic_voice.registry import get_backend  # noqa: E402
from arabic_voice.stt.registry import get_stt_backend  # noqa: E402
from arabic_voice.stt.types import TranscriptionRequest  # noqa: E402
from arabic_voice.types import SynthesisRequest  # noqa: E402

DEFAULT_REF_WAV = REFS_DIR / "uae_ref.wav"
DEFAULT_REF_TXT = REFS_DIR / "uae_ref.txt"
PROMPTS_FILE = SCRIPTS_DIR / "prompts_emirati.txt"


def _default_prompt() -> str:
    if not PROMPTS_FILE.exists():
        return "أهلًا، شحالك؟ تقدر تساعدني أحجز موعد؟"
    for line in PROMPTS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return "أهلًا، شحالك؟"


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    wav = np.asarray(audio, dtype=np.float32).squeeze()
    sf.write(buf, wav, sample_rate, format="WAV")
    return buf.getvalue()


def _audio_duration_s(audio: np.ndarray, sample_rate: int) -> float:
    wav = np.asarray(audio).squeeze()
    if sample_rate <= 0 or wav.size == 0:
        return 0.0
    return float(wav.size) / float(sample_rate)


def _load_uploaded_or_mic(file) -> tuple[np.ndarray, int]:
    """Return float32 mono waveform + sample rate from a Streamlit UploadedFile / audio_input."""
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav, int(sr)


@st.cache_resource(show_spinner="Loading Habibi UAE (first time can take a while)…")
def _habibi_backend():
    b = get_backend("habibi_uae", reload=True)
    t0 = time.perf_counter()
    b.ensure_loaded()
    load_ms = (time.perf_counter() - t0) * 1000.0
    return b, load_ms


@st.cache_resource(show_spinner="Loading Whisper Gulf (first time can take a while)…")
def _gulf_backend():
    b = get_stt_backend("whisper_gulf_small", reload=True)
    t0 = time.perf_counter()
    b.ensure_loaded()
    load_ms = (time.perf_counter() - t0) * 1000.0
    return b, load_ms


def _show_metrics(load_ms: float | None, infer_ms: float, audio_s: float) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Load (ms)", f"{load_ms:.0f}" if load_ms is not None else "— (cached)")
    c2.metric("Infer (ms)", f"{infer_ms:.0f}")
    c3.metric("Audio (s)", f"{audio_s:.2f}")
    rtf = (infer_ms / 1000.0) / audio_s if audio_s > 0 else float("nan")
    c4.metric("RTF", f"{rtf:.2f}×" if audio_s > 0 else "—")


def main() -> None:
    ensure_dirs()
    st.set_page_config(page_title="Arabic Voice Latency Lab", layout="wide")
    st.title("Latency lab — Habibi TTS + Whisper Gulf STT")
    st.caption(
        "Warm models stay in memory across runs. Metrics are wall-clock "
        "(not token streaming). Habibi UAE weights are **CC-BY-NC-SA-4.0** (eval only)."
    )

    with st.sidebar:
        st.header("Models")
        st.markdown(
            "- **Habibi TTS**: MPS preferred, CPU fallback\n"
            "- **Whisper Gulf**: CPU float32 (forced)\n"
        )
        if st.button("Warm both models", type="primary"):
            with st.spinner("Loading Habibi…"):
                _, h_load = _habibi_backend()
            with st.spinner("Loading Whisper Gulf…"):
                _, g_load = _gulf_backend()
            st.success(f"Warm. Habibi load {h_load:.0f} ms · Gulf load {g_load:.0f} ms")
        st.divider()
        st.caption("Ref defaults: `assets/refs/uae_ref.wav`")

    tab_tts, tab_stt = st.tabs(["Habibi TTS", "Whisper Gulf STT"])

    with tab_tts:
        st.subheader("Text → speech")
        text = st.text_area("Text", value=_default_prompt(), height=120)
        ref_col1, ref_col2 = st.columns(2)
        with ref_col1:
            ref_path = st.text_input("Ref WAV path", value=str(DEFAULT_REF_WAV))
        with ref_col2:
            default_ref_text = (
                DEFAULT_REF_TXT.read_text(encoding="utf-8").strip()
                if DEFAULT_REF_TXT.exists()
                else ""
            )
            ref_text = st.text_input("Ref transcript", value=default_ref_text)

        if st.button("Synthesize", type="primary", key="tts_go"):
            if not text.strip():
                st.error("Enter some text.")
            elif not Path(ref_path).exists():
                st.error(f"Ref WAV not found: {ref_path}")
            else:
                try:
                    backend, cold_load_ms = _habibi_backend()
                    show_load = cold_load_ms if "habibi_load_shown" not in st.session_state else None
                    st.session_state["habibi_load_shown"] = True
                    t0 = time.perf_counter()
                    result = backend.synthesize(
                        SynthesisRequest(
                            text=text.strip(),
                            ref_audio=Path(ref_path),
                            ref_text=ref_text.strip() or None,
                        )
                    )
                    infer_ms = (time.perf_counter() - t0) * 1000.0
                    audio_s = _audio_duration_s(result.audio, result.sample_rate)
                    _show_metrics(show_load, infer_ms, audio_s)
                    st.audio(_wav_bytes(result.audio, result.sample_rate), format="audio/wav")
                    if result.meta:
                        st.json(result.meta)
                except Exception as exc:
                    st.exception(exc)

    with tab_stt:
        st.subheader("Speech → text")
        st.caption("Record with the mic or upload a WAV/MP3. Prefer short clips for fair RTF.")
        mic = None
        if hasattr(st, "audio_input"):
            mic = st.audio_input("Record")
        else:
            st.warning("Upgrade Streamlit (≥1.32) for mic input.")
        upload = st.file_uploader("Or upload audio", type=["wav", "mp3", "flac", "ogg", "m4a"])

        source = mic or upload
        if st.button("Transcribe", type="primary", key="stt_go"):
            if source is None:
                st.error("Record or upload audio first.")
            else:
                try:
                    wav, sr = _load_uploaded_or_mic(source)
                    backend, cold_load_ms = _gulf_backend()
                    show_load = cold_load_ms if "gulf_load_shown" not in st.session_state else None
                    st.session_state["gulf_load_shown"] = True
                    t0 = time.perf_counter()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = Path(tmp.name)
                        sf.write(str(tmp_path), wav, sr)
                    try:
                        result = backend.transcribe(
                            TranscriptionRequest(audio=tmp_path, language="ar", sample_rate=sr)
                        )
                    finally:
                        tmp_path.unlink(missing_ok=True)
                    infer_ms = (time.perf_counter() - t0) * 1000.0
                    audio_s = _audio_duration_s(wav, sr)
                    _show_metrics(show_load, infer_ms, audio_s)
                    st.markdown("**Transcript**")
                    st.write(result.text or "_(empty)_")
                    if result.meta:
                        st.json(result.meta)
                except Exception as exc:
                    st.exception(exc)


if __name__ == "__main__":
    main()
