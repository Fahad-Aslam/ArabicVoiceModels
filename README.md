# Arabic Voice Models — Emirati TTS + STT Multi-Wrapper (Mac)

Shared Python interface + CLI to **qualitatively A/B open-source Emirati / Gulf Arabic TTS and STT** backends on this Mac (CPU / Apple MPS / MLX). Pick winners for your orchestrator after listening and reading transcripts.

STT research notes: [`docs/STT_RESEARCH.md`](docs/STT_RESEARCH.md).

## Environment (important)

Use **Python 3.13** (not 3.14). Torch / NeuCodec / NeMo stacks break on 3.14 with current `torchao`.

```bash
cd /Users/fahad/Documents/ArabicVoiceModels
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[piper]"          # lightest — start here
arabic-voice list
arabic-voice synth -b piper_emirati -t "أهلًا، شحالك؟" -o outputs/piper_hello.wav
```

Install additional backends as needed (prefer **one extra group at a time** — Habibi/Qwen/NeMo pin different `transformers` / `huggingface_hub` versions):

```bash
pip install -e ".[audar]"    # GGUF Audar Flash (CPU) — needs HF login for NeuCodec
pip install -e ".[habibi]"   # Habibi UAE (MPS/CPU; NC license)
pip install -e ".[qwen]"     # Qwen3 Emirati (~1.7B; may OOM)
# emirati_vits → use a dedicated venv (see below)
```

If this machine already has a working env at `.venv313`, activate that instead of recreating.

### Audar: gated NeuCodec

`audar_flash` downloads GGUF freely, but encoding/decoding uses gated [`neuphonic/neucodec`](https://huggingface.co/neuphonic/neucodec):

1. Accept the model terms on Hugging Face  
2. `huggingface-cli login` (or `export HF_TOKEN=hf_...`)  
3. Re-run synthesis

## Demo path map (~20s samples)

Checked-in qualitative demos (gitignored elsewhere under `outputs/`):

| Path | What |
|------|------|
| `outputs/tts_demo/` | ~20s Emirati TTS WAVs + `prompt.txt` + per-backend notes |
| `outputs/asr_demo/` | Same audio transcribed by STT backends → `*.txt` transcripts |
| `outputs/tts_demo/README.md` | TTS listen order |
| `outputs/asr_demo/README.md` | ASR compare notes |

## Orchestrator API

```python
from arabic_voice import get_backend, SynthesisRequest
from arabic_voice import get_stt_backend, TranscriptionRequest

tts = get_backend("piper_emirati")
result = tts.synthesize(SynthesisRequest(text="ترا الجو وايد حلو اليوم"))
result.write_wav("outputs/out.wav")

asr = get_stt_backend("whisper_baseline")
tr = asr.transcribe(TranscriptionRequest(audio="outputs/out.wav", language="ar"))
print(tr.text)
```

TTS backends implement `info` / `load` / `synthesize`. STT backends implement `info` / `load` / `transcribe`.

## TTS backends

| ID | Model | License | Mac | Commercial |
|----|-------|---------|-----|------------|
| `piper_emirati` | [Piper Emirati female](https://huggingface.co/vadimbelsky/arabic-emirati-female-piper) | MIT | Excellent (ONNX/CPU) | Yes |
| `audar_flash` | [Audar-TTS V1 Flash](https://huggingface.co/audarai/Audar-TTS-V1-Flash) | AudarAI OL | Good (GGUF/CPU; HF login) | Yes |
| `habibi_uae` | [Habibi-TTS UAE](https://huggingface.co/SWivid/Habibi-TTS) | **CC-BY-NC-SA-4.0** (UAE weights) | Fair (MPS/CPU) | **No** |
| `emirati_vits` | [Emirati VITS male](https://huggingface.co/vadimbelsky/emirati-vits-male-1.0) | Apache-2.0 | Challenging (NeMo/Pynini) | Yes |
| `qwen_emirati` | [Qwen3.5-TTS Emirati](https://huggingface.co/vadimbelsky/qwen3.5-TTS-Emirati) | Apache-2.0 | Heavy / may OOM | Yes |

Deferred on Mac: Magpie-Emirates, Emirati FastPitch (CUDA/NeMo Magpie pins).

## STT backends

| ID | Model | License | Mac | Commercial |
|----|-------|---------|-----|------------|
| `jais_asr` | [IFM/Jais-ASR](https://huggingface.co/IFM/Jais-ASR) | Apache-2.0 | Heavy (~12GB; often OOM) | Yes |
| `audar_asr_flash` | [Audar-ASR V1 Flash](https://huggingface.co/audarai/Audar-ASR-V1-Flash) | AudarAI OL | Good (Transformers) | Yes |
| `qwen_asr_uae` | [Qwen3-ASR Arabic UAE](https://huggingface.co/vadimbelsky/qwen3-asr-arabic-uae) | Apache-2.0 | Heavy (MPS; may OOM) | Yes |
| `whisper_baseline` | Whisper large-v3-turbo (mlx / faster-whisper) | MIT | Excellent | Yes |
| `whisper_arabic_ft` | [Arabic 4-dialect FT](https://huggingface.co/dev-ahmedhany/whisper-large-v3-arabic-ft-v3) | Apache-2.0 | Good (CT2 int8) | Yes |
| `whisper_gulf_small` | [Gulf Whisper-small](https://huggingface.co/otozz/whisper-small-dialect_gulf) | MIT | Excellent | Yes |
| `wav2vec_emirati` | [wav2vec2 Emirati](https://huggingface.co/eabayed/wav2vec2emiratidialict_1) | **GPL-3.0** | Good | **No** (GPL) |

Install STT extras (prefer one group at a time if deps conflict):

```bash
pip install -e ".[asr-whisper]"   # mlx-whisper + faster-whisper — start here
pip install -e ".[asr-qwen]"      # Audar ASR Flash + Qwen ASR UAE
pip install -e ".[asr-jais]"      # Jais-ASR (large)
pip install -e ".[asr-wav2vec]"   # legacy Emirati CTC (GPL)
```

## CLI

```bash
# TTS
arabic-voice list
arabic-voice synth -b piper_emirati -t "هني المكتب، تفضل اقعد"
arabic-voice compare -t "أهلًا، شحالك؟" --backends piper_emirati,audar_flash,habibi_uae
arabic-voice compare-prompts --backends piper_emirati --limit 3

# Cloning backends need a short ref clip — see assets/refs/README.md
arabic-voice synth -b habibi_uae -t "ترا الدرب صوب دبي اليوم وايد زحمة" \
  --ref-audio assets/refs/uae_ref.wav --ref-text "$(cat assets/refs/uae_ref.txt)"

# STT
arabic-voice asr list
arabic-voice asr transcribe -b whisper_baseline -a outputs/tts_demo/piper_emirati.wav
arabic-voice asr compare -a outputs/tts_demo/piper_emirati.wav \
  --backends whisper_baseline,whisper_gulf_small,whisper_arabic_ft
```

Prompts live in [`scripts/prompts_emirati.txt`](scripts/prompts_emirati.txt). WAVs / transcripts land under `outputs/`.

## Streamlit latency lab (Habibi + Whisper Gulf)

Interactive wall-clock timing for Habibi TTS and Whisper Gulf STT (warm models via `@st.cache_resource`):

```bash
source .venv313/bin/activate   # or your Python 3.13 venv
pip install -e ".[habibi,asr-whisper,ui]"
streamlit run scripts/streamlit_latency.py
```

Use the **Habibi TTS** tab to synthesize and hear audio; use **Whisper Gulf STT** with mic or upload. Metrics show load ms, infer ms, audio duration, and RTF.

## Reference audio (Habibi / Audar)

Place:

- `assets/refs/uae_ref.wav` + `uae_ref.txt`
- optional `assets/refs/audar_ref.wav` + `audar_ref.txt`

Bootstrap a ref from Piper:

```bash
arabic-voice synth -b piper_emirati \
  -t "$(cat assets/refs/uae_ref.txt)" \
  -o assets/refs/uae_ref.wav
cp assets/refs/uae_ref.wav assets/refs/audar_ref.wav
cp assets/refs/uae_ref.txt assets/refs/audar_ref.txt
```

## Emirati VITS on macOS (dedicated venv)

`emirati_vits` needs the **ScienceSoft NeMo fork** (Emirati G2P). Stock NeMo 3 removed `VitsModel`, and NeMo deps conflict with Habibi/Qwen in one venv.

**Recommended:** follow the upstream Mac guide in a separate environment:

1. [INSTALL_MAC.md](https://huggingface.co/vadimbelsky/emirati-vits-male-1.0/blob/main/INSTALL_MAC.md)  
2. `pip install -e ".[vits]"` in that venv  
3. `pip install --no-build-isolation 'git+https://github.com/VadzimBelski-ScienceSoft/NeMo.git'`  

Then `arabic-voice synth -b emirati_vits -t "..."`. Without the bilingual normalizer, numbers/dates may sound worse.

## Qwen on Mac

The Emirati fine-tune is ~1.7B. Expect MPS memory pressure. On OOM the backend raises a clear error — use Piper/VITS instead for long prompts.

## Qualitative workflow

1. `pip install -e ".[piper,audar,habibi]"` on Python 3.13  
2. Bootstrap refs (above); log in to HF for Audar  
3. `arabic-voice compare-prompts --backends piper_emirati,audar_flash,habibi_uae`  
4. Listen under `outputs/prompts_*/`  
5. Tell us which backend to finalize for the orchestrator  

## License notes

- This **wrapper code** is MIT.  
- **Habibi UAE / Unified** TTS weights are non-commercial (SADA/Mixat). Fine for local eval; not for commercial redistribution without a separate license.  
- **`wav2vec_emirati`** is GPL-3.0 — keep it out of proprietary product binaries unless you accept GPL terms.  
- Always check each checkpoint card before shipping product audio or transcripts.  
