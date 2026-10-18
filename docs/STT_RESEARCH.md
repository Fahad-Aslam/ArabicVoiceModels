# Emirati / Gulf Arabic STT Research

Open-source speech-to-text for an Emirati listening orchestrator (Mac-first). Companion to the TTS bake-off in this repo.

## Reality check

- **2024 Mixat era:** zero-shot Whisper/MMS/ArTST on Emirati–English podcasts was essentially unusable (often **>100% WER**; Mixat medium Whisper ~**168%** overall).
- **Casablanca UAE:** Whisper-large-v3 zero-shot still ~**50–60%+ WER**.
- **2026 leap:** **Jais-ASR** and **Audar-ASR** publish the first credible open Mixat Emirati numbers (~**18–19% WER**). Falcon-ASR looks strong in demos but has **no public weights**.
- Treat synthetic UAE WER claims (e.g. Qwen UAE FT ~10%) as **unverified** until A/B on your own mic audio / Mixat Part2 / Casablanca UAE.

## Recommended wrap shortlist

| Priority | ID | Model | Emirati/Gulf | Mac | License |
|----------|-----|-------|--------------|-----|---------|
| 1 | `jais_asr` | [IFM/Jais-ASR](https://huggingface.co/IFM/Jais-ASR) | Explicit `AE`; Mixat CS ~17.8% WER | Heavy (~12 GB) | Apache-2.0 |
| 2 | `audar_asr_flash` | [Audar-ASR-V1-Flash](https://huggingface.co/audarai/Audar-ASR-V1-Flash) | Gulf/Emirati claimed; streaming/GGUF | Good | AudarAI Open |
| 3 | `qwen_asr_uae` | [qwen3-asr-arabic-uae](https://huggingface.co/vadimbelsky/qwen3-asr-arabic-uae) | Emirati FT (synthetic train) | Heavy MPS | Apache-2.0 |
| 4 | `whisper_baseline` | mlx / faster-whisper `large-v3-turbo` | Multilingual ZS + EN CS | Excellent | MIT |
| 5 | `whisper_arabic_ft` | [whisper-large-v3-arabic-ft-v3](https://huggingface.co/dev-ahmedhany/whisper-large-v3-arabic-ft-v3) | Casablanca UAE lift | Good | Apache-2.0 |
| 6 | `whisper_gulf_small` | [otozz gulf Whisper](https://huggingface.co/otozz/whisper-small-dialect_gulf) | MASC Gulf FT | Excellent | Whisper MIT |
| 7 | `wav2vec_emirati` | [eabayed wav2vec2 Emirati](https://huggingface.co/eabayed/wav2vec2emiratidialict_1) | Legacy Emirati CTC | Good | GPL-3.0 |

## Datasets

| Dataset | Hours | License | Role |
|---------|-------|---------|------|
| Mixat | ~15h Emirati–EN CS | CC BY-NC-SA 4.0 | Best modern Emirati eval/train (research only) |
| Alsanaa | ~4.5h traditional Emirati | Unclear / cite source | Heritage ASR paper |
| emirates-dialect-speech | ~4h segmented | `other` | TTS/ASR chunks |
| EmiratiDialictShows | 467 clips | AFL-3.0 | Small benchmark |
| Casablanca UAE | ~6h val/test | CC BY-NC-ND 4.0 | Gold Emirati eval |
| SADA / Khaliji | hundreds of hours Saudi TV | CC BY-NC-SA 4.0 | Gulf warm-start (NC) |
| Ramsa | 41h Emirati | Restricted DUA | Best future resource; not public download |
| UAE_100K | large | Unclear | Audit before commercial use |

**Commercial gate:** do not train shippable weights on Mixat / SADA / Casablanca. Prefer proprietary Emirati + Common Voice AR + Apache/MIT bases (Whisper, Qwen3-ASR, Jais, Audar Flash).

## Streaming (orchestrator turn-taking)

Mac-first pattern:

```
Mic 16 kHz → Silero VAD / end-of-utterance
  → [optional partial UI]
  → final: mlx-whisper turbo or Audar Flash / Qwen3-ASR
```

True low-latency RNNT streaming (Nemotron, etc.) is a later Linux/NVIDIA (or MLX port) track. This repo ships **offline** `transcribe(wav)` first.

## Fine-tune notes

1. Whisper QLoRA on `large-v3-turbo` → merge → CTranslate2/faster-whisper for serving.
2. Qwen3-ASR official SFT: freeze audio encoder, FT decoder; UAE recipe matches `vadimbelsky/qwen3-asr-arabic-uae`.
3. Jais/Audar already cover much of the Emirati gap — validate before investing in your own FT.

## Skip as primary Emirati listener

- Falcon-ASR (no self-host weights)
- Closed APIs (Google, AssemblyAI, Whisper API)
- Vosk AR as free-conversation primary
- Unadapted MMS / Seamless for commercial (NC + dialect gaps)
- FunASR open streaming (Chinese/English focus)

## Sources

- Mixat (SIGUL 2024); Casablanca (EMNLP 2024); Alsanaa Emirati ASR (ICNLSP 2025); Ramsa (LREC 2026)
- otozz multi-dialect Whisper ([arXiv:2506.02627](https://arxiv.org/abs/2506.02627))
- Qwen3-ASR; IFM Jais-ASR blog; Audar-ASR cards
- Open Universal Arabic ASR Leaderboard ([arXiv:2412.13788](https://arxiv.org/abs/2412.13788))
