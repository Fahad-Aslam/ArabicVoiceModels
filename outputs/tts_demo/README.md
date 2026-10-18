# TTS demo (~20s Emirati)

**Prompt** (also in `prompt.txt`):

> أهلًا، شحالك؟ تقدر تساعدني أحجز موعد؟ ترا الدرب صوب دبي اليوم وايد زحمة، خلنا نطلع بدري. السعر ألف وخمس مية درهم، والكود متاح بعد باچر. هني المكتب، تفضل اقعد، تبي قهوة عربية ولا شاهي؟ السلام عليكم، معاك المساعد الآلي لبنك الإمارات. كيف أقدر أخدمك؟

## Listen order

| File | Duration | Notes |
|------|----------|-------|
| `piper_emirati.wav` | ~20.7s | MIT, ONNX/CPU — default bake-off pick |
| `habibi_uae.wav` | ~20.0s | **CC-BY-NC-SA-4.0** — eval only |
| `audar_flash` | — | Needs HF login for gated `neuphonic/neucodec` |

Regenerate: `python scripts/generate_demos.py`
