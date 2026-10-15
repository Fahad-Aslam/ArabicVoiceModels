# Reference audio for cloning backends

`habibi_uae` and `audar_flash` need a short reference clip (≈5–15 seconds) plus its transcript.

## Expected files

| File | Used by |
|------|---------|
| `uae_ref.wav` + `uae_ref.txt` | Habibi UAE (preferred) |
| `audar_ref.wav` + `audar_ref.txt` | Audar Flash (preferred) |

If only `uae_ref.wav` is present, Audar will fall back to it.

## Emirati Habibi example transcript

The Habibi paper/assets UAE prompt transcript is:

```text
قمنا نشتريها بشكل متكرر أو لما نلقى ستايل يعجبنا وحياناً هذا الستايل ما نحبه.
```

You can also pass refs on the CLI:

```bash
arabic-voice synth -b habibi_uae -t "ترا الجو وايد حلو اليوم" \
  --ref-audio assets/refs/uae_ref.wav \
  --ref-text "قمنا نشتريها بشكل متكرر أو لما نلقى ستايل يعجبنا وحياناً هذا الستايل ما نحبه."
```

After `pip install habibi-tts`, Habibi may also find packaged `UAE.wav` assets automatically.
