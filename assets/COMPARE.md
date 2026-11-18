# Stock vs filtered vs smooth vs warm vs human

All files are 24 kHz, voice **F2**, seed 0, 7.95 s, from the same text
(16-bit for raw stock, 24-bit for processed):

> The quick brown fox jumps over the lazy dog. She sells seashells by the
> seashore, and the old lighthouse keeper counted the ships each evening.

| file | what it does | floor | p10-p90 | HF flat 4-8k / 8-12k | inter-harm | STOI |
|---|---|---:|---:|---:|---:|---:|
| `tiny_voca_stock.wav` | raw model output | −71.8 | 18.7 | 0.700 / 0.654 | −25.6 | 1.000 |
| `tiny_voca_filtered.wav` | tone + de-hiss (`clean`) | −85.0 | 18.3 | 0.683 / 0.654 | −26.2 | 0.999 |
| `tiny_voca_smooth.wav` | `clean` + HF contrast 1.5 | −81.3 | 17.4 | 0.532 / 0.496 | −30.0 | 0.999 |
| `tiny_voca_warm.wav` | `smooth` + 3.5 dB 180–700 Hz body | −84.5 | 20.2 | 0.532 / 0.493 | −29.9 | 0.996 |
| `tiny_voca_human.wav` | `warm` + room tone/reflections + breaths + expander | −58.9 | 21.2 | 0.551 / 0.507 | −29.6 | 0.979 |
| human reference (LJSpeech) | different speaker/recording | −56.7 | 22.0 | 0.50 / 0.36 | −29.6 | — |

The **sizzle** is HF flatness: the Qwen target itself is ~0.72/0.67, the model
~0.70/0.65, human speech ~0.50/0.36. All processed files are loudness-matched
to the stock RMS (−18.5 dBFS) for a fair A/B. `smooth`/`warm` use a per-frame HF
spectral-contrast boost (γ=1.5) that preserves the HF energy but makes the
harmonics/formants stand out above the broadband noise. `--contrast 2.0`
goes further; `strong` uses that.

Regenerate:

```bash
python -m tiny_voca.infer --checkpoint weights/tiny-voca.pt --voice F2 \
  --text "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore, and the old lighthouse keeper counted the ships each evening." \
  --output stock.wav
python -m tiny_voca.postfilter --input stock.wav --output filtered.wav --voice F2 --preset clean
python -m tiny_voca.postfilter --input stock.wav --output smooth.wav   --voice F2 --preset smooth
python -m tiny_voca.postfilter --input stock.wav --output warm.wav     --voice F2 --preset warm
python -m tiny_voca.humanize  --input stock.wav --output human.wav      --voice F2 --preset human
```

`humanize` writes 24-bit by default; `--subtype PCM_16` adds TPDF dither.


## Experimental branch: HF harmonic/formant resynthesis

On the `hf-harmonic-formant-synth` branch, `tiny-voca-hfsynth` rebuilds the
7-12 kHz magnitude from an F0-guided harmonic model. It leaves 4-8 kHz alone
and targets the top-octave sizzle.

| file | HF flat 4-8k | HF flat 8-12k | STOI vs warm |
|---|---:|---:|---:|
| `tiny_voca_warm.wav` (F2) | 0.532 | 0.493 | 1.000 |
| `tiny_voca_hfsynth.wav` (F2) | 0.534 | **0.378** | 1.000 |
| `tiny_voca_hfsynth_strong.wav` (F2) | 0.531 | **0.326** | 1.000 |
| `tiny_voca_warm_m2.wav` (M2) | 0.574 | 0.495 | 1.000 |
| `tiny_voca_hfsynth_m2.wav` (M2) | 0.577 | **0.382** | 1.000 |
| human reference | 0.50 | 0.36 | — |

See `postproc/HF_SYNTH.md` for the method and caveats.
