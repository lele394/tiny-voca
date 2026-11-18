# Stock vs filtered vs warm vs human

All files are 24 kHz, voice **F2**, seed 0, 7.95 s, generated from the same
text (16-bit for the raw stock, 24-bit for the processed files):

> The quick brown fox jumps over the lazy dog. She sells seashells by the
> seashore, and the old lighthouse keeper counted the ships each evening.

| file | what it does | quiet floor | active p10-p90 | STOI vs stock |
|---|---|---:|---:|---:|
| `tiny_voca_stock.wav` | raw model output, peak-limited | −71.8 dBFS | 18.7 dB | 1.000 |
| `tiny_voca_filtered.wav` | tone + de-hiss (`clean`) | −85.5 dBFS | 18.3 dB | 0.999 |
| `tiny_voca_warm.wav` | `warm`: +3.5 dB 180–700 Hz body, smoother de-hiss | −83.2 dBFS | 21.3 dB | 0.997 |
| `tiny_voca_human.wav` | `human`: warm + room tone/reflections + breaths + slow expander | −57.9 dBFS | 22.3 dB | 0.980 |

The human reference (LJSpeech, different speaker/recording) sits around
−57 dBFS quiet floor and 22 dB active p10-p90, so `tiny_voca_human.wav` is the
closest of the four to that balance. The trade is a deliberate, quiet room
floor instead of digital silence.

Regenerate:

```bash
python -m tiny_voca.infer --checkpoint weights/tiny-voca.pt --voice F2 \
  --text "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore, and the old lighthouse keeper counted the ships each evening." \
  --output stock.wav
python -m tiny_voca.postfilter --input stock.wav --output filtered.wav --voice F2 --preset clean
python -m tiny_voca.postfilter --input stock.wav --output warm.wav --voice F2 --preset warm
python -m tiny_voca.humanize  --input stock.wav --output human.wav --voice F2 --preset human
```

`humanize` writes 24-bit PCM by default; use `--subtype PCM_16` for a dithered
16-bit file, or `--subtype FLOAT` for float samples.
