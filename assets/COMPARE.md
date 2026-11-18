# Stock vs filtered

Both files are 24 kHz, 16-bit PCM, voice **F2**, seed 0, 7.95 s, generated from
the same text:

> The quick brown fox jumps over the lazy dog. She sells seashells by the
> seashore, and the old lighthouse keeper counted the ships each evening.

| file | what it is |
|---|---|
| `tiny_voca_stock.wav` | raw model output, peak-limited to 0.99 |
| `tiny_voca_filtered.wav` | the same waveform through `--postfilter clean` |

The filtered version applies the voicing-adaptive tone correction, removes
about 13 dB of pause hiss, high-passes at 55 Hz and limits the peak. The
speech body is deliberately left close to the original; the largest audible
difference should be the quieter gaps.

Regenerate with:

```bash
python -m tiny_voca.infer --checkpoint weights/tiny-voca.pt --voice F2 \
  --text "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore, and the old lighthouse keeper counted the ships each evening." \
  --output stock.wav
python -m tiny_voca.postfilter --input stock.wav --output filtered.wav --voice F2 --preset clean
```
