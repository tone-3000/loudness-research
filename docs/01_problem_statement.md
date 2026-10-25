# 01 — Problem statement and current implementation

## What `metadata.loudness` is today

The trainer stamps a single number into every exported `.nam` file:

```python
# nam/models/base.py (BaseNet)
    def _metadata_loudness(self, gain: float = 1.0, db: bool = True) -> float:
        """
        How loud is this model when given a standardized input?
        In dB

        :param gain: Multiplies input signal
        """
        x = self._metadata_loudness_x().to(self.device)
        y = self._at_nominal_settings(gain * x)
        loudness = _torch.sqrt(_torch.mean(_torch.square(y)))
        if db:
            loudness = 20.0 * _torch.log10(loudness)
        return loudness.item()
```

i.e. **the flat, unweighted RMS level (dBFS) of the model's output when fed
`nam/models/_resources/loudness_input.wav`**. For A2 (`SlimmableContainer`)
files the container's value is copied from the highest-quality submodel
(`PackedWaveNet._sync_container_metadata_to_highest_quality_submodel`), and
`_ScaleOutputHook._adjust_metadata_loudness` shifts it by `20·log10(scale)`
when output normalisation rescales `head_scale` (#668).

The stimulus file (added in #135, March 2023, unchanged since) measured in
this study:

| property | value |
|---|---|
| duration / rate / format | 1.0 s, 48 kHz, mono, 24-bit |
| RMS | −14.97 dBFS |
| peak | −2.95 dBFS (crest 12.0 dB) |
| spectrum | pink (PSD slope −10.5 dB/decade 50 Hz–10 kHz), Gaussian (excess kurtosis −0.05) |
| **DC offset** | **−0.0397 (−28 dBFS)** |
| energy below 80 Hz | ≈ 18 % of total (−7.5 dB) |
| energy below 40 Hz | ≈ 12 % |
| energy above 5 kHz | ≈ 17 % |
| stationarity | 100 ms segment RMS within −16.0 … −14.2 dBFS |

For comparison, the 34 guitar and bass demo DIs shipped with the TONE3000 web player ([`neural-amp-modeler-wasm`](https://github.com/tone-3000/neural-amp-modeler-wasm), `ui/public/inputs`; the 18 used in this study are in `data/audio/inputs`):

| | guitar DIs | bass DIs |
|---|---|---|
| RMS / peak | ≈ −28 / −8…−13 dBFS | ≈ −28 / −8…−13 dBFS |
| crest factor | 15–24 dB | 15–28 dB |
| spectral centroid | 200–1000 Hz | 190–390 Hz |
| energy below 80 Hz | 0.03 %–6 % (−35…−12 dB) | 10–25 % (−10…−6 dB) |
| energy above 5 kHz | 0.0001 %–0.1 % (−59…−30 dB) | ≈ −35…−49 dB |

So the stimulus drives every model (a) ~13 dB hotter in RMS and ~5–9 dB hotter
in peak than real playing, (b) with 2–3 orders of magnitude more energy above
5 kHz than any instrument produces, (c) with far more sub-80 Hz energy than a
guitar, and (d) with a DC offset no instrument produces. None of those matter
for a linear, flat device; all of them matter for a nonlinear, band-shaping
one, which is what every amp, pedal and cab is.

## How the engines consume it

All three TONE3000 engines, and the upstream NAM plugin, do the same thing:

```
gain_dB = clamp(TARGET − metadata.loudness, −12, +12),   TARGET = −18 dB
```

| engine | location | notes |
|---|---|---|
| TONE3000 plugin | `Processor.cpp` | last NAM block always normalises; mid-chain blocks use calibrated hand-off when `output_level_dbu` is present, otherwise normalise; user-adjustable target (`targetLoudness`, default −18) |
| TONE3000 pedal | `t3k_pedal.cpp` `NormalizeDb()` | same constants, fixed −18 |
| TONE3000 web player | `neural-amp-modeler-wasm/wasm/nam-engine.cpp` | same constants |
| upstream NAM plugin | `NeuralAmpModelerPlugin/NeuralAmpModeler.cpp` | "Normalized" output mode, −18, no clamp |

IR blocks are handled differently per engine, which interacts with the
question of amp-only models:

* pedal and web player: IR kernel scaled to **unity in-band gain** (mean power over
  64 log-spaced points, 100 Hz–5 kHz, = 0 dB) — a cab "leaves the loudness of
  the normalised NAM block essentially unchanged".
* plugin: IR **unit-energy** normalised then a fixed −18 dB cab pad (lands
  ≈ 11 dB below unity; the ±24 dB block knobs absorb it).

## The complaint

Users report that after normalisation some models are still noticeably louder
or quieter than others. The reports cluster around: amp-only ("DI / no cab")
captures vs. full-rig captures; bass vs. guitar material; small vintage amps
that "choke" on sub-bass (issue #588); and high-gain vs. clean models.

## Goal of this study

Find **one** stimulus + measurement recipe, used identically for every model
regardless of gear type, that minimises the spread of *perceived* listening
level across models after the engines apply `TARGET − loudness`, when real
guitar **and** real bass are played through them, at whatever level the
player happens to hit the model — and quantify the engine-side conventions
(target, clamp, IR gain) that sit on top of it. The stimulus should stay a
short noise like today's, and the result should be an in-place improvement
of `metadata.loudness`, not a new key.
