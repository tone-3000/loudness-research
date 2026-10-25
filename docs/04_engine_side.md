# 04 — Engine side: how the number is used

The trainer stamps one scalar; the engines turn it into a gain. Several of
the "models normalise to different volumes" reports are about this half of
the system, and a better number will not fix them. This document lists what
each engine does today, what the data says about each design choice, and
what should change.

## 1. Current behaviour (as of this study)

| engine | gain | clamp | target | notes |
|---|---|---|---|---|
| upstream NAM plugin (`NeuralAmpModelerPlugin/NeuralAmpModeler.cpp`) | `-18 − loudness` | none | fixed −18 | gain added into the output knob's dB; "Normalize" toggle disabled when metadata absent |
| TONE3000 plugin (`Processor.cpp`) | `target − loudness` | ±12 dB | user param `targetLoudness` (−60…0, default −18) | last NAM block only; mid-chain NAM blocks use the calibrated hand-off (`output_level_dbu − inputCalibrationLevel`) when calibration is on. Sanity check: loudness must be finite and in (−100, 0], else unity |
| TONE3000 pedal (`t3k_pedal.cpp`) | `-18 − loudness` | ±12 dB | fixed −18 | same chain logic as the plugin; reads the value from the `.namb` header |
| TONE3000 web player (`nam-engine.cpp`) | `-18 − loudness` | ±12 dB | fixed −18 | smoothed gain |

IR gain conventions downstream of the model differ:

| engine | IR gain | effect on an amp-only model's listening level |
|---|---|---|
| plugin | unit-energy kernel, then fixed −18 dB "cab pad" | ≈ −11 dB vs the NAM-only block — absorbed by block / main out knobs |
| pedal, web player | unity *in-band* gain: mean power over 64 log-spaced points 100 Hz–5 kHz = 0 dB | ≈ 0 dB; a NAM→IR preset sits at the same level as NAM-only |

The upstream plugin does not touch IR gain beyond the user's knob.

## 2. What the stamped value actually measures today

`results/side_experiments.csv` (all 261 corpus models, PyTorch, shipped
stimulus):

* **Operating point.** The shipped file is −14.97 dBFS RMS with −2.95 dBFS
  peaks. Driving the pink stimulus from −27 to −15 dBFS RMS changes the
  output of the median model by only **1.0 dB** (median slope 0.09 dB/dB);
  from −15 to −3 dBFS the median slope is **0.03 dB/dB**. 64 % of models
  have slope < 0.3 between −27 and −15, 18 % are essentially linear
  (slope > 0.7). In other words, for most of the catalogue the current
  number is the model's *ceiling* — the level it clips at — not the level it
  produces at playing level, and the linear minority (clean amps, EQ-type
  outboard, light pedals, DI-level captures) is measured on a different
  footing from the rest. This is the mechanism behind "clean models come
  out quiet / high-gain models come out loud" (`03_method_and_results.md`
  §4.1 quantifies it against the perceptual gold).
* **DC offset.** Removing the −28 dBFS DC offset from the shipped file
  changes the stamped value by +0.02 dB on average (SD 0.11); 0.8 % of
  models move by more than 0.5 dB (max 1.0 dB). Not a cause, but trivially
  fixable — the proposed stimulus is DC-free.
* **Settling.** With a 6 s stationary stimulus, the first second is on
  average 0.15 dB louder than the last; 25 % of models differ by > 0.25 dB
  and 3 % by > 0.5 dB (sag / slow-recovery models). The proposed burst train
  trims 30 ms at each burst edge and averages seven bursts, which is enough
  (`03_method_and_results.md` §4.2, length test).

None of these mechanisms involves gear type, so the fix to the calculation
stays gear-agnostic, as the brief requires.

## 3. Target and clamp

* **Target.** The −18 dB target is a historical convention (the first
  stimulus happened to produce ≈ −18 dB from typical models, so unity gain
  was the common case). It is only a constant offset; the TONE3000 plugin
  already exposes it as a user parameter. The proposed calculation includes
  a +3 dB calibration constant chosen so that the catalogue **median is
  unchanged** (−19.2 dB on the corpus), so the −18 target gives the median
  model the same gain as today and no engine needs a new target or a
  metadata version switch.
* **Clamp.** ±12 dB protects against junk metadata and against the +30 dB a
  very quiet (DI-level or attenuated) capture would otherwise get. Today the
  stamped values in the corpus span −41.8 … −3.3 dB (SD 6.0) and 8.0 % of
  models hit the clamp and *do not get fully normalised* — those are the
  "unfixable" outliers in user reports. The proposed value spreads *wider*
  (SD 7.7, −51 … −6 dB) because it reflects real level differences at
  playing level rather than ceilings: with the −18 target, 13.4 % of models
  exceed ±12 dB (all on the boost side), 8.4 % exceed ±15, 4.2 % exceed ±18;
  the extreme is a genuinely quiet capture needing +33 dB
  (`results/migration.csv`). Recommendation: widen the clamp to **±18 dB**,
  or make it asymmetric (−18 … +15: boost is what exposes noise floors and
  training artefacts; cut is harmless). Keep the finite / range sanity
  check.
* **Metadata sanity window.** All three TONE3000 engines reject
  `loudness > 0`. Any definition that could legitimately exceed 0 dB (a phon
  scale, an un-offset K level) would be silently ignored by shipped engines;
  the proposed definition stays in dBFS-like units and is negative for every
  model in the corpus (max −5.7 dB) for this reason.
* **Transition.** An old stamp differs from the new one by 5.6 dB SD per
  model — exactly today's error — so a catalogue with mixed old and new
  stamps is no worse than today for old models and fixed for new ones.
  The trainer `version` field already in every `.nam` tells them apart if an
  engine wants to flag or re-stamp old files; the `.namb` header and the
  loaders need no new fields.

## 4. Interaction with input calibration

The calibration path and the normalisation path are deliberately
independent in the TONE3000 plugin: calibration scales the input into the
model's reference frame (`inputCalibrationLevel − input_level_dbu`),
normalisation scales the output. 84 % of catalogue models carry
`input_level_dbu` and 78 % carry `output_level_dbu`.

This matters for the loudness metadata in two ways:

1. **The stimulus level is defined in dBFS, not in the model's calibrated
   frame.** A calibrated-frame stimulus ("pink noise at 0 dBu through this
   model") would be more physically meaningful but would reintroduce a
   dependence on metadata that 16 % of models lack, and would make the
   metadata of two captures of the same device at different interface gains
   disagree. The study therefore keeps fixed dBFS levels and treats the
   calibrated case as a downstream concern.
2. **Where the engine applies calibration, the operating point changes.**
   With calibration on, a user with a hot interface drives the model harder
   than the stimulus did, and vice versa. How much the output moves per dB
   of input is a property of the device being modelled, and no scalar can
   remove it. This is exactly why the proposed stimulus sweeps the levels
   users play at and averages over them instead of picking one.

## 5. Amp-only models and IRs

Amp-only (`gear = amp`) models are intended to be followed by a cabinet IR,
which removes most energy below ~70–100 Hz and above ~5 kHz. Whether the
*metadata* should anticipate this depends on what the engine does with IR
gain:

* With the pedal / web player **unity in-band** convention, the IR is
  level-neutral in the band that dominates loudness. Measured on the
  corpus, a real cab IR lowers an amp-only model's perceived loudness by
  1.4 phon on average (SD 0.8) while leaving its LUFS within a fraction of
  a dB (`results/figures/amp_ir_effect_*.png`). The K-weighted calculation
  already places amp-only models level with other gear *once the cab is
  on*; heard raw they are ≈ 1.5 dB louder, which is correct (they are
  fizzier raw). No cab-anticipating band limit in the metadata is needed.
* With the plugin's **unit-energy + −18 dB pad** convention, every IR block
  sits ≈ 11 dB under unity; the gain is constant per IR, so it does not harm
  consistency *between models*, but it does make NAM-only and NAM→IR presets
  differ in level, and it is unrelated to the metadata.

Recommendation: align the plugin's IR gain with the pedal / web player
unity in-band convention (or make the cab pad equal to the in-band gain of
the kernel so the two conventions coincide). This is independent of the
metadata change.

## 6. Options for runtime levelling (beyond a better scalar)

A scalar fixes the between-model offset averaged over playing levels. What
remains (`03_method_and_results.md` §4.4) is each model's own
level-dependence seen from a level other than the player's. Options, in
increasing order of intrusiveness:

1. **Stamp the level curve**: the seven per-burst output levels of the
   proposed stimulus are already a complete input-level → output-level
   curve. Stored as an extra array (`loudness_curve`), an engine with a slow
   input-level follower can interpolate the expected output level and pick
   the gain for the level the player is actually at. Simulated on the
   corpus, this holds the residual at ≈ 1.5–2 dB SD at *every* player level
   instead of 1.6 dB at the centre rising to ≈ 4.5 dB at ±12 dB
   (`results/curve_*.csv`). Backward compatible: old engines ignore the
   array.
2. **Runtime LUFS matching** (the "auto-gain" of pedal multi-FX): measure
   the model output with a slow integrator and converge to the target.
   Removes the dependence on metadata entirely but changes dynamics across
   long time scales and is a product decision rather than a metadata fix.

The single-number change removes most of today's spread on its own and is
the recommendation; (1) is a possible later step.
