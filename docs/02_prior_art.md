# 02 — Prior art

## A. History in the NAM repositories

Searched issues and PRs in `sdatkinson/neural-amp-modeler`,
`sdatkinson/NeuralAmpModelerCore`, `sdatkinson/NeuralAmpModelerPlugin` for
*loudness*, *normalize*, *volume*, *output level*, *calibration*.

### Trainer (`neural-amp-modeler`)

| ref | date | summary | relevance |
|---|---|---|---|
| [#134](https://github.com/sdatkinson/neural-amp-modeler/issues/134), [#136](https://github.com/sdatkinson/neural-amp-modeler/issues/136), [#135 (commit 2928d24)](https://github.com/sdatkinson/neural-amp-modeler/commit/2928d242a931d9e51e3331f4eac875599753c4dc) | Mar 2023 | "Models compute their output level": introduced `_loudness()` as `10·log10(mean(y²))` of the response to `loudness_input.wav`. | Origin of the current calculation. No design rationale for the stimulus recorded. |
| [#171 LUFS model output normalization](https://github.com/sdatkinson/neural-amp-modeler/issues/171) | 2023 | User requested LUFS normalisation: "certain guitar or bass tones having a lot of bass frequency content may unintentionally skew the RMS measurement". Maintainer: "If you want to modify the code in this repo to compute LUFS instead, I'll happily take the PR" → later closed: "the current non-weighted loudness calculation seems fine". | Exactly the bass-vs-guitar concern; never tested quantitatively. |
| [#588 Improvement to calculation of loudness metadata](https://github.com/sdatkinson/neural-amp-modeler/issues/588) | Aug 2025 – Jun 2026, **open** | rerdavies (TooB NAM author): a family of commercial vintage-Fender models get loudness values up to 4.5 dB off; the models "audibly choke" on the sub-60 Hz content / apparent DC offsets of the pink stimulus (confirmed by the model author as realistic rectifier/transformer sag, ~1/13 s time constant). Proposed: remove < ~100 Hz from the stimulus, pre-warm the model before measuring, and derive a backward-compatible offset to preserve the −18 reference. Maintainer tried "low- and high-passed variations" **by ear**, "didn't find something that I liked", un-assigned; suggested (1) switch to LUFS, (2) feed the stimulus at a calibrated physical level. rerdavies later: "high frequency input signals cause problems with some amp models … band limiting produces slightly better results". | The most direct precedent. Both hypotheses (LF and HF content of the stimulus) are tested here. Also note the maintainer's observation that **the vast majority of normalised models land ≈ −9 dBFS** peak — consistent, so the problem is the tails. |
| [#668 Compensate metadata.loudness when `_ScaleOutputHook` scales head_scale](https://github.com/sdatkinson/neural-amp-modeler/pull/668) | 2026 | Keeps `loudness` correct under output-RMS normalisation. | Any new measure must remain shift-equivariant under output gain so this hook stays exact. |
| [#480 How to determine calibration levels](https://github.com/sdatkinson/neural-amp-modeler/issues/480) | 2024 | dBu ↔ dBFS calibration docs. | Background for the "calibrated stimulus level" idea. |

### Core (`NeuralAmpModelerCore`)

| ref | summary |
|---|---|
| [#16](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/16), [#17](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/17), [PR #18](https://github.com/sdatkinson/NeuralAmpModelerCore/pull/18) | Store loudness in the model; "Output loudness normalization" (originally inside the DSP core with a `TARGET_DSP_LOUDNESS`). |
| [#58](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/58), [#82](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/82)/[PR #85](https://github.com/sdatkinson/NeuralAmpModelerCore/pull/85), [#86](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/86)/[PR #87](https://github.com/sdatkinson/NeuralAmpModelerCore/pull/87), [PR #80](https://github.com/sdatkinson/NeuralAmpModelerCore/pull/80) | Normalisation removed from the core; the core only *carries* `loudness`, the host applies it. (Good: the host can change target/clamp without touching the core.) |
| [#158 Source of the "loudness" metadata item](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/158) | Points at `base.py` L122; maintainer: "rather successful at achieving its goal". |
| [#159 Guidance on Calibration](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/159) (comment [3239702643](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/159#issuecomment-3239702643)) | Clarifies `input_level_dbu` = dBu (RMS) that corresponds to **0 dBFS peak**. Contains the first report of the vintage-Fender "+5 dB too loud when normalised" family and the suggested fix (HPF ~100 Hz, pre-warm). |
| [#195](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/195), [#262](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/262) | `std::optional` for loudness; A2 container metadata must be at the top level and is taken from the highest-quality submodel. |

### Plugin (`NeuralAmpModelerPlugin`)

| ref | summary |
|---|---|
| [#153 auto-level the main output loudness](https://github.com/sdatkinson/NeuralAmpModelerPlugin/issues/153) → [PR #175 Normalize output loudness](https://github.com/sdatkinson/NeuralAmpModelerPlugin/pull/175) | Origin of the −18 dB target and the "Normalize" switch. |
| [#183](https://github.com/sdatkinson/NeuralAmpModelerPlugin/issues/183)/[PR #192](https://github.com/sdatkinson/NeuralAmpModelerPlugin/pull/192), [#376 Set normalization on by default](https://github.com/sdatkinson/NeuralAmpModelerPlugin/issues/376) | Normalisation is the default UX; "a literal copy of the reamp as-captured is a niche desire". |
| [#529 output calibration](https://github.com/sdatkinson/NeuralAmpModelerPlugin/issues/529), [#294](https://github.com/sdatkinson/NeuralAmpModelerPlugin/issues/294), [#542](https://github.com/sdatkinson/NeuralAmpModelerPlugin/issues/542) | Raw / Normalized / Calibrated output modes and the dBu math. "You'll keep the output mode on normalized because that's what makes the models all the same (digital) loudness." |
| [#313 IR Normalization on NAM plugin IR loader](https://github.com/sdatkinson/NeuralAmpModelerPlugin/issues/313) (open), trainer [#288](https://github.com/sdatkinson/neural-amp-modeler/issues/288) | IR level is "an accident of capture/export"; nothing upstream yet. The TONE3000 engines solved it two different ways (unity in-band vs unit-energy −18). |

**Takeaways from the history**

1. The stimulus was never designed against instrument signals; it was a
   convenient broadband probe. Every subsequent discussion treated the −18
   target as the thing to preserve and the stimulus as the thing to tweak.
2. The only prior attempt at improving it (#588) was evaluated by ear on a
   handful of models and abandoned. There has been no corpus-scale,
   perception-model-based evaluation; that gap is what this study fills.
3. Two independent reporters converged on the same two defects: too much
   energy below ~100 Hz and above ~5 kHz in the probe.
4. The maintainer's own suggestions — LUFS instead of flat RMS, and a
   stimulus level anchored to calibration — are both in our candidate set.

## B. Loudness measurement standards

* **ITU-R BS.1770-5 (11/2023)** — K-weighting (stage 1: +4 dB high shelf
  modelling the head, stage 2: RLB high-pass ≈ 38 Hz), mean square, 400 ms
  blocks with 75 % overlap, absolute gate −70 LKFS, relative gate −10 LU.
  A 0 dBFS 997 Hz sine reads −3.01 LKFS. [ITU](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I%21%21PDF-E.pdf)
* **EBU Tech 3341** (EBU Mode: Momentary 400 ms, Short-term 3 s, Integrated)
  and **Tech 3342** (Loudness Range). **R128 s1** for short-form content: use
  Programme Loudness plus **Maximum Short-term Loudness**; LRA is not
  meaningful for short items. [EBU 3341](https://tech.ebu.ch/files/live/sites/tech/files/shared/tech/tech3341v4_0.pdf), [R128 s1](https://tech.ebu.ch/files/live/sites/tech/files/shared/r/r128s1.pdf)
* **ISO 532-1:2017 (Zwicker)** — time-varying loudness from 1/3-octave band
  levels through excitation patterns, specific loudness, temporal
  integration; **N5** (loudness exceeded 5 % of the time) is the standard
  single value for time-varying sounds; the note in §6.4 points out that the
  energetic mean of loudness level correlates at least as well for music.
  Implemented in `mosqito` (`loudness_zwtv`). [ISO](https://www.iso.org/standard/63077.html)
* **ISO 532-2 (Moore–Glasberg)** stationary only; **ISO 532-3** time-varying
  extension; **ECMA-418-2 (Sottek hearing model)** best agreement with
  listening data for time-varying/impulsive sounds but uses its own sone_HMS
  scale. For a corpus study, Zwicker ISO 532-1 is the pragmatic reference
  because it is standardised for arbitrary time-varying signals and freely
  implemented. [JASA ISO 532-3](https://pubs.aip.org/asa/jasa/article/145/3_Supplement/1897/636629/The-loudness-model-used-in-ISO532-3-Development), [ECMA-418-2](https://ecma-international.org/wp-content/uploads/ECMA-418-2_4th_edition_june_2025.pdf)
* **A-weighting (IEC 61672)** — designed for low-level pure tones (40 phon);
  Aarts (AES preprint, 1990) found A-weighting "can result in misleading
  conclusions" for loudspeaker level matching while Zwicker loudness matched
  listener adjustments best. [Aarts](https://www.sps.tue.nl/rmaarts/RMA_papers/aar90pu.pdf)

## C. Perception of distorted / instrument signals

* Durbridge, Hill & Taylor, *The effects of distortion on the perception of
  loudness in live sound* (Reproduced Sound 2015): clipped/compressive
  distortion must be *played louder* (by LUFS) to be judged equally loud as
  the clean version; inter-listener spread grows with distortion severity.
  Implication: any energy-based measure (RMS, LUFS) will systematically rate
  high-gain models slightly "too loud" relative to clean ones, and no
  stimulus tweak fixes that — only a perceptual model, or a deliberate
  gain-dependent correction, can. [paper](https://adamjhill.com/AJHILL/wp-content/uploads/2020/02/Durbridge-Hill-Taylor-RS2015.pdf)
* Herbst, *Influence of distortion on guitar chord structures* (Huddersfield):
  "the strongest positive correlation was measured between loudness and
  spectral centroid, suggesting that sharper sounds increase the perceived
  loudness"; amplifiers differed mainly in loudness and sharpness.
  [paper](https://eprints.hud.ac.uk/id/eprint/33379/1/Herbst%20Influence%20of%20distortion%20on%20guitar%20chord%20structures.pdf)
* Schlittenlacher et al., *Instantaneous and overall loudness of music*:
  N5 and the energetic mean of loudness level both correlate highly with
  overall judgements of rock music (which has little dynamic range); N5
  overestimates for wide-dynamic-range material. [paper](https://discovery.ucl.ac.uk/id/eprint/10181604/1/2014%20Instantaneous%20and%20overall%20loudness%20of%20music.pdf)

## D. Cabinets and what an IR removes

* Celestion Vintage 30: official response 70 Hz–5 kHz; measured IRs "roll off
  rapidly above 5–6 kHz", presence peaks ≈ 2.3 and 3.5 kHz. 4×12 closed-back
  adds 80–300 Hz weight. [midi-audio-expert](https://midi-audio-expert.com/celestion-vintage-30-ir-frequency-response/)
* Ampeg SVT-410HE: 60 Hz–18 kHz ±3 dB (horn), −10 dB at 43 Hz; typical
  bass IRs peak ≈ 100 Hz with a tweeter resonance 2–3 kHz. [Ampeg manual](https://ampeg.com/data/6/0a00050909c6661d97e9057aa/application/pdf/)
* Measured on the study's own IRs (relative to 200 Hz–2 kHz mean):
  celestion −7 dB @50 Hz, +0 @5 k, −25 @8 k, −41 @12 k; mesa −17 @50, −26 @8 k;
  eminence −33 @50, −26 @8 k; ampeg (bass) +15 @50, +17 @100, +2 @5 k, −36 @8 k.

## E. How other ecosystems do it

Helix / Quad Cortex / Fractal / ToneX have **no automatic per-capture level
metadata**; the community practice is a reference preset and manual trimming
of a post-chain output block, with LUFS meters recommended as a starting
point and "finish by ear at performance volume". Quad Cortex's capture
guidance normalises *at capture time* (−12 dBFS for clean, −6 dBFS for
distorted amps) which is a human-in-the-loop version of the same idea.
NAM's stamped `loudness` is therefore already ahead of the field; the
question is only how well the stamp predicts perception.
[faderandknob](https://faderandknob.com/blog/level-match-modeler-presets), [QC wiki](https://quadcortex.wiki/Capture_Techniques), [Line 6 forum](https://line6.com/support/topic/63753-balancing-volume-levels/)
