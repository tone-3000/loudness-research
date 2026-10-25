# Test material

All files are 48 kHz mono WAV, taken from the demo material shipped with the
TONE3000 web player, [`tone-3000/neural-amp-modeler-wasm`](https://github.com/tone-3000/neural-amp-modeler-wasm)
(`ui/public/inputs`, `ui/public/irs`), and redistributed here under that
repository's terms.

## `inputs/` — DI recordings

Twelve guitar and six bass DIs. The study uses the loudest 6 s window of
each (8 s for the listening test); see `scripts/stimuli.py`.

| guitar | bass |
|---|---|
| Brit, Cream, Fear, Hotrod, Mayer, Metalcore, Smooth, Tomb, Jazz Trot, Slide Lead, Pluck, Power Thrash | Downtown, Drivin', Garden, Rollin', Smokin', Frogger |

"Power Thrash - Guitar" was contributed to the web player by
[@pippress](https://www.tone3000.com/pipppriss); the other DIs were recorded
by TONE3000.

## `irs/` — cabinet impulse responses

`celestion.wav` and `mesa.wav` (guitar cabinets), `ampeg.wav` (bass
cabinet). Applied to amp-only models with unity in-band gain
(100 Hz – 5 kHz), the convention used by the TONE3000 pedal and web player.
