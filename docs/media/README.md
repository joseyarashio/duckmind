# Screenshots and motion clips

Pictures of the duckmind simulator page and short clips of the duck's motions.
Everything here was made in the simulator (MuJoCo physics under upstream `robotd --sim`).
Nothing here was filmed on a real robot; duckmind has not run on real hardware.

## Licence: CC BY-NC-SA 4.0 (not Apache-2.0)

These files show the **Microduck 3D model by Pollen Robotics**, whose meshes are licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). So the files in
`docs/media/` are licensed **CC BY-NC-SA 4.0** too, not Apache-2.0 like the code:

- Attribution: "duckmind simulator renders; Microduck 3D model © Pollen Robotics, CC BY-NC-SA 4.0".
- Non-commercial use only.
- Share changed versions under the same licence.

duckmind is not affiliated with Pollen Robotics. See [THIRD_PARTY.md](../../THIRD_PARTY.md).

## Screenshots (`screens/`)

`<view>-<desktop|phone>.<lang>.webp`: desktop is 1440×900, phone is 390×844 (shown at 2×);
`lang` is `zh-Hant` (繁體中文) or `en` (English).

| View | What it shows |
|---|---|
| `main` | The simulator: the 3D home, the duck's clock, what it is doing now, the command strip |
| `train` | The training tab (the trick book) next to the command strip |
| `motions` | The motion demo tab (every trained motion and its test result) |
| `map` | The home map, opened large (desktop only: the phone layout does not show it) |
| `data` | The data page (the skills the duck is learning) |

## Clips (`clips/`)

Animated WebP, 432×324, 12 frames a second, a few seconds each. Recorded from the
simulator page's 3D view in a test server with a throwaway duck (growth day 49), with the
page's text hidden, so the clips have no words in them. The full list of motions is in
[docs/motions.md](../motions.md).

| Clip | Motion | How it was started |
|---|---|---|
| `walk.webp` | Walking (Pollen v5 walking policy) | the page's remote control |
| `quick-turn.webp` | Quick turn on the spot (`quick_turn_left`) | motion demo button |
| `bow.webp` | Bow (`bow`) | motion demo button |
| `lie-low.webp` | Lie low, then rise (`lie_low`) | motion demo button |
| `shake.webp` | Shake it off (`shake_2`) | motion demo button |
| `stretch.webp` | Standing stretch (`stretch_tall`) | motion demo button |
| `startle-hop-back.webp` | Startle, then two small steps back (`startle_3` + `hop_back`) | motion demo button |
| `freeze.webp` | Crouch and freeze (`freeze_3`) | motion demo button |
| `hop-over.webp` | Step over a sill (`hop_over_4`) | motion demo button |
| `spin.webp` | Spin (trick) | command "spin" |
| `flip.webp` | Flip, a forward roll (trick, Pollen's `roulade` policy) | command "flip" |
| `sit-stand.webp` | Sit down, stand up | commands "sit", "stand up" |
| `getup.webp` | Pushed over, gets back up (righting reflex) | the page's "push" |
| `play-dead.webp` | Play dead (emotion keyframes) | emotion button |
| `emotion-angry.webp` | Angry (emotion keyframes from Pollen's `microduck_emotions`) | emotion button |
