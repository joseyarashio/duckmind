# Screenshots, video, motion clips and README banners

Pictures of the duckmind simulator page, a screen recording of the page, short clips of
the duck's motions, and the banners at the top of the READMEs.
Everything here was made with the simulator (MuJoCo physics under upstream `robotd --sim`,
or, for the banners' duck, the page's own 3D drawing of the robot model).
Nothing here was filmed on a real robot; duckmind has not run on real hardware.

## Licence: CC BY-NC-SA 4.0 (not Apache-2.0)

These files show the **Microduck 3D model by Pollen Robotics**, whose meshes are licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). So the files in
`docs/media/` are licensed **CC BY-NC-SA 4.0** too, not Apache-2.0 like the code:

- Attribution: "duckmind simulator renders; Microduck 3D model © Pollen Robotics, CC BY-NC-SA 4.0".
- Non-commercial use only.
- Share changed versions under the same licence.

duckmind is not affiliated with Pollen Robotics. See [THIRD_PARTY.md](../../THIRD_PARTY.md).

## README banners (`hero/`)

The banner at the top of each README, one per README language: `hero.<lang>.png`
(2400×1260) and `hero.<lang>.webp` (the same picture, smaller, for the READMEs), with
`lang` one of `en`, `zh-Hant`, `zh-Hans`, `es`, `fr`, `ja`, `ko`. They are drawn from the
sources in `hero/src/`:

| File | What it is |
|---|---|
| `generate.py` | Writes `hero.<lang>.html` from `template.html` and the text for each language, and takes each picture with headless Chrome (then `cwebp` for the WebP) |
| `template.html`, `hero.<lang>.html` | The banner page; the `hero.<lang>.html` files are generated, do not edit them |
| `duck.png` | The duck: the official Microduck 3D model drawn the way the simulator page's 3D view draws it, on a transparent background |
| `render_duck.py`, `render-duck.html` | How `duck.png` is made: the simulator's robot description (the MJCF bodies, meshes and colours from the `microduck_rl` checkout that setup downloads), the page's vendored three.js and the page's lights; the `STAND` keyframe from the official `scene.xml` with the legs apart as in a step and the head turned to the camera, a 3/4 view |
| `duck-eye.svg` | The simulator page's own mark (the duck with the eye, from the page header) |
| `screenshot.webp` | A screenshot of the simulator page (desktop, page in 繁體中文): the duck 蛋黃 napping, sitting down; the line under the commands says it did not do the flip it was asked for, and the banner's speech bubble says just that |
| `fonts/` | Noto Sans JP, KR and SC cut down to the characters the banners use (SIL Open Font License 1.1, `fonts/OFL-NotoSansCJK.txt`) |

Every font is a local file: Anton, DM Sans, JetBrains Mono and Noto Sans TC are the page's
own vendored fonts in `src/duckmind/simweb/static/fonts/` (SIL Open Font License 1.1, the
`OFL-*.txt` files there), the rest are in `hero/src/fonts/`. Nothing is loaded from a font
service, so the pictures come out the same each time. The banners say what is true today:
the simulator works, moving to a real robot is planned and not validated, it runs on your
own computer with the cloud voice off unless you turn it on.

How to draw them again (from the repository root):

```bash
# the duck (needs the microduck_rl checkout from setup)
DUCKMIND_MICRODUCK_MJCF=~/.duckmind/upstream/microduck_rl/src/mjlab_microduck/robot/microduck \
  uv run python docs/media/hero/src/render_duck.py
# after changing Japanese, Korean or Simplified Chinese text: the font subsets
uv run --with fonttools --with brotli python docs/media/hero/src/generate.py --subset-fonts --html-only
# the banners
python3 docs/media/hero/src/generate.py
```

The banners are under the same CC BY-NC-SA 4.0 licence as everything in `docs/media/`,
because they show Pollen's Microduck 3D model; the fonts in them stay under the OFL.

## Video (`video/`)

A real screen recording of the simulator page as you see it in the browser (desktop,
1440×900, page in 繁體中文), with nothing hidden and nothing staged: what the duck does is
what its brain decided at that moment.

| File | What it is |
|---|---|
| `simulator.mp4` | The full cut: H.264, 1440×900, 30 frames a second, 59 s |
| `simulator.webp` | A shorter cut for the READMEs (GitHub does not play MP4 files from the repo inline): animated WebP, 960×600, 12 frames a second, 36 s |

What `simulator.mp4` shows:

| Time | What happens |
|---|---|
| 0:00–0:15 | The duck lives on its own: walks about looking at things, sits down by itself, nibbles at something, looks around. The 「現在在做」 card and the brain tab on the right change as it decides. |
| 0:15–0:21 | 「叫名字」 (call its name): it turns round to look. |
| 0:21–0:30 | Command 「坐下」 (sit): it sits down; then 「站起來」 (stand up): it stands. |
| 0:30–0:41 | Training tab: 「鞠躬」 (bow), a trick it has not learned yet. It bows; 「稱讚」 (praise) right after, and the bow's bar goes from 0% to 6%. |
| 0:41–0:47 | 「拍手」 (clap): it startles, then freezes and stares. |
| 0:47–0:52 | 「近看小鴨」: the camera moves in close. |
| 0:52–0:59 | 「放大」: the 3D view is enlarged. |

`simulator.webp` has the same parts, shorter, without the close-up step.

How it was made: a test server on this code with a throwaway life (seed 9202, growth day 60,
the `real_home` home, no nest, the ToF sensor on), the duck's clock set to 10:00 with the
page's 「上午」 button. The page ran in Chrome for Testing; its frames were recorded through
the Chrome DevTools Protocol's screencast, and the page's own buttons were clicked by a
Puppeteer script. Everything the duck did in reply was its own. The video is cut together
from three recordings made one after another in the same session, in a different order:
the clap is from the first one (there the clap scared the duck, and it then refused to sit),
the bow from the third (an untaught trick is done only now and then: in the third recording
it bowed on the 1st and the 7th of 7 asks; the video shows the 7th, the one that was praised). Nothing else was
changed: no hidden page parts, no posed body, no speed-up. Encoded with ffmpeg (H.264,
CRF 22, faststart) and `img2webp` (lossy, quality 55).

This video is under the same CC BY-NC-SA 4.0 licence as everything in `docs/media/`.

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
