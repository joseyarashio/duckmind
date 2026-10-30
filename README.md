**English** | [繁體中文](README.zh-Hant.md) | [简体中文](README.zh-Hans.md) | [Español](README.es.md) | [Français](README.fr.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

# duckmind

duckmind gives a small robot duck a mind of its own: who it is, its moods, its
memories, the people it knows, what it wants to do, and what it learns. It is written
for robots compatible with Microduck, and comes with a simulator you can open in your
browser.

> **Unofficial project.** duckmind is not affiliated with, endorsed by, or sponsored
> by Pollen Robotics. Microduck is a name of Pollen Robotics. duckmind uses Pollen's
> open-source code, robot model and policies, which you download from their upstream
> sources during setup.

## See it

<p align="center"><a href="docs/media/video/simulator.mp4"><img src="docs/media/video/simulator.webp" alt="A screen recording of the duckmind simulator page: the duck wanders, sits and looks around on its own, then reacts to its name, sits and stands up when told, bows and is praised" width="880"></a></p>

<p align="center"><sub>A real screen recording of the simulator page (page in 繁體中文), not staged: first the duck lives on its own (wanders, sits down, nibbles, looks around, and the "doing now" card and its brain show what it decides); then it turns when its name is called, sits and stands up when told, does a bow it is still learning and gets praised (the training tab goes up to 6%), startles at a clap, and the camera moves in close and the view is enlarged. Cut together from three recordings of the same session; nothing else was changed. <a href="docs/media/video/simulator.mp4">Watch in full quality (MP4, 59 s)</a>.</sub></p>

| <img src="docs/media/clips/walk.webp" width="220" alt="Walk"> | <img src="docs/media/clips/bow.webp" width="220" alt="Bow"> | <img src="docs/media/clips/lie-low.webp" width="220" alt="Lie low"> |
|:---:|:---:|:---:|
| Walk | Bow | Lie low |
| <img src="docs/media/clips/spin.webp" width="220" alt="Spin"> | <img src="docs/media/clips/flip.webp" width="220" alt="Flip"> | <img src="docs/media/clips/getup.webp" width="220" alt="Gets back up"> |
| Spin | Flip | Gets back up |

All screenshots and clips: [docs/media](docs/media/README.md) (licensed CC BY-NC-SA 4.0, because they show Pollen's Microduck 3D model). Everything the duck can do, sorted by kind: [docs/motions.md](docs/motions.md).

## Status

- **Simulator:** works. The duck lives, grows, walks around a simulated home, and
  remembers people, on your computer.
- **Real hardware: not validated.** duckmind has not been tested on a real robot.
  The board release is a candidate only. Read [SAFETY.md](SAFETY.md) first.

## What it is

duckmind decides *what* the duck wants to do. It does not drive the motors. Pollen's
`robotd` and its trained policies move the body, and their safety rules stay in
charge:

```text
cognition -> behavior intent -> embodied skill -> robotd -> policy -> safety -> body
                                      ^                              |
                                      +-------- robot.state --------+
```

In the simulator, MuJoCo physics replaces the body below `robotd`, so the upper
layers are the same for the simulator and a real robot.

## What this project is about

In one sentence: **a little robot duck that slowly grows into a pet that truly belongs to your family.**

- It isn't a remote-control toy, and it isn't a robot acting out a script. Every move it makes, it decides for itself, after it "notices something and wants something".
- It only gets its own senses: eyes (the camera), ears, a distance sensor, and the feel of its own body. The simulator's "answer key" (where things really are, what the map looks like) stays hidden from it, just like for a real duck.
- **The neural duck is the living interface; growing up is the core.** The moment-to-moment reactions (getting startled, curious, sleepy, playful) come from the "neural duck" brain; who it turns into, day after day, comes from the growth system in "Duck Core". You can't have one without the other.

### Every duck is different: the random seed

- When it hatches, it gets a number called a "seed". The same seed grows the same genes; a different seed gives you a different duck.
- The seed sets its starting personality leanings (a bit clingy, a bit of an explorer, slow and dreamy…), how quick and how random its movements are, and a set of calls all its own.
- The seed is only the starting point. What happens to it afterwards, and how the family treats it, makes it "branch out" in different directions: give the same seed to two families, and after a while you'll have two different ducks.
- Same seed plus same experiences gives the same result, every time. That makes research and debugging possible, and it means you can trace "why did it turn out like this?".

### Raising it: it really grows up

- Its age is counted in "growth days", not power-on hours: a day only counts once it has been awake long enough.
- From Hatchling to Duckling, Juvenile and Adult, new things unlock as it grows. For example, the flip usually comes around the Juvenile stage (about 45 growth days); the exact day depends on how fast each duck grows.
- How you look after it shapes which kind of duck it becomes: Explorer, Dreamer, Snuggler or Songbird.
- It remembers: the people it knows, what it likes, where it once fell and should steer clear, what home looks like.
- Its calls evolve generation by generation (a call lineage): it starts out babbling, the calls the family praises with a "good duck!" are kept, and bit by bit they become its own song.
- You can teach it tricks: ask for one, praise it or give it a treat the moment it gets it right, and it gets better with practice. Leave a trick too long and it gets rusty, but it never completely forgets one it has learned.

### From simulator to real body: one day it can be "burned into" a real duck

- Right now everything runs in a physics simulator on your computer (MuJoCo plus Pollen's robotd simulation). Body physics, motor delay and battery are all modelled as closely to a real Microduck as we can.
- **A duck is a file**: its seed, genes, growth record, what it has learned, its calls. The design goal: a duck you raised in the simulator, once connected to a real Microduck, is the same duck in a real body — name, personality, memories and tricks all still there.
- The brain can stay on your home computer while the body becomes a real little duck; it's the same brain code, and the only differences are the body and the calibration of its senses.
- **To be honest:** the real duck hasn't been validated on real hardware yet, and "export/import a duck file" and "remote body" are planned features. See the Status section above and [SAFETY.md](SAFETY.md).

### What you can play with and do right now

- **Interact with it:** clap, call its name, pat its head, throw a ball, pick it up, turn off the lights, and watch how its reflexes and moods respond.
- **Give commands (in six languages):** Sit, Stand up, Come, Look at me, Stop, Follow me, Go home; tap the floor to send it there (experimental); or switch to Remote mode.
- **Teach it tricks:** Lie down, Bow, Shake, Spin, Flip; the "Training" tab shows how well it knows each one.
- **See inside its head:** what it wants to do right now, why, whether it's scared, how far it has grown.
- **Furnish the home:** place furniture, door thresholds and rugs yourself, and watch how it crosses thresholds, goes around walls and finds its way home.
- **Train new moves:** train new motion networks with the same tools (mjlab, on an A100 or your own machine) and add them to its move list.
- The full list of moves is in [docs/motions.md](docs/motions.md) (繁中: [docs/motions.zh-Hant.md](docs/motions.zh-Hant.md)).

## How it works

The duck's brain has two speeds: fast reflex arcs (startle, freeze, righting, stumble stop) that act at once, and a slower "cortex" where drives and feelings compete to choose what to do next, with a little seeded randomness. It senses only through its own sensors, remembers in layers (from this moment, to the places and people it learned, to an event log that lasts its whole life), and grows over days. The details — and an honest list of what does not work yet — are in [docs/how-it-works.md](docs/how-it-works.md).

## The simulator page speaks six languages

The simulator page is in 繁體中文 (the default), English, 日本語, Español, 简体中文
and 한국어. Switch the language on the page; everything the page, the server and the
duck's brain show follows it. Names the family gives (the duck's name, people's
names) are kept as they are.

## Quick start

You need Python 3.12 or newer, `git`, and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/joseyarashio/duckmind.git
cd duckmind
uv run --python 3.12 --extra sim python -m duckmind.setup_cmd
```

`python -m duckmind.setup_cmd` downloads what duckmind uses but does not ship:

- Pollen's `microduck` code (robotd and the duck's sounds) at a pinned commit,
- Pollen's `microduck_rl` code and 3D robot model at a pinned commit. The 3D meshes
  are licensed CC BY-NC-SA 4.0 (non-commercial). setup shows that licence and
  downloads them only if you type "yes",
- Pollen's v5 policy set from Hugging Face, checked file by file against pinned
  sha256 values.

Files go under `~/.duckmind` by default (`DUCKMIND_HOME` changes that). setup never
changes something that is already there.

Then start the simulator page. setup prints this command with your own path:

```bash
uv run --python 3.12 --extra sim python -m duckmind.simweb --port 8765 \
  --microduck-mjcf ~/.duckmind/upstream/microduck_rl/src/mjlab_microduck/robot/microduck
# open http://127.0.0.1:8765/
```

The server listens only on your own computer. More in
[docs/validation/run-duck-simulator.md](docs/validation/run-duck-simulator.md); live
physics with robotd is in [docs/validation/run-live-sim.md](docs/validation/run-live-sim.md).

The cloud voice (Google Gemini Live) is off by default. Read [PRIVACY.md](PRIVACY.md)
before you turn it on.

## Documentation

- [Architecture](architecture.md) and the [source review](docs/architecture-review.md)
- Contracts: [embodied skills](docs/contracts/embodied-skill.md),
  [System 1 / System 2](docs/contracts/system1-system2.md),
  [memory](docs/contracts/memory.md), [provenance](docs/contracts/provenance.md),
  [simulator](docs/contracts/simulator.md)
- [Upstream validation](docs/upstream-validation.md)
- Running: [the simulator page](docs/validation/run-duck-simulator.md),
  [live physics](docs/validation/run-live-sim.md),
  [the pet simulation](docs/validation/run-pet-sim.md)
- Checks: [replay and adversarial validation](docs/validation/replay-adversarial.md),
  [learning crash recovery](docs/validation/learning-crash-recovery.md),
  [robotd candidate comparison](docs/validation/robotd-candidate.md)
- Training: [off-board RL](docs/training/official-rl.md),
  [local CPU training](docs/training/local-cpu.md)
- The board: [deployment](docs/deploy/board.md), [offline release](docs/deploy/release.md),
  [ready-to-burn gates](docs/ready-to-burn.md)
- The page: [six languages](docs/i18n.md), [design standard](docs/simulator-style.md)

## Development

```bash
uv run --python 3.12 --extra dev pytest
uv run --python 3.12 --extra dev ruff check src tests
uv run --python 3.12 --extra dev ruff format --check src tests
uv run --python 3.12 --extra dev mypy
```

Extras: `sim` (MuJoCo physics), `train` (policy training with mjlab), `kws` (the
name keyword detector). Read [CONTRIBUTING.md](CONTRIBUTING.md) before you send a
change; coding agents also read [AGENTS.md](AGENTS.md). Security problems:
[SECURITY.md](SECURITY.md).

## Using, citing or modifying it: please credit the author

The author of this project is **joseyarashio**. If you use it, cite it, adapt it or build something on top of it, please credit the author and link the source, for example:

> duckmind by joseyarashio — https://github.com/joseyarashio/duckmind

- The code is Apache-2.0. Under section 4 of the licence, when you share the original or a modified version, you must include [LICENSE](LICENSE), keep the credits in [NOTICE](NOTICE), and say in each file you changed that you changed it.
- To cite it in an article or a paper, use [CITATION.cff](CITATION.cff) in this repository.
- The screenshots and clips ([docs/media](docs/media/README.md)) are licensed separately under CC BY-NC-SA 4.0: give credit, no commercial use, and share adaptations under the same licence, because they show Pollen's 3D model.

## Licence

duckmind is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE),
[NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md).

Some third-party material is not Apache-2.0 and is not in this repository. The most
important: the Microduck 3D meshes are CC BY-NC-SA 4.0 (non-commercial only); setup
downloads them from Pollen's repository only after you accept that licence. The test
voices in `tests/fixtures/voice/` were made with Qwen3-TTS (Apache-2.0). The
recordings in `tests/fixtures/live/` are Gemini model output, kept for tests only.
