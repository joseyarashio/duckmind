# Contributing to duckmind

Thank you for helping. duckmind is a small project; these rules keep it working for
everyone.

By sending a contribution you agree that it is licensed under the Apache License 2.0,
like the rest of the project (see [LICENSE](LICENSE)).

duckmind is an unofficial project. It is not affiliated with, endorsed by, or
sponsored by Pollen Robotics. Please do not describe your change as official.

## Getting started

Requires Python 3.12+, `git` and [`uv`](https://docs.astral.sh/uv/).

```bash
uv run --python 3.12 python -m duckmind.setup_cmd      # the upstream checkouts and the v5 policies
uv run --python 3.12 --extra dev pytest -m "not heavy" # the fast tier: under three minutes
uv run --python 3.12 --extra dev ruff check src tests
uv run --python 3.12 --extra dev ruff format --check src tests
uv run --python 3.12 --extra dev mypy
```

Tests need no network and no API key; anything that would call a cloud service uses recorded
answers. [Section 5](#5-tests) says which tests run when.

## The rules

### 1. No machine paths, no personal data

- Never write an absolute path of your own machine (your home folder, your user
  name, a drive letter) into code, tests or docs. Use `duckmind.paths` (`duckmind_home()`,
  `data_dir()`, `policy_dir()`, `upstream_root()`, ...). Every folder has an
  environment variable and an optional `paths.toml` entry; see `src/duckmind/paths.py`.
- No real names, e-mail addresses, IP addresses, host names, tokens or keys in code,
  tests, fixtures or commit messages. Test keys are built at run time, never written
  out literally.
- Names a family gives (the duck's name, people's names) are data. The demo duck is
  called 蛋黃.

### 2. The simulator page speaks six languages

繁體中文 (`zh-Hant`) is the default and the source. English, 日本語, Español,
简体中文 and 한국어 are translations in the same plain style.

- Every page change, and every new page feature, ships its visible text in all six
  languages in the same change. This covers static text, buttons, tooltips, aria
  labels, canvas labels, and the text the server or the duck's brain sends to the
  page.
- Text goes through the i18n catalogs and message keys, never hard-coded. Use the
  tools: `python -m duckmind.i18n add` / `set`, then `format`, `manifest` and
  `check`. A new meaning gets a new key. The full guide is [docs/i18n.md](docs/i18n.md).
- Names the family gives are not translated.
- Check the layout in the longest language too (usually Español or 日本語).
- New words in 日本語, 简体中文 or 한국어 also need the page's font subsets rebuilt:
  `uv run --with fonttools --with brotli python -m duckmind.i18n.fonts subset`.
- Never send page text to an outside translation service. If you cannot write one
  of the languages, say so in your pull request and leave the key for a maintainer.
- The i18n tests must pass: `PYTHONPATH=src python -m pytest tests/test_i18n_*.py`.

### 3. The simulator page design standard

The page (`src/duckmind/simweb/static/`) has one look, inspired by the Microduck
product page. It is not an official design, and it copies no logos, images or text.
The full standard, with its tokens and component classes, is
[docs/simulator-style.md](docs/simulator-style.md) (in 繁體中文). In short:

- Ink and TV static, cream dot-grid paper, square corners, thick ink borders, hard
  offset shadows.
- Display type is Anton / Noto Sans TC 900. Primary buttons are orange with a magenta
  shadow. The ticker is yellow.
- Use the existing design tokens and component classes. Do not add colours, rounded
  corners or soft shadows.
- Keep every element id, `data-*` hook and script-used class intact. Keep scenario
  controls inside a `[data-scenarios]` container.
- Fonts and scripts are bundled in `static/fonts/` and `static/vendor/` with their
  licences. Do not load anything from a CDN.
- Before calling a page change done, check 1440×900 and 390×844: the controls work,
  the console has no errors, and nothing scrolls sideways.

### 4. Licences of third-party material

- Never commit the Microduck 3D meshes (`.stl`, CC BY-NC-SA 4.0), policy weights
  (`.onnx`), datasets, or audio made by a non-commercial model. `python -m duckmind.setup_cmd`
  fetches what is allowed to be fetched.
- A port of upstream code keeps the upstream licence notice in its file header and
  gets a line in [NOTICE](NOTICE). New third-party material gets a row in
  [THIRD_PARTY.md](THIRD_PARTY.md).
- Call Pollen Robotics' material "Pollen's" or "upstream", never "official", in
  anything a user reads.

### 5. Tests

- A behaviour change comes with a test that fails before the change and passes after
  it. Test behaviour, not the source text: a test that reads a source file and looks for a
  string in it proves nothing about what the code does. (Three kinds of test do read files on
  purpose: the i18n key coverage, the import-boundary guards that parse the code, and the checks
  of a frozen copy's bytes.)
- The safety layers (see [SAFETY.md](SAFETY.md)) must never be weakened to make a
  test pass.
- **The tiers.** While you work, run the fast tier: `pytest -m "not heavy"`. It is everything
  that needs no MuJoCo and is not measured slow, and takes under three minutes on a laptop
  (measured figures: `docs/validation/test-tiers.md`). A plain `pytest` is the whole suite
  (about half an hour with the `sim` extra); it runs before a merge, not after every edit.
  CI runs the fast tier on every push.
- **The markers.** `tests/tiers.py` says what each test is; `tests/conftest.py` puts the
  markers on while pytest collects, so a test file needs none:

  | Marker | What it means | Run just these |
  |---|---|---|
  | `heavy` | `render`, `physics` and the measured-slow tests | `pytest -m heavy` |
  | `physics` | steps or compiles MuJoCo; skips without the `sim` extra and the duck model | `pytest -m physics` |
  | `render` | renders the duck's camera (numpy, MuJoCo's offscreen context) | `pytest -m render` |
  | `node` | runs a page script in Node; skips without Node | `pytest -m node` |
  | `subprocess` | starts a child process | `pytest -m subprocess` |
  | `timing` | sleeps or polls on the wall clock: the first to fail on a busy machine | `pytest -m timing` |

  The physics and render tests need `uv run --python 3.12 --extra dev --extra sim pytest` and
  the robot model, found through `DUCKMIND_MICRODUCK_MJCF` or the checkout
  `python -m duckmind.setup_cmd` made. No CI job runs them (`.github/workflows/ci.yml` says
  what a runner for them would need).
- **A new test** is in the fast tier unless it says otherwise. A new file that needs MuJoCo
  says so itself: `pytestmark = [pytest.mark.physics]` (that makes it heavy too). A test that
  takes more than half a second goes into `tests/tiers.py`'s `SLOW`; the fast tier lists its
  own tests that took 2 s or more when it ends. Renaming a test that `tests/tiers.py` names
  stops the run until the name is changed there too.
- **Waiting.** Wait for the thing itself (an event, a condition, a callback, an answer on the
  socket), with a generous timeout, not for a length of time. A test that has to sleep is
  marked `timing`.

## Pull requests

- One topic per pull request, with a short description of what changed and why.
- Say which checks you ran. For page changes, say which screen sizes and languages
  you looked at.
- Security problems: please do not open a public issue; see [SECURITY.md](SECURITY.md).
- How a pull request lands: this repository is published from the maintainer's development
  line, so a merged change arrives in a sync commit instead of a GitHub "merge". Your commit is
  applied with your authorship kept, and the sync commit credits you as a co-author; the pull
  request is then closed with a note pointing to that commit.
