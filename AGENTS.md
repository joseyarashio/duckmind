# duckmind — rules for coding agents

These rules are for AI coding agents (and humans) working in this repository. The
longer version, with the reasons, is [CONTRIBUTING.md](CONTRIBUTING.md).

- **No machine paths.** Never write an absolute path of the machine you run on into
  code, tests or docs. Folders outside the repo come from `duckmind.paths`
  (`duckmind_home()`, `data_dir()`, `policy_dir()`, `v5_policy_dir()`,
  `upstream_root()`, `microduck_mjcf_dir()`), which honour the `DUCKMIND_*`
  environment variables and an optional `paths.toml`.
- **No personal data or secrets** in code, tests, fixtures or commit messages: no
  real names, e-mail addresses, IP addresses, host names, tokens or API keys. Test
  keys are built at run time.
- **Simulator page design standard:** every change to the simulator page
  (`src/duckmind/simweb/static/`), and every new page feature, follows
  [docs/simulator-style.md](docs/simulator-style.md): ink and TV static, cream
  dot-grid paper, square corners, thick ink borders, hard offset shadows, Anton /
  Noto Sans TC 900 display type, orange primary buttons with a magenta shadow, a
  yellow ticker. Use its tokens and component classes; do not add colours, rounded
  corners or soft shadows. Keep every element id, `data-*` hook and script-used class
  intact, and keep scenario controls inside a `[data-scenarios]` container. Before
  calling a page change done, check 1440×900 and 390×844: working controls, no
  console errors, no sideways scroll.
- **Six languages on the page:** 繁體中文 is the default and the source; English,
  日本語, Español, 简体中文 and 한국어 are translations in the same plain style.
  Every page change ships its visible text in all six languages in the same change:
  static text, buttons, tooltips, aria labels, canvas labels, and the text the server
  or the duck's brain sends to the page. Send text through the i18n catalogs and
  message keys with `python -m duckmind.i18n add` / `set`, then `format`, `manifest`
  and `check` ([docs/i18n.md](docs/i18n.md)); a new meaning gets a new key. Names
  the family gives (the duck's name, people's names) are data and are not translated.
  Check the layout in the longest language. The i18n tests must pass. Never send
  page text to an outside translation service.
- **Plain words.** Text for the people who use the duck is short and plain.
- **Licences.** Never commit the Microduck 3D meshes (`.stl`, CC BY-NC-SA 4.0),
  `.onnx` weights, datasets, or audio from a non-commercial model. Ports of upstream
  code keep their notice and get a line in [NOTICE](NOTICE); new third-party
  material gets a row in [THIRD_PARTY.md](THIRD_PARTY.md). Call Pollen Robotics'
  material "Pollen's" or "upstream", never "official", in text users read.
- **Safety.** Never weaken the safety layers in [SAFETY.md](SAFETY.md). Do not add
  code that calls `robot.enable`, sends servo commands, or talks to a real robot
  unless the task says so explicitly. Test servers bind to `127.0.0.1`.
- **Tests.** A behaviour change comes with a test that fails before and passes after.
  Test behaviour, not source text. Tests must not need the network or an API key.
- **Nothing leaves the machine** without the maintainer asking: no pushes, uploads,
  releases or calls to outside services from an agent session.
