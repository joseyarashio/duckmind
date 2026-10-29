# Offline application release bundle

This is an application bundle for an **already provisioned official Microduck**,
not a flashable OS image. The author's robot has not arrived yet. A passing bundle is a
software candidate, not clearance for physical or unsupervised play.

## Evidence before packaging

Freeze Python sources, then rerun all five source-bound evidence scenarios:

1. [Official pet service simulation](../validation/run-pet-sim.md): run both
   `HOME` and official-default `SIT` starting poses with actual `robotd`, v5
   ONNX and MuJoCo; execution, paired gaze-body command replay, synthetic
   owner feedback and passive restart. Each pet report is bound to its sibling
   body trace by hash and interaction ID. Physical hardware and real owner
   preference are not tested.
2. [Official named-skill simulation](../validation/run-live-sim.md): fresh
   `kick_left` trace, offline wire/oracle validation and command-by-command
   replay through `EmbodiedSkillLayer` and `RecordedBodyPort`. No ball/contact
   goal, so task success is unknown even when execution passes.
3. Run `python -m duckmind.train_sim` at `--days 30` and `--days 365`, writing
   `docs/validation/evidence/30day-twins.json` and
   `docs/validation/evidence/365day-twins.json`: synthetic cognitive twins,
   bounded long-horizon preference updates and causal ablation. This does not
   train a gait policy.

Each report records `duckmind_source_sha256`. Packaging refuses reports from
different source bytes. This binds evidence to code, not just a version string
or a dirty Git HEAD. Upstream pins, model and simulator identities remain in
the reports; they are external prerequisites, not bundled runtime dependencies.
The release gate compares SHA-256 for each of the six configured ONNX files
across HOME-positive, SIT-positive, HOME-negative and named-skill runs; the
official manifest alone does not contain file hashes. This binds simulator
inputs, not the actual bytes resident in a running daemon or real robot. It
also requires matching robotd, ONNX Runtime, scene and body-server file hashes
across the scenarios, rather than relying on upstream Git commit IDs alone.
This is not a hash of the entire upstream checkout; record its clean state and
review any future source or dependency change separately.

## Build on the development host

```bash
uv run --locked --python 3.12 --extra dev python -m duckmind.release prepare \
  --repo . --release-id 0.1.0-rc29 --output dist/duckmind-0.1.0-rc29
```

Use a new release ID/output for every candidate. Existing outputs are never
overwritten. Network access may be required on the build host for locked dev
and build tools; board installation of the wheel itself uses `--no-index
--no-deps`. Python 3.12+, venv/pip, systemd and the official robot stack must
already be present on the board.

The command runs tests, Ruff checks/format checks and mypy, then builds a fresh
wheel with the pinned backend. A whole-input fingerprint covers `src`, `tests`,
`deploy`, `docs`, `.gitignore`, README, architecture, pyproject and uv.lock.
Changes during checks, build or copying invalidate the candidate. The builder
rechecks the copied wheel, copied evidence and input fingerprint before atomic
publication. Source/test bytecode caches are excluded. Do not edit inputs
while preparing a release.

The current offline profile permits only the pure-Python Duck Mind wheel with
no third-party runtime dependencies. Required wheel metadata, console entry
points and every RECORD hash are checked. Adding runtime dependencies or native
extensions requires extending the offline dependency closure and SBOM first;
the release builder intentionally refuses them today.

## Bundle contents and verification

- `dist/*.whl`: the exact source-matched application artifact.
- `src/`, `tests/`, `deploy/`, `docs/` and project settings: reproducible inputs.
- `checks.json`: commands/results bound to the whole-input and Python hashes.
- `sbom.cdx.json`: CycloneDX 1.6 application-wheel inventory. It deliberately
  excludes the host OS, Python, official `robotd`, policies and dev extras;
  **it is not a complete board SBOM**.
- `manifest.json`: Git HEAD/dirty state, wheel/input hashes, evidence inventory,
  file hashes, and `physical_hardware_gate=not_run`.
- `SHA256SUMS`: payload and manifest transport checksums.

```bash
uv run --locked --python 3.12 --extra dev python -m duckmind.release verify dist/duckmind-0.1.0-rc29
```

The verifier checks file inventory, hashes, checksums, wheel/source identity,
evidence gates, check identity and SBOM consistency. To verify copied bytes on
Linux without running the bundle's Python code, change into the bundle directory
and run `sha256sum -c SHA256SUMS`. Compare the manifest SHA-256 with the build
host's separately retained value before trusting it.

New bundles receive the nine-file semantic simulator/replay gate. For rollback
inspection, the verifier also accepts the exact, hard-coded manifest SHA-256
values of RC15–RC22 after all payload, wheel, check and SBOM integrity checks
pass. RC20–RC22 have nine evidence files but predate the policy-file/scene/ORT
identity gate; the pin does **not** claim they satisfy today's semantics. The
verifier does **not** execute their bundled Python. A newly built or modified
bundle cannot claim historical status merely by recomputing its checksums.
RC2–RC14 are not covered by this current-verifier compatibility
window; use a separately reviewed historical verification procedure before
considering any of them for recovery. Integrity verification is not approval
to deploy an old application or enable physical play.

Checksums detect corruption and inconsistency, **not authorship**. An attacker
who can replace every file can also replace checksums and test claims. This
candidate is unsigned; use an authenticated transfer channel and a separately
trusted manifest digest. Cryptographic signing and a full board-image inventory
remain release-hardening work.

After verification, follow the [board checklist](board.md) and invoke its
installer with the wheel path, the exact manifest wheel SHA-256, and release ID.
The installer archives the matching systemd unit alongside the application.
Keep the real-play approval drop-in absent until the supervised bench trial.
Do not install or invoke a real robot from the simulator/release workflow.

## Evidence limits

The package verifier does not simulate the physics again, execute Linux systemd,
establish physical safety, approve a head profile, or exercise a real owner
episode. Installer fault tests use isolated command doubles. Actual board boot,
service permissions, power-loss deployment recovery and physical stops require
separate gates in [ready-to-burn](../ready-to-burn.md).
