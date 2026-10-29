# Security policy

## Reporting a problem

Please do not open a public issue for a security problem. Report it privately through
GitHub's "Report a vulnerability" button on the repository's Security tab
(github.com/joseyarashio/duckmind/security). Include what you found, how to
reproduce it, and what it could let someone do.

You will get an answer as soon as the maintainer can; this is a one-person hobby
project, so please allow a few weeks for a fix. Please give that time before you
publish anything.

## What counts

- Anything that lets someone else control the duck, its simulator or the machine it
  runs on: the simulator web server, the `duckmindd` socket, the release bundle
  checks, the policy file checks.
- Anything that leaks an API key, a recording, or what the family said.
- A way around the safety layers in [SAFETY.md](SAFETY.md).
- A download in `python -m duckmind.setup_cmd` that is not checked (upstream checkouts must be at
  their pinned commits; policy files must match their pinned sha256).

## What to know

- duckmind has **not** been validated on a real robot. See [SAFETY.md](SAFETY.md).
- The simulator server is meant for your own computer. It binds to `127.0.0.1` by
  default and has no login. Do not expose it to the internet or an untrusted network.
- API keys belong in your system's key store or environment, never in the repository.
  See [PRIVACY.md](PRIVACY.md) for what cloud features send.
- Only the latest commit on the default branch gets fixes.
