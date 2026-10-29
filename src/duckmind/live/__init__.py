"""duckmind.live — the duck's cloud brain: one Gemini Live session (live voice, LV1).

* :mod:`~duckmind.live.session` — the persistent session (audio + pictures up, tool calls back);
* :mod:`~duckmind.live.tools` — the five tools the model may call (it only proposes);
* :mod:`~duckmind.live.soul` — the system instruction and the periodic status text;
* :mod:`~duckmind.live.echo`, :mod:`~duckmind.live.meter` — the echo gate and the usage meter;
* :mod:`~duckmind.live.transport` — the line to the other end; :mod:`~duckmind.live.genai_transport`
  (the real one), :mod:`~duckmind.live.fake` (rule-based), :mod:`~duckmind.live.cassette`
  (record and replay);
* :mod:`~duckmind.live.keys` — the API key's whereabouts and keeping it out of everything.

Notes: ``docs/design/live-voice/LV1-live-session.md``.
"""
