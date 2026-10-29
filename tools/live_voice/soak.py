"""A long real session: how a Live connection behaves past the documented limits.

The documentation says a session without compression ends after 2 minutes with video (15 minutes
audio only) and a connection after about 10 minutes (with a GoAway first); compression and the
resumption handle are what keep a session going. This runs the duck's session for a long time on
the real service — the microphone's room noise, one picture a second, the status text every five
seconds — through the very :class:`~duckmind.live.session.LiveSession` the duck uses, and says what
happened: when the server said go-away, whether the session resumed, what it cost per minute.

    PYTHONPATH=src python tools/live_voice/soak.py [--seconds 660] [--no-video]

Needs the key (Keychain or ``$GEMINI_API_KEY``, read by the connection). Prints one JSON line per
notable event (redacted) and a summary; nothing is stored but the log at ``--log``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from duckmind import paths
from duckmind.live import scenario as S
from duckmind.live.genai_transport import GenaiTransport
from duckmind.live.keys import redact
from duckmind.live.meter import UsageCaps, UsageMeter
from duckmind.live.session import FRAME_S, LiveSession, SessionConfig
from duckmind.live.soul import ContextSnapshot

LOG_DEFAULT = paths.data_dir() / "live-voice" / "soak-log.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seconds", type=float, default=660.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--log", default=str(LOG_DEFAULT))
    args = parser.parse_args(argv)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file = S.load_scenario_file()
    run = S.ScenarioRun(file, "silence", "explorer", 7, require_bank=False)
    picture = (S.EYE_DIR / "eye_toy.jpg").read_bytes()
    clock = S.RealtimeClock()
    session = LiveSession(
        GenaiTransport(clock=clock),
        run.soul_text(),
        clock=clock,
        config=SessionConfig(),
        meter=UsageMeter(
            UsageCaps(tokens=400_000, audio_s=args.seconds + 60, frames=int(args.seconds) + 60)
        ),
    )
    mic = run.make_mic()
    seen = 0
    notes = 0
    started = time.monotonic()
    states: list[tuple[float, str]] = []
    step = 0

    def emit(**fields: Any) -> None:
        line = redact(
            json.dumps(
                {"at": datetime.now(UTC).isoformat(timespec="seconds"), **fields},
                ensure_ascii=False,
            )
        )
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    emit(soak="start", seconds=args.seconds, video=not args.no_video)
    while True:
        t = step * FRAME_S
        if t > args.seconds:
            break
        clock.wait_until(t)
        frame = mic.tick(t, [], wall=t)
        if step % 250 == 0:
            session.set_context(
                ContextSnapshot(behaviour="坐著看四周", notes=(f"第 {step // 50} 秒",))
            )
        session.feed_audio(frame)
        if not args.no_video and step % 50 == 0:
            session.feed_image(picture)
        session.pump()
        step += 1
        if not states or states[-1][1] != session.state:
            states.append((round(t, 1), session.state))
            emit(t=round(t, 1), state=session.state, connections=session.status()["connections"])
        while seen < len(session.log):
            entry = session.log[seen]
            seen += 1
            if entry["kind"] in (
                "go_away",
                "closed",
                "error",
                "opened",
                "tool_cancel",
                "interrupted",
                "heard",
                "said",
                "call",
                "fault",
            ):
                emit(entry={k: v for k, v in entry.items() if k != "wall"})
        if step % 3000 == 0:  # every minute
            status = session.status()
            emit(
                t=round(t, 1),
                minute=status["usage"]["this_minute"],
                context_tokens=status["usage"]["context_tokens"],
            )
    session.close()
    status = session.status()
    emit(
        soak="end",
        wall_s=round(time.monotonic() - started, 1),
        connections=status["connections"],
        resumes=status["resumes"],
        events=status["events"],
        usage=status["usage"]["totals"],
        by_modality=status["usage"]["by_modality"],
        last_error=status["last_error"],
        faults=status["faults"],
        notes=notes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
