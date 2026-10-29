"""Growth G4: fast-forward a throwaway test duck (docs/design/GROWTH-G4-2026-10-11.md).

To see growth without waiting weeks, a *test duck* is hatched in a headless world (the neural duck's
brain in a synthetic room, no physics), lives N duck-days under a chosen care style — attentive,
neglectful, rough or praising — and the page shows what it became, in the same component as the
duck the server lives. It reuses the acceptance harness of the growth gates
(``tools/growth/regrow_harness.py`` on top of ``care_harness.py``): the experience bridge, Duck Core
at the calendar's pace, the body regrown every night while it sleeps.

Never on the live life:

* the test duck lives in a **temporary folder** of its own (made fresh, removed afterwards), never in
  a duck's folder;
* a folder that is, holds, or lies inside a real duck's record (the server's ``--life`` folder, any
  folder with a ``life.json`` or a Duck Core ``core.json``, anything under ``~/.duckmind``) is
  refused (:class:`Refused`) — by the manager, by the CLI, and again by the worker process;
* it runs in a **separate process** (``python -m duckmind.life.fastforward --worker``), niced, so the
  live body loop keeps its pace; **one job at a time** (a second start is :class:`Busy`);
* the page's button cannot name a folder at all.

CLI (also the tool the page's button uses):

    python -m duckmind.life.fastforward --days 30 --style attentive --style rough --out DIR
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from duckmind.i18n import Msg, msg

STYLES = ("attentive", "neglectful", "rough", "praising")
DEFAULT_DAYS = 30
MAX_DAYS = 60
DEFAULT_SEED = 4107  # 蛋黃's seed: what a duck of its nature would have become
MAX_SEED = 999_999
MAX_STYLES = 2
POLL_S = 0.5
KILL_AFTER_S = 3.0
# a duck-day of the headless world costs about 1.4 s while it is a hatchling that cannot walk and 4-5 s once it
# walks (the brain's own step is ~1.4 ms; docs/design/GROWTH-G4-2026-10-11.md has the measured speed)


class FastForwardError(ValueError):
    """A request that cannot be done, said in words (a :class:`~duckmind.i18n.Msg`)."""


class Busy(FastForwardError):
    """A fast-forward is already running (one at a time)."""


class Refused(FastForwardError):
    """A folder that belongs to a real duck."""


class Invalid(FastForwardError):
    """A request outside the limits."""


# ── folders a test duck may never touch ───────────────────────────────────────────────────────


def default_protected() -> list[Path]:
    """Every duck a person keeps by default: ``~/.duckmind`` (the lives, the old save, the eyes)."""
    return [Path.home() / ".duckmind"]


def _related(a: Path, b: Path) -> bool:
    """Is ``a`` the same as ``b``, inside it, or does it hold it?"""
    return a == b or b in a.parents or a in b.parents


def check_workdir(workdir: Path, protected: Iterable[Path] = ()) -> Path:
    """The folder a test duck may be given, resolved; :class:`Refused` for one that is, holds or
    lies in a real duck's. (An empty or new folder is fine.)"""
    path = Path(workdir).expanduser().resolve()
    for guarded in (*protected, *default_protected()):
        other = Path(guarded).expanduser().resolve()
        if _related(path, other):
            raise Refused(msg("web.growth.ff.refused_folder", folder=str(path)))
    # a folder that already looks like a duck's (or is inside one) is somebody's, whoever they are
    for folder in (path, *path.parents):
        if (folder / "life.json").exists() or (folder / "core.json").exists():
            raise Refused(msg("web.growth.ff.refused_record", folder=str(path)))
    if path.is_dir():  # (a folder that holds a duck, one level down: a folder of lives)
        for child in path.iterdir():
            if child.is_dir() and (
                (child / "life.json").exists() or (child / "core.json").exists()
            ):
                raise Refused(msg("web.growth.ff.refused_record", folder=str(path)))
    return path


def tools_dir() -> Path | None:
    """The acceptance harness (``tools/growth`` of the repository), None when this is not a checkout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "growth"
        if (candidate / "regrow_harness.py").is_file():
            return candidate
    return None


def check_request(days: Any, styles: Any, seed: Any) -> tuple[int, list[str], int]:
    """A request as validated numbers and words (:class:`Invalid` otherwise)."""
    if type(days) is not int or not 1 <= days <= MAX_DAYS:
        raise Invalid(msg("web.growth.ff.bad_days", max=MAX_DAYS))
    if isinstance(styles, str):
        styles = [styles]
    if not isinstance(styles, (list, tuple)) or not 1 <= len(styles) <= MAX_STYLES:
        raise Invalid(msg("web.growth.ff.bad_styles", max=MAX_STYLES))
    chosen: list[str] = []
    for style in styles:
        if style not in STYLES:
            raise Invalid(msg("web.growth.ff.bad_styles", max=MAX_STYLES))
        if style not in chosen:
            chosen.append(str(style))
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise Invalid(msg("web.growth.ff.bad_seed", max=MAX_SEED))
    return days, chosen, seed


# ── the worker: one test duck's life, in its own process ──────────────────────────────────────


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _regrow_status(keep: dict[str, Any]) -> dict[str, Any]:
    """The test duck's regrows, shaped as ``LifeHost._regrow_status`` shapes the live duck's."""
    from duckmind.life import regrow as rg

    regrower = keep["regrower"]
    reports = keep["reports"]
    return {
        "count": len(reports),
        "growth_day": int(keep["session"].core.s.gd),
        "applied_gd": regrower.applied.gd,
        "due": bool(regrower.due),
        "waiting_for": regrower.refused if regrower.due else None,
        "problem": None,
        "last": reports[-1] if reports else None,
        "history": reports[-8:],
        "genome": rg.genome_numbers(keep["brain"].genome),
    }


def view_of_test_duck(keep: dict[str, Any], style: str) -> dict[str, Any]:
    """The growth view (annotated with its ``_i18n`` words) of a test duck the harness kept."""
    from duckmind.life import growth_view as gv
    from duckmind.simweb import i18nwire

    session = keep["session"]
    core = session.core
    tally = gv.Tally()
    tally.update(session.inputs)
    view = gv.build(
        core,
        tally=tally,
        name="",
        live=False,
        branch_id=core.s.branch,
        pinned=False,
        nature_branch=None,
        learner=getattr(keep["brain"], "learn", None),
        regrow=_regrow_status(keep),
        voice=None,
        ignored=getattr(keep["exp"], "ignored", {}),
        delivered={"refused": 0},
    )
    view["style"] = style
    return i18nwire.growth_answer(view, names=[core.s.name])


def run_worker(style: str, days: int, seed: int, out: Path, protect: Sequence[Path]) -> int:
    """One test duck's life under ``style`` for ``days`` duck-days in ``out``: writes
    ``progress.json`` after each day and ``result.json`` at the end."""
    out = check_workdir(out, protect)
    out.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.nice(10)  # (a guest on the machine: the live body loop keeps its pace)
    harness = tools_dir()
    if harness is None:
        _write_json(out / "result.json", {"ok": False, "error": "no tools/growth here"})
        return 2
    sys.path.insert(0, str(harness))
    import regrow_harness  # type: ignore[import-not-found]

    started = time.time()

    def progress(day: int, total: int) -> None:
        _write_json(
            out / "progress.json",
            {"day": day, "days": total, "elapsed_s": round(time.time() - started, 1)},
        )

    progress(0, days)
    keep: dict[str, Any] = {}
    result = regrow_harness.run_life(
        seed,
        style,
        days=days,
        care_from=0,
        care_days=days,
        do_exam=False,
        progress=progress,
        keep=keep,
    )
    view = view_of_test_duck(keep, style)
    # the test duck's own Duck Core save, kept next to the result (a test record, not a duck's:
    # no life.json; it holds only the inputs the harness gave it)
    from duckmind.duckcore import api

    _write_json(out / "test-duck-save.json", api.save_json(keep["session"]))
    _write_json(
        out / "result.json",
        {
            "ok": True,
            "style": style,
            "days": days,
            "seed": seed,
            "wall_s": result["wall_s"],
            "regrows": result["regrows"],
            "missed_nights": result["missed_nights"],
            "events_given": result["events_given"],
            "view": view,
        },
    )
    return 0


# ── the manager: one job at a time, in the background ─────────────────────────────────────────


class _Run:
    def __init__(self, style: str, folder: Path, proc: subprocess.Popen[bytes]) -> None:
        self.style = style
        self.folder = folder
        self.proc = proc
        self.state = "running"  # running | done | failed | cancelled
        self.error: Msg | str | None = None
        self.result: dict[str, Any] | None = None
        self.day = 0
        self.wall_s = 0.0


class _Job:
    def __init__(self, days: int, styles: list[str], seed: int, workdir: Path, own: bool) -> None:
        self.id = f"{int(time.time() * 1000):x}"
        self.days = days
        self.styles = styles
        self.seed = seed
        self.workdir = workdir
        self.own = own
        self.runs: list[_Run] = []
        self.state = "running"
        self.started = time.time()
        self.finished: float | None = None
        self.cancel = False


class FastForward:
    """Starts, watches and cancels the background fast-forward; one job at a time."""

    def __init__(
        self,
        protected: Iterable[Path] = (),
        *,
        python: str | None = None,
        tmp_root: Path | None = None,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.protected = [Path(p) for p in protected]
        self.python = python or sys.executable
        self.tmp_root = tmp_root
        self._wall = wall
        self._lock = threading.Lock()
        self._job: _Job | None = None
        self._thread: threading.Thread | None = None

    # ── start / cancel ───────────────────────────────────────────────────────────────────

    def start(
        self,
        *,
        days: Any = DEFAULT_DAYS,
        styles: Any = ("attentive",),
        seed: Any = DEFAULT_SEED,
        workdir: Path | None = None,
    ) -> dict[str, Any]:
        """Begin a fast-forward. ``workdir``: a folder to keep the test ducks' files in (the CLI);
        None (the page): a fresh temporary one, removed afterwards. :class:`Refused` if it is a
        real duck's, :class:`Busy` if one is running, :class:`Invalid` for numbers out of range."""
        days, chosen, seed = check_request(days, styles, seed)
        if tools_dir() is None:
            raise Invalid(msg("web.growth.ff.no_harness"))
        own = workdir is None
        with self._lock:
            if self._job is not None and self._job.state == "running":
                raise Busy(msg("web.growth.ff.busy"))
            if own:
                root = self.tmp_root
                if root is not None:
                    root.mkdir(parents=True, exist_ok=True)
                folder = check_workdir(
                    Path(tempfile.mkdtemp(prefix="duckmind-ff-", dir=root)), self.protected
                )
            else:
                assert workdir is not None
                folder = check_workdir(workdir, self.protected)
                folder.mkdir(parents=True, exist_ok=True)
            if self._job is not None and self._job.own:
                shutil.rmtree(self._job.workdir, ignore_errors=True)
            job = _Job(days, chosen, seed, folder, own)
            try:
                for style in chosen:
                    job.runs.append(self._spawn(job, style))
            except Exception:
                for run in job.runs:
                    run.proc.kill()
                if own:
                    shutil.rmtree(folder, ignore_errors=True)
                raise
            self._job = job
            self._thread = threading.Thread(target=self._watch, args=(job,), daemon=True)
            self._thread.start()
        return self.status()

    def _spawn(self, job: _Job, style: str) -> _Run:
        folder = job.workdir / f"run-{style}"
        folder.mkdir(parents=True, exist_ok=True)
        command = [
            self.python,
            "-m",
            "duckmind.life.fastforward",
            "--worker",
            "--style",
            style,
            "--days",
            str(job.days),
            "--seed",
            str(job.seed),
            "--out",
            str(folder),
        ]
        for guarded in self.protected:
            command += ["--protect", str(guarded)]
        log = (folder / "worker.log").open("wb")
        try:
            proc = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
            )
        finally:
            log.close()
        return _Run(style, folder, proc)

    def cancel(self) -> bool:
        """Stop the running job (its worker processes, by their own pids). True if one was running."""
        with self._lock:
            job = self._job
            if job is None or job.state != "running":
                return False
            job.cancel = True
            for run in job.runs:
                if run.proc.poll() is None:
                    run.proc.terminate()
        return True

    def shutdown(self) -> None:
        """The server is stopping: no worker outlives it, no temporary folder stays."""
        self.cancel()
        thread = self._thread
        if thread is not None:
            thread.join(KILL_AFTER_S + 2)
        job = self._job
        if job is not None and job.own:
            shutil.rmtree(job.workdir, ignore_errors=True)

    # ── watching ─────────────────────────────────────────────────────────────────────────

    def _watch(self, job: _Job) -> None:
        asked_kill = None
        while True:
            alive = 0
            for run in job.runs:
                self._read_progress(run)
                if run.proc.poll() is None:
                    alive += 1
            if job.cancel and alive:
                asked_kill = asked_kill or time.time()
                if time.time() - asked_kill > KILL_AFTER_S:
                    for run in job.runs:
                        if run.proc.poll() is None:
                            run.proc.kill()
            if not alive:
                break
            time.sleep(POLL_S)
        with self._lock:
            for run in job.runs:
                self._finish(run, job.cancel)
            job.finished = self._wall()
            if job.cancel:
                job.state = "cancelled"
            elif all(r.state == "done" for r in job.runs):
                job.state = "done"
            else:
                job.state = "failed"
            if job.own:
                # the results are in memory now; the folder was only the worker's desk
                shutil.rmtree(job.workdir, ignore_errors=True)

    @staticmethod
    def _read_progress(run: _Run) -> None:
        try:
            data = json.loads((run.folder / "progress.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        run.day = int(data.get("day", run.day))
        run.wall_s = float(data.get("elapsed_s", run.wall_s))

    @staticmethod
    def _finish(run: _Run, cancelled: bool) -> None:
        code = run.proc.returncode
        if cancelled and code != 0:
            run.state = "cancelled"
            return
        try:
            result = json.loads((run.folder / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result = None
        if code == 0 and isinstance(result, dict) and result.get("ok"):
            run.state = "done"
            run.result = result
            run.day = int(result.get("days", run.day))
            run.wall_s = float(result.get("wall_s", run.wall_s))
            return
        run.state = "failed"
        try:
            tail = (run.folder / "worker.log").read_text(encoding="utf-8", errors="replace")
            run.error = tail.strip().splitlines()[-1][:300] if tail.strip() else f"exit {code}"
        except OSError:
            run.error = f"exit {code}"

    # ── what the page reads ──────────────────────────────────────────────────────────────

    def status(self, *, results: bool = False) -> dict[str, Any]:
        with self._lock:
            job = self._job
            if job is None:
                return {"state": "idle", "limits": self._limits()}
            end = job.finished if job.finished is not None else self._wall()
            runs = []
            for run in job.runs:
                if run.state == "running":
                    self._read_progress(run)
                pct = 100 if run.state == "done" else min(99, run.day * 100 // max(1, job.days))
                runs.append(
                    {
                        "style": run.style,
                        "state": run.state,
                        "day": run.day,
                        "days": job.days,
                        "pct": pct,
                        "wall_s": round(run.wall_s, 1),
                        "error": None if run.error is None else str(run.error),
                    }
                )
            out: dict[str, Any] = {
                "state": job.state,
                "id": job.id,
                "days": job.days,
                "styles": list(job.styles),
                "seed": job.seed,
                "elapsed_s": round(end - job.started, 1),
                "runs": runs,
                "limits": self._limits(),
            }
            done = [r for r in job.runs if r.state == "done" and r.result]
            if job.state == "done" and done:
                total_days = sum(job.days for _ in done)
                out["speed"] = {
                    "days": total_days,
                    "seconds": out["elapsed_s"],
                    "days_per_s": round(total_days / max(0.1, out["elapsed_s"]), 2),
                }
            if results and done:
                out["results"] = {
                    r.style: {
                        "view": r.result["view"],  # type: ignore[index]
                        "wall_s": r.wall_s,
                        "regrows": r.result["regrows"],  # type: ignore[index]
                        "missed_nights": r.result["missed_nights"],  # type: ignore[index]
                    }
                    for r in done
                }
            return out

    @staticmethod
    def _limits() -> dict[str, Any]:
        return {
            "styles": list(STYLES),
            "days_default": DEFAULT_DAYS,
            "days_max": MAX_DAYS,
            "seed_default": DEFAULT_SEED,
            "max_styles": MAX_STYLES,
        }


# ── the command line ──────────────────────────────────────────────────────────────────────────


def _table(results: dict[str, dict[str, Any]]) -> str:
    """The key numbers of the test ducks side by side, in plain text."""
    rows = [
        ("growth day", lambda v: v["identity"]["growth_day"]),
        ("stage", lambda v: v["identity"]["stage"]["id"]),
        ("branch", lambda v: v["identity"]["branch"]["id"]),
        ("bond to the family %", lambda v: v["bonds"]["family_pct"]),
        ("liked things", lambda v: sum(1 for p in v["learned"]["prefs"] if p["liked"])),
        ("places to avoid", lambda v: len(v["learned"]["avoid_spots"])),
        ("falls", lambda v: v["learned"]["falls"]),
        ("calls learned", lambda v: len(v["voice"]["calls"])),
        ("experiences", lambda v: v["experiences"]["total"]),
        ("regrows", lambda v: (v["regrow"] or {}).get("count")),
    ]
    styles = list(results)
    width = max(len(name) for name, _ in rows)
    lines = ["".ljust(width) + "  " + "  ".join(s.ljust(12) for s in styles)]
    for name, getter in rows:
        cells = [str(getter(results[s]["view"])).ljust(12) for s in styles]
        lines.append(name.ljust(width) + "  " + "  ".join(cells))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fast-forward a throwaway test duck in a headless world (never a real duck's folder)."
    )
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    ap.add_argument("--style", action="append", choices=STYLES, help="up to two, side by side")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, help="a folder for the test ducks' files (kept)")
    ap.add_argument(
        "--protect", action="append", type=Path, default=[], help="a folder never to use"
    )
    ap.add_argument("--worker", action="store_true", help="(internal) run one duck in this process")
    args = ap.parse_args(argv)
    styles = args.style or ["attentive"]
    if args.worker:
        if len(styles) != 1 or args.out is None:
            ap.error("--worker takes one --style and --out")
        try:
            return run_worker(styles[0], args.days, args.seed, args.out, args.protect)
        except Refused as error:
            print(str(error), file=sys.stderr)
            return 3
    manager = FastForward(args.protect)
    try:
        manager.start(days=args.days, styles=styles, seed=args.seed, workdir=args.out)
    except FastForwardError as error:
        print(str(error), file=sys.stderr)
        return 3
    started = time.time()
    try:
        while manager.status()["state"] == "running":
            status = manager.status()
            line = "  ".join(f"{r['style']} {r['day']}/{r['days']}" for r in status["runs"])
            print(f"\r{line}  ({time.time() - started:.0f} s)", end="", flush=True, file=sys.stderr)
            time.sleep(1.0)
    except KeyboardInterrupt:
        manager.cancel()
        print("\ncancelled", file=sys.stderr)
        manager.shutdown()
        return 130
    final = manager.status(results=True)
    print(file=sys.stderr)
    manager.shutdown()
    if final["state"] != "done":
        for run in final["runs"]:
            if run["error"]:
                print(f"{run['style']}: {run['error']}", file=sys.stderr)
        return 1
    print(_table(final["results"]))
    speed = final.get("speed") or {}
    print(
        f"{speed.get('days')} duck-days in {final['elapsed_s']} s "
        f"({speed.get('days_per_s')} duck-days a second)"
    )
    if args.out is not None:
        _write_json(Path(args.out) / "fastforward.json", final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
