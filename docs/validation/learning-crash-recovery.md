# Learning transaction crash-recovery gate

Status: local **process-crash** validation, not a board power-cut result.
Tested with Python 3.12.13 / SQLite 3.53.2 on the development Mac. The
physical board's Python-linked SQLite version and storage behavior remain
unknown until delivery.

Duck Mind uses `journal_mode=WAL` and `synchronous=FULL`, with one writer and
an outer `BEGIN IMMEDIATE` transaction for each committed learning update and
its provenance event. In SQLite's [WAL design](https://www.sqlite.org/wal.html#how_wal_works),
the commit record is the visibility boundary; [transaction documentation](https://www.sqlite.org/lang_transaction.html)
describes rollback of an unfinished transaction. SQLite documents the extra
WAL sync at each commit under [`synchronous=FULL`](https://www.sqlite.org/pragma.html#pragma_synchronous).
These settings support the intended durability design but do not, by
themselves, prove the board's power supply or filesystem.

`tests/test_store_crash.py` runs a separate Python process and calls
`os._exit` after these actual SQLite operations, without closing the database:

| Cut point | Expected after reopen | Retry |
|---|---|---|
| learning-update `INSERT` | no update, no learning event | commits once |
| learning-event `INSERT` | no update, no learning event | commits once |
| outer `COMMIT` returns | one update and one learning event | idempotent/no second change |
| event commit, before Duck Self snapshot | event survives, snapshot lags | startup replay applies it once |

The learning cuts also check SQLite `quick_check`, the episode/outcome evidence,
the final bounded preference and the applied update ID. The event/snapshot cut
checks health and replay across two restarts. Schema-creation crash points have
separate tests in the same file. The tests do **not** pull power,
simulate an OS/kernel crash, exercise eMMC/SD flush semantics, or prove that
a backup made by copying only the main `.sqlite3` file is complete. Use the
SQLite backup API or a stopped service for board backups.

Before real-play approval, record `python3 -c 'import sqlite3; print(sqlite3.sqlite_version)'`
on the board and confirm the distribution's SQLite patch status. SQLite's
[WAL-reset advisory](https://www.sqlite.org/wal.html#the_wal_reset_bug)
identifies a rare multi-connection write/checkpoint bug fixed in 3.51.3 and
selected backports. Duck Mind's single-writer lock reduces exposure but is
not proof that an unknown board build is patched. Keep other tools from
writing or checkpointing the live DB, and hold board approval if its linked
SQLite is affected without a verified backport.
