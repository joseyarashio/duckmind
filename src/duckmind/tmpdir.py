"""The short local folder for unix sockets and the live body's state.

A unix socket path must fit ``sun_path`` (104 bytes on macOS), and macOS's own temp folder
(``/var/folders/...``) is too long, so the code has always put its sockets under ``/private/tmp``.
Linux has no ``/private`` (and creating it needs root): there the system temp folder (``/tmp``,
short enough; ``$TMPDIR`` is honoured by :func:`tempfile.gettempdir`) is the same thing.
"""

from __future__ import annotations

import os
import tempfile

MAC_SHORT_TMP = "/private/tmp"


def short_tmp() -> str:
    """``/private/tmp`` where it exists (macOS: unchanged behaviour), else the temp folder."""
    return MAC_SHORT_TMP if os.path.isdir(MAC_SHORT_TMP) else tempfile.gettempdir()
