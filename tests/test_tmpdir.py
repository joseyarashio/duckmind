"""The short socket/state folder: /private/tmp where it exists (macOS), else the temp folder."""

from __future__ import annotations

import os
import tempfile

import pytest

from duckmind import tmpdir


def test_mac_short_tmp_is_used_where_it_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os.path, "isdir", lambda p: p == tmpdir.MAC_SHORT_TMP)
    assert tmpdir.short_tmp() == "/private/tmp"


def test_linux_has_no_private_tmp_so_the_temp_folder_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    assert tmpdir.short_tmp() == tempfile.gettempdir()


def test_the_folder_is_short_enough_for_a_unix_socket() -> None:
    # sun_path is 104 bytes on macOS and 108 on Linux; the longest socket the sensors make
    # is dm-sensors-<port>/mediad/media.sock
    path = os.path.join(tmpdir.short_tmp(), "dm-sensors-65535", "mediad", "media.sock")
    assert len(os.fsencode(path)) < 104
