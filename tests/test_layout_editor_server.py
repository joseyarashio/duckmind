"""The layout editor's server side: saved layouts, the checks, apply (no physics here)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from duckmind.duckcore.api import Config
from duckmind.simweb import homeview, layouts
from duckmind.simweb.server import SimServer, SimState, make_handler
from duckmind.world import homelayout


@pytest.fixture()
def server(tmp_path: Path):
    homeview.set_home("real_home")
    state = SimState(Config(seed=7, name="測試"))
    state.layouts = layouts.LayoutStore(tmp_path / "layouts")
    state.layout_default = "real_home"
    httpd = SimServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state, tmp_path
    httpd.shutdown()
    httpd.server_close()
    homeview.set_home(None)


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())


def _post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_editor_gets_the_home_the_library_and_the_checks(server) -> None:
    base, _, _ = server
    state = _get(f"{base}/api/layout")
    assert state["ok"] and state["id"] == "real_home" and state["check"]["ok"]
    assert len(state["layout"]["placements"]) > 40
    assert "placements_expanded" not in state["layout"]  # the file's form, not the pieces
    assert {r["id"] for r in state["rooms"]} == {"living", "hall", "bedroom", "kitchen"}
    assert state["rooms"][0]["_i18n"]["zh"]["k"] == "world.homes.real_home.rooms.living.zh"
    lib = _get(f"{base}/api/layout/library")
    assert lib["ok"] and len(lib["items"]) >= 50 and lib["classes"]
    default = _get(f"{base}/api/layout?which=default")
    assert default["id"] == "real_home"


def test_check_save_load_delete_and_names(server) -> None:
    base, _, tmp = server
    layout = _get(f"{base}/api/layout")["layout"]
    code, out = _post(f"{base}/api/layout/check", {"layout": layout})
    assert code == 200 and out["ok"] and out["problems"] == []
    layout["placements"].append(
        {"id": "cu", "lib": "cushion", "x_mm": -2500, "y_mm": -1300, "yaw_mrad": 0}
    )
    code, out = _post(f"{base}/api/layout/save", {"name": "我的 flat_1", "layout": layout})
    assert code == 200 and out["ok"] and out["saved"][0]["name"] == "我的 flat_1"
    assert (tmp / "layouts" / "我的 flat_1.json").is_file()
    code, out = _post(f"{base}/api/layout/load", {"name": "我的 flat_1"})
    assert out["ok"] and any(p["id"] == "cu" for p in out["layout"]["placements"])
    for bad in ("../x", "a/b", "", "x" * 41, "active", ".hidden"):
        code, out = _post(f"{base}/api/layout/save", {"name": bad, "layout": layout})
        assert out["ok"] is False and out["code"] == "bad_name", bad
    assert _post(f"{base}/api/layout/load", {"name": "nope"})[1]["code"] == "not_found"
    assert _post(f"{base}/api/layout/delete", {"name": "我的 flat_1"})[1]["saved"] == []
    # a layout with a problem is not saved
    layout["placements"].append({"id": "w", "lib": "wardrobe", "x_mm": -900, "y_mm": -1000})
    code, out = _post(f"{base}/api/layout/save", {"name": "bad", "layout": layout})
    assert out["ok"] is False and out["code"] == "invalid" and out["problems"]
    assert not (tmp / "layouts" / "bad.json").exists()
    assert (
        _post(f"{base}/api/layout/save", {"name": "x", "layout": "not an object"})[1]["code"]
        == "bad_layout"
    )


def test_apply_writes_the_layout_for_the_physics_and_remembers_it(server) -> None:
    base, state, tmp = server
    layout = _get(f"{base}/api/layout")["layout"]
    layout["placements"].append({"id": "cu", "lib": "cushion", "x_mm": -2500, "y_mm": -1300})
    layout["id"], layout["zh"] = "custom_test", "試試看"
    code, out = _post(f"{base}/api/layout/apply", {"layout": layout})
    assert code == 200 and out["ok"]
    path = Path(out["home"])
    assert path.parent == tmp / "layouts" / "applied" and path.is_file()
    assert homelayout.load(str(path)).thing("cu").lib == "cushion"  # the physics loads it by path
    assert homeview.asked_home() == str(path)
    assert _get(f"{base}/api/layout")["id"] == "custom_test"
    assert state.layouts.remembered() == str(path)
    # an invalid layout is refused and nothing changes
    layout["placements"].append({"id": "w", "lib": "wardrobe", "x_mm": -900, "y_mm": -1000})
    out = _post(f"{base}/api/layout/apply", {"layout": layout})[1]
    assert out["ok"] is False and out["code"] == "invalid"
    assert homeview.asked_home() == str(path)
    # back to the default
    out = _post(f"{base}/api/layout/reset", {})[1]
    assert out["ok"] and homeview.asked_home() == "real_home" and state.layouts.remembered() is None


def test_the_store_refuses_what_it_should(tmp_path: Path) -> None:
    store = layouts.LayoutStore(None)
    with pytest.raises(layouts.LayoutProblem):
        store.save("a", {})
    store = layouts.LayoutStore(tmp_path)
    with pytest.raises(layouts.LayoutProblem):
        store.save("a", {"junk": "x" * layouts.MAX_BYTES})
    with pytest.raises(layouts.LayoutProblem):
        layouts.validate_upload({"placements": [{}] * (layouts.MAX_PLACEMENTS + 1)})
    assert store.remembered() is None
    store.remember(str(tmp_path / "gone.json"))
    assert store.remembered() is None  # a file that is gone is not remembered
