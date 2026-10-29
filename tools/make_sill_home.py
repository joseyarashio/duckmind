"""A test room with two sills for hop_over (bug 7, 2026-10-09): ``default_home`` without the nest, the duck on the mat,
and on the free west half a 2 cm sill (y = +1000 mm) and a 4 cm sill (y = -1000 mm), each 40 mm deep and 1.8 m wide.

    python tools/make_sill_home.py OUT.json
    DUCKMIND_LIVE_HOME=OUT.json python -m duckmind.simweb ...   (a layout file is a home)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src/duckmind/world/homes/default_home.json"
SILLS = (("sill_20", 20, 1000), ("sill_40", 40, -1000))  # id, height mm, y mm
X_MM = -1500
DEPTH_MM = 40
WIDTH_MM = 1800


def build() -> dict:
    d = json.loads(SRC.read_text())
    d["id"] = "sill_home"
    d["zh"] = "有兩道門檻（2 公分、4 公分）的家：預設的家去掉充電窩，給小鴨跨門檻的測試用"
    d["nest"] = None
    d["spawn"] = "mat"
    d["variants"] = {}
    for name, height, y in SILLS:
        d["elements"].append(
            {
                "id": name,
                "kind": "threshold",
                "cls": "furniture",
                "zh": f"門檻 {height} mm",
                "shape": "box",
                "pos_mm": [X_MM, y, 0],
                "size_mm": [WIDTH_MM, DEPTH_MM, height],
                "colour": "#a88a66",
            }
        )
    return d


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(build(), ensure_ascii=False, indent=1))
