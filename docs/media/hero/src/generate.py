"""Build the README hero banners: hero.<lang>.html here, hero.<lang>.png (2400×1260) and
hero.<lang>.webp one folder up.

Run from the repository root (Google Chrome and cwebp needed):

    python3 docs/media/hero/src/generate.py            # all seven languages
    python3 docs/media/hero/src/generate.py en ja      # some of them
    python3 docs/media/hero/src/generate.py --html-only

After changing any Japanese, Korean or Simplified Chinese text, first rebuild the fonts/ subsets
(needs fontTools; the full Noto fonts are read from ~/.cache/duckmind-fonts, the same cache
``python -m duckmind.i18n.fonts subset`` uses, and downloaded from google/fonts if missing):

    uv run --with fonttools --with brotli python docs/media/hero/src/generate.py --subset-fonts

The pages use only files in this repository: duck.png (render_duck.py), duck-eye.svg (the
simulator page's mark), screenshot.webp (a simulator screenshot), the page's vendored OFL fonts
in src/duckmind/simweb/static/fonts/ (Anton, DM Sans, JetBrains Mono, Noto Sans TC) and
fonts/ here (Noto Sans JP/KR/SC cut down to the banners' own characters). A small web server
on 127.0.0.1 (``--port``, 8782 by default) serves the repository while Chrome takes each picture.
"""

from __future__ import annotations

import argparse
import functools
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
REPO = HERE.parents[3]
FONTS = "../../../../src/duckmind/simweb/static/fonts/"
FONT_CACHE = Path.home() / ".cache" / "duckmind-fonts"
GOOGLE_FONTS_REPO = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
# family → (google/fonts folder, full variable font, the subset in fonts/)
CJK_FONTS = {
    "Noto Sans JP": ("notosansjp", "NotoSansJP[wght].ttf", "noto-sans-jp.woff2"),
    "Noto Sans KR": ("notosanskr", "NotoSansKR[wght].ttf", "noto-sans-kr.woff2"),
    "Noto Sans SC": ("notosanssc", "NotoSansSC[wght].ttf", "noto-sans-sc.woff2"),
}

MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
GITHUB = "GITHUB: JOSEYARASHIO/DUCKMIND"
DUCK_NAME = "蛋黃"  # the duck in screenshot.webp; a name the family gave, never translated

CONFIGS: dict[str, dict] = {
    "en": {
        "lang": "en",
        "cjk": [],
        "title": "duckmind — README hero (English)",
        "partner_left": "UNOFFICIAL • FOR MICRODUCK-COMPATIBLE ROBOTS • OPEN-SOURCE PET BRAIN",
        "partner_right": "PYTHON 3.12 + MUJOCO • CODE: APACHE-2.0",
        "kicker_tag": "A PET BRAIN FOR MICRODUCK-COMPATIBLE ROBOT DUCKS",
        "hero_subheadline": "MOODS • MEMORIES • GROWING UP",
        "hero_desc": "Gives a small robot duck a mind of its own. <strong>Not a scripted remote-control toy</strong>: it has moods and curiosity, remembers people, learns tricks and slowly grows up. Try it today in a 3D physics simulator in your browser.",
        "btn_primary": "OPEN THE 3D SIMULATOR →",
        "btn_secondary": "GITHUB REPO ★",
        "frame_title": "DUCKMIND SIMWEB // 3D SIMULATOR",
        "comic_badge": "DECIDES ON ITS OWN",
        "comic_text": "Told to flip, it didn't: it was sitting there, napping.",
        "robot_badge": "★ SIM-TO-REAL: PLANNED ★",
        "marquee_items": [
            "FAST REFLEXES + SLOW CORTEX (SYSTEM 1 & 2)",
            "SEEDED PERSONALITY",
            "4 GROWTH STAGES",
            "EVOLVING CALL LINEAGE",
            "RUNS LOCALLY • CLOUD OPTIONAL",
        ],
        "credit": "Duck: simulator render of Pollen Robotics' Microduck 3D model (CC BY-NC-SA 4.0).",
        "footnote": "* Unofficial project. duckmind is not affiliated with or endorsed by Pollen Robotics. Microduck is a trademark of Pollen Robotics.",
    },
    "zh-Hant": {
        "lang": "zh-Hant",
        "cjk": [],
        "title": "duckmind — README 主圖（繁體中文）",
        "partner_left": "非官方 • 為相容 MICRODUCK 的機器鴨而寫 • 開源寵物大腦",
        "partner_right": "PYTHON 3.12 + MUJOCO • 程式碼：APACHE-2.0",
        "kicker_tag": "為相容 MICRODUCK 的機器鴨打造的寵物大腦",
        "hero_subheadline": "有情緒 • 會記得 • 會長大",
        "hero_desc": "給機器小鴨一個自己的心智。<strong>不是照著劇本跑的遙控玩具</strong>：牠有情緒、會好奇、記得人、學得會把戲，也會慢慢長大。現在就能在瀏覽器裡的 3D 物理模擬器跟牠玩。",
        "btn_primary": "啟動 3D 模擬器 →",
        "btn_secondary": "GITHUB 開源專案 ★",
        "frame_title": "DUCKMIND SIMWEB // 3D 模擬器",
        "comic_badge": "自主決策",
        "comic_text": "叫牠翻跟斗，牠沒照做：正坐著打瞌睡。",
        "robot_badge": "★ SIM-TO-REAL · 規劃中 ★",
        "marquee_items": [
            "快反射 + 慢皮層 (SYSTEM 1 & 2)",
            "隨機種子個性",
            "4 階段成長",
            "叫聲譜系演化",
            "預設本機運行 • 雲端選用",
        ],
        "credit": "小鴨：模擬器算出的 Pollen Robotics Microduck 3D 模型（CC BY-NC-SA 4.0）。",
        "footnote": "* 非官方專案。duckmind 與 Pollen Robotics 無關，沒有得到 Pollen Robotics 的認可或贊助。Microduck 是 Pollen Robotics 的商標。",
    },
    "zh-Hans": {
        "lang": "zh-Hans",
        "cjk": ["Noto Sans SC"],
        "title": "duckmind — README 主图（简体中文）",
        "partner_left": "非官方 • 为兼容 MICRODUCK 的机器鸭而写 • 开源宠物大脑",
        "partner_right": "PYTHON 3.12 + MUJOCO • 代码：APACHE-2.0",
        "kicker_tag": "为兼容 MICRODUCK 的机器鸭打造的宠物大脑",
        "hero_subheadline": "有情绪 • 会记得 • 会长大",
        "hero_desc": "给机器小鸭一个自己的心智。<strong>不是照着剧本跑的遥控玩具</strong>：它有情绪、会好奇、记得人、学得会把戏，也会慢慢长大。现在就能在浏览器里的 3D 物理模拟器跟它玩。",
        "btn_primary": "启动 3D 模拟器 →",
        "btn_secondary": "GITHUB 开源项目 ★",
        "frame_title": "DUCKMIND SIMWEB // 3D 模拟器",
        "comic_badge": "自主决策",
        "comic_text": "叫它翻跟头，它没照做：正坐着打瞌睡。",
        "robot_badge": "★ SIM-TO-REAL · 规划中 ★",
        "marquee_items": [
            "快反射 + 慢皮层 (SYSTEM 1 & 2)",
            "随机种子个性",
            "4 阶段成长",
            "叫声谱系演化",
            "默认本机运行 • 云端可选",
        ],
        "credit": "小鸭：模拟器渲染的 Pollen Robotics Microduck 3D 模型（CC BY-NC-SA 4.0）。",
        "footnote": "* 非官方项目。duckmind 与 Pollen Robotics 无关，没有得到 Pollen Robotics 的认可或赞助。Microduck 是 Pollen Robotics 的商标。",
    },
    "es": {
        "lang": "es",
        "cjk": [],
        "title": "duckmind — imagen principal del README (Español)",
        "partner_left": "NO OFICIAL • PARA ROBOTS COMPATIBLES CON MICRODUCK • CEREBRO DE MASCOTA ABIERTO",
        "partner_right": "PYTHON 3.12 + MUJOCO • CÓDIGO: APACHE-2.0",
        "kicker_tag": "UN CEREBRO DE MASCOTA PARA PATOS ROBOT COMPATIBLES CON MICRODUCK",
        "hero_subheadline": "EMOCIONES • RECUERDOS • CRECE",
        "hero_desc": "Le da a un pequeño pato robot una mente propia. <strong>No es un juguete teledirigido con guion</strong>: tiene emociones y curiosidad, recuerda a las personas, aprende trucos y crece poco a poco. Pruébalo hoy en un simulador de física 3D en tu navegador.",
        "btn_primary": "ABRIR EL SIMULADOR 3D →",
        "btn_secondary": "REPOSITORIO GITHUB ★",
        "frame_title": "DUCKMIND SIMWEB // SIMULADOR 3D",
        "comic_badge": "DECIDE POR SÍ MISMO",
        "comic_text": "Le pedí una voltereta y no la hizo: estaba sentado, echando una siesta.",
        "robot_badge": "★ SIM-TO-REAL: PREVISTO ★",
        "marquee_items": [
            "REFLEJOS RÁPIDOS + CORTEZA LENTA (SYSTEM 1 & 2)",
            "PERSONALIDAD POR SEMILLA",
            "4 ETAPAS DE CRECIMIENTO",
            "LINAJE DE CANTOS QUE EVOLUCIONA",
            "FUNCIONA EN LOCAL • NUBE OPCIONAL",
        ],
        "credit": "Pato: render del simulador del modelo 3D Microduck de Pollen Robotics (CC BY-NC-SA 4.0).",
        "footnote": "* Proyecto no oficial. duckmind no está afiliado a Pollen Robotics ni cuenta con su aprobación o patrocinio. Microduck es una marca de Pollen Robotics.",
    },
    "fr": {
        "lang": "fr",
        "cjk": [],
        "title": "duckmind — image principale du README (Français)",
        "partner_left": "NON OFFICIEL • POUR ROBOTS COMPATIBLES MICRODUCK • CERVEAU DE COMPAGNON OPEN SOURCE",
        "partner_right": "PYTHON 3.12 + MUJOCO • CODE : APACHE-2.0",
        "kicker_tag": "UN CERVEAU DE COMPAGNON POUR CANARDS ROBOTS COMPATIBLES MICRODUCK",
        "hero_subheadline": "ÉMOTIONS • SOUVENIRS • IL GRANDIT",
        "hero_desc": "Donne à un petit canard robot un esprit à lui. <strong>Pas un jouet téléguidé qui suit un script</strong> : il a des humeurs et de la curiosité, se souvient des gens, apprend des tours et grandit peu à peu. Essayez-le dès aujourd'hui dans un simulateur physique 3D, dans votre navigateur.",
        "btn_primary": "OUVRIR LE SIMULATEUR 3D →",
        "btn_secondary": "PROJET GITHUB ★",
        "frame_title": "DUCKMIND SIMWEB // SIMULATEUR 3D",
        "comic_badge": "DÉCIDE TOUT SEUL",
        "comic_text": "On lui demande une roulade, il ne la fait pas : il somnole, assis.",
        "robot_badge": "★ SIM-TO-REAL : PRÉVU ★",
        "marquee_items": [
            "RÉFLEXES RAPIDES + CORTEX LENT (SYSTEM 1 & 2)",
            "PERSONNALITÉ PAR GRAINE",
            "4 STADES DE CROISSANCE",
            "LIGNÉE DE CRIS ÉVOLUTIVE",
            "TOURNE EN LOCAL • CLOUD EN OPTION",
        ],
        "credit": "Canard : rendu du simulateur, modèle 3D Microduck de Pollen Robotics (CC BY-NC-SA 4.0).",
        "footnote": "* Projet non officiel. duckmind n'est ni affilié à ni approuvé par Pollen Robotics. Microduck est une marque de Pollen Robotics.",
    },
    "ja": {
        "lang": "ja",
        "cjk": ["Noto Sans JP"],
        "title": "duckmind — README メイン画像（日本語）",
        "partner_left": "非公式 • MICRODUCK 互換ロボット向け • オープンソースのペット脳",
        "partner_right": "PYTHON 3.12 + MUJOCO • コード：APACHE-2.0",
        "kicker_tag": "MICRODUCK 互換のロボットアヒルのためのペット脳",
        "hero_subheadline": "気分 • 記憶 • 成長",
        "hero_desc": "小さなロボットアヒルに、自分の心を。<strong>台本どおりに動くリモコンのおもちゃではありません</strong>。気分や好奇心があり、人を覚え、芸を覚え、少しずつ成長します。いまはブラウザの 3D 物理シミュレーターで遊べます。",
        "btn_primary": "3D シミュレーターを開く →",
        "btn_secondary": "GITHUB リポジトリ ★",
        "frame_title": "DUCKMIND SIMWEB // 3D シミュレーター",
        "comic_badge": "自分で決める",
        "comic_text": "宙返りを頼んだけど、やらなかった。座ってうたた寝中。",
        "robot_badge": "★ SIM-TO-REAL・計画中 ★",
        "marquee_items": [
            "速い反射 + 遅い皮質 (SYSTEM 1 & 2)",
            "シードで決まる個性",
            "4 段階の成長",
            "進化する鳴き声の系統",
            "標準はローカル実行 • クラウドは任意",
        ],
        "credit": "アヒル：シミュレーターで描いた Pollen Robotics の Microduck 3D モデル（CC BY-NC-SA 4.0）。",
        "footnote": "* 非公式プロジェクトです。duckmind は Pollen Robotics とは関係がなく、承認や支援も受けていません。Microduck は Pollen Robotics の商標です。",
    },
    "ko": {
        "lang": "ko",
        "cjk": ["Noto Sans KR"],
        "title": "duckmind — README 대표 이미지 (한국어)",
        "partner_left": "비공식 • MICRODUCK 호환 로봇용 • 오픈 소스 반려 두뇌",
        "partner_right": "PYTHON 3.12 + MUJOCO • 코드: APACHE-2.0",
        "kicker_tag": "MICRODUCK 호환 로봇 오리를 위한 반려 두뇌",
        "hero_subheadline": "감정 • 기억 • 성장",
        "hero_desc": "작은 로봇 오리에게 자기만의 마음을 줍니다. <strong>각본대로 움직이는 리모컨 장난감이 아닙니다</strong>. 기분과 호기심이 있고, 사람을 기억하고, 재주를 배우며, 조금씩 자랍니다. 지금은 브라우저의 3D 물리 시뮬레이터에서 만나 볼 수 있어요.",
        "btn_primary": "3D 시뮬레이터 열기 →",
        "btn_secondary": "GITHUB 저장소 ★",
        "frame_title": "DUCKMIND SIMWEB // 3D 시뮬레이터",
        "comic_badge": "스스로 결정",
        "comic_text": "구르기를 시켰지만 하지 않았어요. 앉아서 졸고 있었거든요.",
        "robot_badge": "★ SIM-TO-REAL · 계획 중 ★",
        "marquee_items": [
            "빠른 반사 + 느린 피질 (SYSTEM 1 & 2)",
            "시드로 정해지는 개성",
            "4단계 성장",
            "진화하는 울음소리 계보",
            "기본은 로컬 실행 • 클라우드는 선택",
        ],
        "credit": "오리: 시뮬레이터로 그린 Pollen Robotics의 Microduck 3D 모델 (CC BY-NC-SA 4.0).",
        "footnote": "* 비공식 프로젝트입니다. duckmind는 Pollen Robotics와 관련이 없으며 승인이나 후원을 받지 않았습니다. Microduck은 Pollen Robotics의 상표입니다.",
    },
}

TEMPLATE = (HERE / "template.html").read_text(encoding="utf-8")


def page(code: str) -> str:
    cfg = CONFIGS[code]
    families = ", ".join(f"'{family}'" for family in [*cfg["cjk"], "Noto Sans TC"])
    faces = "\n".join(
        f"  @font-face {{ font-family: '{family}'; font-style: normal; font-weight: 100 900;"
        f" src: url(fonts/{CJK_FONTS[family][2]}) format('woff2'); }}"
        for family in cfg["cjk"]
    )
    marquee = "\n".join(
        f'  <div class="marquee-item"><span class="star-bullet">★</span><span>{item}</span></div>'
        for item in [*cfg["marquee_items"], GITHUB]
    )
    values = {
        **{key: value for key, value in cfg.items() if isinstance(value, str)},
        "font_links": f'<link rel="stylesheet" href="{FONTS}fonts.css">',
        "font_faces": faces,
        "cjk_families": families,
        "marquee_html": marquee,
        "comic_text": f"{cfg['comic_text']} <span>(Zzz • {DUCK_NAME})</span>",
    }
    return TEMPLATE.format(**values)


def visible_text(markup: str) -> str:
    body = markup[markup.index("<body>") :]
    body = re.sub(r"<script>.*?</script>", "", body, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", "", body))


def subset_fonts() -> None:
    """fonts/noto-sans-{jp,kr,sc}.woff2: every character the banners using that family show."""
    from fontTools import subset
    from fontTools.ttLib import TTFont

    (HERE / "fonts").mkdir(exist_ok=True)
    FONT_CACHE.mkdir(parents=True, exist_ok=True)
    for family, (folder, filename, output) in CJK_FONTS.items():
        chars = {" "}
        for code, cfg in CONFIGS.items():
            if family in cfg["cjk"]:
                chars |= {c for c in visible_text(page(code)) if not c.isspace()}
        source = FONT_CACHE / filename
        if not source.is_file():
            quoted = filename.replace("[", "%5B").replace("]", "%5D")
            with urllib.request.urlopen(
                GOOGLE_FONTS_REPO + folder + "/" + quoted, timeout=120
            ) as response:
                source.write_bytes(response.read())
        font = TTFont(source)
        options = subset.Options()
        options.flavor = "woff2"
        options.layout_features = ["*"]
        options.name_IDs = ["*"]
        options.name_languages = ["*"]
        options.notdef_outline = True
        subsetter = subset.Subsetter(options)
        subsetter.populate(text="".join(sorted(chars)))
        subsetter.subset(font)
        font.flavor = "woff2"
        font.save(HERE / "fonts" / output)
        print(
            f"fonts/{output}: {len(chars)} characters, {(HERE / 'fonts' / output).stat().st_size // 1024} KB"
        )


def chrome_binary() -> str:
    for candidate in (
        os.environ.get("CHROME"),
        MAC_CHROME,
        shutil.which("google-chrome"),
        shutil.which("chromium"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    sys.exit("Google Chrome not found: set CHROME to its binary")


def render(code: str, port: int) -> Path:
    png = OUT / f"hero.{code}.png"
    webp = OUT / f"hero.{code}.webp"
    png.unlink(missing_ok=True)
    url = f"http://127.0.0.1:{port}/docs/media/hero/src/hero.{code}.html"
    with tempfile.TemporaryDirectory() as profile:
        result = subprocess.run(
            [
                chrome_binary(),
                "--headless=new",
                f"--user-data-dir={profile}",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                "--window-size=1200,630",
                "--virtual-time-budget=15000",
                f"--screenshot={png}",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    if not png.is_file():
        sys.exit(f"{code}: no picture\n{result.stderr[-2000:]}")
    subprocess.run(
        ["cwebp", "-quiet", "-q", "82", "-m", "6", "-metadata", "none", str(png), "-o", str(webp)],
        check=True,
    )
    return png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "langs", nargs="*", metavar="LANG", help=f"some of {', '.join(CONFIGS)} (default: all)"
    )
    parser.add_argument(
        "--subset-fonts", action="store_true", help="rebuild fonts/ first (needs fontTools)"
    )
    parser.add_argument("--html-only", action="store_true")
    parser.add_argument("--port", type=int, default=8782)
    args = parser.parse_args(argv)
    langs = args.langs or list(CONFIGS)
    if unknown := [code for code in langs if code not in CONFIGS]:
        parser.error(f"unknown language: {', '.join(unknown)}")

    if args.subset_fonts:
        subset_fonts()
    for code in langs:
        (HERE / f"hero.{code}.html").write_text(page(code), encoding="utf-8")
    if args.html_only:
        return 0

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), functools.partial(Quiet, directory=str(REPO))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for code in langs:
            png = render(code, args.port)
            webp = png.with_suffix(".webp")
            print(
                f"{code}: {png.name} {png.stat().st_size // 1024} KB, {webp.name} {webp.stat().st_size // 1024} KB"
            )
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
