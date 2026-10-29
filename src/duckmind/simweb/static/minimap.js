/* The minimap on the view (owner 2026-10-11): the home from above, small, with the duck's live place.
 *
 * It sits in a corner of the stage (bottom right) and draws the layout the physics runs (the same page layout
 * the 3D view builds the room from: /api/home, or /api/body's home_json when the home changes, e.g. after the
 * layout editor's Apply) with the shared top-down drawing of plan2d.js: walls, doorways, furniture and toys as
 * simple shapes, the room names. The duck is a dot with a heading tick; the family, when they are in the home,
 * are small dots. The duck's place is the body's ground truth (/api/body `frame.truth`), which the page may
 * show but the brain never reads. Nothing is polled here: it listens to the answers the page already asks for
 * (`duck:body`, `duck:home`). With no physics body (the demo duck, Duck Core's own mode) or no layout it hides.
 * Fold it with its button (remembered by this viewer, in this browser); a tap on the map makes it bigger for a
 * moment. Every word comes from the catalogs (I18N). Nothing here decides anything for the duck.
 *
 * What the duck itself knows (MAP-EXPLORE 2026-10-12): with the duck's lasting map on, a faint layer in the plan colour shows
 * its OWN map (/api/neuro/knows: a coarse run-length coded grid of 0.2 m squares — floor it has seen clear, things it learned,
 * not sure; uncoloured = never seen). It is drawn where the duck believes it is, laid
 * over the true plan by the body's true pose, so its drift shows. Its button in the title bar toggles it (remembered by this
 * viewer); the button is there only while the duck keeps a map. */
"use strict";

(function miniMap() {
  const root = document.getElementById("miniMap");
  const plan = window.DuckPlan;
  if (!root || !plan) return;
  const svgEl = document.getElementById("miniMapSvg");
  const fold = document.getElementById("miniMapFold");
  const body = document.getElementById("miniMapBody");
  const area = root.closest(".stage-area");
  const hud = document.getElementById("hud");
  const viewBox = document.getElementById("duckView");
  const FOLD_KEY = "duck.minimap.fold";
  const KNOWS_KEY = "duck.minimap.knows";
  const KNOWS_MS = 3000; // (how often its map is asked for while the layer shows; 15 s while it is off, to know whether to offer it)
  const knowsBtn = document.getElementById("miniMapKnows");
  const BIG_MS = 12000;
  const PHONE = window.matchMedia ? window.matchMedia("(max-width: 760px)") : null;
  const t = (key, params) => (window.I18N ? window.I18N.t(key, params) : key);
  const svg = plan.svg;

  const S = {
    layout: null,
    fp: "",
    live: false, // the physics body runs (the map is meaningful)
    duck: null, // {x, y, yaw} mm, mrad
    people: [], // [{who, x, y}]
    things: new Map(), // id -> toy position now
    nodes: null, // {duck, people, things: Map(id -> {node, x, y})}
    radius: 150,
    asking: 0,
    raf: 0,
    sizeTimer: 0,
    folded: false,
    knowsOn: true, // the viewer wants the duck's own map drawn
    knowsAvail: false, // the duck keeps a map (the server says so)
    knows: null, // the last /api/neuro/knows answer
    knowsTimer: 0,
  };

  /* ---------------- fold: remembered per viewer ---------------- */

  const store = {
    get(key = FOLD_KEY) {
      try {
        return window.localStorage.getItem(key);
      } catch (error) {
        return null;
      }
    },
    set(value, key = FOLD_KEY) {
      try {
        window.localStorage.setItem(key, value);
      } catch (error) {
        /* storage may be blocked: the choice just is not remembered */
      }
    },
  };

  // (words are written once the catalogs are in; I18N.attr then remembers them for a language switch)
  const ready = window.I18N && window.I18N.ready ? window.I18N.ready : Promise.resolve();
  const words = (el, attr, key) => {
    if (!window.I18N) return;
    const write = () => window.I18N.attr(el, attr, key);
    ready.then(write, write);
  };

  function setFolded(folded, remember) {
    S.folded = folded;
    root.dataset.state = folded ? "folded" : "open";
    fold.setAttribute("aria-expanded", String(!folded));
    for (const attr of ["aria-label", "title"]) words(fold, attr, folded ? "page.minimap.expand" : "page.minimap.collapse");
    if (folded) setBig(false);
    if (remember) store.set(folded ? "folded" : "open");
    settle();
    planKnows(0);
  }

  // (no choice yet: open on a desktop, folded on a phone, where the view is small)
  const saved = store.get();
  setFolded(saved ? saved === "folded" : !!(PHONE && PHONE.matches), false);
  fold.addEventListener("click", () => setFolded(!S.folded, true));

  /* ---------------- bigger for a moment ---------------- */

  function setBig(big) {
    root.dataset.size = big ? "large" : "normal";
    for (const attr of ["aria-label", "title"]) words(body, attr, big ? "page.minimap.shrink" : "page.minimap.enlarge");
    clearTimeout(S.sizeTimer);
    if (big) S.sizeTimer = setTimeout(() => setBig(false), BIG_MS);
    settle();
  }
  body.addEventListener("click", () => setBig(root.dataset.size !== "large"));

  /* ---------------- the plan ---------------- */

  // (a wide letter — Chinese, Japanese, Korean — takes about one em; a Latin one about 0.6)
  const isWide = (ch) => ch.codePointAt(0) >= 0x2e80;
  const roomName = (r) => (window.I18N ? window.I18N.field(r, "zh") : r.zh) || r.id;

  function labelFont(text, width) {
    let units = 0;
    for (const ch of text) units += isWide(ch) ? 1 : 0.62;
    return Math.max(150, Math.min(340, (width * 0.9) / Math.max(units, 1)));
  }

  function drawLabels() {
    const layer = S.nodes && S.nodes.labels;
    if (!layer || !S.layout) return;
    layer.replaceChildren();
    for (const r of S.layout.rooms || []) {
      const [x0, y0, x1, y1] = r.rect_mm;
      const text = roomName(r);
      const size = labelFont(text, x1 - x0 - 140);
      const node = svg("text", { x: x0 + 70, y: -(y1 - 60 - size * 0.8), class: "mm-label", style: `font-size:${size}px` }, layer);
      node.textContent = text;
    }
  }

  function build(layout) {
    S.layout = layout;
    S.fp = layout.fingerprint || "";
    svgEl.replaceChildren();
    const [bx0, by0, bx1, by1] = layout.bounds_mm || [-3000, -3000, 3000, 3000];
    const w = bx1 - bx0;
    const h = by1 - by0;
    svgEl.setAttribute("viewBox", `${bx0} ${-by1} ${w} ${h}`);
    svgEl.style.aspectRatio = `${w} / ${h}`;
    S.aspect = w / h;
    const rooms = svg("g", { class: "mm-rooms" }, svgEl);
    for (const r of layout.rooms || []) {
      const [x0, y0, x1, y1] = r.rect_mm;
      svg("rect", { x: x0, y: -y1, width: x1 - x0, height: y1 - y0, class: "mm-room" }, rooms);
    }
    const m = layout.mat;
    if (m && m.drawn !== false) svg("circle", { cx: m.x_mm, cy: -m.y_mm, r: m.radius_mm, class: "mm-mat" }, svgEl);
    const fixed = { elements: (layout.elements || []).filter((e) => e.flags && e.flags.page), things: layout.things || [] };
    const toys = plan.drawFixed(svg("g", { class: "mm-fixed-layer" }, svgEl), fixed, { cls: "mm-fixed", kindClass: "mm-k-", minLine: 45, sort: true, wrapThings: true });
    for (const d of layout.doorways || []) {
      const [ax, ay] = d.from_mm;
      const [bx, by] = d.to_mm;
      svg("line", { x1: ax, y1: -ay, x2: bx, y2: -by, class: "mm-door" }, svgEl);
    }
    const knows = svg("g", { class: "mm-knows-layer" }, svgEl);
    const labels = svg("g", { class: "mm-labels" }, svgEl);
    const people = svg("g", { class: "mm-people" }, svgEl);
    S.radius = Math.max(w, h) * 0.034; // (a dot about 5 px across at the normal size)
    const duck = svg("g", { class: "mm-duck", visibility: "hidden" }, svgEl);
    svg("circle", { r: S.radius * 1.75, class: "mm-halo" }, duck);
    svg("circle", { r: S.radius, class: "mm-dot" }, duck);
    svg("line", { x1: 0, y1: 0, x2: S.radius * 2.1, y2: 0, class: "mm-tick" }, duck);
    S.nodes = { labels, people, duck, toys, knows, peopleNodes: new Map() };
    drawKnows();
    drawLabels();
    S.things = new Map();
    paint();
    settle();
  }

  /* ---------------- what moves ---------------- */

  function paint() {
    S.raf = 0;
    const n = S.nodes;
    if (!n) return;
    const d = S.duck;
    if (d) {
      n.duck.setAttribute("transform", `translate(${d.x} ${-d.y}) rotate(${(-d.yaw * 180) / Math.PI / 1000})`);
      n.duck.setAttribute("visibility", "visible");
    } else {
      n.duck.setAttribute("visibility", "hidden");
    }
    // the family in the home: a dot each (names are data and are not drawn)
    const seen = new Set();
    for (const p of S.people) {
      seen.add(p.who);
      let node = n.peopleNodes.get(p.who);
      if (!node) {
        node = svg("circle", { r: S.radius * 0.8, class: "mm-person" }, n.people);
        n.peopleNodes.set(p.who, node);
      }
      node.setAttribute("cx", p.x);
      node.setAttribute("cy", -p.y);
    }
    for (const [who, node] of n.peopleNodes) {
      if (!seen.has(who)) {
        node.remove();
        n.peopleNodes.delete(who);
      }
    }
    // a toy that was moved (a ball rolled, a sock carried): its pieces follow
    for (const [id, now] of S.things) {
      const toy = n.toys.get(id);
      if (!toy) continue;
      const dx = now.x - toy.x;
      const dy = now.y - toy.y;
      toy.node.setAttribute("transform", Math.abs(dx) + Math.abs(dy) > 15 ? `translate(${dx} ${-dy})` : "");
    }
  }

  function schedule() {
    if (!S.raf) S.raf = requestAnimationFrame(paint);
  }

  const finite = (v) => typeof v === "number" && Number.isFinite(v);

  function takeBody(info) {
    const live = !!(info && info.enabled);
    if (live !== S.live) {
      S.live = live;
      settle();
      planKnows(0);
    }
    if (!live) return;
    if (info.home_json && info.home_json.layout) {
      if (info.home_json.fingerprint !== S.fp) build(info.home_json.layout);
    } else {
      // (the body's home is not the one drawn here: ask for the new one — after the editor's Apply, say)
      const world = info.world || {};
      const fp = world.home && world.home.fingerprint;
      if (fp && S.fp && fp !== S.fp) askHome();
    }
    const truth = info.available && info.frame ? info.frame.truth : null;
    S.duck = truth && truth.frame === "world" && finite(truth.x_mm) && finite(truth.y_mm) ? { x: truth.x_mm, y: truth.y_mm, yaw: finite(truth.yaw_mrad) ? truth.yaw_mrad : 0 } : null;
    const world = info.world || {};
    S.people = (Array.isArray(world.people) ? world.people : [])
      .filter((p) => p && Array.isArray(p.pos_mm) && finite(p.pos_mm[0]) && finite(p.pos_mm[1]) && p.state !== "away")
      .map((p) => ({ who: String(p.who), x: p.pos_mm[0], y: p.pos_mm[1] }));
    S.things = new Map();
    for (const th of world.things || []) {
      if (th && th.id && Array.isArray(th.pos_mm) && finite(th.pos_mm[0]) && finite(th.pos_mm[1])) S.things.set(th.id, { x: th.pos_mm[0], y: th.pos_mm[1] });
    }
    schedule();
  }

  async function askHome() {
    if (S.asking && Date.now() - S.asking < 2500) return;
    S.asking = Date.now();
    try {
      const home = await fetch("/api/home", { cache: "no-store" }).then((r) => r.json());
      if (home && home.layout && home.fingerprint && home.fingerprint !== S.fp) build(home.layout);
    } catch (error) {
      /* the page asks again with the next answer */
    }
  }

  window.addEventListener("duck:body", (event) => takeBody(event.detail));
  window.addEventListener("duck:home", (event) => {
    const home = event.detail;
    if (home && home.layout && home.fingerprint !== S.fp) build(home.layout);
  });

  /* ---------------- what the duck itself knows (its own map) ---------------- */

  const KNOWS_CLASS = { b: "mm-kn-free", c: "mm-kn-thing", d: "mm-kn-unsure" };

  function drawKnows() {
    const layer = S.nodes && S.nodes.knows;
    if (!layer) return;
    layer.replaceChildren();
    const k = S.knows;
    if (!S.knowsOn || !k || !k.on || !k.world || !k.grid || !(k.grid.w > 0)) return;
    const g = k.grid;
    const w = k.world;
    // map frame (m, y up) -> the plan (mm, y down): translate to where the map's origin lies, turn, flip y, metres to mm
    const frame = svg("g", { transform: `translate(${w.x * 1000} ${-w.y * 1000}) rotate(${(-w.yaw * 180) / Math.PI}) scale(1000 -1000)` }, layer);
    const cell = g.cell_m;
    const [ox, oy] = g.origin;
    let at = 0;
    for (const [, ch, count] of String(g.rle).matchAll(/([a-d])(\d+)/g)) {
      let n = Number(count);
      const cls = KNOWS_CLASS[ch];
      while (n > 0) {
        const row = Math.floor(at / g.w);
        const col = at % g.w;
        const run = Math.min(n, g.w - col); // (a run is cut at the row's end: one rectangle a row)
        if (cls) svg("rect", { x: ox + col * cell, y: oy + row * cell, width: run * cell, height: cell, class: cls }, frame);
        at += run;
        n -= run;
      }
    }
  }

  function setKnowsAvail(avail) {
    if (avail === S.knowsAvail) return;
    S.knowsAvail = avail;
    if (knowsBtn) knowsBtn.hidden = !avail;
    settle();
  }

  function setKnowsOn(on, remember) {
    S.knowsOn = on;
    if (knowsBtn) knowsBtn.setAttribute("aria-pressed", String(on));
    if (remember) store.set(on ? "on" : "off", KNOWS_KEY);
    drawKnows();
    planKnows(0);
  }

  async function askKnows() {
    S.knowsTimer = 0;
    if (!S.live || document.hidden) return planKnows();
    try {
      const answer = await fetch("/api/neuro/knows", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null));
      S.knows = answer && answer.on ? answer : null;
      setKnowsAvail(!!S.knows);
    } catch (error) {
      S.knows = null; // (the server is away: asked again later)
      setKnowsAvail(false);
    }
    drawKnows();
    planKnows();
  }

  function planKnows(delay) {
    clearTimeout(S.knowsTimer);
    const wait = delay !== undefined ? delay : S.knowsOn && !S.folded ? KNOWS_MS : KNOWS_MS * 5;
    S.knowsTimer = setTimeout(askKnows, wait);
  }

  if (knowsBtn) {
    for (const attr of ["aria-label", "title"]) words(knowsBtn, attr, "page.minimap.knows");
    knowsBtn.addEventListener("click", () => setKnowsOn(!S.knowsOn, true));
    setKnowsOn(store.get(KNOWS_KEY) !== "off", false);
  }
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) planKnows(0);
  });
  if (window.I18N && typeof window.I18N.onChange === "function") window.I18N.onChange(drawLabels);

  /* ---------------- where it sits ---------------- */

  // Shown only while there is a physics body and a layout to draw. It sits on the left edge of the view, in the
  // left column between the 物理真相 label (above) and the 現在在做 card, the HUD (below), and it never lies over
  // another card of the view:
  //  - below it, the HUD and the duck's words (when shown): the map lifts above them (--mm-lift);
  //  - above it, the clock, the 物理真相 label (and, if it is wide enough to reach them, the view's buttons and
  //    the eye window): the map is only as big as the room between allows (--mm-fit);
  //  - beside it, the enlarged view's floating panel: the map is only as wide as the room left of it;
  //  - it never goes past the middle of the view (a phone: a smaller map).
  // Where there is no room the map keeps its title bar, then only its icon, still on the left.
  const live = (el) => !!el && !el.hidden && el.getClientRects().length > 0;
  const SMALLEST = 80; // (px: a map smaller than this is no use)

  function settle() {
    const show = S.live && !!S.layout;
    if (root.hidden === show) root.hidden = !show;
    if (!area) return;
    requestAnimationFrame(() => {
      const shown = !root.hidden && root.getClientRects().length > 0;
      if (!shown) return;
      const areaBox = area.getBoundingClientRect();
      const rootBox = root.getBoundingClientRect();
      const head = root.querySelector(".mm-head");
      const bar = head.offsetHeight + 4;
      const widest = Math.max(head.scrollWidth, parseFloat(getComputedStyle(root).getPropertyValue("--mm-w")) || 0) + 4;
      const right = rootBox.left + widest; // (its widest extent: it does not flip back and forth)
      const across = (box) => box.left < right + 8 && box.right > rootBox.left - 8;
      // below it: the cards of the bottom row
      let lift = 0;
      const below = [hud, document.getElementById("duck3dCaption")];
      for (const card of below) {
        if (!live(card) || (card.id === "duck3dCaption" && !card.classList.contains("show"))) continue;
        const box = card.getBoundingClientRect();
        if (across(box)) lift = Math.max(lift, Math.ceil(areaBox.bottom - box.top - 2));
      }
      root.style.setProperty("--mm-lift", `${lift}px`);
      // above it: the cards of the top rows that reach over it
      let top = areaBox.top + 10;
      const above = [document.getElementById("duck3dClock"), document.getElementById("stageMode"), area.querySelector(".stage-tools")];
      const side = area.querySelector(".stage-side");
      if (side) above.push(...side.children);
      for (const card of above) {
        if (!live(card)) continue;
        const box = card.getBoundingClientRect();
        if (across(box)) top = Math.max(top, box.bottom);
      }
      const room = areaBox.bottom - (10 + lift) - (top + 8);
      // beside it: not past the middle of the view, nor under the floating panel
      let wide = Math.floor(areaBox.width / 2) - 14 - 4;
      const panel = document.getElementById("ctlPanel");
      const big = !!(viewBox && viewBox.classList.contains("is-big"));
      if (big && viewBox.dataset.ctl === "open" && live(panel)) wide = Math.min(wide, Math.floor(panel.getBoundingClientRect().left - areaBox.left - 10 - 12 - 4));
      const byRoom = Math.floor((room - bar) * (S.aspect || 1));
      const fit = Math.min(byRoom, wide);
      root.style.setProperty("--mm-fit", `${Math.max(SMALLEST, fit)}px`);
      // (a view with room for the title bar but not for a map keeps the title bar; with room for neither, only the icon)
      root.dataset.fit = room < bar || wide < 96 ? "icon" : fit < SMALLEST ? "head" : "";
    });
  }

  if ("ResizeObserver" in window) {
    const watcher = new ResizeObserver(settle);
    if (area) watcher.observe(area);
    if (hud) watcher.observe(hud);
    for (const id of ["duck3dCaption", "duck3dClock", "stageMode"]) {
      const card = document.getElementById(id);
      if (card) watcher.observe(card);
    }
    watcher.observe(root);
  }
  window.addEventListener("resize", settle);
  window.addEventListener("duck:neuro", settle);
  if (hud) new MutationObserver(settle).observe(hud, { attributes: true, attributeFilter: ["hidden"] });
  const speech = document.getElementById("duck3dCaption");
  if (speech) new MutationObserver(settle).observe(speech, { childList: true, attributes: true, attributeFilter: ["class"] });
  if (viewBox) new MutationObserver(settle).observe(viewBox, { attributes: true, attributeFilter: ["class", "data-ctl"] });

  // The 3D view asks for /api/home at start and tells the page (`duck:home`); this asks too, once, in case that
  // answer came before this script ran (the 2D duck, a slow start).
  askHome();
  window.duckMiniMap = { setFolded, setBig };
})();
