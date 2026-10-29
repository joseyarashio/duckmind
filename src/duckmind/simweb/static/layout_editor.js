/* The layout editor on the 資料 page (owner 2026-10-10; docs/design/HOME-FORMAT.md).
 *
 * A top-down map of the home that runs. The viewer drags items from the object library onto it (or taps an
 * item, then the map), moves, rotates and deletes them, checks the result, saves it under a name, opens a saved
 * one, resets to the default home, and applies it to the running simulator. The layout is plain data (a home
 * file with "placements"); nothing here decides anything for the duck. Every word comes from the catalogs
 * (I18N); item names are page.lib.item.<id>. The server checks (POST /api/layout/check) and applies. */
"use strict";

(function layoutEditor() {
  const root = document.getElementById("homeEditor");
  if (!root) return;
  const $ = (id) => document.getElementById(id);
  const t = (key, params) => (window.I18N ? window.I18N.t(key, params) : key);
  const SVGNS = "http://www.w3.org/2000/svg";
  const PASS = ["under_foot", "step", "hop", "block", "soft", "movable"];
  const MRAD_PER_DEG = Math.PI / 180 * 1000;
  const GRID_MM = 50;
  const UNDO_MAX = 60;
  const SNAP_KEY = "duck.layout.snap";

  const els = {
    map: $("leMap"),
    status: $("leStatus"),
    library: $("leLibrary"),
    cats: $("leCats"),
    legend: $("leLegend"),
    sel: $("leSelection"),
    problems: $("leProblems"),
    problemsTitle: $("leProblemsTitle"),
    name: $("leName"),
    saved: $("leSaved"),
    running: $("leRunning"),
    snap: $("leSnap"),
  };

  const S = {
    srv: null, // the home that runs, as the server sent it (GET /api/layout)
    lib: new Map(), // item id -> library item
    libList: [],
    catList: [],
    layout: null, // the working copy (a home file: placements)
    base: "", // JSON of the layout the simulator runs (to say "changed")
    selected: null, // a placement id
    armed: null, // a library item id waiting for a tap on the map
    cat: "all",
    problems: [],
    reach: null,
    undo: [],
    redo: [],
    snap: true,
    view: null, // {x, y, w, h} in mm, y up: the part of the home on screen
    checkTimer: 0,
    checkSeq: 0,
    busy: false,
    loaded: false,
    drag: null,
    pan: null,
    ghost: null,
    waitApply: false,
  };
  try {
    S.snap = window.localStorage.getItem(SNAP_KEY) !== "0";
  } catch (error) {
    /* storage may be blocked: snapping is on */
  }

  /* ---------------- small helpers ---------------- */

  const el = (tag, cls, text, parent) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  };
  const svg = (tag, attrs, parent) => {
    const node = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, String(v));
    if (parent) parent.appendChild(node);
    return node;
  };
  const rot = (x, y, mrad) => {
    const a = mrad / 1000;
    const c = Math.cos(a);
    const s = Math.sin(a);
    return [c * x - s * y, s * x + c * y];
  };
  const wrap = (mrad) => {
    const turn = 2000 * Math.PI;
    let v = (mrad + turn / 2) % turn;
    if (v < 0) v += turn;
    return Math.round(v - turn / 2);
  };
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const itemName = (id) => (S.lib.has(id) ? t(`page.lib.item.${id}`) : id);
  const say = (key, params, tone) => {
    els.status.dataset.tone = tone || "";
    window.I18N.text(els.status, key, params);
  };
  const placements = () => S.layout.placements;
  const find = (id) => placements().find((p) => p.id === id) || null;

  async function api(path, body) {
    const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const res = await fetch(path, opts);
    return res.json();
  }

  /* ---------------- the map ---------------- */

  const CLASS_OF_KIND = { floor: 0, doorway: 1, cable: 2, furniture: 1, clutter: 3, toy: 3 };

  function drawParts(item, p, group) {
    // every felt part, flat, in world mm (y up: the map flips y at the end)
    const yaw = p.yaw_mrad || 0;
    for (const part of item.draw) {
      let at = part.at;
      let size = part.size;
      let from = part.from;
      let to = part.to;
      if (p.length_mm && item.resize_x) {
        const k = p.length_mm;
        if (size) size = [k, size[1]];
        if (from) {
          from = [-k / 2, from[1]];
          to = [k / 2, to[1]];
        }
      }
      const colour = p.colour || part.colour || item.colour;
      const common = { fill: colour, "data-part": "1" };
      if (part.shape === "box" || part.shape === "slab") {
        const [w, d] = size;
        const pts = [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]].map(([a, b]) => {
          const [rx, ry] = rot(at[0] + a, at[1] + b, yaw);
          return `${p.x_mm + rx},${-(p.y_mm + ry)}`;
        });
        svg("polygon", { ...common, points: pts.join(" ") }, group);
      } else if (part.shape === "capsule") {
        const [ax, ay] = rot(from[0], from[1], yaw);
        const [bx, by] = rot(to[0], to[1], yaw);
        svg("line", { x1: p.x_mm + ax, y1: -(p.y_mm + ay), x2: p.x_mm + bx, y2: -(p.y_mm + by), style: `stroke:${colour};stroke-width:${Math.max(part.r * 2, 14)}px;stroke-linecap:round`, "data-part": "1" }, group);
      } else {
        const [rx, ry] = rot(at[0], at[1], yaw);
        svg("circle", { ...common, cx: p.x_mm + rx, cy: -(p.y_mm + ry), r: part.r }, group);
      }
    }
  }

  function hitBox(item, p, group) {
    // a thin cable or a coin-sized item is hard to hit: an invisible, fatter box answers the pointer
    const w = Math.max(item.footprint_mm[0], 90);
    const d = Math.max(item.footprint_mm[1], 90);
    const len = p.length_mm && item.resize_x ? p.length_mm : w;
    const pts = [[-len / 2, -d / 2], [len / 2, -d / 2], [len / 2, d / 2], [-len / 2, d / 2]].map(([a, b]) => {
      const [rx, ry] = rot(a, b, p.yaw_mrad || 0);
      return `${p.x_mm + rx},${-(p.y_mm + ry)}`;
    });
    svg("polygon", { points: pts.join(" "), class: "le-hit" }, group);
  }

  // (the fixed walls, furniture and toys are drawn by the one shared top-down drawing, plan2d.js, which the
  // minimap on the view uses too)
  function drawFixed(group) {
    window.DuckPlan.drawFixed(group, S.srv.fixed, { cls: "le-fixed", kindClass: "le-k-" });
  }

  function render() {
    const map = els.map;
    if (!S.srv) return;
    map.replaceChildren();
    const v = S.view;
    map.setAttribute("viewBox", `${v.x} ${-(v.y + v.h)} ${v.w} ${v.h}`);
    map.classList.toggle("is-zoomed", v.w < S.srv.bounds_mm[2] - S.srv.bounds_mm[0] + 100);
    const rooms = svg("g", { class: "le-rooms" }, map);
    for (const r of S.srv.rooms) {
      const [x0, y0, x1, y1] = r.rect_mm;
      svg("rect", { x: x0, y: -y1, width: x1 - x0, height: y1 - y0, class: "le-room" }, rooms);
    }
    // 1 m ticks
    const grid = svg("g", { class: "le-grid" }, map);
    const [bx0, by0, bx1, by1] = S.srv.bounds_mm;
    for (let x = Math.ceil(bx0 / 1000) * 1000; x <= bx1; x += 1000) svg("line", { x1: x, y1: -by1, x2: x, y2: -by0 }, grid);
    for (let y = Math.ceil(by0 / 1000) * 1000; y <= by1; y += 1000) svg("line", { x1: bx0, y1: -y, x2: bx1, y2: -y }, grid);
    drawFixed(svg("g", { class: "le-fixed-layer" }, map));
    const m = S.srv.mat;
    if (m) svg("circle", { cx: m.x_mm, cy: -m.y_mm, r: m.radius_mm, class: "le-mat" }, map);
    const items = svg("g", { class: "le-items" }, map);
    const bad = new Set();
    for (const p of S.problems) {
      if (p.place) bad.add(p.place);
      if (p.other_place) bad.add(p.other_place);
    }
    const order = [...placements()].sort((a, b) => {
      const ia = S.lib.get(a.lib);
      const ib = S.lib.get(b.lib);
      const ka = ia ? (CLASS_OF_KIND[ia.category] ?? 1) : 1;
      const kb = ib ? (CLASS_OF_KIND[ib.category] ?? 1) : 1;
      return ka - kb;
    });
    for (const p of order) {
      const item = S.lib.get(p.lib);
      if (!item) continue;
      const g = svg("g", { class: `le-item le-pass-${item.passability}`, "data-id": p.id }, items);
      if (p.id === S.selected) g.classList.add("is-selected");
      if (bad.has(p.id)) g.classList.add("is-bad");
      drawParts(item, p, g);
      hitBox(item, p, g);
    }
    // the duck's start
    const sp = S.srv.spawn;
    if (sp) svg("circle", { cx: sp.x_mm, cy: -sp.y_mm, r: 70, class: "le-spawn" }, map);
    // room names (the map is not flipped, so the text reads upright)
    const labels = svg("g", { class: "le-labels" }, map);
    for (const r of S.srv.rooms) {
      const [x0, y0, x1, y1] = r.rect_mm;
      const tx = svg("text", { x: x0 + 90, y: -(y1 - 230), class: "le-label" }, labels);
      tx.textContent = roomName(r);
    }
    renderSelection();
    updateButtons();
  }

  // (a room's name is the home's own words: the server sends its message beside it)
  const roomName = (r) => (window.I18N ? window.I18N.field(r, "zh") : r.zh) || r.id;

  /* ---------------- pointer: place, move, pan ---------------- */

  function worldAt(event) {
    const pt = els.map.createSVGPoint();
    pt.x = event.clientX;
    pt.y = event.clientY;
    const inv = els.map.getScreenCTM().inverse();
    const q = pt.matrixTransform(inv);
    return [q.x, -q.y];
  }
  const snap = (v) => (S.snap ? Math.round(v / GRID_MM) * GRID_MM : Math.round(v));

  const capture = (event) => {
    try {
      els.map.setPointerCapture(event.pointerId);
    } catch (error) {
      /* a pointer that is already gone: the drag just ends */
    }
  };

  els.map.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !S.srv) return;
    const hit = event.target.closest("[data-id]");
    const [wx, wy] = worldAt(event);
    if (S.armed) {
      place(S.armed, snap(wx), snap(wy));
      event.preventDefault();
      return;
    }
    if (hit) {
      const p = find(hit.dataset.id);
      if (!p) return;
      select(p.id);
      S.drag = { id: p.id, dx: p.x_mm - wx, dy: p.y_mm - wy, moved: false, before: clone(placements()) };
      capture(event);
      event.preventDefault();
      return;
    }
    const fixed = event.target.closest("[data-fixed]");
    if (fixed) {
      select(null);
      say("page.layout.fixed_note", {}, "info");
    } else {
      select(null);
    }
    S.pan = { x: event.clientX, y: event.clientY, view: { ...S.view }, moved: false };
    capture(event);
  });

  els.map.addEventListener("pointermove", (event) => {
    if (S.drag) {
      const [wx, wy] = worldAt(event);
      const p = find(S.drag.id);
      if (!p) return;
      const nx = snap(wx + S.drag.dx);
      const ny = snap(wy + S.drag.dy);
      if (nx !== p.x_mm || ny !== p.y_mm) {
        p.x_mm = nx;
        p.y_mm = ny;
        S.drag.moved = true;
        render();
      }
    } else if (S.pan) {
      const box = els.map.getBoundingClientRect();
      const k = S.pan.view.w / box.width;
      const dx = (event.clientX - S.pan.x) * k;
      const dy = (event.clientY - S.pan.y) * k;
      if (Math.abs(event.clientX - S.pan.x) + Math.abs(event.clientY - S.pan.y) > 4) S.pan.moved = true;
      S.view.x = S.pan.view.x - dx;
      S.view.y = S.pan.view.y + dy;
      clampView();
      els.map.setAttribute("viewBox", `${S.view.x} ${-(S.view.y + S.view.h)} ${S.view.w} ${S.view.h}`);
    }
  });

  const endPointer = () => {
    if (S.drag) {
      const d = S.drag;
      S.drag = null;
      if (d.moved) {
        pushUndo(d.before);
        changed();
      }
    }
    S.pan = null;
  };
  els.map.addEventListener("pointerup", endPointer);
  els.map.addEventListener("pointercancel", endPointer);
  els.map.addEventListener(
    "wheel",
    (event) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      zoom(event.deltaY < 0 ? 1.15 : 1 / 1.15, worldAt(event));
    },
    { passive: false },
  );

  /* ---------------- the working layout ---------------- */

  function pushUndo(before) {
    S.undo.push(before);
    if (S.undo.length > UNDO_MAX) S.undo.shift();
    S.redo = [];
  }

  function newId(lib) {
    let n = 1;
    const used = new Set(placements().map((p) => p.id));
    while (used.has(`${lib}_${n}`)) n += 1;
    return `${lib}_${n}`;
  }

  function place(libId, x, y) {
    const item = S.lib.get(libId);
    if (!item) return;
    pushUndo(clone(placements()));
    const p = { id: newId(libId), lib: libId, x_mm: x, y_mm: y, yaw_mrad: 0 };
    if (item.resize_x) p.length_mm = Math.min(Math.max(900, item.resize_x[0]), item.resize_x[1]);
    placements().push(p);
    S.selected = p.id;
    S.armed = null; // one tap, one item (a tap on the item again picks it up to place another)
    els.map.classList.remove("is-armed");
    renderLibrary();
    changed();
    say("page.layout.status.placed", { name: itemName(libId) }, "ok");
  }

  function select(id) {
    S.selected = id;
    render();
  }

  function mutate(fn) {
    const p = find(S.selected);
    if (!p) return;
    pushUndo(clone(placements()));
    fn(p);
    changed();
  }

  function changed() {
    render();
    scheduleCheck();
    updateRunning();
  }

  function undo() {
    if (!S.undo.length) return;
    S.redo.push(clone(placements()));
    S.layout.placements = S.undo.pop();
    if (!find(S.selected)) S.selected = null;
    changed();
  }
  function redo() {
    if (!S.redo.length) return;
    S.undo.push(clone(placements()));
    S.layout.placements = S.redo.pop();
    if (!find(S.selected)) S.selected = null;
    changed();
  }

  /* ---------------- zoom ---------------- */

  function fitView() {
    const [x0, y0, x1, y1] = S.srv.bounds_mm;
    const pad = 60;
    S.view = { x: x0 - pad, y: y0 - pad, w: x1 - x0 + 2 * pad, h: y1 - y0 + 2 * pad };
  }
  function clampView() {
    const [x0, y0, x1, y1] = S.srv.bounds_mm;
    const pad = 400;
    S.view.x = Math.min(Math.max(S.view.x, x0 - pad), x1 + pad - S.view.w);
    S.view.y = Math.min(Math.max(S.view.y, y0 - pad), y1 + pad - S.view.h);
  }
  function zoom(k, about) {
    const [x0, , x1] = S.srv.bounds_mm;
    const full = x1 - x0 + 120;
    const w = Math.min(full, Math.max(1200, S.view.w / k));
    const h = (S.view.h / S.view.w) * w;
    const [ax, ay] = about || [S.view.x + S.view.w / 2, S.view.y + S.view.h / 2];
    S.view = { x: ax - ((ax - S.view.x) / S.view.w) * w, y: ay - ((ay - S.view.y) / S.view.h) * h, w, h };
    clampView();
    render();
  }

  /* ---------------- the library ---------------- */

  function renderLibrary() {
    els.library.replaceChildren();
    for (const item of S.libList) {
      if (S.cat !== "all" && item.category !== S.cat) continue;
      const b = el("button", "btn small le-item-btn", undefined, els.library);
      b.type = "button";
      b.dataset.lib = item.id;
      b.dataset.pass = item.passability;
      const swatch = el("span", "le-swatch", undefined, b);
      swatch.style.background = item.colour;
      el("span", "le-item-name", itemName(item.id), b);
      el("span", `le-pass-chip le-pass-${item.passability}`, t(`page.lib.class.${item.passability}`), b);
      b.setAttribute("aria-pressed", String(S.armed === item.id));
    }
  }

  function renderCats() {
    els.cats.replaceChildren();
    const mk = (id, label) => {
      const b = el("button", "btn small le-cat", label, els.cats);
      b.type = "button";
      b.dataset.cat = id;
      b.setAttribute("aria-pressed", String(S.cat === id));
    };
    mk("all", t("page.layout.cat.all"));
    for (const c of S.catList) mk(c, t(`page.lib.cat.${c}`));
  }

  function renderLegend() {
    els.legend.replaceChildren();
    for (const c of PASS) {
      const chip = el("li", `le-pass-chip le-pass-${c}`, undefined, els.legend);
      chip.textContent = t(`page.lib.class.${c}`);
    }
  }

  els.cats.addEventListener("click", (event) => {
    const b = event.target.closest("[data-cat]");
    if (!b) return;
    S.cat = b.dataset.cat;
    renderCats();
    renderLibrary();
  });

  // tap an item (then the map), or drag it onto the map
  let press = null;
  els.library.addEventListener("pointerdown", (event) => {
    const b = event.target.closest("[data-lib]");
    if (!b || event.button !== 0) return;
    press = { id: b.dataset.lib, x: event.clientX, y: event.clientY, dragging: false, pointer: event.pointerId };
  });
  window.addEventListener("pointermove", (event) => {
    if (!press || event.pointerId !== press.pointer) return;
    if (!press.dragging && Math.hypot(event.clientX - press.x, event.clientY - press.y) > 8 && event.pointerType === "mouse") {
      press.dragging = true;
      S.ghost = el("div", "le-ghost", itemName(press.id), document.body);
    }
    if (press.dragging && S.ghost) {
      S.ghost.style.left = `${event.clientX + 10}px`;
      S.ghost.style.top = `${event.clientY + 10}px`;
    }
  });
  const releaseLibrary = (event) => {
    if (!press || event.pointerId !== press.pointer) return;
    const p = press;
    press = null;
    if (S.ghost) {
      S.ghost.remove();
      S.ghost = null;
    }
    if (p.dragging) {
      const box = els.map.getBoundingClientRect();
      if (event.clientX >= box.left && event.clientX <= box.right && event.clientY >= box.top && event.clientY <= box.bottom) {
        const [wx, wy] = worldAt(event);
        place(p.id, snap(wx), snap(wy));
      }
      return;
    }
    // a tap: arm the item (or drop the arm)
    S.armed = S.armed === p.id ? null : p.id;
    renderLibrary();
    if (S.armed) say("page.layout.status.armed", { name: itemName(S.armed) }, "info");
    else say("page.layout.status.idle", {}, "");
    els.map.classList.toggle("is-armed", Boolean(S.armed));
  };
  window.addEventListener("pointerup", releaseLibrary);
  window.addEventListener("pointercancel", releaseLibrary);

  /* ---------------- the selected item ---------------- */

  function renderSelection() {
    const box = els.sel;
    const p = find(S.selected);
    box.replaceChildren();
    if (!p) {
      el("p", "hint", t("page.layout.sel.none"), box);
      return;
    }
    const item = S.lib.get(p.lib);
    const head = el("div", "le-sel-head", undefined, box);
    const sw = el("span", "le-swatch", undefined, head);
    sw.style.background = item.colour;
    el("strong", "le-sel-name", itemName(p.lib), head);
    el("span", `le-pass-chip le-pass-${item.passability}`, t(`page.lib.class.${item.passability}`), head);
    const fields = el("div", "le-fields", undefined, box);
    const num = (key, label, value, step, min, max, onSet) => {
      const wrapNode = el("label", "le-field", undefined, fields);
      el("span", undefined, label, wrapNode);
      const input = el("input", undefined, undefined, wrapNode);
      input.type = "number";
      input.step = String(step);
      if (min !== undefined) input.min = String(min);
      if (max !== undefined) input.max = String(max);
      input.value = String(value);
      input.dataset.field = key;
      input.addEventListener("change", () => {
        const v = Math.round(Number(input.value));
        if (!Number.isFinite(v)) return;
        mutate((q) => onSet(q, v));
      });
    };
    num("x", t("page.layout.sel.x"), p.x_mm, 10, undefined, undefined, (q, v) => { q.x_mm = v; });
    num("y", t("page.layout.sel.y"), p.y_mm, 10, undefined, undefined, (q, v) => { q.y_mm = v; });
    num("yaw", t("page.layout.sel.yaw"), Math.round((p.yaw_mrad || 0) / MRAD_PER_DEG), 5, -180, 180, (q, v) => { q.yaw_mrad = wrap(Math.round(v * MRAD_PER_DEG)); });
    if (item.resize_x) num("length", t("page.layout.sel.length"), p.length_mm || item.footprint_mm[0], 50, item.resize_x[0], item.resize_x[1], (q, v) => { q.length_mm = Math.min(Math.max(v, item.resize_x[0]), item.resize_x[1]); });
    const facts = el("dl", "le-facts", undefined, box);
    const fact = (label, value) => {
      el("dt", undefined, label, facts);
      el("dd", undefined, value, facts);
    };
    fact(t("page.layout.sel.size"), t("page.layout.sel.size_value", { w: item.footprint_mm[0], d: item.footprint_mm[1], h: item.height_mm }));
    if (item.mass_g) fact(t("page.layout.sel.mass"), t("page.layout.sel.mass_value", { g: item.mass_g }));
    fact(t("page.layout.sel.friction"), item.friction.toFixed(1));
    fact(t("page.layout.sel.contact"), t(item.contact === "soft" ? "page.layout.sel.contact_soft" : "page.layout.sel.contact_rigid"));
    fact(t("page.layout.sel.kind"), t(item.kind === "thing" ? "page.layout.sel.kind_thing" : "page.layout.sel.kind_fixed"));
  }

  /* ---------------- checks ---------------- */

  function scheduleCheck() {
    window.clearTimeout(S.checkTimer);
    S.checkTimer = window.setTimeout(runCheck, 250);
  }

  async function runCheck() {
    const seq = ++S.checkSeq;
    let res;
    try {
      res = await api("/api/layout/check", { layout: S.layout });
    } catch (error) {
      return;
    }
    if (seq !== S.checkSeq) return;
    applyCheck(res);
  }

  function applyCheck(res) {
    S.problems = (res.problems || []).map(enrich);
    S.reach = res.reach || null;
    renderProblems();
    render();
  }

  function enrich(p) {
    // which placement a problem is about (a piece's id is its placement's id plus a part name)
    const placeOf = (id) => {
      if (!id) return "";
      const direct = placements().find((q) => q.id === id || id.startsWith(`${q.id}_`));
      return direct ? direct.id : "";
    };
    return { ...p, place: p.place || placeOf(p.id), other_place: p.other_place || placeOf(p.other) };
  }

  function nameFor(p, which) {
    const id = which === "other" ? p.other : p.id;
    const place = which === "other" ? p.other_place : p.place;
    if (place) {
      const q = find(place);
      if (q) return itemName(q.lib);
    }
    const lib = which === "other" ? p.other_lib : p.lib;
    if (lib && S.lib.has(lib)) return itemName(lib);
    const rawKind = which === "other" ? p.other_kind : p.kind;
    const kind = /^(nest|dock)_/.test(rawKind || "") ? "nest" : rawKind;
    if (kind) {
      const key = `page.layout.kind.${kind}`;
      const word = t(key);
      if (word !== key) return word;
    }
    const room = (S.srv?.rooms || []).find((r) => r.id === id);
    if (room) return roomName(room);
    return t("page.layout.kind.other");
  }

  function renderProblems() {
    const ul = els.problems;
    ul.replaceChildren();
    if (!S.problems.length) {
      const li = el("li", "le-ok", t("page.layout.problems.ok"), ul);
      li.dataset.tone = "ok";
    }
    const said = new Set();
    for (const p of S.problems) {
      const key = `page.layout.problem.${p.code}`;
      let text = t(key, { name: nameFor(p, "id"), other: nameFor(p, "other") });
      if (text === key) text = t("page.layout.problem.other", { code: p.code });
      if (said.has(text)) continue;
      said.add(text);
      const li = el("li", "le-problem", undefined, ul);
      const b = el("button", "btn small", text, li);
      b.type = "button";
      if (p.place) b.addEventListener("click", () => select(p.place));
    }
    if (S.reach && S.reach.rooms) {
      const li = el("li", "le-reach", undefined, ul);
      li.textContent = t("page.layout.problems.reach", { pct: Math.round(S.reach.reached_pct) });
    }
  }

  /* ---------------- save, open, reset, apply ---------------- */

  // (the home keeps its id, so its rooms and pieces keep their translated names; the layout's own name is data)
  function layoutFor(name) {
    const out = clone(S.layout);
    if (!out.derived_from) out.derived_from = out.id;
    out.zh = name || t("page.layout.custom_name");
    return out;
  }

  async function save() {
    const name = els.name.value.trim();
    if (!name) {
      say("page.layout.status.need_name", {}, "warn");
      els.name.focus();
      return;
    }
    const res = await api("/api/layout/save", { name, layout: layoutFor(name) });
    if (res.ok) {
      say("page.layout.status.saved", { name }, "ok");
      renderSaved(res.saved);
    } else say(...failure(res), "warn");
  }

  function failure(res) {
    const key = `page.layout.error.${res.code || "other"}`;
    const word = t(key);
    if (word === key) return ["page.layout.error.other", { code: res.code || res.error || "?" }];
    return [key, { max: res.max, name: res.name }];
  }

  function renderSaved(list) {
    const sel = els.saved;
    sel.replaceChildren();
    if (!list || !list.length) {
      const o = el("option", undefined, t("page.layout.saved.none"), sel);
      o.value = "";
      sel.disabled = true;
    } else {
      sel.disabled = false;
      for (const item of list) {
        const o = el("option", undefined, item.name, sel);
        o.value = item.name;
      }
    }
    updateButtons();
  }

  async function openSaved() {
    const name = els.saved.value;
    if (!name) return;
    const res = await api("/api/layout/load", { name });
    if (!res.ok) return say(...failure(res), "warn");
    adopt(res.layout, res.check);
    els.name.value = name;
    say("page.layout.status.opened", { name }, "ok");
  }

  async function deleteSaved() {
    const name = els.saved.value;
    if (!name) return;
    const res = await api("/api/layout/delete", { name });
    if (!res.ok) return say(...failure(res), "warn");
    renderSaved(res.saved);
    say("page.layout.status.deleted", { name }, "ok");
  }

  function adopt(layout, check) {
    pushUndo(clone(placements()));
    S.layout = layout;
    if (!S.layout.placements) S.layout.placements = [];
    S.selected = null;
    if (check) applyCheck(check);
    else scheduleCheck();
    render();
    updateRunning();
  }

  async function resetToDefault() {
    const res = await api("/api/layout?which=default");
    if (!res.ok) return say(...failure(res), "warn");
    adopt(res.layout, res.check);
    say("page.layout.status.reset", {}, "ok");
  }

  function download() {
    const name = els.name.value.trim() || "layout";
    const blob = new Blob([JSON.stringify(layoutFor(els.name.value.trim()), null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${name}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  function upload(file) {
    const reader = new FileReader();
    reader.onload = async () => {
      let layout;
      try {
        layout = JSON.parse(String(reader.result));
      } catch (error) {
        return say("page.layout.error.bad_layout", {}, "warn");
      }
      const res = await api("/api/layout/check", { layout });
      if (res.ok === false && res.code) return say(...failure(res), "warn");
      adopt(layout, res);
      say("page.layout.status.opened_file", {}, "ok");
    };
    reader.readAsText(file);
  }

  async function applyNow() {
    if (S.busy) return;
    S.busy = true;
    updateButtons();
    say("page.layout.status.applying", {}, "info");
    let res;
    try {
      res = await api("/api/layout/apply", { layout: layoutFor(els.name.value.trim()) });
    } catch (error) {
      S.busy = false;
      updateButtons();
      return say("page.layout.error.other", { code: String(error) }, "warn");
    }
    if (!res.ok) {
      S.busy = false;
      updateButtons();
      if (res.code === "invalid") {
        applyCheck({ problems: res.problems, reach: S.reach });
        return say("page.layout.error.invalid", {}, "warn");
      }
      return say(...failure(res), "warn");
    }
    S.base = JSON.stringify(S.layout.placements);
    // the physics reloads: wait for the body, then say where the duck is
    say("page.layout.status.reloading", {}, "info");
    const t0 = Date.now();
    for (;;) {
      await new Promise((r) => window.setTimeout(r, 1500));
      let st = null;
      try {
        st = await api("/api/layout");
      } catch (error) {
        st = null;
      }
      if (st && st.ok && !st.applying) {
        S.srv = st;
        S.busy = false;
        updateButtons();
        updateRunning();
        say(st.restored === "there" ? "page.layout.status.done_there" : st.restored === "spawn" ? "page.layout.status.done_spawn" : "page.layout.status.done", {}, "ok");
        render();
        return;
      }
      if (Date.now() - t0 > 120000) {
        S.busy = false;
        updateButtons();
        return say("page.layout.status.slow", {}, "warn");
      }
    }
  }

  function updateRunning() {
    if (!S.srv) return;
    const dirty = JSON.stringify(S.layout.placements) !== S.base;
    els.running.dataset.dirty = String(dirty);
    window.I18N.text(els.running, dirty ? "page.layout.running.changed" : "page.layout.running.same");
  }

  function updateButtons() {
    const p = find(S.selected);
    for (const b of root.querySelectorAll("[data-needs-sel]")) b.disabled = !p;
    const undoBtn = root.querySelector('[data-le="undo"]');
    if (undoBtn) undoBtn.disabled = !S.undo.length;
    const redoBtn = root.querySelector('[data-le="redo"]');
    if (redoBtn) redoBtn.disabled = !S.redo.length;
    const apply = root.querySelector('[data-le="apply"]');
    if (apply) apply.disabled = S.busy || S.problems.length > 0;
    const saveBtn = root.querySelector('[data-le="save"]');
    if (saveBtn) saveBtn.disabled = S.problems.length > 0 || !S.srv?.can_save;
    const open = root.querySelector('[data-le="open"]');
    const del = root.querySelector('[data-le="delete"]');
    if (open) open.disabled = els.saved.disabled || !els.saved.value;
    if (del) del.disabled = els.saved.disabled || !els.saved.value;
  }

  /* ---------------- buttons and keys ---------------- */

  root.addEventListener("click", (event) => {
    const b = event.target.closest("[data-le]");
    if (!b || b.disabled) return;
    const act = b.dataset.le;
    const rotate = (deg) => mutate((p) => { p.yaw_mrad = wrap((p.yaw_mrad || 0) + Math.round(deg * MRAD_PER_DEG)); });
    if (act === "rot-left") rotate(15);
    else if (act === "rot-right") rotate(-15);
    else if (act === "rot-90") rotate(90);
    else if (act === "delete-item") {
      const id = S.selected;
      pushUndo(clone(placements()));
      S.layout.placements = placements().filter((q) => q.id !== id);
      S.selected = null;
      changed();
      say("page.layout.status.removed", {}, "ok");
    } else if (act === "duplicate") {
      const p = find(S.selected);
      if (!p) return;
      pushUndo(clone(placements()));
      const q = { ...clone(p), id: newId(p.lib), x_mm: p.x_mm + 200, y_mm: p.y_mm - 200 };
      placements().push(q);
      S.selected = q.id;
      changed();
    } else if (act === "undo") undo();
    else if (act === "redo") redo();
    else if (act === "zoom-in") zoom(1.4);
    else if (act === "zoom-out") zoom(1 / 1.4);
    else if (act === "fit") {
      fitView();
      render();
    } else if (act === "save") save();
    else if (act === "open") openSaved();
    else if (act === "delete") deleteSaved();
    else if (act === "reset") resetToDefault();
    else if (act === "apply") applyNow();
    else if (act === "download") download();
    else if (act === "upload") $("leFile").click();
  });
  $("leFile").addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) upload(file);
    event.target.value = "";
  });
  els.snap.addEventListener("change", () => {
    S.snap = els.snap.checked;
    try {
      window.localStorage.setItem(SNAP_KEY, S.snap ? "1" : "0");
    } catch (error) {
      /* the choice is just not remembered */
    }
  });
  els.saved.addEventListener("change", updateButtons);
  els.name.addEventListener("keydown", (event) => {
    if (event.key === "Enter") save();
  });

  root.addEventListener("keydown", (event) => {
    const inField = event.target.matches("input, select, textarea");
    if (event.key === "Escape") {
      if (S.armed) {
        S.armed = null;
        els.map.classList.remove("is-armed");
        renderLibrary();
        say("page.layout.status.idle", {}, "");
      } else if (S.selected) select(null);
      return;
    }
    if (inField) return;
    const p = find(S.selected);
    if (!p) return;
    const step = event.shiftKey ? 50 : 10;
    const move = (dx, dy) => {
      event.preventDefault();
      mutate((q) => { q.x_mm += dx; q.y_mm += dy; });
    };
    if (event.key === "ArrowLeft") move(-step, 0);
    else if (event.key === "ArrowRight") move(step, 0);
    else if (event.key === "ArrowUp") move(0, step);
    else if (event.key === "ArrowDown") move(0, -step);
    else if (event.key === "r" || event.key === "R") {
      event.preventDefault();
      mutate((q) => { q.yaw_mrad = wrap((q.yaw_mrad || 0) + Math.round((event.shiftKey ? -15 : 15) * MRAD_PER_DEG)); });
    } else if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      root.querySelector('[data-le="delete-item"]').click();
    }
  });

  /* ---------------- first load ---------------- */

  function renderAll() {
    renderCats();
    renderLibrary();
    renderLegend();
    renderSaved(S.srv ? S.srv.saved : []);
    renderProblems();
    render();
    updateRunning();
  }

  async function load() {
    if (S.loaded) return;
    S.loaded = true;
    try {
      const [lib, srv] = await Promise.all([api("/api/layout/library"), api("/api/layout")]);
      if (!lib.ok || !srv.ok) throw new Error("layout api");
      S.libList = lib.items;
      S.catList = lib.categories;
      for (const item of lib.items) S.lib.set(item.id, item);
      S.srv = srv;
      S.layout = clone(srv.layout);
      if (!S.layout.placements) S.layout.placements = [];
      S.base = JSON.stringify(S.layout.placements);
      els.snap.checked = S.snap;
      fitView();
      S.problems = (srv.check.problems || []).map(enrich);
      S.reach = srv.check.reach;
      renderAll();
      say("page.layout.status.idle", {}, "");
    } catch (error) {
      S.loaded = false;
      say("page.layout.error.load", {}, "warn");
    }
  }

  // the editor loads when the 資料 view opens (the library and the home are not needed before)
  const ready = window.I18N && window.I18N.ready ? window.I18N.ready : Promise.resolve();
  const open = () => {
    if (document.body.classList.contains("view-data")) ready.then(load, load);
  };
  window.addEventListener("duck:view", open);
  open();
  if (window.I18N && typeof window.I18N.onChange === "function") {
    window.I18N.onChange(() => {
      if (S.srv) renderAll();
    });
  }
  window.duckLayout = { state: S, reload: () => { S.loaded = false; return load(); } };
})();
