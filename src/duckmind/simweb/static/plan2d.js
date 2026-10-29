/* The home seen from above, in SVG (shared by the layout editor on the 資料 page and the minimap on the view).
 *
 * Both draw the same page layout the server sends (homelayout's page JSON: every wall, piece of furniture and
 * toy in world millimetres), so there is one drawing of "the fixed home" and not two. World mm, y up; an SVG
 * has y down, so every y is written negated and a viewBox of `x, -(y + h), w, h` shows the plan the right way
 * up. Only the geometry lives here; the colours and strokes are the caller's CSS (the classes it passes in).
 * Nothing here decides anything for the duck. */
"use strict";

(function planTwoD() {
  const SVGNS = "http://www.w3.org/2000/svg";

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

  // (a rectangle w x d centred on (x, y) and turned by yaw, as an SVG polygon's points)
  function boxPoints(x, y, w, d, yaw, ox = 0, oy = 0) {
    return [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]]
      .map(([a, b]) => {
        const [rx, ry] = rot(a + ox, b + oy, yaw);
        return `${x + rx},${-(y + ry)}`;
      })
      .join(" ");
  }

  // What lies on the floor is drawn first, then what stands on it, then the walls.
  const KIND_LAYER = { rug: 0, mat: 0, threshold: 1, cable: 2, door: 4, window: 5, wall: 6 };
  const layerOf = (kind) => (kind in KIND_LAYER ? KIND_LAYER[kind] : 3);

  /** Draw the fixed pieces of a home into the SVG group `group`.
   *  fixed: {elements: [...], things: [...]} (the page layout's elements and toys).
   *  opts.cls: the class every piece gets ("le-fixed"); opts.kindClass: the prefix of its kind class ("le-k-").
   *  opts.minLine: thinnest line (mm) a cable is drawn with. opts.sort: draw floor things first, walls last.
   *  opts.wrapThings: put each toy's parts in a <g data-thing=id> and return those groups by id. */
  function drawFixed(group, fixed, opts = {}) {
    const base = opts.cls || "plan-fixed";
    const kindClass = opts.kindClass || "plan-k-";
    const minLine = opts.minLine || 12;
    const things = new Map();
    let elements = fixed.elements || [];
    if (opts.sort) elements = [...elements].sort((a, b) => layerOf(a.kind) - layerOf(b.kind));
    for (const e of elements) {
      if (e.kind === "floor" || e.kind === "ceiling" || e.shape === "plane") continue;
      if (e.kind === "skirting") continue;
      if (opts.skipKinds && opts.skipKinds.includes(e.kind)) continue;
      const cls = `${base} ${kindClass}${e.kind}`;
      const attrs = { class: cls, fill: e.colour, "data-fixed": e.id };
      if (!e.flags.collide) attrs["stroke-dasharray"] = "40 30";
      const [x, y] = e.pos_mm;
      if (e.shape === "box" || e.shape === "slab" || e.shape === "ellipsoid") {
        const [w, d] = e.size_mm;
        svg("polygon", { ...attrs, points: boxPoints(x, y, w, d, e.yaw_mrad || 0) }, group);
      } else if (e.shape === "capsule") {
        const [ax, ay, , bx, by] = e.fromto_mm;
        svg("line", { x1: ax, y1: -ay, x2: bx, y2: -by, style: `stroke:${e.colour};stroke-width:${Math.max(e.radius_mm * 2, minLine)}px`, class: cls, "data-fixed": e.id }, group);
      } else {
        svg("circle", { ...attrs, cx: x, cy: -y, r: Math.max(e.radius_mm || 0, e.radius_bottom_mm || 0) }, group);
      }
    }
    for (const th of fixed.things || []) {
      const [x, y] = th.pos_mm;
      const holder = opts.wrapThings ? svg("g", { "data-thing": th.id }, group) : group;
      if (opts.wrapThings) things.set(th.id, { node: holder, x, y });
      for (const part of th.parts) {
        const yaw = th.yaw_mrad || 0;
        const [rx, ry] = rot(part.offset_mm[0], part.offset_mm[1], yaw);
        const cls = `${base} ${kindClass}thing`;
        if (part.shape === "box") {
          const [w, d] = part.size_mm;
          svg("polygon", { points: boxPoints(x + rx, y + ry, w, d, yaw), fill: th.colour, class: cls, "data-fixed": th.id }, holder);
        } else {
          svg("circle", { cx: x + rx, cy: -(y + ry), r: part.radius_mm, fill: th.colour, class: cls, "data-fixed": th.id }, holder);
        }
      }
    }
    return things;
  }

  window.DuckPlan = { SVGNS, svg, rot, boxPoints, drawFixed };
})();
