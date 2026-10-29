/* The home in the 3D view (phase 2 「真的家與真的感測器」, spec §12–§13).
 *
 * The room comes from the same layout the physics runs (the server's /api/home, or
 * /api/body's home_json: homelayout's page JSON, every element and toy in world millimetres),
 * plus the shapes the world adds to it — the visible hand, the family's bodies, the charging
 * place's tags, light and contacts. Everything is built in one Z-up group, so positions and
 * quaternions are MuJoCo's own (x, y on the floor, z up; quaternions [w, x, y, z]); three.js sees
 * it turned by −90° about x (three (x, y, z) = world (x, z, −y)).
 *
 * The charging place is a robot-vacuum-style dock against a wall (owner decision 2026-09-27;
 * world/dock.py): its housing, cyan face panel, ramped floor plate and V guides are layout
 * elements; its tags, status light and contacts (the forward style's floor pads, the backin
 * style's spring plates) come from `nest_parts` in the dock's own frame (origin at the middle
 * of its back on the wall, x into the room). Phase 2's open nest is still drawn for its variants.
 *
 * What moves (x_world through /api/body's `world`): the toys, the hand (while it holds the
 * duck, on the drawn duck's own trunk: the weld holds it there), the family's bodies, the
 * charging light, the backlight or the dock's status light, and which dock contacts are pressed.
 * Walls between the camera and the duck fade (owner decision B4). While the physics runs another
 * world (phase 0's flat floor) the room is drawn translucent: then it really is only on the page.
 *
 * No DOM here; duck3d.js owns the scene. */
import * as THREE from "three";

const PAGE_ONLY_OPACITY = 0.28;
const CONTACT_OPEN = "#c9a227"; // brass: a charging contact nothing presses
const CONTACT_ON = "#1f9d57"; // --green: pressed (the charger's PG)
const FADED_OPACITY = 0.14;
const FADE_KINDS = new Set(["wall", "railing", "gate"]);
const EASE_PER_S = 12;

// A MuJoCo quaternion [w, x, y, z] as three.js's.
function quat(q) {
  return new THREE.Quaternion(q[1], q[2], q[3], q[0]);
}

function m(mm) {
  return (mm || 0) / 1000;
}

function material(colour, opacity = 1, opts = {}) {
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(colour || "#cccccc"),
    roughness: opts.roughness ?? 0.85,
    metalness: opts.metalness ?? 0,
    transparent: opacity < 1,
    opacity,
  });
  if (opts.map) mat.map = opts.map;
  return mat;
}

export function woodTexture(renderer, repeat = [5, 5]) {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 512;
  const g = c.getContext("2d");
  const tones = ["#c9a47a", "#c29b70", "#cfac84", "#bd966b", "#c6a176"];
  const plank = 64;
  for (let row = 0; row < 512 / plank; row++) {
    let x = -((row * 97) % 300);
    while (x < 512) {
      const len = 180 + ((row * 53 + x) % 140);
      g.fillStyle = tones[(row * 3 + Math.abs(Math.round(x / 7))) % tones.length];
      g.fillRect(x, row * plank, len, plank);
      g.strokeStyle = "rgba(90,60,30,0.35)";
      g.lineWidth = 2;
      g.strokeRect(x + 1, row * plank + 1, len - 2, plank - 2);
      g.strokeStyle = "rgba(120,80,40,0.10)";
      for (let k = 0; k < 5; k++) {
        const y = row * plank + 8 + k * 11;
        g.beginPath();
        g.moveTo(x + 4, y);
        g.bezierCurveTo(x + len * 0.3, y + 3, x + len * 0.6, y - 3, x + len - 4, y + 1);
        g.stroke();
      }
      x += len;
    }
  }
  const texture = new THREE.CanvasTexture(c);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(repeat[0], repeat[1]);
  texture.colorSpace = THREE.SRGBColorSpace;
  if (renderer) texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return texture;
}

// A capsule (or cylinder) between two points (metres), in its parent's frame.
function between(geometryOf, from, to) {
  const a = new THREE.Vector3(...from);
  const b = new THREE.Vector3(...to);
  const axis = b.clone().sub(a);
  const length = axis.length();
  const mesh = new THREE.Mesh(geometryOf(Math.max(length, 1e-4)));
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  if (length > 1e-6) mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis.normalize());
  return mesh;
}

// One layout element: pos_mm is [x, y, its lowest z], size_mm its full extents.
function elementMesh(el, renderer) {
  const s = (el.size_mm || [0, 0, 0]).map(m);
  const [x, y, zb] = (el.pos_mm || [0, 0, 0]).map(m);
  const opacity = el.opacity ?? 1;
  const wood = el.texture === "wood" ? woodTexture(renderer, [Math.max(1, s[0] / 1.2), Math.max(1, s[1] / 1.2)]) : null;
  const mat = material(wood ? "#ffffff" : el.colour, opacity, { map: wood, roughness: el.kind === "floor" ? 0.8 : 0.85 });
  let mesh;
  let z = zb + s[2] / 2;
  switch (el.shape) {
    case "plane": {
      mesh = new THREE.Mesh(new THREE.PlaneGeometry(Math.max(s[0], 0.01), Math.max(s[1], 0.01)), mat);
      z = zb;
      break;
    }
    case "cylinder":
    case "frustum": {
      const top = m(el.radius_mm) || s[0] / 2;
      const bottom = el.shape === "frustum" ? m(el.radius_bottom_mm) || top : top;
      const geometry = new THREE.CylinderGeometry(top, bottom, Math.max(s[2], 0.001), 48);
      geometry.rotateX(Math.PI / 2); // three's y axis → MuJoCo's z (top up)
      mesh = new THREE.Mesh(geometry, mat);
      break;
    }
    case "sphere": {
      const r = m(el.radius_mm) || s[0] / 2;
      mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16), mat);
      z = zb + r;
      break;
    }
    case "slab": {
      // A low plate ramped on every side (the dock's floor plate): a box whose top face is
      // pulled in by the ramp on every side.
      const ramp = m(el.ramp_mm);
      const geometry = new THREE.BoxGeometry(Math.max(s[0], 0.001), Math.max(s[1], 0.001), Math.max(s[2], 0.001));
      const pos = geometry.attributes.position;
      const kx = Math.max(0, s[0] / 2 - ramp) / Math.max(s[0] / 2, 1e-6);
      const ky = Math.max(0, s[1] / 2 - ramp) / Math.max(s[1] / 2, 1e-6);
      for (let i = 0; i < pos.count; i++) {
        if (pos.getZ(i) > 0) pos.setXYZ(i, pos.getX(i) * kx, pos.getY(i) * ky, pos.getZ(i));
      }
      pos.needsUpdate = true;
      geometry.computeVertexNormals();
      mesh = new THREE.Mesh(geometry, mat);
      break;
    }
    case "ellipsoid": {
      mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), mat);
      mesh.scale.set(s[0] / 2, s[1] / 2, s[2] / 2);
      break;
    }
    case "capsule": {
      const r = m(el.radius_mm) || s[0] / 2;
      const f = (el.fromto_mm || [x * 1000, y * 1000, zb * 1000 + r * 1000, x * 1000, y * 1000, (zb + s[2]) * 1000 - r * 1000]).map(m);
      const piece = between((length) => new THREE.CapsuleGeometry(r, length, 6, 16), f.slice(0, 3), f.slice(3));
      piece.material = mat;
      piece.castShadow = piece.receiveShadow = true;
      return piece; // (already placed in the world frame)
    }
    default: {
      mesh = new THREE.Mesh(new THREE.BoxGeometry(Math.max(s[0], 0.001), Math.max(s[1], 0.001), Math.max(s[2], 0.001)), mat);
    }
  }
  mesh.position.set(x, y, z);
  mesh.rotation.z = (el.yaw_mrad || 0) / 1000;
  const flat = el.kind === "floor" || el.kind === "rug" || el.kind === "floor_decor" || el.kind === "mat" || el.kind === "dock_plate";
  mesh.castShadow = !flat && el.kind !== "window";
  mesh.receiveShadow = true;
  return mesh;
}

// The world's own primitives (hand, people): MuJoCo sizes (half extents, radii), in their
// body's frame, millimetres.
function primMesh(p) {
  const s = (p.size_mm || [0]).map(m);
  const mat = material(p.colour, 1, { roughness: 0.7 });
  let mesh;
  if (p.fromto_mm) {
    const f = p.fromto_mm.map(m);
    const r = s[0];
    mesh = between((length) => (p.shape === "cylinder" ? new THREE.CylinderGeometry(r, r, length, 16) : new THREE.CapsuleGeometry(r, length, 6, 12)), f.slice(0, 3), f.slice(3));
    mesh.material = mat;
  } else {
    if (p.shape === "sphere") mesh = new THREE.Mesh(new THREE.SphereGeometry(s[0], 20, 14), mat);
    else if (p.shape === "ellipsoid") {
      mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), mat);
      mesh.scale.set(s[0], s[1] ?? s[0], s[2] ?? s[0]);
    } else if (p.shape === "capsule") mesh = new THREE.Mesh(new THREE.CapsuleGeometry(s[0], 2 * (s[1] || 0), 6, 12), mat).rotateX(Math.PI / 2);
    else if (p.shape === "cylinder") mesh = new THREE.Mesh(new THREE.CylinderGeometry(s[0], s[0], 2 * (s[1] || 0.01), 16).rotateX(Math.PI / 2), mat);
    else mesh = new THREE.Mesh(new THREE.BoxGeometry(2 * s[0], 2 * (s[1] ?? s[0]), 2 * (s[2] ?? s[0])), mat);
    const pos = (p.pos_mm || [0, 0, 0]).map(m);
    mesh.position.set(pos[0], pos[1], pos[2]);
  }
  mesh.castShadow = true;
  return mesh;
}

// A toy: its parts from the thing's frame (its lowest point), part offsets at the part's own
// bottom centre, sizes as full extents.
function thingGroup(thing) {
  const group = new THREE.Group();
  const opacity = thing.opacity ?? 1;
  for (const part of thing.parts || []) {
    const s = (part.size_mm || [0, 0, 0]).map(m);
    const [ox, oy, oz] = (part.offset_mm || [0, 0, 0]).map(m);
    const mat = material(thing.colour, opacity, { roughness: 0.55 });
    let mesh;
    if (part.shape === "sphere") {
      const r = m(part.radius_mm) || s[0] / 2;
      mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16), mat);
      mesh.position.set(ox, oy, oz + r);
    } else if (part.shape === "cylinder") {
      const r = m(part.radius_mm) || s[0] / 2;
      mesh = new THREE.Mesh(new THREE.CylinderGeometry(r, r, s[2], 24).rotateX(Math.PI / 2), mat);
      mesh.position.set(ox, oy, oz + s[2] / 2);
    } else {
      mesh = new THREE.Mesh(new THREE.BoxGeometry(s[0], s[1], s[2]), mat);
      mesh.position.set(ox, oy, oz + s[2] / 2);
    }
    mesh.castShadow = mesh.receiveShadow = true;
    group.add(mesh);
  }
  const [x, y, z] = (thing.pos_mm || [0, 0, 0]).map(m);
  group.position.set(x, y, z);
  group.rotation.z = (thing.yaw_mrad || 0) / 1000;
  group.userData.thing = thing.id;
  return group;
}

// An AprilTag's cells (the same bits as the physics' texture) on a canvas.
function tagTexture(cells) {
  const n = cells.length || 1;
  const px = 8;
  const c = document.createElement("canvas");
  c.width = c.height = n * px;
  const g = c.getContext("2d");
  cells.forEach((row, y) =>
    row.forEach((black, x) => {
      g.fillStyle = black ? "#101018" : "#ffffff";
      g.fillRect(x * px, y * px, px, px);
    }),
  );
  const texture = new THREE.CanvasTexture(c);
  texture.magFilter = THREE.NearestFilter;
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

// A point of the nest's own frame (x toward the room, z up) in the world.
function nestPoint(nest, p) {
  const yaw = (nest.heading_mrad || 0) / 1000;
  const [lx, ly, lz] = p.map(m);
  return [m(nest.x_mm) + lx * Math.cos(yaw) - ly * Math.sin(yaw), m(nest.y_mm) + lx * Math.sin(yaw) + ly * Math.cos(yaw), m(nest.z_mm) + lz];
}

// A box between two corners of the nest's own frame (mm), turned with the nest.
function nestBox(nest, min, max, mat) {
  const size = [0, 1, 2].map((i) => Math.max(m(max[i] - min[i]), 0.001));
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(size[0], size[1], size[2]), mat);
  const [x, y, z] = nestPoint(nest, [0, 1, 2].map((i) => (min[i] + max[i]) / 2));
  mesh.position.set(x, y, z);
  mesh.rotation.z = (nest.heading_mrad || 0) / 1000;
  return mesh;
}

function buildNestParts(zup, layout, parts, meshes) {
  const nest = layout.nest;
  if (!nest || !parts || !parts.present) return null;
  const yaw = (nest.heading_mrad || 0) / 1000;
  const out = { group: new THREE.Group(), backlight: null, kind: parts.kind || "open", contacts: [] };
  zup.add(out.group);
  for (const tag of parts.tags || []) {
    const size = m(tag.size_mm);
    const mount = m(tag.mount_mm || tag.size_mm);
    const holder = new THREE.Group();
    const white = new THREE.Mesh(new THREE.PlaneGeometry(mount, mount), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    const face = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshBasicMaterial({ map: tagTexture(tag.cells || []) }));
    face.position.z = 0.0004;
    holder.add(white, face);
    const [x, y, z] = nestPoint(nest, tag.centre_mm || [0, 0, 0]);
    holder.position.set(x, y, z);
    if (tag.where === "board") {
      // Upright, facing the room (+x of the nest): the plane's +z turned to the nest's +x.
      holder.rotation.set(Math.PI / 2, 0, yaw + Math.PI / 2, "ZXY");
      holder.position.x += 0.0015 * Math.cos(yaw);
      holder.position.y += 0.0015 * Math.sin(yaw);
    } else {
      holder.rotation.z = yaw + Math.PI / 2; // lying flat, its top toward the board
      holder.position.z += 0.0008;
    }
    out.group.add(holder);
    meshes.push(white, face);
  }
  // The dock's status light (C4: the backlight's meaning) and its contacts, pressed or not.
  if (parts.light) {
    const [w, h] = (parts.light.size_mm || [160, 10]).map(m);
    const glow = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: 0xe8f6f7 }));
    const [x, y, z] = nestPoint(nest, parts.light.centre_mm || [110, 0, 183]);
    glow.position.set(x, y, z);
    glow.rotation.set(Math.PI / 2, 0, yaw + Math.PI / 2, "ZXY");
    glow.visible = false;
    out.group.add(glow);
    out.backlight = glow;
  }
  for (const contact of parts.contacts || []) {
    const pad = nestBox(nest, contact.min_mm, contact.max_mm, material(CONTACT_OPEN, 1, { roughness: 0.5, metalness: 0.25 }));
    pad.userData.contact = contact.id;
    pad.receiveShadow = true;
    out.group.add(pad);
    out.contacts.push(pad);
    meshes.push(pad);
  }
  if (parts.backlight) {
    const [w, h] = (parts.backlight.size_mm || [200, 190]).map(m);
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: 0xe8f6f7, transparent: true, opacity: 0.95 }));
    const [x, y, z] = nestPoint(nest, parts.backlight.centre_mm || [-400, 0, 120]);
    panel.position.set(x + 0.0008 * Math.cos(yaw), y + 0.0008 * Math.sin(yaw), z);
    panel.rotation.set(Math.PI / 2, 0, yaw + Math.PI / 2, "ZXY");
    panel.visible = false;
    out.group.add(panel);
    out.backlight = panel;
  }
  if (parts.electrode) {
    const disc = new THREE.Mesh(new THREE.CircleGeometry(m(parts.electrode.radius_mm), 64), material("#c9a227", 1, { roughness: 0.35, metalness: 0.8 }));
    disc.position.set(m(nest.x_mm), m(nest.y_mm), m(parts.electrode.top_mm) + 0.0003);
    disc.receiveShadow = true;
    out.group.add(disc);
    meshes.push(disc);
  }
  return out;
}

/** Build the home into `scene` (see the file comment). `payload`: /api/home. Returns what
 * updateWorld / fadeWalls / setGhost / disposeHome need. */
export function buildHome(scene, payload, renderer) {
  const layout = payload && payload.layout;
  if (!layout) return null;
  const zup = new THREE.Group();
  zup.rotation.x = -Math.PI / 2;
  zup.name = "home";
  scene.add(zup);
  const home = {
    zup,
    fingerprint: payload.fingerprint,
    name: payload.name,
    layout,
    meshes: [], // every layout mesh (ghosted together)
    floors: [], // never ghosted: the physics always has a floor
    walls: [], // {mesh, box} faded between the camera and the duck (B4)
    things: new Map(),
    thingTargets: new Map(),
    hand: null,
    handTarget: null,
    people: new Map(),
    bodyTargets: new Map(),
    nest: null,
    led: null,
    ghost: false,
    frame: layout.frame || { x_mm: 0, y_mm: 0, heading_mrad: 0 },
    bounds: layout.bounds_mm || [-3000, -3000, 3000, 3000],
    lamps: ((layout.lights && layout.lights.lamps) || []).map((lamp) => {
      const [x, y, z] = lamp.pos_mm.map(m);
      return { id: lamp.id, three: new THREE.Vector3(x, z, -y), hours: lamp.hours };
    }),
    aids: [],
  };
  for (const el of layout.elements || []) {
    if (el.flags && el.flags.page === false) continue;
    const mesh = elementMesh(el, renderer);
    mesh.name = el.geom || el.id;
    mesh.userData.element = el.id;
    mesh.userData.kind = el.kind;
    zup.add(mesh);
    if (el.kind === "floor") home.floors.push(mesh);
    else home.meshes.push(mesh);
    if (el.kind === "nest_led") home.led = mesh;
    if (FADE_KINDS.has(el.kind)) home.walls.push({ mesh, box: null, opacity: 1 });
  }
  home.nest = buildNestParts(zup, layout, payload.nest_parts, home.meshes);
  for (const thing of layout.things || []) {
    const group = thingGroup(thing);
    zup.add(group);
    home.things.set(thing.id, group);
    group.traverse((child) => child.isMesh && home.meshes.push(child));
  }
  // The world's hand (hidden until it is shown) and the family's bodies (hidden until they come).
  if (payload.hand && payload.hand.parts) {
    const hand = new THREE.Group();
    for (const part of payload.hand.parts) hand.add(primMesh(part));
    hand.visible = false;
    zup.add(hand);
    home.hand = hand;
  }
  for (const [who, person] of Object.entries(payload.people || {})) {
    for (const [body, prims] of Object.entries(person.bodies || {})) {
      const group = new THREE.Group();
      for (const part of prims) group.add(primMesh(part));
      group.visible = false;
      group.userData.who = who;
      zup.add(group);
      home.people.set(body, group);
    }
  }
  // Page-only aids (never in the physics): 50 cm dots along Duck Core's +x from its origin.
  const aids = payload.aids || {};
  const frame = home.frame;
  const fyaw = (frame.heading_mrad || 0) / 1000;
  for (const [ax, ay] of aids.floor_dots_mm || []) {
    const dot = new THREE.Mesh(new THREE.CircleGeometry(0.012, 16), new THREE.MeshBasicMaterial({ color: 0x1f6f6b }));
    dot.position.set(m(frame.x_mm) + m(ax) * Math.cos(fyaw) - m(ay) * Math.sin(fyaw), m(frame.y_mm) + m(ax) * Math.sin(fyaw) + m(ay) * Math.cos(fyaw), 0.003);
    zup.add(dot);
    home.aids.push(dot);
  }
  zup.updateMatrixWorld(true);
  for (const wall of home.walls) wall.box = new THREE.Box3().setFromObject(wall.mesh);
  return home;
}

export function disposeHome(scene, home) {
  if (!home) return;
  scene.remove(home.zup);
  home.zup.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    const mats = Array.isArray(child.material) ? child.material : child.material ? [child.material] : [];
    for (const mat of mats) {
      if (mat.map) mat.map.dispose();
      mat.dispose();
    }
  });
}

/** A Duck Core (plan) pose [x, y, yaw] in the world, by the home's frame (its nest or mat). */
export function planToWorld(frame, x, y, yaw) {
  const f = frame || { x_mm: 0, y_mm: 0, heading_mrad: 0 };
  const fy = (f.heading_mrad || 0) / 1000;
  const c = Math.cos(fy);
  const s = Math.sin(fy);
  return [m(f.x_mm) + c * x - s * y, m(f.y_mm) + s * x + c * y, yaw + fy];
}

/** The world pose [x, y] in Duck Core's frame (distances to its nest or mat). */
export function worldToPlan(frame, x, y) {
  const f = frame || { x_mm: 0, y_mm: 0, heading_mrad: 0 };
  const fy = (f.heading_mrad || 0) / 1000;
  const dx = x - m(f.x_mm);
  const dy = y - m(f.y_mm);
  return [Math.cos(fy) * dx + Math.sin(fy) * dy, -Math.sin(fy) * dx + Math.cos(fy) * dy];
}

function ease(object, target, k) {
  if (!target) return;
  if (!object.userData.placed || object.position.distanceTo(target.pos) > 0.6) {
    object.position.copy(target.pos);
    object.quaternion.copy(target.quat);
    object.userData.placed = true;
    return;
  }
  object.position.lerp(target.pos, k);
  object.quaternion.slerp(target.quat, k);
}

function pose(pos_mm, q) {
  if (!pos_mm) return null;
  return { pos: new THREE.Vector3(m(pos_mm[0]), m(pos_mm[1]), m(pos_mm[2])), quat: q ? quat(q) : new THREE.Quaternion() };
}

/** The world's truth for the page (/api/body `world`): where the toys, the hand and the
 * family's bodies are now, the nest's charging light and backlight. `duck`: the drawn solid
 * duck's trunk {trunk: [x, y, z], quat} (the hand holding it is drawn on it). `charging`:
 * Duck Core's own idea, used when the world does not say. */
export function setWorld(home, world, charging) {
  if (!home) return;
  const w = world || {};
  for (const item of w.things || []) {
    if (home.things.has(item.id)) home.thingTargets.set(item.id, pose(item.pos_mm, item.quat));
  }
  const hand = w.hand || null;
  home.handShown = !!(hand && hand.shown && hand.pos_mm);
  home.handHolding = !!(hand && hand.holding);
  home.handTarget = home.handShown ? pose(hand.pos_mm, hand.quat) : null;
  const bodies = w.bodies || {};
  for (const [name, group] of home.people) {
    const b = bodies[name];
    const shown = !!(b && b.pos_mm && b.pos_mm[2] > -1000);
    if (!shown) {
      group.visible = false;
      group.userData.placed = false;
      home.bodyTargets.delete(name);
    } else home.bodyTargets.set(name, pose(b.pos_mm, b.quat));
  }
  const nest = w.nest || null;
  const on = nest && nest.present ? !!nest.charging : !!charging;
  const dock = !!(home.nest && home.nest.kind === "dock");
  if (home.led && !dock) {
    home.led.material.color.set(on ? "#1f9d57" : "#3a3a44");
    home.led.material.emissive.set(on ? "#1f9d57" : "#000000");
  }
  const glow = w.backlight || null;
  if (home.nest && home.nest.backlight) {
    home.nest.backlight.visible = !!(glow && glow.on);
    if (glow && glow.color) home.nest.backlight.material.color.set(glow.color);
  }
  if (dock) {
    // Which contacts are pressed: the forward pads one by one (a thigh on each), the backin
    // plates together (one probe). Without the world, Duck Core's own idea: all or none.
    const pressed = new Set();
    const probes = (nest && nest.probes) || {};
    for (const probe of Object.values(probes)) {
      if (!probe || !(probe.pressed_mm >= 0)) continue;
      if (probe.pad) pressed.add(probe.pad);
      else if (probe.on_plate) ["l", "r"].forEach((id) => pressed.add(id));
    }
    const all = nest && nest.present ? !!nest.contact : !!charging;
    for (const pad of home.nest.contacts) {
      const lit = all || pressed.has(pad.userData.contact);
      pad.material.color.set(lit ? CONTACT_ON : CONTACT_OPEN);
      pad.material.emissive.set(lit ? CONTACT_ON : "#000000");
      pad.material.emissiveIntensity = lit ? 0.35 : 0;
    }
  }
}

/** Ease everything that moves toward the world's latest truth (called every frame). */
export function stepWorld(home, dt, duck) {
  if (!home) return;
  const k = Math.min(1, dt * EASE_PER_S);
  for (const [id, target] of home.thingTargets) {
    const group = home.things.get(id);
    if (group && target) ease(group, target, k);
  }
  if (home.hand) {
    home.hand.visible = !!home.handShown;
    if (home.handShown && home.handHolding && duck && duck.trunk) {
      // Held by the weld, the hand's frame is the trunk's: drawn on the drawn duck itself.
      home.hand.position.set(duck.trunk[0], duck.trunk[1], duck.trunk[2]);
      home.hand.quaternion.copy(quat(duck.quat));
      home.hand.userData.placed = true;
    } else if (home.handTarget) ease(home.hand, home.handTarget, k);
  }
  for (const [name, target] of home.bodyTargets) {
    const group = home.people.get(name);
    if (!group || !target) continue;
    group.visible = true;
    ease(group, target, k);
  }
}

/** B4: walls, railings and gates between the camera and the duck fade (`target`: the duck, in
 * three.js coordinates). */
export function fadeWalls(home, camera, target, dt) {
  if (!home || !camera || !target) return;
  const from = camera.position;
  const dir = target.clone().sub(from);
  const far = dir.length();
  if (far < 1e-6) return;
  const ray = new THREE.Ray(from.clone(), dir.normalize());
  const hit = new THREE.Vector3();
  const k = Math.min(1, dt * 6);
  for (const wall of home.walls) {
    if (!wall.box) continue;
    const point = ray.intersectBox(wall.box, hit);
    const between = !!point && from.distanceTo(point) < far - 0.08 && !wall.box.containsPoint(target);
    const want = between ? FADED_OPACITY : home.ghost ? PAGE_ONLY_OPACITY : 1;
    wall.opacity += (want - wall.opacity) * k;
    const mat = wall.mesh.material;
    const faded = wall.opacity < 0.985;
    if (mat.transparent !== faded) {
      mat.transparent = faded;
      mat.needsUpdate = true;
    }
    mat.opacity = faded ? wall.opacity : 1;
    mat.depthWrite = !faded;
    wall.mesh.castShadow = !between;
    wall.faded = between;
  }
}

/** Translucent when the physics does not have this home (phase 0's flat floor): then the
 * room really is only on the page. The floor stays: the physics always has one. */
export function setGhost(home, ghost) {
  if (!home || home.ghost === ghost) return;
  home.ghost = ghost;
  for (const mesh of home.meshes) {
    const mat = mesh.material;
    if (mesh.userData.opacity == null) {
      mesh.userData.opacity = mat.opacity;
      mesh.userData.transparent = mat.transparent;
      mesh.userData.castShadow = mesh.castShadow;
    }
    if (home.walls.some((w) => w.mesh === mesh)) continue; // (fadeWalls owns their opacity)
    mat.transparent = ghost || mesh.userData.transparent;
    mat.opacity = ghost ? Math.min(mesh.userData.opacity, PAGE_ONLY_OPACITY) : mesh.userData.opacity;
    mat.depthWrite = !ghost;
    mat.needsUpdate = true;
    mesh.castShadow = ghost ? false : mesh.userData.castShadow;
  }
}

/** How far the whole home reaches (metres), for the camera and the fog. */
export function homeSpan(home) {
  const [x0, y0, x1, y1] = (home && home.bounds) || [-3000, -3000, 3000, 3000];
  return Math.hypot(m(x1 - x0), m(y1 - y0));
}
