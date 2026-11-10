// Runs static/home3d.js in Node with the page's own three.js (the one index.html's import map names) on a home
// the server really makes (tests/test_page_truth_source.py writes it), with no renderer and no DOM but a canvas
// that remembers what was painted on it.
//   node home3d_node_harness.mjs <static dir> <input json>
// input: {payload: /api/home's answer, looks: [{camera: [x, y, z], duck: [x, y, z]}] (world metres), world: {...}, trunk: {...}}
// Prints one JSON value: what was built, what each look faded, where the world's things were drawn.
import fs from "node:fs";
import path from "node:path";
import { register } from "node:module";
import { pathToFileURL } from "node:url";

const [staticDir, inputPath] = process.argv.slice(2);
const input = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const fileUrl = (rel) => pathToFileURL(path.join(staticDir, rel)).href;

// "three" is found the way the page finds it: by its import map.
const page = fs.readFileSync(path.join(staticDir, "index.html"), "utf8");
const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(page)[1]).imports;
const mapped = Object.fromEntries(Object.entries(imports).map(([name, url]) => [name, fileUrl(url)]));
register(
  "data:text/javascript," +
    encodeURIComponent(
      `const mapped = ${JSON.stringify(mapped)};
       export async function resolve(specifier, context, next) {
         for (const [name, url] of Object.entries(mapped)) {
           if (specifier === name) return { url, shortCircuit: true };
           if (name.endsWith("/") && specifier.startsWith(name)) return { url: url + specifier.slice(name.length), shortCircuit: true };
         }
         return next(specifier, context);
       }`,
    ),
);

// A canvas that keeps its filled rectangles (the AprilTag's cells are painted on one).
const canvases = [];
globalThis.document = {
  createElement(tag) {
    const canvas = { tag, width: 0, height: 0, rects: [] };
    const brush = { fillStyle: "" };
    canvas.getContext = () =>
      new Proxy(brush, {
        get(target, name) {
          if (name === "fillRect") return (x, y, w, h) => canvas.rects.push({ x, y, w, h, fill: target.fillStyle });
          return name in target ? target[name] : () => {};
        },
        set(target, name, value) {
          target[name] = value;
          return true;
        },
      });
    canvases.push(canvas);
    return canvas;
  },
};

const THREE = await import("three");
const H = await import(fileUrl("home3d.js"));

const round = (v) => Math.round(v * 1e4) / 10; // metres -> mm, one decimal
// three.js sees the home turned by −90° about x: three (x, y, z) = world (x, z, −y)
const toThree = ([x, y, z]) => new THREE.Vector3(x, z, -y);
const worldBox = (object) => {
  const b = new THREE.Box3().setFromObject(object);
  return [b.min.x, -b.max.z, b.min.y, b.max.x, -b.min.z, b.max.y].map(round);
};
const worldAt = (object) => {
  const p = object.getWorldPosition(new THREE.Vector3());
  return [p.x, -p.z, p.y].map(round);
};

const scene = new THREE.Scene();
const home = H.buildHome(scene, input.payload, null);
scene.updateMatrixWorld(true);

const out = {};
out.elements = home.zup.children
  .filter((child) => child.userData.element)
  .map((mesh) => ({ id: mesh.userData.element, kind: mesh.userData.kind, box: worldBox(mesh), opacity: mesh.material.opacity }));
out.things = [...home.things].map(([id, group]) => ({ id, at: worldAt(group), parts: group.children.length }));
out.hand = home.hand ? { parts: home.hand.children.length, visible: home.hand.visible } : null;
out.people = [...home.people].map(([body, group]) => ({ body, who: group.userData.who, visible: group.visible }));
// every tag's picture, read back from the canvas as rows of cells (1 = black)
out.tags = canvases
  .filter((canvas) => canvas.rects.length && canvas.rects.every((r) => r.w === r.h))
  .map((canvas) => {
    const px = canvas.rects[0].w;
    const cells = Array.from({ length: canvas.height / px }, () => Array(canvas.width / px).fill(null));
    for (const r of canvas.rects) cells[r.y / px][r.x / px] = r.fill === "#ffffff" ? 0 : 1;
    return cells;
  });
out.contacts = home.nest ? home.nest.contacts.map((pad) => pad.userData.contact) : [];
out.frame = home.frame;

// B4: each look is held for a second of frames; which walls are faded then
out.looks = [];
for (const look of input.looks || []) {
  const camera = { position: toThree(look.camera) };
  for (let i = 0; i < 60; i++) H.fadeWalls(home, camera, toThree(look.duck), 1 / 60);
  out.looks.push(
    Object.fromEntries(home.walls.map((wall) => [wall.mesh.userData.element, { faded: !!wall.faded, opacity: wall.mesh.material.opacity, transparent: wall.mesh.material.transparent }])),
  );
}

// the world's truth: the toys, the family and the hand go where it says; a hand that holds the duck is on the drawn trunk
const settle = (world, trunk) => {
  H.setWorld(home, world, false);
  for (let i = 0; i < 60; i++) H.stepWorld(home, 1 / 60, trunk);
  scene.updateMatrixWorld(true);
  return {
    things: Object.fromEntries((world.things || []).map((item) => [item.id, worldAt(home.things.get(item.id))])),
    hand: home.hand ? { visible: home.hand.visible, at: worldAt(home.hand) } : null,
    people: Object.fromEntries([...home.people].map(([body, group]) => [body, group.visible ? worldAt(group) : null])),
  };
};
if (input.world) {
  out.held = settle(input.world, input.trunk);
  out.released = settle({ ...input.world, hand: { ...input.world.hand, holding: false } }, input.trunk);
  out.gone = settle({}, input.trunk);
}

// a home the physics does not run is drawn translucent, except its floor
H.setGhost(home, true);
out.ghost = Object.fromEntries(home.zup.children.filter((child) => child.userData.element).map((mesh) => [mesh.userData.element, mesh.material.opacity]));
H.setGhost(home, false);
out.solid = Object.fromEntries(home.zup.children.filter((child) => child.userData.element).map((mesh) => [mesh.userData.element, mesh.material.opacity]));

console.log(JSON.stringify(out));
