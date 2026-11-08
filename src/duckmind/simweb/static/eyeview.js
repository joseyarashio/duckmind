/* The duck's-eye window's shape (camera beta, 2026-11-09): the picture is as tall or as wide as the head
 * camera the simulated duck carries — the alpha board's quarter-turned IMX219 gives a portrait picture
 * (360×640), the production GC2093 a landscape one (640×360). /api/body's eye answer says the picture's
 * width and height (the latest picture's own, else what this camera gives); app.js shapes the window by
 * it, so neither picture is squeezed. An older server says nothing: the alpha's portrait.
 * Words: none here. */
"use strict";

(function eyeViewModule() {
  const PORTRAIT = { width: 360, height: 640 };

  /* The picture's shape from one eye answer: {width, height, aspect ("w / h", for CSS), landscape}. */
  function eyeShape(eye) {
    const width = Number(eye && eye.width);
    const height = Number(eye && eye.height);
    const known = Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0;
    const w = known ? Math.round(width) : PORTRAIT.width;
    const h = known ? Math.round(height) : PORTRAIT.height;
    return { width: w, height: h, aspect: `${w} / ${h}`, landscape: w > h };
  }

  const api = { eyeShape };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else window.duckEyeView = api;
})();
