import { fileURLToPath } from "node:url";
import { cloudflare } from "@cloudflare/vite-plugin";
import { defineConfig } from "vite";

// The Worker's static assets are the simulator's own vendored fonts (no copy in this folder);
// the Worker serves them under /_edge/fonts/ (assets.runWorkerFirst).
const fonts = fileURLToPath(new URL("../../../src/duckmind/simweb/static/fonts", import.meta.url));

export default defineConfig({
	publicDir: fonts,
	server: { host: "127.0.0.1", port: 8789, strictPort: true },
	plugins: [cloudflare()],
});
