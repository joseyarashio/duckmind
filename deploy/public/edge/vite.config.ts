import { cpSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { cloudflare } from "@cloudflare/vite-plugin";
import { defineConfig, type Plugin } from "vite";

// The Worker's static assets, which the Worker serves under /_edge/ (assets.runWorkerFirst):
//   img/    the pages' pictures: this folder's static/img (the public dir)
//   fonts/  the simulator's own vendored fonts, taken from where they live (no copy in this folder)
const STATIC = fileURLToPath(new URL("./static", import.meta.url));
const FONTS = fileURLToPath(new URL("../../../src/duckmind/simweb/static/fonts", import.meta.url));

const TYPES: Record<string, string> = {
	".css": "text/css; charset=utf-8",
	".woff2": "font/woff2",
	".txt": "text/plain; charset=utf-8",
	".webp": "image/webp",
};

/** fonts/ copied into the client build. `cf dev` answers /_edge/fonts/ and /_edge/img/ from the
 * folders itself, ahead of the Worker (whose ASSETS binding finds no files in development). */
function edgeFiles(): Plugin {
	const folders: Record<string, string> = { "/_edge/fonts": FONTS, "/_edge/img": join(STATIC, "img") };
	return {
		name: "duckmind-edge-files",
		configureServer(server) {
			for (const [prefix, dir] of Object.entries(folders)) {
				server.middlewares.use(prefix, (req, res, next) => {
					const file = join(dir, normalize(decodeURIComponent((req.url ?? "/").split("?")[0]!)));
					const type = TYPES[extname(file)];
					if (!type || !file.startsWith(dir + "/") || !statSync(file, { throwIfNoEntry: false })?.isFile()) return next();
					res.setHeader("content-type", type);
					res.end(readFileSync(file));
				});
			}
		},
		writeBundle(options) {
			if (this.environment.name === "client" && options.dir) cpSync(FONTS, join(options.dir, "fonts"), { recursive: true });
		},
	};
}

export default defineConfig({
	publicDir: STATIC,
	server: { host: "127.0.0.1", port: 8789, strictPort: true },
	// edgeFiles first: its development middleware answers before the Cloudflare plugin hands requests to the Worker.
	plugins: [edgeFiles(), cloudflare()],
});
