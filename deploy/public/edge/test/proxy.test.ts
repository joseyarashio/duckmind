import assert from "node:assert/strict";
import { test } from "node:test";
import { allowedPath } from "../src/core/proxy.ts";

test("the page, its static files and the public API pass", () => {
	for (const p of [
		"/",
		"/index.html",
		"/app.js",
		"/style.css",
		"/i18n.en.page.js",
		"/fonts/fonts.css",
		"/fonts/notosanstc/notosanstc-12.woff2",
		"/vendor/d3/d3.min.js",
		"/mesh/body.stl",
		"/api/ping",
		"/api/public/session",
		"/api/life/duckfile/export",
		"/api/life/fastforward",
		"/api/neuro",
		"/api/neuro/stream",
		"/api/voice/manifest.json",
	]) {
		assert.equal(allowedPath("GET", p), true, p);
	}
	for (const p of ["/api/call", "/api/neuro/command", "/api/neuro/sound/world"]) {
		assert.equal(allowedPath("POST", p), true, p);
	}
});

test("everything else stays at the edge", () => {
	const refused: Array<[string, string]> = [
		["GET", "/lab"],
		["GET", "/data"],
		["GET", "/api/layout"],
		["GET", "/api/lab/anything"],
		["GET", "/api/neuro/remote/ws"],
		["GET", "/api/life/duckfile/import"],
		["POST", "/api/life/duckfile/import"],
		["POST", "/api/layout"],
		["POST", "/api/ping"],
		["POST", "/api/life/fastforward"],
		["GET", "/favicon.ico"],
		["GET", "/secret.py"],
		["GET", "/.env"],
		["GET", "/fonts/../server.py"],
		["GET", "/fonts/%2e%2e/server.py"],
		["GET", "/fonts/..%2fserver.py"],
		["GET", "/fonts\\..\\x"],
		["GET", "/fonts//x"],
		["GET", "/fonts/"],
		["GET", "/mesh/a/b.stl"],
		["GET", "/vendor/.hidden"],
		["GET", "/_edge/status"],
		["PUT", "/api/call"],
		["DELETE", "/"],
		["HEAD", "/"],
		["OPTIONS", "/api/call"],
		["GET", "/" + "a".repeat(3000) + ".js"],
	];
	for (const [m, p] of refused) assert.equal(allowedPath(m, p), false, `${m} ${p}`);
});
