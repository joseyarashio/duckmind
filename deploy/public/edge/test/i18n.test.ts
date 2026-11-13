import assert from "node:assert/strict";
import { test } from "node:test";
import { CATALOGS, LANGS, pickLang, t } from "../src/core/i18n.ts";
import { esc, guidingPage, landingPage, QUICK_START_URL, waitingPage } from "../src/core/pages.ts";

const values = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();

test("every word exists in all six languages with the same values", () => {
	const source = CATALOGS["zh-Hant"];
	for (const lang of LANGS) {
		const cat = CATALOGS[lang];
		assert.deepEqual(Object.keys(cat).sort(), Object.keys(source).sort(), lang);
		for (const [k, v] of Object.entries(source)) {
			assert.ok(cat[k]!.trim().length > 0, `${lang} ${k}`);
			assert.deepEqual(values(cat[k]!), values(v), `${lang} ${k}`);
		}
	}
});

test("the language comes from ?lang, then the saved choice, then the browser, then 繁體中文", () => {
	assert.equal(pickLang("es", "ja", "ko"), "es");
	assert.equal(pickLang(null, "ja", "ko"), "ja");
	assert.equal(pickLang(null, null, "ko-KR,ko;q=0.9"), "ko");
	assert.equal(pickLang(null, null, "zh-CN,zh;q=0.9"), "zh-Hans");
	assert.equal(pickLang(null, null, "zh-TW"), "zh-Hant");
	assert.equal(pickLang(null, null, "de-DE,fr"), "zh-Hant");
	assert.equal(pickLang("xx", null, null), "zh-Hant");
});

test("the landing page carries the non-commercial notice and Pollen's attribution in each language", () => {
	for (const lang of LANGS) {
		const html = landingPage({
			lang, nonce: "n", minutes: 15, stats: { live: 3, queued: 0, max: 10 }, sitekey: "k", scriptUrl: "https://x.test/api.js",
		});
		assert.ok(html.includes(esc(t(lang, "land.nc.text"))), lang);
		assert.ok(html.includes("Pollen Robotics"), lang);
		assert.ok(html.includes("CC BY-NC-SA 4.0"), lang);
		assert.ok(html.includes(`lang="${lang}"`), lang);
		assert.ok(!html.toLowerCase().includes("official pollen"), lang);
	}
});

test("the waiting page shows the place in line", () => {
	const html = waitingPage({ lang: "en", nonce: "n", minutes: 15, view: { phase: "queued", place: 4 } });
	assert.ok(html.includes("You are number 4"));
});

test("a visitor whose duck has ended sees why, in their language", () => {
	const html = landingPage({
		lang: "es", nonce: "n", minutes: 15, stats: { live: 0, queued: 0, max: 10 }, sitekey: "k", scriptUrl: "https://x.test/api.js",
		ended: { phase: "gone", reason: "crash" },
	});
	assert.ok(html.includes(esc(t("es", "gone.title"))));
	assert.ok(html.includes(esc(t("es", "gone.crash"))));
});

test("time up: the page says so, thanks the visitor and offers another duck", () => {
	const html = landingPage({
		lang: "zh-Hant", nonce: "n", minutes: 5, stats: { live: 0, queued: 0, max: 1 }, sitekey: "k", scriptUrl: "https://x.test/api.js",
		ended: { phase: "gone", reason: "time" },
	});
	assert.match(html, /<h1[^>]*>時間到了<\/h1>/);
	assert.ok(html.includes(esc(t("zh-Hant", "gone.note"))));
	assert.match(html, /id="startBtn"[^>]*>再來一隻<svg[^>]*>.*?<\/svg><\/button>/);
});

test("the landing page says what a visit is: a fresh duck, its minutes, one at a time or a few, the .duck file", () => {
	const page = (max: number) =>
		landingPage({ lang: "en", nonce: "n", minutes: 5, stats: { live: 0, queued: 0, max }, sitekey: "k", scriptUrl: "https://x.test/api.js" });
	const one = page(1);
	assert.ok(one.includes("brand-new random duck to play with in its own simulator for 5 minutes"));
	assert.ok(one.includes("One visitor at a time; the others wait in line."));
	assert.ok(one.includes("download it as a .duck file"));
	assert.ok(one.includes('id="startBtn"') && one.includes('href="https://github.com/joseyarashio/duckmind"'), "play and the repository");
	const three = page(3);
	assert.ok(three.includes("Up to 3 visitors at a time; the others wait in line."));
	assert.ok(!three.includes("One visitor at a time"));
});

test("the guiding page says the ducks are adopted and links the README quick start, in each language", () => {
	for (const lang of LANGS) {
		const html = guidingPage({ lang, nonce: "n", minutes: 5 });
		for (const key of ["guide.title", "guide.later", "guide.home.title", "guide.home.text", "guide.home.link"]) {
			assert.ok(html.includes(esc(t(lang, key))), `${lang} ${key}`);
		}
		assert.ok(html.includes(`href="${QUICK_START_URL}"`), lang);
		assert.ok(html.includes("Pollen Robotics"), lang);
		assert.ok(!html.includes("<form"), `${lang}: nothing to start`);
	}
});
