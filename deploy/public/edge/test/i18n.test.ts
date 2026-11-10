import assert from "node:assert/strict";
import { test } from "node:test";
import { CATALOGS, LANGS, pickLang, t } from "../src/core/i18n.ts";
import { esc, landingPage, waitingPage } from "../src/core/pages.ts";

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
