/**
 * The landing page's own Turnstile script, run in Node over a fake page: a fake `turnstile` object
 * records each render and lets the test call the widget's callbacks; timers are driven by hand. A
 * visitor whose check fails, expires, times out or never arrives must get Cloudflare's visible box and
 * one plain line, never a disabled button with nothing to do.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import { t } from "../src/core/i18n.ts";
import { landingPage } from "../src/core/pages.ts";

interface FakeElement {
	id: string;
	dataset: Record<string, string>;
	disabled: boolean;
	hidden: boolean;
	textContent: string;
	innerHTML: string;
	classList: { add(c: string): void; contains(c: string): boolean };
}

interface Render {
	el: FakeElement;
	opts: Record<string, unknown> & { appearance?: string };
	id: string;
	removed: boolean;
}

interface FakePage {
	els: Record<string, FakeElement>;
	renders: Render[];
	/** The widget on the page now (the last one rendered and not removed). */
	readonly widget: Render;
	/** Call one of the widget's callbacks, as Turnstile would. */
	call(name: string, ...args: unknown[]): void;
	advance(ms: number): void;
}

function element(id: string, extra: Partial<FakeElement> = {}): FakeElement {
	const classes = new Set<string>();
	return {
		id,
		dataset: {},
		disabled: false,
		hidden: false,
		textContent: "",
		innerHTML: "",
		classList: { add: (c) => void classes.add(c), contains: (c) => classes.has(c) },
		...extra,
	};
}

/** The landing page in `lang` with its inline head script run; the widget script "loads" at once. */
function page(lang: "zh-Hant" | "en" | "es" = "en"): FakePage {
	const html = landingPage({
		lang, nonce: "n", minutes: 5, stats: { live: 0, queued: 0, max: 1 }, sitekey: "site-key", scriptUrl: "https://x.test/api.js",
	});
	const head = /<script nonce="n">([\s\S]*?)<\/script>/.exec(html)![1]!;
	const hintText = /id="verifyHint"[^>]*>([^<]*)</.exec(html)![1]!;
	const els: Record<string, FakeElement> = {
		turnstile: element("turnstile", { dataset: { sitekey: "site-key" } }),
		startBtn: element("startBtn", { disabled: true }),
		verifyHint: element("verifyHint", { textContent: hintText }),
	};
	const renders: Render[] = [];
	const timers: Array<{ at: number; fn: () => void; live: boolean }> = [];
	let now = 0;
	const turnstile = {
		render(el: FakeElement, opts: Render["opts"]) {
			const r = { el, opts, id: `w${renders.length}`, removed: false };
			renders.push(r);
			return r.id;
		},
		remove(id: string) {
			const r = renders.find((x) => x.id === id);
			if (r) r.removed = true;
		},
	};
	const window: Record<string, unknown> = { turnstile };
	const document = {
		readyState: "complete",
		getElementById: (id: string) => els[id] ?? null,
		addEventListener() {},
	};
	const setTimeout = (fn: () => void, ms: number) => {
		const timer = { at: now + ms, fn, live: true };
		timers.push(timer);
		return timer;
	};
	const clearTimeout = (timer?: { live: boolean }) => {
		if (timer) timer.live = false;
	};
	runInNewContext(head, { window, document, setTimeout, clearTimeout });
	(window.dmTurnstileReady as () => void)();
	return {
		els,
		renders,
		/** The widget on the page now (the last one rendered and not removed). */
		get widget(): Render {
			return renders.filter((r) => !r.removed).at(-1)!;
		},
		call(name: string, ...args: unknown[]) {
			(this.widget.opts[name] as (...a: unknown[]) => unknown)(...args);
		},
		advance(ms: number) {
			now += ms;
			for (const timer of timers) if (timer.live && timer.at <= now) (timer.live = false), timer.fn();
		},
	};
}

test("a check that passes quietly: one hidden widget, the button unlocks, nothing else shows", () => {
	const p = page();
	assert.equal(p.renders.length, 1);
	assert.equal(p.widget.opts.appearance, "interaction-only");
	p.call("callback", "token");
	assert.equal(p.els.startBtn!.disabled, false);
	assert.equal(p.els.verifyHint!.hidden, true);
	p.advance(30_000);
	assert.equal(p.renders.length, 1, "no second widget after a token");
});

for (const [why, act] of [
	["fails (a hung widget, 300030)", (p: FakePage) => p.call("error-callback", "300030")],
	["fails (Cloudflare's always-fail answer, 600010)", (p: FakePage) => p.call("error-callback", "600010")],
	["expires", (p: FakePage) => p.call("expired-callback")],
	["times out", (p: FakePage) => p.call("timeout-callback")],
	["never answers within 8 s", (p: FakePage) => p.advance(8_000)],
] as const) {
	test(`a check that ${why}: Cloudflare's box shows and one plain line says to try again`, () => {
		const p = page("zh-Hant");
		act(p);
		assert.equal(p.widget.opts.appearance, "always", "the visible widget");
		assert.equal(p.widget.el.id, "turnstile");
		assert.equal(p.renders.filter((r) => !r.removed).length, 1, "one widget on the page");
		assert.equal(p.widget.opts.sitekey, "site-key");
		assert.equal(p.widget.opts.action, "start");
		assert.notEqual(p.widget.opts.retry, "never", "Turnstile keeps retrying");
		assert.equal(p.els.startBtn!.disabled, true);
		assert.equal(p.els.verifyHint!.hidden, false);
		assert.equal(p.els.verifyHint!.textContent, t("zh-Hant", "land.check.retry"));
		// The visitor ticks the box: the button unlocks and the line goes away.
		p.call("callback", "token");
		assert.equal(p.els.startBtn!.disabled, false);
		assert.equal(p.els.verifyHint!.hidden, true);
	});
}

test("an unsupported browser (110500) is told to use another browser", () => {
	const p = page("es");
	p.call("error-callback", "110500");
	assert.equal(p.widget.opts.appearance, "always");
	assert.equal(p.els.verifyHint!.textContent, t("es", "land.check.browser"));
	assert.equal(p.els.startBtn!.disabled, true);
});

test("a second failure on the visible box keeps one box and says it again", () => {
	const p = page();
	p.call("error-callback", "300030");
	p.call("error-callback", "600010");
	assert.equal(p.renders.length, 2, "rendered visible once, then Turnstile's own retry");
	assert.equal(p.els.verifyHint!.textContent, t("en", "land.check.retry"));
	p.call("callback", "token");
	p.call("expired-callback");
	assert.equal(p.els.startBtn!.disabled, true);
	assert.equal(p.els.verifyHint!.textContent, t("en", "land.check.retry"));
});
