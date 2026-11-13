/**
 * Public play edge: one Worker, a Gate Durable Object (3 slots, the line and the spending meter) and one DuckSession Durable
 * Object per visitor, which owns that visitor's Container (Durable Object scheduling policy).
 *
 * Settings come from the environment when `cf` evaluates this file, so nothing account- or
 * domain-specific lives in the repository:
 *   DUCK_EDGE_DOMAIN     custom domain for the Worker (a subdomain of a zone on the account); the
 *                        workers.dev address is then switched off. Unset → workers.dev only, and
 *   DUCK_EDGE_HOSTNAMES  must name it (<worker>.<subdomain>.workers.dev) for the Turnstile check.
 *   DUCK_EDGE_REGIONS    container placement, comma-separated (e.g. "APAC"); unset → the location
 *                        nearest to the visitor (Cloudflare's default).
 *   DUCK_EDGE_INSTANCE   container instance type: lite | basic | standard-1..4 | JSON {"vcpu","memoryMib","diskMb"};
 *                        the meter prices it at its worst case (src/core/config.ts PRICING).
 *   DUCK_EDGE_SITEKEY    the Turnstile widget's public sitekey.
 *   DUCK_EDGE_MAX_LIVE   live sessions at once (default 3).
 *   DUCK_EDGE_OPEN       "true" (default) lets visitors in; "false" closes everything: no new ducks,
 *                        the line is emptied and running containers are stopped within half a minute.
 *   DUCK_EDGE_BUDGET_MONTH_USD  container spending per billing month, US$ (default 25: a US$30 cap
 *                        minus the US$5 Workers Paid plan). Each day gets what is left this month
 *                        divided by the days left, so an unused day carries over.
 *   DUCK_EDGE_BILLING_DAY  day of the month (1-28) the account's invoices start (default 1).
 *   DUCK_EDGE_DOCKERFILE / DUCK_EDGE_BUILD_CONTEXT  the duck image (default ../image/Dockerfile with
 *                        the repository root as context, from deploy/public/image).
 * Secrets (never here): TURNSTILE_SECRET and IP_HASH_KEY, uploaded with `cf deploy --secrets-file`.
 *
 * `cf dev` (mode "development") runs without Docker: no Container is attached, and the session
 * forwards to the local stub container (dev/stub-container.mjs) instead. Run the stub first:
 *   node dev/stub-container.mjs & npx cf dev      → http://127.0.0.1:8789/
 */
import { bindings, defineConfig, defineContainer, exports } from "cf/config";
import * as entrypoint from "./src/index.ts" with { type: "cf-worker" };

const REGIONS = ["ENAM", "WNAM", "EEUR", "WEUR", "APAC", "SAM", "ME", "OC", "AFR"];

function regions(value: string | undefined): string[] | undefined {
	const list = (value ?? "")
		.split(",")
		.map((r) => r.trim().toUpperCase())
		.filter(Boolean);
	if (list.length === 0) return undefined;
	const bad = list.filter((r) => !REGIONS.includes(r));
	if (bad.length) throw new Error(`DUCK_EDGE_REGIONS: unknown region ${bad.join(", ")} (use ${REGIONS.join(", ")})`);
	return list;
}

export default defineConfig(({ mode }) => {
	const dev = mode === "development";
	const env = process.env;
	const domain = env.DUCK_EDGE_DOMAIN?.trim() || undefined;
	const placement = regions(env.DUCK_EDGE_REGIONS);
	const stubPort = env.DUCK_EDGE_STUB_PORT?.trim() || "8790";
	const stub = `http://127.0.0.1:${stubPort}`;

	const duckContainer = defineContainer({
		name: "duckmind-public-duck",
		schedulingPolicy: "durable-object",
		// The image the visitor's duck runs (owned by deploy/public/image); built by `cf deploy`.
		images: {
			duck: {
				dockerfile: env.DUCK_EDGE_DOCKERFILE?.trim() || "../image/Dockerfile",
				buildContext: env.DUCK_EDGE_BUILD_CONTEXT?.trim() || "../../..",
			},
		},
		observability: { enabled: true, logs: { enabled: true } },
		// The Durable Object policy's config schema has no `constraints` field yet; the Containers API
		// does (constraints.regions), so it is passed through unchanged.
		...(placement ? { unsafe: { constraints: { regions: placement } } } : {}),
	});

	// Turnstile tokens are accepted only from the public hostname: the custom domain, or (without
	// one) the workers.dev address given in DUCK_EDGE_HOSTNAMES.
	const hostnames = dev ? "localhost,127.0.0.1" : env.DUCK_EDGE_HOSTNAMES?.trim() || domain || "";
	const sitekey = dev ? "local-stub-sitekey" : env.DUCK_EDGE_SITEKEY?.trim() || "";
	if (!dev && !sitekey) throw new Error("set DUCK_EDGE_SITEKEY (the Turnstile widget's sitekey)");
	if (!dev && !hostnames) throw new Error("set DUCK_EDGE_DOMAIN, or DUCK_EDGE_HOSTNAMES=<worker>.<subdomain>.workers.dev");

	return {
		worker: {
			name: "duckmind-public",
			compatibilityDate: "2026-11-04",
			entrypoint,
			assets: { runWorkerFirst: true },
			// With a custom domain the visitor uses only that one address.
			workersDev: dev || !domain,
			...(domain && !dev ? { domains: [domain] } : {}),
			observability: {
				enabled: true,
				redactQueryString: true,
				logs: { enabled: true, invocationLogs: true },
				traces: { enabled: true, headSamplingRate: 0.1 },
			},
			exports: {
				Gate: exports.durableObject({ storage: "sqlite" }),
				DuckSession: dev
					? exports.durableObject({ storage: "sqlite" })
					: exports.durableObject({ storage: "sqlite", container: duckContainer }),
			},
			env: {
				ASSETS: bindings.assets(),
				GATE: bindings.durableObject({ worker: "duckmind-public", exportName: "Gate" }),
				SESSIONS: bindings.durableObject({ worker: "duckmind-public", exportName: "DuckSession" }),
				MAX_LIVE: bindings.text(env.DUCK_EDGE_MAX_LIVE?.trim() || "3"),
				MAX_QUEUE: bindings.text("100"),
				SESSION_MINUTES: bindings.text("5"),
				EXPORT_MINUTES: bindings.text("3"),
				START_TIMEOUT_S: bindings.text("180"),
				IDLE_END_S: bindings.text("120"),
				START_RATE_MAX: bindings.text("3"),
				START_WINDOW_S: bindings.text("3600"),
				MAX_PER_ADDRESS: bindings.text("1"),
				MAX_BODY_BYTES: bindings.text(String(256 * 1024)),
				CONTAINER_INSTANCE: bindings.text(env.DUCK_EDGE_INSTANCE?.trim() || "standard-2"),
				OPEN: bindings.text(env.DUCK_EDGE_OPEN?.trim() || "true"),
				BUDGET_MONTH_USD: bindings.text(env.DUCK_EDGE_BUDGET_MONTH_USD?.trim() || "25"),
				BILLING_DAY: bindings.text(env.DUCK_EDGE_BILLING_DAY?.trim() || "1"),
				TURNSTILE_HOSTNAMES: bindings.text(hostnames),
				// Development uses the stub's own widget and check; production uses Cloudflare's.
				TURNSTILE_SITEKEY: bindings.text(sitekey),
				...(dev
					? {
							DEV_UPSTREAM_URL: bindings.text(stub),
							TURNSTILE_SCRIPT_URL: bindings.text(`${stub}/__stub/turnstile.js`),
							TURNSTILE_VERIFY_URL: bindings.text(`${stub}/__stub/siteverify`),
							TURNSTILE_SECRET: bindings.text("local-stub-secret"),
							IP_HASH_KEY: bindings.text("local-dev-only"),
						}
					: {
							TURNSTILE_SECRET: bindings.secret(),
							IP_HASH_KEY: bindings.secret(),
						}),
			},
		},
		containers: dev ? [] : [duckContainer],
	};
});
