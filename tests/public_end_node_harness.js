// Runs static/public.js in Node over a tiny fake page and network, then a stand-in for the rest of the page
// (a poller every 50 ms and a stream, as app.js and neuro.js do), through one scenario:
//   time       the download time runs out; the front says "ended", then "gone"
//   crash      mid-play the duck's polls get 404 and the front says "gone"
//   stray404   one poll gets 404 but the front says the session is on: nothing ends
//   normal     not a public server: everything passes through, nothing ends
//   alone      the download time runs out with no front (/_edge/status is the server's 404)
// Prints one JSON line: what reached the network after the page went quiet, the stream, where it went.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const scenario = process.argv[2];
const JS = fs.readFileSync(path.join(__dirname, "..", "src", "duckmind", "simweb", "static", "public.js"), "utf8");
const t0 = Date.now();
const log = []; // [ms, path] of every request that reached the network
let statusCalls = 0;
let navigatedTo = null;
let crashed = false;

function answer(status, body, type = "application/json") {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name) => (name.toLowerCase() === "content-type" ? type : null) },
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

function network(url) {
  const p = String(url).split("?")[0];
  log.push([Date.now() - t0, p]);
  if (p === "/api/ping") return answer(200, { ok: true, public: scenario !== "normal" });
  if (p === "/api/public/session") {
    if (crashed) return answer(404, "not found", "text/plain");
    const playing = scenario === "crash" || scenario === "stray404" || scenario === "normal";
    return answer(200, {
      ok: true, name: "Pip", ended: !playing, left_s: playing ? 600 : 0,
      export_left_s: playing ? 780 : 2.6, export_minutes: 3, refuses: [],
    });
  }
  if (p === "/_edge/status") {
    if (scenario === "alone") return answer(404, "not found", "text/plain");
    statusCalls++;
    const gone = scenario === "time" ? statusCalls > 2 : crashed;
    return answer(200, { phase: gone ? "gone" : scenario === "time" ? "ended" : "running" });
  }
  if (p === "/api/body" && (crashed || (scenario === "stray404" && !network.strayed))) {
    network.strayed = true;
    return answer(404, "not found", "text/plain");
  }
  return answer(200, { ok: true });
}

class FakeSource {
  constructor(url) {
    this.url = url;
    this.closed = false;
    log.push([Date.now() - t0, `stream ${String(url).split("?")[0]}`]);
  }
  close() {
    this.closed = true;
  }
}

const noop = () => {};
const el = { classList: { add: noop }, hidden: true };
const document = {
  body: el,
  querySelector: () => null,
  querySelectorAll: () => [],
};
// The page's global object is its window (fetch, EventSource and the rest are found on it, as in a browser).
const window = vm.createContext({
  fetch: async (url) => network(url),
  EventSource: FakeSource,
  location: { replace: (to) => (navigatedTo = to) },
  I18N: { t: (k) => k, text: noop, onChange: noop, ready: Promise.resolve() },
  document,
  performance,
  setTimeout,
  setInterval,
  clearInterval,
  clearTimeout,
  console,
});
vm.runInContext("var window = globalThis;", window);
vm.runInContext(JS, window, { filename: "public.js" });

// The rest of the page: it took fetch and EventSource after public.js, as the real scripts do.
const pageFetch = window.fetch;
const Source = window.EventSource;
const stream = new Source("/api/neuro/stream?since=0");
const poller = setInterval(() => {
  pageFetch("/api/body").catch(noop);
}, 50);
if (scenario === "crash") setTimeout(() => (crashed = true), 600);

setTimeout(() => {
  clearInterval(poller);
  const late = new Source("/api/neuro/stream?since=9");
  const leftAt = log.findIndex(([, p]) => p === "/_edge/status");
  const after = leftAt < 0 ? [] : log.slice(leftAt).map(([, p]) => p);
  process.stdout.write(
    JSON.stringify({
      scenario,
      navigatedTo,
      streamClosed: stream.closed,
      statusCalls: log.filter(([, p]) => p === "/_edge/status").length,
      bodyPolls: log.filter(([, p]) => p === "/api/body").length,
      afterQuiet: [...new Set(after)],
    }) + "\n",
  );
  process.exit(0);
}, Number(process.argv[3] || 4000));
