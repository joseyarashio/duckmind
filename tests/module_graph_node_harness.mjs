// Walks the page's ES modules as a browser loads them, without running them: each module's source is fetched from
// the server, V8 reads its static imports (vm.SourceTextModule: parsed, never evaluated), each one is resolved with
// the page's import map and fetched in turn. (An import() worked out while a script runs is not seen here.)
//   node --experimental-vm-modules module_graph_node_harness.mjs <server url> '{"entries": [paths], "imports": {import map}}'
// Prints {modules: {path: [the paths it imports]}, problems: [what a browser would fail on, in words]}.
import vm from "node:vm";

const [base, inputJson] = process.argv.slice(2);
const { entries, imports } = JSON.parse(inputJson);
const origin = new URL(base).origin;

// A specifier as a URL, the way a browser resolves it: the import map first (a name, then a "prefix/"), else a
// relative or absolute URL; a bare name the map does not have is refused (null).
function resolve(specifier, from) {
  if (specifier in imports) return new URL(imports[specifier], from).href;
  for (const [prefix, target] of Object.entries(imports)) {
    if (prefix.endsWith("/") && specifier.startsWith(prefix)) return new URL(target + specifier.slice(prefix.length), from).href;
  }
  return /^(\.{0,2}\/|[a-z][a-z0-9+.-]*:)/i.test(specifier) ? new URL(specifier, from).href : null;
}

const local = (url) => url.slice(origin.length);
const modules = {};
const problems = [];
const queue = entries.map((entry) => new URL(entry, base).href);
while (queue.length) {
  const url = queue.shift();
  if (local(url) in modules) continue;
  const found = (modules[local(url)] = []);
  const response = await fetch(url);
  const kind = response.headers.get("content-type") || "";
  if (!response.ok || !kind.startsWith("text/javascript")) {
    problems.push(`${local(url)} is not served as a script (${response.status} ${kind})`);
    continue;
  }
  const module = new vm.SourceTextModule(await response.text(), { identifier: url });
  for (const specifier of module.dependencySpecifiers) {
    const to = resolve(specifier, url);
    if (!to) problems.push(`${local(url)} imports "${specifier}", which the import map does not have`);
    else if (new URL(to).origin !== origin) problems.push(`${local(url)} imports "${specifier}" from another site`);
    else {
      found.push(local(to));
      queue.push(to);
    }
  }
}
console.log(JSON.stringify({ modules, problems }));
