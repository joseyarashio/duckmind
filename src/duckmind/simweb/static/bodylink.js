/* 我的小鴨 → 真的小鴨 (item 6, docs/design/neural-duck/UI-REDESIGN-2026-11-05.md; docs/real-microduck.md):
 * which body this duck's brain drives now, said plainly, and the link's own numbers when the body is on
 * another machine. It reads
 *   GET /api/body      enabled / starting / available / error, and remote: the link's stats (url, up,
 *                      rtt_ms_p50, connects, drops, max_gap_ms, deadman_ms), absent for a body run here;
 *                      real: null for a simulated body, else {rehearsal, …} (a real Microduck, or a
 *                      rehearsal body on this computer that plays one)
 * every 5 s while the 我的小鴨 tab is shown. It changes nothing: connecting a real duck is a restart
 * with --body remote://… (the card's steps).
 * Words: none here (page.body.*); the address is data. */
"use strict";

(function bodyLinkModule() {
  const POLL_MS = 5000;

  /* What one /api/body answer says about the body: {state, key, params, stats} (stats: a line of the
     link's numbers, or null). ``answer`` null: the server did not answer. */
  function linkState(answer) {
    if (!answer) return { state: "offline", key: "page.body.state.offline", params: {}, stats: null };
    if (!answer.enabled) return { state: "off", key: "page.body.state.off", params: {}, stats: null };
    const remote = answer.remote;
    if (remote && typeof remote === "object") {
      const url = String(remote.url || "");
      const stats = {
        key: "page.body.stats",
        params: {
          connects: remote.connects || 0,
          drops: remote.drops || 0,
          gap: Math.round(remote.max_gap_ms || 0),
          deadman: Math.round(remote.deadman_ms || 0),
        },
      };
      if (!remote.up) return { state: "remote_down", key: "page.body.state.remote_down", params: { url }, stats };
      const ms = remote.rtt_ms_p50;
      const known = typeof ms === "number";
      // A real duck (or the rehearsal that plays one): said as such, with the same numbers — or, while
      // it is not on yet (`turn_on`), that a person turns it on with the duck's own control.
      if (answer.real && typeof answer.real === "object") {
        if (answer.real.turn_on) return { state: "real_off", key: "page.body.state.real_off", params: { url }, stats };
        const [withMs, without] = answer.real.rehearsal
          ? ["page.body.state.rehearsal", "page.body.state.rehearsal_new"]
          : ["page.body.state.real", "page.body.state.real_new"];
        return { state: "real", key: known ? withMs : without, params: known ? { url, ms: Math.round(ms) } : { url }, stats };
      }
      if (known) return { state: "remote_up", key: "page.body.state.remote_up", params: { url, ms: Math.round(ms) }, stats };
      return { state: "remote_up", key: "page.body.state.remote_up_new", params: { url }, stats };
    }
    if (answer.starting) return { state: "starting", key: "page.body.state.starting", params: {}, stats: null };
    if (answer.error || answer.available === false) return { state: "error", key: "page.body.state.error", params: {}, stats: null };
    return { state: "local", key: "page.body.state.local", params: {}, stats: null };
  }

  function start() {
    const I18N = window.I18N;
    const card = document.getElementById("bodyLink");
    const line = document.getElementById("bodyLinkState");
    const statsLine = document.getElementById("bodyLinkStats");
    const panel = document.getElementById("ctl-duck");
    if (!card || !line || !panel || !I18N) return;
    let asking = false;

    function paint(said) {
      line.dataset.state = said.state;
      I18N.text(line, said.key, said.params);
      if (said.stats) {
        I18N.text(statsLine, said.stats.key, said.stats.params);
        statsLine.hidden = false;
      } else {
        statsLine.hidden = true;
        statsLine.textContent = "";
      }
    }

    // (only while the tab can be seen: the page is on the simulator and 我的小鴨 is the chosen tab)
    const shown = () => !document.hidden && !panel.hidden && !document.body.classList.contains("view-data");

    async function look() {
      if (asking || !shown()) return;
      asking = true;
      let answer = null;
      try {
        const response = await fetch("/api/body", { cache: "no-store" });
        answer = response.ok ? await response.json() : null;
      } catch (error) {
        answer = null;
      }
      asking = false;
      paint(linkState(answer));
    }

    new MutationObserver(look).observe(panel, { attributes: true, attributeFilter: ["hidden"] });
    document.addEventListener("visibilitychange", look);
    window.setInterval(look, POLL_MS);
    look();
  }

  const api = { linkState, POLL_MS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else {
    window.duckBodyLink = api;
    (window.I18N && window.I18N.ready ? window.I18N.ready : Promise.resolve()).then(start, start);
  }
})();
