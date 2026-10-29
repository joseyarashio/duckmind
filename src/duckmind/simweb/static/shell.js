/* The page shell (UI rebuild 2026-10-10, docs/design/UI-REBUILD-2026-10-10.md).
 *
 * Only layout lives here; no element is made or moved, so every id the other scripts use stays
 * where the markup puts it. It does four things:
 *   - the control panel's tabs (逗逗牠, 動作, 情緒, 說話, 換鴨子, 腦; 狀態, 互動, 工具 for Duck Core's own
 *     mode): one tab selected, the rest hidden, arrow keys move between them; a tab whose parts
 *     the mode puts away (or that the server does not offer: 說話) is skipped;
 *   - the panel's open/closed state where it is a sheet (a phone, a tablet) or a floating panel
 *     (the enlarged view): the handle button, a swipe on the handle, the 「腦」 button;
 *   - the two views of the one page: the simulator (/) and 「資料」 (/data), same document, so the
 *     scripts keep working; the 3D view is not drawn while 「資料」 is open;
 *   - --topbar-h, so the app fills exactly what is under the header.
 * Words come from the catalogs (I18N); nothing here decides anything for the duck. */
"use strict";

(function pageShell() {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const view = $("#duckView");
  const tablist = $("#ctlTabs");
  if (!view || !tablist) return;
  const tabs = $$("[data-ctl-tab]", tablist);
  const panels = $$("[data-ctl-panel]");
  const handle = $("#ctlHandle");
  const sheetQuery = window.matchMedia("(max-width: 1023.98px)");
  const KEY = "duck.ctl.tab";
  const state = { tab: "", raf: 0, user: false };

  const store = {
    get() {
      try {
        return window.sessionStorage.getItem(KEY) || "";
      } catch (error) {
        return "";
      }
    },
    set(value) {
      try {
        window.sessionStorage.setItem(KEY, value);
      } catch (error) {
        /* storage may be blocked: the choice just is not remembered */
      }
    },
  };

  // (words are written once the catalogs are in; I18N.text then remembers them for a language switch)
  const ready = window.I18N && window.I18N.ready ? window.I18N.ready : Promise.resolve();
  const words = (el, key) => {
    if (!el || !window.I18N) return;
    ready.then(() => window.I18N.text(el, key), () => window.I18N.text(el, key));
  };

  const shown = (el) => el.getClientRects().length > 0;
  const big = () => view.classList.contains("is-big");
  // The panel is always open in the desktop column; a sheet or the floating panel has a state.
  const floating = () => big() || sheetQuery.matches;
  const isOpen = () => view.dataset.ctl === "open";

  /* ---------------- open and closed ---------------- */

  function setOpen(open) {
    view.dataset.ctl = open ? "open" : "closed";
    handle.setAttribute("aria-expanded", String(open));
    words($("#ctlHandleText"), open ? "page.shell.panel.close" : "page.shell.panel.open");
    syncSheet();
  }

  // (a phone: the app's height is the screen; the sheet takes its share, the view keeps the rest)
  function syncSheet() {
    document.body.classList.toggle("ctl-sheet-open", floating() && isOpen());
  }

  handle.addEventListener("click", () => {
    if (handle.dataset.swiped) {
      delete handle.dataset.swiped;
      return;
    }
    setOpen(!isOpen());
    if (isOpen()) focusTab();
  });

  // A swipe on the handle: up opens, down closes (a tap still toggles).
  let touchY = null;
  handle.addEventListener("pointerdown", (event) => {
    touchY = event.clientY;
  });
  handle.addEventListener("pointerup", (event) => {
    if (touchY == null) return;
    const dy = event.clientY - touchY;
    touchY = null;
    if (Math.abs(dy) < 28) return;
    handle.dataset.swiped = "1";
    setOpen(dy < 0);
  });
  handle.addEventListener("pointercancel", () => {
    touchY = null;
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !isOpen() || !floating() || event.defaultPrevented) return;
    if (!view.contains(document.activeElement) && document.activeElement !== document.body) return;
    setOpen(false);
    handle.focus();
  });

  // Enlarged: the panel floats at the bottom right (open where there is room); back to normal: shut
  // again where it is a sheet.
  new MutationObserver(() => {
    if (view.dataset.bigSeen === String(big())) return;
    view.dataset.bigSeen = String(big());
    setOpen(big() ? window.innerWidth >= 700 : false);
  }).observe(view, { attributes: true, attributeFilter: ["class"] });
  const onQuery = () => syncSheet();
  if (sheetQuery.addEventListener) sheetQuery.addEventListener("change", onQuery);
  else if (sheetQuery.addListener) sheetQuery.addListener(onQuery);

  /* ---------------- tabs ---------------- */

  const usable = () => tabs.filter((tab) => !tab.hidden && shown(tab));

  function select(id, { focus = false, open = false, user = false } = {}) {
    const tab = tabs.find((item) => item.dataset.ctlTab === id);
    if (!tab || tab.hidden) return false;
    state.tab = id;
    for (const item of tabs) {
      const on = item === tab;
      item.setAttribute("aria-selected", String(on));
      item.tabIndex = on ? 0 : -1;
    }
    for (const panel of panels) panel.hidden = panel.dataset.ctlPanel !== id;
    if (user) {
      store.set(id);
      state.user = true;
    }
    if (open && !isOpen() && floating()) setOpen(true);
    if (focus) tab.focus({ preventScroll: true });
    return true;
  }

  function focusTab() {
    const tab = tabs.find((item) => item.dataset.ctlTab === state.tab);
    if (tab && shown(tab)) tab.focus({ preventScroll: true });
  }

  // The selected tab must be one that can be seen: after the mode (or the server) changes what is
  // offered, keep it, else the remembered one, else the first.
  function settle() {
    state.raf = 0;
    if (document.body.classList.contains("view-data")) return;
    const list = usable();
    if (!list.length) return;
    const ids = list.map((tab) => tab.dataset.ctlTab);
    // (what the viewer picked stays while it can be seen; else the remembered tab, else the first —
    // and until the viewer picks one, the first tab that can be seen, as the mode settles)
    let want = state.user && ids.includes(state.tab) ? state.tab : ids.includes(store.get()) ? store.get() : ids[0];
    if (!state.user && !ids.includes(store.get())) want = ids[0];
    if (want !== state.tab || list.find((tab) => tab.dataset.ctlTab === want).getAttribute("aria-selected") !== "true") select(want);
  }

  const settleSoon = () => {
    if (!state.raf) state.raf = requestAnimationFrame(settle);
  };

  tablist.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-ctl-tab]");
    if (tab) select(tab.dataset.ctlTab, { open: true, user: true });
  });

  tablist.addEventListener("keydown", (event) => {
    const keys = ["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp", "Home", "End"];
    if (!keys.includes(event.key)) return;
    const list = usable();
    const at = list.findIndex((tab) => tab === document.activeElement);
    if (at < 0) return;
    event.preventDefault();
    let next = at;
    if (event.key === "Home") next = 0;
    else if (event.key === "End") next = list.length - 1;
    else if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (at + 1) % list.length;
    else next = (at - 1 + list.length) % list.length;
    select(list[next].dataset.ctlTab, { focus: true, user: true });
  });

  // 「腦」 on the view, and any link into a part of the panel (「換一隻小鴨」 → the picker).
  document.addEventListener("click", (event) => {
    const opener = event.target.closest("[data-ctl-open]");
    if (opener) {
      event.preventDefault();
      select(opener.dataset.ctlOpen, { open: true, user: true });
      return;
    }
    const link = event.target.closest('a[href^="#"]');
    if (!link || link.getAttribute("href") === "#") return;
    const target = document.getElementById(link.getAttribute("href").slice(1));
    const panel = target && target.closest("[data-ctl-panel]");
    if (!panel) return;
    event.preventDefault();
    select(panel.dataset.ctlPanel, { open: true, user: true });
    target.scrollIntoView({ block: "nearest" });
  });

  // What is offered changes after the first paint (the mode, the server's live voice): look again.
  new MutationObserver(settleSoon).observe(tablist, { subtree: true, attributes: true, attributeFilter: ["hidden"] });
  new MutationObserver(settleSoon).observe(document.body, { attributes: true, attributeFilter: ["class"] });
  window.addEventListener("duck:neuro", settleSoon);
  window.addEventListener("duck:view", settleSoon);

  // 說話 is offered only when the server runs live voice (live_voice.js shows its card then).
  const voiceCard = $("#liveVoice");
  const voiceTab = $("#ctlTab-voice");
  if (voiceCard && voiceTab) {
    const syncVoice = () => {
      if (voiceTab.hidden !== voiceCard.hidden) voiceTab.hidden = voiceCard.hidden;
    };
    new MutationObserver(syncVoice).observe(voiceCard, { attributes: true, attributeFilter: ["hidden"] });
    syncVoice();
  }

  /* ---------------- the two views: the simulator and 「資料」 ---------------- */

  const simView = $("#viewSim");
  const dataView = $("#viewData");
  const isData = () => location.pathname.replace(/\/+$/, "") === "/data";

  function applyView(scroll) {
    const data = isData();
    document.body.classList.toggle("view-data", data);
    if (simView) simView.hidden = data;
    if (dataView) dataView.hidden = !data;
    for (const link of $$("[data-view-link]")) {
      const on = link.dataset.viewLink === (data ? "data" : "sim");
      if (link.classList.contains("btn")) continue;
      if (on) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
    if (scroll) window.scrollTo(0, 0);
    window.dispatchEvent(new CustomEvent("duck:view", { detail: { view: data ? "data" : "sim" } }));
  }

  document.addEventListener("click", (event) => {
    const link = event.target.closest("[data-view-link]");
    if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const to = new URL(link.href, location.href);
    if (to.origin !== location.origin) return;
    event.preventDefault();
    if (to.pathname !== location.pathname) history.pushState({}, "", to.pathname);
    applyView(true);
  });
  window.addEventListener("popstate", () => applyView(false));

  /* ---------------- the header's height ---------------- */

  const bar = $(".topbar");
  const measure = () => {
    if (bar) document.documentElement.style.setProperty("--topbar-h", `${Math.round(bar.getBoundingClientRect().height)}px`);
  };
  if (bar && "ResizeObserver" in window) new ResizeObserver(measure).observe(bar);
  window.addEventListener("resize", measure);
  measure();

  /* ---------------- the physics label, in the HUD's tooltip ---------------- */

  // The label that said which duck is drawn (#stageMode: 「物理真相」 and its line) is no longer shown over the
  // view (style.css); its two lines, which duck3d.js / app.js still write there, ride on the HUD's tooltip.
  const modeLabel = $("#stageMode");
  const hudBox = $("#hud");
  if (modeLabel && hudBox) {
    const carry = () => {
      const lines = Array.from(modeLabel.children, (part) => part.textContent.trim()).filter(Boolean);
      if (lines.length) hudBox.title = lines.join("\n");
      else hudBox.removeAttribute("title");
    };
    new MutationObserver(carry).observe(modeLabel, { childList: true, subtree: true, characterData: true });
    carry();
  }

  /* ---------------- first paint ---------------- */

  setOpen(false);
  applyView(false);
  settle();
  if (window.I18N && typeof window.I18N.onChange === "function") {
    window.I18N.onChange(() => words($("#ctlHandleText"), isOpen() ? "page.shell.panel.close" : "page.shell.panel.open"));
  }
  window.duckShell = { select, setOpen, applyView };
})();
