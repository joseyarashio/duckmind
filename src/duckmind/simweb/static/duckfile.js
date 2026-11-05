/* 我的小鴨檔案 (docs/duck-file.md): the 我的小鴨 tab's card to export the duck this server lives and to
 * import a duck file as a new life on this computer.
 *   GET  /api/life/duckfile          whether this server lives a duck (else the export is put away)
 *   GET  /api/life/duckfile/export   the download (a plain link)
 *   POST /api/life/duckfile/inspect  the chosen file's bytes: what is in it, is the same duck here
 *   POST /api/life/duckfile/import   the same bytes (?replace=1: replace it, its folder kept as a backup)
 * Every word comes from the catalogs (page.duckfile.*) or the server's _i18n map; the duck's name and
 * folders are data. */
(function () {
  "use strict";

  const ROUTE = "/api/life/duckfile";
  const $ = (id) => document.getElementById(id);
  const I18N = window.I18N;
  let chosen = null; // the chosen file's bytes, while its answer is shown

  // a line of the card: a catalog key with its values, or nothing (forgotten, so a language switch
  // does not bring old words back)
  function line(el, key, params) {
    if (key) {
      I18N.text(el, key, params);
      el.hidden = false;
      return;
    }
    delete el.dataset.i18n;
    delete el.dataset.i18nParams;
    delete el.dataset.i18nField;
    delete el._i18nField;
    el.textContent = "";
    el.hidden = true;
  }

  function note(key, params) {
    line($("duckFileNote"), key, params);
    line($("duckFileBackup"), null);
  }

  function noteField(answer) {
    const el = $("duckFileNote");
    I18N.show(el, answer, "error");
    el.hidden = false;
    line($("duckFileBackup"), null);
  }

  function reset() {
    chosen = null;
    $("duckFileFound").hidden = true;
    $("duckFileReplace").hidden = true;
    $("duckFilePut").hidden = false;
    $("duckFileSame").hidden = true;
    $("duckFileInput").value = "";
  }

  async function post(path, bytes) {
    const response = await fetch(ROUTE + path, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Duckmind-Duckfile": "1" },
      body: bytes,
    });
    return response.json();
  }

  function tricksPractised(duck) {
    return Object.values(duck.tricks || {}).filter((p) => p > 0).length;
  }

  async function inspect(file) {
    reset();
    note("page.duckfile.checking");
    let answer;
    try {
      chosen = await file.arrayBuffer();
      answer = await post("/inspect", chosen);
    } catch (error) {
      chosen = null;
      note("page.duckfile.failed");
      return;
    }
    if (!answer || !answer.ok) {
      chosen = null;
      if (answer && answer.error) noteField(answer);
      else note("page.duckfile.failed");
      return;
    }
    note(null);
    const duck = answer.duck || {};
    I18N.text($("duckFileFoundText"), "page.duckfile.found", {
      name: duck.name,
      day: duck.growth_day,
      people: duck.people || 0,
      tricks: tricksPractised(duck),
    });
    const same = $("duckFileSame");
    if (answer.running_here) {
      I18N.text(same, "page.duckfile.running", { name: duck.name });
      same.hidden = false;
      $("duckFilePut").hidden = true;
    } else if ((answer.same_here || []).length) {
      I18N.text(same, "page.duckfile.same_here", { name: duck.name });
      same.hidden = false;
      $("duckFilePut").hidden = true;
      $("duckFileReplace").hidden = false;
    }
    $("duckFileFound").hidden = false;
  }

  async function put(replace) {
    if (!chosen) return;
    const bytes = chosen;
    $("duckFilePut").disabled = $("duckFileReplace").disabled = true;
    let answer;
    try {
      answer = await post("/import" + (replace ? "?replace=1" : ""), bytes);
    } catch (error) {
      answer = null;
    }
    $("duckFilePut").disabled = $("duckFileReplace").disabled = false;
    if (!answer || !answer.ok) {
      if (answer && answer.error) noteField(answer);
      else note("page.duckfile.failed");
      return;
    }
    reset();
    note("page.duckfile.done", { name: answer.name, folder: answer.folder });
    if (answer.backup) line($("duckFileBackup"), "page.duckfile.backup", { folder: answer.backup });
  }

  async function start() {
    const card = $("duckFile");
    if (!card) return;
    $("duckFileImport").addEventListener("click", () => $("duckFileInput").click());
    $("duckFileInput").addEventListener("change", (event) => {
      const file = event.target.files && event.target.files[0];
      if (file) inspect(file);
    });
    $("duckFilePut").addEventListener("click", () => put(false));
    $("duckFileReplace").addEventListener("click", () => put(true));
    $("duckFileCancel").addEventListener("click", () => {
      reset();
      note(null);
    });
    try {
      const response = await fetch(ROUTE, { cache: "no-store" });
      const answer = await response.json();
      $("duckFileExport").hidden = !answer.life;
      $("duckFileNoLife").hidden = !!answer.life;
    } catch (error) {
      $("duckFileNoLife").hidden = false;
    }
  }

  (I18N ? I18N.ready : Promise.resolve()).then(start, start);
})();
