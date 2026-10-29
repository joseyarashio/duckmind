/* The 「成長」 section of the 資料 page (growth G4, docs/design/GROWTH-G4-2026-10-11.md).
 *
 * One component shows how a duck has grown: who it is, its temperament, what it learnt, who it is
 * close to, what it lived, its nightly regrows, its voice and a diary. It is fed by
 *   GET /api/life/growth          the duck this server lives (polled every 15 s while the page is on 資料)
 *   /api/life/fastforward         throwaway test ducks, fast-forwarded in a background worker (the
 *                                 same component shows them, side by side for two care styles)
 * Words: this file holds none. Page words come from the catalogs (I18N.t, page.growth.*); a
 * server field that is Duck Core's own words (a stage, a habit, a call) is a message the server
 * sent along (I18N.field). Names the family gave are data and are shown as they are.
 * Style: docs/simulator-style.md (cards, bars, chips, buttons; tokens only).
 */
"use strict";

(function growthModule() {
  const I18N = window.I18N;
  const T = (key, values) => (I18N ? I18N.t(key, values) : key);
  const F = (obj, name) => (obj ? (I18N ? I18N.field(obj, name) : obj[name]) ?? "" : "");
  const POLL_MS = 15000;
  const FF_POLL_MS = 1000;

  // every key the component names, spelled out (a test checks the catalogs for each one)
  const KIND_KEYS = {
    touch: "page.growth.exp.k.touch",
    poke: "page.growth.exp.k.poke",
    lift: "page.growth.exp.k.lift",
    hand: "page.growth.exp.k.hand",
    name: "page.growth.exp.k.name",
    voice: "page.growth.exp.k.voice",
    hello: "page.growth.exp.k.hello",
    noise: "page.growth.exp.k.noise",
    company: "page.growth.exp.k.company",
    visit: "page.growth.exp.k.visit",
    play: "page.growth.exp.k.play",
    fall: "page.growth.exp.k.fall",
    sleep: "page.growth.exp.k.sleep",
    dock: "page.growth.exp.k.dock",
    feed: "page.growth.exp.k.feed",
    voiced: "page.growth.exp.k.voiced",
    explore: "page.growth.exp.k.explore",
    teach_trick: "page.growth.exp.k.teach_trick",
    praise: "page.growth.exp.k.praise",
    scold: "page.growth.exp.k.scold",
  };
  const KIND_ORDER = Object.keys(KIND_KEYS);
  const FALL_KEYS = {
    easy: "page.growth.exp.fall.easy",
    hard: "page.growth.exp.fall.hard",
    stuck: "page.growth.exp.fall.stuck",
    helped: "page.growth.exp.fall.helped",
  };
  const IGNORED_KEYS = {
    thing: "page.growth.exp.ignored.thing",
    person: "page.growth.exp.ignored.person",
    noise_self: "page.growth.exp.ignored.noise_self",
    lift_false: "page.growth.exp.ignored.lift_false",
    hand_false: "page.growth.exp.ignored.hand_false",
    company_false: "page.growth.exp.ignored.company_false",
    lie_down_self: "page.growth.exp.ignored.lie_down_self",
    explore_held: "page.growth.exp.ignored.explore_held",
  };
  const IGNORED_OTHER = "page.growth.exp.ignored.other";
  const HOW_KEYS = {
    liked: "page.growth.voice.how.liked",
    promoted: "page.growth.voice.how.promoted",
    song: "page.growth.voice.how.song",
    first_word: "page.growth.voice.how.first_word",
  };
  const WAIT_KEYS = {
    awake: "page.growth.regrow.wait.awake",
    settling: "page.growth.regrow.wait.settling",
    fallen: "page.growth.regrow.wait.fallen",
    posture: "page.growth.regrow.wait.posture",
    skill: "page.growth.regrow.wait.skill",
    event: "page.growth.regrow.wait.event",
    reflex: "page.growth.regrow.wait.reflex",
    escape: "page.growth.regrow.wait.escape",
    emotion: "page.growth.regrow.wait.emotion",
    recent: "page.growth.regrow.wait.recent",
    play: "page.growth.regrow.wait.play",
    motion: "page.growth.regrow.wait.motion",
    flight: "page.growth.regrow.wait.flight",
  };
  const STYLE_KEYS = {
    attentive: "page.growth.ff.style.attentive",
    neglectful: "page.growth.ff.style.neglectful",
    rough: "page.growth.ff.style.rough",
    praising: "page.growth.ff.style.praising",
  };
  const PREF_KEYS = {
    verb: "page.growth.learn.pref_verb",
    moment: "page.growth.learn.pref_moment",
    phrase: "page.growth.learn.pref_phrase",
  };

  /* ---------------- small helpers ---------------- */

  function h(tag, attrs, ...kids) {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(attrs || {})) {
      if (value === undefined || value === null || value === false) continue;
      if (name === "class") node.className = value;
      else if (name === "text") node.textContent = value;
      else if (name.startsWith("on") && typeof value === "function") node.addEventListener(name.slice(2), value);
      else node.setAttribute(name, value === true ? "" : String(value));
    }
    for (const kid of kids.flat()) {
      if (kid === undefined || kid === null || kid === false) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }
  const num = (value) => (I18N ? I18N.number(value) : String(value));
  const pad = (n) => String(n).padStart(2, "0");
  function fmtDate(iso) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "";
    const zh = (x) => `${x.getFullYear()}-${pad(x.getMonth() + 1)}-${pad(x.getDate())}`;
    return I18N ? I18N.date(d, { year: "numeric", month: "2-digit", day: "2-digit" }, zh) : zh(d);
  }
  function fmtWhen(iso) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "";
    const zh = (x) => `${pad(x.getMonth() + 1)}-${pad(x.getDate())} ${pad(x.getHours())}:${pad(x.getMinutes())}`;
    return I18N ? I18N.date(d, { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }, zh) : zh(d);
  }
  const bar = (pct, cls) => h("div", { class: `bar ${cls || ""}`.trim(), role: "presentation" }, h("i", { style: `width:${Math.max(0, Math.min(100, pct))}%` }));
  const joinList = (items) => (I18N ? I18N.list(items) : items.join(", "));
  const card = (accent, titleKey, ...kids) =>
    h("section", { class: `card gv-card accent-${accent}` }, h("h3", { class: "card-title", text: T(titleKey) }), ...kids);
  const hint = (text) => h("p", { class: "hint", text });
  const meter = (label, pct, shown) =>
    h("div", { class: "gv-meter" }, h("span", { class: "gv-meter-name", text: label }), bar(pct), h("span", { class: "val", text: shown === undefined ? `${num(pct)}%` : shown }));
  const sub = (key) => h("h4", { class: "gv-sub", text: T(key) });
  const labelOf = (obj) => F(obj, "label");

  /* ---------------- the cards ---------------- */

  function idCard(view) {
    const id = view.identity;
    const live = view.live !== false;
    const branch = id.branch || {};
    const sourceKey = { family: "page.growth.id.branch_family", growth: "page.growth.id.branch_growth", none: "page.growth.id.branch_none" }[branch.source] || "page.growth.id.branch_none";
    const next = id.next_stage;
    return card(
      "yellow",
      "page.growth.id.title",
      h("div", { class: "gv-id" },
        h("div", { class: "gv-gd" }, h("span", { class: "gv-gd-num", text: num(id.growth_day) }), h("span", { class: "gv-gd-label", text: T("page.growth.id.growth_day") })),
        h("div", { class: "gv-id-main" },
          live ? h("p", { class: "gv-name", text: id.name }) : null,
          h("div", { class: "gv-chips" },
            h("span", { class: "chip stage", text: F(id.stage, "label") }),
            branch.label ? h("span", { class: "chip gd", text: F(branch, "label") }) : null,
          ),
          h("p", { class: "gv-line", text: T(sourceKey) }),
          live ? h("p", { class: "gv-line", text: T("page.growth.id.hatched", { date: fmtDate(id.hatched), days: id.age_days }) }) : null,
          id.title ? h("p", { class: "gv-line", text: T("page.growth.id.epithet", { title: F(id, "title") }) }) : null,
        ),
      ),
      h("p", { class: "gv-line", text: next ? T("page.growth.id.next_stage", { days: next.in_days, gd: next.gd, stage: F(next, "label") }) : T("page.growth.id.grown_up") }),
      h("ol", { class: "gv-stage-list" },
        (id.stages || []).map((stage) =>
          h("li", { class: stage.id === id.stage.id ? "is-now" : "", "aria-current": stage.id === id.stage.id ? "step" : null },
            h("span", { text: F(stage, "label") }),
            h("span", { class: "val", text: T("page.growth.id.stage_from", { gd: stage.gd }) }),
          ),
        ),
      ),
      (view.milestones || []).length
        ? h("div", {}, sub("page.growth.id.milestones"),
            h("ul", { class: "gv-list" },
              view.milestones.slice(-6).map((m) => h("li", { text: T("page.growth.id.milestone", { gd: m.gd, title: F(m, "title") }) }))))
        : null,
    );
  }

  function temperamentCard(view) {
    const temp = view.temperament || { axes: [], nature: [] };
    const quirks = (view.quirks || { items: [], hidden: 0 });
    const lines = quirks.items.map((q) => {
      if (q.key === "tilt_side") return T("page.growth.quirk.tilt", { side: F(q, "side") });
      if (q.key === "favorite_tag") return T("page.growth.quirk.tag", { tag: F(q, "tag") });
      return F(q, "label");
    });
    return card(
      "cyan",
      "page.growth.temp.title",
      hint(T("page.growth.temp.hint")),
      h("div", { class: "gv-meters" }, temp.axes.map((a) => meter(`${F(a, "name")} · ${F(a, "word")}`, a.pct))),
      sub("page.growth.temp.leaning"),
      h("div", { class: "gv-meters" }, temp.nature.map((n) => meter(F(n, "label"), n.pct))),
      sub("page.growth.quirk.title"),
      lines.length ? h("ul", { class: "gv-list" }, lines.map((l) => h("li", { text: l }))) : hint(T("page.growth.quirk.none")),
      quirks.hidden ? hint(T("page.growth.quirk.hidden", { n: quirks.hidden })) : null,
    );
  }

  function learnedCard(view) {
    const learned = view.learned;
    const prefs = learned.prefs || [];
    const mind = learned.mind;
    // (MAP-EXPLORE) how much of its home its own map has seen: absent with the map off, empty before it has seen any floor
    const home = learned.home;
    return card(
      "orange",
      "page.growth.learn.title",
      sub("page.growth.learn.likes"),
      prefs.length
        ? h("div", { class: "gv-meters" }, prefs.map((p) => meter(T(PREF_KEYS[p.kind] || PREF_KEYS.verb, { label: F(p, "label") }), p.pct, `${num(p.pct)}`)))
        : hint(T("page.growth.learn.likes_none")),
      hint(T("page.growth.learn.likes_hint")),
      sub("page.growth.learn.avoid"),
      learned.avoid_spots.length
        ? h("ul", { class: "gv-list" }, learned.avoid_spots.map((s) => h("li", { text: T("page.growth.learn.avoid_spot", { x: num(s.x), y: num(s.y) }) })))
        : hint(T("page.growth.learn.avoid_none")),
      sub("page.growth.learn.counts_title"),
      h("p", { class: "gv-line", text: T("page.growth.learn.counts", { touches: learned.touches, pickups: learned.pickups, falls: learned.falls, meters: num(learned.walked_m) }) }),
      mind
        ? h("p", { class: "gv-line", text: T("page.growth.learn.mind", { things: mind.things, places: mind.places, liked: mind.places_liked, disliked: mind.places_disliked, avoided: mind.places_avoided }) })
        : null,
      sub("page.growth.learn.home_title"),
      !home
        ? hint(T("page.growth.learn.home_off"))
        : home.share_pct === null || home.share_pct === undefined
          ? hint(T("page.growth.learn.home_none"))
          : h("div", { class: "gv-meters" }, meter(T("page.growth.learn.home_share"), home.share_pct, `${num(home.share_pct)}%`)),
      home && home.share_pct !== null && home.share_pct !== undefined
        ? h("p", { class: "gv-line", text: T("page.growth.learn.home_line", { m2: num(home.known_m2), places: home.places, places_new: home.found }) })
        : null,
      home ? hint(T("page.growth.learn.home_hint")) : null,
    );
  }

  function bondCard(view) {
    const bonds = view.bonds;
    return card(
      "magenta",
      "page.growth.bond.title",
      meter(T("page.growth.bond.family"), bonds.family_pct),
      hint(T("page.growth.bond.family_hint")),
      sub("page.growth.bond.people"),
      bonds.people.length
        ? h("div", { class: "gv-people" }, bonds.people.map((p) =>
            h("div", { class: "gv-person" },
              h("div", { class: "gv-meter" }, h("span", { class: "gv-meter-name gv-data", text: p.name }), bar(p.pct), h("span", { class: "val", text: `${num(p.pct)}%` })),
              h("p", { class: "hint", text: T("page.growth.bond.person_stats", { greeted: p.greeted, exchanges: p.exchanges, sightings: p.sightings }) }),
            )))
        : hint(T("page.growth.bond.none")),
      bonds.unnamed ? hint(T("page.growth.bond.unnamed", { n: bonds.unnamed })) : null,
    );
  }

  function recentText(item) {
    const what = T(KIND_KEYS[item.kind] || KIND_KEYS.explore);
    let detail = "";
    if (item.minutes !== undefined) detail = T("page.growth.exp.minutes", { n: item.minutes });
    else if (item.outcome && FALL_KEYS[item.outcome]) detail = T(FALL_KEYS[item.outcome]);
    return detail ? T("page.growth.exp.with", { what, detail }) : what;
  }

  function experienceCard(view) {
    const exp = view.experiences;
    const ignored = Object.entries(exp.ignored || {}).filter(([, n]) => n > 0);
    const total = ignored.reduce((sum, [, n]) => sum + n, 0);
    return card(
      "blue",
      "page.growth.exp.title",
      hint(T("page.growth.exp.intro")),
      h("p", { class: "gv-line", text: T("page.growth.exp.total", { n: exp.total }) }),
      exp.total
        ? h("ul", { class: "gv-tally" }, KIND_ORDER.filter((k) => exp.by_kind[k]).map((k) =>
            h("li", {}, h("span", { text: T(KIND_KEYS[k]) }), h("b", { class: "val", text: num(exp.by_kind[k]) }))))
        : hint(T("page.growth.exp.none")),
      sub("page.growth.exp.recent"),
      exp.recent.length
        ? h("ul", { class: "gv-list" }, exp.recent.map((r) => h("li", { text: T("page.growth.exp.recent_line", { when: fmtWhen(r.at), what: recentText(r) }) })))
        : hint(T("page.growth.exp.recent_none")),
      sub("page.growth.exp.ignored_title"),
      total
        ? h("p", { class: "gv-line", text: T("page.growth.exp.ignored_list", { by: joinList(ignored.map(([by, n]) => T(IGNORED_KEYS[by] || IGNORED_OTHER, { n }))) }) })
        : hint(T("page.growth.exp.ignored_none")),
      exp.refused ? hint(T("page.growth.exp.refused", { n: exp.refused })) : null,
    );
  }

  function stageLabel(view, id) {
    const found = ((view.identity || {}).stages || []).find((s) => s.id === id);
    return found ? F(found, "label") : String(id);
  }

  function branchLabel(view, id) {
    const found = (((view.temperament || {}).nature) || []).find((n) => n.id === id);
    return found ? F(found, "label") : String(id || "-");
  }

  /* the family's bond is kept as 2000..10000; the page says percent */
  const bondPct = (raw) => `${I18N ? I18N.number(raw / 100, 1) : raw / 100}%`;

  function regrowLine(view, change, when) {
    const care = change.care || { bond: [0, 0], prefs: 0 };
    const parts = [
      T("page.growth.regrow.change", {
        when,
        from: change.gd[0],
        to: change.gd[1],
        stageFrom: stageLabel(view, change.stage[0]),
        stageTo: stageLabel(view, change.stage[1]),
        unlocked: (change.unlocked || []).length,
        params: change.params_moved,
        gains: change.gains_moved,
        bondFrom: bondPct(care.bond[0]),
        bondTo: bondPct(care.bond[1]),
        prefs: care.prefs,
      }),
    ];
    if (change.branch && change.branch[0] !== change.branch[1]) {
      parts.push(T("page.growth.regrow.branch_changed", { from: branchLabel(view, change.branch[0]), to: branchLabel(view, change.branch[1]) }));
    }
    return parts.join(" ");
  }

  function regrowCard(view) {
    const r = view.regrow;
    if (!r) return card("purple", "page.growth.regrow.title", hint(T("page.growth.regrow.off")));
    const last = r.last;
    const genome = r.genome;
    return card(
      "purple",
      "page.growth.regrow.title",
      hint(T("page.growth.regrow.intro")),
      h("p", { class: "gv-line", text: T("page.growth.regrow.count", { n: r.count }) }),
      r.due
        ? h("p", { class: "gv-line gv-wait", text: T("page.growth.regrow.waiting", { why: T(WAIT_KEYS[r.waiting_for] || WAIT_KEYS.settling), gd: r.growth_day }) })
        : h("p", { class: "gv-line", text: T("page.growth.regrow.up_to_date", { gd: r.applied_gd === null || r.applied_gd === undefined ? r.growth_day : r.applied_gd }) }),
      r.problem ? h("p", { class: "gv-line gv-warn", text: T("page.growth.regrow.problem", { text: String(r.problem) }) }) : null,
      sub("page.growth.regrow.last_title"),
      last
        ? h("div", {},
            h("p", { class: "gv-line", text: regrowLine(view, last, fmtWhen(last.at)) }),
            h("p", { class: "hint", text: T("page.growth.regrow.waited", { s: num(Math.round(last.waited_s || 0)) }) }))
        : hint(T("page.growth.regrow.none_yet")),
      genome
        ? h("p", { class: "gv-line", text: T("page.growth.regrow.body", { gd: genome.gd, stage: stageLabel(view, genome.stage), leash: num(genome.leash_m), wobble: num(genome.hatchling_wobble), unlocked: genome.unlocked }) })
        : null,
      r.history && r.history.length > 1
        ? h("div", {}, sub("page.growth.regrow.history"),
            h("ul", { class: "gv-list" }, r.history.slice(1).map((c) => h("li", { text: regrowLine(view, c, fmtWhen(c.at)) }))))
        : null,
      hint(T("page.growth.regrow.note")),
    );
  }

  function playCall(clip, note) {
    const voice = window.duckVoice;
    const ok = !!(voice && typeof voice.call === "function" && voice.call([clip]));
    if (!ok) {
      if (voice && typeof voice.refresh === "function") voice.refresh();
      note.textContent = T("page.growth.voice.play_wait");
    } else {
      note.textContent = "";
    }
  }

  function voiceCard(view) {
    const voice = view.voice;
    const live = view.live !== false;
    const note = h("p", { class: "hint gv-play-note", role: "status" });
    const playButton = (call) =>
      live && call.playable
        ? h("button", { class: "btn small", type: "button", text: T("page.growth.voice.play"), "aria-label": T("page.growth.voice.play_aria", { words: F(call, "words") }), onclick: () => playCall(call.clip, note) })
        : null;
    return card(
      "cyan",
      "page.growth.voice.title",
      h("p", { class: "gv-line", text: voice.first_word ? T("page.growth.voice.first_word", { word: F(voice, "first_word") }) : T("page.growth.voice.first_word_none") }),
      h("p", { class: "gv-line", text: T("page.growth.voice.lineage", { n: voice.lineage_size }) }),
      sub("page.growth.voice.calls"),
      voice.calls.length
        ? h("ul", { class: "gv-calls" }, voice.calls.map((call) =>
            h("li", {},
              h("span", { class: "gv-call-words", text: F(call, "words") }),
              h("span", { class: "hint", text: T(HOW_KEYS[call.how] || HOW_KEYS.liked) }),
              playButton(call))))
        : hint(T("page.growth.voice.none")),
      voice.song
        ? h("div", {}, sub("page.growth.voice.song"),
            h("p", { class: "gv-line", text: F(voice.song, "words") }),
            live && voice.song.playable
              ? h("button", { class: "btn small", type: "button", text: T("page.growth.voice.play"), "aria-label": T("page.growth.voice.play_song_aria"), onclick: () => playCall(voice.song.clip, note) })
              : null)
        : null,
      live ? note : hint(T("page.growth.voice.test_silent")),
    );
  }

  function diaryCard(view) {
    const days = view.diary || [];
    return card(
      "yellow",
      "page.growth.diary.title",
      days.length
        ? h("div", { class: "gv-diary" }, days.map((day) => {
            const lived = KIND_ORDER.filter((k) => day.lived[k]).map((k) => T("page.growth.diary.lived_item", { what: T(KIND_KEYS[k]), n: day.lived[k] }));
            return h("div", { class: "gv-day" },
              h("h4", { class: "gv-sub", text: day.gd === null ? fmtDate(day.date) : T("page.growth.diary.day", { date: fmtDate(day.date), gd: day.gd }) }),
              lived.length ? h("p", { class: "hint", text: T("page.growth.diary.lived", { items: joinList(lived) }) }) : null,
              day.lines.length ? h("ul", { class: "gv-list" }, day.lines.map((line) => h("li", { class: `tone-${line.tone}`, text: F(line, "text") }))) : null,
              day.more ? hint(T("page.growth.diary.more", { n: day.more })) : null);
          }))
        : hint(T("page.growth.diary.none")),
    );
  }

  /* One growth view, as a grid of cards. */
  function renderView(view, style) {
    const live = view.live !== false;
    const title = live ? null : h("h3", { class: "gv-result-title", text: T("page.growth.ff.result_title", { style: T(STYLE_KEYS[style] || STYLE_KEYS.attentive) }) });
    return h("div", { class: "gv-block" }, title, h("div", { class: "gv-grid" }, [idCard(view), bondCard(view), learnedCard(view), temperamentCard(view), experienceCard(view), regrowCard(view), voiceCard(view), diaryCard(view)]));
  }

  /* ---------------- the comparison of two test ducks ---------------- */

  const ROWS = [
    ["page.growth.cmp.growth_day", (v) => v.identity.growth_day, null],
    ["page.growth.cmp.stage", (v) => F(v.identity.stage, "label"), null],
    ["page.growth.cmp.branch", (v) => (v.identity.branch && v.identity.branch.label ? F(v.identity.branch, "label") : "-"), null],
    ["page.growth.cmp.bond", (v) => v.bonds.family_pct, "%"],
    ["page.growth.cmp.likes", (v) => v.learned.likes, null],
    ["page.growth.cmp.avoid", (v) => v.learned.avoid_spots.length, null],
    ["page.growth.cmp.falls", (v) => v.learned.falls, null],
    ["page.growth.cmp.touches", (v) => v.learned.touches, null],
    ["page.growth.cmp.calls", (v) => v.voice.calls.length, null],
    ["page.growth.cmp.experiences", (v) => v.experiences.total, null],
    ["page.growth.cmp.regrows", (v) => (v.regrow ? v.regrow.count : 0), null],
  ];

  /* The numbers of two (or one) test ducks side by side: [{key, cells:[...], differs}] */
  function compareRows(views) {
    return ROWS.map(([key, get, unit]) => {
      const raw = views.map(get);
      return { key, cells: raw.map((value) => (typeof value === "number" ? `${num(value)}${unit || ""}` : String(value))), differs: new Set(raw.map(String)).size > 1 };
    });
  }

  function compareTable(styles, views) {
    const rows = compareRows(views);
    return h("div", { class: "gv-compare table-wrap" },
      h("table", {},
        h("thead", {}, h("tr", {}, h("th", { text: T("page.growth.cmp.what") }), styles.map((s) => h("th", { text: T(STYLE_KEYS[s] || STYLE_KEYS.attentive) })))),
        h("tbody", {}, rows.map((row) => h("tr", { class: row.differs ? "differs" : "" }, h("th", { scope: "row", text: T(row.key) }), row.cells.map((c) => h("td", { text: c })))))));
  }

  /* ---------------- the page ---------------- */

  const els = {};
  const state = { live: null, liveKey: "", ff: null, results: null, polling: null, ffPolling: null, error: false };

  function paintLive() {
    const root = els.live;
    if (!root) return;
    root.textContent = "";
    if (state.error && !state.live) {
      els.status.textContent = T("page.growth.offline");
      return;
    }
    if (!state.live) {
      els.status.textContent = state.loaded ? T("page.growth.off") : T("page.growth.loading");
      return;
    }
    els.status.textContent = T("page.growth.updated", { time: fmtWhen(state.live.calendar) });
    root.append(renderView(state.live, ""));
  }

  function fetchJson(url, options) {
    return fetch(url, Object.assign({ cache: "no-store" }, options || {})).then((response) => response.json().then((data) => ({ status: response.status, data })));
  }

  function visible() {
    return document.body.classList.contains("view-data") && !document.hidden;
  }

  function refreshLive() {
    if (!visible()) return;
    fetchJson("/api/life/growth")
      .then(({ data }) => {
        state.error = false;
        state.loaded = true;
        const view = data && data.growth ? data.growth : null;
        const key = JSON.stringify(view, (name, value) => (name === "calendar" ? undefined : value));
        if (key === state.liveKey && state.live) {
          els.status.textContent = T("page.growth.updated", { time: fmtWhen(view.calendar) });
          return;
        }
        state.liveKey = key;
        state.live = view;
        paintLive();
      })
      .catch(() => {
        state.error = true;
        paintLive();
      });
  }

  /* ---- the fast-forward ---- */

  function ffStyles() {
    const a = els.styleA.value;
    const b = els.styleB.value;
    return b && b !== a ? [a, b] : [a];
  }

  /* A day of a hatchling that cannot walk is cheap (about a third of a walking day's work): the
     estimate weighs the days still to come by that, so it does not promise too little early on. */
  const DAY_WEIGHT = (day) => (day < 5 ? 1 : 3.2);
  function etaSeconds(elapsed, done, total) {
    if (!(done > 0) || !(elapsed > 0)) return null;
    let spent = 0;
    let left = 0;
    for (let day = 0; day < total; day += 1) (day < done ? (spent += DAY_WEIGHT(day)) : (left += DAY_WEIGHT(day)));
    return Math.round((elapsed / spent) * left);
  }

  function paintFf() {
    const status = state.ff;
    const running = !!status && status.state === "running";
    els.start.disabled = running;
    els.cancel.disabled = !running;
    els.progress.textContent = "";
    if (!status || status.state === "idle") {
      els.note.textContent = "";
      return;
    }
    for (const run of status.runs || []) {
      const label = T(STYLE_KEYS[run.style] || STYLE_KEYS.attentive);
      const text = run.state === "failed" ? T("page.growth.ff.run_failed", { error: run.error || "" }) : run.state === "cancelled" ? T("page.growth.ff.run_cancelled") : T("page.growth.ff.run_day", { day: run.day, days: run.days });
      els.progress.append(
        h("div", { class: "gv-run" },
          h("span", { class: "gv-run-name", text: label }),
          h("div", { class: "bar", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": run.pct, "aria-label": label }, h("i", { style: `width:${run.pct}%` })),
          h("span", { class: "val", text: text })));
    }
    if (status.state === "running") {
      const slowest = Math.max(...status.runs.map((r) => r.day), 0);
      const eta = etaSeconds(status.elapsed_s, slowest, status.days);
      els.note.textContent = eta === null ? T("page.growth.ff.running") : T("page.growth.ff.running_eta", { s: num(eta) });
    } else if (status.state === "done" && status.speed) {
      els.note.textContent = T("page.growth.ff.done", { days: status.days, seconds: num(status.elapsed_s), rate: num(status.speed.days_per_s) });
    } else if (status.state === "cancelled") {
      els.note.textContent = T("page.growth.ff.cancelled");
    } else if (status.state === "failed") {
      els.note.textContent = T("page.growth.ff.failed");
    }
  }

  function paintResults() {
    const root = els.results;
    root.textContent = "";
    const results = state.results;
    if (!results) return;
    const styles = Object.keys(results);
    if (!styles.length) return;
    const views = styles.map((s) => results[s].view);
    root.append(h("h3", { class: "gv-result-title", text: T("page.growth.cmp.title") }));
    if (styles.length > 1) root.append(compareTable(styles, views));
    root.append(h("div", { class: `gv-pair gv-pair-${styles.length}` }, styles.map((s, i) => renderView(views[i], s))));
  }

  function applyFf(status) {
    state.ff = status;
    paintFf();
    const running = status && status.state === "running";
    if (running && !state.ffPolling) state.ffPolling = setInterval(pollFf, FF_POLL_MS);
    if (!running && state.ffPolling) {
      clearInterval(state.ffPolling);
      state.ffPolling = null;
    }
    if (status && status.state === "done" && (!state.results || state.resultsId !== status.id)) {
      fetchJson("/api/life/fastforward?results=1").then(({ data }) => {
        state.results = (data.fastforward && data.fastforward.results) || null;
        state.resultsId = status.id;
        paintResults();
      });
    }
  }

  function pollFf() {
    fetchJson("/api/life/fastforward").then(({ data }) => applyFf(data.fastforward)).catch(() => {});
  }

  function startFf() {
    const days = Number.parseInt(els.days.value, 10);
    const seed = Number.parseInt(els.seed.value, 10);
    els.problem.textContent = "";
    fetchJson("/api/life/fastforward", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ days, styles: ffStyles(), seed }),
    })
      .then(({ status, data }) => {
        if (status >= 400 || !data.ok) {
          els.problem.textContent = F(data, "error") || T("page.growth.ff.failed");
          return;
        }
        state.results = null;
        paintResults();
        applyFf(data.fastforward);
      })
      .catch(() => {
        els.problem.textContent = T("page.growth.offline");
      });
  }

  function cancelFf() {
    fetchJson("/api/life/fastforward/cancel", { method: "POST" }).then(({ data }) => applyFf(data.fastforward)).catch(() => {});
  }

  function repaint() {
    paintLive();
    paintFf();
    paintResults();
  }

  function init() {
    els.live = document.getElementById("gvLive");
    if (!els.live) return;
    els.status = document.getElementById("gvStatus");
    els.days = document.getElementById("gvFfDays");
    els.styleA = document.getElementById("gvFfStyleA");
    els.styleB = document.getElementById("gvFfStyleB");
    els.seed = document.getElementById("gvFfSeed");
    els.start = document.getElementById("gvFfStart");
    els.cancel = document.getElementById("gvFfCancel");
    els.progress = document.getElementById("gvFfProgress");
    els.note = document.getElementById("gvFfNote");
    els.problem = document.getElementById("gvFfProblem");
    els.results = document.getElementById("gvResults");
    els.start.addEventListener("click", startFf);
    els.cancel.addEventListener("click", cancelFf);
    if (I18N && typeof I18N.onChange === "function") I18N.onChange(repaint);
    window.addEventListener("duck:view", () => {
      refreshLive();
      pollFf();
    });
    document.addEventListener("visibilitychange", refreshLive);
    state.polling = setInterval(refreshLive, POLL_MS);
    refreshLive();
    pollFf();
  }

  const api = { renderView, compareRows, regrowLine, etaSeconds, KIND_KEYS, WAIT_KEYS, STYLE_KEYS, HOW_KEYS, FALL_KEYS, IGNORED_KEYS, PREF_KEYS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else window.duckGrowth = api;
  if (typeof document !== "undefined" && document.getElementById) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
  }
})();
