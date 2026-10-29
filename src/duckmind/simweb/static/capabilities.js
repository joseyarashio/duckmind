/* R1/R2 simulator evidence browser. This reads archived gate evidence only; it never commands robotd. */
(function () {
  "use strict";

  const API = "/api/skill-evidence";
  const SKILL_ORDER = ["march", "bow", "hop", "turn", "lie_low"];
  const STATUS_KEYS = {
    regressed: "page.cap.status.regressed",
    candidate: "page.cap.status.candidate",
    limited: "page.cap.status.limited",
    unverified: "page.cap.status.unverified",
  };
  const RECOMMEND_KEYS = {
    r1: "page.cap.recommend.r1",
    r2_stand_only: "page.cap.recommend.r2_stand_only",
    none: "page.cap.recommend.none",
  };
  const SKILL_KEYS = Object.fromEntries(SKILL_ORDER.map((id) => [id, `page.cap.skill.${id}`]));
  const REASON_KEYS = {
    yaw: "page.cap.reason.yaw",
    housing: "page.cap.reason.housing",
    fall: "page.cap.reason.fall",
    steps: "page.cap.reason.steps",
    post_fall: "page.cap.reason.post_fall",
  };

  const byId = (id) => document.getElementById(id);
  const escText = (node, value) => { node.textContent = value == null ? "" : String(value); };
  const t = (key, params = {}) => window.I18N?.t(key, params) || key;
  const add = (parent, tag, className, text) => {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) escText(el, text);
    parent.appendChild(el);
    return el;
  };

  let evidence = null;
  let selectedSkill = null;
  let selectedCondition = null;
  let videos = [];
  let syncUntil = 0;
  let loadError = null;

  function stopVideos() {
    for (const video of videos) {
      video.pause();
      video.removeAttribute("src");
      video.load();
    }
    videos = [];
  }

  function skillName(id) { return t(SKILL_KEYS[id] || "page.cap.skill.unknown"); }
  function statusName(status) { return t(STATUS_KEYS[status] || STATUS_KEYS.unverified); }
  function conditionLabel(condition, skillId) {
    const servo = t(`page.cap.servo.${condition.servo}`);
    const entry = condition.entry === "all" ? t("page.cap.entry.all") : t(`page.cap.entry.${condition.entry}`);
    const base = condition.height_mm == null
      ? t("page.cap.condition.flat", { servo, delay: condition.delay_ms, entry })
      : t("page.cap.condition.format", {
        servo, delay: condition.delay_ms, entry, height: String(condition.height_mm),
      });
    const directionKey = skillId === "turn" ? {
      left: "page.cap.direction.left",
      right: "page.cap.direction.right",
    }[condition.direction] : null;
    return directionKey ? `${base} · ${t(directionKey)}` : base;
  }

  function renderSkillCards() {
    const root = byId("capSkills");
    root.replaceChildren();
    const items = [...(evidence?.skills || [])].sort((a, b) => {
      const ai = SKILL_ORDER.indexOf(a.id), bi = SKILL_ORDER.indexOf(b.id);
      return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
    });
    for (const skill of items) {
      const button = add(root, "button", "btn cap-skill-card", undefined);
      button.type = "button";
      button.setAttribute("aria-pressed", String(skill.id === selectedSkill));
      add(button, "span", "cap-skill-name", skillName(skill.id));
      const meta = add(button, "span", "cap-skill-meta", undefined);
      add(meta, "span", `cap-status ${skill.status || "unverified"}`, statusName(skill.status));
      button.addEventListener("click", () => {
        selectedSkill = skill.id;
        const first = skill.conditions?.[0];
        selectedCondition = first?.id || null;
        render();
      });
    }
  }

  function selectedItem() {
    return evidence?.skills?.find((item) => item.id === selectedSkill) || null;
  }

  function renderConditionSelect(skill) {
    const select = byId("capCondition");
    select.replaceChildren();
    const conditions = skill?.conditions || [];
    for (const condition of conditions) {
      const option = document.createElement("option");
      option.value = condition.id;
      escText(option, conditionLabel(condition, skill?.id));
      select.appendChild(option);
    }
    select.disabled = conditions.length === 0;
    if (conditions.length) {
      if (!conditions.some((item) => item.id === selectedCondition)) selectedCondition = conditions[0].id;
      select.value = selectedCondition;
    } else {
      selectedCondition = null;
    }
    select.onchange = () => {
      selectedCondition = select.value || null;
      renderCondition();
    };
  }

  function selectedConditionItem(skill) {
    return skill?.conditions?.find((condition) => condition.id === selectedCondition) || null;
  }

  function countPill(parent, key, value, n) {
    if (value == null && key !== "n") return;
    add(parent, "span", "cap-count", t(key, { value: value ?? "—", n: n ?? "—" }));
  }

  function renderVersion(condition, version, skill) {
    const stats = condition?.[version];
    const card = add(byId("capResults"), "article", `card cap-version-card accent-${version === "r1" ? "cyan" : "magenta"}`);
    const head = add(card, "div", "cap-version-title", undefined);
    add(head, "h3", "", version.toUpperCase());
    if (stats) add(head, "span", "cap-status", t("page.cap.sample_count", { n: stats.n }));
    else add(head, "span", "cap-status unverified", t("page.cap.no_result"));
    const recommendation = skill?.recommended || "none";
    add(card, "p", "cap-rec", t("page.cap.recommendation", { value: t(RECOMMEND_KEYS[recommendation] || RECOMMEND_KEYS.none) }));
    if (!stats) return;
    const counts = add(card, "div", "cap-counts", undefined);
    const metric = condition.metric === "crossed_by_skill_end" ? "page.cap.metric.crossed" : "page.cap.metric.success";
    countPill(counts, metric, stats.success, stats.n);
    countPill(counts, "page.cap.metric.falls", stats.falls, stats.n);
    countPill(counts, "page.cap.metric.housing", stats.housing, stats.n);
    countPill(counts, "page.cap.metric.post_falls", stats.post_falls, stats.n);
  }

  function renderVideos(skill, condition) {
    stopVideos();
    const root = byId("capVideos");
    root.replaceChildren();
    for (const version of ["r1", "r2"]) {
      const column = add(root, "section", `cap-video-column ${version}`, undefined);
      add(column, "h3", "", version.toUpperCase());
      const clips = (skill?.clips || []).filter((clip) => clip.version === version && clip.condition_id === condition?.id);
      if (!clips.length) {
        add(column, "p", "cap-empty", t("page.cap.clip.pending", { version: version.toUpperCase() }));
        continue;
      }
      for (const clip of clips) {
        const item = add(column, "article", "cap-video-item", undefined);
        const video = add(item, "video", "", undefined);
        video.controls = true;
        video.preload = "metadata";
        video.playsInline = true;
        video.setAttribute("aria-label", t("page.cap.video.aria", { version: version.toUpperCase(), skill: skillName(skill.id) }));
        const src = typeof clip.src === "string" ? clip.src : "";
        if (src.startsWith("/skill-evidence/") && src.endsWith(".mp4")) video.src = src;
        else {
          video.hidden = true;
          video.removeAttribute("controls");
          add(item, "p", "cap-empty", t("page.cap.clip.pending", { version: version.toUpperCase() }));
        }
        const videoError = add(item, "p", "cap-video-error", t("page.cap.clip.error"));
        videoError.hidden = true;
        const playPair = add(item, "button", "btn cap-play-pair", t("page.cap.clip.play_pair"));
        playPair.type = "button";
        playPair.disabled = video.hidden;
        playPair.addEventListener("click", playPairFromStart);
        video.addEventListener("error", () => {
          video.pause();
          video.hidden = true;
          playPair.disabled = true;
          videoError.hidden = false;
        });
        videos.push(video);
        const resultKey = clip.result === "success" ? "page.cap.clip.success" : "page.cap.clip.failure";
        const caption = add(item, "p", "cap-video-caption", t("page.cap.clip.caption", {
          result: t(resultKey), seed: clip.seed ?? "—", rep: clip.rep ?? "—",
          duration: clip.duration_s ?? "—", sha: clip.model_sha256 ? String(clip.model_sha256).slice(0, 12) : "—",
        }));
        caption.title = clip.model_sha256 || "";
        const reasons = Array.isArray(clip.reasons) ? clip.reasons.filter((reason) => REASON_KEYS[reason]) : [];
        if (reasons.length) {
          const tags = add(item, "div", "cap-reasons", undefined);
          for (const reason of reasons) add(tags, "span", "cap-reason", t(REASON_KEYS[reason]));
        }
        video.addEventListener("play", () => syncVideo(video, "play"));
        video.addEventListener("pause", () => syncVideo(video, "pause"));
        video.addEventListener("seeking", () => syncVideo(video, "seek"));
        video.addEventListener("timeupdate", () => syncVideo(video, "drift"));
      }
    }
  }

  function playPairFromStart() {
    const playable = videos.filter((video) => !video.hidden && video.hasAttribute("src") && !video.error);
    if (!playable.length) return;
    syncUntil = Date.now() + 350;
    for (const video of playable) {
      video.pause();
      try { video.currentTime = 0; } catch (_) { /* Metadata may still be loading. */ }
    }
    for (const video of playable) video.play().catch(() => {});
  }

  function syncVideo(source, action) {
    if (Date.now() < syncUntil) return;
    syncUntil = Date.now() + 350;
    for (const target of videos) {
      if (target === source) continue;
      if (action === "play") {
        if (Math.abs(target.currentTime - source.currentTime) > 0.15) target.currentTime = source.currentTime;
        if (target.paused) target.play().catch(() => {});
      } else if (action === "pause") {
        if (!target.paused) target.pause();
      } else if (action === "seek") {
        target.currentTime = source.currentTime;
      } else if (action === "drift" && !target.paused && Math.abs(target.currentTime - source.currentTime) > 0.35) {
        target.currentTime = source.currentTime;
      }
    }
  }

  function renderCondition() {
    const skill = selectedItem();
    const condition = selectedConditionItem(skill);
    escText(byId("capConditionDetail"), condition ? conditionLabel(condition, skill?.id) : t("page.cap.no_conditions"));
    byId("capResults").replaceChildren();
    renderVersion(condition, "r1", skill);
    renderVersion(condition, "r2", skill);
    renderVideos(skill, condition);
  }

  function render() {
    if (!evidence) {
      renderError();
      return;
    }
    renderSkillCards();
    const skill = selectedItem();
    renderConditionSelect(skill);
    escText(byId("capUpdated"), t("page.cap.updated", { date: evidence.generated_at || "—" }));
    renderCondition();
  }

  function renderError() {
    if (!loadError) return;
    byId("capError").hidden = false;
    escText(byId("capErrorText"), t("page.cap.load_error", { error: loadError }));
    escText(byId("capUpdated"), t("page.cap.no_data"));
  }

  async function load() {
    try {
      const response = await fetch(API, { cache: "no-store", headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      if (!payload || payload.schema_version !== 1 || !Array.isArray(payload.skills)) {
        throw new Error("schema_version 1 required");
      }
      evidence = payload;
      loadError = null;
      selectedSkill = SKILL_ORDER.find((id) => payload.skills.some((item) => item.id === id)) || payload.skills[0]?.id || null;
      const item = selectedItem();
      selectedCondition = item?.conditions?.[0]?.id || null;
      byId("capError").hidden = true;
      render();
    } catch (error) {
      loadError = error?.message || "unknown";
      renderError();
    }
  }

  document.addEventListener("DOMContentLoaded", async () => {
    if (window.I18N?.ready) await window.I18N.ready;
    byId("capRetry").addEventListener("click", load);
    // (the evidence and its clips belong to the 「資料」 page: read them only when it is open)
    if (document.body.classList.contains("view-data")) await load();
    else {
      const open = async (event) => {
        if (!event.detail || event.detail.view !== "data") return;
        window.removeEventListener("duck:view", open);
        await load();
      };
      window.addEventListener("duck:view", open);
    }
    window.I18N?.onChange?.(() => {
      if (evidence) render();
      else renderError();
    });
  });
})();
