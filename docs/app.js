"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const videos = Array.from(document.querySelectorAll("video[data-autoplay]"));
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const number = (value) => typeof value === "number" && Number.isFinite(value);
  const setText = (id, value) => { const node = $(id); if (node) node.textContent = value; };
  const playSafely = (video) => { const promise = video.play(); if (promise) promise.catch(() => {}); };

  if ("IntersectionObserver" in window) {
    const playback = new IntersectionObserver((entries) => {
      entries.forEach(({ target, isIntersecting }) => {
        if (isIntersecting && !reducedMotion.matches && document.visibilityState === "visible") playSafely(target);
        else target.pause();
      });
    }, { threshold: 0.25 });
    videos.forEach((video) => playback.observe(video));
  }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) videos.forEach((video) => video.pause());
  });
  $("replay-pair").addEventListener("click", () => {
    [$("video-before"), $("video-after")].forEach((video) => {
      video.currentTime = 0;
      playSafely(video);
    });
  });

  function localAsset(path) {
    if (typeof path !== "string") return null;
    try {
      const url = new URL(path, document.baseURI);
      return url.origin === location.origin && url.pathname.includes("/assets/videos/") ? url.href : null;
    } catch (_) { return null; }
  }

  function bindVideo(id, experiment, task, episode) {
    const video = $(id);
    const entry = experiment.videos.find((v) => v.task_id === task && v.episode_index === episode);
    if (!entry || entry.success !== true) throw new Error("Selected demonstration is not a verified successful episode");
    const source = localAsset(entry.src), poster = localAsset(entry.poster);
    if (!source || !poster) throw new Error("Invalid local demonstration path");
    // The HTML already provides the published snapshot; avoid interrupting playback when paths match.
    if (video.querySelector("source").src !== source) {
      video.querySelector("source").src = source;
      video.load();
    }
    video.poster = poster;
  }

  async function loadEvidence() {
    try {
      const response = await fetch("data/results.json", { cache: "no-cache" });
      if (!response.ok) throw new Error("Results unavailable");
      const data = await response.json();
      const base = data.experiments.find((e) => e.id === "phase1_baseline");
      const fast = data.experiments.find((e) => e.id === "sampling5_baseline");
      const lora = data.experiments.find((e) => e.id === "phase1_lora");
      const before = base?.latency.find((r) => r.num_steps === 10)?.median_ms;
      const after = fast?.latency.find((r) => r.num_steps === 5)?.median_ms;
      if (!data.status.complete || !number(before) || !number(after) || before <= 0) throw new Error("Incomplete evidence");
      const drop = 100 * (1 - after / before);
      setText("before-video-ms", before.toFixed(1));
      setText("after-video-ms", after.toFixed(1));
      setText("latency-before", before.toFixed(1));
      setText("latency-after", after.toFixed(1));
      setText("headline-reduction", `−${drop.toFixed(1)}%`);
      setText("lead-reduction", `降低 ${drop.toFixed(1)}%`);
      setText("lead-success", `${base.evaluation.pc_success.toFixed(0)}% → ${fast.evaluation.pc_success.toFixed(0)}%`);
      setText("headline-latency-pair", `${before.toFixed(1)} → ${after.toFixed(1)} ms`);
      setText("contribution-reduction", `${drop.toFixed(1)}%`);
      $("optimized-bar").style.width = `${Math.min(100, Math.max(0, 100 * after / before))}%`;
      setText("headline-success", `${fast.evaluation.n_success}/${fast.evaluation.n_episodes}`);
      setText("success-before", `${base.evaluation.pc_success.toFixed(0)}%`);
      setText("success-after", `${fast.evaluation.pc_success.toFixed(0)}%`);
      setText("headline-success-caption", `10 步 ${base.evaluation.pc_success.toFixed(0)}% / 5 步 ${fast.evaluation.pc_success.toFixed(0)}%`);
      setText("headline-episodes", data.experiments.reduce((sum, e) => sum + e.evaluation.n_episodes, 0).toLocaleString());
      const params = 100 * (1 - lora.training.trainable_parameters / base.training.trainable_parameters);
      setText("parameter-reduction", `${params.toFixed(1)}%`);
      bindVideo("video-before", base, 4, 0);
      bindVideo("video-after", fast, 4, 0);
      bindVideo("demo-spatial", fast, 0, 0);
      bindVideo("demo-box", fast, 3, 0);
      bindVideo("demo-drawer", fast, 4, 4);
    } catch (_) {
      const notice = $("data-message");
      notice.hidden = false;
      notice.textContent = "完整数据暂时无法加载；当前展示为已发布实验快照，可通过技术报告查阅记录。";
    }
  }
  loadEvidence();
})();
