"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = { data: null, experiments: [], videos: [], loading: false, videoLimit: 12, renderedVideos: 0 };
  const posterObserver = typeof IntersectionObserver === "function" ? new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        if (entry.target.dataset.lazyPoster) entry.target.poster = entry.target.dataset.lazyPoster;
        delete entry.target.dataset.lazyPoster;
        posterObserver.unobserve(entry.target);
      }
    });
  }, { rootMargin: "200px" }) : null;
  const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
  const list = (value) => Array.isArray(value) ? value : [];
  const text = (value, fallback = "待完成") => typeof value === "string" && value.trim() ? value : fallback;
  const count = (value) => isNumber(value) ? value.toLocaleString("zh-CN") : "—";
  const number = (value, digits = 1) => isNumber(value) ? value.toFixed(digits) : "—";
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  function element(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = content;
    return node;
  }

  function safeURL(value) {
    if (typeof value !== "string" || !value.trim()) return null;
    try {
      const url = new URL(value, document.baseURI);
      return ["http:", "https:"].includes(url.protocol) ? url.href : null;
    } catch (_) { return null; }
  }

  function configureLink(node, value) {
    const href = safeURL(value);
    node.hidden = !href;
    if (!href) return;
    node.href = href;
    if (new URL(href).origin !== location.origin) {
      node.target = "_blank";
      node.rel = "noopener noreferrer";
    } else {
      node.removeAttribute("target");
      node.removeAttribute("rel");
    }
  }

  function isCheck(experiment) {
    return /smoke|pipeline|check|流程/i.test(String(experiment.kind || "")) || /^smoke/i.test(String(experiment.id || ""));
  }

  function kindTag(experiment) {
    return element("span", `kind-tag${isCheck(experiment) ? " check" : ""}`, isCheck(experiment) ? "流程检查" : experiment.kind === "controlled_inference_ablation" ? "推理对照" : "训练配方");
  }

  function percentage(evaluation) {
    if (isNumber(evaluation?.pc_success)) return evaluation.pc_success;
    if (isNumber(evaluation?.n_success) && isNumber(evaluation?.n_episodes) && evaluation.n_episodes > 0) {
      return 100 * evaluation.n_success / evaluation.n_episodes;
    }
    return null;
  }

  function interval(evaluation) {
    const bounds = evaluation?.pc_success_ci95;
    return Array.isArray(bounds) && bounds.length === 2 && bounds.every(isNumber)
      ? `[${number(bounds[0])}%, ${number(bounds[1])}%]` : null;
  }

  function rateDescription(evaluation) {
    const rate = percentage(evaluation);
    if (rate === null) return "待评测";
    const trials = isNumber(evaluation?.n_success) && isNumber(evaluation?.n_episodes)
      ? `${count(evaluation.n_success)}/${count(evaluation.n_episodes)} 回合` : "回合数未记录";
    return `${number(rate)}% · ${trials}`;
  }

  function parameterCount(value) {
    if (!isNumber(value)) return "待记录";
    if (value >= 1e6) return `${number(value / 1e6, 2)} M`;
    if (value >= 1e3) return `${number(value / 1e3, 1)} K`;
    return count(value);
  }

  function duration(value) {
    if (!isNumber(value)) return "待记录";
    if (value < 60) return `${number(value, 1)} 秒`;
    if (value < 3600) return `${number(value / 60, 1)} 分钟`;
    return `${number(value / 3600, 2)} 小时`;
  }

  function cell(primary, secondary, className) {
    const td = element("td");
    td.append(element("span", className || "", primary));
    if (secondary) td.append(element("span", "cell-secondary", secondary));
    return td;
  }

  function setStat(id, value, suffix) {
    const node = $(id);
    node.replaceChildren(element("span", "", value));
    node.append(element("small", "", suffix));
  }

  function showCameraSample(container) {
    container.replaceChildren();
    const image = element("img");
    image.src = "assets/sample_cameras.png";
    image.alt = "真实轨迹中的主视角与腕部相机样例";
    image.addEventListener("error", () => {
      const empty = element("div", "media-placeholder");
      empty.append(element("span", "placeholder-frame", "＋"), element("p", "", "真实实验画面待生成"), element("span", "", "结果随实验记录更新"));
      container.replaceChildren(empty);
      $("hero-media-label").textContent = "等待实验画面";
      $("hero-caption").textContent = "视频和相机样例尚未发布。";
    }, { once: true });
    container.append(image);
    $("hero-media-label").textContent = "双相机轨迹样例";
    $("hero-caption").textContent = "主视角与腕部视角，来自轨迹数据中的同一时刻。";
  }

  function makeVideo(record) {
    const video = element("video");
    video.controls = true;
    video.preload = "none";
    video.playsInline = true;
    video.src = safeURL(record.src);
    const poster = safeURL(record.poster);
    if (poster) {
      if (posterObserver) { video.dataset.lazyPoster = poster; posterObserver.observe(video); }
      else video.poster = poster;
    }
    video.setAttribute("aria-label", text(record.caption, `任务 ${record.task_id ?? "待标注"} 的真实操作回合`));
    video.append(document.createTextNode("当前浏览器无法播放此视频，请查看项目中的原始记录。"));
    return video;
  }

  function renderHero(data) {
    const project = data.project || {};
    if (project.title) {
      $("project-title").textContent = project.title;
      document.title = `${project.title} · 实验记录`;
    }
    if (project.subtitle) $("project-subtitle").textContent = project.subtitle;
    configureLink($("header-repository"), project.repository_url);
    configureLink($("hero-repository"), project.repository_url);
    $("project-status").classList.toggle("is-complete", data.status?.complete === true);
    $("project-status").classList.remove("is-error");
    $("status-label").textContent = text(data.status?.label, "实验状态待更新");
    const updated = project.updated_at;
    if (updated) {
      const date = new Date(updated);
      $("updated-at").textContent = Number.isNaN(date.getTime()) ? `记录更新：${updated}`
        : `记录更新：${new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", timeZone: "America/Los_Angeles" }).format(date)}（洛杉矶）`;
    } else $("updated-at").textContent = "";
    const protocol = data.protocol || {};
    const tasks = Array.isArray(protocol.tasks) ? protocol.tasks.length : protocol.tasks;
    setStat("stat-tasks", isNumber(tasks) ? count(tasks) : "—", isNumber(tasks) ? "个任务" : "待记录");
    setStat("stat-episodes", `${count(protocol.train_episodes)} / ${count(protocol.heldout_episodes)}`, "条轨迹");
    setStat("stat-frames", count(protocol.train_frames), isNumber(protocol.train_frames) ? "帧" : "待记录");
    setStat("stat-seed", count(protocol.seed), isNumber(protocol.seed) ? "固定种子" : "待记录");
    const heroVideo = state.videos.find((v) => !isCheck(v.experiment)) || state.videos[0];
    if (heroVideo) {
      $("hero-visual").replaceChildren(makeVideo(heroVideo));
      $("hero-media-label").textContent = `${heroVideo.experiment.label || heroVideo.experiment.id} · ${isCheck(heroVideo.experiment) ? "流程检查" : "闭环回合"}`;
      $("hero-caption").textContent = text(heroVideo.caption, `任务 ${heroVideo.task_id ?? "待标注"}，真实仿真操作记录。`);
    } else showCameraSample($("hero-visual"));
  }

  function visibleExperiments() {
    return $("experiment-scope").value === "all" ? state.experiments : state.experiments.filter((experiment) => !isCheck(experiment));
  }

  function renderComparison() {
    const experiments = visibleExperiments();
    $("comparison-body").replaceChildren();
    $("success-chart").replaceChildren();
    $("experiment-notes").replaceChildren();
    $("comparison-empty").hidden = experiments.length > 0;
    $("comparison-content").hidden = experiments.length === 0;
    $("comparison-notice").hidden = !experiments.some(isCheck);
    experiments.forEach((experiment, index) => {
      const reused = experiment.kind === "controlled_inference_ablation";
      const training = experiment.training || {};
      const evaluation = experiment.evaluation || {};
      const row = element("tr");
      const method = cell(text(experiment.label, experiment.id || "实验方案"), null, "method-name");
      const tag = element("span", "cell-secondary"); tag.append(kindTag(experiment)); method.append(tag);
      const updates = isNumber(training.steps) ? `${count(training.steps)} / ${count(training.target_steps)}` : "待训练";
      const params = parameterCount(training.trainable_parameters);
      const ratio = isNumber(training.trainable_parameters) && isNumber(training.total_parameters) && training.total_parameters > 0
        ? `${number(100 * training.trainable_parameters / training.total_parameters, 2)}% / 共 ${parameterCount(training.total_parameters)}` : null;
      const rate = percentage(evaluation);
      const rateCell = cell(rate === null ? "待评测" : `${number(rate)}%`,
        rate === null ? null : `${isNumber(evaluation.n_success) && isNumber(evaluation.n_episodes) ? `${count(evaluation.n_success)}/${count(evaluation.n_episodes)} 回合` : "回合数未记录"}${interval(evaluation) ? ` · 95% CI ${interval(evaluation)}` : ""}`,
        rate === null ? "cell-pending" : "cell-success");
      row.append(method, cell(reused ? "沿用基线" : updates, reused ? "同一最终checkpoint" : "已记录 / 计划"), cell(params, ratio),
        cell(reused ? "沿用基线" : isNumber(training.max_logged_allocated_gib) ? `${number(training.max_logged_allocated_gib, 2)} GiB` : "待记录", "训练日志分配值"), cell(reused ? "无需重训" : duration(training.wall_s)), rateCell);
      $("comparison-body").append(row);
      const chartRow = element("div", "chart-row");
      const track = element("div", rate === null ? "chart-track pending" : "chart-track");
      track.setAttribute("role", "img");
      track.setAttribute("aria-label", `${text(experiment.label, experiment.id)}：${rateDescription(evaluation)}${interval(evaluation) ? `，95%区间${interval(evaluation)}` : ""}`);
      if (rate === null) track.textContent = "评测待完成";
      else {
        const bar = element("span", `chart-bar${index % 2 ? " is-secondary" : ""}`);
        bar.style.width = `${clamp(rate, 0, 100)}%`; track.append(bar);
      }
      chartRow.append(element("span", "chart-label", text(experiment.label, experiment.id)), track, element("span", "chart-value", rate === null ? "—" : `${number(rate)}%`));
      $("success-chart").append(chartRow);
      list(experiment.notes).forEach((note) => { if (typeof note === "string" && note.trim()) $("experiment-notes").append(element("li", "", `${text(experiment.label, experiment.id)}：${note}`)); });
    });
    renderLatency(experiments);
    renderTaskResults(experiments);
  }

  function renderLatency(experiments) {
    const hasSamplingComparison = experiments.some((experiment) => experiment.id === "sampling5_baseline");
    const rows = experiments.flatMap((experiment) => list(experiment.latency)
      .filter((measurement) => isNumber(measurement?.median_ms) || isNumber(measurement?.p95_ms))
      .filter((measurement) => !(hasSamplingComparison && experiment.id === "phase1_baseline" && measurement.num_steps === 5))
      .map((measurement) => ({ ...measurement, experiment })));
    $("latency-panel").hidden = rows.length === 0;
    $("latency-chart").replaceChildren();
    const maximum = Math.max(1, ...rows.map((row) => isNumber(row.median_ms) ? row.median_ms : 0));
    rows.forEach((measurement) => {
      const row = element("div", "latency-row");
      const family = text(measurement.experiment.label, measurement.experiment.id).replace(/\s*·\s*\d+步采样$/, "");
      const label = `${family} · ${isNumber(measurement.num_steps) ? `${measurement.num_steps}步采样` : "步数未记录"}${measurement.experiment.id === "phase1_lora" && measurement.num_steps === 5 ? "（仅测速）" : ""}`;
      const track = element("div", "chart-track"); track.setAttribute("role", "img");
      track.setAttribute("aria-label", `${label}，推理中位数${number(measurement.median_ms)}毫秒，P95 ${number(measurement.p95_ms)}毫秒`);
      if (isNumber(measurement.median_ms)) { const bar = element("span", "chart-bar"); bar.style.width = `${clamp(measurement.median_ms / maximum * 100, 0, 100)}%`; track.append(bar); }
      else { track.classList.add("pending"); track.textContent = "中位数待记录"; }
      const values = element("span", "latency-values", `${number(measurement.median_ms)} / ${number(measurement.p95_ms)} ms`);
      row.append(element("span", "latency-label", label), track, values); $("latency-chart").append(row);
    });
  }

  function renderTaskResults(experiments) {
    const target = $("task-results"); target.replaceChildren();
    const taskMap = new Map();
    experiments.forEach((experiment) => list(experiment.evaluation?.per_task).forEach((task) => {
      const key = String(task.task_id);
      if (!taskMap.has(key)) taskMap.set(key, { id: task.task_id, instruction: task.instruction, values: new Map() });
      taskMap.get(key).values.set(experiment.id, task);
      if (task.instruction) taskMap.get(key).instruction = task.instruction;
    }));
    if (!taskMap.size) { target.append(element("p", "table-muted", "逐任务评测结果待完成。")); return; }
    const wrapper = element("div", "table-scroll"); wrapper.tabIndex = 0; wrapper.setAttribute("role", "region"); wrapper.setAttribute("aria-label", "逐任务成功率对照，可横向滚动");
    const table = element("table", "task-table"); const head = element("thead"); const header = element("tr");
    ["任务", ...experiments.map((experiment) => text(experiment.label, experiment.id))].forEach((label) => { const th = element("th", "", label); th.scope = "col"; header.append(th); }); head.append(header);
    const body = element("tbody");
    Array.from(taskMap.values()).sort((a, b) => String(a.id).localeCompare(String(b.id), "en", { numeric: true })).forEach((task) => {
      const row = element("tr"); row.append(cell(`任务 ${task.id}`, text(task.instruction, "任务指令待补充"), "method-name"));
      row.firstChild.lastChild.classList.add("instruction");
      experiments.forEach((experiment) => { const result = task.values.get(experiment.id); row.append(cell(rateDescription(result), interval(result) ? `95% CI ${interval(result)}` : null)); }); body.append(row);
    });
    table.append(head, body); wrapper.append(table); target.append(wrapper);
  }

  function fillSelect(id, values, defaultLabel) {
    const select = $(id); const previous = select.value;
    select.replaceChildren(new Option(defaultLabel, "all"));
    values.forEach(([value, label]) => select.add(new Option(label, String(value))));
    select.value = Array.from(select.options).some((option) => option.value === previous) ? previous : "all";
  }

  function prepareVideos() {
    state.videoLimit = 12;
    fillSelect("video-method", state.experiments.filter((e) => state.videos.some((v) => v.experiment.id === e.id)).map((e) => [e.id, text(e.label, e.id)]), "全部方案");
    const tasks = Array.from(new Set(state.videos.filter((v) => v.task_id !== undefined && v.task_id !== null).map((v) => String(v.task_id)))).sort((a, b) => a.localeCompare(b, "en", { numeric: true }));
    fillSelect("video-task", tasks.map((id) => [id, `任务 ${id}`]), "全部任务");
    $("show-failures").disabled = !state.videos.some((v) => v.success === false);
    renderVideos();
  }

  function renderVideos({ append = false } = {}) {
    const method = $("video-method").value; const task = $("video-task").value; const outcome = $("video-outcome").value;
    const videos = state.videos.filter((v) => (method === "all" || String(v.experiment.id) === method)
      && (task === "all" || String(v.task_id) === task)
      && (outcome === "all" || (outcome === "success" && v.success === true) || (outcome === "failed" && v.success === false) || (outcome === "unknown" && v.success !== true && v.success !== false)));
    if (!append) {
      if (posterObserver) $("video-grid").querySelectorAll("video").forEach((video) => posterObserver.unobserve(video));
      $("video-grid").replaceChildren();
      state.renderedVideos = 0;
    }
    const displayCount = Math.min(state.videoLimit, videos.length);
    $("video-count").textContent = state.videos.length ? `当前筛选 ${videos.length} / 共 ${state.videos.length} 条真实回合` : "视频记录待生成";
    $("video-pagination").hidden = videos.length === 0;
    $("video-pagination-note").textContent = `已展示 ${displayCount} / ${videos.length} 条回合`;
    $("load-more-videos").hidden = displayCount >= videos.length;
    $("video-empty").hidden = videos.length > 0;
    if (!videos.length) {
      $("video-empty").firstElementChild.textContent = state.videos.length ? "没有匹配的回合" : "闭环视频待生成";
      $("video-empty").lastElementChild.textContent = state.videos.length ? "调整筛选条件，或重置为全部记录。" : "有可用视频后，会与实验方案、任务和结果一起出现在这里。";
    }
    videos.slice(state.renderedVideos, displayCount).forEach((record) => {
      const card = element("article", "video-card"); const player = element("div", "video-player");
      const video = makeVideo(record); video.addEventListener("error", () => { const note = element("p", "media-error", "此视频暂不可用，请查看原始实验记录。"); player.replaceChildren(note); }, { once: true }); player.append(video);
      const info = element("div", "video-info"); const meta = element("div", "video-meta");
      const label = text(record.experiment.label, record.experiment.id);
      meta.append(element("span", "experiment-label", `${label} · 任务 ${record.task_id ?? "未标注"}${record.episode_index !== null && record.episode_index !== undefined ? ` / 回合 ${record.episode_index}` : ""}`));
      const tags = element("span", "video-tags"); if (isCheck(record.experiment)) tags.append(kindTag(record.experiment));
      const outcome = record.success === true ? ["success", "成功"] : record.success === false ? ["failure", "失败"] : ["unknown", "未标注"];
      tags.append(element("span", `outcome-tag ${outcome[0]}`, outcome[1])); meta.append(tags);
      info.append(meta, element("h3", "", text(record.caption, "仿真闭环中的真实操作回合"))); card.append(player, info); $("video-grid").append(card);
    });
    state.renderedVideos = displayCount;
  }

  function renderProtocol(data) {
    const protocol = data.protocol || {}; const tasks = Array.isArray(protocol.tasks) ? protocol.tasks.length : protocol.tasks;
    const values = [["任务子集", isNumber(tasks) ? `${count(tasks)} 个任务` : text(protocol.tasks)], ["训练轨迹", isNumber(protocol.train_episodes) ? `${count(protocol.train_episodes)} 条` : "待记录"], ["留出轨迹", isNumber(protocol.heldout_episodes) ? `${count(protocol.heldout_episodes)} 条` : "待记录"], ["训练随机种子", count(protocol.seed)], ["训练 / 留出帧", `${count(protocol.train_frames)} / ${count(protocol.heldout_frames)}`]];
    $("protocol-details").replaceChildren(); values.forEach(([label, value]) => { const row = element("div", "protocol-row"); row.append(element("dt", "", label), element("dd", "", value)); $("protocol-details").append(row); });
    $("protocol-notes").replaceChildren(); list(protocol.notes).forEach((note) => $("protocol-notes").append(element("li", "", typeof note === "string" ? note : text(note?.text))));
    $("ablation-list").replaceChildren(); const ablations = list(data.ablations); $("ablation-empty").hidden = ablations.length > 0;
    ablations.forEach((ablation, index) => {
      const card = element("article", "ablation-card"); card.append(element("span", "ablation-index", String(index + 1).padStart(2, "0")));
      const content = element("div"); const header = element("div", "ablation-card-header"); header.append(element("h4", "", text(ablation.label, "对照实验")));
      const statusMap = { completed: "已完成", complete: "已完成", running: "进行中", planned: "待开展", pending: "待完成", failed: "需检查" };
      const status = text(ablation.status, "待完成"); header.append(element("span", `kind-tag${/planned|pending|待/.test(status) ? " neutral" : ""}`, statusMap[status] || status));
      const hypothesis = element("p"); hypothesis.append(element("strong", "", "问题"), document.createTextNode(text(ablation.hypothesis, "研究问题待补充。")));
      const conclusion = element("p", ablation.conclusion ? "" : "pending-conclusion"); conclusion.append(element("strong", "", "观察"), document.createTextNode(text(ablation.conclusion, "实验结果待完成。")));
      content.append(header, hypothesis, conclusion); card.append(content); $("ablation-list").append(card);
    });
  }

  function renderTrainingCurve(data) {
    const figure = $("training-curve");
    const image = $("training-curve-image");
    const src = safeURL(data.project?.training_curve_src);
    figure.hidden = !src;
    if (!src) { image.removeAttribute("src"); return; }
    configureLink($("training-curve-original"), src);
    $("training-curve-original").target = "_blank";
    $("training-curve-original").rel = "noopener noreferrer";
    image.onerror = () => { figure.hidden = true; };
    image.src = src;
  }

  function renderFailureCases(data) {
    const cases = list(data.failure_cases).filter((record) => record && typeof record === "object" && !Array.isArray(record));
    $("failure-cases").hidden = cases.length === 0;
    const container = $("failure-case-list");
    container.replaceChildren();
    cases.forEach((record) => {
      const card = element("article", "failure-case");
      const title = text(record.title, "失败回合记录");
      const imageSrc = safeURL(record.image);
      if (imageSrc) {
        const imageLink = element("a", "failure-case-image");
        configureLink(imageLink, imageSrc);
        imageLink.target = "_blank"; imageLink.rel = "noopener noreferrer";
        imageLink.setAttribute("aria-label", `${title}：在新窗口查看完整原图`);
        const image = element("img");
        image.src = imageSrc; image.alt = `${title}的回合画面`; image.loading = "lazy"; image.decoding = "async"; image.width = 1280; image.height = 640;
        image.addEventListener("error", () => { imageLink.hidden = true; }, { once: true });
        imageLink.append(image, element("span", "failure-image-hint", "查看原图 ↗"));
        card.append(imageLink);
      }
      const content = element("div", "failure-case-content");
      const metadata = element("p", "failure-case-meta");
      const experiment = state.experiments.find((item) => item.id === record.experiment);
      const labels = [text(experiment?.label, text(record.experiment, "实验方案待补充"))];
      if (isNumber(record.task_id) || (typeof record.task_id === "string" && record.task_id.trim())) labels.push(`任务 ${record.task_id}`);
      if (isNumber(record.episode_index) || (typeof record.episode_index === "string" && record.episode_index.trim())) labels.push(`回合 ${record.episode_index}`);
      metadata.textContent = labels.join(" / ");
      content.append(metadata, element("h4", "", title));
      const notes = element("dl", "failure-case-notes");
      [["observation", "观察"], ["hypothesis", "待验证假设"], ["next_test", "下一步"]].forEach(([key, label]) => {
        const row = element("div", `case-note case-note-${key}`);
        row.append(element("dt", "", label), element("dd", "", text(record[key], "待补充。")));
        notes.append(row);
      });
      content.append(notes);
      if (safeURL(record.video)) {
        const videoLink = element("a", "text-link failure-case-video", "打开回合片段 ↗");
        configureLink(videoLink, record.video); videoLink.target = "_blank"; videoLink.rel = "noopener noreferrer";
        content.append(videoLink);
      }
      card.append(content); container.append(card);
    });
  }

  function renderFindings(data) {
    [["conclusions", "conclusions-list", "conclusions-empty"], ["limitations", "limitations-list", "limitations-empty"]].forEach(([key, targetId, emptyId]) => {
      const entries = list(data[key]).map((item) => typeof item === "string" ? item : item?.text || item?.conclusion || item?.label).filter((item) => typeof item === "string" && item.trim());
      $(targetId).replaceChildren(); entries.forEach((entry) => $(targetId).append(element("li", "", entry))); $(emptyId).hidden = entries.length > 0;
    });
    $("evidence-list").replaceChildren();
    const evidence = list(data.evidence).filter((item) => safeURL(item?.href));
    $("evidence-empty").hidden = evidence.length > 0;
    evidence.forEach((record) => { const link = element("a", "evidence-link"); configureLink(link, record.href); link.append(element("span", "", text(record.label, "实验原始记录")), element("span", "", "↗")); $("evidence-list").append(link); });
    renderFailureCases(data);
  }

  async function loadResults() {
    if (state.loading) return;
    state.loading = true; $("main").setAttribute("aria-busy", "true"); $("refresh-results").setAttribute("aria-busy", "true"); $("refresh-results").disabled = true;
    try {
      const response = await fetch("data/results.json", { cache: "no-store", headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error("Invalid result document");
      state.data = data; state.experiments = list(data.experiments).filter((item) => item && typeof item === "object");
      state.videos = state.experiments.flatMap((experiment) => list(experiment.videos).filter((video) => safeURL(video?.src)).map((video) => ({ ...video, experiment })));
      $("load-error").hidden = true; $("raw-results").hidden = false;
      renderHero(data); renderComparison(); prepareVideos(); renderProtocol(data); renderFindings(data); renderTrainingCurve(data);
    } catch (_) {
      $("load-error").textContent = state.data ? "暂时无法刷新记录，当前保留上次成功读取的结果。" : "暂时无法加载实验记录。请稍后刷新，或查看项目仓库。";
      $("load-error").hidden = false;
      if (!state.data) { $("status-label").textContent = "实验记录待加载"; $("project-status").classList.add("is-error"); $("comparison-empty").hidden = false; $("comparison-content").hidden = true; }
    } finally { state.loading = false; $("main").setAttribute("aria-busy", "false"); $("refresh-results").setAttribute("aria-busy", "false"); $("refresh-results").disabled = false; }
  }

  const initialSample = $("initial-camera-sample");
  initialSample.addEventListener("error", () => { if (initialSample.isConnected) showCameraSample($("hero-visual")); }, { once: true });
  if (initialSample.complete && initialSample.naturalWidth === 0) showCameraSample($("hero-visual"));
  $("experiment-scope").addEventListener("change", renderComparison);
  ["video-method", "video-task", "video-outcome"].forEach((id) => $(id).addEventListener("change", () => { state.videoLimit = 12; renderVideos(); }));
  $("reset-filters").addEventListener("click", () => { ["video-method", "video-task", "video-outcome"].forEach((id) => { $(id).value = "all"; }); state.videoLimit = 12; renderVideos(); });
  $("load-more-videos").addEventListener("click", () => { state.videoLimit += 12; renderVideos({ append: true }); });
  $("show-failures").addEventListener("click", () => { $("video-method").value = "all"; $("video-task").value = "all"; $("video-outcome").value = "failed"; state.videoLimit = 12; renderVideos(); $("demos").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); $("video-outcome").focus({ preventScroll: true }); });
  $("refresh-results").addEventListener("click", loadResults);
  loadResults();
})();
