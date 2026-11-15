const $ = (selector, scope = document) => scope.querySelector(selector);
const $$ = (selector, scope = document) => Array.from(scope.querySelectorAll(selector));
const root = document.documentElement;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const lakeSelect = $("#lake-select");
const imageSelect = $("#image-select");
const errorBanner = $("#error-banner");
const mapImage = $("#quality-map");
const mapEmpty = $("#map-empty");
const mapFrame = $("#map-frame");
const mapExpand = $("#map-expand");
const chart = $("#trend-chart");
const analysisForm = $("#analysis-form");
const analysisLakeInput = $("#analysis-lake");
const analysisStartDate = $("#analysis-start-date");
const analysisEndDate = $("#analysis-end-date");
const analysisCadence = $("#analysis-cadence");
const analysisBoundary = $("#analysis-boundary");
const cloudCoverSlider = $("#cloud-cover");
const cloudCoverValue = $("#cloud-cover-value");
const analysisFormMessage = $("#analysis-form-message");
const analysisSubmit = analysisForm.querySelector("button[type='submit']");
const jobStatus = $("#job-status");
const jobState = $("#job-state");
const jobMessage = $("#job-message");
const jobProgressBar = $("#job-progress-bar");
const jobProgressLabel = $("#job-progress-label");
const jobLog = $("#job-log");
const lakeDatalist = $("#lake-options");
const geeConfigHint = $("#gee-config-hint");
const chartTooltip = $("#chart-tooltip");
const tooltipTitle = $("#tooltip-title");
const tooltipGood = $("#tooltip-good");
const tooltipModerate = $("#tooltip-moderate");
const tooltipPoor = $("#tooltip-poor");
const lightbox = $("#lightbox");
const lightboxImage = $("#lightbox-image");
const lightboxCaption = $("#lightbox-caption");

const STAGE_ORDER = ["download", "water_masks", "classification", "severity", "gradcam", "temporal", "validation"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

let imageRows = [];
let temporalRows = [];
let activeJobId = null;
let lastJobState = null;
let trendRange = "all";
let tableQuery = "";
let sort = { key: "image", dir: 1 };
let chartLayout = null;
let chartHover = null;
let chartFrame = null;

async function getJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

function pct(value) {
  const number = Number(value);
  return `${Number.isFinite(number) ? number.toFixed(1) : "0.0"}%`;
}

function pretty(name) {
  return String(name || "").replaceAll("_", " ");
}

function formatAcquisition(name) {
  const match = /(\d{4})(\d{2})$/.exec(String(name || ""));
  if (!match) return String(name || "");
  return `${MONTHS[Number(match[2]) - 1] || match[2]} ${match[1]}`;
}

function capitalize(value) {
  const text = String(value || "");
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}

function cssVar(name) {
  return getComputedStyle(root).getPropertyValue(name).trim();
}

function showError(message) {
  const text = String(message || "");
  errorBanner.textContent = text;
  errorBanner.hidden = !text;
}

function showToast(message, tone = "info") {
  const stack = $("#toast-stack");
  if (!stack) return;
  const toast = document.createElement("div");
  toast.className = `toast toast-${tone}`;
  const dot = document.createElement("span");
  dot.className = "toast-dot";
  const text = document.createElement("span");
  text.textContent = message;
  toast.append(dot, text);
  stack.append(toast);
  requestAnimationFrame(() => toast.classList.add("in"));
  window.setTimeout(() => {
    toast.classList.remove("in");
    window.setTimeout(() => toast.remove(), 320);
  }, 4200);
}

function animateNumber(element, target, options = {}) {
  if (!element) return;
  const { suffix = "", decimals = 0, duration = 800 } = options;
  const value = Number(target);
  const format = (number) => `${number.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}${suffix}`;
  if (!Number.isFinite(value)) {
    element.textContent = target == null ? "--" : String(target);
    return;
  }
  if (reduceMotion || document.hidden) {
    element.textContent = format(value);
    element.dataset.value = String(value);
    return;
  }
  const from = Number(element.dataset.value || 0);
  element.dataset.value = String(value);
  cancelAnimationFrame(Number(element.dataset.raf || 0));
  const start = performance.now();
  const tick = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    element.textContent = format(from + (value - from) * eased);
    if (progress < 1) element.dataset.raf = String(requestAnimationFrame(tick));
  };
  element.dataset.raf = String(requestAnimationFrame(tick));
}

/* ---------- shell interactions ---------- */
function initTheme() {
  $("#theme-toggle")?.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("aquascope-theme", next); } catch (_) {}
    requestAnimationFrame(() => drawTrend(false));
  });
}

function initReveal() {
  const items = $$(".reveal");
  if (reduceMotion || !("IntersectionObserver" in window)) {
    items.forEach((item) => item.classList.add("in"));
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add("in");
      observer.unobserve(entry.target);
    }
  }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
  items.forEach((item) => observer.observe(item));
}

function initNavSpy() {
  const links = $$(".topnav a");
  if (!("IntersectionObserver" in window)) return;
  const sections = links.map((link) => $(link.getAttribute("href"))).filter(Boolean);
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      links.forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`);
      });
    }
  }, { rootMargin: "-42% 0px -52% 0px" });
  sections.forEach((section) => observer.observe(section));
}

function initScroll() {
  let ticking = false;
  const update = () => {
    ticking = false;
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = scrollable > 0 ? Math.min(window.scrollY / scrollable, 1) : 0;
    const progress = $("#scroll-progress");
    if (progress) progress.style.transform = `scaleX(${ratio})`;
    document.body.classList.toggle("scrolled", window.scrollY > 8);
    $("#to-top").classList.toggle("visible", window.scrollY > 640);
  };
  window.addEventListener("scroll", () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(update);
  }, { passive: true });
  $("#to-top").addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
  });
  update();
}

/* ---------- lightbox ---------- */
function openLightbox(src, caption) {
  if (!src) return;
  lightboxImage.src = src;
  lightboxImage.alt = caption || "Enlarged analysis image";
  lightboxCaption.textContent = caption || "";
  lightbox.hidden = false;
  document.body.classList.add("no-scroll");
  void lightbox.offsetWidth;
  lightbox.classList.add("open");
  $(".lightbox-close").focus();
}

function closeLightbox() {
  lightbox.classList.remove("open");
  document.body.classList.remove("no-scroll");
  window.setTimeout(() => {
    lightbox.hidden = true;
    lightboxImage.removeAttribute("src");
  }, 240);
}

function initLightbox() {
  lightbox.addEventListener("click", (event) => {
    if (event.target === lightbox || event.target.closest(".lightbox-close")) closeLightbox();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !lightbox.hidden) closeLightbox();
  });
  mapExpand.addEventListener("click", () => {
    if (mapImage.hidden || !mapImage.src) return;
    openLightbox(mapImage.src, `Water quality map · ${imageSelect.value}`);
  });
}

/* ---------- analysis form ---------- */
function updateCloudCoverValue() {
  const value = `${cloudCoverSlider.value}%`;
  cloudCoverValue.value = value;
  cloudCoverValue.textContent = value;
  cloudCoverSlider.style.setProperty("--fill", value);
}

function updateWorkflow(stage, allDone = false) {
  const index = STAGE_ORDER.indexOf(stage);
  for (const item of $$(".workflow-list li[data-stage]")) {
    const itemIndex = STAGE_ORDER.indexOf(item.dataset.stage);
    item.classList.toggle("active", !allDone && stage !== null && item.dataset.stage === stage);
    item.classList.toggle("done", allDone || (index >= 0 && itemIndex >= 0 && itemIndex < index));
  }
}

function setAnalysisRunning(running) {
  analysisSubmit.disabled = running;
  analysisSubmit.classList.toggle("is-busy", running);
  const label = $(".button-label", analysisSubmit);
  if (label) label.textContent = running ? "Analysis running…" : "Run analysis";
}

function renderJob(job) {
  jobStatus.hidden = false;
  jobStatus.classList.add("is-running");
  jobState.textContent = { queued: "Queued", running: "Running", done: "Complete", failed: "Failed" }[job.state] || job.state;
  jobState.className = `job-state ${job.state}`;
  jobMessage.textContent = job.message || "";
  const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
  jobProgressBar.style.width = `${progress}%`;
  jobProgressLabel.textContent = `${Math.round(progress)}%`;
  if (Array.isArray(job.log) && job.log.length) {
    jobLog.textContent = job.log.slice(-40).join("\n");
    jobLog.scrollTop = jobLog.scrollHeight;
  }
  const terminal = job.state === "done" || job.state === "failed";
  updateWorkflow(job.stage || null, job.state === "done");
  if (terminal) {
    jobStatus.classList.remove("is-running");
    setAnalysisRunning(false);
    showError(job.state === "failed" ? (job.error || job.message || "Analysis failed") : "");
    if (lastJobState !== job.state) {
      showToast(
        job.state === "done" ? "Analysis complete — dashboard refreshed." : "Analysis failed. See the job log.",
        job.state === "done" ? "success" : "error",
      );
    }
  }
  lastJobState = job.state;
}

async function pollJob(id) {
  try {
    const payload = await getJson(`/api/jobs/${encodeURIComponent(id)}`);
    renderJob(payload.job);
    if (payload.job.state === "done" || payload.job.state === "failed") {
      activeJobId = null;
      if (payload.job.state === "done") await refreshDashboard(payload.job.slug);
      return;
    }
  } catch (error) {
    activeJobId = null;
    setAnalysisRunning(false);
    showError(error.message);
    return;
  }
  window.setTimeout(() => pollJob(id), 2000);
}

async function refreshDashboard(slug) {
  try {
    const summary = await getJson("/api/summary");
    renderLakeOptions(summary.lakes);
    const target = summary.lakes.find((lake) => lake.lake === slug);
    if (target) {
      lakeSelect.value = target.lake;
      analysisLakeInput.value = pretty(target.lake);
      await loadLake();
    }
  } catch (error) {
    showError(error.message);
  }
}

async function submitAnalysis(event) {
  event.preventDefault();
  showError("");
  analysisFormMessage.textContent = "";
  analysisEndDate.setCustomValidity("");

  if (analysisStartDate.value && analysisEndDate.value && analysisEndDate.value < analysisStartDate.value) {
    analysisEndDate.setCustomValidity("End date must be on or after the start date.");
  }
  if (!analysisForm.reportValidity()) return;
  if (activeJobId) {
    analysisFormMessage.textContent = "An analysis is already running.";
    return;
  }

  const payload = {
    lake: analysisLakeInput.value.trim(),
    start: analysisStartDate.value,
    end: analysisEndDate.value,
    cadence: analysisCadence.value,
    max_cloud: Number(cloudCoverSlider.value),
  };
  if (analysisBoundary.value.trim()) payload.boundary = analysisBoundary.value.trim();

  setAnalysisRunning(true);
  updateWorkflow("download");
  analysisFormMessage.textContent = "Requesting imagery from Earth Engine…";
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not start the analysis");
    activeJobId = result.job.id;
    lastJobState = null;
    analysisFormMessage.textContent = "Analysis running — results appear when it finishes.";
    renderJob(result.job);
    showToast("Analysis queued. Progress streams below.", "info");
    window.setTimeout(() => pollJob(result.job.id), 1200);
  } catch (error) {
    activeJobId = null;
    setAnalysisRunning(false);
    updateWorkflow(null);
    analysisFormMessage.textContent = error.message;
    showError(error.message);
  }
}

async function loadConfig() {
  if (!geeConfigHint) return;
  try {
    const config = await getJson("/api/config");
    if (!config.gee_installed) {
      geeConfigHint.textContent = "Earth Engine client is not installed. Run: pip install -r requirements.txt";
      geeConfigHint.hidden = false;
    } else if (!config.project_configured) {
      geeConfigHint.textContent = "Earth Engine project is not configured. Authenticate and set EE_PROJECT before running a new lake.";
      geeConfigHint.hidden = false;
    } else {
      geeConfigHint.hidden = true;
    }
  } catch (error) {
    geeConfigHint.hidden = true;
  }
}

/* ---------- options + hero ---------- */
function makeOption(value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  return option;
}

function renderLakeOptions(lakes) {
  lakeSelect.replaceChildren(...lakes.map((lake) => makeOption(lake.lake, pretty(lake.lake))));
  if (lakeDatalist) {
    lakeDatalist.replaceChildren(...lakes.map((lake) => makeOption(pretty(lake.lake), pretty(lake.lake))));
  }
  const total = lakes.reduce((sum, lake) => sum + lake.image_count, 0);
  $("#batch-count").textContent = `${total} acquisitions`;
  animateNumber($("#hero-lakes"), lakes.length);
  animateNumber($("#hero-acquisitions"), total);
}

/* ---------- metrics + donut ---------- */
function setBar(selector, value) {
  const bar = $(selector);
  if (bar) bar.style.width = `${Math.max(0, Math.min(100, Number(value) || 0))}%`;
}

function updateDonut(row) {
  const segments = [
    ["#donut-good", Math.max(0, Math.min(100, Number(row.good_percentage) || 0))],
    ["#donut-moderate", Math.max(0, Math.min(100, Number(row.moderate_percentage) || 0))],
    ["#donut-poor", Math.max(0, Math.min(100, Number(row.poor_percentage) || 0))],
  ];
  let offset = 0;
  for (const [selector, value] of segments) {
    const element = $(selector);
    if (!element) continue;
    const visible = Math.max(value - 0.8, 0);
    element.style.strokeDasharray = `${visible} ${100 - visible}`;
    element.style.strokeDashoffset = `${-offset}`;
    offset += value;
  }
  animateNumber($("#donut-total"), row.classified_patches, { duration: 700 });
  $("#legend-good").textContent = pct(row.good_percentage);
  $("#legend-moderate").textContent = pct(row.moderate_percentage);
  $("#legend-poor").textContent = pct(row.poor_percentage);
}

/* ---------- table ---------- */
function sortValue(row, key) {
  if (key === "dominant") return String(row.dominant_quality || "");
  if (key === "image") return String(row.image || "");
  return Number(row[key]) || 0;
}

function updateSortIndicators() {
  for (const button of $$("th button[data-sort]")) {
    const active = button.dataset.sort === sort.key;
    button.classList.toggle("active", active);
    const indicator = $(".sort-ind", button);
    if (indicator) indicator.textContent = active ? (sort.dir > 0 ? "↑" : "↓") : "↕";
    button.closest("th").setAttribute("aria-sort", active ? (sort.dir > 0 ? "ascending" : "descending") : "none");
  }
}

function visibleRows() {
  const query = tableQuery.trim().toLowerCase();
  const rows = imageRows.filter((row) => !query || String(row.image).toLowerCase().includes(query));
  return rows.sort((a, b) => {
    const left = sortValue(a, sort.key);
    const right = sortValue(b, sort.key);
    if (typeof left === "string") return left.localeCompare(right, undefined, { numeric: true }) * sort.dir;
    return (left - right) * sort.dir;
  });
}

function buildRow(row) {
  const tr = document.createElement("tr");
  tr.dataset.image = row.image;
  tr.classList.toggle("selected", row.image === imageSelect.value);
  tr.tabIndex = 0;
  for (const value of [row.image, pct(row.good_percentage), pct(row.moderate_percentage), pct(row.poor_percentage)]) {
    const td = document.createElement("td");
    td.textContent = value;
    tr.append(td);
  }
  const dominantCell = document.createElement("td");
  const badge = document.createElement("span");
  badge.className = `quality-badge ${String(row.dominant_quality || "").toLowerCase()}`;
  badge.textContent = row.dominant_quality || "--";
  dominantCell.append(badge);
  tr.append(dominantCell);

  const select = () => {
    imageSelect.value = row.image;
    loadImage().catch((error) => showError(error.message));
  };
  tr.addEventListener("click", select);
  tr.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      select();
    }
  });
  return tr;
}

function renderTable() {
  const body = $("#acquisition-rows");
  const rows = visibleRows();
  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.className = "table-empty";
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = imageRows.length ? "No acquisitions match this filter." : "No acquisitions available.";
    tr.append(td);
    body.replaceChildren(tr);
  } else {
    body.replaceChildren(...rows.map(buildRow));
  }
  $("#table-count").textContent = `${rows.length} of ${imageRows.length} images`;
  updateSortIndicators();
}

function initTableControls() {
  $("#table-search").addEventListener("input", (event) => {
    tableQuery = event.target.value;
    renderTable();
  });
  for (const button of $$("th button[data-sort]")) {
    button.addEventListener("click", () => {
      const key = button.dataset.sort;
      if (sort.key === key) {
        sort.dir *= -1;
      } else {
        sort = { key, dir: key === "image" || key === "dominant" ? 1 : -1 };
      }
      renderTable();
    });
  }
}

/* ---------- trend chart ---------- */
function trendRows() {
  if (!temporalRows.length) return [];
  const limit = trendRange === "all" ? temporalRows.length : Math.min(Number(trendRange) || temporalRows.length, temporalRows.length);
  return temporalRows.slice(-limit);
}

function paintChart(progress) {
  if (!chart) return;
  const bounds = chart.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const scale = window.devicePixelRatio || 1;
  chart.width = Math.round(bounds.width * scale);
  chart.height = Math.round(bounds.height * scale);
  const context = chart.getContext("2d");
  context.setTransform(scale, 0, 0, scale, 0, 0);
  const width = bounds.width;
  const height = bounds.height;
  const rows = trendRows();
  const colors = { good_pct: cssVar("--good"), moderate_pct: cssVar("--moderate"), poor_pct: cssVar("--poor") };
  const keys = Object.keys(colors);
  const margin = { top: 18, right: 18, bottom: 40, left: 44 };
  const plotWidth = Math.max(10, width - margin.left - margin.right);
  const plotHeight = Math.max(10, height - margin.top - margin.bottom);

  context.clearRect(0, 0, width, height);
  context.font = '11px "Segoe UI", system-ui, sans-serif';
  context.textBaseline = "middle";
  context.lineWidth = 1;

  for (let tick = 0; tick <= 100; tick += 25) {
    const y = margin.top + plotHeight * (1 - tick / 100);
    context.strokeStyle = cssVar("--chart-grid");
    context.beginPath();
    context.moveTo(margin.left, y);
    context.lineTo(width - margin.right, y);
    context.stroke();
    context.fillStyle = cssVar("--chart-label");
    context.textAlign = "right";
    context.fillText(`${tick}%`, margin.left - 9, y);
  }

  if (!rows.length) {
    context.fillStyle = cssVar("--chart-label");
    context.textAlign = "center";
    context.fillText("No temporal data available", width / 2, height / 2);
    chartLayout = null;
    return;
  }

  const step = rows.length === 1 ? 0 : plotWidth / (rows.length - 1);
  const xFor = (index) => margin.left + (rows.length === 1 ? plotWidth / 2 : index * step);
  chartLayout = { rows, margin, plotWidth, plotHeight, xFor };

  for (const key of keys) {
    context.beginPath();
    rows.forEach((row, index) => {
      const value = Math.max(0, Math.min(100, Number(row[key]) || 0)) * progress;
      const x = xFor(index);
      const y = margin.top + plotHeight * (1 - value / 100);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.strokeStyle = colors[key];
    context.lineWidth = 2.6;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.stroke();

    rows.forEach((row, index) => {
      const value = Math.max(0, Math.min(100, Number(row[key]) || 0)) * progress;
      const x = xFor(index);
      const y = margin.top + plotHeight * (1 - value / 100);
      context.beginPath();
      context.arc(x, y, 3.3, 0, Math.PI * 2);
      context.fillStyle = cssVar("--surface");
      context.fill();
      context.beginPath();
      context.arc(x, y, 3.3, 0, Math.PI * 2);
      context.strokeStyle = colors[key];
      context.lineWidth = 2;
      context.stroke();
    });
  }

  if (chartHover !== null && rows[chartHover]) {
    const x = xFor(chartHover);
    context.save();
    context.strokeStyle = cssVar("--chart-crosshair");
    context.lineWidth = 1;
    context.setLineDash([4, 4]);
    context.beginPath();
    context.moveTo(x, margin.top);
    context.lineTo(x, height - margin.bottom);
    context.stroke();
    context.restore();
    for (const key of keys) {
      const value = Math.max(0, Math.min(100, Number(rows[chartHover][key]) || 0));
      const y = margin.top + plotHeight * (1 - value / 100);
      context.beginPath();
      context.arc(x, y, 5, 0, Math.PI * 2);
      context.fillStyle = colors[key];
      context.fill();
      context.strokeStyle = cssVar("--surface");
      context.lineWidth = 2;
      context.stroke();
    }
  }

  context.fillStyle = cssVar("--chart-label");
  context.textAlign = "center";
  context.textBaseline = "top";
  const labelEvery = Math.max(1, Math.ceil(rows.length / 6));
  rows.forEach((row, index) => {
    if (index % labelEvery && index !== rows.length - 1) return;
    context.fillText(formatAcquisition(row.image), xFor(index), height - margin.bottom + 12, Math.max(54, plotWidth / 4));
  });
}

function drawTrend(animate = false) {
  if (!chart) return;
  cancelAnimationFrame(chartFrame);
  if (!animate || reduceMotion || document.hidden) {
    paintChart(1);
    return;
  }
  const start = performance.now();
  const frame = (now) => {
    const progress = Math.min((now - start) / 750, 1);
    paintChart(1 - Math.pow(1 - progress, 3));
    if (progress < 1) chartFrame = requestAnimationFrame(frame);
  };
  chartFrame = requestAnimationFrame(frame);
}

function updateTooltip(index) {
  if (!chartLayout || !chartLayout.rows[index]) {
    chartTooltip.hidden = true;
    return;
  }
  const row = chartLayout.rows[index];
  tooltipTitle.textContent = formatAcquisition(row.image);
  tooltipGood.textContent = pct(row.good_pct);
  tooltipModerate.textContent = pct(row.moderate_pct);
  tooltipPoor.textContent = pct(row.poor_pct);
  chartTooltip.hidden = false;
  const shell = chart.parentElement.getBoundingClientRect();
  const x = chartLayout.xFor(index);
  chartTooltip.style.left = `${Math.min(Math.max(x, 84), shell.width - 84)}px`;
  chartTooltip.style.top = `${chartLayout.margin.top + 6}px`;
}

function initChartHover() {
  chart.addEventListener("mousemove", (event) => {
    if (!chartLayout) return;
    const bounds = chart.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    let nearest = 0;
    let best = Infinity;
    chartLayout.rows.forEach((row, index) => {
      const distance = Math.abs(chartLayout.xFor(index) - x);
      if (distance < best) {
        best = distance;
        nearest = index;
      }
    });
    chartHover = nearest;
    paintChart(1);
    updateTooltip(nearest);
  });
  chart.addEventListener("mouseleave", () => {
    chartHover = null;
    chartTooltip.hidden = true;
    paintChart(1);
  });

  if ("ResizeObserver" in window) {
    let resizeFrame = null;
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(() => drawTrend(false));
    });
    observer.observe(chart.parentElement);
  } else {
    window.addEventListener("resize", () => drawTrend(false));
  }
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) drawTrend(false);
  });
}

function initTrendControls() {
  for (const button of $$(".segmented [data-range]")) {
    button.addEventListener("click", () => {
      trendRange = button.dataset.range;
      for (const sibling of $$(".segmented [data-range]")) {
        sibling.classList.toggle("active", sibling === button);
      }
      chartHover = null;
      chartTooltip.hidden = true;
      drawTrend(true);
    });
  }
}

/* ---------- lake + image ---------- */
async function loadLake() {
  showError("");
  const lake = lakeSelect.value;
  imageRows = await getJson(`/api/lakes/${encodeURIComponent(lake)}/images`);
  temporalRows = await getJson(`/api/temporal?lake=${encodeURIComponent(lake)}`);
  imageSelect.replaceChildren(...imageRows.map((row) => makeOption(row.image, row.image)));
  $("#lake-title").textContent = pretty(lake);
  $("#hero-latest").textContent = temporalRows.length
    ? formatAcquisition(temporalRows[temporalRows.length - 1].image)
    : "--";
  tableQuery = "";
  const search = $("#table-search");
  if (search) search.value = "";
  renderTable();
  await loadImage();
}

async function loadImage() {
  showError("");
  const lake = lakeSelect.value;
  const image = imageSelect.value;
  const row = imageRows.find((item) => item.image === image);
  if (!row) return;

  const captionLabel = formatAcquisition(image);
  $("#image-caption").textContent = captionLabel && captionLabel !== image
    ? `Acquisition ${captionLabel} · ${image}`
    : `Acquisition ${image}`;
  $("#map-image-label").textContent = formatAcquisition(image);
  animateNumber($("#good-value"), row.good_percentage, { decimals: 1, suffix: "%" });
  animateNumber($("#moderate-value"), row.moderate_percentage, { decimals: 1, suffix: "%" });
  animateNumber($("#poor-value"), row.poor_percentage, { decimals: 1, suffix: "%" });
  setBar("#good-bar", row.good_percentage);
  setBar("#moderate-bar", row.moderate_percentage);
  setBar("#poor-bar", row.poor_percentage);
  $("#dominant-quality").textContent = row.dominant_quality;
  $("#dominant-severity").textContent = row.dominant_severity;
  const severity = String(row.dominant_severity || "").trim();
  const severityBadge = $("#dominant-severity-badge");
  severityBadge.textContent = severity ? `${severity} severity` : "--";
  severityBadge.className = `severity-badge ${severity.toLowerCase()}`;
  $("#readout-image").textContent = image;
  $("#patch-count").textContent = `${row.classified_patches} patches`;
  $("#image-count").textContent = `${imageRows.length} acquisitions`;

  updateDonut(row);
  renderTable();

  mapFrame.classList.add("is-loading");
  mapImage.hidden = false;
  mapEmpty.hidden = true;
  mapExpand.disabled = false;
  mapImage.onload = () => mapFrame.classList.remove("is-loading");
  mapImage.onerror = () => {
    mapFrame.classList.remove("is-loading");
    mapImage.hidden = true;
    mapEmpty.hidden = false;
    mapExpand.disabled = true;
  };
  mapImage.src = `/api/quality-map?lake=${encodeURIComponent(lake)}&image=${encodeURIComponent(image)}&v=${Date.now()}`;

  drawTrend(true);
  await loadGradCAM(lake, image);
}

async function loadGradCAM(lake, image) {
  const grid = $("#gradcam-grid");
  try {
    const classes = await getJson(`/api/gradcam-classes?lake=${encodeURIComponent(lake)}&image=${encodeURIComponent(image)}`);
    if (!classes.length) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "No explanations generated for this acquisition.";
      grid.replaceChildren(empty);
      return;
    }
    grid.replaceChildren(...classes.map((className) => {
      const figure = document.createElement("figure");
      figure.className = "gradcam-item";
      figure.tabIndex = 0;
      figure.setAttribute("role", "button");
      figure.setAttribute("aria-label", `Expand ${className} class Grad-CAM overlay`);
      const imageElement = document.createElement("img");
      imageElement.alt = `${capitalize(className)} class Grad-CAM overlay for ${image}`;
      imageElement.loading = "lazy";
      const src = `/api/gradcam?lake=${encodeURIComponent(lake)}&image=${encodeURIComponent(image)}&class=${encodeURIComponent(className)}`;
      imageElement.src = src;
      const caption = document.createElement("figcaption");
      caption.textContent = `${capitalize(className)} activation`;
      figure.append(imageElement, caption);
      const open = () => openLightbox(src, `${capitalize(className)} activation · ${image}`);
      figure.addEventListener("click", open);
      figure.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          open();
        }
      });
      return figure;
    }));
  } catch (error) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = error.message;
    grid.replaceChildren(empty);
  }
}

/* ---------- boot ---------- */
async function start() {
  initTheme();
  initReveal();
  initNavSpy();
  initScroll();
  initLightbox();
  initChartHover();
  initTableControls();
  initTrendControls();
  updateCloudCoverValue();
  await loadConfig();
  try {
    const summary = await getJson("/api/summary");
    if (!summary.lakes.length) throw new Error("No severity summary is available. Run the inference and severity stages first.");
    renderLakeOptions(summary.lakes);
    await loadLake();
  } catch (error) {
    showError(error.message);
    $("#batch-count").textContent = "Outputs unavailable";
  }
}

lakeSelect.addEventListener("change", () => {
  analysisLakeInput.value = pretty(lakeSelect.value);
  loadLake().catch((error) => showError(error.message));
});
imageSelect.addEventListener("change", () => loadImage().catch((error) => showError(error.message)));
cloudCoverSlider.addEventListener("input", updateCloudCoverValue);
analysisForm.addEventListener("submit", submitAnalysis);

start();
