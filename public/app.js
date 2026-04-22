const metadata = Object.freeze({
  repo: "exp-llm-worker-ux",
  category: "llm",
  purpose: "worker UX 실험",
  priority: "P1",
  trackLabel: "LLM",
  kindLabel: "experiment",
  trackSlug: "llm",
  workloadKind: "llm-chat",
  pagesUrl: "https://ai-webgpu-lab.github.io/exp-llm-worker-ux/",
  repoUrl: "https://github.com/ai-webgpu-lab/exp-llm-worker-ux",
  readmeUrl: "https://github.com/ai-webgpu-lab/exp-llm-worker-ux/blob/main/README.md",
  resultsUrl: "https://github.com/ai-webgpu-lab/exp-llm-worker-ux/blob/main/RESULTS.md"
});

const state = {
  startedAt: performance.now(),
  environment: null,
  probes: {
    webgpu: null,
    frame: null,
    worker: null
  },
  logs: []
};

const knownLimitKeys = [
  "maxTextureDimension1D",
  "maxTextureDimension2D",
  "maxTextureDimension3D",
  "maxBindGroups",
  "maxBindingsPerBindGroup",
  "maxUniformBufferBindingSize",
  "maxStorageBufferBindingSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupStorageSize",
  "maxBufferSize"
];

const elements = {
  metaGrid: document.getElementById("meta-grid"),
  statusRow: document.getElementById("status-row"),
  statusSummary: document.getElementById("status-summary"),
  focusList: document.getElementById("focus-list"),
  nextSteps: document.getElementById("next-steps"),
  metricsGrid: document.getElementById("metrics-grid"),
  environmentJson: document.getElementById("environment-json"),
  resultJson: document.getElementById("result-json"),
  activityLog: document.getElementById("activity-log"),
  detectEnvironment: document.getElementById("detect-environment"),
  runWebgpu: document.getElementById("run-webgpu"),
  runFrame: document.getElementById("run-frame"),
  runWorker: document.getElementById("run-worker"),
  downloadJson: document.getElementById("download-json")
};

function round(value, digits = 2) {
  if (!Number.isFinite(value)) {
    return null;
  }

  const factor = Math.pow(10, digits);
  return Math.round(value * factor) / factor;
}

function percentile(values, ratio) {
  if (!values.length) {
    return null;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * ratio) - 1));
  return sorted[index];
}

function nowIso() {
  return new Date().toISOString();
}

function parseBrowser() {
  const ua = navigator.userAgent;
  const candidates = [
    ["Edg/", "Edge"],
    ["Chrome/", "Chrome"],
    ["Firefox/", "Firefox"],
    ["Version/", "Safari"]
  ];

  for (const [needle, name] of candidates) {
    const marker = ua.indexOf(needle);
    if (marker >= 0) {
      const version = ua.slice(marker + needle.length).split(/[\s)/;]/)[0] || "unknown";
      return { name, version };
    }
  }

  return { name: "Unknown", version: "unknown" };
}

function parseOs() {
  const ua = navigator.userAgent;

  if (/Windows NT/i.test(ua)) {
    const match = ua.match(/Windows NT ([0-9.]+)/i);
    return { name: "Windows", version: match ? match[1] : "unknown" };
  }

  if (/Mac OS X/i.test(ua)) {
    const match = ua.match(/Mac OS X ([0-9_]+)/i);
    return { name: "macOS", version: match ? match[1].replace(/_/g, ".") : "unknown" };
  }

  if (/Android/i.test(ua)) {
    const match = ua.match(/Android ([0-9.]+)/i);
    return { name: "Android", version: match ? match[1] : "unknown" };
  }

  if (/(iPhone|iPad|CPU OS)/i.test(ua)) {
    const match = ua.match(/OS ([0-9_]+)/i);
    return { name: "iOS", version: match ? match[1].replace(/_/g, ".") : "unknown" };
  }

  if (/Linux/i.test(ua)) {
    return { name: "Linux", version: "unknown" };
  }

  return { name: "Unknown", version: "unknown" };
}

function inferDeviceClass() {
  const threads = navigator.hardwareConcurrency || 0;
  const memory = navigator.deviceMemory || 0;
  const mobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);

  if (mobile) {
    if (memory >= 6 && threads >= 8) {
      return "mobile-high";
    }

    return "mobile-mid";
  }

  if (memory >= 16 && threads >= 12) {
    return "desktop-high";
  }

  if (memory >= 8 && threads >= 8) {
    return "desktop-mid";
  }

  if (threads >= 4) {
    return "laptop";
  }

  return "unknown";
}

function baseEnvironment() {
  return {
    browser: parseBrowser(),
    os: parseOs(),
    device: {
      name: navigator.platform || "unknown",
      class: inferDeviceClass(),
      cpu: navigator.hardwareConcurrency ? String(navigator.hardwareConcurrency) + " threads" : "unknown",
      memory_gb: navigator.deviceMemory || undefined,
      power_mode: "unknown"
    },
    gpu: {
      adapter: "unknown",
      required_features: [],
      limits: {}
    },
    backend: "wasm",
    fallback_triggered: true,
    worker_mode: "unknown",
    cache_state: "unknown"
  };
}

function ensureEnvironment() {
  if (!state.environment) {
    state.environment = baseEnvironment();
  }

  return state.environment;
}

function log(message) {
  state.logs.unshift("[" + new Date().toLocaleTimeString() + "] " + message);
  state.logs = state.logs.slice(0, 14);
  renderLogs();
}

function metadataCards() {
  return [
    ["Track", metadata.trackLabel],
    ["Kind", metadata.kindLabel],
    ["Priority", metadata.priority],
    ["Workload", metadata.workloadKind],
    ["Pages URL", metadata.pagesUrl]
  ];
}

function focusItems() {
  const common = [
    "Collect a reproducible browser and device snapshot before adding workload-specific code.",
    "Use the exported JSON as the first draft for reports/raw once you validate it in the target browser."
  ];

  switch (metadata.category) {
    case "template":
      return common.concat([
        "Verify the smallest WebGPU success path and copy that shape into downstream repositories.",
        "Document capability and fallback behavior before adding framework-specific layers."
      ]);
    case "benchmark":
      return common.concat([
        "Replace lightweight frame and worker probes with workload-specific comparison harnesses.",
        "Keep input profiles and environment notes identical across runs."
      ]);
    case "app":
      return common.concat([
        "Check whether the integration surface can acquire GPU resources without blocking the UI.",
        "Turn this probe into the first user-facing end-to-end demo once the core flow exists."
      ]);
    case "graphics":
    case "blackhole":
      return common.concat([
        "Prioritize adapter/device acquisition, frame pacing, and scene-load instrumentation.",
        "Capture visual correctness notes together with frame timing."
      ]);
    default:
      return common.concat([
        "Prioritize adapter readiness, worker offload viability, and result export hygiene.",
        "Replace generic probes with model or runtime-specific metrics as soon as the first harness lands."
      ]);
  }
}

function nextSteps() {
  const steps = [
    "Save an exported JSON after validating it in the target browser and move it into reports/raw/.",
    "Replace generic probes in public/app.js with workload-specific setup and KPI collection.",
    "Update RESULTS.md with the first measured run and record fallback conditions explicitly."
  ];

  if (metadata.category === "template") {
    steps.unshift("Promote the minimal setup path into a copyable starter template for downstream repos.");
  }

  if (metadata.category === "benchmark") {
    steps.unshift("Define the comparison matrix and freeze one shared input profile before collecting numbers.");
  }

  if (metadata.category === "app") {
    steps.unshift("Connect one real user flow and treat this probe as the readiness gate before adding polish.");
  }

  return steps;
}

function renderList(element, items) {
  element.innerHTML = "";
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    element.appendChild(li);
  }
}

function renderMeta() {
  elements.metaGrid.innerHTML = "";

  for (const [label, value] of metadataCards()) {
    const card = document.createElement("article");
    card.className = "meta-card";

    const labelNode = document.createElement("span");
    labelNode.className = "label";
    labelNode.textContent = label;

    const valueNode = document.createElement(label === "Pages URL" ? "a" : "div");
    valueNode.className = "value";
    if (label === "Pages URL") {
      valueNode.href = value;
      valueNode.className = "value link";
    }
    valueNode.textContent = value;

    card.appendChild(labelNode);
    card.appendChild(valueNode);
    elements.metaGrid.appendChild(card);
  }
}

function summarizeStatus() {
  if (!state.environment) {
    return "Environment detection has not run yet.";
  }

  if (!state.probes.webgpu) {
    return "Environment captured. Run the WebGPU probe to see whether the repository can stay on the GPU path.";
  }

  if (!state.probes.webgpu.available) {
    return "Environment captured, but WebGPU is not available. The exported JSON records a fallback path so you can keep the run reproducible.";
  }

  if (!state.probes.frame || !state.probes.worker) {
    return "WebGPU is available. Run the frame and worker probes next to capture baseline responsiveness metrics.";
  }

  return "Environment, WebGPU, frame pacing, and worker round-trip probes are complete. Promote this JSON into reports/raw after validating it against the intended workload.";
}

function renderStatus() {
  const badges = [];

  badges.push({
    tone: state.environment ? "success" : "warn",
    text: state.environment ? "Environment ready" : "Environment pending"
  });

  if (!state.probes.webgpu) {
    badges.push({ tone: "warn", text: "WebGPU probe pending" });
  } else if (state.probes.webgpu.available) {
    badges.push({ tone: "success", text: "WebGPU available" });
  } else {
    badges.push({ tone: "danger", text: "WebGPU unavailable" });
  }

  badges.push({
    tone: state.probes.frame ? "success" : "warn",
    text: state.probes.frame ? "Frame probe done" : "Frame probe pending"
  });
  badges.push({
    tone: state.probes.worker ? "success" : "warn",
    text: state.probes.worker ? "Worker probe done" : "Worker probe pending"
  });

  elements.statusRow.innerHTML = "";
  for (const badge of badges) {
    const node = document.createElement("span");
    node.className = "badge " + badge.tone;
    node.textContent = badge.text;
    elements.statusRow.appendChild(node);
  }

  elements.statusSummary.textContent = summarizeStatus();
}

function metricCards() {
  const cards = [];
  cards.push(["TTI", round(performance.now() - state.startedAt, 1) ? round(performance.now() - state.startedAt, 1) + " ms" : "pending"]);

  if (state.probes.webgpu) {
    cards.push(["WebGPU Init", state.probes.webgpu.initMs ? round(state.probes.webgpu.initMs, 1) + " ms" : state.probes.webgpu.available ? "ready" : "fallback"]);
  } else {
    cards.push(["WebGPU Init", "pending"]);
  }

  if (state.probes.frame) {
    cards.push(["Avg FPS", round(state.probes.frame.avgFps, 1) + " fps"]);
    cards.push(["P95 Frame", round(state.probes.frame.p95FrameMs, 2) + " ms"]);
  } else {
    cards.push(["Avg FPS", "pending"]);
    cards.push(["P95 Frame", "pending"]);
  }

  if (state.probes.worker) {
    cards.push(["Worker RTT", round(state.probes.worker.avgRttMs, 2) + " ms"]);
    cards.push(["Worker P95", round(state.probes.worker.p95RttMs, 2) + " ms"]);
  } else {
    cards.push(["Worker RTT", "pending"]);
    cards.push(["Worker P95", "pending"]);
  }

  return cards;
}

function renderMetrics() {
  elements.metricsGrid.innerHTML = "";

  for (const [label, value] of metricCards()) {
    const card = document.createElement("article");
    card.className = "metric-card";

    const labelNode = document.createElement("span");
    labelNode.className = "label";
    labelNode.textContent = label;

    const valueNode = document.createElement("div");
    valueNode.className = "value";
    valueNode.textContent = value;

    card.appendChild(labelNode);
    card.appendChild(valueNode);
    elements.metricsGrid.appendChild(card);
  }
}

function renderLogs() {
  elements.activityLog.innerHTML = "";

  if (!state.logs.length) {
    const li = document.createElement("li");
    li.textContent = "No probe activity yet.";
    elements.activityLog.appendChild(li);
    return;
  }

  for (const item of state.logs) {
    const li = document.createElement("li");
    li.textContent = item;
    elements.activityLog.appendChild(li);
  }
}

function schemaResult() {
  const environment = ensureEnvironment();
  const webgpu = state.probes.webgpu;

  if (webgpu) {
    environment.backend = webgpu.available ? "webgpu" : "wasm";
    environment.fallback_triggered = !webgpu.available;
    environment.gpu = {
      adapter: webgpu.adapter || "unknown",
      required_features: webgpu.features || [],
      limits: webgpu.limits || {}
    };
  }

  environment.worker_mode = state.probes.worker ? "worker" : "main";

  const initMs = webgpu && webgpu.initMs ? round(webgpu.initMs, 2) : round(performance.now() - state.startedAt, 2);
  const successRate = webgpu ? (webgpu.available ? 1 : 0) : 0.5;
  const errorType = webgpu && webgpu.error ? webgpu.error : "";

  return {
    meta: {
      repo: metadata.repo,
      commit: "bootstrap-generated",
      timestamp: nowIso(),
      owner: "ai-webgpu-lab",
      track: metadata.trackSlug,
      scenario: "baseline-probe",
      notes: metadata.purpose + ". Replace generic probes with workload-specific logic before treating this as a final benchmark."
    },
    environment,
    workload: {
      kind: metadata.workloadKind,
      name: metadata.repo + " baseline probe",
      input_profile: "bootstrap-default"
    },
    metrics: {
      common: {
        time_to_interactive_ms: round(performance.now() - state.startedAt, 2),
        init_ms: initMs,
        success_rate: successRate,
        peak_memory_note: navigator.deviceMemory ? String(navigator.deviceMemory) + " GB reported by browser" : "deviceMemory unavailable",
        error_type: errorType
      }
    },
    status: webgpu ? (webgpu.available ? "success" : "partial") : "partial",
    artifacts: {
      deploy_url: metadata.pagesUrl
    }
  };
}

function renderJson() {
  const environment = state.environment || baseEnvironment();
  elements.environmentJson.textContent = JSON.stringify(environment, null, 2);
  elements.resultJson.textContent = JSON.stringify(schemaResult(), null, 2);
}

async function detectEnvironment() {
  ensureEnvironment();
  log("Captured base environment snapshot.");
  render();
}

function extractLimits(source) {
  const limits = {};

  if (!source) {
    return limits;
  }

  for (const key of knownLimitKeys) {
    if (key in source && Number.isFinite(source[key])) {
      limits[key] = Number(source[key]);
    }
  }

  return limits;
}

async function runWebgpuProbe() {
  ensureEnvironment();
  const startedAt = performance.now();

  if (!("gpu" in navigator)) {
    state.probes.webgpu = {
      available: false,
      initMs: performance.now() - startedAt,
      error: "navigator.gpu unavailable",
      adapter: "unavailable",
      features: [],
      limits: {}
    };
    log("WebGPU probe failed: navigator.gpu is not available in this browser.");
    render();
    return;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("No GPU adapter returned");
    }

    let adapterInfo = null;
    if (typeof adapter.requestAdapterInfo === "function") {
      try {
        adapterInfo = await adapter.requestAdapterInfo();
      } catch (error) {
        adapterInfo = null;
      }
    }

    const device = await adapter.requestDevice();
    const adapterName = (adapterInfo && (adapterInfo.description || adapterInfo.vendor || adapterInfo.architecture)) || "WebGPU adapter";
    const features = Array.from(device.features || []);
    const limits = extractLimits(device.limits || adapter.limits);

    state.probes.webgpu = {
      available: true,
      initMs: performance.now() - startedAt,
      adapter: adapterName,
      features,
      limits
    };
    log("WebGPU probe succeeded with adapter: " + adapterName + ".");
  } catch (error) {
    state.probes.webgpu = {
      available: false,
      initMs: performance.now() - startedAt,
      error: error instanceof Error ? error.message : String(error),
      adapter: "unavailable",
      features: [],
      limits: {}
    };
    log("WebGPU probe failed: " + state.probes.webgpu.error + ".");
  }

  render();
}

async function runFrameProbe() {
  ensureEnvironment();
  const deltas = [];

  await new Promise((resolve) => {
    let previous = 0;
    function step(timestamp) {
      if (previous !== 0) {
        deltas.push(timestamp - previous);
      }
      previous = timestamp;

      if (deltas.length >= 120) {
        resolve();
        return;
      }

      requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
  });

  const avgDelta = deltas.reduce((total, value) => total + value, 0) / deltas.length;
  state.probes.frame = {
    avgFrameMs: avgDelta,
    avgFps: avgDelta > 0 ? 1000 / avgDelta : 0,
    p95FrameMs: percentile(deltas, 0.95)
  };
  log("Frame probe captured " + deltas.length + " frames.");
  render();
}

async function runWorkerProbe() {
  ensureEnvironment();
  const workerScript = "self.onmessage = (event) => { if (event.data === 'ping') { self.postMessage(performance.now()); } };";
  const workerUrl = URL.createObjectURL(new Blob([workerScript], { type: "text/javascript" }));
  const probeWorker = new Worker(workerUrl);
  const roundTrips = [];

  try {
    for (let index = 0; index < 20; index += 1) {
      const sample = await new Promise((resolve, reject) => {
        const startedAt = performance.now();
        const timeout = setTimeout(() => reject(new Error("Worker probe timed out")), 2000);

        probeWorker.onmessage = () => {
          clearTimeout(timeout);
          resolve(performance.now() - startedAt);
        };

        probeWorker.postMessage("ping");
      });
      roundTrips.push(sample);
    }

    const avgRtt = roundTrips.reduce((total, value) => total + value, 0) / roundTrips.length;
    state.probes.worker = {
      avgRttMs: avgRtt,
      p95RttMs: percentile(roundTrips, 0.95)
    };
    log("Worker probe completed with " + roundTrips.length + " round-trips.");
  } catch (error) {
    state.probes.worker = {
      avgRttMs: null,
      p95RttMs: null,
      error: error instanceof Error ? error.message : String(error)
    };
    log("Worker probe failed: " + state.probes.worker.error + ".");
  } finally {
    probeWorker.terminate();
    URL.revokeObjectURL(workerUrl);
  }

  render();
}

function downloadJson() {
  const payload = JSON.stringify(schemaResult(), null, 2);
  const blob = new Blob([payload], { type: "application/json" });
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = metadata.repo + "-baseline-probe.json";
  anchor.click();
  URL.revokeObjectURL(objectUrl);
  log("Downloaded schema-aligned baseline JSON draft.");
}

function render() {
  renderMeta();
  renderStatus();
  renderMetrics();
  renderJson();
}

elements.detectEnvironment.addEventListener("click", detectEnvironment);
elements.runWebgpu.addEventListener("click", runWebgpuProbe);
elements.runFrame.addEventListener("click", runFrameProbe);
elements.runWorker.addEventListener("click", runWorkerProbe);
elements.downloadJson.addEventListener("click", downloadJson);

renderList(elements.focusList, focusItems());
renderList(elements.nextSteps, nextSteps());
log("Baseline probe ready. Capture environment first, then run WebGPU, frame, and worker probes.");
detectEnvironment();
render();
