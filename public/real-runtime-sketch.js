// Real worker-UX comparison runtime sketch for exp-llm-worker-ux.
//
// Gated by ?mode=real-worker-ux. Default deterministic harness path is untouched.
// `loadTransformersFromCdn` is parameterized so tests can inject a stub. The
// adapter exposes a `mode` configuration so consumers can compare worker vs
// main-thread execution against the same model.

const DEFAULT_TRANSFORMERS_VERSION = "3.0.0";
const DEFAULT_TRANSFORMERS_CDN = (version) => `https://esm.sh/@huggingface/transformers@${version}`;
const DEFAULT_MODEL_ID = "Xenova/Phi-3-mini-4k-instruct";

export async function loadTransformersFromCdn({ version = DEFAULT_TRANSFORMERS_VERSION } = {}) {
  const transformers = await import(/* @vite-ignore */ DEFAULT_TRANSFORMERS_CDN(version));
  if (!transformers || typeof transformers.pipeline !== "function") {
    throw new Error("transformers module did not expose pipeline()");
  }
  return { transformers, pipeline: transformers.pipeline, env: transformers.env };
}

export function buildRealWorkerUxAdapter({
  pipeline,
  env,
  version = DEFAULT_TRANSFORMERS_VERSION,
  modelId = DEFAULT_MODEL_ID,
  mode = "main",
  workerFactory = null
}) {
  if (typeof pipeline !== "function") {
    throw new Error("buildRealWorkerUxAdapter requires a callable pipeline");
  }
  if (mode !== "main" && mode !== "worker") {
    throw new Error(`unsupported mode '${mode}' (expected 'main' or 'worker')`);
  }
  if (mode === "worker" && typeof workerFactory !== "function") {
    throw new Error("worker mode requires a workerFactory callable");
  }
  const sanitized = modelId.replace(/[^A-Za-z0-9]/g, "-").toLowerCase();
  const id = `worker-ux-${mode}-${sanitized}-${version.replace(/[^0-9]/g, "")}`;
  let runtime = null;
  let worker = null;

  return {
    id,
    label: `Worker-UX ${mode} ${modelId} (Transformers.js ${version})`,
    version,
    capabilities: ["prefill", "decode", "worker-mode-toggle", "fixed-output-budget"],
    loadType: "async",
    backendHint: "webgpu",
    workerMode: mode,
    isReal: true,
    async loadRuntime({ device = "webgpu", dtype = "q4" } = {}) {
      if (env && typeof env === "object") env.allowRemoteModels = true;
      if (mode === "worker") {
        worker = workerFactory({ modelId, device, dtype, version });
        return worker;
      }
      runtime = await pipeline("text-generation", modelId, { device, dtype });
      return runtime;
    },
    async prefill(_runtime, prompt) {
      const startedAt = performance.now();
      const text = typeof prompt === "string" ? prompt : (prompt && prompt.text) || "";
      const promptTokens = text.trim().split(/\s+/).filter(Boolean).length;
      const prefillMs = performance.now() - startedAt;
      return { promptTokens, prefillMs, text };
    },
    async decode(activeRuntime, prefillResult, outputTokenBudget = 64) {
      const startedAt = performance.now();
      let text = "";
      if (mode === "worker") {
        const target = activeRuntime || worker;
        if (!target || typeof target.run !== "function") {
          throw new Error("worker mode requires loadRuntime() to produce a worker with .run()");
        }
        const reply = await target.run({ prompt: prefillResult.text, maxNewTokens: outputTokenBudget });
        text = (reply && reply.text) || "";
      } else {
        const target = activeRuntime || runtime;
        if (!target) {
          throw new Error("real worker-ux adapter requires loadRuntime() before decode()");
        }
        const output = await target(prefillResult.text, { max_new_tokens: outputTokenBudget, return_full_text: false });
        text = Array.isArray(output) && output[0] && output[0].generated_text ? output[0].generated_text : "";
      }
      const decodeMs = performance.now() - startedAt;
      const tokens = text.split(/\s+/).filter(Boolean).length || outputTokenBudget;
      return {
        tokens,
        decodeMs,
        text,
        mode,
        ttftMs: decodeMs / Math.max(tokens, 1),
        decodeTokPerSec: tokens / Math.max(decodeMs / 1000, 0.001)
      };
    }
  };
}

export async function connectRealWorkerUx({
  registry = typeof window !== "undefined" ? window.__aiWebGpuLabRuntimeRegistry : null,
  loader = loadTransformersFromCdn,
  version = DEFAULT_TRANSFORMERS_VERSION,
  modelId = DEFAULT_MODEL_ID,
  mode = "main",
  workerFactory = null
} = {}) {
  if (!registry) {
    throw new Error("runtime registry not available");
  }
  const { pipeline, env } = await loader({ version });
  if (typeof pipeline !== "function") {
    throw new Error("loaded pipeline is not callable");
  }
  const adapter = buildRealWorkerUxAdapter({ pipeline, env, version, modelId, mode, workerFactory });
  registry.register(adapter);
  return { adapter, pipeline, env };
}

if (typeof window !== "undefined" && window.location && typeof window.location.search === "string") {
  const params = new URLSearchParams(window.location.search);
  if (params.get("mode") === "real-worker-ux" && !window.__aiWebGpuLabRealWorkerUxBootstrapping) {
    window.__aiWebGpuLabRealWorkerUxBootstrapping = true;
    const requestedMode = params.get("workerMode") === "worker" ? "worker" : "main";
    connectRealWorkerUx({ mode: requestedMode }).catch((error) => {
      console.warn(`[real-worker-ux] bootstrap failed: ${error.message}`);
      window.__aiWebGpuLabRealWorkerUxBootstrapError = error.message;
    });
  }
}
