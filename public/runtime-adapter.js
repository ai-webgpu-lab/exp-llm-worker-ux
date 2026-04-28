// Runtime adapter contract for bench-runtime-shootout.
//
// A real runtime (WebLLM, Transformers.js, ORT-Web, etc.) graduates from the
// deterministic harness by implementing this shape and registering itself
// before app.js boots:
//
//   window.__aiWebGpuLabRuntimeRegistry.register(myAdapter);
//
// The harness consults the registry only when it needs adapter metadata for
// the result draft. Existing deterministic scenarios stay unchanged.

class RuntimeAdapterRegistry {
  constructor() {
    this.adapters = new Map();
    this.deterministic = {
      id: "deterministic-mock",
      label: "Deterministic Mock",
      version: "1.0.0",
      capabilities: ["prefill", "decode", "fixed-output-budget"],
      loadType: "synchronous",
      isReal: false
    };
  }

  register(adapter) {
    if (!adapter || typeof adapter !== "object") {
      throw new Error("adapter must be an object");
    }
    for (const field of ["id", "label", "version"]) {
      if (typeof adapter[field] !== "string" || !adapter[field]) {
        throw new Error(`adapter.${field} is required`);
      }
    }
    for (const method of ["loadRuntime", "prefill", "decode"]) {
      if (typeof adapter[method] !== "function") {
        throw new Error(`adapter.${method} must be a function`);
      }
    }
    this.adapters.set(adapter.id, {
      ...adapter,
      isReal: true,
      capabilities: Array.isArray(adapter.capabilities) ? adapter.capabilities : []
    });
    return adapter.id;
  }

  list() {
    return [...this.adapters.values()];
  }

  describe(modeId) {
    const reportRealAdapter = modeId === "adapter-stub" || (typeof modeId === "string" && modeId.startsWith("real-"));
    if (reportRealAdapter) {
      const registered = [...this.adapters.values()];
      if (registered.length === 0) {
        return {
          id: "stub-not-connected",
          label: "Adapter Stub (not connected)",
          status: "not-connected",
          isReal: false,
          version: "n/a",
          capabilities: this.deterministic.capabilities,
          loadType: "stub",
          message: `No real runtime adapter has registered for mode='${modeId}'. Falling back to the deterministic harness.`
        };
      }
      const primary = registered[0];
      return {
        id: primary.id,
        label: primary.label,
        status: "connected",
        isReal: true,
        version: primary.version,
        capabilities: primary.capabilities,
        loadType: primary.loadType || "async",
        message: `Real runtime adapter '${primary.id}' is connected.`
      };
    }
    return {
      ...this.deterministic,
      status: "deterministic",
      message: "Deterministic harness — replace by registering a real adapter."
    };
  }
}

if (typeof window !== "undefined") {
  if (!window.__aiWebGpuLabRuntimeRegistry) {
    window.__aiWebGpuLabRuntimeRegistry = new RuntimeAdapterRegistry();
  }
}

export { RuntimeAdapterRegistry };
