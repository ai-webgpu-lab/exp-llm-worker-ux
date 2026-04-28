self.addEventListener("message", async (event) => {
  if (!event.data || event.data.type !== "run") return;

  const { promptTokens, outputTokens, profile } = event.data;
  const startedAt = performance.now();

  await sleep(profile.initDelayMs);
  const initMs = performance.now() - startedAt;

  const prefillStartedAt = performance.now();
  let consumed = 0;
  while (consumed < promptTokens.length) {
    consumed += profile.prefillChunk;
    await sleep(profile.prefillDelayMs);
  }
  const prefillMs = performance.now() - prefillStartedAt;

  const decodeStartedAt = performance.now();
  let emitted = 0;
  let ttftMs = 0;
  while (emitted < outputTokens.length) {
    await sleep(profile.decodeDelayMs);
    if (emitted === 0) ttftMs = performance.now() - decodeStartedAt;
    const chunk = outputTokens.slice(emitted, emitted + profile.decodeChunk);
    emitted += chunk.length;
    self.postMessage({ type: "chunk", text: chunk.join(" ") });
  }
  const decodeMs = performance.now() - decodeStartedAt;

  self.postMessage({
    type: "complete",
    result: {
      initMs,
      ttftMs,
      prefillTokPerSec: promptTokens.length / Math.max(prefillMs / 1000, 0.001),
      decodeTokPerSec: outputTokens.length / Math.max(decodeMs / 1000, 0.001),
      turnLatencyMs: initMs + prefillMs + decodeMs
    }
  });
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
