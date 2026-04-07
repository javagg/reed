let wasmModule = null;
let wasmReady = false;

async function ensureWasm(jobId) {
  if (wasmReady && wasmModule) {
    return;
  }

  const postLog = (message) => {
    const logMessage = { type: 'log', jobId, message };
    self.postMessage(logMessage);
  };

  postLog('Loading WASM module...');

  const modulePath = '/wasm/reed_wasm_runner.js';
  postLog(`Import JS glue from ${modulePath}`);
  wasmModule = await import(/* @vite-ignore */ modulePath);
  postLog('Instantiate WASM binary');
  await wasmModule.default('/wasm/reed_wasm_runner_bg.wasm');

  wasmReady = true;
  postLog('WASM module initialized');
}

self.onmessage = async (event) => {
  const req = event.data;
  if (!req || req.type !== 'run') {
    return;
  }

  const { jobId, payload } = req;

  try {
    await ensureWasm(jobId);

    // ── wgpu: one-time async init ──────────────────────────────────────────────
    if (payload.backend === 'wasm-gpu') {
      const initMsg = {
        type: 'log',
        jobId,
        message: `Initializing wgpu for ${payload.example} in ${payload.dim}D...`,
      };
      self.postMessage(initMsg);

      const adapterName = await wasmModule.init_wgpu();
      const readyMsg = {
        type: 'log',
        jobId,
        message: `wgpu ready: ${adapterName}`,
      };
      self.postMessage(readyMsg);
    }

    const startMsg = {
      type: 'log',
      jobId,
      message: `Running ${payload.example} in ${payload.dim}D on ${payload.backend} ...`,
    };
    self.postMessage(startMsg);

    // run_example takes a single args object (wasm-bindgen compatible)
    const t0 = performance.now();
    const data = wasmModule.run_example({
      backend: payload.backend,
      example: payload.example,
      dim: payload.dim,
      nelem: payload.nelem,
      p: payload.p,
      q: payload.q,
    });
    const t1 = performance.now();

    const workerTimingLog = {
      type: 'log',
      jobId,
      message: `Worker wall time: ${(t1 - t0).toFixed(3)} ms`,
    };
    self.postMessage(workerTimingLog);

    if (Array.isArray(data.logs)) {
      for (const log of data.logs) {
        const detail = { type: 'log', jobId, message: log };
        self.postMessage(detail);
      }
    }

    // data.backend is now set by run_example itself

    const done = {
      type: 'result',
      jobId,
      data,
    };
    self.postMessage(done);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const err = { type: 'error', jobId, message };
    self.postMessage(err);
  }
};
