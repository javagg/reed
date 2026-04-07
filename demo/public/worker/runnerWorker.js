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

    const startMsg = {
      type: 'log',
      jobId,
      message: `Running ${payload.example} in ${payload.dim}D on ${payload.backend} ...`,
    };
    self.postMessage(startMsg);

    const t0 = performance.now();
    const data = wasmModule.run_example(
      payload.example,
      payload.dim,
      payload.nelem,
      payload.p,
      payload.q,
    );
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

    data.backend = payload.backend;

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
