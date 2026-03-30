<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import type {
  BackendName,
  ExampleName,
  RunPayload,
  RunResult,
  WorkerMessage,
} from './types'

type RunHistoryItem = {
  id: string
  status: 'success' | 'failed'
  payload: RunPayload
  durationMs?: number
  value?: number
  error?: number
  message?: string
}

const examples: Array<{ label: string; value: ExampleName }> = [
  { label: 'ex1_volume', value: 'ex1_volume' },
  { label: 'ex2_surface', value: 'ex2_surface' },
  { label: 'ex3_volume_combined', value: 'ex3_volume_combined' },
  { label: 'poisson', value: 'poisson' },
]

const backends: Array<{ label: string; value: BackendName }> = [
  { label: 'WASM CPU (current)', value: 'wasm-cpu' },
]

const payload = ref<RunPayload>({
  backend: 'wasm-cpu',
  example: 'ex1_volume',
  dim: 3,
  nelem: 8,
  p: 2,
  q: 4,
})

const logs = ref<string[]>([])
const isRunning = ref(false)
const statusText = ref('Idle')
const runResult = ref<RunResult | null>(null)
const currentJobId = ref<string>('')
const runHistory = ref<RunHistoryItem[]>([])

const worker = new Worker(new URL('./workers/runnerWorker.ts', import.meta.url), {
  type: 'module',
})

const canRun = computed(() => {
  const p = payload.value
  return !isRunning.value && p.dim >= 1 && p.dim <= 3 && p.nelem >= 1 && p.p >= 2 && p.q >= 1
})

worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const message = event.data
  if (!message || !('type' in message)) {
    return
  }

  if (!currentJobId.value || message.jobId !== currentJobId.value) {
    return
  }

  if (message.type === 'log') {
    logs.value.push(message.message)
    statusText.value = message.message
    return
  }

  isRunning.value = false
  if (message.type === 'result') {
    runResult.value = message.data
    statusText.value = 'Done'
    logs.value.push(`Finished in ${message.data.duration_ms.toFixed(3)} ms`)
    runHistory.value.unshift({
      id: currentJobId.value,
      status: 'success',
      payload: { ...payload.value },
      durationMs: message.data.duration_ms,
      value: message.data.value,
      error: message.data.error,
    })
  } else {
    runResult.value = null
    statusText.value = 'Failed'
    logs.value.push(`Error: ${message.message}`)
    runHistory.value.unshift({
      id: currentJobId.value,
      status: 'failed',
      payload: { ...payload.value },
      message: message.message,
    })
  }
}

function run() {
  if (!canRun.value) {
    return
  }

  const jobId = `${Date.now()}-${Math.random().toString(16).slice(2)}`
  currentJobId.value = jobId
  isRunning.value = true
  statusText.value = 'Submitting task'
  runResult.value = null
  logs.value = [
    `Submit ${payload.value.example} on ${payload.value.backend}`,
    `dim=${payload.value.dim}, nelem=${payload.value.nelem}, p=${payload.value.p}, q=${payload.value.q}`,
  ]

  worker.postMessage({
    type: 'run',
    jobId,
    payload: { ...payload.value },
  })
}

function clearLogs() {
  logs.value = []
}

onBeforeUnmount(() => {
  worker.terminate()
})
</script>

<template>
  <main class="page">
    <header class="hero">
      <p class="kicker">Reed Web Bench</p>
      <h1>WASM Worker Test Console</h1>
      <p class="subtitle">选择示例后在 Web Worker 中运行 WASM，查看运行过程、结果与耗时，为后续后端切换性能比较做基线。</p>
    </header>

    <section class="panel controls">
      <h2>Run Config</h2>
      <div class="grid">
        <label>
          Backend
          <select v-model="payload.backend">
            <option v-for="item in backends" :key="item.value" :value="item.value">{{ item.label }}</option>
          </select>
        </label>

        <label>
          Example
          <select v-model="payload.example">
            <option v-for="item in examples" :key="item.value" :value="item.value">{{ item.label }}</option>
          </select>
        </label>

        <label>
          Dimension
          <select v-model.number="payload.dim">
            <option :value="1">1D</option>
            <option :value="2">2D</option>
            <option :value="3">3D</option>
          </select>
        </label>

        <label>
          nelem
          <input v-model.number="payload.nelem" type="number" min="1" step="1" />
        </label>

        <label>
          p
          <input v-model.number="payload.p" type="number" min="2" step="1" />
        </label>

        <label>
          q
          <input v-model.number="payload.q" type="number" min="1" step="1" />
        </label>
      </div>

      <div class="actions">
        <button class="run" :disabled="!canRun" @click="run">
          {{ isRunning ? 'Running...' : 'Run Example' }}
        </button>
        <button class="clear" :disabled="isRunning" @click="clearLogs">Clear Logs</button>
        <span class="status">Status: {{ statusText }}</span>
      </div>
    </section>

    <section class="layout">
      <article class="panel logs">
        <h2>Runtime Logs</h2>
        <pre>{{ logs.join('\n') || 'No logs yet.' }}</pre>
      </article>

      <article class="panel result">
        <h2>Run Result</h2>
        <div v-if="runResult" class="metrics">
          <p><strong>Example:</strong> {{ runResult.example }}</p>
          <p><strong>Backend:</strong> {{ runResult.backend }}</p>
          <p><strong>Value:</strong> {{ runResult.value.toFixed(12) }}</p>
          <p v-if="runResult.expected !== undefined"><strong>Expected:</strong> {{ runResult.expected.toFixed(12) }}</p>
          <p v-if="runResult.error !== undefined"><strong>Error:</strong> {{ runResult.error.toExponential(6) }}</p>
          <p><strong>Duration:</strong> {{ runResult.duration_ms.toFixed(3) }} ms</p>
        </div>
        <p v-else class="empty">No result yet.</p>
      </article>
    </section>

    <section class="panel history">
      <h2>History (for backend comparison)</h2>
      <table>
        <thead>
          <tr>
            <th>status</th>
            <th>backend</th>
            <th>example</th>
            <th>dim</th>
            <th>nelem/p/q</th>
            <th>duration(ms)</th>
            <th>error</th>
            <th>message</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in runHistory" :key="item.id">
            <td :class="item.status">{{ item.status }}</td>
            <td>{{ item.payload.backend }}</td>
            <td>{{ item.payload.example }}</td>
            <td>{{ item.payload.dim }}</td>
            <td>{{ item.payload.nelem }}/{{ item.payload.p }}/{{ item.payload.q }}</td>
            <td>{{ item.durationMs?.toFixed(3) ?? '-' }}</td>
            <td>{{ item.error !== undefined ? item.error.toExponential(3) : '-' }}</td>
            <td>{{ item.message ?? '-' }}</td>
          </tr>
          <tr v-if="runHistory.length === 0">
            <td colspan="8" class="empty">No runs yet.</td>
          </tr>
        </tbody>
      </table>
    </section>
  </main>
</template>
