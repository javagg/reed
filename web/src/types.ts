export type ExampleName = 'ex1_volume' | 'ex2_surface' | 'ex3_volume_combined' | 'poisson'

export type BackendName = 'wasm-cpu'

export type RunPayload = {
  backend: BackendName
  example: ExampleName
  dim: number
  nelem: number
  p: number
  q: number
}

export type WorkerLog = {
  type: 'log'
  jobId: string
  message: string
}

export type RunResult = {
  backend: BackendName
  example: string
  dim: number
  nelem: number
  p: number
  q: number
  logs: string[]
  value: number
  expected?: number
  error?: number
  duration_ms: number
}

export type WorkerResult = {
  type: 'result'
  jobId: string
  data: RunResult
}

export type WorkerError = {
  type: 'error'
  jobId: string
  message: string
}

export type WorkerMessage = WorkerLog | WorkerResult | WorkerError
