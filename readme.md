## How to run examples

Current status: ex1/ex2/ex3 examples support 1D, 2D, and 3D with current Reed gallery QFunctions.

```bash
cargo run --example ex1_volume
cargo run --example ex2_surface
cargo run --example ex3_volume_combined
cargo run --example poisson
```

### ex1_volume optional args

```bash
cargo run --example ex1_volume -- --dim 1 --nelem 8 --p 2 --q 4
cargo run --example ex1_volume -- --dim 2 --nelem 8 --p 2 --q 4
cargo run --example ex1_volume -- --dim 3 --nelem 8 --p 2 --q 4
```

### ex2_surface optional args

```bash
cargo run --example ex2_surface -- --dim 1 --nelem 8 --p 2 --q 4
cargo run --example ex2_surface -- --dim 2 --nelem 8 --p 2 --q 4
cargo run --example ex2_surface -- --dim 3 --nelem 8 --p 2 --q 4
cargo run --example ex2_surface -- --dim 3 --nelem 2 --p 2 --q 4 --study --levels 4
```

Note: for 2D/3D `ex2_surface`, the current discrete surface estimate follows the same first-order trend observed in earlier 2D runs, so validation uses a mesh-dependent convergence tolerance.
`--study` doubles `nelem` for each level and prints error/rate/tolerance for convergence inspection.

### ex3_volume_combined optional args

```bash
cargo run --example ex3_volume_combined -- --dim 1 --nelem 8 --p 2 --q 4
cargo run --example ex3_volume_combined -- --dim 2 --nelem 8 --p 2 --q 4
cargo run --example ex3_volume_combined -- --dim 3 --nelem 8 --p 2 --q 4
```

### poisson optional args

```bash
cargo run --example poisson -- --dim 1 --nelem 4 --p 2 --q 4
cargo run --example poisson -- --dim 2 --nelem 4 --p 2 --q 4
cargo run --example poisson -- --dim 3 --nelem 4 --p 2 --q 4
```

## CPU benchmarks

Criterion benchmarks are available at the workspace root and focus on the CPU backend hot path.

```bash
cargo bench --bench cpu_backend
```

To compare parallel vs. serial CPU backend builds with the same benchmark harness:

```bash
./scripts/bench_cpu_compare.sh
```

Current benchmark groups:

- `cpu_poisson_apply`
- `cpu_combined_apply`

## WASM compile

CPU backend can be compiled for `wasm32-unknown-unknown`.

```bash
cargo check --target wasm32-unknown-unknown -p reed-cpu
cargo check --target wasm32-unknown-unknown -p reed
```

## WGPU backend

A new WGPU backend crate is available at `crates/wgpu` and can be enabled with a feature flag.

Build checks:

```bash
cargo check -p reed --features wgpu-backend
cargo check -p reed-wgpu
```

Resource strings:

- CPU: `/cpu/self`
- WGPU: `/gpu/wgpu`

Current implementation stage: backend initialization and object factory wiring are in place, with incremental compute-kernel migration now active.

Progress update:

- `WgpuVector` now includes real compute-shader execution for `set_value`, `scale`, and `axpy` on `f32` vectors.
- `WgpuElemRestriction` now includes a real compute-shader path for `NoTranspose` with offset restrictions on `f32` data.
- `WgpuBasis` now includes a real compute-shader path for `EvalMode::Interp` with `transpose = false` on `f32` data.
- Other numeric types and unimplemented objects still use fallback paths, preserving compatibility while GPU coverage expands incrementally.

## Web benchmark UI

The `web` directory contains a Vue 3 + Vite test UI where you can:

- Select the example (`ex1_volume`, `ex2_surface`, `ex3_volume_combined`, `poisson`)
- Configure `dim`, `nelem`, `p`, `q`
- Run in a Web Worker with WASM backend (`wasm-cpu`)
- View runtime logs, result values, and duration
- Accumulate run history for future backend performance comparison

```bash
cd web
npm install
npm run wasm:build
npm run dev
```

One-command local start:

```bash
cd web
npm run dev:bench
```