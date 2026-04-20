## How to run examples

Current status: ex1/ex2/ex3 examples support 1D, 2D, and 3D with Reed gallery QFunctions. Named entries under `Reed::q_function_by_name` match libCEED’s `ceed-gallery-list.h` where implemented (see `design_mapping.md`, section 5).

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
- `cpu_basis_apply` (includes `Interp` / `Grad` / **`Div`** / **`Curl`** on tensor H1 Lagrange, vector fields `ncomp = dim` for differential modes)
- `cpu_simplex_basis_apply` (`Interp` / `Grad` on triangle / tet `basis_h1_simplex`, scalar `ncomp = 1`)

## Native WGPU microbenchmarks

Criterion benchmarks are also available for the currently GPU-backed native WGPU paths.

```bash
cargo bench --features wgpu-backend --bench wgpu_backend
```

Current WGPU benchmark groups:

- `wgpu_vector_ops`
- `wgpu_elem_restriction`
- `wgpu_basis_interp`
- `wgpu_basis_grad`
- `wgpu_basis_div_curl` (tensor H1 Lagrange, `ncomp = dim`, forward `Div` / `Curl` on `f32`)

Direct CPU/WGPU comparison benchmark:

```bash
cargo bench --features wgpu-backend --bench backend_compare
```

Current comparison groups:

- `compare_vector_axpy`
- `compare_restriction_gather`
- `compare_restriction_transpose`
- `compare_basis_interp`
- `compare_basis_grad`
- `compare_basis_grad_2d` (tensor H1 Lagrange, `dim = 2`, `ncomp = 1`)
- `compare_basis_div_2d` (`ncomp = dim = 2`, forward `Div` only)

These are microbenchmarks for the portions that currently execute real WGPU compute kernels on native `f32` data. They are not end-to-end operator benchmarks yet, because operator assembly/QFunction execution still routes through CPU code paths.

GPU coverage for tensor H1 Lagrange on native **`f32`** includes **`EvalMode::Interp`**, **`Grad`**, **`Div`**, and **`Curl`** (forward and transpose where implemented), with quadrature-side vector layout aligned to CPU `LagrangeBasis` (`iq · (ncomp·dim) + comp·dim + dir`). That path reduces CPU fallback for those basis evaluations; full operators still use the CPU gallery/QFunction path unless otherwise noted.

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

- `WgpuVector` includes compute-shader execution for `set_value`, `scale`, and `axpy` on `f32` vectors.
- `WgpuElemRestriction` includes compute-shader paths for offset-based restrictions on `f32`: gather (`NoTranspose`) and transpose scatter (`Transpose`; serial single-thread kernel for portability on Metal).
- `WgpuBasis` runs `Interp`, `Grad`, `Div`, and `Curl` on `f32` for tensor H1 Lagrange when `ncomp` matches the differential mode (including transpose for these modes), matching CPU `LagrangeBasis` in integration tests.
- Non-`f32` types and other objects still use CPU fallbacks where GPU paths are not implemented.

For a prioritized list of next steps toward libCEED feature parity (QFunction device path, strided restriction on GPU, AtPoints notes, etc.), see **`design_mapping.md` §8.1**.

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