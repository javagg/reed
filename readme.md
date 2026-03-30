## How to run examples

Current status: ex1/ex2/ex3 examples support 1D, 2D, and 3D with current Reed gallery QFunctions.

```bash
cargo run --example ex1_volume
cargo run --example ex2_surface
cargo run --example ex3_volume_combined
cargo run --example poisson_1d
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