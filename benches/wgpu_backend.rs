#[cfg(feature = "wgpu-backend")]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
#[cfg(feature = "wgpu-backend")]
use reed::{EvalMode, QuadMode, Reed};

#[cfg(feature = "wgpu-backend")]
fn build_offsets_1d(nelem: usize, elemsize: usize) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(nelem * elemsize);
    for elem in 0..nelem {
        let start = elem * (elemsize - 1);
        for local in 0..elemsize {
            offsets.push((start + local) as i32);
        }
    }
    offsets
}

#[cfg(feature = "wgpu-backend")]
fn bench_wgpu_vector_ops(c: &mut Criterion) {
    let reed = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let mut group = c.benchmark_group("wgpu_vector_ops");

    for &size in &[4_096usize, 65_536, 1_048_576] {
        group.bench_with_input(BenchmarkId::new("set_value", size), &size, |b, &size| {
            let mut y = reed.vector(size).unwrap();
            b.iter(|| y.set_value(black_box(1.25_f32)).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("scale", size), &size, |b, &size| {
            let seed = vec![1.0_f32; size];
            let mut y = reed.vector_from_slice(&seed).unwrap();
            b.iter(|| {
                y.copy_from_slice(&seed).unwrap();
                y.scale(black_box(0.5_f32)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("axpy", size), &size, |b, &size| {
            let x_seed = vec![0.5_f32; size];
            let y_seed = vec![1.0_f32; size];
            let x = reed.vector_from_slice(&x_seed).unwrap();
            let mut y = reed.vector_from_slice(&y_seed).unwrap();
            b.iter(|| {
                y.copy_from_slice(&y_seed).unwrap();
                y.axpy(black_box(2.0_f32), &*x).unwrap();
            });
        });
    }

    group.finish();
}

#[cfg(feature = "wgpu-backend")]
fn bench_wgpu_elem_restriction(c: &mut Criterion) {
    let reed = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let mut group = c.benchmark_group("wgpu_elem_restriction");

    for &(nelem, elemsize) in &[(16_384usize, 4usize), (32_768, 8)] {
        let lsize = nelem * (elemsize - 1) + 1;
        let offsets = build_offsets_1d(nelem, elemsize);
        let restriction = reed
            .elem_restriction(nelem, elemsize, 1, 1, lsize, &offsets)
            .unwrap();
        let global = (0..lsize)
            .map(|index| ((index % 97) as f32 - 48.0) * 0.125)
            .collect::<Vec<_>>();
        let mut local = vec![0.0_f32; nelem * elemsize];

        group.bench_with_input(
            BenchmarkId::new("gather_no_transpose", format!("{nelem}x{elemsize}")),
            &(nelem, elemsize),
            |b, _| {
                b.iter(|| {
                    restriction
                        .apply(
                            reed::TransposeMode::NoTranspose,
                            black_box(&global),
                            black_box(&mut local),
                        )
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "wgpu-backend")]
fn bench_wgpu_basis_interp(c: &mut Criterion) {
    let reed = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let mut group = c.benchmark_group("wgpu_basis_interp");

    // 1D benchmarks
    for &(p, q, num_elem) in &[(4usize, 6usize, 8_192usize), (8, 10, 4_096)] {
        let basis = reed
            .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
            .unwrap();
        let input_len = num_elem * basis.num_dof();
        let output_len = num_elem * basis.num_qpoints();
        let input = (0..input_len)
            .map(|index| ((index % 31) as f32 - 15.0) * 0.0625)
            .collect::<Vec<_>>();
        let mut output = vec![0.0_f32; output_len];

        group.bench_with_input(
            BenchmarkId::new("interp", format!("dim1_p{p}_q{q}_e{num_elem}")),
            &(p, q, num_elem),
            |b, _| {
                b.iter(|| {
                    basis
                        .apply(
                            num_elem,
                            false,
                            EvalMode::Interp,
                            black_box(&input),
                            black_box(&mut output),
                        )
                        .unwrap();
                });
            },
        );
    }

    // 2D benchmarks
    for &(p, q, num_elem) in &[(4usize, 6usize, 1_024usize), (6, 8, 256)] {
        let basis = reed
            .basis_tensor_h1_lagrange(2, 1, p, q, QuadMode::Gauss)
            .unwrap();
        let input_len = num_elem * basis.num_dof();
        let output_len = num_elem * basis.num_qpoints();
        let input = (0..input_len)
            .map(|index| ((index % 31) as f32 - 15.0) * 0.0625)
            .collect::<Vec<_>>();
        let mut output = vec![0.0_f32; output_len];

        group.bench_with_input(
            BenchmarkId::new("interp", format!("dim2_p{p}_q{q}_e{num_elem}")),
            &(p, q, num_elem),
            |b, _| {
                b.iter(|| {
                    basis
                        .apply(
                            num_elem,
                            false,
                            EvalMode::Interp,
                            black_box(&input),
                            black_box(&mut output),
                        )
                        .unwrap();
                });
            },
        );
    }

    // 3D benchmarks
    for &(p, q, num_elem) in &[(4usize, 6usize, 128usize), (6, 8, 64)] {
        let basis = reed
            .basis_tensor_h1_lagrange(3, 1, p, q, QuadMode::Gauss)
            .unwrap();
        let input_len = num_elem * basis.num_dof();
        let output_len = num_elem * basis.num_qpoints();
        let input = (0..input_len)
            .map(|index| ((index % 31) as f32 - 15.0) * 0.0625)
            .collect::<Vec<_>>();
        let mut output = vec![0.0_f32; output_len];

        group.bench_with_input(
            BenchmarkId::new("interp", format!("dim3_p{p}_q{q}_e{num_elem}")),
            &(p, q, num_elem),
            |b, _| {
                b.iter(|| {
                    basis
                        .apply(
                            num_elem,
                            false,
                            EvalMode::Interp,
                            black_box(&input),
                            black_box(&mut output),
                        )
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "wgpu-backend")]
criterion_group!(
    benches,
    bench_wgpu_vector_ops,
    bench_wgpu_elem_restriction,
    bench_wgpu_basis_interp
);
#[cfg(feature = "wgpu-backend")]
criterion_main!(benches);

#[cfg(not(feature = "wgpu-backend"))]
fn main() {
    eprintln!("Run this benchmark with: cargo bench --features wgpu-backend --bench wgpu_backend");
}