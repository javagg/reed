#[cfg(feature = "wgpu-backend")]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
#[cfg(feature = "wgpu-backend")]
use reed::{EvalMode, QuadMode, Reed, TransposeMode};

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
fn bench_compare_vector_axpy(c: &mut Criterion) {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let mut group = c.benchmark_group("compare_vector_axpy");

    for &size in &[4_096usize, 65_536, 1_048_576] {
        for &(backend, reed) in &[("cpu", &reed_cpu), ("wgpu", &reed_gpu)] {
            group.bench_with_input(
                BenchmarkId::new(backend, size),
                &size,
                |b, &size| {
                    let x_seed = vec![0.5_f32; size];
                    let y_seed = vec![1.0_f32; size];
                    let x = reed.vector_from_slice(&x_seed).unwrap();
                    let mut y = reed.vector_from_slice(&y_seed).unwrap();
                    b.iter(|| {
                        y.copy_from_slice(&y_seed).unwrap();
                        y.axpy(black_box(2.0_f32), &*x).unwrap();
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(feature = "wgpu-backend")]
fn bench_compare_restriction(c: &mut Criterion) {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let mut group = c.benchmark_group("compare_restriction_gather");

    for &(nelem, elemsize) in &[(16_384usize, 4usize), (32_768, 8)] {
        let lsize = nelem * (elemsize - 1) + 1;
        let offsets = build_offsets_1d(nelem, elemsize);
        for &(backend, reed) in &[("cpu", &reed_cpu), ("wgpu", &reed_gpu)] {
            let restriction = reed
                .elem_restriction(nelem, elemsize, 1, 1, lsize, &offsets)
                .unwrap();
            let global = (0..lsize)
                .map(|index| ((index % 97) as f32 - 48.0) * 0.125)
                .collect::<Vec<_>>();
            let mut local = vec![0.0_f32; nelem * elemsize];
            group.bench_with_input(
                BenchmarkId::new(backend, format!("{nelem}x{elemsize}")),
                &(nelem, elemsize),
                |b, _| {
                    b.iter(|| {
                        restriction
                            .apply(
                                TransposeMode::NoTranspose,
                                black_box(&global),
                                black_box(&mut local),
                            )
                            .unwrap();
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(feature = "wgpu-backend")]
fn bench_compare_basis_interp(c: &mut Criterion) {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let mut group = c.benchmark_group("compare_basis_interp");

    for &(p, q, num_elem, transpose) in &[
        (4usize, 6usize, 8_192usize, false),
        (4usize, 6usize, 8_192usize, true),
        (8usize, 10usize, 4_096usize, false),
        (8usize, 10usize, 4_096usize, true),
    ] {
        for &(backend, reed) in &[("cpu", &reed_cpu), ("wgpu", &reed_gpu)] {
            let basis = reed
                .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
                .unwrap();
            let input_len = if transpose {
                num_elem * basis.num_qpoints()
            } else {
                num_elem * basis.num_dof()
            };
            let output_len = if transpose {
                num_elem * basis.num_dof()
            } else {
                num_elem * basis.num_qpoints()
            };
            let input = (0..input_len)
                .map(|index| ((index % 31) as f32 - 15.0) * 0.0625)
                .collect::<Vec<_>>();
            let mut output = vec![0.0_f32; output_len];
            let case = if transpose {
                format!("p{p}_q{q}_e{num_elem}_t")
            } else {
                format!("p{p}_q{q}_e{num_elem}_f")
            };

            group.bench_with_input(BenchmarkId::new(backend, case), &num_elem, |b, _| {
                b.iter(|| {
                    basis
                        .apply(
                            num_elem,
                            transpose,
                            EvalMode::Interp,
                            black_box(&input),
                            black_box(&mut output),
                        )
                        .unwrap();
                });
            });
        }
    }

    group.finish();
}

#[cfg(feature = "wgpu-backend")]
criterion_group!(
    benches,
    bench_compare_vector_axpy,
    bench_compare_restriction,
    bench_compare_basis_interp
);
#[cfg(feature = "wgpu-backend")]
criterion_main!(benches);

#[cfg(not(feature = "wgpu-backend"))]
fn main() {
    eprintln!("Run this benchmark with: cargo bench --features wgpu-backend --bench backend_compare");
}