use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use reed::{ElemTopology, EvalMode, FieldVector, OperatorTrait, QuadMode, Reed};

struct ApplyScenario {
    input: &'static dyn reed::VectorTrait<f64>,
    output: &'static mut dyn reed::VectorTrait<f64>,
    operator: Box<dyn OperatorTrait<f64>>,
}

struct BasisApplyScenario {
    basis: Box<dyn reed::BasisTrait<f64>>,
    input: Vec<f64>,
    output: Vec<f64>,
    num_elem: usize,
    transpose: bool,
    eval_mode: EvalMode,
}

fn leak_box<T>(value: T) -> &'static mut T {
    Box::leak(Box::new(value))
}

fn build_offsets_1d(nelem_1d: usize, p: usize) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(nelem_1d * p);
    for e in 0..nelem_1d {
        let start = e * (p - 1);
        for j in 0..p {
            offsets.push((start + j) as i32);
        }
    }
    offsets
}

fn build_offsets_2d(nelem_1d: usize, p: usize, ndofs_1d: usize) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(nelem_1d * nelem_1d * p * p);
    for ey in 0..nelem_1d {
        for ex in 0..nelem_1d {
            let sy = ey * (p - 1);
            let sx = ex * (p - 1);
            for jy in 0..p {
                for jx in 0..p {
                    offsets.push(((sy + jy) * ndofs_1d + (sx + jx)) as i32);
                }
            }
        }
    }
    offsets
}

fn build_offsets_3d(nelem_1d: usize, p: usize, ndofs_1d: usize) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(nelem_1d * nelem_1d * nelem_1d * p * p * p);
    for ez in 0..nelem_1d {
        for ey in 0..nelem_1d {
            for ex in 0..nelem_1d {
                let sz = ez * (p - 1);
                let sy = ey * (p - 1);
                let sx = ex * (p - 1);
                for jz in 0..p {
                    for jy in 0..p {
                        for jx in 0..p {
                            offsets.push(
                                (((sz + jz) * ndofs_1d + (sy + jy)) * ndofs_1d + (sx + jx)) as i32,
                            );
                        }
                    }
                }
            }
        }
    }
    offsets
}

fn build_coords_components(dim: usize, ndofs_1d: usize) -> Vec<Vec<f64>> {
    let ndofs = ndofs_1d.pow(dim as u32);
    let mut comps = (0..dim).map(|_| vec![0.0_f64; ndofs]).collect::<Vec<_>>();

    if dim == 1 {
        for i in 0..ndofs {
            comps[0][i] = -1.0 + 2.0 * i as f64 / (ndofs - 1) as f64;
        }
    } else if dim == 2 {
        for iy in 0..ndofs_1d {
            for ix in 0..ndofs_1d {
                let i = iy * ndofs_1d + ix;
                comps[0][i] = -1.0 + 2.0 * ix as f64 / (ndofs_1d - 1) as f64;
                comps[1][i] = -1.0 + 2.0 * iy as f64 / (ndofs_1d - 1) as f64;
            }
        }
    } else {
        for iz in 0..ndofs_1d {
            for iy in 0..ndofs_1d {
                for ix in 0..ndofs_1d {
                    let i = (iz * ndofs_1d + iy) * ndofs_1d + ix;
                    comps[0][i] = -1.0 + 2.0 * ix as f64 / (ndofs_1d - 1) as f64;
                    comps[1][i] = -1.0 + 2.0 * iy as f64 / (ndofs_1d - 1) as f64;
                    comps[2][i] = -1.0 + 2.0 * iz as f64 / (ndofs_1d - 1) as f64;
                }
            }
        }
    }

    comps
}

fn build_basis_data(len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| ((index % 19) as f64 - 9.0) * 0.125)
        .collect()
}

fn basis_apply_buffer_sizes(
    dim: usize,
    ncomp: usize,
    ne: usize,
    nd: usize,
    nq: usize,
    transpose: bool,
    eval_mode: EvalMode,
) -> (usize, usize) {
    match (eval_mode, transpose) {
        (EvalMode::Interp, false) => (ne * nd * ncomp, ne * nq * ncomp),
        (EvalMode::Interp, true) => (ne * nq * ncomp, ne * nd * ncomp),
        (EvalMode::Grad, false) => (ne * nd * ncomp, ne * nq * ncomp * dim),
        (EvalMode::Grad, true) => (ne * nq * ncomp * dim, ne * nd * ncomp),
        (EvalMode::Div, false) => (ne * nd * ncomp, ne * nq),
        (EvalMode::Div, true) => (ne * nq, ne * nd * ncomp),
        (EvalMode::Curl, false) if dim == 2 && ncomp == 2 => (ne * nd * ncomp, ne * nq),
        (EvalMode::Curl, true) if dim == 2 && ncomp == 2 => (ne * nq, ne * nd * ncomp),
        (EvalMode::Curl, false) if dim == 3 && ncomp == 3 => (ne * nd * ncomp, ne * nq * 3),
        (EvalMode::Curl, true) if dim == 3 && ncomp == 3 => (ne * nq * 3, ne * nd * ncomp),
        _ => panic!(
            "cpu_backend bench: unsupported (dim={dim}, ncomp={ncomp}, {eval_mode:?}, transpose={transpose})"
        ),
    }
}

fn build_basis_apply(
    dim: usize,
    ncomp: usize,
    p: usize,
    q: usize,
    num_elem: usize,
    transpose: bool,
    eval_mode: EvalMode,
) -> BasisApplyScenario {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let basis = reed
        .basis_tensor_h1_lagrange(dim, ncomp, p, q, QuadMode::Gauss)
        .unwrap();
    let ne = num_elem;
    let nd = basis.num_dof();
    let nq = basis.num_qpoints();
    let d = basis.dim();
    let (input_len, output_len) =
        basis_apply_buffer_sizes(d, ncomp, ne, nd, nq, transpose, eval_mode);

    BasisApplyScenario {
        basis,
        input: build_basis_data(input_len),
        output: vec![0.0; output_len],
        num_elem,
        transpose,
        eval_mode,
    }
}

fn build_simplex_basis_apply(
    topo: ElemTopology,
    poly: usize,
    ncomp: usize,
    q: usize,
    num_elem: usize,
    transpose: bool,
    eval_mode: EvalMode,
) -> BasisApplyScenario {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let basis = reed.basis_h1_simplex(topo, poly, ncomp, q).unwrap();
    let ne = num_elem;
    let nd = basis.num_dof();
    let nq = basis.num_qpoints();
    let d = basis.dim();
    let (input_len, output_len) =
        basis_apply_buffer_sizes(d, ncomp, ne, nd, nq, transpose, eval_mode);

    BasisApplyScenario {
        basis,
        input: build_basis_data(input_len),
        output: vec![0.0; output_len],
        num_elem,
        transpose,
        eval_mode,
    }
}

fn build_poisson_apply(dim: usize, nelem_1d: usize, p: usize, q: usize) -> ApplyScenario {
    let reed = leak_box(Reed::<f64>::init("/cpu/self").unwrap());
    let ndofs_1d = nelem_1d * (p - 1) + 1;
    let ndofs = ndofs_1d.pow(dim as u32);
    let nelem = nelem_1d.pow(dim as u32);
    let elemsize = p.pow(dim as u32);
    let qpts_per_elem = q.pow(dim as u32);

    let offsets = match dim {
        1 => build_offsets_1d(nelem_1d, p),
        2 => build_offsets_2d(nelem_1d, p, ndofs_1d),
        3 => build_offsets_3d(nelem_1d, p, ndofs_1d),
        _ => unreachable!(),
    };

    let comps = build_coords_components(dim, ndofs_1d);

    if dim == 1 {
        let node_coords = &comps[0];
        let r_u = leak_box(
            reed.elem_restriction(nelem, p, 1, 1, ndofs, &offsets)
                .unwrap(),
        );
        let r_q = leak_box(
            reed.strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])
                .unwrap(),
        );
        let b_u = leak_box(
            reed.basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
                .unwrap(),
        );
        let x_coord = leak_box(reed.vector_from_slice(node_coords).unwrap());
        let qdata = leak_box(reed.vector(nelem * q).unwrap());
        qdata.set_value(0.0).unwrap();
        leak_box(
            reed.operator_builder()
                .qfunction(reed.q_function_by_name("Poisson1DBuild").unwrap())
                .field("dx", Some(&**r_u), Some(&**b_u), FieldVector::Active)
                .field("weights", None, Some(&**b_u), FieldVector::None)
                .field("qdata", Some(&**r_q), None, FieldVector::Active)
                .build()
                .unwrap(),
        )
        .apply(&**x_coord, &mut **qdata)
        .unwrap();
        let input = leak_box(reed.vector_from_slice(node_coords).unwrap());
        let output = leak_box(reed.vector(ndofs).unwrap());
        output.set_value(0.0).unwrap();
        let operator = Box::new(
            reed.operator_builder()
                .qfunction(reed.q_function_by_name("Poisson1DApply").unwrap())
                .field("du", Some(&**r_u), Some(&**b_u), FieldVector::Active)
                .field("qdata", Some(&**r_q), None, FieldVector::Passive(&**qdata))
                .field("dv", Some(&**r_u), Some(&**b_u), FieldVector::Active)
                .build()
                .unwrap(),
        );
        return ApplyScenario {
            input: &**input,
            output: &mut **output,
            operator,
        };
    }

    let x_coords = comps.concat();
    let x = leak_box(reed.vector_from_slice(&x_coords).unwrap());
    let r_x = leak_box(
        reed.elem_restriction(nelem, elemsize, dim, ndofs, dim * ndofs, &offsets)
            .unwrap(),
    );
    let r_u = leak_box(
        reed.elem_restriction(nelem, elemsize, 1, 1, ndofs, &offsets)
            .unwrap(),
    );
    let b_x = leak_box(
        reed.basis_tensor_h1_lagrange(dim, dim, p, q, QuadMode::Gauss)
            .unwrap(),
    );
    let b_u = leak_box(
        reed.basis_tensor_h1_lagrange(dim, 1, p, q, QuadMode::Gauss)
            .unwrap(),
    );
    let qdata_comp = dim * dim;
    let r_q = leak_box(
        reed.strided_elem_restriction(
            nelem,
            qpts_per_elem,
            qdata_comp,
            nelem * qpts_per_elem * qdata_comp,
            [1, qpts_per_elem as i32, (qpts_per_elem * qdata_comp) as i32],
        )
        .unwrap(),
    );
    let qdata = leak_box(reed.vector(nelem * qpts_per_elem * qdata_comp).unwrap());
    qdata.set_value(0.0).unwrap();

    let qf_build = if dim == 2 {
        "Poisson2DBuild"
    } else {
        "Poisson3DBuild"
    };
    let op_build = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name(qf_build).unwrap())
        .field("dx", Some(&**r_x), Some(&**b_x), FieldVector::Active)
        .field("weights", None, Some(&**b_x), FieldVector::None)
        .field("qdata", Some(&**r_q), None, FieldVector::Active)
        .build()
        .unwrap();
    op_build.apply(&**x, &mut **qdata).unwrap();

    let mut u_vals = vec![0.0_f64; ndofs];
    for component in comps.iter().take(dim) {
        for (dst, src) in u_vals.iter_mut().zip(component.iter()) {
            *dst += *src;
        }
    }
    let input = leak_box(reed.vector_from_slice(&u_vals).unwrap());
    let output = leak_box(reed.vector(ndofs).unwrap());
    output.set_value(0.0).unwrap();
    let qf_apply = if dim == 2 {
        "Poisson2DApply"
    } else {
        "Poisson3DApply"
    };
    let operator = Box::new(
        reed.operator_builder()
            .qfunction(reed.q_function_by_name(qf_apply).unwrap())
            .field("du", Some(&**r_u), Some(&**b_u), FieldVector::Active)
            .field("qdata", Some(&**r_q), None, FieldVector::Passive(&**qdata))
            .field("dv", Some(&**r_u), Some(&**b_u), FieldVector::Active)
            .build()
            .unwrap(),
    );

    ApplyScenario {
        input: &**input,
        output: &mut **output,
        operator,
    }
}

fn build_combined_apply(dim: usize, nelem_1d: usize, p: usize, q: usize) -> ApplyScenario {
    let reed = leak_box(Reed::<f64>::init("/cpu/self").unwrap());
    let ndofs_1d = nelem_1d * (p - 1) + 1;
    let ndofs = ndofs_1d.pow(dim as u32);
    let nelem = nelem_1d.pow(dim as u32);
    let elemsize = p.pow(dim as u32);
    let qpts_per_elem = q.pow(dim as u32);
    let offsets = match dim {
        1 => build_offsets_1d(nelem_1d, p),
        2 => build_offsets_2d(nelem_1d, p, ndofs_1d),
        3 => build_offsets_3d(nelem_1d, p, ndofs_1d),
        _ => unreachable!(),
    };

    let r_u = leak_box(
        reed.elem_restriction(nelem, elemsize, 1, 1, ndofs, &offsets)
            .unwrap(),
    );
    let b_u = leak_box(
        reed.basis_tensor_h1_lagrange(dim, 1, p, q, QuadMode::Gauss)
            .unwrap(),
    );
    let comps = build_coords_components(dim, ndofs_1d);
    let x_coords = comps.concat();
    let x_coord = leak_box(reed.vector_from_slice(&x_coords).unwrap());
    let r_x = leak_box(
        reed.elem_restriction(nelem, elemsize, dim, ndofs, dim * ndofs, &offsets)
            .unwrap(),
    );
    let b_x = leak_box(
        reed.basis_tensor_h1_lagrange(dim, dim, p, q, QuadMode::Gauss)
            .unwrap(),
    );

    let r_q_mass = leak_box(
        reed.strided_elem_restriction(
            nelem,
            qpts_per_elem,
            1,
            nelem * qpts_per_elem,
            [1, qpts_per_elem as i32, qpts_per_elem as i32],
        )
        .unwrap(),
    );
    let qdata_mass = leak_box(reed.vector(nelem * qpts_per_elem).unwrap());
    qdata_mass.set_value(0.0).unwrap();
    let op_build_mass = reed
        .operator_builder()
        .qfunction(
            reed.q_function_by_name(match dim {
                1 => "Mass1DBuild",
                2 => "Mass2DBuild",
                _ => "Mass3DBuild",
            })
            .unwrap(),
        )
        .field("dx", Some(&**r_x), Some(&**b_x), FieldVector::Active)
        .field("weights", None, Some(&**b_x), FieldVector::None)
        .field("qdata", Some(&**r_q_mass), None, FieldVector::Active)
        .build()
        .unwrap();
    op_build_mass.apply(&**x_coord, &mut **qdata_mass).unwrap();

    let (r_q_poisson, qdata_poisson) = if dim == 1 {
        let r_q = leak_box(
            reed.strided_elem_restriction(
                nelem,
                qpts_per_elem,
                1,
                nelem * qpts_per_elem,
                [1, qpts_per_elem as i32, qpts_per_elem as i32],
            )
            .unwrap(),
        );
        let qdata_vec = leak_box(reed.vector(nelem * qpts_per_elem).unwrap());
        qdata_vec.set_value(0.0).unwrap();
        let op_poisson_build = reed
            .operator_builder()
            .qfunction(reed.q_function_by_name("Poisson1DBuild").unwrap())
            .field("dx", Some(&**r_x), Some(&**b_x), FieldVector::Active)
            .field("weights", None, Some(&**b_x), FieldVector::None)
            .field("qdata", Some(&**r_q), None, FieldVector::Active)
            .build()
            .unwrap();
        op_poisson_build
            .apply(&**x_coord, &mut **qdata_vec)
            .unwrap();
        (r_q, qdata_vec)
    } else {
        let qdata_comp = dim * dim;
        let r_q = leak_box(
            reed.strided_elem_restriction(
                nelem,
                qpts_per_elem,
                qdata_comp,
                nelem * qpts_per_elem * qdata_comp,
                [1, qpts_per_elem as i32, (qpts_per_elem * qdata_comp) as i32],
            )
            .unwrap(),
        );
        let qdata_vec = leak_box(reed.vector(nelem * qpts_per_elem * qdata_comp).unwrap());
        qdata_vec.set_value(0.0).unwrap();
        let op_build_poisson = reed
            .operator_builder()
            .qfunction(
                reed.q_function_by_name(if dim == 2 {
                    "Poisson2DBuild"
                } else {
                    "Poisson3DBuild"
                })
                .unwrap(),
            )
            .field("dx", Some(&**r_x), Some(&**b_x), FieldVector::Active)
            .field("weights", None, Some(&**b_x), FieldVector::None)
            .field("qdata", Some(&**r_q), None, FieldVector::Active)
            .build()
            .unwrap();
        op_build_poisson
            .apply(&**x_coord, &mut **qdata_vec)
            .unwrap();
        (r_q, qdata_vec)
    };

    let input = leak_box(reed.vector_from_slice(&vec![1.0_f64; ndofs]).unwrap());
    let output = leak_box(reed.vector(ndofs).unwrap());
    output.set_value(0.0).unwrap();
    let operator = Box::new(
        reed.operator_builder()
            .qfunction(
                reed.q_function_interior(
                    1,
                    vec![
                        reed::QFunctionField {
                            name: "u".into(),
                            num_comp: 1,
                            eval_mode: reed::EvalMode::Interp,
                        },
                        reed::QFunctionField {
                            name: "qdata_mass".into(),
                            num_comp: 1,
                            eval_mode: reed::EvalMode::None,
                        },
                        reed::QFunctionField {
                            name: "du".into(),
                            num_comp: dim,
                            eval_mode: reed::EvalMode::Grad,
                        },
                        reed::QFunctionField {
                            name: "qdata_poisson".into(),
                            num_comp: dim * dim,
                            eval_mode: reed::EvalMode::None,
                        },
                    ],
                    vec![
                        reed::QFunctionField {
                            name: "v".into(),
                            num_comp: 1,
                            eval_mode: reed::EvalMode::Interp,
                        },
                        reed::QFunctionField {
                            name: "dv".into(),
                            num_comp: dim,
                            eval_mode: reed::EvalMode::Grad,
                        },
                    ],
                    0,
                    Box::new(move |_ctx, q, inputs, outputs| {
                        let u = inputs[0];
                        let qdata_mass = inputs[1];
                        let du = inputs[2];
                        let qdata_poisson = inputs[3];
                        let (v_out, dv_out) = outputs.split_at_mut(1);
                        let v = &mut v_out[0];
                        let dv = &mut dv_out[0];
                        for i in 0..q {
                            v[i] = u[i] * qdata_mass[i];
                            match dim {
                                1 => {
                                    dv[i] = du[i] * qdata_poisson[i];
                                }
                                2 => {
                                    let du0 = du[i * 2];
                                    let du1 = du[i * 2 + 1];
                                    let g00 = qdata_poisson[i * 4];
                                    let g01 = qdata_poisson[i * 4 + 1];
                                    let g10 = qdata_poisson[i * 4 + 2];
                                    let g11 = qdata_poisson[i * 4 + 3];
                                    dv[i * 2] = g00 * du0 + g01 * du1;
                                    dv[i * 2 + 1] = g10 * du0 + g11 * du1;
                                }
                                3 => {
                                    let du0 = du[i * 3];
                                    let du1 = du[i * 3 + 1];
                                    let du2 = du[i * 3 + 2];
                                    let g00 = qdata_poisson[i * 9];
                                    let g01 = qdata_poisson[i * 9 + 1];
                                    let g02 = qdata_poisson[i * 9 + 2];
                                    let g10 = qdata_poisson[i * 9 + 3];
                                    let g11 = qdata_poisson[i * 9 + 4];
                                    let g12 = qdata_poisson[i * 9 + 5];
                                    let g20 = qdata_poisson[i * 9 + 6];
                                    let g21 = qdata_poisson[i * 9 + 7];
                                    let g22 = qdata_poisson[i * 9 + 8];
                                    dv[i * 3] = g00 * du0 + g01 * du1 + g02 * du2;
                                    dv[i * 3 + 1] = g10 * du0 + g11 * du1 + g12 * du2;
                                    dv[i * 3 + 2] = g20 * du0 + g21 * du1 + g22 * du2;
                                }
                                _ => unreachable!(),
                            }
                        }
                        Ok(())
                    }),
                )
                .unwrap(),
            )
            .field("u", Some(&**r_u), Some(&**b_u), FieldVector::Active)
            .field(
                "qdata_mass",
                Some(&**r_q_mass),
                None,
                FieldVector::Passive(&**qdata_mass),
            )
            .field("du", Some(&**r_u), Some(&**b_u), FieldVector::Active)
            .field(
                "qdata_poisson",
                Some(&**r_q_poisson),
                None,
                FieldVector::Passive(&**qdata_poisson),
            )
            .field("v", Some(&**r_u), Some(&**b_u), FieldVector::Active)
            .field("dv", Some(&**r_u), Some(&**b_u), FieldVector::Active)
            .build()
            .unwrap(),
    );

    ApplyScenario {
        input: &**input,
        output: &mut **output,
        operator,
    }
}

fn bench_poisson_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_poisson_apply");
    for &(dim, nelem, p, q) in &[(2, 32, 4, 6), (3, 16, 4, 6)] {
        let scenario = build_poisson_apply(dim, nelem, p, q);
        group.bench_with_input(
            BenchmarkId::new("apply", format!("dim{dim}_n{nelem}_p{p}_q{q}")),
            &(dim, nelem, p, q),
            |b, _| {
                b.iter(|| {
                    scenario.output.set_value(0.0).unwrap();
                    scenario
                        .operator
                        .apply(black_box(scenario.input), black_box(&mut *scenario.output))
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_combined_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_combined_apply");
    for &(dim, nelem, p, q) in &[(2, 24, 4, 6), (3, 16, 4, 6)] {
        let scenario = build_combined_apply(dim, nelem, p, q);
        group.bench_with_input(
            BenchmarkId::new("apply", format!("dim{dim}_n{nelem}_p{p}_q{q}")),
            &(dim, nelem, p, q),
            |b, _| {
                b.iter(|| {
                    scenario.output.set_value(0.0).unwrap();
                    scenario
                        .operator
                        .apply(black_box(scenario.input), black_box(&mut *scenario.output))
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_basis_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_basis_apply");
    for &(dim, num_elem, p, q) in &[(1, 8192, 8, 10), (2, 1024, 4, 6), (3, 256, 4, 6)] {
        for &(eval_mode, transpose, label) in &[
            (EvalMode::Interp, false, "interp"),
            (EvalMode::Interp, true, "interp_t"),
            (EvalMode::Grad, false, "grad"),
            (EvalMode::Grad, true, "grad_t"),
        ] {
            let mut scenario = build_basis_apply(dim, 1, p, q, num_elem, transpose, eval_mode);
            group.bench_with_input(
                BenchmarkId::new(label, format!("dim{dim}_elem{num_elem}_p{p}_q{q}")),
                &(dim, num_elem, p, q, transpose),
                |b, _| {
                    b.iter(|| {
                        scenario
                            .basis
                            .apply(
                                scenario.num_elem,
                                scenario.transpose,
                                scenario.eval_mode,
                                black_box(scenario.input.as_slice()),
                                black_box(scenario.output.as_mut_slice()),
                            )
                            .unwrap();
                    });
                },
            );
        }
    }
    for &(dim, num_elem, p, q) in &[(2, 1024, 4, 6), (3, 256, 4, 6)] {
        let ncomp = dim;
        for &(eval_mode, transpose, label) in &[
            (EvalMode::Div, false, "div"),
            (EvalMode::Div, true, "div_t"),
            (EvalMode::Curl, false, "curl"),
            (EvalMode::Curl, true, "curl_t"),
        ] {
            let mut scenario = build_basis_apply(dim, ncomp, p, q, num_elem, transpose, eval_mode);
            group.bench_with_input(
                BenchmarkId::new(label, format!("dim{dim}_elem{num_elem}_p{p}_q{q}")),
                &(dim, num_elem, p, q, transpose),
                |b, _| {
                    b.iter(|| {
                        scenario
                            .basis
                            .apply(
                                scenario.num_elem,
                                scenario.transpose,
                                scenario.eval_mode,
                                black_box(scenario.input.as_slice()),
                                black_box(scenario.output.as_mut_slice()),
                            )
                            .unwrap();
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_simplex_basis_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_simplex_basis_apply");
    // Triangle P2, q=6; Tet P2, q=5 (see basis_simplex docs)
    for &(topo, poly, q, num_elem) in &[
        (ElemTopology::Triangle, 2usize, 6usize, 2048usize),
        (ElemTopology::Tet, 2usize, 5usize, 512usize),
    ] {
        for &(eval_mode, transpose, label) in &[
            (EvalMode::Interp, false, "interp"),
            (EvalMode::Interp, true, "interp_t"),
            (EvalMode::Grad, false, "grad"),
            (EvalMode::Grad, true, "grad_t"),
        ] {
            let mut scenario =
                build_simplex_basis_apply(topo, poly, 1, q, num_elem, transpose, eval_mode);
            let topo_label = if topo == ElemTopology::Triangle {
                "tri"
            } else {
                "tet"
            };
            group.bench_with_input(
                BenchmarkId::new(label, format!("{topo_label}_p{poly}_q{q}_e{num_elem}")),
                &(poly, q, num_elem),
                |b, _| {
                    b.iter(|| {
                        scenario
                            .basis
                            .apply(
                                scenario.num_elem,
                                scenario.transpose,
                                scenario.eval_mode,
                                black_box(scenario.input.as_slice()),
                                black_box(scenario.output.as_mut_slice()),
                            )
                            .unwrap();
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_poisson_apply,
    bench_combined_apply,
    bench_basis_apply,
    bench_simplex_basis_apply
);
criterion_main!(benches);
