use reed::{EvalMode, FieldVector, OperatorTrait, QFunctionField, QuadMode, Reed, TransposeMode};

fn build_poisson_qdata(node_coords: &[f64], qweights: &[f64]) -> Vec<f64> {
    let q = qweights.len();
    let mut qdata = Vec::with_capacity((node_coords.len() - 1) * q);
    for elem in 0..(node_coords.len() - 1) {
        let jacobian = (node_coords[elem + 1] - node_coords[elem]) * 0.5;
        for &weight in qweights {
            qdata.push(weight / jacobian);
        }
    }
    qdata
}

#[test]
fn test_mass_1d_integral() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;

    let node_coords = vec![-1.0, 0.0, 1.0];
    let x_coord = reed.vector_from_slice(&node_coords).unwrap();
    let mut qdata = reed.vector(nelem * q).unwrap();
    qdata.set_value(0.0).unwrap();

    let ind_x = vec![0, 1, 1, 2];
    let ind_u = ind_x.clone();

    let r_x = reed
        .elem_restriction(nelem, 2, 1, 1, ndofs, &ind_x)
        .unwrap();
    let r_u = reed
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let r_q = reed
        .strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])
        .unwrap();

    let b_x = reed
        .basis_tensor_h1_lagrange(1, 1, 2, q, QuadMode::Gauss)
        .unwrap();
    let b_u = reed
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();

    let build = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Mass1DBuild").unwrap())
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()
        .unwrap();
    build.apply(&*x_coord, &mut *qdata).unwrap();

    let op_mass = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    let u = reed.vector_from_slice(&vec![1.0_f64; ndofs]).unwrap();
    let mut v = reed.vector(ndofs).unwrap();
    v.set_value(0.0).unwrap();
    op_mass.apply(&*u, &mut *v).unwrap();

    let mut values = vec![0.0; ndofs];
    v.copy_to_slice(&mut values).unwrap();
    let sum: f64 = values.iter().sum();
    assert!((sum - 2.0).abs() < 50.0 * f64::EPSILON);
}

#[test]
fn test_poisson_1d_apply() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;

    let node_coords = vec![-1.0, 0.0, 1.0];
    let b_u = reed
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();
    let qdata_values = build_poisson_qdata(&node_coords, b_u.q_weights());
    let qdata = reed.vector_from_slice(&qdata_values).unwrap();

    let ind_u = vec![0, 1, 1, 2];
    let r_u = reed
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let r_q = reed
        .strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])
        .unwrap();

    let op_poisson = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Poisson1DApply").unwrap())
        .field("du", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("dv", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    let u = reed.vector_from_slice(&[0.0_f64, 1.0, 0.0]).unwrap();
    let mut v = reed.vector(ndofs).unwrap();
    v.set_value(0.0).unwrap();
    op_poisson.apply(&*u, &mut *v).unwrap();

    let mut values = vec![0.0; ndofs];
    v.copy_to_slice(&mut values).unwrap();
    let expected = [-1.0_f64, 2.0, -1.0];
    for (actual, reference) in values.iter().zip(expected.iter()) {
        assert!((actual - reference).abs() < 100.0 * f64::EPSILON);
    }
}

#[test]
fn test_custom_closure_qfunction_apply() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;

    let node_coords = vec![-1.0, 0.0, 1.0];
    let x_coord = reed.vector_from_slice(&node_coords).unwrap();
    let mut qdata = reed.vector(nelem * q).unwrap();
    qdata.set_value(0.0).unwrap();

    let ind_x = vec![0, 1, 1, 2];
    let ind_u = ind_x.clone();

    let r_x = reed
        .elem_restriction(nelem, 2, 1, 1, ndofs, &ind_x)
        .unwrap();
    let r_u = reed
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let r_q = reed
        .strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])
        .unwrap();

    let b_x = reed
        .basis_tensor_h1_lagrange(1, 1, 2, q, QuadMode::Gauss)
        .unwrap();
    let b_u = reed
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();

    let build = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Mass1DBuild").unwrap())
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()
        .unwrap();
    build.apply(&*x_coord, &mut *qdata).unwrap();

    let custom_qf = reed
        .q_function_interior(
            1,
            vec![
                QFunctionField {
                    name: "u".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Interp,
                },
                QFunctionField {
                    name: "qdata".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::None,
                },
            ],
            vec![QFunctionField {
                name: "v".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
            Box::new(|q, inputs, outputs| {
                let u = inputs[0];
                let qdata = inputs[1];
                let v = &mut outputs[0];
                for i in 0..q {
                    v[i] = u[i] * qdata[i];
                }
                Ok(())
            }),
        )
        .unwrap();

    let op_mass = reed
        .operator_builder()
        .qfunction(custom_qf)
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    let u = reed.vector_from_slice(&vec![1.0_f64; ndofs]).unwrap();
    let mut v = reed.vector(ndofs).unwrap();
    v.set_value(0.0).unwrap();
    op_mass.apply(&*u, &mut *v).unwrap();

    let mut values = vec![0.0; ndofs];
    v.copy_to_slice(&mut values).unwrap();
    let sum: f64 = values.iter().sum();
    assert!((sum - 2.0).abs() < 50.0 * f64::EPSILON);
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_backend_init() {
    let reed = Reed::<f64>::init("/gpu/wgpu").unwrap();
    assert_eq!(reed.resource(), "/gpu/wgpu");
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_vector_basic_ops() {
    let reed = Reed::<f32>::init("/gpu/wgpu").unwrap();

    let mut y = reed.vector(4).unwrap();
    y.set_value(2.0).unwrap();
    y.scale(0.5).unwrap();

    let x = reed.vector_from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    y.axpy(2.0, &*x).unwrap();

    let mut out = [0.0_f32; 4];
    y.copy_to_slice(&mut out).unwrap();
    let expected = [3.0_f32, 5.0, 7.0, 9.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1.0e-5);
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_elem_restriction_no_transpose_offset_f32() {
    let reed = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let r = reed
        .elem_restriction(2, 2, 1, 1, 3, &[0, 1, 1, 2])
        .unwrap();

    let global = vec![10.0_f32, 20.0, 30.0];
    let mut local = vec![0.0_f32; 4];
    r.apply(TransposeMode::NoTranspose, &global, &mut local)
        .unwrap();
    assert_eq!(local, vec![10.0_f32, 20.0, 20.0, 30.0]);
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_elem_restriction_transpose_fallback() {
    let reed = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let r = reed
        .elem_restriction(2, 2, 1, 1, 3, &[0, 1, 1, 2])
        .unwrap();

    let local = vec![10.0_f32, 20.0, 20.0, 30.0];
    let mut gathered = vec![0.0_f32; 3];
    r.apply(TransposeMode::Transpose, &local, &mut gathered)
        .unwrap();
    assert_eq!(gathered, vec![10.0_f32, 40.0, 30.0]);
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_interp_matches_cpu() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();

    let b_cpu = reed_cpu
        .basis_tensor_h1_lagrange(1, 1, 3, 4, QuadMode::Gauss)
        .unwrap();
    let b_gpu = reed_gpu
        .basis_tensor_h1_lagrange(1, 1, 3, 4, QuadMode::Gauss)
        .unwrap();

    let num_elem = 2usize;
    let u = vec![0.0_f32, 1.0, 2.0, 1.5, -0.5, 0.25];
    let mut v_cpu = vec![0.0_f32; num_elem * b_cpu.num_qpoints() * b_cpu.num_comp()];
    let mut v_gpu = vec![0.0_f32; num_elem * b_gpu.num_qpoints() * b_gpu.num_comp()];

    b_cpu
        .apply(num_elem, false, EvalMode::Interp, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(num_elem, false, EvalMode::Interp, &u, &mut v_gpu)
        .unwrap();

    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 1.0e-5);
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_interp_transpose_matches_cpu() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();

    let b_cpu = reed_cpu
        .basis_tensor_h1_lagrange(1, 1, 3, 4, QuadMode::Gauss)
        .unwrap();
    let b_gpu = reed_gpu
        .basis_tensor_h1_lagrange(1, 1, 3, 4, QuadMode::Gauss)
        .unwrap();

    let num_elem = 2usize;
    let u = vec![0.5_f32, -0.25, 1.0, 2.0, -1.0, 0.75, 0.25, -0.5];
    let mut v_cpu = vec![0.0_f32; num_elem * b_cpu.num_dof() * b_cpu.num_comp()];
    let mut v_gpu = vec![0.0_f32; num_elem * b_gpu.num_dof() * b_gpu.num_comp()];

    b_cpu
        .apply(num_elem, true, EvalMode::Interp, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(num_elem, true, EvalMode::Interp, &u, &mut v_gpu)
        .unwrap();

    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 1.0e-5);
    }
}
