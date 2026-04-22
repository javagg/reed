use reed::{
    CeedMatrix, CeedMatrixStorage, CompositeOperator, CpuOperator, CsrMatrix, ElemRestrictionTrait,
    ElemTopology, EvalMode, FieldVector, OperatorAssembleKind, OperatorTrait, OperatorTransposeRequest, QFunctionCategory,
    QFunctionContext, QFunctionField, QuadMode, QFUNCTION_INTERIOR_GALLERY_NAMES,
    QFUNCTION_LIBCEED_MAIN_GALLERY_NAMES, Reed, ReedError, ReedResult, TransposeMode, VectorTrait,
};

#[test]
fn test_qfunction_interior_gallery_names_resolve_via_reed() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    for &name in QFUNCTION_INTERIOR_GALLERY_NAMES {
        reed.q_function_by_name(name)
            .unwrap_or_else(|e| panic!("gallery name {name:?} should resolve: {e:?}"));
    }
}

#[test]
fn test_qfunction_libceed_main_gallery_names_resolve_via_reed() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    for &name in QFUNCTION_LIBCEED_MAIN_GALLERY_NAMES {
        reed.q_function_by_name(name)
            .unwrap_or_else(|e| panic!("libCEED main gallery {name:?} should resolve: {e:?}"));
    }
    reed.q_function_by_name("IdentityScalar")
        .expect("IdentityScalar alias (registration-style name)");
    reed.q_function_by_name("ScaleScalar")
        .expect("ScaleScalar alias (registration-style name)");
}

#[test]
fn test_qfunction_exterior_closure_reports_exterior_category() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let qf = reed
        .q_function_exterior(
            1,
            vec![QFunctionField {
                name: "u".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
            vec![QFunctionField {
                name: "v".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
            0,
            Box::new(
                |_ctx: &[u8],
                 q: usize,
                 inputs: &[&[f64]],
                 outputs: &mut [&mut [f64]]|
                 -> ReedResult<()> {
                    for i in 0..q {
                        outputs[0][i] = inputs[0][i];
                    }
                    Ok(())
                },
            ),
        )
        .unwrap();
    assert_eq!(qf.q_function_category(), QFunctionCategory::Exterior);
}

#[test]
fn test_cpu_operator_libceed_dense_linear_assemble_and_fdm_stub() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 1usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = 2usize;
    let ind_u = vec![0i32, 1];
    let r_u = reed
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let r_q = reed
        .strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])
        .unwrap();
    let b_u = reed
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();
    let mut qdata = reed.vector(nelem * q).unwrap();
    qdata.set_value(1.0).unwrap();
    let op: CpuOperator<'_, f64> = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    assert!(OperatorTrait::operator_supports_assemble(
        &op,
        OperatorAssembleKind::Diagonal
    ));
    assert!(OperatorTrait::operator_supports_assemble(
        &op,
        OperatorAssembleKind::LinearSymbolic
    ));
    assert!(OperatorTrait::operator_supports_assemble(
        &op,
        OperatorAssembleKind::LinearNumeric
    ));
    assert!(OperatorTrait::operator_supports_assemble(
        &op,
        OperatorAssembleKind::LinearCsrNumeric
    ));
    assert!(OperatorTrait::operator_supports_assemble(
        &op,
        OperatorAssembleKind::FdmElementInverse
    ));

    OperatorTrait::linear_assemble_symbolic(&op).unwrap();
    assert_eq!(op.dense_linear_assembly_n(), Some(2));
    assert!(!op.dense_linear_assembly_numeric_ready());
    OperatorTrait::linear_assemble(&op).unwrap();
    assert_eq!(op.dense_linear_assembly_n(), Some(2));
    assert!(op.dense_linear_assembly_numeric_ready());
    let (n, a) = op.assembled_linear_matrix_col_major().expect("dense matrix after assemble");
    assert_eq!(n, 2);
    assert_eq!(a.len(), 4);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (a[i + j * n] - a[j + i * n]).abs() < 1e-12,
                "symmetry failed at ({i},{j})"
            );
        }
    }
    let mut diag = reed.vector(n).unwrap();
    diag.set_value(0.0).unwrap();
    OperatorTrait::linear_assemble_diagonal(&op, &mut *diag).unwrap();
    for i in 0..n {
        assert!(
            (diag.as_slice()[i] - a[i + i * n]).abs() < 1e-11,
            "diagonal vs dense diag at {i}"
        );
    }

    let mut acc = reed.vector(n).unwrap();
    acc.set_value(1.0).unwrap();
    OperatorTrait::linear_assemble_add_diagonal(&op, &mut *acc).unwrap();
    for i in 0..n {
        assert!(
            (acc.as_slice()[i] - 1.0 - diag.as_slice()[i]).abs() < 1e-11,
            "add_diagonal should add true diag to prior contents at {i}"
        );
    }

    let inv = OperatorTrait::operator_create_fdm_element_inverse(&op).unwrap();
    let x = reed.vector_from_slice(&[1.0_f64, -0.5]).unwrap();
    let mut ax = reed.vector(2).unwrap();
    ax.set_value(0.0).unwrap();
    OperatorTrait::apply(&op, &*x, &mut *ax).unwrap();
    let mut recovered = reed.vector(2).unwrap();
    recovered.set_value(0.0).unwrap();
    OperatorTrait::apply(inv.as_ref(), &*ax, &mut *recovered).unwrap();
    for i in 0..2 {
        assert!(
            (recovered.as_slice()[i] - x.as_slice()[i]).abs() < 1e-10,
            "FDM inverse roundtrip i={i}"
        );
    }
    let mut recovered_t = reed.vector(2).unwrap();
    recovered_t.set_value(0.0).unwrap();
    OperatorTrait::apply_with_transpose(
        inv.as_ref(),
        OperatorTransposeRequest::Adjoint,
        &*ax,
        &mut *recovered_t,
    )
    .unwrap();
    for i in 0..2 {
        assert!(
            (recovered_t.as_slice()[i] - x.as_slice()[i]).abs() < 1e-10,
            "symmetric mass: inv adjoint roundtrip i={i}"
        );
    }

    // libCEED `LinearAssembleAdd`: reset dense to `A`, then add columns again -> `2A` in the slot.
    OperatorTrait::linear_assemble(&op).unwrap();
    OperatorTrait::linear_assemble_add(&op).unwrap();
    let (_n2, a_twice) = op.assembled_linear_matrix_col_major().unwrap();
    for i in 0..n {
        for j in 0..n {
            assert!(
                (a_twice[i + j * n] - 2.0 * a[i + j * n]).abs() < 1e-11,
                "linear_assemble_add should double columns at ({i},{j})"
            );
        }
    }
    let inv_after_add = OperatorTrait::operator_create_fdm_element_inverse(&op).unwrap();
    let mut recovered_after_add = reed.vector(2).unwrap();
    recovered_after_add.set_value(0.0).unwrap();
    OperatorTrait::apply(inv_after_add.as_ref(), &*ax, &mut *recovered_after_add).unwrap();
    for i in 0..2 {
        assert!(
            (recovered_after_add.as_slice()[i] - x.as_slice()[i]).abs() < 1e-10,
            "FDM inverse should use canonical A even after linear_assemble_add i={i}"
        );
    }
    let (_n_after_add, dense_after_add) = op.assembled_linear_matrix_col_major().unwrap();
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dense_after_add[i + j * n] - a_twice[i + j * n]).abs() < 1e-11,
                "FDM inverse creation should not mutate dense slot at ({i},{j})"
            );
        }
    }

    op.clear_dense_linear_assembly().unwrap();
    assert_eq!(op.dense_linear_assembly_n(), None);
    assert!(!op.dense_linear_assembly_numeric_ready());
    assert!(
        op.assembled_linear_matrix_col_major().is_none(),
        "assembled matrix should be gone after clear_dense_linear_assembly"
    );
    assert!(OperatorTrait::linear_assemble(&op).is_err());
    OperatorTrait::linear_assemble_symbolic(&op).unwrap();
    OperatorTrait::linear_assemble(&op).unwrap();
    let (_, a_after_clear) = op.assembled_linear_matrix_col_major().unwrap();
    for i in 0..n {
        for j in 0..n {
            assert!(
                (a_after_clear[i + j * n] - a[i + j * n]).abs() < 1e-11,
                "reassemble after clear should match original A at ({i},{j})"
            );
        }
    }

    let op2: CpuOperator<'_, f64> = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();
    assert!(OperatorTrait::linear_assemble(&op2).is_err());
}

#[test]
fn test_cpu_operator_ceed_matrix_handle_and_jacobi_inverse_paths() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 1usize;
    let p = 2usize;
    let q = 1usize;
    let ndofs = 2usize;
    let ind_u = vec![0i32, 1];
    let r_u = reed
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let b_u = reed
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();
    let op: CpuOperator<'_, f64> = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Identity").unwrap())
        .field("input", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("output", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    let mut dense = CeedMatrix::<f64>::dense_col_major_symbolic(ndofs, ndofs).unwrap();
    op.linear_assemble_ceed_matrix(&mut dense).unwrap();
    let dense_once = match dense.storage() {
        CeedMatrixStorage::DenseColMajor { values, .. } => values.clone(),
        _ => panic!("expected dense handle"),
    };
    op.linear_assemble_add_ceed_matrix(&mut dense).unwrap();
    match dense.storage() {
        CeedMatrixStorage::DenseColMajor { values, .. } => {
            for i in 0..values.len() {
                assert!((values[i] - 2.0 * dense_once[i]).abs() < 1e-12);
            }
        }
        _ => panic!("expected dense handle"),
    }

    let pat = r_u.assembled_csr_pattern().unwrap();
    let mut csr = CeedMatrix::<f64>::csr_symbolic(pat);
    op.linear_assemble_ceed_matrix(&mut csr).unwrap();
    let csr_once = match csr.storage() {
        CeedMatrixStorage::Csr(m) => m.values.clone(),
        _ => panic!("expected csr handle"),
    };
    op.linear_assemble_add_ceed_matrix(&mut csr).unwrap();
    match csr.storage() {
        CeedMatrixStorage::Csr(m) => {
            for (i, &v) in m.values.iter().enumerate() {
                assert!((v - 2.0 * csr_once[i]).abs() < 1e-12);
            }
        }
        _ => panic!("expected csr handle"),
    }

    let jac = op.operator_create_fdm_element_inverse_jacobi().unwrap();
    let x = reed.vector_from_slice(&[2.0_f64, -3.0]).unwrap();
    let mut y = reed.vector(ndofs).unwrap();
    y.set_value(0.0).unwrap();
    jac.apply(&*x, &mut *y).unwrap();
    let mut diag = reed.vector(ndofs).unwrap();
    diag.set_value(0.0).unwrap();
    op.linear_assemble_diagonal(&mut *diag).unwrap();
    for i in 0..ndofs {
        assert!(
            (y.as_slice()[i] - x.as_slice()[i] / diag.as_slice()[i]).abs() < 1e-12,
            "jacobi inverse action mismatch at {i}"
        );
    }
}

#[test]
fn test_mass_csr_assembly_matches_dense_columns() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;
    let ind_u = vec![0i32, 1, 1, 2];
    let r_u = reed
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let r_q = reed
        .strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])
        .unwrap();
    let b_u = reed
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();
    let mut qdata = reed.vector(nelem * q).unwrap();
    qdata.set_value(1.0).unwrap();

    let op: CpuOperator<'_, f64> = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    assert!(OperatorTrait::operator_supports_assemble(
        &op,
        OperatorAssembleKind::LinearCsrNumeric
    ));
    let pat = ElemRestrictionTrait::assembled_csr_pattern(r_u.as_ref()).unwrap();
    let csr = OperatorTrait::linear_assemble_csr_matrix(&op, &pat).unwrap();

    let mut csr_acc = CsrMatrix {
        pattern: pat.clone(),
        values: vec![0.0_f64; pat.nnz()],
    };
    OperatorTrait::linear_assemble_csr_matrix_add(&op, &mut csr_acc).unwrap();
    for k in 0..pat.nnz() {
        assert!(
            (csr_acc.values[k] - csr.values[k]).abs() < 1e-11,
            "CSR add from zero should match linear_assemble at k={k}"
        );
    }
    OperatorTrait::linear_assemble_csr_matrix_add(&op, &mut csr_acc).unwrap();
    for k in 0..pat.nnz() {
        assert!(
            (csr_acc.values[k] - 2.0 * csr.values[k]).abs() < 1e-11,
            "second CSR add should double at k={k}"
        );
    }

    OperatorTrait::linear_assemble_symbolic(&op).unwrap();
    OperatorTrait::linear_assemble(&op).unwrap();
    let (n, dense) = op.assembled_linear_matrix_col_major().unwrap();
    assert_eq!(n, ndofs);
    for row in 0..n {
        let r0 = csr.pattern.row_ptr[row];
        let r1 = csr.pattern.row_ptr[row + 1];
        for k in r0..r1 {
            let col = csr.pattern.col_ind[k];
            let c = csr.values[k];
            let d = dense[row + col * n];
            assert!(
                (c - d).abs() < 1e-11,
                "CSR vs dense mismatch at ({row},{col}): csr={c} dense={d}"
            );
        }
    }

    let x = reed.vector_from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
    let mut y_apply = reed.vector(ndofs).unwrap();
    y_apply.set_value(0.0).unwrap();
    OperatorTrait::apply(&op, &*x, &mut *y_apply).unwrap();
    let mut y_csr = vec![0.0_f64; ndofs];
    csr.mul_vec(x.as_slice(), &mut y_csr).unwrap();
    for i in 0..ndofs {
        assert!(
            (y_apply.as_slice()[i] - y_csr[i]).abs() < 1e-10,
            "CSR matvec vs apply at {i}"
        );
    }
}

#[test]
fn test_poisson_1d_csr_assembly_matches_dense_columns() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;

    let node_coords = vec![-1.0, 0.0, 1.0];
    let x_coord = reed.vector_from_slice(&node_coords).unwrap();
    let mut qdata = reed.vector(nelem * q).unwrap();
    qdata.set_value(0.0).unwrap();

    let ind_x = vec![0i32, 1, 1, 2];
    let ind_u = vec![0i32, 1, 1, 2];

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

    let build_poisson = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Poisson1DBuild").unwrap())
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()
        .unwrap();
    build_poisson.apply(&*x_coord, &mut *qdata).unwrap();

    let op: CpuOperator<'_, f64> = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Poisson1DApply").unwrap())
        .field("du", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("dv", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    assert!(OperatorTrait::operator_supports_assemble(
        &op,
        OperatorAssembleKind::LinearCsrNumeric
    ));
    let csr = op.linear_assemble_csr_from_elem_restriction(r_u.as_ref()).unwrap();

    let pat = csr.pattern.clone();
    let mut csr_acc = CsrMatrix {
        pattern: pat.clone(),
        values: vec![0.0_f64; pat.nnz()],
    };
    OperatorTrait::linear_assemble_csr_matrix_add(&op, &mut csr_acc).unwrap();
    for k in 0..pat.nnz() {
        assert!(
            (csr_acc.values[k] - csr.values[k]).abs() < 1e-9,
            "Poisson CSR add from zero at k={k}"
        );
    }
    OperatorTrait::linear_assemble_csr_matrix_add(&op, &mut csr_acc).unwrap();
    for k in 0..pat.nnz() {
        assert!(
            (csr_acc.values[k] - 2.0 * csr.values[k]).abs() < 1e-9,
            "Poisson second CSR add at k={k}"
        );
    }

    OperatorTrait::linear_assemble_symbolic(&op).unwrap();
    OperatorTrait::linear_assemble(&op).unwrap();
    let (n, dense) = op.assembled_linear_matrix_col_major().unwrap();
    assert_eq!(n, ndofs);
    for row in 0..n {
        let r0 = csr.pattern.row_ptr[row];
        let r1 = csr.pattern.row_ptr[row + 1];
        for k in r0..r1 {
            let col = csr.pattern.col_ind[k];
            let c = csr.values[k];
            let d = dense[row + col * n];
            assert!(
                (c - d).abs() < 1e-9,
                "Poisson CSR vs dense mismatch at ({row},{col}): csr={c} dense={d}"
            );
        }
    }

    let x = reed.vector_from_slice(&[1.0_f64, -0.5, 2.0]).unwrap();
    let mut y_apply = reed.vector(ndofs).unwrap();
    y_apply.set_value(0.0).unwrap();
    OperatorTrait::apply(&op, &*x, &mut *y_apply).unwrap();
    let mut y_csr = vec![0.0_f64; ndofs];
    csr.mul_vec(x.as_slice(), &mut y_csr).unwrap();
    for i in 0..ndofs {
        assert!(
            (y_apply.as_slice()[i] - y_csr[i]).abs() < 1e-9,
            "Poisson CSR matvec vs apply at {i}"
        );
    }
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
fn test_cpu_operator_check_ready_and_apply_io_length() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;
    let ind_u = vec![0, 1, 1, 2];
    let r_u = reed
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let r_q = reed
        .strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])
        .unwrap();
    let b_u = reed
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();
    let mut qdata = reed.vector(nelem * q).unwrap();
    qdata.set_value(1.0).unwrap();

    let op = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    op.check_ready().unwrap();

    let bad_in = reed.vector_from_slice(&[1.0_f64, 2.0]).unwrap();
    let mut out = reed.vector(ndofs).unwrap();
    out.set_value(0.0).unwrap();
    let err = op.apply(&*bad_in, &mut *out).unwrap_err();
    assert!(
        matches!(err, ReedError::Operator(_)),
        "expected Operator error, got {err:?}"
    );
}

#[test]
fn test_cpu_operator_label_transpose_forward_and_symmetric_mass_adjoint() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;
    let ind_u = vec![0, 1, 1, 2];
    let r_u = reed
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let r_q = reed
        .strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])
        .unwrap();
    let b_u = reed
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();
    let mut qdata = reed.vector(nelem * q).unwrap();
    qdata.set_value(1.0).unwrap();

    let op = reed
        .operator_builder()
        .operator_label("mass L2 smoke")
        .qfunction(reed.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    assert_eq!(OperatorTrait::operator_label(&op), Some("mass L2 smoke"));

    let u = reed.vector_from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
    let mut y_fwd = reed.vector(ndofs).unwrap();
    let mut y_path = reed.vector(ndofs).unwrap();
    y_fwd.set_value(0.0).unwrap();
    y_path.set_value(0.0).unwrap();
    op.apply(&*u, &mut *y_fwd).unwrap();
    op.apply_with_transpose(OperatorTransposeRequest::Forward, &*u, &mut *y_path)
        .unwrap();
    for i in 0..ndofs {
        assert!(
            (y_fwd.as_slice()[i] - y_path.as_slice()[i]).abs() < 50.0 * f64::EPSILON,
            "dof {i}"
        );
    }

    let w = reed.vector_from_slice(&[0.5_f64, 1.0, 1.5]).unwrap();
    let mut y_adj = reed.vector(ndofs).unwrap();
    let mut y_sym = reed.vector(ndofs).unwrap();
    y_adj.set_value(0.0).unwrap();
    y_sym.set_value(0.0).unwrap();
    op.apply_with_transpose(OperatorTransposeRequest::Adjoint, &*w, &mut *y_adj)
        .unwrap();
    op.apply(&*w, &mut *y_sym).unwrap();
    for i in 0..ndofs {
        assert!(
            (y_adj.as_slice()[i] - y_sym.as_slice()[i]).abs() < 50.0 * f64::EPSILON,
            "dof {i}: adjoint vs forward symmetric mass"
        );
    }
}

/// Discrete adjoint with a **passive** second input declared as [`EvalMode::Weight`] (same qp kernel as `MassApply`).
#[test]
fn test_cpu_operator_mass_apply_interp_times_weight_adjoint_matches_forward() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;
    let ind_u = vec![0, 1, 1, 2];
    let r_u = reed
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let b_u = reed
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();
    let passive_dummy = reed.vector_from_slice(&[0.0_f64]).unwrap();

    let op = reed
        .operator_builder()
        .qfunction(
            reed.q_function_by_name("MassApplyInterpTimesWeight")
                .unwrap(),
        )
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("w", None, Some(&*b_u), FieldVector::Passive(&*passive_dummy))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    let u = reed.vector_from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
    let mut y_fwd = reed.vector(ndofs).unwrap();
    let mut y_path = reed.vector(ndofs).unwrap();
    y_fwd.set_value(0.0).unwrap();
    y_path.set_value(0.0).unwrap();
    op.apply(&*u, &mut *y_fwd).unwrap();
    op.apply_with_transpose(OperatorTransposeRequest::Forward, &*u, &mut *y_path)
        .unwrap();
    for i in 0..ndofs {
        assert!(
            (y_fwd.as_slice()[i] - y_path.as_slice()[i]).abs() < 50.0 * f64::EPSILON,
            "dof {i}"
        );
    }

    let w = reed.vector_from_slice(&[0.5_f64, 1.0, 1.5]).unwrap();
    let mut y_adj = reed.vector(ndofs).unwrap();
    let mut y_sym = reed.vector(ndofs).unwrap();
    y_adj.set_value(0.0).unwrap();
    y_sym.set_value(0.0).unwrap();
    op.apply_with_transpose(OperatorTransposeRequest::Adjoint, &*w, &mut *y_adj)
        .unwrap();
    op.apply(&*w, &mut *y_sym).unwrap();
    for i in 0..ndofs {
        assert!(
            (y_adj.as_slice()[i] - y_sym.as_slice()[i]).abs() < 50.0 * f64::EPSILON,
            "dof {i}: adjoint vs forward (interp × qp-weight slot)"
        );
    }
}

/// Named-buffer `apply_field_buffers_with_transpose(Adjoint)` with a passive **`EvalMode::Weight`** input
/// (regression: domain scatter must skip non-active input fields).
#[test]
fn test_cpu_operator_mass_apply_interp_times_weight_named_buffers_adjoint_inner_product() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;
    let ind_u = vec![0, 1, 1, 2];
    let r_u = reed
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let b_u = reed
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();
    let passive_dummy = reed.vector_from_slice(&[0.0_f64]).unwrap();

    let op = reed
        .operator_builder()
        .qfunction(
            reed.q_function_by_name("MassApplyInterpTimesWeight")
                .unwrap(),
        )
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("w", None, Some(&*b_u), FieldVector::Passive(&*passive_dummy))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    let u = reed.vector_from_slice(&[1.0_f64, 0.5, -0.25]).unwrap();
    let w = reed.vector_from_slice(&[0.3, -0.7, 0.05]).unwrap();
    let mut v = reed.vector(ndofs).unwrap();
    v.set_value(0.0).unwrap();
    let ins = [("u", &*u as &dyn VectorTrait<f64>)];
    let mut outs = [("v", &mut *v as &mut dyn VectorTrait<f64>)];
    op.apply_field_buffers(&ins, &mut outs).unwrap();

    let mut du = reed.vector(ndofs).unwrap();
    du.set_value(0.0).unwrap();
    let range_in = [("v", &*w as &dyn VectorTrait<f64>)];
    let mut domain_out = [("u", &mut *du as &mut dyn VectorTrait<f64>)];
    op.apply_field_buffers_with_transpose(
        OperatorTransposeRequest::Adjoint,
        &range_in,
        &mut domain_out,
    )
    .unwrap();

    let lhs: f64 = v
        .as_slice()
        .iter()
        .zip(w.as_slice().iter())
        .map(|(a, b)| a * b)
        .sum();
    let rhs: f64 = u
        .as_slice()
        .iter()
        .zip(du.as_slice().iter())
        .map(|(a, b)| a * b)
        .sum();
    assert!(
        (lhs - rhs).abs() < 1e-9_f64.max(1e-9 * lhs.abs()),
        "named-buffer adjoint inner product: lhs={lhs} rhs={rhs}"
    );
}

/// Two active input fields (`u`, `aux`) need per-field global buffers: [`OperatorTrait::apply`] errors;
/// `apply_field_buffers` on the assembled CPU operator succeeds (libCEED-style multi-vector apply).
#[test]
fn test_cpu_operator_two_active_inputs_require_field_buffers() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;
    let ind_u = vec![0, 1, 1, 2];
    let r_u = reed
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let r_q = reed
        .strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])
        .unwrap();
    let b_u = reed
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();
    let mut qdata = reed.vector(nelem * q).unwrap();
    qdata.set_value(1.0).unwrap();

    let qf = reed
        .q_function_interior(
            1,
            vec![
                QFunctionField {
                    name: "u".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Interp,
                },
                QFunctionField {
                    name: "aux".into(),
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
            0,
            Box::new(|_ctx, q, inputs, outputs| {
                let u = inputs[0];
                let aux = inputs[1];
                let v = &mut outputs[0];
                for i in 0..q {
                    v[i] = u[i] + aux[i];
                }
                Ok(())
            }),
        )
        .unwrap();

    let op = reed
        .operator_builder()
        .qfunction(qf)
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("aux", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    assert!(op.requires_field_named_buffers());
    assert!(op.active_global_dof_len().is_err());
    assert!(!OperatorTrait::operator_supports_assemble(
        &op,
        OperatorAssembleKind::Diagonal
    ));

    let u = reed.vector_from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
    let aux = reed.vector_from_slice(&[10.0_f64, 20.0, 30.0]).unwrap();
    let mut v = reed.vector(ndofs).unwrap();
    v.set_value(0.0).unwrap();

    let err = op.apply(&*u, &mut *v).unwrap_err();
    assert!(matches!(err, ReedError::Operator(_)));
    let msg = err.to_string();
    assert!(
        msg.contains("apply_field_buffers"),
        "expected hint to apply_field_buffers, got {msg:?}"
    );

    let mut out = reed.vector(ndofs).unwrap();
    out.set_value(0.0).unwrap();
    let inputs = [
        ("u", &*u as &dyn VectorTrait<f64>),
        ("aux", &*aux as &dyn VectorTrait<f64>),
    ];
    let mut outputs = [("v", &mut *out as &mut dyn VectorTrait<f64>)];
    op.apply_field_buffers(&inputs, &mut outputs).unwrap();

    let mut buf = vec![0.0_f64; ndofs];
    out.copy_to_slice(&mut buf).unwrap();
    assert!(buf.iter().all(|x| x.is_finite()));
    assert!(buf.iter().any(|x| *x > 1e-12));
}

/// Two active outputs that receive the same quadrature values must assemble identical global vectors.
#[test]
fn test_cpu_operator_two_active_outputs_apply_field_buffers_agree() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;
    let ind_u = vec![0, 1, 1, 2];
    let r_u = reed
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let b_u = reed
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();

    let qf = reed
        .q_function_interior(
            1,
            vec![QFunctionField {
                name: "u".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
            vec![
                QFunctionField {
                    name: "out_a".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Interp,
                },
                QFunctionField {
                    name: "out_b".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Interp,
                },
            ],
            0,
            Box::new(|_ctx, nq, inputs, outputs| {
                let u = inputs[0];
                for i in 0..nq {
                    let v = u[i];
                    outputs[0][i] = v;
                    outputs[1][i] = v;
                }
                Ok(())
            }),
        )
        .unwrap();

    let op = reed
        .operator_builder()
        .qfunction(qf)
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("out_a", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("out_b", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    assert!(op.requires_field_named_buffers());

    let u = reed.vector_from_slice(&[1.0_f64, 0.5, -0.25]).unwrap();
    let mut out_a = reed.vector(ndofs).unwrap();
    let mut out_b = reed.vector(ndofs).unwrap();
    out_a.set_value(0.0).unwrap();
    out_b.set_value(0.0).unwrap();

    let inputs = [("u", &*u as &dyn VectorTrait<f64>)];
    let mut outputs = [
        ("out_a", &mut *out_a as &mut dyn VectorTrait<f64>),
        ("out_b", &mut *out_b as &mut dyn VectorTrait<f64>),
    ];
    op.apply_field_buffers(&inputs, &mut outputs).unwrap();

    let mut va = vec![0.0_f64; ndofs];
    let mut vb = vec![0.0_f64; ndofs];
    out_a.copy_to_slice(&mut va).unwrap();
    out_b.copy_to_slice(&mut vb).unwrap();
    for i in 0..ndofs {
        assert!(
            (va[i] - vb[i]).abs() < 1e-10,
            "dof {}: out_a={} out_b={}",
            i,
            va[i],
            vb[i]
        );
    }
}

/// Two identical `MassApply` `CpuOperator`s composed with `composite_operator_refs` equals `2 *` one apply.
#[test]
fn test_composite_operator_refs_two_mass_matches_double_single() {
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

    let build_mass = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Mass1DBuild").unwrap())
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()
        .unwrap();
    build_mass.apply(&*x_coord, &mut *qdata).unwrap();

    let op_a = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();
    let op_b = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    let composite = reed.composite_operator_refs(&[&op_a, &op_b]).unwrap();
    assert!(OperatorTrait::operator_supports_assemble(
        &composite,
        OperatorAssembleKind::Diagonal
    ));
    assert!(!OperatorTrait::operator_supports_assemble(
        &composite,
        OperatorAssembleKind::LinearCsrNumeric
    ));
    assert!(!OperatorTrait::operator_supports_assemble(
        &composite,
        OperatorAssembleKind::FdmElementInverse
    ));
    let fdm_err = match OperatorTrait::operator_create_fdm_element_inverse(&composite) {
        Err(e) => e,
        Ok(_) => panic!("expected FDM inverse on composite to fail"),
    };
    assert!(
        fdm_err.to_string().contains("CompositeOperatorBorrowed"),
        "{fdm_err:?}"
    );

    let u = reed.vector_from_slice(&vec![1.0_f64; ndofs]).unwrap();
    let mut y_composite = reed.vector(ndofs).unwrap();
    y_composite.set_value(0.0).unwrap();
    composite.apply(&*u, &mut *y_composite).unwrap();

    let op_once = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();
    let mut y_single = reed.vector(ndofs).unwrap();
    y_single.set_value(0.0).unwrap();
    op_once.apply(&*u, &mut *y_single).unwrap();
    for x in y_single.as_mut_slice() {
        *x *= 2.0;
    }

    for (a, b) in y_composite
        .as_slice()
        .iter()
        .zip(y_single.as_slice().iter())
    {
        assert!((a - b).abs() < 50.0 * f64::EPSILON, "composite {a} vs 2*single {b}");
    }
}

/// Additive composition only supports single-buffer `apply`; a multi-active `CpuOperator` is rejected.
#[test]
fn test_composite_operator_refs_rejects_multi_active_suboperator() {
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

    let build_mass = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Mass1DBuild").unwrap())
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()
        .unwrap();
    build_mass.apply(&*x_coord, &mut *qdata).unwrap();

    let op_mass = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    let qf_dual = reed
        .q_function_interior(
            1,
            vec![
                QFunctionField {
                    name: "u".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Interp,
                },
                QFunctionField {
                    name: "aux".into(),
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
            0,
            Box::new(|_ctx, nq, inputs, outputs| {
                let u = inputs[0];
                let aux = inputs[1];
                let v = &mut outputs[0];
                for i in 0..nq {
                    v[i] = u[i] + aux[i];
                }
                Ok(())
            }),
        )
        .unwrap();

    let op_dual = reed
        .operator_builder()
        .qfunction(qf_dual)
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("aux", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .unwrap();

    assert!(op_dual.requires_field_named_buffers());

    let comp = reed.composite_operator_refs(&[&op_mass, &op_dual]).unwrap();

    // Mixed single-buffer apply is rejected because one sub-operator requires named buffers.
    let u_single = reed.vector_from_slice(&vec![1.0_f64; ndofs]).unwrap();
    let mut v_single = reed.vector(ndofs).unwrap();
    v_single.set_value(0.0).unwrap();
    let err = comp
        .apply(&*u_single, &mut *v_single)
        .expect_err("single-buffer apply should be rejected for mixed named-buffer composite");
    assert!(matches!(err, ReedError::Operator(_)));
    assert!(err.to_string().contains("apply_field_buffers"));

    // Named-buffer path should work and match sum of sub-operators.
    let u = reed.vector_from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
    let aux = reed.vector_from_slice(&[0.5_f64, -1.0, 2.0]).unwrap();
    let mut y_comp = reed.vector(ndofs).unwrap();
    y_comp.set_value(0.0).unwrap();
    let inputs = [
        ("u", &*u as &dyn VectorTrait<f64>),
        ("aux", &*aux as &dyn VectorTrait<f64>),
    ];
    let mut outputs = [("v", &mut *y_comp as &mut dyn VectorTrait<f64>)];
    comp.apply_field_buffers(&inputs, &mut outputs).unwrap();

    let mut y_ref = reed.vector(ndofs).unwrap();
    y_ref.set_value(0.0).unwrap();
    op_mass.apply_add(&*u, &mut *y_ref).unwrap();
    let dual_in = [
        ("u", &*u as &dyn VectorTrait<f64>),
        ("aux", &*aux as &dyn VectorTrait<f64>),
    ];
    let mut dual_out = [("v", &mut *y_ref as &mut dyn VectorTrait<f64>)];
    op_dual.apply_add_field_buffers(&dual_in, &mut dual_out).unwrap();
    for i in 0..ndofs {
        assert!((y_comp.as_slice()[i] - y_ref.as_slice()[i]).abs() < 1e-11);
    }
}

#[test]
fn test_poisson_1d_apply() {
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
    let ind_u = vec![0, 1, 1, 2];

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

    let build_poisson = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Poisson1DBuild").unwrap())
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()
        .unwrap();
    build_poisson.apply(&*x_coord, &mut *qdata).unwrap();

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

    let w = reed.vector_from_slice(&[0.25_f64, -0.5, 0.75]).unwrap();
    let mut z_adj = reed.vector(ndofs).unwrap();
    let mut z_fwd = reed.vector(ndofs).unwrap();
    z_adj.set_value(0.0).unwrap();
    z_fwd.set_value(0.0).unwrap();
    op_poisson
        .apply_with_transpose(OperatorTransposeRequest::Adjoint, &*w, &mut *z_adj)
        .unwrap();
    op_poisson.apply(&*w, &mut *z_fwd).unwrap();
    for i in 0..ndofs {
        assert!(
            (z_adj.as_slice()[i] - z_fwd.as_slice()[i]).abs() < 200.0 * f64::EPSILON,
            "adjoint vs forward at dof {i}"
        );
    }
}

/// `elem_restriction_at_points` delegates to the same CPU implementation as `elem_restriction`
/// (libCEED `CeedElemRestrictionCreateAtPoints` naming only).
#[test]
fn test_elem_restriction_ceed_int_offsets_rejects_overflow() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let bad = [i64::from(i32::MAX) + 1];
    assert!(reed
        .elem_restriction_ceed_int_offsets(1, 1, 1, 1, 1, &bad)
        .is_err());
}

#[test]
fn test_elem_restriction_ceed_int_offsets_matches_i32() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let ndofs = 3usize;
    let off_i32 = [0i32, 1, 1, 2];
    let off_i64: [i64; 4] = [0, 1, 1, 2];
    let r32 = reed
        .elem_restriction(nelem, 2, 1, 1, ndofs, &off_i32)
        .unwrap();
    let r64 = reed
        .elem_restriction_ceed_int_offsets(nelem, 2, 1, 1, ndofs, &off_i64)
        .unwrap();
    let g = vec![10.0_f64, 20.0, 30.0];
    let mut a = vec![0.0_f64; 4];
    let mut b = vec![0.0_f64; 4];
    r32.apply(TransposeMode::NoTranspose, &g, &mut a).unwrap();
    r64.apply(TransposeMode::NoTranspose, &g, &mut b).unwrap();
    assert_eq!(a, b);
}

#[test]
fn test_strided_elem_restriction_ceed_int_strides_matches_i32() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let q = 3usize;
    let s32 = [1i32, q as i32, q as i32];
    let s64 = [1i64, q as i64, q as i64];
    let r32 = reed
        .strided_elem_restriction(nelem, q, 1, nelem * q, s32)
        .unwrap();
    let r64 = reed
        .strided_elem_restriction_ceed_int_strides(nelem, q, 1, nelem * q, s64)
        .unwrap();
    let global: Vec<f64> = (1..=nelem * q).map(|i| i as f64).collect();
    let mut a = vec![0.0_f64; nelem * q];
    let mut b = vec![0.0_f64; nelem * q];
    r32.apply(TransposeMode::NoTranspose, &global, &mut a)
        .unwrap();
    r64.apply(TransposeMode::NoTranspose, &global, &mut b)
        .unwrap();
    assert_eq!(a, b);
}

/// Nested `CompositeOperator` (libCEED-style multi-level additive composition).
#[test]
fn test_nested_composite_operator() {
    struct ScaleOp {
        n: usize,
        scale: f64,
    }
    impl OperatorTrait<f64> for ScaleOp {
        fn global_vector_len_hint(&self) -> Option<usize> {
            Some(self.n)
        }
        fn apply(
            &self,
            input: &dyn VectorTrait<f64>,
            output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            output.set_value(0.0)?;
            self.apply_add(input, output)
        }
        fn apply_add(
            &self,
            input: &dyn VectorTrait<f64>,
            output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            for i in 0..self.n {
                output.as_mut_slice()[i] += self.scale * input.as_slice()[i];
            }
            Ok(())
        }
        fn linear_assemble_diagonal(
            &self,
            assembled: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            assembled.set_value(0.0)?;
            for i in 0..self.n {
                assembled.as_mut_slice()[i] = self.scale;
            }
            Ok(())
        }

        fn linear_assemble_add_diagonal(
            &self,
            assembled: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            for i in 0..self.n {
                assembled.as_mut_slice()[i] += self.scale;
            }
            Ok(())
        }
    }

    let inner = CompositeOperator::new(vec![
        Box::new(ScaleOp { n: 2, scale: 1.0 }) as Box<dyn OperatorTrait<f64>>,
        Box::new(ScaleOp { n: 2, scale: 2.0 }),
    ])
    .unwrap();
    let outer = CompositeOperator::new(vec![
        Box::new(inner),
        Box::new(ScaleOp { n: 2, scale: 10.0 }),
    ])
    .unwrap();

    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let x = reed.vector_from_slice(&[1.0_f64, 1.0]).unwrap();
    let mut y = reed.vector(2).unwrap();
    y.set_value(0.0).unwrap();
    outer.apply(&*x, &mut *y).unwrap();
    // (1+2+10) * 1 = 13 per entry
    assert_eq!(y.as_slice(), &[13.0, 13.0]);
}

#[test]
fn test_cpu_elem_restriction_at_points_matches_elem_restriction() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let nelem = 2usize;
    let elemsize = 2usize;
    let lsize = 3usize;
    let offsets = vec![0i32, 1, 1, 2];

    let r_offset = reed
        .elem_restriction(nelem, elemsize, 1, 1, lsize, &offsets)
        .unwrap();
    let r_at_points = reed
        .elem_restriction_at_points(nelem, elemsize, 1, 1, lsize, &offsets)
        .unwrap();

    let global = vec![1.0_f64, 2.0, 3.0];
    let mut local_a = vec![0.0_f64; nelem * elemsize];
    let mut local_b = vec![0.0_f64; nelem * elemsize];
    r_offset
        .apply(TransposeMode::NoTranspose, &global, &mut local_a)
        .unwrap();
    r_at_points
        .apply(TransposeMode::NoTranspose, &global, &mut local_b)
        .unwrap();
    assert_eq!(local_a, local_b);

    let mut out_a = vec![0.0_f64; lsize];
    let mut out_b = vec![0.0_f64; lsize];
    r_offset
        .apply(TransposeMode::Transpose, &local_a, &mut out_a)
        .unwrap();
    r_at_points
        .apply(TransposeMode::Transpose, &local_b, &mut out_b)
        .unwrap();
    assert_eq!(out_a, out_b);
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
            0,
            Box::new(|_ctx, q, inputs, outputs| {
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

/// `CpuOperator` assembly and qfunction execution stay on the host; field objects may be
/// `WgpuVector` / `WgpuElemRestriction` / `WgpuBasis` when `Reed` is `/gpu/wgpu`. This matches the
/// libCEED-style split where the operator graph is logical and sub-parts can use device execution.
#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_hybrid_mass_operator_apply_matches_cpu() {
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;
    let ind_u = [0i32, 1, 1, 2];
    let strides = [1i32, q as i32, q as i32];

    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();

    let r_u_cpu = reed_cpu
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let r_q_cpu = reed_cpu
        .strided_elem_restriction(nelem, q, 1, nelem * q, strides)
        .unwrap();
    let b_u_cpu = reed_cpu
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();

    let r_u_gpu = reed_gpu
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let r_q_gpu = reed_gpu
        .strided_elem_restriction(nelem, q, 1, nelem * q, strides)
        .unwrap();
    let b_u_gpu = reed_gpu
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();

    let u_vals = [1.0_f32, 0.5, -0.25];
    let qdata_host = [1.0_f32; 4];

    let u_cpu = reed_cpu.vector_from_slice(&u_vals).unwrap();
    let q_cpu = reed_cpu.vector_from_slice(&qdata_host).unwrap();
    let mut v_cpu = reed_cpu.vector(ndofs).unwrap();
    v_cpu.set_value(0.0).unwrap();

    let op_cpu = reed_cpu
        .operator_builder()
        .qfunction(reed_cpu.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u_cpu), Some(&*b_u_cpu), FieldVector::Active)
        .field("qdata", Some(&*r_q_cpu), None, FieldVector::Passive(&*q_cpu))
        .field("v", Some(&*r_u_cpu), Some(&*b_u_cpu), FieldVector::Active)
        .build()
        .unwrap();
    op_cpu.check_ready().unwrap();
    op_cpu.apply(&*u_cpu, &mut *v_cpu).unwrap();

    let u_gpu = reed_gpu.vector_from_slice(&u_vals).unwrap();
    let q_gpu = reed_gpu.vector_from_slice(&qdata_host).unwrap();
    let mut v_gpu = reed_gpu.vector(ndofs).unwrap();
    v_gpu.set_value(0.0).unwrap();

    let op_gpu = reed_gpu
        .operator_builder()
        .qfunction(reed_gpu.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u_gpu), Some(&*b_u_gpu), FieldVector::Active)
        .field("qdata", Some(&*r_q_gpu), None, FieldVector::Passive(&*q_gpu))
        .field("v", Some(&*r_u_gpu), Some(&*b_u_gpu), FieldVector::Active)
        .build()
        .unwrap();
    op_gpu.check_ready().unwrap();
    op_gpu.apply(&*u_gpu, &mut *v_gpu).unwrap();

    let mut buf_cpu = vec![0.0_f32; ndofs];
    let mut buf_gpu = vec![0.0_f32; ndofs];
    v_cpu.copy_to_slice(&mut buf_cpu).unwrap();
    v_gpu.copy_to_slice(&mut buf_gpu).unwrap();
    for i in 0..ndofs {
        assert!(
            (buf_cpu[i] - buf_gpu[i]).abs() < 5.0e-4,
            "dof {i}: cpu {} gpu {}",
            buf_cpu[i],
            buf_gpu[i]
        );
    }
}

/// Same hybrid WGPU field stack as [`test_wgpu_hybrid_mass_operator_apply_matches_cpu`], but
/// exercises `OperatorTrait::apply_with_transpose` (`Forward` delegates to `apply`; `Adjoint`
/// matches forward for symmetric `MassApply` with constant quadrature data).
#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_hybrid_mass_operator_transpose_matches_cpu() {
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;
    let ind_u = [0i32, 1, 1, 2];
    let strides = [1i32, q as i32, q as i32];

    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();

    let r_u_cpu = reed_cpu
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let r_q_cpu = reed_cpu
        .strided_elem_restriction(nelem, q, 1, nelem * q, strides)
        .unwrap();
    let b_u_cpu = reed_cpu
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();

    let r_u_gpu = reed_gpu
        .elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)
        .unwrap();
    let r_q_gpu = reed_gpu
        .strided_elem_restriction(nelem, q, 1, nelem * q, strides)
        .unwrap();
    let b_u_gpu = reed_gpu
        .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
        .unwrap();

    let qdata_host = [1.25_f32; 4];
    let w_vals = [0.25_f32, -1.0, 2.0];

    let q_cpu = reed_cpu.vector_from_slice(&qdata_host).unwrap();
    let q_gpu = reed_gpu.vector_from_slice(&qdata_host).unwrap();

    let op_cpu = reed_cpu
        .operator_builder()
        .qfunction(reed_cpu.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u_cpu), Some(&*b_u_cpu), FieldVector::Active)
        .field("qdata", Some(&*r_q_cpu), None, FieldVector::Passive(&*q_cpu))
        .field("v", Some(&*r_u_cpu), Some(&*b_u_cpu), FieldVector::Active)
        .build()
        .unwrap();

    let op_gpu = reed_gpu
        .operator_builder()
        .qfunction(reed_gpu.q_function_by_name("MassApply").unwrap())
        .field("u", Some(&*r_u_gpu), Some(&*b_u_gpu), FieldVector::Active)
        .field("qdata", Some(&*r_q_gpu), None, FieldVector::Passive(&*q_gpu))
        .field("v", Some(&*r_u_gpu), Some(&*b_u_gpu), FieldVector::Active)
        .build()
        .unwrap();

    let w_cpu = reed_cpu.vector_from_slice(&w_vals).unwrap();
    let w_gpu = reed_gpu.vector_from_slice(&w_vals).unwrap();

    let mut y_fwd_cpu = reed_cpu.vector(ndofs).unwrap();
    y_fwd_cpu.set_value(0.0).unwrap();
    op_cpu
        .apply_with_transpose(
            OperatorTransposeRequest::Forward,
            &*w_cpu,
            &mut *y_fwd_cpu,
        )
        .unwrap();
    let mut y_apply_cpu = reed_cpu.vector(ndofs).unwrap();
    y_apply_cpu.set_value(0.0).unwrap();
    op_cpu.apply(&*w_cpu, &mut *y_apply_cpu).unwrap();
    for i in 0..ndofs {
        assert!(
            (y_fwd_cpu.as_slice()[i] - y_apply_cpu.as_slice()[i]).abs() < 1.0e-5,
            "cpu forward vs apply dof {i}"
        );
    }

    let mut y_fwd_gpu = reed_gpu.vector(ndofs).unwrap();
    y_fwd_gpu.set_value(0.0).unwrap();
    op_gpu
        .apply_with_transpose(
            OperatorTransposeRequest::Forward,
            &*w_gpu,
            &mut *y_fwd_gpu,
        )
        .unwrap();
    let mut y_apply_gpu = reed_gpu.vector(ndofs).unwrap();
    y_apply_gpu.set_value(0.0).unwrap();
    op_gpu.apply(&*w_gpu, &mut *y_apply_gpu).unwrap();
    for i in 0..ndofs {
        assert!(
            (y_fwd_gpu.as_slice()[i] - y_apply_gpu.as_slice()[i]).abs() < 1.0e-4,
            "gpu forward vs apply dof {i}"
        );
    }

    let mut y_adj_cpu = reed_cpu.vector(ndofs).unwrap();
    y_adj_cpu.set_value(0.0).unwrap();
    op_cpu
        .apply_with_transpose(
            OperatorTransposeRequest::Adjoint,
            &*w_cpu,
            &mut *y_adj_cpu,
        )
        .unwrap();
    for i in 0..ndofs {
        assert!(
            (y_adj_cpu.as_slice()[i] - y_apply_cpu.as_slice()[i]).abs() < 5.0e-4,
            "cpu adjoint vs forward dof {i}"
        );
    }

    let mut y_adj_gpu = reed_gpu.vector(ndofs).unwrap();
    y_adj_gpu.set_value(0.0).unwrap();
    op_gpu
        .apply_with_transpose(
            OperatorTransposeRequest::Adjoint,
            &*w_gpu,
            &mut *y_adj_gpu,
        )
        .unwrap();
    for i in 0..ndofs {
        assert!(
            (y_adj_gpu.as_slice()[i] - y_apply_gpu.as_slice()[i]).abs() < 5.0e-4,
            "gpu adjoint vs forward dof {i}"
        );
    }

    for i in 0..ndofs {
        assert!(
            (y_adj_cpu.as_slice()[i] - y_adj_gpu.as_slice()[i]).abs() < 5.0e-4,
            "cpu vs gpu adjoint dof {i}"
        );
    }
}

// `GpuRuntime::mass_apply_qp_f32_host` / `mass_apply_qp_transpose_accumulate_f32_host`:
// host slice upload + MassApply qp shaders + readback (no CpuOperator wiring yet).
#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_gpu_runtime_mass_apply_qp_host_bridge() {
    use reed::{GpuRuntime, WgpuBackend};

    let backend = WgpuBackend::<f32>::new();
    let Some(rt_arc) = backend.gpu_runtime() else {
        return;
    };
    let rt: &GpuRuntime = rt_arc.as_ref();

    let n = 256usize;
    let u: Vec<f32> = (0..n).map(|i| (i as f32) * 0.017 + 0.2).collect();
    let q: Vec<f32> = (0..n).map(|i| 0.9 + (i as f32) * 0.003).collect();
    let mut v = vec![0.0_f32; n];
    rt.mass_apply_qp_f32_host(&u, &q, &mut v).unwrap();
    for i in 0..n {
        let exp = u[i] * q[i];
        assert!((v[i] - exp).abs() < 2.0e-3, "forward i={i}");
    }

    let dv: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 + 0.4).collect();
    let mut du: Vec<f32> = (0..n).map(|i| (i as f32) * 0.12).collect();
    let du_before = du.clone();
    rt.mass_apply_qp_transpose_accumulate_f32_host(&dv, &q, &mut du)
        .unwrap();
    for i in 0..n {
        let exp = du_before[i] + dv[i] * q[i];
        assert!((du[i] - exp).abs() < 2.0e-3, "transpose i={i}");
    }
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
    let r = reed.elem_restriction(2, 2, 1, 1, 3, &[0, 1, 1, 2]).unwrap();

    let global = vec![10.0_f32, 20.0, 30.0];
    let mut local = vec![0.0_f32; 4];
    r.apply(TransposeMode::NoTranspose, &global, &mut local)
        .unwrap();
    assert_eq!(local, vec![10.0_f32, 20.0, 20.0, 30.0]);
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_elem_restriction_transpose_offset_matches_cpu() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let r_cpu = reed_cpu
        .elem_restriction(2, 2, 1, 1, 3, &[0, 1, 1, 2])
        .unwrap();
    let r_gpu = reed_gpu
        .elem_restriction(2, 2, 1, 1, 3, &[0, 1, 1, 2])
        .unwrap();

    let local = vec![10.0_f32, 20.0, 20.0, 30.0];
    let mut g_cpu = vec![0.0_f32; 3];
    let mut g_gpu = vec![0.0_f32; 3];
    r_cpu
        .apply(TransposeMode::Transpose, &local, &mut g_cpu)
        .unwrap();
    r_gpu
        .apply(TransposeMode::Transpose, &local, &mut g_gpu)
        .unwrap();
    assert_eq!(g_cpu, g_gpu);
    assert_eq!(g_cpu, vec![10.0_f32, 40.0, 30.0]);
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_elem_restriction_no_transpose_strided_f32() {
    let reed = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let nelem = 2usize;
    let q = 3usize;
    let r = reed
        .strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])
        .unwrap();

    let mut global = vec![0.0_f32; nelem * q];
    for i in 0..global.len() {
        global[i] = (i + 1) as f32;
    }
    let mut local = vec![0.0_f32; nelem * q];
    r.apply(TransposeMode::NoTranspose, &global, &mut local)
        .unwrap();
    assert_eq!(local, global);
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_elem_restriction_transpose_strided_matches_cpu() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let nelem = 2usize;
    let q = 3usize;
    let strides = [1, q as i32, q as i32];
    let r_cpu = reed_cpu
        .strided_elem_restriction(nelem, q, 1, nelem * q, strides)
        .unwrap();
    let r_gpu = reed_gpu
        .strided_elem_restriction(nelem, q, 1, nelem * q, strides)
        .unwrap();

    let local: Vec<f32> = (0..nelem * q).map(|i| (i + 1) as f32).collect();
    let mut g_cpu = vec![0.0_f32; nelem * q];
    let mut g_gpu = vec![0.0_f32; nelem * q];
    r_cpu
        .apply(TransposeMode::Transpose, &local, &mut g_cpu)
        .unwrap();
    r_gpu
        .apply(TransposeMode::Transpose, &local, &mut g_gpu)
        .unwrap();
    assert_eq!(g_cpu, g_gpu);
    assert_eq!(g_cpu, local);
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_elem_restriction_no_transpose_offset_f64() {
    let reed = Reed::<f64>::init("/gpu/wgpu").unwrap();
    let r = reed.elem_restriction(2, 2, 1, 1, 3, &[0, 1, 1, 2]).unwrap();

    let global = vec![10.0_f64, 20.0, 30.0];
    let mut local = vec![0.0_f64; 4];
    r.apply(TransposeMode::NoTranspose, &global, &mut local)
        .unwrap();
    assert_eq!(local, vec![10.0_f64, 20.0, 20.0, 30.0]);
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_elem_restriction_no_transpose_strided_f64() {
    let reed = Reed::<f64>::init("/gpu/wgpu").unwrap();
    let nelem = 2usize;
    let q = 3usize;
    let r = reed
        .strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])
        .unwrap();

    let mut global = vec![0.0_f64; nelem * q];
    for i in 0..global.len() {
        global[i] = (i + 1) as f64;
    }
    let mut local = vec![0.0_f64; nelem * q];
    r.apply(TransposeMode::NoTranspose, &global, &mut local)
        .unwrap();
    assert_eq!(local, global);
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_elem_restriction_transpose_offset_f64_matches_cpu() {
    let reed_cpu = Reed::<f64>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f64>::init("/gpu/wgpu").unwrap();
    let r_cpu = reed_cpu
        .elem_restriction(2, 2, 1, 1, 3, &[0, 1, 1, 2])
        .unwrap();
    let r_gpu = reed_gpu
        .elem_restriction(2, 2, 1, 1, 3, &[0, 1, 1, 2])
        .unwrap();

    let local = vec![10.0_f64, 20.0, 20.0, 30.0];
    let mut g_cpu = vec![0.0_f64; 3];
    let mut g_gpu = vec![0.0_f64; 3];
    r_cpu
        .apply(TransposeMode::Transpose, &local, &mut g_cpu)
        .unwrap();
    r_gpu
        .apply(TransposeMode::Transpose, &local, &mut g_gpu)
        .unwrap();
    assert_eq!(g_cpu, g_gpu);
    assert_eq!(g_cpu, vec![10.0_f64, 40.0, 30.0]);
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_elem_restriction_transpose_strided_f64_matches_cpu() {
    let reed_cpu = Reed::<f64>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f64>::init("/gpu/wgpu").unwrap();
    let nelem = 2usize;
    let q = 3usize;
    let strides = [1, q as i32, q as i32];
    let r_cpu = reed_cpu
        .strided_elem_restriction(nelem, q, 1, nelem * q, strides)
        .unwrap();
    let r_gpu = reed_gpu
        .strided_elem_restriction(nelem, q, 1, nelem * q, strides)
        .unwrap();

    let local: Vec<f64> = (0..nelem * q).map(|i| (i + 1) as f64).collect();
    let mut g_cpu = vec![0.0_f64; nelem * q];
    let mut g_gpu = vec![0.0_f64; nelem * q];
    r_cpu
        .apply(TransposeMode::Transpose, &local, &mut g_cpu)
        .unwrap();
    r_gpu
        .apply(TransposeMode::Transpose, &local, &mut g_gpu)
        .unwrap();
    assert_eq!(g_cpu, g_gpu);
    assert_eq!(g_cpu, local);
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
fn test_wgpu_basis_grad_matches_cpu() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();

    let b_cpu = reed_cpu
        .basis_tensor_h1_lagrange(1, 1, 3, 4, QuadMode::Gauss)
        .unwrap();
    let b_gpu = reed_gpu
        .basis_tensor_h1_lagrange(1, 1, 3, 4, QuadMode::Gauss)
        .unwrap();
    let dim = b_cpu.dim();

    let num_elem = 2usize;
    let u = vec![0.0_f32, 1.0, 2.0, 1.5, -0.5, 0.25];
    let mut v_cpu = vec![0.0_f32; num_elem * b_cpu.num_qpoints() * b_cpu.num_comp() * dim];
    let mut v_gpu = vec![0.0_f32; num_elem * b_gpu.num_qpoints() * b_gpu.num_comp() * dim];

    b_cpu
        .apply(num_elem, false, EvalMode::Grad, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(num_elem, false, EvalMode::Grad, &u, &mut v_gpu)
        .unwrap();

    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 1.0e-4);
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_grad_transpose_matches_cpu() {
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
        .apply(num_elem, true, EvalMode::Grad, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(num_elem, true, EvalMode::Grad, &u, &mut v_gpu)
        .unwrap();

    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 1.0e-4);
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_grad_matches_cpu_2d_scalar() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();

    let b_cpu = reed_cpu
        .basis_tensor_h1_lagrange(2, 1, 3, 4, QuadMode::Gauss)
        .unwrap();
    let b_gpu = reed_gpu
        .basis_tensor_h1_lagrange(2, 1, 3, 4, QuadMode::Gauss)
        .unwrap();
    let dim = b_cpu.dim();

    let num_elem = 2usize;
    let u = (0..num_elem * b_cpu.num_dof())
        .map(|i| (i as f32) * 0.11 - 0.4)
        .collect::<Vec<_>>();
    let mut v_cpu = vec![0.0_f32; num_elem * b_cpu.num_qpoints() * dim];
    let mut v_gpu = vec![0.0_f32; num_elem * b_gpu.num_qpoints() * dim];

    b_cpu
        .apply(num_elem, false, EvalMode::Grad, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(num_elem, false, EvalMode::Grad, &u, &mut v_gpu)
        .unwrap();

    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 1.0e-4, "cpu={a} gpu={b}");
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_grad_transpose_matches_cpu_2d_scalar() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();

    let b_cpu = reed_cpu
        .basis_tensor_h1_lagrange(2, 1, 3, 4, QuadMode::Gauss)
        .unwrap();
    let b_gpu = reed_gpu
        .basis_tensor_h1_lagrange(2, 1, 3, 4, QuadMode::Gauss)
        .unwrap();

    let num_elem = 2usize;
    let nq = b_cpu.num_qpoints();
    let u = (0..num_elem * nq * 2)
        .map(|i| (i as f32) * 0.02 - 0.5)
        .collect::<Vec<_>>();
    let mut v_cpu = vec![0.0_f32; num_elem * b_cpu.num_dof()];
    let mut v_gpu = vec![0.0_f32; num_elem * b_gpu.num_dof()];

    b_cpu
        .apply(num_elem, true, EvalMode::Grad, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(num_elem, true, EvalMode::Grad, &u, &mut v_gpu)
        .unwrap();

    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 1.0e-4, "cpu={a} gpu={b}");
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_grad_matches_cpu_2d_vector() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();

    let b_cpu = reed_cpu
        .basis_tensor_h1_lagrange(2, 2, 3, 4, QuadMode::Gauss)
        .unwrap();
    let b_gpu = reed_gpu
        .basis_tensor_h1_lagrange(2, 2, 3, 4, QuadMode::Gauss)
        .unwrap();
    let dim = b_cpu.dim();

    let num_elem = 2usize;
    let u = (0..num_elem * b_cpu.num_dof() * 2)
        .map(|i| (i as f32) * 0.07 - 1.0)
        .collect::<Vec<_>>();
    let mut v_cpu = vec![0.0_f32; num_elem * b_cpu.num_qpoints() * 2 * dim];
    let mut v_gpu = vec![0.0_f32; num_elem * b_gpu.num_qpoints() * 2 * dim];

    b_cpu
        .apply(num_elem, false, EvalMode::Grad, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(num_elem, false, EvalMode::Grad, &u, &mut v_gpu)
        .unwrap();

    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 1.0e-4, "cpu={a} gpu={b}");
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_grad_transpose_matches_cpu_2d_vector() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();

    let b_cpu = reed_cpu
        .basis_tensor_h1_lagrange(2, 2, 3, 4, QuadMode::Gauss)
        .unwrap();
    let b_gpu = reed_gpu
        .basis_tensor_h1_lagrange(2, 2, 3, 4, QuadMode::Gauss)
        .unwrap();

    let num_elem = 2usize;
    let nq = b_cpu.num_qpoints();
    let qcomp = 2 * 2;
    let u = (0..num_elem * nq * qcomp)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect::<Vec<_>>();
    let mut v_cpu = vec![0.0_f32; num_elem * b_cpu.num_dof() * 2];
    let mut v_gpu = vec![0.0_f32; num_elem * b_gpu.num_dof() * 2];

    b_cpu
        .apply(num_elem, true, EvalMode::Grad, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(num_elem, true, EvalMode::Grad, &u, &mut v_gpu)
        .unwrap();

    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 2.0e-4, "cpu={a} gpu={b}");
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_div_matches_cpu_2d() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let b_cpu = reed_cpu
        .basis_tensor_h1_lagrange(2, 2, 3, 4, QuadMode::Gauss)
        .unwrap();
    let b_gpu = reed_gpu
        .basis_tensor_h1_lagrange(2, 2, 3, 4, QuadMode::Gauss)
        .unwrap();
    let ne = 2usize;
    let u = vec![0.1_f32; ne * b_cpu.num_dof() * 2];
    let mut v_cpu = vec![0.0_f32; ne * b_cpu.num_qpoints()];
    let mut v_gpu = vec![0.0_f32; ne * b_gpu.num_qpoints()];
    b_cpu
        .apply(ne, false, EvalMode::Div, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(ne, false, EvalMode::Div, &u, &mut v_gpu)
        .unwrap();
    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 2.0e-4, "cpu={a} gpu={b}");
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_div_transpose_matches_cpu_2d() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let b_cpu = reed_cpu
        .basis_tensor_h1_lagrange(2, 2, 3, 4, QuadMode::Gauss)
        .unwrap();
    let b_gpu = reed_gpu
        .basis_tensor_h1_lagrange(2, 2, 3, 4, QuadMode::Gauss)
        .unwrap();
    let ne = 2usize;
    let u = (0..ne * b_cpu.num_qpoints())
        .map(|i| (i as f32) * 0.03 - 0.5)
        .collect::<Vec<_>>();
    let mut v_cpu = vec![0.0_f32; ne * b_cpu.num_dof() * 2];
    let mut v_gpu = vec![0.0_f32; ne * b_gpu.num_dof() * 2];
    b_cpu
        .apply(ne, true, EvalMode::Div, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(ne, true, EvalMode::Div, &u, &mut v_gpu)
        .unwrap();
    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 2.0e-4, "cpu={a} gpu={b}");
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_curl2d_transpose_matches_cpu() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let b_cpu = reed_cpu
        .basis_tensor_h1_lagrange(2, 2, 3, 4, QuadMode::Gauss)
        .unwrap();
    let b_gpu = reed_gpu
        .basis_tensor_h1_lagrange(2, 2, 3, 4, QuadMode::Gauss)
        .unwrap();
    let ne = 2usize;
    let u = (0..ne * b_cpu.num_qpoints())
        .map(|i| (i as f32) * 0.04 - 0.2)
        .collect::<Vec<_>>();
    let mut v_cpu = vec![0.0_f32; ne * b_cpu.num_dof() * 2];
    let mut v_gpu = vec![0.0_f32; ne * b_gpu.num_dof() * 2];
    b_cpu
        .apply(ne, true, EvalMode::Curl, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(ne, true, EvalMode::Curl, &u, &mut v_gpu)
        .unwrap();
    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 2.0e-4, "cpu={a} gpu={b}");
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_curl3d_transpose_matches_cpu() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let b_cpu = reed_cpu
        .basis_tensor_h1_lagrange(3, 3, 3, 4, QuadMode::Gauss)
        .unwrap();
    let b_gpu = reed_gpu
        .basis_tensor_h1_lagrange(3, 3, 3, 4, QuadMode::Gauss)
        .unwrap();
    let ne = 2usize;
    let u = (0..ne * b_cpu.num_qpoints() * 3)
        .map(|i| (i as f32) * 0.02 - 0.35)
        .collect::<Vec<_>>();
    let mut v_cpu = vec![0.0_f32; ne * b_cpu.num_dof() * 3];
    let mut v_gpu = vec![0.0_f32; ne * b_gpu.num_dof() * 3];
    b_cpu
        .apply(ne, true, EvalMode::Curl, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(ne, true, EvalMode::Curl, &u, &mut v_gpu)
        .unwrap();
    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 2.0e-4, "cpu={a} gpu={b}");
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_curl2d_matches_cpu() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let b_cpu = reed_cpu
        .basis_tensor_h1_lagrange(2, 2, 3, 4, QuadMode::Gauss)
        .unwrap();
    let b_gpu = reed_gpu
        .basis_tensor_h1_lagrange(2, 2, 3, 4, QuadMode::Gauss)
        .unwrap();
    let ne = 2usize;
    let u = vec![0.12_f32; ne * b_cpu.num_dof() * 2];
    let mut v_cpu = vec![0.0_f32; ne * b_cpu.num_qpoints()];
    let mut v_gpu = vec![0.0_f32; ne * b_gpu.num_qpoints()];
    b_cpu
        .apply(ne, false, EvalMode::Curl, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(ne, false, EvalMode::Curl, &u, &mut v_gpu)
        .unwrap();
    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 2.0e-4, "cpu={a} gpu={b}");
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_curl3d_matches_cpu() {
    let reed_cpu = Reed::<f32>::init("/cpu/self").unwrap();
    let reed_gpu = Reed::<f32>::init("/gpu/wgpu").unwrap();
    let b_cpu = reed_cpu
        .basis_tensor_h1_lagrange(3, 3, 2, 3, QuadMode::Gauss)
        .unwrap();
    let b_gpu = reed_gpu
        .basis_tensor_h1_lagrange(3, 3, 2, 3, QuadMode::Gauss)
        .unwrap();
    let ne = 2usize;
    let u = vec![0.07_f32; ne * b_cpu.num_dof() * 3];
    let mut v_cpu = vec![0.0_f32; ne * b_cpu.num_qpoints() * 3];
    let mut v_gpu = vec![0.0_f32; ne * b_gpu.num_qpoints() * 3];
    b_cpu
        .apply(ne, false, EvalMode::Curl, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(ne, false, EvalMode::Curl, &u, &mut v_gpu)
        .unwrap();
    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 2.0e-4, "cpu={a} gpu={b}");
    }
}

#[test]
fn test_qfunction_context_applies_in_closure() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let mut ctx = QFunctionContext::new(8);
    ctx.write_f64_le(0, 3.0).unwrap();
    let qf = reed
        .q_function_interior(
            1,
            vec![QFunctionField {
                name: "x".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
            vec![QFunctionField {
                name: "y".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
            8,
            Box::new(|ctx_b, q, inputs, outputs| {
                let scale = QFunctionContext::read_f64_le_bytes(ctx_b, 0)?;
                for i in 0..q {
                    outputs[0][i] = scale * inputs[0][i];
                }
                Ok(())
            }),
        )
        .unwrap();
    let mut out = vec![0.0_f64; 4];
    let inp = vec![1.0_f64; 4];
    qf.apply(ctx.as_bytes(), 4, &[inp.as_slice()], &mut [&mut out])
        .unwrap();
    assert_eq!(out, vec![3.0; 4]);
}

#[test]
fn test_qfunction_context_f32_i32_roundtrip_in_closure() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let mut ctx = QFunctionContext::new(8);
    ctx.write_f32_le(0, 2.5).unwrap();
    ctx.write_i32_le(4, -100).unwrap();

    let qf = reed
        .q_function_interior(
            1,
            vec![QFunctionField {
                name: "u".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
            vec![QFunctionField {
                name: "v".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
            8,
            Box::new(|ctx_b, q, inputs, outputs| {
                let a = QFunctionContext::read_f32_le_bytes(ctx_b, 0)?;
                let b = QFunctionContext::read_i32_le_bytes(ctx_b, 4)?;
                for i in 0..q {
                    outputs[0][i] = f64::from(inputs[0][i]) * f64::from(a) + f64::from(b);
                }
                Ok(())
            }),
        )
        .unwrap();

    let mut out = vec![0.0_f64; 3];
    let inp = vec![1.0_f64, 2.0_f64, 3.0_f64];
    qf.apply(ctx.as_bytes(), 3, &[inp.as_slice()], &mut [&mut out])
        .unwrap();
    assert_eq!(out, vec![-97.5, -95.0, -92.5]);
}

#[test]
fn test_lagrange_vector_div_curl_smoke() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let ne = 2usize;
    let b2 = reed
        .basis_tensor_h1_lagrange(2, 2, 3, 4, QuadMode::Gauss)
        .unwrap();
    let u2 = vec![0.5_f64; ne * b2.num_dof() * 2];
    let mut div2 = vec![0.0_f64; ne * b2.num_qpoints()];
    let mut curl2 = vec![0.0_f64; ne * b2.num_qpoints()];
    b2.apply(ne, false, EvalMode::Div, &u2, &mut div2).unwrap();
    b2.apply(ne, false, EvalMode::Curl, &u2, &mut curl2)
        .unwrap();

    let b3 = reed
        .basis_tensor_h1_lagrange(3, 3, 3, 4, QuadMode::Gauss)
        .unwrap();
    let u3 = vec![0.5_f64; ne * b3.num_dof() * 3];
    let mut div3 = vec![0.0_f64; ne * b3.num_qpoints()];
    let mut curl3 = vec![0.0_f64; ne * b3.num_qpoints() * 3];
    b3.apply(ne, false, EvalMode::Div, &u3, &mut div3).unwrap();
    b3.apply(ne, false, EvalMode::Curl, &u3, &mut curl3)
        .unwrap();
}

#[test]
fn test_cpu_simplex_p3_factory_dims_and_constant_field() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let tri = reed
        .basis_h1_simplex(ElemTopology::Triangle, 3, 1, 6)
        .unwrap();
    assert_eq!(tri.num_dof(), 10);
    assert_eq!(tri.num_qpoints(), 6);
    let tet = reed.basis_h1_simplex(ElemTopology::Tet, 3, 1, 5).unwrap();
    assert_eq!(tet.num_dof(), 20);
    assert_eq!(tet.num_qpoints(), 5);
    let ne = 1usize;
    let u = vec![1.0_f64; ne * tet.num_dof()];
    let mut v = vec![0.0_f64; ne * tet.num_qpoints()];
    tet.apply(ne, false, EvalMode::Interp, &u, &mut v).unwrap();
    for &x in &v {
        assert!(
            (x - 1.0).abs() < 1e-11,
            "constant P3 tet interp should be 1, got {x}"
        );
    }
}

#[test]
fn test_qfunction_context_scale_field_layout_and_dirty() {
    let mut ctx =
        QFunctionContext::from_field_layout(QFunctionContext::gallery_scale_fields()).unwrap();
    assert!(!ctx.host_needs_device_upload());
    ctx.write_field_f64("alpha", 1.25).unwrap();
    assert!(ctx.host_needs_device_upload());
    assert!((ctx.read_field_f64("alpha").unwrap() - 1.25).abs() < 1e-14);
    ctx.mark_host_synced_to_device();
    assert!(!ctx.host_needs_device_upload());
}

#[test]
fn test_line_simplex_basis_weights_sum() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let b = reed.basis_h1_simplex(ElemTopology::Line, 2, 1, 3).unwrap();
    let sum: f64 = b.q_weights().iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
}

#[test]
fn test_at_points_gallery_aliases_resolve() {
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    for name in [
        "MassApplyAtPoints",
        "MassApplyInterpTimesWeightAtPoints",
        "ScaleAtPoints",
        "IdentityAtPoints",
        "Poisson2DApplyAtPoints",
    ] {
        reed.q_function_by_name(name).unwrap_or_else(|e| {
            panic!("AtPoints gallery alias {name:?} should resolve: {e:?}")
        });
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

#[cfg(feature = "wgpu-backend")]
#[test]
fn test_wgpu_basis_weight_transpose_matches_cpu() {
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
        .apply(num_elem, true, EvalMode::Weight, &u, &mut v_cpu)
        .unwrap();
    b_gpu
        .apply(num_elem, true, EvalMode::Weight, &u, &mut v_gpu)
        .unwrap();

    for (a, b) in v_cpu.iter().zip(v_gpu.iter()) {
        assert!((a - b).abs() < 1.0e-5);
    }
}
