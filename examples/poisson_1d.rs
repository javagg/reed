use reed::{FieldVector, OperatorTrait, QuadMode, Reed};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reed = Reed::<f64>::init("/cpu/self")?;
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;

    let node_coords = vec![-1.0, 0.0, 1.0];
    let qdata_values = {
        let basis = reed.basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)?;
        build_poisson_qdata(&node_coords, basis.q_weights())
    };
    let qdata = reed.vector_from_slice(&qdata_values)?;

    let ind_u = vec![0, 1, 1, 2];
    let r_u = reed.elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)?;
    let r_q = reed.strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])?;
    let b_u = reed.basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)?;

    let op_poisson = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Poisson1DApply")?)
        .field("du", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("dv", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()?;

    let u = reed.vector_from_slice(&[0.0_f64, 1.0, 0.0])?;
    let mut v = reed.vector(ndofs)?;
    v.set_value(0.0)?;
    op_poisson.apply(&*u, &mut *v)?;

    let mut values = vec![0.0; ndofs];
    v.copy_to_slice(&mut values)?;
    println!("poisson operator output: {values:?}");
    Ok(())
}
