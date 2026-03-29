use reed::{FieldVector, OperatorTrait, QuadMode, Reed};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reed = Reed::<f64>::init("/cpu/self")?;
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;

    let node_coords = vec![-1.0, 0.0, 1.0];
    let x_coord = reed.vector_from_slice(&node_coords)?;
    let mut qdata = reed.vector(nelem * q)?;
    qdata.set_value(0.0)?;

    let ind_x = vec![0, 1, 1, 2];
    let ind_u = ind_x.clone();

    let r_x = reed.elem_restriction(nelem, 2, 1, 1, ndofs, &ind_x)?;
    let r_u = reed.elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)?;
    let r_q = reed.strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])?;

    let b_x = reed.basis_tensor_h1_lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    let b_u = reed.basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)?;

    let build = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Mass1DBuild")?)
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()?;
    build.apply(&*x_coord, &mut *qdata)?;

    let op_mass = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply")?)
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()?;

    let u = reed.vector_from_slice(&vec![1.0_f64; ndofs])?;
    let mut v = reed.vector(ndofs)?;
    v.set_value(0.0)?;
    op_mass.apply(&*u, &mut *v)?;

    let mut values = vec![0.0; ndofs];
    v.copy_to_slice(&mut values)?;
    println!("mass operator output: {values:?}");
    Ok(())
}
