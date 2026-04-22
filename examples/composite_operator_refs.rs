//! Compose two `CpuOperator` mass applies with `Reed::composite_operator_refs` (same scope as mesh).
//!
//! Contrasts with `composite_operator.rs`, which uses `'static` `Box<dyn>` scaling ops. Here both
//! sub-operators borrow shared `ElemRestriction` / `Basis` / `qdata`, analogous to libCEED
//! composing operators under one `Ceed` context.
//! Sub-operators must use single-buffer `OperatorTrait::apply` (not multi-active `apply_field_buffers`).

use reed::{FieldVector, OperatorTrait, QuadMode, Reed, ReedResult};

fn main() -> ReedResult<()> {
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

    reed.operator_builder()
        .qfunction(reed.q_function_by_name("Mass1DBuild")?)
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()?
        .apply(&*x_coord, &mut *qdata)?;

    let op_a = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply")?)
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()?;
    let op_b = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply")?)
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()?;

    op_a.check_ready()?;
    op_b.check_ready()?;
    let composite = reed.composite_operator_refs(&[&op_a, &op_b])?;
    composite.check_ready()?;

    let u = reed.vector_from_slice(&vec![1.0_f64; ndofs])?;
    let mut y = reed.vector(ndofs)?;
    y.set_value(0.0)?;
    composite.apply(&*u, &mut *y)?;

    println!("composite of two MassApply on u=[1,1,1] => {:?}", y.as_slice());
    Ok(())
}
