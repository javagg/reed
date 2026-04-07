use crate::{enums::TransposeMode, error::ReedResult, scalar::Scalar};

/// 自由度限制算符 trait
#[cfg(not(target_arch = "wasm32"))]
pub trait ElemRestrictionTrait<T: Scalar>: Send + Sync {
    fn num_elements(&self) -> usize;
    fn num_dof_per_elem(&self) -> usize;
    fn num_global_dof(&self) -> usize;
    fn num_comp(&self) -> usize;
    fn apply(&self, t_mode: TransposeMode, u: &[T], v: &mut [T]) -> ReedResult<()>;
    fn local_size(&self) -> usize {
        self.num_elements() * self.num_dof_per_elem() * self.num_comp()
    }
}

#[cfg(target_arch = "wasm32")]
pub trait ElemRestrictionTrait<T: Scalar> {
    fn num_elements(&self) -> usize;
    fn num_dof_per_elem(&self) -> usize;
    fn num_global_dof(&self) -> usize;
    fn num_comp(&self) -> usize;
    fn apply(&self, t_mode: TransposeMode, u: &[T], v: &mut [T]) -> ReedResult<()>;
    fn local_size(&self) -> usize {
        self.num_elements() * self.num_dof_per_elem() * self.num_comp()
    }
}
