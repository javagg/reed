use crate::{error::ReedResult, scalar::Scalar, vector::VectorTrait};

/// 完整弱形式算符 trait
///
/// 组合 ElemRestriction + Basis + QFunction，实现：
///   v = A(u) = Eᵀ Bᵀ D B E u
#[cfg(not(target_arch = "wasm32"))]
pub trait OperatorTrait<T: Scalar>: Send + Sync {
    fn apply(&self, input: &dyn VectorTrait<T>, output: &mut dyn VectorTrait<T>) -> ReedResult<()>;
    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()>;
    fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()>;
}

#[cfg(target_arch = "wasm32")]
pub trait OperatorTrait<T: Scalar> {
    fn apply(&self, input: &dyn VectorTrait<T>, output: &mut dyn VectorTrait<T>) -> ReedResult<()>;
    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()>;
    fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()>;
}
