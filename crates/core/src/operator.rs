use crate::{error::ReedResult, scalar::Scalar, vector::VectorTrait};

/// 完整弱形式算符 trait
///
/// 组合 ElemRestriction + Basis + QFunction，实现：
///   v = A(u) = Eᵀ Bᵀ D B E u
pub trait OperatorTrait<T: Scalar>: Send + Sync {
    /// 应用算符：output = A(input)
    fn apply(&self, input: &dyn VectorTrait<T>, output: &mut dyn VectorTrait<T>) -> ReedResult<()>;

    /// 累加应用算符：output += A(input)
    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()>;

    /// 组装线性算符的对角线
    fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()>;
}
