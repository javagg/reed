use crate::{enums::TransposeMode, error::ReedResult, scalar::Scalar};

/// 自由度限制算符 trait
///
/// E  (NoTranspose): u_local[e,i] = u_global[offsets[e,i]]
/// Eᵀ (Transpose)  : u_global[offsets[e,i]] += u_local[e,i]
pub trait ElemRestrictionTrait<T: Scalar>: Send + Sync {
    /// 单元总数
    fn num_elements(&self) -> usize;

    /// 每单元自由度数
    fn num_dof_per_elem(&self) -> usize;

    /// 全局向量自由度数
    fn num_global_dof(&self) -> usize;

    /// 分量数
    fn num_comp(&self) -> usize;

    /// 应用限制算符
    fn apply(&self, t_mode: TransposeMode, u: &[T], v: &mut [T]) -> ReedResult<()>;

    /// 本地向量大小 = num_elements * num_dof_per_elem * num_comp
    fn local_size(&self) -> usize {
        self.num_elements() * self.num_dof_per_elem() * self.num_comp()
    }
}
