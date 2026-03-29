use crate::{enums::EvalMode, error::ReedResult, scalar::Scalar};

/// 参考单元基函数 trait
pub trait BasisTrait<T: Scalar>: Send + Sync {
    /// 空间拓扑维度
    fn dim(&self) -> usize;

    /// 每单元自由度数量（插值节点数）
    fn num_dof(&self) -> usize;

    /// 每单元积分点数量
    fn num_qpoints(&self) -> usize;

    /// 分量数量（标量场为 1）
    fn num_comp(&self) -> usize;

    /// 应用基函数算符
    ///
    /// - `transpose = false`：正向，u_local → v_qpt
    /// - `transpose = true` ：转置，v_qpt → u_local
    /// - `eval_mode`：指定求值类型
    /// - `num_elem`：批量处理的单元数
    fn apply(
        &self,
        num_elem: usize,
        transpose: bool,
        eval_mode: EvalMode,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<()>;

    /// 积分权重（长度 = num_qpoints()）
    fn q_weights(&self) -> &[T];

    /// 积分点坐标（参考单元，行主序 [nqpts × dim]）
    fn q_ref(&self) -> &[T];
}
