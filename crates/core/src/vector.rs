use crate::{enums::NormType, error::ReedResult, scalar::Scalar};

/// 抽象数值向量 trait
pub trait VectorTrait<T: Scalar>: Send + Sync {
    /// 向量长度
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 将数据复制进向量（来自主机切片）
    fn copy_from_slice(&mut self, data: &[T]) -> ReedResult<()>;

    /// 将数据复制出向量（到主机切片）
    fn copy_to_slice(&self, data: &mut [T]) -> ReedResult<()>;

    /// 将所有元素设为常量
    fn set_value(&mut self, val: T) -> ReedResult<()>;

    /// AXPY: self = alpha * x + self
    fn axpy(&mut self, alpha: T, x: &dyn VectorTrait<T>) -> ReedResult<()>;

    /// 向量缩放: self *= alpha
    fn scale(&mut self, alpha: T) -> ReedResult<()>;

    /// 计算向量范数
    fn norm(&self, norm_type: NormType) -> ReedResult<T>;

    /// 获取内部数据的只读引用（用于 CPU 后端）
    fn as_slice(&self) -> &[T];

    /// 获取内部数据的可变引用（用于 CPU 后端）
    fn as_mut_slice(&mut self) -> &mut [T];
}
