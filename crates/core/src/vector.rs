use crate::{enums::NormType, error::ReedResult, scalar::Scalar};

/// 抽象数值向量 trait
#[cfg(not(target_arch = "wasm32"))]
pub trait VectorTrait<T: Scalar>: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn copy_from_slice(&mut self, data: &[T]) -> ReedResult<()>;
    fn copy_to_slice(&self, data: &mut [T]) -> ReedResult<()>;
    fn set_value(&mut self, val: T) -> ReedResult<()>;
    fn axpy(&mut self, alpha: T, x: &dyn VectorTrait<T>) -> ReedResult<()>;
    fn scale(&mut self, alpha: T) -> ReedResult<()>;
    fn norm(&self, norm_type: NormType) -> ReedResult<T>;
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
}

#[cfg(target_arch = "wasm32")]
pub trait VectorTrait<T: Scalar> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn copy_from_slice(&mut self, data: &[T]) -> ReedResult<()>;
    fn copy_to_slice(&self, data: &mut [T]) -> ReedResult<()>;
    fn set_value(&mut self, val: T) -> ReedResult<()>;
    fn axpy(&mut self, alpha: T, x: &dyn VectorTrait<T>) -> ReedResult<()>;
    fn scale(&mut self, alpha: T) -> ReedResult<()>;
    fn norm(&self, norm_type: NormType) -> ReedResult<T>;
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
}
