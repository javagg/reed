use crate::{enums::EvalMode, error::ReedResult, scalar::Scalar};

/// QFunction 字段描述符
#[derive(Debug, Clone)]
pub struct QFunctionField {
    pub name: String,
    pub num_comp: usize,
    pub eval_mode: EvalMode,
}

/// 积分点点态算符 trait
#[cfg(not(target_arch = "wasm32"))]
pub trait QFunctionTrait<T: Scalar>: Send + Sync {
    fn inputs(&self) -> &[QFunctionField];
    fn outputs(&self) -> &[QFunctionField];
    fn apply(&self, q: usize, inputs: &[&[T]], outputs: &mut [&mut [T]]) -> ReedResult<()>;
}

#[cfg(target_arch = "wasm32")]
pub trait QFunctionTrait<T: Scalar> {
    fn inputs(&self) -> &[QFunctionField];
    fn outputs(&self) -> &[QFunctionField];
    fn apply(&self, q: usize, inputs: &[&[T]], outputs: &mut [&mut [T]]) -> ReedResult<()>;
}

/// 用户闭包类型别名
#[cfg(not(target_arch = "wasm32"))]
pub type QFunctionClosure<T> =
    dyn Fn(usize, &[&[T]], &mut [&mut [T]]) -> ReedResult<()> + Send + Sync;

#[cfg(target_arch = "wasm32")]
pub type QFunctionClosure<T> =
    dyn Fn(usize, &[&[T]], &mut [&mut [T]]) -> ReedResult<()>;

/// 基于闭包的 QFunction 实现。
pub struct ClosureQFunction<T: Scalar> {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
    closure: Box<QFunctionClosure<T>>,
}

impl<T: Scalar> ClosureQFunction<T> {
    pub fn new(
        inputs: Vec<QFunctionField>,
        outputs: Vec<QFunctionField>,
        closure: Box<QFunctionClosure<T>>,
    ) -> Self {
        Self {
            inputs,
            outputs,
            closure,
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for ClosureQFunction<T> {
    fn inputs(&self) -> &[QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[QFunctionField] {
        &self.outputs
    }

    fn apply(&self, q: usize, inputs: &[&[T]], outputs: &mut [&mut [T]]) -> ReedResult<()> {
        (self.closure)(q, inputs, outputs)
    }
}
