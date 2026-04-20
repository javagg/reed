use crate::{enums::EvalMode, error::ReedResult, scalar::Scalar};

/// QFunction 字段描述符
#[derive(Debug, Clone)]
pub struct QFunctionField {
    pub name: String,
    pub num_comp: usize,
    pub eval_mode: EvalMode,
}

/// 积分点点态算符 trait
///
/// `ctx` is always `context_byte_len()` bytes (often empty). This mirrors libCEED's
/// `CeedQFunctionContext` passed into user kernels.
#[cfg(not(target_arch = "wasm32"))]
pub trait QFunctionTrait<T: Scalar>: Send + Sync {
    /// Byte length of `ctx` passed to [`Self::apply`]. Zero means `ctx` is empty.
    fn context_byte_len(&self) -> usize {
        0
    }

    fn inputs(&self) -> &[QFunctionField];
    fn outputs(&self) -> &[QFunctionField];
    fn apply(
        &self,
        ctx: &[u8],
        q: usize,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> ReedResult<()>;
}

#[cfg(target_arch = "wasm32")]
pub trait QFunctionTrait<T: Scalar> {
    fn context_byte_len(&self) -> usize {
        0
    }

    fn inputs(&self) -> &[QFunctionField];
    fn outputs(&self) -> &[QFunctionField];
    fn apply(
        &self,
        ctx: &[u8],
        q: usize,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> ReedResult<()>;
}

/// 用户闭包类型别名（`ctx` 为 qfunction 上下文字节切片）
#[cfg(not(target_arch = "wasm32"))]
pub type QFunctionClosure<T> =
    dyn Fn(&[u8], usize, &[&[T]], &mut [&mut [T]]) -> ReedResult<()> + Send + Sync;

#[cfg(target_arch = "wasm32")]
pub type QFunctionClosure<T> = dyn Fn(&[u8], usize, &[&[T]], &mut [&mut [T]]) -> ReedResult<()>;

/// 基于闭包的 QFunction 实现。
pub struct ClosureQFunction<T: Scalar> {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
    context_byte_len: usize,
    closure: Box<QFunctionClosure<T>>,
}

impl<T: Scalar> ClosureQFunction<T> {
    pub fn new(
        inputs: Vec<QFunctionField>,
        outputs: Vec<QFunctionField>,
        context_byte_len: usize,
        closure: Box<QFunctionClosure<T>>,
    ) -> Self {
        Self {
            inputs,
            outputs,
            context_byte_len,
            closure,
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for ClosureQFunction<T> {
    fn context_byte_len(&self) -> usize {
        self.context_byte_len
    }

    fn inputs(&self) -> &[QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        ctx: &[u8],
        q: usize,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> ReedResult<()> {
        (self.closure)(ctx, q, inputs, outputs)
    }
}
