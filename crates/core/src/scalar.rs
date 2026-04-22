use num_traits::{Float, NumAssign};
use std::fmt::Debug;

/// Scalar type bound, supports f32 / f64.
pub trait Scalar: Float + NumAssign + Send + Sync + Copy + Debug + 'static {
    const ZERO: Self;
    const ONE: Self;
    const EPSILON: Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0_f32;
    const ONE: Self = 1.0_f32;
    const EPSILON: Self = f32::EPSILON;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0_f64;
    const ONE: Self = 1.0_f64;
    const EPSILON: Self = f64::EPSILON;
}
