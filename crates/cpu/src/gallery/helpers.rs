//! Shared helpers for scalar-generic gallery QFunctions (libCEED-compatible context bytes).

use num_traits::NumCast;
use reed_core::{error::ReedResult, scalar::Scalar, ReedError};

/// Near-singular Jacobian threshold (magnitude); scales with `T` via [`NumCast`].
#[inline]
pub(crate) fn singular_jacobian_tol<T: Scalar>() -> T {
    NumCast::from(1e-12_f64).expect("singular_jacobian_tol")
}

/// libCEED `Scale` context: one `f64` little-endian (8 bytes), applied as [`Scalar`] `alpha`.
#[inline]
pub(crate) fn scale_alpha_from_libceed_context<T: Scalar>(ctx: &[u8]) -> ReedResult<T> {
    if ctx.len() < 8 {
        return Err(ReedError::QFunction(
            "Scale expects 8-byte context (f64 LE scale)".into(),
        ));
    }
    let b: [u8; 8] = ctx[0..8]
        .try_into()
        .map_err(|_| ReedError::QFunction("Scale: context slice".into()))?;
    let a64 = f64::from_le_bytes(b);
    NumCast::from(a64).ok_or_else(|| {
        ReedError::QFunction("Scale: could not convert context f64 to scalar".into())
    })
}
