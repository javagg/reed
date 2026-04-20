//! Host CPU backend: [`CpuBackend`] implements [`reed_core::reed::Backend`] for any [`Scalar`]
//! (`f32` / `f64`) on vectors, restrictions, bases, and [`operator::CpuOperator`].
//!
//! Gallery QFunctions from [`q_function_by_name`] implement [`QFunctionTrait`] for any [`Scalar`]
//! (`f32` / `f64`). libCEED-style contexts that embed `f64` (e.g. [`Scale`](gallery::Scale)) are read
//! as little-endian doubles and cast to `T`.

pub mod basis_lagrange;
pub mod basis_simplex;
pub mod composite_operator;
pub mod elem_restriction;
pub mod gallery;
pub mod operator;
pub mod vector;

use basis_lagrange::LagrangeBasis;
use basis_simplex::SimplexBasis;
use elem_restriction::CpuElemRestriction;
use reed_core::{
    basis::BasisTrait,
    elem_restriction::ElemRestrictionTrait,
    enums::{ElemTopology, QuadMode},
    error::ReedResult,
    qfunction::QFunctionTrait,
    reed::Backend,
    scalar::Scalar,
    vector::VectorTrait,
    ReedError,
};
use vector::CpuVector;

pub use composite_operator::CompositeOperator;
pub use gallery::{
    Identity, IdentityScalar, Mass1DBuild, Mass2DBuild, Mass3DBuild, MassApply, Poisson1DApply,
    Poisson1DBuild, Poisson2DApply, Poisson2DBuild, Poisson3DApply, Poisson3DBuild, Scale,
    ScaleScalar, Vec2Dot, Vec3Dot, Vector2MassApply, Vector2Poisson1DApply, Vector2Poisson2DApply,
    Vector3MassApply, Vector3Poisson1DApply, Vector3Poisson2DApply, Vector3Poisson3DApply,
};
pub use operator::{CpuOperator, FieldVector, OperatorBuilder};

pub struct CpuBackend<T: Scalar> {
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> Default for CpuBackend<T> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar> CpuBackend<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: Scalar> Backend<T> for CpuBackend<T> {
    fn resource_name(&self) -> &str {
        "/cpu/self"
    }

    fn create_vector(&self, size: usize) -> ReedResult<Box<dyn VectorTrait<T>>> {
        Ok(Box::new(CpuVector::<T>::new(size)))
    }

    fn create_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i32],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Ok(Box::new(CpuElemRestriction::<T>::new_offset(
            nelem, elemsize, ncomp, compstride, lsize, offsets,
        )?))
    }

    fn create_strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Ok(Box::new(CpuElemRestriction::<T>::new_strided(
            nelem, elemsize, ncomp, lsize, strides,
        )?))
    }

    fn create_basis_tensor_h1_lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        Ok(Box::new(LagrangeBasis::<T>::new(dim, ncomp, p, q, qmode)?))
    }

    fn create_basis_h1_simplex(
        &self,
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        Ok(Box::new(SimplexBasis::<T>::new(topo, poly, ncomp, q)?))
    }
}

pub fn q_function_by_name<T: Scalar>(name: &str) -> ReedResult<Box<dyn QFunctionTrait<T>>> {
    match name {
        "Mass1DBuild" => Ok(Box::new(Mass1DBuild::default())),
        "Mass2DBuild" => Ok(Box::new(Mass2DBuild::default())),
        "Mass3DBuild" => Ok(Box::new(Mass3DBuild::default())),
        "MassApply" => Ok(Box::new(MassApply::default())),
        "Poisson1DApply" => Ok(Box::new(Poisson1DApply::default())),
        "Poisson1DBuild" => Ok(Box::new(Poisson1DBuild::default())),
        "Poisson2DBuild" => Ok(Box::new(Poisson2DBuild::default())),
        "Poisson2DApply" => Ok(Box::new(Poisson2DApply::default())),
        "Poisson3DBuild" => Ok(Box::new(Poisson3DBuild::default())),
        "Poisson3DApply" => Ok(Box::new(Poisson3DApply::default())),
        "Vec2Dot" => Ok(Box::new(Vec2Dot::new())),
        "Vec3Dot" => Ok(Box::new(Vec3Dot::new())),
        "Identity" => Ok(Box::new(Identity::default())),
        "Identity to scalar" => Ok(Box::new(IdentityScalar::default())),
        "Scale" => Ok(Box::new(Scale::default())),
        "Scale (scalar)" => Ok(Box::new(ScaleScalar::default())),
        "Vector2MassApply" => Ok(Box::new(Vector2MassApply::new())),
        "Vector2Poisson1DApply" => Ok(Box::new(Vector2Poisson1DApply::new())),
        "Vector2Poisson2DApply" => Ok(Box::new(Vector2Poisson2DApply::new())),
        "Vector3MassApply" => Ok(Box::new(Vector3MassApply::new())),
        "Vector3Poisson1DApply" => Ok(Box::new(Vector3Poisson1DApply::new())),
        "Vector3Poisson2DApply" => Ok(Box::new(Vector3Poisson2DApply::new())),
        "Vector3Poisson3DApply" => Ok(Box::new(Vector3Poisson3DApply::new())),
        other => Err(ReedError::QFunction(format!(
            "unknown CPU gallery qfunction '{}'",
            other
        ))),
    }
}
