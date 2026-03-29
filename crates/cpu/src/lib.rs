pub mod basis_lagrange;
pub mod elem_restriction;
pub mod gallery;
pub mod operator;
pub mod vector;

use basis_lagrange::LagrangeBasis;
use elem_restriction::CpuElemRestriction;
use reed_core::{
    basis::BasisTrait, elem_restriction::ElemRestrictionTrait, enums::QuadMode, error::ReedResult,
    qfunction::QFunctionTrait, reed::Backend, scalar::Scalar, vector::VectorTrait, ReedError,
};
use vector::CpuVector;

pub use gallery::{Mass1DBuild, MassApply, Poisson1DApply};
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
}

pub fn q_function_by_name(name: &str) -> ReedResult<Box<dyn QFunctionTrait<f64>>> {
    match name {
        "Mass1DBuild" => Ok(Box::new(Mass1DBuild::default())),
        "MassApply" => Ok(Box::new(MassApply::default())),
        "Poisson1DApply" => Ok(Box::new(Poisson1DApply::default())),
        other => Err(ReedError::QFunction(format!(
            "unknown CPU gallery qfunction '{}'",
            other
        ))),
    }
}
