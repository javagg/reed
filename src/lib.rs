pub use reed_core::{
    Backend, BasisTrait, ClosureQFunction, ElemRestrictionTrait, ElemTopology, EvalMode,
    OperatorTrait, QFunctionClosure, QFunctionField, QFunctionTrait, QuadMode, ReedError,
    ReedResult, Scalar, TransposeMode, VectorTrait,
};
pub use reed_cpu::{q_function_by_name, CpuBackend, FieldVector, OperatorBuilder};
#[cfg(feature = "wgpu-backend")]
pub use reed_wgpu::WgpuBackend;

use std::sync::Arc;

pub struct Reed<T: Scalar> {
    inner: reed_core::Reed<T>,
}

impl<T: Scalar> Reed<T> {
    pub fn init(resource: &str) -> ReedResult<Self> {
        if matches!(resource, "/cpu/self" | "/cpu/self/ref") {
            return Ok(Self {
                inner: reed_core::Reed::from_backend(Arc::new(CpuBackend::<T>::new())),
            });
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(resource, "/gpu/wgpu" | "/gpu/wgpu/ref") {
            return Ok(Self {
                inner: reed_core::Reed::from_backend(Arc::new(WgpuBackend::<T>::new())),
            });
        }

        #[cfg(not(feature = "wgpu-backend"))]
        if matches!(resource, "/gpu/wgpu" | "/gpu/wgpu/ref") {
            return Err(ReedError::BackendNotSupported(
                "wgpu backend is disabled; build with feature 'wgpu-backend'".into(),
            ));
        }

        Err(ReedError::BackendNotSupported(resource.into()))
    }

    /// Build a Reed context from a pre-configured backend.
    pub fn from_backend(backend: Arc<dyn Backend<T>>) -> Self {
        Self {
            inner: reed_core::Reed::from_backend(backend),
        }
    }

    pub fn resource(&self) -> String {
        self.inner.resource()
    }

    pub fn vector(&self, n: usize) -> ReedResult<Box<dyn VectorTrait<T>>> {
        self.inner.vector(n)
    }

    pub fn vector_from_slice(&self, data: &[T]) -> ReedResult<Box<dyn VectorTrait<T>>> {
        self.inner.vector_from_slice(data)
    }

    pub fn elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i32],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        self.inner
            .elem_restriction(nelem, elemsize, ncomp, compstride, lsize, offsets)
    }

    pub fn strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        self.inner
            .strided_elem_restriction(nelem, elemsize, ncomp, lsize, strides)
    }

    pub fn basis_tensor_h1_lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        self.inner.basis_tensor_h1_lagrange(dim, ncomp, p, q, qmode)
    }

    pub fn basis_h1_simplex(
        &self,
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        self.inner.basis_h1_simplex(topo, poly, ncomp, q)
    }

    pub fn operator_builder<'a>(&'a self) -> OperatorBuilder<'a, T> {
        OperatorBuilder::new()
    }

    pub fn q_function_interior(
        &self,
        vector_length: usize,
        inputs: Vec<QFunctionField>,
        outputs: Vec<QFunctionField>,
        closure: Box<QFunctionClosure<T>>,
    ) -> ReedResult<Box<dyn QFunctionTrait<T>>> {
        let _ = self;
        if vector_length == 0 {
            return Err(ReedError::InvalidArgument(
                "qfunction vector_length must be greater than zero".into(),
            ));
        }
        Ok(Box::new(ClosureQFunction::new(inputs, outputs, closure)))
    }
}

impl Reed<f64> {
    pub fn q_function_by_name(&self, name: &str) -> ReedResult<Box<dyn QFunctionTrait<f64>>> {
        let _ = self;
        q_function_by_name(name)
    }
}
