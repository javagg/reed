use reed_core::{
    basis::BasisTrait, elem_restriction::ElemRestrictionTrait, enums::QuadMode, error::ReedResult,
    reed::Backend, scalar::Scalar, vector::VectorTrait, ReedError,
};
use std::sync::Arc;

mod runtime;
mod basis;
mod elem_restriction;
mod vector;

use basis::WgpuBasis;
use elem_restriction::WgpuElemRestriction;
use runtime::GpuRuntime;
use vector::WgpuVector;

pub struct WgpuBackend<T: Scalar> {
    gpu_available: bool,
    adapter_name: Option<String>,
    runtime: Option<Arc<GpuRuntime>>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> Default for WgpuBackend<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> WgpuBackend<T> {
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }));

        let (gpu_available, adapter_name, runtime) = if let Some(adapter) = adapter {
            let info = adapter.get_info();
            let rt = GpuRuntime::new(&adapter).map(GpuRuntime::shared);
            (rt.is_some(), Some(format!("{} ({:?})", info.name, info.backend)), rt)
        } else {
            (false, None, None)
        };

        Self {
            gpu_available,
            adapter_name,
            runtime,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    pub fn adapter_name(&self) -> Option<&str> {
        self.adapter_name.as_deref()
    }
}

impl<T: Scalar> Backend<T> for WgpuBackend<T> {
    fn resource_name(&self) -> &str {
        "/gpu/wgpu"
    }

    fn create_vector(&self, size: usize) -> ReedResult<Box<dyn VectorTrait<T>>> {
        Ok(Box::new(WgpuVector::<T>::new(size, self.runtime.clone())))
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
        Ok(Box::new(WgpuElemRestriction::<T>::new_offset(
            nelem,
            elemsize,
            ncomp,
            compstride,
            lsize,
            offsets,
            self.runtime.clone(),
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
        Ok(Box::new(WgpuElemRestriction::<T>::new_strided(
            nelem,
            elemsize,
            ncomp,
            lsize,
            strides,
            self.runtime.clone(),
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
        Ok(Box::new(WgpuBasis::<T>::new(
            dim,
            ncomp,
            p,
            q,
            qmode,
            self.runtime.clone(),
        )?))
    }
}

pub fn wgpu_available() -> bool {
    let backend = WgpuBackend::<f64>::new();
    backend.is_gpu_available()
}

pub fn wgpu_adapter_name() -> Option<String> {
    let backend = WgpuBackend::<f64>::new();
    backend.adapter_name().map(ToOwned::to_owned)
}

pub fn require_wgpu() -> ReedResult<()> {
    if wgpu_available() {
        Ok(())
    } else {
        Err(ReedError::BackendNotSupported(
            "no suitable wgpu adapter found".into(),
        ))
    }
}
