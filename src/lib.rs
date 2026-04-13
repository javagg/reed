pub use reed_core::{
    Backend, BasisTrait, ClosureQFunction, ElemRestrictionTrait, EvalMode,
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

/// Canonical external solver backend IDs shared across subprojects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalSolverBackend {
    HypreRs,
    PetscRs,
    Mumps,
    Mkl,
}

/// Backend request surface exposed by reed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReedBackendRequest {
    NativeLinger,
    GpuWgpu,
    ExternalSolver(ExternalSolverBackend),
}

/// Compile-time/runtime capability snapshot for reed backend routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReedBackendCapabilities {
    pub gpu_wgpu: bool,
    pub hypre_rs: bool,
    pub petsc_rs: bool,
    pub mumps: bool,
    pub mkl: bool,
    pub wasm_target: bool,
}

impl ReedBackendCapabilities {
    pub fn detect() -> Self {
        Self {
            gpu_wgpu: cfg!(feature = "wgpu-backend"),
            // Solver backend execution currently routes through native linger.
            hypre_rs: cfg!(feature = "hypre-rs"),
            petsc_rs: cfg!(feature = "petsc-rs"),
            mumps: cfg!(feature = "mumps"),
            mkl: cfg!(feature = "mkl"),
            wasm_target: cfg!(target_arch = "wasm32"),
        }
    }
}

/// Deterministic backend-selection result used for diagnostics/integration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReedBackendSelectionReport {
    pub requested: Option<ReedBackendRequest>,
    pub effective_resource: String,
    pub capabilities: ReedBackendCapabilities,
    pub note: String,
}

impl<T: Scalar> Reed<T> {
    /// Return capability snapshot used by reed backend selection.
    pub fn backend_capabilities() -> ReedBackendCapabilities {
        ReedBackendCapabilities::detect()
    }

    /// Resolve requested backend into a deterministic reed runtime resource.
    pub fn backend_selection_report(
        requested: Option<ReedBackendRequest>,
    ) -> ReedBackendSelectionReport {
        let caps = ReedBackendCapabilities::detect();
        resolve_backend_request(requested, caps)
    }

    /// Initialize reed with a backend request and return selection diagnostics.
    pub fn init_with_backend(
        requested: ReedBackendRequest,
    ) -> ReedResult<(Self, ReedBackendSelectionReport)> {
        let report = Self::backend_selection_report(Some(requested));
        let reed = Self::init(&report.effective_resource)?;
        Ok((reed, report))
    }

    /// Parse a canonical backend/resource string into a backend request.
    ///
    /// Returns `None` when the resource is unknown.
    pub fn parse_backend_request(resource: &str) -> Option<ReedBackendRequest> {
        match resource {
            "/cpu/self" | "/cpu/self/ref" | "/native/linger" => {
                Some(ReedBackendRequest::NativeLinger)
            }
            "/gpu/wgpu" | "/gpu/wgpu/ref" => Some(ReedBackendRequest::GpuWgpu),
            "/solver/hypre-rs" => Some(ReedBackendRequest::ExternalSolver(
                ExternalSolverBackend::HypreRs,
            )),
            "/solver/petsc-rs" | "/solver/petsc-ffi" => Some(
                ReedBackendRequest::ExternalSolver(ExternalSolverBackend::PetscRs),
            ),
            "/solver/mumps" => Some(ReedBackendRequest::ExternalSolver(
                ExternalSolverBackend::Mumps,
            )),
            "/solver/mkl" => {
                Some(ReedBackendRequest::ExternalSolver(ExternalSolverBackend::Mkl))
            }
            _ => None,
        }
    }

    /// Initialize reed from canonical backend/resource strings.
    ///
    /// Accepted examples: `/native/linger`, `/gpu/wgpu`, `/solver/mkl`,
    /// `/solver/petsc-rs`.
    pub fn init_with_backend_resource(
        resource: &str,
    ) -> ReedResult<(Self, ReedBackendSelectionReport)> {
        let requested = Self::parse_backend_request(resource)
            .ok_or_else(|| ReedError::BackendNotSupported(resource.into()))?;
        Self::init_with_backend(requested)
    }

    pub fn init(resource: &str) -> ReedResult<Self> {
        if matches!(resource, "/cpu/self" | "/cpu/self/ref") {
            return Ok(Self {
                inner: reed_core::Reed::from_backend(Arc::new(CpuBackend::<T>::new())),
            });
        }

        #[cfg(feature = "hypre-rs")]
        if resource == "/solver/hypre-rs" {
            return Ok(Self {
                inner: reed_core::Reed::from_backend(Arc::new(CpuBackend::<T>::new())),
            });
        }

        #[cfg(feature = "petsc-rs")]
        if resource == "/solver/petsc-rs" {
            return Ok(Self {
                inner: reed_core::Reed::from_backend(Arc::new(CpuBackend::<T>::new())),
            });
        }

        #[cfg(feature = "mumps")]
        if resource == "/solver/mumps" {
            return Ok(Self {
                inner: reed_core::Reed::from_backend(Arc::new(CpuBackend::<T>::new())),
            });
        }

        #[cfg(feature = "mkl")]
        if resource == "/solver/mkl" {
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

fn resolve_backend_request(
    requested: Option<ReedBackendRequest>,
    caps: ReedBackendCapabilities,
) -> ReedBackendSelectionReport {
    match requested {
        None | Some(ReedBackendRequest::NativeLinger) => ReedBackendSelectionReport {
            requested,
            effective_resource: "/cpu/self".to_string(),
            capabilities: caps,
            note: "No explicit external backend requested; using native linger/cpu path in reed.".to_string(),
        },
        Some(ReedBackendRequest::GpuWgpu) => {
            if caps.wasm_target {
                ReedBackendSelectionReport {
                    requested,
                    effective_resource: "/cpu/self".to_string(),
                    capabilities: caps,
                    note: "Requested gpu/wgpu on wasm32 target; using deterministic fallback to native linger/cpu path.".to_string(),
                }
            } else if caps.gpu_wgpu {
                ReedBackendSelectionReport {
                    requested,
                    effective_resource: "/gpu/wgpu".to_string(),
                    capabilities: caps,
                    note: "Requested gpu/wgpu and feature is enabled; using reed wgpu backend.".to_string(),
                }
            } else {
                ReedBackendSelectionReport {
                    requested,
                    effective_resource: "/cpu/self".to_string(),
                    capabilities: caps,
                    note: "Requested gpu/wgpu but feature wgpu-backend is disabled; using deterministic fallback to native linger/cpu path.".to_string(),
                }
            }
        }
        Some(ReedBackendRequest::ExternalSolver(backend)) => {
            let (name, enabled, resource) = match backend {
                ExternalSolverBackend::HypreRs => ("hypre-rs", caps.hypre_rs, "/solver/hypre-rs"),
                ExternalSolverBackend::PetscRs => ("petsc-rs", caps.petsc_rs, "/solver/petsc-rs"),
                ExternalSolverBackend::Mumps => ("mumps", caps.mumps, "/solver/mumps"),
                ExternalSolverBackend::Mkl => ("mkl", caps.mkl, "/solver/mkl"),
            };

            if caps.wasm_target {
                ReedBackendSelectionReport {
                    requested,
                    effective_resource: "/cpu/self".to_string(),
                    capabilities: caps,
                    note: format!(
                        "Requested {} on wasm32 target; external solver backends are unavailable and deterministically fall back to native linger/cpu path.",
                        name
                    ),
                }
            } else if enabled {
                ReedBackendSelectionReport {
                    requested,
                    effective_resource: resource.to_string(),
                    capabilities: caps,
                    note: format!(
                        "Requested {}. Capability is enabled; reed uses {} as an executable placeholder route currently backed by CPU adapter until dedicated solver wiring lands.",
                        name, resource
                    ),
                }
            } else {
                ReedBackendSelectionReport {
                    requested,
                    effective_resource: "/cpu/self".to_string(),
                    capabilities: caps,
                    note: format!(
                        "Requested {}, but capability is disabled; using deterministic fallback to native linger/cpu path.",
                        name
                    ),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_report_defaults_to_native_cpu() {
        let rep = Reed::<f64>::backend_selection_report(None);
        assert_eq!(rep.effective_resource, "/cpu/self");
    }

    #[test]
    fn backend_report_gpu_without_feature_falls_back() {
        let rep = Reed::<f64>::backend_selection_report(Some(ReedBackendRequest::GpuWgpu));
        if !rep.capabilities.gpu_wgpu || rep.capabilities.wasm_target {
            assert_eq!(rep.effective_resource, "/cpu/self");
            assert!(rep.note.contains("fallback"));
        }
    }

    #[test]
    fn backend_report_mkl_defaults_to_fallback_until_wiring() {
        let rep = Reed::<f64>::backend_selection_report(Some(ReedBackendRequest::ExternalSolver(
            ExternalSolverBackend::Mkl,
        )));
        if rep.capabilities.mkl && !rep.capabilities.wasm_target {
            assert_eq!(rep.effective_resource, "/solver/mkl");
        } else {
            assert_eq!(rep.effective_resource, "/cpu/self");
        }
        assert!(rep.note.contains("mkl"));
    }

    #[test]
    fn parse_backend_request_supports_canonical_paths() {
        assert_eq!(
            Reed::<f64>::parse_backend_request("/native/linger"),
            Some(ReedBackendRequest::NativeLinger)
        );
        assert_eq!(
            Reed::<f64>::parse_backend_request("/solver/petsc-rs"),
            Some(ReedBackendRequest::ExternalSolver(ExternalSolverBackend::PetscRs))
        );
        assert_eq!(
            Reed::<f64>::parse_backend_request("/solver/petsc-ffi"),
            Some(ReedBackendRequest::ExternalSolver(ExternalSolverBackend::PetscRs))
        );
    }
}
