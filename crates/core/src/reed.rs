use crate::{
    basis::BasisTrait,
    elem_restriction::ElemRestrictionTrait,
    enums::*,
    error::{ReedError, ReedResult},
    scalar::Scalar,
    vector::VectorTrait,
};
use std::sync::{Arc, Mutex};

/// 后端工厂 trait（各后端实现此 trait）
///
/// On WASM targets, the `Send + Sync` bounds are omitted because wgpu::Device
/// is not thread-safe in the browser's single-threaded environment.
#[cfg(not(target_arch = "wasm32"))]
pub trait Backend<T: Scalar>: Send + Sync {
    fn resource_name(&self) -> &str;

    fn create_vector(&self, size: usize) -> ReedResult<Box<dyn VectorTrait<T>>>;

    fn create_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i32],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>>;

    fn create_strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>>;

    fn create_basis_tensor_h1_lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;

    /// Create an H1 Lagrange basis on a simplex reference element.
    ///
    /// # Parameters
    /// * `topo`  — `ElemTopology::Triangle` or `ElemTopology::Tet`.
    /// * `poly`  — polynomial order (1 = P1, 2 = P2).
    /// * `ncomp` — number of field components.
    /// * `q`     — number of quadrature points (see `SimplexBasis` docs for
    ///             valid values per topology).
    fn create_basis_h1_simplex(
        &self,
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;
}

/// On WASM, wgpu::Device is not Send+Sync so neither is the Backend trait.
#[cfg(target_arch = "wasm32")]
pub trait Backend<T: Scalar> {
    fn resource_name(&self) -> &str;

    fn create_vector(&self, size: usize) -> ReedResult<Box<dyn VectorTrait<T>>>;

    fn create_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i32],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>>;

    fn create_strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>>;

    fn create_basis_tensor_h1_lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;

    fn create_basis_h1_simplex(
        &self,
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;
}

/// Reed 顶层库上下文
pub struct Reed<T: Scalar> {
    backend: Arc<Mutex<Arc<dyn Backend<T>>>>,
}

impl<T: Scalar> Reed<T> {
    /// 从已有后端创建（主要用于测试和库内部）
    pub fn from_backend(backend: Arc<dyn Backend<T>>) -> Self {
        Self {
            backend: Arc::new(Mutex::new(backend)),
        }
    }

    pub fn resource(&self) -> String {
        (**self.backend.lock().unwrap()).resource_name().to_owned()
    }

    // ── Vector 工厂 ──

    pub fn vector(&self, n: usize) -> ReedResult<Box<dyn VectorTrait<T>>> {
        (**self.backend.lock().unwrap()).create_vector(n)
    }

    pub fn vector_from_slice(&self, data: &[T]) -> ReedResult<Box<dyn VectorTrait<T>>> {
        let mut v = (**self.backend.lock().unwrap()).create_vector(data.len())?;
        v.copy_from_slice(data)?;
        Ok(v)
    }

    // ── ElemRestriction 工厂 ──

    pub fn elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i32],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        (**self.backend.lock().unwrap())
            .create_elem_restriction(nelem, elemsize, ncomp, compstride, lsize, offsets)
    }

    pub fn strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        (**self.backend.lock().unwrap())
            .create_strided_elem_restriction(nelem, elemsize, ncomp, lsize, strides)
    }

    /// Restriction with `elemsize = npoints_per_elem` (dofs indexed per quadrature point per element).
    ///
    /// Same implementation as [`Self::elem_restriction`]; aligns with libCEED
    /// `CeedElemRestrictionCreateAtPoints` naming. `offsets.len()` must be `nelem * npoints_per_elem`.
    pub fn elem_restriction_at_points(
        &self,
        nelem: usize,
        npoints_per_elem: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i32],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        let expected = nelem.checked_mul(npoints_per_elem).ok_or_else(|| {
            ReedError::InvalidArgument("elem_restriction_at_points: size overflow".into())
        })?;
        if offsets.len() != expected {
            return Err(ReedError::InvalidArgument(format!(
                "elem_restriction_at_points: offsets.len() {} != nelem * npoints_per_elem ({})",
                offsets.len(),
                expected
            )));
        }
        self.elem_restriction(nelem, npoints_per_elem, ncomp, compstride, lsize, offsets)
    }

    // ── Basis 工厂 ──

    pub fn basis_tensor_h1_lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        (**self.backend.lock().unwrap()).create_basis_tensor_h1_lagrange(dim, ncomp, p, q, qmode)
    }

    /// Create an H1 Lagrange basis on a simplex reference element.
    ///
    /// See [`Backend::create_basis_h1_simplex`] for parameter details.
    pub fn basis_h1_simplex(
        &self,
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        (**self.backend.lock().unwrap()).create_basis_h1_simplex(topo, poly, ncomp, q)
    }

    /// 获取后端引用
    pub fn backend(&self) -> &Arc<Mutex<Arc<dyn Backend<T>>>> {
        &self.backend
    }
}

/// 通过资源字符串初始化 Reed 上下文
///
/// 支持的资源：
/// - "/cpu/self" 或 "/cpu/self/ref" → CPU 后端
pub fn init<T: Scalar>(resource: &str) -> ReedResult<Reed<T>> {
    let _resource = resource;
    // 后端注册在 reed-cpu crate 中完成
    // 这里提供一个查找机制
    Err(ReedError::BackendNotSupported(format!(
        "No backend registered for resource '{}'. \
         Use Reed::from_backend() or enable the appropriate backend crate.",
        resource
    )))
}
