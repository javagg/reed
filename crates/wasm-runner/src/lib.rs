use js_sys::Date;
use reed::{FieldVector, OperatorTrait, QuadMode, Reed};
use serde::Serialize;
use serde::Deserialize;

use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm-gpu")]
use reed_wgpu::WgpuBackend;

#[cfg(feature = "wasm-gpu")]
use std::sync::Arc;


#[cfg(feature = "wasm-gpu")]
use std::thread_local;

#[cfg(feature = "wasm-gpu")]
thread_local! {
    /// Stores the initialized wgpu backend (Arc so it can be cloned into closures).
    static WGPU_BACKEND: std::cell::RefCell<Option<Arc<WgpuBackend<f64>>>> =
        std::cell::RefCell::new(None);
}

/// Newtype wrapper so we can implement reed_core::Backend in this crate (orphan rule).
#[cfg(feature = "wasm-gpu")]
pub struct WgpuBackendWrap(pub Arc<WgpuBackend<f64>>);

#[cfg(feature = "wasm-gpu")]
impl reed_core::Backend<f64> for WgpuBackendWrap {
    fn resource_name(&self) -> &str {
        reed_core::Backend::<f64>::resource_name(&*self.0)
    }

    fn create_vector(&self, size: usize) -> reed_core::ReedResult<Box<dyn reed_core::VectorTrait<f64>>> {
        reed_core::Backend::<f64>::create_vector(&*self.0, size)
    }

    fn create_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i32],
    ) -> reed_core::ReedResult<Box<dyn reed_core::ElemRestrictionTrait<f64>>> {
        reed_core::Backend::<f64>::create_elem_restriction(&*self.0, nelem, elemsize, ncomp, compstride, lsize, offsets)
    }

    fn create_strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
    ) -> reed_core::ReedResult<Box<dyn reed_core::ElemRestrictionTrait<f64>>> {
        reed_core::Backend::<f64>::create_strided_elem_restriction(&*self.0, nelem, elemsize, ncomp, lsize, strides)
    }

    fn create_basis_tensor_h1_lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: reed_core::enums::QuadMode,
    ) -> reed_core::ReedResult<Box<dyn reed_core::BasisTrait<f64>>> {
        reed_core::Backend::<f64>::create_basis_tensor_h1_lagrange(&*self.0, dim, ncomp, p, q, qmode)
    }
}

#[cfg(feature = "wasm-gpu")]
/// Initialize the wgpu backend asynchronously.
/// Returns a Promise that resolves to the adapter name, or rejects on error.
#[wasm_bindgen]
pub async fn init_wgpu() -> Result<String, JsValue> {
    let backend = WgpuBackend::<f64>::new_async().await;
    let name = backend
        .adapter_name()
        .map(String::from)
        .unwrap_or_else(|| "unknown".to_string());
    let _ = backend.is_gpu_available();

    let shared: Arc<WgpuBackend<f64>> = Arc::new(backend);
    WGPU_BACKEND.with(|cell| {
        *cell.borrow_mut() = Some(shared);
    });

    Ok(name)
}

/// WASM-side check for wgpu availability.
#[wasm_bindgen]
pub fn wgpu_is_available() -> bool {
    #[cfg(feature = "wasm-gpu")]
    {
        WGPU_BACKEND.with(|cell| cell.borrow().is_some())
    }
    #[cfg(not(feature = "wasm-gpu"))]
    {
        false
    }
}

#[derive(Serialize)]
struct RunResult {
    example: String,
    backend: String,
    dim: usize,
    nelem: usize,
    p: usize,
    q: usize,
    logs: Vec<String>,
    value: f64,
    expected: Option<f64>,
    error: Option<f64>,
    duration_ms: f64,
}

#[derive(Deserialize)]
struct RunExampleArgs {
    #[serde(rename = "backend")]
    backend: String,
    example: String,
    dim: usize,
    nelem: usize,
    p: usize,
    q: usize,
}

/// Entry point called by the JS worker.
/// `args` is a plain JS object decoded from JSON.
#[wasm_bindgen]
pub fn run_example(args: JsValue) -> Result<JsValue, JsValue> {
    let parsed: RunExampleArgs = serde_wasm_bindgen::from_value(args)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let backend_str = parsed.backend.as_str();
    let mut logs = vec![
        format!("Initialize runner for {}", parsed.example),
        format!(
            "Backend: {}, Build: {}",
            backend_str,
            if cfg!(feature = "wasm-gpu") {
                "wasm-gpu"
            } else {
                "wasm-cpu"
            }
        ),
        format!(
            "Build operators for dim={}, nelem={}, p={}, q={}",
            parsed.dim, parsed.nelem, parsed.p, parsed.q
        ),
    ];

    let reed: Reed<f64> = match backend_str {
        "wasm-cpu" => {
            logs.push("Using CPU backend".to_string());
            Reed::init("/cpu/self")
                .map_err(|e| JsValue::from_str(&format!("failed to init CPU backend: {e}")))?
        }
        "wasm-gpu" => {
            logs.push("Using wgpu backend".to_string());
            #[cfg(feature = "wasm-gpu")]
            {
                let backend = WGPU_BACKEND
                    .with(|cell| cell.borrow().clone())
                    .ok_or_else(|| JsValue::from_str("wgpu not initialized; call init_wgpu() first"))?;
                Reed::from_backend(Arc::new(WgpuBackendWrap(backend)) as Arc<dyn reed_core::Backend<f64>>)
            }
            #[cfg(not(feature = "wasm-gpu"))]
            {
                return Err(JsValue::from_str(
                    "wgpu backend not compiled in; rebuild with --features wasm-gpu",
                ));
            }
        }
        _ => {
            return Err(JsValue::from_str(&format!("unknown backend: {backend_str}")));
        }
    };

    logs.push("Apply operator and collect output".to_string());
    let t0 = Date::now();

    let (value, expected, error) = match parsed.example.as_str() {
        "ex1_volume" => run_ex1(&reed, parsed.dim, parsed.nelem, parsed.p, parsed.q, &mut logs),
        "ex2_surface" => run_ex2(&reed, parsed.dim, parsed.nelem, parsed.p, parsed.q, &mut logs),
        "ex3_volume_combined" => {
            run_ex3(&reed, parsed.dim, parsed.nelem, parsed.p, parsed.q, &mut logs)
        }
        "poisson" => run_poisson(&reed, parsed.dim, parsed.nelem, parsed.p, parsed.q, &mut logs),
        _ => Err(format!("unknown example: {}", parsed.example)),
    }
    .map_err(|e| JsValue::from_str(&e))?;

    let t1 = Date::now();

    let result = RunResult {
        example: parsed.example,
        backend: backend_str.to_string(),
        dim: parsed.dim,
        nelem: parsed.nelem,
        p: parsed.p,
        q: parsed.q,
        logs,
        value,
        expected: if expected == 0.0 { None } else { Some(expected) },
        error: if error == 0.0 { None } else { Some(error) },
        duration_ms: t1 - t0,
    };

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

// ─── Helper: convert Reed<f64> result to (value, expected, error) ────────────

fn build_offsets_1d(nelem_1d: usize, p: usize) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(nelem_1d * p);
    for e in 0..nelem_1d {
        let start = e * (p - 1);
        for j in 0..p {
            offsets.push((start + j) as i32);
        }
    }
    offsets
}

fn build_offsets_2d(nelem_1d: usize, p: usize, ndofs_1d: usize) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(nelem_1d * nelem_1d * p * p);
    for ey in 0..nelem_1d {
        for ex in 0..nelem_1d {
            let sy = ey * (p - 1);
            let sx = ex * (p - 1);
            for jy in 0..p {
                for jx in 0..p {
                    let gi = (sy + jy) * ndofs_1d + (sx + jx);
                    offsets.push(gi as i32);
                }
            }
        }
    }
    offsets
}

fn build_offsets_3d(nelem_1d: usize, p: usize, ndofs_1d: usize) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(nelem_1d * nelem_1d * nelem_1d * p * p * p);
    for ez in 0..nelem_1d {
        for ey in 0..nelem_1d {
            for ex in 0..nelem_1d {
                let sz = ez * (p - 1);
                let sy = ey * (p - 1);
                let sx = ex * (p - 1);
                for jz in 0..p {
                    for jy in 0..p {
                        for jx in 0..p {
                            let gi = ((sz + jz) * ndofs_1d + (sy + jy)) * ndofs_1d + (sx + jx);
                            offsets.push(gi as i32);
                        }
                    }
                }
            }
        }
    }
    offsets
}

fn build_coords_components(dim: usize, ndofs_1d: usize) -> Vec<Vec<f64>> {
    let ndofs = ndofs_1d.pow(dim as u32);
    let mut comps = (0..dim).map(|_| vec![0.0_f64; ndofs]).collect::<Vec<_>>();

    if dim == 1 {
        for i in 0..ndofs {
            comps[0][i] = -1.0 + 2.0 * i as f64 / (ndofs - 1) as f64;
        }
    } else if dim == 2 {
        for iy in 0..ndofs_1d {
            for ix in 0..ndofs_1d {
                let i = iy * ndofs_1d + ix;
                comps[0][i] = -1.0 + 2.0 * ix as f64 / (ndofs_1d - 1) as f64;
                comps[1][i] = -1.0 + 2.0 * iy as f64 / (ndofs_1d - 1) as f64;
            }
        }
    } else {
        for iz in 0..ndofs_1d {
            for iy in 0..ndofs_1d {
                for ix in 0..ndofs_1d {
                    let i = (iz * ndofs_1d + iy) * ndofs_1d + ix;
                    comps[0][i] = -1.0 + 2.0 * ix as f64 / (ndofs_1d - 1) as f64;
                    comps[1][i] = -1.0 + 2.0 * iy as f64 / (ndofs_1d - 1) as f64;
                    comps[2][i] = -1.0 + 2.0 * iz as f64 / (ndofs_1d - 1) as f64;
                }
            }
        }
    }
    comps
}

fn build_poisson_qdata_1d(node_coords: &[f64], qweights: &[f64], nelem: usize, p: usize) -> Vec<f64> {
    let mut qdata = Vec::with_capacity(nelem * qweights.len());
    for e in 0..nelem {
        let i0 = e * (p - 1);
        let i1 = i0 + (p - 1);
        let jacobian = 0.5 * (node_coords[i1] - node_coords[i0]);
        for &w in qweights {
            qdata.push(w / jacobian);
        }
    }
    qdata
}

fn setup_common(
    _reed: &Reed<f64>,
    dim: usize,
    nelem_1d: usize,
    p: usize,
    q: usize,
) -> (usize, usize, usize, usize, Vec<i32>, Vec<Vec<f64>>) {
    let ndofs_1d = nelem_1d * (p - 1) + 1;
    let ndofs = ndofs_1d.pow(dim as u32);
    let nelem = nelem_1d.pow(dim as u32);
    let elemsize = p.pow(dim as u32);
    let qpts_per_elem = q.pow(dim as u32);

    let offsets = match dim {
        1 => build_offsets_1d(nelem_1d, p),
        2 => build_offsets_2d(nelem_1d, p, ndofs_1d),
        3 => build_offsets_3d(nelem_1d, p, ndofs_1d),
        _ => unreachable!(),
    };

    let comps = build_coords_components(dim, ndofs_1d);

    (ndofs, nelem, elemsize, qpts_per_elem, offsets, comps)
}

fn run_ex1(
    reed: &Reed<f64>,
    dim: usize,
    nelem_1d: usize,
    p: usize,
    q: usize,
    logs: &mut Vec<String>,
) -> Result<(f64, f64, f64), String> {
    let (ndofs, nelem, elemsize, qpts_per_elem, offsets, comps) =
        setup_common(reed, dim, nelem_1d, p, q);

    let x_coords = comps.concat();
    let x = reed
        .vector_from_slice(&x_coords)
        .map_err(|e| e.to_string())?;

    let r_x = reed
        .elem_restriction(nelem, elemsize, dim, ndofs, dim * ndofs, &offsets)
        .map_err(|e| e.to_string())?;
    let r_u = reed
        .elem_restriction(nelem, elemsize, 1, 1, ndofs, &offsets)
        .map_err(|e| e.to_string())?;
    let b_x = reed
        .basis_tensor_h1_lagrange(dim, dim, p, q, QuadMode::Gauss)
        .map_err(|e| e.to_string())?;
    let b_u = reed
        .basis_tensor_h1_lagrange(dim, 1, p, q, QuadMode::Gauss)
        .map_err(|e| e.to_string())?;
    let r_q = reed
        .strided_elem_restriction(
            nelem,
            qpts_per_elem,
            1,
            nelem * qpts_per_elem,
            [1, qpts_per_elem as i32, qpts_per_elem as i32],
        )
        .map_err(|e| e.to_string())?;

    let mut qdata = reed
        .vector(nelem * qpts_per_elem)
        .map_err(|e| e.to_string())?;
    qdata.set_value(0.0).map_err(|e| e.to_string())?;

    let qf_build = match dim {
        1 => "Mass1DBuild",
        2 => "Mass2DBuild",
        _ => "Mass3DBuild",
    };

    let op_build = reed
        .operator_builder()
        .qfunction(
            reed.q_function_by_name(qf_build)
                .map_err(|e| e.to_string())?,
        )
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()
        .map_err(|e| e.to_string())?;
    op_build.apply(&*x, &mut *qdata).map_err(|e| e.to_string())?;

    let u = reed
        .vector_from_slice(&vec![1.0_f64; ndofs])
        .map_err(|e| e.to_string())?;
    let mut v = reed.vector(ndofs).map_err(|e| e.to_string())?;
    v.set_value(0.0).map_err(|e| e.to_string())?;

    let op_mass = reed
        .operator_builder()
        .qfunction(
            reed.q_function_by_name("MassApply")
                .map_err(|e| e.to_string())?,
        )
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .map_err(|e| e.to_string())?;
    op_mass.apply(&*u, &mut *v).map_err(|e| e.to_string())?;

    let mut values = vec![0.0; ndofs];
    v.copy_to_slice(&mut values).map_err(|e| e.to_string())?;
    let computed = values.iter().sum::<f64>();
    let exact = if dim == 1 {
        2.0
    } else if dim == 2 {
        4.0
    } else {
        8.0
    };
    logs.push(format!("backend: {}", reed.resource()));
    Ok((computed, exact, (computed - exact).abs()))
}

fn run_ex2(
    reed: &Reed<f64>,
    dim: usize,
    nelem_1d: usize,
    p: usize,
    q: usize,
    logs: &mut Vec<String>,
) -> Result<(f64, f64, f64), String> {
    let (ndofs, nelem, elemsize, qpts_per_elem, offsets, comps) =
        setup_common(reed, dim, nelem_1d, p, q);

    if dim == 1 {
        let node_coords = &comps[0];
        let r_u = reed
            .elem_restriction(nelem, p, 1, 1, ndofs, &offsets)
            .map_err(|e| e.to_string())?;
        let r_q = reed
            .strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])
            .map_err(|e| e.to_string())?;
        let b_u = reed
            .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
            .map_err(|e| e.to_string())?;
        let qdata_vals =
            build_poisson_qdata_1d(node_coords, b_u.q_weights(), nelem, p);
        let qdata = reed
            .vector_from_slice(&qdata_vals)
            .map_err(|e| e.to_string())?;

        let u = reed
            .vector_from_slice(node_coords)
            .map_err(|e| e.to_string())?;
        let mut v = reed.vector(ndofs).map_err(|e| e.to_string())?;
        v.set_value(0.0).map_err(|e| e.to_string())?;

        let op = reed
            .operator_builder()
            .qfunction(
                reed.q_function_by_name("Poisson1DApply")
                    .map_err(|e| e.to_string())?,
            )
            .field("du", Some(&*r_u), Some(&*b_u), FieldVector::Active)
            .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
            .field("dv", Some(&*r_u), Some(&*b_u), FieldVector::Active)
            .build()
            .map_err(|e| e.to_string())?;
        op.apply(&*u, &mut *v).map_err(|e| e.to_string())?;

        let mut values = vec![0.0; ndofs];
        v.copy_to_slice(&mut values).map_err(|e| e.to_string())?;
        let computed = values.iter().map(|x| x.abs()).sum::<f64>();
        logs.push(format!("backend: {}", reed.resource()));
        return Ok((computed, 2.0, (computed - 2.0).abs()));
    }

    let x_coords = comps.concat();
    let x = reed
        .vector_from_slice(&x_coords)
        .map_err(|e| e.to_string())?;
    let r_x = reed
        .elem_restriction(nelem, elemsize, dim, ndofs, dim * ndofs, &offsets)
        .map_err(|e| e.to_string())?;
    let r_u = reed
        .elem_restriction(nelem, elemsize, 1, 1, ndofs, &offsets)
        .map_err(|e| e.to_string())?;
    let b_x = reed
        .basis_tensor_h1_lagrange(dim, dim, p, q, QuadMode::Gauss)
        .map_err(|e| e.to_string())?;
    let b_u = reed
        .basis_tensor_h1_lagrange(dim, 1, p, q, QuadMode::Gauss)
        .map_err(|e| e.to_string())?;

    let qdata_comp = dim * dim;
    let r_q = reed
        .strided_elem_restriction(
            nelem,
            qpts_per_elem,
            qdata_comp,
            nelem * qpts_per_elem * qdata_comp,
            [1, qpts_per_elem as i32, (qpts_per_elem * qdata_comp) as i32],
        )
        .map_err(|e| e.to_string())?;
    let mut qdata = reed
        .vector(nelem * qpts_per_elem * qdata_comp)
        .map_err(|e| e.to_string())?;
    qdata.set_value(0.0).map_err(|e| e.to_string())?;

    let qf_build = if dim == 2 {
        "Poisson2DBuild"
    } else {
        "Poisson3DBuild"
    };
    let op_build = reed
        .operator_builder()
        .qfunction(
            reed.q_function_by_name(qf_build)
                .map_err(|e| e.to_string())?,
        )
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()
        .map_err(|e| e.to_string())?;
    op_build.apply(&*x, &mut *qdata).map_err(|e| e.to_string())?;

    let mut u_vals = vec![0.0_f64; ndofs];
    for c in comps.iter().take(dim) {
        for i in 0..ndofs {
            u_vals[i] += c[i];
        }
    }
    let u = reed
        .vector_from_slice(&u_vals)
        .map_err(|e| e.to_string())?;
    let mut v = reed.vector(ndofs).map_err(|e| e.to_string())?;
    v.set_value(0.0).map_err(|e| e.to_string())?;

    let qf_apply = if dim == 2 {
        "Poisson2DApply"
    } else {
        "Poisson3DApply"
    };
    let op = reed
        .operator_builder()
        .qfunction(
            reed.q_function_by_name(qf_apply)
                .map_err(|e| e.to_string())?,
        )
        .field("du", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("dv", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .map_err(|e| e.to_string())?;
    op.apply(&*u, &mut *v).map_err(|e| e.to_string())?;

    let mut values = vec![0.0; ndofs];
    v.copy_to_slice(&mut values).map_err(|e| e.to_string())?;
    let computed = values.iter().map(|x| x.abs()).sum::<f64>();
    let exact = if dim == 2 { 8.0 } else { 24.0 };
    logs.push(format!("backend: {}", reed.resource()));
    Ok((computed, exact, (computed - exact).abs()))
}

fn run_ex3(
    reed: &Reed<f64>,
    dim: usize,
    nelem_1d: usize,
    p: usize,
    q: usize,
    logs: &mut Vec<String>,
) -> Result<(f64, f64, f64), String> {
    let (ndofs, nelem, elemsize, qpts_per_elem, offsets, comps) =
        setup_common(reed, dim, nelem_1d, p, q);

    let r_u = reed
        .elem_restriction(nelem, elemsize, 1, 1, ndofs, &offsets)
        .map_err(|e| e.to_string())?;
    let b_u = reed
        .basis_tensor_h1_lagrange(dim, 1, p, q, QuadMode::Gauss)
        .map_err(|e| e.to_string())?;

    let qf_mass_build = match dim {
        1 => "Mass1DBuild",
        2 => "Mass2DBuild",
        _ => "Mass3DBuild",
    };

    let x_coords = comps.concat();
    let x = reed
        .vector_from_slice(&x_coords)
        .map_err(|e| e.to_string())?;
    let r_x = reed
        .elem_restriction(nelem, elemsize, dim, ndofs, dim * ndofs, &offsets)
        .map_err(|e| e.to_string())?;
    let b_x = reed
        .basis_tensor_h1_lagrange(dim, dim, p, q, QuadMode::Gauss)
        .map_err(|e| e.to_string())?;

    let r_q_mass = reed
        .strided_elem_restriction(
            nelem,
            qpts_per_elem,
            1,
            nelem * qpts_per_elem,
            [1, qpts_per_elem as i32, qpts_per_elem as i32],
        )
        .map_err(|e| e.to_string())?;
    let mut qdata_mass = reed
        .vector(nelem * qpts_per_elem)
        .map_err(|e| e.to_string())?;
    qdata_mass.set_value(0.0).map_err(|e| e.to_string())?;

    let op_build_mass = reed
        .operator_builder()
        .qfunction(
            reed.q_function_by_name(qf_mass_build)
                .map_err(|e| e.to_string())?,
        )
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q_mass), None, FieldVector::Active)
        .build()
        .map_err(|e| e.to_string())?;
    op_build_mass
        .apply(&*x, &mut *qdata_mass)
        .map_err(|e| e.to_string())?;

    let (r_q_poisson, qdata_poisson) = if dim == 1 {
        let qdata_vals =
            build_poisson_qdata_1d(&comps[0], b_u.q_weights(), nelem, p);
        let qdata_vec = reed
            .vector_from_slice(&qdata_vals)
            .map_err(|e| e.to_string())?;
        let r_q = reed
            .strided_elem_restriction(
                nelem,
                qpts_per_elem,
                1,
                nelem * qpts_per_elem,
                [1, qpts_per_elem as i32, qpts_per_elem as i32],
            )
            .map_err(|e| e.to_string())?;
        (r_q, qdata_vec)
    } else {
        let qdata_comp = dim * dim;
        let r_q = reed
            .strided_elem_restriction(
                nelem,
                qpts_per_elem,
                qdata_comp,
                nelem * qpts_per_elem * qdata_comp,
                [1, qpts_per_elem as i32, (qpts_per_elem * qdata_comp) as i32],
            )
            .map_err(|e| e.to_string())?;
        let mut qdata_vec = reed
            .vector(nelem * qpts_per_elem * qdata_comp)
            .map_err(|e| e.to_string())?;
        qdata_vec.set_value(0.0).map_err(|e| e.to_string())?;

        let qf_poisson_build = if dim == 2 {
            "Poisson2DBuild"
        } else {
            "Poisson3DBuild"
        };
        let op_build_poisson = reed
            .operator_builder()
            .qfunction(
                reed.q_function_by_name(qf_poisson_build)
                    .map_err(|e| e.to_string())?,
            )
            .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
            .field("weights", None, Some(&*b_x), FieldVector::None)
            .field("qdata", Some(&*r_q), None, FieldVector::Active)
            .build()
            .map_err(|e| e.to_string())?;
        op_build_poisson
            .apply(&*x, &mut *qdata_vec)
            .map_err(|e| e.to_string())?;
        (r_q, qdata_vec)
    };

    let op_mass = reed
        .operator_builder()
        .qfunction(
            reed.q_function_by_name("MassApply")
                .map_err(|e| e.to_string())?,
        )
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q_mass), None, FieldVector::Passive(&*qdata_mass))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .map_err(|e| e.to_string())?;

    let qf_poisson_apply = if dim == 1 {
        "Poisson1DApply"
    } else if dim == 2 {
        "Poisson2DApply"
    } else {
        "Poisson3DApply"
    };
    let op_diff = reed
        .operator_builder()
        .qfunction(
            reed.q_function_by_name(qf_poisson_apply)
                .map_err(|e| e.to_string())?,
        )
        .field("du", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q_poisson), None, FieldVector::Passive(&*qdata_poisson))
        .field("dv", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .map_err(|e| e.to_string())?;

    let u = reed
        .vector_from_slice(&vec![1.0_f64; ndofs])
        .map_err(|e| e.to_string())?;
    let mut v = reed.vector(ndofs).map_err(|e| e.to_string())?;
    v.set_value(0.0).map_err(|e| e.to_string())?;
    op_mass.apply(&*u, &mut *v).map_err(|e| e.to_string())?;
    op_diff.apply_add(&*u, &mut *v).map_err(|e| e.to_string())?;

    let mut values = vec![0.0; ndofs];
    v.copy_to_slice(&mut values).map_err(|e| e.to_string())?;
    let computed = values.iter().sum::<f64>();
    let exact = if dim == 1 {
        2.0
    } else if dim == 2 {
        4.0
    } else {
        8.0
    };
    logs.push(format!("backend: {}", reed.resource()));
    Ok((computed, exact, (computed - exact).abs()))
}

fn run_poisson(
    reed: &Reed<f64>,
    dim: usize,
    nelem_1d: usize,
    p: usize,
    q: usize,
    logs: &mut Vec<String>,
) -> Result<(f64, f64, f64), String> {
    let (ndofs, nelem, elemsize, qpts_per_elem, offsets, comps) =
        setup_common(reed, dim, nelem_1d, p, q);

    if dim == 1 {
        let node_coords = &comps[0];
        let r_u = reed
            .elem_restriction(nelem, p, 1, 1, ndofs, &offsets)
            .map_err(|e| e.to_string())?;
        let r_q = reed
            .strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])
            .map_err(|e| e.to_string())?;
        let b_u = reed
            .basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)
            .map_err(|e| e.to_string())?;

        let qdata_vals =
            build_poisson_qdata_1d(node_coords, b_u.q_weights(), nelem, p);
        let qdata = reed
            .vector_from_slice(&qdata_vals)
            .map_err(|e| e.to_string())?;
        let u = reed
            .vector_from_slice(node_coords)
            .map_err(|e| e.to_string())?;
        let mut v = reed.vector(ndofs).map_err(|e| e.to_string())?;
        v.set_value(0.0).map_err(|e| e.to_string())?;

        let op = reed
            .operator_builder()
            .qfunction(
                reed.q_function_by_name("Poisson1DApply")
                    .map_err(|e| e.to_string())?,
            )
            .field("du", Some(&*r_u), Some(&*b_u), FieldVector::Active)
            .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
            .field("dv", Some(&*r_u), Some(&*b_u), FieldVector::Active)
            .build()
            .map_err(|e| e.to_string())?;
        op.apply(&*u, &mut *v).map_err(|e| e.to_string())?;

        let mut values = vec![0.0; ndofs];
        v.copy_to_slice(&mut values).map_err(|e| e.to_string())?;
        let norm1 = values.iter().map(|x| x.abs()).sum::<f64>();
        logs.push(format!("backend: {}", reed.resource()));
        return Ok((norm1, 0.0, 0.0));
    }

    let x_coords = comps.concat();
    let x = reed
        .vector_from_slice(&x_coords)
        .map_err(|e| e.to_string())?;
    let r_x = reed
        .elem_restriction(nelem, elemsize, dim, ndofs, dim * ndofs, &offsets)
        .map_err(|e| e.to_string())?;
    let r_u = reed
        .elem_restriction(nelem, elemsize, 1, 1, ndofs, &offsets)
        .map_err(|e| e.to_string())?;
    let b_x = reed
        .basis_tensor_h1_lagrange(dim, dim, p, q, QuadMode::Gauss)
        .map_err(|e| e.to_string())?;
    let b_u = reed
        .basis_tensor_h1_lagrange(dim, 1, p, q, QuadMode::Gauss)
        .map_err(|e| e.to_string())?;

    let qdata_comp = dim * dim;
    let r_q = reed
        .strided_elem_restriction(
            nelem,
            qpts_per_elem,
            qdata_comp,
            nelem * qpts_per_elem * qdata_comp,
            [1, qpts_per_elem as i32, (qpts_per_elem * qdata_comp) as i32],
        )
        .map_err(|e| e.to_string())?;
    let mut qdata = reed
        .vector(nelem * qpts_per_elem * qdata_comp)
        .map_err(|e| e.to_string())?;
    qdata.set_value(0.0).map_err(|e| e.to_string())?;

    let qf_build = if dim == 2 {
        "Poisson2DBuild"
    } else {
        "Poisson3DBuild"
    };
    let op_build = reed
        .operator_builder()
        .qfunction(
            reed.q_function_by_name(qf_build)
                .map_err(|e| e.to_string())?,
        )
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()
        .map_err(|e| e.to_string())?;
    op_build.apply(&*x, &mut *qdata).map_err(|e| e.to_string())?;

    let mut u_vals = vec![0.0_f64; ndofs];
    for c in comps.iter().take(dim) {
        for i in 0..ndofs {
            u_vals[i] += c[i];
        }
    }
    let u = reed
        .vector_from_slice(&u_vals)
        .map_err(|e| e.to_string())?;
    let mut v = reed.vector(ndofs).map_err(|e| e.to_string())?;
    v.set_value(0.0).map_err(|e| e.to_string())?;

    let qf_apply = if dim == 2 {
        "Poisson2DApply"
    } else {
        "Poisson3DApply"
    };
    let op = reed
        .operator_builder()
        .qfunction(
            reed.q_function_by_name(qf_apply)
                .map_err(|e| e.to_string())?,
        )
        .field("du", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("dv", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()
        .map_err(|e| e.to_string())?;
    op.apply(&*u, &mut *v).map_err(|e| e.to_string())?;

    let mut values = vec![0.0; ndofs];
    v.copy_to_slice(&mut values).map_err(|e| e.to_string())?;
    let norm1 = values.iter().map(|x| x.abs()).sum::<f64>();
    logs.push(format!("backend: {}", reed.resource()));
    Ok((norm1, 0.0, 0.0))
}
