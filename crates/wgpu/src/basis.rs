use std::{any::TypeId, sync::Arc};

use num_traits::NumCast;
use reed_core::{
    BasisTrait,
    enums::{EvalMode, QuadMode},
    error::ReedResult,
    scalar::Scalar,
    ReedError,
};
use reed_cpu::basis_lagrange::LagrangeBasis;
use wgpu::util::DeviceExt;

use crate::runtime::GpuRuntime;

pub struct WgpuBasis<T: Scalar> {
    cpu_fallback: LagrangeBasis<T>,
    runtime: Option<Arc<GpuRuntime>>,
    interp_matrix_f32: Option<Vec<f32>>,
}

impl<T: Scalar> WgpuBasis<T> {
    pub fn new(
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: QuadMode,
        runtime: Option<Arc<GpuRuntime>>,
    ) -> ReedResult<Self> {
        let cpu_fallback = LagrangeBasis::<T>::new(dim, ncomp, p, q, qmode)?;
        let interp_matrix_f32 = if TypeId::of::<T>() == TypeId::of::<f32>() && runtime.is_some() {
            Some(build_interp_matrix_f32(dim, p, q, qmode)?)
        } else {
            None
        };
        Ok(Self {
            cpu_fallback,
            runtime,
            interp_matrix_f32,
        })
    }

    fn supports_f32_gpu() -> bool {
        TypeId::of::<T>() == TypeId::of::<f32>()
    }

    fn try_apply_interp_gpu(
        &self,
        num_elem: usize,
        transpose: bool,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<bool> {
        if transpose || !Self::supports_f32_gpu() {
            return Ok(false);
        }
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        let Some(interp) = &self.interp_matrix_f32 else {
            return Ok(false);
        };

        let num_dof = self.cpu_fallback.num_dof();
        let num_qpoints = self.cpu_fallback.num_qpoints();
        let ncomp = self.cpu_fallback.num_comp();
        let in_size = num_elem * num_dof * ncomp;
        let out_size = num_elem * num_qpoints * ncomp;
        if u.len() != in_size || v.len() != out_size {
            return Err(ReedError::Basis(format!(
                "interp apply size mismatch: input {}, expected {}; output {}, expected {}",
                u.len(),
                in_size,
                v.len(),
                out_size
            )));
        }

        let Some(u_f32) = u.iter().map(|x| NumCast::from(*x)).collect::<Option<Vec<f32>>>() else {
            return Ok(false);
        };
        let mut v_f32 = vec![0.0_f32; out_size];

        let mat_buffer =
            runtime
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("wgpu-basis-interp-mat"),
                    contents: bytemuck::cast_slice(interp),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-basis-interp-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-basis-interp-v"),
                contents: bytemuck::cast_slice(&v_f32),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params: [u32; 8] = [
            num_elem as u32,
            num_dof as u32,
            num_qpoints as u32,
            ncomp as u32,
            out_size as u32,
            0,
            0,
            0,
        ];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-basis-interp-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-basis-interp-bind"),
            layout: runtime.basis_interp_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buffer.as_entire_binding(),
                },
            ],
        });

        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-basis-interp-readback"),
            size: (out_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-basis-interp-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-basis-interp-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.basis_interp_pipeline());
            pass.set_bind_group(0, &bind, &[]);
            let groups = (out_size as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &v_buffer,
            0,
            &readback,
            0,
            (out_size * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        map_readback_f32(&runtime.device, &readback, &mut v_f32)?;
        for (dst, src) in v.iter_mut().zip(v_f32.iter()) {
            *dst = NumCast::from(*src)
                .ok_or_else(|| ReedError::Basis("f32->T conversion failed during readback".into()))?;
        }
        Ok(true)
    }
}

/// On WASM, wgpu::Device (inside GpuRuntime) is not Send+Sync, so the
/// BasisTrait impl is restricted to non-WASM targets only.
#[cfg(not(target_arch = "wasm32"))]
impl<T: Scalar> BasisTrait<T> for WgpuBasis<T> {
    fn dim(&self) -> usize {
        self.cpu_fallback.dim()
    }

    fn num_dof(&self) -> usize {
        self.cpu_fallback.num_dof()
    }

    fn num_qpoints(&self) -> usize {
        self.cpu_fallback.num_qpoints()
    }

    fn num_comp(&self) -> usize {
        self.cpu_fallback.num_comp()
    }

    fn apply(
        &self,
        num_elem: usize,
        transpose: bool,
        eval_mode: EvalMode,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<()> {
        if matches!(eval_mode, EvalMode::Interp)
            && self.try_apply_interp_gpu(num_elem, transpose, u, v)?
        {
            return Ok(());
        }
        self.cpu_fallback.apply(num_elem, transpose, eval_mode, u, v)
    }

    fn q_weights(&self) -> &[T] {
        self.cpu_fallback.q_weights()
    }

    fn q_ref(&self) -> &[T] {
        self.cpu_fallback.q_ref()
    }
}

fn build_interp_matrix_f32(dim: usize, p: usize, q: usize, qmode: QuadMode) -> ReedResult<Vec<f32>> {
    let probe = LagrangeBasis::<f32>::new(dim, 1, p, q, qmode)?;
    let num_dof = probe.num_dof();
    let num_qpoints = probe.num_qpoints();

    let mut interp = vec![0.0_f32; num_qpoints * num_dof];
    for dof in 0..num_dof {
        let mut u = vec![0.0_f32; num_dof];
        u[dof] = 1.0;
        let mut v = vec![0.0_f32; num_qpoints];
        probe.apply(1, false, EvalMode::Interp, &u, &mut v)?;
        for qpt in 0..num_qpoints {
            interp[qpt * num_dof + dof] = v[qpt];
        }
    }
    Ok(interp)
}

fn map_readback_f32(
    device: &wgpu::Device,
    readback: &wgpu::Buffer,
    out: &mut [f32],
) -> ReedResult<()> {
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    let map_result = rx
        .recv()
        .map_err(|e| ReedError::Basis(format!("map recv error: {e}")))?;
    map_result.map_err(|e| ReedError::Basis(format!("map error: {e:?}")))?;

    let data = slice.get_mapped_range();
    let mapped: &[f32] = bytemuck::cast_slice(&data);
    if mapped.len() != out.len() {
        return Err(ReedError::Basis("basis readback length mismatch".into()));
    }
    out.copy_from_slice(mapped);
    drop(data);
    readback.unmap();
    Ok(())
}
