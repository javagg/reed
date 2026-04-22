//! Device-side QFunction design (Reed / WGSL / WGPU).
//!
//! libCEED models a QFunction as a pointwise kernel over quadrature points with packed I/O and
//! optional context bytes. Reed preserves that on CPU via [`reed_core::QFunctionTrait`]. This
//! module documents the **GPU path** and contains a **minimal runnable prototype** that is not
//! yet wired into [`reed_cpu::operator::CpuOperator`].
//!
//! ## 1. Packed I/O (align with `CpuOperator` staging)
//!
//! Each QFunction input/output slot is a **dense 1D buffer** in quadrature order, element-major:
//! index `e * nqp * C + q * C + c` where `C` is that slot’s `num_comp`. This matches the `Vec<T>`
//! layout passed to [`reed_core::QFunctionTrait::apply`] today. Passive fields (e.g. `qdata`) use
//! the same packing as **read-only** storage bindings.
//!
//! ## 2. `QFunctionContext`
//!
//! Host-owned bytes are copied each dispatch into either a **small uniform** block (typical
//! constants, few scalars) or a **read-only storage** buffer for larger tables. After the host
//! updates context, enqueue a `queue.write_buffer` (or mapped upload) before the operator’s
//! compute pass. [`reed_core::ClosureQFunction`] remains CPU-only unless a compiled-kernel
//! registration path is added later.
//!
//! ## 3. WGSL sources (incremental)
//!
//! - **Gallery**: hand-written (or `include!` template) WGSL per named kernel; parity-tested
//!   against CPU gallery.
//! - **Bind layout**: one group can hold uniform ctx + several `storage, read` inputs + several
//!   `storage, read_write` outputs; larger operators may split passes if binding limits bite.
//!
//! ## 4. Operator integration
//!
//! Restriction/basis still produce host `Vec` quadrature data in [`reed_cpu::operator::CpuOperator`].
//! A device QFunction **uploads those slices → SSBO**, runs compute, then **readbacks** into the
//! output `Vec` before restriction transpose / basis transpose on the host (or future GPU scatter).
//!
//! ## 5. Bring-up types
//!
//! - [`QFunctionPrototypeScaleF32`] — `out[i] = scale * in[i]` (no trait).
//! - [`MassApplyF32Wgpu`] — gallery-compatible scalar [`MassApply`](reed_cpu::MassApply); implements
//!   [`QFunctionTrait`] for `f32` and can be passed to [`reed_cpu::OperatorBuilder::qfunction`].
//! - [`Poisson1DApplyF32Wgpu`] — gallery [`Poisson1DApply`](reed_cpu::Poisson1DApply) (`dv = du *
//!   qdata`); uses the same [`GpuRuntime`] pointwise multiply pipeline as [`MassApplyF32Wgpu`].
//! - [`Poisson2DApplyF32Wgpu`] — gallery [`Poisson2DApply`](reed_cpu::Poisson2DApply); 2×2 block per
//!   point via [`GpuRuntime::qfunction_poisson2d_apply_pipeline`] (same four-slot bind layout).
//! - [`Poisson3DApplyF32Wgpu`] — gallery [`Poisson3DApply`](reed_cpu::Poisson3DApply); 3×3 block per
//!   point via [`GpuRuntime::qfunction_poisson3d_apply_pipeline`].
//! - [`IdentityF32Wgpu`] / [`ScaleF32Wgpu`] — gallery [`Identity`](reed_cpu::Identity) copy and
//!   [`Scale`](reed_cpu::Scale) on `f32` via [`GpuRuntime::qfunction_unary_layout`].
//! - [`Vector2MassApplyF32Wgpu`] — gallery [`Vector2MassApply`](reed_cpu::Vector2MassApply) (`u`,`v`
//!   have `2` components per quadrature point); uses [`GpuRuntime::qfunction_vector2_mass_apply_pipeline`]
//!   with the same bind layout as scalar pointwise multiply.
//! - [`Vector3MassApplyF32Wgpu`] — gallery [`Vector3MassApply`](reed_cpu::Vector3MassApply); uses
//!   [`GpuRuntime::qfunction_vector3_mass_apply_pipeline`] with the same bind layout.
//! - [`Vector2Poisson1DApplyF32Wgpu`] / [`Vector3Poisson1DApplyF32Wgpu`] — gallery
//!   [`Vector2Poisson1DApply`](reed_cpu::Vector2Poisson1DApply) /
//!   [`Vector3Poisson1DApply`](reed_cpu::Vector3Poisson1DApply); numerically the same per-point
//!   scaling as vector mass (reuse `vector2_mass_apply_f32` / `vector3_mass_apply_f32`).
//! - [`Vector2Poisson2DApplyF32Wgpu`] / [`Vector3Poisson2DApplyF32Wgpu`] — gallery
//!   [`Vector2Poisson2DApply`](reed_cpu::Vector2Poisson2DApply) /
//!   [`Vector3Poisson2DApply`](reed_cpu::Vector3Poisson2DApply); shared 2×2 stiffness per point
//!   (`vector2_poisson2d_apply_f32` / `vector3_poisson2d_apply_f32`).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use reed_core::{
    qfunction::QFunctionTrait, QFunctionContext, ReedError, ReedResult,
};
use wgpu::util::DeviceExt;

use crate::runtime::GpuRuntime;

const WGSL_SCALE_PROTO: &str = r#"
struct QfProtoParams {
    num_q: u32,
    _pad0: u32,
    scale: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> qp: QfProtoParams;
@group(0) @binding(1) var<storage, read> q_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_out: array<f32>;

@compute @workgroup_size(256)
fn qf_scale_proto(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    q_out[i] = qp.scale * q_in[i];
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QfProtoParamsHost {
    num_q: u32,
    _pad0: u32,
    scale: f32,
    _pad1: f32,
}

/// One-input one-output `f32` QFunction prototype: pointwise multiply by `scale`.
///
/// Intended for bring-up only; real gallery kernels will share the same buffer/bind conventions
/// documented in this module’s crate-level docs.
pub struct QFunctionPrototypeScaleF32 {
    runtime: Arc<GpuRuntime>,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl QFunctionPrototypeScaleF32 {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let device = &runtime.device;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reed-qf-proto-scale"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-proto-scale"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL_SCALE_PROTO)),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reed-qf-proto-scale-pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reed-qf-proto-scale-pipe"),
            layout: Some(&pipeline_layout),
            module: &sm,
            entry_point: "qf_scale_proto",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Ok(Self {
            runtime,
            layout,
            pipeline,
        })
    }

    /// Writes `out[i] = scale * in[i]` for `i in 0..num_q`.
    ///
    /// `input` / `output` slices must have length at least `num_q`.
    pub fn apply(
        &self,
        num_q: usize,
        scale: f32,
        input: &[f32],
        output: &mut [f32],
    ) -> ReedResult<()> {
        if num_q == 0 {
            return Ok(());
        }
        if input.len() < num_q || output.len() < num_q {
            return Err(ReedError::QFunction(format!(
                "apply: need len >= num_q ({num_q}), got in={} out={}",
                input.len(),
                output.len()
            )));
        }
        let device = &self.runtime.device;
        let queue = &self.runtime.queue;
        let bytes = (num_q * std::mem::size_of::<f32>()) as u64;

        let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reed-qf-proto-in"),
            contents: bytemuck::cast_slice(&input[..num_q]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-qf-proto-out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = QfProtoParamsHost {
            num_q: num_q as u32,
            _pad0: 0,
            scale,
            _pad1: 0.0,
        };
        let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reed-qf-proto-uni"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reed-qf-proto-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reed-qf-proto-enc"),
        });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reed-qf-proto-pass"),
                ..Default::default()
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind, &[]);
            let wg = 256u32;
            let groups = ((num_q as u32) + wg - 1) / wg;
            cpass.dispatch_workgroups(groups, 1, 1);
        }

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-qf-proto-rb"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, bytes);
        queue.submit(std::iter::once(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
            .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
        {
            let data = slice.get_mapped_range();
            let mapped: &[f32] = bytemuck::cast_slice(&data);
            output[..num_q].copy_from_slice(&mapped[..num_q]);
        }
        readback.unmap();
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QfPointwiseMulParamsHost {
    num_q: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// `out[i] = in0[i] * in1[i]` using [`GpuRuntime::qfunction_pointwise_mul_pipeline`].
pub(crate) fn dispatch_qf_pointwise_mul_f32(
    runtime: &GpuRuntime,
    q: usize,
    in0: &[f32],
    in1: &[f32],
    out: &mut [f32],
) -> ReedResult<()> {
    if q == 0 {
        return Ok(());
    }
    if in0.len() < q || in1.len() < q || out.len() < q {
        return Err(ReedError::QFunction(format!(
            "pointwise_mul: need len >= q ({q}), got in0={} in1={} out={}",
            in0.len(),
            in1.len(),
            out.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (q * std::mem::size_of::<f32>()) as u64;

    let in0_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-pw-in0"),
        contents: bytemuck::cast_slice(&in0[..q]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let in1_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-pw-in1"),
        contents: bytemuck::cast_slice(&in1[..q]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-pw-out"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-pw-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-pw-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: in0_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: in1_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-pw-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-pw-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_pointwise_mul_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((q as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-pw-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        out[..q].copy_from_slice(&mapped[..q]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` implementation of the scalar gallery [`MassApply`](reed_cpu::MassApply): `v = u * qdata`.
///
/// I/O layout matches [`QFunctionTrait::apply`] / `CpuOperator` staging (one scalar per quadrature
/// point, contiguous). Host upload and readback occur inside [`QFunctionTrait::apply`].
pub struct MassApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl MassApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::MassApply::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for MassApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "MassApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let qdata = inputs[1];
        let v = &mut outputs[0];
        dispatch_qf_pointwise_mul_f32(&self.runtime, q, u, qdata, v)
    }
}

/// GPU `f32` gallery [`Poisson1DApply`](reed_cpu::Poisson1DApply): `dv[i] = du[i] * qdata[i]`.
///
/// For scalar 1D Poisson apply this matches [`MassApplyF32Wgpu`] numerically; this type carries
/// Poisson field names (`du`, `dv`, …) for [`OperatorBuilder`](reed_cpu::OperatorBuilder) while
/// reusing the shared [`GpuRuntime`] pointwise multiply pipeline.
pub struct Poisson1DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Poisson1DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Poisson1DApply::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Poisson1DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Poisson1DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_pointwise_mul_f32(&self.runtime, q, du, qdata, dv)
    }
}

/// 2D Poisson apply: for each quadrature index `i`, `dv[2*i..]` = `qdata[4*i..4*i+4] * du[2*i..]`.
pub(crate) fn dispatch_qf_poisson2d_apply_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    du: &[f32],
    qdata: &[f32],
    dv: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n_du = num_q.saturating_mul(2);
    let n_qd = num_q.saturating_mul(4);
    if du.len() < n_du || qdata.len() < n_qd || dv.len() < n_du {
        return Err(ReedError::QFunction(format!(
            "poisson2d_apply: need du>={n_du}, qdata>={n_qd}, dv>={n_du}; got du={} qdata={} dv={}",
            du.len(),
            qdata.len(),
            dv.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let du_bytes = (n_du * std::mem::size_of::<f32>()) as u64;

    let du_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p2d-du"),
        contents: bytemuck::cast_slice(&du[..n_du]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p2d-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n_qd]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let dv_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-p2d-dv"),
        size: du_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p2d-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-p2d-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: du_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dv_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-p2d-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-p2d-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_poisson2d_apply_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-p2d-rb"),
        size: du_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&dv_buf, 0, &readback, 0, du_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        dv[..n_du].copy_from_slice(&mapped[..n_du]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` gallery [`Poisson2DApply`](reed_cpu::Poisson2DApply): stiffness `qdata` (4×`q`) times
/// gradient `du` (2×`q`) into `dv` (2×`q`).
pub struct Poisson2DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Poisson2DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Poisson2DApply::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Poisson2DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Poisson2DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_poisson2d_apply_f32(&self.runtime, q, du, qdata, dv)
    }
}

/// 3D Poisson apply: for each quadrature index `i`, `dv[3*i..]` = `qdata[9*i..]` (row-major 3×3) × `du[3*i..]`.
pub(crate) fn dispatch_qf_poisson3d_apply_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    du: &[f32],
    qdata: &[f32],
    dv: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n_du = num_q.saturating_mul(3);
    let n_qd = num_q.saturating_mul(9);
    if du.len() < n_du || qdata.len() < n_qd || dv.len() < n_du {
        return Err(ReedError::QFunction(format!(
            "poisson3d_apply: need du>={n_du}, qdata>={n_qd}, dv>={n_du}; got du={} qdata={} dv={}",
            du.len(),
            qdata.len(),
            dv.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let du_bytes = (n_du * std::mem::size_of::<f32>()) as u64;

    let du_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p3d-du"),
        contents: bytemuck::cast_slice(&du[..n_du]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p3d-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n_qd]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let dv_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-p3d-dv"),
        size: du_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p3d-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-p3d-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: du_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dv_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-p3d-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-p3d-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_poisson3d_apply_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-p3d-rb"),
        size: du_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&dv_buf, 0, &readback, 0, du_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        dv[..n_du].copy_from_slice(&mapped[..n_du]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` gallery [`Poisson3DApply`](reed_cpu::Poisson3DApply).
pub struct Poisson3DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Poisson3DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Poisson3DApply::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Poisson3DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Poisson3DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_poisson3d_apply_f32(&self.runtime, q, du, qdata, dv)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QfUnaryWordCountHost {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QfScaleF32UniformHost {
    n: u32,
    _pad0: u32,
    alpha: f32,
    _pad1: f32,
}

/// Flat `out[..n] = in[..n]` using [`GpuRuntime::qfunction_identity_copy_pipeline`].
pub(crate) fn dispatch_qf_identity_copy_f32(
    runtime: &GpuRuntime,
    n_words: usize,
    input: &[f32],
    output: &mut [f32],
) -> ReedResult<()> {
    if n_words == 0 {
        return Ok(());
    }
    if input.len() < n_words || output.len() < n_words {
        return Err(ReedError::QFunction(format!(
            "identity_copy: need len >= n ({n_words}), got in={} out={}",
            input.len(),
            output.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n_words * std::mem::size_of::<f32>()) as u64;

    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-id-in"),
        contents: bytemuck::cast_slice(&input[..n_words]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-id-out"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfUnaryWordCountHost {
        n: n_words as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-id-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_unary_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-id-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-id-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-id-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_identity_copy_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((n_words as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-id-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        output[..n_words].copy_from_slice(&mapped[..n_words]);
    }
    readback.unmap();
    Ok(())
}

/// Flat `out[i] = alpha * in[i]` using [`GpuRuntime::qfunction_scale_f32_pipeline`].
pub(crate) fn dispatch_qf_scale_f32(
    runtime: &GpuRuntime,
    n_words: usize,
    alpha: f32,
    input: &[f32],
    output: &mut [f32],
) -> ReedResult<()> {
    if n_words == 0 {
        return Ok(());
    }
    if input.len() < n_words || output.len() < n_words {
        return Err(ReedError::QFunction(format!(
            "scale_f32: need len >= n ({n_words}), got in={} out={}",
            input.len(),
            output.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n_words * std::mem::size_of::<f32>()) as u64;

    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-sc-in"),
        contents: bytemuck::cast_slice(&input[..n_words]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-sc-out"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfScaleF32UniformHost {
        n: n_words as u32,
        _pad0: 0,
        alpha,
        _pad1: 0.0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-sc-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_unary_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-sc-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-sc-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-sc-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_scale_f32_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((n_words as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-sc-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        output[..n_words].copy_from_slice(&mapped[..n_words]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` gallery [`Identity`](reed_cpu::Identity): copy packed quadrature values.
pub struct IdentityF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl IdentityF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        Self::with_components(runtime, 1)
    }

    pub fn with_components(runtime: Arc<GpuRuntime>, ncomp: usize) -> ReedResult<Self> {
        let template = reed_cpu::Identity::with_components(ncomp);
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for IdentityF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "IdentityF32Wgpu expects 1 input and 1 output".into(),
            ));
        }
        let ncomp = self.inputs[0].num_comp;
        let n_words = q.saturating_mul(ncomp);
        let u = inputs[0];
        let v = &mut outputs[0];
        dispatch_qf_identity_copy_f32(&self.runtime, n_words, u, v)
    }
}

/// GPU `f32` gallery [`Scale`](reed_cpu::Scale): multiply by `alpha` from 8-byte `f64` LE context.
pub struct ScaleF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl ScaleF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        Self::with_components(runtime, 1)
    }

    pub fn with_components(runtime: Arc<GpuRuntime>, ncomp: usize) -> ReedResult<Self> {
        let template = reed_cpu::Scale::with_components(ncomp);
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for ScaleF32Wgpu {
    fn context_byte_len(&self) -> usize {
        8
    }

    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "ScaleF32Wgpu expects 1 input and 1 output".into(),
            ));
        }
        let alpha64 = QFunctionContext::read_f64_le_bytes(ctx, 0)?;
        let alpha = alpha64 as f32;
        let ncomp = self.inputs[0].num_comp;
        let n_words = q.saturating_mul(ncomp);
        let u = inputs[0];
        let v = &mut outputs[0];
        dispatch_qf_scale_f32(&self.runtime, n_words, alpha, u, v)
    }
}

/// `v[flat] = qdata[flat / 2] * u[flat]` for `flat in 0..2*num_qp` (gallery [`Vector2MassApply`](reed_cpu::Vector2MassApply)).
pub(crate) fn dispatch_qf_vector2_mass_apply_f32(
    runtime: &GpuRuntime,
    num_qp: usize,
    u: &[f32],
    qdata: &[f32],
    v: &mut [f32],
) -> ReedResult<()> {
    if num_qp == 0 {
        return Ok(());
    }
    let n_uv = num_qp.saturating_mul(2);
    if u.len() < n_uv || qdata.len() < num_qp || v.len() < n_uv {
        return Err(ReedError::QFunction(format!(
            "vector2_mass_apply: need u>={n_uv}, qdata>={num_qp}, v>={n_uv}; got u={} qdata={} v={}",
            u.len(),
            qdata.len(),
            v.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let u_bytes = (n_uv * std::mem::size_of::<f32>()) as u64;

    let u_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2m-u"),
        contents: bytemuck::cast_slice(&u[..n_uv]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2m-qdata"),
        contents: bytemuck::cast_slice(&qdata[..num_qp]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let v_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v2m-v"),
        size: u_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_qp as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2m-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v2m-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v2m-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v2m-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vector2_mass_apply_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = n_uv as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v2m-rb"),
        size: u_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&v_buf, 0, &readback, 0, u_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        v[..n_uv].copy_from_slice(&mapped[..n_uv]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` gallery [`Vector2MassApply`](reed_cpu::Vector2MassApply).
pub struct Vector2MassApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vector2MassApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vector2MassApply::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vector2MassApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vector2MassApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let qdata = inputs[1];
        let v = &mut outputs[0];
        dispatch_qf_vector2_mass_apply_f32(&self.runtime, q, u, qdata, v)
    }
}

/// `v[flat] = qdata[flat / 3] * u[flat]` for `flat in 0..3*num_qp` (gallery [`Vector3MassApply`](reed_cpu::Vector3MassApply)).
pub(crate) fn dispatch_qf_vector3_mass_apply_f32(
    runtime: &GpuRuntime,
    num_qp: usize,
    u: &[f32],
    qdata: &[f32],
    v: &mut [f32],
) -> ReedResult<()> {
    if num_qp == 0 {
        return Ok(());
    }
    let n_uv = num_qp.saturating_mul(3);
    if u.len() < n_uv || qdata.len() < num_qp || v.len() < n_uv {
        return Err(ReedError::QFunction(format!(
            "vector3_mass_apply: need u>={n_uv}, qdata>={num_qp}, v>={n_uv}; got u={} qdata={} v={}",
            u.len(),
            qdata.len(),
            v.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let u_bytes = (n_uv * std::mem::size_of::<f32>()) as u64;

    let u_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3m-u"),
        contents: bytemuck::cast_slice(&u[..n_uv]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3m-qdata"),
        contents: bytemuck::cast_slice(&qdata[..num_qp]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let v_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3m-v"),
        size: u_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_qp as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3m-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v3m-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v3m-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v3m-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vector3_mass_apply_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = n_uv as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3m-rb"),
        size: u_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&v_buf, 0, &readback, 0, u_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        v[..n_uv].copy_from_slice(&mapped[..n_uv]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` gallery [`Vector3MassApply`](reed_cpu::Vector3MassApply).
pub struct Vector3MassApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vector3MassApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vector3MassApply::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vector3MassApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vector3MassApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let qdata = inputs[1];
        let v = &mut outputs[0];
        dispatch_qf_vector3_mass_apply_f32(&self.runtime, q, u, qdata, v)
    }
}

/// GPU `f32` gallery [`Vector2Poisson1DApply`](reed_cpu::Vector2Poisson1DApply) — same kernel as [`Vector2MassApplyF32Wgpu`].
pub struct Vector2Poisson1DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vector2Poisson1DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vector2Poisson1DApply::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vector2Poisson1DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vector2Poisson1DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_vector2_mass_apply_f32(&self.runtime, q, du, qdata, dv)
    }
}

/// GPU `f32` gallery [`Vector3Poisson1DApply`](reed_cpu::Vector3Poisson1DApply) — same kernel as [`Vector3MassApplyF32Wgpu`].
pub struct Vector3Poisson1DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vector3Poisson1DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vector3Poisson1DApply::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vector3Poisson1DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vector3Poisson1DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_vector3_mass_apply_f32(&self.runtime, q, du, qdata, dv)
    }
}

/// [`Vector2Poisson2DApply`](reed_cpu::Vector2Poisson2DApply): `du`/`dv` length `4 * num_q`, `qdata` length `4 * num_q`.
pub(crate) fn dispatch_qf_vector2_poisson2d_apply_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    du: &[f32],
    qdata: &[f32],
    dv: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n = num_q.saturating_mul(4);
    if du.len() < n || qdata.len() < n || dv.len() < n {
        return Err(ReedError::QFunction(format!(
            "vector2_poisson2d_apply: need du>={n}, qdata>={n}, dv>={n}; got du={} qdata={} dv={}",
            du.len(),
            qdata.len(),
            dv.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n * std::mem::size_of::<f32>()) as u64;

    let du_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2p2-du"),
        contents: bytemuck::cast_slice(&du[..n]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2p2-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let dv_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v2p2-dv"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2p2-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v2p2-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: du_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dv_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v2p2-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v2p2-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vector2_poisson2d_apply_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v2p2-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&dv_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        dv[..n].copy_from_slice(&mapped[..n]);
    }
    readback.unmap();
    Ok(())
}

/// [`Vector3Poisson2DApply`](reed_cpu::Vector3Poisson2DApply): `du`/`dv` length `6 * num_q`, `qdata` length `4 * num_q`.
pub(crate) fn dispatch_qf_vector3_poisson2d_apply_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    du: &[f32],
    qdata: &[f32],
    dv: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n_du = num_q.saturating_mul(6);
    let n_qd = num_q.saturating_mul(4);
    if du.len() < n_du || qdata.len() < n_qd || dv.len() < n_du {
        return Err(ReedError::QFunction(format!(
            "vector3_poisson2d_apply: need du>={n_du}, qdata>={n_qd}, dv>={n_du}; got du={} qdata={} dv={}",
            du.len(),
            qdata.len(),
            dv.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n_du * std::mem::size_of::<f32>()) as u64;

    let du_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p2-du"),
        contents: bytemuck::cast_slice(&du[..n_du]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p2-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n_qd]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let dv_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3p2-dv"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p2-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v3p2-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: du_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dv_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v3p2-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v3p2-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vector3_poisson2d_apply_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3p2-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&dv_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        dv[..n_du].copy_from_slice(&mapped[..n_du]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` gallery [`Vector2Poisson2DApply`](reed_cpu::Vector2Poisson2DApply).
pub struct Vector2Poisson2DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vector2Poisson2DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vector2Poisson2DApply::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vector2Poisson2DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vector2Poisson2DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_vector2_poisson2d_apply_f32(&self.runtime, q, du, qdata, dv)
    }
}

/// GPU `f32` gallery [`Vector3Poisson2DApply`](reed_cpu::Vector3Poisson2DApply).
pub struct Vector3Poisson2DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vector3Poisson2DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vector3Poisson2DApply::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vector3Poisson2DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vector3Poisson2DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_vector3_poisson2d_apply_f32(&self.runtime, q, du, qdata, dv)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use reed_core::QFunctionContext;

    #[test]
    fn prototype_scale_matches_cpu() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = GpuRuntime::new(&adapter).expect("device");
        let proto = QFunctionPrototypeScaleF32::new(Arc::new(rt)).unwrap();
        let n = 100usize;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut out_gpu = vec![0.0_f32; n];
        proto.apply(n, 2.5, &input, &mut out_gpu).unwrap();
        for i in 0..n {
            assert!(
                (out_gpu[i] - 2.5 * input[i]).abs() < 1.0e-4,
                "i={i} got {} want {}",
                out_gpu[i],
                2.5 * input[i]
            );
        }
    }

    #[test]
    fn mass_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = MassApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::MassApply::default();
        let q = 64usize;
        let u: Vec<f32> = (0..q).map(|i| i as f32 * 0.03).collect();
        let qd: Vec<f32> = (0..q).map(|i| 1.0 + i as f32 * 0.01).collect();
        let mut out_gpu = vec![0.0_f32; q];
        let mut out_cpu = vec![0.0_f32; q];
        gpu
            .apply(&[], q, &[u.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu
            .apply(&[], q, &[u.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn poisson1d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Poisson1DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Poisson1DApply::default();
        let q = 64usize;
        let du: Vec<f32> = (0..q).map(|i| i as f32 * 0.03).collect();
        let qd: Vec<f32> = (0..q).map(|i| 1.0 + i as f32 * 0.01).collect();
        let mut out_gpu = vec![0.0_f32; q];
        let mut out_cpu = vec![0.0_f32; q];
        gpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn poisson2d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Poisson2DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Poisson2DApply::default();
        let q = 40usize;
        let du: Vec<f32> = (0..2 * q).map(|i| (i as f32) * 0.01 - 0.1).collect();
        let qd: Vec<f32> = (0..4 * q)
            .map(|i| 0.25 + (i as f32) * 0.007)
            .collect();
        let mut out_gpu = vec![0.0_f32; 2 * q];
        let mut out_cpu = vec![0.0_f32; 2 * q];
        gpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..2 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 2.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn poisson3d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Poisson3DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Poisson3DApply::default();
        let q = 32usize;
        let du: Vec<f32> = (0..3 * q).map(|i| (i as f32) * 0.02 - 0.3).collect();
        let qd: Vec<f32> = (0..9 * q)
            .map(|i| 0.1 + (i as f32) * 0.005)
            .collect();
        let mut out_gpu = vec![0.0_f32; 3 * q];
        let mut out_cpu = vec![0.0_f32; 3 * q];
        gpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..3 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 3.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn identity_f32_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = IdentityF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Identity::default();
        let q = 32usize;
        let ncomp = 1usize;
        let n = q * ncomp;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.07).collect();
        let mut out_gpu = vec![0.0_f32; n];
        let mut out_cpu = vec![0.0_f32; n];
        gpu
            .apply(&[], q, &[input.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu
            .apply(&[], q, &[input.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        assert_eq!(out_gpu, out_cpu);
    }

    #[test]
    fn identity_f32_wgpu_ncomp3_matches_cpu() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = IdentityF32Wgpu::with_components(rt, 3).unwrap();
        let cpu = reed_cpu::Identity::with_components(3);
        let q = 8usize;
        let n = q * 3;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.11).collect();
        let mut out_gpu = vec![0.0_f32; n];
        let mut out_cpu = vec![0.0_f32; n];
        gpu
            .apply(&[], q, &[input.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu
            .apply(&[], q, &[input.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        assert_eq!(out_gpu, out_cpu);
    }

    #[test]
    fn scale_f32_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = ScaleF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Scale::default();
        let mut ctx = [0u8; 8];
        QFunctionContext::write_f64_le_bytes(&mut ctx, 0, -2.25).unwrap();
        let q = 40usize;
        let input: Vec<f32> = (0..q).map(|i| i as f32 * 0.05).collect();
        let mut out_gpu = vec![0.0_f32; q];
        let mut out_cpu = vec![0.0_f32; q];
        gpu
            .apply(&ctx, q, &[input.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu
            .apply(&ctx, q, &[input.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 2.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn vector2_mass_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector2MassApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector2MassApply::new();
        let q = 48usize;
        let u: Vec<f32> = (0..2 * q).map(|i| i as f32 * 0.02).collect();
        let qd: Vec<f32> = (0..q).map(|i| 0.5 + i as f32 * 0.03).collect();
        let mut out_gpu = vec![0.0_f32; 2 * q];
        let mut out_cpu = vec![0.0_f32; 2 * q];
        gpu
            .apply(&[], q, &[u.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu
            .apply(&[], q, &[u.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..2 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn vector3_mass_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector3MassApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector3MassApply::new();
        let q = 48usize;
        let u: Vec<f32> = (0..3 * q).map(|i| i as f32 * 0.02).collect();
        let qd: Vec<f32> = (0..q).map(|i| 0.5 + i as f32 * 0.03).collect();
        let mut out_gpu = vec![0.0_f32; 3 * q];
        let mut out_cpu = vec![0.0_f32; 3 * q];
        gpu
            .apply(&[], q, &[u.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu
            .apply(&[], q, &[u.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..3 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn vector2_poisson1d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector2Poisson1DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector2Poisson1DApply::new();
        let q = 48usize;
        let du: Vec<f32> = (0..2 * q).map(|i| i as f32 * 0.02).collect();
        let qd: Vec<f32> = (0..q).map(|i| 0.5 + i as f32 * 0.03).collect();
        let mut out_gpu = vec![0.0_f32; 2 * q];
        let mut out_cpu = vec![0.0_f32; 2 * q];
        gpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..2 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn vector3_poisson1d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector3Poisson1DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector3Poisson1DApply::new();
        let q = 48usize;
        let du: Vec<f32> = (0..3 * q).map(|i| i as f32 * 0.02).collect();
        let qd: Vec<f32> = (0..q).map(|i| 0.5 + i as f32 * 0.03).collect();
        let mut out_gpu = vec![0.0_f32; 3 * q];
        let mut out_cpu = vec![0.0_f32; 3 * q];
        gpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..3 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn vector2_poisson2d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector2Poisson2DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector2Poisson2DApply::new();
        let q = 36usize;
        let du: Vec<f32> = (0..4 * q).map(|i| (i as f32) * 0.015 - 0.2).collect();
        let qd: Vec<f32> = (0..4 * q)
            .map(|i| 0.2 + (i as f32) * 0.006)
            .collect();
        let mut out_gpu = vec![0.0_f32; 4 * q];
        let mut out_cpu = vec![0.0_f32; 4 * q];
        gpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..4 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 2.5e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn vector3_poisson2d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector3Poisson2DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector3Poisson2DApply::new();
        let q = 36usize;
        let du: Vec<f32> = (0..6 * q).map(|i| (i as f32) * 0.012 - 0.15).collect();
        let qd: Vec<f32> = (0..4 * q)
            .map(|i| 0.18 + (i as f32) * 0.005)
            .collect();
        let mut out_gpu = vec![0.0_f32; 6 * q];
        let mut out_cpu = vec![0.0_f32; 6 * q];
        gpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu
            .apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..6 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 2.5e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }
}
