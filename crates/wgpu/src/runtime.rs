use std::sync::Arc;

pub struct GpuRuntime {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    set_layout: wgpu::BindGroupLayout,
    set_pipeline: wgpu::ComputePipeline,
    scale_layout: wgpu::BindGroupLayout,
    scale_pipeline: wgpu::ComputePipeline,
    axpy_layout: wgpu::BindGroupLayout,
    axpy_pipeline: wgpu::ComputePipeline,
    restriction_layout: wgpu::BindGroupLayout,
    restriction_pipeline: wgpu::ComputePipeline,
    basis_interp_layout: wgpu::BindGroupLayout,
    basis_interp_pipeline: wgpu::ComputePipeline,
}

impl GpuRuntime {
    /// Synchronous init (native platforms — uses pollster internally).
    pub fn new(adapter: &wgpu::Adapter) -> Option<Self> {
        let req = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("reed-wgpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ));
        let (device, queue) = req.ok()?;
        Some(Self::from_device_queue(device, queue))
    }

    /// Async init for WASM (no pollster — await the WebGPU futures).
    pub async fn new_async(
        instance: &wgpu::Instance,
        power_pref: wgpu::PowerPreference,
        force_fallback: bool,
    ) -> Option<Self> {
        let Some(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: power_pref,
                force_fallback_adapter: force_fallback,
                compatible_surface: None,
            })
            .await else {
            return None;
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("reed-wgpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                },
                None,
            )
            .await
            .ok()?;

        Some(Self::from_device_queue(device, queue))
    }

    /// Build from an already-instantiated device + queue.
    pub fn from_device_queue(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let set_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reed-set-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let scale_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reed-scale-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let axpy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reed-axpy-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let restriction_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reed-restriction-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let basis_interp_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reed-basis-interp-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let set_pipeline = create_pipeline(&device, &set_layout, "set_main");
        let scale_pipeline = create_pipeline(&device, &scale_layout, "scale_main");
        let axpy_pipeline = create_pipeline(&device, &axpy_layout, "axpy_main");
        let restriction_pipeline =
            create_pipeline(&device, &restriction_layout, "restriction_gather_main");
        let basis_interp_pipeline =
            create_pipeline(&device, &basis_interp_layout, "basis_interp_main");

        Self {
            device,
            queue,
            set_layout,
            set_pipeline,
            scale_layout,
            scale_pipeline,
            axpy_layout,
            axpy_pipeline,
            restriction_layout,
            restriction_pipeline,
            basis_interp_layout,
            basis_interp_pipeline,
        }
    }

    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    pub fn set_layout(&self) -> &wgpu::BindGroupLayout {
        &self.set_layout
    }

    pub fn set_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.set_pipeline
    }

    pub fn scale_layout(&self) -> &wgpu::BindGroupLayout {
        &self.scale_layout
    }

    pub fn scale_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.scale_pipeline
    }

    pub fn axpy_layout(&self) -> &wgpu::BindGroupLayout {
        &self.axpy_layout
    }

    pub fn axpy_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.axpy_pipeline
    }

    pub fn restriction_layout(&self) -> &wgpu::BindGroupLayout {
        &self.restriction_layout
    }

    pub fn restriction_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.restriction_pipeline
    }

    pub fn basis_interp_layout(&self) -> &wgpu::BindGroupLayout {
        &self.basis_interp_layout
    }

    pub fn basis_interp_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.basis_interp_pipeline
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("reed-vector-kernels"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(KERNELS_WGSL)),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("reed-vector-pipeline-layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("reed-vector-pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    })
}

const KERNELS_WGSL: &str = r#"
struct Params {
    alpha: f32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
};

struct RestrictionParams {
    nelem: u32,
    elemsize: u32,
    ncomp: u32,
    compstride: u32,
    local_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct BasisInterpParams {
    num_elem: u32,
    num_dof: u32,
    num_qpoints: u32,
    ncomp: u32,
    output_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<uniform> p: Params;

@compute @workgroup_size(64)
fn set_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) {
        y[i] = p.alpha;
    }
}

@compute @workgroup_size(64)
fn scale_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) {
        y[i] = y[i] * p.alpha;
    }
}

@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<uniform> p2: Params;

@compute @workgroup_size(64)
fn axpy_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p2.n) {
        y[i] = p2.alpha * x[i] + y[i];
    }
}

@group(0) @binding(0) var<storage, read> rg_u: array<f32>;
@group(0) @binding(1) var<storage, read> rg_offsets: array<i32>;
@group(0) @binding(2) var<storage, read_write> rg_v: array<f32>;
@group(0) @binding(3) var<uniform> rg_p: RestrictionParams;

@compute @workgroup_size(64)
fn restriction_gather_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= rg_p.local_size) {
        return;
    }

    let per_elem = rg_p.ncomp * rg_p.elemsize;
    let elem = idx / per_elem;
    let rem = idx % per_elem;
    let comp = rem / rg_p.elemsize;
    let local = rem % rg_p.elemsize;

    let offset_idx = elem * rg_p.elemsize + local;
    let base = rg_offsets[offset_idx];
    if (base < 0) {
        rg_v[idx] = 0.0;
        return;
    }
    let g = u32(base) + comp * rg_p.compstride;
    rg_v[idx] = rg_u[g];
}

@group(0) @binding(0) var<storage, read> bi_mat: array<f32>;
@group(0) @binding(1) var<storage, read> bi_u: array<f32>;
@group(0) @binding(2) var<storage, read_write> bi_v: array<f32>;
@group(0) @binding(3) var<uniform> bi_p: BasisInterpParams;

@compute @workgroup_size(64)
fn basis_interp_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bi_p.output_size) {
        return;
    }

    let per_elem_out = bi_p.ncomp * bi_p.num_qpoints;
    let elem = idx / per_elem_out;
    let rem = idx % per_elem_out;
    let comp = rem / bi_p.num_qpoints;
    let qpt = rem % bi_p.num_qpoints;

    var sum = 0.0;
    let u_elem_base = (elem * bi_p.ncomp + comp) * bi_p.num_dof;
    let mat_row_base = qpt * bi_p.num_dof;
    for (var dof = 0u; dof < bi_p.num_dof; dof = dof + 1u) {
        sum = sum + bi_mat[mat_row_base + dof] * bi_u[u_elem_base + dof];
    }
    bi_v[idx] = sum;
}
"#;
