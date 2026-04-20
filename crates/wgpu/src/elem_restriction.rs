use std::{any::TypeId, sync::Arc};

use num_traits::NumCast;
use reed_core::{
    enums::TransposeMode, error::ReedResult, scalar::Scalar, ElemRestrictionTrait, ReedError,
};
use reed_cpu::elem_restriction::CpuElemRestriction;
use wgpu::util::DeviceExt;

use crate::runtime::GpuRuntime;

#[derive(Clone)]
enum RestrictionLayout {
    Offset {
        offsets: Vec<i32>,
        compstride: usize,
    },
    Strided {
        _strides: [i32; 3],
    },
}

pub struct WgpuElemRestriction<T: Scalar> {
    nelem: usize,
    elemsize: usize,
    ncomp: usize,
    lsize: usize,
    layout: RestrictionLayout,
    runtime: Option<Arc<GpuRuntime>>,
    cpu_fallback: CpuElemRestriction<T>,
}

impl<T: Scalar> WgpuElemRestriction<T> {
    pub fn new_offset(
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i32],
        runtime: Option<Arc<GpuRuntime>>,
    ) -> ReedResult<Self> {
        let cpu_fallback = CpuElemRestriction::<T>::new_offset(
            nelem, elemsize, ncomp, compstride, lsize, offsets,
        )?;
        Ok(Self {
            nelem,
            elemsize,
            ncomp,
            lsize,
            layout: RestrictionLayout::Offset {
                offsets: offsets.to_vec(),
                compstride,
            },
            runtime,
            cpu_fallback,
        })
    }

    pub fn new_strided(
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
        runtime: Option<Arc<GpuRuntime>>,
    ) -> ReedResult<Self> {
        let cpu_fallback =
            CpuElemRestriction::<T>::new_strided(nelem, elemsize, ncomp, lsize, strides)?;
        Ok(Self {
            nelem,
            elemsize,
            ncomp,
            lsize,
            layout: RestrictionLayout::Strided { _strides: strides },
            runtime,
            cpu_fallback,
        })
    }

    fn supports_f32_gpu() -> bool {
        TypeId::of::<T>() == TypeId::of::<f32>()
    }

    fn local_size(&self) -> usize {
        self.nelem * self.elemsize * self.ncomp
    }

    fn try_apply_no_transpose_gpu(&self, u: &[T], v: &mut [T]) -> ReedResult<bool> {
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        if !Self::supports_f32_gpu() {
            return Ok(false);
        }

        let (offsets, compstride) = match &self.layout {
            RestrictionLayout::Offset {
                offsets,
                compstride,
            } => (offsets, *compstride),
            RestrictionLayout::Strided { .. } => return Ok(false),
        };

        if u.len() != self.lsize {
            return Err(ReedError::ElemRestriction(format!(
                "input length {} != global size {}",
                u.len(),
                self.lsize
            )));
        }

        let local_size = self.local_size();
        if v.len() != local_size {
            return Err(ReedError::ElemRestriction(format!(
                "output length {} != local size {}",
                v.len(),
                local_size
            )));
        }

        let Some(u_f32) = u
            .iter()
            .map(|x| NumCast::from(*x))
            .collect::<Option<Vec<f32>>>()
        else {
            return Ok(false);
        };
        let mut v_f32 = vec![0.0_f32; local_size];

        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let offsets_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-offsets"),
                contents: bytemuck::cast_slice(offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-v"),
                contents: bytemuck::cast_slice(&v_f32),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params: [u32; 8] = [
            self.nelem as u32,
            self.elemsize as u32,
            self.ncomp as u32,
            compstride as u32,
            local_size as u32,
            self.lsize as u32,
            0,
            0,
        ];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-restriction-bind"),
                layout: runtime.restriction_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: offsets_buffer.as_entire_binding(),
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
            label: Some("wgpu-restriction-readback"),
            size: (local_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-restriction-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-restriction-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.restriction_pipeline());
            pass.set_bind_group(0, &bind, &[]);
            let groups = (local_size as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &v_buffer,
            0,
            &readback,
            0,
            (local_size * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        map_readback_f32(&runtime.device, &readback, &mut v_f32)?;

        for (dst, src) in v.iter_mut().zip(v_f32.iter()) {
            *dst = NumCast::from(*src).ok_or_else(|| {
                ReedError::ElemRestriction("f32->T conversion failed during readback".into())
            })?;
        }
        Ok(true)
    }

    fn try_apply_transpose_gpu(&self, u: &[T], v: &mut [T]) -> ReedResult<bool> {
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        if !Self::supports_f32_gpu() {
            return Ok(false);
        }

        let (offsets, compstride) = match &self.layout {
            RestrictionLayout::Offset {
                offsets,
                compstride,
            } => (offsets, *compstride),
            RestrictionLayout::Strided { .. } => return Ok(false),
        };

        let local_size = self.local_size();
        if u.len() != local_size {
            return Err(ReedError::ElemRestriction(format!(
                "transpose input length {} != local size {}",
                u.len(),
                local_size
            )));
        }
        if v.len() != self.lsize {
            return Err(ReedError::ElemRestriction(format!(
                "transpose output length {} != global size {}",
                v.len(),
                self.lsize
            )));
        }

        let Some(u_f32) = u
            .iter()
            .map(|x| NumCast::from(*x))
            .collect::<Option<Vec<f32>>>()
        else {
            return Ok(false);
        };

        let mut v_f32_host: Vec<f32> = Vec::with_capacity(self.lsize);
        for x in v.iter() {
            let f: f32 = NumCast::from(*x).ok_or_else(|| {
                ReedError::ElemRestriction("transpose: expected f32-compatible values".into())
            })?;
            v_f32_host.push(f);
        }

        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-t-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let offsets_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-t-off"),
                contents: bytemuck::cast_slice(offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-t-v"),
                contents: bytemuck::cast_slice(&v_f32_host),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params: [u32; 8] = [
            self.nelem as u32,
            self.elemsize as u32,
            self.ncomp as u32,
            compstride as u32,
            local_size as u32,
            self.lsize as u32,
            0,
            0,
        ];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-t-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-restriction-t-bind"),
                layout: runtime.restriction_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: offsets_buffer.as_entire_binding(),
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
            label: Some("wgpu-restriction-t-readback"),
            size: (self.lsize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-restriction-t-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-restriction-t-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.restriction_scatter_pipeline());
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &v_buffer,
            0,
            &readback,
            0,
            (self.lsize * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        let mut v_out = vec![0.0_f32; self.lsize];
        map_readback_f32(&runtime.device, &readback, &mut v_out)?;
        for (dst, src) in v.iter_mut().zip(v_out.iter()) {
            *dst = NumCast::from(*src).ok_or_else(|| {
                ReedError::ElemRestriction(
                    "f32->T conversion failed after transpose readback".into(),
                )
            })?;
        }
        Ok(true)
    }
}

impl<T: Scalar> ElemRestrictionTrait<T> for WgpuElemRestriction<T> {
    fn num_elements(&self) -> usize {
        self.nelem
    }

    fn num_dof_per_elem(&self) -> usize {
        self.elemsize
    }

    fn num_global_dof(&self) -> usize {
        self.lsize
    }

    fn num_comp(&self) -> usize {
        self.ncomp
    }

    fn apply(&self, t_mode: TransposeMode, u: &[T], v: &mut [T]) -> ReedResult<()> {
        if matches!(t_mode, TransposeMode::NoTranspose) && self.try_apply_no_transpose_gpu(u, v)? {
            return Ok(());
        }
        if matches!(t_mode, TransposeMode::Transpose) && self.try_apply_transpose_gpu(u, v)? {
            return Ok(());
        }
        self.cpu_fallback.apply(t_mode, u, v)
    }
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
        .map_err(|e| ReedError::ElemRestriction(format!("map recv error: {e}")))?;
    map_result.map_err(|e| ReedError::ElemRestriction(format!("map error: {e:?}")))?;

    let data = slice.get_mapped_range();
    let mapped: &[f32] = bytemuck::cast_slice(&data);
    if mapped.len() != out.len() {
        return Err(ReedError::ElemRestriction(
            "restriction readback length mismatch".into(),
        ));
    }
    out.copy_from_slice(mapped);
    drop(data);
    readback.unmap();
    Ok(())
}
