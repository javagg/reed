use std::{any::TypeId, sync::Arc};

use num_traits::NumCast;
use reed_core::{
    enums::NormType, error::ReedResult, scalar::Scalar, VectorTrait, ReedError,
};
use wgpu::util::DeviceExt;

use crate::runtime::GpuRuntime;

pub struct WgpuVector<T: Scalar> {
    data: Vec<T>,
    runtime: Option<Arc<GpuRuntime>>,
}

impl<T: Scalar> WgpuVector<T> {
    pub fn new(size: usize, runtime: Option<Arc<GpuRuntime>>) -> Self {
        Self {
            data: vec![T::ZERO; size],
            runtime,
        }
    }

    fn supports_f32_gpu() -> bool {
        TypeId::of::<T>() == TypeId::of::<f32>()
    }

    fn as_f32_vec(data: &[T]) -> Option<Vec<f32>> {
        data.iter().map(|v| NumCast::from(*v)).collect::<Option<Vec<f32>>>()
    }

    fn from_f32_into(data: &mut [T], f32_data: &[f32]) -> ReedResult<()> {
        if data.len() != f32_data.len() {
            return Err(ReedError::Vector("size mismatch during gpu readback".into()));
        }
        for (dst, src) in data.iter_mut().zip(f32_data.iter()) {
            *dst = NumCast::from(*src)
                .ok_or_else(|| ReedError::Vector("f32->T conversion failed".into()))?;
        }
        Ok(())
    }

    fn dispatch_set_f32(&mut self, alpha: f32) -> ReedResult<bool> {
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        let Some(mut y) = Self::as_f32_vec(&self.data) else {
            return Ok(false);
        };

        let y_bytes: &[u8] = bytemuck::cast_slice(&y);
        let y_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-vector-y-set"),
                contents: y_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params = [alpha, self.data.len() as f32, 0.0, 0.0];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-vector-params-set"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-vector-bind-set"),
            layout: runtime.set_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p_buffer.as_entire_binding(),
                },
            ],
        });

        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-vector-readback-set"),
            size: (y.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-vector-encoder-set"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-vector-pass-set"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.set_pipeline());
            pass.set_bind_group(0, &bind, &[]);
            let groups = (self.data.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &y_buffer,
            0,
            &readback,
            0,
            (y.len() * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        map_readback_f32(&runtime.device, &readback, &mut y)?;
        Self::from_f32_into(&mut self.data, &y)?;
        Ok(true)
    }

    fn dispatch_scale_f32(&mut self, alpha: f32) -> ReedResult<bool> {
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        let Some(mut y) = Self::as_f32_vec(&self.data) else {
            return Ok(false);
        };

        let y_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-vector-y-scale"),
                contents: bytemuck::cast_slice(&y),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params = [alpha, self.data.len() as f32, 0.0, 0.0];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-vector-params-scale"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-vector-bind-scale"),
            layout: runtime.scale_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p_buffer.as_entire_binding(),
                },
            ],
        });

        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-vector-readback-scale"),
            size: (y.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-vector-encoder-scale"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-vector-pass-scale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.scale_pipeline());
            pass.set_bind_group(0, &bind, &[]);
            let groups = (self.data.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &y_buffer,
            0,
            &readback,
            0,
            (y.len() * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        map_readback_f32(&runtime.device, &readback, &mut y)?;
        Self::from_f32_into(&mut self.data, &y)?;
        Ok(true)
    }

    fn dispatch_axpy_f32(&mut self, alpha: f32, x: &[T]) -> ReedResult<bool> {
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        let Some(mut y) = Self::as_f32_vec(&self.data) else {
            return Ok(false);
        };
        let Some(x_f32) = Self::as_f32_vec(x) else {
            return Ok(false);
        };

        let y_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-vector-y-axpy"),
                contents: bytemuck::cast_slice(&y),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let x_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-vector-x-axpy"),
                contents: bytemuck::cast_slice(&x_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let params = [alpha, self.data.len() as f32, 0.0, 0.0];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-vector-params-axpy"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-vector-bind-axpy"),
            layout: runtime.axpy_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p_buffer.as_entire_binding(),
                },
            ],
        });

        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-vector-readback-axpy"),
            size: (y.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-vector-encoder-axpy"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-vector-pass-axpy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.axpy_pipeline());
            pass.set_bind_group(0, &bind, &[]);
            let groups = (self.data.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &y_buffer,
            0,
            &readback,
            0,
            (y.len() * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        map_readback_f32(&runtime.device, &readback, &mut y)?;
        Self::from_f32_into(&mut self.data, &y)?;
        Ok(true)
    }
}

impl<T: Scalar> VectorTrait<T> for WgpuVector<T> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn copy_from_slice(&mut self, data: &[T]) -> ReedResult<()> {
        if data.len() != self.data.len() {
            return Err(ReedError::Vector(format!(
                "slice length {} != vector length {}",
                data.len(),
                self.data.len()
            )));
        }
        self.data.copy_from_slice(data);
        Ok(())
    }

    fn copy_to_slice(&self, data: &mut [T]) -> ReedResult<()> {
        if data.len() != self.data.len() {
            return Err(ReedError::Vector(format!(
                "slice length {} != vector length {}",
                data.len(),
                self.data.len()
            )));
        }
        data.copy_from_slice(&self.data);
        Ok(())
    }

    fn set_value(&mut self, val: T) -> ReedResult<()> {
        if Self::supports_f32_gpu() {
            if let Some(alpha) = NumCast::from(val) {
                if self.dispatch_set_f32(alpha)? {
                    return Ok(());
                }
            }
        }

        for x in &mut self.data {
            *x = val;
        }
        Ok(())
    }

    fn axpy(&mut self, alpha: T, x: &dyn VectorTrait<T>) -> ReedResult<()> {
        if x.len() != self.data.len() {
            return Err(ReedError::Vector(format!(
                "axpy: x length {} != self length {}",
                x.len(),
                self.data.len()
            )));
        }

        if Self::supports_f32_gpu() {
            if let Some(alpha_f32) = NumCast::from(alpha) {
                if self.dispatch_axpy_f32(alpha_f32, x.as_slice())? {
                    return Ok(());
                }
            }
        }

        for (yi, &xi) in self.data.iter_mut().zip(x.as_slice().iter()) {
            *yi += alpha * xi;
        }
        Ok(())
    }

    fn scale(&mut self, alpha: T) -> ReedResult<()> {
        if Self::supports_f32_gpu() {
            if let Some(alpha_f32) = NumCast::from(alpha) {
                if self.dispatch_scale_f32(alpha_f32)? {
                    return Ok(());
                }
            }
        }

        for x in &mut self.data {
            *x *= alpha;
        }
        Ok(())
    }

    fn norm(&self, norm_type: NormType) -> ReedResult<T> {
        let result = match norm_type {
            NormType::One => {
                let mut sum = T::ZERO;
                for &x in &self.data {
                    sum += x.abs();
                }
                sum
            }
            NormType::Two => {
                let mut sum = T::ZERO;
                for &x in &self.data {
                    sum += x * x;
                }
                sum.sqrt()
            }
            NormType::Max => {
                let mut max = T::ZERO;
                for &x in &self.data {
                    let a = x.abs();
                    if a > max {
                        max = a;
                    }
                }
                max
            }
        };
        Ok(result)
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
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
        .map_err(|e| ReedError::Vector(format!("map recv error: {e}")))?;
    map_result.map_err(|e| ReedError::Vector(format!("map error: {e:?}")))?;

    let data = slice.get_mapped_range();
    let mapped: &[f32] = bytemuck::cast_slice(&data);
    if mapped.len() != out.len() {
        return Err(ReedError::Vector("readback length mismatch".into()));
    }
    out.copy_from_slice(mapped);
    drop(data);
    readback.unmap();
    Ok(())
}
