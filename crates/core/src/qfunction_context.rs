//! User data passed into [`crate::qfunction::QFunctionTrait::apply`], analogous to
//! libCEED's `CeedQFunctionContext` (opaque bytes sized per qfunction).

use crate::error::{ReedError, ReedResult};

/// Byte buffer holding per-operator user state for qfunctions (coefficients, flags, etc.).
#[derive(Debug, Clone)]
pub struct QFunctionContext {
    data: Vec<u8>,
}

impl QFunctionContext {
    /// Allocate a zero-filled buffer of `byte_len` bytes.
    pub fn new(byte_len: usize) -> Self {
        Self {
            data: vec![0u8; byte_len],
        }
    }

    /// Take ownership of raw bytes (e.g. from serialization).
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { data }
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    #[inline]
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Read `f64` at `offset` (little-endian). `offset` must be 8-byte aligned within the buffer.
    pub fn read_f64_le(&self, offset: usize) -> ReedResult<f64> {
        if offset + 8 > self.data.len() {
            return Err(ReedError::QFunction(format!(
                "read_f64_le: offset {} + 8 exceeds context length {}",
                offset,
                self.data.len()
            )));
        }
        let b: [u8; 8] = self.data[offset..offset + 8]
            .try_into()
            .map_err(|_| ReedError::QFunction("read_f64_le: slice length".into()))?;
        Ok(f64::from_le_bytes(b))
    }

    /// Write `f64` at `offset` (little-endian).
    pub fn write_f64_le(&mut self, offset: usize, v: f64) -> ReedResult<()> {
        if offset + 8 > self.data.len() {
            return Err(ReedError::QFunction(format!(
                "write_f64_le: offset {} + 8 exceeds context length {}",
                offset,
                self.data.len()
            )));
        }
        self.data[offset..offset + 8].copy_from_slice(&v.to_le_bytes());
        Ok(())
    }

    /// Read `f32` at `offset` (little-endian).
    pub fn read_f32_le(&self, offset: usize) -> ReedResult<f32> {
        if offset + 4 > self.data.len() {
            return Err(ReedError::QFunction(format!(
                "read_f32_le: offset {} + 4 exceeds context length {}",
                offset,
                self.data.len()
            )));
        }
        let b: [u8; 4] = self.data[offset..offset + 4]
            .try_into()
            .map_err(|_| ReedError::QFunction("read_f32_le: slice length".into()))?;
        Ok(f32::from_le_bytes(b))
    }

    /// Write `f32` at `offset` (little-endian).
    pub fn write_f32_le(&mut self, offset: usize, v: f32) -> ReedResult<()> {
        if offset + 4 > self.data.len() {
            return Err(ReedError::QFunction(format!(
                "write_f32_le: offset {} + 4 exceeds context length {}",
                offset,
                self.data.len()
            )));
        }
        self.data[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
        Ok(())
    }

    /// Read `i32` at `offset` (little-endian).
    pub fn read_i32_le(&self, offset: usize) -> ReedResult<i32> {
        if offset + 4 > self.data.len() {
            return Err(ReedError::QFunction(format!(
                "read_i32_le: offset {} + 4 exceeds context length {}",
                offset,
                self.data.len()
            )));
        }
        let b: [u8; 4] = self.data[offset..offset + 4]
            .try_into()
            .map_err(|_| ReedError::QFunction("read_i32_le: slice length".into()))?;
        Ok(i32::from_le_bytes(b))
    }

    /// Write `i32` at `offset` (little-endian).
    pub fn write_i32_le(&mut self, offset: usize, v: i32) -> ReedResult<()> {
        if offset + 4 > self.data.len() {
            return Err(ReedError::QFunction(format!(
                "write_i32_le: offset {} + 4 exceeds context length {}",
                offset,
                self.data.len()
            )));
        }
        self.data[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_roundtrip() {
        let mut c = QFunctionContext::new(16);
        c.write_f64_le(0, 3.25).unwrap();
        c.write_f64_le(8, -1.0).unwrap();
        assert!((c.read_f64_le(0).unwrap() - 3.25).abs() < 1e-15);
        assert!((c.read_f64_le(8).unwrap() + 1.0).abs() < 1e-15);
    }

    #[test]
    fn f32_i32_roundtrip() {
        let mut c = QFunctionContext::new(12);
        c.write_f32_le(0, 1.25).unwrap();
        c.write_i32_le(4, -7).unwrap();
        c.write_i32_le(8, 0x1234_5678_u32 as i32).unwrap();
        assert!((c.read_f32_le(0).unwrap() - 1.25).abs() < 1e-6);
        assert_eq!(c.read_i32_le(4).unwrap(), -7);
        assert_eq!(c.read_i32_le(8).unwrap(), 0x1234_5678_u32 as i32);
    }
}
