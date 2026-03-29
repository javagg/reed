use reed_core::{
    enums::NormType, error::ReedResult, scalar::Scalar, vector::VectorTrait, ReedError,
};

/// CPU 向量实现，内部存储 `Vec<T>`
pub struct CpuVector<T: Scalar> {
    data: Vec<T>,
}

impl<T: Scalar> CpuVector<T> {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![T::ZERO; size],
        }
    }

    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: Scalar> VectorTrait<T> for CpuVector<T> {
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
        for x in self.data.iter_mut() {
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
        let x_slice = x.as_slice();
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            self.data
                .par_iter_mut()
                .zip(x_slice.par_iter())
                .for_each(|(yi, &xi)| {
                    *yi += alpha * xi;
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for (yi, &xi) in self.data.iter_mut().zip(x_slice.iter()) {
                *yi += alpha * xi;
            }
        }
        Ok(())
    }

    fn scale(&mut self, alpha: T) -> ReedResult<()> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            self.data.par_iter_mut().for_each(|x| *x *= alpha);
        }
        #[cfg(not(feature = "parallel"))]
        {
            for x in self.data.iter_mut() {
                *x *= alpha;
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_basic() {
        let mut v = CpuVector::<f64>::new(5);
        assert_eq!(v.len(), 5);
        v.set_value(3.0).unwrap();
        assert_eq!(v.data(), &[3.0, 3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_vector_copy() {
        let mut v = CpuVector::<f64>::new(3);
        v.copy_from_slice(&[1.0, 2.0, 3.0]).unwrap();
        let mut out = [0.0; 3];
        v.copy_to_slice(&mut out).unwrap();
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_axpy() {
        let mut y = CpuVector::from_vec(vec![1.0, 2.0, 3.0]);
        let x = CpuVector::from_vec(vec![10.0, 20.0, 30.0]);
        y.axpy(2.0, &x).unwrap();
        assert_eq!(y.data(), &[21.0, 42.0, 63.0]);
    }

    #[test]
    fn test_vector_scale() {
        let mut v = CpuVector::from_vec(vec![1.0, 2.0, 3.0]);
        v.scale(2.0).unwrap();
        assert_eq!(v.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vector_norm() {
        let v = CpuVector::from_vec(vec![3.0_f64, -4.0]);
        assert!((v.norm(NormType::One).unwrap() - 7.0).abs() < 1e-14);
        assert!((v.norm(NormType::Two).unwrap() - 5.0).abs() < 1e-14);
        assert!((v.norm(NormType::Max).unwrap() - 4.0).abs() < 1e-14);
    }
}
