//! Sum of sub-operators, analogous to libCEED’s `CeedCompositeOperator` additive apply.
//!
//! Each sub-operator must map the same global input/output vector space (same `VectorTrait::len()`).

use reed_core::{
    error::{ReedError, ReedResult},
    operator::OperatorTrait,
    scalar::Scalar,
    vector::VectorTrait,
};

use crate::vector::CpuVector;

/// `y = sum_i A_i x` for [`OperatorTrait::apply`]; [`OperatorTrait::apply_add`] accumulates all sub-operators.
pub struct CompositeOperator<T: Scalar> {
    ops: Vec<Box<dyn OperatorTrait<T>>>,
}

impl<T: Scalar> CompositeOperator<T> {
    pub fn new(ops: Vec<Box<dyn OperatorTrait<T>>>) -> ReedResult<Self> {
        if ops.is_empty() {
            return Err(ReedError::Operator(
                "CompositeOperator requires at least one sub-operator".into(),
            ));
        }
        Ok(Self { ops })
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl<T: Scalar> OperatorTrait<T> for CompositeOperator<T> {
    fn apply(&self, input: &dyn VectorTrait<T>, output: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        output.set_value(T::ZERO)?;
        for op in &self.ops {
            op.apply_add(input, output)?;
        }
        Ok(())
    }

    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        for op in &self.ops {
            op.apply_add(input, output)?;
        }
        Ok(())
    }

    fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        let n = assembled.len();
        assembled.set_value(T::ZERO)?;
        let mut tmp = CpuVector::new(n);
        for op in &self.ops {
            tmp.set_value(T::ZERO)?;
            op.linear_assemble_diagonal(&mut tmp)?;
            for i in 0..n {
                assembled.as_mut_slice()[i] = assembled.as_slice()[i] + tmp.as_slice()[i];
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::CpuVector;

    /// `y += s x`; diagonal of `s I` is `s` on each component.
    struct ScaleOp {
        n: usize,
        scale: f64,
    }

    impl OperatorTrait<f64> for ScaleOp {
        fn apply(
            &self,
            input: &dyn VectorTrait<f64>,
            output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            output.set_value(0.0)?;
            self.apply_add(input, output)
        }

        fn apply_add(
            &self,
            input: &dyn VectorTrait<f64>,
            output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            for i in 0..self.n {
                output.as_mut_slice()[i] += self.scale * input.as_slice()[i];
            }
            Ok(())
        }

        fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<f64>) -> ReedResult<()> {
            assembled.set_value(0.0)?;
            for i in 0..self.n {
                assembled.as_mut_slice()[i] = self.scale;
            }
            Ok(())
        }
    }

    #[test]
    fn composite_new_rejects_empty() {
        assert!(matches!(
            CompositeOperator::<f64>::new(vec![]),
            Err(ReedError::Operator(_))
        ));
    }

    #[test]
    fn composite_apply_sums_suboperators() {
        let n = 3;
        let ops: Vec<Box<dyn OperatorTrait<f64>>> = vec![
            Box::new(ScaleOp { n, scale: 2.0 }),
            Box::new(ScaleOp { n, scale: 3.0 }),
        ];
        let comp = CompositeOperator::new(ops).unwrap();
        let x = CpuVector::from_vec(vec![1.0, 2.0, 3.0]);
        let mut y = CpuVector::new(n);
        comp.apply(&x, &mut y).unwrap();
        assert_eq!(y.as_slice(), &[5.0, 10.0, 15.0]);
    }

    #[test]
    fn composite_diagonal_sums_suboperators() {
        let n = 2;
        let ops: Vec<Box<dyn OperatorTrait<f64>>> = vec![
            Box::new(ScaleOp { n, scale: 1.0 }),
            Box::new(ScaleOp { n, scale: 4.0 }),
        ];
        let comp = CompositeOperator::new(ops).unwrap();
        let mut d = CpuVector::new(n);
        comp.linear_assemble_diagonal(&mut d).unwrap();
        assert_eq!(d.as_slice(), &[5.0, 5.0]);
    }
}
