//! Minimal `CompositeOperator` demo (libCEED `CeedCompositeOperator`-style additive apply).
//!
//! Uses two simple scaling sub-operators on `R^n` so the composite needs no borrowed mesh data
//! (`Box<dyn OperatorTrait>` is `'static`). For **assembled** `CpuOperator`s that reference
//! restrictions/bases, lifetimes are tied to those objects; build the composite in the same
//! scope as the mesh (see integration tests) or use patterns that yield `'static` operators.

use reed::{OperatorTrait, Reed, ReedResult, VectorTrait};

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

    fn linear_assemble_diagonal(
        &self,
        assembled: &mut dyn VectorTrait<f64>,
    ) -> ReedResult<()> {
        assembled.set_value(0.0)?;
        for i in 0..self.n {
            assembled.as_mut_slice()[i] = self.scale;
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reed = Reed::<f64>::init("/cpu/self")?;
    let n = 3usize;
    let composite = reed.composite_operator(vec![
        Box::new(ScaleOp { n, scale: 2.0 }),
        Box::new(ScaleOp { n, scale: 3.0 }),
    ])?;

    let x = reed.vector_from_slice(&[1.0_f64, 2.0, 3.0])?;
    let mut y = reed.vector(n)?;
    y.set_value(0.0)?;
    composite.apply(&*x, &mut *y)?;

    println!("y = (2I + 3I) x = 5 x  =>  {:?}", y.as_slice());
    assert_eq!(y.as_slice(), &[5.0, 10.0, 15.0]);

    let mut d = reed.vector(n)?;
    d.set_value(0.0)?;
    composite.linear_assemble_diagonal(&mut *d)?;
    println!("diag composite = sum of diagonals => {:?}", d.as_slice());
    assert_eq!(d.as_slice(), &[5.0, 5.0, 5.0]);
    Ok(())
}
