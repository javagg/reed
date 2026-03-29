use reed_core::{
    enums::EvalMode,
    error::ReedResult,
    qfunction::{QFunctionField, QFunctionTrait},
    ReedError,
};

pub struct Poisson1DApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for Poisson1DApply {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "du".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Grad,
                },
                QFunctionField {
                    name: "qdata".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::None,
                },
            ],
            outputs: vec![QFunctionField {
                name: "dv".into(),
                num_comp: 1,
                eval_mode: EvalMode::Grad,
            }],
        }
    }
}

impl QFunctionTrait<f64> for Poisson1DApply {
    fn inputs(&self) -> &[QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[QFunctionField] {
        &self.outputs
    }

    fn apply(&self, q: usize, inputs: &[&[f64]], outputs: &mut [&mut [f64]]) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Poisson1DApply expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        for i in 0..q {
            dv[i] = du[i] * qdata[i];
        }
        Ok(())
    }
}
