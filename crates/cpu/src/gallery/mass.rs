use reed_core::{
    enums::EvalMode,
    error::ReedResult,
    qfunction::{QFunctionField, QFunctionTrait},
    ReedError,
};

pub struct Mass1DBuild {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for Mass1DBuild {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "dx".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Grad,
                },
                QFunctionField {
                    name: "weights".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Weight,
                },
            ],
            outputs: vec![QFunctionField {
                name: "qdata".into(),
                num_comp: 1,
                eval_mode: EvalMode::None,
            }],
        }
    }
}

impl QFunctionTrait<f64> for Mass1DBuild {
    fn inputs(&self) -> &[QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[QFunctionField] {
        &self.outputs
    }

    fn apply(&self, q: usize, inputs: &[&[f64]], outputs: &mut [&mut [f64]]) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Mass1DBuild expects 2 inputs and 1 output".into(),
            ));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        for i in 0..q {
            qdata[i] = dx[i].abs() * weights[i];
        }
        Ok(())
    }
}

pub struct MassApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for MassApply {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "u".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Interp,
                },
                QFunctionField {
                    name: "qdata".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::None,
                },
            ],
            outputs: vec![QFunctionField {
                name: "v".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
        }
    }
}

impl QFunctionTrait<f64> for MassApply {
    fn inputs(&self) -> &[QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[QFunctionField] {
        &self.outputs
    }

    fn apply(&self, q: usize, inputs: &[&[f64]], outputs: &mut [&mut [f64]]) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "MassApply expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let qdata = inputs[1];
        let v = &mut outputs[0];
        for i in 0..q {
            v[i] = u[i] * qdata[i];
        }
        Ok(())
    }
}
