use reed_core::{
    basis::BasisTrait,
    elem_restriction::ElemRestrictionTrait,
    enums::{EvalMode, TransposeMode},
    error::ReedResult,
    operator::OperatorTrait,
    qfunction::QFunctionTrait,
    scalar::Scalar,
    vector::VectorTrait,
    QFunctionContext, ReedError,
};

pub enum FieldVector<'a, T: Scalar> {
    Active,
    Passive(&'a dyn VectorTrait<T>),
    None,
}

pub struct OperatorField<'a, T: Scalar> {
    name: String,
    restriction: Option<&'a dyn ElemRestrictionTrait<T>>,
    basis: Option<&'a dyn BasisTrait<T>>,
    vector: FieldVector<'a, T>,
}

#[derive(Clone, Copy)]
struct InputPlan {
    field_index: usize,
    eval_mode: EvalMode,
}

#[derive(Clone, Copy)]
struct OutputPlan {
    field_index: usize,
    eval_mode: EvalMode,
}

pub struct OperatorBuilder<'a, T: Scalar> {
    qfunction: Option<Box<dyn QFunctionTrait<T>>>,
    qfunction_context: Option<QFunctionContext>,
    fields: Vec<OperatorField<'a, T>>,
}

impl<'a, T: Scalar> Default for OperatorBuilder<'a, T> {
    fn default() -> Self {
        Self {
            qfunction: None,
            qfunction_context: None,
            fields: Vec::new(),
        }
    }
}

impl<'a, T: Scalar> OperatorBuilder<'a, T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn qfunction(mut self, qfunction: Box<dyn QFunctionTrait<T>>) -> Self {
        self.qfunction = Some(qfunction);
        self
    }

    /// User [`QFunctionContext`] buffer; byte length must match
    /// [`QFunctionTrait::context_byte_len`] of the configured qfunction (often zero).
    pub fn qfunction_context(mut self, ctx: QFunctionContext) -> Self {
        self.qfunction_context = Some(ctx);
        self
    }

    pub fn field(
        mut self,
        name: impl Into<String>,
        restriction: Option<&'a dyn ElemRestrictionTrait<T>>,
        basis: Option<&'a dyn BasisTrait<T>>,
        vector: FieldVector<'a, T>,
    ) -> Self {
        self.fields.push(OperatorField {
            name: name.into(),
            restriction,
            basis,
            vector,
        });
        self
    }

    pub fn build(self) -> ReedResult<CpuOperator<'a, T>> {
        let qfunction = self
            .qfunction
            .ok_or_else(|| ReedError::Operator("operator builder requires a qfunction".into()))?;
        let ctx_need = qfunction.context_byte_len();
        let qfunction_context = match (self.qfunction_context, ctx_need) {
            (Some(c), need) if c.byte_len() != need => {
                return Err(ReedError::Operator(format!(
                    "QFunctionContext length {} does not match qfunction.context_byte_len() {}",
                    c.byte_len(),
                    need
                )));
            }
            (Some(c), _) => c,
            (None, 0) => QFunctionContext::new(0),
            (None, need) => {
                return Err(ReedError::Operator(format!(
                    "qfunction requires {} byte(s) of QFunctionContext; call .qfunction_context(...)",
                    need
                )));
            }
        };
        let input_plans = qfunction
            .inputs()
            .iter()
            .map(|descriptor| {
                Ok(InputPlan {
                    field_index: CpuOperator::field_index_by_name(&self.fields, &descriptor.name)?,
                    eval_mode: descriptor.eval_mode,
                })
            })
            .collect::<ReedResult<Vec<_>>>()?;
        let output_plans = qfunction
            .outputs()
            .iter()
            .map(|descriptor| {
                Ok(OutputPlan {
                    field_index: CpuOperator::field_index_by_name(&self.fields, &descriptor.name)?,
                    eval_mode: descriptor.eval_mode,
                })
            })
            .collect::<ReedResult<Vec<_>>>()?;
        let num_qfunction_inputs = input_plans.len();
        let num_qfunction_outputs = output_plans.len();
        let num_elem = self
            .fields
            .iter()
            .find_map(|field| {
                field
                    .restriction
                    .map(|restriction| restriction.num_elements())
            })
            .ok_or_else(|| {
                ReedError::Operator(
                    "operator builder requires at least one restricted field".into(),
                )
            })?;
        let num_qpoints = self
            .fields
            .iter()
            .find_map(|field| field.basis.map(|basis| basis.num_qpoints()))
            .or_else(|| {
                self.fields.iter().find_map(|field| {
                    field
                        .restriction
                        .map(|restriction| restriction.num_dof_per_elem())
                })
            })
            .ok_or_else(|| {
                ReedError::Operator(
                    "operator builder requires at least one basis or restriction".into(),
                )
            })?;
        Ok(CpuOperator {
            qfunction,
            qfunction_context,
            fields: self.fields,
            input_plans,
            output_plans,
            num_elem,
            num_qpoints,
            num_qfunction_inputs,
            num_qfunction_outputs,
        })
    }
}

pub struct CpuOperator<'a, T: Scalar> {
    qfunction: Box<dyn QFunctionTrait<T>>,
    qfunction_context: QFunctionContext,
    fields: Vec<OperatorField<'a, T>>,
    input_plans: Vec<InputPlan>,
    output_plans: Vec<OutputPlan>,
    num_elem: usize,
    num_qpoints: usize,
    num_qfunction_inputs: usize,
    num_qfunction_outputs: usize,
}

impl<'a, T: Scalar> CpuOperator<'a, T> {
    fn field_index_by_name(fields: &[OperatorField<'a, T>], name: &str) -> ReedResult<usize> {
        fields
            .iter()
            .position(|field| field.name == name)
            .ok_or_else(|| ReedError::Operator(format!("field '{}' not found", name)))
    }

    fn qpoint_component_count(
        field: &OperatorField<'a, T>,
        eval_mode: EvalMode,
    ) -> ReedResult<usize> {
        match eval_mode {
            EvalMode::None => {
                if let Some(restriction) = field.restriction {
                    Ok(restriction.num_comp())
                } else {
                    Err(ReedError::Operator(format!(
                        "field '{}' without basis requires a restriction to infer component count",
                        field.name
                    )))
                }
            }
            EvalMode::Weight => Ok(1),
            EvalMode::Interp => field.basis.map(|basis| basis.num_comp()).ok_or_else(|| {
                ReedError::Operator(format!("field '{}' requires basis", field.name))
            }),
            EvalMode::Grad => field
                .basis
                .map(|basis| basis.num_comp() * basis.dim())
                .ok_or_else(|| {
                    ReedError::Operator(format!("field '{}' requires basis", field.name))
                }),
            EvalMode::Div => {
                let basis = field.basis.ok_or_else(|| {
                    ReedError::Operator(format!("field '{}' requires basis for Div", field.name))
                })?;
                if basis.num_comp() != basis.dim() {
                    return Err(ReedError::Operator(format!(
                        "field '{}': EvalMode::Div requires basis.num_comp() == basis.dim() (vector field), got comp {} dim {}",
                        field.name,
                        basis.num_comp(),
                        basis.dim()
                    )));
                }
                Ok(1)
            }
            EvalMode::Curl => {
                let basis = field.basis.ok_or_else(|| {
                    ReedError::Operator(format!("field '{}' requires basis for Curl", field.name))
                })?;
                match (basis.dim(), basis.num_comp()) {
                    (2, 2) => Ok(1),
                    (3, 3) => Ok(3),
                    _ => Err(ReedError::Operator(format!(
                        "field '{}': EvalMode::Curl requires (dim, ncomp) = (2, 2) or (3, 3), got dim {} comp {}",
                        field.name,
                        basis.dim(),
                        basis.num_comp()
                    ))),
                }
            }
        }
    }

    fn prepare_input_into(
        &self,
        field: &OperatorField<'a, T>,
        eval_mode: EvalMode,
        active_input: &dyn VectorTrait<T>,
        local_buffer: &mut Vec<T>,
        q_buffer: &mut Vec<T>,
    ) -> ReedResult<()> {
        if matches!(eval_mode, EvalMode::Weight) {
            let basis = field.basis.ok_or_else(|| {
                ReedError::Operator(format!("field '{}' requires basis for Weight", field.name))
            })?;
            q_buffer.resize(self.num_elem * basis.num_qpoints(), T::ZERO);
            basis.apply(self.num_elem, false, EvalMode::Weight, &[], q_buffer)?;
            return Ok(());
        }

        let source = match field.vector {
            FieldVector::Active => active_input.as_slice(),
            FieldVector::Passive(vector) => vector.as_slice(),
            FieldVector::None => {
                return Err(ReedError::Operator(format!(
                    "field '{}' has no vector source",
                    field.name
                )));
            }
        };

        let local = if let Some(restriction) = field.restriction {
            local_buffer.resize(restriction.local_size(), T::ZERO);
            restriction.apply(TransposeMode::NoTranspose, source, local_buffer)?;
            local_buffer.as_slice()
        } else {
            source
        };

        if let Some(basis) = field.basis {
            let qcomp = Self::qpoint_component_count(field, eval_mode)?;
            q_buffer.resize(self.num_elem * basis.num_qpoints() * qcomp, T::ZERO);
            basis.apply(self.num_elem, false, eval_mode, local, q_buffer)?;
        } else {
            q_buffer.clear();
            q_buffer.extend_from_slice(local);
        }
        Ok(())
    }

    fn scatter_output_to_slice(
        &self,
        field: &OperatorField<'a, T>,
        eval_mode: EvalMode,
        q_output: &[T],
        local_buffer: &mut Vec<T>,
        active_output: &mut [T],
    ) -> ReedResult<()> {
        let local = if let Some(basis) = field.basis {
            local_buffer.resize(self.num_elem * basis.num_dof() * basis.num_comp(), T::ZERO);
            basis.apply(self.num_elem, true, eval_mode, q_output, local_buffer)?;
            local_buffer.as_slice()
        } else {
            q_output
        };

        match field.vector {
            FieldVector::Active => {
                if let Some(restriction) = field.restriction {
                    restriction.apply(TransposeMode::Transpose, &local, active_output)
                } else {
                    if active_output.len() != local.len() {
                        return Err(ReedError::Operator(format!(
                            "output length {} != local length {} for field '{}'",
                            active_output.len(),
                            local.len(),
                            field.name
                        )));
                    }
                    #[cfg(feature = "parallel")]
                    {
                        use rayon::prelude::*;
                        active_output
                            .par_iter_mut()
                            .zip(local.par_iter())
                            .for_each(|(dst, src)| *dst += *src);
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        for (dst, src) in active_output.iter_mut().zip(local.iter()) {
                            *dst += *src;
                        }
                    }
                    Ok(())
                }
            }
            FieldVector::Passive(_) | FieldVector::None => Err(ReedError::Operator(format!(
                "output field '{}' must be active",
                field.name
            ))),
        }
    }

    fn execute(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
        add: bool,
    ) -> ReedResult<()> {
        if !add {
            output.set_value(T::ZERO)?;
        }
        let output_slice = output.as_mut_slice();

        // Allocate workspace buffers on each call to avoid Mutex overhead
        // The allocation cost is negligible compared to the compute work
        let mut q_inputs: Vec<Vec<T>> =
            (0..self.num_qfunction_inputs).map(|_| Vec::new()).collect();
        let mut q_outputs: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();
        let mut input_locals: Vec<Vec<T>> =
            (0..self.num_qfunction_inputs).map(|_| Vec::new()).collect();
        let mut output_locals: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();

        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            self.prepare_input_into(
                field,
                plan.eval_mode,
                input,
                &mut input_locals[slot],
                &mut q_inputs[slot],
            )?;
        }

        for (slot, descriptor) in self.qfunction.outputs().iter().enumerate() {
            q_outputs[slot].resize(
                self.num_elem * self.num_qpoints * descriptor.num_comp,
                T::ZERO,
            );
        }

        let input_slices = q_inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut output_slices = q_outputs
            .iter_mut()
            .map(Vec::as_mut_slice)
            .collect::<Vec<_>>();
        self.qfunction.apply(
            self.qfunction_context.as_bytes(),
            self.num_elem * self.num_qpoints,
            &input_slices,
            &mut output_slices,
        )?;

        for (slot, plan) in self.output_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            self.scatter_output_to_slice(
                field,
                plan.eval_mode,
                &q_outputs[slot],
                &mut output_locals[slot],
                output_slice,
            )?;
        }
        Ok(())
    }
}

impl<'a, T: Scalar> OperatorTrait<T> for CpuOperator<'a, T> {
    fn apply(&self, input: &dyn VectorTrait<T>, output: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        self.execute(input, output, false)
    }

    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        self.execute(input, output, true)
    }

    fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        assembled.set_value(T::ZERO)?;
        let n = assembled.len();
        for i in 0..n {
            let mut input = vec![T::ZERO; n];
            input[i] = T::ONE;
            let x = crate::vector::CpuVector::from_vec(input);
            let mut y = crate::vector::CpuVector::new(n);
            self.apply(&x, &mut y)?;
            assembled.as_mut_slice()[i] = y.as_slice()[i];
        }
        Ok(())
    }
}
