use reed_core::{
    basis::BasisTrait,
    elem_restriction::ElemRestrictionTrait,
    enums::{EvalMode, TransposeMode},
    error::ReedResult,
    operator::OperatorTrait,
    qfunction::QFunctionTrait,
    scalar::Scalar,
    vector::VectorTrait,
    ReedError,
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

pub struct OperatorBuilder<'a, T: Scalar> {
    qfunction: Option<Box<dyn QFunctionTrait<T>>>,
    fields: Vec<OperatorField<'a, T>>,
}

impl<'a, T: Scalar> Default for OperatorBuilder<'a, T> {
    fn default() -> Self {
        Self {
            qfunction: None,
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
            fields: self.fields,
            num_elem,
            num_qpoints,
        })
    }
}

pub struct CpuOperator<'a, T: Scalar> {
    qfunction: Box<dyn QFunctionTrait<T>>,
    fields: Vec<OperatorField<'a, T>>,
    num_elem: usize,
    num_qpoints: usize,
}

impl<'a, T: Scalar> CpuOperator<'a, T> {
    fn qpoint_component_count(field: &OperatorField<'a, T>, eval_mode: EvalMode) -> ReedResult<usize> {
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
            EvalMode::Interp => field
                .basis
                .map(|basis| basis.num_comp())
                .ok_or_else(|| ReedError::Operator(format!("field '{}' requires basis", field.name))),
            EvalMode::Grad => field
                .basis
                .map(|basis| basis.num_comp() * basis.dim())
                .ok_or_else(|| ReedError::Operator(format!("field '{}' requires basis", field.name))),
            other => Err(ReedError::Operator(format!(
                "eval mode {:?} not implemented in operator sizing",
                other
            ))),
        }
    }

    fn field_by_name(&self, name: &str) -> ReedResult<&OperatorField<'a, T>> {
        self.fields
            .iter()
            .find(|field| field.name == name)
            .ok_or_else(|| ReedError::Operator(format!("field '{}' not found", name)))
    }

    fn prepare_input(
        &self,
        field: &OperatorField<'a, T>,
        eval_mode: EvalMode,
        active_input: &dyn VectorTrait<T>,
    ) -> ReedResult<Vec<T>> {
        if matches!(eval_mode, EvalMode::Weight) {
            let basis = field.basis.ok_or_else(|| {
                ReedError::Operator(format!("field '{}' requires basis for Weight", field.name))
            })?;
            let mut qdata = vec![T::ZERO; self.num_elem * basis.num_qpoints()];
            basis.apply(self.num_elem, false, EvalMode::Weight, &[], &mut qdata)?;
            return Ok(qdata);
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
            let mut local = vec![T::ZERO; restriction.local_size()];
            restriction.apply(TransposeMode::NoTranspose, source, &mut local)?;
            local
        } else {
            source.to_vec()
        };

        if let Some(basis) = field.basis {
            let qcomp = Self::qpoint_component_count(field, eval_mode)?;
            let mut qdata = vec![T::ZERO; self.num_elem * basis.num_qpoints() * qcomp];
            basis.apply(self.num_elem, false, eval_mode, &local, &mut qdata)?;
            Ok(qdata)
        } else {
            Ok(local)
        }
    }

    fn scatter_output(
        &self,
        field: &OperatorField<'a, T>,
        eval_mode: EvalMode,
        q_output: &[T],
        active_output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        let local = if let Some(basis) = field.basis {
            let mut local = vec![T::ZERO; self.num_elem * basis.num_dof() * basis.num_comp()];
            basis.apply(self.num_elem, true, eval_mode, q_output, &mut local)?;
            local
        } else {
            q_output.to_vec()
        };

        match field.vector {
            FieldVector::Active => {
                if let Some(restriction) = field.restriction {
                    restriction.apply(
                        TransposeMode::Transpose,
                        &local,
                        active_output.as_mut_slice(),
                    )
                } else {
                    let out = active_output.as_mut_slice();
                    if out.len() != local.len() {
                        return Err(ReedError::Operator(format!(
                            "output length {} != local length {} for field '{}'",
                            out.len(),
                            local.len(),
                            field.name
                        )));
                    }
                    for (dst, src) in out.iter_mut().zip(local.iter()) {
                        *dst += *src;
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

        let q_inputs = self
            .qfunction
            .inputs()
            .iter()
            .map(|descriptor| {
                let field = self.field_by_name(&descriptor.name)?;
                self.prepare_input(field, descriptor.eval_mode, input)
            })
            .collect::<ReedResult<Vec<_>>>()?;

        let mut q_outputs = self
            .qfunction
            .outputs()
            .iter()
            .map(|descriptor| vec![T::ZERO; self.num_elem * self.num_qpoints * descriptor.num_comp])
            .collect::<Vec<_>>();

        for elem in 0..self.num_elem {
            let input_slices = self
                .qfunction
                .inputs()
                .iter()
                .zip(q_inputs.iter())
                .map(|(descriptor, buffer)| {
                    let per_elem = self.num_qpoints * descriptor.num_comp;
                    let start = elem * per_elem;
                    &buffer[start..start + per_elem]
                })
                .collect::<Vec<_>>();
            let mut output_slices = self
                .qfunction
                .outputs()
                .iter()
                .zip(q_outputs.iter_mut())
                .map(|(descriptor, buffer)| {
                    let per_elem = self.num_qpoints * descriptor.num_comp;
                    let start = elem * per_elem;
                    &mut buffer[start..start + per_elem]
                })
                .collect::<Vec<_>>();
            self.qfunction
                .apply(self.num_qpoints, &input_slices, &mut output_slices)?;
        }

        for (descriptor, q_output) in self.qfunction.outputs().iter().zip(q_outputs.iter()) {
            let field = self.field_by_name(&descriptor.name)?;
            self.scatter_output(field, descriptor.eval_mode, q_output, output)?;
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
