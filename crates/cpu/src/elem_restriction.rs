use reed_core::{
    elem_restriction::ElemRestrictionTrait, enums::TransposeMode, error::ReedResult,
    scalar::Scalar, ReedError,
};

#[derive(Clone)]
enum RestrictionLayout {
    Offset {
        offsets: Vec<i32>,
        compstride: usize,
    },
    Strided {
        strides: [i32; 3],
    },
}

#[derive(Clone)]
pub struct CpuElemRestriction<T: Scalar> {
    nelem: usize,
    elemsize: usize,
    ncomp: usize,
    lsize: usize,
    layout: RestrictionLayout,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> CpuElemRestriction<T> {
    pub fn new_offset(
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i32],
    ) -> ReedResult<Self> {
        if offsets.len() != nelem * elemsize {
            return Err(ReedError::InvalidArgument(format!(
                "offsets length {} != nelem*elemsize {}",
                offsets.len(),
                nelem * elemsize
            )));
        }
        Ok(Self {
            nelem,
            elemsize,
            ncomp,
            lsize,
            layout: RestrictionLayout::Offset {
                offsets: offsets.to_vec(),
                compstride,
            },
            _marker: std::marker::PhantomData,
        })
    }

    pub fn new_strided(
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
    ) -> ReedResult<Self> {
        Ok(Self {
            nelem,
            elemsize,
            ncomp,
            lsize,
            layout: RestrictionLayout::Strided { strides },
            _marker: std::marker::PhantomData,
        })
    }

    fn global_index(&self, elem: usize, comp: usize, local: usize) -> ReedResult<usize> {
        let idx = match &self.layout {
            RestrictionLayout::Offset {
                offsets,
                compstride,
            } => {
                let base = offsets[elem * self.elemsize + local];
                if base < 0 {
                    return Err(ReedError::ElemRestriction(format!(
                        "negative offset {} at element {}, local {}",
                        base, elem, local
                    )));
                }
                base as usize + comp * *compstride
            }
            RestrictionLayout::Strided { strides } => {
                let index =
                    local as i32 * strides[0] + comp as i32 * strides[1] + elem as i32 * strides[2];
                if index < 0 {
                    return Err(ReedError::ElemRestriction(format!(
                        "negative strided index {} at element {}, comp {}, local {}",
                        index, elem, comp, local
                    )));
                }
                index as usize
            }
        };
        if idx >= self.lsize {
            return Err(ReedError::ElemRestriction(format!(
                "global index {} out of bounds for lsize {}",
                idx, self.lsize
            )));
        }
        Ok(idx)
    }

    fn local_index(&self, elem: usize, comp: usize, local: usize) -> usize {
        ((elem * self.ncomp + comp) * self.elemsize) + local
    }
}

impl<T: Scalar> ElemRestrictionTrait<T> for CpuElemRestriction<T> {
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
        let local_size = self.local_size();
        match t_mode {
            TransposeMode::NoTranspose => {
                if u.len() != self.lsize {
                    return Err(ReedError::ElemRestriction(format!(
                        "input length {} != global size {}",
                        u.len(),
                        self.lsize
                    )));
                }
                if v.len() != local_size {
                    return Err(ReedError::ElemRestriction(format!(
                        "output length {} != local size {}",
                        v.len(),
                        local_size
                    )));
                }
                for elem in 0..self.nelem {
                    for comp in 0..self.ncomp {
                        for local in 0..self.elemsize {
                            let g = self.global_index(elem, comp, local)?;
                            let l = self.local_index(elem, comp, local);
                            v[l] = u[g];
                        }
                    }
                }
            }
            TransposeMode::Transpose => {
                if u.len() != local_size {
                    return Err(ReedError::ElemRestriction(format!(
                        "input length {} != local size {}",
                        u.len(),
                        local_size
                    )));
                }
                if v.len() != self.lsize {
                    return Err(ReedError::ElemRestriction(format!(
                        "output length {} != global size {}",
                        v.len(),
                        self.lsize
                    )));
                }
                for elem in 0..self.nelem {
                    for comp in 0..self.ncomp {
                        for local in 0..self.elemsize {
                            let g = self.global_index(elem, comp, local)?;
                            let l = self.local_index(elem, comp, local);
                            v[g] += u[l];
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reed_core::enums::TransposeMode;

    #[test]
    fn test_offset_restriction_roundtrip() {
        let r = CpuElemRestriction::<f64>::new_offset(2, 2, 1, 1, 3, &[0, 1, 1, 2]).unwrap();
        let global = vec![10.0, 20.0, 30.0];
        let mut local = vec![0.0; 4];
        r.apply(TransposeMode::NoTranspose, &global, &mut local)
            .unwrap();
        assert_eq!(local, vec![10.0, 20.0, 20.0, 30.0]);

        let mut gathered = vec![0.0; 3];
        r.apply(TransposeMode::Transpose, &local, &mut gathered)
            .unwrap();
        assert_eq!(gathered, vec![10.0, 40.0, 30.0]);
    }
}
