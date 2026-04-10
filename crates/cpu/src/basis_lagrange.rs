use reed_core::{basis::BasisTrait, enums::EvalMode, error::ReedResult, scalar::Scalar, ReedError};

#[cfg(feature = "parallel")]
const PAR_MIN_ELEMS_PER_TASK: usize = 128;

pub struct LagrangeBasis<T: Scalar> {
    dim: usize,
    ncomp: usize,
    p: usize,
    q: usize,
    num_dof: usize,
    num_qpoints: usize,
    q_ref: Vec<T>,
    weights: Vec<T>,
    interp: Vec<T>,
    grad: Vec<T>,
}

impl<T: Scalar> LagrangeBasis<T> {
    pub fn new(
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: reed_core::QuadMode,
    ) -> ReedResult<Self> {
        if !(1..=3).contains(&dim) {
            return Err(ReedError::Basis(format!(
                "current CPU basis supports dim in 1..=3, got {}",
                dim
            )));
        }
        if p < 2 {
            return Err(ReedError::Basis(format!("p must be >= 2, got {}", p)));
        }
        if q < 1 {
            return Err(ReedError::Basis(format!("q must be >= 1, got {}", q)));
        }

        let nodes = gauss_lobatto_nodes(p)?;
        let (q_ref_f64, weights_f64) = match qmode {
            reed_core::QuadMode::Gauss => gauss_quadrature(q)?,
            reed_core::QuadMode::GaussLobatto => gauss_lobatto_quadrature(q)?,
        };
        let num_dof = p.pow(dim as u32);
        let num_qpoints = q.pow(dim as u32);
        let q_ref_tensor = build_tensor_qref::<T>(&q_ref_f64, dim)?;
        let weights_tensor = build_tensor_weights::<T>(&weights_f64, dim)?;

        let interp = build_interp::<T>(&nodes, &q_ref_f64)?;
        let grad = build_grad::<T>(&nodes, &q_ref_f64)?;

        Ok(Self {
            dim,
            ncomp,
            p,
            q,
            num_dof,
            num_qpoints,
            q_ref: q_ref_tensor,
            weights: weights_tensor,
            interp,
            grad,
        })
    }
}

impl<T: Scalar> BasisTrait<T> for LagrangeBasis<T> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn num_dof(&self) -> usize {
        self.num_dof
    }

    fn num_qpoints(&self) -> usize {
        self.num_qpoints
    }

    fn num_comp(&self) -> usize {
        self.ncomp
    }

    fn apply(
        &self,
        num_elem: usize,
        transpose: bool,
        eval_mode: EvalMode,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<()> {
        match eval_mode {
            EvalMode::Interp => {
                let in_size = if transpose {
                    num_elem * self.num_qpoints * self.ncomp
                } else {
                    num_elem * self.num_dof * self.ncomp
                };
                let out_size = if transpose {
                    num_elem * self.num_dof * self.ncomp
                } else {
                    num_elem * self.num_qpoints * self.ncomp
                };
                if u.len() != in_size || v.len() != out_size {
                    return Err(ReedError::Basis(format!(
                        "interp apply size mismatch: input {}, expected {}; output {}, expected {}",
                        u.len(),
                        in_size,
                        v.len(),
                        out_size
                    )));
                }
                let in_stride = in_size / num_elem;
                let out_stride = out_size / num_elem;
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    u.par_chunks(in_stride)
                        .zip(v.par_chunks_mut(out_stride))
                        .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                        .for_each(|(u_elem, v_elem)| self.apply_interp_elem(transpose, u_elem, v_elem));
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for (u_elem, v_elem) in u.chunks(in_stride).zip(v.chunks_mut(out_stride)) {
                        self.apply_interp_elem(transpose, u_elem, v_elem);
                    }
                }
            }
            EvalMode::Grad => {
                let qcomp = self.ncomp * self.dim;
                let in_size = if transpose {
                    num_elem * self.num_qpoints * qcomp
                } else {
                    num_elem * self.num_dof * self.ncomp
                };
                let out_size = if transpose {
                    num_elem * self.num_dof * self.ncomp
                } else {
                    num_elem * self.num_qpoints * qcomp
                };
                if u.len() != in_size || v.len() != out_size {
                    return Err(ReedError::Basis(format!(
                        "grad apply size mismatch: input {}, expected {}; output {}, expected {}",
                        u.len(),
                        in_size,
                        v.len(),
                        out_size
                    )));
                }
                let in_stride = in_size / num_elem;
                let out_stride = out_size / num_elem;
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    u.par_chunks(in_stride)
                        .zip(v.par_chunks_mut(out_stride))
                        .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                        .for_each(|(u_elem, v_elem)| self.apply_grad_elem(transpose, u_elem, v_elem));
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for (u_elem, v_elem) in u.chunks(in_stride).zip(v.chunks_mut(out_stride)) {
                        self.apply_grad_elem(transpose, u_elem, v_elem);
                    }
                }
            }
            EvalMode::Weight => {
                if transpose {
                    return Err(ReedError::Basis(
                        "weight evaluation does not support transpose".into(),
                    ));
                }
                if v.len() != num_elem * self.num_qpoints {
                    return Err(ReedError::Basis(format!(
                        "weight output length {} != expected {}",
                        v.len(),
                        num_elem * self.num_qpoints
                    )));
                }
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    v.par_chunks_mut(self.num_qpoints)
                        .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                        .for_each(|v_elem| v_elem.copy_from_slice(&self.weights));
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for v_elem in v.chunks_mut(self.num_qpoints) {
                        v_elem.copy_from_slice(&self.weights);
                    }
                }
            }
            other => {
                return Err(ReedError::Basis(format!(
                    "eval mode {:?} not implemented in CPU basis",
                    other
                )));
            }
        }
        Ok(())
    }

    fn q_weights(&self) -> &[T] {
        &self.weights
    }

    fn q_ref(&self) -> &[T] {
        &self.q_ref
    }
}

impl<T: Scalar> LagrangeBasis<T> {
    fn apply_interp_elem(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        match self.dim {
            1 => self.apply_interp_elem_1d(transpose, u_elem, v_elem),
            2 => self.apply_interp_elem_2d(transpose, u_elem, v_elem),
            3 => self.apply_interp_elem_3d(transpose, u_elem, v_elem),
            _ => unreachable!(),
        }
    }

    fn apply_grad_elem(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        match self.dim {
            1 => self.apply_grad_elem_1d(transpose, u_elem, v_elem),
            2 => self.apply_grad_elem_2d(transpose, u_elem, v_elem),
            3 => self.apply_grad_elem_3d(transpose, u_elem, v_elem),
            _ => unreachable!(),
        }
    }

    fn apply_interp_elem_1d(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        for comp in 0..self.ncomp {
            if transpose {
                for px in 0..self.p {
                    let mut sum = T::ZERO;
                    for qx in 0..self.q {
                        sum += self.interp[qx * self.p + px] * u_elem[qx * self.ncomp + comp];
                    }
                    v_elem[comp * self.p + px] = sum;
                }
            } else {
                for qx in 0..self.q {
                    let mut sum = T::ZERO;
                    for px in 0..self.p {
                        sum += self.interp[qx * self.p + px] * u_elem[comp * self.p + px];
                    }
                    v_elem[qx * self.ncomp + comp] = sum;
                }
            }
        }
    }

    fn apply_interp_elem_2d(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        let qp = self.q * self.p;
        let qq = self.q * self.q;
        let pp = self.p * self.p;
        let mut tmp = vec![T::ZERO; qp];

        for comp in 0..self.ncomp {
            if transpose {
                tmp.fill(T::ZERO);
                for py in 0..self.p {
                    for qx in 0..self.q {
                        let mut sum = T::ZERO;
                        for qy in 0..self.q {
                            sum += self.interp[qy * self.p + py]
                                * u_elem[(qy * self.q + qx) * self.ncomp + comp];
                        }
                        tmp[py * self.q + qx] = sum;
                    }
                }
                for py in 0..self.p {
                    for px in 0..self.p {
                        let mut sum = T::ZERO;
                        for qx in 0..self.q {
                            sum += self.interp[qx * self.p + px] * tmp[py * self.q + qx];
                        }
                        v_elem[comp * pp + py * self.p + px] = sum;
                    }
                }
            } else {
                let u_comp = &u_elem[comp * pp..(comp + 1) * pp];
                for py in 0..self.p {
                    for qx in 0..self.q {
                        let mut sum = T::ZERO;
                        for px in 0..self.p {
                            sum += self.interp[qx * self.p + px] * u_comp[py * self.p + px];
                        }
                        tmp[py * self.q + qx] = sum;
                    }
                }
                for qy in 0..self.q {
                    for qx in 0..self.q {
                        let mut sum = T::ZERO;
                        for py in 0..self.p {
                            sum += self.interp[qy * self.p + py] * tmp[py * self.q + qx];
                        }
                        v_elem[(qy * self.q + qx) * self.ncomp + comp] = sum;
                    }
                }
            }
        }

        let _ = qq;
    }

    fn apply_interp_elem_3d(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        let p2 = self.p * self.p;
        let q2 = self.q * self.q;
        let ppp = p2 * self.p;
        let qqq = q2 * self.q;
        let mut tmp_x = vec![T::ZERO; self.q * p2];
        let mut tmp_xy = vec![T::ZERO; q2 * self.p];

        for comp in 0..self.ncomp {
            if transpose {
                tmp_xy.fill(T::ZERO);
                for pz in 0..self.p {
                    for py in 0..self.p {
                        for qx in 0..self.q {
                            let mut sum = T::ZERO;
                            for qz in 0..self.q {
                                for qy in 0..self.q {
                                    let qpt = (qz * q2) + (qy * self.q) + qx;
                                    sum += self.interp[qz * self.p + pz]
                                        * self.interp[qy * self.p + py]
                                        * u_elem[qpt * self.ncomp + comp];
                                }
                            }
                            tmp_xy[(pz * self.p + py) * self.q + qx] = sum;
                        }
                    }
                }
                for pz in 0..self.p {
                    for py in 0..self.p {
                        for px in 0..self.p {
                            let mut sum = T::ZERO;
                            for qx in 0..self.q {
                                sum += self.interp[qx * self.p + px]
                                    * tmp_xy[(pz * self.p + py) * self.q + qx];
                            }
                            v_elem[comp * ppp + (pz * p2 + py * self.p + px)] = sum;
                        }
                    }
                }
            } else {
                let u_comp = &u_elem[comp * ppp..(comp + 1) * ppp];
                for pz in 0..self.p {
                    for py in 0..self.p {
                        for qx in 0..self.q {
                            let mut sum = T::ZERO;
                            for px in 0..self.p {
                                sum += self.interp[qx * self.p + px]
                                    * u_comp[pz * p2 + py * self.p + px];
                            }
                            tmp_x[(pz * self.p + py) * self.q + qx] = sum;
                        }
                    }
                }
                for pz in 0..self.p {
                    for qy in 0..self.q {
                        for qx in 0..self.q {
                            let mut sum = T::ZERO;
                            for py in 0..self.p {
                                sum += self.interp[qy * self.p + py]
                                    * tmp_x[(pz * self.p + py) * self.q + qx];
                            }
                            tmp_xy[(pz * q2) + (qy * self.q) + qx] = sum;
                        }
                    }
                }
                for qz in 0..self.q {
                    for qy in 0..self.q {
                        for qx in 0..self.q {
                            let mut sum = T::ZERO;
                            for pz in 0..self.p {
                                sum += self.interp[qz * self.p + pz]
                                    * tmp_xy[(pz * q2) + (qy * self.q) + qx];
                            }
                            v_elem[((qz * q2) + (qy * self.q) + qx) * self.ncomp + comp] = sum;
                        }
                    }
                }
            }
        }

        let _ = qqq;
    }

    fn apply_grad_elem_1d(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        for comp in 0..self.ncomp {
            if transpose {
                for px in 0..self.p {
                    let mut sum = T::ZERO;
                    for qx in 0..self.q {
                        sum += self.grad[qx * self.p + px] * u_elem[qx * self.ncomp + comp];
                    }
                    v_elem[comp * self.p + px] = sum;
                }
            } else {
                for qx in 0..self.q {
                    let mut sum = T::ZERO;
                    for px in 0..self.p {
                        sum += self.grad[qx * self.p + px] * u_elem[comp * self.p + px];
                    }
                    v_elem[qx * self.ncomp + comp] = sum;
                }
            }
        }
    }

    fn apply_grad_elem_2d(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        let qcomp = self.ncomp * 2;
        let pp = self.p * self.p;
        let mut tmp_interp_x = vec![T::ZERO; self.q * self.p];
        let mut tmp_grad_x = vec![T::ZERO; self.q * self.p];

        for comp in 0..self.ncomp {
            if transpose {
                let mut accum_x = vec![T::ZERO; pp];
                let mut accum_y = vec![T::ZERO; pp];

                for py in 0..self.p {
                    for qx in 0..self.q {
                        let mut sum_x = T::ZERO;
                        let mut sum_y = T::ZERO;
                        for qy in 0..self.q {
                            let base = (qy * self.q + qx) * qcomp + comp * 2;
                            sum_x += self.interp[qy * self.p + py] * u_elem[base];
                            sum_y += self.grad[qy * self.p + py] * u_elem[base + 1];
                        }
                        tmp_grad_x[py * self.q + qx] = sum_x;
                        tmp_interp_x[py * self.q + qx] = sum_y;
                    }
                }

                for py in 0..self.p {
                    for px in 0..self.p {
                        let mut sum_x = T::ZERO;
                        let mut sum_y = T::ZERO;
                        for qx in 0..self.q {
                            sum_x += self.grad[qx * self.p + px] * tmp_grad_x[py * self.q + qx];
                            sum_y += self.interp[qx * self.p + px] * tmp_interp_x[py * self.q + qx];
                        }
                        let dst = comp * pp + py * self.p + px;
                        accum_x[py * self.p + px] = sum_x;
                        accum_y[py * self.p + px] = sum_y;
                        v_elem[dst] = accum_x[py * self.p + px] + accum_y[py * self.p + px];
                    }
                }
            } else {
                let u_comp = &u_elem[comp * pp..(comp + 1) * pp];
                for py in 0..self.p {
                    for qx in 0..self.q {
                        let mut sum_interp = T::ZERO;
                        let mut sum_grad = T::ZERO;
                        for px in 0..self.p {
                            let value = u_comp[py * self.p + px];
                            sum_interp += self.interp[qx * self.p + px] * value;
                            sum_grad += self.grad[qx * self.p + px] * value;
                        }
                        tmp_interp_x[py * self.q + qx] = sum_interp;
                        tmp_grad_x[py * self.q + qx] = sum_grad;
                    }
                }

                for qy in 0..self.q {
                    for qx in 0..self.q {
                        let mut dx = T::ZERO;
                        let mut dy = T::ZERO;
                        for py in 0..self.p {
                            dx += self.interp[qy * self.p + py] * tmp_grad_x[py * self.q + qx];
                            dy += self.grad[qy * self.p + py] * tmp_interp_x[py * self.q + qx];
                        }
                        let dst = (qy * self.q + qx) * qcomp + comp * 2;
                        v_elem[dst] = dx;
                        v_elem[dst + 1] = dy;
                    }
                }
            }
        }
    }

    fn apply_grad_elem_3d(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        let qcomp = self.ncomp * 3;
        let p2 = self.p * self.p;
        let q2 = self.q * self.q;
        let ppp = p2 * self.p;
        let mut tmp_x = vec![T::ZERO; self.q * p2];
        let mut tmp_y = vec![T::ZERO; q2 * self.p];
        let mut accum = vec![T::ZERO; ppp];

        for comp in 0..self.ncomp {
            if transpose {
                accum.fill(T::ZERO);

                for direction in 0..3 {
                    for pz in 0..self.p {
                        for qy in 0..self.q {
                            for qx in 0..self.q {
                                let mut sum = T::ZERO;
                                for qz in 0..self.q {
                                    let base = ((qz * q2) + (qy * self.q) + qx) * qcomp + comp * 3 + direction;
                                    let bz = if direction == 2 {
                                        self.grad[qz * self.p + pz]
                                    } else {
                                        self.interp[qz * self.p + pz]
                                    };
                                    sum += bz * u_elem[base];
                                }
                                tmp_y[(pz * q2) + (qy * self.q) + qx] = sum;
                            }
                        }
                    }

                    for pz in 0..self.p {
                        for py in 0..self.p {
                            for qx in 0..self.q {
                                let mut sum = T::ZERO;
                                for qy in 0..self.q {
                                    let by = if direction == 1 {
                                        self.grad[qy * self.p + py]
                                    } else {
                                        self.interp[qy * self.p + py]
                                    };
                                    sum += by * tmp_y[(pz * q2) + (qy * self.q) + qx];
                                }
                                tmp_x[(pz * self.p + py) * self.q + qx] = sum;
                            }
                        }
                    }

                    for pz in 0..self.p {
                        for py in 0..self.p {
                            for px in 0..self.p {
                                let mut sum = T::ZERO;
                                for qx in 0..self.q {
                                    let bx = if direction == 0 {
                                        self.grad[qx * self.p + px]
                                    } else {
                                        self.interp[qx * self.p + px]
                                    };
                                    sum += bx * tmp_x[(pz * self.p + py) * self.q + qx];
                                }
                                accum[pz * p2 + py * self.p + px] += sum;
                            }
                        }
                    }
                }

                let dst = &mut v_elem[comp * ppp..(comp + 1) * ppp];
                dst.copy_from_slice(&accum);
            } else {
                let u_comp = &u_elem[comp * ppp..(comp + 1) * ppp];
                for direction in 0..3 {
                    for pz in 0..self.p {
                        for py in 0..self.p {
                            for qx in 0..self.q {
                                let mut sum = T::ZERO;
                                for px in 0..self.p {
                                    let bx = if direction == 0 {
                                        self.grad[qx * self.p + px]
                                    } else {
                                        self.interp[qx * self.p + px]
                                    };
                                    sum += bx * u_comp[pz * p2 + py * self.p + px];
                                }
                                tmp_x[(pz * self.p + py) * self.q + qx] = sum;
                            }
                        }
                    }

                    for pz in 0..self.p {
                        for qy in 0..self.q {
                            for qx in 0..self.q {
                                let mut sum = T::ZERO;
                                for py in 0..self.p {
                                    let by = if direction == 1 {
                                        self.grad[qy * self.p + py]
                                    } else {
                                        self.interp[qy * self.p + py]
                                    };
                                    sum += by * tmp_x[(pz * self.p + py) * self.q + qx];
                                }
                                tmp_y[(pz * q2) + (qy * self.q) + qx] = sum;
                            }
                        }
                    }

                    for qz in 0..self.q {
                        for qy in 0..self.q {
                            for qx in 0..self.q {
                                let mut sum = T::ZERO;
                                for pz in 0..self.p {
                                    let bz = if direction == 2 {
                                        self.grad[qz * self.p + pz]
                                    } else {
                                        self.interp[qz * self.p + pz]
                                    };
                                    sum += bz * tmp_y[(pz * q2) + (qy * self.q) + qx];
                                }
                                let dst = ((qz * q2) + (qy * self.q) + qx) * qcomp + comp * 3 + direction;
                                v_elem[dst] = sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn tensor_contract<T: Scalar>(
    b: &[T],
    u: &[T],
    v: &mut [T],
    q: usize,
    p: usize,
    transpose: bool,
) {
    if transpose {
        for pi in 0..p {
            let mut sum = T::ZERO;
            for qi in 0..q {
                sum += b[qi * p + pi] * u[qi];
            }
            v[pi] = sum;
        }
    } else {
        for qi in 0..q {
            let mut sum = T::ZERO;
            for pi in 0..p {
                sum += b[qi * p + pi] * u[pi];
            }
            v[qi] = sum;
        }
    }
}

fn to_scalar<T: Scalar>(value: f64) -> ReedResult<T> {
    T::from(value).ok_or_else(|| ReedError::Basis(format!("failed to convert {} to scalar", value)))
}

fn legendre(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    let mut pnm1 = 1.0;
    let mut pn = x;
    for k in 2..=n {
        let kf = k as f64;
        let pk = ((2.0 * kf - 1.0) * x * pn - (kf - 1.0) * pnm1) / kf;
        pnm1 = pn;
        pn = pk;
    }
    let dp = (n as f64) * (x * pn - pnm1) / (x * x - 1.0);
    (pn, dp)
}

pub fn gauss_quadrature(n: usize) -> ReedResult<(Vec<f64>, Vec<f64>)> {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let m = n.div_ceil(2);
    for i in 0..m {
        let nf = n as f64;
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        for _ in 0..100 {
            let (pn, dpn) = legendre(n, x);
            let dx = -pn / dpn;
            x += dx;
            if dx.abs() < 1.0e-14 {
                break;
            }
        }
        let (_, dpn) = legendre(n, x);
        let w = 2.0 / ((1.0 - x * x) * dpn * dpn);
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    Ok((nodes, weights))
}

pub fn gauss_lobatto_nodes(n: usize) -> ReedResult<Vec<f64>> {
    if n < 2 {
        return Err(ReedError::Basis(format!(
            "gauss_lobatto_nodes requires n>=2, got {}",
            n
        )));
    }
    if n == 2 {
        return Ok(vec![-1.0, 1.0]);
    }

    let mut nodes = vec![-1.0; n];
    nodes[n - 1] = 1.0;
    for (i, node) in nodes.iter_mut().enumerate().take(n - 1).skip(1) {
        let mut x = -(std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos();
        for _ in 0..100 {
            let (pnm1, dpnm1) = legendre(n - 1, x);
            let ddpnm1 = (2.0 * x * dpnm1 - (n as f64 - 1.0) * n as f64 * pnm1) / (1.0 - x * x);
            let dx = -dpnm1 / ddpnm1;
            x += dx;
            if dx.abs() < 1.0e-14 {
                break;
            }
        }
        *node = x;
    }
    Ok(nodes)
}

pub fn gauss_lobatto_quadrature(n: usize) -> ReedResult<(Vec<f64>, Vec<f64>)> {
    let nodes = gauss_lobatto_nodes(n)?;
    let mut weights = vec![0.0; n];
    let nn = n as f64;
    for i in 0..n {
        let (pnm1, _) = legendre(n - 1, nodes[i]);
        weights[i] = 2.0 / (nn * (nn - 1.0) * pnm1 * pnm1);
    }
    Ok((nodes, weights))
}

fn barycentric_weights(nodes: &[f64]) -> Vec<f64> {
    let mut weights = vec![1.0; nodes.len()];
    for j in 0..nodes.len() {
        let mut w = 1.0;
        for (k, &xk) in nodes.iter().enumerate() {
            if j != k {
                w *= nodes[j] - xk;
            }
        }
        weights[j] = 1.0 / w;
    }
    weights
}

fn build_interp<T: Scalar>(nodes: &[f64], qref: &[f64]) -> ReedResult<Vec<T>> {
    let bary = barycentric_weights(nodes);
    let mut interp = Vec::with_capacity(qref.len() * nodes.len());
    for &x in qref {
        let mut exact = None;
        for (j, &node) in nodes.iter().enumerate() {
            if (x - node).abs() < 1.0e-14 {
                exact = Some(j);
                break;
            }
        }
        if let Some(index) = exact {
            for j in 0..nodes.len() {
                interp.push(to_scalar::<T>(if j == index { 1.0 } else { 0.0 })?);
            }
            continue;
        }
        let denom: f64 = nodes
            .iter()
            .enumerate()
            .map(|(j, &node)| bary[j] / (x - node))
            .sum();
        for (j, &node) in nodes.iter().enumerate() {
            interp.push(to_scalar::<T>((bary[j] / (x - node)) / denom)?);
        }
    }
    Ok(interp)
}

fn build_grad<T: Scalar>(nodes: &[f64], qref: &[f64]) -> ReedResult<Vec<T>> {
    let bary = barycentric_weights(nodes);
    let interp = build_interp::<T>(nodes, qref)?;
    let interp_f64 = interp
        .iter()
        .map(|value| value.to_f64().unwrap())
        .collect::<Vec<_>>();
    let mut grad = Vec::with_capacity(qref.len() * nodes.len());
    for (qi, &x) in qref.iter().enumerate() {
        let exact = nodes.iter().position(|&node| (x - node).abs() < 1.0e-14);
        if let Some(j_exact) = exact {
            for i in 0..nodes.len() {
                if i == j_exact {
                    let mut sum = 0.0;
                    for m in 0..nodes.len() {
                        if m != i {
                            sum += 1.0 / (nodes[i] - nodes[m]);
                        }
                    }
                    grad.push(to_scalar::<T>(sum)?);
                } else {
                    grad.push(to_scalar::<T>(
                        bary[i] / (bary[j_exact] * (nodes[j_exact] - nodes[i])),
                    )?);
                }
            }
            continue;
        }
        let s1: f64 = nodes
            .iter()
            .enumerate()
            .map(|(j, &node)| bary[j] / (x - node))
            .sum();
        let s2: f64 = nodes
            .iter()
            .enumerate()
            .map(|(j, &node)| bary[j] / ((x - node) * (x - node)))
            .sum();
        for i in 0..nodes.len() {
            let li = interp_f64[qi * nodes.len() + i];
            let value = li * (s2 / s1 - 1.0 / (x - nodes[i]));
            grad.push(to_scalar::<T>(value)?);
        }
    }
    Ok(grad)
}

fn build_tensor_qref<T: Scalar>(qref_1d: &[f64], dim: usize) -> ReedResult<Vec<T>> {
    let mut q_ref = Vec::with_capacity(qref_1d.len().pow(dim as u32) * dim);
    match dim {
        1 => {
            for &x in qref_1d {
                q_ref.push(to_scalar::<T>(x)?);
            }
        }
        2 => {
            for &y in qref_1d {
                for &x in qref_1d {
                    q_ref.push(to_scalar::<T>(x)?);
                    q_ref.push(to_scalar::<T>(y)?);
                }
            }
        }
        3 => {
            for &z in qref_1d {
                for &y in qref_1d {
                    for &x in qref_1d {
                        q_ref.push(to_scalar::<T>(x)?);
                        q_ref.push(to_scalar::<T>(y)?);
                        q_ref.push(to_scalar::<T>(z)?);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    Ok(q_ref)
}

fn build_tensor_weights<T: Scalar>(weights_1d: &[f64], dim: usize) -> ReedResult<Vec<T>> {
    let mut weights = Vec::with_capacity(weights_1d.len().pow(dim as u32));
    match dim {
        1 => {
            for &w in weights_1d {
                weights.push(to_scalar::<T>(w)?);
            }
        }
        2 => {
            for &wy in weights_1d {
                for &wx in weights_1d {
                    weights.push(to_scalar::<T>(wx * wy)?);
                }
            }
        }
        3 => {
            for &wz in weights_1d {
                for &wy in weights_1d {
                    for &wx in weights_1d {
                        weights.push(to_scalar::<T>(wx * wy * wz)?);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_weights_sum_to_two() {
        let (_, weights) = gauss_quadrature(3).unwrap();
        assert!((weights.iter().sum::<f64>() - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_interp_of_constant() {
        let basis = LagrangeBasis::<f64>::new(1, 1, 3, 4, reed_core::QuadMode::Gauss).unwrap();
        let u = vec![2.0; 3];
        let mut v = vec![0.0; 4];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        for value in v {
            assert!((value - 2.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn test_2d_weights_sum_to_four() {
        let basis = LagrangeBasis::<f64>::new(2, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        assert!((basis.q_weights().iter().sum::<f64>() - 4.0).abs() < 1.0e-12);
        assert_eq!(basis.num_dof(), 4);
        assert_eq!(basis.num_qpoints(), 4);
    }

    #[test]
    fn test_2d_interp_of_constant() {
        let basis = LagrangeBasis::<f64>::new(2, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        let u = vec![3.5; 4];
        let mut v = vec![0.0; 4];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        for value in v {
            assert!((value - 3.5).abs() < 1.0e-12);
        }
    }

    #[test]
    fn test_2d_grad_of_linear_function() {
        let basis = LagrangeBasis::<f64>::new(2, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        let nodes = [-1.0_f64, 1.0];
        let mut u = Vec::new();
        for &y in &nodes {
            for &x in &nodes {
                u.push(2.0 * x - 3.0 * y + 1.0);
            }
        }
        let mut grad = vec![0.0; 4 * 2];
        basis.apply(1, false, EvalMode::Grad, &u, &mut grad).unwrap();
        for qpt in 0..4 {
            assert!((grad[qpt * 2] - 2.0).abs() < 1.0e-12);
            assert!((grad[qpt * 2 + 1] + 3.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn test_3d_weights_sum_to_eight() {
        let basis = LagrangeBasis::<f64>::new(3, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        assert!((basis.q_weights().iter().sum::<f64>() - 8.0).abs() < 1.0e-12);
        assert_eq!(basis.num_dof(), 8);
        assert_eq!(basis.num_qpoints(), 8);
    }

    #[test]
    fn test_3d_grad_of_linear_function() {
        let basis = LagrangeBasis::<f64>::new(3, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        let nodes = [-1.0_f64, 1.0];
        let mut u = Vec::new();
        for &z in &nodes {
            for &y in &nodes {
                for &x in &nodes {
                    u.push(2.0 * x - 3.0 * y + 4.0 * z + 1.0);
                }
            }
        }
        let mut grad = vec![0.0; 8 * 3];
        basis.apply(1, false, EvalMode::Grad, &u, &mut grad).unwrap();
        for qpt in 0..8 {
            assert!((grad[qpt * 3] - 2.0).abs() < 1.0e-12);
            assert!((grad[qpt * 3 + 1] + 3.0).abs() < 1.0e-12);
            assert!((grad[qpt * 3 + 2] - 4.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn test_3d_interp_transpose_matches_naive() {
        let basis = LagrangeBasis::<f64>::new(3, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        let u = vec![
            0.5, -1.0, 2.0, 0.25, 1.5, -0.75, 0.1, 3.0,
        ];
        let mut v = vec![0.0; 8];
        basis.apply(1, true, EvalMode::Interp, &u, &mut v).unwrap();

        let mut expected = vec![0.0; 8];
        for pz in 0..2 {
            for py in 0..2 {
                for px in 0..2 {
                    let mut sum = 0.0;
                    for qz in 0..2 {
                        for qy in 0..2 {
                            for qx in 0..2 {
                                let qpt = (qz * 4) + (qy * 2) + qx;
                                sum += basis.interp[qx * 2 + px]
                                    * basis.interp[qy * 2 + py]
                                    * basis.interp[qz * 2 + pz]
                                    * u[qpt];
                            }
                        }
                    }
                    expected[pz * 4 + py * 2 + px] = sum;
                }
            }
        }

        for (got, want) in v.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-12);
        }
    }

    #[test]
    fn test_3d_grad_transpose_matches_naive() {
        let basis = LagrangeBasis::<f64>::new(3, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        let u = vec![
            0.5, -1.0, 2.0,
            0.25, 1.5, -0.75,
            0.1, 3.0, -2.0,
            1.25, -0.5, 0.75,
            -1.0, 0.2, 1.8,
            2.2, -1.3, 0.4,
            0.6, 0.7, -0.8,
            -0.9, 1.1, 2.4,
        ];
        let mut v = vec![0.0; 8];
        basis.apply(1, true, EvalMode::Grad, &u, &mut v).unwrap();

        let mut expected = vec![0.0; 8];
        for pz in 0..2 {
            for py in 0..2 {
                for px in 0..2 {
                    let mut sum = 0.0;
                    for qz in 0..2 {
                        for qy in 0..2 {
                            for qx in 0..2 {
                                let qpt = (qz * 4) + (qy * 2) + qx;
                                let ux = u[qpt * 3];
                                let uy = u[qpt * 3 + 1];
                                let uz = u[qpt * 3 + 2];
                                sum += basis.grad[qx * 2 + px]
                                    * basis.interp[qy * 2 + py]
                                    * basis.interp[qz * 2 + pz]
                                    * ux;
                                sum += basis.interp[qx * 2 + px]
                                    * basis.grad[qy * 2 + py]
                                    * basis.interp[qz * 2 + pz]
                                    * uy;
                                sum += basis.interp[qx * 2 + px]
                                    * basis.interp[qy * 2 + py]
                                    * basis.grad[qz * 2 + pz]
                                    * uz;
                            }
                        }
                    }
                    expected[pz * 4 + py * 2 + px] = sum;
                }
            }
        }

        for (got, want) in v.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-12);
        }
    }
}
