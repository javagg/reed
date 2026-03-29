use reed_core::{basis::BasisTrait, enums::EvalMode, error::ReedResult, scalar::Scalar, ReedError};

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
                for elem in 0..num_elem {
                    let u_elem = &u[elem * in_size / num_elem..(elem + 1) * in_size / num_elem];
                    let v_elem = &mut v[elem * out_size / num_elem..(elem + 1) * out_size / num_elem];
                    self.apply_interp_elem(transpose, u_elem, v_elem);
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
                for elem in 0..num_elem {
                    let u_elem = &u[elem * in_size / num_elem..(elem + 1) * in_size / num_elem];
                    let v_elem = &mut v[elem * out_size / num_elem..(elem + 1) * out_size / num_elem];
                    self.apply_grad_elem(transpose, u_elem, v_elem);
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
                for elem in 0..num_elem {
                    let offset = elem * self.num_qpoints;
                    v[offset..offset + self.num_qpoints].copy_from_slice(&self.weights);
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
        for comp in 0..self.ncomp {
            if transpose {
                for dof in 0..self.num_dof {
                    let mut sum = T::ZERO;
                    for qpt in 0..self.num_qpoints {
                        sum += self.interp_entry(qpt, dof) * u_elem[qpt * self.ncomp + comp];
                    }
                    v_elem[comp * self.num_dof + dof] = sum;
                }
            } else {
                for qpt in 0..self.num_qpoints {
                    let mut sum = T::ZERO;
                    for dof in 0..self.num_dof {
                        sum += self.interp_entry(qpt, dof) * u_elem[comp * self.num_dof + dof];
                    }
                    v_elem[qpt * self.ncomp + comp] = sum;
                }
            }
        }
    }

    fn apply_grad_elem(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        let qcomp = self.ncomp * self.dim;
        for comp in 0..self.ncomp {
            if transpose {
                for dof in 0..self.num_dof {
                    let mut sum = T::ZERO;
                    for qpt in 0..self.num_qpoints {
                        for d in 0..self.dim {
                            sum += self.grad_entry(d, qpt, dof)
                                * u_elem[qpt * qcomp + comp * self.dim + d];
                        }
                    }
                    v_elem[comp * self.num_dof + dof] = sum;
                }
            } else {
                for qpt in 0..self.num_qpoints {
                    for d in 0..self.dim {
                        let mut sum = T::ZERO;
                        for dof in 0..self.num_dof {
                            sum += self.grad_entry(d, qpt, dof) * u_elem[comp * self.num_dof + dof];
                        }
                        v_elem[qpt * qcomp + comp * self.dim + d] = sum;
                    }
                }
            }
        }
    }

    fn interp_entry(&self, qpt: usize, dof: usize) -> T {
        match self.dim {
            1 => self.interp[qpt * self.p + dof],
            2 => {
                let (qx, qy) = decode_2d(qpt, self.q);
                let (px, py) = decode_2d(dof, self.p);
                self.interp[qx * self.p + px] * self.interp[qy * self.p + py]
            }
            3 => {
                let (qx, qy, qz) = decode_3d(qpt, self.q);
                let (px, py, pz) = decode_3d(dof, self.p);
                self.interp[qx * self.p + px]
                    * self.interp[qy * self.p + py]
                    * self.interp[qz * self.p + pz]
            }
            _ => unreachable!(),
        }
    }

    fn grad_entry(&self, direction: usize, qpt: usize, dof: usize) -> T {
        match self.dim {
            1 => self.grad[qpt * self.p + dof],
            2 => {
                let (qx, qy) = decode_2d(qpt, self.q);
                let (px, py) = decode_2d(dof, self.p);
                match direction {
                    0 => self.grad[qx * self.p + px] * self.interp[qy * self.p + py],
                    1 => self.interp[qx * self.p + px] * self.grad[qy * self.p + py],
                    _ => unreachable!(),
                }
            }
            3 => {
                let (qx, qy, qz) = decode_3d(qpt, self.q);
                let (px, py, pz) = decode_3d(dof, self.p);
                match direction {
                    0 => {
                        self.grad[qx * self.p + px]
                            * self.interp[qy * self.p + py]
                            * self.interp[qz * self.p + pz]
                    }
                    1 => {
                        self.interp[qx * self.p + px]
                            * self.grad[qy * self.p + py]
                            * self.interp[qz * self.p + pz]
                    }
                    2 => {
                        self.interp[qx * self.p + px]
                            * self.interp[qy * self.p + py]
                            * self.grad[qz * self.p + pz]
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
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

fn decode_2d(index: usize, n: usize) -> (usize, usize) {
    (index % n, index / n)
}

fn decode_3d(index: usize, n: usize) -> (usize, usize, usize) {
    let plane = n * n;
    let z = index / plane;
    let rem = index % plane;
    let y = rem / n;
    let x = rem % n;
    (x, y, z)
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
}
