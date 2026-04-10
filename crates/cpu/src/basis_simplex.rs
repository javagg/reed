//! Simplex basis functions for triangles and tetrahedra.
//!
//! Implements [`BasisTrait`] for H1-conforming Lagrange bases on simplex
//! reference elements:
//!
//! | Type | Topology | DOFs | Exact up to polynomial degree |
//! |------|----------|------|-------------------------------|
//! | P1 triangle | Tri3 | 3 | linear |
//! | P2 triangle | Tri6 | 6 | quadratic |
//! | P1 tet | Tet4 | 4 | linear |
//! | P2 tet | Tet10 | 10 | quadratic |
//!
//! ## Reference elements
//!
//! **Triangle** — vertices (0,0), (1,0), (0,1).
//!
//! **Tetrahedron** — vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1).
//!
//! ## Quadrature rules
//!
//! Pass the desired number of quadrature points via the `q` constructor argument.
//!
//! | `q` (triangle) | degree exact | `q` (tet) | degree exact |
//! |----------------|--------------|-----------|--------------|
//! | 1 | 1 | 1 | 1 |
//! | 3 | 2 | 4 | 2 |
//! | 4 | 3 | 5 | 3 |
//! | 6 | 4 | — | — |
//! | 7 | 5 | — | — |
//!
//! ## Memory layout (matches [`LagrangeBasis`](super::basis_lagrange::LagrangeBasis))
//!
//! * `interp`  — row-major `[nqpts × num_dof]`
//! * `grad`    — row-major `[nqpts × num_dof × dim]`,
//!               stored as `[qpt][dof][d]` ↔ index `(qpt*num_dof + dof)*dim + d`
//!
//! **Element buffers passed to `apply`:**
//!
//! * Forward interp : `u=[ncomp × num_dof]`, `v=[nqpts × ncomp]`
//! * Forward grad   : `u=[ncomp × num_dof]`, `v=[nqpts × ncomp × dim]`
//!   (`qcomp = ncomp*dim`, layout `v[qpt*qcomp + comp*dim + d]`)
//! * Weight         : `v=[nqpts]` (per element, repeated `num_elem` times)

use reed_core::{
    basis::BasisTrait,
    enums::{ElemTopology, EvalMode},
    error::{ReedError, ReedResult},
    scalar::Scalar,
};

#[cfg(feature = "parallel")]
const PAR_MIN_ELEMS_PER_TASK: usize = 128;

// ── SimplexBasis ──────────────────────────────────────────────────────────────

/// H1 Lagrange basis on triangle or tetrahedron reference elements.
pub struct SimplexBasis<T: Scalar> {
    #[allow(dead_code)]
    topo: ElemTopology,
    dim: usize,
    ncomp: usize,
    num_dof: usize,
    num_qpoints: usize,
    /// Quadrature point coordinates, row-major `[nqpts × dim]`.
    q_ref: Vec<T>,
    /// Quadrature weights, length `nqpts`.
    weights: Vec<T>,
    /// Interpolation matrix, row-major `[nqpts × num_dof]`.
    interp: Vec<T>,
    /// Gradient tensor, layout `[nqpts × num_dof × dim]`.
    grad: Vec<T>,
}

impl<T: Scalar> SimplexBasis<T> {
    /// Construct a simplex basis.
    ///
    /// # Parameters
    /// * `topo`  — element topology (`ElemTopology::Triangle` or `ElemTopology::Tet`).
    /// * `poly`  — polynomial order (1 = P1, 2 = P2).
    /// * `ncomp` — number of field components (1 for scalar problems).
    /// * `q`     — number of quadrature points (see module-level table for valid
    ///             values per topology).
    ///
    /// # Errors
    /// Returns `ReedError::Basis` for unsupported topology/poly/q combinations.
    pub fn new(topo: ElemTopology, poly: usize, ncomp: usize, q: usize) -> ReedResult<Self> {
        let (dim, num_dof) = match (topo, poly) {
            (ElemTopology::Triangle, 1) => (2, 3),
            (ElemTopology::Triangle, 2) => (2, 6),
            (ElemTopology::Tet, 1) => (3, 4),
            (ElemTopology::Tet, 2) => (3, 10),
            _ => {
                return Err(ReedError::Basis(format!(
                    "SimplexBasis: unsupported (topology={:?}, poly={})",
                    topo, poly
                )))
            }
        };

        // Quadrature rule ---------------------------------------------------
        let (q_ref_f64, weights_f64) = match topo {
            ElemTopology::Triangle => tri_quadrature(q)?,
            ElemTopology::Tet => tet_quadrature(q)?,
            _ => unreachable!(),
        };
        let num_qpoints = q_ref_f64.len() / dim;

        // Convert to target scalar type.
        let q_ref: Vec<T> = q_ref_f64
            .iter()
            .map(|&v| to_t::<T>(v))
            .collect::<ReedResult<_>>()?;
        let weights: Vec<T> = weights_f64
            .iter()
            .map(|&v| to_t::<T>(v))
            .collect::<ReedResult<_>>()?;

        // Interpolation & gradient tables -----------------------------------
        let qpts: Vec<[f64; 3]> = (0..num_qpoints)
            .map(|qi| {
                let mut pt = [0.0f64; 3];
                for d in 0..dim {
                    pt[d] = q_ref_f64[qi * dim + d];
                }
                pt
            })
            .collect();

        let mut interp = vec![0.0f64; num_qpoints * num_dof];
        let mut grad = vec![0.0f64; num_qpoints * num_dof * dim];

        for (qi, pt) in qpts.iter().enumerate() {
            let (phi, dphi) = match (topo, poly) {
                (ElemTopology::Triangle, 1) => tri_p1_basis(pt[0], pt[1]),
                (ElemTopology::Triangle, 2) => tri_p2_basis(pt[0], pt[1]),
                (ElemTopology::Tet, 1) => tet_p1_basis(pt[0], pt[1], pt[2]),
                (ElemTopology::Tet, 2) => tet_p2_basis(pt[0], pt[1], pt[2]),
                _ => unreachable!(),
            };
            for dof in 0..num_dof {
                interp[qi * num_dof + dof] = phi[dof];
                for d in 0..dim {
                    grad[(qi * num_dof + dof) * dim + d] = dphi[dof * dim + d];
                }
            }
        }

        let interp_t: Vec<T> = interp
            .iter()
            .map(|&v| to_t::<T>(v))
            .collect::<ReedResult<_>>()?;
        let grad_t: Vec<T> = grad
            .iter()
            .map(|&v| to_t::<T>(v))
            .collect::<ReedResult<_>>()?;

        Ok(Self {
            topo,
            dim,
            ncomp,
            num_dof,
            num_qpoints,
            q_ref,
            weights,
            interp: interp_t,
            grad: grad_t,
        })
    }

    // ── accessor helpers ───────────────────────────────────────────────────

    #[inline]
    fn interp_val(&self, qpt: usize, dof: usize) -> T {
        self.interp[qpt * self.num_dof + dof]
    }

    /// `grad[(qpt * num_dof + dof) * dim + d]`
    #[inline]
    fn grad_val(&self, qpt: usize, dof: usize, d: usize) -> T {
        self.grad[(qpt * self.num_dof + dof) * self.dim + d]
    }

    // ── element-level apply ────────────────────────────────────────────────

    fn apply_interp_elem(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        for comp in 0..self.ncomp {
            if transpose {
                // v: [ncomp × num_dof],  u: [nqpts × ncomp]
                for dof in 0..self.num_dof {
                    let mut sum = T::ZERO;
                    for qpt in 0..self.num_qpoints {
                        sum += self.interp_val(qpt, dof) * u_elem[qpt * self.ncomp + comp];
                    }
                    v_elem[comp * self.num_dof + dof] += sum;
                }
            } else {
                // u: [ncomp × num_dof],  v: [nqpts × ncomp]
                for qpt in 0..self.num_qpoints {
                    let mut sum = T::ZERO;
                    for dof in 0..self.num_dof {
                        sum += self.interp_val(qpt, dof) * u_elem[comp * self.num_dof + dof];
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
                // u: [nqpts × ncomp × dim],  v: [ncomp × num_dof]
                for dof in 0..self.num_dof {
                    let mut sum = T::ZERO;
                    for qpt in 0..self.num_qpoints {
                        for d in 0..self.dim {
                            sum += self.grad_val(qpt, dof, d)
                                * u_elem[qpt * qcomp + comp * self.dim + d];
                        }
                    }
                    v_elem[comp * self.num_dof + dof] += sum;
                }
            } else {
                // u: [ncomp × num_dof],  v: [nqpts × ncomp × dim]
                for qpt in 0..self.num_qpoints {
                    for d in 0..self.dim {
                        let mut sum = T::ZERO;
                        for dof in 0..self.num_dof {
                            sum += self.grad_val(qpt, dof, d) * u_elem[comp * self.num_dof + dof];
                        }
                        v_elem[qpt * qcomp + comp * self.dim + d] = sum;
                    }
                }
            }
        }
    }
}

// ── BasisTrait impl ───────────────────────────────────────────────────────────

impl<T: Scalar> BasisTrait<T> for SimplexBasis<T> {
    fn dim(&self) -> usize { self.dim }
    fn num_dof(&self) -> usize { self.num_dof }
    fn num_qpoints(&self) -> usize { self.num_qpoints }
    fn num_comp(&self) -> usize { self.ncomp }

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
                let in_stride = if transpose {
                    self.num_qpoints * self.ncomp
                } else {
                    self.num_dof * self.ncomp
                };
                let out_stride = if transpose {
                    self.num_dof * self.ncomp
                } else {
                    self.num_qpoints * self.ncomp
                };
                check_sizes(u, in_stride * num_elem, v, out_stride * num_elem, "interp")?;
                // zero output for accumulation in transpose path
                if transpose { v.fill(T::ZERO); }
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
                let in_stride = if transpose {
                    self.num_qpoints * qcomp
                } else {
                    self.num_dof * self.ncomp
                };
                let out_stride = if transpose {
                    self.num_dof * self.ncomp
                } else {
                    self.num_qpoints * qcomp
                };
                check_sizes(u, in_stride * num_elem, v, out_stride * num_elem, "grad")?;
                if transpose { v.fill(T::ZERO); }
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
                    "SimplexBasis: eval mode {:?} not implemented",
                    other
                )));
            }
        }
        Ok(())
    }

    fn q_weights(&self) -> &[T] { &self.weights }
    fn q_ref(&self) -> &[T] { &self.q_ref }
}

// ── shape functions ───────────────────────────────────────────────────────────

/// P1 triangle basis: φ and ∇φ at (x,y).
///
/// Returns `(phi[3], dphi[3*2])` where `dphi[i*2+d]` = ∂φᵢ/∂xd.
/// Gradients are constant (independent of x,y).
fn tri_p1_basis(x: f64, y: f64) -> (Vec<f64>, Vec<f64>) {
    let _ = (x, y); // gradients are constant
    // φ₀ = 1-x-y,  φ₁ = x,  φ₂ = y
    let phi = vec![1.0 - x - y, x, y];
    // ∇φ₀ = (-1,-1),  ∇φ₁ = (1,0),  ∇φ₂ = (0,1)
    let dphi = vec![
        -1.0, -1.0, // dof 0
         1.0,  0.0, // dof 1
         0.0,  1.0, // dof 2
    ];
    (phi, dphi)
}

/// P2 triangle basis: φ and ∇φ at (x,y).
///
/// Node ordering (standard serendipity):
/// ```text
///  2
///  |\\
///  5  4
///  |    \\
///  0--3--1
/// ```
/// DOF 0=(0,0), 1=(1,0), 2=(0,1), 3=(½,0), 4=(½,½), 5=(0,½).
fn tri_p2_basis(x: f64, y: f64) -> (Vec<f64>, Vec<f64>) {
    let l0 = 1.0 - x - y;
    let l1 = x;
    let l2 = y;
    let phi = vec![
        l0 * (2.0 * l0 - 1.0), // 0: vertex (0,0)
        l1 * (2.0 * l1 - 1.0), // 1: vertex (1,0)
        l2 * (2.0 * l2 - 1.0), // 2: vertex (0,1)
        4.0 * l0 * l1,         // 3: midpoint (½,0)
        4.0 * l1 * l2,         // 4: midpoint (½,½)
        4.0 * l0 * l2,         // 5: midpoint (0,½)
    ];
    // ∂φ/∂x (d=0), ∂φ/∂y (d=1)
    // dl0/dx=-1, dl0/dy=-1; dl1/dx=1, dl1/dy=0; dl2/dx=0, dl2/dy=1
    let dphi = vec![
        // dof 0: φ = l0*(2l0-1)  → ∂/∂x = (-1)*(4l0-1), ∂/∂y = (-1)*(4l0-1)
        -(4.0*l0 - 1.0), -(4.0*l0 - 1.0),
        // dof 1: φ = l1*(2l1-1)  → ∂/∂x = 4l1-1, ∂/∂y = 0
        4.0*l1 - 1.0, 0.0,
        // dof 2: φ = l2*(2l2-1)  → ∂/∂x = 0, ∂/∂y = 4l2-1
        0.0, 4.0*l2 - 1.0,
        // dof 3: φ = 4*l0*l1     → ∂/∂x = 4*(l0-l1), ∂/∂y = -4*l1
        4.0*(l0 - l1), -4.0*l1,
        // dof 4: φ = 4*l1*l2     → ∂/∂x = 4*l2, ∂/∂y = 4*l1
        4.0*l2, 4.0*l1,
        // dof 5: φ = 4*l0*l2     → ∂/∂x = -4*l2, ∂/∂y = 4*(l0-l2)
        -4.0*l2, 4.0*(l0 - l2),
    ];
    (phi, dphi)
}

/// P1 tet basis: φ and ∇φ at (x,y,z).
///
/// DOF 0=(0,0,0), 1=(1,0,0), 2=(0,1,0), 3=(0,0,1).
fn tet_p1_basis(x: f64, y: f64, z: f64) -> (Vec<f64>, Vec<f64>) {
    let phi = vec![1.0 - x - y - z, x, y, z];
    // ∇φ₀=(-1,-1,-1), ∇φ₁=(1,0,0), ∇φ₂=(0,1,0), ∇φ₃=(0,0,1)
    let dphi = vec![
        -1.0, -1.0, -1.0,
         1.0,  0.0,  0.0,
         0.0,  1.0,  0.0,
         0.0,  0.0,  1.0,
    ];
    let _ = (x, y, z);
    (phi, dphi)
}

/// P2 tet basis: φ and ∇φ at (x,y,z).
///
/// 10 DOFs: 4 vertices + 6 edge midpoints.
///
/// Vertex ordering: V0=(0,0,0), V1=(1,0,0), V2=(0,1,0), V3=(0,0,1).
/// Edge midpoints:  E4=(½,0,0), E5=(½,½,0), E6=(0,½,0),
///                  E7=(0,0,½), E8=(½,0,½), E9=(0,½,½)...
///
/// Wait, standard ordering for Tet10 edges:
/// E4 = midpoint V0-V1 = (½,0,0)
/// E5 = midpoint V1-V2 = (½,½,0)
/// E6 = midpoint V0-V2 = (0,½,0)
/// E7 = midpoint V0-V3 = (0,0,½)
/// E8 = midpoint V1-V3 = (½,0,½)
/// E9 = midpoint V2-V3 = (0,½,½)
fn tet_p2_basis(x: f64, y: f64, z: f64) -> (Vec<f64>, Vec<f64>) {
    let l0 = 1.0 - x - y - z;
    let l1 = x;
    let l2 = y;
    let l3 = z;
    let phi = vec![
        l0 * (2.0*l0 - 1.0), // 0: V0
        l1 * (2.0*l1 - 1.0), // 1: V1
        l2 * (2.0*l2 - 1.0), // 2: V2
        l3 * (2.0*l3 - 1.0), // 3: V3
        4.0*l0*l1,           // 4: E4 midpoint V0-V1
        4.0*l1*l2,           // 5: E5 midpoint V1-V2
        4.0*l0*l2,           // 6: E6 midpoint V0-V2
        4.0*l0*l3,           // 7: E7 midpoint V0-V3
        4.0*l1*l3,           // 8: E8 midpoint V1-V3
        4.0*l2*l3,           // 9: E9 midpoint V2-V3
    ];
    // dl0=(−1,−1,−1), dl1=(1,0,0), dl2=(0,1,0), dl3=(0,0,1)
    let dphi = vec![
        // dof 0: ∂/∂x = -(4l0-1), ∂/∂y = -(4l0-1), ∂/∂z = -(4l0-1)
        -(4.0*l0-1.0), -(4.0*l0-1.0), -(4.0*l0-1.0),
        // dof 1: ∂/∂x = 4l1-1, ∂/∂y = 0, ∂/∂z = 0
        4.0*l1-1.0, 0.0, 0.0,
        // dof 2: ∂/∂x = 0, ∂/∂y = 4l2-1, ∂/∂z = 0
        0.0, 4.0*l2-1.0, 0.0,
        // dof 3: ∂/∂x = 0, ∂/∂y = 0, ∂/∂z = 4l3-1
        0.0, 0.0, 4.0*l3-1.0,
        // dof 4: φ=4l0l1 → ∂/∂x=4(l0-l1), ∂/∂y=-4l1, ∂/∂z=-4l1
        4.0*(l0-l1), -4.0*l1, -4.0*l1,
        // dof 5: φ=4l1l2 → ∂/∂x=4l2, ∂/∂y=4l1, ∂/∂z=0
        4.0*l2, 4.0*l1, 0.0,
        // dof 6: φ=4l0l2 → ∂/∂x=-4l2, ∂/∂y=4(l0-l2), ∂/∂z=-4l2
        -4.0*l2, 4.0*(l0-l2), -4.0*l2,
        // dof 7: φ=4l0l3 → ∂/∂x=-4l3, ∂/∂y=-4l3, ∂/∂z=4(l0-l3)
        -4.0*l3, -4.0*l3, 4.0*(l0-l3),
        // dof 8: φ=4l1l3 → ∂/∂x=4l3, ∂/∂y=0, ∂/∂z=4l1
        4.0*l3, 0.0, 4.0*l1,
        // dof 9: φ=4l2l3 → ∂/∂x=0, ∂/∂y=4l3, ∂/∂z=4l2
        0.0, 4.0*l3, 4.0*l2,
    ];
    (phi, dphi)
}

// ── quadrature rules ─────────────────────────────────────────────────────────

/// Triangle Gauss quadrature rules (reference triangle area = 1/2).
///
/// Returns `(ref_coords, weights)` where `ref_coords` is row-major `[q×2]`.
///
/// | q | degree exact |
/// |---|--------------|
/// | 1 | 1 |
/// | 3 | 2 |
/// | 4 | 3 |
/// | 6 | 4 |
/// | 7 | 5 |
fn tri_quadrature(q: usize) -> ReedResult<(Vec<f64>, Vec<f64>)> {
    match q {
        1 => {
            // Centroid rule (degree 1 exact)
            let pts = vec![1.0/3.0, 1.0/3.0];
            let wts = vec![0.5];
            Ok((pts, wts))
        }
        3 => {
            // Degree 2 exact (Dunavant / midpoint rule)
            let a = 1.0/6.0;
            let b = 2.0/3.0;
            let pts = vec![
                a, a,
                b, a,
                a, b,
            ];
            let wts = vec![1.0/6.0, 1.0/6.0, 1.0/6.0];
            Ok((pts, wts))
        }
        4 => {
            // Degree 3 exact (Dunavant 4-point, one negative weight)
            let pts = vec![
                1.0/3.0, 1.0/3.0,
                0.2,     0.2,
                0.6,     0.2,
                0.2,     0.6,
            ];
            let wts = vec![
                -27.0/96.0,
                25.0/96.0,
                25.0/96.0,
                25.0/96.0,
            ];
            Ok((pts, wts))
        }
        6 => {
            // Degree 4 exact (Dunavant 6-point)
            // Group 1 (a1 symmetry, 3 points)
            let a1 = 0.445948490915965_f64;
            let b1 = 0.108103018168070_f64;
            let w1 = 0.111690794839005_f64;
            // Group 2 (a2 symmetry, 3 points)
            let a2 = 0.091576213509771_f64;
            let b2 = 0.816847572980459_f64;
            let w2 = 0.054975871827661_f64;
            let pts = vec![
                a1, a1,
                b1, a1,
                a1, b1,
                a2, a2,
                b2, a2,
                a2, b2,
            ];
            let wts = vec![w1, w1, w1, w2, w2, w2];
            Ok((pts, wts))
        }
        7 => {
            // Degree 5 exact (Dunavant 7-point).
            // Weights are for reference triangle with area = 0.5.
            // Closed-form via sqrt(15):
            //   w_center = 9/40 * (1/2) — reference area factor
            //   a_inner  = (6 - sqrt(15)) / 21,  w_inner = (155 - sqrt(15)) / 1200
            //   a_outer  = (6 + sqrt(15)) / 21,  w_outer = (155 + sqrt(15)) / 1200
            // (Dunavant 1985; weights sum to 1 on unit-area triangle, halved here.)
            let sq15 = 15.0_f64.sqrt();
            let w1 = 9.0 / 80.0; // 9/40 * 1/2
            let a_inner = (6.0 - sq15) / 21.0;
            let b_inner = 1.0 - 2.0 * a_inner;
            let w_inner = (155.0 - sq15) / 2400.0;
            let a_outer = (6.0 + sq15) / 21.0;
            let b_outer = 1.0 - 2.0 * a_outer;
            let w_outer = (155.0 + sq15) / 2400.0;
            let pts = vec![
                1.0/3.0, 1.0/3.0,
                a_outer, a_outer,
                b_outer, a_outer,
                a_outer, b_outer,
                a_inner, a_inner,
                b_inner, a_inner,
                a_inner, b_inner,
            ];
            let wts = vec![w1, w_outer, w_outer, w_outer, w_inner, w_inner, w_inner];
            Ok((pts, wts))
        }
        _ => Err(ReedError::Basis(format!(
            "triangle quadrature: unsupported q={q} (valid: 1,3,4,6,7)"
        ))),
    }
}

/// Tetrahedron Gauss quadrature rules (reference tet volume = 1/6).
///
/// Returns `(ref_coords, weights)` where `ref_coords` is row-major `[q×3]`.
///
/// | q | degree exact |
/// |---|--------------|
/// | 1 | 1 |
/// | 4 | 2 |
/// | 5 | 3 |
fn tet_quadrature(q: usize) -> ReedResult<(Vec<f64>, Vec<f64>)> {
    match q {
        1 => {
            let pts = vec![0.25, 0.25, 0.25];
            let wts = vec![1.0/6.0];
            Ok((pts, wts))
        }
        4 => {
            // Degree 2 exact (symmetric 4-point rule)
            // a = (5 - sqrt(5)) / 20,  b = (5 + 3*sqrt(5)) / 20
            let sq5 = 5.0_f64.sqrt();
            let a = (5.0 - sq5) / 20.0;  // ≈ 0.1381966
            let b = (5.0 + 3.0*sq5) / 20.0; // ≈ 0.5854102
            let w = 1.0/24.0;
            let pts = vec![
                a, a, a,
                b, a, a,
                a, b, a,
                a, a, b,
            ];
            let wts = vec![w, w, w, w];
            Ok((pts, wts))
        }
        5 => {
            // Degree 3 exact (Keast 5-point rule with negative weight)
            // Point at centroid with negative weight
            let pts = vec![
                0.25,         0.25,        0.25,
                1.0/6.0, 1.0/6.0, 1.0/6.0,
                0.5,     1.0/6.0, 1.0/6.0,
                1.0/6.0, 0.5,     1.0/6.0,
                1.0/6.0, 1.0/6.0, 0.5,
            ];
            let wts = vec![
                -4.0/30.0,
                9.0/120.0,
                9.0/120.0,
                9.0/120.0,
                9.0/120.0,
            ];
            Ok((pts, wts))
        }
        _ => Err(ReedError::Basis(format!(
            "tet quadrature: unsupported q={q} (valid: 1,4,5)"
        ))),
    }
}

// ── utilities ─────────────────────────────────────────────────────────────────

fn to_t<T: Scalar>(v: f64) -> ReedResult<T> {
    T::from(v).ok_or_else(|| {
        ReedError::Basis(format!("SimplexBasis: failed to convert {v} to scalar"))
    })
}

fn check_sizes<T>(
    u: &[T],
    u_expected: usize,
    v: &[T],
    v_expected: usize,
    mode: &str,
) -> ReedResult<()> {
    if u.len() != u_expected || v.len() != v_expected {
        return Err(ReedError::Basis(format!(
            "SimplexBasis {mode} size mismatch: \
             input {} (expected {}), output {} (expected {})",
            u.len(), u_expected, v.len(), v_expected
        )));
    }
    Ok(())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    // ── partition-of-unity ────────────────────────────────────────────────

    #[test]
    fn tri_p1_partition_of_unity() {
        for &(x, y) in &[(0.1, 0.2), (0.5, 0.3), (1.0/3.0, 1.0/3.0)] {
            let (phi, _) = tri_p1_basis(x, y);
            let sum: f64 = phi.iter().sum();
            assert!((sum - 1.0).abs() < TOL, "PoU failed at ({x},{y}): sum={sum}");
        }
    }

    #[test]
    fn tri_p2_partition_of_unity() {
        for &(x, y) in &[(0.1, 0.2), (0.5, 0.25), (0.25, 0.5)] {
            let (phi, _) = tri_p2_basis(x, y);
            let sum: f64 = phi.iter().sum();
            assert!((sum - 1.0).abs() < TOL, "PoU failed at ({x},{y}): sum={sum}");
        }
    }

    #[test]
    fn tet_p1_partition_of_unity() {
        for &(x, y, z) in &[(0.1,0.2,0.3),(0.25,0.25,0.25)] {
            let (phi, _) = tet_p1_basis(x, y, z);
            let sum: f64 = phi.iter().sum();
            assert!((sum - 1.0).abs() < TOL, "PoU failed at ({x},{y},{z}): sum={sum}");
        }
    }

    #[test]
    fn tet_p2_partition_of_unity() {
        for &(x, y, z) in &[(0.1,0.2,0.1),(0.2,0.2,0.2)] {
            let (phi, _) = tet_p2_basis(x, y, z);
            let sum: f64 = phi.iter().sum();
            assert!((sum - 1.0).abs() < TOL, "PoU failed at ({x},{y},{z}): sum={sum}");
        }
    }

    // ── gradient consistency via finite differences ───────────────────────

    // Centered finite difference — O(h²) error so 1e-8 tolerance is safe with h=1e-5.
    fn fd_check_tri(poly: usize, x: f64, y: f64, h: f64) {
        let basis_fn: fn(f64, f64) -> (Vec<f64>, Vec<f64>) =
            if poly == 1 { tri_p1_basis } else { tri_p2_basis };
        let (_, dphi) = basis_fn(x, y);
        let (phi_px, _) = basis_fn(x + h, y);
        let (phi_mx, _) = basis_fn(x - h, y);
        let (phi_py, _) = basis_fn(x, y + h);
        let (phi_my, _) = basis_fn(x, y - h);
        let num_dof = dphi.len() / 2;
        for i in 0..num_dof {
            let fd_x = (phi_px[i] - phi_mx[i]) / (2.0 * h);
            let fd_y = (phi_py[i] - phi_my[i]) / (2.0 * h);
            let an_x = dphi[i*2];
            let an_y = dphi[i*2+1];
            assert!(
                (fd_x - an_x).abs() < 1e-8,
                "tri P{poly} dof {i} ∂/∂x: FD={fd_x:.10}, analytic={an_x:.10}"
            );
            assert!(
                (fd_y - an_y).abs() < 1e-8,
                "tri P{poly} dof {i} ∂/∂y: FD={fd_y:.10}, analytic={an_y:.10}"
            );
        }
    }

    #[test]
    fn tri_p1_gradient_fd() { fd_check_tri(1, 0.2, 0.3, 1e-6); }
    #[test]
    fn tri_p2_gradient_fd() { fd_check_tri(2, 0.2, 0.3, 1e-6); }

    // ── quadrature weight sums ────────────────────────────────────────────

    #[test]
    fn tri_quad_weight_sums() {
        for q in [1usize, 3, 4, 6, 7] {
            let (_, wts) = tri_quadrature(q).unwrap();
            let sum: f64 = wts.iter().sum();
            assert!(
                (sum - 0.5).abs() < 1e-13,
                "tri q={q}: weights sum to {sum}, expected 0.5"
            );
        }
    }

    #[test]
    fn tet_quad_weight_sums() {
        for q in [1usize, 4, 5] {
            let (_, wts) = tet_quadrature(q).unwrap();
            let sum: f64 = wts.iter().sum();
            assert!(
                (sum - 1.0/6.0).abs() < 1e-13,
                "tet q={q}: weights sum to {sum}, expected 1/6"
            );
        }
    }

    // ── interp exactly reproduces linear fields ───────────────────────────

    #[test]
    fn tri_p1_basis_apply_interp() {
        // u = x at the 3 P1 nodes: u[0]=(0,0)→0, u[1]=(1,0)→1, u[2]=(0,1)→0
        // At q=3 Gauss points, interpolated value should equal x-coordinate.
        let basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, 1, 1, 3).unwrap();
        let u = vec![0.0_f64, 1.0, 0.0]; // ncomp=1 × num_dof=3
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        let (q_ref, _) = tri_quadrature(3).unwrap();
        for (qi, vv) in v.iter().enumerate() {
            let expected = q_ref[qi*2]; // x-coordinate of qpt
            assert!((*vv - expected).abs() < TOL, "qpt {qi}: got {vv}, expected {expected}");
        }
    }

    #[test]
    fn tet_p1_basis_apply_interp() {
        // u = y at P1 tet nodes: (0,0,0)→0, (1,0,0)→0, (0,1,0)→1, (0,0,1)→0
        let basis = SimplexBasis::<f64>::new(ElemTopology::Tet, 1, 1, 4).unwrap();
        let u = vec![0.0_f64, 0.0, 1.0, 0.0]; // u=y at each vertex
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        let (q_ref, _) = tet_quadrature(4).unwrap();
        for (qi, vv) in v.iter().enumerate() {
            let expected = q_ref[qi*3 + 1]; // y-coordinate of qpt
            assert!((*vv - expected).abs() < TOL, "qpt {qi}: got {vv}, expected {expected}");
        }
    }

    // ── weight mode ───────────────────────────────────────────────────────

    #[test]
    fn tri_p1_weight_sums_to_area() {
        let basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, 1, 1, 3).unwrap();
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Weight, &[], &mut v).unwrap();
        let sum: f64 = v.iter().sum();
        assert!((sum - 0.5).abs() < TOL, "weight sum={sum}, expected 0.5 (triangle area)");
    }

    #[test]
    fn tet_p1_weight_sums_to_volume() {
        let basis = SimplexBasis::<f64>::new(ElemTopology::Tet, 1, 1, 4).unwrap();
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Weight, &[], &mut v).unwrap();
        let sum: f64 = v.iter().sum();
        assert!((sum - 1.0/6.0).abs() < TOL, "weight sum={sum}, expected 1/6 (tet volume)");
    }
}
