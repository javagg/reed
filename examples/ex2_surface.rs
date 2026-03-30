/// libCEED ex2-surface 对应示例（当前 Reed 支持 1D/2D/3D）
///
/// 通过 Poisson 算子计算边界测度：
/// - 1D: 2
/// - 2D: 8
/// - 3D: 24

use reed::{FieldVector, OperatorTrait, QuadMode, Reed};
use std::env;

fn parse_arg(args: &[String], key: &str, default: usize) -> usize {
    args.windows(2)
        .find_map(|w| (w[0] == key).then(|| w[1].parse::<usize>().ok()).flatten())
        .unwrap_or(default)
}

fn build_offsets_1d(nelem_1d: usize, p: usize) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(nelem_1d * p);
    for e in 0..nelem_1d {
        let start = e * (p - 1);
        for j in 0..p {
            offsets.push((start + j) as i32);
        }
    }
    offsets
}

fn build_offsets_2d(nelem_1d: usize, p: usize, ndofs_1d: usize) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(nelem_1d * nelem_1d * p * p);
    for ey in 0..nelem_1d {
        for ex in 0..nelem_1d {
            let sy = ey * (p - 1);
            let sx = ex * (p - 1);
            for jy in 0..p {
                for jx in 0..p {
                    let gi = (sy + jy) * ndofs_1d + (sx + jx);
                    offsets.push(gi as i32);
                }
            }
        }
    }
    offsets
}

fn build_offsets_3d(nelem_1d: usize, p: usize, ndofs_1d: usize) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(nelem_1d * nelem_1d * nelem_1d * p * p * p);
    for ez in 0..nelem_1d {
        for ey in 0..nelem_1d {
            for ex in 0..nelem_1d {
                let sz = ez * (p - 1);
                let sy = ey * (p - 1);
                let sx = ex * (p - 1);
                for jz in 0..p {
                    for jy in 0..p {
                        for jx in 0..p {
                            let gi = ((sz + jz) * ndofs_1d + (sy + jy)) * ndofs_1d + (sx + jx);
                            offsets.push(gi as i32);
                        }
                    }
                }
            }
        }
    }
    offsets
}

fn build_coords_components(dim: usize, ndofs_1d: usize) -> Vec<Vec<f64>> {
    let ndofs = ndofs_1d.pow(dim as u32);
    let mut comps = (0..dim).map(|_| vec![0.0_f64; ndofs]).collect::<Vec<_>>();

    if dim == 1 {
        for i in 0..ndofs {
            comps[0][i] = -1.0 + 2.0 * i as f64 / (ndofs - 1) as f64;
        }
    } else if dim == 2 {
        for iy in 0..ndofs_1d {
            for ix in 0..ndofs_1d {
                let i = iy * ndofs_1d + ix;
                comps[0][i] = -1.0 + 2.0 * ix as f64 / (ndofs_1d - 1) as f64;
                comps[1][i] = -1.0 + 2.0 * iy as f64 / (ndofs_1d - 1) as f64;
            }
        }
    } else {
        for iz in 0..ndofs_1d {
            for iy in 0..ndofs_1d {
                for ix in 0..ndofs_1d {
                    let i = (iz * ndofs_1d + iy) * ndofs_1d + ix;
                    comps[0][i] = -1.0 + 2.0 * ix as f64 / (ndofs_1d - 1) as f64;
                    comps[1][i] = -1.0 + 2.0 * iy as f64 / (ndofs_1d - 1) as f64;
                    comps[2][i] = -1.0 + 2.0 * iz as f64 / (ndofs_1d - 1) as f64;
                }
            }
        }
    }

    comps
}

fn build_poisson_qdata_1d(node_coords: &[f64], qweights: &[f64], nelem: usize, p: usize) -> Vec<f64> {
    let q = qweights.len();
    let mut qdata = Vec::with_capacity(nelem * q);
    for e in 0..nelem {
        let i0 = e * (p - 1);
        let i1 = i0 + (p - 1);
        let jacobian = 0.5 * (node_coords[i1] - node_coords[i0]);
        for &w in qweights {
            qdata.push(w / jacobian);
        }
    }
    qdata
}

fn surface_tol(dim: usize, nelem_1d: usize) -> f64 {
    if dim == 1 {
        1.0e-8
    } else if dim == 2 {
        4.1 / nelem_1d as f64
    } else {
        24.0 / nelem_1d as f64
    }
}

fn run_surface(
    dim: usize,
    nelem_1d: usize,
    p: usize,
    q: usize,
    print_summary: bool,
    check_tolerance: bool,
) -> Result<(f64, f64, f64, f64), Box<dyn std::error::Error>> {
    let reed = Reed::<f64>::init("/cpu/self")?;

    let ndofs_1d = nelem_1d * (p - 1) + 1;
    let ndofs = ndofs_1d.pow(dim as u32);
    let nelem = nelem_1d.pow(dim as u32);
    let elemsize = p.pow(dim as u32);
    let qpts_per_elem = q.pow(dim as u32);

    let offsets = match dim {
        1 => build_offsets_1d(nelem_1d, p),
        2 => build_offsets_2d(nelem_1d, p, ndofs_1d),
        3 => build_offsets_3d(nelem_1d, p, ndofs_1d),
        _ => unreachable!(),
    };

    let comps = build_coords_components(dim, ndofs_1d);

    if dim == 1 {
        let node_coords = &comps[0];
        let r_u = reed.elem_restriction(nelem, p, 1, 1, ndofs, &offsets)?;
        let r_q = reed.strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])?;
        let b_u = reed.basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)?;

        let qdata_vals = build_poisson_qdata_1d(node_coords, b_u.q_weights(), nelem, p);
        let qdata = reed.vector_from_slice(&qdata_vals)?;

        let u = reed.vector_from_slice(node_coords)?;
        let mut v = reed.vector(ndofs)?;
        v.set_value(0.0)?;

        let op_diff = reed
            .operator_builder()
            .qfunction(reed.q_function_by_name("Poisson1DApply")?)
            .field("du", Some(&*r_u), Some(&*b_u), FieldVector::Active)
            .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
            .field("dv", Some(&*r_u), Some(&*b_u), FieldVector::Active)
            .build()?;
        op_diff.apply(&*u, &mut *v)?;

        let mut values = vec![0.0; ndofs];
        v.copy_to_slice(&mut values)?;
        let computed = values.iter().map(|x| x.abs()).sum::<f64>();
        let exact = 2.0_f64;
        let error = (computed - exact).abs();
        let tol = surface_tol(dim, nelem_1d);

        if print_summary {
            println!("libCEED ex2-surface (Reed 1D)");
            println!("nelem_1d={nelem_1d}, p={p}, q={q}");
            println!("Exact value    : {:.12}", exact);
            println!("Computed value : {:.12}", computed);
            println!("Error          : {:.12e}", error);
        }

        if check_tolerance && error > tol {
            return Err(format!("error too large: {:.3e} > tol {:.3e}", error, tol).into());
        }
        return Ok((exact, computed, error, tol));
    }

    let x_coord_data = comps.concat();
    let x_coord = reed.vector_from_slice(&x_coord_data)?;

    let r_x = reed.elem_restriction(nelem, elemsize, dim, ndofs, dim * ndofs, &offsets)?;
    let r_u = reed.elem_restriction(nelem, elemsize, 1, 1, ndofs, &offsets)?;

    let qdata_comp = dim * dim;
    let r_q = reed.strided_elem_restriction(
        nelem,
        qpts_per_elem,
        qdata_comp,
        nelem * qpts_per_elem * qdata_comp,
        [1, qpts_per_elem as i32, (qpts_per_elem * qdata_comp) as i32],
    )?;

    let b_x = reed.basis_tensor_h1_lagrange(dim, dim, p, q, QuadMode::Gauss)?;
    let b_u = reed.basis_tensor_h1_lagrange(dim, 1, p, q, QuadMode::Gauss)?;

    let mut qdata = reed.vector(nelem * qpts_per_elem * qdata_comp)?;
    qdata.set_value(0.0)?;

    let qf_build = if dim == 2 { "Poisson2DBuild" } else { "Poisson3DBuild" };
    let op_build = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name(qf_build)?)
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()?;
    op_build.apply(&*x_coord, &mut *qdata)?;

    let mut u_vals = vec![0.0_f64; ndofs];
    for c in comps.iter().take(dim) {
        for i in 0..ndofs {
            u_vals[i] += c[i];
        }
    }
    let u = reed.vector_from_slice(&u_vals)?;
    let mut v = reed.vector(ndofs)?;
    v.set_value(0.0)?;

    let qf_apply = if dim == 2 { "Poisson2DApply" } else { "Poisson3DApply" };
    let op_diff = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name(qf_apply)?)
        .field("du", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("dv", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()?;
    op_diff.apply(&*u, &mut *v)?;

    let mut values = vec![0.0; ndofs];
    v.copy_to_slice(&mut values)?;
    let computed = values.iter().map(|x| x.abs()).sum::<f64>();

    let exact = if dim == 2 { 8.0_f64 } else { 24.0_f64 };
    let error = (computed - exact).abs();
    let tol = surface_tol(dim, nelem_1d);

    if print_summary {
        println!("libCEED ex2-surface (Reed {dim}D)");
        println!("nelem_1d={nelem_1d}, p={p}, q={q}");
        println!("Exact value    : {:.12}", exact);
        println!("Computed value : {:.12}", computed);
        println!("Error          : {:.12e}", error);
    }

    if check_tolerance && error > tol {
        return Err(format!("error too large: {:.3e} > tol {:.3e}", error, tol).into());
    }

    Ok((exact, computed, error, tol))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("Usage: cargo run --example ex2_surface -- [--dim D] [--nelem N] [--p P] [--q Q]");
        println!("  --dim D     1, 2 or 3 (default 1)");
        println!("  --nelem N   elements per dimension (default 8)");
        println!("  --p P       element polynomial order (default 2)");
        println!("  --q Q       quadrature points per dimension (default p+2)");
        println!("  --study     run a convergence study by doubling nelem");
        println!("  --levels L  number of study levels (default 4)");
        return Ok(());
    }

    let dim = parse_arg(&args, "--dim", 1);
    let nelem = parse_arg(&args, "--nelem", 8);
    let p = parse_arg(&args, "--p", 2);
    let q = parse_arg(&args, "--q", p + 2);
    let study = args.iter().any(|a| a == "--study");
    let levels = parse_arg(&args, "--levels", 4);

    if !(1..=3).contains(&dim) {
        return Err("--dim must be 1, 2 or 3".into());
    }
    if nelem < 1 {
        return Err("--nelem must be >= 1".into());
    }
    if p < 2 {
        return Err("--p must be >= 2".into());
    }
    if q < 1 {
        return Err("--q must be >= 1".into());
    }
    if levels < 1 {
        return Err("--levels must be >= 1".into());
    }

    if study {
        println!("ex2_surface convergence study (dim={dim}, p={p}, q={q})");
        println!("{:>8} {:>18} {:>12} {:>12}", "nelem", "error", "rate", "tol");

        let mut prev_error: Option<f64> = None;
        for level in 0..levels {
            let nelem_level = nelem * (1 << level);
            let (_, _, error, tol) = run_surface(dim, nelem_level, p, q, false, false)?;
            let rate_str = if let Some(prev) = prev_error {
                if error > 0.0 {
                    format!("{:.4}", (prev / error).log2())
                } else {
                    "inf".to_string()
                }
            } else {
                "-".to_string()
            };

            println!("{:>8} {:>18.10e} {:>12} {:>12.4e}", nelem_level, error, rate_str, tol);
            prev_error = Some(error);
        }
        Ok(())
    } else {
        let _ = run_surface(dim, nelem, p, q, true, true)?;
        Ok(())
    }
}
