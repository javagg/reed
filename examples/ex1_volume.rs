/// libCEED ex1-volume 对应示例（当前 Reed 支持 1D/2D/3D）
///
/// 通过质量算子计算测度：
/// - 1D: 区间长度，理论值 2
/// - 2D: 正方形面积，理论值 4
/// - 3D: 立方体体积，理论值 8

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

fn build_coords(dim: usize, ndofs_1d: usize) -> Vec<f64> {
    let ndofs = ndofs_1d.pow(dim as u32);
    if dim == 1 {
        return (0..ndofs)
            .map(|i| -1.0 + 2.0 * i as f64 / (ndofs - 1) as f64)
            .collect();
    }

    let mut comps = (0..dim).map(|_| vec![0.0_f64; ndofs]).collect::<Vec<_>>();
    if dim == 2 {
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
    comps.concat()
}

fn run_volume(
    dim: usize,
    nelem_1d: usize,
    p: usize,
    q: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let reed = Reed::<f64>::init("/cpu/self")?;

    let ndofs_1d = nelem_1d * (p - 1) + 1;
    let nelem = nelem_1d.pow(dim as u32);
    let ndofs = ndofs_1d.pow(dim as u32);
    let qpts_per_elem = q.pow(dim as u32);
    let elemsize = p.pow(dim as u32);

    let offsets = match dim {
        1 => build_offsets_1d(nelem_1d, p),
        2 => build_offsets_2d(nelem_1d, p, ndofs_1d),
        3 => build_offsets_3d(nelem_1d, p, ndofs_1d),
        _ => unreachable!(),
    };

    let x_coords = build_coords(dim, ndofs_1d);
    let x_coord = reed.vector_from_slice(&x_coords)?;

    let (qf_build_name, exact) = match dim {
        1 => ("Mass1DBuild", 2.0_f64),
        2 => ("Mass2DBuild", 4.0_f64),
        3 => ("Mass3DBuild", 8.0_f64),
        _ => unreachable!(),
    };

    let r_x = reed.elem_restriction(nelem, elemsize, dim, ndofs, dim * ndofs, &offsets)?;
    let b_x = reed.basis_tensor_h1_lagrange(dim, dim, p, q, QuadMode::Gauss)?;

    let r_u = reed.elem_restriction(nelem, elemsize, 1, 1, ndofs, &offsets)?;
    let b_u = reed.basis_tensor_h1_lagrange(dim, 1, p, q, QuadMode::Gauss)?;

    let r_q = reed.strided_elem_restriction(
        nelem,
        qpts_per_elem,
        1,
        nelem * qpts_per_elem,
        [1, qpts_per_elem as i32, qpts_per_elem as i32],
    )?;

    let mut qdata = reed.vector(nelem * qpts_per_elem)?;
    qdata.set_value(0.0)?;

    let op_build = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name(qf_build_name)?)
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()?;
    op_build.apply(&*x_coord, &mut *qdata)?;

    let op_mass = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply")?)
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()?;

    let u = reed.vector_from_slice(&vec![1.0_f64; ndofs])?;
    let mut v = reed.vector(ndofs)?;
    v.set_value(0.0)?;
    op_mass.apply(&*u, &mut *v)?;

    let mut values = vec![0.0; ndofs];
    v.copy_to_slice(&mut values)?;
    let computed = values.iter().sum::<f64>();
    let error = (computed - exact).abs();

    println!("libCEED ex1-volume (Reed {dim}D)");
    println!("nelem_1d={nelem_1d}, p={p}, q={q}");
    println!("Exact value    : {:.12}", exact);
    println!("Computed value : {:.12}", computed);
    println!("Error          : {:.12e}", error);

    let tol = if dim == 1 { 2.0e3 * f64::EPSILON } else { 1.0e-9 };
    if error > tol {
        return Err(format!("error too large: {:.3e} > tol {:.3e}", error, tol).into());
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("Usage: cargo run --example ex1_volume -- [--dim D] [--nelem N] [--p P] [--q Q]");
        println!("  --dim D     1, 2 or 3 (default 1)");
        println!("  --nelem N   elements per dimension (default 8)");
        println!("  --p P       element polynomial order (default 2)");
        println!("  --q Q       quadrature points per dimension (default p+2)");
        return Ok(());
    }

    let dim = parse_arg(&args, "--dim", 1);
    let nelem = parse_arg(&args, "--nelem", 8);
    let p = parse_arg(&args, "--p", 2);
    let q = parse_arg(&args, "--q", p + 2);

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

    run_volume(dim, nelem, p, q)
}
