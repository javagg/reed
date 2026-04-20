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
                    offsets.push(((sy + jy) * ndofs_1d + (sx + jx)) as i32);
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
                            offsets.push(
                                (((sz + jz) * ndofs_1d + (sy + jy)) * ndofs_1d + (sx + jx)) as i32,
                            );
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

fn run_poisson(
    dim: usize,
    nelem_1d: usize,
    p: usize,
    q: usize,
) -> Result<(), Box<dyn std::error::Error>> {
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

        let x = reed.vector_from_slice(node_coords)?;
        let mut qdata = reed.vector(nelem * q)?;
        qdata.set_value(0.0)?;
        let op_build = reed
            .operator_builder()
            .qfunction(reed.q_function_by_name("Poisson1DBuild")?)
            .field("dx", Some(&*r_u), Some(&*b_u), FieldVector::Active)
            .field("weights", None, Some(&*b_u), FieldVector::None)
            .field("qdata", Some(&*r_q), None, FieldVector::Active)
            .build()?;
        op_build.apply(&*x, &mut *qdata)?;

        let u = reed.vector_from_slice(node_coords)?;
        let mut v = reed.vector(ndofs)?;
        v.set_value(0.0)?;

        let op = reed
            .operator_builder()
            .qfunction(reed.q_function_by_name("Poisson1DApply")?)
            .field("du", Some(&*r_u), Some(&*b_u), FieldVector::Active)
            .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
            .field("dv", Some(&*r_u), Some(&*b_u), FieldVector::Active)
            .build()?;
        op.apply(&*u, &mut *v)?;

        let mut values = vec![0.0; ndofs];
        v.copy_to_slice(&mut values)?;
        println!("Poisson example (1D)");
        println!("nelem_1d={nelem_1d}, p={p}, q={q}");
        println!(
            "output norm1 = {:.12e}",
            values.iter().map(|x| x.abs()).sum::<f64>()
        );
        return Ok(());
    }

    let x_coords = comps.concat();
    let x = reed.vector_from_slice(&x_coords)?;
    let r_x = reed.elem_restriction(nelem, elemsize, dim, ndofs, dim * ndofs, &offsets)?;
    let r_u = reed.elem_restriction(nelem, elemsize, 1, 1, ndofs, &offsets)?;
    let b_x = reed.basis_tensor_h1_lagrange(dim, dim, p, q, QuadMode::Gauss)?;
    let b_u = reed.basis_tensor_h1_lagrange(dim, 1, p, q, QuadMode::Gauss)?;

    let qdata_comp = dim * dim;
    let r_q = reed.strided_elem_restriction(
        nelem,
        qpts_per_elem,
        qdata_comp,
        nelem * qpts_per_elem * qdata_comp,
        [1, qpts_per_elem as i32, (qpts_per_elem * qdata_comp) as i32],
    )?;
    let mut qdata = reed.vector(nelem * qpts_per_elem * qdata_comp)?;
    qdata.set_value(0.0)?;

    let qf_build = if dim == 2 {
        "Poisson2DBuild"
    } else {
        "Poisson3DBuild"
    };
    let op_build = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name(qf_build)?)
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()?;
    op_build.apply(&*x, &mut *qdata)?;

    let mut u_vals = vec![0.0_f64; ndofs];
    for c in comps.iter().take(dim) {
        for i in 0..ndofs {
            u_vals[i] += c[i];
        }
    }
    let u = reed.vector_from_slice(&u_vals)?;
    let mut v = reed.vector(ndofs)?;
    v.set_value(0.0)?;

    let qf_apply = if dim == 2 {
        "Poisson2DApply"
    } else {
        "Poisson3DApply"
    };
    let op = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name(qf_apply)?)
        .field("du", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("dv", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()?;
    op.apply(&*u, &mut *v)?;

    let mut values = vec![0.0; ndofs];
    v.copy_to_slice(&mut values)?;
    println!("Poisson example ({dim}D)");
    println!("nelem_1d={nelem_1d}, p={p}, q={q}");
    println!(
        "output norm1 = {:.12e}",
        values.iter().map(|x| x.abs()).sum::<f64>()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("Usage: cargo run --example poisson_1d -- [--dim D] [--nelem N] [--p P] [--q Q]");
        println!("  --dim D     1, 2 or 3 (default 1)");
        println!("  --nelem N   elements per dimension (default 4)");
        println!("  --p P       element polynomial order (default 2)");
        println!("  --q Q       quadrature points per dimension (default p+2)");
        return Ok(());
    }

    let dim = parse_arg(&args, "--dim", 1);
    let nelem = parse_arg(&args, "--nelem", 4);
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

    run_poisson(dim, nelem, p, q)
}
