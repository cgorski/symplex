use symplex::polysys::{LinearSolution, linsolve, solve_numeric_system};
use symplex::prelude::*;

fn show(label: &str, v: &[Ex]) {
    let s: Vec<String> = v.iter().map(|e| format!("{e}")).collect();
    eprintln!("{label}: {s:?}");
}

#[test]
fn scratch_solve_variants() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    eprintln!("0: {:?}", ctx.int(0).solve(&x).map(|v| v.len()));
    eprintln!("1: {:?}", ctx.int(1).solve(&x).map(|v| v.len()));
    eprintln!("a: {:?}", a.solve(&x).map(|v| v.len()));
    eprintln!("exp: {:?}", x.exp().solve(&x).map(|v| v.len()));
    eprintln!("sin=2: {:?}", (&x.sin() - 2).solve(&x).map(|v| v.len()));
    show("x^3-27", &(&x.powi(3) - 27).solve(&x).unwrap());
    show("x^3-2", &(&x.powi(3) - 2).solve(&x).unwrap());
    show("x^5-2", &(&x.powi(5) - 2).solve(&x).unwrap());
    show("x^4+4", &(&x.powi(4) + 4).solve(&x).unwrap());
    show("(x-1)/x", &(&(&x - 1) / &x).solve(&x).unwrap_or_default());
    show("x^2-a", &(&x.powi(2) - &a).solve(&x).unwrap());
    show(
        "sin(2x+1)=1/2",
        &(&(&x * 2 + 1).sin() - &ctx.rational(1, 2)).solve(&x).unwrap(),
    );
    let g = (&x.sin() - &ctx.rational(1, 2)).solve_general(&x).unwrap();
    show("general sin", &g.solutions);
    show("params", &g.parameters);
    let g = (&x.cos().powi(2) - &ctx.rational(1, 4)).solve_general(&x).unwrap();
    show("general cos^2=1/4", &g.solutions);
    let g = (&x.tan() - 1).solve_general(&x).unwrap();
    show("general tan", &g.solutions);
    eprintln!("as_set 0: {}", ctx.int(0).solve_as_set(&x));
    eprintln!("as_set 1: {}", ctx.int(1).solve_as_set(&x));
    eprintln!("|x-2|<3: {}", (&(&x - 2).abs() - 3).solve_lt(&x));
    eprintln!("|x-2|>3: {}", (&(&x - 2).abs() - 3).solve_gt(&x));
    eprintln!("|x-a|<=3: {}", (&(&x - &a).abs() - 3).solve_le(&x));
    eprintln!("2|x|+1>0: {}", (&x.abs() * 2 + 1).solve_gt(&x));
    eprintln!("|x|+1<0: {}", (&x.abs() + 1).solve_lt(&x));
    eprintln!(
        "check_solution x^2+1 at 2i: {:?}",
        (&x.powi(2) + 1).check_solution(&x, &(&ctx.i_unit() * 2))
    );
    eprintln!(
        "check_solution x^2+1 at i: {:?}",
        (&x.powi(2) + 1).check_solution(&x, &ctx.i_unit())
    );
}

#[test]
fn scratch_linsolve() {
    let ctx = Context::new();
    let (x, y, z, a, b) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("z"),
        ctx.symbol("a"),
        ctx.symbol("b"),
    );
    let sol = linsolve(&[&a * &x + &y - 1, &x - &y - &b], &[x.clone(), y.clone()]).unwrap();
    eprintln!("{sol:?}");
    if let LinearSolution::Unique(p) = &sol {
        for (v, val) in p {
            eprintln!("{v} = {val}");
        }
    }
    let sol = linsolve(
        &[&x + &y + &z - 6, &x - &y + 1],
        &[x.clone(), y.clone(), z.clone()],
    )
    .unwrap();
    eprintln!("{sol:?}");
    if let LinearSolution::Parametric { solution, free } = &sol {
        for (v, val) in solution {
            eprintln!("{v} = {val}");
        }
        show("free", free);
    }
    let sol = linsolve(
        &[&x + &y - 3, &x - &y - 1, &x * 2 - 4],
        &[x.clone(), y.clone()],
    )
    .unwrap();
    eprintln!("overdetermined: {sol:?}");
    let sol = solve_numeric_system(
        &[&x.powi(2) + &y.powi(2) - 1, &y - &x],
        &[x.clone(), y.clone()],
        &[1.0, 1.0],
    )
    .unwrap();
    eprintln!("newton: {sol:?}");
}

#[test]
fn scratch_polysys() {
    use symplex::polysys::solve_system_ex;
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let cases: Vec<(&str, Vec<Ex>, Vec<Ex>)> = vec![
        (
            "circle-line",
            vec![&x.powi(2) + &y.powi(2) - 1, &y - &x],
            vec![x.clone(), y.clone()],
        ),
        (
            "x^2=2,y=x+1",
            vec![&x.powi(2) - 2, &y - &x - 1],
            vec![x.clone(), y.clone()],
        ),
        (
            "xy=1,x+y=3",
            vec![&x * &y - 1, &x + &y - 3],
            vec![x.clone(), y.clone()],
        ),
        (
            "circle4-parabola",
            vec![&x.powi(2) + &y.powi(2) - 4, &x.powi(2) - &y - 2],
            vec![x.clone(), y.clone()],
        ),
        (
            "3var",
            vec![&x + &y + &z, &x * &y + &y * &z + &z * &x + 3, &x * &y * &z],
            vec![x.clone(), y.clone(), z.clone()],
        ),
        (
            "3var-chain",
            vec![&x.powi(2) - 2, &y - &x - 1, &z - &x * &y],
            vec![x.clone(), y.clone(), z.clone()],
        ),
        (
            "linear",
            vec![&x + &y - 3, &x - &y - 1],
            vec![x.clone(), y.clone()],
        ),
        (
            "inconsistent",
            vec![&x.powi(2) - 1, &x - 2],
            vec![x.clone()],
        ),
        (
            "x^2=y,y^2=2",
            vec![&x.powi(2) - &y, &y.powi(2) - 2],
            vec![x.clone(), y.clone()],
        ),
    ];
    for (label, eqs, vars) in cases {
        match solve_system_ex(&eqs, &vars) {
            Ok(sols) => {
                eprintln!("{label}: {} solutions", sols.len());
                for s in &sols {
                    let strs: Vec<String> = s.iter().map(|e| format!("{e}")).collect();
                    // verify
                    let mut max_res = 0.0f64;
                    for eq in &eqs {
                        let mut r = eq.clone();
                        for (v, val) in vars.iter().zip(s) {
                            r = r.subs(v, val);
                        }
                        let (re, im) = r.eval_complex64().unwrap_or((f64::NAN, 0.0));
                        max_res = max_res.max(re.hypot(im));
                    }
                    eprintln!("   {strs:?}  residual {max_res:.2e}");
                }
            }
            Err(e) => eprintln!("{label}: Err {e}"),
        }
    }
    eprintln!(
        "underdetermined: {:?}",
        solve_system_ex(&[&x + &y - 1], &[x.clone(), y.clone()]).map(|v| v.len())
    );
    eprintln!(
        "positive-dim: {:?}",
        solve_system_ex(&[&x * &y], &[x.clone(), y.clone()]).map(|v| v.len())
    );
}

#[test]
fn scratch_numeric() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = x.atan().solve_numeric(&x, 3.0, 100, 1e-12);
    eprintln!("atan from 3: {r:?}");
    let r = (&x - &x.cos()).solve_numeric(&x, 1.0, 50, 1e-12);
    eprintln!("x-cos: {r:?}");
    let r = (&x.powi(3) - &x * 2 + 2).solve_numeric(&x, -1.0, 100, 1e-12);
    eprintln!("cubic from -1: {r:?}");
}
