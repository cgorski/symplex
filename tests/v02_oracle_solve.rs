//! SymPy oracle, 0.2 surface — solvers: general periodic solutions
//! (`solve_general` vs `solveset(…, S.Reals)` enumerated in a window),
//! `linsolve` (unique / symbolic / parametric / inconsistent), polynomial
//! systems, ODE initial-value problems and linear recurrences.

mod v02_oracle_common;

use symplex::prelude::*;
use v02_oracle_common::*;

/// Library bugs surfaced by this file (strict xfail — see common module).
const KNOWN_BUGS: &[KnownBug] = &[(
    "rsolve",
    "linear",
    "third_order_irrational_roots",
    // BUG: rsolve_linear([-6, 5, -1, 1], None, n, [1, 0, 0]) does not
    // terminate (>10 min) — the characteristic cubic r^3 - r^2 + 5r - 6 has
    // one real irrational and two complex roots.  SymPy's rsolve returns
    // a closed form in CRootOf within a second.
    "rsolve_linear hangs on a cubic characteristic polynomial with irrational roots",
)];

/// Regression: `solve_system_ex` used to return points that were not
/// solutions (Cardano cube roots of negative radicands were evaluated on
/// the principal complex branch).
#[test]
fn bug_solve_system_ex_wrong_x_component() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f = &x.powi(2) + &y - 3;
    let g = &x - &y.powi(2) + 1;
    let sols = symplex::polysys::solve_system_ex(&[f.clone(), g.clone()], &[x.clone(), y.clone()])
        .unwrap();
    assert_eq!(sols.len(), 4);
    for s in &sols {
        let r = f.subs(&x, &s[0]).subs(&y, &s[1]).eval_complex64().unwrap();
        assert!(
            r.0.hypot(r.1) < 1e-8,
            "residual {r:?} for {}, {}",
            s[0],
            s[1]
        );
    }
}

/// Regression: `solve_system_ex` used to claim this solvable system has
/// no solutions (biquadratic eliminant hit a 0/0 in Ferrari's method).
#[test]
fn bug_solve_system_ex_empty_for_solvable_system() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let sols =
        symplex::polysys::solve_system_ex(&[&x * &y - 1, &x.powi(2) - &y.powi(2) - 3], &[x, y])
            .unwrap();
    assert_eq!(sols.len(), 4, "got {sols:?}");
}

/// Reproducer: `rsolve_linear` hangs (run with a timeout!).
#[test]
#[ignore = "BUG: rsolve_linear([-6, 5, -1, 1], None, n, [1, 0, 0]) does not terminate"]
fn bug_rsolve_linear_hangs_on_irrational_cubic_roots() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let coeffs = [ctx.int(-6), ctx.int(5), ctx.int(-1), ctx.int(1)];
    let sol =
        symplex::rsolve::rsolve_linear(&coeffs, None, &n, &[ctx.int(1), ctx.int(0), ctx.int(0)])
            .unwrap();
    // a(0..) = 1, 0, 0, 6, 6, 6, 42, 78, 78, 330, ...
    assert!((sol.subs_i64(&n, 3).eval_f64().unwrap() - 6.0).abs() < 1e-9);
}

fn parse_all(ctx: &Context, strs: &[String]) -> Result<Vec<Ex>, String> {
    strs.iter().map(|s| parse(ctx, s)).collect()
}

// ── solve_general ──────────────────────────────────────────────────────

#[test]
fn solve_general_periodic_families() {
    run_with_known_bugs("solve", "general_periodic", KNOWN_BUGS, |ctx, fx| {
        let f = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let x = ctx.symbol(fx.str("variable").unwrap_or("x"));
        let window = fx.field("window").and_then(|v| v.as_f64()).unwrap_or(10.0);
        let want: Vec<f64> = fx
            .field("solutions_in_window")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();
        let fam = match f.solve_general(&x) {
            Ok(g) => g,
            Err(e) => return Status::NotImplemented(format!("{e}")),
        };
        // Every member of every family must satisfy the equation …
        let mut got: Vec<f64> = Vec::new();
        for kk in -8..=8 {
            for sol in fam.instance(kk) {
                let Ok(v) = sol.eval_complex64() else {
                    return Status::NotImplemented(format!(
                        "non-numeric solution instance {}",
                        truncate(&sol)
                    ));
                };
                if v.1.abs() > 1e-9 {
                    continue; // complex instance: not in the real solution set
                }
                let resid = f.subs(&x, &sol).eval_complex64();
                match resid {
                    Ok(r) if r.0.hypot(r.1) < 1e-8 => {}
                    Ok(r) => {
                        return Status::Fail(format!(
                            "solution {} has residual {:?}",
                            truncate(&sol),
                            r
                        ));
                    }
                    Err(e) => return Status::NotImplemented(format!("cannot check residual: {e}")),
                }
                if v.0.abs() <= window + 1e-9 {
                    got.push(v.0);
                }
            }
            if fam.parameters.is_empty() {
                break;
            }
        }
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        got.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
        // … and the set of real solutions inside the window must be complete.
        if got.len() != want.len() {
            return Status::Fail(format!(
                "solution count in |x|<={window}: symplex={} sympy={} (symplex={got:?}, sympy={want:?}; families={:?})",
                got.len(),
                want.len(),
                fam.solutions
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            ));
        }
        for (g, w) in got.iter().zip(&want) {
            if !approx_eq_tol(*g, *w, 1e-7) {
                return Status::Fail(format!("solutions differ: symplex={got:?} sympy={want:?}"));
            }
        }
        Status::Pass
    });
}

// ── linsolve ───────────────────────────────────────────────────────────

fn linsolve_inputs(ctx: &Context, fx: &Fixture) -> Result<(Vec<Ex>, Vec<Ex>), String> {
    let eqs = parse_all(ctx, &fx.str_list("equations"))?;
    let vars: Vec<Ex> = fx
        .str_list("variables")
        .iter()
        .map(|v| ctx.symbol(v))
        .collect();
    Ok((eqs, vars))
}

fn check_linear_values(ctx: &Context, pairs: &[(Ex, Ex)], fx: &Fixture) -> Status {
    let pts = fx.eval_points("eval_points");
    if pts.is_empty() {
        return Status::SkippedOracle("no oracle values".into());
    }
    for pt in &pts {
        for (var, val) in pairs {
            let name = var.to_string();
            let Some(want) = pt.values.get(&name) else {
                return Status::Fail(format!("oracle has no value for {name}"));
            };
            let Num::Finite(re, im) = want else {
                return Status::Fail(format!("oracle value for {name} is {want}"));
            };
            match eval_at(val, ctx, &pt.subs) {
                Some(got) if complex_matches(got, (*re, *im), TOLERANCE) => {}
                Some(got) => {
                    return Status::Fail(format!(
                        "{name} = {} → {got:?} at {:?}, sympy={want}",
                        truncate(val),
                        pt.subs
                    ));
                }
                None => {
                    return Status::NotImplemented(format!(
                        "cannot evaluate {name} = {}",
                        truncate(val)
                    ));
                }
            }
        }
    }
    Status::Pass
}

#[test]
fn linsolve_unique_solutions() {
    run_with_known_bugs("linsolve", "unique", KNOWN_BUGS, |ctx, fx| {
        let (eqs, vars) = match linsolve_inputs(ctx, fx) {
            Ok(v) => v,
            Err(e) => return Status::NotImplemented(e),
        };
        match linsolve(&eqs, &vars) {
            Ok(LinearSolution::Unique(pairs)) => check_linear_values(ctx, &pairs, fx),
            Ok(other) => Status::Fail(format!("expected a unique solution, got {other:?}")),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

#[test]
fn linsolve_symbolic_coefficients() {
    run_with_known_bugs("linsolve", "symbolic", KNOWN_BUGS, |ctx, fx| {
        let (eqs, vars) = match linsolve_inputs(ctx, fx) {
            Ok(v) => v,
            Err(e) => return Status::NotImplemented(e),
        };
        match linsolve(&eqs, &vars) {
            Ok(LinearSolution::Unique(pairs)) => check_linear_values(ctx, &pairs, fx),
            Ok(other) => Status::Fail(format!(
                "expected a (generic) unique solution, got {other:?}"
            )),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

#[test]
fn linsolve_parametric_families() {
    run_with_known_bugs("linsolve", "parametric", KNOWN_BUGS, |ctx, fx| {
        let (eqs, vars) = match linsolve_inputs(ctx, fx) {
            Ok(v) => v,
            Err(e) => return Status::NotImplemented(e),
        };
        let want_free = fx.u64("free_count").unwrap_or(0) as usize;
        match linsolve(&eqs, &vars) {
            Ok(LinearSolution::Parametric { solution, free }) => {
                if free.len() != want_free {
                    return Status::Fail(format!(
                        "free variables: symplex={} sympy={want_free}",
                        free.len()
                    ));
                }
                // Substitute the free variables with fixed values and verify
                // every equation is satisfied.
                let free_vals = [3i64, -2, 5, 7];
                let mut subs = solution.clone();
                for (i, fv) in free.iter().enumerate() {
                    let v = ctx.int(free_vals[i % free_vals.len()]);
                    for (_, val) in subs.iter_mut() {
                        *val = val.subs(fv, &v);
                    }
                }
                for eq in &eqs {
                    let mut e = eq.clone();
                    for (var, val) in &subs {
                        e = e.subs(var, val);
                    }
                    match e.eval_f64() {
                        Ok(r) if r.abs() < 1e-9 => {}
                        Ok(r) => {
                            return Status::Fail(format!(
                                "residual {r} for {} with {subs:?}",
                                truncate(eq)
                            ));
                        }
                        Err(err) => {
                            return Status::NotImplemented(format!(
                                "cannot evaluate residual: {err}"
                            ));
                        }
                    }
                }
                Status::Pass
            }
            Ok(other) => Status::Fail(format!("expected a parametric solution, got {other:?}")),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

#[test]
fn linsolve_inconsistent_systems() {
    run_with_known_bugs("linsolve", "inconsistent", KNOWN_BUGS, |ctx, fx| {
        let (eqs, vars) = match linsolve_inputs(ctx, fx) {
            Ok(v) => v,
            Err(e) => return Status::NotImplemented(e),
        };
        match linsolve(&eqs, &vars) {
            Ok(LinearSolution::Inconsistent) => Status::Pass,
            Ok(other) => Status::Fail(format!("expected Inconsistent, got {other:?}")),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

// ── nonlinear polynomial systems ───────────────────────────────────────

#[test]
fn nonlinear_polynomial_systems() {
    run_with_known_bugs("nonlinear_system", "polynomial", KNOWN_BUGS, |ctx, fx| {
        let (eqs, vars) = match linsolve_inputs(ctx, fx) {
            Ok(v) => v,
            Err(e) => return Status::NotImplemented(e),
        };
        let want: Vec<Vec<(f64, f64)>> = fx
            .field("solutions")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|tup| {
                        tup.as_array()?
                            .iter()
                            .map(|v| match Num::from_json(v)? {
                                Num::Finite(re, im) => Some((re, im)),
                                _ => None,
                            })
                            .collect()
                    })
                    .collect()
            })
            .unwrap_or_default();
        let sols = match symplex::polysys::solve_system_ex(&eqs, &vars) {
            Ok(s) => s,
            Err(e) => return Status::NotImplemented(format!("{e}")),
        };
        let mut got: Vec<Vec<(f64, f64)>> = Vec::new();
        for sol in &sols {
            let mut tup = Vec::new();
            for v in sol {
                match v.eval_complex64() {
                    Ok(c) => tup.push(c),
                    Err(e) => {
                        return Status::NotImplemented(format!(
                            "non-numeric solution {}: {e}",
                            truncate(v)
                        ));
                    }
                }
            }
            // Every reported solution must satisfy every equation.
            for eq in &eqs {
                let mut e = eq.clone();
                for (var, val) in vars.iter().zip(sol) {
                    e = e.subs(var, val);
                }
                if let Ok(r) = e.eval_complex64()
                    && r.0.hypot(r.1) > 1e-7
                {
                    return Status::Fail(format!(
                        "solution {tup:?} has residual {r:?} in {}",
                        truncate(eq)
                    ));
                }
            }
            got.push(tup);
        }
        let key = |t: &Vec<(f64, f64)>| -> Vec<(i64, i64)> {
            t.iter()
                .map(|(re, im)| ((re * 1e7).round() as i64, (im * 1e7).round() as i64))
                .collect()
        };
        got.sort_by_key(|t| key(t));
        let mut want = want;
        want.sort_by_key(|t| key(t));
        if got.len() != want.len() {
            return Status::Fail(format!(
                "solution count: symplex={} sympy={} ({got:?} vs {want:?})",
                got.len(),
                want.len()
            ));
        }
        for (g, w) in got.iter().zip(&want) {
            for (gc, wc) in g.iter().zip(w) {
                if !complex_matches(*gc, *wc, 1e-7) {
                    return Status::Fail(format!(
                        "solutions differ: symplex={got:?} sympy={want:?}"
                    ));
                }
            }
        }
        Status::Pass
    });
}

// ── ODE initial-value problems ─────────────────────────────────────────

#[test]
fn ode_initial_value_problems() {
    run_with_known_bugs("ode_ivp", "linear", KNOWN_BUGS, |ctx, fx| {
        let Some(ode) = fx.field("ode").and_then(|v| v.as_object()) else {
            return Status::SkippedOracle("no ode".into());
        };
        let x = ctx.symbol(fx.str("variable").unwrap_or("x"));
        let y = ctx.symbol(fx.str("func").unwrap_or("y"));
        let coeffs: Vec<Ex> = match ode.get("coeffs").and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .filter_map(|c| c.as_str())
                .map(|c| parse(ctx, c))
                .collect::<Result<Vec<_>, _>>()
        }) {
            Some(Ok(c)) => c,
            Some(Err(e)) => return Status::NotImplemented(e),
            None => return Status::SkippedOracle("no coeffs".into()),
        };
        let rhs = match parse(ctx, ode.get("rhs").and_then(|v| v.as_str()).unwrap_or("0")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        // Σ cᵢ y⁽ⁱ⁾ − rhs = 0
        let mut lhs = ctx.zero();
        let mut deriv = y.clone();
        for (i, c) in coeffs.iter().enumerate() {
            if i > 0 {
                deriv = deriv.formal_diff(&x);
            }
            lhs = &lhs + &(c * &deriv);
        }
        let eq = &lhs - &rhs;
        let mut ics: Vec<(usize, Ex, Ex)> = Vec::new();
        for ic in fx
            .field("ics")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
        {
            let Some(arr) = ic.as_array() else { continue };
            let order = arr[0].as_u64().unwrap_or(0) as usize;
            let (Ok(x0), Ok(val)) = (
                parse(ctx, arr[1].as_str().unwrap_or("0")),
                parse(ctx, arr[2].as_str().unwrap_or("0")),
            ) else {
                return Status::NotImplemented("bad initial condition".into());
            };
            ics.push((order, x0, val));
        }
        match eq.solve_ode_ivp(&y, &x, &ics) {
            Ok(sol) => {
                if sol.contains(&y) {
                    return Status::NotImplemented(format!("implicit solution {}", truncate(&sol)));
                }
                compare_eval_points(ctx, &sol, fx, "eval_points", TOLERANCE)
            }
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

// ── recurrences ────────────────────────────────────────────────────────

#[test]
fn rsolve_linear_recurrences() {
    run_with_known_bugs("rsolve", "linear", KNOWN_BUGS, |ctx, fx| {
        let n = ctx.symbol(fx.str("variable").unwrap_or("n"));
        let coeffs = match parse_all(ctx, &fx.str_list("coeffs")) {
            Ok(c) => c,
            Err(e) => return Status::NotImplemented(e),
        };
        let ics = match parse_all(ctx, &fx.str_list("ics")) {
            Ok(c) => c,
            Err(e) => return Status::NotImplemented(e),
        };
        let forcing = match fx.str("forcing") {
            Some(s) => match parse(ctx, s) {
                Ok(e) => Some(e),
                Err(e) => return Status::NotImplemented(e),
            },
            None => None,
        };
        let want: Vec<f64> = fx
            .field("values")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| Num::from_json(v)?.re()).collect())
            .unwrap_or_default();
        let sol = match symplex::rsolve::rsolve_linear(&coeffs, forcing.as_ref(), &n, &ics) {
            Ok(s) => s,
            Err(e) => return Status::NotImplemented(format!("{e}")),
        };
        if sol.has_unevaluated() {
            return Status::NotImplemented(format!("unevaluated {}", truncate(&sol)));
        }
        for (i, w) in want.iter().enumerate() {
            match sol.subs_i64(&n, i as i64).eval_complex64() {
                Ok(v) if complex_matches(v, (*w, 0.0), 1e-9) => {}
                Ok(v) => {
                    return Status::Fail(format!(
                        "a({i}) = {v:?}, expected {w} (closed form {})",
                        truncate(&sol)
                    ));
                }
                Err(e) => {
                    return Status::NotImplemented(format!(
                        "cannot evaluate a({i}) of {}: {e}",
                        truncate(&sol)
                    ));
                }
            }
        }
        Status::Pass
    });
}
