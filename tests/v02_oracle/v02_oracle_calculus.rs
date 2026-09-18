//! SymPy oracle, 0.2 surface — calculus: definite/improper/numeric integrals,
//! symbolic sums and products, convergence, one-sided limits, asymptotic
//! series, residues.  One `#[test]` per fixture subcategory; see the header
//! of `v02_oracle_common/mod.rs` for the honesty policy.

use super::v02_oracle_common;

use symplex::prelude::*;
use v02_oracle_common::*;

/// Library bugs surfaced by this file (strict xfail — see common module).
const KNOWN_BUGS: &[KnownBug] = &[];

/// Regression: `series_at_infinity(atan(x))` used to return the garbage
/// expression `atan(zoo)` as its constant term.
#[test]
fn bug_series_at_infinity_atan_returns_atan_zoo() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // `n_terms` bounds the exponent of 1/x (exclusive), as for
    // `series_at_infinity(x/(x+1), 3)` = 1 - 1/x + 1/x^2; six terms give
    // pi/2 - 1/x + 1/(3x^3) - 1/(5x^5), accurate to ~1e-8 at x = 10.
    let s = x
        .atan()
        .try_series_at_infinity(&x, 6)
        .expect("a Laurent series");
    assert!(!s.contains(&ctx.complex_infinity()), "got {s}");
    let at_10 = s.subs_i64(&x, 10).eval_f64().unwrap();
    assert!(
        (at_10 - 10f64.atan()).abs() < 1e-6,
        "series at x=10 gives {at_10}"
    );
}

// ── definite integrals ─────────────────────────────────────────────────

fn integrand_and_bounds(ctx: &Context, fx: &Fixture) -> Result<(Ex, Ex, Ex, Ex), String> {
    if let Some(assume) = fx.field("assumptions").and_then(|v| v.as_object()) {
        for (name, kind) in assume {
            let props = match kind.as_str() {
                Some("positive") => vec![Assumption::Positive],
                Some("real") => vec![Assumption::Real],
                _ => vec![],
            };
            ctx.symbol_with(name, &props);
        }
    }
    let f = parse(ctx, fx.str("input").ok_or("no input")?)?;
    let x = ctx.symbol(fx.str("variable").unwrap_or("x"));
    let lo = parse(ctx, fx.str("lower").ok_or("no lower")?)?;
    let hi = parse(ctx, fx.str("upper").ok_or("no upper")?)?;
    Ok((f, x, lo, hi))
}

fn definite_integral(ctx: &Context, fx: &Fixture) -> Status {
    let (f, x, lo, hi) = match integrand_and_bounds(ctx, fx) {
        Ok(v) => v,
        Err(e) => return Status::NotImplemented(e),
    };
    let result = match f.try_integrate_definite(&x, &lo, &hi) {
        Ok(r) => r,
        Err(SymplexError::Divergent { .. }) => {
            return match fx.num("value") {
                Some(Num::PosInf | Num::NegInf | Num::ComplexInf | Num::NaN) => Status::Pass,
                Some(v) => Status::Fail(format!("symplex says divergent, sympy={v}")),
                None => Status::NotImplemented("symplex says divergent, no oracle value".into()),
            };
        }
        Err(e) => return Status::NotImplemented(format!("{e}")),
    };
    if !fx.eval_points("eval_points").is_empty() {
        return compare_eval_points(ctx, &result, fx, "eval_points", TOLERANCE);
    }
    match fx.num("value") {
        Some(v) => compare_constant(ctx, &result, &v, TOLERANCE),
        None => Status::SkippedOracle("no oracle value".into()),
    }
}

#[test]
fn definite_integral_proper() {
    run("definite_integral", "proper", definite_integral);
}

#[test]
fn definite_integral_improper() {
    run("definite_integral", "improper", definite_integral);
}

#[test]
fn definite_integral_infinite() {
    run("definite_integral", "infinite", definite_integral);
}

#[test]
fn definite_integral_symmetric() {
    run("definite_integral", "symmetric", definite_integral);
}

#[test]
fn definite_integral_parametric() {
    run("definite_integral", "parametric", definite_integral);
}

#[test]
fn definite_integral_numeric_quadrature() {
    run("definite_integral", "numeric", |ctx, fx| {
        let (f, x, lo, hi) = match integrand_and_bounds(ctx, fx) {
            Ok(v) => v,
            Err(e) => return Status::NotImplemented(e),
        };
        let Some(Num::Finite(re, _)) = fx.num("value") else {
            return Status::SkippedOracle("no finite oracle value".into());
        };
        match f.integrate_numeric(&x, &lo, &hi) {
            // Adaptive G7/K15 targets ~1e-10; allow 1e-8 relative.
            Ok(v) if approx_eq_tol(v, re, 1e-8) => Status::Pass,
            Ok(v) => Status::Fail(format!("integrate_numeric={v}, sympy={re}")),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

// ── summation / products ───────────────────────────────────────────────

fn sum_or_product(ctx: &Context, fx: &Fixture, is_product: bool) -> Status {
    let f = match parse(ctx, fx.str("input").unwrap_or("")) {
        Ok(e) => e,
        Err(e) => return Status::NotImplemented(e),
    };
    let k = ctx.symbol(fx.str("variable").unwrap_or("k"));
    let lo = match parse(ctx, fx.str("lower").unwrap_or("1")) {
        Ok(e) => e,
        Err(e) => return Status::NotImplemented(e),
    };
    let hi = match parse(ctx, fx.str("upper").unwrap_or("n")) {
        Ok(e) => e,
        Err(e) => return Status::NotImplemented(e),
    };
    let outcome = if is_product {
        f.try_product_over(&k, &lo, &hi)
    } else {
        f.try_summation(&k, &lo, &hi)
    };
    let result = match outcome {
        Ok(r) => r,
        Err(SymplexError::Divergent { .. }) => {
            return match fx.num("value") {
                Some(Num::PosInf | Num::NegInf | Num::ComplexInf | Num::NaN) => Status::Pass,
                Some(v) => Status::Fail(format!("symplex says divergent, sympy={v}")),
                None => Status::NotImplemented("symplex says divergent, no oracle value".into()),
            };
        }
        Err(e) => return Status::NotImplemented(format!("{e}")),
    };
    if !fx.eval_points("eval_points").is_empty() {
        // Closed form in `n`: compare against brute-force values at n = 1..6.
        return compare_eval_points(ctx, &result, fx, "eval_points", TOLERANCE);
    }
    match fx.num("value") {
        Some(v) => compare_constant(ctx, &result, &v, TOLERANCE),
        None => Status::SkippedOracle("no oracle value".into()),
    }
}

#[test]
fn summation_finite_symbolic_upper_bound() {
    run("summation", "finite_symbolic", |ctx, fx| {
        sum_or_product(ctx, fx, false)
    });
}

#[test]
fn summation_infinite() {
    run("summation", "infinite", |ctx, fx| {
        sum_or_product(ctx, fx, false)
    });
}

#[test]
fn summation_divergent() {
    run("summation", "divergent", |ctx, fx| {
        let f = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let k = ctx.symbol("k");
        match f.try_summation(&k, &ctx.int(1), &ctx.infinity()) {
            Err(SymplexError::Divergent { .. }) => Status::Pass,
            Err(e) => Status::NotImplemented(format!("{e}")),
            Ok(r) => match extended_value(ctx, &r) {
                Some(Num::PosInf | Num::NegInf) => Status::Pass,
                _ => Status::Fail(format!("divergent sum evaluated to {}", truncate(&r))),
            },
        }
    });
}

#[test]
fn summation_convergence_test() {
    run("summation", "convergence", |ctx, fx| {
        let f = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let k = ctx.symbol("k");
        let Some(expected) = fx.bool("value_bool") else {
            return Status::SkippedOracle("no oracle verdict".into());
        };
        match f.is_convergent(&k) {
            Some(v) if v == expected => Status::Pass,
            Some(v) => Status::Fail(format!("is_convergent={v}, expected {expected}")),
            None => Status::NotImplemented("is_convergent undecided".into()),
        }
    });
}

#[test]
fn product_finite_symbolic_upper_bound() {
    run("product", "finite_symbolic", |ctx, fx| {
        sum_or_product(ctx, fx, true)
    });
}

#[test]
fn product_infinite() {
    run("product", "infinite", |ctx, fx| {
        sum_or_product(ctx, fx, true)
    });
}

// ── one-sided limits ───────────────────────────────────────────────────

fn one_sided_limit(ctx: &Context, fx: &Fixture) -> Status {
    let f = match parse(ctx, fx.str("input").unwrap_or("")) {
        Ok(e) => e,
        Err(e) => return Status::NotImplemented(e),
    };
    let x = ctx.symbol(fx.str("variable").unwrap_or("x"));
    let pt = match parse(ctx, fx.str("point").unwrap_or("0")) {
        Ok(e) => e,
        Err(e) => return Status::NotImplemented(e),
    };
    let Some(expected) = fx.num("value") else {
        return Status::SkippedOracle("no oracle value".into());
    };
    let dir = if fx.subcategory == "left" {
        Direction::Left
    } else {
        Direction::Right
    };
    match f.try_limit_dir(&x, &pt, dir) {
        Ok(r) => compare_constant(ctx, &r, &expected, TOLERANCE),
        Err(e) => Status::NotImplemented(format!("{e}")),
    }
}

#[test]
fn limit_from_the_left() {
    run("limit", "left", one_sided_limit);
}

#[test]
fn limit_from_the_right() {
    run("limit", "right", one_sided_limit);
}

// ── series at infinity ─────────────────────────────────────────────────

/// Decompose `Σ c·x^e` into `(e, c)` pairs by evaluating each term at
/// x = 1, 2, 3 (works for any real exponent; returns None on anything that
/// is not a monomial in `x`).
fn monomial_coefficients(sum: &Ex, x: &Ex) -> Option<Vec<(f64, (f64, f64))>> {
    let expanded = sum.expand();
    let terms = if expanded.expr_type() == ExprType::Add {
        expanded.args()
    } else {
        vec![expanded]
    };
    let mut out = Vec::new();
    for t in terms {
        let c = t.subs_i64(x, 1).eval_complex64().ok()?;
        let at2 = t.subs_i64(x, 2).eval_complex64().ok()?;
        let at3 = t.subs_i64(x, 3).eval_complex64().ok()?;
        let cm = c.0.hypot(c.1);
        if cm < 1e-300 {
            continue;
        }
        let r2 = at2.0.hypot(at2.1) / cm;
        let r3 = at3.0.hypot(at3.1) / cm;
        let e = r2.log2();
        // Consistency: 3^e must match too, otherwise not a monomial.
        if (r3 - 3f64.powf(e)).abs() > 1e-6 * r3.max(1.0) {
            return None;
        }
        out.push((e, c));
    }
    Some(out)
}

#[test]
fn series_at_infinity_laurent_coefficients() {
    run_with_known_bugs("series_at_infinity", "laurent", KNOWN_BUGS, |ctx, fx| {
        let f = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let x = ctx.symbol(fx.str("variable").unwrap_or("x"));
        let n_terms = fx.u64("n_terms").unwrap_or(6) as u32;
        let want: Vec<(f64, (f64, f64))> = fx
            .field("coefficients")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|c| {
                        let e = c.get("exp")?.as_f64()?;
                        match Num::from_json(c.get("value")?)? {
                            Num::Finite(re, im) => Some((e, (re, im))),
                            _ => None,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        if want.is_empty() {
            return Status::SkippedOracle("no oracle coefficients".into());
        }
        let ser = match f.try_series_at_infinity(&x, n_terms) {
            Ok(s) => s,
            Err(e) => return Status::NotImplemented(format!("{e}")),
        };
        if ser.contains(&ctx.complex_infinity()) || ser.contains(&ctx.nan()) {
            return Status::Fail(format!(
                "series_at_infinity returned a non-series object: {}",
                truncate(&ser)
            ));
        }
        let Some(got) = monomial_coefficients(&ser, &x) else {
            return Status::NotImplemented(format!(
                "result is not a sum of monomials in x: {}",
                truncate(&ser)
            ));
        };
        // Compare every exponent present in either side down to the coarser
        // truncation order; coefficients missing on one side are zero.
        let min_got = got.iter().map(|(e, _)| *e).fold(f64::INFINITY, f64::min);
        let min_want = want.iter().map(|(e, _)| *e).fold(f64::INFINITY, f64::min);
        let cutoff = min_got.max(min_want);
        let coeff_at = |list: &[(f64, (f64, f64))], e: f64| -> (f64, f64) {
            list.iter()
                .filter(|(ee, _)| (ee - e).abs() < 1e-9)
                .fold((0.0, 0.0), |acc, (_, c)| (acc.0 + c.0, acc.1 + c.1))
        };
        let mut exps: Vec<f64> = got.iter().chain(want.iter()).map(|(e, _)| *e).collect();
        exps.sort_by(|a, b| b.partial_cmp(a).unwrap());
        exps.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let mut compared = 0;
        let mut mismatches = Vec::new();
        for e in exps {
            if e < cutoff - 1e-9 {
                continue;
            }
            let g = coeff_at(&got, e);
            let w = coeff_at(&want, e);
            compared += 1;
            if !complex_matches(g, w, 1e-9) {
                mismatches.push(format!("x^{e}: symplex={g:?} sympy={w:?}"));
            }
        }
        if compared < 2 {
            return Status::NotImplemented(format!(
                "too few comparable terms in {}",
                truncate(&ser)
            ));
        }
        if mismatches.is_empty() {
            Status::Pass
        } else {
            Status::Fail(format!("{} ({})", mismatches.join("; "), truncate(&ser)))
        }
    });
}

// ── residues ───────────────────────────────────────────────────────────

fn residue(ctx: &Context, fx: &Fixture) -> Status {
    let f = match parse(ctx, fx.str("input").unwrap_or("")) {
        Ok(e) => e,
        Err(e) => return Status::NotImplemented(e),
    };
    let z = ctx.symbol(fx.str("variable").unwrap_or("z"));
    let Some(expected) = fx.num("value") else {
        return Status::SkippedOracle("no oracle value".into());
    };
    let r = if fx.str("point") == Some("oo") {
        f.residue_at_infinity(&z)
    } else {
        let pt = match parse(ctx, fx.str("point").unwrap_or("0")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        f.residue(&z, &pt)
    };
    compare_constant(ctx, &r, &expected, TOLERANCE)
}

#[test]
fn residue_at_finite_poles() {
    run("residue", "finite", residue);
}

#[test]
fn residue_at_infinity() {
    run("residue", "infinity", residue);
}
