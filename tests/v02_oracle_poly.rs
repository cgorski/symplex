//! SymPy oracle, 0.2 surface — polynomial algebra: multivariate factoring,
//! `factor_list`, resultants, discriminants, square-free decomposition,
//! numeric roots, gcd/lcm, partial fractions, division, decomposition.

mod v02_oracle_common;

use symplex::prelude::*;
use v02_oracle_common::*;

const KNOWN_BUGS: &[KnownBug] = &[];

/// `(factor string, multiplicity)` pairs from the fixture.
fn oracle_factors(fx: &Fixture, key: &str) -> Vec<(String, u32)> {
    fx.field(key)
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    let arr = p.as_array()?;
                    Some((arr[0].as_str()?.to_string(), arr[1].as_u64()? as u32))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Total degree of a monomial (structural walk over Mul / Pow / Symbol).
fn monomial_degree(t: &Ex) -> Option<u32> {
    match t.expr_type() {
        ExprType::Number | ExprType::Constant => Some(0),
        ExprType::Symbol => Some(1),
        ExprType::Neg => monomial_degree(&t.args()[0]),
        ExprType::Mul => t.args().iter().map(monomial_degree).sum(),
        ExprType::Pow => {
            let a = t.args();
            let base_deg = monomial_degree(&a[0])?;
            let exp = a[1].as_i64()?;
            if exp < 0 {
                return None;
            }
            Some(base_deg * exp as u32)
        }
        _ => None,
    }
}

/// Total degree of a polynomial in all its free symbols (expand, then take
/// the maximum monomial degree).
fn total_degree(_ctx: &Context, e: &Ex) -> Option<u32> {
    let e = e.expand();
    let terms = if e.expr_type() == ExprType::Add {
        e.args()
    } else {
        vec![e]
    };
    terms.iter().map(monomial_degree).max().flatten()
}

/// Compare the *shape* of a factorization: multiset of (total degree,
/// multiplicity), and the product reconstructs the input numerically.
fn check_factorization(
    ctx: &Context,
    input: &Ex,
    content: &Ex,
    factors: &[(Ex, u32)],
    fx: &Fixture,
) -> Status {
    let want = oracle_factors(fx, "factors");
    let want_content = fx.str("content").unwrap_or("1");
    // Shape: sorted (degree, multiplicity) lists must agree.
    let mut got_shape: Vec<(u32, u32)> = Vec::new();
    for (f, m) in factors {
        match total_degree(ctx, f) {
            Some(d) => got_shape.push((d, *m)),
            None => {
                return Status::NotImplemented(format!("factor {} is not polynomial", truncate(f)));
            }
        }
    }
    let mut want_shape: Vec<(u32, u32)> = Vec::new();
    for (f, m) in &want {
        let fe = match parse(ctx, f) {
            Ok(e) => e,
            Err(e) => return Status::SkippedOracle(e),
        };
        want_shape.push((total_degree(ctx, &fe).unwrap_or(0), *m));
    }
    got_shape.sort();
    want_shape.sort();
    // Reconstruct and compare numerically with the input.
    let mut prod = content.clone();
    for (f, m) in factors {
        prod = &prod * &f.powi(i64::from(*m));
    }
    let recon = compare_eval_points(ctx, &prod, fx, "eval_points", TOLERANCE);
    if let Status::Fail(r) = recon {
        return Status::Fail(format!("factor product != input: {r}"));
    }
    if got_shape != want_shape {
        return Status::Fail(format!(
            "factorization shape (degree, mult): symplex={got_shape:?} sympy={want_shape:?} (symplex: {} · {:?}, sympy: {} · {:?})",
            truncate(content),
            factors
                .iter()
                .map(|(f, m)| format!("({f})^{m}"))
                .collect::<Vec<_>>(),
            want_content,
            want
        ));
    }
    let _ = input;
    Status::Pass
}

#[test]
fn factor_multivariate() {
    run_with_known_bugs("factor", "multivariate", KNOWN_BUGS, |ctx, fx| {
        let f = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let (content, factors) = f.factor_list_all();
        check_factorization(ctx, &f, &content, &factors, fx)
    });
}

#[test]
fn factor_list_univariate() {
    run_with_known_bugs("factor", "univariate_list", KNOWN_BUGS, |ctx, fx| {
        let f = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let x = ctx.symbol(fx.str("variable").unwrap_or("x"));
        let (content, factors) = f.factor_list(&x);
        check_factorization(ctx, &f, &content, &factors, fx)
    });
}

fn two_polys(ctx: &Context, fx: &Fixture) -> Result<(Ex, Ex, Ex), String> {
    let f = parse(ctx, fx.str("input_f").ok_or("no input_f")?)?;
    let g = parse(ctx, fx.str("input_g").ok_or("no input_g")?)?;
    let x = ctx.symbol(fx.str("variable").unwrap_or("x"));
    Ok((f, g, x))
}

fn constant_or_points(ctx: &Context, r: &Ex, fx: &Fixture) -> Status {
    if !fx.eval_points("eval_points").is_empty() {
        return compare_eval_points(ctx, r, fx, "eval_points", TOLERANCE);
    }
    match fx.num("value") {
        Some(v) => compare_constant(ctx, r, &v, TOLERANCE),
        None => Status::SkippedOracle("no oracle value".into()),
    }
}

fn resultant(ctx: &Context, fx: &Fixture) -> Status {
    let (f, g, x) = match two_polys(ctx, fx) {
        Ok(v) => v,
        Err(e) => return Status::NotImplemented(e),
    };
    match f.resultant(&g, &x) {
        Some(r) => constant_or_points(ctx, &r, fx),
        None => Status::NotImplemented("resultant returned None".into()),
    }
}

#[test]
fn resultant_integer() {
    run_with_known_bugs("resultant", "integer", KNOWN_BUGS, resultant);
}

#[test]
fn resultant_symbolic_coefficients() {
    run_with_known_bugs("resultant", "symbolic", KNOWN_BUGS, resultant);
}

fn discriminant(ctx: &Context, fx: &Fixture) -> Status {
    let f = match parse(ctx, fx.str("input").unwrap_or("")) {
        Ok(e) => e,
        Err(e) => return Status::NotImplemented(e),
    };
    let x = ctx.symbol(fx.str("variable").unwrap_or("x"));
    match f.discriminant(&x) {
        Some(r) => constant_or_points(ctx, &r, fx),
        None => Status::NotImplemented("discriminant returned None".into()),
    }
}

#[test]
fn discriminant_integer() {
    run_with_known_bugs("discriminant", "integer", KNOWN_BUGS, discriminant);
}

#[test]
fn discriminant_symbolic_coefficients() {
    run_with_known_bugs("discriminant", "symbolic", KNOWN_BUGS, discriminant);
}

#[test]
fn sqf_list_univariate() {
    run_with_known_bugs("sqf_list", "univariate", KNOWN_BUGS, |ctx, fx| {
        let f = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let x = ctx.symbol(fx.str("variable").unwrap_or("x"));
        let Some((content, parts)) = f.sqf_list(&x) else {
            return Status::NotImplemented("sqf_list returned None".into());
        };
        // Oracle parts: [factor, multiplicity, degree]
        let mut want: Vec<(u32, u32)> = fx
            .field("parts")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|p| {
                        let arr = p.as_array()?;
                        Some((arr[1].as_u64()? as u32, arr[2].as_u64()? as u32))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut got: Vec<(u32, u32)> = Vec::new();
        for (p, m) in &parts {
            match p.degree(&x) {
                Some(d) => got.push((*m, d as u32)),
                None => {
                    return Status::NotImplemented(format!("part {} not polynomial", truncate(p)));
                }
            }
        }
        got.sort();
        want.sort();
        let mut prod = content.clone();
        for (p, m) in &parts {
            prod = &prod * &p.powi(i64::from(*m));
        }
        if let Status::Fail(r) = compare_eval_points(ctx, &prod, fx, "eval_points", TOLERANCE) {
            return Status::Fail(format!("sqf product != input: {r}"));
        }
        if got != want {
            return Status::Fail(format!(
                "(multiplicity, degree) lists differ: symplex={got:?} sympy={want:?}"
            ));
        }
        Status::Pass
    });
}

#[test]
fn nroots_all_complex_roots() {
    run_with_known_bugs("nroots", "complex", KNOWN_BUGS, |ctx, fx| {
        let f = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let x = ctx.symbol(fx.str("variable").unwrap_or("x"));
        let want: Vec<Num> = fx
            .field("roots")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(Num::from_json).collect())
            .unwrap_or_default();
        let Some(want) = nums_to_complex(&want) else {
            return Status::SkippedOracle("non-finite oracle root".into());
        };
        match f.nroots(&x, 15) {
            // Repeated roots limit attainable accuracy to ~1e-8 in double precision.
            Ok(roots) => compare_complex_multisets(roots, want, 1e-6),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

fn gcd_or_lcm(ctx: &Context, fx: &Fixture) -> Status {
    let (f, g, x) = match two_polys(ctx, fx) {
        Ok(v) => v,
        Err(e) => return Status::NotImplemented(e),
    };
    let r = if fx.subcategory == "gcd" {
        f.poly_gcd(&g, &x)
    } else {
        f.poly_lcm(&g, &x)
    };
    let Some(r) = r else {
        return Status::NotImplemented(format!("poly_{} returned None", fx.subcategory));
    };
    let want_deg = fx.u64("degree").unwrap_or(0);
    match r.degree(&x) {
        Some(d) if d as u64 == want_deg => {}
        Some(d) => {
            return Status::Fail(format!(
                "degree: symplex={d} ({}) sympy={want_deg}",
                truncate(&r)
            ));
        }
        None => return Status::NotImplemented(format!("result {} not polynomial", truncate(&r))),
    }
    // gcd/lcm are unique up to a nonzero constant: the ratio to SymPy's
    // result must be the same at every eval point.
    let pts = fx.eval_points("eval_points");
    let mut ratio: Option<f64> = None;
    for pt in &pts {
        let Some(Num::Finite(w, _)) = pt.value else {
            continue;
        };
        if w.abs() < 1e-9 {
            continue;
        }
        let Some((gv, _)) = eval_at(&r, ctx, &pt.subs) else {
            return Status::NotImplemented(format!("cannot evaluate {}", truncate(&r)));
        };
        let q = gv / w;
        match ratio {
            None => ratio = Some(q),
            Some(prev) if approx_eq(prev, q) => {}
            Some(prev) => {
                return Status::Fail(format!(
                    "{} is not a constant multiple of SymPy's {}: ratios {prev} vs {q}",
                    truncate(&r),
                    fx.str("sympy_result").unwrap_or("?")
                ));
            }
        }
    }
    if ratio.is_none() {
        return Status::SkippedOracle("no usable eval points".into());
    }
    Status::Pass
}

#[test]
fn poly_gcd() {
    run_with_known_bugs("poly_gcd_lcm", "gcd", KNOWN_BUGS, gcd_or_lcm);
}

#[test]
fn poly_lcm() {
    run_with_known_bugs("poly_gcd_lcm", "lcm", KNOWN_BUGS, gcd_or_lcm);
}

#[test]
fn apart_partial_fractions() {
    run_with_known_bugs("apart", "univariate", KNOWN_BUGS, |ctx, fx| {
        let f = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let x = ctx.symbol(fx.str("variable").unwrap_or("x"));
        let r = f.partial_fractions(&x);
        // A partial-fraction result must be a sum whose terms each have a
        // denominator of degree ≤ that of an irreducible factor power; we
        // check the weaker structural property that no term keeps the whole
        // original denominator when the input had ≥ 2 distinct factors.
        let status = compare_eval_points(ctx, &r, fx, "eval_points", TOLERANCE);
        if !matches!(status, Status::Pass) {
            return status;
        }
        let (_, den_in) = f.together().as_numer_denom();
        let in_deg = den_in.degree(&x).unwrap_or(0);
        let terms = if r.expr_type() == ExprType::Add {
            r.args()
        } else {
            vec![r.clone()]
        };
        let max_term_den_deg = terms
            .iter()
            .map(|t| t.together().as_numer_denom().1.degree(&x).unwrap_or(0))
            .max()
            .unwrap_or(0);
        if terms.len() == 1
            && in_deg >= 2
            && max_term_den_deg == in_deg
            && fx
                .str("sympy_result")
                .is_some_and(|s| s.contains(" + ") || s.contains(" - "))
        {
            return Status::NotImplemented(format!("not decomposed: {}", truncate(&r)));
        }
        Status::Pass
    });
}

#[test]
fn poly_div_quotient_and_remainder() {
    run_with_known_bugs("poly_div", "univariate", KNOWN_BUGS, |ctx, fx| {
        let (f, g, x) = match two_polys(ctx, fx) {
            Ok(v) => v,
            Err(e) => return Status::NotImplemented(e),
        };
        let Some((q, r)) = f.poly_div(&g, &x) else {
            return Status::NotImplemented("poly_div returned None".into());
        };
        let sq = compare_eval_points(ctx, &q, fx, "quotient_points", TOLERANCE);
        if !matches!(sq, Status::Pass) {
            return match sq {
                Status::Fail(m) => Status::Fail(format!("quotient: {m}")),
                other => other,
            };
        }
        let rem_pts = fx.eval_points("remainder_points");
        if rem_pts.is_empty() {
            // remainder is zero
            return if r.eval().is_zero_structural() {
                Status::Pass
            } else {
                Status::Fail(format!("remainder should be 0, got {}", truncate(&r)))
            };
        }
        match compare_eval_points(ctx, &r, fx, "remainder_points", TOLERANCE) {
            Status::Fail(m) => Status::Fail(format!("remainder: {m}")),
            other => other,
        }
    });
}

#[test]
fn decompose_functional_composition() {
    run_with_known_bugs("decompose", "univariate", KNOWN_BUGS, |ctx, fx| {
        let f = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let x = ctx.symbol(fx.str("variable").unwrap_or("x"));
        let parts = f.decompose(&x);
        if parts.is_empty() {
            return Status::NotImplemented("decompose returned nothing".into());
        }
        // Recompose outermost-first: f = p0(p1(...(pk(x)))).
        let mut comp = parts.last().unwrap().clone();
        for p in parts.iter().rev().skip(1) {
            comp = p.subs(&x, &comp);
        }
        if let Status::Fail(m) = compare_eval_points(ctx, &comp, fx, "eval_points", TOLERANCE) {
            return Status::Fail(format!("composition != input: {m}"));
        }
        let want_count = fx.u64("component_count").unwrap_or(1) as usize;
        if parts.len() < want_count {
            return Status::NotImplemented(format!(
                "found {} components ({:?}), SymPy found {want_count}",
                parts.len(),
                parts.iter().map(ToString::to_string).collect::<Vec<_>>()
            ));
        }
        Status::Pass
    });
}
