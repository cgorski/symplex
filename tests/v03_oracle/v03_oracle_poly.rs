//! SymPy oracle, 0.3 surface — the polynomial view (`Poly`, `Ex::as_poly`),
//! symbolic-coefficient `degree` / `coeff` / `leading_coeff`, `ratsimp`
//! against `sympy.cancel`, exact least-squares `poly_fit_exact` and Brent
//! root bracketing.
//!
//! Fixtures: `tests/fixtures/v03_cross_validation.json`
//! (`scripts/generate_v03_fixtures.py`); the runner and statuses are shared
//! with the 0.2 oracle (`v02_oracle_common/mod.rs`).
//!
//! Comparison rules:
//!
//! * rational coefficients / values are compared **exactly** as
//!   `Ratio<BigInt>` (the oracle stores them as `"-3/4"` strings);
//! * symbolic coefficients (`a + 1`, `1/a`, `sin(a)`) are compared at the
//!   fixture's parameter points with relative tolerance `1e-9`;
//! * `ratsimp` is compared with `cancel` at 5 exact rational points of the
//!   *original* expression (so any correct normal form agrees) plus the
//!   total degrees of numerator and denominator, never by string;
//! * `nroots` is compared as a sorted complex multiset (`1e-6`, repeated
//!   roots limit `f64` accuracy);
//! * Brent roots are compared with `nsolve(…, prec=30)` to `1e-9`.

use super::oracle_common;

use std::collections::BTreeMap;
use std::sync::OnceLock;

use num_bigint::BigInt;
use num_rational::Ratio;
use oracle_common::*;
use serde_json::Value;
use symplex::optimize::{RootOpts, brent_root, poly_fit_exact};
use symplex::poly_ex::Poly;
use symplex::prelude::*;

type Q = Ratio<BigInt>;

const KNOWN_BUGS: &[KnownBug] = &[];

// ── v03 fixture file ───────────────────────────────────────────────────

static V03_FILE: OnceLock<FixtureFile> = OnceLock::new();

fn v03() -> &'static FixtureFile {
    V03_FILE.get_or_init(|| {
        let json = include_str!("../fixtures/v03_cross_validation.json");
        let file: FixtureFile = serde_json::from_str(json).expect("v03 fixture JSON must parse");
        assert_eq!(
            file.fixture_count,
            file.fixtures.len(),
            "fixture_count is stale"
        );
        file
    })
}

fn run_v03(
    category: &str,
    subcategory: &str,
    process: impl Fn(&Context, &Fixture) -> Status + Send + Sync + 'static,
) {
    let file = v03();
    run_fixtures(
        &file.fixtures,
        &file.generated_by,
        category,
        Some(subcategory),
        KNOWN_BUGS,
        process,
    );
}

// ── Helpers ────────────────────────────────────────────────────────────

fn parse_q(s: &str) -> Result<Q, String> {
    s.trim()
        .parse::<Q>()
        .map_err(|e| format!("bad rational {s:?}: {e}"))
}

fn input(ctx: &Context, fx: &Fixture) -> Result<Ex, Status> {
    parse(ctx, fx.str("input").unwrap_or("")).map_err(Status::NotImplemented)
}

fn gens(ctx: &Context, fx: &Fixture, key: &str) -> Vec<Ex> {
    fx.str_list(key).iter().map(|g| ctx.symbol(g)).collect()
}

fn param_points(fx: &Fixture) -> Vec<BTreeMap<String, f64>> {
    fx.field("param_points")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_object)
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                        .collect()
                })
                .collect()
        })
        .unwrap_or_default()
}

fn u32_list(v: Option<&Value>) -> Vec<u32> {
    v.and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|u| u as u32))
                .collect()
        })
        .unwrap_or_default()
}

/// Compare a symplex coefficient with an oracle coefficient record
/// `{"coeff": sstr, "rational": bool, "values": [...]}`: exactly when the
/// oracle coefficient is rational, numerically at `pts` otherwise.
fn check_coeff(
    ctx: &Context,
    got: &Ex,
    rec: &Value,
    pts: &[BTreeMap<String, f64>],
    what: &str,
) -> Result<(), Status> {
    let want_str = rec.get("coeff").and_then(Value::as_str).unwrap_or("?");
    if rec.get("rational").and_then(Value::as_bool) == Some(true) {
        let want = parse_q(want_str).map_err(Status::SkippedOracle)?;
        return match got.eval().as_rational() {
            Some(g) if g == want => Ok(()),
            Some(g) => Err(Status::Fail(format!("{what}: symplex={g} sympy={want}"))),
            None => Err(Status::Fail(format!(
                "{what}: symplex={} is not a rational number, sympy={want}",
                truncate(got)
            ))),
        };
    }
    let vals: Vec<Num> = rec
        .get("values")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Num::from_json).collect())
        .unwrap_or_default();
    if vals.is_empty() || vals.len() != pts.len() {
        return Err(Status::SkippedOracle(format!(
            "{what}: oracle has {} values for {} parameter points",
            vals.len(),
            pts.len()
        )));
    }
    let mut evaluated = 0;
    for (pt, want) in pts.iter().zip(&vals) {
        let Num::Finite(re, im) = want else { continue };
        match eval_at(got, ctx, pt) {
            Some(v) => {
                evaluated += 1;
                if !complex_matches(v, (*re, *im), 1e-9) {
                    return Err(Status::Fail(format!(
                        "{what}: at {pt:?} symplex={} = ({}, {}i), sympy={want_str} = ({re}, {im}i)",
                        truncate(got),
                        v.0,
                        v.1
                    )));
                }
            }
            None => {
                return Err(Status::NotImplemented(format!(
                    "{what}: cannot evaluate {} at {pt:?}",
                    truncate(got)
                )));
            }
        }
    }
    if evaluated == 0 {
        return Err(Status::SkippedOracle(format!(
            "{what}: no finite oracle value"
        )));
    }
    Ok(())
}

fn as_poly_or(e: &Ex, gens: &[Ex]) -> Result<Poly, Status> {
    let refs: Vec<&Ex> = gens.iter().collect();
    e.as_poly(&refs).ok_or_else(|| {
        Status::NotImplemented(format!(
            "as_poly returned None for {} in {:?}",
            truncate(e),
            gens.iter().map(ToString::to_string).collect::<Vec<_>>()
        ))
    })
}

// ═══════════════════════════════════════════════════════════════════════
// poly
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn poly_as_dict_terms_and_coefficients() {
    run_v03("poly", "as_dict", |ctx, fx| {
        let e = match input(ctx, fx) {
            Ok(e) => e,
            Err(s) => return s,
        };
        let gens = gens(ctx, fx, "gens");
        let p = match as_poly_or(&e, &gens) {
            Ok(p) => p,
            Err(s) => return s,
        };
        let pts = param_points(fx);
        let want: Vec<&Value> = fx
            .field("terms")
            .and_then(Value::as_array)
            .map(|a| a.iter().collect())
            .unwrap_or_default();
        let got = p.terms();
        if got.len() != want.len() {
            return Status::Fail(format!(
                "term count: symplex={} sympy={} (symplex terms: {:?}, sympy: {})",
                got.len(),
                want.len(),
                got.iter()
                    .map(|(m, c)| format!("{m:?}: {c}"))
                    .collect::<Vec<_>>(),
                fx.str("sympy_result").unwrap_or("?")
            ));
        }
        for rec in &want {
            let exps = u32_list(rec.get("exps"));
            let Some((_, c)) = got.iter().find(|(m, _)| *m == exps) else {
                return Status::Fail(format!(
                    "monomial {exps:?} (coeff {}) missing from symplex terms {:?}",
                    rec.get("coeff").and_then(Value::as_str).unwrap_or("?"),
                    got.iter().map(|(m, _)| m.clone()).collect::<Vec<_>>()
                ));
            };
            if let Err(s) = check_coeff(ctx, c, rec, &pts, &format!("coeff of {exps:?}")) {
                return s;
            }
            // coeff_monomial must agree with terms().
            match p.coeff_monomial(&exps) {
                Ok(cm) if cm == *c => {}
                Ok(cm) => {
                    return Status::Fail(format!(
                        "coeff_monomial({exps:?}) = {cm} but terms() has {c}"
                    ));
                }
                Err(err) => return Status::Fail(format!("coeff_monomial({exps:?}): {err}")),
            }
        }
        // terms() must be in descending lexicographic order (SymPy's order).
        if got.windows(2).any(|w| w[0].0 <= w[1].0) {
            return Status::Fail(format!(
                "terms() not in descending lex order: {:?}",
                got.iter().map(|(m, _)| m.clone()).collect::<Vec<_>>()
            ));
        }
        if let Some(n) = fx.u64("num_terms")
            && p.num_terms() as u64 != n
        {
            return Status::Fail(format!("num_terms: symplex={} sympy={n}", p.num_terms()));
        }
        if let Some(d) = fx.u64("total_degree")
            && p.total_degree() != Some(d as u32)
        {
            return Status::Fail(format!(
                "total_degree: symplex={:?} sympy={d}",
                p.total_degree()
            ));
        }
        Status::Pass
    });
}

#[test]
fn poly_degree_total_and_per_generator() {
    run_v03("poly", "degree", |ctx, fx| {
        let e = match input(ctx, fx) {
            Ok(e) => e,
            Err(s) => return s,
        };
        let gens = gens(ctx, fx, "gens");
        let p = match as_poly_or(&e, &gens) {
            Ok(p) => p,
            Err(s) => return s,
        };
        let want_total = fx.u64("total_degree").map(|d| d as u32);
        let want_list = u32_list(fx.field("degree_list"));
        if p.total_degree() != want_total {
            return Status::Fail(format!(
                "total_degree: symplex={:?} sympy={want_total:?}",
                p.total_degree()
            ));
        }
        if p.degree_list() != want_list {
            return Status::Fail(format!(
                "degree_list: symplex={:?} sympy={want_list:?}",
                p.degree_list()
            ));
        }
        for (g, d) in gens.iter().zip(&want_list) {
            if p.degree_in(g) != Some(*d) {
                return Status::Fail(format!(
                    "degree_in({g}): symplex={:?} sympy={d}",
                    p.degree_in(g)
                ));
            }
            // Ex::degree treats the other generators as symbolic coefficients.
            if e.degree(g) != Some(*d as usize) {
                return Status::Fail(format!(
                    "Ex::degree({g}): symplex={:?} sympy={d}",
                    e.degree(g)
                ));
            }
        }
        if let Some(h) = fx.bool("is_homogeneous")
            && p.is_homogeneous() != h
        {
            return Status::Fail(format!(
                "is_homogeneous: symplex={} sympy={h}",
                p.is_homogeneous()
            ));
        }
        Status::Pass
    });
}

#[test]
fn poly_leading_coefficient_and_monomial() {
    run_v03("poly", "LC", |ctx, fx| {
        let e = match input(ctx, fx) {
            Ok(e) => e,
            Err(s) => return s,
        };
        let gens = gens(ctx, fx, "gens");
        let p = match as_poly_or(&e, &gens) {
            Ok(p) => p,
            Err(s) => return s,
        };
        let pts = param_points(fx);
        let Some(rec) = fx.field("LC") else {
            return Status::SkippedOracle("no LC".into());
        };
        let want_lm = u32_list(fx.field("LM"));
        if p.leading_monomial().as_deref() != Some(want_lm.as_slice()) {
            return Status::Fail(format!(
                "leading monomial: symplex={:?} sympy={want_lm:?}",
                p.leading_monomial()
            ));
        }
        if let Err(s) = check_coeff(ctx, &p.leading_coeff(), rec, &pts, "Poly::leading_coeff") {
            return s;
        }
        if gens.len() == 1 {
            match e.leading_coeff(&gens[0]) {
                Some(lc) => {
                    if let Err(s) = check_coeff(ctx, &lc, rec, &pts, "Ex::leading_coeff") {
                        return s;
                    }
                }
                None => return Status::Fail("Ex::leading_coeff returned None".into()),
            }
        }
        Status::Pass
    });
}

#[test]
fn poly_all_coeffs_dense_univariate() {
    run_v03("poly", "all_coeffs", |ctx, fx| {
        let e = match input(ctx, fx) {
            Ok(e) => e,
            Err(s) => return s,
        };
        let v = ctx.symbol(fx.str("var").unwrap_or("x"));
        let p = match as_poly_or(&e, std::slice::from_ref(&v)) {
            Ok(p) => p,
            Err(s) => return s,
        };
        let pts = param_points(fx);
        let want: Vec<&Value> = fx
            .field("all_coeffs")
            .and_then(Value::as_array)
            .map(|a| a.iter().collect())
            .unwrap_or_default();
        let Some(got) = p.all_coeffs() else {
            return Status::Fail("Poly::all_coeffs returned None for a univariate poly".into());
        };
        if got.len() != want.len() {
            return Status::Fail(format!(
                "all_coeffs length: symplex={} sympy={}",
                got.len(),
                want.len()
            ));
        }
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            if let Err(s) = check_coeff(ctx, g, w, &pts, &format!("Poly::all_coeffs[{i}]")) {
                return s;
            }
        }
        // Ex::coeffs is the same list in ascending order.
        let Some(asc) = e.coeffs(&v) else {
            return Status::Fail("Ex::coeffs returned None".into());
        };
        if asc.len() != want.len() {
            return Status::Fail(format!(
                "Ex::coeffs length: symplex={} sympy={}",
                asc.len(),
                want.len()
            ));
        }
        for (i, (g, w)) in asc.iter().rev().zip(&want).enumerate() {
            if let Err(s) = check_coeff(ctx, g, w, &pts, &format!("Ex::coeffs (desc)[{i}]")) {
                return s;
            }
        }
        if let Some(d) = fx.u64("degree")
            && p.degree_in(&v) != Some(d as u32)
        {
            return Status::Fail(format!("degree: symplex={:?} sympy={d}", p.degree_in(&v)));
        }
        Status::Pass
    });
}

#[test]
fn poly_eval_exact_rational_points() {
    run_v03("poly", "eval", |ctx, fx| {
        let e = match input(ctx, fx) {
            Ok(e) => e,
            Err(s) => return s,
        };
        let gens = gens(ctx, fx, "gens");
        let p = match as_poly_or(&e, &gens) {
            Ok(p) => p,
            Err(s) => return s,
        };
        let Some(points) = fx.field("points").and_then(Value::as_array) else {
            return Status::SkippedOracle("no points".into());
        };
        for pt in points {
            let vals: Vec<Q> = match pt
                .get("values")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .map(|s| parse_q(s.as_str().unwrap_or("")))
                        .collect::<Result<Vec<_>, _>>()
                })
                .unwrap_or_else(|| Err("no values".into()))
            {
                Ok(v) => v,
                Err(e) => return Status::SkippedOracle(e),
            };
            let want = match parse_q(pt.get("result").and_then(Value::as_str).unwrap_or("")) {
                Ok(w) => w,
                Err(e) => return Status::SkippedOracle(e),
            };
            let exs: Vec<Ex> = vals.iter().map(|q| ctx.from_ratio(q.clone())).collect();
            let refs: Vec<&Ex> = exs.iter().collect();
            match p.eval(&refs) {
                Ok(v) => match v.as_rational() {
                    Some(g) if g == want => {}
                    Some(g) => {
                        return Status::Fail(format!("eval at {vals:?}: symplex={g} sympy={want}"));
                    }
                    None => {
                        return Status::Fail(format!(
                            "eval at {vals:?}: symplex={} is not rational",
                            truncate(&v)
                        ));
                    }
                },
                Err(err) => return Status::NotImplemented(format!("eval: {err}")),
            }
        }
        Status::Pass
    });
}

#[test]
fn poly_nroots_complex_multiset() {
    run_v03("poly", "nroots", |ctx, fx| {
        let e = match input(ctx, fx) {
            Ok(e) => e,
            Err(s) => return s,
        };
        let v = ctx.symbol(fx.str("var").unwrap_or("x"));
        let p = match as_poly_or(&e, std::slice::from_ref(&v)) {
            Ok(p) => p,
            Err(s) => return s,
        };
        let want: Vec<Num> = fx
            .field("roots")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Num::from_json).collect())
            .unwrap_or_default();
        let Some(want) = nums_to_complex(&want) else {
            return Status::SkippedOracle("non-finite oracle root".into());
        };
        let roots = match p.nroots(15) {
            Ok(r) => r,
            Err(err) => return Status::NotImplemented(format!("Poly::nroots: {err}")),
        };
        if let Some(d) = fx.u64("degree")
            && roots.len() as u64 != d
        {
            return Status::Fail(format!("root count: symplex={} degree={d}", roots.len()));
        }
        if let Status::Fail(r) = compare_complex_multisets(roots, want.clone(), 1e-6) {
            return Status::Fail(format!("Poly::nroots: {r}"));
        }
        match e.nroots(&v, 15) {
            Ok(r) => compare_complex_multisets(r, want, 1e-6),
            Err(err) => Status::NotImplemented(format!("Ex::nroots: {err}")),
        }
    });
}

#[test]
fn poly_degree_with_symbolic_coefficients() {
    run_v03("poly", "degree_symbolic", |ctx, fx| {
        let e = match input(ctx, fx) {
            Ok(e) => e,
            Err(s) => return s,
        };
        let v = ctx.symbol(fx.str("var").unwrap_or("x"));
        let Some(want) = fx.u64("degree") else {
            return Status::SkippedOracle("no degree".into());
        };
        match e.degree(&v) {
            Some(d) if d as u64 == want => {}
            Some(d) => return Status::Fail(format!("Ex::degree: symplex={d} sympy={want}")),
            None => return Status::Fail(format!("Ex::degree returned None, sympy={want}")),
        }
        match e.as_poly(&[&v]) {
            Some(p) => match p.degree_in(&v) {
                Some(d) if d as u64 == want => Status::Pass,
                other => Status::Fail(format!("Poly::degree_in: symplex={other:?} sympy={want}")),
            },
            None => Status::Fail("as_poly returned None for a polynomial input".into()),
        }
    });
}

#[test]
fn poly_coeff_with_symbolic_coefficients() {
    run_v03("poly", "coeff_symbolic", |ctx, fx| {
        let e = match input(ctx, fx) {
            Ok(e) => e,
            Err(s) => return s,
        };
        let v = ctx.symbol(fx.str("var").unwrap_or("x"));
        let power = fx.u64("power").unwrap_or(0) as usize;
        let pts = param_points(fx);
        let Some(rec) = fx.field("coeff") else {
            return Status::SkippedOracle("no coeff".into());
        };
        let Some(c) = e.coeff(&v, power) else {
            return Status::Fail("Ex::coeff returned None".into());
        };
        if let Err(s) = check_coeff(ctx, &c, rec, &pts, &format!("Ex::coeff({v}, {power})")) {
            return s;
        }
        let Some(p) = e.as_poly(&[&v]) else {
            return Status::Fail("as_poly returned None for a polynomial input".into());
        };
        match p.coeff_monomial(&[power as u32]) {
            Ok(cm) => {
                if let Err(s) = check_coeff(ctx, &cm, rec, &pts, "Poly::coeff_monomial") {
                    return s;
                }
            }
            Err(err) => return Status::Fail(format!("coeff_monomial: {err}")),
        }
        Status::Pass
    });
}

// ═══════════════════════════════════════════════════════════════════════
// ratsimp vs cancel
// ═══════════════════════════════════════════════════════════════════════

/// Exact rational substitutions `(symbol name, value)`.
type ExactSubs = Vec<(String, Q)>;

/// `(substitutions, value)` pairs with exact rationals.
fn exact_points(fx: &Fixture) -> Result<Vec<(ExactSubs, Q)>, String> {
    let Some(arr) = fx.field("eval_points").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(arr.len());
    for p in arr {
        let subs = p
            .get("subs")
            .and_then(Value::as_object)
            .ok_or("eval point without subs")?;
        let mut pairs = Vec::with_capacity(subs.len());
        for (k, v) in subs {
            pairs.push((k.clone(), parse_q(v.as_str().unwrap_or(""))?));
        }
        let value = parse_q(p.get("value").and_then(Value::as_str).unwrap_or(""))?;
        out.push((pairs, value));
    }
    Ok(out)
}

fn eval_exact(ctx: &Context, e: &Ex, subs: &ExactSubs) -> Option<Q> {
    let syms: Vec<Ex> = subs.iter().map(|(k, _)| ctx.symbol(k)).collect();
    let vals: Vec<Ex> = subs
        .iter()
        .map(|(_, q)| ctx.from_ratio(q.clone()))
        .collect();
    let pairs: Vec<(&Ex, &Ex)> = syms.iter().zip(&vals).collect();
    e.subs_map(&pairs).eval().as_rational()
}

/// Total degree of a polynomial expression in `gens`; zero counts as 0
/// (SymPy's `Poly(0, …).total_degree()`).
fn total_degree_in(e: &Ex, gens: &[Ex]) -> Result<u32, String> {
    let refs: Vec<&Ex> = gens.iter().collect();
    let p = e
        .as_poly(&refs)
        .ok_or_else(|| format!("{} is not polynomial in the fixture symbols", truncate(e)))?;
    Ok(p.total_degree().unwrap_or(0))
}

#[test]
fn ratsimp_matches_cancel_at_points_and_in_degree() {
    run_v03("ratsimp", "cancel", |ctx, fx| {
        let e = match input(ctx, fx) {
            Ok(e) => e,
            Err(s) => return s,
        };
        let gens = gens(ctx, fx, "symbols");
        let pts = match exact_points(fx) {
            Ok(p) if !p.is_empty() => p,
            Ok(_) => return Status::SkippedOracle("no eval points".into()),
            Err(err) => return Status::SkippedOracle(err),
        };
        let r = e.ratsimp();
        if r.has_unevaluated() {
            return Status::NotImplemented(format!("unevaluated result: {}", truncate(&r)));
        }
        for (subs, want) in &pts {
            // Sanity: the parsed input itself must reproduce the oracle value.
            match eval_exact(ctx, &e, subs) {
                Some(v) if v == *want => {}
                other => {
                    return Status::Fail(format!(
                        "original expression at {subs:?}: symplex={other:?} sympy={want}"
                    ));
                }
            }
            match eval_exact(ctx, &r, subs) {
                Some(v) if v == *want => {}
                Some(v) => {
                    return Status::Fail(format!(
                        "ratsimp = {} at {subs:?}: symplex={v} sympy={want} (cancel = {})",
                        truncate(&r),
                        fx.str("sympy_result").unwrap_or("?")
                    ));
                }
                None => {
                    return Status::Fail(format!(
                        "ratsimp = {} does not evaluate to a rational at {subs:?} (cancel = {})",
                        truncate(&r),
                        fx.str("sympy_result").unwrap_or("?")
                    ));
                }
            }
        }
        // Degrees of the single fraction.
        let (n, d) = r.as_numer_denom();
        let want_n = fx.u64("numer_degree").unwrap_or(0) as u32;
        let want_d = fx.u64("denom_degree").unwrap_or(0) as u32;
        let (dn, dd) = match (total_degree_in(&n, &gens), total_degree_in(&d, &gens)) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(err), _) | (_, Err(err)) => {
                return Status::Fail(format!(
                    "ratsimp = {} is not a single polynomial fraction: {err}",
                    truncate(&r)
                ));
            }
        };
        if (dn, dd) != (want_n, want_d) {
            return Status::Fail(format!(
                "degrees (numer, denom): symplex=({dn}, {dd}) from {} / {}, sympy=({want_n}, {want_d}) from {}",
                truncate(&n),
                truncate(&d),
                fx.str("sympy_result").unwrap_or("?")
            ));
        }
        Status::Pass
    });
}

// ═══════════════════════════════════════════════════════════════════════
// optimize
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn optimize_polyfit_exact_least_squares() {
    run_v03("optimize", "polyfit_exact", |ctx, fx| {
        let degree = fx.u64("degree").unwrap_or(0) as usize;
        let pts: Vec<(Q, Q)> = match fx
            .field("points")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|p| {
                        let arr = p.as_array().ok_or("point is not an array")?;
                        Ok((
                            parse_q(arr[0].as_str().unwrap_or(""))?,
                            parse_q(arr[1].as_str().unwrap_or(""))?,
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .unwrap_or_else(|| Err("no points".into()))
        {
            Ok(p) => p,
            Err(err) => return Status::SkippedOracle(err),
        };
        let want: Vec<Q> = match fx
            .str_list("coeffs")
            .iter()
            .map(|s| parse_q(s))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(w) => w,
            Err(err) => return Status::SkippedOracle(err),
        };
        let got = match poly_fit_exact(&pts, degree) {
            Ok(c) => c,
            Err(err) => return Status::NotImplemented(format!("poly_fit_exact: {err}")),
        };
        if got != want {
            return Status::Fail(format!(
                "poly_fit_exact: symplex={:?} sympy={:?}",
                got.iter().map(ToString::to_string).collect::<Vec<_>>(),
                want.iter().map(ToString::to_string).collect::<Vec<_>>()
            ));
        }
        // Ex::poly_fit_points must produce the same polynomial.
        let x = ctx.symbol("x");
        let ex_pts: Vec<(Ex, Ex)> = pts
            .iter()
            .map(|(px, py)| (ctx.from_ratio(px.clone()), ctx.from_ratio(py.clone())))
            .collect();
        let p = match Ex::poly_fit_points(ctx, &ex_pts, &x, degree) {
            Ok(p) => p,
            Err(err) => return Status::NotImplemented(format!("poly_fit_points: {err}")),
        };
        for (k, w) in want.iter().enumerate() {
            match p.coeff(&x, k).and_then(|c| c.eval().as_rational()) {
                Some(c) if c == *w => {}
                other => {
                    return Status::Fail(format!(
                        "poly_fit_points coefficient of x^{k}: symplex={other:?} sympy={w} (p = {})",
                        truncate(&p)
                    ));
                }
            }
        }
        Status::Pass
    });
}

#[test]
fn optimize_brent_root_bracket() {
    run_v03("optimize", "brent_root", |ctx, fx| {
        let f = match input(ctx, fx) {
            Ok(e) => e,
            Err(s) => return s,
        };
        let x = ctx.symbol(fx.str("var").unwrap_or("x"));
        let bracket: Vec<f64> = fx
            .field("bracket")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        let (Some(&lo), Some(&hi)) = (bracket.first(), bracket.get(1)) else {
            return Status::SkippedOracle("no bracket".into());
        };
        let Some(want) = fx.field("root_f64").and_then(Value::as_f64) else {
            return Status::SkippedOracle("no root".into());
        };
        let check = |r: f64, what: &str| -> Result<(), Status> {
            if (r - want).abs() <= 1e-9 {
                Ok(())
            } else {
                Err(Status::Fail(format!(
                    "{what}: symplex={r} sympy={want} (|Δ| = {:.3e})",
                    (r - want).abs()
                )))
            }
        };
        // Primary: compiled expression through find_root_bracket.
        let primary = f.find_root_bracket(&x, lo, hi);
        // Secondary: the raw Brent kernel on an exact-evaluation closure.
        let g = |t: f64| -> f64 {
            let mut subs = BTreeMap::new();
            subs.insert(x.to_string(), t);
            eval_at(&f, ctx, &subs).map_or(f64::NAN, |(re, _)| re)
        };
        let secondary = brent_root(g, lo, hi, &RootOpts::default());
        match (primary, secondary) {
            (Ok(r1), Ok(r2)) => {
                if let Err(s) = check(r1, "find_root_bracket") {
                    return s;
                }
                if let Err(s) = check(r2, "brent_root") {
                    return s;
                }
                Status::Pass
            }
            (Ok(r1), Err(e2)) => {
                if let Err(s) = check(r1, "find_root_bracket") {
                    return s;
                }
                Status::Fail(format!("brent_root on the evaluation closure failed: {e2}"))
            }
            (Err(e1), Ok(r2)) => {
                if let Err(s) = check(r2, "brent_root") {
                    return s;
                }
                Status::NotImplemented(format!(
                    "find_root_bracket cannot compile {}: {e1} (brent_root on the closure agrees)",
                    truncate(&f)
                ))
            }
            (Err(e1), Err(e2)) => {
                Status::NotImplemented(format!("find_root_bracket: {e1}; brent_root: {e2}"))
            }
        }
    });
}
