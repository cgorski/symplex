//! Integration tests for the 0.2 series work: the truncated-series engine
//! behind `Ex::series` / `Ex::maclaurin`, asymptotic expansions
//! (`series_at_infinity`), the `Ex`-based `FormalPowerSeries`, convergence
//! tests, and the `Ex`-based finite-difference API.

use num_bigint::BigInt;
use num_rational::Ratio;
use symplex::finite_diff::{apply_finite_diff, equispaced_grid, finite_diff_weights};
use symplex::formal_series::FormalPowerSeries;
use symplex::prelude::*;

fn rat(p: i64, q: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(p), BigInt::from(q))
}

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Check `series ≈ f` numerically at a small point (truncation error ≪ tol).
fn check_series(f: &Ex, s: &Ex, x: &Ex, at: (i64, i64), tol: f64) {
    let ctx = f.context();
    let pt = ctx.rational(at.0, at.1);
    let sv = s.subs(x, &pt).eval_f64().unwrap();
    let fv = f.subs(x, &pt).eval_f64().unwrap();
    assert!(
        approx(sv, fv, tol),
        "series of {f}: {s}\n  at {}/{}: {sv} vs {fv}",
        at.0,
        at.1
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Series engine
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn maclaurin_of_special_functions_is_exact_and_fast() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: Vec<(Ex, &str)> = vec![
        (x.tan(), "17/315*x^7 + 2/15*x^5 + 1/3*x^3 + x"),
        (x.tanh(), "-17/315*x^7 + 2/15*x^5 - 1/3*x^3 + x"),
        (x.asin(), "5/112*x^7 + 3/40*x^5 + 1/6*x^3 + x"),
        (x.asinh(), "-5/112*x^7 + 3/40*x^5 - 1/6*x^3 + x"),
        (x.atan(), "-1/7*x^7 + 1/5*x^5 - 1/3*x^3 + x"),
        (x.atanh(), "1/7*x^7 + 1/5*x^5 + 1/3*x^3 + x"),
        (x.sinh(), "1/5040*x^7 + 1/120*x^5 + 1/6*x^3 + x"),
        (
            x.lambertw(),
            "16807/720*x^7 - 54/5*x^6 + 125/24*x^5 - 8/3*x^4 + 3/2*x^3 - x^2 + x",
        ),
        (
            (&ctx.int(1) + &x).ln(),
            "1/7*x^7 - 1/6*x^6 + 1/5*x^5 - 1/4*x^4 + 1/3*x^3 - 1/2*x^2 + x",
        ),
    ];
    for (f, expected) in cases {
        let s = f.maclaurin(&x, 8);
        assert_eq!(s.to_string(), expected, "maclaurin({f}, 8)");
    }
    // High order stays fast (closed-form coefficients, no 40 derivatives).
    let start = std::time::Instant::now();
    let s = x.tan().maclaurin(&x, 40);
    assert!(start.elapsed().as_secs_f64() < 5.0, "tan series too slow");
    assert!(s.to_string().contains("x^39"), "{s}");
    let s = x.erf().maclaurin(&x, 6);
    check_series(&x.erf(), &s, &x, (1, 10), 1e-8);
}

#[test]
fn series_of_compositions_products_and_quotients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    // exp(sin x) = 1 + x + x²/2 − x⁴/8 − x⁵/15 + …
    let f = x.sin().exp();
    assert_eq!(
        f.maclaurin(&x, 6).to_string(),
        "-1/15*x^5 - 1/8*x^4 + 1/2*x^2 + x + 1"
    );
    // sin(x)/x → 1 − x²/6 + x⁴/120
    let f = &x.sin() / &x;
    assert_eq!(f.maclaurin(&x, 5).to_string(), "1/120*x^4 - 1/6*x^2 + 1");
    // tan x = sin x / cos x — via arithmetic
    let f = &x.sin() / &x.cos();
    assert_eq!(f.maclaurin(&x, 6).to_string(), "2/15*x^5 + 1/3*x^3 + x");
    // 1/(1 − x − x²): Fibonacci generating function
    let f = &one / &(&(&one - &x) - &x.powi(2));
    assert_eq!(
        f.maclaurin(&x, 7).to_string(),
        "13*x^6 + 8*x^5 + 5*x^4 + 3*x^3 + 2*x^2 + x + 1"
    );
    // (1 + x)^x
    let f = (&one + &x).pow(&x);
    assert_eq!(f.maclaurin(&x, 4).to_string(), "-1/2*x^3 + x^2 + 1");
    // sqrt(1 + sin x)
    let f = (&one + &x.sin()).sqrt();
    check_series(&f, &f.maclaurin(&x, 6), &x, (1, 20), 1e-9);
    // cos(x)² + sin(x)² = 1 exactly to every order
    let f = &x.cos().powi(2) + &x.sin().powi(2);
    assert_eq!(f.maclaurin(&x, 12).to_string(), "1");
    // e^{a x} with a symbolic parameter
    let a = ctx.symbol("a");
    let f = (&a * &x).exp();
    assert_eq!(f.maclaurin(&x, 3).to_string(), "1/2*a^2*x^2 + a*x + 1");
    // Laurent: 1/sin x = 1/x + x/6 + 7x³/360
    let f = &one / &x.sin();
    assert_eq!(f.maclaurin(&x, 4).to_string(), "7/360*x^3 + 1/x + 1/6*x");
    // cot x − 1/x = −x/3 − x³/45
    let f = &(&x.cos() / &x.sin()) - &(&one / &x);
    assert_eq!(f.maclaurin(&x, 5).to_string(), "-1/45*x^3 - 1/3*x");
}

#[test]
fn series_around_nonzero_points_and_via_series_expr() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ln x around 1
    let s = x.ln().series(&x, &ctx.int(1), 4);
    check_series(&x.ln(), &s, &x, (11, 10), 1e-4);
    // sin x around π/2 = 1 − (x−π/2)²/2 + …
    let pi_2 = &ctx.pi() / 2;
    let s = x.sin().series(&x, &pi_2, 4);
    let v = s
        .subs(&x, &(&pi_2 + &ctx.rational(1, 10)))
        .eval_f64()
        .unwrap();
    assert!(
        approx(v, (std::f64::consts::FRAC_PI_2 + 0.1).sin(), 1e-5),
        "{s}"
    );
    // Around a symbolic point a (Taylor with symbolic derivatives)
    let a = ctx.symbol("a");
    let s = x.exp().series(&x, &a, 3);
    let at = s
        .subs_i64(&a, 1)
        .subs(&x, &ctx.rational(11, 10))
        .eval_f64()
        .unwrap();
    assert!(approx(at, (1.1f64).exp(), 1e-3), "{s}");
}

#[test]
fn puiseux_and_log_singularities_stay_unevaluated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.sqrt() * &x.sin();
    let s = f.series(&x, &ctx.int(0), 5);
    assert!(s.has_unevaluated(), "{s}");
    assert!(s.to_string().starts_with("Series("), "{s}");
    assert!(f.try_series(&x, &ctx.int(0), 5).is_err());
    assert!(x.ln().try_maclaurin(&x, 4).is_err());
    assert!(x.sin().ln().try_maclaurin(&x, 4).is_err());
    // x^(3/2) alone is Puiseux too
    assert!(x.pow(&ctx.rational(3, 2)).try_maclaurin(&x, 4).is_err());
    // but (x²)^(1/2)·… where the fractional power resolves to an even power is fine:
    let f = x.powi(4).sqrt();
    assert_eq!(f.maclaurin(&x, 5).to_string(), "x^2");
    // and (x²)^(1/2) = |x| is NOT expanded as x
    assert!(x.powi(2).sqrt().try_maclaurin(&x, 5).is_err());
}

#[test]
fn asymptotic_expansions_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let f = &x / &(&x + 1);
    let s = f.series_at_infinity(&x, 4);
    assert_eq!(s.to_string(), "x^(-2) - 1/x - x^(-3) + 1");
    let f = (&x.powi(2) + 1).sqrt() - &x;
    let s = f.series_at_infinity(&x, 6);
    assert_eq!(s.to_string(), "-1/8*x^(-3) + 1/16*x^(-5) + 1/2*1/x");
    // x·sin(1/x) = 1 − 1/(6x²) + 1/(120x⁴)
    let f = &x * &(&one / &x).sin();
    let s = f.series_at_infinity(&x, 5);
    let expected = &(&one - &(&x.powi(-2) / 6)) + &(&x.powi(-4) / 120);
    assert_eq!((&s - &expected).expand().to_string(), "0", "{s}");
    // (x+1)/(x−1) as x → −∞
    let f = &(&x + 1) / &(&x - 1);
    let s = f.series_at_neg_infinity(&x, 3);
    let v = s.subs_i64(&x, -1000).eval_f64().unwrap();
    assert!(approx(v, 999.0 / 1001.0, 1e-8), "{s}");
    // Ex::series with point = oo routes to the asymptotic expansion
    let s2 = (&x / &(&x + 1)).series(&x, &ctx.infinity(), 4);
    assert_eq!(s2.to_string(), "x^(-2) - 1/x - x^(-3) + 1");
    // Something with no expansion at infinity (essential singularity e^x): unevaluated
    let s = x.exp().series_at_infinity(&x, 3);
    assert!(s.has_unevaluated(), "{s}");
    assert!(x.exp().try_series_at_infinity(&x, 3).is_err());
    // e^{1/x} does have one: 1 + 1/x + 1/(2x²)
    let s = (&one / &x).exp().series_at_infinity(&x, 3);
    let expected = &(&one + &x.powi(-1)) + &(&x.powi(-2) / 2);
    assert_eq!((&s - &expected).expand().to_string(), "0", "{s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Formal power series
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fps_coefficients_general_terms_and_truncation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let k = ctx.symbol("k");
    let e = x.exp().fps_maclaurin(&x);
    assert!(e.has_closed_form());
    assert_eq!(e.coefficient_rational(7), Some(rat(1, 5040)));
    assert_eq!(e.general_term(&k).unwrap().to_string(), "1/k!");
    assert_eq!(e.truncate(4).to_string(), "1/6*x^3 + 1/2*x^2 + x + 1");
    assert_eq!(e.coefficients(3).len(), 3);
    assert_eq!(e.variable().to_string(), "x");
    assert_eq!(e.point().to_string(), "0");
    // general term of geometric series
    let g = (&ctx.int(1) - &x).powi(-1).fps_maclaurin(&x);
    let gt = g.general_term(&k).unwrap();
    for i in 0..6 {
        assert_eq!(gt.subs_i64(&k, i).eval().to_string(), "1");
    }
    // ln(1 + x): general term (−1)^(k+1)/k, a_0 = 0
    let l = (&ctx.int(1) + &x).ln().fps_maclaurin(&x);
    let gt = l.general_term(&k).unwrap();
    assert_eq!(gt.subs_i64(&k, 0).eval().to_string(), "0");
    assert_eq!(gt.subs_i64(&k, 3).eval().to_string(), "1/3");
    assert_eq!(gt.subs_i64(&k, 4).eval().to_string(), "-1/4");
    // symbolic-parameter coefficients: exp(a x)
    let a = ctx.symbol("a");
    let ea = (&a * &x).exp().fps_maclaurin(&x);
    assert_eq!(ea.coefficient(2).to_string(), "1/2*a^2");
    assert!(ea.coefficient_rational(2).is_none());
    // non-elementary → engine, no general term
    let f = (&x.exp() * &x.cos()).fps_maclaurin(&x);
    assert!(!f.has_closed_form());
    assert!(f.general_term(&k).is_none());
    assert_eq!(f.coefficient_rational(3), Some(rat(-1, 3)));
    assert_eq!(f.truncate(4).to_string(), "-1/3*x^3 + x + 1");
    // derived series still carry a general term through add/scale/derivative
    let d = e
        .add(&x.sin().fps_maclaurin(&x))
        .unwrap()
        .scale(&ctx.int(2))
        .derivative();
    assert!(d.has_closed_form());
    let gt = d.general_term(&k).unwrap();
    for i in 0..6usize {
        let direct = d.coefficient(i).eval_f64().unwrap();
        let formula = gt.subs_i64(&k, i as i64).eval_f64().unwrap();
        assert!(
            approx(direct, formula, 1e-12),
            "k={i}: {direct} vs {formula}"
        );
    }
}

#[test]
fn fps_arithmetic_identities() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = x.sin().fps_maclaurin(&x);
    let c = x.cos().fps_maclaurin(&x);
    let e = x.exp().fps_maclaurin(&x);
    // sin·cos = sin(2x)/2
    let sc = s.mul(&c).unwrap();
    let half_sin2x = (&x * 2).sin().fps_maclaurin(&x).scale(&ctx.rational(1, 2));
    for k in 0..10 {
        assert_eq!(
            sc.coefficient_rational(k),
            half_sin2x.coefficient_rational(k),
            "k={k}"
        );
    }
    // (1/cos)·cos = 1
    let one = c.inverse().unwrap().mul(&c).unwrap();
    assert_eq!(one.coefficient_rational(0), Some(rat(1, 1)));
    for k in 1..10 {
        assert_eq!(one.coefficient_rational(k), Some(rat(0, 1)), "k={k}");
    }
    // sec x = 1 + x²/2 + 5x⁴/24 + 61x⁶/720
    let sec = c.inverse().unwrap();
    assert_eq!(sec.coefficient_rational(6), Some(rat(61, 720)));
    // tan = sin · sec ; reversion(tan) = atan
    let tan = s.mul(&sec).unwrap();
    assert_eq!(tan.coefficient_rational(5), Some(rat(2, 15)));
    let atan = tan.reversion().unwrap();
    assert_eq!(
        atan.truncate(8).to_string(),
        "-1/7*x^7 + 1/5*x^5 - 1/3*x^3 + x"
    );
    // reversion(sin) = asin ; reversion(e^x − 1) = ln(1 + x)
    let asin = s.reversion().unwrap();
    assert_eq!(asin.coefficient_rational(5), Some(rat(3, 40)));
    let em1 = e
        .sub(&FormalPowerSeries::from_coefficients(
            &x,
            &ctx.int(0),
            &[ctx.int(1)],
        ))
        .unwrap();
    let ln1p = em1.reversion().unwrap();
    assert_eq!(
        ln1p.truncate(5).to_string(),
        "-1/4*x^4 + 1/3*x^3 - 1/2*x^2 + x"
    );
    // compose: cos(sin x)
    let cs = c.compose(&s).unwrap();
    check_series(&x.sin().cos(), &cs.truncate(8), &x, (1, 5), 1e-6);
    // integral of 1/(1+x) = ln(1+x)
    let inv = (&ctx.int(1) + &x).powi(-1).fps_maclaurin(&x).integral();
    for k in 0..8 {
        assert_eq!(
            inv.coefficient_rational(k),
            ln1p.coefficient_rational(k),
            "k={k}"
        );
    }
    // derivative of e^x is e^x
    let de = e.derivative();
    for k in 0..8 {
        assert_eq!(de.coefficient_rational(k), e.coefficient_rational(k));
    }
    // preconditions
    assert!(s.inverse().is_err());
    assert!(c.reversion().is_err());
    assert!(e.compose(&c).is_err());
    let y = ctx.symbol("y");
    assert!(s.add(&y.sin().fps_maclaurin(&y)).is_err());
}

#[test]
fn fps_about_nonzero_point_and_laurent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ln x about 1: a_k = (−1)^(k+1)/k
    let l = x.ln().fps(&x, &ctx.int(1));
    assert_eq!(l.coefficient_rational(0), Some(rat(0, 1)));
    assert_eq!(l.coefficient_rational(1), Some(rat(1, 1)));
    assert_eq!(l.coefficient_rational(2), Some(rat(-1, 2)));
    let t = l.truncate(3);
    let v = t.subs(&x, &ctx.rational(11, 10)).eval_f64().unwrap();
    assert!(approx(v, 1.1f64.ln(), 1e-3), "{t}");
    // 1/(x (1 − x)) has a simple pole: truncate shows 1/x
    let f = &ctx.int(1) / &(&x * &(&ctx.int(1) - &x));
    let s = f.fps_maclaurin(&x);
    assert_eq!(s.coefficient_rational(2), Some(rat(1, 1)));
    assert_eq!(s.truncate(3).to_string(), "x^2 + x + 1/x + 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Convergence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergence_tests_are_decisive_when_they_should_be() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let one = ctx.int(1);
    let m1 = ctx.int(-1);
    let cases: Vec<(Ex, Option<bool>, &str)> = vec![
        (k.powi(-2), Some(true), "1/k²"),
        (k.powi(-1), Some(false), "1/k"),
        (&one / &(&k * &k.ln().powi(2)), Some(true), "1/(k ln² k)"),
        (&one / &(&k * &k.ln()), Some(false), "1/(k ln k)"),
        (m1.pow(&k) / &k, Some(true), "(−1)^k/k"),
        (&k.factorial() / &k.pow(&k), Some(true), "k!/k^k"),
        (&ctx.int(2).pow(&k) / &k.factorial(), Some(true), "2^k/k!"),
        (&k / &(&k + 1), Some(false), "k/(k+1)"),
        (k.pow(&ctx.rational(-1, 2)), Some(false), "1/√k"),
        (&k.powi(10) / &ctx.int(2).pow(&k), Some(true), "k^10/2^k"),
        (
            &k.factorial() / &ctx.int(10).pow(&k),
            Some(false),
            "k!/10^k",
        ),
        (
            &(&k * 2).factorial() / &(&k.factorial().powi(2) * &ctx.int(4).pow(&k)),
            Some(false),
            "C(2k,k)/4^k",
        ),
        (
            &(&k * 2).factorial() / &(&k.factorial().powi(2) * &ctx.int(5).pow(&k)),
            Some(true),
            "C(2k,k)/5^k",
        ),
        (&k.sin() / &k.powi(2), Some(true), "sin k/k²"),
        (&k.sin() / &k, None, "sin k/k"),
        (k.powi(-1) - (&k + 1).powi(-1), Some(true), "1/k − 1/(k+1)"),
        (&one / &(&k.powi(2) + 1), Some(true), "1/(k²+1)"),
        (&k / &(&k.powi(2) + 1), Some(false), "k/(k²+1)"),
        (
            ctx.rational(1, 2).pow(&k) * &k.powi(3),
            Some(true),
            "k³/2^k",
        ),
        (k.pow(&k) / k.factorial(), Some(false), "k^k/k!"),
        (k.pow(&k) / (&k * 3).factorial(), Some(true), "k^k/(3k)!"),
        (ctx.e().pow(&k) / &k.factorial(), Some(true), "e^k/k!"),
        (ctx.pi().pow(&(-&k)), Some(true), "π^{-k}"),
        (ctx.int(1).pow(&k), Some(false), "1^k"),
    ];
    for (body, expected, label) in cases {
        assert_eq!(body.is_convergent(&k), expected, "Σ {label}  ({body})");
    }
    // absolute convergence
    let alt = m1.pow(&k) / &k;
    assert_eq!(alt.is_absolutely_convergent(&k), Some(false));
    let alt2 = m1.pow(&k) / &k.powi(2);
    assert_eq!(alt2.is_absolutely_convergent(&k), Some(true));
    assert_eq!(
        (&k.sin() / &k.powi(2)).is_absolutely_convergent(&k),
        Some(true)
    );
    // symbolic parameter: decided when the factorial dominates, otherwise unknown
    let x = ctx.symbol("x");
    assert_eq!((x.pow(&k) / k.factorial()).is_convergent(&k), Some(true));
    assert_eq!(x.pow(&k).is_convergent(&k), None);
}

// ═══════════════════════════════════════════════════════════════════════════
// Finite differences (Ex-based API)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn finite_difference_ex_api() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let h = ctx.symbol("h");
    let grid = equispaced_grid(&x, &h, 2);
    assert_eq!(grid.len(), 5);
    // five-point second derivative weights: [-1, 16, -30, 16, -1]/(12h²)
    let w = finite_diff_weights(2, &grid, &x);
    let scaled: Vec<String> = w
        .iter()
        .map(|wi| (wi * &h.powi(2) * 12).simplify().to_string())
        .collect();
    assert_eq!(scaled, ["-1", "16", "-30", "16", "-1"]);
    // apply to f = x⁴ values: exact 12x² + O(h⁴) term vanishes for 5 points → exactly 12x²
    let ys: Vec<Ex> = grid.iter().map(|p| p.powi(4)).collect();
    let d2 = apply_finite_diff(2, &grid, &ys, &x).unwrap().expand();
    assert_eq!(d2.to_string(), "12*x^2");
    // Ex::differentiate_finite: first derivative of sin on a 3-point stencil, numerically
    let d = x
        .sin()
        .differentiate_finite(&x, &equispaced_grid(&x, &h, 1), 1);
    let v = d
        .subs_i64(&x, 1)
        .subs(&h, &ctx.rational(1, 1000))
        .eval_f64()
        .unwrap();
    assert!(approx(v, 1f64.cos(), 1e-6), "{d}");
    // replacement of formal Derivative nodes, nested → order 2
    let e = x.powi(3).formal_diff(&x).formal_diff(&x);
    let d2 = e
        .differentiate_finite(&x, &equispaced_grid(&x, &h, 1), 0)
        .expand();
    assert_eq!(d2.to_string(), "6*x");
    // errors
    assert!(apply_finite_diff(1, &grid, &ys[..3], &x).is_err());
}
