//! Formal power series beyond Taylor series: Puiseux series, symbolic
//! powers `x^a·f(x)`, Laurent operands in series arithmetic, and refusals
//! (never `nan` coefficients).
//!
//! Up to 0.47 every function without a Laurent expansion (`√x·eˣ`,
//! `x^a·eˣ`, `exp(√x)`, `ln x`) gave `FormalPowerSeries(nan)`, and series
//! arithmetic ignored the negative powers of Laurent operands.  Oracles:
//! SymPy 1.14 `fps(f, x, 0).truncate(n)`, binomial/exponential series, and
//! the numeric remainder `f(x₀) − truncate(n)(x₀) = O(x₀ⁿ)`.

use symplex::formal_series::FormalPowerSeries;
use symplex::prelude::*;

fn series(ctx: &Context, s: &str) -> FormalPowerSeries {
    let x = ctx.symbol("x");
    ctx.parse(s).unwrap().fps_maclaurin(&x)
}

fn coeffs(s: &FormalPowerSeries, n: usize) -> Vec<String> {
    s.coefficients(n).iter().map(ToString::to_string).collect()
}

/// `f(x₀) − truncate(n)(x₀)` is `O(x₀ⁿ)`: below `bound` at `x₀ = 10⁻³`
/// (`a = 1/3` for a symbolic exponent).
fn remainder_ok(ctx: &Context, f: &str, s: &FormalPowerSeries, n: usize, bound: f64) {
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let x0 = ctx.rational(1, 1000);
    let a0 = ctx.rational(1, 3);
    let at = |e: &Ex| e.subs(&a, &a0).subs(&x, &x0).eval_f64().unwrap();
    let fv = at(&ctx.parse(f).unwrap());
    let t = s.truncate(n);
    let tv = at(&t);
    assert!(
        (fv - tv).abs() < bound,
        "{f}: truncate({n}) = {t} differs by {}",
        (fv - tv).abs()
    );
}

/// `√x·eˣ` was `FormalPowerSeries(nan)`.  SymPy 1.14
/// `fps(sqrt(x)*exp(x)).truncate(4)` = `sqrt(x) + x**(3/2) + x**(5/2)/2 +
/// x**(7/2)/6 + O(x**4)`; the general term `1/k!` of `x^(k + 1/2)`.
#[test]
fn sqrt_times_exp_is_a_puiseux_series() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let s = series(&ctx, "sqrt(x)*exp(x)");
    assert!(!s.is_refused());
    assert_eq!(s.offset().to_string(), "1/2");
    assert_eq!(s.ramification(), 1);
    assert_eq!(coeffs(&s, 4), ["1", "1", "1/2", "1/6"]);
    assert_eq!(s.exponent(3).to_string(), "7/2");
    assert_eq!(s.general_term(&k).unwrap().to_string(), "1/k!");
    assert_eq!(
        s.truncate(4).to_string(),
        "sqrt(x) + x^(3/2) + 1/2*x^(5/2) + 1/6*x^(7/2)"
    );
    remainder_ok(&ctx, "sqrt(x)*exp(x)", &s, 4, 1e-11);
}

/// `x^(1/3)·sin x` was `nan`.  SymPy 1.14 `fps(x**(1/3)*sin(x)).truncate(4)`
/// = `x**(4/3) - x**(10/3)/6 + O(x**4)`.
#[test]
fn cube_root_times_sin() {
    let ctx = Context::new();
    let s = series(&ctx, "x^(1/3)*sin(x)");
    assert_eq!(s.offset().to_string(), "1/3");
    assert_eq!(coeffs(&s, 4), ["0", "1", "0", "-1/6"]);
    assert_eq!(s.truncate(4).to_string(), "x^(4/3) - 1/6*x^(10/3)");
    remainder_ok(&ctx, "x^(1/3)*sin(x)", &s, 4, 1e-11);
}

/// `x^a·eˣ` was `nan`; now `x^a·Σ xᵏ/k!`.  SymPy 1.14
/// `fps(x**a*exp(x)).truncate(4)` = `x**a + x**(a + 1) + x**(a + 2)/2 +
/// x**(a + 3)/6 + O(x**(a + 4))`.  Also `x^a/(1 − x)` and a symbolic power in
/// a denominator, `ln(1 − x)/x^(a+1) = x^(−a−1)·(−x − x²/2 − …)`.
#[test]
fn symbolic_power_times_series() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let s = series(&ctx, "x^a*exp(x)");
    assert_eq!(s.offset().to_string(), "a");
    assert_eq!(s.exponent(2).to_string(), "a + 2");
    assert_eq!(s.general_term(&k).unwrap().to_string(), "1/k!");
    assert_eq!(
        s.truncate(4).to_string(),
        "x^a + x^(a + 1) + 1/2*x^(a + 2) + 1/6*x^(a + 3)"
    );
    remainder_ok(&ctx, "x^a*exp(x)", &s, 4, 1e-11);
    let g = series(&ctx, "x^a/(1 - x)");
    assert_eq!(coeffs(&g, 4), ["1", "1", "1", "1"]);
    let l = series(&ctx, "x^(-1)/x^a*log(1 - x)");
    assert!(!l.is_refused());
    assert_eq!(coeffs(&l, 4), ["0", "-1", "-1/2", "-1/3"]);
    remainder_ok(&ctx, "x^(-1)/x^a*log(1 - x)", &l, 3, 1e-8);
}

/// `exp(√x)` was `nan`; now `Σ x^(k/2)/k!` with ramification 2 and general
/// term `1/k!`.  SymPy 1.14 `fps(exp(sqrt(x))).truncate(2)` = `1 + sqrt(x) +
/// x/2 + x**(3/2)/6 + O(x**2)`.
#[test]
fn exp_of_sqrt_is_ramified() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let s = series(&ctx, "exp(sqrt(x))");
    assert_eq!(s.ramification(), 2);
    assert_eq!(s.offset().to_string(), "0");
    assert_eq!(coeffs(&s, 5), ["1", "1", "1/2", "1/6", "1/24"]);
    assert_eq!(s.exponent(3).to_string(), "3/2");
    assert_eq!(s.general_term(&k).unwrap().to_string(), "1/k!");
    remainder_ok(&ctx, "exp(sqrt(x))", &s, 3, 1e-9);
}

/// `cos √x` and `sin √x` were `nan`.  `cos √x = Σ (−1)ᵏ xᵏ/(2k)!` is a power
/// series in `x` (SymPy 1.14 `fps(cos(sqrt(x))).truncate(4)` = `1 - x/2 +
/// x**2/24 - x**3/720 + O(x**4)`); `sin √x = √x·Σ (−1)ᵏ xᵏ/(2k+1)!`.
#[test]
fn even_and_odd_functions_of_sqrt() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let c = series(&ctx, "cos(sqrt(x))");
    assert_eq!((c.ramification(), c.offset().to_string()), (1, "0".into()));
    assert_eq!(coeffs(&c, 4), ["1", "-1/2", "1/24", "-1/720"]);
    let gt = c.general_term(&k).unwrap();
    for (i, want) in [1.0, -0.5, 1.0 / 24.0, -1.0 / 720.0].iter().enumerate() {
        let v = gt.subs_i64(&k, i as i64).eval_f64().unwrap();
        assert!((v - want).abs() < 1e-15, "cos √x general term at {i}: {gt}");
    }
    let s = series(&ctx, "sin(sqrt(x))");
    assert_eq!(
        (s.ramification(), s.offset().to_string()),
        (1, "1/2".into())
    );
    assert_eq!(coeffs(&s, 3), ["1", "-1/6", "1/120"]);
    remainder_ok(&ctx, "sin(sqrt(x))", &s, 3, 1e-9);
}

/// `√(1 + √x) = Σ C(1/2, k) x^(k/2)` was `nan`.  SymPy 1.14 refuses
/// (`fps(sqrt(1 + sqrt(x)))` returns the input); the binomial series gives
/// `1, 1/2, −1/8, 1/16, −5/128`.
#[test]
fn nested_radical_is_puiseux() {
    let ctx = Context::new();
    let s = series(&ctx, "sqrt(1 + sqrt(x))");
    assert_eq!(s.ramification(), 2);
    assert_eq!(coeffs(&s, 5), ["1", "1/2", "-1/8", "1/16", "-5/128"]);
    remainder_ok(&ctx, "sqrt(1 + sqrt(x))", &s, 3, 1e-8);
}

/// `eˣ/√x` (negative offset), `√(2x)·eˣ` and `√x + x` (a sum: ramification
/// 2) were `nan`.
/// SymPy 1.14 `fps(exp(x)/sqrt(x)).truncate(2)` = `1/sqrt(x) + sqrt(x) +
/// x**(3/2)/2 + O(x**2)`.
#[test]
fn negative_offsets_and_sums() {
    let ctx = Context::new();
    let s = series(&ctx, "exp(x)/sqrt(x)");
    assert_eq!(s.offset().to_string(), "-1/2");
    assert_eq!(
        s.truncate(2).to_string(),
        "1/sqrt(x) + sqrt(x) + 1/2*x^(3/2)"
    );
    // (2x)^(1/2) = √2·x^(1/2) (a positive constant factor; up to 0.47 `nan`)
    let r = series(&ctx, "sqrt(2*x)*exp(x)");
    assert_eq!(r.offset().to_string(), "1/2");
    assert_eq!(coeffs(&r, 2), ["sqrt(2)", "sqrt(2)"]);
    let t = series(&ctx, "sqrt(x) + x");
    assert_eq!(t.truncate(4).to_string(), "x + sqrt(x)");
    // terms with exponents a and a + 1
    let u = series(&ctx, "x^a + x^a*sin(x)");
    assert_eq!(u.offset().to_string(), "a");
    assert_eq!(coeffs(&u, 4), ["1", "1", "0", "-1/6"]);
}

/// Functions with no Laurent or Puiseux expansion gave `nan` coefficients;
/// now they are refused: `is_refused`, `Err` from the fallible accessors,
/// unevaluated stand-ins and a formal `Series` node (SymPy 1.14 `fps` returns
/// the input for each).  `1/x + x^a` is refused too (exponents differing by
/// a symbol cannot share one `x^ρ·Σ`).
#[test]
fn no_expansion_is_refused_never_nan() {
    let ctx = Context::new();
    for f in ["log(x)", "exp(1/x)", "abs(x)", "x^x", "1/x + x^a"] {
        let s = series(&ctx, f);
        assert!(s.is_refused(), "{f}");
        assert!(s.try_coefficient(0).is_err(), "{f}");
        assert!(s.try_truncate(3).is_err(), "{f}");
        let t = s.truncate(3).to_string();
        assert!(t.starts_with("Series("), "{f}: {t}");
        for c in s.coefficients(4) {
            let c = c.to_string();
            assert!(!c.contains("nan") && !c.contains("zoo"), "{f}: {c}");
        }
        let x = ctx.symbol("x");
        assert!(s.add(&x.exp().fps_maclaurin(&x)).is_err(), "{f}");
        assert!(
            s.derivative().is_refused() && s.integral().is_refused(),
            "{f}"
        );
    }
}

/// Arithmetic dropped the negative powers of Laurent operands: up to 0.47
/// `fps(eˣ/x).inverse()` gave `1 − x/2 + x²/12` (the inverse of
/// `(eˣ − 1)/x`), `fps(1/x)·fps(x)` gave `0`, `d/dx fps(1/x)` gave `0`,
/// `∫ fps(1/x² + 1/x + 1)` gave `x`, `exp ∘ fps(1/x)` gave `1` and
/// `reversion(fps(x + 1/x))` gave `x`.  Oracle: the exact functions
/// `x·e⁻ˣ`, `1`, `−1/x²`, `x − 1/x + ln x` (the residue becomes a log term).
#[test]
fn laurent_operands_in_arithmetic() {
    let ctx = Context::new();
    let inv = series(&ctx, "exp(x)/x").inverse().unwrap();
    assert_eq!(inv.truncate(4).to_string(), "1/2*x^3 - x^2 + x");
    let one = series(&ctx, "1/x").mul(&series(&ctx, "x")).unwrap();
    assert_eq!(one.truncate(4).to_string(), "1");
    assert_eq!(
        series(&ctx, "1/x").derivative().truncate(2).to_string(),
        "-x^(-2)"
    );
    let i = series(&ctx, "1/x^2 + 1/x + 1").integral();
    assert_eq!(i.truncate(3).to_string(), "x - 1/x + ln(x)");
    assert!(i.try_truncate(3).is_ok());
    assert!(i.mul(&series(&ctx, "x")).is_err(), "log term in a product");
    let ii = series(&ctx, "1/x").integral().integral();
    assert_eq!(ii.truncate(3).to_string(), "-x + x*ln(x)");
    let e = series(&ctx, "exp(x)");
    assert!(e.compose(&series(&ctx, "1/x")).is_err());
    assert!(series(&ctx, "x + 1/x").reversion().is_err());
}

/// Arithmetic on Puiseux series (all `nan` up to 0.47): offsets add and fold
/// (`√x · √x·eˣ = x·eˣ`), `1/(√x·eˣ) = x^(−1/2)·e⁻ˣ`, term-wise derivative
/// and integral, and `exp ∘ √x = exp(√x)` (up to 0.47 `compose` refused the
/// inner series).
#[test]
fn puiseux_arithmetic() {
    let ctx = Context::new();
    let p = series(&ctx, "sqrt(x)*exp(x)");
    let xe = series(&ctx, "sqrt(x)").mul(&p).unwrap();
    assert_eq!(xe.offset().to_string(), "0");
    assert_eq!(coeffs(&xe, 4), ["0", "1", "1", "1/2"]);
    let inv = p.inverse().unwrap();
    assert_eq!(inv.offset().to_string(), "-1/2");
    assert_eq!(coeffs(&inv, 4), ["1", "-1", "1/2", "-1/6"]);
    // same frame: x^(1/2 + k); the x^(−1/2) term is index −1, like a pole
    let d = p.derivative();
    assert_eq!(coeffs(&d, 3), ["3/2", "5/4", "7/12"]);
    assert_eq!(d.truncate(1).to_string(), "1/(2*sqrt(x)) + 3/2*sqrt(x)");
    assert_eq!(
        p.integral().truncate(3).to_string(),
        "2/3*x^(3/2) + 2/5*x^(5/2)"
    );
    let c = series(&ctx, "exp(x)")
        .compose(&series(&ctx, "sqrt(x)"))
        .unwrap();
    assert_eq!(c.ramification(), 2);
    assert_eq!(coeffs(&c, 5), ["1", "1", "1/2", "1/6", "1/24"]);
    // a sum needing a common ramification: x^(1/2) + x^(1/3) → d = 6
    let s = series(&ctx, "sqrt(x)")
        .add(&series(&ctx, "x^(1/3)"))
        .unwrap();
    assert_eq!(s.ramification(), 6);
    assert_eq!(s.truncate(1).to_string(), "sqrt(x) + cbrt(x)");
}

/// About a non-zero point: `√(x − 1)·eˣ` at 1 was `nan`; now
/// `e·(x − 1)^(1/2) + e·(x − 1)^(3/2) + …` (`eˣ = e·e^(x−1)`).
#[test]
fn puiseux_about_a_nonzero_point() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx
        .parse("sqrt(x - 1)*exp(x)")
        .unwrap()
        .fps(&x, &ctx.int(1));
    assert_eq!(s.offset().to_string(), "1/2");
    assert_eq!(s.coefficient(1).to_string(), "E");
    let t = s.truncate(3);
    let at = ctx.rational(1001, 1000);
    let want = (0.001f64).sqrt() * 1.001f64.exp();
    let got = t.subs(&x, &at).eval_f64().unwrap();
    assert!((got - want).abs() < 1e-10, "{t}");
}
