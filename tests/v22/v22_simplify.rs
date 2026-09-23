//! 0.23 — every rewrite preserves the value over ℂ.
//!
//! A symbol without assumptions may be complex, and every function takes
//! its principal branch.  `fuzz_simplify` (comparing complex values at real
//! and complex points since 0.23) found rewrites that were right only on the
//! reals, or only for positive arguments; each is pinned here with the point
//! where the old rewrite was wrong.  Reference values are mpmath 1.3 at
//! `mp.dps = 30` (`symplex/.venv`) and SymPy 1.14, quoted with the call.

// Reference values are quoted at the 17 digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use symplex::prelude::*;

/// `e` at `x = v` as a complex number (16 digits).
fn at(e: &Ex, x: &Ex, v: &Ex) -> Complex64 {
    e.subs(x, v)
        .eval_complex64()
        .unwrap_or_else(|err| panic!("{e} at {v}: {err}"))
}

/// `a` and `b` have the same value at `x = v`.
fn same_value(a: &Ex, b: &Ex, x: &Ex, v: &Ex) {
    let (za, zb) = (at(a, x, v), at(b, x, v));
    assert!(
        (za - zb).norm() <= 1e-12 * za.norm().max(1.0),
        "{a} = {za} but {b} = {zb} at x = {v}"
    );
}

fn close(z: Complex64, re: f64, im: f64, label: &str) {
    let want = Complex64::new(re, im);
    assert!(
        (z - want).norm() <= 1e-13 * want.norm().max(1.0),
        "{label}: got {z}, expected {want}"
    );
}

fn complex(ctx: &Context, re: (i64, i64), im: (i64, i64)) -> Ex {
    ctx.rational(re.0, re.1) + ctx.i_unit() * ctx.rational(im.0, im.1)
}

// ── logarithms: only what holds for every complex value ────────────────

#[test]
fn simplify_keeps_a_sum_of_logs_of_unknown_sign() {
    // ln(−1) + ln(−1) = 2πi but ln((−1)·(−1)) = 0.  SymPy:
    // simplify(log(x) + log(y)) = log(x) + log(y).
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f = &x.ln() + &y.ln();
    assert_eq!(f.simplify(), f);
    let (p, q) = (
        ctx.symbol_with("p", &[Assumption::Positive]),
        ctx.symbol_with("q", &[Assumption::Positive]),
    );
    // SymPy: simplify(log(p) + log(q)) = log(p*q)
    assert_eq!(format!("{}", (&p.ln() + &q.ln()).simplify()), "ln(p*q)");
}

#[test]
fn simplify_keeps_the_branch_of_a_negated_log() {
    // fuzz_simplify: (−ln x + sin|x|)/x became (sin|x| + ln(1/x))/x, which
    // differs by 2πi/x at x = −5/7 (−ln(−a) = −ln a − iπ, ln(−1/a) = −ln a + iπ).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (-x.ln() + x.abs().sin()) / &x;
    same_value(&f, &f.simplify(), &x, &ctx.rational(-5, 7));
    let g = &x.ln() + &x.asinh().tanh().ln();
    same_value(&g, &g.simplify(), &x, &ctx.rational(-5, 7));
}

#[test]
fn log_combine_joins_a_positive_log_with_any_other() {
    // arg 2 = 0, so ln 2 + ln x = ln(2x) for every complex x.
    // SymPy: logcombine(log(2) + log(x)) = log(2*x).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.int(2).ln() + x.ln();
    let g = f.log_combine();
    assert_eq!(format!("{g}"), "ln(2*x)");
    for v in [ctx.rational(-5, 7), complex(&ctx, (-3, 4), (5, 4))] {
        same_value(&f, &g, &x, &v);
    }
}

#[test]
fn expand_log_leaves_powers_outside_the_principal_range() {
    // SymPy: expand_log(log(x**2)) = log(x**2), expand_log(log(1/x)) = log(1/x);
    // ln((−2)²) = ln 4 but 2·ln(−2) = ln 4 + 2πi.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for f in [x.powi(2).ln(), x.powi(-1).ln()] {
        assert_eq!(f.expand_log(), f);
    }
    // SymPy: expand_log(log(sqrt(x))) = log(x)/2
    assert_eq!(format!("{}", x.sqrt().ln().expand_log()), "1/2*ln(x)");
    // The forced form is still there, and exact for positive x.
    let forced = x.powi(2).ln().expand_log_with(true);
    assert_eq!(format!("{forced}"), "2*ln(x)");
    same_value(&x.powi(2).ln(), &forced, &x, &ctx.rational(7, 5));
}

// ── identities of real analysis need a known-real argument ─────────────

#[test]
fn ln_of_exp_stays_for_a_complex_argument() {
    // ln(e^z) = z only for Im z ∈ (−π, π]: ln(e^(1/3 + 4i)) = 1/3 + (4 − 2π)i.
    // mpmath: log(exp(mpc(1/3, 4))) = 0.33333333333333333 - 2.2831853071795865j
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp().ln();
    assert_eq!(f.simplify(), f);
    let v = complex(&ctx, (1, 3), (4, 1));
    close(
        at(&f, &x, &v),
        1.0 / 3.0,
        -2.2831853071795865,
        "ln(exp(1/3+4i))",
    );
    let r = ctx.symbol_with("r", &[Assumption::Real]);
    assert_eq!(r.exp().ln().simplify(), r);
}

#[test]
fn sqrt_of_a_square_stays_for_a_complex_argument() {
    // √(z²) = z for Re z > 0, not |z|: mpmath sqrt(mpc(0.5, 1.5)**2) = 0.5 + 1.5j.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2).sqrt();
    assert_eq!(f.simplify(), f);
    let v = complex(&ctx, (1, 2), (3, 2));
    close(at(&f, &x, &v), 0.5, 1.5, "sqrt((1/2+3i/2)^2)");
    for g in [&f / &x, &x * &f + &x, (&f - &x) / &x] {
        same_value(&g, &g.simplify(), &x, &v);
    }
    let r = ctx.symbol_with("r", &[Assumption::Real]);
    assert_eq!(format!("{}", r.powi(2).sqrt().simplify()), "abs(r)");
}

#[test]
fn inverse_hyperbolic_of_hyperbolic_stays_for_a_complex_argument() {
    // mpmath: asinh(sinh(1+2j)) = -1.0 + 1.1415926535897932j,
    //         atanh(tanh(1+2j)) =  1.0 - 1.1415926535897932j,
    //         acosh(cosh(1+2j)) =  1.0 + 2.0j   (not |1 + 2i| = √5).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let v = ctx.int(1) + ctx.i_unit() * 2;
    let cases: [(Ex, f64, f64); 3] = [
        (x.sinh().asinh(), -1.0, 1.1415926535897932),
        (x.tanh().atanh(), 1.0, -1.1415926535897932),
        (x.cosh().acosh(), 1.0, 2.0),
    ];
    for (f, re, im) in cases {
        assert_eq!(f.simplify(), f, "{f} must stay for complex x");
        close(at(&f, &x, &v), re, im, &f.to_string());
    }
    // For real arguments they simplify — further than SymPy 1.14, which
    // leaves asinh(sinh(r)), atanh(tanh(r)) and acosh(cosh(r)) as they are.
    let r = ctx.symbol_with("r", &[Assumption::Real]);
    assert_eq!(r.sinh().asinh().simplify(), r);
    assert_eq!(r.tanh().atanh().simplify(), r);
    assert_eq!(r.cosh().acosh().simplify(), r.abs());
}

#[test]
fn a_power_of_exp_merges_only_on_the_principal_branch() {
    // mpmath: sqrt(exp(4j)) = 0.41614683654714239 - 0.9092974268256817j,
    // while exp(2j) = -0.41614683654714239 + 0.9092974268256817j.
    // SymPy: sqrt(exp(x)) stays, exp(r)**y = exp(r*y), exp(x)**2 = exp(2*x).
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f = x.exp().sqrt();
    assert_eq!(f.eval(), f);
    close(
        at(&f, &x, &(ctx.i_unit() * 4)),
        0.41614683654714239,
        -0.9092974268256817,
        "sqrt(exp(4i))",
    );
    assert_eq!(x.exp().powi(2).eval(), (&x * 2).exp());
    let r = ctx.symbol_with("r", &[Assumption::Real]);
    assert_eq!(r.exp().pow(&y).eval(), (&r * &y).exp());
    // fuzz_simplify: sqrt(exp(x)·exp(cosh x)) at 1/3 + 4i.
    let g = (x.exp() * x.cosh().exp()).sqrt();
    same_value(&g, &g.simplify(), &x, &complex(&ctx, (1, 3), (4, 1)));
}

#[test]
fn limits_still_merge_powers_of_exp() {
    // The limit variable runs along the positive reals (a positive dummy
    // inside Gruntz, as in SymPy), so exp(x)^(1/x) → e.  SymPy: E.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp().pow(&(1 / &x));
    assert_eq!(f.limit(&x, &ctx.infinity()), ctx.e());
    // SymPy: limit(sqrt(x**2)/x, x, oo) = 1
    assert_eq!(
        (x.powi(2).sqrt() / &x).limit(&x, &ctx.infinity()),
        ctx.int(1)
    );
}

// ── one branch for rational powers ─────────────────────────────────────

#[test]
fn cube_root_of_a_negative_number_is_principal() {
    // SymPy: cbrt(-8) = 2*(-1)**(1/3); N: 1.0 + 1.73205080756888*I.
    let ctx = Context::new();
    let c = ctx.int(-8).cbrt().eval();
    // `cbrt(…)` is how `…^(1/3)` prints: this is 2·(−1)^(1/3).
    assert_eq!(format!("{c}"), "2*cbrt(-1)");
    assert_eq!(c, ctx.int(2) * ctx.int(-1).pow(&ctx.rational(1, 3)));
    close(
        c.eval_complex64().unwrap(),
        1.0,
        1.7320508075688773,
        "cbrt(-8)",
    );
}

#[test]
fn simplify_cbrt_squared_agrees_everywhere() {
    // simplify(cbrt(x)²) = x^(2/3) (SymPy's answer), which is right at
    // x = −2 only if (−2)^(1/3) is the principal root there too.
    // mpmath: (mpc(-2)**(1/3))**2 = (-2)**(2/3) = -0.79370052598409974 + 1.3747296369986026j
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.cbrt().powi(2);
    let g = f.simplify();
    assert_eq!(format!("{g}"), "x^(2/3)");
    for v in [
        ctx.int(-2),
        ctx.rational(-5, 7),
        complex(&ctx, (-3, 4), (5, 4)),
    ] {
        same_value(&f, &g, &x, &v);
    }
    close(
        at(&g, &x, &ctx.int(-2)),
        -0.79370052598409974,
        1.3747296369986026,
        "(-2)^(2/3)",
    );
}

#[test]
fn exp_of_a_log_of_a_negative_is_the_principal_power() {
    // fuzz_simplify: exp(x·ln(−3x)) → (−3x)^x, which at x = 1/3 folded to the
    // real root (−1)^(1/3) = −1; both are exp(iπ/3) = 1/2 + (√3/2)i.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x * (&x * -3).ln()).exp();
    let v = ctx.rational(1, 3);
    same_value(&f, &f.simplify(), &x, &v);
    close(
        at(&f.simplify(), &x, &v),
        0.5,
        0.8660254037844386,
        "(-1)^(1/3)",
    );
}

#[test]
fn real_root_gives_the_real_odd_root() {
    // SymPy: real_root(-8, 3) = -2, real_root(-32, 5) = -2, real_root(-4, 2) = 2*I,
    //        N(real_root(-2, 3)) = -1.25992104989487
    let ctx = Context::new();
    assert_eq!(ctx.int(-8).real_root(3).unwrap(), ctx.int(-2));
    assert_eq!(ctx.int(-32).real_root(5).unwrap(), ctx.int(-2));
    assert_eq!(
        ctx.int(-4).real_root(2).unwrap().eval(),
        ctx.int(2) * ctx.i_unit()
    );
    let v = ctx.int(-2).real_root(3).unwrap().eval_f64().unwrap();
    assert!((v + 2f64.cbrt()).abs() < 1e-15, "{v}");
    assert!(ctx.int(8).real_root(0).is_err());
    // For a symbol not known real it is SymPy's Piecewise on im(x) = 0.
    let x = ctx.symbol("x");
    let pw = x.real_root(3).unwrap();
    assert!(format!("{pw}").starts_with("Piecewise"), "{pw}");
    close(
        at(&pw, &x, &ctx.int(-27)),
        -3.0,
        0.0,
        "real_root(x, 3) at -27",
    );
}

// ── refine ─────────────────────────────────────────────────────────────

#[test]
fn refine_rewrites_only_the_square_root_it_matched() {
    // fuzz_simplify: cos(√(|x|²)) simplified to |x| — refine's √(r²) → |r|
    // replaced the whole expression.  mpmath: cos(1/3) = 0.94495694631473766.
    // SymPy: simplify(cos(sqrt(Abs(x)**2))) = cos(Abs(x)),
    //        refine(sqrt(r**2) + 1) = Abs(r) + 1.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.abs().powi(2).sqrt().cos();
    let g = f.simplify();
    assert_eq!(g, x.abs().cos());
    close(
        at(&g, &x, &ctx.rational(1, 3)),
        0.94495694631473766,
        0.0,
        "cos(1/3)",
    );
    let r = ctx.symbol_with("r", &[Assumption::Real]);
    assert_eq!((r.powi(2).sqrt() + 1).refine(), r.abs() + 1);
}
