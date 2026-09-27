//! Integration with parameters over ℂ (decision D4, 0.31): in integration
//! the variable is real, and every other symbol is what its assumptions
//! say — an unassumed parameter may be complex.  So `ln|u|` (and the real
//! forms built on it) is written only where `u` is real for every real `x`
//! under the declared assumptions; otherwise the principal `ln u`, which is
//! an antiderivative for complex parameters too.  Definite integrals refuse
//! a Newton–Leibniz difference across a branch cut of the antiderivative.
//!
//! Each test says what was wrong before; every reference value cites the
//! oracle call that produced it (SymPy 1.14, mpmath 1.3.0).

// Reference values are quoted at the digits the oracle printed them.
#![allow(clippy::excessive_precision)]

use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap_or_else(|e| panic!("{s}: {e}"))
}

/// `F′ = f` (to `1e-9`) at the real sample points, with the parameters
/// bound first.  Binding first matters: `diff` differentiates `ln|u|` with
/// the real-parameter rule (`u′/u`) while its argument has free symbols,
/// which would hide exactly the bug these tests pin.
fn assert_antiderivative_at(
    ctx: &Context,
    f: &Ex,
    big_f: &Ex,
    x: &Ex,
    binding: &[(&str, &str)],
    points: &[&str],
) {
    assert!(
        !big_f.has_unevaluated(),
        "∫ {f} stayed unevaluated: {big_f}"
    );
    let bind = |e: &Ex| {
        binding.iter().fold(e.clone(), |acc, (s, v)| {
            acc.subs(&ctx.symbol(s), &p(ctx, v))
        })
    };
    let residual = bind(big_f).diff(x) - bind(f);
    for pt in points {
        let v = residual
            .subs(x, &p(ctx, pt))
            .eval_complex64()
            .unwrap_or_else(|e| panic!("∫ {f} = {big_f} at {binding:?}: residual at {pt}: {e}"));
        assert!(
            v.norm() < 1e-9,
            "∫ {f} = {big_f} at {binding:?}: F′ − f = {v} at x = {pt}"
        );
    }
}

const COMPLEX_BINDINGS: [[(&str, &str); 2]; 3] = [
    [("a", "7/5 + 3/11*I"), ("b", "2/3 - 5/7*I")],
    [("a", "-6/5 - 2/3*I"), ("b", "-1/5 + 8/7*I")],
    [("a", "-1/2 + 3/2*I"), ("b", "3/4 + 2/5*I")],
];

const POINTS: [&str; 5] = ["1/3", "7/5", "-5/7", "13/4", "-13/4"];

// ═══════════════════════════════════════════════════════════════════════════
// Indefinite integrals
// ═══════════════════════════════════════════════════════════════════════════

/// An unassumed parameter counted as real (`realness.rs`: "a symbol
/// without declared assumptions counts as a real"), so `∫ dx/(a·x + 1)`
/// was `ln|a·x + 1|/a`: its derivative `re(conj(u)·a)/|u|²` is not
/// `1/(a·x + 1)` for a non-real `a` — a silently wrong antiderivative.  The
/// principal logarithm is an antiderivative for every `a` (on a real
/// interval where `a·x + 1 < 0` it differs from `ln|a·x + 1|` by the
/// constant `iπ/a`).  The complex-parameter hunter found 150 such answers
/// in 600 integrands.
///
/// SymPy: `integrate(1/(a*x + 1), x)` → `log(a*x + 1)/a`;
/// `integrate(1/(a + x), x)` → `log(a + x)`; `integrate(x/(a + x**2), x)`
/// → `log(a + x**2)/2`; `integrate((a + 2*x)/(a*x + x**2 + 1), x)` →
/// `log(a*x + x**2 + 1)`; `integrate(exp(a*x)/(exp(a*x) + 1), x)` →
/// `log(exp(a*x) + 1)/a`; `integrate(tan(a*x), x)` →
/// `Piecewise((-log(cos(a*x))/a, Ne(a, 0)), (0, True))`;
/// `integrate(x/(a + b*x), x)` → `-a*log(a + b*x)/b**2 + x/b`.
#[test]
fn unassumed_parameters_give_principal_logarithms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases = [
        (
            "1/(a*x + 1)",
            "Piecewise(ln(a*x + 1)/a if a != 0, x if True)",
        ),
        ("1/(x + a)", "ln(a + x)"),
        ("x/(x^2 + a)", "1/2*ln(x^2 + a)"),
        ("(2*x + a)/(x^2 + a*x + 1)", "ln(a*x + x^2 + 1)"),
        (
            "exp(a*x)/(exp(a*x) + 1)",
            "Piecewise(ln(exp(a*x) + 1)/a if a != 0, 1/2*x if True)",
        ),
        (
            "tan(a*x)",
            "Piecewise(-ln(cos(a*x))/a if a != 0, 0 if True)",
        ),
        (
            "x/(a + b*x)",
            "Piecewise((-a*ln(b*x + a)/b + x)/b if b != 0, x^2/(2*a) if True)",
        ),
    ];
    for (f, shown) in cases {
        let f = p(&ctx, f);
        let big_f = f.integrate(&x);
        assert_eq!(big_f.to_string(), shown, "∫ {f}");
        for binding in &COMPLEX_BINDINGS {
            assert_antiderivative_at(&ctx, &f, &big_f, &x, binding, &POINTS);
        }
    }
}

/// Users who want `ln|a·x + 1|` declare `a` real: then `a·x + 1` is real for
/// every real `x`, and the real-interval antiderivative (the crate's
/// value-add over SymPy's `log(a*x + 1)/a`, which is off by `iπ/a` where
/// `a·x + 1 < 0`) is kept.  Checked by differentiation at real values of
/// both signs, at points on both sides of the pole.
///
/// SymPy (with `a = Symbol('a', real=True)`): `integrate(1/(a*x + 1), x)`
/// → `log(a*x + 1)/a`; `integrate(tan(a*x), x)` →
/// `Piecewise((-log(cos(a*x))/a, Ne(a, 0)), (0, True))`.
#[test]
fn declared_real_parameters_keep_ln_abs() {
    let ctx = Context::new();
    ctx.symbol_with("a", &[Assumption::Real]).unwrap();
    let x = ctx.symbol("x");
    let cases = [
        (
            "1/(a*x + 1)",
            "Piecewise(ln(abs(a*x + 1))/a if a != 0, x if True)",
        ),
        ("1/(x + a)", "ln(abs(a + x))"),
        (
            "tan(a*x)",
            "Piecewise(-ln(abs(cos(a*x)))/a if a != 0, 0 if True)",
        ),
    ];
    for (f, shown) in cases {
        let f = p(&ctx, f);
        let big_f = f.integrate(&x);
        assert_eq!(big_f.to_string(), shown, "∫ {f}");
        for a in ["6/5", "-6/5", "5/3"] {
            assert_antiderivative_at(&ctx, &f, &big_f, &x, &[("a", a)], &POINTS);
        }
    }
}

/// Numeric coefficients keep `ln|x − 2|` (the real-interval value-add over
/// SymPy's `log(x - 2)`), and so does an `ln|u|` whose argument has no
/// parameter even when a parameter multiplies it: `cos x` is real.
///
/// SymPy: `integrate(1/(x - 2), x)` → `log(x - 2)`;
/// `integrate(a*tan(x), x)` → `-a*log(cos(x))`.
#[test]
fn parameter_free_arguments_keep_ln_abs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases = [
        ("1/(x - 2)", "ln(abs(x - 2))"),
        ("a*tan(x)", "-a*ln(abs(cos(x)))"),
        ("a/(x - 2) + 1", "a*ln(abs(x - 2)) + x"),
    ];
    for (f, shown) in cases {
        let f = p(&ctx, f);
        let big_f = f.integrate(&x);
        assert_eq!(big_f.to_string(), shown, "∫ {f}");
        for binding in &COMPLEX_BINDINGS {
            assert_antiderivative_at(&ctx, &f, &big_f, &x, binding, &["1/3", "7/5", "13/4"]);
        }
    }
}

/// `√((x + a)²)` was rewritten to `|x + a|` inside the integrator (the
/// nested-power flattening read an unassumed `a` as real), and the
/// unevaluated answer named a different integral: at `a = i`, `x = 0` the
/// integrand is `√(−1) = i`, not `|i| = 1`.  For a declared-real `a` the
/// rewrite is an identity and stays.
///
/// SymPy: `sqrt((x + a)**2)` stays as it is; `sqrt((0 + I)**2)` → `I`;
/// `Abs(I)` → 1.
#[test]
fn square_root_of_a_square_with_a_complex_parameter_is_not_an_absolute_value() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(&ctx, "sqrt((x + a)^2)");
    assert_eq!(f.integrate(&x).to_string(), "Integral(sqrt((a + x)^2), x)");
    let ctx = Context::new();
    ctx.symbol_with("a", &[Assumption::Real]).unwrap();
    let x = ctx.symbol("x");
    let f = p(&ctx, "sqrt((x + a)^2)");
    assert_eq!(f.integrate(&x).to_string(), "Integral(abs(a + x), x)");
}

/// The degenerate-parameter wrapper solved a denominator of a partial
/// answer, `(a + 1)·x + a`, for `a` and substituted the "value"
/// `a = −x/(x + 1)`, which depends on the variable; the resulting
/// integrand had a zero denominator in disguise and panicked the rational
/// integrator ("division by zero polynomial").  Elsewhere the same
/// "value" became a condition on the variable: `∫ (a + 2 − a·x)⁻² +
/// sin(a·x)·cos x dx` was a `Piecewise` over `a ≠ 2/(x − 1)` with
/// unevaluated integrals in its other branch.  Found by the
/// complex-parameter hunter.
///
/// SymPy: `integrate((2*x + a)/(x**2 + a*x - 2), x)` →
/// `log(a*x + x**2 - 2)`; the second checked by differentiation at
/// complex `a`.
///
/// Updated in 0.31 (the rational integrator over the parameters' field):
/// the second term, pinned unevaluated here before, now integrates —
/// SymPy: `integrate(1/(x*((a + 1)*x + a)), x)` →
/// `(log(x) - log(a/(a + 1) + x))/a`, with the degenerate cases `a = 0`,
/// `a = −1` as branches; checked by differentiation at complex `a`.
#[test]
fn degenerate_parameter_values_are_constants() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(&ctx, "(2*x + a)/(x^2 + a*x - 2) + 1/(x*((a + 1)*x + a))");
    let big_f = f.integrate(&x);
    assert_eq!(
        big_f.to_string(),
        "Piecewise(Piecewise(-ln(a/(a + 1) + x)/a + ln(abs(x))/a + ln(a*x + x^2 - 2) if a != 0, \
         -1/x + ln(abs(x^2 - 2)) if True) if a != -1, -ln(abs(x)) + ln(abs(x^2 - x - 2)) if True)"
    );
    for binding in &COMPLEX_BINDINGS {
        assert_antiderivative_at(&ctx, &f, &big_f, &x, binding, &POINTS);
    }
    let f = p(&ctx, "(a + 2 - a*x)^(-2) + sin(a*x)*cos(x)");
    let big_f = f.integrate(&x);
    assert!(!big_f.to_string().contains("(x - 1)"), "{big_f}");
    for binding in &COMPLEX_BINDINGS {
        assert_antiderivative_at(&ctx, &f, &big_f, &x, binding, &POINTS);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Definite integrals
// ═══════════════════════════════════════════════════════════════════════════

/// `∫₀¹ dx/(x² + 2a·x + a² + 1)` was `atan(a + 1) − atan(a)` for every `a`.
/// For `a = −1/2 + 3i/2` the path `x + a` crosses the cut of `atan` at
/// `x = 1/2` and that is 2.5536, not the integral's −0.5880: the
/// antiderivative jumps by `π` where the integrand is smooth.  A line with
/// a real slope is now written in logarithms whose imaginary parts do not
/// depend on `x` (SymPy's form), exact for every complex `a`; a declared-
/// real `a` keeps the arctangents.
///
/// SymPy: `integrate(1/(x**2 + 2*a*x + a**2 + 1), (x, 0, 1))` →
/// `I*log(a - I)/2 - I*log(a + I)/2 - I*log(a + 1 - I)/2 +
/// I*log(a + 1 + I)/2`.  mpmath (`mp.dps = 30`, 40 agrees):
/// `quad(lambda t: 1/(t**2 + 2*a*t + a**2 + 1), [0, 0.25, 0.5, 0.75, 1])`
/// at `a = mpc(-1/2, 3/2)` → `-0.588002603547567551245611080625`, at
/// `a = mpc(7/5, 3/11)` → `0.217547831573327492628767158175 -
/// 0.0508761242001240039558645667798j`; `atan(a + 1) - atan(a)` at
/// `a = mpc(-1/2, 3/2)` → `2.55359005004222568721703230265`.
///
/// Updated in 0.31 (second pass): for an unassumed `a` the integral is now
/// refused — the poles `x = −a ± i` are real where `Im a = ±1` (at
/// `a = −1/2 + i` one is `x = 1/2`, on the path), and the singularity scan
/// no longer drops them as non-real.  The same `a` values are checked
/// through `a = c + (3/2)i`, `c + (3/11)i` with `c` declared real.
#[test]
fn definite_arctangent_of_a_complex_line_becomes_logarithms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(&ctx, "1/(x^2 + 2*a*x + a^2 + 1)");
    let v = f.integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    assert!(v.has_unevaluated(), "{v}");
    let ctx = Context::new();
    let c = ctx.symbol_with("c", &[Assumption::Real]).unwrap();
    let x = ctx.symbol("x");
    for (im_a, value, re, im) in [
        ("3/2", "-1/2", -0.588002603547567551245611080625, 0.0),
        (
            "3/11",
            "7/5",
            0.217547831573327492628767158175,
            -0.0508761242001240039558645667798,
        ),
    ] {
        let a = format!("(c + {im_a}*I)");
        let f = p(&ctx, &format!("1/(x^2 + 2*{a}*x + {a}^2 + 1)"));
        let v = f.integrate_definite(&x, &ctx.int(0), &ctx.int(1));
        assert!(!v.has_unevaluated(), "{v}");
        let z = v.subs(&c, &p(&ctx, value)).eval_complex64().unwrap();
        assert!(
            (z.re - re).abs() < 1e-12 && (z.im - im).abs() < 1e-12,
            "{v} at a = {value}: {z}"
        );
    }
    let ctx = Context::new();
    ctx.symbol_with("a", &[Assumption::Real]).unwrap();
    let x = ctx.symbol("x");
    let f = p(&ctx, "1/(x^2 + 2*a*x + a^2 + 1)");
    let v = f.integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    assert_eq!(v.to_string(), "-atan(a) + atan(a + 1)");
}

/// `∫₀¹ atan(x + a) dx` was `(a + 1)·atan(a + 1) − a·atan(a) −
/// ln((a + 1)² + 1)/2 + ln(a² + 1)/2` for every `a`; at `a = −1/2 + 3i/2`
/// that is `2.3128·i`, the integral `0.7420·i` (the integrand itself jumps
/// at `x = 1/2`, where `x + a` crosses the cut of `atan`).  The arguments
/// of `atan` and `ln` in the antiderivative may cross their cuts for
/// complex `a`, so the integral is refused; for a declared-real `a` the
/// difference stands.
///
/// mpmath (`mp.dps = 30`, 45 agrees): `quad(lambda t: atan(t + a), [0,
/// 1/2, 1])` at `a = mpc(-1/2, 3/2)` → `0.742027157291361925130665850468j`;
/// the old closed form there → `2.31282348408625854436198754211j`.
#[test]
fn definite_integral_refused_where_a_cut_may_be_crossed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(&ctx, "atan(x + a)");
    let v = f.integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    assert_eq!(v.to_string(), "Integral(atan(a + x), x, 0, 1)");
    let ctx = Context::new();
    ctx.symbol_with("a", &[Assumption::Real]).unwrap();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let f = p(&ctx, "atan(x + a)");
    let v = f.integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    assert!(!v.has_unevaluated(), "{v}");
    // SymPy: integrate(atan(x - 1/2), (x, 0, 1)) → 0 (odd about x = 1/2).
    let z = v.subs(&a, &p(&ctx, "-1/2")).eval_complex64().unwrap();
    assert!(z.norm() < 1e-12, "{v} at a = -1/2: {z}");
}

/// A logarithm whose argument has an imaginary part independent of `x`
/// never crosses its cut along the real path: `∫₀¹ dx/(x + c − i)` is the
/// Newton–Leibniz difference.
///
/// Updated in 0.31 (second pass): this was asserted for an unassumed `a` in
/// `∫₀¹ dx/(x + a + i)`, but the pole `x = −a − i` is real where
/// `Im a = −1` (on the path for `a = −1/2 − i`), so for a possibly complex
/// `a` the integral is refused now; with `c` declared real the pole is never
/// real and the difference stands.
///
/// SymPy: `integrate(1/(x + a + I), (x, 0, 1))` →
/// `-log(a + I) + log(a + 1 + I)`.  mpmath (`mp.dps = 30`, 40 agrees):
/// `quad(lambda t: 1/(t + a + 1j), [0, 1])` at `a = mpc(-3/2, -2)` (that is
/// `t − 3/2 − i`, `c = −3/2` below) →
/// `-0.47775572251371818072636405417 + 0.519146114246522951771454379553j`.
#[test]
fn definite_logarithm_with_constant_imaginary_part_is_kept() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(&ctx, "1/(x + a + I)");
    let v = f.integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    assert!(v.has_unevaluated(), "{v}");
    let ctx = Context::new();
    let a = ctx.symbol_with("c", &[Assumption::Real]).unwrap();
    let x = ctx.symbol("x");
    let f = p(&ctx, "1/(x + c - I)");
    let v = f.integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    assert_eq!(v.to_string(), "-ln(c - I) + ln(c - I + 1)");
    let z = v.subs(&a, &p(&ctx, "-3/2")).eval_complex64().unwrap();
    assert!(
        (z.re + 0.47775572251371818072636405417).abs() < 1e-12
            && (z.im - 0.519146114246522951771454379553).abs() < 1e-12,
        "{z}"
    );
}

/// `d ln|g|/dx = g′/g` only for a `g` real on the real line under the
/// declared assumptions: up to 0.30 `diff` counted an undeclared parameter
/// as real, so `d ln|a·x + 1|/dx` was `a/(a·x + 1)`, `(1 + i)/2` at
/// `a = i`, `x = 1`.  A declared-real parameter keeps the short form.
///
/// SymPy (x real, a unassumed):
/// `N(diff(log(Abs(a*x+1)), x).subs({a: I, x: 1}))` → `0.5`;
/// `diff(log(Abs(ar*x+1)), x)` for a real `ar` → `ar/(ar*x + 1)`.
#[test]
fn derivative_of_ln_abs_with_a_complex_parameter() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let d = (&(&a * &x) + 1).abs().ln().diff(&x);
    let at = d.subs(&a, &ctx.i_unit()).subs(&x, &ctx.int(1));
    let v = at.eval_complex64().unwrap();
    assert!(
        (v.re - 0.5).abs() < 1e-14 && v.im.abs() < 1e-14,
        "{d} at a = i, x = 1: {v}"
    );

    let ar = ctx.symbol_with("ar", &[Assumption::Real]).unwrap();
    let g = &(&ar * &x) + 1;
    assert_eq!(g.abs().ln().diff(&x), &ar / &g);
}
