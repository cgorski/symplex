//! Integration with symbolic coefficients (0.31, second pass): the
//! rational integrator over the field of the parameters, `ℚ(p₁, …)(x)`;
//! the `together_deep` panic on a denominator that vanishes identically;
//! poles whose realness depends on a complex parameter in definite
//! integrals.
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
/// bound first (as in `v30_integrate.rs`).
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

const BINDINGS: [[(&str, &str); 2]; 5] = [
    [("a", "7/5 + 3/11*I"), ("b", "2/3 - 5/7*I")],
    [("a", "-6/5 - 2/3*I"), ("b", "-1/5 + 8/7*I")],
    [("a", "-1/2 + 3/2*I"), ("b", "3/4 + 2/5*I")],
    [("a", "6/5"), ("b", "3/4")],
    [("a", "-6/5"), ("b", "-3/4")],
];
const POINTS: [&str; 5] = ["1/3", "7/5", "-5/7", "13/4", "-13/4"];

// ═══════════════════════════════════════════════════════════════════════════
// The together_deep panic
// ═══════════════════════════════════════════════════════════════════════════

/// `∫ 1/(x·(−x/(x + 1) + x·(−x/(x + 1) + 1))) + … dx` panicked with
/// "division by zero polynomial" (`generic.rs`, via `together_deep` from
/// the rational integrator), and so did `together`, `as_numer_denom`: the
/// inner sum cancels to 0, the reciprocal's denominator is the literal 0,
/// and the polynomial LCM of the denominators divided by it.  Also a
/// denominator that is zero only as a polynomial (`x(x + 1) − x² − x`),
/// and `cancel` of `0/0` in that form (`gcd(0, 0) = 0` was the divisor).
/// The expression is undefined everywhere: `together` gives `zoo`, the
/// integral stays unevaluated.
///
/// SymPy: `cancel(e)` → `zoo`, `ratsimp(e)` → `zoo` (and
/// `integrate(e, x)` → `zoo*x`, `apart(e, x)` raises ZeroDivisionError).
#[test]
fn identically_vanishing_denominators_do_not_panic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = p(
        &ctx,
        "1/(x*(-x/(x + 1) + x*(-x/(x + 1) + 1))) + (-x/(x + 1) + 2*x)/(-x^2/(x + 1) + x^2 - 2)",
    );
    assert_eq!(e.together().to_string(), "zoo");
    let (_, d) = e.as_numer_denom();
    assert_eq!(d.to_string(), "0");
    assert!(e.integrate(&x).has_unevaluated());
    let _ = (
        e.cancel(&x),
        e.ratsimp(),
        e.partial_fractions(&x),
        e.simplify(),
    );
    for s in [
        "1/(x*(x + 1) - x^2 - x) + 1/x",
        "1/(x + 1) + 1/(x*(x + 1) - x^2 - x)",
        "(x^2 + x - x*(x + 1))/(x*(x + 1) - x^2 - x)",
    ] {
        let e = p(&ctx, s);
        let _ = (e.together(), e.as_numer_denom(), e.cancel(&x), e.ratsimp());
        let _ = (e.partial_fractions(&x), e.integrate(&x));
    }
}

/// A looped regression over nested rational expressions of the panic's
/// kind (seeded, deterministic): sums, products and reciprocals of
/// `x + c`, `x·(…)`, and sums that cancel to zero, with a parameter.  Every
/// rational-function entry point must return (the panic above was found by
/// the complex-parameter hunter in 1,500 cases; the parametric-rational
/// hunter's `nest` mode ran 600 more).
#[test]
fn nested_rational_expressions_never_panic() {
    fn next(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn leaf(s: &mut u64) -> String {
        const C: [&str; 6] = ["1", "2", "-1", "a", "(a + 1)", "(-a)"];
        let c = C[(next(s) % 6) as usize];
        match next(s) % 3 {
            0 => "x".to_string(),
            1 => c.to_string(),
            _ => format!("(x + {c})"),
        }
    }
    fn nested(s: &mut u64, depth: u32) -> String {
        if depth == 0 || next(s).is_multiple_of(4) {
            return leaf(s);
        }
        match next(s) % 5 {
            0 => format!("({} + {})", nested(s, depth - 1), nested(s, depth - 1)),
            1 => format!("({}*{})", nested(s, depth - 1), nested(s, depth - 1)),
            2 => format!("1/({})", nested(s, depth - 1)),
            3 => format!("({})/({})", nested(s, depth - 1), nested(s, depth - 1)),
            _ => {
                let c = leaf(s);
                let z = format!("(-x/(x + {c}) + x*(-x/(x + {c}) + 1))");
                format!("(1/(x*{z}) + {})", nested(s, depth - 1))
            }
        }
    }
    let mut seed = 20_260_927u64;
    for _ in 0..40 {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let src = nested(&mut seed, 3);
        let e = p(&ctx, &src);
        let _ = (e.together(), e.as_numer_denom(), e.cancel(&x), e.ratsimp());
        let _ = (e.partial_fractions(&x), e.integrate(&x));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational functions over ℚ(parameters)
// ═══════════════════════════════════════════════════════════════════════════

/// These stayed unevaluated: the rational integrator worked over `ℚ` only,
/// and the parametric routes knew `1/(a·x + b)`, `1/(x² + k)` and a few
/// quadratic shapes.  Now Hermite reduction and the logarithmic part run
/// over `ℚ(a, b)`; each answer is checked by differentiation at complex and
/// real parameter values.
///
/// SymPy (unassumed `a`, `b`):
/// `integrate(1/(x**2 + a*x + b), x)` → `sqrt(1/(a**2 - 4*b))*log(-a**2*sqrt(1/(a**2 - 4*b))/2 + a/2 + 2*b*sqrt(1/(a**2 - 4*b)) + x) - …`;
/// `integrate(1/((x + a)*(x + b)), x)` → `-log(a + x)/(a - b) + log(b + x)/(a - b)` (printed unsimplified);
/// `integrate(1/(x*(a + b*x)), x)` → `(log(x) - log(a/b + x))/a`;
/// `integrate(x/((x + a)*(x + b)), x)` → `a*log(x + …)/(a - b) - b*log(x + …)/(a - b)`;
/// `integrate(1/(a**2 - x**2), x)` → `(-log(-a + x)/2 + log(a + x)/2)/a`;
/// `integrate(1/(a*x**2 + 1), x)` → `-sqrt(-1/a)*log(x - sqrt(-1/a))/2 + sqrt(-1/a)*log(x + sqrt(-1/a))/2`;
/// `integrate(1/(x + a)**2/(x + b), x)` → `1/(a**2 - a*b + x*(a - b)) + log(…)/(a - b)**2 - log(…)/(a - b)**2`;
/// `integrate((a*x**3 + b)/(x**4 + 1), x)` → closed form with `log(x**2 ± sqrt(2)*x + 1)`, `atan`;
/// `integrate(x/(x**4 + a), x)` → `-sqrt(-1/a)*log(-a*sqrt(-1/a) + x**2)/4 + sqrt(-1/a)*log(a*sqrt(-1/a) + x**2)/4`;
/// `integrate((x + a)/(x**2 + b)**2, x)` → `a*(-sqrt(-1/b**3)*log(…)/4 + …) + (a*x - b)/(2*b**2 + 2*b*x**2)`;
/// `integrate(1/((x + a)*(x + b)*(x + c)), x)` timed out after 10 s.
#[test]
fn rational_functions_with_parameters_integrate() {
    for src in [
        "1/(x^2 + a*x + b)",
        "1/((x + a)*(x + b))",
        "1/(x*(a + b*x))",
        "x/((x + a)*(x + b))",
        "1/(a^2 - x^2)",
        "1/(a*x^2 + 1)",
        "1/((x + a)^2*(x + b))",
        "(a*x^3 + b)/(x^4 + 1)",
        "x/(x^4 + a)",
        "(x + a)/(x^2 + b)^2",
        "1/((x + a)*(x + b)*(x + 1))",
        "(a*x^2 + b)/((x - a)^2*(x^2 + b))",
    ] {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let f = p(&ctx, src);
        let big_f = f.integrate(&x);
        for binding in &BINDINGS {
            assert_antiderivative_at(&ctx, &f, &big_f, &x, binding, &POINTS);
        }
    }
}

/// The exact forms: principal logarithms with monic arguments `x − r` for
/// unassumed parameters (decision D4; SymPy's `ratint` forms), the
/// degenerate cases as `Piecewise` branches.  `∫ dx/(x² − a²)` was
/// `atan(x/√(−a²))/√(−a²)`, `∫ dx/(x² + a)` was `atan(x/√a)/√a` for every
/// `a`: an arctangent is written only where the quadratic is real.
///
/// SymPy: `integrate(1/(x**2 - a**2), x)` → `(log(-a + x)/2 - log(a + x)/2)/a`;
/// `integrate(1/(x**2 + a), x)` →
/// `-sqrt(-1/a)*log(x - a*sqrt(-1/a))/2 + sqrt(-1/a)*log(x + a*sqrt(-1/a))/2`;
/// `integrate(1/((x + a)*(x + b)), x)` → `-log(a + x)/(a - b) + log(b + x)/(a - b)`.
#[test]
fn parametric_rational_forms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases = [
        (
            "1/(x^2 - a^2)",
            "Piecewise(-ln(a + x)/(2*a) + ln(-a + x)/(2*a) if a != 0, -1/x if True)",
        ),
        (
            "1/(x^2 + a)",
            "Piecewise((-ln(x + sqrt(-a)) + ln(x - sqrt(-a)))/(2*sqrt(-a)) if a != 0, -1/x if True)",
        ),
        (
            "1/((x + a)*(x + b))",
            "Piecewise(-ln(a + x)/(a - b) + ln(b + x)/(a - b) if b != a, -1/(a + x) if True)",
        ),
        (
            "1/(x^2 + a*x + b)",
            "(-ln(1/2*a + x + 1/2*sqrt(a^2 - 4*b)) + ln(1/2*a + x - 1/2*sqrt(a^2 - 4*b)))/sqrt(a^2 - 4*b)",
        ),
    ];
    for (src, want) in cases {
        assert_eq!(p(&ctx, src).integrate(&x).to_string(), want, "∫ {src}");
    }
}

/// With the parameters declared real the quadratic is real, and where
/// `−Δ ≥ 0` is proved the arctangent form comes back; where the sign of
/// the discriminant is open the logarithms stay (valid for both signs).
///
/// SymPy (`a`, `b` positive): `integrate(1/(x**2 + a), x)` →
/// `atan(x/sqrt(a))/sqrt(a)`; (`a` real) `integrate(1/(x**2 + a**2), x)` →
/// `atan(x/a)/a`; `integrate(1/(a*x**2 + b), x)` (positive) →
/// `atan(sqrt(a)*x/sqrt(b))/(sqrt(a)*sqrt(b))`.
#[test]
fn arctangent_only_for_real_quadratics() {
    let ctx = Context::new();
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let x = ctx.symbol("x");
    let big_f = (x.powi(2) + &a).powi(-1).integrate(&x);
    assert_eq!(
        big_f.to_string(),
        "Piecewise(atan(x/sqrt(a))/sqrt(a) if a != 0, -1/x if True)"
    );
    let ctx = Context::new();
    let a = ctx.symbol_with("a", &[Assumption::Real]).unwrap();
    let x = ctx.symbol("x");
    let big_f = (x.powi(2) + a.powi(2)).powi(-1).integrate(&x);
    assert_eq!(
        big_f.to_string(),
        "Piecewise(atan(x/a)/a if a != 0, -1/x if True)"
    );
    // Real `a`, open sign of `−a`: logarithms, correct for either sign.
    let f = (x.powi(2) + &a).powi(-1);
    let big_f = f.integrate(&x);
    assert!(!big_f.to_string().contains("atan"), "{big_f}");
    for v in ["6/5", "-6/5"] {
        assert_antiderivative_at(&ctx, &f, &big_f, &x, &[("a", v)], &POINTS);
    }
}

/// `∫ x·(1 + a²x²)·atan(a·x) dx` stayed unevaluated, and so did
/// `∫ x·(1 + x²)·atan x dx`: the three-factor rule takes `u = x`, while
/// by parts needs `dv = x·(1 + a²x²)`; the product of the polynomial
/// factors is multiplied out first now.  The remaining rational integral
/// has parametric coefficients (the new integrator).
///
/// SymPy: `integrate(x*(1 + a**2*x**2)*atan(a*x), x)` →
/// `Piecewise((a**2*x**4*atan(a*x)/4 - a*x**3/12 + x**2*atan(a*x)/2 - x/(4*a) + atan(a*x)/(4*a**2), Ne(a, 0)), (0, True))`;
/// `integrate(x*(1 + x**2)*atan(x), x)` →
/// `x**4*atan(x)/4 - x**3/12 + x**2*atan(x)/2 - x/4 + atan(x)/4`.
#[test]
fn polynomial_factors_times_atan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(&ctx, "x*(1 + a^2*x^2)*atan(a*x)");
    let big_f = f.integrate(&x);
    for binding in &BINDINGS {
        assert_antiderivative_at(&ctx, &f, &big_f, &x, binding, &POINTS);
    }
    assert_eq!(
        p(&ctx, "x*(1 + x^2)*atan(x)").integrate(&x).to_string(),
        "-1/12*x^3 - 1/4*x + 1/4*atan(x) + (1/4*x^4 + 1/2*x^2)*atan(x)"
    );
}

/// The degenerate-case wrapper solved the denominator `a − b` for `a` and
/// then for `b`, and nested the whole case analysis once more for `b = a`,
/// which the first branch already covers: `∫ dx/((a − b)·x + 1)` came out
/// as `Piecewise(Piecewise(ln(…)/(a − b) if b ≠ a, x) if a ≠ b, x)`, and
/// `∫ dx/((x + a)(x + b)(x + c))` nested six levels deep (now three: one
/// level per pair of coinciding factors).
///
/// SymPy: `integrate(1/((a - b)*x + 1), x)` → `log(x*(a - b) + 1)/(a - b)`
/// (no case split); at `b = a` the integrand is 1.
#[test]
fn degenerate_cases_are_not_nested_twice() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        p(&ctx, "1/((a - b)*x + 1)").integrate(&x).to_string(),
        "Piecewise(ln(x*(a - b) + 1)/(a - b) if b != a, x if True)"
    );
    let big_f = p(&ctx, "1/((x + a)*(x + b)*(x + c))").integrate(&x);
    assert_eq!(big_f.to_string().matches("Piecewise").count(), 6, "{big_f}");
}

/// Found by the parametric-rational hunter, each unevaluated before:
/// a denominator that is itself a nested fraction (`x·(−x/(x + 2) + …)`
/// is `x²/(x + 2)`: the `ℚ` integrator's polynomial conversion failed);
/// one linear factor written as a nested fraction or as two factors (the
/// `(a·x + b)ⁿ` routes, which the new integrator leaves such denominators
/// to, miss them); and a degenerate-case branch that was itself partly
/// unevaluated (`b = 1 − a` makes the second term undefined), which made
/// the whole answer unevaluated.
///
/// SymPy: `integrate(1/(x*(-x/(x + 2) + x*(-x/(x + 2) + 1))), x)` →
/// `log(x) - 2/x`; `integrate(1/((a + 1)/x + 1), x)` →
/// `x + (-a - 1)*log(a + x + 1)`;
/// `integrate(2*a/((2*a*x - 2)**2*(1 - a*x)), x)` →
/// `a/(4*a**3*x**2 - 8*a**2*x + 4*a)`;
/// `integrate(x + 1/(-x/(a + b + x) + x*(-x/(a + b + x) + 1)), x)` →
/// `(x**2*(a/2 + b/2 - 1/2) + x + (a + b)*log(x))/(a + b - 1)`.
#[test]
fn nested_and_split_denominators() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        p(&ctx, "1/(x*(-x/(x + 2) + x*(-x/(x + 2) + 1)))")
            .integrate(&x)
            .to_string(),
        "-2/x + ln(abs(x))"
    );
    assert_eq!(
        p(&ctx, "1/((a + 1)/x + 1)").integrate(&x).to_string(),
        "x + (-a - 1)*ln(a + x + 1)"
    );
    for src in [
        "2*a/((2*a*x - 2)^2*(1 - a*x))",
        "x + 1/(-x/(a + b + x) + x*(-x/(a + b + x) + 1))",
    ] {
        let f = p(&ctx, src);
        let big_f = f.integrate(&x);
        for binding in &BINDINGS {
            assert_antiderivative_at(&ctx, &f, &big_f, &x, binding, &POINTS);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Definite integrals: poles that are real for some complex parameter
// ═══════════════════════════════════════════════════════════════════════════

/// The singularity scan dropped every parametric pole containing `i`
/// (it read the parameters as real): `∫₀¹ dx/(x + a − i)` was
/// `ln(a − i + 1) − ln(a − i)` for an unassumed `a`, finite also for
/// `a = −1/2 + i`, where the pole `x = −a + i = 1/2` lies on the path and
/// the integral diverges.  An unassumed parameter may be complex
/// (decision D4), so the integral stays unevaluated; for a declared-real
/// `a` the pole is never real and the closed form is returned.
///
/// SymPy: `integrate(1/(x + a - I), (x, 0, 1))` →
/// `-log(a - I) + log(a + 1 - I)` (for every `a`).  mpmath (`mp.dps = 30`,
/// 40 agrees): `quad(lambda t: 1/(t + mpf(1)/2 - 1j), [0, 1])` →
/// `0.47775572251371818072636405417 + 0.519146114246522951771454379553j`.
#[test]
fn definite_integral_refuses_a_pole_of_undecided_realness() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (lo, hi) = (ctx.int(0), ctx.int(1));
    let v = p(&ctx, "1/(x + a - I)").integrate_definite(&x, &lo, &hi);
    assert!(v.has_unevaluated(), "{v}");

    let ctx = Context::new();
    let a = ctx.symbol_with("a", &[Assumption::Real]).unwrap();
    let x = ctx.symbol("x");
    let (lo, hi) = (ctx.int(0), ctx.int(1));
    let v = p(&ctx, "1/(x + a - I)").integrate_definite(&x, &lo, &hi);
    assert!(!v.has_unevaluated(), "{v}");
    let z = v.subs(&a, &ctx.rational(1, 2)).eval_complex64().unwrap();
    assert!((z.re - 0.47775572251371818).abs() < 1e-12, "{z}");
    assert!((z.im - 0.51914611424652295).abs() < 1e-12, "{z}");
}
