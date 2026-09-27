//! The integrator's cost (0.31, fourth pass) and the integrand gaps the
//! third pass left: rational functions whose numerator carries more than
//! four parameters, the self-check of `RootSum` answers, degenerate
//! parameter cases, `∫ f(ln x) dx`, the `x⁸ + 1` partial fractions behind
//! `∫ sinh x·tanh 4x dx`; and `together` of `0/0`.
//!
//! Each test says what was wrong before; every reference value cites the
//! oracle call that produced it (SymPy 1.14, mpmath 1.3.0, or Rubi's
//! optimal antiderivative from `rubi-harness/suite`).

// Reference values are quoted at the digits the oracle printed them.
#![allow(clippy::excessive_precision)]

use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap_or_else(|e| panic!("{s}: {e}"))
}

/// `F′ = f` (to `1e-9`) at the real sample points, with the parameters
/// bound first (as in `v30_integrate3.rs`).
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

const POINTS: [&str; 5] = ["1/3", "7/5", "-5/7", "13/4", "-13/4"];

/// `F(hi) − F(lo)` to 15 digits against the oracle's value.
fn assert_difference(ctx: &Context, big_f: &Ex, x: &Ex, lo: &str, hi: &str, want: f64) {
    let d = big_f.subs(x, &p(ctx, hi)) - big_f.subs(x, &p(ctx, lo));
    let got: f64 = d
        .eval_decimal(25)
        .unwrap_or_else(|e| panic!("{big_f} on [{lo}, {hi}]: {e}"))
        .parse()
        .unwrap();
    assert!(
        (got - want).abs() < 1e-15 * want.abs().max(1.0),
        "{big_f} on [{lo}, {hi}]: {got} ≠ {want}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational functions with parameters in the numerator
// ═══════════════════════════════════════════════════════════════════════════

/// The rational integrator over `ℚ(params)` refused more than four
/// parameters in all, so `∫ x²(d + e·x² + f·x⁴ + g·x⁶)/(a + b·x² + c·x⁴)² dx`
/// (seven) went through by parts and the distribution of the numerator:
/// every piece integrated and checked on its own, a separate `RootSum` for
/// each, 1 s and a 56 KB answer (the slowest Rubi entries of 0.31).
/// Parameters that occur only in the numerator now stay in the numerators
/// of the coefficient field: one `RootSum`, 2 KB.  The degenerate case
/// `c = b²/(4a)` no longer carries the unreachable sub-case `a = 0` (whose
/// substituted integrand evaluated to `0`).
///
/// Oracle: Rubi, `1.2.2.6 P(x) (d x)^m (a+b x^2+c x^4)^p.mac` entry 127
/// (an `atan` form); mpmath `quad(x²(d + e x² + f x⁴ + g x⁶)/(a + b x² +
/// c x⁴)², [0, 1])` at `a, …, g = 6/5, 3/4, 5/3, 2/7, 11/9, 13/6, 5/8` →
/// `0.117743702576528807252985221432`.
#[test]
fn numerator_parameters_stay_in_the_coefficient_field() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(
        &ctx,
        "x^2*(d + e_*x^2 + f*x^4 + g*x^6)/(a + b*x^2 + c*x^4)^2",
    );
    let big_f = f.integrate(&x);
    let shown = big_f.to_string();
    assert_eq!(shown.matches("RootSum(").count(), 1, "{shown}");
    assert!(shown.len() < 5_000, "{} bytes: {shown}", shown.len());
    assert!(!shown.contains(", 0 if True"), "{shown}");
    let binding = [
        ("a", "6/5"),
        ("b", "3/4"),
        ("c", "5/3"),
        ("d", "2/7"),
        ("e_", "11/9"),
        ("f", "13/6"),
        ("g", "5/8"),
    ];
    assert_antiderivative_at(&ctx, &f, &big_f, &x, &binding, &POINTS);
    let bound = binding.iter().fold(big_f.clone(), |acc, (s, v)| {
        acc.subs(&ctx.symbol(s), &p(&ctx, v))
    });
    assert_difference(&ctx, &bound, &x, "0", "1", 0.117743702576528807252985221432);
}

/// Twelve parameters in all are more than the self-check can bind (eight
/// generic values): such an integrand keeps the route through the pieces
/// of its numerator, each checked on its own, rather than an answer from
/// the rational integrator that no check could test.
///
/// Oracle: Rubi, `1.2.3.5 P(x) (d x)^m (a+b x^n+c x^(2 n))^p.mac` entry 1
/// (verified by differentiation here).
#[test]
fn nine_numerator_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(
        &ctx,
        "(d + e_*x + f*x^2 + g*x^3 + h*x^4 + j*x^5 + k*x^6 + l*x^7 + m*x^8)/(a + b*x^3 + c*x^6)",
    );
    let big_f = f.integrate(&x);
    let binding = [
        ("a", "6/5"),
        ("b", "3/4"),
        ("c", "5/3"),
        ("d", "2/7"),
        ("e_", "11/9"),
        ("f", "13/6"),
        ("g", "5/8"),
        ("h", "-3/7"),
        ("j", "7/4"),
        ("k", "1/9"),
        ("l", "-5/6"),
        ("m", "4/3"),
    ];
    assert_antiderivative_at(&ctx, &f, &big_f, &x, &binding, &POINTS);
}

// ═══════════════════════════════════════════════════════════════════════════
// ∫ f(ln x) dx
// ═══════════════════════════════════════════════════════════════════════════

/// `∫ sin(ln x) dx` and its relatives stayed unevaluated: no route
/// substituted `x = e^w`.  The substitution `w = ln(c·xⁿ)` turns them into
/// an exponential times trigonometric functions of `w`.
///
/// Oracle: SymPy `integrate(sin(log(x)), x)` → `x*sin(log(x))/2 -
/// x*cos(log(x))/2`; `integrate(cos(log(x)), x)` → `x*sin(log(x))/2 +
/// x*cos(log(x))/2`; `integrate(x**2*sin(log(x)), x)` →
/// `3*x**3*sin(log(x))/10 - x**3*cos(log(x))/10`;
/// `integrate(sin(3*log(x)), x)` → `x*sin(3*log(x))/10 -
/// 3*x*cos(3*log(x))/10`; `integrate(sin(log(x))/x**2, x)` →
/// `-sin(log(x))/(2*x) - cos(log(x))/(2*x)`; mpmath
/// `quad(sin(log(x)), [1, 2])` → `0.369722374949662674571702917803`.
#[test]
fn trigonometric_functions_of_a_logarithm() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (f, want) in [
        ("sin(ln(x))", "-1/2*x*cos(ln(x)) + 1/2*x*sin(ln(x))"),
        ("cos(ln(x))", "1/2*x*sin(ln(x)) + 1/2*x*cos(ln(x))"),
        (
            "x^2*sin(ln(x))",
            "-1/10*x^3*cos(ln(x)) + 3/10*x^3*sin(ln(x))",
        ),
        ("sin(3*ln(x))", "-3/10*x*cos(3*ln(x)) + 1/10*x*sin(3*ln(x))"),
        ("sin(ln(x))/x^2", "-sin(ln(x))/(2*x) - cos(ln(x))/(2*x)"),
    ] {
        let f = p(&ctx, f);
        let big_f = f.integrate(&x);
        assert_eq!(big_f.to_string(), want, "∫ {f}");
        assert_antiderivative_at(&ctx, &f, &big_f, &x, &[], &POINTS);
    }
    let big_f = p(&ctx, "sin(ln(x))").integrate(&x);
    assert_difference(&ctx, &big_f, &x, "1", "2", 0.369722374949662674571702917803);
}

/// Hyperbolic functions and parameters: `∫ sinh(ln x) dx` (SymPy leaves it
/// unevaluated) is `x²/4 − ln|x|/2`, `∫ sin(a·ln x) dx` has the cases
/// `a = ±i` once each (the wrap of the inner `w`-integral and of the
/// outer one nested them twice before the outer one learnt to keep an
/// inner case analysis), `∫ x²·sin(a + b·ln(c·xⁿ)) dx` is Rubi's family
/// `4.7.5` (247 of its 250 entries were unevaluated).
///
/// Oracle: mpmath `quad(sinh(log(x)), [1, 2])` →
/// `0.403426409720027345291383939271`; SymPy
/// `integrate(sin(a*log(x)), x)` → `Piecewise((-I*x**2/4 + I*log(x)/2,
/// Eq(a, -I)), (I*x**2/4 - I*log(x)/2, Eq(a, I)),
/// (-a*x*cos(a*log(x))/(a**2 + 1) + x*sin(a*log(x))/(a**2 + 1), True))`;
/// Rubi `4.7.5` entry 1: `-b*n*x^3*cos(a+b*log(c*x^n))/(9+b^2*n^2) +
/// (3*x^3*sin(a+b*log(c*x^n)))/(9+b^2*n^2)`; mpmath
/// `quad(x**2*sin(1/2 + 3/2*log(2*x**(3/4))), [1, 2])` →
/// `2.02074637061058368822704623231`.
#[test]
fn logarithms_inside_hyperbolic_functions_and_with_parameters() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(&ctx, "sinh(ln(x))");
    let big_f = f.integrate(&x);
    assert_eq!(big_f.to_string(), "1/4*x^2 - 1/2*ln(abs(x))");
    assert_difference(&ctx, &big_f, &x, "1", "2", 0.403426409720027345291383939271);

    let f = p(&ctx, "sin(a*ln(x))");
    let big_f = f.integrate(&x);
    let shown = big_f.to_string();
    assert_eq!(shown.matches("if a != I").count(), 1, "{shown}");
    assert_eq!(shown.matches("if a != -I").count(), 1, "{shown}");
    for binding in [[("a", "7/5 + 3/11*I")], [("a", "6/5")], [("a", "-6/5")]] {
        assert_antiderivative_at(&ctx, &f, &big_f, &x, &binding, &["1/3", "7/5", "13/4"]);
    }

    let f = p(&ctx, "x^2*sin(a + b*ln(c*x^n))");
    let big_f = f.integrate(&x);
    let binding = [("a", "1/2"), ("b", "3/2"), ("c", "2"), ("n", "3/4")];
    assert_antiderivative_at(&ctx, &f, &big_f, &x, &binding, &["1/3", "7/5", "13/4"]);
    let bound = binding.iter().fold(big_f.clone(), |acc, (s, v)| {
        acc.subs(&ctx.symbol(s), &p(&ctx, v))
    });
    assert_difference(&ctx, &bound, &x, "1", "2", 2.02074637061058368822704623231);
}

/// A logarithm of a linear argument with no other `x` (`w = ln(a + b·x)`,
/// `dx = e^w/b·dw`), and a sum of such terms, which the stage-level
/// substitutions never saw one by one: `∫ (1 − 2x)·cos(ln(2x)/3 + 1) dx`
/// integrated but its multiplied-out form did not (the extended
/// normal-form hunter, mode `formsl`).
///
/// Oracle: Rubi `3.5 Logarithm functions.mac`-style `∫ sin(log(a + b x))`
/// = `(a + b x)(sin(log(a + b x)) − cos(log(a + b x)))/(2b)` (verified by
/// differentiation here); mpmath `quad(sin(log(1 + 2*x)), [0, 1])` →
/// `0.576808464130852978035521622228`.
#[test]
fn logarithm_of_a_linear_argument_and_sums_of_such_terms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(&ctx, "sin(ln(a + b*x))");
    let big_f = f.integrate(&x);
    assert_antiderivative_at(
        &ctx,
        &f,
        &big_f,
        &x,
        &[("a", "6/5"), ("b", "3/4")],
        &["1/3", "7/5", "13/4"],
    );
    let big_f = p(&ctx, "sin(ln(1 + 2*x))").integrate(&x);
    assert_difference(&ctx, &big_f, &x, "0", "1", 0.576808464130852978035521622228);
    let f = p(&ctx, "-2*x*cos(-1/3*ln(2*x) - 1) + cos(-1/3*ln(2*x) - 1)");
    let big_f = f.integrate(&x);
    assert!(!big_f.to_string().contains("000"), "{big_f}");
    assert_antiderivative_at(&ctx, &f, &big_f, &x, &[], &POINTS);
}

/// `∫ cosh x·coth 4x dx` integrated, but the same integrand multiplied and
/// divided by `x² + 1` did not: the normal form that cancels such a factor
/// refused a denominator with a transcendental generator (`tanh 4x`).
/// Found by the extended normal-form hunter (mode `formsl`: 167 of these
/// in 2,000 cases, 0 now).
///
/// Oracle: mpmath `quad(cosh(x)*coth(4*x), [1, 2])` →
/// `2.45180309760761410600481440851`.
#[test]
fn common_factor_over_a_transcendental_denominator_cancels() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = p(
        &ctx,
        "((cosh(x)*coth(4*x))*x^2 + cosh(x)*coth(4*x))/(x^2 + 1)",
    );
    let big_f = f.integrate(&x);
    assert_antiderivative_at(&ctx, &f, &big_f, &x, &[], &["1/3", "7/5", "13/4"]);
    assert_difference(&ctx, &big_f, &x, "1", "2", 2.45180309760761410600481440851);
}

// ═══════════════════════════════════════════════════════════════════════════
// The partial fractions of x⁸ + 1
// ═══════════════════════════════════════════════════════════════════════════

/// `∫ sinh x·tanh 4x dx` came out as a 140 KB sum of nested `cos(π/8)`,
/// `sin(π/8)` products: through `u = eˣ` it is
/// `∫ (u² − 1)(u⁸ − 1)/(2u²(u⁸ + 1)) du`, whose algebraic part the
/// heuristic integrator decomposed over the roots of `u⁸ + 1` without
/// ever collecting the products.  A closed
/// form of that remainder larger than 2,000 nodes now gives way to the
/// `RootSum` of the Rothstein–Trager factor (degree 4, solved in radicals).
///
/// Oracle: SymPy `integrate(sinh(x)*tanh(4*x), x)` stays unevaluated; Rubi
/// `6.7.1` entry 202:
/// `sinh(x) - 1/4*atan(2*sinh(x)/sqrt(2-sqrt(2)))*sqrt(2-sqrt(2)) -
/// 1/4*atan(2*sinh(x)/sqrt(2+sqrt(2)))*sqrt(2+sqrt(2))`; mpmath
/// `quad(sinh(x)*tanh(4*x), [0, 1])` → `0.51702502053565553602981184637`,
/// `quad(cosh(x)*tanh(4*x), [0, 1])` → `0.998476410446793202718735894168`.
#[test]
fn partial_fractions_over_the_roots_of_x8_plus_1_stay_compact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (f, want) in [
        ("sinh(x)*tanh(4*x)", 0.51702502053565553602981184637),
        ("cosh(x)*tanh(4*x)", 0.998476410446793202718735894168),
    ] {
        let f = p(&ctx, f);
        let big_f = f.integrate(&x);
        let shown = big_f.to_string();
        assert!(shown.len() < 2_000, "∫ {f}: {} bytes", shown.len());
        assert_antiderivative_at(&ctx, &f, &big_f, &x, &[], &POINTS);
        assert_difference(&ctx, &big_f, &x, "0", "1", want);
    }
    let f = p(&ctx, "(x^2 - 1)*(x^8 - 1)/(2*x^2*(x^8 + 1))");
    let big_f = f.integrate(&x);
    assert!(big_f.to_string().len() < 2_000, "{big_f}");
    assert_antiderivative_at(&ctx, &f, &big_f, &x, &[], &POINTS);
}

// ═══════════════════════════════════════════════════════════════════════════
// Substitutions that need trigsimp still work
// ═══════════════════════════════════════════════════════════════════════════

/// The u-substitution skips `trigsimp` when the quotient `remaining/u′`
/// takes different values at two sample points (it cannot become
/// constant); the quotients that are constants only up to an identity
/// (`sin 2x/(2 sin x cos x)`) still reach it.  Before, every quotient went
/// through `trigsimp`'s seven strategies: 60 % of the time of the Rubi
/// entries that stay unevaluated.
///
/// Oracle: SymPy `integrate(sin(2*x)*exp(sin(x)**2), x)` →
/// `exp(sin(x)**2)`; `integrate(sin(2*x)/(1 + sin(x)**2), x)` →
/// `log(sin(x)**2 + 1)`.
#[test]
fn substitutions_through_a_trigonometric_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (f, want) in [
        ("sin(2*x)*exp(sin(x)^2)", "exp(sin(x)^2)"),
        ("sin(2*x)/(1 + sin(x)^2)", "ln(abs(sin(x)^2 + 1))"),
    ] {
        let f = p(&ctx, f);
        assert_eq!(f.integrate(&x).to_string(), want, "∫ {f}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// together of 0/0
// ═══════════════════════════════════════════════════════════════════════════

/// `together((x(x + 1) − x² − x)/(−x/(x + 1) + x·(−x/(x + 1) + 1)))` gave
/// `zoo` while `ratsimp` gave `nan`: the denominator cancels to the literal
/// 0, and `together` did not see that the numerator vanishes too (it is 0
/// only once multiplied out).  `0/0` is undefined for every `x`: `nan`
/// from both, the crate's rule elsewhere.  A numerator that does not
/// vanish still gives `zoo`.
///
/// Oracle: SymPy `together(e)` → `nan`, `ratsimp(e)` → `0`, `cancel(e)` →
/// `0` (these two return the zero numerator before they look at the
/// denominator, which is not the rational function's value);
/// `together(1/(-x/(x + 1) + x*(-x/(x + 1) + 1)))` → `zoo`.
#[test]
fn together_of_zero_over_zero_is_nan() {
    let ctx = Context::new();
    let e = p(
        &ctx,
        "(x*(x + 1) - x^2 - x)/(-x/(x + 1) + x*(-x/(x + 1) + 1))",
    );
    assert_eq!(e.together().to_string(), "nan");
    assert_eq!(e.ratsimp().to_string(), "nan");
    let e = p(&ctx, "1/(-x/(x + 1) + x*(-x/(x + 1) + 1))");
    assert_eq!(e.together().to_string(), "zoo");
    assert_eq!(e.ratsimp().to_string(), "zoo");
    // Unchanged: ordinary fractions.
    let e = p(&ctx, "1/x + 1/(x + 1)");
    assert_eq!(e.together().to_string(), "(2*x + 1)/(x^2 + x)");
}
