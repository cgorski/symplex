//! Exact special values (`eval` and the canonical folds of function
//! applications): the bug hunt of the closed forms at special arguments.
//! Every reference value cites mpmath 1.3.0 (50 digits) or SymPy 1.14.

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `li z = Ei(Log z)`, so `li(e^y) = Ei(y)` only for `Im y ∈ (−π, π]`.
/// Before: `li(exp(4i))` → `Ei(4i)` (mpmath `li(exp(4j))` =
/// `0.35201818981023262327 − 3.2874912358609361478j`, `ei(4j)` =
/// `−0.14098169788693041164 + 3.3289994657439496773j`), `li(exp(1 + 4i))`
/// → `Ei(1 + 4i)` (mpmath `li` = `0.76743935230521806568 −
/// 3.8755821309410486062j`), and `li(exp(x))` → `Ei(x)` for a complex
/// symbol (wrong at `x = −1.3 + 4.1i`).  SymPy 1.14 keeps `li(exp(y))`.
/// Inside the strip, on its edge `Im y = π` (`e^{1+πi} = −e`; mpmath
/// `li(-e)` = `ei(1 + πj)` = `−0.02225078580297302039 +
/// 3.9479261440774262340j`) and for a real symbol the fold stays.
#[test]
fn li_of_exp_folds_to_ei_only_on_the_principal_strip() {
    let ctx = Context::new();
    for s in [
        "li(exp(4*I))",
        "li(exp(1 + 4*I))",
        "li(exp(-7*I/2))",
        "li(exp(x))",
    ] {
        let e = parse(&ctx, s);
        assert_eq!(e.eval(), e, "{s}");
    }
    for (s, want) in [
        ("li(exp(3*I))", "Ei(3*I)"),
        ("li(exp(1 + pi*I))", "Ei(1 + pi*I)"),
        ("li(exp(2))", "Ei(2)"),
        ("li(E)", "Ei(1)"),
    ] {
        assert_eq!(parse(&ctx, s).eval(), parse(&ctx, want), "{s}");
    }
    let t = ctx.symbol_with("t", &[Assumption::Real]).unwrap();
    assert_eq!(t.exp().li(), t.ei());
}

/// `Ai′(x)` and `Bi′(x)` oscillate with the growing amplitude `|x|^(1/4)/√π`
/// as `x → −∞` (DLMF 9.7.10, 9.7.12; mpmath `airyai(-10**6, 1)` =
/// `17.706164485139947379`, `airyai(-10**6 - 2, 1)` =
/// `−8.5599208806807098082`): no limit.  Before:
/// `airyaiprime(-oo)` and `airybiprime(-oo)` were `0` (SymPy 1.14 keeps
/// `airyaiprime(-oo)`; its `airybiprime(-oo)` → `0` is the same mistake).
/// `Ai(−∞) = Bi(−∞) = 0` stay.
#[test]
fn airy_derivatives_have_no_value_at_minus_infinity() {
    let ctx = Context::new();
    for s in ["airyaiprime(-oo)", "airybiprime(-oo)"] {
        let e = parse(&ctx, s);
        assert_eq!(e.eval(), e, "{s}");
    }
    for s in ["airyai(-oo)", "airybi(-oo)"] {
        assert_eq!(parse(&ctx, s).eval(), ctx.int(0), "{s}");
    }
}

/// `Y_{−n−1/2}(z) = (−1)ⁿ J_{n+1/2}(z)` vanishes at `z = 0`.  Before:
/// `bessely(-1/2, 0)`, `bessely(-3/2, 0)`, `bessely(-9/2, 0)` were `zoo`
/// (as SymPy 1.14's); mpmath `bessely(-0.5, 0)` = `0.0`,
/// `bessely(-1.5, 1e-20)` = `−2.6596152026762178529e-31`.  The other
/// orders keep their poles.
#[test]
fn bessel_y_of_negative_half_integer_order_vanishes_at_zero() {
    let ctx = Context::new();
    for s in ["bessely(-1/2, 0)", "bessely(-3/2, 0)", "bessely(-9/2, 0)"] {
        assert_eq!(parse(&ctx, s), ctx.int(0), "{s}");
        assert_eq!(parse(&ctx, s).eval_decimal(20).unwrap(), "0", "{s}");
    }
    for s in [
        "bessely(1/2, 0)",
        "bessely(-1/3, 0)",
        "bessely(2, 0)",
        "besselk(-1/2, 0)",
    ] {
        assert_eq!(parse(&ctx, s), ctx.complex_infinity(), "{s}");
    }
    assert_eq!(parse(&ctx, "bessely(0, 0)"), ctx.neg_infinity());
}

/// `eval` folds the applications its own rules build, so a second `eval`
/// changes nothing.  Before, each of these needed two: `|exp(e + 3 − πi)|`
/// stopped at `|exp(3 + e)|` (`exp(e + 3 − πi) = −exp(3 + e)`),
/// `polylog(1, exp(iπ/3))` at `−ln(1/2 − √3·i/2)` (= `iπ/3`; mpmath
/// `polylog(1, exp(pi*j/3))` = `1.0471975511965977462j`, SymPy 1.14
/// `-log(1/2 - sqrt(3)*I/2)` → `I*pi/3`), `elliptic_f(−π/2, i)` at
/// `−elliptic_f(π/2, i)` (= `−K(i)`, mpmath `ellipf(-pi/2, 1j)` =
/// `−ellipk(1j)`), `expint(0, e⁻³)` at `exp(−e⁻³)/exp(−3)`.
#[test]
fn eval_is_idempotent_on_the_forms_its_rules_build() {
    let ctx = Context::new();
    for (s, want) in [
        ("abs(exp(E + 3 - pi*I))", "exp(3 + E)"),
        ("abs(-2*exp(sqrt(2)))", "2*exp(sqrt(2))"),
        ("polylog(1, exp(I*pi/3))", "I*pi/3"),
        ("polylog(1, 2)", "-pi*I"),
        ("elliptic_f(-pi/2, I)", "-elliptic_k(I)"),
        ("expint(0, exp(-3))", "exp(3)*exp(-exp(-3))"),
    ] {
        let e = parse(&ctx, s).eval();
        assert_eq!(e, parse(&ctx, want).eval(), "{s}");
        assert_eq!(e.eval(), e, "{s}");
    }
}

/// A function of `nan` is `nan`.  Before: `KroneckerDelta(nan, nan)` was `1`
/// (a value for an undefined comparison; SymPy 1.14 leaves it unevaluated),
/// `KroneckerDelta(nan, 1)`, `polygamma(nan, 1)` and `polygamma(1, nan)`
/// stayed (SymPy 1.14: `polygamma(nan, 1)` → `nan`).
#[test]
fn kronecker_delta_and_polygamma_of_nan_are_nan() {
    let ctx = Context::new();
    for s in [
        "KroneckerDelta(nan, nan)",
        "KroneckerDelta(nan, 1)",
        "polygamma(nan, 1)",
        "polygamma(1, nan)",
    ] {
        assert_eq!(parse(&ctx, s).eval(), ctx.nan(), "{s}");
    }
    assert_eq!(parse(&ctx, "KroneckerDelta(2, 2)"), ctx.int(1));
}

/// An order between real constants whose difference is a rational number
/// is decided by `eval`.  Before: `tanh(1) < tanh(1)` stayed, and `evalf`
/// decided it true (the sign test of `+0`), so `Piecewise((π, tanh 1 <
/// tanh 1), (0, True))` evaluated to `π` (SymPy 1.14: `0`).
#[test]
fn order_of_real_constants_with_a_rational_difference_is_decided() {
    let ctx = Context::new();
    for (s, want) in [
        ("Piecewise(pi if tanh(1) < tanh(1), 0 if True)", "0"),
        ("Piecewise(pi if zeta(3) > zeta(3), 0 if True)", "0"),
        ("Piecewise(pi if tanh(1) <= tanh(1), 0 if True)", "pi"),
        ("Piecewise(pi if pi + 1 > pi, 0 if True)", "pi"),
        ("Piecewise(pi if sqrt(2) - 1/2 > sqrt(2), 0 if True)", "0"),
    ] {
        assert_eq!(parse(&ctx, s).eval(), parse(&ctx, want), "{s}");
    }
    let pw = parse(&ctx, "Piecewise(pi if tanh(1) < tanh(1), 0 if True)");
    assert_eq!(pw.eval_decimal(20).unwrap(), "0");
    // Not real, or not a number apart: left alone.
    for s in [
        "Piecewise(pi if I + 1 > I, 0 if True)",
        "Piecewise(pi if x + 1 > x, 0 if True)",
    ] {
        let e = parse(&ctx, s);
        assert_eq!(e.eval(), e, "{s}");
    }
}
