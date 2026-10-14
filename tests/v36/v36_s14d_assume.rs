//! Tests of session 14 (s14d): 1/0 claimed finite, delta off the reals.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, src: &str) -> Ex {
    ctx.parse(src).unwrap()
}

/// `1/max(−Γ(1/2), 0)` and `1/min(W(π), 0)` are `1/0 = zoo` (`Γ(1/2) = √π`
/// and `W(π)` are positive, so both bases are 0), as are `1/floor(W(1/2))`,
/// `1/ceiling(−W(1/2))` and `1/H(−W(1))`.  Up to 0.50 the assumption
/// system called all five finite and real (`Some(true)`, from "real^integer
/// is real" and "finite^finite is finite" applied to a base not known
/// nonzero), while `eval` gave `zoo`.
///
/// Oracle: mpmath `gamma(0.5)` = 1.7724538509…, `lambertw(pi)` =
/// 1.0736581947…, `lambertw(0.5)` = 0.3517337112…, `lambertw(1)` =
/// 0.5671432904…; SymPy 1.14: `1/Max(-gamma(S(1)/2), 0)`,
/// `1/floor(LambertW(S(1)/2))`, `1/Heaviside(-LambertW(1))` are `zoo`.
#[test]
fn reciprocal_of_a_hidden_zero_is_not_called_finite_or_real() {
    for src in [
        "1/max(-gamma(1/2), 0)",
        "1/min(LambertW(pi), 0)",
        "1/floor(LambertW(1/2))",
        "1/ceiling(-LambertW(1/2))",
        "1/Heaviside(-LambertW(1))",
        "max(-gamma(1/2), 0)^(-2)",
    ] {
        let ctx = Context::new();
        let e = parse(&ctx, src);
        assert_ne!(ctx.query(&e, Props::FINITE), Some(true), "{src}");
        assert_ne!(e.is_finite(), Some(true), "{src}");
        assert_ne!(e.is_real(), Some(true), "{src}");
        assert_ne!(e.is_positive(), Some(true), "{src}");
        assert_eq!(e.eval().to_string(), "zoo", "{src}");
        assert!(e.eval_decimal(15).is_err(), "{src}");
    }
    // The base itself is decided: 0.
    let ctx = Context::new();
    assert_eq!(parse(&ctx, "min(LambertW(pi), 0)").is_zero(), Some(true));
}

/// A negative power of a constant is finite only where the constant is
/// nonzero: the structural rules prove that for few constants, a certified
/// value for the others (`ln 3`, `π − 3`, `1 − Catalan`, `sin 2`), and a
/// hidden 0 has neither.  The reciprocals of nonzero constants stay real
/// and finite (the rule of 0.50 called every one so, the zeros included);
/// a symbol-dependent base keeps the generic reading of the module (real
/// away from its poles: `1/x` for a real `x`).
///
/// Oracle: mpmath `log(3)` = 1.0986…, `pi − 3` = 0.14159…, `1 − catalan` =
/// 0.08403…, `sin(2)` = 0.90929…; `1/(atan(1/2) + atan(1/3) − π/4)` is
/// `1/0` (Euler's identity).
#[test]
fn a_negative_power_of_a_constant_needs_it_nonzero() {
    let ctx = Context::new();
    for src in ["1/ln(3)", "1/(pi - 3)", "(1 - Catalan)^(-2)", "sin(2)^(-3)"] {
        let e = parse(&ctx, src);
        assert_eq!(ctx.query(&e, Props::REAL), Some(true), "{src}");
        assert_eq!(ctx.query(&e, Props::FINITE), Some(true), "{src}");
    }
    for src in [
        "1/(atan(1/2) + atan(1/3) - pi/4)",
        "(sin(1)^2 + cos(1)^2 - 1)^(-2)",
    ] {
        let e = parse(&ctx, src);
        assert_ne!(ctx.query(&e, Props::REAL), Some(true), "{src}");
        assert_ne!(ctx.query(&e, Props::FINITE), Some(true), "{src}");
        assert_ne!(e.is_real(), Some(true), "{src}");
    }
    let _x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    assert_eq!(parse(&ctx, "1/x").is_real(), Some(true));
}

/// The value of `min`/`max` of reals is one of the arguments: `min(p, q)` of
/// positive `p`, `q` is positive, `max(x, 1) ≥ 1` for a real `x`, `min(x,
/// −1) ≤ −1`, and so are their reciprocals.  Up to 0.50 `min` and `max`
/// had no sign (`None`).
///
/// Oracle: SymPy 1.14 (`MinMaxBase`): `Min(p, q).is_positive`,
/// `Max(x, 1).is_positive`, `Min(x, -1).is_negative` and
/// `(1/Min(p, q)).is_positive` are `True`.
#[test]
fn min_and_max_of_signed_arguments_have_a_sign() {
    let ctx = Context::new();
    for (name, a) in [
        ("x", Assumption::Real),
        ("y", Assumption::Real),
        ("p", Assumption::Positive),
        ("q", Assumption::Positive),
    ] {
        let _declared = ctx.symbol_with(name, &[a]).unwrap();
    }
    for src in [
        "min(p, q)",
        "max(x, 1)",
        "max(p, x)",
        "1/min(p, q)",
        "1/max(x, 1)",
    ] {
        let e = parse(&ctx, src);
        assert_eq!(e.is_positive(), Some(true), "{src}");
        assert_eq!(e.is_finite(), Some(true), "{src}");
    }
    for src in ["min(x, -1)", "1/min(x, -1)"] {
        let e = parse(&ctx, src);
        assert_eq!(e.is_negative(), Some(true), "{src}");
        assert_eq!(e.is_real(), Some(true), "{src}");
    }
    assert_eq!(parse(&ctx, "min(x, y)").is_positive(), None);
}

/// `min(atan(1/2) + atan(1/3) − π/4, 0)`: the first argument is exactly 0
/// (Euler's `atan(1/2) + atan(1/3) = π/4`), so `eval` gives 0 and `1/min(…)`
/// evaluates to `zoo`.  Up to 0.50 `eval` decided the comparison only by a
/// certified nonzero difference, kept the `min`, and left `1/min(…)` a
/// `1/0` in disguise (`eval_decimal`: precision exhausted).  It also kept
/// `min(Ai(√2), 1/10)`: an argument had to be finite by the assumptions,
/// which claim nothing for `Ai` of a constant; a certified value now shows
/// it finite.
///
/// Oracle: the identity `tan(atan(1/2) + atan(1/3)) = (1/2 + 1/3)/(1 −
/// 1/6) = 1`; mpmath at 30 digits: `atan(0.5) + atan(1/3) − pi/4` = 0.0,
/// `airyai(sqrt(2))` = 0.080507310075939356, `lambertw(pi)` = 1.0736….
/// (SymPy 1.14 rejects the `Min` of the hidden zero: "not comparable".)
#[test]
fn min_of_an_exact_zero_and_zero_evaluates_to_zero() {
    let ctx = Context::new();
    let m = parse(&ctx, "min(atan(1/2) + atan(1/3) - pi/4, 0)");
    assert_eq!(m.eval().to_string(), "0");
    let r = parse(&ctx, "1/min(atan(1/2) + atan(1/3) - pi/4, 0)");
    assert_eq!(r.eval().to_string(), "zoo");
    // A decided order still picks the extreme value.
    assert_eq!(
        parse(&ctx, "max(LambertW(pi), 1)").eval().to_string(),
        parse(&ctx, "LambertW(pi)").to_string()
    );
    // Finite by its value, not by the assumptions.
    assert_eq!(
        parse(&ctx, "min(airyai(sqrt(2)), 1/10)").eval().to_string(),
        parse(&ctx, "airyai(sqrt(2))").to_string()
    );
    assert_eq!(
        parse(&ctx, "max(1/(pi - 3), 7)").eval().to_string(),
        parse(&ctx, "1/(pi - 3)").to_string()
    );
}

/// `1/max(Γ(1/3)·Γ(2/3) − 2π/√3, 0)` is `1/0` (reflection formula:
/// `Γ(1/3)·Γ(2/3) = π/sin(π/3) = 2π/√3`).  `eval` keeps the `max` (its zero
/// is not decided there); up to 0.50 `is_real` was nevertheless `Some(true)`
/// (realness of a constant from an imaginary part that is 0 wherever the
/// constant is defined, with no value to show it finite).
///
/// Oracle: the reflection formula; mpmath at 30 digits: `gamma(1/3) *
/// gamma(2/3) − 2*pi/sqrt(3)` = 3.9e−31 (0 to the precision).
#[test]
fn a_constant_without_a_value_is_not_called_real() {
    let ctx = Context::new();
    let e = parse(&ctx, "1/max(gamma(1/3)*gamma(2/3) - 2*pi/sqrt(3), 0)");
    assert_ne!(e.is_real(), Some(true));
    assert_ne!(e.is_finite(), Some(true));
    assert!(e.eval_decimal(15).is_err());
}

/// `δ` is a distribution on the real line: no value off it.  Up to 0.50
/// `eval_decimal` and `eval_f64` gave `0` for `δ(i)`, `δ(1 + i)`, `δ(√2·i)`
/// and `δ(W(−1))` (`W₀(−1)` = −0.3181 + 1.3372i), `is_real` was
/// `Some(true)` and `compile` folded them to 0, while `is_zero` was `None`
/// and the limit engine treats `δ` off the reals as undefined.  An argument
/// only real to the working precision is refused too; a real one gives 0.
///
/// Oracle: SymPy 1.14 `DiracDelta(I)` raises `ValueError("Function defined
/// only for Real Values…")`; mpmath `lambertw(-1)` = −0.318131505204764 +
/// 1.33723570143069j; `atan(1/2) + atan(1/3) − π/4 = 0` (Euler).
#[test]
fn dirac_delta_off_the_reals_has_no_value() {
    for src in [
        "DiracDelta(I)",
        "DiracDelta(1 + I)",
        "DiracDelta(sqrt(-2))",
        "DiracDelta(LambertW(-1))",
        "1 + DiracDelta(I)",
    ] {
        let ctx = Context::new();
        let e = parse(&ctx, src);
        let err = e.eval_decimal(30).unwrap_err();
        assert!(err.to_string().contains("non-real"), "{src}: {err}");
        assert!(e.eval_f64().is_err(), "{src}");
        assert_ne!(e.is_real(), Some(true), "{src}");
        assert!(e.compile(&[]).is_err(), "{src}");
    }
    let ctx = Context::new();
    let hidden = parse(&ctx, "DiracDelta(1 + I*(atan(1/2) + atan(1/3) - pi/4))");
    assert!(hidden.eval_decimal(15).is_err());
    let real = parse(&ctx, "DiracDelta(sqrt(2) - 1)");
    assert_eq!(real.eval_decimal(15).unwrap(), "0");
    assert_eq!(real.eval_f64().unwrap(), 0.0);
    // At a real point of a compiled `δ(x)`: 0 (NaN at 0, as before).
    let x = ctx.symbol("x");
    let f = x.dirac_delta().compile(&["x"]).unwrap();
    assert_eq!(f.call(&[0.5]), 0.0);
    assert!(f.call(&[0.0]).is_nan());
}
