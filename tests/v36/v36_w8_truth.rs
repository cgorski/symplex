//! Tests of the 0.49 refactor (w8): three-valued zero tests in the limit engine, one Kleene logic, certified equality.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use std::time::{Duration, Instant};

use symplex::prelude::*;

/// `c(a) = sin(1)·(a − 1153/877)·(a − 1789/2311)`, multiplied out: a
/// constant in the positive parameter `a` that vanishes at exactly two
/// values of it — the two points at which the series engine sampled a
/// parametric constant to decide whether it vanishes identically.
fn two_point_constant(ctx: &Context) -> Ex {
    ctx.parse("sin(1)*a^2 - (1153/877 + 1789/2311)*sin(1)*a + 1153*1789/(877*2311)*sin(1)")
        .unwrap()
}

/// Up to 0.48 a constant with parameters was taken to vanish identically
/// when it was 0 to 30 digits at two sample points: `c(a)` above, which is
/// 0 only at those two points, became the zero coefficient of
/// `1/(c(a) + x)`, whose series at `0` was `1/x` and the limit of
/// `x/(c(a) + x)` at `0` was `1`.  Now two zero samples only make the exact
/// tests run, and a nonzero value at another point decides.
///
/// Oracle (exact algebra): `c(1) = sin 1·(−276/877)·(522/2311) ≠ 0`, so for
/// generic `a` the series is `1/c − x/c² + O(x²)` and `x/(c + x) → 0`;
/// at `a = 1153/877` or `a = 1789/2311`, `c = 0` and the quotient is `1`.
#[test]
fn a_parametric_constant_zero_at_two_points_is_not_zero() {
    let ctx = Context::new();
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let x = ctx.symbol("x");
    let c = two_point_constant(&ctx);
    let s = (&c + &x).powi(-1).series(&x, &ctx.int(0), 2);
    let want = &c.powi(-1) - &(&x / &c.powi(2));
    assert!((&s - &want).simplify().is_zero_structural(), "series = {s}");

    // A `Piecewise`: `1` at the two roots of `c`, `0` elsewhere.
    let l = (&x / &(&c + &x)).try_limit(&x, &ctx.int(0)).unwrap();
    for v in [1, 2] {
        let at = l.subs(&a, &ctx.int(v)).eval_f64().unwrap();
        assert_eq!(at, 0.0, "limit = {l}");
    }
}

/// A constant that is 0 by the reflection formula of `Γ` decides a leading
/// term.  Up to 0.48 its value, 0 to some 1,100 bits, was taken for a
/// proof (the same answers, on numeric evidence); a zero that is not
/// proved now ends the expansion (a refusal), so this one is proved:
/// `Γ(1/3)·Γ(2/3)` is rewritten as `π/sin(π/3)`.
///
/// Oracle: `Γ(z)Γ(1 − z) = π/sin πz`, so the denominator is `x`: the
/// limit is `1` and the series `1/x`.
#[test]
fn a_zero_by_the_reflection_formula_is_proved() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let k = ctx.parse("gamma(1/3)*gamma(2/3) - 2*pi/sqrt(3)").unwrap();
    let l = (&x / &(&k + &x)).try_limit(&x, &ctx.int(0)).unwrap();
    assert_eq!(l.to_string(), "1");
    let s = (&k + &x).powi(-1).try_series(&x, &ctx.int(0), 2).unwrap();
    assert_eq!(s.to_string(), "1/x");
}

/// A parametric leading coefficient `a + 1 − √(a² + 2a + 1)` of the Gruntz
/// algorithm, 0 for `a > 0`: proved by taking the square `(a + 1)²` out
/// of the root (the samples alone, 0 at two points, prove nothing).  A
/// strict reading of the samples without that rewrite refused the limit
/// (SymPy's test suite, `test_issue_...` of the D15 limit corpus).
///
/// Oracle: SymPy 1.14 `limit(x*(1 + a) - sqrt(vc + vb*x + x**2*(1 + a)**2),
/// x, oo)` for positive symbols is `-vb/(2*(a + 1))`: `−1/2` at `a = 1`,
/// `vb = 2`.
#[test]
fn a_parametric_leading_coefficient_is_proved_zero() {
    let ctx = Context::new();
    let positive = |s: &str| ctx.symbol_with(s, &[Assumption::Positive]).unwrap();
    let (a, vb, vc, x) = (positive("a"), positive("vb"), positive("vc"), positive("x"));
    let f = ctx
        .parse("x*(1 + a) - sqrt(vc + vb*x + x^2*(1 + a)^2)")
        .unwrap();
    let l = f.try_limit(&x, &ctx.infinity()).unwrap();
    let v = l
        .subs(&a, &ctx.int(1))
        .subs(&vb, &ctx.int(2))
        .subs(&vc, &ctx.int(3))
        .eval_f64()
        .unwrap();
    assert!((v + 0.5).abs() < 1e-14, "limit = {l}");
}

/// Constants that are not sums were never tested for 0: a product, power
/// or function value counted as nonzero.  `J₋₁/₂(π/2)` (the limit of
/// `besselj(−1/2, elliptic_e(x))` at 0) is 0, and `(25 − y/5)/J₋₁/₂(π/2)`
/// was returned as a finite limit; `min(0, Catalan − 1/6, bernoulli(1/2))`
/// is 0, and `1/min(0, …)` was the two-sided limit of `1/min(x, Catalan −
/// 1/6, bernoulli(1/2))`.  Now they are evaluated: the first is refused,
/// the second has the one-sided limits `±∞` and no two-sided one.
///
/// Oracles: `J₋₁/₂(z) = √(2/(πz))·cos z` (DLMF 10.16.1), 0 at `z = π/2`;
/// mpmath `catalan − 1/6` = `0.7493`, SymPy `bernoulli(1/2).evalf()` =
/// `0.7302` (both positive, so the minimum is `x` near 0).
#[test]
fn a_function_value_at_a_zero_is_not_taken_for_nonzero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx
        .parse("(25 - y/5)/besselj(-1/2, elliptic_e(x))")
        .unwrap();
    match f.try_limit(&x, &ctx.int(0)) {
        Err(_) => {}
        Ok(l) => assert!(l.to_string().contains("oo"), "limit = {l}"),
    }
    let g = ctx
        .parse("1/Min(x, Catalan - 1/6, bernoulli(1/2))")
        .unwrap();
    let zero = ctx.int(0);
    assert_eq!(
        g.try_limit_dir(&x, &zero, Direction::Right).unwrap(),
        ctx.infinity()
    );
    assert_eq!(
        g.try_limit_dir(&x, &zero, Direction::Left).unwrap(),
        ctx.neg_infinity()
    );
    assert!(g.try_limit_dir(&x, &zero, Direction::Both).is_err());
}

/// A coefficient below the reach of the deep zero search was taken for 0:
/// the series of `coth(x + 1385)` at 0 was the constant `coth(1385)`, its
/// coefficients `−csch²(1385) ≈ −4·10⁻¹²⁰³`, … dropped (golden case 86000,
/// `coth(x + euler_number(8))`).  Now the expansion is refused.
///
/// Oracle: mpmath at 1,300 digits, `−1/sinh(1385)²` =
/// `−4.0396628·10⁻¹²⁰³`.
#[test]
fn a_coefficient_below_the_deep_search_is_not_dropped() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("coth(x + 1385)").unwrap();
    match f.try_series(&x, &ctx.int(0), 4) {
        Err(_) => {}
        Ok(s) => assert!(s.to_string().contains('x'), "series = {s}"),
    }
}

/// The `fuzz_calculus` timeout after the three-valued zero tests (0.49):
/// at `x = −1/3` Gruntz's exponential form of `√(tanh x)` leaves the
/// constant term `√((e^{−2/3} − 1)/(e^{−2/3} + 1)) − √(−tanh(1/3))`, a 0
/// that no exact test proved.  The undecided coefficient cut the short
/// expansion, so a ten-term one followed, whose coefficients (sums of huge
/// constants) were each evaluated: over 100 s in release (0.05 s at
/// 0.48).  Now the two radicands are shown equal (a rational function of
/// exponentials), and an expansion cut by an undecided coefficient is not
/// widened — the second limit, whose zero `K₁/₂(1) − √(π/2)·e⁻¹` is
/// not proved, is refused at once.  Bounds for a debug build under load
/// (release: 0.05 s and 0.2 s).
///
/// Oracle: mpmath `diff(lambda x: sqrt(tanh(x)), -1/3)` =
/// `−0.790649435247292391869…·i` (the principal branch; both one-sided
/// quotients at `10⁻¹²` agree).
#[test]
fn a_radical_spelled_two_ways_does_not_widen_the_expansion() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx
        .parse("(sqrt(tanh(x)) - sqrt(tanh(-1/3)))/(x + 1/3)")
        .unwrap();
    let t = Instant::now();
    let l = f.try_limit(&x, &ctx.rational(-1, 3)).unwrap();
    assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
    let z = l.eval_complex64().unwrap();
    assert!(
        z.re.abs() < 1e-12 && (z.im + 0.790_649_435_247_292_4).abs() < 1e-12,
        "{l} = {z}"
    );
    let g = ctx
        .parse("(sqrt(x + besselk(1/2, 1)) - sqrt(sqrt(pi/2)*exp(-1)))/x")
        .unwrap();
    let t = Instant::now();
    let _ = g.try_limit(&x, &ctx.int(0));
    assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
}

/// `Quantity::is_zero`/`equals` return `bool` and mean "proved": an
/// undecided answer is `false`, which reads as "not zero".  The
/// three-valued `try_is_zero`/`try_equals` (new) keep the third case.
///
/// Oracle: `K_{1/2}(z) = √(π/(2z))·e^{−z}` (DLMF 10.39.2), so `k =
/// K_{1/2}(1) − √(π/2)·e^{−1}` is 0, which `Ex::is_zero` does not decide
/// (`None`).  (Up to 0.49 the undecided zero here was `Γ(1/3)Γ(2/3) −
/// 2π/√3`, which `is_zero` now proves by the reflection formula.)
#[test]
fn quantity_zero_tests_say_proved_or_undecided() {
    use symplex::units::*;
    let ctx = Context::new();
    let k = ctx.parse("besselk(1/2, 1) - sqrt(pi/2)*exp(-1)").unwrap();
    assert_eq!(k.is_zero(), None);
    let q = Length::from_ex(k.clone());
    assert!(!q.is_zero());
    assert_eq!(q.try_is_zero(), None);

    let one = Length::from_ex(ctx.int(1));
    assert!(!one.is_zero());
    assert_eq!(one.try_is_zero(), Some(false));
    assert!(Length::from_ex(ctx.int(0)).is_zero());
    assert_eq!(Length::from_ex(ctx.int(0)).try_is_zero(), Some(true));

    let r = Length::from_ex(ctx.parse("besselk(1/2, 1)").unwrap());
    let s = Length::from_ex(ctx.parse("sqrt(pi/2)*exp(-1)").unwrap());
    assert!(!r.equals(&s));
    assert_eq!(r.try_equals(&s), None);
    assert_eq!(one.try_equals(&one), Some(true));
    assert_eq!(one.try_equals(&s), Some(false));
}
