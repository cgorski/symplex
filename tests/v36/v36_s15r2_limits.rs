//! Tests of session 17, wave 16 (s15r2): limits, singularities and root searches; slow limits and series.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

use std::time::Duration;

use crate::perf_assert::assert_fast;

/// `singularities(1/den, x, [lo, hi])`, displayed.
fn poles(den: &str, lo: i64, hi: i64) -> Result<String, SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _y = ctx.symbol("y");
    let dom = ctx.interval(&ctx.int(lo), &ctx.int(hi), IntervalKind::Closed);
    let f = 1 / ctx.parse(den).unwrap();
    f.singularities(&x, Some(&dom)).map(|s| s.to_string())
}

/// `limit(src, x, point)` from `dir`, displayed.
fn lim_at(src: &str, point: &str, dir: Direction) -> Result<String, SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _y = ctx.symbol("y");
    let e = ctx.parse(src).unwrap();
    let p = ctx.parse(point).unwrap();
    e.try_limit_dir(&x, &p, dir).map(|v| v.to_string())
}

/// `limit(src, x, 0)` from `dir`, displayed.
fn lim(src: &str, dir: Direction) -> Result<String, SymplexError> {
    lim_at(src, "0", dir)
}

/// `series(src, x, point, order)`, displayed (a formal `Series(…)` when
/// there is no expansion).
fn ser_at(src: &str, point: &str, order: u32) -> String {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _y = ctx.symbol("y");
    let e = ctx.parse(src).unwrap();
    let p = ctx.parse(point).unwrap();
    e.series(&x, &p, order).to_string()
}

// ── (a) R2.1: families with a complex offset and step ──────────────────────

/// The general solution of `(−2)^x = 4` is `x = (ln 4 + 2πin)/(ln 2 + iπ)`,
/// real at `n = 1` only.  `enumerate_family` dropped the family because its
/// offset (`n = 0`) is not real: `singularities(1/((−2)^x − 4), x, [0, 10])`
/// was `∅`.  Oracle: exact substitution, `(−2)² − 4 = 0` (mpmath
/// `power(-2, 2) - 4 = 0.0` at 30 and 60 digits); `(−2)^(x+1) = 4` at 1,
/// `(−2)³ = −8`, `(−2)^(x/2) = 4` at 4 (was `∅` each), `(−2)^(x/2) = −2` at 2
/// (was refused: "cannot locate the members … numerically").
#[test]
fn a_complex_family_reaches_the_real_axis_at_one_index() {
    assert_eq!(poles("(-2)^x - 4", 0, 10).unwrap(), "{2}");
    assert_eq!(poles("(-2)^(x + 1) - 4", 0, 10).unwrap(), "{1}");
    assert_eq!(poles("(-2)^x + 8", 0, 10).unwrap(), "{3}");
    assert_eq!(poles("(-2)^(3*x) - 64", 0, 10).unwrap(), "{2}");
    assert_eq!(poles("(-2)^(x/2) - 4", 0, 10).unwrap(), "{4}");
    assert_eq!(poles("(-2)^(x/2) + 2", 0, 10).unwrap(), "{2}");
    // the real member outside the window, or at its end
    assert_eq!(poles("(-2)^x - 4", 3, 10).unwrap(), "EmptySet");
    assert_eq!(poles("(-2)^x - 4", 2, 10).unwrap(), "{2}");
    assert_eq!(poles("(-2)^x - 16", 0, 3).unwrap(), "EmptySet");
    // a negative member: (−1/2)^(−2) = 4
    assert_eq!(poles("(-1/2)^x - 4", -10, 10).unwrap(), "{-2}");
}

/// The same for other complex bases and exponents: `(1 + i)^x = 2i` at 2
/// (the family's offset is the real member; it was refused, "cannot decide
/// whether the family … has real members"), `(2i)^x = −4` at 2,
/// `e^{(1+i)x} = e^{2π}` at `2π`, `(−2)^(πx) = 4` at `2/π`,
/// `e^{x(1 + iπ/3)} = −e³` at 3 (each was `∅`).  Oracle: exact substitution
/// (mpmath: `power(mpc(1, 1), 2) - 2j = 0`, `power(mpc(0, 2), 2) + 4 = 0`,
/// `exp((1+1j)*2*pi) - exp(2*pi) = −1.2e-58j` at 60 digits, `power(-2,
/// pi*(2/pi)) - 4 = 0`).
#[test]
fn complex_bases_and_rates() {
    assert_eq!(poles("(1 + I)^x - 2*I", 0, 10).unwrap(), "{2}");
    assert_eq!(poles("(2*I)^x + 4", 0, 10).unwrap(), "{2}");
    assert_eq!(
        poles("exp((1 + I)*x) - exp(2*pi)", 0, 10).unwrap(),
        "{2*pi}"
    );
    assert_eq!(poles("(-2)^(x*pi) - 4", 0, 10).unwrap(), "{2/pi}");
    assert_eq!(poles("exp(x*(1 + I*pi/3)) + exp(3)", 0, 10).unwrap(), "{3}");
    assert_eq!(poles("(-3)^x - 9", 0, 10).unwrap(), "{2}");
}

/// A family that never reaches the real axis stays empty: `|(−2)^x| = 3`
/// only at `x = log₂ 3`, where `(−2)^x − 3 = −2.2087 − 2.8938i` (mpmath
/// `power(-2, log(3)/log(2)) - 3`); `|(−8)^x| = 4` only at `2/3`, where
/// `(−8)^(2/3) − 4 = −6 + 3.4641i`.  (Unchanged; the index `−im a/im b` is
/// not an integer.)
#[test]
fn a_complex_family_off_the_axis_has_no_member() {
    assert_eq!(poles("(-2)^x - 3", 0, 10).unwrap(), "EmptySet");
    assert_eq!(poles("(-8)^x - 4", 0, 10).unwrap(), "EmptySet");
    assert_eq!(poles("(-8)^x + 2", 0, 10).unwrap(), "EmptySet");
}

/// Families with a real step are enumerated as before, and one whose step
/// holds a parameter is refused, not dropped.
#[test]
fn real_families_are_unchanged() {
    assert_eq!(poles("(-1)^x - 1", 0, 10).unwrap(), "{0, 10, 2, 4, 6, 8}");
    assert_eq!(poles("I^x + 1", 0, 10).unwrap(), "{10, 2, 6}");
    assert_eq!(poles("exp(I*x) - 1", 0, 10).unwrap(), "{0, 2*pi}");
    assert_eq!(poles("cos(x)", 0, 10).unwrap(), "{1/2*pi, 3/2*pi, 5/2*pi}");
    assert_eq!(poles("sin(x) - 2", 0, 10).unwrap(), "EmptySet");
    assert!(poles("(-2)^x - y", 0, 10).is_err());
}

// ── (b) R2.2: γ(s, 0) = 0 only for re s > 0 ────────────────────────────────

/// `eval` folded `γ(s, 0) = 0` for every `s` but a number `≤ 0`, and the
/// table had no pole place at `z = 0`: `limit(lowergamma(y, x), x, 0)` was
/// `Piecewise((0, y ≠ 0))`, `limit(lowergamma(−√2, x), x, 0⁺)` and
/// `limit(lowergamma(i, x), x, 0⁺)` were 0, `limit(lowergamma(−1/2, x), x,
/// 0)` was the "value" `lowergamma(−1/2, 0)`, and `√x·γ(−1/2, x)` tended to
/// 0.  `γ(s, x) ~ xˢ/s` (DLMF 8.7.1).  Oracle: mpmath `gammainc(-0.5, 0,
/// 1e-20) = −2.0e10`, `gammainc(-0.5, 0, 1e-40) = −2.0e20` (so `√x·γ → −2`),
/// `gammainc(-sqrt(2), 0, 1e-40) = −2.6e56`, `gammainc(1j, 0, x)` = `−0.878 +
/// 0.478i` at `1e-20` and `0.840 + 0.543i` at `1e-40` (no limit).  The
/// engines have no expansion of `γ(s, x)` at 0, so these are refused.
#[test]
fn lowergamma_at_zero_is_zero_only_for_a_positive_real_part() {
    for src in [
        "lowergamma(y, x)",
        "lowergamma(-1/2, x)",
        "lowergamma(-sqrt(2), x)",
        "lowergamma(I, x)",
        "sqrt(x)*lowergamma(-1/2, x)",
        "x*lowergamma(y, x)",
        "exp(lowergamma(y, x^2))",
    ] {
        assert!(lim(src, Direction::Right).is_err(), "{src}");
        assert!(lim(src, Direction::Both).is_err(), "{src}");
    }
    let ctx = Context::new();
    let y = ctx.symbol("y");
    for (src, folded) in [
        ("lowergamma(y, 0)", false),
        ("lowergamma(-sqrt(2), 0)", false),
        ("lowergamma(I, 0)", false),
        ("lowergamma(pi - 4, 0)", false),
        ("lowergamma(sqrt(2), 0)", true),
        ("lowergamma(1 + I, 0)", true),
        ("lowergamma(pi - 3, 0)", true),
        ("lowergamma(5/2, 0)", true),
    ] {
        let v = ctx.parse(src).unwrap().eval();
        assert_eq!(v == ctx.int(0), folded, "{src}: {v}");
    }
    assert_eq!(y.is_zero(), None);
}

/// Where `re s > 0` is known the limit is still 0 (mpmath `gammainc(sqrt(2),
/// 0, 1e-40) = 1.9e-57`, `gammainc(1+1j, 0, 1e-40) = (1.5 + 6.9i)e-41`), also
/// for an order that varies with `x` (`γ` is continuous in both arguments at
/// `(s₀, 0)`, `s₀ > 0`), and for a symbol declared positive.
#[test]
fn lowergamma_at_zero_with_a_positive_order() {
    for src in [
        "lowergamma(sqrt(2), x)",
        "lowergamma(1 + I, x)",
        "lowergamma(1/3, x)",
        "lowergamma(3, x)",
        "lowergamma(abs(y) + 1, x)",
        "lowergamma(x + 1, x)",
    ] {
        assert_eq!(lim(src, Direction::Both).unwrap(), "0", "{src}");
    }
    assert_eq!(lim("uppergamma(x + 1, x)", Direction::Both).unwrap(), "1");
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    assert_eq!(ctx.int(0).lowergamma(&p).eval(), ctx.int(0));
    let v = x.lowergamma(&p).try_limit(&x, &ctx.int(0)).unwrap();
    assert_eq!(v, ctx.int(0));
}

// ── (c) R2.3: the finite part next to an infinity in a parametric direction ─

/// `Γ(0, x) = E₁(x) → +∞ − iπ` from the left of 0 and `+∞` from the right.
/// With a factor of unknown direction the Gruntz algorithm's `sign(y −
/// 1)·∞` was the value on both sides, and the two-sided limit was claimed:
/// for `y = 2` that reads `∞`, while `limit(Γ(0, x), x, 0)` is refused (the
/// sides differ by `−iπ`).  Oracle: mpmath `(y − 1)·gammainc(0, −1e-40)` =
/// `−27.4579 + 0.942478i` for `y = 7/10` (`−(y − 1)π = 0.942478`), `91.5262
/// − 3.14159i` for `y = 2`, `3.14159 + 91.5262i` for `y = 1 + i`; from the
/// right the imaginary parts are 0.
#[test]
fn a_parametric_factor_keeps_the_finite_part_of_an_infinity() {
    let f = "(y - 1)*uppergamma(0, x)";
    assert_eq!(lim(f, Direction::Right).unwrap(), "sign(y - 1)*oo");
    assert_eq!(
        lim(f, Direction::Left).unwrap(),
        "-(y - 1)*pi*I + sign(y - 1)*oo"
    );
    assert!(lim(f, Direction::Both).is_err());
    // y·ln x → −sign(y)·∞ + iπy from the left (mpmath `y*log(-1e-40)`:
    // imaginary part `πy`), y·Ci(x) likewise (`ci(-x) = ci(x) + iπ`).
    assert_eq!(
        lim("y*ln(x)", Direction::Left).unwrap(),
        "y*pi*I - sign(y)*oo"
    );
    assert_eq!(lim("y*ln(x)", Direction::Right).unwrap(), "-sign(y)*oo");
    assert!(lim("y*ln(x)", Direction::Both).is_err());
    assert!(lim("y*Ci(x)", Direction::Both).is_err());
    // a sum, where no factor can be taken out: still no two-sided claim
    assert!(lim("(y - 1)*uppergamma(0, x) + x", Direction::Both).is_err());
    // the numeric factors, as before
    assert_eq!(
        lim("2*uppergamma(0, x)", Direction::Left).unwrap(),
        "-2*pi*I + oo"
    );
    // real on both sides: the two-sided limit stays
    assert_eq!(lim("y/x^2", Direction::Both).unwrap(), "sign(y)*oo");
    assert_eq!(
        lim("(y - 1)*ln(x^2)", Direction::Both).unwrap(),
        "sign(-2*y + 2)*oo"
    );
}

/// `γ(−1, 1)` has no value (`γ(s, 1)` has a pole at `s = −1`: mpmath
/// `gammainc(-1 + 1e-20, 0, 1) = −1.0e20`, `gammainc(-1 - 1e-20, 0, 1) =
/// +1.0e20`), but the constant was returned as a limit value:
/// `limit(lowergamma(−1, 1) + x, x, 0)` was `lowergamma(−1, 1)`.
#[test]
fn lowergamma_at_a_pole_of_its_order_is_no_limit_value() {
    assert!(lim("lowergamma(-1, 1) + x", Direction::Both).is_err());
    assert!(lim("lowergamma(-2, 3)*cos(x)", Direction::Both).is_err());
    assert!(lim("lowergamma(0, exp(x))", Direction::Both).is_err());
    // a finite constant is still one: mpmath gammainc(-0.5, 0, 1) = −3.72305…
    assert_eq!(
        lim("lowergamma(-1/2, exp(x))", Direction::Both).unwrap(),
        "lowergamma(-1/2, 1)"
    );
}

// ── (d) R2.5: a sum of step parts that vanishes on a region; b^x, b < 0 ────

/// `U₃(t) = 8t³ − 4t` at `t = max(0, y)` is 0 for every `y ≤ 0`: a sum that
/// vanishes on a region of the parameter although none of its factors
/// does, which the zero test read as "nonzero for generic parameters".
/// Golden case 120222: `limit(B(U₃(max(x, −4/5, y)), Ai′(x − y)), x, 0)` was
/// `Piecewise((B(8·max(0, y)³ − 4·max(0, y), Ai′(−y)), y ≠ 0))`, the value of
/// `B` at its pole `B(0, ·)` for every `y < 0`.  Oracle: mpmath at `y =
/// −13/10`, `beta(8x³ − 4x, airyai(1.3 + x, derivative=1))` = `−2.5e19` at
/// `x = 1e-20` and `+2.5e19` at `x = −1e-20` (`B(a, b) ~ 1/a`).  Refused now.
#[test]
fn a_sum_of_step_parts_vanishing_on_a_region_is_a_possible_pole() {
    for src in [
        "beta(chebyshev_u(3, Max(x, -4/5, y)), airyaiprime(x - y))",
        "gamma(8*Max(x, y)^3 - 4*Max(x, y))",
        "gamma(Heaviside(y) - 1 + x)",
        "gamma(y - Max(x, y))",
        "x/(Heaviside(y) - 1 + x)",
        "x/(Max(1, y) - 1 + x)",
        "1/(Max(0, y)^2 + Max(0, y) + x)",
    ] {
        assert!(lim(src, Direction::Both).is_err(), "{src}");
    }
    // sums of step parts that vanish nowhere keep their limits
    assert_eq!(lim("x/(Max(1, y) + x)", Direction::Both).unwrap(), "0");
    assert_eq!(
        lim("gamma(1 + Max(x, y))", Direction::Both).unwrap(),
        "Gamma(max(0, y) + 1)"
    );
}

/// The roots `Ex::solve` gives for `src = 0`, `x` declared real.
fn real_roots(src: &str) -> Result<Vec<String>, SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let e = ctx.parse(src).unwrap();
    e.solve(&x)
        .map(|v| v.iter().map(|r| r.to_string()).collect())
}

/// A polynomial in `b^x` with a negative base was refused ("not polynomial
/// … transcendental solver could not find solutions"): the substitution `t
/// = b^x` wanted a positive rational base.  `(−2)^(2x) − 6·(−2)^x + 8 = (t −
/// 2)(t − 4)`: `(−2)^x = 4` at `x = 2`, `(−2)^x = 2` at no real `x`
/// (`|(−2)^x| = 2` at `x = 1`, where it is `−2`).  Oracle: exact
/// substitution (`16 − 24 + 8 = 0`; at `x = 1`: `4 + 12 + 8 = 24`); `t² − 2t
/// − 8 = (t − 4)(t + 2)`: `x = 2` and `x = 1` (`4 + 4 − 8 = 0`); `t² + 6t + 8
/// = (t + 2)(t + 4)`: `x = 1` only (`4 − 12 + 8 = 0`; `(−2)^x = −4` has no
/// real root).
#[test]
fn quadratics_in_a_power_with_a_negative_base() {
    assert_eq!(real_roots("(-2)^(2*x) - 6*(-2)^x + 8").unwrap(), ["2"]);
    assert_eq!(real_roots("(-2)^(2*x) - 2*(-2)^x - 8").unwrap(), ["2", "1"]);
    assert_eq!(real_roots("(-2)^(2*x) + 6*(-2)^x + 8").unwrap(), ["1"]);
    assert_eq!(real_roots("(-3)^(2*x) - 12*(-3)^x + 27").unwrap(), ["2"]);
    assert_eq!(real_roots("(-2)^(x + 1) + (-2)^x + 4").unwrap(), ["2"]);
    // other bases that are no powers of an integer: t = (2/3)^x ∈ {2, 3}
    assert_eq!(
        real_roots("(2/3)^(2*x) - 5*(2/3)^x + 6").unwrap(),
        ["ln(3)/ln(2/3)", "ln(2)/ln(2/3)"]
    );
    // positive integer bases as before
    assert_eq!(real_roots("4^x - 6*2^x + 8").unwrap(), ["2", "1"]);
    // and the pole search sees them: (−1)^x ∈ {1, 2} at the even integers
    assert_eq!(poles("(-2)^(2*x) - 6*(-2)^x + 8", 0, 10).unwrap(), "{2}");
    assert_eq!(poles("(-2)^(2*x) - 2*(-2)^x - 8", 0, 10).unwrap(), "{1, 2}");
    assert_eq!(
        poles("(-1)^(2*x) - 3*(-1)^x + 2", 0, 5).unwrap(),
        "{0, 2, 4}"
    );
}

// ── (e) B-C06-02: a function discontinuous at the point ────────────────────

/// `KroneckerDelta(1, x + 1)` is 1 at `x = 0` and 0 at every other `x`: it
/// has no expansion at 0 (the series is refused, as for the spelling
/// `KroneckerDelta(0, x)`; roadmap B-C06-02 reported `1`, fixed before
/// 0.58), and its limit there is 0.  The limit substituted the point:
/// `limit(KroneckerDelta(1, x + 1), x, 0)` was 1, `limit(exp(KroneckerDelta(1,
/// x + 1)), x, 0)` was `e`, `limit(x + KroneckerDelta(0, x), x, 0)` was 1,
/// `limit(KroneckerDelta(x, 3), x, 3)` was 1.  Oracle: the definition (`1 ≠
/// 1 + x` for every `x ≠ 0`: the function is 0 at `x = ±1e-20`).
#[test]
fn kronecker_delta_at_the_point_where_its_arguments_meet() {
    for src in [
        "KroneckerDelta(1, x + 1)",
        "KroneckerDelta(0, x)",
        "KroneckerDelta(x, 0)",
        "KroneckerDelta(x^2, 0)",
        "KroneckerDelta(1, cos(x))",
        "x + KroneckerDelta(1, x + 1)",
        "sin(x)*KroneckerDelta(1, x + 1)/x",
    ] {
        assert_eq!(lim(src, Direction::Both).unwrap(), "0", "{src}");
        assert_eq!(lim(src, Direction::Right).unwrap(), "0", "{src}");
        assert!(ser_at(src, "0", 4).starts_with("Series("), "{src}");
    }
    assert_eq!(
        lim("exp(KroneckerDelta(1, x + 1))", Direction::Both).unwrap(),
        "1"
    );
    assert_eq!(
        lim_at("KroneckerDelta(x, 3)", "3", Direction::Both).unwrap(),
        "0"
    );
    assert_eq!(
        lim_at("KroneckerDelta(0, 1/x)", "oo", Direction::Both).unwrap(),
        "0"
    );
    // away from the meeting point the value is the limit, and the series
    assert_eq!(
        lim("KroneckerDelta(2, x + 1)", Direction::Both).unwrap(),
        "0"
    );
    assert_eq!(ser_at("KroneckerDelta(2, x + 1)", "0", 4), "0");
    // a parameter: 0 for every y (at y = 0 too, where the old value was 1)
    assert_eq!(
        lim("KroneckerDelta(y, x)", Direction::Both).unwrap(),
        "Piecewise(0 if y == 0, KroneckerDelta(0, y) if y != 0)"
    );
}

// ── (f) R2.4: a function without an expansion and with a formal derivative ──

/// `B(a, b)` has no expansion rule and its derivative stays formal, so a
/// series holding it is refused; but only after the limits of the
/// function, the limits of `x·f` (to count the useful pole retries) and
/// the retries themselves: `series(B(x, x)/(x + 5) − sign(Li₂(3)), x, 0, 4)`
/// took 0.58 s to refuse in a release build, `−2/C(B(x, x), 3/2)` 1.17 s,
/// `atan2(B(x, √2/x), x·ln(−2))` 0.37 s, `B(Cx, asech x)` 0.36 s, `B(x²,
/// i)/x` 0.23 s, and `γ(−1, x + γ) − x − 1` (no value at any `x`) 0.37 s
/// (golden seeds 7784, 4382, 7181, 1536, 6915, 1965: +12 % series CPU since
/// 0.57).  Now the formal first derivative ends the fallback before any
/// limit is taken (under 1 ms each).  The answers are the same refusals.
#[test]
fn a_series_with_a_formal_first_derivative_is_refused_at_once() {
    for src in [
        "beta(x, x)/(5 + x) - sign(polylog(2, 3))",
        "-2/binomial(beta(x, x), 3/2)",
        "atan2(beta(x, sqrt(2)/x), x*ln(-2))",
        "beta(x*Catalan, asech(x))",
        "beta(x^2, I)/x",
        "lowergamma(-1, EulerGamma + x) - (x + 1)",
    ] {
        let s = assert_fast!(
            Duration::from_secs(1),
            ser_at(src, "0", 4),
            "series of {src}"
        );
        assert!(s.starts_with("Series("), "{src}: {s}");
    }
    // a pole that the retry does expand is still expanded
    assert_eq!(ser_at("1/(x*cos(x))", "0", 4), "5/24*x^3 + 1/x + 1/2*x");
}

// ── (g) R6.3, (h) R6.4: slow inputs of the nightly fuzz runs ───────────────

/// `fuzz_calculus` input `bc61d72d0ea7`: `limit(|(x − 1)^tanh(x − 1)|^(x − 1),
/// x, 1, '−')` was refused after 1.2 s in a release build (5.6 s in the fuzz
/// build, a timeout under the sanitizer): `leadterm_add_by_series`
/// multiplied out truncated series with only their tree size charged to the
/// budget.  Every `expand` there is now charged by the terms it distributes
/// (0.05 s).  The limit is 1 (mpmath: `0.99999999999999999977` at `x = 1 −
/// 1e-10`, `1.0` at `1 − 1e-20`); the engine still refuses it.
#[test]
fn a_cancelling_leading_term_is_refused_within_the_budget() {
    let r = assert_fast!(
        Duration::from_secs(5),
        lim_at("abs((x - 1)^tanh(x - 1))^(x - 1)", "1", Direction::Left),
        "the limit of fuzz input bc61d72d0ea7"
    );
    if let Ok(v) = r {
        assert_eq!(v, "1");
    }
}

/// `fuzz_calculus` input `c9feedf1241f`: `series(atanh(π^x + x^(3/2)), x, 1,
/// 4)` takes 1.8 s in a release build (6.7 s in the fuzz build): composing
/// the Taylor series of `atanh` at `π + 1` with the inner series decides
/// every partial sum and every coefficient of every power by a certified
/// evaluation.  Deciding once per exponent brought it to 0.25 s but left
/// cancelling sums in the powers (`limit(Chi(acsch(π − x)), x, ∞)` went from
/// 0.45 s to 7.7 s), so that change is not in the tree (R6.4 is open): this
/// test pins the value only.  Oracle: mpmath
/// `taylor(lambda x: atanh(pi**x + x**1.5), 1, 3)` = `[0.2463165002033 −
/// 1.5707963267949i, −0.3155043241305, 0.2616186995171, −0.2002291540915]`,
/// whose value at `x = 1.1` is `0.217182025631332 − 1.570796326794897i`.
#[test]
fn the_series_of_a_composition_at_a_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("atanh(pi^x + x^(3/2))").unwrap();
    let s = f.series(&x, &ctx.int(1), 4);
    assert!(!s.has_unevaluated(), "{s}");
    let v = s.subs(&x, &ctx.rational(11, 10)).eval_complex64().unwrap();
    assert!((v.re - 0.217182025631332).abs() < 1e-12, "{v}");
    assert!((v.im + std::f64::consts::FRAC_PI_2).abs() < 1e-12, "{v}");
}

// ── two fixes found in the golden diff of this wave ────────────────────────

/// Golden case 51383: `x²·E₁(KroneckerDelta(x + 1, −1))` is `x²·E₁(0)` at
/// every `x ≠ −2`, and `E₁(0)` is infinite (mpmath `e1(0) = +inf`): there is
/// no finite limit at 0.  0.58 refused it; the first version of this wave's
/// rewrite of the delta (replaced by its eventual value 0, the product then
/// read by its leading term) returned `0`.  `gruntz::rewrite_tractable` now
/// refuses an expression that the replacement leaves at a pole: the
/// two-sided limit and the limit from the left are refused, as in 0.58.
/// (The limit from the right is `0` in 0.58 and here — the product rule
/// takes the constant `E₁(0)` for a finite factor, as it does for `x²·E₁(0)`
/// itself: an open wrong answer, not asserted.)
#[test]
fn a_kronecker_delta_at_its_eventual_value_may_leave_a_pole() {
    let f = "x^2*E1(KroneckerDelta(x + 1, -1))";
    assert!(lim(f, Direction::Both).is_err());
    assert!(lim(f, Direction::Left).is_err());
    // where the eventual value is regular the limit is taken: 0·E₁(1)
    assert_eq!(
        lim("x^2*E1(1 + KroneckerDelta(x + 1, -1))", Direction::Both).unwrap(),
        "0"
    );
}

/// Golden cases 36758 and 92545: `max(x, y, √2) ≥ √2` is never 0, so
/// `limit(csc(max(x, y, √2)), x, 0)` is `1/sin(max(0, y, √2))` for every `y`
/// (0.58), and `limit(B(π, S(max(e, x, y))), x, 0)` is `B(π, S(max(e, 0,
/// y)))`.  The first version of this wave's step-value test tried every
/// argument of the `max` as its value, the dominated constant 0 too (`sin 0
/// = 0`, `S(0) = 0`: a pole), and refused both.  Oracle: mpmath
/// `csc(max(x, y, sqrt(2)))` at `x = ±1e-20` = `1.0123855798601852666` for
/// `y = 7/10` (`csc(sqrt(2))`) and `1.0997501702946164668` for `y = 2`
/// (`1/sin(2)`); `beta(pi, fresnels(max(e, x, y)))` = `1.2776062101785174419`
/// for `y = 7/10` (`beta(pi, fresnels(e))`), `1.0524622740577137559` for `y =
/// 3`.
#[test]
fn a_dominated_constant_is_never_the_value_of_a_max() {
    assert_eq!(
        lim("csc(Max(x, y, sqrt(2)))", Direction::Both).unwrap(),
        "1/sin(max(0, y, sqrt(2)))"
    );
    assert_eq!(
        lim("beta(pi, fresnels(Max(E, x, y)))", Direction::Both).unwrap(),
        "B(pi, fresnels(max(E, 0, y)))"
    );
    // a constant that can be the value still makes a possible pole:
    // max(0, y) = 0 for every y ≤ 0
    assert!(lim("csc(Max(x, y))", Direction::Both).is_err());
}
