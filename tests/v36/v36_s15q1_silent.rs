//! Tests of session 15 (s15q1): limits by substitution at a pole, solve missing roots.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

/// `limit(src, x, 0)` from `dir`, displayed.
fn lim(src: &str, dir: Direction) -> Result<String, SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _y = ctx.symbol("y");
    let e = ctx.parse(src).unwrap();
    e.try_limit_dir(&x, &ctx.int(0), dir).map(|v| v.to_string())
}

/// `∫_lo^hi dx / den` for a real `x`.
fn integral_of_reciprocal(den: &str, lo: &str, hi: &str) -> Result<Ex, SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let f = 1 / ctx.parse(den).unwrap();
    f.try_integrate_definite(&x, &ctx.parse(lo).unwrap(), &ctx.parse(hi).unwrap())
}

/// The roots `Ex::solve` gives for `src = 0`, `x` declared real.
fn real_roots(src: &str) -> Result<Vec<String>, SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let e = ctx.parse(src).unwrap();
    e.solve(&x)
        .map(|v| v.iter().map(|r| r.to_string()).collect())
}

// ── limits: direct substitution never stops at a pole of the table ─────────

/// `B(x, y) = Γ(x)Γ(y)/Γ(x + y) ~ 1/x` at 0, a pole for every `y`: the
/// limit was the value `B(0, y)` (session 15 P2, ~420 golden pairs).  Now
/// the substitution refuses and the engine decides: `B(x, 1) = 1/x` is `+∞`
/// from the right and `−∞` from the left, and with a plain (complex) `y` the
/// direction's imaginary part is not decided, so the limit is refused.
/// Oracle: mpmath `beta(±1e-30, 1) = ±1e30`.
#[test]
fn beta_at_a_pole_is_no_value() {
    assert!(lim("beta(x, y)", Direction::Both).is_err());
    assert!(lim("beta(y, x)", Direction::Right).is_err());
    assert_eq!(lim("beta(x, 1)", Direction::Right).unwrap(), "oo");
    assert_eq!(lim("beta(x, 1)", Direction::Left).unwrap(), "-oo");
    assert_eq!(lim("beta(x, x)", Direction::Right).unwrap(), "oo");
}

/// `0·B(0, y)` folded to 0: `x·B(x, 3x)` was 0 at `x → 0` (the limit is
/// `4/3`, `B(x, 3x) ~ 4/(3x)`), `x·B(erf x, γ)` was 0 (`√π/2`), `x·B(e, x)`
/// was 0 (1).  Oracle: mpmath at `x = 1e-30`: `x·beta(x, 3x) =
/// 1.3333333333333333333`, `x·beta(erf x, euler) = 0.88622692545275801365`
/// (`√π/2`), `x·beta(e, x) = 1`.
#[test]
fn zero_times_a_beta_pole_is_not_zero() {
    assert_eq!(lim("x*beta(x, 3*x)", Direction::Both).unwrap(), "4/3");
    assert_eq!(
        lim("x*beta(erf(x), EulerGamma)", Direction::Both).unwrap(),
        "1/2*sqrt(pi)"
    );
    assert_eq!(lim("x*beta(E, x)", Direction::Both).unwrap(), "1");
}

/// `B(x − 1, 1 + 2x)` at 0 was `B(−1, 1)` (whose poles cancel to `−1` along
/// `a + b = 0`); along this path the limit is `−3`.  Oracle: mpmath
/// `beta(−1 ± 1e-30, 1 ± 2e-30) = −3.0000000000000000000000000000`.
#[test]
fn beta_where_the_poles_cancel_depends_on_the_path() {
    assert_eq!(lim("beta(x - 1, 1 + 2*x)", Direction::Both).unwrap(), "-3");
    assert_eq!(lim("beta(x - 1, 1)", Direction::Both).unwrap(), "-1");
}

/// The other functions with poles in their table: `γ(x, 1)` at 0 was
/// `γ(0, 1)` (`γ(s, 1) ~ 1/s`: mpmath `gammainc(1e-30, 0, 1) = 1.0e30`),
/// `K_y(x)` was `K_y(0)` (infinite for every `y`: `besselk(0.7, 1e-30) =
/// 1.05e21`), `Γ(y, x)` was `Γ(y)` and `Li_y(1 + x)` was `ζ(y)` (true only
/// for `re y > 0`, `re y > 1`), `ψ^(y)(x)` was `ψ^(y)(0)`.  Now refused.
#[test]
fn table_functions_at_their_poles_are_refused() {
    for src in [
        "lowergamma(x, 1)",
        "besselk(y, x)",
        "uppergamma(y, x)",
        "polylog(y, 1 + x)",
        "polygamma(y, x)",
        "besselj(y, x)",
    ] {
        assert!(lim(src, Direction::Right).is_err(), "{src}");
    }
}

/// An argument that reaches the pole place for every value of the
/// parameter: `L_y(0) = 1` (`₁F₁(−y; 1; 0)`, DLMF 18.11.2), so `K₁(L_y(x) − 1)`
/// and `Γ(L_y(x) − 1)` at 0 were the infinite `K₁(L_y(0) − 1)` and `Γ(L_y(0)
/// − 1)` (by substitution and by the composition rule of `Γ`).  Now refused.
#[test]
fn arguments_identically_at_the_pole_place_are_refused() {
    assert!(lim("besselk(1, laguerre(y, x) - 1)", Direction::Both).is_err());
    assert!(lim("gamma(laguerre(y, x) - 1)", Direction::Both).is_err());
}

/// `Γ(−1/2, x¹²)` and `Γ(0, x²)` at 0 were `uppergamma(−1/2, 0)` and
/// `uppergamma(0, 0)`, values at the pole; the limits are `+∞`.  Oracle:
/// mpmath `gammainc(−1/2, 1e-360) = 2.0e180`, `gammainc(0, 1e-60) = 137.58`
/// (`−ln t − γ`).
#[test]
fn upper_gamma_at_zero_tends_to_infinity() {
    assert_eq!(
        lim("uppergamma(-1/2, x^12)", Direction::Both).unwrap(),
        "oo"
    );
    assert_eq!(lim("uppergamma(0, x^2)", Direction::Both).unwrap(), "oo");
}

/// What the table folds to a finite constant where only the slot at the
/// pole place varies is still a value by substitution: `J₁(0) = 0`,
/// `Li₂(1) = π²/6`, `Γ(2, 0) = 1`, `E₂(0) = 1` (unchanged).  Oracle: DLMF
/// 10.2.2, 25.12.1/25.6.1, 8.4.4, 8.19.6 (`E_p(0) = 1/(p − 1)`).
#[test]
fn finite_folds_at_a_pole_place_are_kept() {
    assert_eq!(lim("besselj(1, x)", Direction::Both).unwrap(), "0");
    assert_eq!(
        lim("polylog(2, 1 + x)", Direction::Both).unwrap(),
        "1/6*pi^2"
    );
    assert_eq!(lim("uppergamma(2, x)", Direction::Both).unwrap(), "1");
    assert_eq!(lim("expint(2, x)", Direction::Both).unwrap(), "1");
}

/// `J_x(0)` at `x = 0`: the folded `J₀(0) = 1` is no limit when the order
/// varies (`J_ν(0) = 0` for `ν > 0`, infinite for `ν < 0` not an integer):
/// the substitution took it.  Oracle: DLMF 10.2.2 (`J_ν(z) ~ (z/2)^ν/Γ(ν + 1)`).
#[test]
fn a_finite_fold_is_no_limit_when_another_slot_varies() {
    assert!(lim("besselj(x, 0)", Direction::Right).is_err());
}

/// The series of `ψ^(1/2)(x)` at 0 was a Taylor series with the constant
/// `polygamma(1/2, 0)`: for a non-integer order `ψ^(m)(x)` is a multiple of
/// `ζ(m + 1, x) ~ x^{−m−1}` (mpmath `zeta(3/2, 1e-6) = 1.0e9`).  Now refused.
#[test]
fn series_of_polygamma_at_its_pole_is_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.parse("polygamma(1/2, x)").unwrap();
    let s = e.series(&x, &ctx.int(0), 4);
    assert!(s.has_unevaluated(), "{s}");
}

// ── solve: the principal roots of a periodic exponential are not all ─────

/// N2's fuzz_calculus input `s7Ozs3KbMLOzs7Ozs7OzkXKbdpFym3aZIQ==`:
/// `solve(−1/(−1)^x = −1, x)` gives `[0]`, while every even integer solves
/// it.  That is `Ex::solve`'s documented convention for periodic functions
/// (the principal branch of `ln`: `(−1)^x = e^{iπx} = 1` at `iπx = 0`;
/// SymPy 1.14 `solve` gives `[0]` too), and `solve_as_set` has the family
/// `2n`.  Oracle: `e^{2πin} = 1`.
#[test]
fn periodic_exponential_principal_root_and_family() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.parse("-1/(-1)^x + 1").unwrap();
    let roots: Vec<String> = e.solve(&x).unwrap().iter().map(|r| r.to_string()).collect();
    assert_eq!(roots, ["0"]);
    assert_eq!(
        e.solve_as_set(&x).to_string(),
        "ImageSet(Lambda(_n, 2*_n), Integers)"
    );
}

/// The crate's real root searches read the same principal list as every
/// real zero: `∫₁³ dx/((−1)^x − 1)` was `−2`, across the simple pole at
/// `x = 2` (`e^{iπx} − 1 ≈ iπ(x − 2)`; its principal value is `−1`),
/// `∫₂⁶ dx/(i^x − 1)` was `−4` (pole at 4), `∫₁¹⁰ dx/(2^{ix} − 1)` was
/// `−9.03 + 0.09i` (pole at `2π/ln 2 ≈ 9.06`), `∫₁³ dx/(cosh(iπx) − 1)` was
/// `0` (double pole at 2).  Now no value is returned.  Oracle: the zeros
/// `x = 2n`, `4n`, `2πn/ln 2`, `2n` of the denominators.
#[test]
fn integrals_across_poles_of_periodic_exponentials_are_not_answered() {
    for (den, lo, hi) in [
        ("(-1)^x - 1", "1", "3"),
        ("exp(I*pi*x) - 1", "1", "3"),
        ("I^x - 1", "2", "6"),
        ("2^(I*x) - 1", "1", "10"),
        ("cosh(I*pi*x) - 1", "1", "3"),
    ] {
        assert!(integral_of_reciprocal(den, lo, hi).is_err(), "{den}");
    }
}

/// `Ex::singularities` took the principal root list of a non-trigonometric
/// denominator for all its real zeros: `1/((−1)^x − 1)` on `[0, 10]` was
/// `{0}`, `1/(i^x + 1)` was `{2}`.  Now the general solve's families are
/// enumerated, as for `sin`/`cos`/`tan`.  Oracle: `e^{iπx} = 1` at even `x`,
/// `e^{iπx/2} = −1` at `x = 2 + 4n`.
#[test]
fn singularities_of_periodic_exponentials_are_all_listed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let dom = ctx.interval(&ctx.int(0), &ctx.int(10), IntervalKind::Closed);
    let s = |src: &str| {
        let f = 1 / ctx.parse(src).unwrap();
        f.singularities(&x, Some(&dom)).unwrap().to_string()
    };
    assert_eq!(s("(-1)^x - 1"), "{0, 10, 2, 4, 6, 8}");
    assert_eq!(s("I^x + 1"), "{10, 2, 6}");
}

/// Without a real zero the integral is still taken: `∫₀¹ dx/((−1)^x − 2)`
/// (`|(−1)^x| = 1`).  Oracle: mpmath `quad(1/(exp(iπt) − 2), [0, 1]) =
/// −0.5 − 0.17484957628302988900i`.
#[test]
fn integral_of_a_periodic_exponential_without_real_zeros() {
    let v = integral_of_reciprocal("(-1)^x - 2", "0", "1")
        .unwrap()
        .eval_complex64()
        .unwrap();
    assert!(
        (v.re + 0.5).abs() < 1e-13 && (v.im + 0.174_849_576_283_029_9).abs() < 1e-13,
        "{v}"
    );
}

/// `(−2)^x = 4` for a real `x`: the one real root 2 lies on the branch `k =
/// 1` of the logarithm, not the principal one (`ln 4/(ln 2 + iπ)`, not
/// real), and `Ex::solve` gave `[]`; `(1 + i)^x = 2i` was refused ("cannot
/// decide whether … is real").  Now the real roots over every branch:
/// `|b^x| = |b|^x = |c|` fixes `x`, the phase decides.  Oracle: `(−2)² = 4`,
/// `(1 + i)² = 2i`, and `|(−1)^x| = 1 ≠ 2`, `(−2)^x = 3` needs `x = ln 3/ln
/// 2`, where `arg = π·ln 3/ln 2 ≢ 0 (mod 2π)`.
#[test]
fn real_roots_of_powers_with_negative_and_complex_bases() {
    assert_eq!(real_roots("(-2)^x - 4").unwrap(), ["2"]);
    assert_eq!(real_roots("(1+I)^x - 2*I").unwrap(), ["2"]);
    assert_eq!(real_roots("(-8)^x + 512").unwrap(), ["3"]);
    assert!(matches!(
        real_roots("(-1)^x - 2"),
        Err(SymplexError::NoSolution { .. })
    ));
    assert!(matches!(
        real_roots("(-2)^x - 3"),
        Err(SymplexError::NoSolution { .. })
    ));
}

/// With a pole on another branch located, `∫₁³ dx/((−2)^x − 4)` is divergent
/// (it was refused: "no antiderivative in closed form").  Oracle: `(−2)^x −
/// 4` vanishes at 2 with derivative `4·(ln 2 + iπ) ≠ 0`, a simple pole.
#[test]
fn integral_across_a_pole_on_another_branch_diverges() {
    assert!(matches!(
        integral_of_reciprocal("(-2)^x - 4", "1", "3"),
        Err(SymplexError::Divergent { .. })
    ));
}
