//! Tests of session 15 (s15p1): evalf honesty: underflow, infinite verdicts, values at zoo.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::errors::SymplexError;
use symplex::prelude::*;
use symplex::stats::Distribution;

fn parse(ctx: &Context, src: &str) -> Ex {
    ctx.parse(src).unwrap()
}

/// `eval_decimal` refuses a value that lies below the exponent range (as
/// `erfc(10⁵)`; certainly not 0 at exact arguments, "not known to be 0 or
/// not" at rounded ones), and `eval_f64` gives its correctly rounded `0.0`.
fn assert_underflow(e: &Ex, what: &str) {
    let d = e.eval_decimal(30);
    assert!(
        matches!(&d, Err(SymplexError::Unevaluable { reason }) if reason.contains("underflows")),
        "{what}: {d:?}"
    );
    assert_eq!(e.eval_f64().unwrap(), 0.0, "{what}");
}

/// The incomplete beta and lower incomplete gamma functions whose
/// prefactor `exp(…)` falls below the exponent range: their routes return
/// 0 there, which counted as an exact 0 (the accuracy layer held a 0 from
/// exact arguments as an underflow only for `erfc`, `Γ`, `Ei`, Airy,
/// Bessel, `Γ(s, x)`, `E_ν`).  Before: `betainc_regularized(1.4·10³⁰, 18, 0,
/// ½).eval_decimal(30)` was `"0"`, likewise `betainc(…)`, the upper tail
/// `betainc_regularized(18, 1.4·10³⁰, ½, 1)` and `lowergamma(10¹⁰, ½)`;
/// `beta(10³⁰, 10¹⁰)` was refused "beyond the exponent range" in `eval_f64`
/// too, and `is_zero()` of the underflowed tail was `Some(true)` (`ln` of it
/// was rewritten `iπ + ln(−…)`).  The tiny value still vanishes next to 1.
///
/// Oracle: DLMF 8.17.8 in log space, `I_x(a, b) ≤ x^a(1 − x)^b/(a·B(a, b))·(1
/// − x·(a + b)/(a + 1))⁻¹` (the hypergeometric terms' ratio): `log₁₀
/// I_½(1.4·10³⁰, 18) ≤ −4.2144·10²⁹`; mpmath 1.3 `gammainc(10**10, 0, 0.5)`
/// = `1.39008323929302576037551224027e-3010299967`, `log₁₀ B(10³⁰, 10¹⁰)` =
/// `−2.04342944823·10¹¹` (`loggamma`); all below `2^(−2³¹) ≈ 10^(−6.46·10⁸)`,
/// and positive (integrals of positive functions; the reversed limits
/// negative).
#[test]
fn incomplete_beta_and_gamma_below_the_exponent_range_are_refused() {
    let ctx = Context::new();
    for src in [
        "betainc_regularized(14*10^29, 18, 0, 1/2)",
        "betainc_regularized(18, 14*10^29, 1/2, 1)",
        "betainc_regularized(14*10^29, 18, 1/2, 0)",
        "betainc_regularized(10^10, 1/2, 0, 1/2)",
        "betainc(14*10^29, 18, 0, 1/2)",
        "lowergamma(10^10, 1/2)",
        "lowergamma(14*10^29, 1/2)",
        "beta(10^30, 10^10)",
    ] {
        let e = parse(&ctx, src);
        assert_underflow(&e, src);
        let d = e.eval_decimal(30);
        assert!(
            matches!(&d, Err(SymplexError::Unevaluable { reason }) if reason.contains("not 0 but underflows")),
            "{src}: {d:?}"
        );
        assert_eq!(e.is_zero(), Some(false), "{src}");
    }
    // Rounded arguments: refused as well (up to 0.58 `0`).
    for src in [
        "betainc_regularized(14*10^29 + 1/3, 18, 0, 1/2)",
        "betainc_regularized(5*10^99, 1/2, 0, 1 - 10^-90)",
    ] {
        assert_underflow(&parse(&ctx, src), src);
    }
    for src in [
        "1 - betainc_regularized(14*10^29, 18, 0, 1/2)",
        "lowergamma(10^10, 1/2) + 1",
    ] {
        assert_eq!(parse(&ctx, src).eval_decimal(30).unwrap(), "1", "{src}");
    }
    // In range, unchanged (mpmath `gammainc(3*10**7, 0, 0.5)`).
    assert_eq!(
        parse(&ctx, "lowergamma(3*10^7, 1/2)")
            .eval_decimal(20)
            .unwrap(),
        "2.727797177730232667e-9030908"
    );
}

/// The statistics routes through the incomplete beta function (L2's
/// reproducers).  Before: `Beta(18, 1.4·10³⁰).sf(½)` printed `0`, as did
/// the Student t survival function at `ν = 10¹⁰⁰`, `t = 10⁵`: its limit
/// `ν/(ν + t²) = 1/(1 + 10⁻⁹⁰)` is rounded, and the bound of the integrand
/// there took `log₂` of the limit in `f64` (0), keeping `2³¹³` where the
/// true sensitivity is below `2^(−7·10⁹)`: the underflowed value was 0
/// within that bound.
///
/// Oracle: `sf = I_{1−½}(1.4·10³⁰, 18)` as above; the t tail is
/// `½·I_{ν/(ν+t²)}(ν/2, ½) ≤ (1 − 10⁻⁹⁰)^(5·10⁹⁹) ≈ e^(−5·10⁹)` (and the
/// normal limit `erfc(10⁵/√2)/2 ≈ 10^(−2.2·10⁹)`), below `2^(−2³¹)`.
#[test]
fn distribution_tails_below_the_exponent_range_are_refused() {
    let ctx = Context::new();
    let b = Distribution::beta(ctx.int(18), parse(&ctx, "14*10^29"));
    assert_underflow(&b.sf(&ctx.rational(1, 2)), "Beta(18, 1.4e30).sf(1/2)");
    let t = Distribution::student_t(parse(&ctx, "10^100"));
    assert_underflow(&t.sf(&parse(&ctx, "10^5")), "t(1e100).sf(1e5)");
}

/// `evalf` tells an infinite value (`SymplexError::Infinite`: a function at
/// a pole of exact arguments, or a value built on one) from one it cannot
/// compute (`Unevaluable`); a singular point met only by an argument that
/// rounded onto it is no such claim.  Before: both were `Unevaluable`.
///
/// Oracle: mpmath 1.3 `ellippi(pi, 1)` = `-inf`, `expint(1, 0)` = `+inf`;
/// `erfinv(9/4)` has no real value (`|erf x| < 1`); DLMF 19.12.1 `K(1 − ε) =
/// ln(4/√ε) + O(ε ln ε)` = `ln 4 + 30 ln 10` = `70.46384715094126113937420788…`
/// at `ε = 10⁻⁶⁰`; `E_n(0) = 1/(n − 1)` (DLMF 8.19.6) = `10⁴⁰`.
#[test]
fn evalf_tells_an_infinite_value_from_one_it_cannot_compute() {
    let ctx = Context::new();
    for src in [
        "elliptic_pi(pi, 1)",
        "1/elliptic_pi(pi, 1)",
        "expint(1, 0)",
        "elliptic_k(1)",
    ] {
        let d = parse(&ctx, src).eval_decimal(15);
        assert!(
            matches!(d, Err(SymplexError::Infinite { .. })),
            "{src}: {d:?}"
        );
    }
    let d = parse(&ctx, "erfinv(9/4)").eval_decimal(15);
    assert!(matches!(d, Err(SymplexError::Unevaluable { .. })), "{d:?}");
    assert_eq!(
        parse(&ctx, "elliptic_k(1 - 10^-60)")
            .eval_decimal(30)
            .unwrap(),
        "70.4638471509412611393742078834"
    );
    assert_eq!(
        parse(&ctx, "expint(1 + 10^-40, 0)")
            .eval_decimal(10)
            .unwrap(),
        "1e40"
    );
}

/// The root acceptance refuses a candidate whose value meets an
/// unevaluated pole.  Before: `solve(x − elliptic_pi(π, 1), x)` claimed
/// `[elliptic_pi(π, 1)]` (which is `−∞`), `x·elliptic_pi(π, 1) − 1` claimed
/// `[1/elliptic_pi(π, 1)]` (`x = 0` gives `0·∞ − 1`, no value) and `x² −
/// elliptic_pi(π, 1)` claimed `±√elliptic_pi(π, 1)`.  A constant `evalf`
/// cannot evaluate for another reason is no evidence of an infinity: the
/// root `erfinv(9/4)` is kept as before (M1 found 229 such golden roots).
///
/// Oracle: mpmath 1.3 `ellippi(pi, 1)` = `-inf`, `ellippi(0.5, 1)` = `+inf`
/// (SymPy 1.14: `elliptic_pi(n, 1)` = `−∞/sign(n − 1)`).
#[test]
fn roots_at_an_unevaluated_pole_are_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "x - elliptic_pi(pi, 1)",
        "x*elliptic_pi(pi, 1) - 1",
        "x^2 - elliptic_pi(pi, 1)",
        "x + elliptic_pi(1/2, 1)",
    ] {
        let roots = parse(&ctx, src).solve(&x);
        assert!(
            matches!(&roots, Err(SymplexError::ComputationFailed { reason, .. }) if reason.contains("meets an infinity")),
            "{src}: {roots:?}"
        );
    }
    let roots = parse(&ctx, "x - erfinv(9/4)").solve(&x).unwrap();
    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0].to_string(), "erfinv(9/4)");
}

/// The closed forms at `zoo` follow the values the functions tend to in
/// every direction: a polynomial of degree `d ≥ 1` is `zoo`, of degree 0
/// (or the zero polynomial) its constant; `Li_s(zoo)` is `−1` for `s = 0`
/// and 0 for `s < 0`.  Before: `legendre(3, zoo)`, `chebyshev_t(3, zoo)`,
/// `chebyshev_t(−3, zoo)`, `laguerre(2, zoo)`, `assoc_laguerre(2, 1, zoo)`,
/// `gegenbauer(3, −2, zoo)`, `gegenbauer(3, −1, zoo)`, `jacobi(2, −3, 0, zoo)`,
/// `betainc(1, 2, 0, zoo)`, `polylog(−1, zoo)`, `polylog(0, zoo)` were `nan`
/// (`zoo + zoo` in the expanded closed form).  `E₀(z) = e^{−z}/z` and the
/// incomplete gamma functions have no limit at `zoo` and stay unevaluated
/// (changed in 0.59 from `nan`: `eval` takes no closed form at an untabulated
/// `zoo` argument, so that the node reads as when it is built, where `f − f`
/// is 0; s15q3), and `x + chebyshev_t(3, zoo) = 0` keeps its "no solution".
///
/// Oracle: the explicit polynomials (SymPy 1.14 `legendre`, `chebyshevt`,
/// `laguerre`, `assoc_laguerre`; DLMF 18.5.10 for `C₃^(−2)(x) = −4x`,
/// `C₃^(−1) = 0`; DLMF 18.5.7 for `P₂^(−3,0) = 1`; `∫₀^z (1 − t) dt = z −
/// z²/2`) evaluated by mpmath at `10¹²e^{iθ}` in 16 directions:
/// `min |p| > 10⁶·min |p(10⁶e^{iθ})|`; `Li₋₁(z) = z/(1 − z)²`, `Li₀(z) = z/(1 −
/// z)`: within `10⁻¹²` of 0 and −1 there.
#[test]
fn values_at_zoo_follow_the_limit_in_every_direction() {
    let ctx = Context::new();
    for (src, value) in [
        ("legendre(3, zoo)", "zoo"),
        ("legendre(0, zoo)", "1"),
        ("chebyshev_t(3, zoo)", "zoo"),
        ("chebyshev_t(-3, zoo)", "zoo"),
        ("chebyshev_u(-1, zoo)", "0"),
        ("laguerre(2, zoo)", "zoo"),
        ("assoc_laguerre(2, 1, zoo)", "zoo"),
        ("gegenbauer(3, -2, zoo)", "zoo"),
        ("gegenbauer(3, -1, zoo)", "0"),
        ("jacobi(2, -3, 0, zoo)", "1"),
        ("betainc(1, 2, 0, zoo)", "zoo"),
        ("betainc_regularized(1, 2, zoo, 1/2)", "zoo"),
        ("polylog(-1, zoo)", "0"),
        ("polylog(0, zoo)", "-1"),
        ("polylog(2, zoo)", "zoo"),
        ("expint(0, zoo)", "expint(0, zoo)"),
        ("uppergamma(2, zoo)", "uppergamma(2, zoo)"),
        ("lowergamma(2, zoo)", "lowergamma(2, zoo)"),
    ] {
        assert_eq!(parse(&ctx, src).eval().to_string(), value, "{src}");
    }
    let x = ctx.symbol("x");
    let r = parse(&ctx, "x + chebyshev_t(3, zoo)").solve(&x);
    assert!(matches!(r, Err(SymplexError::NoSolution { .. })), "{r:?}");
}
