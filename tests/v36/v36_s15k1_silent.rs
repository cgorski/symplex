//! Tests of session 15 (s15k1): silent wrong answers in canon, eval and the assumptions.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, src: &str) -> Ex {
    ctx.parse(src).unwrap()
}

/// The folds of `eval` that hold for every finite value of one argument
/// when the other is special (`Γ(s, ∞) = 0`, `Γ(s, 0) = Γ(s)`, `γ(s, 0) =
/// 0`, `E_ν(∞) = 0`, `J_ν(∞) = 0`) fired on any order: up to 0.56
/// `uppergamma(elliptic_k(besseli(0, 0)), ∞)` (the order evaluates to `K(1)
/// = zoo`) was 0 and `is_zero()` was `Some(true)`, and so were
/// `uppergamma(zoo, ∞)`, `lowergamma(zoo, 0)`, `expint(zoo, ∞)`,
/// `expint(elliptic_pi(π, 1), ∞)` (`Π(π | 1) = −∞`, not evaluated),
/// `uppergamma(y·∞, ∞)`, `uppergamma((1 + i)·∞, ∞)` and `uppergamma(y + ∞,
/// ∞)`.  Now they stay; finite orders keep folding.
///
/// Oracle: the iterated limits differ, so there is no value: mpmath
/// `gammainc(1000, 100)` = 4.02·10²⁵⁶⁴ while `gammainc(10, 1000)` =
/// 5.1·10⁻⁴⁰⁸ (`Γ(s, x) → 0` as `x → ∞` for each `s`, `→ Γ(s) → ∞` as `s →
/// +∞` for each `x`); `expint(1000, 100)` = 3.4·10⁻⁴⁷ but `expint(−1000,
/// 1000)` = 2.0·10⁻⁴³⁶ and `E_{−s}(x) ~ Γ(1 + s)/x^{1+s}` grows without
/// bound in `s` for fixed `x`; DLMF 19.6.1 (`K(1) = ∞`).
#[test]
fn folds_at_a_special_argument_need_a_finite_other_argument() {
    let ctx = Context::new();
    let k = parse(&ctx, "uppergamma(elliptic_k(besseli(0, 0)), oo)");
    assert_ne!(k.is_zero(), Some(true), "{k}");
    assert_ne!(k.eval().to_string(), "0");
    for src in [
        "uppergamma(zoo, oo)",
        "uppergamma(zoo, 0)",
        "lowergamma(zoo, 0)",
        "expint(zoo, oo)",
        "expint(elliptic_k(besseli(0, 0)), oo)",
        "expint(elliptic_pi(pi, 1), oo)",
        "uppergamma(elliptic_pi(pi, 1), oo)",
        "uppergamma(y*oo, oo)",
        "uppergamma((1 + I)*oo, oo)",
        "uppergamma(y + oo, oo)",
        "besselj(elliptic_pi(pi, 1), oo)",
    ] {
        let e = parse(&ctx, src);
        assert_ne!(e.eval().to_string(), "0", "{src}");
        assert_ne!(e.is_zero(), Some(true), "{src}");
    }
    let x = ctx.symbol("x");
    let p = parse(&ctx, "x*uppergamma(elliptic_k(besseli(0, 0)), oo)");
    assert!(p.solve(&x).is_err(), "solve of {p}: {:?}", p.solve(&x));
    // Finite orders still fold (DLMF 8.11.2, 8.19.1, 10.7.8).
    for (src, value) in [
        ("uppergamma(2, oo)", "0"),
        ("uppergamma(s, oo)", "0"),
        ("expint(3, oo)", "0"),
        ("lowergamma(3, oo)", "2"),
        ("lowergamma(s, 0)", "0"),
        ("besselj(n, oo)", "0"),
    ] {
        assert_eq!(parse(&ctx, src).eval().to_string(), value, "{src}");
    }
}

/// A function left unevaluated at `zoo` has no value (`zoo` has no
/// direction, so no value at infinity applies), so `0·f(zoo)` is `nan`:
/// up to 0.56 `0·uppergamma(zoo, ∞)`, `0·Γ(zoo)` and `0·atan(zoo)` were 0
/// (SymPy 1.14 too).  A function at a finite argument keeps the canonical
/// `0·f = 0`, and so does a polynomial of degree 0 (the constant 1).
///
/// Oracle: `Γ` has no limit at `zoo`: mpmath `gamma(100)` = 9.3·10¹⁵⁵,
/// `gamma(100j)` = −2.8·10⁻⁷¹ + 1.5·10⁻⁶⁹i, poles on the negative axis;
/// `atan(t) → ±π/2` as `t → ±∞`; `uppergamma(zoo, ∞)` as in the test
/// above.
#[test]
fn zero_times_a_function_left_at_complex_infinity_is_undefined() {
    let ctx = Context::new();
    for src in [
        "0*uppergamma(zoo, oo)",
        "0*gamma(zoo)",
        "0*atan(zoo)",
        "0*x*erf(zoo)",
    ] {
        assert_eq!(parse(&ctx, src).to_string(), "nan", "{src}");
    }
    for src in [
        "0*gamma(y)",
        "0*uppergamma(2, oo)",
        "0*Heaviside(y)",
        "0*legendre(0, zoo)",
        "0*jacobi(0, zoo, 1, x)",
    ] {
        assert_eq!(parse(&ctx, src).to_string(), "0", "{src}");
    }
}

/// `Min`/`Max` read their arguments as real (where the order is defined),
/// so `±∞ + c` is `±∞` there and `+∞` absorbs a `Max` none of whose
/// arguments is certified non-real; `W(∞ + c)` is `∞` (as `ln(∞ + c)`).  Up to 0.56 `Max(−3, ∞, y)`, `Min(y + ∞, y)` and `W(y + ∞)`
/// stayed, and `solve` refused `bessely(2, φ) + x/e − tanh(Max(−3, ∞, y))`
/// (golden 62040; 0.54 solved it) and `x + W(y + ∞)` (no solution).
///
/// Oracle: SymPy 1.14 `Max(oo, y)` → `oo`, `Min(y + oo, y)` → `y` for a
/// real `y`; `tanh(∞) = 1`; `W(z) = ln z − ln ln z + o(1)` (DLMF 4.13.10),
/// whose imaginary part tends to `arg z → 0` along `t + c`.  `ζ(3/7 + i)` is not real (mpmath
/// `zeta(3/7 + 1j)` = 0.11 − 0.66i), so `Max(∞, ζ(3/7 + i))` has no value.
#[test]
fn min_and_max_read_infinite_sums_as_infinities() {
    let ctx = Context::new();
    for (src, value) in [
        ("max(oo, y)", "oo"),
        ("max(-3, oo, y)", "oo"),
        ("tanh(max(-3, oo, y))", "1"),
        ("min(y + oo, y)", "y"),
        ("max(y - oo, y)", "y"),
        ("min(y - oo, y)", "-oo"),
        ("exp(max(ceiling(oo), y))", "oo"),
        ("LambertW(y + oo)", "oo"),
    ] {
        assert_eq!(parse(&ctx, src).to_string(), value, "{src}");
    }
    assert_ne!(parse(&ctx, "max(oo, zeta(3/7 + I))").to_string(), "oo");
    let x = ctx.symbol("x");
    let e = parse(&ctx, "bessely(2, GoldenRatio) + x/E - tanh(max(-3, oo, y))");
    let roots: Vec<String> = e.solve(&x).unwrap().iter().map(|r| r.to_string()).collect();
    assert_eq!(roots, ["(-bessely(2, GoldenRatio) + 1)*E"]);
    let w = parse(&ctx, "x + LambertW(y + oo)");
    assert!(w.solve(&x).is_err(), "{:?}", w.solve(&x));
}

/// A piecewise parameter is not "nonzero for generic values": `0^y` is 0
/// on the half plane `re y > 0`, `δ(1/2, y)` everywhere but `y = 1/2`,
/// `⌊y⌋` on `[0, 1)`, `H(y)` for `y < 0`.  Up to 0.56 the series engine
/// divided by them: `series(sin(x)/(x + 0^y), x, 0, 4)` was `x/0^y −
/// x²/(0^y)² + …`, `limit(sin(x)/(x + 0^y), x, 0)` was 0, `series(1/(x +
/// δ(1/2, y)))` and `series(1/(x + ⌊y⌋))` were Taylor series in
/// `1/δ(1/2, y)`, `1/⌊y⌋`, and `cancel(0^y·(x + 1)/(0^y·(x + 2)))` was `(x
/// + 1)/(x + 2)`.  Now they are refused (and `cancel` keeps the input).
///
/// Oracle: at `y = 1` (`0^1 = 0`; SymPy `0**Symbol('y', positive=True)` →
/// 0) `sin(x)/x → 1`, not 0, and `1/x` has no Taylor series; `δ(1/2, 0) =
/// 0`, `⌊1/2⌋ = 0`.  An analytic parameter keeps its generic reading:
/// `series(1/(x + y))` is `1/y − x/y² + …`.
#[test]
fn piecewise_parameters_are_not_generic_nonzero_constants() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    for src in [
        "sin(x)/(x + 0^y)",
        "1/(x + 0^y)",
        "1/(x + KroneckerDelta(1/2, y))",
        "sin(x)/(x + KroneckerDelta(1/2, y))",
        "1/(x + floor(y))",
        "1/(x + Heaviside(y))",
    ] {
        let e = parse(&ctx, src);
        assert!(e.try_series(&x, &zero, 4).is_err(), "series of {src}");
        assert!(e.try_limit(&x, &zero).is_err(), "limit of {src}");
    }
    let c = parse(&ctx, "0^y*(x + 1)/(0^y*(x + 2))");
    assert_eq!(c.cancel(&x).to_string(), c.to_string());
    let generic = parse(&ctx, "1/(x + y)");
    assert_eq!(
        generic.try_series(&x, &zero, 2).unwrap().to_string(),
        "-x*y^(-2) + 1/y"
    );
}

/// The Kronecker delta of a fraction and an integer is 0: up to 0.56
/// `KroneckerDelta(1/2, n)` stayed for an integer `n`.  With a plain `y` it
/// stays (it is 1 at `y = 1/2`).
///
/// Oracle: SymPy 1.14 `KroneckerDelta(S(1)/2, Symbol('n', integer=True))`
/// → 0, `KroneckerDelta(S(1)/2, y)` stays.
#[test]
fn kronecker_delta_of_a_fraction_and_an_integer_is_zero() {
    let ctx = Context::new();
    ctx.symbol_with("n", &[Assumption::Integer]).unwrap();
    assert_eq!(parse(&ctx, "KroneckerDelta(1/2, n)").to_string(), "0");
    assert_eq!(parse(&ctx, "KroneckerDelta(n + 1/3, 2*n)").to_string(), "0");
    assert_eq!(
        parse(&ctx, "KroneckerDelta(1/2, y)").to_string(),
        "KroneckerDelta(1/2, y)"
    );
}

/// The exact zero test related `e` and `exp(c)` only through the canonical
/// forms: written in exponentials, `exp(−e − 1)` became `exp(−1)·exp(−e)`
/// and `exp(−1)` was a generator independent of `e`, so up to 0.56 the
/// constant below (0) had `is_zero()` `None`.  Now the numeric parts of the
/// exponents and `e` are powers of one unit.
///
/// Oracle (exact): with `u = e^{1+e}`, `u·e^{−e−1} = 1`, so the bracket is
/// `−(2 + e)·e/(1 + e) + e = −e/(1 + e)` and the whole is `−e + e = 0`.
#[test]
fn e_and_exponentials_of_numbers_are_related_in_the_zero_test() {
    let ctx = Context::new();
    let z = parse(
        &ctx,
        "(1 + E)*(-((1 + E)*exp(1 + E) + exp(1 + E))*exp(-E - 1)*E/(1 + E) + E) + E",
    );
    assert_eq!(z.is_zero(), Some(true));
    let nz = parse(
        &ctx,
        "(1 + E)*(-((1 + E)*exp(1 + E) + exp(1 + E))*exp(-E - 1)*E/(1 + E) + E) + 2*E",
    );
    assert_eq!(nz.is_zero(), Some(false));
}

/// The orthogonal polynomials take their values at negative integer
/// degrees by the identities of their defining functions; up to 0.56 every
/// one stayed, and series ended in `K(jacobi(−1, 2, 3, i) + 1)` = `K(1)` =
/// ∞.  `P_{−m} = P_{m−1}`, `T_{−m} = T_m`, `U_{−1} = 0`, `U_{−m} =
/// −U_{m−2}`, `L_{−m}(x) = eˣL_{m−1}(−x)` (the Legendre and Laguerre
/// functions), `P^μ_{−m} = P^μ_{m−1}`; the parametrised Jacobi, Gegenbauer
/// and generalised Laguerre families are the zero polynomial there (SymPy's
/// Gegenbauer convention; what their derivative formulas need at `n = 0`,
/// `C₀⁽λ⁾′ = 2λ·C_{−1}^{(λ+1)} = 0`); `H_{−m}` (a Hermite function, not a
/// polynomial) stays.  `P_0^{−1}(x) = √((1 − x)/(1 + x))` (Ferrers, DLMF
/// 14.5.12) is not 0: up to 0.56 `assoc_legendre(0, −1, x)` was.
///
/// Oracle: mpmath (20 digits) `legendre(−3, 0.3)` = −0.365 = `P₂(0.3)`,
/// `chebyt(−2, 0.3)` = −0.82 = `T₂`, `chebyu(−1, 0.3)` = 0, `chebyu(−3, 0.3)`
/// = −0.6 = `−U₁`, `jacobi(−1, 2, 3, 1j)` = 0, `jacobi(−2, 2, 3, 0.3)` = 0,
/// `gegenbauer(−1, 0.7, 0.3)` = 0, `laguerre(−1, 2, 0.3)` = 0, `legenp(0,
/// −1, 0.5)` = 0.57735 = `√(1/3)`; `laguerre(−1 + 10⁻¹⁵, 0, 0.3)` = 1.34985881
/// = `e^{0.3}` (the Laguerre function `₁F₁(−ν; 1; x)` at `ν = −1`; mpmath's
/// `laguerre(−1, 0, ·)` takes its binomial prefactor as 0 at that joint
/// pole); SymPy 1.14 `legendre(-3, y)` → `3*y**2/2 - 1/2`, `chebyshevu(-3,
/// y)` → `-2*y`, `laguerre(-2, y)` → `(y + 1)*exp(y)`, `gegenbauer(-1, y,
/// z)` and `gegenbauer(-1, 1/2, z)` → 0.
#[test]
fn orthogonal_polynomials_at_negative_degrees() {
    let ctx = Context::new();
    for (src, value) in [
        ("legendre(-1, z)", "1"),
        ("legendre(-3, z)", "3/2*z^2 - 1/2"),
        ("chebyshev_t(-2, z)", "2*z^2 - 1"),
        ("chebyshev_u(-1, z)", "0"),
        ("chebyshev_u(-3, z)", "-2*z"),
        ("laguerre(-1, z)", "exp(z)"),
        ("laguerre(-2, z)", "(z + 1)*exp(z)"),
        ("jacobi(-1, 2, 3, I)", "0"),
        ("jacobi(-2, a, b, z)", "0"),
        ("gegenbauer(-1, a, z)", "0"),
        ("gegenbauer(-1, 1/2, z)", "0"),
        ("assoc_laguerre(-1, 2, z)", "0"),
        ("assoc_legendre(-2, 1, z)", "-sqrt(-z^2 + 1)"),
        // The Hermite function (0.59; this row pinned "stays" before):
        // `H_{−1}(z) = (√π/2)·e^{z²}·erfc z` (DLMF 7.18, 12.7.6).
        ("hermite(-1, z)", "1/2*sqrt(pi)*exp(z^2)*erfc(z)"),
    ] {
        assert_eq!(parse(&ctx, src).eval().to_string(), value, "{src}");
    }
    for src in ["assoc_legendre(0, -1, z)", "assoc_legendre(-1, -1, z)"] {
        assert_ne!(parse(&ctx, src).eval().to_string(), "0", "{src}");
    }
    assert_eq!(
        parse(&ctx, "elliptic_k(x^3 + jacobi(-1, 2, 3, I) + 1)")
            .eval()
            .to_string(),
        "elliptic_k(x^3 + 1)"
    );
}

/// Limits and series never divide by a constant that is 0 without looking
/// like it: `J_{1/2}(z) = √(2/(πz))·sin z` and `J_{−1/2}(z) = √(2/(πz))·cos z`
/// (DLMF 10.16.1), so `J_{1/2}(π) = 0` and `J_{−1/2}(π/2) = 0`.  Up to 0.55 `limit(1/besselj(−1/2, π/2 + x), x, 0)` was
/// `1/besselj(−1/2, π/2)` and `limit(sin(x)/(sin(x)·J_{1/2}(π) + x²), x, 0)`
/// was `1/J_{1/2}(π)` (L'Hôpital's rule and the series route both divided
/// by the unproved zero); they are refused.  A piecewise parameter in a
/// leading coefficient is not a generic nonzero constant either:
/// `series(acosh(P₂(x))/(x² + H(y)), x, 0)` divided by `H(y)`, which is 0
/// for `y < 0`.
#[test]
fn limits_never_divide_by_a_hidden_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    for src in [
        "sin(x)/(sin(x)*besselj(1/2, pi) + x^2)",
        "(cos(x) - 1)/(x^2*besselj(1/2, pi) + x^3)",
        "sin(x)/(x*besselj(-1/2, pi/2 + x))",
        "1/besselj(-1/2, pi/2 + x)",
    ] {
        let r = parse(&ctx, src).try_limit(&x, &zero);
        assert!(r.is_err(), "{src}: {r:?}");
    }
    let s = parse(&ctx, "acosh(legendre(2, x))/(x^2 + Heaviside(y))").series(&x, &zero, 4);
    assert!(!s.to_string().contains("/H(y)"), "{s}");
    // `KroneckerDelta(a, b)` is a step in `x`: a Taylor series only where
    // `a − b` is a nonzero number at the point (then it is constant nearby).
    // Up to 0.55 `series(KroneckerDelta(0, x), x, 0)` was `1` (it is 0 for
    // `x ≠ 0`) and `series(KroneckerDelta(x, y), x, 0)` was `KroneckerDelta(0, y)`.
    for src in [
        "KroneckerDelta(0, x)",
        "sin(KroneckerDelta(0, x))",
        "acsc(KroneckerDelta(x, y))^(-2)",
    ] {
        let s = parse(&ctx, src).series(&x, &zero, 4);
        assert!(s.has_unevaluated(), "{src}: {s}");
    }
    let s = parse(&ctx, "x + KroneckerDelta(1, x)").series(&x, &zero, 4);
    assert_eq!(s.to_string(), "x");
}
