//! Tests of session 15 (s15h1): values at infinity for the remaining functions, the even prime.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, src: &str) -> Ex {
    ctx.parse(src).unwrap()
}

/// Library and special functions at `±i·∞` fold to their values when they
/// are built, so `0·f(i·∞)` is `0·∞` where `f` is infinite: up to 0.53
/// they stayed unevaluated and `0·besselj(1, i·∞)` was 0 (`J₁(i·∞) =
/// i·∞`), as were `0·Si(i·∞)`, `0·airyai(i·∞)`; `0/Γ(i·∞)` was 0 (`0/0`).
///
/// Oracle: DLMF 10.27.6 (`J_ν(±iy) = e^{±iνπ/2}I_ν(y)`), 10.27.11 (`Y_ν(iy)
/// ~ e^{i(ν+1)π/2}I_ν(y)`), 6.5.5–6.5.6 (`Si(iy) = i·Shi(y)`, `Ci(iy) =
/// Chi(y) + iπ/2`), 5.4.3 (`|Γ(iy)|² = π/(y·sinh πy)`); mpmath at `y = 10³,
/// 10⁶` (30 and 60 digits): `besselj(1, 10**6*1j)` = 1.21·10⁴³⁴²⁹¹·i,
/// `bessely(0, 10**6*1j)` = 1.21·10⁴³⁴²⁹¹·i, `besseli(1, 10**6*1j)` =
/// −7.3·10⁻⁴·i, `besselk(0, -10**6*1j)` = 1.25·10⁻³·e^{0.136πi}, `airyai(10**6*1j)`
/// = 1.19·10²⁰⁴⁷²⁸³⁸⁰ with a phase turning with `y`, `chi(10**6*1j)` =
/// 1.5708i, `shi(10**6*1j)` = 1.5708i, `erfi(10**6*1j)` = i,
/// `fresnels(10**6*1j)` = −0.5i, `fresnelc(10**6*1j)` = 0.5i,
/// `ellipk(10**6*1j)` = 8.3·10⁻³·e^{0.22πi}, `ellipe(10**6*1j)` =
/// 1000·e^{−iπ/4}, `expint(1, 10**6*1j)` = 10⁻⁶·e^{−0.386πi},
/// `gammainc(1/2, 10**6*1j)` = 10⁻³·e^{−0.136πi}, `gamma(10**3*1j)` =
/// 5·10⁻⁶⁸⁴, `digamma(10**6*1j)` = 13.8155 + 1.5708i, `ci(10**6*1j)` =
/// 1.5·10⁴³⁴²⁸⁸ + 1.5708i, `ei(10**6*1j)` = −3.5·10⁻⁷ + 3.1415917i,
/// `li(10**6*1j)` = 9652.7 + 77407.2i at 10³ … both parts diverging.
/// SymPy 1.14 agrees on `besselj(1, oo*I)` → `oo*I`, `Si(oo*I)` → `oo*I`,
/// `Shi(oo*I)` → `I*pi/2`, `erfi(oo*I)` → `I`, `fresnels(oo*I)` → `-I/2`,
/// `elliptic_k(oo*I)` → 0; it keeps `besselj(1, -oo*I)` as nan.
#[test]
fn functions_at_imaginary_infinity_fold_to_their_values() {
    let ctx = Context::new();
    for (src, value) in [
        ("besselj(1, I*oo)", "I*oo"),
        ("besselj(1, -I*oo)", "-I*oo"),
        ("besselj(2, I*oo)", "-oo"),
        ("besselj(0, -I*oo)", "oo"),
        ("besselj(1/2, I*oo)", "zoo"),
        ("bessely(0, I*oo)", "I*oo"),
        ("bessely(1, I*oo)", "-oo"),
        ("bessely(0, -I*oo)", "-I*oo"),
        ("besseli(1, I*oo)", "0"),
        ("besseli(n, -I*oo)", "0"),
        ("besselk(0, -I*oo)", "0"),
        ("airyai(I*oo)", "zoo"),
        ("airybiprime(-I*oo)", "zoo"),
        ("Chi(I*oo)", "1/2*pi*I"),
        ("Shi(-I*oo)", "-1/2*pi*I"),
        ("erfi(I*oo)", "I"),
        ("fresnels(I*oo)", "-1/2*I"),
        ("fresnelc(I*oo)", "1/2*I"),
        ("elliptic_k(I*oo)", "0"),
        ("elliptic_e(I*oo)", "zoo"),
        ("expint(1, I*oo)", "0"),
        ("expint(0, -I*oo)", "0"),
        ("uppergamma(1/2, I*oo)", "0"),
        ("uppergamma(-1, I*oo)", "0"),
        ("lowergamma(0, I*oo)", "zoo"),
        ("gamma(I*oo)", "0"),
        ("factorial(-I*oo)", "0"),
        ("loggamma(I*oo)", "zoo"),
        ("digamma(I*oo)", "1/2*pi*I + oo"),
        ("lambertw(-I*oo)", "-1/2*pi*I + oo"),
        ("Si(I*oo)", "I*oo"),
        ("Si(-I*oo)", "-I*oo"),
        ("Ci(I*oo)", "1/2*pi*I + oo"),
        ("Ei(-I*oo)", "-pi*I"),
        ("li(I*oo)", "zoo"),
        ("polygamma(2, I*oo)", "0"),
    ] {
        let e = parse(&ctx, src);
        assert_eq!(e.to_string(), value, "{src} when built");
        assert_eq!(e.eval().to_string(), value, "{src} evaluated");
    }
    assert_eq!(
        parse(&ctx, "lowergamma(1/2, I*oo)").eval().to_string(),
        "sqrt(pi)"
    );
    for src in [
        "0*besselj(1, I*oo)",
        "0*Si(I*oo)",
        "0*airyai(I*oo)",
        "0/gamma(I*oo)",
        "0*besselj(n, I*oo)",
    ] {
        assert_eq!(parse(&ctx, src).to_string(), "nan", "{src}");
    }
    // Finite values keep the product 0.
    assert_eq!(parse(&ctx, "0*Chi(I*oo)").to_string(), "0");
    let x = ctx.symbol("x");
    assert!(
        parse(&ctx, "x*besselj(1, I*oo)").solve(&x).is_err(),
        "x·i·∞ has no root (up to 0.53 solve refused the constant factor)"
    );
}

/// The orthogonal polynomials of an integer degree at `i^t·∞` take the sign
/// of their leading coefficient and the parity of their degree, also when
/// built: up to 0.53 `hermite(1, ∞)` stayed (`solve(x·hermite(1, ∞))` gave
/// `[0]`, where `x·∞` has no root) and `eval` expanded the polynomial at
/// `∞`: `legendre(3, ∞)`, `laguerre(2, ∞)`, `gegenbauer(1, 0, ∞)` and
/// `jacobi(1, −1, −1, ∞)` gave `nan`, `jacobi(1, a, b, ∞)` gave `(a + 1)·∞ +
/// (b + 1)·∞` (`nan` at `a = 0`, `b = −3`, where the value is `−∞`).  A
/// polynomial whose leading coefficient vanishes keeps its lower terms
/// (`C₂^{(−1)} = 1`, `C₁^{(0)} = 0`); a symbolic degree has no known value
/// there, so `0·hermite(n, ∞)` is `nan` (was 0).
///
/// Oracle: the explicit polynomials (`P₃ = (5x³ − 3x)/2`, `L₃ = (−x³ + 9x² −
/// 18x + 6)/6`, `U₋₂ = −U₀ = −1`, `C₂^{(−1)}(x) = 1`, `P₁^{(−1,−1)} = 0`,
/// DLMF 18.5.7, 18.9.1); mpmath `hermite(-0.5, 10**6*1j)` =
/// 7.07·10⁻⁴·e^{−iπ/4}, `legendre(-0.5, 10**6)` = 7.2·10⁻³ (→ 0),
/// `laguerre(-0.5, 0, -10**6)` = 5.6·10⁻⁴; SymPy 1.14 `hermite(1, oo)` → oo,
/// `laguerre(3, oo)` → -oo, `legendre(-3, oo)` → oo.
#[test]
fn orthogonal_polynomials_at_infinity_fold_to_their_values() {
    let ctx = Context::new();
    for (src, value) in [
        ("hermite(1, oo)", "oo"),
        ("hermite(0, oo)", "1"),
        ("legendre(3, oo)", "oo"),
        ("legendre(3, -oo)", "-oo"),
        ("legendre(3, I*oo)", "-I*oo"),
        ("legendre(-1, oo)", "1"),
        ("legendre(-1/2, -oo)", "0"),
        ("laguerre(2, oo)", "oo"),
        ("laguerre(3, oo)", "-oo"),
        ("chebyshev_t(3, -oo)", "-oo"),
        ("chebyshev_u(-1, oo)", "0"),
        ("chebyshev_u(-2, I*oo)", "-1"),
        ("hermite(-1/2, I*oo)", "0"),
        ("hermite(1/2, -oo)", "-oo"),
        ("gegenbauer(1, 0, oo)", "0"),
        ("gegenbauer(2, -1, oo)", "1"),
        ("gegenbauer(3, -2, oo)", "-oo"),
        ("gegenbauer(3, 1/2, -oo)", "-oo"),
        ("jacobi(1, -1, -1, oo)", "0"),
        ("jacobi(2, 1, 2, oo)", "oo"),
        ("assoc_laguerre(3, a, -oo)", "oo"),
        ("assoc_legendre(2, 1, oo)", "-I*oo"),
        ("assoc_legendre(2, 2, oo)", "-oo"),
    ] {
        let e = parse(&ctx, src);
        assert_eq!(e.to_string(), value, "{src} when built");
        assert_eq!(e.eval().to_string(), value, "{src} evaluated");
    }
    let x = ctx.symbol("x");
    assert_eq!(parse(&ctx, "x*hermite(1, oo)").to_string(), "x*oo");
    assert!(parse(&ctx, "x*hermite(1, oo)").solve(&x).is_err());
    assert_eq!(parse(&ctx, "0*hermite(n, oo)").to_string(), "nan");
    // A parameter without a value: no closed form at `∞`.
    let j = parse(&ctx, "jacobi(1, a, b, oo)");
    assert_eq!(j.eval().to_string(), "jacobi(1, a, b, oo)");
    // The degree slot is not tabulated: `0·f` keeps its 0.
    assert_eq!(parse(&ctx, "0*legendre(oo, 1/2)").to_string(), "0");
}

/// `F(φ | ±∞) = F(0 | m) = 0`, `Π(n | i^t·∞) = Π(i^t·∞ | m) = 0` (unless the
/// other argument is 1), when built: up to 0.53 they stayed, so
/// `x·elliptic_f(0, ∞)` was a product (solve refused the constant factor;
/// in 0.52 it gave `[0]`) and `euler_number(5)/elliptic_pi(−∞, π⁴)` was 0
/// (`E₅ = 0`: `0/0`).
///
/// Oracle: SymPy 1.14 `elliptic_f(z, oo)` → 0, `elliptic_pi(oo, m)` → 0;
/// mpmath `ellipf(1, -10**6)` = 7.7·10⁻³, `ellippi(-10**6, pi**4)` =
/// 1.57·10⁻³ (→ 0 as `π/(2√|n|)`), `ellippi(2, 10**6*1j)` = 7.2·10⁻³.
#[test]
fn elliptic_integrals_at_infinity_fold_to_zero() {
    let ctx = Context::new();
    for src in [
        "elliptic_f(0, oo)",
        "elliptic_f(1, -oo)",
        "elliptic_f(x, I*oo)",
        "elliptic_pi(-oo, pi^4)",
        "elliptic_pi(oo, m)",
        "elliptic_pi(2, I*oo)",
    ] {
        assert_eq!(parse(&ctx, src).to_string(), "0", "{src}");
    }
    let x = ctx.symbol("x");
    assert_eq!(parse(&ctx, "x*elliptic_f(0, oo)").to_string(), "0");
    let q = parse(&ctx, "euler_number(5)/elliptic_pi(-oo, pi^4)");
    assert_eq!(q.eval().to_string(), "nan");
    // `Π(1 | m)` is a pole: no finite value at `m = ∞`.
    assert_eq!(parse(&ctx, "0*elliptic_pi(1, oo)").to_string(), "nan");
    // `F(i·∞ | m) = i·K(1 − m)` is finite and not tabulated.
    assert_eq!(parse(&ctx, "0*elliptic_f(I*oo, 1/2)").to_string(), "0");
    let _ = x;
}

/// A constant argument next to the infinite one must be known finite (and,
/// for `Π`, known not to be 1) before the value at infinity is folded:
/// before H1b `elliptic_pi(besseli(0, 0), ∞)` was built as 0 and evaluated
/// to 0 (`besseli(0, 0) = 1`, `Π(1 | m) = zoo`: 0.53 evaluated it to
/// `zoo`), `elliptic_pi(∞, besseli(0, 0))` was 0 (`Π(n | 1) = −∞` for `n >
/// 1`), and `elliptic_f(elliptic_pi(π, 1), ∞)` and
/// `besselj(∞, elliptic_k(besseli(0, 0)))` were 0 (the first argument is
/// infinite: `Π(π | 1) = −∞`, `K(1) = ∞`).  A constant with a certified
/// value still folds.
///
/// Oracle: `I₀(0) = 1` (DLMF 10.25.2); `Π(1 | m)` diverges (`∫₀^{π/2}
/// sec²θ/√(1 − m sin²θ) dθ`, SymPy 1.14 `elliptic_pi(1, m)` → zoo),
/// `Π(n | 1) = ∫₀^{π/2} dθ/((1 − n sin²θ) cos θ)` diverges (SymPy:
/// `elliptic_pi(n, 1)` → `-oo/sign(n - 1)`), `K(1) = ∞` (DLMF 19.6.1);
/// mpmath (30 and 60 digits) `ellipf(ei(3)**e, m)` with `ei(3)**e` = 513.39:
/// 2.71 at `m = −10⁶`, 0.00497 at `−10¹²`, 0.00051 − 0.00497i at `10¹²`;
/// `ellippi(10**12, sqrt(2))` = −1.0·10⁻¹² − 1.57·10⁻⁶i,
/// `ellippi(10**12, chi(-0.8))` = 7.0·10⁻¹³ − 1.57·10⁻⁶i (30 digits).
#[test]
fn constant_arguments_must_be_known_finite_and_off_the_pole() {
    let ctx = Context::new();
    for src in [
        "elliptic_pi(besseli(0, 0), oo)",
        "elliptic_pi(oo, besseli(0, 0))",
        "elliptic_f(elliptic_pi(pi, 1), oo)",
        "besselj(oo, elliptic_k(besseli(0, 0)))",
    ] {
        let e = parse(&ctx, src);
        assert_eq!(e.to_string(), src, "{src} when built");
        assert_ne!(e.eval().to_string(), "0", "{src} evaluated");
    }
    assert_eq!(
        parse(&ctx, "elliptic_pi(besseli(0, 0), oo)")
            .eval()
            .to_string(),
        "zoo"
    );
    for src in [
        "elliptic_f(Ei(3)^E, oo)",
        "elliptic_pi(oo, sqrt(2))",
        "elliptic_pi(oo, Chi(-4/5))",
    ] {
        assert_eq!(parse(&ctx, src).to_string(), "0", "{src}");
    }
}

/// Factorials, binomials, polylogarithms, `W_k`, the integer sequences and
/// the Bessel functions of an infinite order take their values at infinity
/// when built: up to 0.53 `rising_factorial(e + ∞, 1)` stayed (solve
/// refused `x·rising_factorial(∞, 1)`), `polylog(0, ∞)` and
/// `polylog(−1, ∞)` evaluated to `nan` (`z/(1 − z)` at `∞`), and
/// `0·besselj(∞, 1)` was `nan` although `J_∞(1) = 0`.
///
/// Oracle: mpmath `rf(10**6, -1)` = 10⁻⁶ (SymPy 1.14: `rf(oo, -1)` → oo,
/// wrong), `binomial(10**6, 0.5)` = 1128.4 (SymPy: nan), `polylog(0,
/// 10**6)` = −1.000001, `polylog(-1, 10**6*1j)` = 10⁻⁶·i,
/// `polylog(2, -10**6)` = −97.08 (`−ln²x/2 − π²/6`), `lambertw(10**6, 1)`
/// = 11.38 + 5.8i (→ `∞ + 2πi`), `besselj(10**3, 1)` = 2.3·10⁻²⁸⁶⁹,
/// `catalan(10**6*1j)` = 5.6·10⁻¹⁰, `harmonic(10**6*1j)` = 13.8155 +
/// 1.5708i, `rf(-0.5, 10**6)` < 0.
#[test]
fn factorials_sequences_and_polylogs_at_infinity() {
    let ctx = Context::new();
    for (src, value) in [
        ("rising_factorial(E + oo, 1)", "oo"),
        ("rising_factorial(oo, -1)", "0"),
        ("falling_factorial(-oo, 3)", "-oo"),
        ("rising_factorial(-1/2, oo)", "-oo"),
        ("rising_factorial(2, I*oo)", "0"),
        ("binomial(oo, 1/2)", "oo"),
        ("binomial(-oo, 2)", "oo"),
        ("polylog(0, oo)", "-1"),
        ("polylog(-1, I*oo)", "0"),
        ("polylog(2, -oo)", "-oo"),
        ("polylog(1, oo)", "-pi*I - oo"),
        ("polylog(oo, 1/2)", "1/2"),
        ("lambertw(oo, 1)", "2*pi*I + oo"),
        ("lambertw(-oo)", "pi*I + oo"),
        ("fibonacci(oo)", "oo"),
        ("catalan(I*oo)", "0"),
        ("harmonic(-I*oo)", "-1/2*pi*I + oo"),
        ("besselj(oo, 1)", "0"),
        ("besseli(oo, x)", "0"),
        ("beta(1, oo)", "0"),
        ("beta(-1/2, oo)", "-oo"),
        ("polygamma(1, oo)", "0"),
    ] {
        let e = parse(&ctx, src);
        assert_eq!(e.to_string(), value, "{src} when built");
        assert_eq!(e.eval().to_string(), value, "{src} evaluated");
    }
    assert_eq!(parse(&ctx, "0*besselj(oo, 1)").to_string(), "0");
    assert_eq!(parse(&ctx, "0*fibonacci(-oo)").to_string(), "nan");
}

/// The infinite atoms answer every public query: `i·∞`, `−i·∞`, `(1 +
/// i)·∞` are infinite, not finite, not real, not complex and have no sign;
/// `∞ + i` has no sign either.  Up to 0.53 `(i·∞).is_finite()` was `None`
/// (SymPy 1.14: `False`), and so were its realness and sign, and `(∞ +
/// i).is_positive()`.
///
/// Oracle: the definitions (`c·∞` for a finite `c ≠ 0` is infinite in the
/// direction of `c`; an order exists on the extended reals only); SymPy
/// 1.14: `(oo*I).is_finite` → False, `(oo*I).is_extended_real` → False,
/// `(oo + I).is_extended_positive` → False.
#[test]
fn infinite_atoms_through_the_public_queries() {
    let ctx = Context::new();
    for src in ["I*oo", "-I*oo", "(1 + I)*oo", "oo + I", "zoo"] {
        let e = parse(&ctx, src);
        assert_eq!(e.is_finite(), Some(false), "{src} finite");
        assert_eq!(e.query(Props::INFINITE), Some(true), "{src} infinite");
        assert_eq!(e.is_real(), Some(false), "{src} real");
        assert_eq!(e.is_complex(), Some(false), "{src} complex");
        assert_eq!(e.is_zero(), Some(false), "{src} zero");
        assert_eq!(e.is_positive(), Some(false), "{src} positive");
        assert_eq!(e.is_negative(), Some(false), "{src} negative");
        assert_eq!(e.query(Props::INTEGER), Some(false), "{src} integer");
    }
    assert_eq!(parse(&ctx, "oo").is_positive(), Some(true));
    assert_eq!(parse(&ctx, "-oo").is_negative(), Some(true));
}

/// A symbol whose assumptions pin it to one value is that value: an even
/// prime is 2, a symbol declared zero (or nonnegative and not positive) is
/// 0, a positive integer that is neither prime nor composite is 1.  The
/// assumptions decide the expressions in it and the zero tests substitute
/// it: up to 0.53 `(⌈n⌉ − n)/(n − 2)` was 0 and `(x + 3)(n² − 4)/((n² −
/// 4)(y + 1))` was `(x + 3)/(y + 1)` for an even prime `n` (both `0/0`;
/// SymPy 1.14 gives 0 and the quotient too), `z/z` was 1 for `z` declared
/// zero, and `(n − 2).is_zero()` was `None`.
///
/// Oracle: `n = 2` is the only even prime, so each is the expression at
/// `n = 2`: `0/0`, `(x + 3)·0/(0·(y + 1))`, `(2! − 2)(y + 1)/((2! − 2)e^y)`
/// are `nan`; `sign(0) = 0`, `2 − 3 < 0`.
#[test]
fn a_symbol_pinned_to_a_value_is_that_value() {
    let ctx = Context::new();
    ctx.symbol_with("n", &[Assumption::Even, Assumption::Prime])
        .unwrap();
    ctx.symbol_with("z", &[Assumption::Zero]).unwrap();
    ctx.symbol_with(
        "m",
        &[
            Assumption::Integer,
            Assumption::Positive,
            Assumption::NotPrime,
            Assumption::NotComposite,
        ],
    )
    .unwrap();
    ctx.symbol_with("u", &[Assumption::NonNegative, Assumption::NotPositive])
        .unwrap();
    for src in [
        "(ceiling(n) - n)/(n - 2)",
        "(x + 3)*(n^2 - 4)/((n^2 - 4)*(y + 1))",
        "(factorial(n) - 2)*(y + 1)/((factorial(n) - 2)*exp(y))",
        "((-1)^n - 1)/(n^2 - 4 + cos(r)^2 + sin(r)^2 - 1) + y",
        "z/z",
        "0/z",
        "(x + 3)*z/(z*(y + 1))",
        "(m - 1)/(m - 1)",
    ] {
        assert_eq!(parse(&ctx, src).to_string(), "nan", "{src}");
    }
    assert_eq!(parse(&ctx, "n - 2").is_zero(), Some(true));
    assert_eq!(parse(&ctx, "n^2 - 4").is_zero(), Some(true));
    assert_eq!(parse(&ctx, "n - 3").is_negative(), Some(true));
    assert_eq!(parse(&ctx, "sign(n - 2)").to_string(), "0");
    assert_eq!(parse(&ctx, "m - 1").is_zero(), Some(true));
    assert_eq!(parse(&ctx, "u").is_zero(), Some(true));
    // `cancel` reads `n` as an indeterminate: the denominator it divides
    // out is decided with `n = 2` (up to 0.53 the quotient was 0).
    let x = ctx.symbol("x");
    let q = parse(&ctx, "((x + 1)^2 - x^2 - 2*x - 1)/((n - 2)*(x + 2))");
    assert_eq!(q.cancel(&x).to_string(), "nan");
    // Contradictions are still refused when declared.
    assert!(
        ctx.symbol_with("w", &[Assumption::Positive, Assumption::NonPositive])
            .is_err()
    );
}

/// Functions without a value at an infinite argument make `0·f` `nan`:
/// `ψ(−∞)` meets the poles, `ζ(i·∞)` grows without a limit, `H(i·∞)` has no
/// value off the real axis.  Up to 0.53 the three products were 0.
///
/// Oracle: `ψ(x) = ψ(1 − x) − π·cot(πx)` (DLMF 5.5.4), `|ζ(it)| =
/// |χ(it)|·|ζ(1 − it)|` with `|χ(it)| ~ (t/2π)^{1/2}` (mpmath `zeta(10**6*1j)`
/// = −364.5 + 257.2i at 10³ … growing), `H` defined on ℝ (SymPy 1.14:
/// `Heaviside(I)` raises).
#[test]
fn zero_times_a_function_without_a_value_at_infinity_is_nan() {
    let ctx = Context::new();
    for src in [
        "0*digamma(-oo)",
        "0*zeta(I*oo)",
        "0*Heaviside(I*oo)",
        "0*DiracDelta(-I*oo)",
        "0*dirichlet_eta(I*oo)",
        "0*erfinv(oo)",
    ] {
        assert_eq!(parse(&ctx, src).to_string(), "nan", "{src}");
    }
    assert_eq!(parse(&ctx, "0*zeta(oo + 1)").to_string(), "0");
}
