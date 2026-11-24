//! Tests of session 17, wave 16 (s15r3): aborts, panics and quadratic loops at huge sizes.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use std::time::Duration;

use symplex::prelude::*;
use symplex::stats::Distribution;

use crate::perf_assert::{CpuTimer, assert_fast};

// ── R3.1: FormalPowerSeries::coefficients ──────────────────────────────

/// Before (0.58): `coefficients(n)` collected `n` entries whatever `n` was;
/// `sin(x).fps_maclaurin(x).coefficients(1 << 40)` asked the allocator for
/// 17,592,186,044,416 bytes and the process aborted ("memory allocation of
/// … bytes failed"), and there was no `try_` twin.  Now a length beyond
/// what a series lists (2¹⁴ + 1) is refused at once: the plain function
/// returns the documented empty list and `try_coefficients` an error.
/// Oracle for the kept values: the Maclaurin coefficients of `sin`.
#[test]
fn fps_coefficients_of_a_huge_length_are_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = x.sin().fps_maclaurin(&x);
    for n in [1usize << 20, 1 << 31, 1 << 40, usize::MAX] {
        let listed = assert_fast!(Duration::from_secs(1), s.coefficients(n));
        assert!(listed.is_empty(), "n = {n}");
        let tried = assert_fast!(Duration::from_secs(1), s.try_coefficients(n));
        assert!(
            matches!(tried, Err(SymplexError::ComputationFailed { .. })),
            "n = {n}"
        );
    }
    let show = |c: Vec<Ex>| c.iter().map(|c| c.to_string()).collect::<Vec<_>>();
    let want = ["0", "1", "0", "-1/6", "0", "1/120"];
    assert_eq!(show(s.coefficients(6)), want);
    assert_eq!(show(s.try_coefficients(6).unwrap()), want);
    assert!(s.coefficients(0).is_empty());
    assert!(s.try_coefficients(0).unwrap().is_empty());
}

/// `try_coefficients` follows `try_coefficient`: a refused series (`ln x`
/// at 0 has no expansion) is an error for every length, and the plain list
/// keeps the stand-ins.  A polynomial lists all of its own coefficients,
/// also beyond the 2¹⁴ + 1 a computed series lists (before and now), and
/// refuses beyond its length.
#[test]
fn fps_try_coefficients_follows_try_coefficient() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let refused = x.ln().fps_maclaurin(&x);
    assert!(refused.is_refused());
    assert!(refused.try_coefficients(3).is_err());
    assert_eq!(refused.coefficients(3).len(), 3);

    let own = (1usize << 14) + 5;
    let coeffs: Vec<Ex> = (0..own).map(|k| ctx.int((k % 7) as i64)).collect();
    let p = FormalPowerSeries::from_coefficients(&x, &ctx.int(0), &coeffs);
    let listed = p.try_coefficients(own).unwrap();
    assert_eq!(listed.len(), own);
    assert_eq!(listed[own - 1].to_string(), ((own - 1) % 7).to_string());
    assert_eq!(p.coefficients(own).len(), own);
    assert!(p.try_coefficients(own + 1).is_err());
    assert!(p.coefficients(own + 1).is_empty());
}

// ── R3.3: verification of a factorization by exponent arithmetic ───────

/// Before (0.58): `factor_multivariate(x^N·y + x^N)` verified its answer by
/// multiplying the product by the factor `x` once per unit of the
/// multiplicity `N` of the monomial content: 5.1 s at `N = 2²⁴`, a quarter
/// of an hour at `2³¹` (measured: > 20 s, extrapolated).  Now the monomial
/// factors are verified by one addition of exponents: 0.000 s for both.
/// Oracle: `x^N·y + x^N = x^N·(y + 1)`, checked on the exponents.
#[test]
fn multivariate_factor_with_a_huge_monomial_content_is_immediate() {
    use symplex::factor_zassenhaus::factor_multivariate;
    use symplex::linprog::qi;
    let y: MultiPoly = MultiPoly::var(2, 1);
    let y1 = y.add(&MultiPoly::from_int(2, 1));
    for n in [5u32, 1 << 24, 1 << 31, u32::MAX] {
        let big: MultiPoly = MultiPoly::monomial(qi(1), vec![n, 0]);
        let f = big.mul(&y).add(&big);
        let (content, factors) = assert_fast!(Duration::from_secs(2), factor_multivariate(&f))
            .unwrap_or_else(|| panic!("no factorization for n = {n}"));
        assert_eq!(content, qi(1), "n = {n}");
        let x: MultiPoly = MultiPoly::var(2, 0);
        assert_eq!(factors.len(), 2, "n = {n}: {factors:?}");
        assert!(factors.contains(&(x, n)), "n = {n}: {factors:?}");
        assert!(factors.contains(&(y1.clone(), 1)), "n = {n}: {factors:?}");
    }
    // A small case still multiplies back to the input, powers included:
    // x²·y·(x + y)³.
    let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    let s = x.add(&y);
    let f = x.mul(&x).mul(&y).mul(&s.pow(3));
    let (content, factors) = factor_multivariate(&f).expect("factors");
    let mut back = MultiPoly::constant(2, content);
    for (g, m) in &factors {
        back = back.mul(&g.pow(*m));
    }
    assert_eq!(back, f);
    assert!(factors.contains(&(s, 3)), "{factors:?}");
}

/// Before (0.58): the univariate `factor_zassenhaus(x^N + x^(N−1))` took
/// 0.8 s at `N = 4000` and 25 s at `N = 20,000` (release): the power of `x`
/// went through the square-free decomposition (divisions of the full
/// degree) and was multiplied back once per unit of its multiplicity.  Now
/// it is read off the coefficients and verified by a shift: 1 ms at
/// 20,000, 0.1 s at 2²⁰.  Oracle: `x^N + x^(N−1) = x^(N−1)·(x + 1)`.
#[test]
fn univariate_factor_with_a_huge_power_of_x_is_immediate() {
    use symplex::factor_zassenhaus::{Poly as DensePoly, factor_zassenhaus_with_content};
    use symplex::linprog::qi;
    let n = 20_000usize;
    let mut coeffs = vec![qi(0); n + 1];
    coeffs[n] = qi(3);
    coeffs[n - 1] = qi(3);
    let f = DensePoly::from_coeffs(coeffs);
    let (content, factors) =
        assert_fast!(Duration::from_secs(5), factor_zassenhaus_with_content(&f));
    assert_eq!(content, qi(3));
    let x = DensePoly::from_coeffs(vec![qi(0), qi(1)]);
    let x1 = DensePoly::from_coeffs(vec![qi(1), qi(1)]);
    assert_eq!(factors, vec![(x.clone(), 19_999), (x1.clone(), 1)]);
    // Powers of x next to repeated factors: x³·(x + 1)²·(x − 2).
    let x2 = DensePoly::from_coeffs(vec![qi(-2), qi(1)]);
    let g = &(&(&(&(&x * &x) * &x) * &x1) * &x1) * &x2;
    let (c, fs) = factor_zassenhaus_with_content(&g);
    assert_eq!(c, qi(1));
    assert_eq!(fs, vec![(x2, 1), (x, 3), (x1, 2)]);
}

// ── R3.4: images of `transformed` beyond and at the edge of the doubles ─

fn atoms(d: &Distribution) -> Vec<Ex> {
    d.support().as_points().expect("a finite table")
}

/// Before (0.58): an image beyond the `f64` range had no key, so
/// `Distribution::transformed` compared it with every entry by `equals`:
/// `sinh X` of a 1000-sided die (290 images beyond the range) took 3.7 s in
/// release, a 2000-sided die 25 s, a 5000-sided one more than 50 s and
/// 1.4 GB.  Now such an image is keyed by `ln |image|` (0.2 s for 5000
/// sides).  Oracle: `sinh` is strictly increasing, so the 800 images are
/// distinct, each with the mass 1/800; `cosh` is even, so the images of
/// `−750..=750` merge in pairs (`2/1501` each, `1/1501` for `cosh 0 = 1`)
/// although those beyond `|k| = 710` have no `f64`.
#[test]
fn transformed_images_beyond_the_doubles_are_keyed() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let timer = CpuTimer::start();
    let die = Distribution::die(ctx.int(800));
    let s = die.transformed(&x, &x.sinh())?;
    let values = atoms(&s);
    assert_eq!(values.len(), 800);
    let mass = ctx.rational(1, 800);
    for k in [1i64, 709, 710, 711, 800] {
        assert_eq!(s.density(&ctx.int(k).sinh()).eval(), mass, "k = {k}");
    }
    let e = die.transformed(&x, &x.exp())?;
    assert_eq!(atoms(&e).len(), 800);
    assert_eq!(e.density(&ctx.int(800).exp()).eval(), mass);

    let sym = Distribution::discrete_uniform(ctx.int(-750), ctx.int(750));
    let c = sym.transformed(&x, &x.cosh())?;
    assert_eq!(atoms(&c).len(), 751);
    assert_eq!(c.density(&ctx.int(0).cosh()).eval(), ctx.rational(1, 1501));
    for k in [1i64, 700, 711, 750] {
        assert_eq!(
            c.density(&ctx.int(k).cosh()).eval(),
            ctx.rational(2, 1501),
            "k = {k}"
        );
    }
    timer.assert_within(Duration::from_secs(60), "three transformed tables");
    Ok(())
}

/// Before (0.58): `tanh k` is the double `1.0` from `k = 19` on, so the
/// images of a die shared one key and were compared pairwise by `equals`:
/// 15.7 s for 300 sides and 24.8 s for 350 (release).  Now two images that
/// share a double are told apart by their residuals `image − double`
/// (`tanh k − 1 = −2/(e^{2k} + 1)`): 8 ms for 350 sides.  Oracle: `tanh` is
/// strictly increasing (300 distinct images of mass 1/300), and `tanh² `
/// is even: the images of `−60..=60` merge in pairs.
#[test]
fn transformed_images_sharing_a_double_are_told_apart() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let timer = CpuTimer::start();
    let t = Distribution::die(ctx.int(300)).transformed(&x, &x.tanh())?;
    assert_eq!(atoms(&t).len(), 300);
    let mass = ctx.rational(1, 300);
    for k in [1i64, 18, 19, 20, 299, 300] {
        assert_eq!(t.density(&ctx.int(k).tanh()).eval(), mass, "k = {k}");
    }
    let sym = Distribution::discrete_uniform(ctx.int(-60), ctx.int(60));
    let sq = sym.transformed(&x, &x.tanh().pow(&ctx.int(2)))?;
    assert_eq!(atoms(&sq).len(), 61);
    for k in [1i64, 19, 30, 60] {
        assert_eq!(
            sq.density(&ctx.int(k).tanh().pow(&ctx.int(2))).eval(),
            ctx.rational(2, 121),
            "k = {k}"
        );
    }
    timer.assert_within(Duration::from_secs(60), "two transformed tables");
    Ok(())
}

// ── B-C06-03: the stand-in coefficients of a refused series ────────────

/// Before (0.58): the coefficients of a series that does not exist were
/// Taylor stand-ins — `fps(1/x + x^a)` and `fps(sqrt(x^3))` at 0 gave
/// `Subs(1/x + x^a, x, 0)`, `Subs(Derivative(1/x + x^a, x), x, 0)`,
/// `1/2*Subs(Derivative(Derivative(…), x), x, 0)` — which name derivatives
/// at a point where the function has none.  Now the stand-in is the
/// residue `Res_{x=0} f·x^−(k+1)` (Cauchy's formula for a Laurent
/// coefficient, unevaluated); the series is refused as before
/// (`is_refused`, `Err` from the fallible accessors, a formal `Series`).
#[test]
fn refused_series_coefficients_are_not_taylor_stand_ins() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for f in ["1/x + x^a", "sqrt(x^3)", "abs(x)", "log(x)", "exp(1/x)"] {
        let s = ctx.parse(f).unwrap().fps_maclaurin(&x);
        assert!(s.is_refused(), "{f}");
        assert!(s.try_coefficient(2).is_err(), "{f}");
        assert!(s.try_coefficients(3).is_err(), "{f}");
        assert!(s.truncate(3).to_string().starts_with("Series("), "{f}");
        for (k, c) in s.coefficients(4).iter().enumerate() {
            let c = c.to_string();
            assert!(c.starts_with("Residue("), "{f}, k = {k}: {c}");
            assert!(
                !c.contains("Derivative") && !c.contains("Subs") && !c.contains("nan"),
                "{f}, k = {k}: {c}"
            );
        }
    }
    let s = ctx.parse("1/x + x^a").unwrap().fps_maclaurin(&x);
    assert_eq!(s.coefficient(0).to_string(), "Residue((1/x + x^a)/x, x, 0)");
    assert_eq!(
        s.coefficient(2).to_string(),
        "Residue(x^(-3)*(1/x + x^a), x, 0)"
    );
}

// ── B-C14-01: Fibonacci, Lucas and primorial numbers of a huge index ───

/// Before (0.58): `fibonacci(2⁴⁰)` (a value of 95 GB), `lucas(2⁴⁰)` and
/// `primorial(2⁴⁰)` (2⁴⁰ calls of `nextprime`) never returned; the caller
/// had to bound the index and there was no `try_` twin.  Now an index
/// beyond `±2²⁷` (the first `5·10⁶` primes) is refused at once: `Err` from
/// `try_fibonacci`, `try_lucas`, `try_primorial`, and the documented `0`
/// (not a value of any of the three at such an index) from the plain
/// functions.  Oracle for the kept values: `F₉₀ = 2880067194370816120`,
/// `L₄₀ = 228826127`, `p₅# = 2310`, `F₋₈ = −21`.
#[test]
fn sequences_of_a_huge_index_are_refused() {
    use symplex::ntheory::{fibonacci, lucas, primorial, try_fibonacci, try_lucas, try_primorial};
    use symplex::num_bigint::BigInt;
    let zero = BigInt::from(0);
    for n in [(1i64 << 27) + 1, 1 << 31, 1 << 40, i64::MAX, -(1i64 << 40)] {
        assert_eq!(assert_fast!(Duration::from_secs(1), fibonacci(n)), zero);
        assert_eq!(assert_fast!(Duration::from_secs(1), lucas(n)), zero);
        assert!(try_fibonacci(n).is_err() && try_lucas(n).is_err(), "{n}");
    }
    let beyond_i64 = BigInt::from(u64::MAX) * BigInt::from(u64::MAX);
    assert!(try_fibonacci(beyond_i64.clone()).is_err());
    assert_eq!(lucas(-beyond_i64), zero);
    for n in [5_000_001u64, 1 << 31, 1 << 40, u64::MAX] {
        assert_eq!(assert_fast!(Duration::from_secs(1), primorial(n)), zero);
        assert!(try_primorial(n).is_err(), "{n}");
    }
    assert_eq!(
        try_fibonacci(90).unwrap(),
        BigInt::from(2_880_067_194_370_816_120i64)
    );
    assert_eq!(fibonacci(-8), BigInt::from(-21));
    assert_eq!(try_lucas(40).unwrap(), BigInt::from(228_826_127));
    assert_eq!(try_primorial(5).unwrap(), BigInt::from(2310));
    assert_eq!(primorial(0), BigInt::from(1));
    // The product of the primes up to a bound has the same cap (the
    // 5·10⁶-th prime, 86,028,121): `primorial_up_to(2³¹)` listed 10⁸ primes.
    use symplex::ntheory::{primorial_up_to, try_primorial_up_to};
    for n in [86_028_122u64, 1 << 31, 1 << 40, u64::MAX] {
        assert_eq!(
            assert_fast!(Duration::from_secs(1), primorial_up_to(n)),
            zero
        );
        assert!(try_primorial_up_to(n).is_err(), "{n}");
    }
    assert_eq!(try_primorial_up_to(10).unwrap(), BigInt::from(210));
    assert_eq!(primorial_up_to(1), BigInt::from(1));
    // The cap itself is an index that is computed (not here: 17 s).
    assert!(try_fibonacci(1000).unwrap().bits() == 694);
}

// ── B-C14-02: a cyclic convolution of a huge length ────────────────────

/// Before (0.58): `convolution_cyclic(a, b, n)` allocated its `n` entries
/// first: `n = 2⁴⁰` aborted the process ("memory allocation of
/// 70368744177664 bytes failed") and `n = usize::MAX` panicked ("capacity
/// overflow").  Now a length beyond both the linear length and 2²⁴ is
/// refused: the documented empty vector, and `Err` from
/// `try_convolution_cyclic`.  Oracle (SymPy): `convolution([1, 2, 3],
/// [4, 5, 6], cycle=3) == [31, 31, 28]`, `cycle=6` pads with a zero.
#[test]
fn cyclic_convolution_of_a_huge_length_is_refused() {
    use symplex::discrete::{convolution_cyclic, try_convolution_cyclic};
    use symplex::linprog::qi;
    let q = |v: &[i64]| v.iter().map(|&t| qi(t)).collect::<Vec<_>>();
    let (a, b) = (q(&[1, 2, 3]), q(&[4, 5, 6]));
    for n in [(1usize << 24) + 1, 1 << 31, 1 << 40, usize::MAX] {
        let r = assert_fast!(Duration::from_secs(1), convolution_cyclic(&a, &b, n));
        assert!(r.is_empty(), "n = {n}");
        assert!(matches!(
            try_convolution_cyclic(&a, &b, n),
            Err(SymplexError::ComputationFailed { .. })
        ));
    }
    assert_eq!(convolution_cyclic(&a, &b, 3), q(&[31, 31, 28]));
    assert_eq!(
        try_convolution_cyclic(&a, &b, 6).unwrap(),
        q(&[4, 13, 28, 27, 18, 0])
    );
    assert!(try_convolution_cyclic(&a, &b, 0).unwrap().is_empty());
    assert!(try_convolution_cyclic(&[], &b, 4).unwrap().is_empty());
}

// ── B-C10-01: the Z-transform of a huge power of the index ─────────────

/// Before (0.58): `z_transform(n^k)` applied `−z·d/dz` `k − 1` times
/// whatever `k` was: `n^400` took 2.9 s and 500 MB, `n^2000` and `n^65537`
/// ran past 20 s and 1.8 GB towards an allocation failure.  Now a power of
/// the index beyond 256 is refused at once (`ComputationFailed`), alone or
/// as a factor.  Oracle for a kept value: `Z{n³} = z(z² + 4z + 1)/(z − 1)⁴`,
/// which is `33/8` at `z = 3` (`Σ n³/3ⁿ`).
#[test]
fn z_transform_of_a_huge_power_is_refused() {
    let ctx = Context::new();
    let (n, z) = (ctx.symbol("n"), ctx.symbol("z"));
    for src in [
        "n^65537",
        "n^257",
        "n^4294967296",
        "n^65537*2^n",
        "n^300*sin(n)",
    ] {
        let e = ctx.parse(src).unwrap();
        let r = assert_fast!(Duration::from_secs(1), e.z_transform(&n, &z));
        assert!(
            matches!(r, Err(SymplexError::ComputationFailed { .. })),
            "{src}: {r:?}"
        );
    }
    let cube = ctx.parse("n^3").unwrap().z_transform(&n, &z).unwrap();
    assert_eq!(cube.subs(&z, &ctx.int(3)).simplify(), ctx.rational(33, 8));
    assert!(ctx.parse("n^40").unwrap().z_transform(&n, &z).is_ok());
}

// ── B-C04-03, B-C04-04: gcd, powers and values of huge polynomials ─────

/// Before (0.58): `MultiPoly::gcd(x^(2³¹ − 1) − 1, x² − 1)` left the dense
/// routes (degree beyond 2²⁴) for the pseudo-remainder sequence, whose
/// first division takes 2³⁰ steps: no answer after 20 s.  Now a
/// pseudo-division is refused after 2¹⁸ steps (0.6 s in release):
/// `try_gcd` is `None` and `gcd` its documented refusal value (the gcd of
/// the integer contents, here 1).  Below 2²⁴ nothing changes: `gcd(x^1001
/// − 1, x² − 1) = x − 1`.
#[test]
fn gcd_of_a_polynomial_of_degree_two_billion_is_refused() {
    use symplex::linprog::qi;
    let one: MultiPoly = MultiPoly::from_int(1, 1);
    let binomial = |n: u32| MultiPoly::monomial(qi(1), vec![n]).sub(&one);
    let (huge, square) = (binomial((1 << 31) - 1), binomial(2));
    let timer = CpuTimer::start();
    assert_eq!(MultiPoly::try_gcd(&huge, &square), None);
    assert_eq!(MultiPoly::gcd(&huge, &square), one);
    timer.assert_within(Duration::from_secs(120), "two refused gcds");
    let x: MultiPoly = MultiPoly::var(1, 0);
    assert_eq!(MultiPoly::gcd(&binomial(1001), &square), x.sub(&one));
}

/// Before (0.58): a power had no size check: `Poly::pow(u32::MAX)` of
/// `x + 1` (4·10⁹ terms) ran until the memory was exhausted (no answer
/// after 20 s).  Now a polynomial of two or more terms is raised to at
/// most 2¹⁶ and a coefficient of `b` bits to at most `2²⁷/(b − 1)`:
/// `InvalidArgument` from `Poly::pow`, `None` from `MultiPoly::try_pow`.
/// A monomial keeps its power (`x^(2³² − 1)`), and small powers are
/// unchanged (`(x + 1)³`).
#[test]
fn power_of_a_polynomial_beyond_the_size_bound_is_refused() {
    use symplex::linprog::qi;
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = Poly::new(&(&x + 1), &[&x]).unwrap();
    for n in [(1u32 << 16) + 1, 1 << 31, u32::MAX] {
        let r = assert_fast!(Duration::from_secs(1), p.pow(n));
        assert!(
            matches!(r, Err(SymplexError::InvalidArgument { .. })),
            "n = {n}"
        );
    }
    assert_eq!(
        p.pow(3).unwrap().to_ex().to_string(),
        "x^3 + 3*x^2 + 3*x + 1"
    );
    let mono = Poly::new(&x, &[&x]).unwrap().pow(u32::MAX).unwrap();
    assert_eq!(mono.total_degree(), Some(u32::MAX));

    let mx: MultiPoly = MultiPoly::var(1, 0);
    let s = mx.add(&MultiPoly::from_int(1, 1));
    assert_eq!(s.try_pow(u32::MAX), None);
    assert_eq!(s.try_pow(2), Some(s.mul(&s)));
    // (3x)^(2³¹): the coefficient 3^(2³¹) has 3.4·10⁹ bits.
    let three_x = mx.scale(&qi(3));
    assert_eq!(
        assert_fast!(Duration::from_secs(1), three_x.try_pow(1 << 31)),
        None
    );
    assert_eq!(three_x.try_pow(2), Some(mx.mul(&mx).scale(&qi(9))));
}

/// Before (0.58): `MultiPoly::eval` tabulated every power of the point up
/// to the degree: `x^(2³¹) + 1` at `x = 1` asked for a vector of 68 GB
/// (abort), `x^(10⁶) + 1` at `x = 3` for 10⁶ integers of 100 KB on average
/// (killed at 32 GB).  Now only the powers of the exponents that occur are
/// formed, and a power beyond 2²⁷ bits (2²² in a denominator) is an error.
/// Oracle: `1^n + 1 = 2`; `3^20000 + 1` has `⌊20000·log₂ 3⌋ + 1 = 31700`
/// bits and is `≡ 2 (mod 4)` and `≡ 3^(20000 mod 6) + 1 = 3² + 1 (mod 7)`;
/// `(1/2)^100 + 1 = (2^100 + 1)/2^100`.
#[test]
fn value_of_a_sparse_polynomial_of_huge_degree() {
    use symplex::linprog::qi;
    use symplex::num_bigint::BigInt;
    use symplex::num_rational::Ratio;
    let one: MultiPoly = MultiPoly::from_int(1, 1);
    let p = |n: u32| MultiPoly::monomial(qi(1), vec![n]).add(&one);
    let huge = p(1 << 31);
    assert_eq!(
        assert_fast!(Duration::from_secs(1), huge.eval(&[qi(1)])).unwrap(),
        qi(2)
    );
    assert_eq!(huge.eval(&[qi(-1)]).unwrap(), qi(2));
    assert_eq!(huge.eval(&[qi(0)]).unwrap(), qi(1));
    for v in [qi(2), qi(-3), Ratio::new(BigInt::from(1), BigInt::from(2))] {
        let r = assert_fast!(Duration::from_secs(1), huge.eval(std::slice::from_ref(&v)));
        assert!(
            matches!(r, Err(SymplexError::ComputationFailed { .. })),
            "{v}"
        );
    }
    let v = p(20_000).eval(&[qi(3)]).unwrap();
    assert!(v.is_integer());
    assert_eq!(v.numer().bits(), 31_700);
    assert_eq!(v.numer() % BigInt::from(4), BigInt::from(2));
    assert_eq!(v.numer() % BigInt::from(7), BigInt::from(3));
    let half = Ratio::new(BigInt::from(1), BigInt::from(2));
    let two_100 = BigInt::from(1u8) << 100u32;
    assert_eq!(
        p(100).eval(&[half]).unwrap(),
        Ratio::new(&two_100 + BigInt::from(1), two_100)
    );
    // A dense polynomial is unchanged: (x + 1)^5 at 2/3 is (5/3)^5.
    let x: MultiPoly = MultiPoly::var(1, 0);
    let dense = x.add(&one).pow(5);
    let point = Ratio::new(BigInt::from(2), BigInt::from(3));
    assert_eq!(
        dense.eval(&[point]).unwrap(),
        Ratio::new(BigInt::from(3125), BigInt::from(243))
    );
}

// ── B-C18-01: a deeply nested tree ─────────────────────────────────────

/// Before (0.58): `Context::from_tree` converted a tree by a function that
/// called itself once per level: `sin(sin(…sin(x)…))` nested 100,000 deep
/// overflowed the stack ("thread … has overflowed its stack", an abort).
/// Now the conversion walks an explicit stack.  Oracle: the same
/// expression built by applying `sin` 100,000 times.
#[test]
fn deeply_nested_tree_converts_without_recursion() {
    use symplex::tree::ExprTree;
    let depth = 100_000;
    let ctx = Context::new();
    let mut tree = ExprTree::Symbol { name: "x".into() };
    let mut want = ctx.symbol("x");
    for _ in 0..depth {
        tree = ExprTree::Sin {
            arg: Box::new(tree),
        };
        want = want.sin();
    }
    let got = ctx.from_tree(&tree);
    assert_eq!(got, want);
    // The tree is taken apart level by level (its own drop is recursive).
    while let ExprTree::Sin { arg } = tree {
        tree = *arg;
    }
    // Children in boxes and in lists, one to four of them, are found by
    // the walk: each tree converts back to the expression it was made of.
    let x = ctx.symbol("x");
    let unevaluated = x.pow(&x).exp().integrate(&x);
    let parsed = [
        "x^2 + sin(x)*exp(y) - 3/7",
        "sqrt(x + 1)/(y - 2) + log(x)^3",
        "abs(x - y)*cos(2*x) + z^(x*y)",
    ];
    let exprs: Vec<Ex> = parsed
        .iter()
        .map(|src| ctx.parse(src).unwrap())
        .chain([unevaluated])
        .collect();
    for e in exprs {
        let src = e.to_string();
        assert_eq!(ctx.from_tree(&e.to_tree()), e, "{src}");
    }
}
