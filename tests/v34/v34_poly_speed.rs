//! Slowness in the polynomial layer with large coefficients (the
//! construction-path hunt after 0.35 and the hand-off items): gcds,
//! cancellation, factorisation, real-root counting, extended gcds and
//! rational integration.  Every call must finish well within its limit with
//! the value it had before; the reference values are SymPy 1.14's
//! (`target/scratch/pp/py/*.py` in the work tree that produced them).

use std::time::{Duration, Instant};

use num_bigint::BigInt;
use num_traits::Pow;
use symplex::prelude::*;

/// Generous for a debug build on a loaded CI machine: the release times
/// are milliseconds.
const LIMIT: Duration = Duration::from_secs(5);

fn timed<T>(what: &str, f: impl FnOnce() -> T) -> T {
    let t0 = Instant::now();
    let r = f();
    let dt = t0.elapsed();
    assert!(dt < LIMIT, "{what} took {dt:?}");
    r
}

/// `7^k + c`, a deterministic stand-in for a random huge integer.
fn big(k: u32, c: i64) -> BigInt {
    Pow::pow(BigInt::from(7), k) + c
}

/// `Σ_{k=1}^{10} 1/(x + n_k)` with 3,000-digit `n_k`.
fn fraction_sum(ctx: &Context) -> (Ex, Vec<BigInt>) {
    let x = ctx.symbol("x");
    let ns: Vec<BigInt> = (1..=10).map(|k| big(3550 + k, 11 * i64::from(k))).collect();
    let mut sum = ctx.int(0);
    for n in &ns {
        sum = &sum + &(&x + &ctx.from_bigint(n.clone())).powi(-1);
    }
    (sum, ns)
}

/// The value of the fraction sum at `x = 1/3`, exactly.
fn fraction_sum_at_third(ctx: &Context, ns: &[BigInt]) -> Ex {
    let third = ctx.rational(1, 3);
    let mut v = ctx.int(0);
    for n in ns {
        v = &v + &(&third + &ctx.from_bigint(n.clone())).powi(-1);
    }
    v
}

/// Checks a single fraction `p/q` equal to the fraction sum: `deg p = 9`
/// with leading coefficient 10, `deg q = 10`, and the value at `1/3`.
/// SymPy: `cancel(Σ 1/(x + n_k))` is `(10*x**9 + …)/(x**10 + …)`, the
/// numerator and denominator coprime (`gcd` → 1).
fn check_single_fraction(ctx: &Context, r: &Ex, ns: &[BigInt]) {
    let x = ctx.symbol("x");
    let (p, q) = r.as_numer_denom();
    assert_eq!(p.degree(&x), Some(9));
    assert_eq!(q.degree(&x), Some(10));
    assert_eq!(p.coeff(&x, 9), Some(ctx.int(10)));
    let third = ctx.rational(1, 3);
    assert_eq!(r.subs(&x, &third), fraction_sum_at_third(ctx, ns));
}

/// Before: 15.7 s of CPU in a release build (43 s with 5,000-digit
/// constants) — the heuristic gcd of the numerator and denominator
/// evaluated them by multiplying reduced fractions (a binary gcd against
/// the denominator `1` per product, on 300,000-digit values); now 20 ms: an
/// image modulo one word prime proves them coprime.
#[test]
fn ratsimp_of_ten_fractions_with_3000_digit_constants() {
    let ctx = Context::new();
    let (sum, ns) = fraction_sum(&ctx);
    let r = timed("ratsimp", || sum.ratsimp());
    check_single_fraction(&ctx, &r, &ns);
}

/// Before: 0.55 s (release; 1.5 s with 5,000-digit constants), in the
/// rational divisions of `together`'s least common multiple (a binary gcd
/// per coefficient operation) and the remainder sequence of its gcds; now
/// 60 ms with the modular gcd and gcd-free integer arithmetic.
#[test]
fn together_of_ten_fractions_with_3000_digit_constants() {
    let ctx = Context::new();
    let (sum, ns) = fraction_sum(&ctx);
    let r = timed("together", || sum.together());
    check_single_fraction(&ctx, &r, &ns);
}

/// Before: 2.8 s (release) for the heuristic gcd in four variables of
/// degrees (10, 19, 9, 12) (SymPy: 0.06 s); now 57 ms (integer heuristic
/// with Lehmer's integer gcd).  SymPy 1.14 `gcd(expand(A*B), expand(A*C))`
/// = `y**2*z*A` (`B` and `C` share the monomial `y²z`).
#[test]
fn gcd_of_four_variable_products_with_30_digit_coefficients() {
    let ctx = Context::new();
    let a = ctx
        .parse(
            "890458120250254458077947439060*w^4*x^6*y*z^3 \
             + 442156157520785028051943080131*w*x^8*y^3*z^4 \
             - 686114261398001756708667951654*w*x^8*y^2*z^2 \
             + 577620180229282151538606202949*x^6*y^3*z^5 \
             + 197825952881374641344148225481*x^2*y",
        )
        .unwrap();
    let b = ctx
        .parse(
            "-452790671812125719356157710742*w^6*x^8*y^6*z^7 \
             + 795141664136704373756445434863*w^6*x*y^2*z^7 \
             + 323774601736514287423735220895*w^2*y^3*z^2 \
             + 962175701930452328338016738988*x^11*y^3*z^4 \
             - 325545253970444884676426075023*x^7*y^2*z",
        )
        .unwrap();
    let c = ctx
        .parse(
            "543153413127262130835948510224*w^4*x^6*y^3*z^11 \
             - 402345464725708604760695339209*w^4*x^3*y^4*z^3 \
             - 652484924896491184332767815530*w^4*x*y^5*z^10 \
             + 147016042136910700499172050351*w^2*x*y^4*z^2 \
             + 936723407987457194988969871697*x^5*y^5*z^4",
        )
        .unwrap();
    let f = (&a * &b).expand();
    let g = (&a * &c).expand();
    let h = timed("gcd_all", || f.gcd_all(&g)).unwrap();
    let y2z = &ctx.symbol("y").powi(2) * &ctx.symbol("z");
    assert_eq!(h, (&y2z * &a).expand());
}

/// Before: 3.9 s (release): linear Hensel lifting, one step per power of
/// the prime up to the Mignotte bound (`3^21000`); now quadratic lifting.
/// SymPy 1.14 `factor_list(expand(f))` lists the three factors.
#[test]
fn factor_of_a_product_with_5000_digit_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (a, b, c) = (big(5900, 1), big(5901, -2), big(5902, 3));
    let product = &(&(&x + &ctx.from_bigint(a)) * &(&x + &ctx.from_bigint(b)))
        * &(&x.powi(2) + &ctx.from_bigint(c));
    let f = product.expand();
    let r = timed("factor", || f.factor(&x));
    assert_eq!(r, product);
}

/// Before: 0.53 s (release; 1.3 s with 5,000-digit coefficients) — each
/// of the four sign queries isolated every real root by Descartes bisection
/// from the root bound `2^12000` down to the roots near 1; now the half-line
/// is cut around the root magnitudes first (4 ms).  SymPy 1.14
/// `Poly(f).count_roots()` = 2 (real roots in `(0, 1)` and near `7^3551`),
/// so the sign is not constant: no fact (SymPy `f.is_positive` with a real
/// `x` → `None`; `((x**2 - a)**2 + 1).is_positive` → `True`).
#[test]
fn sign_of_a_quartic_with_3000_digit_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let f = &(&(&(&x.powi(4) - &(&ctx.from_bigint(big(3551, 0)) * &x.powi(3)))
        + &(&ctx.from_bigint(big(3550, 1)) * &x.powi(2)))
        - &(&ctx.from_bigint(big(3550, 3)) * &x))
        + &ctx.from_bigint(big(3550, 5));
    assert_eq!(timed("is_positive", || f.is_positive()), None);
    // A positive definite one is decided: (x² − a)² + 1.
    let g = &(&x.powi(2) - &ctx.from_bigint(big(3550, 0))).powi(2) + &ctx.int(1);
    assert_eq!(
        timed("is_positive", || g.expand().is_positive()),
        Some(true)
    );
}

/// Before: 2.0 s (release; 4.5 s at degrees 12 and 11) — the cofactors'
/// 1,000-digit fractions were interned with a hash that walked their
/// continued fractions (Euclid on the digits); now 30 ms with a
/// linear-time hash.  The Bézout identity holds and the
/// cofactors have the minimal degrees (SymPy `gcdex` returns the same
/// unique pair).
#[test]
fn extended_gcd_with_30_digit_rational_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut digit30 = || -> BigInt {
        let mut v = BigInt::from(0);
        for _ in 0..30 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            v = v * 10 + BigInt::from(state % 10);
        }
        v + 1
    };
    let mut poly = |deg: usize| -> Ex {
        let mut p = ctx.int(0);
        for i in 0..=deg {
            let (n, d) = (digit30(), digit30());
            let sign = if i % 3 == 1 { -1 } else { 1 };
            let c = &ctx.from_bigint(n * sign) / &ctx.from_bigint(d);
            p = &p + &(&c * &x.powi(i as i64));
        }
        p
    };
    let f = poly(10);
    let g = poly(9);
    let e = timed("poly_gcdex", || f.poly_gcdex(&g, &x)).unwrap();
    assert_eq!(e.gcd, ctx.int(1));
    assert!(e.x.degree(&x).unwrap_or(0) < 9 && e.y.degree(&x).unwrap_or(0) < 10);
    // x·f + y·g = 1, checked at two points (the polynomial identity has
    // degree 18, its coefficients thousand-digit fractions).
    for t in [ctx.int(2), ctx.rational(-1, 3)] {
        let at = |p: &Ex| p.subs(&x, &t);
        assert_eq!(&(&at(&e.x) * &at(&f)) + &(&at(&e.y) * &at(&g)), ctx.int(1));
    }
}

/// Before: 3.8 s (release), in Hermite reduction's rational divisions and
/// gcds (binary gcds per coefficient operation, a remainder sequence of
/// degree 21 with 600-digit coefficients); now 0.16 s.  The
/// antiderivative differentiates back at `x = 2`.
#[test]
fn rational_integrand_with_30_digit_parameters() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx
        .parse(
            "(394872958374659283746592837465 + 718293847561029384756102938475*x)^10\
             /(918273645546372819283746556473 + 564738291029384756473829102938*x)^21",
        )
        .unwrap();
    let big_f = timed("integrate", || f.integrate(&x));
    assert!(!big_f.has_unevaluated());
    let two = ctx.int(2);
    assert_eq!(big_f.diff(&x).subs(&x, &two), f.subs(&x, &two));
}
