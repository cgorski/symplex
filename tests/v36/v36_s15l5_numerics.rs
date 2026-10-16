//! Tests of session 15 (s15l5): the numerics statistics depends on.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;
use symplex::stats::Distribution;

/// The value dump of the nightly statistics oracle (`scripts/stats_oracle/`).
#[path = "v36_s15l5_oracle.rs"]
mod oracle;

/// The significant digits and the decimal exponent of a decimal string
/// (`"-1.25e-3"` → `("125", -3)`).
fn digits_of(s: &str) -> (String, i64) {
    let s = s.trim_start_matches('-');
    let (mant, exp) = s
        .split_once('e')
        .map_or((s, 0), |(m, e)| (m, e.parse().unwrap_or(0)));
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let all = format!("{int}{frac}");
    let lead = all.len() - all.trim_start_matches('0').len();
    let sig = all.trim_start_matches('0').to_string();
    (sig, exp + int.len() as i64 - 1 - lead as i64)
}

/// `got` agrees with the reference `want` in its first `n` significant digits.
fn assert_digits(got: &str, want: &str, n: usize) {
    let (g, ge) = digits_of(got);
    let (w, we) = digits_of(want);
    assert!(
        ge == we && g.len() >= n && w.len() >= n && g[..n] == w[..n],
        "{got} does not agree with {want} to {n} digits"
    );
}

/// `B(1347/500, 6.062·10¹⁸)` was refused (`PrecisionExhausted`, "achieved
/// 0"): the `Beta` node formed `Γ(a)Γ(b)/Γ(a + b)`, and `Γ(b)`, `Γ(a + b)`
/// both overflow the exponent range.  mpmath at 50 and 80 digits:
/// `beta(mpf(1347)/500, 6062*10**15) = 3.8583366123022995378109578543958293e-51`.
#[test]
fn beta_with_a_huge_shape_is_evaluated() {
    let ctx = Context::new();
    let b = ctx
        .rational(1347, 500)
        .beta(&ctx.parse("6062000000000000000").unwrap());
    assert_digits(
        &b.eval_decimal(30).unwrap(),
        "3.8583366123022995378109578543958293e-51",
        28,
    );
    let f = b.eval_f64().unwrap();
    assert!((f / 3.858_336_612_302_299_5e-51 - 1.0).abs() < 1e-15, "{f}");
}

/// The density of a Student t with `ν = 10⁹` was refused at every point:
/// its normalisation `B(ν/2, ½)` overflowed through `Γ(ν/2)`.  mpmath:
/// `exp(loggamma((n+1)/2) - loggamma(n/2))/sqrt(n*pi)*(1+1/n)**(-(n+1)/2)`
/// at `n = 10⁹`, `x = 1`: `0.24197072439815798758866908562967725`, and
/// `beta(1/2, n/2) = 7.9266545971936856757407709927036212e-5`.
#[test]
fn student_t_density_with_a_billion_degrees_of_freedom() {
    let ctx = Context::new();
    let t = Distribution::student_t(ctx.parse("10^9").unwrap());
    let pdf = t.density(&ctx.int(1));
    assert_digits(
        &pdf.eval_decimal(30).unwrap(),
        "0.24197072439815798758866908562967725",
        28,
    );
    let b = ctx.rational(1, 2).beta(&ctx.parse("10^9/2").unwrap());
    assert_digits(
        &b.eval_decimal(30).unwrap(),
        "7.9266545971936856757407709927036212e-5",
        28,
    );
}

/// The survival function of `Gamma(10⁸, 1)` two standard deviations above
/// its mean, `uppergamma(s, x)/Γ(s)`, was refused: the prefactor `xˢe⁻ˣ`
/// of Legendre's fraction overflowed and the infinity went on
/// (`PrecisionExhausted`); likewise the CDF of a χ² with `2·10⁸` degrees of
/// freedom there (`lowergamma(s, x)/Γ(s)`, "overflows the exponent range").
/// Both incomplete gamma functions and `Γ(s)` are now held scaled.  mpmath
/// at 45 and 70 digits (Legendre's fraction with `exp(s ln x − x −
/// loggamma(s))`): `Q(10⁸, 10⁸ + 2·10⁴) = 0.0227555307748552026062488263158625775`,
/// `P = 0.977244469225144797393751173684137423`.
#[test]
fn gamma_and_chi_squared_tails_at_shapes_beyond_the_range_of_gamma() {
    let ctx = Context::new();
    let s = ctx.parse("10^8").unwrap();
    let x = ctx.parse("10^8 + 20000").unwrap();
    let sf = Distribution::gamma(s, ctx.int(1)).sf(&x);
    assert_digits(
        &sf.eval_decimal(30).unwrap(),
        "0.0227555307748552026062488263158625775",
        28,
    );
    let f = sf.eval_f64().unwrap();
    assert!((f / 0.022_755_530_774_855_2 - 1.0).abs() < 1e-15, "{f}");
    let chi2 = Distribution::chi_squared(ctx.parse("2*10^8").unwrap());
    let cdf = chi2.cdf(&ctx.parse("2*10^8 + 40000").unwrap());
    assert_digits(
        &cdf.eval_decimal(30).unwrap(),
        "0.977244469225144797393751173684137423",
        28,
    );
}

/// `uppergamma(591978335, 592100000)/Γ(591978335)` (five standard deviations
/// above the mean) was `PrecisionExhausted`: the infinite value of the
/// overflowing `uppergamma` went on.  mpmath at 45 and 70 digits:
/// `2.864087892772782262941758858285e-7`.
#[test]
fn regularized_upper_incomplete_gamma_in_a_far_tail_of_a_huge_shape() {
    let ctx = Context::new();
    let q = ctx
        .parse("uppergamma(591978335, 592100000)/gamma(591978335)")
        .unwrap();
    assert_digits(
        &q.eval_decimal(30).unwrap(),
        "2.864087892772782262941758858285e-7",
        28,
    );
}

/// `binomial(k + r − 1, k)` with `Γ(n + 1)` and `Γ(n − k + 1)` beyond the
/// exponent range and a non-integer `k`-complement was NaN, refused with
/// `PrecisionExhausted` — and with it the mass function of
/// `NegativeBinomial(10⁷/999999, 10⁻⁶)` at `k = 1000000100` (found by the
/// nightly oracle's seed set).  mpmath at 50 and 80 digits, from `loggamma`:
/// `C = 2.7562435999474308743383637419439216e+75`, the mass
/// `1.3980255810797926275188088167645289e-419`.
#[test]
fn binomial_coefficient_of_huge_non_integer_arguments() {
    let ctx = Context::new();
    let r = ctx.rational(10_000_000, 999_999);
    let k = ctx.int(1_000_000_100);
    let c = (&k + &r - ctx.one()).binomial(&k);
    assert_digits(
        &c.eval_decimal(30).unwrap(),
        "2.7562435999474308743383637419439216e+75",
        28,
    );
    let nb = Distribution::negative_binomial(r, ctx.rational(1, 1_000_000));
    let pmf = nb.density(&k);
    assert_digits(
        &pmf.eval_decimal(30).unwrap(),
        "1.3980255810797926275188088167645289e-419",
        28,
    );
}
