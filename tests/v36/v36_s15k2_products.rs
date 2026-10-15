//! Tests of session 15 (s15k2): finite products, the Gosper API, sums at the boundary.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

/// `expr` at the integers `n` of `ns`, as strings.
fn values_at(e: &Ex, n: &Ex, ns: &[i64]) -> Vec<String> {
    ns.iter()
        .map(|&v| e.subs_i64(n, v).eval().to_string())
        .collect()
}

fn natural(ctx: &Context, name: &str) -> Ex {
    ctx.symbol_with(name, &[Assumption::Integer, Assumption::NonNegative])
        .unwrap()
}

/// Reversed limits were the empty product (`Π_{k=5}^{2} k = 1`, also through
/// `eval` of a formal `Product`).  D13: Karr's convention, `Π_{k=a}^{b} =
/// 1/Π_{k=b+1}^{a−1}` for `b < a − 1` (SymPy's), which keeps
/// `Π_{a}^{b}·Π_{b+1}^{c} = Π_{a}^{c}` for all integers.  Oracle: exact
/// partial products, `1/(3·4) = 1/12`, `1/((1/2)·(1/3)) = 6`; `Π_{k=3}^{2}`
/// is empty; `Π_{k=3}^{−2} k = 1/((−1)·0·1·2)` is `1/0`.
#[test]
fn reversed_products_follow_karrs_convention() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    assert_eq!(
        k.product_over(&k, &ctx.int(5), &ctx.int(2)).to_string(),
        "1/12"
    );
    assert_eq!(
        k.product_over(&k, &ctx.int(3), &ctx.int(2)).to_string(),
        "1"
    );
    let inv = ctx.int(1) / &k;
    let formal = Ex::symbolic_product(&inv, &k, &ctx.int(4), &ctx.int(1));
    assert_eq!(formal.eval().to_string(), "6");
    assert_eq!(
        Ex::symbolic_product(&k, &k, &ctx.int(5), &ctx.int(2))
            .eval_decimal(10)
            .unwrap(),
        "0.08333333333"
    );
    // A factor 0 in the reflected range: 1/0.
    assert_eq!(
        k.product_over(&k, &ctx.int(3), &ctx.int(-2)).to_string(),
        "zoo"
    );
    // Additivity across the reversed range: Π_{1}^{-3}·Π_{-2}^{4} = Π_{1}^{4}.
    let f = &k + ctx.rational(1, 2);
    let a = f.product_over(&k, &ctx.int(1), &ctx.int(-3));
    let b = f.product_over(&k, &ctx.int(-2), &ctx.int(4));
    let c = f.product_over(&k, &ctx.int(1), &ctx.int(4));
    assert_eq!((&a * &b).eval().to_string(), c.to_string());
}

/// `evalf` of a formal `Product` with reversed limits gave the empty product
/// `1` where `eval` cannot multiply it out (1501 factors, beyond the
/// enumeration limit).  Oracle: mpmath at 30 and 60 digits,
/// `1/Π_{k=−1499}^{1} (2 + sin k) = 3.1037015565939193996·10⁻⁴⁰⁷`.
#[test]
fn evalf_of_a_reversed_product_is_the_reciprocal() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.int(2) + k.sin();
    let p = Ex::symbolic_product(&body, &k, &ctx.int(2), &ctx.int(-1500));
    let v = p.eval_decimal(15).unwrap();
    assert!(v.starts_with("3.10370155659392"), "{v}");
    assert!(v.ends_with("e-407"), "{v}");
}

/// `Π_{k=1}^{n} (2k − 1)` was `(2n)!/(2ⁿ·n!)` for every `n`: at `n = −1` it is
/// `(−2)!/((1/2)·(−1)!) = zoo/zoo = nan`, where Karr's product is
/// `1/(2·0 − 1) = −1` (D13).  Legendre's duplication `Γ(m + 1/2) =
/// (2m)!√π/(4ᵐ·m!)` is an identity of values only for `m ≥ 0`; for a plain
/// `n` the closed form keeps `Γ(n + 1/2)`, which holds at every integer.
/// Oracle: exact Karr partial products (`n = −3`: `1/((−5)(−3)(−1)) = −1/15`;
/// `n = 4`: `1·3·5·7 = 105`).
#[test]
fn half_integer_gamma_forms_hold_at_negative_upper_limits() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let body = ctx.int(2) * &k - 1;
    let p = body.product_over(&k, &ctx.int(1), &n);
    assert_eq!(p.to_string(), "2^n*Gamma(n + 1/2)/sqrt(pi)");
    assert_eq!(
        values_at(&p, &n, &[-3, -2, -1, 0, 1, 4]),
        ["-1/15", "1/3", "-1", "1", "1", "105"]
    );
    let m = natural(&ctx, "m");
    let p = body.product_over(&k, &ctx.int(1), &m);
    assert_eq!(p.to_string(), "2^(-m)*(2*m)!/m!");
    // The content and the leading coefficient no longer print as (1/2)^n·2^n.
    let half = (&k - ctx.rational(1, 2)).product_over(&k, &ctx.int(1), &n);
    assert_eq!(half.to_string(), "Gamma(n + 1/2)/sqrt(pi)");
}

/// A product of non-integer powers was claimed for every `n`:
/// `Π_{k=1}^{n} √(k + 5/2) = √(8Γ(n + 7/2)/(15√π))` gives `4i/√15` at
/// `n = −4`, where Karr's product `1/(√(−1/2)·√(1/2)·√(3/2)·√(5/2))` is
/// `−4i/√15` (the principal roots multiply only while the factors are
/// positive; D13).  The closed form holds for `n ≥ −3` and comes with that
/// condition for a plain `n`; with a factor 0 below the lower limit
/// (`−k·√(k + 5/2)` at `k = 0`) the product is undefined below and the form
/// holds wherever it is defined.  Oracle: mpmath, `1/(sqrt(-0.5)·sqrt(0.5)·
/// sqrt(1.5)·sqrt(2.5)) = −1.0327955589886445027i` (30 and 60 digits).
#[test]
fn non_integer_powers_carry_their_condition() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let body = (&k + ctx.rational(5, 2)).sqrt();
    let p = body.product_over(&k, &ctx.int(1), &n);
    let s = p.to_string();
    assert!(
        s.starts_with("Piecewise(sqrt(8*Gamma(n + 7/2)/(15*sqrt(pi))) if n >= -3, "),
        "{s}"
    );
    let below = p.subs_i64(&n, -4).eval_complex64().unwrap();
    assert!(below.re.abs() < 1e-15 && (below.im + 1.032_795_558_988_644_5).abs() < 1e-14);
    let at = p.subs_i64(&n, -3).eval_complex64().unwrap();
    assert!((at.re - (8.0f64 / 15.0).sqrt()).abs() < 1e-14 && at.im.abs() < 1e-15);
    let err = body.try_product_over(&k, &ctx.int(1), &n).unwrap_err();
    assert!(err.to_string().contains("n >= -3"), "{err}");
    let m = natural(&ctx, "m");
    let p = body.product_over(&k, &ctx.int(1), &m);
    assert_eq!(p.to_string(), "sqrt(8*Gamma(m + 7/2)/(15*sqrt(pi)))");
    let zero_below = (-&k * &body).product_over(&k, &ctx.int(1), &n);
    assert!(
        !zero_below.to_string().starts_with("Piecewise"),
        "{zero_below}"
    );
}

/// `Ex::gosper_sum` called Gosper's algorithm without the checks
/// `summation` makes: `Σ_{k=−1}^{n} C(k + 2, k)/2ᵏ` was `nan` for every `n`
/// (the antidifference is undefined at the lower limit, where the term
/// `C(1, −1) = 0` vanishes by convention).  It now sums that term apart and
/// returns the form with its condition `n ≥ −1` (below, the form has poles
/// at `n = −2, −3`; the sum is 0).  Oracle: exact partial sums with
/// Graham–Knuth–Patashnik binomials, `0 + 1 + 3/2 + 6/4 + 10/8 = 21/4`
/// at `n = 3`; `n = −3`: `−C(0, −2)·4 = 0`.
#[test]
fn gosper_sum_is_checked() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let t = ctx.parse("binomial(k + 2, k)/2^k").unwrap();
    let s = Ex::symbolic_sum(&t, &k, &ctx.int(-1), &n).gosper_sum(&k);
    assert!(s.to_string().starts_with("Piecewise("), "{s}");
    assert!(s.to_string().contains("if n >= -1"), "{s}");
    assert_eq!(values_at(&s, &n, &[-3, -1, 0, 3]), ["0", "0", "1", "21/4"]);
    let concrete = Ex::symbolic_sum(&t, &k, &ctx.int(-1), &ctx.int(3)).gosper_sum(&k);
    assert_eq!(concrete.to_string(), "21/4");
    let kk = &k * &k.factorial();
    let s = Ex::symbolic_sum(&kk, &k, &ctx.int(0), &n).gosper_sum(&k);
    assert_eq!(s.to_string(), "(n + 1)! - 1");
    // Not Gosper-summable: unchanged.
    let sin = Ex::symbolic_sum(&k.sin(), &k, &ctx.int(0), &n);
    assert_eq!(sin.gosper_sum(&k), sin);
}

/// `Σ_{k≥−1} C(k + 2, k)/2ᵏ` stayed a `Sum`: the negative-binomial series
/// required a lower limit `≥ 0`.  The terms at `k = lo, …, −1` are added
/// (`C(1, −1) = 0`, GKP), so it is 8 like the sum from 0; `Σ_{k≥−3} C(k + 2,
/// 2)/2ᵏ` adds `C(−1, 2)·2³ = 8` to 8.  Oracle: `Σ_{k≥0} C(k + 2, 2)/2ᵏ =
/// (1 − 1/2)⁻³ = 8` (mpmath `nsum`, 30 and 60 digits: 8.0).
#[test]
fn negative_binomial_series_from_a_negative_lower_limit() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let inf = ctx.infinity();
    let t = ctx.parse("binomial(k + 2, k)/2^k").unwrap();
    assert_eq!(t.summation(&k, &ctx.int(-1), &inf).to_string(), "8");
    assert_eq!(t.summation(&k, &ctx.int(-3), &inf).to_string(), "8");
    let u = ctx.parse("binomial(k + 2, 2)/2^k").unwrap();
    assert_eq!(u.summation(&k, &ctx.int(-3), &inf).to_string(), "16");
}

/// With a symbolic lower limit `m` a Gosper form came with the condition
/// `n ≥ m − 1` (`Σ_{k=m}^{n} k·k! = Piecewise(((n + 1)! − m!, n ≥ m − 1),
/// …)`) though it holds wherever the sum is defined: the form for the lower
/// limit 0 holds (the term is undefined at every negative `k` and finite
/// from 0 on), and Karr's additivity `Σ_m^n = Σ_0^n − Σ_0^{m−1}` carries it
/// to every `m`.  A form with a pole at an integer `m` where the sum is
/// defined keeps its condition (`C(k + 2, k)/2ᵏ`).  Oracle: exact Karr
/// partial sums (`m = 3, n = 1`: `−Σ_{k=2}^{2} k·k! = −4 = 2! − 3!`).
#[test]
fn symbolic_lower_limit_gosper_forms_hold_everywhere() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let m = ctx.symbol("m");
    let s = (&k * &k.factorial()).summation(&k, &m, &n);
    assert_eq!(s.to_string(), "-m! + (n + 1)!");
    assert_eq!(s.subs_i64(&m, 3).subs_i64(&n, 1).eval().to_string(), "-4");
    assert_eq!(s.subs_i64(&m, 2).subs_i64(&n, 5).eval().to_string(), "718");
    let central = ctx
        .parse("binomial(2*k, k)/4^k")
        .unwrap()
        .summation(&k, &m, &n);
    assert!(!central.to_string().starts_with("Piecewise"), "{central}");
    assert_eq!(
        central.subs_i64(&m, 1).subs_i64(&n, 2).eval().to_string(),
        "7/8"
    );
    let shifted = ctx
        .parse("binomial(k + 2, k)/2^k")
        .unwrap()
        .summation(&k, &m, &n);
    assert!(shifted.to_string().contains("if n >= m - 1"), "{shifted}");
}

/// Since 0.55 `Σ_{k≥2} k·xᵏ/3ᵏ` printed
/// `x*(-27*x^2 + 6*x*(9*x - 27))/(-81*x^2 + 486*x - 729)` (0.54:
/// `1/3*x*(-1/3*x + 1)^(-2) - 1/3*x`): the closed form from `y²/(1 − y)` over
/// one denominator.  The sum from 0 less the skipped terms is kept when
/// shorter.  Oracle: `Σ_{k≥0} k·yᵏ = y/(1 − y)²`, at `x = 1` (`y = 1/3`):
/// `3/4 − 1/3 = 5/12`.
#[test]
fn infinite_geometric_sums_keep_the_shorter_form() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let x = ctx.symbol("x");
    let t = &k * &x.pow(&k) / ctx.int(3).pow(&k);
    let s = t.summation(&k, &ctx.int(2), &ctx.infinity());
    assert_eq!(
        s.to_string(),
        "Piecewise(3*x*(-x + 3)^(-2) - 1/3*x if 1 > 1/3*abs(x), Sum(k*x^k/3^k, k=2..oo) if True)"
    );
    assert_eq!(s.subs(&x, &ctx.int(1)).eval().to_string(), "5/12");
    // Already short: unchanged.
    let g = x.pow(&k).summation(&k, &ctx.int(2), &ctx.infinity());
    assert!(
        g.to_string().starts_with("Piecewise(x^2/(-x + 1) if "),
        "{g}"
    );
}
