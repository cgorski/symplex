//! Tests of session 15 (s15j2): finite sums for every integer bound, summation gaps.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

/// `Σ_{k=lo}^{n} body` at the integers `n` of `ns`, as strings.
fn values_at(s: &Ex, n: &Ex, ns: &[i64]) -> Vec<String> {
    ns.iter()
        .map(|&v| s.subs_i64(n, v).eval().to_string())
        .collect()
}

fn natural(ctx: &Context, name: &str) -> Ex {
    ctx.symbol_with(name, &[Assumption::Integer, Assumption::NonNegative])
        .unwrap()
}

/// Reversed limits were the empty sum (`Σ_{k=5}^{2} k = 0`, also through
/// `eval` and `eval_decimal` of a formal `Sum`).  D12: Karr's convention,
/// `Σ_{k=a}^{b} = −Σ_{k=b+1}^{a−1}` for `b < a − 1` (SymPy's), which keeps
/// `Σ_{a}^{b} + Σ_{b+1}^{c} = Σ_{a}^{c}` for all integers.  Oracle: exact
/// partial sums, `−(3 + 4) = −7`, `−(1/2 + 1/3) = −5/6`; `Σ_{k=3}^{2}` is
/// empty.
#[test]
fn reversed_limits_follow_karrs_convention() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let s = k.summation(&k, &ctx.int(5), &ctx.int(2));
    assert_eq!(s.to_string(), "-7");
    assert_eq!(k.summation(&k, &ctx.int(3), &ctx.int(2)).to_string(), "0");
    let inv = ctx.int(1) / &k;
    let formal = Ex::symbolic_sum(&inv, &k, &ctx.int(4), &ctx.int(1));
    assert_eq!(formal.eval().to_string(), "-5/6");
    assert_eq!(formal.eval_decimal(10).unwrap(), "-0.8333333333");
    // A term at a pole in the reflected range: undefined.
    let s = inv.summation(&k, &ctx.int(2), &ctx.int(-2));
    assert_eq!(s.to_string(), "zoo");
}

/// `Σ_{k=0}^{n} C(n, k)` was `2ⁿ` for every `n`: `1/2` at `n = −1`, where the
/// sum is empty (D12; SymPy 1.14 gives `2**n` too).  The identity holds for
/// `n ≥ 0`: `Piecewise((2ⁿ, n ≥ 0), (Sum, True))` for a plain `n`, `2ⁿ` for
/// an `n` declared non-negative; `eval` of the formal sum keeps it.  Oracle:
/// exact partial sums with Graham–Knuth–Patashnik binomials (`n = −3`:
/// `−Σ_{k=−2}^{−1} C(−3, k) = 0`; `n = 5`: 32).
#[test]
fn binomial_row_sum_holds_for_nonnegative_n() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let body = n.binomial(&k);
    let s = body.summation(&k, &ctx.int(0), &n);
    assert_eq!(
        s.to_string(),
        "Piecewise(2^n if n >= 0, Sum(C(n, k), k=0..n) if True)"
    );
    assert_eq!(values_at(&s, &n, &[-3, -1, 0, 5]), ["0", "0", "1", "32"]);
    let formal = Ex::symbolic_sum(&body, &k, &ctx.int(0), &n);
    assert_eq!(formal.eval().to_string(), "Sum(C(n, k), k=0..n)");
    let err = body.try_summation(&k, &ctx.int(0), &n).unwrap_err();
    assert!(err.to_string().contains("n >= 0"), "{err}");
    let m = natural(&ctx, "m");
    let s = m.binomial(&k).summation(&k, &ctx.int(0), &m);
    assert_eq!(s.to_string(), "2^m");
}

/// The other identities with `C(n, k)` were claimed for every `n` too:
/// `Σ k·C(n, k) = n·2ⁿ⁻¹` gave `−1/4` at `n = −1` (the sum is 0), `Σ_{k=1}^{n}
/// C(n, k) = 2ⁿ − 1` gave `−1/2` there (the sum is `−C(−1, 0) = −1`).  Those
/// whose closed form is 0 at every negative `n` hold everywhere and stay
/// unconditional: `Σ C(n, k)² = C(2n, n)` (`C(−2, −1) = 0`), `Σ (−1)ᵏ C(n, k)
/// = [n = 0]`.  Oracle: exact partial sums (Karr, GKP).
#[test]
fn binomial_identities_at_negative_n() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let c = n.binomial(&k);
    let s = (&k * &c).summation(&k, &ctx.int(0), &n);
    assert!(
        s.to_string()
            .starts_with("Piecewise(n*2^(n - 1) if n >= 0, "),
        "{s}"
    );
    assert_eq!(values_at(&s, &n, &[-4, -1, 3]), ["0", "0", "12"]);
    let s = c.summation(&k, &ctx.int(1), &n);
    assert_eq!(values_at(&s, &n, &[-2, -1, 0, 4]), ["-1", "-1", "0", "15"]);
    let s = c.powi(2).summation(&k, &ctx.int(0), &n);
    assert_eq!(s.to_string(), "C(2*n, n)");
    assert_eq!(values_at(&s, &n, &[-3, -1, 4]), ["0", "0", "70"]);
    let s = (&c * ctx.int(-1).pow(&k)).summation(&k, &ctx.int(0), &n);
    assert_eq!(s.to_string(), "Piecewise(1 if n == 0, 0 if n != 0)");
}

/// Gosper's antidifference of a term that vanishes by convention at the
/// lower limit was undefined there: `Σ_{k=−1}^{n} C(k + 2, k)/2ᵏ` was `nan`
/// for every `n` (`C(1, −1) = 0`, the certificate divides by `k + 1`).  The
/// lower-limit term is summed apart; the closed form holds for `n ≥ −1`
/// (below, the sum is 0 and the polynomial part of the form is not).
/// Oracle: exact partial sums, `n = 3`: `1 + 3/2 + 3/2 + 5/4 = 21/4`, `n = 7`:
/// `121/16` (SymPy `summation(binomial(k+2,k)/2**k, (k, 0, 7))`).
#[test]
fn gosper_closed_form_defined_at_the_lower_limit() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let body = ctx.parse("binomial(k+2, k)/2^k").unwrap();
    let s = body.summation(&k, &ctx.int(-1), &n);
    assert!(!s.to_string().contains("nan"), "{s}");
    assert_eq!(
        values_at(&s, &n, &[-4, -2, -1, 0, 3]),
        ["0", "0", "0", "1", "21/4"]
    );
    let m = natural(&ctx, "m");
    let s = body.summation(&k, &ctx.int(-1), &m);
    assert!(!s.has_unevaluated(), "{s}");
    assert_eq!(values_at(&s, &m, &[3, 7]), ["21/4", "121/16"]);
}

/// Gosper's closed forms are checked below the lower limit (Karr): where
/// the terms are undefined below (`k·k!`) or 0 together with the form's
/// dependence on `n` (`C(2k, k)/4ᵏ`, `k/(k + 1)!`), the form holds for every
/// `n` and stays unconditional.  Oracle: exact partial sums; `Σ_{k=0}^{4}
/// C(2k, k)/4ᵏ = 315/128`, at `n = −3` the reversed range of zero terms
/// (`C(−4, −2) = C(−2, −1) = 0`) gives 0; `Σ_{k=0}^{−5} k/(k + 1)! =
/// −Σ_{k=−4}^{−1} k/(k + 1)! = 1`.
#[test]
fn gosper_closed_forms_hold_below_the_lower_limit() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let s = ctx
        .parse("binomial(2*k, k)/4^k")
        .unwrap()
        .summation(&k, &ctx.int(0), &n);
    assert_eq!(s.to_string(), "2*(n + 1)*C(2*n + 2, n + 1)/4^(n + 1)");
    assert_eq!(values_at(&s, &n, &[-3, -1, 4]), ["0", "0", "315/128"]);
    let s = ctx
        .parse("k/factorial(k+1)")
        .unwrap()
        .summation(&k, &ctx.int(0), &n);
    assert_eq!(s.to_string(), "(-n - 2)/(n + 2)! + 1");
    assert_eq!(values_at(&s, &n, &[-5, -2, 3]), ["1", "1", "23/24"]);
    let s = ctx
        .parse("k*factorial(k)")
        .unwrap()
        .summation(&k, &ctx.int(0), &n);
    assert_eq!(s.to_string(), "(n + 1)! - 1");
}

/// A closed form undefined as a whole was returned for a sum defined for
/// some limits: `Σ_{k=0}^{n} k!/((k + 2)!·(k − 3))` was `zoo` (the term has
/// a pole at `k = 3` that the rational pole test does not see through the
/// factorials), `Σ_{k=−1}^{n} k·k!` was `zoo` (a pole at `k = −1`).  The sum
/// stays unevaluated, as for the rational terms with a pole in the range.
/// Oracle: `Σ_{k=0}^{2} 1/((k + 1)(k + 2)(k − 3)) = −1/6 − 1/12 − 1/12 =
/// −1/3`, finite.
#[test]
fn undefined_closed_forms_are_not_returned() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let body = ctx.parse("factorial(k)/(factorial(k+2)*(k-3))").unwrap();
    let s = body.summation(&k, &ctx.int(0), &n);
    assert!(s.to_string().starts_with("Sum("), "{s}");
    assert_eq!(s.subs_i64(&n, 2).eval().to_string(), "-1/3");
    let s = ctx
        .parse("k*factorial(k)")
        .unwrap()
        .summation(&k, &ctx.int(-1), &n);
    assert!(s.to_string().starts_with("Sum("), "{s}");
}

/// `is_convergent` returned `None` for a geometric-type term whose ratio the
/// assumptions bound, while `summation` decided it: `Σ (1/(a + 1))ᵏ` with a
/// positive `a` (`summation`: `1/a` from 1), `Σ (a/(a + 1))ᵏ/k²` (`Li₂`).  It
/// now decides `|ratio| < 1` as `summation` does; `Σ (a + 1)ᵏ/k²` diverges
/// (`summation` gave the `Sum`, now `∞`), `Σ C(2k, k)/(4a + 4)ᵏ` converges
/// (rate `4/(4a + 4) = 1/(a + 1)`).  A plain ratio stays undecided (D2).
/// Oracle: the ratio test, `|r| < 1` ⇔ convergence for `Σ P(k)·rᵏ`.
#[test]
fn convergence_agrees_with_summation_for_symbolic_ratios() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let conv = |src: &str| ctx.parse(src).unwrap().is_convergent(&k);
    assert_eq!(conv("(1/(a+1))^k"), Some(true));
    assert_eq!(conv("(a+1)^k"), Some(false));
    assert_eq!(conv("(a/(a+1))^k/k^2"), Some(true));
    assert_eq!(conv("(a+1)^k/k^2"), Some(false));
    assert_eq!(conv("binomial(2*k, k)*(1/(4*a+4))^k"), Some(true));
    assert_eq!(conv("x^k/k^2"), None);
    let s = ctx
        .parse("(a+1)^k/k^2")
        .unwrap()
        .summation(&k, &ctx.int(1), &ctx.infinity());
    assert_eq!(s.to_string(), "oo");
    let s = ctx
        .parse("(1/(a+1))^k")
        .unwrap()
        .is_absolutely_convergent(&k);
    assert_eq!(s, Some(true));
}

/// A geometric ratio written as a division, `1/2ᵏ` (the node `(2ᵏ)^(−1)`),
/// was not read by the negative-binomial rule: `Σ_{k≥0} C(k + c, c)/2ᵏ`
/// stayed a `Sum` while `Σ C(k + c, c)·(1/2)ᵏ` gave `(1/2)^(−c−1)`.
/// Oracle: the binomial series `Σ C(k + c, k)xᵏ = (1 − x)^(−c−1)`; mpmath
/// `nsum(lambda k: binomial(k + 1.5, 1.5)/2**k, [0, inf]) =
/// 5.65685424949238019520` `= 2^2.5`.
#[test]
fn division_spelling_of_a_geometric_ratio() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let c = ctx.symbol("c");
    let inf = ctx.infinity();
    let divided = ctx
        .parse("binomial(k + c, c)/2^k")
        .unwrap()
        .summation(&k, &ctx.int(0), &inf);
    let power = ctx
        .parse("binomial(k + c, c)*(1/2)^k")
        .unwrap()
        .summation(&k, &ctx.int(0), &inf);
    assert_eq!(divided.to_string(), power.to_string());
    let at = divided.subs(&c, &ctx.rational(3, 2));
    assert_eq!(at.eval_decimal(20).unwrap(), "5.6568542494923801952");
}

/// The `Sum` held in the last branch of a conditional closed form (D2, D12)
/// behaves: `eval` keeps the `Piecewise` (no recursion into its own `Sum`),
/// substituting a value selects the branch (`x = 1/2`: `Li₂(1/2)`, `x = 2`:
/// the `Sum`, which diverges to `oo`; `n = −2`: the reversed formal sum, 0),
/// `simplify` keeps it, and the JSON round trip is the identity.  Oracle:
/// `Li₂(1/2) = π²/12 − ln²2/2`; exact partial sums.
#[test]
fn conditional_closed_forms_substitute_and_round_trip() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let x = ctx.symbol("x");
    let n = ctx.symbol("n");
    let li2 = (x.pow(&k) / k.powi(2)).summation(&k, &ctx.int(1), &ctx.infinity());
    let row = n.binomial(&k).summation(&k, &ctx.int(0), &n);
    for s in [&li2, &row] {
        assert_eq!(s.eval().to_string(), s.to_string());
        assert_eq!(s.simplify().to_string(), s.to_string());
        let back = ctx.from_json(&s.to_json().unwrap()).unwrap();
        assert_eq!(back.to_string(), s.to_string());
    }
    assert_eq!(
        li2.subs(&x, &ctx.rational(1, 2)).eval().to_string(),
        "-1/2*ln(2)^2 + 1/12*pi^2"
    );
    assert_eq!(li2.subs(&x, &ctx.int(2)).eval().to_string(), "oo");
    assert_eq!(row.subs_i64(&n, -2).simplify().to_string(), "0");
    assert_eq!(row.subs_i64(&n, 6).eval().to_string(), "64");
}
