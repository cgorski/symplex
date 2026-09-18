//! Integration tests for the 0.2 symbolic summation / product engine:
//! `Ex::summation`, `Ex::try_summation`, `Ex::product_over`,
//! `Ex::try_product_over`, `Ex::hypergeometric_ratio`, and the `eval()`
//! path through `Sum` nodes.
//!
//! Every closed form with a symbolic bound is verified against direct
//! enumeration for several concrete `n`; constants are checked numerically.

use symplex::prelude::*;

/// Σ_{k=lo}^{hi} body(k) by brute force (exact).
fn brute_sum(body: &Ex, k: &Ex, lo: i64, hi: i64) -> Ex {
    let ctx = body.context();
    let mut acc = ctx.int(0);
    for i in lo..=hi {
        acc = &acc + &body.subs_i64(k, i);
    }
    acc.eval()
}

fn brute_prod(body: &Ex, k: &Ex, lo: i64, hi: i64) -> Ex {
    let ctx = body.context();
    let mut acc = ctx.int(1);
    for i in lo..=hi {
        acc = &acc * &body.subs_i64(k, i);
    }
    acc.eval()
}

/// Assert `closed(n) == Σ_{k=lo}^{n} body` for several `n`.
fn check_closed_sum(body: &Ex, k: &Ex, n: &Ex, lo: i64, closed: &Ex, ns: &[i64]) {
    assert!(
        !closed.has_unevaluated(),
        "expected a closed form for Σ {body}, got {closed}"
    );
    for &nv in ns {
        let expected = brute_sum(body, k, lo, nv);
        let got = closed.subs_i64(n, nv).eval();
        assert_eq!(
            got.to_string(),
            expected.to_string(),
            "Σ_{{k={lo}}}^{{{nv}}} {body}: closed form {closed}"
        );
    }
}

fn check_closed_prod(body: &Ex, k: &Ex, n: &Ex, lo: i64, closed: &Ex, ns: &[i64]) {
    assert!(
        !closed.has_unevaluated(),
        "expected a closed form for Π {body}, got {closed}"
    );
    for &nv in ns {
        let expected = brute_prod(body, k, lo, nv);
        let got = closed.subs_i64(n, nv).eval();
        assert_eq!(
            got.to_string(),
            expected.to_string(),
            "Π_{{k={lo}}}^{{{nv}}} {body}: closed form {closed}"
        );
    }
}

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

// ═══════════════════════════════════════════════════════════════════════════
// Faulhaber
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn faulhaber_all_powers_up_to_ten() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    for p in 0..=10i64 {
        let body = k.powi(p);
        let s = body.summation(&k, &ctx.int(1), &n);
        check_closed_sum(&body, &k, &n, 1, &s, &[0, 1, 2, 3, 7, 12]);
    }
}

#[test]
fn faulhaber_known_shapes() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let s1 = k.summation(&k, &ctx.int(1), &n);
    // n(n+1)/2 expanded
    assert_eq!(s1.to_string(), "1/2*n^2 + 1/2*n");
    let s2 = k.powi(2).summation(&k, &ctx.int(1), &n);
    assert_eq!(s2.to_string(), "1/3*n^3 + 1/2*n^2 + 1/6*n");
    // Σ_{k=0}^{n} 1 = n + 1
    let s0 = ctx.int(1).summation(&k, &ctx.int(0), &n);
    assert_eq!(s0.to_string(), "n + 1");
}

#[test]
fn polynomial_with_symbolic_coefficients_and_general_lower_bound() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let a = ctx.symbol("a");
    let body = &(&a * &k.powi(2)) + &(&k * 3) - &ctx.int(5);
    let s = body.summation(&k, &ctx.int(2), &n);
    assert!(!s.has_unevaluated(), "{s}");
    for nv in [2i64, 3, 6, 11] {
        for av in [1i64, 7] {
            let expected = brute_sum(&body.subs_i64(&a, av), &k, 2, nv);
            let got = s.subs_i64(&n, nv).subs_i64(&a, av).eval();
            assert_eq!(got.to_string(), expected.to_string());
        }
    }
    // (k+1)² needs expansion
    let body = (&k + 1).powi(2);
    let s = body.summation(&k, &ctx.int(1), &n);
    check_closed_sum(&body, &k, &n, 1, &s, &[1, 2, 5, 9]);
}

#[test]
fn big_numeric_range_uses_closed_form() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    // Σ_{k=1}^{1_000_000} k = 500000500000 (beyond the enumeration limit)
    let s = k.summation(&k, &ctx.int(1), &ctx.int(1_000_000));
    assert_eq!(s.to_string(), "500000500000");
    let s = k.powi(3).summation(&k, &ctx.int(1), &ctx.int(100_000));
    // (n(n+1)/2)² with n = 10^5
    assert_eq!(s.to_string(), "25000500002500000000");
}

#[test]
fn small_numeric_ranges_are_enumerated_exactly() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let s = (ctx.int(1) / &k).summation(&k, &ctx.int(1), &ctx.int(4));
    assert_eq!(s.to_string(), "25/12");
    let s = k.sin().summation(&k, &ctx.int(1), &ctx.int(2));
    assert_eq!(s.to_string(), "sin(1) + sin(2)");
    // empty range
    let s = k.summation(&k, &ctx.int(5), &ctx.int(1));
    assert_eq!(s.to_string(), "0");
    let p = k.product_over(&k, &ctx.int(5), &ctx.int(1));
    assert_eq!(p.to_string(), "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Geometric / arithmetico-geometric
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn geometric_numeric_ratio() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    for r in [2i64, 3, -2] {
        let body = ctx.int(r).pow(&k);
        let s = body.summation(&k, &ctx.int(0), &n);
        check_closed_sum(&body, &k, &n, 0, &s, &[0, 1, 4, 9]);
    }
    let body = ctx.rational(1, 3).pow(&k);
    let s = body.summation(&k, &ctx.int(1), &n);
    check_closed_sum(&body, &k, &n, 1, &s, &[1, 2, 6]);
}

#[test]
fn arithmetico_geometric_numeric_ratio() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let two_k = ctx.int(2).pow(&k);
    let three_k = ctx.int(3).pow(&k);
    let bodies = [
        &k * &two_k,
        &k.powi(2) * &two_k,
        &(&k.powi(2) + &k) * &three_k,
        &(&k * 5 + 7) * &ctx.rational(1, 2).pow(&k),
    ];
    for body in bodies {
        let s = body.summation(&k, &ctx.int(0), &n);
        check_closed_sum(&body, &k, &n, 0, &s, &[0, 1, 3, 6, 10]);
    }
}

#[test]
fn geometric_symbolic_ratio_is_piecewise() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let r = ctx.symbol("r");
    let body = r.pow(&k);
    let s = body.summation(&k, &ctx.int(0), &n);
    assert!(s.to_string().contains("Piecewise"), "{s}");
    for rv in [2i64, 5, -3] {
        let sr = s.subs_i64(&r, rv);
        check_closed_sum(&body.subs_i64(&r, rv), &k, &n, 0, &sr, &[0, 1, 4]);
    }
    // r = 1 branch: n + 1
    assert_eq!(s.subs_i64(&r, 1).subs_i64(&n, 4).eval().to_string(), "5");
    // (a + b k) r^k with all symbolic
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let body = &(&a + &(&b * &k)) * &r.pow(&k);
    let s = body.summation(&k, &ctx.int(0), &n);
    let sub = |e: &Ex| e.subs_i64(&a, 2).subs_i64(&b, 3).subs_i64(&r, 4);
    check_closed_sum(&sub(&body), &k, &n, 0, &sub(&s), &[0, 1, 3, 5]);
}

#[test]
fn infinite_geometric_convergent() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let inf = ctx.infinity();
    let half = ctx.rational(1, 2);
    assert_eq!(
        half.pow(&k).summation(&k, &ctx.int(0), &inf).to_string(),
        "2"
    );
    assert_eq!(
        half.pow(&k).summation(&k, &ctx.int(1), &inf).to_string(),
        "1"
    );
    // Σ (2/3)^k = 3
    assert_eq!(
        ctx.rational(2, 3)
            .pow(&k)
            .summation(&k, &ctx.int(0), &inf)
            .to_string(),
        "3"
    );
    // Σ (−1/2)^k = 2/3
    assert_eq!(
        ctx.rational(-1, 2)
            .pow(&k)
            .summation(&k, &ctx.int(0), &inf)
            .to_string(),
        "2/3"
    );
    // Σ e^{-k} = 1/(1 − e^{-1})
    let s = (-&k).exp().summation(&k, &ctx.int(0), &inf);
    let v = s.eval_f64().unwrap();
    assert!(approx(v, 1.0 / (1.0 - (-1f64).exp()), 1e-12), "{s} = {v}");
}

#[test]
fn infinite_arithmetico_geometric() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let inf = ctx.infinity();
    let half = ctx.rational(1, 2);
    // Σ k/2^k = 2, Σ k²/2^k = 6, Σ k³/2^k = 26
    assert_eq!(
        (&k * &half.pow(&k))
            .summation(&k, &ctx.int(0), &inf)
            .to_string(),
        "2"
    );
    assert_eq!(
        (&k.powi(2) * &half.pow(&k))
            .summation(&k, &ctx.int(0), &inf)
            .to_string(),
        "6"
    );
    assert_eq!(
        (&k.powi(3) * &half.pow(&k))
            .summation(&k, &ctx.int(0), &inf)
            .to_string(),
        "26"
    );
    // Σ k (1/3)^k = 3/4
    assert_eq!(
        (&k * &ctx.rational(1, 3).pow(&k))
            .summation(&k, &ctx.int(0), &inf)
            .to_string(),
        "3/4"
    );
}

#[test]
fn infinite_geometric_divergent_and_symbolic_ratio() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let inf = ctx.infinity();
    // Σ 2^k diverges → oo ; try_ → Divergent
    assert_eq!(
        ctx.int(2)
            .pow(&k)
            .summation(&k, &ctx.int(0), &inf)
            .to_string(),
        "oo"
    );
    assert!(matches!(
        ctx.int(2).pow(&k).try_summation(&k, &ctx.int(0), &inf),
        Err(SymplexError::Divergent { .. })
    ));
    // |r| = 1 with r = −1: oscillating divergence
    assert!(matches!(
        ctx.int(-1).pow(&k).try_summation(&k, &ctx.int(0), &inf),
        Err(SymplexError::Divergent { .. })
    ));
    // Σ (−2)^k: oscillating divergence keeps the Sum node, try_ → Divergent
    let osc = ctx.int(-2).pow(&k).summation(&k, &ctx.int(0), &inf);
    assert!(osc.to_string().contains("Sum"), "{osc}");
    assert!(matches!(
        ctx.int(-2).pow(&k).try_summation(&k, &ctx.int(0), &inf),
        Err(SymplexError::Divergent { .. })
    ));
    // Σ x^k with symbolic x: convergence unknown (no |x| < 1 assumption
    // vocabulary) → unevaluated rather than a conditional answer
    let x = ctx.symbol("x");
    let s = x.pow(&k).summation(&k, &ctx.int(0), &inf);
    assert!(s.has_unevaluated());
    assert!(x.pow(&k).try_summation(&k, &ctx.int(0), &inf).is_err());
    // Even a positive symbol is not enough (x = 2 would diverge)
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    assert!(p.pow(&k).summation(&k, &ctx.int(0), &inf).has_unevaluated());
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational functions: telescoping & harmonic numbers
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn telescoping_finite_sums() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let one = ctx.int(1);
    let bodies = [
        &one / &(&k * &(&k + 1)),
        &one / &(&k * &(&k + 2)),
        &one / &(&(&k * 2 - 1) * &(&k * 2 + 1)),
        &one / &(&(&k + 1) * &(&k + 3)),
        &(&k * 2 + 1) / &(&k.powi(2) * &(&k + 1).powi(2)),
        &one / &(&k * &(&k + 1) * &(&k + 2)),
        &k / &(&(&k + 1) * &(&k + 2) * &(&k + 3)),
    ];
    for body in bodies {
        let s = body.summation(&k, &ctx.int(1), &n);
        check_closed_sum(&body, &k, &n, 1, &s, &[1, 2, 3, 7, 15]);
    }
    // Explicit form: Σ 1/(k(k+1)) = 1 − 1/(n+1)
    let s = (&one / &(&k * &(&k + 1))).summation(&k, &ctx.int(1), &n);
    assert_eq!(s.subs_i64(&n, 9).eval().to_string(), "9/10");
    // Σ_{k=1}^{n} 1/((2k−1)(2k+1)) = n/(2n+1)
    let s = (&one / &(&(&k * 2 - 1) * &(&k * 2 + 1))).summation(&k, &ctx.int(1), &n);
    assert_eq!(s.subs_i64(&n, 10).eval().to_string(), "10/21");
}

#[test]
fn telescoping_infinite_sums() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let inf = ctx.infinity();
    let one = ctx.int(1);
    let cases: [(Ex, i64, &str); 6] = [
        (&one / &(&k * &(&k + 1)), 1, "1"),
        (&one / &(&k * &(&k + 2)), 1, "3/4"),
        (&one / &(&(&k * 2 - 1) * &(&k * 2 + 1)), 1, "1/2"),
        (&one / &(&k * &(&k + 1) * &(&k + 2)), 1, "1/4"),
        (&(&k * 2 + 1) / &(&k.powi(2) * &(&k + 1).powi(2)), 1, "1"),
        (&one / &(&k * &(&k + 1)), 3, "1/3"),
    ];
    for (body, lo, expected) in cases {
        let s = body.summation(&k, &ctx.int(lo), &inf);
        assert_eq!(s.to_string(), expected, "Σ_{{k≥{lo}}} {body}");
        // Sanity: partial sums approach the value
        let partial = brute_sum(&body, &k, lo, 400).eval_f64().unwrap();
        let v = s.eval_f64().unwrap();
        assert!(approx(partial, v, 1e-2), "{body}: partial {partial} vs {v}");
    }
    // Explicit non-trivial pole combination: 1/k − 2/(k+1) + 1/(k+2)
    let body = &(&one / &k) - &(&ctx.int(2) / &(&k + 1)) + &(&one / &(&k + 2));
    let s = body.summation(&k, &ctx.int(1), &inf);
    assert_eq!(s.to_string(), "1/2");
}

#[test]
fn harmonic_type_sums() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let inf = ctx.infinity();
    let s = k.powi(-1).summation(&k, &ctx.int(1), &n);
    assert_eq!(s.to_string(), "harmonic(n)");
    assert_eq!(s.subs_i64(&n, 5).eval().to_string(), "137/60");
    // Σ_{k=3}^{n} 1/(k+2) = H_{n+2} − H_4
    let s = (&ctx.int(1) / &(&k + 2)).summation(&k, &ctx.int(3), &n);
    check_closed_sum(&(&ctx.int(1) / &(&k + 2)), &k, &n, 3, &s, &[3, 4, 8]);
    // Σ_{k=0}^{n} 1/(2k+1) — digamma difference, numerically checked
    let body = &ctx.int(1) / &(&k * 2 + 1);
    let s = body.summation(&k, &ctx.int(0), &n);
    assert!(!s.has_unevaluated(), "{s}");
    let got = s.subs_i64(&n, 6).eval_f64().unwrap();
    let expected = brute_sum(&body, &k, 0, 6).eval_f64().unwrap();
    assert!(approx(got, expected, 1e-10), "{s}: {got} vs {expected}");
    // Divergence of the harmonic series
    assert_eq!(
        k.powi(-1).summation(&k, &ctx.int(1), &inf).to_string(),
        "oo"
    );
    assert!(matches!(
        k.powi(-1).try_summation(&k, &ctx.int(1), &inf),
        Err(SymplexError::Divergent { .. })
    ));
    // Σ k/(k+1) diverges to +∞
    assert_eq!(
        (&k / &(&k + 1))
            .summation(&k, &ctx.int(1), &inf)
            .to_string(),
        "oo"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Zeta-type constants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn zeta_even_closed_forms() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let inf = ctx.infinity();
    let pi = std::f64::consts::PI;
    let cases = [
        (2i64, pi.powi(2) / 6.0),
        (4, pi.powi(4) / 90.0),
        (6, pi.powi(6) / 945.0),
        (8, pi.powi(8) / 9450.0),
    ];
    for (p, expected) in cases {
        let s = k.powi(-p).summation(&k, &ctx.int(1), &inf);
        assert!(!s.has_unevaluated(), "ζ({p}) → {s}");
        assert!(
            approx(s.eval_f64().unwrap(), expected, 1e-12),
            "ζ({p}) = {s}"
        );
    }
    assert_eq!(
        k.powi(-2).summation(&k, &ctx.int(1), &inf).to_string(),
        "1/6*pi^2"
    );
    // Shifted start: Σ_{k≥2} 1/k² = π²/6 − 1
    let s = k.powi(-2).summation(&k, &ctx.int(2), &inf);
    assert!(
        approx(s.eval_f64().unwrap(), pi * pi / 6.0 - 1.0, 1e-12),
        "{s}"
    );
    // Scaled: Σ 3/k⁴
    let s = (&ctx.int(3) * &k.powi(-4)).summation(&k, &ctx.int(1), &inf);
    assert!(
        approx(s.eval_f64().unwrap(), 3.0 * pi.powi(4) / 90.0, 1e-12),
        "{s}"
    );
}

#[test]
fn zeta_odd_uses_zeta_node() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let inf = ctx.infinity();
    // Σ 1/k³ = ζ(3) (Apéry's constant), Σ 1/k⁵ = ζ(5)
    let s = k.powi(-3).summation(&k, &ctx.int(1), &inf);
    assert_eq!(s.to_string(), "zeta(3)");
    assert!(!s.has_unevaluated());
    assert!(approx(
        s.eval_f64().unwrap(),
        1.202_056_903_159_594_3,
        1e-12
    ));
    assert_eq!(
        k.powi(-3)
            .try_summation(&k, &ctx.int(1), &inf)
            .unwrap()
            .to_string(),
        "zeta(3)"
    );
    assert_eq!(
        k.powi(-5).summation(&k, &ctx.int(1), &inf).to_string(),
        "zeta(5)"
    );
    // Shifted start: Σ_{k≥2} 1/k³ = ζ(3) − 1
    let s = k.powi(-3).summation(&k, &ctx.int(2), &inf);
    assert!(
        approx(s.eval_f64().unwrap(), 1.202_056_903_159_594_3 - 1.0, 1e-12),
        "{s}"
    );
    // Odd powers of odd integers: Σ_{k≥0} 1/(2k+1)³ = (7/8) ζ(3)
    let s = (&k * 2 + 1).powi(-3).summation(&k, &ctx.int(0), &inf);
    assert!(
        approx(
            s.eval_f64().unwrap(),
            7.0 / 8.0 * 1.202_056_903_159_594_3,
            1e-12
        ),
        "{s}"
    );
    // Alternating: Σ (−1)^(k+1)/k³ = (3/4) ζ(3)
    let s = (ctx.int(-1).pow(&(&k + 1)) * k.powi(-3)).summation(&k, &ctx.int(1), &inf);
    assert!(
        approx(s.eval_f64().unwrap(), 0.75 * 1.202_056_903_159_594_3, 1e-12),
        "{s}"
    );
}

#[test]
fn catalan_constant_from_alternating_odd_squares() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let inf = ctx.infinity();
    // Σ_{k≥0} (−1)^k/(2k+1)² = G
    let body = ctx.int(-1).pow(&k) / (&k * 2 + 1).powi(2);
    let s = body.summation(&k, &ctx.int(0), &inf);
    assert_eq!(s.to_string(), "Catalan");
    assert!(approx(s.eval_f64().unwrap(), 0.915_965_594_177_219, 1e-12));
    // Shifted start: Σ_{k≥1} (−1)^k/(2k+1)² = G − 1
    let s = body.summation(&k, &ctx.int(1), &inf);
    assert!(
        approx(s.eval_f64().unwrap(), 0.915_965_594_177_219 - 1.0, 1e-12),
        "{s}"
    );
    // β(4) has no closed form: unevaluated, not wrong
    let s = (ctx.int(-1).pow(&k) / (&k * 2 + 1).powi(4)).summation(&k, &ctx.int(0), &inf);
    assert!(s.has_unevaluated(), "{s}");
}

#[test]
fn odd_denominator_p_series() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let inf = ctx.infinity();
    let pi = std::f64::consts::PI;
    // Σ_{k≥0} 1/(2k+1)² = π²/8
    let s = (&k * 2 + 1).powi(-2).summation(&k, &ctx.int(0), &inf);
    assert_eq!(s.to_string(), "1/8*pi^2");
    // Σ_{k≥0} 1/(2k+1)⁴ = π⁴/96
    let s = (&k * 2 + 1).powi(-4).summation(&k, &ctx.int(0), &inf);
    assert!(
        approx(s.eval_f64().unwrap(), pi.powi(4) / 96.0, 1e-12),
        "{s}"
    );
    // Shifted start: Σ_{k≥1} 1/(2k+1)² = π²/8 − 1
    let s = (&k * 2 + 1).powi(-2).summation(&k, &ctx.int(1), &inf);
    assert!(
        approx(s.eval_f64().unwrap(), pi * pi / 8.0 - 1.0, 1e-12),
        "{s}"
    );
}

#[test]
fn alternating_p_series_constants() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let inf = ctx.infinity();
    let pi = std::f64::consts::PI;
    let m1 = ctx.int(-1);
    // Σ (−1)^(k+1)/k = ln 2
    let s = (m1.pow(&(&k + 1)) / &k).summation(&k, &ctx.int(1), &inf);
    assert_eq!(s.to_string(), "ln(2)");
    // Σ (−1)^k/k = −ln 2
    let s = (m1.pow(&k) / &k).summation(&k, &ctx.int(1), &inf);
    assert!(approx(s.eval_f64().unwrap(), -(2f64).ln(), 1e-12), "{s}");
    // Σ (−1)^(k+1)/k² = π²/12
    let s = (m1.pow(&(&k + 1)) * k.powi(-2)).summation(&k, &ctx.int(1), &inf);
    assert_eq!(s.to_string(), "1/12*pi^2");
    // Σ (−1)^(k+1)/k⁴ = 7π⁴/720
    let s = (m1.pow(&(&k + 1)) * k.powi(-4)).summation(&k, &ctx.int(1), &inf);
    assert!(
        approx(s.eval_f64().unwrap(), 7.0 * pi.powi(4) / 720.0, 1e-12),
        "{s}"
    );
    // Starting later: Σ_{k≥2} (−1)^k/k = 1 − ln 2
    let s = (m1.pow(&k) / &k).summation(&k, &ctx.int(2), &inf);
    assert!(
        approx(s.eval_f64().unwrap(), 1.0 - (2f64).ln(), 1e-12),
        "{s}"
    );
}

#[test]
fn alternating_odd_denominator_constants() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let inf = ctx.infinity();
    let pi = std::f64::consts::PI;
    let m1 = ctx.int(-1);
    // Σ (−1)^k/(2k+1) = π/4
    let s = (m1.pow(&k) / (&k * 2 + 1)).summation(&k, &ctx.int(0), &inf);
    assert_eq!(s.to_string(), "1/4*pi");
    // Shifted start: Σ_{k≥1} (−1)^k/(2k+1) = π/4 − 1
    let s = (m1.pow(&k) / (&k * 2 + 1)).summation(&k, &ctx.int(1), &inf);
    assert!(approx(s.eval_f64().unwrap(), pi / 4.0 - 1.0, 1e-12), "{s}");
    // Σ (−1)^k/(2k+1)³ = π³/32
    let s = (m1.pow(&k) / (&k * 2 + 1).powi(3)).summation(&k, &ctx.int(0), &inf);
    assert!(
        approx(s.eval_f64().unwrap(), pi.powi(3) / 32.0, 1e-12),
        "{s}"
    );
    // Σ_{k≥1} (−1)^(k+1)/(2k−1) = π/4  (β = −1/2, so the (−1)^n sign factor matters)
    let s = (m1.pow(&(&k + 1)) / (&k * 2 - 1)).summation(&k, &ctx.int(1), &inf);
    assert!(approx(s.eval_f64().unwrap(), pi / 4.0, 1e-12), "{s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Binomial sums
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn binomial_identities() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let x = ctx.symbol("x");
    let c = n.binomial(&k);
    let zero = ctx.int(0);
    assert_eq!(c.summation(&k, &zero, &n).to_string(), "2^n");
    let with_n = |body: &Ex, closed: &Ex| {
        for nv in [0i64, 1, 2, 5, 8] {
            let expected = brute_sum(&body.subs_i64(&n, nv), &k, 0, nv);
            let got = closed.subs_i64(&n, nv).eval();
            assert_eq!(
                got.to_string(),
                expected.to_string(),
                "{body} at n={nv}: {closed}"
            );
        }
    };
    let body = &k * &c;
    let s = body.summation(&k, &zero, &n);
    assert!(!s.has_unevaluated());
    with_n(&body, &s);
    let body = &k.powi(2) * &c;
    let s = body.summation(&k, &zero, &n);
    assert!(!s.has_unevaluated());
    with_n(&body, &s);
    let body = c.powi(2);
    let s = body.summation(&k, &zero, &n);
    assert_eq!(s.to_string(), "C(2*n, n)");
    with_n(&body, &s);
    let body = &c / &(&k + 1);
    let s = body.summation(&k, &zero, &n);
    assert!(!s.has_unevaluated(), "{s}");
    with_n(&body, &s);
    let body = &c * &x.pow(&k);
    let s = body.summation(&k, &zero, &n);
    assert_eq!(s.to_string(), "(x + 1)^n");
    let body = &c * &ctx.int(3).pow(&k);
    assert_eq!(body.summation(&k, &zero, &n).to_string(), "4^n");
    let body = &c * &ctx.int(-1).pow(&k);
    let s = body.summation(&k, &zero, &n);
    assert_eq!(s.subs_i64(&n, 0).eval().to_string(), "1");
    assert_eq!(s.subs_i64(&n, 7).eval().to_string(), "0");
    // Σ k C(n,k) x^k = n x (1+x)^(n−1)
    let body = &(&k * &c) * &x.pow(&k);
    let s = body.summation(&k, &zero, &n);
    assert!(!s.has_unevaluated(), "{s}");
    for nv in [1i64, 3, 5] {
        let expected = brute_sum(&body.subs_i64(&n, nv).subs_i64(&x, 2), &k, 0, nv);
        let got = s.subs_i64(&n, nv).subs_i64(&x, 2).eval();
        assert_eq!(got.to_string(), expected.to_string());
    }
}

#[test]
fn binomial_polynomial_weights() {
    // Σ P(k) C(n,k) x^k for arbitrary polynomials P via the falling-factorial basis.
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let c = n.binomial(&k);
    let check = |body: &Ex, closed: &Ex| {
        assert!(!closed.has_unevaluated(), "{body} → {closed}");
        for nv in [0i64, 1, 2, 3, 6, 9] {
            for xv in [1i64, 2, -3] {
                let b = body.subs_i64(&n, nv).subs_i64(&x, xv);
                let expected = brute_sum(&b, &k, 0, nv);
                let got = closed.subs_i64(&n, nv).subs_i64(&x, xv).eval();
                assert_eq!(
                    got.to_string(),
                    expected.to_string(),
                    "{body} at n={nv}, x={xv}: {closed}"
                );
            }
        }
    };
    // Σ k(k−1) C(n,k) = n(n−1) 2^(n−2)
    let body = &(&k * &(&k - 1)) * &c;
    let s = body.summation(&k, &zero, &n);
    assert_eq!(s.to_string(), "n*2^(n - 2)*(n - 1)");
    check(&body, &s);
    // Σ k³ C(n,k) = n²(n+3) 2^(n−3)
    let body = &k.powi(3) * &c;
    check(&body, &body.summation(&k, &zero, &n));
    // Σ (k² + 1) C(n,k) x^k
    let body = &(&k.powi(2) + 1) * &c * &x.pow(&k);
    check(&body, &body.summation(&k, &zero, &n));
    // Σ (2k − n)² C(n,k) = n 2^n   (variance of the binomial distribution)
    let body = &(&k * 2 - &n).powi(2) * &c;
    let s = body.summation(&k, &zero, &n);
    check(&body, &s);
    assert_eq!(s.subs_i64(&n, 10).eval().to_string(), "10240");
}

// ═══════════════════════════════════════════════════════════════════════════
// Power series recognition
// ═══════════════════════════════════════════════════════════════════════════

/// Compare `Σ body` (from `lo` to ∞) with `expected` numerically at `x = 7/10`.
fn check_power_series(body: &Ex, k: &Ex, lo: i64, x: &Ex, expected: &Ex) {
    let ctx = body.context();
    let s = body.summation(k, &ctx.int(lo), &ctx.infinity());
    assert!(!s.has_unevaluated(), "Σ {body} → {s}");
    let sv = s.subs(x, &ctx.rational(7, 10)).eval_f64().unwrap();
    let ev = expected.subs(x, &ctx.rational(7, 10)).eval_f64().unwrap();
    assert!(
        approx(sv, ev, 1e-10),
        "Σ {body} = {s}, expected {expected}: {sv} vs {ev}"
    );
}

#[test]
fn power_series_exponential_family() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let x = ctx.symbol("x");
    let kf = k.factorial();
    let two_k = &k * 2;
    let two_k1 = &k * 2 + 1;
    let m1 = ctx.int(-1);
    assert_eq!(
        (x.pow(&k) / &kf)
            .summation(&k, &ctx.int(0), &ctx.infinity())
            .to_string(),
        "exp(x)"
    );
    check_power_series(&(x.pow(&two_k1) / two_k1.factorial()), &k, 0, &x, &x.sinh());
    check_power_series(&(x.pow(&two_k) / two_k.factorial()), &k, 0, &x, &x.cosh());
    check_power_series(&(&k * &x.pow(&k) / &kf), &k, 0, &x, &(&x * &x.exp()));
    check_power_series(&((&x * 2).pow(&k) / &kf), &k, 0, &x, &(&x * 2).exp());
    check_power_series(
        &(m1.pow(&k) * x.pow(&two_k) / kf.clone()),
        &k,
        0,
        &x,
        &(-&x.powi(2)).exp(),
    );
    // Numeric arguments: Σ 1/k! = e, Σ 1/(2k)! = cosh 1
    let one = ctx.int(1);
    assert_eq!(
        (&one / &kf)
            .summation(&k, &ctx.int(0), &ctx.infinity())
            .to_string(),
        "E"
    );
    let s = (&one / &two_k.factorial()).summation(&k, &ctx.int(0), &ctx.infinity());
    assert!(approx(s.eval_f64().unwrap(), 1f64.cosh(), 1e-12), "{s}");
}

#[test]
fn power_series_trigonometric_and_bessel() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let m1 = ctx.int(-1);
    let kf = k.factorial();
    let two_k = &k * 2;
    let two_k1 = &k * 2 + 1;
    check_power_series(
        &(m1.pow(&k) * x.pow(&two_k1) / two_k1.factorial()),
        &k,
        0,
        &x,
        &x.sin(),
    );
    check_power_series(
        &(m1.pow(&k) * x.pow(&two_k) / two_k.factorial()),
        &k,
        0,
        &x,
        &x.cos(),
    );
    // J₀(x) = Σ (−1)^k (x/2)^(2k) / k!²
    check_power_series(
        &(m1.pow(&k) * x.pow(&two_k) / (ctx.int(4).pow(&k) * kf.powi(2))),
        &k,
        0,
        &x,
        &x.bessel_j(&zero),
    );
    // I₀ has no numeric evaluator yet: check the symbolic result only.
    let i0 = (x.pow(&two_k) / (ctx.int(4).pow(&k) * kf.powi(2))).summation(&k, &zero, &inf);
    assert_eq!(i0.to_string(), x.bessel_i(&zero).to_string());
    // Σ (−1)^k/(2k+1)! = sin 1
    let s = (m1.pow(&k) / &two_k1.factorial()).summation(&k, &zero, &inf);
    assert!(approx(s.eval_f64().unwrap(), 1f64.sin(), 1e-12), "{s}");
}

#[test]
fn power_series_restricted_domain_needs_numeric_argument() {
    // atan / atanh / asin / ln(1−x) entries converge only for |x| ≤ 1, so a
    // symbolic x stays unevaluated while a numeric x inside the disc evaluates.
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let inf = ctx.infinity();
    let m1 = ctx.int(-1);
    let kf = k.factorial();
    let two_k = &k * 2;
    let two_k1 = &k * 2 + 1;
    let h = ctx.rational(1, 2);
    // atan(1/2), atanh(1/2), asin(1/2) = π/6
    let s = (m1.pow(&k) * h.pow(&two_k1) / &two_k1).summation(&k, &zero, &inf);
    assert!(approx(s.eval_f64().unwrap(), 0.5f64.atan(), 1e-12), "{s}");
    let s = (h.pow(&two_k1) / &two_k1).summation(&k, &zero, &inf);
    assert!(approx(s.eval_f64().unwrap(), 0.5f64.atanh(), 1e-12), "{s}");
    let s = (two_k.factorial() * h.pow(&two_k1) / (ctx.int(4).pow(&k) * kf.powi(2) * &two_k1))
        .summation(&k, &zero, &inf);
    assert!(
        approx(s.eval_f64().unwrap(), std::f64::consts::PI / 6.0, 1e-12),
        "{s}"
    );
    assert!(
        (m1.pow(&k) * x.pow(&two_k1) / &two_k1)
            .summation(&k, &zero, &inf)
            .has_unevaluated()
    );
    // −ln(1−x) with numeric x inside the disc
    let s = (ctx.rational(1, 2).pow(&k) / &k).summation(&k, &one, &inf);
    assert_eq!(s.to_string(), "ln(2)");
    let s = (ctx.rational(1, 3).pow(&k) / &k).summation(&k, &one, &inf);
    assert!(
        approx(s.eval_f64().unwrap(), -(2f64 / 3.0).ln(), 1e-12),
        "{s}"
    );
    // ln(1+x) at x = 1/2 via the alternating form: Σ (−1)^(k+1) (1/2)^k/k = ln(3/2)
    let s = (m1.pow(&(&k + 1)) * ctx.rational(1, 2).pow(&k) / &k).summation(&k, &one, &inf);
    assert!(approx(s.eval_f64().unwrap(), 1.5f64.ln(), 1e-12), "{s}");
    // Outside the disc: Σ 2^k/k diverges
    assert!(matches!(
        (ctx.int(2).pow(&k) / &k).try_summation(&k, &one, &inf),
        Err(SymplexError::Divergent { .. })
    ));
    // Symbolic x with a restricted domain stays unevaluated
    assert!((x.pow(&k) / &k).summation(&k, &one, &inf).has_unevaluated());
}

#[test]
fn power_series_skipped_leading_terms_and_erf() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let m1 = ctx.int(-1);
    let kf = k.factorial();
    let two_k1 = &k * 2 + 1;
    // Skipped leading terms: Σ_{k≥2} x^k/k! = e^x − 1 − x
    let s = (x.pow(&k) / &kf).summation(&k, &ctx.int(2), &inf);
    let sv = s.subs(&x, &ctx.rational(1, 2)).eval_f64().unwrap();
    assert!(approx(sv, 0.5f64.exp() - 1.5, 1e-12), "{s}");
    // Σ (−1)^k/(k!(2k+1)) = (√π/2) erf(1)
    let s = (m1.pow(&k) / (&kf * &two_k1)).summation(&k, &zero, &inf);
    assert!(!s.has_unevaluated(), "{s}");
    assert!(
        approx(s.eval_f64().unwrap(), 0.746_824_132_812_427_2, 1e-10),
        "{s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Gosper & hypergeometric ratio
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gosper_summable_terms() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let body = &k * &k.factorial();
    let s = body.summation(&k, &ctx.int(0), &n);
    check_closed_sum(&body, &k, &n, 0, &s, &[0, 1, 2, 5, 7]);
    // k/(k+1)! = 1/k! − 1/(k+1)!  → Σ_{k=1}^{n} = 1 − 1/(n+1)!
    let body = &k / &(&k + 1).factorial();
    let s = body.summation(&k, &ctx.int(1), &n);
    check_closed_sum(&body, &k, &n, 1, &s, &[1, 2, 4, 6]);
    // Also via gosper_sum on a Sum node
    let sum_node = Ex::symbolic_sum(&body, &k, &ctx.int(1), &n);
    let g = sum_node.gosper_sum(&k);
    assert!(!g.has_unevaluated());
    assert_eq!(
        g.subs_i64(&n, 4).eval().to_string(),
        brute_sum(&body, &k, 1, 4).to_string()
    );
    // Infinite Gosper-summable: Σ_{k≥1} k/(k+1)! = 1
    let s = body.summation(&k, &ctx.int(1), &ctx.infinity());
    assert_eq!(s.to_string(), "1");
}

#[test]
fn gosper_infinite_with_factorial_ratio_tail() {
    // Σ_{k≥0} k³·k!/(k+5)! — Gosper finds g(N) = P(N)·(N+1)!/(N+6)! + 25/288
    // and the tail P(N)·(N+1)!/(N+6)! → 0 (degree 4 over degree 5).
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let body = &k.factorial() * &k.powi(3) / (&k + 5).factorial();
    let s = body.summation(&k, &ctx.int(0), &n);
    check_closed_sum(&body, &k, &n, 0, &s, &[0, 1, 2, 5, 8]);
    let s = body.summation(&k, &ctx.int(0), &ctx.infinity());
    assert_eq!(s.to_string(), "25/288");
    // Partial sums in floating point: k³/((k+1)(k+2)(k+3)(k+4)(k+5)), tail ~ 1/N.
    let partial: f64 = (0..200_000u64)
        .map(|k| {
            let k = k as f64;
            k.powi(3) / ((k + 1.0) * (k + 2.0) * (k + 3.0) * (k + 4.0) * (k + 5.0))
        })
        .sum();
    assert!(approx(partial, 25.0 / 288.0, 1e-4), "partial {partial}");
}

#[test]
fn hypergeometric_ratio_api() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let x = ctx.symbol("x");
    let r = k.factorial().hypergeometric_ratio(&k).unwrap();
    assert_eq!(r.to_string(), "k + 1");
    let r = (&k * &k.factorial()).hypergeometric_ratio(&k).unwrap();
    // (k+1)²/k
    assert_eq!(r.subs_i64(&k, 3).eval().to_string(), "16/3");
    let r = (x.pow(&k) / k.factorial())
        .hypergeometric_ratio(&k)
        .unwrap();
    assert_eq!(r.subs_i64(&k, 4).eval().to_string(), "1/5*x");
    let r = n.binomial(&k).hypergeometric_ratio(&k).unwrap();
    assert_eq!(r.subs_i64(&n, 6).subs_i64(&k, 1).eval().to_string(), "5/2");
    let r = (ctx.int(1) / (&k * &(&k + 1)))
        .hypergeometric_ratio(&k)
        .unwrap();
    assert_eq!(r.subs_i64(&k, 2).eval().to_string(), "1/2");
    assert!(k.sin().hypergeometric_ratio(&k).is_none());
    assert!(k.pow(&k).hypergeometric_ratio(&k).is_none());
    assert!(k.harmonic().hypergeometric_ratio(&k).is_none());
}

#[test]
fn not_summable_stays_unevaluated_and_linearity_is_partial() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let s = k.sin().summation(&k, &ctx.int(1), &n);
    assert!(s.has_unevaluated());
    assert!(s.to_string().starts_with("Sum("), "{s}");
    assert!(matches!(
        k.sin().try_summation(&k, &ctx.int(1), &n),
        Err(SymplexError::ComputationFailed { .. })
    ));
    // partial: Σ (k + sin k) = n(n+1)/2 + Σ sin k
    let s = (&k + &k.sin()).summation(&k, &ctx.int(1), &n);
    assert!(s.has_unevaluated());
    let t = s.to_string();
    assert!(t.contains("n^2") && t.contains("Sum(sin(k)"), "{t}");
    assert!((&k + &k.sin()).try_summation(&k, &ctx.int(1), &n).is_err());
    // Σ_{k≥1} 1/k^(3/2) converges but has no closed form → unevaluated, not wrong
    let s = k
        .pow(&ctx.rational(-3, 2))
        .summation(&k, &ctx.int(1), &ctx.infinity());
    assert!(s.has_unevaluated(), "{s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// eval() integration through Sum nodes
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_path_uses_the_engine() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    // symbolic upper bound via eval()
    let s = Ex::symbolic_sum(&k.powi(5), &k, &ctx.int(1), &n).eval();
    assert!(!s.has_unevaluated(), "{s}");
    assert_eq!(s.subs_i64(&n, 6).eval().to_string(), "12201");
    // infinite via eval()
    let s = Ex::symbolic_sum(&k.powi(-2), &k, &ctx.int(1), &ctx.infinity()).eval();
    assert_eq!(s.to_string(), "1/6*pi^2");
    // divergent via eval() → oo
    let s = Ex::symbolic_sum(&k.powi(-1), &k, &ctx.int(1), &ctx.infinity()).eval();
    assert_eq!(s.to_string(), "oo");
    // closed_form_sum still works
    let s = Ex::symbolic_sum(&(&ctx.int(2).pow(&k) * &k), &k, &ctx.int(0), &n).closed_form_sum();
    assert_eq!(s.subs_i64(&n, 5).eval().to_string(), "258");
    // Doubly infinite: Σ_{k=−∞}^{∞} 2^{−|k|}… use a rational: Σ 1/(k²+1) has no closed form,
    // but Σ_{k=-∞}^{-1} 2^k = 1.
    let s = ctx
        .int(2)
        .pow(&k)
        .summation(&k, &ctx.neg_infinity(), &ctx.int(-1));
    assert_eq!(s.to_string(), "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Products
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn products_factorial_and_power_forms() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let one = ctx.int(1);
    let a = ctx.symbol("a");
    assert_eq!(k.product_over(&k, &one, &n).to_string(), "n!");
    assert_eq!(ctx.int(3).product_over(&k, &one, &n).to_string(), "3^n");
    assert_eq!(k.powi(2).product_over(&k, &one, &n).to_string(), "n!^2");
    // Π 2k = 2^n n!
    let p = (&k * 2).product_over(&k, &one, &n);
    assert_eq!(p.to_string(), "2^n*n!");
    check_closed_prod(&(&k * 2), &k, &n, 1, &p, &[1, 2, 5]);
    // Π (2k − 1) = (2n)!/(2^n n!)
    let body = &k * 2 - 1;
    let p = body.product_over(&k, &one, &n);
    assert_eq!(p.to_string(), "2^(-n)*(2*n)!/n!");
    check_closed_prod(&body, &k, &n, 1, &p, &[1, 2, 4, 6]);
    // Π a^k = a^(n(n+1)/2);  Π e^k = e^(n(n+1)/2);  Π 2^(k²) = 2^(Σ k²)
    let p = a.pow(&k).product_over(&k, &one, &n);
    assert_eq!(p.subs_i64(&n, 4).eval().to_string(), "a^10");
    let p = k.exp().product_over(&k, &one, &n);
    assert_eq!(p.subs_i64(&n, 4).eval().to_string(), "exp(10)");
    let p = ctx.int(2).pow(&k.powi(2)).product_over(&k, &one, &n);
    assert_eq!(p.subs_i64(&n, 3).eval().to_string(), "16384");
    // Split Π f·g:  Π k·2^k = n!·2^(n(n+1)/2)
    let body = &k * &ctx.int(2).pow(&k);
    let p = body.product_over(&k, &one, &n);
    check_closed_prod(&body, &k, &n, 1, &p, &[1, 2, 4]);
    let body = &k.powi(2) * &(&k + 1);
    let p = body.product_over(&k, &one, &n);
    check_closed_prod(&body, &k, &n, 1, &p, &[1, 2, 4]);
}

#[test]
fn products_rational_telescoping() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let one = ctx.int(1);
    assert_eq!(
        (&one + &k.powi(-1)).product_over(&k, &one, &n).to_string(),
        "n + 1"
    );
    let body = &one - &k.powi(-2);
    let p = body.product_over(&k, &ctx.int(2), &n);
    check_closed_prod(&body, &k, &n, 2, &p, &[2, 3, 6, 9]);
    assert_eq!(
        body.product_over(&k, &ctx.int(2), &ctx.infinity())
            .to_string(),
        "1/2"
    );
    // Π (k+1)/k telescopes to n + 1 as a rational function product
    let body = &(&k + 1) / &k;
    assert_eq!(body.product_over(&k, &one, &n).to_string(), "n + 1");
    // Π_{k=1}^{n} k/(k+2) = 2/((n+1)(n+2))
    let body = &k / &(&k + 2);
    let p = body.product_over(&k, &one, &n);
    check_closed_prod(&body, &k, &n, 1, &p, &[1, 2, 5]);
    assert_eq!(
        body.product_over(&k, &one, &ctx.infinity()).to_string(),
        "0"
    );
}

#[test]
fn products_symbolic_shift_gamma_ratio() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let a = ctx.symbol("a");
    let one = ctx.int(1);
    // Π_{k=1}^{n} (k + a) = Γ(n + a + 1)/Γ(a + 1) = (a + n)!/a!
    let p = (&k + &a).product_over(&k, &one, &n);
    assert_eq!(p.to_string(), "(a + n)!/a!");
    for (nv, av) in [(3i64, 2i64), (5, 1), (4, 7)] {
        let expected = brute_prod(&(&k + &a).subs_i64(&a, av), &k, 1, nv);
        let got = p.subs_i64(&n, nv).subs_i64(&a, av).eval();
        assert_eq!(got.to_string(), expected.to_string(), "{p}");
    }
    // Π_{k=0}^{n} (k + a) = Γ(a + n + 1)/Γ(a)  (rising factorial a^(n+1))
    let p = (&k + &a).product_over(&k, &ctx.int(0), &n);
    assert_eq!(p.to_string(), "(a + n)!/Gamma(a)");
    for (nv, av) in [(2i64, 3i64), (4, 1)] {
        let expected = brute_prod(&(&k + &a).subs_i64(&a, av), &k, 0, nv);
        let got = p.subs_i64(&n, nv).subs_i64(&a, av).eval();
        assert_eq!(got.to_string(), expected.to_string(), "{p}");
    }
}

#[test]
fn products_unevaluated_and_try_variant() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let p = k.sin().product_over(&k, &ctx.int(1), &n);
    assert!(p.has_unevaluated());
    assert!(p.to_string().starts_with("Product("), "{p}");
    assert!(k.sin().try_product_over(&k, &ctx.int(1), &n).is_err());
    assert_eq!(
        k.try_product_over(&k, &ctx.int(1), &n).unwrap().to_string(),
        "n!"
    );
    // numeric enumeration
    assert_eq!(
        k.product_over(&k, &ctx.int(1), &ctx.int(6)).to_string(),
        "720"
    );
    // Π_{k≥1} 2 diverges
    assert!(matches!(
        ctx.int(2)
            .try_product_over(&k, &ctx.int(1), &ctx.infinity()),
        Err(SymplexError::Divergent { .. })
    ));
}
