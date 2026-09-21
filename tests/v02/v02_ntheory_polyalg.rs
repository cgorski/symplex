//! Integration tests for the 0.2 polynomial-algebra methods on `Ex`
//! (resultant, discriminant, square-free, division, decomposition, roots).

use proptest::prelude::*;
use symplex::prelude::*;

fn ctx_x() -> (Context, Ex) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    (ctx, x)
}

fn s(e: &Ex) -> String {
    format!("{e}")
}

// ── resultant / discriminant ────────────────────────────────────────────

#[test]
fn resultant_detects_common_roots() {
    let (_ctx, x) = ctx_x();
    let f = &x.powi(2) - 1;
    assert_eq!(s(&f.resultant(&(&x - 1), &x).unwrap()), "0");
    assert_eq!(s(&f.resultant(&(&x - 3), &x).unwrap()), "8");
    // res(x² + 1, x² − 1) = 4
    assert_eq!(s(&(&x.powi(2) + 1).resultant(&f, &x).unwrap()), "4");
    assert!(x.sin().resultant(&f, &x).is_none());
}

#[test]
fn discriminant_matches_closed_forms() {
    let (ctx, x) = ctx_x();
    // quadratic b² − 4ac
    for (a, b, c) in [(1i64, 3i64, 1i64), (2, -5, 3), (1, 0, 1), (3, 6, 3)] {
        let f = &x.powi(2) * a + &x * b + c;
        let d = f.discriminant(&x).unwrap();
        assert_eq!(s(&d), (b * b - 4 * a * c).to_string(), "{f}");
    }
    // cubic x³ + px + q: −4p³ − 27q²
    for (p, q) in [(-1i64, 1i64), (2, 3), (-3, 2)] {
        let f = &x.powi(3) + &x * p + q;
        let d = f.discriminant(&x).unwrap();
        assert_eq!(s(&d), (-4 * p * p * p - 27 * q * q).to_string(), "{f}");
    }
    assert_eq!(s(&(&x + 5).discriminant(&x).unwrap()), "1");
    assert!(ctx.int(7).discriminant(&x).is_none());
}

// ── square-free ─────────────────────────────────────────────────────────

#[test]
fn sqf_list_and_square_free_part() {
    let (_ctx, x) = ctx_x();
    // 3 (x − 1)² (x + 2)³ (x² + 1)
    let f = ((&x - 1).powi(2) * (&x + 2).powi(3) * (&x.powi(2) + 1) * 3).expand();
    let (content, parts) = f.sqf_list(&x).unwrap();
    assert_eq!(s(&content), "3");
    let described: Vec<(String, u32)> = parts.iter().map(|(p, m)| (s(p), *m)).collect();
    assert_eq!(
        described,
        vec![
            ("x^2 + 1".to_string(), 1),
            ("x - 1".to_string(), 2),
            ("x + 2".to_string(), 3)
        ]
    );
    let sf = f.square_free_part(&x).unwrap();
    assert_eq!(sf.degree(&x), Some(4));
    assert_eq!(sf.is_squarefree(&x), Some(true));
    assert_eq!(f.is_squarefree(&x), Some(false));
    assert_eq!((&x.powi(2) + 1).is_squarefree(&x), Some(true));
}

// ── division ────────────────────────────────────────────────────────────

#[test]
fn poly_division_identities() {
    let (_ctx, x) = ctx_x();
    let f = &x.powi(5) + &x.powi(3) * 2 - &x + 7;
    let g = &x.powi(2) + &x - 1;
    let (q, r) = f.poly_div(&g, &x).unwrap();
    assert!(r.degree(&x).unwrap_or(0) < 2);
    let back = (&q * &g + &r).expand();
    assert_eq!(s(&back), s(&f));
    assert_eq!(s(&f.poly_quo(&g, &x).unwrap()), s(&q));
    assert_eq!(s(&f.poly_rem(&g, &x).unwrap()), s(&r));
    // Division by zero → None.
    assert!(f.poly_div(&_ctx.int(0), &x).is_none());
    // Exact division.
    let (q, r) = (&x.powi(3) - 8).poly_div(&(&x - 2), &x).unwrap();
    assert_eq!(s(&q), "x^2 + 2*x + 4");
    assert_eq!(s(&r), "0");
}

#[test]
fn poly_gcdex_bezout_identity() {
    let (_ctx, x) = ctx_x();
    let f = (&x.powi(2) - 1) * (&x + 3);
    let g = (&x - 1) * (&x.powi(2) + 1);
    let (s_, t, gcd) = f.expand().poly_gcdex(&g.expand(), &x).unwrap();
    assert_eq!(s(&gcd), "x - 1");
    let check = (&s_ * &f + &t * &g).expand();
    assert_eq!(s(&check), "x - 1");
    // Coprime → gcd 1.
    let (_, _, one) = (&x.powi(2) + 1).poly_gcdex(&(&x + 1), &x).unwrap();
    assert_eq!(s(&one), "1");
}

// ── structure ───────────────────────────────────────────────────────────

#[test]
fn decompose_examples() {
    let (_ctx, x) = ctx_x();
    let f = &x.powi(4) + &x.powi(2) * 2 + 1;
    let parts: Vec<String> = f.decompose(&x).iter().map(s).collect();
    assert_eq!(parts, vec!["x^2 + 2*x + 1", "x^2"]);

    // (x³ + x)∘(x² + 1) = (x²+1)³ + (x²+1)
    let h = &x.powi(2) + 1;
    let f = (h.powi(3) + &h).expand();
    let parts = f.decompose(&x);
    assert_eq!(parts.len(), 2);
    // Re-compose to verify.
    let recomposed = parts[0].poly_compose(&parts[1], &x).unwrap();
    assert_eq!(s(&recomposed.expand()), s(&f));

    // Prime degree → indecomposable.
    let g = &x.powi(5) - &x - 1;
    assert_eq!(g.decompose(&x).len(), 1);
    // Non-polynomial → empty.
    assert!(x.sin().decompose(&x).is_empty());
    // Chebyshev-like: T4 = 8x⁴ − 8x² + 1 = (8x² − 8x + 1)∘x²
    let t4 = &x.powi(4) * 8 - &x.powi(2) * 8 + 1;
    let parts: Vec<String> = t4.decompose(&x).iter().map(s).collect();
    assert_eq!(parts, vec!["8*x^2 - 8*x + 1", "x^2"]);
}

#[test]
fn content_primitive_leading_coeff_monic() {
    let (ctx, x) = ctx_x();
    let f = &x.powi(3) * -6 + &x * 9;
    let (c, p) = f.content_primitive(&x);
    assert_eq!(s(&c), "-3");
    assert_eq!(s(&p), "2*x^3 - 3*x");
    assert_eq!(s(&f.leading_coeff(&x).unwrap()), "-6");
    assert_eq!(s(&f.monic(&x).unwrap()), "x^3 - 3/2*x");
    // Non-polynomial: (1, self).
    let g = x.exp();
    let (c, p) = g.content_primitive(&x);
    assert_eq!(s(&c), "1");
    assert_eq!(s(&p), s(&g));
    assert!(ctx.int(0).monic(&x).is_none());
}

#[test]
fn compose_shift_reverse_interpolate() {
    let (ctx, x) = ctx_x();
    let f = &x.powi(3) - &x;
    assert_eq!(
        s(&f.poly_compose(&(&x + 1), &x).unwrap()),
        s(&f.poly_shift(&x, &ctx.int(1)).unwrap())
    );
    assert_eq!(
        s(&f.poly_shift(&x, &ctx.int(1)).unwrap()),
        "x^3 + 3*x^2 + 2*x"
    );
    assert_eq!(
        s(&(&x.powi(3) * 2 + &x * 3 + 5).poly_reverse(&x).unwrap()),
        "5*x^3 + 3*x^2 + 2"
    );
    // Reverse is an involution when f(0) ≠ 0.
    let g = &x.powi(4) + &x * 7 - 2;
    let rr = g.poly_reverse(&x).unwrap().poly_reverse(&x).unwrap();
    assert_eq!(s(&rr), s(&g));
    // Interpolation through exact rationals.
    let pts = [
        (ctx.int(-1), ctx.int(4)),
        (ctx.int(0), ctx.int(1)),
        (ctx.int(1), ctx.int(0)),
        (ctx.int(2), ctx.int(1)),
    ];
    let p = Ex::poly_interpolate(&pts, &x).unwrap();
    assert_eq!(s(&p), "x^2 - 2*x + 1");
    // Duplicate abscissae → None; non-rational → None.
    assert!(
        Ex::poly_interpolate(&[(ctx.int(1), ctx.int(2)), (ctx.int(1), ctx.int(3))], &x).is_none()
    );
    assert!(Ex::poly_interpolate(&[(ctx.pi(), ctx.int(2))], &x).is_none());
}

// ── real roots ──────────────────────────────────────────────────────────

#[test]
fn sturm_root_counting() {
    let (ctx, x) = ctx_x();
    let f = &x.powi(3) - &x; // roots −1, 0, 1
    assert_eq!(f.count_real_roots(&x), Some(3));
    assert_eq!(
        f.count_real_roots_in(&x, &ctx.int(-1), &ctx.int(1)),
        Some(3)
    );
    assert_eq!(f.count_real_roots_in(&x, &ctx.int(0), &ctx.int(1)), Some(2));
    assert_eq!(
        f.count_real_roots_in(&x, &ctx.rational(1, 2), &ctx.int(10)),
        Some(1)
    );
    assert_eq!(
        f.count_real_roots_in(&x, &ctx.neg_infinity(), &ctx.rational(-1, 2)),
        Some(1)
    );
    assert_eq!(
        f.count_real_roots_in(&x, &ctx.neg_infinity(), &ctx.infinity()),
        Some(3)
    );
    assert_eq!(f.count_real_roots_in(&x, &ctx.int(5), &ctx.int(1)), Some(0));
    assert_eq!((&x.powi(2) + 1).count_real_roots(&x), Some(0));
    // Repeated roots counted once.
    assert_eq!((&x - 2).powi(3).expand().count_real_roots(&x), Some(1));
    assert!(f.count_real_roots_in(&x, &ctx.pi(), &ctx.int(1)).is_none());
    // Wilkinson-style: 10 real roots.
    let mut w = ctx.int(1);
    for k in 1..=10 {
        w *= &x - k;
    }
    assert_eq!(w.expand().count_real_roots(&x), Some(10));
}

#[test]
fn real_root_isolation_brackets_every_root() {
    let (_ctx, x) = ctx_x();
    // roots: ±√3, −1/2, 0 (rational), ±√2, 3
    let f = ((&x.powi(2) - 3) * (&x * 2 + 1) * &x * (&x.powi(2) - 2) * (&x - 3)).expand();
    let iv = f.real_roots_isolate(&x);
    assert_eq!(iv.len(), 7);
    let expected = [
        -(3f64.sqrt()),
        -(2f64.sqrt()),
        -0.5,
        0.0,
        2f64.sqrt(),
        3f64.sqrt(),
        3.0,
    ];
    for (interval, root) in iv.iter().zip(expected.iter()) {
        let lo = interval.lower.eval_f64().unwrap();
        let hi = interval.upper.eval_f64().unwrap();
        assert!(lo <= *root && *root <= hi, "{root} not in [{lo}, {hi}]");
        assert!(hi - lo <= 1.0 / 1024.0 + 1e-12);
        // A Sturm cell is `(lo, hi]`; an exact hit is the point `[r, r]`.
        match interval.kind {
            IntervalKind::LeftOpen => assert!(lo < hi, "{interval}"),
            IntervalKind::Closed => assert_eq!(lo, hi, "{interval}"),
            other => panic!("unexpected kind {other:?} for {interval}"),
        }
    }
    // 0 is hit exactly by the bisection and reported as a point.
    assert_eq!(iv[3].kind, IntervalKind::Closed);
    assert_eq!(iv[3].lower, iv[3].upper);
    assert!(x.sin().real_roots_isolate(&x).is_empty());
}

// ── numeric roots ───────────────────────────────────────────────────────

#[test]
fn nroots_all_complex_roots() {
    let (_ctx, x) = ctx_x();
    // x^5 − x − 1: one real root ≈ 1.1673, two conjugate pairs
    let f = &x.powi(5) - &x - 1;
    let roots = f.nroots(&x, 15).unwrap();
    assert_eq!(roots.len(), 5);
    let real: Vec<_> = roots.iter().filter(|r| r.1.abs() < 1e-9).collect();
    assert_eq!(real.len(), 1);
    assert!((real[0].0 - 1.1673039782614187).abs() < 1e-9);
    // Every root satisfies |f(z)| ≈ 0 (evaluate with complex arithmetic).
    for (re, im) in &roots {
        let z = (*re, *im);
        // Horner in complex arithmetic: coefficients of x^5 − x − 1.
        let coeffs = [-1.0, -1.0, 0.0, 0.0, 0.0, 1.0];
        let mut acc = (0.0f64, 0.0f64);
        for c in coeffs.iter().rev() {
            acc = (acc.0 * z.0 - acc.1 * z.1 + c, acc.0 * z.1 + acc.1 * z.0);
        }
        assert!(acc.0.abs() < 1e-8 && acc.1.abs() < 1e-8, "residual {acc:?}");
    }
    // Multiplicity: (x − 1)² (x + 2)
    let g = ((&x - 1).powi(2) * (&x + 2)).expand();
    let roots = g.nroots(&x, 15).unwrap();
    assert_eq!(roots.len(), 3);
    assert!((roots[0].0 + 2.0).abs() < 1e-10);
    assert!((roots[1].0 - 1.0).abs() < 1e-10 && (roots[2].0 - 1.0).abs() < 1e-10);
    // Errors.
    assert!(x.sin().nroots(&x, 10).is_err());
    assert!(_ctx.int(3).nroots(&x, 10).is_err());
}

// ── property tests ──────────────────────────────────────────────────────

fn shared_ctx() -> &'static Context {
    use std::sync::OnceLock;
    static CTX: OnceLock<Context> = OnceLock::new();
    CTX.get_or_init(Context::new)
}

fn arb_poly(max_deg: usize) -> impl Strategy<Value = Ex> {
    prop::collection::vec(-5i64..6, 1..=max_deg + 1).prop_map(|coeffs| {
        let ctx = shared_ctx();
        let x = ctx.symbol("x");
        coeffs
            .iter()
            .enumerate()
            .fold(ctx.int(0), |acc, (i, &c)| acc + x.powi(i as i64) * c)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Division identity f = q·g + r with deg r < deg g.
    #[test]
    fn prop_division_identity(f in arb_poly(6), g in arb_poly(3)) {
        let x = shared_ctx().symbol("x");
        prop_assume!(g.degree(&x).is_some());
        let (q, r) = f.poly_div(&g, &x).unwrap();
        let back = (&q * &g + &r).expand();
        prop_assert_eq!(format!("{back}"), format!("{}", f.expand()));
        if let Some(dr) = r.degree(&x) {
            prop_assert!(dr < g.degree(&x).unwrap());
        }
    }

    /// Resultant is zero iff there is a common factor.
    #[test]
    fn prop_resultant_zero_iff_common_factor(a in arb_poly(2), b in arb_poly(2), c in arb_poly(2)) {
        let x = shared_ctx().symbol("x");
        prop_assume!(a.degree(&x).is_some_and(|d| d >= 1));
        prop_assume!(b.degree(&x).is_some() && c.degree(&x).is_some());
        let f = (&a * &b).expand();
        let g = (&a * &c).expand();
        let r = f.resultant(&g, &x).unwrap();
        prop_assert_eq!(format!("{r}"), "0");
    }

    /// Sturm count agrees with the number of distinct real roots found numerically.
    #[test]
    fn prop_sturm_matches_nroots(f in arb_poly(5)) {
        let x = shared_ctx().symbol("x");
        prop_assume!(f.degree(&x).is_some_and(|d| d >= 1));
        let count = f.count_real_roots(&x).unwrap();
        let sf = f.square_free_part(&x).unwrap();
        let roots = sf.nroots(&x, 20).unwrap();
        let real = roots.iter().filter(|(_, im)| im.abs() < 1e-7).count();
        prop_assert_eq!(count, real, "{}", f);
    }

    /// Discriminant is zero iff not square-free.
    #[test]
    fn prop_discriminant_zero_iff_repeated_root(f in arb_poly(4)) {
        let x = shared_ctx().symbol("x");
        prop_assume!(f.degree(&x).is_some_and(|d| d >= 2));
        let d = f.discriminant(&x).unwrap();
        let is_zero = format!("{d}") == "0";
        prop_assert_eq!(is_zero, f.is_squarefree(&x) == Some(false));
    }
}
