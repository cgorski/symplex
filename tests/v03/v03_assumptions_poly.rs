//! symplex 0.3.4 — exact sign facts for univariate polynomials with rational
//! coefficients in a real-assumed symbol, decided by square-free factoring
//! and Sturm sequences when the structural assumption rules are silent.

use symplex::prelude::*;

fn signs(e: &Ex) -> (Option<bool>, Option<bool>, Option<bool>, Option<bool>) {
    (
        e.is_positive(),
        e.is_nonnegative(),
        e.is_negative(),
        e.is_nonpositive(),
    )
}

#[test]
fn positive_definite_quadratics_are_known_positive() {
    let ctx = Context::new();
    let u = ctx.symbol_with("u", &[Assumption::NonNegative]);
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    // u ≥ 0: every coefficient positive, and one with a negative middle term.
    assert_eq!(
        signs(&(3 * &u.powi(2) + 2 * &u + 1)),
        (Some(true), Some(true), Some(false), Some(false))
    );
    assert_eq!(
        signs(&(3 * &u.powi(2) - 2 * &u + 1)),
        (Some(true), Some(true), Some(false), Some(false))
    );
    // Real x: x² − 2x + 2 = (x − 1)² + 1 > 0.
    assert_eq!(
        signs(&(&x.powi(2) - 2 * &x + 2)),
        (Some(true), Some(true), Some(false), Some(false))
    );
    assert_eq!((&x.powi(2) - 2 * &x + 2).is_nonzero(), Some(true));
    // Negative definite.
    assert_eq!(
        signs(&(-&x.powi(4) - 1)),
        (Some(false), Some(false), Some(true), Some(true))
    );
}

#[test]
fn zeros_inside_the_domain_are_respected() {
    let ctx = Context::new();
    let u = ctx.symbol_with("u", &[Assumption::NonNegative]);
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    // (x − 1)² ≥ 0 but touches zero: nonneg, not positive, not decided negative.
    assert_eq!(
        signs(&(&x.powi(2) - 2 * &x + 1)),
        (None, Some(true), Some(false), None)
    );
    // Sign changes inside the domain stay undecided.
    assert_eq!(signs(&(&x.powi(2) - 1)), (None, None, None, None));
    assert_eq!(signs(&(&u.powi(2) - &u)), (None, None, None, None));
    assert_eq!(signs(&(&p - &p.powi(2))), (None, None, None, None));
    // A root exactly at the excluded endpoint: p² + p > 0 for p > 0 (zero only at 0).
    assert_eq!(
        signs(&(&p.powi(2) + &p)),
        (Some(true), Some(true), Some(false), Some(false))
    );
    // The same polynomial for u ≥ 0 is only non-negative.
    assert_eq!(
        signs(&(&u.powi(2) + &u)),
        (None, Some(true), Some(false), None)
    );
}

#[test]
fn negative_domains_and_odd_polynomials() {
    let ctx = Context::new();
    let n = ctx.symbol_with("n", &[Assumption::Negative]);
    let m = ctx.symbol_with("m", &[Assumption::NonPositive]);
    // n³ − n = n(n − 1)(n + 1) changes sign at −1 inside n < 0.
    assert_eq!(signs(&(&n.powi(3) - &n)), (None, None, None, None));
    // n³ + n = n(n² + 1) < 0 for n < 0.
    assert_eq!(
        signs(&(&n.powi(3) + &n)),
        (Some(false), Some(false), Some(true), Some(true))
    );
    // m³ + m ≤ 0 for m ≤ 0 (zero at 0): non-positive, hence not positive.
    assert_eq!(
        signs(&(&m.powi(3) + &m)),
        (Some(false), None, None, Some(true))
    );
    // Odd degree over all of ℝ is never sign-definite.
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    assert_eq!(signs(&(&x.powi(3) + &x + 1)), (None, None, None, None));
}

#[test]
fn not_applied_without_realness_or_with_parameters() {
    let ctx = Context::new();
    let z = ctx.symbol("z"); // possibly complex
    assert_eq!(signs(&(&z.powi(2) + 1)), (None, None, None, None));
    let (x, a) = (ctx.symbol_with("x", &[Assumption::Real]), ctx.symbol("a"));
    // Two symbols: not a univariate polynomial with rational coefficients.
    assert_eq!(signs(&(&a * &x.powi(2) + 1)), (None, None, None, None));
    // Non-polynomial.
    assert_eq!(
        signs(&(&x.sin().powi(2) + 1)),
        (Some(true), Some(true), Some(false), Some(false))
    );
}

#[test]
fn downstream_simplifications_use_the_new_facts() {
    let ctx = Context::new();
    let u = ctx.symbol_with("u", &[Assumption::NonNegative]);
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let q = 3 * &u.powi(2) + 2 * &u + 1;
    assert_eq!(q.abs().simplify(), q);
    assert_eq!(q.powi(2).sqrt().simplify(), q);
    assert_eq!(q.sign().simplify(), ctx.int(1));
    assert_eq!(q.ln().is_real(), Some(true));
    assert_eq!(
        q.compare_numeric(&ctx.int(0)),
        Some(std::cmp::Ordering::Greater)
    );
    let r = &x.powi(2) - 2 * &x + 2;
    assert_eq!(r.abs().simplify(), r);
    assert_eq!(r.gt(&ctx.int(0)).eval().to_string(), "True");
    // Still honest where it cannot decide.
    let s = &x.powi(2) - 1;
    assert_eq!(s.abs().simplify(), s.abs());
}

#[test]
fn high_degree_is_decided_and_fast() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    // x^20 + x^10 + 1 > 0 on ℝ.
    let t = std::time::Instant::now();
    let p = &x.powi(20) + &x.powi(10) + 1;
    assert_eq!(p.is_positive(), Some(true));
    assert!(
        t.elapsed() < std::time::Duration::from_secs(2),
        "{:?}",
        t.elapsed()
    );
    // Beyond the degree guard the answer is simply unknown, never wrong.
    let big = &x.powi(30) + 1;
    assert!(matches!(big.is_positive(), None | Some(true)));
}
