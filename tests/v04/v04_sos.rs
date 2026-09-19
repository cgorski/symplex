//! symplex 0.6 — sums-of-squares certificates: `prove_sos`, the exact
//! Gram-matrix pipeline (numerical SDP, facial reduction, rounding,
//! rational LDLᵀ), refutation, JSON round trip and the Lean export.
//!
//! `fixtures/sos_certificates.lean` is the exact text these certificates
//! emitted when it was compiled against Mathlib (Lean 4.30.0) with
//! `linter.style.longLine` enabled; the emitter must reproduce it byte for
//! byte.

use num_traits::Signed;
use symplex::certificates::{SosCertificate, SosOpts, SosOutcome, is_sos, prove_sos};
use symplex::lean::LeanOpts;
use symplex::linprog::{q, qi};
use symplex::prelude::*;

const COMPILED: &str = include_str!("../fixtures/sos_certificates.lean");

fn proved(out: SosOutcome) -> SosCertificate {
    match out {
        SosOutcome::Proved(c) => {
            assert!(c.verify(), "certificate must re-verify: {c}");
            c
        }
        other => panic!("expected a certificate, got {other:?}"),
    }
}

fn sos(goal: &Ex, vars: &[Ex]) -> SosCertificate {
    proved(prove_sos(goal, vars, &SosOpts::default()).unwrap())
}

#[test]
fn emitted_lean_matches_the_mathlib_compiled_fixture() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let xy = [x.clone(), y.clone()];
    let xyz = [x.clone(), y.clone(), z.clone()];
    let cases: Vec<(&str, Ex, &[Ex])> = vec![
        (
            "two_squares",
            ((&x - 1).powi(2) + (&y - 1).powi(2)).expand(),
            &xy,
        ),
        ("pd_quadratic", x.powi(2) - &x * &y + y.powi(2) + 1, &xy),
        (
            "square_of_quadratic",
            (x.powi(2) - y.powi(2)).powi(2).expand(),
            &xy,
        ),
        (
            "product_of_squares",
            ((&x - &y).powi(2) * (&x + 1).powi(2)).expand(),
            &xy,
        ),
        (
            "univariate",
            ((x.powi(2) + 1) * (&x - 1).powi(2)).expand(),
            std::slice::from_ref(&x),
        ),
        (
            "quartic_pd",
            x.powi(4) + y.powi(4) + &x * &y * ctx.rational(1, 2) + 1,
            &xy,
        ),
        (
            "three_vars",
            (x.powi(2) + y.powi(2) + z.powi(2) - &x * &y - &y * &z - &x * &z).expand(),
            &xyz,
        ),
        (
            "amgm3",
            x.powi(4) + y.powi(4) + z.powi(4) - &x * &y * &z * 4 + 1,
            &xyz,
        ),
    ];
    let mut text = String::from("import Mathlib\nset_option linter.style.longLine true\n\n");
    for (name, goal, vars) in &cases {
        let c = sos(goal, vars);
        text.push_str(&c.to_lean(name).unwrap());
        text.push('\n');
    }
    assert_eq!(text, COMPILED);
    assert!(text.lines().all(|l| l.chars().count() <= 100));
}

#[test]
fn decompositions_are_exact_and_minimal_where_expected() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let xy = [x.clone(), y.clone()];
    // A perfect square is recovered as one square.
    let c = sos(&(x.powi(2) - y.powi(2)).powi(2).expand(), &xy);
    assert_eq!(c.rank(), 1);
    assert_eq!(c.squares()[0].0, qi(1));
    // (x − 1)² + (y − 1)²: rank 2, Gram matrix rank 2 on the basis (1, x, y).
    let c = sos(&((&x - 1).powi(2) + (&y - 1).powi(2)).expand(), &xy);
    assert_eq!(c.rank(), 2);
    assert_eq!(c.basis().len(), 3);
    assert_eq!(c.gram().rank(), 2);
    let (lhs, rhs) = c.identity();
    assert!((lhs - rhs).expand().is_zero_structural());
    // AM–GM in three variables: x⁴ + y⁴ + z⁴ + 1 ≥ 4xyz.
    let c = sos(
        &(x.powi(4) + y.powi(4) + z.powi(4) - &x * &y * &z * 4 + 1),
        &[x.clone(), y.clone(), z.clone()],
    );
    assert!(c.rank() <= 6);
    let hints = c.lean_hints(&LeanOpts::default()).unwrap();
    assert_eq!(hints.len(), c.rank());
    assert!(hints.iter().all(|h| h.starts_with("sq_nonneg (")));
    // The Gram matrix reproduces the goal and is PSD (checked by verify).
    assert!(c.gram().is_positive_semidefinite());
}

#[test]
fn refutations_and_honest_unknowns() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let xy = [x.clone(), y.clone()];
    // Negative somewhere: exact witness.
    match prove_sos(
        &(x.powi(4) + y.powi(4) - 4 * &x * &y + 1),
        &xy,
        &SosOpts::default(),
    )
    .unwrap()
    {
        SosOutcome::Refuted { point, value, .. } => {
            assert!(value.is_negative());
            let mut e = x.powi(4) + y.powi(4) - 4 * &x * &y + 1;
            for (v, q) in &point {
                e = e.subs(v, &ctx.from_ratio(q.clone()));
            }
            assert_eq!(e.eval().as_rational().unwrap(), value);
        }
        other => panic!("{other:?}"),
    }
    // Odd degree.
    assert_eq!(
        is_sos(&(x.powi(3) - &x + 1), std::slice::from_ref(&x)),
        Some(false)
    );
    // Motzkin: non-negative on ℝ² but not a sum of squares — Unknown, never Proved.
    let motzkin = x.powi(4) * y.powi(2) + x.powi(2) * y.powi(4) - 3 * x.powi(2) * y.powi(2) + 1;
    match prove_sos(&motzkin, &xy, &SosOpts::default()).unwrap() {
        SosOutcome::Unknown(u) => assert!(!u.reason.is_empty()),
        other => panic!("Motzkin must not be certified: {other:?}"),
    }
    // Errors.
    assert!(prove_sos(&x.sin(), std::slice::from_ref(&x), &SosOpts::default()).is_err());
    assert!(prove_sos(&(&x + &y), std::slice::from_ref(&x), &SosOpts::default()).is_err());
}

#[test]
fn random_sums_of_squares_are_recovered_and_negatives_never_proved() {
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> i64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) % 7) as i64 - 3
        }
    }
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let xy = [x.clone(), y.clone()];
    for seed in 1..=12u64 {
        let mut g = Lcg(seed * 7919);
        // Σ of two or three random quadratic squares (degree-4 goal).
        let k = 2 + (seed % 2) as usize;
        let mut goal = ctx.zero();
        for _ in 0..k {
            let p = x.powi(2) * g.next()
                + &x * &y * g.next()
                + y.powi(2) * g.next()
                + &x * g.next()
                + &y * g.next()
                + g.next();
            goal += p.powi(2);
        }
        let goal = goal.expand();
        if goal.is_zero_structural() {
            continue;
        }
        // The analytic centre maximises the Gram determinant, so the
        // decomposition found is generally *not* the k-term one; it is exact
        // nonetheless.
        let c = sos(&goal, &xy);
        assert!(c.rank() >= 1 && c.rank() <= c.basis().len());
        let lean = c.to_lean(&format!("random_{seed}")).unwrap();
        assert!(lean.lines().all(|l| l.chars().count() <= 100), "{lean}");
        // Subtract enough to make it negative somewhere: never Proved.
        let bad = &goal - 1000;
        match prove_sos(&bad, &xy, &SosOpts::default()).unwrap() {
            SosOutcome::Refuted { value, .. } => assert!(value.is_negative()),
            SosOutcome::Unknown { .. } => {}
            SosOutcome::Proved(c) => panic!("seed {seed}: proved a negative goal: {c}"),
        }
    }
}

#[test]
fn json_round_trip_reverifies_and_rejects_tampering() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let c = sos(
        &(x.powi(2) - &x * &y + y.powi(2) + 1),
        &[x.clone(), y.clone()],
    );
    let json = c.to_json().unwrap();
    let other = Context::new();
    let back = SosCertificate::from_json(&other, &json).unwrap();
    assert_eq!(back.to_string(), c.to_string());
    assert_eq!(back.to_lean("t").unwrap(), c.to_lean("t").unwrap());
    let mut d = c.to_data();
    d.gram[1] = "9/1".to_string(); // breaks the identity
    assert!(SosCertificate::from_data(&other, &d).is_err());
    let mut d = c.to_data();
    d.gram[0] = "-1/1".to_string(); // not PSD
    assert!(SosCertificate::from_data(&other, &d).is_err());
    assert!(SosCertificate::from_json(&other, "{").is_err());
    let _ = q(1, 2);
}
