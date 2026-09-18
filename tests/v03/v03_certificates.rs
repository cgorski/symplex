//! symplex 0.3.2 — `certificates`: exact Handelman certificates of
//! non-negativity on a box, exact refutation, and Lean 4 export.
//!
//! The Lean texts pinned in `lean_output_is_stable` were compiled against
//! Mathlib (Lean 4.30.0) when this file was written; a change here means the
//! emitted proof changed and should be re-checked, not just re-pinned.

use symplex::certificates::{
    BoxOutcome, Certificate, is_nonnegative_on_box, prove_nonnegative_on_box,
};
use symplex::linprog::Feasibility;
use symplex::num_bigint::BigInt;
use symplex::num_rational::Ratio;
use symplex::num_traits::Zero;
use symplex::poly_ex::Poly;
use symplex::prelude::*;

type Q = Ratio<BigInt>;

fn q(n: i64, d: i64) -> Q {
    Q::new(BigInt::from(n), BigInt::from(d))
}

fn proved(out: Result<BoxOutcome, SymplexError>) -> Certificate {
    match out {
        Ok(BoxOutcome::Proved(c)) => c,
        other => panic!("expected a certificate, got {other:?}"),
    }
}

/// Evaluate the certificate identity at random rational points of the box
/// and check the products are individually non-negative there.
fn check_identity_numerically(cert: &Certificate) {
    let ctx = cert.goal().context();
    let (lhs, rhs) = cert.identity();
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..25 {
        let mut subs: Vec<(Ex, Ex)> = Vec::new();
        for b in cert.bounds() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let t = q((seed % 1000) as i64, 1000);
            let (lo, hi) = (b.lo.as_rational().unwrap(), b.hi.as_rational().unwrap());
            let point = &lo + (&hi - &lo) * t;
            subs.push((b.var.clone(), ctx.from_ratio(point)));
        }
        let refs: Vec<(&Ex, &Ex)> = subs.iter().map(|(a, b)| (a, b)).collect();
        let l = lhs.subs_map(&refs).eval().as_rational().unwrap();
        let r = rhs.subs_map(&refs).eval().as_rational().unwrap();
        assert_eq!(l, r, "identity fails at {subs:?}");
        assert!(l >= Q::zero(), "goal negative at {subs:?}: {l}");
        for t in cert.terms() {
            let v = cert
                .product(t)
                .to_ex()
                .subs_map(&refs)
                .eval()
                .as_rational()
                .unwrap();
            assert!(v >= Q::zero(), "product negative on the box");
            assert!(t.weight > Q::zero());
        }
    }
}

fn unit_square(ctx: &Context, x: &Ex, y: &Ex) -> Vec<(Ex, Ex, Ex)> {
    vec![
        (x.clone(), ctx.int(0), ctx.int(1)),
        (y.clone(), ctx.int(0), ctx.int(1)),
    ]
}

// ── Proofs ──────────────────────────────────────────────────────────────

#[test]
fn quarter_bound_has_a_degree_two_certificate() {
    let ctx = Context::new();
    let (r, f) = (ctx.symbol("r"), ctx.symbol("f"));
    let goal = ctx.rational(1, 4) - (&r - &f / 2).powi(2);
    let bounds = [
        (r.clone(), ctx.int(0), ctx.rational(1, 2)),
        (f.clone(), ctx.int(0), ctx.int(1)),
    ];
    let cert = proved(prove_nonnegative_on_box(&goal, &bounds, 2));
    assert!(cert.verify());
    assert_eq!(cert.degree(), 2);
    assert!(!cert.terms().is_empty());
    check_identity_numerically(&cert);
    // Degree 1 is not enough for a quadratic goal.
    assert!(matches!(
        prove_nonnegative_on_box(&goal, &bounds, 1),
        Ok(BoxOutcome::Unknown { degree: 1, .. })
    ));
}

#[test]
fn single_product_certificates_are_found_exactly() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let bounds = [(x.clone(), ctx.int(0), ctx.int(1))];
    let cert = proved(prove_nonnegative_on_box(&(&x * (1 - &x)), &bounds, 2));
    assert_eq!(cert.terms().len(), 1);
    let t = &cert.terms()[0];
    assert_eq!(
        (t.lower_powers.clone(), t.upper_powers.clone()),
        (vec![1], vec![1])
    );
    assert_eq!(t.weight, q(1, 1));
    assert!(cert.verify());
}

#[test]
fn cubic_with_roots_outside_the_box() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let goal = (&x - 1) * (&x - 2) * (&x - 3);
    let cert = proved(prove_nonnegative_on_box(
        &goal,
        &[(x.clone(), ctx.int(3), ctx.int(10))],
        3,
    ));
    assert!(cert.verify());
    check_identity_numerically(&cert);
    // Same goal on [2, 10] is false (negative on (2, 3)).
    match prove_nonnegative_on_box(&goal, &[(x.clone(), ctx.int(2), ctx.int(10))], 3).unwrap() {
        BoxOutcome::Refuted { point, value } => {
            assert!(value < Q::zero());
            assert!(point[0] > q(2, 1) && point[0] < q(3, 1), "{point:?}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn three_variables_degree_two() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    // 3 − xy − yz − zx ≥ 0 on the unit cube (each product ≤ 1).
    let goal = 3 - &x * &y - &y * &z - &z * &x;
    let bounds = [
        (x.clone(), ctx.int(0), ctx.int(1)),
        (y.clone(), ctx.int(0), ctx.int(1)),
        (z.clone(), ctx.int(0), ctx.int(1)),
    ];
    let cert = proved(prove_nonnegative_on_box(&goal, &bounds, 2));
    assert!(cert.verify());
    check_identity_numerically(&cert);
}

#[test]
fn certificates_prefer_sparse_low_degree_products() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // 3x + 2y − 1 on [1/2, 3] × [0, 1] = 3(x − 1/2) + 2y + 1/2: three terms
    // (constant, x − 1/2, y), none using the upper bounds.
    let goal = 3 * &x + 2 * &y - 1;
    let bounds = [
        (x.clone(), ctx.rational(1, 2), ctx.int(3)),
        (y.clone(), ctx.int(0), ctx.int(1)),
    ];
    let cert = proved(prove_nonnegative_on_box(&goal, &bounds, 2));
    assert_eq!(cert.degree(), 1, "{cert}");
    assert_eq!(cert.terms().len(), 3, "{cert}");
    assert!(
        cert.terms()
            .iter()
            .all(|t| t.upper_powers.iter().all(|&e| e == 0))
    );
}

// ── Refutation and unknown ──────────────────────────────────────────────

#[test]
fn false_inequalities_are_refuted_with_exact_points() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let bounds = unit_square(&ctx, &x, &y);
    match prove_nonnegative_on_box(&(&x * &y - ctx.rational(1, 2)), &bounds, 2).unwrap() {
        BoxOutcome::Refuted { point, value } => {
            assert_eq!(point.len(), 2);
            let v = &point[0] * &point[1] - q(1, 2);
            assert_eq!(v, value);
            assert!(value < Q::zero());
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        is_nonnegative_on_box(&(&x * &y - 1), &bounds, 3),
        Some(false)
    );
    assert_eq!(
        is_nonnegative_on_box(&(1 - &x * &y), &bounds, 3),
        Some(true)
    );
}

#[test]
fn interior_zero_has_no_handelman_certificate() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // (x − 1)² + (y − 1)² touches zero at the interior point (1, 1):
    // non-negative, but Handelman certificates cannot represent it.
    let goal = (&x - 1).powi(2) + (&y - 1).powi(2);
    let bounds = [
        (x.clone(), ctx.int(0), ctx.int(2)),
        (y.clone(), ctx.int(0), ctx.int(2)),
    ];
    for d in 1..=3 {
        match prove_nonnegative_on_box(&goal, &bounds, d).unwrap() {
            BoxOutcome::Unknown { degree, farkas } => {
                assert_eq!(degree, d);
                assert!(
                    farkas.is_some(),
                    "a Farkas vector is expected at degree {d}"
                );
            }
            other => panic!("degree {d}: {other:?}"),
        }
    }
    assert_eq!(is_nonnegative_on_box(&goal, &bounds, 3), None);
}

// ── Input validation ────────────────────────────────────────────────────

#[test]
fn invalid_inputs_are_errors() {
    let ctx = Context::new();
    let (x, y, a) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a"));
    let ok = [(x.clone(), ctx.int(0), ctx.int(1))];
    assert!(prove_nonnegative_on_box(&x, &[], 2).is_err());
    // Non-literal bound.
    assert!(prove_nonnegative_on_box(&x, &[(x.clone(), ctx.int(0), a.clone())], 2).is_err());
    // lo ≥ hi.
    assert!(prove_nonnegative_on_box(&x, &[(x.clone(), ctx.int(1), ctx.int(1))], 2).is_err());
    // Repeated variable.
    assert!(
        prove_nonnegative_on_box(
            &x,
            &[
                (x.clone(), ctx.int(0), ctx.int(1)),
                (x.clone(), ctx.int(0), ctx.int(2))
            ],
            2
        )
        .is_err()
    );
    // Goal mentions a symbol that is not bounded / non-polynomial goal.
    assert!(prove_nonnegative_on_box(&(&x + &y), &ok, 2).is_err());
    assert!(prove_nonnegative_on_box(&(&a * &x), &ok, 2).is_err());
    assert!(prove_nonnegative_on_box(&x.sin(), &ok, 2).is_err());
    assert_eq!(is_nonnegative_on_box(&x.sin(), &ok, 2), None);
}

// ── Poly::express_as_nonneg_combination ─────────────────────────────────

#[test]
fn express_as_nonneg_combination_matches_hand_computation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let goal = (&x + 1).powi(2).as_poly(&[&x]).unwrap();
    let b1 = (&x + 1).as_poly(&[&x]).unwrap();
    let b2 = (&x.powi(2) - 1).as_poly(&[&x]).unwrap();
    match goal.express_as_nonneg_combination(&[&b1, &b2]).unwrap() {
        Feasibility::Feasible(w) => assert_eq!(w, vec![q(2, 1), q(1, 1)]),
        other => panic!("{other:?}"),
    }
    // x² + 1 is not a non-negative combination of x + 1 and x² − 1
    // (it would need a negative multiple of x + 1 to cancel the x term).
    let g2 = (&x.powi(2) + 1).as_poly(&[&x]).unwrap();
    assert!(
        !g2.express_as_nonneg_combination(&[&b1, &b2])
            .unwrap()
            .is_feasible()
    );
    // Errors.
    assert!(goal.express_as_nonneg_combination(&[]).is_err());
    let y = ctx.symbol("y");
    let other_gens = Poly::new(&(&y + 1), &[&y]).unwrap();
    assert!(goal.express_as_nonneg_combination(&[&other_gens]).is_err());
}

// ── Lean export ─────────────────────────────────────────────────────────

#[test]
fn lean_output_is_stable() {
    let ctx = Context::new();
    let (r, f, x, y) = (
        ctx.symbol("r"),
        ctx.symbol("f"),
        ctx.symbol("x"),
        ctx.symbol("y"),
    );

    let quarter = proved(prove_nonnegative_on_box(
        &(ctx.rational(1, 4) - (&r - &f / 2).powi(2)),
        &[
            (r.clone(), ctx.int(0), ctx.rational(1, 2)),
            (f.clone(), ctx.int(0), ctx.int(1)),
        ],
        2,
    ));
    assert_eq!(
        quarter.to_lean("quarter_bound").unwrap(),
        "theorem quarter_bound (r f : ℝ) (h_r_lo : (0 : ℝ) ≤ r) (h_r_hi : r ≤ (1 / 2 : ℝ)) (h_f_lo : (0 : ℝ) ≤ f) (h_f_hi : f ≤ (1 : ℝ)) :\n    0 ≤ -(f ^ 2 / 4) + f * r - r ^ 2 + (1 / 4 : ℝ) := by\n  nlinarith [mul_nonneg (sub_nonneg.mpr h_r_hi) (sub_nonneg.mpr h_f_hi), mul_nonneg (sub_nonneg.mpr h_f_lo) (sub_nonneg.mpr h_f_hi), mul_nonneg (sub_nonneg.mpr h_r_lo) (sub_nonneg.mpr h_r_hi), mul_nonneg (sub_nonneg.mpr h_r_lo) (sub_nonneg.mpr h_f_lo)]\n"
    );

    let one_product = proved(prove_nonnegative_on_box(
        &(&x * (1 - &x)),
        &[(x.clone(), ctx.int(0), ctx.int(1))],
        2,
    ));
    assert_eq!(
        one_product.to_lean("x_one_minus_x").unwrap(),
        "theorem x_one_minus_x (x : ℝ) (h_x_lo : (0 : ℝ) ≤ x) (h_x_hi : x ≤ (1 : ℝ)) :\n    0 ≤ -x ^ 2 + x := by\n  nlinarith [mul_nonneg (sub_nonneg.mpr h_x_lo) (sub_nonneg.mpr h_x_hi)]\n"
    );

    // Linear certificate: `linarith`, unused bounds underscored.
    let linear = proved(prove_nonnegative_on_box(
        &(3 * &x + 2 * &y - 1),
        &[
            (x.clone(), ctx.rational(1, 2), ctx.int(3)),
            (y.clone(), ctx.int(0), ctx.int(1)),
        ],
        1,
    ));
    assert_eq!(
        linear.to_lean("linear_only").unwrap(),
        "theorem linear_only (x y : ℝ) (h_x_lo : (1 / 2 : ℝ) ≤ x) (_h_x_hi : x ≤ (3 : ℝ)) (h_y_lo : (0 : ℝ) ≤ y) (_h_y_hi : y ≤ (1 : ℝ)) :\n    0 ≤ 3 * x + 2 * y - 1 := by\n  linarith [sub_nonneg.mpr h_y_lo, sub_nonneg.mpr h_x_lo]\n"
    );
}

#[test]
fn lean_theorem_names_and_identifiers_are_sanitised() {
    let ctx = Context::new();
    let odd = ctx.symbol("x-1");
    let cert = proved(prove_nonnegative_on_box(
        &(&odd * (1 - &odd)),
        &[(odd.clone(), ctx.int(0), ctx.int(1))],
        2,
    ));
    let text = cert.to_lean("my theorem").unwrap();
    assert!(
        text.starts_with("theorem «my theorem» («x-1» : ℝ)"),
        "{text}"
    );
    assert!(text.contains("h_x-1_lo"), "{text}");
}

#[test]
fn certificate_display_shows_the_identity_and_the_box() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cert = proved(prove_nonnegative_on_box(
        &(&x * (1 - &x)),
        &[(x.clone(), ctx.int(0), ctx.int(1))],
        2,
    ));
    assert_eq!(cert.to_string(), "-x^2 + x = x*(-x + 1), 0 ≤ x ≤ 1");
    let (lhs, rhs) = cert.identity();
    assert_eq!((lhs - rhs).expand().to_string(), "0");
}
