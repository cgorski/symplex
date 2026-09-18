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
        "theorem quarter_bound (r f : ℝ) (h_r_lo : (0 : ℝ) ≤ r) (h_r_hi : r ≤ (1 / 2 : ℝ))\n    (h_f_lo : (0 : ℝ) ≤ f) (h_f_hi : f ≤ (1 : ℝ)) :\n    0 ≤ -(f ^ 2 / 4) + f * r - r ^ 2 + (1 / 4 : ℝ) := by\n  nlinarith [mul_nonneg (sub_nonneg.mpr h_r_hi) (sub_nonneg.mpr h_f_hi),\n    mul_nonneg (sub_nonneg.mpr h_f_lo) (sub_nonneg.mpr h_f_hi),\n    mul_nonneg (sub_nonneg.mpr h_r_lo) (sub_nonneg.mpr h_r_hi),\n    mul_nonneg (sub_nonneg.mpr h_r_lo) (sub_nonneg.mpr h_f_lo)]\n"
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
        "theorem linear_only (x y : ℝ) (h_x_lo : (1 / 2 : ℝ) ≤ x) (_h_x_hi : x ≤ (3 : ℝ))\n    (h_y_lo : (0 : ℝ) ≤ y) (_h_y_hi : y ≤ (1 : ℝ)) :\n    0 ≤ 3 * x + 2 * y - 1 := by\n  linarith [sub_nonneg.mpr h_y_lo, sub_nonneg.mpr h_x_lo]\n"
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

// ═══════════════════════════════════════════════════════════════════════════
// Square factors in box certificates (interior even-multiplicity zeros)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn box_certificate_splits_off_a_square_factor() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let square = unit_square(&ctx, &x, &y);
    // (x − 1/2)² · (1 − xy): a double zero along x = 1/2 inside the box.
    let goal = (&x - ctx.rational(1, 2)).powi(2) * (1 - &x * &y);
    let cert = proved(prove_nonnegative_on_box(&goal, &square, 2));
    let g = cert.square().expect("square factor");
    // factor_list_all returns the primitive factor 2x − 1.
    assert_eq!(g.to_string(), "Poly(2*x - 1, x, y)");
    assert!(cert.verify());
    check_identity_numerically(&cert);
    let lean = cert.to_lean("sq_box").unwrap();
    assert!(
        lean.contains("mul_nonneg (sq_nonneg (2 * x - 1))"),
        "{lean}"
    );
    // A pure square: h is the constant 1/4.
    let pure = proved(prove_nonnegative_on_box(
        &(&x - ctx.rational(1, 2)).powi(2),
        &square[..1],
        2,
    ));
    assert_eq!(
        pure.to_string(),
        "x^2 - x + 1/4 = 1/4*(2*x - 1)^2, 0 ≤ x ≤ 1"
    );
    assert_eq!(
        pure.to_lean("pure_square").unwrap(),
        "theorem pure_square (x : ℝ) (_h_x_lo : (0 : ℝ) ≤ x) (_h_x_hi : x ≤ (1 : ℝ)) :\n    0 ≤ x ^ 2 - x + (1 / 4 : ℝ) := by\n  nlinarith [sq_nonneg (2 * x - 1)]\n"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Half-lines and the real line
// ═══════════════════════════════════════════════════════════════════════════

use symplex::certificates::{
    HalfLineCertificate, HalfLineOutcome, Ray, prove_nonnegative_on_halfline,
    prove_nonnegative_on_reals,
};

fn hl_proved(out: Result<HalfLineOutcome, SymplexError>) -> HalfLineCertificate {
    match out {
        Ok(HalfLineOutcome::Proved(c)) => c,
        other => panic!("expected a half-line certificate, got {other:?}"),
    }
}

/// Sample the half-line and check the identity and the sign of the goal.
fn check_halfline_numerically(cert: &HalfLineCertificate) {
    let ctx = cert.goal().context();
    let (lhs, rhs) = cert.identity();
    let a = cert.endpoint().as_rational().unwrap();
    for i in 0..40 {
        let t = q(i * i, 7); // 0, 1/7, 4/7, …
        let x = match cert.ray() {
            Ray::AtLeast => &a + &t,
            Ray::AtMost => &a - &t,
        };
        let xe = ctx.from_ratio(x);
        let l = lhs.subs(cert.var(), &xe).eval().as_rational().unwrap();
        let r = rhs.subs(cert.var(), &xe).eval().as_rational().unwrap();
        assert_eq!(l, r);
        let g = cert
            .goal()
            .to_ex()
            .subs(cert.var(), &xe)
            .eval()
            .as_rational()
            .unwrap();
        assert!(g >= Q::zero());
    }
}

#[test]
fn halfline_shift_certificate_reads_off_coefficients() {
    let ctx = Context::new();
    let j = ctx.symbol("j");
    let p = (&j - 1) * (&j - 3);
    let cert = hl_proved(prove_nonnegative_on_halfline(
        &p,
        &j,
        &ctx.int(3),
        Ray::AtLeast,
        10,
    ));
    assert_eq!(cert.polya_power(), 0);
    assert_eq!(
        cert.coefficients()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["0", "2", "1"]
    );
    assert!(cert.square().is_none());
    assert!(cert.verify());
    check_halfline_numerically(&cert);
    assert_eq!(
        cert.to_lean("shift_only").unwrap(),
        "theorem shift_only (j : ℝ) (h_j_lo : (3 : ℝ) ≤ j) :\n    0 ≤ j ^ 2 - 4 * j + 3 := by\n  have hk : 0 ≤ j - (3 : ℝ) := sub_nonneg.mpr h_j_lo\n  nlinarith [hk, pow_nonneg hk 2]\n"
    );
}

#[test]
fn halfline_polya_multiplier_when_shift_alone_fails() {
    let ctx = Context::new();
    let j = ctx.symbol("j");
    // j² − j + 1 > 0 on [0, ∞) but has a negative coefficient: (1 + j)(j² − j + 1) = j³ + 1.
    let cert = hl_proved(prove_nonnegative_on_halfline(
        &(&j.powi(2) - &j + 1),
        &j,
        &ctx.int(0),
        Ray::AtLeast,
        10,
    ));
    assert_eq!(cert.polya_power(), 1);
    assert_eq!(
        cert.coefficients()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["1", "0", "0", "1"]
    );
    assert!(cert.verify());
    check_halfline_numerically(&cert);
    assert_eq!(
        cert.to_lean("polya_needed").unwrap(),
        "theorem polya_needed (j : ℝ) (h_j_lo : (0 : ℝ) ≤ j) :\n    0 ≤ j ^ 2 - j + 1 := by\n  have hk : 0 ≤ j - (0 : ℝ) := sub_nonneg.mpr h_j_lo\n  have hpos : 0 < (1 + (j - (0 : ℝ))) ^ 1 := pow_pos (by linarith) 1\n  have hprod : 0 ≤ (1 + (j - (0 : ℝ))) ^ 1 * (j ^ 2 - j + 1) := by nlinarith [pow_nonneg hk 3]\n  exact nonneg_of_mul_nonneg_right hprod hpos\n"
    );
    // A tighter minimum needs a larger exponent; the budget is respected.
    let tight = 4 * &j.powi(2) - 6 * &j + 3;
    let cert = hl_proved(prove_nonnegative_on_halfline(
        &tight,
        &j,
        &ctx.int(0),
        Ray::AtLeast,
        20,
    ));
    assert_eq!(cert.polya_power(), 12);
    assert!(cert.verify());
    assert!(matches!(
        prove_nonnegative_on_halfline(&tight, &j, &ctx.int(0), Ray::AtLeast, 5).unwrap(),
        HalfLineOutcome::Unknown { max_polya_power: 5 }
    ));
}

#[test]
fn halfline_square_factor_for_interior_double_zero() {
    let ctx = Context::new();
    let j = ctx.symbol("j");
    // (j − 5)²(j² + 1) on [3, ∞): zero at 5 inside the half-line.
    let goal = (&j - 5).powi(2) * (&j.powi(2) + 1);
    let cert = hl_proved(prove_nonnegative_on_halfline(
        &goal,
        &j,
        &ctx.int(3),
        Ray::AtLeast,
        10,
    ));
    assert_eq!(
        cert.square().map(ToString::to_string),
        Some("Poly(j - 5, j)".into())
    );
    assert_eq!(cert.polya_power(), 0);
    assert!(cert.verify());
    check_halfline_numerically(&cert);
    assert_eq!(
        cert.to_lean("square_inside").unwrap(),
        "theorem square_inside (j : ℝ) (h_j_lo : (3 : ℝ) ≤ j) :\n    0 ≤ j ^ 4 - 10 * j ^ 3 + 26 * j ^ 2 - 10 * j + 25 := by\n  have hk : 0 ≤ j - (3 : ℝ) := sub_nonneg.mpr h_j_lo\n  nlinarith [sq_nonneg (j - 5), mul_nonneg (sq_nonneg (j - 5)) (hk),\n    mul_nonneg (sq_nonneg (j - 5)) (pow_nonneg hk 2)]\n"
    );
}

#[test]
fn halfline_at_most_direction_and_rational_endpoint() {
    let ctx = Context::new();
    let j = ctx.symbol("j");
    let cert = hl_proved(prove_nonnegative_on_halfline(
        &((3 - &j) * (5 - &j)),
        &j,
        &ctx.int(3),
        Ray::AtMost,
        10,
    ));
    assert_eq!(*cert.ray(), Ray::AtMost);
    assert_eq!(cert.shift_expr().to_string(), "-j + 3");
    assert!(cert.verify());
    check_halfline_numerically(&cert);
    let lean = cert.to_lean("at_most").unwrap();
    assert!(lean.contains("(h_j_hi : j ≤ (3 : ℝ))"), "{lean}");
    assert!(
        lean.contains("have hk : 0 ≤ (3 : ℝ) - j := sub_nonneg.mpr h_j_hi"),
        "{lean}"
    );

    let lin = hl_proved(prove_nonnegative_on_halfline(
        &(2 * &j - 1),
        &j,
        &ctx.rational(1, 2),
        Ray::AtLeast,
        10,
    ));
    assert_eq!(
        lin.to_lean("rational_endpoint").unwrap(),
        "theorem rational_endpoint (j : ℝ) (h_j_lo : (1 / 2 : ℝ) ≤ j) :\n    0 ≤ 2 * j - 1 := by\n  have hk : 0 ≤ j - (1 / 2 : ℝ) := sub_nonneg.mpr h_j_lo\n  linarith [hk]\n"
    );
    // Constant goal.
    let c = hl_proved(prove_nonnegative_on_halfline(
        &ctx.int(7),
        &j,
        &ctx.int(0),
        Ray::AtLeast,
        1,
    ));
    assert_eq!(c.coefficients(), &[q(7, 1)]);
}

#[test]
fn halfline_refutation_and_errors() {
    let ctx = Context::new();
    let (j, a) = (ctx.symbol("j"), ctx.symbol("a"));
    match prove_nonnegative_on_halfline(&(&j - 4), &j, &ctx.int(3), Ray::AtLeast, 10).unwrap() {
        HalfLineOutcome::Refuted { point, value } => {
            assert!(point >= q(3, 1));
            assert!(value < Q::zero());
            assert_eq!(&point - q(4, 1), value);
        }
        other => panic!("{other:?}"),
    }
    // Sign change strictly inside the half-line.
    let p = (&j - 1) * (&j - 3);
    match prove_nonnegative_on_halfline(&p, &j, &ctx.int(2), Ray::AtLeast, 10).unwrap() {
        HalfLineOutcome::Refuted { point, value } => {
            // p(2) = −1 already refutes it; any point of [2, 3) would do.
            assert!(point >= q(2, 1) && point < q(3, 1), "{point}");
            assert!(value < Q::zero());
        }
        other => panic!("{other:?}"),
    }
    // Errors: symbolic endpoint, parameter coefficient, non-polynomial.
    assert!(prove_nonnegative_on_halfline(&j, &j, &a, Ray::AtLeast, 5).is_err());
    assert!(prove_nonnegative_on_halfline(&(&a * &j), &j, &ctx.int(0), Ray::AtLeast, 5).is_err());
    assert!(prove_nonnegative_on_halfline(&j.sin(), &j, &ctx.int(0), Ray::AtLeast, 5).is_err());
}

#[test]
fn real_line_certificate_splits_into_two_halves() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cert = prove_nonnegative_on_reals(&(&x.powi(2) - &x + 1), &x, &ctx.int(0), 10)
        .unwrap()
        .unwrap();
    assert!(cert.verify());
    assert_eq!(cert.upper.polya_power(), 1);
    assert_eq!(cert.lower.polya_power(), 0);
    let lean = cert.to_lean("pos_quadratic_reals").unwrap();
    assert!(lean.starts_with("theorem pos_quadratic_reals (x : ℝ) : 0 ≤ x ^ 2 - x + 1 := by\n  rcases le_total (0 : ℝ) x with h_x_lo | h_x_hi\n"), "{lean}");
    assert!(lean.contains("exact nonneg_of_mul_nonneg_right hprod hpos"));
    assert!(lean.contains("nlinarith [hk, pow_nonneg hk 2]"));
    // Splitting at 0 puts the double zero at 2 inside the upper half-line,
    // which then needs the square factor; the lower half does not.
    let sq =
        prove_nonnegative_on_reals(&((&x - 2).powi(2) * (&x.powi(2) + 3)), &x, &ctx.int(0), 10)
            .unwrap()
            .unwrap();
    assert!(sq.verify());
    assert_eq!(
        sq.upper.square().map(ToString::to_string),
        Some("Poly(x - 2, x)".into())
    );
    assert!(sq.lower.square().is_none());
    // Splitting exactly at the zero needs no square factor on either side.
    let at_zero =
        prove_nonnegative_on_reals(&((&x - 2).powi(2) * (&x.powi(2) + 3)), &x, &ctx.int(2), 10)
            .unwrap()
            .unwrap();
    assert!(at_zero.upper.square().is_none() && at_zero.lower.square().is_none());
    // A polynomial that is negative somewhere has no real-line certificate.
    assert!(
        prove_nonnegative_on_reals(&(&x.powi(2) - 1), &x, &ctx.int(0), 10)
            .unwrap()
            .is_none()
    );
}

#[test]
fn lean_output_respects_mathlib_line_width() {
    use symplex::lean::{MATHLIB_LINE_WIDTH, wrap_lean};
    let ctx = Context::new();
    let (r, f) = (ctx.symbol("r"), ctx.symbol("f"));
    let cert = proved(prove_nonnegative_on_box(
        &(ctx.rational(1, 4) - (&r - &f / 2).powi(2)),
        &[
            (r.clone(), ctx.int(0), ctx.rational(1, 2)),
            (f.clone(), ctx.int(0), ctx.int(1)),
        ],
        2,
    ));
    let text = cert.to_lean("quarter_bound").unwrap();
    assert!(
        text.lines()
            .all(|l| l.chars().count() <= MATHLIB_LINE_WIDTH),
        "{text}"
    );
    // Hint lists break after commas and keep `nlinarith [` together.
    assert!(text.contains("  nlinarith [mul_nonneg"), "{text}");
    assert!(text.contains("),\n    mul_nonneg"), "{text}");
    // Wrapping is idempotent and never changes the token stream.
    assert_eq!(wrap_lean(&text, MATHLIB_LINE_WIDTH), text);
    fn tokens(t: &str) -> Vec<&str> {
        t.split_whitespace().collect()
    }
    let narrow = wrap_lean(&text, 60);
    assert!(narrow.lines().all(|l| l.chars().count() <= 60), "{narrow}");
    assert_eq!(tokens(&narrow), tokens(&text));
    // Guillemet identifiers are never split.
    let odd = wrap_lean(
        "theorem t («a long name» : ℝ) («another long one» : ℝ) : 0 ≤ 1 := by\n  linarith\n",
        40,
    );
    assert!(
        odd.contains("«a long name»") && odd.contains("«another long one»"),
        "{odd}"
    );
}
