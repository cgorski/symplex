//! symplex 0.3 — the public sparse polynomial view `Poly` (`symplex::poly_ex`)
//! and the exact interval sign tests `poly_is_nonnegative_on` /
//! `poly_is_positive_on`.

use num_bigint::BigInt;
use num_rational::Ratio;
use proptest::prelude::*;
use symplex::matrix::Matrix;
use symplex::multipoly::MultiPoly;
use symplex::poly_ex::Poly;
use symplex::polysys::linsolve_matrix;
use symplex::prelude::*;

/// Wall-clock hang guard (scaled up on shared CI runners).
fn time_budget(secs: u64) -> std::time::Duration {
    let mult = if std::env::var_os("CI").is_some() {
        5
    } else {
        1
    };
    std::time::Duration::from_secs(secs * mult)
}

fn s<T: std::fmt::Display>(e: &T) -> String {
    format!("{e}")
}

fn rat(p: i64, q: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(p), BigInt::from(q))
}

fn ctx_xy() -> (Context, Ex, Ex) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    (ctx, x, y)
}

/// `j·r² + (j + 1)·r·f + 3` — the motivating parametric polynomial.
fn motivating() -> (Context, Ex, Ex, Ex, Ex) {
    let ctx = Context::new();
    let (j, r, f) = (ctx.symbol("j"), ctx.symbol("r"), ctx.symbol("f"));
    let e = &j * r.powi(2) + (&j + 1) * &r * &f + 3;
    (ctx, j, r, f, e)
}

// ═══════════════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn new_univariate_with_symbolic_coefficients() {
    let (_ctx, j, r, _f, e) = motivating();
    let p = Poly::new(&e, &[&r]).unwrap();
    assert_eq!(p.num_gens(), 1);
    assert_eq!(p.num_terms(), 3);
    assert_eq!(p.degree_in(&r), Some(2));
    assert_eq!(p.leading_coeff(), j);
    assert!(!p.has_rational_coeffs());
}

#[test]
fn new_bivariate_view_of_the_same_expression() {
    let (ctx, j, r, f, e) = motivating();
    let p = Poly::new(&e, &[&r, &f]).unwrap();
    // Terms: j·r², (j+1)·r·f, 3
    assert_eq!(p.num_terms(), 3);
    assert_eq!(p.coeff_monomial(&[2, 0]).unwrap(), j);
    assert_eq!(p.coeff_monomial(&[1, 1]).unwrap(), (&j + 1).eval());
    assert_eq!(p.coeff_monomial(&[0, 0]).unwrap(), ctx.int(3));
}

#[test]
fn new_all_symbols_as_generators_gives_rational_coefficients() {
    let (_ctx, j, r, f, e) = motivating();
    let p = Poly::new(&e, &[&r, &f, &j]).unwrap();
    assert!(p.has_rational_coeffs());
    assert_eq!(p.num_terms(), 4); // j r², j r f, r f, 3
    assert_eq!(p.total_degree(), Some(3));
}

#[test]
fn as_poly_is_new() {
    let (_ctx, x, y) = ctx_xy();
    let e = &x * &y + 1;
    let a = e.as_poly(&[&x, &y]).unwrap();
    let b = Poly::new(&e, &[&x, &y]).unwrap();
    assert!(a.equals(&b));
}

#[test]
fn new_expands_products_and_powers() {
    let (_ctx, x, y) = ctx_xy();
    let e = (&x + &y).powi(3) * (&x - &y);
    let p = Poly::new(&e, &[&x, &y]).unwrap();
    assert_eq!(p.total_degree(), Some(4));
    assert!(p.is_homogeneous());
    assert_eq!(p.to_ex(), e.expand());
}

#[test]
fn new_none_for_generator_inside_function() {
    let ctx = Context::new();
    let r = ctx.symbol("r");
    assert!(Poly::new(&r.sin(), &[&r]).is_none());
    assert!(Poly::new(&(r.exp() + &r), &[&r]).is_none());
    assert!(Poly::new(&r.ln(), &[&r]).is_none());
}

#[test]
fn new_none_for_negative_power_of_generator() {
    let ctx = Context::new();
    let r = ctx.symbol("r");
    assert!(Poly::new(&r.powi(-1), &[&r]).is_none());
    assert!(Poly::new(&(ctx.int(1) / &r + r.powi(2)), &[&r]).is_none());
}

#[test]
fn new_none_for_symbolic_or_fractional_power_of_generator() {
    let ctx = Context::new();
    let (r, f) = (ctx.symbol("r"), ctx.symbol("f"));
    assert!(Poly::new(&r.pow(&f), &[&r]).is_none());
    assert!(Poly::new(&r.sqrt(), &[&r]).is_none());
    // …but r^f is fine as a coefficient when only f... no: f is in the exponent.
    assert!(Poly::new(&r.pow(&f), &[&f]).is_none());
    // A power of a non-generator is an opaque coefficient.
    assert!(Poly::new(&(r.pow(&f) * ctx.symbol("z")), &[&ctx.symbol("z")]).is_some());
}

#[test]
fn new_none_for_generator_in_exponent() {
    let ctx = Context::new();
    let (r, a) = (ctx.symbol("r"), ctx.symbol("a"));
    assert!(Poly::new(&a.pow(&r), &[&r]).is_none());
    assert!(Poly::new(&ctx.int(2).pow(&r), &[&r]).is_none());
}

#[test]
fn new_none_for_empty_duplicate_or_non_symbol_generators() {
    let (ctx, x, y) = ctx_xy();
    let e = &x + &y;
    assert!(Poly::new(&e, &[]).is_none());
    assert!(Poly::new(&e, &[&x, &x]).is_none());
    assert!(Poly::new(&e, &[&(&x + 1)]).is_none());
    assert!(Poly::new(&e, &[&ctx.int(2)]).is_none());
    assert!(Poly::new(&e, &[&x.sin()]).is_none());
}

#[test]
fn new_zero_polynomial() {
    let (ctx, x, _y) = ctx_xy();
    let p = Poly::new(&ctx.zero(), &[&x]).unwrap();
    assert!(p.is_zero());
    assert_eq!(p.num_terms(), 0);
    assert_eq!(p.total_degree(), None);
    assert_eq!(p.degree_in(&x), None);
    assert_eq!(p.leading_coeff(), ctx.zero());
    assert!(p.leading_term().is_none());
    assert!(p.to_ex().is_zero_structural());
    // Cancelling terms also give zero.
    let q = Poly::new(&((&x + 1).powi(2) - x.powi(2) - &x * 2 - 1), &[&x]).unwrap();
    assert!(q.is_zero());
}

#[test]
fn new_constant_polynomial() {
    let (ctx, x, _y) = ctx_xy();
    let a = ctx.symbol("a");
    let p = Poly::new(&(&a + 5), &[&x]).unwrap();
    assert!(p.is_ground());
    assert!(!p.is_zero());
    assert_eq!(p.total_degree(), Some(0));
    assert_eq!(p.degree_in(&x), Some(0));
    assert_eq!(p.leading_coeff(), (&a + 5).eval());
}

#[test]
fn from_terms_sums_duplicates_and_drops_zeros() {
    let (ctx, x, y) = ctx_xy();
    let p = Poly::from_terms(
        &ctx,
        &[&x, &y],
        vec![
            (vec![1, 0], ctx.int(2)),
            (vec![1, 0], ctx.int(3)),
            (vec![0, 1], ctx.int(4)),
            (vec![0, 1], ctx.int(-4)),
        ],
    )
    .unwrap();
    assert_eq!(p.num_terms(), 1);
    assert_eq!(p.to_ex(), &x * 5);
}

#[test]
fn from_terms_rejects_bad_input() {
    let (ctx, x, y) = ctx_xy();
    assert!(Poly::from_terms(&ctx, &[], vec![]).is_err());
    assert!(Poly::from_terms(&ctx, &[&x], vec![(vec![1, 1], ctx.int(1))]).is_err());
    assert!(Poly::from_terms(&ctx, &[&x], vec![(vec![1], y.clone())]).is_ok());
    // A coefficient may not mention a generator.
    assert!(Poly::from_terms(&ctx, &[&x], vec![(vec![1], x.clone())]).is_err());
    assert!(Poly::from_terms(&ctx, &[&x, &y], vec![(vec![0, 0], x.sin())]).is_err());
}

#[test]
fn zero_one_constant_constructors() {
    let (ctx, x, y) = ctx_xy();
    let z = Poly::zero(&ctx, &[&x, &y]).unwrap();
    assert!(z.is_zero());
    let o = Poly::one(&ctx, &[&x, &y]).unwrap();
    assert!(o.is_ground() && !o.is_zero());
    assert_eq!(o.to_ex(), ctx.one());
    let c = Poly::constant(&ctx, &[&x], &ctx.rational(2, 3)).unwrap();
    assert_eq!(c.to_ex(), ctx.rational(2, 3));
    assert!(Poly::constant(&ctx, &[&x], &x).is_err());
    assert!(Poly::zero(&ctx, &[]).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Term access
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn terms_are_lex_descending() {
    let (_ctx, x, y) = ctx_xy();
    let e = &y.powi(3) + &x * &y + x.powi(2) * 2 + 7;
    let p = Poly::new(&e, &[&x, &y]).unwrap();
    let monoms = p.monoms();
    assert_eq!(monoms, vec![vec![2, 0], vec![1, 1], vec![0, 3], vec![0, 0]]);
    let coeffs: Vec<String> = p.coeffs().iter().map(s).collect();
    assert_eq!(coeffs, ["2", "1", "1", "7"]);
    let terms = p.terms();
    assert_eq!(terms.len(), 4);
    assert_eq!(terms[0].0, vec![2, 0]);
    assert_eq!(s(&terms[0].1), "2");
}

#[test]
fn terms_order_depends_on_generator_order() {
    let (_ctx, x, y) = ctx_xy();
    let e = &x * y.powi(2) + x.powi(2) * &y;
    let pxy = Poly::new(&e, &[&x, &y]).unwrap();
    let pyx = Poly::new(&e, &[&y, &x]).unwrap();
    assert_eq!(pxy.leading_monomial(), Some(vec![2, 1]));
    assert_eq!(pyx.leading_monomial(), Some(vec![2, 1]));
    // Same expression, different generator order → different exponent vectors.
    assert_eq!(pxy.monoms(), vec![vec![2, 1], vec![1, 2]]);
    assert!(!pxy.equals(&pyx));
}

#[test]
fn coeff_monomial_present_absent_and_wrong_length() {
    let (ctx, x, y) = ctx_xy();
    let a = ctx.symbol("a");
    let p = Poly::new(&(&a * &x * &y - &y * 3 + 1), &[&x, &y]).unwrap();
    assert_eq!(p.coeff_monomial(&[1, 1]).unwrap(), a);
    assert_eq!(p.coeff_monomial(&[0, 1]).unwrap(), ctx.int(-3));
    assert_eq!(p.coeff_monomial(&[0, 0]).unwrap(), ctx.int(1));
    assert_eq!(p.coeff_monomial(&[5, 0]).unwrap(), ctx.zero());
    assert!(p.coeff_monomial(&[1]).is_err());
    assert!(p.coeff_monomial(&[1, 1, 1]).is_err());
}

#[test]
fn degrees() {
    let (_ctx, x, y) = ctx_xy();
    let z = _ctx.symbol("z");
    let e = x.powi(3) * &y + y.powi(2) * z.powi(4) + 1;
    let p = Poly::new(&e, &[&x, &y, &z]).unwrap();
    assert_eq!(p.total_degree(), Some(6));
    assert_eq!(p.degree_in(&x), Some(3));
    assert_eq!(p.degree_in(&y), Some(2));
    assert_eq!(p.degree_in(&z), Some(4));
    assert_eq!(p.degree_list(), vec![3, 2, 4]);
    // Not a generator.
    assert_eq!(p.degree_in(&_ctx.symbol("w")), None);
}

#[test]
fn leading_term_coeff_monomial_lex() {
    let (ctx, x, y) = ctx_xy();
    let a = ctx.symbol("a");
    // In lex with x > y, x·y⁵ beats y⁹ even though its total degree is lower.
    let e = &a * &x * y.powi(5) - y.powi(9) * 4;
    let p = Poly::new(&e, &[&x, &y]).unwrap();
    assert_eq!(p.leading_monomial(), Some(vec![1, 5]));
    assert_eq!(p.leading_coeff(), a);
    let (m, c) = p.leading_term().unwrap();
    assert_eq!(m, vec![1, 5]);
    assert_eq!(c, a);
}

#[test]
fn all_coeffs_dense_highest_first() {
    let (ctx, x, _y) = ctx_xy();
    let a = ctx.symbol("a");
    let p = Poly::new(&(&a * x.powi(3) - &x + 2), &[&x]).unwrap();
    let cs: Vec<String> = p.all_coeffs().unwrap().iter().map(s).collect();
    assert_eq!(cs, ["a", "0", "-1", "2"]);
    // Zero → [0]; multivariate → None.
    let z = Poly::zero(&ctx, &[&x]).unwrap();
    assert_eq!(z.all_coeffs().unwrap(), vec![ctx.zero()]);
    let m = Poly::new(&(&x + ctx.symbol("y")), &[&x, &ctx.symbol("y")]).unwrap();
    assert!(m.all_coeffs().is_none());
}

#[test]
fn all_coeffs_matches_ex_coeffs_reversed() {
    let (_ctx, x, _y) = ctx_xy();
    let e = (&x * 3 - 2).powi(4) + &x;
    let mut from_ex = e.coeffs(&x).unwrap();
    from_ex.reverse();
    let from_poly = Poly::new(&e, &[&x]).unwrap().all_coeffs().unwrap();
    assert_eq!(from_ex, from_poly);
}

// ═══════════════════════════════════════════════════════════════════════════
// Predicates
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn predicates_univariate_linear_ground() {
    let (ctx, x, y) = ctx_xy();
    let lin = Poly::new(&(&x * 2 + &y - 1), &[&x, &y]).unwrap();
    assert!(lin.is_linear());
    assert!(!lin.is_univariate());
    assert!(!lin.is_ground());
    assert!(!lin.is_homogeneous());
    let bil = Poly::new(&(&x * &y), &[&x, &y]).unwrap();
    assert!(!bil.is_linear(), "xy has total degree 2");
    assert!(bil.is_homogeneous());
    let uni = Poly::new(&(&x + 1), &[&x]).unwrap();
    assert!(uni.is_univariate());
    let z = Poly::zero(&ctx, &[&x]).unwrap();
    assert!(z.is_zero() && z.is_ground() && z.is_linear() && z.is_homogeneous());
}

#[test]
fn has_rational_coeffs() {
    let (ctx, x, _y) = ctx_xy();
    let a = ctx.symbol("a");
    assert!(
        Poly::new(&(x.powi(2) * ctx.rational(1, 3) + 1), &[&x])
            .unwrap()
            .has_rational_coeffs()
    );
    assert!(
        !Poly::new(&(x.powi(2) * &a + 1), &[&x])
            .unwrap()
            .has_rational_coeffs()
    );
    assert!(
        !Poly::new(&(x.powi(2) * ctx.pi()), &[&x])
            .unwrap()
            .has_rational_coeffs()
    );
    assert!(Poly::zero(&ctx, &[&x]).unwrap().has_rational_coeffs());
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_exact_rationals() {
    let (ctx, x, y) = ctx_xy();
    // x²y − y/2 + 1/3 at (x, y) = (3/2, −4/5)
    let e = x.powi(2) * &y - &y / 2 + ctx.rational(1, 3);
    let p = Poly::new(&e, &[&x, &y]).unwrap();
    let v = p
        .eval(&[&ctx.rational(3, 2), &ctx.rational(-4, 5)])
        .unwrap();
    // 9/4 · (−4/5) + 2/5 + 1/3 = −9/5 + 2/5 + 1/3 = −7/5 + 1/3 = −16/15
    assert_eq!(v.as_rational().unwrap(), rat(-16, 15));
}

#[test]
fn eval_matches_substitution_into_expression() {
    let (ctx, x, y) = ctx_xy();
    let e = (&x - &y * 2).powi(3) + &x * &y;
    let p = Poly::new(&e, &[&x, &y]).unwrap();
    for (px, py) in [(1i64, 2i64), (-3, 5), (7, -1), (0, 0)] {
        let via_poly = p.eval(&[&ctx.int(px), &ctx.int(py)]).unwrap();
        let via_subs = e.subs_map(&[(&x, &ctx.int(px)), (&y, &ctx.int(py))]).eval();
        assert_eq!(via_poly, via_subs);
    }
}

#[test]
fn eval_with_symbolic_coefficients_and_values() {
    let (ctx, x, _y) = ctx_xy();
    let a = ctx.symbol("a");
    let p = Poly::new(&(&a * x.powi(2) + &x), &[&x]).unwrap();
    let v = p.eval(&[&ctx.int(3)]).unwrap();
    assert_eq!(v, (&a * 9 + 3).eval());
    // Evaluating at another symbol just substitutes.
    let t = ctx.symbol("t");
    let v = p.eval(&[&t]).unwrap();
    assert_eq!(v, (&a * t.powi(2) + &t).eval());
}

#[test]
fn eval_wrong_arity_is_error() {
    let (ctx, x, y) = ctx_xy();
    let p = Poly::new(&(&x + &y), &[&x, &y]).unwrap();
    assert!(p.eval(&[&ctx.int(1)]).is_err());
    assert!(p.eval(&[]).is_err());
    assert!(p.eval(&[&ctx.int(1), &ctx.int(2), &ctx.int(3)]).is_err());
}

#[test]
fn eval_zero_polynomial_is_zero() {
    let (ctx, x, _y) = ctx_xy();
    let z = Poly::zero(&ctx, &[&x]).unwrap();
    assert!(z.eval(&[&ctx.int(17)]).unwrap().is_zero_structural());
}

#[test]
fn eval_gen_removes_the_generator() {
    let (ctx, x, y) = ctx_xy();
    let p = Poly::new(&(x.powi(2) * &y + &x + &y), &[&x, &y]).unwrap();
    let q = p.eval_gen(&x, &ctx.int(2)).unwrap();
    assert_eq!(q.gens(), std::slice::from_ref(&y));
    assert_eq!(q.to_ex(), &y * 5 + 2);
    let r = p.eval_gen(&y, &ctx.rational(1, 2)).unwrap();
    assert_eq!(r.gens(), std::slice::from_ref(&x));
    assert_eq!(r.to_ex(), x.powi(2) / 2 + &x + ctx.rational(1, 2));
}

#[test]
fn eval_gen_with_a_polynomial_value_in_the_other_generator() {
    let (_ctx, x, y) = ctx_xy();
    let p = Poly::new(&(x.powi(2) + &y), &[&x, &y]).unwrap();
    // x := y + 1  →  (y + 1)² + y = y² + 3y + 1
    let q = p.eval_gen(&x, &(&y + 1)).unwrap();
    assert_eq!(q.to_ex(), y.powi(2) + &y * 3 + 1);
}

#[test]
fn eval_gen_last_generator_gives_ground_poly() {
    let (ctx, x, _y) = ctx_xy();
    let p = Poly::new(&(x.powi(2) + 1), &[&x]).unwrap();
    let g = p.eval_gen(&x, &ctx.int(3)).unwrap();
    assert_eq!(g.num_gens(), 0);
    assert!(g.is_ground());
    assert_eq!(g.to_ex(), ctx.int(10));
}

#[test]
fn eval_gen_errors() {
    let (ctx, x, y) = ctx_xy();
    let p = Poly::new(&(x.powi(2) + &y), &[&x, &y]).unwrap();
    // Not a generator.
    assert!(p.eval_gen(&ctx.symbol("z"), &ctx.int(1)).is_err());
    // Value mentions the generator itself.
    assert!(p.eval_gen(&x, &(&x + 1)).is_err());
    // Value not polynomial in the remaining generator.
    assert!(p.eval_gen(&x, &y.sin()).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn add_sub_mul_basic() {
    let (_ctx, x, y) = ctx_xy();
    let p = Poly::new(&(&x + &y), &[&x, &y]).unwrap();
    let q = Poly::new(&(&x - &y), &[&x, &y]).unwrap();
    assert_eq!(p.add(&q).unwrap().to_ex(), &x * 2);
    assert_eq!(p.sub(&q).unwrap().to_ex(), &y * 2);
    assert_eq!(p.mul(&q).unwrap().to_ex(), x.powi(2) - y.powi(2));
    assert!(p.sub(&p).unwrap().is_zero());
}

#[test]
fn arithmetic_with_symbolic_coefficients_expands_coefficients() {
    let (ctx, x, _y) = ctx_xy();
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let p = Poly::new(&(&a * &x + &b), &[&x]).unwrap();
    let q = Poly::new(&(&b * &x + &a), &[&x]).unwrap();
    let prod = p.mul(&q).unwrap();
    // (ax + b)(bx + a) = ab x² + (a² + b²) x + ab
    assert_eq!(prod.coeff_monomial(&[2]).unwrap(), (&a * &b).eval());
    assert_eq!(
        prod.coeff_monomial(&[1]).unwrap(),
        (a.powi(2) + b.powi(2)).eval()
    );
    assert_eq!(prod.coeff_monomial(&[0]).unwrap(), (&a * &b).eval());
    // And it agrees with expanding the expression product.
    let direct = Poly::new(&(p.to_ex() * q.to_ex()), &[&x]).unwrap();
    assert!(prod.equals(&direct));
}

#[test]
fn arithmetic_gens_mismatch_is_error() {
    let (_ctx, x, y) = ctx_xy();
    let p = Poly::new(&(&x + 1), &[&x]).unwrap();
    let q = Poly::new(&(&y + 1), &[&y]).unwrap();
    let pq = Poly::new(&(&x + &y), &[&x, &y]).unwrap();
    assert!(p.add(&q).is_err());
    assert!(p.sub(&pq).is_err());
    assert!(p.mul(&q).is_err());
    assert!(Poly::monomial_basis(&[&p, &q]).is_err());
}

#[test]
fn neg_scale_pow() {
    let (ctx, x, y) = ctx_xy();
    let a = ctx.symbol("a");
    let p = Poly::new(&(&x - &y), &[&x, &y]).unwrap();
    assert_eq!(p.neg().to_ex(), &y - &x);
    assert!(p.neg().add(&p).unwrap().is_zero());
    assert_eq!(p.scale(&ctx.int(3)).unwrap().to_ex(), &x * 3 - &y * 3);
    assert_eq!(p.scale(&a).unwrap().coeff_monomial(&[1, 0]).unwrap(), a);
    assert!(p.scale(&x).is_err(), "scale by a generator is rejected");
    assert_eq!(p.pow(0).unwrap().to_ex(), ctx.one());
    assert!(p.pow(1).unwrap().equals(&p));
    assert_eq!(p.pow(3).unwrap().to_ex(), (&x - &y).powi(3).expand());
    assert_eq!(p.pow(5).unwrap().num_terms(), 6);
}

#[test]
fn derivative_partial() {
    let (ctx, x, y) = ctx_xy();
    let a = ctx.symbol("a");
    let p = Poly::new(&(&a * x.powi(3) * &y + y.powi(2) - &x), &[&x, &y]).unwrap();
    let dx = p.derivative(&x).unwrap();
    assert_eq!(dx.to_ex(), &a * x.powi(2) * &y * 3 - 1);
    let dy = p.derivative(&y).unwrap();
    assert_eq!(dy.to_ex(), &a * x.powi(3) + &y * 2);
    // Matches Ex::diff.
    assert_eq!(dx.to_ex(), p.to_ex().diff(&x).expand());
    assert!(p.derivative(&a).is_err());
    // Derivative of a constant is zero.
    assert!(
        Poly::one(&ctx, &[&x])
            .unwrap()
            .derivative(&x)
            .unwrap()
            .is_zero()
    );
}

#[test]
fn derivative_commutes_with_multiplication_product_rule() {
    let (_ctx, x, y) = ctx_xy();
    let p = Poly::new(&(x.powi(2) + &y), &[&x, &y]).unwrap();
    let q = Poly::new(&(&x * &y - 1), &[&x, &y]).unwrap();
    let lhs = p.mul(&q).unwrap().derivative(&x).unwrap();
    let rhs = p
        .derivative(&x)
        .unwrap()
        .mul(&q)
        .unwrap()
        .add(&p.mul(&q.derivative(&x).unwrap()).unwrap())
        .unwrap();
    assert!(lhs.equals(&rhs));
}

// ═══════════════════════════════════════════════════════════════════════════
// Round trips and equality
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn to_ex_round_trip_rational() {
    let (_ctx, x, y) = ctx_xy();
    let e = (&x * 2 - &y + 3).powi(2) * (&x + 1);
    let p = Poly::new(&e, &[&x, &y]).unwrap();
    let back = Poly::new(&p.to_ex(), &[&x, &y]).unwrap();
    assert!(p.equals(&back));
    assert_eq!(p.to_ex(), e.expand());
}

#[test]
fn to_ex_round_trip_symbolic_coefficients() {
    let (ctx, j, r, f, e) = motivating();
    let _ = ctx;
    for gens in [vec![&r], vec![&f], vec![&r, &f], vec![&f, &r], vec![&j, &r]] {
        let p = Poly::new(&e, &gens).unwrap();
        let back = Poly::new(&p.to_ex(), &gens).unwrap();
        assert!(p.equals(&back), "{p} vs {back}");
        assert!((p.to_ex() - &e).expand().is_zero_structural());
    }
}

#[test]
fn to_ex_round_trip_after_arithmetic() {
    let (ctx, j, r, f, e) = motivating();
    let _ = (ctx, j);
    let p = Poly::new(&e, &[&r, &f]).unwrap();
    let q = p.mul(&p).unwrap().add(&p.derivative(&r).unwrap()).unwrap();
    let back = Poly::new(&q.to_ex(), &[&r, &f]).unwrap();
    assert!(q.equals(&back));
}

#[test]
fn equals_is_structural_on_normalised_coefficients() {
    let (ctx, x, _y) = ctx_xy();
    let a = ctx.symbol("a");
    let p = Poly::new(&((&a + 1) * &x), &[&x]).unwrap();
    let q = Poly::from_terms(&ctx, &[&x], vec![(vec![1], &a + 1)]).unwrap();
    let r = Poly::from_terms(
        &ctx,
        &[&x],
        vec![(vec![1], a.clone()), (vec![1], ctx.one())],
    )
    .unwrap();
    assert!(p.equals(&q));
    assert!(p.equals(&r));
    let different = Poly::new(&((&a + 2) * &x), &[&x]).unwrap();
    assert!(!p.equals(&different));
}

#[test]
fn display_lists_expression_then_generators() {
    let (_ctx, x, y) = ctx_xy();
    let p = Poly::new(&(&x * &y + 1), &[&x, &y]).unwrap();
    assert_eq!(s(&p), "Poly(x*y + 1, x, y)");
    let (_ctx2, j, r, f, e) = motivating();
    let _ = (j, f);
    let q = Poly::new(&e, &[&r]).unwrap();
    let shown = s(&q);
    assert!(
        shown.starts_with("Poly(") && shown.ends_with(", r)"),
        "{shown}"
    );
}

/// `Display` follows `terms()` (lex-descending), not the arena's canonical
/// order, so the leading term comes first and coefficients are printed with
/// their sign folded into the separator.
#[test]
fn display_prints_terms_in_lex_descending_order() {
    let (ctx, x, y) = ctx_xy();
    let p = Poly::new(&(&x + 2 * &y).powi(3), &[&x, &y]).unwrap();
    assert_eq!(s(&p), "Poly(x^3 + 6*x^2*y + 12*x*y^2 + 8*y^3, x, y)");
    // Leading term first even when the arena would print it elsewhere.
    let q = Poly::new(&(&x.powi(3) + 4 * &x.powi(2) * &y - &y.powi(3)), &[&x, &y]).unwrap();
    assert_eq!(s(&q), "Poly(x^3 + 4*x^2*y - y^3, x, y)");
    // Negative leading coefficient, rational coefficients, constant term.
    let r = Poly::new(
        &(-&x.powi(2) + ctx.rational(3, 2) * &x - ctx.rational(1, 4)),
        &[&x],
    )
    .unwrap();
    assert_eq!(s(&r), "Poly(-x^2 + 3/2*x - 1/4, x)");
    // Symbolic coefficients: sums are parenthesised, negatives fold into
    // the separator, constant terms are printed bare.
    let (_c, j, rr, f, e) = motivating();
    let m = Poly::new(&e, &[&rr, &f]).unwrap();
    assert_eq!(s(&m), "Poly(j*r^2 + (j + 1)*r*f + 3, r, f)");
    let n = Poly::new(&(-2 * &j * &rr + &j + 1), &[&rr]).unwrap();
    assert_eq!(s(&n), "Poly(-2*j*r + j + 1, r)");
    // Zero polynomial.
    let z = Poly::zero(&ctx, &[&x, &y]).unwrap();
    assert_eq!(s(&z), "Poly(0, x, y)");
    // Round trip: the printed body re-parses to the same polynomial.
    let body = s(&q);
    let body = &body["Poly(".len()..body.len() - ", x, y".len() - 1];
    let back = Poly::new(&ctx.parse(body).unwrap(), &[&x, &y]).unwrap();
    assert!(back.equals(&q), "{body}");
}

#[test]
fn debug_and_clone_work() {
    let (_ctx, x, _y) = ctx_xy();
    let p = Poly::new(&(&x + 1), &[&x]).unwrap();
    let q = p.clone();
    assert!(p.equals(&q));
    assert!(format!("{p:?}").contains("Poly"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Content, primitive part, monic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn content_and_primitive_rational() {
    let (ctx, x, y) = ctx_xy();
    // 4x²y − 6xy + 10y  =  2 · (2x²y − 3xy + 5y)
    let p = Poly::new(&(x.powi(2) * &y * 4 - &x * &y * 6 + &y * 10), &[&x, &y]).unwrap();
    let (c, prim) = p.content_and_primitive().unwrap();
    assert_eq!(c, ctx.int(2));
    assert_eq!(prim.to_ex(), x.powi(2) * &y * 2 - &x * &y * 3 + &y * 5);
    // Fractions: x/2 + 1/3 = (1/6)(3x + 2)
    let q = Poly::new(&(&x / 2 + ctx.rational(1, 3)), &[&x]).unwrap();
    let (c, prim) = q.content_and_primitive().unwrap();
    assert_eq!(c, ctx.rational(1, 6));
    assert_eq!(prim.to_ex(), &x * 3 + 2);
    // Negative leading coefficient moves the sign into the content.
    let n = Poly::new(&(x.powi(2) * -4 + &x * 6), &[&x]).unwrap();
    let (c, prim) = n.content_and_primitive().unwrap();
    assert_eq!(c, ctx.int(-2));
    assert_eq!(prim.to_ex(), x.powi(2) * 2 - &x * 3);
}

#[test]
fn content_and_primitive_symbolic_is_none_zero_is_zero() {
    let (ctx, x, _y) = ctx_xy();
    let a = ctx.symbol("a");
    assert!(
        Poly::new(&(&a * &x), &[&x])
            .unwrap()
            .content_and_primitive()
            .is_none()
    );
    let (c, prim) = Poly::zero(&ctx, &[&x])
        .unwrap()
        .content_and_primitive()
        .unwrap();
    assert!(c.is_zero_structural() && prim.is_zero());
}

#[test]
fn monic_divides_by_leading_coefficient() {
    let (ctx, x, y) = ctx_xy();
    let p = Poly::new(&(x.powi(2) * 2 + &x * &y * 4 - 6), &[&x, &y]).unwrap();
    let m = p.monic().unwrap();
    assert_eq!(m.leading_coeff(), ctx.one());
    assert_eq!(m.to_ex(), x.powi(2) + &x * &y * 2 - 3);
    assert!(Poly::zero(&ctx, &[&x]).unwrap().monic().is_none());
    assert!(
        Poly::new(&(ctx.symbol("a") * &x), &[&x])
            .unwrap()
            .monic()
            .is_none()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Bridge to MultiPoly (Gröbner machinery)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn to_multipoly_from_multipoly_round_trip() {
    let (ctx, x, y) = ctx_xy();
    let e = x.powi(2) * &y * ctx.rational(3, 4) - &y.powi(2) + &x * 5 - 1;
    let p = Poly::new(&e, &[&x, &y]).unwrap();
    let mp = p.to_multipoly().unwrap();
    assert_eq!(mp.num_vars(), 2);
    assert_eq!(mp.num_terms(), 4);
    assert_eq!(mp.coeff(&[2, 1]), Some(&rat(3, 4)));
    let back = Poly::from_multipoly(&ctx, &[&x, &y], &mp).unwrap();
    assert!(back.equals(&p));
    assert_eq!(back.to_ex(), e.expand());
}

#[test]
fn to_multipoly_none_for_symbolic_coefficients() {
    let (ctx, x, _y) = ctx_xy();
    let a = ctx.symbol("a");
    assert!(
        Poly::new(&(&a * &x), &[&x])
            .unwrap()
            .to_multipoly()
            .is_none()
    );
    assert!(
        Poly::new(&(ctx.pi() * &x), &[&x])
            .unwrap()
            .to_multipoly()
            .is_none()
    );
}

#[test]
fn from_multipoly_arity_mismatch_is_error() {
    let (ctx, x, y) = ctx_xy();
    let mp: MultiPoly = MultiPoly::var(2, 0);
    assert!(Poly::from_multipoly(&ctx, &[&x], &mp).is_err());
    assert!(Poly::from_multipoly(&ctx, &[&x, &y], &mp).is_ok());
    assert!(Poly::from_multipoly(&ctx, &[&x, &x], &mp).is_err());
}

#[test]
fn multipoly_gcd_through_the_bridge() {
    let (ctx, x, y) = ctx_xy();
    let f = Poly::new(&((&x + &y) * (&x - &y)), &[&x, &y]).unwrap();
    let g = Poly::new(&((&x + &y).powi(2)), &[&x, &y]).unwrap();
    let d = MultiPoly::gcd(&f.to_multipoly().unwrap(), &g.to_multipoly().unwrap());
    let d = Poly::from_multipoly(&ctx, &[&x, &y], &d).unwrap();
    assert_eq!(d.to_ex(), &x + &y);
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric roots
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nroots_univariate_rational() {
    let (_ctx, x, _y) = ctx_xy();
    let p = Poly::new(&((&x - 1) * (&x + 2) * (x.powi(2) + 1)), &[&x]).unwrap();
    let roots = p.nroots(15).unwrap();
    assert_eq!(roots.len(), 4);
    let reals: Vec<f64> = roots
        .iter()
        .filter(|(_, im)| im.abs() < 1e-9)
        .map(|(re, _)| *re)
        .collect();
    assert_eq!(reals.len(), 2);
    assert!((reals[0] + 2.0).abs() < 1e-9 && (reals[1] - 1.0).abs() < 1e-9);
}

#[test]
fn nroots_rejects_multivariate_symbolic_and_constant() {
    let (ctx, x, y) = ctx_xy();
    assert!(
        Poly::new(&(&x + &y), &[&x, &y])
            .unwrap()
            .nroots(10)
            .is_err()
    );
    assert!(
        Poly::new(&(ctx.symbol("a") * &x + 1), &[&x])
            .unwrap()
            .nroots(10)
            .is_err()
    );
    assert!(Poly::one(&ctx, &[&x]).unwrap().nroots(10).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Families: monomial basis and coefficient matrix
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn monomial_basis_is_union_lex_descending() {
    let (_ctx, x, y) = ctx_xy();
    let p1 = Poly::new(&(&x * &y + 1), &[&x, &y]).unwrap();
    let p2 = Poly::new(&(x.powi(2) - &y), &[&x, &y]).unwrap();
    let basis = Poly::monomial_basis(&[&p1, &p2]).unwrap();
    assert_eq!(basis, vec![vec![2, 0], vec![1, 1], vec![0, 1], vec![0, 0]]);
    assert!(Poly::monomial_basis(&[]).is_err());
}

#[test]
fn coefficient_matrix_layout() {
    let (ctx, x, _y) = ctx_xy();
    let p1 = Poly::new(&(&x + 1), &[&x]).unwrap();
    let p2 = Poly::new(&(x.powi(2) - 1), &[&x]).unwrap();
    let basis = Poly::monomial_basis(&[&p1, &p2]).unwrap();
    let m = Poly::coefficient_matrix(&[&p1, &p2], &basis).unwrap();
    assert_eq!(m.shape(), (3, 2));
    assert_eq!(m, matrix![ctx, [0, 1], [1, 0], [1, -1]]);
    // Wrong exponent length or empty inputs are errors.
    assert!(Poly::coefficient_matrix(&[&p1], &[vec![1, 1]]).is_err());
    assert!(Poly::coefficient_matrix(&[], &basis).is_err());
    assert!(Poly::coefficient_matrix(&[&p1], &[]).is_err());
}

/// Solve `goal = Σ λⱼ hⱼ` exactly and return the λ's in order.
fn certificate(goal: &Poly, hs: &[&Poly]) -> Option<Vec<Ex>> {
    let mut all: Vec<&Poly> = hs.to_vec();
    all.push(goal);
    let basis = Poly::monomial_basis(&all).unwrap();
    let m = Poly::coefficient_matrix(hs, &basis).unwrap();
    let rhs: Vec<Ex> = basis
        .iter()
        .map(|mono| goal.coeff_monomial(mono).unwrap())
        .collect();
    let b = Matrix::col_vector(rhs);
    match linsolve_matrix(&m, &b).unwrap() {
        LinearSolution::Unique(pairs) => Some(pairs.into_iter().map(|(_, v)| v).collect()),
        LinearSolution::Parametric { .. } => None,
        LinearSolution::Inconsistent => None,
    }
}

#[test]
fn coefficient_matrix_solves_a_univariate_certificate() {
    let (ctx, x, _y) = ctx_xy();
    let h1 = Poly::new(&(&x + 1), &[&x]).unwrap();
    let h2 = Poly::new(&(x.powi(2) - 1), &[&x]).unwrap();
    // (x + 1)² = 2·(x + 1) + 1·(x² − 1)
    let goal = Poly::new(&(&x + 1).powi(2), &[&x]).unwrap();
    let lam = certificate(&goal, &[&h1, &h2]).unwrap();
    assert_eq!(lam, vec![ctx.int(2), ctx.int(1)]);
    // Nonnegative multipliers: a valid Positivstellensatz-style certificate.
    assert!(lam.iter().all(|l| l.as_rational().unwrap() >= rat(0, 1)));
    // Reconstruct.
    let rebuilt = h1
        .scale(&lam[0])
        .unwrap()
        .add(&h2.scale(&lam[1]).unwrap())
        .unwrap();
    assert!(rebuilt.equals(&goal));
}

#[test]
fn coefficient_matrix_solves_a_multivariate_certificate_with_fractions() {
    let (ctx, x, y) = ctx_xy();
    let h1 = Poly::new(&(x.powi(2) + &y * 2), &[&x, &y]).unwrap();
    let h2 = Poly::new(&(&x * &y - &y), &[&x, &y]).unwrap();
    let h3 = Poly::new(&(y.powi(2) + 1), &[&x, &y]).unwrap();
    // goal = (1/2) h1 + 3 h2 + (2/3) h3
    let goal = h1
        .scale(&ctx.rational(1, 2))
        .unwrap()
        .add(&h2.scale(&ctx.int(3)).unwrap())
        .unwrap()
        .add(&h3.scale(&ctx.rational(2, 3)).unwrap())
        .unwrap();
    let lam = certificate(&goal, &[&h1, &h2, &h3]).unwrap();
    assert_eq!(
        lam,
        vec![ctx.rational(1, 2), ctx.int(3), ctx.rational(2, 3)]
    );
}

#[test]
fn coefficient_matrix_detects_impossible_certificate() {
    let (_ctx, x, _y) = ctx_xy();
    let h1 = Poly::new(&(&x + 1), &[&x]).unwrap();
    let h2 = Poly::new(&(&x * 2 + 2), &[&x]).unwrap();
    // x² is not in the span of {x + 1, 2x + 2}.
    let goal = Poly::new(&x.powi(2), &[&x]).unwrap();
    assert!(certificate(&goal, &[&h1, &h2]).is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// Interval sign tests on Ex
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nonneg_perfect_square_on_the_real_line() {
    let (ctx, x, _y) = ctx_xy();
    let sq = x.powi(2) - &x * 2 + 1;
    assert_eq!(
        sq.poly_is_nonnegative_on(&x, &ctx.neg_infinity(), &ctx.infinity()),
        Some(true)
    );
    assert_eq!(
        sq.poly_is_positive_on(&x, &ctx.neg_infinity(), &ctx.infinity()),
        Some(false)
    );
}

#[test]
fn nonneg_cubic_on_half_lines() {
    let (ctx, x, _y) = ctx_xy();
    let f = x.powi(3) - &x;
    let inf = ctx.infinity();
    assert_eq!(f.poly_is_nonnegative_on(&x, &ctx.int(2), &inf), Some(true));
    assert_eq!(f.poly_is_nonnegative_on(&x, &ctx.int(1), &inf), Some(true));
    assert_eq!(f.poly_is_positive_on(&x, &ctx.int(1), &inf), Some(false));
    assert_eq!(f.poly_is_positive_on(&x, &ctx.int(2), &inf), Some(true));
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.int(-2), &inf),
        Some(false)
    );
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.neg_infinity(), &ctx.int(-1)),
        Some(false)
    );
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.neg_infinity(), &ctx.int(-2)),
        Some(false)
    );
    // −(x³ − x) ≤ 0 on [2, ∞)
    assert_eq!(
        (-&f).poly_is_nonnegative_on(&x, &ctx.int(2), &inf),
        Some(false)
    );
}

#[test]
fn positive_irreducible_quadratic_everywhere() {
    let (ctx, x, _y) = ctx_xy();
    let f = x.powi(2) + 1;
    assert_eq!(
        f.poly_is_positive_on(&x, &ctx.neg_infinity(), &ctx.infinity()),
        Some(true)
    );
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.neg_infinity(), &ctx.infinity()),
        Some(true)
    );
    assert_eq!(
        (-&f).poly_is_positive_on(&x, &ctx.neg_infinity(), &ctx.infinity()),
        Some(false)
    );
}

#[test]
fn sign_on_bounded_intervals_between_roots() {
    let (ctx, x, _y) = ctx_xy();
    let f = x.powi(3) - &x; // roots −1, 0, 1; negative on (0, 1), positive on (−1, 0)
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.int(-1), &ctx.int(0)),
        Some(true)
    );
    assert_eq!(
        f.poly_is_positive_on(&x, &ctx.int(-1), &ctx.int(0)),
        Some(false)
    );
    assert_eq!(
        f.poly_is_positive_on(&x, &ctx.rational(-1, 2), &ctx.rational(-1, 4)),
        Some(true)
    );
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.int(0), &ctx.int(1)),
        Some(false)
    );
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.rational(1, 2), &ctx.int(1)),
        Some(false)
    );
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.rational(-1, 2), &ctx.rational(1, 2)),
        Some(false)
    );
}

#[test]
fn even_multiplicity_root_inside_interval() {
    let (ctx, x, _y) = ctx_xy();
    // (x − 1)²·(x + 3) ≥ 0 on [−3, ∞), touching zero at 1.
    let f = ((&x - 1).powi(2) * (&x + 3)).expand();
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.int(-3), &ctx.infinity()),
        Some(true)
    );
    assert_eq!(
        f.poly_is_positive_on(&x, &ctx.int(0), &ctx.int(2)),
        Some(false)
    );
    assert_eq!(
        f.poly_is_positive_on(&x, &ctx.int(-2), &ctx.int(0)),
        Some(true)
    );
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.int(-4), &ctx.int(0)),
        Some(false)
    );
}

#[test]
fn root_at_an_endpoint() {
    let (ctx, x, _y) = ctx_xy();
    let f = x.powi(2) - 4; // roots ±2
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.int(2), &ctx.int(10)),
        Some(true)
    );
    assert_eq!(
        f.poly_is_positive_on(&x, &ctx.int(2), &ctx.int(10)),
        Some(false)
    );
    assert_eq!(
        f.poly_is_positive_on(&x, &ctx.rational(201, 100), &ctx.int(10)),
        Some(true)
    );
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.int(-2), &ctx.int(2)),
        Some(false)
    );
}

#[test]
fn single_point_and_empty_intervals() {
    let (ctx, x, _y) = ctx_xy();
    let f = -x.powi(2);
    // At the single point 0 the value is 0.
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.int(0), &ctx.int(0)),
        Some(true)
    );
    assert_eq!(
        f.poly_is_positive_on(&x, &ctx.int(0), &ctx.int(0)),
        Some(false)
    );
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.int(1), &ctx.int(1)),
        Some(false)
    );
    // Empty interval: vacuously true.
    assert_eq!(
        f.poly_is_nonnegative_on(&x, &ctx.int(1), &ctx.int(0)),
        Some(true)
    );
    assert_eq!(
        f.poly_is_positive_on(&x, &ctx.infinity(), &ctx.int(0)),
        Some(true)
    );
}

#[test]
fn constants_and_zero_polynomial() {
    let (ctx, x, _y) = ctx_xy();
    let (ninf, inf) = (ctx.neg_infinity(), ctx.infinity());
    assert_eq!(
        ctx.int(0).poly_is_nonnegative_on(&x, &ninf, &inf),
        Some(true)
    );
    assert_eq!(ctx.int(0).poly_is_positive_on(&x, &ninf, &inf), Some(false));
    assert_eq!(ctx.int(5).poly_is_positive_on(&x, &ninf, &inf), Some(true));
    assert_eq!(
        ctx.rational(-1, 3).poly_is_nonnegative_on(&x, &ninf, &inf),
        Some(false)
    );
}

#[test]
fn sign_tests_none_for_non_polynomial_symbolic_or_bad_endpoints() {
    let (ctx, x, y) = ctx_xy();
    let (ninf, inf) = (ctx.neg_infinity(), ctx.infinity());
    assert_eq!(x.sin().poly_is_nonnegative_on(&x, &ninf, &inf), None);
    assert_eq!(
        (&y * x.powi(2)).poly_is_nonnegative_on(&x, &ninf, &inf),
        None
    );
    assert_eq!(x.powi(-2).poly_is_positive_on(&x, &ctx.int(1), &inf), None);
    assert_eq!((x.powi(2) + 1).poly_is_positive_on(&x, &y, &inf), None);
    assert_eq!(
        (x.powi(2) + 1).poly_is_positive_on(&x, &ctx.int(0), &ctx.pi()),
        None
    );
}

#[test]
fn sign_tests_agree_with_dense_sampling() {
    let (ctx, x, _y) = ctx_xy();
    let polys = [
        (x.powi(4) - x.powi(2) * 5 + 4).expand(), // roots ±1, ±2
        (x.powi(3) - &x * 3 + 1).expand(),
        ((&x - 1).powi(2) * (&x - 3)).expand(),
        (x.powi(2) * 2 - &x + ctx.rational(1, 8)).expand(),
        (-x.powi(4) + 3).expand(),
    ];
    let bounds: [(i64, i64); 6] = [(-3, -2), (-2, 0), (0, 1), (1, 3), (-3, 3), (2, 5)];
    for f in &polys {
        for &(lo, hi) in &bounds {
            let nonneg = f
                .poly_is_nonnegative_on(&x, &ctx.int(lo), &ctx.int(hi))
                .unwrap();
            let pos = f
                .poly_is_positive_on(&x, &ctx.int(lo), &ctx.int(hi))
                .unwrap();
            // Sample 201 points exactly.
            let mut all_nonneg = true;
            let mut all_pos = true;
            for k in 0..=200i64 {
                let t = ctx.from_ratio(rat(lo * 200 + (hi - lo) * k, 200));
                let v = f.subs(&x, &t).eval().as_rational().unwrap();
                if v < rat(0, 1) {
                    all_nonneg = false;
                }
                if v <= rat(0, 1) {
                    all_pos = false;
                }
            }
            // Sampling can only over-approximate the exact answer.
            if !all_nonneg {
                assert!(
                    !nonneg,
                    "{f} on [{lo}, {hi}]: sampled negative but nonneg=true"
                );
            }
            if !all_pos {
                assert!(!pos, "{f} on [{lo}, {hi}]: sampled ≤ 0 but positive=true");
            }
            if nonneg {
                assert!(all_nonneg, "{f} on [{lo}, {hi}]");
            }
            if pos {
                assert!(all_pos, "{f} on [{lo}, {hi}]");
            }
        }
    }
}

#[test]
fn sign_tests_are_fast_on_a_degree_20_polynomial() {
    let (ctx, x, _y) = ctx_xy();
    let start = std::time::Instant::now();
    let f = (x.powi(2) + 1).powi(10) - &x * 3;
    let r = f.poly_is_positive_on(&x, &ctx.int(1), &ctx.infinity());
    assert_eq!(r, Some(true));
    assert!(
        start.elapsed() < time_budget(5),
        "took {:?}",
        start.elapsed()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Properties
// ═══════════════════════════════════════════════════════════════════════════

/// A random polynomial in x, y with small integer coefficients (as an
/// unexpanded expression so `new` has to do work).
fn poly_strategy() -> impl Strategy<Value = Vec<(u8, u8, i8)>> {
    prop::collection::vec((0u8..=3, 0u8..=3, -5i8..=5), 1..=5)
}

fn build(ctx: &Context, x: &Ex, y: &Ex, terms: &[(u8, u8, i8)]) -> Ex {
    let mut e = ctx.zero();
    for &(ex, ey, c) in terms {
        e += x.powi(i64::from(ex)) * y.powi(i64::from(ey)) * i64::from(c);
    }
    e
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn prop_ring_identities(a in poly_strategy(), b in poly_strategy(), c in poly_strategy()) {
        let (ctx, x, y) = ctx_xy();
        let gens = [&x, &y];
        let p = Poly::new(&build(&ctx, &x, &y, &a), &gens).unwrap();
        let q = Poly::new(&build(&ctx, &x, &y, &b), &gens).unwrap();
        let r = Poly::new(&build(&ctx, &x, &y, &c), &gens).unwrap();
        // commutativity
        prop_assert!(p.add(&q).unwrap().equals(&q.add(&p).unwrap()));
        prop_assert!(p.mul(&q).unwrap().equals(&q.mul(&p).unwrap()));
        // (p + q) − q = p
        prop_assert!(p.add(&q).unwrap().sub(&q).unwrap().equals(&p));
        // distributivity
        let lhs = p.mul(&q.add(&r).unwrap()).unwrap();
        let rhs = p.mul(&q).unwrap().add(&p.mul(&r).unwrap()).unwrap();
        prop_assert!(lhs.equals(&rhs));
        // p − p = 0, p + (−p) = 0
        prop_assert!(p.sub(&p).unwrap().is_zero());
        prop_assert!(p.add(&p.neg()).unwrap().is_zero());
    }

    #[test]
    fn prop_to_ex_round_trip_and_expand_agreement(a in poly_strategy(), b in poly_strategy()) {
        let (ctx, x, y) = ctx_xy();
        let gens = [&x, &y];
        let ea = build(&ctx, &x, &y, &a);
        let eb = build(&ctx, &x, &y, &b);
        let p = Poly::new(&ea, &gens).unwrap();
        let q = Poly::new(&eb, &gens).unwrap();
        let prod = p.mul(&q).unwrap();
        prop_assert_eq!(prod.to_ex(), (&ea * &eb).expand());
        let back = Poly::new(&prod.to_ex(), &gens).unwrap();
        prop_assert!(back.equals(&prod));
        // Evaluation agrees with substitution.
        let (vx, vy) = (ctx.rational(3, 2), ctx.rational(-2, 7));
        let via_poly = prod.eval(&[&vx, &vy]).unwrap();
        let via_subs = (&ea * &eb).subs_map(&[(&x, &vx), (&y, &vy)]).eval();
        prop_assert_eq!(via_poly, via_subs);
    }

    #[test]
    fn prop_derivative_matches_diff(a in poly_strategy()) {
        let (ctx, x, y) = ctx_xy();
        let e = build(&ctx, &x, &y, &a);
        let p = Poly::new(&e, &[&x, &y]).unwrap();
        prop_assert_eq!(p.derivative(&x).unwrap().to_ex(), e.diff(&x).expand());
        prop_assert_eq!(p.derivative(&y).unwrap().to_ex(), e.diff(&y).expand());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Property: interval sign tests vs. an independent root-based oracle
// ═══════════════════════════════════════════════════════════════════════════

/// A polynomial given by its rational roots with multiplicities, a leading
/// sign, and an optional irreducible quadratic factor (no real roots).
#[derive(Clone, Debug)]
struct RootSpec {
    roots: Vec<(i64, i64, u32)>, // (numerator, denominator, multiplicity)
    negative: bool,
    quadratic: bool, // multiply by (x^2 + 1)
}

fn root_spec() -> impl Strategy<Value = RootSpec> {
    (
        prop::collection::vec((-6i64..=6, 1i64..=4, 1u32..=3), 0..=3),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(roots, negative, quadratic)| RootSpec {
            roots,
            negative,
            quadratic,
        })
}

/// An interval endpoint: a small rational, or infinite.
#[derive(Clone, Copy, Debug)]
enum End {
    Fin(i64, i64),
    Inf,
}

fn endpoint() -> impl Strategy<Value = End> {
    prop_oneof![
        4 => (-8i64..=8, 1i64..=4).prop_map(|(n, d)| End::Fin(n, d)),
        1 => Just(End::Inf),
    ]
}

/// Sign of `f` at a rational point from the root list (independent of the
/// library): sign(c) · ∏ sign(x − rᵢ)^{mᵢ}.
fn oracle_sign(spec: &RootSpec, at: &Ratio<BigInt>) -> i32 {
    let mut s: i32 = if spec.negative { -1 } else { 1 };
    for &(n, d, m) in &spec.roots {
        let r = rat(n, d);
        let diff = at - &r;
        if diff == Ratio::from(BigInt::from(0)) {
            return 0;
        }
        if diff < Ratio::from(BigInt::from(0)) && m % 2 == 1 {
            s = -s;
        }
    }
    s
}

/// Independent decision: `f ≥ 0` (resp. `> 0`) on `[lo, hi]`.
fn oracle_nonneg(
    spec: &RootSpec,
    lo: Option<Ratio<BigInt>>,
    hi: Option<Ratio<BigInt>>,
    strict: bool,
) -> bool {
    let inside_open =
        |r: &Ratio<BigInt>| lo.as_ref().is_none_or(|l| r > l) && hi.as_ref().is_none_or(|h| r < h);
    let inside_closed = |r: &Ratio<BigInt>| {
        lo.as_ref().is_none_or(|l| r >= l) && hi.as_ref().is_none_or(|h| r <= h)
    };
    // A root of odd multiplicity strictly inside flips the sign.
    for &(n, d, m) in &spec.roots {
        let r = rat(n, d);
        if m % 2 == 1 && inside_open(&r) {
            return false;
        }
        if strict && inside_closed(&r) {
            return false;
        }
    }
    // Otherwise the sign is constant on the interval away from roots:
    // sample a point in the interior that is not a root.
    let roots: Vec<Ratio<BigInt>> = spec.roots.iter().map(|&(n, d, _)| rat(n, d)).collect();
    let mut sample = match (&lo, &hi) {
        (Some(l), Some(h)) => (l + h) / Ratio::from(BigInt::from(2)),
        (Some(l), None) => l + Ratio::from(BigInt::from(1)),
        (None, Some(h)) => h - Ratio::from(BigInt::from(1)),
        (None, None) => Ratio::from(BigInt::from(0)),
    };
    // Nudge off any root (stay inside the interval).
    let mut step = rat(1, 1000);
    while roots.contains(&sample) {
        sample += &step;
        step /= Ratio::from(BigInt::from(2));
    }
    if let (Some(l), Some(h)) = (&lo, &hi)
        && l == h
    {
        // Degenerate interval: the value at the single point decides.
        let s = oracle_sign(spec, l);
        return if strict { s > 0 } else { s >= 0 };
    }
    oracle_sign(spec, &sample) > 0
}

fn build_from_roots(ctx: &Context, x: &Ex, spec: &RootSpec) -> Ex {
    let mut f = if spec.negative {
        ctx.int(-1)
    } else {
        ctx.int(1)
    };
    for &(n, d, m) in &spec.roots {
        f *= (x - ctx.rational(n, d)).powi(i64::from(m));
    }
    if spec.quadratic {
        f *= x.powi(2) + 1;
    }
    f.expand()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn prop_interval_sign_matches_root_oracle(spec in root_spec(), a in endpoint(), b in endpoint()) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let f = build_from_roots(&ctx, &x, &spec);
        // Order the finite endpoints; `Inf` on the left means −∞.
        let (lo, hi) = match (a, b) {
            (End::Fin(n1, d1), End::Fin(n2, d2)) => {
                let (r1, r2) = (rat(n1, d1), rat(n2, d2));
                if r1 <= r2 { (Some(r1), Some(r2)) } else { (Some(r2), Some(r1)) }
            }
            (End::Fin(n, d), End::Inf) => (Some(rat(n, d)), None),
            (End::Inf, End::Fin(n, d)) => (None, Some(rat(n, d))),
            (End::Inf, End::Inf) => (None, None),
        };
        let lo_ex = lo.clone().map_or_else(|| ctx.neg_infinity(), |r| ctx.from_ratio(r));
        let hi_ex = hi.clone().map_or_else(|| ctx.infinity(), |r| ctx.from_ratio(r));

        let got_nonneg = f.poly_is_nonnegative_on(&x, &lo_ex, &hi_ex);
        let got_pos = f.poly_is_positive_on(&x, &lo_ex, &hi_ex);
        let want_nonneg = oracle_nonneg(&spec, lo.clone(), hi.clone(), false);
        let want_pos = oracle_nonneg(&spec, lo, hi, true);
        prop_assert_eq!(got_nonneg, Some(want_nonneg), "f = {} nonneg on [{}, {}]", f, lo_ex, hi_ex);
        prop_assert_eq!(got_pos, Some(want_pos), "f = {} positive on [{}, {}]", f, lo_ex, hi_ex);
    }
}
