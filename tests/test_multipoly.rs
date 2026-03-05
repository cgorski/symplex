//! Integration tests for the `multipoly` module — sparse multivariate
//! polynomials over ℚ.

use num_bigint::BigInt;
use num_rational::Ratio;
use symplex::multipoly::MultiPoly;

// ── Helper ─────────────────────────────────────────────────────────────────

fn rat(n: i64) -> Ratio<BigInt> {
    Ratio::from_integer(BigInt::from(n))
}

fn rat_frac(p: i64, q: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(p), BigInt::from(q))
}

// ── 1. Zero polynomial ────────────────────────────────────────────────────

#[test]
fn zero_polynomial() {
    let z = MultiPoly::zero(3);
    assert!(z.is_zero());
    assert_eq!(z.num_vars(), 3);
    assert_eq!(z.num_terms(), 0);
    assert_eq!(z.total_degree(), None);
    assert_eq!(format!("{z}"), "0");
}

// ── 2. Constant polynomial ────────────────────────────────────────────────

#[test]
fn constant_polynomial() {
    let c = MultiPoly::from_int(2, 5);
    assert!(!c.is_zero());
    assert_eq!(c.num_terms(), 1);
    assert_eq!(c.total_degree(), Some(0));
    assert_eq!(c.eval(&[rat(99), rat(99)]), rat(5));
}

// ── 3. Variable polynomial ────────────────────────────────────────────────

#[test]
fn variable_polynomial() {
    let x = MultiPoly::var(3, 0);
    assert!(!x.is_zero());
    assert_eq!(x.num_terms(), 1);
    assert_eq!(x.total_degree(), Some(1));
    assert_eq!(x.degree_in(0), 1);
    assert_eq!(x.degree_in(1), 0);
    assert_eq!(x.degree_in(2), 0);
    // x evaluated at (7, ?, ?) = 7
    assert_eq!(x.eval(&[rat(7), rat(0), rat(0)]), rat(7));
}

// ── 4. Add polynomials ────────────────────────────────────────────────────

#[test]
fn add_polynomials() {
    let x = MultiPoly::var(2, 0);
    let y = MultiPoly::var(2, 1);
    let sum = &x + &y;
    assert_eq!(sum.num_terms(), 2);
    assert_eq!(sum.total_degree(), Some(1));
    // (x + y) at (3, 4) = 7
    assert_eq!(sum.eval(&[rat(3), rat(4)]), rat(7));
}

// ── 5. Add like terms ─────────────────────────────────────────────────────

#[test]
fn add_like_terms() {
    let x = MultiPoly::var(2, 0);
    let two_x = &x + &x;
    // Should combine into a single term 2x
    assert_eq!(two_x.num_terms(), 1);
    assert_eq!(two_x.eval(&[rat(5), rat(0)]), rat(10));

    // x + 2x = 3x
    let three_x = &x + &two_x;
    assert_eq!(three_x.num_terms(), 1);
    assert_eq!(three_x.eval(&[rat(1), rat(0)]), rat(3));
}

// ── 6. Subtract to zero ──────────────────────────────────────────────────

#[test]
fn subtract_to_zero() {
    let x = MultiPoly::var(3, 1);
    let diff = &x - &x;
    assert!(diff.is_zero());
    assert_eq!(diff.num_terms(), 0);
    assert_eq!(format!("{diff}"), "0");
}

// ── 7. Multiply monomials ─────────────────────────────────────────────────

#[test]
fn multiply_monomials() {
    let x = MultiPoly::var(2, 0);
    let y = MultiPoly::var(2, 1);
    let xy = &x * &y;
    assert_eq!(xy.num_terms(), 1);
    assert_eq!(xy.total_degree(), Some(2));
    assert_eq!(xy.degree_in(0), 1);
    assert_eq!(xy.degree_in(1), 1);
    // xy at (3, 5) = 15
    assert_eq!(xy.eval(&[rat(3), rat(5)]), rat(15));
}

// ── 8. Multiply polynomials: (x+1)(x-1) = x²-1 ──────────────────────────

#[test]
fn multiply_polynomials() {
    let x = MultiPoly::var(1, 0);
    let one = MultiPoly::from_int(1, 1);
    let x_plus_1 = &x + &one;
    let x_minus_1 = &x - &one;
    let product = &x_plus_1 * &x_minus_1;
    // x² - 1
    assert_eq!(product.num_terms(), 2);
    assert_eq!(product.total_degree(), Some(2));
    // At x=3: 9 - 1 = 8
    assert_eq!(product.eval(&[rat(3)]), rat(8));
    // At x=1: 1 - 1 = 0
    assert_eq!(product.eval(&[rat(1)]), rat(0));
}

// ── 9. Multiply multivariate: (x+y)² = x² + 2xy + y² ────────────────────

#[test]
fn multiply_multivariate() {
    let x = MultiPoly::var(2, 0);
    let y = MultiPoly::var(2, 1);
    let x_plus_y = &x + &y;
    let squared = &x_plus_y * &x_plus_y;
    // x² + 2xy + y² → 3 terms
    assert_eq!(squared.num_terms(), 3);
    assert_eq!(squared.total_degree(), Some(2));
    // At (2, 3): 4 + 12 + 9 = 25
    assert_eq!(squared.eval(&[rat(2), rat(3)]), rat(25));
}

// ── 10. Total degree ──────────────────────────────────────────────────────

#[test]
fn total_degree() {
    // x²y³ has total degree 5
    let mono = MultiPoly::monomial(rat(1), vec![2, 3]);
    assert_eq!(mono.total_degree(), Some(5));
}

// ── 11. Degree in variable ────────────────────────────────────────────────

#[test]
fn degree_in_variable() {
    // x²y³
    let mono = MultiPoly::monomial(rat(1), vec![2, 3]);
    assert_eq!(mono.degree_in(0), 2);
    assert_eq!(mono.degree_in(1), 3);
}

// ── 12. Evaluate ──────────────────────────────────────────────────────────

#[test]
fn evaluate() {
    // p(x, y) = x² + y
    let x = MultiPoly::var(2, 0);
    let y = MultiPoly::var(2, 1);
    let x_sq = &x * &x;
    let p = &x_sq + &y;
    // p(3, 7) = 9 + 7 = 16
    assert_eq!(p.eval(&[rat(3), rat(7)]), rat(16));
}

// ── 13. Partial derivative ∂/∂x(x²y) = 2xy ──────────────────────────────

#[test]
fn partial_derivative_x() {
    // x²y = monomial with coeff 1, exponents [2, 1]
    let p = MultiPoly::monomial(rat(1), vec![2, 1]);
    let dp_dx = p.partial_derivative(0);
    // Should be 2xy: monomial with coeff 2, exponents [1, 1]
    assert_eq!(dp_dx.num_terms(), 1);
    assert_eq!(dp_dx.eval(&[rat(3), rat(5)]), rat(30)); // 2*3*5 = 30
}

// ── 14. Partial derivative ∂/∂y(x²y) = x² ───────────────────────────────

#[test]
fn partial_derivative_y() {
    let p = MultiPoly::monomial(rat(1), vec![2, 1]);
    let dp_dy = p.partial_derivative(1);
    // Should be x²: monomial with coeff 1, exponents [2, 0]
    assert_eq!(dp_dy.num_terms(), 1);
    assert_eq!(dp_dy.eval(&[rat(4), rat(999)]), rat(16)); // 4² = 16
}

// ── 15. Scale ─────────────────────────────────────────────────────────────

#[test]
fn scale() {
    let x = MultiPoly::var(2, 0);
    let y = MultiPoly::var(2, 1);
    let p = &x + &y;
    let scaled = p.scale(&rat(3));
    // 3(x + y) at (2, 5) = 21
    assert_eq!(scaled.eval(&[rat(2), rat(5)]), rat(21));
    assert_eq!(scaled.num_terms(), 2);
}

// ── 16. Leading term grevlex ──────────────────────────────────────────────

#[test]
fn leading_term_grevlex() {
    // p = x² + xy + y²
    let x = MultiPoly::var(2, 0);
    let y = MultiPoly::var(2, 1);
    let x2 = &x * &x;
    let xy = &x * &y;
    let y2 = &y * &y;
    let p = &(&x2 + &xy) + &y2;

    let (lt_exp, lt_coeff) = p.leading_term_grevlex().unwrap();
    // In grevlex, all three have total degree 2.
    // [2,0] vs [1,1] vs [0,2]:
    //   Rightmost differing for [2,0] vs [1,1]: idx 1 → 0 < 1 → [2,0] > [1,1]
    //   Rightmost differing for [2,0] vs [0,2]: idx 1 → 0 < 2 → [2,0] > [0,2]
    // So leading term is x² = [2, 0]
    assert_eq!(lt_exp, &vec![2u32, 0]);
    assert_eq!(*lt_coeff, rat(1));
}

// ── 17. Display ───────────────────────────────────────────────────────────

#[test]
fn display() {
    let z = MultiPoly::zero(2);
    assert_eq!(format!("{z}"), "0");

    let x = MultiPoly::var(2, 0);
    let s = format!("{x}");
    assert_eq!(s, "x0");

    let c = MultiPoly::from_int(2, -3);
    assert_eq!(format!("{c}"), "-3");

    // x + 1
    let one = MultiPoly::from_int(2, 1);
    let p = &x + &one;
    let s = format!("{p}");
    assert!(s.contains("x0"));
    assert!(s.contains("1"));
}

// ── 18. Num terms ─────────────────────────────────────────────────────────

#[test]
fn num_terms() {
    // x² + xy + y² has 3 terms
    let x = MultiPoly::var(2, 0);
    let y = MultiPoly::var(2, 1);
    let p = &(&(&x * &x) + &(&x * &y)) + &(&y * &y);
    assert_eq!(p.num_terms(), 3);
}

// ── 19. Negate polynomial ─────────────────────────────────────────────────

#[test]
fn neg_polynomial() {
    let x = MultiPoly::var(2, 0);
    let one = MultiPoly::from_int(2, 1);
    let p = &x + &one; // x + 1
    let neg_p = -&p; // -x - 1
    assert_eq!(neg_p.num_terms(), 2);
    // (-x - 1) at (3, 0) = -4
    assert_eq!(neg_p.eval(&[rat(3), rat(0)]), rat(-4));

    // p + (-p) = 0
    let should_be_zero = &p + &neg_p;
    assert!(should_be_zero.is_zero());
}

// ── 20. Constant times poly: 0 * p = 0 ───────────────────────────────────

#[test]
fn constant_times_poly() {
    let x = MultiPoly::var(2, 0);
    let y = MultiPoly::var(2, 1);
    let p = &x + &y;
    let zero_scaled = p.scale(&rat(0));
    assert!(zero_scaled.is_zero());
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn monomial_constructor() {
    let m = MultiPoly::monomial(rat(7), vec![3, 0, 2]);
    assert_eq!(m.num_vars(), 3);
    assert_eq!(m.num_terms(), 1);
    assert_eq!(m.total_degree(), Some(5));
    // 7 * 2^3 * 1^0 * 3^2 = 7 * 8 * 9 = 504
    assert_eq!(m.eval(&[rat(2), rat(1), rat(3)]), rat(504));
}

#[test]
fn monomial_zero_coeff() {
    let m = MultiPoly::monomial(rat(0), vec![1, 2]);
    assert!(m.is_zero());
}

#[test]
fn substitute_variable() {
    // p(x, y) = x² + 3xy + y²
    let x = MultiPoly::var(2, 0);
    let y = MultiPoly::var(2, 1);
    let three = MultiPoly::from_int(2, 3);
    let p = &(&(&x * &x) + &(&three * &(&x * &y))) + &(&y * &y);

    // Substitute x = 2 → p(2, y) = 4 + 6y + y² (1-variable polynomial)
    let q = p.substitute(0, &rat(2));
    assert_eq!(q.num_vars(), 1);
    // Evaluate q at y=3: 4 + 18 + 9 = 31
    assert_eq!(q.eval(&[rat(3)]), rat(31));
    // Cross-check: p(2, 3) = 4 + 18 + 9 = 31
    assert_eq!(p.eval(&[rat(2), rat(3)]), rat(31));
}

#[test]
fn partial_derivative_constant() {
    let c = MultiPoly::from_int(2, 42);
    let dc = c.partial_derivative(0);
    assert!(dc.is_zero());
}

#[test]
fn partial_derivative_higher_degree() {
    // p(x) = x^3 in 1 variable
    let x = MultiPoly::var(1, 0);
    let x3 = &(&x * &x) * &x;
    let dp = x3.partial_derivative(0); // 3x²
    assert_eq!(dp.num_terms(), 1);
    assert_eq!(dp.eval(&[rat(2)]), rat(12)); // 3*4 = 12
    let d2p = dp.partial_derivative(0); // 6x
    assert_eq!(d2p.eval(&[rat(5)]), rat(30)); // 6*5 = 30
}

#[test]
fn eval_with_rationals() {
    // p(x, y) = x + y, evaluate at (1/2, 1/3) = 5/6
    let x = MultiPoly::var(2, 0);
    let y = MultiPoly::var(2, 1);
    let p = &x + &y;
    let result = p.eval(&[rat_frac(1, 2), rat_frac(1, 3)]);
    assert_eq!(result, rat_frac(5, 6));
}

#[test]
fn leading_coeff_grevlex() {
    let p = MultiPoly::monomial(rat(7), vec![2, 3]);
    assert_eq!(*p.leading_coeff_grevlex().unwrap(), rat(7));
}

#[test]
fn leading_term_zero_poly() {
    let z = MultiPoly::zero(2);
    assert!(z.leading_term_grevlex().is_none());
    assert!(z.leading_coeff_grevlex().is_none());
}

#[test]
fn operator_overloads_owned() {
    // Test that owned-value operators work
    let x = MultiPoly::var(2, 0);
    let y = MultiPoly::var(2, 1);
    let sum = x.clone() + y.clone();
    assert_eq!(sum.num_terms(), 2);
    let diff = x.clone() - y.clone();
    assert_eq!(diff.num_terms(), 2);
    let prod = x.clone() * y.clone();
    assert_eq!(prod.num_terms(), 1);
    let neg = -x.clone();
    assert_eq!(neg.eval(&[rat(5), rat(0)]), rat(-5));
}

#[test]
fn distributive_law() {
    // a * (b + c) == a*b + a*c
    let x = MultiPoly::var(2, 0);
    let y = MultiPoly::var(2, 1);
    let one = MultiPoly::from_int(2, 1);

    let a = &x + &one; // x + 1
    let b = &y;
    let c = &x;

    let lhs = &a * &(b + c); // (x+1)(y+x)
    let rhs = &(&a * b) + &(&a * c); // (x+1)*y + (x+1)*x

    // Evaluate at a common point to check equality
    let pt = [rat(3), rat(7)];
    assert_eq!(lhs.eval(&pt), rhs.eval(&pt));
    // Also check (2, 5)
    let pt2 = [rat(2), rat(5)];
    assert_eq!(lhs.eval(&pt2), rhs.eval(&pt2));
}

#[test]
fn degree_in_zero_poly() {
    let z = MultiPoly::zero(3);
    assert_eq!(z.degree_in(0), 0);
    assert_eq!(z.degree_in(1), 0);
    assert_eq!(z.degree_in(2), 0);
}

#[test]
fn from_int_zero() {
    let z = MultiPoly::from_int(2, 0);
    assert!(z.is_zero());
}

#[test]
fn scale_by_fraction() {
    let x = MultiPoly::var(1, 0);
    let half_x = x.scale(&rat_frac(1, 2));
    assert_eq!(half_x.eval(&[rat(6)]), rat(3)); // (1/2)*6 = 3
}

#[test]
fn display_negative_leading() {
    // -x should display as "-x0"
    let x = MultiPoly::var(1, 0);
    let neg_x = -&x;
    let s = format!("{neg_x}");
    assert_eq!(s, "-x0");
}

#[test]
fn display_multiterm() {
    // x² - 1 in one variable
    let x = MultiPoly::var(1, 0);
    let one = MultiPoly::from_int(1, 1);
    let p = &(&x * &x) - &one;
    let s = format!("{p}");
    // Should be "x0^2 - 1"
    assert_eq!(s, "x0^2 - 1");
}

#[test]
fn multiply_by_zero() {
    let x = MultiPoly::var(2, 0);
    let z = MultiPoly::zero(2);
    let product = &x * &z;
    assert!(product.is_zero());
}

#[test]
fn add_zero_identity() {
    let x = MultiPoly::var(2, 0);
    let z = MultiPoly::zero(2);
    let sum = &x + &z;
    assert_eq!(sum.eval(&[rat(7), rat(0)]), rat(7));
    assert_eq!(sum.num_terms(), 1);
}

#[test]
#[should_panic(expected = "incompatible variable counts")]
fn incompatible_add_panics() {
    let a = MultiPoly::var(2, 0);
    let b = MultiPoly::var(3, 0);
    let _ = &a + &b;
}

#[test]
#[should_panic(expected = "var_index")]
fn var_index_out_of_range_panics() {
    let _ = MultiPoly::var(2, 5);
}

#[test]
fn three_variable_polynomial() {
    // p(x, y, z) = xyz + x + y + z + 1
    let x = MultiPoly::var(3, 0);
    let y = MultiPoly::var(3, 1);
    let z = MultiPoly::var(3, 2);
    let one = MultiPoly::from_int(3, 1);
    let xyz = &(&x * &y) * &z;
    let p = &(&(&(&xyz + &x) + &y) + &z) + &one;
    assert_eq!(p.num_terms(), 5);
    // p(2, 3, 5) = 30 + 2 + 3 + 5 + 1 = 41
    assert_eq!(p.eval(&[rat(2), rat(3), rat(5)]), rat(41));
}
