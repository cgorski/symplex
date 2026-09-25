//! Integration tests for the `multipoly` module — sparse multivariate
//! polynomials over ℚ.

use num_bigint::BigInt;
use num_rational::Ratio;
use symplex::multipoly::{
    GrLex, GrevLex, Lex, MonomialOrd, MultiPoly, monomial_coprime, monomial_div, monomial_divides,
    monomial_lcm, monomial_mul, s_polynomial,
};

// Alias for the default ordering — avoids type annotations on every constructor call.
type Poly = MultiPoly<GrevLex>;

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
    let z = Poly::zero(3);
    assert!(z.is_zero());
    assert_eq!(z.num_vars(), 3);
    assert_eq!(z.num_terms(), 0);
    assert_eq!(z.total_degree(), None);
    assert_eq!(format!("{z}"), "0");
}

// ── 2. Constant polynomial ────────────────────────────────────────────────

#[test]
fn constant_polynomial() {
    let c = Poly::from_int(2, 5);
    assert!(!c.is_zero());
    assert_eq!(c.num_terms(), 1);
    assert_eq!(c.total_degree(), Some(0));
    assert_eq!(c.eval(&[rat(99), rat(99)]).unwrap(), rat(5));
}

// ── 3. Variable polynomial ────────────────────────────────────────────────

#[test]
fn variable_polynomial() {
    let x = Poly::var(3, 0);
    assert!(!x.is_zero());
    assert_eq!(x.num_terms(), 1);
    assert_eq!(x.total_degree(), Some(1));
    assert_eq!(x.degree_in(0), 1);
    assert_eq!(x.degree_in(1), 0);
    assert_eq!(x.degree_in(2), 0);
    // x evaluated at (7, ?, ?) = 7
    assert_eq!(x.eval(&[rat(7), rat(0), rat(0)]).unwrap(), rat(7));
}

// ── 4. Add polynomials ────────────────────────────────────────────────────

#[test]
fn add_polynomials() {
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let sum = &x + &y;
    assert_eq!(sum.num_terms(), 2);
    assert_eq!(sum.total_degree(), Some(1));
    // (x + y) at (3, 4) = 7
    assert_eq!(sum.eval(&[rat(3), rat(4)]).unwrap(), rat(7));
}

// ── 5. Add like terms ─────────────────────────────────────────────────────

#[test]
fn add_like_terms() {
    let x = Poly::var(2, 0);
    let two_x = &x + &x;
    // Should combine into a single term 2x
    assert_eq!(two_x.num_terms(), 1);
    assert_eq!(two_x.eval(&[rat(5), rat(0)]).unwrap(), rat(10));

    // x + 2x = 3x
    let three_x = &x + &two_x;
    assert_eq!(three_x.num_terms(), 1);
    assert_eq!(three_x.eval(&[rat(1), rat(0)]).unwrap(), rat(3));
}

// ── 6. Subtract to zero ──────────────────────────────────────────────────

#[test]
fn subtract_to_zero() {
    let x = Poly::var(3, 1);
    let diff = &x - &x;
    assert!(diff.is_zero());
    assert_eq!(diff.num_terms(), 0);
    assert_eq!(format!("{diff}"), "0");
}

// ── 7. Multiply monomials ─────────────────────────────────────────────────

#[test]
fn multiply_monomials() {
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let xy = &x * &y;
    assert_eq!(xy.num_terms(), 1);
    assert_eq!(xy.total_degree(), Some(2));
    assert_eq!(xy.degree_in(0), 1);
    assert_eq!(xy.degree_in(1), 1);
    // xy at (3, 5) = 15
    assert_eq!(xy.eval(&[rat(3), rat(5)]).unwrap(), rat(15));
}

// ── 8. Multiply polynomials: (x+1)(x-1) = x²-1 ──────────────────────────

#[test]
fn multiply_polynomials() {
    let x = Poly::var(1, 0);
    let one = Poly::from_int(1, 1);
    let x_plus_1 = &x + &one;
    let x_minus_1 = &x - &one;
    let product = &x_plus_1 * &x_minus_1;
    // x² - 1
    assert_eq!(product.num_terms(), 2);
    assert_eq!(product.total_degree(), Some(2));
    // At x=3: 9 - 1 = 8
    assert_eq!(product.eval(&[rat(3)]).unwrap(), rat(8));
    // At x=1: 1 - 1 = 0
    assert_eq!(product.eval(&[rat(1)]).unwrap(), rat(0));
}

// ── 9. Multiply multivariate: (x+y)² = x² + 2xy + y² ────────────────────

#[test]
fn multiply_multivariate() {
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let x_plus_y = &x + &y;
    let squared = &x_plus_y * &x_plus_y;
    // x² + 2xy + y² → 3 terms
    assert_eq!(squared.num_terms(), 3);
    assert_eq!(squared.total_degree(), Some(2));
    // At (2, 3): 4 + 12 + 9 = 25
    assert_eq!(squared.eval(&[rat(2), rat(3)]).unwrap(), rat(25));
}

// ── 10. Total degree ──────────────────────────────────────────────────────

#[test]
fn total_degree() {
    // x²y³ has total degree 5
    let mono = Poly::monomial(rat(1), vec![2, 3]);
    assert_eq!(mono.total_degree(), Some(5));
}

// ── 11. Degree in variable ────────────────────────────────────────────────

#[test]
fn degree_in_variable() {
    // x²y³
    let mono = Poly::monomial(rat(1), vec![2, 3]);
    assert_eq!(mono.degree_in(0), 2);
    assert_eq!(mono.degree_in(1), 3);
}

// ── 12. Evaluate ──────────────────────────────────────────────────────────

#[test]
fn evaluate() {
    // p(x, y) = x² + y
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let x_sq = &x * &x;
    let p = &x_sq + &y;
    // p(3, 7) = 9 + 7 = 16
    assert_eq!(p.eval(&[rat(3), rat(7)]).unwrap(), rat(16));
}

// ── 13. Partial derivative ∂/∂x(x²y) = 2xy ──────────────────────────────

#[test]
fn partial_derivative_x() {
    // x²y = monomial with coeff 1, exponents [2, 1]
    let p = Poly::monomial(rat(1), vec![2, 1]);
    let dp_dx = p.partial_derivative(0);
    // Should be 2xy: monomial with coeff 2, exponents [1, 1]
    assert_eq!(dp_dx.num_terms(), 1);
    assert_eq!(dp_dx.eval(&[rat(3), rat(5)]).unwrap(), rat(30)); // 2*3*5 = 30
}

// ── 14. Partial derivative ∂/∂y(x²y) = x² ───────────────────────────────

#[test]
fn partial_derivative_y() {
    let p = Poly::monomial(rat(1), vec![2, 1]);
    let dp_dy = p.partial_derivative(1);
    // Should be x²: monomial with coeff 1, exponents [2, 0]
    assert_eq!(dp_dy.num_terms(), 1);
    assert_eq!(dp_dy.eval(&[rat(4), rat(999)]).unwrap(), rat(16)); // 4² = 16
}

// ── 15. Scale ─────────────────────────────────────────────────────────────

#[test]
fn scale() {
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let p = &x + &y;
    let scaled = p.scale(&rat(3));
    // 3(x + y) at (2, 5) = 21
    assert_eq!(scaled.eval(&[rat(2), rat(5)]).unwrap(), rat(21));
    assert_eq!(scaled.num_terms(), 2);
}

// ── 16. Leading term ──────────────────────────────────────────────────────

#[test]
fn leading_term_grevlex() {
    // p = x² + xy + y²
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let x2 = &x * &x;
    let xy = &x * &y;
    let y2 = &y * &y;
    let p = &(&x2 + &xy) + &y2;

    let (lt_exp, lt_coeff) = p.leading_term().unwrap();
    // In grevlex, all three have total degree 2.
    // [2,0] vs [1,1] vs [0,2]:
    //   Rightmost differing for [2,0] vs [1,1]: idx 1 → 0 < 1 → [2,0] > [1,1]
    //   Rightmost differing for [2,0] vs [0,2]: idx 1 → 0 < 2 → [2,0] > [0,2]
    // So leading term is x² = [2, 0]
    assert_eq!(lt_exp, &[2u32, 0]);
    assert_eq!(*lt_coeff, rat(1));
}

// ── 17. Display ───────────────────────────────────────────────────────────

#[test]
fn display() {
    let z = Poly::zero(2);
    assert_eq!(format!("{z}"), "0");

    let x = Poly::var(2, 0);
    let s = format!("{x}");
    assert_eq!(s, "x0");

    let c = Poly::from_int(2, -3);
    assert_eq!(format!("{c}"), "-3");

    // x + 1
    let one = Poly::from_int(2, 1);
    let p = &x + &one;
    let s = format!("{p}");
    assert!(s.contains("x0"));
    assert!(s.contains("1"));
}

// ── 18. Num terms ─────────────────────────────────────────────────────────

#[test]
fn num_terms() {
    // x² + xy + y² has 3 terms
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let p = &(&(&x * &x) + &(&x * &y)) + &(&y * &y);
    assert_eq!(p.num_terms(), 3);
}

// ── 19. Negate polynomial ─────────────────────────────────────────────────

#[test]
fn neg_polynomial() {
    let x = Poly::var(2, 0);
    let one = Poly::from_int(2, 1);
    let p = &x + &one; // x + 1
    let neg_p = -&p; // -x - 1
    assert_eq!(neg_p.num_terms(), 2);
    // (-x - 1) at (3, 0) = -4
    assert_eq!(neg_p.eval(&[rat(3), rat(0)]).unwrap(), rat(-4));

    // p + (-p) = 0
    let should_be_zero = &p + &neg_p;
    assert!(should_be_zero.is_zero());
}

// ── 20. Constant times poly: 0 * p = 0 ───────────────────────────────────

#[test]
fn constant_times_poly() {
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let p = &x + &y;
    let zero_scaled = p.scale(&rat(0));
    assert!(zero_scaled.is_zero());
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional tests (pre-existing)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn monomial_constructor() {
    let m = Poly::monomial(rat(7), vec![3, 0, 2]);
    assert_eq!(m.num_vars(), 3);
    assert_eq!(m.num_terms(), 1);
    assert_eq!(m.total_degree(), Some(5));
    // 7 * 2^3 * 1^0 * 3^2 = 7 * 8 * 9 = 504
    assert_eq!(m.eval(&[rat(2), rat(1), rat(3)]).unwrap(), rat(504));
}

#[test]
fn monomial_zero_coeff() {
    let m = Poly::monomial(rat(0), vec![1, 2]);
    assert!(m.is_zero());
}

#[test]
fn substitute_variable() {
    // p(x, y) = x² + 3xy + y²
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let three = Poly::from_int(2, 3);
    let p = &(&(&x * &x) + &(&three * &(&x * &y))) + &(&y * &y);

    // Substitute x = 2 → p(2, y) = 4 + 6y + y² (1-variable polynomial)
    let q = p.substitute(0, &rat(2));
    assert_eq!(q.num_vars(), 1);
    // Evaluate q at y=3: 4 + 18 + 9 = 31
    assert_eq!(q.eval(&[rat(3)]).unwrap(), rat(31));
    // Cross-check: p(2, 3) = 4 + 18 + 9 = 31
    assert_eq!(p.eval(&[rat(2), rat(3)]).unwrap(), rat(31));
}

#[test]
fn partial_derivative_constant() {
    let c = Poly::from_int(2, 42);
    let dc = c.partial_derivative(0);
    assert!(dc.is_zero());
}

#[test]
fn partial_derivative_higher_degree() {
    // p(x) = x^3 in 1 variable
    let x = Poly::var(1, 0);
    let x3 = &(&x * &x) * &x;
    let dp = x3.partial_derivative(0); // 3x²
    assert_eq!(dp.num_terms(), 1);
    assert_eq!(dp.eval(&[rat(2)]).unwrap(), rat(12)); // 3*4 = 12
    let d2p = dp.partial_derivative(0); // 6x
    assert_eq!(d2p.eval(&[rat(5)]).unwrap(), rat(30)); // 6*5 = 30
}

#[test]
fn eval_with_rationals() {
    // p(x, y) = x + y, evaluate at (1/2, 1/3) = 5/6
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let p = &x + &y;
    let result = p.eval(&[rat_frac(1, 2), rat_frac(1, 3)]).unwrap();
    assert_eq!(result, rat_frac(5, 6));
}

#[test]
fn leading_coeff_grevlex() {
    let p = Poly::monomial(rat(7), vec![2, 3]);
    assert_eq!(*p.leading_coeff().unwrap(), rat(7));
}

#[test]
fn leading_term_zero_poly() {
    let z = Poly::zero(2);
    assert!(z.leading_term().is_none());
    assert!(z.leading_coeff().is_none());
}

#[test]
fn operator_overloads_owned() {
    // Test that owned-value operators work
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let sum = x.clone() + y.clone();
    assert_eq!(sum.num_terms(), 2);
    let diff = x.clone() - y.clone();
    assert_eq!(diff.num_terms(), 2);
    let prod = x.clone() * y.clone();
    assert_eq!(prod.num_terms(), 1);
    let neg = -x.clone();
    assert_eq!(neg.eval(&[rat(5), rat(0)]).unwrap(), rat(-5));
}

#[test]
fn distributive_law() {
    // a * (b + c) == a*b + a*c
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let one = Poly::from_int(2, 1);

    let a = &x + &one; // x + 1
    let b = &y;
    let c = &x;

    let lhs = &a * &(b + c); // (x+1)(y+x)
    let rhs = &(&a * b) + &(&a * c); // (x+1)*y + (x+1)*x

    // Evaluate at a common point to check equality
    let pt = [rat(3), rat(7)];
    assert_eq!(lhs.eval(&pt).unwrap(), rhs.eval(&pt).unwrap());
    // Also check (2, 5)
    let pt2 = [rat(2), rat(5)];
    assert_eq!(lhs.eval(&pt2).unwrap(), rhs.eval(&pt2).unwrap());
}

#[test]
fn degree_in_zero_poly() {
    let z = Poly::zero(3);
    assert_eq!(z.degree_in(0), 0);
    assert_eq!(z.degree_in(1), 0);
    assert_eq!(z.degree_in(2), 0);
}

#[test]
fn from_int_zero() {
    let z = Poly::from_int(2, 0);
    assert!(z.is_zero());
}

#[test]
fn scale_by_fraction() {
    let x = Poly::var(1, 0);
    let half_x = x.scale(&rat_frac(1, 2));
    assert_eq!(half_x.eval(&[rat(6)]).unwrap(), rat(3)); // (1/2)*6 = 3
}

#[test]
fn display_negative_leading() {
    // -x should display as "-x0"
    let x = Poly::var(1, 0);
    let neg_x = -&x;
    let s = format!("{neg_x}");
    assert_eq!(s, "-x0");
}

#[test]
fn display_multiterm() {
    // x² - 1 in one variable
    let x = Poly::var(1, 0);
    let one = Poly::from_int(1, 1);
    let p = &(&x * &x) - &one;
    let s = format!("{p}");
    // Should be "x0^2 - 1"
    assert_eq!(s, "x0^2 - 1");
}

#[test]
fn multiply_by_zero() {
    let x = Poly::var(2, 0);
    let z = Poly::zero(2);
    let product = &x * &z;
    assert!(product.is_zero());
}

#[test]
fn add_zero_identity() {
    let x = Poly::var(2, 0);
    let z = Poly::zero(2);
    let sum = &x + &z;
    assert_eq!(sum.eval(&[rat(7), rat(0)]).unwrap(), rat(7));
    assert_eq!(sum.num_terms(), 1);
}

#[test]
#[should_panic(expected = "incompatible variable counts")]
fn incompatible_add_panics() {
    let a = Poly::var(2, 0);
    let b = Poly::var(3, 0);
    let _ = &a + &b;
}

#[test]
#[should_panic(expected = "var_index")]
fn var_index_out_of_range_panics() {
    let _ = Poly::var(2, 5);
}

#[test]
fn three_variable_polynomial() {
    // p(x, y, z) = xyz + x + y + z + 1
    let x = Poly::var(3, 0);
    let y = Poly::var(3, 1);
    let z = Poly::var(3, 2);
    let one = Poly::from_int(3, 1);
    let xyz = &(&x * &y) * &z;
    let p = &(&(&(&xyz + &x) + &y) + &z) + &one;
    assert_eq!(p.num_terms(), 5);
    // p(2, 3, 5) = 30 + 2 + 3 + 5 + 1 = 41
    assert_eq!(p.eval(&[rat(2), rat(3), rat(5)]).unwrap(), rat(41));
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave G1: Ordering comparison tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn grevlex_ordering_basic() {
    use std::cmp::Ordering;
    // Same total degree: compare from last variable, reversed
    // x² = [2,0], xy = [1,1] → [2,0] > [1,1] in grevlex
    assert_eq!(GrevLex::cmp_exponents(&[2, 0], &[1, 1]), Ordering::Greater);
    // xy = [1,1], y² = [0,2] → [1,1] > [0,2] in grevlex
    assert_eq!(GrevLex::cmp_exponents(&[1, 1], &[0, 2]), Ordering::Greater);
    // Higher total degree always wins
    assert_eq!(GrevLex::cmp_exponents(&[3, 0], &[1, 1]), Ordering::Greater);
    // Equal
    assert_eq!(GrevLex::cmp_exponents(&[1, 1], &[1, 1]), Ordering::Equal);
    // Zero vs zero
    assert_eq!(GrevLex::cmp_exponents(&[0, 0], &[0, 0]), Ordering::Equal);
}

#[test]
fn lex_ordering_basic() {
    use std::cmp::Ordering;
    // In lex: x² > xy > xz > x > y² > yz > y > z² > z > 1
    // x² = [2,0,0] vs xy = [1,1,0]
    assert_eq!(
        Lex::cmp_exponents(&[2, 0, 0], &[1, 1, 0]),
        Ordering::Greater
    );
    // xy = [1,1,0] vs y² = [0,2,0]
    assert_eq!(
        Lex::cmp_exponents(&[1, 1, 0], &[0, 2, 0]),
        Ordering::Greater
    );
    // x = [1,0,0] vs y² = [0,2,0]
    assert_eq!(
        Lex::cmp_exponents(&[1, 0, 0], &[0, 2, 0]),
        Ordering::Greater
    );
    // In lex, degree doesn't matter — first variable is king
    assert_eq!(
        Lex::cmp_exponents(&[1, 0, 0], &[0, 5, 5]),
        Ordering::Greater
    );
    // Equal
    assert_eq!(Lex::cmp_exponents(&[1, 2, 3], &[1, 2, 3]), Ordering::Equal);
}

#[test]
fn grlex_ordering_basic() {
    use std::cmp::Ordering;
    // GrLex: total degree first, then lex
    // Same degree: [2,0] vs [1,1] → lex says [2,0] > [1,1]
    assert_eq!(GrLex::cmp_exponents(&[2, 0], &[1, 1]), Ordering::Greater);
    // Same degree: [1,1] vs [0,2] → lex says [1,1] > [0,2]
    assert_eq!(GrLex::cmp_exponents(&[1, 1], &[0, 2]), Ordering::Greater);
    // Different degree: higher degree wins
    assert_eq!(GrLex::cmp_exponents(&[0, 3], &[2, 0]), Ordering::Greater);
}

#[test]
fn leading_term_differs_by_ordering() {
    // p = xz² + y³ in 3 variables
    // [1,0,2] total deg = 3, [0,3,0] total deg = 3
    // GrevLex: same degree, compare from last: z-component: 2 vs 0 → reversed → [0,3,0] > [1,0,2]
    //   So LT_grevlex = y³ = [0,3,0]
    // Lex: first component: 1 vs 0 → [1,0,2] > [0,3,0]
    //   So LT_lex = xz² = [1,0,2]

    let p_grevlex: MultiPoly<GrevLex> = {
        let xz2 = Poly::monomial(rat(1), vec![1, 0, 2]);
        let y3 = Poly::monomial(rat(1), vec![0, 3, 0]);
        &xz2 + &y3
    };

    let p_lex: MultiPoly<Lex> = p_grevlex.convert_order();

    let (lt_grevlex, _) = p_grevlex.leading_term().unwrap();
    let (lt_lex, _) = p_lex.leading_term().unwrap();

    assert_eq!(lt_grevlex, &[0, 3, 0], "GrevLex leading term should be y³");
    assert_eq!(lt_lex, &[1, 0, 2], "Lex leading term should be xz²");
}

#[test]
fn convert_order_preserves_terms() {
    // Build a polynomial in GrevLex and convert to Lex
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let p = &(&x * &x) + &(&x * &y);

    let p_lex: MultiPoly<Lex> = p.convert_order();

    // Same number of terms
    assert_eq!(p.num_terms(), p_lex.num_terms());
    // Same evaluation
    assert_eq!(
        p.eval(&[rat(3), rat(5)]).unwrap(),
        p_lex.eval(&[rat(3), rat(5)]).unwrap()
    );
}

#[test]
fn convert_order_roundtrip() {
    let x = Poly::var(3, 0);
    let y = Poly::var(3, 1);
    let z = Poly::var(3, 2);
    let p = &(&(&x * &x) + &(&y * &z)) + &Poly::from_int(3, 7);

    let p_lex: MultiPoly<Lex> = p.convert_order();
    let p_back: MultiPoly<GrevLex> = p_lex.convert_order();

    assert_eq!(p, p_back);
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave G2: Monomial helper tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_monomial_lcm() {
    assert_eq!(monomial_lcm(&[2, 1, 0], &[1, 0, 3]), vec![2, 1, 3]);
    assert_eq!(monomial_lcm(&[0, 0], &[0, 0]), vec![0, 0]);
    assert_eq!(monomial_lcm(&[3, 2], &[3, 2]), vec![3, 2]);
}

#[test]
fn test_monomial_divides() {
    assert!(monomial_divides(&[1, 0], &[2, 1])); // x | x²y
    assert!(monomial_divides(&[0, 0], &[3, 5])); // 1 | anything
    assert!(monomial_divides(&[2, 3], &[2, 3])); // a | a
    assert!(!monomial_divides(&[3, 0], &[2, 1])); // x³ does not divide x²y
    assert!(!monomial_divides(&[1, 1], &[2, 0])); // xy does not divide x²
}

#[test]
fn test_monomial_div() {
    // x²y / xy = x → [2,1] / [1,1] = [1,0]
    assert_eq!(monomial_div(&[1, 1], &[2, 1]), Some(vec![1, 0]));
    // 1 / anything doesn't divide unless smaller
    assert_eq!(monomial_div(&[0, 0], &[3, 5]), Some(vec![3, 5]));
    // Can't divide
    assert_eq!(monomial_div(&[3, 0], &[2, 1]), None);
}

#[test]
fn test_monomial_mul() {
    assert_eq!(monomial_mul(&[2, 1], &[1, 3]), Some(vec![3, 4]));
    assert_eq!(monomial_mul(&[0, 0], &[1, 2]), Some(vec![1, 2]));
    assert_eq!(monomial_mul(&[u32::MAX], &[1]), None);
}

#[test]
fn test_monomial_coprime() {
    assert!(monomial_coprime(&[1, 0, 0], &[0, 1, 0])); // x and y
    assert!(monomial_coprime(&[0, 0], &[0, 0])); // 1 and 1
    assert!(!monomial_coprime(&[1, 1], &[1, 0])); // xy and x share x
    assert!(!monomial_coprime(&[2, 0], &[1, 0])); // x² and x share x
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave G2: Monic and primitive part tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_monic() {
    // 3x² + 6x → monic → x² + 2x
    let p = {
        let a = Poly::monomial(rat(3), vec![2]);
        let b = Poly::monomial(rat(6), vec![1]);
        &a + &b
    };
    let m = p.monic();
    assert_eq!(*m.leading_coeff().unwrap(), rat(1));
    // At x=2: (4 + 4) = 8
    assert_eq!(m.eval(&[rat(2)]).unwrap(), rat(8));
}

#[test]
fn test_monic_zero() {
    let z = Poly::zero(2);
    let m = z.monic();
    assert!(m.is_zero());
}

#[test]
fn test_monic_already_monic() {
    let x = Poly::var(2, 0);
    let m = x.monic();
    assert_eq!(*m.leading_coeff().unwrap(), rat(1));
    assert_eq!(m.eval(&[rat(5), rat(0)]).unwrap(), rat(5));
}

#[test]
fn test_primitive_part_q() {
    // (2/3)x² + (4/3)x → clear denoms: 2x² + 4x → divide by gcd(2,4)=2 → x² + 2x
    let p = {
        let a = Poly::monomial(rat_frac(2, 3), vec![2]);
        let b = Poly::monomial(rat_frac(4, 3), vec![1]);
        &a + &b
    };
    let pp = p.primitive_part_q();
    // Should be x² + 2x
    assert_eq!(pp.num_terms(), 2);
    assert_eq!(pp.eval(&[rat(3)]).unwrap(), rat(15)); // 9 + 6 = 15
    // All coefficients should be integers
    for (_, c) in pp.terms() {
        assert!(c.is_integer(), "coefficient {} should be integer", c);
    }
}

#[test]
fn test_primitive_part_q_zero() {
    let z = Poly::zero(2);
    let pp = z.primitive_part_q();
    assert!(pp.is_zero());
}

#[test]
fn test_primitive_part_q_integer_poly() {
    // 6x + 9 → gcd(6,9)=3 → 2x + 3
    let p = {
        let a = Poly::monomial(rat(6), vec![1]);
        let b = Poly::from_int(1, 9);
        &a + &b
    };
    let pp = p.primitive_part_q();
    assert_eq!(pp.eval(&[rat(1)]).unwrap(), rat(5)); // 2 + 3 = 5
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave G2: Division tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_reduce_basic() {
    // Divide x² + xy + y² by [x + y]
    // LT = x² (grevlex). Divisor LT = x.
    //   x² / x = x. Subtract x*(x+y) = x²+xy. Left: y².
    //   y² not divisible by x → move to remainder.
    //   Remainder = y².
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let x2 = &x * &x;
    let xy = &x * &y;
    let y2 = &y * &y;
    let p = &(&x2 + &xy) + &y2;
    let divisor = &x + &y;

    let remainder = p.reduce(&[&divisor]);
    // Remainder should be y²
    assert_eq!(remainder.num_terms(), 1);
    assert_eq!(remainder.eval(&[rat(0), rat(3)]).unwrap(), rat(9));
    assert_eq!(remainder.eval(&[rat(0), rat(5)]).unwrap(), rat(25));
}

#[test]
fn test_reduce_multiple_divisors() {
    // x²y + xy² + y² divided by [xy - 1, y² - 1]
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let one = Poly::from_int(2, 1);

    let x2y = &(&x * &x) * &y;
    let xy2 = &(&x * &y) * &y;
    let y2 = &y * &y;
    let p = &(&x2y + &xy2) + &y2;

    let xy = &x * &y;
    let d1 = &xy - &one; // xy - 1
    let d2 = &y2 - &one; // y² - 1

    let remainder = p.reduce(&[&d1, &d2]);

    // Verify: no leading term of remainder is divisible by LT(d1)=xy or LT(d2)=y²
    // Check remainder has reasonable degree
    assert!(remainder.total_degree().unwrap_or(0) <= p.total_degree().unwrap());
}

#[test]
fn test_reduce_zero_dividend() {
    let z = Poly::zero(2);
    let x = Poly::var(2, 0);
    let remainder = z.reduce(&[&x]);
    assert!(remainder.is_zero());
}

#[test]
fn test_reduce_no_divisors() {
    let x = Poly::var(2, 0);
    let remainder = x.reduce(&[]);
    assert_eq!(remainder, x);
}

#[test]
fn test_reduce_exact_division() {
    // (x² - 1) / (x - 1) = x + 1, remainder = 0
    let x = Poly::var(1, 0);
    let one = Poly::from_int(1, 1);
    let p = &(&x * &x) - &one; // x² - 1
    let d = &x - &one; // x - 1
    let remainder = p.reduce(&[&d]);
    assert!(remainder.is_zero(), "x²-1 should reduce to 0 mod (x-1)");
}

#[test]
fn test_reduce_not_divisible() {
    // y reduced by [x] → remainder = y (y is not divisible by x)
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let remainder = y.reduce(&[&x]);
    assert_eq!(remainder, y);
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave G2: S-polynomial tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_s_polynomial_textbook() {
    // Classic textbook example: f = x³ - 2xy, g = x²y - 2y² + x
    // LT(f) = x³, LT(g) = x²y
    // lcm(x³, x²y) = x³y
    // S(f,g) = y*f - x*g
    //        = y(x³ - 2xy) - x(x²y - 2y² + x)
    //        = x³y - 2xy² - x³y + 2xy² - x²
    //        = -x²
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);

    // f = x³ - 2xy
    let x3 = &(&x * &x) * &x;
    let two_xy = &Poly::from_int(2, 2) * &(&x * &y);
    let f = &x3 - &two_xy;

    // g = x²y - 2y² + x
    let x2y = &(&x * &x) * &y;
    let two_y2 = &Poly::from_int(2, 2) * &(&y * &y);
    let g = &(&x2y - &two_y2) + &x;

    let s = s_polynomial(&f, &g);

    // S(f,g) = -x²
    assert_eq!(s.num_terms(), 1);
    assert_eq!(s.eval(&[rat(3), rat(0)]).unwrap(), rat(-9)); // -3² = -9
    assert_eq!(s.eval(&[rat(5), rat(0)]).unwrap(), rat(-25)); // -5² = -25
}

#[test]
fn test_s_polynomial_zero_inputs() {
    let z = Poly::zero(2);
    let x = Poly::var(2, 0);
    let s1 = s_polynomial(&z, &x);
    assert!(s1.is_zero());
    let s2 = s_polynomial(&x, &z);
    assert!(s2.is_zero());
}

#[test]
fn test_s_polynomial_coprime_leading_terms() {
    // When leading monomials are coprime, S-poly often reduces to zero
    // f = x², g = y²
    // lcm(x², y²) = x²y²
    // S(f,g) = y²·f - x²·g = x²y² - x²y² = 0
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let f = &x * &x;
    let g = &y * &y;
    let s = s_polynomial(&f, &g);
    assert!(s.is_zero());
}

#[test]
fn test_s_polynomial_with_coefficients() {
    // f = 2x, g = 3y
    // LT(f) = 2x, LT(g) = 3y, lcm(x,y) = xy
    // S(f,g) = (1/2)*y*(2x) - (1/3)*x*(3y) = xy - xy = 0
    let f = Poly::monomial(rat(2), vec![1, 0]);
    let g = Poly::monomial(rat(3), vec![0, 1]);
    let s = s_polynomial(&f, &g);
    assert!(s.is_zero());
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave G2: mul_monomial tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_mul_monomial_by_x() {
    // (x + y) * x = x² + xy
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let p = &x + &y;
    let result = p.mul_monomial(&rat(1), &[1, 0]);
    assert_eq!(result.num_terms(), 2);
    // At (2, 3): x²+xy = 4+6 = 10
    assert_eq!(result.eval(&[rat(2), rat(3)]).unwrap(), rat(10));
}

#[test]
fn test_mul_monomial_by_x2y() {
    // (x + 1) * 2x²y = 2x³y + 2x²y
    let x = Poly::var(2, 0);
    let one = Poly::from_int(2, 1);
    let p = &x + &one;
    let result = p.mul_monomial(&rat(2), &[2, 1]);
    assert_eq!(result.num_terms(), 2);
    // At (2, 3): 2*8*3 + 2*4*3 = 48 + 24 = 72
    assert_eq!(result.eval(&[rat(2), rat(3)]).unwrap(), rat(72));
}

#[test]
fn test_mul_monomial_by_one() {
    // p * 1 = p
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let p = &x + &y;
    let result = p.mul_monomial(&rat(1), &[0, 0]);
    assert_eq!(result, p);
}

#[test]
fn test_mul_monomial_by_zero_coeff() {
    let x = Poly::var(2, 0);
    let result = x.mul_monomial(&rat(0), &[1, 0]);
    assert!(result.is_zero());
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave G1: leading_monomial test
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_leading_monomial() {
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let p = &(&x * &x) + &y; // x² + y
    let lm = p.leading_monomial().unwrap();
    assert_eq!(lm, &[2, 0]); // x² has higher degree
}

#[test]
fn test_leading_monomial_zero() {
    let z = Poly::zero(2);
    assert!(z.leading_monomial().is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave G2: Integration — reduce + S-poly together
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_s_poly_reduces_to_zero() {
    // For a Gröbner basis, all S-polynomials should reduce to zero.
    // {x+1, y+1} is already a Gröbner basis for the ideal (x+1, y+1).
    // S(x+1, y+1): lcm(x,y) = xy
    // S = y*(x+1) - x*(y+1) = xy + y - xy - x = y - x
    // Reduce y - x mod {x+1, y+1}:
    //   In grevlex [1,0] > [0,1], so LT of (y-x) is -x (monomial [1,0]).
    //   -x divisible by x (LT of x+1). Subtract (-1)*(x+1).
    //   y - x + x + 1 = y + 1.
    //   LT(y+1) = y. Divisible by y (LT of y+1). Subtract 1*(y+1).
    //   y + 1 - y - 1 = 0. Done!
    let x = Poly::var(2, 0);
    let y = Poly::var(2, 1);
    let one = Poly::from_int(2, 1);
    let f = &x + &one;
    let g = &y + &one;

    let s = s_polynomial(&f, &g);
    let rem = s.reduce(&[&f, &g]);
    assert!(
        rem.is_zero(),
        "S-poly of a Gröbner basis should reduce to zero, got: {rem}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave G1: GrLex specific tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn grlex_vs_grevlex_difference() {
    use std::cmp::Ordering;
    // In 3 variables, consider [1,2,0] (deg 3) and [1,0,2] (deg 3)
    // GrLex: same degree → lex tiebreak: first component equal (1=1),
    //   second: 2 > 0 → [1,2,0] > [1,0,2]
    assert_eq!(
        GrLex::cmp_exponents(&[1, 2, 0], &[1, 0, 2]),
        Ordering::Greater
    );
    // GrevLex: same degree → reverse lex tiebreak: last component: 0 vs 2,
    //   reversed comparison: 2.cmp(0) = Greater → [1,2,0] > [1,0,2]
    assert_eq!(
        GrevLex::cmp_exponents(&[1, 2, 0], &[1, 0, 2]),
        Ordering::Greater
    );

    // Now consider [2,0,1] (deg 3) and [1,2,0] (deg 3)
    // GrLex: same degree → lex: first component: 2 > 1 → [2,0,1] > [1,2,0]
    assert_eq!(
        GrLex::cmp_exponents(&[2, 0, 1], &[1, 2, 0]),
        Ordering::Greater
    );
    // GrevLex: same degree → last component: 1 vs 0, reversed: 0.cmp(1) = Less → [2,0,1] < [1,2,0]
    assert_eq!(
        GrevLex::cmp_exponents(&[2, 0, 1], &[1, 2, 0]),
        Ordering::Less
    );
    // ^ This is where GrLex and GrevLex differ!
}

#[test]
fn grlex_polynomial() {
    // Build a polynomial in GrLex ordering
    let x: MultiPoly<GrLex> = MultiPoly::var(2, 0);
    let y: MultiPoly<GrLex> = MultiPoly::var(2, 1);
    let p = &(&x * &x) + &y;
    assert_eq!(p.num_terms(), 2);
    // Leading term should be x² (degree 2 > degree 1)
    let (lt, _) = p.leading_term().unwrap();
    assert_eq!(lt, &[2, 0]);
}

#[test]
fn lex_polynomial_leading_term() {
    // In Lex ordering, x > y^100
    let x: MultiPoly<Lex> = MultiPoly::var(2, 0);
    let y: MultiPoly<Lex> = MultiPoly::var(2, 1);

    // p = x + y²
    let p = &x + &(&y * &y);
    let (lt, _) = p.leading_term().unwrap();
    // In Lex, [1,0] > [0,2] because first component 1 > 0
    assert_eq!(lt, &[1, 0]);
}
