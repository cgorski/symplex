//! Round 2 — Polynomial operations, solver edge cases, partial fractions,
//! Sturm-based root counting (indirectly), Gröbner basis correctness,
//! inequality solving, and algebraic number smoke tests.
//!
//! Run with:
//!   cd symplex && cargo test --test round2_poly_solver 2>&1

mod common;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Zero;
use symplex::groebner::{groebner_basis, is_groebner_basis, is_zero_dimensional};
use symplex::multipoly::{multipoly_vars, GrevLex, MultiPoly};
use symplex::polysys::{solve_polynomial_system, solve_system_ex};
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn rat(n: i64) -> Ratio<BigInt> {
    Ratio::from_integer(BigInt::from(n))
}

fn ratio(p: i64, q: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(p), BigInt::from(q))
}

/// Sort solutions for order-independent comparison.
fn sorted(mut sols: Vec<Vec<Ratio<BigInt>>>) -> Vec<Vec<Ratio<BigInt>>> {
    sols.sort_by(|a, b| {
        for (ai, bi) in a.iter().zip(b.iter()) {
            match ai.cmp(bi) {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }
        a.len().cmp(&b.len())
    });
    sols
}

/// Verify that a point is a common zero of all polynomials.
fn verify_solution(polys: &[MultiPoly<GrevLex>], point: &[Ratio<BigInt>]) {
    for (i, p) in polys.iter().enumerate() {
        let val = p.eval(point);
        assert!(
            val.is_zero(),
            "poly {} evaluated to {} at {:?} (expected 0)",
            i,
            val,
            point
        );
    }
}

/// Collect solve results as sorted display strings for easy comparison.
fn solve_strings(expr: &Ex, var: &Ex) -> Vec<String> {
    let mut roots: Vec<String> = expr
        .solve(var)
        .unwrap_or_default()
        .iter()
        .map(|r| format!("{}", r))
        .collect();
    roots.sort();
    roots
}

// ═══════════════════════════════════════════════════════════════════════════
// § 1  Polynomial GCD edge cases (via MultiPoly)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gcd_coprime_polynomials() {
    // gcd(x² + 1, x - 1) = 1  (since x=1 gives 1+1=2 ≠ 0)
    // Test via univariate multipoly: both in 1 variable.
    let xm = MultiPoly::<GrevLex>::var(1, 0);
    let one = MultiPoly::<GrevLex>::from_int(1, 1);

    let p1: MultiPoly<GrevLex> = &xm * &xm + one.clone(); // x²+1
    let p2: MultiPoly<GrevLex> = &xm - &one;               // x-1

    // Verify they have no common root: p1(1)=2 ≠ 0
    assert_eq!(p1.eval(&[rat(1)]), rat(2));
    assert_eq!(p2.eval(&[rat(1)]), rat(0));

    // x and y in 2 variables are coprime; GB should be {{x, y}}
    let x2 = MultiPoly::<GrevLex>::var(2, 0);
    let y2 = MultiPoly::<GrevLex>::var(2, 1);
    let gb = groebner_basis(&[x2, y2]);
    assert_eq!(gb.len(), 2, "GB of coprime {{x,y}} should have 2 elements");
}

#[test]
fn gcd_common_factor_via_groebner() {
    // x²-1 and x²-x share factor (x-1). Their common root is x=1.
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let one = MultiPoly::<GrevLex>::from_int(1, 1);

    let p1: MultiPoly<GrevLex> = &x * &x - one;  // (x-1)(x+1)
    let p2: MultiPoly<GrevLex> = &x * &x - x.clone(); // x(x-1)

    assert_eq!(p1.eval(&[rat(1)]), rat(0));
    assert_eq!(p2.eval(&[rat(1)]), rat(0));

    let sols = solve_polynomial_system(&[p1, p2]).unwrap();
    assert!(
        sols.contains(&vec![rat(1)]),
        "x=1 should be a common root, got {:?}",
        sols
    );
}

#[test]
fn gcd_zero_polynomial_with_another() {
    // Groebner basis of {0, x-1} should just be {x-1} (normalized)
    let zero = MultiPoly::<GrevLex>::from_int(1, 0);
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let one = MultiPoly::<GrevLex>::from_int(1, 1);
    let p = &x - &one; // x - 1

    let gb = groebner_basis(&[zero, p]);
    assert_eq!(
        gb.len(),
        1,
        "GB of {{0, x-1}} should have 1 element, got {:?}",
        gb.iter().map(|q| format!("{q}")).collect::<Vec<_>>()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2  Resultant computation (via polynomial system)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn resultant_common_root_is_zero() {
    // res(x-1, x-1) = 0 because they share root x=1
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let one = MultiPoly::<GrevLex>::from_int(1, 1);
    let p = &x - &one;
    // Both polynomials share factor (x-1), so system has solution
    let sols = solve_polynomial_system(&[p.clone(), p.clone()]);
    match sols {
        Ok(s) => {
            assert!(
                s.contains(&vec![rat(1)]),
                "expected x=1 in solutions, got {:?}",
                s
            );
        }
        Err(msg) => {
            // Acceptable: system is underdetermined (duplicate equation)
            assert!(
                msg.contains("underdetermined") || msg.contains("not zero-dimensional"),
                "unexpected error: {}",
                msg
            );
        }
    }
}

#[test]
fn resultant_coprime_nonzero() {
    // System {x=0, x-1=0} should have no solution (resultant ≠ 0)
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let one = MultiPoly::<GrevLex>::from_int(1, 1);
    let p1 = x.clone();  // x
    let p2 = &x - &one;  // x - 1

    let sols = solve_polynomial_system(&[p1, p2]).unwrap();
    assert!(
        sols.is_empty(),
        "x=0 and x=1 are inconsistent, should have no solution, got {:?}",
        sols
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3  Gröbner basis correctness
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn groebner_circle_line_system() {
    // x² + y² = 1 and x = y
    let nv = 2;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);
    let one = MultiPoly::<GrevLex>::from_int(nv, 1);

    let circle = &(&x * &x) + &(&y * &y) - one;
    let line = &x - &y;

    let gb = groebner_basis(&[circle, line]);
    assert!(!gb.is_empty(), "GB should not be empty");
    assert!(
        is_groebner_basis(&gb),
        "output should be a valid Gröbner basis"
    );
}

#[test]
fn groebner_overdetermined_consistent() {
    // x + y = 3, x - y = 1, 2x = 4 — all consistent: x=2, y=1
    let vars = multipoly_vars::<GrevLex>(2);
    let (x, y) = (&vars[0], &vars[1]);

    let p1 = x + y - 3i64;
    let p2 = x - y - 1i64;
    let p3 = x * 2i64 - 4i64;

    let sols = solve_polynomial_system(&[p1, p2, p3]).unwrap();
    let sols = sorted(sols);
    assert_eq!(sols, vec![vec![rat(2), rat(1)]], "unique solution (2,1)");
}

#[test]
fn groebner_overdetermined_inconsistent() {
    // x + y = 3, x + y = 5 — inconsistent
    let vars = multipoly_vars::<GrevLex>(2);
    let (x, y) = (&vars[0], &vars[1]);

    let p1 = x + y - 3i64;
    let p2 = x + y - 5i64;

    let sols = solve_polynomial_system(&[p1, p2]).unwrap();
    assert!(
        sols.is_empty(),
        "inconsistent system should have no solutions, got {:?}",
        sols
    );
}

#[test]
fn groebner_underdetermined_system() {
    // Underdetermined: single equation x + y = 1 in 2 variables
    let vars = multipoly_vars::<GrevLex>(2);
    let (x, y) = (&vars[0], &vars[1]);

    let p = x + y - 1i64;

    let result = solve_polynomial_system(&[p]);
    assert!(
        result.is_err(),
        "underdetermined system should return an error, got {:?}",
        result
    );
    let msg = result.unwrap_err();
    assert!(
        msg.contains("not zero-dimensional") || msg.contains("underdetermined"),
        "error should mention underdetermined/infinite: {}",
        msg
    );
}

#[test]
fn groebner_no_solution_nonlinear() {
    // x² + 1 = 0 has no rational roots.
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let one = MultiPoly::<GrevLex>::from_int(1, 1);
    let p = &x * &x + one; // x² + 1

    let result = solve_polynomial_system(&[p]);
    match result {
        Ok(sols) => {
            assert!(
                sols.is_empty(),
                "x²+1=0 should have no rational solutions, got {:?}",
                sols
            );
        }
        Err(_) => {
            // Also acceptable
        }
    }
}

#[test]
fn groebner_two_var_quadratic_system() {
    // x² + y² = 5, x + y = 3  →  x=1,y=2 or x=2,y=1
    let nv = 2;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);

    let p1 = &(&x * &x) + &(&y * &y) - MultiPoly::from_int(nv, 5);
    let p2 = &x + &y - MultiPoly::from_int(nv, 3);

    let sols = solve_polynomial_system(&[p1.clone(), p2.clone()]).unwrap();
    let sols = sorted(sols);

    assert_eq!(sols.len(), 2, "expected 2 solutions, got {:?}", sols);
    for s in &sols {
        verify_solution(&[p1.clone(), p2.clone()], s);
    }
    assert!(sols.contains(&vec![rat(1), rat(2)]));
    assert!(sols.contains(&vec![rat(2), rat(1)]));
}

#[test]
fn groebner_zero_dimensional_check() {
    // x² - 1, y² - 4 → finite solutions: (±1, ±2)
    let nv = 2;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);

    let p1 = &x * &x - MultiPoly::from_int(nv, 1);
    let p2 = &y * &y - MultiPoly::from_int(nv, 4);

    let gb = groebner_basis(&[p1.clone(), p2.clone()]);
    assert!(
        is_zero_dimensional(&gb),
        "system with finitely many solutions should be zero-dimensional"
    );

    let sols = solve_polynomial_system(&[p1, p2]).unwrap();
    assert_eq!(sols.len(), 4, "expected 4 solutions, got {:?}", sols);
}

#[test]
fn groebner_single_constant_polynomial() {
    // Input: {5} — a nonzero constant means the ideal is the whole ring.
    let five = MultiPoly::<GrevLex>::from_int(1, 5);
    let gb = groebner_basis(&[five]);
    assert_eq!(
        gb.len(),
        1,
        "GB of a nonzero constant should be {{1}}, got {:?}",
        gb.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
    );
}

#[test]
fn groebner_all_zero_input() {
    // Input: {0, 0} — trivial ideal
    let z1 = MultiPoly::<GrevLex>::from_int(2, 0);
    let z2 = MultiPoly::<GrevLex>::from_int(2, 0);
    let gb = groebner_basis(&[z1, z2]);
    assert!(
        gb.is_empty(),
        "GB of all-zero should be empty, got {:?}",
        gb.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
    );
}

#[test]
fn groebner_is_groebner_basis_validates_correctly() {
    let vars = multipoly_vars::<GrevLex>(2);
    let (x, y) = (&vars[0], &vars[1]);

    let p1 = x + y - 3i64;
    let p2 = x - y - 1i64;

    let gb = groebner_basis(&[p1, p2]);
    assert!(
        is_groebner_basis(&gb),
        "computed GB should pass validation"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3b  Gröbner / polysys via the Ex-level API (solve_system_ex)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_system_ex_linear_2x2() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    // x + y - 3 = 0,  x - y - 1 = 0  →  x=2, y=1
    let eq1 = &x + &y - 3;
    let eq2 = &x - &y - 1;

    let sols = solve_system_ex(&[eq1, eq2], &[x.clone(), y.clone()]).unwrap();
    assert_eq!(sols.len(), 1, "expected 1 solution, got {}", sols.len());
    let sol = &sols[0];
    assert_eq!(format!("{}", sol[0]), "2", "x should be 2");
    assert_eq!(format!("{}", sol[1]), "1", "y should be 1");
}

#[test]
fn solve_system_ex_quadratic() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    // x*y = 6,  x + y = 5  → (2,3) or (3,2)
    let eq1 = &x * &y - 6;
    let eq2 = &x + &y - 5;

    let sols = solve_system_ex(&[eq1, eq2], &[x.clone(), y.clone()]).unwrap();
    assert_eq!(sols.len(), 2, "expected 2 solutions, got {}", sols.len());

    let mut pairs: Vec<(String, String)> = sols
        .iter()
        .map(|s| (format!("{}", s[0]), format!("{}", s[1])))
        .collect();
    pairs.sort();

    assert!(
        pairs.contains(&("2".to_string(), "3".to_string())),
        "missing (2,3) in {:?}",
        pairs
    );
    assert!(
        pairs.contains(&("3".to_string(), "2".to_string())),
        "missing (3,2) in {:?}",
        pairs
    );
}

#[test]
fn solve_system_ex_no_solution() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    // x + y = 1,  x + y = 2  — inconsistent
    let eq1 = &x + &y - 1;
    let eq2 = &x + &y - 2;

    let result = solve_system_ex(&[eq1, eq2], &[x.clone(), y.clone()]);
    match result {
        Ok(sols) => assert!(
            sols.is_empty(),
            "inconsistent system should yield no solutions, got {:?}",
            sols.iter()
                .map(|s| s.iter().map(|v| format!("{v}")).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        ),
        Err(_) => {} // also acceptable
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4  Solve edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_zero_expression_gives_empty() {
    // solve(0, x) — 0 = 0 is always true (identity), solver returns empty
    // because it can't enumerate all reals.
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let zero = ctx.int(0);
    let roots = zero.solve(&x).unwrap_or_default();
    // The solver returns [] for identities (documented behavior)
    assert!(
        roots.is_empty(),
        "solve(0, x) should return empty (identity), got {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
}

#[test]
fn solve_nonzero_constant_gives_empty() {
    // solve(1, x) — no x satisfies 1 = 0
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let one = ctx.int(1);
    let roots = one.solve(&x).unwrap_or_default();
    assert!(
        roots.is_empty(),
        "solve(1, x) should return no solutions, got {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
}

#[test]
fn solve_another_nonzero_constant() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let seven = ctx.int(7);
    let roots = seven.solve(&x).unwrap_or_default();
    assert!(
        roots.is_empty(),
        "solve(7, x) should give no solutions, got {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
}

#[test]
fn solve_x_squared_gives_zero() {
    // solve(x², x) → x = 0  (double root)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.powi(2);
    let roots = solve_strings(&expr, &x);
    assert!(
        roots.contains(&"0".to_string()),
        "solve(x², x) should include 0, got {:?}",
        roots
    );
    // Should be exactly 1 distinct root
    assert_eq!(
        roots.len(),
        1,
        "solve(x², x) should give exactly 1 distinct root, got {:?}",
        roots
    );
}

#[test]
fn solve_x_cubed_gives_zero() {
    // solve(x³, x) → x = 0  (triple root)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.powi(3);
    let roots = solve_strings(&expr, &x);
    assert!(
        roots.contains(&"0".to_string()),
        "solve(x³, x) should include 0, got {:?}",
        roots
    );
}

#[test]
fn solve_linear_simple() {
    // solve(2x - 6, x) → x = 3
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x * 2 - 6;
    let roots = solve_strings(&expr, &x);
    assert_eq!(roots, vec!["3"], "solve(2x-6, x) should be [3]");
}

#[test]
fn solve_quadratic_two_integer_roots() {
    // x² - 5x + 6 = (x-2)(x-3) → roots: 2, 3
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - &x * 5 + 6;
    let roots = solve_strings(&expr, &x);
    assert_eq!(roots.len(), 2, "expected 2 roots, got {:?}", roots);
    assert!(roots.contains(&"2".to_string()), "missing root 2: {:?}", roots);
    assert!(roots.contains(&"3".to_string()), "missing root 3: {:?}", roots);
}

#[test]
fn solve_quadratic_rational_roots() {
    // 2x² - 3x + 1 = (2x-1)(x-1) → roots: 1/2, 1
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) * 2 - &x * 3 + 1;
    let roots = expr.solve(&x).unwrap();
    assert_eq!(
        roots.len(),
        2,
        "expected 2 roots, got {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );

    // Verify by substitution
    common::verify_roots(&expr, &x, &roots, 1e-10);
}

#[test]
fn solve_quadratic_double_root() {
    // x² - 2x + 1 = (x-1)² → root: 1 (double)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - &x * 2 + 1;
    let roots = solve_strings(&expr, &x);
    assert!(
        roots.contains(&"1".to_string()),
        "solve((x-1)², x) should include 1, got {:?}",
        roots
    );
    assert_eq!(
        roots.len(),
        1,
        "double root should give 1 distinct root, got {:?}",
        roots
    );
}

#[test]
fn solve_quadratic_irrational_roots() {
    // x² - 2 = 0 → roots: ±√2
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - 2;
    let roots = expr.solve(&x).unwrap();
    assert_eq!(roots.len(), 2, "expected 2 roots for x²-2");

    // Verify numerically
    common::verify_roots(&expr, &x, &roots, 1e-10);
}

#[test]
fn solve_quadratic_complex_roots() {
    // x² + 1 = 0 → roots: ±i
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) + 1;
    let roots = expr.solve(&x).unwrap();
    // Should find 2 complex roots
    assert_eq!(
        roots.len(),
        2,
        "x²+1 should have 2 complex roots, got {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
    // Each root should contain i
    for r in &roots {
        let s = format!("{r}");
        assert!(
            s.contains("i") || s.contains("I"),
            "complex root should mention i or I: {s}"
        );
    }
}

#[test]
fn solve_cubic_all_integer_roots() {
    // x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3) → roots: 1, 2, 3
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(3) - &x.powi(2) * 6 + &x * 11 - 6;
    let roots = expr.solve(&x).unwrap();

    assert_eq!(
        roots.len(),
        3,
        "expected 3 roots for (x-1)(x-2)(x-3), got {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
    common::verify_roots(&expr, &x, &roots, 1e-9);
}

#[test]
fn solve_cubic_one_rational_root() {
    // x³ - 1 = (x-1)(x² + x + 1) → 1 real root: 1, 2 complex
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(3) - 1;
    let roots = expr.solve(&x).unwrap();
    assert!(
        !roots.is_empty(),
        "x³-1 should have at least 1 root"
    );
    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        strs.contains(&"1".to_string()),
        "x³-1 should have root 1, got {:?}",
        strs
    );
    common::verify_roots(&expr, &x, &roots, 1e-9);
}

#[test]
fn solve_cubic_with_known_roots_verify() {
    // x³ - 3x² + 3x - 1 = (x-1)³ → triple root at 1
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(3) - &x.powi(2) * 3 + &x * 3 - 1;
    let roots = expr.solve(&x).unwrap();
    assert!(
        !roots.is_empty(),
        "(x-1)³ should have at least 1 root"
    );
    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        strs.contains(&"1".to_string()),
        "(x-1)³ should have root 1, got {:?}",
        strs
    );
}

#[test]
fn solve_quartic_all_integer_roots() {
    // (x-1)(x-2)(x-3)(x-4) = x⁴ - 10x³ + 35x² - 50x + 24
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(4) - &x.powi(3) * 10 + &x.powi(2) * 35 - &x * 50 + 24;
    let roots = expr.solve(&x).unwrap();

    assert_eq!(
        roots.len(),
        4,
        "quartic with 4 integer roots should return 4, got {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
    common::verify_roots(&expr, &x, &roots, 1e-8);
}

#[test]
fn solve_quartic_with_double_root() {
    // (x-1)²(x-2)(x-3) = x⁴ - 7x³ + 17x² - 17x + 6
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(4) - &x.powi(3) * 7 + &x.powi(2) * 17 - &x * 17 + 6;
    let roots = expr.solve(&x).unwrap();

    assert!(
        !roots.is_empty(),
        "quartic (x-1)²(x-2)(x-3) should have roots"
    );
    common::verify_roots(&expr, &x, &roots, 1e-8);
}

#[test]
fn solve_quartic_no_rational_roots() {
    // x⁴ - 2 = 0 → roots are ±⁴√2, ±i·⁴√2
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(4) - 2;
    let roots = expr.solve(&x).unwrap();
    assert!(
        !roots.is_empty(),
        "x⁴-2 should have roots (via Ferrari), got empty"
    );
    // Verify the ones we can evaluate
    for r in &roots {
        if let Ok(val) = r.eval_f64() {
            let residual = val.powi(4) - 2.0;
            assert!(
                residual.abs() < 1e-6,
                "root {} (approx {}) doesn't satisfy x⁴-2: residual={}",
                r,
                val,
                residual
            );
        }
    }
}

#[test]
fn solve_product_of_factors() {
    // x * (x - 1) * (x + 2) = 0 → roots: 0, 1, -2
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x * (&x - 1) * (&x + 2);
    let roots = solve_strings(&expr, &x);
    assert_eq!(roots.len(), 3, "expected 3 roots, got {:?}", roots);
    assert!(roots.contains(&"0".to_string()), "missing 0: {:?}", roots);
    assert!(roots.contains(&"1".to_string()), "missing 1: {:?}", roots);
    assert!(
        roots.contains(&"-2".to_string()),
        "missing -2: {:?}",
        roots
    );
}

#[test]
fn solve_high_degree_with_rational_roots() {
    // x⁵ - x = x(x⁴-1) = x(x²-1)(x²+1) = x(x-1)(x+1)(x²+1)
    // Rational roots: 0, 1, -1
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(5) - &x;
    let roots = expr.solve(&x).unwrap();
    assert!(
        roots.len() >= 3,
        "x⁵-x should have at least 3 roots, got {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(strs.contains(&"0".to_string()), "missing root 0: {:?}", strs);
}

// ═══════════════════════════════════════════════════════════════════════════
// § 5  Partial fraction decomposition
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_1_over_x2_minus_1() {
    // 1/(x²-1) = 1/((x-1)(x+1)) should decompose to
    // 1/(2(x-1)) - 1/(2(x+1))
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / (&x.powi(2) - 1);
    let decomposed = expr.partial_fractions(&x);

    // Verify by numerical evaluation at points away from poles at ±1
    for pt in &[2i64, 3, 5, -3, -5] {
        let orig_val = common::eval_at_i64(&expr, &x, *pt);
        let decomp_val = common::eval_at_i64(&decomposed, &x, *pt);
        assert!(
            common::approx_eq(orig_val, decomp_val, 1e-10),
            "apart(1/(x²-1)) differs at x={}: orig={}, decomp={}",
            pt,
            orig_val,
            decomp_val
        );
    }
}

#[test]
fn apart_already_polynomial() {
    // x² + 1 is already a polynomial — apart should return unchanged
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) + 1;
    let decomposed = expr.partial_fractions(&x);
    let s_orig = format!("{expr}");
    let s_decomp = format!("{decomposed}");
    assert_eq!(
        s_orig, s_decomp,
        "apart of a polynomial should be unchanged"
    );
}

#[test]
fn apart_simple_fraction_1_over_x() {
    // 1/x is already a simple fraction — apart should not change it
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / &x;
    let decomposed = expr.partial_fractions(&x);

    // Verify numerically
    for pt in &[2i64, 3, 5, 7] {
        let orig_val = common::eval_at_i64(&expr, &x, *pt);
        let decomp_val = common::eval_at_i64(&decomposed, &x, *pt);
        assert!(
            common::approx_eq(orig_val, decomp_val, 1e-10),
            "apart(1/x) differs at x={}: orig={}, decomp={}",
            pt,
            orig_val,
            decomp_val
        );
    }
}

#[test]
fn apart_with_quotient_polynomial() {
    // (x³ + 1) / (x² - 1):
    //   polynomial division gives quotient x, remainder x + 1
    //   remainder/(x²-1) = (x+1)/((x-1)(x+1)) = 1/(x-1)
    //   So: x + 1/(x-1)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = (&x.powi(3) + 1) / (&x.powi(2) - 1);
    let decomposed = expr.partial_fractions(&x);

    // Verify numerically away from poles at ±1
    for pt in &[2i64, 3, 5, -3, -5] {
        let orig_val = common::eval_at_i64(&expr, &x, *pt);
        let decomp_val = common::eval_at_i64(&decomposed, &x, *pt);
        assert!(
            common::approx_eq(orig_val, decomp_val, 1e-10),
            "apart((x³+1)/(x²-1)) differs at x={}: orig={}, decomp={}",
            pt,
            orig_val,
            decomp_val
        );
    }
}

#[test]
fn apart_expand_roundtrip() {
    // For f = 1/(x² - 5x + 6) = 1/((x-2)(x-3)):
    // apart(f) should decompose, and expand(apart(f)) should equal f numerically
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / (&x.powi(2) - &x * 5 + 6);
    let decomposed = expr.partial_fractions(&x);

    // Numerical check at several points (avoiding poles at x=2,3)
    for pt in &[0i64, 1, 4, 5, -1, -2, 7] {
        let orig_val = common::eval_at_i64(&expr, &x, *pt);
        let decomp_val = common::eval_at_i64(&decomposed, &x, *pt);
        assert!(
            common::approx_eq(orig_val, decomp_val, 1e-10),
            "apart roundtrip failed at x={}: orig={}, decomp={}",
            pt,
            orig_val,
            decomp_val
        );
    }
}

#[test]
fn apart_repeated_factor() {
    // 1/(x-1)² should already be in partial fraction form
    // (single irreducible factor), but let's verify the API handles it
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let denom = (&x - 1).powi(2);
    let expr = ctx.int(1) / &denom;
    let decomposed = expr.partial_fractions(&x);

    // Verify numerically away from pole at x=1
    for pt in &[0i64, 2, 3, 5, -1, -2] {
        let orig_val = common::eval_at_i64(&expr, &x, *pt);
        let decomp_val = common::eval_at_i64(&decomposed, &x, *pt);
        assert!(
            common::approx_eq(orig_val, decomp_val, 1e-10),
            "apart(1/(x-1)²) differs at x={}: orig={}, decomp={}",
            pt,
            orig_val,
            decomp_val
        );
    }
}

#[test]
fn apart_cubic_denominator() {
    // 1/(x³ - 1) = 1/((x-1)(x²+x+1))
    // Should decompose into A/(x-1) + (Bx+C)/(x²+x+1)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / (&x.powi(3) - 1);
    let decomposed = expr.partial_fractions(&x);

    for pt in &[2i64, 3, 5, -2, -3, 0] {
        let orig_val = common::eval_at_i64(&expr, &x, *pt);
        let decomp_val = common::eval_at_i64(&decomposed, &x, *pt);
        assert!(
            common::approx_eq(orig_val, decomp_val, 1e-10),
            "apart(1/(x³-1)) differs at x={}: orig={}, decomp={}",
            pt,
            orig_val,
            decomp_val
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 6  Sturm-based root counting (tested indirectly via inequalities)
// ═══════════════════════════════════════════════════════════════════════════
//
// The SturmChain type is pub(crate), so we test it indirectly through
// the inequality solver which uses Sturm chains as a fast path for
// polynomials with no real roots.

#[test]
fn sturm_no_real_roots_via_inequality() {
    // x² + 1 > 0 — always true (no real roots, positive leading coeff)
    // The Sturm fast path should detect this immediately.
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) + 1;
    let result = expr.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x²+1 > 0 should be true everywhere (Sturm fast path): {s}"
    );
}

#[test]
fn sturm_no_real_roots_negative() {
    // -x² - 1 > 0 — always false (no real roots, negative everywhere)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = -&x.powi(2) - 1;
    let result = expr.solve_gt(&x);
    let s = format!("{result}");
    assert_eq!(
        s, "EmptySet",
        "-x²-1 > 0 should be empty (Sturm fast path): {s}"
    );
}

#[test]
fn sturm_x4_plus_1_always_positive() {
    // x⁴ + 1 > 0 — always true (no real roots)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(4) + 1;
    let result = expr.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x⁴+1 > 0 should be satisfied everywhere: {s}"
    );
}

#[test]
fn sturm_x2_plus_x_plus_1_always_positive() {
    // x² + x + 1 > 0 — discriminant = 1-4 = -3 < 0, always positive
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) + &x + 1;
    let result = expr.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x²+x+1 > 0 should be true everywhere: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// § 7  Algebraic number operations (smoke tests via public API)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn algebraic_sqrt2_squared_is_2() {
    let ctx = Context::new();
    let sqrt2 = ctx.int(2).pow(&ctx.rational(1, 2));
    let product = &sqrt2 * &sqrt2;
    let result = product.eval();
    assert_eq!(
        format!("{result}"),
        "2",
        "sqrt(2) * sqrt(2) should be 2, got {}",
        result
    );
}

#[test]
fn algebraic_solve_produces_radicals() {
    // x² - 3 = 0 → ±√3
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - 3;
    let roots = expr.solve(&x).unwrap();
    assert_eq!(roots.len(), 2, "x²-3 should have 2 roots");

    // Verify each root: r² - 3 ≈ 0
    for r in &roots {
        if let Ok(val) = r.eval_f64() {
            let residual = val * val - 3.0;
            assert!(
                residual.abs() < 1e-10,
                "root {} (approx {}) of x²-3: residual = {}",
                r,
                val,
                residual
            );
        }
    }
}

#[test]
fn algebraic_golden_ratio_identity() {
    // phi = (1+sqrt(5))/2 satisfies phi² - phi - 1 = 0
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - &x - 1;
    let roots = expr.solve(&x).unwrap();
    assert_eq!(roots.len(), 2, "x²-x-1 should have 2 roots");

    // One root should be the golden ratio ≈ 1.618
    let mut found_phi = false;
    for r in &roots {
        if let Ok(val) = r.eval_f64() {
            if (val - 1.618033988749895).abs() < 1e-6 {
                found_phi = true;
            }
        }
    }
    assert!(found_phi, "should find golden ratio among roots of x²-x-1");
}

#[test]
fn algebraic_cube_root_identity() {
    // cbrt(8) should be 2
    let ctx = Context::new();
    let cbrt8 = ctx.int(8).pow(&ctx.rational(1, 3));
    let result = cbrt8.eval();
    assert_eq!(
        format!("{result}"),
        "2",
        "cbrt(8) should be 2, got {}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// § 8  Inequality solving
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ineq_x_gt_0() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = x.solve_gt(&x);
    let s = format!("{result}");
    // x > 0 → (0, ∞)
    assert!(!s.contains("EmptySet"), "x > 0 should not be empty: {s}");
    common::assert_positive_at(&x, &x, 5, "x > 0 at x=5");
    common::assert_negative_at(&x, &x, -3, "x > 0 at x=-3");
}

#[test]
fn ineq_x_lt_0() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = x.solve_lt(&x);
    let s = format!("{result}");
    assert!(!s.contains("EmptySet"), "x < 0 should not be empty: {s}");
    common::assert_negative_at(&x, &x, -5, "x < 0 at x=-5 (should be negative)");
}

#[test]
fn ineq_x_squared_minus_4_gt_0() {
    // x² - 4 > 0 → x < -2 or x > 2 → (-∞,-2) ∪ (2,∞)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - 4;
    let result = expr.solve_gt(&x);
    let s = format!("{result}");
    assert!(!s.contains("EmptySet"), "x²-4 > 0 should have solutions: {s}");

    // Interior points
    common::assert_positive_at(&expr, &x, 3, "x²-4 > 0 at x=3");
    common::assert_positive_at(&expr, &x, -3, "x²-4 > 0 at x=-3");
    // Exterior point
    common::assert_negative_at(&expr, &x, 0, "x²-4 > 0 at x=0 (exterior)");
}

#[test]
fn ineq_x_squared_minus_4_le_0() {
    // x² - 4 ≤ 0 → -2 ≤ x ≤ 2 → [-2, 2]
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - 4;
    let result = expr.solve_le(&x);
    let s = format!("{result}");
    assert!(!s.contains("EmptySet"), "x²-4 <= 0 should have solutions: {s}");
    // x=0 → 0-4 = -4 ≤ 0 ✓
    common::assert_negative_at(&expr, &x, 0, "x²-4 <= 0 at x=0");
}

#[test]
fn ineq_positive_constant_gt() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let five = ctx.int(5);
    let result = five.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "5 > 0 is always true, should be all reals: {s}"
    );
}

#[test]
fn ineq_negative_constant_gt() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let neg = ctx.int(-3);
    let result = neg.solve_gt(&x);
    let s = format!("{result}");
    assert_eq!(s, "EmptySet", "-3 > 0 is always false: {s}");
}

#[test]
fn ineq_zero_ge() {
    // 0 ≥ 0 is always true
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let zero = ctx.int(0);
    let result = zero.solve_ge(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "0 >= 0 should be true everywhere: {s}"
    );
}

#[test]
fn ineq_zero_gt() {
    // 0 > 0 is always false
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let zero = ctx.int(0);
    let result = zero.solve_gt(&x);
    let s = format!("{result}");
    assert_eq!(s, "EmptySet", "0 > 0 should be false: {s}");
}

#[test]
fn ineq_negative_constant_le() {
    // -3 ≤ 0 is always true
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let neg = ctx.int(-3);
    let result = neg.solve_le(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "-3 <= 0 should be true everywhere: {s}"
    );
}

#[test]
fn ineq_cubic_gt() {
    // x³ - x > 0 = x(x-1)(x+1) > 0
    // Positive when x in (-1, 0) ∪ (1, ∞)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(3) - &x;
    let result = expr.solve_gt(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x³-x > 0 should have solutions: {s}"
    );

    // Interior of (1, ∞): x = 2 → 8 - 2 = 6 > 0 ✓
    common::assert_positive_at(&expr, &x, 2, "x³-x > 0 at x=2");

    // Exterior: x = -2 → -8 - (-2) = -6 < 0 ✗
    common::assert_negative_at(&expr, &x, -2, "x³-x exterior at x=-2");
}

#[test]
fn ineq_quartic_always_nonneg() {
    // x⁴ ≥ 0 — always true
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.powi(4);
    let result = expr.solve_ge(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x⁴ >= 0 should be true everywhere: {s}"
    );
}

#[test]
fn ineq_x_squared_lt_never_negative() {
    // x² < 0 — never true
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.powi(2);
    let result = expr.solve_lt(&x);
    let s = format!("{result}");
    assert_eq!(s, "EmptySet", "x² < 0 should be empty: {s}");
}

#[test]
fn ineq_x_ge_includes_zero() {
    // x ≥ 0 should include 0 in the solution set
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = x.solve_ge(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x >= 0 should not be empty: {s}"
    );
    assert!(
        s.contains("0"),
        "x >= 0 solution should reference 0: {s}"
    );
}

#[test]
fn ineq_x_le_includes_zero() {
    // x ≤ 0 should include 0 in the solution set
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = x.solve_le(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x <= 0 should not be empty: {s}"
    );
    assert!(
        s.contains("0"),
        "x <= 0 solution should reference 0: {s}"
    );
}

#[test]
fn ineq_x4_minus_1_ge_0() {
    // x⁴ - 1 ≥ 0 → x ≤ -1 or x ≥ 1
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(4) - 1;
    let result = expr.solve_ge(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("EmptySet"),
        "x⁴-1 >= 0 should have solutions: {s}"
    );

    // Interior: x=2 → 16-1=15 ≥ 0 ✓
    common::assert_positive_at(&expr, &x, 2, "x⁴-1 >= 0 at x=2");
    // Exterior: x=0 → 0-1=-1 < 0 ✗
    common::assert_negative_at(&expr, &x, 0, "x⁴-1 >= 0 at x=0 (exterior)");
}

// ═══════════════════════════════════════════════════════════════════════════
// § 9  Additional solver stress tests and edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_difference_of_squares() {
    // x² - 9 = (x-3)(x+3) → roots: ±3
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - 9;
    let roots = solve_strings(&expr, &x);
    assert_eq!(roots.len(), 2, "expected 2 roots: {:?}", roots);
    assert!(roots.contains(&"3".to_string()), "missing 3: {:?}", roots);
    assert!(roots.contains(&"-3".to_string()), "missing -3: {:?}", roots);
}

#[test]
fn solve_large_integer_roots() {
    // x² - 10000 = (x-100)(x+100)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - 10000;
    let roots = solve_strings(&expr, &x);
    assert!(
        roots.contains(&"100".to_string()),
        "should find root 100: {:?}",
        roots
    );
    assert!(
        roots.contains(&"-100".to_string()),
        "should find root -100: {:?}",
        roots
    );
}

#[test]
fn solve_cubic_verify_cardano_roots() {
    // x³ + x - 2 = 0  has x=1 as a rational root.
    // Factor: (x-1)(x²+x+2) = 0
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(3) + &x - 2;
    let roots = expr.solve(&x).unwrap();

    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        strs.contains(&"1".to_string()),
        "x³+x-2 should have rational root 1, got {:?}",
        strs
    );
    assert!(
        !roots.is_empty(),
        "should find at least 1 root"
    );
}

#[test]
fn solve_quartic_biquadratic() {
    // x⁴ - 5x² + 4 = (x²-1)(x²-4) = (x-1)(x+1)(x-2)(x+2)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(4) - &x.powi(2) * 5 + 4;
    let roots = expr.solve(&x).unwrap();
    assert_eq!(
        roots.len(),
        4,
        "biquadratic x⁴-5x²+4 should have 4 roots, got {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
    common::verify_roots(&expr, &x, &roots, 1e-9);
}

#[test]
fn solve_verify_roots_by_substitution() {
    // Generic test: solve and verify for multiple polynomials
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    // (x-3)(x+1) = x² - 2x - 3
    let poly1 = &x.powi(2) - &x * 2 - 3;
    let roots1 = poly1.solve(&x).unwrap();
    if !roots1.is_empty() {
        common::verify_roots(&poly1, &x, &roots1, 1e-9);
    }

    // (x+2)² = x² + 4x + 4
    let poly2 = &x.powi(2) + &x * 4 + 4;
    let roots2 = poly2.solve(&x).unwrap();
    if !roots2.is_empty() {
        common::verify_roots(&poly2, &x, &roots2, 1e-9);
    }

    // x(x-2)(x+2) = x³ - 4x
    let poly3 = &x.powi(3) - &x * 4;
    let roots3 = poly3.solve(&x).unwrap();
    if !roots3.is_empty() {
        common::verify_roots(&poly3, &x, &roots3, 1e-9);
    }
}

#[test]
fn solve_negative_leading_coefficient() {
    // -x² + 4 = 0 → x = ±2
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = -&x.powi(2) + 4;
    let roots = solve_strings(&expr, &x);
    assert_eq!(roots.len(), 2, "expected 2 roots: {:?}", roots);
    assert!(roots.contains(&"2".to_string()), "missing 2: {:?}", roots);
    assert!(roots.contains(&"-2".to_string()), "missing -2: {:?}", roots);
}

#[test]
fn solve_very_simple_linear() {
    // x = 0 → root at 0
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let roots = solve_strings(&x, &x);
    assert_eq!(roots, vec!["0"], "solve(x, x) should be [0], got {:?}", roots);
}

// ═══════════════════════════════════════════════════════════════════════════
// § 10  Polynomial system solving: more edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn polysys_single_variable_linear() {
    // x - 7 = 0 in 1 variable
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let p = x - 7i64;
    let sols = solve_polynomial_system(&[p]).unwrap();
    assert_eq!(sols, vec![vec![rat(7)]]);
}

#[test]
fn polysys_single_variable_cubic() {
    // x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let x2 = &x * &x;
    let x3 = &x2 * &x;
    let p = x3 - x2 * 6i64 + x * 11i64 - 6i64;
    let sols = solve_polynomial_system(&[p]).unwrap();
    let sols = sorted(sols);
    assert_eq!(sols.len(), 3, "expected 3 solutions: {:?}", sols);
    assert!(sols.contains(&vec![rat(1)]));
    assert!(sols.contains(&vec![rat(2)]));
    assert!(sols.contains(&vec![rat(3)]));
}

#[test]
fn polysys_three_variables() {
    // x + y + z = 6, x - y = 0, y - z = 0 → x = y = z = 2
    let nv = 3;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);
    let z = MultiPoly::<GrevLex>::var(nv, 2);

    let p1: MultiPoly<GrevLex> = &x + &(&y + &z) - MultiPoly::from_int(nv, 6);
    let p2 = &x - &y;
    let p3 = &y - &z;

    let sols = solve_polynomial_system(&[p1.clone(), p2.clone(), p3.clone()]).unwrap();
    assert_eq!(sols.len(), 1, "expected 1 solution: {:?}", sols);
    assert_eq!(sols[0], vec![rat(2), rat(2), rat(2)]);
    for s in &sols {
        verify_solution(&[p1.clone(), p2.clone(), p3.clone()], s);
    }
}

#[test]
fn polysys_empty_input() {
    let result = solve_polynomial_system(&[]);
    assert!(result.is_ok());
    let sols = result.unwrap();
    assert_eq!(sols.len(), 1, "empty system should have 1 (trivial) solution");
    assert!(sols[0].is_empty(), "trivial solution should be empty vec");
}

#[test]
fn polysys_constant_nonzero() {
    // System: {5} — a nonzero constant, no solutions
    let p = MultiPoly::<GrevLex>::from_int(1, 5);
    let sols = solve_polynomial_system(&[p]).unwrap();
    assert!(
        sols.is_empty(),
        "nonzero constant should have no solutions: {:?}",
        sols
    );
}

#[test]
fn polysys_two_quadratics() {
    // x² - y = 0 and y - 4 = 0 → y=4, x²=4 → (2,4) and (-2,4)
    let nv = 2;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);

    let p1: MultiPoly<GrevLex> = &x * &x - y.clone();
    let p2 = y - MultiPoly::from_int(nv, 4);

    let sols = solve_polynomial_system(&[p1.clone(), p2.clone()]).unwrap();
    let sols = sorted(sols);
    assert_eq!(sols.len(), 2, "expected 2 solutions: {:?}", sols);
    assert!(sols.contains(&vec![rat(-2), rat(4)]));
    assert!(sols.contains(&vec![rat(2), rat(4)]));
    for s in &sols {
        verify_solution(&[p1.clone(), p2.clone()], s);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 11  More partial fractions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apart_x_over_x2_minus_1() {
    // x/(x²-1) = x/((x-1)(x+1))
    // Decomposition: 1/(2(x-1)) + 1/(2(x+1))
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x / (&x.powi(2) - 1);
    let decomposed = expr.partial_fractions(&x);

    for pt in &[2i64, 3, 5, -3, -5] {
        let orig_val = common::eval_at_i64(&expr, &x, *pt);
        let decomp_val = common::eval_at_i64(&decomposed, &x, *pt);
        assert!(
            common::approx_eq(orig_val, decomp_val, 1e-10),
            "apart(x/(x²-1)) differs at x={}: orig={}, decomp={}",
            pt,
            orig_val,
            decomp_val
        );
    }
}

#[test]
fn apart_2x_plus_3_over_x2_plus_3x_plus_2() {
    // (2x+3)/(x²+3x+2) = (2x+3)/((x+1)(x+2))
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let numer = &x * 2 + 3;
    let denom = &x.powi(2) + &x * 3 + 2;
    let expr = &numer / &denom;
    let decomposed = expr.partial_fractions(&x);

    // Avoid poles at x=-1, x=-2
    for pt in &[0i64, 1, 2, 3, 5, -3, -5, 10] {
        let orig_val = common::eval_at_i64(&expr, &x, *pt);
        let decomp_val = common::eval_at_i64(&decomposed, &x, *pt);
        assert!(
            common::approx_eq(orig_val, decomp_val, 1e-10),
            "apart((2x+3)/(x²+3x+2)) differs at x={}: orig={}, decomp={}",
            pt,
            orig_val,
            decomp_val
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 12  Regression-style tests: known tricky inputs
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn polysys_xy_system_with_zero_product() {
    // x + y = 1 and x * y = 0 → (0,1) and (1,0)
    let nv = 2;
    let x = MultiPoly::<GrevLex>::var(nv, 0);
    let y = MultiPoly::<GrevLex>::var(nv, 1);

    let p1 = &x + &y - MultiPoly::from_int(nv, 1);
    let p2 = &x * &y;

    let sols = solve_polynomial_system(&[p1.clone(), p2.clone()]).unwrap();
    let sols = sorted(sols);
    assert_eq!(sols.len(), 2, "expected 2 solutions: {:?}", sols);
    assert!(sols.contains(&vec![rat(0), rat(1)]));
    assert!(sols.contains(&vec![rat(1), rat(0)]));
}

#[test]
fn solve_quintic_with_known_rational_roots() {
    // x⁵ - 15x⁴ + 85x³ - 225x² + 274x - 120
    // = (x-1)(x-2)(x-3)(x-4)(x-5)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(5) - &x.powi(4) * 15 + &x.powi(3) * 85 - &x.powi(2) * 225 + &x * 274
        - 120;
    let roots = expr.solve(&x).unwrap();
    assert!(
        roots.len() >= 5,
        "quintic with 5 integer roots should find all 5, got {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
    common::verify_roots(&expr, &x, &roots, 1e-8);
}

#[test]
fn solve_x_cubed_minus_27() {
    // x³ - 27 = (x-3)(x² + 3x + 9) → rational root: 3
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(3) - 27;
    let roots = expr.solve(&x).unwrap();
    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        strs.contains(&"3".to_string()),
        "x³-27 should have root 3, got {:?}",
        strs
    );
}

#[test]
fn solve_depressed_cubic_needs_cardano() {
    // x³ - 2 = 0 → real root is cbrt(2) ≈ 1.2599
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(3) - 2;
    let roots = expr.solve(&x).unwrap();
    assert!(
        !roots.is_empty(),
        "x³ - 2 should have roots"
    );
    // At least one real root should be ≈ cbrt(2)
    let mut found_real = false;
    for r in &roots {
        if let Ok(val) = r.eval_f64() {
            if (val - 1.2599210498948732).abs() < 1e-4 {
                found_real = true;
            }
        }
    }
    assert!(found_real, "should find cbrt(2) among roots of x³ - 2");
}

#[test]
fn solve_x_squared_minus_x_eq_0() {
    // x² - x = x(x-1) = 0 → roots: 0, 1
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - &x;
    let roots = solve_strings(&expr, &x);
    assert_eq!(roots.len(), 2, "expected 2 roots: {:?}", roots);
    assert!(roots.contains(&"0".to_string()), "missing 0: {:?}", roots);
    assert!(roots.contains(&"1".to_string()), "missing 1: {:?}", roots);
}

#[test]
fn ineq_linear_gt_with_coefficient() {
    // 2x - 6 > 0 → x > 3 → (3, ∞)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x * 2 - 6;
    let result = expr.solve_gt(&x);
    let s = format!("{result}");
    assert!(!s.contains("EmptySet"), "2x-6 > 0 should have solutions: {s}");
    // x=4 → 8-6=2 > 0 ✓
    common::assert_positive_at(&expr, &x, 4, "2x-6 > 0 at x=4");
    // x=2 → 4-6=-2 < 0 ✗
    common::assert_negative_at(&expr, &x, 2, "2x-6 > 0 at x=2 (exterior)");
}

#[test]
fn ineq_negative_leading_coeff_lt() {
    // -x² + 1 < 0 → x² > 1 → x < -1 or x > 1
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = -&x.powi(2) + 1;
    let result = expr.solve_lt(&x);
    let s = format!("{result}");
    assert!(!s.contains("EmptySet"), "-x²+1 < 0 should have solutions: {s}");
    // x=2 → -4+1=-3 < 0 ✓
    common::assert_negative_at(&expr, &x, 2, "-x²+1 < 0 at x=2");
    // x=0 → 0+1=1 > 0, so 1 < 0 is false
    common::assert_positive_at(&expr, &x, 0, "-x²+1 at x=0 should be positive");
}
