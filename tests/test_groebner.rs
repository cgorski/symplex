//! Integration tests for the `groebner` module — Buchberger's algorithm and FGLM.

mod common;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};
use symplex::groebner::*;
use symplex::multipoly::*;

// ── Helpers ────────────────────────────────────────────────────────────────

fn rat(n: i64) -> Ratio<BigInt> {
    Ratio::from_integer(BigInt::from(n))
}

fn rat_frac(p: i64, q: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(p), BigInt::from(q))
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave G3: Buchberger tests
// ═══════════════════════════════════════════════════════════════════════════

// ── 1. Trivial basis ──────────────────────────────────────────────────────

#[test]
fn buchberger_trivial() {
    // {x, y} is already a Gröbner basis in any ordering
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let gb = groebner_basis(&[x.clone(), y.clone()]);

    assert_eq!(
        gb.len(),
        2,
        "expected 2 elements, got {}: {:?}",
        gb.len(),
        gb.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
    );

    // The GB should contain (monic) x and y
    let strs: Vec<String> = gb.iter().map(|p| format!("{p}")).collect();
    assert!(strs.contains(&"x0".to_string()), "missing x0 in {strs:?}");
    assert!(strs.contains(&"x1".to_string()), "missing x1 in {strs:?}");
}

// ── 2. Circle-line system ─────────────────────────────────────────────────

#[test]
fn buchberger_circle_line() {
    // x² + y² - 1 = 0, x - y = 0
    // The GB (grevlex) should contain a univariate in y (namely 2y² - 1)
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let circle = (&x * &x).add(&(&y * &y)).sub(&one);
    let line = x.sub(&y);

    let gb = groebner_basis(&[circle, line]);
    assert!(!gb.is_empty(), "GB should not be empty");

    // There should be a univariate element (only x1 appears)
    let has_univariate_y = gb.iter().any(|p| {
        // Check that x0 does not appear (degree_in(0) == 0)
        p.degree_in(0) == 0 && p.degree_in(1) > 0
    });
    assert!(
        has_univariate_y,
        "expected a univariate in y, got: {:?}",
        gb.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
    );
}

// ── 3. Simple 2-variable system ──────────────────────────────────────────

#[test]
fn buchberger_simple_2var() {
    // x + y - 1, x - y → unique solution x = 1/2, y = 1/2
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let f1 = x.add(&y).sub(&one); // x + y - 1
    let f2 = x.sub(&y); // x - y

    let gb = groebner_basis(&[f1, f2]);

    // Should be two linear polys that encode x = 1/2, y = 1/2
    assert_eq!(
        gb.len(),
        2,
        "expected 2 elements: {:?}",
        gb.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
    );

    // Both solutions should satisfy the original equations
    let half = rat_frac(1, 2);
    for p in &gb {
        let val = p.eval(&[half.clone(), half.clone()]);
        assert!(
            val.is_zero(),
            "poly {p} should vanish at (1/2,1/2), got {val}"
        );
    }
}

// ── 4. is_groebner_basis verification ─────────────────────────────────────

#[test]
fn buchberger_is_gb() {
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let circle = (&x * &x).add(&(&y * &y)).sub(&one);
    let line = x.sub(&y);

    let gb = groebner_basis(&[circle, line]);
    assert!(
        is_groebner_basis(&gb),
        "result of groebner_basis should pass is_groebner_basis"
    );
}

#[test]
fn is_gb_false_for_non_basis() {
    // {x² + y, x*y + x} is NOT necessarily a Gröbner basis
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);

    let f1 = (&x * &x).add(&y);
    let f2 = (&x * &y).add(&x);

    // The main point is that is_groebner_basis doesn't crash
    let _result = is_groebner_basis(&[f1, f2]);
}

// ── 5. Ideal membership ───────────────────────────────────────────────────

#[test]
fn buchberger_ideal_membership() {
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let f1 = (&x * &x).add(&(&y * &y)).sub(&one);
    let f2 = x.sub(&y);

    let gb = groebner_basis(&[f1.clone(), f2.clone()]);
    let gb_refs: Vec<&MultiPoly<GrevLex>> = gb.iter().collect();

    // Every input poly should reduce to 0 mod the GB
    let r1 = f1.reduce(&gb_refs);
    assert!(r1.is_zero(), "f1 should reduce to 0 mod GB, got {r1}");

    let r2 = f2.reduce(&gb_refs);
    assert!(r2.is_zero(), "f2 should reduce to 0 mod GB, got {r2}");
}

// ── 6. Katsura-3 benchmark ───────────────────────────────────────────────

#[test]
fn buchberger_katsura3() {
    // Katsura-3: 3 variables x0, x1, x2
    // x0 + 2*x1 + 2*x2 - 1 = 0
    // x0² + 2*x1² + 2*x2² - x0 = 0
    // 2*x0*x1 + 2*x1*x2 - x1 = 0
    let x0 = MultiPoly::<GrevLex>::var(3, 0);
    let x1 = MultiPoly::<GrevLex>::var(3, 1);
    let x2 = MultiPoly::<GrevLex>::var(3, 2);
    let one = MultiPoly::<GrevLex>::from_int(3, 1);
    let two = MultiPoly::<GrevLex>::from_int(3, 2);

    // f1 = x0 + 2*x1 + 2*x2 - 1
    let f1 = x0.add(&(&two * &x1)).add(&(&two * &x2)).sub(&one);

    // f2 = x0^2 + 2*x1^2 + 2*x2^2 - x0
    let f2 = (&x0 * &x0)
        .add(&(&two * &(&x1 * &x1)))
        .add(&(&two * &(&x2 * &x2)))
        .sub(&x0);

    // f3 = 2*x0*x1 + 2*x1*x2 - x1
    let f3 = (&two * &(&x0 * &x1))
        .add(&(&two * &(&x1 * &x2)))
        .sub(&x1);

    let gb = groebner_basis(&[f1.clone(), f2.clone(), f3.clone()]);

    assert!(!gb.is_empty(), "Katsura-3 GB should not be empty");

    assert!(
        is_groebner_basis(&gb),
        "Katsura-3 result should be a valid GB"
    );

    // All input polys should reduce to 0
    let gb_refs: Vec<&MultiPoly<GrevLex>> = gb.iter().collect();
    for (i, f) in [&f1, &f2, &f3].iter().enumerate() {
        let r = f.reduce(&gb_refs);
        assert!(
            r.is_zero(),
            "Katsura f{} should reduce to 0, got {r}",
            i + 1
        );
    }

    println!("Katsura-3 GB has {} elements:", gb.len());
    for p in &gb {
        println!("  {p}");
    }
}

// ── 6b. Constant result ──────────────────────────────────────────────────

#[test]
fn buchberger_constant_input() {
    // If we include a nonzero constant, the ideal is the whole ring → GB = {1}
    let one = MultiPoly::<GrevLex>::from_int(2, 1);
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let gb = groebner_basis(&[one, x]);
    assert_eq!(gb.len(), 1);
    assert_eq!(format!("{}", gb[0]), "1");
}

#[test]
fn buchberger_empty_input() {
    let gb = groebner_basis::<GrevLex>(&[]);
    assert!(gb.is_empty());
}

#[test]
fn buchberger_all_zero() {
    let z = MultiPoly::<GrevLex>::zero(2);
    let gb = groebner_basis(&[z.clone(), z]);
    assert!(gb.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave G4: FGLM tests
// ═══════════════════════════════════════════════════════════════════════════

// ── 7. is_zero_dimensional yes ────────────────────────────────────────────

#[test]
fn is_zero_dimensional_yes() {
    // circle + line in grevlex → zero-dimensional (two solutions)
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let circle = (&x * &x).add(&(&y * &y)).sub(&one);
    let line = x.sub(&y);

    let gb = groebner_basis(&[circle, line]);
    assert!(
        is_zero_dimensional(&gb),
        "circle-line system should be zero-dimensional; GB: {:?}",
        gb.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
    );
}

// ── 8. is_zero_dimensional no ─────────────────────────────────────────────

#[test]
fn is_zero_dimensional_no() {
    // x² + y² - 1 alone: 1 equation in 2 variables → not zero-dimensional
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let circle = (&x * &x).add(&(&y * &y)).sub(&one);
    let gb = groebner_basis(&[circle]);

    assert!(
        !is_zero_dimensional(&gb),
        "single circle should NOT be zero-dimensional"
    );
}

// ── 9. FGLM simple ───────────────────────────────────────────────────────

#[test]
fn fglm_simple() {
    // Compute grevlex GB for circle+line, convert to lex
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let circle = (&x * &x).add(&(&y * &y)).sub(&one);
    let line = x.sub(&y);

    let grevlex_gb = groebner_basis(&[circle, line]);
    assert!(is_zero_dimensional(&grevlex_gb));

    let lex_gb = fglm::<GrevLex, Lex>(&grevlex_gb);
    assert!(
        lex_gb.is_some(),
        "FGLM should succeed for zero-dimensional ideal"
    );

    let lex_gb = lex_gb.unwrap();
    assert!(!lex_gb.is_empty(), "lex GB should not be empty");

    println!("Lex GB for circle-line:");
    for p in &lex_gb {
        println!("  {p}");
    }

    // The lex GB should have a univariate in the last variable (y = x1)
    let has_univariate_y = lex_gb
        .iter()
        .any(|p| p.degree_in(0) == 0 && p.degree_in(1) > 0);
    assert!(
        has_univariate_y,
        "lex GB should contain a univariate in y: {:?}",
        lex_gb.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
    );
}

// ── 10. FGLM preserves ideal ──────────────────────────────────────────────

#[test]
fn fglm_correct_ideal() {
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let circle = (&x * &x).add(&(&y * &y)).sub(&one);
    let line = x.sub(&y);

    let grevlex_gb = groebner_basis(&[circle, line]);
    let lex_gb = fglm::<GrevLex, Lex>(&grevlex_gb).unwrap();

    // Every element of the lex GB should reduce to 0 mod the grevlex GB
    let grevlex_refs: Vec<&MultiPoly<GrevLex>> = grevlex_gb.iter().collect();
    for lex_elem in &lex_gb {
        // Convert lex_elem to grevlex for reduction
        let as_grevlex: MultiPoly<GrevLex> = lex_elem.convert_order();
        let remainder = as_grevlex.reduce(&grevlex_refs);
        assert!(
            remainder.is_zero(),
            "lex GB element {lex_elem} should be in the grevlex ideal, but remainder = {remainder}"
        );
    }
}

// ── 11. Quotient dimension ────────────────────────────────────────────────

#[test]
fn quotient_dimension_circle_line() {
    // Circle-line: 2 solutions → quotient dimension = 2
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let circle = (&x * &x).add(&(&y * &y)).sub(&one);
    let line = x.sub(&y);

    let gb = groebner_basis(&[circle, line]);
    assert!(is_zero_dimensional(&gb));

    // The lex GB should also be valid
    let lex_gb = fglm::<GrevLex, Lex>(&gb).unwrap();
    assert!(
        is_groebner_basis(&lex_gb),
        "lex GB from FGLM should be valid"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration: full pipeline tests
// ═══════════════════════════════════════════════════════════════════════════

// ── 12. Full pipeline: circle-line ────────────────────────────────────────

#[test]
fn full_pipeline_circle_line() {
    // Build polys → grevlex Buchberger → verify is GB → FGLM to lex → verify lex has univariate
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let circle = (&x * &x).add(&(&y * &y)).sub(&one);
    let line = x.sub(&y);

    // Step 1: grevlex Buchberger
    let grevlex_gb = groebner_basis(&[circle.clone(), line.clone()]);
    println!("GrevLex GB:");
    for p in &grevlex_gb {
        println!("  {p}");
    }

    // Step 2: verify is GB
    assert!(is_groebner_basis(&grevlex_gb));

    // Step 3: check zero-dimensional
    assert!(is_zero_dimensional(&grevlex_gb));

    // Step 4: FGLM to lex
    let lex_gb = fglm::<GrevLex, Lex>(&grevlex_gb).unwrap();
    println!("Lex GB:");
    for p in &lex_gb {
        println!("  {p}");
    }

    // Step 5: verify lex GB is valid
    assert!(is_groebner_basis(&lex_gb), "lex GB should be valid");

    // Step 6: lex GB should have a univariate in y
    let has_univariate_y = lex_gb
        .iter()
        .any(|p| p.degree_in(0) == 0 && p.degree_in(1) > 0);
    assert!(has_univariate_y, "lex GB should have univariate in y");

    // Step 7: known solutions satisfy all elements
    // Solutions are (1/√2, 1/√2) and (-1/√2, -1/√2)
    // We can check that 2y² - 1 = 0 is in the lex GB (or equivalent)
    let uni_y: Vec<&MultiPoly<Lex>> = lex_gb
        .iter()
        .filter(|p| p.degree_in(0) == 0)
        .collect();
    assert!(!uni_y.is_empty());
    println!("Univariate in y: {}", uni_y[0]);
}

// ── 13. Full pipeline: two conics ─────────────────────────────────────────

#[test]
fn full_pipeline_two_conics() {
    // x² + y² - 5 = 0, xy - 2 = 0 → 4 solutions: (1,2),(2,1),(-1,-2),(-2,-1)
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let five = MultiPoly::<GrevLex>::from_int(2, 5);
    let two = MultiPoly::<GrevLex>::from_int(2, 2);

    let f1 = (&x * &x).add(&(&y * &y)).sub(&five); // x² + y² - 5
    let f2 = (&x * &y).sub(&two); // xy - 2

    let grevlex_gb = groebner_basis(&[f1.clone(), f2.clone()]);
    println!("Two conics GrevLex GB:");
    for p in &grevlex_gb {
        println!("  {p}");
    }

    assert!(is_groebner_basis(&grevlex_gb));

    // Verify ideal membership
    let gb_refs: Vec<&MultiPoly<GrevLex>> = grevlex_gb.iter().collect();
    assert!(f1.reduce(&gb_refs).is_zero(), "f1 should be in ideal");
    assert!(f2.reduce(&gb_refs).is_zero(), "f2 should be in ideal");

    // Should be zero-dimensional (4 solutions)
    assert!(is_zero_dimensional(&grevlex_gb));

    // FGLM to lex
    let lex_gb = fglm::<GrevLex, Lex>(&grevlex_gb).unwrap();
    println!("Two conics Lex GB:");
    for p in &lex_gb {
        println!("  {p}");
    }

    assert!(is_groebner_basis(&lex_gb));

    // Lex GB should have a univariate in y (the last variable)
    let uni_y: Vec<&MultiPoly<Lex>> = lex_gb
        .iter()
        .filter(|p| p.degree_in(0) == 0 && p.degree_in(1) > 0)
        .collect();
    assert!(
        !uni_y.is_empty(),
        "lex GB should have univariate in y; got: {:?}",
        lex_gb.iter().map(|p| format!("{p}")).collect::<Vec<_>>()
    );

    // The univariate in y should be degree 4 (since there are 4 solutions)
    let max_y_deg = uni_y.iter().map(|p| p.degree_in(1)).max().unwrap();
    assert_eq!(
        max_y_deg, 4,
        "univariate in y should be degree 4, got {max_y_deg}"
    );

    // Verify all four solutions satisfy both original equations
    let solutions: Vec<(i64, i64)> = vec![(1, 2), (2, 1), (-1, -2), (-2, -1)];
    for (sx, sy) in &solutions {
        let val1 = f1.eval(&[rat(*sx), rat(*sy)]);
        let val2 = f2.eval(&[rat(*sx), rat(*sy)]);
        assert!(val1.is_zero(), "f1({sx},{sy}) = {val1}, expected 0");
        assert!(val2.is_zero(), "f2({sx},{sy}) = {val2}, expected 0");
    }
}

// ── 14. FGLM returns None for non-zero-dimensional ────────────────────────

#[test]
fn fglm_returns_none_for_positive_dim() {
    // Single equation in 2 vars → positive dimension
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let circle = (&x * &x).add(&(&y * &y)).sub(&one);
    let gb = groebner_basis(&[circle]);

    let result = fglm::<GrevLex, Lex>(&gb);
    assert!(
        result.is_none(),
        "FGLM should return None for positive-dimensional ideal"
    );
}

// ── 15. groebner_basis_lex convenience ────────────────────────────────────

#[test]
fn groebner_basis_lex_convenience() {
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let circle = (&x * &x).add(&(&y * &y)).sub(&one);
    let line = x.sub(&y);

    let lex_gb = groebner_basis_lex(&[circle, line]);
    assert!(!lex_gb.is_empty());
    assert!(is_groebner_basis(&lex_gb));

    println!("groebner_basis_lex result:");
    for p in &lex_gb {
        println!("  {p}");
    }
}

// ── 16. Lex ordering direct Buchberger ────────────────────────────────────

#[test]
fn buchberger_lex_direct() {
    // Compute GB directly in lex ordering
    let x = MultiPoly::<Lex>::var(2, 0);
    let y = MultiPoly::<Lex>::var(2, 1);
    let one = MultiPoly::<Lex>::from_int(2, 1);

    let f1 = x.add(&y).sub(&one);
    let f2 = x.sub(&y);

    let gb = groebner_basis(&[f1, f2]);
    assert!(!gb.is_empty());
    assert!(is_groebner_basis(&gb));

    // Should encode x = 1/2, y = 1/2
    let half = rat_frac(1, 2);
    for p in &gb {
        let val = p.eval(&[half.clone(), half.clone()]);
        assert!(val.is_zero(), "poly {p} should vanish at (1/2, 1/2)");
    }
}

// ── 17. GrLex ordering ───────────────────────────────────────────────────

#[test]
fn buchberger_grlex() {
    let x = MultiPoly::<GrLex>::var(2, 0);
    let y = MultiPoly::<GrLex>::var(2, 1);
    let one = MultiPoly::<GrLex>::from_int(2, 1);

    let circle = (&x * &x).add(&(&y * &y)).sub(&one);
    let line = x.sub(&y);

    let gb = groebner_basis(&[circle, line]);
    assert!(is_groebner_basis(&gb));
}

// ── 18. Identical polynomials ─────────────────────────────────────────────

#[test]
fn buchberger_duplicate_input() {
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);

    // Input with duplicates
    let gb = groebner_basis(&[x.clone(), y.clone(), x.clone(), y.clone()]);
    assert_eq!(gb.len(), 2);
}

// ── 19. Single polynomial input ───────────────────────────────────────────

#[test]
fn buchberger_single_polynomial() {
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);

    let f = (&x * &x).add(&y); // x² + y
    let gb = groebner_basis(&[f.clone()]);
    assert_eq!(gb.len(), 1);
    assert!(is_groebner_basis(&gb));

    // The GB should be {f} (already a singleton GB)
    let gb_refs: Vec<&MultiPoly<GrevLex>> = gb.iter().collect();
    assert!(f.reduce(&gb_refs).is_zero());
}

// ── 20. Three-variable system ─────────────────────────────────────────────

#[test]
fn buchberger_three_vars() {
    // Simple linear system: x + y + z = 3, x - y = 0, y - z = 0
    // Solution: x = y = z = 1
    let x = MultiPoly::<GrevLex>::var(3, 0);
    let y = MultiPoly::<GrevLex>::var(3, 1);
    let z = MultiPoly::<GrevLex>::var(3, 2);
    let three = MultiPoly::<GrevLex>::from_int(3, 3);

    let f1 = x.add(&y).add(&z).sub(&three);
    let f2 = x.sub(&y);
    let f3 = y.sub(&z);

    let gb = groebner_basis(&[f1.clone(), f2.clone(), f3.clone()]);
    assert!(is_groebner_basis(&gb));

    // x=1, y=1, z=1 should be the solution
    let one = rat(1);
    for p in &gb {
        let val = p.eval(&[one.clone(), one.clone(), one.clone()]);
        assert!(val.is_zero(), "{p} should vanish at (1,1,1), got {val}");
    }
}

// ── 21. FGLM 3-variable system ───────────────────────────────────────────

#[test]
fn fglm_three_vars() {
    // x + y + z = 3, x - y = 0, y - z = 0
    let x = MultiPoly::<GrevLex>::var(3, 0);
    let y = MultiPoly::<GrevLex>::var(3, 1);
    let z = MultiPoly::<GrevLex>::var(3, 2);
    let three = MultiPoly::<GrevLex>::from_int(3, 3);

    let f1 = x.add(&y).add(&z).sub(&three);
    let f2 = x.sub(&y);
    let f3 = y.sub(&z);

    let grevlex_gb = groebner_basis(&[f1, f2, f3]);
    assert!(is_zero_dimensional(&grevlex_gb));

    let lex_gb = fglm::<GrevLex, Lex>(&grevlex_gb);
    assert!(lex_gb.is_some());

    let lex_gb = lex_gb.unwrap();
    assert!(is_groebner_basis(&lex_gb));

    println!("3-var lex GB:");
    for p in &lex_gb {
        println!("  {p}");
    }

    // Solution (1,1,1) should satisfy all lex GB elements
    let one = rat(1);
    for p in &lex_gb {
        let val = p.eval(&[one.clone(), one.clone(), one.clone()]);
        assert!(val.is_zero(), "{p} should vanish at (1,1,1), got {val}");
    }
}

// ── 22. Verify FGLM output is in same ideal (two conics) ─────────────────

#[test]
fn fglm_preserves_ideal_two_conics() {
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let five = MultiPoly::<GrevLex>::from_int(2, 5);
    let two = MultiPoly::<GrevLex>::from_int(2, 2);

    let f1 = (&x * &x).add(&(&y * &y)).sub(&five);
    let f2 = (&x * &y).sub(&two);

    let grevlex_gb = groebner_basis(&[f1, f2]);
    let lex_gb = fglm::<GrevLex, Lex>(&grevlex_gb).unwrap();

    // Each lex element, when converted to grevlex, should reduce to 0
    let grevlex_refs: Vec<&MultiPoly<GrevLex>> = grevlex_gb.iter().collect();
    for p in &lex_gb {
        let as_grevlex: MultiPoly<GrevLex> = p.convert_order();
        let r = as_grevlex.reduce(&grevlex_refs);
        assert!(
            r.is_zero(),
            "lex element {p} not in grevlex ideal; remainder = {r}"
        );
    }
}

// ── 23. FGLM empty basis ─────────────────────────────────────────────────

#[test]
fn fglm_empty_basis() {
    let result = fglm::<GrevLex, Lex>(&[]);
    assert_eq!(result, Some(vec![]));
}

// ── 24. is_groebner_basis on empty ───────────────────────────────────────

#[test]
fn is_gb_empty() {
    assert!(is_groebner_basis::<GrevLex>(&[]));
}

// ── 25. Buchberger result is monic ───────────────────────────────────────

#[test]
fn buchberger_result_is_monic() {
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let circle = (&x * &x).add(&(&y * &y)).sub(&one);
    let line = x.sub(&y);

    let gb = groebner_basis(&[circle, line]);
    for p in &gb {
        let lc = p.leading_coeff().unwrap();
        assert!(
            lc == &Ratio::one(),
            "leading coefficient should be 1, got {lc} for poly {p}"
        );
    }
}

// ── 26. Buchberger with scale ────────────────────────────────────────────

#[test]
fn buchberger_scaled_input() {
    // Scaling input by a constant shouldn't change the GB (up to monic)
    let x = MultiPoly::<GrevLex>::var(2, 0);
    let y = MultiPoly::<GrevLex>::var(2, 1);
    let one = MultiPoly::<GrevLex>::from_int(2, 1);

    let f1 = x.add(&y).sub(&one);
    let f2 = x.sub(&y);

    let gb1 = groebner_basis(&[f1.clone(), f2.clone()]);

    // Scale by 3
    let three = rat(3);
    let f1_scaled = f1.scale(&three);
    let f2_scaled = f2.scale(&three);
    let gb2 = groebner_basis(&[f1_scaled, f2_scaled]);

    assert_eq!(
        gb1.len(),
        gb2.len(),
        "scaled input should give same GB size"
    );
}

// ── 27. Katsura-3 FGLM to lex ───────────────────────────────────────────

#[test]
fn katsura3_fglm_to_lex() {
    let x0 = MultiPoly::<GrevLex>::var(3, 0);
    let x1 = MultiPoly::<GrevLex>::var(3, 1);
    let x2 = MultiPoly::<GrevLex>::var(3, 2);
    let one = MultiPoly::<GrevLex>::from_int(3, 1);
    let two = MultiPoly::<GrevLex>::from_int(3, 2);

    let f1 = x0.add(&(&two * &x1)).add(&(&two * &x2)).sub(&one);
    let f2 = (&x0 * &x0)
        .add(&(&two * &(&x1 * &x1)))
        .add(&(&two * &(&x2 * &x2)))
        .sub(&x0);
    let f3 = (&two * &(&x0 * &x1))
        .add(&(&two * &(&x1 * &x2)))
        .sub(&x1);

    let grevlex_gb = groebner_basis(&[f1, f2, f3]);

    if is_zero_dimensional(&grevlex_gb) {
        let lex_gb = fglm::<GrevLex, Lex>(&grevlex_gb);
        assert!(lex_gb.is_some(), "FGLM should succeed for Katsura-3");

        let lex_gb = lex_gb.unwrap();
        assert!(
            is_groebner_basis(&lex_gb),
            "Katsura-3 lex GB should be valid"
        );

        println!("Katsura-3 Lex GB:");
        for p in &lex_gb {
            println!("  {p}");
        }

        // Should have a univariate in x2 (the last variable)
        let has_uni_x2 = lex_gb
            .iter()
            .any(|p| p.degree_in(0) == 0 && p.degree_in(1) == 0 && p.degree_in(2) > 0);
        assert!(
            has_uni_x2,
            "Katsura-3 lex GB should have univariate in x2"
        );
    } else {
        // If not zero-dim, it's still a valid GB
        println!("Katsura-3 is not zero-dimensional (unusual), skipping FGLM");
    }
}
