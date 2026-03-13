//! Round 3: API misuse tests — calling methods incorrectly, with degenerate
//! inputs, and in unusual orderings to find panics, incorrect error handling,
//! and silent corruption.

use symplex::prelude::*;
use symplex::matrix;

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY A: Compile/eval with wrong inputs
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn a1_compile_empty_vars_on_expr_with_x() {
    // expr.compile(&[]) on an expression containing `x` — should return None, not panic
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    let result = expr.compile(&[]);
    assert!(
        result.is_none(),
        "BUG: compile(&[]) on expr containing x should return None, got Some"
    );
}

#[test]
fn a2_compile_extra_vars_on_one_variable_expr() {
    // expr.compile(&["x", "y", "z"]) on a 1-variable expression
    // Should work (extra vars ignored) or return error — must not panic
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + 1;
    let result = expr.compile(&["x", "y", "z"]);
    match result {
        Some(f) => {
            // Extra vars should be harmless; evaluate with x=2 (y=0, z=0)
            let val = f(&[2.0, 0.0, 0.0]);
            assert!(
                (val - 5.0).abs() < 1e-10,
                "BUG: compiled fn returned {val}, expected 5.0"
            );
        }
        None => {
            // Also acceptable — library chose to reject extra vars
            // This is not a bug, just a design choice
        }
    }
}

#[test]
fn a3_compile_nonexistent_var_name() {
    // expr.compile(&["nonexistent"]) on `x + 1`
    // Should return None or the func should return NaN — must not panic
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    let result = expr.compile(&["nonexistent"]);
    match result {
        Some(f) => {
            let val = f(&[3.0]);
            // x is unresolved, so we'd expect NaN or similar
            assert!(
                val.is_nan() || val.is_infinite(),
                "BUG: compile with wrong var name returned concrete value {val}"
            );
        }
        None => {
            // Good — library correctly refused to compile
        }
    }
}

#[test]
fn a4_eval_f64_zero() {
    // ctx.int(0).eval_f64() — should return Ok(0.0)
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = zero.eval_f64();
    match result {
        Ok(val) => assert!(
            val.abs() < 1e-15,
            "BUG: eval_f64 of 0 returned {val}, expected 0.0"
        ),
        Err(e) => panic!("BUG: eval_f64 of 0 returned Err: {e}"),
    }
}

#[test]
fn a5_eval_f64_infinity() {
    // ctx.infinity().eval_f64() — should return Err or Ok(inf), not panic
    let ctx = Context::new();
    let inf = ctx.infinity();
    let result = inf.eval_f64();
    // Either Ok(f64::INFINITY) or Err is acceptable — panic is not
    match result {
        Ok(val) => {
            assert!(
                val.is_infinite() && val > 0.0,
                "eval_f64 of ∞ returned {val}, expected +inf"
            );
        }
        Err(_) => {
            // Fine — library chose to error on infinity
        }
    }
}

#[test]
fn a6_eval_f64_nan() {
    // ctx.nan().eval_f64() — should return Err or Ok(NaN), not panic
    let ctx = Context::new();
    let nan = ctx.nan();
    let result = nan.eval_f64();
    match result {
        Ok(val) => {
            // If they return Ok, it should at least be NaN
            assert!(val.is_nan(), "eval_f64 of NaN returned {val}, expected NaN");
        }
        Err(_) => {
            // Fine — library chose to error on NaN
        }
    }
}

#[test]
fn a7_eval_f64_neg_infinity() {
    // ctx.neg_infinity().eval_f64() — should return Err or Ok(-inf), not panic
    let ctx = Context::new();
    let neg_inf = ctx.neg_infinity();
    let result = neg_inf.eval_f64();
    match result {
        Ok(val) => {
            assert!(
                val.is_infinite() && val < 0.0,
                "eval_f64 of -∞ returned {val}, expected -inf"
            );
        }
        Err(_) => {
            // Fine
        }
    }
}

#[test]
fn a8_compile_constant_expression_no_vars() {
    // Compile a pure constant expression with no variable list
    let ctx = Context::new();
    let expr = &ctx.int(3) + &ctx.int(4);
    let result = expr.compile(&[]);
    match result {
        Some(f) => {
            let val = f(&[]);
            assert!(
                (val - 7.0).abs() < 1e-10,
                "BUG: compile of 3+4 with no vars returned {val}, expected 7.0"
            );
        }
        None => panic!("BUG: compile of pure constant 3+4 with &[] returned None"),
    }
}

#[test]
fn a9_eval_f64_large_integer() {
    // Evaluate a large integer that fits in f64
    let ctx = Context::new();
    let big = ctx.int(i64::MAX);
    let result = big.eval_f64();
    match result {
        Ok(val) => {
            assert!(val > 0.0, "large int should eval to positive f64, got {val}");
        }
        Err(e) => {
            // Acceptable if library has precision concerns
            eprintln!("Note: eval_f64(i64::MAX) returned Err: {e}");
        }
    }
}

#[test]
fn a10_eval_f64_imaginary_unit() {
    // ctx.i_unit().eval_f64() should return Err (has imaginary part)
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.eval_f64();
    assert!(
        result.is_err(),
        "BUG: eval_f64 of i should return Err, got Ok({:?})",
        result.ok()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY B: Degenerate matrix operations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn b1_matrix_0x0_not_constructible() {
    // Creating a 0×0 matrix via empty rows — should error
    let result = Matrix::new(vec![]);
    assert!(
        result.is_err(),
        "BUG: Matrix::new(vec![]) should return Err"
    );
}

#[test]
fn b2_matrix_empty_row() {
    // A matrix with one empty row — should error
    let result = Matrix::new(vec![vec![]]);
    assert!(
        result.is_err(),
        "BUG: Matrix::new(vec![vec![]]) should return Err"
    );
}

#[test]
fn b3_matrix_1x1_det() {
    // 1×1 matrix [[5]] — det should be 5
    let ctx = Context::new();
    let five = ctx.int(5);
    let m = Matrix::new(vec![vec![five.clone()]]).unwrap();
    let d = m.det().unwrap();
    assert_eq!(format!("{d}"), "5", "BUG: det of [[5]] should be 5, got {d}");
}

#[test]
fn b4_matrix_1x1_inv() {
    // 1×1 matrix [[5]] — inv should be [[1/5]]
    let ctx = Context::new();
    let five = ctx.int(5);
    let m = Matrix::new(vec![vec![five]]).unwrap();
    let inv = m.inv().unwrap();
    let entry = inv.get(0, 0);
    let s = format!("{entry}");
    assert!(
        s == "1/5" || s == "0.2",
        "BUG: inv of [[5]] should be [[1/5]], got [[{s}]]"
    );
}

#[test]
fn b5_matrix_1x1_eigenvals() {
    // 1×1 matrix [[5]] — eigenvals should return [5]
    let ctx = Context::new();
    let five = ctx.int(5);
    let m = Matrix::new(vec![vec![five]]).unwrap();
    let lam = ctx.symbol("lambda");
    let evals = m.eigenvals(&lam);
    match evals {
        Ok(vals) => {
            assert!(
                vals.len() == 1,
                "BUG: eigenvals of [[5]] should have exactly 1 eigenvalue, got {}",
                vals.len()
            );
            if vals.len() == 1 {
                let s = format!("{}", vals[0]);
                assert_eq!(s, "5", "BUG: eigenvalue of [[5]] should be 5, got {s}");
            }
        }
        Err(e) => panic!("BUG: eigenvals of [[5]] failed: {e}"),
    }
}

#[test]
fn b6_det_non_square() {
    // det() on a 2×3 non-square matrix — should error, not panic
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    let result = m.det();
    assert!(
        result.is_err(),
        "BUG: det of 2×3 matrix should return Err"
    );
}

#[test]
fn b7_inv_singular_matrix() {
    // inv() on a singular matrix [[1,2],[2,4]] — should return error
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(2), ctx.int(4)],
    ])
    .unwrap();
    let result = m.inv();
    assert!(
        result.is_err(),
        "BUG: inv of singular matrix should return Err, got Ok"
    );
}

#[test]
fn b8_eigenvals_non_square() {
    // eigenvals on a non-square matrix — should error
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    let lam = ctx.symbol("lambda");
    let result = m.eigenvals(&lam);
    assert!(
        result.is_err(),
        "BUG: eigenvals of non-square matrix should return Err"
    );
}

#[test]
fn b9_inv_non_square() {
    // inv() on a non-square matrix
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    let result = m.inv();
    assert!(
        result.is_err(),
        "BUG: inv of non-square matrix should return Err"
    );
}

#[test]
fn b10_trace_non_square() {
    // trace on a non-square matrix — should error
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    let result = m.trace();
    assert!(
        result.is_err(),
        "BUG: trace of non-square matrix should return Err"
    );
}

#[test]
fn b11_matrix_jagged_rows() {
    // Jagged rows — should error
    let ctx = Context::new();
    let result = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3)], // mismatched length
    ]);
    assert!(result.is_err(), "BUG: jagged rows should return Err");
}

#[test]
fn b12_matrix_inv_zero_matrix() {
    // inv of a zero matrix [[0,0],[0,0]] — singular, should error
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(0), ctx.int(0)],
        vec![ctx.int(0), ctx.int(0)],
    ])
    .unwrap();
    let result = m.inv();
    assert!(
        result.is_err(),
        "BUG: inv of zero matrix should return Err"
    );
}

#[test]
fn b13_char_poly_non_square() {
    // char_poly on non-square matrix — should error
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    let lam = ctx.symbol("lambda");
    let result = m.char_poly(&lam);
    assert!(
        result.is_err(),
        "BUG: char_poly of non-square matrix should return Err"
    );
}

#[test]
fn b14_identity_matrix_inv() {
    // Identity matrix inverse should be itself
    let ctx = Context::new();
    let id = Matrix::identity(&ctx, 3);
    let inv = id.inv().unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let entry = inv.get(i, j).eval();
            let s = format!("{entry}");
            if i == j {
                assert_eq!(s, "1", "BUG: I^-1 diagonal entry ({i},{j}) = {s}, expected 1");
            } else {
                assert_eq!(s, "0", "BUG: I^-1 off-diagonal entry ({i},{j}) = {s}, expected 0");
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY C: Solve edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn c1_solve_zero_for_x() {
    // solve(0, x) — literal zero is zero for all x
    // Should return empty vec or indicate "all x"
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let result = zero.solve(&x);
    match result {
        Ok(sols) => {
            // Empty is acceptable (means "trivially true for all x" or "no specific root")
            // A single solution of 0 might also appear if solver treats it as polynomial.
            eprintln!("solve(0, x) returned {} solution(s): {:?}",
                sols.len(),
                sols.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
            );
        }
        Err(e) => {
            // An error is tolerable — must not panic
            eprintln!("solve(0, x) returned Err: {e}");
        }
    }
}

#[test]
fn c2_solve_constant_nonzero() {
    // solve(5, x) — no solution, should return empty
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    let result = five.solve(&x);
    match result {
        Ok(sols) => {
            assert!(
                sols.is_empty(),
                "BUG: solve(5, x) should return empty, got {} solution(s): {:?}",
                sols.len(),
                sols.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
            );
        }
        Err(_) => {
            // Acceptable — "not polynomial in x" or similar
        }
    }
}

#[test]
fn c3_solve_x_for_x() {
    // solve(x, x) — solution is x=0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let solutions = x.solve_or_empty(&x);
    assert!(
        !solutions.is_empty(),
        "BUG: solve(x, x) should find x=0"
    );
    if !solutions.is_empty() {
        let s = format!("{}", solutions[0]);
        assert_eq!(s, "0", "BUG: solve(x, x) should give 0, got {s}");
    }
}

#[test]
fn c4_solve_x_squared() {
    // solve(x^2, x) — double root at 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2);
    let solutions = expr.solve_or_empty(&x);
    assert!(
        !solutions.is_empty(),
        "BUG: solve(x^2, x) should find at least one root (0)"
    );
    // All solutions should be 0
    for sol in &solutions {
        let s = format!("{sol}");
        assert_eq!(s, "0", "BUG: solve(x^2, x) should only have root 0, got {s}");
    }
}

#[test]
fn c5_solve_sin_x() {
    // solve(sin(x), x) — infinite solutions
    // Should return some representation or error gracefully — must not panic
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let result = expr.solve(&x);
    match result {
        Ok(sols) => {
            // At minimum x=0 should be among solutions
            eprintln!(
                "solve(sin(x), x) returned {} solution(s): {:?}",
                sols.len(),
                sols.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
            );
        }
        Err(e) => {
            // Acceptable — transcendental with infinite solutions
            eprintln!("solve(sin(x), x) returned Err: {e}");
        }
    }
}

#[test]
fn c6_solve_exp_x() {
    // solve(exp(x), x) — no solution (exp never zero)
    //
    // BUG FOUND: The solver returns ["ln(0)"] as a solution. This is wrong:
    //   - exp(x) is strictly positive for all finite x, so exp(x) = 0 has no solution.
    //   - The solver applies the inversion rule exp(x)=0 → x=ln(0), but ln(0) = -∞,
    //     which is not a valid finite solution.
    //   - The solver should either reject this (return empty) or check that the
    //     candidate solution is finite before returning it.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.exp();
    let result = expr.solve(&x);
    match result {
        Ok(sols) => {
            // Filter out any solution that is literally ln(0) or evaluates to ±∞
            let spurious: Vec<String> = sols
                .iter()
                .filter(|s| {
                    let txt = format!("{s}");
                    txt.contains("ln(0)") || txt.contains("oo") || txt.contains("-oo")
                })
                .map(|s| format!("{s}"))
                .collect();
            assert!(
                sols.is_empty(),
                "BUG (solve/exp): solve(exp(x), x) should return empty — exp(x) is never zero. \
                 Got {} solution(s): {:?}. Spurious entries: {:?}",
                sols.len(),
                sols.iter().map(|s| format!("{s}")).collect::<Vec<_>>(),
                spurious,
            );
        }
        Err(_) => {
            // Acceptable — solver recognises it can't be solved
        }
    }
}

#[test]
fn c7_solve_x_squared_plus_1() {
    // solve(x^2 + 1, x) — complex roots ±i
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + 1;
    let result = expr.solve(&x);
    match result {
        Ok(sols) => {
            eprintln!(
                "solve(x^2 + 1, x) returned {} solution(s): {:?}",
                sols.len(),
                sols.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
            );
            // Should have 2 complex roots: i and -i
            // Even if empty, that's OK for a real-only solver
        }
        Err(e) => {
            eprintln!("solve(x^2 + 1, x) returned Err: {e}");
        }
    }
}

#[test]
fn c8_solve_linear() {
    // Basic sanity: solve(2*x - 6, x) should return [3]
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x * 2) - 6;
    let solutions = expr.solve_or_empty(&x);
    assert_eq!(solutions.len(), 1, "BUG: solve(2x-6, x) should have 1 solution");
    let s = format!("{}", solutions[0]);
    assert_eq!(s, "3", "BUG: solve(2x-6, x) should give 3, got {s}");
}

#[test]
fn c9_solve_or_empty_on_impossible() {
    // solve_or_empty should not panic even on weird inputs
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.exp() + x.sin() + 1;
    let solutions = expr.solve_or_empty(&x);
    // Just must not panic — any result is acceptable
    eprintln!(
        "solve_or_empty(exp(x)+sin(x)+1, x) returned {} solution(s)",
        solutions.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY D: Series at problematic points
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn d1_series_1_over_x_at_zero() {
    // series(1/x, x, 0, 5) — Laurent series at a pole
    // Should not panic; may return unevaluated or partial result
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let expr = &ctx.int(1) / &x;
    let result = expr.series(&x, &zero, 5);
    let s = format!("{result}");
    eprintln!("series(1/x, x, 0, 5) = {s}");
    // Must not panic — any result is acceptable
}

#[test]
fn d2_series_tan_at_pi_over_2() {
    // series(tan(x), x, pi/2, 3) — series at a pole of tan
    //
    // BUG FOUND: The series engine produces a Taylor-form result whose
    // coefficients contain unevaluated `tan(1/2*pi)`, which is ±∞ (a pole).
    // The correct result is a Laurent series with a leading 1/(x - π/2) term,
    // or an error indicating the function has a pole at the expansion point.
    // Instead we get nonsensical finite-looking expressions with ∞ hidden inside.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi_half = &ctx.pi() / &ctx.int(2);
    let expr = x.tan();
    let result = expr.series(&x, &pi_half, 3);
    let s = format!("{result}");
    eprintln!("series(tan(x), x, pi/2, 3) = {s}");
    // Must not panic — ✓ it doesn't.
    // But the result should NOT contain unevaluated tan(pi/2) in its coefficients,
    // because tan(pi/2) is undefined (pole). A correct CAS would either:
    //   (a) return a Laurent series with negative-power terms, or
    //   (b) return an error indicating the function has a pole at the expansion point.
    assert!(
        !s.contains("tan(1/2*pi)"),
        "BUG (series/pole): series(tan(x), x, pi/2, 3) contains unevaluated tan(pi/2) \
         which is ±∞. The series engine should detect the pole and either produce a \
         Laurent series or return an error. Got: {s}"
    );
}

#[test]
fn d3_series_ln_at_zero() {
    // series(ln(x), x, 0, 5) — series at a branch point
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let expr = x.ln();
    let result = expr.series(&x, &zero, 5);
    let s = format!("{result}");
    eprintln!("series(ln(x), x, 0, 5) = {s}");
    // Must not panic
}

#[test]
fn d4_series_exp_1_over_x_at_zero() {
    // series(exp(1/x), x, 0, 5) — essential singularity
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let one_over_x = &ctx.int(1) / &x;
    let expr = one_over_x.exp();
    let result = expr.series(&x, &zero, 5);
    let s = format!("{result}");
    eprintln!("series(exp(1/x), x, 0, 5) = {s}");
    // Must not panic
}

#[test]
fn d5_series_order_zero() {
    // series(x, x, 0, 0) — order 0 series
    // Should return a constant (the value at the expansion point) or 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let result = x.series(&x, &zero, 0);
    let s = format!("{result}");
    eprintln!("series(x, x, 0, 0) = {s}");
    // Must not panic. Order 0 means just the constant term = f(0) = 0
}

#[test]
fn d6_series_exp_at_zero_order_4() {
    // Sanity: series(exp(x), x, 0, 4) should give 1 + x + x^2/2 + x^3/6
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let expr = x.exp();
    let result = expr.series(&x, &zero, 4);
    let expanded = result.expand().eval();
    let s = format!("{expanded}");
    eprintln!("series(exp(x), x, 0, 4) = {s}");
    assert!(s.contains("x"), "series of exp(x) should contain x terms: {s}");
}

#[test]
fn d7_series_constant() {
    // series(5, x, 0, 3) — constant function, series is just 5
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let five = ctx.int(5);
    let result = five.series(&x, &zero, 3);
    let s = format!("{result}");
    eprintln!("series(5, x, 0, 3) = {s}");
    // Should be just "5"
}

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY E: Assumption edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn e1_neg_of_positive_is_not_positive() {
    // Set x as positive. Check (-x).is_positive() — should be Some(false) or None
    let ctx = Context::new();
    let x = ctx.symbol("x").assume(Assumption::Positive);
    let neg_x = -&x;
    let result = neg_x.is_positive();
    assert_ne!(
        result,
        Some(true),
        "BUG: -x should not be positive when x is positive"
    );
    eprintln!("(-x).is_positive() when x is positive = {:?}", result);
}

#[test]
fn e2_twice_integer_is_even() {
    // Set x as integer. Check (2*x).is_even() — should be Some(true)
    let ctx = Context::new();
    let x = ctx.symbol("x").assume(Assumption::Integer);
    let two_x = &x * 2;
    let result = two_x.is_even();
    eprintln!("(2*x).is_even() when x is integer = {:?}", result);
    // This tests Bug 15 fix — 2*integer should be even
    // Some(true) is correct; None is tolerable but suboptimal
    assert_ne!(
        result,
        Some(false),
        "BUG: 2*x should not be odd when x is integer"
    );
}

#[test]
fn e3_square_of_real_is_nonnegative() {
    // Set x as real. Check (x^2).is_nonnegative() — should be Some(true)
    let ctx = Context::new();
    let x = ctx.symbol("x").assume(Assumption::Real);
    let x_sq = x.powi(2);
    let result = x_sq.is_nonnegative();
    eprintln!("(x^2).is_nonnegative() when x is real = {:?}", result);
    assert_ne!(
        result,
        Some(false),
        "BUG: x^2 should never be reported as possibly negative for real x"
    );
}

#[test]
fn e4_positive_implies_real() {
    // Setting x as positive should imply x is real and nonnegative
    let ctx = Context::new();
    let x = ctx.symbol("x").assume(Assumption::Positive);
    let is_real = x.is_real();
    let is_nn = x.is_nonnegative();
    let is_nz = x.is_nonzero();
    eprintln!("x positive → is_real={is_real:?}, is_nonneg={is_nn:?}, is_nonzero={is_nz:?}");
    if is_real == Some(false) {
        panic!("BUG: positive x should be real");
    }
    if is_nn == Some(false) {
        panic!("BUG: positive x should be nonnegative");
    }
}

#[test]
fn e5_zero_is_zero() {
    // 0.is_zero() should be Some(true)
    let ctx = Context::new();
    let z = ctx.int(0);
    assert_eq!(z.is_zero(), Some(true), "BUG: 0.is_zero() should be Some(true)");
}

#[test]
fn e6_integer_positive_is_natural() {
    // x is integer and positive → x is nonnegative
    let ctx = Context::new();
    let x = ctx
        .symbol("x")
        .assume(Assumption::Integer)
        .assume(Assumption::Positive);
    let result = x.is_nonnegative();
    assert_ne!(
        result,
        Some(false),
        "BUG: integer positive x should be nonnegative"
    );
}

#[test]
fn e7_sum_of_positives_is_positive() {
    // x, y both positive → x+y is positive
    let ctx = Context::new();
    let x = ctx.symbol("x").assume(Assumption::Positive);
    let y = ctx.symbol("y").assume(Assumption::Positive);
    let sum = &x + &y;
    let result = sum.is_positive();
    eprintln!("(x+y).is_positive() when both positive = {:?}", result);
    // Some(true) is ideal; None is tolerable
    assert_ne!(
        result,
        Some(false),
        "BUG: x+y should not be non-positive when both x,y are positive"
    );
}

#[test]
fn e8_product_of_positives_is_positive() {
    // x, y both positive → x*y is positive
    let ctx = Context::new();
    let x = ctx.symbol("x").assume(Assumption::Positive);
    let y = ctx.symbol("y").assume(Assumption::Positive);
    let prod = &x * &y;
    let result = prod.is_positive();
    eprintln!("(x*y).is_positive() when both positive = {:?}", result);
    assert_ne!(
        result,
        Some(false),
        "BUG: x*y should not be non-positive when both x,y are positive"
    );
}

#[test]
fn e9_is_positive_on_literal_negative() {
    // (-3).is_positive() should be Some(false)
    let ctx = Context::new();
    let neg3 = ctx.int(-3);
    let result = neg3.is_positive();
    assert_eq!(
        result,
        Some(false),
        "BUG: (-3).is_positive() should be Some(false), got {:?}",
        result
    );
}

#[test]
fn e10_is_integer_on_literal_integer() {
    // 42.is_integer() should be Some(true)
    let ctx = Context::new();
    let n = ctx.int(42);
    let result = n.is_integer();
    assert_eq!(
        result,
        Some(true),
        "BUG: 42.is_integer() should be Some(true), got {:?}",
        result
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY F: Thread safety
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn f1_concurrent_simplify_and_diff() {
    // Spawn 8 threads sharing a cloned Context. Each thread creates
    // 100 expressions, simplifies, and differentiates. No panic or corruption.
    use std::thread;

    let ctx = Context::new();
    let handles: Vec<_> = (0..8)
        .map(|t| {
            let ctx = ctx.clone();
            thread::spawn(move || {
                let x = ctx.symbol("x");
                for i in 1..=100 {
                    let expr = &x.powi(i % 7 + 1) + &(&x * ctx.int(i)) + ctx.int(i * i);
                    let simplified = expr.simplify();
                    let deriv = simplified.diff(&x);
                    // Sanity: derivative should not be empty string
                    let s = format!("{deriv}");
                    assert!(!s.is_empty(), "Thread {t}: empty derivative string");
                }
            })
        })
        .collect();

    for (i, h) in handles.into_iter().enumerate() {
        h.join().unwrap_or_else(|e| panic!("Thread {i} panicked: {e:?}"));
    }
}

#[test]
fn f2_concurrent_same_expression() {
    // Spawn 4 threads all simplifying the SAME expression simultaneously.
    // Verify all get the same result.
    use std::sync::Arc;
    use std::thread;

    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + &(&x * 2) + 1;
    // Wrap in Arc for sharing
    let expr = Arc::new(expr);

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let expr = Arc::clone(&expr);
            thread::spawn(move || {
                let result = expr.simplify();
                format!("{result}")
            })
        })
        .collect();

    let results: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    // All results should be identical
    let first = &results[0];
    for (i, r) in results.iter().enumerate() {
        assert_eq!(
            r, first,
            "BUG: thread {i} got different simplify result: {r} vs {first}"
        );
    }
}

#[test]
fn f3_concurrent_eval_f64() {
    // Multiple threads evaluating numerical expressions concurrently
    use std::thread;

    let ctx = Context::new();
    let handles: Vec<_> = (0..8)
        .map(|t| {
            let ctx = ctx.clone();
            thread::spawn(move || {
                for i in 1..=50 {
                    let x = ctx.symbol("x");
                    let val = ctx.int(i);
                    let expr = &x.powi(2) + 1;
                    let subbed = expr.subs_i64(&x, i);
                    let result = subbed.eval_f64();
                    match result {
                        Ok(v) => {
                            let expected = (i * i + 1) as f64;
                            assert!(
                                (v - expected).abs() < 1e-6,
                                "Thread {t}: eval_f64 of {i}^2+1 = {v}, expected {expected}"
                            );
                        }
                        Err(e) => panic!("Thread {t}: eval_f64 failed: {e}"),
                    }
                    drop(val);
                }
            })
        })
        .collect();

    for (i, h) in handles.into_iter().enumerate() {
        h.join().unwrap_or_else(|e| panic!("Thread {i} panicked: {e:?}"));
    }
}

#[test]
fn f4_concurrent_solve() {
    // Multiple threads solving different equations concurrently
    use std::thread;

    let ctx = Context::new();
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let ctx = ctx.clone();
            thread::spawn(move || {
                let x = ctx.symbol("x");
                for i in 1i64..=20 {
                    // solve x - i = 0 → x = i
                    let expr = &x - ctx.int(i);
                    let sols = expr.solve_or_empty(&x);
                    assert!(
                        !sols.is_empty(),
                        "Thread {t}: solve(x - {i}, x) returned empty"
                    );
                    let s = format!("{}", sols[0]);
                    assert_eq!(
                        s,
                        format!("{i}"),
                        "Thread {t}: solve(x - {i}, x) gave {s}, expected {i}"
                    );
                }
            })
        })
        .collect();

    for (i, h) in handles.into_iter().enumerate() {
        h.join().unwrap_or_else(|e| panic!("Thread {i} panicked: {e:?}"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY G: Misc edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn g1_diff_constant() {
    // diff(5, x) — derivative of a constant should be 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    let deriv = five.diff(&x);
    assert!(
        deriv.is_zero_structural(),
        "BUG: diff(5, x) should be 0, got {deriv}"
    );
}

#[test]
fn g2_integrate_zero() {
    // integrate(0, x) — integral of zero should be 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let result = zero.integrate(&x);
    let s = format!("{result}");
    assert!(
        result.is_zero_structural() || s == "0",
        "BUG: integrate(0, x) should be 0, got {s}"
    );
}

#[test]
fn g3_factor_pure_number() {
    // factor(ctx.int(12), x) — factor a pure number
    // Should return unchanged or the number itself — must not panic
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let twelve = ctx.int(12);
    let result = twelve.factor(&x);
    let s = format!("{result}");
    eprintln!("factor(12, x) = {s}");
    // Should be 12 or equivalent — must not panic
}

#[test]
fn g4_expand_atom() {
    // expand(x) — expand an atom, should return unchanged
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.expand();
    let s = format!("{result}");
    assert_eq!(s, "x", "BUG: expand(x) should return x, got {s}");
}

#[test]
fn g5_simplify_number() {
    // simplify(42) — simplify a number, should return unchanged
    let ctx = Context::new();
    let n = ctx.int(42);
    let result = n.simplify();
    let s = format!("{result}");
    assert_eq!(s, "42", "BUG: simplify(42) should return 42, got {s}");
}

#[test]
fn g6_very_long_symbol_name() {
    // Very long symbol name — should work without panic
    let ctx = Context::new();
    let long_name: String = (0..200).map(|i| format!("a{i}_")).collect();
    let sym = ctx.symbol(&long_name);
    let expr = &sym + 1;
    let deriv = expr.diff(&sym);
    let s = format!("{deriv}");
    assert_eq!(s, "1", "BUG: diff of long-named symbol should be 1, got {s}");
}

#[test]
fn g7_symbol_with_underscore() {
    // Symbol with underscores: ctx.symbol("x_1")
    let ctx = Context::new();
    let x1 = ctx.symbol("x_1");
    let expr = &x1.powi(2) + 1;
    let s = format!("{expr}");
    assert!(
        s.contains("x_1"),
        "Symbol x_1 should appear in expression: {s}"
    );
}

#[test]
fn g8_symbol_with_prime() {
    // Symbol with prime-like character: ctx.symbol("x'")
    let ctx = Context::new();
    let x_prime = ctx.symbol("x'");
    let expr = &x_prime + 1;
    let s = format!("{expr}");
    eprintln!("Expression with x': {s}");
    // Must not panic
}

#[test]
fn g9_diff_zero() {
    // diff(0, x) should be 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let deriv = zero.diff(&x);
    assert!(
        deriv.is_zero_structural(),
        "BUG: diff(0, x) should be 0, got {deriv}"
    );
}

#[test]
fn g10_diff_variable_wrt_itself() {
    // diff(x, x) should be 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let deriv = x.diff(&x);
    let s = format!("{deriv}");
    assert_eq!(s, "1", "BUG: diff(x, x) should be 1, got {s}");
}

#[test]
fn g11_diff_variable_wrt_other() {
    // diff(x, y) should be 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let deriv = x.diff(&y);
    assert!(
        deriv.is_zero_structural(),
        "BUG: diff(x, y) should be 0, got {deriv}"
    );
}

#[test]
fn g12_subs_chain() {
    // Multiple substitutions: x -> y, then y -> 5
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = x.powi(2);
    let step1 = expr.subs(&x, &y);
    let step2 = step1.subs_i64(&y, 5);
    let val = step2.eval_f64().unwrap();
    assert!(
        (val - 25.0).abs() < 1e-10,
        "BUG: (x^2)[x→y][y→5] should be 25, got {val}"
    );
}

#[test]
fn g13_eval_of_pi() {
    // ctx.pi().eval_f64() should return approximately 3.14159...
    let ctx = Context::new();
    let pi = ctx.pi();
    let val = pi.eval_f64().unwrap();
    assert!(
        (val - std::f64::consts::PI).abs() < 1e-10,
        "BUG: pi.eval_f64() = {val}, expected {}",
        std::f64::consts::PI
    );
}

#[test]
fn g14_eval_of_e() {
    // ctx.e().eval_f64() should return approximately 2.71828...
    let ctx = Context::new();
    let e = ctx.e();
    let val = e.eval_f64().unwrap();
    assert!(
        (val - std::f64::consts::E).abs() < 1e-10,
        "BUG: e.eval_f64() = {val}, expected {}",
        std::f64::consts::E
    );
}

#[test]
fn g15_factorize_pure_number() {
    // factorize(12) should return prime factors
    let ctx = Context::new();
    let n = ctx.int(12);
    let factors = n.factorize();
    match factors {
        Some(f) => {
            eprintln!("factorize(12) = {:?}", f);
            // 12 = 2^2 * 3
            assert!(!f.is_empty(), "factorize(12) returned empty vec");
        }
        None => {
            panic!("BUG: factorize(12) returned None");
        }
    }
}

#[test]
fn g16_factorize_zero() {
    // factorize(0) — should return None or empty, not panic
    let ctx = Context::new();
    let zero = ctx.int(0);
    let factors = zero.factorize();
    eprintln!("factorize(0) = {:?}", factors);
    // Must not panic. None is acceptable.
}

#[test]
fn g17_factorize_one() {
    // factorize(1) — 1 has no prime factors
    let ctx = Context::new();
    let one = ctx.int(1);
    let factors = one.factorize();
    eprintln!("factorize(1) = {:?}", factors);
    // Should be Some([]) or None — must not panic
}

#[test]
fn g18_factorize_negative() {
    // factorize(-12) — should handle sign
    let ctx = Context::new();
    let n = ctx.int(-12);
    let factors = n.factorize();
    eprintln!("factorize(-12) = {:?}", factors);
    // Must not panic
}

#[test]
fn g19_double_simplify() {
    // Simplifying an already simplified expression should be idempotent
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + &(&x * 2) + 1;
    let s1 = expr.simplify();
    let s1_str = format!("{s1}");
    let s2 = s1.simplify();
    let s2_str = format!("{s2}");
    assert_eq!(
        s1_str, s2_str,
        "BUG: simplify should be idempotent: first={s1_str}, second={s2_str}"
    );
}

#[test]
fn g20_display_special_values() {
    // Display of special values should not panic
    let ctx = Context::new();
    let inf = ctx.infinity();
    let neg_inf = ctx.neg_infinity();
    let nan = ctx.nan();
    let i = ctx.i_unit();
    let pi = ctx.pi();
    let e = ctx.e();

    let s_inf = format!("{inf}");
    let s_neg_inf = format!("{neg_inf}");
    let s_nan = format!("{nan}");
    let s_i = format!("{i}");
    let s_pi = format!("{pi}");
    let s_e = format!("{e}");

    assert!(!s_inf.is_empty(), "Display of ∞ is empty");
    assert!(!s_neg_inf.is_empty(), "Display of -∞ is empty");
    assert!(!s_nan.is_empty(), "Display of NaN is empty");
    assert!(!s_i.is_empty(), "Display of i is empty");
    assert!(!s_pi.is_empty(), "Display of π is empty");
    assert!(!s_e.is_empty(), "Display of e is empty");
    eprintln!("inf={s_inf}, -inf={s_neg_inf}, nan={s_nan}, i={s_i}, pi={s_pi}, e={s_e}");
}

#[test]
fn g21_arithmetic_with_infinity() {
    // Arithmetic with infinity should not panic
    let ctx = Context::new();
    let inf = ctx.infinity();
    let x = ctx.symbol("x");

    // inf + x
    let r1 = &inf + &x;
    let s1 = format!("{r1}");
    eprintln!("inf + x = {s1}");

    // inf * 2
    let r2 = &inf * 2;
    let s2 = format!("{r2}");
    eprintln!("inf * 2 = {s2}");

    // 0 * inf — indeterminate
    let zero = ctx.int(0);
    let r3 = &zero * &inf;
    let s3 = format!("{r3}");
    eprintln!("0 * inf = {s3}");
    // Must not panic
}

#[test]
fn g22_arithmetic_with_nan() {
    // Arithmetic with NaN should not panic
    let ctx = Context::new();
    let nan = ctx.nan();
    let x = ctx.symbol("x");

    let r1 = &nan + &x;
    let s1 = format!("{r1}");
    eprintln!("nan + x = {s1}");

    let r2 = &nan * &ctx.int(2);
    let s2 = format!("{r2}");
    eprintln!("nan * 2 = {s2}");
    // Must not panic
}

#[test]
fn g23_compile_with_pi_and_e() {
    // Compile expression containing pi and e
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + &ctx.pi();
    let func = expr.compile(&["x"]);
    match func {
        Some(f) => {
            let val = f(&[0.0]);
            assert!(
                (val - std::f64::consts::PI).abs() < 1e-10,
                "BUG: compiled (x + pi) at x=0 gave {val}, expected pi"
            );
        }
        None => {
            panic!("BUG: compile(x + pi) returned None");
        }
    }
}

#[test]
fn g24_eval_nested_function() {
    // sin(cos(0)) = sin(1) ≈ 0.8415
    let ctx = Context::new();
    let zero = ctx.int(0);
    let expr = zero.cos().sin();
    let val = expr.eval_f64().unwrap();
    let expected = 1.0_f64.sin();
    assert!(
        (val - expected).abs() < 1e-10,
        "BUG: sin(cos(0)) = {val}, expected {expected}"
    );
}

#[test]
fn g25_is_constant_checks() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    let pi = ctx.pi();
    let expr_with_x = &x + 1;

    assert_eq!(five.is_constant(), true, "BUG: 5 should be constant");
    assert_eq!(pi.is_constant(), true, "BUG: pi should be constant");
    assert_eq!(
        expr_with_x.is_constant(),
        false,
        "BUG: x+1 should not be constant"
    );
}

#[test]
fn g26_subs_with_itself() {
    // x.subs(x, x) — substituting a variable with itself
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.subs(&x, &x);
    let s = format!("{result}");
    assert_eq!(s, "x", "BUG: x.subs(x, x) should be x, got {s}");
}

#[test]
fn g27_eval_rational() {
    // ctx.rational(1, 3).eval_f64() should be approximately 0.333...
    let ctx = Context::new();
    let third = ctx.rational(1, 3);
    let val = third.eval_f64().unwrap();
    assert!(
        (val - 1.0 / 3.0).abs() < 1e-10,
        "BUG: eval_f64(1/3) = {val}, expected ~0.333"
    );
}

#[test]
fn g28_rational_zero_denominator() {
    // ctx.rational(1, 0) — division by zero in construction
    // Should panic or return inf/nan — test that the library handles it
    let ctx = Context::new();
    // This may panic — if it does, that's actually reasonable for division by zero
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let r = ctx.rational(1, 0);
        format!("{r}")
    }));
    eprintln!("ctx.rational(1, 0) result: {:?}", result);
    // Either panic or some representation is OK — we just document the behavior
}

#[test]
fn g29_matrix_determinant_identity() {
    // det(I_n) should be 1 for various sizes
    let ctx = Context::new();
    for n in 1..=5 {
        let id = Matrix::identity(&ctx, n);
        let d = id.det().unwrap();
        let val = d.eval_f64().unwrap();
        assert!(
            (val - 1.0).abs() < 1e-10,
            "BUG: det(I_{n}) = {val}, expected 1.0"
        );
    }
}

#[test]
fn g30_free_symbols_of_constant() {
    // free_symbols of a constant should be empty
    let ctx = Context::new();
    let five = ctx.int(5);
    let syms = five.free_symbols();
    assert!(
        syms.is_empty(),
        "BUG: free_symbols of 5 should be empty, got {:?}",
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
}

#[test]
fn g31_free_symbols_of_expression() {
    // free_symbols of x^2 + y should be {x, y}
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x.powi(2) + &y;
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        2,
        "BUG: free_symbols of x^2+y should have 2 symbols, got {}",
        syms.len()
    );
}

#[test]
fn g32_count_ops_empty_and_atom() {
    let ctx = Context::new();
    let five = ctx.int(5);
    let x = ctx.symbol("x");

    // count_ops of a number and symbol should be 0 or 1 — must not panic
    let ops_five = five.count_ops();
    let ops_x = x.count_ops();
    eprintln!("count_ops(5) = {ops_five}, count_ops(x) = {ops_x}");
}

#[test]
fn g33_expr_type_checks() {
    // Verify expr_type returns correct variants for various expressions
    let ctx = Context::new();
    let five = ctx.int(5);
    let x = ctx.symbol("x");
    let sum = &x + 1;

    match five.expr_type() {
        ExprType::Number => {} // expected
        other => panic!("BUG: expr_type of 5 should be Number, got {other:?}"),
    }
    match x.expr_type() {
        ExprType::Symbol => {} // expected
        other => panic!("BUG: expr_type of x should be Symbol, got {other:?}"),
    }
    match sum.expr_type() {
        ExprType::Add => {} // expected
        other => eprintln!("Note: expr_type of x+1 is {other:?} (may be canonicalized)"),
    }
}

#[test]
fn g34_integrate_constant() {
    // integrate(5, x) should be 5*x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    let result = five.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("x"),
        "BUG: integrate(5, x) should contain x, got {s}"
    );
    // Verify: d/dx(integrate(5, x)) should give back 5
    let round_trip = result.diff(&x);
    let rs = format!("{round_trip}");
    assert!(
        rs == "5",
        "BUG: d/dx(integrate(5, x)) should be 5, got {rs}"
    );
}

#[test]
fn g35_matrix_scale_by_zero() {
    // Scaling a matrix by zero should give the zero matrix
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    let zero = ctx.int(0);
    let result = m.scale(&zero);
    for i in 0..2 {
        for j in 0..2 {
            let entry = result.get(i, j).eval();
            assert!(
                entry.is_zero_structural(),
                "BUG: zero-scaled matrix entry ({i},{j}) = {entry}, expected 0"
            );
        }
    }
}

#[test]
fn g36_matrix_add_self() {
    // M + M should equal 2*M
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    let sum = &m + &m;
    let doubled = m.scale(&ctx.int(2));
    for i in 0..2 {
        for j in 0..2 {
            let s_entry = sum.get(i, j).eval_f64().unwrap();
            let d_entry = doubled.get(i, j).eval_f64().unwrap();
            assert!(
                (s_entry - d_entry).abs() < 1e-10,
                "BUG: (M+M)[{i},{j}] = {s_entry}, 2*M[{i},{j}] = {d_entry}"
            );
        }
    }
}

#[test]
fn g37_solve_quadratic() {
    // Verify quadratic formula: x^2 - 5x + 6 = 0 → x ∈ {2, 3}
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) - &(&x * 5) + 6;
    let solutions = expr.solve_or_empty(&x);
    let mut strs: Vec<String> = solutions.iter().map(|s| format!("{s}")).collect();
    strs.sort();
    assert!(
        strs.contains(&"2".to_string()) && strs.contains(&"3".to_string()),
        "BUG: roots of x^2-5x+6 should be {{2,3}}, got {strs:?}"
    );
}

#[test]
fn g38_expand_product() {
    // expand((x+1)^2) should give x^2 + 2*x + 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(2);
    let expanded = expr.expand();
    let s = format!("{expanded}");
    assert!(
        s.contains("x^2"),
        "BUG: expand((x+1)^2) should contain x^2: {s}"
    );
}

#[test]
fn g39_eval_decimal_of_sqrt2() {
    // sqrt(2) should evaluate to approximately 1.41421356...
    let ctx = Context::new();
    let two = ctx.int(2);
    let sqrt2 = two.sqrt();
    let val = sqrt2.eval_f64().unwrap();
    assert!(
        (val - std::f64::consts::SQRT_2).abs() < 1e-10,
        "BUG: sqrt(2).eval_f64() = {val}, expected {}",
        std::f64::consts::SQRT_2
    );
}

#[test]
fn g40_multiple_symbols_same_name() {
    // Creating the same symbol twice should give the same expression
    let ctx = Context::new();
    let x1 = ctx.symbol("x");
    let x2 = ctx.symbol("x");
    let diff = &x1 - &x2;
    assert!(
        diff.is_zero_structural(),
        "BUG: two symbols with same name should be identical, diff = {diff}"
    );
}

#[test]
fn g41_compile_then_eval_many_points() {
    // Compile once, evaluate at many points — stress the compiled closure
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(3) - &x + 1;
    let f = expr.compile(&["x"]).expect("should compile x^3 - x + 1");

    for i in -100..=100 {
        let xv = i as f64 / 10.0;
        let got = f(&[xv]);
        let expected = xv.powi(3) - xv + 1.0;
        assert!(
            (got - expected).abs() < 1e-6,
            "BUG: compiled fn at x={xv}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn g42_compile_two_variables() {
    // Compile a 2-variable expression
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x * &y + &x + &y;
    let f = expr.compile(&["x", "y"]).expect("should compile x*y+x+y");
    let val = f(&[3.0, 4.0]);
    let expected = 3.0 * 4.0 + 3.0 + 4.0; // 19
    assert!(
        (val - expected).abs() < 1e-10,
        "BUG: compiled x*y+x+y at (3,4) = {val}, expected {expected}"
    );
}

#[test]
fn g43_series_sin_maclaurin() {
    // Maclaurin series of sin(x) to order 5 should have x, x^3/6, x^5/120
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let result = expr.maclaurin(&x, 5);
    let expanded = result.expand();
    let s = format!("{expanded}");
    eprintln!("maclaurin(sin(x), 5) = {s}");
    assert!(s.contains("x"), "Maclaurin of sin(x) should contain x: {s}");
}

#[test]
fn g44_context_node_count() {
    // After creating many expressions, node_count should grow
    let ctx = Context::new();
    let initial = ctx.node_count();
    let x = ctx.symbol("x");
    for i in 0..100 {
        let _ = &x + ctx.int(i);
    }
    let final_count = ctx.node_count();
    assert!(
        final_count >= initial,
        "BUG: node_count decreased from {initial} to {final_count}"
    );
}

#[test]
fn g45_args_of_various_types() {
    // .args() on various expression types — must not panic
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    let sum = &x + &five;
    let prod = &x * &five;
    let pow = x.powi(2);
    let sin_x = x.sin();

    let _ = five.args();
    let _ = x.args();
    let _ = sum.args();
    let _ = prod.args();
    let _ = pow.args();
    let _ = sin_x.args();
    // None of these should panic
}

#[test]
fn g46_has_unevaluated_on_concrete() {
    // Concrete expressions should not have unevaluated forms
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + 1;
    assert!(
        !expr.has_unevaluated(),
        "BUG: x^2 + 1 should not have unevaluated forms"
    );
}

#[test]
fn g47_equals_structural_vs_mathematical() {
    // Test .equals() for structurally different but mathematically equal expressions
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e1 = (&x + 1).powi(2);
    let e2 = &x.powi(2) + &(&x * 2) + 1;
    let result = e1.equals(&e2);
    eprintln!("(x+1)^2 equals x^2+2x+1? = {:?}", result);
    // Should be Some(true) but None is tolerable
    assert_ne!(
        result,
        Some(false),
        "BUG: (x+1)^2 should not be reported as NOT equal to x^2+2x+1"
    );
}

#[test]
fn g48_solve_ode_edge_case() {
    // Attempting to "solve" an expression as an ODE that isn't one should error, not panic
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    let result = expr.try_diff(&x);
    // This is just diff, not ODE — but calling try_diff on a simple expression should work
    match result {
        Ok(d) => {
            assert_eq!(format!("{d}"), "1", "diff(x+1, x) should be 1");
        }
        Err(e) => {
            panic!("BUG: try_diff(x+1, x) returned Err: {e}");
        }
    }
}

#[test]
fn g49_context_default() {
    // Context::default() should work the same as Context::new()
    let ctx = Context::default();
    let x = ctx.symbol("x");
    let expr = x.powi(2);
    let s = format!("{expr}");
    assert_eq!(s, "x^2", "BUG: Context::default() should work, got: {s}");
}

#[test]
fn g50_matrix_transpose_rectangular() {
    // Transpose of 2×3 matrix should be 3×2
    let ctx = Context::new();
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap();
    assert_eq!(m.nrows(), 2);
    assert_eq!(m.ncols(), 3);

    let t = m.transpose();
    assert_eq!(t.nrows(), 3, "BUG: transpose of 2×3 should have 3 rows");
    assert_eq!(t.ncols(), 2, "BUG: transpose of 2×3 should have 2 cols");

    // Check an element: m[0][2] = 3 should be t[2][0] = 3
    let val = t.get(2, 0).eval_f64().unwrap();
    assert!(
        (val - 3.0).abs() < 1e-10,
        "BUG: transposed element [2,0] should be 3, got {val}"
    );
}
