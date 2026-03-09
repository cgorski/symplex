//! Integration tests for the control systems module:
//! state-space models, transfer functions, and Routh-Hurwitz stability.

use symplex::control::{is_routh_stable, routh_array, StateSpace, TransferFunction};
use symplex::matrix::Matrix;

// ═══════════════════════════════════════════════════════════════════════════
// 1. StateSpace: dimensions
// ═══════════════════════════════════════════════════════════════════════════

use symplex::prelude::*;
#[test]
fn state_space_dimensions() {
    let __ctx = Context::new();
    // 2 states, 1 input, 1 output
    let a = Matrix::new(vec![
        vec![__ctx.int(0), __ctx.int(1)],
        vec![__ctx.int(-2), __ctx.int(-3)],
    ]).unwrap();
    let b = Matrix::new(vec![vec![__ctx.int(0)], vec![__ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![__ctx.int(1), __ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    assert_eq!(ss.num_states(), 2);
    assert_eq!(ss.num_inputs(), 1);
    assert_eq!(ss.num_outputs(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. StateSpace: poles of a 2×2 system
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn state_space_poles_2x2() {
    let __ctx = Context::new();
    // A = [[0, 1], [-2, -3]]
    // Characteristic polynomial: s^2 + 3s + 2 = (s+1)(s+2)
    // Poles at s = -1 and s = -2
    let a = Matrix::new(vec![
        vec![__ctx.int(0), __ctx.int(1)],
        vec![__ctx.int(-2), __ctx.int(-3)],
    ]).unwrap();
    let b = Matrix::new(vec![vec![__ctx.int(0)], vec![__ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![__ctx.int(1), __ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    let s = __ctx.symbol("s");
    let poles = ss.poles(&s);
    assert_eq!(poles.len(), 2, "Expected 2 poles, got {}", poles.len());

    let mut pole_strs: Vec<String> = poles.iter().map(|p| format!("{p}")).collect();
    pole_strs.sort();
    assert_eq!(
        pole_strs,
        vec!["-1", "-2"],
        "Poles should be -1 and -2, got: {pole_strs:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. StateSpace: characteristic polynomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn state_space_char_poly() {
    let __ctx = Context::new();
    // A = [[0, 1], [-2, -3]]
    // det(sI - A) = det([[s, -1], [2, s+3]]) = s(s+3) + 2 = s^2 + 3s + 2
    let a = Matrix::new(vec![
        vec![__ctx.int(0), __ctx.int(1)],
        vec![__ctx.int(-2), __ctx.int(-3)],
    ]).unwrap();
    let b = Matrix::new(vec![vec![__ctx.int(0)], vec![__ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![__ctx.int(1), __ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    let s = __ctx.symbol("s");
    let cp = ss.char_poly(&s);

    // Verify that the roots of the char poly are the poles
    let at_neg1 = cp.subs(&s, &__ctx.int(-1)).simplify();
    assert_eq!(
        format!("{at_neg1}"),
        "0",
        "char_poly(-1) should be 0, got: {at_neg1}"
    );

    let at_neg2 = cp.subs(&s, &__ctx.int(-2)).simplify();
    assert_eq!(
        format!("{at_neg2}"),
        "0",
        "char_poly(-2) should be 0, got: {at_neg2}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. StateSpace: controllability (controllable system)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn state_space_controllability() {
    let __ctx = Context::new();
    // A = [[0, 1], [0, 0]], B = [[0], [1]]
    // Controllability matrix: [B, AB] = [[0, 1], [1, 0]] → rank 2 → controllable
    let a = Matrix::new(vec![
        vec![__ctx.int(0), __ctx.int(1)],
        vec![__ctx.int(0), __ctx.int(0)],
    ]).unwrap();
    let b = Matrix::new(vec![vec![__ctx.int(0)], vec![__ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![__ctx.int(1), __ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    assert!(ss.is_controllable(), "System should be controllable");
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. StateSpace: not controllable
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn state_space_not_controllable() {
    let __ctx = Context::new();
    // A = [[1, 0], [0, 2]], B = [[1], [0]]
    // Controllability matrix: [B, AB] = [[1, 1], [0, 0]] → rank 1 ≠ 2 → not controllable
    let a = Matrix::new(vec![
        vec![__ctx.int(1), __ctx.int(0)],
        vec![__ctx.int(0), __ctx.int(2)],
    ]).unwrap();
    let b = Matrix::new(vec![vec![__ctx.int(1)], vec![__ctx.int(0)]]).unwrap();
    let c = Matrix::new(vec![vec![__ctx.int(1), __ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    assert!(
        !ss.is_controllable(),
        "System should NOT be controllable"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. StateSpace: observability
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn state_space_observability() {
    let __ctx = Context::new();
    // A = [[0, 1], [0, 0]], C = [[1, 0]]
    // Observability matrix: [C; CA] = [[1, 0], [0, 1]] → rank 2 → observable
    let a = Matrix::new(vec![
        vec![__ctx.int(0), __ctx.int(1)],
        vec![__ctx.int(0), __ctx.int(0)],
    ]).unwrap();
    let b = Matrix::new(vec![vec![__ctx.int(0)], vec![__ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![__ctx.int(1), __ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    assert!(ss.is_observable(), "System should be observable");
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. StateSpace: stable system
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn state_space_stable() {
    let __ctx = Context::new();
    // A = [[0, 1], [-2, -3]] → eigenvalues -1, -2 → stable
    let a = Matrix::new(vec![
        vec![__ctx.int(0), __ctx.int(1)],
        vec![__ctx.int(-2), __ctx.int(-3)],
    ]).unwrap();
    let b = Matrix::new(vec![vec![__ctx.int(0)], vec![__ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![__ctx.int(1), __ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    let stability = ss.is_stable();
    assert_eq!(stability, Some(true), "System should be stable");
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. StateSpace: unstable system
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn state_space_unstable() {
    let __ctx = Context::new();
    // A = [[1, 0], [0, -1]] → eigenvalues 1, -1 → unstable (has positive eigenvalue)
    let a = Matrix::new(vec![
        vec![__ctx.int(1), __ctx.int(0)],
        vec![__ctx.int(0), __ctx.int(-1)],
    ]).unwrap();
    let b = Matrix::new(vec![vec![__ctx.int(1)], vec![__ctx.int(0)]]).unwrap();
    let c = Matrix::new(vec![vec![__ctx.int(1), __ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    let stability = ss.is_stable();
    assert_eq!(stability, Some(false), "System should be unstable");
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. TransferFunction: poles
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn transfer_function_poles() {
    let __ctx = Context::new();
    // G(s) = 1 / (s^2 + 3s + 2) = 1 / ((s+1)(s+2))
    // Poles at s = -1 and s = -2
    let s = __ctx.symbol("s");
    let tf = TransferFunction::new(
        __ctx.int(1),
        &s * &s + &s * 3 + 2,
        s.clone(),
    );

    let poles = tf.poles();
    assert_eq!(poles.len(), 2, "Expected 2 poles, got {}", poles.len());

    let mut pole_strs: Vec<String> = poles.iter().map(|p| format!("{p}")).collect();
    pole_strs.sort();
    assert_eq!(
        pole_strs,
        vec!["-1", "-2"],
        "Poles should be -1 and -2, got: {pole_strs:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. TransferFunction: zeros
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn transfer_function_zeros() {
    let __ctx = Context::new();
    // G(s) = (s + 1) / (s^2 + 3s + 2)
    // Zero at s = -1
    let s = __ctx.symbol("s");
    let tf = TransferFunction::new(
        &s + 1,
        &s * &s + &s * 3 + 2,
        s.clone(),
    );

    let zeros = tf.zeros();
    assert_eq!(zeros.len(), 1, "Expected 1 zero, got {}", zeros.len());
    assert_eq!(
        format!("{}", zeros[0]),
        "-1",
        "Zero should be at -1, got: {}",
        zeros[0]
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. TransferFunction: DC gain
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn transfer_function_dc_gain() {
    let __ctx = Context::new();
    // G(s) = 5 / (s + 2)
    // DC gain = G(0) = 5/2
    let s = __ctx.symbol("s");
    let tf = TransferFunction::new(__ctx.int(5), &s + 2, s.clone());

    let gain = tf.dc_gain();
    let val = gain.eval_f64().unwrap();
    assert!(
        (val - 2.5).abs() < 1e-10,
        "DC gain should be 5/2 = 2.5, got: {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. TransferFunction: series connection
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn transfer_function_series() {
    let __ctx = Context::new();
    // G1(s) = 1/(s+1), G2(s) = 1/(s+2)
    // G_series = 1/((s+1)(s+2))
    // DC gain of series: G1(0)*G2(0) = 1*1/2 = 1/2
    let s = __ctx.symbol("s");
    let g1 = TransferFunction::new(__ctx.int(1), &s + 1, s.clone());
    let g2 = TransferFunction::new(__ctx.int(1), &s + 2, s.clone());

    let gs = g1.series(&g2);

    // Evaluate at s=0: should be 1/((0+1)(0+2)) = 1/2
    let val = gs.eval_at(&__ctx.int(0)).eval_f64().unwrap();
    assert!(
        (val - 0.5).abs() < 1e-10,
        "Series DC gain should be 0.5, got: {val}"
    );

    // Evaluate at s=1: should be 1/((1+1)(1+2)) = 1/6
    let val_1 = gs.eval_at(&__ctx.int(1)).eval_f64().unwrap();
    assert!(
        (val_1 - 1.0 / 6.0).abs() < 1e-10,
        "Series at s=1 should be 1/6, got: {val_1}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. TransferFunction: parallel connection
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn transfer_function_parallel() {
    let __ctx = Context::new();
    // G1(s) = 1/(s+1), G2(s) = 1/(s+2)
    // G_par = 1/(s+1) + 1/(s+2) = (2s+3)/((s+1)(s+2))
    // At s=0: 1/1 + 1/2 = 3/2
    let s = __ctx.symbol("s");
    let g1 = TransferFunction::new(__ctx.int(1), &s + 1, s.clone());
    let g2 = TransferFunction::new(__ctx.int(1), &s + 2, s.clone());

    let gp = g1.parallel(&g2);

    let val = gp.eval_at(&__ctx.int(0)).eval_f64().unwrap();
    assert!(
        (val - 1.5).abs() < 1e-10,
        "Parallel DC gain should be 1.5, got: {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. TransferFunction: negative unity feedback
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn transfer_function_feedback() {
    let __ctx = Context::new();
    // G(s) = 10/(s+1)
    // G_cl = G/(1+G) = 10/(s+1) / (1 + 10/(s+1))
    //      = 10/(s+1) / ((s+1+10)/(s+1))
    //      = 10/(s+11)
    // DC gain of closed loop: 10/11
    let s = __ctx.symbol("s");
    let g = TransferFunction::new(__ctx.int(10), &s + 1, s.clone());

    let gcl = g.feedback();

    let val = gcl.eval_at(&__ctx.int(0)).eval_f64().unwrap();
    let expected = 10.0 / 11.0;
    assert!(
        (val - expected).abs() < 1e-10,
        "Feedback DC gain should be 10/11 ≈ {expected}, got: {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. Routh array: stable polynomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn routh_array_stable() {
    let __ctx = Context::new();
    // s^3 + 2s^2 + 3s + 4
    // Coefficients: [1, 2, 3, 4]
    //
    // Routh table:
    // Row 0: 1  3
    // Row 1: 2  4
    // Row 2: (2*3 - 1*4)/2 = 2/2 = 1
    // Row 3: (1*4 - 2*0)/1 = 4
    //
    // First column: [1, 2, 1, 4] — all positive → stable
    let coeffs = vec![
        __ctx.int(1),
        __ctx.int(2),
        __ctx.int(3),
        __ctx.int(4),
    ];

    let table = routh_array(&coeffs);
    assert_eq!(table.len(), 4, "Routh array should have 4 rows");

    // Verify first column entries are all positive
    let stability = is_routh_stable(&coeffs);
    assert_eq!(
        stability,
        Some(true),
        "s^3 + 2s^2 + 3s + 4 should be Routh-stable"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 16. Routh array: unstable polynomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn routh_array_unstable() {
    let __ctx = Context::new();
    // s^3 + 2s^2 + s + 8
    // Coefficients: [1, 2, 1, 8]
    //
    // Routh table:
    // Row 0: 1  1
    // Row 1: 2  8
    // Row 2: (2*1 - 1*8)/2 = -6/2 = -3
    // Row 3: (-3*8 - 2*0)/(-3) = 8
    //
    // First column: [1, 2, -3, 8] — sign change → unstable
    let coeffs = vec![
        __ctx.int(1),
        __ctx.int(2),
        __ctx.int(1),
        __ctx.int(8),
    ];

    let stability = is_routh_stable(&coeffs);
    assert_eq!(
        stability,
        Some(false),
        "s^3 + 2s^2 + s + 8 should be Routh-unstable"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 17. Controllability matrix size
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn controllability_matrix_size() {
    let __ctx = Context::new();
    // 3 states, 2 inputs → controllability matrix should be 3×6
    let a = Matrix::new(vec![
        vec![__ctx.int(1), __ctx.int(0), __ctx.int(0)],
        vec![__ctx.int(0), __ctx.int(2), __ctx.int(0)],
        vec![__ctx.int(0), __ctx.int(0), __ctx.int(3)],
    ]).unwrap();
    let b = Matrix::new(vec![
        vec![__ctx.int(1), __ctx.int(0)],
        vec![__ctx.int(0), __ctx.int(1)],
        vec![__ctx.int(0), __ctx.int(0)],
    ]).unwrap();
    let c = Matrix::new(vec![vec![
        __ctx.int(1),
        __ctx.int(0),
        __ctx.int(0),
    ]]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0), __ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    let cm = ss.controllability_matrix();
    assert_eq!(cm.nrows(), 3, "Controllability matrix should have 3 rows");
    assert_eq!(
        cm.ncols(),
        6,
        "Controllability matrix should have 3*2=6 cols"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 18. Observability matrix size
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn observability_matrix_size() {
    let __ctx = Context::new();
    // 2 states, 2 outputs → observability matrix should be 4×2
    let a = Matrix::new(vec![
        vec![__ctx.int(0), __ctx.int(1)],
        vec![__ctx.int(-2), __ctx.int(-3)],
    ]).unwrap();
    let b = Matrix::new(vec![vec![__ctx.int(0)], vec![__ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![
        vec![__ctx.int(1), __ctx.int(0)],
        vec![__ctx.int(0), __ctx.int(1)],
    ]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0)], vec![__ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    let om = ss.observability_matrix();
    assert_eq!(
        om.nrows(),
        4,
        "Observability matrix should have 2*2=4 rows"
    );
    assert_eq!(om.ncols(), 2, "Observability matrix should have 2 cols");
}

// ═══════════════════════════════════════════════════════════════════════════
// 19. TransferFunction: eval_at specific values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn transfer_function_eval_at() {
    let __ctx = Context::new();
    // G(s) = (s + 3) / (s + 1)
    // G(0) = 3/1 = 3
    // G(1) = 4/2 = 2
    // G(2) = 5/3
    let s = __ctx.symbol("s");
    let tf = TransferFunction::new(&s + 3, &s + 1, s.clone());

    let val0 = tf.eval_at(&__ctx.int(0)).eval_f64().unwrap();
    assert!(
        (val0 - 3.0).abs() < 1e-10,
        "G(0) should be 3, got: {val0}"
    );

    let val1 = tf.eval_at(&__ctx.int(1)).eval_f64().unwrap();
    assert!(
        (val1 - 2.0).abs() < 1e-10,
        "G(1) should be 2, got: {val1}"
    );

    let val2 = tf.eval_at(&__ctx.int(2)).eval_f64().unwrap();
    assert!(
        (val2 - 5.0 / 3.0).abs() < 1e-10,
        "G(2) should be 5/3, got: {val2}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 20. TransferFunction: feedback_with controller
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn transfer_function_feedback_with() {
    let __ctx = Context::new();
    // G(s) = 10/(s+1), H(s) = 2/(s+5)
    // G_cl = G/(1+G*H) = (10*(s+5)) / ((s+1)(s+5) + 10*2)
    //      = 10(s+5) / (s^2+6s+5+20)
    //      = 10(s+5) / (s^2+6s+25)
    // At s=0: 10*5 / (0+0+25) = 50/25 = 2
    let s = __ctx.symbol("s");
    let g = TransferFunction::new(__ctx.int(10), &s + 1, s.clone());
    let h = TransferFunction::new(__ctx.int(2), &s + 5, s.clone());

    let gcl = g.feedback_with(&h);

    let val = gcl.eval_at(&__ctx.int(0)).eval_f64().unwrap();
    assert!(
        (val - 2.0).abs() < 1e-10,
        "Feedback_with DC gain should be 2.0, got: {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 21. TransferFunction: Display format
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn transfer_function_display() {
    let __ctx = Context::new();
    let s = __ctx.symbol("s");
    let tf = TransferFunction::new(__ctx.int(1), &s + 1, s.clone());
    let display = format!("{tf}");
    // Should contain a "/" separator
    assert!(
        display.contains('/'),
        "Display should show fraction: {display}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 22. Routh array: second-order stable
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn routh_array_second_order_stable() {
    let __ctx = Context::new();
    // s^2 + 3s + 2 = (s+1)(s+2)
    // Coefficients: [1, 3, 2]
    // Routh table:
    // Row 0: 1  2
    // Row 1: 3  0
    // Row 2: (3*2 - 1*0)/3 = 2
    // First column: [1, 3, 2] → all positive → stable
    let coeffs = vec![__ctx.int(1), __ctx.int(3), __ctx.int(2)];
    let stability = is_routh_stable(&coeffs);
    assert_eq!(stability, Some(true), "s^2 + 3s + 2 should be Routh-stable");
}

// ═══════════════════════════════════════════════════════════════════════════
// 23. Routh array: single coefficient
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn routh_array_single_coeff() {
    let __ctx = Context::new();
    // Constant polynomial: just [5]
    let coeffs = vec![__ctx.int(5)];
    let table = routh_array(&coeffs);
    assert_eq!(table.len(), 1);
    let stability = is_routh_stable(&coeffs);
    assert_eq!(stability, Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// 24. StateSpace: char_poly evaluated at non-roots is nonzero
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn state_space_char_poly_nonzero_at_non_root() {
    let __ctx = Context::new();
    // A = [[0, 1], [-2, -3]], char poly roots are -1 and -2
    // Evaluate at s=0: should be 0^2 + 3*0 + 2 = 2 (nonzero)
    let a = Matrix::new(vec![
        vec![__ctx.int(0), __ctx.int(1)],
        vec![__ctx.int(-2), __ctx.int(-3)],
    ]).unwrap();
    let b = Matrix::new(vec![vec![__ctx.int(0)], vec![__ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![__ctx.int(1), __ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    let s = __ctx.symbol("s");
    let cp = ss.char_poly(&s);
    let at_zero = cp.subs(&s, &__ctx.int(0)).simplify();
    let val = at_zero.eval_f64().unwrap();
    assert!(
        (val - 2.0).abs() < 1e-10,
        "char_poly(0) should be 2, got: {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 25. StateSpace: 1×1 system
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn state_space_1x1_system() {
    let __ctx = Context::new();
    // Simple first-order system: dx/dt = -2x + u, y = x
    let a = Matrix::new(vec![vec![__ctx.int(-2)]]).unwrap();
    let b = Matrix::new(vec![vec![__ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![__ctx.int(1)]]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    assert_eq!(ss.num_states(), 1);
    assert_eq!(ss.num_inputs(), 1);
    assert_eq!(ss.num_outputs(), 1);
    assert!(ss.is_controllable());
    assert!(ss.is_observable());

    let s = __ctx.symbol("s");
    let poles = ss.poles(&s);
    assert_eq!(poles.len(), 1);
    assert_eq!(format!("{}", poles[0]), "-2");
}

// ═══════════════════════════════════════════════════════════════════════════
// 26. TransferFunction: DC gain of constant TF
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn transfer_function_constant_dc_gain() {
    let __ctx = Context::new();
    // G(s) = 5/1 — constant gain
    let s = __ctx.symbol("s");
    let tf = TransferFunction::new(__ctx.int(5), __ctx.int(1), s.clone());

    let gain = tf.dc_gain().eval_f64().unwrap();
    assert!(
        (gain - 5.0).abs() < 1e-10,
        "Constant TF DC gain should be 5, got: {gain}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 27. StateSpace: Display formatting
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn state_space_display() {
    let __ctx = Context::new();
    let a = Matrix::new(vec![
        vec![__ctx.int(0), __ctx.int(1)],
        vec![__ctx.int(-2), __ctx.int(-3)],
    ]).unwrap();
    let b = Matrix::new(vec![vec![__ctx.int(0)], vec![__ctx.int(1)]]).unwrap();
    let c = Matrix::new(vec![vec![__ctx.int(1), __ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    let display = format!("{ss}");
    assert!(
        display.contains("StateSpace"),
        "Display should include 'StateSpace': {display}"
    );
    assert!(
        display.contains("n=2"),
        "Display should include state count: {display}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 28. Routh: first-order polynomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn routh_first_order() {
    let __ctx = Context::new();
    // s + 3 → coeffs [1, 3]
    // Table: row0=[1], row1=[3]
    // First col: [1, 3] → stable
    let coeffs = vec![__ctx.int(1), __ctx.int(3)];
    let stability = is_routh_stable(&coeffs);
    assert_eq!(stability, Some(true), "s + 3 should be stable");
}

// ═══════════════════════════════════════════════════════════════════════════
// 29. StateSpace: not observable
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn state_space_not_observable() {
    let __ctx = Context::new();
    // A = [[1, 0], [0, 2]], C = [[1, 0]]
    // Observability matrix: [C; CA] = [[1, 0], [1, 0]] → rank 1 ≠ 2 → not observable
    let a = Matrix::new(vec![
        vec![__ctx.int(1), __ctx.int(0)],
        vec![__ctx.int(0), __ctx.int(2)],
    ]).unwrap();
    let b = Matrix::new(vec![vec![__ctx.int(1)], vec![__ctx.int(0)]]).unwrap();
    let c = Matrix::new(vec![vec![__ctx.int(1), __ctx.int(0)]]).unwrap();
    let d = Matrix::new(vec![vec![__ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d);

    assert!(
        !ss.is_observable(),
        "System should NOT be observable"
    );
}
