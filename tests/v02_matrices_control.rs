//! symplex 0.2 — control-systems additions: `StateSpace::poles()` without a
//! dummy variable, `TransferFunction::to_state_space` and
//! `StateSpace::to_transfer_function` round trips.

use symplex::prelude::*;

#[test]
fn poles_without_dummy_variable() {
    let ctx = Context::new();
    let ss = StateSpace::new(
        matrix![ctx, [0, 1], [-6, -5]],
        matrix![ctx, [0], [1]],
        matrix![ctx, [1, 0]],
        matrix![ctx, [0]],
    );
    let mut poles: Vec<String> = ss.poles().iter().map(|p| p.to_string()).collect();
    poles.sort();
    assert_eq!(poles, ["-2", "-3"]);
    assert_eq!(ss.is_stable(), Some(true));
    let unstable = StateSpace::new(
        matrix![ctx, [1, 1], [0, 1]],
        matrix![ctx, [0], [1]],
        matrix![ctx, [1, 0]],
        matrix![ctx, [0]],
    );
    // repeated pole 1 (multiplicity 2)
    assert_eq!(unstable.poles().len(), 2);
    assert_eq!(unstable.is_stable(), Some(false));
}

#[test]
fn tf_to_state_space_controllable_canonical_form() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    // G(s) = (2s + 3) / (s³ + 4s² + 5s + 6)
    let g = TransferFunction::from_coeffs(&[3, 2], &[6, 5, 4, 1], &s);
    let ss = g.to_state_space().unwrap();
    assert_eq!(ss.a, matrix![ctx, [0, 1, 0], [0, 0, 1], [-6, -5, -4]]);
    assert_eq!(ss.b, matrix![ctx, [0], [0], [1]]);
    assert_eq!(ss.c, matrix![ctx, [3, 2, 0]]);
    assert_eq!(ss.d, matrix![ctx, [0]]);
    assert!(ss.is_controllable());
    // Poles of the realisation are the roots of the denominator
    assert_eq!(ss.char_poly(&s).expand(), g.den.expand());
    // Round trip back to the transfer function
    let back = ss.to_transfer_function(&s).unwrap();
    assert_eq!(back.num, g.num.expand());
    assert_eq!(back.den, g.den.expand());
}

#[test]
fn tf_with_direct_feedthrough_and_non_monic_denominator() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    // G(s) = (2s² + s + 1) / (2s² + 4s + 6)  →  normalise: (s² + s/2 + 1/2)/(s² + 2s + 3)
    let g = TransferFunction::from_coeffs(&[1, 1, 2], &[6, 4, 2], &s);
    let ss = g.to_state_space().unwrap();
    assert_eq!(ss.d, Matrix::new(vec![vec![ctx.int(1)]]).unwrap());
    assert_eq!(ss.a, matrix![ctx, [0, 1], [-3, -2]]);
    // C = [b0 − a0 b2, b1 − a1 b2] = [1/2 − 3, 1/2 − 2] = [−5/2, −3/2]
    assert_eq!(
        ss.c,
        Matrix::new(vec![vec![ctx.rational(-5, 2), ctx.rational(-3, 2)]]).unwrap()
    );
    let back = ss.to_transfer_function(&s).unwrap();
    // Same rational function: cross-multiply
    let lhs = (&back.num * &g.den).expand();
    let rhs = (&g.num * &back.den).expand();
    assert!(
        (&lhs - &rhs).expand().is_zero_structural(),
        "{lhs} vs {rhs}"
    );
}

#[test]
fn tf_to_state_space_symbolic_coefficients() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    let (k, wn, zeta) = (
        ctx.symbol("K"),
        ctx.symbol_with("omega_n", &[Assumption::Positive]),
        ctx.symbol("zeta"),
    );
    // Standard second-order system K ωn² / (s² + 2ζωn s + ωn²)
    let num = &k * &wn.powi(2);
    let den = &(&s.powi(2) + &(&(&(&zeta * 2) * &wn) * &s)) + &wn.powi(2);
    let g = TransferFunction::new(num.clone(), den.clone(), s.clone());
    let ss = g.to_state_space().unwrap();
    assert_eq!(ss.num_states(), 2);
    assert_eq!(ss.a.get(1, 0), &(-&wn.powi(2)));
    assert_eq!(ss.a.get(1, 1), &(-&(&(&zeta * 2) * &wn)));
    assert_eq!(ss.c.get(0, 0), &num);
    let back = ss.to_transfer_function(&s).unwrap();
    assert!((&back.den - &den).expand().is_zero_structural());
    assert!((&back.num - &num).expand().is_zero_structural());
}

#[test]
fn tf_to_state_space_errors() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    // improper
    let improper = TransferFunction::from_coeffs(&[0, 0, 1], &[1, 1], &s);
    assert!(matches!(
        improper.to_state_space(),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // constant denominator
    let constant = TransferFunction::from_coeffs(&[1], &[2], &s);
    assert!(constant.to_state_space().is_err());
    // non-polynomial
    let bad = TransferFunction::new(ctx.int(1), s.sin(), s.clone());
    assert!(bad.to_state_space().is_err());
    // MIMO → to_transfer_function errors
    let mimo = StateSpace::new(
        matrix![ctx, [0, 1], [-2, -3]],
        matrix![ctx, [1, 0], [0, 1]],
        matrix![ctx, [1, 0]],
        matrix![ctx, [0, 0]],
    );
    assert!(matches!(
        mimo.to_transfer_function(&s),
        Err(SymplexError::InvalidArgument { .. })
    ));
}
