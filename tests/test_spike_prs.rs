//! Spike: verify GenPoly<RationalFn> resultant gives the right answer for x³-1
use symplex::prelude::*;

#[test]
fn resultant_x3_minus_1_via_apart_and_integrate() {
    // Instead of testing at the Poly level (which is pub(crate)),
    // verify end-to-end that the pieces work correctly.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    
    // The apart decomposition of 1/(x³-1) should produce
    // 1/3·1/(x-1) + linear/quadratic
    let expr = 1 / (&x.powi(3) - 1);
    let apart = expr.partial_fractions(&x);
    
    // Verify apart is numerically correct at x=2
    let orig_2 = expr.eval_f64_with(&[(&x, 2)]).unwrap();
    let apart_2 = apart.eval_f64_with(&[(&x, 2)]).unwrap();
    assert!((orig_2 - apart_2).abs() < 1e-14, "apart should be numerically exact");
    
    // Verify apart integration is numerically correct
    let anti = apart.integrate(&x);
    assert!(!anti.has_unevaluated(), "apart integration should fully evaluate");
    
    let f3 = anti.eval_f64_with(&[(&x, 3)]).unwrap();
    let f2 = anti.eval_f64_with(&[(&x, 2)]).unwrap();
    let integral = f3 - f2;
    
    // Ground truth: ∫₂³ 1/(x³-1) dx ≈ 0.07539
    assert!((integral - 0.07539).abs() < 0.001, 
        "∫₂³ 1/(x³-1) dx ≈ 0.07539, got {integral}");
}
