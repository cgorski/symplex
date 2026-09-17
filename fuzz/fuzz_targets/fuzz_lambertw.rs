#![no_main]

use libfuzzer_sys::fuzz_target;
use symplex::prelude::*;

fuzz_target!(|data: &[u8]| {
    // Need at least 8 bytes to construct an f64
    if data.len() < 8 {
        return;
    }

    // Construct an f64 from the first 8 bytes
    let raw_bytes: [u8; 8] = data[..8].try_into().unwrap();
    let raw_f64 = f64::from_le_bytes(raw_bytes);

    // Skip NaN, infinity, and values outside the domain of the principal branch.
    // W(x) is defined for x >= -1/e on the principal branch.
    if raw_f64.is_nan() || raw_f64.is_infinite() || raw_f64.abs() > 1e15 {
        return;
    }

    // Map to a reasonable range: [-0.36, 100].
    // -1/e ≈ -0.3679, so -0.36 is safely inside the domain.
    let x_f64 = raw_f64.rem_euclid(100.36) - 0.36;

    let ctx = Context::new();

    // Build LambertW(x) as a symplex expression
    let x_val = ctx.rational((x_f64 * 1000.0).round() as i64, 1000);
    let w_expr = x_val.lambertw();

    // Attempt numerical evaluation — must never panic
    let w_f64 = match w_expr.eval_f64() {
        Ok(v) => v,
        Err(_) => return, // Evaluation might fail for edge-case inputs — that's fine
    };

    // Skip if W(x) evaluated to NaN or infinity
    if w_f64.is_nan() || w_f64.is_infinite() {
        return;
    }

    // Verify the defining property: W(x) · e^{W(x)} = x
    // This is the fundamental identity that must hold for any correct
    // LambertW implementation.
    let lhs = w_f64 * w_f64.exp();
    let x_approx = (x_f64 * 1000.0).round() / 1000.0; // match the rational we constructed

    let diff = (lhs - x_approx).abs();
    let scale = x_approx.abs().max(1.0);

    // Relative tolerance: W·e^W should equal x to reasonable precision.
    // We use a generous tolerance because we're going through rational
    // approximation and f64 eval, not arbitrary-precision.
    assert!(
        diff / scale < 1e-6,
        "LambertW identity violated: W({x_approx}) = {w_f64}, \
         but W(x)·exp(W(x)) = {lhs}, expected {x_approx}, diff = {diff}"
    );

    // Also verify Display doesn't panic
    let _ = format!("{w_expr}");

    // Also verify LaTeX doesn't panic
    let _ = w_expr.to_latex();

    // Also verify pretty printing doesn't panic
    let _ = w_expr.pretty();
    let _ = w_expr.pretty_ascii();

    // Test differentiation doesn't panic
    let x_sym = ctx.symbol("fuzz_lw_x");
    let w_sym = x_sym.lambertw();
    let _ = w_sym.diff(&x_sym);
});
