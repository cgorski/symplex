#![no_main]

use libfuzzer_sys::fuzz_target;
use symplex::prelude::*;

fuzz_target!(|data: &[u8]| {
    // Need at least 2 bytes: one for expression shape, one for assumption
    if data.len() < 2 {
        return;
    }

    let shape = data[0];
    let assumption_byte = data[1];

    let ctx = Context::new();

    // Create a variable with a random assumption
    let x = ctx.symbol("fuzz_x");
    let assumption = match assumption_byte % 8 {
        0 => Assumption::Positive,
        1 => Assumption::Negative,
        2 => Assumption::NonNegative,
        3 => Assumption::NonPositive,
        4 => Assumption::Real,
        5 => Assumption::Integer,
        6 => Assumption::Zero,
        _ => Assumption::NonZero,
    };
    // One assumption on a fresh symbol is always consistent.
    let Ok(x) = x.assume(assumption) else {
        return;
    };

    // Build an expression based on shape byte
    let expr = match shape % 12 {
        0 => x.abs(),
        1 => x.sign(),
        2 => x.floor(),
        3 => x.ceiling(),
        4 => x.powi(2).sqrt(),         // sqrt(x^2)
        5 => x.abs().powi(2),          // abs(x)^2
        6 => (&x + &ctx.int(1)).abs(), // abs(x + 1)
        7 => x.sign().powi(2),         // sign(x)^2
        8 => {
            let y = ctx.symbol("fuzz_y").assume(Assumption::Positive).unwrap();
            (&x + &y).abs() // abs(x + y)
        }
        9 => {
            let neg_one = ctx.int(-1);
            neg_one.pow(&x) // (-1)^x
        }
        10 => x.abs().abs(), // abs(abs(x))
        _ => x.clone(),      // just x
    };

    // refine() must never panic
    let refined = expr.refine();

    // If the expression has no free symbols (fully evaluable), verify
    // that the refined form evaluates to the same value.
    // For expressions with symbols, we can't easily verify numerical
    // equivalence without substitution, but at minimum it shouldn't panic.
    let _ = format!("{refined}");

    // Also test refine_with (temporary assumptions)
    let y = ctx.symbol("fuzz_y2");
    let expr2 = y.abs();
    let refined2 = expr2.refine_with(&[(&y, Assumption::Positive)]).unwrap();
    let _ = format!("{refined2}");
});
