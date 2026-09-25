//! 0.25 — panics reachable from user input: an empty symbol name has a
//! `Result`-returning path everywhere a name can come from outside, and a
//! panicking `refine_with` no longer leaves temporary assumptions behind.

use symplex::prelude::*;
use symplex::stats::{Distribution, RandomVariable};

#[test]
fn empty_symbol_names_are_errors_on_the_fallible_paths() {
    let ctx = Context::new();
    assert!(ctx.try_symbol("").is_err());
    assert_eq!(ctx.try_symbol("x").unwrap(), ctx.symbol("x"));
    let normal = Distribution::normal(ctx.int(0), ctx.int(1));
    assert!(RandomVariable::try_new(&ctx, "", normal.clone()).is_err());
    assert!(RandomVariable::try_new(&ctx, "X", normal).is_ok());
}

#[test]
fn a_contradiction_in_refine_with_leaves_the_context_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol_with("y", &[Assumption::Positive]).unwrap();
    let e = &x.abs() + &y.abs();
    // The hypothesis on x is fine; the one on y contradicts its stored
    // assumption.  Before 0.25 the panic left x positive; since 0.29 it is
    // an error, reported before anything is applied.
    assert!(matches!(
        e.refine_with(&[(&x, Assumption::Positive), (&y, Assumption::Negative)]),
        Err(SymplexError::ContradictoryAssumptions { .. })
    ));
    assert_eq!(x.is_positive(), None);
    assert_eq!(y.is_positive(), Some(true));
    // And the context still works normally.
    assert_eq!(
        e.refine_with(&[(&x, Assumption::Positive)]).unwrap(),
        &x + &y
    );
    assert_eq!(x.is_positive(), None);
}
