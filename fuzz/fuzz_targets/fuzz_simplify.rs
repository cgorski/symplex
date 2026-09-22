//! Value preservation of the rewriting transforms.
//!
//! For an elementary expression `f` (see `common::expr`), each of
//! `simplify`, `expand`, `factor`, `together`, `cancel`, `ratsimp` and
//! `simplify_trig` must return an expression with the same value at every
//! sample point where both are finite reals (30-digit evaluation, relative
//! 1e-9), and `simplify` must be idempotent.  The 0.21 audit found the
//! `pow_pow` collapse (`(x²)^(3/2) → x³`) exactly this way.
#![no_main]

#[path = "common/mod.rs"]
mod common;

use libfuzzer_sys::fuzz_target;
use symplex::prelude::*;

/// A named rewriting transform `(f, x) ↦ g`.
type Transform = (&'static str, fn(&Ex, &Ex) -> Ex);

fuzz_target!(|data: &[u8]| {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut b = common::Bytes::new(data);
    let f = common::expr(&ctx, &x, &mut b, 4);
    let transforms: [Transform; 7] = [
        ("simplify", |e, _| e.simplify()),
        ("expand", |e, _| e.expand()),
        ("factor", |e, x| e.factor(x)),
        ("together", |e, _| e.together()),
        ("cancel", |e, x| e.cancel(x)),
        ("ratsimp", |e, _| e.ratsimp()),
        ("simplify_trig", |e, _| e.simplify_trig()),
    ];
    let which = b.u8() as usize % transforms.len();
    let (name, t) = transforms[which];
    let g = t(&f, &x);
    if let Some((v, fa, fb)) = common::close_at(&ctx, &f, &g, &x) {
        panic!("{name} changed the value: f = {f}, {name}(f) = {g}, at x = {v}: {fa:e} vs {fb:e}");
    }
    if name == "simplify" {
        let gg = g.simplify();
        assert_eq!(gg, g, "simplify is not idempotent on f = {f}: {g} → {gg}");
    }
});
