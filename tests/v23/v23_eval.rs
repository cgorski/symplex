//! 0.24 — exact evaluation keeps branch-cut values exact.
//!
//! `fuzz_simplify` found `√((e^{ln(−2)})^{−3})` evaluating to `−(√2/4)·i` at
//! 30 digits (and `+(√2/4)·i` at 50): `eval` rewrote the inner power to
//! `exp(−3·ln 2 − 3πi)`, and the numeric exponential of that carried a
//! rounding residue in its imaginary part whose sign chose the branch of the
//! square root.  `eval` now folds `exp(ln w) = w` and splits off an
//! imaginary π-multiple (`exp(w + ikπ) = ±exp(w)`, `±i·exp(w)`).
//! mpmath (mp.dps = 50): sqrt(mpc(-1/8, 0)) = 0.35355339059327376220042218105242j.

use symplex::prelude::*;

#[test]
fn exp_of_ln_folds() {
    // SymPy: exp(log(x)) = x (at construction).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.ln().exp().eval(), x);
    assert_eq!(ctx.int(-2).ln().exp().eval(), ctx.int(-2));
}

#[test]
fn exp_with_an_imaginary_pi_multiple_folds_its_unit() {
    // exp(ln 2 − 3πi) = 2·e^{−3πi} = −2.
    let ctx = Context::new();
    let e = (ctx.int(2).ln() - ctx.pi() * ctx.i_unit() * 3).exp().eval();
    assert_eq!(e.eval(), ctx.int(-2));
}

#[test]
fn square_root_of_a_negative_through_exp_ln_is_principal_at_every_precision() {
    let ctx = Context::new();
    let f = ctx.int(-2).ln().exp().powi(-3).sqrt();
    for digits in [16, 30, 50] {
        let s = f.eval_decimal(digits).unwrap();
        assert!(s.starts_with("0.353553390593273"), "{digits} digits: {s}");
        assert!(
            s.ends_with("*i") && !s.starts_with('-'),
            "{digits} digits: {s}"
        );
    }
}
