//! SymPy oracle, 0.2 surface — integral transforms: Laplace, Fourier
//! (SymPy's `fourier_transform` uses `e^{-2πikx}`, i.e. symplex's
//! `FourierConvention::Ordinary`), and Mellin.  Transformed expressions are
//! compared numerically at several frequency points.

use super::v02_oracle_common;

use symplex::prelude::*;
use v02_oracle_common::*;

/// Library bugs surfaced by this file (strict xfail — see common module).
///
/// Oracle values SymPy cannot give are corrected in the fixture instead,
/// with an `oracle_correction` field (re-applied by the generator):
/// - `mellin: 1/(1 + x)**3` and `1/(1 + x)**2` at `s = 1` — SymPy 1.14's
///   closed forms `pi*(s - 2)*(s - 1)/(2*sin(pi*s))` and `pi*(1 - s)/sin(pi*s)`
///   are `0/0` there, inside the strips `(0, 3)` and `(0, 2)`, and the points
///   were stored as `nan` and skipped (0.46 and before); the transforms are
///   `Γ(1)Γ(2)/Γ(3) = 1/2` and `Γ(1)Γ(1)/Γ(2) = 1` (mpmath 1.3.0 `quad` of
///   `1/(1 + x)³` and `1/(1 + x)²` over `[0, ∞)`), and symplex's `B(s, 3 − s)`,
///   `B(s, 2 − s)` give them.
const KNOWN_BUGS: &[KnownBug] = &[];

#[test]
fn laplace_transform_forward() {
    run_with_known_bugs("laplace", "forward", KNOWN_BUGS, |ctx, fx| {
        let f = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let t = ctx.symbol(fx.str("time_var").unwrap_or("t"));
        let s = ctx.symbol(fx.str("freq_var").unwrap_or("s"));
        match f.try_laplace(&t, &s) {
            Ok(r) => compare_eval_points(ctx, &r, fx, "eval_points", TOLERANCE),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

#[test]
fn fourier_transform_ordinary_convention() {
    run_with_known_bugs("fourier", "ordinary", KNOWN_BUGS, |ctx, fx| {
        let f = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let x = ctx.symbol(fx.str("time_var").unwrap_or("x"));
        let k = ctx.symbol(fx.str("freq_var").unwrap_or("k"));
        match f.fourier_transform_with(&x, &k, FourierConvention::Ordinary) {
            Ok(r) => compare_eval_points(ctx, &r, fx, "eval_points", TOLERANCE),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

#[test]
fn mellin_transform_forward() {
    run_with_known_bugs("mellin", "forward", KNOWN_BUGS, |ctx, fx| {
        let f = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let x = ctx.symbol(fx.str("space_var").unwrap_or("x"));
        let s = ctx.symbol(fx.str("freq_var").unwrap_or("s"));
        match f.mellin_transform(&x, &s) {
            Ok((r, _strip)) => compare_eval_points(ctx, &r, fx, "eval_points", TOLERANCE),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}
