//! SymPy oracle, 0.2 surface — integral transforms: Laplace, Fourier
//! (SymPy's `fourier_transform` uses `e^{-2πikx}`, i.e. symplex's
//! `FourierConvention::Ordinary`), and Mellin.  Transformed expressions are
//! compared numerically at several frequency points.

use super::v02_oracle_common;

use symplex::prelude::*;
use v02_oracle_common::*;

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
