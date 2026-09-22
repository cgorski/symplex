//! Shared input decoding for the symbolic fuzz targets (`#[path]`-included
//! by each; not a target itself).
//!
//! [`expr`] turns bytes into an elementary expression in one symbol, depth
//! ≤ 4, over `+ − × ÷`, integer powers `−3..=3`, `sqrt`, `sin`, `cos`,
//! `tan`, `exp`, `ln`, `atan`, `abs` and small rational constants.  Rational
//! powers other than `sqrt` are left out on purpose: the three evaluators
//! disagree on odd roots of negative numbers (a known, separately tracked
//! issue), and every mismatch would be that one.
//!
//! [`close_at`] compares two expressions numerically at a few exact
//! rational points, at 30 significant digits so cancellation in one form
//! does not masquerade as a disagreement.  Points where either side is not a
//! finite real number are skipped.
#![allow(dead_code)]

use symplex::prelude::*;

pub struct Bytes<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Bytes<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Bytes { data, pos: 0 }
    }
    pub fn u8(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }
    pub fn exhausted(&self) -> bool {
        self.pos >= self.data.len()
    }
}

fn leaf(ctx: &Context, x: &Ex, b: &mut Bytes) -> Ex {
    match b.u8() % 6 {
        0..=2 => x.clone(),
        3 => ctx.int(i64::from(b.u8() % 7) - 3),
        _ => {
            let n = i64::from(b.u8() % 9) - 4;
            let d = i64::from(b.u8() % 4) + 1;
            ctx.rational(n, d)
        }
    }
}

/// An elementary expression in `x` decoded from `b`.
pub fn expr(ctx: &Context, x: &Ex, b: &mut Bytes, depth: u32) -> Ex {
    if depth == 0 || b.exhausted() {
        return leaf(ctx, x, b);
    }
    let op = b.u8() % 18;
    let d = depth - 1;
    match op {
        0 | 1 => leaf(ctx, x, b),
        2 | 3 => expr(ctx, x, b, d) + expr(ctx, x, b, d),
        4 => expr(ctx, x, b, d) - expr(ctx, x, b, d),
        5 | 6 => expr(ctx, x, b, d) * expr(ctx, x, b, d),
        7 => expr(ctx, x, b, d) / expr(ctx, x, b, d),
        8 => expr(ctx, x, b, d).powi(i64::from(b.u8() % 7) - 3),
        9 => expr(ctx, x, b, d).sqrt(),
        10 => expr(ctx, x, b, d).sin(),
        11 => expr(ctx, x, b, d).cos(),
        12 => expr(ctx, x, b, d).tan(),
        13 => expr(ctx, x, b, d).exp(),
        14 => expr(ctx, x, b, d).ln(),
        15 => expr(ctx, x, b, d).atan(),
        16 => expr(ctx, x, b, d).abs(),
        _ => {
            let c = leaf(ctx, x, b);
            c * expr(ctx, x, b, d)
        }
    }
}

/// `e` at `x = v` as an `f64`, through a 30-digit evaluation; `None` when
/// it is not a finite real number (or does not evaluate).
pub fn value_at(e: &Ex, x: &Ex, v: &Ex) -> Option<f64> {
    let s = e.subs(x, v).eval_decimal(30).ok()?;
    let f: f64 = s.parse().ok()?;
    f.is_finite().then_some(f)
}

/// The sample points: exact rationals on both sides of 0, away from the
/// usual special values.
pub fn points(ctx: &Context) -> [Ex; 4] {
    [
        ctx.rational(1, 3),
        ctx.rational(7, 5),
        ctx.rational(-5, 7),
        ctx.rational(13, 4),
    ]
}

/// Do `a` and `b` agree at every sample point where both are finite reals?
/// Returns the first disagreement `(point, a, b)`.
pub fn close_at(ctx: &Context, a: &Ex, b: &Ex, x: &Ex) -> Option<(Ex, f64, f64)> {
    for v in points(ctx) {
        if let (Some(fa), Some(fb)) = (value_at(a, x, &v), value_at(b, x, &v)) {
            let scale = fa.abs().max(fb.abs()).max(1.0);
            if (fa - fb).abs() > 1e-9 * scale {
                return Some((v, fa, fb));
            }
        }
    }
    None
}
