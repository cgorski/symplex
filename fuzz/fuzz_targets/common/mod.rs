//! Shared input decoding for the symbolic fuzz targets (`#[path]`-included
//! by each; not a target itself).
//!
//! [`expr`] turns bytes into an elementary expression in one symbol, depth
//! ≤ 4, over `+ − × ÷`, integer powers `−3..=3`, `sqrt`, `sin`, `cos`,
//! `tan`, `exp`, `ln`, `atan`, `abs` and small rational constants
//! ([`Grammar::Elementary`], the `fuzz_integrate` / `fuzz_poly` trees).
//! [`expr_in`] with [`Grammar::Full`] adds the functions whose principal
//! branch a rewrite can get wrong: `sinh`, `cosh`, `tanh`, their inverses,
//! `asin`, `acos` and rational powers `x^(p/q)` with `q ≤ 5`.
//!
//! Two comparisons:
//!
//! * [`close_at`] — finite *real* values at real points (30 significant
//!   digits, so cancellation in one form does not masquerade as a
//!   disagreement); a point where either side is not a finite real is
//!   skipped.  `fuzz_integrate` uses it: the integrator's real-variable
//!   convention (`∫ dx/x = ln|x|`) only promises `F′ = f` on ℝ.
//! * [`close_at_complex`] — complex values at real *and* complex points.
//!   symplex is a CAS over ℂ (a symbol without assumptions may be complex,
//!   principal branch throughout), so a rewrite must preserve the value
//!   everywhere, including where it is not real: `ln x + ln y → ln(x·y)` is
//!   wrong at `x = y = −1`, `√(x²) → |x|` at `x = i`.  The complex points sit
//!   in the second quadrant, just below the negative real axis (the branch
//!   cut of `ln` and `√`) and above `Im x = π` (where `ln(eˣ) ≠ x`).
//!   A point is skipped where either side fails to evaluate, or is not
//!   continuous there (its value moves under a `10⁻¹²` nudge in the real or
//!   the imaginary direction): an expression whose exact value lies *on* a
//!   branch cut — `acosh(−x + e^{ln x})` is `acosh(0)`, on the cut of
//!   `acosh` — is evaluated with rounding noise that picks a side, and a
//!   wrong rewrite differs in a whole neighbourhood anyway.
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

/// Which functions [`expr_in`] may use.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Grammar {
    /// `+ − × ÷`, integer powers, `sqrt`, `sin`, `cos`, `tan`, `exp`, `ln`,
    /// `atan`, `abs`.  The byte → expression map is fixed: a corpus built
    /// with it keeps decoding to the same expressions.
    Elementary,
    /// Elementary plus `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`,
    /// `asin`, `acos` and rational powers `x^(p/q)`, `q ∈ 2..=5`.
    Full,
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

/// An elementary expression in `x` decoded from `b` ([`Grammar::Elementary`]).
pub fn expr(ctx: &Context, x: &Ex, b: &mut Bytes, depth: u32) -> Ex {
    expr_in(ctx, x, b, depth, Grammar::Elementary)
}

/// An expression in `x` decoded from `b`, over the functions of `g`.
pub fn expr_in(ctx: &Context, x: &Ex, b: &mut Bytes, depth: u32, g: Grammar) -> Ex {
    if depth == 0 || b.exhausted() {
        return leaf(ctx, x, b);
    }
    let ops = match g {
        Grammar::Elementary => 18,
        Grammar::Full => 28,
    };
    let op = b.u8() % ops;
    let d = depth - 1;
    let sub = |b: &mut Bytes| expr_in(ctx, x, b, d, g);
    match op {
        0 | 1 => leaf(ctx, x, b),
        2 | 3 => sub(b) + sub(b),
        4 => sub(b) - sub(b),
        5 | 6 => sub(b) * sub(b),
        7 => sub(b) / sub(b),
        8 => sub(b).powi(i64::from(b.u8() % 7) - 3),
        9 => sub(b).sqrt(),
        10 => sub(b).sin(),
        11 => sub(b).cos(),
        12 => sub(b).tan(),
        13 => sub(b).exp(),
        14 => sub(b).ln(),
        15 => sub(b).atan(),
        16 => sub(b).abs(),
        17 => {
            let c = leaf(ctx, x, b);
            c * sub(b)
        }
        // Grammar::Full only.
        18 => sub(b).sinh(),
        19 => sub(b).cosh(),
        20 => sub(b).tanh(),
        21 => sub(b).asinh(),
        22 => sub(b).acosh(),
        23 => sub(b).atanh(),
        24 => sub(b).asin(),
        25 => sub(b).acos(),
        _ => {
            let base = sub(b);
            let q = i64::from(b.u8() % 4) + 2;
            let p = match i64::from(b.u8() % 7) - 3 {
                0 => 1,
                p => p,
            };
            base.pow(&ctx.rational(p, q))
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

/// `e` at `x = v` as a complex number, through a 30-digit evaluation;
/// `None` when it does not evaluate or a part is not finite.
pub fn complex_value_at(e: &Ex, x: &Ex, v: &Ex) -> Option<Complex64> {
    parse_complex(&e.subs(x, v).eval_decimal(30).ok()?)
}

/// Parse what `eval_decimal` prints for a complex value: `a`, `b*i`, `i`,
/// `-i`, `a + b*i`, `a - b*i`, `a + i`, `a - i` (the parts may be in
/// scientific notation, `1.5e-40`, whose `-` has no spaces around it).
fn parse_complex(s: &str) -> Option<Complex64> {
    let imag = |t: &str| -> Option<f64> {
        match t {
            "i" => Some(1.0),
            "-i" => Some(-1.0),
            _ => t.strip_suffix("*i")?.parse().ok(),
        }
    };
    let z = if let Some((re, im)) = s.split_once(" + ") {
        Complex64::new(re.parse().ok()?, imag(im)?)
    } else if let Some((re, im)) = s.split_once(" - ") {
        Complex64::new(re.parse().ok()?, -imag(im)?)
    } else if s.ends_with('i') {
        Complex64::new(0.0, imag(s)?)
    } else {
        Complex64::new(s.parse().ok()?, 0.0)
    };
    (z.re.is_finite() && z.im.is_finite()).then_some(z)
}

/// The real sample points: exact rationals on both sides of 0, away from
/// the usual special values.
pub fn points(ctx: &Context) -> [Ex; 4] {
    [
        ctx.rational(1, 3),
        ctx.rational(7, 5),
        ctx.rational(-5, 7),
        ctx.rational(13, 4),
    ]
}

/// The complex sample points `a + b·i`: first quadrant, second quadrant
/// (doubling its argument crosses the negative real axis), just below the
/// negative real axis, and `Im = 4 > π`.
pub fn complex_points(ctx: &Context) -> [Ex; 4] {
    let z = |a: (i64, i64), b: (i64, i64)| {
        ctx.rational(a.0, a.1) + ctx.i_unit() * ctx.rational(b.0, b.1)
    };
    [z((1, 2), (3, 2)), z((-3, 4), (5, 4)), z((-2, 1), (-1, 3)), z((1, 3), (4, 1))]
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

/// `e` at `x = v` if it evaluates there and is continuous at `v`: the
/// values at `v + δ` and `v + iδ` (`δ = 10⁻¹²`) are within `10⁻⁶` of it,
/// relative.  `None` at a pole, on a branch cut, or where it does not
/// evaluate.
pub fn continuous_value_at(ctx: &Context, e: &Ex, x: &Ex, v: &Ex) -> Option<Complex64> {
    let z = complex_value_at(e, x, v)?;
    let delta = ctx.rational(1, 1_000_000_000_000);
    for nudge in [delta.clone(), ctx.i_unit() * &delta] {
        let w = complex_value_at(e, x, &(v + &nudge))?;
        if (w - z).norm() > 1e-6 * z.norm().max(1.0) {
            return None;
        }
    }
    Some(z)
}

/// Do `a` and `b` have the same complex value at every real and complex
/// sample point where both evaluate and are continuous (see
/// [`continuous_value_at`])?  Returns the first disagreement `(point, a, b)`.
pub fn close_at_complex(
    ctx: &Context,
    a: &Ex,
    b: &Ex,
    x: &Ex,
) -> Option<(Ex, Complex64, Complex64)> {
    for v in points(ctx).into_iter().chain(complex_points(ctx)) {
        if let (Some(za), Some(zb)) = (
            continuous_value_at(ctx, a, x, &v),
            continuous_value_at(ctx, b, x, &v),
        ) {
            let scale = za.norm().max(zb.norm()).max(1.0);
            if (za - zb).norm() > 1e-9 * scale {
                return Some((v, za, zb));
            }
        }
    }
    None
}
