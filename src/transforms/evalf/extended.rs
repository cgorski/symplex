//! Values below the exponent range of `BigFloat`.
//!
//! astro-float's exponent is an `i32`: a number below about `2^(−2.1·10⁹)`
//! comes out 0 — `exp(−10¹⁰) ≈ 2^(−1.44·10¹⁰)`, `erfc(10⁵)`, `Ei(−10¹⁰)`.
//! Such a zero carries an underflow bound (`accuracy::finish`), and before
//! 0.34 that was all that was known of it: a sum of such numbers whose signs
//! differ was `0 ± 2^(−2.1·10⁹)` and printed `0` — `exp(−10¹⁰) −
//! exp(−2·10¹⁰)` (truly `9.28·10^(−4342944820)`), `Ei(−10¹⁰) + exp(−10¹⁰)`,
//! and `log(1 + exp(−10¹⁰))`.
//!
//! mpmath's exponents are unbounded (an `mpf` is `(sign, man, exp, bc)` with
//! Python integers, `libmp/libmpf.py`), and SymPy's `add_terms`
//! (`core/evalf.py`) adds the terms of a sum as integer mantissas at
//! integer exponents, dropping a term only when it lies more than the
//! working precision below the partial sum.  Here a value below the range is
//! held beside the ordinary placeholder `0 ± underflow` as a mantissa in
//! range and a binary scale, `m·2^k` with `|m| ∈ [½, 1)` and an error bound
//! on `m` ([`Scaled`]), for the nodes that can propagate it:
//!
//! | node | from a value below the range (`s = m·2^k`) | from ordinary arguments whose value underflowed |
//! |---|---|---|
//! | `exp z` | `1 + s + s²/2 + …` ([`Ext::Near`]) | `e^r·2^k`, `k = round(re z/ln 2)`, `r = z − k·ln 2` |
//! | `x·y·…` | the mantissas multiplied, the scales added | the same |
//! | `b^q` | `mⁿ·2^(nk)` for an integer `abs(n) ≤ 1024`, else `exp(q·(Log m + k·ln 2))`; `(1 + s)^q` by the binomial series | `exp(q·Log b)` |
//! | `a + b + …` | SymPy's `add_terms`: the largest terms at a common scale, the terms more than `prec + 64` bits below them in the error, an exact cancellation of the largest moving on to the next | — |
//! | `ln z` | `Log m + k·ln 2`, in range; `ln(1 + s) = s − s²/2 + …` | — |
//! | `sin`, `tan`, `asin`, `atan` (and hyperbolic) | their series, `s ∓ s³/6 …` | — |
//! | `erf` | `2s/√π·(1 + O(s²))` | — |
//! | `cos`, `cosh` | `1 ∓ s²/2 + s⁴/24` | — |
//! | `−z`, `conj`, `re`, `im`, `abs` | of the mantissa | — |
//! | `Ei(x)`, `erfc(x)` | — | the asymptotic series times `exp` at its scale |
//! | `J_ν(x)`, `I_ν(x)` (`ν > 0`, `x² ≤ ν + 1`) | — | `e^L·₀F₁(; ν + 1; ∓x²/4)`, `L = ν·ln(abs(x)/2) − ln Γ(ν + 1)` |
//!
//! Beside its mantissa and scale, a value below the range is held as a
//! polynomial in such values with exact rational coefficients, plus a
//! remainder ([`Poly`]): the series above to the fifth order, `exp(g)` of
//! an exact `g` as a monomial whose products add the exponents
//! (`exp(−10¹⁰)²` and `exp(−2·10¹⁰)` are one monomial), any other value
//! (`Ei(−10¹⁰)`, a product with an inexact factor) as a monomial of its
//! own.  Sums and products of them are exact on the coefficients, so a
//! cancellation of the largest terms leaves the next ones: before 0.34
//! only the first order was kept, `ln(1 + e^(−10¹⁰)) − e^(−10¹⁰)` (truly
//! `−e^(−2·10¹⁰)/2`) and `sin(e^(−10¹⁰)) − e^(−10¹⁰)` (`−e^(−3·10¹⁰)/6`)
//! were a zero ball the zero search shrank, and printed `0`.  A sum whose
//! terms all cancel within the remainder of a series is undecided, and
//! refused ("not known to be 0"); a cancellation of different monomials
//! (`e^(−10¹⁰)·(sin²1 + cos²1) − e^(−10¹⁰)`) is numeric, a zero ball of the
//! working precision for the zero search, as before.
//!
//! A result back in range replaces the ordinary value: `(exp(−10¹⁰) −
//! exp(−2·10¹⁰))·…`, `ln(exp(−10¹⁰) − exp(−2·10¹⁰)) = −10¹⁰` (refused before:
//! the ordinary logarithm of `0 ± underflow`).  A sum whose largest terms
//! are exact and in range keeps the terms below the range as well
//! ([`Ext::Near`]), so `ln(1 + exp(−10¹⁰))` is `exp(−10¹⁰)·(1 + O(e^(−10¹⁰)))`
//! rather than `ln 1 = 0`.
//!
//! Only the bottom of the range: a value above it (`exp(10¹⁰)`) is refused
//! where it arises, as before ("overflows the exponent range"), and so is
//! anything that would come back from a scaled value above the range —
//! except a factor of a product with a factor below the range: a negative
//! power of a value below the range, or an `exp` above it, is scaled for
//! the product ([`beyond_power`]; `besselj(10⁹, 1)/besselj(10⁹, 2)`,
//! `exp(−10¹⁰)/exp(−10¹⁰ + 1)`, refused before 0.35).  Any
//! other node sees the placeholder: an underflow its own rules handle
//! (`accuracy::underflow_nonzero`), or an argument whose value is unknown.
//! A function of a value that underflowed without a scaled form (`K₀(10¹⁰)`,
//! `Ai(10¹⁰)`) has none either ([`View::Lost`]): in a sum it is an error
//! term of the size of the underflow bound.

use astro_float::{BigFloat, Consts, RoundingMode};
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::bigcomplex::{Complex, c_div, c_one, c_powi, c_zero};
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;

use super::accuracy::{self, Absorbed, Bound, ErrExp, Nonzero};
use super::exact::{self, ExactMap, Gauss};

/// The bottom of the exponent range.
const MIN_EXP: i64 = astro_float::EXPONENT_MIN as i64;

/// The top of the exponent range.
const MAX_EXP: i64 = astro_float::EXPONENT_MAX as i64;

/// The bits kept clear of either end of the exponent range: a value whose
/// magnitude is within them of the bottom is held scaled, and one within
/// them of the top is not produced.
fn margin(prec: usize) -> i64 {
    2 * prec as i64 + 256
}

/// `m·2^k`: a value below the exponent range as a mantissa in range and a
/// binary scale, with the error bound `eb` of the mantissa (the value's
/// error is `eb·2^k`, per part).  `|m| ∈ [½, 1)` unless `m` is 0 (a zero
/// ball of radius `eb·2^k`).
#[derive(Clone, Debug)]
pub(super) struct Scaled {
    pub(super) m: Complex,
    pub(super) k: i64,
    pub(super) eb: Bound,
    /// The value as a polynomial in values below the range, when this is
    /// the value of a node (`None` for a number alone).
    pub(super) poly: Option<Box<Poly>>,
}

/// What is known of a node beyond its ordinary value.
#[derive(Clone, Debug)]
pub(super) enum Ext {
    /// The value lies below the exponent range; the ordinary value is the
    /// placeholder `0 ± underflow`.
    Beyond(Scaled),
    /// The value is `p + d` with `p` an exact complex rational (the
    /// ordinary value, up to `d`) and `d` below the range: `1 + exp(−10¹⁰)`,
    /// `exp(exp(−10¹⁰))`, `8/3 − 5/3·cos(exp(−10¹⁰))`.
    Near(Gauss, Scaled),
}

/// The extended values of the nodes of one evaluation.
pub(super) type ExtMap = FxHashMap<ExprId, Ext>;

/// The extended evaluation of a node.
pub(super) enum Outcome {
    /// The value is in range: it replaces the ordinary one.
    InRange(Complex, Bound),
    /// The value lies below the range.
    Beyond(Scaled),
    /// The ordinary value stands, and the node is `p + d` ([`Ext::Near`]).
    Near(Gauss, Scaled),
}

impl Scaled {
    /// `m·2^k ± eb·2^k`, a number alone.
    fn plain(m: Complex, k: i64, eb: Bound) -> Scaled {
        Scaled {
            m,
            k,
            eb,
            poly: None,
        }
    }

    /// The number alone, without its polynomial.
    fn bare(&self) -> Scaled {
        Scaled::plain(self.m.clone(), self.k, self.eb)
    }

    /// `log₂` of an upper bound on the absolute value, its error included
    /// (`−∞` for an exact 0).
    fn lg_upper(&self) -> f64 {
        accuracy::lsum(accuracy::lg_abs(&self.m), self.eb.joint()) + self.k as f64
    }

    /// `log₂ abs(value)` (absolute, rounded up; `−∞` for 0).
    pub(super) fn lg_value(&self) -> f64 {
        accuracy::lg_abs(&self.m) + self.k as f64
    }

    /// `log₂` of an upper bound on `abs(value)`'s exponent: `mag(m) + k`.
    fn top(&self) -> Option<i64> {
        accuracy::mag(&self.m).map(|t| t.saturating_add(self.k))
    }

    /// Is the value certainly not 0, and with which sign?
    pub(super) fn nonzero(&self) -> Option<Nonzero> {
        if self.eb.is_unknown() || accuracy::contains_zero(&self.m, self.eb.joint()) {
            return None;
        }
        Some(if accuracy::exactly_real(&self.m, self.eb) {
            if self.m.0.is_negative() {
                Nonzero::Negative
            } else {
                Nonzero::Positive
            }
        } else {
            Nonzero::Unsigned
        })
    }

    /// The ordinary value and bound that stand for it: 0 within the
    /// underflow bound, on each part that is not exactly 0.
    pub(super) fn placeholder(&self, prec: usize) -> (Complex, Bound) {
        let part = |x: &BigFloat, e: ErrExp| {
            if accuracy::exact_zero(x, e) {
                accuracy::EXACT
            } else {
                accuracy::UNDERFLOW
            }
        };
        (
            c_zero(prec),
            Bound {
                re: part(&self.m.0, self.eb.re),
                im: part(&self.m.1, self.eb.im),
            },
        )
    }

    /// A number in range with the direction of the value, at the bottom of
    /// the range: for the zero, sign and realness tests of the crate (its
    /// magnitude is not the value's).  `None` for a zero ball.
    pub(super) fn representative(&self, prec: usize) -> Option<Complex> {
        let t = accuracy::mag(&self.m)?;
        let (z, _) = scale_with_loss(&self.m, self.eb, MIN_EXP + margin(prec) - t);
        Some(z)
    }

    /// `ln abs(value)`, for a magnitude estimate.
    pub(super) fn ln_abs(&self) -> Option<f64> {
        let t = accuracy::mag(&self.m)?;
        let lg = accuracy::lg_abs(&self.m);
        lg.is_finite()
            .then(|| (lg - t as f64 + (t + self.k) as f64) * std::f64::consts::LN_2)
    }
}

/// `±2^(EXPONENT_MIN + margin)`: a stand-in for a number below the range
/// of known sign (see [`Scaled::representative`]).
pub(super) fn tiny(negative: bool, prec: usize) -> BigFloat {
    let mut t = BigFloat::from_i32(if negative { -1 } else { 1 }, prec);
    t.set_exponent(i32::try_from(MIN_EXP + margin(prec)).unwrap_or(astro_float::EXPONENT_MIN));
    t
}

/// `x·2^j` exactly, or `None` when a nonzero `x` would leave the range.
fn scale_part(x: &BigFloat, j: i64) -> Option<BigFloat> {
    if x.is_zero() {
        return Some(x.clone());
    }
    let e = i64::from(x.exponent()?).checked_add(j)?;
    if e <= MIN_EXP || e >= MAX_EXP {
        return None;
    }
    let mut y = x.clone();
    y.set_exponent(i32::try_from(e).ok()?);
    Some(y)
}

/// `z·2^j` with its bound `eb·2^j`; a part that would fall below the range
/// (the smaller part of a value at its bottom) is 0 with its magnitude
/// added to its error.  `j` must not lift a part above the range.
fn scale_with_loss(z: &Complex, eb: Bound, j: i64) -> (Complex, Bound) {
    let part = |x: &BigFloat, e: ErrExp| -> (BigFloat, ErrExp) {
        let e2 = accuracy::shift(e, j as f64);
        match scale_part(x, j) {
            Some(y) => (y, e2),
            None => {
                let lost = accuracy::part_lg(x) + j as f64;
                (
                    BigFloat::new(x.mantissa_max_bit_len().unwrap_or(64)),
                    accuracy::lsum(e2, lost),
                )
            }
        }
    };
    let (re, ere) = part(&z.0, eb.re);
    let (im, eim) = part(&z.1, eb.im);
    ((re, im), Bound { re: ere, im: eim })
}

/// `v ± b` as `m·2^k` with `abs(m) ∈ [½, 1)`.
fn normalized(v: &Complex, b: Bound) -> Option<Scaled> {
    match accuracy::mag(v) {
        None => Some(Scaled::plain(v.clone(), 0, b)),
        Some(t) => {
            let (m, eb) = scale_with_loss(v, b, -t);
            Some(Scaled::plain(m, t, eb))
        }
    }
}

/// `s` as the outcome of a node: in range again, below it, or `None` when
/// its bound is unknown or it would lie at the top of the range (refused by
/// the ordinary evaluation).
fn finish(s: Scaled, prec: usize) -> Option<Outcome> {
    if s.eb.is_unknown() {
        return None;
    }
    let margin = margin(prec);
    match accuracy::mag(&s.m) {
        None => {
            if s.eb.is_exact() {
                return Some(Outcome::InRange(c_zero(prec), Bound::EXACT));
            }
            let radius = s.eb.joint() + s.k as f64;
            if radius > (MIN_EXP + margin) as f64 {
                if radius >= (MAX_EXP - margin) as f64 {
                    return None;
                }
                let k = s.k as f64;
                let eb = Bound {
                    re: accuracy::shift(s.eb.re, k),
                    im: accuracy::shift(s.eb.im, k),
                };
                return Some(Outcome::InRange(s.m, eb));
            }
            Some(Outcome::Beyond(s))
        }
        Some(t) => {
            let total = t.checked_add(s.k)?;
            if total >= MAX_EXP - margin {
                return None;
            }
            if total > MIN_EXP + margin {
                let (v, eb) = scale_with_loss(&s.m, s.eb, s.k);
                return Some(Outcome::InRange(v, eb));
            }
            let (m, eb) = scale_with_loss(&s.m, s.eb, -t);
            Some(Outcome::Beyond(Scaled {
                m,
                k: total,
                eb,
                poly: s.poly,
            }))
        }
    }
}

/// [`finish`] for the value of node `id` computed as a number: below the
/// range it is a monomial of its own.
fn finish_node(s: Scaled, id: ExprId, prec: usize) -> Option<Outcome> {
    finish_atom(s, Mono::node(id), prec)
}

/// [`finish`] for a value that is the monomial `mono` (a number without a
/// polynomial of its own).
fn finish_atom(s: Scaled, mono: Mono, prec: usize) -> Option<Outcome> {
    Some(match finish(s, prec)? {
        Outcome::Beyond(s) if s.poly.is_none() => {
            let poly = Poly::atom(mono, &s);
            Outcome::Beyond(Scaled {
                poly: Some(Box::new(poly)),
                ..s
            })
        }
        other => other,
    })
}

/// `p + d` for the value of node `id`, `d` computed as a number (a
/// monomial of its own when it has no polynomial).
fn near_node(p: Gauss, d: Scaled, id: ExprId) -> Outcome {
    Outcome::Near(p, with_poly(d, id))
}

/// `s`, the value of node `c` (or its part beyond an exact `p`), with a
/// polynomial: its own, or itself as a monomial.
fn with_poly(mut s: Scaled, c: ExprId) -> Scaled {
    if s.poly.is_none() {
        s.poly = Some(Box::new(Poly::atom(Mono::node(c), &s)));
    }
    s
}

/// The polynomial of `s`, the value of node `c` (see [`with_poly`]).
fn poly_of(c: ExprId, s: &Scaled) -> Poly {
    match &s.poly {
        Some(p) => (**p).clone(),
        None => Poly::atom(Mono::node(c), s),
    }
}

/// A child as the extended evaluation sees it.
enum View {
    /// `m·2^k`; `true` when it lies below the range (an [`Ext::Beyond`]),
    /// `false` for an ordinary value normalised.
    Plain(Scaled, bool),
    /// `p + d` ([`Ext::Near`]).
    Near(Gauss, Scaled),
    /// A value that underflowed without a scaled form: `0 ± b`.
    Lost(Bound),
}

/// Lookups shared by the node rules.
struct Ctx<'a> {
    arena: &'a Arena,
    cache: &'a FxHashMap<ExprId, Complex>,
    errs: &'a FxHashMap<ExprId, Bound>,
    exts: &'a ExtMap,
    exact: &'a ExactMap,
}

impl Ctx<'_> {
    fn view(&self, c: ExprId) -> Option<View> {
        match self.exts.get(&c) {
            Some(Ext::Beyond(s)) => return Some(View::Plain(s.clone(), true)),
            Some(Ext::Near(p, d)) => return Some(View::Near(p.clone(), d.clone())),
            None => {}
        }
        let (v, b) = self.ordinary(c)?;
        if accuracy::mag(v).is_none() && accuracy::is_underflow(b.joint()) {
            return Some(View::Lost(b));
        }
        Some(View::Plain(normalized(v, b)?, false))
    }

    /// The exact value of `c`, a rational, `i` or an exact composite.
    fn exact_of(&self, c: ExprId) -> Option<Gauss> {
        exact::operand(self.arena, c, self.exact)
    }

    /// The ordinary value and bound of `c`, when finite with a bound.
    fn ordinary(&self, c: ExprId) -> Option<(&Complex, Bound)> {
        let v = self.cache.get(&c)?;
        let b = *self.errs.get(&c)?;
        let finite = !(v.0.is_inf() || v.0.is_nan() || v.1.is_inf() || v.1.is_nan());
        (finite && !b.is_unknown()).then_some((v, b))
    }

    /// `c` as an ordinary value normalised, its [`Ext::Near`] record
    /// ignored (the ordinary value is `p` within the underflow bound).
    fn plain_or_near(&self, c: ExprId) -> Option<(Scaled, bool)> {
        match self.view(c)? {
            View::Plain(s, beyond) => Some((s, beyond)),
            View::Near(..) => {
                let (v, b) = self.ordinary(c)?;
                Some((normalized(v, b)?, false))
            }
            View::Lost(_) => None,
        }
    }
}

/// Does node `id` call for the extended evaluation: a child has an
/// extended value (or is a power of one, [`beyond_power`]), or its
/// ordinary value underflowed (`underflowed`)?
pub(super) fn wanted(arena: &Arena, id: ExprId, underflowed: bool, exts: &ExtMap) -> bool {
    underflowed
        || (!exts.is_empty()
            && arena.node(id).children().iter().any(|c| {
                exts.contains_key(c)
                    || matches!(arena.node(*c), ExprNode::Pow(b, _) if exts.contains_key(b))
            }))
}

/// The extended evaluation of node `id` (see the module documentation), or
/// `None` when the ordinary value stands.  `ordinary` is the node's
/// ordinary value, when it has one; `underflowed`: it is 0 with an
/// underflow bound.  A nonzero term a sum loses below its error bound is
/// recorded in `ab` ([`Absorbed`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn extend(
    arena: &Arena,
    id: ExprId,
    underflowed: bool,
    cache: &FxHashMap<ExprId, Complex>,
    errs: &FxHashMap<ExprId, Bound>,
    exts: &ExtMap,
    exact: &ExactMap,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
    ab: &mut Option<Absorbed>,
) -> Option<Outcome> {
    let cx = Ctx {
        arena,
        cache,
        errs,
        exts,
        exact,
    };
    match arena.node(id) {
        ExprNode::Exp(c) => match cx.view(*c)? {
            View::Plain(s, true) => {
                let fallback = s.bare();
                match series(&poly_of(*c, &s), &EXPM1, prec, rm) {
                    Some(d) => near(one(), d, prec, rm),
                    None => Some(near_node(one(), first_order(fallback, 1), id)),
                }
            }
            View::Plain(..) if underflowed => {
                let (z, b) = cx.ordinary(*c)?;
                if !z.1.is_zero() {
                    super::check_trig_arg(&z.1, 0, prec, arena).ok()?;
                }
                let s = exp_scaled(z, b, prec, rm, cc)?;
                // `exp` of an exact argument: the monomial `e^g`.
                let mono = match cx.exact_of(*c) {
                    Some(g) => Mono::exp(g),
                    None => Mono::node(id),
                };
                finish_atom(s, mono, prec)
            }
            _ => None,
        },
        ExprNode::Ln(c) => match cx.view(*c)? {
            View::Plain(s, true) => {
                let (v, b) = ln_scaled(&s, prec, rm, cc)?;
                Some(Outcome::InRange(v, b))
            }
            View::Near(p, d) if p == one() => match series(&poly_of(*c, &d), &LOG1P, prec, rm) {
                Some(l) => settle_poly(l, prec, rm, ab),
                None => finish_node(first_order(d, 1), id, prec),
            },
            _ => None,
        },
        ExprNode::Mul(children) => {
            if let Some(full) = poly_product(&cx, children, prec, rm) {
                return settle_poly(full, prec, rm, ab);
            }
            let mut factors = Vec::with_capacity(children.len());
            let mut beyond = false;
            for &c in children.iter() {
                let (s, b) = match cx.plain_or_near(c) {
                    Some(f) => f,
                    None => (beyond_power(&cx, c, prec, rm, cc)?, true),
                };
                beyond |= b;
                factors.push(s);
            }
            if !beyond && !underflowed {
                return None;
            }
            finish_node(product(&factors, prec, rm)?, id, prec)
        }
        ExprNode::Pow(b, e) => {
            if exts.contains_key(e) {
                return None;
            }
            let (x, ex) = cx.ordinary(*e)?;
            let q = arena.as_num(*e);
            match cx.view(*b)? {
                View::Plain(s, true) => {
                    if let Some(q) = q
                        && let Some(full) = poly_power(&poly_of(*b, &s), q, x, ex, prec, rm, cc)
                    {
                        return settle_poly(full, prec, rm, ab);
                    }
                    finish_node(power(&s, q, x, ex, prec, rm, cc)?, id, prec)
                }
                View::Plain(s, false) if underflowed => {
                    finish_node(power(&s, q, x, ex, prec, rm, cc)?, id, prec)
                }
                View::Near(p, d) if p == one() => {
                    // `(1 + d)^q` by the binomial series for an exact `q`.
                    if let Some(q) = q
                        && let Some(dq) = binomial(&poly_of(*b, &d), q, prec, rm)
                    {
                        return near(one(), dq, prec, rm);
                    }
                    // `(1 + d)^x = 1 + x·d·(1 + O(d))`.
                    let xd = product(&[d.bare(), normalized(x, ex)?], prec, rm)?;
                    Some(near_node(one(), first_order(xd, 1), id))
                }
                _ => None,
            }
        }
        ExprNode::Add(children) => sum_node(&cx, id, children, prec, rm, ab),
        ExprNode::Neg(c) => match cx.view(*c)? {
            View::Plain(s, true) => finish(negated(with_poly(s, *c)), prec),
            View::Near(p, d) => Some(Outcome::Near(p.neg(), negated(with_poly(d, *c)))),
            _ => None,
        },
        ExprNode::Conjugate(c) => match cx.view(*c)? {
            View::Plain(s, true) => finish_node(conjugated(with_poly(s, *c)), id, prec),
            View::Near(p, d) => Some(near_node(
                Gauss {
                    re: p.re,
                    im: -p.im,
                },
                conjugated(with_poly(d, *c)),
                id,
            )),
            _ => None,
        },
        ExprNode::Re(c) | ExprNode::Im(c) | ExprNode::Abs(c) => {
            let View::Plain(s, true) = cx.view(*c)? else {
                return None;
            };
            let (m, eb) = match arena.node(id) {
                ExprNode::Re(_) => (s.m.0.clone(), s.eb.re),
                ExprNode::Im(_) => (s.m.1.clone(), s.eb.im),
                // `abs` is 1-Lipschitz.
                _ => {
                    let a = crate::base::bigcomplex::c_abs(&s.m, prec, rm);
                    let r = accuracy::rounding(&(a.clone(), BigFloat::new(prec)), prec);
                    (a, accuracy::lsum(s.eb.joint(), r))
                }
            };
            let z = (m, BigFloat::new(prec));
            finish_node(Scaled::plain(z, s.k, Bound::real(eb)), id, prec)
        }
        // `erf s = 2/√π·s·(1 + O(s²))`.
        ExprNode::Erf(c) => {
            let View::Plain(s, true) = cx.view(*c)? else {
                return None;
            };
            let s = first_order(s, 2);
            let wp = prec + 16;
            let pi = cc.pi(wp, rm).clone();
            let two_over_sqrt_pi = BigFloat::from_i32(2, wp).div(&pi.sqrt(wp, rm), wp, rm);
            let f = normalized(
                &(two_over_sqrt_pi, BigFloat::new(wp)),
                Bound::real(-(prec as f64) - 8.0),
            )?;
            finish_node(product(&[s, f], prec, rm)?, id, prec)
        }
        // Odd functions with `f′(0) = 1`: their series, `f(s) = s + c₃s³ + c₅s⁵ + O(s⁷)`.
        ExprNode::Sin(c)
        | ExprNode::Tan(c)
        | ExprNode::Sinh(c)
        | ExprNode::Tanh(c)
        | ExprNode::Asin(c)
        | ExprNode::Atan(c)
        | ExprNode::Asinh(c)
        | ExprNode::Atanh(c) => {
            let View::Plain(s, true) = cx.view(*c)? else {
                return None;
            };
            let table = match arena.node(id) {
                ExprNode::Sin(_) => &SIN,
                ExprNode::Tan(_) => &TAN,
                ExprNode::Sinh(_) => &SINH,
                ExprNode::Tanh(_) => &TANH,
                ExprNode::Asin(_) => &ASIN,
                ExprNode::Atan(_) => &ATAN,
                ExprNode::Asinh(_) => &ASINH,
                _ => &ATANH,
            };
            match series(&poly_of(*c, &s), table, prec, rm) {
                Some(f) => settle_poly(f, prec, rm, ab),
                None => finish_node(first_order(s, 2), id, prec),
            }
        }
        // `cos s = 1 − s²/2 + s⁴/24 + O(s⁶)`, `cosh s = 1 + s²/2 + …`.
        ExprNode::Cos(c) | ExprNode::Cosh(c) => {
            let View::Plain(s, true) = cx.view(*c)? else {
                return None;
            };
            let is_cos = matches!(arena.node(id), ExprNode::Cos(_));
            let table = if is_cos { &COSM1 } else { &COSHM1 };
            if let Some(d) = series(&poly_of(*c, &s), table, prec, rm) {
                return near(one(), d, prec, rm);
            }
            let s = s.bare();
            let mut sq = product(&[s.clone(), s], prec, rm)?;
            sq.k = sq.k.checked_sub(1)?;
            if is_cos {
                sq = negated(sq);
            }
            Some(near_node(one(), first_order(sq, 1), id))
        }
        ExprNode::Ei(c) if underflowed => {
            let (x, b) = cx.ordinary(*c)?;
            if !accuracy::exactly_real(x, b) || !x.0.is_negative() {
                return None;
            }
            finish_node(ei_scaled(&x.0, b.re, prec, rm, cc)?, id, prec)
        }
        ExprNode::Erfc(c) if underflowed => {
            let (x, b) = cx.ordinary(*c)?;
            if !accuracy::exactly_real(x, b) || !super::bf_strictly_positive(&x.0) {
                return None;
            }
            finish_node(erfc_scaled(&x.0, b.re, prec, rm, cc)?, id, prec)
        }
        ExprNode::Apply(sid, args) if underflowed && args.len() == 2 => {
            let alternating = match arena.lib_fn(*sid) {
                Some(LibFn::BesselJ) => true,
                Some(LibFn::BesselI) => false,
                _ => return None,
            };
            let (nu, nb) = cx.ordinary(args[0])?;
            let (x, xb) = cx.ordinary(args[1])?;
            if !accuracy::exactly_real(nu, nb) || !accuracy::exactly_real(x, xb) {
                return None;
            }
            // `J_{−n} = (−1)ⁿ J_n`, `I_{−n} = I_n` for an exact integer order.
            let reflected = nu.0.is_negative() && nu.0.is_int() && nb.is_exact();
            let order = if reflected { nu.0.abs() } else { nu.0.clone() };
            let s = bessel_scaled(&order, nb.re, &x.0, xb.re, alternating, prec, rm, cc)?;
            let flip = reflected && alternating && super::bessel_order::is_odd(&nu.0);
            finish_node(if flip { negated(s) } else { s }, id, prec)
        }
        _ => None,
    }
}

/// The exact 0.
fn zero() -> Gauss {
    Gauss::real(Q::zero())
}

/// The exact 1.
fn one() -> Gauss {
    Gauss::real(Q::one())
}

/// The exact `n/d`.
fn ratio(n: i64, d: i64) -> Gauss {
    Gauss::real(Q::new(n.into(), d.into()))
}

/// `p + d` ([`Outcome::Near`]) for the polynomial `d` (without a constant
/// term).
fn near(p: Gauss, d: Poly, prec: usize, rm: RoundingMode) -> Option<Outcome> {
    Some(Outcome::Near(p, numeric(d, prec, rm)?))
}

/// `s·(1 + O(s^n))`: `s` with the relative error `abs(s)^n` added (the
/// next term of a series in a value below the range, negligible but
/// counted).  A number alone: it no longer has `s`'s polynomial.
fn first_order(mut s: Scaled, n: i64) -> Scaled {
    s.poly = None;
    if let Some(t) = s.top() {
        let rel = accuracy::lg_abs(&s.m) + (n as f64) * (t as f64);
        s.eb = Bound {
            re: accuracy::lsum(s.eb.re, rel),
            im: if accuracy::exactly_real(&s.m, s.eb) {
                s.eb.im
            } else {
                accuracy::lsum(s.eb.im, rel)
            },
        };
    }
    s
}

fn negated(s: Scaled) -> Scaled {
    Scaled {
        m: (s.m.0.neg(), s.m.1.neg()),
        poly: s.poly.map(|p| Box::new(p.negated())),
        ..s
    }
}

/// The conjugate; its polynomial when every monomial is an `exp` (the
/// conjugate of another value is not one of the monomials).
fn conjugated(s: Scaled) -> Scaled {
    Scaled {
        m: (s.m.0, s.m.1.neg()),
        poly: s.poly.and_then(|p| p.conjugated().map(Box::new)),
        ..s
    }
}

/// `Π sᵢ`: the mantissas multiplied with their bounds, the scales added.
fn product(factors: &[Scaled], prec: usize, rm: RoundingMode) -> Option<Scaled> {
    let parts: Vec<(&Complex, Bound)> = factors.iter().map(|s| (&s.m, s.eb)).collect();
    let (m, eb) = accuracy::mul_with_bound(&parts, prec, rm);
    let k = factors.iter().try_fold(0i64, |a, s| a.checked_add(s.k))?;
    Some(Scaled::plain(m, k, eb))
}

/// Bits of `abs(k)`.
fn bits(k: i64) -> usize {
    (64 - k.unsigned_abs().leading_zeros()) as usize
}

/// `log₂(e^R − 1)` for a radius `R = 2^r` (`r = −∞` for an exact
/// argument); unknown beyond `R = 512`.
fn lg_expm1(r: ErrExp) -> ErrExp {
    if accuracy::is_exact(r) {
        return accuracy::EXACT;
    }
    if accuracy::is_unknown(r) || r > 9.0 {
        return accuracy::UNKNOWN;
    }
    if r < -60.0 {
        // `e^R − 1 = R·(1 + R/2 + …)`.
        return r + 1e-12;
    }
    r.exp2().exp_m1().log2() + 1e-12
}

/// `e^z` as `m·2^k`: `k = round(re z/ln 2)` and `m = e^(z − k·ln 2)`, the
/// reduction at `prec + 64` bits beyond `k`'s, so that `m` keeps `prec`
/// bits (`zb`: `z`'s bound).  The bound of `m` is `abs(m)·(e^R − 1)` with
/// `R` the radius of the reduced argument (as for `exp` in
/// `accuracy::node_error`), plus its rounding.
fn exp_scaled(
    z: &Complex,
    zb: Bound,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Scaled> {
    let x = super::bigfloat_to_f64_rounded(&z.0, RoundingMode::ToEven).ok()?;
    if !x.is_finite() || x.abs() > 4e18 {
        return None;
    }
    let k = (x * std::f64::consts::LOG2_E).round() as i64;
    let p = prec + bits(k) + 64;
    let kl = BigFloat::from_i64(k, 64).mul(&cc.ln_2(p, rm), p, rm);
    let r = z.0.sub(&kl, p, rm);
    // `k·ln 2` and the difference are each rounded at `p` bits.
    let reduction = (bits(k) as f64 + 2.0) - p as f64;
    let radius = accuracy::lsum(zb.joint(), reduction);
    let real = accuracy::exactly_real(z, zb);
    let m: Complex = if real {
        (r.exp(prec, rm, cc), BigFloat::new(prec))
    } else {
        super::c_exp(&(r, z.1.clone()), prec, rm, cc)
    };
    let lv = accuracy::lg_abs(&m);
    let rel = lg_expm1(radius);
    let part = |x: &BigFloat| {
        let round = accuracy::rounding(&(x.clone(), BigFloat::new(prec)), prec) + 1.0;
        accuracy::lsum(lv + rel, round)
    };
    let eb = Bound {
        re: part(&m.0),
        im: if real { accuracy::EXACT } else { part(&m.1) },
    };
    let t = accuracy::mag(&m)?;
    let (m, eb) = scale_with_loss(&m, eb, -t);
    Some(Scaled::plain(m, k.checked_add(t)?, eb))
}

/// `Log(m·2^k) = Log m + k·ln 2` (`2^k` is positive: the argument is
/// `m`'s) at `prec` bits, with its bound: `r/(abs(m) − r)` for the radius
/// `r` of `m` (at most `2r/abs(m)`; `None` when `r > abs(m)/2`), the
/// roundings, and no bound when the side of the cut is undecidable.  The
/// imaginary part is exactly 0 for an exactly real positive `m`.
fn ln_scaled(
    s: &Scaled,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<(Complex, Bound)> {
    let lm = accuracy::lg_abs_low(&s.m);
    let r = s.eb.joint();
    if accuracy::mag(&s.m).is_none() || s.eb.is_unknown() || r > lm - 1.0 {
        return None;
    }
    let real = accuracy::exactly_real(&s.m, s.eb);
    let positive = real && s.m.0.is_positive();
    if !real && accuracy::cut_undecided(&s.m, s.eb, accuracy::Cut::NegativeReal) {
        return None;
    }
    let p = prec + bits(s.k) + 64;
    let (lre, lim) = if positive {
        (s.m.0.ln(p, rm, cc), BigFloat::new(prec))
    } else {
        super::c_ln(&s.m, p, rm, cc)
    };
    let kl = BigFloat::from_i64(s.k, 64).mul(&cc.ln_2(p, rm), p, rm);
    let re = super::round_to(lre.add(&kl, p, rm), prec, rm);
    let im = super::round_to(lim, prec, rm);
    let prop = r - lm + 1.0;
    let reduction = (bits(s.k) as f64 + 2.0) - p as f64;
    let rounding = |x: &BigFloat| accuracy::rounding(&(x.clone(), BigFloat::new(prec)), prec) + 1.0;
    let eb = Bound {
        re: accuracy::lsum(accuracy::lsum(prop, reduction), rounding(&re)),
        im: if positive {
            accuracy::EXACT
        } else {
            accuracy::lsum(prop, rounding(&im))
        },
    };
    Some(((re, im), eb))
}

/// `b^x` for `b = s` (`q`: the exponent's literal value, `x ± ex` its
/// value): an integer power up to 1024 by multiplication (`mⁿ·2^(nk)`, the
/// bound as for any power, `accuracy::pow_bound`); otherwise `exp(x·Log b)`
/// with `Log b` at the precision its size needs.
fn power(
    s: &Scaled,
    q: Option<&crate::base::numeric::Q>,
    x: &Complex,
    ex: Bound,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Scaled> {
    accuracy::mag(&s.m)?;
    if let Some(q) = q
        && q.is_integer()
        && let Some(n) = num_traits::ToPrimitive::to_i64(&q.to_integer())
        && n.unsigned_abs() <= 1024
    {
        let wp = prec + 16;
        let pos = c_powi(&s.m, n.unsigned_abs() as usize, wp, rm);
        let v = if n >= 0 {
            pos
        } else {
            c_div(&c_one(wp), &pos, wp, rm)
        };
        let v = (
            super::round_to(v.0, prec, rm),
            super::round_to(v.1, prec, rm),
        );
        let lv = accuracy::lg_abs(&v);
        let prop = accuracy::pow_bound(Some(q), &s.m, s.eb, x, ex, lv);
        let rounding =
            |y: &BigFloat| accuracy::rounding(&(y.clone(), BigFloat::new(prec)), prec) + 1.0;
        let part = |e: ErrExp, y: &BigFloat| {
            if accuracy::exact_zero(y, e) {
                e
            } else {
                accuracy::lsum(e, rounding(y))
            }
        };
        let eb = Bound {
            re: part(prop.re, &v.0),
            im: part(prop.im, &v.1),
        };
        return Some(Scaled::plain(v, s.k.checked_mul(n)?, eb));
    }
    let p = prec + bits(s.k) + 64;
    let (l, lb) = ln_scaled(s, p, rm, cc)?;
    let (w, wb) = accuracy::mul_with_bound(&[(&l, lb), (x, ex)], p, rm);
    exp_scaled(&w, wb, prec, rm, cc)
}

/// A factor `c` of a product that has no value of its own because it lies
/// above the exponent range (`exponent_range_error`): a power `b^e` of a
/// base `b` below the range (a negative power), or `exp(g)` of an argument
/// in range.  Its scaled value, `k` unbounded, for the product with a
/// factor below the range to bring back: `besselj(10⁹, 1)/besselj(10⁹, 2)
/// = 2.17·10^(−301029996)`, `erfc(10⁵)/erfc(10⁵ + 1)`,
/// `exp(−10¹⁰)/exp(−10¹⁰ + 1) = e⁻¹` (all refused before 0.35).  A product above the range is
/// still refused ([`finish`]).
fn beyond_power(
    cx: &Ctx<'_>,
    c: ExprId,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Scaled> {
    match cx.arena.node(c) {
        ExprNode::Pow(b, e) => {
            let Some(Ext::Beyond(s)) = cx.exts.get(b) else {
                return None;
            };
            if cx.exts.contains_key(e) {
                return None;
            }
            let (x, ex) = cx.ordinary(*e)?;
            power(s, cx.arena.as_num(*e), x, ex, prec, rm, cc)
        }
        ExprNode::Exp(g) => {
            if cx.exts.contains_key(g) {
                return None;
            }
            let (z, b) = cx.ordinary(*g)?;
            if !super::bf_strictly_positive(&z.0) {
                return None;
            }
            if !z.1.is_zero() {
                super::check_trig_arg(&z.1, 0, prec, cx.arena).ok()?;
            }
            exp_scaled(z, b, prec, rm, cc)
        }
        _ => None,
    }
}

/// A sum with a term below the range (SymPy's `add_terms`, at a common
/// scale): see the module documentation.  The exact rational terms (and the
/// exact parts of [`Ext::Near`] terms) are added exactly: when they and the
/// terms below the range are all there is, the sum is `p + d` (or `d` when
/// `p` is 0 — `1 − cos(e^(−10¹⁰))`), their polynomials added
/// ([`settle_poly`]).
fn sum_node(
    cx: &Ctx<'_>,
    id: ExprId,
    children: &[ExprId],
    prec: usize,
    rm: RoundingMode,
    ab: &mut Option<Absorbed>,
) -> Option<Outcome> {
    let mut beyond_terms = Vec::with_capacity(children.len());
    let mut ordinary_terms = Vec::new();
    let mut p = zero();
    let mut lost = Bound::EXACT;
    // The sum of the polynomials, while it stays within its limits.
    let mut full = Some(Poly::zero());
    for &c in children {
        if !cx.exts.contains_key(&c)
            && let Some(q) = cx.exact_of(c)
        {
            p = p.add(&q);
            if !p.small() {
                return None;
            }
            continue;
        }
        match cx.view(c)? {
            View::Plain(s, true) => {
                full = full.and_then(|f| f.plus(poly_of(c, &s)));
                beyond_terms.push(s);
            }
            View::Plain(s, false) => ordinary_terms.push(s),
            View::Near(q, d) => {
                p = p.add(&q);
                if !p.small() {
                    return None;
                }
                full = full.and_then(|f| f.plus(poly_of(c, &d)));
                beyond_terms.push(d);
            }
            View::Lost(b) => {
                lost = Bound {
                    re: accuracy::lsum(lost.re, b.re),
                    im: accuracy::lsum(lost.im, b.im),
                };
            }
        }
    }
    if beyond_terms.is_empty() {
        return None;
    }
    if ordinary_terms.is_empty() && lost.is_exact() {
        if let Some(f) = full.and_then(|f| f.plus(Poly::constant(p.clone()))) {
            return settle_poly(f, prec, rm, ab);
        }
        let d = add_terms(&beyond_terms, Bound::EXACT, prec, rm)?;
        return if p.is_zero() {
            finish_node(d, id, prec)
        } else {
            Some(near_node(p, d, id))
        };
    }
    // The terms by shape (see `note_lost`): a number in range is a term of
    // its own, a value below the range has the shapes of its polynomial.
    let mut shaped: Vec<(Option<Mono>, Scaled)> = Vec::new();
    for s in &ordinary_terms {
        shaped.push((None, s.bare()));
    }
    for s in &beyond_terms {
        match s.poly.as_ref().and_then(|p| shaped_terms(p, prec, rm)) {
            Some(terms) => shaped.extend(terms),
            None => shaped.push((None, s.bare())),
        }
    }
    let mut terms = ordinary_terms;
    terms.append(&mut beyond_terms);
    if !p.is_zero() {
        let (v, b) = exact::to_value(&p, prec, rm);
        let v = normalized(&v, b)?;
        shaped.push((None, v.clone()));
        terms.push(v);
    }
    let s = add_terms(&terms, lost, prec, rm)?;
    // The value is a number alone from here: what lies below its error is
    // lost.
    note_lost(shaped, s.eb.joint() + s.k as f64, prec, rm, ab);
    // A ball that holds 0 only because of a term without a scaled form says
    // nothing the placeholder does not.
    if !lost.is_exact() {
        s.nonzero()?;
    }
    finish_node(s.bare(), id, prec)
}

/// `Σ terms ± lost` (`lost`: error-only terms, in absolute `log₂`), each
/// part on its own as SymPy's `evalf_add` adds the real and the imaginary
/// parts ([`part_sum`]): an imaginary part far below the real one is still
/// known (it decides the side of a cut), then both at the larger scale.
fn add_terms(terms: &[Scaled], lost: Bound, prec: usize, rm: RoundingMode) -> Option<Scaled> {
    let (sr, kr, er) = part_sum(
        terms.iter().map(|t| (&t.m.0, t.k, t.eb.re)),
        lost.re,
        prec,
        rm,
    )?;
    let (si, ki, ei) = part_sum(
        terms.iter().map(|t| (&t.m.1, t.k, t.eb.im)),
        lost.im,
        prec,
        rm,
    )?;
    // A zero part's scale is its error's: `0 ± 2^e` at scale 0.
    let k = match (sr.is_zero(), si.is_zero()) {
        (true, true) => kr.max(ki),
        (false, true) => kr,
        (true, false) => ki,
        (false, false) => kr.max(ki),
    };
    let part = |s: BigFloat, ks: i64, e: ErrExp| -> (BigFloat, ErrExp) {
        let (z, b) = scale_with_loss(&(s, BigFloat::new(prec)), Bound::real(e), ks - k);
        (z.0, b.re)
    };
    let (re, ebr) = part(sr, kr, er);
    let (im, ebi) = part(si, ki, ei);
    Some(Scaled::plain((re, im), k, Bound { re: ebr, im: ebi }))
}

/// `Σ xᵢ·2^(kᵢ) ± lost` for one part (`xᵢ ± eᵢ` at the scale `kᵢ`): SymPy's
/// `add_terms` — the terms within `prec + 64` bits of the largest, at its
/// scale `K` (`accuracy::add_with_bound`), the smaller ones in the error;
/// when the largest cancel exactly (exact terms, an exact zero sum) the next
/// ones take their place.  `(S, K, e)`: the sum is `S·2^K ± 2^(e + K)`; a
/// zero sum (a cancellation) keeps the scale of its terms, so that its ball
/// is a relative one the zero search of `evalf` can shrink.
fn part_sum<'a>(
    parts: impl Iterator<Item = (&'a BigFloat, i64, ErrExp)>,
    lost: ErrExp,
    prec: usize,
    rm: RoundingMode,
) -> Option<(BigFloat, i64, ErrExp)> {
    let window = prec as i64 + 64;
    let mut extra = lost;
    // (value, scale, bound, top)
    let mut live: Vec<(&BigFloat, i64, ErrExp, i64)> = Vec::new();
    for (x, k, e) in parts {
        match accuracy::part_mag(x) {
            Some(t) => live.push((x, k, e, t.checked_add(k)?)),
            None => extra = accuracy::lsum(extra, accuracy::shift(e, k as f64)),
        }
    }
    loop {
        let Some(kmax) = live.iter().map(|t| t.3).max() else {
            // Only error terms: `0 ± 2^extra`, at the scale of the error.
            if accuracy::is_exact(extra) || accuracy::is_unknown(extra) {
                return Some((BigFloat::new(prec), 0, extra));
            }
            let k = extra.ceil();
            if k.is_nan() || k.abs() >= 9e18 {
                return None;
            }
            return Some((BigFloat::new(prec), k as i64, extra - k));
        };
        let (head, below): (Vec<_>, Vec<_>) = live.iter().partition(|t| t.3 >= kmax - window);
        let mut shifted: Vec<(Complex, Bound)> = Vec::with_capacity(head.len());
        for &&(x, k, e, _) in &head {
            let j = k - kmax;
            shifted.push((
                (scale_part(x, j)?, BigFloat::new(prec)),
                Bound::real(accuracy::shift(e, j as f64)),
            ));
        }
        let refs: Vec<(&Complex, Bound)> = shifted.iter().map(|(z, b)| (z, *b)).collect();
        let (sum, sb) = accuracy::add_with_bound(&refs, prec, rm);
        if sum.0.is_zero() && accuracy::is_exact(sb.re) {
            live = below.into_iter().copied().collect();
            continue;
        }
        let mut eb = sb.re;
        for &&(x, k, e, _) in &below {
            let m = accuracy::lsum(accuracy::part_lg(x), e);
            eb = accuracy::lsum(eb, accuracy::shift(m, (k - kmax) as f64));
        }
        eb = accuracy::lsum(eb, accuracy::shift(extra, -(kmax as f64)));
        return Some((sum.0, kmax, eb));
    }
}

/// `Ei(x)` for a large negative `x` as `m·2^k`: the asymptotic factor
/// `Σ k!/xᵏ / x` (as `evalf::arb_ei` sums it) times `e^x` at its scale.  The
/// bound: the factor's and `e^x`'s relative errors, and `x`'s error through
/// `Ei′/Ei = 1/(x·Ei e^(−x))` ≈ 1 (counted twice).
fn ei_scaled(
    x: &BigFloat,
    xe: ErrExp,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Scaled> {
    let wp = prec + 32;
    let (inv_x, sum) = super::ei_asymptotic_series(x, wp, rm);
    let e = exp_scaled(
        &(x.clone(), BigFloat::new(prec)),
        Bound::real(xe),
        wp,
        rm,
        cc,
    )?;
    let m = e.m.0.mul(&inv_x, wp, rm).mul(&sum, wp, rm);
    let m = super::round_to(m, prec, rm);
    scaled_by(
        e,
        (m, BigFloat::new(prec)),
        accuracy::shift(xe, 1.0),
        wp,
        prec,
    )
}

/// `erfc(x)` for a large `x` as `m·2^k`: the asymptotic sum over `x√π`
/// (as `evalf::erfc_asymptotic` sums it) times `e^(−x²)` at its scale, with
/// `x²` exact enough for its exponential (`2·log₂ x` bits more).  The bound:
/// the relative errors of the sum and of `e^(−x²)` (whose argument carries
/// `2x·err(x)`), and `x`'s error through the factor `1/x`.
fn erfc_scaled(
    x: &BigFloat,
    xe: ErrExp,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Scaled> {
    let wp = prec + 32;
    let t = accuracy::mag(&(x.clone(), BigFloat::new(64)))?;
    let px = wp + 2 * usize::try_from(t.max(0)).ok()? + 16;
    let x_sq = x.mul(x, px, rm);
    let arg_err = accuracy::lsum(
        accuracy::shift(xe, (t + 1) as f64),
        accuracy::rounding(&(x_sq.clone(), BigFloat::new(64)), px),
    );
    let (series, x_sqrt_pi) = super::erfc_asymptotic_series(x, wp, rm, cc);
    let e = exp_scaled(
        &(x_sq.neg(), BigFloat::new(prec)),
        Bound::real(arg_err),
        wp,
        rm,
        cc,
    )?;
    let m = e.m.0.mul(&series, wp, rm).div(&x_sqrt_pi, wp, rm);
    let m = super::round_to(m, prec, rm);
    scaled_by(
        e,
        (m, BigFloat::new(prec)),
        accuracy::shift(xe, -(t as f64) + 2.0),
        wp,
        prec,
    )
}

/// `J_ν(x)` (`alternating`) or `I_ν(x)` below the range as `m·2^k`, for a
/// real order `ν > 0` and a real `x ≠ 0` (an integer order for `x < 0`)
/// with `y = x²/(4(ν + 1)) ≤ 1/4` — where they underflow:
///
/// ```text
/// J_ν(x), I_ν(x) = ±e^L·S,  L = ν·ln(abs(x)/2) − ln Γ(ν + 1),
/// S = Σ_k (∓x²/4)^k/(k!·(ν + 1)_k)
/// ```
///
/// (mpmath's `besselj`, `(z/2)^ν/Γ(ν + 1)·₀F₁(; ν + 1; ∓z²/4)`, with
/// unbounded exponents; `L` from `bessel_order::log_power_over_gamma`,
/// `e^L` at its scale by [`exp_scaled`]).  The terms of `S` fall by `y` or
/// more, so `S ∈ [3/4, e^(1/4)]`; it is summed until a term is below
/// `2^(−wp−8)`, and its error is at most `(K + 8)·2^(2−wp)` relative for `K`
/// terms (running products of `k` roundings each, `Σ k·y^k ≤ 1/2`).  An
/// inexact argument adds `2·(abs(∂ln f/∂x)·err(x) + abs(∂ln f/∂ν)·err(ν))`
/// relative, with `abs(∂ln f/∂x) ≤ (ν + 1)/abs(x) + abs(x)` (`ν/x` and
/// `S′/S`) and `abs(∂ln f/∂ν) ≤ abs(ln(abs(x)/2)) + ln(ν + 1) + 2`
/// (`abs(ψ(ν + 1)) ≤ ln(ν + 1) + 1`), over balls of relative radius
/// below `1/16`.  Before 0.35 `log(besselj(8345185991999992, 61/10²⁷))`
/// was refused.
#[allow(clippy::too_many_arguments)]
fn bessel_scaled(
    nu: &BigFloat,
    nu_err: ErrExp,
    x: &BigFloat,
    x_err: ErrExp,
    alternating: bool,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Scaled> {
    if !super::bf_strictly_positive(nu) || x.is_zero() || x.is_nan() || x.is_inf() {
        return None;
    }
    let integer = nu.is_int();
    if x.is_negative() && !(integer && accuracy::is_exact(nu_err)) {
        return None;
    }
    let nu_f = super::bigfloat_to_f64_rounded(nu, rm).ok()?;
    let x_abs = x.abs();
    let lg_x = accuracy::part_lg(&x_abs);
    let xf = super::bigfloat_to_f64_rounded(&x_abs, rm).ok()?;
    if !nu_f.is_finite() || !xf.is_finite() || xf * xf > nu_f + 1.0 {
        return None;
    }
    // The arguments' errors, relative to the value.
    let times = |a: f64, e: ErrExp| {
        if accuracy::is_exact(e) {
            accuracy::EXACT
        } else {
            a.max(f64::MIN_POSITIVE).log2() + e
        }
    };
    // In `log₂`, for an `x` beyond the `f64` range (`2^(−10⁶)`).
    let dx = if accuracy::is_exact(x_err) {
        accuracy::EXACT
    } else {
        accuracy::lsum((nu_f + 1.0).log2() - lg_x, lg_x) + x_err
    };
    let ln_half_x = ((lg_x - 1.0) * std::f64::consts::LN_2).abs();
    let dnu = times(ln_half_x + (nu_f + 1.0).ln() + 2.0, nu_err);
    let args_rel = accuracy::lsum(dx, dnu) + 1.0;
    if !accuracy::is_exact(x_err) && x_err - lg_x > -4.0 {
        return None;
    }
    if !accuracy::is_exact(args_rel) && args_rel > -4.0 {
        return None;
    }
    let wp = prec + 32;
    let two = BigFloat::from_i32(2, 64);
    let x_half = x_abs.div(&two, wp + 8, rm);
    let l = super::bessel_order::log_power_over_gamma(nu, &x_half, wp, rm, cc).ok()?;
    let e = exp_scaled(
        &(l, BigFloat::new(prec)),
        Bound::real(-((wp + 12) as f64)),
        wp,
        rm,
        cc,
    )?;
    // S, its terms by running products.
    let q = x_half.mul(&x_half, wp, rm);
    let q = if alternating { q.neg() } else { q };
    let mut term = BigFloat::from_i32(1, wp);
    let mut sum = term.clone();
    let mut k: u64 = 0;
    loop {
        k += 1;
        if k > (2 * wp) as u64 {
            return None;
        }
        let kb = BigFloat::from_u64(k, 64);
        let nk = nu.add(&kb, super::exact_bits(nu, wp), rm);
        term = term.mul(&q, wp, rm).div(&kb.mul(&nk, wp, rm), wp, rm);
        if term.is_zero()
            || term
                .exponent()
                .is_some_and(|t| i64::from(t) < -(wp as i64) - 8)
        {
            break;
        }
        sum = sum.add(&term, wp, rm);
    }
    let s_rel = ((k + 8) as f64).log2() + 2.0 - wp as f64;
    let negative = x.is_negative() && super::bessel_order::is_odd(nu);
    let m = e.m.0.mul(&sum, wp, rm);
    let m = super::round_to(if negative { m.neg() } else { m }, prec, rm);
    scaled_by(
        e,
        (m, BigFloat::new(prec)),
        accuracy::lsum(s_rel, args_rel),
        wp,
        prec,
    )
}

/// `e·f` for `e = exp_scaled(…)` and the computed product `m = e.m·f`
/// (real): `m·2^(e.k)` with `e`'s relative error, the relative error
/// `rel` of the other factor, and the roundings at `wp` and `prec`.
fn scaled_by(e: Scaled, m: Complex, rel: ErrExp, wp: usize, prec: usize) -> Option<Scaled> {
    let lm = accuracy::lg_abs(&m);
    if !lm.is_finite() {
        return None;
    }
    let e_rel = e.eb.re - accuracy::lg_abs_low(&e.m);
    let work = 3.0 - wp as f64;
    let round = accuracy::rounding(&m, prec) + 1.0;
    let eb = accuracy::lsum(lm + accuracy::lsum(accuracy::lsum(e_rel, rel), work), round);
    let t = accuracy::mag(&m)?;
    let (m, eb) = scale_with_loss(&m, Bound::real(eb), -t);
    Some(Scaled::plain(m, e.k.checked_add(t)?, eb))
}

// ─── Polynomials in values below the range ──────────────────────────────

/// Most terms of a [`Poly`]; beyond, the value is a number alone.
const MAX_TERMS: usize = 48;

/// Largest integer power of a polynomial of several terms.
const MAX_POLY_POWER: i64 = 8;

/// A monomial of a [`Poly`]: `e^g·Π vᵢ^nᵢ·Π fⱼ^mⱼ` — `g` exact (`exp` of an
/// exact argument below the range), `vᵢ` the value of node `i` below the
/// range known as a number (`Ei(−10¹⁰)`), `fⱼ` an inexact factor in range
/// of a product (`sin²1 + cos²1`).  Without its factors it is the
/// monomial's [`shape`](Mono::shape): the size of its value up to a number in
/// range.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Mono {
    exp: Gauss,
    /// `(node, n)`, sorted by node: values below the range.
    nodes: Vec<(ExprId, u32)>,
    /// `(node, n)`, sorted by node: factors in range.
    factors: Vec<(ExprId, u32)>,
}

/// `Π xᵢ^nᵢ · Π yⱼ^mⱼ` of two sorted lists of `(node, power)`.
fn merge_powers(a: &[(ExprId, u32)], b: &[(ExprId, u32)]) -> Option<Vec<(ExprId, u32)>> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some(&(x, n)), Some(&(y, m))) if x == y => {
                out.push((x, n.checked_add(m)?));
                i += 1;
                j += 1;
            }
            (Some(&(x, n)), Some(&(y, _))) if x < y => {
                out.push((x, n));
                i += 1;
            }
            (_, Some(&t)) => {
                out.push(t);
                j += 1;
            }
            (Some(&t), None) => {
                out.push(t);
                i += 1;
            }
            (None, None) => break,
        }
    }
    Some(out)
}

impl Mono {
    fn one() -> Mono {
        Mono {
            exp: zero(),
            nodes: Vec::new(),
            factors: Vec::new(),
        }
    }

    fn exp(g: Gauss) -> Mono {
        Mono {
            exp: g,
            ..Mono::one()
        }
    }

    fn node(id: ExprId) -> Mono {
        Mono {
            nodes: vec![(id, 1)],
            ..Mono::one()
        }
    }

    fn factor(id: ExprId) -> Mono {
        Mono {
            factors: vec![(id, 1)],
            ..Mono::one()
        }
    }

    fn is_one(&self) -> bool {
        self.exp.is_zero() && self.nodes.is_empty() && self.factors.is_empty()
    }

    /// Only `exp` of exact arguments (or 1)?
    fn is_exp(&self) -> bool {
        self.nodes.is_empty() && self.factors.is_empty()
    }

    /// The monomial without its factors in range.
    fn shape(&self) -> Mono {
        Mono {
            exp: self.exp.clone(),
            nodes: self.nodes.clone(),
            factors: Vec::new(),
        }
    }

    fn times(&self, o: &Mono) -> Option<Mono> {
        let exp = self.exp.add(&o.exp);
        if !exp.small() {
            return None;
        }
        Some(Mono {
            exp,
            nodes: merge_powers(&self.nodes, &o.nodes)?,
            factors: merge_powers(&self.factors, &o.factors)?,
        })
    }

    fn pow(&self, n: u32) -> Option<Mono> {
        let exp = self.exp.mul(&Gauss::real(Q::from_integer(n.into())));
        if !exp.small() {
            return None;
        }
        let raise = |xs: &[(ExprId, u32)]| {
            xs.iter()
                .map(|&(x, m)| Some((x, m.checked_mul(n)?)))
                .collect::<Option<Vec<_>>>()
        };
        Some(Mono {
            exp,
            nodes: raise(&self.nodes)?,
            factors: raise(&self.factors)?,
        })
    }
}

/// `c·μ` with the value of `μ` (a number alone).
#[derive(Clone, Debug)]
struct Term {
    c: Gauss,
    mono: Mono,
    v: Scaled,
}

impl Term {
    /// `log₂` of an upper bound on `abs(c·μ)`.
    fn lg(&self) -> f64 {
        lg_gauss(&self.c) + self.v.lg_upper()
    }
}

/// A value as a polynomial in values below the range with exact complex
/// rational coefficients, `Σ cⱼ·μⱼ + r` with `abs(re r) ≤ 2^rem.re` and
/// `abs(im r) ≤ 2^rem.im` (the remainders of the series, absolute
/// `log₂`).  Sums and products are exact on the coefficients: the terms
/// that cancel are gone, with their errors, and the next ones are the
/// value.
#[derive(Clone, Debug)]
pub(super) struct Poly {
    terms: Vec<Term>,
    rem: Bound,
}

/// `log₂` of an upper bound on `abs(c)` (`−∞` for 0).
fn lg_gauss(c: &Gauss) -> f64 {
    // `abs(n/d) < 2^(bits(n) − bits(d) + 1)`.
    let lq = |q: &Q| {
        if q.is_zero() {
            f64::NEG_INFINITY
        } else {
            q.numer().bits() as f64 - q.denom().bits() as f64 + 1.0
        }
    };
    accuracy::lsum(lq(&c.re), lq(&c.im))
}

/// `cⁿ` exactly, `None` past the size limit.
fn gauss_pow(c: &Gauss, mut n: u32) -> Option<Gauss> {
    let mut out = one();
    let mut base = c.clone();
    while n > 0 {
        if n & 1 == 1 {
            out = out.mul(&base);
            if !out.small() {
                return None;
            }
        }
        n >>= 1;
        if n > 0 {
            base = base.mul(&base);
            if !base.small() {
                return None;
            }
        }
    }
    Some(out)
}

/// The value 1.
fn unit(prec: usize) -> Scaled {
    let half = BigFloat::from_f64(0.5, prec.max(64));
    Scaled::plain((half, BigFloat::new(prec)), 1, Bound::EXACT)
}

impl Poly {
    fn zero() -> Poly {
        Poly {
            terms: Vec::new(),
            rem: Bound::EXACT,
        }
    }

    fn constant(c: Gauss) -> Poly {
        let mut p = Poly::zero();
        if !c.is_zero() {
            p.terms.push(Term {
                c,
                mono: Mono::one(),
                v: unit(64),
            });
        }
        p
    }

    /// The monomial `mono` of value `v`.
    fn atom(mono: Mono, v: &Scaled) -> Poly {
        Poly {
            terms: vec![Term {
                c: one(),
                mono,
                v: v.bare(),
            }],
            rem: Bound::EXACT,
        }
    }

    /// Is the value exactly real?
    fn is_real(&self) -> bool {
        accuracy::is_exact(self.rem.im)
            && self
                .terms
                .iter()
                .all(|t| t.c.im.is_zero() && accuracy::exactly_real(&t.v.m, t.v.eb))
    }

    /// `log₂` of an upper bound on the absolute value.
    fn lg(&self) -> f64 {
        self.terms
            .iter()
            .fold(self.rem.joint(), |l, t| accuracy::lsum(l, t.lg()))
    }

    /// Add the term `t` (`None` past the limits).
    fn push(&mut self, t: Term) -> Option<()> {
        if let Some(i) = self.terms.iter().position(|u| u.mono == t.mono) {
            let c = self.terms[i].c.add(&t.c);
            if !c.small() {
                return None;
            }
            if c.is_zero() {
                self.terms.remove(i);
            } else {
                self.terms[i].c = c;
            }
        } else if !t.c.is_zero() {
            if self.terms.len() >= MAX_TERMS {
                return None;
            }
            self.terms.push(t);
        }
        Some(())
    }

    fn plus(mut self, o: Poly) -> Option<Poly> {
        for t in o.terms {
            self.push(t)?;
        }
        self.rem = Bound {
            re: accuracy::lsum(self.rem.re, o.rem.re),
            im: accuracy::lsum(self.rem.im, o.rem.im),
        };
        Some(self)
    }

    fn scaled(mut self, c: &Gauss) -> Option<Poly> {
        if c.is_zero() {
            return Some(Poly::zero());
        }
        for t in &mut self.terms {
            t.c = t.c.mul(c);
            if !t.c.small() {
                return None;
            }
        }
        let lc = lg_gauss(c);
        self.rem = if c.im.is_zero() {
            Bound {
                re: accuracy::shift(self.rem.re, lc),
                im: accuracy::shift(self.rem.im, lc),
            }
        } else {
            Bound::both(accuracy::shift(self.rem.joint(), lc))
        };
        Some(self)
    }

    fn negated(mut self) -> Poly {
        for t in &mut self.terms {
            t.c = t.c.neg();
        }
        self
    }

    /// The conjugate, when every monomial is an `exp` (or 1).
    fn conjugated(mut self) -> Option<Poly> {
        for t in &mut self.terms {
            if !t.mono.is_exp() {
                return None;
            }
            t.c.im = -t.c.im.clone();
            t.mono.exp.im = -t.mono.exp.im.clone();
            t.v = conjugated(t.v.bare());
        }
        Some(self)
    }

    /// The constant term and the rest.
    fn split_constant(mut self) -> (Gauss, Poly) {
        match self.terms.iter().position(|t| t.mono.is_one()) {
            Some(i) => {
                let t = self.terms.remove(i);
                (t.c, self)
            }
            None => (zero(), self),
        }
    }

    /// The terms alone.
    fn without_rem(&self) -> Poly {
        Poly {
            terms: self.terms.clone(),
            rem: Bound::EXACT,
        }
    }
}

/// `a·b`; a product of terms below `floor`, or below the remainder the
/// product has anyway, goes into the remainder.
fn poly_mul(a: &Poly, b: &Poly, floor: ErrExp, prec: usize, rm: RoundingMode) -> Option<Poly> {
    let real = a.is_real() && b.is_real();
    // `(A + ra)(B + rb) − AB` is bounded by `abs(a)·rb + abs(b)·ra`.
    let rem = accuracy::lsum(
        accuracy::shift(b.rem.joint(), a.lg()),
        accuracy::shift(a.rem.joint(), b.lg()),
    );
    let cut = floor.max(rem);
    let mut out = Poly::zero();
    let mut dropped = accuracy::EXACT;
    for ta in &a.terms {
        for tb in &b.terms {
            let l = ta.lg() + tb.lg();
            if l < cut {
                dropped = accuracy::lsum(dropped, l);
                continue;
            }
            let c = ta.c.mul(&tb.c);
            if !c.small() {
                return None;
            }
            let v = if ta.mono.is_one() {
                tb.v.clone()
            } else if tb.mono.is_one() {
                ta.v.clone()
            } else {
                product(&[ta.v.clone(), tb.v.clone()], prec, rm)?
            };
            out.push(Term {
                c,
                mono: ta.mono.times(&tb.mono)?,
                v,
            })?;
        }
    }
    let total = accuracy::lsum(rem, dropped);
    out.rem = if real {
        Bound::real(total)
    } else {
        Bound::both(total)
    };
    Some(out)
}

/// The value of the term `c·μ` as a number.
fn term_value(t: &Term, prec: usize, rm: RoundingMode) -> Option<Scaled> {
    Some(if t.c == one() {
        t.v.clone()
    } else if t.c == one().neg() {
        negated(t.v.clone())
    } else {
        let (cv, cb) = exact::to_value(&t.c, prec, rm);
        product(&[normalized(&cv, cb)?, t.v.clone()], prec, rm)?
    })
}

/// The terms of `p` as numbers, with their shapes ([`Mono::shape`]).
fn shaped_terms(p: &Poly, prec: usize, rm: RoundingMode) -> Option<Vec<(Option<Mono>, Scaled)>> {
    p.terms
        .iter()
        .map(|t| Some((Some(t.mono.shape()), term_value(t, prec, rm)?)))
        .collect()
}

/// Record in `ab` what a value that becomes a number alone, with the error
/// bound `2^radius` (absolute), loses of its terms (`shaped`) ([`Absorbed`]):
/// the sum of the terms of each shape (`None`: a shape of its own) that is
/// certainly not 0 and smaller than that bound, and each term certainly not
/// 0 and smaller than the error of the sum of its shape.  Terms of one
/// shape differ by factors in range, so they may cancel numerically — a zero
/// ball, not a loss: `e^(−10¹⁰)·(sin²1 + cos²1) − e^(−10¹⁰)` is 0 to the
/// precision, while `sin(e^(−10¹⁰))·(sin²1 + cos²1) − e^(−10¹⁰)` loses
/// `−(sin²1 + cos²1)·e^(−3·10¹⁰)/6`, of another shape, and
/// `e^(−10¹⁰)·(sin²1 + cos²1) − e^(−10¹⁰) + 10⁻⁷⁰⁰·e^(−10¹⁰)` the
/// last term, of the same shape but below the error of the first two.
fn note_lost(
    shaped: Vec<(Option<Mono>, Scaled)>,
    radius: ErrExp,
    prec: usize,
    rm: RoundingMode,
    ab: &mut Option<Absorbed>,
) {
    if !radius.is_finite() {
        return;
    }
    let mut groups: Vec<(Option<Mono>, Vec<Scaled>)> = Vec::new();
    for (shape, v) in shaped {
        let at = shape
            .as_ref()
            .and_then(|m| groups.iter().position(|(g, _)| g.as_ref() == Some(m)));
        match at {
            Some(i) => groups[i].1.push(v),
            None => groups.push((shape, vec![v])),
        }
    }
    for (_, vs) in groups {
        let Some(g) = add_terms(&vs, Bound::EXACT, prec, rm) else {
            continue;
        };
        if g.nonzero().is_some() {
            Absorbed::note(ab, g.lg_value(), radius);
        }
        if vs.len() > 1 {
            let within = g.eb.joint() + g.k as f64;
            for v in &vs {
                if v.nonzero().is_some() {
                    Absorbed::note(ab, v.lg_value(), within);
                }
            }
        }
    }
}

/// Record in `ab` what the value `s` below the range — the value of the
/// root, read as the number `m·2^k` — hides of its polynomial below its error
/// ([`note_lost`]): `e^(−10¹⁰)·(sin²1 + cos²1) − e^(−10¹⁰) + e^(−2·10¹⁰)`
/// is a zero ball at the scale of `e^(−10¹⁰)` with the term `e^(−2·10¹⁰)`.
pub(super) fn note_hidden(s: &Scaled, prec: usize, rm: RoundingMode, ab: &mut Option<Absorbed>) {
    if let Some(poly) = &s.poly
        && let Some(shaped) = shaped_terms(poly, prec, rm)
    {
        note_lost(shaped, s.eb.joint() + s.k as f64, prec, rm, ab);
    }
}

/// The value of the polynomial `p` as a number (SymPy's `add_terms` over
/// its terms, [`add_terms`]), `p` attached.
fn numeric(p: Poly, prec: usize, rm: RoundingMode) -> Option<Scaled> {
    let mut values = Vec::with_capacity(p.terms.len());
    for t in &p.terms {
        values.push(term_value(t, prec, rm)?);
    }
    let s = add_terms(&values, p.rem, prec, rm)?;
    Some(Scaled {
        poly: Some(Box::new(p)),
        ..s
    })
}

/// The outcome of a node whose value is the polynomial `full`: `p + d` for
/// a constant term `p ≠ 0` ([`Outcome::Near`]), otherwise `d` in range or
/// below it ([`finish`]).  `None` — the ordinary placeholder stands, and is
/// refused as "not known to be 0" — when the terms cancel to within the
/// remainder of the series: `sin(e^(−10¹⁰)) − e^(−10¹⁰) + e^(−3·10¹⁰)/6 −
/// e^(−5·10¹⁰)/120` is `−e^(−7·10¹⁰)/5040 + …`, beyond the fifth order.
fn settle_poly(
    full: Poly,
    prec: usize,
    rm: RoundingMode,
    ab: &mut Option<Absorbed>,
) -> Option<Outcome> {
    let (p, rest) = full.split_constant();
    if !p.is_zero() {
        return near(p, rest, prec, rm);
    }
    let undecided_rem = !rest.rem.is_exact();
    let bare = undecided_rem.then(|| rest.without_rem());
    let d = numeric(rest, prec, rm)?;
    if let Some(bare) = bare
        && d.nonzero().is_none()
    {
        if bare.terms.is_empty() {
            return None;
        }
        // The terms alone: a value that the remainder hides, or a zero
        // ball no wider than the remainder (the zero search cannot shrink
        // it).
        let rem = d.poly.as_ref().map_or(accuracy::UNKNOWN, |p| p.rem.joint());
        let d0 = numeric(bare, prec, rm)?;
        if d0.nonzero().is_some() || d0.lg_upper() <= rem + 1.0 {
            return None;
        }
    }
    let poly = d.poly.clone();
    let out = finish(d, prec)?;
    // Back in range, the value is a number alone: the terms below its error
    // are lost (`sin(e^(−10¹⁰))·F/e^(−10¹⁰) = F − F·e^(−2·10¹⁰)/6 + …`).
    if let (Outcome::InRange(_, b), Some(poly)) = (&out, poly)
        && let Some(shaped) = shaped_terms(&poly, prec, rm)
    {
        note_lost(shaped, b.joint(), prec, rm, ab);
    }
    Some(out)
}

/// The product of the children of a `Mul` when one has a polynomial at
/// least, `None` otherwise.  An exact factor scales the coefficients; any
/// other factor with an ordinary value is a monomial of its own (`F` in
/// `F·sin(e^(−10¹⁰)) = F·e^(−10¹⁰) − F/6·e^(−3·10¹⁰) + …`), so that the
/// terms of the other factors stay apart.  Before 0.35 such a product was a
/// number alone: `sin(e^(−10¹⁰))·(sin²1 + cos²1) − e^(−10¹⁰)` (truly
/// `−e^(−3·10¹⁰)/6`) cancelled to a zero ball and printed `0`.
fn poly_product(cx: &Ctx<'_>, children: &[ExprId], prec: usize, rm: RoundingMode) -> Option<Poly> {
    let mut acc = Poly::constant(one());
    let mut any = false;
    for &c in children {
        let f = match cx.exts.get(&c) {
            Some(Ext::Beyond(s)) => poly_of(c, s),
            Some(Ext::Near(p, d)) => Poly::constant(p.clone()).plus(poly_of(c, d))?,
            None => {
                if let Some(q) = cx.exact_of(c) {
                    acc = acc.scaled(&q)?;
                    continue;
                }
                let View::Plain(s, false) = cx.view(c)? else {
                    return None;
                };
                acc = poly_mul(
                    &acc,
                    &Poly::atom(Mono::factor(c), &s),
                    accuracy::EXACT,
                    prec,
                    rm,
                )?;
                continue;
            }
        };
        any = true;
        acc = poly_mul(&acc, &f, accuracy::EXACT, prec, rm)?;
    }
    any.then_some(acc)
}

/// `v^q` for an exact `q` (`x ± ex` its value): a monomial's power — any
/// integer `1 ≤ n ≤ 1024`, or `(e^g)^q = e^(qg)` for a real `g` (the
/// principal power) — or a polynomial's integer power up to
/// [`MAX_POLY_POWER`].  `None` otherwise (the number alone).
#[allow(clippy::too_many_arguments)]
fn poly_power(
    v: &Poly,
    q: &Q,
    x: &Complex,
    ex: Bound,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Poly> {
    let single = match v.terms.as_slice() {
        [t] if v.rem.is_exact() => Some(t),
        _ => None,
    };
    if !q.is_integer() {
        let t = single?;
        if t.c != one() || !t.mono.is_exp() || !t.mono.exp.im.is_zero() {
            return None;
        }
        let g = t.mono.exp.mul(&Gauss::real(q.clone()));
        if !g.small() {
            return None;
        }
        let value = power(&t.v, Some(q), x, ex, prec, rm, cc)?;
        return Some(Poly::atom(Mono::exp(g), &value));
    }
    let n = q.to_integer().to_i64()?;
    if n < 1 {
        return None;
    }
    if let Some(t) = single
        && n <= 1024
    {
        let n32 = u32::try_from(n).ok()?;
        let value = power(&t.v, Some(q), x, ex, prec, rm, cc)?;
        return Some(Poly {
            terms: vec![Term {
                c: gauss_pow(&t.c, n32)?,
                mono: t.mono.pow(n32)?,
                v: value,
            }],
            rem: Bound::EXACT,
        });
    }
    if n > MAX_POLY_POWER {
        return None;
    }
    let mut acc = v.clone();
    for _ in 1..n {
        acc = poly_mul(&acc, v, accuracy::EXACT, prec, rm)?;
    }
    Some(acc)
}

/// A power series `f(V) − f(0) = Σ cₙ·Vⁿ` at 0, to the order below `next`;
/// every coefficient beyond is at most 1 in absolute value.
struct Series {
    coeffs: &'static [(u32, i64, i64)],
    next: u32,
}

const EXPM1: Series = Series {
    coeffs: &[(1, 1, 1), (2, 1, 2), (3, 1, 6), (4, 1, 24), (5, 1, 120)],
    next: 6,
};
const LOG1P: Series = Series {
    coeffs: &[(1, 1, 1), (2, -1, 2), (3, 1, 3), (4, -1, 4), (5, 1, 5)],
    next: 6,
};
const COSM1: Series = Series {
    coeffs: &[(2, -1, 2), (4, 1, 24)],
    next: 6,
};
const COSHM1: Series = Series {
    coeffs: &[(2, 1, 2), (4, 1, 24)],
    next: 6,
};
const SIN: Series = Series {
    coeffs: &[(1, 1, 1), (3, -1, 6), (5, 1, 120)],
    next: 7,
};
const SINH: Series = Series {
    coeffs: &[(1, 1, 1), (3, 1, 6), (5, 1, 120)],
    next: 7,
};
const TAN: Series = Series {
    coeffs: &[(1, 1, 1), (3, 1, 3), (5, 2, 15)],
    next: 7,
};
const TANH: Series = Series {
    coeffs: &[(1, 1, 1), (3, -1, 3), (5, 2, 15)],
    next: 7,
};
const ASIN: Series = Series {
    coeffs: &[(1, 1, 1), (3, 1, 6), (5, 3, 40)],
    next: 7,
};
const ASINH: Series = Series {
    coeffs: &[(1, 1, 1), (3, -1, 6), (5, 3, 40)],
    next: 7,
};
const ATAN: Series = Series {
    coeffs: &[(1, 1, 1), (3, -1, 3), (5, 1, 5)],
    next: 7,
};
const ATANH: Series = Series {
    coeffs: &[(1, 1, 1), (3, 1, 3), (5, 1, 5)],
    next: 7,
};

/// `f(V) − f(0)` for `V = v` (no constant term) by the series `s`.
fn series(v: &Poly, s: &Series, prec: usize, rm: RoundingMode) -> Option<Poly> {
    let coeffs: Vec<(u32, Gauss)> = s.coeffs.iter().map(|&(n, a, b)| (n, ratio(a, b))).collect();
    series_with(v, &coeffs, Some((s.next, 0.0)), 1.0, prec, rm)
}

/// `Σ cₙ·Vⁿ` for `V = v` (no constant term, `abs(V) < 2⁻⁶⁴`), with the
/// remainder: for `tail = Some((N, λ))` the terms from `Vᴺ` on, at most
/// `2·2^λ·abs(V)ᴺ` (their coefficients at most `2^λ`, decreasing geometric
/// against `abs(V)`; `None`: the series ends), and `v`'s own remainder
/// through `abs(f′) ≤ 2^lf1` near 0.
fn series_with(
    v: &Poly,
    coeffs: &[(u32, Gauss)],
    tail: Option<(u32, f64)>,
    lf1: f64,
    prec: usize,
    rm: RoundingMode,
) -> Option<Poly> {
    let l = v.lg();
    // (A NaN bound is not small either.)
    let small = l < -64.0;
    if !small || v.terms.iter().any(|t| t.mono.is_one()) {
        return None;
    }
    let cut = match tail {
        Some((n, lc)) => 1.0 + lc + f64::from(n) * l,
        None => accuracy::EXACT,
    };
    let real = v.is_real() && coeffs.iter().all(|(_, c)| c.im.is_zero());
    let bare = v.without_rem();
    let mut out = Poly::zero();
    let mut pow = bare.clone();
    let mut n = 1;
    for (order, c) in coeffs {
        while n < *order {
            pow = poly_mul(&pow, &bare, cut, prec, rm)?;
            n += 1;
        }
        out = out.plus(pow.clone().scaled(c)?)?;
    }
    let r = accuracy::lsum(cut, accuracy::shift(v.rem.joint(), lf1));
    let r = if real { Bound::real(r) } else { Bound::both(r) };
    out.rem = Bound {
        re: accuracy::lsum(out.rem.re, r.re),
        im: accuracy::lsum(out.rem.im, r.im),
    };
    Some(out)
}

/// `(1 + V)^q − 1` for `V = v` and an exact `q`: the binomial series to the
/// fifth order (the principal power: `Log(1 + V)` is analytic at `V = 0`).
fn binomial(v: &Poly, q: &Q, prec: usize, rm: RoundingMode) -> Option<Poly> {
    let mut coeffs = Vec::new();
    let mut c = Q::one();
    for n in 1..=6u32 {
        c = c * (q - Q::from_integer((n - 1).into())) / Q::from_integer(n.into());
        if n <= 5 && !c.is_zero() {
            coeffs.push((n, Gauss::real(c.clone())));
        }
    }
    // `C(q, n + 1)/C(q, n) = (q − n)/(n + 1)`: beyond the sixth term the
    // coefficients grow at most by `abs(q) + 1` per order, so the tail is
    // geometric while `(abs(q) + 1)·abs(V) ≤ ½`.
    let lq = lg_gauss(&Gauss::real(q.abs() + Q::one()));
    let small = v.lg() + lq < -64.0;
    if !small {
        return None;
    }
    let tail = (!c.is_zero()).then(|| (6, lg_gauss(&Gauss::real(c))));
    // `abs(f′) = abs(q)·abs(1 + ξ)^(q − 1) ≤ 2·abs(q)` near 0.
    series_with(v, &coeffs, tail, 1.0 + lq, prec, rm)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf(x: f64) -> BigFloat {
        BigFloat::from_f64(x, 128)
    }

    /// `e^(−10¹⁰)` scaled: mpmath `mp.dps=30; exp(-mpf(10)**10)` =
    /// `9.27858442032487257807314229893e-4342944820`; its `log₂` is
    /// `−1.4426950408889634·10¹⁰`.
    #[test]
    fn exp_of_minus_ten_to_the_ten() {
        let mut cc = Consts::new().unwrap();
        let z = (bf(-1e10), BigFloat::new(128));
        let s = exp_scaled(&z, Bound::EXACT, 128, RoundingMode::ToEven, &mut cc).unwrap();
        let lg = (accuracy::lg_abs(&s.m) - accuracy::mag(&s.m).unwrap() as f64)
            + (accuracy::mag(&s.m).unwrap() + s.k) as f64;
        assert!((lg / -1.4426950408889634e10 - 1.0).abs() < 1e-15, "{lg}");
        assert!(s.eb.re < -120.0, "{:?}", s.eb);
        assert!(accuracy::is_exact(s.eb.im));
        assert_eq!(s.nonzero(), Some(Nonzero::Positive));
    }
}
