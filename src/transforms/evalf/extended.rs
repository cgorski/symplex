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
//! | `exp z` | `1 + s` ([`Ext::Near`]) | `e^r·2^k`, `k = round(re z/ln 2)`, `r = z − k·ln 2` |
//! | `x·y·…` | the mantissas multiplied, the scales added | the same |
//! | `b^q` | `mⁿ·2^(nk)` for an integer `abs(n) ≤ 1024`, else `exp(q·(Log m + k·ln 2))` | `exp(q·Log b)` |
//! | `a + b + …` | SymPy's `add_terms`: the largest terms at a common scale, the terms more than `prec + 64` bits below them in the error, an exact cancellation of the largest moving on to the next | — |
//! | `ln z` | `Log m + k·ln 2`, in range; `ln(1 + s) = s·(1 + O(s))` | — |
//! | `sin`, `tan`, `asin`, `atan` (and hyperbolic), `erf` | `s`, `2s/√π` (they vanish simply at 0) | — |
//! | `cos`, `cosh` | `1 ∓ s²/2` | — |
//! | `−z`, `conj`, `re`, `im`, `abs` | of the mantissa | — |
//! | `Ei(x)`, `erfc(x)` | — | the asymptotic series times `exp` at its scale |
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
//! anything that would come back from a scaled value above the range.  Any
//! other node sees the placeholder: an underflow its own rules handle
//! (`accuracy::underflow_nonzero`), or an argument whose value is unknown.
//! A function of a value that underflowed without a scaled form (`K₀(10¹⁰)`,
//! `Ai(10¹⁰)`) has none either ([`View::Lost`]): in a sum it is an error
//! term of the size of the underflow bound.

use astro_float::{BigFloat, Consts, RoundingMode};
use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::bigcomplex::{Complex, c_div, c_one, c_powi, c_zero};
use crate::base::node::{ExprId, ExprNode};

use super::accuracy::{self, Bound, ErrExp, Nonzero};
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
        None => Some(Scaled {
            m: v.clone(),
            k: 0,
            eb: b,
        }),
        Some(t) => {
            let (m, eb) = scale_with_loss(v, b, -t);
            Some(Scaled { m, k: t, eb })
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
            Some(Outcome::Beyond(Scaled { m, k: total, eb }))
        }
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
/// extended value, or its ordinary value underflowed (`underflowed`)?
pub(super) fn wanted(arena: &Arena, id: ExprId, underflowed: bool, exts: &ExtMap) -> bool {
    underflowed
        || (!exts.is_empty()
            && arena
                .node(id)
                .children()
                .iter()
                .any(|c| exts.contains_key(c)))
}

/// The extended evaluation of node `id` (see the module documentation), or
/// `None` when the ordinary value stands.  `ordinary` is the node's
/// ordinary value, when it has one; `underflowed`: it is 0 with an
/// underflow bound.
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
            View::Plain(s, true) => Some(one_plus(first_order(s, 1))),
            View::Plain(..) if underflowed => {
                let (z, b) = cx.ordinary(*c)?;
                if !z.1.is_zero() {
                    super::check_trig_arg(&z.1, 0, prec, arena).ok()?;
                }
                finish(exp_scaled(z, b, prec, rm, cc)?, prec)
            }
            _ => None,
        },
        ExprNode::Ln(c) => match cx.view(*c)? {
            View::Plain(s, true) => {
                let (v, b) = ln_scaled(&s, prec, rm, cc)?;
                Some(Outcome::InRange(v, b))
            }
            View::Near(p, d) if p == one() => finish(first_order(d, 1), prec),
            _ => None,
        },
        ExprNode::Mul(children) => {
            let mut factors = Vec::with_capacity(children.len());
            let mut beyond = false;
            for &c in children.iter() {
                let (s, b) = cx.plain_or_near(c)?;
                beyond |= b;
                factors.push(s);
            }
            if !beyond && !underflowed {
                return scaled_near(&cx, children, prec, rm);
            }
            finish(product(&factors, prec, rm)?, prec)
        }
        ExprNode::Pow(b, e) => {
            if exts.contains_key(e) {
                return None;
            }
            let (x, ex) = cx.ordinary(*e)?;
            match cx.view(*b)? {
                View::Plain(s, beyond) if beyond || underflowed => {
                    finish(power(&s, arena.as_num(*e), x, ex, prec, rm, cc)?, prec)
                }
                // `(1 + d)^x = 1 + x·d·(1 + O(d))`.
                View::Near(p, d) if p == one() => {
                    let xd = product(&[d, normalized(x, ex)?], prec, rm)?;
                    Some(one_plus(first_order(xd, 1)))
                }
                _ => None,
            }
        }
        ExprNode::Add(children) => sum_node(&cx, children, prec, rm),
        ExprNode::Neg(c) => match cx.view(*c)? {
            View::Plain(s, true) => finish(negated(s), prec),
            View::Near(p, d) => Some(Outcome::Near(p.neg(), negated(d))),
            _ => None,
        },
        ExprNode::Conjugate(c) => match cx.view(*c)? {
            View::Plain(s, true) => finish(conjugated(s), prec),
            View::Near(p, d) => Some(Outcome::Near(
                Gauss {
                    re: p.re,
                    im: -p.im,
                },
                conjugated(d),
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
            finish(
                Scaled {
                    m: z,
                    k: s.k,
                    eb: Bound::real(eb),
                },
                prec,
            )
        }
        // Odd functions with `f′(0) = 1` (`2/√π` for `erf`): `f(s) = c·s·(1 + O(s²))`.
        ExprNode::Sin(c)
        | ExprNode::Tan(c)
        | ExprNode::Sinh(c)
        | ExprNode::Tanh(c)
        | ExprNode::Asin(c)
        | ExprNode::Atan(c)
        | ExprNode::Asinh(c)
        | ExprNode::Atanh(c)
        | ExprNode::Erf(c) => {
            let View::Plain(s, true) = cx.view(*c)? else {
                return None;
            };
            let s = first_order(s, 2);
            if matches!(arena.node(id), ExprNode::Erf(_)) {
                let wp = prec + 16;
                let pi = cc.pi(wp, rm).clone();
                let two_over_sqrt_pi = BigFloat::from_i32(2, wp).div(&pi.sqrt(wp, rm), wp, rm);
                let f = normalized(
                    &(two_over_sqrt_pi, BigFloat::new(wp)),
                    Bound::real(-(prec as f64) - 8.0),
                )?;
                return finish(product(&[s, f], prec, rm)?, prec);
            }
            finish(s, prec)
        }
        // `cos s = 1 − s²/2·(1 + O(s²))`, `cosh s = 1 + s²/2·(…)`.
        ExprNode::Cos(c) | ExprNode::Cosh(c) => {
            let View::Plain(s, true) = cx.view(*c)? else {
                return None;
            };
            let mut sq = product(&[s.clone(), s], prec, rm)?;
            sq.k = sq.k.checked_sub(1)?;
            if matches!(arena.node(id), ExprNode::Cos(_)) {
                sq = negated(sq);
            }
            Some(one_plus(first_order(sq, 1)))
        }
        ExprNode::Ei(c) if underflowed => {
            let (x, b) = cx.ordinary(*c)?;
            if !accuracy::exactly_real(x, b) || !x.0.is_negative() {
                return None;
            }
            finish(ei_scaled(&x.0, b.re, prec, rm, cc)?, prec)
        }
        ExprNode::Erfc(c) if underflowed => {
            let (x, b) = cx.ordinary(*c)?;
            if !accuracy::exactly_real(x, b) || !super::bf_strictly_positive(&x.0) {
                return None;
            }
            finish(erfc_scaled(&x.0, b.re, prec, rm, cc)?, prec)
        }
        _ => None,
    }
}

/// `c·(p + d) = c·p + c·d` for one [`Ext::Near`] factor and exact
/// rational ones (`−cos(e^(−10¹⁰))`, the canonical form of a
/// subtraction, `5/3·cos(…)`), so that the sum it enters still sees `c·d`.
fn scaled_near(
    cx: &Ctx<'_>,
    children: &[ExprId],
    prec: usize,
    rm: RoundingMode,
) -> Option<Outcome> {
    let mut near: Option<(Gauss, Scaled)> = None;
    let mut c = one();
    for &ch in children {
        match cx.exts.get(&ch) {
            Some(Ext::Near(p, d)) if near.is_none() => near = Some((p.clone(), d.clone())),
            Some(_) => return None,
            None => {
                c = c.mul(&cx.exact_of(ch)?);
                if !c.small() {
                    return None;
                }
            }
        }
    }
    let (p, d) = near?;
    let cp = c.mul(&p);
    if c.is_zero() || !cp.small() {
        return None;
    }
    let (cv, cb) = exact::to_value(&c, prec, rm);
    let cd = product(&[d, normalized(&cv, cb)?], prec, rm)?;
    Some(Outcome::Near(cp, cd))
}

/// The exact 1.
fn one() -> Gauss {
    Gauss::real(crate::base::numeric::Q::from_integer(1.into()))
}

/// `1 + d` ([`Outcome::Near`]).
fn one_plus(d: Scaled) -> Outcome {
    Outcome::Near(one(), d)
}

/// `s·(1 + O(s^n))`: `s` with the relative error `abs(s)^n` added (the
/// next term of a series in a value below the range, negligible but
/// counted).
fn first_order(mut s: Scaled, n: i64) -> Scaled {
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
        ..s
    }
}

fn conjugated(s: Scaled) -> Scaled {
    Scaled {
        m: (s.m.0, s.m.1.neg()),
        ..s
    }
}

/// `Π sᵢ`: the mantissas multiplied with their bounds, the scales added.
fn product(factors: &[Scaled], prec: usize, rm: RoundingMode) -> Option<Scaled> {
    let parts: Vec<(&Complex, Bound)> = factors.iter().map(|s| (&s.m, s.eb)).collect();
    let (m, eb) = accuracy::mul_with_bound(&parts, prec, rm);
    let k = factors.iter().try_fold(0i64, |a, s| a.checked_add(s.k))?;
    Some(Scaled { m, k, eb })
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
    Some(Scaled {
        m,
        k: k.checked_add(t)?,
        eb,
    })
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
        return Some(Scaled {
            m: v,
            k: s.k.checked_mul(n)?,
            eb,
        });
    }
    let p = prec + bits(s.k) + 64;
    let (l, lb) = ln_scaled(s, p, rm, cc)?;
    let (w, wb) = accuracy::mul_with_bound(&[(&l, lb), (x, ex)], p, rm);
    exp_scaled(&w, wb, prec, rm, cc)
}

/// A sum with a term below the range (SymPy's `add_terms`, at a common
/// scale): see the module documentation.  The exact rational terms (and the
/// exact parts of [`Ext::Near`] terms) are added exactly: when they and the
/// terms below the range are all there is, the sum is `p + d` (or `d` when
/// `p` is 0 — `1 − cos(e^(−10¹⁰))`).
fn sum_node(cx: &Ctx<'_>, children: &[ExprId], prec: usize, rm: RoundingMode) -> Option<Outcome> {
    let mut beyond_terms = Vec::with_capacity(children.len());
    let mut ordinary_terms = Vec::new();
    let mut p = Gauss::real(crate::base::numeric::Q::from_integer(0.into()));
    let mut lost = Bound::EXACT;
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
            View::Plain(s, true) => beyond_terms.push(s),
            View::Plain(s, false) => ordinary_terms.push(s),
            View::Near(q, d) => {
                p = p.add(&q);
                if !p.small() {
                    return None;
                }
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
        let d = add_terms(&beyond_terms, Bound::EXACT, prec, rm)?;
        return if p.is_zero() {
            finish(d, prec)
        } else {
            Some(Outcome::Near(p, d))
        };
    }
    let mut terms = ordinary_terms;
    terms.append(&mut beyond_terms);
    if !p.is_zero() {
        let (v, b) = exact::to_value(&p, prec, rm);
        terms.push(normalized(&v, b)?);
    }
    let s = add_terms(&terms, lost, prec, rm)?;
    // A ball that holds 0 only because of a term without a scaled form says
    // nothing the placeholder does not.
    if !lost.is_exact() {
        s.nonzero()?;
    }
    finish(s, prec)
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
    Some(Scaled {
        m: (re, im),
        k,
        eb: Bound { re: ebr, im: ebi },
    })
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
    Some(Scaled {
        m,
        k: e.k.checked_add(t)?,
        eb,
    })
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
