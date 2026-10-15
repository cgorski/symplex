//! Improper integrals: whether an `f64` quadrature over an infinite range
//! can have seen the whole integral, and proofs of divergence.
//!
//! # The tail beyond `f64`
//!
//! An `f64` quadrature of a range infinite towards `+∞` samples the
//! integrand up to `x ≈ 10³⁰⁸` at most, through a compiled body whose
//! intermediate values overflow long before: `|x|/(x² + 1)` is `0` once `x²`
//! overflows (`|x| > 1.3·10¹⁵⁴`), and the bisection of the smoothed far piece
//! (`u ↦ x = 1/u − 1`) then meets a function that vanishes beyond the cut,
//! resolves it, and reports the truncated integral with a small error
//! estimate.  Before 0.47 `eval_f64` of `∫_{−∞}^{∞} |x|/((x² + 1)π) dx`,
//! which diverges like `(2/π)·ln R`, was `225.93` (`ln(1.8·10³⁰⁸)/π`), and
//! `integrate_numeric` of it the same; `∫_{−∞}^{∞} (x + 4)/(x² + 5) dx`
//! came out as its principal value, both halves cut at the same place.
//!
//! [`f64_tail_trusted`] looks where `f64` cannot: `evalf`'s node evaluator
//! (whose exponent range ends near `2^(2³¹)`) gives a certified bound on
//! `g = |x·f(x)|`, the integrand per unit of `ln x`, at `|x| = 2¹⁶, 2²⁴, 2³²,
//! 2⁶⁴, …, 2²⁰⁴⁸`.  Between two samples the mass `∫ g d(ln x)` is that of the power
//! law through them (an upper bound for a power law, `≤ g₀·min(Δ, 1/α)` at
//! decay rate `α`); beyond the last it is `g/α` at the last rate, infinite
//! when `g` does not decay.  An `f64` value is trusted when that mass beyond
//! `2¹⁶` is below its tolerance, or when the mass beyond `2¹⁰²⁴` (`> f64::MAX`)
//! is and the compiled body agrees with the certified values at the samples
//! within the `f64` range and at `2¹⁰⁰⁰` (no cut-off by overflow).
//!
//! What the integrand does beyond the last sample is not read from the
//! samples.  A sample that underflowed the exponent range bounds the value
//! there and nothing else — `e^{x − 10¹⁰⁰⁰}` is below `2^(−10³⁰⁰⁰)` at
//! every sample and diverges — and a growth set off by a constant beyond
//! `2²⁰⁴⁸` is not seen at all (`x⁻²e^{x/10⁷⁰⁰}` looks like `x⁻²` there):
//! up to 0.52 `∫₀^∞ e^{−x}·e^{x²/10¹⁴⁰⁰ − 10¹⁰} dx` and
//! `∫₁^∞ x⁻²e^{x/10⁷⁰⁰} dx`, both divergent, were `0` and `1`, and
//! `e^{x − 10¹⁰⁰⁰}` was refused only because `exp` gave no bound at its
//! samples.  So the integrand's constants must lie within `2^(±1024)`
//! (`quad::Integrand::constant_scale`: terms with constants within `2^(±S)`
//! cross over within `2^(2S)`, and the samples reach `2²⁰⁴⁸`), and the
//! exact limit of `x·f(x)` at the end ([`tail_decay`]) must be 0 when a
//! sample measured nothing (it underflowed, or was read as 0 below
//! [`NEGLIGIBLE`]), and not proved nonzero when every sample measured a
//! value (an oscillation such as `1/((1 + x⁴)(2 + cos(x/3)))` has no
//! limit for the engine, and its samples show the decay).  It stays a
//! heuristic as the quadrature's own error estimate is — a bump between
//! the last sample and the asymptotic regime made by fractional powers
//! alone (`e^{2x^(1/1000) − x^(1/999) − 10⁵}` peaks near `2^999000`), or a
//! decay that sets in only beyond `2²⁰⁴⁸`, is not seen — but the cut-off
//! by overflow, behind every wrong value of the divergence hunt, any growth
//! at the end and the features of large constants are.  An integrand
//! refused here goes to the arbitrary-precision quadrature, whose tanh-sinh
//! walk samples it out to `x ≈ 10¹¹⁰⁰⁰⁰` and refuses terms that do not
//! decay.
//!
//! # Divergence
//!
//! [`divergence`] proves that an integral the arbitrary-precision quadrature
//! refused diverges, by the comparison tests at each end and breakpoint `c`
//! (SymPy's `Integral` leaves this to `integrate`; its `evalf` returns
//! mpmath's estimate whatever it is): `∫ f` diverges near `c` when `(x −
//! c)·f(x)` — `x·f(x)` at `±∞` — tends to a nonzero or infinite limit, the
//! integrand then being comparable with `1/(x − c)`; and when `(x −
//! c)·|ln|x − c||·f(x)` (`x·ln x·f(x)`) does, comparable with `1/(x ln x)`
//! (Bertrand's test).  The one-sided limit at a finite `c` is taken at `u →
//! ∞` of `x = c ± 1/u`, as the definite integrator's own comparison test
//! does.  A limit that is not proved nonzero (0, an oscillation, a limit the
//! engine does not find) proves nothing: the refusal stays.

use super::ball::CBall;
use super::wp::{RM, Wp};
use crate::base::bigfloat;
use astro_float::{BigFloat, Consts};
use tracing::debug;

use super::accuracy::{self, ErrExp};
use super::quad::{Integrand, scale2, small};
use super::{Settled, ZeroSearch, evalf_settled};
use crate::base::arena::Arena;
use crate::base::bigcomplex::c_sub;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;

/// The working precision of the samples.
const PREC: usize = 128;

/// Binary exponents of the samples `|x| = 2^e` of [`f64_tail_trusted`]:
/// from `2¹⁶` on, before an exponential in the body overflows the exponent
/// range (`e^x` does past `x ≈ 1.5·10⁹`, and `1/cosh x` then has no value),
/// and before the compiled body of a power overflows (`x⁴⁰` past `2²⁵`).
const EXPONENTS: [i32; 9] = [16, 24, 32, 64, 128, 256, 512, 1024, 2048];

/// The sample beyond which `f64` cannot reach (`2¹⁰²⁴ > f64::MAX`).
const F64_END: i32 = 1024;

/// The last sample within the `f64` range, for the comparison.
const F64_LAST: i32 = 1000;

/// The relative difference below which the compiled body is taken to agree
/// with the certified value (`log₂`): an `f64` evaluation loses a few digits
/// at most, a cut-off all of them.
const AGREEMENT: f64 = -20.0;

/// `log₂ g` below which a sample counts as 0: far below any `f64` tolerance,
/// and above the bound of a value that underflowed the exponent range
/// (`e^{−x}` at `x = 2⁶⁴` is 0 within `2^(−2³¹)`, a bound that grows
/// with `x` by the factor `|x|` alone).
const NEGLIGIBLE: ErrExp = -1.0e5;

/// `log₂` of the mass `∫ g d(ln x)` between the samples `|x| = 2^e₀` and
/// `2^e₁`, where `log₂ g` is at most `g₀` and `g₁`, and the decay rate `α`
/// of `g` per unit of `ln x` (`≤ 0`: it does not decay; `∞`: it vanishes).
fn segment(e0: i32, g0: ErrExp, e1: i32, g1: ErrExp) -> (ErrExp, f64) {
    let span = f64::from(e1 - e0) * std::f64::consts::LN_2;
    match (accuracy::is_exact(g0), accuracy::is_exact(g1)) {
        (true, true) => (accuracy::EXACT, f64::INFINITY),
        // An exact zero at the far sample: no rate, the mass at most `g₀`
        // over the span.
        (false, true) => (g0 + span.log2(), f64::INFINITY),
        (true, false) => (g1 + span.log2(), -1.0),
        (false, false) => {
            let alpha = (g0 - g1) * std::f64::consts::LN_2 / span;
            let mass = if alpha > 0.0 {
                g0 + span.min(1.0 / alpha).log2()
            } else {
                g1 + span.log2()
            };
            (mass, alpha)
        }
    }
}

/// Can an `f64` quadrature of `body` (in `var`) over a range infinite
/// towards `+∞` (`upper`) or `−∞`, which evaluates the integrand by
/// `compiled`, have seen all of its integral beyond `|x| = 2¹⁶` but
/// `allowed`?  The samples say so ([`samples_trusted`]), and the exact
/// limit of `x·f(x)` at the end ([`tail_decay`]) is 0 when a sample
/// measured nothing (it underflowed, or fell below [`NEGLIGIBLE`]), and is
/// not proved nonzero otherwise.  See the module documentation.
pub(super) fn f64_tail_trusted(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    upper: bool,
    compiled: &dyn Fn(f64) -> f64,
    allowed: f64,
    cc: &mut Consts,
) -> bool {
    match samples_trusted(arena, body, var, upper, compiled, allowed, cc) {
        Sampled::Refused => false,
        Sampled::Measured => tail_decay(arena, body, var, upper) != TailDecay::Persists,
        Sampled::Negligible => tail_decay(arena, body, var, upper) == TailDecay::Vanishes,
    }
}

/// What the certified samples of [`samples_trusted`] say of an `f64` tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sampled {
    /// They do not bound its mass by the tolerance.
    Refused,
    /// They bound it, every sample a value (or an exact 0).
    Measured,
    /// They bound it, but some sample measured nothing: a value that
    /// underflowed, or one below [`NEGLIGIBLE`], read as 0.
    Negligible,
}

/// What the exact limit of `x·f(x)` at an infinite end says of an
/// integrand `f` ([`tail_decay`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TailDecay {
    /// The limit is 0: `f` decays faster than `1/x`.
    Vanishes,
    /// The limit is infinite or a certified nonzero number: `∫ f` diverges
    /// at that end (comparison with `1/x`).
    Persists,
    /// No limit was found, or it was not decided.
    Unknown,
}

/// The memo of [`tail_decay`]: an integral's tail is checked again at every
/// working precision and for each bound of a family of integrals of one
/// density (`P(X > x)` in a quantile search).
type DecayKey = (u64, ExprId, ExprId, bool);

thread_local! {
    static DECAYS: std::cell::RefCell<Vec<(DecayKey, TailDecay)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Verdicts kept.
const DECAY_MEMO_CAP: usize = 16;

/// The limit of `x·f(x)` as `x → +∞` (`upper`) or `−∞`, `f` being `body`
/// in `var`, computed exactly in a private arena.  What an integrand does
/// beyond the last point a quadrature sampled is not read from the values
/// there — an underflow at every one of them, a growth set off by a constant
/// beyond them — which say nothing of it: `e^{x − 10¹⁰⁰⁰}` is below
/// `2^(−10³⁰⁰⁰)` out to `x = 2²⁰⁴⁸` and diverges, and `x⁻²e^{x/10⁷⁰⁰}`
/// looks like `x⁻²` there.
fn tail_decay(arena: &Arena, body: ExprId, var: ExprId, upper: bool) -> TailDecay {
    let key = (arena.serial(), body, var, upper);
    if let Some(known) = DECAYS.with(|m| m.borrow().iter().find(|e| e.0 == key).map(|e| e.1)) {
        return known;
    }
    let decay = limit_of_x_f(arena, body, var, upper);
    debug!(
        upper,
        ?decay,
        "evalf: the integrand's decay at an infinite end"
    );
    DECAYS.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() >= DECAY_MEMO_CAP {
            m.remove(0);
        }
        m.push((key, decay));
    });
    decay
}

/// [`tail_decay`], computed.
fn limit_of_x_f(arena: &Arena, body: ExprId, var: ExprId, upper: bool) -> TailDecay {
    if walk::post_order_ids(arena, body).len() > MAX_NODES {
        return TailDecay::Unknown;
    }
    let mut pa = Arena::with_config(arena.config.clone());
    let mut map = rustc_hash::FxHashMap::default();
    let body = crate::base::compact::transfer_subtree(arena, &mut pa, body, &mut map);
    let var = crate::base::compact::transfer_subtree(arena, &mut pa, var, &mut map);
    let u = crate::transforms::subs::fresh_symbol(&mut pa, "__improper_u", &[body]);
    let at = if upper { u } else { pa.neg(u) };
    let fu = crate::transforms::subs::subs(&mut pa, body, var, at);
    let h = pa.mul(&[fu, u]);
    let h = crate::transforms::eval::eval(&mut pa, h);
    let inf = pa.infinity();
    match pa.limit_expr(h, u, inf) {
        Ok(l) if l == pa.zero() => TailDecay::Vanishes,
        Ok(l) if nonzero_limit(&pa, l) => TailDecay::Persists,
        _ => TailDecay::Unknown,
    }
}

/// [`f64_tail_trusted`] by the certified samples alone.
fn samples_trusted(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    upper: bool,
    compiled: &dyn Fn(f64) -> f64,
    allowed: f64,
    cc: &mut Consts,
) -> Sampled {
    if allowed.is_nan() || allowed < 0.0 {
        return Sampled::Refused;
    }
    // An eighth of the tolerance for each of the far mass and the
    // differences.
    let lg_allowed = allowed.log2() - 3.0;
    let mut f = Integrand::new(arena, body, var);
    // The samples must reach where the integrand's terms cross over.
    let last = f64::from(EXPONENTS[EXPONENTS.len() - 1]);
    if 2.0 * f.constant_scale(&mut Wp::new(PREC, cc)) > last {
        debug!(
            upper,
            "evalf: constants beyond the reach of the far-tail samples"
        );
        return Sampled::Refused;
    }
    let negligible = std::cell::Cell::new(false);
    // `log₂` of a bound on `|x·f(x)|` at `x = ±2^e`, and `f(x)`.
    let mut sample = |e: i32| -> Option<(ErrExp, (BigFloat, BigFloat))> {
        let x = scale2(&small(1), e);
        let x = if upper { x } else { x.neg() };
        // A power of `x` rounds at `PREC` bits by `2^(k·e − PREC)` in
        // absolute terms, which `exp(−x²)` turns into no bound at all: then
        // with as many bits as the powers have.
        let e_bits = usize::try_from(e).unwrap_or(0);
        let mut found = None;
        for prec in [PREC, 2 * e_bits + PREC, 4 * e_bits + PREC] {
            let evaluated = f.eval(&x, accuracy::EXACT, &mut Wp::new(prec, cc));
            match evaluated {
                Ok(CBall { mid: v, err: b })
                    if !(b.is_unknown()
                        || v.0.is_nan()
                        || v.1.is_nan()
                        || v.0.is_inf()
                        || v.1.is_inf()) =>
                {
                    found = Some((v, b));
                    break;
                }
                other => debug!(e, upper, prec, ?other, "evalf: a far-tail sample failed"),
            }
        }
        let (v, b) = found?;
        let size = accuracy::lsum(accuracy::lg_abs(&v), b.joint());
        let g = accuracy::shift(size, f64::from(e));
        debug!(e, upper, g, "evalf: the far tail of an f64 quadrature");
        if g < NEGLIGIBLE {
            negligible.set(true);
            return Some((accuracy::EXACT, v));
        }
        Some((g, v))
    };
    let verdict = || {
        if negligible.get() {
            Sampled::Negligible
        } else {
            Sampled::Measured
        }
    };
    let mut samples: Vec<(i32, ErrExp, (BigFloat, BigFloat))> = Vec::new();
    let mut mid = accuracy::EXACT;
    let mut far = accuracy::EXACT;
    for &e in &EXPONENTS {
        let Some((g, v)) = sample(e) else {
            return Sampled::Refused;
        };
        if let Some(&(pe, pg, _)) = samples.last() {
            let (mass, alpha) = segment(pe, pg, e, g);
            if pe < F64_END {
                mid = accuracy::lsum(mid, mass);
            } else {
                far = accuracy::lsum(far, mass);
            }
            // The rest, if `g` goes on decaying at this rate.
            let rest = if accuracy::is_exact(g) {
                accuracy::EXACT
            } else if alpha > 0.0 {
                g - alpha.log2()
            } else {
                accuracy::UNKNOWN
            };
            if accuracy::lsum(accuracy::lsum(mid, far), rest) <= lg_allowed {
                return verdict();
            }
            if e == EXPONENTS[EXPONENTS.len() - 1] {
                far = accuracy::lsum(far, rest);
            }
        }
        samples.push((e, g, v));
    }
    // The mass beyond `2¹⁶` matters.  Beyond `2¹⁰²⁴` no `f64` node lies.
    if far > lg_allowed {
        return Sampled::Refused;
    }
    // Within the `f64` range the compiled body must agree with the certified
    // values, up to a difference that carries no mass.
    let Some((g_last, v_last)) = sample(F64_LAST) else {
        return Sampled::Refused;
    };
    let span = f64::from(F64_LAST - EXPONENTS[0]) * std::f64::consts::LN_2;
    let floor = lg_allowed - span.log2();
    samples.retain(|s| s.0 < F64_END);
    samples.push((F64_LAST, g_last, v_last));
    for (e, g, v) in &samples {
        if *g <= floor {
            continue;
        }
        let t = f64::from(*e).exp2();
        let c = compiled(if upper { t } else { -t });
        if !c.is_finite() {
            return Sampled::Refused;
        }
        let d = c_sub(
            v,
            &(bigfloat::from_f64(c, 64), BigFloat::new(PREC)),
            PREC,
            RM,
        );
        let diff = accuracy::shift(accuracy::lg_abs(&d), f64::from(*e));
        if diff > (g + AGREEMENT).max(floor) {
            return Sampled::Refused;
        }
    }
    verdict()
}

/// Where the comparison tests of [`divergence`] look: a finite point
/// approached from above or from below, or an infinite end.
#[derive(Clone, Copy, Debug)]
pub(super) enum Approach {
    /// `x → c⁺`.
    Above(ExprId),
    /// `x → c⁻`.
    Below(ExprId),
    /// `x → +∞`.
    PosInf,
    /// `x → −∞`.
    NegInf,
}

/// Most nodes of an integrand whose divergence is sought (the limits are
/// symbolic).
const MAX_NODES: usize = 80;

/// A proof that `∫ body d(var)` diverges at one of `approaches` (all in
/// `arena`, a private arena the caller owns): the comparison tests of the
/// module documentation.  `None` when none is found.
pub(super) fn divergence(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    approaches: &[Approach],
) -> Option<String> {
    if walk::post_order_ids(arena, body).len() > MAX_NODES {
        return None;
    }
    let u = crate::transforms::subs::fresh_symbol(arena, "__improper_u", &[body]);
    let inf = arena.infinity();
    let one = arena.one();
    let ln_u = arena.ln(u);
    let inv_u = arena.div(one, u);
    for &approach in approaches {
        // `f` at the point `u` stands for, and the factor that makes it
        // `(x − c)·f`.
        let (fu, scale, shown) = match approach {
            Approach::PosInf => (
                crate::transforms::subs::subs(arena, body, var, u),
                u,
                "∞".to_string(),
            ),
            Approach::NegInf => {
                let neg = arena.neg(u);
                (
                    crate::transforms::subs::subs(arena, body, var, neg),
                    u,
                    "−∞".to_string(),
                )
            }
            Approach::Above(c) | Approach::Below(c) => {
                let x = match approach {
                    Approach::Above(_) => arena.add(&[c, inv_u]),
                    _ => arena.sub(c, inv_u),
                };
                let side = if matches!(approach, Approach::Above(_)) {
                    "⁺"
                } else {
                    "⁻"
                };
                (
                    crate::transforms::subs::subs(arena, body, var, x),
                    inv_u,
                    format!("{}{side}", arena.display(c)),
                )
            }
        };
        let first = arena.mul(&[fu, scale]);
        let second = arena.mul(&[fu, scale, ln_u]);
        for (h, test) in [
            (first, "comparison with 1/x"),
            (second, "comparison with 1/(x ln x)"),
        ] {
            let h = crate::transforms::eval::eval(arena, h);
            let Ok(l) = arena.limit_expr(h, u, inf) else {
                continue;
            };
            if nonzero_limit(arena, l) {
                return Some(format!(
                    "the integrand is not integrable as x → {shown} ({test}: the limit is {})",
                    arena.display(l)
                ));
            }
        }
    }
    None
}

/// Is the limit `l` infinite (`±∞`) or a constant certified nonzero?
fn nonzero_limit(arena: &Arena, l: ExprId) -> bool {
    if l == arena.infinity() || l == arena.neg_infinity() {
        return true;
    }
    if !walk::free_symbols(arena, l).is_empty() || walk::has_unevaluated(arena, l) {
        return false;
    }
    // An infinity inside a value, `nan`, a set of limit points: no number.
    let undecided = walk::post_order_ids(arena, l).iter().any(|&id| {
        let node = arena.node(id);
        node.is_set_node()
            || matches!(
                node,
                ExprNode::Infinity
                    | ExprNode::NegInfinity
                    | ExprNode::ComplexInfinity
                    | ExprNode::NaN
            )
    });
    if undecided {
        return false;
    }
    matches!(
        evalf_settled(arena, l, 16, ZeroSearch::Number),
        Ok((z, Settled::Certified)) if !(z.0.is_zero() && z.1.is_zero())
    )
}
