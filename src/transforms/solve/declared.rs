//! Declared domains of the unknown (part of [`solve`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Declared domains of the unknown
// ═══════════════════════════════════════════════════════════════════════════
//
// The public solvers answer over the declared domain of the unknown, as
// SymPy's `solve` does with `check_assumptions`: a candidate that certainly
// violates an assumption of the variable is dropped (`x` positive:
// `x² − 4 = 0` gives `[2]`, `x + 3 = 0` and `x = 0` give `[]`; `m` integer:
// `2m − 3 = 0` gives `[]`), one whose membership is not decided is kept
// (SymPy keeps those too: `x − a` for a plain `a`).  Up to 0.42 the
// assumptions of the unknown were ignored.  The internal callers of
// [`known_real_roots`] (partial fractions, singularities, residues, …) want every
// root and are not filtered.

/// The properties of a value compared with the declared assumptions of the
/// unknown ([`assumptions_hold`]).  Not `commutative`, `complex`, `finite`,
/// `hermitian`, …: every candidate is a finite complex number.
const DOMAIN_PROPS: [crate::base::assumptions::Props; 17] = {
    use crate::base::assumptions::Props;
    [
        Props::REAL,
        Props::RATIONAL,
        Props::INTEGER,
        Props::ALGEBRAIC,
        Props::TRANSCENDENTAL,
        Props::IRRATIONAL,
        Props::IMAGINARY,
        Props::POSITIVE,
        Props::NEGATIVE,
        Props::NONNEGATIVE,
        Props::NONPOSITIVE,
        Props::ZERO,
        Props::NONZERO,
        Props::EVEN,
        Props::ODD,
        Props::PRIME,
        Props::COMPOSITE,
    ]
};

/// The declared assumptions of `var`, closed under the implication rules,
/// when they restrict its values (`None` for a plain symbol, one declared
/// only `complex`, `finite`, …, and for a non-symbol).
pub(crate) fn declared_domain(
    arena: &Arena,
    var: ExprId,
) -> Option<crate::base::assumptions::Assumptions> {
    let ExprNode::Symbol(sid) = *arena.node(var) else {
        return None;
    };
    let mut declared = arena.symbol_assumptions(sid);
    declared.forward_chain();
    DOMAIN_PROPS
        .iter()
        .any(|&p| declared.query(p).is_some())
        .then_some(declared)
}

/// The signs of the real and imaginary parts of a constant from a
/// certified evaluation: one atom each ([`SignSet::ZERO`]: the part is
/// exactly 0), or [`SignSet::REAL`] for both parts when the evaluation is
/// not certified (a part is a real number of unknown sign).  Up to 0.49 two
/// `Option<Ordering>`s, read by `== Some(Less)` and `matches!(…, Some(Less |
/// Greater))` at each use.
///
/// [`SignSet::ZERO`]: crate::base::truth::SignSet::ZERO
/// [`SignSet::REAL`]: crate::base::truth::SignSet::REAL
#[derive(Clone, Copy)]
pub(super) struct PartSigns {
    pub(super) re: crate::base::truth::SignSet,
    pub(super) im: crate::base::truth::SignSet,
}

impl PartSigns {
    /// Certainly not real: a certified nonzero imaginary part.
    pub(super) fn nonreal(self) -> bool {
        self.im.is_nonzero().is_true()
    }

    /// Certainly not 0: a certified nonzero part.
    fn nonzero(self) -> bool {
        self.nonreal() || self.re.is_nonzero().is_true()
    }

    /// Certainly 0: both parts certified 0.
    fn zero(self) -> bool {
        self.re.is_zero().is_true() && self.im.is_zero().is_true()
    }

    /// Are the signs certified?
    pub(super) fn certified(self) -> bool {
        self.re.is_single() && self.im.is_single()
    }
}

/// Strict sign of a `BigFloat`: a zero of either sign is `Equal`.
fn bigfloat_sign(x: &astro_float::BigFloat) -> std::cmp::Ordering {
    if x.is_zero() {
        std::cmp::Ordering::Equal
    } else if x.is_negative() {
        std::cmp::Ordering::Less
    } else {
        std::cmp::Ordering::Greater
    }
}

pub(super) fn part_signs(arena: &Arena, value: ExprId) -> PartSigns {
    use crate::base::truth::SignSet;
    use crate::transforms::evalf::{Settled, ZeroSearch};
    match crate::transforms::evalf::evalf_settled(arena, value, ROOT_CHECK_DIGITS, ZeroSearch::Cap)
    {
        Ok((z, Settled::Certified)) => PartSigns {
            re: SignSet::from_ordering(bigfloat_sign(&z.0)),
            im: SignSet::from_ordering(bigfloat_sign(&z.1)),
        },
        _ => PartSigns {
            re: SignSet::REAL,
            im: SignSet::REAL,
        },
    }
}

/// Is the constant `value` certainly not an integer: not real, or at a
/// certified nonzero distance below 1 from the integer `k` nearest to a
/// 16-digit approximation (so strictly between `k − 1` and `k + 1`, and
/// not `k`)?
fn certainly_not_integer(arena: &mut Arena, value: ExprId, signs: PartSigns) -> bool {
    if signs.nonreal() {
        return true;
    }
    let Ok(z) = crate::transforms::evalf::evalf_complex64(arena, value) else {
        return false;
    };
    if !z.re.is_finite() || z.re.abs() > 1e15 {
        return false;
    }
    let k = arena.int(z.re.round() as i64);
    let d = arena.sub(value, k);
    let d = crate::transforms::eval::eval(arena, d);
    if let Some(r) = arena.as_num(d) {
        return !r.is_integer();
    }
    if !part_signs(arena, d).re.is_nonzero().is_true() {
        return false;
    }
    crate::transforms::evalf::evalf_complex64(arena, d).is_ok_and(|w| w.re.abs() < 0.99)
}

/// The rational `q` with `e = q·π`, for a constant `e`.
fn rational_multiple_of_pi(arena: &mut Arena, e: ExprId) -> Option<Q> {
    if arena.is_zero_structural(e) {
        return Some(Q::zero());
    }
    let pi = arena.pi;
    let q = arena.div(e, pi);
    let q = crate::transforms::eval::eval(arena, q);
    arena.as_num(q).cloned()
}

/// Is the constant `value` certainly transcendental: a nonzero rational
/// multiple of `π`, `e^q`, `q·ln r` or `W(q)` for nonzero rationals `q`
/// and a positive rational `r ≠ 1` (Lindemann–Weierstrass: `e^a` is
/// transcendental for algebraic `a ≠ 0`, and `W(q) = a` would make
/// `e^a = q/a` algebraic)?
fn certainly_transcendental(arena: &mut Arena, value: ExprId) -> bool {
    if rational_multiple_of_pi(arena, value).is_some_and(|q| !q.is_zero()) {
        return true;
    }
    let nonzero_rational = |arena: &Arena, e: ExprId| arena.as_num(e).is_some_and(|q| !q.is_zero());
    let log_of_rational = |arena: &Arena, e: ExprId| match *arena.node(e) {
        ExprNode::Ln(r) => arena
            .as_num(r)
            .is_some_and(|r| r.is_positive() && !r.is_one()),
        _ => false,
    };
    match arena.node(value) {
        ExprNode::Exp(q) | ExprNode::LambertW(q) => nonzero_rational(arena, *q),
        // the other branches `W_k(q)`
        ExprNode::Lib(crate::base::libfn::LibFn::LambertW, args) if args.len() == 2 => {
            nonzero_rational(arena, args[0])
        }
        ExprNode::Ln(_) => log_of_rational(arena, value),
        ExprNode::Mul(ch) if ch.len() == 2 => {
            nonzero_rational(arena, ch[0]) && log_of_rational(arena, ch[1])
        }
        _ => false,
    }
}

/// Is `e^{is}` (so `sin s`, `cos s`) algebraic: `s` a sum of rational
/// multiples of `π`, `asin q`, `acos q`, `atan q` for algebraic `q`
/// (`e^{i·asin q} = √(1 − q²) + iq`, `e^{i·atan q} = (1 + iq)/√(1 + q²)`,
/// and a rational power of an algebraic number is algebraic)?
fn exp_i_is_algebraic(arena: &mut Arena, s: ExprId) -> bool {
    let terms: Vec<ExprId> = match arena.node(s) {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => vec![s],
    };
    terms.into_iter().all(|t| {
        let atom = match arena.node(t) {
            ExprNode::Mul(ch) if ch.len() == 2 && arena.as_num(ch[0]).is_some() => ch[1],
            ExprNode::Neg(a) => *a,
            _ => t,
        };
        match *arena.node(atom) {
            ExprNode::Pi => true,
            ExprNode::Num(_) => arena.is_zero_structural(atom),
            ExprNode::Asin(q) | ExprNode::Acos(q) | ExprNode::Atan(q) => {
                arena.as_num(q).is_some()
                    || (crate::base::walk::free_symbols(arena, q).is_empty()
                        && crate::base::walk::post_order_ids(arena, q).len() <= MAX_MINPOLY_NODES
                        && crate::poly::algebraic::minimal_polynomial(arena, q).is_some())
            }
            _ => false,
        }
    })
}

/// Largest constant (in nodes) whose minimal polynomial
/// [`certainly_irrational`] looks for.
const MAX_MINPOLY_NODES: usize = 40;

/// Is the real constant `value` certainly irrational: a nonzero rational
/// multiple of `π`, or an algebraic number whose (verified) minimal
/// polynomial has degree above 1 (`√3`, `(1 + √5)/2`)?
fn certainly_irrational(arena: &mut Arena, value: ExprId) -> bool {
    if certainly_transcendental(arena, value) {
        return true;
    }
    // a root of an irreducible polynomial over ℚ of degree above 1
    if let ExprNode::RootOf(p, x, _) = *arena.node(value) {
        return polybridge::expr_to_poly(arena, p, x).is_some_and(|q| {
            q.degree().is_some_and(|d| d > 1)
                && crate::poly::factor_zassenhaus::is_irreducible_z(&q) == Some(true)
        });
    }
    if crate::base::walk::post_order_ids(arena, value).len() > MAX_MINPOLY_NODES {
        return false;
    }
    crate::poly::algebraic::minimal_polynomial(arena, value)
        .and_then(|p| p.degree())
        .is_some_and(|d| d > 1)
}

/// Does a certified evaluation show that the constant `value` lacks the
/// property `prop` (declared true of the unknown)?
fn violates_numerically(
    arena: &mut Arena,
    value: ExprId,
    prop: crate::base::assumptions::Props,
    s: PartSigns,
) -> bool {
    use crate::base::assumptions::Props;
    let zero = s.zero();
    let re_negative = s.re.is_negative().is_true();
    let re_positive = s.re.is_positive().is_true();
    let re_nonzero = s.re.is_nonzero().is_true();
    match prop {
        Props::REAL | Props::IRRATIONAL => s.nonreal(),
        Props::RATIONAL => s.nonreal() || certainly_irrational(arena, value),
        Props::ALGEBRAIC => certainly_transcendental(arena, value),
        Props::POSITIVE => s.nonreal() || re_negative || zero,
        Props::NEGATIVE => s.nonreal() || re_positive || zero,
        Props::NONNEGATIVE => s.nonreal() || re_negative,
        Props::NONPOSITIVE => s.nonreal() || re_positive,
        Props::ZERO => s.nonreal() || re_nonzero,
        Props::IMAGINARY => re_nonzero,
        // `1` is neither prime nor composite (the assumption system leaves
        // it open)
        Props::PRIME | Props::COMPOSITE
            if arena
                .as_num(value)
                .is_some_and(|r| r.is_integer() && *r <= Q::one()) =>
        {
            true
        }
        Props::INTEGER | Props::EVEN | Props::ODD | Props::PRIME | Props::COMPOSITE => {
            certainly_not_integer(arena, value, s)
        }
        _ => false,
    }
}

/// `id`, or `0` when it is a constant that is exactly 0 without being
/// written 0 (`atan(1/2) + atan(1/3) − π/4`, `cos(π/16) − √(2 + √(2 +
/// √2))/2`: the zero tests of
/// [`constant_zero_decision`](crate::transforms::eval::constant_zero_decision)).
/// Only a value whose `f64` evaluation is within `10⁻¹⁰` of 0 is tested.
pub(crate) fn exact_zero_form(arena: &mut Arena, id: ExprId) -> ExprId {
    let near_zero = crate::base::walk::free_symbols(arena, id).is_empty()
        && !arena.is_zero_structural(id)
        && crate::transforms::evalf::evalf_complex64(arena, id).is_ok_and(|z| z.norm() < 1e-10);
    if near_zero && crate::transforms::eval::constant_zero_decision(arena, id) == Some(true) {
        arena.zero
    } else {
        id
    }
}

/// Does `value` satisfy every assumption declared on the symbol `var`?
/// `Some(false)` when it certainly violates one: by the assumption system
/// (`−a` is not positive for a positive `a`, `3/2` is not an integer), or
/// for a constant by a certified evaluation (`−√2` is not positive, `√2`
/// not an integer, `asin 2` not real); `None` when it is not decided.
/// `Some(true)` for a non-symbol or a symbol without restrictions.
pub(crate) fn assumptions_hold(arena: &mut Arena, var: ExprId, value: ExprId) -> Option<bool> {
    match undecided_assumptions(arena, var, value) {
        None => Some(false),
        Some(open) if open.is_empty() => Some(true),
        Some(_) => None,
    }
}

/// The declared properties of `var` that `value` is not known to have
/// ([`assumptions_hold`]): empty when it has them all, `None` when it
/// certainly lacks one.
fn undecided_assumptions(
    arena: &mut Arena,
    var: ExprId,
    value: ExprId,
) -> Option<crate::base::assumptions::Props> {
    use crate::base::assumptions::AssumptionCache;
    let Some(declared) = declared_domain(arena, var) else {
        return Some(crate::base::assumptions::Props::empty());
    };
    let value = decision_form(arena, value);
    use crate::base::assumptions::Props;
    let mut cache = AssumptionCache::new();
    let constant = crate::base::walk::free_symbols(arena, value).is_empty()
        && !crate::base::walk::has_unevaluated(arena, value);
    // The constant whose signs decide those of a parametric `value`
    // ([`phase_constant`]), computed when first needed.
    let mut phase: Option<Option<Phase>> = None;
    let mut facts: Vec<(ExprId, PartSigns, Option<bool>)> = Vec::new();
    let mut undecided = Props::empty();
    for prop in DOMAIN_PROPS {
        let Some(want) = declared.query(prop) else {
            continue;
        };
        match cache.query(arena, value, prop) {
            Some(got) if got == want => continue,
            Some(_) => return None,
            None => {}
        }
        let sign_kind = matches!(
            prop,
            Props::REAL
                | Props::POSITIVE
                | Props::NEGATIVE
                | Props::NONNEGATIVE
                | Props::NONPOSITIVE
                | Props::ZERO
                | Props::NONZERO
                | Props::IMAGINARY
        );
        if !constant && want && implies_real(prop) && nonreal_logarithm(arena, value) {
            return None;
        }
        let subject = if constant {
            Some(value)
        } else if sign_kind {
            let ph = phase.get_or_insert_with(|| phase_of(arena, value)).clone();
            match ph {
                Some(p) if p.spread.is_zero() && !p.maybe_zero => Some(p.w),
                Some(p) => {
                    if want && arc_excludes(arena, &p, prop) {
                        return None;
                    }
                    None
                }
                None => None,
            }
        } else {
            None
        };
        if let Some(w) = subject {
            let (s, r) = match facts.iter().find(|f| f.0 == w) {
                Some(&(_, s, r)) => (s, r),
                None => {
                    let s = part_signs(arena, w);
                    let mut reals = AssumptionCache::new();
                    let r = crate::transforms::realness::constant_realness_as_declared(
                        arena,
                        w,
                        ROOT_CHECK_DIGITS,
                        &mut reals,
                    );
                    facts.push((w, s, r));
                    (s, r)
                }
            };
            if want && violates_numerically(arena, w, prop, s) {
                return None;
            }
            if holds_numerically(prop, want, s, r) {
                continue;
            }
        }
        undecided |= prop;
    }
    Some(undecided)
}

/// `value` in a form the decisions read better: `exp(q·ln m)` (rational
/// `q`, positive rational `m`) is `m^q` (`ln(x)/ln 10 = 2` gives
/// `exp(2·ln 10)`, which is `100`, odd for nobody).
fn decision_form(arena: &mut Arena, value: ExprId) -> ExprId {
    let ExprNode::Exp(arg) = *arena.node(value) else {
        return value;
    };
    let (q, ln) = match arena.node(arg) {
        ExprNode::Mul(ch) if ch.len() == 2 => match arena.as_num(ch[0]) {
            Some(q) => (q.clone(), ch[1]),
            None => return value,
        },
        ExprNode::Ln(_) => (Q::one(), arg),
        _ => return value,
    };
    let ExprNode::Ln(m) = *arena.node(ln) else {
        return value;
    };
    if !arena.as_num(m).is_some_and(|m| m.is_positive()) {
        return value;
    }
    let q = arena.num_ratio(q);
    let p = arena.pow(m, q);
    crate::transforms::eval::eval(arena, p)
}

/// Does the property imply that the value is real?
fn implies_real(prop: crate::base::assumptions::Props) -> bool {
    use crate::base::assumptions::Props;
    matches!(
        prop,
        Props::REAL
            | Props::POSITIVE
            | Props::NEGATIVE
            | Props::NONNEGATIVE
            | Props::NONPOSITIVE
            | Props::RATIONAL
            | Props::IRRATIONAL
            | Props::INTEGER
            | Props::EVEN
            | Props::ODD
            | Props::PRIME
            | Props::COMPOSITE
    )
}

/// Is `value` `q·ln B` (`q` a nonzero rational) with `B` negative, whose
/// imaginary part is `q·π`?  (`exp(x) = a` for a negative `a` gives
/// `ln a`, not real; SymPy 1.14 `log(a).is_real` is `False`.)
fn nonreal_logarithm(arena: &Arena, value: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    let ln = match arena.node(value) {
        ExprNode::Mul(ch) if ch.len() == 2 => match (arena.as_num(ch[0]), arena.node(ch[1])) {
            (Some(q), ExprNode::Ln(_)) if !q.is_zero() => ch[1],
            _ => return false,
        },
        ExprNode::Neg(inner) => *inner,
        _ => value,
    };
    let ExprNode::Ln(b) = *arena.node(ln) else {
        return false;
    };
    AssumptionCache::new().query(arena, b, Props::NEGATIVE) == Some(true)
}

/// The argument of a parametric value `|…|·w·e^{iθ}` ([`phase_of`]):
/// `θ ∈ (−spread·π, spread·π]`, and the value may be `0` (`maybe_zero`).
#[derive(Clone)]
struct Phase {
    w: ExprId,
    spread: Q,
    maybe_zero: bool,
}

/// For a `value` with parameters that is `c·∏ Bᵢ^rᵢ` (rational `rᵢ`, `c` a
/// constant, also as a sum of such terms with the same parametric part),
/// its argument from the principal branch: a positive `B` contributes `0`,
/// a negative one `r·π` (`B^r = |B|^r·e^{iπr}`), a nonnegative or
/// nonpositive one the same or the value `0`, and one of unknown sign with
/// `0 < r < 1` the range `(−rπ, rπ]` (or `0`).  With only strict signs
/// `w = c·e^{iπΣrᵢ[Bᵢ < 0]}` has the sign and realness of `value`: for a
/// negative `a`, `∛a` is not real (`w = e^{iπ/3}`; SymPy 1.14 also says
/// `False`), `(−1/2 + i√3/2)·∛a` is negative (`w = −1`); with a range,
/// [`arc_excludes`] decides what the value cannot be (`√(1 − b)` is never
/// negative, `−√(1 − b)` never positive).
fn phase_of(arena: &mut Arena, value: ExprId) -> Option<Phase> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let first = match arena.node(value) {
        ExprNode::Add(ch) => *ch.first()?,
        _ => value,
    };
    let factors: Vec<ExprId> = match arena.node(first) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![first],
    };
    let parametric: Vec<ExprId> = factors
        .into_iter()
        .filter(|&f| !crate::base::walk::free_symbols(arena, f).is_empty())
        .collect();
    if parametric.is_empty() {
        return None;
    }
    let mut cache = AssumptionCache::new();
    let mut turns = Q::zero();
    let mut spread = Q::zero();
    let mut maybe_zero = false;
    for &f in &parametric {
        let (base, r) = match *arena.node(f) {
            ExprNode::Pow(b, e) => (b, arena.as_num(e)?.clone()),
            _ => (f, Q::one()),
        };
        let known =
            |cache: &mut AssumptionCache, p: Props| cache.query(arena, base, p) == Some(true);
        if known(&mut cache, Props::NEGATIVE) {
            turns += r;
        } else if known(&mut cache, Props::POSITIVE) {
            // argument 0
        } else if known(&mut cache, Props::NONPOSITIVE) {
            // `0^r` is `0` for `r > 0`, undefined (no value) for `r < 0`
            turns += r.clone();
            maybe_zero |= r.is_positive();
        } else if known(&mut cache, Props::NONNEGATIVE) {
            maybe_zero |= r.is_positive();
        } else if r.is_positive() && r < Q::one() {
            spread += r;
            maybe_zero |= !known(&mut cache, Props::NONZERO);
        } else {
            return None;
        }
    }
    if spread >= Q::one() {
        return None;
    }
    let m = arena.mul(&parametric);
    let c = arena.div(value, m);
    let c = crate::transforms::expand::expand(arena, c);
    let c = crate::transforms::eval::eval(arena, c);
    if !crate::base::walk::free_symbols(arena, c).is_empty() {
        return None;
    }
    let turns = arena.num_ratio(turns);
    let angle = arena.mul(&[turns, arena.pi, arena.i_unit]);
    let rotation = arena.exp(angle);
    let w = arena.mul(&[c, rotation]);
    let w = crate::transforms::eval::eval(arena, w);
    Some(Phase {
        w,
        spread,
        maybe_zero,
    })
}

/// Does the argument range of `p` show that the value cannot have the
/// property `prop` (declared true of the unknown)?  The range is
/// `arg w + (−spread·π, spread·π]`; an angle within `10⁻⁹` of an end is
/// taken to be in it.
fn arc_excludes(arena: &mut Arena, p: &Phase, prop: crate::base::assumptions::Props) -> bool {
    use crate::base::assumptions::Props;
    use std::f64::consts::PI;
    if !part_signs(arena, p.w).certified() {
        return false;
    }
    let Ok(w) = crate::transforms::evalf::evalf_complex64(arena, p.w) else {
        return false;
    };
    let r = w.norm();
    if r.is_nan() || r <= 0.0 || r.is_infinite() {
        return false;
    }
    let phi = w.arg();
    let half = num_traits::ToPrimitive::to_f64(&p.spread).unwrap_or(1.0) * PI;
    // is the angle `t` (mod 2π) possibly in `(phi − half, phi + half]`?
    let reaches = |t: f64| {
        let mut d = (t - phi) % (2.0 * PI);
        if d > PI {
            d -= 2.0 * PI;
        } else if d <= -PI {
            d += 2.0 * PI;
        }
        d.abs() <= half + 1e-9
    };
    let zero_ok = p.maybe_zero;
    match prop {
        Props::POSITIVE => !reaches(0.0),
        Props::NEGATIVE => !reaches(PI),
        Props::REAL => !zero_ok && !reaches(0.0) && !reaches(PI),
        Props::NONNEGATIVE => !zero_ok && !reaches(0.0),
        Props::NONPOSITIVE => !zero_ok && !reaches(PI),
        Props::IMAGINARY => !reaches(PI / 2.0) && !reaches(-PI / 2.0),
        _ => false,
    }
}

/// Does a certified evaluation (signs `s`, realness `real`) show that a
/// constant has the property `prop` (`want`) of the sign and realness
/// kind?  `false` when it does not decide.
fn holds_numerically(
    prop: crate::base::assumptions::Props,
    want: bool,
    s: PartSigns,
    real: Option<bool>,
) -> bool {
    use crate::base::assumptions::Props;
    let is_real = real == Some(true);
    let not_real = real == Some(false) || s.nonreal();
    let pos = is_real && s.re.is_positive().is_true();
    let neg = is_real && s.re.is_negative().is_true();
    let zero = s.zero();
    let re_nonzero = s.re.is_nonzero().is_true();
    let nonzero = s.nonzero();
    match (prop, want) {
        (Props::REAL, true) => is_real,
        (Props::REAL, false) => not_real,
        (Props::POSITIVE, true) => pos,
        (Props::POSITIVE, false) => neg || zero || not_real,
        (Props::NEGATIVE, true) => neg,
        (Props::NEGATIVE, false) => pos || zero || not_real,
        (Props::NONNEGATIVE, true) => pos || zero,
        (Props::NONNEGATIVE, false) => neg || not_real,
        (Props::NONPOSITIVE, true) => neg || zero,
        (Props::NONPOSITIVE, false) => pos || not_real,
        (Props::NONZERO, true) | (Props::ZERO, false) => nonzero,
        (Props::IMAGINARY, false) => is_real || re_nonzero,
        _ => false,
    }
}

/// The candidates in `values` that do not certainly violate the declared
/// assumptions of `var` ([`assumptions_hold`]): an undecided one with
/// parameters is `Err` when its membership may fail on a set of parameter
/// values of positive measure ([`in_domain_only_for_some_parameters`]),
/// and kept otherwise; an undecided constant is `Err` with the reason —
/// one that is 0 to the precision of the zero search
/// ([`zero_to_precision`]), or whose membership no exact or certified test
/// decides.  Up to 0.50 the latter was kept: a real `x` got the roots
/// `ln(⅕·4^(1/4)·(±√(8√2 − 4·4^(3/4)) − 2·2^(3/4)))/ln 2` of `4^(x + 3/4) +
/// 2·2^(x + 3/4) + 1 = 0`, logarithms of negative numbers whose value no
/// evaluation could settle (the radicand is 0); a root of a real unknown
/// must be proved real.
pub(crate) fn retain_admissible(
    arena: &mut Arena,
    var: ExprId,
    values: Vec<ExprId>,
) -> Result<Vec<ExprId>, String> {
    let Some(declared) = declared_domain(arena, var) else {
        return Ok(values);
    };
    let real_unknown =
        crate::base::truth::Truth::from(declared.query(crate::base::assumptions::Props::REAL))
            .is_true();
    let mut kept = Vec::with_capacity(values.len());
    for v in values {
        match assumptions_hold(arena, var, v) {
            Some(false) => {}
            Some(true) => kept.push(v),
            None => {
                if zero_to_precision(arena, v) {
                    let shown = arena.display(v).to_string();
                    let name = arena.display(var).to_string();
                    return Err(format!(
                        "cannot decide whether the solution {shown} lies in the declared domain \
                         of {name}: it is 0 to {ROOT_CHECK_DIGITS} digits, and not known to be 0"
                    ));
                }
                match constant_realness(arena, v) {
                    Some(crate::base::truth::Truth::False) if real_unknown => continue,
                    Some(crate::base::truth::Truth::Unknown) if real_unknown => {
                        let shown = arena.display(v).to_string();
                        let name = arena.display(var).to_string();
                        return Err(format!(
                            "cannot decide whether the solution {shown} is real (the declared \
                             domain of {name})"
                        ));
                    }
                    _ => {}
                }
                if in_domain_only_for_some_parameters(arena, var, v, &declared) {
                    let shown = arena.display(v).to_string();
                    let name = arena.display(var).to_string();
                    return Err(format!(
                        "the solution {shown} lies in the declared domain of {name} only for some \
                         values of the parameters, which their assumptions do not decide \
                         (solve_as_set gives the solution set)"
                    ));
                }
                kept.push(v);
            }
        }
    }
    Ok(kept)
}

/// The exact condition on its symbols under which the value `value` is
/// real, when it has the shape `A + C·u^(p/q)` with `A`, `C` and `u` real by
/// the assumptions of the symbols, `C` nonzero and `q > 1`: `u ≥ 0` (`u >
/// 0` for `p < 0`, where `u = 0` is a pole) — for `u < 0` the principal
/// power `|u|^(p/q)·e^(iπp/q)` is not real, since `p/q` is not an integer.
/// A radicand that is itself of that shape is decided the same way when
/// `|p| < q` (then `u^(p/q)` is real only for a real `u ≥ 0`): `√(a −
/// √b)` is real exactly when `b ≥ 0 ∧ a − √b ≥ 0`.  `None` for other
/// shapes (two radicals, a coefficient that may be 0).
pub(crate) fn realness_condition(arena: &mut Arena, value: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::base::truth::Truth;
    const MAX_NESTING: usize = 4;
    let mut cache = AssumptionCache::new();
    let mut known =
        |arena: &Arena, e: ExprId, p: Props| Truth::from(cache.query(arena, e, p)).is_true();
    let mut conds: Vec<ExprId> = Vec::new();
    let mut current = value;
    for _ in 0..MAX_NESTING {
        if known(arena, current, Props::REAL) {
            return Some(arena.and(&conds));
        }
        // the one term not known real, `C·u^(p/q)`
        let terms: Vec<ExprId> = match arena.node(current) {
            ExprNode::Add(ch) => ch.to_vec(),
            _ => vec![current],
        };
        let open: Vec<ExprId> = terms
            .into_iter()
            .filter(|&t| !known(arena, t, Props::REAL))
            .collect();
        let [term] = open[..] else {
            return None;
        };
        let mut factors: Vec<ExprId> = Vec::new();
        let mut stack = vec![term];
        while let Some(f) = stack.pop() {
            match arena.node(f) {
                ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
                ExprNode::Neg(inner) => stack.push(*inner),
                _ => factors.push(f),
            }
        }
        let mut radical: Option<(ExprId, Q)> = None;
        for f in factors {
            if known(arena, f, Props::REAL) && known(arena, f, Props::NONZERO) {
                continue;
            }
            let ExprNode::Pow(u, e) = *arena.node(f) else {
                return None;
            };
            let r = arena.as_num(e)?.clone();
            if r.is_integer() || radical.is_some() {
                return None;
            }
            radical = Some((u, r));
        }
        let (u, r) = radical?;
        let zero = arena.zero;
        conds.push(if r.is_positive() {
            arena.ge(u, zero)
        } else {
            arena.gt(u, zero)
        });
        if known(arena, u, Props::REAL) {
            return Some(arena.and(&conds));
        }
        if r.numer().abs() >= *r.denom() {
            return None;
        }
        current = u;
    }
    None
}

/// Is the root `value`, with parameters, in the declared domain of `var`
/// only for some parameter values, so that a list holding it would be
/// wrong on a set of positive measure?  Undecided realness, sign,
/// integrality or rationality fail there (`x² = y` for a real `x` and a
/// real `y`: `±√y` is real only for `y ≥ 0`); an undecided `≠ 0`,
/// irrationality or "not an integer" fails on a set of measure zero only
/// (the generic convention: `x − a` for a nonzero `x` gives `a`).  Up to
/// 0.52 every root with parameters whose membership was undecided was
/// returned (the radical hunt of 0.51: 829 of 16,000 solves).
fn in_domain_only_for_some_parameters(
    arena: &mut Arena,
    var: ExprId,
    value: ExprId,
    declared: &crate::base::assumptions::Assumptions,
) -> bool {
    use crate::base::assumptions::Props;
    if crate::base::walk::free_symbols(arena, value).is_empty() {
        return false;
    }
    let Some(open) = undecided_assumptions(arena, var, value) else {
        return false;
    };
    open.iter().any(|p| {
        let generic = if crate::base::truth::Truth::from(declared.query(p)).is_true() {
            Props::NONZERO | Props::IRRATIONAL | Props::TRANSCENDENTAL
        } else {
            Props::ZERO
                | Props::INTEGER
                | Props::RATIONAL
                | Props::ALGEBRAIC
                | Props::EVEN
                | Props::ODD
                | Props::PRIME
                | Props::COMPOSITE
        };
        !generic.contains(p)
    })
}

/// Is the constant `value` real: by the assumption system or a certified
/// evaluation ([`constant_realness_as_declared`]), else by the facts `eval`
/// decides from its value ([`constant_facts`], with the exact zero tests:
/// `∛(1 − √(8√2 − 4·4^(3/4)))` is `1`)?  `None` for a value with symbols
/// or an unevaluated part (not decided here).
///
/// [`constant_realness_as_declared`]: crate::transforms::realness::constant_realness_as_declared
/// [`constant_facts`]: crate::transforms::eval::constant_facts
fn constant_realness(arena: &mut Arena, value: ExprId) -> Option<crate::base::truth::Truth> {
    use crate::base::truth::Truth;
    use crate::base::walk;
    if !walk::free_symbols(arena, value).is_empty() || walk::has_unevaluated(arena, value) {
        return None;
    }
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    let by_value = Truth::from(crate::transforms::realness::constant_realness_as_declared(
        arena,
        value,
        ROOT_CHECK_DIGITS,
        &mut reals,
    ));
    Some(match by_value {
        Truth::Unknown => Truth::from(
            crate::transforms::eval::constant_facts(arena, value)
                .query(crate::base::assumptions::Props::REAL),
        ),
        decided => decided,
    })
}

/// Is the constant `value` zero to the precision of the zero search
/// without being certified zero or nonzero?  Such a member of a solution
/// list is undecided against a domain bounded at `0` (positive, negative,
/// nonzero, …) however many digits are computed: before 0.45 it was
/// kept, a root that may be `0` returned for a positive unknown.
fn zero_to_precision(arena: &mut Arena, value: ExprId) -> bool {
    use crate::base::walk;
    use crate::transforms::evalf::{Settled, ZeroSearch};
    if !walk::free_symbols(arena, value).is_empty() || walk::has_unevaluated(arena, value) {
        return false;
    }
    matches!(
        crate::transforms::evalf::evalf_settled(arena, value, ROOT_CHECK_DIGITS, ZeroSearch::Cap),
        Ok((_, Settled::ZeroToPrecision))
    )
}

/// The solutions `values` of `expr = 0` (from `solve`) restricted to the
/// declared domain of `var` ([`retain_admissible`]).  `Err` with the reason
/// when every one of them is outside the domain while a member of a
/// periodic family through them is in it (`x` negative: `sin x = 1/2`
/// has the principal solutions `π/6, 5π/6`, and `−7π/6`): the empty list
/// would claim there is no solution, as SymPy's `solve` does.
pub(crate) fn restrict_to_domain(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    values: Vec<ExprId>,
) -> Result<Vec<ExprId>, String> {
    if values.is_empty() || declared_domain(arena, var).is_none() {
        return Ok(values);
    }
    let kept = retain_admissible(arena, var, values)?;
    if !kept.is_empty() || polybridge::expr_to_poly(arena, expr, var).is_some() {
        return Ok(kept);
    }
    let Some(PeriodicCandidates {
        families, param: n, ..
    }) = raw_families(arena, expr, var, &SolveCx::principal(Domain::Real), true)
    else {
        return Ok(kept);
    };
    for fam in families {
        for k in [1i64, -1, 2, -2, 3, -3, 4, -4] {
            let kk = arena.int(k);
            let member = crate::transforms::subs::subs(arena, fam, n, kk);
            let member = crate::transforms::eval::eval(arena, member);
            if assumptions_hold(arena, var, member) == Some(true)
                && member_is_root(arena, expr, var, member)
            {
                let shown = arena.display(member).to_string();
                let name = arena.display(var).to_string();
                return Err(format!(
                    "every principal-branch solution is outside the declared domain of {name}, \
                     but other branches lie in it (e.g. {shown}); see solve_general"
                ));
            }
        }
    }
    Ok(kept)
}

/// Has `expr = 0` no solution at all in the declared domain of `var`?  For
/// `Ex::solve` when [`restrict_to_domain`] dropped every principal root:
/// decided by the general solution over that domain
/// ([`solve_general_declared`]), complete, each family without a member in
/// the domain ([`family_in_domain`]) and each other root certainly outside
/// it.  For a real `x`, `eˣ + e^(ix) = 0` has the principal root
/// `iπ/(1 − i)` and the solutions `(2n + 1)πi/(1 − i)`, none real; up to
/// 0.49 `Ex::solve` answered `Ok([])`, which also means "no principal
/// root in the domain, other branches not looked at".  `false` for a
/// polynomial (its empty list of roots in the domain is an answer) and
/// when not decided.
pub(crate) fn none_in_domain(arena: &mut Arena, expr: ExprId, var: ExprId) -> bool {
    if declared_domain(arena, var).is_none() || polybridge::expr_to_poly(arena, expr, var).is_some()
    {
        return false;
    }
    if let Some(coeffs) = symbolic_poly_coeffs(arena, expr, var)
        && coeffs.iter().all(|&c| !expr_contains_var(arena, c, var))
    {
        return false;
    }
    let n = branch_parameter(arena, &[expr, var]);
    let roots = match solve_general_declared(arena, expr, var, n) {
        Solutions::Complete(RootSet::Empty(_)) => return true,
        complete @ Solutions::Complete(RootSet::Points(_) | RootSet::Families { .. }) => {
            complete.claimed_points()
        }
        _ => return false,
    };
    // (the parameter of the rays of `family_in_domain`, never returned)
    let k = branch_parameter(arena, &[expr, var, n]);
    for r in roots {
        let outside = if crate::base::walk::contains(arena, r, n) {
            matches!(family_in_domain(arena, var, r, n, k), FamilyInDomain::Empty)
        } else {
            assumptions_hold(arena, var, r) == Some(false)
        };
        if !outside {
            return false;
        }
    }
    true
}

/// The members of a periodic family in the declared domain of the unknown
/// ([`family_in_domain`]).
pub(crate) enum FamilyInDomain {
    /// Every member.
    All,
    /// No member.
    Empty,
    /// Exactly one member.
    Point(ExprId),
    /// The members of the rays `family(n₀ + k)` or `family(n₀ − k)`, `k ≥ 0`,
    /// written with the nonnegative integer parameter `k` given.
    Rays(Vec<ExprId>),
    /// Not decided.
    Undecided,
}

/// The members of the periodic family `fam` (linear in the integer
/// parameter `n`, which ranges over ℤ) in the declared domain of `var`:
/// all or none when the assumption system decides the family as a whole
/// (`2nπ + π/6` is real, `2nπi + i` is not), otherwise for constant
/// families and a domain on the real line given by signs (positive,
/// nonnegative, negative, nonpositive, nonzero, real): the rays of a real
/// progression cut by `0` (`x` positive: `nπ` is `π + kπ`, `k ≥ 0`), the
/// one real member of a family with a non-real step, or none for a real
/// step and a non-real offset.  `Undecided` for discrete domains (integer,
/// rational, …), symbolic families and members not certified.
pub(crate) fn family_in_domain(
    arena: &mut Arena,
    var: ExprId,
    fam: ExprId,
    n: ExprId,
    k: ExprId,
) -> FamilyInDomain {
    use crate::base::assumptions::Props;
    let Some(declared) = declared_domain(arena, var) else {
        return FamilyInDomain::All;
    };
    match assumptions_hold(arena, var, fam) {
        Some(true) => return FamilyInDomain::All,
        Some(false) => return FamilyInDomain::Empty,
        None => {}
    }
    let member = |arena: &mut Arena, j: i64| -> ExprId {
        let jj = arena.int(j);
        let m = crate::transforms::subs::subs(arena, fam, n, jj);
        crate::transforms::eval::eval(arena, m)
    };
    let s0 = member(arena, 0);
    let s1 = member(arena, 1);
    let d = arena.sub(s1, s0);
    let d = crate::transforms::eval::eval(arena, d);
    if !crate::base::walk::free_symbols(arena, s0).is_empty()
        || !crate::base::walk::free_symbols(arena, d).is_empty()
    {
        return FamilyInDomain::Undecided;
    }
    // The decisions below read the family as the progression `s₀ + n·d`;
    // one that is not (`√(2nπ + π/6)` from `sin(x²) = 1/2`, whose members
    // for `n < 0` are imaginary) is left to the intersection with the
    // domain.  Before 0.45 the real `s₀` and `d` of `√(2nπ + π/6)` made the
    // whole family real, and the solution set of `sin(x²) = 1/2` for a real
    // `x` contained `i·√(11π/6)`.
    let nd = arena.mul(&[n, d]);
    let progression = arena.add(&[s0, nd]);
    let gap = arena.sub(fam, progression);
    let gap = crate::transforms::expand::expand(arena, gap);
    let gap = crate::transforms::eval::eval(arena, gap);

    if !arena.is_zero_structural(gap) {
        return FamilyInDomain::Undecided;
    }
    // A family `s₀ + n·c·π` (`c ≠ 0` rational) with `sin s₀` algebraic in a
    // domain of algebraic numbers (integer, rational, even, prime, …):
    // every member has an algebraic sine (`sin(s₀ + θ)` for `θ ∈ ℚπ`), and
    // the sine of a nonzero algebraic number is transcendental
    // (Lindemann–Weierstrass), so `0` is the only member that can be in
    // the domain (`nπ`, `2nπ + asin(1/3)`, `nπ + atan(1/2)` for an integer
    // `x`).
    let algebraic_domain = [Props::INTEGER, Props::RATIONAL, Props::ALGEBRAIC]
        .iter()
        .any(|&p| declared.query(p) == Some(true));
    if algebraic_domain
        && let Some(c) = rational_multiple_of_pi(arena, d)
        && !c.is_zero()
        && exp_i_is_algebraic(arena, s0)
    {
        // the only index at which the member could be `0`
        let zero_at = match rational_multiple_of_pi(arena, s0) {
            Some(r) => {
                let j = -r / c;
                Some(j.is_integer())
            }
            None => match crate::transforms::evalf::evalf_complex64(arena, s0) {
                Ok(z) if z.re.is_finite() => {
                    let t = -z.re
                        / (num_traits::ToPrimitive::to_f64(&c).unwrap_or(1.0)
                            * std::f64::consts::PI);
                    let m = member(arena, t.round() as i64);
                    if arena.is_zero_structural(m) {
                        Some(true)
                    } else {
                        part_signs(arena, m).nonzero().then_some(false)
                    }
                }
                _ => None,
            },
        };
        let zero = arena.zero;
        return match zero_at {
            Some(false) => FamilyInDomain::Empty,
            Some(true) => match assumptions_hold(arena, var, zero) {
                Some(true) => FamilyInDomain::Point(zero),
                Some(false) => FamilyInDomain::Empty,
                None => FamilyInDomain::Undecided,
            },
            None => FamilyInDomain::Undecided,
        };
    }
    let discrete = [
        Props::INTEGER,
        Props::RATIONAL,
        Props::ALGEBRAIC,
        Props::IRRATIONAL,
        Props::TRANSCENDENTAL,
        Props::EVEN,
        Props::ODD,
        Props::PRIME,
        Props::COMPOSITE,
    ]
    .iter()
    .any(|&p| declared.query(p).is_some());

    // (A discrete domain on the real line is decided below only for a
    // non-real step, when at most one member is real.)
    if discrete && declared.query(Props::REAL) != Some(true) {
        return FamilyInDomain::Undecided;
    }
    let (Ok(z0), Ok(zd)) = (
        crate::transforms::evalf::evalf_complex64(arena, s0),
        crate::transforms::evalf::evalf_complex64(arena, d),
    ) else {
        return FamilyInDomain::Undecided;
    };
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    let d_real = crate::transforms::realness::constant_realness_as_declared(
        arena,
        d,
        ROOT_CHECK_DIGITS,
        &mut reals,
    );
    let s_real = crate::transforms::realness::constant_realness_as_declared(
        arena,
        s0,
        ROOT_CHECK_DIGITS,
        &mut reals,
    );
    // The index `t` of the member on the boundary (`Im = 0`, or `0`): the
    // members decided are those next to it.
    let near = |t: f64| t.is_finite() && t.abs() < 1e12;
    let holds = |p: Props| declared.query(p) == Some(true);
    let ray = |arena: &mut Arena, from: i64, up: bool| -> ExprId {
        let from = arena.int(from);
        let idx = if up {
            arena.add(&[from, k])
        } else {
            arena.sub(from, k)
        };
        let r = crate::transforms::subs::subs(arena, fam, n, idx);
        let r = crate::transforms::expand::expand(arena, r);
        crate::transforms::eval::eval(arena, r)
    };
    // The family without its member `0`: the rays on either side of it.
    let without_zero = |arena: &mut Arena| -> FamilyInDomain {
        let neg_s0 = arena.neg(s0);
        let q = arena.div(neg_s0, d);
        let q = crate::transforms::eval::eval(arena, q);
        let at = if let Some(r) = arena.as_num(q) {
            Some(r.is_integer().then(|| r.to_integer()))
        } else {
            match crate::transforms::evalf::evalf_complex64(arena, q) {
                Ok(z) if near(z.re) && near(z.im) => {
                    let j = arena.int(z.re.round() as i64);
                    let off = arena.sub(q, j);
                    let off = crate::transforms::eval::eval(arena, off);
                    part_signs(arena, off).nonzero().then_some(None)
                }
                _ => None,
            }
        };
        match at {
            Some(None) => FamilyInDomain::All,
            Some(Some(j)) => match i64::try_from(j) {
                Ok(j) => {
                    let up = ray(arena, j + 1, true);
                    let down = ray(arena, j - 1, false);
                    FamilyInDomain::Rays(vec![up, down])
                }
                Err(_) => FamilyInDomain::Undecided,
            },
            None => FamilyInDomain::Undecided,
        }
    };
    if holds(Props::IMAGINARY) || declared.query(Props::REAL) == Some(false) {
        return FamilyInDomain::Undecided;
    }
    if declared.query(Props::REAL) != Some(true) {
        // A domain in ℂ restricted only by `≠ 0`.
        let only_nonzero = DOMAIN_PROPS
            .iter()
            .all(|&p| p == Props::NONZERO || p == Props::ZERO || declared.query(p).is_none());
        return if only_nonzero && holds(Props::NONZERO) {
            without_zero(arena)
        } else {
            FamilyInDomain::Undecided
        };
    }
    if d_real == Some(false) {
        // A non-real step: at most one real member, at `n = −Im s₀ / Im d`.
        let t = -z0.im / zd.im;
        if !near(t) {
            return FamilyInDomain::Undecided;
        }
        // Every other member is at least half a step off the real line.
        let m = member(arena, t.round() as i64);
        return match assumptions_hold(arena, var, m) {
            Some(true) => FamilyInDomain::Point(m),
            Some(false) => FamilyInDomain::Empty,
            None => FamilyInDomain::Undecided,
        };
    }
    if discrete || d_real != Some(true) || !part_signs(arena, d).re.is_nonzero().is_true() {
        return FamilyInDomain::Undecided;
    }
    match s_real {
        Some(true) => {}
        Some(false) => return FamilyInDomain::Empty,
        None => return FamilyInDomain::Undecided,
    }
    // A real progression `s₀ + n·d`; the domain is a half-line, the line
    // without 0, or the line.
    let lower = holds(Props::POSITIVE) || holds(Props::NONNEGATIVE);
    let upper = holds(Props::NEGATIVE) || holds(Props::NONPOSITIVE);
    let t = -z0.re / zd.re;
    if !near(t) || (lower && upper) {
        return FamilyInDomain::Undecided;
    }
    if lower || upper {
        // The admissible members are `n ≥ n₀` or `n ≤ n₀`: find the
        // boundary between two decided neighbours near `t`.
        let increasing = (zd.re > 0.0) == lower;
        let base = t.floor() as i64;
        for j in (base - 2)..=(base + 3) {
            let (inside, outside) = if increasing { (j, j - 1) } else { (j, j + 1) };
            let mi = member(arena, inside);
            let mo = member(arena, outside);
            if assumptions_hold(arena, var, mi) == Some(true)
                && assumptions_hold(arena, var, mo) == Some(false)
            {
                return FamilyInDomain::Rays(vec![ray(arena, inside, increasing)]);
            }
        }
        return FamilyInDomain::Undecided;
    }
    if holds(Props::NONZERO) {
        return without_zero(arena);
    }
    // real (the family is real)
    FamilyInDomain::All
}
