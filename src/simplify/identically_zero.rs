//! Denominators that vanish identically only by an identity of the
//! functions in them: `tan x·cos x − sin x`, `sin²x + cos²x − 1`,
//! `cosh²x − sinh²x − 1`, `sin 2x − 2·sin x·cos x`.
//!
//! The residue test of
//! [`may_have_vanishing_denominator`](crate::poly::polybridge::may_have_vanishing_denominator)
//! takes every generator (`sin x`, `tan x`, `eˣ`) as an independent
//! indeterminate, so it finds exactly the denominators that vanish as
//! rational functions of the generators, `(x + 1)² − x² − 2x − 1`.  The
//! rest are decided here, at about the cost of one numeric evaluation per
//! denominator:
//!
//! * **not identically zero** when a certified evaluation at a sample
//!   point is nonzero (`evalf` with its error bound): decides almost every
//!   denominator at the first point;
//! * **identically zero** only when it is zero at both sample points and
//!   then shown to be exactly 0: a constant by the exact zero test,
//!   anything else as a rational function of exponentials once `sin`,
//!   `cos`, `tan` and their hyperbolic counterparts are written with
//!   `exp` (a decision procedure for that class, which `simplify` alone is
//!   not: it leaves `tanh x·cosh x − sinh x`), or by `simplify` reaching
//!   the structural 0.  A numeric zero alone proves nothing (`ln(x²) −
//!   2·ln x` is 0 on the right half-plane and `−2πi` elsewhere: the sample
//!   points lie in both half-planes, and the exact tests would not confirm
//!   it either).
//!
//! The sample points respect the declared assumptions of each symbol
//! (positive, real, integer, …); a symbol without any takes Gaussian
//! rational values off the real axis.

use std::cell::Cell;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, Zero};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::base::arena::Arena;
use crate::base::assumptions::Props;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;
use crate::simplify::ratsimp::Vanishing;

/// Digits to which a sample value is certified.
const SAMPLE_DIGITS: u32 = 20;

/// Sample points per test (one for a constant).
const SAMPLE_POINTS: u64 = 2;

/// Nesting of the `simplify` confirmations: the `simplify` of a candidate
/// denominator does not look for identically vanishing denominators of its
/// own (it still has the polynomial test of `vanishing_denominator`).
const MAX_DEPTH: u32 = 1;

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// One level of [`DEPTH`], released on drop (also when unwinding).
struct DepthGuard;

impl DepthGuard {
    fn enter() -> Option<Self> {
        DEPTH.with(|d| {
            (d.get() < MAX_DEPTH).then(|| {
                d.set(d.get() + 1);
                DepthGuard
            })
        })
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

fn nested() -> bool {
    DEPTH.with(|d| d.get() >= MAX_DEPTH)
}

/// A power with a numeric exponent met on the structural nodes of an
/// expression (sums, products, negations and integer powers, through
/// which `ratsimp` sees).
struct Power {
    power: ExprId,
    base: ExprId,
    negative: bool,
}

/// The powers of `expr` with a numeric exponent, outside every generator
/// (function arguments and the bases of non-integer powers are not
/// entered).  An explicit stack.
fn structural_powers(arena: &Arena, expr: ExprId) -> Vec<Power> {
    let mut out = Vec::new();
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Add(children) | ExprNode::Mul(children) => stack.extend(children.iter()),
            ExprNode::Neg(child) => stack.push(*child),
            ExprNode::Pow(base, e) => {
                let Some(q) = arena.as_num(*e) else {
                    continue;
                };
                if q.is_integer() {
                    stack.push(*base);
                }
                if !q.is_zero() {
                    out.push(Power {
                        power: id,
                        base: *base,
                        negative: q.is_negative(),
                    });
                }
            }
            _ => {}
        }
    }
    out
}

/// Could `d` vanish identically other than as a rational function of its
/// generators?  Not when it is an atom, when every generator is a free
/// symbol (then the residue test is exact), or when it holds an infinity,
/// `nan` or an unevaluated node (no numeric value to sample).
fn may_vanish_by_identity(arena: &Arena, d: ExprId) -> bool {
    if arena.node(d).is_atom() || walk::has_unevaluated(arena, d) {
        return false;
    }
    let mut transcendental = false;
    for id in walk::post_order_ids(arena, d) {
        match arena.node(id) {
            ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN => return false,
            ExprNode::Num(_)
            | ExprNode::Symbol(_)
            | ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Neg(_) => {}
            ExprNode::Pow(_, e) if arena.as_num(*e).is_some_and(|q| q.is_integer()) => {}
            _ => transcendental = true,
        }
    }
    transcendental
}

/// What the sample evaluations of an expression showed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sampled {
    /// A certified nonzero value at a sample point.
    Nonzero,
    /// Zero (exactly, or to the precision of the zero search) at every
    /// sample point that evaluated, and at least one did.
    Zero,
    /// No sample point evaluated.
    Unknown,
}

/// splitmix64.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The value of the symbol `sym` at sample point `k`: an exact rational,
/// integer or Gaussian rational consistent with its declared assumptions.
/// A symbol without assumptions lies off the real axis, in the right
/// half-plane at `k = 0` and in the left one at `k = 1`, where also
/// `|Re| > π/2` and `|Im| > π`: outside the strips where `atan(tan x) =
/// x`, `ln(e^x) = x` and `ln(x²) = 2·ln x` hold, so that those
/// near-identities show a nonzero value at once.
fn sample_value(arena: &mut Arena, sym: ExprId, k: u64) -> ExprId {
    let props = match arena.node(sym) {
        ExprNode::Symbol(sid) => arena.symbol_assumptions(*sid),
        _ => crate::base::assumptions::Assumptions::default(),
    };
    let has = |p: Props| props.query(p) == Some(true);
    if has(Props::ZERO) {
        return arena.zero;
    }
    let h = mix(u64::from(sym.0) ^ k.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    let pick = |table: &[i64], salt: u32| table[((h >> salt) % table.len() as u64) as usize];
    let sign: i64 = if has(Props::NEGATIVE) || has(Props::NONPOSITIVE) {
        -1
    } else if has(Props::POSITIVE) || has(Props::NONNEGATIVE) || k.is_multiple_of(2) {
        1
    } else {
        -1
    };
    if has(Props::INTEGER) {
        let n = if has(Props::PRIME) {
            pick(&[3, 5, 7, 11], 0)
        } else if has(Props::EVEN) {
            pick(&[2, 4, 6, 8], 0)
        } else if has(Props::ODD) {
            pick(&[3, 5, 7, 9], 0)
        } else {
            pick(&[2, 3, 4, 5], 0)
        };
        return arena.int(sign * n);
    }
    let far = !k.is_multiple_of(2);
    let magnitude = if far {
        arena.rational(sign * pick(&[9, 10, 11, 13], 8), 4)
    } else {
        arena.rational(
            sign * pick(&[7, 9, 11, 13, 17, 19], 8),
            pick(&[5, 6, 7, 8], 16),
        )
    };
    if has(Props::REAL) {
        return magnitude;
    }
    let i = arena.i_unit();
    if has(Props::IMAGINARY) {
        return arena.mul(&[magnitude, i]);
    }
    let im_sign = if (h >> 40) & 1 == 0 { 1 } else { -1 };
    let im = if far {
        arena.rational(im_sign * pick(&[14, 15, 16, 18], 24), 4)
    } else {
        arena.rational(im_sign * pick(&[3, 5, 7, 9], 24), pick(&[4, 5, 7], 32))
    };
    let im = arena.mul(&[im, i]);
    arena.add(&[magnitude, im])
}

/// Evaluate `e` at the sample points ([`sample_value`] for each free
/// symbol; one point for a constant), with certified digits.
fn sample(arena: &mut Arena, e: ExprId) -> Sampled {
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    let syms = walk::free_symbols(arena, e);
    let points = if syms.is_empty() { 1 } else { SAMPLE_POINTS };
    let mut zero = false;
    for k in 0..points {
        let pairs: Vec<(ExprId, ExprId)> = syms
            .iter()
            .map(|&s| (s, sample_value(arena, s, k)))
            .collect();
        let at = crate::transforms::subs::subs_map(arena, e, &pairs);
        if let Some(r) = arena.as_num(at) {
            if r.is_zero() {
                zero = true;
                continue;
            }
            return Sampled::Nonzero;
        }
        match evalf_settled(arena, at, SAMPLE_DIGITS, ZeroSearch::Cap) {
            Ok((z, Settled::Certified)) if !(z.0.is_zero() && z.1.is_zero()) => {
                return Sampled::Nonzero;
            }
            Ok(_) => zero = true,
            Err(_) => {}
        }
    }
    if zero {
        Sampled::Zero
    } else {
        Sampled::Unknown
    }
}

/// Largest `|k|` of a power `exp(m/L)^k` built by [`in_exponentials`].
/// The conversion to a rational function has its own size budget; up to
/// 0.33 this was 256, which refused the constants `subs` makes of
/// `exp(x + y) − exp(x)·exp(y)` at rational points (`exp(−46/11)·exp(−13/7)
/// − exp(−465/77)`: `k = −465`).
const MAX_EXP_POWER: i64 = 4096;

/// The terms `c·m` of the exponent `arg` multiplied out, each with its
/// rational coefficient `c` split off (`m = 1` for a number).
fn exponent_terms(arena: &mut Arena, arg: ExprId) -> Vec<(Q, ExprId)> {
    let expanded = crate::transforms::expand::expand(arena, arg);
    let terms: Vec<ExprId> = match arena.node(expanded) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expanded],
    };
    terms
        .into_iter()
        .map(|t| {
            let mut coeff = Q::one();
            let mut rest = t;
            // `−(2·x)` is `(−1, 2·x)`, then `(2, x)`: a few steps at most.
            for _ in 0..4 {
                let (c, r) = arena.as_coeff_term(rest);
                if r == rest {
                    break;
                }
                coeff *= c;
                rest = r;
            }
            (coeff, rest)
        })
        .collect()
}

/// `e` with the trigonometric and hyperbolic functions written as
/// exponentials ([`rewrite_as_exp`](crate::simplify::rewrite::rewrite_as_exp))
/// and every `exp(Σ cⱼ·mⱼ)` as `Π exp(mⱼ/Lⱼ)^(cⱼ·Lⱼ)`, `Lⱼ` the least
/// common denominator of the coefficients of `mⱼ` throughout `e`: a
/// rational function of exponentials of distinct monomials, which are
/// algebraically independent, so the identity `sin²x + cos²x = 1` or
/// `tanh x·cosh x = sinh x` becomes a polynomial identity in `e^{ix}` or
/// `e^x` (and `i² = −1`).  Both rewrites are identities over ℂ.  `None`
/// when a power would exceed [`MAX_EXP_POWER`].
fn in_exponentials(arena: &mut Arena, e: ExprId) -> Option<ExprId> {
    let e = crate::simplify::rewrite::rewrite_as_exp(arena, e);
    let order = walk::post_order_ids(arena, e);
    let mut lcd: FxHashMap<ExprId, BigInt> = FxHashMap::default();
    for &id in &order {
        if let ExprNode::Exp(arg) = *arena.node(id) {
            for (c, m) in exponent_terms(arena, arg) {
                let l = lcd.entry(m).or_insert_with(BigInt::one);
                *l = l.lcm(c.denom());
            }
        }
    }
    if lcd.is_empty() {
        return Some(e);
    }
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for &id in &order {
        let rebuilt = walk::rebuild_with_cache(arena, id, &cache);
        let out = match *arena.node(rebuilt) {
            ExprNode::Exp(arg) => {
                let mut factors: Vec<ExprId> = Vec::new();
                for (c, m) in exponent_terms(arena, arg) {
                    let l = lcd
                        .get(&m)
                        .cloned()
                        .unwrap_or_else(BigInt::one)
                        .lcm(c.denom());
                    let k = (&c * Q::from_integer(l.clone())).to_integer();
                    if k.abs() > BigInt::from(MAX_EXP_POWER) {
                        return None;
                    }
                    let inv_l = arena.num_ratio(Q::new(BigInt::one(), l));
                    let base_arg = arena.mul(&[inv_l, m]);
                    let base = arena.exp(base_arg);
                    let k = arena.num_ratio(Q::from_integer(k));
                    factors.push(arena.pow(base, k));
                }
                arena.mul(&factors)
            }
            _ => rebuilt,
        };
        cache.insert(id, out);
    }
    cache.get(&e).copied()
}

/// Is `e`, which evaluated to 0 at the sample points, exactly 0?  A
/// constant by the exact zero test
/// ([`is_zero_checked`](crate::poly::algebraic::is_zero_checked)); then
/// as a rational function of exponentials ([`in_exponentials`]), which
/// decides the identities of `sin`, `cos`, `tan`, their hyperbolic
/// counterparts and `exp` with arguments polynomial in the symbols; then
/// by `simplify` reaching the structural 0 (identities of radicals and
/// logarithms under assumptions).
fn confirmed_zero(arena: &mut Arena, e: ExprId) -> bool {
    let Some(_guard) = DepthGuard::enter() else {
        return false;
    };
    if walk::free_symbols(arena, e).is_empty()
        && let Some(decided) = crate::poly::algebraic::is_zero_checked(arena, e)
    {
        return decided;
    }
    if let Some(r) = in_exponentials(arena, e)
        && crate::simplify::ratsimp::vanishes_as_rational_function(arena, r)
    {
        return true;
    }
    let opts = crate::simplify::simplify_engine::SimplifyOpts::default();
    let s = crate::simplify::simplify_engine::unified_simplify(arena, e, &opts).expr;
    arena.is_zero_structural(s)
}

/// Does `e` vanish identically — for every value of its free symbols
/// allowed by their assumptions — by any identity `simplify` knows?
/// `true` only when `e` is zero at the sample points **and** reduces
/// exactly to 0 (see the module docs); `false` when undecided.
///
/// A costly part of `e` — a special function at a parameter of magnitude
/// 10⁵ or more, an orthogonal polynomial of degree 200 or more
/// ([`with_costly_parts_as_symbols`](crate::base::canon::with_costly_parts_as_symbols))
/// — is never evaluated: it is taken for a symbol, so `e` vanishes when it
/// does for every value of that part (`sin²u + cos²u − 1` for `u =
/// jacobi(999, 1/3, 1/5, 1/7)`).  Up to 0.34 the sample and the exact
/// confirmation computed `u`: `ratsimp` and `expand` of `((x + 1)² − x² −
/// 2x − 1)/(sin²u + cos²u − 1)` took 119 s and 113 s in a debug build
/// (now milliseconds, the same `nan`).
pub(crate) fn vanishes_by_identity(arena: &mut Arena, e: ExprId) -> bool {
    if arena.is_zero_structural(e) {
        return true;
    }
    if nested() || walk::has_unevaluated(arena, e) {
        return false;
    }
    let e = if crate::base::canon::costly_to_evaluate(arena, e) {
        match crate::base::canon::with_costly_parts_as_symbols(arena, e) {
            Some(general) => general,
            None => return false,
        }
    } else {
        e
    };
    sample(arena, e) == Sampled::Zero && confirmed_zero(arena, e)
}

/// Is the numerator of `expr` (`as_numer_denom`) identically zero: 0 once
/// multiplied out ([`numerator_expands_to_zero`](crate::simplify::ratsimp::numerator_expands_to_zero)),
/// or by an identity ([`vanishes_by_identity`])?  The test that turns the
/// `zoo` of a vanishing denominator into the `nan` of `0/0`:
/// `(x + 1)·(sin 2x − 2·sin x·cos x)/(cosh²x − sinh²x − 1)` is `nan`.
pub(crate) fn numerator_vanishes_identically(arena: &mut Arena, expr: ExprId) -> bool {
    if crate::simplify::ratsimp::numerator_expands_to_zero(arena, expr) {
        return true;
    }
    let (n, _) = crate::poly::polybridge::fraction_parts(arena, expr);
    vanishes_by_identity(arena, n)
}

/// The value of `expr` when the base of a negative power in it (outside
/// every function argument) vanishes identically by an identity of its
/// functions (`tan x·cos x − sin x`): `ratsimp`'s arithmetic of `1/0 =
/// zoo` with those bases 0 ([`ratsimp_vanishing`](crate::simplify::ratsimp::ratsimp_vanishing)).
/// `nan` for `0/0` when the numerator is 0 once multiplied out; the caller
/// tests a numerator that vanishes by an identity
/// ([`numerator_vanishes_identically`]).  `None` when no such base is
/// found (the common case, at the cost of one certified evaluation per
/// base with a function in it) or when `ratsimp` does not apply.
pub(crate) fn vanishing_by_identity(arena: &mut Arena, expr: ExprId) -> Option<ExprId> {
    if nested() {
        return None;
    }
    vanishing_among(arena, expr, |_| true)
}

/// The products of the rational skeleton of `expr` (outside every
/// generator).  An explicit stack.
fn structural_products(arena: &Arena, expr: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Mul(children) => {
                out.push(id);
                stack.extend(children.iter());
            }
            ExprNode::Add(children) => stack.extend(children.iter()),
            ExprNode::Neg(child) => stack.push(*child),
            ExprNode::Pow(base, e) if arena.as_num(*e).is_some_and(|q| q.is_integer()) => {
                stack.push(*base);
            }
            _ => {}
        }
    }
    out
}

/// [`vanishing_by_identity`] for the denominators of the `0/0` candidates
/// only ([`identity_denominator_candidates`](crate::base::canon::identity_denominator_candidates):
/// a product with a factor that may vanish identically by the residue
/// test), for `powsimp`, which must stay cheap on ordinary input (it runs
/// inside Gruntz) and multiplies such a numerator out to 0: up to 0.33
/// `simplify_powers((eˣ·e⁻ˣ − 1)/(tan x·cos x − sin x))` was `0`
/// (`simplify`: `nan`).  Residue tests only, unless there is such a
/// candidate; `expand` and `ratsimp` select their candidates the same way.
pub(crate) fn vanishing_by_identity_of_candidates(
    arena: &mut Arena,
    expr: ExprId,
) -> Option<ExprId> {
    if nested() {
        return None;
    }
    let products = structural_products(arena, expr);
    let candidates = crate::base::canon::identity_denominator_candidates(arena, &products);
    if candidates.is_empty() {
        return None;
    }
    vanishing_among(arena, expr, |b| candidates.contains(&b))
}

/// [`vanishing_by_identity`] with the bases tested restricted to `wanted`.
fn vanishing_among(
    arena: &mut Arena,
    expr: ExprId,
    wanted: impl Fn(ExprId) -> bool,
) -> Option<ExprId> {
    let powers = structural_powers(arena, expr);
    let mut decided: FxHashSet<ExprId> = FxHashSet::default();
    let mut zero_bases: FxHashSet<ExprId> = FxHashSet::default();
    for p in &powers {
        if !p.negative
            || !wanted(p.base)
            || !decided.insert(p.base)
            || !may_vanish_by_identity(arena, p.base)
        {
            continue;
        }
        if vanishes_by_identity(arena, p.base) {
            zero_bases.insert(p.base);
        }
    }
    if zero_bases.is_empty() {
        return None;
    }
    let mut vanishing: FxHashMap<ExprId, Vanishing> = FxHashMap::default();
    for &b in &zero_bases {
        vanishing.insert(b, Vanishing::Zero);
    }
    for p in &powers {
        if zero_bases.contains(&p.base) {
            let v = if p.negative {
                Vanishing::Pole
            } else {
                Vanishing::Zero
            };
            vanishing.insert(p.power, v);
        }
    }
    tracing::debug!(
        bases = zero_bases.len(),
        "simplify: a denominator vanishes by an identity"
    );
    crate::simplify::ratsimp::ratsimp_vanishing(arena, expr, &vanishing)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(ctx: &crate::api::context::Context, s: &str) -> crate::api::expr::Ex {
        ctx.parse(s).unwrap()
    }

    #[test]
    fn identities_vanish_and_branch_cuts_do_not() {
        let ctx = crate::api::context::Context::new();
        for (src, zero) in [
            ("tan(x)*cos(x) - sin(x)", true),
            ("sin(x)^2 + cos(x)^2 - 1", true),
            ("cosh(x)^2 - sinh(x)^2 - 1", true),
            ("sin(2*x) - 2*sin(x)*cos(x)", true),
            // `simplify` alone does not reach 0 for these two.
            ("tanh(x)*cosh(x) - sinh(x)", true),
            ("cosh(2*x) - cosh(x)^2 - sinh(x)^2", true),
            ("sin(x/2)^2 - (1 - cos(x))/2", true),
            ("exp(x + y) - exp(x)*exp(y)", true),
            ("log(x^2) - 2*log(x)", false),
            ("log(exp(x)) - x", false),
            ("atan(tan(x)) - x", false),
            ("sqrt(x^2) - x", false),
            ("sin(x)^2 + cos(x)^2 - 2", false),
        ] {
            let e = parse(&ctx, src);
            let mut inner = e.inner.write();
            let got = vanishes_by_identity(&mut inner.arena, e.raw_id());
            assert_eq!(got, zero, "{src}");
        }
    }

    #[test]
    fn log_identity_holds_for_positive_symbols() {
        let ctx = crate::api::context::Context::new();
        let _ = ctx
            .symbol_with("p", &[crate::base::assumptions::Assumption::Positive])
            .unwrap();
        let e = parse(&ctx, "log(p^2) - 2*log(p)");
        let mut inner = e.inner.write();
        let arena = &mut inner.arena;
        // Zero at every positive sample point.
        assert_eq!(sample(arena, e.raw_id()), Sampled::Zero);
    }

    #[test]
    fn sample_points_respect_assumptions() {
        let ctx = crate::api::context::Context::new();
        let n = ctx
            .symbol_with("n", &[crate::base::assumptions::Assumption::Negative])
            .unwrap();
        let k = ctx
            .symbol_with("k", &[crate::base::assumptions::Assumption::Integer])
            .unwrap();
        let mut inner = n.inner.write();
        let arena = &mut inner.arena;
        for point in 0..SAMPLE_POINTS {
            let v = sample_value(arena, n.raw_id(), point);
            assert!(arena.as_num(v).is_some_and(|q| q.is_negative()));
            let v = sample_value(arena, k.raw_id(), point);
            assert!(arena.as_num(v).is_some_and(|q| q.is_integer()));
        }
    }
}
