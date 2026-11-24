//! Windowed sums: `Σ_{k = lo}^{hi} f(k)` over a long (or infinite) range
//! whose terms are concentrated around a mode far from the ends.
//!
//! The moment sums of the lattice laws have this shape — the entropy
//! `−Σ P(k) ln P(k)` of `Binomial(10⁶, p)`, `E[ln(1 + X)]` of
//! `Poisson(10⁵)`, a tail of a hypergeometric law: a hundred thousand
//! terms or more, all but a few thousand of them below any precision asked
//! for.  Up to 0.58 a finite range of more than 10⁴ terms was refused, and
//! an infinite one was summed from its lower limit (refused beyond 20,000
//! terms).  Here the terms are summed outward from the mode, and each side
//! stops where the rest is *proved* below the precision.
//!
//! # The summand
//!
//! The expression tree of `f` is split ([`plan`]) into
//!
//! * **constants** (free of `k`), evaluated once;
//! * **weights** `W(k)`: hypergeometric factors, `W(k + 1)/W(k) =
//!   C·P(k)/Q(k)` with `P, Q ∈ ℤ[k]` read from the structure by
//!   [`hypsum::shape_of`](super::hypsum) (powers `c^(mk + b)`, factorials,
//!   `Γ`, binomials of arguments linear in `k`, integer powers of
//!   polynomials), and here also `exp(a·k + b + Σ ln Gᵢ(k))` for
//!   hypergeometric `Gᵢ` (the form `e^{ln P(k)}` the entropy of a binomial
//!   law is written in).  `C` is a real constant, 1 or recovered as a ball
//!   from two consecutive values;
//! * **composites** over them: `+`, `·`, `−`, a power with a numeric
//!   exponent, `ln`.
//!
//! A weight is evaluated once, at the mode, and moved by its recurrence
//! with exact `P(k)`, `Q(k)` (real balls, [`RBall`]); the composites are
//! evaluated per term from those values by the ordinary node rules
//! ([`eval_ball`](super::eval_ball)), so every term carries its own bound.
//! The recurrence is used only across indices where neither `P` nor `Q`
//! vanishes: an integer root of either ends the route (a pole of a `Γ`, or
//! a binomial coefficient beyond its support).
//!
//! # The rest beyond an edge
//!
//! For an edge `K` of the window and a direction `d = ±1`, every node `X`
//! gets an envelope ([`Env`])
//!
//! ```text
//! |X(K + d·j)| ≤ 2^(a + b·j) · (1 + j)^p        for every j ≥ 0 inside the range,
//! ```
//!
//! proved bottom-up: a constant from its ball; `k` itself (`|K ± j| ≤
//! max(1, |K|)·(1 + j)`); sums (`max` of rates and degrees), products
//! (sums of them), positive integer powers.  For a weight, `2^a` bounds
//! `|W(K)|` and `2^b = ρ` bounds the step ratio on the whole rest:
//! `|C|·|P(x)/Q(x)| ≤ ρ` for every real `x` between the edge and the end
//! of the range.  That is **proved from the terms, not assumed**: `P·Q`
//! has no real root there (Descartes isolation on the exact polynomial,
//! `poly::sturm`), so both keep their signs, and for a rational `ρ` just
//! above the ratio at the edge (the largest one of a log-concave weight;
//! the far end and the limit at infinity are tried too) the polynomial
//! `ρ·|Q| − c·|P|`, `c ≥ |C|` rational, is positive at the edge and has
//! no real root on the rest.  Downward the step ratio is `Q/(C·P)`.
//!
//! A node proved positive on the rest also gets a bound of its logarithm
//! ([`LogEnv`]), `|ln X(K + d·j)| ≤ c₀ + j·(c₁ + δ·ln(1 + j))`: a
//! positive constant; a polynomial without a root on the rest and positive
//! at the edge (at integers it is at least `1/D`, `D` its denominator); a
//! positive weight, whose step ratio is at most `ρ` and at least
//! `c/(Σ|qᵢ|·max(1, |k|)^deg Q)` since `|P(k)| ≥ 1` at an integer that is
//! not a root; products and numeric powers of such nodes.  Then `ln X` is
//! real and `|ln X| ≤ (c₀ + c₁ + δ)·(1 + j)²`.  That covers `P(k)·ln P(k)`.
//!
//! With the envelope `(a, b, p)` of the summand and `b < 0`, `x = 2^b`,
//! `n = ⌈p⌉`,
//!
//! ```text
//! Σ_{j ≥ 1} 2^(a + bj)·(1 + j)^p ≤ 2^a · x · 2ⁿ · n! / (1 − x)^(n + 1)
//! ```
//!
//! (`(j + 2)ⁿ ≤ 2ⁿ(j + 1)ⁿ ≤ 2ⁿ·n!·C(j + n, n)` and `Σ C(j + n, n)·xʲ =
//! (1 − x)^(−n−1)`).  The sweep stops at the first edge where this is
//! below the working precision relative to the sum so far; the bound that
//! was proved there is added to the error of the result, on both sides.
//! All envelope arithmetic is in `f64` logarithms rounded up.
//!
//! # Declining
//!
//! The route returns `None` — and the caller's older routes run — when a
//! node has no rule (a complex constant, a function without an envelope),
//! when a proof fails three times (a weight that is not eventually
//! monotone), when a root of `P` or `Q` lies in the window, or when the
//! window would exceed [`MAX_WINDOW_TERMS`] (a standard deviation beyond
//! about 2,000: `Binomial(10⁸, p)`; an `f64` estimate of the width refuses
//! that at once).  It never returns a value without the
//! proved bounds of both rests.
//!
//! mpmath's `nsum` (BSD) extrapolates such sums with an error estimate;
//! nothing is taken from it.

use astro_float::BigFloat;
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::{FxHashMap, FxHashSet};

use super::accuracy::{self, Bound};
use super::ball::{Balls, CBall, Cache, RBall};
use super::hypsum::{Shape, shape_of};
use super::wp::{RM, Wp};
use crate::base::arena::Arena;
use crate::base::bigcomplex::{c_add, c_zero};
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::{Q, bigint_to_bigfloat};
use crate::base::walk;
use crate::poly::dense::Poly;
use crate::poly::sturm::{SturmChain, cauchy_bound};

/// A finite range of at most this many terms keeps its term-by-term sum
/// (the families write the entropy of a law with more atoms as a `Sum`:
/// each term of a hypergeometric law's costs milliseconds from scratch).
pub(super) const MIN_RANGE_TERMS: u32 = 64;

/// An infinite sum of a hypergeometric term is windowed when its terms
/// peak at least this far above the lower limit (below, the series routes
/// of `hypsum` sum from the limit by exact recurrences as before).  A
/// term with other factors (`λᵏ·ln(k!)/k!`) is windowed wherever it peaks.
pub(super) const MIN_MODE_OFFSET: u32 = 64;

/// Most terms of a window (about three seconds at 30 digits).
pub(super) const MAX_WINDOW_TERMS: u64 = 1 << 16;

/// Working bits above the requested precision: the drift of the
/// recurrences (a few ulps per step) and the roundings of the partial sums
/// over [`MAX_WINDOW_TERMS`] terms.
const GUARD_BITS: usize = 48;

/// Most proofs of a rest attempted per side.
const MAX_CHECKS: u32 = 48;

/// Proofs that may fail per side before the route declines.
const MAX_UNPROVED: u32 = 3;

/// Highest degree of a polynomial formed here.
const MAX_DEGREE: usize = 64;

/// Largest index (in bits) a window may reach.
const MAX_INDEX_BITS: u64 = 62;

const LN2: f64 = std::f64::consts::LN_2;

/// `x` rounded up by more than the `f64` rounding of the arithmetic it
/// came from (infinities unchanged).
fn up(x: f64) -> f64 {
    if x.is_finite() {
        x + x.abs() * 1e-12 + 1e-12
    } else {
        x
    }
}

/// Is `x` a positive number (not NaN)?
fn positive(x: f64) -> bool {
    x > 0.0
}

/// `log₂|x|` of an integer to about `2⁻⁵⁰` (`−∞` for 0): a guide, and
/// with [`up`] a bound.
fn lg_int(x: &BigInt) -> f64 {
    if x.is_zero() {
        return f64::NEG_INFINITY;
    }
    let bits = x.bits();
    let shift = bits.saturating_sub(64);
    let top = (x.abs() >> shift).to_f64().unwrap_or(f64::INFINITY);
    top.log2() + shift as f64
}

/// `log₂` of a positive rational, rounded up.
fn lg_q_up(x: &Q) -> f64 {
    up(up(lg_int(x.numer())) - (lg_int(x.denom()) - 1e-9))
}

/// `p(k)` by Horner's rule.
fn eval_int(coeffs: &[BigInt], k: &BigInt) -> BigInt {
    let mut acc = BigInt::zero();
    for c in coeffs.iter().rev() {
        acc = acc * k + c;
    }
    acc
}

fn int_q(n: &BigInt) -> Q {
    Q::from_integer(n.clone())
}

/// `|X(K + d·j)| ≤ 2^(a + b·j)·(1 + j)^p` for every `j ≥ 0` inside the
/// range (see the module documentation).
#[derive(Clone, Copy, Debug)]
struct Env {
    a: f64,
    b: f64,
    p: f64,
}

impl Env {
    fn plus(self, o: Env) -> Env {
        Env {
            a: accuracy::lsum(self.a, o.a),
            b: self.b.max(o.b),
            p: self.p.max(o.p),
        }
    }

    fn times(self, o: Env) -> Env {
        Env {
            a: up(self.a + o.a),
            b: up(self.b + o.b),
            p: self.p + o.p,
        }
    }

    /// `|X|^r` for `r > 0`.
    fn power(self, r: f64) -> Env {
        Env {
            a: up(self.a * r),
            b: up(self.b * r),
            p: self.p * r,
        }
    }

    /// `log₂` of a bound on `Σ_{j ≥ 1} 2^(a + bj)·(1 + j)^p`, for `b < 0`.
    fn rest(self) -> Option<f64> {
        if !positive(-self.b) || !self.p.is_finite() || self.p > 64.0 || self.a.is_nan() {
            return None;
        }
        if self.a == f64::NEG_INFINITY {
            return Some(f64::NEG_INFINITY);
        }
        let n = self.p.max(0.0).ceil();
        // 1 − 2^b = −expm1(b·ln 2), rounded down.
        let gap = -(self.b * LN2).exp_m1() * (1.0 - 1e-12);
        if !positive(gap) {
            return None;
        }
        let mut lg_fact = 0.0;
        let mut i = 2.0;
        while i <= n {
            lg_fact += f64::log2(i);
            i += 1.0;
        }
        Some(up(
            self.a + self.b + n + up(lg_fact) - (n + 1.0) * gap.log2()
        ))
    }
}

/// `X > 0` and `|ln X(K + d·j)| ≤ c0 + j·(c1 + d·ln(1 + j))` (natural
/// logarithms) for every `j ≥ 0` inside the range.
#[derive(Clone, Copy, Debug)]
struct LogEnv {
    c0: f64,
    c1: f64,
    d: f64,
}

impl LogEnv {
    fn plus(self, o: LogEnv) -> LogEnv {
        LogEnv {
            c0: up(self.c0 + o.c0),
            c1: up(self.c1 + o.c1),
            d: self.d + o.d,
        }
    }

    fn scaled(self, r: f64) -> LogEnv {
        LogEnv {
            c0: up(self.c0 * r),
            c1: up(self.c1 * r),
            d: self.d * r,
        }
    }
}

/// A hypergeometric factor of the summand.
struct Weight {
    id: ExprId,
    /// Its post-order (for the evaluation at the mode).
    order: Vec<ExprId>,
    /// `W(k + 1)/W(k) = C·p(k)/q(k)`, integer coefficients.
    p: Vec<BigInt>,
    q: Vec<BigInt>,
    /// `C` is not 1: it is recovered from two consecutive values.
    constant: bool,
}

/// The summand split into constants, weights and composites.
struct Plan {
    body: ExprId,
    var: ExprId,
    weights: Vec<Weight>,
    consts: Vec<ExprId>,
    /// The composite nodes, children first.
    composites: Vec<ExprId>,
    shapes: FxHashMap<ExprId, Shape>,
    /// The weights that are factors of the summand (of its first addend
    /// that has any), not arguments of a logarithm: their product peaks
    /// where the summand does.  A guide for the start of the window.
    guide: Vec<usize>,
}

/// The weights reached from `body` through products, negations and powers
/// (of the first addend that has any); every weight when there is none.
fn guide_weights(
    arena: &Arena,
    body: ExprId,
    weights: &[Weight],
    composite: &FxHashSet<ExprId>,
) -> Vec<usize> {
    let mut roots = vec![body];
    let mut tried = 0u32;
    while let Some(root) = roots.pop() {
        tried += 1;
        if tried > 64 {
            break;
        }
        let mut found = Vec::new();
        let mut addends = Vec::new();
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if let Some(i) = weights.iter().position(|w| w.id == id) {
                if !found.contains(&i) {
                    found.push(i);
                }
                continue;
            }
            if !composite.contains(&id) {
                continue;
            }
            match arena.node(id) {
                ExprNode::Mul(children) => stack.extend(children.iter().copied()),
                ExprNode::Neg(c) | ExprNode::Pow(c, _) => stack.push(*c),
                ExprNode::Add(children) => addends.extend(children.iter().rev().copied()),
                _ => {}
            }
        }
        if !found.is_empty() {
            return found;
        }
        roots.extend(addends);
    }
    (0..weights.len()).collect()
}

/// The ratio of `exp(arg)`: `arg` a sum of terms that are free of `k`,
/// linear in `k` (one factor of degree at most 1 times constants), or the
/// logarithm of a hypergeometric term.  The constant part of the ratio is
/// left to be recovered from the values.
fn exp_shape(arena: &Arena, arg: ExprId, shapes: &FxHashMap<ExprId, Shape>) -> Option<Shape> {
    let terms: Vec<ExprId> = match arena.node(arg) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![arg],
    };
    let linear = |s: &Shape| match s {
        Shape::Free(_) => Some(0u32),
        Shape::Poly(p) if p.degree().unwrap_or(0) <= 1 => Some(1),
        _ => None,
    };
    let (mut num, mut den) = (Poly::one(), Poly::one());
    for t in terms {
        if shapes.get(&t).and_then(linear).is_some() {
            continue;
        }
        match arena.node(t) {
            ExprNode::Ln(u) => match shapes.get(u)? {
                Shape::Ratio(p, q, _) => {
                    num = num.mul(p);
                    den = den.mul(q);
                    if num.degree().unwrap_or(0) > MAX_DEGREE
                        || den.degree().unwrap_or(0) > MAX_DEGREE
                    {
                        return None;
                    }
                }
                _ => return None,
            },
            ExprNode::Mul(children) => {
                let mut degree = 0;
                for c in children.iter() {
                    degree += shapes.get(c).and_then(linear)?;
                }
                if degree != 1 {
                    return None;
                }
            }
            _ => return None,
        }
    }
    Some(Shape::Ratio(num, den, true))
}

/// `P`, `Q` scaled to integer coefficients by a common factor.
fn integer_coefficients(p: &Poly, q: &Poly) -> (Vec<BigInt>, Vec<BigInt>) {
    let mut lcm = BigInt::one();
    for c in p.coeffs().iter().chain(q.coeffs()) {
        lcm = lcm.lcm(c.denom());
    }
    let scale = |poly: &Poly| -> Vec<BigInt> {
        poly.coeffs()
            .iter()
            .map(|c| (c * int_q(&lcm)).to_integer())
            .collect()
    };
    (scale(p), scale(q))
}

fn poly_of(coeffs: &[BigInt]) -> Poly {
    Poly::from_coeffs(coeffs.iter().map(int_q).collect())
}

/// Splits the summand (see the module documentation); `None` when a node
/// has no rule or there is no weight.
fn plan(arena: &Arena, body: ExprId, var: ExprId) -> Option<Plan> {
    let post = walk::post_order_ids(arena, body);
    let mut shapes: FxHashMap<ExprId, Shape> = FxHashMap::default();
    for &id in &post {
        let shape = shape_of(arena, id, var, &shapes).or_else(|| match arena.node(id) {
            ExprNode::Exp(a) => exp_shape(arena, *a, &shapes),
            _ => None,
        });
        if let Some(s) = shape {
            shapes.insert(id, s);
        }
    }
    let mut weights = Vec::new();
    let mut consts = Vec::new();
    let mut composite: FxHashSet<ExprId> = FxHashSet::default();
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack = vec![body];
    while let Some(id) = stack.pop() {
        if id == var || !seen.insert(id) {
            continue;
        }
        let node = arena.node(id);
        match shapes.get(&id) {
            Some(Shape::Free(_)) => {
                consts.push(id);
                continue;
            }
            Some(Shape::Ratio(p, q, constant))
                if !matches!(node, ExprNode::Mul(_) | ExprNode::Neg(_)) =>
            {
                if p.is_zero() || q.is_zero() {
                    return None;
                }
                let (pi, qi) = integer_coefficients(p, q);
                weights.push(Weight {
                    id,
                    order: walk::post_order_ids(arena, id),
                    p: pi,
                    q: qi,
                    constant: *constant,
                });
                continue;
            }
            _ => {}
        }
        match node {
            ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) | ExprNode::Ln(_) => {}
            ExprNode::Pow(_, e) if arena.as_num(*e).is_some() => {}
            _ => return None,
        }
        composite.insert(id);
        stack.extend(node.children());
    }
    if weights.is_empty() {
        return None;
    }
    let composites = post
        .iter()
        .copied()
        .filter(|id| composite.contains(id))
        .collect();
    let guide = guide_weights(arena, body, &weights, &composite);
    Some(Plan {
        body,
        var,
        weights,
        consts,
        composites,
        shapes,
        guide,
    })
}

/// A real ball of an evaluated node; `None` when it is not exactly real,
/// not finite or without a bound.
fn real_ball(b: &CBall) -> Option<RBall> {
    if b.err.is_unknown() || !b.is_exactly_real() || b.mid.0.is_nan() || b.mid.0.is_inf() {
        return None;
    }
    Some(RBall {
        m: b.mid.0.clone(),
        r: b.err.re,
    })
}

fn complex_ball(r: &RBall, wp: usize) -> CBall {
    CBall::new((r.m.clone(), BigFloat::new(wp)), Bound::real(r.r))
}

/// Has `poly` no real root in `[a, b]` (`b = None`: in `[a, ∞)`)?
fn root_free(poly: &Poly, a: &BigInt, b: Option<&BigInt>) -> bool {
    let Some(degree) = poly.degree() else {
        return false;
    };
    if degree == 0 {
        return true;
    }
    if degree > 2 * MAX_DEGREE {
        return false;
    }
    let a = int_q(a);
    let b = match b {
        Some(b) => int_q(b),
        // Beyond Cauchy's bound there is no root.
        None => cauchy_bound(poly).max(a.clone()) + Q::one(),
    };
    SturmChain::new(poly).count_roots_in_closed(&a, &b) == 0
}

/// The constant `C` of a weight's ratio and what the bounds need of it.
struct Constant {
    /// `C` and `1/C` (`None`: `C = 1`).
    c: Option<RBall>,
    c_inv: Option<RBall>,
    positive: bool,
}

/// The state of a sweep: the index and the weights there.
#[derive(Clone)]
struct State {
    k: BigInt,
    vals: Vec<RBall>,
}

struct Sum<'a, 'w, 'c> {
    arena: &'a Arena,
    plan: &'a Plan,
    lo: &'a BigInt,
    hi: Option<&'a BigInt>,
    constants: Vec<Constant>,
    /// The constants' balls (kept), and per term the index, the weights
    /// and the composites.
    balls: Balls,
    const_balls: Vec<(ExprId, RBall)>,
    w: &'w mut Wp<'c>,
    prec: usize,
    used: u64,
}

impl Sum<'_, '_, '_> {
    /// The weights at `k + 1` (`up`) or `k − 1` from those at `k`; `None`
    /// at a root of `P` or `Q`.
    fn step(&self, st: &mut State, up: bool) -> Option<()> {
        let wp = self.w.bits();
        let index = if up { st.k.clone() } else { &st.k - 1 };
        for ((wt, c), val) in self
            .plan
            .weights
            .iter()
            .zip(&self.constants)
            .zip(st.vals.iter_mut())
        {
            let (pk, qk) = (eval_int(&wt.p, &index), eval_int(&wt.q, &index));
            if pk.is_zero() || qk.is_zero() {
                return None;
            }
            // (Not reduced: only its value is read.)
            let sign = if qk.is_negative() == pk.is_negative() {
                BigInt::one()
            } else {
                -BigInt::one()
            };
            let (ratio, constant) = if up {
                (Q::new_raw(sign * pk.abs(), qk.abs()), &c.c)
            } else {
                (Q::new_raw(sign * qk.abs(), pk.abs()), &c.c_inv)
            };
            let mut next = val.mul(&RBall::from_q(&ratio, wp), wp);
            if let Some(c) = constant {
                next = next.mul(c, wp);
            }
            *val = next;
        }
        if up {
            st.k += 1;
        } else {
            st.k -= 1;
        }
        (st.k.bits() <= MAX_INDEX_BITS).then_some(())
    }

    /// The summand at `st.k` with its bound.
    fn term(&mut self, st: &State) -> Option<CBall> {
        let wp = self.w.bits();
        let k = (bigint_to_bigfloat(&st.k, wp), BigFloat::new(wp));
        self.balls.insert(self.plan.var, CBall::exact(k));
        for (wt, val) in self.plan.weights.iter().zip(&st.vals) {
            self.balls.insert(wt.id, complex_ball(val, wp));
        }
        for &id in &self.plan.composites {
            let ball = super::eval_ball(self.arena, id, Cache::Balls(&self.balls), self.w).ok()?;
            if ball.err.is_unknown() || ball.mid.0.is_nan() || ball.mid.0.is_inf() {
                return None;
            }
            self.balls.insert(id, ball);
        }
        self.balls.get(&self.plan.body).cloned()
    }

    /// The envelope of weight `i` beyond the edge `st.k` (and of its
    /// logarithm when it is positive there), with the proofs of the module
    /// documentation; `None` when one fails.
    fn weight_env(&self, i: usize, st: &State, up_side: bool) -> Option<(Env, Option<LogEnv>)> {
        let wt = &self.plan.weights[i];
        let c = &self.constants[i];
        let val = &st.vals[i];
        // The indices of the steps of the rest, `[a, b]`.
        let (a, b): (BigInt, Option<BigInt>) = if up_side {
            (st.k.clone(), self.hi.map(|h| h - 1))
        } else {
            (self.lo.clone(), Some(&st.k - 1))
        };
        // The step ratio is `c·|num|/|den|`.
        let (num, den, constant) = if up_side {
            (&wt.p, &wt.q, &c.c)
        } else {
            (&wt.q, &wt.p, &c.c_inv)
        };
        let (num_poly, den_poly) = (poly_of(num), poly_of(den));
        if !root_free(&num_poly.mul(&den_poly), &a, b.as_ref()) {
            return None;
        }
        let (num_a, den_a) = (eval_int(num, &a), eval_int(den, &a));
        let (s_num, s_den) = (num_a.signum(), den_a.signum());
        // `c_hi ≥ |c|` as a rational, `lg_c_lo ≤ log₂|c|`.
        let (c_hi, lg_c_lo) = match constant {
            None => (Q::one(), 0.0),
            Some(ball) => {
                let (hi, lo) = (ball.hi(), ball.lo());
                if !hi.is_finite() || !lo.is_finite() || hi.abs() > 1e6 {
                    return None;
                }
                let whole = hi.floor();
                let frac = Q::from_float((hi - whole).exp2() * (1.0 + 1e-12))?;
                let shift = i32::try_from(crate::base::cast::f64_to_i64(whole)?).ok()?;
                let two = Q::from_integer(BigInt::from(2));
                (frac * two.pow(shift), lo)
            }
        };
        let abs_num = num_poly.scale(&int_q(&s_num));
        let abs_den = den_poly.scale(&int_q(&s_den));
        let margin = Q::one() + Q::new(BigInt::one(), BigInt::from(1u32 << 20));
        let at = |i: &BigInt| -> Option<Q> {
            let d = eval_int(den, i).abs();
            (!d.is_zero()).then(|| &c_hi * Q::new(eval_int(num, i).abs(), d))
        };
        let near = if up_side { Some(a.clone()) } else { b.clone() };
        let far = if up_side { b.clone() } else { Some(a.clone()) };
        let mut candidates: Vec<Q> = Vec::with_capacity(3);
        candidates.extend(near.as_ref().and_then(&at));
        candidates.extend(far.as_ref().and_then(&at));
        if b.is_none() && num.len() == den.len() {
            // The limit of the ratio at infinity.
            if let (Some(n), Some(d)) = (num.last(), den.last()) {
                candidates.push(&c_hi * Q::new(n.abs(), d.abs()));
            }
        }
        let mut lg_rho = None;
        for t in candidates {
            if !t.is_positive() {
                continue;
            }
            let rho = &t * &margin;
            let f = abs_den.scale(&rho).sub(&abs_num.scale(&c_hi));
            let positive_at = |x: &BigInt| f.eval(&int_q(x)).is_positive();
            if positive_at(&a)
                && b.as_ref().is_none_or(&positive_at)
                && root_free(&f, &a, b.as_ref())
            {
                lg_rho = Some(lg_q_up(&rho));
                break;
            }
        }
        let lg_rho = lg_rho?;
        let env = Env {
            a: val.hi(),
            b: lg_rho,
            p: 0.0,
        };
        // The logarithm, for a positive weight: every step ratio is at most
        // `ρ` and at least `c_lo/(Σ|denᵢ|·max(1, |k|)^deg)`.
        let positive = c.positive && (&s_num * &s_den).is_positive() && val.certainly_positive();
        let log = positive.then(|| {
            let degree = den.len().saturating_sub(1) as f64;
            let total: BigInt = den.iter().map(BigInt::abs).sum();
            let lk = lg_int(&st.k).max(0.0);
            let low = (up(lg_int(&total)) + up(degree * lk) - lg_c_lo).max(0.0);
            LogEnv {
                c0: up(LN2 * val.hi().abs().max(val.lo().abs())),
                c1: up(LN2 * lg_rho.abs().max(up(low))),
                d: degree,
            }
        });
        Some((env, log))
    }

    /// A polynomial positive on the values of the rest: its logarithm's
    /// envelope and `log₂ D` (`poly(k) ≥ 1/D` at the integers).
    fn poly_log(&self, poly: &Poly, st: &State, up_side: bool) -> Option<(LogEnv, f64)> {
        let (a, b): (BigInt, Option<BigInt>) = if up_side {
            (st.k.clone(), self.hi.cloned())
        } else {
            (self.lo.clone(), Some(st.k.clone()))
        };
        if !root_free(poly, &a, b.as_ref()) || !poly.eval(&int_q(&a)).is_positive() {
            return None;
        }
        let mut lcm = BigInt::one();
        let mut total = Q::zero();
        for c in poly.coeffs() {
            lcm = lcm.lcm(c.denom());
            total += c.abs();
        }
        let degree = poly.degree()? as f64;
        let lk = lg_int(&st.k).max(0.0);
        let lg_d = up(lg_int(&lcm));
        let c0 = up(LN2 * lg_d.max(lg_q_up(&total) + up(degree * lk)));
        Some((
            LogEnv {
                c0,
                c1: 0.0,
                d: degree,
            },
            lg_d,
        ))
    }

    /// Is the polynomial 0 at the lower end of the values of the rest and
    /// positive at every other integer there?
    fn poly_nonneg(&self, poly: &Poly, st: &State, up_side: bool) -> bool {
        let (a, b): (BigInt, Option<BigInt>) = if up_side {
            (st.k.clone(), self.hi.cloned())
        } else {
            (self.lo.clone(), Some(st.k.clone()))
        };
        if !poly.eval(&int_q(&a)).is_zero() {
            return false;
        }
        let next = &a + 1;
        if b.as_ref().is_some_and(|b| &next > b) {
            return true;
        }
        poly.eval(&int_q(&next)).is_positive() && root_free(poly, &next, b.as_ref())
    }

    /// `log₂` of a proved bound on the sum of `|f|` beyond the edge `st.k`
    /// on one side, and the rate `b` of the summand's envelope.
    fn rest(&self, st: &State, up_side: bool) -> Option<(f64, f64)> {
        let arena = self.arena;
        let mut envs: FxHashMap<ExprId, Env> = FxHashMap::default();
        let mut logs: FxHashMap<ExprId, LogEnv> = FxHashMap::default();
        let mut poly_low: FxHashMap<ExprId, f64> = FxHashMap::default();
        let mut nonneg: FxHashSet<ExprId> = FxHashSet::default();
        for (id, ball) in &self.const_balls {
            envs.insert(
                *id,
                Env {
                    a: ball.hi(),
                    b: 0.0,
                    p: 0.0,
                },
            );
            if ball.certainly_positive() {
                logs.insert(
                    *id,
                    LogEnv {
                        c0: up(LN2 * ball.hi().abs().max(ball.lo().abs())),
                        c1: 0.0,
                        d: 0.0,
                    },
                );
            }
        }
        for i in 0..self.plan.weights.len() {
            let (env, log) = self.weight_env(i, st, up_side)?;
            let id = self.plan.weights[i].id;
            envs.insert(id, env);
            if let Some(log) = log {
                logs.insert(id, log);
            }
        }
        envs.insert(
            self.plan.var,
            Env {
                a: up(lg_int(&st.k).max(0.0)),
                b: 0.0,
                p: 1.0,
            },
        );
        let var = std::iter::once(self.plan.var);
        for id in var.chain(self.plan.composites.iter().copied()) {
            let node = arena.node(id);
            let env = if id == self.plan.var {
                None
            } else {
                Some(match node {
                    ExprNode::Add(children) => {
                        let mut acc: Option<Env> = None;
                        for c in children.iter() {
                            let e = *envs.get(c)?;
                            acc = Some(acc.map_or(e, |a| a.plus(e)));
                        }
                        acc?
                    }
                    ExprNode::Mul(children) => {
                        let mut acc: Option<Env> = None;
                        let mut log = Some(LogEnv {
                            c0: 0.0,
                            c1: 0.0,
                            d: 0.0,
                        });
                        for c in children.iter() {
                            let e = *envs.get(c)?;
                            acc = Some(acc.map_or(e, |a| a.times(e)));
                            log = match (log, logs.get(c)) {
                                (Some(l), Some(m)) => Some(l.plus(*m)),
                                _ => None,
                            };
                        }
                        if let Some(log) = log {
                            logs.insert(id, log);
                        }
                        acc?
                    }
                    ExprNode::Neg(c) => *envs.get(c)?,
                    ExprNode::Pow(base, e) => {
                        let q = arena.as_num(*e)?;
                        let r = crate::base::numeric::ratio_to_f64(q)?;
                        let base_env = *envs.get(base)?;
                        let base_log = logs.get(base).copied();
                        if let Some(log) = base_log {
                            logs.insert(id, log.scaled(up(r.abs())));
                        }
                        if q.is_integer() && q.is_positive() && r <= 64.0 {
                            base_env.power(r)
                        } else if q.is_positive() && r <= 64.0 && nonneg.contains(base) {
                            // A root of a polynomial that is not negative: real.
                            base_env.power(up(r))
                        } else if base_log.is_some()
                            && let Some(&lg_d) = poly_low.get(base)
                        {
                            // A power of a positive polynomial: real.
                            if q.is_positive() {
                                base_env.power(up(r))
                            } else {
                                Env {
                                    a: up(up(-r) * lg_d),
                                    b: 0.0,
                                    p: 0.0,
                                }
                            }
                        } else {
                            return None;
                        }
                    }
                    ExprNode::Ln(u) => {
                        let l = logs.get(u)?;
                        Env {
                            a: up(up(l.c0 + l.c1 + l.d).log2()),
                            b: 0.0,
                            p: 2.0,
                        }
                    }
                    _ => return None,
                })
            };
            if let Some(env) = env {
                envs.insert(id, env);
            }
            // A polynomial in `k` that is positive (or not negative) on the
            // rest.
            if let Some(Shape::Poly(poly)) = self.plan.shapes.get(&id) {
                if !logs.contains_key(&id)
                    && let Some((log, lg_d)) = self.poly_log(poly, st, up_side)
                {
                    logs.insert(id, log);
                    poly_low.insert(id, lg_d);
                }
                if poly_low.contains_key(&id) || self.poly_nonneg(poly, st, up_side) {
                    nonneg.insert(id);
                }
            }
        }
        let env = *envs.get(&self.plan.body)?;
        Some((env.rest()?, env.b))
    }

    /// Adds the terms beyond the mode on one side until the rest is proved
    /// below the precision, or the range ends: `log₂` of the bound on the
    /// rest (`−∞`: nothing is left).  `None`: declined.
    fn sweep(&mut self, mut st: State, up_side: bool, sum: &mut super::SumAcc) -> Option<f64> {
        let mut checks = 0u32;
        let mut unproved = 0u32;
        let mut next_check = 0u64;
        let mut steps = 0u64;
        let mut last_lg = f64::INFINITY;
        loop {
            let at_end = if up_side {
                self.hi.is_some_and(|h| &st.k >= h)
            } else {
                &st.k <= self.lo
            };
            if at_end {
                return Some(f64::NEG_INFINITY);
            }
            let lg_sum = accuracy::lg_abs(&sum.value);
            let lg_tol = lg_sum - self.prec as f64 - 4.0;
            if steps >= next_check && last_lg <= lg_tol {
                checks += 1;
                if checks > MAX_CHECKS {
                    return None;
                }
                match self.rest(&st, up_side) {
                    Some((lg_rest, _)) if lg_rest <= lg_tol => return Some(lg_rest),
                    Some((lg_rest, rate)) => {
                        // About `deficit/|rate|` more steps at this rate; the
                        // rate improves outward, so somewhat fewer.
                        let more = (0.6 * (lg_rest - lg_tol) / -rate).clamp(4.0, 16384.0);
                        let more = crate::base::cast::f64_to_i64(more.ceil()).unwrap_or(16384);
                        next_check = steps + more.unsigned_abs();
                    }
                    None => {
                        unproved += 1;
                        if unproved > MAX_UNPROVED {
                            return None;
                        }
                        next_check = steps + (steps / 4).max(16);
                    }
                }
            }
            self.step(&mut st, up_side)?;
            steps += 1;
            self.used += 1;
            if self.used > MAX_WINDOW_TERMS {
                return None;
            }
            let term = self.term(&st)?;
            last_lg = accuracy::lg_abs(&term.mid);
            sum.add(&term.mid, term.err);
        }
    }
}

/// `log₂` of the step ratio at `k` of the product of the guide weights, in
/// `f64`.
fn growth(plan: &Plan, constants: &[Constant], k: &BigInt) -> f64 {
    let mut g = 0.0;
    for &i in &plan.guide {
        let wt = &plan.weights[i];
        g += lg_int(&eval_int(&wt.p, k)) - lg_int(&eval_int(&wt.q, k));
        if let Some(c) = &constants[i].c {
            g += c.hi();
        }
    }
    g
}

/// Whether a window at `mode` for `prec` bits would clearly exceed
/// [`MAX_WINDOW_TERMS`]: the fall of the terms over half the budget on a
/// side with room for it — the logarithms of the step ratios at the
/// offsets 0, 1, 2, 4, … from the mode, by the trapezoid rule (within a
/// factor 1.5 for a ratio like `1 − c/k`, where one sample at the far end
/// would be off by thousands of bits) — against the precision and the
/// polynomial factors of the bound.  An `f64` estimate that only refuses
/// early (the older routes then run, unless they are as hopeless); the
/// budget itself is counted.
fn width(
    plan: &Plan,
    constants: &[Constant],
    mode: &BigInt,
    lo: &BigInt,
    hi: Option<&BigInt>,
    prec: usize,
) -> Width {
    let half = MAX_WINDOW_TERMS / 2;
    let fall = |up_side: bool| -> f64 {
        // Bits lost per step at the offset `o` (never negative).
        let rate = |o: u64| -> f64 {
            let g = if up_side {
                -growth(plan, constants, &(mode + o))
            } else {
                growth(plan, constants, &(mode - o - 1u32))
            };
            g.max(0.0)
        };
        let (mut total, mut at, mut before) = (0.0, 0u64, rate(0));
        let mut o = 1u64;
        while o <= half {
            let r = rate(o);
            total += (before + r) / 2.0 * (o - at) as f64;
            (at, before) = (o, r);
            o *= 2;
        }
        total
    };
    let mut least = f64::INFINITY;
    if hi.is_none_or(|h| &(mode + half) < h) {
        least = least.min(fall(true));
    }
    if &(mode - half) > lo {
        least = least.min(fall(false));
    }
    if least.is_nan() || least >= prec as f64 + 64.0 {
        Width::Fits
    } else if 0.61 * least < 0.9 * prec as f64 && decays_geometrically(plan, constants) {
        Width::Hopeless
    } else {
        Width::Wide
    }
}

/// Does the step ratio of the product of the guide weights tend to a limit
/// of modulus below 1 (or to 0)?  Then the terms fall geometrically in the
/// end, and the only other route is `hypsum`'s dominated series from the
/// lower limit; a ratio tending to modulus 1 (`(−1)ᵏ/ln k`, `ln k/k²`) is
/// left to the routes for alternating and smooth terms.  An `f64` reading
/// of the leading coefficients: a guide.
fn decays_geometrically(plan: &Plan, constants: &[Constant]) -> bool {
    let (mut degree, mut lg) = (0i64, 0.0);
    for &i in &plan.guide {
        let wt = &plan.weights[i];
        degree += wt.p.len() as i64 - wt.q.len() as i64;
        if let (Some(p), Some(q)) = (wt.p.last(), wt.q.last()) {
            lg += lg_int(p) - lg_int(q);
        }
        if let Some(c) = &constants[i].c {
            lg += c.hi();
        }
    }
    degree < 0 || (degree == 0 && positive(-lg))
}

/// What [`width`] estimates.
enum Width {
    /// The window may fit the budget.
    Fits,
    /// It would not: the window is not tried.
    Wide,
    /// Nor would 20,000 terms from the lower limit (`hypsum`'s budget for
    /// a dominated series, 0.61 of half a window) bring the terms down by
    /// the precision.
    Hopeless,
}

/// The outcome of [`windowed_sum`].
pub(super) enum Windowed {
    /// The sum with its proved bound.
    Value(CBall),
    /// An infinite sum of a term that is not hypergeometric whose terms
    /// fall too slowly for a window, and for the term-by-term routes.
    TooWide,
    /// The route does not apply or cannot prove its bounds.
    Declined,
}

/// The index where the product of the guide weights stops growing, by
/// bisection on `f64` logarithms of the step ratios: a guide (the window
/// is right wherever it starts).
fn guide_mode(
    plan: &Plan,
    constants: &[Constant],
    lo: &BigInt,
    hi: Option<&BigInt>,
) -> Option<BigInt> {
    let growth = |k: &BigInt| growth(plan, constants, k);
    if hi.is_some_and(|h| h <= lo) || !positive(growth(lo)) {
        return Some(lo.clone());
    }
    let mut a = lo.clone();
    let mut b = match hi {
        Some(h) => {
            let last: BigInt = h - 1;
            if positive(growth(&last)) {
                return Some(h.clone());
            }
            last
        }
        None => {
            let mut step = BigInt::one();
            loop {
                let c = &a + &step;
                if !positive(growth(&c)) {
                    break c;
                }
                a = c;
                step *= 2;
                if step.bits() > MAX_INDEX_BITS {
                    return None;
                }
            }
        }
    };
    while &b - &a > BigInt::one() {
        let mid: BigInt = (&a + &b) >> 1;
        if positive(growth(&mid)) {
            a = mid;
        } else {
            b = mid;
        }
    }
    Some(b)
}

/// `Σ_{var = lo}^{hi} body` (`hi = None`: to infinity) by a window around
/// the mode with proved bounds on both rests (see the module
/// documentation), at the precision of `w`.  Declined when the route does
/// not apply or cannot prove its bounds; for an infinite sum of a
/// hypergeometric term also when the mode lies within `min_offset` of `lo`.
pub(super) fn windowed_sum(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: &BigInt,
    hi: Option<&BigInt>,
    min_offset: u32,
    w: &mut Wp<'_>,
) -> Windowed {
    let mut hopeless = false;
    match window(arena, body, var, lo, hi, min_offset, w, &mut hopeless) {
        Some(ball) => Windowed::Value(ball),
        None if hopeless => Windowed::TooWide,
        None => Windowed::Declined,
    }
}

/// [`windowed_sum`]; `hopeless` is set when the refusal is final.
#[allow(clippy::too_many_arguments)]
fn window(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: &BigInt,
    hi: Option<&BigInt>,
    min_offset: u32,
    w: &mut Wp<'_>,
    hopeless: &mut bool,
) -> Option<CBall> {
    if lo.bits() > MAX_INDEX_BITS || hi.is_some_and(|h| h.bits() > MAX_INDEX_BITS || h < lo) {
        return None;
    }
    let plan = plan(arena, body, var)?;
    let prec = w.bits();
    let wp = prec + GUARD_BITS;
    let mut ww = w.at(wp);

    // The constants, once.
    let mut balls = Balls::default();
    let mut const_balls = Vec::with_capacity(plan.consts.len());
    for &id in &plan.consts {
        let order = walk::post_order_ids(arena, id);
        let ball = super::evaluate_tree(arena, id, &order, None, true, &mut ww).ok()?;
        const_balls.push((id, real_ball(&ball)?));
        balls.insert(id, ball);
    }

    // The constant of each weight's ratio, from two consecutive values.
    let weight_at = |wt: &Weight, k: &BigInt, ww: &mut Wp<'_>| -> Option<RBall> {
        let ball = super::evaluate_at_integer(arena, wt.id, &wt.order, var, k, ww).ok()?;
        real_ball(&ball)
    };
    let probes: Vec<BigInt> = match hi {
        Some(h) => vec![(lo + h) >> 1, lo.clone(), h - 1],
        None => vec![lo.clone(), lo + 1, lo + 16],
    };
    let mut constants = Vec::with_capacity(plan.weights.len());
    for wt in &plan.weights {
        if !wt.constant {
            constants.push(Constant {
                c: None,
                c_inv: None,
                positive: true,
            });
            continue;
        }
        let mut found = None;
        for k in &probes {
            let (pk, qk) = (eval_int(&wt.p, k), eval_int(&wt.q, k));
            if pk.is_zero() || qk.is_zero() || hi.is_some_and(|h| k >= h) || k < lo {
                continue;
            }
            let (Some(w0), Some(w1)) =
                (weight_at(wt, k, &mut ww), weight_at(wt, &(k + 1), &mut ww))
            else {
                continue;
            };
            let Some(inverse) = w0.recip(wp) else {
                continue;
            };
            let c = w1
                .mul(&inverse, wp)
                .mul(&RBall::from_q(&Q::new(qk, pk), wp), wp);
            if let Some(c_inv) = c.recip(wp) {
                found = Some(Constant {
                    positive: c.certainly_positive(),
                    c: Some(c),
                    c_inv: Some(c_inv),
                });
                break;
            }
        }
        constants.push(found?);
    }

    let mode = guide_mode(&plan, &constants, lo, hi)?;
    let hypergeometric = plan.shapes.contains_key(&body);
    if hi.is_none() && hypergeometric && &mode - lo < BigInt::from(min_offset) {
        return None;
    }
    match width(&plan, &constants, &mode, lo, hi, prec) {
        Width::Fits => {}
        refused => {
            tracing::debug!("evalf: Sum — the window around the mode would be too long");
            *hopeless = matches!(refused, Width::Hopeless) && hi.is_none() && !hypergeometric;
            return None;
        }
    }
    let mut vals = Vec::with_capacity(plan.weights.len());
    for wt in &plan.weights {
        vals.push(weight_at(wt, &mode, &mut ww)?);
    }
    let anchor = State { k: mode, vals };
    let mut run = Sum {
        arena,
        plan: &plan,
        lo,
        hi,
        constants,
        balls,
        const_balls,
        w: &mut ww,
        prec,
        used: 1,
    };
    let mut sum = super::SumAcc::new(wp);
    let first = run.term(&anchor)?;
    sum.add(&first.mid, first.err);
    let above = run.sweep(anchor.clone(), true, &mut sum)?;
    let below = run.sweep(anchor, false, &mut sum)?;
    let terms = run.used;
    let total = sum.finish();
    if total.err.is_unknown() {
        return None;
    }
    let rest = accuracy::lsum(above, below);
    let value = c_add(&total.mid, &c_zero(prec), prec, RM);
    let rounding = accuracy::rounding(&value, prec);
    let re = accuracy::lsum(accuracy::lsum(total.err.re, rest), rounding);
    // The rest is real (real constants and weights, logarithms of positive
    // nodes only); a window term that is not exactly real takes it on both
    // parts.
    let im = if total.is_exactly_real() {
        total.err.im
    } else {
        accuracy::lsum(accuracy::lsum(total.err.im, rest), rounding)
    };
    tracing::debug!(
        terms,
        above,
        below,
        "evalf: Sum over a window around the mode"
    );
    Some(CBall::new(value, Bound { re, im }))
}
