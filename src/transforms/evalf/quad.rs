//! Arbitrary-precision quadrature of a `DefiniteIntegral` node, after
//! mpmath's `quad` (`mpmath/calculus/quadrature.py`, BSD licence:
//! `TanhSinh.calc_nodes`, `TanhSinh.sum_next`, `QuadratureRule.summation`
//! and `transform_nodes`, `GaussLegendre.calc_nodes`; adaptive bisection as
//! in its `quadsubdiv`).
//!
//! # Algorithm
//!
//! The default rule is tanh-sinh (double exponential): `x = tanh(π/2·sinh
//! t)` maps `t ∈ ℝ` onto `(−1, 1)` with weights `w = π/2·cosh t/cosh²(π/2·
//! sinh t)` that decay double exponentially, so an integrand with an
//! algebraic or logarithmic singularity at an endpoint becomes a rapidly
//! decaying function of `t` and the trapezoidal sum with step `h = 2⁻ᵏ`
//! (level `k`) converges about as fast as for an analytic one.  As in mpmath
//! level 1 has `h = ½` and the centre, and each further level adds the odd
//! multiples of `h`, reusing the previous sum.  An interval `[a, b]` is
//! mapped linearly; `[a, ∞)` by `x = a − 1 + 2/(s + 1)` and `(−∞, b]` by
//! `x = b + 1 − 2/(s + 1)` (mpmath's `transform_nodes`); `(−∞, ∞)` is
//! split at 0 (and at the breakpoints) so that each half must converge on
//! its own — mpmath folds it to `f(x) + f(−x)` on `[0, ∞)`, which gives a
//! finite "principal value" for `∫ x dx`.
//!
//! Unlike mpmath, the distance `δ = 1 − |s|` of a node to the nearer end of
//! `(−1, 1)` is computed directly (`2q/(1 + q)`, `q = e^{−π sinh t}`) rather
//! than as `1 − tanh(…)`, and a node is placed at its distance `Cδ` from its
//! nearer endpoint `e` (`C = (b − a)/2`): the caller supplies the body
//! shifted to each bound and breakpoint, `body(e ± x)`, evaluated at `x =
//! Cδ`, so that an algebraic or logarithmic singularity at `e` is seen
//! exactly however close the node (`(x − 1/3)^(−9/10)` shifted is
//! `x^(−9/10)`).  The walk outwards stops where the terms are negligible,
//! not where `1 − |s|` underflows the working precision: mpmath at 50
//! digits gets 28 digits of `∫₀¹ dx/√(1 − x)` and 6 of `∫₀¹ x^(−9/10) dx`
//! (with an error estimate of `10⁻⁵³` for the first).  Towards an infinite
//! end it does not stop before its nodes reach where the integrand's terms
//! cross over (`Integrand::crossover_reach`): within `2^(2S)`, where
//! `2^(±S)` holds the integrand's constants, for orders of growth at least
//! 1 apart, and near `2^(L/D)` for two terms whose orders differ by `D <
//! 1` and whose sizes at `x = 2` differ by `2^L`.  Up to 0.52 the walk
//! stopped at `x ≈ 10⁸` whatever they were (`∫₁^∞ x⁻²e^{x/10⁷⁰⁰} dx`,
//! which diverges, was `1`), and up to 0.53 at `2^(2S)` (`∫₁^∞
//! x⁻²e^{x^(1/1000)} dx`, which diverges, was `2.7210`); an integrand whose
//! terms cross over beyond the last node (`t = 12`, `x ≈ 2³⁶⁸⁸³⁶`) is
//! refused.
//!
//! A node whose value underflowed the exponent range (a 0 with an underflow
//! bound) is *blind*: it bounds the integrand at that node and says nothing
//! of its size next to it.  A rule none of whose nodes saw a value does not
//! converge (the other rule is tried; a value seen beats none), and a
//! quadrature none of whose pieces did is refused: the mass, if any, lies
//! between the nodes — the moment generating function of a Weibull
//! distribution of shape `31/2` and scale `6·10⁻⁶` has all its mass within
//! `10⁻⁶` of `6·10⁻⁶` and underflows at every node of `[0, 1]` and
//! `[1, ∞)`.  (Up to 0.52 such nodes had no bound at all, `exp` of a
//! ball beyond `2⁹` wide refusing even far below the exponent range.)
//!
//! The integrand is the symbolic body, evaluated at each node by `evalf`'s
//! own node evaluator with its error bounds (the nodes that do not depend on
//! the variable once per precision).  A node is evaluated at the working
//! precision first; when its error, weighted, exceeds its share of the
//! tolerance — a cancellation the shift did not remove, `1 − (1 − x)²` — it
//! is re-evaluated at more bits (with the node held to the precision that
//! keeps its distance to an endpoint without a shifted body), up to a cap.
//!
//! Interior points where the integrand is not smooth are split off first:
//! those of the definite integrator's breakpoint scanner (poles, branch
//! points, kinks of `|·|`, steps, `Piecewise` boundaries), its periodic
//! families enumerated between the exact bounds; the kinks and steps it
//! cannot place (`sign(cos x²)`, `floor(x³ − x)`) are located by sampling
//! their argument and narrowing each crossing between certified signs to
//! `2^−(p + 16)` of the range, and a family of more than 10,000 members, or
//! a crossing that cannot be located, refuses the integral.  A finite piece
//! on which tanh-sinh resolves nothing by level 6 (an oscillation its nodes,
//! crowded at the ends, do not resolve) is tried by Gauss–Legendre of
//! increasing degree, accepted on the same criterion; a piece that converges
//! by neither within its share of the work is bisected, as mpmath's
//! `quadsubdiv` does, and an infinite one loses a finite piece next to its
//! end (`[a, ∞)` at `max(a + 1, 2a)`) — the algebraic map converges slowly on
//! an exponentially decaying integrand, and the tail left is small.  A
//! finite piece over which a `sin`/`cos`/`tan` of a linear argument
//! oscillates more often than any bisection could resolve refuses the
//! integral at once (`Quad::unresolvable`) instead of spending the work
//! budget on it (and the closed form is sought next).  A piece integrated
//! after others needs only a share of their tolerance in absolute terms.
//!
//! # What is certified
//!
//! At working precision `p` a piece is accepted at level `k ≥ 4` when
//!
//! * the estimates of levels `k − 1` and `k` differ by at most `τ =
//!   max(2^−(p − 48)·Σ|w f|, 2·R)` (the trapezoidal error of tanh-sinh falls
//!   doubly exponentially in `1/h`: each level about doubles the correct
//!   digits, so the difference bounds the error of level `k − 1` and far
//!   exceeds that of level `k`); Gauss–Legendre likewise, two successive
//!   degrees from 24 points on,
//! * the terms at the two outermost nodes on each side, spaced `½` in `t`,
//!   are below `2^−(p + 24)·Σ|w f|` (the truncated tail), and
//! * `R`, the sum of the weighted error bounds of the integrand values plus
//!   the rounding of the weights and of the sum, is accounted for rigorously.
//!
//! The reported bound is the difference of the last two levels plus the tail
//! and `R`; `evalf` prints only the digits it covers, re-evaluates at a
//! higher precision otherwise, and refuses beyond its limit.  The first
//! criterion is mpmath's (its estimate is the same difference, extrapolated
//! more aggressively); it is a heuristic, not a proof: an integrand with a
//! feature narrower than the node spacing at level 4 that no node sees, or
//! an unresolved interior singularity the scanner misses whose effect on two
//! successive levels happens to agree, can pass it (so can a pair of
//! crossings of a kink argument between two of its samples).  Up to 0.49 the
//! kinks the scanner could not place were not split off at all, and the
//! criterion passed across them: `∫₀³ sign(cos x² − 1/7) dx` was `0.5178`
//! (truly `0.5360`).  A piece that does not
//! meet it within the work limit reports the difference it reached — `evalf`
//! certifies the digits that covers — and an integral whose terms do not
//! decay (a divergent or oscillatory infinite integral) is refused.
//!
//! # Cost
//!
//! A node is a handful of `BigFloat` operations (computed once per
//! quadrature and shared by all pieces) plus one evaluation of the
//! variable-dependent part of the body; level `k` has about `9·2^(k−1)`
//! nodes, at most [`MAX_EVALS`] evaluations in all per working precision
//! (a deterministic [`Work`] budget: up to 0.48 there was also a 20 s
//! wall-clock limit, so the answer depended on the speed of the machine).
//! A quadrature that ran out of nodes is remembered ([`recall`]): `evalf`'s
//! search at higher precisions gets its answer again at once.  Gauss–Legendre nodes (`O(n²)` operations
//! each set) are cached per thread.

use super::ball::CBall;
use super::wp::{RM, Wp};
use crate::base::bigfloat;
use astro_float::{BigFloat, Consts};
use num_bigint::BigInt;
use num_rational::Ratio;
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::debug;

use super::accuracy::{self, Bound, ErrExp};
use super::{
    ErrMap, eval_node_with_error, evaluate_tree, exponent_range_error, is_condition, root_cause,
};
use crate::base::arena::Arena;
use crate::base::bigcomplex::{Complex, c_add, c_sub, c_zero};
use crate::base::budget::{BudgetKind, Work};
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;
use num_traits::{One, Zero};

/// Bits by which the tolerance at working precision `p` is wider than
/// `2⁻ᵖ` (relative to `Σ|w f|`): `evalf`'s first precision carries about 60
/// bits more than the digits need (`3.4d + 64` bits for `3.32d + 4`, and at
/// least 128 bits for at most 64), so `p − 48` leaves 12 to spare.
const TARGET_GUARD: usize = 48;
/// Bits added to the working precision for the integrand's values.
const EVAL_GUARD: usize = 16;
/// Bits added to the evaluation precision for the nodes and weights.
const NODE_GUARD: usize = 24;
/// The sum's precision over the evaluation precision.
const SUM_GUARD: usize = 16;
/// The first level whose agreement with the level before is accepted.
const MIN_LEVEL: u32 = 4;
/// The deepest level (`h = 2⁻¹²`).
const MAX_LEVEL: u32 = 12;
/// A side of the walk does not stop before `t = 5/2` (`δ ≈ 10⁻⁸`, `x ≈
/// 10⁸` on an infinite range), however small its terms — nor, towards an
/// infinite end, before its nodes reach where the integrand's terms cross
/// over (see [`Integrand::crossover_reach`]).
const T_MIN_HALVES: u64 = 5;
/// Nor does it go beyond `t = 12` (`δ ≈ 10⁻¹¹⁰⁰⁰⁰`): terms that have not
/// decayed by then belong to a divergent integral.
const T_MAX_HALVES: u64 = 24;

/// `log₂` of the distance from the centre of an infinite piece reached by
/// the level-1 node `t = j/2`: `2/δ ≈ e^{π·sinh t}` (`2³⁶⁸⁸³⁶` at `t = 12`).
fn walk_reach(j: u64) -> f64 {
    std::f64::consts::PI * (j as f64 / 2.0).sinh() * std::f64::consts::LOG2_E
}
/// Most integrand evaluations for one integral (all pieces, refinements
/// included): the [`Work`] budget of kind
/// [`BudgetKind::QuadratureEvaluations`].
const MAX_EVALS: u64 = 30_000;
/// Most pieces (breakpoints and bisections).
const MAX_PIECES: usize = 64;
/// Deepest bisection of a piece.
const MAX_DEPTH: u32 = 8;
/// Periods of an oscillation beyond which a piece is not resolved: a
/// Gauss–Legendre rule of `n` points needs about `πP/2` of them for `P`
/// periods, and the highest degree has 192.
const MAX_PERIODS_PER_PIECE: f64 = 256.0;
/// The absolute tolerance a piece may stop at is `2^−FLOOR_SHARE` of the
/// tolerance of the pieces done before it (see `Quad::abs_floor`).
const FLOOR_SHARE: f64 = 6.0;
/// A node's weighted error is accepted below `2^−NODE_SHARE` of the
/// tolerance (the tolerance covers 2¹⁴ such nodes four times over).
const NODE_SHARE: f64 = 18.0;
/// The highest Gauss–Legendre degree (`3·2^(m−1)` points: 192).
const MAX_GL_DEGREE: u32 = 7;
/// The first Gauss–Legendre degree whose agreement with the one before is
/// accepted (24 against 12 points).
const MIN_GL_DEGREE: u32 = 4;
/// Gauss–Legendre node sets kept per thread.
const GL_CACHE: usize = 8;
/// Precisions kept in the integrand's constant cache.
const MAX_PRECISIONS: usize = 24;

// ═══════════════════════════════════════════════════════════════════════════
// The integrand
// ═══════════════════════════════════════════════════════════════════════════

/// The integrand `root` as a function of `var`, evaluated at a `BigFloat`
/// node with `evalf`'s error bounds: the nodes that depend on `var` in
/// post-order by [`eval_node_with_error`] at every node, the others once per
/// precision by [`evaluate_tree`].
pub(super) struct Integrand<'a> {
    arena: &'a Arena,
    root: ExprId,
    var: ExprId,
    /// The nodes that depend on `var`, in post-order, without `var` itself
    /// and without conditions (`Piecewise` reads their operands).
    dependent: Vec<ExprId>,
    /// The variable-free nodes the dependent ones read, with their
    /// post-orders.
    frontier: Vec<(ExprId, Vec<ExprId>)>,
    /// Per precision: the values and bounds of the frontier, and the latest
    /// values of the dependent nodes.
    scratch: FxHashMap<usize, (FxHashMap<ExprId, Complex>, ErrMap)>,
}

impl<'a> Integrand<'a> {
    pub(super) fn new(arena: &'a Arena, root: ExprId, var: ExprId) -> Self {
        let post = walk::post_order_ids(arena, root);
        let mut dep: FxHashSet<ExprId> = FxHashSet::default();
        for &id in &post {
            if id == var || arena.node(id).children().iter().any(|c| dep.contains(c)) {
                dep.insert(id);
            }
        }
        let mut dependent = Vec::new();
        let mut frontier = Vec::new();
        let mut seen: FxHashSet<ExprId> = FxHashSet::default();
        if !dep.contains(&root) {
            frontier.push((root, post.clone()));
        }
        for &id in &post {
            if !dep.contains(&id) {
                continue;
            }
            let node = arena.node(id);
            if id != var && (id == root || !is_condition(node)) {
                dependent.push(id);
            }
            let mut stack: Vec<ExprId> = node.children().to_vec();
            while let Some(c) = stack.pop() {
                if dep.contains(&c) {
                    continue;
                }
                let child = arena.node(c);
                if is_condition(child) {
                    stack.extend(child.children());
                    continue;
                }
                if seen.insert(c) {
                    frontier.push((c, walk::post_order_ids(arena, c)));
                }
            }
        }
        Integrand {
            arena,
            root,
            var,
            dependent,
            frontier,
            scratch: FxHashMap::default(),
        }
    }

    /// The value and error bound of the integrand at `x ± 2^x_err`,
    /// evaluated at `prec` bits.
    pub(super) fn eval(
        &mut self,
        x: &BigFloat,
        x_err: ErrExp,
        w: &mut Wp<'_>,
    ) -> Result<CBall, SymplexError> {
        let prec = w.bits();
        self.eval_frontier(w);
        let Some((cache, errs)) = self.scratch.get_mut(&prec) else {
            return Err(internal("integrand cache"));
        };
        cache.insert(self.var, (x.clone(), BigFloat::new(prec)));
        errs.insert(self.var, Bound::real(x_err));
        let mut failed: FxHashMap<ExprId, SymplexError> = FxHashMap::default();
        for &id in &self.dependent {
            let evaluated = eval_node_with_error(self.arena, id, cache, errs, w).and_then(
                |CBall { mid: v, err: b }| match exponent_range_error(
                    self.arena, id, &v, cache, errs,
                ) {
                    Some(e) => Err(e),
                    None => Ok((v, b)),
                },
            );
            match evaluated {
                Ok((v, b)) => {
                    cache.insert(id, v);
                    errs.insert(id, b);
                }
                Err(e) => {
                    cache.remove(&id);
                    errs.remove(&id);
                    let e = root_cause(self.arena.node(id), e, &failed);
                    if id == self.root {
                        return Err(e);
                    }
                    failed.insert(id, e);
                }
            }
        }
        let value = cache
            .get(&self.root)
            .cloned()
            .ok_or_else(|| SymplexError::Unevaluable {
                reason: "the integrand has no value at a quadrature node".into(),
            })?;
        let bound = errs.get(&self.root).copied().unwrap_or(Bound::UNKNOWN);
        Ok(CBall::new(value, bound))
    }

    /// The scale of the integrand's constants: the largest `abs(log₂ abs(c))`
    /// over its variable-free parts `c` (evaluated at `w`'s precision; 0
    /// for none, `∞` for one without a finite value).  Where the terms of an
    /// integrand whose constants lie within `2^(±S)` cross over — the roots
    /// of `a·xᵐ = b·xⁿ` — lies within `2^(2S)`, so samples or nodes out to
    /// there see the integrand's features; `e^{x − 10¹⁰⁰⁰ − x²/10³⁰⁰⁰}`
    /// underflows out to `x ≈ 10¹⁰⁰⁰` and peaks at `10³⁰⁰⁰/2`.  For powers
    /// whose exponents differ by less than 1 that is not far enough
    /// (`x^(1/1000)` against `x^(1/999)` crosses at `2⁹⁹⁹⁰⁰⁰`): see
    /// [`Integrand::crossover_reach`].
    pub(super) fn constant_scale(&mut self, w: &mut Wp<'_>) -> f64 {
        self.eval_frontier(w);
        let Some((cache, _)) = self.scratch.get(&w.bits()) else {
            return f64::INFINITY;
        };
        let mut scale = 0.0f64;
        for (c, _) in &self.frontier {
            let Some(v) = cache.get(c) else {
                return f64::INFINITY;
            };
            for part in [&v.0, &v.1] {
                if !part.is_zero() {
                    scale = scale.max(accuracy::part_lg(part).abs());
                }
            }
        }
        scale
    }

    /// `log₂` of the distance from 0 within which the terms of the
    /// integrand cross over: `2S` for the [`constant_scale`](Integrand::constant_scale)
    /// `S` (terms `a·x^m`, `b·x^n` with constants within `2^(±S)` and orders
    /// at least 1 apart are equal within `2^(2S)`), or more for terms whose
    /// orders of growth differ by `D < 1` ([`close_reach`](Integrand::close_reach):
    /// `a·x^m = b·x^n` at `log₂|x| = log₂|b/a|/D`, read off the terms' sizes
    /// at `x = 2`).  The terms `2x^(1/1000)` and `x^(1/999)` of
    /// the exponent of `e^{2x^(1/1000) − x^(1/999) − 10⁵}` cross at
    /// `2⁹⁹⁹⁰⁰⁰`, where the integrand, below `e^{−10⁵}` at every point a
    /// sample or a node reaches, peaks at `e^{10²⁹⁷}`; up to 0.53 the scale
    /// of the constants alone (`2S`, 33 bits) was taken as that distance, and
    /// `integrate_numeric` of it over `[0, ∞)` was `0`.  `∞` for constants
    /// without a finite value.
    pub(super) fn crossover_reach(&mut self, w: &mut Wp<'_>) -> f64 {
        let twice = 2.0 * self.constant_scale(w);
        twice.max(self.close_reach(w))
    }

    /// The part of [`crossover_reach`](Integrand::crossover_reach) due to
    /// terms whose orders of growth differ by less than 1: `log₂|x|` of the
    /// farthest crossing at a large `|x|` ([`close_orders`](Integrand::close_orders)),
    /// `−∞` for none.  `a·x^m = b·x^n` (`m > n`) at `log₂|x| = log₂|b/a|/D`,
    /// read off the terms' sizes `t(2)` at `x = 2`: `(log₂|tₙ(2)| −
    /// log₂|tₘ(2)|)/D + 1`, a crossing at a small `|x|` when negative.  A
    /// logarithm `b·ln x` against a small power `a·x^D` meets it `log₂ ln
    /// x/D` farther out, and their rates of growth meet at `x^D = b/(aD)`,
    /// `log₂(1/D)/D` farther out (where `ln(x⁻²e^{5x^(1/100)})` turns from
    /// decreasing to increasing: at `2⁵³²`, although the power is the
    /// larger term from `x = 2` on).  Where a size is not known, `2S/D` in
    /// either direction.
    pub(super) fn close_reach(&mut self, w: &mut Wp<'_>) -> f64 {
        self.close_points(w)
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// `log₂|x|` of the points where terms of close orders of growth meet
    /// or where their rates do, the latter the extrema of a sum `a·x^m −
    /// b·x^n` (`m·a·x^m = n·b·x^n`, `log₂(n/m)/D` before they meet): the
    /// bump of `e^{1.5x^(1/30) − x^(1/29) − 500}`, `e^{1800}` high at
    /// `x ≈ 2⁴⁶⁶`, lies between the far-tail samples `2²⁵⁶` and `2⁵¹²`, where
    /// the integrand is below `e^{−400}`, and up to 0.53 `integrate_numeric`
    /// of it over `[0, ∞)` was `8.9·10⁻²¹⁵`.  See [`close_reach`](Integrand::close_reach).
    pub(super) fn close_points(&mut self, w: &mut Wp<'_>) -> Vec<f64> {
        let pairs = self.close_orders();
        if pairs.is_empty() {
            return Vec::new();
        }
        let twice = 2.0 * self.constant_scale(w);
        // The terms' sizes at `x = 2` (the nodes' values stay in the cache).
        let two = small(2);
        let _ = self.eval(&two, accuracy::EXACT, w);
        let ln2 = std::f64::consts::LN_2;
        let size = |cache: &FxHashMap<ExprId, Complex>, m: &Size| -> Option<f64> {
            let lg = |id: &ExprId| {
                cache
                    .get(id)
                    .map(accuracy::lg_abs)
                    .filter(|l| l.is_finite())
            };
            match m {
                Size::Of(id) => lg(id),
                // `|e·ln|b||` from `log₂|b|`.
                Size::LogOf(b, e) => {
                    let l = (lg(b)? * ln2 * e).abs().log2();
                    l.is_finite().then_some(l)
                }
            }
        };
        let mut points = Vec::new();
        let cache = self.scratch.get(&w.bits()).map(|s| &s.0);
        for p in &pairs {
            let sizes = cache.and_then(|c| Some((size(c, &p.top)?, size(c, &p.other)?)));
            // `log₂|b/a| = log₂|tₙ(2)/tₘ(2)| + D`, and `log₂|b| ≤
            // log₂|b·ln 2| + 0.53` for a logarithm.
            let slack = if p.logarithm { 0.6 } else { 0.0 };
            let lead = sizes.map_or(twice, |(top, other)| other - top + slack);
            if p.logarithm {
                // `a·x^D = b·ln x` at the larger root of `D·L = log₂|b/a| +
                // log₂(L·ln 2)`, `L = log₂ x` (the logarithm may overtake the
                // power first): the iteration from above converges to it.
                let mut l = (lead.max(0.0) + 64.0) / p.gap;
                for _ in 0..30 {
                    l = ((lead + (l.max(2.0) * ln2).log2()) / p.gap).max(1.0);
                }
                points.push(l + 1.0);
                points.push((lead - p.gap.log2()) / p.gap + 1.0);
            } else {
                let r = lead / p.gap + 1.0;
                points.push(r);
                // The rates: `log₂(n/m)/D` earlier, for a lesser order `n > 0`.
                if let Some(ratio) = p.ratio {
                    points.push(r + ratio.log2() / p.gap);
                }
            }
        }
        points.retain(|r| r.is_finite());
        points
    }

    /// The pairs of terms of the integrand that compete at infinity with
    /// orders of growth, as powers of the variable, less than 1 apart (`D =
    /// 1/999000` for `2x^(1/1000) − x^(1/999)`): in a sum the terms of the
    /// leading order against the others, in a product the `exp` factors with
    /// the leading order of exponent against the logarithms of the powers
    /// (`D = 1/1000` for `x⁻²·e^{x^(1/1000)}`, which decays as `x⁻²` out to
    /// `2¹⁴²⁰⁰`) and the other `exp` factors.  Orders are exact rationals:
    /// `x`, a power of a term with an order, a product, a sum, `ln` and the
    /// bounded functions (order 0); a factor `e^u` whose `u` grows has none,
    /// nor has any other function, and a sum or product with a term without
    /// an order is not compared.  At most [`MAX_PAIRS`] pairs (beyond, the
    /// first ones).
    fn close_orders(&self) -> Vec<Close> {
        use num_traits::Signed;
        let one = Q::one();
        let depends: FxHashSet<ExprId> = self.dependent.iter().copied().collect();
        let mut orders: FxHashMap<ExprId, Option<Q>> = FxHashMap::default();
        orders.insert(self.var, Some(Q::one()));
        let mut pairs: Vec<Close> = Vec::new();
        let mut note = |terms: &[(Size, Q, bool)]| {
            let Some(top) = terms.iter().map(|t| &t.1).max().cloned() else {
                return;
            };
            for (lead, _, _) in terms.iter().filter(|t| t.1 == top) {
                for (other, p, logarithm) in terms {
                    let gap = &top - p;
                    if gap.is_positive() && gap < one && pairs.len() < MAX_PAIRS {
                        pairs.push(Close {
                            top: lead.clone(),
                            other: other.clone(),
                            gap: crate::base::numeric::ratio_to_f64(&gap).unwrap_or(1.0),
                            logarithm: *logarithm,
                            ratio: (p.is_positive() && top.is_positive())
                                .then(|| crate::base::numeric::ratio_to_f64(&(p / &top)))
                                .flatten(),
                        });
                    }
                }
            }
        };
        for &id in &self.dependent {
            let order_of = |c: &ExprId| -> Option<Q> {
                match orders.get(c) {
                    Some(o) => o.clone(),
                    None if depends.contains(c) => None,
                    None => Some(Q::zero()),
                }
            };
            let dependent = |c: &ExprId| *c == self.var || depends.contains(c);
            let order = match self.arena.node(id) {
                ExprNode::Add(children) => {
                    // A term of order 0 that depends on `x` may grow as a
                    // logarithm.
                    let terms: Option<Vec<(Size, Q, bool)>> = children
                        .iter()
                        .map(|c| {
                            order_of(c)
                                .map(|q| (Size::Of(*c), q.clone(), dependent(c) && q.is_zero()))
                        })
                        .collect();
                    terms.and_then(|terms| {
                        note(&terms);
                        terms.into_iter().map(|t| t.1).max()
                    })
                }
                ExprNode::Mul(children) => {
                    // The exponents of the `exp` factors against each other
                    // and against the logarithms of the powers.
                    let logs: Option<Vec<(Size, Q, bool)>> = children
                        .iter()
                        .filter(|c| dependent(c))
                        .filter_map(|c| match self.arena.node(*c) {
                            ExprNode::Exp(u) => Some(order_of(u).map(|q| {
                                (
                                    Size::Of(*u),
                                    if q.is_positive() { q } else { Q::zero() },
                                    false,
                                )
                            })),
                            ExprNode::Pow(b, e) => {
                                let e = crate::base::numeric::ratio_to_f64(self.arena.as_num(*e)?)?;
                                Some(order_of(c).map(|_| (Size::LogOf(*b, e), Q::zero(), true)))
                            }
                            _ if *c == self.var => {
                                Some(Some((Size::LogOf(self.var, 1.0), Q::zero(), true)))
                            }
                            // Bounded or unknown factors: not compared.
                            _ => None,
                        })
                        .collect();
                    if let Some(logs) = logs
                        && logs.iter().any(|t| t.1.is_positive())
                    {
                        note(&logs);
                    }
                    children
                        .iter()
                        .map(order_of)
                        .try_fold(Q::zero(), |s, q| q.map(|q| s + q))
                }
                ExprNode::Pow(b, e) => match self.arena.as_num(*e) {
                    Some(e) => order_of(b).map(|q| q * e),
                    None => None,
                },
                ExprNode::Exp(u) => order_of(u).filter(|q| !q.is_positive()).map(|_| Q::zero()),
                ExprNode::Neg(u) | ExprNode::Abs(u) => order_of(u),
                ExprNode::Ln(u) => order_of(u).map(|_| Q::zero()),
                ExprNode::Sin(_)
                | ExprNode::Cos(_)
                | ExprNode::Atan(_)
                | ExprNode::Tanh(_)
                | ExprNode::Erf(_) => Some(Q::zero()),
                _ => None,
            };
            orders.insert(id, order);
        }
        pairs
    }

    /// The highest angular frequency `abs(a)` of a `sin`, `cos` or `tan` of
    /// a linear argument `a·x + b` in the integrand (0 for none): it
    /// oscillates `abs(a)·w/(2π)` times over a piece of width `w`.
    fn frequency(&self) -> f64 {
        let mut top = 0.0f64;
        for &id in &self.dependent {
            if let ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) = self.arena.node(id)
                && let Some(c) = linear_coefficient(self.arena, *a, self.var)
            {
                top = top.max(c);
            }
        }
        top
    }

    /// The values of the variable-free parts at `w`'s precision, computed
    /// once ([`Integrand::eval`]).
    fn eval_frontier(&mut self, w: &mut Wp<'_>) {
        let prec = w.bits();
        if self.scratch.contains_key(&prec) {
            return;
        }
        if self.scratch.len() >= MAX_PRECISIONS {
            self.scratch.clear();
        }
        let mut cache: FxHashMap<ExprId, Complex> = FxHashMap::default();
        let mut errs = ErrMap::default();
        for (c, order) in &self.frontier {
            if let Ok(CBall { mid: v, err: b }) =
                evaluate_tree(self.arena, *c, order, None, false, w)
            {
                cache.insert(*c, v);
                errs.insert(*c, b);
            }
        }
        self.scratch.insert(prec, (cache, errs));
    }
}

/// Most pairs of terms [`Integrand::close_orders`] reports.
const MAX_PAIRS: usize = 64;

/// The size of a term at `x = 2` for [`Integrand::crossover_reach`]: a node's
/// value, or `e·ln|b|` of a power `bᵉ` (a factor `e·ln b` of `ln|f|`).
#[derive(Clone, Debug)]
enum Size {
    Of(ExprId),
    LogOf(ExprId, f64),
}

/// Two terms whose orders of growth at infinity are `gap < 1` apart
/// ([`Integrand::close_orders`]); `logarithm` when the lesser one grows as
/// a logarithm; `ratio` the lesser order over the greater when both are
/// positive.
#[derive(Clone, Debug)]
struct Close {
    top: Size,
    other: Size,
    gap: f64,
    logarithm: bool,
    ratio: Option<f64>,
}

/// `abs(a)` when `e` is `a·var + b` with a rational `a` and a `b` free of
/// `var` (`var`, `a·var`, or a sum of one such term and constants).
fn linear_coefficient(arena: &Arena, e: ExprId, var: ExprId) -> Option<f64> {
    let term = |t: ExprId| -> Option<f64> {
        if t == var {
            return Some(1.0);
        }
        let ExprNode::Mul(children) = arena.node(t) else {
            return None;
        };
        let mut coefficient = 1.0f64;
        let mut found = false;
        for &c in children.iter() {
            if c == var && !found {
                found = true;
            } else {
                coefficient *= crate::base::numeric::ratio_to_f64(arena.as_num(c)?)?.abs();
            }
        }
        found.then_some(coefficient)
    };
    if let ExprNode::Add(children) = arena.node(e) {
        let mut dependent = children.iter().filter(|&&c| walk::contains(arena, c, var));
        let (Some(&t), None) = (dependent.next(), dependent.next()) else {
            return None;
        };
        return term(t);
    }
    term(e)
}

/// Is `body` (in `var`, both in `arena`, no other free symbol) bounded away
/// from 0 on the closed interval `[lo, hi]` of the real line (`lo ≤ hi`)?
/// The body is evaluated as a quadrature node is, at the midpoint with the
/// half-width (and the midpoint's rounding) as the variable's error, which
/// encloses its values over the whole interval; `true` when the box of
/// that ball excludes 0 (its real or its imaginary part keeps a sign),
/// `false` when no bound shows it (a singularity, a zero or a step in the
/// interval, an interval too wide for the bound).
///
/// The definite integrator's proof that a pole of a rational factor is not
/// integrable rests on it: the other factors stay away from 0 next to the
/// pole, so they cannot cancel it.
pub(crate) fn nonzero_on(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: &Ratio<BigInt>,
    hi: &Ratio<BigInt>,
) -> bool {
    /// The working precision of the evaluation.
    const PREC: usize = 128;
    if lo > hi {
        return false;
    }
    let Ok(mut cc) = Consts::new() else {
        return false;
    };
    let two = Ratio::from_integer(BigInt::from(2));
    let centre = (lo + hi) / &two;
    let Some(half) = crate::base::numeric::ratio_to_f64(&((hi - lo) / &two)) else {
        return false;
    };
    if !half.is_finite() {
        return false;
    }
    // The midpoint is within `2^(−PREC)` of `centre` relatively
    // (`ratio_to_bigfloat`); the half-width as an `f64` within a few ulps.
    let x = crate::base::numeric::ratio_to_bigfloat(&centre, PREC, RM);
    let rounding = ulp2(&x, PREC);
    let width = if half > 0.0 {
        half.log2() + 1e-9
    } else {
        accuracy::EXACT
    };
    let x_err = accuracy::lsum(width, rounding);
    let mut f = Integrand::new(arena, body, var);
    let Ok(CBall { mid: v, err: b }) = f.eval(&x, x_err, &mut Wp::new(PREC, &mut cc)) else {
        return false;
    };
    if b.is_unknown() || v.0.is_nan() || v.1.is_nan() || v.0.is_inf() || v.1.is_inf() {
        return false;
    }
    // `|part| ≥ 2^(mag − 1)`: clear of its error with a bit to spare.
    let clear = |part: &BigFloat, e: ErrExp| {
        accuracy::part_mag(part).is_some_and(|m| (m - 1) as f64 > e + 1.0)
    };
    clear(&v.0, b.re) || clear(&v.1, b.im)
}

fn internal(what: &str) -> SymplexError {
    SymplexError::ComputationFailed {
        operation: "evalf",
        reason: format!("quadrature: missing {what}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Endpoints and pieces
// ═══════════════════════════════════════════════════════════════════════════

/// A finite endpoint of a piece, available at any precision.
#[derive(Clone, Debug)]
pub(super) enum End {
    /// An exact rational.
    Rational(Ratio<BigInt>),
    /// An exact binary number (a split point chosen by the bisection).
    Float(BigFloat),
    /// A real constant expression, evaluated at the precision a node needs.
    Expr(ExprId),
    /// A value known to a fixed precision (a bound that depends on an
    /// enclosing bound variable), with its error bound.
    Value(BigFloat, ErrExp),
}

impl End {
    /// The end as an exact expression in `pa` (a rational or a constant
    /// expression); `None` for a binary split point or a value known to a
    /// fixed precision.
    pub(super) fn exact(&self, pa: &mut Arena) -> Option<ExprId> {
        match self {
            End::Rational(q) => Some(pa.num_ratio(q.clone())),
            End::Expr(id) => Some(*id),
            End::Float(_) | End::Value(..) => None,
        }
    }

    /// Are `self` and `other` the same number by construction?
    fn same(&self, other: &End) -> bool {
        match (self, other) {
            (End::Rational(a), End::Rational(b)) => a == b,
            (End::Expr(a), End::Expr(b)) => a == b,
            (End::Float(a), End::Float(b)) => a.cmp(b) == Some(0),
            _ => false,
        }
    }
}

/// A piece of the integration range and its map from the tanh-sinh
/// variable `s ∈ (−1, 1)`.
#[derive(Clone, Debug)]
pub(super) enum Piece {
    /// `[a, b]`: `x = (a + b)/2 + C·s`, `C = (b − a)/2`.
    Finite(End, End),
    /// `[a, ∞)`: `x = a − 1 + 2/(s + 1)`.
    Upper(End),
    /// `(−∞, b]`: `x = b + 1 − 2/(s + 1)`.
    Lower(End),
}

/// Which side of the centre a node lies on: towards the lower or the upper
/// limit of the piece.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Lo,
    Hi,
}

/// A tanh-sinh abscissa `t > 0`: the distance `δ = 1 − |s|` of its node to
/// the end of `(−1, 1)` and its weight `w`, with a bound on their relative
/// error (`log₂`).
struct TsNode {
    delta: BigFloat,
    w: BigFloat,
    rel: ErrExp,
}

/// A node placed in a piece: `x` (with the error bound of the endpoint it
/// was placed from and its rounding), its full weight `W` (step included)
/// with a bound on its relative error, and the precision at which `x` keeps
/// the working precision relative to its distance to that endpoint.
struct Placed {
    x: BigFloat,
    x_err: ErrExp,
    w: BigFloat,
    w_rel: ErrExp,
    need: usize,
    /// The shifted body to evaluate at `x` (an index into `variants`), or
    /// the body itself.
    which: Option<usize>,
}

/// The weighted value of a node: `W·f(x)`, a bound on its error, whether
/// `f(x)` was exactly real, and whether it was seen: not a 0 that
/// underflowed the exponent range ([`underflowed`]), which says nothing of
/// the integrand's size next to the node.
struct Term {
    value: Complex,
    err: ErrExp,
    real: bool,
    seen: bool,
}

/// The result of one piece.
struct PieceOut {
    value: Complex,
    /// Absolute error bound (`log₂`).
    err: ErrExp,
    /// `log₂ Σ|W f|` (with the values' error bounds).
    l1: ErrExp,
    converged: bool,
    /// The failure was the discretisation's (bisection may help), not the
    /// accuracy of the values.
    discretisation: bool,
    real: bool,
    /// The most bits two successive levels agreed to, relative to
    /// `Σ|W f|`.
    digits: f64,
    /// Some node was seen ([`Term::seen`]).  A piece where none was is
    /// blind: its sum is 0 and its bound that of the underflows, whatever
    /// lies between the nodes (see [`Quad::run`]).
    seen: bool,
}

/// The result of a quadrature: the sum, its error bound (`log₂`), whether
/// every value was exactly real, whether every piece converged, and whether
/// the pieces that did not were limited by the discretisation (not by the
/// accuracy of the values, which a higher precision improves).
pub(super) struct RunOut {
    pub(super) value: Complex,
    pub(super) err: ErrExp,
    pub(super) real: bool,
    pub(super) converged: bool,
    pub(super) discretisation: bool,
}

/// Is the integrand's value `v ± b` at a node a 0 that underflowed the
/// exponent range (no measurement of its size, unlike an exact 0 or a 0
/// within a rounding error)?
fn underflowed(v: &Complex, b: Bound) -> bool {
    v.0.is_zero() && v.1.is_zero() && accuracy::is_underflow(b.joint())
}

/// `log₂|x|` (−∞ for 0).
fn lg(x: &BigFloat) -> f64 {
    accuracy::part_lg(x)
}

/// Two units in the last place of `x` at `p` bits (`log₂`), exact for 0.
fn ulp2(x: &BigFloat, p: usize) -> ErrExp {
    accuracy::part_mag(x).map_or(accuracy::EXACT, |m| (m - p as i64 + 1) as f64)
}

/// `j·2⁻ᵏ`, exactly.
fn dyadic(j: u64, k: u32) -> BigFloat {
    let mut t = BigFloat::from_u64(j, 64);
    if let Some(e) = t.exponent() {
        t.set_exponent(e - k as i32);
    }
    t
}

/// `x·2^s` for an integer `s`, exactly.
pub(super) fn scale2(x: &BigFloat, s: i32) -> BigFloat {
    let mut y = x.clone();
    if !y.is_zero()
        && let Some(e) = y.exponent()
    {
        y.set_exponent(e + s);
    }
    y
}

pub(super) fn small(n: u64) -> BigFloat {
    BigFloat::from_u64(n, 64)
}

/// Round `p` up to a multiple of 32 bits (fewer distinct precisions for
/// the integrand's constant cache).
fn quantize(p: usize) -> usize {
    p.div_ceil(32) * 32
}

// ═══════════════════════════════════════════════════════════════════════════
// The quadrature
// ═══════════════════════════════════════════════════════════════════════════

/// The state of one integral's quadrature.
pub(super) struct Quad<'a> {
    f: Integrand<'a>,
    /// The body with the variable shifted to an anchor (a bound or a
    /// breakpoint `e`): `body(e + x)` (`true`) or `body(e − x)` (`false`),
    /// for the nodes next to `e`, placed at their distance from it.
    variants: Vec<(End, bool, Integrand<'a>)>,
    /// `evalf`'s working precision for the integral.
    prec: usize,
    /// The integrand's precision.
    pe: usize,
    /// The highest precision a node may be evaluated at.
    p_cap: usize,
    work: Work,
    ends: FxHashMap<(ExprId, usize), (BigFloat, ErrExp)>,
    /// The tanh-sinh abscissae computed so far, by level and index (shared
    /// by both sides and all pieces).
    nodes: FxHashMap<(u32, u64), std::rc::Rc<TsNode>>,
    /// An absolute tolerance every piece may stop at (`log₂`): a share of
    /// the tolerance of the pieces done so far, so that a piece far smaller
    /// than the integral (the tail of a split infinite range) is not held to
    /// its own relative accuracy.
    abs_floor: ErrExp,
    /// Where the integrand's terms cross over ([`Integrand::crossover_reach`]),
    /// once computed.
    reach: Option<f64>,
    /// The integrand's frequency ([`Integrand::frequency`]), once computed.
    frequency: Option<f64>,
}

impl<'a> Quad<'a> {
    /// A quadrature of `body` in `var` (both in `arena`) for `evalf`'s
    /// working precision `prec`.
    pub(super) fn new(
        arena: &'a Arena,
        body: ExprId,
        var: ExprId,
        variants: Vec<(End, bool, ExprId)>,
        prec: usize,
    ) -> Self {
        let pe = prec + EVAL_GUARD;
        let max = arena.config.max_evalf_precision as usize;
        Quad {
            f: Integrand::new(arena, body, var),
            variants: variants
                .into_iter()
                .map(|(e, plus, b)| (e, plus, Integrand::new(arena, b, var)))
                .collect(),
            prec,
            pe,
            p_cap: (12 * pe + 1024).min(max.max(pe)),
            work: Work::new(BudgetKind::QuadratureEvaluations, MAX_EVALS),
            ends: FxHashMap::default(),
            nodes: FxHashMap::default(),
            abs_floor: accuracy::EXACT,
            reach: None,
            frequency: None,
        }
    }

    /// [`Integrand::crossover_reach`] of the body, computed once.
    fn reach(&mut self, cc: &mut Consts) -> f64 {
        if let Some(s) = self.reach {
            return s;
        }
        let s = self.f.crossover_reach(&mut Wp::new(self.pe, cc));
        self.reach = Some(s);
        s
    }

    /// A quarter of the evaluations: a first try that has a fallback.
    pub(super) fn lighten(&mut self) {
        self.work = Work::new(BudgetKind::QuadratureEvaluations, MAX_EVALS / 4);
    }

    /// The value of a finite endpoint at `p` bits and its error bound.
    pub(super) fn end_value(
        &mut self,
        e: &End,
        w: &mut Wp<'_>,
    ) -> Result<(BigFloat, ErrExp), SymplexError> {
        let p = w.bits();
        match e {
            End::Rational(q) => {
                let v = crate::base::numeric::ratio_to_bigfloat(q, p, RM);
                let err = accuracy::rational_part(q, &v, p);
                Ok((v, err))
            }
            End::Float(v) => Ok((v.clone(), accuracy::EXACT)),
            End::Value(v, err) => Ok((v.clone(), *err)),
            End::Expr(id) => {
                if let Some(hit) = self.ends.get(&(*id, p)) {
                    return Ok(hit.clone());
                }
                let order = walk::post_order_ids(self.f.arena, *id);
                let CBall { mid: v, err: b } =
                    evaluate_tree(self.f.arena, *id, &order, None, true, w)?;
                if !accuracy::exact_zero(&v.1, b.im) || b.is_unknown() {
                    return Err(SymplexError::Unevaluable {
                        reason: "integration bounds of a definite integral must be real".into(),
                    });
                }
                let out = (v.0, b.re);
                self.ends.insert((*id, p), out.clone());
                Ok(out)
            }
        }
    }

    /// The shifted body for the nodes next to `end` on its upper side
    /// (`plus`) or its lower side.
    fn variant(&self, end: &End, plus: bool) -> Option<usize> {
        self.variants
            .iter()
            .position(|(e, p, _)| *p == plus && e.same(end))
    }

    /// The tanh-sinh abscissa `t = j·2⁻ᵏ` at `pw` bits, computed once.
    fn ts_node(&mut self, j: u64, k: u32, w: &mut Wp<'_>) -> std::rc::Rc<TsNode> {
        if let Some(n) = self.nodes.get(&(k, j)) {
            return n.clone();
        }
        let n = std::rc::Rc::new(self.compute_ts_node(j, k, w));
        self.nodes.insert((k, j), n.clone());
        n
    }

    /// [`Quad::ts_node`], computed.
    fn compute_ts_node(&self, j: u64, k: u32, wpc: &mut Wp<'_>) -> TsNode {
        let pw = wpc.bits();
        let t = dyadic(j, k);
        let pi = wpc.cc.pi(pw, RM);
        let s = t.sinh(pw, RM, wpc.cc);
        let c = t.cosh(pw, RM, wpc.cc);
        let arg = pi.mul(&s, pw, RM); // π·sinh t = 2u
        let q = arg.neg().exp(pw, RM, wpc.cc); // e^{−2u}
        let opq = small(1).add(&q, pw, RM);
        let delta = scale2(&q, 1).div(&opq, pw, RM); // 2q/(1 + q) = 1 − tanh u
        let w = scale2(&pi.mul(&c, pw, RM).mul(&q, pw, RM), 1).div(&opq.mul(&opq, pw, RM), pw, RM);
        // `exp` of an argument known to a relative 2^−pw loses its magnitude.
        let a = (j as f64 * (-(k as f64)).exp2()).sinh() * std::f64::consts::PI;
        let rel = (1.0 + a).log2() + 4.0 - pw as f64;
        TsNode { delta, w, rel }
    }

    /// Place node `node` (or the centre, `None`) of level `k` in `piece` at
    /// `p` bits.  `half` is `C` of a finite piece.
    #[allow(clippy::too_many_arguments)]
    fn place(
        &mut self,
        piece: &Piece,
        half: &(BigFloat, ErrExp),
        side: Option<Side>,
        node: Option<&TsNode>,
        k: u32,
        wpc: &mut Wp<'_>,
    ) -> Result<Placed, SymplexError> {
        let p = wpc.bits();
        let pw = self.pe + NODE_GUARD;
        let step = -(k as i32);
        let (c, c_err) = half;
        let c_rel = if c.is_zero() {
            accuracy::EXACT
        } else {
            accuracy::shift(*c_err, -lg(c))
        };
        let need_for = |end: &BigFloat, dist: &BigFloat, pe: usize| -> usize {
            match (accuracy::part_mag(end), accuracy::part_mag(dist)) {
                (Some(me), Some(md)) if me > md => pe + (me - md) as usize + 8,
                _ => pe,
            }
        };
        let pe = self.pe;
        let one = small(1);
        let two = small(2);
        let (Some(node), Some(side)) = (node, side) else {
            // The centre: s = 0, w = π/2.
            let pi_half = scale2(&wpc.cc.pi(pw, RM), -1);
            return Ok(match piece {
                Piece::Finite(a, _) => {
                    let (av, ae) = self.end_value(a, wpc)?;
                    let x = av.add(c, p, RM);
                    Placed {
                        x_err: accuracy::lsum(accuracy::lsum(ae, *c_err), ulp2(&x, p)),
                        x,
                        w: scale2(&c.mul(&pi_half, pw, RM), step),
                        w_rel: accuracy::lsum(c_rel, 2.0 - pw as f64),
                        need: p,
                        which: None,
                    }
                }
                Piece::Upper(a) => {
                    let (av, ae) = self.end_value(a, wpc)?;
                    let x = av.add(&one, p, RM);
                    Placed {
                        x_err: accuracy::lsum(ae, ulp2(&x, p)),
                        x,
                        w: scale2(&pi_half, step + 1),
                        w_rel: 2.0 - pw as f64,
                        need: p,
                        which: None,
                    }
                }
                Piece::Lower(b) => {
                    let (bv, be) = self.end_value(b, wpc)?;
                    let x = bv.sub(&one, p, RM);
                    Placed {
                        x_err: accuracy::lsum(be, ulp2(&x, p)),
                        x,
                        w: scale2(&pi_half, step + 1),
                        w_rel: 2.0 - pw as f64,
                        need: p,
                        which: None,
                    }
                }
            });
        };
        let d = &node.delta;
        let base_rel = node.rel + 3.0;
        Ok(match (piece, side) {
            (Piece::Finite(a, _), Side::Lo) | (Piece::Finite(_, a), Side::Hi) => {
                let dist = c.mul(d, pw, RM);
                if let Some(i) = self.variant(a, side == Side::Lo) {
                    return Ok(Placed {
                        x_err: ulp2(&dist, pw),
                        x: dist,
                        w: scale2(&c.mul(&node.w, pw, RM), step),
                        w_rel: accuracy::lsum(c_rel, base_rel),
                        need: p,
                        which: Some(i),
                    });
                }
                let (av, ae) = self.end_value(a, wpc)?;
                let x = if side == Side::Lo {
                    av.add(&dist, p, RM)
                } else {
                    av.sub(&dist, p, RM)
                };
                Placed {
                    x_err: accuracy::lsum(ae, ulp2(&x, p)),
                    need: need_for(&av, &dist, pe),
                    x,
                    w: scale2(&c.mul(&node.w, pw, RM), step),
                    w_rel: accuracy::lsum(c_rel, base_rel),
                    which: None,
                }
            }
            // The finite end of a half-infinite piece: x = a ± δ/(2 − δ).
            (Piece::Upper(a), Side::Lo) | (Piece::Lower(a), Side::Hi) => {
                let tmd = two.sub(d, pw, RM);
                let dist = d.div(&tmd, pw, RM);
                let factor = scale2(&one.div(&tmd.mul(&tmd, pw, RM), pw, RM), 1);
                if let Some(i) = self.variant(a, side == Side::Lo) {
                    return Ok(Placed {
                        x_err: ulp2(&dist, pw),
                        x: dist,
                        w: scale2(&node.w.mul(&factor, pw, RM), step),
                        w_rel: base_rel + 1.0,
                        need: p,
                        which: Some(i),
                    });
                }
                let (av, ae) = self.end_value(a, wpc)?;
                let x = if side == Side::Lo {
                    av.add(&dist, p, RM)
                } else {
                    av.sub(&dist, p, RM)
                };
                Placed {
                    x_err: accuracy::lsum(ae, ulp2(&x, p)),
                    need: need_for(&av, &dist, pe),
                    x,
                    w: scale2(&node.w.mul(&factor, pw, RM), step),
                    w_rel: base_rel + 1.0,
                    which: None,
                }
            }
            // The infinite end: x = a − 1 + 2/δ, x = b + 1 − 2/δ.
            (Piece::Upper(a), Side::Hi) | (Piece::Lower(a), Side::Lo) => {
                let far = scale2(&one.div(d, pw, RM), 1);
                let (av, ae) = self.end_value(a, wpc)?;
                let x = if side == Side::Hi {
                    av.sub(&one, p, RM).add(&far, p, RM)
                } else {
                    av.add(&one, p, RM).sub(&far, p, RM)
                };
                let factor = scale2(&one.div(&d.mul(d, pw, RM), pw, RM), 1);
                Placed {
                    x_err: accuracy::lsum(ae, ulp2(&x, p)),
                    need: p,
                    x,
                    w: scale2(&node.w.mul(&factor, pw, RM), step),
                    w_rel: base_rel + 1.0,
                    which: None,
                }
            }
        })
    }

    /// The weighted value `W·f(x)` of a node, evaluated at the working
    /// precision first and, when its weighted error exceeds the larger of
    /// `budget` and its own size `·2^−(pe − 8)`, again with the node held to
    /// more bits, up to the cap.
    #[allow(clippy::too_many_arguments)]
    fn term(
        &mut self,
        piece: &Piece,
        half: &(BigFloat, ErrExp),
        side: Option<Side>,
        node: Option<&TsNode>,
        k: u32,
        budget: ErrExp,
        cc: &mut Consts,
    ) -> Result<Term, SymplexError> {
        let psum = self.pe + SUM_GUARD;
        let mut p = self.pe;
        loop {
            self.work.charge(1, "evalf")?;
            let placed = self.place(piece, half, side, node, k, &mut Wp::new(p, cc))?;
            let f = match placed.which.and_then(|i| self.variants.get_mut(i)) {
                Some((_, _, g)) => g,
                None => &mut self.f,
            };
            let evaluated = f.eval(&placed.x, placed.x_err, &mut Wp::new(p, cc));
            let next = match evaluated {
                Ok(CBall { mid: fv, err: fb })
                    if !fb.is_unknown()
                        && !(fv.0.is_nan() || fv.0.is_inf() || fv.1.is_nan() || fv.1.is_inf()) =>
                {
                    let value = (fv.0.mul(&placed.w, psum, RM), fv.1.mul(&placed.w, psum, RM));
                    let size = accuracy::lg_abs(&value);
                    let err = accuracy::lsum(
                        accuracy::lsum(
                            accuracy::shift(fb.joint(), lg(&placed.w)),
                            accuracy::shift(placed.w_rel, size),
                        ),
                        accuracy::shift(1.0 - psum as f64, size),
                    );
                    let target = budget.max(size - (self.pe as f64 - 8.0));
                    let real = accuracy::exactly_real(&fv, fb);
                    if err <= target || p >= self.p_cap {
                        let seen = !underflowed(&fv, fb);
                        return Ok(Term {
                            value,
                            err,
                            real,
                            seen,
                        });
                    }
                    let deficit = (err - target).clamp(0.0, 1e6).ceil() as usize;
                    placed.need.max(p + deficit + 16)
                }
                Ok(_) | Err(_) if p >= self.p_cap => {
                    let why = match evaluated {
                        Err(e) => format!(": {e}"),
                        Ok(_) => String::new(),
                    };
                    return Err(not_converged(&format!(
                        "the integrand has no error bound at a node (x ≈ {}) even at {} bits{why}",
                        super::f64_of(&placed.x)
                            .map_or_else(|_| "?".to_string(), |v| format!("{v:e}")),
                        self.p_cap
                    )));
                }
                _ => placed.need.max(2 * p),
            };
            p = quantize(next).max(p + 32).min(self.p_cap);
        }
    }

    /// Integrate one piece by tanh-sinh, levels 1 to at most [`MAX_LEVEL`].
    fn piece(&mut self, piece: &Piece, cc: &mut Consts) -> Result<PieceOut, QuadError> {
        let psum = self.pe + SUM_GUARD;
        let pw = self.pe + NODE_GUARD;
        let half = match piece {
            Piece::Finite(a, b) => {
                let (av, ae) = self.end_value(a, &mut Wp::new(pw, cc))?;
                let (bv, be) = self.end_value(b, &mut Wp::new(pw, cc))?;
                let c = scale2(&bv.sub(&av, pw, RM), -1);
                let err = accuracy::lsum(accuracy::lsum(ae, be), ulp2(&c, pw));
                (c, err)
            }
            _ => (small(1), accuracy::EXACT),
        };
        // Level 1: the centre and the walk outwards on each side.
        let mut sum = c_zero(psum);
        let mut l1 = accuracy::EXACT; // log₂ Σ(|W f| + err)
        let mut rerr = accuracy::EXACT; // log₂ Σ err
        let mut real = true;
        let mut additions = 0usize;
        let centre = self.term(piece, &half, None, None, 1, accuracy::EXACT, cc)?;
        real &= centre.real;
        let mut seen = centre.seen;
        l1 = accuracy::lsum(
            l1,
            accuracy::lsum(accuracy::lg_abs(&centre.value), centre.err),
        );
        rerr = accuracy::lsum(rerr, centre.err);
        sum = c_add(&sum, &centre.value, psum, RM);
        additions += 1;
        // The walk outwards on each side: the level-1 terms' sizes, until
        // two in a row are negligible (and not before `t = 3`).
        let mut sizes: [Vec<ErrExp>; 2] = [Vec::new(), Vec::new()];
        for (si, side) in [Side::Lo, Side::Hi].into_iter().enumerate() {
            // Towards an infinite end the walk does not stop before its
            // nodes reach where the integrand's terms cross over
            // ([`Integrand::crossover_reach`]): beyond `x ≈ 10⁸` (`t = 5/2`)
            // a growth set off by a large constant is not seen, and up to
            // 0.52 `∫₁^∞ x⁻²e^{x/10⁷⁰⁰} dx` was `1`; up to 0.53 the walk
            // stopped short of the crossing of close powers, and
            // `∫₁^∞ x⁻²e^{x^(1/1000)} dx`, which diverges, was `2.7210`.  Where
            // no node reaches, nothing is seen.
            let reach_needed = match (piece, side) {
                (Piece::Upper(_), Side::Hi) | (Piece::Lower(_), Side::Lo) => self.reach(cc),
                _ => f64::NEG_INFINITY,
            };
            if reach_needed > walk_reach(T_MAX_HALVES) {
                debug!(
                    reach_needed,
                    "quadrature: the terms cross over beyond the nodes"
                );
                return Err(not_converged(&format!(
                    "the integrand's terms cross over near |x| = 2^{reach_needed:.0}, beyond the \
                     reach of the quadrature's nodes"
                ))
                .into());
            }
            let mut quiet = 0;
            let mut j = 1u64;
            loop {
                if j > T_MAX_HALVES {
                    debug!(
                        ?side,
                        sizes = ?sizes[si],
                        l1,
                        "quadrature: the walk found no negligible terms"
                    );
                    return Err(QuadError::NotDecaying);
                }
                let node = self.ts_node(j, 1, &mut Wp::new(pw, cc));
                let budget = accuracy::shift(l1, -((self.pe as f64) + 8.0));
                let t = self.term(piece, &half, Some(side), Some(node.as_ref()), 1, budget, cc)?;
                real &= t.real;
                seen |= t.seen;
                let size = accuracy::lsum(accuracy::lg_abs(&t.value), t.err);
                l1 = accuracy::lsum(l1, size);
                rerr = accuracy::lsum(rerr, t.err);
                sum = c_add(&sum, &t.value, psum, RM);
                additions += 1;
                sizes[si].push(size);
                let threshold = accuracy::shift(l1, -((self.pe as f64) + 8.0));
                if size <= threshold || accuracy::is_exact(size) {
                    quiet += 1;
                } else {
                    quiet = 0;
                }
                // `log₂` of the distance of the node from the centre on an
                // infinite piece, `2/δ`.
                let reach = 1.0 - lg(&node.delta);
                if quiet >= 2 && (j >= T_MAX_HALVES || (j >= T_MIN_HALVES && reach >= reach_needed))
                {
                    break;
                }
                j += 1;
            }
        }
        // The finer levels stop at the first level-1 node from which every
        // term was negligible (`cut`, in halves of `t`); the terms from there
        // on, and the omitted tail beyond them (below the last two), bound the
        // truncation.
        let threshold = accuracy::shift(l1, -((self.pe as f64) + 8.0));
        let mut cut = [0u64; 2];
        let mut tail = accuracy::EXACT;
        for (si, s) in sizes.iter().enumerate() {
            let mut first = s.len();
            while first > 0 && (s[first - 1] <= threshold || accuracy::is_exact(s[first - 1])) {
                first -= 1;
            }
            // `s[i]` is the node `j = i + 1`; the finer levels fill in up to
            // the first quiet one.
            cut[si] = (first as u64 + 1).min(s.len() as u64);
            for &v in &s[first.min(s.len())..] {
                tail = accuracy::lsum(tail, v);
            }
        }
        let ends = cut;
        let mut prev = sum.clone();
        let mut d_hist: Vec<ErrExp> = Vec::new();
        let mut out_err = accuracy::UNKNOWN;
        let mut converged = false;
        let mut discretisation = true;
        let mut digits = 0.0f64;
        for k in 2..=MAX_LEVEL {
            let new_nodes: u64 = ends.iter().map(|&e| e << (k - 2)).sum();
            if new_nodes > self.work.left() {
                debug!(
                    k,
                    "quadrature: the next level exceeds the evaluation budget"
                );
                break;
            }
            let budget = accuracy::shift(
                accuracy::shift(l1, -1.0),
                -((self.prec - TARGET_GUARD.min(self.prec)) as f64) - NODE_SHARE,
            )
            .max(accuracy::shift(self.abs_floor, -NODE_SHARE));
            let mut level = c_zero(psum);
            let mut level_l1 = accuracy::EXACT;
            let mut level_err = accuracy::EXACT;
            for (si, side) in [Side::Lo, Side::Hi].into_iter().enumerate() {
                let top = ends[si] << (k - 1);
                let mut j = 1u64;
                while j <= top {
                    let node = self.ts_node(j, k, &mut Wp::new(pw, cc));
                    let t =
                        self.term(piece, &half, Some(side), Some(node.as_ref()), k, budget, cc)?;
                    real &= t.real;
                    seen |= t.seen;
                    level_l1 =
                        accuracy::lsum(level_l1, accuracy::lsum(accuracy::lg_abs(&t.value), t.err));
                    level_err = accuracy::lsum(level_err, t.err);
                    level = c_add(&level, &t.value, psum, RM);
                    additions += 1;
                    j += 2;
                }
            }
            let halved = (scale2(&prev.0, -1), scale2(&prev.1, -1));
            sum = c_add(&halved, &level, psum, RM);
            additions += 1;
            l1 = accuracy::lsum(accuracy::shift(l1, -1.0), level_l1);
            rerr = accuracy::lsum(accuracy::shift(rerr, -1.0), level_err);
            let diff = accuracy::lg_abs(&c_sub(&sum, &prev, psum, RM));
            let rounding = accuracy::shift(l1, (additions as f64).log2() + 1.0 - psum as f64);
            let r = accuracy::lsum(rerr, rounding);
            let e = accuracy::lsum(accuracy::lsum(diff, r), tail);
            out_err = e;
            prev = sum.clone();
            let rel_tol = accuracy::shift(l1, -((self.prec - TARGET_GUARD.min(self.prec)) as f64))
                .max(self.abs_floor);
            let tol = rel_tol.max(accuracy::shift(r, 1.0));
            debug!(k, diff, tol, r, tail, "quadrature: level done");
            if k >= MIN_LEVEL && diff <= tol && tail <= tol - 2.0 {
                converged = true;
                digits = digits.max(l1 - diff);
                break;
            }
            if r > rel_tol && diff <= tol {
                // The values' error, not the discretisation, limits it.
                discretisation = false;
            }
            d_hist.push(diff - l1);
            digits = digits.max(l1 - diff);
            if self.hopeless(k, &d_hist, rel_tol - l1, &ends) {
                break;
            }
        }
        Ok(PieceOut {
            value: sum,
            err: out_err,
            l1,
            converged,
            discretisation,
            real,
            digits,
            seen,
        })
    }

    /// Is a piece unlikely to converge by the last level or within the
    /// budget?  `rel` holds `log₂` of the level differences relative to
    /// `Σ|W f|`, `target` the relative tolerance (both negative when small).
    /// Tanh-sinh about doubles the correct digits per level; from level 6 on
    /// fewer than 8 correct bits, a difference that falls slower than by a
    /// factor of `2^(1.3·digits)` (algebraic convergence: an unresolved
    /// singularity), or more levels than are left, give the piece up — to a
    /// bisection when it is finite.
    fn hopeless(&self, k: u32, rel: &[ErrExp], target: ErrExp, ends: &[u64; 2]) -> bool {
        let Some(&last) = rel.last() else {
            return false;
        };
        let next_cost: u64 = ends.iter().map(|&e| e << k.saturating_sub(1)).sum();
        if next_cost > self.work.left() {
            return true;
        }
        if k < 6 {
            return false;
        }
        let digits = -last;
        let needed = -target;
        if digits >= needed {
            return false;
        }
        // Nothing resolved by level 6 (an oscillation the step does not
        // resolve, a feature between the nodes): a bisection may.
        if digits < 8.0 {
            return true;
        }
        let levels = (needed / digits).log2().ceil().max(1.0) as u32;
        if levels > MAX_LEVEL.saturating_sub(k) {
            return true;
        }
        if rel.len() >= 2 {
            let before = -rel[rel.len() - 2];
            if before > 0.0 && digits < 1.3 * before {
                return true;
            }
        }
        false
    }

    /// The positive Gauss–Legendre abscissae of degree `m` (`n = 3·2^(m−1)`
    /// points, `m ≥ 2`) at `pw` bits with their weights, and a bound on their
    /// relative error.
    fn gl_nodes(&self, m: u32, pw: usize) -> GlNodes {
        let hit = GL_NODES.with(|c| {
            c.borrow()
                .iter()
                .find(|e| e.0 == (m, pw))
                .map(|e| e.1.clone())
        });
        if let Some(n) = hit {
            return n;
        }
        let nodes = std::rc::Rc::new(gauss_legendre(m, pw));
        GL_NODES.with(|c| {
            let mut c = c.borrow_mut();
            if c.len() >= GL_CACHE {
                c.remove(0);
            }
            c.push(((m, pw), nodes.clone()));
        });
        nodes
    }

    /// Integrate a finite piece by Gauss–Legendre quadrature of increasing
    /// degree, accepted when two successive degrees agree as tanh-sinh's
    /// levels must (mpmath's `GaussLegendre`): no node clusters at the ends,
    /// so a smooth oscillating integrand needs several times fewer nodes,
    /// while an endpoint singularity does not converge.
    fn gl_piece(&mut self, piece: &Piece, cc: &mut Consts) -> Result<PieceOut, SymplexError> {
        let pe = self.pe;
        let psum = pe + SUM_GUARD;
        let pw = pe + NODE_GUARD;
        let one = small(1);
        // `x = centre + C·s` on a finite piece; on a half-infinite one
        // `x = e ∓ 1 ± 2/(s + 1)` with weight factor `2/(s + 1)²` (mpmath's
        // `transform_nodes`), `centre` holding `e ∓ 1`.
        let (centre, c, end_err, upper) = match piece {
            Piece::Finite(a, b) => {
                let (av, ae) = self.end_value(a, &mut Wp::new(pw, cc))?;
                let (bv, be) = self.end_value(b, &mut Wp::new(pw, cc))?;
                let c = scale2(&bv.sub(&av, pw, RM), -1);
                let mid = scale2(&av.add(&bv, pw, RM), -1);
                let err = accuracy::lsum(accuracy::lsum(ae, be), ulp2(&mid, pw));
                (mid, c, err, None)
            }
            Piece::Upper(a) => {
                let (av, ae) = self.end_value(a, &mut Wp::new(pw, cc))?;
                (av.sub(&one, pw, RM), one.clone(), ae, Some(true))
            }
            Piece::Lower(b) => {
                let (bv, be) = self.end_value(b, &mut Wp::new(pw, cc))?;
                (bv.add(&one, pw, RM), one.clone(), be, Some(false))
            }
        };
        let rel_tol_bits = (self.prec - TARGET_GUARD.min(self.prec)) as f64;
        let mut prev: Option<Complex> = None;
        let mut out = PieceOut {
            value: c_zero(psum),
            err: accuracy::UNKNOWN,
            l1: accuracy::EXACT,
            converged: false,
            discretisation: true,
            real: true,
            digits: 0.0,
            seen: false,
        };
        let mut rel_hist: Vec<ErrExp> = Vec::new();
        for m in 2..=MAX_GL_DEGREE {
            let nodes = self.gl_nodes(m, pw);
            let (ref list, node_rel) = *nodes;
            if 2 * list.len() as u64 > self.work.left() {
                break;
            }
            let mut sum = c_zero(psum);
            let mut l1 = accuracy::EXACT;
            let mut rerr = accuracy::EXACT;
            let mut real = true;
            let mut seen = false;
            for (r, w) in list.iter() {
                for sign in [false, true] {
                    self.work.charge(1, "evalf")?;
                    let s = if sign { r.neg() } else { r.clone() };
                    let (x, weight) = match upper {
                        None => (centre.add(&c.mul(&s, pw, RM), pe, RM), c.mul(w, pw, RM)),
                        Some(up) => {
                            let sp1 = s.add(&one, pw, RM);
                            let u = scale2(&one.div(&sp1, pw, RM), 1);
                            let x = if up {
                                centre.add(&u, pe, RM)
                            } else {
                                centre.sub(&u, pe, RM)
                            };
                            (x, scale2(&w.div(&sp1.mul(&sp1, pw, RM), pw, RM), 1))
                        }
                    };
                    let x_err = accuracy::lsum(end_err, ulp2(&x, pe));
                    let CBall { mid: fv, err: fb } =
                        self.f.eval(&x, x_err, &mut Wp::new(pe, cc))?;
                    if fb.is_unknown()
                        || fv.0.is_nan()
                        || fv.0.is_inf()
                        || fv.1.is_nan()
                        || fv.1.is_inf()
                    {
                        return Err(not_converged("the integrand has no error bound at a node"));
                    }
                    let value = (fv.0.mul(&weight, psum, RM), fv.1.mul(&weight, psum, RM));
                    let size = accuracy::lg_abs(&value);
                    let err = accuracy::lsum(
                        accuracy::lsum(
                            accuracy::shift(fb.joint(), lg(&weight)),
                            accuracy::shift(node_rel + 2.0, size),
                        ),
                        accuracy::shift(1.0 - psum as f64, size),
                    );
                    real &= accuracy::exactly_real(&fv, fb);
                    seen |= !underflowed(&fv, fb);
                    l1 = accuracy::lsum(l1, accuracy::lsum(size, err));
                    rerr = accuracy::lsum(rerr, err);
                    sum = c_add(&sum, &value, psum, RM);
                }
            }
            let rounding =
                accuracy::shift(l1, ((2 * list.len()) as f64).log2() + 1.0 - psum as f64);
            let r = accuracy::lsum(rerr, rounding);
            if let Some(p) = &prev {
                let diff = accuracy::lg_abs(&c_sub(&sum, p, psum, RM));
                let rel_tol = accuracy::shift(l1, -rel_tol_bits).max(self.abs_floor);
                let tol = rel_tol.max(accuracy::shift(r, 1.0));
                debug!(m, diff, tol, r, "quadrature: Gauss–Legendre degree done");
                out.err = accuracy::lsum(diff, r);
                out.converged = m >= MIN_GL_DEGREE && diff <= tol;
                // The values' error, not the discretisation, limits it.
                out.discretisation = r <= rel_tol || diff > tol;
                out.digits = out.digits.max(l1 - diff);
                rel_hist.push(diff - l1);
            }
            out.value = sum.clone();
            out.l1 = l1;
            out.real = real;
            out.seen = seen;
            if out.converged {
                break;
            }
            // Fewer than 2 correct bits at 96 points: give it up.
            if m >= 6 && rel_hist.last().is_some_and(|&v| v > -2.0) {
                break;
            }
            prev = Some(sum);
        }
        Ok(out)
    }

    /// Integrate over `pieces`, bisecting a finite piece that does not
    /// converge.
    pub(super) fn run(&mut self, pieces: Vec<Piece>, cc: &mut Consts) -> Result<RunOut, QuadError> {
        let psum = self.pe + SUM_GUARD;
        let mut total = c_zero(psum);
        let mut err = accuracy::EXACT;
        let mut l1 = accuracy::EXACT;
        let mut real = true;
        let mut converged = true;
        let mut discretisation = true;
        let mut seen = false;
        let mut count = pieces.len();
        // Pieces with their bisection depth, and whether to try
        // Gauss–Legendre first (a parent on which tanh-sinh resolved nothing).
        let mut queue: Vec<(Piece, u32, bool)> =
            pieces.into_iter().rev().map(|p| (p, 0, false)).collect();
        let gl_only = GAUSS_LEGENDRE.with(std::cell::Cell::get);
        while let Some((piece, depth, gl_first)) = queue.pop() {
            let finite = match &piece {
                Piece::Finite(a, b) => Some((a.clone(), b.clone())),
                _ => None,
            };
            let mut best: Option<PieceOut> = None;
            let mut ts_digits = f64::INFINITY;
            let order: &[bool] = if gl_only {
                &[true]
            } else if gl_first {
                &[true, false]
            } else {
                &[false, true]
            };
            for &gl in order {
                let out = match (gl, gl_only) {
                    (true, true) => self.gl_piece(&piece, cc)?,
                    (true, false) => match self.gl_piece(&piece, cc) {
                        Ok(out) if finite.is_some() => out,
                        // A node Gauss–Legendre cannot use: tanh-sinh decides.
                        _ => continue,
                    },
                    (false, _) => self.piece(&piece, cc)?,
                };
                if !gl {
                    ts_digits = out.digits;
                }
                // A blind rule (no node had a value) converges on nothing
                // (its underflow bounds also pass for an accuracy limit):
                // the other rule, whose nodes lie elsewhere, may see the
                // integrand, and a value seen beats none.
                let done = (out.converged || !out.discretisation) && out.seen;
                if best.as_ref().is_none_or(|b| {
                    if out.seen == b.seen {
                        out.converged || out.err < b.err
                    } else {
                        out.seen
                    }
                }) {
                    if gl && out.converged {
                        debug!(depth, "quadrature: Gauss–Legendre converged");
                    }
                    best = Some(out);
                }
                if done {
                    break;
                }
            }
            let Some(out) = best else {
                return Err(internal("piece result").into());
            };
            let bisect = !out.converged
                && out.discretisation
                && depth < MAX_DEPTH
                && count < MAX_PIECES
                && self.work.left() > self.work.cap() / 8;
            if bisect && let Some(periods) = self.unresolvable(&piece, depth, cc) {
                debug!(
                    depth,
                    periods, "quadrature: a piece oscillates beyond resolution"
                );
                return Err(not_converged(&format!(
                    "the integrand oscillates about {periods:.3e} times over a piece of the range, \
                     more than the quadrature resolves"
                ))
                .into());
            }
            if bisect {
                let osc = ts_digits < 8.0;
                count += 1;
                // A finite piece is bisected; an infinite one loses a finite
                // piece next to its end (mpmath's `quadsubdiv`: `[a, ∞)` at
                // `max(a + 1, 2a)`), the tail coming after it.
                match &piece {
                    Piece::Finite(a, b) => {
                        let m = self.midpoint(a, b, cc)?;
                        debug!(depth, "quadrature: bisecting a piece that did not converge");
                        queue.push((Piece::Finite(m.clone(), b.clone()), depth + 1, osc));
                        queue.push((Piece::Finite(a.clone(), m), depth + 1, osc));
                    }
                    Piece::Upper(a) => {
                        let m = self.step_out(a, true, cc)?;
                        debug!(
                            depth,
                            "quadrature: splitting an infinite piece that did not converge"
                        );
                        queue.push((Piece::Upper(m.clone()), depth + 1, false));
                        queue.push((Piece::Finite(a.clone(), m), depth + 1, osc));
                    }
                    Piece::Lower(b) => {
                        let m = self.step_out(b, false, cc)?;
                        debug!(
                            depth,
                            "quadrature: splitting an infinite piece that did not converge"
                        );
                        queue.push((Piece::Lower(m.clone()), depth + 1, false));
                        queue.push((Piece::Finite(m, b.clone()), depth + 1, osc));
                    }
                }
                continue;
            }
            total = c_add(&total, &out.value, psum, RM);
            err = accuracy::lsum(err, out.err);
            l1 = accuracy::lsum(l1, out.l1);
            self.abs_floor = accuracy::shift(
                l1,
                -((self.prec - TARGET_GUARD.min(self.prec)) as f64) - FLOOR_SHARE,
            );
            real &= out.real;
            converged &= out.converged;
            discretisation &= out.converged || out.discretisation;
            seen |= out.seen;
        }
        if !seen {
            debug!("quadrature: every node underflowed");
            return Err(not_converged(BLIND).into());
        }
        debug!(
            pieces = count,
            evaluations = self.work.spent(),
            nodes = self.nodes.len(),
            "quadrature: done"
        );
        // The additions of the pieces.
        if count > 1 {
            err = accuracy::lsum(
                err,
                accuracy::shift(l1, (count as f64).log2() + 1.0 - psum as f64),
            );
        }
        Ok(RunOut {
            value: total,
            err,
            real,
            converged,
            discretisation,
        })
    }

    /// The number of times the integrand oscillates over `piece`, when no
    /// bisection from `depth` on could resolve it: more than
    /// [`MAX_PERIODS_PER_PIECE`] over each of the pieces of [`MAX_DEPTH`],
    /// or more than [`MAX_PIECES`] such pieces hold ([`Integrand::frequency`];
    /// `None` for an infinite piece or no oscillation known).  Up to 0.52
    /// such a piece was bisected until the evaluations ran out: `eval_f64` of
    /// the characteristic function of a Weibull distribution of shape `1/200`
    /// at `t = −5·10⁷` spent 11 s on `∫₀^∞ cos(5·10⁷x)·f(x) dx`.
    fn unresolvable(&mut self, piece: &Piece, depth: u32, cc: &mut Consts) -> Option<f64> {
        let Piece::Finite(a, b) = piece else {
            return None;
        };
        let frequency = *self.frequency.get_or_insert_with(|| self.f.frequency());
        if frequency <= 0.0 {
            return None;
        }
        let p = 64;
        let (av, _) = self.end_value(a, &mut Wp::new(p, cc)).ok()?;
        let (bv, _) = self.end_value(b, &mut Wp::new(p, cc)).ok()?;
        let width = super::f64_of(&bv.sub(&av, p, RM)).ok()?.abs();
        let periods = frequency * width / std::f64::consts::TAU;
        let finest = periods / f64::from(MAX_DEPTH.saturating_sub(depth)).exp2();
        let most = MAX_PERIODS_PER_PIECE * MAX_PIECES as f64;
        (finest > MAX_PERIODS_PER_PIECE || periods > most).then_some(periods)
    }

    /// The split point of an infinite piece from its finite end `e`:
    /// `max(e + 1, 2e)` upwards, `min(e − 1, 2e)` downwards, exact for a
    /// rational end.
    fn step_out(&mut self, e: &End, up: bool, cc: &mut Consts) -> Result<End, SymplexError> {
        use num_traits::Signed;
        if let End::Rational(q) = e {
            let one = Ratio::from_integer(BigInt::from(1));
            let step = if q.abs() > one { q.abs() } else { one };
            return Ok(End::Rational(if up { q + step } else { q - step }));
        }
        let p = self.pe + NODE_GUARD;
        let (v, _) = self.end_value(e, &mut Wp::new(p, cc))?;
        let one = small(1);
        let step = if v.abs().cmp(&one).is_some_and(|c| c > 0) {
            v.abs()
        } else {
            one
        };
        Ok(End::Float(if up {
            v.add(&step, p, RM)
        } else {
            v.sub(&step, p, RM)
        }))
    }

    /// A split point of `[a, b]`: the exact midpoint of rationals, otherwise
    /// a binary number near it.
    fn midpoint(&mut self, a: &End, b: &End, cc: &mut Consts) -> Result<End, SymplexError> {
        if let (End::Rational(x), End::Rational(y)) = (a, b) {
            return Ok(End::Rational((x + y) / BigInt::from(2)));
        }
        let p = self.pe + NODE_GUARD;
        let (av, _) = self.end_value(a, &mut Wp::new(p, cc))?;
        let (bv, _) = self.end_value(b, &mut Wp::new(p, cc))?;
        Ok(End::Float(scale2(&av.add(&bv, p, RM), -1)))
    }
}

/// Why a quadrature none of whose nodes had a value is refused.
const BLIND: &str = "the integrand underflowed the exponent range at every node: its mass, \
                     if any, lies between them";

/// Why a quadrature whose terms do not decay is refused.
const NOT_DECAYING: &str = "the integrand does not decay towards an end of the interval (the \
                            integral may diverge, or oscillate without decaying)";

fn not_converged(why: &str) -> SymplexError {
    SymplexError::ComputationFailed {
        operation: "evalf",
        reason: format!("quadrature of the definite integral did not converge: {why}"),
    }
}

/// Split the range from `lo` to `hi` (`None`: infinite in its direction)
/// at the sorted interior points `points` into pieces; `(−∞, ∞)` without a
/// point is split at 0.
pub(super) fn pieces(lo: Option<End>, hi: Option<End>, points: Vec<End>) -> Vec<Piece> {
    let mut cuts = points;
    if lo.is_none() && hi.is_none() && cuts.is_empty() {
        cuts.push(End::Rational(Ratio::from_integer(BigInt::from(0))));
    }
    let mut out = Vec::with_capacity(cuts.len() + 1);
    let mut left = lo;
    for c in cuts {
        out.push(match left {
            Some(a) => Piece::Finite(a, c.clone()),
            None => Piece::Lower(c.clone()),
        });
        left = Some(c);
    }
    out.push(match (left, hi) {
        (Some(a), Some(b)) => Piece::Finite(a, b),
        (Some(a), None) => Piece::Upper(a),
        (None, Some(b)) => Piece::Lower(b),
        (None, None) => Piece::Upper(End::Rational(Ratio::from_integer(BigInt::from(0)))),
    });
    out
}

/// Are the finite ends `a` and `b` provably the same point?
pub(super) fn same_end(a: &End, b: &End) -> bool {
    a.same(b)
}

thread_local! {
    /// Gauss–Legendre on every piece instead of tanh-sinh (see [`RuleGuard`]).
    static GAUSS_LEGENDRE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The rule of the quadratures on this thread while it lives:
/// Gauss–Legendre when set so (`Ex::integrate_numeric_decimal` with
/// `QuadRule::GaussLegendre`), restored on drop.
pub(super) struct RuleGuard {
    previous: bool,
}

impl RuleGuard {
    pub(super) fn set(gauss_legendre: bool) -> RuleGuard {
        RuleGuard {
            previous: GAUSS_LEGENDRE.with(|c| c.replace(gauss_legendre)),
        }
    }
}

impl Drop for RuleGuard {
    fn drop(&mut self) {
        GAUSS_LEGENDRE.with(|c| c.set(self.previous));
    }
}

/// The Gauss–Legendre node sets computed on this thread, by degree and
/// precision.
type GlNodes = std::rc::Rc<(Vec<(BigFloat, BigFloat)>, ErrExp)>;

thread_local! {
    static GL_NODES: std::cell::RefCell<Vec<((u32, usize), GlNodes)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `(P_n(r), P_{n−1}(r))` by the three-term recurrence, in `f64`.
fn legendre_f64(n: usize, r: f64) -> (f64, f64) {
    let (mut t1, mut t2) = (1.0, 0.0);
    for j in 1..=n {
        let jf = j as f64;
        let t3 = ((2.0 * jf - 1.0) * r * t1 - (jf - 1.0) * t2) / jf;
        t2 = t1;
        t1 = t3;
    }
    (t1, t2)
}

/// The positive roots of the Legendre polynomial `P_n`, `n = 3·2^(m−1)`
/// (`m ≥ 2`), with the Gauss–Legendre weights `2/((1 − r²)·P_n′(r)²)`, at
/// `pw` bits, and a bound on their relative error (`log₂`): as mpmath's
/// `GaussLegendre.calc_nodes`, from the asymptotic first guess
/// `cos(π(j − ¼)/(n + ½))` by Newton's iteration on the recurrence — in
/// `f64` first, then doubling the correct bits at `pw` bits.
fn gauss_legendre(m: u32, pw: usize) -> (Vec<(BigFloat, BigFloat)>, ErrExp) {
    let n: usize = 3 << m.saturating_sub(1);
    let nb = small(n as u64);
    let one = small(1);
    let mut out = Vec::with_capacity(n / 2);
    for j in 1..=n / 2 {
        let mut r = (std::f64::consts::PI * (j as f64 - 0.25) / (n as f64 + 0.5)).cos();
        for _ in 0..10 {
            let (t1, t2) = legendre_f64(n, r);
            let dp = n as f64 * (r * t1 - t2) / (r * r - 1.0);
            let a = t1 / dp;
            r -= a;
            if a.abs() < 1e-15 {
                break;
            }
        }
        let mut rb = bigfloat::from_f64(r, pw);
        let mut bits = 40usize;
        let t4 = loop {
            let (mut t1, mut t2) = (one.clone(), BigFloat::new(pw));
            for jj in 1..=n as u64 {
                let a = rb.mul(&t1, pw, RM).mul(&small(2 * jj - 1), pw, RM);
                let b = t2.mul(&small(jj - 1), pw, RM);
                let t3 = a.sub(&b, pw, RM).div(&small(jj), pw, RM);
                t2 = t1;
                t1 = t3;
            }
            let den = rb.mul(&rb, pw, RM).sub(&one, pw, RM);
            let t4 = nb
                .mul(&rb.mul(&t1, pw, RM).sub(&t2, pw, RM), pw, RM)
                .div(&den, pw, RM);
            rb = rb.sub(&t1.div(&t4, pw, RM), pw, RM);
            bits *= 2;
            if bits >= pw + 16 {
                break t4;
            }
        };
        let om = one.sub(&rb.mul(&rb, pw, RM), pw, RM);
        let w = scale2(&one, 1).div(&om.mul(&t4.mul(&t4, pw, RM), pw, RM), pw, RM);
        out.push((rb, w));
    }
    (out, (n as f64).log2() + 6.0 - pw as f64)
}

// ═══════════════════════════════════════════════════════════════════════════
// Failures remembered across precisions
// ═══════════════════════════════════════════════════════════════════════════

/// An integral by its arena ([`Arena::serial`]) and node ids.  Up to 0.48
/// the arena was named by its address and node count, which a later arena
/// on the same thread can have again (the golden harness's fresh context
/// per case did): it would have read the earlier arena's failure.
pub(super) type MemoKey = [u64; 5];

/// A quadrature that did not converge for want of nodes, not of precision
/// (see [`remember`]).
struct Failure {
    key: MemoKey,
    prec: usize,
    outcome: Result<(Complex, ErrExp, bool), QuadError>,
}

thread_local! {
    /// The latest quadratures that did not converge for want of nodes.
    static FAILURES: std::cell::RefCell<Vec<Failure>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Failures kept.
const MEMO_CAP: usize = 16;

/// The key of `∫_lo^hi body d(var)` in `arena` (nodes are only appended,
/// so the ids keep naming the same nodes).
pub(super) fn memo_key(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
) -> MemoKey {
    [
        arena.serial(),
        u64::from(body.index()),
        u64::from(var.index()),
        u64::from(lo.index()),
        u64::from(hi.index()),
    ]
}

/// The outcome of a quadrature of `key` at a precision up to `prec` that
/// did not converge for want of nodes: at `prec` it would not either (a
/// higher precision only tightens the tolerance), so `evalf`'s search for
/// more digits gets the same answer at once instead of the same work again.
pub(super) fn recall(
    key: &MemoKey,
    prec: usize,
) -> Option<Result<(Complex, ErrExp, bool), QuadError>> {
    FAILURES.with(|f| {
        f.borrow()
            .iter()
            .find(|e| e.key == *key && e.prec <= prec)
            .map(|e| e.outcome.clone())
    })
}

/// Remember a quadrature of `key` at `prec` that did not converge for want
/// of nodes.
pub(super) fn remember(
    key: MemoKey,
    prec: usize,
    outcome: Result<(Complex, ErrExp, bool), QuadError>,
) {
    FAILURES.with(|f| {
        let mut f = f.borrow_mut();
        f.retain(|e| e.key != key);
        if f.len() >= MEMO_CAP {
            f.remove(0);
        }
        f.push(Failure { key, prec, outcome });
    });
}

/// The closed form of an integral whose quadrature refused (`evalf`'s exact
/// route): the private arena and the value, the proof that it diverges, or
/// `Ok(None)` when there is none.
pub(super) type ClosedForm = std::rc::Rc<Result<Option<(Arena, ExprId)>, SymplexError>>;

thread_local! {
    /// The latest closed forms sought, by integral.
    static CLOSED_FORMS: std::cell::RefCell<Vec<(MemoKey, ClosedForm)>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// Is a closed form being sought on this thread?
    static SEEKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Clears [`SEEKING`] when the search ends, also by unwinding.
struct Seeking;

impl Drop for Seeking {
    fn drop(&mut self) {
        SEEKING.with(|s| s.set(false));
    }
}

/// The closed form of the integral `key` by `seek` (the definite
/// integrator), sought once per integral: `evalf`'s search for more digits
/// asks again at every precision, and the answer is the same.  `None`
/// while another closed form is being sought on this thread — the
/// integrator evaluating a bound or a breakpoint that holds a definite
/// integral does not start a search of its own.
pub(super) fn closed_form(
    key: MemoKey,
    seek: impl FnOnce() -> Result<Option<(Arena, ExprId)>, SymplexError>,
) -> Option<ClosedForm> {
    let known = recall_closed_form(&key);
    if known.is_some() {
        return known;
    }
    if SEEKING.with(std::cell::Cell::get) {
        return None;
    }
    SEEKING.with(|s| s.set(true));
    let found = {
        let _seeking = Seeking;
        std::rc::Rc::new(seek())
    };
    CLOSED_FORMS.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() >= CLOSED_FORM_CAP {
            m.remove(0);
        }
        m.push((key, found.clone()));
    });
    Some(found)
}

/// The closed form of the integral `key` when it has been sought
/// ([`closed_form`]).
pub(super) fn recall_closed_form(key: &MemoKey) -> Option<ClosedForm> {
    CLOSED_FORMS.with(|m| {
        m.borrow()
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, c)| c.clone())
    })
}

/// Closed forms kept.
const CLOSED_FORM_CAP: usize = 8;

/// Why a quadrature gave no value.  The two causes `evalf` acts on — terms
/// that do not decay, and a failure for want of nodes — are told apart by
/// kind; up to 0.48 they were found by searching the message for
/// [`NOT_DECAYING`], `"budget"` or `"time limit"`, so an integrand whose
/// own error message mentioned a budget counted as one.
#[derive(Clone, Debug)]
pub(super) enum QuadError {
    /// The terms do not decay towards an end of the range, or of a piece
    /// at a breakpoint ([`NOT_DECAYING`]).
    NotDecaying,
    /// Any other failure (the evaluation budget is
    /// [`SymplexError::BudgetExhausted`]).
    Failed(SymplexError),
}

impl From<SymplexError> for QuadError {
    fn from(e: SymplexError) -> Self {
        QuadError::Failed(e)
    }
}

impl From<QuadError> for SymplexError {
    fn from(e: QuadError) -> Self {
        match e {
            QuadError::NotDecaying => not_converged(NOT_DECAYING),
            QuadError::Failed(e) => e,
        }
    }
}

impl QuadError {
    /// A failure for want of nodes (a higher precision fails too), or a
    /// proof of divergence?
    pub(super) fn is_node_failure(&self) -> bool {
        matches!(
            self,
            QuadError::NotDecaying
                | QuadError::Failed(
                    SymplexError::Divergent { .. }
                        | SymplexError::BudgetExhausted {
                            kind: BudgetKind::QuadratureEvaluations,
                            ..
                        }
                )
        )
    }

    /// A refusal for want of a value that a proof of divergence may
    /// replace (any but an evaluation error of the integrand)?
    pub(super) fn may_diverge(&self) -> bool {
        matches!(
            self,
            QuadError::NotDecaying
                | QuadError::Failed(
                    SymplexError::ComputationFailed { .. } | SymplexError::BudgetExhausted { .. }
                )
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_f64(x: &BigFloat) -> f64 {
        super::super::f64_of(x).unwrap_or(f64::NAN)
    }

    /// The 12-point Gauss–Legendre rule (`m = 3`) integrates `x^22` exactly:
    /// `Σ w·(r²²·2) = 2/23`, and its weights sum to 2.
    #[test]
    fn gauss_legendre_nodes_are_exact_to_degree_2n_minus_1() {
        let p = 200;
        let (nodes, rel) = gauss_legendre(3, p);
        assert_eq!(nodes.len(), 6);
        assert!(rel < -180.0);
        let mut sum_w = BigFloat::new(p);
        let mut moment = BigFloat::new(p);
        for (r, w) in &nodes {
            sum_w = sum_w.add(&scale2(w, 1), p, RM);
            moment = moment.add(&scale2(&w.mul(&r.powi(22, p, RM), p, RM), 1), p, RM);
        }
        let two = small(2);
        let exact = two.div(&small(23), p, RM);
        assert!(lg(&sum_w.sub(&two, p, RM)) < -180.0, "{}", to_f64(&sum_w));
        assert!(
            lg(&moment.sub(&exact, p, RM)) < -180.0,
            "{}",
            to_f64(&moment)
        );
    }

    /// Abscissae and steps are exact binary numbers.
    #[test]
    fn dyadic_steps_are_exact() {
        assert_eq!(to_f64(&dyadic(3, 2)), 0.75);
        assert_eq!(to_f64(&scale2(&small(5), -3)), 0.625);
        assert_eq!(to_f64(&scale2(&BigFloat::new(64), 7)), 0.0);
        assert_eq!(quantize(129), 160);
    }

    /// `pieces` splits at the interior points and the whole line at 0.
    #[test]
    fn pieces_split_the_range() {
        let zero = || End::Rational(Ratio::from_integer(BigInt::from(0)));
        let p = pieces(None, None, Vec::new());
        assert!(matches!(
            &p[..],
            [
                Piece::Lower(End::Rational(_)),
                Piece::Upper(End::Rational(_))
            ]
        ));
        let p = pieces(
            Some(zero()),
            None,
            vec![End::Rational(Ratio::from_integer(BigInt::from(1)))],
        );
        assert!(matches!(&p[..], [Piece::Finite(..), Piece::Upper(_)]));
        assert!(same_end(&zero(), &zero()));
    }
}
