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
//! (with an error estimate of `10⁻⁵³` for the first).
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
//! points, kinks of `|·|`, steps, `Piecewise` boundaries).  A finite piece
//! on which tanh-sinh resolves nothing by level 6 (an oscillation its nodes,
//! crowded at the ends, do not resolve) is tried by Gauss–Legendre of
//! increasing degree, accepted on the same criterion; a piece that converges
//! by neither within its share of the work is bisected, as mpmath's
//! `quadsubdiv` does, and an infinite one loses a finite piece next to its
//! end (`[a, ∞)` at `max(a + 1, 2a)`) — the algebraic map converges slowly on
//! an exponentially decaying integrand, and the tail left is small.  A piece
//! integrated after others needs only a share of their tolerance in absolute
//! terms.
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
//! successive levels happens to agree, can pass it.  A piece that does not
//! meet it within the work limit reports the difference it reached — `evalf`
//! certifies the digits that covers — and an integral whose terms do not
//! decay (a divergent or oscillatory infinite integral) is refused.
//!
//! # Cost
//!
//! A node is a handful of `BigFloat` operations (computed once per
//! quadrature and shared by all pieces) plus one evaluation of the
//! variable-dependent part of the body; level `k` has about `9·2^(k−1)`
//! nodes, at most [`MAX_EVALS`] evaluations in all and [`TIME_LIMIT`] of
//! wall-clock time per working precision.  A quadrature that ran out of
//! nodes is remembered ([`recall`]): `evalf`'s search at higher precisions
//! gets its answer again at once.  Gauss–Legendre nodes (`O(n²)` operations
//! each set) are cached per thread.

use astro_float::{BigFloat, Consts, RoundingMode};
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
use crate::base::errors::SymplexError;
use crate::base::node::ExprId;
use crate::base::walk;

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
/// 10⁸` on an infinite range), however small its terms.
const T_MIN_HALVES: u64 = 5;
/// Nor does it go beyond `t = 12` (`δ ≈ 10⁻¹¹⁰⁰⁰⁰`): terms that have not
/// decayed by then belong to a divergent integral.
const T_MAX_HALVES: u64 = 24;
/// Most integrand evaluations for one integral (all pieces, refinements
/// included).
const MAX_EVALS: usize = 30_000;
/// Most pieces (breakpoints and bisections).
const MAX_PIECES: usize = 64;
/// Deepest bisection of a piece.
const MAX_DEPTH: u32 = 8;
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
/// Wall-clock limit of one quadrature (one working precision): the
/// evaluation budget bounds the nodes, but an integrand of special functions
/// at a thousand bits can take milliseconds a node.
const TIME_LIMIT: std::time::Duration = std::time::Duration::from_secs(20);

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
        prec: usize,
        rm: RoundingMode,
        cc: &mut Consts,
    ) -> Result<(Complex, Bound), SymplexError> {
        if !self.scratch.contains_key(&prec) {
            if self.scratch.len() >= MAX_PRECISIONS {
                self.scratch.clear();
            }
            let mut cache: FxHashMap<ExprId, Complex> = FxHashMap::default();
            let mut errs = ErrMap::default();
            for (c, order) in &self.frontier {
                if let Ok((v, b)) = evaluate_tree(self.arena, *c, order, None, false, prec, rm, cc)
                {
                    cache.insert(*c, v);
                    errs.insert(*c, b);
                }
            }
            self.scratch.insert(prec, (cache, errs));
        }
        let Some((cache, errs)) = self.scratch.get_mut(&prec) else {
            return Err(internal("integrand cache"));
        };
        cache.insert(self.var, (x.clone(), BigFloat::new(prec)));
        errs.insert(self.var, Bound::real(x_err));
        let mut failed: FxHashMap<ExprId, SymplexError> = FxHashMap::default();
        for &id in &self.dependent {
            let evaluated = eval_node_with_error(self.arena, id, cache, errs, prec, rm, cc)
                .and_then(
                    |(v, b)| match exponent_range_error(self.arena, id, &v, cache, errs) {
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
        Ok((value, bound))
    }
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

/// The weighted value of a node: `W·f(x)`, a bound on its error, and
/// whether `f(x)` was exactly real.
struct Term {
    value: Complex,
    err: ErrExp,
    real: bool,
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
    rm: RoundingMode,
    evals_left: usize,
    /// The evaluation budget `evals_left` started from.
    evals_total: usize,
    /// The integral is refused past this instant ([`TIME_LIMIT`]).
    deadline: std::time::Instant,
    ends: FxHashMap<(ExprId, usize), (BigFloat, ErrExp)>,
    /// The tanh-sinh abscissae computed so far, by level and index (shared
    /// by both sides and all pieces).
    nodes: FxHashMap<(u32, u64), std::rc::Rc<TsNode>>,
    /// An absolute tolerance every piece may stop at (`log₂`): a share of
    /// the tolerance of the pieces done so far, so that a piece far smaller
    /// than the integral (the tail of a split infinite range) is not held to
    /// its own relative accuracy.
    abs_floor: ErrExp,
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
        rm: RoundingMode,
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
            rm,
            evals_left: MAX_EVALS,
            evals_total: MAX_EVALS,
            deadline: std::time::Instant::now() + TIME_LIMIT,
            ends: FxHashMap::default(),
            nodes: FxHashMap::default(),
            abs_floor: accuracy::EXACT,
        }
    }

    /// A quarter of the evaluations and an eighth of the time: a first try
    /// that has a fallback.
    pub(super) fn lighten(&mut self) {
        self.evals_left = MAX_EVALS / 4;
        self.evals_total = MAX_EVALS / 4;
        self.deadline = std::time::Instant::now() + TIME_LIMIT / 8;
    }

    /// The value of a finite endpoint at `p` bits and its error bound.
    pub(super) fn end_value(
        &mut self,
        e: &End,
        p: usize,
        cc: &mut Consts,
    ) -> Result<(BigFloat, ErrExp), SymplexError> {
        match e {
            End::Rational(q) => {
                let v = crate::base::numeric::ratio_to_bigfloat(q, p, self.rm);
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
                let (v, b) = evaluate_tree(self.f.arena, *id, &order, None, true, p, self.rm, cc)?;
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
    fn ts_node(&mut self, j: u64, k: u32, pw: usize, cc: &mut Consts) -> std::rc::Rc<TsNode> {
        if let Some(n) = self.nodes.get(&(k, j)) {
            return n.clone();
        }
        let n = std::rc::Rc::new(self.compute_ts_node(j, k, pw, cc));
        self.nodes.insert((k, j), n.clone());
        n
    }

    /// [`Quad::ts_node`], computed.
    fn compute_ts_node(&self, j: u64, k: u32, pw: usize, cc: &mut Consts) -> TsNode {
        let rm = self.rm;
        let t = dyadic(j, k);
        let pi = cc.pi(pw, rm);
        let s = t.sinh(pw, rm, cc);
        let c = t.cosh(pw, rm, cc);
        let arg = pi.mul(&s, pw, rm); // π·sinh t = 2u
        let q = arg.neg().exp(pw, rm, cc); // e^{−2u}
        let opq = small(1).add(&q, pw, rm);
        let delta = scale2(&q, 1).div(&opq, pw, rm); // 2q/(1 + q) = 1 − tanh u
        let w = scale2(&pi.mul(&c, pw, rm).mul(&q, pw, rm), 1).div(&opq.mul(&opq, pw, rm), pw, rm);
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
        p: usize,
        cc: &mut Consts,
    ) -> Result<Placed, SymplexError> {
        let rm = self.rm;
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
            let pi_half = scale2(&cc.pi(pw, rm), -1);
            return Ok(match piece {
                Piece::Finite(a, _) => {
                    let (av, ae) = self.end_value(a, p, cc)?;
                    let x = av.add(c, p, rm);
                    Placed {
                        x_err: accuracy::lsum(accuracy::lsum(ae, *c_err), ulp2(&x, p)),
                        x,
                        w: scale2(&c.mul(&pi_half, pw, rm), step),
                        w_rel: accuracy::lsum(c_rel, 2.0 - pw as f64),
                        need: p,
                        which: None,
                    }
                }
                Piece::Upper(a) => {
                    let (av, ae) = self.end_value(a, p, cc)?;
                    let x = av.add(&one, p, rm);
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
                    let (bv, be) = self.end_value(b, p, cc)?;
                    let x = bv.sub(&one, p, rm);
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
                let dist = c.mul(d, pw, rm);
                if let Some(i) = self.variant(a, side == Side::Lo) {
                    return Ok(Placed {
                        x_err: ulp2(&dist, pw),
                        x: dist,
                        w: scale2(&c.mul(&node.w, pw, rm), step),
                        w_rel: accuracy::lsum(c_rel, base_rel),
                        need: p,
                        which: Some(i),
                    });
                }
                let (av, ae) = self.end_value(a, p, cc)?;
                let x = if side == Side::Lo {
                    av.add(&dist, p, rm)
                } else {
                    av.sub(&dist, p, rm)
                };
                Placed {
                    x_err: accuracy::lsum(ae, ulp2(&x, p)),
                    need: need_for(&av, &dist, pe),
                    x,
                    w: scale2(&c.mul(&node.w, pw, rm), step),
                    w_rel: accuracy::lsum(c_rel, base_rel),
                    which: None,
                }
            }
            // The finite end of a half-infinite piece: x = a ± δ/(2 − δ).
            (Piece::Upper(a), Side::Lo) | (Piece::Lower(a), Side::Hi) => {
                let tmd = two.sub(d, pw, rm);
                let dist = d.div(&tmd, pw, rm);
                let factor = scale2(&one.div(&tmd.mul(&tmd, pw, rm), pw, rm), 1);
                if let Some(i) = self.variant(a, side == Side::Lo) {
                    return Ok(Placed {
                        x_err: ulp2(&dist, pw),
                        x: dist,
                        w: scale2(&node.w.mul(&factor, pw, rm), step),
                        w_rel: base_rel + 1.0,
                        need: p,
                        which: Some(i),
                    });
                }
                let (av, ae) = self.end_value(a, p, cc)?;
                let x = if side == Side::Lo {
                    av.add(&dist, p, rm)
                } else {
                    av.sub(&dist, p, rm)
                };
                Placed {
                    x_err: accuracy::lsum(ae, ulp2(&x, p)),
                    need: need_for(&av, &dist, pe),
                    x,
                    w: scale2(&node.w.mul(&factor, pw, rm), step),
                    w_rel: base_rel + 1.0,
                    which: None,
                }
            }
            // The infinite end: x = a − 1 + 2/δ, x = b + 1 − 2/δ.
            (Piece::Upper(a), Side::Hi) | (Piece::Lower(a), Side::Lo) => {
                let far = scale2(&one.div(d, pw, rm), 1);
                let (av, ae) = self.end_value(a, p, cc)?;
                let x = if side == Side::Hi {
                    av.sub(&one, p, rm).add(&far, p, rm)
                } else {
                    av.add(&one, p, rm).sub(&far, p, rm)
                };
                let factor = scale2(&one.div(&d.mul(d, pw, rm), pw, rm), 1);
                Placed {
                    x_err: accuracy::lsum(ae, ulp2(&x, p)),
                    need: p,
                    x,
                    w: scale2(&node.w.mul(&factor, pw, rm), step),
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
        let rm = self.rm;
        let psum = self.pe + SUM_GUARD;
        let mut p = self.pe;
        loop {
            if self.evals_left == 0 {
                return Err(not_converged("the evaluation budget was exhausted"));
            }
            if std::time::Instant::now() > self.deadline {
                return Err(not_converged("the time limit was reached"));
            }
            self.evals_left -= 1;
            let placed = self.place(piece, half, side, node, k, p, cc)?;
            let f = match placed.which.and_then(|i| self.variants.get_mut(i)) {
                Some((_, _, g)) => g,
                None => &mut self.f,
            };
            let evaluated = f.eval(&placed.x, placed.x_err, p, rm, cc);
            let next = match evaluated {
                Ok((fv, fb))
                    if !fb.is_unknown()
                        && !(fv.0.is_nan() || fv.0.is_inf() || fv.1.is_nan() || fv.1.is_inf()) =>
                {
                    let value = (fv.0.mul(&placed.w, psum, rm), fv.1.mul(&placed.w, psum, rm));
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
                        return Ok(Term { value, err, real });
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
                        super::bigfloat_to_f64_rounded(&placed.x, rm)
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
    fn piece(&mut self, piece: &Piece, cc: &mut Consts) -> Result<PieceOut, SymplexError> {
        let rm = self.rm;
        let psum = self.pe + SUM_GUARD;
        let pw = self.pe + NODE_GUARD;
        let half = match piece {
            Piece::Finite(a, b) => {
                let (av, ae) = self.end_value(a, pw, cc)?;
                let (bv, be) = self.end_value(b, pw, cc)?;
                let c = scale2(&bv.sub(&av, pw, rm), -1);
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
        l1 = accuracy::lsum(
            l1,
            accuracy::lsum(accuracy::lg_abs(&centre.value), centre.err),
        );
        rerr = accuracy::lsum(rerr, centre.err);
        sum = c_add(&sum, &centre.value, psum, rm);
        additions += 1;
        // The walk outwards on each side: the level-1 terms' sizes, until
        // two in a row are negligible (and not before `t = 3`).
        let mut sizes: [Vec<ErrExp>; 2] = [Vec::new(), Vec::new()];
        for (si, side) in [Side::Lo, Side::Hi].into_iter().enumerate() {
            let mut quiet = 0;
            let mut j = 1u64;
            loop {
                if j > T_MAX_HALVES {
                    return Err(not_converged(NOT_DECAYING));
                }
                let node = self.ts_node(j, 1, pw, cc);
                let budget = accuracy::shift(l1, -((self.pe as f64) + 8.0));
                let t = self.term(piece, &half, Some(side), Some(node.as_ref()), 1, budget, cc)?;
                real &= t.real;
                let size = accuracy::lsum(accuracy::lg_abs(&t.value), t.err);
                l1 = accuracy::lsum(l1, size);
                rerr = accuracy::lsum(rerr, t.err);
                sum = c_add(&sum, &t.value, psum, rm);
                additions += 1;
                sizes[si].push(size);
                let threshold = accuracy::shift(l1, -((self.pe as f64) + 8.0));
                if size <= threshold || accuracy::is_exact(size) {
                    quiet += 1;
                } else {
                    quiet = 0;
                }
                if quiet >= 2 && j >= T_MIN_HALVES {
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
            if new_nodes as usize > self.evals_left {
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
                    let node = self.ts_node(j, k, pw, cc);
                    let t =
                        self.term(piece, &half, Some(side), Some(node.as_ref()), k, budget, cc)?;
                    real &= t.real;
                    level_l1 =
                        accuracy::lsum(level_l1, accuracy::lsum(accuracy::lg_abs(&t.value), t.err));
                    level_err = accuracy::lsum(level_err, t.err);
                    level = c_add(&level, &t.value, psum, rm);
                    additions += 1;
                    j += 2;
                }
            }
            let halved = (scale2(&prev.0, -1), scale2(&prev.1, -1));
            sum = c_add(&halved, &level, psum, rm);
            additions += 1;
            l1 = accuracy::lsum(accuracy::shift(l1, -1.0), level_l1);
            rerr = accuracy::lsum(accuracy::shift(rerr, -1.0), level_err);
            let diff = accuracy::lg_abs(&c_sub(&sum, &prev, psum, rm));
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
        if next_cost as usize > self.evals_left {
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
        let nodes = std::rc::Rc::new(gauss_legendre(m, pw, self.rm));
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
        let rm = self.rm;
        let pe = self.pe;
        let psum = pe + SUM_GUARD;
        let pw = pe + NODE_GUARD;
        let one = small(1);
        // `x = centre + C·s` on a finite piece; on a half-infinite one
        // `x = e ∓ 1 ± 2/(s + 1)` with weight factor `2/(s + 1)²` (mpmath's
        // `transform_nodes`), `centre` holding `e ∓ 1`.
        let (centre, c, end_err, upper) = match piece {
            Piece::Finite(a, b) => {
                let (av, ae) = self.end_value(a, pw, cc)?;
                let (bv, be) = self.end_value(b, pw, cc)?;
                let c = scale2(&bv.sub(&av, pw, rm), -1);
                let mid = scale2(&av.add(&bv, pw, rm), -1);
                let err = accuracy::lsum(accuracy::lsum(ae, be), ulp2(&mid, pw));
                (mid, c, err, None)
            }
            Piece::Upper(a) => {
                let (av, ae) = self.end_value(a, pw, cc)?;
                (av.sub(&one, pw, rm), one.clone(), ae, Some(true))
            }
            Piece::Lower(b) => {
                let (bv, be) = self.end_value(b, pw, cc)?;
                (bv.add(&one, pw, rm), one.clone(), be, Some(false))
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
        };
        let mut rel_hist: Vec<ErrExp> = Vec::new();
        for m in 2..=MAX_GL_DEGREE {
            let nodes = self.gl_nodes(m, pw);
            let (ref list, node_rel) = *nodes;
            if 2 * list.len() > self.evals_left {
                break;
            }
            let mut sum = c_zero(psum);
            let mut l1 = accuracy::EXACT;
            let mut rerr = accuracy::EXACT;
            let mut real = true;
            for (r, w) in list.iter() {
                for sign in [false, true] {
                    if self.evals_left == 0 {
                        return Err(not_converged("the evaluation budget was exhausted"));
                    }
                    self.evals_left -= 1;
                    let s = if sign { r.neg() } else { r.clone() };
                    let (x, weight) = match upper {
                        None => (centre.add(&c.mul(&s, pw, rm), pe, rm), c.mul(w, pw, rm)),
                        Some(up) => {
                            let sp1 = s.add(&one, pw, rm);
                            let u = scale2(&one.div(&sp1, pw, rm), 1);
                            let x = if up {
                                centre.add(&u, pe, rm)
                            } else {
                                centre.sub(&u, pe, rm)
                            };
                            (x, scale2(&w.div(&sp1.mul(&sp1, pw, rm), pw, rm), 1))
                        }
                    };
                    let x_err = accuracy::lsum(end_err, ulp2(&x, pe));
                    let (fv, fb) = self.f.eval(&x, x_err, pe, rm, cc)?;
                    if fb.is_unknown()
                        || fv.0.is_nan()
                        || fv.0.is_inf()
                        || fv.1.is_nan()
                        || fv.1.is_inf()
                    {
                        return Err(not_converged("the integrand has no error bound at a node"));
                    }
                    let value = (fv.0.mul(&weight, psum, rm), fv.1.mul(&weight, psum, rm));
                    let size = accuracy::lg_abs(&value);
                    let err = accuracy::lsum(
                        accuracy::lsum(
                            accuracy::shift(fb.joint(), lg(&weight)),
                            accuracy::shift(node_rel + 2.0, size),
                        ),
                        accuracy::shift(1.0 - psum as f64, size),
                    );
                    real &= accuracy::exactly_real(&fv, fb);
                    l1 = accuracy::lsum(l1, accuracy::lsum(size, err));
                    rerr = accuracy::lsum(rerr, err);
                    sum = c_add(&sum, &value, psum, rm);
                }
            }
            let rounding =
                accuracy::shift(l1, ((2 * list.len()) as f64).log2() + 1.0 - psum as f64);
            let r = accuracy::lsum(rerr, rounding);
            if let Some(p) = &prev {
                let diff = accuracy::lg_abs(&c_sub(&sum, p, psum, rm));
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
    pub(super) fn run(
        &mut self,
        pieces: Vec<Piece>,
        cc: &mut Consts,
    ) -> Result<RunOut, SymplexError> {
        let rm = self.rm;
        let psum = self.pe + SUM_GUARD;
        let mut total = c_zero(psum);
        let mut err = accuracy::EXACT;
        let mut l1 = accuracy::EXACT;
        let mut real = true;
        let mut converged = true;
        let mut discretisation = true;
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
                let done = out.converged || !out.discretisation;
                if best
                    .as_ref()
                    .is_none_or(|b| out.converged || out.err < b.err)
                {
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
                return Err(internal("piece result"));
            };
            let bisect = !out.converged
                && out.discretisation
                && depth < MAX_DEPTH
                && count < MAX_PIECES
                && self.evals_left > self.evals_total / 8;
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
            total = c_add(&total, &out.value, psum, rm);
            err = accuracy::lsum(err, out.err);
            l1 = accuracy::lsum(l1, out.l1);
            self.abs_floor = accuracy::shift(
                l1,
                -((self.prec - TARGET_GUARD.min(self.prec)) as f64) - FLOOR_SHARE,
            );
            real &= out.real;
            converged &= out.converged;
            discretisation &= out.converged || out.discretisation;
        }
        debug!(
            pieces = count,
            evaluations = self.evals_total.saturating_sub(self.evals_left),
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
        let (v, _) = self.end_value(e, p, cc)?;
        let one = small(1);
        let step = if v.abs().cmp(&one).is_some_and(|c| c > 0) {
            v.abs()
        } else {
            one
        };
        Ok(End::Float(if up {
            v.add(&step, p, self.rm)
        } else {
            v.sub(&step, p, self.rm)
        }))
    }

    /// A split point of `[a, b]`: the exact midpoint of rationals, otherwise
    /// a binary number near it.
    fn midpoint(&mut self, a: &End, b: &End, cc: &mut Consts) -> Result<End, SymplexError> {
        if let (End::Rational(x), End::Rational(y)) = (a, b) {
            return Ok(End::Rational((x + y) / BigInt::from(2)));
        }
        let p = self.pe + NODE_GUARD;
        let (av, _) = self.end_value(a, p, cc)?;
        let (bv, _) = self.end_value(b, p, cc)?;
        Ok(End::Float(scale2(&av.add(&bv, p, self.rm), -1)))
    }
}

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
fn gauss_legendre(m: u32, pw: usize, rm: RoundingMode) -> (Vec<(BigFloat, BigFloat)>, ErrExp) {
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
        let mut rb = BigFloat::from_f64(r, pw);
        let mut bits = 40usize;
        let t4 = loop {
            let (mut t1, mut t2) = (one.clone(), BigFloat::new(pw));
            for jj in 1..=n as u64 {
                let a = rb.mul(&t1, pw, rm).mul(&small(2 * jj - 1), pw, rm);
                let b = t2.mul(&small(jj - 1), pw, rm);
                let t3 = a.sub(&b, pw, rm).div(&small(jj), pw, rm);
                t2 = t1;
                t1 = t3;
            }
            let den = rb.mul(&rb, pw, rm).sub(&one, pw, rm);
            let t4 = nb
                .mul(&rb.mul(&t1, pw, rm).sub(&t2, pw, rm), pw, rm)
                .div(&den, pw, rm);
            rb = rb.sub(&t1.div(&t4, pw, rm), pw, rm);
            bits *= 2;
            if bits >= pw + 16 {
                break t4;
            }
        };
        let om = one.sub(&rb.mul(&rb, pw, rm), pw, rm);
        let w = scale2(&one, 1).div(&om.mul(&t4.mul(&t4, pw, rm), pw, rm), pw, rm);
        out.push((rb, w));
    }
    (out, (n as f64).log2() + 6.0 - pw as f64)
}

// ═══════════════════════════════════════════════════════════════════════════
// Failures remembered across precisions
// ═══════════════════════════════════════════════════════════════════════════

/// An integral by its arena (address and size) and node ids.
pub(super) type MemoKey = [usize; 6];

/// A quadrature that did not converge for want of nodes, not of precision
/// (see [`remember`]).
struct Failure {
    key: MemoKey,
    prec: usize,
    outcome: Result<(Complex, ErrExp, bool), SymplexError>,
}

thread_local! {
    /// The latest quadratures that did not converge for want of nodes.
    static FAILURES: std::cell::RefCell<Vec<Failure>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Failures kept.
const MEMO_CAP: usize = 16;

/// The key of `∫_lo^hi body d(var)` in `arena` (the arena does not change
/// while `evalf` holds it).
pub(super) fn memo_key(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
) -> MemoKey {
    [
        std::ptr::from_ref(arena) as usize,
        arena.node_count(),
        body.0 as usize,
        var.0 as usize,
        lo.0 as usize,
        hi.0 as usize,
    ]
}

/// The outcome of a quadrature of `key` at a precision up to `prec` that
/// did not converge for want of nodes: at `prec` it would not either (a
/// higher precision only tightens the tolerance), so `evalf`'s search for
/// more digits gets the same answer at once instead of the same work again.
pub(super) fn recall(
    key: &MemoKey,
    prec: usize,
) -> Option<Result<(Complex, ErrExp, bool), SymplexError>> {
    FAILURES.with(|f| {
        f.borrow()
            .iter()
            .find(|e| e.key == *key && e.prec <= prec)
            .map(|e| match &e.outcome {
                Ok(v) => Ok(v.clone()),
                Err(err) => Err(super::duplicate_error(err)),
            })
    })
}

/// Remember a quadrature of `key` at `prec` that did not converge for want
/// of nodes.
pub(super) fn remember(
    key: MemoKey,
    prec: usize,
    outcome: Result<(Complex, ErrExp, bool), SymplexError>,
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

/// Is `e` a failure for want of nodes (a higher precision fails too), or a
/// proof of divergence?
pub(super) fn is_node_failure(e: &SymplexError) -> bool {
    matches!(e, SymplexError::Divergent { .. })
        || matches!(e, SymplexError::ComputationFailed { reason, .. }
        if reason.contains(NOT_DECAYING) || reason.contains("budget") || reason.contains("time limit"))
}

/// Is `e` the refusal of a quadrature whose terms do not decay towards an
/// end — of the range, or of a piece at a breakpoint ([`NOT_DECAYING`])?
pub(super) fn is_not_decaying(e: &SymplexError) -> bool {
    matches!(e, SymplexError::ComputationFailed { reason, .. } if reason.contains(NOT_DECAYING))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_f64(x: &BigFloat) -> f64 {
        super::super::bigfloat_to_f64_rounded(x, RoundingMode::ToEven).unwrap_or(f64::NAN)
    }

    /// The 12-point Gauss–Legendre rule (`m = 3`) integrates `x^22` exactly:
    /// `Σ w·(r²²·2) = 2/23`, and its weights sum to 2.
    #[test]
    fn gauss_legendre_nodes_are_exact_to_degree_2n_minus_1() {
        let rm = RoundingMode::ToEven;
        let p = 200;
        let (nodes, rel) = gauss_legendre(3, p, rm);
        assert_eq!(nodes.len(), 6);
        assert!(rel < -180.0);
        let mut sum_w = BigFloat::new(p);
        let mut moment = BigFloat::new(p);
        for (r, w) in &nodes {
            sum_w = sum_w.add(&scale2(w, 1), p, rm);
            moment = moment.add(&scale2(&w.mul(&r.powi(22, p, rm), p, rm), 1), p, rm);
        }
        let two = small(2);
        let exact = two.div(&small(23), p, rm);
        assert!(lg(&sum_w.sub(&two, p, rm)) < -180.0, "{}", to_f64(&sum_w));
        assert!(
            lg(&moment.sub(&exact, p, rm)) < -180.0,
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
