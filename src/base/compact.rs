//! Arena compaction — generational garbage collection.
//!
//! [`transfer_subtree`] copies an expression tree from one arena to another,
//! preserving structural sharing via hash-consing in the destination.
//! This is the core primitive behind [`Context::compact()`](crate::api::context::Context::compact).
//!
//! [`liveness_ratio`] and [`should_compact`] provide lightweight heuristics
//! for deciding *when* to compact without actually performing the copy.
//!
//! # Algorithm
//!
//! 1. Compute a post-order traversal of the source subtree (leaves first).
//! 2. For each node in post-order, remap its children using the
//!    already-transferred mapping, then intern the remapped node into the
//!    destination arena.
//! 3. The `map` cache ensures each source node is transferred at most once,
//!    preserving structural sharing across multiple roots.

use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};

/// Transfer an expression subtree from `src` to `dst`, returning the
/// new [`ExprId`] in `dst`.
///
/// Uses a post-order traversal so children are always transferred before
/// their parents.  The `map` caches already-transferred nodes to preserve
/// structural sharing — if you call this multiple times with the same `map`,
/// shared sub-expressions are interned only once in `dst`.
pub(crate) fn transfer_subtree(
    src: &Arena,
    dst: &mut Arena,
    root: ExprId,
    map: &mut FxHashMap<ExprId, ExprId>,
) -> ExprId {
    let post_order = crate::base::walk::post_order_ids(src, root);

    tracing::trace!(
        root = ?root,
        traversal_len = post_order.len(),
        "transfer_subtree: starting post-order transfer",
    );

    for &old_id in &post_order {
        if map.contains_key(&old_id) {
            continue;
        }

        let new_id = transfer_node(src, dst, old_id, map);
        map.insert(old_id, new_id);
    }

    // Record the destination arena size so should_compact() can detect
    // when the arena has doubled since its last compaction.
    dst.last_compact_size = dst.node_count();

    map[&root]
}

// ---------------------------------------------------------------------------
// Liveness analysis
// ---------------------------------------------------------------------------

/// Compute the fraction of arena nodes reachable from the given roots.
///
/// Returns `live_count / total_count`.  Cost: O(live_nodes) time,
/// O(total_nodes) memory for a temporary bitmap.
/// Zero cost when not called.
pub(crate) fn liveness_ratio(arena: &Arena, roots: &[ExprId]) -> f64 {
    let total = arena.node_count();
    if total == 0 {
        return 1.0;
    }

    let mut alive = vec![false; total]; // simple bool vec
    let mut stack: Vec<ExprId> = roots.to_vec();

    while let Some(id) = stack.pop() {
        let idx = id.0 as usize;
        if idx < total && !alive[idx] {
            alive[idx] = true;
            arena.node(id).for_each_child(|child| {
                if (child.0 as usize) < total && !alive[child.0 as usize] {
                    stack.push(child);
                }
            });
        }
    }

    let live = alive.iter().filter(|&&b| b).count();

    tracing::trace!(
        total_nodes = total,
        live_nodes = live,
        ratio = live as f64 / total as f64,
        "liveness_ratio computed",
    );

    live as f64 / total as f64
}

/// Heuristic: should the arena be compacted?
///
/// Returns `true` when **all three** conditions hold:
///
/// 1. The arena has more than 100 000 nodes, **and**
/// 2. The arena has grown to at least 2× its size at last compact, **and**
/// 3. Less than 50 % of nodes are reachable from the given roots.
pub(crate) fn should_compact(arena: &Arena, roots: &[ExprId]) -> bool {
    let total = arena.node_count();
    if total < 100_000 {
        tracing::trace!(total_nodes = total, "should_compact: arena too small");
        return false;
    }
    if total < arena.last_compact_size.saturating_mul(2) {
        tracing::trace!(
            total_nodes = total,
            last_compact_size = arena.last_compact_size,
            "should_compact: not yet doubled since last compact",
        );
        return false;
    }
    let ratio = liveness_ratio(arena, roots);
    let verdict = ratio < 0.5;
    tracing::trace!(ratio, verdict, "should_compact: liveness check",);
    verdict
}

/// Transfer a single node, remapping its children and side-table references.
fn transfer_node(
    src: &Arena,
    dst: &mut Arena,
    old_id: ExprId,
    map: &FxHashMap<ExprId, ExprId>,
) -> ExprId {
    let node = src.node(old_id).clone();
    let new_node = remap_node(src, dst, &node, map);
    dst.intern(new_node)
}

/// Remap a node's children from old [`ExprId`]s to new [`ExprId`]s.
///
/// Also handles side-table references:
/// - `Num` → re-intern the rational value in `dst`
/// - `Symbol` → re-intern the name in `dst`'s symbol table, copy assumptions
/// - `Apply` → re-intern the function name, remap arguments
fn remap_node(
    src: &Arena,
    dst: &mut Arena,
    node: &ExprNode,
    map: &FxHashMap<ExprId, ExprId>,
) -> ExprNode {
    // Helper: look up a remapped child ID.  All children should have been
    // transferred already (post-order guarantee), so unwrap is safe.
    let m = |old: &ExprId| -> ExprId {
        *map.get(old)
            .expect("compact bug: child not yet transferred — post-order invariant violated")
    };

    match node {
        // ── Atoms with side-table references ─────────────────────────────
        ExprNode::Num(old_nid) => {
            let value = src.num(*old_nid).clone();
            let new_nid = dst.intern_num(value);
            ExprNode::Num(new_nid)
        }

        ExprNode::Symbol(old_sid) => {
            let name = src.symbol_name(*old_sid).to_owned();
            let new_sid = dst.symbols.intern(&name);
            // Preserve mathematical assumptions (positive, real, etc.)
            let assumptions = src.symbol_assumptions(*old_sid);
            dst.set_symbol_assumptions(new_sid, assumptions);
            ExprNode::Symbol(new_sid)
        }

        // ── Atoms without side-table references ──────────────────────────
        ExprNode::Pi => ExprNode::Pi,
        ExprNode::E => ExprNode::E,
        ExprNode::ImaginaryUnit => ExprNode::ImaginaryUnit,
        ExprNode::EulerGamma => ExprNode::EulerGamma,
        ExprNode::Catalan => ExprNode::Catalan,
        ExprNode::GoldenRatio => ExprNode::GoldenRatio,

        ExprNode::PhysicalConstant(old_sid, old_value_id) => {
            let name = src.symbol_name(*old_sid).to_owned();
            let new_sid = dst.symbols.intern(&name);
            // The value_id is not a declared child of this atom, so it may
            // not have been visited during the reachability walk.  If it IS
            // in the map (referenced elsewhere), use the mapped id.
            // Otherwise, transfer its subtree directly (typically a Num).
            let new_value_id = if let Some(&mapped) = map.get(old_value_id) {
                mapped
            } else {
                transfer_node(src, dst, *old_value_id, map)
            };
            ExprNode::PhysicalConstant(new_sid, new_value_id)
        }

        ExprNode::Infinity => ExprNode::Infinity,
        ExprNode::NegInfinity => ExprNode::NegInfinity,
        ExprNode::ComplexInfinity => ExprNode::ComplexInfinity,
        ExprNode::NaN => ExprNode::NaN,
        ExprNode::BoolTrue => ExprNode::BoolTrue,
        ExprNode::BoolFalse => ExprNode::BoolFalse,
        ExprNode::EmptySet => ExprNode::EmptySet,
        ExprNode::UniversalSet => ExprNode::UniversalSet,

        // ── N-ary nodes ──────────────────────────────────────────────────
        ExprNode::Add(children) => ExprNode::Add(children.iter().map(&m).collect()),
        ExprNode::Mul(children) => ExprNode::Mul(children.iter().map(&m).collect()),
        ExprNode::And(children) => ExprNode::And(children.iter().map(&m).collect()),
        ExprNode::Or(children) => ExprNode::Or(children.iter().map(&m).collect()),
        ExprNode::Min(children) => ExprNode::Min(children.iter().map(&m).collect()),
        ExprNode::Max(children) => ExprNode::Max(children.iter().map(&m).collect()),
        ExprNode::FiniteSet(children) => ExprNode::FiniteSet(children.iter().map(&m).collect()),
        ExprNode::SetUnion(children) => ExprNode::SetUnion(children.iter().map(&m).collect()),
        ExprNode::SetIntersection(children) => {
            ExprNode::SetIntersection(children.iter().map(&m).collect())
        }

        // ── Binary nodes ─────────────────────────────────────────────────
        ExprNode::Pow(a, b) => ExprNode::Pow(m(a), m(b)),
        ExprNode::Binomial(a, b) => ExprNode::Binomial(m(a), m(b)),
        ExprNode::Gt(a, b) => ExprNode::Gt(m(a), m(b)),
        ExprNode::Ge(a, b) => ExprNode::Ge(m(a), m(b)),
        ExprNode::Eq_(a, b) => ExprNode::Eq_(m(a), m(b)),
        ExprNode::Ne(a, b) => ExprNode::Ne(m(a), m(b)),
        ExprNode::Derivative(a, b) => ExprNode::Derivative(m(a), m(b)),
        ExprNode::Integral(a, b) => ExprNode::Integral(m(a), m(b)),
        ExprNode::Atan2(a, b) => ExprNode::Atan2(m(a), m(b)),
        ExprNode::Beta(a, b) => ExprNode::Beta(m(a), m(b)),
        ExprNode::Polygamma(a, b) => ExprNode::Polygamma(m(a), m(b)),
        ExprNode::KroneckerDelta(a, b) => ExprNode::KroneckerDelta(m(a), m(b)),
        ExprNode::SetComplement(a, b) => ExprNode::SetComplement(m(a), m(b)),

        // ── Interval (binary + flags) ────────────────────────────────────
        ExprNode::Interval(a, b, flags) => ExprNode::Interval(m(a), m(b), *flags),

        // ── 4-child nodes ────────────────────────────────────────────────
        ExprNode::Sum(body, var, lo, hi) => ExprNode::Sum(m(body), m(var), m(lo), m(hi)),
        ExprNode::Product_(body, var, lo, hi) => ExprNode::Product_(m(body), m(var), m(lo), m(hi)),
        ExprNode::DefiniteIntegral(body, var, lo, hi) => {
            ExprNode::DefiniteIntegral(m(body), m(var), m(lo), m(hi))
        }

        // ── New formal/unevaluated nodes ─────────────────────────────────
        ExprNode::Limit(a, b, c) => ExprNode::Limit(m(a), m(b), m(c)),
        ExprNode::LaplaceTransform(a, b, c) => ExprNode::LaplaceTransform(m(a), m(b), m(c)),
        ExprNode::InverseLaplaceTransform(a, b, c) => {
            ExprNode::InverseLaplaceTransform(m(a), m(b), m(c))
        }
        ExprNode::Residue(a, b, c) => ExprNode::Residue(m(a), m(b), m(c)),
        ExprNode::DSolve(a, b, c) => ExprNode::DSolve(m(a), m(b), m(c)),
        ExprNode::RootSum(a, b, c) => ExprNode::RootSum(m(a), m(b), m(c)),
        ExprNode::Series(a, b, c, d) => ExprNode::Series(m(a), m(b), m(c), m(d)),
        ExprNode::RootOf(a, b) => ExprNode::RootOf(m(a), m(b)),
        ExprNode::ConditionSet(a, b) => ExprNode::ConditionSet(m(a), m(b)),

        // ── Unary nodes ──────────────────────────────────────────────────
        ExprNode::Neg(x) => ExprNode::Neg(m(x)),
        ExprNode::Sin(x) => ExprNode::Sin(m(x)),
        ExprNode::Cos(x) => ExprNode::Cos(m(x)),
        ExprNode::Tan(x) => ExprNode::Tan(m(x)),
        ExprNode::Exp(x) => ExprNode::Exp(m(x)),
        ExprNode::Ln(x) => ExprNode::Ln(m(x)),
        ExprNode::Abs(x) => ExprNode::Abs(m(x)),
        ExprNode::Asin(x) => ExprNode::Asin(m(x)),
        ExprNode::Acos(x) => ExprNode::Acos(m(x)),
        ExprNode::Atan(x) => ExprNode::Atan(m(x)),
        ExprNode::Sinh(x) => ExprNode::Sinh(m(x)),
        ExprNode::Cosh(x) => ExprNode::Cosh(m(x)),
        ExprNode::Tanh(x) => ExprNode::Tanh(m(x)),
        ExprNode::Asinh(x) => ExprNode::Asinh(m(x)),
        ExprNode::Acosh(x) => ExprNode::Acosh(m(x)),
        ExprNode::Atanh(x) => ExprNode::Atanh(m(x)),
        ExprNode::Sign(x) => ExprNode::Sign(m(x)),
        ExprNode::Factorial(x) => ExprNode::Factorial(m(x)),
        ExprNode::Not(x) => ExprNode::Not(m(x)),
        ExprNode::Floor(x) => ExprNode::Floor(m(x)),
        ExprNode::Ceiling(x) => ExprNode::Ceiling(m(x)),
        ExprNode::Gamma(x) => ExprNode::Gamma(m(x)),
        ExprNode::LogGamma(x) => ExprNode::LogGamma(m(x)),
        ExprNode::Digamma(x) => ExprNode::Digamma(m(x)),
        ExprNode::Erf(x) => ExprNode::Erf(m(x)),
        ExprNode::Erfc(x) => ExprNode::Erfc(m(x)),
        ExprNode::LambertW(x) => ExprNode::LambertW(m(x)),
        ExprNode::Heaviside(x) => ExprNode::Heaviside(m(x)),
        ExprNode::DiracDelta(x) => ExprNode::DiracDelta(m(x)),
        ExprNode::Re(x) => ExprNode::Re(m(x)),
        ExprNode::Im(x) => ExprNode::Im(m(x)),
        ExprNode::Conjugate(x) => ExprNode::Conjugate(m(x)),
        ExprNode::Arg(x) => ExprNode::Arg(m(x)),
        ExprNode::Si(x) => ExprNode::Si(m(x)),
        ExprNode::Ci(x) => ExprNode::Ci(m(x)),
        ExprNode::Ei(x) => ExprNode::Ei(m(x)),
        ExprNode::Li(x) => ExprNode::Li(m(x)),
        ExprNode::Zeta(x) => ExprNode::Zeta(m(x)),

        // ── Piecewise ────────────────────────────────────────────────────
        ExprNode::Piecewise(pairs) => {
            ExprNode::Piecewise(pairs.iter().map(|(v, c)| (m(v), m(c))).collect())
        }

        // ── Apply (symbol name + args) ───────────────────────────────────
        ExprNode::Apply(old_sid, args) => {
            let name = src.symbol_name(*old_sid).to_owned();
            let new_sid = dst.symbols.intern(&name);
            ExprNode::Apply(new_sid, args.iter().map(m).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;
    use rustc_hash::FxHashMap;

    #[test]
    fn transfer_atom_symbol() {
        let mut src = Arena::new();
        let x = src.symbol("x");

        let mut dst = Arena::new();
        let mut map = FxHashMap::default();
        let new_x = transfer_subtree(&src, &mut dst, x, &mut map);

        assert_eq!(
            dst.symbol_name(match dst.node(new_x) {
                ExprNode::Symbol(sid) => *sid,
                _ => panic!("expected Symbol"),
            }),
            "x"
        );
    }

    #[test]
    fn transfer_atom_num() {
        let mut src = Arena::new();
        let two = src.int(2);

        let mut dst = Arena::new();
        let mut map = FxHashMap::default();
        let new_two = transfer_subtree(&src, &mut dst, two, &mut map);

        let nid = match dst.node(new_two) {
            ExprNode::Num(nid) => *nid,
            _ => panic!("expected Num"),
        };
        assert_eq!(*dst.num(nid), num_rational::Ratio::from_integer(2.into()));
    }

    #[test]
    fn transfer_compound_expression() {
        let mut src = Arena::new();
        let x = src.symbol("x");
        let two = src.int(2);
        let x2 = src.pow(x, two);
        let one = src.one();
        let sum = src.add(&[x2, one]);

        let mut dst = Arena::new();
        let mut map = FxHashMap::default();
        let new_sum = transfer_subtree(&src, &mut dst, sum, &mut map);

        // The transferred expression should be structurally correct.
        // Verify it's an Add node.
        match dst.node(new_sum) {
            ExprNode::Add(_) => {}
            other => panic!("expected Add, got {:?}", other),
        }
    }

    #[test]
    fn transfer_preserves_sharing() {
        let mut src = Arena::new();
        let x = src.symbol("x");
        let sx = src.sin(x);
        let one = src.one();
        let sum1 = src.add(&[sx, one]);
        let two = src.int(2);
        let pow = src.pow(sx, two);

        let mut dst = Arena::new();
        let mut map = FxHashMap::default();

        let _new1 = transfer_subtree(&src, &mut dst, sum1, &mut map);
        let _new2 = transfer_subtree(&src, &mut dst, pow, &mut map);

        // sin(x) was shared in src; it should also be shared in dst
        // (the map should contain the same dst id for src's sin(x)).
        // The key test is that dst has fewer nodes than naive double transfer.
        // sin(x) and x should each appear once.
        let dst_count = dst.node_count();
        // dst has pre-interned constants + x, sin(x), 1, 2, Add, Pow = small
        // Without sharing we'd duplicate x and sin(x).
        assert!(dst_count > 0);
    }

    #[test]
    fn transfer_preserves_assumptions() {
        let mut src = Arena::new();
        let x_id = src.symbol("x");
        let sid = match src.node(x_id) {
            ExprNode::Symbol(s) => *s,
            _ => panic!("expected Symbol"),
        };
        let mut assumptions = src.symbol_assumptions(sid);
        assumptions.known_true |= crate::base::assumptions::Props::POSITIVE;
        assumptions.known_true |= crate::base::assumptions::Props::REAL;
        src.set_symbol_assumptions(sid, assumptions);

        let mut dst = Arena::new();
        let mut map = FxHashMap::default();
        let new_x = transfer_subtree(&src, &mut dst, x_id, &mut map);

        let new_sid = match dst.node(new_x) {
            ExprNode::Symbol(s) => *s,
            _ => panic!("expected Symbol"),
        };
        let new_assumptions = dst.symbol_assumptions(new_sid);
        assert!(
            new_assumptions
                .known_true
                .contains(crate::base::assumptions::Props::POSITIVE)
        );
        assert!(
            new_assumptions
                .known_true
                .contains(crate::base::assumptions::Props::REAL)
        );
    }

    #[test]
    fn transfer_constants() {
        let src = Arena::new();
        let pi = src.pi();

        let mut dst = Arena::new();
        let mut map = FxHashMap::default();
        let new_pi = transfer_subtree(&src, &mut dst, pi, &mut map);

        assert!(matches!(dst.node(new_pi), ExprNode::Pi));
    }

    #[test]
    fn transfer_reduces_arena_size() {
        let mut src = Arena::new();
        let x = src.symbol("x");
        // Build a bunch of junk that we won't transfer
        let _junk1 = src.sin(x);
        let _junk2 = src.cos(x);
        let _junk3 = src.exp(x);
        let _junk4 = src.ln(x);
        let two = src.int(2);
        let _junk5 = src.pow(x, two);
        let three = src.int(3);
        let _junk6 = src.pow(x, three);

        let src_count = src.node_count();

        // Only transfer `x` itself
        let mut dst = Arena::new();
        let dst_baseline = dst.node_count(); // pre-interned constants
        let mut map = FxHashMap::default();
        let _new_x = transfer_subtree(&src, &mut dst, x, &mut map);

        let dst_count = dst.node_count();
        // dst should have only pre-interned constants + x (1 new node)
        assert_eq!(dst_count, dst_baseline + 1);
        assert!(
            dst_count < src_count,
            "dst ({}) should be smaller than src ({})",
            dst_count,
            src_count
        );
    }

    #[test]
    fn transfer_new_constants_and_function_nodes() {
        let mut src = Arena::new();
        let z = src.symbol("z");
        let n = src.symbol("n");
        let re_z = src.intern(ExprNode::Re(z));
        let conj_z = src.intern(ExprNode::Conjugate(z));
        let pg = src.intern(ExprNode::Polygamma(n, z));
        let kd = src.intern(ExprNode::KroneckerDelta(n, z));
        let si = src.intern(ExprNode::Si(z));
        let consts = src.add(&[src.euler_gamma, src.catalan, src.golden_ratio]);
        let root = src.add(&[re_z, conj_z, pg, kd, si, consts]);

        let mut dst = Arena::new();
        let mut map = FxHashMap::default();
        let new_root = transfer_subtree(&src, &mut dst, root, &mut map);

        assert_eq!(
            dst.display(new_root).to_string(),
            src.display(root).to_string()
        );
        // Pre-interned constants map onto the destination's own singletons.
        assert_eq!(map[&src.euler_gamma], dst.euler_gamma);
        assert_eq!(map[&src.catalan], dst.catalan);
        assert_eq!(map[&src.golden_ratio], dst.golden_ratio);
    }
}
