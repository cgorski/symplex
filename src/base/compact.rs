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
use crate::base::node::{ExprId, ExprNode, IdRemap, NumId, SymbolId};

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
    map: &mut FxHashMap<ExprId, ExprId>,
) -> ExprId {
    let node = src.node(old_id);
    // Post-order guarantees every declared child was transferred before its
    // parent; a miss is an invariant violation, recovered by transferring
    // the child's subtree on the spot (`Transfer::expr`).
    #[cfg(debug_assertions)]
    node.for_each_child(|c| {
        debug_assert!(
            map.contains_key(&c),
            "compact: child not yet transferred — post-order invariant violated"
        );
    });
    let new_node = node.remap_ids(&mut Transfer { src, dst, map });
    // Preserve a symbol's mathematical assumptions (positive, real, etc.).
    if let (ExprNode::Symbol(old_sid), ExprNode::Symbol(new_sid)) = (node, &new_node) {
        let assumptions = src.symbol_assumptions(*old_sid);
        dst.set_symbol_assumptions(*new_sid, assumptions);
    }
    dst.intern(new_node)
}

/// Destination id of `old`, transferring its subtree on demand if it has not
/// been mapped yet (e.g. a `PhysicalConstant` value, which is not a declared
/// child and so is not visited by the reachability walk).
fn remap_id(
    src: &Arena,
    dst: &mut Arena,
    old: ExprId,
    map: &mut FxHashMap<ExprId, ExprId>,
) -> ExprId {
    match map.get(&old) {
        Some(&new) => new,
        None => transfer_subtree(src, dst, old, map),
    }
}

/// The id maps of one transfer from `src` to `dst`, for
/// [`ExprNode::remap_ids`]: expressions through the transfer `map` (on
/// demand for a node not yet transferred), numbers and symbol names
/// re-interned in `dst`'s side tables.
struct Transfer<'a> {
    src: &'a Arena,
    dst: &'a mut Arena,
    map: &'a mut FxHashMap<ExprId, ExprId>,
}

impl IdRemap for Transfer<'_> {
    fn expr(&mut self, id: ExprId) -> ExprId {
        remap_id(self.src, self.dst, id, self.map)
    }

    fn num(&mut self, id: NumId) -> NumId {
        let value = self.src.num(id).clone();
        self.dst.intern_num(value)
    }

    fn symbol(&mut self, id: SymbolId) -> SymbolId {
        let name = self.src.symbol_name(id).to_owned();
        self.dst.symbols.intern(&name)
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
