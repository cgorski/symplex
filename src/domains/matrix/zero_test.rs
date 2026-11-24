//! Three-valued scalar helpers (shared with matrix_decomp / quaternion) (part of [`matrix`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Three-valued scalar helpers (shared with matrix_decomp / quaternion)
// ═══════════════════════════════════════════════════════════════════════════

/// Three-valued zero test for a scalar expression.
///
/// Layers: structural zero → assumption system → `eval().simplify()` →
/// numeric evaluation for constants → `together().simplify()` (common
/// denominator, catches rational-function identities such as
/// `1 − 2sin²θ/(2cosθ + 2) − cosθ`).  Returns `Unknown` when the value cannot
/// be decided symbolically (e.g. a free symbol without assumptions).
pub(crate) fn ex_is_zero(e: &Ex) -> Truth {
    if e.is_zero_structural() {
        return Truth::True;
    }
    // A rational function over ℚ(radicals, i) that vanishes identically is
    // zero for every value of its symbols; a constant one is decided
    // exactly either way.  (Before 0.31 `det` of `[[√2, 2], [a, √2·a]]`
    // was not recognised as zero and `inv` divided by it.)
    let rf = rational_function_is_zero(e);
    if rf.is_true() || (rf.is_false() && e.free_symbols().is_empty()) {
        return rf;
    }
    let t = Truth::from(e.is_zero());
    if t.is_known() {
        return t;
    }
    let s = e.eval().simplify();
    if s.is_zero_structural() {
        return Truth::True;
    }
    let t = Truth::from(s.is_zero());
    if t.is_known() {
        return t;
    }
    // Numeric constants (including `RootOf`, whose bound variable
    // `is_constant` would report as free): a single `RootOf` over ℚ is
    // decided exactly in its number field; anything else by the certified
    // test (certified digits, the exact test for algebraic numbers, the
    // deep zero search) — undecided stays `None`.  Before 0.30 an `f64`
    // value of exactly `0.0` was zero and one above `10⁻¹²` nonzero.
    let t = rootof_field_is_zero(&s);
    if t.is_known() {
        return t;
    }
    if s.eval_complex64().is_ok() {
        let mut inner = s.inner.write();
        return Truth::from(crate::poly::algebraic::is_zero_checked(
            &mut inner.arena,
            s.raw_id(),
        ));
    }
    // Symbolic with fractions: bring over a common denominator and retry.
    // Skipped when radicals are present: `together` then routes through a
    // canonicalisation path that is not yet robust for products of radicals.
    if !has_radical(&s) {
        let t = s.together();
        if t != s {
            let t = t.simplify();
            if t.is_zero_structural() {
                return Truth::True;
            }
            return Truth::from(t.is_zero());
        }
    }
    Truth::Unknown
}

/// Exact zero test for a constant built from rationals and one `RootOf(g,
/// k)` node by `+`, `·` and integer powers, with `g` irreducible over ℚ:
/// the expression is evaluated in the number field `ℚ[t]/(g)` (`RootOf ↦
/// t`), where it is zero iff its residue is — for every root of `g`, so
/// the index `k` does not matter.  `None` for any other shape (another
/// atom, two different `RootOf`s, a reducible or linear `g`, a division
/// by an expression that vanishes).
///
/// This is the zero test of Gaussian elimination on `A − λI` for a
/// rational `A` and an eigenvalue `λ = RootOf(g, k)`: every entry is such
/// an expression.  Before 0.30 those entries were decided by an `f64`
/// evaluation with an absolute threshold, so a matrix with small entries
/// had pivots of size `10⁻¹⁵` taken for zero (two "eigenvectors" for a
/// simple eigenvalue, none for another, `Av ≠ λv`).
pub(crate) fn rootof_field_is_zero(e: &Ex) -> Truth {
    Truth::from(rootof_field_residue(e).map(|r| r.is_zero()))
}

/// The residue of `e` in `ℚ[t]/(g)` for [`rootof_field_is_zero`]; `None`
/// for any other shape.
fn rootof_field_residue(e: &Ex) -> Option<crate::poly::dense::Poly> {
    use crate::base::node::{ExprId, ExprNode};
    use crate::poly::dense::Poly;
    let inner = e.inner.read();
    let arena = &inner.arena;
    // Pass 1: the shape, and the single `RootOf` atom.
    let mut root: Option<ExprId> = None;
    let mut stack = vec![e.raw_id()];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::RootOf(..) => {
                if root.is_some_and(|r| r != id) {
                    return None;
                }
                root = Some(id);
            }
            ExprNode::Num(_) => {}
            ExprNode::Add(ch) | ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
            ExprNode::Pow(b, x) => {
                if !arena.as_num(*x).is_some_and(|r| r.is_integer()) {
                    return None;
                }
                stack.push(*b);
            }
            _ => return None,
        }
    }
    let root = root?;
    let ExprNode::RootOf(p, v, _) = arena.node(root) else {
        return None;
    };
    let g = crate::poly::polybridge::expr_to_poly(arena, *p, *v)?;
    if g.degree()? < 2 || !irreducible_cached(&g) {
        return None;
    }
    let reduce = |a: Poly| a.rem(&g);
    let invert = |a: &Poly| -> Option<Poly> {
        if a.is_zero() {
            return None;
        }
        let e = Poly::extended_gcd(a, &g);
        // `g` irreducible and `a ≢ 0`: the gcd is a nonzero constant.
        let c = e.gcd.leading_coeff()?.clone();
        if e.gcd.degree() != Some(0) {
            return None;
        }
        Some(reduce(e.x.scale(&(Q::from_integer(1.into()) / c))))
    };
    // Pass 2: post-order evaluation in ℚ[t]/(g).
    let mut val: rustc_hash::FxHashMap<ExprId, Poly> = rustc_hash::FxHashMap::default();
    let mut stack = vec![(e.raw_id(), false)];
    while let Some((id, ready)) = stack.pop() {
        if val.contains_key(&id) {
            continue;
        }
        let node = arena.node(id);
        let children: Vec<ExprId> = match node {
            ExprNode::Add(ch) | ExprNode::Mul(ch) => ch.to_vec(),
            ExprNode::Pow(b, _) => vec![*b],
            _ => vec![],
        };
        if !ready && !children.is_empty() {
            stack.push((id, true));
            stack.extend(children.into_iter().map(|c| (c, false)));
            continue;
        }
        let r = match node {
            ExprNode::Num(_) => Poly::constant(arena.as_num(id)?.clone()),
            ExprNode::RootOf(..) => reduce(Poly::x()),
            ExprNode::Add(ch) => {
                let mut acc = Poly::zero();
                for c in ch {
                    acc = acc.add(val.get(c)?);
                }
                acc
            }
            ExprNode::Mul(ch) => {
                let mut acc = Poly::one();
                for c in ch {
                    acc = reduce(acc.mul(val.get(c)?));
                }
                acc
            }
            ExprNode::Pow(b, x) => {
                let n = arena.as_num(*x)?.to_integer();
                let base = val.get(b)?;
                let mut base = if n < 0.into() {
                    invert(base)?
                } else {
                    base.clone()
                };
                let mut k = num_traits::Signed::abs(&n);
                let mut acc = Poly::one();
                let two = BigInt::from(2);
                while k > 0.into() {
                    if (&k % &two) == 1.into() {
                        acc = reduce(acc.mul(&base));
                    }
                    k /= &two;
                    if k > 0.into() {
                        base = reduce(base.mul(&base));
                    }
                }
                acc
            }
            _ => return None,
        };
        val.insert(id, r);
    }
    val.remove(&e.raw_id())
}

/// Is `e` identically zero as a rational function over ℚ of its free
/// symbols?  `e` must be built from rationals and symbols by sums,
/// products and integer powers (read by explicit post-order walks);
/// `None` for any other node (a function, a radical, a constant such as
/// `π`).
///
/// `e` is first evaluated exactly at up to [`RF_POINTS`] fixed rational
/// points of large height: a nonzero value proves `e ≢ 0` (the common
/// case, linear in the size of the expression DAG).  When a value is zero
/// the numerator `P` of `e = P/Q` is expanded exactly and `e ≡ 0` iff
/// `P = 0`.  If that expansion exceeds [`RF_TERM_BUDGET`] terms, zero
/// values at all the points are only a hint (a nonzero `P` of degree `d`
/// vanishes at a point of height `h` with probability about `d/h`,
/// Schwartz–Zippel), which an exact test on a grid confirms or refutes
/// ([`rf_numerator_vanishes_on_grid`]); `None` when the grid is too large.
/// Up to 0.49 the hint was the answer: `Some(true)` on probabilistic
/// evidence.
///
/// The pivots of symbolic Gaussian elimination are such rational
/// functions.  Before 0.30 `linsolve` decided them by `simplify`, which
/// does not always bring an identically-zero combination to `0` (it then
/// divided by zero: a "unique" solution of `nan`s for a rank-deficient
/// system, or a consistent system reported inconsistent), and
/// `rref`/`rank`/`nullspace` only structurally (`rank` of
/// `[[2t − 2, 3t − 3], [6, 9]]` was 2, and null-space vectors were not in
/// the kernel).
///
/// Expressions that also contain radicals of rationals, the golden ratio or
/// `i` are decided exactly over `ℚ(radicals, i)` by
/// [`algebraic_function_is_zero`](crate::domains::linalg::algebraic_function_is_zero):
/// before 0.31 they returned `None` here, and `linsolve` then decided a
/// pivot mixing `√2` and a symbol by `simplify` (a consistent system was
/// "Inconsistent"), `rank`/`rref`/`nullspace` structurally (`rank` of
/// `[[√2 − 3, −2], [3√3 − √6, 2√3]]` was 2).
pub(crate) fn rational_function_is_zero(e: &Ex) -> Truth {
    rational_function_is_zero_q(e)
        .or_else(|| Truth::from(crate::domains::linalg::algebraic_function_is_zero(e)))
}

/// [`rational_function_is_zero`] for rational functions over ℚ (no
/// algebraic constants).
pub(super) fn rational_function_is_zero_q(e: &Ex) -> Truth {
    use num_traits::Zero;
    let inner = e.inner.read();
    let arena = &inner.arena;
    let syms = crate::base::walk::free_symbols(arena, e.raw_id());
    let mut zero_at_all = true;
    for k in 0..RF_POINTS {
        match rf_value_at(arena, e.raw_id(), &syms, k) {
            // Not rational-function arithmetic.
            None => return Truth::Unknown,
            Some(Some(v)) if !v.is_zero() => return Truth::False,
            Some(Some(_)) => {}
            // A denominator vanishes at this point: no information.
            Some(None) => zero_at_all = false,
        }
    }
    // No nonzero value: settle it exactly unless the numerator is too big.
    let exact = rf_numerator_is_zero(arena, e.raw_id(), &syms);
    if exact.is_known() {
        return exact;
    }
    // Zero at every point, each denominator nonzero there: the hint that
    // the exact test on a grid checks.
    if zero_at_all {
        return rf_numerator_vanishes_on_grid(arena, e.raw_id(), &syms);
    }
    Truth::Unknown
}

/// Most evaluation points of [`rf_numerator_vanishes_on_grid`].
const RF_GRID_POINTS: u64 = 4096;

/// Is the numerator `P` of `root = P/Q` (over ℚ in `syms`, the fraction
/// built node by node without cancellation: `a/b + c/d = (ad + cb)/(bd)`,
/// `(a/b)⁻ⁿ = bⁿ/aⁿ`) the zero polynomial?  Decided exactly from its
/// values on a grid: a polynomial of degree at most `dᵢ` in the `i`-th
/// variable that vanishes on `S₁ × … × Sₖ` with `|Sᵢ| = dᵢ + 1` is zero
/// (by induction on `k`: each coefficient in the last variable vanishes on
/// the smaller grid), and a nonzero value shows `P ≠ 0`.  The degrees are
/// bounded node by node.  `Unknown` when the grid would have more than
/// [`RF_GRID_POINTS`] points, or for a node that is not rational-function
/// arithmetic.
///
/// The caller has seen `root` evaluate to 0 at points where no
/// denominator vanishes, so `Q ≠ 0` and `root ≡ 0` iff `P = 0`.
pub(super) fn rf_numerator_vanishes_on_grid(
    arena: &crate::base::arena::Arena,
    root: crate::base::node::ExprId,
    syms: &[crate::base::node::ExprId],
) -> Truth {
    // `None` (every `?`): a node that is not rational-function arithmetic,
    // or a grid that is too large.
    let grid_test = || -> Option<bool> {
        use crate::base::node::{ExprId, ExprNode};
        use num_traits::Zero;
        let order = crate::base::walk::post_order_ids(arena, root);
        let index: rustc_hash::FxHashMap<ExprId, usize> =
            syms.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let exponent = |x: ExprId| -> Option<i64> {
            let k = arena.as_num(x)?;
            if !k.is_integer() {
                return None;
            }
            num_traits::ToPrimitive::to_i64(&k.to_integer()).filter(|k| (-64..=64).contains(k))
        };
        // Degree bounds (numerator, denominator) per variable.
        let nv = syms.len();
        let mut degs: rustc_hash::FxHashMap<ExprId, (Vec<u64>, Vec<u64>)> =
            rustc_hash::FxHashMap::default();
        for &id in &order {
            let r = match arena.node(id) {
                ExprNode::Num(_) => (vec![0; nv], vec![0; nv]),
                ExprNode::Symbol(_) => {
                    let mut n = vec![0; nv];
                    n[*index.get(&id)?] = 1;
                    (n, vec![0; nv])
                }
                ExprNode::Neg(b) => degs.get(b)?.clone(),
                ExprNode::Add(ch) => {
                    let (mut n, mut d) = (vec![0u64; nv], vec![0u64; nv]);
                    for c in ch {
                        let (cn, cd) = degs.get(c)?;
                        for i in 0..nv {
                            n[i] = (n[i].saturating_add(cd[i])).max(cn[i].saturating_add(d[i]));
                            d[i] = d[i].saturating_add(cd[i]);
                        }
                    }
                    (n, d)
                }
                ExprNode::Mul(ch) => {
                    let (mut n, mut d) = (vec![0u64; nv], vec![0u64; nv]);
                    for c in ch {
                        let (cn, cd) = degs.get(c)?;
                        for i in 0..nv {
                            n[i] = n[i].saturating_add(cn[i]);
                            d[i] = d[i].saturating_add(cd[i]);
                        }
                    }
                    (n, d)
                }
                ExprNode::Pow(b, x) => {
                    let k = exponent(*x)?;
                    let (bn, bd) = degs.get(b)?;
                    let m = k.unsigned_abs();
                    let scale = |v: &[u64]| v.iter().map(|&e| e.saturating_mul(m)).collect();
                    if k >= 0 {
                        (scale(bn), scale(bd))
                    } else {
                        (scale(bd), scale(bn))
                    }
                }
                _ => return None,
            };
            degs.insert(id, r);
        }
        let bounds = &degs.get(&root)?.0;
        let mut points: u64 = 1;
        for &d in bounds {
            points = points.saturating_mul(d.saturating_add(1));
            if points > RF_GRID_POINTS {
                return None;
            }
        }
        // Every point of {0, …, d₁} × … × {0, …, dₖ}, in mixed radix.
        let one = Q::from_integer(BigInt::from(1));
        for p in 0..points {
            let mut rest = p;
            let coords: Vec<Q> = bounds
                .iter()
                .map(|&d| {
                    let c = rest % (d + 1);
                    rest /= d + 1;
                    Q::from_integer(BigInt::from(c))
                })
                .collect();
            let mut val: rustc_hash::FxHashMap<ExprId, (Q, Q)> = rustc_hash::FxHashMap::default();
            for &id in &order {
                let r = match arena.node(id) {
                    ExprNode::Num(_) => (arena.as_num(id)?.clone(), one.clone()),
                    ExprNode::Symbol(_) => (coords[*index.get(&id)?].clone(), one.clone()),
                    ExprNode::Neg(b) => {
                        let (n, d) = val.get(b)?;
                        (-n.clone(), d.clone())
                    }
                    ExprNode::Add(ch) => {
                        let (mut n, mut d) = (Q::zero(), one.clone());
                        for c in ch {
                            let (cn, cd) = val.get(c)?;
                            n = &n * cd + cn * &d;
                            d *= cd;
                        }
                        (n, d)
                    }
                    ExprNode::Mul(ch) => {
                        let (mut n, mut d) = (one.clone(), one.clone());
                        for c in ch {
                            let (cn, cd) = val.get(c)?;
                            n *= cn;
                            d *= cd;
                        }
                        (n, d)
                    }
                    ExprNode::Pow(b, x) => {
                        let k = exponent(*x)?;
                        let (bn, bd) = val.get(b)?;
                        let m = i32::try_from(k.unsigned_abs()).ok()?;
                        if k >= 0 {
                            (num_traits::Pow::pow(bn, m), num_traits::Pow::pow(bd, m))
                        } else {
                            (num_traits::Pow::pow(bd, m), num_traits::Pow::pow(bn, m))
                        }
                    }
                    _ => return None,
                };
                val.insert(id, r);
            }
            if !val.get(&root)?.0.is_zero() {
                return Some(false);
            }
        }
        Some(true)
    };
    Truth::from(grid_test())
}

/// Number of evaluation points of [`rational_function_is_zero`].
pub(crate) const RF_POINTS: u64 = 3;

/// Largest intermediate polynomial (in terms) that
/// [`rational_function_is_zero`] expands exactly.
pub(crate) const RF_TERM_BUDGET: usize = 4000;

/// The rational value of `root` with symbol `syms[i]` set to point `k`'s
/// `i`-th coordinate.  `None` for a node that is not rational-function
/// arithmetic; `Some(None)` when a division by zero occurs.
pub(super) fn rf_value_at(
    arena: &crate::base::arena::Arena,
    root: crate::base::node::ExprId,
    syms: &[crate::base::node::ExprId],
    k: u64,
) -> Option<Option<Q>> {
    use crate::base::node::{ExprId, ExprNode};
    use num_traits::Zero;
    let point = |i: usize| rf_sample_point(i, k);
    let index: rustc_hash::FxHashMap<ExprId, usize> =
        syms.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    let mut val: rustc_hash::FxHashMap<ExprId, Q> = rustc_hash::FxHashMap::default();
    let mut stack = vec![(root, false)];
    while let Some((id, ready)) = stack.pop() {
        if val.contains_key(&id) {
            continue;
        }
        let node = arena.node(id);
        let children: Vec<ExprId> = match node {
            ExprNode::Add(ch) | ExprNode::Mul(ch) => ch.to_vec(),
            ExprNode::Pow(b, _) | ExprNode::Neg(b) => vec![*b],
            _ => vec![],
        };
        if !ready && !children.is_empty() {
            stack.push((id, true));
            stack.extend(children.into_iter().map(|c| (c, false)));
            continue;
        }
        let v = match node {
            ExprNode::Num(_) => arena.as_num(id)?.clone(),
            ExprNode::Symbol(_) => point(*index.get(&id)?),
            ExprNode::Neg(b) => -val.get(b)?.clone(),
            ExprNode::Add(ch) => {
                let mut acc = Q::zero();
                for c in ch {
                    acc += val.get(c)?;
                }
                acc
            }
            ExprNode::Mul(ch) => {
                let mut acc = Q::from_integer(BigInt::from(1));
                for c in ch {
                    acc *= val.get(c)?;
                }
                acc
            }
            ExprNode::Pow(b, x) => {
                let n = arena.as_num(*x)?;
                if !n.is_integer() {
                    return None;
                }
                let n = num_traits::ToPrimitive::to_i32(&n.to_integer())?;
                if !(-64..=64).contains(&n) {
                    return None;
                }
                let b = val.get(b)?;
                if n < 0 && b.is_zero() {
                    return Some(None);
                }
                num_traits::Pow::pow(b, n)
            }
            _ => return None,
        };
        val.insert(id, v);
    }
    Some(val.get(&root).cloned())
}

/// Coordinate `i` of the `k`-th sample point of the generic zero tests
/// ([`rational_function_is_zero`]): fixed, reproducible rationals of height
/// about 2⁴⁰.
pub(crate) fn rf_sample_point(i: usize, k: u64) -> Q {
    let mut z =
        (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xD1B5_4A32_D192_ED03);
    let mut next = || {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };
    let num = BigInt::from(next() >> 23) - BigInt::from(1u64 << 40);
    let den = BigInt::from((next() >> 24) | 1);
    Q::new(num, den)
}

/// Is the numerator of `root = P/Q` (over ℚ in `syms`) the zero
/// polynomial?  `Unknown` when an intermediate polynomial exceeds
/// [`RF_TERM_BUDGET`] terms, when a denominator is the zero polynomial, or
/// for a node that is not rational-function arithmetic.
pub(super) fn rf_numerator_is_zero(
    arena: &crate::base::arena::Arena,
    root: crate::base::node::ExprId,
    syms: &[crate::base::node::ExprId],
) -> Truth {
    Truth::from(rf_numerator(arena, root, syms).map(|p| p.is_zero()))
}

/// The expanded numerator for [`rf_numerator_is_zero`]; `None` as there.
fn rf_numerator(
    arena: &crate::base::arena::Arena,
    root: crate::base::node::ExprId,
    syms: &[crate::base::node::ExprId],
) -> Option<crate::poly::multipoly::MultiPoly<crate::poly::multipoly::GrevLex>> {
    use crate::base::node::{ExprId, ExprNode};
    use crate::poly::multipoly::{GrevLex, MultiPoly};
    type Mp = MultiPoly<GrevLex>;
    let nv = syms.len();
    let index: rustc_hash::FxHashMap<ExprId, usize> =
        syms.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    let mut val: rustc_hash::FxHashMap<ExprId, (Mp, Mp)> = rustc_hash::FxHashMap::default();
    let mut stack = vec![(root, false)];
    while let Some((id, ready)) = stack.pop() {
        if val.contains_key(&id) {
            continue;
        }
        let node = arena.node(id);
        let children: Vec<ExprId> = match node {
            ExprNode::Add(ch) | ExprNode::Mul(ch) => ch.to_vec(),
            ExprNode::Pow(b, _) | ExprNode::Neg(b) => vec![*b],
            _ => vec![],
        };
        if !ready && !children.is_empty() {
            stack.push((id, true));
            stack.extend(children.into_iter().map(|c| (c, false)));
            continue;
        }
        let r = match node {
            ExprNode::Num(_) => (
                Mp::constant(nv, arena.as_num(id)?.clone()),
                Mp::from_int(nv, 1),
            ),
            ExprNode::Symbol(_) => (Mp::var(nv, *index.get(&id)?), Mp::from_int(nv, 1)),
            ExprNode::Neg(b) => {
                let (n, d) = val.get(b)?;
                (n.neg(), d.clone())
            }
            ExprNode::Add(ch) => {
                let (mut n, mut d) = (Mp::zero(nv), Mp::from_int(nv, 1));
                for c in ch {
                    let (cn, cd) = val.get(c)?;
                    if *cd == d {
                        n = n.add(cn);
                    } else {
                        n = n.mul(cd).add(&cn.mul(&d));
                        d = d.mul(cd);
                    }
                    if n.num_terms().max(d.num_terms()) > RF_TERM_BUDGET {
                        return None;
                    }
                }
                (n, d)
            }
            ExprNode::Mul(ch) => {
                let (mut n, mut d) = (Mp::from_int(nv, 1), Mp::from_int(nv, 1));
                for c in ch {
                    let (cn, cd) = val.get(c)?;
                    n = n.mul(cn);
                    d = d.mul(cd);
                    if n.num_terms().max(d.num_terms()) > RF_TERM_BUDGET {
                        return None;
                    }
                }
                (n, d)
            }
            ExprNode::Pow(b, x) => {
                let k = arena.as_num(*x)?;
                if !k.is_integer() {
                    return None;
                }
                let k = num_traits::ToPrimitive::to_i64(&k.to_integer())?;
                let e = u32::try_from(k.unsigned_abs()).ok().filter(|&e| e <= 64)?;
                let (bn, bd) = val.get(b)?;
                let (n, d) = if k >= 0 {
                    (bn.pow(e), bd.pow(e))
                } else {
                    if bn.is_zero() {
                        return None;
                    }
                    (bd.pow(e), bn.pow(e))
                };
                if n.num_terms().max(d.num_terms()) > RF_TERM_BUDGET {
                    return None;
                }
                (n, d)
            }
            _ => return None,
        };
        if r.1.is_zero() {
            return None;
        }
        val.insert(id, r);
    }
    val.remove(&root).map(|(numerator, _)| numerator)
}

/// `is_irreducible_z(g) == Some(true)`, remembering the last polynomial
/// asked about (an elimination asks about the same `g` for every entry).
pub(super) fn irreducible_cached(g: &crate::poly::dense::Poly) -> bool {
    use std::cell::RefCell;
    thread_local! {
        static LAST: RefCell<Option<(crate::poly::dense::Poly, bool)>> = const { RefCell::new(None) };
    }
    if let Some(b) = LAST.with(|c| {
        c.borrow()
            .as_ref()
            .and_then(|(p, b)| (p == g).then_some(*b))
    }) {
        return b;
    }
    let b = Truth::from(crate::poly::factor_zassenhaus::is_irreducible_z(g)).is_true();
    LAST.with(|c| *c.borrow_mut() = Some((g.clone(), b)));
    b
}

/// Does `e` contain a power with a numeric non-integer exponent (a radical)?
pub(super) fn has_radical(e: &Ex) -> bool {
    use crate::base::node::ExprNode;
    let inner = e.inner.read();
    let arena = &inner.arena;
    let mut stack = vec![e.raw_id()];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let ExprNode::Pow(_, exp) = arena.node(id)
            && let Some(r) = arena.as_num(*exp)
            && !r.is_integer()
        {
            return true;
        }
        stack.extend(arena.node(id).children());
    }
    false
}

/// `exp(a·t + b·t + c)` → `exp(t·(a + b) + c)` throughout `e`: the
/// exponents of `matrix_exp_t` collected in `t` (`simplify` distributes
/// `t·(1/2 + √37/2)`; the integrator reads the collected form, and the
/// variation-of-parameters integrals of `solve_ode_system_nonhomogeneous`
/// stayed unevaluated otherwise).
pub(super) fn collect_exponents(e: &Ex, t: &Ex) -> Ex {
    use crate::base::node::{ExprId, ExprNode};
    let mut inner = e.inner.write();
    let arena = &mut inner.arena;
    let t_id = t.raw_id();
    let targets: Vec<(ExprId, ExprId)> = crate::base::walk::post_order_ids(arena, e.raw_id())
        .into_iter()
        .filter_map(|id| match arena.node(id) {
            ExprNode::Exp(arg) if crate::base::walk::contains(arena, *arg, t_id) => {
                Some((id, *arg))
            }
            _ => None,
        })
        .collect();
    let mut out = e.raw_id();
    for (exp_id, arg) in targets {
        let Some(coeffs) = crate::transforms::solve::symbolic_poly_coeffs(arena, arg, t_id) else {
            continue;
        };
        if coeffs.len() != 2 {
            continue;
        }
        let slope = crate::transforms::eval::eval(arena, coeffs[1]);
        let lin = arena.mul(&[slope, t_id]);
        let new_arg = arena.add(&[lin, coeffs[0]]);
        if new_arg == arg {
            continue;
        }
        let new_exp = arena.exp(new_arg);
        out = arena.subs_structural(out, exp_id, new_exp);
    }
    drop(inner);
    e.wrap(out)
}

/// Rewrite `sin(−c)` → `−sin(c)` and `cos(−c)` → `cos(c)` for *numeric*
/// literals `c > 0`.  The canonicaliser applies these parity rules to
/// symbolic negations (`sin(−x)`) but not to negative number literals, so
/// `½e^{i} + ½e^{−i}` would otherwise stay `½cos(1) + ½cos(−1) + …`.
pub(super) fn fix_trig_parity(e: &Ex) -> Ex {
    use crate::base::node::ExprNode;
    use num_traits::Signed;
    // Pass 1 (read lock): find sin/cos nodes with a negative numeric argument.
    let targets: Vec<(crate::base::node::ExprId, bool, Q)> = {
        let inner = e.inner.read();
        let arena = &inner.arena;
        let mut out = Vec::new();
        let mut stack = vec![e.raw_id()];
        let mut seen = rustc_hash::FxHashSet::default();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let (is_sin, arg) = match arena.node(id) {
                ExprNode::Sin(a) => (true, *a),
                ExprNode::Cos(a) => (false, *a),
                _ => {
                    stack.extend(arena.node(id).children());
                    continue;
                }
            };
            if let Some(r) = arena.as_num(arg)
                && r.is_negative()
            {
                out.push((id, is_sin, r.clone()));
            }
        }
        out
    };
    if targets.is_empty() {
        return e.clone();
    }
    // Pass 2 (no lock held): build the replacements, then substitute by id.
    let ctx = e.context();
    let map: rustc_hash::FxHashMap<_, Ex> = targets
        .into_iter()
        .map(|(id, is_sin, r)| {
            let pos = ctx.from_ratio(-r);
            (id, if is_sin { -pos.sin() } else { pos.cos() })
        })
        .collect();
    e.replace(|v| map.get(&v.id()).cloned())
}

/// Does `e` contain a `RootOf` node?
pub(super) fn has_root_of(e: &Ex) -> bool {
    use crate::base::node::ExprNode;
    let inner = e.inner.read();
    let arena = &inner.arena;
    let mut stack = vec![e.raw_id()];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if matches!(arena.node(id), ExprNode::RootOf(..)) {
            return true;
        }
        stack.extend(arena.node(id).children());
    }
    false
}

/// Is `e` an eigenvalue in radicals of a symbolic cubic or quartic
/// ([`parametric_radical_roots`]): free symbols under a root of index ≥ 3
/// or under a nested square root?
pub(super) fn is_parametric_radical(e: &Ex) -> bool {
    !e.free_symbols().is_empty() && (has_nested_higher_root(e) || is_nested_radical(e))
}

/// Does `e` contain Cardano's cube root `∛(… + √R)` of the symbols — a root
/// of index ≥ 3 whose radicand holds a radical and a free symbol?  Those are
/// the eigenvalues whose real values (*casus irreducibilis*) come out of
/// complex cube roots, imaginary parts cancelling only exactly; a binomial's
/// `∛p·ωᵏ` is not one.
pub(super) fn has_parametric_cardano_root(e: &Ex) -> bool {
    use crate::base::node::ExprNode;
    let inner = e.inner.read();
    let arena = &inner.arena;
    let radical = |id| {
        matches!(arena.node(id), ExprNode::Pow(_, x)
            if arena.as_num(*x).is_some_and(|r| !r.is_integer()))
    };
    let mut stack = vec![e.raw_id()];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let ExprNode::Pow(base, exp) = arena.node(id)
            && arena
                .as_num(*exp)
                .is_some_and(|r| *r.denom() > BigInt::from(2))
            && arena.as_num(*base).is_none()
        {
            let below = crate::base::walk::post_order_ids(arena, *base);
            if below.iter().any(|&j| radical(j))
                && below
                    .iter()
                    .any(|&j| matches!(arena.node(j), ExprNode::Symbol(_)))
            {
                return true;
            }
        }
        stack.extend(arena.node(id).children());
    }
    false
}

/// Does `e` contain a root of index ≥ 3 of a non-rational radicand (the
/// cube roots of Cardano's formulas over `ℚ(√p…, i)`)?
pub(crate) fn has_nested_higher_root(e: &Ex) -> bool {
    use crate::base::node::ExprNode;
    let inner = e.inner.read();
    let arena = &inner.arena;
    let mut stack = vec![e.raw_id()];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let ExprNode::Pow(base, exp) = arena.node(id)
            && let Some(r) = arena.as_num(*exp)
            && *r.denom() > BigInt::from(2)
            && arena.as_num(*base).is_none()
        {
            return true;
        }
        stack.extend(arena.node(id).children());
    }
    false
}

/// Did an operation fail on [`EXPRESSION_BUDGET`]?
pub(crate) fn is_swell(e: &SymplexError) -> bool {
    matches!(e, SymplexError::ExpressionSwell { .. })
}

/// `e` with the operation `operation` when it is an expression swell; the
/// other errors through `otherwise`.
pub(crate) fn swell_or(
    operation: &'static str,
    e: SymplexError,
    otherwise: impl FnOnce(SymplexError) -> SymplexError,
) -> SymplexError {
    match e {
        SymplexError::ExpressionSwell { nodes, .. } => {
            SymplexError::ExpressionSwell { operation, nodes }
        }
        e => otherwise(e),
    }
}

/// The final clean-up of an entry of `P·f(J)·P⁻¹` (`matrix_exp`,
/// `matrix_sqrt`, `matrix_log`, `matrix_pow_symbolic`): `simplify`, except
/// for entries with `RootOf` values or Cardano cube roots, where it can do
/// nothing (a casus-irreducibilis radical does not denest) and takes
/// about a second per entry on a 3×3 matrix over `ℚ(√2, √3, i)` — constant
/// folding is all they get.
pub(crate) fn simplify_function_entry(e: &Ex) -> Ex {
    if has_root_of(e) || has_nested_higher_root(e) {
        e.eval()
    } else {
        e.simplify()
    }
}

/// Three-valued "is strictly positive" test for a scalar expression: the
/// assumption system and certified evaluation ([`Ex::is_positive`]), of `e`
/// and of `e.eval().simplify()`.
///
/// Up to 0.49 an undecided value was then decided by the sign of its `f64`
/// value, which is `0.0` for a value below the exponent range or zero to
/// the precision reached: `[[e^{−10³⁰} − e^{−2·10³⁰}]]` (positive) was "not
/// positive definite" and `[[e^{−10⁻⁵⁰⁰⁰} − 1]]` (negative) positive
/// semidefinite ([`ex_is_nonnegative`]).
pub(crate) fn ex_is_positive(e: &Ex) -> Truth {
    Truth::from(e.is_positive()).or_else(|| Truth::from(e.eval().simplify().is_positive()))
}

/// Three-valued "is non-negative" test for a scalar expression, as
/// [`ex_is_positive`] (no `f64` sign either).
pub(crate) fn ex_is_nonnegative(e: &Ex) -> Truth {
    Truth::from(e.is_nonnegative()).or_else(|| Truth::from(e.eval().simplify().is_nonnegative()))
}

/// Square root of `e` with rational radicands rationalised: for a positive
/// rational `p/q` returns `√(p·q) / q` (so `√(1/2)` becomes `√2/2`, matching
/// the form the evaluator produces for `cos(π/4)`), otherwise `e.sqrt()`.
///
/// Used wherever a norm is divided out (Gram–Schmidt, quaternion axis
/// extraction) so that products of the resulting radicals cancel
/// structurally instead of needing `simplify()`.
pub(crate) fn sqrt_rationalized(e: &Ex) -> Ex {
    if let Some(r) = e.as_rational()
        && r.numer().sign() == num_bigint::Sign::Plus
    {
        let ctx = e.context();
        let p = ctx.from_bigint(r.numer().clone());
        let q = ctx.from_bigint(r.denom().clone());
        return &(&p * &q).sqrt() / &q;
    }
    e.sqrt()
}
