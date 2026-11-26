//! Products: three-factor by parts, trig product-to-sum (part of [`integrate`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Products: three-factor by parts, trig product-to-sum
// ═══════════════════════════════════════════════════════════════════════════

/// The size the cyclic rule bounds before it works: an upper bound of the
/// degree of `expr` in the variable, read off the tree without multiplying
/// anything out.  The variable counts 1, a product adds, a sum (and any
/// function, for its arguments) takes the largest, and a power with a
/// numeric exponent `q` multiplies by `⌈|q|⌉`: `x⁷⁶⁰` is 760, `x^(759/2)` is
/// 380, `(1 + 323·x²)⁻¹⁶²` is 324, `eˣ·sin x` is 1.  Deterministic, linear in
/// the tree, saturating.
pub(super) fn degree_bound(arena: &Arena, expr: ExprId, var_sym: SymbolId) -> usize {
    size_in(arena, expr, var_sym, false)
}

/// The size by parts bounds before it searches: the number of levels the
/// chain from a polynomial `u` needs before `u` is used up.  As
/// [`degree_bound`], but a power of anything other than a constant counts
/// its exponent only: `x⁷⁶⁰` is 760 and `x¹⁰⁰ + x³` is 100 (a level lowers the
/// degree by one), `(1 + x⁸)¹²` is 12 (a level lowers the outer power:
/// `∫ (1 + x⁸)¹²·ln x dx` is found that way).
fn chain_length(arena: &Arena, u: ExprId, var_sym: SymbolId) -> usize {
    size_in(arena, u, var_sym, true)
}

/// [`degree_bound`], or with `outer_power` [`chain_length`].
fn size_in(arena: &Arena, expr: ExprId, var_sym: SymbolId, outer_power: bool) -> usize {
    let mut size: FxHashMap<ExprId, usize> = FxHashMap::default();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let of = |child: &ExprId| size.get(child).copied().unwrap_or(0);
        let d = match arena.node(id) {
            ExprNode::Symbol(s) => usize::from(*s == var_sym),
            ExprNode::Num(_) => 0,
            ExprNode::Mul(children) => children
                .iter()
                .fold(0usize, |sum, child| sum.saturating_add(of(child))),
            ExprNode::Pow(base, exp) => match arena.as_num(*exp) {
                Some(q) => {
                    let steps = num_traits::ToPrimitive::to_usize(&q.abs().ceil().to_integer())
                        .unwrap_or(usize::MAX);
                    if outer_power {
                        of(base).min(1).saturating_mul(steps)
                    } else {
                        of(base).saturating_mul(steps)
                    }
                }
                None => of(base).max(of(exp)),
            },
            _ => arena.children(id).iter().map(of).max().unwrap_or(0),
        };
        size.insert(id, d);
    }
    size.get(&expr).copied().unwrap_or(0)
}

/// The largest [`chain_length`] of a polynomial `u` for which by parts
/// ([`try_by_parts_pair`]) searches for `∫ v·du` when `dv` is transcendental
/// ([`transcendental_in`]).  Every level of by parts shortens the chain of
/// such a `u` by one and keeps the transcendental factor, and the search has
/// twenty levels (the depth of `integrate_node`): a chain longer than twenty
/// cannot reach its end.  Up to 0.60.0 nothing bounded the
/// attempt.  After `x = u²`, `∫ x^(759/2)·e^(−x/2) dx` (the mean of a
/// truncated `χ²(759)` law) went down all twenty levels from `u⁷⁶⁰`, both
/// orderings and every other strategy at each, with their polynomial
/// arithmetic at degree 760: 14 s for an unevaluated integral, 5.7 s for
/// `x^(481/2)` (`fuzz_stats`, 2026-11).  Beyond the bound `∫ v·du` is taken
/// only if it is at hand ([`BEYOND_BOUND_DEPTH`]), and a remaining integral
/// that is a multiple of the integrand is recognised as before
/// ([`by_parts_cycle`]: `∫ x⁵⁰⁰·|x| dx`).
///
/// Where 64 comes from.  A build that records [`degree_bound`] (never less
/// than the chain length) at every attempt and at every success (the bounds
/// off; 2026-11-26, on the base of 0.60.1) gave, for a polynomial `u` and a
/// transcendental `dv`: largest success 20
/// in `rubi-harness --check` (`x²⁰·eˣ`, the chain of twenty levels), 3 in the
/// test suite, 6 in the golden corpus; largest size offered 101, 9·10¹⁰
/// and 14.  The hunt of the change (`x^(n/2)·e^(−x/2)` times a polynomial or
/// rational factor, `n` to 2000) has successes up to 99, all of them with
/// `∫ v·du` at hand (`x¹⁰⁰·e^(−x/2)/(1 + x)`: the quotient's terms in rising
/// order).  A polynomial times `eˣ` is integrated up to degree 50 and no
/// further, with or without the bound.  The bound is not for an algebraic
/// `dv`: there the rule succeeds up to degree 468 in Rubi (`x³⁶·(a +
/// b·x³⁷)¹²`, a polynomial; `(a + b·x³)⁸·x⁻⁴³`, where a level lowers the
/// outer power).
pub(super) const MAX_BY_PARTS_POLY_DEGREE: usize = 64;

/// The depth of the search for `∫ v·du` beyond [`MAX_BY_PARTS_POLY_DEGREE`]:
/// two levels, which reach an integral that this `integrate` call has
/// solved already (the memo of `integrate_node`).  The terms of
/// `x¹⁰⁰·e^(−x/2)/(1 + x)` after the division are integrated in rising order,
/// and `∫ x⁹⁶·e^(−x/2) dx` is by parts twice and then the term `x⁹⁴·e^(−x/2)`
/// (found in 0.60.0; with no search at all it was lost, with one level too).
const BEYOND_BOUND_DEPTH: usize = 2;

/// The largest [`degree_bound`] of a factor that the cyclic rule
/// (`try_cyclic_ibp`) works on.  The rule integrates one factor twice and
/// differentiates the other twice before it can see whether the integral
/// comes back, whatever their size.  Up to 0.60.0: after `x = u²`,
/// `∫ e^(−x/8)·x^(321/2)·(1 + 323·x)^(−162) dx` (the moment generating
/// function of `F(323, 1)`) cost two Hermite reductions of a denominator of
/// degree 324, 41 s for an unevaluated integral (`fuzz_stats`, 2026-11).
///
/// Where 64 comes from (the same recording build): the largest size at
/// which the rule finds its cycle is 13 in `rubi-harness --check`, 9 in the
/// test suite, 3 in the golden corpus and 1 in the hunt (`e^(200x)·
/// sin(201x)`); the largest size offered is 386, 9·10¹⁰, 16 and 1001.
pub(super) const MAX_CYCLIC_IBP_DEGREE: usize = 64;

/// Does `expr` hold a function of the variable that is not algebraic: any
/// node that depends on the variable other than a sum, a product and a power
/// with an exponent free of it (`eˣ`, `erf x`, `ln x`, `2ˣ`; not `√x`, `xᵐ`)?
fn transcendental_in(arena: &Arena, expr: ExprId, var_sym: SymbolId) -> bool {
    let mut depends: FxHashSet<ExprId> = FxHashSet::default();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let node = arena.node(id);
        let on_var = match node {
            ExprNode::Symbol(s) => *s == var_sym,
            _ => arena.children(id).iter().any(|c| depends.contains(c)),
        };
        if !on_var {
            continue;
        }
        match node {
            ExprNode::Symbol(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => {}
            ExprNode::Pow(_, exp) if !depends.contains(exp) => {}
            _ => return true,
        }
        depends.insert(id);
    }
    false
}

/// Is `u` a polynomial of the variable beyond [`MAX_BY_PARTS_POLY_DEGREE`],
/// and `dv` transcendental?  Read off the two trees, before any work.
fn polynomial_beyond_by_parts(
    arena: &Arena,
    u: ExprId,
    dv: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> bool {
    is_polynomial_in(arena, u, var, var_sym)
        && chain_length(arena, u, var_sym) > MAX_BY_PARTS_POLY_DEGREE
        && transcendental_in(arena, dv, var_sym)
}

/// `∫ u(x)·g(x)·h(x) dx` by parts with `dv = g·h`: `u` a logarithm, an
/// inverse trigonometric or another by-parts function of the variable
/// first (LIATE), then a polynomial `P` (e.g. `x·eˣ·sin x`).  Up to 0.46
/// only `u = P` was tried: `∫ x·eˣ·ln x dx` needed `∫ Ei(x) dx` that way
/// (SymPy's test_integrals: `(x − 1)·eˣ·ln x − eˣ + Ei(x)`, which `u = ln x`,
/// `v = (x − 1)·eˣ` gives at once; the ODE `y″ + 2y′ + y = ln x` needs it).
pub(super) fn try_by_parts_poly_times_pair(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if rational_product(arena, dependent, var_sym) {
        return None;
    }
    // One `exp`/`sin`/`cos`/`sinh`/`cosh` of a linear argument over a
    // rational function is the special-function table's (`∫ eˣ·(x − 1)/x² dx
    // = eˣ/x` there; by parts with `u = x − 1` left `Ei` terms that cancel
    // only numerically).
    if special_times_rational(arena, dependent, var, var_sym) {
        return None;
    }
    // Without a polynomial factor the rule is not for this integrand
    // (`∫ ln x/(x·(ln²x + 2)) dx` is the substitution `u = ln x`).
    let poly_idx = dependent
        .iter()
        .position(|&d| is_polynomial_in(arena, d, var, var_sym))?;
    // A logarithm or inverse function of a linear argument (its `du` is a
    // simple rational function); for `atan((t + ½)⁶)` the remaining `∫ v·du`
    // only fed long failing searches (`∫ atan(x³)/(√x − ½) dx`: 77 ms → 1.1 s).
    let mut order: Vec<(u8, usize)> = dependent
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| {
            let rank = liate_rank(arena, d, var, var_sym);
            (rank <= 2
                && is_by_parts_candidate(arena, d, var, var_sym)
                && single_argument(arena, d)
                    .is_some_and(|g| linear_in_structurally(arena, g, var_sym)))
            .then_some((rank, i))
        })
        .collect();
    order.sort_unstable();
    order.push((3, poly_idx));
    for (_, idx) in order {
        let u = dependent[idx];
        let dv = remaining_product(arena, dependent, idx);
        if special_v_without_route(arena, u, dv, var, var_sym) {
            continue;
        }
        // A polynomial `dv` multiplied out, so that `v` is one polynomial
        // (`x·(1 + x²)` by parts gave `x·(x³/3 + x) − x⁴/12 − x²/2`).
        let dv = if is_polynomial_in(arena, dv, var, var_sym) {
            crate::transforms::expand::expand(arena, dv)
        } else {
            dv
        };
        let v = integrate_node(arena, dv, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, v) || has_root_sum_of_var(arena, v, var_sym) {
            continue;
        }
        let du = crate::transforms::diff::diff(arena, u, var);
        let v_du = arena.mul(&[v, du]);
        let u_dv = arena.mul(&[u, dv]);
        match by_parts_cycle(arena, u, v, u_dv, v_du, var_sym) {
            Cycle::Solved(result) => return Some(result),
            Cycle::Hopeless => continue,
            Cycle::No => {}
        }
        let v_du = crate::transforms::expand::expand(arena, v_du);
        let rest = integrate_node(arena, v_du, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, rest) {
            continue;
        }
        let uv = arena.mul(&[u, v]);
        return Some(arena.sub(uv, rest));
    }
    None
}

/// What [`by_parts_cycle`] found.
enum Cycle {
    /// `∫ v·du` is not a constant multiple of the integrand.
    No,
    /// `∫ v·du = c·I`: `I = u·v/(1 + c)`.
    Solved(ExprId),
    /// `∫ v·du = −I`: by parts says `I = u·v + I`, nothing.
    Hopeless,
}

/// Is the integral left by parts, `∫ v·du`, a constant multiple `c` of the
/// integrand `u·dv` itself?  Then `I = u·v − c·I` gives `I = u·v/(1 + c)`
/// at once (SymPy's `manualintegrate` solves such cyclic parts the same
/// way), and for `c = −1` by parts proves nothing.  Before, the integrator
/// integrated `v·du` — the same integral again — by parts again, one level
/// deeper each time down to the depth limit, every other strategy running
/// at every level: `∫ x²·sign(x)` (`v = x·sign(x)`, `v·du = 2x²·sign(x)`)
/// is the core of `∫ (√x + |x|)/(x + x⁻² + 1) dx`, which took 74 s to be
/// refused (`fuzz_integrate`, 2026-10-01).  The ratio is the canonical
/// quotient, so only a structural multiple is recognised.
fn by_parts_cycle(
    arena: &mut Arena,
    u: ExprId,
    v: ExprId,
    u_dv: ExprId,
    v_du: ExprId,
    var_sym: SymbolId,
) -> Cycle {
    let ratio = arena.div(v_du, u_dv);
    let c = crate::transforms::eval::eval(arena, ratio);
    if contains_var(arena, c, var_sym)
        || contains_non_finite(arena, c)
        || arena.is_zero_structural(c)
    {
        return Cycle::No;
    }
    let one = arena.one;
    let one_plus_c = arena.add(&[one, c]);
    let one_plus_c = crate::transforms::eval::eval(arena, one_plus_c);
    if arena.is_zero_structural(one_plus_c) {
        return Cycle::Hopeless;
    }
    tracing::debug!("by parts: the remaining integral is a multiple of the integrand");
    let uv = arena.mul(&[u, v]);
    Cycle::Solved(arena.div(uv, one_plus_c))
}

/// Is the product of the factors `dependent` a rational function of the
/// variable over ℚ (sums, products and integer powers of the variable and
/// of rational numbers) with a denominator (a polynomial is not the
/// rational integrator's)?  Then by parts is no strategy: Hermite's reduction
/// and Rothstein–Trager are complete there, and by parts takes `u` = a
/// polynomial factor and `dv` = the reciprocal of the rest, whose `v`
/// (logarithms, arctangents, a `RootSum` or the explicit roots of a
/// quartic) leaves `∫ v·du` harder than the integrand.  Up to 0.44 the
/// algebraic remainder `(−x³/6 + …)/(x⁴ − x³ + x² − x + 2)` of
/// `∫ dx/(x⁵ + x + 2)`, which `try_risch_rational` hands to the heuristic
/// integrator, went down 8 levels of by parts and sum distribution before
/// it gave up (0.5 s of its 0.57 s), for the same final answer.  (With
/// parameters by parts still runs: the rational integrator over `ℚ(a, …)`
/// leaves products such as `(a + b·x)ᵐ·(c + d·x)ⁿ` to it.)
fn rational_product(arena: &Arena, dependent: &[ExprId], var_sym: SymbolId) -> bool {
    let has_denominator = dependent.iter().any(|&d| {
        matches!(arena.node(d), ExprNode::Pow(b, e)
            if arena.as_num(*e).is_some_and(|q| q.is_integer() && q.is_negative())
                && contains_var(arena, *b, var_sym))
    });
    has_denominator
        && dependent.iter().all(|&d| {
            crate::base::walk::post_order_ids(arena, d)
                .into_iter()
                .all(|id| match arena.node(id) {
                    ExprNode::Num(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => {
                        true
                    }
                    ExprNode::Pow(_, e) => arena.as_num(*e).is_some_and(|q| q.is_integer()),
                    ExprNode::Symbol(s) => *s == var_sym,
                    _ => false,
                })
        })
}

/// Does `e` hold a `RootSum` that depends on the variable?  As a by-parts
/// `v` it leaves `∫ v·du` with no route (see [`integrate_impl`]: every
/// strategy would take the `ln(x − t)` of its body for a function of `x`
/// alone).
fn has_root_sum_of_var(arena: &Arena, e: ExprId, var_sym: SymbolId) -> bool {
    crate::base::walk::post_order_ids(arena, e)
        .iter()
        .any(|&id| {
            matches!(arena.node(id), ExprNode::RootSum(..)) && contains_var(arena, id, var_sym)
        })
}

/// Is the product of `dependent` one `exp`, `sin`, `cos`, `sinh`, `cosh` of
/// a linear argument times a rational function with a denominator?
fn special_times_rational(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> bool {
    let mut special = 0usize;
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &f in dependent {
        match LinearSpecial::of(arena, f) {
            Some((_, arg)) if symbolic_linear_coeff_of(arena, arg, var, var_sym).is_some() => {
                special += 1;
            }
            _ => rest.push(f),
        }
    }
    if special != 1 || rest.is_empty() {
        return false;
    }
    let r = arena.mul(&rest);
    if !is_rational_in(arena, r, var_sym) {
        return false;
    }
    let (_, den) = crate::poly::polybridge::as_numer_denom(arena, r);
    contains_var(arena, den, var_sym)
}

/// Would by parts with this `u` and `dv` build a `v` of `Ei`, `Si`/`Ci`,
/// `Shi`/`Chi` (a function `exp`, `sin`, `cos`, `sinh`, `cosh` of a linear
/// argument times a rational function with a denominator in the variable)
/// only to meet `∫ v·du` with a `du` that is no polynomial, which no route
/// integrates (`Si(x − r)·R(x)` for a proper rational `R`)?  Building that `v`
/// over the complex roots of `x⁸ + 1` and verifying it took 10–30 ms in
/// failing searches (`∫ cos(√x)·atan(x²) dx`: 9 ms in 0.43, 118 ms in 0.45;
/// agent B14's fuzz units).  Checked on the structure, before `v` is built.
fn special_v_without_route(
    arena: &mut Arena,
    u: ExprId,
    dv: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> bool {
    if is_polynomial_in(arena, u, var, var_sym) {
        return false;
    }
    let factors: Vec<ExprId> = match arena.node(dv) {
        ExprNode::Mul(children) => children.to_vec(),
        _ => vec![dv],
    };
    let mut special = 0usize;
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    for f in factors {
        if !contains_var(arena, f, var_sym) {
            continue;
        }
        match LinearSpecial::of(arena, f) {
            Some((_, arg)) if symbolic_linear_coeff_of(arena, arg, var, var_sym).is_some() => {
                special += 1;
            }
            _ => rest.push(f),
        }
    }
    if special != 1 || rest.is_empty() {
        return false;
    }
    let r = arena.mul(&rest);
    if !is_rational_in(arena, r, var_sym) {
        return false;
    }
    let (_, den) = crate::poly::polybridge::as_numer_denom(arena, r);
    if !contains_var(arena, den, var_sym) {
        return false;
    }
    let du = crate::transforms::diff::diff(arena, u, var);
    !is_polynomial_in(arena, du, var, var_sym)
}

/// Integration by parts for a product of two dependent factors, one a
/// by-parts candidate `u` (a polynomial, a logarithm or an inverse
/// trigonometric function) and the other `dv` with a closed form:
/// `∫ u·dv = u·v − ∫ v·du`.  The LIATE-preferred ordering is tried first;
/// never for a rational function ([`rational_product`]).
pub(super) fn try_by_parts_pair(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if rational_product(arena, dependent, var_sym) {
        return None;
    }
    let orderings = {
        let r0 = liate_rank(arena, dependent[0], var, var_sym);
        let r1 = liate_rank(arena, dependent[1], var, var_sym);
        tracing::debug!(u_rank = r0, dv_rank = r1, "by-parts LIATE ordering");
        if r0 <= r1 {
            [(0usize, 1usize), (1, 0)]
        } else {
            [(1, 0), (0, 1)]
        }
    };
    for (u_idx, dv_idx) in orderings {
        let u = dependent[u_idx];
        let dv = dependent[dv_idx];

        // Check that u is a by-parts candidate (polynomial or ln)
        if !is_by_parts_candidate(arena, u, var, var_sym) {
            continue;
        }
        if special_v_without_route(arena, u, dv, var, var_sym) {
            continue;
        }

        // Check that dv is directly integrable (deep check: reject
        // results that contain nested unevaluated Integral nodes,
        // e.g. Add(Integral(..), Integral(..)) which has a non-
        // Integral top node but is still not fully evaluated).
        let v = integrate_node(arena, dv, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, v) {
            tracing::trace!("by-parts: v = ∫dv has unevaluated nodes, skipping this ordering");
            continue; // dv not integrable
        }
        if has_root_sum_of_var(arena, v, var_sym) {
            continue;
        }

        // Compute du = d(u)/dx
        let du = crate::transforms::diff::diff(arena, u, var);

        // Compute ∫ v·du dx
        let v_du = arena.mul(&[v, du]);
        let u_dv = arena.mul(&[u, dv]);
        match by_parts_cycle(arena, u, v, u_dv, v_du, var_sym) {
            Cycle::Solved(result) => return Some(result),
            Cycle::Hopeless => continue,
            Cycle::No => {}
        }
        // Beyond the size bound a short search for `∫ v·du`: an integral
        // at hand only (see `MAX_BY_PARTS_POLY_DEGREE`).
        let rest_depth = if polynomial_beyond_by_parts(arena, u, dv, var, var_sym) {
            (depth - 1).min(BEYOND_BOUND_DEPTH)
        } else {
            depth - 1
        };
        let integral_v_du = integrate_node(arena, v_du, var, var_sym, rest_depth);

        // Check if the remaining integral was resolved (deep check:
        // an Add of unevaluated Integrals should not be accepted).
        if crate::base::walk::has_unevaluated(arena, integral_v_du) {
            tracing::trace!("by-parts: ∫v·du has unevaluated nodes, skipping this ordering");
            continue; // Remaining integral not solvable
        }

        // Success: ∫ u·dv = u·v - ∫ v·du
        tracing::debug!("integration by parts succeeded");
        let u_v = arena.mul(&[u, v]);
        return Some(arena.sub(u_v, integral_v_du));
    }
    None
}

/// `∫ P₁(x)·P₂(x)·…·g(x) dx` with two or more polynomial factors: their
/// product multiplied out, and the integrand with one polynomial factor
/// retried.  `∫ x·(1 + a²x²)·atan(a·x) dx` (even `∫ x·(1 + x²)·atan x dx`)
/// stayed unevaluated: the three-factor rule takes `u = x`, while by parts
/// needs `dv = x·(1 + a²x²)` (0.31, the complex-parameter hunt).
pub(super) fn try_merge_polynomial_factors(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let (polys, mut rest): (Vec<ExprId>, Vec<ExprId>) = dependent
        .iter()
        .partition(|&&d| is_polynomial_in(arena, d, var, var_sym));
    if polys.len() < 2 || rest.is_empty() {
        return None;
    }
    let p = arena.mul(&polys);
    let p = crate::transforms::expand::expand(arena, p);
    rest.push(p);
    let merged = arena.mul(&rest);
    let r = integrate_node(arena, merged, var, var_sym, depth - 1);
    (!crate::base::walk::has_unevaluated(arena, r)).then_some(r)
}

/// `∫ (s₁ + … + s_k)·g dx = Σ ∫ sᵢ·g dx` for a product with a sum factor
/// (or a square to fourth power of a non-polynomial sum, multiplied out),
/// when every piece has a closed form (the first piece without one ends
/// the attempt).  A sum that is not a polynomial in the variable is taken
/// first: polynomial factors are the `u` of by parts.  Linearity only split
/// sums at the top: `∫ (sin t + cos t)·eᵗ dt` stayed unevaluated (by parts
/// wants a polynomial or a logarithm for `u`, and the cyclic rule does not
/// see `(−cos t − sin t)·eᵗ` as `−1` times the integrand) while
/// `∫ (eᵗ·sin t + eᵗ·cos t) dt` worked (0.31, normal-form hunt).
pub(super) fn try_distribute_sum_factor(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if dependent.len() < 2 || depth < 2 {
        return None;
    }
    // A radical of the variable (`√(sec x)`, `(a + b·x)^m`) makes every
    // piece as hard as the product, whose substitutions see it whole: on
    // the Rubi suite such attempts cost 30 s and found almost nothing.
    let radical = dependent.iter().any(|&d| {
        crate::base::walk::post_order_ids(arena, d)
            .iter()
            .any(|&id| {
                matches!(arena.node(id), ExprNode::Pow(b, e)
                if !arena.as_num(*e).is_some_and(|q| q.is_integer())
                    && contains_var(arena, *b, var_sym))
            })
    });
    if radical {
        return None;
    }
    // A rational function over ℚ is the rational integrator's (Hermite and
    // Rothstein–Trager are complete there); its pieces only repeat that
    // work — inside the algebraic-remainder fallback of
    // `try_risch_rational`, five nested splittings made `∫ sinh x·tanh 4x`
    // twenty times slower.
    let product = arena.mul(dependent);
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, product);
    if crate::poly::polybridge::expr_to_poly(arena, numer, var).is_some()
        && crate::poly::polybridge::expr_to_poly(arena, denom, var).is_some()
    {
        return None;
    }
    let is_sum = |arena: &Arena, d: ExprId| matches!(arena.node(d), ExprNode::Add(_));
    // A small power of a sum that is not a polynomial, `(a + b·sin u)³`,
    // multiplied out is a sum as well.
    let is_power_of_sum = |arena: &Arena, d: ExprId| {
        matches!(arena.node(d), ExprNode::Pow(b, n)
        if is_sum(arena, *b)
            && !is_polynomial_in(arena, *b, var, var_sym)
            && arena.as_num(*n).is_some_and(|q| {
                q.is_integer() && *q >= Q::from_integer(2.into()) && *q <= Q::from_integer(4.into())
            }))
    };
    let idx = dependent
        .iter()
        .position(|&d| is_sum(arena, d) && !is_polynomial_in(arena, d, var, var_sym))
        .or_else(|| dependent.iter().position(|&d| is_sum(arena, d)))
        .or_else(|| dependent.iter().position(|&d| is_power_of_sum(arena, d)))?;
    let sum = if is_sum(arena, dependent[idx]) {
        dependent[idx]
    } else {
        crate::transforms::expand::expand(arena, dependent[idx])
    };
    let ExprNode::Add(terms) = arena.node(sum).clone() else {
        return None;
    };
    let rest = remaining_product(arena, dependent, idx);
    let mut parts: SmallVec<[ExprId; 6]> = SmallVec::new();
    for &term in terms.iter() {
        let piece = arena.mul(&[term, rest]);
        let big_f = integrate_node(arena, piece, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, big_f) {
            return None;
        }
        parts.push(big_f);
    }
    Some(arena.add(&parts))
}

/// Rewrite products of `sin`/`cos` factors via product-to-sum identities
/// and retry (e.g. `x·sin x·cos x = x·sin(2x)/2`).
pub(super) fn try_trig_product_to_sum(
    arena: &mut Arena,
    expr: ExprId,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let trig_count = dependent
        .iter()
        .filter(|&&d| matches!(arena.node(d), ExprNode::Sin(_) | ExprNode::Cos(_)))
        .count();
    if trig_count < 2 {
        return None;
    }
    let combined = arena.trig_combine_expr(expr);
    let combined = crate::transforms::eval::eval(arena, combined);
    if combined == expr {
        return None;
    }
    let combined = crate::transforms::expand::expand(arena, combined);
    let result = integrate_node(arena, combined, var, var_sym, depth - 1);
    if crate::base::walk::has_unevaluated(arena, result) {
        return None;
    }
    Some(result)
}
