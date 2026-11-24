//! mul vanishing (part of [`canon`](super)).

use super::*;

/// The sums whose vanishing identically makes `f` vanish identically:
/// `f` itself if it is a sum, those of the factors of a product, of the
/// base of a positive power, and of the argument `a` of `g(a)` for a `g`
/// with `g(0) = 0` (`sin`, `tan`, `sinh`, `tanh`, their inverses, `|·|`,
/// `sign`, `W`, `floor`, `ceiling`).  An explicit walk down those nodes.
/// `expand` and `ratsimp` test them by an identity beside a pole
/// (`√(tan x·cos x − sin x)` is 0 because `tan x·cos x − sin x` is): there
/// a factor that is undefined rather than 0 (a product with an infinite
/// factor inside) makes the product `nan` all the same.
pub(crate) fn zero_candidate_sums(arena: &Arena, f: ExprId) -> SmallVec<[ExprId; 4]> {
    let mut sums: SmallVec<[ExprId; 4]> = SmallVec::new();
    match arena.node(f) {
        ExprNode::Add(_) => {
            sums.push(f);
            return sums;
        }
        node if node.is_atom() => return sums,
        _ => {}
    }
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: SmallVec<[ExprId; 8]> = smallvec![f];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Add(_) => sums.push(id),
            ExprNode::Mul(children) => stack.extend_from_slice(children),
            ExprNode::Neg(inner) => stack.push(*inner),
            ExprNode::Pow(base, e) if arena.as_num(*e).is_some_and(Signed::is_positive) => {
                stack.push(*base);
            }
            node => stack.extend(node.unary_arg_if(zero_through)),
        }
    }
    sums
}

/// The functions [`not_finite_at_infinity`] examines at an infinite
/// argument: the elementary functions but `atan`, `atanh` (bounded), and
/// `|·|`, `⌊·⌋`, `⌈·⌉`, `erf`, `erfc`, `Γ`, `ln Γ`, `ψ`, `ζ`, `n!`.
pub(super) fn examined_at_infinity(f: crate::base::func::Func) -> bool {
    use crate::base::func::Func;
    (f.elementary().is_some() && !matches!(f, Func::Atan | Func::Atanh))
        || matches!(
            f,
            Func::Abs
                | Func::Floor
                | Func::Ceiling
                | Func::Erf
                | Func::Erfc
                | Func::Gamma
                | Func::LogGamma
                | Func::Digamma
                | Func::Zeta
                | Func::Factorial
        )
}

/// The one-argument functions a zero of the argument passes through
/// (`f(0) = 0`) for [`zero_candidate_sums`] and [`sum_under_zero_chain`]:
/// the odd elementary functions (`sin`, `tan`, `asin`, `atan` and their
/// hyperbolic kin) and `|·|`, `sign`, `W`, `⌊·⌋`, `⌈·⌉`.
pub(super) fn zero_through(f: crate::base::func::Func) -> bool {
    use crate::base::func::{Func, Parity};
    (f.elementary().is_some() && matches!(f.spec().parity, Parity::Odd(_)))
        || matches!(
            f,
            Func::Abs | Func::Sign | Func::LambertW0 | Func::Floor | Func::Ceiling
        )
}

/// `false` when `f` certainly does not vanish identically by one of its
/// [`zero_candidate_sums`] (the residue test, no expansion).
pub(crate) fn may_vanish_identically_as_factor(arena: &Arena, f: ExprId) -> bool {
    zero_candidate_sums(arena, f)
        .into_iter()
        .any(|s| arena.may_vanish_identically(s))
}

/// Does `f` vanish identically because one of its [`zero_candidate_sums`]
/// does ([`sum_vanishes_identically`])?  `x·(x + 1) − x² − x` does, so do
/// `sin(−x/(x + 1) + x·(−x/(x + 1) + 1))` and `√((x + 2)·((x + y)² − x²
/// − 2xy − y²))`; `cos(x·(x + 1) − x² − x)` and `sin²x + cos²x − 1` (not
/// 0 as a rational function of `sin x`, `cos x`) do not.
pub(crate) fn vanishes_identically_as_factor(arena: &mut Arena, f: ExprId) -> bool {
    zero_candidate_sums(arena, f)
        .into_iter()
        .any(|s| sum_vanishes_identically(arena, s))
}

/// Does a base of the product meet a positive and a negative power of
/// itself (`s·s⁻¹`, `s²·s⁻³`), while it vanishes identically
/// ([`factor_vanishes`])?  Then the product is `0·zoo =
/// nan` at every point, not the cancelled `s⁰ = 1`: up to 0.32
/// `(x·(x + 1) − x² − x)/(x·(x + 1) − x² − x)` was `1` (SymPy 1.14 too;
/// symplex's `together`, `ratsimp` and `simplify` give `nan` for `0/0`),
/// and up to 0.33 so were `(exp(2x) − exp(x)²)/(exp(2x) − exp(x)²)` (SymPy:
/// `nan`, it folds `exp(x)**2`) and `(sin² 1 + cos² 1 − 1)/(sin² 1 + cos² 1 − 1)`.
/// A base that vanishes by an identity of its functions
/// ([`vanishes_by_function_identity`]) too: up to 0.52 `(sin²x + cos²x −
/// 1)·(y + 1)/(sin²x + cos²x − 1)` was `y + 1` in every operation.
/// Only a base whose exponents have both signs is tested (a symbol at no
/// cost: it holds no sum; one declared zero is 0, so `z·z⁻¹` is `nan`).
pub(super) fn cancels_vanishing_sum(
    arena: &mut Arena,
    bases: &FxHashMap<ExprId, SmallVec<[ExprId; 4]>>,
) -> bool {
    let mixed: SmallVec<[ExprId; 2]> = bases
        .iter()
        .filter(|&(&b, exps)| {
            exps.len() > 1
                && (!arena.node(b).is_atom() || pinned_zero_symbol(arena, b))
                && exps
                    .iter()
                    .any(|&e| arena.as_num(e).is_some_and(Signed::is_positive))
                && exps
                    .iter()
                    .any(|&e| arena.as_num(e).is_some_and(Signed::is_negative))
        })
        .map(|(&b, _)| b)
        .collect();
    mixed
        .into_iter()
        .any(|b| factor_vanishes(arena, b) || vanishes_by_function_identity(arena, b))
}

/// Does `d` hold a function application or another non-rational generator
/// (`sin x`, `tan x`, `eˣ`, `√x`) in its rational skeleton, so that it could
/// vanish by an identity of its functions (`tan x·cos x − sin x`) that the
/// residue test does not see?  Not an atom, and not an expression in
/// numbers and symbols only (there the residue test is exact).
pub(super) fn may_vanish_by_identity(arena: &Arena, d: ExprId) -> bool {
    if arena.node(d).is_atom() {
        return false;
    }
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: SmallVec<[ExprId; 16]> = smallvec![d];
    let mut function = false;
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN => return false,
            ExprNode::Num(_) | ExprNode::Symbol(_) => {}
            ExprNode::Add(children) | ExprNode::Mul(children) => stack.extend_from_slice(children),
            ExprNode::Neg(inner) => stack.push(*inner),
            ExprNode::Pow(base, e) if arena.as_num(*e).is_some_and(|q| q.is_integer()) => {
                stack.push(*base);
            }
            _ => function = true,
        }
    }
    function
}

/// The denominators of the `0/0` candidates among the products `nodes`:
/// the bases `d` of negative powers that could vanish by an identity of
/// their functions ([`may_vanish_by_identity`]: `tan x·cos x − sin x`), in a
/// product with another factor that may vanish identically by the residue
/// test ([`may_vanish_identically_as_factor`]: `(x + 1)² − x² − 2x − 1`).
/// Only those denominators are worth the numeric identity test
/// ([`Arena::vanishes_by_identity`]) where it is not affordable for every
/// denominator (`expand`, `trigsimp`): multiplying out the numerator would
/// give `0·d⁻¹ = 0` where the value is `0/0`.  Residue tests only.
pub(crate) fn identity_denominator_candidates(
    arena: &Arena,
    nodes: &[ExprId],
) -> SmallVec<[ExprId; 2]> {
    let negative_base = |c: ExprId| match arena.node(c) {
        ExprNode::Pow(base, e) if arena.as_num(*e).is_some_and(Signed::is_negative) => Some(*base),
        _ => None,
    };
    let mut transcendental: FxHashMap<ExprId, bool> = FxHashMap::default();
    let mut may_vanish: FxHashMap<ExprId, bool> = FxHashMap::default();
    let mut out: SmallVec<[ExprId; 2]> = SmallVec::new();
    for &id in nodes {
        let ExprNode::Mul(children) = arena.node(id) else {
            continue;
        };
        let mut denominators: SmallVec<[ExprId; 2]> = SmallVec::new();
        for &c in children {
            if let Some(b) = negative_base(c)
                && *transcendental
                    .entry(b)
                    .or_insert_with(|| may_vanish_by_identity(arena, b))
            {
                denominators.push(b);
            }
        }
        if denominators.is_empty() {
            continue;
        }
        let zero_candidate = children.iter().any(|&c| {
            negative_base(c).is_none()
                && *may_vanish
                    .entry(c)
                    .or_insert_with(|| may_vanish_identically_as_factor(arena, c))
        });
        if !zero_candidate {
            continue;
        }
        for d in denominators {
            if !out.contains(&d) {
                out.push(d);
            }
        }
    }
    out
}

/// The nodes among `order` (a post-order of an expression) that are `zoo`
/// (or `nan`) at every point because a negative power of a base that
/// vanishes (`zero`, decided by the caller) is in them outside every other
/// negative power: that power, and the sums, products, negations and
/// positive numeric powers above it (`y/s − 2` and its square for `s`
/// zero; not `1/(y/s − 2)`).  A factor beside one of them that vanishes
/// makes its product `0·zoo = nan`; `ratsimp` and `expand` test the
/// factors beside these, not only beside the powers themselves.
/// Structural; one pass over `order`.
pub(crate) fn nodes_over_vanishing_bases(
    arena: &Arena,
    order: &[ExprId],
    zero: impl Fn(ExprId) -> bool,
) -> FxHashSet<ExprId> {
    let mut infinite: FxHashSet<ExprId> = FxHashSet::default();
    for &id in order {
        let is_infinite = match arena.node(id) {
            ExprNode::Add(ch) | ExprNode::Mul(ch) => ch.iter().any(|c| infinite.contains(c)),
            ExprNode::Neg(c) => infinite.contains(c),
            ExprNode::Pow(b, e) => match arena.as_num(*e) {
                Some(q) if q.is_negative() => zero(*b),
                Some(_) => infinite.contains(b),
                None => false,
            },
            _ => false,
        };
        if is_infinite {
            infinite.insert(id);
        }
    }
    infinite
}

/// The `zoo + zoo` candidates among the sums `nodes`: in a sum with exactly
/// one term over a denominator that vanishes (`vanishes`, decided by the
/// caller), the denominators of the other terms that could vanish by an
/// identity of their functions ([`may_vanish_by_identity`]): if one does,
/// the sum is `zoo + zoo = nan`, not the `zoo` of the first term (whose
/// reciprocal is a `0` that hides the `nan`).  `((x + 6)²/((x + y)² − x² −
/// 2xy − y²) + (cos x + 2)/(sin 2x − 2·sin x·cos x))`.  Structural tests
/// only; empty unless a denominator vanishes.
pub(crate) fn sum_partner_denominators(
    arena: &Arena,
    nodes: &[ExprId],
    vanishes: impl Fn(ExprId) -> bool,
) -> SmallVec<[ExprId; 2]> {
    let denominators = |t: ExprId| -> SmallVec<[ExprId; 2]> {
        let factors: &[ExprId] = match arena.node(t) {
            ExprNode::Mul(children) => children,
            _ => std::slice::from_ref(&t),
        };
        factors
            .iter()
            .filter_map(|&f| match arena.node(f) {
                ExprNode::Pow(base, e) if arena.as_num(*e).is_some_and(Signed::is_negative) => {
                    Some(*base)
                }
                _ => None,
            })
            .collect()
    };
    let mut out: SmallVec<[ExprId; 2]> = SmallVec::new();
    for &id in nodes {
        let ExprNode::Add(terms) = arena.node(id) else {
            continue;
        };
        let per_term: SmallVec<[SmallVec<[ExprId; 2]>; 6]> =
            terms.iter().map(|&t| denominators(t)).collect();
        let infinite: SmallVec<[bool; 6]> = per_term
            .iter()
            .map(|ds| ds.iter().any(|&d| vanishes(d)))
            .collect();
        if infinite.iter().filter(|&&b| b).count() != 1 {
            continue;
        }
        for (ds, &inf) in per_term.iter().zip(&infinite) {
            if inf {
                continue;
            }
            for &d in ds {
                if !out.contains(&d) && may_vanish_by_identity(arena, d) {
                    out.push(d);
                }
            }
        }
    }
    out
}

/// Does the factor `f` vanish identically: by one of its sums as a
/// rational function of its generators ([`vanishes_identically_as_factor`],
/// which relates `exp(2x)` and `exp(x)²`), or, for a constant, exactly
/// ([`constant_vanishes`]), itself or by the constant sum under its
/// positive powers and functions with `g(0) = 0` ([`sum_under_zero_chain`]:
/// `√(tan 2·cos 2 − sin 2)`, which the numeric test does not confirm as a
/// whole)?  The test of the rare paths of the canonical constructors
/// (`0·d⁻¹`, `zoo·f`, `s·s⁻¹`, like terms over `d`).
///
/// A symbol whose assumptions pin it to a value is that value
/// ([`with_pinned_values`]): `n − 2` and `n! − 2` vanish for an even prime
/// `n`, and a symbol declared zero vanishes.
pub(crate) fn factor_vanishes(arena: &mut Arena, f: ExprId) -> bool {
    let f = with_pinned_values(arena, f).unwrap_or(f);
    arena.is_zero_structural(f)
        || vanishes_identically_as_factor(arena, f)
        || constant_vanishes(arena, f)
        || sum_under_zero_chain(arena, f).is_some_and(|s| s != f && constant_vanishes(arena, s))
}

/// `f` with every free symbol whose assumptions pin it to one value
/// ([`Assumptions::pinned_value`](crate::base::assumptions::Assumptions::pinned_value):
/// `0` for `zero`, `2` for `even ∧ prime`, `1` for a positive integer
/// neither prime nor composite) replaced by that value, rebuilt by the
/// canonical constructors: what the zero tests of the rare paths
/// ([`factor_vanishes`], [`vanishes_by_function_identity`]) and
/// `identically_zero::vanishes_by_identity` decide, since the residue test
/// takes a symbol for an indeterminate.  Up to 0.53 an even prime `n` was
/// not known to be 2: `(⌈n⌉ − n)/(n − 2)` was 0 and `(x + 3)(n² − 4)/((n² −
/// 4)(y + 1))` was `(x + 3)/(y + 1)` (both `nan`; SymPy 1.14 gives 0 and
/// the quotient too).  `None` when `f` has no such symbol, or a binder
/// (whose bound variable the rebuild would replace too).
pub(crate) fn with_pinned_values(arena: &mut Arena, f: ExprId) -> Option<ExprId> {
    if arena.free_symbol_mask(f) == 0 {
        return None;
    }
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for s in crate::base::walk::free_symbols(arena, f) {
        if let ExprNode::Symbol(sid) = *arena.node(s)
            && let Some(v) = arena.symbol_assumptions(sid).pinned_value()
        {
            let value = arena.int(v);
            cache.insert(s, value);
        }
    }
    if cache.is_empty() || arena.facts(f).has_binder() {
        return None;
    }
    for id in crate::base::walk::post_order_ids(arena, f) {
        if cache.contains_key(&id) {
            continue;
        }
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        if rebuilt != id {
            cache.insert(id, rebuilt);
        }
    }
    cache.get(&f).copied()
}

/// Is `id` an atom that may vanish: a symbol pinned to the value 0
/// ([`with_pinned_values`])?  The rare paths skip every other atom.
pub(super) fn pinned_zero_symbol(arena: &Arena, id: ExprId) -> bool {
    matches!(*arena.node(id), ExprNode::Symbol(sid)
        if arena.symbol_assumptions(sid).pinned_value() == Some(0))
}

/// The sum `s` that `f` is a positive power of, or a function with `g(0) =
/// 0` of (`sin`, `|·|`, … as in [`zero_candidate_sums`]), possibly nested:
/// `f` vanishes with `s`.  Not through a product, whose other factors may
/// be infinite where `s` is 0 (`sin(s·w)` with `w = 1/(s/s′ + 2)` is
/// `sin(0·zoo)`, `nan`).  `None` when there is no such sum.
fn sum_under_zero_chain(arena: &Arena, f: ExprId) -> Option<ExprId> {
    let mut id = f;
    // A chain, not a tree: one step per node of `f`.
    loop {
        id = match arena.node(id) {
            ExprNode::Add(_) => return Some(id),
            ExprNode::Neg(inner) => *inner,
            ExprNode::Pow(base, e) if arena.as_num(*e).is_some_and(Signed::is_positive) => *base,
            node => node.unary_arg_if(zero_through)?,
        };
    }
}

/// Is `f` a constant: no free symbol, no infinity or `nan`, no unevaluated
/// node?  An explicit walk that stops at the first symbol.
pub(super) fn is_numeric_constant(arena: &Arena, f: ExprId) -> bool {
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: SmallVec<[ExprId; 16]> = smallvec![f];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Symbol(_)
            | ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN => return false,
            node => node.for_each_child(|c| stack.push(c)),
        }
    }
    !crate::base::walk::has_unevaluated(arena, f)
}

/// Is the constant `f` (no free symbol: `sin² 1 + cos² 1 − 1`, `e² −
/// exp(1)²`, `(1 + √2)² − 3 − 2√2`) exactly 0?  A constant is decided
/// numerically: not when the assumptions know it nonzero (`1 + √2`, `π`)
/// or a certified evaluation is nonzero (almost every constant, at the cost
/// of one evaluation); 0 when it is zero to that precision **and** exactly
/// 0 by the exact tests of `simplify` (algebraic numbers, rational
/// functions of exponentials; see
/// [`Arena::vanishes_by_identity`]).  Up to 0.33 `0/(sin² 1 + cos² 1 −
/// 1)` was `0` (SymPy 1.14 too) — so `eval` of `(e² − exp(1)²)/(sin² 1 +
/// cos² 1 − 1)`, whose numerator evaluates to 0, was `0`, and so was
/// `subs` of a point into `((x + 1)² − x² − 2x − 1)/(sin² x + cos² x −
/// 1)`; both are `nan`.  Only on the rare paths of [`factor_vanishes`], not
/// for an atom, and not inside a confirmation ([`vanishing_check_active`]).
///
/// A constant too costly to evaluate ([`costly_to_evaluate`]) is 0 only
/// by an identity that holds whatever the values of its costly parts,
/// which are never evaluated: the identity test takes them for symbols
/// ([`with_costly_parts_as_symbols`]), so `sin²u + cos²u − 1` is 0 for
/// `u = besselj(10⁵, 1)`.  Up to 0.34 such a constant was never taken for
/// 0, and `(sin²u + cos²u − 1)·zoo` was `zoo` (it is `0·zoo = nan`).  It
/// is tested only when the residue test does not rule it out
/// ([`Arena::may_vanish_identically`], which relates `sin u`, `cos u`, `tan
/// u` whatever `u` is), so a costly constant that is not 0 costs no more
/// than before.
pub(crate) fn constant_vanishes(arena: &mut Arena, f: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    if arena.node(f).is_atom() || vanishing_check_active() || !is_numeric_constant(arena, f) {
        return false;
    }
    if nonzero_by_form(arena, f) {
        return false;
    }
    if AssumptionCache::new().query(arena, f, Props::NONZERO) == Some(true) {
        return false;
    }
    if costly_to_evaluate(arena, f) && !arena.may_vanish_identically(f) {
        return false;
    }
    arena.vanishes_by_identity(f)
}

/// Is the constant `f` nonzero by its form alone: `Γ(z)` and `z! = Γ(z +
/// 1)` (`Γ` has no zeros), `eᶻ`, and a power `bᶻ` of such a constant or of
/// a nonzero number (`e^{z·ln b}`), whatever the (finite) `z`?  The numeric
/// test of [`constant_vanishes`] evaluated them: `((i!)!)!…` times `zoo`
/// (38 nested factorials of complex numbers, a parser fuzz input) took 0.15
/// s at rising precision once complex `Γ` evaluated (0.44), `33^{(i!)!…}`
/// 0.05 s (release builds); 0.43 refused the evaluation at once.  A chain of
/// bases, one step per node.
fn nonzero_by_form(arena: &Arena, f: ExprId) -> bool {
    let mut id = f;
    loop {
        match arena.node(id) {
            ExprNode::Gamma(_) | ExprNode::Factorial(_) | ExprNode::Exp(_) => return true,
            ExprNode::Pow(base, _) => match arena.as_num(*base) {
                Some(q) => return !q.is_zero(),
                None => id = *base,
            },
            _ => return false,
        }
    }
}

/// `f` with each of its costly parts — the outermost subexpressions that
/// [`is_costly_node`] (`besselj(10⁵, 1)`, `jacobi(999, …)`) — replaced by a
/// symbol of its own (`__costly_constant_k`), so that a zero test of the
/// result never evaluates them: an identity of the result holds for every
/// value of those parts, so at theirs — unless their value is a
/// singularity of a term (`tan u·cos u − sin u` at `cos u = 0`), where `f`
/// is undefined and the rare paths that ask give `nan` either way.  The
/// zero tests of `simplify` ([`Arena::vanishes_by_identity`]) and of the
/// canonical constructors ([`sum_vanishes_identically`]) use it.  `None`
/// when `f` has no costly part (or one is left).  One walk.
pub(crate) fn with_costly_parts_as_symbols(arena: &mut Arena, f: ExprId) -> Option<ExprId> {
    let mut costly: SmallVec<[ExprId; 2]> = SmallVec::new();
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: SmallVec<[ExprId; 16]> = smallvec![f];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if is_costly_node(arena, id) {
            costly.push(id);
        } else {
            arena.node(id).for_each_child(|c| stack.push(c));
        }
    }
    if costly.is_empty() {
        return None;
    }
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for (k, &c) in costly.iter().enumerate() {
        let symbol = arena.symbol(&format!("__costly_constant_{k}"));
        cache.insert(c, symbol);
    }
    for id in crate::base::walk::post_order_ids(arena, f) {
        if cache.contains_key(&id) || arena.node(id).is_atom() {
            continue;
        }
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        if rebuilt != id {
            cache.insert(id, rebuilt);
        }
    }
    let general = cache.get(&f).copied().unwrap_or(f);
    (general != f && !costly_to_evaluate(arena, general)).then_some(general)
}

/// The magnitude of a numeric parameter from which [`costly_to_evaluate`]
/// declines the numeric zero test (`besselj(10⁸, x)` took 5 s before 0.34).
const COSTLY_PARAMETER: i64 = 100_000;

/// The degree of an orthogonal polynomial from which [`costly_to_evaluate`]
/// declines the zero test: the exact test of its confirmation computes the
/// polynomial's rational value (`gegenbauer(999, 1/3, 1/3)` 0.75 s,
/// `jacobi(999, 1/3, 1/5, 1/7)` 2.8 s; at degree 199 at most 0.16 s).
const COSTLY_DEGREE: i64 = 200;

/// Is the numeric zero test of the constant `f` ([`constant_vanishes`])
/// beyond the budget of a canonical constructor, which runs on every node
/// built: does `f` apply a special function (`Γ`, `ψ⁽ⁿ⁾`, `ζ`, Bessel, the
/// incomplete `Γ`, … — anything but arithmetic, powers and the elementary
/// functions) to a number of magnitude [`COSTLY_PARAMETER`] or more, or an
/// orthogonal polynomial of degree [`COSTLY_DEGREE`] or more?  The cost of
/// evaluating some of those grows with their orders; such a constant is not
/// tested (not taken for 0, as before 0.34).  Up to 0.34-dev building
/// `cos(besselj(8345185991999992, 61/10²⁷))/sin(…)` never finished: the
/// test evaluated the Bessel function, a multiplication per unit of its
/// order (`evalf` takes it at once now; this bounds the next such kernel).
/// An explicit walk.
pub(crate) fn costly_to_evaluate(arena: &Arena, f: ExprId) -> bool {
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: SmallVec<[ExprId; 16]> = smallvec![f];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if is_costly_node(arena, id) {
            return true;
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }
    false
}

/// Is the node `id` itself what makes [`costly_to_evaluate`] true: a
/// special function with a numeric argument of magnitude
/// [`COSTLY_PARAMETER`] or more, or an orthogonal polynomial of degree
/// [`COSTLY_DEGREE`] or more?
fn is_costly_node(arena: &Arena, id: ExprId) -> bool {
    use crate::base::libfn::LibFn;
    let node = arena.node(id);
    let special = match node {
        ExprNode::Apply(..) | ExprNode::Lib(..) | ExprNode::Internal(..) => true,
        _ => node.func(arena).is_some_and(|(f, _)| f.is_special()),
    };
    if !special {
        return false;
    }
    if let ExprNode::Lib(
        LibFn::Legendre
        | LibFn::ChebyshevT
        | LibFn::ChebyshevU
        | LibFn::Hermite
        | LibFn::Laguerre
        | LibFn::Gegenbauer
        | LibFn::Jacobi
        | LibFn::AssocLegendre
        | LibFn::AssocLaguerre,
        args,
    ) = node
    {
        let degree_bound = Q::from_integer(BigInt::from(COSTLY_DEGREE));
        if args
            .first()
            .and_then(|&n| arena.as_num(n))
            .is_some_and(|q| q.abs() >= degree_bound)
        {
            return true;
        }
    }
    let bound = Q::from_integer(BigInt::from(COSTLY_PARAMETER));
    let mut costly = false;
    node.for_each_child(|c| costly |= arena.as_num(c).is_some_and(|q| q.abs() >= bound));
    costly
}

thread_local! {
    /// Set while [`sum_vanishes_identically`] confirms a zero by multiplying
    /// out: the products built meanwhile are not tested again, and `expand`
    /// does not look for vanishing denominators
    /// ([`vanishing_check_active`]).
    static IN_VANISHING_CHECK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Is [`sum_vanishes_identically`] confirming a zero on this thread?
pub(crate) fn vanishing_check_active() -> bool {
    IN_VANISHING_CHECK
        .try_with(std::cell::Cell::get)
        .unwrap_or(true)
}

/// Clears [`IN_VANISHING_CHECK`] when dropped (also on unwinding).
pub(super) struct VanishingCheckGuard;

impl VanishingCheckGuard {
    /// `None` when a check is already running on this thread.
    pub(super) fn enter() -> Option<Self> {
        IN_VANISHING_CHECK
            .try_with(|active| !active.replace(true))
            .unwrap_or(false)
            .then_some(VanishingCheckGuard)
    }
}

impl Drop for VanishingCheckGuard {
    fn drop(&mut self) {
        let _ = IN_VANISHING_CHECK.try_with(|active| active.set(false));
    }
}

/// Is the sum `s` identically 0 as a rational function of its generators
/// (`x·(x + 1) − x² − x`, `x/(x + 1) + 1/(x + 1) − 1`)?  The residue at a
/// pseudo-random point ([`Arena::may_vanish_identically`], one pass, no
/// expansion) rejects almost every sum; only a residue 0 is confirmed, by
/// the numerator of `s` as one fraction being 0 once multiplied out
/// ([`Arena::vanishes_identically`]).  `false` for anything but a sum,
/// inside another such confirmation, and for a sum with a denominator that
/// may vanish identically itself ([`Arena::may_have_vanishing_denominator`]):
/// `1/((x + y)² − x² − 2xy − y²) + 2` is `zoo + 2 = zoo`, not 0.
///
/// The confirmation evaluates (`eval` of the numerator multiplied out),
/// which computes an orthogonal polynomial of high degree exactly: a sum
/// with a costly part ([`costly_to_evaluate`]) is confirmed with its
/// costly parts as symbols ([`with_costly_parts_as_symbols`]).  Up to
/// 0.34 the residue test rejected `sin²u + cos²u − 1` before any
/// confirmation; it relates `sin` and `cos` now, and building `(sin²u +
/// cos²u − 1)·zoo` for `u = jacobi(999, 1/3, 1/5, 1/7)` would have computed
/// `u` (seconds).
pub(crate) fn sum_vanishes_identically(arena: &mut Arena, s: ExprId) -> bool {
    if !matches!(arena.node(s), ExprNode::Add(_))
        || !arena.may_vanish_identically(s)
        || arena.may_have_vanishing_denominator(s)
    {
        return false;
    }
    let Some(_guard) = VanishingCheckGuard::enter() else {
        return false;
    };
    let s = if costly_to_evaluate(arena, s) {
        match with_costly_parts_as_symbols(arena, s) {
            Some(general) => general,
            None => return false,
        }
    } else {
        s
    };
    let (numerator, _) = arena.as_numer_denom_expr(s);
    arena.vanishes_identically(numerator)
}
