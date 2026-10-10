//! Rational-function normal form (`ratsimp`).
//!
//! An expression built from numbers, `Add`, `Mul`, `Neg` and integer
//! powers over a set of *generators* — the free symbols together with every
//! maximal non-rational subexpression (function applications, constants
//! such as π or e, powers with non-integer exponents, …) — is a rational
//! function `P/Q` with `P, Q ∈ ℚ[generators]`.  [`ratsimp`] computes that
//! single fraction, cancels `gcd(P, Q)` with the heuristic multivariate GCD,
//! clears denominators so that both parts have integer coefficients with no
//! common integer factor, makes the leading coefficient of `Q` positive, and
//! rebuilds `P / Q`.
//!
//! This mirrors what SymPy's `cancel` does: opaque subexpressions are
//! treated as independent indeterminates, so `sin(x)² / sin(x)` becomes
//! `sin(x)` but no trigonometric identity is applied.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::base::arena::Arena;
use crate::base::canon::Everywhere;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;
use crate::poly::multipoly::{GrevLex, Lex, MonomialOrd, MultiPoly};
use crate::poly::polybridge::multipoly_to_expr;

type RatPoly = MultiPoly<GrevLex>;

/// Largest number of terms any intermediate polynomial may reach before the
/// conversion gives up and the input is returned unchanged.
const MAX_TERMS: usize = 20_000;

/// Largest total degree any intermediate polynomial may reach.  Keeps
/// exponent arithmetic far away from `u32` overflow.
const MAX_DEGREE: u32 = 1 << 20;

/// Rational-function normal form of `expr` (see the module docs).
///
/// Returns `expr` unchanged when it contains `±∞`, `zoo`, `NaN` or an
/// unevaluated node, or when an intermediate polynomial exceeds the size
/// budget.  A negative power of a subexpression that is identically zero
/// (a zero polynomial in the generators) is a fraction with denominator
/// `0`, carried through the arithmetic like any other: the result is
/// `zoo` for `P/0` with `P ≠ 0` and `nan` for `0/0`, as SymPy's `ratsimp`
/// and [`together`](crate::poly::polybridge::together) give.  Up to 0.31
/// `ratsimp(1/(x·(−x/(x + 1) + x·(−x/(x + 1) + 1))))` returned its
/// input while `together` gave `zoo`.
///
/// A generator is 0 when its argument is (`√(x·(x + 1) − x² − x)`): `P/0`
/// is `nan` when the numerator is 0 once multiplied out with its function
/// arguments ([`numerator_expands_to_zero`]), as `together` finds.  Up to
/// 0.31 `ratsimp(√(x·(x + 1) − x² − x)/(x·(x + 2) − x² − 2x))` was `zoo`.
///
/// A denominator that vanishes only as a function of its generators —
/// `exp(2) − exp(1)²`, `exp(2x) − exp(x)²` (the generators `exp(2x)` and
/// `exp(x)` are not independent), a constant `sin² 1 + cos² 1 − 1` — is 0
/// too ([`vanishing_bases`]).  Up to 0.33 `ratsimp(((x + 1)² − x² − 2x −
/// 1)/(exp(2) − exp(1)²))` was `0`, and `ratsimp((xy − 2)²/((exp(x/2)² −
/// exp(x))/(exp(2) − exp(1)²) + (y + 1)/(exp(2) − exp(1)²)))` multiplied
/// the zero `exp(2) − exp(1)²` into the numerator (a value 0 at every
/// point); both are `nan`.
pub(crate) fn ratsimp(arena: &mut Arena, expr: ExprId) -> ExprId {
    if !is_admissible(arena, expr) {
        return expr;
    }
    let vanishing = vanishing_bases(arena, expr, &FxHashMap::default());
    let gens = collect_generators(arena, expr);
    let Some((p, q)) = to_rational_function_tracked(arena, expr, &gens, &mut false, &vanishing)
    else {
        return expr;
    };
    let out = rebuild(arena, &p, &q, &gens).unwrap_or(expr);
    if out == arena.complex_infinity && numerator_expands_to_zero(arena, expr) {
        return arena.nan;
    }
    out
}

/// The bases of the negative powers in the rational skeleton of `expr`
/// (outside every generator) that vanish identically
/// ([`factor_vanishes`](crate::base::canon::factor_vanishes): as rational
/// functions of their generators with exponentials and radicals related,
/// `exp(2x) − exp(x)²`, or exactly for a constant), as [`Vanishing`] values
/// for [`to_rational_function_tracked`]: the base `0`, its negative powers
/// the fraction `1/0`, its positive ones `0`.  The polynomial arithmetic
/// sees only the zeros of independent generators.  One residue pass per
/// distinct base (and one numeric evaluation for a constant one); empty
/// for almost every input.
///
/// So is the denominator of a `0/0` candidate that vanishes by an identity
/// of its functions
/// ([`identity_denominator_candidates`](crate::base::canon::identity_denominator_candidates),
/// [`vanishes_by_identity`](crate::simplify::identically_zero::vanishes_by_identity)):
/// `((x + 1)² − x² − 2x − 1)/(tan x·cos x − sin x)` is `nan`, not the `0` of
/// the numerator multiplied out (SymPy 1.14's `ratsimp`: `0`).  Other
/// identities stay unapplied, as in SymPy: `ratsimp(1/(x/s + 1/s))` with
/// `s = sin 2x − 2·sin x·cos x` is `s/(x + 1)`.
///
/// Once something in `expr` vanishes — a denominator above, or a factor
/// of a product by the residue test — every denominator that may vanish
/// by an identity of its functions is tested as well ([`identity_bases`]):
/// with a zero in the expression, a pole anywhere makes `0·zoo` or `zoo +
/// zoo`.  Up to 0.33 `ratsimp((cosh²x − sinh²x − 1)⁻¹·(xy − 1)/(y +
/// 1/(x·(x + 4) − x² − 4x)))` was `0` (`zoo·0`, `nan`).  `known`: bases
/// already known to vanish (`Vanishing::Zero`), as [`ratsimp_vanishing`]
/// is given them; a factor beside one of their poles is tested too.
///
/// A generator that is `nan` at every point — a function at an argument
/// that is `nan` or `zoo` where the function has no value
/// ([`everywhere_values`](crate::base::canon::everywhere_values): `sin(c/d)`
/// for a constant `d` that is 0, `sin(zoo) = nan`) — takes the value `0/0`
/// ([`Vanishing::Undefined`]), which every operation keeps.  Up to 0.34
/// `ratsimp(sin((√2 + e)/(sin²1 + cos²1 − 1))·y²·((√y + √x)² − x − y −
/// 2√x·√y))` was `0` (`nan·0`; SymPy 1.14 gives `0`, deliberately
/// different).  The arguments are walked once; only the bases of their
/// negative powers get a zero test.  A generator found `zoo` is left
/// alone: those tests do not see a zero by an identity of `sin` and `cos`
/// (`√(x/(sin²x + cos²x − 1) + y/0)` is `√(zoo + zoo) = nan`).
fn vanishing_bases(
    arena: &mut Arena,
    expr: ExprId,
    known: &FxHashMap<ExprId, Vanishing>,
) -> FxHashMap<ExprId, Vanishing> {
    // (power, base, negative?) of the skeleton, its products and its sums;
    // an explicit stack.
    let mut powers: Vec<(ExprId, ExprId, bool)> = Vec::new();
    let mut products: Vec<ExprId> = Vec::new();
    let mut sums: Vec<ExprId> = Vec::new();
    let mut generators: Vec<ExprId> = Vec::new();
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Mul(children) => {
                products.push(id);
                stack.extend(children.iter());
            }
            ExprNode::Add(children) => {
                sums.push(id);
                stack.extend(children.iter());
            }
            ExprNode::Neg(child) => stack.push(*child),
            ExprNode::Pow(base, e) => {
                if let Some(n) = integer_exponent(arena, *e) {
                    stack.push(*base);
                    if n != 0 && !arena.node(*base).is_atom() {
                        powers.push((id, *base, n < 0));
                    }
                } else if let Some(q) = arena.as_num(*e)
                    && !arena.node(*base).is_atom()
                {
                    // A generator `s^(p/q)`: `0` or `1/0` when `s` vanishes
                    // (`√(exp(2x) − exp(x)²)`).
                    powers.push((id, *base, q.is_negative()));
                    generators.push(id);
                } else {
                    generators.push(id);
                }
            }
            node if !node.is_atom() => generators.push(id),
            _ => {}
        }
    }
    let mut vanishing: FxHashMap<ExprId, Vanishing> = FxHashMap::default();
    let mut decided: FxHashMap<ExprId, bool> = FxHashMap::default();
    for &(_, base, _) in &powers {
        if known.get(&base) == Some(&Vanishing::Zero) {
            decided.insert(base, true);
        }
    }
    for &(_, base, negative) in &powers {
        if negative && !decided.contains_key(&base) {
            let zero = crate::base::canon::factor_vanishes(arena, base);
            decided.insert(base, zero);
        }
    }
    let candidates = crate::base::canon::identity_denominator_candidates(arena, &products);
    let mut by_identity: FxHashSet<ExprId> = FxHashSet::default();
    for &d in &candidates {
        if decided.get(&d) != Some(&true)
            && by_identity.insert(d)
            && crate::simplify::identically_zero::vanishes_by_identity(arena, d)
        {
            decided.insert(d, true);
        }
    }
    let something_vanishes = !candidates.is_empty() || decided.values().any(|&zero| zero);
    for d in identity_bases(arena, &powers, &products, something_vanishes) {
        if decided.get(&d) != Some(&true)
            && by_identity.insert(d)
            && crate::simplify::identically_zero::vanishes_by_identity(arena, d)
        {
            decided.insert(d, true);
        }
    }
    // A second term over a denominator zero by an identity: `zoo + zoo`.
    if decided.values().any(|&zero| zero) {
        let partners = crate::base::canon::sum_partner_denominators(arena, &sums, |d| {
            decided.get(&d) == Some(&true)
        });
        for d in partners {
            if decided.get(&d) != Some(&true)
                && crate::simplify::identically_zero::vanishes_by_identity(arena, d)
            {
                decided.insert(d, true);
            }
        }
    }
    // A factor beside such a pole that vanishes too makes the product `0/0`
    // (`((√x + √y)² − x − y − 2√x·√y)/(e³ − exp(1)³)`: the polynomial
    // arithmetic does not reduce `(√x)²` to `x`), also by an identity of
    // its functions: up to 0.33 `(sin²x + cos²x − sin²1 − cos²1)/(e^(x+y)
    // − eˣ·eʸ)` was `P/0 = zoo` there, and `1/((x − 2)² + …)` became `0`
    // (it is `nan`).  A pole is any negative power of such a base, or a
    // factor that is `zoo` by one
    // ([`nodes_over_vanishing_bases`](crate::base::canon::nodes_over_vanishing_bases)):
    // up to 0.34 `(sin²x + cos²x − 1)²/√(cosh²x − sinh²x − 1)` and
    // `(cosh²x − sinh²x − 1)·(y/(sin 2x − 2·sin x·cos x) − 2)` were `zoo`
    // (`0·zoo = nan`).  A factor vanishes too when one of the sums that
    // make it vanish does (`√(tan x·cos x − sin x)`,
    // [`zero_candidate_sums`](crate::base::canon::zero_candidate_sums)).
    let infinite = if decided.values().any(|&zero| zero) {
        let order = walk::post_order_ids(arena, expr);
        crate::base::canon::nodes_over_vanishing_bases(arena, &order, |b| {
            decided.get(&b) == Some(&true)
        })
    } else {
        FxHashSet::default()
    };
    for &prod in &products {
        let ExprNode::Mul(children) = arena.node(prod).clone() else {
            continue;
        };
        if !children.iter().any(|c| infinite.contains(c)) {
            continue;
        }
        for c in children {
            let positive = match arena.node(c) {
                ExprNode::Pow(_, e) => integer_exponent(arena, *e).is_none_or(|n| n > 0),
                _ => true,
            };
            if positive
                && !infinite.contains(&c)
                && !arena.node(c).is_atom()
                && !vanishing.contains_key(&c)
                && (crate::base::canon::factor_vanishes(arena, c)
                    || crate::simplify::identically_zero::vanishes_by_identity(arena, c)
                    || crate::base::canon::zero_candidate_sums(arena, c)
                        .into_iter()
                        .any(|s| {
                            s != c
                                && crate::simplify::identically_zero::vanishes_by_identity(arena, s)
                        }))
            {
                vanishing.insert(c, Vanishing::Zero);
            }
        }
    }
    for &(power, base, negative) in &powers {
        if decided.get(&base) == Some(&true) {
            vanishing.insert(base, Vanishing::Zero);
            let v = if negative {
                Vanishing::Pole
            } else {
                Vanishing::Zero
            };
            vanishing.insert(power, v);
        }
    }
    // With a zero in the skeleton, a generator whose argument has a pole by
    // an identity of its functions makes `0·nan`: up to 0.39
    // `ratsimp(sin(1/(tan x·cos x − sin x))·(x·(x + 2) − x² − 2x))` was `0`
    // (`sin(zoo)·0 = nan`; `simplify` and `expand` gave `nan`).  The bases
    // of the negative powers inside the generators get the identity test
    // only then.
    let zero_in_skeleton = something_vanishes
        || decided.values().any(|&zero| zero)
        || has_zero_factor(arena, &products);
    let mut argument_zeros: Vec<ExprId> = Vec::new();
    if zero_in_skeleton {
        let mut inner: Vec<ExprId> = Vec::new();
        for &g in &generators {
            for id in walk::post_order_ids(arena, g) {
                if let ExprNode::Pow(b, e) = arena.node(id)
                    && arena.as_num(*e).is_some_and(|q| q.is_negative())
                    && !inner.contains(b)
                    && may_vanish_by_identity(arena, *b)
                {
                    inner.push(*b);
                }
            }
        }
        for b in inner {
            if decided.get(&b) == Some(&true)
                || (by_identity.insert(b)
                    && crate::simplify::identically_zero::vanishes_by_identity(arena, b))
            {
                argument_zeros.push(b);
            }
        }
    }
    let values = crate::base::canon::everywhere_values(arena, &generators, &argument_zeros);
    for (&g, value) in generators.iter().zip(values) {
        if value == Everywhere::Undefined {
            vanishing.insert(g, Vanishing::Undefined);
        }
    }
    vanishing
}

/// The bases of the negative powers among `powers` that may vanish by an
/// identity of their functions ([`may_vanish_by_identity`]), when
/// something in the expression vanishes: `something_vanishes` (a
/// denominator, or a `0/0` candidate), or a factor of one of the
/// `products` that may vanish identically by the residue test
/// ([`may_vanish_identically_as_factor`](crate::base::canon::may_vanish_identically_as_factor):
/// `(x·(x − 3) − x² + 3x)·(x + (x + 2)²/(cosh²x − sinh²x − 1))`).  Empty
/// otherwise: the numeric identity test is not run on ordinary input.
fn identity_bases(
    arena: &Arena,
    powers: &[(ExprId, ExprId, bool)],
    products: &[ExprId],
    something_vanishes: bool,
) -> Vec<ExprId> {
    let mut bases: Vec<ExprId> = Vec::new();
    for &(_, base, negative) in powers {
        if negative && !bases.contains(&base) && may_vanish_by_identity(arena, base) {
            bases.push(base);
        }
    }
    if bases.is_empty() || something_vanishes {
        return bases;
    }
    if has_zero_factor(arena, products) {
        bases
    } else {
        Vec::new()
    }
}

/// Has one of the `products` a factor (not a negative power) that may
/// vanish identically by the residue test
/// ([`may_vanish_identically_as_factor`](crate::base::canon::may_vanish_identically_as_factor))?
fn has_zero_factor(arena: &Arena, products: &[ExprId]) -> bool {
    products.iter().any(|&p| {
        let ExprNode::Mul(children) = arena.node(p) else {
            return false;
        };
        children.iter().any(|&c| {
            let positive = match arena.node(c) {
                ExprNode::Pow(_, e) => arena.as_num(*e).is_none_or(|q| q.is_positive()),
                _ => true,
            };
            positive
                && !arena.node(c).is_atom()
                && crate::base::canon::may_vanish_identically_as_factor(arena, c)
        })
    })
}

/// Could `d` vanish identically other than as a rational function of its
/// generators: a non-atom with a function in it (outside the integer
/// powers, sums and products), without an infinity, `nan` or an
/// unevaluated node?
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

/// Is the numerator of `expr` (`as_numer_denom`) structurally 0 once
/// multiplied out (`expand`, then `eval`, which reach into function
/// arguments)?
pub(crate) fn numerator_expands_to_zero(arena: &mut Arena, expr: ExprId) -> bool {
    let (n, _) = crate::poly::polybridge::fraction_parts(arena, expr);
    let expanded = crate::transforms::expand::expand(arena, n);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    arena.is_zero_structural(expanded)
}

/// The rational normal form `P/Q` of `expr` over its generators (as for
/// [`ratsimp`]) written as `Σₘ m·(Pₘ/Qₘ)`: the monomials `m` of the
/// generators in `keys` (for the integrator, the transcendental
/// subexpressions that depend on the variable, `sin x`, `e^{2x}`,
/// `ln(a·x + 2)`) collected, and each coefficient `Pₘ/Q` reduced by its own
/// gcd.  When `Q` involves a key generator nothing is collected, and the
/// result is `P/Q` with `gcd(P, Q)` cancelled if that gcd is not a
/// constant: `(cosh x·x² + cosh x)/((x² + 1)·tanh 4x)` is `cosh x/tanh 4x`
/// (up to 0.31 it was `None`, and `∫ cosh x·coth 4x dx` multiplied and
/// divided by `x² + 1` stayed unevaluated).  `None` when there is a
/// single monomial whose coefficient has no common factor with `Q` (the
/// form would only be `ratsimp`'s expanded fraction), when `ratsimp` would
/// return `expr` unchanged, or when `Q` is a constant.
/// `(x²·cos x + cos x)/(x² + 1)` is `cos x`;
/// `(−10·ln(a·x + 2)·(x + a)(x + b) + 3)/((x + a)(x + b))` is
/// `−10·ln(a·x + 2) + 3/((x + a)(x + b))`.
pub(crate) fn collect_reduced_terms(
    arena: &mut Arena,
    expr: ExprId,
    keys: impl Fn(&Arena, ExprId) -> bool,
) -> Option<ExprId> {
    if !is_admissible(arena, expr) {
        return None;
    }
    let gens = collect_generators(arena, expr);
    let (p, q) = to_rational_function(arena, expr, &gens)?;
    if p.is_zero() || q.is_zero() {
        return None;
    }
    let key: Vec<bool> = gens.iter().map(|&g| keys(arena, g)).collect();
    let has_key = |exp: &[u32]| exp.iter().zip(&key).any(|(&e, &k)| k && e > 0);
    if q.terms().any(|(exp, _)| has_key(exp)) {
        if RatPoly::gcd(&p, &q).total_degree().unwrap_or(0) == 0 {
            return None;
        }
        let out = rebuild(arena, &p, &q, &gens)?;
        return (out != expr).then_some(out);
    }
    // Numerator terms grouped by their key monomial.
    let mut groups: Vec<(Vec<u32>, RatPoly)> = Vec::new();
    for (exp, c) in p.terms() {
        let key_exp: Vec<u32> = exp
            .iter()
            .zip(&key)
            .map(|(&e, &k)| if k { e } else { 0 })
            .collect();
        let rest_exp: Vec<u32> = exp
            .iter()
            .zip(&key)
            .map(|(&e, &k)| if k { 0 } else { e })
            .collect();
        let term = RatPoly::monomial(c.clone(), rest_exp);
        match groups.iter_mut().find(|(m, _)| *m == key_exp) {
            Some((_, acc)) => *acc = acc.add(&term),
            None => groups.push((key_exp, term)),
        }
    }
    if q.total_degree().unwrap_or(0) == 0
        || (groups.len() == 1 && RatPoly::gcd(&groups[0].1, &q).total_degree().unwrap_or(0) == 0)
    {
        return None;
    }
    groups.sort_by(|a, b| a.0.cmp(&b.0));
    let mut terms: Vec<ExprId> = Vec::with_capacity(groups.len());
    for (key_exp, coeff) in groups {
        let reduced = rebuild(arena, &coeff, &q, &gens)?;
        let monomial = multipoly_to_expr(arena, &RatPoly::monomial(Q::one(), key_exp), &gens);
        terms.push(arena.mul(&[monomial, reduced]));
    }
    let out = match terms.len() {
        1 => terms[0],
        _ => arena.add(&terms),
    };
    (out != expr).then_some(out)
}

/// No infinities, NaN or unevaluated nodes anywhere in the tree.
fn is_admissible(arena: &Arena, expr: ExprId) -> bool {
    if walk::has_unevaluated(arena, expr) {
        return false;
    }
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN => return false,
            node => stack.extend(node.children()),
        }
    }
    true
}

/// Integer exponent of a `Pow` node, if it is one within the degree budget.
fn integer_exponent(arena: &Arena, exp: ExprId) -> Option<i64> {
    let r = arena.as_num(exp)?;
    if !r.is_integer() {
        return None;
    }
    let n: i64 = r.to_integer().try_into().ok()?;
    if n.unsigned_abs() > u64::from(MAX_DEGREE) {
        return None;
    }
    Some(n)
}

/// Is this node handled structurally (as opposed to being a generator)?
fn is_structural(arena: &Arena, id: ExprId) -> bool {
    match arena.node(id) {
        ExprNode::Num(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => true,
        ExprNode::Pow(_, exp) => integer_exponent(arena, *exp).is_some(),
        _ => false,
    }
}

/// The maximal non-structural subexpressions of `expr`, in canonical order.
fn collect_generators(arena: &Arena, expr: ExprId) -> Vec<ExprId> {
    let mut gens: Vec<ExprId> = Vec::new();
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if is_structural(arena, id) {
            match arena.node(id) {
                ExprNode::Pow(base, _) => stack.push(*base),
                node => stack.extend(node.children()),
            }
        } else {
            gens.push(id);
        }
    }
    gens.sort_by(|a, b| {
        arena
            .sort_key(*a)
            .cmp(arena.sort_key(*b))
            .then_with(|| a.0.cmp(&b.0))
    });
    gens
}

/// Is `p` the constant polynomial 1?
fn is_one(p: &RatPoly) -> bool {
    p.num_terms() == 1 && p.total_degree() == Some(0) && p.leading_coeff().is_some_and(One::is_one)
}

/// Enforce the size budget.
fn within_budget(p: &RatPoly) -> bool {
    p.num_terms() <= MAX_TERMS && p.total_degree().unwrap_or(0) <= MAX_DEGREE
}

/// `base^n` for `n ≥ 0` by repeated squaring, `None` if the budget is hit.
fn pow(base: &RatPoly, mut n: u64) -> Option<RatPoly> {
    let nv = base.num_vars();
    let mut result = RatPoly::from_int(nv, 1);
    let mut sq = base.clone();
    while n > 0 {
        if n & 1 == 1 {
            result = result.mul(&sq);
            if !within_budget(&result) {
                return None;
            }
        }
        n >>= 1;
        if n > 0 {
            sq = sq.mul(&sq);
            if !within_budget(&sq) {
                return None;
            }
        }
    }
    Some(result)
}

/// `p1/q1 + p2/q2` over the least common multiple of the denominators (a
/// zero denominator has none: the product is used).
///
/// Two zero denominators make `0/0`, `nan`: each fraction is `zoo` (`P/0`)
/// or `nan` (`0/0`) at every point, and `zoo + zoo`, `nan + …` are `nan`.
/// Up to 0.33 they were added as fractions over the common denominator 0,
/// `(p1 + p2)/0 = zoo`, so `1/0 + 0/0` was `zoo` and `x·y/(1/(x·(x + 3) −
/// x² − 3x) + ((x + y)² − x² − 2xy − y²)/(x·(y + 4) − xy − 4x))` became
/// `x·y/zoo = 0` in `ratsimp` and `simplify` (SymPy 1.14's `simplify`:
/// `nan`).
fn add_fractions(p1: &RatPoly, q1: &RatPoly, p2: &RatPoly, q2: &RatPoly) -> (RatPoly, RatPoly) {
    if q1.is_zero() && q2.is_zero() {
        let nv = q1.num_vars();
        return (RatPoly::zero(nv), RatPoly::zero(nv));
    }
    if q1 == q2 {
        return (p1.add(p2), q1.clone());
    }
    if q1.is_zero() || q2.is_zero() {
        return (p1.mul(q2).add(&p2.mul(q1)), q1.mul(q2));
    }
    if is_one(q1) {
        return (p1.mul(q2).add(p2), q2.clone());
    }
    if is_one(q2) {
        return (p1.add(&p2.mul(q1)), q1.clone());
    }
    let l = RatPoly::lcm(q1, q2);
    match (l.div_exact(q1), l.div_exact(q2)) {
        (Some(m1), Some(m2)) => (p1.mul(&m1).add(&p2.mul(&m2)), l),
        _ => (p1.mul(q2).add(&p2.mul(q1)), q1.mul(q2)),
    }
}

/// Convert `expr` to `(P, Q)` over the generators.  Iterative post-order
/// over the structural nodes; generators are leaves.
fn to_rational_function(
    arena: &Arena,
    expr: ExprId,
    gens: &[ExprId],
) -> Option<(RatPoly, RatPoly)> {
    to_rational_function_tracked(arena, expr, gens, &mut false, &FxHashMap::default())
}

/// The value a node of `expr` is given in [`ratsimp_vanishing`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Vanishing {
    /// A subexpression that vanishes identically (`tan x·cos x − sin x`),
    /// or a positive power of one: `0`.
    Zero,
    /// A negative power of such a subexpression: the fraction `1/0`.
    Pole,
    /// A generator that is `nan` at every point (`sin(c/0)`): the fraction
    /// `0/0`, which every operation keeps.
    Undefined,
}

/// Is `expr` zero as a rational function `P/Q` of its generators, with
/// the imaginary unit among them taken modulo `i² + 1` (`P` reduces to 0
/// and `Q` does not)?  The exact zero test behind
/// [`identically_zero`](crate::simplify::identically_zero) once the
/// trigonometric and hyperbolic functions are written as exponentials,
/// where `i` is everywhere (`sin x = (e^{ix} − e^{−ix})/(2i)`).  `false`
/// when undecided: an infinity or an unevaluated node in `expr`, or the
/// size budget.
pub(crate) fn vanishes_as_rational_function(arena: &mut Arena, expr: ExprId) -> bool {
    if !is_admissible(arena, expr) {
        return false;
    }
    let gens = collect_generators(arena, expr);
    let Some((p, q)) = to_rational_function(arena, expr, &gens) else {
        return false;
    };
    let i = arena.i_unit();
    let Some(t) = gens.iter().position(|&g| g == i) else {
        return p.is_zero() && !q.is_zero();
    };
    // i^k = (−1)^(k div 2)·i^(k mod 2)
    let modulo_i = |f: &RatPoly| -> Option<RatPoly> {
        let terms: Vec<(Vec<u32>, Q)> = f
            .terms()
            .map(|(exp, c)| {
                let mut e = exp.to_vec();
                let k = e.get(t).copied().unwrap_or(0);
                if let Some(slot) = e.get_mut(t) {
                    *slot = k % 2;
                }
                let c = if (k / 2) % 2 == 1 {
                    -c.clone()
                } else {
                    c.clone()
                };
                (e, c)
            })
            .collect();
        RatPoly::from_terms(f.num_vars(), terms)
    };
    match (modulo_i(&p), modulo_i(&q)) {
        (Some(p), Some(q)) => p.is_zero() && !q.is_zero(),
        _ => false,
    }
}

/// [`ratsimp_zero_denominator`] for an `expr` in which the nodes of
/// `vanishing` are known to vanish identically by an identity the
/// polynomial arithmetic cannot see (`sin²x + cos²x − 1`): each takes its
/// [`Vanishing`] value and the rest is carried through the arithmetic of
/// `1/0 = zoo` — `nan` for `0/0`, `zoo` for `P/0`, and the zero absorbed
/// by a further division (`x/(1 + 1/0) = 0`).  `nan` too for `P/0` with a
/// numerator `P` that is 0 once multiplied out ([`numerator_expands_to_zero`]);
/// a numerator that vanishes only by an identity is the caller's to test.
/// `None` when `ratsimp` does not apply (an infinity or an unevaluated node
/// in `expr`, or the size budget).
pub(crate) fn ratsimp_vanishing(
    arena: &mut Arena,
    expr: ExprId,
    vanishing: &FxHashMap<ExprId, Vanishing>,
) -> Option<ExprId> {
    if !is_admissible(arena, expr) {
        return None;
    }
    let mut vanishing = vanishing.clone();
    for (id, v) in vanishing_bases(arena, expr, &vanishing) {
        vanishing.entry(id).or_insert(v);
    }
    let gens = collect_generators(arena, expr);
    let (p, q) = to_rational_function_tracked(arena, expr, &gens, &mut false, &vanishing)?;
    let out = rebuild(arena, &p, &q, &gens)?;
    if out == arena.complex_infinity && numerator_expands_to_zero(arena, expr) {
        return Some(arena.nan);
    }
    Some(out)
}

/// The value of `expr` in the arithmetic of `1/0 = zoo` when a negative
/// power in it (outside every generator) has a base that vanishes
/// identically: `ratsimp`'s normal form, where that power is a fraction
/// with denominator 0 — `nan` for `0/0`, `zoo` for `P/0`, and the zero
/// absorbed by a further division (`x/(1 + 1/0) = x/zoo = 0`).  `None`
/// when no such power is met (or `ratsimp` does not apply).
pub(crate) fn ratsimp_zero_denominator(arena: &mut Arena, expr: ExprId) -> Option<ExprId> {
    if !is_admissible(arena, expr) {
        return None;
    }
    let vanishing = vanishing_bases(arena, expr, &FxHashMap::default());
    let gens = collect_generators(arena, expr);
    let mut zero_denominator = false;
    let (p, q) =
        to_rational_function_tracked(arena, expr, &gens, &mut zero_denominator, &vanishing)?;
    if !zero_denominator {
        return None;
    }
    let out = rebuild(arena, &p, &q, &gens)?;
    if out == arena.complex_infinity && numerator_expands_to_zero(arena, expr) {
        return Some(arena.nan);
    }
    Some(out)
}

/// [`to_rational_function`], setting `zero_denominator` when the base of
/// a negative power is the zero polynomial, with the nodes of `vanishing`
/// taking their [`Vanishing`] values (`0`, or the fraction `1/0`).
fn to_rational_function_tracked(
    arena: &Arena,
    expr: ExprId,
    gens: &[ExprId],
    zero_denominator: &mut bool,
    vanishing: &FxHashMap<ExprId, Vanishing>,
) -> Option<(RatPoly, RatPoly)> {
    let nv = gens.len();
    let gen_index: FxHashMap<ExprId, usize> =
        gens.iter().enumerate().map(|(i, &g)| (g, i)).collect();
    let mut cache: FxHashMap<ExprId, (RatPoly, RatPoly)> = FxHashMap::default();
    let one = RatPoly::from_int(nv, 1);
    // (node, children already pushed?)
    let mut stack: Vec<(ExprId, bool)> = vec![(expr, false)];

    while let Some(&(id, expanded)) = stack.last() {
        if cache.contains_key(&id) {
            stack.pop();
            continue;
        }
        if let Some(v) = vanishing.get(&id) {
            let value = match v {
                Vanishing::Zero => (RatPoly::zero(nv), one.clone()),
                Vanishing::Pole => {
                    *zero_denominator = true;
                    (one.clone(), RatPoly::zero(nv))
                }
                Vanishing::Undefined => {
                    *zero_denominator = true;
                    (RatPoly::zero(nv), RatPoly::zero(nv))
                }
            };
            cache.insert(id, value);
            stack.pop();
            continue;
        }
        if let Some(&i) = gen_index.get(&id) {
            cache.insert(id, (RatPoly::var(nv, i), one.clone()));
            stack.pop();
            continue;
        }
        let node = arena.node(id).clone();
        if !expanded {
            if let Some(top) = stack.last_mut() {
                top.1 = true;
            }
            match &node {
                ExprNode::Pow(base, _) => stack.push((*base, false)),
                n => {
                    for child in n.children() {
                        stack.push((child, false));
                    }
                }
            }
            continue;
        }
        let value: (RatPoly, RatPoly) = match &node {
            ExprNode::Num(nid) => (RatPoly::constant(nv, arena.num(*nid).clone()), one.clone()),
            ExprNode::Add(children) => {
                let mut acc_p = RatPoly::zero(nv);
                let mut acc_q = one.clone();
                for c in children.iter() {
                    let (p, q) = cache.get(c)?;
                    let (np, nq) = add_fractions(&acc_p, &acc_q, p, q);
                    if !within_budget(&np) || !within_budget(&nq) {
                        return None;
                    }
                    acc_p = np;
                    acc_q = nq;
                }
                (acc_p, acc_q)
            }
            ExprNode::Mul(children) => {
                let mut acc_p = one.clone();
                let mut acc_q = one.clone();
                for c in children.iter() {
                    let (p, q) = cache.get(c)?;
                    acc_p = acc_p.mul(p);
                    acc_q = acc_q.mul(q);
                    if !within_budget(&acc_p) || !within_budget(&acc_q) {
                        return None;
                    }
                }
                (acc_p, acc_q)
            }
            ExprNode::Neg(inner) => {
                let (p, q) = cache.get(inner)?;
                (p.neg(), q.clone())
            }
            ExprNode::Pow(base, exp) => {
                let n = integer_exponent(arena, *exp)?;
                let (p, q) = cache.get(base)?;
                if n >= 0 {
                    (pow(p, n.unsigned_abs())?, pow(q, n.unsigned_abs())?)
                } else {
                    // A zero `p` gives the denominator 0 (see `ratsimp`).
                    *zero_denominator |= p.is_zero();
                    (pow(q, n.unsigned_abs())?, pow(p, n.unsigned_abs())?)
                }
            }
            // Unreachable: non-structural nodes are generators.
            _ => return None,
        };
        cache.insert(id, value);
        stack.pop();
    }
    cache.remove(&expr)
}

/// Leading coefficient under the lexicographic order.
fn lex_leading_coeff(p: &RatPoly) -> Option<Q> {
    let mut best: Option<(&[u32], &Q)> = None;
    for (exp, c) in p.terms() {
        match best {
            Some((be, _)) if Lex::cmp_exponents(exp, be) != std::cmp::Ordering::Greater => {}
            _ => best = Some((exp, c)),
        }
    }
    best.map(|(_, c)| c.clone())
}

/// Cancel, normalise to integer-primitive parts and rebuild `P / Q`
/// (`zoo` for `P/0`, `nan` for `0/0`).
fn rebuild(arena: &mut Arena, p: &RatPoly, q: &RatPoly, gens: &[ExprId]) -> Option<ExprId> {
    if q.is_zero() {
        return Some(if p.is_zero() {
            arena.nan
        } else {
            arena.complex_infinity
        });
    }
    if p.is_zero() {
        return Some(arena.zero);
    }
    let mut p = p.clone();
    let mut q = q.clone();

    // Cancel the polynomial GCD.
    let g = RatPoly::gcd(&p, &q);
    if g.total_degree().unwrap_or(0) > 0
        && let (Some(pq), Some(qq)) = (p.div_exact(&g), q.div_exact(&g))
    {
        p = pq;
        q = qq;
    }

    // Integer coefficients: P/Q = (Pz/dP) / (Qz/dQ) = (Pz·dQ) / (Qz·dP).
    let (dp, pz) = p.clear_denominators();
    let (dq, qz) = q.clear_denominators();
    p = pz.scale(&Ratio::from_integer(dq));
    q = qz.scale(&Ratio::from_integer(dp));

    // Remove the common integer content.
    let c = p.integer_content().gcd(&q.integer_content());
    if !c.is_zero() && !c.is_one() {
        let inv = Ratio::new(BigInt::one(), c);
        p = p.scale(&inv);
        q = q.scale(&inv);
    }

    // Positive leading coefficient in the denominator.
    if lex_leading_coeff(&q).is_some_and(|lc| lc.is_negative()) {
        p = p.neg();
        q = q.neg();
    }

    let p_expr = multipoly_to_expr(arena, &p, gens);
    if is_one(&q) {
        return Some(p_expr);
    }
    let q_expr = multipoly_to_expr(arena, &q, gens);
    Some(arena.div(p_expr, q_expr))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn show(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    #[test]
    fn linear_over_linear_cancels_and_normalises() {
        let mut a = Arena::new();
        let j = a.symbol("j");
        let one = a.one;
        let two = a.int(2);
        // (j² − 1)/(2j) − (j − 1)/2  →  (j − 1)/(2j)
        let j2 = a.pow(j, two);
        let num1 = a.sub(j2, one);
        let den1 = a.mul(&[two, j]);
        let t1 = a.div(num1, den1);
        let num2 = a.sub(j, one);
        let t2 = a.div(num2, two);
        let e = a.sub(t1, t2);
        let r = ratsimp(&mut a, e);
        let expected = a.div(num2, den1);
        assert_eq!(r, expected, "got {}", show(&a, r));
    }

    #[test]
    fn opaque_subexpressions_are_generators() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let s = a.sin(x);
        let two = a.int(2);
        let s2 = a.pow(s, two);
        let e = a.div(s2, s);
        assert_eq!(ratsimp(&mut a, e), s);
    }

    #[test]
    fn infinity_is_left_alone() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let inf = a.infinity;
        let e = a.add(&[x, inf]);
        assert_eq!(ratsimp(&mut a, e), e);
    }

    #[test]
    fn already_normal_input_is_unchanged() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let one = a.one;
        let e = a.add(&[x, one]);
        assert_eq!(ratsimp(&mut a, e), e);
        let inv = a.div(one, x);
        assert_eq!(ratsimp(&mut a, inv), inv);
    }
}
