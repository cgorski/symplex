//! [`ValueClass`]: what kind of value an expression is — finite, `+∞`,
//! `−∞`, a directed infinity off the real axis, `zoo`, `nan`, or not known
//! — decided in one place, so that a fold written for one class cannot fire
//! at another.
//!
//! The folds at special arguments (`Γ(s, ∞) = 0`, `J_ν(±∞) = 0`, `∞^q = ∞`,
//! `atan(±∞) = ±π/2`) used to compare their arguments with `arena.infinity`
//! and its siblings by hand, and each area grew its own "is this argument
//! finite" test (`finite_slot`, `finite_argument`, `finite_order` in
//! `canon`, `contains_infinity` in `eval`, `holds_infinity`,
//! `not_known_finite` in `solve`).  They disagreed at the edges, and each
//! disagreement was a silent wrong answer: a fold written for `±∞` fired at
//! `zoo`, at an infinity of unknown direction (`y·∞`), at an unevaluated
//! pole (`elliptic_k(besseli(0, 0))`, which is `K(1) = ∞`), or at a constant
//! not known to be finite — `uppergamma(elliptic_k(besseli(0, 0)), ∞)` was
//! 0 with `is_zero() = Some(true)` up to 0.56.  Here the question has one
//! answer, and the folds ask it.  See CONTRIBUTING.md, "Values at
//! infinity".
//!
//! # Which function
//!
//! | Need | Call |
//! |---|---|
//! | is `x` one of the atoms `∞`, `−∞`, `zoo`, `nan` (an O(1) identity test that decides a fold at a literal atom) | [`ValueClass::special_atom`] |
//! | is `x` proved finite (an argument a fold holds for "every finite value" of) | [`ValueClass::of`]`(…).`[`is_finite_proved`](ValueClass::is_finite_proved) |
//! | is `x` exactly `+∞` or `−∞` | [`ValueClass::of`]`(…).`[`real_infinity`](ValueClass::real_infinity) |
//! | does `x` hold a special atom anywhere inside | [`holds_special_atom`] |
//! | likewise, outside the bounds of formal integrals, sums and limits (a candidate value) | [`holds_special_atom_outside_formal`] |
//! | does `x` hold `±∞` anywhere inside (a closed form taken there may meet `∞ − ∞`) | [`holds_real_infinity`] |
//!
//! # The classes
//!
//! [`ValueClass::of`] is structural first and evaluates only symbol-free
//! constants:
//!
//! - **The atoms.**  `∞` is [`PosInf`](ValueClass::PosInf), `−∞`
//!   [`NegInf`](ValueClass::NegInf), `zoo`
//!   [`ComplexInf`](ValueClass::ComplexInf) (no direction), `nan`
//!   [`NaN`](ValueClass::NaN).
//! - **Directed infinities.**  `c·∞` for a nonzero Gaussian rational `c`
//!   (the canonical spellings `i·∞`, `−i·∞`, `(1 + i)·∞`) is
//!   [`DirectedInf`](ValueClass::DirectedInf) off the real axis (and `±∞`
//!   on it, though the constructors never leave `2·∞`).  A direction of
//!   unknown value (`y·∞`, `(ln 2 − 1)·∞` before its sign is certified) is
//!   [`Unknown`](ValueClass::Unknown): `y·∞` is `nan` at `y = 0`.
//! - **Finite.**  A number, a named constant (`π`, `e`, `i`, `γ`, `G`, `φ`),
//!   a symbol, and an expression with symbols and no special atom (a
//!   generic point, as everywhere in the tables of values: `1/x` is finite
//!   away from its pole).  A symbol-free constant without a special atom is
//!   finite when the assumptions prove it (`AssumptionCache::query`,
//!   `FINITE`) or its value certifies ([`Arena::part_signs`]).
//! - **Infinite by evaluation.**  A constant that is not finite and whose
//!   own value `evalf` proves infinite (`SymplexError::Infinite`, a
//!   function at a pole of exact arguments: `elliptic_pi(π, 1)` is `−∞`,
//!   `Ci(0)` unevaluated) is [`Infinite`](ValueClass::Infinite), its
//!   direction not decided.  One whose evaluation meets such a value inside
//!   (`1/elliptic_pi(π, 1)`) is `Unknown`, as with an atom inside; a
//!   constant `evalf` cannot evaluate for another reason (`erfinv(9/4)`) is
//!   `Unknown` too.  Up to 0.58 an unevaluated pole was `Unknown` like any
//!   other constant without a value.  These answers are memoised under
//!   `FactKey::Finite` and `FactKey::Infinite`.
//! - **Everything else** holding a special atom (`∞ + i`, `sin ∞`,
//!   `chebyshev_t(2, y·∞)`, a formal integral with an infinite bound) is
//!   `Unknown`: `of` does not decide finiteness through an infinity.  A
//!   caller that may accept a value the assumptions call finite asks them
//!   itself (`solve` does, with the public, every-point reading).
//!
//! # Corner cases
//!
//! - **`±∞ + c`.**  A sum with an infinite term is `Unknown`, not `PosInf`:
//!   its imaginary part is kept (`∞ + i` is not extended real).  `Min` and
//!   `Max`, which read their arguments as real, take `±∞ + c` as `±∞`
//!   themselves (`canon`'s `real_infinity`).
//! - **Formal nodes.**  An infinity in the bound of an integral, sum or
//!   limit counts as held: such a node may diverge, and is finite here only
//!   by a certified value… which `of` does not seek once a special atom is
//!   inside (the stricter of the readings the helpers it replaced took).
//! - **Symbols declared infinite.**  A symbol is a generic point here, as
//!   it was in the tables; the assumption `infinite` is not consulted.
//!
//! [`Arena::part_signs`]: crate::base::arena::Arena::part_signs

use rustc_hash::FxHashSet;

use crate::base::arena::{Arena, FactKey};
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::truth::SignSet;

/// `(a, b)` for the complex rational `a + b·i`.
type GaussianRational = (Q, Q);

/// The class of a value: finite, one of the special values, or not known.
/// Decided by [`ValueClass::of`]; read with the queries below, never by
/// comparing with the arena's atoms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ValueClass {
    /// Proved finite: a number, a named constant, a symbol, an expression
    /// with symbols and no special atom (a generic point), or a constant
    /// known finite by the assumptions or with a certified value.
    Finite,
    /// Exactly `+∞`.
    PosInf,
    /// Exactly `−∞`.
    NegInf,
    /// A nonzero finite multiple of `∞` off the real axis (`i·∞`, `(1 +
    /// i)·∞`).
    DirectedInf,
    /// `zoo`, the infinity without a direction.
    ComplexInf,
    /// `nan`, no value.
    NaN,
    /// Proved infinite by its evaluation (`SymplexError::Infinite`): an
    /// unevaluated pole such as `elliptic_pi(π, 1)` (`−∞`), the direction
    /// not decided.
    Infinite,
    /// Not decided: it may be finite, infinite or undefined (`y·∞`, `∞ + i`,
    /// `sin ∞`, an unevaluated pole).
    Unknown,
}

/// The direction of a real infinity ([`ValueClass::real_infinity`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Sign {
    /// `+∞`.
    Positive,
    /// `−∞`.
    Negative,
}

impl ValueClass {
    /// The class of `x` when it is one of the atoms `∞`, `−∞`, `zoo`,
    /// `nan` (an identity test, O(1)); `None` for every other expression.
    /// Use it where a fold is taken at a literal atom (`atan(∞) = π/2`),
    /// with a `match` that says what happens at each of them.
    #[inline]
    pub(crate) fn special_atom(arena: &Arena, x: ExprId) -> Option<ValueClass> {
        if x == arena.infinity {
            Some(ValueClass::PosInf)
        } else if x == arena.neg_infinity {
            Some(ValueClass::NegInf)
        } else if x == arena.complex_infinity {
            Some(ValueClass::ComplexInf)
        } else if x == arena.nan {
            Some(ValueClass::NaN)
        } else {
            None
        }
    }

    /// The class of `x` (see the module documentation): structural first;
    /// a symbol-free constant without a special atom is asked of the
    /// assumptions and, failing them, of its certified value (memoised).
    pub(crate) fn of(arena: &Arena, x: ExprId) -> ValueClass {
        if let Some(class) = Self::special_atom(arena, x) {
            return class;
        }
        match arena.node(x) {
            ExprNode::Num(_)
            | ExprNode::Symbol(_)
            | ExprNode::Pi
            | ExprNode::E
            | ExprNode::ImaginaryUnit
            | ExprNode::EulerGamma
            | ExprNode::Catalan
            | ExprNode::GoldenRatio => return ValueClass::Finite,
            ExprNode::Mul(factors) if factors.contains(&arena.infinity) => {
                return directed(arena, factors);
            }
            _ => {}
        }
        if holds_special_atom(arena, x) {
            return ValueClass::Unknown;
        }
        if !arena.is_symbol_free(x) {
            return ValueClass::Finite;
        }
        finite_constant(arena, x)
    }

    /// Is the value proved finite ([`Finite`](ValueClass::Finite))?
    #[inline]
    pub(crate) const fn is_finite_proved(self) -> bool {
        matches!(self, ValueClass::Finite)
    }

    /// The direction of `+∞` and `−∞`; `None` for every other class.
    #[inline]
    pub(crate) const fn real_infinity(self) -> Option<Sign> {
        match self {
            ValueClass::PosInf => Some(Sign::Positive),
            ValueClass::NegInf => Some(Sign::Negative),
            _ => None,
        }
    }
}

/// The class of `c·∞`, the product of `factors` (one of them `∞`): the
/// direction `c` must be a nonzero Gaussian rational (a number, `i`, `q·i`
/// or a sum of them); any other direction is not known.
fn directed(arena: &Arena, factors: &[ExprId]) -> ValueClass {
    use num_traits::{One, Signed, Zero};
    // `re + im·i`: the product of the factors other than `∞`.
    let (mut re, mut im) = (Q::one(), Q::zero());
    let mut infinities = 0usize;
    for &f in factors {
        if f == arena.infinity {
            infinities += 1;
            continue;
        }
        let Some((a, b)) = gaussian_rational(arena, f) else {
            return ValueClass::Unknown;
        };
        (re, im) = (&re * &a - &im * &b, &re * &b + &im * &a);
    }
    if infinities != 1 {
        return ValueClass::Unknown;
    }
    match (re.is_zero(), im.is_zero()) {
        (true, true) => ValueClass::NaN,
        (_, false) => ValueClass::DirectedInf,
        (false, true) if re.is_positive() => ValueClass::PosInf,
        (false, true) => ValueClass::NegInf,
    }
}

/// `(a, b)` when `x = a + b·i` for rationals `a`, `b`: a number, `i`,
/// `q·i`, or a sum of them.
fn gaussian_rational(arena: &Arena, x: ExprId) -> Option<GaussianRational> {
    use num_traits::{One, Zero};
    let term = |t: ExprId| -> Option<(Q, Q)> {
        if let Some(q) = arena.as_num(t) {
            return Some((q.clone(), Q::zero()));
        }
        if t == arena.i_unit {
            return Some((Q::zero(), Q::one()));
        }
        match arena.node(t) {
            ExprNode::Mul(cs) => match **cs {
                [c, u] if u == arena.i_unit => Some((Q::zero(), arena.as_num(c)?.clone())),
                _ => None,
            },
            _ => None,
        }
    };
    match arena.node(x) {
        ExprNode::Add(terms) => {
            let (mut re, mut im) = (Q::zero(), Q::zero());
            for &t in terms.iter() {
                let (a, b) = term(t)?;
                re += a;
                im += b;
            }
            Some((re, im))
        }
        _ => term(x),
    }
}

/// [`ValueClass::of`] for a symbol-free constant without a special atom:
/// finite when the assumptions prove it or its value certifies, infinite
/// when its own value is ([`infinite_constant`]).  Memoised under
/// [`FactKey::Finite`] (`UNKNOWN`, any finite sign, when proved; `EMPTY`
/// when not).
fn finite_constant(arena: &Arena, x: ExprId) -> ValueClass {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::base::truth::Truth;
    let finite = match arena.constant_fact(x, FactKey::Finite) {
        Some(memo) => !memo.is_empty(),
        None => {
            let finite = Truth::from(AssumptionCache::new().query(arena, x, Props::FINITE))
                .is_true()
                || arena.part_signs(x).is_some();
            let memo = if finite {
                SignSet::UNKNOWN
            } else {
                SignSet::EMPTY
            };
            arena.remember_constant_fact(x, FactKey::Finite, memo);
            finite
        }
    };
    if finite {
        ValueClass::Finite
    } else if infinite_constant(arena, x) {
        ValueClass::Infinite
    } else {
        ValueClass::Unknown
    }
}

/// Is the constant `x`, not proved finite, itself infinite by its
/// evaluation: it meets a value proved infinite ([`Arena::meets_infinity`])
/// and none of its arguments does (otherwise the infinity is inside,
/// `1/elliptic_pi(π, 1)`, and `x` may be anything).  Memoised under
/// [`FactKey::Infinite`] (`UNKNOWN` when proved, `EMPTY` when not).
fn infinite_constant(arena: &Arena, x: ExprId) -> bool {
    if let Some(memo) = arena.constant_fact(x, FactKey::Infinite) {
        return !memo.is_empty();
    }
    let infinite = arena.meets_infinity(x)
        && !arena
            .node(x)
            .children()
            .iter()
            .any(|&c| arena.meets_infinity(c));
    let memo = if infinite {
        SignSet::UNKNOWN
    } else {
        SignSet::EMPTY
    };
    arena.remember_constant_fact(x, FactKey::Infinite, memo);
    infinite
}

/// Does `x` hold `∞`, `−∞`, `zoo` or `nan` anywhere inside (also in the
/// bounds of formal integrals, sums and limits)?
pub(crate) fn holds_special_atom(arena: &Arena, x: ExprId) -> bool {
    holds(arena, x, |class| class.is_some())
}

/// Does `x` hold `∞`, `−∞`, `zoo` or `nan` outside the formal integrals,
/// sums, products, limits and series (whose bounds they may be: `∫₀^∞`,
/// `Σ_{k=1}^∞` are values in their own right)?  It looks inside function
/// arguments: `chebyshev_t(2, y·∞)`, `Γ(y·∞)`, `sin ∞`.
pub(crate) fn holds_special_atom_outside_formal(arena: &Arena, x: ExprId) -> bool {
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack = vec![x];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if ValueClass::special_atom(arena, id).is_some() {
            return true;
        }
        match arena.node(id) {
            ExprNode::Integral(..)
            | ExprNode::DefiniteIntegral(..)
            | ExprNode::Limit(..)
            | ExprNode::Sum(..)
            | ExprNode::Product_(..)
            | ExprNode::Series(..) => {}
            node => node.for_each_child(|c| stack.push(c)),
        }
    }
    false
}

/// Does `x` hold `∞` or `−∞` anywhere inside?  A closed form taken at such
/// an argument may meet `∞ − ∞` (`eval` takes none there).
pub(crate) fn holds_real_infinity(arena: &Arena, x: ExprId) -> bool {
    holds(arena, x, |class| {
        class.is_some_and(|c| c.real_infinity().is_some())
    })
}

/// Does a node of `x` satisfy `wanted` (given its
/// [`ValueClass::special_atom`])?  A walk of the DAG, each node once.
fn holds(arena: &Arena, x: ExprId, wanted: impl Fn(Option<ValueClass>) -> bool) -> bool {
    if arena.node(x).is_atom() {
        return wanted(ValueClass::special_atom(arena, x));
    }
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack = vec![x];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if wanted(ValueClass::special_atom(arena, id)) {
            return true;
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atoms_and_directions() {
        let mut a = Arena::new();
        assert_eq!(ValueClass::of(&a, a.infinity), ValueClass::PosInf);
        assert_eq!(ValueClass::of(&a, a.neg_infinity), ValueClass::NegInf);
        assert_eq!(
            ValueClass::of(&a, a.complex_infinity),
            ValueClass::ComplexInf
        );
        assert_eq!(ValueClass::of(&a, a.nan), ValueClass::NaN);
        let i_inf = a.mul(&[a.i_unit, a.infinity]);
        assert_eq!(ValueClass::of(&a, i_inf), ValueClass::DirectedInf);
        let one_plus_i = a.add(&[a.one, a.i_unit]);
        let diag = a.mul(&[one_plus_i, a.infinity]);
        assert_eq!(ValueClass::of(&a, diag), ValueClass::DirectedInf);
        let y = a.symbol("y");
        let y_inf = a.mul(&[y, a.infinity]);
        assert_eq!(ValueClass::of(&a, y_inf), ValueClass::Unknown);
        let shifted = a.add(&[a.infinity, a.i_unit]);
        assert_eq!(ValueClass::of(&a, shifted), ValueClass::Unknown);
        assert_eq!(
            ValueClass::of(&a, a.infinity).real_infinity(),
            Some(Sign::Positive)
        );
        assert_eq!(ValueClass::of(&a, i_inf).real_infinity(), None);
        assert_eq!(ValueClass::special_atom(&a, y), None);
    }

    #[test]
    fn finite_values() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let inv = a.pow(x, a.neg_one);
        assert!(ValueClass::of(&a, inv).is_finite_proved());
        assert!(ValueClass::of(&a, a.pi).is_finite_proved());
        let two = a.int(2);
        let ln2 = a.ln(two);
        assert!(ValueClass::of(&a, ln2).is_finite_proved());
        // A second call reads the memo.
        assert!(ValueClass::of(&a, ln2).is_finite_proved());
        let x_inf = a.add(&[x, a.infinity]);
        let sin = a.sin(x_inf);
        assert!(!ValueClass::of(&a, sin).is_finite_proved());
        assert!(holds_special_atom(&a, sin));
        assert!(holds_real_infinity(&a, sin));
        let zoo_sum = a.atan(a.complex_infinity);
        assert!(holds_special_atom(&a, zoo_sum));
        assert!(!holds_real_infinity(&a, zoo_sum));
        assert!(!holds_special_atom(&a, inv));
    }

    #[test]
    fn poles_proved_by_evaluation() {
        let mut a = Arena::new();
        let pi = a.pi;
        let pole = a.lib_apply(crate::base::libfn::LibFn::EllipticPi, &[pi, a.one]);
        assert_eq!(ValueClass::of(&a, pole), ValueClass::Infinite);
        assert!(!ValueClass::of(&a, pole).is_finite_proved());
        assert_eq!(ValueClass::of(&a, pole).real_infinity(), None);
        // The infinity inside: not the value's class.
        let inv = a.pow(pole, a.neg_one);
        assert_eq!(ValueClass::of(&a, inv), ValueClass::Unknown);
        // Off the pole: finite.
        let half = a.rational(1, 2);
        let off = a.lib_apply(crate::base::libfn::LibFn::EllipticPi, &[half, half]);
        assert_eq!(ValueClass::of(&a, off), ValueClass::Finite);
    }

    #[test]
    fn formal_bounds_are_not_values() {
        let mut a = Arena::new();
        let t = a.symbol("t");
        let minus_t = a.neg(t);
        let body = a.exp(minus_t);
        let integral = a.definite_integral(body, t, a.zero, a.infinity);
        assert!(holds_special_atom(&a, integral));
        assert!(!holds_special_atom_outside_formal(&a, integral));
        assert!(!ValueClass::of(&a, integral).is_finite_proved());
        let sin_inf = a.sin(a.infinity);
        let sum = a.add(&[integral, sin_inf]);
        assert!(holds_special_atom_outside_formal(&a, sum));
    }
}
