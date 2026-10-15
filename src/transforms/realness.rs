//! Realness of constants: exact through the assumption system, else by
//! certified numerical evaluation.  In `transforms` (not `base`) because
//! the numerical half needs `evalf`.

use crate::base::arena::Arena;
use crate::base::assumptions::{AssumptionCache, Props};
use crate::base::node::{ExprId, ExprNode};
use crate::base::truth::Truth;
use crate::base::walk;

/// How [`constant_realness_with`] reads a symbol without declared
/// assumptions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unassumed {
    /// As a real parameter (the limit engine's reading, see
    /// [`constant_realness`]).
    Real,
    /// As what the crate's model says it is: a complex number, so a
    /// constant built from it is real only when the assumption system
    /// proves so for every complex value (`|a|`, `re(a)`, `a·conj(a)`).
    Complex,
}

/// Is the constant `c` real, for real values of its parameters?
/// `Some(true)`: known real; `Some(false)`: known not real; `None`:
/// undecided.  A symbol without declared assumptions counts as a real
/// parameter here (the reading of the limit engine, `limit.rs`, whose
/// limits are taken along real values); a declared one keeps its
/// declaration.  The integrator and `apart` use
/// [`constant_realness_as_declared`] instead, where such a symbol is a
/// complex parameter (decision D4, 0.31).
///
/// First the assumption system: `a + b`, `√2`, `ln 3` are real, `i`,
/// `√(−2)` are not; `√a` and `√(−a)` are undecided, since a real `a` may
/// have either sign.  Then, for a constant without parameters,
/// numerically: its value to `digits` correct digits with an imaginary part
/// beyond them (two digits of margin) is not real (`√(1/2 − √5/2)`).  If
/// the imaginary part is within them, `im(c)` is evaluated on its own and
/// `c` is real when that is a certified exact 0 (`√(3 − √5)`, which the
/// assumption system leaves open).  `evalf` refuses a value it cannot
/// certify (`PrecisionExhausted`), and then the answer is `None`; so it is
/// for a non-zero `im(c)` below the digits, which may be a rounding residue
/// (the imaginary part `evalf` (0.28) returns for a real `RootOf` root) or
/// a genuinely tiny imaginary part.  An `im(c)` that is only zero to the
/// precision reached is no proof: then `eval`'s exact realness tests decide
/// ([`constant_facts`](crate::transforms::eval::constant_facts): the exact
/// conjugation test over radicals and roots of unity, the minimal
/// polynomial), else `None`.  Up to 0.48 it counted as real:
/// `Γ(2 + i·d)` with `d = e^{10⁻¹⁵⁰} − 1 − 10⁻¹⁵⁰ ≈ 5·10⁻³⁰¹` (imaginary
/// part `2.1·10⁻³⁰¹`) was real, and `lim_{x→∞} (Γ(2 + i·d) − 1)·x` was
/// `0`.
pub(crate) fn constant_realness(
    arena: &mut Arena,
    c: ExprId,
    digits: u32,
    reals: &mut AssumptionCache,
) -> Option<bool> {
    constant_realness_with(arena, c, digits, reals, Unassumed::Real)
}

/// Is the constant `c` real under the **declared** assumptions of its
/// parameters?  As [`constant_realness`], except that a symbol without
/// declared assumptions is a complex parameter, as everywhere in the
/// crate's model: `a + b`, `a·x₀`, `√a` are undecided (`None`) for such
/// symbols and real (or undecided, for `√a`) once `a` and `b` are
/// declared real; `|a|`, `re(a)` are real either way.
///
/// This is the rule behind decision D4 (0.31): the integrator writes
/// `ln|u|` (and the real forms built on conjugate pairs) only where `u` is
/// real for every real value of the variable under the declared
/// assumptions of every other symbol.  Up to 0.30 the integrator and
/// `apart` read an unassumed symbol as a real parameter, so
/// `∫ dx/(a·x + 1)` was `ln|a·x + 1|/a`, whose derivative is not
/// `1/(a·x + 1)` for a non-real `a`.  The caller must pass a cache that
/// was not primed by [`constant_realness`] (which declares the unassumed
/// symbols real in it).
pub(crate) fn constant_realness_as_declared(
    arena: &mut Arena,
    c: ExprId,
    digits: u32,
    reals: &mut AssumptionCache,
) -> Option<bool> {
    constant_realness_with(arena, c, digits, reals, Unassumed::Complex)
}

fn constant_realness_with(
    arena: &mut Arena,
    c: ExprId,
    digits: u32,
    reals: &mut AssumptionCache,
    unassumed: Unassumed,
) -> Option<bool> {
    use crate::base::assumptions::Assumptions;
    match arena.node(c) {
        ExprNode::Num(_)
        | ExprNode::Pi
        | ExprNode::E
        | ExprNode::EulerGamma
        | ExprNode::Catalan
        | ExprNode::GoldenRatio => return Some(true),
        ExprNode::ImaginaryUnit => return Some(false),
        _ => {}
    }
    let params = walk::free_symbols(arena, c);
    if unassumed == Unassumed::Real {
        for &s in &params {
            if let ExprNode::Symbol(sid) = *arena.node(s)
                && arena.symbol_assumptions(sid) == Assumptions::default()
            {
                let mut real = Assumptions::default();
                real.known_true |= Props::REAL;
                reals.set_symbol_assumptions(s, real);
            }
        }
    }
    if let Some(known) = reals.query(arena, c, Props::REAL) {
        return Some(known);
    }
    if !params.is_empty() {
        return None;
    }
    // `evalf_complex` returns certified digits or a 0 that is only zero to
    // the precision reached, which is real to any digits: an imaginary part
    // beyond the digits is certified.
    let z = crate::transforms::evalf::evalf_complex(arena, c, digits).ok()?;
    if !crate::transforms::evalf::is_real_to_digits(&z, digits.saturating_sub(2)) {
        return Some(false);
    }
    let im = arena.im(c);
    let im = crate::transforms::eval::eval(arena, im);
    if walk::has_unevaluated(arena, im) {
        return None;
    }
    let search = crate::transforms::evalf::ZeroSearch::Cap;
    let w = crate::transforms::evalf::evalf_verdict(arena, im, digits, search).ok()?;
    match w.proven() {
        Some(w) => (w.0.is_zero() && w.1.is_zero()).then_some(true),
        None => {
            let facts = crate::transforms::eval::constant_facts(arena, c);
            (Truth::from(facts.query(Props::REAL)).is_true()).then_some(true)
        }
    }
}

/// The sign of `im c` for a symbol-free constant `c`, proved: `ZERO` for
/// a constant the assumption system proves real, otherwise the sign of
/// the imaginary part of its certified evaluation (`evalf` keeps a part
/// next to a much larger one only when its error ball excludes 0), or of
/// `im c` evaluated on its own (a certified exact 0 included).
/// [`SignSet::REAL`](crate::base::truth::SignSet::REAL) when undecided: an
/// imaginary part 0 to the precision reached, a constant with symbols or
/// one that does not evaluate.
///
/// Which side of the real axis a value lies on decides the side of a
/// branch cut; up to 0.52 the definite integrator read an `f64` imaginary
/// part below `10⁻¹²` of the value as 0 (on the axis), and the integrator
/// a crossing of `Ei`'s cut `2·10⁻¹⁴` left of its branch point as one off
/// the cut: `∫₀² e^{(1+i)x}/(x − 1 + 10⁻¹⁴i) dx` (`4.7265 + 2.6665i`) was
/// `−9.6453 + 11.8945i`, off by `2πi·e^{1+i}`.
pub(crate) fn imaginary_part_sign(arena: &mut Arena, c: ExprId) -> crate::base::truth::SignSet {
    use crate::base::truth::SignSet;
    if !walk::free_symbols(arena, c).is_empty() || walk::has_unevaluated(arena, c) {
        return SignSet::REAL;
    }
    let mut cache = AssumptionCache::new();
    if Truth::from(cache.query(arena, c, Props::REAL)).is_true() {
        return SignSet::ZERO;
    }
    let sign = |p: &astro_float::BigFloat| {
        if crate::base::bigfloat::is_strictly_positive(p) {
            SignSet::POSITIVE
        } else {
            SignSet::NEGATIVE
        }
    };
    use crate::transforms::evalf::{self, Settled, ZeroSearch};
    if let Ok((z, Settled::Certified)) = evalf::evalf_settled(arena, c, 16, ZeroSearch::Cap)
        && !z.1.is_zero()
    {
        return sign(&z.1);
    }
    let im = arena.im(c);
    let im = crate::transforms::eval::eval(arena, im);
    if arena.is_zero_structural(im) {
        return SignSet::ZERO;
    }
    if walk::has_unevaluated(arena, im) {
        return SignSet::REAL;
    }
    match evalf::evalf_verdict(arena, im, 16, ZeroSearch::Cap)
        .ok()
        .and_then(crate::base::truth::Verdict::proven)
    {
        Some(w) if w.0.is_zero() && w.1.is_zero() => SignSet::ZERO,
        Some(w) if w.1.is_zero() => sign(&w.0),
        _ => SignSet::REAL,
    }
}

/// Is the symbol `s` declared real (directly or through a declaration that
/// implies it, such as `positive` or `integer`)?  A symbol without
/// declared assumptions is not: it is a complex parameter in the crate's
/// model (decision D4, see [`constant_realness_as_declared`]).
pub(crate) fn symbol_declared_real(arena: &Arena, s: crate::base::node::SymbolId) -> bool {
    let mut declared = arena.symbol_assumptions(s);
    declared.normalize_declared();
    Truth::from(declared.query(Props::REAL)).is_true()
}
