//! Realness of constants: exact through the assumption system, else by
//! certified numerical evaluation.  In `transforms` (not `base`) because
//! the numerical half needs `evalf`.

use crate::base::arena::Arena;
use crate::base::assumptions::{AssumptionCache, Props};
use crate::base::node::{ExprId, ExprNode};
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
/// `c` is real when that is exactly 0 (`√(3 − √5)`, which the assumption
/// system leaves open).  `evalf` refuses a value it cannot certify
/// (`PrecisionExhausted`), and then the answer is `None`; so it is for a
/// non-zero `im(c)` below the digits, which may be a rounding residue (the
/// imaginary part `evalf` (0.28) returns for a real `RootOf` root) or a
/// genuinely tiny imaginary part.
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
    let z = crate::transforms::evalf::evalf_complex(arena, c, digits).ok()?;
    if !crate::transforms::evalf::is_real_to_digits(&z, digits.saturating_sub(2)) {
        return Some(false);
    }
    let im = arena.im(c);
    let im = crate::transforms::eval::eval(arena, im);
    if walk::has_unevaluated(arena, im) {
        return None;
    }
    let w = crate::transforms::evalf::evalf_complex(arena, im, digits).ok()?;
    (w.0.is_zero() && w.1.is_zero()).then_some(true)
}

/// Is the symbol `s` declared real (directly or through a declaration that
/// implies it, such as `positive` or `integer`)?  A symbol without
/// declared assumptions is not: it is a complex parameter in the crate's
/// model (decision D4, see [`constant_realness_as_declared`]).
pub(crate) fn symbol_declared_real(arena: &Arena, s: crate::base::node::SymbolId) -> bool {
    let mut declared = arena.symbol_assumptions(s);
    declared.normalize_declared();
    declared.query(Props::REAL) == Some(true)
}
