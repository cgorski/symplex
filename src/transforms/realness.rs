//! Realness of constants: exact through the assumption system, else by
//! certified numerical evaluation.  In `transforms` (not `base`) because
//! the numerical half needs `evalf`.

use crate::base::arena::Arena;
use crate::base::assumptions::{AssumptionCache, Props};
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;

/// Is the constant `c` real, for real values of its parameters?
/// `Some(true)`: known real; `Some(false)`: known not real; `None`:
/// undecided.  A symbol without declared assumptions counts as a real
/// parameter (the convention of the integrator and of `apart`, whose
/// constants come from real data); a declared one keeps its declaration.
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
    for &s in &params {
        if let ExprNode::Symbol(sid) = *arena.node(s)
            && arena.symbol_assumptions(sid) == Assumptions::default()
        {
            let mut real = Assumptions::default();
            real.known_true |= Props::REAL;
            reals.set_symbol_assumptions(s, real);
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
