//! Mellin transform (table-based) and its inverse.
//!
//! `M{f}(s) = ∫₀^∞ x^{s−1} f(x) dx`, defined on the *fundamental strip*
//! `a < Re(s) < b` in which the integral converges. Every forward transform
//! returns the strip as a boolean condition alongside the transform.
//!
//! The public entry points are the `Ex` methods `mellin_transform` and
//! `inverse_mellin_transform`.
//!
//! # Table
//!
//! | `f(x)`                     | `M{f}(s)`                       | strip                 |
//! |----------------------------|---------------------------------|-----------------------|
//! | `e^{−x}`                   | `Γ(s)`                          | `Re s > 0`            |
//! | `e^{−x²}`                  | `Γ(s/2)/2`                      | `Re s > 0`            |
//! | `e^{−x^b}` (`b > 0`)       | `Γ(s/b)/b`                      | `Re s > 0`            |
//! | `1/(1+x)`                  | `π/sin(πs)`                     | `0 < Re s < 1`        |
//! | `1/(1+x)^ν`                | `B(s, ν − s)`                   | `0 < Re s < Re ν`     |
//! | `H(1−x)·x^a`               | `1/(s + a)`                     | `Re s > −a`           |
//! | `H(x−1)·x^a`               | `−1/(s + a)`                    | `Re s < −a`           |
//! | `H(1−x)·(1−x)^b`           | `B(s, b + 1)`                   | `Re s > 0`            |
//! | `sin x`                    | `Γ(s) sin(πs/2)`                | `−1 < Re s < 1`       |
//! | `cos x`                    | `Γ(s) cos(πs/2)`                | `0 < Re s < 1`        |
//! | `ln(1+x)`                  | `π/(s sin(πs))`                 | `−1 < Re s < 0`       |
//!
//! Rules: linearity (strips intersect), `x^a f(x) → F(s + a)`,
//! `f(ax) → a^{−s} F(s)` (`a > 0`), `f(x^b) → F(s/b)/|b|`,
//! `f′(x) → −(s − 1) F(s − 1)`, `ln(x)·f(x) → F′(s)`.
//!
//! Symbols other than the transform variables are treated as **real**
//! parameters; sign conditions are checked through the assumption system
//! and an unprovable condition is an error rather than a guess.
//!
//! # Inverse
//!
//! The inverse recognises the right-hand sides of the table (with the
//! shift and scaling rules applied in reverse). Where a transform belongs to
//! several strips (`1/(s + a)`) the strip `Re s > −a` — i.e. the function
//! supported on `(0, 1)` — is chosen; this is documented on
//! `Ex::inverse_mellin_transform`.
//!
//! `Ex::inverse_mellin_transform_in_strip` (since 0.60.1) takes the strip:
//! every table entry knows the strip of its function and its poles, and on
//! another pole-free strip the function is the table's one minus (to the
//! left) or plus (to the right) the residues of `F(s)·x^{−s}` at the poles
//! crossed (`entry_on_strip`).

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::calculus::transform_util::{
    Factors, Linear, any_in_scope, depends_on, fail, for_each_child_in_scope, linear_in, need_sign,
    param_sign, rebuild_in_scope, require_symbol, scope_post_order, split_factors,
};

/// Maximum nesting of rule applications.
const MAX_RULE_DEPTH: usize = 6;

/// Fundamental strip `lower < Re(s) < upper` (either end may be open).
#[derive(Clone, Copy, Debug)]
struct Strip {
    lower: Option<ExprId>,
    upper: Option<ExprId>,
}

impl Strip {
    fn all() -> Self {
        Self {
            lower: None,
            upper: None,
        }
    }

    fn above(lower: ExprId) -> Self {
        Self {
            lower: Some(lower),
            upper: None,
        }
    }

    fn between(lower: ExprId, upper: ExprId) -> Self {
        Self {
            lower: Some(lower),
            upper: Some(upper),
        }
    }

    fn below(upper: ExprId) -> Self {
        Self {
            lower: None,
            upper: Some(upper),
        }
    }

    /// Shift the strip by `a`: the strip of `F(s + a)` is that of `F`
    /// translated by `−a`.
    fn shifted(self, arena: &mut Arena, a: ExprId) -> Self {
        let sh = |arena: &mut Arena, b: Option<ExprId>| {
            b.map(|b| {
                let d = arena.sub(b, a);
                crate::transforms::eval::eval(arena, d)
            })
        };
        Self {
            lower: sh(arena, self.lower),
            upper: sh(arena, self.upper),
        }
    }

    /// Scale the strip by `b > 0` (for `F(s/b)`): bounds multiply by `b`.
    fn scaled(self, arena: &mut Arena, b: ExprId, positive: bool) -> Self {
        let sc = |arena: &mut Arena, e: Option<ExprId>| {
            e.map(|e| {
                let p = arena.mul(&[e, b]);
                crate::transforms::eval::eval(arena, p)
            })
        };
        let (lo, up) = (sc(arena, self.lower), sc(arena, self.upper));
        if positive {
            Self {
                lower: lo,
                upper: up,
            }
        } else {
            Self {
                lower: up,
                upper: lo,
            }
        }
    }

    /// Intersection of two strips (`max` of lower bounds, `min` of upper).
    fn intersect(self, arena: &mut Arena, other: Strip) -> Result<Self, SymplexError> {
        let lower = match (self.lower, other.lower) {
            (None, x) | (x, None) => x,
            (Some(a), Some(b)) => Some(extremum(arena, a, b, true)),
        };
        let upper = match (self.upper, other.upper) {
            (None, x) | (x, None) => x,
            (Some(a), Some(b)) => Some(extremum(arena, a, b, false)),
        };
        if let (Some(l), Some(u)) = (lower, upper) {
            let d = arena.sub(u, l);
            if let Some(sgn) = crate::calculus::limit::const_sign(arena, d)
                && sgn <= 0
            {
                return Err(fail(
                    "mellin_transform",
                    "the fundamental strips of the summands do not overlap",
                ));
            }
        }
        Ok(Self { lower, upper })
    }

    /// Is this strip proved to lie inside `outer`?  Bounds compare when
    /// they are the same expression or their difference has a known sign.
    fn within(self, arena: &mut Arena, outer: Strip) -> bool {
        let lower = match (outer.lower, self.lower) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(o), Some(w)) => proved_le(arena, o, w),
        };
        let upper = match (outer.upper, self.upper) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(o), Some(w)) => proved_le(arena, w, o),
        };
        lower && upper
    }

    /// The strip as a boolean condition on `Re(s)`.
    fn condition(self, arena: &mut Arena, s: ExprId) -> ExprId {
        let re_s = arena.re(s);
        let mut parts: SmallVec<[ExprId; 2]> = SmallVec::new();
        if let Some(l) = self.lower {
            parts.push(arena.gt(re_s, l));
        }
        if let Some(u) = self.upper {
            parts.push(arena.gt(u, re_s));
        }
        match parts.len() {
            0 => arena.bool_true(),
            1 => parts[0],
            _ => arena.and(&parts),
        }
    }
}

/// `max(a, b)` / `min(a, b)` for strip bounds: decided when the difference
/// has a known sign, otherwise a formal `Max`/`Min` node.
fn extremum(arena: &mut Arena, a: ExprId, b: ExprId, want_max: bool) -> ExprId {
    if a == b {
        return a;
    }
    let d = arena.sub(a, b);
    match crate::calculus::limit::const_sign(arena, d) {
        Some(sgn) if sgn > 0 => {
            if want_max {
                a
            } else {
                b
            }
        }
        Some(_) => {
            if want_max {
                b
            } else {
                a
            }
        }
        None => {
            let args: SmallVec<[ExprId; 4]> = smallvec::smallvec![a, b];
            if want_max {
                arena.intern(ExprNode::Max(args))
            } else {
                arena.intern(ExprNode::Min(args))
            }
        }
    }
}

/// `a ≤ b`, proved: the same expression, or a difference of known sign.
fn proved_le(arena: &mut Arena, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    let d = arena.sub(b, a);
    crate::calculus::limit::const_sign(arena, d).is_some_and(|sgn| sgn >= 0)
}

/// `a < b`, proved by the sign of the difference.
fn proved_lt(arena: &mut Arena, a: ExprId, b: ExprId) -> bool {
    let d = arena.sub(b, a);
    crate::calculus::limit::const_sign(arena, d).is_some_and(|sgn| sgn > 0)
}

/// A family of poles of a table entry on the real axis: `start`, `start +
/// step`, `start + 2·step`, … (one pole when `step` is `None`).
#[derive(Clone, Copy)]
struct PoleFamily {
    start: ExprId,
    step: Option<ExprId>,
}

/// Poles of one family examined before [`entry_on_strip`] gives up.
const MAX_POLES_PER_FAMILY: usize = 64;

/// The function whose Mellin transform is the table entry `atom` **on the
/// strip `want`**, given the table's function `f0` for the entry, the
/// strip `home` on which `f0` has that transform, and the entry's poles.
///
/// * `want` inside `home`: `f0`.
/// * `want` to the left of `home`, whose left end is a pole: the line of
///   integration of the inversion integral moves left across the poles
///   `p` with `want.upper ≤ p ≤ home.lower`, and the function is `f0 − Σ
///   Res_{s=p} atom(s)·x^{−s}` (`Γ(s)` on `−1 < Re s < 0` is the transform
///   of `e^{−x} − 1`).  To the right, `f0 + Σ Res` over `home.upper ≤ p ≤
///   want.lower`.
/// * Refused: a pole inside `want`; an end of `home` that is not a pole
///   (there the transform integral of `f0` stops converging because of the
///   growth of `f0`, and beyond it `atom` is the transform of no function:
///   `Γ(s)·cos(πs/2)` for `Re s > 1`); a position that cannot be decided
///   (symbolic bounds); more than [`MAX_POLES_PER_FAMILY`] poles to cross.
#[allow(clippy::too_many_arguments)]
fn entry_on_strip(
    arena: &mut Arena,
    atom: ExprId,
    f0: ExprId,
    home: Strip,
    families: &[PoleFamily],
    want: Strip,
    s: ExprId,
    x: ExprId,
) -> Result<ExprId, SymplexError> {
    const OP: &str = "inverse_mellin_transform_in_strip";
    if want.within(arena, home) {
        return Ok(f0);
    }
    // The side `want` lies on, and the closed range of poles to cross.
    let (leftwards, range_lo, range_hi) = match (want.upper, home.lower, want.lower, home.upper) {
        (Some(u), Some(l), _, _) if proved_le(arena, u, l) => (true, u, l),
        (_, _, Some(l), Some(u)) if proved_le(arena, u, l) => (false, u, l),
        _ => {
            return Err(fail(
                OP,
                "a pole of the transform lies inside the strip, or the strip cannot be placed \
                 among the poles",
            ));
        }
    };
    let undecided = || fail(OP, "a pole cannot be placed relative to the strip");
    // The end of `home` that the line of integration leaves through.
    let end = if leftwards { range_hi } else { range_lo };
    let mut crossed: Vec<ExprId> = Vec::new();
    let mut end_is_a_pole = false;
    for family in families {
        let mut finished = false;
        let rising = match family.step {
            Some(step) => match crate::calculus::limit::const_sign(arena, step) {
                Some(sgn) if sgn > 0 => true,
                Some(sgn) if sgn < 0 => false,
                _ => return Err(undecided()),
            },
            None => false,
        };
        for n in 0..MAX_POLES_PER_FAMILY {
            let p = match family.step {
                Some(step) if n > 0 => {
                    let k = arena.int(n as i64);
                    let off = arena.mul(&[k, step]);
                    let p = arena.add(&[family.start, off]);
                    crate::transforms::eval::eval(arena, p)
                }
                _ => family.start,
            };
            // Inside the strip asked for: no function has this transform there.
            let above_lower = want.lower.is_none_or(|l| proved_lt(arena, l, p));
            let below_upper = want.upper.is_none_or(|u| proved_lt(arena, p, u));
            if above_lower && below_upper {
                return Err(fail(OP, "a pole of the transform lies inside the strip"));
            }
            if proved_le(arena, range_lo, p) && proved_le(arena, p, range_hi) {
                crossed.push(p);
                if proved_le(arena, end, p) && proved_le(arena, p, end) {
                    end_is_a_pole = true;
                }
            } else if !(proved_lt(arena, p, range_lo) || proved_lt(arena, range_hi, p)) {
                return Err(undecided());
            }
            // The family has left the part of the axis that matters: the
            // range crossed and the strip asked for.
            let past = if rising {
                proved_lt(arena, range_hi, p) && want.upper.is_none_or(|u| proved_le(arena, u, p))
            } else {
                proved_lt(arena, p, range_lo) && want.lower.is_none_or(|l| proved_le(arena, p, l))
            };
            if past || family.step.is_none() {
                finished = true;
                break;
            }
        }
        if !finished {
            return Err(fail(
                OP,
                "too many poles between the strip and the table's strip",
            ));
        }
    }
    if !end_is_a_pole {
        return Err(fail(
            OP,
            "the transform is that of no function on this strip (the table's strip does not \
             end at a pole)",
        ));
    }
    let neg_s = arena.neg(s);
    let kernel = arena.pow(x, neg_s);
    let integrand = arena.mul(&[atom, kernel]);
    let mut terms = vec![f0];
    for p in crossed {
        let r = match crate::calculus::residue::residue(arena, integrand, s, p) {
            Ok(r) if !depends_on(arena, r, s) && !crate::base::walk::has_unevaluated(arena, r) => r,
            // The residue engine refuses the poles of `Γ` (it takes the
            // numerator for non-analytic); they are simple.
            _ => simple_pole_residue(arena, atom, s, x, p)?,
        };
        terms.push(if leftwards { arena.neg(r) } else { r });
    }
    let sum = arena.add(&terms);
    Ok(crate::transforms::eval::eval(arena, sum))
}

/// `Res_{s=p} atom(s)·x^{−s}` for a simple pole `p` of `atom`: `x^{−p}·lim_{s→p}
/// (s − p)·atom(s)`, the limit by the limit engine.  A limit that is not a
/// finite constant (a pole of higher order, a limit the engine does not
/// find) is a refusal.
fn simple_pole_residue(
    arena: &mut Arena,
    atom: ExprId,
    s: ExprId,
    x: ExprId,
    p: ExprId,
) -> Result<ExprId, SymplexError> {
    const OP: &str = "inverse_mellin_transform_in_strip";
    let distance = arena.sub(s, p);
    let scaled = arena.mul(&[distance, atom]);
    let c = crate::calculus::limit::limit(arena, scaled, s, p)?;
    if crate::base::value_class::ValueClass::special_atom(arena, c).is_some()
        || depends_on(arena, c, s)
        || crate::base::walk::has_unevaluated(arena, c)
    {
        return Err(fail(OP, "a residue could not be computed"));
    }
    let neg_p = arena.neg(p);
    let power = arena.pow(x, neg_p);
    Ok(arena.mul(&[c, power]))
}

/// The table's function `f0` of the entry `atom`, placed on the strip
/// asked for ([`entry_on_strip`]); `f0` itself when no strip is asked for
/// (`Ex::inverse_mellin_transform`).
#[allow(clippy::too_many_arguments)]
fn table_entry(
    arena: &mut Arena,
    atom: ExprId,
    f0: ExprId,
    home: Strip,
    families: &[PoleFamily],
    want: Option<Strip>,
    s: ExprId,
    x: ExprId,
) -> Result<Option<ExprId>, SymplexError> {
    match want {
        None => Ok(Some(f0)),
        Some(want) => entry_on_strip(arena, atom, f0, home, families, want, s, x).map(Some),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Mellin transform of `expr` (a function of `x`) as a function of `s`,
/// together with the fundamental strip as a boolean condition on `Re(s)`.
pub(crate) fn mellin_transform(
    arena: &mut Arena,
    expr: ExprId,
    x: ExprId,
    s: ExprId,
) -> Result<(ExprId, ExprId), SymplexError> {
    const OP: &str = "mellin_transform";
    require_symbol(arena, x, OP, "the variable")?;
    require_symbol(arena, s, OP, "the transform variable")?;
    if x == s {
        return Err(SymplexError::InvalidArgument {
            operation: OP,
            reason: "the variable and the transform variable must be distinct".into(),
        });
    }
    // `Piecewise` inputs with linear conditions become sums of steps (the
    // Fourier transform's conversion), expanded so that each term is one
    // step times powers; before 0.31 every `Piecewise` was refused.
    // (A `Piecewise` under a binder of `x` is not a piece of the function
    // being transformed: `transform_util`.)
    let expr = if any_in_scope(arena, expr, x, |node| {
        matches!(node, ExprNode::Piecewise(_))
    }) {
        let e = crate::calculus::fourier_transform::eliminate_piecewise(arena, expr, x, OP)?;
        crate::transforms::expand::expand(arena, e)
    } else {
        expr
    };
    let mut result = Err(fail(OP, "no transform rule applies"));
    for candidate in steps_on_positive_axis(arena, expr, x) {
        result = forward(arena, candidate, x, s, 0);
        if result.is_ok() {
            break;
        }
    }
    let (f, strip) = result?;
    let f = crate::transforms::eval::eval(arena, f);
    if depends_on(arena, f, x) || crate::base::walk::has_unevaluated(arena, f) {
        return Err(fail(OP, "result still depends on the variable"));
    }
    let cond = strip.condition(arena, s);
    Ok((f, cond))
}

/// The forms of `expr` the forward rules are tried on, in order.  Heaviside
/// steps are read on `x > 0`: a step with threshold `≤ 0` is `1`
/// (rising) or `0` (falling).  When writing every rising step
/// `H(x − c)` as `1 − H(c − x)` makes the step-free part cancel, the
/// function has compact support and is returned in that falling-step form,
/// whose terms `g(x)·H(c − x)` have overlapping strips `Re s > …`.  The
/// window `H(x − 1) − H(x − 2)` as it stands is a difference of two
/// transforms on `Re s < 0`, and the triangle `x·H(1 − x) + (2 − x)·H(x − 1)
/// − (2 − x)·H(x − 2)` mixes `Re s > −1` with `Re s < −1` ("strips do not
/// overlap"); in falling form both are integrals over bounded intervals.
/// That form comes first, then `expr` with the thresholds read, then its
/// expansion (a polynomial times a step, `(x − 2)²·H(x − 2)`, monomial by
/// monomial).  A step under a binder of `x` is not read (the bound `x` of
/// `Σ_{x=−3}^{3} H(x + 1)` is negative too: up to 0.59 that sum became 7,
/// not 9/2).
fn steps_on_positive_axis(arena: &mut Arena, expr: ExprId, x: ExprId) -> Vec<ExprId> {
    let has_step = any_in_scope(arena, expr, x, |node| {
        matches!(node, ExprNode::Heaviside(_))
    });
    if !has_step {
        return vec![expr];
    }
    // (threshold sign, rising) of a step, when decidable.
    let classify = |arena: &mut Arena, arg: ExprId| -> Option<(i32, bool, ExprId)> {
        let Linear { a, b } = linear_in(arena, arg, x)?;
        let sa = param_sign(arena, a).filter(|&v| v != 0)?;
        let ratio = arena.div(b, a);
        let ratio = arena.neg(ratio);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        Some((param_sign(arena, ratio)?, sa > 0, ratio))
    };
    let rebuild = |arena: &mut Arena, falling_form: bool| -> ExprId {
        let post = scope_post_order(arena, expr, x);
        let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
        for id in post {
            let rebuilt = rebuild_in_scope(arena, id, x, &cache);
            let new = match arena.node(rebuilt).clone() {
                ExprNode::Heaviside(arg) => match classify(arena, arg) {
                    Some((sg, rising, _)) if sg <= 0 => {
                        if rising {
                            arena.one()
                        } else {
                            arena.zero()
                        }
                    }
                    Some((_, true, c)) if falling_form => {
                        let cmx = arena.sub(c, x);
                        let h = arena.heaviside(cmx);
                        let one = arena.one();
                        arena.sub(one, h)
                    }
                    _ => rebuilt,
                },
                _ => rebuilt,
            };
            cache.insert(id, new);
        }
        cache.get(&expr).copied().unwrap_or(expr)
    };
    let plain = rebuild(arena, false);
    let mut out = Vec::with_capacity(3);
    let falling = rebuild(arena, true);
    if falling != plain {
        let falling = crate::transforms::expand::expand(arena, falling);
        let terms: Vec<ExprId> = match arena.node(falling) {
            ExprNode::Add(ch) => ch.to_vec(),
            _ => vec![falling],
        };
        let (stepped, free): (Vec<ExprId>, Vec<ExprId>) = terms.into_iter().partition(|&t| {
            any_in_scope(arena, t, x, |node| matches!(node, ExprNode::Heaviside(_)))
        });
        let free_sum = arena.add(&free);
        let free_sum = crate::transforms::expand::expand(arena, free_sum);
        let free_sum = crate::transforms::eval::eval(arena, free_sum);
        if !stepped.is_empty() && arena.is_zero_structural(free_sum) {
            out.push(arena.add(&stepped));
        }
    }
    out.push(plain);
    let expanded = crate::transforms::expand::expand(arena, plain);
    if expanded != plain {
        out.push(expanded);
    }
    out
}

/// Inverse Mellin transform of `expr` (a function of `s`) as a function of `x`.
pub(crate) fn inverse_mellin_transform(
    arena: &mut Arena,
    expr: ExprId,
    s: ExprId,
    x: ExprId,
) -> Result<ExprId, SymplexError> {
    const OP: &str = "inverse_mellin_transform";
    require_symbol(arena, s, OP, "the transform variable")?;
    require_symbol(arena, x, OP, "the variable")?;
    if x == s {
        return Err(SymplexError::InvalidArgument {
            operation: OP,
            reason: "the variable and the transform variable must be distinct".into(),
        });
    }
    let f = inverse(arena, expr, s, x, 0, None)?;
    let f = crate::transforms::eval::eval(arena, f);
    if depends_on(arena, f, s) || crate::base::walk::has_unevaluated(arena, f) {
        return Err(fail(OP, "result still depends on the transform variable"));
    }
    Ok(f)
}

/// The function of `x` whose Mellin transform is `expr` (a function of
/// `s`) **on the strip `lower < Re s < upper`** (`-oo` and `oo` for an open
/// side).
///
/// A transform determines its function only together with its strip: `−1/s`
/// is the transform of `H(x − 1)` on `Re s < 0` and of `−H(1 − x)` on
/// `Re s > 0`.  Sums, constant factors, `a^{±s}` and shifts `G(s + a)` carry the
/// strip to the table entries (a shift moves it by `a`), and each entry is
/// placed on its strip by [`entry_on_strip`]: the table's function when the
/// strip lies inside the entry's own, corrected by the residues at the
/// poles crossed when it lies beyond a pole, and a refusal when a pole lies
/// inside the strip, when the strip is empty, or when the strip cannot be
/// placed among the poles.
pub(crate) fn inverse_mellin_transform_in_strip(
    arena: &mut Arena,
    expr: ExprId,
    s: ExprId,
    x: ExprId,
    lower: ExprId,
    upper: ExprId,
) -> Result<ExprId, SymplexError> {
    use crate::base::value_class::ValueClass;
    const OP: &str = "inverse_mellin_transform_in_strip";
    require_symbol(arena, s, OP, "the transform variable")?;
    require_symbol(arena, x, OP, "the variable")?;
    if x == s {
        return Err(SymplexError::InvalidArgument {
            operation: OP,
            reason: "the variable and the transform variable must be distinct".into(),
        });
    }
    let invalid = |reason: &str| SymplexError::InvalidArgument {
        operation: OP,
        reason: reason.into(),
    };
    for bound in [lower, upper] {
        if depends_on(arena, bound, s) || depends_on(arena, bound, x) {
            return Err(invalid(
                "a bound of the strip depends on a transform variable",
            ));
        }
    }
    let lower = match ValueClass::special_atom(arena, lower) {
        None => Some(lower),
        Some(ValueClass::NegInf) => None,
        Some(_) => {
            return Err(invalid(
                "the lower bound of the strip is a real number or -oo",
            ));
        }
    };
    let upper = match ValueClass::special_atom(arena, upper) {
        None => Some(upper),
        Some(ValueClass::PosInf) => None,
        Some(_) => {
            return Err(invalid(
                "the upper bound of the strip is a real number or oo",
            ));
        }
    };
    if let (Some(l), Some(u)) = (lower, upper)
        && proved_le(arena, u, l)
    {
        return Err(invalid("the strip is empty"));
    }
    let f = inverse(arena, expr, s, x, 0, Some(Strip { lower, upper }))?;
    let f = crate::transforms::eval::eval(arena, f);
    if depends_on(arena, f, s) || crate::base::walk::has_unevaluated(arena, f) {
        return Err(fail(OP, "result still depends on the transform variable"));
    }
    Ok(f)
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared helpers
// ═══════════════════════════════════════════════════════════════════════════
//
// `split_factors`, `linear_in` (`a·v + b`), the dependence test `depends_on`
// and the sign and error helpers are in `calculus::transform_util`, shared
// with the Laplace, Z and Fourier transforms.

/// `expr = x^k` (any constant exponent `k`, including `x` itself) → `Some(k)`.
fn power_of_var(arena: &Arena, expr: ExprId, v: ExprId) -> Option<ExprId> {
    if expr == v {
        return Some(arena.one());
    }
    if let ExprNode::Pow(base, e) = arena.node(expr)
        && *base == v
        && !depends_on(arena, *e, v)
    {
        return Some(*e);
    }
    None
}

/// `expr = (1 + x)^{−ν}` → `Some(ν)` (as a positive quantity), or
/// `1/(1 + x)` → `Some(1)`.
fn one_plus_x_power(arena: &mut Arena, expr: ExprId, x: ExprId) -> Option<ExprId> {
    let one = arena.one();
    let one_plus_x = arena.add(&[one, x]);
    if let ExprNode::Pow(base, e) = arena.node(expr).clone()
        && base == one_plus_x
        && !depends_on(arena, e, x)
    {
        let nu = arena.neg(e);
        return Some(crate::transforms::eval::eval(arena, nu));
    }
    None
}

/// `arg = −c·x^b` → `Some((c, b))` with `c`, `b` free of `x`.
fn neg_coeff_power(arena: &mut Arena, arg: ExprId, x: ExprId) -> Option<(ExprId, ExprId)> {
    let (coeff, rest) = match arena.node(arg).clone() {
        ExprNode::Neg(inner) => (arena.one(), inner),
        ExprNode::Mul(ch) => {
            let mut consts: SmallVec<[ExprId; 4]> = SmallVec::new();
            let mut dep: Option<ExprId> = None;
            for &c in &ch {
                if depends_on(arena, c, x) {
                    if dep.is_some() {
                        return None;
                    }
                    dep = Some(c);
                } else {
                    consts.push(c);
                }
            }
            let c = arena.mul(&consts);
            let c = arena.neg(c);
            (crate::transforms::eval::eval(arena, c), dep?)
        }
        _ => return None,
    };
    let b = power_of_var(arena, rest, x)?;
    Some((coeff, b))
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward transform
// ═══════════════════════════════════════════════════════════════════════════

fn forward(
    arena: &mut Arena,
    expr: ExprId,
    x: ExprId,
    s: ExprId,
    depth: usize,
) -> Result<(ExprId, Strip), SymplexError> {
    const OP: &str = "mellin_transform";
    if depth > MAX_RULE_DEPTH {
        return Err(fail(OP, "rule nesting too deep"));
    }
    let expr = crate::transforms::eval::eval(arena, expr);
    let node = arena.node(expr).clone();

    // Linearity: strips intersect.
    if let ExprNode::Add(children) = &node {
        let kids = children.clone();
        let mut terms = Vec::with_capacity(kids.len());
        let mut strip = Strip::all();
        for child in kids {
            let (f, st) = forward(arena, child, x, s, depth)?;
            terms.push(f);
            strip = strip.intersect(arena, st)?;
        }
        return Ok((arena.add(&terms), strip));
    }
    if let ExprNode::Neg(inner) = node {
        let (f, st) = forward(arena, inner, x, s, depth)?;
        return Ok((arena.neg(f), st));
    }
    if !depends_on(arena, expr, x) {
        return Err(fail(
            OP,
            "a non-zero constant has no Mellin transform (the integral diverges)",
        ));
    }

    // Constant factors.
    let Factors { consts, deps } = split_factors(arena, expr, x);
    if !consts.is_empty() {
        let body = arena.mul(&deps);
        let (f, st) = forward(arena, body, x, s, depth)?;
        let mut all = consts;
        all.push(f);
        return Ok((arena.mul(&all), st));
    }

    // Power rule: x^a · g(x) → G(s + a).
    if let ExprNode::Mul(children) = &node {
        let kids: Vec<ExprId> = children.iter().copied().collect();
        let mut a_total: Option<ExprId> = None;
        let mut rest: Vec<ExprId> = Vec::new();
        for &k in &kids {
            if let Some(e) = power_of_var(arena, k, x) {
                a_total = Some(match a_total {
                    Some(t) => arena.add(&[t, e]),
                    None => e,
                });
            } else {
                rest.push(k);
            }
        }
        if let Some(a) = a_total {
            if rest.is_empty() {
                return Err(fail(OP, "a pure power of x has no Mellin transform"));
            }
            let g = arena.mul(&rest);
            let (gf, st) = forward(arena, g, x, s, depth + 1)?;
            let s_plus_a = arena.add(&[s, a]);
            let shifted = crate::transforms::subs::subs(arena, gf, s, s_plus_a);
            return Ok((shifted, st.shifted(arena, a)));
        }
        // ln(x)·g(x) → G′(s)
        if let Some(idx) = rest
            .iter()
            .position(|&k| matches!(arena.node(k), ExprNode::Ln(inner) if *inner == x))
        {
            let g_factors: Vec<ExprId> = rest
                .iter()
                .copied()
                .enumerate()
                .filter(|&(j, _)| j != idx)
                .map(|(_, k)| k)
                .collect();
            if !g_factors.is_empty() {
                let g = arena.mul(&g_factors);
                let (gf, st) = forward(arena, g, x, s, depth + 1)?;
                let d = crate::transforms::diff::diff(arena, gf, s);
                return Ok((d, st));
            }
        }
        // Step windows: H(1 − x)·g(x), H(x − 1)·g(x).
        if let Some(r) = forward_step_product(arena, &rest, x, s, depth)? {
            return Ok(r);
        }
    }

    // Table entries for a single function.
    if let Some(r) = forward_table(arena, expr, &node, x, s)? {
        return Ok(r);
    }

    // (c + g(x))^k with a positive constant c ≠ 1 → c^k (1 + g(x)/c)^k, so
    // that the scaling rule can reduce `1/(2 + x)` to `1/(1 + x)`.
    if let ExprNode::Pow(base, k) = node
        && !depends_on(arena, k, x)
        && let ExprNode::Add(terms) = arena.node(base).clone()
    {
        let one = arena.one();
        let (consts, deps): (Vec<ExprId>, Vec<ExprId>) = terms
            .iter()
            .copied()
            .partition(|&t| !depends_on(arena, t, x));
        if !consts.is_empty() && !deps.is_empty() {
            let c = arena.add(&consts);
            if c != one && param_sign(arena, c) == Some(1) {
                let g = arena.add(&deps);
                let g_over_c = arena.div(g, c);
                let inner = arena.add(&[one, g_over_c]);
                let p = arena.pow(inner, k);
                let ck = arena.pow(c, k);
                let (f, st) = forward(arena, p, x, s, depth + 1)?;
                return Ok((arena.mul(&[ck, f]), st));
            }
        }
    }

    // Derivative rule: f′(x) → −(s − 1) F(s − 1).
    if let ExprNode::Derivative(body, var) = node
        && var == x
    {
        let (f, st) = forward(arena, body, x, s, depth + 1)?;
        let one = arena.one();
        let s_m1 = arena.sub(s, one);
        let shifted = crate::transforms::subs::subs(arena, f, s, s_m1);
        let neg_s_m1 = arena.neg(s_m1);
        let r = arena.mul(&[neg_s_m1, shifted]);
        let neg_one = arena.neg_one();
        return Ok((r, st.shifted(arena, neg_one)));
    }

    // Scaling / power-substitution rules: f(a·x), f(x^b).
    if let Some(r) = forward_substitution_rules(arena, expr, x, s, depth)? {
        return Ok(r);
    }

    // Rational functions by residues.
    if let Some(r) = forward_rational(arena, expr, x, s) {
        return Ok(r);
    }
    // Damped oscillations e^{−αx}·cos βx, e^{−αx}·sin βx.
    if let Some(r) = forward_damped_trig(arena, expr, x, s) {
        return Ok(r);
    }

    Err(fail(
        OP,
        format!("no transform rule applies to {}", arena.display(expr)),
    ))
}

/// `M{e^{−αx} cos βx}(s) = Γ(s)·((α − iβ)^{−s} + (α + iβ)^{−s})/2` on `Re s > 0`
/// and `M{e^{−αx} sin βx}(s) = Γ(s)·((α − iβ)^{−s} − (α + iβ)^{−s})/(2i)` on
/// `Re s > −1`, for numbers `α > 0`, `β ≠ 0` (`M{e^{−cx}} = Γ(s)c^{−s}`
/// holds for `Re c > 0` with the principal power).  Before 0.46
/// `e^{−x}·sin x` was refused; SymPy: `mellin_transform(exp(-x)*sin(x), x,
/// s)` = `2**(-s/2)*gamma(s)*sin(pi*s/4)` on `(-1, oo)`.
fn forward_damped_trig(
    arena: &mut Arena,
    expr: ExprId,
    x: ExprId,
    s: ExprId,
) -> Option<(ExprId, Strip)> {
    let ExprNode::Mul(factors) = arena.node(expr).clone() else {
        return None;
    };
    if factors.len() != 2 {
        return None;
    }
    let mut alpha = None;
    let mut trig = None;
    for f in factors {
        match arena.node(f).clone() {
            ExprNode::Exp(arg) => {
                let Linear { a, b } = linear_in(arena, arg, x)?;
                if !arena.is_zero_structural(b) {
                    return None;
                }
                let neg = arena.neg(a);
                alpha = Some(crate::transforms::eval::eval(arena, neg));
            }
            ExprNode::Sin(arg) | ExprNode::Cos(arg) => {
                let Linear { a: b, b: c } = linear_in(arena, arg, x)?;
                if !arena.is_zero_structural(c) {
                    return None;
                }
                trig = Some((matches!(arena.node(f), ExprNode::Sin(_)), b));
            }
            _ => return None,
        }
    }
    let (alpha, (is_sin, beta)) = (alpha?, trig?);
    let a = arena.as_num(alpha)?.clone();
    if !num_traits::Signed::is_positive(&a)
        || arena.as_num(beta).is_none_or(num_traits::Zero::is_zero)
    {
        return None;
    }
    let i = arena.i_unit();
    let ib = arena.mul(&[i, beta]);
    let minus = arena.sub(alpha, ib);
    let plus = arena.add(&[alpha, ib]);
    let neg_s = arena.neg(s);
    let p_minus = arena.pow(minus, neg_s);
    let p_plus = arena.pow(plus, neg_s);
    let g = arena.gamma(s);
    let (bracket, lower) = if is_sin {
        let d = arena.sub(p_minus, p_plus);
        let two = arena.int(2);
        let two_i = arena.mul(&[two, i]);
        (arena.div(d, two_i), arena.neg_one())
    } else {
        let sum = arena.add(&[p_minus, p_plus]);
        let half = arena.rational(1, 2);
        (arena.mul(&[half, sum]), arena.zero())
    };
    let v = arena.mul(&[g, bracket]);
    let v = crate::transforms::eval::eval(arena, v);
    Some((v, Strip::above(lower)))
}

/// `M{R}(s)` for a rational `R` over ℚ with no pole on `(0, ∞)`, by the
/// keyhole contour: with `R = x^e·R̃`, `R̃(0)` finite and non-zero, `M{R}(s)
/// = M{R̃}(s + e)` and `M{R̃}(σ) = π/sin(πσ) · Σ Res[(−z)^{σ−1} R̃(z)]` over
/// all poles (principal power; `−z` is off the cut at every pole) on the
/// strip `0 < Re σ < deg D̃ − deg Ñ`, i.e. `−e < Re s < deg D − deg N`.
/// (`1/(1 + x)` gives `π/sin πs`; a double pole at `−1`, `1/(1 + x)²`, gives
/// `π(1 − s)/sin πs = B(s, 2 − s)`.)  Before 0.46 only the binomial forms
/// of the table were known: `1/(x² + x + 1)` and `1/((x + 1)(x + 2))` were
/// refused.  SymPy: `mellin_transform(1/((x + 1)*(x + 2)), x, s)` =
/// `(-2**s*pi/2 + pi)/sin(pi*s)` on `0 < re(s) < 2` — our sum over the poles
/// `−1`, `−2` is `π(1 − 2^{s−1})/sin πs`.
fn forward_rational(
    arena: &mut Arena,
    expr: ExprId,
    x: ExprId,
    s: ExprId,
) -> Option<(ExprId, Strip)> {
    use crate::calculus::residue::{self, RationalQ};
    let r = residue::rational_over_q(arena, expr, x)?;
    let gap = r.degree_gap();
    let zeros = |p: &crate::poly::Poly| {
        p.coeffs()
            .iter()
            .take_while(|c| num_traits::Zero::is_zero(*c))
            .count()
    };
    let (n0, m0) = (zeros(&r.numer), zeros(&r.denom));
    let strip_off = |p: &crate::poly::Poly, k: usize| {
        crate::poly::Poly::from_coeffs(p.coeffs().get(k..).unwrap_or(&[]).to_vec())
    };
    let reduced = RationalQ {
        numer: strip_off(&r.numer, n0),
        denom: strip_off(&r.denom, m0),
    };
    let e = i64::try_from(n0).ok()? - i64::try_from(m0).ok()?;
    if reduced.numer.is_zero() || reduced.degree_gap() < 1 || gap > i64::from(i32::MAX) {
        return None;
    }
    // No pole on (0, ∞) (0 itself is not a root of the reduced denominator).
    let bound = crate::poly::sturm::cauchy_bound(&reduced.denom);
    let (_, sqf) = reduced.denom.sqf_list();
    if sqf.iter().any(|(p, _)| {
        p.degree().is_some_and(|d| d > 0)
            && crate::poly::sturm::SturmChain::new(p)
                .count_roots_in_closed(&num_rational::Ratio::from_integer(0.into()), &bound)
                > 0
    }) {
        return None;
    }
    let e_id = arena.int(e);
    let sigma = arena.add(&[s, e_id]);
    let one = arena.one();
    let sigma_m1 = arena.sub(sigma, one);
    let neg_x = arena.neg(x);
    let weight = arena.pow(neg_x, sigma_m1);
    let sum = residue::rational_residue_sum(arena, &reduced, weight, x, |_| true)?;
    let pi = arena.pi();
    let pi_sigma = arena.mul(&[pi, sigma]);
    let sine = arena.sin(pi_sigma);
    let v = arena.mul(&[pi, sum]);
    let v = arena.div(v, sine);
    let v = crate::transforms::eval::eval(arena, v);
    let lower = arena.int(-e);
    let upper = arena.int(gap);
    Some((v, Strip::between(lower, upper)))
}

fn forward_table(
    arena: &mut Arena,
    expr: ExprId,
    node: &ExprNode,
    x: ExprId,
    s: ExprId,
) -> Result<Option<(ExprId, Strip)>, SymplexError> {
    const OP: &str = "mellin_transform";
    let zero = arena.zero();
    let one = arena.one();
    let two = arena.int(2);
    let pi = arena.pi();
    let neg_one = arena.neg_one();
    match node.clone() {
        // e^{−c x^b} → c^{−s/b} Γ(s/b)/b, Re s > 0 (b > 0)
        ExprNode::Exp(arg) => {
            let Some((c, b)) = neg_coeff_power(arena, arg, x) else {
                return Ok(None);
            };
            match param_sign(arena, c) {
                Some(1) => {}
                Some(_) => return Err(fail(OP, "exp(c·x^b) with c ≥ 0 has no Mellin transform")),
                None => return Err(need_sign(OP, arena, c, "c > 0 in exp(-c x^b)")),
            }
            let bs = match param_sign(arena, b) {
                Some(sg) if sg != 0 => sg,
                Some(_) => return Ok(None),
                None => return Err(need_sign(OP, arena, b, "the sign of b in exp(-c x^b)")),
            };
            let s_over_b = arena.div(s, b);
            let g = arena.gamma(s_over_b);
            let neg_s_over_b = arena.neg(s_over_b);
            let c_pow = if c == one {
                one
            } else {
                arena.pow(c, neg_s_over_b)
            };
            let abs_b = if bs > 0 { b } else { arena.neg(b) };
            let num = arena.mul(&[c_pow, g]);
            let r = arena.div(num, abs_b);
            let strip = if bs > 0 {
                Strip::above(zero)
            } else {
                Strip::below(zero)
            };
            Ok(Some((r, strip)))
        }
        // 1/(1+x)^ν → B(s, ν − s), 0 < Re s < ν
        ExprNode::Pow(..) => {
            let Some(nu) = one_plus_x_power(arena, expr, x) else {
                return Ok(None);
            };
            match param_sign(arena, nu) {
                Some(1) => {}
                Some(_) => return Err(fail(OP, "(1+x)^k with k ≥ 0 has no Mellin transform")),
                None => return Err(need_sign(OP, arena, nu, "ν > 0 in (1+x)^(-ν)")),
            }
            if nu == one {
                let pis = arena.mul(&[pi, s]);
                let sn = arena.sin(pis);
                let r = arena.div(pi, sn);
                return Ok(Some((r, Strip::between(zero, one))));
            }
            let nu_minus_s = arena.sub(nu, s);
            let b = arena.beta(s, nu_minus_s);
            Ok(Some((b, Strip::between(zero, nu))))
        }
        // H(c − x) → c^s/s (Re s > 0);  H(x − c) → −c^s/s (Re s < 0), c > 0
        // (before 0.31 only c = 1; `H(x − 5/2)` was refused).
        ExprNode::Heaviside(arg) => {
            let Some(Linear { a, b }) = linear_in(arena, arg, x) else {
                return Ok(None);
            };
            let ratio = arena.div(b, a);
            let ratio = arena.neg(ratio);
            let ratio = crate::transforms::eval::eval(arena, ratio);
            if param_sign(arena, ratio) != Some(1) {
                return Ok(None);
            }
            let cs = arena.pow(ratio, s);
            let c_over_s = arena.div(cs, s);
            match param_sign(arena, a) {
                Some(sg) if sg < 0 => Ok(Some((c_over_s, Strip::above(zero)))),
                Some(sg) if sg > 0 => Ok(Some((arena.neg(c_over_s), Strip::below(zero)))),
                _ => Ok(None),
            }
        }
        // sin x → Γ(s) sin(πs/2), −1 < Re s < 1;  cos x → Γ(s) cos(πs/2), 0 < Re s < 1
        ExprNode::Sin(arg) | ExprNode::Cos(arg) => {
            if arg != x {
                return Ok(None); // sin(a x): scaling rule
            }
            let g = arena.gamma(s);
            let pis = arena.mul(&[pi, s]);
            let half = arena.div(pis, two);
            if matches!(node, ExprNode::Sin(_)) {
                let t = arena.sin(half);
                Ok(Some((arena.mul(&[g, t]), Strip::between(neg_one, one))))
            } else {
                let t = arena.cos(half);
                Ok(Some((arena.mul(&[g, t]), Strip::between(zero, one))))
            }
        }
        // ln(1 + x) → π/(s sin(πs)), −1 < Re s < 0
        ExprNode::Ln(arg) => {
            let one_plus_x = arena.add(&[one, x]);
            if arg != one_plus_x {
                return Ok(None);
            }
            let pis = arena.mul(&[pi, s]);
            let sn = arena.sin(pis);
            let den = arena.mul(&[s, sn]);
            let r = arena.div(pi, den);
            Ok(Some((r, Strip::between(neg_one, zero))))
        }
        _ => Ok(None),
    }
}

/// `H(1 − x)·(1 − x)^b → B(s, b + 1)`, `H(1 − x)·g(x)` and `H(x − 1)·g(x)`
/// for `g` a power of `x` (handled by the power rule after peeling the step).
fn forward_step_product(
    arena: &mut Arena,
    factors: &[ExprId],
    x: ExprId,
    s: ExprId,
    depth: usize,
) -> Result<Option<(ExprId, Strip)>, SymplexError> {
    const OP: &str = "mellin_transform";
    let one = arena.one();
    let zero = arena.zero();
    let one_minus_x = arena.sub(one, x);
    // Some((true, H)) = H(c − x), Some((false, H)) = H(x − c), for a
    // threshold c > 0 (c = 1 only before 0.31).
    let mut step: Option<(bool, ExprId)> = None;
    let mut rest: Vec<ExprId> = Vec::new();
    for &f in factors {
        if step.is_none()
            && let ExprNode::Heaviside(arg) = arena.node(f).clone()
            && let Some(Linear { a, b }) = linear_in(arena, arg, x)
            && let Some(sa) = param_sign(arena, a).filter(|&v| v != 0)
        {
            let ratio = arena.div(b, a);
            let ratio = arena.neg(ratio);
            let ratio = crate::transforms::eval::eval(arena, ratio);
            if param_sign(arena, ratio) == Some(1) {
                step = Some((sa < 0, f));
                continue;
            }
        }
        rest.push(f);
    }
    let Some((inside, step_expr)) = step else {
        return Ok(None);
    };
    let unit_step = arena.node(step_expr).clone() == ExprNode::Heaviside(one_minus_x);
    if rest.len() == 1 && inside && unit_step {
        // (1 − x)^b → B(s, b + 1), Re s > 0, b > −1
        if let ExprNode::Pow(base, b) = arena.node(rest[0]).clone()
            && base == one_minus_x
            && !depends_on(arena, b, x)
        {
            let b_plus_1 = arena.add(&[b, one]);
            match param_sign(arena, b_plus_1) {
                Some(1) => {}
                Some(_) => return Err(fail(OP, "(1 − x)^b with b ≤ −1 is not integrable at 1")),
                None => return Err(need_sign(OP, arena, b_plus_1, "b > −1 in (1 − x)^b")),
            }
            let r = arena.beta(s, b_plus_1);
            return Ok(Some((r, Strip::above(zero))));
        }
    }
    if rest.is_empty() {
        return Ok(None); // plain step: table
    }
    // H·x^a: peel the powers, transform the bare step, shift.
    let mut a_total: Option<ExprId> = None;
    for &r in &rest {
        let Some(e) = power_of_var(arena, r, x) else {
            return Ok(None);
        };
        a_total = Some(match a_total {
            Some(t) => arena.add(&[t, e]),
            None => e,
        });
    }
    let a = a_total.unwrap_or(zero);
    let (sf, st) = forward(arena, step_expr, x, s, depth + 1)?;
    let s_plus_a = arena.add(&[s, a]);
    let shifted = crate::transforms::subs::subs(arena, sf, s, s_plus_a);
    Ok(Some((shifted, st.shifted(arena, a))))
}

/// `f(a·x) → a^{−s} F(s)` and `f(x^b) → F(s/b)/|b|`, tried on the innermost
/// `a·x` / `x^b` sub-expression.
fn forward_substitution_rules(
    arena: &mut Arena,
    expr: ExprId,
    x: ExprId,
    s: ExprId,
    depth: usize,
) -> Result<Option<(ExprId, Strip)>, SymplexError> {
    let mut scales: Vec<ExprId> = Vec::new();
    let mut powers: Vec<ExprId> = Vec::new();
    let mut stack = vec![expr];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let node = arena.node(id);
        match node {
            ExprNode::Mul(ch) if ch.contains(&x) => {
                let rest_const = ch
                    .iter()
                    .filter(|&&c| c != x)
                    .all(|&c| !depends_on(arena, c, x));
                if rest_const && ch.len() > 1 {
                    scales.push(id);
                }
            }
            ExprNode::Pow(base, e) if *base == x && !depends_on(arena, *e, x) => {
                powers.push(*e);
            }
            _ => {}
        }
        for_each_child_in_scope(arena, id, x, |c| stack.push(c));
    }

    for mul_id in scales.into_iter().take(3) {
        let Some(Linear { a, .. }) = linear_in(arena, mul_id, x) else {
            continue;
        };
        match param_sign(arena, a) {
            Some(1) => {}
            _ => continue,
        }
        let x_over_a = arena.div(x, a);
        let g = crate::transforms::subs::subs(arena, expr, x, x_over_a);
        let g = crate::transforms::eval::eval(arena, g);
        if g == expr {
            continue;
        }
        if let Ok((gf, st)) = forward(arena, g, x, s, depth + 1) {
            let neg_s = arena.neg(s);
            let a_pow = arena.pow(a, neg_s);
            return Ok(Some((arena.mul(&[a_pow, gf]), st)));
        }
    }

    for b in powers.into_iter().take(3) {
        let bs = match param_sign(arena, b) {
            Some(sg) if sg != 0 => sg,
            _ => continue,
        };
        if b == arena.one() {
            continue;
        }
        let one = arena.one();
        let inv_b = arena.div(one, b);
        let x_root = arena.pow(x, inv_b);
        let g = crate::transforms::subs::subs(arena, expr, x, x_root);
        // (x^{1/b})^b → x is valid for x > 0 (the Mellin domain), hence `force`.
        let g = crate::simplify::powsimp::powdenest_with(arena, g, true);
        let g = crate::transforms::eval::eval(arena, g);
        if g == expr {
            continue;
        }
        if let Ok((gf, st)) = forward(arena, g, x, s, depth + 1) {
            let s_over_b = arena.div(s, b);
            let scaled = crate::transforms::subs::subs(arena, gf, s, s_over_b);
            let abs_b = if bs > 0 { b } else { arena.neg(b) };
            let r = arena.div(scaled, abs_b);
            return Ok(Some((r, st.scaled(arena, b, bs > 0))));
        }
    }
    Ok(None)
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse transform
// ═══════════════════════════════════════════════════════════════════════════

fn inverse(
    arena: &mut Arena,
    expr: ExprId,
    s: ExprId,
    x: ExprId,
    depth: usize,
    want: Option<Strip>,
) -> Result<ExprId, SymplexError> {
    const OP: &str = "inverse_mellin_transform";
    if depth > MAX_RULE_DEPTH {
        return Err(fail(OP, "rule nesting too deep"));
    }
    let expr = crate::transforms::eval::eval(arena, expr);
    let node = arena.node(expr).clone();

    if let ExprNode::Add(children) = &node {
        let kids = children.clone();
        let mut terms = Vec::with_capacity(kids.len());
        for child in kids {
            terms.push(inverse(arena, child, s, x, depth, want)?);
        }
        return Ok(arena.add(&terms));
    }
    if let ExprNode::Neg(inner) = node {
        let f = inverse(arena, inner, s, x, depth, want)?;
        return Ok(arena.neg(f));
    }
    if !depends_on(arena, expr, s) {
        return Err(fail(
            OP,
            "a constant is not the Mellin transform of a function",
        ));
    }

    // Constant factors and scaling factors a^{±s}.
    let Factors { consts, deps } = split_factors(arena, expr, s);
    let mut scale: Option<(ExprId, bool)> = None; // (a, is a^{-s})
    let mut body_factors: Vec<ExprId> = Vec::new();
    for &d in &deps {
        if scale.is_none()
            && let ExprNode::Pow(base, e) = arena.node(d).clone()
            && !depends_on(arena, base, s)
            && let Some(Linear { a: k, b: b0 }) = linear_in(arena, e, s)
            && arena.is_zero_structural(b0)
            && (k == arena.one() || k == arena.neg_one())
            && param_sign(arena, base) == Some(1)
        {
            scale = Some((base, k == arena.neg_one()));
        } else {
            body_factors.push(d);
        }
    }
    if !consts.is_empty() || scale.is_some() {
        if body_factors.is_empty() {
            return Err(fail(OP, "a pure exponential in s is not in the table"));
        }
        let body = arena.mul(&body_factors);
        // A constant factor and `a^{±s}` (entire, without zeros) leave the
        // strip where it is.
        let mut f = inverse(arena, body, s, x, depth + 1, want)?;
        if let Some((a, is_neg)) = scale {
            // a^{−s} F(s) ↔ f(a x);  a^{s} F(s) ↔ f(x/a)
            let arg = if is_neg {
                arena.mul(&[a, x])
            } else {
                arena.div(x, a)
            };
            f = crate::transforms::subs::subs(arena, f, x, arg);
        }
        let mut all = consts;
        all.push(f);
        return Ok(arena.mul(&all));
    }

    if let Some(r) = inverse_table(arena, expr, &node, s, x, want)? {
        return Ok(r);
    }

    // Shift rule: F(s) = G(s + a) ↔ f = x^a g(x).
    if let Some(r) = inverse_shift(arena, expr, s, x, depth, want)? {
        return Ok(r);
    }

    Err(fail(
        OP,
        format!("no inverse rule applies to {}", arena.display(expr)),
    ))
}

fn inverse_table(
    arena: &mut Arena,
    expr: ExprId,
    node: &ExprNode,
    s: ExprId,
    x: ExprId,
    want: Option<Strip>,
) -> Result<Option<ExprId>, SymplexError> {
    let one = arena.one();
    let two = arena.int(2);
    let pi = arena.pi();
    let zero = arena.zero;
    let minus_one = arena.neg_one();
    // The poles `0, −1, −2, …` and `1, 2, 3, …`.
    let left_from_zero = PoleFamily {
        start: zero,
        step: Some(minus_one),
    };
    let right_from_one = PoleFamily {
        start: one,
        step: Some(one),
    };
    match node.clone() {
        // Γ(s/b) → b·e^{−x^b};  Γ(s) → e^{−x}
        ExprNode::Gamma(arg) => {
            let Some(Linear { a: k, b: b0 }) = linear_in(arena, arg, s) else {
                return Ok(None);
            };
            if !arena.is_zero_structural(b0) {
                return Ok(None); // Γ(s + a): shift rule
            }
            // arg = k·s = s/b with b = 1/k
            let b = arena.div(one, k);
            let b = crate::transforms::eval::eval(arena, b);
            let xb = arena.pow(x, b);
            let nxb = arena.neg(xb);
            let e = arena.exp(nxb);
            let abs_b = match param_sign(arena, b) {
                Some(sg) if sg > 0 => b,
                Some(sg) if sg < 0 => arena.neg(b),
                _ => return Ok(None),
            };
            let f0 = arena.mul(&[abs_b, e]);
            // `Γ(s/b)`: the transform on `Re s > 0` for `b > 0` and on `Re
            // s < 0` for `b < 0`; poles at `s = −n·b`.
            let home = if abs_b == b {
                Strip::above(zero)
            } else {
                Strip::below(zero)
            };
            let step = arena.neg(b);
            let poles = [PoleFamily {
                start: zero,
                step: Some(step),
            }];
            table_entry(arena, expr, f0, home, &poles, want, s, x)
        }
        // B(s, ν − s) → (1 + x)^{−ν};  B(s, b + 1) → H(1 − x)(1 − x)^b
        ExprNode::Beta(p, q) => {
            let (p, q) = if p == s { (p, q) } else { (q, p) };
            if p != s {
                return Ok(None);
            }
            if let Some(Linear { a: k, b: c }) = linear_in(arena, q, s) {
                if k == arena.neg_one() {
                    // q = ν − s: on `0 < Re s < ν`; poles `−n` and `ν + n`.
                    let one_plus_x = arena.add(&[one, x]);
                    let neg_nu = arena.neg(c);
                    let f0 = arena.pow(one_plus_x, neg_nu);
                    let poles = [
                        left_from_zero,
                        PoleFamily {
                            start: c,
                            step: Some(one),
                        },
                    ];
                    return table_entry(
                        arena,
                        expr,
                        f0,
                        Strip::between(zero, c),
                        &poles,
                        want,
                        s,
                        x,
                    );
                }
                if arena.is_zero_structural(k) {
                    // q = b + 1: on `Re s > 0`; poles `−n`.
                    let b = arena.sub(c, one);
                    let one_minus_x = arena.sub(one, x);
                    let h = arena.heaviside(one_minus_x);
                    let p = arena.pow(one_minus_x, b);
                    let f0 = arena.mul(&[h, p]);
                    return table_entry(
                        arena,
                        expr,
                        f0,
                        Strip::above(zero),
                        &[left_from_zero],
                        want,
                        s,
                        x,
                    );
                }
            }
            Ok(None)
        }
        // 1/sin(πs) → 1/(π(1 + x))
        ExprNode::Pow(base, e)
            if e == arena.neg_one() && matches!(arena.node(base), ExprNode::Sin(_)) =>
        {
            let ExprNode::Sin(arg) = arena.node(base).clone() else {
                return Ok(None);
            };
            let pis = arena.mul(&[pi, s]);
            if arg != pis {
                return Ok(None);
            }
            let one_plus_x = arena.add(&[one, x]);
            let den = arena.mul(&[pi, one_plus_x]);
            let f0 = arena.div(one, den);
            // On `0 < Re s < 1`; poles at the integers.
            table_entry(
                arena,
                expr,
                f0,
                Strip::between(zero, one),
                &[left_from_zero, right_from_one],
                want,
                s,
                x,
            )
        }
        // 1/(s + a) → x^a H(1 − x) on `Re s > −a`, −x^a H(x − 1) on `Re s <
        // −a`; without a strip the first (the strip right of the pole).
        ExprNode::Pow(base, e) if e == arena.neg_one() => {
            let Some(Linear { a: k, b: a }) = linear_in(arena, base, s) else {
                return Ok(None);
            };
            if k != one {
                return Ok(None);
            }
            let pole = arena.neg(a);
            let pole = crate::transforms::eval::eval(arena, pole);
            let right = want.is_none_or(|w| w.within(arena, Strip::above(pole)));
            let step_arg = if right {
                arena.sub(one, x)
            } else if want.is_some_and(|w| w.within(arena, Strip::below(pole))) {
                arena.sub(x, one)
            } else {
                return Err(fail(
                    "inverse_mellin_transform_in_strip",
                    "the pole of the transform lies inside the strip, or cannot be placed \
                     relative to it",
                ));
            };
            let h = arena.heaviside(step_arg);
            let h = if right { h } else { arena.neg(h) };
            if arena.is_zero_structural(a) {
                return Ok(Some(h));
            }
            let xa = arena.pow(x, a);
            Ok(Some(arena.mul(&[xa, h])))
        }
        ExprNode::Mul(children) => {
            let kids: Vec<ExprId> = children.iter().copied().collect();
            // Γ(s)·sin(πs/2) → sin x;  Γ(s)·cos(πs/2) → cos x
            let gamma_s = arena.gamma(s);
            if kids.len() == 2 && kids.contains(&gamma_s) {
                let other = if kids[0] == gamma_s { kids[1] } else { kids[0] };
                let pis = arena.mul(&[pi, s]);
                let half = arena.div(pis, two);
                let sin_half = arena.sin(half);
                let cos_half = arena.cos(half);
                if other == sin_half {
                    // On `−1 < Re s < 1`; poles `−1, −3, …` (the zeros of
                    // the sine cancel the even ones).
                    let f0 = arena.sin(x);
                    let minus_two = arena.int(-2);
                    let poles = [PoleFamily {
                        start: minus_one,
                        step: Some(minus_two),
                    }];
                    return table_entry(
                        arena,
                        expr,
                        f0,
                        Strip::between(minus_one, one),
                        &poles,
                        want,
                        s,
                        x,
                    );
                }
                if other == cos_half {
                    // On `0 < Re s < 1`; poles `0, −2, −4, …`.
                    let f0 = arena.cos(x);
                    let minus_two = arena.int(-2);
                    let poles = [PoleFamily {
                        start: zero,
                        step: Some(minus_two),
                    }];
                    return table_entry(
                        arena,
                        expr,
                        f0,
                        Strip::between(zero, one),
                        &poles,
                        want,
                        s,
                        x,
                    );
                }
            }
            // π/sin(πs) → 1/(1 + x);   π/(s sin(πs)) → ln(1 + x)
            let pis = arena.mul(&[pi, s]);
            let sin_pis = arena.sin(pis);
            let inv_sin = arena.pow(sin_pis, arena.neg_one());
            let inv_s = arena.pow(s, arena.neg_one());
            let mut has_pi = false;
            let mut has_inv_sin = false;
            let mut has_inv_s = false;
            let mut other = false;
            for &k in &kids {
                if k == pi {
                    has_pi = true;
                } else if k == inv_sin {
                    has_inv_sin = true;
                } else if k == inv_s {
                    has_inv_s = true;
                } else {
                    other = true;
                }
            }
            if has_inv_sin && !other {
                // π may already have been split off as a constant factor.
                let one_plus_x = arena.add(&[one, x]);
                let base = if has_inv_s {
                    arena.ln(one_plus_x)
                } else {
                    arena.div(one, one_plus_x)
                };
                let f0 = if has_pi { base } else { arena.div(base, pi) };
                // `1/(1 + x)` on `0 < Re s < 1`, `ln(1 + x)` on `−1 < Re s <
                // 0`; poles at the integers.
                let (home, poles) = if has_inv_s {
                    (
                        Strip::between(minus_one, zero),
                        [
                            PoleFamily {
                                start: minus_one,
                                step: Some(minus_one),
                            },
                            PoleFamily {
                                start: zero,
                                step: Some(one),
                            },
                        ],
                    )
                } else {
                    (Strip::between(zero, one), [left_from_zero, right_from_one])
                };
                return table_entry(arena, expr, f0, home, &poles, want, s, x);
            }
            // Γ(s)Γ(ν − s)/Γ(ν) written out → (1 + x)^{−ν}
            let mut gamma_args: Vec<(ExprId, bool)> = Vec::new(); // (arg, inverted)
            let mut rest = false;
            for &k in &kids {
                match arena.node(k).clone() {
                    ExprNode::Gamma(a) => gamma_args.push((a, false)),
                    ExprNode::Pow(b, e) if e == arena.neg_one() => match arena.node(b).clone() {
                        ExprNode::Gamma(a) => gamma_args.push((a, true)),
                        _ => rest = true,
                    },
                    _ => rest = true,
                }
            }
            // Γ(s)Γ(ν − s)/Γ(ν) → (1 + x)^{−ν}; without the 1/Γ(ν) factor (it may
            // have been evaluated to a number and split off) → Γ(ν)(1 + x)^{−ν}.
            if !rest && (gamma_args.len() == 2 || gamma_args.len() == 3) {
                let plain: Vec<ExprId> = gamma_args.iter().filter(|g| !g.1).map(|g| g.0).collect();
                let inv: Vec<ExprId> = gamma_args.iter().filter(|g| g.1).map(|g| g.0).collect();
                if plain.len() == 2 && inv.len() <= 1 && plain.contains(&s) {
                    let other = if plain[0] == s { plain[1] } else { plain[0] };
                    if let Some(Linear { a: k, b: nu }) = linear_in(arena, other, s)
                        && k == arena.neg_one()
                        && (inv.is_empty() || inv[0] == nu)
                    {
                        let one_plus_x = arena.add(&[one, x]);
                        let neg_nu = arena.neg(nu);
                        let p = arena.pow(one_plus_x, neg_nu);
                        let f0 = if inv.is_empty() {
                            let g_nu = arena.gamma(nu);
                            arena.mul(&[g_nu, p])
                        } else {
                            p
                        };
                        // On `0 < Re s < ν`; poles `−n` and `ν + n`.
                        let poles = [
                            left_from_zero,
                            PoleFamily {
                                start: nu,
                                step: Some(one),
                            },
                        ];
                        return table_entry(
                            arena,
                            expr,
                            f0,
                            Strip::between(zero, nu),
                            &poles,
                            want,
                            s,
                            x,
                        );
                    }
                }
            }
            let _ = expr;
            Ok(None)
        }
        _ => Ok(None),
    }
}

/// `F(s) = G(s + a)` for some sub-expression `s + a` → `x^a · M⁻¹{G}`.
fn inverse_shift(
    arena: &mut Arena,
    expr: ExprId,
    s: ExprId,
    x: ExprId,
    depth: usize,
    want: Option<Strip>,
) -> Result<Option<ExprId>, SymplexError> {
    let mut shifts: Vec<ExprId> = Vec::new();
    let mut stack = vec![expr];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let node = arena.node(id);
        if let ExprNode::Add(ch) = node
            && ch.contains(&s)
        {
            let rest_const = ch
                .iter()
                .filter(|&&c| c != s)
                .all(|&c| !depends_on(arena, c, s));
            if rest_const && ch.len() > 1 {
                shifts.push(id);
            }
        }
        for_each_child_in_scope(arena, id, s, |c| stack.push(c));
    }
    for add_id in shifts.into_iter().take(3) {
        let Some(Linear { a: k, b: a }) = linear_in(arena, add_id, s) else {
            continue;
        };
        if k != arena.one() {
            continue;
        }
        // expr = G(s + a) → G(u) = expr|_{s = u − a}
        let s_minus_a = arena.sub(s, a);
        let g = crate::transforms::subs::subs(arena, expr, s, s_minus_a);
        let g = crate::transforms::eval::eval(arena, g);
        if g == expr {
            continue;
        }
        // `F(s) = G(s + a)` on a strip is `G(u)` on that strip moved by `a`.
        let want_g = want.map(|w| {
            let neg_a = arena.neg(a);
            w.shifted(arena, neg_a)
        });
        if let Ok(gi) = inverse(arena, g, s, x, depth + 1, want_g) {
            let xa = arena.pow(x, a);
            return Ok(Some(arena.mul(&[xa, gi])));
        }
    }
    let _: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    Ok(None)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn exp_gives_gamma_with_right_half_plane() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let s = a.symbol("s");
        let nx = a.neg(x);
        let e = a.exp(nx);
        let (f, cond) = mellin_transform(&mut a, e, x, s).unwrap();
        let gamma_s = a.gamma(s);
        assert_eq!(f, gamma_s);
        assert_eq!(display(&a, cond), "re(s) > 0");
    }

    #[test]
    fn one_over_one_plus_x() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let s = a.symbol("s");
        let one = a.one();
        let d = a.add(&[one, x]);
        let e = a.div(one, d);
        let (f, cond) = mellin_transform(&mut a, e, x, s).unwrap();
        let pi = a.pi();
        let pis = a.mul(&[pi, s]);
        let sn = a.sin(pis);
        let expected = a.div(pi, sn);
        assert_eq!(f, expected, "{}", display(&a, f));
        let c = display(&a, cond);
        assert!(c.contains("re(s) > 0") && c.contains("1 > re(s)"), "{c}");
    }

    #[test]
    fn variables_must_be_distinct_symbols() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let e = a.exp(x);
        assert!(matches!(
            mellin_transform(&mut a, e, x, x),
            Err(SymplexError::InvalidArgument { .. })
        ));
        let two = a.int(2);
        assert!(matches!(
            inverse_mellin_transform(&mut a, e, two, x),
            Err(SymplexError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn inverse_of_gamma_is_exp() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let s = a.symbol("s");
        let g = a.gamma(s);
        let f = inverse_mellin_transform(&mut a, g, s, x).unwrap();
        let nx = a.neg(x);
        let e = a.exp(nx);
        assert_eq!(f, e);
    }
}
