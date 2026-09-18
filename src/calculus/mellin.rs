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

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};

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

fn fail(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::ComputationFailed {
        operation,
        reason: reason.into(),
    }
}

fn param_sign(arena: &mut Arena, e: ExprId) -> Option<i32> {
    crate::calculus::limit::const_sign(arena, e)
}

fn need_sign(op: &'static str, arena: &Arena, e: ExprId, cond: &str) -> SymplexError {
    fail(
        op,
        format!(
            "requires {cond} (declare the sign of {} with Assumption::Positive / Assumption::Negative)",
            arena.display(e)
        ),
    )
}

fn require_symbol(
    arena: &Arena,
    id: ExprId,
    operation: &'static str,
    what: &str,
) -> Result<(), SymplexError> {
    if matches!(arena.node(id), ExprNode::Symbol(_)) {
        Ok(())
    } else {
        Err(SymplexError::InvalidArgument {
            operation,
            reason: format!("{what} must be a symbol, got {}", arena.display(id)),
        })
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
    let (f, strip) = forward(arena, expr, x, s, 0)?;
    let f = crate::transforms::eval::eval(arena, f);
    if crate::base::walk::contains(arena, f, x) || crate::base::walk::has_unevaluated(arena, f) {
        return Err(fail(OP, "result still depends on the variable"));
    }
    let cond = strip.condition(arena, s);
    Ok((f, cond))
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
    let f = inverse(arena, expr, s, x, 0)?;
    let f = crate::transforms::eval::eval(arena, f);
    if crate::base::walk::contains(arena, f, s) || crate::base::walk::has_unevaluated(arena, f) {
        return Err(fail(OP, "result still depends on the transform variable"));
    }
    Ok(f)
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Split a product into `(constant factors, v-dependent factors)`.
fn split_factors(arena: &Arena, expr: ExprId, v: ExprId) -> (Vec<ExprId>, Vec<ExprId>) {
    let children: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Mul(ch) => ch.iter().copied().collect(),
        _ => vec![expr],
    };
    let mut consts = Vec::new();
    let mut deps = Vec::new();
    for c in children {
        if crate::base::walk::contains(arena, c, v) {
            deps.push(c);
        } else {
            consts.push(c);
        }
    }
    (consts, deps)
}

/// `expr = a·v + b` with `a`, `b` free of `v`.
fn linear_in(arena: &mut Arena, expr: ExprId, v: ExprId) -> Option<(ExprId, ExprId)> {
    if expr == v {
        return Some((arena.one(), arena.zero()));
    }
    if !crate::base::walk::contains(arena, expr, v) {
        return Some((arena.zero(), expr));
    }
    match arena.node(expr).clone() {
        ExprNode::Neg(inner) => {
            let (a, b) = linear_in(arena, inner, v)?;
            Some((arena.neg(a), arena.neg(b)))
        }
        ExprNode::Mul(children) => {
            let mut coeff: SmallVec<[ExprId; 4]> = SmallVec::new();
            let mut seen = false;
            for &c in &children {
                if c == v {
                    if seen {
                        return None;
                    }
                    seen = true;
                } else if crate::base::walk::contains(arena, c, v) {
                    return None;
                } else {
                    coeff.push(c);
                }
            }
            if !seen {
                return None;
            }
            let a = if coeff.is_empty() {
                arena.one()
            } else {
                arena.mul(&coeff)
            };
            Some((a, arena.zero()))
        }
        ExprNode::Add(children) => {
            let mut a_terms: SmallVec<[ExprId; 4]> = SmallVec::new();
            let mut b_terms: SmallVec<[ExprId; 4]> = SmallVec::new();
            for &c in &children {
                let (a, b) = linear_in(arena, c, v)?;
                if !arena.is_zero_structural(a) {
                    a_terms.push(a);
                }
                if !arena.is_zero_structural(b) {
                    b_terms.push(b);
                }
            }
            let a = arena.add(&a_terms);
            let b = arena.add(&b_terms);
            Some((a, b))
        }
        _ => None,
    }
}

/// `expr = x^k` (any constant exponent `k`, including `x` itself) → `Some(k)`.
fn power_of_var(arena: &Arena, expr: ExprId, v: ExprId) -> Option<ExprId> {
    if expr == v {
        return Some(arena.one());
    }
    if let ExprNode::Pow(base, e) = arena.node(expr)
        && *base == v
        && !crate::base::walk::contains(arena, *e, v)
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
        && !crate::base::walk::contains(arena, e, x)
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
                if crate::base::walk::contains(arena, c, x) {
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
    if !crate::base::walk::contains(arena, expr, x) {
        return Err(fail(
            OP,
            "a non-zero constant has no Mellin transform (the integral diverges)",
        ));
    }

    // Constant factors.
    let (consts, deps) = split_factors(arena, expr, x);
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
        && !crate::base::walk::contains(arena, k, x)
        && let ExprNode::Add(terms) = arena.node(base).clone()
    {
        let one = arena.one();
        let (consts, deps): (Vec<ExprId>, Vec<ExprId>) = terms
            .iter()
            .copied()
            .partition(|&t| !crate::base::walk::contains(arena, t, x));
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

    Err(fail(
        OP,
        format!("no transform rule applies to {}", arena.display(expr)),
    ))
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
        // H(1 − x) → 1/s (Re s > 0);  H(x − 1) → −1/s (Re s < 0)
        ExprNode::Heaviside(arg) => {
            let Some((a, b)) = linear_in(arena, arg, x) else {
                return Ok(None);
            };
            let ratio = arena.div(b, a);
            let ratio = arena.neg(ratio);
            let ratio = crate::transforms::eval::eval(arena, ratio);
            if ratio != one {
                return Ok(None); // H(c − x) with c ≠ 1: scaling rule
            }
            let inv_s = arena.div(one, s);
            match param_sign(arena, a) {
                Some(sg) if sg < 0 => Ok(Some((inv_s, Strip::above(zero)))),
                Some(sg) if sg > 0 => Ok(Some((arena.neg(inv_s), Strip::below(zero)))),
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
    let x_minus_one = arena.sub(x, one);
    let mut step: Option<bool> = None; // Some(true) = H(1 − x), Some(false) = H(x − 1)
    let mut rest: Vec<ExprId> = Vec::new();
    for &f in factors {
        match arena.node(f).clone() {
            ExprNode::Heaviside(arg) if arg == one_minus_x && step.is_none() => step = Some(true),
            ExprNode::Heaviside(arg) if arg == x_minus_one && step.is_none() => step = Some(false),
            _ => rest.push(f),
        }
    }
    let Some(inside) = step else {
        return Ok(None);
    };
    if rest.len() == 1 && inside {
        // (1 − x)^b → B(s, b + 1), Re s > 0, b > −1
        if let ExprNode::Pow(base, b) = arena.node(rest[0]).clone()
            && base == one_minus_x
            && !crate::base::walk::contains(arena, b, x)
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
    let step_expr = if inside {
        arena.heaviside(one_minus_x)
    } else {
        arena.heaviside(x_minus_one)
    };
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
                    .all(|&c| !crate::base::walk::contains(arena, c, x));
                if rest_const && ch.len() > 1 {
                    scales.push(id);
                }
            }
            ExprNode::Pow(base, e) if *base == x && !crate::base::walk::contains(arena, *e, x) => {
                powers.push(*e);
            }
            _ => {}
        }
        node.for_each_child(|c| stack.push(c));
    }

    for mul_id in scales.into_iter().take(3) {
        let Some((a, _)) = linear_in(arena, mul_id, x) else {
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
            terms.push(inverse(arena, child, s, x, depth)?);
        }
        return Ok(arena.add(&terms));
    }
    if let ExprNode::Neg(inner) = node {
        let f = inverse(arena, inner, s, x, depth)?;
        return Ok(arena.neg(f));
    }
    if !crate::base::walk::contains(arena, expr, s) {
        return Err(fail(
            OP,
            "a constant is not the Mellin transform of a function",
        ));
    }

    // Constant factors and scaling factors a^{±s}.
    let (consts, deps) = split_factors(arena, expr, s);
    let mut scale: Option<(ExprId, bool)> = None; // (a, is a^{-s})
    let mut body_factors: Vec<ExprId> = Vec::new();
    for &d in &deps {
        if scale.is_none()
            && let ExprNode::Pow(base, e) = arena.node(d).clone()
            && !crate::base::walk::contains(arena, base, s)
            && let Some((k, b0)) = linear_in(arena, e, s)
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
        let mut f = inverse(arena, body, s, x, depth + 1)?;
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

    if let Some(r) = inverse_table(arena, expr, &node, s, x)? {
        return Ok(r);
    }

    // Shift rule: F(s) = G(s + a) ↔ f = x^a g(x).
    if let Some(r) = inverse_shift(arena, expr, s, x, depth)? {
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
) -> Result<Option<ExprId>, SymplexError> {
    let one = arena.one();
    let two = arena.int(2);
    let pi = arena.pi();
    match node.clone() {
        // Γ(s/b) → b·e^{−x^b};  Γ(s) → e^{−x}
        ExprNode::Gamma(arg) => {
            let Some((k, b0)) = linear_in(arena, arg, s) else {
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
            Ok(Some(arena.mul(&[abs_b, e])))
        }
        // B(s, ν − s) → (1 + x)^{−ν};  B(s, b + 1) → H(1 − x)(1 − x)^b
        ExprNode::Beta(p, q) => {
            let (p, q) = if p == s { (p, q) } else { (q, p) };
            if p != s {
                return Ok(None);
            }
            if let Some((k, c)) = linear_in(arena, q, s) {
                if k == arena.neg_one() {
                    // q = ν − s
                    let one_plus_x = arena.add(&[one, x]);
                    let neg_nu = arena.neg(c);
                    return Ok(Some(arena.pow(one_plus_x, neg_nu)));
                }
                if arena.is_zero_structural(k) {
                    // q = b + 1
                    let b = arena.sub(c, one);
                    let one_minus_x = arena.sub(one, x);
                    let h = arena.heaviside(one_minus_x);
                    let p = arena.pow(one_minus_x, b);
                    return Ok(Some(arena.mul(&[h, p])));
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
            Ok(Some(arena.div(one, den)))
        }
        // 1/(s + a) → x^a H(1 − x)   (strip Re s > −a)
        ExprNode::Pow(base, e) if e == arena.neg_one() => {
            let Some((k, a)) = linear_in(arena, base, s) else {
                return Ok(None);
            };
            if k != one {
                return Ok(None);
            }
            let one_minus_x = arena.sub(one, x);
            let h = arena.heaviside(one_minus_x);
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
                    return Ok(Some(arena.sin(x)));
                }
                if other == cos_half {
                    return Ok(Some(arena.cos(x)));
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
                if has_pi {
                    return Ok(Some(base));
                }
                return Ok(Some(arena.div(base, pi)));
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
                    if let Some((k, nu)) = linear_in(arena, other, s)
                        && k == arena.neg_one()
                        && (inv.is_empty() || inv[0] == nu)
                    {
                        let one_plus_x = arena.add(&[one, x]);
                        let neg_nu = arena.neg(nu);
                        let p = arena.pow(one_plus_x, neg_nu);
                        if inv.is_empty() {
                            let g_nu = arena.gamma(nu);
                            return Ok(Some(arena.mul(&[g_nu, p])));
                        }
                        return Ok(Some(p));
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
                .all(|&c| !crate::base::walk::contains(arena, c, s));
            if rest_const && ch.len() > 1 {
                shifts.push(id);
            }
        }
        node.for_each_child(|c| stack.push(c));
    }
    for add_id in shifts.into_iter().take(3) {
        let Some((k, a)) = linear_in(arena, add_id, s) else {
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
        if let Ok(gi) = inverse(arena, g, s, x, depth + 1) {
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
