//! General u-substitution (part of [`integrate`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// General u-substitution
// ═══════════════════════════════════════════════════════════════════════════

/// Try general u-substitution on a product integrand.
///
/// For each dependent factor `g(u)` where `u = u(x)`, this checks
/// whether the remaining factors equal `du/dx` times a constant `c`.
/// When they do, `∫ c · (du/dx) · g(u) dx = c · G(u)` where `G` is
/// the antiderivative of `g` with respect to `u`.
///
/// The technique works by:
/// 1. Extracting candidate inner arguments from function / power nodes.
/// 2. Computing `du/dx` via [`crate::transforms::diff::diff`].
/// 3. Forming `remaining / du` and checking it is free of `var`.
/// 4. Substituting `u → var` in the factor, integrating, then
///    substituting back.
pub(super) fn try_u_substitution(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    for (i, &factor) in dependent.iter().enumerate() {
        let candidates = u_sub_candidates(arena, factor, var_sym);

        for u_expr in candidates {
            // Skip the trivial u = var case (already handled elsewhere)
            if u_expr == var {
                continue;
            }

            // Compute du/dx
            let du = crate::transforms::diff::diff(arena, u_expr, var);
            if du == arena.zero {
                continue;
            }

            // Product of the remaining dependent factors
            let remaining_expr = remaining_product(arena, dependent, i);

            // quotient = remaining / du — if free of var, we have our constant
            let quotient = arena.div(remaining_expr, du);

            // Try the raw quotient first; fall back to polynomial cancellation,
            // then to trigonometric simplification (e.g. sec²/(1 + tan²) = 1).
            // A quotient that depends on `var` only through `u` joins the
            // factor: `∫ g(u)·h(u)·u′ dx` (up to 0.31 only a constant
            // quotient counted, and `∫ cos³x·(a + b·cos²x)³·sin x dx` came
            // out of the Weierstrass substitution with `tan(x/2)⁴²`).
            let placeholder = arena.symbol("__usub_u");
            // `u = c·xᵏ + d`: every `xʲ` with `j/k` an integer is
            // `((u − d)/c)^{j/k}` (exact), so `x = 1/u` for `u = 1/x`.  `∫ e^{1/x}/x dx`
            // is `−∫ eᵘ/u du`, `∫ e^{x²}/x dx` is `∫ eᵘ/(2u) du` (unevaluated up to
            // 0.43; SymPy: `−Ei(1/x)`, `Ei(x²)/2`).  Only for a `c` known to be
            // non-zero: `∫ e^{a/x}/x dx` stays unevaluated for an unassumed `a`
            // (`−Ei(a/x)` is undefined at `a = 0`).
            let monomial_u = affine_monomial_parts(arena, u_expr, var, var_sym)
                .filter(|&(c, _, _)| constant_nonzero(arena, c));
            let inverse = monomial_u.is_some();
            let to_u = |arena: &mut Arena, e: ExprId| -> ExprId {
                let mut s = arena.subs_structural(e, u_expr, placeholder);
                let Some((c, k, d)) = monomial_u.as_ref() else {
                    return s;
                };
                if !contains_var(arena, s, var_sym) {
                    return s;
                }
                let u_minus_d = arena.sub(placeholder, *d);
                let u_over_c = arena.div(u_minus_d, *c);
                for id in crate::base::walk::post_order_ids(arena, s) {
                    if let ExprNode::Pow(b, j) = *arena.node(id)
                        && b == var
                        && let Some(r) = arena.as_num(j).map(|j| j / k)
                        && r.is_integer()
                    {
                        let r = arena.num_ratio(r);
                        let u_r = arena.pow(u_over_c, r);
                        s = arena.subs_structural(s, id, u_r);
                    }
                }
                let r = Q::one() / k;
                if r.is_integer() {
                    let r = arena.num_ratio(r);
                    let u_r = arena.pow(u_over_c, r);
                    s = arena.subs_structural(s, var, u_r);
                }
                s
            };
            let mut through_u: Option<ExprId> = None;
            let coeff = if !contains_var(arena, quotient, var_sym) {
                quotient
            } else {
                let cancelled = arena.cancel_expr(quotient, var);
                if !contains_var(arena, cancelled, var_sym) {
                    cancelled
                } else {
                    // No rewriting of `sin`, `cos`, … can make a quotient
                    // that provably depends on `var` constant.
                    let varies = var_outside_trig_arguments(arena, cancelled, var_sym)
                        || varies_numerically(arena, cancelled, var);
                    let trig = if varies {
                        cancelled
                    } else {
                        let trig = arena.trigsimp_expr(cancelled);
                        crate::transforms::eval::eval(arena, trig)
                    };
                    if !contains_var(arena, trig, var_sym) {
                        trig
                    } else {
                        // (For a linear `u` this is a renaming that adds
                        // nothing: every rule takes linear arguments.)
                        if !contains_var(arena, du, var_sym) {
                            continue;
                        }
                        let q_in_u = to_u(arena, quotient);
                        if contains_var(arena, q_in_u, var_sym) {
                            continue;
                        }
                        through_u = Some(quotient);
                        arena.one
                    }
                }
            };
            // `du` may be a constant that is zero without being so
            // structurally, and `trigsimp` shows it: `u = k·x + 1` with
            // `k = sin²1 + cos²1 − 1` gave the coefficient `zoo`, and
            // `∫ x/u dx = zoo` once the self-check stopped passing the
            // earlier candidate with the vanishing denominator `k`.
            if contains_non_finite(arena, coeff) {
                continue;
            }

            // The factor must depend on `var` only through `u`: with `u`
            // replaced by a fresh symbol nothing of `var` may remain
            // (otherwise `g(var)` below would silently absorb it).
            let factor = match through_u {
                Some(q) => arena.mul(&[factor, q]),
                None => factor,
            };
            let factor_in_u = to_u(arena, factor);
            if contains_var(arena, factor_in_u, var_sym) {
                continue;
            }

            // Replace u(x) → var inside the factor to get g(var),
            // integrate g(var) w.r.t. var, then substitute var → u(x) back.
            let g_of_var = if inverse {
                arena.subs_structural(factor_in_u, placeholder, var)
            } else {
                arena.subs_structural(factor, u_expr, var)
            };
            let g_integrated = integrate_node(arena, g_of_var, var, var_sym, depth);

            // If the inner integral is unevaluated, this candidate didn't help
            if matches!(arena.node(g_integrated), ExprNode::Integral(_, _)) {
                continue;
            }
            // `u = e^{λ·x + μ}` with `λ` not real runs along a spiral or a
            // circle through every direction: a logarithm (or another cut
            // function) of `u` jumps once per turn.  Up to 0.54 `∫
            // e^{(1+i)x}/(e^{(1+i)x} − 3) dx` was `ln(e^{(1+i)x} − 3)/(1 + i)`,
            // up to 0.55 `∫ e^{ix}/(e^{ix} − 3) dx` was `−i·ln(e^{ix} − 3)`,
            // which jumps at `x = π`; the exponential substitution keeps such
            // logarithms continuous ([`spiral_continuous`]).
            if has_cut_function(arena, g_integrated, var_sym)
                && crate::base::walk::post_order_ids(arena, u_expr)
                    .into_iter()
                    .any(|id| {
                        let ExprNode::Exp(l) = *arena.node(id) else {
                            return false;
                        };
                        contains_var(arena, l, var_sym)
                            && symbolic_linear_coeff_of(arena, l, var, var_sym)
                                .is_some_and(|(alpha, _)| off_a_ray(arena, alpha))
                    })
            {
                continue;
            }

            // G(u) — substitute var back to u(x)
            let antideriv = arena.subs_structural(g_integrated, var, u_expr);
            let antideriv = match continuous_through_tan_of_u(
                arena,
                g_integrated,
                antideriv,
                u_expr,
                var,
                var_sym,
            ) {
                Some(continuous) => continuous,
                None if jump_kept_undecided(arena, g_integrated, var_sym) => {
                    match tan_argument_of(arena, u_expr, var_sym) {
                        Some(w) => crate::transforms::trig_integ::atan_tan_terms_to_argument_in(
                            arena,
                            antideriv,
                            w,
                            Some(var_sym),
                        ),
                        None => antideriv,
                    }
                }
                None => continue,
            };
            tracing::debug!("u-substitution succeeded");
            return Some(arena.mul(&[coeff, antideriv]));
        }
    }
    None
}

/// The back-substituted `antideriv = G(u(x))` of a u-substitution, made
/// continuous across the poles of a `tan w` through which `u` depends on
/// `x` ([`crate::transforms::trig_integ::add_tan_floor`]).  `g` is `G` in
/// `var` (the stand-in for `u`).  Up to 0.37 `∫ tan x/(tan x + 2) dx` was
/// `… + 1/5·atan(tan x)`, which drops by `π/5` at every `x = π/2 + kπ`.
///
/// `antideriv` unchanged when `u` has no `tan` of `var`; `None` (the
/// candidate is not used) when `u` depends on `var` otherwise than through
/// one `tan w`, or the jump `G(U(+∞)) − G(U(−∞))` of `G(U(s))`, `u = U(tan w)`,
/// is not decided.
fn continuous_through_tan_of_u(
    arena: &mut Arena,
    g: ExprId,
    antideriv: ExprId,
    u_expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let Some(w) = tan_argument_of(arena, u_expr, var_sym) else {
        return Some(antideriv);
    };
    let tan_w = arena.tan(w);
    let s = arena.symbol("__tan_s");
    let ExprNode::Symbol(s_sym) = *arena.node(s) else {
        return None;
    };
    let u_of_s = arena.subs_structural(u_expr, tan_w, s);
    if contains_var(arena, u_of_s, var_sym) {
        return None;
    }
    let h = arena.subs_structural(g, var, u_of_s);
    let (antideriv, jump) = match crate::transforms::trig_integ::infinity_jump(arena, h, s, s_sym) {
        Some(jump) => (antideriv, jump),
        // Parameters not declared real: the jump as for real ones, of the
        // answer with the logarithms the stage exit writes `ln u` (up to
        // 0.37 the answer was kept uncorrected: `∫ tan x/(a + tan x) dx`
        // stepped by `iπ·a/(a² + 1)` at every pole of `tan x`).
        None => {
            let all = crate::base::walk::free_symbols(arena, w)
                .into_iter()
                .any(|v| match *arena.node(v) {
                    ExprNode::Symbol(sid) => {
                        sid != var_sym
                            && !crate::transforms::realness::symbol_declared_real(arena, sid)
                    }
                    _ => true,
                });
            let g = crate::transforms::trig_integ::plain_logs(arena, g, var_sym, all);
            let h = arena.subs_structural(g, var, u_of_s);
            let jump =
                crate::transforms::trig_integ::infinity_jump_real_parameters(arena, h, s, s_sym)?;
            (arena.subs_structural(g, var, u_expr), jump)
        }
    };
    Some(crate::transforms::trig_integ::add_tan_floor_in(
        arena,
        antideriv,
        w,
        jump,
        Some(var_sym),
    ))
}

/// The argument `w` of the first `tan w` (with `w` depending on `var`) in
/// `u` (post-order).
fn tan_argument_of(arena: &Arena, u: ExprId, var_sym: SymbolId) -> Option<ExprId> {
    crate::base::walk::post_order_ids(arena, u)
        .into_iter()
        .find_map(|id| match *arena.node(id) {
            ExprNode::Tan(w) if contains_var(arena, w, var_sym) => Some(w),
            _ => None,
        })
}

/// Does `var` occur in `q` outside the arguments of its trigonometric and
/// hyperbolic functions (`x²·sin x`, `eˣ/cos x`)?  `trigsimp` rewrites
/// those functions only, so it cannot make such a `q` free of `var`: on a
/// sample of the Rubi suite (0.31) none of 4,550 such quotients became
/// constant (the canonical form already cancels equal factors).
fn var_outside_trig_arguments(arena: &Arena, q: ExprId, var_sym: SymbolId) -> bool {
    let mut stack = vec![q];
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Tan(_)
            | ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Tanh(_) => {}
            ExprNode::Symbol(s) if *s == var_sym => return true,
            node => node.for_each_child(|c| stack.push(c)),
        }
    }
    false
}

/// Sample points for [`varies_numerically`].
const VARIES_POINTS: [(i64, i64); 2] = [(3, 7), (5, 11)];

/// Does `q` provably depend on `var`: does it take clearly different
/// values (to 15 digits) at two sample points, with its other free symbols
/// at the generic values of [`FTC_PARAMETER_VALUES`]?  `false` when it
/// does not evaluate at both points or the values agree.  No rewriting
/// can then make `q` free of `var`, and [`try_u_substitution`] skips
/// `trigsimp`, which tries seven strategies on each quotient and found a
/// constant in 1 of 9,500 calls on a sample of the Rubi suite (0.31;
/// `sin 2x/(8 sin x cos x) = 1/4`), while taking 50 % of the time of the
/// integrands that stay unevaluated.
///
/// Inside an `integrate` call the answer is remembered until the outermost
/// call ends ([`VARIES_MEMO`]).
fn varies_numerically(arena: &mut Arena, q: ExprId, var: ExprId) -> bool {
    let inside = INTEGRATE_DEPTH.with(std::cell::Cell::get) > 0;
    let key = (arena.serial(), q, var);
    if inside && let Some(found) = VARIES_MEMO.with(|m| m.borrow().get(&key).copied()) {
        return found;
    }
    let varies = varies_numerically_uncached(arena, q, var);
    if inside {
        VARIES_MEMO.with(|m| m.borrow_mut().insert(key, varies));
    }
    varies
}

thread_local! {
    /// [`varies_numerically`] during the current top-level `integrate`
    /// (cleared with [`INTEGRATE_MEMO`]), by (arena, quotient, variable).
    /// The u-substitution meets the same quotients for every degenerate
    /// case of the parameters and every depth it is retried at: on the Rubi
    /// suite's 4.5.4.2 (0.38) its 45,000 tests took 6.5 s, a fifth of the
    /// integration time, and 24,700 of them were repeats.
    pub(super) static VARIES_MEMO: std::cell::RefCell<FxHashMap<(u64, ExprId, ExprId), bool>> =
        std::cell::RefCell::new(FxHashMap::default());
}

/// [`varies_numerically`] without the memo.
fn varies_numerically_uncached(arena: &mut Arena, q: ExprId, var: ExprId) -> bool {
    let mut params: Vec<(String, ExprId)> = Vec::new();
    for s in crate::base::walk::free_symbols(arena, q) {
        if s == var {
            continue;
        }
        let ExprNode::Symbol(sid) = *arena.node(s) else {
            return false;
        };
        params.push((arena.symbol_name(sid).to_owned(), s));
    }
    if params.len() > FTC_PARAMETER_VALUES.len() {
        return false;
    }
    params.sort_by(|a, b| a.0.cmp(&b.0));
    let mut bound = q;
    for (k, &(_, s)) in params.iter().enumerate() {
        let (p, d) = FTC_PARAMETER_VALUES[k];
        let value = arena.rational(p, d);
        bound = crate::transforms::subs::subs(arena, bound, s, value);
    }
    let mut values = Vec::with_capacity(VARIES_POINTS.len());
    for &(p, d) in &VARIES_POINTS {
        let point = arena.rational(p, d);
        let at = crate::transforms::subs::subs(arena, bound, var, point);
        let at = crate::transforms::eval::eval(arena, at);
        if crate::base::walk::has_unevaluated(arena, at) {
            return false;
        }
        let Ok(v) = crate::transforms::evalf::evalf_complex(arena, at, 15) else {
            return false;
        };
        values.push(v);
    }
    let (Some(scale), Some(gap)) = (
        crate::transforms::evalf::abs_to_f64(&values[0]),
        crate::transforms::evalf::distance_to_f64(&values[0], &values[1]),
    ) else {
        return false;
    };
    // Only skips `trigsimp`: the coefficient is right either way.
    scale.is_finite() && gap.is_finite() && gap > 1e-8 * scale.max(1.0)
}

/// Collect candidate `u`-expressions from a single factor.
///
/// For function nodes (`sin`, `cos`, `exp`, …) the inner argument and the
/// node itself are returned.  For `Pow(base, exp)` the base is returned
/// (enabling e.g. `u = x² + 1` inside `(x²+1)^{-1}`), and for `Pow` and
/// `Add` factors also the function nodes found inside them (`u = sin x`
/// for `(sin²x + 1)^{-1}` or `sin x + 1`), so that `∫ cos x·R(sin x) dx`
/// is a substitution rather than a Weierstrass problem.
fn u_sub_candidates(arena: &Arena, factor: ExprId, var_sym: SymbolId) -> SmallVec<[ExprId; 4]> {
    let mut out: SmallVec<[ExprId; 4]> = SmallVec::new();
    let node = arena.node(factor).clone();
    // The elementary functions and `|·|` (`Func::elementary`).
    if let Some((f, args)) = node.func(arena)
        && (f.elementary().is_some() || f == crate::base::func::Func::Abs)
        && let [inner] = args[..]
        && contains_var(arena, inner, var_sym)
    {
        out.push(inner);
        // Also try the function node itself as a candidate.
        // E.g., for ln(x), try u = ln(x) (not just u = x).
        // This enables ∫ 1/(x·ln(x)) dx via u = ln(x), du = 1/x dx.
        out.push(factor);
        return out;
    }
    match node {
        ExprNode::Pow(base, _exp) if contains_var(arena, base, var_sym) => {
            out.push(base);
            // The argument of a function under the power: `u = x²` for
            // `x·cos(x²)⁻¹` (up to 0.46 only `u = cos x²` and functions inside the
            // argument were tried, and `∫ x·sec(x²) dx` stayed unevaluated;
            // SymPy's test_manual).
            if let Some(inner) = function_argument(arena, base)
                && contains_var(arena, inner, var_sym)
                && !out.contains(&inner)
            {
                out.push(inner);
            }
            inner_function_candidates(arena, base, var_sym, &mut out);
        }
        ExprNode::Add(_) => {
            inner_function_candidates(arena, factor, var_sym, &mut out);
        }
        // A library function itself: `∫ J₀(x)·J₁(x) dx = −J₀(x)²/2` (`u = J₀`,
        // `du = −J₁ dx`; SymPy's test_meijerint), and its arguments other
        // than the bare variable: `u = x²` for `∫ x·J₁(x²) dx = −J₀(x²)/2`
        // (unevaluated up to 0.46; SymPy 1.14 integrates it).
        ExprNode::Apply(_, args) | ExprNode::Lib(_, args)
            if args.iter().any(|&a| contains_var(arena, a, var_sym)) =>
        {
            out.push(factor);
            for &a in &args {
                if contains_var(arena, a, var_sym)
                    && !matches!(arena.node(a), ExprNode::Symbol(s) if *s == var_sym)
                    && !out.contains(&a)
                {
                    out.push(a);
                }
            }
        }
        _ => {}
    }
    out
}

/// The argument `g` of `sin g`, `cos g`, `tan g`, `exp g`, `ln g`, `sinh g`,
/// `cosh g`, `tanh g`.
fn function_argument(arena: &Arena, e: ExprId) -> Option<ExprId> {
    match *arena.node(e) {
        ExprNode::Sin(g)
        | ExprNode::Cos(g)
        | ExprNode::Tan(g)
        | ExprNode::Exp(g)
        | ExprNode::Ln(g)
        | ExprNode::Sinh(g)
        | ExprNode::Cosh(g)
        | ExprNode::Tanh(g) => Some(g),
        _ => None,
    }
}

/// `∫ xᵐ·G(x) dx` where `x` occurs in `G` only in integer powers `xʲ` with
/// `k | j` and `k | m + 1` for some `k ≥ 2`: `u = xᵏ`, `∫ u^{(m+1)/k − 1}·G/k du`
/// (every `xʲ = u^{j/k}`), and `F(xᵏ)` for its antiderivative `F` (the chain
/// rule, `(xᵏ)^{(m+1)/k − 1} = x^{m+1−k}` exactly).  `∫ x·e^{x⁴} dx` is
/// `∫ e^{u²} du/2 = √π·erfi(x²)/4`, `∫ x·sin(x⁴) dx` a Fresnel integral;
/// `u = x²` is no subexpression, and up to 0.46 they stayed unevaluated
/// (SymPy's test_manual: `√π·erfi(x²)/4`).
pub(super) fn try_power_substitution(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    use num_integer::Integer;
    if depth < 2 {
        return None;
    }
    let int_exp = |arena: &Arena, e: ExprId| -> Option<i64> {
        if e == var {
            return Some(1);
        }
        let ExprNode::Pow(b, n) = *arena.node(e) else {
            return None;
        };
        if b != var {
            return None;
        }
        arena
            .as_num(n)
            .filter(|q| q.is_integer())
            .and_then(|q| num_traits::ToPrimitive::to_i64(q.numer()))
    };
    let mut m: i64 = 0;
    let mut others: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &d in dependent {
        match int_exp(arena, d) {
            Some(e) => m += e,
            None => others.push(d),
        }
    }
    if others.is_empty() {
        return None;
    }
    let g = arena.mul(&others);
    // The powers `xʲ` of `G`; a bare `x` left over after replacing them
    // makes the substitution fail below.
    let mut k: i64 = m + 1;
    let mut powers: Vec<(ExprId, i64)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, g) {
        if id != var
            && let Some(e) = int_exp(arena, id)
        {
            k = k.gcd(&e);
            powers.push((id, e));
        }
    }
    if powers.is_empty() || k < 2 {
        return None;
    }
    // Only for a function that needs it: `e^{x⁴}`, `sin x⁴`, `sec x²`, `√(1 +
    // x⁴)` of a power.  A rational `G` is the rational integrator's, and
    // `ln(xᵏ) = k·ln x` gains nothing: after `x = t²`, `∫ ln(t²)·t/(t⁶ + t² +
    // 2/3) dt` went to `u = t²` and dilogarithms over a cubic's roots, and
    // `∫ (√x + ln x)/(x³ + x + 2/3) dx` ran for minutes (B14's fuzz unit
    // e6cc5985; 72 ms in 0.46).
    let needs = crate::base::walk::post_order_ids(arena, g)
        .into_iter()
        .any(|id| {
            let arg = match *arena.node(id) {
                ExprNode::Exp(a)
                | ExprNode::Sin(a)
                | ExprNode::Cos(a)
                | ExprNode::Tan(a)
                | ExprNode::Sinh(a)
                | ExprNode::Cosh(a)
                | ExprNode::Tanh(a)
                | ExprNode::Erf(a)
                | ExprNode::Erfc(a) => a,
                ExprNode::Pow(b, e) if arena.as_num(e).is_some_and(|q| !q.is_integer()) => b,
                _ => return false,
            };
            powers
                .iter()
                .any(|&(p, _)| crate::base::walk::contains(arena, arg, p))
        });
    if !needs {
        return None;
    }
    let u = arena.symbol("__pwu");
    let ExprNode::Symbol(u_sym) = *arena.node(u) else {
        return None;
    };
    // Larger powers first, so that `x` itself is replaced last.
    powers.sort_by_key(|p| std::cmp::Reverse(p.1));
    powers.dedup();
    let mut g_u = g;
    for &(node, e) in &powers {
        let ue = arena.int(e / k);
        let rep = arena.pow(u, ue);
        g_u = arena.subs_structural(g_u, node, rep);
    }
    if contains_var(arena, g_u, var_sym) {
        return None;
    }
    let lead = arena.int((m + 1) / k - 1);
    let u_lead = arena.pow(u, lead);
    let inv_k = arena.rational(1, k);
    let integrand = arena.mul(&[inv_k, u_lead, g_u]);
    let integrand = crate::transforms::eval::eval(arena, integrand);
    let res = integrate_node(arena, integrand, u, u_sym, depth - 1);
    if crate::base::walk::has_unevaluated(arena, res) {
        return None;
    }
    let k_id = arena.int(k);
    let x_k = arena.pow(var, k_id);
    let back = arena.subs_structural(res, u, x_k);
    tracing::debug!("integrate: power substitution u = x^k");
    Some(crate::transforms::eval::eval(arena, back))
}

/// Elementary function nodes (`sin g`, `exp g`, … with `g` depending on
/// `var`) inside `expr`, in post-order, deduplicated, at most four in total.
fn inner_function_candidates(
    arena: &Arena,
    expr: ExprId,
    var_sym: SymbolId,
    out: &mut SmallVec<[ExprId; 4]>,
) {
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if out.len() >= 4 {
            return;
        }
        if id == expr || out.contains(&id) {
            continue;
        }
        let is_candidate = match arena.node(id) {
            ExprNode::Sin(inner)
            | ExprNode::Cos(inner)
            | ExprNode::Tan(inner)
            | ExprNode::Exp(inner)
            | ExprNode::Ln(inner)
            | ExprNode::Sinh(inner)
            | ExprNode::Cosh(inner)
            | ExprNode::Tanh(inner)
            // `u = erf x` for `∫ erf(x)·e^{−x²}/(1 + erf³x − erf²x − erf x) dx`
            // (SymPy's test_heurisch; unevaluated up to 0.46).
            | ExprNode::Erf(inner)
            | ExprNode::Erfc(inner) => contains_var(arena, *inner, var_sym),
            _ => false,
        };
        if is_candidate {
            out.push(id);
        }
    }
}

/// Build the product of all elements in `children` except index `skip`.
pub(super) fn remaining_product(arena: &mut Arena, children: &[ExprId], skip: usize) -> ExprId {
    let parts: SmallVec<[ExprId; 4]> = children
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != skip)
        .map(|(_, &c)| c)
        .collect();
    match parts.len() {
        0 => arena.one,
        1 => parts[0],
        _ => arena.mul(&parts),
    }
}
