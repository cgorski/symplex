//! Special function integration table (part of [`integrate`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Special function integration table
// ═══════════════════════════════════════════════════════════════════════════

/// The functions `f` of [`try_special_function_integral`]: `∫ f(α·x + β)/x dx`
/// is an `Ei`, `Si`/`Ci` or `Shi`/`Chi`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LinearSpecial {
    Exp,
    Sin,
    Cos,
    Sinh,
    Cosh,
}

impl LinearSpecial {
    /// The function and its argument when `e` is one of them.
    pub(super) fn of(arena: &Arena, e: ExprId) -> Option<(Self, ExprId)> {
        match *arena.node(e) {
            ExprNode::Exp(a) => Some((Self::Exp, a)),
            ExprNode::Sin(a) => Some((Self::Sin, a)),
            ExprNode::Cos(a) => Some((Self::Cos, a)),
            ExprNode::Sinh(a) => Some((Self::Sinh, a)),
            ExprNode::Cosh(a) => Some((Self::Cosh, a)),
            _ => None,
        }
    }

    fn apply(self, arena: &mut Arena, a: ExprId) -> ExprId {
        match self {
            Self::Exp => arena.exp(a),
            Self::Sin => arena.sin(a),
            Self::Cos => arena.cos(a),
            Self::Sinh => arena.sinh(a),
            Self::Cosh => arena.cosh(a),
        }
    }

    /// `(g, s)` with `f′ = s·g`.
    fn derivative(self) -> (Self, i64) {
        match self {
            Self::Exp => (Self::Exp, 1),
            Self::Sin => (Self::Cos, 1),
            Self::Cos => (Self::Sin, -1),
            Self::Sinh => (Self::Cosh, 1),
            Self::Cosh => (Self::Sinh, 1),
        }
    }
}

/// Largest power `k` of a linear denominator `ℓᵏ` that
/// [`try_special_function_integral`] reduces by parts.
const MAX_SPECIAL_ORDER: i64 = 12;

/// Is the constant `e` written with a negative sign: a negative number, or a
/// product with a negative numeric factor?
fn has_negative_sign(arena: &Arena, e: ExprId) -> bool {
    match arena.node(e) {
        ExprNode::Num(_) => arena.as_num(e).is_some_and(Signed::is_negative),
        ExprNode::Neg(_) => true,
        ExprNode::Mul(children) => children
            .iter()
            .any(|&c| arena.as_num(c).is_some_and(Signed::is_negative)),
        _ => false,
    }
}

/// `f(c)` for `f` = `sin`, `cos`, `sinh`, `cosh` and a constant `c`, with a
/// negative sign of `c` taken out by parity (`sin(−1) = −sin 1`), which the
/// canonical form does not do.
fn parity_folded(arena: &mut Arena, f: LinearSpecial, c: ExprId) -> ExprId {
    if !has_negative_sign(arena, c) {
        return f.apply(arena, c);
    }
    let neg = arena.neg(c);
    let neg = crate::transforms::eval::eval(arena, neg);
    let v = f.apply(arena, neg);
    match f {
        LinearSpecial::Sin | LinearSpecial::Sinh => arena.neg(v),
        _ => v,
    }
}

/// `∫ f(w + β′)·ℓ⁻¹ dℓ` for `w = κ·ℓ` (`κ`, `β′` free of `ℓ`): `e^{β′}·Ei(w)`,
/// `sin β′·Ci(w) + cos β′·Si(w)`, `cos β′·Ci(w) − sin β′·Si(w)` (SymPy's
/// `EiRule`, `SiRule`, `CiRule`), `Shi(w)`, `Chi(w)` for `β′ = 0` and
/// `(e^{β′}·Ei(w) ∓ e^{−β′}·Ei(−w))/2` for `sinh`, `cosh` otherwise.  (SymPy's
/// `sinh β′·Chi(w) + cosh β′·Shi(w)` has two terms of the size `e^{|β′|+|w|}`
/// that cancel to `e^{|w|−|β′|}` on one side: `∫ cosh(12x)/(x + 2)` lost 21
/// digits for `x > −2`; each `Ei` term is of its own size.)  For a `w` with
/// a negative sign the even `Ci`, `Chi` take `−w` and the odd `Si`, `Shi`
/// change sign with it (`∫ cos(3x)/x` and `∫ cos(−3x)/x` are both `Ci(3x)`);
/// `Ei` has no symmetry.
fn linear_special_base(arena: &mut Arena, fam: LinearSpecial, w: ExprId, beta: ExprId) -> ExprId {
    let zero_shift = arena.is_zero_structural(beta);
    if matches!(fam, LinearSpecial::Sinh | LinearSpecial::Cosh) && !zero_shift {
        let neg_w = arena.neg(w);
        let neg_w = crate::transforms::expand::expand(arena, neg_w);
        let neg_beta = arena.neg(beta);
        let neg_beta = crate::transforms::eval::eval(arena, neg_beta);
        let up = linear_special_base(arena, LinearSpecial::Exp, w, beta);
        let down = linear_special_base(arena, LinearSpecial::Exp, neg_w, neg_beta);
        let half = arena.rational(1, 2);
        let combined = if fam == LinearSpecial::Sinh {
            arena.sub(up, down)
        } else {
            arena.add(&[up, down])
        };
        return arena.mul(&[half, combined]);
    }
    let (w, odd_sign) = if fam != LinearSpecial::Exp && has_negative_sign(arena, w) {
        let neg = arena.neg(w);
        (crate::transforms::expand::expand(arena, neg), -1)
    } else {
        (w, 1)
    };
    let odd_sign = arena.int(odd_sign);
    match fam {
        LinearSpecial::Exp => {
            let ei = arena.ei(w);
            if zero_shift {
                ei
            } else {
                let e = arena.exp(beta);
                arena.mul(&[e, ei])
            }
        }
        LinearSpecial::Sin | LinearSpecial::Cos => {
            let si = arena.si(w);
            let si = arena.mul(&[odd_sign, si]);
            if zero_shift {
                return if fam == LinearSpecial::Sin {
                    si
                } else {
                    arena.ci(w)
                };
            }
            let ci = arena.ci(w);
            let s = parity_folded(arena, LinearSpecial::Sin, beta);
            let c = parity_folded(arena, LinearSpecial::Cos, beta);
            if fam == LinearSpecial::Sin {
                let a = arena.mul(&[s, ci]);
                let b = arena.mul(&[c, si]);
                arena.add(&[a, b])
            } else {
                let a = arena.mul(&[c, ci]);
                let b = arena.mul(&[s, si]);
                arena.sub(a, b)
            }
        }
        LinearSpecial::Sinh => {
            let shi = special_apply(arena, crate::base::libfn::LibFn::Shi, w);
            arena.mul(&[odd_sign, shi])
        }
        LinearSpecial::Cosh => special_apply(arena, crate::base::libfn::LibFn::Chi, w),
    }
}

/// `Jₖ = ∫ f(L)·ℓ⁻ᵏ dℓ` for `L = κ·ℓ + β′` (`k ≥ 1`), by parts:
/// `Jₖ[f] = −f(L)·ℓ^{1−k}/(k − 1) + s·κ/(k − 1)·J_{k−1}[g]` with `f′ = s·g`,
/// down to [`linear_special_base`].  Computed upwards for `f` and `g` together.
#[allow(clippy::too_many_arguments)]
fn linear_special_power(
    arena: &mut Arena,
    fam: LinearSpecial,
    big_l: ExprId,
    ell: ExprId,
    kappa: ExprId,
    w: ExprId,
    beta: ExprId,
    k: i64,
) -> ExprId {
    let partner = fam.derivative().0;
    let fams: SmallVec<[LinearSpecial; 2]> = if partner == fam {
        SmallVec::from_slice(&[fam])
    } else {
        SmallVec::from_slice(&[fam, partner])
    };
    let mut level: SmallVec<[ExprId; 2]> = fams
        .iter()
        .map(|&f| linear_special_base(arena, f, w, beta))
        .collect();
    for j in 2..=k {
        let inv = arena.rational(1, j - 1);
        let power = arena.int(1 - j);
        let ell_pow = arena.pow(ell, power);
        let mut next: SmallVec<[ExprId; 2]> = SmallVec::new();
        for &f in &fams {
            let (g, s) = f.derivative();
            let prev_g = level[fams.iter().position(|&h| h == g).unwrap_or(0)];
            let f_l = f.apply(arena, big_l);
            let neg_inv = arena.neg(inv);
            let boundary = arena.mul(&[neg_inv, f_l, ell_pow]);
            let s = arena.int(s);
            let carried = arena.mul(&[s, inv, kappa, prev_g]);
            next.push(arena.add(&[boundary, carried]));
        }
        level = next;
    }
    level[0]
}

/// `∫ f(α·x + β)·R(x) dx` for `f` = `exp`, `sin`, `cos`, `sinh`, `cosh` and a
/// rational `R` whose partial fractions ([`crate::transforms::apart::apart`])
/// are a polynomial and terms `c·ℓ⁻ᵏ` over linear `ℓ = γ·x + δ`: with
/// `κ = α/γ`, `β′ = β − α·δ/γ` the term is `c/γ·∫ f(κ·ℓ + β′)·ℓ⁻ᵏ dℓ`, by
/// parts down to `Ei`, `Si`/`Ci`, `Shi`/`Chi` ([`linear_special_power`]);
/// the polynomial times `f` goes back to `integrate_node`.  Up to 0.43 only
/// `eˣ/x`, `sin x/x`, `cos x/x`, `sinh x/x`, `cosh x/x` were recognised:
/// `∫ e^{12x}/x dx` stayed unevaluated (SymPy: `Ei(12·x)`) and with it 52
/// Riccati equations of the ODE solver, as did `∫ e^{2x}/(x + 1)`,
/// `∫ eˣ/x²`, `∫ sin(3x)/x`.  For a rate `α` that may vanish, the cases
/// `α = 0` come with the answer ([`vanishing_parameter_values`]).  A
/// denominator with a factor irreducible over ℚ of degree ≥ 2 goes to
/// [`special_over_rational_factors`] (up to 0.44 `∫ e^{2x}/(x² − 2) dx`,
/// `∫ sin x/(x² − 2) dx` stayed unevaluated).
///
/// The arguments `κ·ℓ` vanish only at the root of `ℓ`, a pole of the term,
/// so the branch cuts of `Ei`, `Ci`, `Chi` on the negative axis are met
/// there alone for a real `κ` (`Ei` takes its real value on the axis).
pub(super) fn try_special_function_integral(
    arena: &mut Arena,
    dependent: &[ExprId],
    constants: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let mut special: Option<(ExprId, LinearSpecial, ExprId, ExprId, ExprId)> = None;
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &d in dependent {
        if special.is_none()
            && let Some((fam, arg)) = LinearSpecial::of(arena, d)
            && let Some((alpha, beta)) = symbolic_linear_coeff_of(arena, arg, var, var_sym)
        {
            special = Some((d, fam, arg, alpha, beta));
        } else {
            rest.push(d);
        }
    }
    let (f_node, fam, big_l, alpha, beta) = special?;
    if rest.is_empty() {
        return None;
    }
    let r = arena.mul(&rest);
    if !is_rational_in(arena, r, var_sym) {
        return None;
    }
    let (_, denom) = crate::poly::polybridge::as_numer_denom(arena, r);
    if !contains_var(arena, denom, var_sym) {
        return None;
    }
    let cases = vanishing_parameter_values(arena, alpha, var_sym)?;
    let parts = crate::transforms::apart::apart(arena, r, var);
    let terms: SmallVec<[ExprId; 6]> = match arena.node(parts) {
        ExprNode::Add(children) => children.clone(),
        _ => SmallVec::from_slice(&[parts]),
    };
    let mut poly_terms: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut pieces: SmallVec<[ExprId; 6]> = SmallVec::new();
    let mut linear = true;
    for t in terms {
        if !contains_var(arena, t, var_sym) || is_polynomial_in(arena, t, var, var_sym) {
            poly_terms.push(t);
            continue;
        }
        let (c, dep) = match split_constant_factors(arena, t, var_sym) {
            Some((c, dep)) => (arena.mul(&c), dep),
            None => (arena.one, t),
        };
        let Some((ell, k, gamma, delta)) = linear_power_parts(arena, dep, var, var_sym) else {
            linear = false;
            break;
        };
        if k > MAX_SPECIAL_ORDER {
            return None;
        }
        let kappa = arena.div(alpha, gamma);
        let kappa = crate::transforms::eval::eval(arena, kappa);
        let shift = arena.mul(&[kappa, delta]);
        let beta_shifted = arena.sub(beta, shift);
        let beta_shifted = crate::transforms::eval::eval(arena, beta_shifted);
        let w = arena.mul(&[kappa, ell]);
        let w = crate::transforms::expand::expand(arena, w);
        let j = linear_special_power(arena, fam, big_l, ell, kappa, w, beta_shifted, k);
        let c_over_gamma = arena.div(c, gamma);
        pieces.push(arena.mul(&[c_over_gamma, j]));
    }
    if !linear {
        // A factor irreducible over ℚ of degree ≥ 2 (`x² − 2`, `x³ + x + 1`):
        // the partial fractions over ℚ, each summed over the roots.
        let special = LinearSpecialArg {
            fam,
            big_l,
            alpha,
            beta,
        };
        let over_q = special_over_rational_factors(arena, &special, r, var)?;
        pieces = SmallVec::from_slice(&[over_q.proper]);
        poly_terms.clear();
        if !arena.is_zero_structural(over_q.polynomial) {
            poly_terms.push(over_q.polynomial);
        }
    }
    if !poly_terms.is_empty() {
        // P·f(L) as a sum of monomials times f(L) (`expand` of the product
        // would also split `e^{αx+β}` into `e^{αx}·e^β`).
        let p = arena.add(&poly_terms);
        let p = crate::transforms::expand::expand(arena, p);
        let monomials: SmallVec<[ExprId; 6]> = match arena.node(p) {
            ExprNode::Add(children) => children.clone(),
            _ => SmallVec::from_slice(&[p]),
        };
        let products: SmallVec<[ExprId; 6]> =
            monomials.iter().map(|&m| arena.mul(&[m, f_node])).collect();
        let p_f = arena.add(&products);
        let v = integrate_node(arena, p_f, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, v) {
            return None;
        }
        pieces.push(v);
    }
    tracing::debug!("integrate: exponential-integral family over linear denominators");
    let sum = arena.add(&pieces);
    // `Ci`/`Chi` next to the real logarithm `ln|x|` of the polynomial part
    // (`∫ (1 − cos 4x)/x dx = ln|x| − Ci(4x)`): `Ci` of the real `|w|`, or the
    // sum steps by `iπ` where the integrand is finite.
    let has_real_log = crate::base::walk::post_order_ids(arena, sum)
        .into_iter()
        .any(|id| matches!(*arena.node(id), ExprNode::Ln(a) if matches!(arena.node(a), ExprNode::Abs(_))));
    let sum = if has_real_log {
        ci_real_on_real_line(arena, sum, var, var_sym)
    } else {
        sum
    };
    let generic = wrap_with_constants(arena, sum, constants);
    let mut all: SmallVec<[ExprId; 6]> = constants.iter().copied().collect();
    all.extend(dependent.iter().copied());
    let integrand = arena.mul(&all);
    // A sum over non-real roots cannot be checked while `Ei`, `Ci`, `Si`,
    // `Chi`, `Shi` of complex arguments do not evaluate: a refused candidate
    // leaves the later routes their chance.
    if !linear && candidate_rejected(arena, integrand, generic, var, var_sym) {
        return None;
    }
    with_parameter_cases(arena, generic, integrand, &cases, var, var_sym, depth)
}

/// `Σ f(rₖ)/|g′(rₖ)|·H(x − rₖ)` over the real roots `rₖ` of `g` (all of them,
/// numeric, simple), an antiderivative of `f·δ(g)`.
pub(super) fn delta_simple_roots(
    arena: &mut Arena,
    f: ExprId,
    g: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    let roots = real_roots(arena, g, var)?;
    if roots.len() < 2 {
        return None;
    }
    let dg = crate::transforms::diff::diff(arena, g, var);
    let mut terms: Vec<ExprId> = Vec::with_capacity(roots.len());
    for r in roots {
        let slope = crate::transforms::subs::subs(arena, dg, var, r);
        let slope = crate::transforms::eval::eval(arena, slope);
        let s = crate::transforms::evalf::eval_const_f64(arena, slope)?;
        // A nonzero printed digit is certified; 0 (a double root?) refuses.
        if s == 0.0 || !s.is_finite() {
            return None;
        }
        let at = crate::transforms::subs::subs(arena, f, var, r);
        let at = crate::transforms::eval::eval(arena, at);
        if contains_non_finite(arena, at) {
            return None;
        }
        let abs_slope = arena.abs(slope);
        let x_minus_r = arena.sub(var, r);
        let h = arena.heaviside(x_minus_r);
        let t = arena.div(at, abs_slope);
        terms.push(arena.mul(&[t, h]));
    }
    let sum = arena.add(&terms);
    Some(crate::transforms::eval::eval(arena, sum))
}

/// `∫ xᵃ·e^{c·xᵖ + d} dx = eᵈ·x^{a+1}·z^{−s}·γ(s, z)/p` with `z = −c·xᵖ`,
/// `s = (a + 1)/p` (constants `a`, `c ≠ 0`, `d`, a rational `p > 0`; DLMF
/// 8.8.13 `dγ(s, z)/dz = z^{s−1}·e^{−z}`).  `z^{−s}·γ(s, z)` is entire in `z`
/// (DLMF 8.7.1), so the answer is continuous wherever `x^{a+1}` is; the
/// upper function `Γ(s, z)` differs by `x^{a+1}·z^{−s}·Γ(s)`, constant on
/// each side of `x = 0` but not the same constant, and jumped there
/// (`∫ e^{−x³} dx`, found by the hunt).  The last resort for the family: the elementary and `erf`
/// cases come from the earlier routes.  Up to 0.46 `∫ x^y·e^{−x} dx`,
/// `∫ xⁿ·e^{−x} dx` (`n` an integer symbol), `∫ x^{1/3}·e^{−x/8} dx` and
/// `∫ e^{−x³} dx` stayed unevaluated (SymPy's test_integrals, test_manual
/// and test_meijerint answer them with `lowergamma`/`uppergamma`).
pub(super) fn try_incomplete_gamma(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let mut a_terms: SmallVec<[ExprId; 2]> = SmallVec::new();
    let mut exp_arg: Option<ExprId> = None;
    for &d in dependent {
        match *arena.node(d) {
            ExprNode::Exp(g) if exp_arg.is_none() => exp_arg = Some(g),
            ExprNode::Pow(b, e) if b == var && !contains_var(arena, e, var_sym) => a_terms.push(e),
            _ if d == var => a_terms.push(arena.one),
            _ => return None,
        }
    }
    let g = exp_arg?;
    // g = c·xᵖ + d
    let terms: Vec<ExprId> = match arena.node(g) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![g],
    };
    let mut shift: SmallVec<[ExprId; 2]> = SmallVec::new();
    let mut mono: Option<(ExprId, Q)> = None;
    for t in terms {
        if !contains_var(arena, t, var_sym) {
            shift.push(t);
            continue;
        }
        if mono.is_some() {
            return None;
        }
        let (c, n) = monomial_parts(arena, t, var, var_sym)?;
        let n = arena.as_num(n)?.clone();
        mono = Some((c, n));
    }
    let (c, p) = mono?;
    // A negative power (`e^{1/x}`) is the substitution `x = 1/t`'s.
    if !p.is_positive() || arena.is_zero_structural(c) {
        return None;
    }
    let a = if a_terms.is_empty() {
        arena.zero
    } else {
        arena.add(&a_terms)
    };
    let one = arena.one;
    let a1 = arena.add(&[a, one]);
    let a1 = crate::transforms::eval::eval(arena, a1);
    let p_id = arena.num_ratio(p.clone());
    let s = arena.div(a1, p_id);
    let s = crate::transforms::eval::eval(arena, s);
    // A numeric `s` that is a positive integer (an elementary answer) or a
    // half-integer (`erf`) is the other routes'.
    if let Some(sv) = arena.as_num(s)
        && ((sv.is_integer() && sv.is_positive())
            || *(sv * Q::from_integer(2.into())).denom() == num_bigint::BigInt::from(1))
    {
        return None;
    }
    let p_id2 = arena.num_ratio(p);
    let x_p = arena.pow(var, p_id2);
    let neg_c = arena.neg(c);
    let z = arena.mul(&[neg_c, x_p]);
    let z = crate::transforms::eval::eval(arena, z);
    let gamma = arena.lib_apply(crate::base::libfn::LibFn::LowerGamma, &[s, z]);
    let neg_s = arena.neg(s);
    let z_pow = arena.pow(z, neg_s);
    let x_a1 = arena.pow(var, a1);
    let one = arena.one;
    let inv_p = arena.div(one, p_id);
    let mut factors: SmallVec<[ExprId; 5]> = SmallVec::from_slice(&[inv_p, x_a1, z_pow, gamma]);
    if !shift.is_empty() {
        let d = arena.add(&shift);
        factors.push(arena.exp(d));
    }
    tracing::debug!("integrate: x^a·e^(c·x^p) as an incomplete gamma function");
    let r = arena.mul(&factors);
    Some(crate::transforms::eval::eval(arena, r))
}

/// `∫ f(n, …, a·x + b) dx` for a classical orthogonal polynomial `f` of any
/// (also symbolic) degree `n` by its derivative rule read backwards
/// (DLMF 18.9.19–18.9.23, Abramowitz–Stegun 22.8): `∫ Pₙ = (Pₙ₊₁ −
/// Pₙ₋₁)/(2n + 1)`, `∫ Tₙ = (Tₙ₊₁/(n + 1) − Tₙ₋₁/(n − 1))/2`, `∫ Uₙ =
/// Tₙ₊₁/(n + 1)`, `∫ Hₙ = Hₙ₊₁/(2(n + 1))`, `∫ Lₙ = Lₙ − Lₙ₊₁`, `∫ Lₙ⁽ᵅ⁾ =
/// −Lₙ₊₁⁽ᵅ⁻¹⁾`, `∫ Cₙ⁽ᵅ⁾ = Cₙ₊₁⁽ᵅ⁻¹⁾/(2(α − 1))`, `∫ Pₙ⁽ᵅʲᵝ⁾ =
/// 2·Pₙ₊₁⁽ᵅ⁻¹ʲᵝ⁻¹⁾/(n + α + β)`, each over `a`; the vanishing denominators
/// are degenerate cases of the wrap.  Up to 0.46 none was integrated
/// (SymPy's test_manual integrates them all).
pub(super) fn orthogonal_polynomial_integral(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use crate::base::libfn::LibFn;
    let ExprNode::Lib(f, args) = arena.node(expr).clone() else {
        return None;
    };
    let g = *args.last()?;
    let params = &args[..args.len() - 1];
    if params.is_empty() || params.iter().any(|&p| contains_var(arena, p, var_sym)) {
        return None;
    }
    let (a, _) = symbolic_linear_coeff_of(arena, g, var, var_sym)?;
    let n = params[0];
    let one = arena.one;
    let two = arena.int(2);
    let n1 = arena.add(&[n, one]);
    let n1 = crate::transforms::eval::eval(arena, n1);
    let minus_one = arena.int(-1);
    let nm1 = arena.add(&[n, minus_one]);
    let nm1 = crate::transforms::eval::eval(arena, nm1);
    let r = match (f, params.len()) {
        (LibFn::Legendre, 1) => {
            let up = arena.lib_apply(f, &[n1, g]);
            let down = arena.lib_apply(f, &[nm1, g]);
            let diff = arena.sub(up, down);
            let two_n = arena.mul(&[two, n]);
            let den = arena.add(&[two_n, one]);
            arena.div(diff, den)
        }
        (LibFn::ChebyshevT, 1) => {
            let up = arena.lib_apply(f, &[n1, g]);
            let down = arena.lib_apply(f, &[nm1, g]);
            let t1 = arena.div(up, n1);
            let t2 = arena.div(down, nm1);
            let diff = arena.sub(t1, t2);
            arena.div(diff, two)
        }
        (LibFn::ChebyshevU, 1) => {
            let t = arena.lib_apply(LibFn::ChebyshevT, &[n1, g]);
            arena.div(t, n1)
        }
        (LibFn::Hermite, 1) => {
            let h = arena.lib_apply(f, &[n1, g]);
            let den = arena.mul(&[two, n1]);
            arena.div(h, den)
        }
        (LibFn::Laguerre, 1) => {
            let up = arena.lib_apply(f, &[n1, g]);
            arena.sub(expr, up)
        }
        (LibFn::AssocLaguerre, 2) => {
            let alpha = arena.add(&[params[1], minus_one]);
            let up = arena.lib_apply(f, &[n1, alpha, g]);
            arena.neg(up)
        }
        (LibFn::Gegenbauer, 2) => {
            let alpha = arena.add(&[params[1], minus_one]);
            let up = arena.lib_apply(f, &[n1, alpha, g]);
            let den = arena.mul(&[two, alpha]);
            arena.div(up, den)
        }
        (LibFn::Jacobi, 3) => {
            let alpha = arena.add(&[params[1], minus_one]);
            let beta = arena.add(&[params[2], minus_one]);
            let up = arena.lib_apply(f, &[n1, alpha, beta, g]);
            let den = arena.add(&[n, params[1], params[2]]);
            let num = arena.mul(&[two, up]);
            arena.div(num, den)
        }
        _ => return None,
    };
    tracing::debug!("integrate: orthogonal polynomial by its derivative rule");
    Some(arena.div(r, a))
}

/// `∫ xᵐ·Z_ν(a·x) dx` for `Z` = `J`, `Y`, `I`, `K` and `m = ν + 1` or `m = 1 − ν`
/// (DLMF 10.6.6, 10.29.4: `(x^{ν+1}Z_{ν+1})′ = x^{ν+1}Z_ν`, `(x^{−ν}Z_ν)′ =
/// −x^{−ν}Z_{ν+1}`, the signs of `I`, `K` as there), and `∫ Z_{±1}(a·x + b)`.
/// `∫ x·J₀(x) dx = x·J₁(x)`, `∫ J₁(x) dx = −J₀(x)` stayed unevaluated up to
/// 0.46 (SymPy's test_meijerint).
pub(super) fn bessel_power_integral(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use crate::base::libfn::LibFn;
    let mut bessel: Option<(LibFn, ExprId, ExprId)> = None;
    let mut m_terms: SmallVec<[ExprId; 2]> = SmallVec::new();
    for &d in dependent {
        if let ExprNode::Lib(
            f @ (LibFn::BesselJ | LibFn::BesselY | LibFn::BesselI | LibFn::BesselK),
            args,
        ) = arena.node(d).clone()
            && args.len() == 2
            && !contains_var(arena, args[0], var_sym)
            && bessel.is_none()
        {
            bessel = Some((f, args[0], args[1]));
        } else if d == var {
            m_terms.push(arena.one);
        } else if let ExprNode::Pow(b, e) = *arena.node(d)
            && b == var
            && !contains_var(arena, e, var_sym)
        {
            m_terms.push(e);
        } else {
            return None;
        }
    }
    let (f, nu, g) = bessel?;
    let (a, b) = symbolic_linear_coeff_of(arena, g, var, var_sym)?;
    let m = if m_terms.is_empty() {
        arena.zero
    } else {
        arena.add(&m_terms)
    };
    let m = crate::transforms::eval::eval(arena, m);
    if !arena.is_zero_structural(b) && !arena.is_zero_structural(m) {
        return None;
    }
    let one = arena.one;
    let nu1 = arena.add(&[nu, one]);
    let nu1 = crate::transforms::eval::eval(arena, nu1);
    let minus_one = arena.int(-1);
    let num1 = arena.add(&[nu, minus_one]);
    let num1 = crate::transforms::eval::eval(arena, num1);
    let up_case = arena.sub(m, nu1);
    let up_case = crate::transforms::eval::eval(arena, up_case);
    let one_minus_nu = arena.sub(one, nu);
    let down_case = arena.sub(m, one_minus_nu);
    let down_case = crate::transforms::eval::eval(arena, down_case);
    let (order, sign) = if arena.is_zero_structural(up_case) {
        (nu1, if f == LibFn::BesselK { -1 } else { 1 })
    } else if arena.is_zero_structural(down_case) {
        (num1, if f == LibFn::BesselI { 1 } else { -1 })
    } else {
        return None;
    };
    let z = arena.lib_apply(f, &[order, g]);
    let x_m = arena.pow(var, m);
    let s = arena.int(sign);
    let r = arena.mul(&[s, x_m, z]);
    tracing::debug!("integrate: Bessel function times a power by its derivative rule");
    Some(arena.div(r, a))
}

/// Most factors `exp`, `sin`, `cos`, `sinh`, `cosh` (with multiplicity) that
/// [`try_special_product_as_exponentials`] multiplies out.
const MAX_EXPONENTIAL_PRODUCT: i64 = 4;

/// `∫ f₁(L₁)·…·fₖ(Lₖ)·R(x) dx` for two or more factors (with multiplicity)
/// `f` = `exp`, `sin`, `cos`, `sinh`, `cosh` of linear arguments with numeric
/// coefficients and a rational `R` with a denominator: the factors written
/// as exponentials (`cos x = (e^{ix} + e^{−ix})/2`), multiplied out into
/// terms `c·e^{α·x + β}·R(x)`, each the exponential-integral family of
/// [`try_special_function_integral`] at a complex rate `α` (`Ei` at complex
/// points).  `∫ eˣ·cos x/x dx`, `∫ sin x·e^{−ix}/x dx`, `∫ e^{−x}·sin x/(x²
/// + 1) dx` stayed unevaluated up to 0.46 (SymPy's test_integrals; SymPy
/// answers the first two in this form).  `None` when an `Ei` argument
/// crosses its cut, the negative axis, at a real point where the integrand
/// is finite ([`ei_crosses_cut`]): the answer would jump there.
pub(super) fn try_special_product_as_exponentials(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    if depth < 3 || dependent.len() < 2 {
        return None;
    }
    let mut specials: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut count: i64 = 0;
    for &d in dependent {
        let (base, power) = match arena.node(d) {
            ExprNode::Pow(b, e) => match arena.as_num(*e).and_then(|q| {
                (q.is_integer() && q.is_positive())
                    .then(|| num_traits::ToPrimitive::to_i64(q.numer()))
                    .flatten()
            }) {
                Some(k) => (*b, k),
                None => (d, 0),
            },
            _ => (d, 1),
        };
        let special = power > 0
            && LinearSpecial::of(arena, base).is_some_and(|(_, arg)| {
                symbolic_linear_coeff_of(arena, arg, var, var_sym).is_some()
                    && crate::base::walk::free_symbols(arena, arg)
                        .iter()
                        .all(|&s| matches!(arena.node(s), ExprNode::Symbol(t) if *t == var_sym))
            });
        if special {
            count += power;
            specials.push(d);
        } else {
            rest.push(d);
        }
    }
    if !(2..=MAX_EXPONENTIAL_PRODUCT).contains(&count) || rest.is_empty() {
        return None;
    }
    let r = arena.mul(&rest);
    if !is_rational_in(arena, r, var_sym) {
        return None;
    }
    let (_, denom) = crate::poly::polybridge::as_numer_denom(arena, r);
    if !contains_var(arena, denom, var_sym) {
        return None;
    }
    let s = arena.mul(&specials);
    let s = linearized_trig_product(arena, &specials, var, var_sym).unwrap_or_else(|| {
        let s = arena.rewrite_as_exp_expr(s);
        let s = crate::transforms::eval::eval(arena, s);
        crate::transforms::expand::expand(arena, s)
    });
    let terms: Vec<ExprId> = match arena.node(s) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![s],
    };
    let mut pieces: SmallVec<[ExprId; 8]> = SmallVec::new();
    for t in terms {
        let factors: Vec<ExprId> = match arena.node(t) {
            ExprNode::Mul(children) => children.to_vec(),
            _ => vec![t],
        };
        let merged = merge_exponential_factors(arena, &factors, var, var_sym).unwrap_or(t);
        let integrand = arena.mul(&[merged, r]);
        let integrand = crate::transforms::eval::eval(arena, integrand);
        let piece = integrate_node(arena, integrand, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, piece) {
            return None;
        }
        pieces.push(piece);
    }
    let sum = arena.add(&pieces);
    let sum = ei_continuous_on_real_line(arena, sum, var, var_sym)?;
    let sum = ci_real_on_real_line(arena, sum, var, var_sym);
    tracing::debug!("integrate: exponential and trigonometric factors as complex exponentials");
    Some(sum)
}

/// The product of the sine and cosine factors `specials` (all of them
/// such, of linear arguments) as a sum of terms with one sine or cosine
/// each (`cos²x = (1 + cos 2x)/2`, `sin x·cos 3x = (sin 4x − sin 2x)/2`), so
/// that their integrals over a rational function come out in `Si`, `Ci`
/// rather than in `Ei` at imaginary points.  `None` for other factors or
/// when a term keeps two of them.
fn linearized_trig_product(
    arena: &mut Arena,
    specials: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let trig = |arena: &Arena, e: ExprId| {
        let base = match *arena.node(e) {
            ExprNode::Pow(b, _) => b,
            _ => e,
        };
        matches!(
            LinearSpecial::of(arena, base),
            Some((LinearSpecial::Sin | LinearSpecial::Cos, _))
        )
    };
    if !specials.iter().all(|&d| trig(arena, d)) {
        return None;
    }
    let s = arena.mul(specials);
    let s = crate::transforms::expand::expand(arena, s);
    let s = match crate::transforms::trig_integ::linearize_trig_powers(arena, &[s], var, var_sym) {
        Some(l) => l,
        None => s,
    };
    let s = arena.trig_combine_expr(s);
    let s = crate::transforms::eval::eval(arena, s);
    let s = crate::transforms::expand::expand(arena, s);
    let terms: Vec<ExprId> = match arena.node(s) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![s],
    };
    for t in terms {
        let factors: Vec<ExprId> = match arena.node(t) {
            ExprNode::Mul(children) => children.to_vec(),
            _ => vec![t],
        };
        let n = factors
            .iter()
            .filter(|&&f| contains_var(arena, f, var_sym))
            .count();
        let single = factors.iter().all(|&f| {
            !contains_var(arena, f, var_sym)
                || matches!(
                    LinearSpecial::of(arena, f),
                    Some((LinearSpecial::Sin | LinearSpecial::Cos, _))
                )
        });
        if n > 1 || !single {
            return None;
        }
    }
    Some(s)
}

/// `e` with every `Ci(w)`, `Chi(w)` of a real linear `w = κ·x + β` written
/// `Ci(w) − ln w + ln|w|`: `Ci(z) − ln z` is entire, so this is `Ci(|w|)` for
/// real `x`, real on both sides of the zero of `w`, where the principal
/// `Ci(w)` has `+iπ`.  A sum of such terms whose integrand is finite there
/// (`∫ sin²(2x)/x dx = (ln|x| − Ci(4x))/2`) is then continuous; with `Ci(4x)`
/// it stepped by `iπ/2` at 0 (found by the hunt).
fn ci_real_on_real_line(arena: &mut Arena, e: ExprId, var: ExprId, var_sym: SymbolId) -> ExprId {
    let mut out = e;
    for id in crate::base::walk::post_order_ids(arena, e) {
        let w = match arena.node(id) {
            ExprNode::Ci(w) => *w,
            ExprNode::Lib(crate::base::libfn::LibFn::Chi, args) if args.len() == 1 => args[0],
            _ => continue,
        };
        if !contains_var(arena, w, var_sym) {
            continue;
        }
        let Some((k, b)) = symbolic_linear_coeff_of(arena, w, var, var_sym) else {
            continue;
        };
        // Proved real: `ln|w| − ln w` is locally constant only for a real
        // `w` (up to 0.52 an imaginary part below `10⁻¹²` of the value was
        // taken for 0).
        let real = |arena: &mut Arena, c: ExprId| {
            crate::transforms::realness::imaginary_part_sign(arena, c)
                .is_zero()
                .is_true()
                && crate::transforms::evalf::evalf_complex64(arena, c)
                    .is_ok_and(|z| z.re.is_finite())
        };
        if !real(arena, k) || !real(arena, b) {
            continue;
        }
        let ln_w = arena.ln(w);
        let abs_w = arena.abs(w);
        let ln_abs = arena.ln(abs_w);
        let fix = arena.sub(ln_abs, ln_w);
        let fixed = arena.add(&[id, fix]);
        out = arena.subs_structural(out, id, fixed);
    }
    if out == e {
        e
    } else {
        crate::transforms::eval::eval(arena, out)
    }
}

/// `e` with every `Ei(p·x + q)` (numeric `p`, `q`) whose argument crosses
/// the negative real axis — the cut, where `Ei` jumps by `2πi` — at a real
/// `x` written `Ei(z) − ln z + ln(x − c)` for its zero `c = −q/p`: `Ei(z) −
/// ln z` is entire, and `ln(x − c)` keeps off its cut for real `x` (`c` is
/// not real when the crossing is away from it), so the antiderivative is
/// continuous where its integrand is, with the same derivative.  (A real
/// argument stays on the axis, where `Ei` takes its real value; one with a
/// constant non-zero imaginary part never meets it.)  `None` when an `Ei`
/// of the variable has an argument of another form.
fn ei_continuous_on_real_line(
    arena: &mut Arena,
    e: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let mut out = e;
    for id in crate::base::walk::post_order_ids(arena, e) {
        let ExprNode::Ei(z) = *arena.node(id) else {
            continue;
        };
        if !contains_var(arena, z, var_sym) {
            continue;
        }
        let (p, q) = symbolic_linear_coeff_of(arena, z, var, var_sym)?;
        // Where the argument meets the real axis, and on which side of the
        // branch point, by certified signs.  Up to 0.52 they were `f64`
        // values within `10⁻¹²`: a crossing `2·10⁻¹⁴` left of 0 counted as
        // off the cut, and `∫₀² e^{(1+i)x}/(x − 1 + 10⁻¹⁴i) dx` (`4.7265 +
        // 2.6665i`) was `−9.6453 + 11.8945i`, the jump `2πi·e^{1+i}` of `Ei`
        // at `x ≈ 1` left in.
        use crate::base::truth::SignSet;
        let p_side = crate::transforms::realness::imaginary_part_sign(arena, p);
        if p_side == SignSet::REAL {
            return None;
        }
        if p_side == SignSet::ZERO {
            continue;
        }
        let zero_c = arena.div(q, p);
        let zero_c = arena.neg(zero_c);
        let zero_c = crate::transforms::expand::expand(arena, zero_c);
        let zero_c = crate::transforms::eval::eval(arena, zero_c);
        let c_side = crate::transforms::realness::imaginary_part_sign(arena, zero_c);
        if c_side == SignSet::REAL {
            return None;
        }
        if c_side == SignSet::ZERO {
            // The zero `c` of the argument is real: `Ei(z) − ln z` is entire
            // and `ln|x − c|` the real logarithm, whose singularities cancel
            // between the terms where the integrand is finite at `c`
            // (`∫ e^{−x}·sin x/x dx` stepped by `π` at 0 with the principal
            // logarithms of `Ei((−1 ± i)·x)`, found by the hunt).
            let x_minus_c = arena.sub(var, zero_c);
            let abs = arena.abs(x_minus_c);
            let ln_abs = arena.ln(abs);
            let ln_z = arena.ln(z);
            let continuous = arena.sub(ln_abs, ln_z);
            let continuous = arena.add(&[id, continuous]);
            out = arena.subs_structural(out, id, continuous);
            continue;
        }
        // At the real `x₀ = −im q/im p` the argument is `re(p·x₀ + q) =
        // −im(p̄·q)/im p`: on the cut (negative) when `im(p̄·q)` has the sign
        // of `im p`.
        let p_bar = arena.conjugate(p);
        let pq = arena.mul(&[p_bar, q]);
        let pq = crate::transforms::expand::expand(arena, pq);
        let pq = crate::transforms::eval::eval(arena, pq);
        let cross_side = crate::transforms::realness::imaginary_part_sign(arena, pq);
        if cross_side == SignSet::REAL || cross_side == SignSet::ZERO {
            return None;
        }
        if cross_side != p_side {
            continue;
        }
        // c = −q·p̄/(p·p̄), so that a Gaussian rational comes out as `a + b·i`.
        let num = arena.mul(&[q, p_bar]);
        let num = crate::transforms::expand::expand(arena, num);
        let den = arena.mul(&[p, p_bar]);
        let den = crate::transforms::expand::expand(arena, den);
        let c = arena.div(num, den);
        let c = arena.neg(c);
        let c = crate::transforms::expand::expand(arena, c);
        let c = crate::transforms::eval::eval(arena, c);
        let x_minus_c = arena.sub(var, c);
        let x_minus_c = crate::transforms::eval::eval(arena, x_minus_c);
        let ln_z = arena.ln(z);
        let ln_lin = arena.ln(x_minus_c);
        let continuous = arena.sub(ln_lin, ln_z);
        let continuous = arena.add(&[id, continuous]);
        out = arena.subs_structural(out, id, continuous);
    }
    Some(crate::transforms::eval::eval(arena, out))
}

/// `(ℓ, k, γ, δ)` when `dep` is `ℓ⁻ᵏ` for an integer `k ≥ 1` and a linear
/// `ℓ = γ·x + δ`.
fn linear_power_parts(
    arena: &mut Arena,
    dep: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<(ExprId, i64, ExprId, ExprId)> {
    let ExprNode::Pow(ell, e) = *arena.node(dep) else {
        return None;
    };
    let k = arena
        .as_num(e)
        .filter(|q| q.is_integer() && q.is_negative())?;
    let k = i64::try_from(-k.to_integer()).ok()?;
    let (gamma, delta) = symbolic_linear_coeff_of(arena, ell, var, var_sym)?;
    Some((ell, k, gamma, delta))
}

/// The function `f` of a [`try_special_function_integral`] integrand and its
/// linear argument `L = α·x + β`.
struct LinearSpecialArg {
    fam: LinearSpecial,
    /// `L` itself.
    big_l: ExprId,
    alpha: ExprId,
    beta: ExprId,
}

/// What [`special_over_rational_factors`] makes of `∫ f(L)·R(x) dx`.
struct SpecialOverQ {
    /// The antiderivative for the proper part of `R`.
    proper: ExprId,
    /// The polynomial part of `R`, whose product with `f(L)` the caller
    /// integrates.
    polynomial: ExprId,
}

/// Largest degree of the denominator of `R` in
/// [`special_over_rational_factors`] and [`try_log_times_rational`].
const MAX_SPECIAL_DENOM_DEGREE: usize = 24;

/// Largest degree (in the variable, numerator and denominator together, as
/// written) of a rational function that [`rational_over_q`] multiplies out.
const MAX_RATIONAL_DEGREE: u64 = 64;

/// A bound on the degree in `var` of the rational expression `e` multiplied
/// out (the degrees of the factors added, of the terms maximised, of a
/// power multiplied by its exponent, whatever its sign); `None` for a
/// power with a non-integer exponent.
fn rational_degree_bound(arena: &Arena, e: ExprId, var: ExprId) -> Option<u64> {
    let mut deg: FxHashMap<ExprId, u64> = FxHashMap::default();
    for id in crate::base::walk::post_order_ids(arena, e) {
        let d = if id == var {
            1
        } else {
            match arena.node(id) {
                ExprNode::Add(ch) => ch.iter().map(|c| deg.get(c).copied().unwrap_or(0)).max()?,
                ExprNode::Mul(ch) => ch.iter().try_fold(0u64, |acc, c| {
                    acc.checked_add(deg.get(c).copied().unwrap_or(0))
                })?,
                ExprNode::Neg(c) => deg.get(c).copied().unwrap_or(0),
                ExprNode::Pow(b, k) => {
                    let k = arena.as_num(*k).filter(|q| q.is_integer())?;
                    let k = u64::try_from(k.to_integer().magnitude()).ok()?;
                    deg.get(b).copied().unwrap_or(0).checked_mul(k)?
                }
                _ => 0,
            }
        };
        deg.insert(id, d);
    }
    deg.get(&e).copied()
}

/// `r` as `N/D` with `N, D ∈ ℚ[x]` and `D` not constant; `None` when a
/// coefficient is not rational or the degree is larger than
/// [`MAX_RATIONAL_DEGREE`].
fn rational_over_q(arena: &mut Arena, r: ExprId, var: ExprId) -> Option<(Poly, Poly)> {
    if rational_degree_bound(arena, r, var)? > MAX_RATIONAL_DEGREE {
        return None;
    }
    let (n, d) = crate::poly::polybridge::as_numer_denom(arena, r);
    let n = crate::transforms::expand::expand(arena, n);
    let n = crate::transforms::eval::eval(arena, n);
    let d = crate::transforms::expand::expand(arena, d);
    let d = crate::transforms::eval::eval(arena, d);
    let p = crate::poly::polybridge::expr_to_poly(arena, n, var)?;
    let q = crate::poly::polybridge::expr_to_poly(arena, d, var)?;
    (q.degree()? >= 1).then_some((p, q))
}

/// `∫ f(L)·R(x) dx` for `R ∈ ℚ(x)` whose denominator has a factor
/// irreducible over ℚ of degree ≥ 2, from the partial fractions `A/gᵏ` of
/// its proper part over ℚ (`deg A < deg g`):
///
/// - a linear `g` as in [`try_special_function_integral`];
/// - otherwise the poles of order `k ≥ 2` reduced first
///   ([`reduce_special_poles`]: terms `f(L)·S` with `S ∈ ℚ(x)`, and simple
///   poles left), then every simple fraction `M/g` split over the roots `ρ`
///   of `g`: `M/g = Σ_ρ r(ρ)/(x − ρ)` with the residue `r = M·g′⁻¹ mod g`
///   (the first step of Bronstein's full partial fraction decomposition,
///   SymPy's `apart_list_full_decomposition`), and `∫ f(L)/(x − ρ)` an `Ei`,
///   `Si`/`Ci` or `Shi`/`Chi` of `α·(x − ρ)` ([`linear_special_base`]),
///   summed over the roots in radicals or as a `RootSum`
///   ([`factor_roots`]).
///
/// The two terms of a pair of conjugate roots are conjugate on the real
/// line, and each is continuous there: `α·(x − ρ)` never meets the cut of
/// `Ei`, `Ci`, `Chi` (the negative axis) for a non-real `ρ`, and for a real
/// one only at `x = ρ`, a pole of the integrand.  `None` for coefficients
/// outside ℚ, a pole of order `k ≥ 2` with an irrational `α`, or beyond the
/// budgets.
fn special_over_rational_factors(
    arena: &mut Arena,
    special: &LinearSpecialArg,
    r: ExprId,
    var: ExprId,
) -> Option<SpecialOverQ> {
    let (numer, denom) = rational_over_q(arena, r, var)?;
    if denom.degree()? > MAX_SPECIAL_DENOM_DEGREE {
        return None;
    }
    let (quotient, remainder) = numer.div_rem(&denom);
    let polynomial = crate::poly::polybridge::poly_to_expr(arena, &quotient, var);
    let (content, factors) = denom.factor_over_z();
    if content.is_zero() {
        return None;
    }
    let scaled = remainder.scale(&(Q::one() / content));
    let mut pieces: SmallVec<[ExprId; 6]> = SmallVec::new();
    for term in crate::transforms::apart::partial_fractions_over_q(&scaled, &factors) {
        let (a, g, k) = (term.numer, term.factor, term.power);
        if a.is_zero() {
            continue;
        }
        if i64::from(k) > MAX_SPECIAL_ORDER {
            return None;
        }
        let piece = if g.degree() == Some(1) {
            linear_factor_special(arena, special, &a, &g, k, var)
        } else {
            special_over_irreducible(arena, special, &a, &g, k, var)?
        };
        pieces.push(piece);
    }
    let proper = match pieces.len() {
        0 => arena.zero,
        1 => pieces[0],
        _ => arena.add(&pieces),
    };
    Some(SpecialOverQ { proper, polynomial })
}

/// `∫ f(L)·a/gᵏ dx` for a constant `a` and a linear `g = g₁·x + g₀`:
/// `a/g₁ᵏ·∫ f(L)·ℓ⁻ᵏ dℓ` with `ℓ = x + g₀/g₁` ([`linear_special_power`]).
fn linear_factor_special(
    arena: &mut Arena,
    special: &LinearSpecialArg,
    a: &Poly,
    g: &Poly,
    k: u32,
    var: ExprId,
) -> ExprId {
    let g1 = g.coeff(1);
    let mut c = a.coeff(0);
    for _ in 0..k {
        c /= &g1;
    }
    let delta = arena.num_ratio(g.coeff(0) / &g1);
    let ell = arena.add(&[var, delta]);
    let kappa = special.alpha;
    let shift = arena.mul(&[kappa, delta]);
    let beta_shifted = arena.sub(special.beta, shift);
    let beta_shifted = crate::transforms::eval::eval(arena, beta_shifted);
    let w = arena.mul(&[kappa, ell]);
    let w = crate::transforms::expand::expand(arena, w);
    let j = linear_special_power(
        arena,
        special.fam,
        special.big_l,
        ell,
        kappa,
        w,
        beta_shifted,
        i64::from(k),
    );
    let c = arena.num_ratio(c);
    arena.mul(&[c, j])
}

/// The functions met in `∫ f(L)·R(x) dx` by parts: `[(f, 0, s)]` for
/// `f = exp` (`f′ = s·f`), else `[(f, 1, s), (g, 0, s′)]` (`sin`/`cos`,
/// `sinh`/`cosh`): each with the index of the function its derivative is
/// a multiple of, and the sign of that multiple (`f′(L) = s·α·g(L)`).
fn special_closure(fam: LinearSpecial) -> SmallVec<[(LinearSpecial, usize, i64); 2]> {
    let (g, s) = fam.derivative();
    if g == fam {
        SmallVec::from_slice(&[(fam, 0, s)])
    } else {
        let (_, s2) = g.derivative();
        SmallVec::from_slice(&[(fam, 1, s), (g, 0, s2)])
    }
}

/// `∫ Σⱼ Fⱼ(L)·Mⱼ/g dx` with simple poles only, and the part `Σⱼ Fⱼ(L)·Bⱼ/gᵏ⁻¹`
/// of the antiderivative taken off before ([`reduce_special_poles`]).
struct PoleReduction {
    /// The numerators `Bⱼ` over `gᵏ⁻¹`.
    boundary: Vec<Poly>,
    /// The numerators `Mⱼ` over `g` (`deg Mⱼ < deg g`).
    simple: Vec<Poly>,
}

/// `g′⁻¹ mod g` for a square-free `g`.
fn inverse_derivative_mod(g: &Poly) -> Option<Poly> {
    let eg = Poly::extended_gcd(&g.derivative(), g);
    if eg.gcd.degree() != Some(0) {
        return None;
    }
    let c = eg.gcd.coeff(0);
    Some(eg.x.scale(&(Q::one() / c)).rem(g))
}

/// The order of the poles of `∫ Σⱼ Fⱼ(L)·Nⱼ/gᵏ dx` lowered to 1 (`g`
/// square-free, `k ≥ 2`, `deg Nⱼ < k·deg g`, `F` the [`special_closure`]
/// with `Fᵢ′ = sᵢ·α·F_{p(i)}`, `α` rational), as Hermite's reduction does for
/// rational functions: with `Sⱼ = Bⱼ/g^{m−1}`,
/// `(Σⱼ Fⱼ·Sⱼ)′ = Σⱼ Fⱼ·(Sⱼ′ + Σ_{p(i)=j} sᵢ·α·Sᵢ)`, whose part of order `m`
/// is `−(m − 1)·Bⱼ·g′/gᵐ`; `Bⱼ = −Nⱼ·((m − 1)·g′)⁻¹ mod g` cancels the
/// order `m` of the integrand, and the rest is of order `m − 1`.
fn reduce_special_poles(
    g: &Poly,
    k: u32,
    mut numer: Vec<Poly>,
    closure: &[(LinearSpecial, usize, i64)],
    alpha: &Q,
) -> Option<PoleReduction> {
    let inv = inverse_derivative_mod(g)?;
    let gp = g.derivative();
    let n = closure.len();
    let mut boundary = vec![Poly::zero(); n];
    for m in (2..=k).rev() {
        let m1 = Q::from_integer((m - 1).into());
        let b: Vec<Poly> = numer
            .iter()
            .map(|nj| (&nj.rem(g) * &inv).scale(&(-Q::one() / &m1)).rem(g))
            .collect();
        let g_pow = g.pow((k - m) as usize);
        let mut next = Vec::with_capacity(n);
        for j in 0..n {
            boundary[j] = &boundary[j] + &(&b[j] * &g_pow);
            let top = &numer[j] + &(&b[j] * &gp).scale(&m1);
            let (q, rem) = top.div_rem(g);
            if !rem.is_zero() {
                return None;
            }
            let mut cross = b[j].derivative();
            for (i, &(_, p, s)) in closure.iter().enumerate() {
                if p == j {
                    let si = alpha * Q::from_integer(s.into());
                    cross = &cross + &b[i].scale(&si);
                }
            }
            next.push(&q - &cross);
        }
        numer = next;
    }
    Some(PoleReduction {
        boundary,
        simple: numer,
    })
}

/// `∫ f(L)·a/gᵏ dx` for `g` irreducible over ℚ of degree ≥ 2 (see
/// [`special_over_rational_factors`]).
fn special_over_irreducible(
    arena: &mut Arena,
    special: &LinearSpecialArg,
    a: &Poly,
    g: &Poly,
    k: u32,
    var: ExprId,
) -> Option<ExprId> {
    let closure = special_closure(special.fam);
    let mut numer = vec![Poly::zero(); closure.len()];
    numer[0] = a.clone();
    let reduced = if k == 1 {
        PoleReduction {
            boundary: vec![Poly::zero(); closure.len()],
            simple: numer,
        }
    } else {
        let alpha = arena.as_num(special.alpha)?.clone();
        reduce_special_poles(g, k, numer, &closure, &alpha)?
    };
    let mut terms: SmallVec<[ExprId; 4]> = SmallVec::new();
    if k >= 2 {
        let g_expr = crate::poly::polybridge::poly_to_expr(arena, g, var);
        let den = if k == 2 {
            g_expr
        } else {
            let e = arena.int(i64::from(k) - 1);
            arena.pow(g_expr, e)
        };
        for (b, &(f, _, _)) in reduced.boundary.iter().zip(closure.iter()) {
            if b.is_zero() {
                continue;
            }
            let b_expr = crate::poly::polybridge::poly_to_expr(arena, b, var);
            let s = arena.div(b_expr, den);
            let f_l = f.apply(arena, special.big_l);
            terms.push(arena.mul(&[s, f_l]));
        }
    }
    let inv = inverse_derivative_mod(g)?;
    let residues: Vec<Poly> = reduced.simple.iter().map(|m| (m * &inv).rem(g)).collect();
    if residues.iter().any(|r| !r.is_zero()) {
        let roots = factor_roots(arena, g, var)?;
        let sum = sum_over_roots(arena, g, &roots, |arena, rho| {
            let x_minus = arena.sub(var, rho);
            let w = arena.mul(&[special.alpha, x_minus]);
            let w = crate::transforms::expand::expand(arena, w);
            let shift = arena.mul(&[special.alpha, rho]);
            let beta = arena.add(&[special.beta, shift]);
            let beta = crate::transforms::eval::eval(arena, beta);
            let mut parts: SmallVec<[ExprId; 2]> = SmallVec::new();
            for (r, &(f, _, _)) in residues.iter().zip(closure.iter()) {
                if r.is_zero() {
                    continue;
                }
                let c = residue_at(arena, r, rho);
                let base = linear_special_base(arena, f, w, beta);
                parts.push(arena.mul(&[c, base]));
            }
            arena.add(&parts)
        });
        terms.push(sum);
    }
    Some(match terms.len() {
        0 => arena.zero,
        1 => terms[0],
        _ => arena.add(&terms),
    })
}

/// `r(ρ)` for a polynomial `r ∈ ℚ[t]`, multiplied out when that is no
/// larger (`u·ρ + v` with `ρ = p ± √q` collects to `u·p + v ± u·√q`, and
/// `ρ²` with `ρ = (1 + i)/√2` is `i`; Cardano's roots multiplied out are
/// larger).
fn residue_at(arena: &mut Arena, r: &Poly, rho: ExprId) -> ExprId {
    let e = crate::poly::polybridge::poly_to_expr(arena, r, rho);
    let e = crate::transforms::eval::eval(arena, e);
    if matches!(arena.node(rho), ExprNode::Symbol(_)) {
        return e;
    }
    let expanded = crate::transforms::expand::expand(arena, e);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    let size = |arena: &Arena, id: ExprId| crate::base::walk::post_order_ids(arena, id).len();
    if size(arena, expanded) <= size(arena, e) {
        expanded
    } else {
        e
    }
}

/// Largest degree of an irreducible factor whose roots [`factor_roots`]
/// asks `solve` for (the quadratic formula, Cardano, Ferrari).
const MAX_EXPLICIT_ROOT_DEGREE: usize = 4;

/// Largest root (nodes written out) that [`factor_roots`] uses explicitly;
/// Ferrari's roots of a general quartic are far larger and stay implicit
/// (with them, `∫ ln(−x)·(−x² + 2x − 2)/(x⁴ − x² + 2x + 1) dx` was a 16 KB
/// answer).
const MAX_EXPLICIT_ROOT_SIZE: usize = 48;

/// The number of nodes of `e` written out as a tree, counted up to `cap`.
fn written_size(arena: &Arena, e: ExprId, cap: usize) -> usize {
    let mut size: FxHashMap<ExprId, usize> = FxHashMap::default();
    for id in crate::base::walk::post_order_ids(arena, e) {
        let mut s: usize = 1;
        arena
            .node(id)
            .for_each_child(|c| s = s.saturating_add(size.get(&c).copied().unwrap_or(1)));
        size.insert(id, s.min(cap.saturating_add(1)));
    }
    size.get(&e).copied().unwrap_or(1)
}

/// Largest degree of an irreducible factor summed over in a `RootSum`.
const MAX_ROOT_SUM_DEGREE: usize = 16;

/// How the roots of an irreducible factor of a denominator appear in an
/// antiderivative.
enum FactorRoots {
    /// Every root in radicals, from `solve`.
    Explicit(Vec<ExprId>),
    /// A `RootSum` over the factor.
    Implicit,
}

/// The roots of `g` (irreducible over ℚ, degree ≥ 2): explicit when `solve`
/// writes every one of them in radicals of at most
/// [`MAX_EXPLICIT_ROOT_SIZE`] nodes, and the real ones without `i` (Cardano's
/// formula writes the three real roots of `x³ − 3x² + 1` with complex cube
/// roots, whose rounded imaginary parts would decide the side of the cut of
/// `Ei` and `Ci` at every point); otherwise implicit, up to
/// [`MAX_ROOT_SUM_DEGREE`] (a `RootSum` is evaluated at its certified roots,
/// the real ones exactly real).
fn factor_roots(arena: &mut Arena, g: &Poly, var: ExprId) -> Option<FactorRoots> {
    let deg = g.degree()?;
    if deg > MAX_ROOT_SUM_DEGREE {
        return None;
    }
    if deg <= MAX_EXPLICIT_ROOT_DEGREE {
        let g_expr = crate::poly::polybridge::poly_to_expr(arena, g, var);
        let roots: Vec<ExprId> = crate::transforms::solve::known_real_roots(arena, g_expr, var);
        let small = roots.iter().all(|&r| {
            let ids = crate::base::walk::post_order_ids(arena, r);
            written_size(arena, r, MAX_EXPLICIT_ROOT_SIZE) <= MAX_EXPLICIT_ROOT_SIZE
                && !ids.iter().any(|&id| {
                    id == var
                        || matches!(
                            arena.node(id),
                            ExprNode::RootOf(..) | ExprNode::Symbol(_) | ExprNode::Integral(..)
                        )
                })
        });
        let written_real = roots
            .iter()
            .filter(|&&r| {
                !crate::base::walk::post_order_ids(arena, r)
                    .iter()
                    .any(|&id| matches!(arena.node(id), ExprNode::ImaginaryUnit))
            })
            .count();
        if roots.len() == deg
            && small
            && written_real == crate::poly::sturm::SturmChain::new(g).count_real_roots()
        {
            return Some(FactorRoots::Explicit(roots));
        }
    }
    Some(FactorRoots::Implicit)
}

/// `Σ_{g(ρ)=0} body(ρ)`: the explicit sum, or `RootSum(g(t), t ↦ body(t))`.
fn sum_over_roots(
    arena: &mut Arena,
    g: &Poly,
    roots: &FactorRoots,
    mut body: impl FnMut(&mut Arena, ExprId) -> ExprId,
) -> ExprId {
    match roots {
        FactorRoots::Explicit(rs) => {
            let terms: SmallVec<[ExprId; 4]> = rs.iter().map(|&rho| body(arena, rho)).collect();
            arena.add(&terms)
        }
        FactorRoots::Implicit => {
            let t = arena.symbol("__rs_t");
            let b = body(arena, t);
            let g_t = crate::poly::polybridge::poly_to_expr(arena, g, t);
            arena.intern(ExprNode::RootSum(g_t, b, t))
        }
    }
}

/// `∫ ln(ℓ)⁻ⁿ dx` for `ℓ = α·x + β` and an integer `1 ≤ n ≤`
/// [`MAX_SPECIAL_ORDER`]: `t = ln ℓ` makes it `∫ eᵗ·t⁻ⁿ dt/α`, by parts
/// `Jₙ = −ℓ·ln(ℓ)^{1−n}/(n − 1) + J_{n−1}/(n − 1)` down to `J₁ = li(ℓ)`
/// (SymPy's `LiRule`).  Up to 0.43 only `∫ dx/ln x` was recognised: `∫
/// dx/ln(2x + 1)` and `∫ dx/ln(x)²` stayed unevaluated.  `li` has its
/// singularity at `ℓ = 1`, a pole of the integrand, and the integrand is
/// real for `ℓ > 0` only.
pub(super) fn try_inverse_log_power(
    arena: &mut Arena,
    base: ExprId,
    exp: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let ExprNode::Ln(ell) = *arena.node(base) else {
        return None;
    };
    let n = arena
        .as_num(exp)
        .filter(|q| q.is_integer() && q.is_negative())?;
    let n = i64::try_from(-n.to_integer()).ok()?;
    if n > MAX_SPECIAL_ORDER {
        return None;
    }
    let (alpha, _) = symbolic_linear_coeff_of(arena, ell, var, var_sym)?;
    let cases = vanishing_parameter_values(arena, alpha, var_sym)?;
    let mut j = arena.li(ell);
    for m in 2..=n {
        let inv = arena.rational(1, m - 1);
        let power = arena.int(1 - m);
        let ln_pow = arena.pow(base, power);
        let neg_inv = arena.neg(inv);
        let boundary = arena.mul(&[neg_inv, ell, ln_pow]);
        let carried = arena.mul(&[inv, j]);
        j = arena.add(&[boundary, carried]);
    }
    let generic = arena.div(j, alpha);
    let integrand = arena.pow(base, exp);
    with_parameter_cases(arena, generic, integrand, &cases, var, var_sym, depth)
}

/// The product of `factors` with every power `sⁿ` (`n` a negative integer)
/// of a sum `s = g·h` whose terms share a factor `g = c·xᵏ`, with a
/// transcendental function of `x` in `h`, or `g = c·e^u`, written `gⁿ·hⁿ`
/// ([`Arena::factor_terms_pair_expr`]; exact for an integer `n`); `None`
/// when there is no such power.  (Algebraic cofactors are the rational and
/// radical integrators' business: splitting the `t·(…)` of the integrands
/// that the trigonometric substitutions build sent the Rubi suite's 4.1.2.1
/// from 9 s to 220 s, 13 timeouts, for answers with `tan(…)¹²²`.)  The
/// canonical form keeps `x·ln x − x` a sum, so the substitution `u = ln x`
/// did not see the `1/x` of `∫ dx/(x·ln x − x)`, which stayed unevaluated
/// up to 0.43 while `∫ dx/(x·(ln x − 1))` worked (SymPy:
/// `log(log(x) − 1)`).
pub(super) fn split_common_factors(
    arena: &mut Arena,
    factors: &[ExprId],
    var_sym: SymbolId,
) -> Option<ExprId> {
    let mut changed = false;
    let mut out: SmallVec<[ExprId; 6]> = SmallVec::new();
    for &f in factors {
        if let ExprNode::Pow(base, e) = *arena.node(f)
            && matches!(arena.node(base), ExprNode::Add(_))
            && arena
                .as_num(e)
                .is_some_and(|q| q.is_integer() && q.is_negative())
        {
            let (g, h) = arena.factor_terms_pair_expr(base);
            if contains_var(arena, g, var_sym)
                && contains_var(arena, h, var_sym)
                && ((is_monomial_in_var(arena, g, var_sym)
                    && has_transcendental_of_var(arena, h, var_sym))
                    || is_exponential_of_var(arena, g, var_sym))
            {
                let g_e = arena.pow(g, e);
                let h_e = arena.pow(h, e);
                out.push(g_e);
                out.push(h_e);
                changed = true;
                continue;
            }
        }
        out.push(f);
    }
    changed.then(|| arena.mul(&out))
}

/// `∫ e^{a·x² + b·x + c} dx` for coefficients the rational branch of the
/// `Exp` arm does not take (`√2`, a parameter): with `s² = −a`,
/// `√π/(2s)·e^{c − b²/(4a)}·erf(s·x − b/(2s))` when `a` is written with a
/// negative sign, else with `t² = a`, `√π/(2t)·e^{c − b²/(4a)}·erfi(t·x +
/// b/(2t))` (SymPy's `ErfRule`).  Both hold for every complex `a ≠ 0`
/// (only `s² = −a`, `t² = a` is used), and `erf`, `erfi` are entire; the
/// case `a = 0` comes from the stage's wrap (the denominator `√a`).  Up to
/// 0.43 `∫ e^{−a·x²} dx` stayed unevaluated.
pub(super) fn gaussian_integral(
    arena: &mut Arena,
    q: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let (a, b, c) = quadratic_argument_coeffs(arena, q, var, var_sym)?;
    if arena.is_zero_structural(a) {
        return None;
    }
    let two = arena.int(2);
    let four = arena.int(4);
    let b2 = arena.pow(b, two);
    let four_a = arena.mul(&[four, a]);
    let b2_4a = arena.div(b2, four_a);
    let shift = arena.sub(c, b2_4a);
    let shift = crate::transforms::eval::eval(arena, shift);
    let pi = arena.pi();
    let sqrt_pi = arena.sqrt(pi);
    let negative = has_negative_sign(arena, a);
    let r = if negative {
        let neg_a = arena.neg(a);
        let neg_a = crate::transforms::eval::eval(arena, neg_a);
        arena.sqrt(neg_a)
    } else {
        arena.sqrt(a)
    };
    let rx = arena.mul(&[r, var]);
    let two_r = arena.mul(&[two, r]);
    let b_2r = arena.div(b, two_r);
    let front = arena.div(sqrt_pi, two_r);
    let special = if negative {
        let arg = arena.sub(rx, b_2r);
        arena.erf(arg)
    } else {
        let arg = arena.add(&[rx, b_2r]);
        special_apply(arena, crate::base::libfn::LibFn::Erfi, arg)
    };
    Some(if arena.is_zero_structural(shift) {
        arena.mul(&[front, special])
    } else {
        let e = arena.exp(shift);
        arena.mul(&[front, e, special])
    })
}

/// `∫ sin(q) dx` (`sine`) or `∫ cos(q) dx` for `q = a·x² + b·x + c`:
/// `√π/√(2a)·(cos θ·S(u) − sin θ·C(u))` and `√π/√(2a)·(cos θ·C(u) + sin θ·S(u))`
/// with `u = (2a·x + b)/(√(2a)·√π)`, `θ = b²/(4a) − c` and the Fresnel
/// integrals `S`, `C` (SymPy's `FresnelSRule`, `FresnelCRule`): `πu²/2 = q +
/// θ`, and only `√(2a)² = 2a` is used, so it holds for every complex `a ≠ 0`;
/// `S`, `C` are entire.  A leading coefficient written with a negative sign
/// is made positive first (`sin q = −sin(−q)`, `cos q = cos(−q)`), which keeps
/// `√(2a)` real for real coefficients.  Up to 0.43 `∫ sin(x²) dx` stayed
/// unevaluated.
pub(super) fn fresnel_integral(
    arena: &mut Arena,
    sine: bool,
    q: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use crate::base::libfn::LibFn;
    let (a, b, c) = quadratic_argument_coeffs(arena, q, var, var_sym)?;
    if arena.is_zero_structural(a) {
        return None;
    }
    let (a, b, c, sign) = if has_negative_sign(arena, a) {
        let flip = |arena: &mut Arena, e: ExprId| {
            let n = arena.neg(e);
            crate::transforms::eval::eval(arena, n)
        };
        (
            flip(arena, a),
            flip(arena, b),
            flip(arena, c),
            if sine { -1 } else { 1 },
        )
    } else {
        (a, b, c, 1)
    };
    let two = arena.int(2);
    let four = arena.int(4);
    let b2 = arena.pow(b, two);
    let four_a = arena.mul(&[four, a]);
    let b2_4a = arena.div(b2, four_a);
    let theta = arena.sub(b2_4a, c);
    let theta = crate::transforms::eval::eval(arena, theta);
    let two_a = arena.mul(&[two, a]);
    let root = arena.sqrt(two_a);
    let pi = arena.pi();
    let sqrt_pi = arena.sqrt(pi);
    let two_a_x = arena.mul(&[two_a, var]);
    let lin = arena.add(&[two_a_x, b]);
    let scale = arena.mul(&[root, sqrt_pi]);
    let u = arena.div(lin, scale);
    let s_u = special_apply(arena, LibFn::FresnelS, u);
    let c_u = special_apply(arena, LibFn::FresnelC, u);
    let (main, other) = if sine { (s_u, c_u) } else { (c_u, s_u) };
    let body = if arena.is_zero_structural(theta) {
        main
    } else {
        let ct = parity_folded(arena, LinearSpecial::Cos, theta);
        let st = parity_folded(arena, LinearSpecial::Sin, theta);
        let m = arena.mul(&[ct, main]);
        let o = arena.mul(&[st, other]);
        if sine {
            arena.sub(m, o)
        } else {
            arena.add(&[m, o])
        }
    };
    let front = arena.div(sqrt_pi, root);
    let sign = arena.int(sign);
    let result = arena.mul(&[sign, front, body]);
    Some(crate::transforms::eval::eval(arena, result))
}

/// Largest degree of the polynomial of [`try_poly_times_quadratic`].
const MAX_GAUSSIAN_DEGREE: usize = 12;

/// The coefficients `[p₀, …, pₙ]` (free of the variable) of the polynomial
/// `p` in the variable, multiplied out; `None` when `p` is not one or its
/// degree exceeds `max_degree`.
pub(super) fn poly_coeffs_in(
    arena: &mut Arena,
    p: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    max_degree: usize,
) -> Option<Vec<ExprId>> {
    let p = crate::transforms::expand::expand(arena, p);
    let terms: SmallVec<[ExprId; 6]> = match arena.node(p) {
        ExprNode::Add(children) => children.clone(),
        _ => SmallVec::from_slice(&[p]),
    };
    let mut by_degree: Vec<Vec<ExprId>> = Vec::new();
    for t in terms {
        let (c, n) = if !contains_var(arena, t, var_sym) {
            (t, 0usize)
        } else {
            let (c, dep) = match split_constant_factors(arena, t, var_sym) {
                Some((c, dep)) => (arena.mul(&c), dep),
                None => (arena.one, t),
            };
            let n = if dep == var {
                1
            } else {
                let ExprNode::Pow(b, e) = *arena.node(dep) else {
                    return None;
                };
                if b != var {
                    return None;
                }
                let e = arena
                    .as_num(e)
                    .filter(|q| q.is_integer() && !q.is_negative())?;
                usize::try_from(e.to_integer()).ok()?
            };
            (c, n)
        };
        if n > max_degree {
            return None;
        }
        if by_degree.len() <= n {
            by_degree.resize(n + 1, Vec::new());
        }
        by_degree[n].push(c);
    }
    Some(
        by_degree
            .into_iter()
            .map(|cs| match cs.len() {
                0 => arena.zero,
                1 => cs[0],
                _ => arena.add(&cs),
            })
            .collect(),
    )
}

/// `∫ P(x)·f(q) dx` for a polynomial `P` and `f(q)` = `e^q`, `sin q`, `cos q`
/// with `q = a·x² + b·x + c` (`a ≠ 0`): `xⁿ·f(q) = xⁿ⁻¹·(q′ − b)·f(q)/(2a)`,
/// and by parts on `xⁿ⁻¹·q′·f(q)`
///
/// - `Iₙ = xⁿ⁻¹·e^q/(2a) − (n − 1)/(2a)·Iₙ₋₂ − b/(2a)·Iₙ₋₁`,
/// - `Sₙ = (−xⁿ⁻¹·cos q + (n − 1)·Cₙ₋₂ − b·Sₙ₋₁)/(2a)`,
/// - `Cₙ = (xⁿ⁻¹·sin q − (n − 1)·Sₙ₋₂ − b·Cₙ₋₁)/(2a)`,
///
/// down to `∫ f(q)` (`erf`/`erfi`, Fresnel), which is integrated only when
/// the answer needs it.  By parts with `u = P` took `dv = e^q`, an `erf`,
/// and `∫ x²·e^{−x²} dx` stayed unevaluated up to 0.43 (SymPy: `−x·e^{−x²}/2 +
/// √π·erf(x)/4`).
pub(super) fn try_poly_times_quadratic(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let mut special: Option<(LinearSpecial, ExprId, ExprId, ExprId)> = None;
    let mut polys: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &d in dependent {
        if special.is_none()
            && let Some((fam, q)) = LinearSpecial::of(arena, d)
            && matches!(
                fam,
                LinearSpecial::Exp | LinearSpecial::Sin | LinearSpecial::Cos
            )
            && let Some((a, b, _)) = quadratic_argument_coeffs(arena, q, var, var_sym)
            && !arena.is_zero_structural(a)
        {
            special = Some((fam, q, a, b));
        } else if is_polynomial_in(arena, d, var, var_sym) {
            polys.push(d);
        } else {
            return None;
        }
    }
    let (fam, q, a, b) = special?;
    if polys.is_empty() {
        return None;
    }
    let p = arena.mul(&polys);
    let coeffs = poly_coeffs_in(arena, p, var, var_sym, MAX_GAUSSIAN_DEGREE)?;
    let fams: SmallVec<[LinearSpecial; 2]> = if fam == LinearSpecial::Exp {
        SmallVec::from_slice(&[LinearSpecial::Exp])
    } else {
        SmallVec::from_slice(&[LinearSpecial::Sin, LinearSpecial::Cos])
    };
    // The integrals `∫ f(q)` as placeholders until the answer is known.
    let bases: SmallVec<[ExprId; 2]> = fams
        .iter()
        .enumerate()
        .map(|(k, _)| arena.symbol(&format!("__gauss_base{k}")))
        .collect();
    let two = arena.int(2);
    let two_a = arena.mul(&[two, a]);
    let minus_one = arena.int(-1);
    let inv_2a = arena.pow(two_a, minus_one);
    let at = |f: LinearSpecial| fams.iter().position(|&g| g == f).unwrap_or(0);
    // table[n][k] = ∫ xⁿ·fams[k](q)
    let mut table: Vec<SmallVec<[ExprId; 2]>> = vec![bases.clone()];
    for n in 1..coeffs.len() {
        let n_minus_1 = i64::try_from(n - 1).ok()?;
        let power = arena.int(n_minus_1);
        let x_pow = arena.pow(var, power);
        let mut row: SmallVec<[ExprId; 2]> = SmallVec::new();
        for &f in &fams {
            // xⁿ⁻¹·F(q) with F′ = f (F = e^q, −cos q, sin q), and the
            // integral by parts leaves −(n − 1)·∫ xⁿ⁻²·F(q).
            let (antider, minus_partner) = match f {
                LinearSpecial::Exp => (arena.exp(q), (LinearSpecial::Exp, 1)),
                LinearSpecial::Sin => {
                    let c = arena.cos(q);
                    (arena.neg(c), (LinearSpecial::Cos, -1))
                }
                _ => (arena.sin(q), (LinearSpecial::Sin, 1)),
            };
            let mut terms: SmallVec<[ExprId; 3]> = SmallVec::new();
            terms.push(arena.mul(&[x_pow, antider]));
            if n >= 2 {
                let (g, s) = minus_partner;
                let coeff = arena.int(-s * n_minus_1);
                terms.push(arena.mul(&[coeff, table[n - 2][at(g)]]));
            }
            if !arena.is_zero_structural(b) {
                let neg_b = arena.neg(b);
                terms.push(arena.mul(&[neg_b, table[n - 1][at(f)]]));
            }
            let sum = arena.add(&terms);
            row.push(arena.mul(&[inv_2a, sum]));
        }
        table.push(row);
    }
    let k = at(fam);
    let mut parts: SmallVec<[ExprId; 8]> = SmallVec::new();
    for (n, &c) in coeffs.iter().enumerate() {
        if !arena.is_zero_structural(c) {
            parts.push(arena.mul(&[c, table[n][k]]));
        }
    }
    let sum = arena.add(&parts);
    let mut result = crate::transforms::eval::eval(arena, sum);
    for (&f, &base) in fams.iter().zip(&bases) {
        if !crate::base::walk::contains(arena, result, base) {
            continue;
        }
        let f_q = f.apply(arena, q);
        let integral = integrate_node(arena, f_q, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, integral) {
            return None;
        }
        result = crate::transforms::subs::subs(arena, result, base, integral);
    }
    tracing::debug!("integrate: polynomial times a function of a quadratic, by parts");
    Some(crate::transforms::eval::eval(arena, result))
}

/// The dilogarithm integrals `∫ ln(β + α·xᵏ)/x dx = ln β·ln|x| −
/// Li₂(−α·xᵏ/β)/k` for a positive `β` (`ln(β·(1 + z)) = ln β + ln(1 + z)`
/// needs it) and a number `k ≠ 0`, and
/// `∫ ln(x)/(β + α·x) dx = (ln x·ln(1 + α·x/β) + Li₂(−α·x/β))/α` (an identity of
/// analytic functions; SymPy's `PolylogRule` and `manualintegrate`'s
/// answers).  `Li₂(z)` has its cut on `z ≥ 1`, where `ln(1 − z)` has its own
/// (`evalf` takes both from the same side), so the answers are continuous
/// where the integrands are.  Up to 0.43 both stayed unevaluated (by parts
/// turned one into the other).
pub(super) fn try_dilog_integral(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    use crate::base::libfn::LibFn;
    let [f, g] = dependent[..] else {
        return None;
    };
    let li2 = |arena: &mut Arena, z: ExprId| -> ExprId {
        let two = arena.int(2);
        arena.lib_apply(LibFn::PolyLog, &[two, z])
    };
    let is_inverse = |arena: &Arena, e: ExprId| -> Option<ExprId> {
        match *arena.node(e) {
            ExprNode::Pow(base, k)
                if arena.as_num(k).is_some_and(|q| *q == -Q::one())
                    && contains_var(arena, base, var_sym) =>
            {
                Some(base)
            }
            _ => None,
        }
    };
    for (ln_factor, other) in [(f, g), (g, f)] {
        let ExprNode::Ln(ell) = *arena.node(ln_factor) else {
            continue;
        };
        let Some(den) = is_inverse(arena, other) else {
            continue;
        };
        if den == var && ell != var {
            // ln(β + α·xᵏ)/x
            let ExprNode::Add(terms) = arena.node(ell).clone() else {
                return None;
            };
            let (dep, free): (SmallVec<[ExprId; 4]>, SmallVec<[ExprId; 4]>) = terms
                .iter()
                .copied()
                .partition(|&t| contains_var(arena, t, var_sym));
            let [monomial] = dep[..] else {
                return None;
            };
            if free.is_empty() {
                return None;
            }
            let (alpha, k) = monomial_parts(arena, monomial, var, var_sym)?;
            arena.as_num(k)?;
            let beta = arena.add(&free);
            let positive = arena.as_num(beta).is_some_and(Signed::is_positive)
                || Truth::from(crate::base::assumptions::AssumptionCache::new().query(
                    arena,
                    beta,
                    crate::base::assumptions::Props::POSITIVE,
                ))
                .is_true();
            if !positive {
                return None;
            }
            let ratio = arena.div(alpha, beta);
            let x_k = arena.pow(var, k);
            let kx = arena.mul(&[ratio, x_k]);
            let z = arena.neg(kx);
            let z = crate::transforms::eval::eval(arena, z);
            let dilog = li2(arena, z);
            let minus_one = arena.int(-1);
            let inv_k = arena.pow(k, minus_one);
            let dilog = arena.mul(&[minus_one, inv_k, dilog]);
            if beta == arena.one {
                return Some(dilog);
            }
            let ln_beta = arena.ln(beta);
            let abs_x = arena.abs(var);
            let ln_x = arena.ln(abs_x);
            let log_part = arena.mul(&[ln_beta, ln_x]);
            return Some(arena.add(&[log_part, dilog]));
        }
        if ell == var && den != var {
            // ln(x)/(β + α·x)
            let (alpha, beta) = symbolic_linear_coeff_of(arena, den, var, var_sym)?;
            if arena.is_zero_structural(beta) {
                return None;
            }
            let k = arena.div(alpha, beta);
            let kx = arena.mul(&[k, var]);
            let kx = crate::transforms::eval::eval(arena, kx);
            let one = arena.one;
            let one_kx = arena.add(&[one, kx]);
            let ln_one_kx = arena.ln(one_kx);
            let product = arena.mul(&[ln_factor, ln_one_kx]);
            let z = arena.neg(kx);
            let dilog = li2(arena, z);
            let sum = arena.add(&[product, dilog]);
            return Some(arena.div(sum, alpha));
        }
    }
    None
}

/// `Li₂(z)`, the `polylog(2, z)` node.
fn dilog(arena: &mut Arena, z: ExprId) -> ExprId {
    let two = arena.int(2);
    arena.lib_apply(crate::base::libfn::LibFn::PolyLog, &[two, z])
}

/// `∫ ln(L)·R(x) dx` for a linear `L = a·x + b` (`a`, `b` rational) and
/// `R ∈ ℚ(x)` with a non-constant denominator.  Hermite's reduction
/// `∫ R = g + ∫ h` (`g` rational, `h = M/D` proper with `D` square-free)
/// gives by parts `ln(L)·g − ∫ a·g/L dx + ∫ ln(L)·h dx`: the middle integral
/// is rational, and `h = Σ_ρ r(ρ)/(x − ρ)` over the roots of `D` (residues
/// `r = M·D′⁻¹ mod D` on each irreducible factor, summed as in
/// [`special_over_rational_factors`]) with
/// `∫ ln(L)/(x − ρ) dx = ln(L)·ln(1 − L/c) + Li₂(L/c)` for `c = a·ρ + b ≠ 0`
/// (`u = L` turns it into `∫ ln(u)/(u − c) du`) and `ln(L)²/2` for `c = 0`.
/// By parts alone (`u = ln L`) leaves `∫ ln(1 − L/c)/L`, and up to 0.44
/// `∫ ln(x)/(x² − 2) dx`, `∫ ln(2x + 1)/(x − 3) dx` stayed unevaluated.
/// With `nonlinear_only` (the call before by parts) only when the
/// denominator of `h` has a factor of degree ≥ 2: over linear factors by
/// parts may meet [`try_dilog_integral`] and keeps its turn.
///
/// The integrand is real for `L > 0`.  There the terms of a non-real `c`
/// never meet the cuts of `ln(1 − z)` and `Li₂(z)` (`z = L/c` is real only for
/// a real `c`), and a pair of conjugate roots gives conjugate terms; for a
/// real `c > 0` both cuts start at `L = c`, a pole of the integrand, beyond
/// which their imaginary parts `π·ln L` and `−π·ln(L/c)` add up to a constant.
pub(super) fn try_log_times_rational(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
    nonlinear_only: bool,
) -> Option<ExprId> {
    if depth < 2 {
        return None;
    }
    let idx = dependent
        .iter()
        .position(|&d| matches!(arena.node(d), ExprNode::Ln(_)))?;
    let ln_l = dependent[idx];
    let ExprNode::Ln(ell) = *arena.node(ln_l) else {
        return None;
    };
    let lp = crate::poly::polybridge::expr_to_poly(arena, ell, var)?;
    if lp.degree() != Some(1) {
        return None;
    }
    let (a, b) = (lp.coeff(1), lp.coeff(0));
    let rest = remaining_product(arena, dependent, idx);
    if !is_rational_in(arena, rest, var_sym) {
        return None;
    }
    let (numer, denom) = rational_over_q(arena, rest, var)?;
    if denom.degree()? > MAX_SPECIAL_DENOM_DEGREE {
        return None;
    }
    let hr = crate::calculus::risch::hermite::hermite_reduce(&numer, &denom)?;
    if hr.h_numer.is_zero() {
        // Elementary: by parts' business.
        return None;
    }
    let (content, factors) = hr.h_denom.factor_over_z();
    if content.is_zero()
        || (nonlinear_only
            && factors
                .iter()
                .all(|(g, _)| g.degree().is_none_or(|d| d < 2)))
    {
        return None;
    }
    let mut terms: SmallVec<[ExprId; 6]> = SmallVec::new();
    if !hr.g_numer.is_zero() {
        let gn = crate::poly::polybridge::poly_to_expr(arena, &hr.g_numer, var);
        let gd = crate::poly::polybridge::poly_to_expr(arena, &hr.g_denom, var);
        let g = arena.div(gn, gd);
        terms.push(arena.mul(&[ln_l, g]));
        let a_e = arena.num_ratio(a.clone());
        let minus_one = arena.int(-1);
        let inv_l = arena.pow(ell, minus_one);
        let dv = arena.mul(&[a_e, g, inv_l]);
        let dv = crate::transforms::eval::eval(arena, dv);
        let v = integrate_node(arena, dv, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, v) {
            return None;
        }
        terms.push(arena.neg(v));
    }
    let scaled = hr.h_numer.scale(&(Q::one() / content));
    for term in crate::transforms::apart::partial_fractions_over_q(&scaled, &factors) {
        let (m, g, k) = (term.numer, term.factor, term.power);
        if m.is_zero() {
            continue;
        }
        if k != 1 {
            return None;
        }
        let inv = inverse_derivative_mod(&g)?;
        let r = (&m * &inv).rem(&g);
        let roots = if g.degree() == Some(1) {
            let rho = arena.num_ratio(-(g.coeff(0) / g.coeff(1)));
            FactorRoots::Explicit(vec![rho])
        } else {
            factor_roots(arena, &g, var)?
        };
        let sum = sum_over_roots(arena, &g, &roots, |arena, rho| {
            let res = residue_at(arena, &r, rho);
            let a_e = arena.num_ratio(a.clone());
            let b_e = arena.num_ratio(b.clone());
            let a_rho = arena.mul(&[a_e, rho]);
            let c = arena.add(&[a_rho, b_e]);
            let c = crate::transforms::eval::eval(arena, c);
            let term = if arena.is_zero_structural(c) {
                let two = arena.int(2);
                let sq = arena.pow(ln_l, two);
                let half = arena.rational(1, 2);
                arena.mul(&[half, sq])
            } else {
                let z = arena.div(ell, c);
                let z = crate::transforms::eval::eval(arena, z);
                let one = arena.one;
                let one_minus = arena.sub(one, z);
                let one_minus = crate::transforms::eval::eval(arena, one_minus);
                let ln2 = arena.ln(one_minus);
                let product = arena.mul(&[ln_l, ln2]);
                let li = dilog(arena, z);
                arena.add(&[product, li])
            };
            arena.mul(&[res, term])
        });
        terms.push(sum);
    }
    tracing::debug!("integrate: a logarithm times a rational function, by its residues");
    let big_f = arena.add(&terms);
    let product = arena.mul(dependent);
    (!candidate_rejected(arena, product, big_f, var, var_sym)).then_some(big_f)
}

/// Does `h` contain a logarithm, exponential, trigonometric or hyperbolic
/// function of the variable?
fn has_transcendental_of_var(arena: &Arena, h: ExprId, var_sym: SymbolId) -> bool {
    crate::base::walk::post_order_ids(arena, h)
        .into_iter()
        .any(|id| match arena.node(id) {
            ExprNode::Ln(a)
            | ExprNode::Exp(a)
            | ExprNode::Sin(a)
            | ExprNode::Cos(a)
            | ExprNode::Tan(a)
            | ExprNode::Sinh(a)
            | ExprNode::Cosh(a)
            | ExprNode::Tanh(a) => contains_var(arena, *a, var_sym),
            _ => false,
        })
}

/// Is `g` an exponential `e^u` (times constants) with `u` depending on the
/// variable?  (`e^w/(e^{2w}·w + e^{2w}/2)`, what the substitution `x = e^w`
/// makes of `∫ dx/(x²·ln x + x²/2)`, is `e^{−w}/(w + 1/2)`.)
fn is_exponential_of_var(arena: &Arena, g: ExprId, var_sym: SymbolId) -> bool {
    let is_exp = |f: ExprId| matches!(arena.node(f), ExprNode::Exp(_));
    match arena.node(g) {
        ExprNode::Mul(children) => {
            children.iter().filter(|&&c| is_exp(c)).count() == 1
                && children
                    .iter()
                    .all(|&c| is_exp(c) || !contains_var(arena, c, var_sym))
        }
        _ => is_exp(g),
    }
}

/// Steps of `integrate_node` for the integrand of [`split_common_factors`].
pub(super) const SPLIT_STEPS: u64 = 1_000;

/// Is `g` the variable to a numeric power times constants (`x`, `3·x²`)?
fn is_monomial_in_var(arena: &Arena, g: ExprId, var_sym: SymbolId) -> bool {
    let is_power = |f: ExprId| match arena.node(f) {
        ExprNode::Symbol(s) => *s == var_sym,
        ExprNode::Pow(b, k) => {
            matches!(arena.node(*b), ExprNode::Symbol(s) if *s == var_sym)
                && arena.as_num(*k).is_some()
        }
        _ => false,
    };
    match arena.node(g) {
        ExprNode::Mul(children) => children
            .iter()
            .all(|&c| is_power(c) || !contains_var(arena, c, var_sym)),
        _ => is_power(g),
    }
}

/// Is the constant `c` certainly non-zero: a non-zero number, non-zero by
/// the declared assumptions, or a symbol-free constant whose certified
/// value is not 0?
pub(super) fn constant_nonzero(arena: &mut Arena, c: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    if let Some(q) = arena.as_num(c) {
        return !q.is_zero();
    }
    if Truth::from(AssumptionCache::new().query(arena, c, Props::NONZERO)).is_true() {
        return true;
    }
    crate::base::walk::free_symbols(arena, c).is_empty()
        && !crate::base::walk::has_unevaluated(arena, c)
        && matches!(
            evalf_settled(arena, c, 20, ZeroSearch::Cap),
            Ok((z, Settled::Certified)) if !z.0.is_zero() || !z.1.is_zero()
        )
}

/// The `(parameter, value)` pairs at which the constant `alpha` vanishes,
/// for an answer that holds for `α ≠ 0` only (`Ei(α·x)` for `∫ e^{αx}/x`):
/// empty when `α` is certainly non-zero, or inside a rationalising
/// substitution ([`GENERIC_ONLY`]); `None` (no answer) when `α` has several
/// parameters or its zeros are not found.  Values the parameter's
/// assumptions exclude are left out.
fn vanishing_parameter_values(
    arena: &mut Arena,
    alpha: ExprId,
    var_sym: SymbolId,
) -> Option<Vec<(ExprId, ExprId)>> {
    if constant_nonzero(arena, alpha) || GENERIC_ONLY.with(std::cell::Cell::get) {
        return Some(Vec::new());
    }
    if contains_var(arena, alpha, var_sym) {
        return None;
    }
    let syms = crate::base::walk::free_symbols(arena, alpha);
    let [s] = syms[..] else {
        return None;
    };
    let solutions = crate::transforms::solve::known_real_roots(arena, alpha, s);
    if solutions.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for v in solutions {
        if !crate::base::walk::free_symbols(arena, v).is_empty() || contains_non_finite(arena, v) {
            return None;
        }
        if Truth::from(crate::transforms::solve::assumptions_hold(arena, s, v)).is_false()
            || nonreal_value_of_real_symbol(arena, s, v)
        {
            continue;
        }
        out.push((s, v));
    }
    Some(out)
}

/// `generic` with the cases `sᵢ = vᵢ` of [`vanishing_parameter_values`]:
/// `Piecewise((generic, s ≠ v), (∫ f|_{s=v}, True))`, nested; `None` when a
/// case has no closed form.
fn with_parameter_cases(
    arena: &mut Arena,
    generic: ExprId,
    integrand: ExprId,
    cases: &[(ExprId, ExprId)],
    var: ExprId,
    var_sym: SymbolId,
    depth: usize,
) -> Option<ExprId> {
    let mut out = generic;
    for &(s, v) in cases {
        let at = crate::transforms::subs::subs(arena, integrand, s, v);
        let at = crate::transforms::eval::eval(arena, at);
        if has_zero_denominator(arena, at) {
            continue;
        }
        let g = integrate_node(arena, at, var, var_sym, depth - 1);
        if crate::base::walk::has_unevaluated(arena, g) {
            return None;
        }
        let condition = arena.ne_(s, v);
        let true_cond = arena.bool_true;
        out = arena.piecewise(&[(out, condition), (g, true_cond)]);
    }
    Some(out)
}

/// The node of a library function with one argument.
pub(super) fn special_apply(
    arena: &mut Arena,
    f: crate::base::libfn::LibFn,
    arg: ExprId,
) -> ExprId {
    arena.lib_apply(f, &[arg])
}

/// Split the factors of a product into those free of `var` and those
/// depending on it.  Dependent factors have their nested powers flattened
/// (`Pow(Pow(a, m), n)` → `Pow(a, m·n)`, or `Pow(|a|, m·n)` when that is
/// the real-valued identity, e.g. `sqrt(x²) = |x|`): the canon layer only
/// merges integer exponents (branch-cut safety), but for integration we
/// need e.g. `Pow(Pow(x²+1, 1/2), -1)` → `Pow(x²+1, -1/2)`.
pub(super) fn partition_factors(
    arena: &mut Arena,
    children: &[ExprId],
    var_sym: SymbolId,
) -> (SmallVec<[ExprId; 4]>, SmallVec<[ExprId; 4]>) {
    let mut constants: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut dependent: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &child in children {
        if contains_var(arena, child, var_sym) {
            let flat = flatten_nested_pow(arena, child, var_sym)
                .or_else(|| exp_power_as_exp(arena, child, var_sym))
                .unwrap_or(child);
            dependent.push(flat);
        } else {
            constants.push(child);
        }
    }
    (constants, dependent)
}

/// `exp(u)^c` as `exp(c·u)` (`c` free of the variable), which holds for
/// an integer `c`, and for every `c` when `u` is real at every real value
/// of the variable ([`real_for_real_variable`]): then `ln(eᵘ) = u`.  The
/// canonical form keeps `exp(x)^√37` (for a complex `x` the identity
/// fails), and up to 0.31 `∫ exp(x)^√37·cos 3x dx` stayed unevaluated
/// while `∫ e^{√37·x}·cos 3x dx` worked (normal-form hunt).
pub(super) fn exp_power_as_exp(
    arena: &mut Arena,
    expr: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let ExprNode::Pow(base, c) = arena.node(expr).clone() else {
        return None;
    };
    let ExprNode::Exp(u) = arena.node(base).clone() else {
        return None;
    };
    if contains_var(arena, c, var_sym) {
        return None;
    }
    let integer = arena.as_num(c).is_some_and(|q| q.is_integer());
    if !integer && !real_for_real_variable(arena, u, var_sym) {
        return None;
    }
    let cu = arena.mul(&[c, u]);
    Some(arena.exp(cu))
}

/// Is `u` real at every real value of the variable?  Its variable-dependent
/// nodes must be real-to-real operations defined on the whole line (those
/// of [`real_on_reals`]); its maximal variable-free subexpressions are
/// decided by [`constant_is_real`] (`√37`, a declared-real `a`).
fn real_for_real_variable(arena: &mut Arena, u: ExprId, var_sym: SymbolId) -> bool {
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    let mut stack = vec![u];
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if !contains_var(arena, id, var_sym) {
            if !constant_is_real(arena, id, &mut reals) {
                return false;
            }
            continue;
        }
        let whole_line = match arena.node(id) {
            ExprNode::Symbol(s) => *s == var_sym,
            ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Exp(_)
            | ExprNode::Atan(_)
            | ExprNode::Sinh(_)
            | ExprNode::Cosh(_)
            | ExprNode::Tanh(_)
            | ExprNode::Abs(_) => true,
            ExprNode::Pow(_, exp) => arena
                .as_num(*exp)
                .is_some_and(|r| r.is_integer() && !r.is_negative()),
            _ => false,
        };
        if !whole_line {
            return false;
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }
    true
}

/// Largest degree of a product that [`expand_polynomial_product`] expands.
const MAX_EXPANDED_DEGREE: u32 = 64;

/// The product of `dependent` multiplied out when every factor is a
/// polynomial in the variable (sums, products and non-negative integer
/// powers of it and of constants) and one is a power of a sum, `(a +
/// b·x²)³` — a sum of monomials integrates term by term.  Up to 0.31
/// `∫ x³·(a + b·x²)³ dx` went through by parts three times (a nested
/// answer), and the substitution `u = cos x` of `∫ cos³x·(a + b·cos²x)³·sin x`
/// did not finish within its depth.  `None` otherwise or beyond
/// [`MAX_EXPANDED_DEGREE`].
pub(super) fn expand_polynomial_product(
    arena: &mut Arena,
    dependent: &[ExprId],
    var_sym: SymbolId,
) -> Option<ExprId> {
    if dependent.len() < 2 {
        return None;
    }
    let mut has_power_of_sum = false;
    let mut degree: u32 = 0;
    for &d in dependent {
        // (the degree of the factor, bounded by exponent products)
        let mut stack: Vec<(ExprId, u32)> = vec![(d, 1)];
        let mut factor_degree: u32 = 0;
        while let Some((id, mult)) = stack.pop() {
            if !contains_var(arena, id, var_sym) {
                continue;
            }
            match arena.node(id) {
                ExprNode::Symbol(_) => factor_degree = factor_degree.max(mult),
                ExprNode::Add(ch) => stack.extend(ch.iter().map(|&c| (c, mult))),
                ExprNode::Mul(ch) => {
                    // A product's degree is at most the sum over its factors:
                    // bound it by the factor count times the largest.
                    let k = u32::try_from(ch.len()).ok()?;
                    stack.extend(ch.iter().map(|&c| (c, mult.saturating_mul(k))));
                }
                ExprNode::Pow(b, n) => {
                    let k = arena
                        .as_num(*n)
                        .filter(|q| q.is_integer() && !q.is_negative())?;
                    let k = u32::try_from(k.to_integer()).ok()?;
                    if matches!(arena.node(*b), ExprNode::Add(_)) && k >= 2 {
                        has_power_of_sum = true;
                    }
                    stack.push((*b, mult.saturating_mul(k)));
                }
                _ => return None,
            }
        }
        degree = degree.saturating_add(factor_degree);
    }
    if !has_power_of_sum || degree > MAX_EXPANDED_DEGREE {
        return None;
    }
    let product = arena.mul(dependent);
    let expanded = crate::transforms::expand::expand(arena, product);
    (expanded != product).then_some(expanded)
}

/// The product of `dependent` with its exponential factors whose exponents
/// are polynomials in the variable merged into one, `exp(u)·exp(v) =
/// exp(u + v)` (true for all complex `u`, `v`); `None` when fewer than two
/// such factors.  The canonical form keeps `exp(x − 2)·exp(−x)` and
/// `exp(√5·x)·exp(−x/3 − 1)`, whose integrals stayed unevaluated up to 0.31
/// (the first is the constant `e⁻²`).  Exponentials of other exponents
/// stay apart: `exp(x)·exp(exp(x))` is the substitution `u = eˣ`.
pub(super) fn merge_exponential_factors(
    arena: &mut Arena,
    dependent: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let mut exponents: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
    for &d in dependent {
        match arena.node(d) {
            ExprNode::Exp(u) if is_polynomial_in(arena, *u, var, var_sym) => exponents.push(*u),
            _ => rest.push(d),
        }
    }
    if exponents.len() < 2 {
        return None;
    }
    let sum = arena.add(&exponents);
    let merged = arena.exp(sum);
    rest.push(merged);
    Some(arena.mul(&rest))
}

/// `(c, f)` with `expr = c·f` when `expr` is a product with factors `c` free
/// of the variable and at least one factor `f` that depends on it (nested
/// powers flattened as in [`partition_factors`]); `None` otherwise.
pub(super) fn split_constant_factors(
    arena: &mut Arena,
    expr: ExprId,
    var_sym: SymbolId,
) -> Option<(SmallVec<[ExprId; 4]>, ExprId)> {
    let ExprNode::Mul(children) = arena.node(expr).clone() else {
        return None;
    };
    let (constants, dependent) = partition_factors(arena, &children, var_sym);
    if constants.is_empty() || dependent.is_empty() {
        return None;
    }
    let f = if dependent.len() == 1 {
        dependent[0]
    } else {
        arena.mul(&dependent)
    };
    Some((constants, f))
}

/// `c·F` for the integrand `expr = c·f` whose constant factors `c` were
/// split off (linearity) and `F = ∫ f`; `None` when that is not an
/// antiderivative of `expr`.
///
/// `F` may differ from `f` where `f` is complex (the real-variable
/// convention: `F′ = f` is checked only where `f` is real), and for a
/// constant `c` that is not real, `c·f` is real exactly at some such
/// points.  `∫ √3·i·√x·|√x| dx` was `√3·i·x²·sign(√x)/2` (`∫ √x·|√x| dx =
/// x²·sign(√x)/2` differs from its integrand only at `x < 0`, where that
/// is imaginary): at `x = −5/7` the integrand is the real `−1.2372` and the
/// answer's derivative `+1.2372` (up to 0.46; the CI Fuzz workflow's
/// `fuzz_integrate`).  So `c·F` is checked as an antiderivative of `c·f` when
/// `c` is not known to be real and `f` or `F` is not analytic in the
/// variable ([`non_analytic_in`]); an `F` built from analytic functions has
/// `F′ = f` along the whole real line once it holds where `f` is real.
pub(super) fn constants_times_antiderivative(
    arena: &mut Arena,
    expr: ExprId,
    f: ExprId,
    big_f: ExprId,
    constants: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let result = wrap_with_constants(arena, big_f, constants);
    if !non_analytic_in(arena, f, var_sym) && !non_analytic_in(arena, big_f, var_sym) {
        return Some(result);
    }
    let c = arena.mul(constants);
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    if constant_is_real(arena, c, &mut reals) {
        return Some(result);
    }
    (!candidate_rejected(arena, expr, result, var, var_sym)).then_some(result)
}

/// Multiply a result by constant factors (if any).
pub(super) fn wrap_with_constants(
    arena: &mut Arena,
    result: ExprId,
    constants: &[ExprId],
) -> ExprId {
    if constants.is_empty() {
        result
    } else {
        let mut all: SmallVec<[ExprId; 4]> = constants.iter().copied().collect();
        all.push(result);
        arena.mul(&all)
    }
}
