//! nth-order linear constant-coefficient ODEs: Σ a_k y^(k) = g(x) (part of [`ode`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// nth-order linear constant-coefficient ODEs: Σ a_k y^(k) = g(x)
// ═══════════════════════════════════════════════════════════════════════════

/// Highest derivative order recognised by the nth-order solver.
const MAX_ODE_ORDER: usize = 12;

/// A linear constant-coefficient ODE `Σ a_k·y^(k) + g(x) = 0`.
pub(super) struct LinearCcOde {
    /// `a_0 … a_n` (ascending derivative order), `a_n ≠ 0`.
    pub(super) coeffs: Vec<Q>,
    /// The forcing terms `g(x)` as they appear in the zero-form expression.
    pub(super) forcing: Vec<ExprId>,
}

/// Derivative chain `[y, y', y'', …]` up to [`MAX_ODE_ORDER`].
pub(super) fn derivative_chain(arena: &mut Arena, func: ExprId, var: ExprId) -> Vec<ExprId> {
    let mut chain = vec![func];
    for k in 1..=MAX_ODE_ORDER {
        let d = arena.intern(ExprNode::Derivative(chain[k - 1], var));
        chain.push(d);
    }
    chain
}

/// Recognise `Σ a_k·y^(k) + g(x) = 0` with rational `a_k`.
pub(super) fn extract_linear_cc(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Option<LinearCcOde> {
    use num_traits::Zero;
    let chain = derivative_chain(arena, func, var);
    let children: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![expr],
    };
    let mut coeffs = vec![Q::zero(); MAX_ODE_ORDER + 1];
    let mut forcing = Vec::new();
    for child in children {
        let (coeff, term) = arena.as_coeff_term(child);
        if let Some(k) = chain.iter().position(|&d| d == term) {
            coeffs[k] += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            forcing.push(child);
        } else {
            return None;
        }
    }
    while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
        coeffs.pop();
    }
    if coeffs.len() < 2 {
        return None; // no derivative at all
    }
    Some(LinearCcOde { coeffs, forcing })
}

/// Gaussian rational `re + im·i` — exact arithmetic for the complex
/// exponential shift used by undetermined coefficients.
#[derive(Clone, Debug, PartialEq)]
struct CQ {
    re: Q,
    im: Q,
}

impl CQ {
    fn new(re: Q, im: Q) -> Self {
        Self { re, im }
    }
    fn real(re: Q) -> Self {
        Self {
            re,
            im: num_traits::Zero::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        num_traits::Zero::is_zero(&self.re) && num_traits::Zero::is_zero(&self.im)
    }
    fn add(&self, o: &CQ) -> CQ {
        CQ::new(&self.re + &o.re, &self.im + &o.im)
    }
    fn sub(&self, o: &CQ) -> CQ {
        CQ::new(&self.re - &o.re, &self.im - &o.im)
    }
    fn mul(&self, o: &CQ) -> CQ {
        CQ::new(
            &self.re * &o.re - &self.im * &o.im,
            &self.re * &o.im + &self.im * &o.re,
        )
    }
    fn div(&self, o: &CQ) -> CQ {
        let denom = &o.re * &o.re + &o.im * &o.im;
        let num = self.mul(&CQ::new(o.re.clone(), -o.im.clone()));
        CQ::new(num.re / &denom, num.im / &denom)
    }
    fn scale(&self, r: &Q) -> CQ {
        CQ::new(&self.re * r, &self.im * r)
    }
}

/// Evaluate `p^(j)(λ) / j!` for a rational polynomial `p` (ascending
/// coefficients) at a Gaussian rational `λ`.
fn shifted_coefficient(p: &[Q], j: usize, lambda: &CQ) -> CQ {
    // p^(j)(λ)/j! = Σ_{k≥j} C(k, j) a_k λ^{k-j}
    let mut acc = CQ::real(num_traits::Zero::zero());
    let mut lambda_pow = CQ::real(num_traits::One::one());
    for (k, a_k) in p.iter().enumerate().skip(j) {
        let binom = Q::from_integer(binomial(k as u64, j as u64));
        let term = lambda_pow.scale(&(a_k * binom));
        acc = acc.add(&term);
        lambda_pow = lambda_pow.mul(lambda);
    }
    acc
}

/// Trigonometric flavour of a forcing term.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TrigKind {
    None,
    Cos,
    Sin,
}

/// One forcing term `c·x^d·e^{ax}·{1 | cos(bx) | sin(bx)}`.
pub(super) struct ForcingTerm {
    coeff: Q,
    degree: usize,
    a: Q,
    b: Q,
    kind: TrigKind,
}

/// Parse a single additive forcing term (free of `y`).
pub(super) fn parse_forcing_term(
    arena: &mut Arena,
    term: ExprId,
    var: ExprId,
) -> Option<ForcingTerm> {
    use num_traits::{Signed, Zero};
    let (mut coeff, t) = arena.as_coeff_term(term);
    let factors: Vec<ExprId> = if t == arena.one {
        Vec::new()
    } else {
        match arena.node(t).clone() {
            ExprNode::Mul(c) => c.to_vec(),
            _ => vec![t],
        }
    };
    let mut degree = 0usize;
    let mut a = Q::zero();
    let mut b = Q::zero();
    let mut kind = TrigKind::None;
    let mut seen_exp = false;
    for f in factors {
        if f == var {
            degree += 1;
            continue;
        }
        match arena.node(f).clone() {
            ExprNode::Pow(base, exp) if base == var => {
                let n = arena.as_num(exp)?.clone();
                if !n.is_integer() || n.is_negative() {
                    return None;
                }
                degree += usize::try_from(n.to_integer()).ok()?;
            }
            ExprNode::Exp(inner) => {
                if seen_exp {
                    return None;
                }
                let (rate, c) = extract_linear_numeric(arena, inner, var)?;
                if !c.is_zero() {
                    return None;
                }
                seen_exp = true;
                a = rate;
            }
            ExprNode::Sin(inner) => {
                if kind != TrigKind::None {
                    return None;
                }
                let (freq, c) = extract_linear_numeric(arena, inner, var)?;
                if !c.is_zero() || freq.is_zero() {
                    return None;
                }
                kind = TrigKind::Sin;
                if freq.is_negative() {
                    coeff = -coeff;
                    b = -freq;
                } else {
                    b = freq;
                }
            }
            ExprNode::Cos(inner) => {
                if kind != TrigKind::None {
                    return None;
                }
                let (freq, c) = extract_linear_numeric(arena, inner, var)?;
                if !c.is_zero() || freq.is_zero() {
                    return None;
                }
                kind = TrigKind::Cos;
                b = freq.abs();
            }
            _ => {
                let r = arena.as_num(f)?;
                coeff *= r.clone();
            }
        }
    }
    Some(ForcingTerm {
        coeff,
        degree,
        a,
        b,
        kind,
    })
}

/// Build `Σ c_k·x^k` from rational coefficients.
fn rat_poly_expr(arena: &mut Arena, coeffs: &[Q], var: ExprId) -> ExprId {
    build_polynomial_expr(arena, coeffs, var)
}

/// Particular solution of `Σ a_k y^(k) = q(x)·e^{ax}·{1|cos bx|sin bx}` via the
/// complex exponential shift: with `λ = a + bi`, `y_p = Re/Im(w(x)·e^{λx})`
/// where `w` solves the triangular system `Σ_j (p^(j)(λ)/j!)·w^(j) = q`.
/// Resonance (λ a root of multiplicity `s`) is handled by the extra
/// factor `x^s` implicit in the degree of `w`.
fn particular_for_group(
    arena: &mut Arena,
    p: &[Q],
    q: &[Q],
    a: &Q,
    b: &Q,
    kind: TrigKind,
    var: ExprId,
) -> Option<ExprId> {
    use num_traits::Zero;
    let lambda = CQ::new(a.clone(), b.clone());
    let n = p.len() - 1;
    // c_j = p^(j)(λ)/j!
    let c: Vec<CQ> = (0..=n)
        .map(|j| shifted_coefficient(p, j, &lambda))
        .collect();
    let s = c.iter().position(|cj| !cj.is_zero())?;
    let d = q.len().checked_sub(1)?;
    let m = d + s;
    let mut w = vec![CQ::real(Q::zero()); m + 1];
    // Solve from the top degree down: for k = d..0,
    //   Σ_{j≥s} c_j·(k+j)!/k!·w_{k+j} = q_k
    for k in (0..=d).rev() {
        let mut rhs = CQ::real(q[k].clone());
        for j in (s + 1)..=n {
            if k + j > m {
                continue;
            }
            let fact = falling_factorial_rat(k + j, j);
            rhs = rhs.sub(&c[j].mul(&w[k + j]).scale(&fact));
        }
        let lead = c[s].scale(&falling_factorial_rat(k + s, s));
        w[k + s] = rhs.div(&lead);
    }
    let wr: Vec<Q> = w.iter().map(|z| z.re.clone()).collect();
    let wi: Vec<Q> = w.iter().map(|z| z.im.clone()).collect();
    let wr_x = rat_poly_expr(arena, &wr, var);
    let wi_x = rat_poly_expr(arena, &wi, var);
    let exp_ax = if a.is_zero() {
        arena.one
    } else {
        let a_id = ode_ratio_to_expr(arena, a);
        let ax = arena.mul(&[a_id, var]);
        arena.exp(ax)
    };
    let body = match kind {
        TrigKind::None => wr_x,
        TrigKind::Cos | TrigKind::Sin => {
            let b_id = ode_ratio_to_expr(arena, b);
            let bx = arena.mul(&[b_id, var]);
            let cos_bx = arena.cos(bx);
            let sin_bx = arena.sin(bx);
            if kind == TrigKind::Cos {
                // Re(w e^{ibx}) = wr cos - wi sin
                let t1 = arena.mul(&[wr_x, cos_bx]);
                let t2 = arena.mul(&[wi_x, sin_bx]);
                arena.sub(t1, t2)
            } else {
                // Im(w e^{ibx}) = wr sin + wi cos
                let t1 = arena.mul(&[wr_x, sin_bx]);
                let t2 = arena.mul(&[wi_x, cos_bx]);
                arena.add(&[t1, t2])
            }
        }
    };
    let y_p = arena.mul(&[exp_ax, body]);
    let y_p = crate::transforms::expand::expand(arena, y_p);
    Some(crate::transforms::eval::eval(arena, y_p))
}

/// A forcing term `c·x^d·e^{ax}·{1 | cos(bx) | sin(bx)}` whose constants
/// `c`, `a`, `b` are real numbers that are not all rational (`e^{√2·x}`,
/// `e^{x/2}·cos(√3·x/2)`), for [`particular_for_group_sym`].
struct SymForcingTerm {
    coeff: ExprId,
    degree: usize,
    a: ExprId,
    b: ExprId,
    kind: TrigKind,
}

/// Is `e` a real constant: free of symbols and of `i`, with a finite real
/// value?
///
/// The `f64` test only selects the real-split method of
/// [`particular_for_group_sym`], whose coefficients are rational in the
/// constants and stay right for a non-real one (s15g4: with
/// `asin(1 + 10⁻⁴⁰)`, imaginary part `−1.4·10⁻²⁰`, as the frequency the
/// residual is 0).
pub(super) fn is_real_constant(arena: &Arena, e: ExprId) -> bool {
    crate::base::walk::free_symbols(arena, e).is_empty()
        && real_radical_form(arena, e)
        && crate::transforms::evalf::evalf_complex64(arena, e)
            .is_ok_and(|z| z.re.is_finite() && z.im == 0.0)
}

/// `(rate, offset)` with `e = rate·var + offset`, both real constants.
fn linear_real_constants(arena: &mut Arena, e: ExprId, var: ExprId) -> Option<(ExprId, ExprId)> {
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, e, var)?;
    if coeffs.len() != 2 || !coeffs.iter().all(|&c| is_real_constant(arena, c)) {
        return None;
    }
    Some((coeffs[1], coeffs[0]))
}

/// [`parse_forcing_term`] with real constant (not necessarily rational)
/// coefficient, rates and frequency.  A phase-free `cos`/`sin`; an offset
/// in the exponent goes into the coefficient.
fn parse_forcing_term_sym(arena: &mut Arena, term: ExprId, var: ExprId) -> Option<SymForcingTerm> {
    let factors: Vec<ExprId> = match arena.node(term).clone() {
        ExprNode::Mul(c) => c.to_vec(),
        ExprNode::Neg(inner) => vec![arena.neg_one, inner],
        _ => vec![term],
    };
    let mut coeff = vec![];
    let mut degree = 0usize;
    let (mut a, mut b) = (arena.zero, arena.zero);
    let mut kind = TrigKind::None;
    let mut seen_exp = false;
    for f in factors {
        if f == var {
            degree += 1;
            continue;
        }
        if !contains_sym_id(arena, f, var) {
            if !is_real_constant(arena, f) {
                return None;
            }
            coeff.push(f);
            continue;
        }
        match arena.node(f).clone() {
            ExprNode::Pow(base, exp) if base == var => {
                let n = arena.as_num(exp)?.clone();
                if !n.is_integer() || n.is_negative() {
                    return None;
                }
                degree += usize::try_from(n.to_integer()).ok()?;
            }
            ExprNode::Exp(inner) if !seen_exp => {
                let (rate, offset) = linear_real_constants(arena, inner, var)?;
                seen_exp = true;
                a = rate;
                if !arena.is_zero_structural(offset) {
                    coeff.push(arena.exp(offset));
                }
            }
            // b^(k·x + m) = e^{(k·x + m)·ln b} for a constant b > 0 (`2ˣ`)
            ExprNode::Pow(base, exp)
                if !seen_exp
                    && !contains_sym_id(arena, base, var)
                    && is_real_constant(arena, base)
                    && crate::transforms::evalf::evalf_complex64(arena, base)
                        .is_ok_and(|z| z.re > 0.0) =>
            {
                let (k, m) = linear_real_constants(arena, exp, var)?;
                let ln_b = arena.ln(base);
                let rate = arena.mul(&[k, ln_b]);
                seen_exp = true;
                a = crate::transforms::eval::eval(arena, rate);
                if !arena.is_zero_structural(m) {
                    coeff.push(arena.pow(base, m));
                }
            }
            ExprNode::Sin(inner) | ExprNode::Cos(inner) if kind == TrigKind::None => {
                let (freq, offset) = linear_real_constants(arena, inner, var)?;
                if !arena.is_zero_structural(offset) {
                    return None;
                }
                // The sign of a nonzero certified digit; 0 refuses.
                let negative = crate::transforms::evalf::evalf_complex64(arena, freq)
                    .ok()
                    .filter(|z| z.re != 0.0)?
                    .re
                    < 0.0;
                let is_sin = matches!(arena.node(f), ExprNode::Sin(_));
                kind = if is_sin { TrigKind::Sin } else { TrigKind::Cos };
                b = if negative { arena.neg(freq) } else { freq };
                b = crate::transforms::eval::eval(arena, b);
                if negative && is_sin {
                    coeff.push(arena.neg_one);
                }
            }
            _ => return None,
        }
    }
    let coeff = arena.mul(&coeff);
    Some(SymForcingTerm {
        coeff: crate::transforms::eval::eval(arena, coeff),
        degree,
        a: crate::transforms::eval::eval(arena, a),
        b,
        kind,
    })
}

/// Does `e` contain the expression `var` (a symbol)?
pub(super) fn contains_sym_id(arena: &Arena, e: ExprId, var: ExprId) -> bool {
    match arena.node(var) {
        ExprNode::Symbol(s) => contains_sym(arena, e, *s),
        _ => true,
    }
}

/// A complex number `re + i·im` with real constant parts, for
/// [`particular_for_group_sym`] (the symbolic counterpart of [`CQ`]).
#[derive(Clone, Copy)]
struct CEx {
    re: ExprId,
    im: ExprId,
}

impl CEx {
    fn norm(arena: &mut Arena, re: ExprId, im: ExprId) -> CEx {
        let mut parts = [re, im];
        for p in &mut parts {
            let e = crate::transforms::expand::expand(arena, *p);
            *p = crate::transforms::eval::eval(arena, e);
        }
        CEx {
            re: parts[0],
            im: parts[1],
        }
    }
    fn add(self, arena: &mut Arena, o: CEx) -> CEx {
        let re = arena.add(&[self.re, o.re]);
        let im = arena.add(&[self.im, o.im]);
        CEx::norm(arena, re, im)
    }
    fn sub(self, arena: &mut Arena, o: CEx) -> CEx {
        let re = arena.sub(self.re, o.re);
        let im = arena.sub(self.im, o.im);
        CEx::norm(arena, re, im)
    }
    fn mul(self, arena: &mut Arena, o: CEx) -> CEx {
        let rr = arena.mul(&[self.re, o.re]);
        let ii = arena.mul(&[self.im, o.im]);
        let ri = arena.mul(&[self.re, o.im]);
        let ir = arena.mul(&[self.im, o.re]);
        let re = arena.sub(rr, ii);
        let im = arena.add(&[ri, ir]);
        CEx::norm(arena, re, im)
    }
    fn scale(self, arena: &mut Arena, r: &Q) -> CEx {
        let r = ode_ratio_to_expr(arena, r);
        let re = arena.mul(&[r, self.re]);
        let im = arena.mul(&[r, self.im]);
        CEx::norm(arena, re, im)
    }
    fn div(self, arena: &mut Arena, o: CEx) -> CEx {
        // (x + iy)/(u + iv) = ((xu + yv) + i(yu − xv))/(u² + v²)
        let two = arena.int(2);
        let u2 = arena.pow(o.re, two);
        let v2 = arena.pow(o.im, two);
        let d = arena.add(&[u2, v2]);
        let d = crate::transforms::expand::expand(arena, d);
        let d = crate::transforms::eval::eval(arena, d);
        let xu = arena.mul(&[self.re, o.re]);
        let yv = arena.mul(&[self.im, o.im]);
        let yu = arena.mul(&[self.im, o.re]);
        let xv = arena.mul(&[self.re, o.im]);
        let re = arena.add(&[xu, yv]);
        let im = arena.sub(yu, xv);
        let mut parts = [re, im];
        for p in &mut parts {
            let e = crate::transforms::expand::expand(arena, *p);
            let e = crate::transforms::eval::eval(arena, e);
            let q = arena.div(e, d);
            *p = crate::simplify::radsimp::rationalize_denom(arena, q);
        }
        CEx::norm(arena, parts[0], parts[1])
    }
    /// `True` for an exact zero, `False` for a value certainly away from
    /// zero, `Unknown` when undecided (not structurally zero but
    /// numerically tiny).
    fn is_zero(self, arena: &Arena) -> Truth {
        if arena.is_zero_structural(self.re) && arena.is_zero_structural(self.im) {
            return Truth::True;
        }
        let (Ok(re), Ok(im)) = (
            crate::transforms::evalf::evalf_complex64(arena, self.re),
            crate::transforms::evalf::evalf_complex64(arena, self.im),
        ) else {
            return Truth::Unknown;
        };
        if re.norm().max(im.norm()) > 1e-8 {
            Truth::False
        } else {
            Truth::Unknown
        }
    }
}

/// [`particular_for_group`] for real constant `a`, `b` and polynomial
/// coefficients `q` that are not all rational: the same exponential shift
/// with `λ = a + b·i` in exact arithmetic on real constants ([`CEx`]); the
/// vanishing of `p⁽ʲ⁾(λ)` (resonance: `e^{√2·x}` against the root `√2` of
/// `r⁴ − 2`) must be decided exactly, or the term is refused.
fn particular_for_group_sym(
    arena: &mut Arena,
    p: &[Q],
    q: &[ExprId],
    a: ExprId,
    b: ExprId,
    kind: TrigKind,
    var: ExprId,
) -> Option<ExprId> {
    let lambda = CEx { re: a, im: b };
    let zero = CEx {
        re: arena.zero,
        im: arena.zero,
    };
    let n = p.len() - 1;
    // c_j = Σ_{k≥j} C(k, j)·a_k·λ^{k−j}
    let mut c = Vec::with_capacity(n + 1);
    for j in 0..=n {
        let mut acc = zero;
        let mut lambda_pow = CEx {
            re: arena.one,
            im: arena.zero,
        };
        for (k, a_k) in p.iter().enumerate().skip(j) {
            let coef = a_k * Q::from_integer(binomial(k as u64, j as u64));
            let term = lambda_pow.scale(arena, &coef);
            acc = acc.add(arena, term);
            lambda_pow = lambda_pow.mul(arena, lambda);
        }
        c.push(acc);
    }
    let mut s = None;
    for (j, cj) in c.iter().enumerate() {
        match cj.is_zero(arena) {
            Truth::True => {}
            Truth::False => {
                s = Some(j);
                break;
            }
            Truth::Unknown => return None,
        }
    }
    let s = s?;
    let d = q.len().checked_sub(1)?;
    let m = d + s;
    let mut w = vec![zero; m + 1];
    for k in (0..=d).rev() {
        let mut rhs = CEx {
            re: q[k],
            im: arena.zero,
        };
        for j in (s + 1)..=n {
            if k + j > m {
                continue;
            }
            let t = c[j]
                .mul(arena, w[k + j])
                .scale(arena, &falling_factorial_rat(k + j, j));
            rhs = rhs.sub(arena, t);
        }
        let lead = c[s].scale(arena, &falling_factorial_rat(k + s, s));
        w[k + s] = rhs.div(arena, lead);
    }
    let poly = |arena: &mut Arena, parts: &[ExprId]| -> ExprId {
        let mut terms = Vec::with_capacity(parts.len());
        for (k, &ck) in parts.iter().enumerate() {
            let e = arena.int(k as i64);
            let xk = arena.pow(var, e);
            terms.push(arena.mul(&[ck, xk]));
        }
        let sum = arena.add(&terms);
        crate::transforms::eval::eval(arena, sum)
    };
    // each coefficient as one reduced fraction (`(ln²2 + 3)/(ln²2 + 3)²` came
    // out unreduced for the rate `ln 2` of `2ˣ`)
    let reduce = |arena: &mut Arena, e: ExprId| -> ExprId {
        if arena.as_num(e).is_some() {
            return e;
        }
        let r = crate::simplify::ratsimp::ratsimp(arena, e);
        let r = crate::transforms::eval::eval(arena, r);
        let size = |arena: &Arena, x: ExprId| {
            crate::transforms::pattern::tree_size_capped(arena, x, 10_000)
        };
        if size(arena, r) <= size(arena, e) {
            r
        } else {
            e
        }
    };
    let wr: Vec<ExprId> = w.iter().map(|z| reduce(arena, z.re)).collect();
    let wi: Vec<ExprId> = w.iter().map(|z| reduce(arena, z.im)).collect();
    let wr_x = poly(arena, &wr);
    let wi_x = poly(arena, &wi);
    let ax = arena.mul(&[a, var]);
    let exp_ax = arena.exp(ax);
    let body = match kind {
        TrigKind::None => wr_x,
        TrigKind::Cos | TrigKind::Sin => {
            let bx = arena.mul(&[b, var]);
            let cos_bx = arena.cos(bx);
            let sin_bx = arena.sin(bx);
            if kind == TrigKind::Cos {
                let t1 = arena.mul(&[wr_x, cos_bx]);
                let t2 = arena.mul(&[wi_x, sin_bx]);
                arena.sub(t1, t2)
            } else {
                let t1 = arena.mul(&[wr_x, sin_bx]);
                let t2 = arena.mul(&[wi_x, cos_bx]);
                arena.add(&[t1, t2])
            }
        }
    };
    let y_p = arena.mul(&[exp_ax, body]);
    let y_p = crate::transforms::expand::expand(arena, y_p);
    Some(crate::transforms::eval::eval(arena, y_p))
}

/// `sinh u → (eᵘ − e⁻ᵘ)/2`, `cosh u → (eᵘ + e⁻ᵘ)/2`, then expanded:
/// the forcing terms of undetermined coefficients (`y‴ − 2y = sinh x` was
/// refused, its forcing not being of the form `xᵈ·e^{ax}·cos/sin`).
/// `None` when `terms` have no hyperbolic function.
fn hyperbolic_forcing_as_exp(arena: &mut Arena, terms: &[ExprId]) -> Option<Vec<ExprId>> {
    let has_hyperbolic = |arena: &Arena, e: ExprId| {
        crate::base::walk::post_order_ids(arena, e)
            .iter()
            .any(|&id| matches!(arena.node(id), ExprNode::Sinh(_) | ExprNode::Cosh(_)))
    };
    if !terms.iter().any(|&t| has_hyperbolic(arena, t)) {
        return None;
    }
    let sum = arena.add(terms);
    let post = crate::base::walk::post_order_ids(arena, sum);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Sinh(u) | ExprNode::Cosh(u) => {
                let is_sinh = matches!(arena.node(rebuilt), ExprNode::Sinh(_));
                let pos = arena.exp(u);
                let neg_u = arena.neg(u);
                let neg = arena.exp(neg_u);
                let neg = if is_sinh { arena.neg(neg) } else { neg };
                let s = arena.add(&[pos, neg]);
                let half = ode_ratio_to_expr(arena, &Q::new(1.into(), 2.into()));
                arena.mul(&[half, s])
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    let out = cache.get(&sum).copied().unwrap_or(sum);
    let out = crate::transforms::expand::expand(arena, out);
    let out = crate::transforms::eval::eval(arena, out);
    Some(match arena.node(out) {
        ExprNode::Add(c) => c.to_vec(),
        _ if arena.is_zero_structural(out) => Vec::new(),
        _ => vec![out],
    })
}

/// Pairs `c·r·e^{kx} ± c·r·e^{−kx}` (rational `c`, `k > 0`, the same rest
/// `r`) of the sum `e` written back as `2c·r·cosh(kx)`, `2c·r·sinh(kx)`:
/// the particular solution for a hyperbolic forcing term, found after
/// [`hyperbolic_forcing_as_exp`] (`y″ = sinh x` gives `sinh x` again, not
/// `eˣ/2 − e⁻ˣ/2`).
fn exp_pairs_as_hyperbolic(arena: &mut Arena, e: ExprId, var: ExprId) -> ExprId {
    let terms: Vec<ExprId> = match arena.node(e) {
        ExprNode::Add(c) => c.to_vec(),
        _ => return e,
    };
    // (rate k, coefficient c, rest r) of `c·r·e^{kx}`
    let mut parsed: Vec<Option<(Q, Q, ExprId)>> = Vec::with_capacity(terms.len());
    for &t in &terms {
        let (c, body) = arena.as_coeff_term(t);
        let factors: Vec<ExprId> = match arena.node(body) {
            ExprNode::Mul(fs) => fs.to_vec(),
            _ => vec![body],
        };
        let mut rate = None;
        let mut rest = Vec::with_capacity(factors.len());
        for f in factors {
            match *arena.node(f) {
                ExprNode::Exp(arg) if rate.is_none() => {
                    match extract_linear_numeric(arena, arg, var) {
                        Some((k, c0))
                            if num_traits::Zero::is_zero(&c0) && !num_traits::Zero::is_zero(&k) =>
                        {
                            rate = Some(k)
                        }
                        _ => rest.push(f),
                    }
                }
                _ => rest.push(f),
            }
        }
        parsed.push(rate.map(|k| (k, c, arena.mul(&rest))));
    }
    let mut used = vec![false; terms.len()];
    let mut out = Vec::with_capacity(terms.len());
    for i in 0..terms.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let Some((k, c, rest)) = parsed[i].clone() else {
            out.push(terms[i]);
            continue;
        };
        let partner = (0..terms.len()).find(|&j| {
            !used[j]
                && parsed[j].as_ref().is_some_and(|(k2, c2, r2)| {
                    *k2 == -k.clone() && *r2 == rest && (*c2 == c || *c2 == -c.clone())
                })
        });
        let Some(j) = partner else {
            out.push(terms[i]);
            continue;
        };
        used[j] = true;
        let Some((_, c2, _)) = parsed[j].clone() else {
            continue;
        };
        // c·e^{kx} + c2·e^{−kx} with |k| written as the positive rate
        let (pos_c, neg_c) = if k.is_positive() { (c, c2) } else { (c2, c) };
        let rate = ode_ratio_to_expr(arena, &k.abs());
        let arg = arena.mul(&[rate, var]);
        let (func, scale) = if pos_c == neg_c {
            (arena.cosh(arg), pos_c)
        } else {
            (arena.sinh(arg), pos_c)
        };
        let two_c = ode_ratio_to_expr(arena, &(Q::from_integer(2.into()) * scale));
        out.push(arena.mul(&[two_c, rest, func]));
    }
    let sum = arena.add(&out);
    crate::transforms::eval::eval(arena, sum)
}

/// `n·(n-1)·…·(n-j+1)` as a rational.
pub(super) fn falling_factorial_rat(n: usize, j: usize) -> Q {
    let mut r = Q::from_integer(1.into());
    for i in 0..j {
        r *= Q::from_integer(((n - i) as i64).into());
    }
    r
}

/// Fundamental solutions of `Σ a_k y^(k) = 0` from the square-free
/// factorisation of the characteristic polynomial: `x^j e^{rx}` for a
/// real root of multiplicity `> j`, and `x^j e^{αx} cos(βx)`,
/// `x^j e^{αx} sin(βx)` for a complex pair `α ± βi` (`β > 0`): a real
/// fundamental system, as SymPy's `nth_linear_constant_coeff_homogeneous`.
/// The roots are those of [`characteristic_roots`]: radicals where the
/// solver gives them with real and imaginary parts in real radicals
/// (Cardano's for `r³ + r² + 2`), `RootOf` otherwise.
pub(super) fn homogeneous_basis_cc(
    arena: &mut Arena,
    coeffs: &[Q],
    var: ExprId,
) -> Option<Vec<ExprId>> {
    let p = crate::poly::Poly::from_coeffs(coeffs.to_vec());
    let r_sym = arena.symbol("__r_cc");
    let mut basis: Vec<ExprId> = Vec::new();
    for (factor, mult) in p.squarefree_factors() {
        for root in characteristic_roots(arena, &factor, r_sym, var)? {
            for j in 0..mult {
                match root {
                    CharRoot::Real(r) => basis.push(mode_real(arena, r, j, var)),
                    CharRoot::Pair { re, im } => {
                        let (c, s) = mode_complex(arena, re, im, j, var);
                        basis.push(c);
                        basis.push(s);
                    }
                }
            }
        }
    }
    Some(basis)
}

/// A root of a real characteristic polynomial as [`homogeneous_basis_cc`]
/// uses it: a real root, or a non-real pair `re ± i·im` with `im > 0`.
#[derive(Clone, Copy, Debug)]
enum CharRoot {
    /// A real root (an expression free of `i`).
    Real(ExprId),
    /// The pair `re ± i·im`, both parts real expressions, `im > 0`.
    Pair { re: ExprId, im: ExprId },
}

/// The distinct roots of the square-free rational polynomial `f` (one
/// entry per real root or conjugate pair).  First the solver's roots of
/// `f` in radicals ([`radical_char_roots`]); failing that, each factor
/// irreducible over ℚ in radicals or, when the radicals have no real form
/// (degree ≥ 5, a casus-irreducibilis cubic, Ferrari's complex cube
/// roots), as `RootOf` ([`rootof_char_roots`]).  Before, every root
/// containing `i` was taken as non-real and paired by structural equality
/// of the parts: an irreducible cubic (`r³ + r² + 2`, `r³ − 3r + 1`) gave
/// no basis (the ODE was refused), and `RootOf` roots of a quintic were
/// used as complex exponentials `e^{x·RootOf(…)}`.
fn characteristic_roots(
    arena: &mut Arena,
    f: &crate::poly::Poly,
    r_sym: ExprId,
    var: ExprId,
) -> Option<Vec<CharRoot>> {
    if let Some(roots) = radical_char_roots(arena, f, r_sym) {
        return Some(roots);
    }
    let (_, irreducible) = f.factor_over_z();
    let mut out = Vec::new();
    for (g, _) in &irreducible {
        // A single factor is `f` itself, whose radicals just failed.
        let radical = if irreducible.len() > 1 {
            radical_char_roots(arena, g, r_sym)
        } else {
            None
        };
        let roots = match radical {
            Some(r) => r,
            None => rootof_char_roots(arena, g, var)?,
        };
        out.extend(roots);
    }
    Some(out)
}

/// Relative size of the imaginary part below which a root value is taken
/// as real by [`radical_char_roots`] (the exact count of real roots decides
/// how many are; this only orders them), and above [`IM_NONREAL`] non-real.
const IM_REAL: f64 = 1e-9;
/// See [`IM_REAL`].
const IM_NONREAL: f64 = 1e-6;

/// The roots of the square-free rational polynomial `f` from the solver,
/// in radicals: `None` unless there are `deg f` of them, none a `RootOf`,
/// each real root has a real form (free of `i`) and each non-real pair has
/// real and imaginary parts free of `i` and of unevaluated `re`/`im`.  How
/// many roots are real is the exact count of
/// [`SturmChain`](crate::poly::sturm::SturmChain); which ones, their
/// values (the solver's spelling says nothing: a casus-irreducibilis root
/// is real), and every part is checked against the value.
fn radical_char_roots(
    arena: &mut Arena,
    f: &crate::poly::Poly,
    r_sym: ExprId,
) -> Option<Vec<CharRoot>> {
    let deg = f.degree()?;
    let f_expr = crate::poly::polybridge::poly_to_expr(arena, f, r_sym);
    let sols = match crate::transforms::solve::solve_classified_real(arena, f_expr, r_sym)
        .points_if_complete()
    {
        Some(sols) if sols.len() == deg => sols,
        _ => return None,
    };
    let mut values = Vec::with_capacity(deg);
    for &s in &sols {
        if crate::base::walk::contains(arena, s, r_sym) || has_rootof(arena, s) {
            return None;
        }
        let z = crate::transforms::evalf::evalf_complex64(arena, s).ok()?;
        if !(z.re.is_finite() && z.im.is_finite()) {
            return None;
        }
        values.push(z);
    }
    let n_real = crate::poly::sturm::SturmChain::new(f).count_real_roots();
    let rel_im = |z: &num_complex::Complex64| z.im.abs() / z.norm().max(1.0);
    let mut order: Vec<usize> = (0..deg).collect();
    order.sort_by(|&a, &b| rel_im(&values[a]).total_cmp(&rel_im(&values[b])));
    if (n_real > 0 && rel_im(&values[order[n_real - 1]]) > IM_REAL)
        || (n_real < deg && rel_im(&values[order[n_real]]) < IM_NONREAL)
    {
        return None;
    }
    let mut is_real = vec![false; deg];
    for &k in &order[..n_real] {
        is_real[k] = true;
    }
    // Checks of the exact parts against the values: they can only refuse.
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * (1.0 + b.abs());
    let i_unit = arena.i_unit;
    let mut used = vec![false; deg];
    let mut out = Vec::new();
    for i in 0..deg {
        if used[i] {
            continue;
        }
        used[i] = true;
        if is_real[i] {
            let mut r = sols[i];
            if crate::base::walk::contains(arena, r, i_unit) {
                r = real_imag_parts(arena, r).0;
                if !real_radical_form(arena, r) {
                    return None;
                }
                let z = crate::transforms::evalf::evalf_complex64(arena, r).ok()?;
                if !close(z.re, values[i].re) || z.im.abs() > IM_REAL * (1.0 + z.re.abs()) {
                    return None;
                }
            }
            out.push(CharRoot::Real(r));
            continue;
        }
        // The conjugate partner: the unused non-real root nearest conj(zᵢ).
        let conj = values[i].conj();
        let j = (0..deg)
            .filter(|&k| !used[k] && !is_real[k])
            .min_by(|&a, &b| {
                (values[a] - conj)
                    .norm()
                    .total_cmp(&(values[b] - conj).norm())
            })?;
        if (values[j] - conj).norm() > 1e-9 * (1.0 + conj.norm()) {
            return None;
        }
        used[j] = true;
        let pos = if values[i].im > 0.0 { i } else { j };
        let (re, im) = real_imag_parts(arena, sols[pos]);
        if !real_radical_form(arena, re) || !real_radical_form(arena, im) {
            return None;
        }
        let zr = crate::transforms::evalf::evalf_complex64(arena, re).ok()?;
        let zi = crate::transforms::evalf::evalf_complex64(arena, im).ok()?;
        if !close(zr.re, values[pos].re) || !close(zi.re, values[pos].im) {
            return None;
        }
        out.push(CharRoot::Pair { re, im });
    }
    Some(out)
}

/// `(re, im)` of the constant `z` with real and imaginary parts of real
/// radicals resolved and the signs of `cos`/`sin` arguments normalised:
/// [`Arena::as_real_imag_expr`] leaves `re(cbrt(28/27 − √87/9))` when it
/// cannot tell the radicand is positive (Cardano's complex roots of
/// `r³ + r² + 2`), and gives `cos(−2π/9) + cos(2π/9)` for a
/// casus-irreducibilis root.  `re(u) → u`, `im(u) → 0` when `u` is real
/// by [`is_real_radical`]; `cos(−a) → cos(a)`, `sin(−a) → −sin(a)` for a
/// negative rational multiple `−a` of `π`.
pub(super) fn real_imag_parts(arena: &mut Arena, z: ExprId) -> (ExprId, ExprId) {
    // √u of a negative real radical u is i·√(−u) (principal branch):
    // `as_real_imag` leaves `re(√(√5/2 − 3/2))`, so the biquadratic
    // `r⁴ + 3r² + 1` fell back to `RootOf`.
    let z = {
        let post = crate::base::walk::post_order_ids(arena, z);
        let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
        for &id in &post {
            let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
            let new = match *arena.node(rebuilt) {
                // Certified negative beyond the margin; nearer 0 the part
                // stays unresolved and the caller refuses.
                ExprNode::Pow(b, e)
                    if arena
                        .as_num(e)
                        .is_some_and(|q| *q == Q::new(1.into(), 2.into()))
                        && is_real_radical(arena, b)
                        && crate::transforms::evalf::evalf_complex64(arena, b)
                            .is_ok_and(|v| v.re < -1e-12) =>
                {
                    let neg_b = arena.neg(b);
                    let neg_b = crate::transforms::eval::eval(arena, neg_b);
                    let root = arena.pow(neg_b, e);
                    let i_unit = arena.i_unit;
                    arena.mul(&[i_unit, root])
                }
                _ => rebuilt,
            };
            cache.insert(id, new);
        }
        let z = cache.get(&z).copied().unwrap_or(z);
        crate::transforms::eval::eval(arena, z)
    };
    let (re, im) = arena.as_real_imag_expr(z);
    let mut parts = [re, im];
    for part in &mut parts {
        let post = crate::base::walk::post_order_ids(arena, *part);
        let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
        for &id in &post {
            let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
            let new = match arena.node(rebuilt).clone() {
                ExprNode::Re(u) if is_real_radical(arena, u) => u,
                ExprNode::Im(u) if is_real_radical(arena, u) => arena.zero,
                ExprNode::Cos(a) | ExprNode::Sin(a) => {
                    let (c, t) = arena.as_coeff_term(a);
                    if c.is_negative() && t == arena.pi {
                        let pos = ode_ratio_to_expr(arena, &(-c));
                        let pos = arena.mul(&[pos, t]);
                        if matches!(arena.node(rebuilt), ExprNode::Cos(_)) {
                            arena.cos(pos)
                        } else {
                            let s = arena.sin(pos);
                            arena.neg(s)
                        }
                    } else {
                        rebuilt
                    }
                }
                _ => rebuilt,
            };
            cache.insert(id, new);
        }
        let out = cache.get(part).copied().unwrap_or(*part);
        *part = crate::transforms::eval::eval(arena, out);
    }
    (parts[0], parts[1])
}

/// Is the constant `u` real by construction: free of `i`, with every
/// fractional power and logarithm taken of a positive value (decided
/// numerically, with a margin), and real at its value?
fn is_real_radical(arena: &Arena, u: ExprId) -> bool {
    if !real_radical_form(arena, u) || !crate::base::walk::free_symbols(arena, u).is_empty() {
        return false;
    }
    // The constants are radicals of rationals from the solver: realness
    // follows from every base being positive (a certified value beyond the
    // margin), innermost first; the `f64` imaginary parts only confirm it.
    let positive = |b: ExprId| {
        crate::transforms::evalf::evalf_complex64(arena, b)
            .is_ok_and(|z| z.im == 0.0 && z.re > 1e-12 * (1.0 + z.re.abs()))
    };
    let mut stack = vec![u];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let node = arena.node(n);
        match node {
            ExprNode::Pow(b, e) if !arena.as_num(*e).is_some_and(Q::is_integer) => {
                if !positive(*b) {
                    return false;
                }
            }
            ExprNode::Ln(b) if !positive(*b) => return false,
            _ => {}
        }
        node.for_each_child(|c| stack.push(c));
    }
    crate::transforms::evalf::evalf_complex64(arena, u)
        .is_ok_and(|z| z.im == 0.0 && z.re.is_finite())
}

/// Is `e` free of `i` and of the nodes `re`, `im`, `arg`, `conjugate`
/// (a real part that [`Arena::as_real_imag_expr`] could not take)?
pub(super) fn real_radical_form(arena: &Arena, e: ExprId) -> bool {
    let i_unit = arena.i_unit;
    let mut stack = vec![e];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if n == i_unit {
            return false;
        }
        let node = arena.node(n);
        if matches!(
            node,
            ExprNode::Re(_) | ExprNode::Im(_) | ExprNode::Arg(_) | ExprNode::Conjugate(_)
        ) {
            return false;
        }
        node.for_each_child(|c| stack.push(c));
    }
    true
}

/// `(re h(z), im h(z))` for the rational polynomial `h` at `z = re + i·im`
/// with real `re`, `im`: Horner's rule on the pair of parts.
fn horner_complex(
    arena: &mut Arena,
    h: &crate::poly::Poly,
    re: ExprId,
    im: ExprId,
) -> (ExprId, ExprId) {
    let (mut a, mut b) = (arena.zero, arena.zero);
    for c in h.coeffs().iter().rev() {
        // (a + i·b)·(re + i·im) + c
        let a_re = arena.mul(&[a, re]);
        let b_im = arena.mul(&[b, im]);
        let neg_b_im = arena.neg(b_im);
        let c = ode_ratio_to_expr(arena, c);
        let a_new = arena.add(&[a_re, neg_b_im, c]);
        let a_im = arena.mul(&[a, im]);
        let b_re = arena.mul(&[b, re]);
        let b_new = arena.add(&[a_im, b_re]);
        a = crate::transforms::eval::eval(arena, a_new);
        b = crate::transforms::eval::eval(arena, b_new);
    }
    (a, b)
}

/// Does `e` contain a `RootOf`?
pub(super) fn has_rootof(arena: &Arena, e: ExprId) -> bool {
    let mut stack = vec![e];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let node = arena.node(n);
        if matches!(node, ExprNode::RootOf(..)) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// Precision (bits) of the root balls [`rootof_char_roots`] certifies.
const ROOTOF_PREC: usize = 128;

/// The roots of the rational polynomial `g` (square-free, normally
/// irreducible) as `RootOf(g, k)`, SymPy's `CRootOf` form: a real root
/// `RootOf(g, k)`, a non-real pair `re(RootOf(g, k)) ± i·im(RootOf(g, k))`
/// for the index `k` of its member with positive imaginary part.  Which
/// roots are real, and the order the indices refer to, are those of the
/// certified root balls `RootOf` is evaluated with; `None` when a ball is
/// not certified.  The bound variable is named apart from `var`.
fn rootof_char_roots(
    arena: &mut Arena,
    g: &crate::poly::Poly,
    var: ExprId,
) -> Option<Vec<CharRoot>> {
    let deg = g.degree()?;
    let balls = crate::poly::roots::certified_roots(g, ROOTOF_PREC);
    if balls.len() != deg
        || balls.iter().any(|b| b.radius.is_none())
        || balls.iter().filter(|b| b.real).count()
            != crate::poly::sturm::SturmChain::new(g).count_real_roots()
    {
        return None;
    }
    let var_name = match arena.node(var) {
        ExprNode::Symbol(s) => arena.symbol_name(*s).to_string(),
        _ => String::new(),
    };
    let name = ["r", "s", "u", "w"]
        .into_iter()
        .find(|n| *n != var_name)
        .unwrap_or("r");
    let bound = arena.symbol(name);
    let g_expr = crate::poly::polybridge::poly_to_expr(arena, g, bound);
    let mut out = Vec::new();
    let mut pairs = 0usize;
    for (k, ball) in balls.iter().enumerate() {
        let idx = arena.int(i64::try_from(k).ok()?);
        let root = arena.intern(ExprNode::RootOf(g_expr, bound, idx));
        if ball.real {
            out.push(CharRoot::Real(root));
        } else if crate::base::bigfloat::is_sign_positive(&ball.value.1) {
            let re = arena.re(root);
            let im = arena.im(root);
            out.push(CharRoot::Pair { re, im });
            pairs += 1;
        }
    }
    (out.len() + pairs == deg).then_some(out)
}

/// Is `expr = 0` linear in `func` with coefficients free of `var` and
/// `func` (`Σ aₖ·y⁽ᵏ⁾ + g(x)`, each term `aₖ·y⁽ᵏ⁾` or free of `func`)?
/// Its homogeneous solutions are then translation invariant.
pub(crate) fn is_linear_autonomous(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
) -> bool {
    let ExprNode::Symbol(func_sym) = *arena.node(func) else {
        return false;
    };
    let chain = derivative_chain(arena, func, var);
    let children: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![expr],
    };
    let mut has_func = false;
    for child in children {
        let (_, term) = arena.as_coeff_term(child);
        if symbolic_linear_term(arena, term, &chain, var, func_sym).is_some() {
            has_func = true;
        } else if contains_sym(arena, child, func_sym) {
            return false;
        }
    }
    has_func
}

/// Is `expr = 0` a Cauchy–Euler equation `Σ aₖ·xᵏ·y⁽ᵏ⁾ + g(x) = 0`
/// (`aₖ` free of `var` and `func`)?  Its homogeneous solutions are then
/// invariant under `x → λ·x`.
pub(crate) fn is_cauchy_euler(arena: &mut Arena, expr: ExprId, func: ExprId, var: ExprId) -> bool {
    let ExprNode::Symbol(func_sym) = *arena.node(func) else {
        return false;
    };
    let ExprNode::Symbol(var_sym) = *arena.node(var) else {
        return false;
    };
    let chain = derivative_chain(arena, func, var);
    let children: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![expr],
    };
    let mut has_func = false;
    for child in children {
        if !contains_sym(arena, child, func_sym) {
            continue;
        }
        let factors: Vec<ExprId> = match arena.node(child) {
            ExprNode::Mul(fs) => fs.to_vec(),
            ExprNode::Neg(inner) => vec![arena.neg_one, *inner],
            _ => vec![child],
        };
        let mut order = None;
        let mut rest = Vec::with_capacity(factors.len());
        for f in factors {
            match chain.iter().position(|&d| d == f) {
                Some(k) if order.is_none() => order = Some(k),
                Some(_) => return false,
                None if contains_sym(arena, f, func_sym) => return false,
                None => rest.push(f),
            }
        }
        let Some(k) = order else {
            return false;
        };
        // rest / x^k free of x
        let r = arena.mul(&rest);
        let xk = {
            let e = arena.int(i64::try_from(k).unwrap_or(0));
            arena.pow(var, e)
        };
        let q = arena.div(r, xk);
        let q = crate::transforms::eval::eval(arena, q);
        if contains_sym(arena, q, var_sym) {
            return false;
        }
        has_func = true;
    }
    has_func
}

/// The solution of `Σ aₖ·y⁽ᵏ⁾ = g(x)` (rational `aₖ`, order `n ≥ 2`) with
/// `y⁽ᵏ⁾(x₀) = values[k]` for `k < n`, when the characteristic polynomial
/// `p` is square-free and has an irreducible factor of degree `≥ 3`
/// (roots in Cardano's radicals or `RootOf`, see [`characteristic_roots`]);
/// `None` otherwise (the generic fit applies).
///
/// With `y = yₚ + Σ cᵣ·e^{r·(x − x₀)}` the conditions are the Vandermonde
/// system `Σ cᵣ·rᵏ = wₖ = values[k] − yₚ⁽ᵏ⁾(x₀)`, whose inverse is given by
/// the Lagrange polynomials: `cᵣ = Σₖ bₖ(r)·wₖ / p′(r)` with
/// `p(t) = (t − r)·Σ bₖ(r)·tᵏ`.  Reduced modulo the irreducible factor
/// `g` of `r`, `bₖ/p′ = hₖ mod g` has rational coefficients, so a
/// non-real pair contributes `e^{α(x−x₀)}·(A·cos β(x−x₀) + B·sin β(x−x₀))`
/// with `A = 2·Σ wₖ·re hₖ(r)`, `B = −2·Σ wₖ·im hₖ(r)` (`hₖ(re + i·im)` by
/// Horner's rule on the pair of parts).  The generic fit (symbolic
/// elimination) did not finish in a minute for `y⁽⁵⁾ − y′ + y = 0` (over
/// `re`/`im` of `RootOf`s) nor for `y‴ + y″ + 2y = 0` with `y(1/3) = 2/3`,
/// `y′(1/3) = 5/4`, `y″(1/3) = −3/7` (over Cardano's radicals).
pub(crate) fn linear_cc_ivp_lagrange(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    x0: ExprId,
    values: &[ExprId],
) -> Option<ExprId> {
    use crate::poly::Poly;
    let ExprNode::Symbol(func_sym) = *arena.node(func) else {
        return None;
    };
    let Some(ode) = extract_linear_cc(arena, expr, func, var, func_sym) else {
        return euler_ivp_lagrange(arena, expr, func, var, x0, values);
    };
    let n = ode.coeffs.len() - 1;
    if n < 2 || values.len() != n {
        return None;
    }
    let p = Poly::from_coeffs(ode.coeffs.clone()).make_monic();
    if p.squarefree_factors().iter().any(|(_, m)| *m != 1) {
        return None;
    }
    let r_sym = arena.symbol("__r_cc");
    let (_, irreducible) = p.factor_over_z();
    if irreducible.iter().all(|(g, _)| g.degree().unwrap_or(0) < 3) {
        return None;
    }
    let mut groups: Vec<(Poly, Vec<CharRoot>)> = Vec::with_capacity(irreducible.len());
    for (g, _) in irreducible {
        let roots = match radical_char_roots(arena, &g, r_sym) {
            Some(r) => r,
            None => rootof_char_roots(arena, &g, var)?,
        };
        groups.push((g, roots));
    }
    // The particular solution and the conditions left for the modes.
    let general = try_nth_order_linear_const_coeff(arena, expr, func, var, func_sym)?;
    let mut y_p = general.solution;
    let zero = arena.zero;
    for &c in &general.constants {
        y_p = crate::transforms::subs::subs(arena, y_p, c, zero);
    }
    let y_p = crate::transforms::eval::eval(arena, y_p);
    let mut w = Vec::with_capacity(n);
    let mut d = y_p;
    for &v in values {
        let at = crate::transforms::subs::subs(arena, d, var, x0);
        let wk = arena.sub(v, at);
        w.push(crate::transforms::eval::eval(arena, wk));
        d = crate::transforms::diff::diff(arena, d, var);
    }
    // b_{n−1} = 1, b_{k−1} = a_k + t·b_k (p monic).
    let t_poly = Poly::from_coeffs(vec![Q::from_integer(0.into()), Q::from_integer(1.into())]);
    let mut b = vec![Poly::from_coeffs(vec![Q::from_integer(1.into())]); n];
    for k in (1..n).rev() {
        let ak = Poly::from_coeffs(vec![p.coeff(k)]);
        b[k - 1] = ak.add(&t_poly.mul(&b[k]));
    }
    let dp = p.derivative();
    let t_sym = arena.symbol("__t_cc");
    let shift = arena.sub(var, x0);
    let two = arena.int(2);
    let mut terms = vec![y_p];
    for (g, roots) in &groups {
        // x·p′ + y·g = gcd = 1 (monic): x = 1/p′ mod g.
        let e = Poly::extended_gcd(&dp.rem(g), g);
        if e.gcd.degree() != Some(0) {
            return None;
        }
        let inv_dp = e.x;
        let h: Vec<Poly> = b.iter().map(|bk| bk.mul(&inv_dp).rem(g)).collect();
        for root in roots {
            match *root {
                CharRoot::Real(r) => {
                    let mut c = Vec::with_capacity(n);
                    for (hk, wk) in h.iter().zip(&w) {
                        let hk = crate::poly::polybridge::poly_to_expr(arena, hk, t_sym);
                        let at = crate::transforms::subs::subs(arena, hk, t_sym, r);
                        c.push(arena.mul(&[*wk, at]));
                    }
                    let c = arena.add(&c);
                    let rx = arena.mul(&[r, shift]);
                    let e = arena.exp(rx);
                    terms.push(arena.mul(&[c, e]));
                }
                CharRoot::Pair { re, im } => {
                    let (mut a, mut bb) = (Vec::with_capacity(n), Vec::with_capacity(n));
                    for (hk, wk) in h.iter().zip(&w) {
                        let (h_re, h_im) = horner_complex(arena, hk, re, im);
                        a.push(arena.mul(&[*wk, h_re]));
                        bb.push(arena.mul(&[*wk, h_im]));
                    }
                    let a = arena.add(&a);
                    let a = arena.mul(&[two, a]);
                    let bsum = arena.add(&bb);
                    let neg_two = arena.int(-2);
                    let bcoef = arena.mul(&[neg_two, bsum]);
                    let ax = arena.mul(&[re, shift]);
                    let env = arena.exp(ax);
                    let bx = arena.mul(&[im, shift]);
                    let cos_bx = arena.cos(bx);
                    let sin_bx = arena.sin(bx);
                    let ca = arena.mul(&[a, cos_bx]);
                    let sb = arena.mul(&[bcoef, sin_bx]);
                    let osc = arena.add(&[ca, sb]);
                    terms.push(arena.mul(&[env, osc]));
                }
            }
        }
    }
    let sol = arena.add(&terms);
    Some(crate::transforms::eval::eval(arena, sol))
}

/// `x^j·e^{r·x}` (with `e^{0}` folded away).
fn mode_real(arena: &mut Arena, r: ExprId, j: usize, var: ExprId) -> ExprId {
    let x_pow = match j {
        0 => arena.one,
        1 => var,
        _ => {
            let e = arena.int(j as i64);
            arena.pow(var, e)
        }
    };
    if arena.is_zero_structural(r) {
        return x_pow;
    }
    let rx = arena.mul(&[r, var]);
    let exp_rx = arena.exp(rx);
    let m = arena.mul(&[x_pow, exp_rx]);
    crate::transforms::eval::eval(arena, m)
}

/// `(x^j e^{αx} cos βx, x^j e^{αx} sin βx)`.
fn mode_complex(
    arena: &mut Arena,
    alpha: ExprId,
    beta: ExprId,
    j: usize,
    var: ExprId,
) -> (ExprId, ExprId) {
    let envelope = mode_real(arena, alpha, j, var);
    let bx = arena.mul(&[beta, var]);
    let cos_bx = arena.cos(bx);
    let sin_bx = arena.sin(bx);
    let c = arena.mul(&[envelope, cos_bx]);
    let s = arena.mul(&[envelope, sin_bx]);
    (
        crate::transforms::eval::eval(arena, c),
        crate::transforms::eval::eval(arena, s),
    )
}

/// Solve `Σ a_k y^(k) + g(x) = 0` of any order `≥ 2` with rational
/// coefficients, where `g` is a sum of `poly·exp·{sin|cos}` terms (or
/// empty).  Homogeneous part via the characteristic polynomial (repeated
/// and complex roots handled), particular part via undetermined
/// coefficients with resonance handling.
pub(super) fn try_nth_order_linear_const_coeff(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Option<OdeResult> {
    use num_traits::Zero;
    let ode = extract_linear_cc(arena, expr, func, var, func_sym)?;
    let n = ode.coeffs.len() - 1;
    if n < 2 {
        return None;
    }
    tracing::debug!(order = n, "ode: nth-order linear constant-coefficient");

    // Forcing: L[y] = -g(x).  Parse and group by (a, b, kind); hyperbolic
    // functions as exponentials, terms with irrational constants apart.
    let hyperbolic = hyperbolic_forcing_as_exp(arena, &ode.forcing);
    let had_hyperbolic = hyperbolic.is_some();
    let mut forcing = hyperbolic.unwrap_or(ode.forcing);
    // A product of sums (`(x + ln 2)·eˣ`) is parsed term by term.
    if forcing.iter().any(|&t| {
        parse_forcing_term(arena, t, var).is_none()
            && parse_forcing_term_sym(arena, t, var).is_none()
    }) {
        let sum = arena.add(&forcing);
        let sum = crate::transforms::expand::expand(arena, sum);
        let sum = crate::transforms::eval::eval(arena, sum);
        forcing = match arena.node(sum) {
            ExprNode::Add(c) => c.to_vec(),
            _ => vec![sum],
        };
    }
    let mut groups: Vec<((Q, Q, TrigKind), Vec<Q>)> = Vec::new();
    let mut sym_groups: Vec<((ExprId, ExprId, TrigKind), Vec<ExprId>)> = Vec::new();
    for &term in &forcing {
        let Some(ft) = parse_forcing_term(arena, term, var) else {
            let ft = parse_forcing_term_sym(arena, term, var)?;
            let key = (ft.a, ft.b, ft.kind);
            let index = match sym_groups.iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    sym_groups.push((key, Vec::new()));
                    sym_groups.len() - 1
                }
            };
            let entry = &mut sym_groups[index].1;
            if entry.len() <= ft.degree {
                entry.resize(ft.degree + 1, arena.zero);
            }
            // move to the right-hand side
            let slot = arena.sub(entry[ft.degree], ft.coeff);
            entry[ft.degree] = crate::transforms::eval::eval(arena, slot);
            continue;
        };
        let key = (ft.a.clone(), ft.b.clone(), ft.kind);
        let entry = match groups.iter_mut().find(|(k, _)| *k == key) {
            Some(e) => e,
            None => {
                groups.push((key, Vec::new()));
                groups.last_mut()?
            }
        };
        if entry.1.len() <= ft.degree {
            entry.1.resize(ft.degree + 1, Q::zero());
        }
        entry.1[ft.degree] -= ft.coeff; // move to the right-hand side
    }

    let basis = homogeneous_basis_cc(arena, &ode.coeffs, var)?;
    if basis.len() != n {
        return None;
    }
    let mut constants = Vec::with_capacity(n);
    let mut terms = Vec::with_capacity(n + groups.len());
    for (i, &b) in basis.iter().enumerate() {
        let c = arena.symbol(&format!("C{}", i + 1));
        constants.push(c);
        terms.push(arena.mul(&[c, b]));
    }
    let mut particular = Vec::with_capacity(groups.len() + sym_groups.len());
    for ((a, b, kind), q) in &groups {
        if q.iter().all(Zero::is_zero) {
            continue;
        }
        let y_p = particular_for_group(arena, &ode.coeffs, q, a, b, *kind, var)?;
        particular.push(y_p);
    }
    for ((a, b, kind), q) in &sym_groups {
        if q.iter().all(|&c| arena.is_zero_structural(c)) {
            continue;
        }
        let y_p = particular_for_group_sym(arena, &ode.coeffs, q, *a, *b, *kind, var)?;
        particular.push(y_p);
    }
    if had_hyperbolic && !particular.is_empty() {
        let y_p = arena.add(&particular);
        let y_p = crate::transforms::expand::expand(arena, y_p);
        let y_p = crate::transforms::eval::eval(arena, y_p);
        particular = vec![exp_pairs_as_hyperbolic(arena, y_p, var)];
    }
    terms.extend(particular);
    let solution = arena.add(&terms);
    let solution = crate::transforms::eval::eval(arena, solution);
    Some(OdeResult {
        solution,
        constants,
        branches: Vec::new(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear ODEs with constant symbolic coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// `(k, c)` if `term` is `c·y⁽ᵏ⁾` with `c` free of `x` and `y`.
fn symbolic_linear_term(
    arena: &mut Arena,
    term: ExprId,
    chain: &[ExprId],
    var: ExprId,
    func_sym: SymbolId,
) -> Option<(usize, ExprId)> {
    if let Some(k) = chain.iter().position(|&d| d == term) {
        return Some((k, arena.one));
    }
    let ExprNode::Mul(factors) = arena.node(term).clone() else {
        return None;
    };
    let mut order = None;
    let mut rest = Vec::with_capacity(factors.len());
    for f in factors {
        if let Some(k) = chain.iter().position(|&d| d == f) {
            if order.replace(k).is_some() {
                return None;
            }
        } else if crate::base::walk::contains(arena, f, var) || contains_sym(arena, f, func_sym) {
            return None;
        } else {
            rest.push(f);
        }
    }
    Some((order?, arena.mul(&rest)))
}

/// `e` with every `Piecewise` whose first condition is a parameter
/// inequality (`a ≠ 0`, conjunctions of those; free of `var`) replaced by
/// its first, generic branch; `None` if another `Piecewise` remains.
pub(super) fn generic_branches(arena: &mut Arena, e: ExprId, var: ExprId) -> Option<ExprId> {
    fn is_generic(arena: &Arena, c: ExprId) -> bool {
        match arena.node(c) {
            ExprNode::Ne(..) => true,
            ExprNode::And(cs) => cs
                .iter()
                .all(|&k| matches!(arena.node(k), ExprNode::Ne(..))),
            _ => false,
        }
    }
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Piecewise(pairs) => {
                let (v, c) = *pairs.first()?;
                if crate::base::walk::contains(arena, c, var) || !is_generic(arena, c) {
                    return None;
                }
                v
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    let r = cache.get(&e).copied().unwrap_or(e);
    Some(crate::transforms::eval::eval(arena, r))
}

/// Is `e` identically zero after expansion (structurally)?
fn expands_to_zero(arena: &mut Arena, e: ExprId) -> bool {
    let e = crate::transforms::eval::eval(arena, e);
    let e = crate::transforms::expand::expand(arena, e);
    let e = crate::transforms::eval::eval(arena, e);
    arena.is_zero_structural(e)
}

/// Solve `Σ a_k·y⁽ᵏ⁾ = g(x)` whose constant coefficients are not all
/// rational (`y″ + a·y = 0`, `y″ + a·y′ + y = 0`): the roots of the
/// characteristic polynomial in radicals (principal branches), each checked
/// by substitution and given its multiplicity by the vanishing derivatives,
/// so the basis `xʲ·e^{r·x}`; SymPy's form, `C1·e^{−x√(−a)} +
/// C2·e^{x√(−a)}` for `y″ + a·y = 0`, with no case split on the sign of
/// `a` (the exponentials cover both signs: `e^{±x√(−a)}` are the real
/// exponentials for `a < 0` and `e^{±ix√a}` for `a > 0`).  The solution is
/// the generic one: it assumes the leading coefficient non-zero and roots
/// that are distinct as expressions distinct as values (at `a = 0` the
/// basis `e^{±x√(−a)}` degenerates, as in SymPy).  A second-order
/// equation with a forcing term gets its particular solution by variation
/// of parameters; higher orders with forcing are left to other methods.
pub(super) fn try_linear_cc_symbolic(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Option<OdeResult> {
    let chain = derivative_chain(arena, func, var);
    let children: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![expr],
    };
    let mut coeffs: Vec<Vec<ExprId>> = vec![Vec::new(); MAX_ODE_ORDER + 1];
    let mut forcing = Vec::new();
    for child in children {
        if let Some((k, c)) = symbolic_linear_term(arena, child, &chain, var, func_sym) {
            coeffs[k].push(c);
        } else if !contains_sym(arena, child, func_sym) {
            forcing.push(child);
        } else {
            return None;
        }
    }
    let mut a: Vec<ExprId> = Vec::with_capacity(coeffs.len());
    for cs in &coeffs {
        let s = match cs.len() {
            0 => arena.zero,
            1 => cs[0],
            _ => arena.add(cs),
        };
        a.push(crate::transforms::eval::eval(arena, s));
    }
    while a.len() > 1 && a.last().is_some_and(|&c| arena.is_zero_structural(c)) {
        a.pop();
    }
    let n = a.len() - 1;
    // Order 1 is the first-order linear solver's; all-rational coefficients
    // are `try_nth_order_linear_const_coeff`'s.
    if n < 2 || a.iter().all(|&c| arena.as_num(c).is_some()) {
        return None;
    }
    if !forcing.is_empty() && n != 2 {
        return None;
    }
    let r = arena.symbol("__r_sym");
    let mut terms = Vec::with_capacity(n + 1);
    for (k, &c) in a.iter().enumerate() {
        let rk = match k {
            0 => arena.one,
            1 => r,
            _ => {
                let ke = arena.int(k as i64);
                arena.pow(r, ke)
            }
        };
        terms.push(arena.mul(&[c, rk]));
    }
    let p = arena.add(&terms);
    let p = crate::transforms::eval::eval(arena, p);
    // Factor over ℤ[parameters] first (`r³ + a·r = r·(r² + a)`, `r² +
    // (a + 1)r + a = (r + 1)(r + a)`: the solver alone finds neither the
    // cubic's roots nor the rational roots of the quadratic, only
    // `√((a − 1)²)`); a factor linear in `r` gives its root directly, the
    // others go to the solver.  If `p` does not factor this way it is
    // solved whole.
    let (_, factors) = crate::simplify::factor::factor_list(arena, p, Some(r));
    let mut basis = Vec::with_capacity(n);
    let mut exps: Vec<(ExprId, usize)> = Vec::new();
    for (f, m) in factors {
        if !crate::base::walk::contains(arena, f, r) {
            continue; // a (generically non-zero) constant factor
        }
        let coeffs = crate::poly::polybridge::poly_coefficients(arena, f, r);
        let roots: Vec<ExprId> = match coeffs.as_deref() {
            Some([c0, c1]) => {
                let q = arena.div(*c0, *c1);
                vec![arena.neg(q)]
            }
            _ => known_roots(arena, f, r)
                .into_iter()
                .map(|s| s.value)
                .collect(),
        };
        if roots.is_empty() {
            return None;
        }
        // Multiplicity of each root in `f`: the number of derivatives
        // f, f′, … that vanish there (the first must, or the root is not
        // verified); in `p` it is that times the factor's multiplicity.
        for r0 in roots {
            let r0 = crate::transforms::eval::eval(arena, r0);
            if crate::base::walk::contains(arena, r0, var) || exps.iter().any(|&(e, _)| e == r0) {
                return None;
            }
            let mut d = f;
            let mut mult = 0usize;
            while mult < n {
                let at = crate::transforms::subs::subs(arena, d, r, r0);
                if !expands_to_zero(arena, at) {
                    break;
                }
                mult += 1;
                d = crate::transforms::diff::diff(arena, d, r);
            }
            if mult == 0 {
                return None;
            }
            let mult = mult * m as usize;
            exps.push((r0, mult));
            for j in 0..mult {
                basis.push(mode_real(arena, r0, j, var));
            }
        }
    }
    if basis.len() != n {
        return None;
    }
    let homogeneous = real_symbolic_basis(arena, &exps, var).unwrap_or_else(|| basis.clone());
    let mut constants = Vec::with_capacity(n);
    let mut sum = Vec::with_capacity(n + 1);
    for (i, &b) in homogeneous.iter().enumerate() {
        let c = arena.symbol(&format!("C{}", i + 1));
        constants.push(c);
        sum.push(arena.mul(&[c, b]));
    }
    if !forcing.is_empty() {
        // a₂y″ + a₁y′ + a₀y = −f:  y_p = −y₁∫y₂g/W + y₂∫y₁g/W, g = −f/a₂.
        let (y1, y2) = (basis[0], basis[1]);
        let f = arena.add(&forcing);
        let nf = arena.neg(f);
        let g = arena.div(nf, a[2]);
        let g = crate::transforms::eval::eval(arena, g);
        // W = y₁y₂′ − y₂y₁′: (r₂ − r₁)e^{(r₁+r₂)x}, or e^{2rx} for a double root.
        let w = if exps.len() == 2 {
            let d = arena.sub(exps[1].0, exps[0].0);
            let s = arena.add(&[exps[0].0, exps[1].0]);
            let sx = arena.mul(&[s, var]);
            let e = arena.exp(sx);
            arena.mul(&[d, e])
        } else {
            let two = arena.int(2);
            let sx = arena.mul(&[two, exps[0].0, var]);
            arena.exp(sx)
        };
        let w = crate::transforms::eval::eval(arena, w);
        let i1 = {
            let num = arena.mul(&[y2, g]);
            let q = arena.div(num, w);
            let q = crate::transforms::eval::eval(arena, q);
            integrate_forms(arena, q, var)
        };
        if integral_failed(arena, i1) {
            return None;
        }
        let i2 = {
            let num = arena.mul(&[y1, g]);
            let q = arena.div(num, w);
            let q = crate::transforms::eval::eval(arena, q);
            integrate_forms(arena, q, var)
        };
        if integral_failed(arena, i2) {
            return None;
        }
        let t1 = arena.mul(&[y1, i1]);
        let t1 = arena.neg(t1);
        let t2 = arena.mul(&[y2, i2]);
        let yp = arena.add(&[t1, t2]);
        // The integrals of `e^{r·x}·g` come back as `Piecewise` on the
        // parameters (`r ≠ 0`); the homogeneous part is already the generic
        // solution, so the generic branch is taken throughout.
        let yp = generic_branches(arena, yp, var)?;
        let yp = expand_combine(arena, yp);
        let yp = crate::poly::polybridge::together(arena, yp);
        let yp = crate::transforms::eval::eval(arena, yp);
        if crate::base::walk::has_unevaluated(arena, yp) {
            return None;
        }
        sum.push(yp);
    }
    let solution = arena.add(&sum);
    let solution = crate::transforms::eval::eval(arena, solution);
    // The particular solution took the generic branch of its integrals:
    // it is returned only if it verifies.
    if !forcing.is_empty() && !checkodesol(arena, expr, solution, func, var) {
        return None;
    }
    Some(OdeResult {
        solution,
        constants,
        branches: Vec::new(),
    })
}

/// The basis `xʲ·e^{r·x}` of [`try_linear_cc_symbolic`] made real where
/// the declared assumptions decide it: a root whose real and imaginary
/// parts are free of `i` and of unevaluated `re`/`im` (`√(−a)` with `a`
/// positive is `i·√a`), with a non-zero imaginary part and its conjugate
/// among the roots at the same multiplicity, gives `xʲ·e^{αx}·cos(βx)` and
/// `xʲ·e^{αx}·sin(βx)` (SymPy's form for a positive `a`: `y″ + a·y = 0`
/// gives `C1·cos(√a·x) + C2·sin(√a·x)`; before, the exponentials
/// `e^{±x√(−a)}`, and the initial-value solution `e^{−i√a·x}/2 + e^{i√a·x}/2`).
/// `None` when no root changes (the generic, assumption-free form stays).
fn real_symbolic_basis(
    arena: &mut Arena,
    exps: &[(ExprId, usize)],
    var: ExprId,
) -> Option<Vec<ExprId>> {
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    let parts: Vec<(ExprId, ExprId)> = exps
        .iter()
        .map(|&(r, _)| {
            // √u with −u declared positive is i·√(−u).
            let post = crate::base::walk::post_order_ids(arena, r);
            let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
            let half = Q::new(1.into(), 2.into());
            for &id in &post {
                let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
                let new = match *arena.node(rebuilt) {
                    ExprNode::Pow(b, e) if arena.as_num(e) == Some(&half) => {
                        let nb = arena.neg(b);
                        let nb = crate::transforms::eval::eval(arena, nb);
                        if Truth::from(reals.query(
                            arena,
                            nb,
                            crate::base::assumptions::Props::POSITIVE,
                        ))
                        .is_true()
                        {
                            let s = arena.pow(nb, e);
                            let i_unit = arena.i_unit;
                            arena.mul(&[i_unit, s])
                        } else {
                            rebuilt
                        }
                    }
                    _ => rebuilt,
                };
                cache.insert(id, new);
            }
            let r = cache.get(&r).copied().unwrap_or(r);
            let r = crate::transforms::eval::eval(arena, r);
            let (re, im) = arena.as_real_imag_expr(r);
            (
                crate::transforms::eval::eval(arena, re),
                crate::transforms::eval::eval(arena, im),
            )
        })
        .collect();
    let mut used = vec![false; exps.len()];
    let mut out = Vec::new();
    let mut changed = false;
    for i in 0..exps.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let (r, mult) = exps[i];
        let (re, im) = parts[i];
        let decided = !arena.is_zero_structural(im)
            && real_radical_form(arena, re)
            && real_radical_form(arena, im);
        let partner = decided
            .then(|| {
                (0..exps.len()).find(|&k| {
                    if used[k] || exps[k].1 != mult {
                        return false;
                    }
                    let (re2, im2) = parts[k];
                    let dre = arena.sub(re, re2);
                    let sim = arena.add(&[im, im2]);
                    expands_to_zero(arena, dre) && expands_to_zero(arena, sim)
                })
            })
            .flatten();
        match partner {
            Some(k) => {
                used[k] = true;
                changed = true;
                // β with the sign it has structurally positive when known.
                let beta = if arena.as_num(im).is_some_and(|q| q.is_negative())
                    || matches!(arena.node(im), ExprNode::Neg(_))
                {
                    parts[k].1
                } else {
                    im
                };
                for j in 0..mult {
                    let (c, s) = mode_complex(arena, re, beta, j, var);
                    out.push(c);
                    out.push(s);
                }
            }
            None => {
                for j in 0..mult {
                    out.push(mode_real(arena, r, j, var));
                }
            }
        }
    }
    changed.then_some(out)
}
