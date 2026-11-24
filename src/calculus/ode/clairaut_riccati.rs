//! Clairaut: y = x·y' + f(y') (part of [`ode`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Clairaut: y = x·y' + f(y')
// ═══════════════════════════════════════════════════════════════════════════

/// Recognise the Clairaut form `y = x·y' + f(y')` and return `f(p)` with
/// `p` standing for `y'`.
pub(super) fn clairaut_f(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
    p: ExprId,
) -> Option<ExprId> {
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));
    if !expr_contains(arena, expr, dy_dx) || expr_contains(arena, expr, d2y_dx2) {
        return None;
    }
    let g = crate::transforms::subs::subs(arena, expr, dy_dx, p);
    // Linear in y with a constant coefficient.
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, g, func)?;
    if coeffs.len() != 2 {
        return None;
    }
    let c_y = coeffs[1];
    let c_0 = coeffs[0];
    let p_sym = match arena.node(p) {
        ExprNode::Symbol(s) => *s,
        _ => return None,
    };
    if contains_sym(arena, c_y, var_sym) || contains_sym(arena, c_y, p_sym) {
        return None;
    }
    // g / c_y = y + c_0/c_y  must equal  y - x·p - f(p)  ⇒  f = -c_0/c_y - x·p
    let ratio = arena.div(c_0, c_y);
    let neg_ratio = arena.neg(ratio);
    let xp = arena.mul(&[var, p]);
    let f = arena.sub(neg_ratio, xp);
    let f = crate::transforms::eval::eval(arena, f);
    let f = crate::transforms::expand::expand(arena, f);
    let f = crate::transforms::eval::eval(arena, f);
    if contains_sym(arena, f, var_sym) || contains_sym(arena, f, func_sym) {
        return None;
    }
    Some(f)
}

/// Solve a Clairaut equation `y = x·y' + f(y')`: the general solution is
/// the family of lines `y = C·x + f(C)`.  (The singular envelope solution
/// is not returned.)
pub(super) fn try_clairaut(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    let p = arena.symbol("__clairaut_p");
    let f = clairaut_f(arena, expr, func, var, func_sym, var_sym, p)?;
    tracing::debug!("ode: Clairaut form recognised");
    let c1 = arena.symbol("C1");
    let f_c = crate::transforms::subs::subs(arena, f, p, c1);
    let cx = arena.mul(&[c1, var]);
    let solution = arena.add(&[cx, f_c]);
    let solution = crate::transforms::eval::eval(arena, solution);
    Some(OdeResult {
        solution,
        constants: vec![c1],
        branches: Vec::new(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Riccati: y' = q0(x) + q1(x)·y + q2(x)·y²
// ═══════════════════════════════════════════════════════════════════════════

/// Extract `(q0, q1, q2)` from a Riccati equation in zero form,
/// `c·y' - (q0 + q1·y + q2·y²) = 0` with `c` a nonzero constant.
/// Requires `q2 ≠ 0` and `q0 ≠ 0` (otherwise the equation is Bernoulli /
/// linear) and coefficients free of `y` and `y'`.
pub(super) fn riccati_coeffs(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));
    if expr_contains(arena, expr, d2y_dx2) {
        return None;
    }
    let (rest, c) = extract_m_n(arena, expr, func, var)?;
    if crate::base::walk::contains(arena, c, func) || crate::base::walk::contains(arena, c, var) {
        return None;
    }
    if crate::base::walk::contains(arena, rest, dy_dx) {
        return None;
    }
    // y' = -rest/c
    let neg_rest = arena.neg(rest);
    let rhs = arena.div(neg_rest, c);
    let rhs = crate::transforms::eval::eval(arena, rhs);
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, rhs, func)?;
    if coeffs.len() != 3 {
        return None;
    }
    let (q0, q1, q2) = (coeffs[0], coeffs[1], coeffs[2]);
    if arena.is_zero_structural(q0) || arena.is_zero_structural(q2) {
        return None;
    }
    Some((q0, q1, q2))
}

/// Solve a Riccati equation `y' = q0(x) + q1(x)·y + q2(x)·y²` given a known
/// particular solution `y_p(x)`.
///
/// The substitution `y = y_p + 1/v` turns the equation into the linear
/// ODE `v' + (q1 + 2·q2·y_p)·v = -q2`, which is solved with [`dsolve`];
/// the result is `y = y_p + 1/v`.
///
/// Returns `None` if `expr` is not of Riccati form, if `particular` does
/// not satisfy it, or if the linear equation for `v` cannot be solved.
pub fn solve_riccati(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    particular: ExprId,
) -> Option<OdeResult> {
    let (q0, q1, q2) = riccati_coeffs(arena, expr, func, var)?;
    // Verify the particular solution.
    if !checkodesol(arena, expr, particular, func, var) {
        // Try the derivative form directly: y_p' - (q0 + q1 y_p + q2 y_p²).
        let yp_prime = crate::transforms::diff::diff(arena, particular, var);
        let yp2 = arena.mul(&[particular, particular]);
        let t1 = arena.mul(&[q1, particular]);
        let t2 = arena.mul(&[q2, yp2]);
        let rhs = arena.add(&[q0, t1, t2]);
        let residual = arena.sub(yp_prime, rhs);
        let residual = crate::transforms::eval::eval(arena, residual);
        let residual = crate::transforms::expand::expand(arena, residual);
        let residual = crate::transforms::eval::eval(arena, residual);
        if !arena.is_zero_structural(residual) {
            return None;
        }
    }
    let _ = q0;
    let v = arena.symbol("__riccati_v");
    let dv = arena.intern(ExprNode::Derivative(v, var));
    // v' + (q1 + 2 q2 y_p) v + q2 = 0
    let two = arena.int(2);
    let two_q2_yp = arena.mul(&[two, q2, particular]);
    let coeff = arena.add(&[q1, two_q2_yp]);
    let coeff = crate::transforms::eval::eval(arena, coeff);
    let coeff_v = arena.mul(&[coeff, v]);
    let lin = arena.add(&[dv, coeff_v, q2]);
    let lin = crate::transforms::eval::eval(arena, lin);
    let v_res = dsolve(arena, lin, v, var)?;
    if crate::base::walk::contains(arena, v_res.solution, v) {
        return None; // implicit
    }
    let neg_one = arena.neg_one;
    let inv_v = arena.pow(v_res.solution, neg_one);
    let solution = arena.add(&[particular, inv_v]);
    let solution = crate::transforms::eval::eval(arena, solution);
    Some(OdeResult {
        solution,
        constants: v_res.constants,
        branches: Vec::new(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational Riccati: rational particular solutions found, general solution
// built from them (SymPy's `1st_rational_riccati`)
// ═══════════════════════════════════════════════════════════════════════════

/// A rational function `num/den` over ℚ in one variable, `den` monic and
/// coprime to `num` (`0/1` for zero).
#[derive(Clone, Debug, PartialEq)]
struct RatFn {
    num: crate::poly::Poly,
    den: crate::poly::Poly,
}

impl RatFn {
    fn new(num: crate::poly::Poly, den: crate::poly::Poly) -> Option<Self> {
        use crate::poly::Poly;
        if den.is_zero() {
            return None;
        }
        if num.is_zero() {
            return Some(Self {
                num: Poly::zero(),
                den: Poly::one(),
            });
        }
        let g = Poly::gcd(&num, &den);
        let (num, den) = if g.degree().unwrap_or(0) > 0 {
            (num.div(&g), den.div(&g))
        } else {
            (num, den)
        };
        let inv = Q::one() / den.leading_coeff()?.clone();
        Some(Self {
            num: num.scale(&inv),
            den: den.scale(&inv),
        })
    }
    fn poly(p: crate::poly::Poly) -> Self {
        Self {
            num: p,
            den: crate::poly::Poly::one(),
        }
    }
    fn constant(c: Q) -> Self {
        Self::poly(crate::poly::Poly::constant(c))
    }
    fn is_zero(&self) -> bool {
        self.num.is_zero()
    }
    fn add(&self, o: &Self) -> Option<Self> {
        Self::new(
            self.num.mul(&o.den).add(&o.num.mul(&self.den)),
            self.den.mul(&o.den),
        )
    }
    fn sub(&self, o: &Self) -> Option<Self> {
        self.add(&o.scale(&-Q::one()))
    }
    fn mul(&self, o: &Self) -> Option<Self> {
        Self::new(self.num.mul(&o.num), self.den.mul(&o.den))
    }
    fn div(&self, o: &Self) -> Option<Self> {
        if o.is_zero() {
            return None;
        }
        Self::new(self.num.mul(&o.den), self.den.mul(&o.num))
    }
    fn scale(&self, c: &Q) -> Self {
        Self::new(self.num.scale(c), self.den.clone()).unwrap_or_else(|| self.clone())
    }
    fn deriv(&self) -> Option<Self> {
        let n = self
            .num
            .derivative()
            .mul(&self.den)
            .sub(&self.num.mul(&self.den.derivative()));
        Self::new(n, self.den.mul(&self.den))
    }
    /// `deg den − deg num` (the valuation at infinity; large for zero).
    fn val_inf(&self) -> i64 {
        let d = |p: &crate::poly::Poly| i64::try_from(p.degree().unwrap_or(0)).unwrap_or(0);
        if self.is_zero() {
            return i64::MAX / 4;
        }
        d(&self.den) - d(&self.num)
    }
    fn to_expr(&self, arena: &mut Arena, var: ExprId) -> ExprId {
        let n = crate::poly::polybridge::poly_to_expr(arena, &self.num, var);
        if self.den.is_constant() {
            return n;
        }
        let d = crate::poly::polybridge::poly_to_expr(arena, &self.den, var);
        arena.div(n, d)
    }
    fn from_expr(arena: &mut Arena, e: ExprId, var: ExprId) -> Option<Self> {
        let t = crate::poly::polybridge::together(arena, e);
        let (n, d) = crate::poly::polybridge::as_numer_denom(arena, t);
        let n = crate::poly::polybridge::expr_to_poly(arena, n, var)?;
        let d = crate::poly::polybridge::expr_to_poly(arena, d, var)?;
        Self::new(n, d)
    }
}

/// The first `count` power-series coefficients of `p/q` at 0 (`q(0) ≠ 0`).
fn power_series(p: &crate::poly::Poly, q: &crate::poly::Poly, count: usize) -> Option<Vec<Q>> {
    use num_traits::Zero;
    let q0 = q.coeff(0);
    if q0.is_zero() {
        return None;
    }
    let qd = q.degree().unwrap_or(0);
    let mut g: Vec<Q> = Vec::with_capacity(count);
    for k in 0..count {
        let mut s = p.coeff(k);
        for j in 1..=k.min(qd) {
            s -= q.coeff(j) * &g[k - j];
        }
        g.push(s / &q0);
    }
    Some(g)
}

/// The Laurent coefficients of `f` at the pole `p` of order `m`: entry `i`
/// is the coefficient of `(x − p)^{i − m}`, `i < count`.
fn laurent_at_pole(f: &RatFn, p: &Q, m: usize, count: usize) -> Option<Vec<Q>> {
    use num_traits::Zero;
    let num = f.num.taylor_shift(p);
    let den = f.den.taylor_shift(p);
    if (0..m).any(|k| !den.coeff(k).is_zero()) {
        return None;
    }
    let den1 = crate::poly::Poly::from_coeffs(den.coeffs().get(m..)?.to_vec());
    power_series(&num, &den1, count)
}

/// The coefficients `rᵢ` of `f(x) = Σ rᵢ·x^{−v−i}` at infinity
/// (`v = val_inf f`), `i < count`.
fn laurent_at_infinity(f: &RatFn, count: usize) -> Option<Vec<Q>> {
    let rev = |p: &crate::poly::Poly| {
        let mut c = p.coeffs().to_vec();
        c.reverse();
        crate::poly::Poly::from_coeffs(c)
    };
    power_series(&rev(&f.num), &rev(&f.den), count)
}

/// `√q` when it is rational.
fn rational_sqrt(q: &Q) -> Option<Q> {
    if q.is_negative() {
        return None;
    }
    let (n, d) = (q.numer(), q.denom());
    let (rn, rd) = (n.sqrt(), d.sqrt());
    (&rn * &rn == *n && &rd * &rd == *d).then(|| Q::new(rn, rd))
}

/// Most finite poles of `a(x)` whose choices [`rational_riccati_particulars`]
/// combines (`2^(poles + 1)` combinations).
const MAX_RICCATI_POLES: usize = 6;
/// Largest degree of the auxiliary polynomial it solves for.
const MAX_RICCATI_AUX_DEGREE: i64 = 24;

/// The rational solutions of the normal Riccati equation `y′ + y² = a(x)`
/// (`a` rational over ℚ with rational poles), by Algorithm 11 of N. T. Vo,
/// *Rational and Algebraic Solutions of First-Order Algebraic ODEs* (PhD
/// thesis, RISC 2016), after SymPy's `riccati.py`: from the Laurent
/// expansions of `a` at its poles and at infinity, every candidate
/// `ȳ = Σ cᵢⱼ/(x − xᵢ)ʲ + Σ dₖ·xᵏ` with a degree bound `m` for the
/// auxiliary polynomial `P` (`P″ + 2ȳP′ + (ȳ′ + ȳ² − a)·P = 0`), and
/// `y = ȳ + P′/P` for each `P` found.  Only rational choices of the
/// square roots are followed (a solution through an irrational one is not
/// found); poles that are not rational make it give up.  Each solution is
/// checked exactly.  Unlike SymPy's, a polynomial `a` (no finite pole) is
/// handled too (`y′ + y² = x² + 1` has `y = x`).
fn rational_riccati_particulars(a: &RatFn) -> Option<Vec<RatFn>> {
    use crate::poly::Poly;
    use num_traits::{ToPrimitive, Zero};
    let one = Q::one();
    let two = Q::from_integer(2.into());
    // Poles and their orders.
    let mut poles: Vec<(Q, usize)> = Vec::new();
    if !a.den.is_constant() {
        let (_, factors) = a.den.factor_over_z();
        for (f, mult) in factors {
            if f.degree() != Some(1) {
                return None;
            }
            poles.push((-(f.coeff(0) / f.coeff(1)), usize::try_from(mult).ok()?));
        }
    }
    if poles.len() > MAX_RICCATI_POLES {
        return None;
    }
    let val_inf = a.val_inf();
    // Necessary conditions.
    if poles.iter().any(|&(_, m)| m != 1 && !m.is_multiple_of(2))
        || !(val_inf >= 2 || (val_inf <= 0 && val_inf % 2 == 0))
    {
        return Some(Vec::new());
    }
    // c-vectors per pole (entry j: coefficient of 1/(x − xᵢ)^{j+1}).
    let mut c_choices: Vec<Vec<Vec<Q>>> = Vec::with_capacity(poles.len());
    for (p, m) in &poles {
        let mut choices = Vec::new();
        if *m == 1 {
            choices.push(vec![one.clone()]);
        } else if *m == 2 {
            let ser = laurent_at_pole(a, p, 2, 1)?;
            let disc = &one + Q::from_integer(4.into()) * &ser[0];
            if disc.is_zero() {
                choices.push(vec![&one / &two]);
            } else if let Some(s) = rational_sqrt(&disc) {
                choices.push(vec![(&one + &s) / &two]);
                choices.push(vec![(&one - &s) / &two]);
            }
        } else {
            let ri = m / 2;
            // ser[key] = coefficient of (x − p)^{−key}.
            let raw = laurent_at_pole(a, p, *m, *m + 1)?;
            let ser = |key: usize| raw.get(*m - key).cloned().unwrap_or_else(Q::zero);
            let Some(lead) = rational_sqrt(&ser(2 * ri)) else {
                c_choices.push(Vec::new());
                continue;
            };
            for sign in [one.clone(), -one.clone()] {
                let mut c = vec![Q::zero(); ri];
                c[ri - 1] = &sign * &lead;
                if c[ri - 1].is_zero() {
                    continue;
                }
                let mut sm = Q::zero();
                let mut s_last = ri;
                for s in (1..ri).rev() {
                    sm = Q::zero();
                    for j in s + 1..ri {
                        sm += &c[j - 1] * &c[ri + s - j - 1];
                    }
                    if s != 1 {
                        c[s - 1] = (ser(ri + s) - &sm) / (&two * &c[ri - 1]);
                    }
                    s_last = s;
                }
                let ri_q = Q::from_integer(ri.into());
                c[0] = (ser(ri + s_last) - &sm - &ri_q * &c[ri - 1]) / (&two * &c[ri - 1]);
                if !choices.contains(&c) {
                    choices.push(c);
                }
            }
        }
        c_choices.push(choices);
    }
    // d-vectors: (d_0 … d_N, d_{−1}).
    let mut d_choices: Vec<(Vec<Q>, Q)> = Vec::new();
    if val_inf < 0 {
        let n = crate::base::cast::bounded_len(-val_inf / 2, crate::base::cast::MAX_SERIES_TERMS)?;
        let r = laurent_at_infinity(a, n + 2)?;
        // ser(j) = coefficient of x^j = r_{2N − j}.
        let ser = |j: i64| -> Q {
            let i = 2 * i64::try_from(n).unwrap_or(0) - j;
            usize::try_from(i)
                .ok()
                .and_then(|i| r.get(i).cloned())
                .unwrap_or_else(Q::zero)
        };
        let n_i = i64::try_from(n).ok()?;
        if let Some(lead) = rational_sqrt(&ser(2 * n_i)) {
            for sign in [one.clone(), -one.clone()] {
                let mut d = vec![Q::zero(); n + 1];
                d[n] = &sign * &lead;
                if d[n].is_zero() {
                    continue;
                }
                let mut sm = Q::zero();
                for s in (-1..n_i).rev() {
                    sm = Q::zero();
                    for j in (s + 1)..n_i {
                        let (ju, ku) =
                            (usize::try_from(j).ok()?, usize::try_from(n_i + s - j).ok()?);
                        sm += &d[ju] * &d[ku];
                    }
                    if s >= 0 {
                        let su = usize::try_from(s).ok()?;
                        d[su] = (ser(n_i + s) - &sm) / (&two * &d[n]);
                    }
                }
                let n_q = Q::from_integer(n.into());
                let dm1 = (ser(n_i - 1) - &n_q * &d[n] - &sm) / (&two * &d[n]);
                if !d_choices.iter().any(|(v, w)| *v == d && *w == dm1) {
                    d_choices.push((d, dm1));
                }
            }
        }
    } else if val_inf == 0 {
        let r = laurent_at_infinity(a, 2)?;
        if let Some(d0) = rational_sqrt(&r[0])
            && !d0.is_zero()
        {
            for sign in [one.clone(), -one.clone()] {
                let d0s = &sign * &d0;
                let dm1 = &r[1] / (&two * &d0s);
                d_choices.push((vec![d0s], dm1));
            }
        }
    } else {
        let s_inf = if val_inf == 2 {
            laurent_at_infinity(a, 1)?[0].clone()
        } else {
            Q::zero()
        };
        let disc = &one + Q::from_integer(4.into()) * &s_inf;
        if disc.is_zero() {
            d_choices.push((Vec::new(), &one / &two));
        } else if let Some(s) = rational_sqrt(&disc) {
            d_choices.push((Vec::new(), (&one + &s) / &two));
            d_choices.push((Vec::new(), (&one - &s) / &two));
        }
    }
    // Every combination of one c-vector per pole and one d-vector.
    let mut found: Vec<RatFn> = Vec::new();
    let mut idx = vec![0usize; c_choices.len()];
    if c_choices.iter().any(Vec::is_empty) || d_choices.is_empty() {
        return Some(found);
    }
    loop {
        for (d, dm1) in &d_choices {
            let mut m = dm1.clone();
            for (k, ch) in c_choices.iter().enumerate() {
                m -= &ch[idx[k]][0];
            }
            if m.is_integer() && !m.is_negative() {
                let m = m.to_integer().to_i64()?;
                if m <= MAX_RICCATI_AUX_DEGREE {
                    // ȳ
                    let mut ybar = RatFn::poly(Poly::from_coeffs(d.clone()));
                    for (k, (p, _)) in poles.iter().enumerate() {
                        let lin = Poly::from_coeffs(vec![-p.clone(), one.clone()]);
                        let mut pw = Poly::one();
                        for cij in &c_choices[k][idx[k]] {
                            pw = pw.mul(&lin);
                            let term = RatFn::new(Poly::constant(cij.clone()), pw.clone())?;
                            ybar = ybar.add(&term)?;
                        }
                    }
                    if let Some(y) = riccati_aux_solution(a, &ybar, usize::try_from(m).ok()?)
                        && !found.contains(&y)
                    {
                        found.push(y);
                    }
                }
            }
        }
        // Next combination.
        let mut k = 0;
        loop {
            if k == idx.len() {
                return Some(found);
            }
            idx[k] += 1;
            if idx[k] < c_choices[k].len() {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
}

/// `ȳ + P′/P` for a monic polynomial `P` of degree `m` with
/// `P″ + 2ȳ·P′ + (ȳ′ + ȳ² − a)·P = 0` (undetermined coefficients), when
/// it solves `y′ + y² = a` (checked exactly).
fn riccati_aux_solution(a: &RatFn, ybar: &RatFn, m: usize) -> Option<RatFn> {
    use crate::poly::Poly;
    use num_traits::Zero;
    let a0 = ybar.deriv()?.add(&ybar.mul(ybar)?)?.sub(a)?;
    let a1 = ybar.scale(&Q::from_integer(2.into()));
    // Clear denominators: D·P″ + B₁·P′ + B₀·P = 0.
    let lcm = {
        let g = Poly::gcd(&a0.den, &a1.den);
        a0.den.mul(&a1.den).div(&g)
    };
    let b0 = a0.num.mul(&lcm.div(&a0.den));
    let b1 = a1.num.mul(&lcm.div(&a1.den));
    let apply = |p: &Poly| -> Poly {
        let d1 = p.derivative();
        let d2 = d1.derivative();
        lcm.mul(&d2).add(&b1.mul(&d1)).add(&b0.mul(p))
    };
    let mono = |k: usize| Poly::monomial(Q::one(), k);
    let rhs = apply(&mono(m)).neg();
    let cols: Vec<Poly> = (0..m).map(|k| apply(&mono(k))).collect();
    let rows = cols
        .iter()
        .map(|c| c.degree().unwrap_or(0))
        .chain(std::iter::once(rhs.degree().unwrap_or(0)))
        .max()
        .unwrap_or(0)
        + 1;
    // Gaussian elimination over ℚ on [cols | rhs].
    let mut mat: Vec<Vec<Q>> = (0..rows)
        .map(|r| {
            let mut row: Vec<Q> = cols.iter().map(|c| c.coeff(r)).collect();
            row.push(rhs.coeff(r));
            row
        })
        .collect();
    let mut pivots: Vec<(usize, usize)> = Vec::new();
    let mut row = 0;
    for col in 0..m {
        let Some(pr) = (row..rows).find(|&r| !mat[r][col].is_zero()) else {
            continue;
        };
        mat.swap(row, pr);
        let inv = Q::one() / mat[row][col].clone();
        for v in mat[row].iter_mut() {
            *v *= &inv;
        }
        let pivot_row = mat[row].clone();
        for (r, target) in mat.iter_mut().enumerate() {
            if r != row && !target[col].is_zero() {
                let f = target[col].clone();
                for (t, p) in target.iter_mut().zip(&pivot_row) {
                    *t -= &f * p;
                }
            }
        }
        pivots.push((row, col));
        row += 1;
    }
    if (row..rows).any(|r| !mat[r][m].is_zero()) {
        return None;
    }
    let mut coeffs = vec![Q::zero(); m + 1];
    coeffs[m] = Q::one();
    for (r, c) in pivots {
        coeffs[c] = mat[r][m].clone();
    }
    let p = Poly::from_coeffs(coeffs);
    let y = ybar.add(&RatFn::new(p.derivative(), p)?)?;
    // Exact check: y′ + y² − a = 0.
    y.deriv()?.add(&y.mul(&y)?)?.sub(a)?.is_zero().then_some(y)
}

/// Solve the Riccati equation `y′ = b₀ + b₁·y + b₂·y²` with rational
/// coefficients over ℚ (`b₀, b₂ ≠ 0`) from its rational particular
/// solutions ([`rational_riccati_particulars`] on the normal form
/// `v′ + v² = a(x)`, `v = −b₂·y − b₂′/(2b₂) − b₁/2`): with two particular
/// solutions `v₁, v₂` the general solution is `(v₂·u − v₁)/(u − 1)`,
/// `u = C1·exp(∫(v₂ − v₁))`; with one, `v₁ + 1/z`, `z′ = 2v₁·z + 1`; with
/// `a = 0`, `v = 1/(x + C1)`.  SymPy's `1st_rational_riccati` (BSD; see
/// THIRD-PARTY-NOTICES.md).  `y′ = y² − 2/x²` needed a particular solution
/// passed to [`solve_riccati`]; `dsolve` refused it.
pub(super) fn try_rational_riccati(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<OdeResult> {
    let rhs = first_order_rhs(arena, expr, func, var)?;
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, rhs, func)?;
    if coeffs.len() != 3 || coeffs.iter().any(|&c| contains_sym(arena, c, func_sym)) {
        return None;
    }
    // Rational functions of `var` alone.
    for &c in &coeffs {
        if crate::base::walk::free_symbols(arena, c)
            .iter()
            .any(|&s| s != var)
        {
            return None;
        }
    }
    let b0 = RatFn::from_expr(arena, coeffs[0], var)?;
    let b1 = RatFn::from_expr(arena, coeffs[1], var)?;
    let b2 = RatFn::from_expr(arena, coeffs[2], var)?;
    if b0.is_zero() || b2.is_zero() {
        return None;
    }
    // a = −b₀b₂ + b₁²/4 − b₁′/2 + 3b₂′²/(4b₂²) + b₁b₂′/(2b₂) − b₂″/(2b₂)
    let q = |n: i64, d: i64| Q::new(n.into(), d.into());
    let b1p = b1.deriv()?;
    let b2p = b2.deriv()?;
    let b2pp = b2p.deriv()?;
    let a = b0
        .mul(&b2)?
        .scale(&q(-1, 1))
        .add(&b1.mul(&b1)?.scale(&q(1, 4)))?
        .sub(&b1p.scale(&q(1, 2)))?
        .add(&b2p.mul(&b2p)?.div(&b2.mul(&b2)?)?.scale(&q(3, 4)))?
        .add(&b1.mul(&b2p)?.div(&b2)?.scale(&q(1, 2)))?
        .sub(&b2pp.div(&b2)?.scale(&q(1, 2)))?;
    let c1 = arena.symbol("C1");
    // The general solution v of the normal form.
    let v = if a.is_zero() {
        let s = arena.add(&[var, c1]);
        let m1 = arena.int(-1);
        arena.pow(s, m1)
    } else {
        let parts = rational_riccati_particulars(&a)?;
        let parts: Vec<ExprId> = parts.iter().map(|p| p.to_expr(arena, var)).collect();
        match parts.as_slice() {
            [] => return None,
            [v1] => {
                // v₁ + 1/z, z = e^{F}·(C1 + ∫ e^{−F}), F = ∫ 2v₁.
                let two = arena.int(2);
                let tv = arena.mul(&[two, *v1]);
                let f = integrate_forms(arena, tv, var);
                if integral_failed(arena, f) {
                    return None;
                }
                let ef = exp_of_log_sum(arena, f);
                let m1 = arena.int(-1);
                let emf = arena.pow(ef, m1);
                let emf = crate::transforms::eval::eval(arena, emf);
                let g = integrate_forms(arena, emf, var);
                if integral_failed(arena, g) {
                    return None;
                }
                let inner = arena.add(&[c1, g]);
                let z = arena.mul(&[ef, inner]);
                let inv_z = arena.pow(z, m1);
                arena.add(&[*v1, inv_z])
            }
            [v1, v2, ..] => {
                let neg_v1 = arena.neg(*v1);
                let diff = arena.add(&[*v2, neg_v1]);
                let diff = crate::transforms::eval::eval(arena, diff);
                let f = integrate_forms(arena, diff, var);
                if integral_failed(arena, f) {
                    return None;
                }
                let ef = exp_of_log_sum(arena, f);
                let u = arena.mul(&[c1, ef]);
                let v2u = arena.mul(&[*v2, u]);
                let num = arena.add(&[v2u, neg_v1]);
                let m1 = arena.int(-1);
                let den = arena.add(&[u, m1]);
                arena.div(num, den)
            }
        }
    };
    // y = −v/b₂ + bp, bp = −b₂′/(2b₂²) − b₁/(2b₂).
    let bp = b2p
        .div(&b2.mul(&b2)?)?
        .scale(&q(-1, 2))
        .sub(&b1.div(&b2)?.scale(&q(1, 2)))?;
    let inv_b2 = RatFn::constant(Q::one()).div(&b2)?.scale(&q(-1, 1));
    let inv_b2 = inv_b2.to_expr(arena, var);
    let bp = bp.to_expr(arena, var);
    let vy = arena.mul(&[inv_b2, v]);
    let sol = arena.add(&[vy, bp]);
    let sol = crate::transforms::eval::eval(arena, sol);
    let sol = crate::simplify::ratsimp::ratsimp(arena, sol);
    let sol = crate::transforms::eval::eval(arena, sol);
    if !crate::base::walk::contains(arena, sol, c1)
        || crate::base::walk::has_unevaluated(arena, sol)
        || !family_verifies(arena, expr, sol, func, var, c1)
    {
        return None;
    }
    Some(OdeResult {
        solution: sol,
        constants: vec![c1],
        branches: Vec::new(),
    })
}
