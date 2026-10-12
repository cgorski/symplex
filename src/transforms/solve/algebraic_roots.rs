//! Roots of a polynomial whose coefficients are algebraic numbers (`√3`,
//! `∛2`, `i`, the golden ratio, …): the equations in `t` that the
//! substitutions of [`super`] produce (`sin 4x = √3·sin x` in `t =
//! tan(x/2)`, `e^(3x) − √2·eˣ = 1/4` in `t = eˣ`, `cos³x = √2·cos x` in
//! `t = cos x`), and polynomial equations over fields beyond square roots
//! (`(x − ∛2)·(x − i)² = 0`).
//!
//! SymPy's `roots` over an algebraic field: the polynomial is factored over
//! the field `K` its coefficients generate (Trager's norm method,
//! [`crate::poly::numfield`], SymPy's `dup_ext_factor`), and each
//! irreducible factor is solved by the quadratic formula, Cardano's
//! formulas (the trigonometric form for three real roots), a factor in
//! `t²` through its roots in `t²`, or else through the roots of its norm
//! over ℚ, of which those of the factor are told apart by a certified
//! evaluation: the norm `N = ∏ g^σ` vanishes at each of its roots, so when
//! all but `deg g` of them certainly are not roots of `g`, the others are.
//! Every root is finally substituted into the polynomial (zero to the
//! precision of the zero search) — `None` when anything is not decided.

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::poly::numfield::{self, AlgNum, ExtField, KPoly};
use crate::poly::traits::Ring;
use num_bigint::BigInt;

/// Largest degree in the variable of a polynomial this module factors.
const MAX_DEGREE: usize = 32;

/// Largest `[K : ℚ]·deg f` (bounded by the product of the degrees of the
/// atoms) of a polynomial factored only for its factors of degree ≥ 5.
const MAX_BEYOND_QUARTIC_NORM_DEGREE: usize = 16;

/// Largest nesting of the `g(t) = h(t²)` reduction.
const MAX_EVEN_REDUCTIONS: u32 = 3;

/// The algebraic numbers in the coefficients of the polynomial `expr` in
/// `var` (its non-polynomial atoms), at least one; `None` when an atom
/// involves `var` or a free symbol.
fn algebraic_atoms(arena: &Arena, expr: ExprId, var: ExprId) -> Option<Vec<ExprId>> {
    let mut atoms: Vec<ExprId> = Vec::new();
    for g in numfield::generators(arena, &[expr]) {
        if g == var || arena.as_num(g).is_some() || atoms.contains(&g) {
            continue;
        }
        if crate::base::walk::contains(arena, g, var)
            || !crate::base::walk::free_symbols(arena, g).is_empty()
        {
            return None;
        }
        atoms.push(g);
    }
    (!atoms.is_empty()).then_some(atoms)
}

/// Is some algebraic coefficient of the polynomial `expr` in `var` other
/// than a square root (or `i`): `∛2`, `2^(1/4)`?  (`Ex::solve` takes the
/// polynomials over square-root towers itself.)  `false` when there is no
/// such atom or an atom is not a constant.
pub(super) fn beyond_square_roots(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    let Some(atoms) = algebraic_atoms(arena, expr, var) else {
        return false;
    };
    atoms.iter().any(|&a| match arena.node(a) {
        ExprNode::ImaginaryUnit => false,
        ExprNode::Pow(_, e) => !arena
            .as_num(*e)
            .is_some_and(|q| q.denom() == &BigInt::from(2)),
        _ => true,
    })
}

/// Every distinct root of the polynomial `expr` in `var` whose
/// coefficients are algebraic numbers, not all rational (see the module
/// documentation).  `None` when a coefficient is not such a number, the
/// field or a norm is beyond the budgets of [`crate::poly::numfield`], a
/// factor is not solved, or a root is not confirmed — and, with
/// `beyond_quartic`, when no irreducible factor over the field has degree
/// 5 or more (the polynomials whose roots `Ex::solve` writes in radicals
/// itself).
pub(super) fn number_field_roots(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    beyond_quartic: bool,
) -> Option<Vec<ExprId>> {
    let expanded = crate::transforms::expand::expand(arena, expr);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    let atoms = algebraic_atoms(arena, expanded, var)?;
    // The product of the degrees bounds `[K : ℚ]`; beyond the field budget
    // the search for a primitive element (minimal polynomials of sums of the
    // atoms) would take seconds before giving up.
    let bound = field_degree_bound(arena, &atoms)?;
    // A polynomial that `Ex::solve` most likely solves itself is only
    // factored when that is cheap: a norm of degree 40 (`√2·(x + √3)²·(x −
    // 1 + i)·(x + 1/2)²` over `ℚ(√2, √3, i)`) took 3 s.
    if beyond_quartic {
        let degree = super::symbolic_poly_coeffs(arena, expanded, var)?
            .len()
            .saturating_sub(1);
        if bound.saturating_mul(degree) > MAX_BEYOND_QUARTIC_NORM_DEGREE {
            return None;
        }
    }
    let mut ef = ExtField::for_atoms(arena, &atoms)?;
    let f = ef.kpoly_of(arena, expanded, var)?;
    let degree = f.degree()?;
    if degree == 0 || degree > MAX_DEGREE {
        return None;
    }
    let fac = ef.field.factor(&f)?;
    if beyond_quartic
        && fac
            .factors
            .iter()
            .all(|(g, _)| g.degree().unwrap_or(0) <= 4)
    {
        return None;
    }
    let mut roots: Vec<ExprId> = Vec::with_capacity(degree);
    for (g, _) in &fac.factors {
        for r in factor_roots(arena, &ef, g, var, 0)? {
            let r = crate::transforms::eval::eval(arena, r);
            if !roots.contains(&r) {
                roots.push(r);
            }
        }
    }
    for &r in &roots {
        if !super::member_is_root(arena, expanded, var, r) {
            tracing::warn!("solve: a root over a number field does not check; refused");
            return None;
        }
    }
    Some(roots)
}

/// A bound on `[K : ℚ]` for the field generated by `atoms`, `None` beyond
/// [`numfield::MAX_FIELD_DEGREE`]: the product over the atoms of their
/// degrees, the rational powers `b^(p/q)` of one rational `b` counted
/// together (they lie in `ℚ(b^(1/L))`, `L` the lcm of the `q`: `∛2` and
/// `∛4 = 2^(2/3)` from `(x − ∛2)²` give 3, not 9).
fn field_degree_bound(arena: &mut Arena, atoms: &[ExprId]) -> Option<usize> {
    let mut bound: usize = 1;
    let mut radical_groups: Vec<(ExprId, BigInt)> = Vec::new();
    for &a in atoms {
        if let ExprNode::Pow(b, e) = *arena.node(a)
            && arena.as_num(b).is_some()
            && let Some(q) = arena.as_num(e)
        {
            let den = q.denom().clone();
            match radical_groups.iter_mut().find(|(base, _)| *base == b) {
                Some((_, l)) => *l = num_integer::Integer::lcm(l, &den),
                None => radical_groups.push((b, den)),
            }
            continue;
        }
        let d = numfield::algebraic_minpoly(arena, a)?.degree()?;
        bound = bound.checked_mul(d)?;
        if bound > numfield::MAX_FIELD_DEGREE {
            return None;
        }
    }
    for (_, l) in radical_groups {
        let l = num_traits::ToPrimitive::to_usize(&l)?;
        bound = bound.checked_mul(l)?;
        if bound > numfield::MAX_FIELD_DEGREE {
            return None;
        }
    }
    Some(bound)
}

fn rat(n: i64, d: i64) -> AlgNum {
    AlgNum::Rat(Q::new(BigInt::from(n), BigInt::from(d)))
}

fn sqrt_of(arena: &mut Arena, e: ExprId) -> ExprId {
    let half = arena.rational(1, 2);
    let r = arena.pow(e, half);
    crate::transforms::eval::eval(arena, r)
}

/// The roots of a factor `g` (square-free) over the field of `ef`.
fn factor_roots(
    arena: &mut Arena,
    ef: &ExtField,
    g: &KPoly,
    var: ExprId,
    depth: u32,
) -> Option<Vec<ExprId>> {
    let g = g.make_monic();
    let d = g.degree()?;
    let c = |k: usize| g.coeff(k);
    match d {
        0 => Some(Vec::new()),
        1 => Some(vec![ef.element_to_expr(arena, &Ring::neg(&c(0)))]),
        2 => Some(quadratic_roots(arena, ef, &c(1), &c(0))),
        3 => cubic_roots(arena, ef, &c(2), &c(1), &c(0)),
        _ => {
            if depth < MAX_EVEN_REDUCTIONS && (1..=d).step_by(2).all(|k| c(k).is_zero()) {
                // g(t) = h(t²): t = ±√s for the roots s of h
                let h = KPoly::from_coeffs((0..=d / 2).map(|k| c(2 * k)).collect());
                let mut out = Vec::with_capacity(d);
                for s in factor_roots(arena, ef, &h, var, depth + 1)? {
                    let r = sqrt_of(arena, s);
                    out.push(r);
                    out.push(arena.neg(r));
                }
                return Some(out);
            }
            norm_roots(arena, ef, &g, var)
        }
    }
}

/// `t² + b·t + c`: `−b/2 ± √(b²/4 − c)`.
fn quadratic_roots(arena: &mut Arena, ef: &ExtField, b: &AlgNum, c: &AlgNum) -> Vec<ExprId> {
    let disc = Ring::sub(&Ring::mul(&Ring::mul(b, b), &rat(1, 4)), c);
    let r = Ring::neg(&Ring::mul(b, &rat(1, 2)));
    let r = ef.element_to_expr(arena, &r);
    let disc = ef.element_to_expr(arena, &disc);
    let disc = crate::transforms::eval::eval(arena, disc);
    let sq = sqrt_of(arena, disc);
    let minus = arena.neg(sq);
    vec![arena.add(&[r, sq]), arena.add(&[r, minus])]
}

/// `−1/2 + (√3/2)·i` and its conjugate.
fn cube_roots_of_unity(arena: &mut Arena) -> [ExprId; 2] {
    let three = arena.int(3);
    let sqrt3 = sqrt_of(arena, three);
    let half = arena.rational(1, 2);
    let minus_half = arena.rational(-1, 2);
    let i = arena.i_unit;
    let im = arena.mul(&[half, sqrt3, i]);
    let w = arena.add(&[minus_half, im]);
    let neg_im = arena.neg(im);
    let w_bar = arena.add(&[minus_half, neg_im]);
    [w, w_bar]
}

/// The real cube root of a real constant `e` of sign `sign`.
fn real_cbrt(arena: &mut Arena, e: ExprId, sign: i8) -> ExprId {
    let third = arena.rational(1, 3);
    match sign {
        0 => arena.zero,
        s if s > 0 => arena.pow(e, third),
        _ => {
            let minus = arena.neg(e);
            let minus = crate::transforms::eval::eval(arena, minus);
            let r = arena.pow(minus, third);
            arena.neg(r)
        }
    }
}

/// The sign of a constant when it is real and decided.
fn real_sign(arena: &mut Arena, e: ExprId) -> Option<i8> {
    crate::poly::algebraic::sign_checked(arena, e)
}

/// `t³ + a·t² + b·t + c`, square-free, by Cardano's formulas on
/// `y³ + p·y + q` (`t = y − a/3`): for real `p`, `q` the trigonometric
/// form `y = 2√(−p/3)·cos(θ/3 − 2πk/3)` when the three roots are real
/// (`Δ = q²/4 + p³/27 < 0`) and real cube roots otherwise; for complex
/// ones the principal `u = ∛(−q/2 + √Δ)` and `v = −p/(3u)`.  The formulas
/// are exact whatever the branches (`u·v = −p/3` holds by construction).
fn cubic_roots(
    arena: &mut Arena,
    ef: &ExtField,
    a: &AlgNum,
    b: &AlgNum,
    c: &AlgNum,
) -> Option<Vec<ExprId>> {
    let a2 = Ring::mul(a, a);
    let p = Ring::sub(b, &Ring::mul(&a2, &rat(1, 3)));
    let q = Ring::add(
        &Ring::sub(
            &Ring::mul(&Ring::mul(&a2, a), &rat(2, 27)),
            &Ring::mul(&Ring::mul(a, b), &rat(1, 3)),
        ),
        c,
    );
    let shift = ef.element_to_expr(arena, &Ring::neg(&Ring::mul(a, &rat(1, 3))));
    let [w, w_bar] = cube_roots_of_unity(arena);
    let ys: Vec<ExprId> = if p.is_zero() {
        // y³ = −q
        let mq = ef.element_to_expr(arena, &Ring::neg(&q));
        let mq = crate::transforms::eval::eval(arena, mq);
        let r = match real_sign(arena, mq) {
            Some(s) => real_cbrt(arena, mq, s),
            None => {
                let third = arena.rational(1, 3);
                arena.pow(mq, third)
            }
        };
        vec![r, arena.mul(&[w, r]), arena.mul(&[w_bar, r])]
    } else {
        let disc = Ring::add(
            &Ring::mul(&Ring::mul(&q, &q), &rat(1, 4)),
            &Ring::mul(&Ring::mul(&Ring::mul(&p, &p), &p), &rat(1, 27)),
        );
        if disc.is_zero() {
            return None;
        }
        let p_e = ef.element_to_expr(arena, &p);
        let q_e = ef.element_to_expr(arena, &q);
        let disc_e = ef.element_to_expr(arena, &disc);
        let disc_e = crate::transforms::eval::eval(arena, disc_e);
        let real = real_sign(arena, p_e).is_some() && real_sign(arena, q_e).is_some();
        let disc_sign = if real { real_sign(arena, disc_e) } else { None };
        match disc_sign {
            Some(s) if s < 0 => {
                // three real roots: 2√(−p/3)·cos(θ/3 − 2πk/3),
                // θ = acos((3q/(2p))·√(−3/p))
                let m = ef.element_to_expr(arena, &Ring::neg(&Ring::mul(&p, &rat(1, 3))));
                let m = sqrt_of(arena, m);
                let two = arena.int(2);
                let amp = arena.mul(&[two, m]);
                let ratio = ef.element_to_expr(
                    arena,
                    &Ring::mul(
                        &Ring::mul(&q, &rat(3, 2)),
                        &crate::poly::traits::Field::inv(&p),
                    ),
                );
                let inv = ef.element_to_expr(
                    arena,
                    &Ring::mul(&rat(-3, 1), &crate::poly::traits::Field::inv(&p)),
                );
                let inv = sqrt_of(arena, inv);
                let arg = arena.mul(&[ratio, inv]);
                let arg = crate::transforms::eval::eval(arena, arg);
                let theta = arena.acos(arg);
                let third = arena.rational(1, 3);
                let theta3 = arena.mul(&[third, theta]);
                let mut out = Vec::with_capacity(3);
                for k in 0..3i64 {
                    let turn = arena.rational(-2 * k, 3);
                    let phase = arena.mul(&[turn, arena.pi]);
                    let angle = arena.add(&[theta3, phase]);
                    let cos = arena.cos(angle);
                    out.push(arena.mul(&[amp, cos]));
                }
                out
            }
            Some(_) => {
                // one real root: real cube roots u, v of −q/2 ± √Δ
                let minus_half = arena.rational(-1, 2);
                let half_q = arena.mul(&[minus_half, q_e]);
                let sq = sqrt_of(arena, disc_e);
                let plus = arena.add(&[half_q, sq]);
                let plus = crate::transforms::eval::eval(arena, plus);
                let neg_sq = arena.neg(sq);
                let minus = arena.add(&[half_q, neg_sq]);
                let minus = crate::transforms::eval::eval(arena, minus);
                let (su, sv) = (real_sign(arena, plus)?, real_sign(arena, minus)?);
                let u = real_cbrt(arena, plus, su);
                let v = real_cbrt(arena, minus, sv);
                let wu = arena.mul(&[w, u]);
                let wbv = arena.mul(&[w_bar, v]);
                let wbu = arena.mul(&[w_bar, u]);
                let wv = arena.mul(&[w, v]);
                vec![
                    arena.add(&[u, v]),
                    arena.add(&[wu, wbv]),
                    arena.add(&[wbu, wv]),
                ]
            }
            None => {
                // principal u = ∛(−q/2 + √Δ), v = −p/(3u)
                let minus_half = arena.rational(-1, 2);
                let half_q = arena.mul(&[minus_half, q_e]);
                let sq = sqrt_of(arena, disc_e);
                let rad = arena.add(&[half_q, sq]);
                let rad = crate::transforms::eval::eval(arena, rad);
                let third = arena.rational(1, 3);
                let u = arena.pow(rad, third);
                let minus_third = arena.rational(-1, 3);
                let mut out = Vec::with_capacity(3);
                for wk in [arena.one, w, w_bar] {
                    let uk = arena.mul(&[wk, u]);
                    let inv = arena.pow(uk, arena.neg_one);
                    let vk = arena.mul(&[minus_third, p_e, inv]);
                    out.push(arena.add(&[uk, vk]));
                }
                out
            }
        }
    };
    Some(ys.into_iter().map(|y| arena.add(&[y, shift])).collect())
}

/// The roots of `g` (monic, square-free over `K`) among those of its norm
/// `N(g) = ∏ g^σ` over ℚ (radicals through degree 4, `RootOf` beyond):
/// the roots of `N(g)` at which `g` is certainly not zero are left out,
/// and the others are the roots of `g` when exactly `deg g` remain (each
/// root of `g` is one of `N(g)`, and `g` has `deg g` distinct ones).
fn norm_roots(arena: &mut Arena, ef: &ExtField, g: &KPoly, var: ExprId) -> Option<Vec<ExprId>> {
    let d = g.degree()?;
    let norm = ef.field.norm(g)?;
    let (_, factors) = norm.factor_over_z();
    let g_expr = ef.kpoly_to_expr(arena, g, var);
    let mut kept: Vec<ExprId> = Vec::with_capacity(d);
    for (h, _) in &factors {
        let hd = h.degree().unwrap_or(0);
        if hd == 0 {
            continue;
        }
        let sols = super::solve_rational_poly(arena, var, h);
        if sols.len() != hd {
            return None;
        }
        for s in sols {
            if !super::certainly_not_a_root(arena, g_expr, var, s.value) {
                kept.push(s.value);
                if kept.len() > d {
                    return None;
                }
            }
        }
    }
    (kept.len() == d).then_some(kept)
}
