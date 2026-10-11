//! Polynomials over a simple algebraic number field `K = ℚ(α) = ℚ[t]/(m(t))`
//! and their factorization by Trager's norm method.
//!
//! * [`AlgNum`] — an element of `K`, a [`Field`], so that
//!   [`GenPoly<AlgNum>`] ([`KPoly`]) has Euclidean division, gcd, the
//!   extended gcd and Yun's square-free decomposition.  A rational element
//!   is stored as such (`AlgNum::Rat`), so `zero()` and `one()` need no
//!   field; every other element carries the (shared) minimal polynomial.
//! * [`NumberField::norm`] — `N(f)(x) = res_t(m(t), f(x, t)) = ∏ f^σ(x)`
//!   over the conjugates, computed by evaluation at `x = 0, 1, …, d·n` and
//!   interpolation.
//! * [`NumberField::factor`] — square-free decomposition over `K`, then for
//!   each square-free part `f` Trager's algorithm (B. M. Trager,
//!   "Algebraic factoring and rational function integration", *Proc.
//!   SYMSAC '76*, 219–226; `alg_factor`): find `s` with `N(f(x − sα))`
//!   square-free, factor that norm over ℤ (Berlekamp–Zassenhaus) and take
//!   `gcd(f(x − sα), gᵢ(x))` over `K` for each irreducible factor `gᵢ`,
//!   shifted back by `x ↦ x + sα`.  The structure follows SymPy's
//!   `dup_sqf_norm` / `dup_ext_factor`
//!   (`sympy/polys/sqfreetools.py`, `sympy/polys/factortools.py`).
//!   Every factorization is verified by multiplying back; the norm degree
//!   `[K : ℚ]·deg f` is limited by [`MAX_NORM_DEGREE`] (beyond it the
//!   factorization is refused, `None`).
//! * The bridge to expressions: [`ExtField`] picks a primitive element for
//!   the algebraic numbers met in the coefficients (`√2`, `i`, `∛2`, …),
//!   expresses each of them in it (a root of its minimal polynomial over
//!   `K`, certified by an exact zero test), and converts polynomials in a
//!   variable with such coefficients to and from [`KPoly`].

use std::sync::Arc;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::{FxHashMap, FxHashSet};

use super::dense::Poly;
use super::generic::GenPoly;
use super::traits::{EuclideanDomain, Field, Ring};
use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;

type Q = Ratio<BigInt>;

fn qzero() -> Q {
    <Q as Zero>::zero()
}

fn qone() -> Q {
    <Q as One>::one()
}

/// Largest `[K : ℚ]·deg f` for which a norm is computed and factored over ℤ.
pub(crate) const MAX_NORM_DEGREE: usize = 64;

/// Largest degree `[K : ℚ]` of a field built from the algebraic numbers of
/// an expression.
pub(crate) const MAX_FIELD_DEGREE: usize = 8;

/// Largest degree in the variable of a polynomial converted from an
/// expression.
const MAX_KPOLY_DEGREE: usize = 200;

/// Shifts `s = 0, 1, −1, 2, …` tried for a square-free norm.
const MAX_SHIFTS: usize = 40;

/// Largest number of decimal digits of any coefficient (numerator or
/// denominator) in an input polynomial or a field element.
const MAX_COEFF_DIGITS: u64 = 200;

/// Largest size in bits of a coefficient of a norm handed to the
/// factorization over ℤ.
const MAX_NORM_COEFF_BITS: u64 = 4_000;

// ═══════════════════════════════════════════════════════════════════════════
// The coefficient field
// ═══════════════════════════════════════════════════════════════════════════

/// An element of `ℚ(α)`: a rational number, or a polynomial in `t` of degree
/// `1 ≤ deg < [K : ℚ]` reduced modulo the (monic, irreducible) minimal
/// polynomial it carries.
#[derive(Clone, Debug)]
pub(crate) enum AlgNum {
    /// A rational element.
    Rat(Q),
    /// `p(α)` with `p` of degree at least 1, reduced modulo `m`.
    Elt(Poly, Arc<Poly>),
}

impl AlgNum {
    /// `p(α)` reduced modulo `m`.
    fn reduce(p: Poly, m: &Arc<Poly>) -> Self {
        let r = if p.degree().unwrap_or(0) >= m.degree().unwrap_or(0) {
            p.rem(m)
        } else {
            p
        };
        match r.degree() {
            None => AlgNum::Rat(qzero()),
            Some(0) => AlgNum::Rat(r.coeff(0)),
            Some(_) => AlgNum::Elt(r, Arc::clone(m)),
        }
    }

    /// The representative polynomial in `t` (degree below `[K : ℚ]`).
    pub(crate) fn to_poly(&self) -> Poly {
        match self {
            AlgNum::Rat(q) => Poly::constant(q.clone()),
            AlgNum::Elt(p, _) => p.clone(),
        }
    }

    /// The rational value, if the element is rational.
    pub(crate) fn as_rational(&self) -> Option<&Q> {
        match self {
            AlgNum::Rat(q) => Some(q),
            AlgNum::Elt(..) => None,
        }
    }

    /// Largest number of decimal digits of a numerator or denominator.
    fn digits(&self) -> u64 {
        let q_digits = |q: &Q| q.numer().bits().max(q.denom().bits()) * 3 / 10 + 1;
        match self {
            AlgNum::Rat(q) => q_digits(q),
            AlgNum::Elt(p, _) => p.coeffs().iter().map(q_digits).max().unwrap_or(0),
        }
    }
}

impl PartialEq for AlgNum {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AlgNum::Rat(a), AlgNum::Rat(b)) => a == b,
            (AlgNum::Elt(p, _), AlgNum::Elt(q, _)) => p == q,
            _ => false,
        }
    }
}

impl Ring for AlgNum {
    fn zero() -> Self {
        AlgNum::Rat(qzero())
    }

    fn one() -> Self {
        AlgNum::Rat(qone())
    }

    fn is_zero(&self) -> bool {
        matches!(self, AlgNum::Rat(q) if Zero::is_zero(q))
    }

    fn is_one(&self) -> bool {
        matches!(self, AlgNum::Rat(q) if One::is_one(q))
    }

    fn add(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (AlgNum::Rat(a), AlgNum::Rat(b)) => AlgNum::Rat(Ring::add(a, b)),
            (AlgNum::Rat(a), AlgNum::Elt(p, m)) | (AlgNum::Elt(p, m), AlgNum::Rat(a)) => {
                AlgNum::Elt(p.add(&Poly::constant(a.clone())), Arc::clone(m))
            }
            (AlgNum::Elt(p, m), AlgNum::Elt(q, _)) => AlgNum::reduce(p.add(q), m),
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        Ring::add(self, &Ring::neg(rhs))
    }

    fn mul(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (AlgNum::Rat(a), AlgNum::Rat(b)) => AlgNum::Rat(Ring::mul(a, b)),
            (AlgNum::Rat(a), AlgNum::Elt(p, m)) | (AlgNum::Elt(p, m), AlgNum::Rat(a)) => {
                if Zero::is_zero(a) {
                    AlgNum::Rat(qzero())
                } else {
                    AlgNum::Elt(p.scale(a), Arc::clone(m))
                }
            }
            (AlgNum::Elt(p, m), AlgNum::Elt(q, _)) => AlgNum::reduce(p.mul(q), m),
        }
    }

    fn neg(&self) -> Self {
        match self {
            AlgNum::Rat(a) => AlgNum::Rat(-a),
            AlgNum::Elt(p, m) => AlgNum::Elt(p.neg(), Arc::clone(m)),
        }
    }
}

impl EuclideanDomain for AlgNum {
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        if other.is_zero() {
            return (Self::zero(), self.clone());
        }
        (Field::div(self, other), Self::zero())
    }
}

impl Field for AlgNum {
    fn div(&self, other: &Self) -> Self {
        Ring::mul(self, &other.inv())
    }

    /// `1/a` by the extended gcd of `a(t)` and `m(t)`.  The minimal
    /// polynomial is irreducible, so every nonzero element is invertible;
    /// the zero element (never inverted by the algorithms here) gives `0`.
    fn inv(&self) -> Self {
        match self {
            AlgNum::Rat(a) => {
                if Zero::is_zero(a) {
                    AlgNum::Rat(qzero())
                } else {
                    AlgNum::Rat(qone() / a)
                }
            }
            AlgNum::Elt(p, m) => {
                let e = Poly::extended_gcd(p, m);
                if e.gcd.degree() != Some(0) {
                    return AlgNum::Rat(qzero());
                }
                let g = e.gcd.coeff(0);
                AlgNum::reduce(e.x.scale(&(qone() / g)), m)
            }
        }
    }
}

/// A polynomial in `x` over `ℚ(α)`.
pub(crate) type KPoly = GenPoly<AlgNum>;

/// A factorization over `K`: `f = lc · ∏ factorᵢ^multᵢ` with monic factors.
#[derive(Clone, Debug)]
pub(crate) struct KFactorization {
    /// The leading coefficient of the input.
    pub(crate) lc: AlgNum,
    /// Monic irreducible factors with multiplicities.
    pub(crate) factors: Vec<(KPoly, u32)>,
    /// `false` when the factorization of a norm over ℤ was not certified
    /// complete (a factor may then be reducible).
    pub(crate) complete: bool,
}

/// The field `ℚ[t]/(m(t))` for a monic irreducible `m`.
#[derive(Clone, Debug)]
pub(crate) struct NumberField {
    m: Arc<Poly>,
}

impl NumberField {
    /// The field defined by `minpoly` (made monic), which must be
    /// irreducible over ℚ of degree at least 1.
    pub(crate) fn new(minpoly: &Poly) -> Option<Self> {
        if minpoly.degree().unwrap_or(0) == 0 {
            return None;
        }
        Some(NumberField {
            m: Arc::new(minpoly.make_monic()),
        })
    }

    /// `[K : ℚ]`.
    pub(crate) fn degree(&self) -> usize {
        self.m.degree().unwrap_or(0)
    }

    /// The generator `α`.
    pub(crate) fn generator(&self) -> AlgNum {
        AlgNum::reduce(Poly::x(), &self.m)
    }

    /// A polynomial over ℚ as a polynomial over `K`.
    pub(crate) fn lift(&self, f: &Poly) -> KPoly {
        KPoly::from_coeffs(f.coeffs().iter().map(|c| AlgNum::Rat(c.clone())).collect())
    }

    /// The norm `N(f)(x) = res_t(m(t), f(x, t))`, a polynomial over ℚ of
    /// degree `[K : ℚ]·deg f`, by evaluation at `x = 0, …, d·n` and
    /// interpolation (`m` is monic, so `res_t(m, g) = ∏ g(αᵢ)` over the
    /// conjugates `αᵢ`).  `None` beyond [`MAX_NORM_DEGREE`].
    pub(crate) fn norm(&self, f: &KPoly) -> Option<Poly> {
        let n = f.degree()?;
        let total = self.degree().checked_mul(n)?;
        if total > MAX_NORM_DEGREE {
            return None;
        }
        let coeffs: Vec<Poly> = f.coeffs().iter().map(AlgNum::to_poly).collect();
        let mut points: Vec<(i64, Q)> = Vec::with_capacity(total + 1);
        for x0 in 0..=total {
            let x0q = Q::from_integer(BigInt::from(x0));
            let mut g = Poly::zero();
            for c in coeffs.iter().rev() {
                g = g.scale(&x0q).add(c);
            }
            let r = if g.is_zero() {
                qzero()
            } else {
                Poly::resultant(&self.m, &g)
            };
            points.push((x0 as i64, r));
        }
        Some(super::interp::interpolate_at_integers(&points))
    }

    /// The irreducible monic factors over `K` of a monic square-free `f`
    /// (Trager), with a flag: `false` when the factorization of the norm
    /// over ℤ was not certified complete.  `None` when no shift up to
    /// [`MAX_SHIFTS`] gives a square-free norm, or beyond the norm budget.
    pub(crate) fn factor_squarefree(&self, f: &KPoly) -> Option<(Vec<KPoly>, bool)> {
        let n = f.degree()?;
        if n == 0 {
            return Some((vec![], true));
        }
        if n == 1 {
            return Some((vec![f.make_monic()], true));
        }
        let alpha = self.generator();
        let mut found: Option<(AlgNum, KPoly, Poly)> = None;
        for k in 0..MAX_SHIFTS {
            // s = 0, 1, −1, 2, −2, …
            let s = if k % 2 == 1 {
                (k / 2 + 1) as i64
            } else {
                -((k / 2) as i64)
            };
            let shift = Ring::mul(&alpha, &AlgNum::Rat(Q::from_integer(BigInt::from(s))));
            let g = f.taylor_shift(&Ring::neg(&shift));
            let norm = self.norm(&g)?;
            if norm
                .coeffs()
                .iter()
                .any(|c| c.numer().bits().max(c.denom().bits()) > MAX_NORM_COEFF_BITS)
            {
                return None;
            }
            if norm.is_squarefree() == Some(true) {
                found = Some((shift, g, norm));
                break;
            }
        }
        let (shift, g, norm) = found?;
        let (_, nfactors, complete) = super::factor_zassenhaus::factor_zassenhaus_checked(&norm);
        if nfactors.len() <= 1 {
            return Some((vec![f.make_monic()], complete));
        }
        let mut out: Vec<KPoly> = Vec::with_capacity(nfactors.len());
        for (h, _) in &nfactors {
            let q = KPoly::gcd(&g, &self.lift(h));
            if q.degree().unwrap_or(0) == 0 {
                continue;
            }
            out.push(q.taylor_shift(&shift).make_monic());
        }
        let mut back = KPoly::one();
        for q in &out {
            back = back.mul(q);
        }
        if back != f.make_monic() {
            tracing::warn!("numfield: Trager factors do not multiply back; refused");
            return None;
        }
        Some((out, complete))
    }

    /// Factor `f` over `K`: leading coefficient times monic irreducible
    /// factors with multiplicities, verified by multiplying back.  `None`
    /// for a zero or constant `f`, coefficients above the size budget, or
    /// when [`factor_squarefree`](Self::factor_squarefree) gives up.
    pub(crate) fn factor(&self, f: &KPoly) -> Option<KFactorization> {
        if f.degree().unwrap_or(0) == 0 {
            return None;
        }
        if f.coeffs().iter().any(|c| c.digits() > MAX_COEFF_DIGITS) {
            return None;
        }
        let lc = f.leading_coeff()?.clone();
        let monic = f.make_monic();
        let mut factors: Vec<(KPoly, u32)> = Vec::new();
        let mut complete = true;
        for (p, k) in monic.squarefree_factors() {
            let (parts, ok) = self.factor_squarefree(&p)?;
            complete &= ok;
            let k = u32::try_from(k).ok()?;
            for q in parts {
                factors.push((q, k));
            }
        }
        let mut back = KPoly::constant(lc.clone());
        for (q, k) in &factors {
            for _ in 0..*k {
                back = back.mul(q);
            }
        }
        if back != *f {
            tracing::warn!("numfield: factorization does not multiply back; refused");
            return None;
        }
        Some(KFactorization {
            lc,
            factors,
            complete,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Expressions with algebraic coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// The number field generated by some algebraic numbers of an expression,
/// with the expression `α` of its primitive element and every algebraic
/// number met so far expressed in `K`.
#[derive(Clone, Debug)]
pub(crate) struct ExtField {
    pub(crate) field: NumberField,
    /// The primitive element as an expression.
    pub(crate) alpha: ExprId,
    /// Algebraic atoms (`√2`, `i`, `2^(1/3)`, …) and their values in `K`.
    atoms: FxHashMap<ExprId, AlgNum>,
    /// When `α` is a combination of several atoms: a basis of `K` over ℚ
    /// made of products of the atoms (`1, √2, √3, √6`), in which elements
    /// are written out (rather than as powers of `α = √2 + √3 + √6`).
    basis: Option<NiceBasis>,
}

/// A basis of `K` over ℚ given by expressions, with the matrix taking the
/// coefficients of an element in `1, α, …, α^{d−1}` to its coordinates.
#[derive(Clone, Debug)]
struct NiceBasis {
    exprs: Vec<ExprId>,
    inv: Vec<Vec<Q>>,
}

/// The coefficients of `a` in `1, α, …, α^{d−1}`.
fn coords(a: &AlgNum, d: usize) -> Vec<Q> {
    let p = a.to_poly();
    (0..d).map(|k| p.coeff(k)).collect()
}

/// The inverse of a square rational matrix (Gauss–Jordan), `None` if it is
/// singular.
fn invert(m: &[Vec<Q>]) -> Option<Vec<Vec<Q>>> {
    let n = m.len();
    let mut a: Vec<Vec<Q>> = m
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { qone() } else { qzero() }));
            r
        })
        .collect();
    for col in 0..n {
        let piv = (col..n).find(|&r| !Zero::is_zero(&a[r][col]))?;
        a.swap(col, piv);
        let inv = qone() / &a[col][col];
        for v in a[col].iter_mut() {
            *v = &*v * &inv;
        }
        for r in 0..n {
            if r != col && !Zero::is_zero(&a[r][col]) {
                let f = a[r][col].clone();
                let pivot_row = a[col].clone();
                for (v, p) in a[r].iter_mut().zip(&pivot_row) {
                    *v = &*v - &(&f * p);
                }
            }
        }
    }
    Some(a.into_iter().map(|r| r[n..].to_vec()).collect())
}

/// Is `id` a node that the polynomial conversion handles structurally
/// (numbers, sums, products, negations, integer powers)?
fn is_structural(arena: &Arena, id: ExprId) -> bool {
    match arena.node(id) {
        ExprNode::Num(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => true,
        ExprNode::Pow(_, e) => arena.as_num(*e).is_some_and(|q| q.is_integer()),
        _ => false,
    }
}

/// The maximal non-structural subexpressions (generators) of the given
/// expressions, in first-seen order, deduplicated.  An explicit stack.
pub(crate) fn generators(arena: &Arena, exprs: &[ExprId]) -> Vec<ExprId> {
    let mut out: Vec<ExprId> = Vec::new();
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: Vec<ExprId> = exprs.iter().rev().copied().collect();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if is_structural(arena, id) {
            match arena.node(id) {
                ExprNode::Pow(b, _) => stack.push(*b),
                node => {
                    let mut ch = node.children();
                    ch.reverse();
                    stack.extend(ch);
                }
            }
        } else {
            out.push(id);
        }
    }
    out
}

/// Is `id` an algebraic number this module can place in a field: no free
/// symbols, and a minimal polynomial over ℚ of degree at most
/// [`MAX_FIELD_DEGREE`] (`√2`, `i`, `∛2`, the golden ratio, …)?
pub(crate) fn algebraic_minpoly(arena: &mut Arena, id: ExprId) -> Option<Poly> {
    if !walk::free_symbols(arena, id).is_empty() {
        return None;
    }
    let mp = super::algebraic::minimal_polynomial(arena, id)?;
    let d = mp.degree()?;
    if d == 0 || d > MAX_FIELD_DEGREE {
        return None;
    }
    Some(mp.make_monic())
}

impl ExtField {
    /// The field generated by the algebraic numbers `atoms` (at least one),
    /// with a primitive element: one of them when it generates the others,
    /// else a small integer combination `β₁ + c₂β₂ + …`.  `None` when no
    /// candidate of degree at most [`MAX_FIELD_DEGREE`] generates them all,
    /// or an atom is not a recognised algebraic number.
    pub(crate) fn for_atoms(arena: &mut Arena, atoms: &[ExprId]) -> Option<Self> {
        let mut uniq: Vec<(ExprId, Poly)> = Vec::new();
        for &a in atoms {
            if uniq.iter().any(|(b, _)| *b == a) {
                continue;
            }
            let mp = algebraic_minpoly(arena, a)?;
            uniq.push((a, mp));
        }
        if uniq.is_empty() {
            return None;
        }
        // Candidates: each atom (largest degree first), then combinations.
        let mut order: Vec<usize> = (0..uniq.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(uniq[i].1.degree().unwrap_or(0)));
        let mut candidates: Vec<(ExprId, Option<Poly>)> = order
            .iter()
            .map(|&i| (uniq[i].0, Some(uniq[i].1.clone())))
            .collect();
        if uniq.len() > 1 {
            for c in 1..=3i64 {
                let mut terms = Vec::with_capacity(uniq.len());
                for (j, (a, _)) in uniq.iter().enumerate() {
                    let k = arena.int(1 + (j as i64) * (c - 1));
                    terms.push(arena.mul(&[k, *a]));
                }
                candidates.push((arena.add(&terms), None));
            }
        }
        'cand: for (alpha, mp) in candidates {
            let mp = match mp {
                Some(mp) => mp,
                None => match algebraic_minpoly(arena, alpha) {
                    Some(mp) => mp,
                    None => continue,
                },
            };
            let Some(field) = NumberField::new(&mp) else {
                continue;
            };
            let mut ef = ExtField {
                field,
                alpha,
                atoms: FxHashMap::default(),
                basis: None,
            };
            let generator = ef.field.generator();
            ef.atoms.insert(alpha, generator);
            for (a, amp) in &uniq {
                if ef.atoms.contains_key(a) {
                    continue;
                }
                match ef.locate(arena, *a, amp) {
                    Some(v) => {
                        ef.atoms.insert(*a, v);
                    }
                    None => continue 'cand,
                }
            }
            if !uniq.iter().any(|(a, _)| *a == alpha) {
                ef.basis = ef.nice_basis(arena, &uniq);
            }
            return Some(ef);
        }
        None
    }

    /// Products `∏ βⱼ^eⱼ` (`0 ≤ eⱼ < deg βⱼ`, by increasing total degree)
    /// of the atoms, the first `[K : ℚ]` of them that are linearly
    /// independent over ℚ; `None` if they do not span `K` or there are too
    /// many candidates.
    fn nice_basis(&self, arena: &mut Arena, atoms: &[(ExprId, Poly)]) -> Option<NiceBasis> {
        let d = self.field.degree();
        let degs: Vec<usize> = atoms.iter().map(|(_, m)| m.degree().unwrap_or(1)).collect();
        let total: usize = degs.iter().try_fold(1usize, |acc, &k| acc.checked_mul(k))?;
        if total > 64 {
            return None;
        }
        let mut monos: Vec<Vec<usize>> = (0..total)
            .map(|mut idx| {
                degs.iter()
                    .map(|&k| {
                        let e = idx % k;
                        idx /= k;
                        e
                    })
                    .collect()
            })
            .collect();
        monos.sort_by_key(|e| (e.iter().sum::<usize>(), e.clone()));
        let mut exprs: Vec<ExprId> = Vec::with_capacity(d);
        let mut cols: Vec<Vec<Q>> = Vec::with_capacity(d);
        // Row-echelon copy of the chosen vectors, for the independence test.
        let mut echelon: Vec<(usize, Vec<Q>)> = Vec::new();
        for e in &monos {
            if exprs.len() == d {
                break;
            }
            let mut value = AlgNum::one();
            for ((a, _), &k) in atoms.iter().zip(e) {
                let v = self.atoms.get(a)?;
                value = Ring::mul(&value, &v.pow_usize(k));
            }
            let vec = coords(&value, d);
            let mut red = vec.clone();
            for (p, row) in &echelon {
                if !Zero::is_zero(&red[*p]) {
                    let f = red[*p].clone() / &row[*p];
                    for (x, y) in red.iter_mut().zip(row) {
                        *x = &*x - &(&f * y);
                    }
                }
            }
            let Some(p) = red.iter().position(|x| !Zero::is_zero(x)) else {
                continue;
            };
            echelon.push((p, red));
            let mut parts: Vec<ExprId> = Vec::new();
            for ((a, _), &k) in atoms.iter().zip(e) {
                match k {
                    0 => {}
                    1 => parts.push(*a),
                    _ => {
                        let ke = arena.int(k as i64);
                        parts.push(arena.pow(*a, ke));
                    }
                }
            }
            exprs.push(match parts.len() {
                0 => arena.one,
                1 => parts[0],
                _ => arena.mul(&parts),
            });
            cols.push(vec);
        }
        if exprs.len() != d {
            return None;
        }
        // M has the chosen vectors as columns.
        let m: Vec<Vec<Q>> = (0..d)
            .map(|r| cols.iter().map(|c| c[r].clone()).collect())
            .collect();
        let inv = invert(&m)?;
        Some(NiceBasis { exprs, inv })
    }

    /// `β` (with minimal polynomial `mp`) as an element of `K`: the root of
    /// `mp` in `K` whose expression differs from `β` by an exact zero
    /// (`exact_is_zero`), `None` if `β ∉ K` or nothing is certified.
    fn locate(&self, arena: &mut Arena, beta: ExprId, mp: &Poly) -> Option<AlgNum> {
        let d = mp.degree()?;
        if d == 1 {
            return Some(AlgNum::Rat(-mp.coeff(0) / mp.coeff(1)));
        }
        if !self.field.degree().is_multiple_of(d) {
            return None;
        }
        let fac = self.field.factor(&self.field.lift(mp))?;
        for (q, _) in &fac.factors {
            if q.degree() != Some(1) {
                continue;
            }
            let root = Ring::neg(&q.coeff(0));
            let root_expr = self.element_to_expr(arena, &root);
            let diff = {
                let neg = arena.neg(root_expr);
                arena.add(&[beta, neg])
            };
            if super::algebraic::exact_is_zero(arena, diff) == Some(true) {
                return Some(root);
            }
        }
        None
    }

    /// The value in `K` of an algebraic atom, located (and remembered) on
    /// first use.
    fn atom_value(&mut self, arena: &mut Arena, atom: ExprId) -> Option<AlgNum> {
        if let Some(v) = self.atoms.get(&atom) {
            return Some(v.clone());
        }
        let mp = algebraic_minpoly(arena, atom)?;
        let v = self.locate(arena, atom, &mp)?;
        self.atoms.insert(atom, v.clone());
        Some(v)
    }

    /// Convert `expr`, a polynomial in `var` whose coefficients are built
    /// from rationals and algebraic numbers of `K`, to a [`KPoly`].  `None`
    /// for anything else (a symbol other than `var`, a function, a negative
    /// power of something that depends on `var` or is 0, …).  Iterative
    /// post-order walk.
    pub(crate) fn kpoly_of(
        &mut self,
        arena: &mut Arena,
        expr: ExprId,
        var: ExprId,
    ) -> Option<KPoly> {
        let order = walk::post_order_ids(arena, expr);
        let mut cache: FxHashMap<ExprId, KPoly> = FxHashMap::default();
        for &id in &order {
            if cache.contains_key(&id) {
                continue;
            }
            let value = if id == var {
                KPoly::x()
            } else if !is_structural(arena, id) {
                // Children of generators are visited too; skip them unless
                // they are reached structurally.
                match self.atom_value(arena, id) {
                    Some(v) => KPoly::constant(v),
                    None => continue,
                }
            } else {
                let node = arena.node(id).clone();
                match node {
                    ExprNode::Num(nid) => KPoly::constant(AlgNum::Rat(arena.num(nid).clone())),
                    ExprNode::Add(ch) => {
                        let mut acc = KPoly::zero();
                        for c in ch.iter() {
                            acc = acc.add(cache.get(c)?);
                        }
                        acc
                    }
                    ExprNode::Mul(ch) => {
                        let mut acc = KPoly::one();
                        for c in ch.iter() {
                            acc = acc.mul(cache.get(c)?);
                            if acc.degree().unwrap_or(0) > MAX_KPOLY_DEGREE {
                                return None;
                            }
                        }
                        acc
                    }
                    ExprNode::Neg(inner) => cache.get(&inner)?.neg(),
                    ExprNode::Pow(base, e) => {
                        let k = arena.as_num(e)?.to_integer();
                        let b = cache.get(&base)?;
                        if k.is_negative() {
                            // Only a nonzero constant may be inverted.
                            if b.degree() != Some(0) {
                                return None;
                            }
                            let inv = b.coeff(0).inv();
                            let n = usize::try_from(-k).ok()?;
                            if n > MAX_KPOLY_DEGREE {
                                return None;
                            }
                            KPoly::constant(inv.pow_usize(n))
                        } else {
                            let n = usize::try_from(k).ok()?;
                            let deg = b.degree().unwrap_or(0);
                            if deg.checked_mul(n)? > MAX_KPOLY_DEGREE
                                || (deg == 0 && n > MAX_KPOLY_DEGREE)
                            {
                                return None;
                            }
                            b.pow(n)
                        }
                    }
                    _ => return None,
                }
            };
            if value.coeffs().iter().any(|c| c.digits() > MAX_COEFF_DIGITS) {
                return None;
            }
            cache.insert(id, value);
        }
        cache.remove(&expr)
    }

    /// The rational coordinates of `a` in the basis its expression is
    /// written in (the products of atoms of [`NiceBasis`], else the powers
    /// of `α`).
    fn coordinates(&self, a: &AlgNum) -> Vec<Q> {
        let d = self.field.degree();
        let v = coords(a, d);
        match &self.basis {
            Some(b) => b
                .inv
                .iter()
                .map(|row| row.iter().zip(&v).map(|(x, y)| x * y).sum())
                .collect(),
            None => v,
        }
    }

    /// The positive rational `s` for which `s·f` has integer coordinates
    /// (in the basis of [`element_to_expr`](Self::element_to_expr)) with no
    /// common factor: `(2x − √2)/4` for `x − √2/2` is `s = 2`.  `1` for
    /// the zero polynomial.
    pub(crate) fn primitive_scale(&self, f: &KPoly) -> Q {
        let cs: Vec<Q> = f
            .coeffs()
            .iter()
            .flat_map(|c| self.coordinates(c))
            .collect();
        let mut l: BigInt = One::one();
        for q in &cs {
            l = num_integer::Integer::lcm(&l, q.denom());
        }
        let lq = Q::from_integer(l.clone());
        let mut g: BigInt = Zero::zero();
        for q in &cs {
            let n = q * &lq;
            g = num_integer::Integer::gcd(&g, n.numer());
        }
        if Zero::is_zero(&g) {
            return qone();
        }
        Q::new(l, g)
    }

    /// The expression `Σ cⱼ αʲ` of an element.
    pub(crate) fn element_to_expr(&self, arena: &mut Arena, a: &AlgNum) -> ExprId {
        match a {
            AlgNum::Rat(q) => {
                let nid = arena.intern_num(q.clone());
                arena.intern(ExprNode::Num(nid))
            }
            AlgNum::Elt(_, _) if self.basis.is_some() => {
                let Some(b) = &self.basis else {
                    return arena.zero;
                };
                let v = coords(a, self.field.degree());
                let mut terms = Vec::with_capacity(b.exprs.len());
                for (row, &e) in b.inv.iter().zip(&b.exprs) {
                    let c: Q = row.iter().zip(&v).map(|(x, y)| x * y).sum();
                    if Zero::is_zero(&c) {
                        continue;
                    }
                    let cid = arena.num_ratio(c);
                    terms.push(arena.mul(&[cid, e]));
                }
                match terms.len() {
                    0 => arena.zero,
                    1 => terms[0],
                    _ => arena.add(&terms),
                }
            }
            AlgNum::Elt(p, _) => {
                let mut terms = Vec::with_capacity(p.coeffs().len());
                for (j, c) in p.coeffs().iter().enumerate() {
                    if Zero::is_zero(c) {
                        continue;
                    }
                    let nid = arena.intern_num(c.clone());
                    let cid = arena.intern(ExprNode::Num(nid));
                    let pw = match j {
                        0 => arena.one,
                        1 => self.alpha,
                        _ => {
                            let e = arena.int(j as i64);
                            arena.pow(self.alpha, e)
                        }
                    };
                    terms.push(arena.mul(&[cid, pw]));
                }
                match terms.len() {
                    0 => arena.zero,
                    1 => terms[0],
                    _ => arena.add(&terms),
                }
            }
        }
    }

    /// The expression `Σ cₖ(α) varᵏ` of a polynomial over `K`.
    pub(crate) fn kpoly_to_expr(&self, arena: &mut Arena, f: &KPoly, var: ExprId) -> ExprId {
        let mut terms = Vec::with_capacity(f.coeffs().len());
        for (k, c) in f.coeffs().iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let cid = self.element_to_expr(arena, c);
            let pw = match k {
                0 => arena.one,
                1 => var,
                _ => {
                    let e = arena.int(k as i64);
                    arena.pow(var, e)
                }
            };
            terms.push(arena.mul(&[cid, pw]));
        }
        match terms.len() {
            0 => arena.zero,
            1 => terms[0],
            _ => arena.add(&terms),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64) -> Q {
        Q::from_integer(BigInt::from(n))
    }

    fn poly(cs: &[i64]) -> Poly {
        Poly::from_coeffs(cs.iter().map(|&c| q(c)).collect())
    }

    #[test]
    fn norm_of_x_minus_sqrt2_is_x2_minus_2() {
        let k = NumberField::new(&poly(&[-2, 0, 1])).unwrap();
        let a = k.generator();
        let f = KPoly::from_coeffs(vec![Ring::neg(&a), AlgNum::one()]);
        assert_eq!(k.norm(&f).unwrap(), poly(&[-2, 0, 1]));
    }

    #[test]
    fn x2_minus_2_splits_over_q_sqrt2() {
        let k = NumberField::new(&poly(&[-2, 0, 1])).unwrap();
        let f = k.lift(&poly(&[-2, 0, 1]));
        let fac = k.factor(&f).unwrap();
        assert_eq!(fac.factors.len(), 2);
        assert!(fac.complete);
        assert!(
            fac.factors
                .iter()
                .all(|(g, m)| g.degree() == Some(1) && *m == 1)
        );
    }

    #[test]
    fn x4_plus_1_over_q_i_has_two_quadratic_factors() {
        let k = NumberField::new(&poly(&[1, 0, 1])).unwrap();
        let f = k.lift(&poly(&[1, 0, 0, 0, 1]));
        let fac = k.factor(&f).unwrap();
        assert_eq!(fac.factors.len(), 2);
        assert!(fac.factors.iter().all(|(g, _)| g.degree() == Some(2)));
    }

    #[test]
    fn repeated_factor_over_q_sqrt2() {
        // (x − √2)² = x² − 2√2 x + 2
        let k = NumberField::new(&poly(&[-2, 0, 1])).unwrap();
        let a = k.generator();
        let two_a = Ring::mul(&a, &AlgNum::Rat(q(-2)));
        let f = KPoly::from_coeffs(vec![AlgNum::Rat(q(2)), two_a, AlgNum::one()]);
        let fac = k.factor(&f).unwrap();
        assert_eq!(fac.factors.len(), 1);
        assert_eq!(fac.factors[0].1, 2);
    }

    #[test]
    fn inverse_in_q_cbrt2() {
        let k = NumberField::new(&poly(&[-2, 0, 0, 1])).unwrap();
        let a = k.generator();
        let b = Ring::add(&a, &AlgNum::one());
        let prod = Ring::mul(&b, &b.inv());
        assert!(prod.is_one());
    }
}
