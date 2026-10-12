//! Rational functions in one variable over `ℚ(parameters)` and over an
//! algebraic number field: the coefficient field [`PFrac`] of rational
//! functions in the parameters, the bridge from expressions, and
//! [`cancel_full`], the public `Ex::cancel`.
//!
//! An expression rational in `var` is read, after [`together_deep`]
//! (sums of fractions combined), as `P/Q` with `P, Q ∈ ℚ[var, y₁, …, yₖ]`,
//! the `yⱼ` being its other generators: free symbols, and opaque
//! subexpressions free of `var` (`π`, `√2`, `sin a`).  [`PFrac`] is the
//! field `ℚ(y₁, …, yₖ)`, so that [`GenPoly<PFrac>`] has division with
//! remainder and the extended gcd in `var` (used by partial fractions).
//!
//! `cancel_full` keeps the rational-coefficient [`cancel`] where it applies
//! and otherwise works, as SymPy's `cancel` does, over the field of
//! fractions of the parameters (the gcd of `P` and `Q` in
//! `ℤ[var, y]`), or over `ℚ(α)` when the only other generators are
//! algebraic numbers (`√2`, `i`, …; the gcd over the number field,
//! [`numfield`]), which SymPy does with `extension=`.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

use super::generic::GenPoly;
use super::multipoly::{GrevLex, Lex, MultiPoly};
use super::numfield::{self, AlgNum, ExtField, KPoly};
use super::polybridge::{cancel, fraction_parts, multipoly_to_expr, together_deep};
use super::traits::{EuclideanDomain, Field, Ring};
use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;

type Q = Ratio<BigInt>;
type MPoly = MultiPoly<GrevLex>;

/// Largest number of terms of any polynomial built from an expression.
const MAX_TERMS: usize = 20_000;

/// Largest total degree of any polynomial built from an expression.
const MAX_DEGREE: u32 = 400;

fn qzero() -> Q {
    <Q as Zero>::zero()
}

fn qone() -> Q {
    <Q as One>::one()
}

fn bigint_one() -> BigInt {
    <BigInt as One>::one()
}

// ═══════════════════════════════════════════════════════════════════════════
// The field ℚ(y₁, …, yₖ)
// ═══════════════════════════════════════════════════════════════════════════

/// An element of `ℚ(y₁, …, yₖ)`: a rational number, or `n/d` with `n, d`
/// coprime polynomials (`d` integer-primitive with a positive leading
/// coefficient), not both constant.  All polynomials of one computation
/// have the same number of variables.
#[derive(Clone, Debug)]
pub(crate) enum PFrac {
    /// A rational constant.
    Rat(Q),
    /// `numerator / denominator`.
    Frac(MPoly, MPoly),
}

impl PFrac {
    /// The element `p` (a polynomial).
    pub(crate) fn from_poly(p: &MPoly) -> Self {
        match p.as_constant() {
            Some(c) => PFrac::Rat(c),
            None => PFrac::Frac(p.clone(), MPoly::from_int(p.num_vars(), 1)),
        }
    }

    /// `n/d` normalised; `d` must be nonzero.  `coprime`: `n` and `d` are
    /// known to have no common factor of positive degree.
    fn make(n: MPoly, d: MPoly, coprime: bool) -> Self {
        if n.is_zero() {
            return PFrac::Rat(qzero());
        }
        let (n, d) = if coprime {
            (n, d)
        } else {
            let g = MPoly::gcd(&n, &d);
            if g.total_degree().unwrap_or(0) > 0 {
                match (n.div_exact(&g), d.div_exact(&g)) {
                    (Some(a), Some(b)) => (a, b),
                    _ => (n, d),
                }
            } else {
                (n, d)
            }
        };
        // Integer-primitive denominator with a positive leading coefficient.
        let (dl, dz) = d.clear_denominators();
        let c = dz.integer_content();
        let neg = dz.leading_coeff().is_some_and(Signed::is_negative);
        let c = if neg { -c } else { c };
        if Zero::is_zero(&c) {
            return PFrac::Rat(qzero());
        }
        let d = dz.scale(&Ratio::new(bigint_one(), c.clone()));
        let n = n.scale(&Ratio::new(dl, c));
        match (n.as_constant(), d.as_constant()) {
            (Some(a), Some(b)) => PFrac::Rat(a / b),
            _ => PFrac::Frac(n, d),
        }
    }

    /// Numerator and denominator as polynomials in `nv` variables.
    pub(crate) fn parts(&self, nv: usize) -> PolyFraction {
        match self {
            PFrac::Rat(q) => PolyFraction {
                numer: MPoly::constant(nv, q.clone()),
                denom: MPoly::from_int(nv, 1),
            },
            PFrac::Frac(n, d) => PolyFraction {
                numer: n.clone(),
                denom: d.clone(),
            },
        }
    }

    fn size_ok(&self) -> bool {
        match self {
            PFrac::Rat(_) => true,
            PFrac::Frac(n, d) => within_budget(n) && within_budget(d),
        }
    }
}

impl PartialEq for PFrac {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PFrac::Rat(a), PFrac::Rat(b)) => a == b,
            (PFrac::Frac(n1, d1), PFrac::Frac(n2, d2)) => n1 == n2 && d1 == d2,
            _ => false,
        }
    }
}

impl Ring for PFrac {
    fn zero() -> Self {
        PFrac::Rat(qzero())
    }

    fn one() -> Self {
        PFrac::Rat(qone())
    }

    fn is_zero(&self) -> bool {
        matches!(self, PFrac::Rat(q) if Zero::is_zero(q))
    }

    fn is_one(&self) -> bool {
        matches!(self, PFrac::Rat(q) if One::is_one(q))
    }

    fn add(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (PFrac::Rat(a), PFrac::Rat(b)) => PFrac::Rat(Ring::add(a, b)),
            (PFrac::Rat(a), PFrac::Frac(n, d)) | (PFrac::Frac(n, d), PFrac::Rat(a)) => {
                // gcd(n + a·d, d) = gcd(n, d) = 1.
                PFrac::make(n.add(&d.scale(a)), d.clone(), true)
            }
            (PFrac::Frac(n1, d1), PFrac::Frac(n2, d2)) => {
                if d1 == d2 {
                    PFrac::make(n1.add(n2), d1.clone(), false)
                } else {
                    PFrac::make(n1.mul(d2).add(&n2.mul(d1)), d1.mul(d2), false)
                }
            }
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        Ring::add(self, &Ring::neg(rhs))
    }

    fn mul(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (PFrac::Rat(a), PFrac::Rat(b)) => PFrac::Rat(Ring::mul(a, b)),
            (PFrac::Rat(a), PFrac::Frac(n, d)) | (PFrac::Frac(n, d), PFrac::Rat(a)) => {
                if Zero::is_zero(a) {
                    PFrac::Rat(qzero())
                } else {
                    PFrac::make(n.scale(a), d.clone(), true)
                }
            }
            (PFrac::Frac(n1, d1), PFrac::Frac(n2, d2)) => {
                PFrac::make(n1.mul(n2), d1.mul(d2), false)
            }
        }
    }

    fn neg(&self) -> Self {
        match self {
            PFrac::Rat(a) => PFrac::Rat(-a),
            PFrac::Frac(n, d) => PFrac::Frac(n.neg(), d.clone()),
        }
    }
}

impl EuclideanDomain for PFrac {
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        if other.is_zero() {
            return (Self::zero(), self.clone());
        }
        (Field::div(self, other), Self::zero())
    }
}

impl Field for PFrac {
    fn div(&self, other: &Self) -> Self {
        Ring::mul(self, &other.inv())
    }

    /// `1/a`; the zero element (never inverted here) gives `0`.
    fn inv(&self) -> Self {
        match self {
            PFrac::Rat(a) => {
                if Zero::is_zero(a) {
                    PFrac::Rat(qzero())
                } else {
                    PFrac::Rat(qone() / a)
                }
            }
            PFrac::Frac(n, d) => PFrac::make(d.clone(), n.clone(), true),
        }
    }
}

/// A polynomial in the main variable over `ℚ(y)`.
pub(crate) type PPoly = GenPoly<PFrac>;

/// A quotient of two polynomials over the generators.
pub(crate) struct PolyFraction {
    pub(crate) numer: MPoly,
    pub(crate) denom: MPoly,
}

fn within_budget(p: &MPoly) -> bool {
    p.num_terms() <= MAX_TERMS && p.total_degree().unwrap_or(0) <= MAX_DEGREE
}

/// Largest product of the term counts of two factors multiplied (the cost
/// of the product, whatever cancels in it).
const MAX_PRODUCT_WORK: usize = 400_000;

/// `a·b` when its cost and its size are within the budget.
fn mul_budget(a: &MPoly, b: &MPoly) -> Option<MPoly> {
    if a.num_terms().saturating_mul(b.num_terms()) > MAX_PRODUCT_WORK {
        return None;
    }
    let p = a.mul(b);
    within_budget(&p).then_some(p)
}

// ═══════════════════════════════════════════════════════════════════════════
// Expressions ↔ polynomials over the generators
// ═══════════════════════════════════════════════════════════════════════════

/// A rational function `numer/denom` of an expression, polynomials in
/// `vars` with `vars[0]` the main variable.
pub(crate) struct RationalInVar {
    pub(crate) vars: Vec<ExprId>,
    pub(crate) numer: MPoly,
    pub(crate) denom: MPoly,
}

/// Which generators besides the main variable are accepted.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Params {
    /// Free symbols other than the main variable only.
    Symbols,
    /// Every generator, also one that depends on the main variable
    /// (`sin x`, `√x`): an independent indeterminate, as in SymPy's
    /// `cancel`.  Only for computations that divide by a gcd and nothing
    /// else, which stay valid whatever relations the generators satisfy.
    Any,
}

/// `expr` (already combined by [`together_deep`]) as `P/Q` over its
/// generators, the main variable first; `None` if a generator is not
/// accepted by `params`, there is no other generator, `Q` is zero, or a
/// polynomial exceeds the size budget.
pub(crate) fn rational_in_var(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    params: Params,
) -> Option<RationalInVar> {
    let (n, d) = fraction_parts(arena, expr);
    let gens = numfield::generators(arena, &[n, d]);
    let mut others: Vec<ExprId> = Vec::new();
    for g in gens {
        if g == var {
            continue;
        }
        let ok = match params {
            Params::Symbols => matches!(arena.node(g), ExprNode::Symbol(_)),
            Params::Any => true,
        };
        if !ok {
            return None;
        }
        others.push(g);
    }
    if others.is_empty() {
        return None;
    }
    others.sort_by(|a, b| {
        arena
            .sort_key(*a)
            .cmp(arena.sort_key(*b))
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut vars = vec![var];
    vars.extend(others);
    let numer = to_multipoly(arena, n, &vars)?;
    let denom = to_multipoly(arena, d, &vars)?;
    if denom.is_zero() {
        return None;
    }
    Some(RationalInVar { vars, numer, denom })
}

/// A polynomial expression in `vars` (structural nodes only, every leaf a
/// number or one of `vars`) as a [`MultiPoly`], within the size budget.
/// Iterative post-order walk.
pub(crate) fn to_multipoly(arena: &Arena, expr: ExprId, vars: &[ExprId]) -> Option<MPoly> {
    let nv = vars.len();
    let index: FxHashMap<ExprId, usize> = vars.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut cache: FxHashMap<ExprId, MPoly> = FxHashMap::default();
    for id in walk::post_order_ids(arena, expr) {
        if let Some(&i) = index.get(&id) {
            cache.insert(id, MPoly::var(nv, i));
            continue;
        }
        let p = match arena.node(id) {
            ExprNode::Num(nid) => MPoly::constant(nv, arena.num(*nid).clone()),
            ExprNode::Add(ch) => {
                let mut acc = MPoly::zero(nv);
                for c in ch.iter() {
                    acc = acc.add(cache.get(c)?);
                    if !within_budget(&acc) {
                        return None;
                    }
                }
                acc
            }
            ExprNode::Mul(ch) => {
                let mut acc = MPoly::from_int(nv, 1);
                for c in ch.iter() {
                    acc = mul_budget(&acc, cache.get(c)?)?;
                }
                acc
            }
            ExprNode::Neg(inner) => cache.get(inner)?.neg(),
            ExprNode::Pow(base, e) => {
                let k = arena.as_num(*e)?;
                if !k.is_integer() || k.is_negative() {
                    return None;
                }
                let n: u32 = k.to_integer().try_into().ok()?;
                let b = cache.get(base)?;
                let bd = b.total_degree().unwrap_or(0);
                if n > MAX_DEGREE || bd.checked_mul(n)? > MAX_DEGREE {
                    return None;
                }
                pow_budget(b, n)?
            }
            // Children of other nodes (inside generators) are skipped; a
            // parent that needs them fails above.
            _ => continue,
        };
        cache.insert(id, p);
    }
    cache.remove(&expr)
}

/// `b^n` by repeated squaring within the budget.
fn pow_budget(b: &MPoly, mut n: u32) -> Option<MPoly> {
    let mut result = MPoly::from_int(b.num_vars(), 1);
    let mut sq = b.clone();
    while n > 0 {
        if n & 1 == 1 {
            result = mul_budget(&result, &sq)?;
        }
        n >>= 1;
        if n > 0 {
            sq = mul_budget(&sq, &sq)?;
        }
    }
    Some(result)
}

/// A polynomial over `[var, y…]` as a polynomial in `var` (index 0) over
/// `ℚ(y)`; the coefficients keep all variables (exponent 0 in `var`).
pub(crate) fn to_ppoly(p: &MPoly) -> PPoly {
    let deg = p.degree_in(0);
    let coeffs = (0..=deg)
        .map(|k| PFrac::from_poly(&p.coeff_in(0, k)))
        .collect();
    PPoly::from_coeffs(coeffs)
}

/// Is every coefficient of `f` within the size budget?
pub(crate) fn ppoly_size_ok(f: &PPoly) -> bool {
    f.coeffs().iter().all(PFrac::size_ok)
}

/// `f` written as `N/c` with `N ∈ ℚ[var, y]` and `c ∈ ℚ[y]` (the lcm of
/// the coefficient denominators), both as polynomials in `nv` variables.
pub(crate) fn ppoly_over_common_denominator(f: &PPoly, nv: usize) -> PolyFraction {
    let mut c = MPoly::from_int(nv, 1);
    for co in f.coeffs() {
        let d = co.parts(nv).denom;
        if d.total_degree().unwrap_or(0) > 0 {
            c = MPoly::lcm(&c, &d);
        }
    }
    // Rational denominators: the lcm of all of them.
    let mut l = bigint_one();
    for co in f.coeffs() {
        for (_, q) in co.parts(nv).numer.terms() {
            l = l.lcm(q.denom());
        }
    }
    let mut num = MPoly::zero(nv);
    for (k, co) in f.coeffs().iter().enumerate() {
        let PolyFraction { numer: n, denom: d } = co.parts(nv);
        let Some(m) = c.div_exact(&d) else {
            continue;
        };
        let mut e = vec![0u32; nv];
        e[0] = k as u32;
        num = num.add(&n.mul(&m).mul_monomial(&qone(), &e));
    }
    if !One::is_one(&l) {
        let lq = Q::from_integer(l);
        num = num.scale(&lq);
        c = c.scale(&lq);
    }
    PolyFraction {
        numer: num,
        denom: c,
    }
}

/// Integer-primitive `p/q` with the lexicographically leading coefficient
/// of `q` positive (`q` nonzero).
fn normalize_pair(p: &MPoly, q: &MPoly) -> PolyFraction {
    let (dp, pz) = p.clear_denominators();
    let (dq, qz) = q.clear_denominators();
    let mut p = pz.scale(&Q::from_integer(dq));
    let mut q = qz.scale(&Q::from_integer(dp));
    let c = p.integer_content().gcd(&q.integer_content());
    if !Zero::is_zero(&c) && !One::is_one(&c) {
        let inv = Ratio::new(bigint_one(), c);
        p = p.scale(&inv);
        q = q.scale(&inv);
    }
    let lex: MultiPoly<Lex> = q.convert_order();
    if lex.leading_coeff().is_some_and(Signed::is_negative) {
        p = p.neg();
        q = q.neg();
    }
    PolyFraction { numer: p, denom: q }
}

/// The expression of `p` as a product of its irreducible factors over ℤ
/// (`(a + 1)²` rather than `a² + 2a + 1`), or expanded when it does not
/// factor or the factorization gives up.
pub(crate) fn factored_expr(arena: &mut Arena, p: &MPoly, vars: &[ExprId]) -> ExprId {
    let Some((content, factors)) = super::factor_zassenhaus::factor_multivariate_many(p) else {
        return multipoly_to_expr(arena, p, vars);
    };
    if factors.len() <= 1 && factors.iter().all(|(_, m)| *m == 1) {
        return multipoly_to_expr(arena, p, vars);
    }
    let mut parts: Vec<ExprId> = Vec::with_capacity(factors.len() + 1);
    if !One::is_one(&content) {
        parts.push(arena.num_ratio(content));
    }
    for (f, m) in &factors {
        let fe = multipoly_to_expr(arena, f, vars);
        parts.push(if *m == 1 {
            fe
        } else {
            let e = arena.int(i64::from(*m));
            arena.pow(fe, e)
        });
    }
    match parts.len() {
        0 => arena.one,
        1 => parts[0],
        _ => arena.mul(&parts),
    }
}

/// The expression `p/q` (`p` alone when `q` is a constant, divided out).
pub(crate) fn quotient_expr(arena: &mut Arena, p: &MPoly, q: &MPoly, vars: &[ExprId]) -> ExprId {
    if let Some(c) = q.as_constant() {
        let pc = p.scale(&(qone() / c));
        return multipoly_to_expr(arena, &pc, vars);
    }
    let pe = multipoly_to_expr(arena, p, vars);
    let qe = multipoly_to_expr(arena, q, vars);
    arena.div(pe, qe)
}

// ═══════════════════════════════════════════════════════════════════════════
// The field ℚ(y₁, …, yₖ)(α)
// ═══════════════════════════════════════════════════════════════════════════

/// An element of `ℚ(y)(α) = ℚ(y)[t]/(m(t))`, `m` the minimal polynomial
/// of an algebraic number `α` over ℚ (irreducible over `ℚ(y)` too: the `y`
/// are transcendental): an element of `ℚ(y)`, or a polynomial in `t` over
/// `ℚ(y)` of degree `1 ≤ deg < deg m` reduced modulo the `m` it carries, as
/// [`AlgNum`] is over ℚ.
#[derive(Clone, Debug)]
pub(crate) enum PAlg {
    /// An element of `ℚ(y)`.
    Base(PFrac),
    /// `p(α)` with `p` of degree at least 1, reduced modulo `m`.
    Elt(PPoly, std::sync::Arc<PPoly>),
}

impl PAlg {
    /// `p(α)` reduced modulo `m`.
    fn reduce(p: PPoly, m: &std::sync::Arc<PPoly>) -> Self {
        let r = if p.degree().unwrap_or(0) >= m.degree().unwrap_or(0) {
            p.rem(m)
        } else {
            p
        };
        match r.degree() {
            None => PAlg::Base(PFrac::zero()),
            Some(0) => PAlg::Base(r.coeff(0)),
            Some(_) => PAlg::Elt(r, std::sync::Arc::clone(m)),
        }
    }

    /// The coefficients in `ℚ(y)` of `1, α, …, α^{d−1}`.
    fn coefficients(&self, d: usize) -> Vec<PFrac> {
        match self {
            PAlg::Base(c) => {
                let mut v = vec![PFrac::zero(); d.max(1)];
                v[0] = c.clone();
                v
            }
            PAlg::Elt(p, _) => (0..d.max(1)).map(|j| p.coeff(j)).collect(),
        }
    }

    fn size_ok(&self) -> bool {
        match self {
            PAlg::Base(c) => c.size_ok(),
            PAlg::Elt(p, _) => ppoly_size_ok(p),
        }
    }
}

impl PartialEq for PAlg {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PAlg::Base(a), PAlg::Base(b)) => a == b,
            (PAlg::Elt(p, _), PAlg::Elt(q, _)) => p == q,
            _ => false,
        }
    }
}

impl Ring for PAlg {
    fn zero() -> Self {
        PAlg::Base(PFrac::zero())
    }

    fn one() -> Self {
        PAlg::Base(PFrac::one())
    }

    fn is_zero(&self) -> bool {
        matches!(self, PAlg::Base(c) if c.is_zero())
    }

    fn is_one(&self) -> bool {
        matches!(self, PAlg::Base(c) if c.is_one())
    }

    fn add(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (PAlg::Base(a), PAlg::Base(b)) => PAlg::Base(Ring::add(a, b)),
            (PAlg::Base(a), PAlg::Elt(p, m)) | (PAlg::Elt(p, m), PAlg::Base(a)) => {
                PAlg::Elt(p.add(&PPoly::constant(a.clone())), std::sync::Arc::clone(m))
            }
            (PAlg::Elt(p, m), PAlg::Elt(q, _)) => PAlg::reduce(p.add(q), m),
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        Ring::add(self, &Ring::neg(rhs))
    }

    fn mul(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (PAlg::Base(a), PAlg::Base(b)) => PAlg::Base(Ring::mul(a, b)),
            (PAlg::Base(a), PAlg::Elt(p, m)) | (PAlg::Elt(p, m), PAlg::Base(a)) => {
                if a.is_zero() {
                    PAlg::zero()
                } else {
                    PAlg::Elt(p.scale(a), std::sync::Arc::clone(m))
                }
            }
            (PAlg::Elt(p, m), PAlg::Elt(q, _)) => PAlg::reduce(p.mul(q), m),
        }
    }

    fn neg(&self) -> Self {
        match self {
            PAlg::Base(a) => PAlg::Base(Ring::neg(a)),
            PAlg::Elt(p, m) => PAlg::Elt(p.neg(), std::sync::Arc::clone(m)),
        }
    }
}

impl EuclideanDomain for PAlg {
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        if other.is_zero() {
            return (Self::zero(), self.clone());
        }
        (Field::div(self, other), Self::zero())
    }
}

impl Field for PAlg {
    fn div(&self, other: &Self) -> Self {
        Ring::mul(self, &other.inv())
    }

    /// `1/a` by the extended gcd of `a(t)` and `m(t)` over `ℚ(y)`; the zero
    /// element (never inverted here) gives `0`.
    fn inv(&self) -> Self {
        match self {
            PAlg::Base(a) => PAlg::Base(a.inv()),
            PAlg::Elt(p, m) => {
                let e = PPoly::extended_gcd(p, m);
                if e.gcd.degree() != Some(0) {
                    return PAlg::zero();
                }
                let g = e.gcd.coeff(0).inv();
                PAlg::reduce(e.x.scale(&g), m)
            }
        }
    }
}

/// A polynomial in the main variable over `ℚ(y)(α)`.
pub(crate) type PAPoly = GenPoly<PAlg>;

/// Is every coefficient of `f` within the size budget?
pub(crate) fn papoly_size_ok(f: &PAPoly) -> bool {
    f.coeffs().iter().all(PAlg::size_ok)
}

/// Largest degree in the main variable of a [`PAPoly`] built from an
/// expression.
const MAX_PAPOLY_DEGREE: usize = 60;

/// The field `ℚ(y₁, …, yₖ)(α)` of the coefficients of an expression rational
/// in a variable whose other generators are the free symbols `yⱼ` and
/// algebraic numbers generating `ℚ(α)` ([`ExtField`]), with the bridge from
/// and to expressions.
pub(crate) struct ParamAlgField {
    ext: ExtField,
    modulus: std::sync::Arc<PPoly>,
    /// `[var, y₁, …, yₖ]`, the variables of the coefficient polynomials
    /// (the main variable with exponent 0, as in [`to_ppoly`]).
    vars: Vec<ExprId>,
}

impl ParamAlgField {
    /// The field of the parameters `params` (free symbols) and the algebraic
    /// numbers `atoms` (at least one); `None` when the atoms do not generate
    /// a number field ([`ExtField::for_atoms`]).
    pub(crate) fn new(
        arena: &mut Arena,
        var: ExprId,
        params: &[ExprId],
        atoms: &[ExprId],
    ) -> Option<Self> {
        let ext = ExtField::for_atoms(arena, atoms)?;
        let m = ext.field.modulus();
        let mp = PPoly::from_coeffs(m.coeffs().iter().map(|c| PFrac::Rat(c.clone())).collect());
        let mut vars = vec![var];
        vars.extend_from_slice(params);
        Some(ParamAlgField {
            ext,
            modulus: std::sync::Arc::new(mp),
            vars,
        })
    }

    fn nv(&self) -> usize {
        self.vars.len()
    }

    /// An element of `ℚ(α)` in the field.
    fn element(&self, a: &AlgNum) -> PAlg {
        match a {
            AlgNum::Rat(q) => PAlg::Base(PFrac::Rat(q.clone())),
            AlgNum::Elt(p, _) => PAlg::reduce(
                PPoly::from_coeffs(p.coeffs().iter().map(|c| PFrac::Rat(c.clone())).collect()),
                &self.modulus,
            ),
        }
    }

    /// A polynomial over `ℚ(α)` in the field.
    pub(crate) fn lift_kpoly(&self, f: &KPoly) -> PAPoly {
        PAPoly::from_coeffs(f.coeffs().iter().map(|c| self.element(c)).collect())
    }

    /// A polynomial over `ℚ[var, y]` in the field.
    pub(crate) fn lift_multipoly(&self, f: &MPoly) -> PAPoly {
        PAPoly::from_coeffs(
            to_ppoly(f)
                .coeffs()
                .iter()
                .map(|c| PAlg::Base(c.clone()))
                .collect(),
        )
    }

    /// The variables `[var, y₁, …, yₖ]`.
    pub(crate) fn vars(&self) -> &[ExprId] {
        &self.vars
    }

    /// The number field of the algebraic numbers.
    pub(crate) fn ext(&mut self) -> &mut ExtField {
        &mut self.ext
    }

    /// `expr`, a polynomial in `var` whose coefficients are built from
    /// rationals, the parameters and the algebraic numbers of the field
    /// (a negative power only of a nonzero constant), as a [`PAPoly`];
    /// `None` for anything else or beyond the size budget.  Iterative
    /// post-order walk.
    pub(crate) fn poly_of(&mut self, arena: &mut Arena, expr: ExprId) -> Option<PAPoly> {
        let nv = self.nv();
        let var = self.vars[0];
        let index: FxHashMap<ExprId, usize> = self
            .vars
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &v)| (v, i))
            .collect();
        let mut cache: FxHashMap<ExprId, PAPoly> = FxHashMap::default();
        for id in walk::post_order_ids(arena, expr) {
            if cache.contains_key(&id) {
                continue;
            }
            let value = if id == var {
                PAPoly::x()
            } else if let Some(&i) = index.get(&id) {
                PAPoly::constant(PAlg::Base(PFrac::from_poly(&MPoly::var(nv, i))))
            } else {
                match arena.node(id).clone() {
                    ExprNode::Num(nid) => {
                        PAPoly::constant(PAlg::Base(PFrac::Rat(arena.num(nid).clone())))
                    }
                    ExprNode::Add(ch) => {
                        let mut acc = PAPoly::zero();
                        for c in ch.iter() {
                            acc = acc.add(cache.get(c)?);
                        }
                        acc
                    }
                    ExprNode::Mul(ch) => {
                        let mut acc = PAPoly::one();
                        for c in ch.iter() {
                            acc = acc.mul(cache.get(c)?);
                            if acc.degree().unwrap_or(0) > MAX_PAPOLY_DEGREE {
                                return None;
                            }
                        }
                        acc
                    }
                    ExprNode::Neg(inner) => cache.get(&inner)?.neg(),
                    ExprNode::Pow(base, e) if arena.as_num(e).is_some_and(|q| q.is_integer()) => {
                        let k = arena.as_num(e)?.to_integer();
                        let b = cache.get(&base)?.clone();
                        let n = usize::try_from(k.abs()).ok()?;
                        let deg = b.degree().unwrap_or(0);
                        if n > MAX_PAPOLY_DEGREE || deg.checked_mul(n)? > MAX_PAPOLY_DEGREE {
                            return None;
                        }
                        if k.is_negative() {
                            if deg != 0 || b.is_zero() {
                                return None;
                            }
                            PAPoly::constant(b.coeff(0).inv()).pow(n)
                        } else {
                            b.pow(n)
                        }
                    }
                    _ => match self.ext.atom_value(arena, id) {
                        Some(v) => PAPoly::constant(self.element(&v)),
                        // Children of generators are visited too; a parent
                        // that needs them fails above.
                        None => continue,
                    },
                }
            };
            if !papoly_size_ok(&value) {
                return None;
            }
            cache.insert(id, value);
        }
        cache.remove(&expr)
    }

    /// The expression of `f` over a common denominator in the parameters:
    /// `(Σᵢ Nᵢ·bᵢ)/c` with `Nᵢ ∈ ℚ[var, y]`, `c ∈ ℚ[y]` (factored) and the
    /// `bᵢ` the basis of `ℚ(α)` its elements are written in
    /// ([`ExtField::written_basis`]).
    pub(crate) fn poly_to_expr(&self, arena: &mut Arena, f: &PAPoly) -> ExprId {
        let nv = self.nv();
        let d = self.ext.field.degree();
        // The coefficients of αʲ as polynomials in `var` over ℚ(y).
        let per_power: Vec<PPoly> = (0..d)
            .map(|j| {
                PPoly::from_coeffs(
                    f.coeffs()
                        .iter()
                        .map(|c| c.coefficients(d)[j].clone())
                        .collect(),
                )
            })
            .collect();
        let parts: Vec<PolyFraction> = per_power
            .iter()
            .map(|p| ppoly_over_common_denominator(p, nv))
            .collect();
        let mut c = MPoly::from_int(nv, 1);
        for p in &parts {
            if !p.numer.is_zero() {
                c = MPoly::lcm(&c, &p.denom);
            }
        }
        let numers: Vec<MPoly> = parts
            .iter()
            .map(|p| match c.div_exact(&p.denom) {
                Some(m) => p.numer.mul(&m),
                None => MPoly::zero(nv),
            })
            .collect();
        let (basis, transform) = self.ext.written_basis(arena);
        let mut coords: Vec<MPoly> = match &transform {
            Some(inv) => inv
                .iter()
                .map(|row| {
                    let mut acc = MPoly::zero(nv);
                    for (q, n) in row.iter().zip(&numers) {
                        if !Zero::is_zero(q) {
                            acc = acc.add(&n.scale(q));
                        }
                    }
                    acc
                })
                .collect(),
            None => numers,
        };
        // Integer coefficients without a common factor, and a denominator
        // whose lexicographically leading coefficient is positive.
        let mut scale = Q::from_integer(c.clear_denominators().0);
        for n in &coords {
            if !n.is_zero() {
                scale = Q::from_integer(scale.numer().lcm(&n.clear_denominators().0));
            }
        }
        let mut g = c.scale(&scale).integer_content();
        for n in &coords {
            if !n.is_zero() {
                g = g.gcd(&n.scale(&scale).integer_content());
            }
        }
        if !Zero::is_zero(&g) {
            scale /= Q::from_integer(g);
        }
        let lex: MultiPoly<Lex> = c.convert_order();
        if lex.leading_coeff().is_some_and(Signed::is_negative) {
            scale = -scale;
        }
        let c = c.scale(&scale);
        for n in &mut coords {
            *n = n.scale(&scale);
        }
        let mut terms: Vec<ExprId> = Vec::new();
        for (n, &b) in coords.iter().zip(&basis) {
            if n.is_zero() {
                continue;
            }
            let ne = multipoly_to_expr(arena, n, &self.vars);
            terms.push(arena.mul(&[ne, b]));
        }
        let numer = match terms.len() {
            0 => arena.zero,
            1 => terms[0],
            _ => arena.add(&terms),
        };
        match c.as_constant() {
            Some(q) => {
                let inv = arena.num_ratio(qone() / q);
                arena.mul(&[inv, numer])
            }
            None => {
                let ce = factored_expr(arena, &c, &self.vars);
                arena.div(numer, ce)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// cancel
// ═══════════════════════════════════════════════════════════════════════════

/// Does the tree hold `±∞`, `zoo`, `nan` or an unevaluated node?
fn inadmissible(arena: &Arena, expr: ExprId) -> bool {
    if walk::has_unevaluated(arena, expr) {
        return true;
    }
    walk::post_order_ids(arena, expr).into_iter().any(|id| {
        matches!(
            arena.node(id),
            ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity | ExprNode::NaN
        )
    })
}

/// `Ex::cancel`: [`cancel`] (rational coefficients, a single quotient) when
/// it changes the expression; otherwise sums and nested fractions are
/// combined ([`together_deep`]) and cancelled, over `ℚ(α)` when the
/// coefficients hold algebraic numbers only ([`cancel_algebraic`]), else as
/// polynomials in all generators ([`cancel_params`]: the other symbols,
/// constants such as `π`, and functions such as `sin x` are independent
/// indeterminates, as in SymPy's `cancel`).  A sum with nothing to cancel
/// comes back combined (SymPy's `cancel` does that); anything else with
/// nothing to cancel comes back unchanged.
///
/// Before 0.43 sums (`1/(x − 1) + 1/((x − 1)(x + 1))`), symbolic
/// parameters (`(x² − a²)/(x − a)`) and algebraic coefficients
/// (`(x² − 2)/(x − √2)`) were returned unchanged.
pub(crate) fn cancel_full(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let r = cancel(arena, expr, var);
    if r != expr || inadmissible(arena, expr) {
        return r;
    }
    let t = together_deep(arena, expr);
    if t == arena.nan || t == arena.complex_infinity {
        return t;
    }
    if t != expr {
        let r = cancel(arena, t, var);
        if r != t {
            return r;
        }
    }
    if let Some(r) = cancel_algebraic(arena, t, var) {
        return r;
    }
    if let Some(r) = cancel_params(arena, t, var) {
        return r;
    }
    t
}

/// Cancel over `ℚ[var, y]`: the gcd of numerator and denominator and their
/// common integer content divided out; `None` when nothing cancels.
fn cancel_params(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    let rv = rational_in_var(arena, expr, var, Params::Any)?;
    let (p, q) = (&rv.numer, &rv.denom);
    if p.is_zero() {
        return Some(arena.zero);
    }
    let g = MPoly::gcd(p, q);
    let (p2, q2) = if g.total_degree().unwrap_or(0) > 0 {
        (p.div_exact(&g)?, q.div_exact(&g)?)
    } else {
        // Common integer content only.
        let (_, pz) = p.clear_denominators();
        let (_, qz) = q.clear_denominators();
        if One::is_one(&pz.integer_content().gcd(&qz.integer_content())) {
            return None;
        }
        (p.clone(), q.clone())
    };
    let r = normalize_pair(&p2, &q2);
    Some(quotient_expr(arena, &r.numer, &r.denom, &rv.vars))
}

/// Cancel over `ℚ(α)[var]` when every generator other than `var` is an
/// algebraic number; `None` when there is none, the field cannot be built,
/// or nothing cancels.
fn cancel_algebraic(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    let (n, d) = fraction_parts(arena, expr);
    let gens = numfield::generators(arena, &[n, d]);
    let atoms: Vec<ExprId> = gens.into_iter().filter(|&g| g != var).collect();
    if atoms.is_empty() || atoms.iter().any(|&g| walk::contains(arena, g, var)) {
        return None;
    }
    let mut ef = ExtField::for_atoms(arena, &atoms)?;
    let p = ef.kpoly_of(arena, n, var)?;
    let q = ef.kpoly_of(arena, d, var)?;
    if q.is_zero() {
        return None;
    }
    if p.is_zero() {
        return Some(arena.zero);
    }
    let g = KPoly::gcd(&p, &q);
    if g.degree().unwrap_or(0) == 0 {
        return None;
    }
    let mut p2 = p.div(&g);
    let mut q2 = q.div(&g);
    // A denominator with an irrational leading coefficient is made monic;
    // then both parts are scaled to coprime integer coefficients in ℤ[α].
    if let Some(lc) = q2.leading_coeff()
        && lc.as_rational().is_none()
    {
        let inv = lc.inv();
        p2 = p2.scale(&inv);
        q2 = q2.scale(&inv);
    }
    let mut scale = ef.primitive_scale(&q2);
    if q2
        .leading_coeff()
        .and_then(AlgNum::as_rational)
        .is_some_and(Signed::is_negative)
    {
        scale = -scale;
    }
    let p2 = p2.scale(&AlgNum::Rat(scale.clone()));
    let q2 = q2.scale(&AlgNum::Rat(scale));
    if q2.degree() == Some(0) {
        let c = q2.coeff(0).inv();
        let pe = ef.kpoly_to_expr(arena, &p2.scale(&c), var);
        return Some(pe);
    }
    let pe = ef.kpoly_to_expr(arena, &p2, var);
    let qe = ef.kpoly_to_expr(arena, &q2, var);
    Some(arena.div(pe, qe))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn show(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn pfrac_arithmetic_normalises() {
        let a: MPoly = MPoly::var(2, 1);
        let one = MPoly::from_int(2, 1);
        // 1/(a − 1) + 1/(a + 1) = 2a/(a² − 1)
        let f = PFrac::make(one.clone(), a.sub(&one), true);
        let g = PFrac::make(one.clone(), a.add(&one), true);
        let s = Ring::add(&f, &g);
        let PolyFraction { numer: n, denom: d } = s.parts(2);
        assert_eq!(n, a.scale(&Q::from_integer(BigInt::from(2))));
        assert_eq!(d, a.mul(&a).sub(&one));
        // (a − 1)·1/(a − 1) = 1
        let h = Ring::mul(&f, &PFrac::from_poly(&a.sub(&one)));
        assert!(h.is_one());
    }

    #[test]
    fn cancel_with_a_parameter() {
        let mut ar = Arena::new();
        let x = ar.symbol("x");
        let a = ar.symbol("a");
        let two = ar.int(2);
        let x2 = ar.pow(x, two);
        let a2 = ar.pow(a, two);
        let na2 = ar.neg(a2);
        let num = ar.add(&[x2, na2]);
        let na = ar.neg(a);
        let den = ar.add(&[x, na]);
        let e = ar.div(num, den);
        let r = cancel_full(&mut ar, e, x);
        assert_eq!(show(&ar, r), "a + x");
    }

    #[test]
    fn cancel_over_q_sqrt2() {
        let mut ar = Arena::new();
        let x = ar.symbol("x");
        let two = ar.int(2);
        let half = ar.rational(1, 2);
        let s2 = ar.pow(two, half);
        let x2 = ar.pow(x, two);
        let m2 = ar.int(-2);
        let num = ar.add(&[x2, m2]);
        let ns2 = ar.neg(s2);
        let den = ar.add(&[x, ns2]);
        let e = ar.div(num, den);
        let r = cancel_full(&mut ar, e, x);
        assert_eq!(show(&ar, r), "x + sqrt(2)");
    }
}
