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

use std::cell::Cell;
use std::fmt;
use std::marker::PhantomData;

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

/// What a computation over `ℚ(y₁, …, yₖ)` ([`PFrac<P>`]) does at each
/// field operation besides the arithmetic: the gcd it cancels with and the
/// budget it charges.  The arithmetic and the canonical form are written
/// once, in [`PFrac`]; up to 0.49 the integrator had its own copy of them
/// (`risch::param_rational::PFrac`) for the sake of its budget.
pub(crate) trait ParamPolicy: Clone + fmt::Debug + PartialEq {
    /// `gcd(a, b)` of two numerators or denominators (`1` to skip the
    /// cancellation: a constant operand, an attempt given up).
    fn gcd(a: &MPoly, b: &MPoly) -> MPoly;
    /// A quotient is about to be normalised ([`PFrac::finish`]).
    fn charge() {}
    /// May the non-zero quotient `num/den` be normalised, or does it
    /// become `0` (an attempt abandoned)?
    fn admit(num: &MPoly, den: &MPoly) -> bool {
        let _ = (num, den);
        true
    }
    /// A product of the field arithmetic has an exponent beyond `u32`
    /// ([`MultiPoly::try_mul`] refused): the computation cannot go on.
    /// After this call [`abandoned`](Self::abandoned) is true, so that
    /// every element counts as zero and every loop ends, and the owner of
    /// the computation discards its result (the integrator at its next
    /// budget check, `apart` at [`ppoly_size_ok`] / [`papoly_size_ok`]).
    /// Up to 0.58 such a product was the asserting [`MultiPoly::mul`]: a
    /// panic.
    fn overflow();
    /// Does every element count as zero (an attempt abandoned)?
    fn abandoned() -> bool;
}

/// Exact arithmetic with no budget: the field of `cancel` and `apart`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Exact;

thread_local! {
    /// Has a product of the computation over [`PFrac<Exact>`] in progress on
    /// this thread left the exponent range ([`ParamPolicy::overflow`])?
    /// Cleared where such a computation starts ([`rational_in_var`],
    /// [`ParamAlgField::new`]); in a thread-local because the field
    /// operations (the [`Ring`]/[`Field`] traits) take no context, as the
    /// integrator's budget is.
    static EXACT_OVERFLOW: Cell<bool> = const { Cell::new(false) };
}

/// A computation over [`PFrac<Exact>`] starts: no product has overflowed.
fn begin_exact() {
    EXACT_OVERFLOW.with(|c| c.set(false));
}

impl ParamPolicy for Exact {
    fn gcd(a: &MPoly, b: &MPoly) -> MPoly {
        let one = MPoly::from_int(a.num_vars(), 1);
        if a.as_constant().is_some() || b.as_constant().is_some() {
            return one;
        }
        // Refused (an exponent of the PRS beyond `u32`): like a product.
        MPoly::try_gcd(a, b).unwrap_or_else(|| {
            Self::overflow();
            one
        })
    }

    fn overflow() {
        EXACT_OVERFLOW.with(|c| c.set(true));
    }

    fn abandoned() -> bool {
        EXACT_OVERFLOW.with(Cell::get)
    }
}

/// An element of `ℚ(y₁, …, yₖ)`.
///
/// `Const` holds the rational constants (so that [`Ring::zero`] and
/// [`Ring::one`] need no variable count); `Frac` a non-constant quotient
/// `num/den` in canonical form: `gcd(num, den) = 1`, `den` with integer
/// coefficients, primitive, positive leading coefficient.  Equal elements
/// therefore have equal representations, and `==` is equality in the
/// field.  All polynomials of one computation have the same number of
/// variables.  `P` decides the gcd and the budget ([`ParamPolicy`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PFrac<P: ParamPolicy = Exact> {
    /// A rational constant.
    Const(Q),
    /// `num / den` (and the policy, which has no data).
    Frac(Box<(MPoly, MPoly, PhantomData<P>)>),
}

/// `a/g` for a divisor `g` of `a` (`None` if it does not divide).
fn div_mp(a: &MPoly, g: &MPoly) -> Option<MPoly> {
    if g.as_constant().is_some_and(|c| One::is_one(&c)) {
        return Some(a.clone());
    }
    a.div_exact(g)
}

/// `a·b` in the field arithmetic of the policy `P`; `None`, after telling
/// the policy ([`ParamPolicy::overflow`]), when an exponent of the product
/// is beyond `u32`.
fn mul_mp<P: ParamPolicy>(a: &MPoly, b: &MPoly) -> Option<MPoly> {
    let p = a.try_mul(b);
    if p.is_none() {
        P::overflow();
    }
    p
}

impl<P: ParamPolicy> PFrac<P> {
    /// `num/den` in canonical form (`den ≠ 0`).
    pub(crate) fn canon(num: MPoly, den: MPoly) -> Self {
        if num.is_zero() || den.is_zero() {
            return PFrac::Const(qzero());
        }
        let g = P::gcd(&num, &den);
        match (div_mp(&num, &g), div_mp(&den, &g)) {
            (Some(n), Some(d)) => PFrac::finish(n, d),
            _ => PFrac::finish(num, den),
        }
    }

    /// `num/den` for coprime `num`, `den` (`den ≠ 0`): the denominator's
    /// rational content and sign moved into the numerator.
    pub(crate) fn finish(num: MPoly, den: MPoly) -> Self {
        P::charge();
        if num.is_zero() || den.is_zero() || !P::admit(&num, &den) {
            return PFrac::Const(qzero());
        }
        let (mut num, mut den) = (num, den);
        let (lcm, dz) = den.clear_denominators();
        let content = dz.integer_content();
        let mut lambda = if Zero::is_zero(&content) {
            Q::from_integer(lcm)
        } else {
            Q::new(lcm, content)
        };
        if den.leading_coeff().is_some_and(Signed::is_negative) {
            lambda = -lambda;
        }
        if !One::is_one(&lambda) {
            num = num.scale(&lambda);
            den = den.scale(&lambda);
        }
        if den.as_constant().is_some()
            && let Some(n) = num.as_constant()
        {
            return PFrac::Const(n);
        }
        PFrac::Frac(Box::new((num, den, PhantomData)))
    }

    /// The element `p` of `ℚ[y₁, …]`.
    pub(crate) fn from_mp(p: MPoly) -> Self {
        if let Some(c) = p.as_constant() {
            return PFrac::Const(c);
        }
        let one = MPoly::from_int(p.num_vars(), 1);
        PFrac::Frac(Box::new((p, one, PhantomData)))
    }

    /// The element `p` (a polynomial).
    pub(crate) fn from_poly(p: &MPoly) -> Self {
        Self::from_mp(p.clone())
    }

    /// The number of variables of the polynomials (`None` for a constant).
    pub(crate) fn num_vars(&self) -> Option<usize> {
        match self {
            PFrac::Const(_) => None,
            PFrac::Frac(f) => Some(f.0.num_vars()),
        }
    }

    /// Numerator and denominator as polynomials in `nv` variables.
    pub(crate) fn parts(&self, nv: usize) -> PolyFraction {
        match self {
            PFrac::Const(q) => PolyFraction {
                numer: MPoly::constant(nv, q.clone()),
                denom: MPoly::from_int(nv, 1),
            },
            PFrac::Frac(f) => PolyFraction {
                numer: f.0.clone(),
                denom: f.1.clone(),
            },
        }
    }

    /// The rational constant, if this is one.
    pub(crate) fn as_const(&self) -> Option<&Q> {
        match self {
            PFrac::Const(q) => Some(q),
            PFrac::Frac(_) => None,
        }
    }

    /// `k·self` for a rational `k` (no gcd needed).
    pub(crate) fn scale_q(&self, k: &Q) -> Self {
        if Zero::is_zero(k) {
            return PFrac::Const(qzero());
        }
        match self {
            PFrac::Const(q) => PFrac::Const(q * k),
            PFrac::Frac(f) => PFrac::Frac(Box::new((f.0.scale(k), f.1.clone(), PhantomData))),
        }
    }

    /// Within the size budget, in a computation that was not abandoned.
    fn size_ok(&self) -> bool {
        !P::abandoned()
            && match self {
                PFrac::Const(_) => true,
                PFrac::Frac(f) => within_budget(&f.0) && within_budget(&f.1),
            }
    }

    /// Both operands as `num/den` over the same variables.
    fn pair(&self, rhs: &Self) -> Option<[PolyFraction; 2]> {
        let nv = self.num_vars().or_else(|| rhs.num_vars())?;
        Some([self.parts(nv), rhs.parts(nv)])
    }

    /// Henrici's sum: with `g = gcd(ad, bd)`, `(an·bd/g + bn·ad/g)/(ad·bd/g)`
    /// reduced by the gcd of that numerator with `g` alone (Knuth, TAOCP
    /// vol. 2, §4.5.1).  The gcds are of the factors, not of the products:
    /// with four parameters the heuristic gcd of the products took seconds.
    fn add_frac(&self, rhs: &Self) -> Self {
        let Some([a, b]) = self.pair(rhs) else {
            return PFrac::Const(qzero());
        };
        let (an, ad, bn, bd) = (&a.numer, &a.denom, &b.numer, &b.denom);
        let henrici = || {
            let g = P::gcd(ad, bd);
            let (ad1, bd1) = (div_mp(ad, &g)?, div_mp(bd, &g)?);
            let num = mul_mp::<P>(an, &bd1)?.add(&mul_mp::<P>(bn, &ad1)?);
            let g2 = P::gcd(&num, &g);
            Some(PFrac::finish(
                div_mp(&num, &g2)?,
                div_mp(&mul_mp::<P>(&ad1, bd)?, &g2)?,
            ))
        };
        let plain = || {
            let num = mul_mp::<P>(an, bd)?.add(&mul_mp::<P>(bn, ad)?);
            Some(PFrac::canon(num, mul_mp::<P>(ad, bd)?))
        };
        // `None` from both: a product overflowed, the computation is
        // abandoned and every element is zero.
        henrici()
            .or_else(plain)
            .unwrap_or_else(|| PFrac::Const(qzero()))
    }

    /// Henrici's product: `(an/g₁·bn/g₂)/(ad/g₂·bd/g₁)` with
    /// `g₁ = gcd(an, bd)`, `g₂ = gcd(bn, ad)`.
    fn mul_frac(&self, rhs: &Self) -> Self {
        let Some([a, b]) = self.pair(rhs) else {
            return PFrac::Const(qzero());
        };
        let (an, ad, bn, bd) = (&a.numer, &a.denom, &b.numer, &b.denom);
        let henrici = || {
            let g1 = P::gcd(an, bd);
            let g2 = P::gcd(bn, ad);
            let num = mul_mp::<P>(&div_mp(an, &g1)?, &div_mp(bn, &g2)?)?;
            let den = mul_mp::<P>(&div_mp(ad, &g2)?, &div_mp(bd, &g1)?)?;
            Some(PFrac::finish(num, den))
        };
        let plain = || Some(PFrac::canon(mul_mp::<P>(an, bn)?, mul_mp::<P>(ad, bd)?));
        // `None` from both: as in `add_frac`.
        henrici()
            .or_else(plain)
            .unwrap_or_else(|| PFrac::Const(qzero()))
    }
}

impl<P: ParamPolicy> Ring for PFrac<P> {
    fn zero() -> Self {
        PFrac::Const(qzero())
    }

    fn one() -> Self {
        PFrac::Const(qone())
    }

    /// Once an attempt is abandoned ([`ParamPolicy::abandoned`]) every
    /// element counts as zero: the operations return zero then, and a
    /// division whose quotient coefficient came out zero would otherwise
    /// never remove the leading term of its remainder (`GenPoly::try_div_rem`
    /// looped for minutes inside the extended gcd of
    /// `1/((d + e·x)²·(a + c·x⁴)³)`); with every coefficient zero each
    /// polynomial normalises to zero and every loop ends.  The result is
    /// discarded at the next budget check.
    fn is_zero(&self) -> bool {
        P::abandoned() || matches!(self, PFrac::Const(q) if Zero::is_zero(q))
    }

    fn add(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (PFrac::Const(a), PFrac::Const(b)) => PFrac::Const(a + b),
            _ if rhs.is_zero() => self.clone(),
            _ if self.is_zero() => rhs.clone(),
            _ => self.add_frac(rhs),
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.neg())
    }

    fn mul(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (PFrac::Const(a), PFrac::Const(b)) => PFrac::Const(a * b),
            (PFrac::Const(a), _) => rhs.scale_q(a),
            (_, PFrac::Const(b)) => self.scale_q(b),
            _ => self.mul_frac(rhs),
        }
    }

    fn neg(&self) -> Self {
        self.scale_q(&-qone())
    }
}

impl<P: ParamPolicy> EuclideanDomain for PFrac<P> {
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        (Field::div(self, other), Self::zero())
    }
}

impl<P: ParamPolicy> Field for PFrac<P> {
    /// Division; by zero it gives zero (the polynomial algorithms divide
    /// only by leading coefficients, which are never zero).
    fn div(&self, other: &Self) -> Self {
        self.mul(&other.inv())
    }

    /// The inverse; that of zero is zero (see [`Field::div`]).
    fn inv(&self) -> Self {
        match self {
            PFrac::Const(q) if Zero::is_zero(q) => PFrac::Const(qzero()),
            PFrac::Const(q) => PFrac::Const(q.recip()),
            PFrac::Frac(f) => PFrac::finish(f.1.clone(), f.0.clone()),
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
    let p = a.try_mul(b)?;
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
    begin_exact();
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
    others.sort_by(|&a, &b| arena.canonical_cmp(a, b));
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

/// Is every coefficient of `f` within the size budget, and the computation
/// it comes from not abandoned ([`ParamPolicy::overflow`]: every element
/// is zero then, and a zero polynomial has no coefficient to fail the
/// test)?
pub(crate) fn ppoly_size_ok(f: &PPoly) -> bool {
    !Exact::abandoned() && f.coeffs().iter().all(PFrac::size_ok)
}

/// `f` written as `N/c` with `N ∈ ℚ[var, y]` and `c ∈ ℚ[y]` (the lcm of
/// the coefficient denominators), both as polynomials in `nv` variables;
/// `None` when an exponent of `c` or `N` would be beyond `u32`.
pub(crate) fn ppoly_over_common_denominator(f: &PPoly, nv: usize) -> Option<PolyFraction> {
    let mut c = MPoly::from_int(nv, 1);
    for co in f.coeffs() {
        let d = co.parts(nv).denom;
        if d.total_degree().unwrap_or(0) > 0 {
            c = MPoly::try_lcm(&c, &d)?;
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
        if n.is_zero() {
            continue;
        }
        // `d` divides the lcm `c`.
        let m = c.div_exact(&d)?;
        let mut e = vec![0u32; nv];
        *e.first_mut()? = u32::try_from(k).ok()?;
        num = num.add(&n.try_mul(&m)?.try_mul_monomial(&qone(), &e)?);
    }
    if !One::is_one(&l) {
        let lq = Q::from_integer(l);
        num = num.scale(&lq);
        c = c.scale(&lq);
    }
    Some(PolyFraction {
        numer: num,
        denom: c,
    })
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

    /// As for [`PFrac`], every element counts as zero once the computation
    /// is abandoned ([`ParamPolicy::overflow`]).
    fn is_zero(&self) -> bool {
        match self {
            PAlg::Base(c) => c.is_zero(),
            PAlg::Elt(..) => Exact::abandoned(),
        }
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

/// Is every coefficient of `f` within the size budget, and the computation
/// not abandoned (see [`ppoly_size_ok`])?
pub(crate) fn papoly_size_ok(f: &PAPoly) -> bool {
    !Exact::abandoned() && f.coeffs().iter().all(PAlg::size_ok)
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
        begin_exact();
        let ext = ExtField::for_atoms(arena, atoms)?;
        let m = ext.field.modulus();
        let mp = PPoly::from_coeffs(m.coeffs().iter().map(|c| PFrac::Const(c.clone())).collect());
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
            AlgNum::Rat(q) => PAlg::Base(PFrac::Const(q.clone())),
            AlgNum::Elt(p, _) => PAlg::reduce(
                PPoly::from_coeffs(p.coeffs().iter().map(|c| PFrac::Const(c.clone())).collect()),
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
                        PAPoly::constant(PAlg::Base(PFrac::Const(arena.num(nid).clone())))
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
    /// ([`ExtField::written_basis`]); `None` when an exponent of the common
    /// denominator or a numerator would be beyond `u32`.
    pub(crate) fn poly_to_expr(&self, arena: &mut Arena, f: &PAPoly) -> Option<ExprId> {
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
            .collect::<Option<_>>()?;
        let mut c = MPoly::from_int(nv, 1);
        for p in &parts {
            if !p.numer.is_zero() {
                c = MPoly::try_lcm(&c, &p.denom)?;
            }
        }
        let numers: Vec<MPoly> = parts
            .iter()
            .map(|p| match c.div_exact(&p.denom) {
                Some(m) => p.numer.try_mul(&m),
                // Only a zero numerator's denominator is outside the lcm.
                None => Some(MPoly::zero(nv)),
            })
            .collect::<Option<_>>()?;
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
        Some(match c.as_constant() {
            Some(q) => {
                let inv = arena.num_ratio(qone() / q);
                arena.mul(&[inv, numer])
            }
            None => {
                let ce = factored_expr(arena, &c, &self.vars);
                arena.div(numer, ce)
            }
        })
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

/// The polynomial part of `Ex::cancel` (which then decides the
/// denominators divided out here: `simplify::ratsimp::cancel`): [`cancel`]
/// (rational coefficients, a single quotient) when
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
        let f = PFrac::<Exact>::finish(one.clone(), a.sub(&one));
        let g = PFrac::<Exact>::finish(one.clone(), a.add(&one));
        let s = Ring::add(&f, &g);
        let PolyFraction { numer: n, denom: d } = s.parts(2);
        assert_eq!(n, a.scale(&Q::from_integer(BigInt::from(2))));
        assert_eq!(d, a.mul(&a).sub(&one));
        // (a − 1)·1/(a − 1) = 1
        let h = Ring::mul(&f, &PFrac::from_poly(&a.sub(&one)));
        assert!(h.is_one());
    }

    /// Up to 0.58 the field arithmetic multiplied with the asserting
    /// `MultiPoly::mul`: `a^(2³¹)·a^(2³¹)`, `a^(2³¹) + 1/(a^(2³¹) + 1)`
    /// and `1/(a^(2³¹) + 1)²` panicked ("MultiPoly::mul: exponent
    /// overflow": the exponent `2³²` does not fit `u32`).  Now the
    /// computation is abandoned: the result is zero, every element counts
    /// as zero (so the loops of the polynomial algorithms end), and the
    /// size test that guards a result fails until the next computation
    /// starts.
    #[test]
    fn pfrac_product_beyond_u32_exponents_abandons() {
        let a: MPoly = MPoly::var(2, 1);
        let one = MPoly::from_int(2, 1);
        let big = a.pow(1 << 31);
        let unit = PPoly::constant(PFrac::one());
        let cases: [(PFrac, PFrac, bool); 3] = [
            (PFrac::from_poly(&big), PFrac::from_poly(&big), true),
            (
                PFrac::from_poly(&big),
                PFrac::finish(one.clone(), big.add(&one)),
                false,
            ),
            (
                PFrac::finish(one.clone(), big.add(&one)),
                PFrac::finish(one.clone(), big.add(&one)),
                true,
            ),
        ];
        for (f, g, product) in cases {
            begin_exact();
            assert!(!Exact::abandoned() && ppoly_size_ok(&unit));
            assert!(!f.is_zero() && !g.is_zero());
            let r = if product {
                Ring::mul(&f, &g)
            } else {
                Ring::add(&f, &g)
            };
            assert!(r.is_zero() && r.as_const().is_some());
            assert!(Exact::abandoned());
            // Everything is zero now, and no result passes the size test,
            // not even a polynomial without coefficients.
            assert!(f.is_zero());
            assert!(!ppoly_size_ok(&unit) && !ppoly_size_ok(&PPoly::zero()));
            assert!(!papoly_size_ok(&PAPoly::zero()));
        }
        // The next computation starts clean, and exponents that fit are
        // exact: a^(2³⁰)·a^(2³⁰) = a^(2³¹).
        begin_exact();
        let half = PFrac::<Exact>::from_poly(&a.pow(1 << 30));
        assert_eq!(Ring::mul(&half, &half), PFrac::from_poly(&big));
        assert!(!Exact::abandoned() && ppoly_size_ok(&unit));
        // The common denominator of a polynomial's coefficients refuses
        // the same way: lcm(a^(2³¹) + 1, a^(2³¹) − 1) has the exponent 2³².
        let f = PPoly::from_coeffs(vec![
            PFrac::finish(one.clone(), big.add(&one)),
            PFrac::finish(one.clone(), big.sub(&one)),
        ]);
        assert!(ppoly_over_common_denominator(&f, 2).is_none());
        let g = PPoly::from_coeffs(vec![
            PFrac::finish(one.clone(), a.add(&one)),
            PFrac::finish(one.clone(), a.sub(&one)),
        ]);
        let PolyFraction { numer, denom } = ppoly_over_common_denominator(&g, 2).unwrap();
        assert_eq!(denom, a.mul(&a).sub(&one));
        // (a − 1) + (a + 1)·x
        let x = MPoly::var(2, 0);
        assert_eq!(numer, a.sub(&one).add(&a.add(&one).mul(&x)));
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
