//! `∫ A(x)/D(x) dx` for polynomials `A`, `D` whose coefficients are
//! polynomials over `ℚ` in parameters `p₁, …, p_k` (symbols other than the
//! variable): the rational integrator of [`super::try_risch_rational`]
//! extended from `ℚ(x)` to `ℚ(p₁, …, p_k)(x)`.
//!
//! The coefficient field `ℚ(p₁, …, p_k)` is [`PFrac`] (a quotient of
//! [`MultiPoly`]s reduced by their gcd), so the univariate algorithms of
//! [`GenPoly`] — Euclidean division, gcd, extended gcd — run over it
//! unchanged.  The steps are Bronstein's (*Symbolic Integration I*, 2nd ed.,
//! §2.1–2.2 and §2.5):
//!
//! 1. polynomial division `A = Q·D + R`, `∫Q` termwise;
//! 2. Hermite reduction (Mack's linear version) of `R/D` to
//!    `Σ Bᵢ/D⁻ᵢ + ∫ H/D*` with `D*` square-free;
//! 3. the logarithmic part of `H/D*` factor by factor over a coprime
//!    factorisation of `D*` — the syntactic factors of the integrand's
//!    denominator refined by gcds, parameter-free factors split over `ℚ`.
//!    A factor's partial-fraction numerator `aₑ` gives the residues
//!    `aₑ(r)/e′(r)` at its roots `r` (the roots of the Rothstein–Trager
//!    resultant restricted to that factor):
//!    * `e = x − r`: `c·ln(x − r)`;
//!    * `e` quadratic: its roots are rational in the parameters when the
//!      discriminant `Δ` is a square in `ℚ(p₁, …)` (`x² − a²`), else
//!      `r± = (−B ± √Δ)/2`, with the square factors of `Δ` taken out of
//!      the root; a single `(μ/2)·ln e` when the two residues agree (the
//!      resultant's double root, `x/(x² + a)`); the `atan` form only when
//!      `e` is real under the declared assumptions and `−Δ ≥ 0` is proved
//!      (`x² + a` for a positive `a`);
//!    * a parameter-free `e` of higher degree: the numerator's
//!      coefficients times the `ℚ`-integrals of `xʲ/e`
//!      ([`super::integrate_rational_function`]); a parametric one only
//!      when `aₑ` is a multiple of `e′`.
//!
//! The logarithms are written `ln|x − r|` with monic arguments: the stage
//! exit of the integrator keeps `|·|` only where `x − r` is real under the
//! declared assumptions and otherwise writes the principal `ln(x − r)`
//! (decision D4, 0.31), which for a non-real `r` is analytic along the
//! whole real line (its argument never meets the negative real axis).
//!
//! The answer is the generic one: valid for all parameter values outside
//! the proper subvariety where a denominator of the computation vanishes
//! (`1/((x + a)(x + b))` at `a = b`), as SymPy's `ratint`; the integrator's
//! degenerate-case wrapper adds the `Piecewise` branches it can find.

use std::cell::Cell;

use num_bigint::BigInt;
use num_traits::Signed;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::poly::dense::Poly;
use crate::poly::generic::GenPoly;
use crate::poly::multipoly::{GrevLex, MultiPoly};
use crate::poly::traits::{EuclideanDomain, Field, Ring};

type MP = MultiPoly<GrevLex>;
type GP = GenPoly<PFrac>;

/// At most this many parameters (the multivariate gcds grow quickly).
const MAX_PARAMS: usize = 4;
/// Largest integer exponent accepted in the integrand: that of the
/// numerator's degree (up to 0.31 it was 16, and `x²³/(a + b·x³)³`,
/// `x¹⁹/(a + b·x⁵)` were refused although their degrees are within the
/// limits below).
const MAX_EXPONENT: i64 = 24;
/// Largest degree in `x` of the denominator after cancellation.
const MAX_DENOM_DEGREE: usize = 10;
/// Largest degree in `x` of the numerator.
const MAX_NUMER_DEGREE: usize = 24;
/// Field operations allowed for one integrand (each normalises by a
/// multivariate gcd); beyond it the route gives up.
const OP_BUDGET: u64 = 60_000;
/// Largest coefficient (numerator plus denominator terms) the field
/// normalises; a larger one ends the attempt (a deterministic bound on the
/// multivariate gcds, whose cost grows with the size of their inputs).
const MAX_COEFF_TERMS: usize = 400;

thread_local! {
    static OPS: Cell<u64> = const { Cell::new(0) };
    static TOO_BIG: Cell<bool> = const { Cell::new(false) };
}

fn reset_budget() {
    OPS.with(|c| c.set(0));
    TOO_BIG.with(|c| c.set(false));
}

fn budget_exceeded() -> bool {
    OPS.with(Cell::get) > OP_BUDGET || TOO_BIG.with(Cell::get)
}

// ═══════════════════════════════════════════════════════════════════════════
// The coefficient field ℚ(p₁, …, p_k)
// ═══════════════════════════════════════════════════════════════════════════

/// An element of `ℚ(p₁, …, p_k)`.
///
/// `Const` holds the rational constants (so that [`Ring::zero`] and
/// [`Ring::one`] need no variable count); `Frac` a non-constant quotient
/// `num/den` in canonical form: `gcd(num, den) = 1`, `den` with integer
/// coefficients, primitive, positive leading coefficient.  Equal elements
/// therefore have equal representations, and `==` is equality in the
/// field.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PFrac {
    Const(Q),
    Frac(Box<(MP, MP)>),
}

/// Bound on `∏ᵥ (min(degᵥ a, degᵥ b) + 1)` for the inputs of a gcd.  The
/// heuristic gcd evaluates one variable after the other at a large
/// integer, so its numbers grow with the product of the degrees: in four
/// parameters two coefficients of total degree 35 (degrees 8, 19, 11, 19
/// in the variables, a product of 43,200) took 6–13 s in a by-parts
/// sub-integral of a Rubi entry; calls up to about 10,000 take below
/// 0.2 s (release build).
const GCD_COST_LIMIT: u64 = 20_000;

/// `gcd(a, b)` in `ℤ[p₁, …]` ([`MultiPoly::gcd`]), `1` at once when either
/// is a constant.  Beyond [`GCD_COST_LIMIT`] the attempt is abandoned
/// (`TOO_BIG`; `1` is returned meanwhile).
fn gcd_mp(a: &MP, b: &MP) -> MP {
    let one = MP::from_int(a.num_vars(), 1);
    if a.as_constant().is_some() || b.as_constant().is_some() || TOO_BIG.with(Cell::get) {
        return one;
    }
    if !gcd_affordable(a, b) {
        TOO_BIG.with(|c| c.set(true));
        return one;
    }
    let t = std::time::Instant::now();
    let g = MP::gcd(a, b);
    let ms = t.elapsed().as_millis();
    if ms > 50 {
        let degs = |p: &MP| {
            (0..p.num_vars())
                .map(|v| p.degree_in(v))
                .collect::<Vec<_>>()
        };
        tracing::debug!(ms, a_degs = ?degs(a), b_degs = ?degs(b), a_tot = ?a.total_degree(), b_tot = ?b.total_degree(), "param_rational: slow gcd");
    }
    g
}

/// Is `gcd(a, b)` within [`GCD_COST_LIMIT`]: `∏ (min(degᵥ a, degᵥ b) + 1)`
/// over the variables?
fn gcd_affordable(a: &MP, b: &MP) -> bool {
    let mut cost: u64 = 1;
    for v in 0..a.num_vars() {
        let m = u64::from(a.degree_in(v).min(b.degree_in(v)));
        cost = cost.saturating_mul(m + 1);
    }
    cost <= GCD_COST_LIMIT
}

/// `a/g` for a divisor `g` of `a` (`None` if it does not divide).
fn div_mp(a: &MP, g: &MP) -> Option<MP> {
    if g.as_constant().is_some_and(|c| c.is_one()) {
        return Some(a.clone());
    }
    a.div_exact(g)
}

impl PFrac {
    /// `num/den` in canonical form (`den ≠ 0`).
    fn canon(num: MP, den: MP) -> Self {
        if num.is_zero() || den.is_zero() {
            return PFrac::Const(Q::zero());
        }
        let g = gcd_mp(&num, &den);
        match (div_mp(&num, &g), div_mp(&den, &g)) {
            (Some(n), Some(d)) => PFrac::finish(n, d),
            _ => PFrac::finish(num, den),
        }
    }

    /// `num/den` for coprime `num`, `den` (`den ≠ 0`): the denominator's
    /// rational content and sign moved into the numerator.
    fn finish(num: MP, den: MP) -> Self {
        OPS.with(|c| c.set(c.get() + 1));
        if num.is_zero() || den.is_zero() {
            return PFrac::Const(Q::zero());
        }
        if TOO_BIG.with(Cell::get) || num.num_terms() + den.num_terms() > MAX_COEFF_TERMS {
            // The attempt is abandoned at the next budget check; until then
            // everything is zero, which ends every loop quickly.
            TOO_BIG.with(|c| c.set(true));
            return PFrac::Const(Q::zero());
        }
        let (mut num, mut den) = (num, den);
        let (lcm, dz) = den.clear_denominators();
        let content = dz.integer_content();
        let mut lambda = if content.is_zero() {
            Q::from_integer(lcm)
        } else {
            Q::new(lcm, content)
        };
        if den.leading_coeff().is_some_and(Signed::is_negative) {
            lambda = -lambda;
        }
        if !lambda.is_one() {
            num = num.scale(&lambda);
            den = den.scale(&lambda);
        }
        if den.as_constant().is_some()
            && let Some(n) = num.as_constant()
        {
            return PFrac::Const(n);
        }
        PFrac::Frac(Box::new((num, den)))
    }

    /// The element `p` of `ℚ[p₁, …]`.
    fn from_mp(p: MP) -> Self {
        if let Some(c) = p.as_constant() {
            return PFrac::Const(c);
        }
        let one = MP::from_int(p.num_vars(), 1);
        PFrac::Frac(Box::new((p, one)))
    }

    fn num_vars(&self) -> Option<usize> {
        match self {
            PFrac::Const(_) => None,
            PFrac::Frac(f) => Some(f.0.num_vars()),
        }
    }

    /// `(num, den)` as polynomials in `nv` variables.
    fn parts(&self, nv: usize) -> (MP, MP) {
        match self {
            PFrac::Const(q) => (MP::constant(nv, q.clone()), MP::from_int(nv, 1)),
            PFrac::Frac(f) => (f.0.clone(), f.1.clone()),
        }
    }

    fn as_const(&self) -> Option<&Q> {
        match self {
            PFrac::Const(q) => Some(q),
            PFrac::Frac(_) => None,
        }
    }

    /// `k·self` for a rational `k` (no gcd needed).
    fn scale_q(&self, k: &Q) -> Self {
        if k.is_zero() {
            return PFrac::Const(Q::zero());
        }
        match self {
            PFrac::Const(q) => PFrac::Const(q * k),
            PFrac::Frac(f) => PFrac::Frac(Box::new((f.0.scale(k), f.1.clone()))),
        }
    }

    /// Both operands as `(num, den)` pairs over the same variables.
    fn pair(&self, rhs: &Self) -> Option<((MP, MP), (MP, MP))> {
        let nv = self.num_vars().or_else(|| rhs.num_vars())?;
        Some((self.parts(nv), rhs.parts(nv)))
    }

    /// Henrici's sum: with `g = gcd(ad, bd)`, `(an·bd/g + bn·ad/g)/(ad·bd/g)`
    /// reduced by the gcd of that numerator with `g` alone (Knuth, TAOCP
    /// vol. 2, §4.5.1).  The gcds are of the factors, not of the products:
    /// with four parameters the heuristic gcd of the products took seconds.
    fn add_frac(&self, rhs: &Self) -> Self {
        let Some(((an, ad), (bn, bd))) = self.pair(rhs) else {
            return PFrac::Const(Q::zero());
        };
        let henrici = || {
            let g = gcd_mp(&ad, &bd);
            let (ad1, bd1) = (div_mp(&ad, &g)?, div_mp(&bd, &g)?);
            let num = an.mul(&bd1).add(&bn.mul(&ad1));
            let g2 = gcd_mp(&num, &g);
            Some(PFrac::finish(
                div_mp(&num, &g2)?,
                div_mp(&ad1.mul(&bd), &g2)?,
            ))
        };
        henrici().unwrap_or_else(|| PFrac::canon(an.mul(&bd).add(&bn.mul(&ad)), ad.mul(&bd)))
    }

    /// Henrici's product: `(an/g₁·bn/g₂)/(ad/g₂·bd/g₁)` with
    /// `g₁ = gcd(an, bd)`, `g₂ = gcd(bn, ad)`.
    fn mul_frac(&self, rhs: &Self) -> Self {
        let Some(((an, ad), (bn, bd))) = self.pair(rhs) else {
            return PFrac::Const(Q::zero());
        };
        let henrici = || {
            let g1 = gcd_mp(&an, &bd);
            let g2 = gcd_mp(&bn, &ad);
            let num = div_mp(&an, &g1)?.mul(&div_mp(&bn, &g2)?);
            let den = div_mp(&ad, &g2)?.mul(&div_mp(&bd, &g1)?);
            Some(PFrac::finish(num, den))
        };
        henrici().unwrap_or_else(|| PFrac::canon(an.mul(&bn), ad.mul(&bd)))
    }
}

impl Ring for PFrac {
    fn zero() -> Self {
        PFrac::Const(Q::zero())
    }

    fn one() -> Self {
        PFrac::Const(Q::one())
    }

    fn is_zero(&self) -> bool {
        matches!(self, PFrac::Const(q) if q.is_zero())
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
        self.scale_q(&-Q::one())
    }
}

impl EuclideanDomain for PFrac {
    fn div_rem(&self, other: &Self) -> (Self, Self) {
        (Field::div(self, other), Self::zero())
    }
}

impl Field for PFrac {
    /// Division; by zero it gives zero (the polynomial algorithms divide
    /// only by leading coefficients, which are never zero).
    fn div(&self, other: &Self) -> Self {
        self.mul(&other.inv())
    }

    /// The inverse; that of zero is zero (see [`Field::div`]).
    fn inv(&self) -> Self {
        match self {
            PFrac::Const(q) if q.is_zero() => PFrac::Const(Q::zero()),
            PFrac::Const(q) => PFrac::Const(q.recip()),
            PFrac::Frac(f) => PFrac::finish(f.1.clone(), f.0.clone()),
        }
    }
}

/// `p′` with the integer multiples taken by scaling (the generic
/// derivative adds a coefficient to itself `k − 1` times).
fn deriv(p: &GP) -> GP {
    let coeffs = p
        .coeffs()
        .iter()
        .enumerate()
        .skip(1)
        .map(|(k, c)| c.scale_q(&Q::from_integer(BigInt::from(k))))
        .collect();
    GP::from_coeffs(coeffs)
}

/// Exact quotient `a/b` over the field (`None` if `b` does not divide `a`).
fn div_exact(a: &GP, b: &GP) -> Option<GP> {
    let (q, r) = a.try_div_rem(b)?;
    r.is_zero().then_some(q)
}

/// A parameter-free polynomial as a [`Poly`] over `ℚ`.
fn to_q_poly(p: &GP) -> Option<Poly> {
    let coeffs = p
        .coeffs()
        .iter()
        .map(|c| c.as_const().cloned())
        .collect::<Option<Vec<Q>>>()?;
    Some(Poly::from_coeffs(coeffs))
}

fn from_q_poly(p: &Poly) -> GP {
    GP::from_coeffs(p.coeffs().iter().cloned().map(PFrac::Const).collect())
}

// ═══════════════════════════════════════════════════════════════════════════
// Square roots in ℚ(p₁, …)
// ═══════════════════════════════════════════════════════════════════════════

/// The exact square root of a multivariate polynomial with a positive
/// leading coefficient, or `None` if it is not a square: the root is
/// built term by term from the top (`S ← S + lt(P − S²)/(2·lt(S))`), each
/// new term below the previous one in the monomial order.
fn mp_sqrt(p: &MP) -> Option<MP> {
    let nv = p.num_vars();
    let (lt_exp, lt_c) = p.leading_term()?;
    if lt_exp.iter().any(|e| e % 2 == 1) {
        return None;
    }
    let root_c = q_sqrt(lt_c)?;
    let lead_exp: Vec<u32> = lt_exp.iter().map(|e| e / 2).collect();
    let two_lead = MP::monomial(&root_c + &root_c, lead_exp.clone());
    let mut s = MP::monomial(root_c, lead_exp);
    // The leading term of P − S² falls strictly in a graded order, so the
    // loop ends; the bound only caps the work.
    for _ in 0..64 + 4 * p.num_terms() {
        let r = p.sub(&s.mul(&s));
        let Some((r_exp, r_c)) = r.leading_term() else {
            return Some(s);
        };
        let (d_exp, d_c) = two_lead.leading_term()?;
        if r_exp.iter().zip(d_exp).any(|(a, b)| a < b) {
            return None;
        }
        let t_exp: Vec<u32> = r_exp.iter().zip(d_exp).map(|(a, b)| a - b).collect();
        if t_exp.iter().zip(d_exp).all(|(a, b)| a == b) {
            return None;
        }
        let t = MP::monomial(r_c / d_c, t_exp);
        s = s.add(&t);
        if s.num_vars() != nv {
            return None;
        }
    }
    None
}

/// A partial factorisation `p = ∏ sᵢ^kᵢ` of a denominator with integer
/// coefficients and positive leading coefficient (as [`PFrac`] keeps
/// them), or `None` when it finds no proper factor: the monomial content,
/// then a gcd-free basis of the rest refined by `gcd(f, ∂f/∂v)` and by
/// pairwise gcds, each element's multiplicity by repeated division, the
/// product checked against `p`.  For display, and for the integrator's
/// degenerate-case wrapper, whose `solve` handles `a·(a − b − 2)²` factor
/// by factor but not expanded.
fn factor_split(p: &MP) -> Option<Vec<(MP, u32)>> {
    let nv = p.num_vars();
    let mono = p.monomial_content();
    let mut rest = p.div_exact(&MP::monomial(Q::one(), mono.clone()))?;
    let mut basis: Vec<MP> = Vec::new();
    if rest.as_constant().is_none() {
        basis.push(rest.clone());
    }
    // Refine: split an element by a proper gcd with a derivative or with
    // another element, until nothing splits (bounded).
    for _ in 0..16 {
        let mut split: Option<(usize, MP)> = None;
        'find: for (i, f) in basis.iter().enumerate() {
            let deg_f = f.total_degree()?;
            for v in f.variables_present() {
                let g = MP::gcd(f, &f.partial_derivative(v));
                if g.total_degree().is_some_and(|d| d > 0 && d < deg_f) {
                    split = Some((i, g));
                    break 'find;
                }
            }
            for (j, h) in basis.iter().enumerate() {
                if i != j {
                    let g = MP::gcd(f, h);
                    if g.total_degree().is_some_and(|d| d > 0 && d < deg_f) {
                        split = Some((i, g));
                        break 'find;
                    }
                }
            }
        }
        let Some((i, g)) = split else { break };
        let f = basis.swap_remove(i);
        let cof = f.div_exact(&g)?;
        for piece in [g, cof] {
            if piece.as_constant().is_none() && !basis.contains(&piece) {
                basis.push(piece);
            }
        }
    }
    let mut factors: Vec<(MP, u32)> = Vec::new();
    for (v, &e) in mono.iter().enumerate() {
        if e > 0 {
            factors.push((MP::var(nv, v), e));
        }
    }
    for b in basis {
        let mut k = 0u32;
        while rest.as_constant().is_none()
            && let Some(q) = rest.div_exact(&b)
        {
            rest = q;
            k += 1;
        }
        if k > 0 {
            factors.push((b, k));
        }
    }
    if factors.len() < 2 && factors.iter().all(|(_, k)| *k == 1) {
        return None;
    }
    // What is left is a constant: fold it into the first factor's scale
    // only when it is 1 (the display keeps integer factors).
    let c = rest.as_constant()?;
    if !c.is_one() {
        return None;
    }
    let product = factors
        .iter()
        .fold(MP::from_int(nv, 1), |acc, (s, k)| acc.mul(&s.pow(*k)));
    (product == *p).then_some(factors)
}

/// The rational square root of a non-negative rational, if any.
fn q_sqrt(q: &Q) -> Option<Q> {
    if q.is_negative() {
        return None;
    }
    let n = int_sqrt(q.numer())?;
    let d = int_sqrt(q.denom())?;
    Some(Q::new(n, d))
}

fn int_sqrt(n: &BigInt) -> Option<BigInt> {
    let r = num_integer::Roots::sqrt(n);
    (&r * &r == *n).then_some(r)
}

/// `Δ = σ²·ρ` with `σ ∈ ℚ(p₁, …)` and a "square-free" radicand
/// `ρ ∈ ℚ[p₁, …]`: the square factors of the integer content (small
/// primes), of the monomial content, and a perfect-square remainder are
/// moved into `σ`.  `ρ = 1` exactly when `Δ` is recognised as a square.
fn split_square(delta: &PFrac, nv: usize) -> (PFrac, MP) {
    let (n, d) = delta.parts(nv);
    // Δ = n/d = (n·d)/d².
    let p = n.mul(&d);
    let mut sigma = PFrac::canon(MP::from_int(nv, 1), d);
    // Content: p = c·p1, p1 integer, primitive, positive leading coefficient.
    let (lcm, pz) = p.clear_denominators();
    let content = pz.integer_content();
    if content.is_zero() {
        return (PFrac::Const(Q::zero()), MP::from_int(nv, 1));
    }
    let negative = pz.leading_coeff().is_some_and(Signed::is_negative);
    let mut p1 = pz.scale(&Q::new(BigInt::one(), content.clone()));
    if negative {
        p1 = p1.neg();
    }
    // |c| = content/lcm = (content·lcm)/lcm².
    let mut w = &content * &lcm;
    let mut s = BigInt::one();
    for prime in [2u32, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let sq = BigInt::from(prime * prime);
        while (&w % &sq).is_zero() {
            w /= &sq;
            s *= prime;
        }
    }
    if let Some(r) = int_sqrt(&w) {
        s *= r;
        w = BigInt::one();
    }
    sigma = sigma.scale_q(&Q::new(s, lcm));
    // Monomial content.
    let mono = p1.monomial_content();
    if mono.iter().any(|&e| e > 0) {
        let half: Vec<u32> = mono.iter().map(|e| e / 2).collect();
        let odd: Vec<u32> = mono.iter().map(|e| e % 2).collect();
        if let Some(rest) = p1.div_exact(&MP::monomial(Q::one(), mono.clone())) {
            p1 = rest.mul(&MP::monomial(Q::one(), odd));
            sigma = sigma.mul(&PFrac::from_mp(MP::monomial(Q::one(), half)));
        }
    }
    // A perfect-square remainder (after taking the odd monomial part out
    // again it would not be one, so test the whole of p1).
    if let Some(root) = mp_sqrt(&p1) {
        sigma = sigma.mul(&PFrac::from_mp(root));
        p1 = MP::from_int(nv, 1);
    }
    let sign = if negative { -Q::one() } else { Q::one() };
    let rho = p1.scale(&Q::from_integer(w)).scale(&sign);
    (sigma, rho)
}

// ═══════════════════════════════════════════════════════════════════════════
// Conversions
// ═══════════════════════════════════════════════════════════════════════════

/// The integrand's view as a rational function in `var` over `ℚ[params]`.
struct Ctx {
    var: ExprId,
    params: Vec<ExprId>,
}

impl Ctx {
    fn nv(&self) -> usize {
        self.params.len()
    }

    /// `p ∈ ℚ[x, p₁, …]` (variable 0 is `x`) as a polynomial in `x`.
    fn to_gp(&self, p: &MP) -> Option<GP> {
        let nv = self.nv();
        let deg = p.degree_in(0) as usize;
        let mut buckets: Vec<Vec<(Vec<u32>, Q)>> = vec![Vec::new(); deg + 1];
        for (e, c) in p.terms() {
            buckets[e[0] as usize].push((e[1..].to_vec(), c.clone()));
        }
        let mut coeffs = Vec::with_capacity(deg + 1);
        for terms in buckets {
            let mp = if terms.is_empty() {
                MP::zero(nv)
            } else {
                MP::from_terms(nv, terms)?
            };
            coeffs.push(PFrac::from_mp(mp));
        }
        Some(GP::from_coeffs(coeffs))
    }

    /// `L·p ∈ ℚ[p₁, …][x]` as a polynomial in `(x, p₁, …)`, `L` the lcm of
    /// the coefficients' denominators.
    fn to_mp(&self, p: &GP) -> Option<MP> {
        let nv = self.nv();
        let mut l = MP::from_int(nv, 1);
        for c in p.coeffs() {
            if let PFrac::Frac(f) = c
                && f.1.as_constant().is_none()
            {
                l = MP::lcm(&l, &f.1);
            }
        }
        let mut terms: Vec<(Vec<u32>, Q)> = Vec::new();
        for (k, c) in p.coeffs().iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let (n, d) = c.parts(nv);
            let scaled = n.mul(&l.div_exact(&d)?);
            let k = u32::try_from(k).ok()?;
            for (e, q) in scaled.terms() {
                let mut exps = Vec::with_capacity(nv + 1);
                exps.push(k);
                exps.extend_from_slice(e);
                terms.push((exps, q.clone()));
            }
        }
        MP::from_terms(nv + 1, terms)
    }

    /// The monic gcd in `ℚ(p₁, …)[x]`, computed as the gcd in
    /// `ℤ[x, p₁, …]` of the numerators ([`MultiPoly::gcd`]): the two differ
    /// by the content in `x`, a unit of `ℚ(p₁, …)`.  Euclid over the field
    /// took minutes where the heuristic gcd takes milliseconds (every
    /// field operation normalises its coefficients by a gcd of its own).
    fn gcd(&self, a: &GP, b: &GP) -> Option<GP> {
        if a.is_zero() {
            return Some(b.make_monic());
        }
        if b.is_zero() {
            return Some(a.make_monic());
        }
        let (am, bm) = (self.to_mp(a)?, self.to_mp(b)?);
        if !gcd_affordable(&am, &bm) {
            TOO_BIG.with(|c| c.set(true));
            return None;
        }
        let g = MP::gcd(&am, &bm);
        Some(self.to_gp(&g)?.make_monic())
    }

    fn mp_expr(&self, arena: &mut Arena, p: &MP) -> ExprId {
        crate::poly::polybridge::multipoly_to_expr(arena, p, &self.params)
    }

    fn coeff_expr(&self, arena: &mut Arena, c: &PFrac) -> ExprId {
        match c {
            PFrac::Const(q) => arena.num_ratio(q.clone()),
            PFrac::Frac(f) => {
                let n = self.mp_expr(arena, &f.0);
                if f.1.as_constant().is_some_and(|d| d.is_one()) {
                    n
                } else {
                    let d = match factor_split(&f.1) {
                        Some(factors) => {
                            let mut ids = Vec::with_capacity(factors.len());
                            for (s, k) in factors {
                                let s_id = self.mp_expr(arena, &s);
                                ids.push(if k == 1 {
                                    s_id
                                } else {
                                    let k_id = arena.int(i64::from(k));
                                    arena.pow(s_id, k_id)
                                });
                            }
                            arena.mul(&ids)
                        }
                        None => self.mp_expr(arena, &f.1),
                    };
                    arena.div(n, d)
                }
            }
        }
    }

    fn poly_expr(&self, arena: &mut Arena, p: &GP) -> ExprId {
        let mut terms = Vec::new();
        for (k, c) in p.coeffs().iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let c_id = self.coeff_expr(arena, c);
            let x_k = self.x_pow(arena, k);
            terms.push(arena.mul(&[c_id, x_k]));
        }
        match terms.len() {
            0 => arena.zero(),
            1 => terms[0],
            _ => arena.add(&terms),
        }
    }

    fn x_pow(&self, arena: &mut Arena, k: usize) -> ExprId {
        match k {
            0 => arena.one(),
            1 => self.var,
            _ => {
                let k_id = arena.int(i64::try_from(k).unwrap_or(i64::MAX));
                arena.pow(self.var, k_id)
            }
        }
    }

    /// Are all parameters occurring in `p`'s coefficients declared real?
    fn declared_real(&self, arena: &Arena, p: &GP) -> bool {
        let nv = self.nv();
        p.coeffs().iter().all(|c| {
            let (n, d) = c.parts(nv);
            let mut vars = n.variables_present();
            vars.extend(d.variables_present());
            vars.into_iter().all(|i| match arena.node(self.params[i]) {
                ExprNode::Symbol(s) => crate::transforms::realness::symbol_declared_real(arena, *s),
                _ => false,
            })
        })
    }
}

/// The parameters of `expr` (its symbols other than `var`, sorted), when
/// `expr` is built from rational numbers, `var` and at most
/// [`MAX_PARAMS`] parameters by sums, products and integer powers; `None`
/// otherwise or when there is no parameter.
fn parameters(arena: &Arena, expr: ExprId, var: ExprId) -> Option<Vec<ExprId>> {
    let mut params = Vec::new();
    let mut has_var = false;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        match arena.node(id) {
            ExprNode::Num(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => {}
            ExprNode::Pow(_, exp) => {
                let ok = arena.as_num(*exp).is_some_and(|r| {
                    r.is_integer()
                        && i64::try_from(r.to_integer()).is_ok_and(|n| n.abs() <= MAX_EXPONENT)
                });
                if !ok {
                    return None;
                }
            }
            ExprNode::Symbol(_) if id == var => has_var = true,
            ExprNode::Symbol(_) => params.push(id),
            _ => return None,
        }
    }
    params.sort_by_key(|id| id.0);
    params.dedup();
    (has_var && !params.is_empty() && params.len() <= MAX_PARAMS).then_some(params)
}

/// Numerator and denominator of `expr` over a common denominator, nested
/// fractions flattened; the denominator keeps its factors as written.
fn numer_denom(arena: &mut Arena, expr: ExprId) -> (ExprId, ExprId) {
    crate::poly::polybridge::fraction_parts(arena, expr)
}

/// The factors of the denominator as written (`(x + a)²·(x + b)` gives
/// `x + a` and `x + b`), as polynomials in `x`: the starting point of the
/// coprime factorisation.
fn syntactic_factors(arena: &Arena, d: ExprId, ctx: &Ctx) -> Vec<GP> {
    let mut vars = vec![ctx.var];
    vars.extend(&ctx.params);
    let factors: Vec<ExprId> = match arena.node(d) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![d],
    };
    let mut out = Vec::new();
    for f in factors {
        let base = match arena.node(f) {
            ExprNode::Pow(b, _) => *b,
            _ => f,
        };
        if let Some(mp) = crate::poly::polybridge::expr_to_multipoly(arena, base, &vars)
            && let Some(p) = ctx.to_gp(&mp)
            && p.degree().is_some_and(|k| k >= 1)
        {
            out.push(p.make_monic());
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry point
// ═══════════════════════════════════════════════════════════════════════════

/// `∫ expr d(var)` for a rational function of `var` whose coefficients are
/// polynomials in other symbols (see the module docs), or `None` when
/// `expr` is not of that shape, is a sum (integrated termwise by the
/// caller), is a polynomial in `var` over one linear factor (left to the
/// `(a·x + b)ⁿ` routes), exceeds the size limits, or has a
/// logarithmic part this route cannot write.  The caller checks the
/// answer by differentiation.
pub(crate) fn integrate_param_rational(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    if matches!(arena.node(expr), ExprNode::Add(_)) {
        return None;
    }
    let params = parameters(arena, expr, var)?;
    tracing::debug!(integrand = %arena.display(expr), "param_rational: attempt");
    reset_budget();
    let ctx = Ctx { var, params };
    let (n_id, d_id) = numer_denom(arena, expr);
    if d_id == arena.one() {
        return None;
    }
    let mut vars = vec![var];
    vars.extend(&ctx.params);
    let n_mp = crate::poly::polybridge::expr_to_multipoly(arena, n_id, &vars)?;
    let d_mp = crate::poly::polybridge::expr_to_multipoly(arena, d_id, &vars)?;
    let mut a = ctx.to_gp(&n_mp)?;
    let mut d = ctx.to_gp(&d_mp)?;
    if d.degree().is_none_or(|k| k == 0) || a.is_zero() {
        return None;
    }
    if a.degree().is_some_and(|k| k > MAX_NUMER_DEGREE)
        || d.degree().is_some_and(|k| k > 2 * MAX_DENOM_DEGREE)
    {
        return None;
    }
    let numerator_param_free = a.coeffs().iter().all(|c| c.as_const().is_some());
    let g = ctx.gcd(&a, &d)?;
    if g.degree().is_some_and(|k| k > 0) {
        a = div_exact(&a, &g)?;
        d = div_exact(&d, &g)?;
    }
    if d.degree()? > MAX_DENOM_DEGREE {
        return None;
    }
    // Monic denominator.
    let lc = d.leading_coeff()?.clone();
    let inv = lc.inv();
    a = a.scale(&inv);
    d = d.scale(&inv);
    if d.degree()? == 0 {
        // A polynomial after cancellation.
        let big_q = integrate_poly(&a);
        return Some(ctx.poly_expr(arena, &big_q));
    }
    // One linear factor, written as one, with a numerator in ℚ[x]
    // (`1/(a·x + 1)`, `x/(a + b·x)²`): the `(a·x + b)ⁿ` routes keep their
    // forms.  (`(2a·x − 2)⁻²·(1 − a·x)⁻¹` is written with two,
    // `1/((a + 1)/x + 1)` as a nested fraction, and `x/(x² + 2a·x + a²)`
    // multiplied out, which those routes miss: up to 0.31 the last stayed
    // unevaluated while `x/(x + a)²` worked.)
    let cands = syntactic_factors(arena, d_id, &ctx);
    let d_sqf = div_exact(&d, &ctx.gcd(&d, &deriv(&d))?)?;
    if d_sqf.degree()? < 2
        && numerator_param_free
        && cands.len() <= 1
        && cands.iter().all(|c| c.degree() == Some(1))
        && !has_nested_fraction(arena, expr)
    {
        return None;
    }
    if let Some(k) = power_substitution_degree(&a, &d) {
        return integrate_substituted(arena, &ctx, &a, &d, &cands, k, expr);
    }
    let result = integrate_gp(arena, &ctx, &a, &d, &cands, expr)?;
    Some(crate::transforms::eval::eval(arena, result))
}

/// Does `e` have a sum with a negative power inside it (`(a + 1)/x + 1`)?
fn has_nested_fraction(arena: &Arena, e: ExprId) -> bool {
    let mut with_negative_power = rustc_hash::FxHashSet::default();
    for id in crate::base::walk::post_order_ids(arena, e) {
        let node = arena.node(id);
        let here = matches!(node, ExprNode::Pow(_, exp)
            if arena.as_num(*exp).is_some_and(|r| r.is_negative()));
        let mut below = false;
        node.for_each_child(|c| below |= with_negative_power.contains(&c));
        if below && matches!(node, ExprNode::Add(_)) {
            return true;
        }
        if here || below {
            with_negative_power.insert(id);
        }
    }
    false
}

/// `k ≥ 2` if `A/D = x^(k−1)·F(x^k)` (as in the `ℚ` integrator).
fn power_substitution_degree(a: &GP, d: &GP) -> Option<usize> {
    let mut g = 0usize;
    for (e, c) in d.coeffs().iter().enumerate() {
        if !c.is_zero() {
            g = num_integer::gcd(g, e);
        }
    }
    for (e, c) in a.coeffs().iter().enumerate() {
        if !c.is_zero() {
            g = num_integer::gcd(g, e + 1);
        }
    }
    (g >= 2).then_some(g)
}

fn compress(p: &GP, k: usize, shift: usize) -> GP {
    GP::from_coeffs(p.coeffs().iter().skip(shift).step_by(k).cloned().collect())
}

/// `∫ x^(k−1)·F(x^k) dx = (1/k)·∫ F(u) du` at `u = x^k`.
fn integrate_substituted(
    arena: &mut Arena,
    ctx: &Ctx,
    a: &GP,
    d: &GP,
    cands: &[GP],
    k: usize,
    label: ExprId,
) -> Option<ExprId> {
    let k_q = Q::from_integer(BigInt::from(k));
    let a_u = compress(a, k, k - 1).scale(&PFrac::Const(k_q.recip()));
    let d_u = compress(d, k, 0);
    let cands_u: Vec<GP> = cands
        .iter()
        .filter(|c| {
            c.coeffs()
                .iter()
                .enumerate()
                .all(|(e, x)| x.is_zero() || e % k == 0)
        })
        .map(|c| compress(c, k, 0))
        .collect();
    let u = arena.symbol("__prr_u");
    let ctx_u = Ctx {
        var: u,
        params: ctx.params.clone(),
    };
    let f_u = integrate_gp(arena, &ctx_u, &a_u, &d_u, &cands_u, label)?;
    let k_id = arena.int(i64::try_from(k).ok()?);
    let x_k = arena.pow(ctx.var, k_id);
    let f_x = crate::transforms::subs::subs(arena, f_u, u, x_k);
    Some(crate::transforms::eval::eval(arena, f_x))
}

fn integrate_poly(p: &GP) -> GP {
    let mut coeffs = vec![PFrac::zero()];
    for (k, c) in p.coeffs().iter().enumerate() {
        coeffs.push(c.scale_q(&Q::new(BigInt::one(), BigInt::from(k + 1))));
    }
    GP::from_coeffs(coeffs)
}

/// `∫ a/d` for a monic `d` of positive degree.
fn integrate_gp(
    arena: &mut Arena,
    ctx: &Ctx,
    a: &GP,
    d: &GP,
    cands: &[GP],
    label: ExprId,
) -> Option<ExprId> {
    let (q, r) = a.try_div_rem(d)?;
    let mut terms: Vec<ExprId> = Vec::new();
    if !q.is_zero() {
        let big_q = integrate_poly(&q);
        terms.push(ctx.poly_expr(arena, &big_q));
    }
    if !r.is_zero() {
        let (rational, h, d_star) = hermite(ctx, &r, d)?;
        if budget_exceeded() {
            return None;
        }
        let basis = coprime_basis(ctx, &d_star, cands)?;
        for (b, dm) in rational {
            let (dm_id, lambda) = factored_expr(arena, ctx, &dm, &basis);
            let b_id = ctx.poly_expr(arena, &b.scale(&lambda));
            terms.push(arena.div(b_id, dm_id));
        }
        if !h.is_zero() {
            terms.extend(log_part(arena, ctx, &h, &d_star, &basis, label)?);
        }
    }
    if budget_exceeded() {
        return None;
    }
    Some(match terms.len() {
        0 => arena.zero(),
        1 => terms[0],
        _ => arena.add(&terms),
    })
}

/// Hermite reduction, Mack's linear version (Bronstein, *Symbolic
/// Integration I*, §2.2, `HermiteReduce`): `∫ a/d = Σ Bᵢ/D⁻ᵢ + ∫ h/d*`
/// with `d* = d/gcd(d, d′)` square-free.  Returns the pairs `(Bᵢ, D⁻ᵢ)`,
/// `h` and `d*`.
type HermiteParts = (Vec<(GP, GP)>, GP, GP);

fn hermite(ctx: &Ctx, a: &GP, d: &GP) -> Option<HermiteParts> {
    let mut d_minus = ctx.gcd(d, &deriv(d))?;
    let d_star = div_exact(d, &d_minus)?;
    let mut a = a.clone();
    let mut out = Vec::new();
    while d_minus.degree()? > 0 {
        if budget_exceeded() {
            return None;
        }
        let dm_p = deriv(&d_minus);
        let d_minus2 = ctx.gcd(&d_minus, &dm_p)?;
        let d_minus_star = div_exact(&d_minus, &d_minus2)?;
        let t = div_exact(&d_star.mul(&dm_p), &d_minus)?.neg();
        let (b, c) = solve_bezout(&t, &d_minus_star, &a)?;
        let bp = deriv(&b);
        a = c.sub(&div_exact(&bp.mul(&d_star), &d_minus_star)?);
        if !b.is_zero() {
            out.push((b, d_minus.clone()));
        }
        d_minus = d_minus2;
    }
    Some((out, a, d_star))
}

/// `(s, t)` with `s·f + t·g = c` and `deg s < deg g`, for coprime `f`, `g`
/// (Bronstein's `ExtendedEuclidean`).
fn solve_bezout(f: &GP, g: &GP, c: &GP) -> Option<(GP, GP)> {
    let e = GP::extended_gcd(f, g);
    if e.gcd.degree()? != 0 {
        return None;
    }
    let inv = e.gcd.leading_coeff()?.inv();
    let s0 = e.x.scale(&inv).mul(c);
    let t0 = e.y.scale(&inv).mul(c);
    let (q, s) = s0.try_div_rem(g)?;
    let t = t0.add(&q.mul(f));
    Some((s, t))
}

/// Largest number of terms of a parametric basis element that
/// [`split_parametric`] factors.
const MAX_SPLIT_TERMS: usize = 40;

/// The factors of positive degree in `x` of a parametric basis element
/// `e` of degree ≥ 3 (monic, square-free) in `ℚ[x, p₁, …]`
/// ([`crate::poly::factor_zassenhaus::factor_multivariate`]: the monomial
/// content, then Kronecker's substitution), each made monic in `x`; `None`
/// when `e` does not split.  Only degree ≥ 3 is factored: a parametric
/// quadratic is split by its discriminant in [`quadratic_log_part`], and
/// a larger factor's logarithmic part is otherwise written only when
/// `aₑ = c·e′` ([`higher_log_part`]).  Up to 0.31 the denominator
/// `x³ + a·x` was one such factor, so `∫ (x² + 1)/(x³ + a·x) dx` stayed
/// unevaluated while `∫ (x² + 1)/(x·(x² + a)) dx` worked.
fn split_parametric(ctx: &Ctx, e: &GP) -> Option<Vec<GP>> {
    let mp = ctx.to_mp(e)?;
    if mp.num_terms() > MAX_SPLIT_TERMS {
        return None;
    }
    let (_, factors) = crate::poly::factor_zassenhaus::factor_multivariate(&mp)?;
    let pieces = factors
        .iter()
        .filter(|(f, _)| f.degree_in(0) > 0)
        .map(|(f, _)| ctx.to_gp(f).map(|g| g.make_monic()))
        .collect::<Option<Vec<GP>>>()?;
    (pieces.len() >= 2).then_some(pieces)
}

/// A coprime factorisation of the square-free, monic `d`: `d` split by
/// gcds with the candidate factors, parameter-free pieces split over `ℚ`,
/// parametric pieces of degree ≥ 3 by [`split_parametric`].
fn coprime_basis(ctx: &Ctx, d: &GP, cands: &[GP]) -> Option<Vec<GP>> {
    let mut basis = vec![d.make_monic()];
    for c in cands {
        let mut next = Vec::with_capacity(basis.len() + 1);
        for e in basis {
            let g = ctx.gcd(&e, c)?;
            match (g.degree(), e.degree()) {
                (Some(dg), Some(de)) if dg > 0 && dg < de => {
                    let rest = div_exact(&e, &g)?;
                    next.push(g.make_monic());
                    next.push(rest.make_monic());
                }
                _ => next.push(e),
            }
        }
        basis = next;
        if budget_exceeded() {
            return None;
        }
    }
    let mut out = Vec::with_capacity(basis.len());
    for e in basis {
        match to_q_poly(&e) {
            Some(p) if e.degree().is_some_and(|k| k >= 2) => {
                let (_, factors) = p.factor_over_z();
                let total: usize = factors
                    .iter()
                    .map(|(f, m)| f.degree().unwrap_or(0) * *m as usize)
                    .sum();
                if factors.is_empty() || total != p.degree().unwrap_or(0) {
                    out.push(e);
                } else {
                    for (f, _) in factors {
                        out.push(from_q_poly(&f).make_monic());
                    }
                }
            }
            None if e.degree().is_some_and(|k| k >= 3) => match split_parametric(ctx, &e) {
                Some(pieces) => out.extend(pieces),
                None => out.push(e),
            },
            _ => out.push(e),
        }
    }
    Some(out)
}

/// `p` (monic, a product of powers of basis elements) written as `λ⁻¹`
/// times a product of powers of polynomials with coefficients in
/// `ℚ[p₁, …]` (`x + a/b` as `b·x + a`): returns the product and `λ`, or
/// `p` expanded and `λ = 1` when `p` is not such a product.
fn factored_expr(arena: &mut Arena, ctx: &Ctx, p: &GP, basis: &[GP]) -> (ExprId, PFrac) {
    let mut rest = p.clone();
    let mut factors: Vec<ExprId> = Vec::new();
    let mut lambda = PFrac::one();
    for e in basis {
        let mut m = 0u32;
        while rest.degree() >= e.degree()
            && let Some(q) = div_exact(&rest, e)
        {
            rest = q;
            m += 1;
        }
        if m > 0 {
            // e = raw/l with l the lcm of its coefficients' denominators.
            let l = e
                .coeffs()
                .iter()
                .fold(MP::from_int(ctx.nv(), 1), |acc, c| match c {
                    PFrac::Frac(f) if f.1.as_constant().is_none() => {
                        if gcd_affordable(&acc, &f.1) {
                            MP::lcm(&acc, &f.1)
                        } else {
                            acc.mul(&f.1)
                        }
                    }
                    _ => acc,
                });
            let l = PFrac::from_mp(l);
            let raw = e.scale(&l);
            let e_id = ctx.poly_expr(arena, &raw);
            lambda = lambda.mul(&l.pow_usize(m as usize));
            factors.push(if m == 1 {
                e_id
            } else {
                let m_id = arena.int(i64::from(m));
                arena.pow(e_id, m_id)
            });
        }
    }
    if rest.degree() != Some(0) || !rest.coeff(0).is_one() || factors.is_empty() {
        return (ctx.poly_expr(arena, p), PFrac::one());
    }
    let product = if factors.len() == 1 {
        factors[0]
    } else {
        arena.mul(&factors)
    };
    (product, lambda)
}

/// `ln|u|`.
fn ln_abs(arena: &mut Arena, u: ExprId) -> ExprId {
    let a = arena.abs(u);
    arena.ln(a)
}

/// The logarithmic part of `∫ h/d` (`d` square-free, monic, `deg h <
/// deg d`) over the coprime factorisation `basis` of `d`.
fn log_part(
    arena: &mut Arena,
    ctx: &Ctx,
    h: &GP,
    d: &GP,
    basis: &[GP],
    label: ExprId,
) -> Option<Vec<ExprId>> {
    let mut terms = Vec::new();
    for e in basis {
        if budget_exceeded() {
            return None;
        }
        // Partial-fraction numerator: h·(d/e)⁻¹ mod e.
        let cof = div_exact(d, e)?;
        let eg = GP::extended_gcd(&cof, e);
        if eg.gcd.degree()? != 0 {
            return None;
        }
        let inv = eg.gcd.leading_coeff()?.inv();
        let (_, a_e) = h.mul(&eg.x.scale(&inv)).try_div_rem(e)?;
        if a_e.is_zero() {
            continue;
        }
        match e.degree()? {
            1 => {
                let e_id = ctx.poly_expr(arena, e);
                let c = ctx.coeff_expr(arena, &a_e.coeff(0));
                let ln = ln_abs(arena, e_id);
                terms.push(arena.mul(&[c, ln]));
            }
            2 => terms.extend(quadratic_log_part(arena, ctx, &a_e, e)?),
            _ => terms.push(higher_log_part(arena, ctx, &a_e, e, label)?),
        }
    }
    Some(terms)
}

/// `σ·√ρ` as an expression (`ρ = ±1` give `σ`, `σ·i`).
fn sqrt_expr(arena: &mut Arena, ctx: &Ctx, sigma: &PFrac, rho: &MP) -> ExprId {
    let s = ctx.coeff_expr(arena, sigma);
    match rho.as_constant() {
        Some(c) if c.is_one() => s,
        Some(c) if (-&c).is_one() => {
            let i = arena.i_unit();
            arena.mul(&[s, i])
        }
        _ => {
            let r = ctx.mp_expr(arena, rho);
            let root = arena.sqrt(r);
            arena.mul(&[s, root])
        }
    }
}

/// `∫ (μx + ν)/(x² + Bx + C)` for a square-free monic quadratic factor.
fn quadratic_log_part(arena: &mut Arena, ctx: &Ctx, a_e: &GP, e: &GP) -> Option<Vec<ExprId>> {
    let nv = ctx.nv();
    let (mu, nu) = (a_e.coeff(1), a_e.coeff(0));
    let (big_b, big_c) = (e.coeff(1), e.coeff(0));
    let half = Q::new(BigInt::one(), BigInt::from(2));
    let half_mu = mu.scale_q(&half);
    // κ = ν − μB/2: c± = μ/2 ± κ/√Δ.
    let kappa = nu.sub(&half_mu.mul(&big_b));
    let e_id = ctx.poly_expr(arena, e);
    if kappa.is_zero() {
        let c = ctx.coeff_expr(arena, &half_mu);
        let ln = ln_abs(arena, e_id);
        return Some(vec![arena.mul(&[c, ln])]);
    }
    let delta = big_b
        .mul(&big_b)
        .sub(&big_c.scale_q(&Q::from_integer(BigInt::from(4))));
    if delta.is_zero() {
        return None;
    }
    let (sigma, rho) = split_square(&delta, nv);
    if rho.as_constant().is_some_and(|c| c.is_one()) {
        // Rational roots r± = (−B ± σ)/2, residues (μr + ν)/(±σ).
        let mut terms = Vec::with_capacity(2);
        for sign in [Q::one(), -Q::one()] {
            let r = big_b.neg().add(&sigma.scale_q(&sign)).scale_q(&half);
            let res = mu.mul(&r).add(&nu).div(&sigma.scale_q(&sign));
            if res.is_zero() {
                continue;
            }
            let lin = GP::from_coeffs(vec![r.neg(), PFrac::one()]);
            let lin_id = ctx.poly_expr(arena, &lin);
            let c = ctx.coeff_expr(arena, &res);
            let ln = ln_abs(arena, lin_id);
            terms.push(arena.mul(&[c, ln]));
        }
        return Some(terms);
    }
    let b_id = ctx.coeff_expr(arena, &big_b);
    let two = arena.int(2);
    let mut terms = Vec::with_capacity(2);
    if !half_mu.is_zero() {
        let c = ctx.coeff_expr(arena, &half_mu);
        let ln = ln_abs(arena, e_id);
        terms.push(arena.mul(&[c, ln]));
    }
    let two_kappa = ctx.coeff_expr(arena, &kappa.scale_q(&Q::from_integer(BigInt::from(2))));
    if ctx.declared_real(arena, e) && nonnegative(arena, ctx, &delta.neg()) {
        // Real, irreducible over ℝ: (μ/2)·ln|e| + (2κ/k)·atan((2x + B)/k),
        // k = √(−Δ).
        let (sigma_k, rho_k) = split_square(&delta.neg(), nv);
        let k = sqrt_expr(arena, ctx, &sigma_k, &rho_k);
        let two_x = arena.mul(&[two, ctx.var]);
        let lin = arena.add(&[two_x, b_id]);
        let arg = arena.div(lin, k);
        let atan = arena.atan(arg);
        let coeff = arena.div(two_kappa, k);
        terms.push(arena.mul(&[coeff, atan]));
        return Some(terms);
    }
    // (κ/s)·(ln|x − r₊| − ln|x − r₋|), r± = (−B ± s)/2, s = √Δ.
    let s = sqrt_expr(arena, ctx, &sigma, &rho);
    let neg_b = arena.neg(b_id);
    let mut lns = Vec::with_capacity(2);
    for sign in [1i64, -1] {
        let sgn = arena.int(sign);
        let ss = arena.mul(&[sgn, s]);
        let num = arena.add(&[neg_b, ss]);
        let r = arena.div(num, two);
        let neg_r = arena.neg(r);
        let lin = arena.add(&[ctx.var, neg_r]);
        lns.push(ln_abs(arena, lin));
    }
    let neg_minus = arena.neg(lns[1]);
    let diff = arena.add(&[lns[0], neg_minus]);
    let kappa_id = ctx.coeff_expr(arena, &kappa);
    let coeff = arena.div(kappa_id, s);
    terms.push(arena.mul(&[coeff, diff]));
    Some(terms)
}

/// Is `c ∈ ℚ(p₁, …)` non-negative for all real values of its parameters,
/// under their declared assumptions (which make them real)?  `c = σ²·ρ`
/// ([`split_square`]) with `σ` real, so it is decided on `ρ` by the
/// assumption system (which proves `4a² + 3b² ≥ 0` for real `a`, `b` since
/// 0.31; before, this route divided out the integer content of `ρ` to get
/// `a²` from `4a²`).
fn nonnegative(arena: &mut Arena, ctx: &Ctx, c: &PFrac) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    if let Some(q) = c.as_const() {
        return !q.is_negative();
    }
    let (_, rho) = split_square(c, ctx.nv());
    if let Some(q) = rho.as_constant() {
        return !q.is_negative();
    }
    let e = ctx.mp_expr(arena, &rho);
    let mut cache = AssumptionCache::new();
    cache.query(arena, e, Props::NONNEGATIVE) == Some(true)
}

/// `∫ aₑ/e` for a factor of degree ≥ 3: over `ℚ` coefficientwise when `e`
/// has no parameters, else the single logarithm `c·ln|e|` when
/// `aₑ = c·e′`, and otherwise the sum over the roots of `e`
/// ([`root_sum_log_part`]).
fn higher_log_part(
    arena: &mut Arena,
    ctx: &Ctx,
    a_e: &GP,
    e: &GP,
    label: ExprId,
) -> Option<ExprId> {
    let ep = deriv(e);
    let c = a_e.leading_coeff()?.div(ep.leading_coeff()?);
    if a_e.degree() == ep.degree() && ep.scale(&c) == *a_e {
        let e_id = ctx.poly_expr(arena, e);
        let c_id = ctx.coeff_expr(arena, &c);
        let ln = ln_abs(arena, e_id);
        return Some(arena.mul(&[c_id, ln]));
    }
    let Some(e_q) = to_q_poly(e) else {
        return root_sum_log_part(arena, ctx, a_e, e);
    };
    let mut terms = Vec::new();
    for (j, cj) in a_e.coeffs().iter().enumerate() {
        if cj.is_zero() {
            continue;
        }
        let mut xj = vec![Q::zero(); j + 1];
        xj[j] = Q::one();
        let f = super::integrate_rational_function(
            arena,
            &Poly::from_coeffs(xj),
            &e_q,
            ctx.var,
            label,
        )?;
        if crate::base::walk::has_unevaluated(arena, f) {
            return None;
        }
        let c_id = ctx.coeff_expr(arena, cj);
        terms.push(arena.mul(&[c_id, f]));
    }
    Some(match terms.len() {
        0 => arena.zero(),
        1 => terms[0],
        _ => arena.add(&terms),
    })
}

/// `Σ_{e(r) = 0} c(r)·ln(x − r)` as a `RootSum` for a parametric factor
/// `e` (monic, square-free, of degree ≥ 3, not split by
/// [`split_parametric`]), with `c = aₑ/e′ mod e` the residue of `aₑ/e` at
/// the root `r` (Bronstein, *Symbolic Integration I*, §2.5: the residues
/// are the roots of the Rothstein–Trager resultant `resₓ(e, aₑ − t·e′)`,
/// and `x − r` is the logarithm's argument; SymPy's `ratint` writes the
/// same sum over the roots `t = c(r)` of that resultant).  Its derivative
/// is `Σ c(r)/(x − r) = aₑ/e` by partial fractions.  Up to 0.31
/// `∫ dx/(x³ + a)` stayed unevaluated (SymPy: `RootSum(27t³a² − 1,
/// t ↦ t·ln(3ta + x))`; here `RootSum(r³ + a, r ↦ −r·ln(x − r)/(3a))`).
fn root_sum_log_part(arena: &mut Arena, ctx: &Ctx, a_e: &GP, e: &GP) -> Option<ExprId> {
    let ep = deriv(e);
    let eg = GP::extended_gcd(&ep, e);
    if eg.gcd.degree()? != 0 {
        return None;
    }
    let inv = eg.gcd.leading_coeff()?.inv();
    let (_, c) = a_e.mul(&eg.x.scale(&inv)).try_div_rem(e)?;
    if budget_exceeded() || c.is_zero() {
        return None;
    }
    let r = arena.symbol("__rs_t");
    let ctx_r = Ctx {
        var: r,
        params: ctx.params.clone(),
    };
    // The polynomial with its coefficients' denominators cleared.
    let (e_r, _) = factored_expr(arena, &ctx_r, e, std::slice::from_ref(e));
    let c_r = ctx_r.poly_expr(arena, &c);
    let neg_r = arena.neg(r);
    let lin = arena.add(&[ctx.var, neg_r]);
    let ln = arena.ln(lin);
    let body = arena.mul(&[c_r, ln]);
    Some(arena.intern(ExprNode::RootSum(e_r, body, r)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mp(nv: usize, terms: &[(&[u32], i64)]) -> MP {
        MP::from_terms(
            nv,
            terms
                .iter()
                .map(|(e, c)| (e.to_vec(), Q::from_integer(BigInt::from(*c))))
                .collect(),
        )
        .expect("valid terms")
    }

    #[test]
    fn pfrac_is_canonical() {
        // (a² − b²)/(a − b) = a + b; 1/(2a) + 1/(2a) = 1/a.
        let num = mp(2, &[(&[2, 0], 1), (&[0, 2], -1)]);
        let den = mp(2, &[(&[1, 0], 1), (&[0, 1], -1)]);
        let q = PFrac::canon(num, den);
        assert_eq!(q, PFrac::from_mp(mp(2, &[(&[1, 0], 1), (&[0, 1], 1)])));
        let half_inv = PFrac::canon(mp(2, &[(&[0, 0], 1)]), mp(2, &[(&[1, 0], 2)]));
        let inv = PFrac::canon(mp(2, &[(&[0, 0], 1)]), mp(2, &[(&[1, 0], 1)]));
        assert_eq!(half_inv.add(&half_inv), inv);
        assert!(inv.sub(&inv).is_zero());
        assert_eq!(inv.mul(&inv.inv()), PFrac::one());
    }

    #[test]
    fn multipoly_square_roots() {
        // (a − 2b)² = a² − 4ab + 4b².
        let sq = mp(2, &[(&[2, 0], 1), (&[1, 1], -4), (&[0, 2], 4)]);
        let root = mp_sqrt(&sq).expect("a square");
        assert_eq!(root.mul(&root), sq);
        assert!(mp_sqrt(&mp(2, &[(&[2, 0], 1), (&[0, 1], -4)])).is_none());
        // −4/a² = (2/a)²·(−1).
        let delta = PFrac::canon(mp(1, &[(&[0], -4)]), mp(1, &[(&[2], 1)]));
        let (sigma, rho) = split_square(&delta, 1);
        assert_eq!(rho.as_constant(), Some(-Q::one()));
        assert_eq!(sigma.mul(&sigma).neg(), delta);
    }
}
