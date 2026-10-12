//! Exact zero test of constants built from trigonometric values at rational
//! multiples of π, roots of unity and radicals — the relations between
//! constants that `eval` decides ([`is_zero`]).
//!
//! # The field
//!
//! `sin`, `cos`, `tan` at `qπ` (`q` rational), `i`, `(−1)^q` and `i^q` are
//! elements of the cyclotomic field `ℚ(ζ)`, `ζ = e^{2πi/N}`, for an `N` the
//! denominators fix: `cos(qπ) = (ζᵏ + ζ⁻ᵏ)/2` and `sin(qπ) = −i·(ζᵏ −
//! ζ⁻ᵏ)/2` with `k = qN/2`, `i = ζ^{N/4}`.  An element is the vector of its
//! `φ(N)` rational coordinates on `1, ζ, …, ζ^{φ(N)−1}` after reduction
//! modulo the cyclotomic polynomial `Φ_N`, which is irreducible: the
//! representation is unique, and 0 is the zero vector.  Relations between
//! the values — `cos(π/7)·cos(2π/7)·cos(4π/7) = −1/8`, `cos 3x = 4cos³x −
//! 3cos x` — hold coordinate by coordinate.
//!
//! # The radicals
//!
//! A radical `b^(1/m)` (`m = 2, 3`; `b^(p/q)` for `q = 2ᵃ3ᶜ` is a chain of
//! them: principal roots compose to the principal root) is a new generator
//! `s` with `sᵐ = r` for an element `r` of the field built so far — the
//! tower `ℚ(ζ)[s₁, …, s_k]`, `s_jᵐʲ = r_j`, whose elements are polynomials
//! in the `s_j` of degrees below `m_j` with coefficients in `ℚ(ζ)`.  Sending
//! each `s_j` to its value is a ring homomorphism to ℂ, but not injective:
//! no relation between the generators is known (`√2` and `ζ₈ + ζ₈⁻¹` are
//! not identified), so an element that is not the zero polynomial may have
//! the value 0.  The value is decided by the norm (Cohen, *A Course in
//! Computational Algebraic Number Theory*, §4.3): for `e = Σ u_j·sʲ` over
//! the top generator `s`, `N(e) = ∏_ω e(ω·s)` over the `m`-th roots of unity
//! `ω` is an element one level down (`u₀² − u₁²·r` for `m = 2`, `u₀³ +
//! u₁³·r + u₂³·r² − 3u₀u₁u₂·r` for `m = 3`), and the value of `e` is one of
//! its factors.  A nonzero norm proves `e ≠ 0`; a zero norm (decided the
//! same way one level down) says that some conjugate `e(ω·s)` is 0, and
//! certified values tell which: `e` is 0 when it is not certainly nonzero
//! and every other conjugate is, or when every coefficient of `e` over `s`
//! is 0 one level down (two conjugates vanish when `s` lies in the field
//! below).  At the bottom the test is exact.  Nothing is guessed: whatever
//! does not settle is `None`.
//!
//! Quotients are numerator/denominator pairs; every denominator is proved
//! nonzero by the same test.  Nothing here recurses over the expression (a
//! post-order walk); the recursions are over the levels of the tower, at
//! most [`MAX_GENERATORS`] deep.
//!
//! # Realness
//!
//! Complex conjugation maps `ζ ↦ ζ⁻¹` and a generator `s = v·ρ^{1/m}` to
//! `v̄·ρ̄^{1/m}`: a root of a real `ρ` is itself or `e^{−2πi/m}` times
//! itself (by the certified sign of `ρ`), any other is a new generator.  A
//! constant `x` is real when `x − x̄` is zero by the test above
//! ([`is_real`]) — the realness of roots by Cardano's and Ferrari's
//! formulas, whose imaginary parts cancel only exactly.  A generator that
//! is the root of a negative real number is evaluated as `(−1)^{1/m}·(−ρ)^{1/m}`
//! from then on: written `ρ^{1/m}` with complex radicals in `ρ`, it is on
//! the branch cut and does not evaluate.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;

/// The largest `φ(N)` taken: a product in `ℚ(ζ)` costs `φ(N)²` rational
/// products.
const MAX_PHI: usize = 192;
/// The most generators of a tower.
const MAX_GENERATORS: usize = 8;
/// The most rationals of an element, `φ(N)·∏ m_j`.
const MAX_SIZE: usize = 4096;
/// The most rational products of one product of dense elements, `φ(N)²·∏
/// c(m_j)` with `c(2) = 3` (Karatsuba) and `c(3) = 9` (a fraction of a
/// second; the elements met are sparse, and `tan(21π/64)` against its
/// nested radicals, at 3·10⁶, takes 13 ms).
const MAX_PRODUCT_COST: usize = 1 << 22;
/// The most nodes of a constant tested.
const MAX_NODES: usize = 400;
/// The largest `N` considered (before `φ(N)` is checked).
const MAX_N: i64 = 4096;
/// The largest integer exponent taken.
const MAX_EXPONENT: u32 = 64;
/// The most coefficient tests of one zero test ([`Tower::coefficients_vanish`]).
const MAX_COEFFICIENT_TESTS: usize = 16;

/// An element of `ℚ(ζ)`: its coordinates on `1, ζ, …, ζ^{φ(N)−1}`.
type Base = Vec<Q>;
/// An element of the tower: its `ℚ(ζ)` coordinates on the monomials
/// `∏ s_jᵉʲ` (`e_j < m_j`), the first generator least significant.
type Elem = Vec<Base>;

/// `ℚ(ζ)`, `ζ = e^{2πi/N}`.
struct Cyclotomic {
    n: i64,
    phi: usize,
    /// `Φ_N`, ascending (monic, `φ(N) + 1` coefficients).
    modulus: Vec<Q>,
}

/// Euler's `φ(n)` (`n ≥ 1`).
fn totient(n: i64) -> i64 {
    let (mut m, mut out, mut p) = (n, n, 2);
    while p * p <= m {
        if m % p == 0 {
            while m % p == 0 {
                m /= p;
            }
            out -= out / p;
        }
        p += 1;
    }
    if m > 1 {
        out -= out / m;
    }
    out
}

/// `Φ_n`, ascending: `Φ_d = (x^d − 1)/∏_{e | d, e < d} Φ_e` over the
/// divisors `d` of `n`, ascending.
fn cyclotomic_polynomial(n: i64) -> Option<Vec<Q>> {
    use crate::poly::dense::Poly;
    let divisors: Vec<i64> = (1..=n).filter(|d| n % d == 0).collect();
    let mut phis: FxHashMap<i64, Poly> = FxHashMap::default();
    for &d in &divisors {
        let mut c = vec![Q::zero(); usize::try_from(d).ok()? + 1];
        c[0] = -Q::one();
        c[usize::try_from(d).ok()?] = Q::one();
        let mut p = Poly::from_coeffs(c);
        for e in divisors.iter().filter(|&&e| e < d && d % e == 0) {
            p = p.div_exact(phis.get(e)?)?;
        }
        phis.insert(d, p);
    }
    Some(phis.remove(&n)?.coeffs().to_vec())
}

fn base_is_zero(x: &[Q]) -> bool {
    x.iter().all(Zero::is_zero)
}

fn elem_is_zero(x: &[Base]) -> bool {
    x.iter().all(|b| base_is_zero(b))
}

fn base_add(x: &[Q], y: &[Q]) -> Base {
    x.iter().zip(y).map(|(a, b)| a + b).collect()
}

fn elem_add(x: &[Base], y: &[Base]) -> Elem {
    x.iter().zip(y).map(|(a, b)| base_add(a, b)).collect()
}

fn elem_neg(x: &[Base]) -> Elem {
    x.iter().map(|b| b.iter().map(|c| -c).collect()).collect()
}

fn elem_scale(x: &[Base], q: &Q) -> Elem {
    x.iter()
        .map(|b| b.iter().map(|c| c * q).collect())
        .collect()
}

impl Cyclotomic {
    fn new(n: i64) -> Option<Self> {
        let phi = usize::try_from(totient(n)).ok()?;
        if phi > MAX_PHI {
            return None;
        }
        let modulus = cyclotomic_polynomial(n)?;
        (modulus.len() == phi + 1).then_some(Self { n, phi, modulus })
    }

    fn zero(&self) -> Base {
        vec![Q::zero(); self.phi]
    }

    fn constant(&self, q: Q) -> Base {
        let mut v = self.zero();
        v[0] = q;
        v
    }

    /// `v` (any length) reduced modulo `Φ_N`.
    fn reduce(&self, mut v: Vec<Q>) -> Base {
        let phi = self.phi;
        for k in (phi..v.len()).rev() {
            if v[k].is_zero() {
                continue;
            }
            let c = std::mem::replace(&mut v[k], Q::zero());
            for (i, m) in self.modulus[..phi].iter().enumerate() {
                if !m.is_zero() {
                    v[k - phi + i] -= &c * m;
                }
            }
        }
        v.resize(phi, Q::zero());
        v
    }

    /// `ζʲ`.
    fn zeta(&self, j: i64) -> Base {
        let e = usize::try_from(j.rem_euclid(self.n)).unwrap_or(0);
        let mut v = vec![Q::zero(); (e + 1).max(self.phi)];
        v[e] = Q::one();
        self.reduce(v)
    }

    fn mul(&self, a: &[Q], b: &[Q]) -> Base {
        if base_is_zero(a) || base_is_zero(b) {
            return self.zero();
        }
        let mut out = vec![Q::zero(); 2 * self.phi - 1];
        for (i, x) in a.iter().enumerate() {
            if x.is_zero() {
                continue;
            }
            for (j, y) in b.iter().enumerate() {
                if !y.is_zero() {
                    out[i + j] += x * y;
                }
            }
        }
        self.reduce(out)
    }
}

/// A generator `s` of the tower: `s^order = radicand`.  It stands for
/// `v·t`, `t` the principal root of `ρ = u/v` (`rho`), so that the radicand
/// `u·v^{m−1}` is an element.
struct Generator {
    order: usize,
    /// An element of the level below the generator.
    radicand: Elem,
    /// An arena constant with the value of `s`.
    value: ExprId,
    /// `ρ = u/v`, elements of the level below.
    rho: Frac,
    /// An arena constant with the value of `t`.
    root: ExprId,
}

struct Tower {
    field: Cyclotomic,
    gens: Vec<Generator>,
    /// `ζʲ` as arena constants, by `j`.
    zeta_exprs: FxHashMap<i64, ExprId>,
    /// The complex conjugate of each generator that has one computed
    /// ([`Tower::conj_gen`]).
    conjugates: Vec<Option<Frac>>,
    /// The coefficient tests [`Tower::coefficients_vanish`] may still run:
    /// each branches the descent over the levels into one per coefficient.
    coefficient_budget: usize,
}

/// A value `num/den` of the tower (`den` not 0: checked separately).
#[derive(Clone)]
struct Frac {
    num: Elem,
    den: Elem,
}

impl Tower {
    fn level(&self) -> usize {
        self.gens.len()
    }

    fn dim(&self, level: usize) -> usize {
        self.gens[..level].iter().map(|g| g.order).product()
    }

    fn zero(&self, level: usize) -> Elem {
        vec![self.field.zero(); self.dim(level)]
    }

    fn base(&self, b: Base, level: usize) -> Elem {
        let mut e = self.zero(level);
        e[0] = b;
        e
    }

    fn constant(&self, q: Q, level: usize) -> Elem {
        self.base(self.field.constant(q), level)
    }

    /// `x` (of a level at most `level`) as an element of `level`.
    fn pad(&self, x: &[Base], level: usize) -> Elem {
        let mut v = x.to_vec();
        v.resize(self.dim(level), self.field.zero());
        v
    }

    fn is_one(&self, x: &[Base]) -> bool {
        x.first()
            .is_some_and(|b| b[0].is_one() && b[1..].iter().all(Zero::is_zero))
            && x[1..].iter().all(|b| base_is_zero(b))
    }

    /// `x·y` at `level`: the product of the polynomials in the top
    /// generator `s`, reduced by `sᵐ = r`.
    fn mul(&self, x: &[Base], y: &[Base], level: usize) -> Elem {
        if level == 0 {
            return vec![self.field.mul(&x[0], &y[0])];
        }
        if elem_is_zero(x) || elem_is_zero(y) {
            return self.zero(level);
        }
        if self.is_one(x) {
            return y.to_vec();
        }
        if self.is_one(y) {
            return x.to_vec();
        }
        let g = &self.gens[level - 1];
        let m = g.order;
        let h = self.dim(level - 1);
        let l = level - 1;
        if m == 2 {
            // (x₀ + x₁s)(y₀ + y₁s) = x₀y₀ + x₁y₁·r + ((x₀ + x₁)(y₀ + y₁)
            // − x₀y₀ − x₁y₁)·s: three products (Karatsuba).
            let (x0, x1) = x.split_at(h);
            let (y0, y1) = y.split_at(h);
            let p = self.mul(x0, y0, l);
            let q = self.mul(x1, y1, l);
            let s = self.mul(&elem_add(x0, x1), &elem_add(y0, y1), l);
            let qr = self.mul(&q, &g.radicand, l);
            let mut out = elem_add(&p, &qr);
            out.extend(elem_add(&s, &elem_neg(&elem_add(&p, &q))));
            return out;
        }
        let mut w: Vec<Elem> = vec![self.zero(level - 1); 2 * m - 1];
        for (j, xj) in x.chunks(h).enumerate() {
            if elem_is_zero(xj) {
                continue;
            }
            for (k, yk) in y.chunks(h).enumerate() {
                if elem_is_zero(yk) {
                    continue;
                }
                let p = self.mul(xj, yk, level - 1);
                w[j + k] = elem_add(&w[j + k], &p);
            }
        }
        for t in (m..2 * m - 1).rev() {
            if elem_is_zero(&w[t]) {
                continue;
            }
            let p = self.mul(&w[t], &g.radicand, level - 1);
            w[t - m] = elem_add(&w[t - m], &p);
        }
        w.truncate(m);
        w.concat()
    }

    fn pow(&self, x: &[Base], n: u32, level: usize) -> Elem {
        let mut out = self.constant(Q::one(), level);
        let mut base = x.to_vec();
        let mut e = n;
        while e > 0 {
            if e & 1 == 1 {
                out = self.mul(&out, &base, level);
            }
            e >>= 1;
            if e > 0 {
                base = self.mul(&base, &base, level);
            }
        }
        out
    }

    /// The norm `∏_ω e(ω·s)` of `e` over the top generator `s` of `level`:
    /// an element of `level − 1`.
    fn norm(&self, e: &[Base], level: usize) -> Option<Elem> {
        let g = &self.gens[level - 1];
        let h = self.dim(level - 1);
        let u: Vec<&[Base]> = e.chunks(h).collect();
        let r = &g.radicand;
        let l = level - 1;
        match g.order {
            2 => {
                let a = self.mul(u[0], u[0], l);
                let b = self.mul(&self.mul(u[1], u[1], l), r, l);
                Some(elem_add(&a, &elem_neg(&b)))
            }
            3 => {
                let r2 = self.mul(r, r, l);
                let c0 = self.pow(u[0], 3, l);
                let c1 = self.mul(&self.pow(u[1], 3, l), r, l);
                let c2 = self.mul(&self.pow(u[2], 3, l), &r2, l);
                let c3 = self.mul(&self.mul(&self.mul(u[0], u[1], l), u[2], l), r, l);
                let three = Q::from_integer(BigInt::from(3));
                let sum = elem_add(&elem_add(&c0, &c1), &c2);
                Some(elem_add(&sum, &elem_neg(&elem_scale(&c3, &three))))
            }
            _ => None,
        }
    }

    /// Is the value of the element `e` of `level` zero?  See the module
    /// documentation.
    fn value_is_zero(&mut self, arena: &mut Arena, e: &[Base], level: usize) -> Option<bool> {
        if level == 0 {
            return Some(base_is_zero(&e[0]));
        }
        let h = self.dim(level - 1);
        if elem_is_zero(&e[h..]) {
            return self.value_is_zero(arena, &e[..h], level - 1);
        }
        let n = self.norm(e, level)?;
        if !self.value_is_zero(arena, &n, level - 1)? {
            return Some(false);
        }
        // A conjugate e(ω·s) is 0.
        let own = self.element_expr(arena, e, level, 0);
        if super::constant_is_nonzero(arena, own) {
            return Some(false);
        }
        for k in 1..self.gens[level - 1].order {
            let other = self.element_expr(arena, e, level, k);
            if !super::constant_is_nonzero(arena, other) {
                return self.coefficients_vanish(arena, e, level);
            }
        }
        Some(true)
    }

    /// `Some(true)` when every coefficient `uⱼ` of `e = Σ uⱼ·sʲ` over the
    /// top generator `s` of `level` has the value 0 (one level down), so
    /// that `e` is 0 whatever `s` is; `None` otherwise.  For when two
    /// conjugates `e(ωᵏ·s)` may vanish, which happens when `s` lies in the
    /// field below (a radical the tower does not identify, such as
    /// `√(21 + 4√5) = 1 + 2√5`): if they all vanish, every `uⱼ·sʲ` does
    /// (Vandermonde).  Up to 0.45 such an element was undecided, and which
    /// of two generators came on top decided it: `a − b` stayed undecided
    /// where `b − a` was decided 0, and `a < b` was decided while `a > b`
    /// was not.
    fn coefficients_vanish(&mut self, arena: &mut Arena, e: &[Base], level: usize) -> Option<bool> {
        self.coefficient_budget = self.coefficient_budget.checked_sub(1)?;
        let h = self.dim(level - 1);
        for j in 0..self.gens[level - 1].order {
            let (start, end) = ((j * h).min(e.len()), ((j + 1) * h).min(e.len()));
            let u = self.pad(&e[start..end], level - 1);
            if !elem_is_zero(&u) && !self.value_is_zero(arena, &u, level - 1)? {
                return None;
            }
        }
        Some(true)
    }

    /// `ζʲ` as an arena constant, `(−1)^(2j/N)`.
    fn zeta_expr(&mut self, arena: &mut Arena, j: i64) -> ExprId {
        if j == 0 {
            return arena.one;
        }
        if let Some(&e) = self.zeta_exprs.get(&j) {
            return e;
        }
        let exp = arena.num_ratio(Ratio::new(BigInt::from(2 * j), BigInt::from(self.field.n)));
        let e = arena.pow(arena.neg_one, exp);
        self.zeta_exprs.insert(j, e);
        e
    }

    /// The element `e` of `level` as an arena constant, with the top
    /// generator `s` replaced by `ωᵏ·s`, `ω = e^{2πi/m}` (`twist = k`).
    fn element_expr(
        &mut self,
        arena: &mut Arena,
        e: &[Base],
        level: usize,
        twist: usize,
    ) -> ExprId {
        let mut terms: Vec<ExprId> = Vec::new();
        for (idx, b) in e.iter().enumerate() {
            if base_is_zero(b) {
                continue;
            }
            let mut coeff_terms: Vec<ExprId> = Vec::new();
            for (j, c) in b.iter().enumerate() {
                if c.is_zero() {
                    continue;
                }
                let z = self.zeta_expr(arena, j as i64);
                let c = arena.num_ratio(c.clone());
                coeff_terms.push(arena.mul(&[c, z]));
            }
            let mut factors = vec![arena.add(&coeff_terms)];
            let mut rest = idx;
            for (j, g) in self.gens[..level].iter().enumerate() {
                let k = rest % g.order;
                rest /= g.order;
                if k == 0 {
                    continue;
                }
                let kk = arena.int(k as i64);
                factors.push(arena.pow(g.value, kk));
                if twist > 0 && j + 1 == level {
                    let w = arena.rational((2 * twist * k) as i64, g.order as i64);
                    factors.push(arena.pow(arena.neg_one, w));
                }
            }
            terms.push(arena.mul(&factors));
        }
        arena.add(&terms)
    }

    /// `a + b`, both at the current level.
    fn frac_add(&self, a: &Frac, b: &Frac) -> Frac {
        let l = self.level();
        let (an, ad, bn, bd) = (
            self.pad(&a.num, l),
            self.pad(&a.den, l),
            self.pad(&b.num, l),
            self.pad(&b.den, l),
        );
        if ad == bd {
            return Frac {
                num: elem_add(&an, &bn),
                den: ad,
            };
        }
        Frac {
            num: elem_add(&self.mul(&an, &bd, l), &self.mul(&bn, &ad, l)),
            den: self.mul(&ad, &bd, l),
        }
    }

    fn frac_mul(&self, a: &Frac, b: &Frac) -> Frac {
        let l = self.level();
        Frac {
            num: self.mul(&self.pad(&a.num, l), &self.pad(&b.num, l), l),
            den: self.mul(&self.pad(&a.den, l), &self.pad(&b.den, l), l),
        }
    }

    fn frac_of(&self, num: Elem) -> Frac {
        let l = self.level();
        Frac {
            num: self.pad(&num, l),
            den: self.constant(Q::one(), l),
        }
    }

    /// A new generator `s = v·t`, `t` the principal `m`-th root of `ρ =
    /// u/v` with the value `root`: `sᵐ = u·v^{m−1}`.  The value `t = s/v`.
    fn push_root(&mut self, arena: &mut Arena, m: usize, rho: Frac, root: ExprId) -> Option<Frac> {
        let l = self.level();
        let cost = |m: usize| if m == 2 { 3 } else { m * m };
        let product_cost = self.gens.iter().map(|g| cost(g.order)).product::<usize>()
            * cost(m)
            * self.field.phi
            * self.field.phi;
        if self.gens.len() >= MAX_GENERATORS
            || self.dim(l) * m * self.field.phi > MAX_SIZE
            || product_cost > MAX_PRODUCT_COST
        {
            return None;
        }
        let (u, v) = (self.pad(&rho.num, l), self.pad(&rho.den, l));
        let radicand = self.mul(&u, &self.pow(&v, (m - 1) as u32, l), l);
        let value = if self.is_one(&v) {
            root
        } else {
            let v_expr = self.element_expr(arena, &v, l, 0);
            arena.mul(&[v_expr, root])
        };
        self.gens.push(Generator {
            order: m,
            radicand,
            value,
            rho: Frac {
                num: u,
                den: v.clone(),
            },
            root,
        });
        self.conjugates.push(None);
        let mut s = self.zero(l + 1);
        s[self.dim(l)] = self.field.constant(Q::one());
        Some(Frac {
            num: s,
            den: self.pad(&v, l + 1),
        })
    }

    /// The arena constant of the value `num/den`.
    fn frac_expr(&mut self, arena: &mut Arena, x: &Frac) -> ExprId {
        let l = self.level();
        let (n, d) = (self.pad(&x.num, l), self.pad(&x.den, l));
        let n = self.element_expr(arena, &n, l, 0);
        if self.is_one(&d) {
            return n;
        }
        let d = self.element_expr(arena, &d, l, 0);
        arena.div(n, d)
    }

    /// `a/b` (`b` not 0).
    fn frac_div(&self, a: &Frac, b: &Frac) -> Frac {
        self.frac_mul(
            a,
            &Frac {
                num: b.den.clone(),
                den: b.num.clone(),
            },
        )
    }

    /// The complex conjugate of a base element: `ζʲ ↦ ζ⁻ʲ`.
    fn conj_base(&self, b: &[Q]) -> Base {
        let mut out = self.field.zero();
        for (j, c) in b.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let z = self.field.zeta(-(j as i64));
            for (o, x) in out.iter_mut().zip(&z) {
                if !x.is_zero() {
                    *o += c * x;
                }
            }
        }
        out
    }

    /// The complex conjugate of the element `e` of `level`, as a value of
    /// the current level (Horner in the top generator `s`: `ē = Σ ū_k·s̄ᵏ`).
    fn conj_elem(&mut self, arena: &mut Arena, e: &[Base], level: usize) -> Option<Frac> {
        if level == 0 {
            let b = self.conj_base(&e[0]);
            return Some(self.frac_of(self.base(b, 0)));
        }
        let h = self.dim(level - 1);
        let m = self.gens[level - 1].order;
        let chunks: Vec<Elem> = e.chunks(h).map(<[Base]>::to_vec).collect();
        if elem_is_zero(&e[h..]) {
            return self.conj_elem(arena, &chunks[0], level - 1);
        }
        let c = self.conj_gen(arena, level - 1)?;
        let mut acc = self.conj_elem(arena, &chunks[m - 1], level - 1)?;
        for k in (0..m - 1).rev() {
            let lower = self.conj_elem(arena, &chunks[k], level - 1)?;
            acc = self.frac_add(&self.frac_mul(&acc, &c), &lower);
        }
        Some(acc)
    }

    /// The complex conjugate of the generator `j`, `s̄ = v̄·t̄`, memoised.
    /// For `ρ = u/v` real (`u·v̄ − ū·v = 0`, decided as any value) `t̄ = t`
    /// when `ρ > 0` and `t̄ = e^{−2πi/m}·t` when `ρ < 0` (the principal root
    /// of a negative number has the argument `π/m`), the sign certified;
    /// otherwise `t̄` is the principal root of `ρ̄` (conjugation commutes
    /// with the principal root off the negative axis), a new generator.
    ///
    /// A real negative `ρ` is on the branch cut, where its root does not
    /// evaluate when `ρ` is written with complex radicals (Cardano): the
    /// generator's value is rewritten `v·(−1)^{1/m}·(−ρ)^{1/m}` then.
    fn conj_gen(&mut self, arena: &mut Arena, j: usize) -> Option<Frac> {
        if let Some(c) = &self.conjugates[j] {
            return Some(c.clone());
        }
        let m = self.gens[j].order;
        let rho = self.gens[j].rho.clone();
        let cu = self.conj_elem(arena, &rho.num, j)?;
        let cv = self.conj_elem(arena, &rho.den, j)?;
        let rho_bar = self.frac_div(&cu, &cv);
        // ρ − ρ̄ = (u·d − n·v)/(v·d) for ρ̄ = n/d.
        let l = self.level();
        let (u, v) = (self.pad(&rho.num, l), self.pad(&rho.den, l));
        let (n, d) = (self.pad(&rho_bar.num, l), self.pad(&rho_bar.den, l));
        let diff = elem_add(&self.mul(&u, &d, l), &elem_neg(&self.mul(&n, &v, l)));
        let s = {
            let mut s = self.zero(j + 1);
            s[self.dim(j)] = self.field.constant(Q::one());
            self.pad(&s, l)
        };
        let conj = if self.value_is_zero(arena, &diff, l)? {
            let rho_expr = self.frac_expr(arena, &rho);
            let negative = match super::certified_real_sign(arena, rho_expr)? {
                std::cmp::Ordering::Less => true,
                std::cmp::Ordering::Greater => false,
                std::cmp::Ordering::Equal => return None,
            };
            // s̄ = v̄·ω^{−k}·s/v, ω = e^{2πi/m}.
            let mut factor = self.frac_of(self.constant(Q::one(), 0));
            if negative {
                if self.field.n % (m as i64) != 0 {
                    return None;
                }
                let w = self.field.zeta(-self.field.n / m as i64);
                factor = self.frac_of(self.base(w, 0));
                // The value off the cut: v·(−1)^{1/m}·(−ρ)^{1/m}.
                let inv_m = arena.rational(1, m as i64);
                let unit = arena.pow(arena.neg_one, inv_m);
                let minus = arena.neg(rho_expr);
                let magnitude = arena.pow(minus, inv_m);
                let root = arena.mul(&[unit, magnitude]);
                let value = if self.is_one(&rho.den) {
                    root
                } else {
                    let v_expr = self.element_expr(arena, &rho.den, j, 0);
                    arena.mul(&[v_expr, root])
                };
                self.gens[j].root = root;
                self.gens[j].value = value;
            }
            let num = self.frac_mul(&self.frac_mul(&cv, &factor), &self.frac_of(s));
            self.frac_div(&num, &self.frac_of(v))
        } else {
            // ρ̄ is often the radicand of another generator already (the
            // two cube roots `∛(a ± b·i)` of Cardano's formula): conjugation
            // commutes with the principal root off the negative axis, so `t̄`
            // is that generator's `t`.
            let found = (0..self.gens.len()).find(|&k| {
                k != j && self.gens[k].order == m && {
                    let (rn, rd) = (
                        self.pad(&self.gens[k].rho.num, l),
                        self.pad(&self.gens[k].rho.den, l),
                    );
                    self.mul(&rn, &d, l) == self.mul(&n, &rd, l)
                }
            });
            let t_bar = match found {
                Some(k) => {
                    let mut s_k = self.zero(k + 1);
                    s_k[self.dim(k)] = self.field.constant(Q::one());
                    let rd = self.gens[k].rho.den.clone();
                    Frac {
                        num: self.pad(&s_k, l),
                        den: self.pad(&rd, l),
                    }
                }
                None => {
                    let root = arena.conjugate(self.gens[j].root);
                    self.push_root(arena, m, rho_bar, root)?
                }
            };
            self.frac_mul(&cv, &t_bar)
        };
        self.conjugates[j] = Some(conj.clone());
        Some(conj)
    }
}

/// Can the node `id` be part of a constant [`is_zero`] takes (a cheap test
/// before the field is built)?
fn supported(arena: &Arena, id: ExprId) -> bool {
    match arena.node(id) {
        ExprNode::Num(_)
        | ExprNode::Add(_)
        | ExprNode::Mul(_)
        | ExprNode::Neg(_)
        | ExprNode::ImaginaryUnit
        | ExprNode::GoldenRatio
        | ExprNode::Pi
        | ExprNode::Cos(_)
        | ExprNode::Sin(_)
        | ExprNode::Tan(_) => true,
        ExprNode::Pow(_, e) => arena.as_num(*e).is_some(),
        _ => false,
    }
}

/// `lcm(a, b)` within [`MAX_N`].
fn bounded_lcm(a: i64, b: i64) -> Option<i64> {
    let l = a.lcm(&b);
    (l <= MAX_N).then_some(l)
}

/// The prime factors (2 and 3 only) of `q`, with repetition.
fn root_chain(q: i64) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    let mut q = q;
    for p in [2i64, 3] {
        while q % p == 0 {
            q /= p;
            out.push(p as usize);
        }
    }
    (q == 1).then_some(out)
}

/// `N` for the atoms of `order`: twice the denominator of every multiple
/// of π under `cos`, four times under `sin` and `tan`, `2q` for `(−1)^(p/q)`,
/// `4q` for `i^(p/q)`, `4` for `i`.  `None` when an argument of `sin`, `cos`,
/// `tan` is not a rational multiple of π or `N` exceeds [`MAX_N`].
fn field_order(arena: &Arena, order: &[ExprId]) -> Option<i64> {
    // Denominators beyond `MAX_N` give `None` before any product overflows.
    let denominator = |q: &Q| q.denom().to_i64().filter(|&d| d <= MAX_N);
    let mut n: i64 = 2;
    for &id in order {
        let need = match arena.node(id) {
            ExprNode::Cos(a) => 2 * denominator(&super::as_pi_multiple(arena, *a)?)?,
            ExprNode::Sin(a) | ExprNode::Tan(a) => {
                bounded_lcm(4, 2 * denominator(&super::as_pi_multiple(arena, *a)?)?)?
            }
            ExprNode::ImaginaryUnit => 4,
            ExprNode::Pow(b, e) if *b == arena.neg_one || *b == arena.i_unit => {
                let q = denominator(arena.as_num(*e)?)?;
                if *b == arena.neg_one { 2 * q } else { 4 * q }
            }
            // The conjugate of a cube root of a negative number is
            // `e^{−2πi/3}` times it ([`Tower::conj_gen`]).
            ExprNode::Pow(_, e)
                if arena
                    .as_num(*e)
                    .is_some_and(|q| (q.denom() % BigInt::from(3)).is_zero()) =>
            {
                3
            }
            _ => 1,
        };
        n = bounded_lcm(n, need)?;
    }
    Some(n)
}

/// Is the constant `d` zero?  `d` is built from rationals, `i`, `φ`, `sin`,
/// `cos`, `tan` at rational multiples of π, `(−1)^q`, `i^q`, sums, products,
/// integer powers and radicals `b^(p/q)` with `q = 2ᵃ3ᶜ` of such constants
/// (see the module documentation).  `None` when it holds anything else, is
/// too large (more than [`MAX_NODES`] nodes, `φ(N)` above [`MAX_PHI`], more
/// than [`MAX_GENERATORS`] radicals, elements above [`MAX_SIZE`]
/// rationals), divides by a value not proved nonzero, or a certified value
/// does not settle.
pub(super) fn is_zero(arena: &mut Arena, d: ExprId) -> Option<bool> {
    let (mut walk, value) = Walk::over(arena, d)?;
    let tower = &mut walk.tower;
    let top = tower.level();
    let num = tower.pad(&value.num, top);
    let zero = tower.value_is_zero(arena, &num, top)?;
    walk.divisors_nonzero(arena)?;
    Some(zero)
}

/// Is the constant `x` real: is `x − x̄` zero ([`is_zero`]'s test, with the
/// conjugation of the tower, [`Tower::conj_gen`])?  For the realness of a
/// root by Cardano's or Ferrari's formula, whose imaginary parts cancel only
/// exactly: up to 0.44 `√λ` of such a negative `λ` beyond the reach of the
/// minimal polynomial (a Ferrari eigenvalue of a 4×4 matrix at a value of
/// its parameter) did not evaluate.  `None` as for [`is_zero`].
pub(super) fn is_real(arena: &mut Arena, x: ExprId) -> Option<bool> {
    let (mut walk, value) = Walk::over(arena, x)?;
    let tower = &mut walk.tower;
    let l = tower.level();
    let (xn, xd) = (tower.pad(&value.num, l), tower.pad(&value.den, l));
    let cn = tower.conj_elem(arena, &xn, l)?;
    let cd = tower.conj_elem(arena, &xd, l)?;
    let x_bar = tower.frac_div(&cn, &cd);
    // x − x̄ = (xn·d − n·xd)/(xd·d) for x̄ = n/d.
    let l = tower.level();
    let (xn, xd) = (tower.pad(&xn, l), tower.pad(&xd, l));
    let (n, d) = (tower.pad(&x_bar.num, l), tower.pad(&x_bar.den, l));
    let diff = elem_add(&tower.mul(&xn, &d, l), &elem_neg(&tower.mul(&n, &xd, l)));
    let real = tower.value_is_zero(arena, &diff, l)?;
    walk.divisors_nonzero(arena)?;
    Some(real)
}

/// The state of [`is_zero`]'s walk over the constant.
struct Walk {
    tower: Tower,
    /// The value of every node met so far that has one.
    values: FxHashMap<ExprId, Frac>,
    /// `b^(1/q)` by `(b, q)`.
    roots: FxHashMap<(ExprId, i64), Frac>,
    /// Denominators to prove nonzero, with their levels.
    divisors: Vec<(Elem, usize)>,
}

impl Walk {
    /// The walk over the constant `d`, and its value (see [`is_zero`] for
    /// the limits).
    fn over(arena: &mut Arena, d: ExprId) -> Option<(Walk, Frac)> {
        let order = walk::post_order_ids(arena, d);
        if order.len() > MAX_NODES || !order.iter().all(|&id| supported(arena, id)) {
            return None;
        }
        let n = field_order(arena, &order)?;
        let mut walk = Walk {
            tower: Tower {
                field: Cyclotomic::new(n)?,
                gens: Vec::new(),
                zeta_exprs: FxHashMap::default(),
                conjugates: Vec::new(),
                coefficient_budget: MAX_COEFFICIENT_TESTS,
            },
            values: FxHashMap::default(),
            roots: FxHashMap::default(),
            divisors: Vec::new(),
        };
        // A node without a value (`π` under a `cos`) leaves its parents
        // without one.
        for &id in &order {
            if let Some(v) = walk.value_of(arena, id) {
                walk.values.insert(id, v);
            }
        }
        let value = walk.values.remove(&d)?;
        Some((walk, value))
    }

    /// `Some(())` when every denominator met is proved nonzero.
    fn divisors_nonzero(&mut self, arena: &mut Arena) -> Option<()> {
        for (x, l) in std::mem::take(&mut self.divisors) {
            let x = self.tower.pad(&x, l);
            if self.tower.value_is_zero(arena, &x, l)? {
                return None;
            }
        }
        Some(())
    }

    /// The value of the node `id`, whose children have theirs (or `None`).
    fn value_of(&mut self, arena: &mut Arena, id: ExprId) -> Option<Frac> {
        let half = Q::new(BigInt::from(1), BigInt::from(2));
        let n = self.tower.field.n;
        let node = arena.node(id).clone();
        let tower = &mut self.tower;
        Some(match node {
            ExprNode::Num(k) => {
                let q = arena.num(k).clone();
                tower.frac_of(tower.constant(q, 0))
            }
            ExprNode::ImaginaryUnit => tower.frac_of(tower.base(tower.field.zeta(n / 4), 0)),
            ExprNode::Neg(a) => {
                let v = self.values.get(&a)?;
                Frac {
                    num: elem_neg(&v.num),
                    den: v.den.clone(),
                }
            }
            ExprNode::Add(cs) => {
                let mut acc = tower.frac_of(tower.zero(0));
                for c in cs.iter() {
                    acc = tower.frac_add(&acc, self.values.get(c)?);
                }
                acc
            }
            ExprNode::Mul(cs) => {
                let mut acc = tower.frac_of(tower.constant(Q::one(), 0));
                for c in cs.iter() {
                    acc = tower.frac_mul(&acc, self.values.get(c)?);
                }
                acc
            }
            ExprNode::Cos(a) | ExprNode::Sin(a) | ExprNode::Tan(a) => {
                let q = super::as_pi_multiple(arena, a)?;
                let k = q * Q::from_integer(BigInt::from(n / 2));
                if !k.is_integer() {
                    return None;
                }
                let k = k.to_integer().to_i64()?;
                let f = &tower.field;
                let cos = elem_scale(&[base_add(&f.zeta(k), &f.zeta(-k))], &half);
                // sin x = −i·(e^{ix} − e^{−ix})/2, −i = ζ^{3N/4}.
                let sin = || {
                    let m = 3 * n / 4;
                    let minus = f.zeta(m - k).iter().map(|c| -c).collect::<Vec<_>>();
                    elem_scale(&[base_add(&f.zeta(m + k), &minus)], &half)
                };
                match node {
                    ExprNode::Cos(_) => tower.frac_of(cos),
                    ExprNode::Sin(_) => tower.frac_of(sin()),
                    _ => {
                        let l = tower.level();
                        let tan = Frac {
                            num: tower.pad(&sin(), l),
                            den: tower.pad(&cos, l),
                        };
                        self.divisors.push((cos, 0));
                        tan
                    }
                }
            }
            ExprNode::GoldenRatio => {
                // φ = (1 + √5)/2.
                let five = arena.int(5);
                let five_value = tower.frac_of(tower.constant(Q::from_integer(BigInt::from(5)), 0));
                let root = radical(arena, tower, &mut self.roots, five, &five_value, 2)?;
                let l = tower.level();
                let one = tower.frac_of(tower.constant(Q::one(), l));
                let sum = tower.frac_add(&one, &root);
                Frac {
                    num: elem_scale(&sum.num, &half),
                    den: sum.den,
                }
            }
            ExprNode::Pow(b, e) => {
                let ex = arena.as_num(e)?.clone();
                let p = ex.numer().to_i64()?;
                let q = ex.denom().to_i64()?;
                if b == arena.neg_one || b == arena.i_unit {
                    let scale = if b == arena.neg_one {
                        n / (2 * q)
                    } else {
                        n / (4 * q)
                    };
                    return Some(
                        tower.frac_of(tower.base(tower.field.zeta(p.checked_mul(scale)?), 0)),
                    );
                }
                let base = self.values.get(&b)?.clone();
                let root = if q == 1 {
                    base
                } else {
                    radical(arena, tower, &mut self.roots, b, &base, q)?
                };
                let k = u32::try_from(p.unsigned_abs())
                    .ok()
                    .filter(|&k| k <= MAX_EXPONENT)?;
                let l = tower.level();
                let (num, den) = (tower.pad(&root.num, l), tower.pad(&root.den, l));
                let (num, den) = (tower.pow(&num, k, l), tower.pow(&den, k, l));
                if p < 0 {
                    self.divisors.push((num.clone(), l));
                    Frac { num: den, den: num }
                } else {
                    Frac { num, den }
                }
            }
            _ => return None,
        })
    }
}

/// `b^(1/q)` (principal) for the value `base` of the arena constant `b`,
/// `q = 2ᵃ3ᶜ`: a chain of generators, memoised by `(b, q)`.  For
/// `t = (u/v)^(1/m)` the generator is `s = v·t`, `sᵐ = u·v^{m−1}` (an
/// element, not a quotient), and `t = s/v`.
fn radical(
    arena: &mut Arena,
    tower: &mut Tower,
    roots: &mut FxHashMap<(ExprId, i64), Frac>,
    b: ExprId,
    base: &Frac,
    q: i64,
) -> Option<Frac> {
    if let Some(r) = roots.get(&(b, q)) {
        return Some(r.clone());
    }
    let chain = root_chain(q)?;
    let mut cur = base.clone();
    let mut cur_expr = b;
    for m in chain {
        let l = tower.level();
        let (u, v) = (tower.pad(&cur.num, l), tower.pad(&cur.den, l));
        if elem_is_zero(&u) {
            cur = tower.frac_of(tower.zero(l));
            continue;
        }
        let inv_m = arena.rational(1, m as i64);
        let t_expr = arena.pow(cur_expr, inv_m);
        cur = tower.push_root(arena, m, Frac { num: u, den: v }, t_expr)?;
        cur_expr = t_expr;
    }
    roots.insert((b, q), cur.clone());
    Some(cur)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cyclotomic_polynomials() {
        let p = |n: i64| -> Vec<i64> {
            cyclotomic_polynomial(n)
                .unwrap()
                .iter()
                .map(|c| c.to_integer().to_i64().unwrap())
                .collect()
        };
        assert_eq!(p(1), vec![-1, 1]);
        assert_eq!(p(4), vec![1, 0, 1]);
        assert_eq!(p(12), vec![1, 0, -1, 0, 1]);
        assert_eq!(p(15), vec![1, -1, 0, 1, -1, 1, 0, -1, 1]);
        assert_eq!(totient(64), 32);
        assert_eq!(totient(105), 48);
    }

    #[test]
    fn roots_of_unity_reduce() {
        let f = Cyclotomic::new(8).unwrap();
        // ζ₈⁴ = −1, ζ₈⁸ = 1, ζ₈·ζ₈⁷ = 1.
        assert_eq!(f.zeta(4), f.constant(-Q::one()));
        assert_eq!(f.zeta(8), f.constant(Q::one()));
        assert_eq!(f.mul(&f.zeta(1), &f.zeta(7)), f.constant(Q::one()));
    }
}
