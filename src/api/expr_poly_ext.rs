//! Polynomial-algebra methods on [`Ex`]: resultant, discriminant, square-free, division, numeric roots.
//!
//! Every method here treats `self` as a univariate polynomial in an
//! explicitly supplied variable `var`, converting through
//! the crate-internal `polybridge` module to the exact dense
//! [`Poly`] representation, performing the computation
//! there, and converting back.  Methods that return `Option` yield `None`
//! when the expression is not polynomial in `var` (or when the operation is
//! undefined, e.g. the discriminant of a constant).

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::api::expr::{Ex, Expr, Numeric};
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;
use crate::base::node::{ExprId, ExprNode};
use crate::poly::Poly;
use crate::poly::polybridge::{expr_to_poly, poly_to_expr};
use crate::poly::sturm::SturmChain;

// ═══════════════════════════════════════════════════════════════════════════
// Arena-level helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Interpret an expression as an exact rational endpoint, accepting `±∞`
/// as `None` for the corresponding side.  Returns `Err(())` for anything
/// else (symbols, π, …).
pub(crate) enum Endpoint {
    /// A finite rational endpoint.
    Finite(Ratio<BigInt>),
    /// `-∞`.
    NegInf,
    /// `+∞`.
    PosInf,
}

fn endpoint(arena: &Arena, id: ExprId) -> Option<Endpoint> {
    match arena.node(id) {
        ExprNode::Num(nid) => Some(Endpoint::Finite(arena.num(*nid).clone())),
        ExprNode::Infinity => Some(Endpoint::PosInf),
        ExprNode::NegInfinity => Some(Endpoint::NegInf),
        _ => None,
    }
}

/// Sign of a rational: `+1`, `0` or `-1`.
fn sign_of(r: &Ratio<BigInt>) -> i8 {
    if r.is_positive() {
        1
    } else if r.is_negative() {
        -1
    } else {
        0
    }
}

/// Exact decision of `f ≥ 0` (or `f > 0` when `strict`) on the closed
/// interval between the two endpoints.
pub(crate) fn poly_sign_on_interval(f: &Poly, lo: &Endpoint, hi: &Endpoint, strict: bool) -> bool {
    // Empty interval: vacuously true.
    match (lo, hi) {
        (Endpoint::PosInf, _) | (_, Endpoint::NegInf) => return true,
        (Endpoint::Finite(a), Endpoint::Finite(b)) if a > b => return true,
        _ => {}
    }
    if f.is_zero() {
        return !strict;
    }
    let deg = f.degree().unwrap_or(0);
    if deg == 0 {
        let s = sign_of(&f.coeff(0));
        return if strict { s > 0 } else { s >= 0 };
    }

    // Every real root lies strictly inside (-bound, bound), so the bound
    // can stand in for an infinite endpoint.
    let bound = crate::poly::sturm::cauchy_bound(f) + Ratio::one();
    let a = match lo {
        Endpoint::Finite(r) => r.clone(),
        _ => -bound.clone(),
    };
    let b = match hi {
        Endpoint::Finite(r) => r.clone(),
        _ => bound,
    };

    // A single point.
    if a == b {
        let s = sign_of(&f.eval(&a));
        return if strict { s > 0 } else { s >= 0 };
    }

    // 1. A root of odd multiplicity strictly inside (a, b) is a sign change.
    let (_content, parts) = f.sqf_list();
    let mut odd = Poly::from_int(1);
    for (p, m) in &parts {
        if m % 2 == 1 {
            odd = &odd * p;
        }
    }
    if odd.degree().unwrap_or(0) > 0 {
        let chain = SturmChain::new(&odd);
        let half_open = chain.count_roots_in(&a, &b); // roots in (a, b]
        let at_b = usize::from(odd.eval(&b).is_zero());
        if half_open.saturating_sub(at_b) > 0 {
            return false;
        }
    }

    // 2. The sign is now constant away from roots; read it off at one point.
    let s = match (lo, hi) {
        (_, Endpoint::PosInf) => f.leading_coeff().map_or(0, sign_of),
        (Endpoint::NegInf, _) => {
            let lc = f.leading_coeff().map_or(0, sign_of);
            if deg.is_multiple_of(2) { lc } else { -lc }
        }
        _ => {
            // d + 1 distinct interior points; at most d of them are roots.
            let width = &b - &a;
            let steps = Ratio::from_integer(BigInt::from(deg as i64 + 2));
            let mut s = 0i8;
            for k in 1..=(deg + 1) {
                let t = &a + &width * Ratio::from_integer(BigInt::from(k as i64)) / &steps;
                s = sign_of(&f.eval(&t));
                if s != 0 {
                    break;
                }
            }
            s
        }
    };
    if s <= 0 {
        return false;
    }
    if !strict {
        return true;
    }

    // 3. Strict positivity additionally forbids any root in the closed
    //    interval (the bound endpoints are never roots).
    SturmChain::new(f).count_roots_in_closed(&a, &b) == 0
}

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<Numeric> — polynomial algebra
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    // ── Factoring conveniences ─────────────────────────────────────

    /// Factor over ℤ with the variable(s) inferred from the free symbols.
    ///
    /// With one free symbol this is [`factor`](Self::factor) in that symbol;
    /// with two to four symbols the polynomial is factored as a multivariate
    /// polynomial (Kronecker substitution).  Expressions that are not
    /// polynomial, have no free symbols, or are already irreducible are
    /// returned unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let f = (&x.powi(2) - &y.powi(2)).factor_all();
    /// let s = format!("{f}");
    /// assert!(s.contains("x - y") && s.contains("x + y"), "{s}");
    /// ```
    #[must_use = "returns the factored form; does not modify in place"]
    pub fn factor_all(&self) -> Ex {
        let id = {
            let mut inner = self.inner.write();
            crate::simplify::factor::factor_auto(&mut inner.arena, self.raw_id())
        };
        self.wrap(id)
    }

    /// Factor over ℤ and return the pieces: `(content, [(factor, mult), …])`
    /// with `self = content · ∏ factorᵢ^multᵢ`.
    ///
    /// Non-polynomial or constant input yields `(1, [(self, 1)])`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // 2x³ − 2x² − 2x + 2 = 2 (x − 1)² (x + 1)
    /// let f = &x.powi(3) * 2 - &x.powi(2) * 2 - &x * 2 + 2;
    /// let (content, factors) = f.factor_list(&x);
    /// assert_eq!(format!("{content}"), "2");
    /// assert_eq!(factors.len(), 2);
    /// assert_eq!(factors.iter().map(|(_, m)| *m).max(), Some(2));
    /// ```
    #[must_use]
    pub fn factor_list(&self, var: &Ex) -> (Ex, Vec<(Ex, u32)>) {
        let var_id = self.checked_id(var);
        let (c, fs) = {
            let mut inner = self.inner.write();
            crate::simplify::factor::factor_list(&mut inner.arena, self.raw_id(), Some(var_id))
        };
        (
            self.wrap(c),
            fs.into_iter().map(|(f, m)| (self.wrap(f), m)).collect(),
        )
    }

    /// Like [`factor_list`](Self::factor_list) with the variables inferred
    /// from the free symbols (see [`factor_all`](Self::factor_all)).
    #[must_use]
    pub fn factor_list_all(&self) -> (Ex, Vec<(Ex, u32)>) {
        let (c, fs) = {
            let mut inner = self.inner.write();
            crate::simplify::factor::factor_list(&mut inner.arena, self.raw_id(), None)
        };
        (
            self.wrap(c),
            fs.into_iter().map(|(f, m)| (self.wrap(f), m)).collect(),
        )
    }

    // ── Resultant / discriminant ───────────────────────────────────

    /// Resultant `res_var(self, other)` of two polynomials in `var`.
    ///
    /// The resultant vanishes exactly when the two polynomials share a
    /// root.  Returns `None` if either expression is not polynomial in
    /// `var`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.powi(2) - 1;
    /// let g = &x - 1;
    /// assert_eq!(format!("{}", f.resultant(&g, &x).unwrap()), "0");
    /// let h = &x - 3;
    /// assert_eq!(format!("{}", f.resultant(&h, &x).unwrap()), "8");
    /// ```
    #[must_use]
    pub fn resultant(&self, other: &Ex, var: &Ex) -> Option<Ex> {
        let other_id = self.checked_id(other);
        let var_id = self.checked_id(var);
        let id = {
            let mut inner = self.inner.write();
            let a = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
            let b = expr_to_poly(&inner.arena, other_id, var_id)?;
            let r = Poly::resultant(&a, &b);
            inner.arena.num_ratio(r)
        };
        Some(self.wrap(id))
    }

    /// Discriminant of `self` as a polynomial in `var`.
    ///
    /// `disc(f) = (−1)^{n(n−1)/2} · res(f, f′) / lc(f)`; it is zero iff the
    /// polynomial has a repeated root.  Returns `None` for non-polynomial
    /// or constant input.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // ax² + bx + c has discriminant b² − 4ac: x² + 3x + 1 → 5
    /// let f = &x.powi(2) + &x * 3 + 1;
    /// assert_eq!(format!("{}", f.discriminant(&x).unwrap()), "5");
    /// // (x − 1)² has a repeated root
    /// let g = &x.powi(2) - &x * 2 + 1;
    /// assert_eq!(format!("{}", g.discriminant(&x).unwrap()), "0");
    /// ```
    #[must_use]
    pub fn discriminant(&self, var: &Ex) -> Option<Ex> {
        let var_id = self.checked_id(var);
        let id = {
            let mut inner = self.inner.write();
            let f = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
            let d = f.discriminant()?;
            inner.arena.num_ratio(d)
        };
        Some(self.wrap(id))
    }

    // ── Square-free ────────────────────────────────────────────────

    /// Square-free decomposition over ℤ: `(content, [(a₁, 1), (a₂, 2), …])`
    /// with `self = content · ∏ aᵢ^i`, each `aᵢ` square-free and pairwise
    /// coprime.
    ///
    /// Returns `None` for non-polynomial or constant input.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // x⁵ − x⁴ − x + 1 = (x − 1)(x⁴ − 1) = (x − 1)² · (x + 1)(x² + 1)
    /// let f = &x.powi(5) - &x.powi(4) - &x + 1;
    /// let (content, parts) = f.sqf_list(&x).unwrap();
    /// assert_eq!(format!("{content}"), "1");
    /// let mults: Vec<u32> = parts.iter().map(|(_, m)| *m).collect();
    /// assert_eq!(mults, vec![1, 2]);
    /// ```
    #[must_use]
    pub fn sqf_list(&self, var: &Ex) -> Option<(Ex, Vec<(Ex, u32)>)> {
        let var_id = self.checked_id(var);
        let (c, parts) = {
            let mut inner = self.inner.write();
            let f = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
            if f.degree().unwrap_or(0) == 0 {
                return None;
            }
            let (content, parts) = f.sqf_list();
            let c = inner.arena.num_ratio(content);
            let parts: Vec<(ExprId, u32)> = parts
                .iter()
                .map(|(p, m)| (poly_to_expr(&mut inner.arena, p, var_id), *m))
                .collect();
            (c, parts)
        };
        Some((
            self.wrap(c),
            parts.into_iter().map(|(p, m)| (self.wrap(p), m)).collect(),
        ))
    }

    /// Square-free part: the product of the distinct irreducible factors of
    /// `self` (primitive, positive leading coefficient).
    ///
    /// Returns `None` for non-polynomial or constant input.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = (&x - 1).powi(3) * (&x + 2);
    /// let sf = f.expand().square_free_part(&x).unwrap();
    /// assert_eq!(format!("{sf}"), "x^2 + x - 2");
    /// ```
    #[must_use]
    pub fn square_free_part(&self, var: &Ex) -> Option<Ex> {
        let var_id = self.checked_id(var);
        let id = {
            let mut inner = self.inner.write();
            let f = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
            if f.degree().unwrap_or(0) == 0 {
                return None;
            }
            let (_c, parts) = f.sqf_list();
            let mut prod = Poly::from_int(1);
            for (p, _) in &parts {
                prod = &prod * p;
            }
            poly_to_expr(&mut inner.arena, &prod, var_id)
        };
        Some(self.wrap(id))
    }

    /// Is `self` square-free as a polynomial in `var` (no repeated roots)?
    ///
    /// Returns `None` for non-polynomial or constant input.
    #[must_use]
    pub fn is_squarefree(&self, var: &Ex) -> Option<bool> {
        let var_id = self.checked_id(var);
        let inner = self.inner.read();
        let f = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
        f.is_squarefree()
    }

    /// Is `self` irreducible over ℚ as a polynomial in `var`?
    ///
    /// Returns `None` for non-polynomial or constant input.  Non-unit
    /// content is ignored (`2x + 2` is irreducible).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!((&x.powi(4) + 1).is_irreducible(&x), Some(true));
    /// assert_eq!((&x.powi(4) + 4).is_irreducible(&x), Some(false));
    /// assert_eq!(ctx.int(3).is_irreducible(&x), None);
    /// ```
    #[must_use]
    pub fn is_irreducible(&self, var: &Ex) -> Option<bool> {
        let var_id = self.checked_id(var);
        let inner = self.inner.read();
        let f = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
        crate::poly::factor_zassenhaus::is_irreducible_z(&f)
    }

    // ── Division ───────────────────────────────────────────────────

    /// Polynomial division with remainder: `(quotient, remainder)` with
    /// `self = quotient · other + remainder` and
    /// `deg remainder < deg other`.
    ///
    /// Returns `None` if either expression is not polynomial in `var` or
    /// if `other` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.powi(3) + 1;
    /// let g = &x + 1;
    /// let (q, r) = f.poly_div(&g, &x).unwrap();
    /// assert_eq!(format!("{q}"), "x^2 - x + 1");
    /// assert_eq!(format!("{r}"), "0");
    /// ```
    #[must_use]
    pub fn poly_div(&self, other: &Ex, var: &Ex) -> Option<(Ex, Ex)> {
        let other_id = self.checked_id(other);
        let var_id = self.checked_id(var);
        let (q, r) = {
            let mut inner = self.inner.write();
            let a = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
            let b = expr_to_poly(&inner.arena, other_id, var_id)?;
            if b.is_zero() {
                return None;
            }
            let (q, r) = a.div_rem(&b);
            let q = poly_to_expr(&mut inner.arena, &q, var_id);
            let r = poly_to_expr(&mut inner.arena, &r, var_id);
            (q, r)
        };
        Some((self.wrap(q), self.wrap(r)))
    }

    /// Polynomial quotient (see [`poly_div`](Self::poly_div)).
    #[must_use]
    pub fn poly_quo(&self, other: &Ex, var: &Ex) -> Option<Ex> {
        self.poly_div(other, var).map(|(q, _)| q)
    }

    /// Polynomial remainder (see [`poly_div`](Self::poly_div)).
    #[must_use]
    pub fn poly_rem(&self, other: &Ex, var: &Ex) -> Option<Ex> {
        self.poly_div(other, var).map(|(_, r)| r)
    }

    /// Extended Euclidean algorithm: `(s, t, g)` with
    /// `s · self + t · other = g = gcd(self, other)` and `g` monic.
    ///
    /// Returns `None` if either expression is not polynomial in `var`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.powi(2) - 1;
    /// let g = &x.powi(2) - &x * 2 + 1;   // (x − 1)²
    /// let (s, t, gcd) = f.poly_gcdex(&g, &x).unwrap();
    /// assert_eq!(format!("{gcd}"), "x - 1");
    /// let check = (&s * &f + &t * &g).expand();
    /// assert_eq!(format!("{check}"), "x - 1");
    /// ```
    #[must_use]
    pub fn poly_gcdex(&self, other: &Ex, var: &Ex) -> Option<(Ex, Ex, Ex)> {
        let other_id = self.checked_id(other);
        let var_id = self.checked_id(var);
        let (s, t, g) = {
            let mut inner = self.inner.write();
            let a = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
            let b = expr_to_poly(&inner.arena, other_id, var_id)?;
            let (s, t, g) = Poly::extended_gcd(&a, &b);
            let s = poly_to_expr(&mut inner.arena, &s, var_id);
            let t = poly_to_expr(&mut inner.arena, &t, var_id);
            let g = poly_to_expr(&mut inner.arena, &g, var_id);
            (s, t, g)
        };
        Some((self.wrap(s), self.wrap(t), self.wrap(g)))
    }

    // ── Structure ──────────────────────────────────────────────────

    /// Functional decomposition `self = g₁ ∘ g₂ ∘ … ∘ gₖ` into
    /// indecomposable polynomials, outermost first.
    ///
    /// Indecomposable polynomials return `vec![self]`; non-polynomial or
    /// constant input returns an empty vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // x⁴ + 2x² + 1 = (x² + 2x + 1) ∘ x²
    /// let f = &x.powi(4) + &x.powi(2) * 2 + 1;
    /// let parts: Vec<String> = f.decompose(&x).iter().map(|p| format!("{p}")).collect();
    /// assert_eq!(parts, vec!["x^2 + 2*x + 1", "x^2"]);
    /// ```
    #[must_use]
    pub fn decompose(&self, var: &Ex) -> Vec<Ex> {
        let var_id = self.checked_id(var);
        let ids: Vec<ExprId> = {
            let mut inner = self.inner.write();
            let Some(f) = expr_to_poly(&inner.arena, self.raw_id(), var_id) else {
                return vec![];
            };
            if f.degree().unwrap_or(0) == 0 {
                return vec![];
            }
            f.decompose()
                .iter()
                .map(|p| poly_to_expr(&mut inner.arena, p, var_id))
                .collect()
        };
        ids.into_iter().map(|id| self.wrap(id)).collect()
    }

    /// Split into `(content, primitive_part)` where `content` is the
    /// rational GCD of the coefficients (signed so that the primitive part
    /// has a positive leading coefficient) and `self = content ·
    /// primitive_part`.
    ///
    /// Non-polynomial input returns `(1, self)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.powi(2) * -4 + &x * 6;
    /// let (c, p) = f.content_primitive(&x);
    /// assert_eq!(format!("{c}"), "-2");
    /// assert_eq!(format!("{p}"), "2*x^2 - 3*x");
    /// ```
    #[must_use]
    pub fn content_primitive(&self, var: &Ex) -> (Ex, Ex) {
        let var_id = self.checked_id(var);
        let result = {
            let mut inner = self.inner.write();
            match expr_to_poly(&inner.arena, self.raw_id(), var_id) {
                Some(f) if !f.is_zero() => {
                    let mut c = f.content();
                    let mut p = f.primitive_part();
                    if p.leading_coeff().is_some_and(|lc| lc.is_negative()) {
                        c = -c;
                        p = -&p;
                    }
                    let c = inner.arena.num_ratio(c);
                    let p = poly_to_expr(&mut inner.arena, &p, var_id);
                    Some((c, p))
                }
                _ => None,
            }
        };
        match result {
            Some((c, p)) => (self.wrap(c), self.wrap(p)),
            None => {
                let one = self.inner.read().arena.one;
                (self.wrap(one), self.clone())
            }
        }
    }

    /// Leading coefficient of `self` as a polynomial in `var`.
    ///
    /// Returns `None` for non-polynomial input; the zero polynomial has
    /// leading coefficient `0`.  Coefficients may be exact numbers or
    /// arbitrary expressions free of `var`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    /// let f = &x.powi(3) * 5 - &x + 2;
    /// assert_eq!(format!("{}", f.leading_coeff(&x).unwrap()), "5");
    /// let g = &a * &x.powi(2) + &x + 1;
    /// assert_eq!(g.leading_coeff(&x).unwrap(), a);
    /// assert!(x.sin().leading_coeff(&x).is_none());
    /// ```
    #[must_use]
    pub fn leading_coeff(&self, var: &Ex) -> Option<Ex> {
        let var_id = self.checked_id(var);
        let id = {
            let mut inner = self.inner.write();
            match expr_to_poly(&inner.arena, self.raw_id(), var_id) {
                Some(f) => {
                    let lc = f.leading_coeff().cloned().unwrap_or_else(Ratio::zero);
                    inner.arena.num_ratio(lc)
                }
                None => {
                    let coeffs = crate::api::expr_funcs::symbolic_coeffs_of(
                        &mut inner.arena,
                        self.raw_id(),
                        var_id,
                    )?;
                    match coeffs.last() {
                        Some(&lc) => lc,
                        None => inner.arena.zero,
                    }
                }
            }
        };
        Some(self.wrap(id))
    }

    // ── Sign on an interval ───────────────────────────────────────────────

    /// Is `self`, a univariate polynomial in `var` with rational
    /// coefficients, `≥ 0` at every point of the closed interval `[lo, hi]`?
    ///
    /// The decision is exact: the square-free decomposition isolates the
    /// roots of odd multiplicity (the only places where the sign changes), a
    /// Sturm count checks that none lies strictly inside the interval, and
    /// the constant sign on the rest of the interval is read off at one
    /// point.  Endpoints must be exact rationals or `±∞`
    /// ([`Context::infinity`](crate::api::context::Context::infinity),
    /// [`Context::neg_infinity`](crate::api::context::Context::neg_infinity)).
    /// An empty interval (`lo > hi`) is vacuously `Some(true)`; the zero
    /// polynomial is `Some(true)`.
    ///
    /// Returns `None` for non-polynomial input, symbolic coefficients, or
    /// endpoints that are neither rational nor infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let (ninf, inf) = (ctx.neg_infinity(), ctx.infinity());
    /// // (x − 1)² ≥ 0 everywhere
    /// let sq = &x.powi(2) - &x * 2 + 1;
    /// assert_eq!(sq.poly_is_nonnegative_on(&x, &ninf, &inf), Some(true));
    /// // x³ − x is ≥ 0 on [2, ∞) but not on [−2, ∞)
    /// let f = &x.powi(3) - &x;
    /// assert_eq!(f.poly_is_nonnegative_on(&x, &ctx.int(2), &inf), Some(true));
    /// assert_eq!(f.poly_is_nonnegative_on(&x, &ctx.int(-2), &inf), Some(false));
    /// assert_eq!(x.sin().poly_is_nonnegative_on(&x, &ninf, &inf), None);
    /// ```
    #[must_use]
    pub fn poly_is_nonnegative_on(&self, var: &Ex, lo: &Ex, hi: &Ex) -> Option<bool> {
        self.poly_sign_on(var, lo, hi, false)
    }

    /// Is `self`, a univariate polynomial in `var` with rational
    /// coefficients, `> 0` at every point of the closed interval `[lo, hi]`?
    ///
    /// Same method and conventions as
    /// [`poly_is_nonnegative_on`](Self::poly_is_nonnegative_on), but any
    /// root in the closed interval (including a root at an endpoint or a
    /// root of even multiplicity) makes the answer `Some(false)`, and the
    /// zero polynomial is `Some(false)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let (ninf, inf) = (ctx.neg_infinity(), ctx.infinity());
    /// assert_eq!((&x.powi(2) + 1).poly_is_positive_on(&x, &ninf, &inf), Some(true));
    /// // (x − 1)² touches zero at x = 1
    /// let sq = &x.powi(2) - &x * 2 + 1;
    /// assert_eq!(sq.poly_is_positive_on(&x, &ninf, &inf), Some(false));
    /// assert_eq!(sq.poly_is_positive_on(&x, &ctx.int(2), &inf), Some(true));
    /// ```
    #[must_use]
    pub fn poly_is_positive_on(&self, var: &Ex, lo: &Ex, hi: &Ex) -> Option<bool> {
        self.poly_sign_on(var, lo, hi, true)
    }

    /// Shared implementation of the interval sign tests.
    fn poly_sign_on(&self, var: &Ex, lo: &Ex, hi: &Ex, strict: bool) -> Option<bool> {
        let var_id = self.checked_id(var);
        let lo_id = self.checked_id(lo);
        let hi_id = self.checked_id(hi);
        let inner = self.inner.read();
        let f = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
        let lo_e = endpoint(&inner.arena, lo_id)?;
        let hi_e = endpoint(&inner.arena, hi_id)?;
        drop(inner);
        Some(poly_sign_on_interval(&f, &lo_e, &hi_e, strict))
    }

    /// Monic version of `self` (divide by the leading coefficient).
    ///
    /// Returns `None` for non-polynomial input or the zero polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.powi(2) * 2 + &x * 4;
    /// assert_eq!(format!("{}", f.monic(&x).unwrap()), "x^2 + 2*x");
    /// ```
    #[must_use]
    pub fn monic(&self, var: &Ex) -> Option<Ex> {
        let var_id = self.checked_id(var);
        let id = {
            let mut inner = self.inner.write();
            let f = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
            if f.is_zero() {
                return None;
            }
            let m = f.make_monic();
            poly_to_expr(&mut inner.arena, &m, var_id)
        };
        Some(self.wrap(id))
    }

    /// Composition `self(other)`: substitute the polynomial `other` for
    /// `var` and expand.
    ///
    /// Returns `None` if either expression is not polynomial in `var`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.powi(2) + 1;
    /// let g = &x - 1;
    /// assert_eq!(format!("{}", f.poly_compose(&g, &x).unwrap()), "x^2 - 2*x + 2");
    /// ```
    #[must_use]
    pub fn poly_compose(&self, other: &Ex, var: &Ex) -> Option<Ex> {
        let other_id = self.checked_id(other);
        let var_id = self.checked_id(var);
        let id = {
            let mut inner = self.inner.write();
            let f = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
            let g = expr_to_poly(&inner.arena, other_id, var_id)?;
            let c = f.compose(&g);
            poly_to_expr(&mut inner.arena, &c, var_id)
        };
        Some(self.wrap(id))
    }

    /// Taylor shift `self(var + a)` for a rational constant `a`.
    ///
    /// Returns `None` if `self` is not polynomial in `var` or `a` is not
    /// an exact rational number.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = x.powi(2);
    /// assert_eq!(format!("{}", f.poly_shift(&x, &ctx.int(1)).unwrap()), "x^2 + 2*x + 1");
    /// ```
    #[must_use]
    pub fn poly_shift(&self, var: &Ex, a: &Ex) -> Option<Ex> {
        let var_id = self.checked_id(var);
        let a_id = self.checked_id(a);
        let id = {
            let mut inner = self.inner.write();
            let f = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
            let shift = inner.arena.as_num(a_id)?.clone();
            let s = f.taylor_shift(&shift);
            poly_to_expr(&mut inner.arena, &s, var_id)
        };
        Some(self.wrap(id))
    }

    /// Reciprocal polynomial `varⁿ · self(1/var)` (coefficients reversed).
    ///
    /// Returns `None` if `self` is not polynomial in `var`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.powi(3) * 2 + &x * 3 + 5;
    /// assert_eq!(format!("{}", f.poly_reverse(&x).unwrap()), "5*x^3 + 3*x^2 + 2");
    /// ```
    #[must_use]
    pub fn poly_reverse(&self, var: &Ex) -> Option<Ex> {
        let var_id = self.checked_id(var);
        let id = {
            let mut inner = self.inner.write();
            let f = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
            let r = f.reverse();
            poly_to_expr(&mut inner.arena, &r, var_id)
        };
        Some(self.wrap(id))
    }

    /// Lagrange interpolation: the unique polynomial in `var` of degree
    /// `< points.len()` passing through the given `(x, y)` points.
    ///
    /// All coordinates must be exact rational numbers (integers or
    /// `Context::rational`).  Returns `None` for non-rational coordinates
    /// or duplicate abscissae.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let pts = [(ctx.int(0), ctx.int(1)), (ctx.int(1), ctx.int(2)), (ctx.int(2), ctx.int(5))];
    /// let p = Ex::poly_interpolate(&pts, &x).unwrap();
    /// assert_eq!(format!("{p}"), "x^2 + 1");
    /// ```
    #[must_use]
    pub fn poly_interpolate(points: &[(Ex, Ex)], var: &Ex) -> Option<Ex> {
        let var_id = var.raw_id();
        let id = {
            let mut inner = var.inner.write();
            let mut pts: Vec<(Ratio<BigInt>, Ratio<BigInt>)> = Vec::with_capacity(points.len());
            for (px, py) in points {
                let xi = var.checked_id(px);
                let yi = var.checked_id(py);
                let xr = inner.arena.as_num(xi)?.clone();
                let yr = inner.arena.as_num(yi)?.clone();
                pts.push((xr, yr));
            }
            let p = crate::poly::dense::lagrange_interpolate_points(&pts)?;
            poly_to_expr(&mut inner.arena, &p, var_id)
        };
        Some(var.wrap(id))
    }

    // ── Real roots (Sturm) ─────────────────────────────────────────

    /// Number of distinct real roots of `self` as a polynomial in `var`.
    ///
    /// Uses a Sturm sequence, so the count is exact.  Returns `None` for
    /// non-polynomial or constant input.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!((&x.powi(3) - &x).count_real_roots(&x), Some(3));
    /// assert_eq!((&x.powi(2) + 1).count_real_roots(&x), Some(0));
    /// ```
    #[must_use]
    pub fn count_real_roots(&self, var: &Ex) -> Option<usize> {
        let var_id = self.checked_id(var);
        let inner = self.inner.read();
        let f = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
        if f.degree().unwrap_or(0) == 0 {
            return None;
        }
        Some(SturmChain::new(&f).count_real_roots())
    }

    /// Number of distinct real roots in the closed interval `[lo, hi]`
    /// (Sturm's theorem; SymPy's `Poly.count_roots(inf, sup)`).
    ///
    /// `lo` and `hi` must be exact rational numbers or `±∞`
    /// (`Context::infinity`, `Context::neg_infinity`).  Returns `None` for
    /// non-polynomial or constant input, or non-rational bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.powi(3) - &x;   // roots −1, 0, 1
    /// assert_eq!(f.count_real_roots_in(&x, &ctx.int(0), &ctx.int(5)), Some(2));
    /// assert_eq!(f.count_real_roots_in(&x, &ctx.rational(1, 2), &ctx.infinity()), Some(1));
    /// ```
    #[must_use]
    pub fn count_real_roots_in(&self, var: &Ex, lo: &Ex, hi: &Ex) -> Option<usize> {
        let var_id = self.checked_id(var);
        let lo_id = self.checked_id(lo);
        let hi_id = self.checked_id(hi);
        let inner = self.inner.read();
        let f = expr_to_poly(&inner.arena, self.raw_id(), var_id)?;
        if f.degree().unwrap_or(0) == 0 {
            return None;
        }
        let lo_e = endpoint(&inner.arena, lo_id)?;
        let hi_e = endpoint(&inner.arena, hi_id)?;
        let chain = SturmChain::new(&f);
        let bound = crate::poly::sturm::cauchy_bound(&f) + Ratio::one();
        let lo_r = match lo_e {
            Endpoint::Finite(r) => r,
            Endpoint::NegInf => -bound.clone(),
            Endpoint::PosInf => return Some(0),
        };
        let hi_r = match hi_e {
            Endpoint::Finite(r) => r,
            Endpoint::PosInf => bound,
            Endpoint::NegInf => return Some(0),
        };
        Some(chain.count_roots_in_closed(&lo_r, &hi_r))
    }

    /// Isolating intervals for the distinct real roots of `self` in `var`.
    ///
    /// Each returned [`Interval`] has exact rational endpoints, contains
    /// exactly one real root, and has width at most `1/1024`.  Its `kind`
    /// says where the root may lie: a Sturm bisection cell is the half-open
    /// `(lo, hi]` ([`IntervalKind::LeftOpen`](crate::IntervalKind::LeftOpen);
    /// the root is never `lo` and, since an exact hit is reported
    /// separately, never `hi` either), while a root that a bisection point
    /// lands on exactly is the closed singleton `[r, r]`
    /// ([`IntervalKind::Closed`](crate::IntervalKind::Closed) with
    /// `lower == upper`).  Intervals are sorted.  Non-polynomial or
    /// constant input yields an empty vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let iv = (&x.powi(2) - 2).real_roots_isolate(&x);
    /// assert_eq!(iv.len(), 2);
    /// let lo = iv[1].lower.eval_f64().unwrap();
    /// let hi = iv[1].upper.eval_f64().unwrap();
    /// assert!(lo <= 2f64.sqrt() && 2f64.sqrt() <= hi);
    /// assert_eq!(iv[1].kind, IntervalKind::LeftOpen);
    ///
    /// // x³ − x: the bisection lands on the root 0 exactly, so that one is
    /// // the point [0, 0]; ±1 are bracketed by (lo, hi] cells.
    /// let iv = (&x.powi(3) - &x).real_roots_isolate(&x);
    /// assert_eq!(iv.len(), 3);
    /// assert_eq!(iv[1].kind, IntervalKind::Closed);
    /// assert_eq!(iv[1].lower, ctx.int(0));
    /// assert_eq!(iv[1].upper, ctx.int(0));
    /// ```
    #[must_use]
    pub fn real_roots_isolate(&self, var: &Ex) -> Vec<Interval<Ex>> {
        let var_id = self.checked_id(var);
        let ids: Vec<Interval<ExprId>> = {
            let mut inner = self.inner.write();
            let Some(f) = expr_to_poly(&inner.arena, self.raw_id(), var_id) else {
                return vec![];
            };
            if f.degree().unwrap_or(0) == 0 {
                return vec![];
            }
            let chain = SturmChain::new(&f);
            let width = Ratio::new(BigInt::one(), BigInt::from(1024));
            chain
                .isolate_all_real_roots()
                .into_iter()
                .map(|iv| {
                    let iv = if iv.is_point() {
                        iv
                    } else {
                        chain.refine_interval(&iv, &width)
                    };
                    iv.map(|q| inner.arena.num_ratio(q))
                })
                .collect()
        };
        ids.into_iter()
            .map(|iv| iv.map(|id| self.wrap(id)))
            .collect()
    }

    // ── Numeric roots ──────────────────────────────────────────────

    /// All complex roots of `self` as a polynomial in `var`, numerically,
    /// as `(re, im)` pairs sorted by real then imaginary part.  A `k`-fold
    /// root appears `k` times.
    ///
    /// `digits` requests the working precision (clamped to a sensible
    /// range; the output is `f64` so more than ~16 digits has no visible
    /// effect).  Uses Aberth–Ehrlich simultaneous iteration on each
    /// square-free part.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `self` is not a non-constant polynomial in
    /// `var`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let roots = (&x.powi(2) + 1).nroots(&x, 15).unwrap();
    /// assert_eq!(roots.len(), 2);
    /// assert!(roots.iter().all(|(re, im)| re.abs() < 1e-12 && (im.abs() - 1.0).abs() < 1e-12));
    /// ```
    pub fn nroots(&self, var: &Ex, digits: u32) -> Result<Vec<(f64, f64)>, SymplexError> {
        let var_id = self.checked_id(var);
        let inner = self.inner.read();
        let f = expr_to_poly(&inner.arena, self.raw_id(), var_id).ok_or_else(|| {
            SymplexError::InvalidArgument {
                operation: "nroots",
                reason: "expression is not a polynomial in the given variable".into(),
            }
        })?;
        if f.degree().unwrap_or(0) == 0 {
            return Err(SymplexError::InvalidArgument {
                operation: "nroots",
                reason: "polynomial must have degree ≥ 1".into(),
            });
        }
        // ~3.33 bits per decimal digit, plus guard bits; never below 128.
        let prec_bits = ((digits.clamp(1, 200) as f64) * 3.33).ceil() as usize + 64;
        let prec_bits = prec_bits.max(128);
        Ok(crate::poly::roots::nroots_f64(&f, prec_bits))
    }
}
