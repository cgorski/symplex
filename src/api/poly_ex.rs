//! Public polynomial view: [`Poly`] — an expression seen as a sparse
//! polynomial in an explicit list of generators, with symbolic or exact
//! rational coefficients.
//!
//! A `Poly` is built from an [`Ex`] with [`Poly::new`] (or
//! [`Ex::as_poly`]).  The expression is expanded and its terms are
//! collected by monomial in the generators; every coefficient is an `Ex`
//! that is free of the generators — an exact rational number or a symbolic
//! parameter expression such as `a + 1`.  The view is *exact*: no numeric
//! approximation happens anywhere, and [`Poly::to_ex`] rebuilds an
//! expression equal to the original.
//!
//! Terms are reported in descending lexicographic order of their exponent
//! vectors (the order of SymPy's `Poly.terms()`), so the leading term of
//! `Poly(x² y + x y² + y³, x, y)` is `x² y`.
//!
//! # Examples
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::poly_ex::Poly;
//!
//! let ctx = Context::new();
//! let (x, y, a) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a"));
//! let e = &a * &x.powi(2) + &x * &y * 3 - &y + 1;
//! let p = Poly::new(&e, &[&x, &y]).unwrap();
//! assert_eq!(p.num_terms(), 4);
//! assert_eq!(p.total_degree(), Some(2));
//! assert_eq!(p.leading_coeff(), a);
//! assert_eq!(p.coeff_monomial(&[1, 1]).unwrap(), ctx.int(3));
//! assert_eq!(p.to_ex(), e.expand());
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::api::context::Context;
use crate::api::expr::{Ex, ExprType};
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::domains::matrix::Matrix;
use crate::poly::multipoly::{GrevLex, MultiPoly};
use crate::poly::polybridge;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn invalid(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::InvalidArgument {
        operation,
        reason: reason.into(),
    }
}

/// Wrap an arena id as an `Ex` of `ctx`.
fn wrap(ctx: &Context, id: ExprId) -> Ex {
    Ex::from_raw_parts(ctx.id, Arc::clone(&ctx.inner), id)
}

/// Canonical coefficient form: expanded and evaluated.
fn norm_coeff(arena: &mut Arena, id: ExprId) -> ExprId {
    let e = crate::transforms::expand::expand(arena, id);
    crate::transforms::eval::eval(arena, e)
}

/// Does `expr` mention any of `gens`?
fn mentions_any(arena: &Arena, expr: ExprId, gens: &[ExprId]) -> bool {
    gens.iter()
        .any(|&g| crate::base::walk::contains(arena, expr, g))
}

/// Sum duplicate monomials, normalise coefficients, drop zeros.
fn normalize_terms(arena: &mut Arena, raw: Vec<(Vec<u32>, ExprId)>) -> Vec<(Vec<u32>, ExprId)> {
    let mut buckets: BTreeMap<Vec<u32>, Vec<ExprId>> = BTreeMap::new();
    for (e, c) in raw {
        buckets.entry(e).or_default().push(c);
    }
    let mut out = Vec::with_capacity(buckets.len());
    for (e, cs) in buckets {
        let c = if cs.len() == 1 { cs[0] } else { arena.add(&cs) };
        let c = norm_coeff(arena, c);
        if !arena.is_zero_structural(c) {
            out.push((e, c));
        }
    }
    out
}

/// Build the product `c · Π genᵢ^eᵢ`.
fn monomial_expr(arena: &mut Arena, c: ExprId, exps: &[u32], gens: &[ExprId]) -> ExprId {
    let mut factors: Vec<ExprId> = Vec::with_capacity(exps.len() + 1);
    if c != arena.one {
        factors.push(c);
    }
    for (&e, &g) in exps.iter().zip(gens) {
        match e {
            0 => {}
            1 => factors.push(g),
            _ => {
                let e_id = arena.int(i64::from(e));
                factors.push(arena.pow(g, e_id));
            }
        }
    }
    match factors.len() {
        0 => arena.one,
        1 => factors[0],
        _ => arena.mul(&factors),
    }
}

/// Validate a generator list: distinct symbols of `ctx`.  Cross-context
/// generators trip the usual context guard.
fn validate_gens(
    ctx: &Context,
    gens: &[&Ex],
    operation: &'static str,
) -> Result<Vec<Ex>, SymplexError> {
    let probe = ctx.zero();
    let mut ids: Vec<ExprId> = Vec::with_capacity(gens.len());
    for g in gens {
        let id = probe.checked_id(g);
        if g.expr_type() != ExprType::Symbol {
            return Err(invalid(
                operation,
                format!("generator `{g}` is not a symbol"),
            ));
        }
        if ids.contains(&id) {
            return Err(invalid(operation, format!("duplicate generator `{g}`")));
        }
        ids.push(id);
    }
    Ok(gens.iter().map(|g| (*g).clone()).collect())
}

// ═══════════════════════════════════════════════════════════════════════════
// Poly
// ═══════════════════════════════════════════════════════════════════════════

/// A sparse multivariate polynomial view of an expression over explicit
/// generators, with coefficients that are expressions free of the
/// generators.
///
/// See the [module documentation](self) for an overview.  All arithmetic is
/// exact; coefficients are kept expanded and evaluated so that structurally
/// equal polynomials compare equal with [`Poly::equals`].
#[derive(Clone, Debug)]
pub struct Poly {
    ctx: Context,
    gens: Vec<Ex>,
    /// Non-zero terms keyed by exponent vector (ascending lex order in the
    /// map; every accessor reports them descending).
    terms: BTreeMap<Vec<u32>, Ex>,
}

// ── Construction ───────────────────────────────────────────────────────────

impl Poly {
    /// Internal constructor from already-normalised arena terms.
    fn from_normalized(ctx: &Context, gens: Vec<Ex>, ids: Vec<(Vec<u32>, ExprId)>) -> Poly {
        let terms = ids.into_iter().map(|(e, c)| (e, wrap(ctx, c))).collect();
        Poly {
            ctx: ctx.clone(),
            gens,
            terms,
        }
    }

    /// Internal constructor: normalise raw `(exponents, coefficient)` pairs.
    fn from_raw(ctx: &Context, gens: Vec<Ex>, raw: Vec<(Vec<u32>, ExprId)>) -> Poly {
        let ids = ctx.with_arena_mut(|arena| normalize_terms(arena, raw));
        Self::from_normalized(ctx, gens, ids)
    }

    fn gen_ids(&self) -> Vec<ExprId> {
        self.gens.iter().map(Ex::raw_id).collect()
    }

    fn raw_terms(&self) -> Vec<(Vec<u32>, ExprId)> {
        self.terms
            .iter()
            .map(|(e, c)| (e.clone(), c.raw_id()))
            .collect()
    }

    /// View `expr` as a polynomial in the generators `gens`.
    ///
    /// The expression is expanded and collected by monomial.  Coefficients
    /// are evaluated (`.eval()`) and zero coefficients are dropped.
    ///
    /// Returns `None` when `gens` is empty, contains duplicates or
    /// non-symbols, or when a generator occurs in a non-polynomial position:
    /// inside a function (`sin(x)`), under a negative or non-integer power
    /// (`x⁻¹`, `√x`), or in an exponent (`2^x`, `x^a`).  Other symbols may
    /// appear anywhere — they become part of the coefficients.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let (r, f, j) = (ctx.symbol("r"), ctx.symbol("f"), ctx.symbol("j"));
    /// let e = &j * &r.powi(2) + (&j + 1) * &r * &f + 3;
    /// let p = Poly::new(&e, &[&r]).unwrap();
    /// let coeffs: Vec<String> = p.all_coeffs().unwrap().iter().map(|c| c.to_string()).collect();
    /// assert_eq!(coeffs, ["j", "f*j + f", "3"]);
    ///
    /// assert!(Poly::new(&r.sin(), &[&r]).is_none());
    /// assert!(Poly::new(&r.powi(-1), &[&r]).is_none());
    /// assert!(Poly::new(&r.pow(&f), &[&r]).is_none());
    /// ```
    #[must_use]
    pub fn new(expr: &Ex, gens: &[&Ex]) -> Option<Poly> {
        if gens.is_empty() {
            return None;
        }
        let ctx = expr.context();
        let gens = validate_gens(&ctx, gens, "Poly::new").ok()?;
        let gen_ids: Vec<ExprId> = gens.iter().map(Ex::raw_id).collect();
        let ids = ctx.with_arena_mut(|arena| {
            let raw = polybridge::symbolic_multipoly_terms(arena, expr.raw_id(), &gen_ids)?;
            Some(normalize_terms(arena, raw))
        })?;
        Some(Self::from_normalized(&ctx, gens, ids))
    }

    /// Build a polynomial directly from `(exponent vector, coefficient)`
    /// pairs.  Repeated monomials are summed; zero coefficients are dropped.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `gens` is empty or invalid, an exponent vector
    /// has the wrong length, or a coefficient mentions a generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let p = Poly::from_terms(&ctx, &[&x, &y], vec![(vec![1, 0], ctx.int(2)), (vec![0, 2], ctx.int(-1))]).unwrap();
    /// assert_eq!(p.to_ex(), &x * 2 - &y.powi(2));
    /// assert!(Poly::from_terms(&ctx, &[&x], vec![(vec![1, 1], ctx.int(1))]).is_err());
    /// ```
    pub fn from_terms(
        ctx: &Context,
        gens: &[&Ex],
        terms: Vec<(Vec<u32>, Ex)>,
    ) -> Result<Poly, SymplexError> {
        const OP: &str = "Poly::from_terms";
        if gens.is_empty() {
            return Err(invalid(OP, "at least one generator is required"));
        }
        let gens = validate_gens(ctx, gens, OP)?;
        let gen_ids: Vec<ExprId> = gens.iter().map(Ex::raw_id).collect();
        let probe = ctx.zero();
        let mut raw: Vec<(Vec<u32>, ExprId)> = Vec::with_capacity(terms.len());
        for (e, c) in terms {
            if e.len() != gen_ids.len() {
                return Err(invalid(
                    OP,
                    format!(
                        "exponent vector has length {} but there are {} generators",
                        e.len(),
                        gen_ids.len()
                    ),
                ));
            }
            let cid = probe.checked_id(&c);
            let mentions = ctx.with_arena_mut(|arena| mentions_any(arena, cid, &gen_ids));
            if mentions {
                return Err(invalid(
                    OP,
                    format!("coefficient `{c}` mentions a generator"),
                ));
            }
            raw.push((e, cid));
        }
        Ok(Self::from_raw(ctx, gens, raw))
    }

    /// The zero polynomial over `gens`.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `gens` is empty or invalid.
    pub fn zero(ctx: &Context, gens: &[&Ex]) -> Result<Poly, SymplexError> {
        if gens.is_empty() {
            return Err(invalid("Poly::zero", "at least one generator is required"));
        }
        let gens = validate_gens(ctx, gens, "Poly::zero")?;
        Ok(Self::from_normalized(ctx, gens, vec![]))
    }

    /// The constant polynomial `1` over `gens`.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `gens` is empty or invalid.
    pub fn one(ctx: &Context, gens: &[&Ex]) -> Result<Poly, SymplexError> {
        Self::constant(ctx, gens, &ctx.one())
    }

    /// The constant polynomial `c` over `gens`.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `gens` is empty or invalid, or `c` mentions a
    /// generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    /// let c = Poly::constant(&ctx, &[&x], &(&a + 1)).unwrap();
    /// assert!(c.is_ground());
    /// assert!(Poly::constant(&ctx, &[&x], &x).is_err());
    /// ```
    pub fn constant(ctx: &Context, gens: &[&Ex], c: &Ex) -> Result<Poly, SymplexError> {
        if gens.is_empty() {
            return Err(invalid(
                "Poly::constant",
                "at least one generator is required",
            ));
        }
        let n = gens.len();
        Self::from_terms(ctx, gens, vec![(vec![0; n], c.clone())])
    }

    /// Rebuild a `Poly` from a rational-coefficient [`MultiPoly`] (the
    /// crate's Gröbner-basis representation), mapping variable `i` to
    /// `gens[i]`.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `gens` is invalid or its length differs from
    /// `mp.num_vars()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    /// use symplex::multipoly::MultiPoly;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let mp: MultiPoly = MultiPoly::var(2, 0).mul(&MultiPoly::var(2, 1)) + 1;   // xy + 1
    /// let p = Poly::from_multipoly(&ctx, &[&x, &y], &mp).unwrap();
    /// assert_eq!(p.to_ex(), &x * &y + 1);
    /// assert_eq!(p.to_multipoly().unwrap(), mp);
    /// ```
    pub fn from_multipoly(
        ctx: &Context,
        gens: &[&Ex],
        mp: &MultiPoly<GrevLex>,
    ) -> Result<Poly, SymplexError> {
        const OP: &str = "Poly::from_multipoly";
        if gens.is_empty() {
            return Err(invalid(OP, "at least one generator is required"));
        }
        if gens.len() != mp.num_vars() {
            return Err(invalid(
                OP,
                format!(
                    "{} generators but the polynomial has {} variables",
                    gens.len(),
                    mp.num_vars()
                ),
            ));
        }
        let gens = validate_gens(ctx, gens, OP)?;
        let ids: Vec<(Vec<u32>, ExprId)> = ctx.with_arena_mut(|arena| {
            mp.terms()
                .map(|(e, c)| {
                    let nid = arena.intern_num(c.clone());
                    (e.to_vec(), arena.intern(ExprNode::Num(nid)))
                })
                .collect()
        });
        Ok(Self::from_normalized(ctx, gens, ids))
    }
}

// ── Queries ────────────────────────────────────────────────────────────────

impl Poly {
    /// The generators, in the order given at construction.
    #[must_use]
    pub fn gens(&self) -> &[Ex] {
        &self.gens
    }

    /// Number of generators.
    #[must_use]
    pub fn num_gens(&self) -> usize {
        self.gens.len()
    }

    /// The context this polynomial belongs to.
    #[must_use]
    pub fn context(&self) -> Context {
        self.ctx.clone()
    }

    /// Is this the zero polynomial?
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Is this a constant (every generator has exponent zero)?  The zero
    /// polynomial is ground.
    #[must_use]
    pub fn is_ground(&self) -> bool {
        self.terms.keys().all(|e| e.iter().all(|&x| x == 0))
    }

    /// Exactly one generator?
    #[must_use]
    pub fn is_univariate(&self) -> bool {
        self.gens.len() == 1
    }

    /// Total degree at most one in every term (`a x + b y + c`, but not
    /// `x y`)?  The zero polynomial is linear.
    #[must_use]
    pub fn is_linear(&self) -> bool {
        self.terms.keys().all(|e| e.iter().sum::<u32>() <= 1)
    }

    /// Do all terms have the same total degree?  The zero polynomial is
    /// homogeneous.
    #[must_use]
    pub fn is_homogeneous(&self) -> bool {
        let mut degs = self.terms.keys().map(|e| e.iter().sum::<u32>());
        match degs.next() {
            None => true,
            Some(d) => degs.all(|x| x == d),
        }
    }

    /// Is every coefficient an exact rational number?
    #[must_use]
    pub fn has_rational_coeffs(&self) -> bool {
        self.terms.values().all(|c| c.as_rational().is_some())
    }

    /// Number of non-zero terms.
    #[must_use]
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// All `(exponent vector, coefficient)` pairs in descending
    /// lexicographic order (as SymPy's `Poly.terms()`).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let p = Poly::new(&(&x * &y + &y.powi(3) + &x.powi(2) * 2), &[&x, &y]).unwrap();
    /// let monoms: Vec<Vec<u32>> = p.terms().into_iter().map(|(m, _)| m).collect();
    /// assert_eq!(monoms, vec![vec![2, 0], vec![1, 1], vec![0, 3]]);
    /// ```
    #[must_use]
    pub fn terms(&self) -> Vec<(Vec<u32>, Ex)> {
        self.terms
            .iter()
            .rev()
            .map(|(e, c)| (e.clone(), c.clone()))
            .collect()
    }

    /// Exponent vectors in descending lexicographic order.
    #[must_use]
    pub fn monoms(&self) -> Vec<Vec<u32>> {
        self.terms.keys().rev().cloned().collect()
    }

    /// Coefficients in the order of [`terms`](Self::terms).
    #[must_use]
    pub fn coeffs(&self) -> Vec<Ex> {
        self.terms.values().rev().cloned().collect()
    }

    /// Coefficient of the monomial with exponents `exps`; zero if absent.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `exps.len() != self.num_gens()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let p = Poly::new(&(&x * &y * 7 - &y), &[&x, &y]).unwrap();
    /// assert_eq!(p.coeff_monomial(&[1, 1]).unwrap(), ctx.int(7));
    /// assert_eq!(p.coeff_monomial(&[2, 0]).unwrap(), ctx.int(0));
    /// assert!(p.coeff_monomial(&[1]).is_err());
    /// ```
    pub fn coeff_monomial(&self, exps: &[u32]) -> Result<Ex, SymplexError> {
        if exps.len() != self.gens.len() {
            return Err(invalid(
                "Poly::coeff_monomial",
                format!(
                    "exponent vector has length {} but there are {} generators",
                    exps.len(),
                    self.gens.len()
                ),
            ));
        }
        Ok(self
            .terms
            .get(exps)
            .cloned()
            .unwrap_or_else(|| self.ctx.zero()))
    }

    /// Total degree (largest exponent sum); `None` for the zero polynomial.
    #[must_use]
    pub fn total_degree(&self) -> Option<u32> {
        self.terms.keys().map(|e| e.iter().sum::<u32>()).max()
    }

    /// Degree in one generator; `None` for the zero polynomial or a `var`
    /// that is not a generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let p = Poly::new(&(&x.powi(3) * &y + &y.powi(2)), &[&x, &y]).unwrap();
    /// assert_eq!(p.degree_in(&x), Some(3));
    /// assert_eq!(p.degree_in(&y), Some(2));
    /// assert_eq!(p.degree_list(), vec![3, 2]);
    /// ```
    #[must_use]
    pub fn degree_in(&self, var: &Ex) -> Option<u32> {
        let i = self.gens.iter().position(|g| g == var)?;
        self.terms.keys().map(|e| e[i]).max()
    }

    /// Maximum exponent of each generator (all zeros for the zero
    /// polynomial).
    #[must_use]
    pub fn degree_list(&self) -> Vec<u32> {
        let mut out = vec![0u32; self.gens.len()];
        for e in self.terms.keys() {
            for (o, &x) in out.iter_mut().zip(e) {
                *o = (*o).max(x);
            }
        }
        out
    }

    /// Leading term under the lexicographic order; `None` for zero.
    #[must_use]
    pub fn leading_term(&self) -> Option<(Vec<u32>, Ex)> {
        self.terms
            .last_key_value()
            .map(|(e, c)| (e.clone(), c.clone()))
    }

    /// Leading coefficient under the lexicographic order; `0` for zero.
    #[must_use]
    pub fn leading_coeff(&self) -> Ex {
        self.terms
            .last_key_value()
            .map_or_else(|| self.ctx.zero(), |(_, c)| c.clone())
    }

    /// Leading monomial under the lexicographic order; `None` for zero.
    #[must_use]
    pub fn leading_monomial(&self) -> Option<Vec<u32>> {
        self.terms.last_key_value().map(|(e, _)| e.clone())
    }

    /// Dense coefficient list of a univariate polynomial, highest degree
    /// first and with zeros included (SymPy `all_coeffs`); `None` unless
    /// there is exactly one generator.  The zero polynomial gives `[0]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let p = Poly::new(&(&x.powi(3) * 2 - 5), &[&x]).unwrap();
    /// let cs: Vec<String> = p.all_coeffs().unwrap().iter().map(|c| c.to_string()).collect();
    /// assert_eq!(cs, ["2", "0", "0", "-5"]);
    /// ```
    #[must_use]
    pub fn all_coeffs(&self) -> Option<Vec<Ex>> {
        if self.gens.len() != 1 {
            return None;
        }
        let deg = self.terms.keys().map(|e| e[0]).max().unwrap_or(0);
        let zero = self.ctx.zero();
        let mut out = vec![zero; deg as usize + 1];
        for (e, c) in &self.terms {
            out[(deg - e[0]) as usize] = c.clone();
        }
        Some(out)
    }

    /// Structural equality: same generators and identical (normalised)
    /// coefficients for every monomial.
    #[must_use]
    pub fn equals(&self, other: &Poly) -> bool {
        self.gens == other.gens && self.terms == other.terms
    }
}

// ── Conversion and evaluation ──────────────────────────────────────────────

impl Poly {
    /// Rebuild the expression `Σ cᵢ · monomialᵢ`.
    ///
    /// The result equals the expanded input the view was built from.
    #[must_use]
    pub fn to_ex(&self) -> Ex {
        let gens = self.gen_ids();
        let raw = self.raw_terms();
        let id = self.ctx.with_arena_mut(|arena| {
            let terms: Vec<ExprId> = raw
                .iter()
                .map(|(e, c)| monomial_expr(arena, *c, e, &gens))
                .collect();
            match terms.len() {
                0 => arena.zero,
                1 => terms[0],
                _ => arena.add(&terms),
            }
        });
        wrap(&self.ctx, id)
    }

    /// Evaluate at a point: substitute `values[i]` for `gens[i]` and
    /// evaluate.  Values may be numbers or expressions.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `values.len() != self.num_gens()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let p = Poly::new(&(&x.powi(2) + &y), &[&x, &y]).unwrap();
    /// let v = p.eval(&[&ctx.rational(1, 2), &ctx.rational(1, 3)]).unwrap();
    /// assert_eq!(v, ctx.rational(7, 12));
    /// ```
    pub fn eval(&self, values: &[&Ex]) -> Result<Ex, SymplexError> {
        if values.len() != self.gens.len() {
            return Err(invalid(
                "Poly::eval",
                format!("{} values for {} generators", values.len(), self.gens.len()),
            ));
        }
        let probe = self.ctx.zero();
        let value_ids: Vec<ExprId> = values.iter().map(|v| probe.checked_id(v)).collect();
        let raw = self.raw_terms();
        let id = self.ctx.with_arena_mut(|arena| {
            let terms: Vec<ExprId> = raw
                .iter()
                .map(|(e, c)| monomial_expr(arena, *c, e, &value_ids))
                .collect();
            match terms.len() {
                0 => arena.zero,
                1 => terms[0],
                _ => arena.add(&terms),
            }
        });
        Ok(wrap(&self.ctx, id).eval())
    }

    /// Partial evaluation: substitute `value` for the generator `var`,
    /// which is removed from the generator list.  `value` may itself
    /// involve the remaining generators polynomially.  Evaluating the last
    /// generator yields a ground polynomial with no generators.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `var` is not a generator, if `value` mentions
    /// `var`, or if `value` is not polynomial in the remaining generators.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let p = Poly::new(&(&x.powi(2) * &y + &x + &y), &[&x, &y]).unwrap();
    /// let q = p.eval_gen(&x, &ctx.int(2)).unwrap();          // 4y + 2 + y = 5y + 2
    /// assert_eq!(q.gens().len(), 1);
    /// assert_eq!(q.to_ex(), &y * 5 + 2);
    /// ```
    pub fn eval_gen(&self, var: &Ex, value: &Ex) -> Result<Poly, SymplexError> {
        const OP: &str = "Poly::eval_gen";
        let probe = self.ctx.zero();
        let gen_id = probe.checked_id(var);
        let value_id = probe.checked_id(value);
        let Some(i) = self.gens.iter().position(|g| g == var) else {
            return Err(invalid(OP, format!("`{var}` is not a generator")));
        };
        let remaining: Vec<Ex> = self
            .gens
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != i)
            .map(|(_, g)| g.clone())
            .collect();
        let remaining_ids: Vec<ExprId> = remaining.iter().map(Ex::raw_id).collect();
        let raw = self.raw_terms();
        let result: Result<Vec<(Vec<u32>, ExprId)>, SymplexError> =
            self.ctx.with_arena_mut(|arena| {
                if crate::base::walk::contains(arena, value_id, gen_id) {
                    return Err(invalid(
                        OP,
                        format!(
                            "value `{}` mentions the generator itself",
                            arena.display(value_id)
                        ),
                    ));
                }
                // Substitute: cᵢ · value^{eᵢ} with the generator's exponent removed.
                let mut subst: Vec<ExprId> = Vec::with_capacity(raw.len());
                for (e, c) in &raw {
                    let mut rest: Vec<u32> = e.clone();
                    let k = rest.remove(i);
                    let mut factors = vec![*c];
                    match k {
                        0 => {}
                        1 => factors.push(value_id),
                        _ => {
                            let k_id = arena.int(i64::from(k));
                            factors.push(arena.pow(value_id, k_id));
                        }
                    }
                    let coeff = if factors.len() == 1 {
                        factors[0]
                    } else {
                        arena.mul(&factors)
                    };
                    subst.push(monomial_expr(arena, coeff, &rest, &remaining_ids));
                }
                let total = match subst.len() {
                    0 => arena.zero,
                    1 => subst[0],
                    _ => arena.add(&subst),
                };
                if remaining_ids.is_empty() {
                    let c = norm_coeff(arena, total);
                    return Ok(if arena.is_zero_structural(c) {
                        vec![]
                    } else {
                        vec![(vec![], c)]
                    });
                }
                match polybridge::symbolic_multipoly_terms(arena, total, &remaining_ids) {
                    Some(terms) => Ok(normalize_terms(arena, terms)),
                    None => Err(invalid(
                        OP,
                        "value is not polynomial in the remaining generators",
                    )),
                }
            });
        Ok(Self::from_normalized(&self.ctx, remaining, result?))
    }

    /// Convert to the crate's rational-coefficient [`MultiPoly`]
    /// representation (usable with [`groebner`](crate::groebner) and
    /// [`polysys`](crate::polysys)).  `None` if any coefficient is not an
    /// exact rational number.
    #[must_use]
    pub fn to_multipoly(&self) -> Option<MultiPoly<GrevLex>> {
        let mut terms: Vec<(Vec<u32>, Ratio<BigInt>)> = Vec::with_capacity(self.terms.len());
        for (e, c) in &self.terms {
            terms.push((e.clone(), c.as_rational()?));
        }
        MultiPoly::from_terms(self.gens.len(), terms)
    }

    /// Numeric complex roots of a univariate polynomial with rational
    /// coefficients, as `(re, im)` pairs (see [`Ex::nroots`]).
    ///
    /// # Errors
    ///
    /// `InvalidArgument` unless the polynomial is univariate, has rational
    /// coefficients and degree at least one.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let p = Poly::new(&(&x.powi(2) - 2), &[&x]).unwrap();
    /// let roots = p.nroots(15).unwrap();
    /// assert!((roots[1].0 - 2f64.sqrt()).abs() < 1e-12);
    /// ```
    pub fn nroots(&self, digits: u32) -> Result<Vec<(f64, f64)>, SymplexError> {
        if self.gens.len() != 1 {
            return Err(invalid("Poly::nroots", "polynomial must be univariate"));
        }
        if !self.has_rational_coeffs() {
            return Err(invalid(
                "Poly::nroots",
                "polynomial must have rational coefficients",
            ));
        }
        self.to_ex().nroots(&self.gens[0], digits)
    }
}

// ── Arithmetic ─────────────────────────────────────────────────────────────

impl Poly {
    fn check_same_gens(&self, other: &Poly, operation: &'static str) -> Result<(), SymplexError> {
        if self.gens != other.gens {
            return Err(invalid(operation, "polynomials have different generators"));
        }
        Ok(())
    }

    /// Sum.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if the generators differ.
    pub fn add(&self, other: &Poly) -> Result<Poly, SymplexError> {
        self.check_same_gens(other, "Poly::add")?;
        let mut raw = self.raw_terms();
        raw.extend(other.raw_terms());
        Ok(Self::from_raw(&self.ctx, self.gens.clone(), raw))
    }

    /// Difference.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if the generators differ.
    pub fn sub(&self, other: &Poly) -> Result<Poly, SymplexError> {
        self.check_same_gens(other, "Poly::sub")?;
        let mine = self.raw_terms();
        let theirs = other.raw_terms();
        let ids = self.ctx.with_arena_mut(|arena| {
            let mut raw = mine;
            for (e, c) in theirs {
                raw.push((e, arena.neg(c)));
            }
            normalize_terms(arena, raw)
        });
        Ok(Self::from_normalized(&self.ctx, self.gens.clone(), ids))
    }

    /// Product.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if the generators differ or an exponent overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let s = Poly::new(&(&x + &y), &[&x, &y]).unwrap();
    /// let d = Poly::new(&(&x - &y), &[&x, &y]).unwrap();
    /// assert_eq!(s.mul(&d).unwrap().to_ex(), &x.powi(2) - &y.powi(2));
    /// ```
    pub fn mul(&self, other: &Poly) -> Result<Poly, SymplexError> {
        self.check_same_gens(other, "Poly::mul")?;
        let mine = self.raw_terms();
        let theirs = other.raw_terms();
        let ids: Result<Vec<(Vec<u32>, ExprId)>, SymplexError> = self.ctx.with_arena_mut(|arena| {
            let mut raw: Vec<(Vec<u32>, ExprId)> = Vec::with_capacity(mine.len() * theirs.len());
            for (e1, c1) in &mine {
                for (e2, c2) in &theirs {
                    let mut e = Vec::with_capacity(e1.len());
                    for (a, b) in e1.iter().zip(e2) {
                        e.push(
                            a.checked_add(*b)
                                .ok_or_else(|| invalid("Poly::mul", "exponent overflow"))?,
                        );
                    }
                    raw.push((e, arena.mul(&[*c1, *c2])));
                }
            }
            Ok(normalize_terms(arena, raw))
        });
        Ok(Self::from_normalized(&self.ctx, self.gens.clone(), ids?))
    }

    /// Negation.
    #[must_use]
    pub fn neg(&self) -> Poly {
        let raw = self.raw_terms();
        let ids = self.ctx.with_arena_mut(|arena| {
            let negated: Vec<(Vec<u32>, ExprId)> =
                raw.into_iter().map(|(e, c)| (e, arena.neg(c))).collect();
            normalize_terms(arena, negated)
        });
        Self::from_normalized(&self.ctx, self.gens.clone(), ids)
    }

    /// Multiply every coefficient by `c`.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `c` mentions a generator (use
    /// [`mul`](Self::mul) with a polynomial instead).
    pub fn scale(&self, c: &Ex) -> Result<Poly, SymplexError> {
        let cid = self.ctx.zero().checked_id(c);
        let gens = self.gen_ids();
        let raw = self.raw_terms();
        let ids: Result<Vec<(Vec<u32>, ExprId)>, SymplexError> = self.ctx.with_arena_mut(|arena| {
            if mentions_any(arena, cid, &gens) {
                return Err(invalid("Poly::scale", "scale factor mentions a generator"));
            }
            let scaled: Vec<(Vec<u32>, ExprId)> = raw
                .into_iter()
                .map(|(e, coeff)| (e, arena.mul(&[cid, coeff])))
                .collect();
            Ok(normalize_terms(arena, scaled))
        });
        Ok(Self::from_normalized(&self.ctx, self.gens.clone(), ids?))
    }

    /// Integer power by repeated squaring (`p⁰ = 1`).
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if an exponent overflows `u32`.
    pub fn pow(&self, n: u32) -> Result<Poly, SymplexError> {
        let gens: Vec<&Ex> = self.gens.iter().collect();
        let mut result = Self::one(&self.ctx, &gens)?;
        let mut base = self.clone();
        let mut k = n;
        while k > 0 {
            if k & 1 == 1 {
                result = result.mul(&base)?;
            }
            k >>= 1;
            if k > 0 {
                base = base.mul(&base)?;
            }
        }
        Ok(result)
    }

    /// Partial derivative with respect to a generator.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `var` is not a generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let p = Poly::new(&(&x.powi(3) * &y + &y.powi(2)), &[&x, &y]).unwrap();
    /// assert_eq!(p.derivative(&x).unwrap().to_ex(), &x.powi(2) * &y * 3);
    /// ```
    pub fn derivative(&self, var: &Ex) -> Result<Poly, SymplexError> {
        let Some(i) = self.gens.iter().position(|g| g == var) else {
            return Err(invalid(
                "Poly::derivative",
                format!("`{var}` is not a generator"),
            ));
        };
        let raw = self.raw_terms();
        let ids = self.ctx.with_arena_mut(|arena| {
            let mut out: Vec<(Vec<u32>, ExprId)> = Vec::with_capacity(raw.len());
            for (mut e, c) in raw {
                let k = e[i];
                if k == 0 {
                    continue;
                }
                e[i] -= 1;
                let k_id = arena.int(i64::from(k));
                out.push((e, arena.mul(&[k_id, c])));
            }
            normalize_terms(arena, out)
        });
        Ok(Self::from_normalized(&self.ctx, self.gens.clone(), ids))
    }

    /// Split a rational-coefficient polynomial into `(content, primitive)`
    /// with `self = content · primitive`, the content being the rational
    /// GCD of the coefficients signed so that the primitive part's leading
    /// coefficient is positive.  The zero polynomial gives `(0, 0)`.
    ///
    /// `None` if some coefficient is not an exact rational number.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let p = Poly::new(&(&x.powi(2) * -4 + &x * 6), &[&x]).unwrap();
    /// let (c, prim) = p.content_and_primitive().unwrap();
    /// assert_eq!(c, ctx.int(-2));
    /// assert_eq!(prim.to_ex(), &x.powi(2) * 2 - &x * 3);
    /// ```
    #[must_use]
    pub fn content_and_primitive(&self) -> Option<(Ex, Poly)> {
        let mp = self.to_multipoly()?;
        if mp.is_zero() {
            return Some((self.ctx.zero(), self.clone()));
        }
        let mut num = BigInt::zero();
        let mut den = BigInt::one();
        for (_, c) in mp.terms() {
            num = num.gcd(c.numer());
            den = den.lcm(c.denom());
        }
        let mut content = Ratio::new(num, den);
        if self
            .leading_coeff()
            .as_rational()
            .is_some_and(|lc| lc.is_negative())
        {
            content = -content;
        }
        let inv = Ratio::one() / &content;
        let prim = self.scale(&self.ctx.from_ratio(inv)).ok()?;
        Some((self.ctx.from_ratio(content), prim))
    }

    /// Divide by the leading coefficient so it becomes `1`.
    ///
    /// `None` for the zero polynomial or non-rational coefficients.
    #[must_use]
    pub fn monic(&self) -> Option<Poly> {
        if self.is_zero() {
            return None;
        }
        let lc = self.leading_coeff().as_rational()?;
        if !self.has_rational_coeffs() {
            return None;
        }
        let inv = Ratio::one() / lc;
        self.scale(&self.ctx.from_ratio(inv)).ok()
    }
}

// ── Families of polynomials ────────────────────────────────────────────────

impl Poly {
    /// Union of the monomials of several polynomials over the same
    /// generators, in descending lexicographic order.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `polys` is empty or the generators differ.
    pub fn monomial_basis(polys: &[&Poly]) -> Result<Vec<Vec<u32>>, SymplexError> {
        let Some(first) = polys.first() else {
            return Err(invalid(
                "Poly::monomial_basis",
                "at least one polynomial is required",
            ));
        };
        let mut set: BTreeSet<Vec<u32>> = BTreeSet::new();
        for p in polys {
            first.check_same_gens(p, "Poly::monomial_basis")?;
            set.extend(p.terms.keys().cloned());
        }
        Ok(set.into_iter().rev().collect())
    }

    /// Coefficient matrix of a family of polynomials: row `i` is the
    /// monomial `monos[i]`, column `j` is `polys[j]`, and the entry is the
    /// coefficient of that monomial in that polynomial (zero if absent).
    ///
    /// Together with [`monomial_basis`](Self::monomial_basis) this turns
    /// "find λ with `goal = Σ λⱼ pⱼ`" into the linear system
    /// `M λ = coefficients of goal`.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `polys` or `monos` is empty, the generators
    /// differ, or an exponent vector has the wrong length.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let p1 = Poly::new(&(&x + 1), &[&x]).unwrap();
    /// let p2 = Poly::new(&(&x.powi(2) - 1), &[&x]).unwrap();
    /// let basis = Poly::monomial_basis(&[&p1, &p2]).unwrap();
    /// assert_eq!(basis, vec![vec![2], vec![1], vec![0]]);
    /// let m = Poly::coefficient_matrix(&[&p1, &p2], &basis).unwrap();
    /// assert_eq!(m.shape(), (3, 2));
    /// assert_eq!(m, matrix![ctx, [0, 1], [1, 0], [1, -1]]);
    /// ```
    pub fn coefficient_matrix(polys: &[&Poly], monos: &[Vec<u32>]) -> Result<Matrix, SymplexError> {
        const OP: &str = "Poly::coefficient_matrix";
        let Some(first) = polys.first() else {
            return Err(invalid(OP, "at least one polynomial is required"));
        };
        if monos.is_empty() {
            return Err(invalid(OP, "at least one monomial is required"));
        }
        for p in polys {
            first.check_same_gens(p, OP)?;
        }
        let mut rows: Vec<Vec<Ex>> = Vec::with_capacity(monos.len());
        for m in monos {
            let mut row = Vec::with_capacity(polys.len());
            for p in polys {
                row.push(p.coeff_monomial(m)?);
            }
            rows.push(row);
        }
        Matrix::new(rows)
    }
}

/// `Poly(<terms in lex-descending order>, g₁, g₂, …)`.
///
/// Unlike [`Poly::to_ex`], whose printed form follows the arena's canonical
/// term order, the terms here appear in the same order as
/// [`Poly::terms`], so the leading term is printed first:
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let (x, y, a) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a"));
/// let p = (&x + 2 * &y).powi(2).as_poly(&[&x, &y]).unwrap();
/// assert_eq!(p.to_string(), "Poly(x^2 + 4*x*y + 4*y^2, x, y)");
/// let q = (&a * &x.powi(2) - (&a + 1) * &x - 3).as_poly(&[&x]).unwrap();
/// assert_eq!(q.to_string(), "Poly(a*x^2 + (-a - 1)*x - 3, x)");   // as SymPy prints it
/// ```
impl fmt::Display for Poly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Poly(")?;
        if self.terms.is_empty() {
            write!(f, "0")?;
        }
        for (i, (exps, coeff)) in self.terms.iter().rev().enumerate() {
            let monomial: Vec<String> = exps
                .iter()
                .zip(&self.gens)
                .filter(|(e, _)| **e > 0)
                .map(|(e, g)| {
                    if *e == 1 {
                        g.to_string()
                    } else {
                        format!("{g}^{e}")
                    }
                })
                .collect();
            let monomial = monomial.join("*");

            // Sign and magnitude of the coefficient.
            let (negative, magnitude) = match coeff.as_rational() {
                Some(r) => {
                    let abs = r.abs();
                    let s = if abs.is_integer() {
                        abs.numer().to_string()
                    } else {
                        format!("{}/{}", abs.numer(), abs.denom())
                    };
                    (r.is_negative(), s)
                }
                None => {
                    let s = coeff.to_string();
                    if coeff.expr_type() == ExprType::Add && !monomial.is_empty() {
                        (false, format!("({s})"))
                    } else if let Some(rest) = s.strip_prefix('-') {
                        (true, rest.to_string())
                    } else {
                        (false, s)
                    }
                }
            };

            match (i, negative) {
                (0, false) => {}
                (0, true) => write!(f, "-")?,
                (_, false) => write!(f, " + ")?,
                (_, true) => write!(f, " - ")?,
            }
            if monomial.is_empty() {
                write!(f, "{magnitude}")?;
            } else if magnitude == "1" {
                write!(f, "{monomial}")?;
            } else {
                write!(f, "{magnitude}*{monomial}")?;
            }
        }
        for g in &self.gens {
            write!(f, ", {g}")?;
        }
        write!(f, ")")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Ex::as_poly
// ═══════════════════════════════════════════════════════════════════════════

impl Ex {
    /// View this expression as a [`Poly`] in `gens`; see [`Poly::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    /// let p = (&a * &x.powi(2) + 1).as_poly(&[&x]).unwrap();
    /// assert_eq!(p.degree_in(&x), Some(2));
    /// assert_eq!(p.leading_coeff(), a);
    /// assert!(x.sin().as_poly(&[&x]).is_none());
    /// ```
    #[must_use]
    pub fn as_poly(&self, gens: &[&Ex]) -> Option<Poly> {
        Poly::new(self, gens)
    }
}
