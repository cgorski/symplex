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
//! # Representation
//!
//! A polynomial whose coefficients are all rational literals is stored as
//! an exact [`MultiPoly`] in [`Lex`] order (the order of
//! [`Poly::terms`]), and arithmetic between two such polynomials runs on
//! rationals without touching the expression arena.  As soon as a
//! symbolic coefficient appears the polynomial is stored as one `Ex` per
//! monomial, and mixed operations convert the exact side to that form.
//! Both representations report the same terms in the same order.
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

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, OnceLock};

use num_bigint::BigInt;
use num_complex::Complex64;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

use crate::api::context::Context;
use crate::api::expr::{Ex, ExprType};
use crate::base::arena::Arena;
use crate::base::config::EvalConfig;
use crate::base::errors::SymplexError;
use crate::base::interval::Interval;
use crate::base::node::{ExprId, ExprNode};
use crate::domains::matrix::Matrix;
use crate::poly::multipoly::{GrevLex, Lex, MultiPoly};
use crate::poly::polybridge;
use crate::poly::zpoly::pow_ratio;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn invalid(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(operation, reason)
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

// ── Exact-arithmetic helpers ───────────────────────────────────────────────

/// The rational literal `r` as an arena node (the node `norm_coeff` would
/// produce for it: rationals intern to a unique `Num`).
fn intern_ratio(arena: &mut Arena, r: &Ratio<BigInt>) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(ExprNode::Num(nid))
}

/// Digit count of a rational as the arena's numeric guards measure it
/// (decimal digits of numerator and denominator, sign included).
fn digits(r: &Ratio<BigInt>) -> usize {
    r.numer().to_string().len() + r.denom().to_string().len()
}

/// Would the arena evaluate `base^exp` to a rational literal?  Mirrors the
/// guards of canonical `pow` (`max_pow_exponent`, `max_result_digits`)
/// conservatively — `true` only when the expression path is certain to
/// fold the power, so an exact computation returns the identical node.
fn pow_within_limits(config: &EvalConfig, base: &Ratio<BigInt>, exp: u32) -> bool {
    if exp <= 1 || base.is_one() {
        return true;
    }
    let exp = exp as usize;
    exp <= config.max_pow_exponent
        && exp
            .checked_mul(digits(base))
            .is_some_and(|d| d <= config.max_result_digits)
}

/// Maximum exponent of each variable (all zeros for the zero polynomial).
fn degree_list_of(mp: &MultiPoly<Lex>) -> Vec<u32> {
    let mut out = vec![0u32; mp.num_vars()];
    for (e, _) in mp.terms() {
        for (o, &x) in out.iter_mut().zip(e) {
            *o = (*o).max(x);
        }
    }
    out
}

/// Product of two exact polynomials, `None` on exponent overflow (the
/// check `Poly::mul` performs on the symbolic path).  Accumulated over a
/// common denominator in `ℤ` (see [`crate::poly::zpoly::ZPoly`]).
fn mul_checked(a: &MultiPoly<Lex>, b: &MultiPoly<Lex>) -> Option<MultiPoly<Lex>> {
    a.try_mul(b)
}

/// `base^n` by repeated squaring, `None` on exponent overflow.
fn pow_checked(base: &MultiPoly<Lex>, n: u32) -> Option<MultiPoly<Lex>> {
    base.try_pow(n)
}

/// Exact value of `mp` at the rational point `vals` (one reduction at the
/// end; see [`MultiPoly::eval`]).
fn eval_exact(mp: &MultiPoly<Lex>, vals: &[Ratio<BigInt>]) -> Ratio<BigInt> {
    mp.eval(vals)
}

/// `base^exp` for the exact reading of an expression, or `None` to defer
/// to the expand-based path.  Beyond the exponent being a non-negative
/// integer, the limits of that path are honoured so that both paths accept
/// the same inputs and produce the same coefficients: a power of a sum is
/// expanded only up to `min(max_pow_exponent, 200)`, and a numeric
/// coefficient is raised only within the `pow` guards.
fn exact_pow(
    config: &EvalConfig,
    base: &MultiPoly<Lex>,
    exp: &Ratio<BigInt>,
) -> Option<MultiPoly<Lex>> {
    if !exp.is_integer() || exp.is_negative() {
        return None;
    }
    let n: u32 = exp.to_integer().try_into().ok()?;
    let nv = base.num_vars();
    match n {
        0 => return (!base.is_zero()).then(|| MultiPoly::from_int(nv, 1)),
        1 => return Some(base.clone()),
        _ => {}
    }
    if base.num_terms() > 1 && n as usize > config.max_pow_exponent.min(200) {
        return None;
    }
    if base.terms().any(|(_, c)| !pow_within_limits(config, c, n)) {
        return None;
    }
    match base.num_terms() {
        0 => Some(MultiPoly::zero(nv)),
        1 => {
            let (e, c) = base.terms().next()?;
            let exps: Option<Vec<u32>> = e.iter().map(|&x| x.checked_mul(n)).collect();
            Some(MultiPoly::monomial(pow_ratio(c, n), exps?))
        }
        _ => pow_checked(base, n),
    }
}

/// One factor of a flat term folded into `(exps, coeff)`: a generator, an
/// integer power of one, a number, or a product / negation of those.
fn flat_factor(
    arena: &Arena,
    factor: ExprId,
    gens: &[ExprId],
    exps: &mut [u32],
    coeff: &mut Option<Ratio<BigInt>>,
) -> Option<()> {
    if let Some(i) = gens.iter().position(|&g| g == factor) {
        exps[i] = exps[i].checked_add(1)?;
        return Some(());
    }
    match arena.node(factor) {
        ExprNode::Num(nid) => {
            let c = arena.num(*nid);
            *coeff = Some(match coeff.take() {
                None => c.clone(),
                Some(acc) => acc * c,
            });
            Some(())
        }
        ExprNode::Pow(base, exp) => {
            let i = gens.iter().position(|&g| g == *base)?;
            let k = arena.as_num(*exp)?;
            if !k.is_integer() || k.is_negative() {
                return None;
            }
            let k: u32 = k.to_integer().try_into().ok()?;
            exps[i] = exps[i].checked_add(k)?;
            Some(())
        }
        ExprNode::Mul(children) => {
            for &c in children {
                flat_factor(arena, c, gens, exps, coeff)?;
            }
            Some(())
        }
        ExprNode::Neg(inner) => {
            flat_factor(arena, *inner, gens, exps, coeff)?;
            *coeff = Some(match coeff.take() {
                None => -Ratio::<BigInt>::one(),
                Some(acc) => -acc,
            });
            Some(())
        }
        _ => None,
    }
}

/// An already-expanded expression — a sum of `c · Π genᵢ^kᵢ` terms, the
/// usual shape after canonicalisation — read off as exact terms with no
/// polynomial arithmetic at all.  `None` if some term is not of that flat
/// shape; [`exact_multipoly`] then evaluates the tree.
fn flat_terms(
    arena: &Arena,
    expr: ExprId,
    gens: &[ExprId],
) -> Option<Vec<(Vec<u32>, Ratio<BigInt>)>> {
    let terms: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expr],
    };
    let mut out = Vec::with_capacity(terms.len());
    for term in terms {
        let mut exps = vec![0u32; gens.len()];
        let mut coeff = None;
        flat_factor(arena, term, gens, &mut exps, &mut coeff)?;
        out.push((exps, coeff.unwrap_or_else(Ratio::one)));
    }
    Some(out)
}

/// `expr` as an exact rational-coefficient polynomial in `gens`, by a
/// bottom-up walk over `Num`/`Add`/`Mul`/`Pow`/`Neg` nodes.  `None` when
/// some node is not of that shape (a symbol outside `gens`, a function, a
/// non-integer power, …) or a power exceeds the limits the expand-based
/// path honours ([`exact_pow`]); the caller then takes that path, so the
/// fast path never accepts an input the general one rejects.
fn exact_multipoly(arena: &Arena, expr: ExprId, gens: &[ExprId]) -> Option<MultiPoly<Lex>> {
    let nv = gens.len();
    let mut cache: FxHashMap<ExprId, MultiPoly<Lex>> = FxHashMap::default();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let Some(i) = gens.iter().position(|&g| g == id) {
            cache.insert(id, MultiPoly::var(nv, i));
            continue;
        }
        let poly = match arena.node(id) {
            ExprNode::Num(nid) => MultiPoly::constant(nv, arena.num(*nid).clone()),
            ExprNode::Add(children) => {
                let mut terms: Vec<(Vec<u32>, Ratio<BigInt>)> = Vec::new();
                for c in children {
                    terms.extend(cache.get(c)?.terms().map(|(e, r)| (e.to_vec(), r.clone())));
                }
                MultiPoly::from_distinct_terms(nv, terms)?
            }
            ExprNode::Mul(children) => {
                let mut acc = MultiPoly::from_int(nv, 1);
                for c in children {
                    acc = mul_checked(&acc, cache.get(c)?)?;
                }
                acc
            }
            ExprNode::Pow(base, exp) => {
                exact_pow(&arena.config, cache.get(base)?, arena.as_num(*exp)?)?
            }
            ExprNode::Neg(inner) => cache.get(inner)?.neg(),
            _ => return None,
        };
        cache.insert(id, poly);
    }
    cache.remove(&expr)
}

/// The symbolic term map as an exact polynomial, `None` if some
/// coefficient is not a rational literal.
fn rational_multipoly(nv: usize, terms: &BTreeMap<Vec<u32>, Ex>) -> Option<MultiPoly<Lex>> {
    let mut out: Vec<(Vec<u32>, Ratio<BigInt>)> = Vec::with_capacity(terms.len());
    for (e, c) in terms {
        out.push((e.clone(), c.as_rational()?));
    }
    MultiPoly::from_distinct_terms(nv, out)
}

// ═══════════════════════════════════════════════════════════════════════════
// Poly
// ═══════════════════════════════════════════════════════════════════════════

/// Rational-coefficient terms as an exact [`MultiPoly`] in `Lex` order —
/// the order of [`Poly::terms`] — with the `Ex` coefficients materialised
/// once, on the first request from an accessor that hands out expressions.
#[derive(Clone, Debug)]
struct ExactTerms {
    mp: MultiPoly<Lex>,
    ex: OnceLock<BTreeMap<Vec<u32>, Ex>>,
}

/// Term storage: exact when every coefficient is a rational literal,
/// otherwise one `Ex` per monomial (ascending lex order in the map; every
/// accessor reports them descending).
#[derive(Clone, Debug)]
enum Terms {
    Exact(ExactTerms),
    Symbolic(BTreeMap<Vec<u32>, Ex>),
}

/// A coefficient seen through either representation, for printing.
enum CoeffView<'a> {
    Rational(&'a Ratio<BigInt>),
    Symbolic(&'a Ex),
}

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
    /// Non-zero terms keyed by exponent vector.
    terms: Terms,
}

// ── Construction ───────────────────────────────────────────────────────────

impl Poly {
    /// Internal constructor from an exact polynomial.
    fn from_exact(ctx: &Context, gens: Vec<Ex>, mp: MultiPoly<Lex>) -> Poly {
        Poly {
            ctx: ctx.clone(),
            gens,
            terms: Terms::Exact(ExactTerms {
                mp,
                ex: OnceLock::new(),
            }),
        }
    }

    /// Internal constructor from already-normalised arena terms: exact when
    /// every coefficient is a rational literal, symbolic otherwise.
    fn from_normalized(ctx: &Context, gens: Vec<Ex>, ids: Vec<(Vec<u32>, ExprId)>) -> Poly {
        let rational: Option<Vec<(Vec<u32>, Ratio<BigInt>)>> = {
            let inner = ctx.inner.read();
            ids.iter()
                .map(|(e, c)| inner.arena.as_num(*c).map(|r| (e.clone(), r.clone())))
                .collect()
        };
        match rational.and_then(|terms| MultiPoly::from_distinct_terms(gens.len(), terms)) {
            Some(mp) => Self::from_exact(ctx, gens, mp),
            None => {
                let terms = ids.into_iter().map(|(e, c)| (e, wrap(ctx, c))).collect();
                Poly {
                    ctx: ctx.clone(),
                    gens,
                    terms: Terms::Symbolic(terms),
                }
            }
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

    /// The exact representation, if this polynomial has one.
    fn exact(&self) -> Option<&MultiPoly<Lex>> {
        match &self.terms {
            Terms::Exact(t) => Some(&t.mp),
            Terms::Symbolic(_) => None,
        }
    }

    /// The exact representation, converting a symbolic one whose
    /// coefficients all happen to be rational; `None` otherwise.
    fn lex_multipoly(&self) -> Option<Cow<'_, MultiPoly<Lex>>> {
        match &self.terms {
            Terms::Exact(t) => Some(Cow::Borrowed(&t.mp)),
            Terms::Symbolic(m) => rational_multipoly(self.gens.len(), m).map(Cow::Owned),
        }
    }

    /// Terms as `Ex` coefficients keyed by exponent vector (ascending lex),
    /// materialised once for the exact representation.
    fn ex_terms(&self) -> &BTreeMap<Vec<u32>, Ex> {
        match &self.terms {
            Terms::Symbolic(m) => m,
            Terms::Exact(t) => t.ex.get_or_init(|| {
                let ids: Vec<(Vec<u32>, ExprId)> = self.ctx.with_arena_mut(|arena| {
                    t.mp.terms()
                        .map(|(e, c)| (e.to_vec(), intern_ratio(arena, c)))
                        .collect()
                });
                ids.into_iter()
                    .map(|(e, c)| (e, wrap(&self.ctx, c)))
                    .collect()
            }),
        }
    }

    /// Exponent vectors in ascending lex order.
    fn exponents(&self) -> Box<dyn DoubleEndedIterator<Item = &[u32]> + '_> {
        match &self.terms {
            Terms::Exact(t) => Box::new(t.mp.terms().map(|(e, _)| e)),
            Terms::Symbolic(m) => Box::new(m.keys().map(Vec::as_slice)),
        }
    }

    fn raw_terms(&self) -> Vec<(Vec<u32>, ExprId)> {
        self.ex_terms()
            .iter()
            .map(|(e, c)| (e.clone(), c.raw_id()))
            .collect()
    }

    /// The arena's evaluation guards.
    fn eval_config(&self) -> EvalConfig {
        self.ctx.inner.read().arena.config.clone()
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
    /// appear anywhere — they become part of the coefficients.  When the
    /// `None` is a surprise, [`try_new`](Self::try_new) returns the same
    /// result with the reason (`… occurs under a negative power (a rational
    /// function) in …`), which is worth a debugging round.
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
        Self::try_new(expr, gens).ok()
    }

    /// [`new`](Self::new) with the reason for failure.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` naming the problem: empty, duplicated or
    /// non-symbol generators, or the first generator found in a
    /// non-polynomial position — inside a function, under a negative power
    /// (a rational function), under a fractional or symbolic power, or in an
    /// exponent.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let (j, r) = (ctx.symbol("j"), ctx.symbol("r"));
    /// assert!(Poly::try_new(&(&j * &r + 1), &[&j]).is_ok());
    /// let e = Poly::try_new(&(&r / (&j + 1)), &[&j]).unwrap_err().to_string();
    /// assert!(e.contains("negative power (a rational function)"), "{e}");
    /// let e = Poly::try_new(&j.sin(), &[&j]).unwrap_err().to_string();
    /// assert!(e.contains("inside the function `sin(j)`"), "{e}");
    /// let e = Poly::try_new(&j, &[]).unwrap_err().to_string();
    /// assert!(e.contains("at least one generator"), "{e}");
    /// ```
    pub fn try_new(expr: &Ex, gens: &[&Ex]) -> Result<Poly, SymplexError> {
        const OP: &str = "Poly::new";
        if gens.is_empty() {
            return Err(invalid(OP, "at least one generator is required"));
        }
        let ctx = expr.context();
        let gens = validate_gens(&ctx, gens, OP)?;
        let gen_ids: Vec<ExprId> = gens.iter().map(Ex::raw_id).collect();
        // Fast path: a rational-coefficient polynomial read off the tree
        // directly, without expanding in the arena.
        let exact = {
            let inner = ctx.inner.read();
            let arena = &inner.arena;
            match flat_terms(arena, expr.raw_id(), &gen_ids) {
                Some(terms) => MultiPoly::from_distinct_terms(gen_ids.len(), terms),
                None => exact_multipoly(arena, expr.raw_id(), &gen_ids),
            }
        };
        if let Some(mp) = exact {
            return Ok(Self::from_exact(&ctx, gens, mp));
        }
        let ids = ctx.with_arena_mut(|arena| {
            let raw = polybridge::symbolic_multipoly_terms(arena, expr.raw_id(), &gen_ids)?;
            Some(normalize_terms(arena, raw))
        });
        match ids {
            Some(ids) => Ok(Self::from_normalized(&ctx, gens, ids)),
            None => {
                let reason = ctx.with_arena_mut(|arena| {
                    let expanded = crate::transforms::expand::expand(arena, expr.raw_id());
                    polybridge::non_polynomial_reason(arena, expanded, &gen_ids).or_else(|| {
                        polybridge::non_polynomial_reason(arena, expr.raw_id(), &gen_ids)
                    })
                });
                Err(invalid(
                    OP,
                    reason.unwrap_or_else(|| {
                        format!("`{expr}` is not a polynomial in the generators")
                    }),
                ))
            }
        }
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
        let wrong_length = |e: &[u32]| {
            invalid(
                OP,
                format!(
                    "exponent vector has length {} but there are {} generators",
                    e.len(),
                    gen_ids.len()
                ),
            )
        };
        // Fast path: all coefficients rational literals (which cannot
        // mention a generator).
        let mut rational: Vec<(Vec<u32>, Ratio<BigInt>)> = Vec::with_capacity(terms.len());
        for (e, c) in &terms {
            if e.len() != gen_ids.len() {
                return Err(wrong_length(e));
            }
            probe.checked_id(c);
            match c.as_rational() {
                Some(r) => rational.push((e.clone(), r)),
                None => {
                    rational.clear();
                    break;
                }
            }
        }
        if rational.len() == terms.len()
            && let Some(mp) = MultiPoly::from_distinct_terms(gen_ids.len(), rational)
        {
            return Ok(Self::from_exact(ctx, gens, mp));
        }
        let mut raw: Vec<(Vec<u32>, ExprId)> = Vec::with_capacity(terms.len());
        for (e, c) in terms {
            if e.len() != gen_ids.len() {
                return Err(wrong_length(&e));
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
        let n = gens.len();
        Ok(Self::from_exact(ctx, gens, MultiPoly::zero(n)))
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
        Ok(Self::from_exact(ctx, gens, mp.convert_order()))
    }
}

impl<O: crate::poly::multipoly::MonomialOrd> MultiPoly<O> {
    /// This exact polynomial as an expression over the symbols `gens`
    /// (variable `i` ↦ `gens[i]`): the bridge from the arena-free
    /// [`MultiPoly`] arithmetic to `Ex` for rendering (`to_lean`), for
    /// the certificate provers, or for anything else symbolic.
    ///
    /// # Errors
    ///
    /// As [`Poly::from_multipoly`]: `gens` must have one distinct symbol
    /// per variable.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::multipoly::MultiPoly;
    /// use symplex::linprog::qi;
    ///
    /// let ctx = Context::new();
    /// let (j, r) = (ctx.symbol("j"), ctx.symbol("r"));
    /// let mp: MultiPoly = MultiPoly::var(2, 0).mul(&MultiPoly::var(2, 1)).scale(&qi(2)) + 1;
    /// assert_eq!(mp.to_ex(&ctx, &[&j, &r])?.to_lean()?, "2 * j * r + 1");
    /// # Ok::<(), SymplexError>(())
    /// ```
    pub fn to_ex(&self, ctx: &Context, gens: &[&Ex]) -> Result<Ex, SymplexError> {
        Ok(Poly::from_multipoly(ctx, gens, &self.convert_order())?.to_ex())
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
        self.num_terms() == 0
    }

    /// Is this a constant (every generator has exponent zero)?  The zero
    /// polynomial is ground.
    #[must_use]
    pub fn is_ground(&self) -> bool {
        self.exponents().all(|e| e.iter().all(|&x| x == 0))
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
        self.exponents().all(|e| e.iter().sum::<u32>() <= 1)
    }

    /// Do all terms have the same total degree?  The zero polynomial is
    /// homogeneous.
    #[must_use]
    pub fn is_homogeneous(&self) -> bool {
        let mut degs = self.exponents().map(|e| e.iter().sum::<u32>());
        match degs.next() {
            None => true,
            Some(d) => degs.all(|x| x == d),
        }
    }

    /// Is every coefficient an exact rational number?
    #[must_use]
    pub fn has_rational_coeffs(&self) -> bool {
        match &self.terms {
            Terms::Exact(_) => true,
            Terms::Symbolic(m) => m.values().all(|c| c.as_rational().is_some()),
        }
    }

    /// Number of non-zero terms.
    #[must_use]
    pub fn num_terms(&self) -> usize {
        match &self.terms {
            Terms::Exact(t) => t.mp.num_terms(),
            Terms::Symbolic(m) => m.len(),
        }
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
        self.ex_terms()
            .iter()
            .rev()
            .map(|(e, c)| (e.clone(), c.clone()))
            .collect()
    }

    /// Borrowing iterator over `(exponent vector, coefficient)` pairs in
    /// descending lexicographic order — [`terms`](Self::terms) without the
    /// allocation, for code that evaluates many polynomials in a loop.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let p = Poly::new(&(&x * &y * 3 + &y.powi(2)), &[&x, &y]).unwrap();
    /// let degrees: Vec<u32> = p.terms_iter().map(|(m, _)| m.iter().sum()).collect();
    /// assert_eq!(degrees, vec![2, 2]);
    /// assert_eq!(p.terms_iter().next().unwrap().1, &ctx.int(3));
    /// ```
    pub fn terms_iter(&self) -> impl Iterator<Item = (&[u32], &Ex)> + '_ {
        self.ex_terms().iter().rev().map(|(e, c)| (e.as_slice(), c))
    }

    /// The coefficients as exact rationals, in the order of
    /// [`terms`](Self::terms); `None` if any coefficient is symbolic.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::poly_ex::Poly;
    /// use symplex::linprog::q;
    ///
    /// let ctx = Context::new();
    /// let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    /// let p = Poly::new(&(&x.powi(2) / 2 - &x * 3), &[&x]).unwrap();
    /// assert_eq!(p.coeffs_rational(), Some(vec![q(1, 2), q(-3, 1)]));
    /// assert!(Poly::new(&(&a * &x), &[&x]).unwrap().coeffs_rational().is_none());
    /// ```
    #[must_use]
    pub fn coeffs_rational(&self) -> Option<Vec<Ratio<BigInt>>> {
        match &self.terms {
            Terms::Exact(t) => Some(t.mp.terms().rev().map(|(_, c)| c.clone()).collect()),
            Terms::Symbolic(m) => m.values().rev().map(Ex::as_rational).collect(),
        }
    }

    /// Exponent vectors in descending lexicographic order.
    #[must_use]
    pub fn monoms(&self) -> Vec<Vec<u32>> {
        self.exponents().rev().map(<[u32]>::to_vec).collect()
    }

    /// Coefficients in the order of [`terms`](Self::terms).
    #[must_use]
    pub fn coeffs(&self) -> Vec<Ex> {
        self.ex_terms().values().rev().cloned().collect()
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
        Ok(match &self.terms {
            Terms::Exact(t) => match t.ex.get() {
                Some(m) => m.get(exps).cloned(),
                None => t.mp.coeff(exps).map(|r| self.ctx.from_ratio(r.clone())),
            },
            Terms::Symbolic(m) => m.get(exps).cloned(),
        }
        .unwrap_or_else(|| self.ctx.zero()))
    }

    /// Total degree (largest exponent sum); `None` for the zero polynomial.
    #[must_use]
    pub fn total_degree(&self) -> Option<u32> {
        self.exponents().map(|e| e.iter().sum::<u32>()).max()
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
        self.exponents().map(|e| e[i]).max()
    }

    /// Maximum exponent of each generator (all zeros for the zero
    /// polynomial).
    #[must_use]
    pub fn degree_list(&self) -> Vec<u32> {
        let mut out = vec![0u32; self.gens.len()];
        for e in self.exponents() {
            for (o, &x) in out.iter_mut().zip(e) {
                *o = (*o).max(x);
            }
        }
        out
    }

    /// Leading term under the lexicographic order; `None` for zero.
    #[must_use]
    pub fn leading_term(&self) -> Option<(Vec<u32>, Ex)> {
        match &self.terms {
            Terms::Exact(t) => match t.ex.get() {
                Some(m) => m.last_key_value().map(|(e, c)| (e.clone(), c.clone())),
                None => {
                    t.mp.leading_term()
                        .map(|(e, c)| (e.to_vec(), self.ctx.from_ratio(c.clone())))
                }
            },
            Terms::Symbolic(m) => m.last_key_value().map(|(e, c)| (e.clone(), c.clone())),
        }
    }

    /// Leading coefficient under the lexicographic order; `0` for zero.
    #[must_use]
    pub fn leading_coeff(&self) -> Ex {
        self.leading_term()
            .map_or_else(|| self.ctx.zero(), |(_, c)| c)
    }

    /// Leading monomial under the lexicographic order; `None` for zero.
    #[must_use]
    pub fn leading_monomial(&self) -> Option<Vec<u32>> {
        self.exponents().next_back().map(<[u32]>::to_vec)
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
        let terms = self.ex_terms();
        let deg = terms.keys().map(|e| e[0]).max().unwrap_or(0);
        let zero = self.ctx.zero();
        let mut out = vec![zero; deg as usize + 1];
        for (e, c) in terms {
            out[(deg - e[0]) as usize] = c.clone();
        }
        Some(out)
    }

    /// Structural equality: same generators and identical (normalised)
    /// coefficients for every monomial.
    #[must_use]
    pub fn equals(&self, other: &Poly) -> bool {
        if self.gens != other.gens {
            return false;
        }
        match (&self.terms, &other.terms) {
            (Terms::Exact(a), Terms::Exact(b)) => a.mp == b.mp,
            (Terms::Symbolic(a), Terms::Symbolic(b)) => a == b,
            (Terms::Exact(a), Terms::Symbolic(b)) | (Terms::Symbolic(b), Terms::Exact(a)) => {
                rational_multipoly(self.gens.len(), b).is_some_and(|mp| mp == a.mp)
            }
        }
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
        // Fast path: exact polynomial at a rational point, provided every
        // power involved is one the arena would fold to a literal too.
        if let Some(mp) = self.exact() {
            let rationals: Option<Vec<Ratio<BigInt>>> = {
                let inner = self.ctx.inner.read();
                value_ids
                    .iter()
                    .map(|&v| inner.arena.as_num(v).cloned())
                    .collect()
            };
            if let Some(vals) = rationals {
                let config = self.eval_config();
                let within = degree_list_of(mp)
                    .iter()
                    .zip(&vals)
                    .all(|(&d, v)| pow_within_limits(&config, v, d));
                if within {
                    return Ok(self.ctx.from_ratio(eval_exact(mp, &vals)));
                }
            }
        }
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
        // Fast path: exact polynomial at a rational value of the generator.
        if let Some(mp) = self.exact()
            && let Some(v) = value.as_rational()
            && pow_within_limits(&self.eval_config(), &v, mp.degree_in(i))
        {
            return Ok(Self::from_exact(&self.ctx, remaining, mp.substitute(i, &v)));
        }
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
        self.lex_multipoly().map(|mp| mp.convert_order())
    }

    /// Numeric complex roots of a univariate polynomial with rational
    /// coefficients, as [`Complex64`] values (see [`Ex::nroots`]).
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
    /// assert!((roots[1].re - 2f64.sqrt()).abs() < 1e-12);
    /// assert_eq!(roots[1].im, 0.0);
    /// ```
    pub fn nroots(&self, digits: u32) -> Result<Vec<Complex64>, SymplexError> {
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

    /// The single generator of a univariate polynomial with rational
    /// coefficients, or `None` (the precondition of the real-root and sign
    /// queries below).
    fn univariate_rational_gen(&self) -> Option<&Ex> {
        if self.gens.len() == 1 && self.has_rational_coeffs() {
            self.gens.first()
        } else {
            None
        }
    }

    /// Number of distinct real roots (Sturm's theorem); `None` unless the
    /// polynomial is univariate with rational coefficients.  See
    /// [`Ex::count_real_roots`].
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// assert_eq!((&x.powi(3) - &x).as_poly(&[&x]).unwrap().count_real_roots(), Some(3));
    /// assert_eq!((&x.powi(2) + 1).as_poly(&[&x]).unwrap().count_real_roots(), Some(0));
    /// ```
    #[must_use]
    pub fn count_real_roots(&self) -> Option<usize> {
        let x = self.univariate_rational_gen()?;
        self.to_ex().count_real_roots(x)
    }

    /// Number of distinct real roots in the closed interval `[lo, hi]`
    /// (SymPy's `Poly.count_roots(inf, sup)`); endpoints are exact rationals
    /// or `±∞`.  `None` unless the polynomial is univariate with rational
    /// coefficients.  See [`Ex::count_real_roots_in`].
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let p = (&x.powi(3) - &x).as_poly(&[&x]).unwrap();   // roots −1, 0, 1
    /// assert_eq!(p.count_real_roots_in(&ctx.int(0), &ctx.infinity()), Some(2));
    /// assert_eq!(p.count_real_roots_in(&ctx.rational(1, 2), &ctx.infinity()), Some(1));
    /// assert_eq!(p.count_real_roots_in(&ctx.int(2), &ctx.int(9)), Some(0));
    /// ```
    #[must_use]
    pub fn count_real_roots_in(&self, lo: &Ex, hi: &Ex) -> Option<usize> {
        let x = self.univariate_rational_gen()?;
        self.to_ex().count_real_roots_in(x, lo, hi)
    }

    /// Isolating intervals with exact rational endpoints for the distinct
    /// real roots, sorted — each a half-open `(lo, hi]` Sturm cell or the
    /// closed point `[r, r]` of a root hit exactly; empty unless univariate
    /// with rational coefficients.  See [`Ex::real_roots_isolate`].
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let p = (&x.powi(2) - 2).as_poly(&[&x]).unwrap();
    /// let iv = p.real_roots_isolate();
    /// assert_eq!(iv.len(), 2);
    /// assert_eq!(iv[0].kind, IntervalKind::LeftOpen);
    /// assert!(iv[0].upper.eval_f64().unwrap() <= 0.0 && 0.0 <= iv[1].lower.eval_f64().unwrap());
    /// ```
    #[must_use]
    pub fn real_roots_isolate(&self) -> Vec<Interval<Ex>> {
        match self.univariate_rational_gen() {
            Some(x) => self.to_ex().real_roots_isolate(x),
            None => Vec::new(),
        }
    }

    /// Exact decision `p(x) ≥ 0` for every `x ∈ [lo, hi]` (endpoints
    /// rational or `±∞`); `None` unless univariate with rational
    /// coefficients.  See [`Ex::poly_is_nonnegative_on`].
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let j = ctx.symbol("j");
    /// let p = (&j.powi(3) - 6 * &j.powi(2) + 11 * &j - 6).as_poly(&[&j]).unwrap(); // (j-1)(j-2)(j-3)
    /// assert_eq!(p.is_nonnegative_on(&ctx.int(3), &ctx.infinity()), Some(true));
    /// assert_eq!(p.is_nonnegative_on(&ctx.int(2), &ctx.infinity()), Some(false));
    /// assert_eq!(p.is_positive_on(&ctx.int(3), &ctx.infinity()), Some(false)); // zero at 3
    /// ```
    #[must_use]
    pub fn is_nonnegative_on(&self, lo: &Ex, hi: &Ex) -> Option<bool> {
        let x = self.univariate_rational_gen()?;
        self.to_ex().poly_is_nonnegative_on(x, lo, hi)
    }

    /// Exact decision `p(x) > 0` for every `x ∈ [lo, hi]`; see
    /// [`is_nonnegative_on`](Self::is_nonnegative_on) and
    /// [`Ex::poly_is_positive_on`].
    #[must_use]
    pub fn is_positive_on(&self, lo: &Ex, hi: &Ex) -> Option<bool> {
        let x = self.univariate_rational_gen()?;
        self.to_ex().poly_is_positive_on(x, lo, hi)
    }

    /// Taylor shift of one generator: `p(…, g + a, …)`.
    ///
    /// A shift by the left endpoint turns a half-line question into a sign
    /// question about coefficients: if every coefficient of `p(k + a)` is
    /// non-negative then `p ≥ 0` on `[a, ∞)` (a sufficient, certificate-style
    /// criterion; [`is_nonnegative_on`](Self::is_nonnegative_on) is the exact
    /// one).
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `var` is not a generator or `a` mentions one.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let j = ctx.symbol("j");
    /// let p = (&j.powi(2) - 4 * &j + 3).as_poly(&[&j]).unwrap();      // (j-1)(j-3)
    /// let shifted = p.shift(&j, &ctx.int(3)).unwrap();                 // p(j + 3) = j^2 + 2j
    /// assert_eq!(shifted.to_string(), "Poly(j^2 + 2*j, j)");
    /// assert!(shifted.coeffs().iter().all(|c| c.is_negative() == Some(false)));
    /// ```
    pub fn shift(&self, var: &Ex, a: &Ex) -> Result<Poly, SymplexError> {
        let probe = self.ctx.zero();
        let gen_id = probe.checked_id(var);
        if !self.gens.iter().any(|g| g.raw_id() == gen_id) {
            return Err(invalid("Poly::shift", "not a generator of this polynomial"));
        }
        for g in &self.gens {
            if a.contains(g) {
                return Err(invalid(
                    "Poly::shift",
                    "the shift must not mention a generator",
                ));
            }
        }
        let replacement = var + a;
        let shifted = self.to_ex().subs(var, &replacement);
        let gens: Vec<&Ex> = self.gens.iter().collect();
        Poly::new(&shifted, &gens).ok_or_else(|| SymplexError::ComputationFailed {
            operation: "Poly::shift",
            reason: "shifted expression is not polynomial in the generators".into(),
        })
    }
}

// ── Arithmetic ──────────────────────────────────────────────────────────────────────────

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
        if let (Some(a), Some(b)) = (self.exact(), other.exact()) {
            return Ok(Self::from_exact(&self.ctx, self.gens.clone(), a.add(b)));
        }
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
        if let (Some(a), Some(b)) = (self.exact(), other.exact()) {
            return Ok(Self::from_exact(&self.ctx, self.gens.clone(), a.sub(b)));
        }
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
        if let (Some(a), Some(b)) = (self.exact(), other.exact()) {
            let prod =
                mul_checked(a, b).ok_or_else(|| invalid("Poly::mul", "exponent overflow"))?;
            return Ok(Self::from_exact(&self.ctx, self.gens.clone(), prod));
        }
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
        if let Some(mp) = self.exact() {
            return Self::from_exact(&self.ctx, self.gens.clone(), mp.neg());
        }
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
        if let Some(mp) = self.exact()
            && let Some(r) = c.as_rational()
        {
            return Ok(Self::from_exact(&self.ctx, self.gens.clone(), mp.scale(&r)));
        }
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
        if let Some(mp) = self.exact() {
            let power =
                pow_checked(mp, n).ok_or_else(|| invalid("Poly::mul", "exponent overflow"))?;
            return Ok(Self::from_exact(&self.ctx, self.gens.clone(), power));
        }
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
        if let Some(mp) = self.exact() {
            return Ok(Self::from_exact(
                &self.ctx,
                self.gens.clone(),
                mp.partial_derivative(i),
            ));
        }
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
        let mp = self.lex_multipoly()?;
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
        if mp.leading_coeff().is_some_and(Signed::is_negative) {
            content = -content;
        }
        let inv = Ratio::one() / &content;
        let prim = Self::from_exact(&self.ctx, self.gens.clone(), mp.scale(&inv));
        Some((self.ctx.from_ratio(content), prim))
    }

    /// Divide by the leading coefficient so it becomes `1`.
    ///
    /// `None` for the zero polynomial or non-rational coefficients.
    #[must_use]
    pub fn monic(&self) -> Option<Poly> {
        let mp = self.lex_multipoly()?;
        if mp.is_zero() {
            return None;
        }
        Some(Self::from_exact(&self.ctx, self.gens.clone(), mp.monic()))
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
            set.extend(p.exponents().map(<[u32]>::to_vec));
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

/// Sign and printed magnitude of a rational coefficient.
fn rational_sign_magnitude(r: &Ratio<BigInt>) -> (bool, String) {
    let abs = r.abs();
    let s = if abs.is_integer() {
        abs.numer().to_string()
    } else {
        format!("{}/{}", abs.numer(), abs.denom())
    };
    (r.is_negative(), s)
}

/// Sign and printed magnitude of a coefficient; a symbolic sum in front of
/// a monomial is parenthesised.
fn coeff_sign_magnitude(coeff: &CoeffView<'_>, has_monomial: bool) -> (bool, String) {
    match coeff {
        CoeffView::Rational(r) => rational_sign_magnitude(r),
        CoeffView::Symbolic(coeff) => match coeff.as_rational() {
            Some(r) => rational_sign_magnitude(&r),
            None => {
                let s = coeff.to_string();
                if coeff.expr_type() == ExprType::Add && has_monomial {
                    (false, format!("({s})"))
                } else if let Some(rest) = s.strip_prefix('-') {
                    (true, rest.to_string())
                } else {
                    (false, s)
                }
            }
        },
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
        if self.is_zero() {
            write!(f, "0")?;
        }
        let terms: Box<dyn Iterator<Item = (&[u32], CoeffView<'_>)> + '_> = match &self.terms {
            Terms::Exact(t) => {
                Box::new(t.mp.terms().rev().map(|(e, c)| (e, CoeffView::Rational(c))))
            }
            Terms::Symbolic(m) => Box::new(
                m.iter()
                    .rev()
                    .map(|(e, c)| (e.as_slice(), CoeffView::Symbolic(c))),
            ),
        };
        for (i, (exps, coeff)) in terms.enumerate() {
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

            let (negative, magnitude) = coeff_sign_magnitude(&coeff, !monomial.is_empty());

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
