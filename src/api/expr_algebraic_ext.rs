//! Algebraic-number and polynomial-algebra conveniences on [`Ex`]:
//! minimal polynomials, multivariate gcd/lcm, Gröbner bases, real roots
//! as `RootOf`, factoring modulo a prime, symbolic resultants.  (0.9)
//!
//! Everything here is a thin bridge from `Ex` to the crate's exact
//! polynomial engines (`poly::algebraic`, `poly::multipoly`,
//! `poly::groebner`, `poly::factor_zassenhaus`, `poly::sturm`) and back.
//! Methods that return `Option` yield `None` when the input is not of the
//! required shape (Pattern 4: a query that cannot be answered); methods
//! that validate caller-supplied structure (variable lists, moduli) return
//! `Result` (Pattern 5).

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, Zero};

use crate::api::expr::{Ex, Expr, ExprType, Numeric};
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::poly::Poly;
use crate::poly::groebner::{groebner_basis, groebner_basis_lex};
use crate::poly::multipoly::{GrevLex, Lex, MultiPoly};
use crate::poly::polybridge::{expr_to_multipoly, expr_to_poly, multipoly_to_expr, poly_to_expr};
use crate::poly::sturm::SturmChain;

pub use crate::poly::multipoly::MonomialOrder;

// ═══════════════════════════════════════════════════════════════════════════
// Arena-level helpers
// ═══════════════════════════════════════════════════════════════════════════

fn invalid(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(operation, reason)
}

/// The polynomial with integer coefficients, no common integer factor and
/// positive leading coefficient that is a rational multiple of `p`
/// (SymPy's normalisation of `minimal_polynomial`).
fn integer_primitive(p: &Poly) -> Poly {
    let prim = p.primitive_part();
    match prim.leading_coeff() {
        Some(lc) if lc.is_negative() => -&prim,
        _ => prim,
    }
}

/// Union of the free symbols of `exprs`, in the arena's canonical order.
fn shared_generators(arena: &Arena, exprs: &[ExprId]) -> Vec<ExprId> {
    let mut gens: Vec<ExprId> = Vec::new();
    for &e in exprs {
        for s in crate::base::walk::free_symbols(arena, e) {
            if !gens.contains(&s) {
                gens.push(s);
            }
        }
    }
    gens.sort_by(|a, b| {
        arena
            .sort_key(*a)
            .cmp(arena.sort_key(*b))
            .then_with(|| a.0.cmp(&b.0))
    });
    gens
}

/// Validate a variable list for the Gröbner API: non-empty, distinct
/// symbols of the same context as `probe`.
fn validate_vars(
    probe: &Ex,
    vars: &[Ex],
    operation: &'static str,
) -> Result<Vec<ExprId>, SymplexError> {
    if vars.is_empty() {
        return Err(invalid(operation, "at least one variable is required"));
    }
    let mut ids: Vec<ExprId> = Vec::with_capacity(vars.len());
    for v in vars {
        let id = probe.checked_id(v);
        if v.expr_type() != ExprType::Symbol {
            return Err(invalid(
                operation,
                format!("variable `{v}` is not a symbol"),
            ));
        }
        if ids.contains(&id) {
            return Err(invalid(operation, format!("duplicate variable `{v}`")));
        }
        ids.push(id);
    }
    Ok(ids)
}

/// Convert every expression to a `MultiPoly` over `vars`, naming the first
/// one that is not a polynomial with rational coefficients.
fn to_multipolys(
    arena: &Arena,
    exprs: &[ExprId],
    vars: &[ExprId],
    operation: &'static str,
) -> Result<Vec<MultiPoly<GrevLex>>, SymplexError> {
    exprs
        .iter()
        .map(|&e| {
            expr_to_multipoly(arena, e, vars).ok_or_else(|| {
                invalid(
                    operation,
                    format!(
                        "`{}` is not a polynomial in the given variables with rational coefficients",
                        arena.display(e)
                    ),
                )
            })
        })
        .collect()
}

/// Reduced Gröbner basis of `polys` under `order`, returned in grevlex
/// storage (the order only affects which basis is computed).
fn groebner_in_order(
    polys: &[MultiPoly<GrevLex>],
    order: MonomialOrder,
) -> Vec<MultiPoly<GrevLex>> {
    match order {
        MonomialOrder::GrevLex => groebner_basis(polys),
        MonomialOrder::Lex => groebner_basis_lex(polys)
            .iter()
            .map(MultiPoly::convert_order)
            .collect(),
    }
}

/// Remainder of `f` on division by `basis` under `order`.
fn reduce_in_order(
    f: &MultiPoly<GrevLex>,
    basis: &[MultiPoly<GrevLex>],
    order: MonomialOrder,
) -> MultiPoly<GrevLex> {
    match order {
        MonomialOrder::GrevLex => {
            let refs: Vec<&MultiPoly<GrevLex>> = basis.iter().collect();
            f.reduce(&refs)
        }
        MonomialOrder::Lex => {
            let lex_basis: Vec<MultiPoly<Lex>> =
                basis.iter().map(MultiPoly::convert_order).collect();
            let refs: Vec<&MultiPoly<Lex>> = lex_basis.iter().collect();
            f.convert_order::<Lex>().reduce(&refs).convert_order()
        }
    }
}

/// One distinct real root of a polynomial, before it is interned.
enum RealRoot {
    /// A rational root (of a linear factor).
    Rational(Q),
    /// `RootOf(g, k)`: the `k`-th root of the factor `g` in the (re, im)
    /// order the `RootOf` evaluator uses.
    RootOf { factor: usize, index: usize },
}

/// One distinct real root of a polynomial, located but not yet named:
/// the factor it belongs to and its Sturm isolating interval.
struct RootCell {
    /// Index into [`RealRootCells::factors`].
    factor: usize,
    /// Either the point `[r, r]` of an exact hit or a half-open `(lo, hi]`
    /// Sturm cell containing exactly one root of the factor.
    interval: crate::base::interval::Interval<Q>,
}

/// The square-free factors of a polynomial over ℤ and the cells of its
/// distinct real roots, ascending.  Naming a cell (the Aberth run behind a
/// `RootOf` index) is deferred to [`RealRootCells::name`], so `root_of`
/// pays for one root only.
struct RealRootCells {
    /// Pairwise coprime square-free factors over ℤ (irreducible when the
    /// factorisation was certified complete).
    factors: Vec<Poly>,
    cells: Vec<RootCell>,
    /// `rootof_roots(factors[i])`, computed on first use.
    roots_cache: Vec<Option<Vec<crate::poly::roots::Complex>>>,
}

impl RealRootCells {
    /// Locate the distinct real roots of `f` (a non-constant polynomial
    /// with rational coefficients).
    ///
    /// `f` is factored over ℤ (Berlekamp–Zassenhaus).  When the
    /// factorisation is certified complete every factor is irreducible and
    /// each `RootOf(g, k)` is canonical.  When it is not (the recombination
    /// budget of `factor_zassenhaus` ran out on a factor with very many
    /// modular factors) the uncertified factors are still square-free and
    /// pairwise coprime — the square-free decomposition runs first — so
    /// `RootOf(g, k)` on such a `g` is well defined and evaluates
    /// correctly (the evaluator only needs the `k`-th root of `g` in
    /// (re, im) order); it merely may not be in lowest terms, so two
    /// `RootOf`s naming the same algebraic number could compare unequal.
    /// That is logged and accepted rather than refusing the root.
    ///
    /// Returns `None` when a root cannot be assigned to exactly one
    /// factor.  This is an internal limit, not a statement about `f`; the
    /// public wrapper folds it into its `None` alongside "not a polynomial"
    /// because its return type cannot distinguish the two in a patch
    /// release.
    fn locate(f: &Poly) -> Option<Self> {
        let (_content, parts, complete) =
            crate::poly::factor_zassenhaus::factor_zassenhaus_checked(f);
        if !complete {
            tracing::debug!(
                degree = f.degree().unwrap_or(0),
                "real_roots: factorisation over ℤ not certified complete; naming roots of square-free factors"
            );
        }
        let mut factors: Vec<Poly> = Vec::with_capacity(parts.len());
        let mut chains: Vec<SturmChain> = Vec::with_capacity(parts.len());
        let mut square_free = Poly::from_int(1);
        for (g, _mult) in &parts {
            if g.degree()? == 0 {
                continue;
            }
            square_free = &square_free * g;
            chains.push(SturmChain::new(g));
            factors.push(g.clone());
        }
        if square_free.degree().unwrap_or(0) == 0 {
            return None;
        }

        // Isolating intervals of the product of the distinct factors are
        // pairwise disjoint and sorted, so walking them in order visits
        // every real root ascending; each one belongs to exactly one
        // factor.  Each interval is either the point `[r, r]` of an exact
        // hit or a half-open `(lo, hi]` Sturm cell, so `count_roots_in`
        // (also `(a, b]`) is the right membership test for the latter.
        let intervals = SturmChain::new(&square_free).isolate_all_real_roots();
        let mut cells: Vec<RootCell> = Vec::with_capacity(intervals.len());
        for iv in intervals {
            let owner = factors.iter().zip(&chains).position(|(g, chain)| {
                if iv.is_point() {
                    g.eval(&iv.lower).is_zero()
                } else {
                    chain.count_roots_in(&iv.lower, &iv.upper) == 1
                }
            })?;
            cells.push(RootCell {
                factor: owner,
                interval: iv,
            });
        }
        let roots_cache = vec![None; factors.len()];
        Some(Self {
            factors,
            cells,
            roots_cache,
        })
    }

    /// Name the `i`-th real root: exact for a linear factor, otherwise
    /// `RootOf(g, k)` with `k` from
    /// [`real_root_index`](crate::poly::roots::real_root_index), which
    /// verifies that the `k`-th computed root of `g` lies in the root's
    /// Sturm isolating interval and that the index is stable.  The
    /// computed roots of `g` are cached, so naming every root of `g` costs
    /// one Aberth run.
    ///
    /// `None` when `i` is out of range or the index cannot be verified.
    fn name(&mut self, i: usize) -> Option<RealRoot> {
        let cell = self.cells.get(i)?;
        let g = &self.factors[cell.factor];
        if g.degree() == Some(1) {
            return Some(RealRoot::Rational(-(g.coeff(0) / g.coeff(1))));
        }
        let roots = self.roots_cache[cell.factor].get_or_insert_with(|| {
            crate::poly::roots::rootof_roots(g, crate::poly::roots::ROOTOF_DEFAULT_PREC)
        });
        // `real_root_index` treats the pair as closed `[lo, hi]`, a
        // superset of `(lo, hi]`; the extra endpoint is not a root of `g`
        // (an exact hit is reported as a point instead).
        let index =
            crate::poly::roots::real_root_index(roots, &cell.interval.lower, &cell.interval.upper)?;
        Some(RealRoot::RootOf {
            factor: cell.factor,
            index,
        })
    }
}

/// Determinant of the Sylvester matrix of two coefficient lists (highest
/// degree first, both non-empty with non-zero leading coefficient), i.e.
/// `res(f, g)`.  `f` of degree `m`, `g` of degree `n`: the matrix is
/// `(m + n) × (m + n)` with `n` shifted copies of `f` above `m` shifted
/// copies of `g`.
fn sylvester_resultant(ctx: &crate::api::context::Context, f: &[Ex], g: &[Ex]) -> Option<Ex> {
    let m = f.len().checked_sub(1)?;
    let n = g.len().checked_sub(1)?;
    if m == 0 && n == 0 {
        return Some(ctx.one());
    }
    if m == 0 {
        return Some(f[0].powi(n as i64));
    }
    if n == 0 {
        return Some(g[0].powi(m as i64));
    }
    let size = m + n;
    let zero = ctx.zero();
    let mut rows: Vec<Vec<Ex>> = Vec::with_capacity(size);
    for i in 0..n {
        let mut row = vec![zero.clone(); size];
        for (j, c) in f.iter().enumerate() {
            row[i + j] = c.clone();
        }
        rows.push(row);
    }
    for i in 0..m {
        let mut row = vec![zero.clone(); size];
        for (j, c) in g.iter().enumerate() {
            row[i + j] = c.clone();
        }
        rows.push(row);
    }
    // Invariant: `rows` is a non-empty square `size × size` matrix, so
    // `Matrix::new` and `det` cannot fail.  Their `Err` is an internal
    // error, not "not a polynomial"; the `Option` return of the public
    // callers cannot express that in a patch release, so it is folded into
    // `None` here (and asserted in debug builds).
    let matrix = crate::domains::matrix::Matrix::new(rows);
    debug_assert!(matrix.is_ok(), "Sylvester matrix is square by construction");
    let det = matrix.ok()?.det();
    debug_assert!(det.is_ok(), "determinant of a square matrix over Ex");
    Some(det.ok()?.expand())
}

// ═══════════════════════════════════════════════════════════════════════════
// impl Expr<Numeric> — algebraic numbers and polynomial algebra
// ═══════════════════════════════════════════════════════════════════════════

impl Expr<Numeric> {
    // ── Minimal polynomial ─────────────────────────────────────────

    /// Minimal polynomial over ℚ of an algebraic-number expression, as a
    /// polynomial in `var` (SymPy `minimal_polynomial`).
    ///
    /// `self` must be a constant built from rational numbers, radicals
    /// (`n^{p/q}`), `i`, `φ`, sums, products, negations, and integer or
    /// rational powers of such numbers (`(1 + √2)⁻¹`, `√(3 + 2√2)`; a
    /// fractional power needs a positive real base).  The result has
    /// integer coefficients with no common factor and a positive leading
    /// coefficient, so `3/4` gives `4·var − 3` and `√2 + √3` gives
    /// `var⁴ − 10·var² + 1`.  Returns `None` for input that is not
    /// recognised as algebraic — `π`, `e`, free symbols, transcendental
    /// functions.
    ///
    /// The result is verified, never guessed: the irreducible factor of
    /// each intermediate resultant is chosen by evaluating the number to
    /// 320 bits and accepting the unique factor whose residual is
    /// negligible, and the factorisation itself must be certified complete.
    /// If either check fails the method returns `None` rather than a
    /// polynomial that might be reducible or vanish at a different
    /// conjugate.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let a = ctx.int(2).sqrt() + ctx.int(3).sqrt();
    /// let m = a.minimal_polynomial(&x).unwrap();
    /// assert_eq!(m, &x.powi(4) - &x.powi(2) * 10 + 1);
    ///
    /// let cbrt2 = ctx.int(2).pow(&ctx.rational(1, 3));
    /// assert_eq!(cbrt2.minimal_polynomial(&x).unwrap(), &x.powi(3) - 2);
    ///
    /// assert!(ctx.pi().minimal_polynomial(&x).is_none());
    /// ```
    #[must_use]
    pub fn minimal_polynomial(&self, var: &Ex) -> Option<Ex> {
        use crate::poly::algebraic::{AlgExpr, minimal_polynomial_of};
        let var_id = self.checked_id(var);
        // Detach the expression under a read lock; the computation itself
        // (resultants, factoring, 320-bit evaluation) runs unlocked.
        let alg = {
            let inner = self.inner.read();
            AlgExpr::from_arena(&inner.arena, self.raw_id())
        };
        let mp = alg.and_then(|a| minimal_polynomial_of(&a)).or_else(|| {
            // A product or power the structural recursion does not
            // recognise may become a plain sum once expanded; try that
            // form once before giving up.
            let expanded = self.expand();
            if expanded == *self {
                return None;
            }
            let alg = {
                let inner = self.inner.read();
                AlgExpr::from_arena(&inner.arena, expanded.raw_id())
            }?;
            minimal_polynomial_of(&alg)
        })?;
        let id = {
            let mut inner = self.inner.write();
            poly_to_expr(&mut inner.arena, &integer_primitive(&mp), var_id)
        };
        Some(self.wrap(id))
    }

    // ── Multivariate gcd / lcm ─────────────────────────────────────

    /// Polynomial greatest common divisor of `self` and `other` over ℚ in
    /// all of their free symbols at once (SymPy `gcd(f, g)`).
    ///
    /// Both inputs must be polynomials with rational coefficients; anything
    /// else (`sin(x)`, `1/x`, `π`) gives `None`.  The result is normalised
    /// like [`MultiPoly::gcd`]: integer coefficients, positive leading
    /// coefficient in graded reverse lexicographic order, and integer
    /// content equal to the gcd of the inputs' integer contents once their
    /// denominators are cleared — for polynomials over ℤ this is exactly
    /// the gcd over ℤ (`gcd(2x, 4x) = 2x`).  `gcd(0, 0) = 0`; two rational
    /// constants give their integer gcd.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let f = &x.powi(2) - &y.powi(2);
    /// let g = &x - &y;
    /// assert_eq!(f.gcd_all(&g).unwrap(), g);
    /// assert_eq!((&x * 2).gcd_all(&(&x * 4)).unwrap(), &x * 2);
    /// assert!(x.sin().gcd_all(&x).is_none());
    /// ```
    #[must_use]
    pub fn gcd_all(&self, other: &Ex) -> Option<Ex> {
        self.multipoly_binary(other, MultiPoly::gcd)
    }

    /// Polynomial least common multiple of `self` and `other` over ℚ in
    /// all of their free symbols (SymPy `lcm(f, g)`).
    ///
    /// Same preconditions and normalisation as [`gcd_all`](Self::gcd_all):
    /// `lcm = a·b / gcd(a, b)` computed on the integer-normalised inputs,
    /// so `lcm(2x, 4x) = 4x`; zero if either input is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let f = &x.powi(2) - &y.powi(2);
    /// let g = &x - &y;
    /// assert_eq!(f.lcm_all(&g).unwrap(), f);
    /// assert_eq!((&x * &y).lcm_all(&y.powi(2)).unwrap(), &x * &y.powi(2));
    /// ```
    #[must_use]
    pub fn lcm_all(&self, other: &Ex) -> Option<Ex> {
        self.multipoly_binary(other, MultiPoly::lcm)
    }

    /// Shared driver of `gcd_all` / `lcm_all`.
    fn multipoly_binary(
        &self,
        other: &Ex,
        op: fn(&MultiPoly<GrevLex>, &MultiPoly<GrevLex>) -> MultiPoly<GrevLex>,
    ) -> Option<Ex> {
        let other_id = self.checked_id(other);
        // Convert under a read lock, compute unlocked, intern under a write
        // lock.
        let (gens, a, b) = {
            let inner = self.inner.read();
            let arena = &inner.arena;
            let gens = shared_generators(arena, &[self.raw_id(), other_id]);
            let a = expr_to_multipoly(arena, self.raw_id(), &gens)?;
            let b = expr_to_multipoly(arena, other_id, &gens)?;
            (gens, a, b)
        };
        let result = op(&a, &b);
        let id = {
            let mut inner = self.inner.write();
            multipoly_to_expr(&mut inner.arena, &result, &gens)
        };
        Some(self.wrap(id))
    }

    // ── Gröbner bases ──────────────────────────────────────────────

    /// Reduced Gröbner basis of the ideal generated by `polys` in the
    /// variables `vars` under the monomial order `order` (SymPy
    /// `groebner(polys, *vars, order=…)`).
    ///
    /// Every element of the basis is monic; the list is sorted by leading
    /// monomial, largest first.  An empty `polys` (or all zeros) gives an
    /// empty basis; an ideal containing a non-zero constant gives `[1]`.
    /// `MonomialOrder::GrevLex` runs Buchberger directly;
    /// `MonomialOrder::Lex` computes in grevlex first and converts with
    /// FGLM when the ideal is zero-dimensional, which is much faster than
    /// lex Buchberger and yields the same (unique) reduced basis.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `vars` is empty, contains duplicates or
    /// non-symbols, or if some polynomial is not a polynomial in `vars`
    /// with rational coefficients (other symbols count as non-rational
    /// coefficients).
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::multipoly::MonomialOrder;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let f = &x.powi(2) + &y.powi(2) - 1;
    /// let g = &x - &y;
    /// let vars = [x.clone(), y.clone()];
    /// let basis = Ex::groebner(&[f, g.clone()], &vars, MonomialOrder::Lex).unwrap();
    /// assert_eq!(basis, vec![g, &y.powi(2) - ctx.rational(1, 2)]);
    /// ```
    pub fn groebner(
        polys: &[Ex],
        vars: &[Ex],
        order: MonomialOrder,
    ) -> Result<Vec<Ex>, SymplexError> {
        const OP: &str = "groebner";
        let probe = vars
            .first()
            .ok_or_else(|| invalid(OP, "at least one variable is required"))?;
        let var_ids = validate_vars(probe, vars, OP)?;
        let poly_ids: Vec<ExprId> = polys.iter().map(|p| probe.checked_id(p)).collect();
        // Convert under a read lock, compute unlocked, intern under a write
        // lock.
        let mps = {
            let inner = probe.inner.read();
            to_multipolys(&inner.arena, &poly_ids, &var_ids, OP)?
        };
        let basis = groebner_in_order(&mps, order);
        let ids: Vec<ExprId> = {
            let mut inner = probe.inner.write();
            let arena = &mut inner.arena;
            basis
                .iter()
                .map(|g| multipoly_to_expr(arena, g, &var_ids))
                .collect()
        };
        Ok(ids.into_iter().map(|id| probe.wrap(id)).collect())
    }

    /// Remainder of `self` on multivariate division by `basis` in the
    /// variables `vars` under `order` (SymPy `reduced(f, G)[1]` /
    /// `GroebnerBasis.reduce`).
    ///
    /// When `basis` is a Gröbner basis for `order` this is the unique
    /// normal form of `self` modulo the ideal — zero exactly when `self`
    /// lies in the ideal.  For an arbitrary `basis` it is *a* remainder,
    /// which may depend on the order of the divisors.
    ///
    /// # Errors
    ///
    /// As [`groebner`](Self::groebner): invalid `vars`, or `self` or a
    /// basis element that is not a polynomial in `vars` with rational
    /// coefficients.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::multipoly::MonomialOrder;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// let vars = [x.clone(), y.clone()];
    /// let basis = Ex::groebner(&[&x.powi(2) + &y.powi(2) - 1, &x - &y], &vars, MonomialOrder::Lex).unwrap();
    /// // x² ≡ y² ≡ 1/2 modulo the ideal
    /// let r = x.powi(2).reduce_modulo(&basis, &vars, MonomialOrder::Lex).unwrap();
    /// assert_eq!(r, ctx.rational(1, 2));
    /// assert!((&x.powi(2) - &y.powi(2)).reduce_modulo(&basis, &vars, MonomialOrder::Lex).unwrap().is_zero_structural());
    /// ```
    pub fn reduce_modulo(
        &self,
        basis: &[Ex],
        vars: &[Ex],
        order: MonomialOrder,
    ) -> Result<Ex, SymplexError> {
        const OP: &str = "reduce_modulo";
        let var_ids = validate_vars(self, vars, OP)?;
        let basis_ids: Vec<ExprId> = basis.iter().map(|b| self.checked_id(b)).collect();
        // Convert under a read lock, compute unlocked, intern under a write
        // lock.
        let (f, divisors) = {
            let inner = self.inner.read();
            let arena = &inner.arena;
            // `to_multipolys` returns one polynomial per input expression.
            let Some(f) = to_multipolys(arena, &[self.raw_id()], &var_ids, OP)?
                .into_iter()
                .next()
            else {
                return Err(SymplexError::ComputationFailed {
                    operation: OP,
                    reason: "to_multipolys returned no polynomial for one input".into(),
                });
            };
            (f, to_multipolys(arena, &basis_ids, &var_ids, OP)?)
        };
        let r = reduce_in_order(&f, &divisors, order);
        let id = {
            let mut inner = self.inner.write();
            multipoly_to_expr(&mut inner.arena, &r, &var_ids)
        };
        Ok(self.wrap(id))
    }

    // ── Exact real roots ───────────────────────────────────────────

    /// The distinct real roots of `self` as a polynomial in `var`, as
    /// exact expressions in increasing order (SymPy `real_roots`, except
    /// that a repeated root is listed once, as in
    /// [`count_real_roots`](Self::count_real_roots)).
    ///
    /// Rational roots are returned as numbers.  Every other root is a
    /// `RootOf(g, k)` node, where `g` is the irreducible factor over ℤ
    /// that vanishes there and `k` indexes `g`'s roots sorted by real then
    /// imaginary part — the same node `solve` produces for degree ≥ 5, so
    /// it evaluates numerically (`eval_f64`) and prints as `RootOf(…)`.
    /// The order is decided exactly with Sturm sequences.
    ///
    /// Returns `None` if `self` is not a polynomial in `var` with rational
    /// coefficients or is constant; a polynomial without real roots gives
    /// `Some(vec![])`.  `None` is also returned in the rare case that a
    /// root cannot be *named* reliably: the `RootOf` index of a root is
    /// not stable because another root of the same factor has the same
    /// real part to within `2⁻⁶⁰` (the (re, im) order would then depend
    /// on rounding), or the numeric root finder behind the index did not
    /// converge.  Every `RootOf(g, k)` that is returned has been checked
    /// to evaluate inside the root's exact isolating interval.
    ///
    /// If the factorisation over ℤ cannot be certified complete (the
    /// recombination budget of Berlekamp–Zassenhaus is exhausted — a
    /// polynomial with very many small factors modulo every prime), `g` is
    /// a square-free but possibly reducible factor.  `RootOf(g, k)` is
    /// still well defined and evaluates correctly; it just is not a
    /// canonical form, so two such nodes naming the same number may
    /// compare unequal.  Degree alone is not a limit: a degree-50
    /// irreducible polynomial is certified in well under a second.
    ///
    /// For one root, [`root_of`](Self::root_of) is cheaper: it names only
    /// that root.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let roots = (&x.powi(3) - &x * 2).real_roots(&x).unwrap();   // −√2, 0, √2
    /// assert_eq!(roots.len(), 3);
    /// assert_eq!(roots[1], ctx.int(0));
    /// assert!((roots[2].eval_f64().unwrap() - 2f64.sqrt()).abs() < 1e-12);
    /// assert!((&x.powi(2) + 1).real_roots(&x).unwrap().is_empty());
    /// ```
    #[must_use]
    pub fn real_roots(&self, var: &Ex) -> Option<Vec<Ex>> {
        let var_id = self.checked_id(var);
        // Convert under a read lock, compute unlocked, intern under a write
        // lock.
        let f = {
            let inner = self.inner.read();
            expr_to_poly(&inner.arena, self.raw_id(), var_id)?
        };
        if f.degree().unwrap_or(0) == 0 {
            return None;
        }
        let mut cells = RealRootCells::locate(&f)?;
        let roots: Vec<RealRoot> = (0..cells.cells.len())
            .map(|i| cells.name(i))
            .collect::<Option<_>>()?;
        let ids = self.intern_real_roots(&cells.factors, roots, var_id);
        Some(ids.into_iter().map(|id| self.wrap(id)).collect())
    }

    /// Intern named real roots: rationals as numbers, the rest as
    /// `RootOf(g, k)` with each factor `g` converted once.
    fn intern_real_roots(
        &self,
        factors: &[Poly],
        roots: Vec<RealRoot>,
        var_id: ExprId,
    ) -> Vec<ExprId> {
        let mut inner = self.inner.write();
        let arena = &mut inner.arena;
        let mut factor_ids: Vec<Option<ExprId>> = vec![None; factors.len()];
        roots
            .into_iter()
            .map(|root| match root {
                RealRoot::Rational(r) => arena.num_ratio(r),
                RealRoot::RootOf { factor, index } => {
                    let g_expr = *factor_ids[factor]
                        .get_or_insert_with(|| poly_to_expr(arena, &factors[factor], var_id));
                    let idx = arena.int(index as i64);
                    arena.intern(ExprNode::RootOf(g_expr, idx))
                }
            })
            .collect()
    }

    /// The `index`-th distinct real root of `self` in `var`, counting from
    /// the smallest (0-based) — `real_roots(var)[index]` (SymPy
    /// `rootof(f, index)` for the real roots).
    ///
    /// `None` under the same conditions as [`real_roots`](Self::real_roots),
    /// or if there are at most `index` real roots.  Only the requested
    /// root is named (the numeric root finding behind a `RootOf` index
    /// runs once, for that root's factor), so this is the method to use
    /// for one root of a high-degree polynomial — the root in `(0, 1)` of
    /// a degree-50 binomial-tail polynomial takes a few seconds in a
    /// debug build.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let f = &x.powi(5) - &x - 1;   // one real root ≈ 1.1673
    /// let r = f.root_of(&x, 0).unwrap();
    /// assert!((r.eval_f64().unwrap() - 1.1673039782614187).abs() < 1e-12);
    /// assert!(f.root_of(&x, 1).is_none());
    /// ```
    #[must_use]
    pub fn root_of(&self, var: &Ex, index: usize) -> Option<Ex> {
        let var_id = self.checked_id(var);
        let f = {
            let inner = self.inner.read();
            expr_to_poly(&inner.arena, self.raw_id(), var_id)?
        };
        if f.degree().unwrap_or(0) == 0 {
            return None;
        }
        let mut cells = RealRootCells::locate(&f)?;
        let root = cells.name(index)?;
        let ids = self.intern_real_roots(&cells.factors, vec![root], var_id);
        ids.into_iter().next().map(|id| self.wrap(id))
    }

    // ── Factoring modulo a prime ───────────────────────────────────

    /// Factorisation of `self`, a polynomial in `var`, over the prime
    /// field `GF(p)` (SymPy `factor_list(f, modulus=p)`).
    ///
    /// Returns `(lc, [(factor, multiplicity), …])` with `self ≡ lc · ∏
    /// factorᵢ^multᵢ (mod p)`: `lc` is the leading coefficient reduced mod
    /// `p`, each factor is monic and irreducible over `GF(p)` with
    /// coefficients in `[0, p)`, and the list is sorted by degree then
    /// coefficients.  Rational coefficients are reduced through the
    /// inverse of their denominator.  A polynomial that vanishes
    /// identically mod `p` gives `(0, [])`; a constant gives `(c mod p,
    /// [])`.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` if `p` is not prime, if `p` is `2` or at least
    /// `2³¹` (the finite-field arithmetic supports odd primes below
    /// [`factor_zassenhaus::MAX_PRIME`](crate::factor_zassenhaus::MAX_PRIME)),
    /// if `self` is not a polynomial in `var` with rational coefficients,
    /// or if some coefficient has a denominator divisible by `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// // x² + 1 ≡ (x + 2)(x + 3) (mod 5)
    /// let (lc, factors) = (&x.powi(2) + 1).factor_mod(&x, 5).unwrap();
    /// assert_eq!(lc, ctx.int(1));
    /// assert_eq!(factors, vec![(&x + 2, 1), (&x + 3, 1)]);
    /// // … but irreducible mod 3.
    /// let (_, factors) = (&x.powi(2) + 1).factor_mod(&x, 3).unwrap();
    /// assert_eq!(factors, vec![(&x.powi(2) + 1, 1)]);
    /// assert!((&x.powi(2) + 1).factor_mod(&x, 6).is_err());
    /// ```
    pub fn factor_mod(&self, var: &Ex, p: u64) -> Result<(Ex, Vec<(Ex, u32)>), SymplexError> {
        const OP: &str = "factor_mod";
        let var_id = self.checked_id(var);
        if !crate::domains::ntheory::isprime(p) {
            return Err(invalid(OP, format!("modulus {p} is not prime")));
        }
        if p == 2 || p >= crate::poly::factor_zassenhaus::MAX_PRIME {
            return Err(invalid(
                OP,
                format!(
                    "modulus {p} is not supported: p must be an odd prime below {}",
                    crate::poly::factor_zassenhaus::MAX_PRIME
                ),
            ));
        }
        // Convert under a read lock, compute unlocked, intern under a write
        // lock.
        let f = {
            let inner = self.inner.read();
            expr_to_poly(&inner.arena, self.raw_id(), var_id).ok_or_else(|| {
                invalid(
                    OP,
                    "expression is not a polynomial in the given variable with rational coefficients",
                )
            })?
        };
        let pb = BigInt::from(p);
        if f.coeffs().iter().any(|c| (c.denom() % &pb).is_zero()) {
            return Err(invalid(
                OP,
                format!(
                    "a coefficient has a denominator divisible by {p}, so it has no inverse mod {p}"
                ),
            ));
        }
        let (lc, factors) =
            crate::poly::factor_zassenhaus::factor_mod_p(&f, p).ok_or_else(|| {
                SymplexError::ComputationFailed {
                    operation: OP,
                    reason: "factor_mod_p rejected a valid modulus and polynomial".into(),
                }
            })?;
        let (lc_id, factor_ids) = {
            let mut inner = self.inner.write();
            let arena = &mut inner.arena;
            let lc_id = arena.int(lc as i64);
            let factor_ids: Vec<(ExprId, u32)> = factors
                .iter()
                .map(|(coeffs, m)| {
                    let g = Poly::from_coeffs(
                        coeffs
                            .iter()
                            .map(|&c| Ratio::from_integer(BigInt::from(c)))
                            .collect(),
                    );
                    (poly_to_expr(arena, &g, var_id), *m)
                })
                .collect();
            (lc_id, factor_ids)
        };
        Ok((
            self.wrap(lc_id),
            factor_ids
                .into_iter()
                .map(|(id, m)| (self.wrap(id), m))
                .collect(),
        ))
    }

    // ── Symbolic resultant / discriminant ──────────────────────────

    /// Resultant `res_var(self, other)` of two polynomials in `var` whose
    /// coefficients may be symbolic (SymPy `resultant(f, g, var)`).
    ///
    /// Computed as the determinant of the Sylvester matrix over `Ex`
    /// entries and expanded, so the result is a polynomial in the
    /// parameters; [`resultant`](Self::resultant) is the faster exact
    /// route when every coefficient is rational.  Returns `None` if either
    /// expression is not polynomial in `var` (or `var` is not a symbol).
    /// The resultant of two constants is `1`; if one polynomial is a
    /// constant `c` and the other has degree `d`, the result is `c^d`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    /// // res(x − a, x − b) = g(a) = a − b
    /// let r = (&x - &a).resultant_symbolic(&(&x - &b), &x).unwrap();
    /// assert_eq!(r, &a - &b);
    /// // res(x² + a, x + b) = a + b²
    /// let r = (&x.powi(2) + &a).resultant_symbolic(&(&x + &b), &x).unwrap();
    /// assert_eq!(r, &a + &b.powi(2));
    /// ```
    #[must_use]
    pub fn resultant_symbolic(&self, other: &Ex, var: &Ex) -> Option<Ex> {
        let _ = self.checked_id(other);
        let _ = self.checked_id(var);
        let f = crate::api::poly_ex::Poly::new(self, &[var])?;
        let g = crate::api::poly_ex::Poly::new(other, &[var])?;
        if f.is_zero() || g.is_zero() {
            return Some(self.context().zero());
        }
        let fc = f.all_coeffs()?;
        let gc = g.all_coeffs()?;
        sylvester_resultant(&self.context(), &fc, &gc)
    }

    /// Discriminant of `self` as a polynomial in `var` with possibly
    /// symbolic coefficients (SymPy `discriminant(f, var)`).
    ///
    /// `disc(f) = (−1)^{n(n−1)/2} · res(f, f′) / lc(f)`, evaluated
    /// division-free: the leading coefficient is eliminated from the
    /// Sylvester matrix of `f` and `f′` by one row operation before taking
    /// the determinant, so the result is an expanded polynomial in the
    /// parameters (`b² − 4ac` for `ax² + bx + c`).  Returns `None` for
    /// non-polynomial or constant input; a linear polynomial has
    /// discriminant `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, a, b, c) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"), ctx.symbol("c"));
    /// let quad = &a * &x.powi(2) + &b * &x + &c;
    /// assert_eq!(quad.discriminant_symbolic(&x).unwrap(), &b.powi(2) - &a * &c * 4);
    /// // depressed cubic x³ + px + q: −4p³ − 27q²
    /// let (p, q) = (ctx.symbol("p"), ctx.symbol("q"));
    /// let cubic = &x.powi(3) + &p * &x + &q;
    /// assert_eq!(cubic.discriminant_symbolic(&x).unwrap(), -(&p.powi(3) * 4) - &q.powi(2) * 27);
    /// ```
    #[must_use]
    pub fn discriminant_symbolic(&self, var: &Ex) -> Option<Ex> {
        let _ = self.checked_id(var);
        let f = crate::api::poly_ex::Poly::new(self, &[var])?;
        let coeffs = f.all_coeffs()?; // highest degree first
        let n = coeffs.len().checked_sub(1)?;
        if n == 0 {
            return None;
        }
        let ctx = self.context();
        if n == 1 {
            return Some(ctx.one());
        }
        // Sylvester matrix of f (n + 1 coefficients) and f′ (n coefficients):
        // (2n − 1) × (2n − 1), with n − 1 rows of f above n rows of f′.
        // The first column is a_n at row 0 and n·a_n at row n − 1; the row
        // operation R_{n−1} ← R_{n−1} − n·R_0 leaves a_n alone in the
        // column, so det = a_n · det(minor) and the minor gives disc up to
        // the sign (−1)^{n(n−1)/2}.
        let deriv: Vec<Ex> = coeffs[..n]
            .iter()
            .enumerate()
            .map(|(j, c)| c * ((n - j) as i64))
            .collect();
        let size = 2 * n - 1;
        let zero = ctx.zero();
        let mut rows: Vec<Vec<Ex>> = Vec::with_capacity(size);
        for i in 0..n - 1 {
            let mut row = vec![zero.clone(); size];
            for (j, c) in coeffs.iter().enumerate() {
                row[i + j] = c.clone();
            }
            rows.push(row);
        }
        for i in 0..n {
            let mut row = vec![zero.clone(); size];
            for (j, c) in deriv.iter().enumerate() {
                row[i + j] = c.clone();
            }
            rows.push(row);
        }
        // R_{n−1} − n·R_0 = [0, −a_{n−1}, −2a_{n−2}, …, −n·a_0, 0, …]:
        // entry j is −j · coeffs[j] for 1 ≤ j ≤ n.
        for (j, (slot, c)) in rows[n - 1].iter_mut().zip(coeffs.iter()).enumerate() {
            *slot = -&(c * (j as i64));
        }
        // Drop row 0 and column 0.
        let minor: Vec<Vec<Ex>> = rows[1..].iter().map(|row| row[1..].to_vec()).collect();
        // Invariant: `minor` is a square `(2n − 2) × (2n − 2)` matrix with
        // `n ≥ 2`, so construction and `det` cannot fail; an `Err` would be
        // an internal error, folded into `None` because the return type
        // cannot carry it (see `sylvester_resultant`).
        let matrix = crate::domains::matrix::Matrix::new(minor);
        debug_assert!(
            matrix.is_ok(),
            "discriminant minor is square by construction"
        );
        let det = matrix.ok()?.det();
        debug_assert!(det.is_ok(), "determinant of a square matrix over Ex");
        let det = det.ok()?.expand();
        Some(if (n * (n - 1) / 2) % 2 == 1 {
            -det
        } else {
            det
        })
    }
}
