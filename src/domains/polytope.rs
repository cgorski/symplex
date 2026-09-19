//! Exact convex polyhedra in ℚⁿ given by half-spaces.
//!
//! A [`Polytope`] is an intersection of half-spaces `aᵢ·x + bᵢ ≥ 0` with
//! rational data.  Everything here is exact: vertices come from solving
//! `n × n` systems with [`QMatrix`], emptiness and boundedness from the
//! exact LP, volumes (any dimension) from an exact facet decomposition
//! around the vertex centroid.  The type is meant for the geometric bookkeeping
//! around certificate searches — which cells a decision tree produces,
//! where to cut them, whether two descriptions coincide — not for large
//! polyhedra: [`vertices`](Polytope::vertices) is `O(C(m, n))` linear solves.
//!
//! # Examples
//!
//! ```
//! use symplex::polytope::Polytope;
//! use symplex::linprog::{q, qi};
//!
//! // The triangle 0 ≤ x, 0 ≤ y, x + y ≤ 1, described with a redundant face x ≤ 2.
//! let tri = Polytope::from_rows(&[
//!     (vec![qi(1), qi(0)], qi(0)),     //  x ≥ 0
//!     (vec![qi(0), qi(1)], qi(0)),     //  y ≥ 0
//!     (vec![qi(-1), qi(-1)], qi(1)),   //  1 − x − y ≥ 0
//!     (vec![qi(-1), qi(0)], qi(2)),    //  2 − x ≥ 0   (redundant)
//! ]).unwrap();
//! let v = tri.vertices().unwrap();
//! assert_eq!(v.len(), 3);
//! assert_eq!(tri.volume().unwrap(), q(1, 2));
//! assert!(tri.contains(&[q(1, 4), q(1, 4)]));
//! assert_eq!(tri.irredundant().unwrap().num_halfspaces(), 3);
//! let (left, right) = tri.split(&[qi(-1), qi(0)], q(1, 2));   // cut at x = 1/2
//! assert_eq!(left.volume().unwrap() + right.volume().unwrap(), q(1, 2));
//! ```

use num_bigint::BigInt;
use num_traits::{Signed, Zero};

use crate::api::expr::Ex;
use crate::api::poly_ex::Poly;
use crate::base::errors::SymplexError;
use crate::domains::exact_matrix::QMatrix;
use crate::domains::linprog::{LpProblem, LpStatus, Q};

fn invalid(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::InvalidArgument {
        operation,
        reason: reason.into(),
    }
}

/// Per-coordinate `[min, max]` of a polytope; `None` = unbounded on that side.
pub type BoundingBox = Vec<(Option<Q>, Option<Q>)>;

/// One half-space `coeffs · x + constant ≥ 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HalfSpace {
    /// Coefficients `a`, one per coordinate.
    pub coeffs: Vec<Q>,
    /// The constant `b`.
    pub constant: Q,
}

impl HalfSpace {
    /// The value `a·x + b` at `x`.
    pub fn value(&self, x: &[Q]) -> Q {
        self.coeffs
            .iter()
            .zip(x)
            .fold(self.constant.clone(), |acc, (a, xi)| acc + a * xi)
    }

    /// `true` if `a·x + b ≥ 0`.
    pub fn contains(&self, x: &[Q]) -> bool {
        !self.value(x).is_negative()
    }

    /// The opposite half-space `−a·x − b ≥ 0` (the closed complement's
    /// closure: both share the hyperplane).
    pub fn flipped(&self) -> HalfSpace {
        HalfSpace {
            coeffs: self.coeffs.iter().map(|c| -c).collect(),
            constant: -&self.constant,
        }
    }
}

/// A convex polyhedron `{x ∈ ℚⁿ : aᵢ·x + bᵢ ≥ 0 ∀i}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Polytope {
    dim: usize,
    halfspaces: Vec<HalfSpace>,
}

impl Polytope {
    /// Build from half-spaces; all must have `dim` coefficients.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the list is empty, a
    /// coefficient vector is empty or the lengths differ.
    pub fn new(halfspaces: Vec<HalfSpace>) -> Result<Self, SymplexError> {
        let Some(first) = halfspaces.first() else {
            return Err(invalid(
                "Polytope::new",
                "at least one half-space is required",
            ));
        };
        let dim = first.coeffs.len();
        if dim == 0 {
            return Err(invalid(
                "Polytope::new",
                "half-spaces need at least one coordinate",
            ));
        }
        if let Some((i, h)) = halfspaces
            .iter()
            .enumerate()
            .find(|(_, h)| h.coeffs.len() != dim)
        {
            return Err(invalid(
                "Polytope::new",
                format!(
                    "half-space {i} has {} coefficients, expected {dim}",
                    h.coeffs.len()
                ),
            ));
        }
        Ok(Polytope { dim, halfspaces })
    }

    /// Build from `(coeffs, constant)` rows meaning `coeffs·x + constant ≥ 0`.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn from_rows(rows: &[(Vec<Q>, Q)]) -> Result<Self, SymplexError> {
        Self::new(
            rows.iter()
                .map(|(c, k)| HalfSpace {
                    coeffs: c.clone(),
                    constant: k.clone(),
                })
                .collect(),
        )
    }

    /// Build from affine expressions `hᵢ(x) ≥ 0` in the variables `vars`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if an expression is not affine in
    /// `vars` with rational coefficients, or the lists are empty.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::polytope::Polytope;
    /// use symplex::linprog::q;
    ///
    /// let ctx = Context::new();
    /// let (r, t) = (ctx.symbol("r"), ctx.symbol("t"));
    /// let cell = Polytope::from_exprs(&[r.clone(), ctx.rational(1, 2) - &r, t.clone(), 1 - &t], &[r, t]).unwrap();
    /// assert_eq!(cell.volume().unwrap(), q(1, 2));
    /// ```
    pub fn from_exprs(hyps: &[Ex], vars: &[Ex]) -> Result<Self, SymplexError> {
        const OP: &str = "Polytope::from_exprs";
        if vars.is_empty() {
            return Err(invalid(OP, "at least one variable is required"));
        }
        let gens: Vec<&Ex> = vars.iter().collect();
        let mut halfspaces = Vec::with_capacity(hyps.len());
        for h in hyps {
            let p = Poly::try_new(h, &gens).map_err(|e| match e {
                SymplexError::InvalidArgument { reason, .. } => invalid(OP, reason),
                other => other,
            })?;
            if !p.is_linear() {
                return Err(invalid(OP, format!("`{h}` is not affine in the variables")));
            }
            let mut coeffs = vec![Q::zero(); vars.len()];
            let mut constant = Q::zero();
            for (m, c) in p.terms_iter() {
                let Some(c) = c.as_rational() else {
                    return Err(invalid(OP, format!("`{h}` has a symbolic coefficient")));
                };
                match m.iter().position(|&e| e == 1) {
                    Some(i) => coeffs[i] = c,
                    None => constant = c,
                }
            }
            halfspaces.push(HalfSpace { coeffs, constant });
        }
        Self::new(halfspaces)
    }

    /// The half-spaces as affine expressions `aᵢ·vars + bᵢ` (each `≥ 0`), the
    /// inverse of [`from_exprs`](Self::from_exprs) — e.g. to feed a cell to
    /// [`prove_nonnegative_on_polyhedron`](crate::certificates::prove_nonnegative_on_polyhedron).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if `vars.len() != dim`.
    pub fn to_exprs(&self, vars: &[Ex]) -> Result<Vec<Ex>, SymplexError> {
        if vars.len() != self.dim {
            return Err(invalid(
                "Polytope::to_exprs",
                format!(
                    "{} variables for a {}-dimensional polytope",
                    vars.len(),
                    self.dim
                ),
            ));
        }
        let Some(first) = vars.first() else {
            return Err(invalid(
                "Polytope::to_exprs",
                "at least one variable is required",
            ));
        };
        let ctx = first.context();
        Ok(self
            .halfspaces
            .iter()
            .map(|h| {
                let mut e = ctx.from_ratio(h.constant.clone());
                for (a, v) in h.coeffs.iter().zip(vars) {
                    if !a.is_zero() {
                        e += ctx.from_ratio(a.clone()) * v;
                    }
                }
                e
            })
            .collect())
    }

    /// The ambient dimension `n`.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The half-spaces, in input order.
    pub fn halfspaces(&self) -> &[HalfSpace] {
        &self.halfspaces
    }

    /// Number of half-spaces.
    pub fn num_halfspaces(&self) -> usize {
        self.halfspaces.len()
    }

    /// `true` if `x` satisfies every half-space.
    pub fn contains(&self, x: &[Q]) -> bool {
        x.len() == self.dim && self.halfspaces.iter().all(|h| h.contains(x))
    }

    /// The polytope with one more half-space `coeffs·x + constant ≥ 0`.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs.len() != dim` (a programming error, like a shape
    /// mismatch); use [`try_with_halfspace`](Self::try_with_halfspace) for
    /// a `Result`.
    pub fn with_halfspace(&self, coeffs: &[Q], constant: Q) -> Polytope {
        match self.try_with_halfspace(coeffs, constant) {
            Ok(p) => p,
            Err(e) => panic!("Polytope::with_halfspace: {e}"),
        }
    }

    /// [`with_halfspace`](Self::with_halfspace) returning an error on a
    /// dimension mismatch.
    pub fn try_with_halfspace(&self, coeffs: &[Q], constant: Q) -> Result<Polytope, SymplexError> {
        if coeffs.len() != self.dim {
            return Err(invalid(
                "Polytope::with_halfspace",
                format!(
                    "{} coefficients for a {}-dimensional polytope",
                    coeffs.len(),
                    self.dim
                ),
            ));
        }
        let mut hs = self.halfspaces.clone();
        hs.push(HalfSpace {
            coeffs: coeffs.to_vec(),
            constant,
        });
        Ok(Polytope {
            dim: self.dim,
            halfspaces: hs,
        })
    }

    /// Cut by the hyperplane `coeffs·x + constant = 0`: the two closed
    /// pieces `P ∩ {≥ 0}` and `P ∩ {≤ 0}`.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs.len() != dim`.
    pub fn split(&self, coeffs: &[Q], constant: Q) -> (Polytope, Polytope) {
        let pos = self.with_halfspace(coeffs, constant.clone());
        let neg_coeffs: Vec<Q> = coeffs.iter().map(|c| -c).collect();
        let neg = self.with_halfspace(&neg_coeffs, -constant);
        (pos, neg)
    }

    /// Exact LP `min c·x` over the polytope (variables free).  `Ok(None)`
    /// if the polytope is empty or the objective is unbounded below.
    fn minimize(&self, c: &[Q]) -> Result<Option<(Q, Vec<Q>)>, SymplexError> {
        let mut lp = LpProblem::minimize(c.to_vec());
        for i in 0..self.dim {
            lp = lp.free(i);
        }
        for h in &self.halfspaces {
            lp = lp.ge(h.coeffs.clone(), -&h.constant);
        }
        let sol = lp.solve()?;
        Ok(match sol.status {
            LpStatus::Optimal => sol.objective.map(|v| (v, sol.x)),
            _ => None,
        })
    }

    /// Is the polytope empty?  Exact (an LP feasibility check).
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`] if the LP exceeds its pivot cap.
    pub fn is_empty(&self) -> Result<bool, SymplexError> {
        let zero = vec![Q::zero(); self.dim];
        Ok(self.minimize(&zero)?.is_none())
    }

    /// A point of the polytope (an LP vertex), or `None` if empty.
    ///
    /// # Errors
    ///
    /// As [`is_empty`](Self::is_empty).
    pub fn any_point(&self) -> Result<Option<Vec<Q>>, SymplexError> {
        let zero = vec![Q::zero(); self.dim];
        Ok(self.minimize(&zero)?.map(|(_, x)| x))
    }

    /// Exact coordinate-wise `[min, max]` of the polytope, `None` for a
    /// coordinate that is unbounded on that side; `Ok(None)` if empty.
    ///
    /// # Errors
    ///
    /// As [`is_empty`](Self::is_empty).
    pub fn bounding_box(&self) -> Result<Option<BoundingBox>, SymplexError> {
        if self.is_empty()? {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let mut c = vec![Q::zero(); self.dim];
            c[i] = Q::from_integer(BigInt::from(1));
            let lo = self.minimize(&c)?.map(|(v, _)| v);
            c[i] = Q::from_integer(BigInt::from(-1));
            let hi = self.minimize(&c)?.map(|(v, _)| -v);
            out.push((lo, hi));
        }
        Ok(Some(out))
    }

    /// Is the polytope bounded (a polytope proper)?  The empty set counts
    /// as bounded.
    ///
    /// # Errors
    ///
    /// As [`is_empty`](Self::is_empty).
    pub fn is_bounded(&self) -> Result<bool, SymplexError> {
        Ok(match self.bounding_box()? {
            None => true,
            Some(b) => b.iter().all(|(lo, hi)| lo.is_some() && hi.is_some()),
        })
    }

    /// The vertices (0-dimensional faces), exactly, in no particular order.
    ///
    /// Every choice of `n` half-spaces whose hyperplanes meet in a single
    /// point is solved with [`QMatrix::solve`]; the point is kept if it lies
    /// in the polytope.  Degenerate vertices (more than `n` tight
    /// hyperplanes) appear once.  An unbounded polyhedron still has its
    /// vertices enumerated; use [`is_bounded`](Self::is_bounded) to tell.
    ///
    /// # Errors
    ///
    /// Only propagates internal failures; an empty polytope gives an empty
    /// list.
    pub fn vertices(&self) -> Result<Vec<Vec<Q>>, SymplexError> {
        let n = self.dim;
        let m = self.halfspaces.len();
        if m < n {
            return Ok(Vec::new());
        }
        let mut found: Vec<Vec<Q>> = Vec::new();
        let mut idx: Vec<usize> = (0..n).collect();
        loop {
            // Solve the n×n system  aᵢ·x = −bᵢ  for the chosen half-spaces.
            let a = QMatrix::from_fn(n, n, |r, c| self.halfspaces[idx[r]].coeffs[c].clone());
            let b = QMatrix::from_fn(n, 1, |r, _| -&self.halfspaces[idx[r]].constant);
            if let Ok(x) = a.solve(&b) {
                let point: Vec<Q> = x.col(0);
                if self.contains(&point) && !found.contains(&point) {
                    found.push(point);
                }
            }
            // Next n-combination of 0..m.
            let mut k = n;
            loop {
                if k == 0 {
                    return Ok(found);
                }
                k -= 1;
                if idx[k] < m - n + k {
                    idx[k] += 1;
                    for j in (k + 1)..n {
                        idx[j] = idx[j - 1] + 1;
                    }
                    break;
                }
            }
        }
    }

    /// The half-spaces that are tight at some vertex, i.e. the description
    /// with redundant half-spaces removed — for a **bounded, full-
    /// dimensional** polytope.  (For an unbounded or lower-dimensional
    /// polyhedron a half-space can be irredundant without touching a
    /// vertex; the result is then a superset of the input polyhedron.)
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the polytope has no vertices.
    pub fn irredundant(&self) -> Result<Polytope, SymplexError> {
        let verts = self.vertices()?;
        if verts.is_empty() {
            return Err(invalid(
                "Polytope::irredundant",
                "the polytope has no vertices",
            ));
        }
        let kept: Vec<HalfSpace> = self
            .halfspaces
            .iter()
            .filter(|h| verts.iter().any(|v| h.value(v).is_zero()))
            .cloned()
            .collect();
        Polytope::new(kept)
    }

    /// Centroid (mean) of the vertices, or `None` if there are none.  For a
    /// full-dimensional polytope this is an interior point.
    ///
    /// # Errors
    ///
    /// As [`vertices`](Self::vertices).
    pub fn vertex_centroid(&self) -> Result<Option<Vec<Q>>, SymplexError> {
        let verts = self.vertices()?;
        Ok(centroid(&verts, self.dim))
    }

    /// Exact `n`-dimensional volume (length, area, volume, …) of a bounded
    /// polytope in any dimension.  Zero for an empty or lower-dimensional
    /// polytope.
    ///
    /// Computed by the facet decomposition `vol(P) = (1/n) Σ_F dist(o, F) ·
    /// vol(F)` around the vertex centroid `o`, recursing on each facet's
    /// exact `(n − 1)`-dimensional H-representation (obtained by eliminating
    /// one coordinate); the `‖a_F‖` factors cancel, so every intermediate
    /// value is rational.  Cost grows like the number of faces, which is
    /// fine for the cells of a decision tree in dimension `≤ 5` and the
    /// wrong tool for large polyhedra.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the polyhedron is unbounded.
    ///
    /// ```
    /// use symplex::polytope::Polytope;
    /// use symplex::linprog::{q, qi};
    ///
    /// // The 4-simplex x ≥ 0, Σx ≤ 1 has volume 1/4! = 1/24.
    /// let mut rows: Vec<(Vec<_>, _)> = (0..4).map(|i| { let mut e = vec![qi(0); 4]; e[i] = qi(1); (e, qi(0)) }).collect();
    /// rows.push((vec![qi(-1); 4], qi(1)));
    /// assert_eq!(Polytope::from_rows(&rows).unwrap().volume().unwrap(), q(1, 24));
    /// ```
    pub fn volume(&self) -> Result<Q, SymplexError> {
        const OP: &str = "Polytope::volume";
        if !self.is_bounded()? {
            return Err(invalid(OP, "the polyhedron is unbounded"));
        }
        Ok(self.volume_bounded())
    }

    /// Volume of a polytope already known to be bounded (recursive core).
    fn volume_bounded(&self) -> Q {
        let n = self.dim;
        let verts = match self.vertices() {
            Ok(v) => v,
            Err(_) => return Q::zero(),
        };
        if verts.len() < n + 1 {
            return Q::zero(); // empty or lower-dimensional
        }
        if n == 1 {
            let lo = verts.iter().map(|v| &v[0]).min().cloned();
            let hi = verts.iter().map(|v| &v[0]).max().cloned();
            return match (lo, hi) {
                (Some(l), Some(h)) => h - l,
                _ => Q::zero(),
            };
        }
        let Some(o) = centroid(&verts, n) else {
            return Q::zero();
        };
        // Distinct facet hyperplanes (a half-space listed twice, or scaled,
        // must be counted once).
        let mut seen: Vec<HalfSpace> = Vec::new();
        let mut total = Q::zero();
        for h in &self.halfspaces {
            let Some(norm) = normalized(h) else {
                continue; // a·x + b ≥ 0 with a = 0: not a facet
            };
            if seen.contains(&norm) {
                continue;
            }
            seen.push(norm);
            // Vertices on this hyperplane: fewer than n means no facet.
            let on: usize = verts.iter().filter(|v| h.value(v).is_zero()).count();
            if on < n {
                continue;
            }
            // Eliminate coordinate k with the largest |a_k| ≠ 0:
            //   x_k = −(b + Σ_{i≠k} a_i x_i) / a_k.
            let Some(k) = (0..n)
                .filter(|&i| !h.coeffs[i].is_zero())
                .max_by(|&i, &j| h.coeffs[i].abs().cmp(&h.coeffs[j].abs()))
            else {
                continue;
            };
            let ak = &h.coeffs[k];
            let mut facet_rows: Vec<HalfSpace> = Vec::new();
            for g in &self.halfspaces {
                // g(x) with x_k substituted: coefficients over the n−1 kept coordinates.
                let gk = &g.coeffs[k];
                let mut coeffs: Vec<Q> = Vec::with_capacity(n - 1);
                for i in (0..n).filter(|&i| i != k) {
                    coeffs.push(&g.coeffs[i] - gk * &h.coeffs[i] / ak);
                }
                let constant = &g.constant - gk * &h.constant / ak;
                if coeffs.iter().all(Zero::is_zero) {
                    continue; // constant constraint (the facet's own hyperplane, or a redundant one)
                }
                facet_rows.push(HalfSpace { coeffs, constant });
            }
            let Ok(facet) = Polytope::new(facet_rows) else {
                continue;
            };
            let facet_vol = facet.volume_bounded();
            if facet_vol.is_zero() {
                continue;
            }
            // dist(o, F) · vol_{n−1}(F) = (a·o + b)/‖a‖ · vol(proj) · ‖a‖/|a_k|.
            total += h.value(&o) / ak.abs() * facet_vol;
        }
        total / Q::from_integer(BigInt::from(n))
    }
}

/// `h` scaled so that its first non-zero coefficient is `1` (identifies a
/// hyperplane up to positive scaling); `None` for a trivial half-space.
fn normalized(h: &HalfSpace) -> Option<HalfSpace> {
    let first = h.coeffs.iter().find(|c| !c.is_zero())?;
    Some(HalfSpace {
        coeffs: h.coeffs.iter().map(|c| c / first).collect(),
        constant: &h.constant / first,
    })
}

fn centroid(points: &[Vec<Q>], dim: usize) -> Option<Vec<Q>> {
    if points.is_empty() {
        return None;
    }
    let n = Q::from_integer(BigInt::from(points.len()));
    Some(
        (0..dim)
            .map(|i| points.iter().fold(Q::zero(), |acc, p| acc + &p[i]) / &n)
            .collect(),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Parametric polytopes
// ═══════════════════════════════════════════════════════════════════════════

/// A family of polytopes `{x : hₖ(j, x) ≥ 0}` whose half-spaces are affine
/// in `x` with coefficients polynomial in one parameter `j`.
///
/// [`at`](Self::at) instantiates the family exactly at a rational `j`;
/// [`polytope_at`](Self::polytope_at) / [`vertices_at`](Self::vertices_at) /
/// [`volume_at`](Self::volume_at) do the same with a per-value cache, for
/// tree builders that revisit the same samples.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::polytope::ParametricPolytope;
/// use symplex::linprog::{q, qi};
///
/// let ctx = Context::new();
/// let (j, r, t) = (ctx.symbol("j"), ctx.symbol("r"), ctx.symbol("t"));
/// // 0 ≤ r ≤ 1/2, 0 ≤ t ≤ 1, (2j + 1)·t ≥ j·r + 1
/// let hyps = [r.clone(), ctx.rational(1, 2) - &r, t.clone(), 1 - &t, (&j * 2 + 1) * &t - &j * &r - 1];
/// let mut cell = ParametricPolytope::new(&hyps, &[r, t], &j).unwrap();
/// assert_eq!(cell.volume_at(&qi(2)).unwrap(), q(7, 20));   // ∫₀^{1/2} (1 − (2r + 1)/5) dr
/// assert_eq!(cell.vertices_at(&qi(2)).unwrap().len(), 4);
/// assert!(cell.at(&qi(100)).unwrap().contains(&[q(1, 4), q(3, 4)]));
/// ```
#[derive(Clone, Debug)]
pub struct ParametricPolytope {
    hyps: Vec<Poly>,
    vars: Vec<Ex>,
    param: Ex,
    cache: std::collections::BTreeMap<Q, Instance>,
}

/// A cached instantiation: the polytope and, once computed, its vertices.
type Instance = (Polytope, Option<Vec<Vec<Q>>>);

impl ParametricPolytope {
    /// Build from hypotheses `hₖ(j, x) ≥ 0`, each affine in `vars` with
    /// coefficients polynomial (rational) in `param`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if a hypothesis is not a polynomial
    /// in the variables and the parameter, is not affine in the variables,
    /// or has a symbolic coefficient; or if the lists are empty.
    pub fn new(hyps: &[Ex], vars: &[Ex], param: &Ex) -> Result<Self, SymplexError> {
        const OP: &str = "ParametricPolytope::new";
        if hyps.is_empty() {
            return Err(invalid(OP, "at least one hypothesis is required"));
        }
        if vars.is_empty() {
            return Err(invalid(OP, "at least one variable is required"));
        }
        if vars.contains(param) {
            return Err(invalid(
                OP,
                "the parameter must not be one of the variables",
            ));
        }
        let mut gens: Vec<&Ex> = vars.iter().collect();
        gens.push(param);
        let n = vars.len();
        let mut polys = Vec::with_capacity(hyps.len());
        for h in hyps {
            let p = Poly::try_new(h, &gens).map_err(|e| match e {
                SymplexError::InvalidArgument { reason, .. } => invalid(OP, reason),
                other => other,
            })?;
            if !p.has_rational_coeffs() {
                return Err(invalid(OP, format!("`{h}` has a symbolic coefficient")));
            }
            if p.terms_iter().any(|(m, _)| m[..n].iter().sum::<u32>() > 1) {
                return Err(invalid(OP, format!("`{h}` is not affine in the variables")));
            }
            polys.push(p);
        }
        Ok(ParametricPolytope {
            hyps: polys,
            vars: vars.to_vec(),
            param: param.clone(),
            cache: std::collections::BTreeMap::new(),
        })
    }

    /// The variables, in order.
    pub fn vars(&self) -> &[Ex] {
        &self.vars
    }

    /// The parameter.
    pub fn param(&self) -> &Ex {
        &self.param
    }

    /// The hypotheses as polynomials in `(vars…, param)`.
    pub fn hyps(&self) -> &[Poly] {
        &self.hyps
    }

    /// The polytope at `param = value`, exactly (no cache).
    ///
    /// # Errors
    ///
    /// Only internal failures (the substitution of a rational is always
    /// affine in the variables).
    pub fn at(&self, value: &Q) -> Result<Polytope, SymplexError> {
        let ctx = self.param.context();
        let val = ctx.from_ratio(value.clone());
        let n = self.vars.len();
        let mut rows = Vec::with_capacity(self.hyps.len());
        for h in &self.hyps {
            let p = h.eval_gen(&self.param, &val)?;
            let mut coeffs = vec![Q::zero(); n];
            let mut constant = Q::zero();
            for (m, c) in p.terms_iter() {
                let c = c.as_rational().ok_or_else(|| {
                    invalid(
                        "ParametricPolytope::at",
                        "internal: non-rational coefficient",
                    )
                })?;
                match m.iter().position(|&e| e == 1) {
                    Some(i) => coeffs[i] = c,
                    None => constant = c,
                }
            }
            rows.push(HalfSpace { coeffs, constant });
        }
        Polytope::new(rows)
    }

    fn entry(&mut self, value: &Q) -> Result<&mut Instance, SymplexError> {
        if !self.cache.contains_key(value) {
            let p = self.at(value)?;
            self.cache.insert(value.clone(), (p, None));
        }
        self.cache
            .get_mut(value)
            .ok_or_else(|| invalid("ParametricPolytope::at", "internal: cache"))
    }

    /// The polytope at `param = value`, cached.
    ///
    /// # Errors
    ///
    /// As [`at`](Self::at).
    pub fn polytope_at(&mut self, value: &Q) -> Result<&Polytope, SymplexError> {
        Ok(&self.entry(value)?.0)
    }

    /// The vertices at `param = value`, cached.
    ///
    /// # Errors
    ///
    /// As [`at`](Self::at).
    pub fn vertices_at(&mut self, value: &Q) -> Result<&[Vec<Q>], SymplexError> {
        let e = self.entry(value)?;
        if e.1.is_none() {
            e.1 = Some(e.0.vertices()?);
        }
        Ok(e.1.as_deref().unwrap_or(&[]))
    }

    /// The volume at `param = value` (uses the cached polytope).
    ///
    /// # Errors
    ///
    /// As [`Polytope::volume`].
    pub fn volume_at(&mut self, value: &Q) -> Result<Q, SymplexError> {
        self.polytope_at(value)?.volume()
    }

    /// Is the polytope at `param = value` empty?
    ///
    /// # Errors
    ///
    /// As [`Polytope::is_empty`].
    pub fn is_empty_at(&mut self, value: &Q) -> Result<bool, SymplexError> {
        self.polytope_at(value)?.is_empty()
    }

    /// Does the polytope at `param = value` contain `x`?
    ///
    /// # Errors
    ///
    /// As [`at`](Self::at).
    pub fn contains_at(&mut self, value: &Q, x: &[Q]) -> Result<bool, SymplexError> {
        Ok(self.polytope_at(value)?.contains(x))
    }

    /// Forget every cached instantiation.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::linprog::{q, qi};

    fn cube() -> Polytope {
        // 0 ≤ x, y, z ≤ 1
        let mut rows = Vec::new();
        for i in 0..3 {
            let mut e = vec![qi(0); 3];
            e[i] = qi(1);
            rows.push((e.clone(), qi(0)));
            let mut f = vec![qi(0); 3];
            f[i] = qi(-1);
            rows.push((f, qi(1)));
        }
        Polytope::from_rows(&rows).unwrap()
    }

    #[test]
    fn cube_vertices_volume_box() {
        let c = cube();
        let v = c.vertices().unwrap();
        assert_eq!(v.len(), 8);
        assert_eq!(c.volume().unwrap(), qi(1));
        assert!(c.is_bounded().unwrap());
        assert!(!c.is_empty().unwrap());
        let bb = c.bounding_box().unwrap().unwrap();
        assert_eq!(bb, vec![(Some(qi(0)), Some(qi(1))); 3]);
        assert_eq!(c.vertex_centroid().unwrap().unwrap(), vec![q(1, 2); 3]);
        assert!(c.contains(&[q(1, 2), q(1, 3), q(1, 4)]));
        assert!(!c.contains(&[q(3, 2), q(1, 3), q(1, 4)]));
        assert!(!c.contains(&[q(1, 2), q(1, 3)]));
        // Cut the cube at z = 1/3.
        let (lo, hi) = c.split(&[qi(0), qi(0), qi(-1)], q(1, 3));
        assert_eq!(lo.volume().unwrap(), q(1, 3));
        assert_eq!(hi.volume().unwrap(), q(2, 3));
        assert_eq!(lo.vertices().unwrap().len(), 8);
    }

    #[test]
    fn simplex_and_degenerate_vertex() {
        // Tetrahedron x, y, z ≥ 0, x + y + z ≤ 1: volume 1/6, apex of a
        // pyramid with 4 tight planes when a redundant plane is added.
        let mut rows = vec![
            (vec![qi(1), qi(0), qi(0)], qi(0)),
            (vec![qi(0), qi(1), qi(0)], qi(0)),
            (vec![qi(0), qi(0), qi(1)], qi(0)),
            (vec![qi(-1), qi(-1), qi(-1)], qi(1)),
        ];
        let t = Polytope::from_rows(&rows).unwrap();
        assert_eq!(t.vertices().unwrap().len(), 4);
        assert_eq!(t.volume().unwrap(), q(1, 6));
        // x + y ≤ 1 passes through two vertices without cutting: still 4 vertices.
        rows.push((vec![qi(-1), qi(-1), qi(0)], qi(1)));
        let t2 = Polytope::from_rows(&rows).unwrap();
        assert_eq!(t2.vertices().unwrap().len(), 4);
        assert_eq!(t2.volume().unwrap(), q(1, 6));
        assert_eq!(t2.irredundant().unwrap().num_halfspaces(), 5); // tight at vertices, kept
    }

    #[test]
    fn empty_and_unbounded() {
        let empty = Polytope::from_rows(&[(vec![qi(1)], qi(0)), (vec![qi(-1)], qi(-1))]).unwrap();
        assert!(empty.is_empty().unwrap());
        assert!(empty.vertices().unwrap().is_empty());
        assert_eq!(empty.volume().unwrap(), qi(0));
        assert!(empty.bounding_box().unwrap().is_none());
        assert!(empty.any_point().unwrap().is_none());
        assert!(empty.irredundant().is_err());
        let quadrant =
            Polytope::from_rows(&[(vec![qi(1), qi(0)], qi(0)), (vec![qi(0), qi(1)], qi(0))])
                .unwrap();
        assert!(!quadrant.is_bounded().unwrap());
        assert_eq!(quadrant.vertices().unwrap(), vec![vec![qi(0), qi(0)]]);
        assert!(quadrant.volume().is_err());
        let bb = quadrant.bounding_box().unwrap().unwrap();
        assert_eq!(bb[0], (Some(qi(0)), None));
        assert!(quadrant.any_point().unwrap().is_some());
    }

    #[test]
    fn segment_and_polygon() {
        let seg = Polytope::from_rows(&[(vec![qi(1)], q(1, 3)), (vec![qi(-1)], qi(2))]).unwrap();
        assert_eq!(seg.volume().unwrap(), q(7, 3));
        // Regular-ish hexagon-like polygon: |x| + |y| ≤ 1 (a diamond), area 2.
        let diamond = Polytope::from_rows(&[
            (vec![qi(-1), qi(-1)], qi(1)),
            (vec![qi(1), qi(-1)], qi(1)),
            (vec![qi(-1), qi(1)], qi(1)),
            (vec![qi(1), qi(1)], qi(1)),
        ])
        .unwrap();
        assert_eq!(diamond.vertices().unwrap().len(), 4);
        assert_eq!(diamond.volume().unwrap(), qi(2));
        assert_eq!(diamond.irredundant().unwrap(), diamond);
    }

    #[test]
    fn from_exprs_and_errors() {
        let ctx = crate::api::context::Context::new();
        let (r, t) = (ctx.symbol("r"), ctx.symbol("t"));
        let p =
            Polytope::from_exprs(&[r.clone(), &t - &r, 1 - &t], &[r.clone(), t.clone()]).unwrap();
        assert_eq!(p.volume().unwrap(), q(1, 2));
        assert_eq!(
            p.halfspaces()[1],
            HalfSpace {
                coeffs: vec![qi(-1), qi(1)],
                constant: qi(0)
            }
        );
        let back = p.to_exprs(&[r.clone(), t.clone()]).unwrap();
        assert_eq!(back, vec![r.clone(), &t - &r, 1 - &t]);
        assert!(p.to_exprs(std::slice::from_ref(&r)).is_err());
        assert!(Polytope::from_exprs(&[r.powi(2)], std::slice::from_ref(&r)).is_err());
        assert!(Polytope::from_exprs(&[&r * ctx.symbol("a")], std::slice::from_ref(&r)).is_err());
        assert!(Polytope::from_exprs(std::slice::from_ref(&r), &[]).is_err());
        assert!(Polytope::new(vec![]).is_err());
        assert!(Polytope::from_rows(&[(vec![qi(1)], qi(0)), (vec![qi(1), qi(2)], qi(0))]).is_err());
        assert!(p.try_with_halfspace(&[qi(1)], qi(0)).is_err());
        let four = Polytope::from_rows(&[(vec![qi(1); 4], qi(0))]).unwrap();
        assert!(four.volume().is_err());
        assert_eq!(
            HalfSpace {
                coeffs: vec![qi(2)],
                constant: qi(-1)
            }
            .flipped(),
            HalfSpace {
                coeffs: vec![qi(-2)],
                constant: qi(1)
            }
        );
    }
}
