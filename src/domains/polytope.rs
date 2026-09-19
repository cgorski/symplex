//! Exact convex polyhedra in ℚⁿ given by half-spaces.
//!
//! A [`Polytope`] is an intersection of half-spaces `aᵢ·x + bᵢ ≥ 0` with
//! rational data.  Everything here is exact: vertices come from solving
//! `n × n` systems with [`QMatrix`], emptiness and boundedness from the
//! exact LP, volumes (dimension ≤ 3) from a fan triangulation over the
//! vertex centroid.  The type is meant for the geometric bookkeeping
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

    /// Exact `n`-dimensional volume for `n ≤ 3` (length, area, volume).
    /// Zero for an empty or lower-dimensional polytope.
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] if the polytope is unbounded or
    /// `n > 3`; [`SymplexError::NotImplemented`] is not used.
    pub fn volume(&self) -> Result<Q, SymplexError> {
        const OP: &str = "Polytope::volume";
        if self.dim > 3 {
            return Err(invalid(
                OP,
                format!("volume is implemented for dimension ≤ 3, got {}", self.dim),
            ));
        }
        if !self.is_bounded()? {
            return Err(invalid(OP, "the polyhedron is unbounded"));
        }
        let verts = self.vertices()?;
        let Some(o) = centroid(&verts, self.dim) else {
            return Ok(Q::zero());
        };
        match self.dim {
            1 => {
                let lo = verts.iter().map(|v| &v[0]).min().cloned();
                let hi = verts.iter().map(|v| &v[0]).max().cloned();
                Ok(match (lo, hi) {
                    (Some(l), Some(h)) => h - l,
                    _ => Q::zero(),
                })
            }
            2 => Ok(polygon_area(&verts, &o)),
            _ => {
                // Sum of pyramids over the facets from the interior point o.
                let mut total = Q::zero();
                for h in &self.halfspaces {
                    let face: Vec<Vec<Q>> = verts
                        .iter()
                        .filter(|v| h.value(v).is_zero())
                        .cloned()
                        .collect();
                    if face.len() < 3 {
                        continue;
                    }
                    let Some(fc) = centroid(&face, 3) else {
                        continue;
                    };
                    // Order the face's vertices cyclically in the plane of
                    // the face (project along the largest normal component).
                    let drop = (0..3)
                        .max_by(|&i, &j| h.coeffs[i].abs().cmp(&h.coeffs[j].abs()))
                        .unwrap_or(2);
                    let keep: Vec<usize> = (0..3).filter(|&i| i != drop).collect();
                    let planar: Vec<Vec<Q>> = face
                        .iter()
                        .map(|v| vec![v[keep[0]].clone(), v[keep[1]].clone()])
                        .collect();
                    let pc = vec![fc[keep[0]].clone(), fc[keep[1]].clone()];
                    let order = cyclic_order(&planar, &pc);
                    let k = order.len();
                    for i in 0..k {
                        let a = &face[order[i]];
                        let b = &face[order[(i + 1) % k]];
                        total += tetra_volume(&o, &fc, a, b);
                    }
                }
                Ok(total)
            }
        }
    }
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

/// Indices of `pts` in counter-clockwise order around `center` (exact:
/// half-plane first, then cross product).
fn cyclic_order(pts: &[Vec<Q>], center: &[Q]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..pts.len()).collect();
    let rel = |i: usize| (&pts[i][0] - &center[0], &pts[i][1] - &center[1]);
    let half = |(dx, dy): &(Q, Q)| -> u8 {
        if dy.is_positive() || (dy.is_zero() && !dx.is_negative()) {
            0
        } else {
            1
        }
    };
    idx.sort_by(|&i, &j| {
        let (a, b) = (rel(i), rel(j));
        half(&a).cmp(&half(&b)).then_with(|| {
            // cross(a, b) > 0  ⇔  a comes before b (counter-clockwise).
            let cross = &a.0 * &b.1 - &a.1 * &b.0;
            if cross.is_positive() {
                std::cmp::Ordering::Less
            } else if cross.is_negative() {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
    });
    idx
}

/// Area of the convex polygon with the given vertices (any order) via a
/// fan around the interior point `o`.
fn polygon_area(verts: &[Vec<Q>], o: &[Q]) -> Q {
    let order = cyclic_order(verts, o);
    let k = order.len();
    if k < 3 {
        return Q::zero();
    }
    let mut twice = Q::zero();
    for i in 0..k {
        let a = &verts[order[i]];
        let b = &verts[order[(i + 1) % k]];
        let (ax, ay) = (&a[0] - &o[0], &a[1] - &o[1]);
        let (bx, by) = (&b[0] - &o[0], &b[1] - &o[1]);
        twice += (&ax * &by - &ay * &bx).abs();
    }
    twice / Q::from_integer(BigInt::from(2))
}

/// `|det(a − o, b − o, c − o)| / 6`.
fn tetra_volume(o: &[Q], a: &[Q], b: &[Q], c: &[Q]) -> Q {
    let d = |p: &[Q], i: usize| &p[i] - &o[i];
    let (a0, a1, a2) = (d(a, 0), d(a, 1), d(a, 2));
    let (b0, b1, b2) = (d(b, 0), d(b, 1), d(b, 2));
    let (c0, c1, c2) = (d(c, 0), d(c, 1), d(c, 2));
    let det = &a0 * (&b1 * &c2 - &b2 * &c1) - &a1 * (&b0 * &c2 - &b2 * &c0)
        + &a2 * (&b0 * &c1 - &b1 * &c0);
    det.abs() / Q::from_integer(BigInt::from(6))
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
