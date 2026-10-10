//! Zero-dimensional ideals: the number of distinct solutions and a
//! rational univariate representation, by linear algebra in the quotient
//! algebra `A = ℚ[x₁, …, xₙ]/I`.
//!
//! For a zero-dimensional ideal with reduced Gröbner basis `G`, the normal
//! forms modulo `G` are coordinates in `A` (a finite-dimensional ℚ-vector
//! space spanned by the standard monomials).  The minimal polynomial of an
//! element `f ∈ A` is the first linear dependence among `1, f, f², …`
//! (normal forms of successive multiplications by `f`).
//!
//! * **Distinct solutions.**  `m_i`, the minimal polynomial of `xᵢ`,
//!   generates `I ∩ ℚ[xᵢ]`.  By Seidenberg's lemma (characteristic 0) the
//!   radical is `√I = I + (sqf(m₁(x₁)), …, sqf(mₙ(xₙ)))`, and the number of
//!   distinct complex solutions is `dim ℚ[x]/√I` — the number of standard
//!   monomials of a Gröbner basis of `√I` (Cox, Little and O'Shea, *Using
//!   Algebraic Geometry*, chapter 2, §2; Becker and Weispfenning,
//!   *Gröbner Bases*, §8.2).
//! * **Rational univariate representation.**  A linear form `ℓ` separates
//!   the solutions iff its minimal polynomial `μ` in `ℚ[x]/√I` has degree
//!   equal to the number `N` of distinct solutions; then `1, ℓ, …, ℓ^{N−1}`
//!   is a basis of `ℚ[x]/√I`, every coordinate is `xᵢ = gᵢ(ℓ)` with
//!   `deg gᵢ < N`, and the solutions are `(g₁(t), …, gₙ(t))` for the `N`
//!   roots `t` of `μ` (Rouillier, *Solving zero-dimensional systems through
//!   the rational univariate representation*, AAECC 9 (1999); the
//!   representation here uses the shape-lemma form over `√I`).

use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

use super::dense::Poly;
use super::groebner::{groebner_basis, standard_monomial_count};
use super::multipoly::{GrevLex, MultiPoly};

type Q = Ratio<BigInt>;
type Vector = BTreeMap<Vec<u32>, Q>;

/// Separating linear forms tried before giving up.
const MAX_SEPARATING_TRIES: usize = 24;

/// Largest quotient dimension handled (beyond, the dense linear algebra
/// is not attempted).
const MAX_QUOTIENT_DIMENSION: usize = 512;

/// Incremental Gaussian elimination of normal-form vectors, each row
/// remembering its combination of the generating powers.
struct Echelon {
    /// `(pivot, row, combination)`, pivots descending; the pivot is the
    /// largest monomial of its row.
    rows: Vec<(Vec<u32>, Vector, Vec<Q>)>,
}

impl Echelon {
    fn new() -> Self {
        Echelon { rows: Vec::new() }
    }

    /// Reduce `v` (with combination `comb`) by the rows.
    fn reduce(&self, mut v: Vector, mut comb: Vec<Q>) -> (Vector, Vec<Q>) {
        for (pivot, row, rc) in &self.rows {
            let Some(c) = v.get(pivot).cloned() else {
                continue;
            };
            let factor = &c / &row[pivot];
            for (m, x) in row {
                let e = v.entry(m.clone()).or_insert_with(Q::zero);
                *e -= &factor * x;
                if e.is_zero() {
                    v.remove(m);
                }
            }
            if comb.len() < rc.len() {
                comb.resize(rc.len(), Q::zero());
            }
            for (j, x) in rc.iter().enumerate() {
                comb[j] -= &factor * x;
            }
        }
        (v, comb)
    }

    fn insert(&mut self, v: Vector, comb: Vec<Q>) {
        let Some(pivot) = v.keys().next_back().cloned() else {
            return;
        };
        let pos = self
            .rows
            .iter()
            .position(|(p, _, _)| *p < pivot)
            .unwrap_or(self.rows.len());
        self.rows.insert(pos, (pivot, v, comb));
    }
}

fn to_vector(p: &MultiPoly<GrevLex>) -> Vector {
    p.terms().map(|(e, c)| (e.to_vec(), c.clone())).collect()
}

/// The quotient algebra of a zero-dimensional ideal given by a Gröbner
/// basis in grevlex order.
pub(crate) struct Quotient {
    gb: Vec<MultiPoly<GrevLex>>,
    /// `dim ℚ[x]/I` (solutions counted with multiplicity).
    pub(crate) dimension: usize,
}

/// The minimal polynomial of an element of a quotient algebra and the
/// echelonized powers that express other elements in it.
struct PowerBasis {
    /// Monic, ascending coefficients.
    minpoly: Vec<Q>,
    echelon: Echelon,
}

impl Quotient {
    /// `None` when the basis is empty, not zero-dimensional or too large.
    pub(crate) fn new(gb: &[MultiPoly<GrevLex>]) -> Option<Quotient> {
        let nv = gb.first()?.num_vars();
        if !super::groebner::is_zero_dimensional(gb) {
            return None;
        }
        let dimension = standard_monomial_count(gb, nv, MAX_QUOTIENT_DIMENSION)?;
        if dimension == 0 {
            return None;
        }
        Some(Quotient {
            gb: gb.to_vec(),
            dimension,
        })
    }

    fn normal_form(&self, p: &MultiPoly<GrevLex>) -> MultiPoly<GrevLex> {
        let refs: Vec<&MultiPoly<GrevLex>> = self.gb.iter().collect();
        p.reduce(&refs)
    }

    /// The minimal polynomial of `f` in the algebra.
    fn power_basis(&self, f: &MultiPoly<GrevLex>) -> Option<PowerBasis> {
        let nv = f.num_vars();
        let f = self.normal_form(f);
        let mut cur = MultiPoly::from_int(nv, 1);
        let mut echelon = Echelon::new();
        for k in 0..=self.dimension {
            let mut comb = vec![Q::zero(); k + 1];
            comb[k] = Q::one();
            let (v, comb) = echelon.reduce(to_vector(&cur), comb);
            if v.is_empty() {
                return Some(PowerBasis {
                    minpoly: comb,
                    echelon,
                });
            }
            echelon.insert(v, comb);
            cur = self.normal_form(&cur.mul(&f));
        }
        None
    }
}

/// Square-free part of a univariate polynomial given by ascending
/// coefficients, as a polynomial in variable `v` of `nv`.
fn squarefree_in(coeffs: &[Q], v: usize, nv: usize) -> (MultiPoly<GrevLex>, bool) {
    let p = Poly::from_coeffs(coeffs.to_vec());
    let sf = if p.is_squarefree() == Some(true) {
        p
    } else {
        p.square_free_part()
    };
    let squarefree = sf.degree() == p_degree(coeffs);
    let mut terms = Vec::new();
    for (k, c) in sf.coeffs().iter().enumerate() {
        if !c.is_zero() {
            let mut e = vec![0u32; nv];
            e[v] = k as u32;
            terms.push((e, c.clone()));
        }
    }
    (
        MultiPoly::from_terms(nv, terms).unwrap_or_else(|| MultiPoly::zero(nv)),
        squarefree,
    )
}

fn p_degree(coeffs: &[Q]) -> Option<usize> {
    coeffs.iter().rposition(|c| !c.is_zero())
}

/// The radical of a zero-dimensional ideal and its number of distinct
/// solutions.
pub(crate) struct Radical {
    /// The quotient algebra `ℚ[x]/√I`.
    pub(crate) quotient: Quotient,
}

impl Radical {
    /// `None` when the computation is out of range (see [`Quotient::new`]).
    pub(crate) fn new(gb: &[MultiPoly<GrevLex>]) -> Option<Radical> {
        let q = Quotient::new(gb)?;
        let nv = gb[0].num_vars();
        let mut extra = Vec::new();
        let mut radical = true;
        for v in 0..nv {
            let pb = q.power_basis(&MultiPoly::var(nv, v))?;
            let (sf, is_sf) = squarefree_in(&pb.minpoly, v, nv);
            if !is_sf {
                radical = false;
            }
            extra.push(sf);
        }
        if radical {
            return Some(Radical { quotient: q });
        }
        let mut gens = gb.to_vec();
        gens.extend(extra);
        let rgb = groebner_basis(&gens);
        Some(Radical {
            quotient: Quotient::new(&rgb)?,
        })
    }

    /// Number of distinct complex solutions.
    pub(crate) fn distinct(&self) -> usize {
        self.quotient.dimension
    }

    /// A rational univariate representation: `(μ, [g₁, …, gₙ])` with the
    /// solutions `(g₁(t), …, gₙ(t))` for the roots `t` of the square-free
    /// `μ` (ascending coefficients; `deg μ` = number of solutions).  The
    /// separating form is `xⱼ` for some `j` (the last variables first)
    /// when one separates, else a small integer combination; the third
    /// component is that `j` (`t = xⱼ`), or `None` for a combination.
    /// `None` if no separating form was found.
    pub(crate) fn univariate_representation(&self) -> Option<(Poly, Vec<Poly>, Option<usize>)> {
        let q = &self.quotient;
        let nv = q.gb.first()?.num_vars();
        let n = q.dimension;
        let mut forms: Vec<Vec<i64>> = (0..nv)
            .rev()
            .map(|j| {
                let mut c = vec![0i64; nv];
                c[j] = 1;
                c
            })
            .collect();
        let mut rng = crate::base::rng::SplitMix64::new(0x5EB0_0F0F);
        while forms.len() < MAX_SEPARATING_TRIES {
            let r = 1 + (forms.len() / 6) as u64;
            let mut c: Vec<i64> = (0..nv)
                .map(|_| (rng.next_u64() % (2 * r + 1)) as i64 - r as i64)
                .collect();
            c[nv - 1] = 1;
            forms.push(c);
        }
        for c in forms {
            let mut ell = MultiPoly::zero(nv);
            for (j, &cj) in c.iter().enumerate() {
                if cj != 0 {
                    ell = ell.add(&MultiPoly::var(nv, j).scale(&Q::from_integer(BigInt::from(cj))));
                }
            }
            let pb = q.power_basis(&ell)?;
            if pb.minpoly.len() != n + 1 {
                continue;
            }
            // Express every coordinate in powers of ℓ.
            let mut coords = Vec::with_capacity(nv);
            for v in 0..nv {
                let nf = q.normal_form(&MultiPoly::var(nv, v));
                let (rest, comb) = pb.echelon.reduce(to_vector(&nf), vec![Q::zero(); n]);
                if !rest.is_empty() {
                    return None;
                }
                coords.push(Poly::from_coeffs(comb.into_iter().map(|x| -x).collect()));
            }
            let single = (c.iter().filter(|&&cj| cj != 0).count() == 1)
                .then(|| c.iter().position(|&cj| cj == 1))
                .flatten();
            return Some((Poly::from_coeffs(pb.minpoly), coords, single));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64) -> Q {
        Q::from_integer(BigInt::from(n))
    }

    /// `(x² − 1)² = 0, y = x`: 4 solutions with multiplicity, 2 distinct.
    #[test]
    fn distinct_solutions_of_a_non_radical_ideal() {
        let x: MultiPoly<GrevLex> = MultiPoly::var(2, 0);
        let y: MultiPoly<GrevLex> = MultiPoly::var(2, 1);
        let f = x.mul(&x).sub(&MultiPoly::from_int(2, 1));
        let gb = groebner_basis(&[f.mul(&f), y.sub(&x)]);
        assert_eq!(Quotient::new(&gb).unwrap().dimension, 4);
        assert_eq!(Radical::new(&gb).unwrap().distinct(), 2);
    }

    /// `x² = 2, y² = 3`: no coordinate separates the 4 solutions; a
    /// combination does, with the minimal polynomial of `±√2 ± √3` (or of
    /// another separating form) of degree 4, and the coordinates satisfy
    /// the equations at every root.
    #[test]
    fn univariate_representation_of_two_square_roots() {
        let x: MultiPoly<GrevLex> = MultiPoly::var(2, 0);
        let y: MultiPoly<GrevLex> = MultiPoly::var(2, 1);
        let gb = groebner_basis(&[
            x.mul(&x).sub(&MultiPoly::from_int(2, 2)),
            y.mul(&y).sub(&MultiPoly::from_int(2, 3)),
        ]);
        let rad = Radical::new(&gb).unwrap();
        assert_eq!(rad.distinct(), 4);
        let (mu, coords, single) = rad.univariate_representation().unwrap();
        assert_eq!(mu.degree(), Some(4));
        assert_eq!(single, None);
        for (g, c) in coords.iter().zip([q(2), q(3)]) {
            let sq = &(g * g) - &Poly::constant(c);
            assert!(sq.rem(&mu).is_zero(), "{g:?}");
        }
    }
}
