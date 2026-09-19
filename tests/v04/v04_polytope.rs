//! symplex 0.4 — `symplex::polytope`: exact convex polyhedra from
//! half-spaces (vertices, volume, containment, cutting) and the bridge to
//! the certificate search.

use num_traits::Signed;
use symplex::certificates::{PolyhedronOpts, prove_nonnegative_on_polyhedron};
use symplex::linprog::{q, qi};
use symplex::polytope::{HalfSpace, Polytope};
use symplex::prelude::*;

#[test]
fn a_decision_tree_cell_round_trips_through_exprs_and_certifies() {
    let ctx = Context::new();
    let (r, t) = (ctx.symbol("r"), ctx.symbol("t"));
    let vars = [r.clone(), t.clone()];
    // The unit box cut by t ≥ r and r + t ≤ 3/2.
    let cell = Polytope::from_exprs(
        &[
            r.clone(),
            1 - &r,
            t.clone(),
            1 - &t,
            &t - &r,
            ctx.rational(3, 2) - &r - &t,
        ],
        &vars,
    )
    .unwrap();
    let verts = cell.vertices().unwrap();
    assert_eq!(verts.len(), 4);
    // Vertices (0,0), (3/4,3/4), (1/2,1), (0,1): area 7/16.
    assert_eq!(cell.volume().unwrap(), q(7, 16));
    // Redundancy: 1 − r ≥ 0 is implied (r ≤ t ≤ 1) — but it is tight at a
    // vertex (r = t = 1)?  No: r + t ≤ 3/2 excludes (1, 1); so it is dropped.
    let irr = cell.irredundant().unwrap();
    assert_eq!(irr.num_halfspaces(), 5);
    assert!(!irr.halfspaces().contains(&HalfSpace {
        coeffs: vec![qi(-1), qi(0)],
        constant: qi(1)
    }));
    // Every facet of the cell is a valid goal on the cell (Farkas, λ = 1).
    let hyps = irr.to_exprs(&vars).unwrap();
    for goal in &hyps {
        let out =
            prove_nonnegative_on_polyhedron(goal, &hyps, None, &PolyhedronOpts::default()).unwrap();
        assert!(out.is_proved(), "{goal}");
    }
    // A cut through the centroid splits the area exactly.
    let c = cell.vertex_centroid().unwrap().unwrap();
    let (a, b) = cell.split(&[qi(1), qi(-1)], &c[1] - &c[0]);
    assert_eq!(a.volume().unwrap() + b.volume().unwrap(), q(7, 16));
    assert!(a.contains(&c) && b.contains(&c));
}

#[test]
fn three_dimensional_cells() {
    let ctx = Context::new();
    let (r, t, u) = (ctx.symbol("r"), ctx.symbol("t"), ctx.symbol("u"));
    let vars = [r.clone(), t.clone(), u.clone()];
    // Unit cube with the corner r + t + u ≤ 5/2 cut off: volume 1 − (1/2)³/6.
    let cell = Polytope::from_exprs(
        &[
            r.clone(),
            1 - &r,
            t.clone(),
            1 - &t,
            u.clone(),
            1 - &u,
            ctx.rational(5, 2) - &r - &t - &u,
        ],
        &vars,
    )
    .unwrap();
    assert_eq!(cell.vertices().unwrap().len(), 10);
    assert_eq!(cell.volume().unwrap(), qi(1) - q(1, 48));
    assert!(cell.is_bounded().unwrap());
    let bb = cell.bounding_box().unwrap().unwrap();
    assert_eq!(bb, vec![(Some(qi(0)), Some(qi(1))); 3]);
    // Emptiness and a witness.
    let empty = cell.with_halfspace(&[qi(1), qi(1), qi(1)], qi(-3));
    assert!(empty.is_empty().unwrap());
    assert_eq!(empty.volume().unwrap(), qi(0));
    let pt = cell.any_point().unwrap().unwrap();
    assert!(cell.contains(&pt));
}

#[test]
fn volume_in_any_dimension() {
    // Unit hypercubes and simplices in dimensions 1..=5: 1 and 1/n!.
    let mut fact = qi(1);
    for n in 1..=5usize {
        fact *= qi(n as i64);
        let mut cube = Vec::new();
        let mut simplex = Vec::new();
        for i in 0..n {
            let mut e = vec![qi(0); n];
            e[i] = qi(1);
            cube.push((e.clone(), qi(0)));
            simplex.push((e.clone(), qi(0)));
            let mut f = vec![qi(0); n];
            f[i] = qi(-1);
            cube.push((f, qi(1)));
        }
        simplex.push((vec![qi(-1); n], qi(1)));
        assert_eq!(
            Polytope::from_rows(&cube).unwrap().volume().unwrap(),
            qi(1),
            "cube {n}"
        );
        assert_eq!(
            Polytope::from_rows(&simplex).unwrap().volume().unwrap(),
            qi(1) / &fact,
            "simplex {n}"
        );
    }
    // A 4-D cross-polytope |x₁| + … + |x₄| ≤ 1 has volume 2⁴/4! = 2/3.
    let mut rows = Vec::new();
    for signs in 0..16u32 {
        let coeffs: Vec<_> = (0..4)
            .map(|i| if signs & (1 << i) == 0 { qi(-1) } else { qi(1) })
            .collect();
        rows.push((coeffs, qi(1)));
    }
    assert_eq!(
        Polytope::from_rows(&rows).unwrap().volume().unwrap(),
        q(2, 3)
    );
    // Duplicate and scaled copies of a facet do not double count.
    let tri = Polytope::from_rows(&[
        (vec![qi(1), qi(0)], qi(0)),
        (vec![qi(0), qi(1)], qi(0)),
        (vec![qi(-1), qi(-1)], qi(1)),
        (vec![qi(-2), qi(-2)], qi(2)),
        (vec![qi(3), qi(0)], qi(0)),
    ])
    .unwrap();
    assert_eq!(tri.volume().unwrap(), q(1, 2));
}

#[test]
fn parametric_polytope_instantiates_exactly_and_caches() {
    use symplex::polytope::ParametricPolytope;
    let ctx = Context::new();
    let (j, r, t) = (ctx.symbol("j"), ctx.symbol("r"), ctx.symbol("t"));
    let hyps = [
        r.clone(),
        ctx.rational(1, 2) - &r,
        t.clone(),
        1 - &t,
        (&j * 2 + 1) * &t - &j * &r - 1,
    ];
    let mut cell = ParametricPolytope::new(&hyps, &[r.clone(), t.clone()], &j).unwrap();
    assert_eq!(cell.vars(), &[r.clone(), t.clone()]);
    assert_eq!(cell.param(), &j);
    // At j = 2: area ∫₀^{1/2} (1 − (2r + 1)/5) dr = 7/20; at j = 0: t ≥ 1 → segment, area 0.
    assert_eq!(cell.volume_at(&qi(2)).unwrap(), q(7, 20));
    assert_eq!(cell.volume_at(&qi(0)).unwrap(), qi(0));
    assert!(!cell.is_empty_at(&qi(0)).unwrap());
    assert_eq!(cell.vertices_at(&qi(2)).unwrap().len(), 4);
    assert!(cell.contains_at(&qi(2), &[q(1, 4), q(1, 2)]).unwrap());
    assert!(!cell.contains_at(&qi(2), &[q(1, 4), q(1, 4)]).unwrap());
    // The cached instantiation equals a fresh one.
    let fresh = cell.at(&qi(3)).unwrap();
    assert_eq!(cell.polytope_at(&qi(3)).unwrap(), &fresh);
    cell.clear_cache();
    assert_eq!(
        cell.volume_at(&q(5, 2)).unwrap(),
        cell.at(&q(5, 2)).unwrap().volume().unwrap()
    );
    // The polytope at a sample feeds the certificate search unchanged.
    let hyps_at_2 = cell
        .at(&qi(2))
        .unwrap()
        .to_exprs(&[r.clone(), t.clone()])
        .unwrap();
    for g in &hyps_at_2 {
        assert!(
            prove_nonnegative_on_polyhedron(g, &hyps_at_2, None, &PolyhedronOpts::default())
                .unwrap()
                .is_proved()
        );
    }
    // Errors: quadratic in the variables, symbolic coefficient, parameter among the variables.
    assert!(ParametricPolytope::new(&[&r * &t], &[r.clone(), t.clone()], &j).is_err());
    assert!(
        ParametricPolytope::new(&[&r * ctx.symbol("a")], std::slice::from_ref(&r), &j).is_err()
    );
    assert!(
        ParametricPolytope::new(std::slice::from_ref(&r), &[r.clone(), j.clone()], &j).is_err()
    );
    assert!(ParametricPolytope::new(&[], std::slice::from_ref(&r), &j).is_err());
}

/// The 0.6.1 `ParametricPolytope::at` instantiates through exact
/// `MultiPoly` arithmetic instead of the expression arena.  It must agree,
/// half-space by half-space, with the symbolic route (`subs` + `expand`,
/// then reading the affine coefficients) on random families with rational
/// parameter values — including negative and fractional ones.
#[test]
fn parametric_polytope_exact_route_matches_the_symbolic_route() {
    use symplex::polytope::ParametricPolytope;
    let ctx = Context::new();
    let (j, r, t, u) = (
        ctx.symbol("j"),
        ctx.symbol("r"),
        ctx.symbol("t"),
        ctx.symbol("u"),
    );
    let vars = [r.clone(), t.clone(), u.clone()];
    let mut lcg = Lcg(0x5EED_0611);
    let coef = |lcg: &mut Lcg| -> Ex {
        // A polynomial in j of degree ≤ 3 with small integer/rational coefficients.
        let mut acc = ctx.int(0);
        for d in 0..=3 {
            let n = lcg.range(-6, 7);
            let den = lcg.range(1, 4);
            acc += ctx.rational(n, den) * j.powi(d);
        }
        acc
    };
    for _ in 0..25 {
        let mut hyps = Vec::new();
        for _ in 0..6 {
            let h =
                coef(&mut lcg) * &r + coef(&mut lcg) * &t + coef(&mut lcg) * &u + coef(&mut lcg);
            hyps.push(h);
        }
        let family = ParametricPolytope::new(&hyps, &vars, &j).unwrap();
        for value in [qi(0), qi(3), qi(-2), q(7, 3), q(-5, 4), qi(1000)] {
            let fast = family.at(&value).unwrap();
            // Symbolic route: substitute, expand, read coefficients.
            let jv = ctx.from_ratio(value.clone());
            let slow = Polytope::from_exprs(
                &hyps
                    .iter()
                    .map(|h| h.subs(&j, &jv).expand())
                    .collect::<Vec<_>>(),
                &vars,
            )
            .unwrap();
            assert_eq!(fast, slow, "j = {value}");
        }
    }
}

/// `is_full_dimensional` is one LP, not a volume: it agrees with
/// `volume() > 0` on bounded cells, is defined for unbounded polyhedra, and
/// treats trivial rows (`0·x + b ≥ 0`) correctly.  `interior_point` has
/// strictly positive slack everywhere or is `None`.
#[test]
fn full_dimensionality_and_interior_points() {
    // Bounded: the unit cube and its slices.
    let cube = Polytope::from_rows(&[
        (vec![qi(1), qi(0), qi(0)], qi(0)),
        (vec![qi(-1), qi(0), qi(0)], qi(1)),
        (vec![qi(0), qi(1), qi(0)], qi(0)),
        (vec![qi(0), qi(-1), qi(0)], qi(1)),
        (vec![qi(0), qi(0), qi(1)], qi(0)),
        (vec![qi(0), qi(0), qi(-1)], qi(1)),
    ])
    .unwrap();
    assert!(cube.is_full_dimensional().unwrap());
    let p = cube.interior_point().unwrap().unwrap();
    assert!(cube.halfspaces().iter().all(|h| h.value(&p).is_positive()));
    // A face (z = 0): non-empty, positive area, zero volume, not full-dimensional.
    let face = cube.with_halfspace(&[qi(0), qi(0), qi(-1)], qi(0));
    assert!(!face.is_empty().unwrap());
    assert_eq!(face.volume().unwrap(), qi(0));
    assert!(!face.is_full_dimensional().unwrap());
    assert_eq!(face.interior_point().unwrap(), None);
    // Empty.
    let empty = cube.with_halfspace(&[qi(1), qi(0), qi(0)], qi(-2));
    assert!(!empty.is_full_dimensional().unwrap());
    // Thin but full-dimensional: 0 ≤ x ≤ 1/1000 in the cube.  volume > 0 agrees.
    let thin = cube.with_halfspace(&[qi(-1), qi(0), qi(0)], q(1, 1000));
    assert!(thin.is_full_dimensional().unwrap());
    assert!(thin.volume().unwrap().is_positive());
    // Random cells: agreement with volume > 0 on bounded polytopes.
    let mut lcg = Lcg(0xF00D);
    for _ in 0..40 {
        let mut poly = cube.clone();
        for _ in 0..3 {
            let a: Vec<Q> = (0..3).map(|_| qi(lcg.range(-3, 4))).collect();
            poly = poly.with_halfspace(&a, q(lcg.range(-4, 5), 2));
        }
        assert_eq!(
            poly.is_full_dimensional().unwrap(),
            poly.volume().unwrap().is_positive(),
            "{poly:?}"
        );
    }
    // Unbounded: a wedge is full-dimensional; a ray is not.
    let wedge =
        Polytope::from_rows(&[(vec![qi(1), qi(0)], qi(0)), (vec![qi(-1), qi(1)], qi(0))]).unwrap();
    assert!(wedge.is_full_dimensional().unwrap());
    let ray = wedge.with_halfspace(&[qi(1), qi(-1)], qi(0));
    assert!(!ray.is_full_dimensional().unwrap());
    // Trivial rows: `0 ≥ 0` never counts as a tight constraint; `-1 ≥ 0` is empty.
    let with_trivial = cube.with_halfspace(&[qi(0), qi(0), qi(0)], qi(0));
    assert!(with_trivial.is_full_dimensional().unwrap());
    let contradiction = cube.with_halfspace(&[qi(0), qi(0), qi(0)], qi(-1));
    assert!(!contradiction.is_full_dimensional().unwrap());
    assert_eq!(contradiction.interior_point().unwrap(), None);
    // Hyperplane keys identify a cut with its flip and its rescalings.
    let h = HalfSpace {
        coeffs: vec![qi(0), qi(-3), qi(6)],
        constant: qi(9),
    };
    let n = h.normalized();
    assert_eq!(
        (n.coeffs.clone(), n.constant.clone()),
        (vec![qi(0), qi(1), qi(-2)], qi(-3))
    );
    assert!(h.same_hyperplane(&h.flipped()));
    assert!(h.same_hyperplane(&HalfSpace {
        coeffs: vec![qi(0), q(1, 2), qi(-1)],
        constant: q(-3, 2),
    }));
    assert!(!h.same_hyperplane(&HalfSpace {
        coeffs: vec![qi(0), qi(1), qi(-2)],
        constant: qi(0),
    }));
    assert!(
        HalfSpace {
            coeffs: vec![qi(0)],
            constant: qi(1)
        }
        .is_trivial()
    );
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// Uniform in `lo..hi`.
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo) as u64) as i64
    }
}

#[test]
fn unbounded_polyhedra_are_reported_not_mis_measured() {
    let wedge =
        Polytope::from_rows(&[(vec![qi(1), qi(0)], qi(0)), (vec![qi(-1), qi(1)], qi(0))]).unwrap();
    assert!(!wedge.is_bounded().unwrap());
    assert!(wedge.volume().is_err());
    assert_eq!(wedge.vertices().unwrap(), vec![vec![qi(0), qi(0)]]);
    let bb = wedge.bounding_box().unwrap().unwrap();
    assert_eq!(bb, vec![(Some(qi(0)), None), (Some(qi(0)), None)]);
}
