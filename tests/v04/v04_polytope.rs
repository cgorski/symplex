//! symplex 0.4 — `symplex::polytope`: exact convex polyhedra from
//! half-spaces (vertices, volume, containment, cutting) and the bridge to
//! the certificate search.

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
