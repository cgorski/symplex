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
fn unbounded_polyhedra_are_reported_not_mis_measured() {
    let wedge =
        Polytope::from_rows(&[(vec![qi(1), qi(0)], qi(0)), (vec![qi(-1), qi(1)], qi(0))]).unwrap();
    assert!(!wedge.is_bounded().unwrap());
    assert!(wedge.volume().is_err());
    assert_eq!(wedge.vertices().unwrap(), vec![vec![qi(0), qi(0)]]);
    let bb = wedge.bounding_box().unwrap().unwrap();
    assert_eq!(bb, vec![(Some(qi(0)), None), (Some(qi(0)), None)]);
}
