//! symplex 0.4 — `symplex::polytope`: exact convex polyhedra from
//! half-spaces (vertices, volume, containment, cutting) and the bridge to
//! the certificate search.

use num_traits::{Signed, Zero};
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

// ───────────────────────────────────────────────────────────────────────────
// `Polytope::clip`: incremental cutting of the cached vertex structure
// ───────────────────────────────────────────────────────────────────────────

fn sorted_dedup(pts: &[Vec<Q>]) -> Vec<Vec<Q>> {
    let mut v = pts.to_vec();
    v.sort();
    v.dedup();
    v
}

/// The unit cube `[0, 1]ⁿ`.
fn unit_cube(n: usize) -> Polytope {
    let mut rows = Vec::new();
    for i in 0..n {
        let mut e = vec![qi(0); n];
        e[i] = qi(1);
        rows.push((e.clone(), qi(0)));
        let mut f = vec![qi(0); n];
        f[i] = qi(-1);
        rows.push((f, qi(1)));
    }
    Polytope::from_rows(&rows).unwrap()
}

/// A random half-space with small integer coefficients (never trivial) and
/// a half-integer constant.
fn random_halfspace(lcg: &mut Lcg, n: usize) -> HalfSpace {
    loop {
        let coeffs: Vec<Q> = (0..n).map(|_| qi(lcg.range(-3, 4))).collect();
        if coeffs.iter().any(|c| !c.is_zero()) {
            return HalfSpace {
                coeffs,
                constant: q(lcg.range(-6, 7), 2),
            };
        }
    }
}

/// Counters for the non-vacuity checks of the randomized clip tests.
#[derive(Default, Debug)]
struct ClipStats {
    clips: usize,
    /// Clips where the hyperplane crosses the interior (both pieces full-dimensional).
    proper: usize,
    /// Clips with an original vertex on the hyperplane.
    through_vertex: usize,
    /// Clips leaving one side without vertices.
    one_sided: usize,
    /// Clips whose parent has a degenerate vertex (more than `n` tight half-spaces).
    degenerate_parent: usize,
}

/// The correctness reference: `clip(h).pos` / `.neg` as sets equal the
/// full enumerations of `P ∩ {h ≥ 0}` / `P ∩ {h ≤ 0}`; `on` is their
/// intersection and lies on the hyperplane; the pre-filled pieces report
/// the same vertices, volume and dimension as the enumerated ones; and a
/// second clip of a pre-filled piece (which exercises the tight sets the
/// clip wrote into the cache) again matches the enumeration.
fn check_clip(cell: &Polytope, h: &HalfSpace, next: &HalfSpace, stats: &mut ClipStats) {
    let n = cell.dim();
    let clip = cell.clip(h).unwrap();
    let pos_ref = cell.with_halfspace(&h.coeffs, h.constant.clone());
    let flipped = h.flipped();
    let neg_ref = cell.with_halfspace(&flipped.coeffs, flipped.constant.clone());
    let (pos_set, neg_set) = (sorted_dedup(&clip.pos), sorted_dedup(&clip.neg));
    assert_eq!(
        pos_set,
        sorted_dedup(&pos_ref.vertices().unwrap()),
        "pos of {cell:?} by {h:?}"
    );
    assert_eq!(
        neg_set,
        sorted_dedup(&neg_ref.vertices().unwrap()),
        "neg of {cell:?} by {h:?}"
    );
    // No duplicates are produced: the clip is exact, not "sorted afterwards".
    assert_eq!(clip.pos.len(), pos_set.len(), "duplicate in pos: {clip:?}");
    assert_eq!(clip.neg.len(), neg_set.len(), "duplicate in neg: {clip:?}");
    let on_set = sorted_dedup(&clip.on);
    let both: Vec<Vec<Q>> = pos_set
        .iter()
        .filter(|v| neg_set.contains(v))
        .cloned()
        .collect();
    assert_eq!(on_set, both, "on ≠ pos ∩ neg for {cell:?} by {h:?}");
    assert!(clip.on.iter().all(|v| h.is_tight(v)));
    assert_eq!(clip.on.len(), on_set.len());
    // The pre-filled pieces.
    let pos = clip.pos_polytope(cell, h).unwrap();
    let neg = clip.neg_polytope(cell, h).unwrap();
    assert_eq!(pos, pos_ref);
    assert_eq!(neg, neg_ref);
    assert_eq!(sorted_dedup(&pos.vertices().unwrap()), pos_set);
    assert_eq!(sorted_dedup(&neg.vertices().unwrap()), neg_set);
    assert_eq!(
        pos.volume().unwrap(),
        pos_ref.volume().unwrap(),
        "volume of {pos:?}"
    );
    assert_eq!(
        neg.volume().unwrap(),
        neg_ref.volume().unwrap(),
        "volume of {neg:?}"
    );
    assert_eq!(
        pos.volume().unwrap() + neg.volume().unwrap(),
        cell.volume().unwrap()
    );
    // Dimension from the vertices agrees with the LP (bounded cells).
    let (pf, nf) = (
        pos_ref.is_full_dimensional().unwrap(),
        neg_ref.is_full_dimensional().unwrap(),
    );
    assert_eq!(
        clip.pos_is_full_dimensional(),
        pf,
        "pos dim of {cell:?} by {h:?}"
    );
    assert_eq!(
        clip.neg_is_full_dimensional(),
        nf,
        "neg dim of {cell:?} by {h:?}"
    );
    assert_eq!(pos.is_full_dimensional_from_vertices().unwrap(), pf);
    assert_eq!(neg.is_full_dimensional_from_vertices().unwrap(), nf);
    // Tight sets written into the pieces are the real ones (index-for-index).
    for piece in [&pos, &neg] {
        for (v, tight) in piece.vertices_with_tight().unwrap() {
            let expected: Vec<usize> = piece
                .halfspaces()
                .iter()
                .enumerate()
                .filter(|(_, g)| !g.is_trivial() && g.is_tight(&v))
                .map(|(i, _)| i)
                .collect();
            assert_eq!(tight, expected, "tight set of {v:?} in {piece:?}");
            assert!(tight.len() >= n);
        }
    }
    // A second clip of the pre-filled piece matches its enumeration too.
    let second = pos.clip(next).unwrap();
    let second_ref = pos.with_halfspace(&next.coeffs, next.constant.clone());
    assert_eq!(
        sorted_dedup(&second.pos),
        sorted_dedup(&second_ref.vertices().unwrap()),
        "second clip of {pos:?} by {next:?}"
    );
    let second_neg = neg.clip(next).unwrap();
    let second_neg_ref = neg.with_halfspace(&next.flipped().coeffs, -&next.constant);
    assert_eq!(
        sorted_dedup(&second_neg.neg),
        sorted_dedup(&second_neg_ref.vertices().unwrap()),
        "second clip of {neg:?} by {next:?}"
    );
    // Statistics.
    stats.clips += 1;
    if pf && nf {
        stats.proper += 1;
    }
    if cell.vertices().unwrap().iter().any(|v| h.is_tight(v)) {
        stats.through_vertex += 1;
    }
    if clip.pos.is_empty() || clip.neg.is_empty() {
        stats.one_sided += 1;
    }
    if cell
        .vertices_with_tight()
        .unwrap()
        .iter()
        .any(|(_, t)| t.len() > n)
    {
        stats.degenerate_parent += 1;
    }
}

/// Random bounded cells (the unit cube with up to three random cuts) in
/// dimension `n`, cut by random planes, planes through a vertex, and
/// planes that miss the cell.
fn random_cells_clip(n: usize, cells: usize, cuts: usize, seed: u64) -> ClipStats {
    let mut lcg = Lcg(seed);
    let mut stats = ClipStats::default();
    for _ in 0..cells {
        let mut cell = unit_cube(n);
        for _ in 0..lcg.range(0, 4) {
            let g = random_halfspace(&mut lcg, n);
            cell = cell.with_halfspace(&g.coeffs, g.constant.clone());
        }
        let verts = cell.vertices().unwrap();
        let through = |h: &mut HalfSpace, p: &[Q]| {
            h.constant = -h
                .coeffs
                .iter()
                .zip(p)
                .fold(Q::zero(), |acc, (a, x)| acc + a * x);
        };
        for _ in 0..cuts {
            let mut h = random_halfspace(&mut lcg, n);
            let next = random_halfspace(&mut lcg, n);
            match lcg.range(0, 5) {
                // Through a random vertex of the cell.
                0 if !verts.is_empty() => {
                    let v = verts[lcg.range(0, verts.len() as i64) as usize].clone();
                    through(&mut h, &v);
                }
                // Missing the cell entirely.
                1 => h.constant = qi(100),
                // Through the vertex centroid (interior when the cell is full-dimensional).
                2 | 3 if !verts.is_empty() => {
                    let c = cell.vertex_centroid().unwrap().unwrap();
                    through(&mut h, &c);
                }
                // Through a random point of the cube's interior (quarter grid).
                4 => {
                    let p: Vec<Q> = (0..n).map(|_| q(lcg.range(1, 4), 4)).collect();
                    through(&mut h, &p);
                }
                _ => {}
            }
            check_clip(&cell, &h, &next, &mut stats);
        }
    }
    stats
}

#[test]
fn clip_matches_enumeration_on_random_2d_cells() {
    let s = random_cells_clip(2, 30, 5, 0xC11B_0002);
    assert!(
        s.proper >= 40 && s.through_vertex >= 15 && s.one_sided >= 15 && s.degenerate_parent >= 10,
        "{s:?}"
    );
}

#[test]
fn clip_matches_enumeration_on_random_3d_cells() {
    let s = random_cells_clip(3, 12, 4, 0xC11B_0003);
    assert!(
        s.proper >= 12 && s.through_vertex >= 5 && s.one_sided >= 5 && s.degenerate_parent >= 4,
        "{s:?}"
    );
}

#[test]
fn clip_matches_enumeration_on_random_4d_cells() {
    let s = random_cells_clip(4, 6, 3, 0xC11B_0004);
    assert!(
        s.proper >= 5 && s.through_vertex >= 2 && s.one_sided >= 2,
        "{s:?}"
    );
}

/// Degenerate parents: a cube cut by planes through its vertices (vertices
/// with four tight planes), a pyramid whose apex has four tight planes, and
/// the 4-D cross-polytope (every vertex has eight).  Cuts through vertices,
/// along edges and across.
#[test]
fn clip_handles_degenerate_vertices_exactly() {
    let mut stats = ClipStats::default();
    let mut lcg = Lcg(0xDE6E_0001);
    // Cube with the corner planes x + y + z ≤ 2 (through three vertices) and
    // x + y ≥ z (through four vertices), then x − y ≤ 0 (through two).
    let cube = unit_cube(3)
        .with_halfspace(&[qi(-1), qi(-1), qi(-1)], qi(2))
        .with_halfspace(&[qi(1), qi(1), qi(-1)], qi(0));
    assert!(
        cube.vertices_with_tight()
            .unwrap()
            .iter()
            .any(|(_, t)| t.len() > 3)
    );
    // Square pyramid |x|, |y| ≤ 1 − z, z ≥ 0: apex (0, 0, 1) with four tight planes.
    let pyramid = Polytope::from_rows(&[
        (vec![qi(1), qi(0), qi(-1)], qi(1)),  // x ≥ −(1 − z)
        (vec![qi(-1), qi(0), qi(-1)], qi(1)), // x ≤ 1 − z
        (vec![qi(0), qi(1), qi(-1)], qi(1)),
        (vec![qi(0), qi(-1), qi(-1)], qi(1)),
        (vec![qi(0), qi(0), qi(1)], qi(0)),
    ])
    .unwrap();
    let apex = pyramid.vertices_with_tight().unwrap();
    assert_eq!(apex.len(), 5);
    assert_eq!(apex.iter().filter(|(_, t)| t.len() == 4).count(), 1);
    let mut cross = Vec::new();
    for signs in 0..16u32 {
        let coeffs: Vec<_> = (0..4)
            .map(|i| if signs & (1 << i) == 0 { qi(-1) } else { qi(1) })
            .collect();
        cross.push((coeffs, qi(1)));
    }
    let cross = Polytope::from_rows(&cross).unwrap();
    assert!(
        cross
            .vertices_with_tight()
            .unwrap()
            .iter()
            .all(|(_, t)| t.len() == 8)
    );
    let hs = |c: &[i64], k: Q| HalfSpace {
        coeffs: c.iter().map(|&a| qi(a)).collect(),
        constant: k,
    };
    let n3 = hs(&[1, 1, 1], qi(-1));
    // Cube: through the degenerate vertex (1,1,0) along edges; across; through two vertices.
    for h in [
        hs(&[1, -1, 0], qi(0)),
        hs(&[0, 0, 1], q(-1, 2)),
        hs(&[1, 1, 0], qi(-2)),
        hs(&[1, 0, 0], qi(-1)),
        hs(&[-1, -1, -1], qi(2)),
        hs(&[2, -1, 1], q(-1, 2)),
    ] {
        check_clip(&cube, &h, &n3, &mut stats);
    }
    // Pyramid: through the apex, through the apex and a base vertex, across, along the base.
    for h in [
        hs(&[1, 0, 0], qi(0)),
        hs(&[1, 1, 0], qi(0)),
        hs(&[0, 0, -1], q(1, 2)),
        hs(&[0, 0, 1], qi(0)),
        hs(&[1, 2, 3], q(-1, 2)),
        hs(&[1, 0, 1], qi(-1)),
    ] {
        check_clip(&pyramid, &h, &n3, &mut stats);
    }
    // Cross-polytope: coordinate planes (through six vertices), across, missing.
    let n4 = hs(&[1, 2, 0, -1], q(1, 3));
    for h in [
        hs(&[1, 0, 0, 0], qi(0)),
        hs(&[1, 1, 0, 0], qi(0)),
        hs(&[1, 1, 1, 1], q(-1, 2)),
        hs(&[1, 0, 0, 0], qi(-1)),
        hs(&[3, -1, 2, 1], q(1, 4)),
        hs(&[1, 1, 1, 1], qi(5)),
    ] {
        check_clip(&cross, &h, &n4, &mut stats);
    }
    // Random cuts of the three degenerate parents.
    for _ in 0..6 {
        let h = random_halfspace(&mut lcg, 3);
        check_clip(&cube, &h, &random_halfspace(&mut lcg, 3), &mut stats);
        let h = random_halfspace(&mut lcg, 3);
        check_clip(&pyramid, &h, &random_halfspace(&mut lcg, 3), &mut stats);
    }
    for _ in 0..3 {
        let h = random_halfspace(&mut lcg, 4);
        check_clip(&cross, &h, &random_halfspace(&mut lcg, 4), &mut stats);
    }
    assert_eq!(stats.degenerate_parent, stats.clips, "{stats:?}");
    assert!(
        stats.through_vertex >= 12 && stats.proper >= 12 && stats.one_sided >= 2,
        "{stats:?}"
    );
}

/// Lower-dimensional parents and duplicate descriptions: a face of the cube
/// (every vertex has both `z ≥ 0` and `z ≤ 0` tight), a segment given as a
/// diagonal of the square, and a triangle with a duplicated, rescaled facet.
#[test]
fn clip_on_lower_dimensional_and_duplicate_descriptions() {
    let mut stats = ClipStats::default();
    let face = unit_cube(3).with_halfspace(&[qi(0), qi(0), qi(-1)], qi(0));
    let diagonal = unit_cube(2)
        .with_halfspace(&[qi(1), qi(-1)], qi(0))
        .with_halfspace(&[qi(-1), qi(1)], qi(0));
    let tri = Polytope::from_rows(&[
        (vec![qi(1), qi(0)], qi(0)),
        (vec![qi(0), qi(1)], qi(0)),
        (vec![qi(-1), qi(-1)], qi(1)),
        (vec![qi(-2), qi(-2)], qi(2)),
        (vec![qi(3), qi(0)], qi(0)),
    ])
    .unwrap();
    let h3 = HalfSpace {
        coeffs: vec![qi(1), qi(1), qi(0)],
        constant: qi(-1),
    };
    let h3b = HalfSpace {
        coeffs: vec![qi(2), qi(-1), qi(1)],
        constant: q(-1, 3),
    };
    check_clip(&face, &h3, &h3b, &mut stats);
    check_clip(&face, &h3b, &h3, &mut stats);
    let h2 = HalfSpace {
        coeffs: vec![qi(1), qi(1)],
        constant: qi(-1),
    };
    let h2b = HalfSpace {
        coeffs: vec![qi(-1), qi(0)],
        constant: q(1, 3),
    };
    check_clip(&diagonal, &h2, &h2b, &mut stats);
    check_clip(&diagonal, &h2b, &h2, &mut stats);
    check_clip(&tri, &h2, &h2b, &mut stats);
    check_clip(&tri, &h2b, &h2, &mut stats);
    // The diagonal cut at its midpoint: one crossing, the two halves.
    let clip = diagonal.clip(&h2).unwrap();
    assert_eq!(clip.on, vec![vec![q(1, 2), q(1, 2)]]);
    assert_eq!((clip.pos.len(), clip.neg.len()), (2, 2));
    assert!(!clip.pos_is_full_dimensional() && !clip.neg_is_full_dimensional());
    assert_eq!(stats.clips, 6);
}

/// Trivial cuts (`a = 0`), cuts along an edge, and the error paths.
#[test]
fn clip_trivial_cuts_edges_and_errors() {
    let square = unit_cube(2);
    let all: Vec<Vec<Q>> = sorted_dedup(&square.vertices().unwrap());
    // 0·x + 1 ≥ 0 is all of space: pos = everything, neg = nothing.
    let clip = square
        .clip(&HalfSpace {
            coeffs: vec![qi(0), qi(0)],
            constant: qi(1),
        })
        .unwrap();
    assert_eq!(sorted_dedup(&clip.pos), all);
    assert!(clip.neg.is_empty() && clip.on.is_empty());
    assert!(clip.pos_is_full_dimensional() && !clip.neg_is_full_dimensional());
    // 0 ≥ 0: every vertex is on the hyperplane-less cut; tight sets stay unchanged.
    let zero = HalfSpace {
        coeffs: vec![qi(0), qi(0)],
        constant: qi(0),
    };
    let clip = square.clip(&zero).unwrap();
    assert_eq!(sorted_dedup(&clip.pos), all);
    assert_eq!(sorted_dedup(&clip.neg), all);
    assert_eq!(sorted_dedup(&clip.on), all);
    let with_zero = clip.pos_polytope(&square, &zero).unwrap();
    assert!(
        with_zero
            .vertices_with_tight()
            .unwrap()
            .iter()
            .all(|(_, t)| t.len() == 2)
    );
    assert_eq!(with_zero.volume().unwrap(), qi(1));
    // Along an edge (x = 0): no crossing, two vertices on the plane, one side is the edge.
    let edge = HalfSpace {
        coeffs: vec![qi(-1), qi(0)],
        constant: qi(0),
    };
    let clip = square.clip(&edge).unwrap();
    assert_eq!(clip.on.len(), 2);
    assert_eq!(sorted_dedup(&clip.pos), sorted_dedup(&clip.on));
    assert_eq!(sorted_dedup(&clip.neg), all);
    assert!(!clip.pos_is_full_dimensional() && clip.neg_is_full_dimensional());
    assert_eq!(
        clip.pos_polytope(&square, &edge).unwrap().volume().unwrap(),
        qi(0)
    );
    // A polyhedron without vertices clips to nothing.
    let empty = square.with_halfspace(&[qi(1), qi(0)], qi(-2));
    let clip = empty.clip(&edge).unwrap();
    assert!(clip.pos.is_empty() && clip.neg.is_empty() && clip.on.is_empty());
    // Errors: dimension mismatch; a piece built from the wrong parent.
    assert!(
        square
            .clip(&HalfSpace {
                coeffs: vec![qi(1)],
                constant: qi(0)
            })
            .is_err()
    );
    let clip = square.clip(&edge).unwrap();
    let other = unit_cube(2).with_halfspace(&[qi(1), qi(1)], qi(-1));
    assert!(clip.pos_polytope(&other, &edge).is_err());
    let cube = unit_cube(3);
    assert!(clip.pos_polytope(&cube, &edge).is_err());
    let never_enumerated = unit_cube(2);
    assert!(clip.pos_polytope(&never_enumerated, &edge).is_err());
    assert!(clip.pos_polytope(&square, &edge).is_ok());
}

/// `vertices_with_tight` lists exactly the non-trivial half-spaces through
/// each vertex, and `is_full_dimensional_from_vertices` agrees with the LP
/// on bounded cells (including faces and empty ones).
#[test]
fn tight_sets_and_vertex_based_dimension() {
    let cube = unit_cube(3).with_halfspace(&[qi(0), qi(0), qi(0)], qi(0)); // plus a trivial row
    let tv = cube.vertices_with_tight().unwrap();
    assert_eq!(tv.len(), 8);
    for (v, t) in &tv {
        assert_eq!(t.len(), 3);
        assert!(!t.contains(&6), "a trivial row is never tight");
        assert!(t.iter().all(|&i| cube.halfspaces()[i].is_tight(v)));
        assert!(t.windows(2).all(|w| w[0] < w[1]));
    }
    assert_eq!(
        cube.vertices().unwrap(),
        tv.iter().map(|(v, _)| v.clone()).collect::<Vec<_>>()
    );
    assert!(cube.is_full_dimensional_from_vertices().unwrap());
    let mut lcg = Lcg(0x7164_7000);
    for _ in 0..30 {
        let mut poly = unit_cube(3);
        for _ in 0..lcg.range(1, 4) {
            let g = random_halfspace(&mut lcg, 3);
            poly = poly.with_halfspace(&g.coeffs, g.constant.clone());
        }
        if lcg.range(0, 3) == 0 {
            // Force a face or a lower-dimensional piece.
            let g = random_halfspace(&mut lcg, 3);
            poly = poly
                .with_halfspace(&g.coeffs, g.constant.clone())
                .with_halfspace(&g.flipped().coeffs, -&g.constant);
        }
        assert_eq!(
            poly.is_full_dimensional_from_vertices().unwrap(),
            poly.is_full_dimensional().unwrap(),
            "{poly:?}"
        );
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
