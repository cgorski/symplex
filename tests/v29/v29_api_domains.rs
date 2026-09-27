//! Hunt over the API surfaces not covered by the earlier rounds: pattern
//! matching and rewriting, substitution, polynomial views and helpers, and
//! the domain modules (discrete transforms, combinatorics, dynamics,
//! robotics, variable separation, the exact kernels).  Found by the
//! differential hunter `zz_hunt_api` (exact oracles: reproduction of the
//! subject from match bindings, reconstruction of decompositions, naive
//! reference algorithms, numeric evaluation at rational points).  Every
//! test says what was wrong before; every reference value cites its oracle.

use std::time::{Duration, Instant};

use symplex::combinatorics::{catalan, derangements, stirling1, stirling2};
use symplex::dynamics::{GeneralizedCoordinate, euler_lagrange, manipulator_equation};
use symplex::num_bigint::BigInt;
use symplex::prelude::*;
use symplex::robotics::inverse_kinematics_2dof;

/// `a` and `b` agree at a few rational points (both evaluate there).
fn same_value(ctx: &Context, a: &Ex, b: &Ex, vars: &[&Ex]) -> bool {
    let points = [(3, 7), (-5, 11), (13, 10), (2, 9)];
    for (k, &(p, q)) in points.iter().enumerate() {
        let vals: Vec<Ex> = (0..vars.len())
            .map(|i| ctx.rational(p + i as i64 * (k as i64 + 1), q))
            .collect();
        let pairs: Vec<(&Ex, &Ex)> = vars.iter().copied().zip(vals.iter()).collect();
        let (va, vb) = (
            a.subs_map(&pairs).eval().eval_f64().unwrap(),
            b.subs_map(&pairs).eval().eval_f64().unwrap(),
        );
        if (va - vb).abs() > 1e-9 * (1.0 + va.abs().max(vb.abs())) {
            return false;
        }
    }
    true
}

// ═══════════════════════════════════════════════════════════════════════════
// Rewriting strategies terminate
// ═══════════════════════════════════════════════════════════════════════════

/// `sin(2·a_) → 2·sin(a_)·cos(a_)` matches `sin(9x)` again (`a_ = 9x/2`, the
/// numeric coefficient split SymPy's matcher makes too), so every
/// replacement contains a new redex.  The `TopDown` strategy used to descend
/// into replacements without bound and never finished one pass (hunter
/// `ident` seed 832: over 55 s, killed); the `Innermost` strategy
/// recomputed the same normal forms at every nesting level (the growing
/// rule below: over 20 s).  Now each pass follows at most eight nested
/// replacements and memoises them; the value is preserved.
#[test]
fn a_rewrite_top_down_and_innermost_terminate_on_self_reproducing_rules() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    let dbl = Rule::new("dbl", &(2 * &a).sin(), &(2 * a.sin() * a.cos()));
    let boom = Rule::new(
        "boom",
        &a.sin(),
        &(a.sin() * (&a + 1).sin() * (&a + 2).sin()),
    );
    for (rule, subject, identity) in [(dbl, (18 * &x).sin(), true), (boom, x.sin(), false)] {
        let rules = RuleSet::from_rules(vec![rule]);
        for strategy in [
            RewriteStrategy::TopDown,
            RewriteStrategy::Innermost,
            RewriteStrategy::BottomUp,
        ] {
            let start = Instant::now();
            let out = subject.rewrite_with(&rules, &RewriteOpts::default().strategy(strategy));
            assert!(
                start.elapsed() < Duration::from_secs(20),
                "{strategy:?} took {:?}",
                start.elapsed()
            );
            assert_ne!(out, subject, "{strategy:?} did not rewrite");
            if !identity {
                continue;
            }
            // Value preserved (the rule is an identity; checked at x = 3/7).
            let at = |e: &Ex| e.subs(&x, &ctx.rational(3, 7)).eval_f64().unwrap();
            let (v0, v1) = (at(&subject), at(&out));
            assert!(
                (v0 - v1).abs() < 1e-9 * (1.0 + v0.abs()),
                "{strategy:?}: {v0} vs {v1}"
            );
        }
    }
}

/// A single `TopDown` pass still collapses a tower at its root and rewrites
/// the fresh sub-terms of a replacement (the nesting bound is not reached).
#[test]
fn a_rewrite_top_down_still_rewrites_inside_replacements() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    // tan(a_) → sin(a_)/cos(a_), then sin(a_)/cos(a_) has no redex; the
    // replacement's fresh child sin(2x) is rewritten in the same pass.
    let rules = RuleSet::from_rules(vec![
        Rule::new("tan", &a.tan(), &(a.sin() / a.cos())),
        Rule::new("sin2", &(2 * &a).sin(), &(2 * a.sin() * a.cos())),
    ]);
    let e = (2 * &x).sin().tan();
    let opts = RewriteOpts::single_pass().strategy(RewriteStrategy::TopDown);
    let out = e.rewrite_with(&rules, &opts);
    let inner = 2 * x.sin() * x.cos();
    assert_eq!(out, inner.sin() / inner.cos(), "{out}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial views
// ═══════════════════════════════════════════════════════════════════════════

/// `is_polynomial` was defined as `degree().is_some()`, and the degree of
/// the zero polynomial is `None`, so `0`, `x − x` and `(x+1)² − (x²+2x+1)`
/// were "not polynomials".  SymPy: `S(0).is_polynomial(x)` → `True`, and
/// `((x+1)**2-(x**2+2*x+1)).is_polynomial(x)` → `True`.
#[test]
fn b_zero_polynomial_is_a_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cancelled = (&x + 1).powi(2) - (x.powi(2) + 2 * &x + 1);
    for e in [ctx.int(0), &x - &x, cancelled] {
        assert!(e.is_polynomial(&x), "{e}");
        assert_eq!(e.degree(&x), None, "{e}");
        assert_eq!(e.coeffs(&x), Some(vec![]), "{e}");
    }
    // SymPy: ((x**2-1)/(x-1)).is_polynomial(x) → False (no cancellation).
    assert!(!((x.powi(2) - 1) / (&x - 1)).is_polynomial(&x));
}

/// `Poly::eval` of `3·x^(2³² − 1)` at `x = 1` never finished: the exact
/// fast path was chosen because `1ⁿ` is within every size guard, and it
/// tabulates all the powers `1⁰ … 1^(2³² − 1)`.  The fast path now also
/// needs the degree within `max_pow_exponent`; beyond it the expression
/// path folds `1ⁿ` at once.  (Oracle: exact arithmetic, `3·1ⁿ = 3`; and
/// `3·2^(2³²−1)` stays a power, as `ctx.int(2).powi(…)` does.)
#[test]
fn b_poly_eval_of_huge_degree_at_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = Poly::from_terms(&ctx, &[&x], vec![(vec![u32::MAX], ctx.int(3))]).unwrap();
    let start = Instant::now();
    assert_eq!(p.eval(&[&ctx.int(1)]).unwrap(), ctx.int(3));
    assert_eq!(p.eval_gen(&x, &ctx.int(1)).unwrap().to_ex(), ctx.int(3));
    assert_eq!(
        p.eval(&[&ctx.int(2)]).unwrap(),
        3 * ctx.int(2).powi(i64::from(u32::MAX))
    );
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "{:?}",
        start.elapsed()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Dynamics
// ═══════════════════════════════════════════════════════════════════════════

/// `M·q̈ + C·q̇ + G` from `manipulator_equation` silently disagreed with
/// the Euler–Lagrange equations when the kinetic energy had a term linear
/// in the velocities (gyroscopic, `q₀·q̇₁`) or free of them (centrifugal
/// potential, `½w²q²`): `C` and `G` came from the Hessian and `V` only.
/// Oracle: SymPy `euler_equations(L, [q0(t), q1(t)], t)` for
/// `L = (q0'² + q1'²)/2 + q0·q1'` gives `q0'' − q1'` and `q0' + q1''`; for
/// `L = q'²/2 + w²q²/2` it gives `q'' − w²q`.
#[test]
fn c_manipulator_equation_includes_gyroscopic_and_velocity_free_terms() {
    let ctx = Context::new();
    let q: Vec<Ex> = ctx.symbols(&["q0", "q1"]);
    let qd: Vec<Ex> = ctx.symbols(&["qd0", "qd1"]);
    let qdd: Vec<Ex> = ctx.symbols(&["qdd0", "qdd1"]);
    let t = (qd[0].powi(2) + qd[1].powi(2)) / 2 + &q[0] * &qd[1];
    let v = ctx.int(0);
    let me = manipulator_equation(&t, &v, &[&q[0], &q[1]], &[&qd[0], &qd[1]]).unwrap();
    let lhs: Vec<Ex> = (0..2)
        .map(|i| {
            let mut s = me.gravity[i].clone();
            for j in 0..2 {
                s = &s + &(me.mass.get(i, j) * &qdd[j]) + &(me.coriolis.get(i, j) * &qd[j]);
            }
            s
        })
        .collect();
    let want = [&qdd[0] - &qd[1], &qd[0] + &qdd[1]];
    let vars: Vec<&Ex> = q.iter().chain(&qd).chain(&qdd).collect();
    for i in 0..2 {
        assert!(
            same_value(&ctx, &lhs[i], &want[i], &vars),
            "eq {i}: {}",
            lhs[i]
        );
    }
    // The gyroscopic matrix is skew-symmetric: C = [[0, -1], [1, 0]].
    assert_eq!(me.coriolis.get(0, 1).to_string(), "-1");
    assert_eq!(me.coriolis.get(1, 0).to_string(), "1");
    // Agreement with euler_lagrange itself.
    let coords: Vec<GeneralizedCoordinate> = (0..2)
        .map(|i| GeneralizedCoordinate {
            q: &q[i],
            q_dot: &qd[i],
            q_ddot: &qdd[i],
        })
        .collect();
    let el = euler_lagrange(&t, &v, &coords);
    for i in 0..2 {
        assert!(same_value(&ctx, &lhs[i], &el[i], &vars));
    }

    // Centrifugal potential: G = −w²·q.
    let (w, qq, qqd) = (ctx.symbol("w"), ctx.symbol("q"), ctx.symbol("qd"));
    let t2 = qqd.powi(2) / 2 + w.powi(2) * qq.powi(2) / 2;
    let me2 = manipulator_equation(&t2, &ctx.int(0), &[&qq], &[&qqd]).unwrap();
    assert!(same_value(
        &ctx,
        &me2.gravity[0],
        &-(w.powi(2) * &qq),
        &[&w, &qq]
    ));
}

/// A kinetic energy that is not of degree ≤ 2 in the velocities has no
/// `M(q)·q̈ + C(q, q̇)·q̇ + G(q)` form: refused (it used to return a
/// velocity-dependent "mass matrix" and a meaningless `C`).
#[test]
fn c_manipulator_equation_refuses_non_quadratic_kinetic_energy() {
    let ctx = Context::new();
    let (q, qd) = (ctx.symbol("q"), ctx.symbol("qd"));
    let relativistic = (1 + qd.powi(2)).sqrt();
    let err = manipulator_equation(&relativistic, &ctx.int(0), &[&q], &[&qd]).unwrap_err();
    assert!(err.to_string().contains("degree at most two"), "{err}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Robotics
// ═══════════════════════════════════════════════════════════════════════════

fn fk(l1: f64, l2: f64, t1: f64, t2: f64) -> (f64, f64) {
    (
        l1 * t1.cos() + l2 * (t1 + t2).cos(),
        l1 * t1.sin() + l2 * (t1 + t2).sin(),
    )
}

/// `inverse_kinematics_2dof` solved a Gröbner system over ℚ, which finds
/// only solutions whose sines and cosines are rational, so a reachable
/// target with irrational angles came back empty ("unreachable"): unit
/// links to `(1.5, 0.5)`, or links `(0.5, 1.875)` to `(−1.1408…, −0.9226…)`
/// (hunter `robo` seed 2).  Oracle (mpmath, 20 digits):
/// `θ₂ = ±acos(1/4) = ±1.3181160716528179657`,
/// `θ₁ = atan2(0.5, 1.5) − atan2(sin θ₂, 1 + cos θ₂)` =
/// `−0.33730748142976678947` and `0.98080859022305117627`.
#[test]
fn d_ik_2dof_finds_irrational_solutions() {
    let sols = inverse_kinematics_2dof(1.0, 1.0, 1.5, 0.5);
    assert_eq!(sols.len(), 2, "{sols:?}");
    let want = [
        (-0.337_307_481_429_766_8, 1.318_116_071_652_818),
        (0.980_808_590_223_051_2, -1.318_116_071_652_818),
    ];
    for ((t1, t2), (w1, w2)) in sols.iter().zip(want) {
        assert!(
            (t1 - w1).abs() < 1e-12 && (t2 - w2).abs() < 1e-12,
            "{sols:?}"
        );
    }
    let (l1, l2, x, y) = (
        0.5,
        1.875,
        -1.140_848_806_366_047_8,
        -0.922_627_737_226_277_3,
    );
    let sols = inverse_kinematics_2dof(l1, l2, x, y);
    assert_eq!(sols.len(), 2, "{sols:?}");
    for (t1, t2) in sols {
        let (fx, fy) = fk(l1, l2, t1, t2);
        assert!((fx - x).abs() < 1e-12 && (fy - y).abs() < 1e-12);
    }
}

/// Boundary, unreachable, degenerate targets (FK is the oracle).
#[test]
fn d_ik_2dof_boundary_and_degenerate_targets() {
    // Stretched: 1 + 0.8 = 1.8 within the rounding of the inputs.
    assert_eq!(
        inverse_kinematics_2dof(1.0, 0.8, 1.8, 0.0),
        vec![(0.0, 0.0)]
    );
    // Folded back: r = |l1 − l2| = 0.2, θ₂ = π.
    let folded = inverse_kinematics_2dof(1.0, 0.8, 0.2, 0.0);
    assert_eq!(folded.len(), 1);
    let (fx, fy) = fk(1.0, 0.8, folded[0].0, folded[0].1);
    assert!((fx - 0.2).abs() < 1e-12 && fy.abs() < 1e-12, "{folded:?}");
    // Outside and inside the annulus.
    assert!(inverse_kinematics_2dof(1.0, 0.8, 1.9, 0.0).is_empty());
    assert!(inverse_kinematics_2dof(1.0, 0.8, 0.1, 0.0).is_empty());
    // Folded onto the origin: any θ₁; one representative.
    let origin = inverse_kinematics_2dof(1.0, 1.0, 0.0, 0.0);
    assert_eq!(origin.len(), 1);
    let (fx, fy) = fk(1.0, 1.0, origin[0].0, origin[0].1);
    assert!(fx.abs() < 1e-12 && fy.abs() < 1e-12);
    // A zero-length second link: θ₂ is free.
    let one_link = inverse_kinematics_2dof(2.0, 0.0, 0.0, 2.0);
    assert_eq!(one_link, vec![(std::f64::consts::FRAC_PI_2, 0.0)]);
    // Non-finite input.
    assert!(inverse_kinematics_2dof(f64::NAN, 1.0, 1.0, 1.0).is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Combinatorics
// ═══════════════════════════════════════════════════════════════════════════

/// Inputs that fit in `u64` but whose results cannot be built in memory
/// crashed: `catalan(2⁶³)` overflowed `2n` (a panic; `0` in release),
/// `stirling2(u64::MAX, 2) = 2^(n−1) − 1` aborted the process on the
/// allocation, `stirling1(u64::MAX, 1) = (n−1)!` and `derangements(u64::MAX)`
/// ran forever.  They return `None` now (the result would exceed
/// `MAX_RESULT_BITS`); small values are unchanged — SymPy:
/// `catalan(15)` → 9694845, `stirling(5, 3)` → 25,
/// `stirling(4, 1, kind=1, signed=True)` → −6, `subfactorial(10)` → 1334961.
#[test]
fn e_combinatorics_refuse_results_too_large_to_build() {
    assert_eq!(catalan(1u64 << 63), None);
    assert_eq!(catalan(u64::MAX), None);
    assert_eq!(stirling2(u64::MAX, 2u64), None);
    assert_eq!(stirling1(u64::MAX, 1u64), None);
    assert_eq!(derangements(u64::MAX), None);
    // S(n, n−1) = C(n, 2) is small for every n: still computed.
    let n = BigInt::from(u64::MAX);
    assert_eq!(stirling2(u64::MAX, u64::MAX - 1), Some(&n * (&n - 1) / 2));
    assert_eq!(catalan(15), Some(BigInt::from(9_694_845)));
    assert_eq!(stirling2(5, 3), Some(BigInt::from(25)));
    assert_eq!(stirling1(4, 1), Some(BigInt::from(-6)));
    assert_eq!(derangements(10), Some(BigInt::from(1_334_961)));
}
