//! After 0.29 — the transforms / recurrences / residues / logic / LP /
//! vector / control hunter: `laplace`, `inverse_laplace`, `z_transform`,
//! `inverse_z_transform`, `fourier_transform`, `mellin_transform`,
//! `residue`, `rsolve_linear`, `rsolve_first_order`, `BoolEx` normal forms
//! and decisions, `linprog` certificates, `Polytope`, vector calculus in
//! three coordinate systems, quaternions and the control toolbox, each
//! checked against an independent oracle (quadrature over an own evaluator,
//! trapezoid rules on circles, exact recurrences and long division, truth
//! tables, exact dual / Farkas certificates, numeric roots).
//!
//! Each test says what was wrong before; every reference value cites the
//! oracle call that produced it (SymPy 1.14, mpmath 1.3.0 at the stated
//! `mp.dps`, NumPy `roots`).

use std::time::{Duration, Instant};

use symplex::control::{is_routh_stable, routh_array};
use symplex::num_bigint::BigInt;
use symplex::num_rational::Ratio;
use symplex::prelude::*;
use symplex::quaternion::{EulerConvention, Quaternion};
use symplex::rsolve::{rsolve_first_order, rsolve_linear};

type Q = Ratio<BigInt>;

fn q(p: i64, d: i64) -> Q {
    Q::new(BigInt::from(p), BigInt::from(d))
}

/// `|v − expected| ≤ tol·max(1, |expected|)` for the value of `e`.
fn assert_close(e: &Ex, expected: f64, tol: f64, what: &str) {
    let v = e
        .eval_f64()
        .unwrap_or_else(|err| panic!("{what}: {e} does not evaluate: {err}"));
    assert!(
        (v - expected).abs() <= tol * expected.abs().max(1.0),
        "{what}: {e} = {v}, expected {expected}"
    );
}

/// The exact sequence of `Σ c_j a(n+j) = f(n)` from its initial values
/// (the oracle for every recurrence test: forward iteration in ℚ).
fn iterate(coeffs: &[Q], forcing: impl Fn(i64) -> Q, ics: &[Q], count: usize) -> Vec<Q> {
    let k = coeffs.len() - 1;
    let mut a: Vec<Q> = ics.to_vec();
    while a.len() < count {
        let m = a.len() - k;
        let mut acc = forcing(m as i64);
        for j in 0..k {
            acc -= &coeffs[j] * &a[m + j];
        }
        a.push(acc / &coeffs[k]);
    }
    a
}

/// `sol(n)` equals `want[n]` (exactly when the value is rational, else to
/// 1e-9 relative).
fn assert_sequence(sol: &Ex, n: &Ex, want: &[Q], what: &str) {
    for (k, w) in want.iter().enumerate() {
        let at = sol.subs_i64(n, k as i64).eval();
        match at.as_rational() {
            Some(v) => assert_eq!(&v, w, "{what}: a({k}) from {sol}"),
            None => {
                let wf = w.numer().to_string().parse::<f64>().unwrap()
                    / w.denom().to_string().parse::<f64>().unwrap();
                assert_close(&at, wf, 1e-9, &format!("{what}: a({k})"));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Residues
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn a_residue_needs_an_analytic_numerator_not_a_finite_value() {
    // z·e^{1/z}, z²·sin(1/z) and z·cos(1/(2(z − 1))) have essential
    // singularities.  The numerator's value at the point folded to 0
    // (0·e^{1/0} → 0), so the point was taken as regular and the residue
    // came out 0.  The true residues are
    // mpmath (dps 30): quad(lambda th: f(c + r*expj(th))*r*expj(th), [0, 2*pi])/(2*pi)
    //   z*exp(1/z) at 0              -> 0.5
    //   z**2*sin(1/z) at 0           -> -0.166666666666666666666666666667
    //   z*cos(1/(2*(z-1))) at 1      -> -0.125
    // which the pole formulas cannot produce: refuse honestly.
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let zero = ctx.int(0);
    for (f, point) in [
        ("z*exp(1/z)", ctx.int(0)),
        ("z^2*sin(1/z)", ctx.int(0)),
        ("z*cos(1/2/(z - 1))", ctx.int(1)),
    ] {
        let e = ctx.parse(f).unwrap();
        assert!(
            e.try_residue(&z, &point).is_err(),
            "Res({f}) must not be a number"
        );
        assert!(
            e.residue(&z, &point).has_unevaluated(),
            "Res({f}) stays formal"
        );
    }
    // Poles keep working: Res(e^z/z², 0) = 1 (the documented example).
    let g = ctx.parse("exp(z)/z^2").unwrap();
    assert_eq!(g.try_residue(&z, &zero).unwrap(), ctx.int(1));
}

// ═══════════════════════════════════════════════════════════════════════════
// Recurrences
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn leading_zero_coefficients_keep_the_initial_values() {
    // 0·a(n) + a(n+1) = 3n·3ⁿ, a(0) = 3: the root 0 of the characteristic
    // polynomial was skipped, the initial value ignored, and 3ⁿ(n − 1) with
    // a(0) = −1 returned.  The solution space has KroneckerDelta(n, 0).
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let f = ctx.parse("3*n*3^n").unwrap();
    let sol = rsolve_linear(&[ctx.int(0), ctx.int(1)], Some(&f), &n, &[ctx.int(3)]).unwrap();
    let want = iterate(
        &[q(0, 1), q(1, 1)],
        |m| q(3 * m * 3i64.pow(m as u32), 1),
        &[q(3, 1)],
        10,
    );
    assert_sequence(&sol, &n, &want, "a(n+1) = 3n·3^n");

    // −3·a(n+2) = −n·2ⁿ − (1/2)^{n+1}, a(0) = −2, a(1) = 0 (two free values).
    let f2 = ctx.parse("-n*2^n - (1/2)*(1/2)^n").unwrap();
    let sol2 = rsolve_linear(
        &[ctx.int(0), ctx.int(0), ctx.int(-3)],
        Some(&f2),
        &n,
        &[ctx.int(-2), ctx.int(0)],
    )
    .unwrap();
    let want2 = iterate(
        &[q(0, 1), q(0, 1), q(-3, 1)],
        |m| q(-m * 2i64.pow(m as u32), 1) - q(1, 2i64.pow(m as u32 + 1)),
        &[q(-2, 1), q(0, 1)],
        10,
    );
    assert_sequence(&sol2, &n, &want2, "-3 a(n+2) = f(n)");
}

#[test]
fn complex_characteristic_roots_get_exact_angles_and_exact_fits() {
    // a(n+2) − a(n+1) + a(n) = 0, roots e^{±iπ/3}: the basis was
    // cos(n·atan2(√3/2, 1/2)), which eval does not reduce, so the fitted
    // closed form was a tangle of cos(k·atan2(…)) whose values at integers
    // could not even be certified (a(1) = 0 "precision exhausted").
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let coeffs = [ctx.int(1), ctx.int(-1), ctx.int(1)];
    let sol = rsolve_linear(&coeffs, None, &n, &[ctx.int(1), ctx.int(0)]).unwrap();
    let s = sol.to_string();
    assert!(s.contains("pi") && !s.contains("atan"), "{s}");
    let want = iterate(
        &[q(1, 1), q(-1, 1), q(1, 1)],
        |_| q(0, 1),
        &[q(1, 1), q(0, 1)],
        12,
    );
    assert_sequence(&sol, &n, &want, "e^{±iπ/3}");
}

#[test]
fn high_order_recurrences_fit_their_initial_values_without_swelling() {
    // Order 6 with two complex pairs, order 6 with a squared irreducible
    // cubic, order 8 with (r² − r − 1)²(r² + r + 1)²: the constants were
    // fitted by symbolic elimination over all roots at once (trig of atan2
    // angles, RootOf powers, two quadratic fields) and did not finish.
    type Forcing = Option<(&'static str, fn(i64) -> Q)>;
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let cases: [(&[i64], Forcing, &[i64]); 3] = [
        (
            &[32, -32, 48, -16, 18, -2, 2],
            Some(("n^2 + 3*(-1)^n", |m| {
                q(m * m + 3 * if m % 2 == 0 { 1 } else { -1 }, 1)
            })),
            &[-2, -1, 0, 3, -1, 3],
        ),
        (
            &[2, -4, -2, 0, 6, 4, 2],
            Some(("-n*2^n", |m| q(-m * 2i64.pow(m as u32), 1))),
            &[3, 1, 3, 1, -2, -3],
        ),
        (
            &[1, 4, 6, 4, -1, -4, -2, 0, 1],
            None,
            &[0, -1, -3, 2, 0, 3, 1, -2],
        ),
    ];
    for (c, forcing, ics) in cases {
        let coeffs: Vec<Ex> = c.iter().map(|&v| ctx.int(v)).collect();
        let init: Vec<Ex> = ics.iter().map(|&v| ctx.int(v)).collect();
        let f = forcing.map(|(src, _)| ctx.parse(src).unwrap());
        let start = Instant::now();
        let sol = rsolve_linear(&coeffs, f.as_ref(), &n, &init).unwrap();
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "{c:?} took {:?}",
            start.elapsed()
        );
        let cq: Vec<Q> = c.iter().map(|&v| q(v, 1)).collect();
        let iq: Vec<Q> = ics.iter().map(|&v| q(v, 1)).collect();
        let want = match forcing {
            Some((_, fq)) => iterate(&cq, fq, &iq, c.len() + 6),
            None => iterate(&cq, |_| q(0, 1), &iq, c.len() + 6),
        };
        assert_sequence(&sol, &n, &want, &format!("{c:?}"));
    }
}

#[test]
fn partial_initial_values_over_a_cubic_and_a_gaussian_factor() {
    // (2r + 1)²(r² + 2r + 2)(r³ − r − 1)/4 with six of seven initial values:
    // the global symbolic fit hung.  One constant stays free.
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let c = [
        q(-1, 2),
        q(-3, 1),
        q(-27, 4),
        q(-27, 4),
        q(-3, 2),
        q(13, 4),
        q(3, 1),
        q(1, 1),
    ];
    let coeffs: Vec<Ex> = c.iter().map(|v| ctx.from_ratio(v.clone())).collect();
    let ics = [0i64, -1, 1, 1, -2, 2];
    let init: Vec<Ex> = ics.iter().map(|&v| ctx.int(v)).collect();
    let start = Instant::now();
    let sol = rsolve_linear(&coeffs, None, &n, &init).unwrap();
    assert!(start.elapsed() < Duration::from_secs(20));
    // C1 is a(6): give it a value and compare with the iteration.
    let c1 = ctx.symbol("C1");
    let fixed = sol.subs(&c1, &ctx.rational(5, 7));
    let mut iq: Vec<Q> = ics.iter().map(|&v| q(v, 1)).collect();
    iq.push(q(5, 7));
    let want = iterate(&c, |_| q(0, 1), &iq, 12);
    assert_sequence(&fixed, &n, &want, "partial ics");
}

#[test]
fn first_order_recurrence_with_a_vanishing_coefficient() {
    // a(n+1) = n·a(n), a(0) = 3: the product Π_{k<n} k was written
    // Γ(n)/Γ(0), and 3·Γ(n)/Γ(0) evaluates to nothing.  The sequence is
    // 3, 0, 0, … (forward iteration).
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let sol = rsolve_first_order(&n, &ctx.int(0), &n, Some(&ctx.int(3))).unwrap();
    for (k, want) in [3, 0, 0, 0, 0].iter().enumerate() {
        assert_eq!(
            sol.subs_i64(&n, k as i64).eval(),
            ctx.int(*want),
            "a({k}) from {sol}"
        );
    }
    // With forcing the product formula does not apply: an error, not a value.
    assert!(rsolve_first_order(&n, &ctx.int(1), &n, Some(&ctx.int(3))).is_err());
    // A pole of p at an index leaves a(n) undefined.
    let p = ctx.parse("(n + 1)/(n - 2)").unwrap();
    assert!(rsolve_first_order(&p, &ctx.int(0), &n, Some(&ctx.int(1))).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse transforms of rational functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn inverse_laplace_of_repeated_complex_and_non_monic_poles() {
    // The partial-fraction table had no entry for repeated complex poles or
    // non-monic linear factors: 1/(s² + 4)² and 1/(2s + 1)³ were refused
    // (the docs promised every proper rational function).
    // SymPy: inverse_laplace_transform(1/(s**2+4)**2, s, t)
    //   = (-t*cos(2*t)/8 + sin(2*t)/16)*Heaviside(t)
    // SymPy: inverse_laplace_transform(1/(2*s+1)**3, s, t)
    //   = t**2*exp(-t/2)*Heaviside(t)/16
    let ctx = Context::new();
    let s = ctx.symbol("s");
    let t = ctx.symbol("t");
    let f1 = ctx
        .parse("1/(s^2 + 4)^2")
        .unwrap()
        .try_inverse_laplace(&s, &t)
        .unwrap();
    let f2 = ctx
        .parse("1/(2*s + 1)^3")
        .unwrap()
        .try_inverse_laplace(&s, &t)
        .unwrap();
    for tv in [0.5f64, 1.0, 2.5] {
        let tq = ctx.rational((tv * 2.0) as i64, 2);
        let w1 = -tv * (2.0 * tv).cos() / 8.0 + (2.0 * tv).sin() / 16.0;
        let w2 = tv * tv * (-tv / 2.0).exp() / 16.0;
        assert_close(&f1.subs(&t, &tq), w1, 1e-12, "L^-1 1/(s²+4)²");
        assert_close(&f2.subs(&t, &tq), w2, 1e-12, "L^-1 1/(2s+1)³");
    }
}

#[test]
fn inverse_z_transform_of_complex_and_non_monic_poles() {
    // 1/(z² + 1) and 1/(2z + 1)³ were refused ("cannot invert").  x[n] is
    // the coefficient of z^{-n}; with w = 1/z:
    // SymPy: series(w**2/(1+w**2), w, 0, 10) = w**2 - w**4 + w**6 - w**8 + O(w**10)
    // SymPy: series(w**3/(2+w)**3, w, 0, 8)
    //   = w**3/8 - 3*w**4/16 + 3*w**5/16 - 5*w**6/32 + 15*w**7/128 + O(w**8)
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let n = ctx.symbol("n");
    let x1 = ctx
        .parse("1/(z^2 + 1)")
        .unwrap()
        .inverse_z_transform(&z, &n)
        .unwrap();
    let w1 = [
        q(0, 1),
        q(0, 1),
        q(1, 1),
        q(0, 1),
        q(-1, 1),
        q(0, 1),
        q(1, 1),
        q(0, 1),
        q(-1, 1),
    ];
    assert_sequence(&x1, &n, &w1, "Z^-1 1/(z²+1)");
    let x2 = ctx
        .parse("1/(2*z + 1)^3")
        .unwrap()
        .inverse_z_transform(&z, &n)
        .unwrap();
    let w2 = [
        q(0, 1),
        q(0, 1),
        q(0, 1),
        q(1, 8),
        q(-3, 16),
        q(3, 16),
        q(-5, 32),
        q(15, 128),
    ];
    assert_sequence(&x2, &n, &w2, "Z^-1 1/(2z+1)³");
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward Laplace transforms
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn laplace_of_phase_shifts_products_and_shifted_polynomials() {
    // sin(ωt + φ) was not in the table, so the documented time-shift rule
    // H(t − a)·f(t) → e^{−as}·L{f(t + a)} failed for every trigonometric f;
    // cos² t, sinh t·sin t and t³·H(t − 1/2) were refused as well.
    // SymPy: laplace_transform(sin(t+1), t, s) = (s*cos(1 - pi/2) + cos(1))/(s**2 + 1)
    // SymPy: laplace_transform(Heaviside(t-1)*sin(3*t), t, s)
    //   = (s*cos(3 - pi/2) + 3*cos(3))*exp(-s)/(s**2 + 9)
    // SymPy: laplace_transform(cos(t)**2, t, s) = (s**2 + 2)/(s*(s**2 + 4))
    // SymPy: simplify(laplace_transform(t**3*Heaviside(t-1/2), t, s))
    //   = (s**3 + 6*s**2 + 24*s + 48)*exp(-s/2)/(8*s**4)
    // SymPy: laplace_transform(sinh(t)*sin(t), t, s) = 2*s/(s**4 + 4)
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    type Ref = fn(f64) -> f64;
    let cases: [(&str, Ref); 5] = [
        ("sin(t + 1)", |s| {
            (s * 1f64.sin() + 1f64.cos()) / (s * s + 1.0)
        }),
        ("Heaviside(t - 1)*sin(3*t)", |s| {
            (s * 3f64.sin() + 3.0 * 3f64.cos()) * (-s).exp() / (s * s + 9.0)
        }),
        ("cos(t)^2", |s| (s * s + 2.0) / (s * (s * s + 4.0))),
        ("t^3*Heaviside(t - 1/2)", |s| {
            (s.powi(3) + 6.0 * s * s + 24.0 * s + 48.0) * (-s / 2.0).exp() / (8.0 * s.powi(4))
        }),
        ("sinh(t)*sin(t)", |s| 2.0 * s / (s.powi(4) + 4.0)),
    ];
    for (src, reference) in cases {
        let big_f = ctx.parse(src).unwrap().try_laplace(&t, &s).unwrap();
        for sv in [2.0f64, 2.5, 4.0] {
            let sq = ctx.rational((sv * 2.0) as i64, 2);
            assert_close(&big_f.subs(&s, &sq), reference(sv), 1e-12, src);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Quaternions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn euler_angles_at_gimbal_lock_still_factor_the_rotation() {
    // At gimbal lock both atan2 pairs are atan2(0, 0) = 0: a quarter turn
    // about z came back as (0, 0, 0) in the ZXZ convention — the identity.
    // SymPy: Quaternion.from_axis_angle((0,0,1), pi/2).to_rotation_matrix()
    //   = Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    let ctx = Context::new();
    let (zero, one) = (ctx.int(0), ctx.int(1));
    let quarter = Quaternion::from_axis_angle(&zero, &zero, &one, &(ctx.pi() / 2));
    let (a, b, c) = quarter.to_euler(EulerConvention::ZXZ);
    let r = symplex::robotics::rot_euler(&a, &b, &c, EulerConvention::ZXZ);
    let want = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let got = r.eval_f64().unwrap();
    let same = |x: &[Vec<f64>], y: &[[f64; 3]]| {
        x.iter()
            .zip(y)
            .all(|(rx, ry)| rx.iter().zip(ry).all(|(u, v)| (u - v).abs() < 1e-12))
    };
    assert!(same(&got, &want), "ZXZ ({a}, {b}, {c}): {got:?}");
    // The other conventions at their locks (θ = ±π/2): every lock must
    // reproduce the quaternion's own matrix.
    let cases = [
        (
            EulerConvention::ZYX,
            ctx.rational(1, 3),
            ctx.pi() / 2,
            ctx.rational(1, 5),
        ),
        (
            EulerConvention::XYZ,
            ctx.rational(-2, 3),
            -(ctx.pi() / 2),
            ctx.rational(3, 4),
        ),
        (
            EulerConvention::ZXZ,
            ctx.rational(1, 2),
            ctx.pi(),
            ctx.rational(1, 7),
        ),
    ];
    for (conv, phi, theta, psi) in cases {
        let qe = Quaternion::from_euler(&phi, &theta, &psi, conv);
        let rq = qe.to_rotation_matrix().eval_f64().unwrap();
        let (a, b, c) = qe.to_euler(conv);
        let rb = symplex::robotics::rot_euler(&a, &b, &c, conv)
            .eval_f64()
            .unwrap();
        let agree = rq
            .iter()
            .zip(&rb)
            .all(|(x, y)| x.iter().zip(y).all(|(u, v)| (u - v).abs() < 1e-12));
        assert!(agree, "{conv:?}: {rq:?} vs {rb:?}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Routh–Hurwitz
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn routh_zero_tests_are_exact() {
    // Zeros in the first column were detected with |x| < 1e-30 in f64 and
    // replaced by the number 1e-9: s³ + εs² + s + 2ε (ε = 10⁻³²) came out
    // stable.  mpmath (dps 60): polyroots([1, e, 1, 2*e]) with e = 10**-32
    //   = [-2.0e-32, (5.0e-33 - 1.0j), (5.0e-33 + 1.0j)]  → two roots in the RHP.
    let ctx = Context::new();
    let eps = &ctx.int(1) / &ctx.int(10).powi(32);
    let c = [ctx.int(1), eps.clone(), ctx.int(1), &eps * 2];
    assert_eq!(is_routh_stable(&c), Some(false));
    let col: Vec<Ex> = routh_array(&c)
        .unwrap()
        .iter()
        .map(|r| r[0].clone())
        .collect();
    assert_eq!(col[2], ctx.int(-1), "exact first column {col:?}");

    // A leading zero coefficient is not part of the polynomial: 0·s² + s + 2
    // was unstable.  SymPy: Poly(s + 2, s).nroots() = [-2.00000000000000]
    assert_eq!(
        is_routh_stable(&[ctx.int(0), ctx.int(1), ctx.int(2)]),
        Some(true)
    );

    // ε method: s⁴ + s³ + 2s² + 2s + 3 has a zero pivot; the first column,
    // read as ε → 0⁺, changes sign twice.  mpmath (dps 60):
    // polyroots([1, 1, 2, 2, 3]) has two roots with real part 0.40574194.
    let c4: Vec<Ex> = [1, 1, 2, 2, 3].iter().map(|&v| ctx.int(v)).collect();
    assert_eq!(is_routh_stable(&c4), Some(false));
    let eps_sym = routh_array(&c4).unwrap()[2][0].clone();
    assert_eq!(eps_sym.to_string(), "ε");
    let small = &ctx.int(1) / &ctx.int(10).powi(40);
    let signs: Vec<bool> = routh_array(&c4)
        .unwrap()
        .iter()
        .map(|r| r[0].subs(&eps_sym, &small).eval_f64().unwrap() > 0.0)
        .collect();
    assert_eq!(signs.windows(2).filter(|w| w[0] != w[1]).count(), 2);
}
