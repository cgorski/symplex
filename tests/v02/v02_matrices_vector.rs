//! symplex 0.2 — vector calculus in curvilinear coordinates, line
//! integrals and scalar potentials.

use super::common;

use symplex::matrix::Matrix;
use symplex::prelude::*;
use symplex::vector::{
    CoordinateSystem, curl, curl_in, directional_derivative, divergence, divergence_in, gradient,
    gradient_in, hessian, is_conservative, is_irrotational, is_solenoidal, laplacian, laplacian_in,
    line_integral_scalar, line_integral_vector, scalar_potential,
};

#[test]
fn cartesian_functions_unchanged_and_curl_grad_zero() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let f = &(&x.powi(2) * &y) + &(&z * &x.sin());
    let g = gradient(&f, &[&x, &y, &z]).unwrap();
    assert_eq!(
        g,
        gradient_in(&f, &[&x, &y, &z], CoordinateSystem::Cartesian).unwrap()
    );
    let c = curl(&g, &[&x, &y, &z]).unwrap().simplify();
    assert_eq!(c.is_zero(), Some(true));
    assert_eq!(is_conservative(&g, &[&x, &y, &z]), Some(true));
    assert_eq!(is_irrotational(&g, &[&x, &y, &z]), Some(true));
    // div(curl F) = 0
    let field = Matrix::col_vector(vec![&y * &z, &x * &z.powi(2), &x.exp() * &y]);
    let cf = curl(&field, &[&x, &y, &z]).unwrap();
    assert_eq!(is_solenoidal(&cf, &[&x, &y, &z]), Some(true));
    assert_eq!(
        divergence(&cf, &[&x, &y, &z]).unwrap().expand().simplify(),
        ctx.int(0)
    );
    assert_eq!(
        laplacian(&(&x.powi(2) + &y.powi(2) + &z.powi(2)), &[&x, &y, &z]).unwrap(),
        ctx.int(6)
    );
    assert_eq!(
        hessian(&f, &[&x, &y, &z]).unwrap().is_symmetric(),
        Some(true)
    );
}

#[test]
fn spherical_vs_cartesian_laplacian_of_r_squared_and_harmonics() {
    let ctx = Context::new();
    let (r, th, ph) = (ctx.symbol("r"), ctx.symbol("theta"), ctx.symbol("phi"));
    let vars = [&r, &th, &ph];
    let sph = CoordinateSystem::Spherical;
    assert_eq!(
        laplacian_in(&r.powi(2), &vars, sph).unwrap().simplify(),
        ctx.int(6)
    );
    assert!(
        laplacian_in(&(ctx.int(1) / &r), &vars, sph)
            .unwrap()
            .simplify()
            .is_zero_structural()
    );
    // r^l P_l(cos θ) are harmonic: l = 1 (r cos θ) and l = 2 (r² (3cos²θ − 1)/2)
    let y1 = &r * &th.cos();
    assert!(
        laplacian_in(&y1, &vars, sph)
            .unwrap()
            .simplify()
            .is_zero_structural()
    );
    let y2 = &r.powi(2) * &(&(&th.cos().powi(2) * 3) - &ctx.int(1)) / 2;
    let l2 = laplacian_in(&y2, &vars, sph).unwrap().simplify();
    assert!(l2.is_zero_structural(), "∇²(r² P₂(cos θ)) = {l2}");
    // Non-axisymmetric: r sin θ cos φ (= x) is harmonic
    let xx = &(&r * &th.sin()) * &ph.cos();
    let lx = laplacian_in(&xx, &vars, sph).unwrap().simplify();
    assert!(lx.is_zero_structural(), "∇²x = {lx}");
    // Gradient magnitude of r is 1
    let g = gradient_in(&r, &vars, sph).unwrap();
    assert_eq!(g, matrix![ctx, [1], [0], [0]]);
    // div(r̂ / r²) = 0 (Coulomb field), div(r r̂) = 3
    let coulomb = Matrix::col_vector(vec![ctx.int(1) / r.powi(2), ctx.int(0), ctx.int(0)]);
    assert!(
        divergence_in(&coulomb, &vars, sph)
            .unwrap()
            .simplify()
            .is_zero_structural()
    );
    let radial = Matrix::col_vector(vec![r.clone(), ctx.int(0), ctx.int(0)]);
    assert_eq!(
        divergence_in(&radial, &vars, sph).unwrap().simplify(),
        ctx.int(3)
    );
    // curl(grad f) = 0 in spherical for a generic f
    let f = &(&r.powi(2) * &th.sin()) * &ph.cos();
    let c = curl_in(&gradient_in(&f, &vars, sph).unwrap(), &vars, sph)
        .unwrap()
        .simplify();
    assert_eq!(c.is_zero(), Some(true), "{c}");
}

#[test]
fn cylindrical_identities() {
    let ctx = Context::new();
    let (r, ph, z) = (ctx.symbol("r"), ctx.symbol("phi"), ctx.symbol("z"));
    let vars = [&r, &ph, &z];
    let cyl = CoordinateSystem::Cylindrical;
    assert_eq!(
        cyl.scale_factors(&vars).unwrap(),
        vec![ctx.one(), r.clone(), ctx.one()]
    );
    // ∇²(r² + z²) = 4 + 2 = 6
    assert_eq!(
        laplacian_in(&(&r.powi(2) + &z.powi(2)), &vars, cyl)
            .unwrap()
            .simplify(),
        ctx.int(6)
    );
    // ln r harmonic in the plane, r cos φ (= x) harmonic
    assert!(
        laplacian_in(&r.ln(), &vars, cyl)
            .unwrap()
            .simplify()
            .is_zero_structural()
    );
    assert!(
        laplacian_in(&(&r * &ph.cos()), &vars, cyl)
            .unwrap()
            .simplify()
            .is_zero_structural()
    );
    // Rigid rotation F = r φ̂: curl = 2 ẑ, divergence 0
    let rot = Matrix::col_vector(vec![ctx.int(0), r.clone(), ctx.int(0)]);
    assert_eq!(
        curl_in(&rot, &vars, cyl).unwrap().simplify(),
        matrix![ctx, [0], [0], [2]]
    );
    assert!(
        divergence_in(&rot, &vars, cyl)
            .unwrap()
            .simplify()
            .is_zero_structural()
    );
    // Magnetic field of a wire B = φ̂ / r is curl-free away from the axis
    let wire = Matrix::col_vector(vec![ctx.int(0), ctx.int(1) / &r, ctx.int(0)]);
    assert_eq!(
        curl_in(&wire, &vars, cyl).unwrap().simplify().is_zero(),
        Some(true)
    );
    // Numerical cross-check of the cylindrical Laplacian against Cartesian:
    // f = x² y = r³ cos²φ sin φ
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f_cart = &x.powi(2) * &y;
    let f_cyl = &(&r.powi(3) * &ph.cos().powi(2)) * &ph.sin();
    let lap_cart = laplacian(&f_cart, &[&x, &y, &z]).unwrap();
    let lap_cyl = laplacian_in(&f_cyl, &vars, cyl).unwrap();
    for (rv, pv) in [(1.5f64, 0.3f64), (2.0, 1.1), (0.7, 2.6)] {
        let rr = ctx.rational((rv * 1e6) as i64, 1_000_000);
        let pp = ctx.rational((pv * 1e6) as i64, 1_000_000);
        let xv = rv * pv.cos();
        let yv = rv * pv.sin();
        let a = lap_cyl.subs(&r, &rr).subs(&ph, &pp).eval_f64().unwrap();
        let b = lap_cart
            .subs(&x, &ctx.rational((xv * 1e9) as i64, 1_000_000_000))
            .subs(&y, &ctx.rational((yv * 1e9) as i64, 1_000_000_000))
            .eval_f64()
            .unwrap();
        assert!(
            common::approx_eq(a, b, 1e-6),
            "Laplacian mismatch: {a} vs {b}"
        );
    }
}

#[test]
fn directional_derivative_and_potential() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let f = &(&x * &y) + &z.powi(2);
    let d = matrix![ctx, [0], [0], [1]];
    assert_eq!(
        directional_derivative(&f, &[&x, &y, &z], &d).unwrap(),
        &z * 2
    );

    let phi = &(&x.powi(3) * &y) + &(&z * &y.cos()) + &x.exp();
    let field = gradient(&phi, &[&x, &y, &z]).unwrap();
    let back = scalar_potential(&field, &[&x, &y, &z]).unwrap();
    assert!(
        (&back - &phi).expand().simplify().is_zero_structural(),
        "{back}"
    );
    // Rotational field: Err(ComputationFailed); wrong shape: Err(InvalidArgument)
    let rot = Matrix::col_vector(vec![-&y, x.clone(), ctx.int(0)]);
    assert!(matches!(
        scalar_potential(&rot, &[&x, &y, &z]),
        Err(SymplexError::ComputationFailed { .. })
    ));
    assert!(matches!(
        scalar_potential(&rot, &[&x, &y]),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn line_integrals_scalar_and_vector() {
    let ctx = Context::new();
    let (x, y, z, t) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("z"),
        ctx.symbol("t"),
    );
    // Helix r(t) = (cos t, sin t, t), t ∈ [0, 2π]: length = 2π√2
    let helix = [t.cos(), t.sin(), t.clone()];
    let len = line_integral_scalar(
        &ctx.int(1),
        &[&x, &y, &z],
        &helix,
        &t,
        &ctx.int(0),
        &(ctx.pi() * 2),
    )
    .unwrap();
    let expected = 2.0 * std::f64::consts::PI * 2f64.sqrt();
    assert!(
        common::approx_eq(len.eval_f64().unwrap(), expected, 1e-12),
        "{len}"
    );
    // ∫ z ds along the helix = √2 · (2π)²/2
    let zs =
        line_integral_scalar(&z, &[&x, &y, &z], &helix, &t, &ctx.int(0), &(ctx.pi() * 2)).unwrap();
    let expected = 2f64.sqrt() * (2.0 * std::f64::consts::PI).powi(2) / 2.0;
    assert!(
        common::approx_eq(zs.eval_f64().unwrap(), expected, 1e-12),
        "{zs}"
    );
    // Conservative field: work = φ(end) − φ(start), independent of path
    let phi = &(&x * &y) + &z.powi(2);
    let f = gradient(&phi, &[&x, &y, &z]).unwrap();
    let seg = [t.clone(), &t * 2, &t * 3]; // (0,0,0) → (1,2,3)
    let w_seg =
        line_integral_vector(&f, &[&x, &y, &z], &seg, &t, &ctx.int(0), &ctx.int(1)).unwrap();
    let arc = [t.clone(), &t.powi(2) * 2, &t.powi(3) * 3]; // same endpoints
    let w_arc =
        line_integral_vector(&f, &[&x, &y, &z], &arc, &t, &ctx.int(0), &ctx.int(1)).unwrap();
    assert_eq!(w_seg.simplify(), ctx.int(11));
    assert_eq!(w_arc.simplify(), ctx.int(11));
    // Circulation of (−y, x) around a circle of radius 2 = 2π·4 = 8π
    let rot = Matrix::col_vector(vec![-&y, x.clone()]);
    let circle = [&t.cos() * 2, &t.sin() * 2];
    let circ =
        line_integral_vector(&rot, &[&x, &y], &circle, &t, &ctx.int(0), &(ctx.pi() * 2)).unwrap();
    assert_eq!(circ.simplify(), ctx.pi() * 8);
}

#[test]
fn spherical_requires_three_variables() {
    let ctx = Context::new();
    let (r, th) = (ctx.symbol("r"), ctx.symbol("theta"));
    let err = laplacian_in(&r, &[&r, &th], CoordinateSystem::Spherical).unwrap_err();
    assert!(err.to_string().contains("exactly 3 variables"), "{err}");
}
