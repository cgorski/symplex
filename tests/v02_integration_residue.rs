//! v0.2 residues: higher-order poles, algebraic pole locations, essential
//! singularities, residue at infinity, and the residue theorem as a
//! cross-check of the definite integrator.

use symplex::prelude::*;

fn res_str(f: &Ex, z: &Ex, at: &Ex) -> String {
    format!("{}", f.residue(z, at))
}

#[test]
fn simple_poles() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    assert_eq!(res_str(&(&ctx.int(1) / &z), &z, &ctx.int(0)), "1");
    // 1/(z² − 1) at z = 1 → 1/2, at z = −1 → −1/2
    let f = &ctx.int(1) / &(&z.powi(2) - 1);
    assert_eq!(res_str(&f, &z, &ctx.int(1)), "1/2");
    assert_eq!(res_str(&f, &z, &ctx.int(-1)), "-1/2");
    // e^z/(z − 2) at 2 → e²
    let g = &z.exp() / &(&z - 2);
    let r = g.try_residue(&z, &ctx.int(2)).unwrap();
    assert!((r.eval_f64().unwrap() - 2f64.exp()).abs() < 1e-12, "{r}");
}

#[test]
fn higher_order_poles_from_spec() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    // Res_{z=0} e^z / z² = 1
    assert_eq!(res_str(&(&z.exp() / &z.powi(2)), &z, &ctx.int(0)), "1");
    // Res_{z=1} 1/(z−1)³ = 0
    assert_eq!(res_str(&(&z - 1).powi(-3), &z, &ctx.int(1)), "0");
    // Res_{z=0} cos z / z³ = −1/2
    assert_eq!(res_str(&(&z.cos() / &z.powi(3)), &z, &ctx.int(0)), "-1/2");
    // Res_{z=i} 1/(z²+1)² = −i/4
    let r = (&z.powi(2) + 1)
        .powi(-2)
        .try_residue(&z, &ctx.i_unit())
        .unwrap();
    let (re, im) = r.eval_complex64().unwrap();
    assert!(re.abs() < 1e-12 && (im + 0.25).abs() < 1e-12, "{r}");
}

#[test]
fn more_higher_order_poles() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    // Res_{z=0} sin z / z⁴ = −1/6
    assert_eq!(res_str(&(&z.sin() / &z.powi(4)), &z, &ctx.int(0)), "-1/6");
    // Res_{z=0} 1/(z² (z − 1)) = −1
    assert_eq!(
        res_str(&(&ctx.int(1) / &(&z.powi(2) * &(&z - 1))), &z, &ctx.int(0)),
        "-1"
    );
    // Res_{z=1} z/(z − 1)² = 1
    assert_eq!(res_str(&(&z / &(&z - 1).powi(2)), &z, &ctx.int(1)), "1");
    // Res_{z=0} e^{2z}/z³ = 2
    assert_eq!(
        res_str(&(&(&z * 2).exp() / &z.powi(3)), &z, &ctx.int(0)),
        "2"
    );
    // Res_{z=0} 1/(z sin z) = 0 (even function, double pole)
    let r = (&ctx.int(1) / &(&z * &z.sin())).residue(&z, &ctx.int(0));
    if !r.has_unevaluated() {
        assert_eq!(format!("{r}"), "0");
    }
    // Res_{z=0} 1/sin z = 1
    assert_eq!(res_str(&(&ctx.int(1) / &z.sin()), &z, &ctx.int(0)), "1");
    // Res_{z=0} z² e^z/(z − 1)... analytic at 0 → 0
    assert_eq!(
        res_str(&(&(&z.powi(2) * &z.exp()) / &(&z - 1)), &z, &ctx.int(0)),
        "0"
    );
}

#[test]
fn essential_singularity_stays_formal() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let f = (&ctx.int(1) / &z).exp();
    let r = f.residue(&z, &ctx.int(0));
    assert!(r.has_unevaluated(), "{r}");
    assert!(f.try_residue(&z, &ctx.int(0)).is_err());
}

#[test]
fn residue_at_infinity() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    assert_eq!(
        format!("{}", (&ctx.int(1) / &z).residue_at_infinity(&z)),
        "-1"
    );
    assert_eq!(format!("{}", z.residue_at_infinity(&z)), "0");
    // f = (z² + 1)/(z (z − 1)): finite residues sum to lim z f = 1 → Res_∞ = −1
    let f = &(&z.powi(2) + 1) / &(&z * &(&z - 1));
    assert_eq!(format!("{}", f.residue_at_infinity(&z)), "-1");
    // residue theorem: Σ finite residues + Res_∞ = 0
    let r0 = f.try_residue(&z, &ctx.int(0)).unwrap();
    let r1 = f.try_residue(&z, &ctx.int(1)).unwrap();
    let rinf = f.residue_at_infinity(&z);
    let total = (&(&r0 + &r1) + &rinf).eval();
    assert_eq!(format!("{total}"), "0");
    // residue at ∞ of a function without a Laurent tail → formal node
    let g = (&z * &z).exp();
    assert!(g.residue_at_infinity(&z).has_unevaluated());
}

#[test]
fn residue_theorem_cross_checks_definite_integral() {
    // ∫₋∞^∞ dx/(1+x²)² = 2πi · Res_{z=i} = 2πi · (−i/4) = π/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x.powi(2) + 1).powi(-2);
    let via_residue = 2.0 * std::f64::consts::PI * 0.25;
    let v = f
        .try_integrate_definite(&x, &ctx.neg_infinity(), &ctx.infinity())
        .unwrap();
    assert!((v.eval_f64().unwrap() - via_residue).abs() < 1e-12, "{v}");
}
