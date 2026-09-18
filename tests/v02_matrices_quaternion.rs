//! symplex 0.2 — `Quaternion` operators, conversions and calculus.

mod common;

use symplex::matrix::Matrix;
use symplex::prelude::*;
use symplex::quaternion::EulerConvention;
use symplex::robotics::{rot_euler, rot_x, rot_y, rot_z};

fn qi(ctx: &Context, w: i64, x: i64, y: i64, z: i64) -> Quaternion {
    Quaternion::new(ctx.int(w), ctx.int(x), ctx.int(y), ctx.int(z))
}

fn comps(q: &Quaternion) -> [f64; 4] {
    [
        q.w.eval_f64().unwrap(),
        q.x.eval_f64().unwrap(),
        q.y.eval_f64().unwrap(),
        q.z.eval_f64().unwrap(),
    ]
}

fn rat(ctx: &Context, v: f64) -> Ex {
    ctx.rational((v * 1e6).round() as i64, 1_000_000)
}

#[test]
fn operators_and_algebra_laws() {
    let ctx = Context::new();
    let p = qi(&ctx, 1, 2, 3, 4);
    let q = qi(&ctx, -2, 1, 0, 5);
    let r = qi(&ctx, 3, -1, 2, 1);
    // Non-commutative, associative, distributive
    assert_ne!(&p * &q, &q * &p);
    assert_eq!((&(&p * &q) * &r).eval(), (&p * &(&q * &r)).eval());
    assert_eq!((&p * &(&q + &r)).eval(), (&(&p * &q) + &(&p * &r)).eval());
    // |pq| = |p||q|
    let n_pq = (&p * &q).norm_squared().eval_f64().unwrap();
    let n_p = p.norm_squared().eval_f64().unwrap();
    let n_q = q.norm_squared().eval_f64().unwrap();
    assert!(common::approx_eq(n_pq, n_p * n_q, 1e-9));
    // (pq)* = q* p*
    assert_eq!(
        (&p * &q).conjugate().eval(),
        (&q.conjugate() * &p.conjugate()).eval()
    );
    // Scalars on both sides, division, negation
    assert_eq!(&p * 2, 2 * &p);
    assert_eq!(&p * &ctx.int(3), &ctx.int(3) * &p);
    assert_eq!((&p / 2).w, ctx.rational(1, 2));
    assert_eq!(-&p, &p * -1);
    assert_eq!(&p - &p, Quaternion::zero(&ctx));
    // Inverse
    assert_eq!((&p * &p.inverse()).eval(), Quaternion::identity(&ctx));
    assert_eq!(p.dot(&q), ctx.int(20));
    assert_eq!(format!("{p}"), "(1 + 2i + 3j + 4k)");
    assert_eq!(p.components().len(), 4);
    assert_eq!(p.vector(), matrix![ctx, [2], [3], [4]]);
}

#[test]
fn is_unit_three_valued() {
    let ctx = Context::new();
    let th = ctx.symbol("theta");
    let u = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(1), &ctx.int(0), &th);
    assert_eq!(u.is_unit(), Some(true));
    assert_eq!(qi(&ctx, 1, 1, 0, 0).is_unit(), Some(false));
    assert_eq!(qi(&ctx, 1, 1, 0, 0).normalize().is_unit(), Some(true));
    let a = ctx.symbol("a");
    assert_eq!(
        Quaternion::new(a, ctx.int(0), ctx.int(0), ctx.int(0)).is_unit(),
        None
    );
}

#[test]
fn rotation_matrix_round_trip_symbolic_and_numeric() {
    let ctx = Context::new();
    let th = ctx.symbol_with("theta", &[Assumption::Positive]);
    for r in [rot_x(&th), rot_y(&th), rot_z(&th)] {
        let q = Quaternion::from_rotation_matrix(&r).unwrap();
        let back = q.to_rotation_matrix().simplify();
        assert_eq!(back.equals(&r), Some(true), "{back}");
    }
    // Numeric, including 180° rotations (trace = −1 → non-trace Shepperd branches)
    for r in [
        rot_x(&ctx.pi()),
        rot_y(&ctx.pi()),
        rot_z(&ctx.pi()),
        rot_euler(
            &rat(&ctx, 2.9),
            &rat(&ctx, -1.2),
            &rat(&ctx, 0.4),
            EulerConvention::ZYX,
        ),
    ] {
        let r = r.eval();
        let q = Quaternion::from_rotation_matrix(&r).unwrap();
        assert!(common::approx_eq(
            q.norm_squared().eval_f64().unwrap(),
            1.0,
            1e-12
        ));
        let back = q.to_rotation_matrix().eval_f64().unwrap();
        let orig = r.eval_f64().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    common::approx_eq(back[i][j], orig[i][j], 1e-10),
                    "({i},{j})"
                );
            }
        }
    }
    // Not a rotation
    assert!(
        Quaternion::from_rotation_matrix(&matrix![ctx, [2, 0, 0], [0, 1, 0], [0, 0, 1]]).is_err()
    );
    assert!(Quaternion::from_rotation_matrix(&matrix![ctx, [1, 0], [0, 1]]).is_err());
}

#[test]
fn axis_angle_and_rotate_vector() {
    let ctx = Context::new();
    let axis = matrix![ctx, [0], [0], [1]];
    let angle = ctx.pi() / 2;
    let q = Quaternion::from_axis_angle(&axis[(0, 0)], &axis[(1, 0)], &axis[(2, 0)], &angle);
    let (ax, ang) = q.to_axis_angle().unwrap();
    assert_eq!(ax.eval(), axis);
    assert_eq!(ang.eval(), angle);
    // x̂ → ŷ, ŷ → −x̂
    assert_eq!(
        q.rotate_vector(&matrix![ctx, [1], [0], [0]])
            .unwrap()
            .eval()
            .simplify(),
        matrix![ctx, [0], [1], [0]]
    );
    assert_eq!(
        q.rotate_vector(&matrix![ctx, [0], [1], [0]])
            .unwrap()
            .eval()
            .simplify(),
        matrix![ctx, [-1], [0], [0]]
    );
    // rotate_vector == R·v symbolically
    let th = ctx.symbol("theta");
    let qs = Quaternion::from_axis_angle(&ctx.int(1), &ctx.int(0), &ctx.int(0), &th);
    let v = matrix![ctx, [1], [2], [3]];
    let a = qs.rotate_vector(&v).unwrap().simplify();
    let b = (&qs.to_rotation_matrix() * &v).simplify();
    assert_eq!(a.equals(&b), Some(true));
    assert!(qs.rotate_vector(&matrix![ctx, [1], [2]]).is_err());
    assert!(Quaternion::identity(&ctx).to_axis_angle().is_err());
}

#[test]
fn euler_conversions_agree_with_robotics_and_round_trip() {
    let ctx = Context::new();
    let samples: [(f64, f64, f64); 3] = [(0.3, -0.4, 0.7), (-1.2, 0.9, 2.5), (2.0, 1.1, -0.6)];
    for conv in [
        EulerConvention::ZYX,
        EulerConvention::XYZ,
        EulerConvention::ZXZ,
    ] {
        for (a, b, c) in samples {
            let b = if matches!(conv, EulerConvention::ZXZ) {
                b.abs()
            } else {
                b
            };
            let (ea, eb, ec) = (rat(&ctx, a), rat(&ctx, b), rat(&ctx, c));
            let q = Quaternion::from_euler(&ea, &eb, &ec, conv);
            let m = q.to_rotation_matrix().eval_f64().unwrap();
            let r = rot_euler(&ea, &eb, &ec, conv).eval_f64().unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        common::approx_eq(m[i][j], r[i][j], 1e-10),
                        "{conv:?} ({i},{j})"
                    );
                }
            }
            let (pa, pb, pc) = q.to_euler(conv);
            let got = [
                pa.eval_f64().unwrap(),
                pb.eval_f64().unwrap(),
                pc.eval_f64().unwrap(),
            ];
            let exp = [
                ea.eval_f64().unwrap(),
                eb.eval_f64().unwrap(),
                ec.eval_f64().unwrap(),
            ];
            for k in 0..3 {
                assert!(
                    common::approx_eq(got[k], exp[k], 1e-9),
                    "{conv:?}: {got:?} vs {exp:?}"
                );
            }
        }
    }
}

#[test]
fn slerp_is_geodesic() {
    let ctx = Context::new();
    let q0 = Quaternion::identity(&ctx);
    let q1 = Quaternion::from_axis_angle(
        &ctx.int(0),
        &ctx.int(0),
        &ctx.int(1),
        &(ctx.pi() * ctx.rational(2, 3)),
    );
    let t = ctx.symbol("t");
    let s = q0.slerp(&q1, &t);
    for k in 0..=4 {
        let tv = ctx.rational(k, 4);
        let st = s.subs(&t, &tv).eval();
        let c = comps(&st);
        // slerp(t) is the rotation by t·120° about z
        let ang = (k as f64 / 4.0) * 2.0 * std::f64::consts::PI / 3.0;
        assert!(
            common::approx_eq(c[0], (ang / 2.0).cos(), 1e-9),
            "t={k}/4 w"
        );
        assert!(
            common::approx_eq(c[3], (ang / 2.0).sin(), 1e-9),
            "t={k}/4 z"
        );
        assert!(common::approx_eq(c[0] * c[0] + c[3] * c[3], 1.0, 1e-9));
    }
    // and equals q1^t (power) along the way
    let pw = q1.pow(&ctx.rational(1, 4)).eval();
    let sl = s.subs(&t, &ctx.rational(1, 4)).eval();
    let (a, b) = (comps(&pw), comps(&sl));
    for k in 0..4 {
        assert!(common::approx_eq(a[k], b[k], 1e-9));
    }
}

#[test]
fn exp_ln_inverse_pair() {
    let ctx = Context::new();
    let q = qi(&ctx, 1, 2, 3, 4);
    let back = q.ln().exp().eval();
    let (a, b) = (comps(&back), comps(&q));
    for k in 0..4 {
        assert!(
            common::approx_eq(a[k], b[k], 1e-9),
            "exp(ln q) = q: {a:?} vs {b:?}"
        );
    }
    let th = ctx.symbol_with("theta", &[Assumption::Positive]);
    let n = Quaternion::from_axis_angle(&ctx.int(0), &ctx.int(1), &ctx.int(0), &th);
    let l = n.ln().simplify();
    // ln of a unit rotation quaternion is (θ/2)·axis (for θ ∈ (0, 2π))
    let l_at = l.subs(&th, &rat(&ctx, 1.4)).eval();
    let c = comps(&l_at);
    assert!(
        common::approx_eq(c[0], 0.0, 1e-9) && common::approx_eq(c[2], 0.7, 1e-9),
        "{c:?}"
    );
    // pow(2) doubles the angle
    let sq = n.pow(&ctx.int(2));
    let (ax, ang) = sq.to_axis_angle().unwrap();
    let ang_v = ang.subs(&th, &rat(&ctx, 0.5)).eval_f64().unwrap();
    assert!(common::approx_eq(ang_v, 1.0, 1e-9), "{ang_v}");
    assert!(common::approx_eq(
        ax.subs(&th, &rat(&ctx, 0.5)).eval_f64().unwrap()[1][0],
        1.0,
        1e-9
    ));
}

#[test]
fn kinematics_diff_matches_angular_velocity_formula() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let (w, phase) = (ctx.symbol("omega"), ctx.symbol("phi0"));
    let angle = &(&w * &t) + &phase;
    let q = Quaternion::from_axis_angle(&ctx.int(1), &ctx.int(0), &ctx.int(0), &angle);
    let qdot = q.diff(&t);
    let kin = q.angular_velocity_derivative(&w, &ctx.int(0), &ctx.int(0));
    assert_eq!(qdot.simplify().equals(&kin.simplify()), Some(true));
    let _ = Matrix::identity(&ctx, 1);
}
