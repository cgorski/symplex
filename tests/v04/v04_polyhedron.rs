//! symplex 0.4 — certificates on a parametric polyhedron:
//! `prove_nonnegative_on_polyhedron` / `prove_polyhedron_empty`, the staged
//! LP, exact refutation, and the Lean export.
//!
//! `fixtures/polyhedron_certificates.lean` is the exact text these
//! certificates emitted when it was compiled against Mathlib (Lean 4.30.0)
//! with `linter.style.longLine` enabled and no errors or warnings; the
//! emitter must reproduce it byte for byte.

use num_traits::Signed;
use symplex::certificates::{
    ParamBound, PolyhedronCertificate, PolyhedronLeanNames, PolyhedronOpts, PolyhedronOutcome,
    PolyhedronUnknown, prove_nonnegative_on_polyhedron, prove_polyhedron_empty,
};
use symplex::lean::LeanOpts;
use symplex::linprog::{q, qi};
use symplex::prelude::*;

const COMPILED: &str = include_str!("../fixtures/polyhedron_certificates.lean");

/// The parameter bound `var ≥ lower`.
fn ge(var: &Ex, lower: Ex) -> ParamBound {
    ParamBound {
        var: var.clone(),
        lower,
    }
}

fn proved(out: PolyhedronOutcome) -> PolyhedronCertificate {
    match out {
        PolyhedronOutcome::Proved(c) => {
            assert!(c.verify(), "certificate must re-verify: {c}");
            c
        }
        other => panic!("expected a certificate, got {other:?}"),
    }
}

struct Fixture {
    ctx: Context,
    j: Ex,
    r: Ex,
    t: Ex,
}

impl Fixture {
    fn new() -> Self {
        let ctx = Context::new();
        let (j, r, t) = (ctx.symbol("j"), ctx.symbol("r"), ctx.symbol("t"));
        Fixture { ctx, j, r, t }
    }

    fn prove(&self, goal: &Ex, hyps: &[Ex], j0: Option<i64>) -> PolyhedronCertificate {
        let param = j0.map(|v| ge(&self.j, self.ctx.int(v)));
        proved(
            prove_nonnegative_on_polyhedron(goal, hyps, param.as_ref(), &PolyhedronOpts::default())
                .unwrap(),
        )
    }

    /// The fifteen certificates of the compiled fixture, in file order.
    fn compiled_certificates(&self) -> Vec<(&'static str, PolyhedronCertificate)> {
        let (ctx, j, r, t) = (&self.ctx, &self.j, &self.r, &self.t);
        let half = ctx.rational(1, 2);
        let cell = [
            r.clone(),
            &half - r,
            t.clone(),
            1 - t,
            (j * 2 + 1) * t - j * r - 1,
        ];
        let needs_lambda = [t - r, t + j * r - j - 1];
        let needs_lambda_sq = [t - r, t + j.powi(2) * r - j.powi(2) - 1];
        let kh = [t - r, r.clone()];
        let pw = [r.clone(), 1 - r];
        let empty = [
            r - &half,
            (j * 2 + 1) * t - j * r - 1,
            ctx.rational(1, 4) - t,
        ];
        let empty_lambda = [t - r, t + j * r - j - 1, &half - t];
        let fixed = [r.clone(), 1 - r, t - r];
        let neg_lambda = [t - r, t + (j + 1) * r - j - 2];
        let empty_cert = |hyps: &[Ex], j0: i64| {
            proved(
                prove_polyhedron_empty(hyps, Some(&ge(j, ctx.int(j0))), &PolyhedronOpts::default())
                    .unwrap(),
            )
        };
        vec![
            (
                "cell_goal",
                self.prove(&((j * 2 + 1) * t * 4 - j * r * 4 - r - 3), &cell, Some(2)),
            ),
            ("needs_lambda", self.prove(&(t - 1), &needs_lambda, Some(0))),
            (
                "needs_lambda_sq",
                self.prove(&(t - 1), &needs_lambda_sq, Some(0)),
            ),
            ("k_chain", self.prove(&((j - 2) * t), &kh, Some(2))),
            ("neg_j0", self.prove(&((j + 1) * t), &kh, Some(-1))),
            ("jk_chain", self.prove(&(j * (j - 2) * t), &kh, Some(2))),
            ("pairwise", self.prove(&(r - r.powi(2)), &pw, None)),
            (
                "pairwise_j",
                self.prove(&(j * (r - r.powi(2))), &pw, Some(1)),
            ),
            ("cell_empty", empty_cert(&empty, 2)),
            ("cell_empty_lambda", empty_cert(&empty_lambda, 0)),
            (
                "pure_power",
                self.prove(&(j.powi(2) + t), std::slice::from_ref(t), Some(0)),
            ),
            ("farkas", self.prove(&(t * 2 - r), &fixed, None)),
            (
                "pure_jk",
                self.prove(&(j * (j - 2) + t), std::slice::from_ref(t), Some(2)),
            ),
            (
                "pure_kk",
                self.prove(&((j - 2).powi(2) + t), std::slice::from_ref(t), Some(2)),
            ),
            ("neg_lambda", self.prove(&(t - 1), &neg_lambda, Some(-1))),
        ]
    }
}

#[test]
fn emitted_lean_matches_the_mathlib_compiled_fixture() {
    let f = Fixture::new();
    let mut text = String::from("import Mathlib\nset_option linter.style.longLine true\n\n");
    let certs = f.compiled_certificates();
    for (i, (name, c)) in certs.iter().enumerate() {
        text.push_str(&c.to_lean(name).unwrap());
        if i + 1 < certs.len() {
            text.push('\n');
        }
    }
    assert_eq!(text, COMPILED);
    assert!(text.lines().all(|l| l.chars().count() <= 100));
}

#[test]
fn lambda_is_required_and_minimal() {
    let f = Fixture::new();
    let (j, r, t) = (&f.j, &f.r, &f.t);
    let hyps = [t - r, t + j * r - j - 1];
    let c = f.prove(&(t - 1), &hyps, Some(0));
    assert_eq!(c.lambda_coeffs(), &[q(1, 1), q(1, 1)]);
    assert_eq!(c.lambda(), j + 1);
    assert_eq!(c.degree(), 1);
    assert!(!c.uses_pairwise());
    assert_eq!(c.terms().len(), 2);
    assert_eq!(
        c.to_string(),
        "(j + 1)*(t - 1) = j*h0 + h1; h0 = -r + t, h1 = j*r - j + t - 1; j ≥ 0"
    );
    let Equation { lhs, rhs } = c.identity();
    assert!((lhs - rhs).expand().is_zero_structural());
    // A degree-2 λ is found when needed and not otherwise.
    let sq = [t - r, t + j.powi(2) * r - j.powi(2) - 1];
    let c2 = f.prove(&(t - 1), &sq, Some(0));
    assert_eq!(c2.lambda_coeffs(), &[q(1, 1), q(0, 1), q(1, 1)]);
    // λ forced to 1: no certificate at any staged degree.
    let out = prove_nonnegative_on_polyhedron(
        &(t - 1),
        &hyps,
        Some(&ge(j, f.ctx.int(0))),
        &PolyhedronOpts::default().with_max_lambda_degree(0),
    )
    .unwrap();
    assert!(matches!(
        out,
        PolyhedronOutcome::Unknown(PolyhedronUnknown {
            degree: 3,
            lambda_degree: 0,
            pairwise: true,
            ..
        })
    ));
    // A single-stage search at the right size finds it too.
    let single = prove_nonnegative_on_polyhedron(
        &(t - 1),
        &hyps,
        Some(&ge(j, f.ctx.int(0))),
        &PolyhedronOpts::single(1, 1),
    )
    .unwrap();
    assert!(single.is_proved());
}

#[test]
fn refutation_points_are_exact_and_inside_the_set() {
    let f = Fixture::new();
    let (ctx, j, r, t) = (&f.ctx, &f.j, &f.r, &f.t);
    let cell = [
        r.clone(),
        ctx.rational(1, 2) - r,
        t.clone(),
        1 - t,
        (j * 2 + 1) * t - j * r - 1,
    ];
    let goal = t - ctx.rational(1, 2) - r;
    match prove_nonnegative_on_polyhedron(
        &goal,
        &cell,
        Some(&ge(j, ctx.int(2))),
        &PolyhedronOpts::default(),
    )
    .unwrap()
    {
        PolyhedronOutcome::Refuted { point, value, .. } => {
            assert!(value.is_negative());
            // Re-evaluate everything at the point.
            let subs = |e: &Ex| {
                let mut e = e.clone();
                for (v, val) in &point {
                    e = e.subs(v, &ctx.from_ratio(val.clone()));
                }
                e.eval().as_rational().unwrap()
            };
            assert_eq!(subs(&goal), value);
            for h in &cell {
                assert!(
                    !subs(h).is_negative(),
                    "point must satisfy every hypothesis"
                );
            }
            assert!(subs(j) >= q(2, 1));
        }
        other => panic!("{other:?}"),
    }
    // The emptiness question on a non-empty cell returns a point of it.
    match prove_polyhedron_empty(&cell, Some(&ge(j, ctx.int(2))), &PolyhedronOpts::default())
        .unwrap()
    {
        PolyhedronOutcome::Refuted { point, value, .. } => {
            assert_eq!(value, q(-1, 1));
            assert_eq!(point.len(), 3);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn emptiness_certificates_conclude_false() {
    let f = Fixture::new();
    let (ctx, j, r, t) = (&f.ctx, &f.j, &f.r, &f.t);
    let hyps = [
        r - ctx.rational(1, 2),
        (j * 2 + 1) * t - j * r - 1,
        ctx.rational(1, 4) - t,
    ];
    let c = proved(
        prove_polyhedron_empty(&hyps, Some(&ge(j, ctx.int(2))), &PolyhedronOpts::default())
            .unwrap(),
    );
    assert!(c.proves_emptiness());
    assert_eq!(c.goal().to_ex(), ctx.int(-1));
    let lean = c.to_lean("cell_empty").unwrap();
    assert!(lean.contains("    False := by\n"));
    assert!(!lean.contains("hg"));
    // Without a parameter as well: r ≥ 1 and r ≤ 0.
    let c = proved(prove_polyhedron_empty(&[r - 1, -r], None, &PolyhedronOpts::default()).unwrap());
    assert_eq!(c.to_string(), "-1 = h0 + h1; h0 = r - 1, h1 = -r");
}

#[test]
fn lean_steps_slot_into_an_existing_skeleton() {
    let f = Fixture::new();
    let (ctx, j, r, t) = (&f.ctx, &f.j, &f.r, &f.t);
    let names = PolyhedronLeanNames {
        hyps: &["e3", "e5"],
        param_nonneg: "hJ0",
        shift_nonneg: "hK0",
    };
    // λ ≠ 1 with a degree-2 λ: hg / hg' block and pJJ hint.
    let hyps = [t - r, t + j.powi(2) * r - j.powi(2) - 1];
    let c = f.prove(&(t - 1), &hyps, Some(0));
    let steps = c.lean_steps(&names, &LeanOpts::default()).unwrap();
    assert_eq!(
        steps.haves,
        vec![
            "have e3J := mul_nonneg hJ0 e3",
            "have e3JJ := mul_nonneg hJ0 e3J",
            "have pJJ := mul_nonneg hJ0 hJ0",
        ]
    );
    assert_eq!(steps.hints, vec!["e3JJ", "e5"]);
    assert_eq!(steps.lambda_hints, vec!["hJ0", "pJJ"]);
    assert_eq!(steps.lambda.as_deref(), Some("j ^ 2 + 1"));
    assert_eq!(
        steps.closing,
        vec![
            "have hg : (0 : ℝ) ≤ (j ^ 2 + 1) * (t - 1) := by",
            "  linarith only [e3JJ, e5]",
            "have hg' := nonneg_of_mul_nonneg_right hg (by linarith only [hJ0, pJJ])",
            "linarith only [hg']",
        ]
    );
    let block = steps.to_block("  ");
    assert!(block.starts_with("  have e3J := mul_nonneg hJ0 e3\n"));
    assert!(block.ends_with("  linarith only [hg']\n"));
    // A pairwise product with a K multiplier (j₀ = 2); pairwise terms carry
    // multipliers of degree ≤ 1 only, so j·(j − 2)·(r − r²) is out of reach.
    let c = f.prove(&((j - 2) * (r - r.powi(2))), &[r.clone(), 1 - r], Some(2));
    let steps = c.lean_steps(&names, &LeanOpts::default()).unwrap();
    assert_eq!(
        steps.haves,
        vec![
            "have e3xe5 := mul_nonneg e3 e5",
            "have e3xe5K := mul_nonneg hK0 e3xe5"
        ]
    );
    assert_eq!(steps.hints, vec!["e3xe5K"]);
    assert!(steps.lambda.is_none());
    let out = prove_nonnegative_on_polyhedron(
        &(j * (j - 2) * (r - r.powi(2))),
        &[r.clone(), 1 - r],
        Some(&ge(j, ctx.int(2))),
        &PolyhedronOpts::default(),
    )
    .unwrap();
    assert!(matches!(out, PolyhedronOutcome::Unknown { .. }));
    // Name count mismatch is an error, not a panic.
    let bad = PolyhedronLeanNames {
        hyps: &["e3"],
        param_nonneg: "hJ0",
        shift_nonneg: "hK0",
    };
    assert!(c.lean_steps(&bad, &LeanOpts::default()).is_err());
    let _ = ctx;
}

#[test]
fn certificates_cross_a_trust_boundary_as_json_and_are_reverified() {
    use symplex::certificates::{
        BoxBound, BoxCertificate, HalfLineCertificate, Ray, prove_nonnegative_on_box,
        prove_nonnegative_on_halfline,
    };
    let f = Fixture::new();
    let (ctx, j, r, t) = (&f.ctx, &f.j, &f.r, &f.t);
    // Polyhedron certificate with λ ≠ 1 and a K chain.
    let hyps = [t - r, r.clone()];
    let c = f.prove(&(j * (j - 2) * t), &hyps, Some(2));
    let json = c.to_json().unwrap();
    let other = Context::new();
    let back = PolyhedronCertificate::from_json(&other, &json).unwrap();
    assert_eq!(back.to_string(), c.to_string());
    assert_eq!(back.to_lean("x").unwrap(), c.to_lean("x").unwrap());
    assert_eq!(back.to_data(), c.to_data());
    // Tampering is detected: a changed weight, a changed goal, a bad index.
    let mut d = c.to_data();
    d.terms[0].weight = "2/1".to_string();
    assert!(PolyhedronCertificate::from_data(&other, &d).is_err());
    let mut d = c.to_data();
    d.goal = (t * 2).to_tree();
    assert!(PolyhedronCertificate::from_data(&other, &d).is_err());
    let mut d = c.to_data();
    d.terms[0].hyps = vec![7];
    assert!(PolyhedronCertificate::from_data(&other, &d).is_err());
    let mut d = c.to_data();
    d.lambda = vec!["1/0".to_string()];
    assert!(PolyhedronCertificate::from_data(&other, &d).is_err());
    assert!(PolyhedronCertificate::from_json(&other, "{").is_err());

    // Box certificate with a square factor.
    let x = ctx.symbol("x");
    let out = prove_nonnegative_on_box(
        &((&x - ctx.rational(1, 2)).powi(2) * (&x + 1)),
        &[BoxBound {
            var: x.clone(),
            lo: ctx.int(0),
            hi: ctx.int(1),
        }],
        2,
    )
    .unwrap();
    let bc = out.certificate().expect("box certificate");
    assert!(bc.square().is_some());
    let back = BoxCertificate::from_json(&other, &bc.to_json().unwrap()).unwrap();
    assert_eq!(back.to_string(), bc.to_string());
    assert!(back.square().is_some());
    let mut d = bc.to_data();
    d.terms[0].weight = "3/1".to_string();
    assert!(BoxCertificate::from_data(&other, &d).is_err());

    // Half-line certificate with a Pólya power.
    let out =
        prove_nonnegative_on_halfline(&(x.powi(2) - &x + 1), &x, &ctx.int(0), Ray::AtLeast, 4)
            .unwrap();
    let hc = out.certificate().expect("half-line certificate");
    assert!(hc.polya_power() > 0);
    let back = HalfLineCertificate::from_json(&other, &hc.to_json().unwrap()).unwrap();
    assert_eq!(back.coefficients(), hc.coefficients());
    assert_eq!(back.to_lean("h").unwrap(), hc.to_lean("h").unwrap());
    let mut d = hc.to_data();
    d.polya_power += 1;
    assert!(HalfLineCertificate::from_data(&other, &d).is_err());
    let mut d = hc.to_data();
    d.ray = "sideways".to_string();
    assert!(HalfLineCertificate::from_data(&other, &d).is_err());
}

#[test]
fn prover_reuses_hypotheses_and_matches_the_one_shot_functions() {
    use symplex::certificates::PolyhedronProver;
    let f = Fixture::new();
    let (ctx, j, r, t) = (&f.ctx, &f.j, &f.r, &f.t);
    let half = ctx.rational(1, 2);
    let hyps = [
        r.clone(),
        &half - r,
        t.clone(),
        1 - t,
        (j * 2 + 1) * t - j * r - 1,
    ];
    let prover =
        PolyhedronProver::new(&hyps, Some(&ge(j, ctx.int(2))), &PolyhedronOpts::default()).unwrap();
    assert_eq!(prover.hyps().len(), 5);
    assert_eq!(prover.gens(), &[r.clone(), t.clone(), j.clone()]);
    assert_eq!(prover.parameter(), Some(ge(j, ctx.int(2))));
    for goal in hyps
        .iter()
        .chain([&((j * 2 + 1) * t * 4 - j * r * 4 - r - 3)])
    {
        let a = prover.prove(goal).unwrap();
        let b = prove_nonnegative_on_polyhedron(
            goal,
            &hyps,
            Some(&ge(j, ctx.int(2))),
            &PolyhedronOpts::default(),
        )
        .unwrap();
        let (a, b) = (proved(a), proved(b));
        assert_eq!(a.to_string(), b.to_string());
        assert_eq!(a.to_lean("x").unwrap(), b.to_lean("x").unwrap());
    }
    // Emptiness through the prover, and a refutation naming the sample.
    assert!(!prover.prove_empty().unwrap().is_proved());
    match prover.prove(&(t - &half - r)).unwrap() {
        PolyhedronOutcome::Refuted {
            param_value, point, ..
        } => {
            assert_eq!(param_value, Some(q(2, 1)));
            assert_eq!(
                point.last().map(|(v, q)| (v.clone(), q.clone())),
                Some((j.clone(), q(2, 1)))
            );
        }
        other => panic!("{other:?}"),
    }
    // A goal with a foreign symbol is an error, not a silent extra variable.
    let e = prover.prove(&ctx.symbol("u")).unwrap_err().to_string();
    assert!(e.contains("not a variable of the hypotheses"), "{e}");
}

/// `prove_poly` takes an exact polynomial directly.  Its generators may be
/// a permutation (a tool's `(j, r, t)` order against the prover's sorted
/// `(r, t, j)`) or a subset of the prover's; the certificate, its Lean and
/// `used_hyps` are identical to the expression route.  A `MultiPoly` goal
/// reaches it through `Poly::from_multipoly` without ever being an `Ex`.
#[test]
fn prove_poly_accepts_permuted_and_partial_generators() {
    use symplex::certificates::PolyhedronProver;
    use symplex::poly_ex::Poly;
    let f = Fixture::new();
    let (ctx, j, r, t) = (&f.ctx, &f.j, &f.r, &f.t);
    let hyps = [
        r.clone(),
        ctx.rational(1, 2) - r,
        t.clone(),
        1 - t,
        (j * 2 + 1) * t - j * r - 1,
    ];
    let prover =
        PolyhedronProver::new(&hyps, Some(&ge(j, ctx.int(2))), &PolyhedronOpts::default()).unwrap();
    let goal = (j * 2 + 1) * t * 4 - j * r * 4 - r - 3;
    let via_ex = proved(prover.prove(&goal).unwrap());
    // (j, r, t): the tool's order, parameter first.
    let permuted = Poly::new(&goal, &[j, r, t]).unwrap();
    let via_poly = proved(prover.prove_poly(&permuted).unwrap());
    assert_eq!(via_poly.to_string(), via_ex.to_string());
    assert_eq!(via_poly.to_lean("x").unwrap(), via_ex.to_lean("x").unwrap());
    assert_eq!(via_poly.used_hyps(), via_ex.used_hyps());
    // A subset of the generators: `1 − t + r` (from `r ≥ 0`, `1 − t ≥ 0`)
    // does not mention `j`, and is given in `(t, r)` order.
    let partial = Poly::new(&(1 - t + r), &[t, r]).unwrap();
    let a = proved(prover.prove_poly(&partial).unwrap());
    let b = proved(prover.prove(&(1 - t + r)).unwrap());
    assert_eq!(a.to_string(), b.to_string());
    assert_eq!(a.used_hyps(), vec![0, 3]);
    // And a false goal is refuted the same way through both routes.
    let bad = Poly::new(&(t - r), &[t, r]).unwrap();
    assert!(matches!(
        prover.prove_poly(&bad).unwrap(),
        PolyhedronOutcome::Refuted { .. }
    ));
    // Straight from a MultiPoly in (j, r, t) order.
    let [mj, mr, mt]: [MultiPoly; 3] = [
        MultiPoly::var(3, 0),
        MultiPoly::var(3, 1),
        MultiPoly::var(3, 2),
    ];
    let two_j_plus_1 = mj.scale(&qi(2)) + 1;
    let mp = two_j_plus_1.mul(&mt).scale(&qi(4)) - mj.mul(&mr).scale(&qi(4)) - mr.clone() - 3;
    let from_mp = Poly::from_multipoly(ctx, &[j, r, t], &mp).unwrap();
    assert_eq!(from_mp.to_ex().expand(), goal.expand());
    let c = proved(prover.prove_poly(&from_mp).unwrap());
    assert_eq!(c.to_string(), via_ex.to_string());
    // Foreign generator: an error.
    let foreign = Poly::new(&ctx.symbol("u"), &[&ctx.symbol("u")]).unwrap();
    let e = prover.prove_poly(&foreign).unwrap_err().to_string();
    assert!(e.contains("not a variable of the hypotheses"), "{e}");
}

/// A goal polynomial may carry generators that occur in none of its terms
/// — a tool's polynomial ring has the parameter `j` on every row, whether
/// or not the row mentions it — even when the prover does not know them.
/// Only a generator that *occurs* may be foreign (0.9.1).
#[test]
fn prove_poly_ignores_generators_that_occur_in_no_term() {
    use symplex::certificates::PolyhedronProver;
    use symplex::poly_ex::Poly;
    let f = Fixture::new();
    let (ctx, j, r, t) = (&f.ctx, &f.j, &f.r, &f.t);
    // No parameter: the prover's generators are (r, t) only.
    let hyps = [r.clone(), 1 - r, t - r];
    let prover = PolyhedronProver::new(&hyps, None, &PolyhedronOpts::default()).unwrap();
    assert_eq!(prover.gens(), &[r.clone(), t.clone()]);
    let goal = t * 2 - r;
    let via_ex = proved(prover.prove(&goal).unwrap());
    // The same goal over (j, r, t): `j` has exponent 0 in every term.
    let spare = Poly::new(&goal, &[j, r, t]).unwrap();
    assert_eq!(spare.gens().len(), 3);
    let via_poly = proved(prover.prove_poly(&spare).unwrap());
    assert_eq!(via_poly.to_string(), via_ex.to_string());
    assert_eq!(via_poly.to_lean("x").unwrap(), via_ex.to_lean("x").unwrap());
    assert_eq!(via_poly.used_hyps(), via_ex.used_hyps());
    assert_eq!(via_poly.goal().gens(), prover.gens());
    // Spare generators in any position, and several of them.
    let u = ctx.symbol("u");
    let spare2 = Poly::new(&goal, &[t, &u, r, j]).unwrap();
    let c = proved(prover.prove_poly(&spare2).unwrap());
    assert_eq!(c.to_string(), via_ex.to_string());
    // A false goal is refuted through the spare ring too.
    let bad = Poly::new(&(r - t), &[j, r, t]).unwrap();
    assert!(matches!(
        prover.prove_poly(&bad).unwrap(),
        PolyhedronOutcome::Refuted { .. }
    ));
    // A constant goal over entirely foreign generators is fine (nothing occurs).
    let one = Poly::new(&ctx.int(1), &[j, &u]).unwrap();
    assert!(prover.prove_poly(&one).unwrap().is_proved());
    // But a foreign generator that occurs is still an error.
    let used = Poly::new(&(t * 2 - r + j), &[j, r, t]).unwrap();
    let e = prover.prove_poly(&used).unwrap_err();
    assert!(matches!(e, SymplexError::InvalidArgument { .. }), "{e}");
    assert!(
        e.to_string().contains("not a variable of the hypotheses"),
        "{e}"
    );
    // With a parameter the prover knows `j`; a spare `u` is still ignored.
    let cell = [
        r.clone(),
        ctx.rational(1, 2) - r,
        t.clone(),
        1 - t,
        (j * 2 + 1) * t - j * r - 1,
    ];
    let pj =
        PolyhedronProver::new(&cell, Some(&ge(j, ctx.int(2))), &PolyhedronOpts::default()).unwrap();
    let g = (j * 2 + 1) * t * 4 - j * r * 4 - r - 3;
    let a = proved(pj.prove(&g).unwrap());
    let b = proved(
        pj.prove_poly(&Poly::new(&g, &[&u, j, r, t]).unwrap())
            .unwrap(),
    );
    assert_eq!(a.to_string(), b.to_string());
}

// ═══════════════════════════════════════════════════════════════════════════
// Budgets (0.9.1)
// ═══════════════════════════════════════════════════════════════════════════

fn budget_hit(out: &PolyhedronOutcome) -> Option<symplex::certificates::BudgetHit> {
    match out {
        PolyhedronOutcome::Unknown(u) => u.budget_exhausted,
        _ => None,
    }
}

#[test]
fn pivot_budget_stops_the_search_quickly_and_names_the_limit() {
    use std::time::{Duration, Instant};
    use symplex::certificates::BudgetHit;
    let f = Fixture::new();
    let (ctx, j, r, t) = (&f.ctx, &f.j, &f.r, &f.t);
    let half = ctx.rational(1, 2);
    let cell = [
        r.clone(),
        &half - r,
        t.clone(),
        1 - t,
        (j * 2 + 1) * t - j * r - 1,
    ];
    let goal = (j * 2 + 1) * t * 4 - j * r * 4 - r - 3;
    let needs_lambda = [t - r, t + j * r - j - 1];
    let cases: [(&Ex, &[Ex], i64); 2] = [(&goal, &cell, 2), (&(t - 1), &needs_lambda, 0)];
    for (goal, hyps, j0) in cases {
        let bound = ge(j, ctx.int(j0));
        let param = Some(&bound);
        let free = proved(
            prove_nonnegative_on_polyhedron(goal, hyps, param, &PolyhedronOpts::default()).unwrap(),
        );
        // Three pivots do not get through the first stage LP.
        let started = Instant::now();
        let out = prove_nonnegative_on_polyhedron(
            goal,
            hyps,
            param,
            &PolyhedronOpts::default().with_max_pivots(3),
        )
        .unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
        assert_eq!(budget_hit(&out), Some(BudgetHit::MaxPivots), "{out:?}");
        let PolyhedronOutcome::Unknown(u) = &out else {
            panic!("{out:?}");
        };
        assert!(
            u.to_string().ends_with("; budget exhausted: max_pivots"),
            "{u}"
        );
        // The first stage is the one that was interrupted.
        assert_eq!((u.degree, u.lambda_degree, u.pairwise), (1, 0, false));
        // A budget the search fits inside gives exactly the unbudgeted certificate.
        let roomy = proved(
            prove_nonnegative_on_polyhedron(
                goal,
                hyps,
                param,
                &PolyhedronOpts::default()
                    .with_max_pivots(1_000_000)
                    .with_time_limit(Duration::from_secs(600)),
            )
            .unwrap(),
        );
        assert_eq!(roomy.to_string(), free.to_string());
        assert_eq!(roomy.to_lean("x").unwrap(), free.to_lean("x").unwrap());
        // Monotone in the cap: exhausted up to some cap, then always the
        // same certificate.
        let mut finished = false;
        for cap in (0..400).step_by(7) {
            let out = prove_nonnegative_on_polyhedron(
                goal,
                hyps,
                param,
                &PolyhedronOpts::default().with_max_pivots(cap),
            )
            .unwrap();
            match out {
                PolyhedronOutcome::Unknown(u) => {
                    assert_eq!(u.budget_exhausted, Some(BudgetHit::MaxPivots));
                    assert!(
                        !finished,
                        "exhausted at {cap} after finishing at a smaller cap"
                    );
                }
                PolyhedronOutcome::Proved(c) => {
                    finished = true;
                    assert_eq!(c.to_string(), free.to_string());
                }
                other => panic!("{other:?}"),
            }
        }
        assert!(finished, "400 pivots suffice for these goals");
    }
}

#[test]
fn zero_time_limit_fires_at_the_first_check_and_a_prover_is_reusable() {
    use std::time::{Duration, Instant};
    use symplex::certificates::{BudgetHit, PolyhedronProver};
    let f = Fixture::new();
    let (ctx, j, r, t) = (&f.ctx, &f.j, &f.r, &f.t);
    let half = ctx.rational(1, 2);
    let cell = [
        r.clone(),
        &half - r,
        t.clone(),
        1 - t,
        (j * 2 + 1) * t - j * r - 1,
    ];
    let goal = (j * 2 + 1) * t * 4 - j * r * 4 - r - 3;
    let bound = ge(j, ctx.int(2));
    let param = Some(&bound);
    let out = prove_nonnegative_on_polyhedron(
        &goal,
        &cell,
        param,
        &PolyhedronOpts::default().with_time_limit(Duration::ZERO),
    )
    .unwrap();
    assert_eq!(budget_hit(&out), Some(BudgetHit::Deadline), "{out:?}");
    assert!(
        out.to_string().ends_with("; budget exhausted: deadline"),
        "{out}"
    );
    // An absolute deadline in the past, too — and `prove_empty` as well.
    let past = Instant::now() - Duration::from_millis(1);
    let prover_past =
        PolyhedronProver::new(&cell, param, &PolyhedronOpts::default().with_deadline(past))
            .unwrap();
    assert_eq!(
        budget_hit(&prover_past.prove(&goal).unwrap()),
        Some(BudgetHit::Deadline)
    );
    assert_eq!(
        budget_hit(&prover_past.prove_empty().unwrap()),
        Some(BudgetHit::Deadline)
    );
    assert_eq!(
        budget_hit(
            &prover_past
                .prove_poly(&Poly::new(&goal, &[r, t, j]).unwrap())
                .unwrap()
        ),
        Some(BudgetHit::Deadline)
    );
    // The earlier of `deadline` and `time_limit` applies.
    let both = PolyhedronOpts::default()
        .with_deadline(past)
        .with_time_limit(Duration::from_secs(600));
    assert_eq!(
        budget_hit(&prove_nonnegative_on_polyhedron(&goal, &cell, param, &both).unwrap()),
        Some(BudgetHit::Deadline)
    );
    // A time limit is per call: a prover built once proves goal after goal,
    // each with a fresh allowance, and every certificate is the unbudgeted one.
    let prover = PolyhedronProver::new(
        &cell,
        param,
        &PolyhedronOpts::default().with_time_limit(Duration::from_secs(120)),
    )
    .unwrap();
    let plain = PolyhedronProver::new(&cell, param, &PolyhedronOpts::default()).unwrap();
    for g in cell.iter().chain([&goal]) {
        let a = proved(prover.prove(g).unwrap());
        let b = proved(plain.prove(g).unwrap());
        assert_eq!(a.to_string(), b.to_string());
    }
    assert!(!prover.prove_empty().unwrap().is_proved());
    // Refutations are unchanged under a roomy budget.
    assert!(prover.prove(&(t - &half - r)).unwrap().is_refuted());
    // The options record the budget.
    assert_eq!(prover.opts().time_limit, Some(Duration::from_secs(120)));
    assert_eq!(prover.opts().deadline, None);
    assert_eq!(prover.opts().max_pivots, None);
    assert_eq!(PolyhedronOpts::single(2, 1).max_pivots, None);
    let _ = j;
}

/// A refuted goal under a pivot cap: exhausted, then (once the cap covers
/// the first stage *and* the sample LPs) refuted with the same point.
#[test]
fn refutation_lps_share_the_pivot_budget() {
    use symplex::certificates::BudgetHit;
    let f = Fixture::new();
    let (ctx, j, r, t) = (&f.ctx, &f.j, &f.r, &f.t);
    let half = ctx.rational(1, 2);
    let cell = [
        r.clone(),
        &half - r,
        t.clone(),
        1 - t,
        (j * 2 + 1) * t - j * r - 1,
    ];
    let bad = t - &half - r;
    let bound = ge(j, ctx.int(2));
    let param = Some(&bound);
    let free =
        prove_nonnegative_on_polyhedron(&bad, &cell, param, &PolyhedronOpts::default()).unwrap();
    let PolyhedronOutcome::Refuted {
        point,
        value,
        param_value,
        ..
    } = &free
    else {
        panic!("{free:?}");
    };
    let mut finished = false;
    for cap in 0..300usize {
        let out = prove_nonnegative_on_polyhedron(
            &bad,
            &cell,
            param,
            &PolyhedronOpts::default().with_max_pivots(cap),
        )
        .unwrap();
        match out {
            PolyhedronOutcome::Unknown(u) => {
                assert_eq!(u.budget_exhausted, Some(BudgetHit::MaxPivots));
                assert!(
                    !finished,
                    "exhausted at {cap} after refuting at a smaller cap"
                );
            }
            PolyhedronOutcome::Refuted {
                point: p,
                value: v,
                param_value: pv,
                ..
            } => {
                finished = true;
                assert_eq!((&p, &v, &pv), (point, value, param_value));
            }
            other => panic!("{other:?}"),
        }
    }
    assert!(finished);
}

#[test]
fn symbol_text_renders_the_parameter_as_a_cast_everywhere() {
    let f = Fixture::new();
    let (j, r, t) = (&f.j, &f.r, &f.t);
    let hyps = [t - r, t + j.powi(2) * r - j.powi(2) - 1];
    let c = f.prove(&(t - 1), &hyps, Some(0));
    let opts = LeanOpts::default().with_symbol_text("j", "(j : ℝ)");
    let steps = c
        .lean_steps(
            &PolyhedronLeanNames {
                hyps: &["e0", "e1"],
                param_nonneg: "hJ0",
                shift_nonneg: "hK0",
            },
            &opts,
        )
        .unwrap();
    assert_eq!(steps.lambda.as_deref(), Some("(j : ℝ) ^ 2 + 1"));
    assert_eq!(
        steps.closing[0],
        "have hg : (0 : ℝ) ≤ ((j : ℝ) ^ 2 + 1) * (t - 1) := by"
    );
    // Hypothesis names containing `J` are untouched (no blanket replace).
    assert_eq!(steps.haves[0], "have e0J := mul_nonneg hJ0 e0");
    let lean = c.to_lean_with("cast", &opts).unwrap();
    assert!(
        lean.starts_with("theorem cast (r t j : ℝ) (hj : (0 : ℝ) ≤ (j : ℝ))"),
        "{lean}"
    );
    assert!(
        lean.contains("(h1 : 0 ≤ r * (j : ℝ) ^ 2 - (j : ℝ) ^ 2 + t - 1)"),
        "{lean}"
    );
    assert!(
        lean.contains("have hJ0 : (0 : ℝ) ≤ (j : ℝ) := by linarith"),
        "{lean}"
    );
    // `single_fraction` puts a rational function over one denominator.
    let e = ((-8 * j - 2) / (7 * j + 4)).expand();
    assert_eq!(
        e.to_lean().unwrap(),
        "-(8 * j / (7 * j + 4)) - 2 / (7 * j + 4)"
    );
    assert_eq!(
        e.to_lean_with(&LeanOpts::default().with_single_fraction(true))
            .unwrap(),
        "(-(8 * j) - 2) / (7 * j + 4)"
    );
    assert_eq!(
        e.to_lean_with(
            &LeanOpts::default()
                .with_single_fraction(true)
                .with_symbol_text("j", "(j : ℝ)")
        )
        .unwrap(),
        "(-(8 * (j : ℝ)) - 2) / (7 * (j : ℝ) + 4)"
    );
}

#[test]
fn to_block_wraps_to_the_mathlib_width_and_keeps_by_on_the_have_line() {
    let f = Fixture::new();
    let (j, r, t) = (&f.j, &f.r, &f.t);
    // λ = 1 + j² is forced; a long parameter rendering and long hypothesis
    // names push `have hg : … := by` past 100 columns.
    let hyps = [t - r, t + j.powi(2) * r - j.powi(2) - 1];
    let c = f.prove(&(t - 1), &hyps, Some(0));
    let names = PolyhedronLeanNames {
        hyps: &[
            "e_first_hypothesis_of_the_leaf",
            "e_second_hypothesis_of_the_leaf",
        ],
        param_nonneg: "hJ0_nonneg_parameter",
        shift_nonneg: "hK0",
    };
    let opts = LeanOpts::default().with_symbol_text(
        "j",
        "((j_parameter_of_the_ladder_frame_index_variable : ℕ) : ℝ)",
    );
    let steps = c.lean_steps(&names, &opts).unwrap();
    assert!(
        steps.closing[0].chars().count() + 4 > 100,
        "{}",
        steps.closing[0]
    );
    let block = steps.to_block("    ");
    assert!(block.lines().all(|l| l.chars().count() <= 100), "{block}");
    for l in block.lines() {
        assert!(
            !l.trim_start().starts_with(":="),
            "continuation line starts with :=\n{block}"
        );
    }
    // The wrapped `have hg` statement ends with `:= by` on its last line,
    // and the `linarith only […]` that proves it follows, deeper indented.
    let lines: Vec<&str> = block.lines().collect();
    let idx = lines.iter().position(|l| l.contains("have hg :")).unwrap();
    assert!(lines[idx].starts_with("    have hg :"));
    let end = (idx..lines.len())
        .find(|&i| lines[i].ends_with(":= by"))
        .unwrap();
    assert!(
        lines[end + 1].starts_with("      linarith only ["),
        "{block}"
    );
    assert_eq!(
        steps.to_block_width("", 10_000).lines().count(),
        steps.haves.len() + steps.closing.len()
    );
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }

    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
}

#[test]
fn random_constructed_identities_are_recovered_and_false_goals_never_proved() {
    let f = Fixture::new();
    let (ctx, j, r, t) = (&f.ctx, &f.j, &f.r, &f.t);
    for seed in 1..=25u64 {
        let mut g = Lcg(seed * 7919);
        // Random affine hypotheses with j-dependent coefficients.
        let m = g.range(2, 4) as usize;
        let hyps: Vec<Ex> = (0..m)
            .map(|_| {
                (j * g.range(-2, 2) + g.range(-3, 3)) * r
                    + (j * g.range(-2, 2) + g.range(-3, 3)) * t
                    + j * g.range(0, 2)
                    + g.range(-2, 4)
            })
            .collect();
        // A goal built as (1 + c·j)⁻¹ · (Σ μₖ j^a (j−j₀)^b hₖ + μ₀) is not
        // polynomial in general, so build the *certificate side*
        // directly: goal := Σ μₖ·multₖ·hₖ + μ₀ with λ = 1; the search must
        // find some certificate (not necessarily this one).
        let j0 = g.range(0, 3);
        let mut goal = ctx.int(g.range(0, 3));
        for h in &hyps {
            let (a, b) = (g.range(0, 1), g.range(0, 1));
            let w = ctx.rational(g.range(0, 3), g.range(1, 2));
            goal += w * j.powi(a) * (j - j0).powi(b) * h;
        }
        let goal = goal.expand();
        if goal.is_zero_structural() {
            continue;
        }
        let out = prove_nonnegative_on_polyhedron(
            &goal,
            &hyps,
            Some(&ge(j, ctx.int(j0))),
            &PolyhedronOpts::default(),
        )
        .unwrap();
        let c = proved(out);
        let lean = c.to_lean(&format!("random_{seed}")).unwrap();
        assert!(lean.lines().all(|l| l.chars().count() <= 100), "{lean}");
        // A goal that is negative somewhere on a non-empty cell must never
        // be Proved: subtract a large constant.
        let bad = &goal - 1000;
        match prove_nonnegative_on_polyhedron(
            &bad,
            &hyps,
            Some(&ge(j, ctx.int(j0))),
            &PolyhedronOpts::default(),
        )
        .unwrap()
        {
            PolyhedronOutcome::Proved(c) => {
                // Only possible if the cell is empty for every j ≥ j₀ — then the
                // emptiness certificate must exist too.
                assert!(c.verify());
                assert!(
                    prove_polyhedron_empty(
                        &hyps,
                        Some(&ge(j, ctx.int(j0))),
                        &PolyhedronOpts::default()
                    )
                    .unwrap()
                    .is_proved(),
                    "seed {seed}"
                );
            }
            PolyhedronOutcome::Refuted { value, .. } => assert!(value.is_negative()),
            PolyhedronOutcome::Unknown { .. } => {}
        }
    }
}

/// A prover builds each stage's basis on the first goal that reaches it,
/// so it can be shared by reference across threads: the certificates are
/// the same as a sequential run's, byte for byte, whichever thread first
/// materialised a stage (the basis is a pure function of the hypotheses).
/// Goals here are chosen so that some settle in the first stage, some
/// need `λ`, and one is false, exercising three stages and the refutation.
#[test]
fn shared_prover_across_threads_matches_sequential_certificates() {
    use symplex::certificates::PolyhedronProver;
    let f = Fixture::new();
    let (ctx, j, r, t) = (&f.ctx, &f.j, &f.r, &f.t);
    let half = ctx.rational(1, 2);
    let mut hyps = vec![r.clone(), &half - r, t.clone(), 1 - t];
    for k in 1..=4i64 {
        hyps.push((j * k + 1) * t - j * r - ctx.rational(k, 2 * k + 1));
    }
    let mut goals: Vec<Ex> = hyps.clone();
    for k in 1..=4i64 {
        goals.push((j * k + 1) * t * 3 - j * r * 3 - ctx.rational(3 * k, 2 * k + 1) + &half - r);
    }
    goals.push(t - &half - r); // false on the cell
    goals.push(j * t - j * r); // needs the parameter multiplier
    let prover =
        PolyhedronProver::new(&hyps, Some(&ge(j, ctx.int(1))), &PolyhedronOpts::default()).unwrap();
    let render = |out: PolyhedronOutcome| match out {
        PolyhedronOutcome::Proved(c) => format!("proved {}", c.to_lean("g").unwrap()),
        PolyhedronOutcome::Refuted { value, .. } => format!("refuted {value}"),
        PolyhedronOutcome::Unknown(u) => format!("unknown {u}"),
    };
    // Sequential, on a fresh prover (every stage built by this thread).
    let sequential: Vec<String> = goals
        .iter()
        .map(|g| render(prover.prove(g).unwrap()))
        .collect();
    assert!(sequential.iter().any(|s| s.starts_with("proved")));
    assert!(sequential.iter().any(|s| s.starts_with("refuted")));
    // Concurrent, on a second prover shared by reference: the threads race
    // to build the stages.
    let shared =
        PolyhedronProver::new(&hyps, Some(&ge(j, ctx.int(1))), &PolyhedronOpts::default()).unwrap();
    let concurrent: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = goals
            .iter()
            .map(|g| scope.spawn(|| render(shared.prove(g).unwrap())))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(concurrent, sequential);
}

#[test]
fn many_hypotheses_certify_quickly_with_staging() {
    // A 2-D cell with ten j-dependent facets: every facet of the cell is
    // implied by the others plus the box; all ten goals certify.
    let f = Fixture::new();
    let (ctx, j, r, t) = (&f.ctx, &f.j, &f.r, &f.t);
    let half = ctx.rational(1, 2);
    let mut hyps = vec![r.clone(), &half - r, t.clone(), 1 - t];
    for k in 1..=6i64 {
        hyps.push((j * k + 1) * t - j * r - ctx.rational(k, 2 * k + 1));
    }
    let start = std::time::Instant::now();
    for k in 1..=6i64 {
        let goal = (j * k + 1) * t * 3 - j * r * 3 - ctx.rational(3 * k, 2 * k + 1) + &half - r;
        let c = f.prove(&goal, &hyps, Some(1));
        assert!(c.lambda_is_one() || c.verify());
    }
    assert!(
        start.elapsed().as_secs() < 20,
        "six certificates over ten facets took {:?}",
        start.elapsed()
    );
}
