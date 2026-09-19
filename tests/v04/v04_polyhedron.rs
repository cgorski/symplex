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
    PolyhedronCertificate, PolyhedronLeanNames, PolyhedronOpts, PolyhedronOutcome,
    prove_nonnegative_on_polyhedron, prove_polyhedron_empty,
};
use symplex::lean::LeanOpts;
use symplex::linprog::{q, qi};
use symplex::prelude::*;

const COMPILED: &str = include_str!("../fixtures/polyhedron_certificates.lean");

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
        let lo = j0.map(|v| self.ctx.int(v));
        proved(
            prove_nonnegative_on_polyhedron(
                goal,
                hyps,
                lo.as_ref().map(|lo| (&self.j, lo)),
                &PolyhedronOpts::default(),
            )
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
                prove_polyhedron_empty(hyps, Some((j, &ctx.int(j0))), &PolyhedronOpts::default())
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
    let (lhs, rhs) = c.identity();
    assert!((lhs - rhs).expand().is_zero_structural());
    // A degree-2 λ is found when needed and not otherwise.
    let sq = [t - r, t + j.powi(2) * r - j.powi(2) - 1];
    let c2 = f.prove(&(t - 1), &sq, Some(0));
    assert_eq!(c2.lambda_coeffs(), &[q(1, 1), q(0, 1), q(1, 1)]);
    // λ forced to 1: no certificate at any staged degree.
    let out = prove_nonnegative_on_polyhedron(
        &(t - 1),
        &hyps,
        Some((j, &f.ctx.int(0))),
        &PolyhedronOpts {
            max_lambda_degree: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        out,
        PolyhedronOutcome::Unknown {
            degree: 3,
            lambda_degree: 0,
            pairwise: true
        }
    ));
    // A single-stage search at the right size finds it too.
    let single = prove_nonnegative_on_polyhedron(
        &(t - 1),
        &hyps,
        Some((j, &f.ctx.int(0))),
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
        Some((j, &ctx.int(2))),
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
    match prove_polyhedron_empty(&cell, Some((j, &ctx.int(2))), &PolyhedronOpts::default()).unwrap()
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
        prove_polyhedron_empty(&hyps, Some((j, &ctx.int(2))), &PolyhedronOpts::default()).unwrap(),
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
        Some((j, &ctx.int(2))),
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
        Certificate, HalfLineCertificate, Ray, prove_nonnegative_on_box,
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
    d.terms[0].3 = "2/1".to_string();
    assert!(PolyhedronCertificate::from_data(&other, &d).is_err());
    let mut d = c.to_data();
    d.goal = (t * 2).to_tree();
    assert!(PolyhedronCertificate::from_data(&other, &d).is_err());
    let mut d = c.to_data();
    d.terms[0].0 = vec![7];
    assert!(PolyhedronCertificate::from_data(&other, &d).is_err());
    let mut d = c.to_data();
    d.lambda = vec!["1/0".to_string()];
    assert!(PolyhedronCertificate::from_data(&other, &d).is_err());
    assert!(PolyhedronCertificate::from_json(&other, "{").is_err());

    // Box certificate with a square factor.
    let x = ctx.symbol("x");
    let out = prove_nonnegative_on_box(
        &((&x - ctx.rational(1, 2)).powi(2) * (&x + 1)),
        &[(x.clone(), ctx.int(0), ctx.int(1))],
        2,
    )
    .unwrap();
    let bc = out.certificate().expect("box certificate");
    assert!(bc.square().is_some());
    let back = Certificate::from_json(&other, &bc.to_json().unwrap()).unwrap();
    assert_eq!(back.to_string(), bc.to_string());
    assert!(back.square().is_some());
    let mut d = bc.to_data();
    d.terms[0].2 = "3/1".to_string();
    assert!(Certificate::from_data(&other, &d).is_err());

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
        PolyhedronProver::new(&hyps, Some((j, &ctx.int(2))), &PolyhedronOpts::default()).unwrap();
    assert_eq!(prover.hyps().len(), 5);
    assert_eq!(prover.gens(), &[r.clone(), t.clone(), j.clone()]);
    assert_eq!(prover.parameter(), Some((j, &ctx.int(2))));
    for goal in hyps
        .iter()
        .chain([&((j * 2 + 1) * t * 4 - j * r * 4 - r - 3)])
    {
        let a = prover.prove(goal).unwrap();
        let b = prove_nonnegative_on_polyhedron(
            goal,
            &hyps,
            Some((j, &ctx.int(2))),
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
        PolyhedronProver::new(&hyps, Some((j, &ctx.int(2))), &PolyhedronOpts::default()).unwrap();
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
            Some((j, &ctx.int(j0))),
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
            Some((j, &ctx.int(j0))),
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
                        Some((j, &ctx.int(j0))),
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
