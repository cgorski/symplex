//! Certificates on a parametric polyhedron (symplex 0.4).
//!
//! A decision procedure over a polytope whose facets move with a real
//! parameter `j ≥ j₀` has to show, for each cell, that a goal inequality
//! holds on the cell for *every* `j` — or that the cell is empty.  Both are
//! exact Farkas-type identities with a polynomial multiplier `λ(j)` on the
//! goal, found by a staged exact LP and re-verified with polynomial
//! arithmetic.  Every certificate exports as a Lean 4 / Mathlib theorem, or
//! as the bare `have …`/`linarith only […]` lines for an existing proof.
//!
//! Run with: `cargo run --example polyhedron_certificates`

use symplex::certificates::{
    PolyhedronLeanNames, PolyhedronOpts, PolyhedronOutcome, prove_nonnegative_on_polyhedron,
    prove_polyhedron_empty,
};
use symplex::lean::LeanOpts;
use symplex::prelude::*;

fn main() {
    println!("=== Certificates on a parametric polyhedron ===\n");
    let ctx = Context::new();
    let (j, r, t) = (ctx.symbol("j"), ctx.symbol("r"), ctx.symbol("t"));
    let j0 = ctx.int(2);
    let half = ctx.rational(1, 2);

    // A cell of the (r, t) plane whose facets depend on j:
    //   0 ≤ r ≤ 1/2,  0 ≤ t ≤ 1,  (2j + 1)·t ≥ j·r + 1  (a j-dependent facet).
    let hyps = [
        r.clone(),
        &half - &r,
        t.clone(),
        1 - &t,
        (&j * 2 + 1) * &t - &j * &r - 1,
    ];

    // ── 1. A goal on the cell: λ = 1 suffices ─────────────────────────────
    let goal = (&j * 2 + 1) * &t * 4 - &j * &r * 4 - &r - 3;
    let out =
        prove_nonnegative_on_polyhedron(&goal, &hyps, Some((&j, &j0)), &PolyhedronOpts::default())
            .unwrap();
    match &out {
        PolyhedronOutcome::Proved(c) => {
            println!("certificate: {c}");
            assert!(c.verify());
            println!(
                "λ = {}, terms = {}, degree = {}\n",
                c.lambda(),
                c.terms().len(),
                c.degree()
            );
            println!("{}", c.to_lean("cell_goal").unwrap());
        }
        other => println!("{other:?}"),
    }

    // ── 1b. A goal that needs the multiplier λ(j) ────────────────────────
    // On { t ≥ r,  t + j·r ≥ j + 1 } the goal t − 1 ≥ 0 holds for every j ≥ 0,
    // but its Farkas multipliers are 1/(1 + j) and j/(1 + j): no polynomial
    // combination exists until the goal is multiplied by λ(j) = 1 + j.
    let needs_lambda = [&t - &r, &t + &j * &r - &j - 1];
    let out = prove_nonnegative_on_polyhedron(
        &(&t - 1),
        &needs_lambda,
        Some((&j, &ctx.int(0))),
        &PolyhedronOpts::default(),
    )
    .unwrap();
    match &out {
        PolyhedronOutcome::Proved(c) => {
            println!("certificate: {c}");
            assert!(c.verify());
            assert!(!c.lambda_is_one());
            println!("λ = {}\n", c.lambda());
            println!("{}", c.to_lean("needs_lambda").unwrap());
        }
        other => println!("{other:?}"),
    }
    // λ = 1 alone cannot do it:
    let out = prove_nonnegative_on_polyhedron(
        &(&t - 1),
        &needs_lambda,
        Some((&j, &ctx.int(0))),
        &PolyhedronOpts::single(3, 0),
    )
    .unwrap();
    println!(
        "with λ forced to 1: {}",
        match out {
            PolyhedronOutcome::Unknown { .. } => "no certificate (as expected)".to_string(),
            other => format!("{other:?}"),
        }
    );

    // ── 2. A false claim: refuted with an exact point ────────────────────
    let false_goal = &t - &half - &r;
    match prove_nonnegative_on_polyhedron(
        &false_goal,
        &hyps,
        Some((&j, &j0)),
        &PolyhedronOpts::default(),
    )
    .unwrap()
    {
        PolyhedronOutcome::Refuted { point, value, .. } => {
            let shown: Vec<String> = point.iter().map(|(v, q)| format!("{v} = {q}")).collect();
            println!("refuted: goal = {value} at {}", shown.join(", "));
        }
        other => println!("{other:?}"),
    }

    // ── 3. An empty cell for every j ≥ 2 ─────────────────────────────────
    // r ≥ 1/2 together with (2j + 1)·t ≥ j·r + 1 and t ≤ 1/4 has no point once
    // j ≥ 2 (the facet forces t ≥ (j/2 + 1)/(2j + 1) > 1/4 … exactly at j = 2 the
    // bound is 2/5 > 1/4).
    let empty_hyps = [
        &r - &half,
        (&j * 2 + 1) * &t - &j * &r - 1,
        ctx.rational(1, 4) - &t,
    ];
    match prove_polyhedron_empty(&empty_hyps, Some((&j, &j0)), &PolyhedronOpts::default()).unwrap()
    {
        PolyhedronOutcome::Proved(c) => {
            println!("\nemptiness certificate: {c}");
            let steps = c
                .lean_steps(
                    &PolyhedronLeanNames {
                        hyps: &["e0", "e1", "e2"],
                        param_nonneg: "hJ0",
                        shift_nonneg: "hK0",
                    },
                    &LeanOpts::default(),
                )
                .unwrap();
            println!(
                "\nproof steps for an existing skeleton:\n{}",
                steps.to_block("  ")
            );
            println!("{}", c.to_lean("cell_empty").unwrap());
        }
        other => println!("{other:?}"),
    }

    // ── 4. The same machinery without a parameter: plain Farkas ─────────
    let fixed = [r.clone(), 1 - &r, &t - &r];
    let out =
        prove_nonnegative_on_polyhedron(&(&t * 2 - &r), &fixed, None, &PolyhedronOpts::default())
            .unwrap();
    println!("\nno parameter: {}", out.certificate().unwrap());
}
