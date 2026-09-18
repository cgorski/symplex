//! Exact non-negativity certificates on a box, exported as Lean 4 / Mathlib
//! theorems (symplex 0.3.2).
//!
//! Demonstrates:
//! - `prove_nonnegative_on_box`: a Handelman certificate
//!   `goal = Σ λₖ · Π (xᵢ − lᵢ)^a (uᵢ − xᵢ)^b` found by exact LP and
//!   re-verified with exact polynomial arithmetic;
//! - exact refutation (`BoxOutcome::Refuted`) and the honest
//!   `BoxOutcome::Unknown` when the goal touches zero inside the box;
//! - `Certificate::to_lean`: a theorem whose proof is `linarith`/`nlinarith`
//!   over exactly the products of the certificate — the generated file
//!   compiles against Mathlib as is (run `lake env lean <file>` in any
//!   Mathlib project to check it).
//!
//! Run with: `cargo run --example certificates_to_lean [out.lean]`

use symplex::certificates::{BoxOutcome, prove_nonnegative_on_box};
use symplex::prelude::*;

/// A named claim `0 ≤ goal` on a box, with the certificate degree to search.
type Case = (&'static str, Ex, Vec<(Ex, Ex, Ex)>, u32);

fn main() {
    let ctx = Context::new();
    let (r, f, x, y) = (
        ctx.symbol("r"),
        ctx.symbol("f"),
        ctx.symbol("x"),
        ctx.symbol("y"),
    );
    let unit = |v: &Ex| (v.clone(), ctx.int(0), ctx.int(1));

    let cases: Vec<Case> = vec![
        (
            "quarter_bound",
            ctx.rational(1, 4) - (&r - &f / 2).powi(2),
            vec![(r.clone(), ctx.int(0), ctx.rational(1, 2)), unit(&f)],
            2,
        ),
        ("x_one_minus_x", &x * (1 - &x), vec![unit(&x)], 2),
        ("one_minus_xy", 1 - &x * &y, vec![unit(&x), unit(&y)], 2),
        (
            "cubic_on_interval",
            (&x - 1) * (&x - 2) * (&x - 3),
            vec![(x.clone(), ctx.int(3), ctx.int(10))],
            3,
        ),
        (
            "interior_zero",
            (&x - 1).powi(2) + (&y - 1).powi(2),
            vec![
                (x.clone(), ctx.int(0), ctx.int(2)),
                (y.clone(), ctx.int(0), ctx.int(2)),
            ],
            2,
        ),
        (
            "false_claim",
            &x * &y - ctx.rational(1, 2),
            vec![unit(&x), unit(&y)],
            2,
        ),
    ];

    let mut lean = String::from("import Mathlib\n\n");
    println!("=== Handelman certificates on boxes ===\n");
    for (name, goal, bounds, degree) in &cases {
        println!("--- {name}: 0 ≤ {goal}");
        for (v, lo, hi) in bounds {
            println!("    {lo} ≤ {v} ≤ {hi}");
        }
        match prove_nonnegative_on_box(goal, bounds, *degree).unwrap() {
            BoxOutcome::Proved(cert) => {
                println!(
                    "    PROVED with {} products of degree ≤ {}:",
                    cert.terms().len(),
                    cert.degree()
                );
                println!("    {cert}");
                println!("    re-verified exactly: {}", cert.verify());
                lean.push_str(&cert.to_lean(name).unwrap());
                lean.push('\n');
            }
            BoxOutcome::Refuted { point, value } => {
                let pt: Vec<String> = point.iter().map(ToString::to_string).collect();
                println!("    REFUTED: goal = {value} at ({})", pt.join(", "));
            }
            BoxOutcome::Unknown { degree, .. } => {
                println!(
                    "    UNKNOWN at degree {degree} (the goal has a zero inside the box, so no Handelman certificate exists)"
                );
            }
        }
        println!();
    }

    println!("=== Generated Lean 4 ===\n{lean}");
    if let Some(path) = std::env::args().nth(1) {
        std::fs::write(&path, &lean).expect("write Lean file");
        println!("written to {path}");
    }
}
