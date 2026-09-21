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
//!   Mathlib project to check it);
//! - `prove_nonnegative_on_halfline` / `prove_nonnegative_on_reals`: shift
//!   certificates (`p(a + k)` with non-negative coefficients), Pólya
//!   multipliers `(1 + k)^N`, square factors for interior double zeros, and
//!   a two-case proof for the whole real line.
//!
//! Run with: `cargo run --example certificates_to_lean [out.lean]`

use symplex::certificates::{
    BoxBound, BoxOutcome, HalfLineOutcome, Ray, prove_nonnegative_on_box,
    prove_nonnegative_on_halfline, prove_nonnegative_on_reals,
};
use symplex::prelude::*;

/// A named claim `0 ≤ goal` on a box, with the certificate degree to search.
type Case = (&'static str, Ex, Vec<BoxBound>, u32);

fn main() {
    let ctx = Context::new();
    let (r, f, x, y) = (
        ctx.symbol("r"),
        ctx.symbol("f"),
        ctx.symbol("x"),
        ctx.symbol("y"),
    );
    let unit = |v: &Ex| BoxBound {
        var: v.clone(),
        lo: ctx.int(0),
        hi: ctx.int(1),
    };

    let cases: Vec<Case> = vec![
        (
            "quarter_bound",
            ctx.rational(1, 4) - (&r - &f / 2).powi(2),
            vec![
                BoxBound {
                    var: r.clone(),
                    lo: ctx.int(0),
                    hi: ctx.rational(1, 2),
                },
                unit(&f),
            ],
            2,
        ),
        ("x_one_minus_x", &x * (1 - &x), vec![unit(&x)], 2),
        ("one_minus_xy", 1 - &x * &y, vec![unit(&x), unit(&y)], 2),
        (
            "cubic_on_interval",
            (&x - 1) * (&x - 2) * (&x - 3),
            vec![BoxBound {
                var: x.clone(),
                lo: ctx.int(3),
                hi: ctx.int(10),
            }],
            3,
        ),
        (
            "interior_zero",
            (&x - 1).powi(2) + (&y - 1).powi(2),
            vec![
                BoxBound {
                    var: x.clone(),
                    lo: ctx.int(0),
                    hi: ctx.int(2),
                },
                BoxBound {
                    var: y.clone(),
                    lo: ctx.int(0),
                    hi: ctx.int(2),
                },
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
        for BoxBound { var, lo, hi } in bounds {
            println!("    {lo} ≤ {var} ≤ {hi}");
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
            BoxOutcome::Refuted { point, value, .. } => {
                let pt: Vec<String> = point.iter().map(|(v, q)| format!("{v} = {q}")).collect();
                println!("    REFUTED: goal = {value} at ({})", pt.join(", "));
            }
            BoxOutcome::Unknown(u) => {
                println!(
                    "    UNKNOWN at degree {} (the goal has a zero inside the box, so no Handelman certificate exists)",
                    u.degree
                );
            }
        }
        println!();
    }

    // ── Half-lines and the real line (univariate) ────────────────────────
    let j = ctx.symbol("j");
    println!("=== Half-line certificates ===\n");
    let half: Vec<(&str, Ex, Ex, Ray)> = vec![
        ("shift_only", (&j - 1) * (&j - 3), ctx.int(3), Ray::AtLeast),
        (
            "polya_needed",
            &j.powi(2) - &j + 1,
            ctx.int(0),
            Ray::AtLeast,
        ),
        (
            "square_inside",
            (&j - 5).powi(2) * (&j.powi(2) + 1),
            ctx.int(3),
            Ray::AtLeast,
        ),
        ("at_most", (3 - &j) * (5 - &j), ctx.int(3), Ray::AtMost),
        ("false_halfline", &j - 4, ctx.int(3), Ray::AtLeast),
    ];
    for (name, goal, a, ray) in &half {
        let dom = match ray {
            Ray::AtLeast => format!("{j} ≥ {a}"),
            Ray::AtMost => format!("{j} ≤ {a}"),
        };
        println!("--- {name}: 0 ≤ {goal} for {dom}");
        match prove_nonnegative_on_halfline(goal, &j, a, ray.clone(), 12).unwrap() {
            HalfLineOutcome::Proved(c) => {
                println!(
                    "    PROVED: Pólya exponent {}, square factor {}",
                    c.polya_power(),
                    c.square().map_or("none".to_string(), ToString::to_string)
                );
                println!("    {c}");
                println!("    re-verified exactly: {}", c.verify());
                lean.push_str(&c.to_lean(name).unwrap());
                lean.push('\n');
            }
            HalfLineOutcome::Refuted { point, value, .. } => {
                println!("    REFUTED: goal = {value} at {j} = {}", point[0].1);
            }
            HalfLineOutcome::Unknown(u) => {
                println!("    UNKNOWN within Pólya exponent {}", u.max_polya_power);
            }
        }
        println!();
    }
    println!(
        "--- pos_quadratic_reals: 0 ≤ {} on all of ℝ",
        &j.powi(2) - &j + 1
    );
    let reals = prove_nonnegative_on_reals(&(&j.powi(2) - &j + 1), &j, &ctx.int(0), 12)
        .unwrap()
        .unwrap();
    println!(
        "    PROVED by cases at 0; re-verified exactly: {}\n",
        reals.verify()
    );
    lean.push_str(&reals.to_lean("pos_quadratic_reals").unwrap());
    lean.push('\n');

    println!("=== Generated Lean 4 ===\n{lean}");
    if let Some(path) = std::env::args().nth(1) {
        std::fs::write(&path, &lean).expect("write Lean file");
        println!("written to {path}");
    }
}
