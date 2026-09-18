//! Exact linear programming over ℚ (symplex 0.3).
//!
//! Demonstrates:
//! - `LpProblem`: a two-phase simplex with Bland's rule running entirely on
//!   `Ratio<BigInt>`, so optima, shadow prices and certificates are exact;
//! - shadow prices (`duals`) and the strong-duality identity `cᵀx* = yᵀb`;
//! - an infeasible program with its Farkas certificate, checked explicitly;
//! - `feasible_nonneg`: an exact "is `b` a non-negative combination of
//!   these vectors?" query with `1/3`, `1/7` coefficients that a floating
//!   point solver can only answer up to a tolerance;
//! - `linprog_matrix`: feeding `Matrix` data with `Ex` entries and reading
//!   the optimum back as `Ex`.
//!
//! Run with: `cargo run --example exact_lp`

use num_traits::{Signed, Zero};
use symplex::linprog::{LpProblem, LpStatus, Objective, Q, feasible_nonneg, linprog_matrix, q, qi};
use symplex::prelude::*;

fn show(v: &[Q]) -> String {
    let parts: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("({})", parts.join(", "))
}

fn main() {
    println!("=== Exact linear programming ===\n");

    // ── 1. A textbook LP with exact optimum and shadow prices ────────────
    println!("--- max 5x + 4y  s.t.  6x + 4y ≤ 24,  x + 2y ≤ 6,  −x + y ≤ 1,  y ≤ 2 ---");
    let rows = [
        (vec![qi(6), qi(4)], qi(24)),
        (vec![qi(1), qi(2)], qi(6)),
        (vec![qi(-1), qi(1)], qi(1)),
        (vec![qi(0), qi(1)], qi(2)),
    ];
    let mut p = LpProblem::maximize(vec![qi(5), qi(4)]);
    for (row, rhs) in &rows {
        p = p.le(row.clone(), rhs.clone());
    }
    let sol = p.solve().unwrap();
    println!("status     : {:?}", sol.status);
    println!("x*         : {}", show(&sol.x));
    println!("objective  : {}", sol.objective.clone().unwrap());
    println!("duals y    : {}", show(&sol.duals));
    // Strong duality with x ≥ 0 and rⱼxⱼ = 0: cᵀx* = yᵀb.
    let ytb: Q = rows.iter().zip(&sol.duals).map(|((_, b), y)| b * y).sum();
    println!("yᵀb        : {ytb}   (= cᵀx*, strong duality)");
    assert_eq!(Some(ytb), sol.objective);
    // Complementary slackness: yᵢ·(aᵢx* − bᵢ) = 0.
    for (i, (row, b)) in rows.iter().enumerate() {
        let slack: Q = row.iter().zip(&sol.x).map(|(a, x)| a * x).sum::<Q>() - b;
        assert!((&slack * &sol.duals[i]).is_zero());
        println!("row {i}: slack = {slack:>4}, y = {}", sol.duals[i]);
    }

    // ── 2. Fractional data, fractional optimum ──────────────────────────
    println!("\n--- min x + y  s.t.  x + 2y ≥ 1,  3x + y ≥ 1 ---");
    let sol = LpProblem::minimize(vec![qi(1), qi(1)])
        .ge(vec![qi(1), qi(2)], qi(1))
        .ge(vec![qi(3), qi(1)], qi(1))
        .solve()
        .unwrap();
    println!(
        "x* = {}, objective = {}",
        show(&sol.x),
        sol.objective.clone().unwrap()
    );
    let ctx = Context::new();
    println!("as Ex      : {:?}", sol.x_ex(&ctx));

    // ── 3. Infeasible: Farkas certificate ───────────────────────────────
    println!("\n--- x + y ≤ 1  and  x + y ≥ 2  (x, y ≥ 0) ---");
    let sol = LpProblem::minimize(vec![qi(1), qi(1)])
        .le(vec![qi(1), qi(1)], qi(1))
        .ge(vec![qi(1), qi(1)], qi(2))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::Infeasible);
    let y = sol.farkas.clone().unwrap();
    println!("status     : {:?}", sol.status);
    println!("farkas y   : {}", show(&y));
    // Certificate: y₀ ≥ 0 (≤ row), y₁ ≤ 0 (≥ row), g = Aᵀy ≥ 0 so that the
    // infimum over x ≥ 0 is 0, and 0 > yᵀb.
    let g0 = &y[0] + &y[1];
    let ytb = &y[0] + &(&y[1] * qi(2));
    println!("Aᵀy        : ({g0}, {g0})  ≥ 0");
    println!("yᵀb        : {ytb}  < 0   ⇒ no feasible point exists");
    assert!(!g0.is_negative() && ytb.is_negative());

    // ── 4. Exact non-negative combination (certificate search) ──────────
    println!("\n--- find μ ≥ 0 with  μ₁/3 + μ₂/7 + 2μ₃/5 = 1,  μ₁ + μ₂ + μ₃ = 4 ---");
    let a = vec![vec![q(1, 3), q(1, 7), q(2, 5)], vec![qi(1), qi(1), qi(1)]];
    let b = vec![qi(1), qi(4)];
    match feasible_nonneg(&a, &b).unwrap() {
        Some(mu) => {
            println!("μ = {}", show(&mu));
            for (row, rhs) in a.iter().zip(&b) {
                let lhs: Q = row.iter().zip(&mu).map(|(c, m)| c * m).sum();
                assert_eq!(&lhs, rhs);
            }
            println!("verified exactly: A·μ = b");
        }
        None => println!("no non-negative combination exists"),
    }
    // A target outside the cone has none.
    let none = feasible_nonneg(&[vec![qi(1), qi(1)]], &[qi(-1)]).unwrap();
    println!("x + y = −1 with x, y ≥ 0: {none:?}");

    // ── 5. Matrix interface with Ex entries ─────────────────────────────
    println!("\n--- linprog_matrix with Ex data ---");
    let c = matrix![ctx, [3], [2]];
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(1)],
        vec![ctx.int(1), &ctx.int(1) + &ctx.int(2)], // constant expressions are folded
    ])
    .unwrap();
    let bm = matrix![ctx, [4], [6]];
    let sol = linprog_matrix(Objective::Maximize, &c, Some(&a), Some(&bm), None, None).unwrap();
    println!("A =\n{a}");
    println!(
        "max cᵀx  → x* = {:?}, objective {}",
        sol.x_ex(&ctx),
        sol.objective.unwrap()
    );

    // Symbolic entries are rejected rather than silently approximated.
    let x = ctx.symbol("x");
    let bad = Matrix::new(vec![vec![x, ctx.int(1)]]).unwrap();
    let err = linprog_matrix(
        Objective::Minimize,
        &c,
        Some(&bad),
        Some(&matrix![ctx, [1]]),
        None,
        None,
    )
    .unwrap_err();
    println!("symbolic entry → {err}");

    // ── 6. Unbounded ────────────────────────────────────────────────────
    println!("\n--- max x + y  s.t.  x − y ≤ 1 ---");
    let sol = LpProblem::maximize(vec![qi(1), qi(1)])
        .le(vec![qi(1), qi(-1)], qi(1))
        .solve()
        .unwrap();
    println!("status     : {:?}", sol.status);
    assert_eq!(sol.status, LpStatus::Unbounded);

    println!("\nDone.");
}
