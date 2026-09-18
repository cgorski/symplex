//! Polynomial views and rational normal forms (symplex 0.3).
//!
//! Demonstrates:
//! - `Ex::as_poly` / `Poly::new`: an expression as a sparse polynomial in
//!   explicit generators with symbolic (parametric) coefficients;
//! - `terms`, `coeff_monomial`, `degree_list`, `leading_coeff`,
//!   `all_coeffs`, exact evaluation at rationals and partial evaluation;
//! - `monomial_basis` + `coefficient_matrix` + `linsolve_matrix` to solve
//!   for an exact linear certificate `goal = Σ λᵢ hᵢ`;
//! - `Ex::ratsimp` on a nested fraction and on a `solve` result;
//! - `poly_is_nonnegative_on` / `poly_is_positive_on`: exact sign of a
//!   polynomial on an interval.
//!
//! Run with: `cargo run --example polynomials`

use symplex::matrix::Matrix;
use symplex::poly_ex::Poly;
use symplex::polysys::linsolve_matrix;
use symplex::prelude::*;

fn main() -> Result<(), SymplexError> {
    println!("=== Polynomial views and rational normal forms ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; r, f, j, x, y);

    // ── 1. A polynomial with parameters ──────────────────────────────────
    println!("--- Polynomial view with symbolic coefficients ---");
    let e = &j * r.powi(2) + (&j + 1) * &r * &f + 3;
    println!("e = {e}");
    println!("degree in r: {:?}", e.degree(&r));
    println!(
        "coeffs in r: {:?}",
        e.coeffs(&r)
            .map(|cs| cs.iter().map(|c| c.to_string()).collect::<Vec<_>>())
    );
    println!(
        "leading coefficient in r: {}",
        e.leading_coeff(&r).map_or("?".into(), |c| c.to_string())
    );

    let p = e
        .as_poly(&[&r, &f])
        .ok_or_else(|| SymplexError::InvalidArgument {
            operation: "as_poly",
            reason: "not polynomial".into(),
        })?;
    println!("{p}");
    println!("terms (lex-descending in [r, f]):");
    for (mono, coeff) in p.terms() {
        println!("    r^{} f^{}  ·  {coeff}", mono[0], mono[1]);
    }
    println!("coefficient of r·f: {}", p.coeff_monomial(&[1, 1])?);
    println!(
        "degree list: {:?}, total degree: {:?}",
        p.degree_list(),
        p.total_degree()
    );
    println!("leading coefficient (lex): {}", p.leading_coeff());
    let univariate = e
        .as_poly(&[&r])
        .ok_or_else(|| SymplexError::InvalidArgument {
            operation: "as_poly",
            reason: "not polynomial".into(),
        })?;
    println!(
        "all_coeffs in r (dense, highest first): {:?}",
        univariate
            .all_coeffs()
            .map(|cs| cs.iter().map(|c| c.to_string()).collect::<Vec<_>>())
    );

    // ── 2. Exact evaluation ─────────────────────────────────────────────
    println!("\n--- Exact evaluation ---");
    let q = (x.powi(2) * &y - &y / 2 + ctx.rational(1, 3))
        .as_poly(&[&x, &y])
        .ok_or_else(|| SymplexError::InvalidArgument {
            operation: "as_poly",
            reason: "not polynomial".into(),
        })?;
    let v = q.eval(&[&ctx.rational(3, 2), &ctx.rational(-4, 5)])?;
    println!("{q} at (3/2, −4/5) = {v}");
    let partial = q.eval_gen(&x, &ctx.int(2))?;
    println!("{q} with x = 2  →  {partial}");
    let d = q.derivative(&y)?;
    println!("∂/∂y {q} = {}", d.to_ex());

    // ── 3. A linear certificate ─────────────────────────────────────────
    println!("\n--- Linear certificate: goal = Σ λᵢ hᵢ ---");
    let h1 = (&x + 1).as_poly(&[&x]);
    let h2 = (x.powi(2) - 1).as_poly(&[&x]);
    let goal = (&x + 1).powi(2).as_poly(&[&x]);
    let (Some(h1), Some(h2), Some(goal)) = (h1, h2, goal) else {
        return Err(SymplexError::InvalidArgument {
            operation: "as_poly",
            reason: "not polynomial".into(),
        });
    };
    let basis = Poly::monomial_basis(&[&h1, &h2, &goal])?;
    let m = Poly::coefficient_matrix(&[&h1, &h2], &basis)?;
    let rhs: Vec<Ex> = basis
        .iter()
        .map(|mono| goal.coeff_monomial(mono))
        .collect::<Result<_, _>>()?;
    println!("monomial basis: {basis:?}");
    println!("coefficient matrix (rows = monomials, cols = [h1, h2]):\n{m}");
    match linsolve_matrix(&m, &Matrix::col_vector(rhs))? {
        LinearSolution::Unique(pairs) => {
            let lam: Vec<String> = pairs.iter().map(|(_, v)| v.to_string()).collect();
            println!("(x + 1)² = {}·(x + 1) + {}·(x² − 1)", lam[0], lam[1]);
        }
        other => println!("no unique certificate: {other:?}"),
    }

    // ── 4. Rational normal form ─────────────────────────────────────────
    println!("\n--- ratsimp ---");
    let nested = ctx.int(1) / (&x + ctx.int(1) / &y) + ctx.int(1) / (&y + ctx.int(1) / &x);
    println!("{nested}  →  {}", nested.ratsimp());
    let leftover = (j.powi(2) - 1) / (&j * 2) - (&j - 1) / 2;
    println!("{leftover}  →  {}", leftover.ratsimp());
    // sin(x) and cos(x) are opaque indeterminates: the difference of squares
    // cancels, but no trigonometric identity is applied.
    let trig = (x.sin().powi(2) - x.cos().powi(2)) / (x.sin() - x.cos());
    println!("{trig}  →  {}", trig.ratsimp());

    // (3r − 1)/(j + 1) = (r + 1)/(2j), solved for r: the result is a single fraction.
    let eqn = (&r * 3 - 1) / (&j + 1) - (&r + 1) / (&j * 2);
    let sols = eqn.solve(&r)?;
    println!("solve({eqn} = 0, r) = {}", sols[0]);

    // ── 5. Sign of a polynomial on an interval ──────────────────────────
    println!("\n--- Exact sign on an interval ---");
    let (ninf, inf) = (ctx.neg_infinity(), ctx.infinity());
    let sq = x.powi(2) - &x * 2 + 1;
    println!(
        "{sq} ≥ 0 on ℝ: {:?};  > 0 on ℝ: {:?}",
        sq.poly_is_nonnegative_on(&x, &ninf, &inf),
        sq.poly_is_positive_on(&x, &ninf, &inf)
    );
    let cubic = x.powi(3) - &x;
    println!(
        "{cubic} ≥ 0 on [2, ∞): {:?};  on [−2, ∞): {:?};  on [−1, 0]: {:?}",
        cubic.poly_is_nonnegative_on(&x, &ctx.int(2), &inf),
        cubic.poly_is_nonnegative_on(&x, &ctx.int(-2), &inf),
        cubic.poly_is_nonnegative_on(&x, &ctx.int(-1), &ctx.int(0))
    );
    println!(
        "x² + 1 > 0 on ℝ: {:?}",
        (x.powi(2) + 1).poly_is_positive_on(&x, &ninf, &inf)
    );

    println!("\nDone.");
    Ok(())
}
