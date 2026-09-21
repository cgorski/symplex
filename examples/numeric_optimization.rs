//! Numerical optimisation and root bracketing (symplex 0.3).
//!
//! Demonstrates `symplex::optimize`:
//! - bracketing the root of a transcendental equation defined symbolically
//!   (`find_root_bracket`, Brent–Dekker) and checking it against `newton_root`
//!   with a symbolically differentiated derivative;
//! - Nelder–Mead on Rosenbrock's function, both as a closure (`nelder_mead`)
//!   and as an `Ex` (`minimize_numeric_with`);
//! - Brent's scalar minimiser (`minimize_scalar`) versus golden-section search;
//! - differential evolution on Rastrigin's function (`differential_evolution`,
//!   `minimize_global_numeric`), deterministic for a given seed;
//! - exact rational least squares (`Ex::poly_fit_points`) beside the
//!   floating-point fit (`poly_fit`, `linear_fit`);
//! - the trapezoidal rule (`trapezoid`) on sampled data.
//!
//! Run with: `cargo run --example numeric_optimization`

use std::f64::consts::PI;
use symplex::optimize::{
    DeOpts, LinearFit, MinimizeOpts, RootOpts, differential_evolution, eval_poly, golden_section,
    linear_fit, minimize_scalar, nelder_mead, newton_root, poly_fit, trapezoid,
};
use symplex::prelude::*;

fn main() -> Result<(), SymplexError> {
    println!("=== Numerical Optimisation ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    // ── 1. Root bracketing of a symbolic transcendental equation ─────────
    println!("--- Root bracketing ---");
    // Kepler's equation E − e·sin E = M with eccentricity 0.3 and mean anomaly 1.
    let kepler = &x - x.sin() * ctx.rational(3, 10) - 1;
    let e_anomaly = kepler.find_root_bracket(&x, 0.0, PI)?;
    println!("Kepler  E − 0.3 sin E = 1   ⇒  E = {e_anomaly:.15}");
    let residual = kepler.compile(&["x"])?.call(&[e_anomaly]);
    println!("        residual {residual:.2e}");

    // The same root by Newton, with the derivative computed symbolically.
    let f = kepler.compile(&["x"])?;
    let df = kepler.diff(&x).compile(&["x"])?;
    let newton = newton_root(|t| f(&[t]), |t| df(&[t]), 1.0, &RootOpts::default())?;
    println!(
        "        Newton from 1.0 agrees to {:.1e}",
        (newton - e_anomaly).abs()
    );

    // cos x = x, with a loose tolerance.
    let loose = RootOpts {
        xtol: 1e-6,
        ..RootOpts::default()
    };
    let dottie = (x.cos() - &x).find_root_bracket_with(&x, 0.0, 1.0, &loose)?;
    println!("cos x = x  ⇒  x ≈ {dottie:.7}  (xtol = 1e-6)");

    // A bracket without a sign change is rejected rather than guessed.
    match (&x.powi(2) + 1).find_root_bracket(&x, -1.0, 1.0) {
        Err(e) => println!("x² + 1 on [−1, 1]: {e}"),
        Ok(r) => println!("unexpected root {r}"),
    }
    println!();

    // ── 2. Nelder–Mead on Rosenbrock ─────────────────────────────────────
    println!("--- Nelder–Mead ---");
    let rosen = |p: &[f64]| (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2);
    let opts = MinimizeOpts {
        max_iter: 2000,
        ..MinimizeOpts::default()
    };
    let r = nelder_mead(rosen, &[-1.2, 1.0], &opts)?;
    println!(
        "Rosenbrock from (−1.2, 1): x = ({:.6}, {:.6}), f = {:.2e}",
        r.x[0], r.x[1], r.fun
    );
    println!(
        "        {} iterations, {} evaluations, converged = {}",
        r.iterations, r.evaluations, r.converged
    );

    // Same problem defined symbolically.
    let rosen_ex = (1 - &x).powi(2) + 100 * (&y - &x.powi(2)).powi(2);
    let r = rosen_ex.minimize_numeric_with(&[&x, &y], &[-1.2, 1.0], &opts)?;
    println!(
        "as an Ex `{rosen_ex}`:\n        x = ({:.6}, {:.6}), f = {:.2e}",
        r.x[0], r.x[1], r.fun
    );
    println!();

    // ── 3. Scalar minimisation ───────────────────────────────────────────
    println!("--- Scalar minimisation ---");
    let g = |t: f64| t * t.ln();
    let brent = minimize_scalar(g, 0.1, 2.0, &MinimizeOpts::default())?;
    let golden = golden_section(g, 0.1, 2.0, &MinimizeOpts::default())?;
    println!(
        "x ln x on [0.1, 2]:  Brent  x = {:.10}, f = {:.12}",
        brent.x, brent.value
    );
    println!(
        "                     golden x = {:.10}, f = {:.12}",
        golden.x, golden.value
    );
    println!(
        "                     exact  x = 1/e = {:.10}",
        (-1.0f64).exp()
    );

    // Γ(x) has its minimum on (0, ∞) near 1.4616.
    let m = x.gamma().minimize_scalar_numeric(&x, 1.0, 2.0)?;
    println!("Γ(x) on [1, 2]:      x = {:.8}, Γ = {:.10}", m.x, m.value);
    println!();

    // ── 4. Differential evolution on Rastrigin ───────────────────────────
    println!("--- Differential evolution ---");
    let rastrigin = |p: &[f64]| {
        10.0 * p.len() as f64
            + p.iter()
                .map(|v| v * v - 10.0 * (2.0 * PI * v).cos())
                .sum::<f64>()
    };
    let bounds = [Interval::closed(-5.12, 5.12), Interval::closed(-5.12, 5.12)];
    let r = differential_evolution(rastrigin, &bounds, &DeOpts::default())?;
    println!(
        "Rastrigin 2-D, seed 0:  x = ({:+.2e}, {:+.2e}), f = {:.2e}",
        r.x[0], r.x[1], r.fun
    );
    println!(
        "        {} generations, {} evaluations, converged = {}",
        r.iterations, r.evaluations, r.converged
    );
    let seeded = DeOpts {
        seed: 7,
        ..DeOpts::default()
    };
    let again = differential_evolution(rastrigin, &bounds, &seeded)?;
    let repeat = differential_evolution(rastrigin, &bounds, &seeded)?;
    println!(
        "seed 7: f = {:.2e}; identical on rerun = {}",
        again.fun,
        again == repeat
    );

    // Symbolic form of the same function.
    let two_pi = ctx.pi() * 2;
    let ras_ex =
        20 + &x.powi(2) + &y.powi(2) - 10 * (&two_pi * &x).cos() - 10 * (&two_pi * &y).cos();
    let r = ras_ex.minimize_global_numeric(&[&x, &y], &bounds, &DeOpts::default())?;
    println!(
        "as an Ex: f = {:.2e} at ({:+.2e}, {:+.2e})",
        r.fun, r.x[0], r.x[1]
    );
    println!();

    // ── 5. Polynomial fitting: exact vs floating point ───────────────────
    println!("--- Least-squares polynomial fitting ---");
    // Samples of x²/3 − x/2 + 1/7 at six integers: exact recovery over ℚ.
    let pts: Vec<(Ex, Ex)> = (-2..=3)
        .map(|i| {
            let xi = ctx.int(i);
            let yi =
                &xi.powi(2) * ctx.rational(1, 3) - &xi * ctx.rational(1, 2) + ctx.rational(1, 7);
            (xi, yi.eval())
        })
        .collect();
    let exact = Ex::poly_fit_points(&ctx, &pts, &x, 2)?;
    println!("exact fit through six rational points: {exact}");

    let xs: Vec<f64> = (-2..=3).map(f64::from).collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|t| t * t / 3.0 - t / 2.0 + 1.0 / 7.0)
        .collect();
    let c = poly_fit(&xs, &ys, 2)?;
    println!(
        "float fit (ascending coefficients): [{:.12}, {:.12}, {:.12}]",
        c[0], c[1], c[2]
    );
    println!(
        "        1/7 = {:.12}, −1/2, 1/3 = {:.12}",
        1.0 / 7.0,
        1.0 / 3.0
    );
    println!("        p(10) = {:.6}", eval_poly(&c, 10.0));

    // Inconsistent data: the exact least-squares line.
    let noisy = [
        (ctx.int(0), ctx.int(1)),
        (ctx.int(1), ctx.int(0)),
        (ctx.int(2), ctx.int(4)),
        (ctx.int(3), ctx.int(2)),
    ];
    let line = Ex::poly_fit_points(&ctx, &noisy, &x, 1)?;
    println!("exact least-squares line through (0,1) (1,0) (2,4) (3,2): {line}");
    let LinearFit { slope, intercept } = linear_fit(&[0.0, 1.0, 2.0, 3.0], &[1.0, 0.0, 4.0, 2.0])?;
    println!("float linear_fit: slope = {slope:.12}, intercept = {intercept:.12}");
    println!();

    // ── 6. Trapezoidal rule ──────────────────────────────────────────────
    println!("--- Trapezoidal rule ---");
    let grid: Vec<f64> = (0..=1000).map(|i| i as f64 / 1000.0).collect();
    let samples: Vec<f64> = grid.iter().map(|t| t * t).collect();
    let area = trapezoid(&samples, &grid)?;
    println!(
        "∫₀¹ x² dx ≈ {area:.9} (1001 points); exact 1/3 = {:.9}",
        1.0 / 3.0
    );
    let sin_samples: Vec<f64> = grid.iter().map(|t| (PI * t).sin()).collect();
    println!(
        "∫₀¹ sin(πx) dx ≈ {:.9}; exact 2/π = {:.9}",
        trapezoid(&sin_samples, &grid)?,
        2.0 / PI
    );

    Ok(())
}
