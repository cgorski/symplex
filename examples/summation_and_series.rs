//! Symbolic summation, products, convergence and series (symplex 0.2).
//!
//! Demonstrates:
//! - `summation` / `try_summation`: Faulhaber sums of any degree,
//!   geometric and binomial sums, telescoping, Gosper's algorithm,
//!   p-series (`ζ(2m)` exact, `zeta(p)` for odd `p`, Catalan's constant),
//!   and power-series recognition (`Σ xᵏ/k! = eˣ`);
//! - `product_over`: finite and infinite products;
//! - `is_convergent` / `is_absolutely_convergent` (three-valued);
//! - `hypergeometric_ratio`;
//! - `series_at_infinity` (asymptotic expansions);
//! - `FormalPowerSeries`: lazy exact coefficients, general terms,
//!   arithmetic, composition, inversion and reversion;
//! - `Ex`-based finite differences (Fornberg weights).
//!
//! Run with: `cargo run --example summation_and_series`

use symplex::prelude::*;

fn main() {
    println!("=== Summation, Products and Series ===\n");

    let ctx = Context::new();
    let k = ctx.symbol("k");
    let x = ctx.symbol("x");
    // A positive integer upper bound lets the engine use n! and 2^n freely.
    let n = ctx.symbol_with("n", &[Assumption::Integer, Assumption::Positive]);
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let inf = ctx.infinity();

    let show = |label: &str, value: &Ex| println!("{label:<34} = {value}");

    // ── 1. Finite sums with closed forms ────────────────────────────────
    println!("--- Finite sums ---");
    show("Σ_{k=1}^{n} k", &k.summation(&k, &one, &n));
    show("Σ_{k=1}^{n} k²", &k.powi(2).summation(&k, &one, &n));
    show(
        "Σ_{k=1}^{n} k⁵  (Faulhaber)",
        &k.powi(5).summation(&k, &one, &n),
    );
    show(
        "Σ_{k=0}^{n} 2ᵏ",
        &ctx.int(2).pow(&k).summation(&k, &zero, &n),
    );
    show(
        "Σ_{k=1}^{n} 1/(k(k+1)) (telescoping)",
        &(1 / (&k * (&k + 1))).summation(&k, &one, &n),
    );
    show(
        "Σ_{k=0}^{n} C(n,k)",
        &n.binomial(&k).summation(&k, &zero, &n),
    );
    show(
        "Σ_{k=0}^{n} k·C(n,k)",
        &(&k * &n.binomial(&k)).summation(&k, &zero, &n),
    );
    show(
        "Σ_{k=0}^{n} k·2ᵏ  (Gosper)",
        &(&k * &ctx.int(2).pow(&k)).summation(&k, &zero, &n),
    );
    show("Σ_{k=1}^{n} 1/k", &(1 / &k).summation(&k, &one, &n));
    // A geometric sum with a symbolic ratio is only valid for x ≠ 1, and
    // the engine says so with a Piecewise instead of dividing by zero.
    show("Σ_{k=0}^{n} xᵏ", &x.pow(&k).summation(&k, &zero, &n));

    // ── 2. Infinite series ──────────────────────────────────────────────
    println!("\n--- Infinite series ---");
    show(
        "Σ 1/k²           (Basel)",
        &k.powi(-2).summation(&k, &one, &inf),
    );
    show("Σ 1/k⁴", &k.powi(-4).summation(&k, &one, &inf));
    // Odd zeta values have no known closed form; the answer is symbolic.
    show("Σ 1/k³", &k.powi(-3).summation(&k, &one, &inf));
    show(
        "Σ (−1)ᵏ/(2k+1)²  (Catalan)",
        &(ctx.int(-1).pow(&k) / (&k * 2 + 1).powi(2)).summation(&k, &zero, &inf),
    );
    show(
        "Σ (−1)ᵏ/k",
        &(ctx.int(-1).pow(&k) / &k).summation(&k, &one, &inf),
    );
    show("Σ 1/k!", &(1 / &k.factorial()).summation(&k, &zero, &inf));
    show(
        "Σ xᵏ/k!  (power series)",
        &(&x.pow(&k) / &k.factorial()).summation(&k, &zero, &inf),
    );
    show(
        "Σ 1/k            (diverges)",
        &(1 / &k).summation(&k, &one, &inf),
    );

    // `try_summation` is the pipeline-safe variant.
    match k.sin().try_summation(&k, &one, &n) {
        Ok(v) => println!("Σ sin(k) = {v}"),
        Err(e) => println!("Σ_{{k=1}}^{{n}} sin(k)  → Err: {e}"),
    }

    // ── 3. Convergence tests (decisive answers only) ────────────────────
    println!("\n--- Convergence ---");
    let alt_harmonic = ctx.int(-1).pow(&k) / &k;
    println!(
        "1/k²      convergent:            {:?}",
        k.powi(-2).is_convergent(&k)
    );
    println!(
        "1/k       convergent:            {:?}",
        (1 / &k).is_convergent(&k)
    );
    println!(
        "(−1)ᵏ/k   convergent:            {:?}",
        alt_harmonic.is_convergent(&k)
    );
    println!(
        "(−1)ᵏ/k   absolutely convergent: {:?}",
        alt_harmonic.is_absolutely_convergent(&k)
    );
    // Hypergeometric term ratio t(k+1)/t(k) — the input to Gosper's algorithm.
    let term = ctx.int(2).pow(&k) / &k.factorial();
    println!(
        "ratio of 2ᵏ/k!:                  {}",
        term.hypergeometric_ratio(&k).unwrap()
    );

    // ── 4. Products ─────────────────────────────────────────────────────
    println!("\n--- Products ---");
    show("Π_{k=1}^{n} k", &k.product_over(&k, &one, &n));
    show(
        "Π_{k=1}^{n} (1 + 1/k)",
        &(1 + 1 / &k).product_over(&k, &one, &n),
    );
    show(
        "Π_{k=2}^{∞} (1 − 1/k²)",
        &(1 - k.powi(-2)).product_over(&k, &ctx.int(2), &inf),
    );

    // ── 5. Series expansions ────────────────────────────────────────────
    println!("\n--- Series ---");
    show("exp(x) at 0, 6 terms", &x.exp().series(&x, &zero, 6));
    show(
        "√(x²+1) − x as x → ∞",
        &(&(&x.powi(2) + 1).sqrt() - &x).series_at_infinity(&x, 4),
    );
    show(
        "1/(1−x) as x → ∞",
        &(1 / (1 - &x)).series_at_infinity(&x, 4),
    );

    // ── 6. Formal power series ──────────────────────────────────────────
    println!("\n--- Formal power series (lazy, exact) ---");
    let sin_fps = x.sin().fps_maclaurin(&x);
    let cos_fps = x.cos().fps_maclaurin(&x);
    let exp_fps = x.exp().fps_maclaurin(&x);
    let coeffs = |f: &FormalPowerSeries, m: usize| -> String {
        f.coefficients(m)
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!("exp(x):      [{}]", coeffs(&exp_fps, 7));
    println!("  general term a_k = {}", exp_fps.general_term(&k).unwrap());
    println!("sin(x):      [{}]", coeffs(&sin_fps, 8));
    println!("  general term a_k = {}", sin_fps.general_term(&k).unwrap());
    println!("  a_51 = {}", sin_fps.coefficient(51));
    let sin_cos = sin_fps.mul(&cos_fps).unwrap();
    println!(
        "sin·cos:     [{}]  → {}",
        coeffs(&sin_cos, 6),
        sin_cos.truncate(6)
    );
    let sec = cos_fps.inverse().unwrap();
    println!("1/cos(x):    [{}]", coeffs(&sec, 7));
    let asin = sin_fps.reversion().unwrap();
    println!("sin⁻¹ (reversion): [{}]", coeffs(&asin, 8));
    let gauss = exp_fps.compose(&(-&x.powi(2)).fps_maclaurin(&x)).unwrap();
    println!("exp(−x²) (composition): [{}]", coeffs(&gauss, 7));

    // ── 7. Finite differences ───────────────────────────────────────────
    println!("\n--- Finite differences (Fornberg) ---");
    let h = ctx.symbol("h");
    let stencil = [&x - &h, x.clone(), &x + &h];
    let w = symplex::finite_diff::finite_diff_weights(1, &stencil, &x);
    println!(
        "central first-derivative weights on {{x−h, x, x+h}}: [{}]",
        w.iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let d = x.powi(3).differentiate_finite(&x, &stencil, 1).expand();
    println!("central difference of x³:  {d}   (exact 3x² plus the h² truncation term)");
    let d2 = x.powi(4).differentiate_finite(&x, &stencil, 2).expand();
    println!("second difference of x⁴:   {d2}");

    println!("\n✓ Done!");
}
