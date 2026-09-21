//! Definite, improper and numeric integration (symplex 0.2).
//!
//! Demonstrates:
//! - `integrate_definite`: exact definite integrals, including improper
//!   integrals over infinite ranges and integrands with endpoint
//!   singularities;
//! - divergence detection: `try_integrate_definite` returns
//!   `Err(Divergent)` instead of a silently wrong finite number;
//! - the classical improper-integral table (Gaussian, Dirichlet, Fresnel…),
//!   including symbolic parameters under assumptions;
//! - `Abs` / `Heaviside` / `DiracDelta` / `Piecewise` integrands;
//! - adaptive Gauss–Kronrod quadrature with `integrate_numeric` and
//!   `integrate_numeric_with` + `QuadOpts`;
//! - residues at higher-order poles and the residue at infinity.
//!
//! Run with: `cargo run --example definite_integration`

use symplex::prelude::*;

fn main() {
    println!("=== Definite, Improper and Numeric Integration ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let inf = ctx.infinity();
    let neg_inf = ctx.neg_infinity();

    // ── 1. Proper definite integrals (F(b) − F(a) is *not* enough) ──────
    println!("--- Proper definite integrals ---");
    let poly = expr!(ctx, x ^ 3 - 3 * x ^ 2 + 2 * x);
    println!(
        "∫₀¹ ({poly}) dx          = {}",
        poly.integrate_definite(&x, &zero, &one)
    );
    println!(
        "∫₀^π sin(x) dx               = {}",
        x.sin().integrate_definite(&x, &zero, &ctx.pi())
    );
    // 1/x² has a pole at 0 inside [-1, 1]; a naive F(1) − F(−1) = −2 is wrong.
    let two = ctx.int(2);
    println!(
        "∫₁² dx/x²                    = {}",
        x.powi(-2).integrate_definite(&x, &one, &two)
    );

    // ── 2. Divergence is reported, never hidden ─────────────────────────
    println!("\n--- Divergence detection ---");
    let neg_one = ctx.int(-1);
    match x.powi(-2).try_integrate_definite(&x, &neg_one, &one) {
        Err(SymplexError::Divergent { .. }) => {
            println!("∫₋₁¹ dx/x²  → Err(Divergent)   (interior pole; F(1) − F(−1) would give −2)")
        }
        other => println!("unexpected: {other:?}"),
    }
    match (1 / &x).try_integrate_definite(&x, &one, &inf) {
        Err(SymplexError::Divergent { .. }) => println!("∫₁^∞ dx/x   → Err(Divergent)"),
        other => println!("unexpected: {other:?}"),
    }
    // The non-`try_` variant keeps the integral unevaluated instead.
    let kept = x.powi(-2).integrate_definite(&x, &neg_one, &one);
    println!(
        "integrate_definite keeps it symbolic: {kept}  (has_unevaluated = {})",
        kept.has_unevaluated()
    );

    // ── 3. Improper integrals with closed forms ─────────────────────────
    println!("\n--- Improper integrals ---");
    let show = |label: &str, f: &Ex, lo: &Ex, hi: &Ex| {
        println!("{label:<28} = {}", f.integrate_definite(&x, lo, hi));
    };
    show("∫₀^∞ e^(−x) dx", &(-&x).exp(), &zero, &inf);
    show("∫₋∞^∞ e^(−x²) dx", &(-x.powi(2)).exp(), &neg_inf, &inf);
    show("∫₀^∞ sin(x)/x dx", &(&x.sin() / &x), &zero, &inf);
    show("∫₀^∞ dx/(1+x²)", &(1 / (&x.powi(2) + 1)), &zero, &inf);
    show("∫₀¹ ln(x) dx", &x.ln(), &zero, &one);
    show("∫₀¹ dx/√x", &(1 / &x.sqrt()), &zero, &one);
    show(
        "∫₀^∞ x³ e^(−x) dx",
        &(&x.powi(3) * &(-&x).exp()),
        &zero,
        &inf,
    );
    show("∫₀^∞ x/(e^x − 1) dx", &(&x / (&x.exp() - 1)), &zero, &inf);
    show("∫₀^∞ cos(x²) dx", &x.powi(2).cos(), &zero, &inf);
    show(
        "∫₋₁¹ √(1 − x²) dx",
        &(1 - &x.powi(2)).sqrt(),
        &neg_one,
        &one,
    );

    // Symbolic parameters need assumptions: ∫₀^∞ e^(−a x) dx = 1/a needs a > 0.
    let a = ctx.symbol_with("a", &[Assumption::Positive]);
    let n = ctx.symbol_with("n", &[Assumption::Positive]);
    show(
        "∫₀^∞ e^(−a x) dx  (a > 0)",
        &(-(&a * &x)).exp(),
        &zero,
        &inf,
    );
    show(
        "∫₋∞^∞ e^(−a x²) dx (a > 0)",
        &(-(&a * &x.powi(2))).exp(),
        &neg_inf,
        &inf,
    );
    show(
        "∫₀^∞ e^(−a x) sin(x) dx",
        &(&x.sin() * &(-(&a * &x)).exp()),
        &zero,
        &inf,
    );
    show(
        "∫₀^∞ x^(n−1) e^(−x) dx (n > 0)",
        &(&x.pow(&(&n - 1)) * &(-&x).exp()),
        &zero,
        &inf,
    );

    // ── 4. Non-smooth integrands ────────────────────────────────────────
    println!("\n--- Abs / Heaviside / DiracDelta / Piecewise integrands ---");
    let neg_two = ctx.int(-2);
    let three = ctx.int(3);
    println!(
        "∫₋₂³ |x| dx                  = {}",
        x.abs().integrate_definite(&x, &neg_two, &three)
    );
    println!(
        "∫₋₁¹ H(x) dx                 = {}",
        x.heaviside().integrate_definite(&x, &neg_one, &one)
    );
    println!(
        "∫₋₁¹ δ(x) cos(x) dx          = {}",
        (&x.dirac_delta() * &x.cos()).integrate_definite(&x, &neg_one, &one)
    );
    // Piecewise: x² for x < 1, otherwise 2 − x.
    let pw = Ex::piecewise(&[(&x.powi(2), &x.lt(&one)), (&(2 - &x), &x.ge(&one))]);
    println!(
        "∫₀² piecewise dx             = {}",
        pw.integrate_definite(&x, &zero, &two)
    );

    // ── 5. Numeric quadrature (adaptive Gauss–Kronrod G7/K15) ───────────
    println!("\n--- Numeric quadrature ---");
    let v = x.sin().integrate_numeric(&x, &zero, &ctx.pi()).unwrap();
    println!("∫₀^π sin(x) dx               ≈ {v:.15}");
    let gauss = (-x.powi(2))
        .exp()
        .integrate_numeric(&x, &neg_inf, &inf)
        .unwrap();
    println!(
        "∫₋∞^∞ e^(−x²) dx             ≈ {gauss:.15}   (√π = {:.15})",
        std::f64::consts::PI.sqrt()
    );

    // Something with no elementary antiderivative at all: ∫₀¹ e^(x²) dx.
    let no_closed_form = x.powi(2).exp();
    println!(
        "∫₀¹ e^(x²) dx symbolically   = {}",
        no_closed_form.integrate_definite(&x, &zero, &one)
    );
    let opts = QuadOpts {
        rel_tol: 1e-12,
        ..QuadOpts::default()
    };
    let q = no_closed_form
        .integrate_numeric_with(&x, &zero, &one, &opts)
        .unwrap();
    println!(
        "∫₀¹ e^(x²) dx numerically    ≈ {:.15}  (error estimate {:.1e})",
        q.value, q.error
    );

    // A divergent integral is an error here too, not a random large number.
    match x.powi(-2).integrate_numeric(&x, &neg_one, &one) {
        Err(e) => println!("integrate_numeric(1/x², −1, 1) → Err: {e}"),
        Ok(v) => println!("unexpected value {v}"),
    }

    // ── 6. Residues ─────────────────────────────────────────────────────
    println!("\n--- Residues ---");
    let z = ctx.symbol("z");
    let f = &z.exp() / &z.powi(3); // pole of order 3 at 0: Res = 1/2
    println!("Res_{{z=0}} e^z / z³          = {}", f.residue(&z, &zero));
    let g = 1 / (&z.powi(2) + 1); // simple poles at ±i
    let i = ctx.i_unit();
    println!("Res_{{z=i}} 1/(z²+1)          = {}", g.residue(&z, &i));
    // Sum of all residues (finite + infinity) is zero.
    println!(
        "Res_{{z=∞}} 1/(z²+1)          = {}",
        g.residue_at_infinity(&z)
    );
    let h = (&z + 2) / (&z * (&z - 1));
    println!(
        "Res_{{z=∞}} (z+2)/(z(z−1))    = {}   (= −(Res₀ + Res₁) = −(−2 + 3))",
        h.residue_at_infinity(&z)
    );

    println!("\n✓ Done!");
}
