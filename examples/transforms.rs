//! Integral transforms and directional limits (symplex 0.2).
//!
//! Demonstrates:
//! - Fourier transforms (`fourier_transform`, `fourier_transform_with` +
//!   `FourierConvention`, `inverse_fourier_transform`) — table + rules,
//!   sign conditions checked through assumptions;
//! - Mellin transforms with their fundamental strip, and the inverse;
//! - Laplace-table extensions, `laplace_initial_value` / `laplace_final_value`;
//! - Fourier series on arbitrary intervals (`fourier_series_on`) with exact
//!   coefficients for `sign`, `|x|` and sawtooth waves;
//! - Z-transforms and their inverses;
//! - one-sided limits: `limit_left` / `limit_right` / `limit_dir` and
//!   the two-sided `limit` returning an unevaluated `Limit` node when the
//!   one-sided limits disagree.
//!
//! Run with: `cargo run --example transforms`

use symplex::prelude::*;

fn main() {
    println!("=== Transforms and Limits ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; t, w, s, x, n, z);
    let a = ctx.symbol_with("a", &[Assumption::Positive]);
    let i = ctx.i_unit();
    let zero = ctx.int(0);

    // ── 1. Fourier transform ────────────────────────────────────────────
    println!("--- Fourier transform  F(ω) = ∫ f(t) e^(−iωt) dt ---");
    let ft = |label: &str, f: &Ex| match f.fourier_transform(&t, &w) {
        Ok(v) => println!("{label:<22} → {v}"),
        Err(e) => println!("{label:<22} → Err: {e}"),
    };
    ft("e^(−a|t|)", &(-&a * t.abs()).exp());
    ft(
        "H(t+1) − H(t−1)",
        &((&t + 1).heaviside() - (&t - 1).heaviside()),
    );
    ft("e^(−2t) H(t)", &((&t * -2).exp() * t.heaviside()));
    ft("e^(−t²)", &(-t.powi(2)).exp());
    ft("t e^(−t) H(t)", &(&t * &(-&t).exp() * &t.heaviside()));
    ft("δ(t)", &t.dirac_delta());
    ft("1", &ctx.int(1));
    ft("cos(3t)", &(&t * 3).cos());
    ft("sinc(t)", &t.sinc());
    // An unknown-sign parameter is an error, not a guess.
    let b = ctx.symbol("b");
    ft("e^(−b|t|), b unknown", &(-&b * t.abs()).exp());

    println!();
    let gauss = (-t.powi(2)).exp();
    println!(
        "unitary angular:  {}",
        gauss
            .fourier_transform_with(&t, &w, FourierConvention::UnitaryAngular)
            .unwrap()
    );
    let nu = ctx.symbol("nu");
    println!(
        "ordinary (ν):     e^(−πt²) → {}   (self-dual)",
        (-(ctx.pi() * t.powi(2)))
            .exp()
            .fourier_transform_with(&t, &nu, FourierConvention::Ordinary)
            .unwrap()
    );
    println!(
        "inverse:          2a/(a²+ω²) → {}",
        (&a * 2 / (&a.powi(2) + &w.powi(2)))
            .inverse_fourier_transform(&w, &t)
            .unwrap()
    );
    println!(
        "inverse:          1/(iω+2)   → {}",
        (1 / (&i * &w + 2))
            .inverse_fourier_transform(&w, &t)
            .unwrap()
    );

    // ── 2. Mellin transform ─────────────────────────────────────────────
    println!("\n--- Mellin transform  M{{f}}(s) = ∫₀^∞ x^(s−1) f(x) dx ---");
    let mt = |label: &str, f: &Ex| match f.mellin_transform(&x, &s) {
        Ok((v, strip)) => println!("{label:<14} → {v:<28} on {strip}"),
        Err(e) => println!("{label:<14} → Err: {e}"),
    };
    mt("e^(−x)", &(-&x).exp());
    mt("e^(−x²)", &(-x.powi(2)).exp());
    mt("1/(1+x)", &(1 / (1 + &x)));
    mt("1/(1+x)³", &(1 / (1 + &x).powi(3)));
    mt("x² e^(−3x)", &(x.powi(2) * (&x * -3).exp()));
    mt("ln(1+x)", &(1 + &x).ln());
    mt("sin(x)", &x.sin());
    println!(
        "inverse: Γ(s) → {},   π/sin(πs) → {}",
        s.gamma().inverse_mellin_transform(&s, &x).unwrap(),
        (ctx.pi() / (ctx.pi() * &s).sin())
            .inverse_mellin_transform(&s, &x)
            .unwrap()
    );

    // ── 3. Laplace transform additions ──────────────────────────────────
    println!("\n--- Laplace ---");
    println!(
        "L{{t² e^(−3t)}}   = {}",
        (&t.powi(2) * &(&t * -3).exp()).laplace(&t, &s)
    );
    println!(
        "L{{sin(2t)/t}}    = {}",
        ((&t * 2).sin() / &t).laplace(&t, &s)
    );
    println!(
        "L{{H(t−2)}}       = {}",
        (&t - 2).heaviside().laplace(&t, &s)
    );
    println!("L{{J₀(t)}}        = {}", t.bessel_j(&zero).laplace(&t, &s));
    println!("L{{sinh(at)}}     = {}", (&a * &t).sinh().laplace(&t, &s));
    println!(
        "L⁻¹{{s/(s²+2s+5)}} = {}",
        (&s / (&s.powi(2) + &s * 2 + 5)).inverse_laplace(&s, &t)
    );
    println!(
        "L⁻¹{{e^(−2s)/s}}  = {}",
        ((&s * -2).exp() / &s).inverse_laplace(&s, &t)
    );
    println!(
        "L⁻¹{{1/√s}}       = {}",
        (1 / &s.sqrt()).inverse_laplace(&s, &t)
    );
    // Anything outside the table is an honest unevaluated node.
    println!("L{{erf(t)}}       = {}", t.erf().laplace(&t, &s));

    let f = (&s + 1) / (&s * (&s.powi(2) + &s * 2 + 5));
    println!("\nF(s) = {f}");
    println!(
        "  f(0⁺) by the initial-value theorem = {}",
        f.laplace_initial_value(&s).unwrap()
    );
    println!(
        "  f(∞)  by the final-value theorem   = {}",
        f.laplace_final_value(&s).unwrap()
    );
    match (1 / (&s * (&s - 1))).laplace_final_value(&s) {
        Err(SymplexError::Divergent { reason, .. }) => {
            println!("  1/(s(s−1)): Err(Divergent) — {reason}")
        }
        other => println!("  unexpected: {other:?}"),
    }

    // ── 4. Fourier series ───────────────────────────────────────────────
    println!("\n--- Fourier series on [−π, π] ---");
    let pi = ctx.pi();
    let square = x.sign().fourier_series_on(&x, &(-&pi), &pi, 5).unwrap();
    println!("sign(x): {}", square.truncate(5));
    println!(
        "  b₁ = {},  b₂ = {},  b₃ = {}",
        square.coefficient_b(1),
        square.coefficient_b(2),
        square.coefficient_b(3)
    );
    let tri = x.abs().fourier_series_on(&x, &(-&pi), &pi, 5).unwrap();
    println!("|x|:     a₀ = {},  {}", tri.a0, tri.truncate(3));
    let saw = x.fourier_series_on(&x, &(-&pi), &pi, 4).unwrap();
    println!("x:       {}", saw.truncate(4));
    println!(
        "  ω₀ = {},  complex coefficient c₁ = {}",
        saw.omega0(),
        saw.coefficient_c(1)
    );
    let parab = x
        .powi(2)
        .fourier_series_on(&x, &ctx.int(-1), &ctx.int(1), 3)
        .unwrap();
    println!("x² on [−1, 1]: {}", parab.truncate(3));

    // ── 5. Z-transform ──────────────────────────────────────────────────
    println!("\n--- Z-transform ---");
    let zt = |label: &str, f: &Ex| match f.z_transform(&n, &z) {
        Ok(v) => println!("Z{{{label}}} = {v}"),
        Err(e) => println!("Z{{{label}}} → Err: {e}"),
    };
    zt("aⁿ", &a.pow(&n));
    zt("n", &n);
    zt("n²", &n.powi(2));
    zt("n aⁿ", &(&n * &a.pow(&n)));
    zt("cos(ωn)", &(&w * &n).cos());
    zt("1/n!", &(1 / &n.factorial()));
    println!(
        "Z⁻¹{{z/(z−1)²}} = {},   Z⁻¹{{1/(z−a)}} = {}",
        (&z / (&z - 1).powi(2)).inverse_z_transform(&z, &n).unwrap(),
        (1 / (&z - &a)).inverse_z_transform(&z, &n).unwrap()
    );

    // ── 6. One-sided limits ─────────────────────────────────────────────
    println!("\n--- One-sided limits ---");
    let lim = |label: &str, f: &Ex, at: &Ex| {
        println!(
            "{label:<12} x→{at}⁻: {:<10} x→{at}⁺: {:<10} two-sided: {}",
            f.limit_left(&x, at).to_string(),
            f.limit_right(&x, at).to_string(),
            f.limit(&x, at)
        );
    };
    lim("1/x", &(1 / &x), &zero);
    lim("|x|/x", &(&x.abs() / &x), &zero);
    lim("e^(−1/x)", &(-1 / &x).exp(), &zero);
    lim("⌊x⌋", &x.floor(), &ctx.int(1));
    lim("sin(x)/x", &(&x.sin() / &x), &zero);
    println!(
        "x·ln(x) as x→0⁺ = {}",
        (&x * &x.ln()).limit_right(&x, &zero)
    );
    println!("xˣ as x→0⁺     = {}", x.pow(&x).limit_right(&x, &zero));
    println!("tan(x) as x→π/2⁻ = {}", x.tan().limit_left(&x, &(&pi / 2)));
    println!(
        "limit_dir(…, Direction::Left) of ln(x) at 1 = {}",
        x.ln().limit_dir(&x, &ctx.int(1), Direction::Left)
    );
    println!(
        "Gruntz two-sided: (1+1/x)^x → {},  x^(1/x) → {},  (1−cos x)/x² → {}",
        (1 + 1 / &x).pow(&x).limit(&x, &ctx.infinity()),
        x.pow(&(1 / &x)).limit(&x, &ctx.infinity()),
        ((1 - &x.cos()) / &x.powi(2)).limit(&x, &zero)
    );

    println!("\n✓ Done!");
}
