//! Integration coverage probe — tests ~60 integrals across all categories.
//! This is a developer diagnostic tool, NOT a user-facing example.
//! Run with: cargo run --example probe_integration

use symplex::prelude::*;


fn try_integrate(label: &str, expr: &Ex, var: &Ex) -> bool {
    let result = expr.integrate(var);
    let s = format!("{result}");
    let is_unevaluated = s.contains("Integral(");
    let marker = if is_unevaluated { "❌" } else { "✅" };
    println!("  {marker} ∫ {label} dx = {s}");
    !is_unevaluated
}

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, a);
    let mut pass = 0u32;
    let mut fail = 0u32;

    println!("=== Integration Coverage Probe ===\n");

    // ── Basic ──────────────────────────────────────────────────────────
    println!("--- Basic ---");
    let cases_basic: Vec<(&str, Ex)> = vec![
        ("x^2", expr!(ctx, x ^ 2)),
        ("x^5", expr!(ctx, x ^ 5)),
        ("sin(x)", x.sin()),
        ("cos(x)", x.cos()),
        ("exp(x)", x.exp()),
        ("1/x", ctx.int(1) / &x),
        ("tan(x)", x.tan()),
        ("ln(x)", x.ln()),
    ];
    for (label, expr) in &cases_basic {
        if try_integrate(label, expr, &x) { pass += 1; } else { fail += 1; }
    }

    // ── Integration by parts ──────────────────────────────────────────
    println!("\n--- By-parts ---");
    let cases_bp: Vec<(&str, Ex)> = vec![
        ("x*exp(x)", &x * &x.exp()),
        ("x*sin(x)", &x * &x.sin()),
        ("x*cos(x)", &x * &x.cos()),
        ("x^2*exp(x)", &x.powi(2) * &x.exp()),
        ("x*ln(x)", &x * &x.ln()),
        ("ln(x)^2", x.ln().powi(2)),
        ("x^2*sin(x)", &x.powi(2) * &x.sin()),
        ("x^3*exp(x)", &x.powi(3) * &x.exp()),
    ];
    for (label, expr) in &cases_bp {
        if try_integrate(label, expr, &x) { pass += 1; } else { fail += 1; }
    }

    // ── U-substitution ────────────────────────────────────────────────
    println!("\n--- U-substitution ---");
    let cases_usub: Vec<(&str, Ex)> = vec![
        ("2x*exp(x^2)", &(&ctx.int(2) * &x) * &x.powi(2).exp()),
        ("cos(x)*exp(sin(x))", &x.cos() * &x.sin().exp()),
        ("x/(x^2+1)", &x / &(&x.powi(2) + 1)),
    ];
    for (label, expr) in &cases_usub {
        if try_integrate(label, expr, &x) { pass += 1; } else { fail += 1; }
    }

    // ── Trig powers ──────────────────────────────────────────────────
    println!("\n--- Trig powers ---");
    let cases_trig: Vec<(&str, Ex)> = vec![
        ("sin^2(x)", x.sin().powi(2)),
        ("cos^2(x)", x.cos().powi(2)),
        ("sin^3(x)", x.sin().powi(3)),
        ("sin^4(x)", x.sin().powi(4)),
        ("sec^2(x)", expr!(ctx, sec(x) ^ 2)),
        ("tan^2(x)", x.tan().powi(2)),
        ("sin(x)*cos(x)", &x.sin() * &x.cos()),
        ("sin^2(x)*cos^2(x)", &x.sin().powi(2) * &x.cos().powi(2)),
    ];
    for (label, expr) in &cases_trig {
        if try_integrate(label, expr, &x) { pass += 1; } else { fail += 1; }
    }

    // ── Rational functions ───────────────────────────────────────────
    println!("\n--- Rational functions ---");
    let cases_rat: Vec<(&str, Ex)> = vec![
        ("1/(x^2+1)", ctx.int(1) / &(&x.powi(2) + 1)),
        ("1/(x^2-1)", ctx.int(1) / &(&x.powi(2) - 1)),
        ("x/(x^2+1)^2", &x / &(&x.powi(2) + 1).powi(2)),
        ("1/(x^2+a^2)", ctx.int(1) / &(&x.powi(2) + &a.powi(2))),
        ("(2x+3)/(x^2+x+1)", &(&ctx.int(2) * &x + 3) / &(&x.powi(2) + &x + 1)),
        ("1/(1+x^5)", ctx.int(1) / &(&x.powi(5) + 1)),
    ];
    for (label, expr) in &cases_rat {
        if try_integrate(label, expr, &x) { pass += 1; } else { fail += 1; }
    }

    // ── Sqrt forms ───────────────────────────────────────────────────
    println!("\n--- Sqrt forms ---");
    let cases_sqrt: Vec<(&str, Ex)> = vec![
        ("1/sqrt(1-x^2)", ctx.int(1) / &(&ctx.int(1) - &x.powi(2)).sqrt()),
        ("1/sqrt(x^2+1)", ctx.int(1) / &(&x.powi(2) + 1).sqrt()),
        ("1/sqrt(x^2-1)", ctx.int(1) / &(&x.powi(2) - 1).sqrt()),
        ("sqrt(1-x^2)", (&ctx.int(1) - &x.powi(2)).sqrt()),
        ("sqrt(x^2+1)", (&x.powi(2) + 1).sqrt()),
        ("x/sqrt(x^2+1)", &x / &(&x.powi(2) + 1).sqrt()),
    ];
    for (label, expr) in &cases_sqrt {
        if try_integrate(label, expr, &x) { pass += 1; } else { fail += 1; }
    }

    // ── Cyclic IBP (exp*trig) ────────────────────────────────────────
    println!("\n--- Cyclic IBP ---");
    let cases_cyc: Vec<(&str, Ex)> = vec![
        ("exp(x)*sin(x)", &x.exp() * &x.sin()),
        ("exp(x)*cos(x)", &x.exp() * &x.cos()),
    ];
    for (label, expr) in &cases_cyc {
        if try_integrate(label, expr, &x) { pass += 1; } else { fail += 1; }
    }

    // ── Inverse trig ─────────────────────────────────────────────────
    println!("\n--- Inverse trig ---");
    let cases_inv: Vec<(&str, Ex)> = vec![
        ("asin(x)", x.asin()),
        ("acos(x)", x.acos()),
        ("atan(x)", x.atan()),
    ];
    for (label, expr) in &cases_inv {
        if try_integrate(label, expr, &x) { pass += 1; } else { fail += 1; }
    }

    // ── Parametric (variable coefficients) ───────────────────────────
    println!("\n--- Parametric ---");
    let cases_param: Vec<(&str, Ex)> = vec![
        ("a*x^2", &a * &x.powi(2)),
        ("exp(a*x)", (&a * &x).exp()),
        ("sin(a*x)", (&a * &x).sin()),
        ("x*exp(a*x)", &x * &(&a * &x).exp()),
        ("1/(x^2+a^2)", ctx.int(1) / &(&x.powi(2) + &a.powi(2))),
    ];
    for (label, expr) in &cases_param {
        if try_integrate(label, expr, &x) { pass += 1; } else { fail += 1; }
    }

    // ── Hyperbolic ───────────────────────────────────────────────────
    println!("\n--- Hyperbolic ---");
    let cases_hyp: Vec<(&str, Ex)> = vec![
        ("sinh(x)", x.sinh()),
        ("cosh(x)", x.cosh()),
        ("tanh(x)", x.tanh()),
        ("sech^2(x)", expr!(ctx, sech(x) ^ 2)),
        ("sinh^2(x)", x.sinh().powi(2)),
    ];
    for (label, expr) in &cases_hyp {
        if try_integrate(label, expr, &x) { pass += 1; } else { fail += 1; }
    }

    // ── Completing the square ────────────────────────────────────────
    println!("\n--- Completing the square ---");
    let cases_cs: Vec<(&str, Ex)> = vec![
        ("1/(x^2+2x+5)", ctx.int(1) / &(&x.powi(2) + &ctx.int(2) * &x + 5)),
        ("1/sqrt(x^2+2x+5)", ctx.int(1) / &(&x.powi(2) + &ctx.int(2) * &x + 5).sqrt()),
    ];
    for (label, expr) in &cases_cs {
        if try_integrate(label, expr, &x) { pass += 1; } else { fail += 1; }
    }

    // ── Linear substitution ──────────────────────────────────────────
    println!("\n--- Linear substitution ---");
    let cases_lin: Vec<(&str, Ex)> = vec![
        ("(2x+1)^5", (&ctx.int(2) * &x + 1).powi(5)),
        ("1/(3x+2)", ctx.int(1) / &(&ctx.int(3) * &x + 2)),
        ("sqrt(2x+1)", (&ctx.int(2) * &x + 1).sqrt()),
    ];
    for (label, expr) in &cases_lin {
        if try_integrate(label, expr, &x) { pass += 1; } else { fail += 1; }
    }

    // ── Special / Non-elementary ─────────────────────────────────────
    println!("\n--- Special / Non-elementary ---");
    let cases_spec: Vec<(&str, Ex)> = vec![
        ("exp(-x^2)", (-&x.powi(2)).exp()),
        ("sin(x)/x", &x.sin() / &x),
        ("x^x", x.pow(&x)),
        ("1/(1+x^5)", ctx.int(1) / &(&x.powi(5) + 1)),
        ("ln(ln(x))", x.ln().ln()),
    ];
    for (label, expr) in &cases_spec {
        if try_integrate(label, expr, &x) { pass += 1; } else { fail += 1; }
    }

    // ── Summary ──────────────────────────────────────────────────────
    let total = pass + fail;
    let pct = if total > 0 { (pass as f64 / total as f64) * 100.0 } else { 0.0 };
    println!("\n=== Summary: {pass} passed, {fail} failed, {total} total ({pct:.0}%) ===");

    if fail > 0 {
        std::process::exit(1);
    }
}
