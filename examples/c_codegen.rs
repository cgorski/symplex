//! Code generation in symplex 0.2: C99, Rust with an embedded
//! special-function runtime, compiled closures, and shared CSE.
//!
//! Demonstrates:
//! - `to_c_fn` / `to_c_fn_with_options`: self-contained C99 with only the
//!   `static inline symplex_*` helpers the expression needs (Lambert W,
//!   Bessel, …), `fma`, `float` precision, `assert` domain checks, and
//!   piecewise → ternary chains;
//! - `to_rust_fn` for special functions via the embedded `mod symplex_rt`,
//!   plus `CodegenOptions::{use_mul_add, checked_domain, emit_runtime}` and
//!   `runtime_module()` / `c_runtime()` for multi-function files;
//! - `compile` → `Result<CompiledFn>` (`arity`, `try_call`) and
//!   `compile_many` with shared CSE for gradients;
//! - `cse` / `cse_many`.
//!
//! If a C compiler (`cc`) is on the PATH the generated C is compiled and run,
//! and its output is compared against the compiled Rust closure.
//!
//! Run with: `cargo run --example c_codegen`

use std::process::Command;
use symplex::prelude::*;

fn main() {
    println!("=== Code Generation: C99, Rust, compiled closures ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    // ── 1. A special-function expression → C99 ──────────────────────────
    println!("--- to_c_fn: special functions ---");
    let f = &x.gamma() * &(&x.powi(2) + &y).erf() + &x.lambertw() + &x.bessel_j(&ctx.int(1));
    println!("f(x, y) = {f}\n");
    let c_code = f.to_c_fn("f", &["x", "y"]).unwrap();
    // The helper library is long; show the head and the function itself.
    let lines: Vec<&str> = c_code.lines().collect();
    for line in lines.iter().take(4) {
        println!("{line}");
    }
    let helpers = lines
        .iter()
        .filter(|l| l.starts_with("static inline"))
        .count();
    println!(
        "    … {helpers} static inline helper functions (Lambert W, Bessel J series/Miller/Hankel) …"
    );
    for line in lines
        .iter()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        println!("{line}");
    }

    // ── 2. Elementary expression: fma, powers, CSE temporaries ──────────
    println!("\n--- to_c_fn: elementary ---");
    let g = &x.sin().powi(2) + &(&x * 2 + &y).exp() * 3 + &x.powi(3) * &y;
    println!("g(x, y) = {g}");
    println!("{}", g.to_c_fn("g", &["x", "y"]).unwrap());

    println!("--- to_c_fn_with_options: float, static inline, assert domain checks ---");
    let opts = CodegenOptions {
        precision: Precision::F32,
        inline: true,
        checked_domain: true,
        ..Default::default()
    };
    let h = &x.ln() + &x.sqrt();
    println!("{}", h.to_c_fn_with_options("h32", &["x"], &opts).unwrap());

    println!("--- piecewise → ternary chain ---");
    let pw = Ex::piecewise(&[
        (&x.powi(2), &x.lt(&ctx.int(0))),
        (&x.sqrt(), &x.ge(&ctx.int(0))),
    ]);
    println!("{}", pw.to_c_fn("pw", &["x"]).unwrap());

    // ── 3. Rust output ──────────────────────────────────────────────────
    println!("--- to_rust_fn ---");
    println!("{}\n", g.to_rust_fn("g", &["x", "y"]).unwrap());
    let no_fma = CodegenOptions {
        use_mul_add: false,
        ..Default::default()
    };
    println!("use_mul_add = false:");
    println!(
        "{}\n",
        g.to_rust_fn_with_options("g_plain", &["x", "y"], &no_fma)
            .unwrap()
    );
    let checked = CodegenOptions {
        checked_domain: true,
        ..Default::default()
    };
    println!("checked_domain = true:");
    println!(
        "{}\n",
        h.to_rust_fn_with_options("h", &["x"], &checked).unwrap()
    );

    // Special functions embed a `mod symplex_rt` runtime with just the
    // helpers used.  For a file with many functions, emit it once instead.
    let full = f.to_rust_fn("f", &["x", "y"]).unwrap();
    println!(
        "to_rust_fn(f) is {} lines: `mod symplex_rt {{ … }}` + the function.",
        full.lines().count()
    );
    let no_rt = CodegenOptions {
        emit_runtime: false,
        ..Default::default()
    };
    println!("emit_runtime = false gives only the function:");
    println!(
        "{}",
        f.to_rust_fn_with_options("f", &["x", "y"], &no_rt).unwrap()
    );
    println!(
        "…and CodegenOptions::runtime_module() is the complete runtime ({} lines) to paste once;",
        CodegenOptions::default().runtime_module().lines().count()
    );
    println!(
        "CodegenOptions::c_runtime() is the C equivalent ({} lines).",
        CodegenOptions::default().c_runtime().lines().count()
    );
    println!("\nno_std (cfg-gated `mod math`, `#[inline]`):");
    let nostd = h
        .to_rust_fn_with_options("h_nostd", &["x"], &CodegenOptions::no_std())
        .unwrap();
    // Just the function; the two `mod math` variants precede it.
    let tail: Vec<&str> = nostd.lines().rev().take(5).collect();
    for line in tail.into_iter().rev() {
        println!("{line}");
    }

    // ── 4. Compiled closures ────────────────────────────────────────────
    println!("\n--- compile / compile_many ---");
    let cf = f.compile(&["x", "y"]).unwrap();
    println!(
        "compile(f): arity {} ,  f(2, 1) = {:.15}",
        cf.arity(),
        cf(&[2.0, 1.0])
    );
    match cf.try_call(&[1.0]) {
        Err(e) => println!("try_call with 1 argument → Err: {e}"),
        Ok(v) => println!("unexpected {v}"),
    }
    let z = ctx.symbol("z");
    match (&x + &z).compile(&["x"]) {
        Err(SymplexError::FreeSymbol { name }) => {
            println!("free symbol `{name}` → Err(FreeSymbol), not NaN")
        }
        other => println!("unexpected {other:?}"),
    }
    // Gradient with one shared CSE pass.
    let grad = Ex::compile_many(&[&g.diff(&x), &g.diff(&y)], &["x", "y"]).unwrap();
    let gv = grad.call_vec(&[0.5, 0.25]);
    println!(
        "∇g(0.5, 0.25) = ({:.12}, {:.12})   ({} outputs)",
        gv[0],
        gv[1],
        grad.len()
    );
    let (bindings, rewritten) = Ex::cse_many(&[&g.diff(&x), &g.diff(&y)]);
    println!("cse_many shares {} temporaries:", bindings.len());
    for (name, value) in &bindings {
        println!("   {name} = {value}");
    }
    for r in &rewritten {
        println!("   → {r}");
    }

    // ── 5. Compile and run the C, compare with the Rust closure ─────────
    println!("\n--- C vs Rust closure ---");
    let cg = g.compile(&["x", "y"]).unwrap();
    let expected = cg(&[0.5, 0.25]);
    let expected_f = cf(&[2.0, 1.0]);
    println!("Rust closures: g(0.5, 0.25) = {expected:.12},  f(2, 1) = {expected_f:.12}");
    let dir = std::env::temp_dir().join(format!("symplex_c_codegen_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("gen.c");
    let bin = dir.join("gen");
    // Two generated functions in one translation unit: emit the helper
    // library once with `c_runtime()` and the functions with `emit_runtime = false`.
    let shared = CodegenOptions {
        emit_runtime: false,
        ..Default::default()
    };
    let program = format!(
        "{}\n{}\n{}\n#include <stdio.h>\nint main(void) {{ printf(\"%.12f %.12f\\n\", g(0.5, 0.25), f(2.0, 1.0)); return 0; }}\n",
        shared.c_runtime(),
        g.to_c_fn_with_options("g", &["x", "y"], &shared).unwrap(),
        f.to_c_fn_with_options("f", &["x", "y"], &shared).unwrap()
    );
    std::fs::write(&src, program).unwrap();
    let compiled = Command::new("cc")
        .args(["-std=c99", "-O1", "-o"])
        .arg(&bin)
        .arg(&src)
        .arg("-lm")
        .output();
    match compiled {
        Ok(out) if out.status.success() => {
            let run = Command::new(&bin).output().unwrap();
            let text = String::from_utf8_lossy(&run.stdout);
            let vals: Vec<f64> = text
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            println!(
                "C program:     g(0.5, 0.25) = {:.12},  f(2, 1) = {:.12}",
                vals[0], vals[1]
            );
            println!(
                "agreement: |Δg| = {:.1e}, |Δf| = {:.1e}",
                (vals[0] - expected).abs(),
                (vals[1] - expected_f).abs()
            );
        }
        Ok(out) => println!("cc failed:\n{}", String::from_utf8_lossy(&out.stderr)),
        Err(_) => println!("(no C compiler found on PATH — skipping the C run)"),
    }
    let _ = std::fs::remove_dir_all(&dir);

    println!("\n✓ Done!");
}
