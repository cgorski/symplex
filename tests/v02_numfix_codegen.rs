//! Regression tests for the cfg-gated `mod math` emitted by
//! `CodegenOptions::no_std()` (`MathBackend::CfgGated`).
//!
//! The `no_std` variant called `libm::abs`, which does not exist (`libm`
//! spells it `fabs`), so any generated function using `abs` failed to
//! compile on embedded targets.  These tests pin the fix and audit that
//! every function the generator can emit exists in *both* variants, with
//! only real `libm` names in the `no_std` one.

use std::collections::BTreeSet;
use std::process::Command;

use symplex::codegen::CodegenOptions;
use symplex::prelude::*;

/// Names exported by the `libm` crate (0.2.x) that the generator may use.
const LIBM_API: &[&str] = &[
    "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "ceil", "cos", "cosh",
    "exp", "exp2", "expm1", "fabs", "floor", "fma", "fmax", "fmin", "log", "log1p", "log2", "pow",
    "sin", "sinh", "sqrt", "tan", "tanh",
];

/// An expression that exercises every `math::` helper the generator emits
/// (each numerical-optimisation pattern is kept in its own sub-term so the
/// canonicaliser cannot merge it away).
fn kitchen_sink(ctx: &Context) -> Ex {
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let t = ctx.symbol("t");
    let unary = x.abs()
        + x.sin()
        + x.cos()
        + x.tan()
        + x.exp()
        + x.ln()
        + x.sqrt()
        + x.cbrt()
        + x.asin()
        + x.acos()
        + x.atan()
        + x.sinh()
        + x.cosh()
        + x.tanh()
        + x.asinh()
        + x.acosh()
        + x.atanh()
        + x.floor()
        + x.ceiling()
        + x.sign();
    let binary = y.atan2(&x) + x.min_with(&y) + x.max_with(&y) + x.pow(&y) + x.powi(7);
    let numopt = (y.exp() - 1) * &y            // expm1 (exp(x) alone is CSE'd)
        + (&x + 1).ln() * &t                   // log1p
        + (&x * &y + &t) * &x                  // fma
        + t.sin() * t.cos() + t.sin() + t.cos() // sin_cos
        + ctx.int(2).pow(&x); // exp2
    unary + binary + numopt
}

const ARGS: [&str; 3] = ["x", "y", "t"];

fn no_std_code(ctx: &Context) -> String {
    kitchen_sink(ctx)
        .to_rust_fn_with_options("f", &ARGS, &CodegenOptions::no_std())
        .expect("no_std codegen")
}

/// Split the emitted text into the two `mod math { … }` bodies (std, no_std).
fn math_modules(code: &str) -> (String, String) {
    let std_start = code
        .find("#[cfg(feature = \"std\")]\nmod math {")
        .expect("std module");
    let nostd_start = code
        .find("#[cfg(not(feature = \"std\"))]\nmod math {")
        .expect("no_std module");
    let std_end = code[std_start..].find("\n}\n").unwrap() + std_start;
    let nostd_end = code[nostd_start..].find("\n}\n").unwrap() + nostd_start;
    (
        code[std_start..std_end].to_string(),
        code[nostd_start..nostd_end].to_string(),
    )
}

fn pub_fns(module: &str) -> BTreeSet<String> {
    module
        .match_indices("pub fn ")
        .map(|(i, _)| {
            let rest = &module[i + 7..];
            rest[..rest.find('(').unwrap()].to_string()
        })
        .collect()
}

fn calls(text: &str, prefix: &str) -> BTreeSet<String> {
    text.match_indices(prefix)
        .map(|(i, _)| {
            let rest = &text[i + prefix.len()..];
            rest[..rest.find('(').unwrap()].to_string()
        })
        .collect()
}

#[test]
fn no_std_math_module_uses_libm_fabs() {
    let ctx = Context::new();
    let code = no_std_code(&ctx);
    let (_, nostd) = math_modules(&code);
    assert!(
        nostd.contains("pub fn abs(x: f64) -> f64 { libm::fabs(x as f64) as f64 }"),
        "no_std abs must forward to libm::fabs:\n{nostd}"
    );
    assert!(!code.contains("libm::abs("), "libm has no `abs`:\n{code}");
    // The body actually calls it.
    assert!(
        code.contains("math::abs("),
        "body should use math::abs:\n{code}"
    );
}

#[test]
fn no_std_math_module_only_uses_real_libm_names() {
    let ctx = Context::new();
    let (_, nostd) = math_modules(&no_std_code(&ctx));
    let used = calls(&nostd, "libm::");
    for name in &used {
        assert!(
            LIBM_API.contains(&name.as_str()),
            "`libm::{name}` does not exist"
        );
    }
    assert!(used.contains("fabs") && used.contains("log"));
}

#[test]
fn both_math_variants_define_every_emitted_helper() {
    let ctx = Context::new();
    let code = no_std_code(&ctx);
    let (std_mod, nostd_mod) = math_modules(&code);
    let std_fns = pub_fns(&std_mod);
    let nostd_fns = pub_fns(&nostd_mod);
    assert_eq!(
        std_fns, nostd_fns,
        "std and no_std `mod math` must export the same API"
    );

    // Every `math::name(` in the function body must be defined, and the
    // kitchen sink must actually reach each helper (so this audit cannot
    // silently pass on an expression that emits nothing).
    let body = code.split("mod math {").last().unwrap();
    let body = &body[body.find("\n}\n").unwrap()..];
    let used = calls(body, "math::");
    let expected: BTreeSet<String> = [
        "abs", "sin", "cos", "tan", "exp", "ln", "sqrt", "cbrt", "asin", "acos", "atan", "sinh",
        "cosh", "tanh", "asinh", "acosh", "atanh", "floor", "ceil", "atan2", "min", "max", "powf",
        "powi", "expm1", "log1p", "exp2", "fma", "sin_cos",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    for name in &expected {
        assert!(
            used.contains(name),
            "kitchen sink should emit math::{name}; body:\n{body}"
        );
    }
    for name in &used {
        assert!(
            std_fns.contains(name),
            "math::{name} used but not defined in mod math"
        );
    }
    // `log2` has no stable trigger through the public API but is part of
    // the helper set the generator may emit.
    assert!(std_fns.contains("log2") && nostd_fns.contains("log2"));
}

/// Compile the emitted file with `rustc` (skipped when unavailable).
///
/// * `std` variant: `--cfg feature="std"`.
/// * `no_std` variant: linked against a stub `libm` crate exposing exactly
///   the real `libm` API (`LIBM_API`), so a misspelled name fails to link
///   just as it would against the real crate.
fn compile_variant(code: &str, std: bool, tag: &str) {
    let Ok(out) = Command::new("rustc").arg("--version").output() else {
        eprintln!("rustc not available; skipping compile check");
        return;
    };
    if !out.status.success() {
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "symplex_numfix_codegen_{tag}_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();

    let mut cmd = Command::new("rustc");
    cmd.args(["--crate-type", "lib", "--edition", "2024", "-o"])
        .arg(dir.join("gen.rlib"));
    if std {
        cmd.args(["--cfg", "feature=\"std\""]);
    } else {
        let mut stub = String::from("#![no_std]\n");
        for f in LIBM_API {
            let arity = match *f {
                "atan2" | "pow" | "fmin" | "fmax" => 2,
                "fma" => 3,
                _ => 1,
            };
            let params: Vec<String> = (0..arity).map(|i| format!("_a{i}: f64")).collect();
            stub.push_str(&format!(
                "pub fn {f}({}) -> f64 {{ 0.0 }}\n",
                params.join(", ")
            ));
        }
        let stub_src = dir.join("libm.rs");
        std::fs::write(&stub_src, stub).unwrap();
        let rlib = dir.join("liblibm.rlib");
        let out = Command::new("rustc")
            .args([
                "--crate-type",
                "lib",
                "--crate-name",
                "libm",
                "--edition",
                "2024",
                "-o",
            ])
            .arg(&rlib)
            .arg(&stub_src)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "libm stub failed to compile: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        cmd.arg("--extern").arg(format!("libm={}", rlib.display()));
    }
    let src = dir.join("gen.rs");
    std::fs::write(
        &src,
        format!("#![allow(dead_code, unused_parens, clippy::all)]\n{code}"),
    )
    .unwrap();
    let out = cmd.arg(&src).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        out.status.success(),
        "generated {tag} code failed to compile:\n{stderr}\n--- source ---\n{code}"
    );
}

#[test]
fn no_std_output_compiles_std_variant() {
    let ctx = Context::new();
    compile_variant(&no_std_code(&ctx), true, "std");
}

#[test]
fn no_std_output_compiles_libm_variant() {
    let ctx = Context::new();
    compile_variant(&no_std_code(&ctx), false, "nostd");
}

#[test]
fn no_std_output_compiles_f32_variant() {
    let ctx = Context::new();
    let code = kitchen_sink(&ctx)
        .to_rust_fn_with_options("f", &ARGS, &CodegenOptions::embedded_f32())
        .expect("f32 codegen");
    compile_variant(&code, false, "f32");
}
