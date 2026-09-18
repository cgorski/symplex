//! 0.2 numeric back-ends: Rust code generation (`to_rust_fn*`).
//!
//! Structural checks for every new node family plus one end-to-end test that
//! compiles the generated source with `rustc` and compares the results
//! against `Ex::compile`.

use std::io::Write;
use std::process::Command;

use symplex::matrix::{CodegenOptions, MathBackend, Precision};
use symplex::prelude::*;

/// Basic well-formedness: function header present, balanced delimiters.
fn assert_valid_rust(code: &str, fn_name: &str) {
    assert!(
        code.contains(&format!("pub fn {fn_name}(")),
        "missing `pub fn {fn_name}(` in:\n{code}"
    );
    for (open, close) in [('{', '}'), ('(', ')'), ('[', ']')] {
        let o = code.chars().filter(|&c| c == open).count();
        let c = code.chars().filter(|&c| c == close).count();
        assert_eq!(o, c, "unbalanced {open}{close} in:\n{code}");
    }
}

/// Mirror of `symplex-build`'s `strip_cfg_gated_module` heuristic: a line
/// starting with `#[cfg(` containing `feature`, followed by `mod math {`.
fn strip_cfg_gated_module(code: &str) -> String {
    let mut result = String::new();
    let mut lines = code.lines().peekable();
    while let Some(line) = lines.next() {
        if line.starts_with("#[cfg(")
            && line.contains("feature")
            && lines.peek().is_some_and(|n| n.starts_with("mod math {"))
        {
            lines.next();
            let mut depth = 1;
            while depth > 0 {
                match lines.next() {
                    Some(inner) => {
                        for ch in inner.chars() {
                            if ch == '{' {
                                depth += 1;
                            } else if ch == '}' {
                                depth -= 1;
                            }
                        }
                    }
                    None => break,
                }
            }
            if lines.peek().is_some_and(|n| n.trim().is_empty()) {
                lines.next();
            }
            continue;
        }
        result.push_str(line);
        result.push('\n');
    }
    result.trim_start_matches('\n').to_string()
}

// ═══════════════════════════════════════════════════════════════════════════
// Structural tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn special_functions_use_shared_runtime() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let cases: Vec<(&str, Ex, &str)> = vec![
        ("gamma", x.gamma(), "symplex_rt::gamma(x)"),
        ("lgamma", x.log_gamma(), "symplex_rt::lgamma(x)"),
        ("digamma", x.digamma(), "symplex_rt::digamma(x)"),
        ("erf", x.erf(), "symplex_rt::erf(x)"),
        ("erfc", x.erfc(), "symplex_rt::erfc(x)"),
        ("lambertw", x.lambertw(), "symplex_rt::lambert_w0(x)"),
        ("factorial", x.factorial(), "symplex_rt::factorial(x)"),
        ("beta", x.beta(&y), "symplex_rt::beta(x, y)"),
        ("binomial", x.binomial(&y), "symplex_rt::binomial(x, y)"),
        (
            "besselj",
            x.bessel_j(&ctx.int(2)),
            "symplex_rt::bessel_j(2, x)",
        ),
        (
            "bessely",
            x.bessel_y(&ctx.int(0)),
            "symplex_rt::bessel_y(0, x)",
        ),
        (
            "besseli",
            x.bessel_i(&ctx.int(1)),
            "symplex_rt::bessel_i(1, x)",
        ),
        (
            "besselk",
            x.bessel_k(&ctx.int(-3)),
            "symplex_rt::bessel_k(-3, x)",
        ),
        (
            "legendre",
            x.legendre(&ctx.int(4)),
            "symplex_rt::legendre_p(4, x)",
        ),
        (
            "chebyshev_t",
            x.chebyshev_t(&ctx.int(3)),
            "symplex_rt::chebyshev_t(3, x)",
        ),
        (
            "chebyshev_u",
            x.chebyshev_u(&ctx.int(3)),
            "symplex_rt::chebyshev_u(3, x)",
        ),
        (
            "hermite",
            x.hermite(&ctx.int(5)),
            "symplex_rt::hermite_h(5, x)",
        ),
        (
            "laguerre",
            x.laguerre(&ctx.int(2)),
            "symplex_rt::laguerre_l(2, x)",
        ),
        ("fibonacci", x.fibonacci(), "symplex_rt::fibonacci(x)"),
        ("lucas", x.lucas(), "symplex_rt::lucas(x)"),
        ("harmonic", x.harmonic(), "symplex_rt::harmonic(x)"),
        ("factorial2", x.factorial2(), "symplex_rt::factorial2(x)"),
        (
            "rf",
            x.rising_factorial(&y),
            "symplex_rt::rising_factorial(x, y)",
        ),
        (
            "ff",
            x.falling_factorial(&y),
            "symplex_rt::falling_factorial(x, y)",
        ),
    ];
    for (name, expr, expected_call) in cases {
        let code = expr
            .to_rust_fn(name, &["x", "y"])
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_valid_rust(&code, name);
        assert!(
            code.contains(expected_call),
            "{name}: expected `{expected_call}` in:\n{code}"
        );
        assert_eq!(
            code.matches("mod symplex_rt {").count(),
            1,
            "{name}: exactly one runtime module"
        );
        // The module must define the helper it is used for.
        let helper = expected_call
            .trim_start_matches("symplex_rt::")
            .split('(')
            .next()
            .unwrap();
        assert!(
            code.contains(&format!("pub fn {helper}(")),
            "{name}: runtime must define `{helper}`:\n{code}"
        );
        // Module comes before the function.
        assert!(code.find("mod symplex_rt {").unwrap() < code.find("pub fn ").unwrap());
    }
}

#[test]
fn runtime_module_is_minimal_and_transitive() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let code = x.erf().to_rust_fn("f", &["x"]).unwrap();
    assert!(code.contains("fn calerf("), "erf needs its private kernel");
    assert!(
        !code.contains("pub fn gamma("),
        "erf must not pull in gamma"
    );
    assert!(!code.contains("bessel"));
    let code = x.beta(&x.sin()).to_rust_fn("f", &["x"]).unwrap();
    assert!(code.contains("pub fn gamma("), "beta depends on gamma");
    assert!(code.contains("pub fn lgamma("), "beta depends on lgamma");
    assert!(code.contains("fn sin_pi("), "gamma depends on util");
    // No runtime at all for elementary expressions.
    let code = (x.sin() + x.powi(2)).to_rust_fn("f", &["x"]).unwrap();
    assert!(!code.contains("symplex_rt"));
}

#[test]
fn f32_precision_casts_around_runtime_calls() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let opts = CodegenOptions {
        precision: Precision::F32,
        ..Default::default()
    };
    let code = x
        .gamma()
        .to_rust_fn_with_options("g", &["x"], &opts)
        .unwrap();
    assert!(code.contains("pub fn g(x: f32) -> f32"));
    assert!(
        code.contains("symplex_rt::gamma(x as f64) as f32"),
        "f32 call sites must cast:\n{code}"
    );
    // Runtime itself stays f64.
    assert!(code.contains("pub fn gamma(x: f64) -> f64"));
    let code = x
        .bessel_j(&ctx.int(2))
        .to_rust_fn_with_options("j", &["x"], &opts)
        .unwrap();
    assert!(
        code.contains("symplex_rt::bessel_j(2, x as f64) as f32"),
        "integer order must not be cast:\n{code}"
    );
    let code = x
        .beta(&x.sin())
        .to_rust_fn_with_options("b", &["x"], &opts)
        .unwrap();
    assert!(
        code.contains("symplex_rt::beta(x as f64, x.sin() as f64) as f32"),
        "{code}"
    );
}

#[test]
fn math_backends_rewrite_runtime_primitives() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let libm = CodegenOptions {
        math_backend: MathBackend::Libm,
        ..Default::default()
    };
    let code = x
        .gamma()
        .to_rust_fn_with_options("g", &["x"], &libm)
        .unwrap();
    assert!(code.contains("libm::exp(x)"), "libm prims:\n{code}");
    assert!(!code.contains("x.exp()"));

    let cfg = CodegenOptions::no_std();
    let code = (x.gamma() + x.sin())
        .to_rust_fn_with_options("g", &["x"], &cfg)
        .unwrap();
    assert!(code.contains("#[cfg(feature = \"std\")]\nmod math {"));
    assert!(code.contains("math::sin(x)"));
    assert!(code.contains("mod symplex_rt {"));
    // Order: cfg-gated `mod math` blocks, then the runtime, then the fn.
    let i_math = code.find("mod math {").unwrap();
    let i_rt = code.find("mod symplex_rt {").unwrap();
    let i_fn = code.find("pub fn g(").unwrap();
    assert!(i_math < i_rt && i_rt < i_fn, "preamble order:\n{code}");
    // symplex-build's stripping heuristic must remove exactly the math
    // modules and leave the runtime + function intact.
    let stripped = strip_cfg_gated_module(&code);
    assert!(
        !stripped.contains("mod math {"),
        "strip failed:\n{stripped}"
    );
    assert!(stripped.starts_with("#[allow(dead_code, clippy::all)]\nmod symplex_rt {"));
    assert!(stripped.contains("pub fn g("));
    assert_valid_rust(&stripped, "g");
    // Runtime prims are cfg-gated too, so the file works on no_std + libm.
    assert!(code.contains("#[cfg(not(feature = \"std\"))]\n    #[inline(always)]\n    fn p_exp"));
}

#[test]
fn emit_runtime_false_and_runtime_module_api() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let opts = CodegenOptions {
        emit_runtime: false,
        ..Default::default()
    };
    let code = x
        .gamma()
        .to_rust_fn_with_options("g", &["x"], &opts)
        .unwrap();
    assert!(code.contains("symplex_rt::gamma(x)"));
    assert!(
        !code.contains("mod symplex_rt"),
        "no preamble when disabled"
    );
    let rt = opts.runtime_module();
    assert!(rt.starts_with("#[allow(dead_code, clippy::all)]\nmod symplex_rt {"));
    for helper in [
        "gamma",
        "lgamma",
        "digamma",
        "erf",
        "erfc",
        "lambert_w0",
        "beta",
        "binomial",
        "factorial",
        "bessel_j",
        "bessel_y",
        "bessel_i",
        "bessel_k",
        "legendre_p",
        "chebyshev_t",
        "chebyshev_u",
        "hermite_h",
        "laguerre_l",
        "fibonacci",
        "lucas",
        "harmonic",
        "factorial2",
        "rising_factorial",
        "falling_factorial",
    ] {
        assert!(
            rt.contains(&format!("pub fn {helper}(")),
            "full runtime lacks {helper}"
        );
    }
    assert_eq!(rt.matches('{').count(), rt.matches('}').count());
}

#[test]
fn use_mul_add_and_checked_domain_options() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let e = &(&x * &y) + &x.sin();
    let default = e.to_rust_fn("f", &["x", "y"]).unwrap();
    assert!(default.contains("mul_add"), "default fuses:\n{default}");
    let no_fma = CodegenOptions {
        use_mul_add: false,
        ..Default::default()
    };
    let plain = e
        .to_rust_fn_with_options("f", &["x", "y"], &no_fma)
        .unwrap();
    assert!(!plain.contains("mul_add"), "use_mul_add=false:\n{plain}");
    assert!(plain.contains("x * y") || plain.contains("(x * y)"));

    let checked = CodegenOptions {
        checked_domain: true,
        ..Default::default()
    };
    let code = (x.ln() + x.sqrt() + x.gamma() + y.asin() + y.lambertw())
        .to_rust_fn_with_options("f", &["x", "y"], &checked)
        .unwrap();
    assert!(
        code.contains("debug_assert!(x > 0.0_f64"),
        "ln domain:\n{code}"
    );
    assert!(
        code.contains("debug_assert!(x >= 0.0_f64"),
        "sqrt domain:\n{code}"
    );
    assert!(
        code.contains("debug_assert!((y).abs() <= 1.0_f64"),
        "asin domain:\n{code}"
    );
    assert!(
        code.contains("debug_assert!(y >= -0.36787944117144233_f64"),
        "W domain:\n{code}"
    );
    assert!(
        code.contains("fract() == 0.0_f64)"),
        "gamma pole check:\n{code}"
    );
    assert_valid_rust(&code, "f");
    let unchecked = (x.ln() + x.sqrt()).to_rust_fn("f", &["x"]).unwrap();
    assert!(!unchecked.contains("debug_assert"));
}

#[test]
fn piecewise_booleans_and_elementary_nodes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let zero = ctx.zero();
    let pw = Ex::piecewise(&[
        (&x, &x.gt(&y).and(&x.gt(&zero))),
        (&(-&x), &x.le(&y).or(&x.eq_expr(&zero)).not()),
        (&zero, &x.ne_expr(&y)),
    ]);
    let code = pw.to_rust_fn("pw", &["x", "y"]).unwrap();
    assert_valid_rust(&code, "pw");
    assert!(code.contains("if ((x > y) && (x > 0_f64))"), "and:\n{code}");
    assert!(
        code.contains("else if (!((y >= x) || (x == 0_f64)))") || code.contains("else if (!("),
        "not/or:\n{code}"
    );
    assert!(code.contains("else if (x != y)"), "ne:\n{code}");
    assert!(
        code.contains("else { f64::NAN }"),
        "fallthrough → NaN:\n{code}"
    );

    let e = x.min_with(&y)
        + x.max_with(&y)
        + x.sign()
        + x.heaviside()
        + x.floor()
        + y.ceiling()
        + x.abs()
        + y.atan2(&x)
        + x.dirac_delta();
    let code = e.to_rust_fn("elem", &["x", "y"]).unwrap();
    assert_valid_rust(&code, "elem");
    for needle in [
        ".min(",
        ".max(",
        "if x > 0.0_f64 { 1.0_f64 }",
        ".floor()",
        ".ceil()",
        ".abs()",
        ".atan2(",
    ] {
        assert!(code.contains(needle), "missing {needle}:\n{code}");
    }
}

#[test]
fn matrix_codegen_embeds_runtime_once() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![
        vec![x.gamma(), x.erf()],
        vec![x.bessel_j(&ctx.int(1)), x.sin()],
    ])
    .unwrap();
    let code = m.to_rust_fn("m", &["x"]).unwrap();
    assert_valid_rust(&code, "m");
    assert_eq!(code.matches("mod symplex_rt {").count(), 1);
    assert!(code.contains("pub fn gamma("));
    assert!(code.contains("pub fn erf("));
    assert!(code.contains("pub fn bessel_j("));
    assert!(code.contains("[f64; 4]"));
    // 1×1 matrices close their array literal.
    let one = Matrix::new(vec![vec![x.gamma()]]).unwrap();
    let code = one.to_rust_fn("one", &["x"]).unwrap();
    assert!(
        code.contains("[symplex_rt::gamma(x)]"),
        "1x1 literal:\n{code}"
    );
    assert_valid_rust(&code, "one");
}

#[test]
fn constant_folding_of_special_functions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let code = (x.sin() * ctx.int(5).gamma())
        .to_rust_fn("f", &["x"])
        .unwrap();
    assert!(code.contains("24.0_f64"), "Γ(5) folded:\n{code}");
    assert!(
        !code.contains("symplex_rt"),
        "no runtime needed after folding:\n{code}"
    );
}

#[test]
fn codegen_errors_name_the_offender() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let n = ctx.symbol("n");
    match x.bessel_j(&n).to_rust_fn("f", &["x", "n"]) {
        Err(SymplexError::NotImplemented(msg)) => {
            assert!(msg.contains("besselj"), "{msg}")
        }
        other => panic!("expected NotImplemented, got {other:?}"),
    }
    let tree = symplex::tree::ExprTree::Apply {
        name: "myfunc".to_string(),
        args: vec![symplex::tree::ExprTree::Symbol {
            name: "x".to_string(),
        }],
    };
    let user = ctx.from_tree(&tree);
    match user.to_rust_fn("f", &["x"]) {
        Err(SymplexError::NotImplemented(msg)) => {
            assert!(msg.contains("myfunc"), "{msg}")
        }
        other => panic!("expected NotImplemented, got {other:?}"),
    }
    match (x.sin().sin()).exp().integrate(&x).to_rust_fn("f", &["x"]) {
        Err(SymplexError::NotImplemented(msg)) => assert!(msg.contains("Integral")),
        other => panic!("expected NotImplemented, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// End-to-end: compile the generated Rust with rustc and compare to compile()
// ═══════════════════════════════════════════════════════════════════════════

struct Case {
    name: &'static str,
    expr: Ex,
}

fn battery(ctx: &Context) -> Vec<Case> {
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let zero = ctx.zero();
    let pw = Ex::piecewise(&[(&x, &x.gt(&y)), (&y, &x.le(&y))]);
    vec![
        Case {
            name: "poly",
            expr: &x.powi(2) + &(&x * 3) + 1,
        },
        Case {
            name: "trig_cse",
            expr: &x.sin().powi(2) + &(&x.cos().powi(2) * &y) + &(&x.sin() * &x.cos()),
        },
        Case {
            name: "gamma_family",
            expr: &(&x.gamma() + &x.log_gamma()) + &x.digamma(),
        },
        Case {
            name: "erf_family",
            expr: &(&x.erf() * &y.erfc()) + &(&x + 1).lambertw(),
        },
        Case {
            name: "beta_binomial",
            expr: &x.beta(&y) + &(&y.binomial(&ctx.int(2)) / &y.factorial()),
        },
        Case {
            name: "bessel",
            expr: &(&x.bessel_j(&ctx.int(2)) + &x.bessel_y(&ctx.int(1)))
                + &(&x.bessel_i(&ctx.int(0)) * &x.bessel_k(&ctx.int(3))),
        },
        Case {
            name: "orthopoly",
            expr: &(&(&x.legendre(&ctx.int(4)) + &x.chebyshev_t(&ctx.int(3)))
                + &(&x.chebyshev_u(&ctx.int(2)) + &x.hermite(&ctx.int(3))))
                + &x.laguerre(&ctx.int(2)),
        },
        Case {
            name: "sequences",
            expr: &(&(&y.fibonacci() + &y.lucas()) + &y.harmonic()) + &y.factorial2(),
        },
        Case {
            name: "pochhammer",
            expr: &x.rising_factorial(&ctx.int(3)) + &x.falling_factorial(&ctx.int(2)),
        },
        Case {
            name: "piecewise_elem",
            expr: &(&(&pw + &x.min_with(&y)) + &(&x.max_with(&y) + &x.sign()))
                + &(&(&(&x - &y).heaviside() + &x.floor()) + &(&y.ceiling() + &y.atan2(&x))),
        },
        Case {
            name: "numopt",
            expr: &(&x.exp() - 1) + &(&y + 1).ln(),
        },
        Case {
            name: "roots",
            expr: &x.pow(&ctx.rational(1, 3)) + &x.sqrt(),
        },
        Case {
            name: "neg_gamma",
            expr: (-&x).gamma() * (&zero - &y).erf(),
        },
    ]
}

const POINTS: &[(f64, f64)] = &[
    (0.5, 2.0),
    (1.7, 3.0),
    (3.25, 1.0),
    (7.5, 5.0),
    (0.125, 4.0),
];

#[test]
fn generated_rust_compiles_and_matches_compile() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    if Command::new(&rustc).arg("--version").output().is_err() {
        eprintln!("rustc not available; skipping end-to-end codegen test");
        return;
    }
    let ctx = Context::new();
    let cases = battery(&ctx);

    // One shared runtime, per-function preambles disabled.
    let opts = CodegenOptions {
        emit_runtime: false,
        ..Default::default()
    };
    let mut src = String::new();
    src.push_str("#![allow(unused_parens, clippy::all)]\n");
    src.push_str(&opts.runtime_module());
    src.push_str("\n\n");
    for case in &cases {
        let code = case
            .expr
            .to_rust_fn_with_options(case.name, &["x", "y"], &opts)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_valid_rust(&code, case.name);
        src.push_str(&code);
        src.push_str("\n\n");
    }
    src.push_str("fn main() {\n");
    for case in &cases {
        for (px, py) in POINTS {
            src.push_str(&format!(
                "    println!(\"{{:?}}\", {}({px:?}_f64, {py:?}_f64));\n",
                case.name
            ));
        }
    }
    src.push_str("}\n");

    let dir = std::env::temp_dir().join(format!(
        "symplex_codegen_e2e_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let main_rs = dir.join("main.rs");
    std::fs::File::create(&main_rs)
        .unwrap()
        .write_all(src.as_bytes())
        .unwrap();
    let exe = dir.join("main_bin");
    let out = Command::new(&rustc)
        .args(["--edition", "2021", "-O", "-o"])
        .arg(&exe)
        .arg(&main_rs)
        .output()
        .expect("run rustc");
    assert!(
        out.status.success(),
        "generated code failed to compile:\n{}\n--- source ---\n{src}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&exe).output().expect("run generated binary");
    assert!(run.status.success());
    let stdout = String::from_utf8_lossy(&run.stdout);
    let values: Vec<f64> = stdout
        .lines()
        .map(|l| {
            l.trim()
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("bad output line {l:?}"))
        })
        .collect();
    assert_eq!(values.len(), cases.len() * POINTS.len());
    let _ = std::fs::remove_dir_all(&dir);

    // Compare with the VM.
    let mut idx = 0;
    for case in &cases {
        let f = case.expr.compile(&["x", "y"]).unwrap();
        for &(px, py) in POINTS {
            let want = f(&[px, py]);
            let got = values[idx];
            idx += 1;
            if want.is_nan() {
                assert!(got.is_nan(), "{}({px},{py}): rust={got}, vm=NaN", case.name);
                continue;
            }
            let err = (got - want).abs() / want.abs().max(1.0);
            assert!(
                err < 1e-13,
                "{}({px},{py}): rust={got:?}, vm={want:?}, err {err:e}",
                case.name
            );
        }
    }
}

#[test]
fn cfg_gated_output_compiles_with_std_feature() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    if Command::new(&rustc).arg("--version").output().is_err() {
        eprintln!("rustc not available; skipping");
        return;
    }
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let code = (&(&x.gamma() * &x.sin()) + &(&x.exp() - 1))
        .to_rust_fn_with_options("f", &["x"], &CodegenOptions::no_std())
        .unwrap();
    let src = format!(
        "#![allow(dead_code, unused_parens)]\n{code}\nfn main() {{ println!(\"{{:?}}\", f(2.5_f64)); }}\n"
    );
    let dir = std::env::temp_dir().join(format!("symplex_codegen_cfg_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let main_rs = dir.join("main.rs");
    std::fs::write(&main_rs, &src).unwrap();
    let exe = dir.join("cfg_bin");
    let out = Command::new(&rustc)
        .args(["--edition", "2021", "--cfg", "feature=\"std\"", "-o"])
        .arg(&exe)
        .arg(&main_rs)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cfg-gated code failed to compile:\n{}\n--- source ---\n{src}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    let got: f64 = String::from_utf8_lossy(&run.stdout).trim().parse().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let want = 2.5f64.exp() - 1.0 + symplex_gamma_2_5() * 2.5f64.sin();
    assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
}

/// Γ(2.5) = 3√π/4.
fn symplex_gamma_2_5() -> f64 {
    0.75 * std::f64::consts::PI.sqrt()
}
