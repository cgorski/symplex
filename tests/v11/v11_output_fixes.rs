//! 0.11.1 review fixes: output_fixes — Rust/Python/NumPy/Julia code
//! generation, the Lean proof renderer and Presentation MathML.  Reference
//! values cite SymPy 1.14 / CPython 3.14 (`symplex/.venv/bin/python`).
//!
//! ```text
//! >>> (-8) ** (1/3)                       # Python: complex principal root
//! (1.0000000000000002+1.7320508075688772j)
//! >>> math.copysign(abs(-8) ** (1/3), -8)  # real root, as numpy.cbrt / Rust .cbrt()
//! -2.0
//! >>> pycode(Min())
//! math.inf
//! ```
//!
//! The generated Python is executed with the venv interpreter when it is
//! present, and the generated Rust compiled with `rustc` when available;
//! otherwise those checks are skipped with a note, never failed.

use std::io::Write;
use std::process::Command;

use symplex::lean::{Block, Decl, DeclKind, Proof, Tactic};
use symplex::prelude::*;

const PYTHON: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/.venv/bin/python");

/// Run a Python program with the venv interpreter; `None` (with a note)
/// when the interpreter is absent.
fn run_python(code: &str) -> Option<String> {
    if !std::path::Path::new(PYTHON).is_file() {
        eprintln!("note: {PYTHON} not found; skipping the Python execution check");
        return None;
    }
    let out = Command::new(PYTHON)
        .arg("-c")
        .arg(code)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {PYTHON}: {e}"));
    assert!(
        out.status.success(),
        "python failed on:\n{code}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Compile `src` (a complete Rust program printing one `f64` per line) with
/// `rustc` and return the printed values; `None` when `rustc` is absent.
fn run_rust(src: &str, tag: &str) -> Option<Vec<f64>> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    if Command::new(&rustc).arg("--version").output().is_err() {
        eprintln!("note: rustc not available; skipping the compile check");
        return None;
    }
    let dir = std::env::temp_dir().join(format!(
        "symplex_v11_output_{tag}_{}_{}",
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
        .args(["--edition", "2021", "-o"])
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
    let values = String::from_utf8_lossy(&run.stdout)
        .lines()
        .map(|l| {
            l.trim()
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("{l:?}: {e}"))
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    Some(values)
}

/// The body line(s) of a generated single-expression Rust function.
fn rust_body(e: &Ex, args: &[&str]) -> String {
    let opts = CodegenOptions {
        emit_runtime: false,
        ..Default::default()
    };
    let code = e.to_rust_fn_with_options("f", args, &opts).unwrap();
    code.lines()
        .filter(|l| l.starts_with("    "))
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" ; ")
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Rust printer: a receiver that starts with `-` must be parenthesised
// ═══════════════════════════════════════════════════════════════════════════

/// `-2_f64.powf(x)` parses as `-(2_f64.powf(x))` in Rust; the method call
/// binds tighter than unary minus.  Every method-call emission (`powf`,
/// `powi`, `cbrt`, `exp`, `exp_m1`, `min`, `atan2`, `mul_add`) must wrap a
/// negative literal or a `-(a * b)` product in parentheses.
#[test]
fn rust_negative_receivers_are_parenthesised() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let args = &["x", "y", "z"];
    assert_eq!(rust_body(&ctx.int(-2).pow(&x), args), "(-2_f64).powf(x)");
    assert_eq!(
        rust_body(&ctx.rational(-1, 2).pow(&x), args),
        "(-0.5_f64).powf(x)"
    );
    assert_eq!(
        rust_body(&ctx.int(-3).pow(&(&x + 1)), args),
        "(-3_f64).powf((1_f64 + x))"
    );
    assert_eq!(
        rust_body(&ctx.int(-2).pow(&ctx.rational(1, 3)), args),
        "(-2_f64).cbrt()"
    );
    assert_eq!(rust_body(&(-&x * &y).exp(), args), "(-(x * y)).exp()");
    assert_eq!(rust_body(&(-&x * &y).pow(&z), args), "(-(x * y)).powf(z)");
    assert_eq!(
        rust_body(&((-&x * &y).exp() - 1), args),
        "(-(x * y)).exp_m1()"
    );
    assert_eq!(
        rust_body(&ctx.int(-2).min_with(&x), args),
        "(-2_f64).min(x)"
    );
    assert_eq!(rust_body(&ctx.int(-2).atan2(&x), args), "(-2_f64).atan2(x)");
    // The FMA chain: `-2*x*y + z` fused with a negative first factor.
    assert_eq!(
        rust_body(&(-2 * &x * &y + &z), args),
        "(-2_f64).mul_add((x * y), z)"
    );
    // Positive receivers are untouched.
    assert_eq!(rust_body(&ctx.int(2).pow(&x), args), "x.exp2()");
    assert_eq!(rust_body(&x.pow(&ctx.rational(1, 3)), args), "x.cbrt()");
    assert_eq!(rust_body(&(-&x).exp(), args), "(-x).exp()");
}

/// The generated functions compile and agree with `compile()` at a point
/// where the sign matters.
#[test]
fn rust_negative_receivers_compile_and_match_compile() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let cases: Vec<(&str, Ex)> = vec![
        ("neg_base_powf", ctx.int(-2).pow(&x)),
        ("neg_half_powf", ctx.rational(-1, 2).pow(&x)),
        ("neg_prod_exp", (-&x * &y).exp()),
        ("neg_prod_powf", (-&x * &y).pow(&z)),
        ("neg_prod_expm1", (-&x * &y).exp() - 1),
        ("neg_min", ctx.int(-2).min_with(&x)),
        ("neg_fma", -2 * &x * &y + &z),
        ("neg_root", ctx.int(-2).pow(&ctx.rational(3, 5)) * &x),
    ];
    let opts = CodegenOptions {
        emit_runtime: false,
        ..Default::default()
    };
    let mut src = String::from("#![allow(unused_parens, clippy::all)]\n");
    for (name, e) in &cases {
        src.push_str(
            &e.to_rust_fn_with_options(name, &["x", "y", "z"], &opts)
                .unwrap(),
        );
        src.push_str("\n\n");
    }
    // Two points at which the mis-parsed `-(a.method(…))` differs from the
    // intended `(-a).method(…)`: (-2)^2 = 4 vs -4, min(-2, -3) = -3 vs 3,
    // -2·2·0.5 + 2 = 0 vs -(2 + 2) = -4, …
    let points: &[(f64, f64, f64)] = &[(2.0, 0.5, 2.0), (-3.0, 0.5, 2.0)];
    src.push_str("fn main() {\n");
    for (name, _) in &cases {
        for (px, py, pz) in points {
            src.push_str(&format!(
                "    println!(\"{{:?}}\", {name}({px:?}_f64, {py:?}_f64, {pz:?}_f64));\n"
            ));
        }
    }
    src.push_str("}\n");
    let Some(values) = run_rust(&src, "neg_receivers") else {
        return;
    };
    assert_eq!(values.len(), cases.len() * points.len());
    let mut idx = 0;
    for (name, e) in &cases {
        let f = e.compile(&["x", "y", "z"]).unwrap();
        for &(px, py, pz) in points {
            let want = f(&[px, py, pz]);
            let got = values[idx];
            idx += 1;
            if want.is_nan() {
                assert!(got.is_nan(), "{name}({px},{py},{pz}): rust={got}, vm=NaN");
                continue;
            }
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "{name}({px},{py},{pz}): rust={got:?}, compile()={want:?}\n{src}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Python / NumPy / Julia: real roots for odd denominators
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn python_family_emits_real_roots() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let cbrt = x.pow(&ctx.rational(1, 3));
    let three_fifths = x.pow(&ctx.rational(3, 5));
    let two_fifths = x.pow(&ctx.rational(2, 5));
    // Python (`math`): copysign(|x|^(p/q), x) for odd p, |x|^(p/q) for even p.
    assert_eq!(cbrt.to_python().unwrap(), "math.copysign(abs(x)**(1/3), x)");
    assert_eq!(
        three_fifths.to_python().unwrap(),
        "math.copysign(abs(x)**(3/5), x)"
    );
    assert_eq!(two_fifths.to_python().unwrap(), "abs(x)**(2/5)");
    // … also when the power sits in a denominator.
    assert_eq!(
        (&y / &cbrt).to_python().unwrap(),
        "y/math.copysign(abs(x)**(1/3), x)"
    );
    // NumPy and Julia.
    assert_eq!(cbrt.to_numpy().unwrap(), "numpy.cbrt(x)");
    assert_eq!(
        three_fifths.to_numpy().unwrap(),
        "numpy.copysign(numpy.abs(x)**(3/5), x)"
    );
    assert_eq!(two_fifths.to_numpy().unwrap(), "numpy.abs(x)**(2/5)");
    assert_eq!(cbrt.to_julia().unwrap(), "cbrt(x)");
    assert_eq!(
        three_fifths.to_julia().unwrap(),
        "copysign(abs(x)^(3/5), x)"
    );
    assert_eq!(two_fifths.to_julia().unwrap(), "abs(x)^(2/5)");
    // Even denominators are unchanged (complex for negative bases anyway).
    assert_eq!(x.pow(&ctx.rational(3, 2)).to_python().unwrap(), "x**(3/2)");
    assert_eq!(x.sqrt().to_python().unwrap(), "math.sqrt(x)");
}

/// Executed: `x^(1/3)` at x = −8 is −2.0 (Python's bare `(-8)**(1/3)` is
/// `1+1.73j`), and the odd-denominator idiom agrees with `compile()` (the
/// real-root semantics shared by the Rust and C back ends; `eval_f64` takes
/// the principal complex root and refuses).
#[test]
fn python_real_roots_execute_to_real_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: Vec<Ex> = vec![
        x.pow(&ctx.rational(1, 3)),
        x.pow(&ctx.rational(3, 5)),
        x.pow(&ctx.rational(2, 5)),
        x.pow(&ctx.rational(-1, 3)),
        2 / x.pow(&ctx.rational(1, 3)),
    ];
    let xv = -8.0_f64;
    let mut program = String::from("import math\n");
    program.push_str(&format!("x = {xv:?}\n"));
    let mut expected = Vec::new();
    for e in &cases {
        let code = e.to_python().unwrap();
        program.push_str(&format!("print(repr(float({code})))\n"));
        expected.push(e.compile(&["x"]).unwrap()(&[xv]));
    }
    let def = x
        .pow(&ctx.rational(1, 3))
        .to_python_fn("cbrt", &["x"])
        .unwrap();
    program.push_str(&def);
    program.push_str("print(repr(float(cbrt(x))))\n");
    expected.push(-2.0);
    let Some(out) = run_python(&program) else {
        return;
    };
    let got: Vec<f64> = out
        .lines()
        .map(|l| l.parse::<f64>().unwrap_or_else(|e| panic!("{l}: {e}")))
        .collect();
    assert_eq!(got.len(), expected.len(), "{out}");
    assert_eq!(
        got[0],
        -2.0,
        "cbrt(-8) via {}",
        cases[0].to_python().unwrap()
    );
    for ((g, e), c) in got.iter().zip(&expected).zip(&cases) {
        assert!(
            (g - e).abs() <= 1e-12 * e.abs().max(1.0),
            "python {g} vs symplex {e} for {}",
            c.to_python().unwrap()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Python consistency: factorial on reals, one-factor products
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn python_factorial_is_gamma_on_reals() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // `math.factorial(2.5)` is a TypeError since Python 3.10; Γ(x + 1) is
    // what the Rust/C helpers compute.
    assert_eq!(x.factorial().to_python().unwrap(), "math.gamma(x + 1)");
    assert_eq!(
        (&x - &y).factorial().to_python().unwrap(),
        "math.gamma(x - y + 1)"
    );
    // NumPy has no gamma without SciPy; base Julia none without
    // SpecialFunctions.jl: refused, never a wrong formula.
    assert!(matches!(
        x.factorial().to_numpy(),
        Err(SymplexError::NotImplemented(_))
    ));
    assert!(matches!(
        x.factorial().to_julia(),
        Err(SymplexError::NotImplemented(_))
    ));
    let Some(out) = run_python(&format!(
        "import math\nx = 2.5\nprint(repr({}))",
        x.factorial().to_python().unwrap()
    )) else {
        return;
    };
    let got: f64 = out.parse().unwrap();
    // Γ(3.5) = 3.323350970447843 (mpmath: gamma(3.5)); `compile()` uses the
    // same gamma-based helper as the generated Rust/C.
    let want = x.factorial().compile(&["x"]).unwrap()(&[2.5]);
    assert!((got - 3.323_350_970_447_843).abs() < 1e-12, "{got}");
    assert!((got - want).abs() < 1e-12 * want, "{got} vs {want}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Lean proof renderer
// ═══════════════════════════════════════════════════════════════════════════

/// `Tactic::Apply` lays its atoms out itself and promises never to split
/// inside one; the generic post-wrap must not undo that.
#[test]
fn lean_apply_arguments_are_never_split_by_the_post_wrap() {
    let b = Block::new(vec![Tactic::apply(
        "refine f",
        vec!["(a + b + c + d + e + f)".into(), "?_".into()],
    )]);
    assert_eq!(
        b.render_width("", 20),
        "refine f\n  (a + b + c + d + e + f)\n  ?_\n"
    );
    // Other tactics in the same block are still wrapped.
    let b = Block::new(vec![
        Tactic::apply("refine f", vec!["(a + b + c + d + e + f)".into()]),
        Tactic::raw("linarith only [h0, h1, h2, h3, h4, h5]"),
    ]);
    let text = b.render_width("", 24);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "refine f");
    assert_eq!(lines[1], "  (a + b + c + d + e + f)");
    assert!(lines[2].starts_with("linarith only [h0,"), "{text}");
    assert!(lines[2..].iter().all(|l| l.chars().count() <= 24), "{text}");
    // Inside a bullet the packing is relative to the bullet's column.
    let b = Block::new(vec![Tactic::bullet(Block::new(vec![Tactic::apply(
        "exact g",
        vec!["(x + y + z)".into(), "?_".into()],
    )]))]);
    assert_eq!(
        b.render_width("  ", 22),
        "  · exact g\n      (x + y + z) ?_\n"
    );
}

/// An empty `by` block is a Lean parse error: `skip` keeps it well-formed,
/// as the empty bullet already did.
#[test]
fn lean_empty_by_blocks_render_skip() {
    let b = Block::new(vec![Tactic::have(
        "h",
        Some("T"),
        Proof::by(Block::default()),
    )]);
    assert_eq!(b.render("  "), "  have h : T := by\n    skip\n");
    let b = Block::new(vec![Tactic::bullet(Block::new(vec![
        Tactic::have("h", Some("T"), Proof::by(Block::default())),
        Tactic::raw("exact h"),
    ]))]);
    assert_eq!(
        b.render("  "),
        "  · have h : T := by\n      skip\n    exact h\n"
    );
    let d = Decl::new(DeclKind::Theorem, "t", "True", Block::default());
    assert_eq!(d.render(), "theorem t :\n    True := by\n  skip\n");
    // Non-empty bodies are unchanged.
    let d = Decl::new(
        DeclKind::Example,
        "",
        "True",
        Block::new(vec![Tactic::raw("trivial")]),
    );
    assert_eq!(d.render(), "example :\n    True := by\n  trivial\n");
}

/// Multi-line `Raw` continuation lines sit two columns past the tactic
/// column plus their own leading whitespace (what the docstring now says).
#[test]
fn lean_raw_continuation_lines_keep_their_indentation_plus_two() {
    let b = Block::new(vec![Tactic::raw("refine f a\n  ?_ ?_\nexact h")]);
    assert_eq!(b.render("  "), "  refine f a\n      ?_ ?_\n    exact h\n");
}

/// A doc comment containing `-/` (or `/-`) must not close (or nest) the
/// `/-- … -/` comment.
#[test]
fn lean_doc_comment_delimiters_are_escaped() {
    let d = Decl::new(
        DeclKind::Lemma,
        "l",
        "True",
        Block::new(vec![Tactic::raw("trivial")]),
    )
    .with_doc("Bound from -/ the polytope /- facet");
    let text = d.render();
    assert!(
        text.starts_with("/-- Bound from -\\/ the polytope /\\- facet -/\nlemma l :\n"),
        "{text}"
    );
    // Exactly one comment opener and one closer.
    assert_eq!(text.matches("/--").count(), 1);
    assert_eq!(text.matches("-/").count(), 1);
    assert_eq!(text.matches("/-").count(), 1);
    // Plain docs are untouched.
    let plain = Decl::new(DeclKind::Lemma, "l", "True", Block::default()).with_doc("Plain.");
    assert!(plain.render().starts_with("/-- Plain. -/\n"));
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. MathML: memory, ⅇ / ⅈ, "otherwise"
// ═══════════════════════════════════════════════════════════════════════════

/// 20 000 nested `sin`s render in bounded memory: the cache drops a child's
/// markup once its last reader has consumed it.  (Before the fix this held
/// every prefix of the output at once — 15 GB.)  The layouts that read a
/// grandchild from the cache (`-t` summands, `b^(-n)` factors, `sin(x)^2`)
/// still find it.
#[test]
fn mathml_deep_chain_and_grandchild_reads() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut e = x.clone();
    for _ in 0..20_000 {
        e = e.sin();
    }
    let xml = e.to_mathml().unwrap();
    assert_eq!(xml.matches("<mi>sin</mi>").count(), 20_000);
    assert!(
        xml.ends_with("<mo>)</mo></mrow></mrow></math>"),
        "tail: {}",
        &xml[xml.len() - 40..]
    );

    let y = ctx.symbol("y");
    // Shared sub-expressions (a hash-consed DAG): `x` has several readers.
    let shared = (&x.sin().powi(2) + &x.cos().powi(2)) * &x / &y.powi(2) - &x / 2;
    let xml = shared.to_mathml().unwrap();
    assert!(xml.contains("<msup><mi>sin</mi><mn>2</mn></msup>"), "{xml}");
    assert!(xml.contains("<msup><mi>y</mi><mn>2</mn></msup>"), "{xml}");
    for s in [
        "x*y^(-1)",
        "1 - x/2",
        "-(x+1)^3 - 2*x/y^2",
        "exp(-x^2/2)/sqrt(2*pi)",
        "sin(x)^2 + sin(x)",
        "x^(-2) + x^2 + x",
    ] {
        let xml = ctx.parse(s).unwrap().to_mathml().unwrap();
        assert!(xml.contains("<mi>x</mi>"), "{s}: {xml}");
    }
}

#[test]
fn mathml_e_i_and_otherwise() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // SymPy: <mi>&ExponentialE;</mi>, <mi>&ImaginaryI;</mi> — numeric here.
    assert_eq!(
        (ctx.e() + ctx.i_unit()).to_mathml().unwrap(),
        "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">\
         <mrow><mi>&#x2147;</mi><mo>+</mo><mi>&#x2148;</mi></mrow></math>"
    );
    let pw = Ex::piecewise(&[(&x, &x.gt(&ctx.int(0))), (&(-&x), &ctx.bool_true())]);
    let xml = pw.to_mathml().unwrap();
    assert!(
        xml.contains(
            "<mtd><mrow><mo>-</mo><mi>x</mi></mrow></mtd><mtd><mtext>otherwise</mtext></mtd>"
        ),
        "{xml}"
    );
    assert!(
        xml.contains("<mtext>if&#xA0;</mtext><mrow><mi>x</mi><mo>&gt;</mo><mn>0</mn></mrow>"),
        "{xml}"
    );
    assert!(!xml.contains("<mtext>True</mtext>"), "{xml}");
}
