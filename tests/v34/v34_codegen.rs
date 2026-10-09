//! The code emitters (`compile`, `to_rust_fn`, `to_c_fn`, `to_python_fn`,
//! `to_numpy_fn`) and the shared `f64` special-function runtime, after a
//! differential hunt against `eval_f64` and mpmath.  Each test says what
//! was wrong before and cites its oracle (an `eval_f64` call, or the
//! mpmath 1.3 call at `mp.dps = 30` and `60`, same digits at both).
//!
//! Tests that compile or run generated code skip (with a note on stderr)
//! when `cc`, `rustc` or `python3` (with `numpy` where needed) is absent.

use std::process::Command;
use symplex::matrix::{CodegenOptions, MathBackend};
use symplex::prelude::*;

fn rel(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        a.abs()
    } else {
        ((a - b) / b).abs()
    }
}

/// `f(x)` through `compile()`.
fn vm(src: &str, x: f64) -> f64 {
    let ctx = Context::new();
    ctx.parse(src).unwrap().compile(&["x"]).unwrap().call(&[x])
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("v34_codegen_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn tool(cmd: &str, arg: &str) -> bool {
    let ok = Command::new(cmd)
        .arg(arg)
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        eprintln!("{cmd} not available; skipping");
    }
    ok
}

/// Compile the C function `f(double x)` in `code` and print `f` at `xs`.
fn run_c(code: &str, xs: &[f64], tag: &str) -> Option<Vec<f64>> {
    if !tool("cc", "--version") {
        return None;
    }
    let dir = scratch(tag);
    let mut src = format!("{code}\n#include <stdio.h>\nint main(void) {{\n");
    for x in xs {
        src.push_str(&format!("    printf(\"%.17g\\n\", f({x:e}));\n"));
    }
    src.push_str("    return 0;\n}\n");
    std::fs::write(dir.join("f.c"), src).unwrap();
    let out = Command::new("cc")
        .current_dir(&dir)
        .args(["-O1", "f.c", "-lm", "-o", "f"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}\n{code}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(dir.join("f")).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(
        String::from_utf8_lossy(&run.stdout)
            .lines()
            .map(|l| {
                l.trim()
                    .replace("nan", "NaN")
                    .parse::<f64>()
                    .unwrap_or(f64::NAN)
            })
            .collect(),
    )
}

/// Run Python `code` defining `f(x, y)` and print `repr(f(x, y))` (or the
/// exception's name) for each point; `numpy`: arguments as `numpy.float64`.
fn run_py(code: &str, pts: &[(f64, f64)], numpy: bool, tag: &str) -> Option<Vec<String>> {
    let probe = if numpy { "import numpy" } else { "import math" };
    let imports = if numpy {
        "import math, numpy\nnumpy.seterr(all='ignore')"
    } else {
        "import math"
    };
    let ok = Command::new("python3")
        .args(["-c", probe])
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        eprintln!(
            "python3{} not available; skipping",
            if numpy { " with numpy" } else { "" }
        );
        return None;
    }
    let dir = scratch(tag);
    let wrap = if numpy { "numpy.float64" } else { "float" };
    let mut src = format!("{imports}\n{code}\n");
    for (x, y) in pts {
        src.push_str(&format!(
            "try:\n    print(repr(f({wrap}({x:e}), {wrap}({y:e}))))\nexcept Exception as ex:\n    print(type(ex).__name__)\n"
        ));
    }
    std::fs::write(dir.join("f.py"), src).unwrap();
    let out = Command::new("python3")
        .arg(dir.join("f.py"))
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::to_string)
            .collect(),
    )
}

/// Compile generated Rust as a library (`no_std`: under `#![no_std]`
/// against a stub `libm`); `false` when rustc is absent.
fn rustc_lib(code: &str, no_std: bool, tag: &str) -> bool {
    if !tool("rustc", "--version") {
        return false;
    }
    let dir = scratch(tag);
    let mut cmd = Command::new("rustc");
    cmd.current_dir(&dir)
        .args(["--crate-type", "lib", "--edition", "2021", "-o", "gen.rlib"]);
    let header = if no_std {
        let fns = [
            "sin", "cos", "tan", "exp", "log", "fabs", "sqrt", "cbrt", "asin", "acos", "atan",
            "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "floor", "ceil", "expm1", "log1p",
            "log2", "exp2",
        ];
        let mut stub = String::from("#![no_std]\n");
        for f in fns {
            stub.push_str(&format!("pub fn {f}(_a: f64) -> f64 {{ 0.0 }}\n"));
        }
        for f in ["atan2", "pow", "fmin", "fmax"] {
            stub.push_str(&format!("pub fn {f}(_a: f64, _b: f64) -> f64 {{ 0.0 }}\n"));
        }
        stub.push_str("pub fn fma(_a: f64, _b: f64, _c: f64) -> f64 { 0.0 }\n");
        std::fs::write(dir.join("libm.rs"), stub).unwrap();
        let out = Command::new("rustc")
            .current_dir(&dir)
            .args([
                "--crate-type",
                "lib",
                "--crate-name",
                "libm",
                "--edition",
                "2021",
            ])
            .args(["-o", "liblibm.rlib", "libm.rs"])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        cmd.args(["--extern", "libm=liblibm.rlib"]);
        "#![no_std]\n"
    } else {
        ""
    };
    std::fs::write(
        dir.join("gen.rs"),
        format!("{header}#![allow(dead_code, unused_parens, clippy::all)]\n{code}"),
    )
    .unwrap();
    let out = cmd.arg("gen.rs").output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        out.status.success(),
        "generated code failed to compile:\n{}\n--- source ---\n{code}",
        String::from_utf8_lossy(&out.stderr)
    );
    true
}

// ═══════════════════════════════════════════════════════════════════════════
// The shared runtime (`compile()`, `to_rust_fn`) and its C mirror
// ═══════════════════════════════════════════════════════════════════════════

/// `besselk(n, x)` for tiny x: the trapezoidal rule ran out of steps before
/// the integrand decayed (it stays near its peak up to t ≈ ln(2n/x)) and
/// returned the truncated sum: K₀(1.7·10⁻²¹⁰) was 85.67, K₁(1.7·10⁻²⁹⁴)
/// 1.25·10³⁶, K₅(1.7·10⁻⁵⁶) 2.7·10¹⁶⁶; now the power series of K₀, K₁ and
/// the upward recurrence.  mpmath: `besselk(0, mpf(1.7e-210))` =
/// 483.12817279334583577, `besselk(1, mpf(1.7e-294))` =
/// 5.8823529411764709119e+293, `besselk(5, mpf(1.7e-56))` =
/// 2.7044977064591723143e+281; on the series branch `besselk(2,
/// mpf(0.5))` = 7.5501835512408694366, `besselk(3, 1)` = 7.101262824737944506,
/// `besselk(0, 1)` = 0.42102443824070833334.
#[test]
fn besselk_of_small_arguments() {
    let cases = [
        ("besselk(0, x)", 1.7e-210, 483.128_172_793_345_8),
        ("besselk(1, x)", 1.7e-294, 5.882_352_941_176_471e293),
        ("besselk(5, x)", 1.7e-56, 2.704_497_706_459_172_3e281),
        ("besselk(2, x)", 0.5, 7.550_183_551_240_869),
        ("besselk(3, x)", 1.0, 7.101_262_824_737_944),
        ("besselk(0, x)", 1.0, 0.421_024_438_240_708_3),
    ];
    for (src, x, want) in cases {
        assert!(
            rel(vm(src, x), want) < 1e-14,
            "{src} at {x}: {}",
            vm(src, x)
        );
        let ctx = Context::new();
        let code = ctx.parse(src).unwrap().to_c_fn("f", &["x"]).unwrap();
        if let Some(v) = run_c(&code, &[x], "besselk") {
            assert!(rel(v[0], want) < 1e-14, "C {src} at {x}: {}", v[0]);
        }
    }
}

/// `binomial(x, k)` and `falling_factorial(x, k)` for an integer k and x
/// near a small integer: each factor was formed as `x − k + i`, which lost
/// x next to 0 — `binomial(1.7e-16, 3)` and `falling_factorial(1.7e-16, 3)`
/// were 0, `binomial(1e-10, 3)` had a relative error of 2·10⁻⁶.  mpmath:
/// `binomial(mpf(1.7e-16), 3)` = 5.6666666666666652676e-17,
/// `ff(mpf(1.7e-16), 3)` = 3.3999999999999991605e-16,
/// `binomial(mpf(1e-10), 3)` = 3.3333333328333334548e-11.
/// `falling_factorial(3, 6.1e16)` was NaN (`x − n + 1` rounded onto the
/// pole of the denominator); it is 0 (`eval_f64` of
/// `falling_factorial(3, 61000000000000000)`).
#[test]
fn binomial_and_falling_factorial_near_integers() {
    let cases = [
        ("binomial(x, 3)", 1.7e-16, 5.666_666_666_666_665e-17),
        (
            "falling_factorial(x, 3)",
            1.7e-16,
            3.399_999_999_999_999e-16,
        ),
        ("binomial(x, 3)", 1e-10, 3.333_333_332_833_334e-11),
    ];
    for (src, x, want) in cases {
        assert!(
            rel(vm(src, x), want) < 4e-16,
            "{src} at {x}: {}",
            vm(src, x)
        );
        let ctx = Context::new();
        let code = ctx.parse(src).unwrap().to_c_fn("f", &["x"]).unwrap();
        if let Some(v) = run_c(&code, &[x], "binom") {
            assert!(rel(v[0], want) < 4e-16, "C {src} at {x}: {}", v[0]);
        }
    }
    let ctx = Context::new();
    let want = ctx
        .parse("falling_factorial(3, 61000000000000000)")
        .unwrap()
        .eval_f64()
        .unwrap();
    assert_eq!(want, 0.0);
    assert_eq!(vm("falling_factorial(3, x)", 6.1e16), 0.0);
}

/// `binomial(1/2, k)` for a large k went through ln Γ differences of size
/// k·ln k: a relative error of 5·10⁻⁸ at k = 3.3·10⁷.  Now Γ(c) is
/// reflected and Γ(k − n)/Γ(k + 1) taken by Stirling's difference.
/// mpmath: `binomial(mpf(1)/2, 33000000)` = -1.4880727474949897851e-12.
#[test]
fn binomial_of_a_half_and_a_large_k() {
    let want = -1.488_072_747_494_989_8e-12;
    assert!(rel(vm("binomial(1/2, x)", 33_000_000.0), want) < 1e-14);
    let ctx = Context::new();
    let code = ctx
        .parse("binomial(1/2, x)")
        .unwrap()
        .to_c_fn("f", &["x"])
        .unwrap();
    if let Some(v) = run_c(&code, &[33_000_000.0], "binom_half") {
        assert!(rel(v[0], want) < 1e-14, "C: {}", v[0]);
    }
}

/// `rising_factorial(x, 1/2)` for tiny x lost |ln Γ(x)|·ε (450 ulps at
/// 6.1·10⁻²⁸⁰); the Γ quotient keeps it.  mpmath:
/// `rf(mpf(6.1e-280), mpf(1)/2)` = 1.0811968490523647913e-279.
#[test]
fn rising_factorial_of_a_tiny_argument() {
    let want = 1.081_196_849_052_364_8e-279;
    assert!(rel(vm("rising_factorial(x, 1/2)", 6.1e-280), want) < 4e-16);
}

/// The C helper library mirrored an older runtime: `beta` had no Stirling
/// form, so B(π/x, x + y) at x = −3772/128, y = 386·512 had a relative
/// error of 5·10⁻¹⁰ (`compile()` was right).  mpmath:
/// `beta(pi/x, x + y)` with `x = mpf(-3772)/128`, `y = mpf(386)*512` =
/// -36.965359176771850162.
#[test]
fn c_beta_has_the_stirling_form() {
    let ctx = Context::new();
    let e = ctx.parse("beta(pi/x, x + 197632)").unwrap();
    let want = -36.965_359_176_771_85;
    assert!(rel(e.compile(&["x"]).unwrap().call(&[-29.46875]), want) < 1e-14);
    let code = e.to_c_fn("f", &["x"]).unwrap();
    if let Some(v) = run_c(&code, &[-29.46875], "beta") {
        assert!(rel(v[0], want) < 1e-14, "C: {}", v[0]);
    }
}

/// `erfc(x)` for 26.55 < x < 27.2 is subnormal; Cody's cut-off returned
/// 0.  mpmath: `erfc(mpf(26.75))` = 3.6220391064788048176e-313,
/// `erfc(27)` = 5.237048923789255685e-319.
#[test]
fn erfc_in_the_subnormal_range() {
    // Subnormal references, read from mpmath's digits.
    let want = |s: &str| s.parse::<f64>().unwrap();
    let a = vm("erfc(x)", 26.75);
    assert!(rel(a, want("3.6220391064788048176e-313")) < 1e-10, "{a:e}");
    let b = vm("erfc(x)", 27.0);
    assert!(rel(b, want("5.237048923789255685e-319")) < 1e-4, "{b:e}");
    assert_eq!(vm("erfc(x)", 27.5), 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// A symbol-free subexpression whose naive `f64` formula is finite was
/// printed as that formula even when it cancels: `x·(√(10²⁰ + 1) − 10¹⁰)`
/// was `x * (-10000000000 + 100000000000000000001.sqrt())` = 0 in Rust, C,
/// Python and NumPy (`compile()` folded it).  Oracle: `eval_f64` of
/// `sqrt(10^20+1) - 10^10` = 5e-11 (mpmath at dps 60: 5.0e-11).
#[test]
fn cancelling_constants_are_folded_by_every_emitter() {
    let ctx = Context::new();
    let e = ctx.parse("x*(sqrt(10^20+1) - 10^10)").unwrap();
    let c = ctx
        .parse("sqrt(10^20+1) - 10^10")
        .unwrap()
        .eval_f64()
        .unwrap();
    assert!(rel(c, 5e-11) < 1e-15);
    let rust = e.to_rust_fn("f", &["x", "y"]).unwrap();
    assert!(!rust.contains("sqrt"), "{rust}");
    let code = e.to_c_fn("f", &["x"]).unwrap();
    assert!(!code.contains("sqrt"), "{code}");
    if let Some(v) = run_c(&code, &[2.0], "cancel") {
        assert!(rel(v[0], 2.0 * c) < 1e-15, "C: {}", v[0]);
    }
    let py = e.to_python_fn("f", &["x", "y"]).unwrap();
    assert!(!py.contains("sqrt"), "{py}");
    // An accurate formula is kept.
    let keep = ctx.parse("x*sqrt(2)").unwrap().to_python().unwrap();
    assert_eq!(keep, "x*math.sqrt(2)");
}

/// The canonical form flattens constant factors into the product with the
/// variables, where each overflows or underflows alone:
/// `x·2⁻²⁰⁰⁰·C(2000, 1000)` was 0·∞ = NaN in every back end, `compile()`
/// included, and `x·e⁻⁸⁰⁰·e⁸⁰⁰` NaN (Python: `OverflowError`).  Oracles:
/// `eval_f64` of `binomial(2000,1000)/2^2000` = 0.01783901114585432 (mpmath
/// `binomial(2000, 1000)/mpf(2)**2000` = 0.01783901114585432073), of
/// `exp(-800)*exp(800)` = 1.
#[test]
fn constant_parts_of_products_are_folded() {
    let ctx = Context::new();
    let want = 0.017_839_011_145_854_32;
    let e = ctx.parse("x*binomial(2000,1000)/2^2000").unwrap();
    assert!(rel(e.compile(&["x"]).unwrap().call(&[3.0]), 3.0 * want) < 1e-15);
    let code = e.to_c_fn("f", &["x"]).unwrap();
    if let Some(v) = run_c(&code, &[3.0], "group") {
        assert!(rel(v[0], 3.0 * want) < 1e-15, "C: {}\n{code}", v[0]);
    }
    let e = ctx.parse("x*exp(-800)*exp(800)").unwrap();
    assert_eq!(e.compile(&["x"]).unwrap().call(&[3.0]), 3.0);
    let rust = e.to_rust_fn("f", &["x"]).unwrap();
    assert!(!rust.contains("exp"), "{rust}");
    // Terms of a sum likewise: x + (√(10²⁰ + 1) − 10¹⁰) at x = 2⁻⁴⁰.
    let e = ctx.parse("x + sqrt(10^20+1) - 10^10").unwrap();
    let x = 2f64.powi(-40);
    assert!(rel(e.compile(&["x"]).unwrap().call(&[x]), x + 5e-11) < 1e-15);
}

/// A rational or integer past the `f64` range: Rust printed the integer
/// digits with `_f64` (`overflowing_literals`, a compile error) and
/// `NaN.0_f64` for a rational whose parts overflow (`n as f64 / d as f64`
/// = ∞/∞); C printed `NAN`.  Now the correctly rounded value, ±∞ beyond the
/// range.  Oracle: `eval_f64` of `(1/8)^3/4 + 1/3` nested ten times (an
/// exact rational of 30,000 digits) = 0.33352713968829834.
#[test]
fn literals_past_the_range_compile() {
    let ctx = Context::new();
    let mut s = "1/8".to_string();
    for _ in 0..10 {
        s = format!("(({s})^3/4 + 1/3)");
    }
    let r = ctx.parse(&s).unwrap();
    let want = r.eval_f64().unwrap();
    let e = &ctx.parse("x").unwrap() * &r;
    let rust = e.to_rust_fn("f", &["x"]).unwrap();
    assert!(!rust.contains("NaN.0") && !rust.contains("inf"), "{rust}");
    let c = e.to_c_fn("f", &["x"]).unwrap();
    assert!(!c.contains("NAN"), "{c}");
    if let Some(v) = run_c(&c, &[1.0], "bigrat") {
        assert!(rel(v[0], want) < 1e-15, "C: {}", v[0]);
    }
    let big = ctx.parse("x + 10^400*y").unwrap();
    let rust = big.to_rust_fn("f", &["x", "y"]).unwrap();
    assert!(rust.contains("f64::INFINITY"), "{rust}");
    rustc_lib(&rust, false, "bigint");
}

// ═══════════════════════════════════════════════════════════════════════════
// Powers
// ═══════════════════════════════════════════════════════════════════════════

/// An odd integer exponent beyond 2⁵³ rounds to an even `f64`, so
/// `x^(2⁵³ + 1)` at −1 was +1 in every back end.  Oracle: `eval_f64` of
/// `(-1)^9007199254740993` = −1; of `(-(1 + 2^-52))^9007199254740993` =
/// −e²·(1 + 2⁻⁵²)… ≈ −7.389.
#[test]
fn odd_powers_beyond_2_pow_53_keep_their_sign() {
    let ctx = Context::new();
    let e = ctx.parse("x^9007199254740993").unwrap();
    let f = e.compile(&["x"]).unwrap();
    assert_eq!(f.call(&[-1.0]), -1.0);
    let b = -(1.0 + f64::EPSILON);
    let want = ctx
        .parse("(-(1 + 2^-52))^9007199254740993")
        .unwrap()
        .eval_f64()
        .unwrap();
    assert!(want < -7.0 && rel(f.call(&[b]), want) < 1e-14, "{want}");
    let code = e.to_c_fn("f", &["x"]).unwrap();
    if let Some(v) = run_c(&code, &[-1.0, 2.0f64.powi(-60)], "oddpow") {
        assert_eq!(v[0], -1.0);
        assert_eq!(v[1], 0.0);
    }
    let rust = e.to_rust_fn("f", &["x"]).unwrap();
    rustc_lib(&rust, false, "oddpow");
    let py = e.to_python_fn("f", &["x", "y"]).unwrap();
    if let Some(out) = run_py(&py, &[(-1.0, 0.0)], false, "oddpow") {
        assert_eq!(out[0], "-1.0", "{py}");
    }
}

/// Python's `**` of a negative base and a non-integer exponent is a
/// complex number: `x**y` returned `(8.7e-17+1.41j)` at (−2, ½) where the
/// value is not real (`eval_f64` of `(-2)^(1/2)` is an error, its
/// `eval_complex64` 1.414…i).  Now NaN, as `compile()` and `pow`.
#[test]
fn python_powers_of_negative_bases_are_nan() {
    let ctx = Context::new();
    let e = ctx.parse("x^y").unwrap();
    let py = e.to_python_fn("f", &["x", "y"]).unwrap();
    assert!(ctx.parse("(-2)^(1/2)").unwrap().eval_f64().is_err());
    if let Some(out) = run_py(
        &py,
        &[(-2.0, 0.5), (-2.0, 3.0), (2.0, 0.5)],
        false,
        "powneg",
    ) {
        assert_eq!(out, ["nan", "-8.0", "1.4142135623730951"], "{py}");
    }
    // A base of known sign keeps the plain operator.
    assert_eq!(ctx.parse("2^y").unwrap().to_python().unwrap(), "2**y");
}

// ═══════════════════════════════════════════════════════════════════════════
// Emitted text
// ═══════════════════════════════════════════════════════════════════════════

/// A Rust tail expression starting with a block parses as a statement:
/// `(y/x)^(-5)` was `{ let _p2 = x * x; _p2 * _p2 * x } * y.powi(-5)`,
/// which did not compile.
#[test]
fn rust_tail_blocks_keep_their_parentheses() {
    let ctx = Context::new();
    let rust = ctx
        .parse("(y/x)^(-5)")
        .unwrap()
        .to_rust_fn("f", &["x", "y"])
        .unwrap();
    assert!(rust.contains("({ let _p2"), "{rust}");
    rustc_lib(&rust, false, "tailblock");
}

/// Nested `sign`/`Heaviside` and cubes repeated their operand's text:
/// eight nested `sign`s were 436 KB of Rust and 1.6 MB of Python (3ⁿ and
/// 4ⁿ); twelve took minutes.  The operand is bound once when its own code
/// repeats.  Oracle for the value: `eval_f64` of the expression at x = 2.
#[test]
fn nested_repetition_stays_linear() {
    let ctx = Context::new();
    let mut s = "x - 1".to_string();
    for _ in 0..12 {
        s = format!("sign({s} - 1/3)");
    }
    let mut c = "x".to_string();
    for _ in 0..12 {
        c = format!("(({c})^3/4 + 1/3)");
    }
    let mut h = "x".to_string();
    for _ in 0..12 {
        h = format!("heaviside({h} - 1/2)");
    }
    for src in [&s, &c, &h] {
        let e = ctx.parse(src).unwrap();
        let rust = e.to_rust_fn("f", &["x"]).unwrap();
        assert!(rust.len() < 10_000, "{} bytes", rust.len());
        let py = e.to_python_fn("f", &["x", "y"]).unwrap();
        assert!(py.len() < 10_000, "{} bytes", py.len());
        let want = e
            .subs(&ctx.symbol("x"), &ctx.rational(1, 5))
            .eval_f64()
            .unwrap();
        let got = e.compile(&["x"]).unwrap().call(&[0.2]);
        assert!(rel(got, want) < 1e-12, "{src}: {got} vs {want}");
    }
    let rust = ctx.parse(&s).unwrap().to_rust_fn("f", &["x"]).unwrap();
    rustc_lib(&rust, false, "nested");
}

/// Python raises on a domain error, and a temporary is evaluated before
/// the expression: `Piecewise((√x + y√x, x > 0), (y, True))` hoisted
/// `t0 = math.sqrt(x)` out of its branch and raised `ValueError` at x < 0,
/// where the value is y (`eval_f64` at (−1, 2) = 2).
#[test]
fn python_temporaries_stay_in_their_branch() {
    let ctx = Context::new();
    let e = ctx
        .parse("Piecewise(sqrt(x) + y*sqrt(x) if x > 0, y if True)")
        .unwrap();
    let py = e.to_python_fn("f", &["x", "y"]).unwrap();
    assert!(!py.contains("t0 ="), "{py}");
    if let Some(out) = run_py(&py, &[(-1.0, 2.0), (4.0, 1.0)], false, "branch") {
        assert_eq!(out, ["2.0", "4.0"], "{py}");
    }
    // An occurrence outside the branches is still shared.
    let e = ctx
        .parse("sqrt(x) + Piecewise(sqrt(x)*y if x > 0, y if True)")
        .unwrap();
    assert!(
        e.to_python_fn("f", &["x", "y"])
            .unwrap()
            .contains("t0 = math.sqrt(x)")
    );
}

/// Integers past what the target converts: Python's `x/2**1074` raised
/// `OverflowError` (the denominator is not a float) and NumPy's
/// `numpy.sqrt(100000000000000000001)` a `TypeError` (beyond `int64`).
/// Oracles: `eval_f64` of `2^(-1074)` = 5e-324, of `sqrt(10^20+1)` = 1e10.
#[test]
fn python_and_numpy_big_integers() {
    let ctx = Context::new();
    let e = ctx.parse("x*2^(-1074)").unwrap();
    let py = e.to_python_fn("f", &["x", "y"]).unwrap();
    assert!(py.contains("5e-324"), "{py}");
    if let Some(out) = run_py(&py, &[(3.0, 0.0)], false, "bigpy") {
        assert_eq!(out[0], "1.5e-323", "{py}");
    }
    let e = ctx.parse("x*sqrt(10^20 + 1)").unwrap();
    let np = e.to_numpy_fn("f", &["x", "y"]).unwrap();
    if let Some(out) = run_py(&np, &[(2.0, 0.0)], true, "bignp") {
        // `np.float64(…)` (NumPy 2) or the bare float (NumPy 1).
        let v = out[0]
            .trim_start_matches("np.float64(")
            .trim_end_matches(')');
        assert_eq!(v.parse::<f64>().ok(), Some(2e10), "{np}: {}", out[0]);
    }
}

/// The `no_std` back ends printed `std::f64::consts::PI`, which does not
/// resolve under `#![no_std]`; now `core::`.  (The `Std` back end keeps
/// `std::`.)
#[test]
fn no_std_constants_come_from_core() {
    let ctx = Context::new();
    let e = ctx.parse("pi*sin(x) + E").unwrap();
    let code = e
        .to_rust_fn_with_options("f", &["x"], &CodegenOptions::no_std())
        .unwrap();
    assert!(code.contains("core::f64::consts::PI"), "{code}");
    rustc_lib(&code, true, "nostd");
    let opts = CodegenOptions {
        math_backend: MathBackend::Std,
        ..Default::default()
    };
    let std_code = e.to_rust_fn_with_options("f", &["x"], &opts).unwrap();
    assert!(std_code.contains("std::f64::consts::PI"), "{std_code}");
}

/// A node with a bound variable that depends on the arguments was refused
/// with the name of its bound variable (common subexpressions of the body
/// were lifted out of scope first): `∫ 1/(x⁵ − x − 1) dx` (a `RootSum`)
/// gave "free symbol `__rs_t`".  A constant one is a constant:
/// `x·RootOf(t⁵ − t − 1, 4)` (the real root; roots are ordered by real
/// part) was refused and now compiles (`eval_f64` of the `RootOf` =
/// 1.1673039782614187; mpmath `findroot(lambda t: t**5 - t - 1, 1.1673)` =
/// 1.1673039782614186843).
#[test]
fn bound_variable_nodes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.parse("1/(x^5 - x - 1)").unwrap().integrate(&x);
    assert!(i.to_string().contains("RootSum"), "{i}");
    let errs = [
        i.compile(&["x"]).err().map(|e| e.to_string()),
        i.to_rust_fn("f", &["x"]).err().map(|e| e.to_string()),
        i.to_c_fn("f", &["x"]).err().map(|e| e.to_string()),
        i.to_python_fn("f", &["x"]).err().map(|e| e.to_string()),
    ];
    for e in errs {
        let e = e.expect("refused");
        assert!(e.contains("RootSum") && !e.contains("free symbol"), "{e}");
    }
    let r = ctx.parse("x*RootOf(t^5 - t - 1, t, 4)").unwrap();
    let want = 1.167_303_978_261_418_7;
    assert!(rel(r.compile(&["x"]).unwrap().call(&[2.0]), 2.0 * want) < 1e-15);
    let code = r.to_c_fn("f", &["x"]).unwrap();
    if let Some(v) = run_c(&code, &[2.0], "rootof") {
        assert!(rel(v[0], 2.0 * want) < 1e-15, "C: {}", v[0]);
    }
    // A non-real root is refused as a non-real constant.
    let z = ctx.parse("x*RootOf(t^5 - t - 1, t, 0)").unwrap();
    assert!(
        z.compile(&["x"])
            .unwrap_err()
            .to_string()
            .contains("not real")
    );
}

/// The exact constant −1 (or 1) is what `expm1`/`log1p`/`1/x` stand for: a
/// rational that only rounds to it (`−1 − 10⁻²⁰`) made `exp(x) − 1 − 10⁻²⁰`
/// `x.exp_m1()` / `expm1(x)`, dropping the constant.
#[test]
fn numeric_shortcuts_need_exact_constants() {
    let ctx = Context::new();
    let e = ctx.parse("exp(x) - 1 - 10^-20").unwrap();
    assert!(!e.to_rust_fn("f", &["x"]).unwrap().contains("exp_m1"));
    assert!(!e.to_c_fn("f", &["x"]).unwrap().contains("expm1"));
    let e = ctx.parse("exp(x) - 1").unwrap();
    assert!(e.to_rust_fn("f", &["x"]).unwrap().contains("exp_m1"));
}
