//! 0.21 track: `base::libfn` — the closed registry of library special
//! functions carried as `Apply(name, args)`.
//!
//! Every consumer (`eval`, `evalf`, `diff`, the printers, the numeric back
//! ends, the parser) now matches on [`LibFn`] exhaustively instead of on
//! name strings.  The pins below are the observable behaviour that must
//! not have moved: names, arities, exact values, printer strings, the
//! functions each back end refuses — and the two gaps that were filled
//! (`to_c_fn` of `erfinv`/`erfcinv`; the parser accepting every registry
//! name).

use std::process::Command;

use symplex::base::libfn::{Arity, LibFn};
use symplex::prelude::*;

/// The registry as of 0.21, in declaration order.
const NAMES: [&str; 50] = [
    "factorial2",
    "subfactorial",
    "rising_factorial",
    "falling_factorial",
    "fibonacci",
    "lucas",
    "bernoulli",
    "harmonic",
    "catalan",
    "bell",
    "euler_number",
    "stirling1",
    "stirling2",
    "partition_count",
    "lambertw",
    "besselj",
    "bessely",
    "besseli",
    "besselk",
    "legendre",
    "chebyshev_t",
    "chebyshev_u",
    "hermite",
    "laguerre",
    "erfi",
    "erfinv",
    "erfcinv",
    "expint",
    "Shi",
    "Chi",
    "fresnels",
    "fresnelc",
    "lowergamma",
    "uppergamma",
    "polylog",
    "dirichlet_eta",
    "airyai",
    "airybi",
    "airyaiprime",
    "airybiprime",
    "elliptic_k",
    "elliptic_e",
    "elliptic_f",
    "elliptic_pi",
    "gegenbauer",
    "jacobi",
    "assoc_legendre",
    "assoc_laguerre",
    "betainc",
    "betainc_regularized",
];

/// `f` applied to the first `arity` of the symbols `a, b, c, d`.
fn generic_call(ctx: &Context, f: LibFn) -> Ex {
    let syms = [
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("c"),
        ctx.symbol("d"),
    ];
    let Arity::Fixed(n) = f.arity() else {
        panic!("{f}: every registry function has a fixed arity");
    };
    ctx.apply(f.name(), &syms[..n as usize])
}

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    let err = ((got - want) / want).abs();
    assert!(
        err <= tol,
        "{what}: got {got:.17e}, want {want:.17e}, rel err {err:.2e}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// The registry itself
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn registry_is_the_fifty_named_functions() {
    assert_eq!(LibFn::ALL.len(), 50);
    let names: Vec<&str> = LibFn::ALL.iter().map(|f| f.name()).collect();
    assert_eq!(names, NAMES.to_vec());
    for &f in LibFn::ALL {
        assert_eq!(LibFn::from_name(f.name()), Some(f), "{f}");
        assert_eq!(f.to_string(), f.name());
        assert_eq!(
            LibFn::from_name_ignore_ascii_case(&f.name().to_ascii_uppercase()),
            Some(f),
            "{f}"
        );
    }
    assert_eq!(LibFn::from_name("sin"), None);
    assert_eq!(LibFn::from_name("shi"), None, "from_name is exact");
    assert_eq!(LibFn::from_name_ignore_ascii_case("shi"), Some(LibFn::Shi));
    assert_eq!(LibFn::from_name(""), None);
}

#[test]
fn arity_table() {
    // Spot checks against the `eval` guards the table was derived from.
    let expect = [
        (LibFn::Factorial2, 1),
        (LibFn::RisingFactorial, 2),
        (LibFn::Stirling2, 2),
        (LibFn::LambertW, 1),
        (LibFn::BesselJ, 2),
        (LibFn::Legendre, 2),
        (LibFn::Erfi, 1),
        (LibFn::ExpInt, 2),
        (LibFn::Shi, 1),
        (LibFn::LowerGamma, 2),
        (LibFn::PolyLog, 2),
        (LibFn::AiryAiPrime, 1),
        (LibFn::EllipticK, 1),
        (LibFn::EllipticF, 2),
        (LibFn::EllipticPi, 2),
        (LibFn::Gegenbauer, 3),
        (LibFn::Jacobi, 4),
        (LibFn::AssocLegendre, 3),
        (LibFn::AssocLaguerre, 3),
        (LibFn::BetaInc, 4),
        (LibFn::BetaIncRegularized, 4),
    ];
    for (f, n) in expect {
        assert_eq!(f.arity(), Arity::Fixed(n), "{f}");
        assert!(f.arity().accepts(n as usize));
        assert!(!f.arity().accepts(n as usize + 1));
    }
    // No registry function is variadic, and every arity is 1..=4.
    for &f in LibFn::ALL {
        match f.arity() {
            Arity::Fixed(n) => assert!((1..=4).contains(&n), "{f}: arity {n}"),
            Arity::Range { .. } => panic!("{f}: unexpected variadic arity"),
        }
    }
    let counts: Vec<usize> = (1..=4)
        .map(|k| {
            LibFn::ALL
                .iter()
                .filter(|f| f.arity() == Arity::Fixed(k))
                .count()
        })
        .collect();
    assert_eq!(counts, [25, 19, 3, 3]);
    assert!(Arity::Range { min: 1, max: 3 }.accepts(2));
    assert!(!Arity::Range { min: 1, max: 3 }.accepts(4));
}

// ═══════════════════════════════════════════════════════════════════════════
// eval / evalf: values unchanged, wrong arity left alone
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_folds_exactly_as_before() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: Vec<(Ex, &str)> = vec![
        (ctx.int(10).fibonacci(), "55"),
        (ctx.int(7).lucas(), "29"),
        (ctx.int(6).factorial2(), "48"),
        (ctx.int(4).subfactorial(), "9"),
        (ctx.int(5).stirling2(&ctx.int(2)), "15"),
        (ctx.int(4).stirling1(&ctx.int(2)), "11"),
        (ctx.int(5).partition_count(), "7"),
        (ctx.int(4).harmonic(), "25/12"),
        (ctx.int(4).bernoulli_number(), "-1/30"),
        (ctx.int(4).catalan_number(), "14"),
        (ctx.int(4).bell(), "15"),
        (ctx.int(4).euler_number(), "5"),
        (ctx.int(3).rising_factorial(&ctx.int(4)), "360"),
        (ctx.int(6).falling_factorial(&ctx.int(3)), "120"),
        (ctx.int(0).bessel_j(&ctx.int(0)), "1"),
        (ctx.int(0).bessel_j(&ctx.int(3)), "0"),
        (ctx.int(0).bessel_i(&ctx.int(0)), "1"),
        (ctx.int(0).bessel_y(&ctx.int(0)), "bessely(0, 0)"),
        (x.legendre(&ctx.int(2)), "3/2*x^2 - 1/2"),
        (x.chebyshev_t(&ctx.int(3)), "4*x^3 - 3*x"),
        (x.hermite(&ctx.int(2)), "4*x^2 - 2"),
        (x.laguerre(&ctx.int(1)), "-x + 1"),
        (x.legendre(&ctx.symbol("n")), "legendre(n, x)"),
        (ctx.int(0).erfi(), "0"),
        (ctx.int(0).erfinv(), "0"),
        (ctx.int(1).erfcinv(), "0"),
        (ctx.int(0).expint(&ctx.int(3)), "1/2"),
        (ctx.int(0).elliptic_k(), "1/2*pi"),
        (ctx.int(0).airyai(), "cbrt(3)/(3*Gamma(2/3))"),
        (ctx.int(1).dirichlet_eta(), "ln(2)"),
        (ctx.rational(1, 2).polylog(&ctx.int(1)), "-ln(1/2)"),
        (x.gegenbauer(&ctx.int(1), &ctx.symbol("a")), "2*a*x"),
    ];
    let mut bad = Vec::new();
    for (e, want) in cases {
        let got = format!("{}", e.eval());
        if got != want {
            bad.push(format!("{e}: got {got:?}, want {want:?}"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn wrong_arity_is_left_unevaluated_and_unevaluable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one_arg_bessel = ctx.apply("besselj", &[&x]);
    assert_eq!(one_arg_bessel.eval(), one_arg_bessel);
    assert_eq!(format!("{one_arg_bessel}"), "besselj(x)");
    let fib2 = ctx.apply("fibonacci", &[ctx.int(5), ctx.int(6)]);
    assert_eq!(fib2.eval(), fib2, "two-argument fibonacci is not folded");
    assert_eq!(
        format!("{}", ctx.apply("fibonacci", &[ctx.int(5)]).eval()),
        "5"
    );
    // evalf: a library name with the wrong number of arguments is an error,
    // never a value.
    let erfi2 = ctx.apply("erfi", &[ctx.int(1), ctx.int(2)]);
    let err = erfi2.eval_f64().expect_err("erfi/2 has no value");
    assert!(err.to_string().contains("erfi"), "{err}");
    // A user function of the same shape stays a user function.
    let user = ctx.apply("f", &[&x]);
    assert_eq!(user.eval(), user);
    assert!(user.eval_f64().is_err());
    // Substituted arguments are re-evaluated through the same fold.
    let n = ctx.symbol("n");
    assert_eq!(format!("{}", n.fibonacci().subs_i64(&n, 12).eval()), "144");
}

#[test]
fn evalf_values_unchanged() {
    let ctx = Context::new();
    // mpmath / SciPy references (dps 30).
    let cases: Vec<(Ex, f64)> = vec![
        (ctx.int(1).bessel_j(&ctx.int(0)), 0.7651976865579666), // scipy.special.jv(0, 1)
        (ctx.int(1).bessel_y(&ctx.int(1)), -0.7812128213002887), // scipy.special.yv(1, 1)
        (ctx.int(1).bessel_i(&ctx.int(0)), 1.2660658777520084), // scipy.special.iv(0, 1)
        (ctx.int(1).bessel_k(&ctx.int(0)), 0.42102443824070834), // scipy.special.kv(0, 1)
        (ctx.rational(1, 2).erfi(), 0.614952094696511),         // mpmath.erfi(0.5)
        (ctx.rational(1, 2).erfinv(), 0.4769362762044699),      // scipy.special.erfinv(0.5)
        (ctx.rational(1, 10).erfcinv(), 1.1630871536766743),    // scipy.special.erfcinv(0.1)
        (ctx.rational(1, 2).elliptic_k(), 1.8540746773013719),  // scipy.special.ellipk(0.5)
        (ctx.rational(1, 2).legendre(&ctx.int(3)), -0.4375),    // P_3(1/2)
        (
            ctx.rational(3, 2).uppergamma(&ctx.int(2)),
            0.5578254003710745,
        ), // mpmath.gammainc(2, 1.5)
        (ctx.rational(1, 2).airyai(), 0.23169360648083348),     // scipy.special.airy(0.5)[0]
        (ctx.int(2).expint(&ctx.int(1)), 0.04890051070806112),  // scipy.special.expn(1, 2)
    ];
    for (e, want) in cases {
        let got = e.eval_f64().unwrap_or_else(|err| panic!("{e}: {err}"));
        assert_rel(got, want, 1e-13, &e.to_string());
    }
    // Degree checks keep their message, which names the function.
    let bad = ctx.rational(1, 2).legendre(&ctx.rational(1, 2));
    let err = bad.eval_f64().expect_err("half-integer degree");
    assert!(err.to_string().contains("legendre"), "{err}");
}

#[test]
fn diff_rules_unchanged_and_no_rule_functions_stay_formal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        format!("{}", x.bessel_j(&ctx.int(1)).diff(&x)),
        "-1/2*besselj(2, x) + 1/2*besselj(0, x)"
    );
    assert_eq!(format!("{}", x.erfi().diff(&x)), "2*exp(x^2)/sqrt(pi)");
    assert_eq!(format!("{}", x.airyai().diff(&x)), "airyaiprime(x)");
    assert_eq!(
        format!("{}", x.hermite(&ctx.int(3)).diff(&x)),
        "6*hermite(2, x)"
    );
    // The integer sequences have no derivative rule: a formal derivative,
    // exactly as for a user function.
    for f in [
        LibFn::Factorial2,
        LibFn::Fibonacci,
        LibFn::Harmonic,
        LibFn::PartitionCount,
    ] {
        let e = ctx.apply(f.name(), &[&x]);
        let d = e.diff(&x);
        assert_eq!(
            format!("{d}"),
            format!("Derivative({}(x), x)", f.name()),
            "{f}"
        );
    }
    let s2 = x.stirling2(&ctx.int(2));
    assert_eq!(format!("{}", s2.diff(&x)), "Derivative(stirling2(x, 2), x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Printers
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_and_latex_unchanged() {
    let ctx = Context::new();
    let (x, n) = (ctx.symbol("x"), ctx.symbol("n"));
    assert_eq!(format!("{}", x.bessel_j(&n)), "besselj(n, x)");
    assert_eq!(format!("{}", x.uppergamma(&n)), "uppergamma(n, x)");
    assert_eq!(format!("{}", x.shi()), "Shi(x)");
    assert_eq!(x.erfi().to_latex(), r"\operatorname{erfi}\left(x\right)");
    assert_eq!(
        x.erfinv().to_latex(),
        r"\operatorname{erf}^{-1}\left(x\right)"
    );
    assert_eq!(x.uppergamma(&n).to_latex(), r"\Gamma\left(n, x\right)");
    assert_eq!(
        x.expint(&ctx.int(2)).to_latex(),
        r"\operatorname{E}_{2}\left(x\right)"
    );
    assert_eq!(x.elliptic_f(&n).to_latex(), r"F\left(x\middle| n\right)");
    assert_eq!(
        x.jacobi(&n, &ctx.symbol("a"), &ctx.symbol("b")).to_latex(),
        r"P_{n}^{\left(a,b\right)}\left(x\right)"
    );
    assert_eq!(
        x.betainc_regularized(&ctx.symbol("a"), &ctx.symbol("b"), &ctx.int(0))
            .to_latex(),
        r"\operatorname{I}_{(0, x)}\left(a, b\right)"
    );
    // Functions without a SymPy notation print as plain calls, as do user
    // functions and calls of the wrong arity.
    assert_eq!(x.bessel_j(&n).to_latex(), r"besselj\left(n, x\right)");
    assert_eq!(n.fibonacci().to_latex(), r"fibonacci\left(n\right)");
    assert_eq!(x.legendre(&n).to_latex(), r"legendre\left(n, x\right)");
    assert_eq!(
        ctx.apply("erfi", &[&x, &n]).to_latex(),
        r"erfi\left(x, n\right)"
    );
    assert_eq!(ctx.apply("f", &[&x]).to_latex(), r"f\left(x\right)");
}

#[test]
fn mathml_renders_every_library_function() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let g = x.uppergamma(&a).to_mathml().unwrap();
    assert!(g.contains("<mi>uppergamma</mi>"), "{g}");
    assert!(g.contains("<mo>(</mo>"), "{g}");
    let j = x.bessel_j(&ctx.int(2)).to_mathml().unwrap();
    assert!(j.contains("<msub><mi>J</mi><mn>2</mn></msub>"), "{j}");
    // Every registry function has a presentation (by name unless given a
    // notation), for any arity.
    for &f in LibFn::ALL {
        let e = generic_call(&ctx, f);
        let xml = e.to_mathml().unwrap_or_else(|err| panic!("{f}: {err}"));
        let visible = matches!(
            f,
            LibFn::BesselJ | LibFn::BesselY | LibFn::BesselI | LibFn::BesselK
        ) || xml.contains(&format!("<mi>{}</mi>", f.name()));
        assert!(visible, "{f}: {xml}");
    }
    let user = ctx.apply("f", &[&x]).to_mathml().unwrap();
    assert!(user.contains("<mi>f</mi>"), "{user}");
}

#[test]
fn lean_refuses_every_library_function_by_name() {
    let ctx = Context::new();
    // Mathlib has no spelling for any of the fifty; the error names the
    // function rather than the node kind.
    for &f in LibFn::ALL {
        let e = generic_call(&ctx, f);
        match e.to_lean() {
            Err(SymplexError::NotImplemented(msg)) => {
                assert!(msg.contains(f.name()), "{f}: {msg}");
                assert!(msg.contains("special function"), "{f}: {msg}");
            }
            other => panic!("{f}: {other:?}"),
        }
    }
}

#[test]
fn parser_accepts_every_registry_name_at_its_arity() {
    let ctx = Context::new();
    // Display → parse round-trips for every function the arena can build
    // (`lambertw` parses to the dedicated `LambertW` node instead).
    for &f in LibFn::ALL {
        if f == LibFn::LambertW {
            continue;
        }
        let e = generic_call(&ctx, f);
        let back = ctx
            .parse(&e.to_string())
            .unwrap_or_else(|err| panic!("{f}: {err}"));
        assert_eq!(back, e, "{f}");
    }
    assert_eq!(
        ctx.parse("lambertw(x)").unwrap(),
        ctx.symbol("x").lambertw()
    );
    // Case-insensitive, as every parser name is.
    assert_eq!(ctx.parse("shi(x)").unwrap(), ctx.symbol("x").shi());
    assert_eq!(ctx.parse("SHI(x)").unwrap(), ctx.symbol("x").shi());
    // New in 0.21: the integer sequences and classical polynomials, which
    // Display already printed, parse back.
    assert_eq!(
        format!("{}", ctx.parse("fibonacci(10)").unwrap().eval()),
        "55"
    );
    assert_eq!(
        format!("{}", ctx.parse("legendre(2, x)").unwrap().eval()),
        "3/2*x^2 - 1/2"
    );
    // Wrong arity is still "unknown function", not a guess.
    for src in ["besselj(x)", "erfi(x, y)", "jacobi(a, b, x)", "fibonacci()"] {
        assert!(ctx.parse(src).is_err(), "{src}");
    }
    let err = ctx.parse("besselj(x)").expect_err("wrong arity");
    assert!(err.to_string().contains("unknown function"), "{err}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric back ends
// ═══════════════════════════════════════════════════════════════════════════

fn find_cc() -> Option<String> {
    ["cc", "clang", "gcc"]
        .into_iter()
        .find(|cc| {
            Command::new(cc)
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
        })
        .map(str::to_string)
}

#[test]
fn codegen_c_erfinv_and_erfcinv_compile_to_helpers() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 0.20 refused these ("user-defined Apply node `erfinv`"); the Rust
    // runtime had them all along.
    let code = x.erfinv().to_c_fn("f", &["x"]).unwrap();
    assert!(code.contains("return symplex_erfinv(x);"), "{code}");
    assert!(
        code.contains("static inline double symplex_erfinv(double x)"),
        "{code}"
    );
    let code = x.erfcinv().to_c_fn("g", &["x"]).unwrap();
    assert!(code.contains("return symplex_erfcinv(x);"), "{code}");
    // The Rust and VM back ends still agree with each other.
    let rust = x.erfinv().to_rust_fn("f", &["x"]).unwrap();
    assert!(rust.contains("symplex_rt::erfinv("), "{rust}");
    let f = x.erfinv().compile(&["x"]).unwrap();
    assert_rel(f(&[0.5]), 0.4769362762044699, 1e-15, "compile erfinv(0.5)");
}

/// Compile `code` (a complete `to_c_fn` unit for `double f(double x)`) with
/// `cc`, call `f` at each probe and return the printed values.
fn run_c_unary(cc: &str, code: &str, args: &[f64]) -> Vec<f64> {
    let mut src = String::from("#include <stdio.h>\n");
    src.push_str(code);
    src.push_str("int main(void) {\n");
    for a in args {
        src.push_str(&format!("    printf(\"%.17g\\n\", f({a:?}));\n"));
    }
    src.push_str("    return 0;\n}\n");
    let dir = std::env::temp_dir().join(format!(
        "symplex_libfn_c_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let c_file = dir.join("gen.c");
    std::fs::write(&c_file, &src).unwrap();
    let exe = dir.join("gen_bin");
    let out = Command::new(cc)
        .args(["-std=c99", "-O2", "-Wall", "-Werror", "-o"])
        .arg(&exe)
        .arg(&c_file)
        .arg("-lm")
        .output()
        .expect("run C compiler");
    assert!(
        out.status.success(),
        "generated C failed to compile:\n{}\n--- source ---\n{src}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    assert!(run.status.success());
    let stdout = String::from_utf8_lossy(&run.stdout);
    let _ = std::fs::remove_dir_all(&dir);
    stdout
        .lines()
        .map(|l| {
            l.trim()
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("bad C output {l:?}"))
        })
        .collect()
}

#[test]
fn codegen_c_erfinv_evaluates_like_scipy() {
    let Some(cc) = find_cc() else {
        eprintln!("no C compiler found; skipping end-to-end C test");
        return;
    };
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // scipy.special.erfinv(0.5), erfinv(0.9), erfinv(-0.999);
    // scipy.special.erfcinv(0.1), erfcinv(1e-10), erfcinv(1.5).
    let cases: [(Ex, [(f64, f64); 3]); 2] = [
        (
            x.erfinv(),
            [
                (0.5, 0.4769362762044699),
                (0.9, 1.1630871536766743),
                (-0.999, -2.326753765513524),
            ],
        ),
        (
            x.erfcinv(),
            [
                (0.1, 1.1630871536766743),
                (1e-10, 4.572824967389486),
                (1.5, -0.4769362762044699),
            ],
        ),
    ];
    for (e, probes) in &cases {
        let code = e.to_c_fn("f", &["x"]).unwrap();
        let args: Vec<f64> = probes.iter().map(|p| p.0).collect();
        let got = run_c_unary(&cc, &code, &args);
        assert_eq!(got.len(), probes.len());
        let vm = e.compile(&["x"]).unwrap();
        for ((arg, want), c) in probes.iter().zip(got) {
            assert_rel(c, *want, 1e-14, &format!("C {e} at {arg}"));
            assert_rel(c, vm(&[*arg]), 1e-14, &format!("C vs compile {e} at {arg}"));
        }
    }
}

#[test]
fn codegen_py_names_the_scipy_routine_it_refuses() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // `to_python` is `math`-only and `to_numpy` is `numpy`-only: every
    // library function is still refused (pinned since 0.9), but the error
    // now cites the SciPy routine with the same value.
    let j = x.bessel_j(&ctx.int(0));
    for (what, result) in [
        ("python", j.to_python()),
        ("numpy", j.to_numpy()),
        ("julia", j.to_julia()),
    ] {
        match result {
            Err(SymplexError::NotImplemented(msg)) => {
                assert!(msg.contains("scipy.special.jv"), "{what}: {msg}");
            }
            other => panic!("{what}: {other:?}"),
        }
    }
    let cited = [
        (x.erfinv(), "scipy.special.erfinv"),
        (x.uppergamma(&ctx.int(2)), "scipy.special.gammaincc"),
        (x.elliptic_k(), "scipy.special.ellipk"),
        (x.expint(&ctx.int(2)), "scipy.special.expn"),
        (x.airyai(), "scipy.special.airy"),
        (x.legendre(&ctx.int(2)), "scipy.special.eval_legendre"),
    ];
    for (e, scipy) in cited {
        let err = e.to_python().expect_err("refused");
        assert!(err.to_string().contains(scipy), "{e}: {err}");
    }
    // No SciPy routine: refused with the name alone.
    let err = x.fibonacci().to_python().expect_err("refused");
    assert!(err.to_string().contains("fibonacci"), "{err}");
    assert!(!err.to_string().contains("scipy"), "{err}");
    let err = ctx.apply("f", &[&x]).to_python().expect_err("refused");
    assert!(err.to_string().contains("`f`"), "{err}");
}

#[test]
fn numeric_back_ends_refuse_the_same_functions() {
    let ctx = Context::new();
    // The functions without an `f64` runtime routine are refused by
    // `compile`, `to_rust_fn` and `to_c_fn` alike, by name.
    let refused = [
        LibFn::Subfactorial,
        LibFn::Bernoulli,
        LibFn::Catalan,
        LibFn::Bell,
        LibFn::EulerNumber,
        LibFn::Stirling1,
        LibFn::Stirling2,
        LibFn::PartitionCount,
        LibFn::LambertW,
        LibFn::Erfi,
        LibFn::ExpInt,
        LibFn::Shi,
        LibFn::Chi,
        LibFn::FresnelS,
        LibFn::FresnelC,
        LibFn::LowerGamma,
        LibFn::UpperGamma,
        LibFn::PolyLog,
        LibFn::DirichletEta,
        LibFn::AiryAi,
        LibFn::AiryBi,
        LibFn::AiryAiPrime,
        LibFn::AiryBiPrime,
        LibFn::EllipticK,
        LibFn::EllipticE,
        LibFn::EllipticF,
        LibFn::EllipticPi,
        LibFn::Gegenbauer,
        LibFn::Jacobi,
        LibFn::AssocLegendre,
        LibFn::AssocLaguerre,
        LibFn::BetaInc,
        LibFn::BetaIncRegularized,
    ];
    let params = ["a", "b", "c", "d"];
    for f in refused {
        let e = generic_call(&ctx, f);
        for (what, res) in [
            ("compile", e.compile(&params).map(|_| ())),
            ("to_rust_fn", e.to_rust_fn("f", &params).map(|_| ())),
            ("to_c_fn", e.to_c_fn("f", &params).map(|_| ())),
        ] {
            match res {
                Err(SymplexError::NotImplemented(msg)) => {
                    assert!(msg.contains(f.name()), "{f} {what}: {msg}");
                }
                other => panic!("{f} {what}: {other:?}"),
            }
        }
    }
    // And the seventeen with a routine all compile.
    let x = ctx.symbol("x");
    let supported: Vec<Ex> = vec![
        x.bessel_j(&ctx.int(1)),
        x.bessel_y(&ctx.int(1)),
        x.bessel_i(&ctx.int(1)),
        x.bessel_k(&ctx.int(1)),
        x.legendre(&ctx.int(3)),
        x.chebyshev_t(&ctx.int(3)),
        x.chebyshev_u(&ctx.int(3)),
        x.hermite(&ctx.int(3)),
        x.laguerre(&ctx.int(3)),
        x.fibonacci(),
        x.lucas(),
        x.harmonic(),
        x.factorial2(),
        x.erfinv(),
        x.erfcinv(),
        x.rising_factorial(&ctx.int(2)),
        x.falling_factorial(&ctx.int(2)),
    ];
    assert_eq!(supported.len(), 17);
    for e in supported {
        e.compile(&["x"]).unwrap_or_else(|err| panic!("{e}: {err}"));
        e.to_rust_fn("f", &["x"])
            .unwrap_or_else(|err| panic!("{e}: {err}"));
        e.to_c_fn("f", &["x"])
            .unwrap_or_else(|err| panic!("{e}: {err}"));
    }
}

#[test]
fn names_are_resolved_from_text_not_pre_interned() {
    // A fresh context interns nothing for the registry: the node count is
    // the 0.20 constant set, and the first user symbol is unaffected by
    // library nodes built before it (`SymbolId`s are only ever assigned in
    // the user's interning order).
    let ctx = Context::new();
    let fresh = ctx.node_count();
    assert_eq!(fresh, Context::new().node_count());
    let x = ctx.symbol("x");
    assert_eq!(ctx.node_count(), fresh + 1);
    let _ = x.erfi();
    assert_eq!(
        ctx.node_count(),
        fresh + 2,
        "one Apply node, no hidden symbols"
    );
    // The same name built two ways is the same node.
    assert_eq!(ctx.apply("erfi", &[&x]), x.erfi());
    assert_eq!(
        ctx.apply("besselj", &[ctx.int(2), x.clone()]),
        x.bessel_j(&ctx.int(2))
    );
}
