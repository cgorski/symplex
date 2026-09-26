//! After 0.29 — the leftovers of the hand-off: residues at essential
//! singularities, inverse trigonometric functions of special algebraic
//! values, performance of polynomial composition / symbolic resultants /
//! real-root isolation with huge denominators, the emitted code of the
//! back ends, and a hunt through units and non-polynomial inequalities.
//!
//! Each test says what was wrong before; every reference value cites the
//! oracle call that produced it (SymPy 1.14, mpmath 1.3.0 at the stated
//! `mp.dps`).

use symplex::prelude::*;

/// `|v − expected| ≤ tol·max(1, |expected|)` for the value of `e`.
fn assert_close(e: &Ex, expected: f64, tol: f64, what: &str) {
    let v = e
        .eval_f64()
        .unwrap_or_else(|err| panic!("{what}: {e} does not evaluate: {err}"));
    assert!(
        (v - expected).abs() <= tol * expected.abs().max(1.0),
        "{what}: {e} = {v}, expected {expected}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Residues at essential singularities
// ═══════════════════════════════════════════════════════════════════════════

/// The residue at an essential singularity of `P(z)·h(1/(z − z0))` (`P`
/// polynomial, `h` entire) is a finite sum of Laurent coefficients.  Up to
/// 0.29 these residues were `0`; after the first fix they were refused.
///
/// mpmath 1.3.0, dps 30, `res(f, c, r) = quad(lambda th: f(c + r*expj(th))
/// * r*expj(th), [0, pi/2, pi, 3*pi/2, 2*pi]) / (2*pi)`:
///   `z**2*exp(1/z)`, c = 0, r = 1        -> 0.1666666666666666666666667
///   `z*exp(1/z)`, c = 0, r = 1           -> 0.5
///   `z**2*sin(1/z)`, c = 0, r = 1        -> -0.1666666666666666666666667
///   `exp(1/z)`, c = 0, r = 1             -> 1.0
///   `exp(-1/z)`, c = 0, r = 1            -> -1.0
///   `z*cos(1/(2*(z-1)))`, c = 1, r = 1/2 -> -0.125
///   `z**5*exp(1/z**2)`, c = 0, r = 1     -> 0.1666666666666666666666667
///   `cosh(1/z)*z**3`, c = 0, r = 1       -> 0.04166666666666666666666667
///   `z**3*cos(1/z)*sin(1/z)`, c = 0      -> -2.5e-32 (0)
///   `erf(1/z)*z`, c = 0, r = 1           -> -4.5e-32 (0)
///   `2**(1/z)*z`, c = 0, r = 1           -> 0.2402265069591007123335513
#[test]
fn residues_at_essential_singularities_times_polynomials() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    for (f, point, expected) in [
        ("z^2*exp(1/z)", 0, ctx.rational(1, 6)),
        ("z*exp(1/z)", 0, ctx.rational(1, 2)),
        ("z^2*sin(1/z)", 0, ctx.rational(-1, 6)),
        ("exp(1/z)", 0, ctx.int(1)),
        ("exp(-1/z)", 0, ctx.int(-1)),
        ("z*cos(1/2/(z - 1))", 1, ctx.rational(-1, 8)),
        ("z^5*exp(1/z^2)", 0, ctx.rational(1, 6)),
        ("cosh(1/z)*z^3", 0, ctx.rational(1, 24)),
        ("z^3*cos(1/z)*sin(1/z)", 0, ctx.int(0)),
        ("erf(1/z)*z", 0, ctx.int(0)),
    ] {
        let e = ctx.parse(f).unwrap();
        let r = e.try_residue(&z, &ctx.int(point));
        assert_eq!(r.ok(), Some(expected), "Res({f}, {point})");
    }
    let r = ctx.parse("2^(1/z)*z").unwrap().try_residue(&z, &ctx.int(0));
    assert_close(&r.unwrap(), 0.2402265069591007, 1e-14, "Res(2^(1/z) z, 0)");
    // A parameter: Res(z²·e^{a/z}, 0) = a³/3! (the Laurent coefficient).
    let a = ctx.symbol("a");
    let r = ctx
        .parse("z^2*exp(a/z)")
        .unwrap()
        .try_residue(&z, &ctx.int(0));
    assert_eq!(r.unwrap(), (&a.powi(3) / 6).eval());
}

/// With a rational factor that has other poles the Laurent series is
/// infinite; it sums to the closed form `−Res_∞ − Σ Res_p` over the other
/// poles.  Refused before.
///
/// mpmath 1.3.0, dps 30, `res(f, 0, 1/2)` as above:
///   `exp(1/z)/(z-1)`       -> -1.718281828459045235360287   (1 − e)
///   `exp(1/z)/(z-1)**2`    -> 2.718281828459045235360287    (e)
///   `sin(1/z)/(z**2+1)`    -> 1.175201193643801456882382    (sinh 1)
///   `exp(1/z)/(z**3-2)`    -> -0.5104414854823119670160028
#[test]
fn residues_at_essential_singularities_times_rational_functions() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    for (f, expected) in [
        ("exp(1/z)/(z-1)", -1.718281828459045),
        ("exp(1/z)/(z-1)^2", std::f64::consts::E),
        ("sin(1/z)/(z^2+1)", 1.1752011936438014),
        ("exp(1/z)/(z^3-2)", -0.510441485482312),
    ] {
        let r = ctx.parse(f).unwrap().try_residue(&z, &ctx.int(0));
        let r = r.unwrap_or_else(|e| panic!("Res({f}, 0): {e}"));
        assert_close(&r, expected, 1e-13, f);
    }
    let r = ctx
        .parse("exp(1/z)/(z-1)")
        .unwrap()
        .try_residue(&z, &ctx.int(0));
    assert_eq!(r.unwrap(), ctx.parse("1 - E").unwrap());
}

/// What the residue theorem cannot close stays formal: `e^z·e^{1/z}` and
/// `e^{z+1/z}` are essential at `∞` too (the residue is the Bessel value
/// `I₁(2)` = mpmath `besseli(1, 2)` = 1.590636854637329063382254, equal
/// to `res(exp(z)*exp(1/z), 0, 1)`), `1/(e^{1/z} − 1)` has poles
/// accumulating at 0, `√z·e^{1/z}` a branch point.
#[test]
fn residues_that_do_not_close_stay_formal() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    for f in [
        "exp(z)*exp(1/z)",
        "exp(z + 1/z)",
        "1/(exp(1/z) - 1)",
        "sqrt(z)*exp(1/z)",
        "exp(1/(z - 1))*exp(1/z)",
    ] {
        let e = ctx.parse(f).unwrap();
        assert!(e.try_residue(&z, &ctx.int(0)).is_err(), "Res({f}, 0)");
    }
    // An entire function has residue 0 at ∞ (formal before).
    let g = ctx.parse("exp(z^2)").unwrap();
    assert_eq!(g.residue_at_infinity(&z), ctx.int(0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse trigonometric functions of special algebraic values
// ═══════════════════════════════════════════════════════════════════════════

/// `eval` reduced only `0`, `±1/2`, `±1` (and `atan(±1)`); every other
/// `cos(kπ/n)` / `tan(kπ/n)` stayed unevaluated, whatever its radical form.
/// Reference: SymPy 1.14 `sympify(c)` (which reduces the ones marked `*`)
/// and, for all, `nsimplify(N(sympify(c)/pi, 50), tolerance=1e-40)`.
#[test]
fn inverse_trig_of_special_algebraic_values_reduce() {
    let ctx = Context::new();
    for (c, k, n) in [
        ("atan2(sqrt(3)/2, 1/2)", 1, 3),      // * pi/3
        ("atan(sqrt(3))", 1, 3),              // * pi/3
        ("acos(sqrt(2)/2)", 1, 4),            // * pi/4
        ("acos(-sqrt(3)/2)", 5, 6),           // * 5*pi/6
        ("asin(sqrt(2+sqrt(3))/2)", 5, 12),   //   5/12
        ("asin(-sqrt(2+sqrt(3))/2)", -5, 12), //   -5/12
        ("acos((sqrt(6)+sqrt(2))/4)", 1, 12), // * pi/12
        ("acos((sqrt(6)-sqrt(2))/4)", 5, 12), // * 5*pi/12
        ("acos(1/(sqrt(6)-sqrt(2)))", 1, 12), //   1/12
        ("acos((1+sqrt(5))/4)", 1, 5),        // * pi/5
        ("acos(-(1+sqrt(5))/4)", 4, 5),       // * 4*pi/5
        ("acos(GoldenRatio/2)", 1, 5),        //   1/5
        ("acos((sqrt(5)-1)/4)", 2, 5),        // * 2*pi/5
        ("asin(sqrt(10+2*sqrt(5))/4)", 2, 5), //   2/5
        ("asin(sqrt(10-2*sqrt(5))/4)", 1, 5), //   1/5
        ("acos(sqrt(2-sqrt(2))/2)", 3, 8),    // * 3*pi/8
        ("atan(2-sqrt(3))", 1, 12),           // * pi/12
        ("atan(-2-sqrt(3))", -5, 12),         // * -5*pi/12
        ("atan(sqrt(5-2*sqrt(5)))", 1, 5),    // * pi/5
        ("atan(1/sqrt(3))", 1, 6),            // * pi/6
        ("atan(sqrt(2)-1)", 1, 8),            // * pi/8
        ("atan(1+sqrt(2))", 3, 8),            // * 3*pi/8
        ("atan(sqrt(1-2/sqrt(5)))", 1, 10),   // * pi/10
        ("atan2(-sqrt(2), -sqrt(6))", -5, 6), // * -5*pi/6
        ("atan2(sqrt(2), -sqrt(2))", 3, 4),   // * 3*pi/4
        ("atan2(-1, sqrt(3))", -1, 6),        // * -pi/6
    ] {
        let got = ctx.parse(c).unwrap().eval();
        let want = (&ctx.pi() * &ctx.rational(k, n)).eval();
        assert_eq!(got, want, "{c}");
    }
}

/// Values that are not special stay unevaluated — including values within
/// `10⁻³⁰` of a special one (the numeric candidate is rejected by the
/// minimal polynomial).  SymPy 1.14 leaves the first four unevaluated;
/// `nsimplify(N(sympify(c)/pi, 50), tolerance=1e-40)` is not a small
/// rational for any of them.
#[test]
fn inverse_trig_of_other_values_stays_put() {
    let ctx = Context::new();
    for c in [
        "atan2(sqrt(3), 2)",
        "acos(sqrt(2)/3)",
        "atan(sqrt(2))",
        "asin(sqrt(3)/3)",
        "acos(sqrt(3)/2 + 10^(-30))",
        "acos(sqrt(3)/2 + 10^(-30)*sqrt(2))",
        "atan(2 + sqrt(3) + 10^(-40)*sqrt(5))",
    ] {
        let got = ctx.parse(c).unwrap().eval();
        assert!(!got.contains(&ctx.pi()), "{c} -> {got}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Emitted code agrees with compile()
// ═══════════════════════════════════════════════════════════════════════════

/// A Python interpreter: the project venv's, else `python3` on the path.
fn python() -> Option<String> {
    let venv = concat!(env!("CARGO_MANIFEST_DIR"), "/.venv/bin/python");
    if std::path::Path::new(venv).is_file() {
        return Some(venv.to_string());
    }
    std::process::Command::new("python3")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| "python3".to_string())
}

/// A constant whose formula goes through a complex value was emitted as
/// that formula: `abs(atanh(9))` (1.5747…, which `compile()` folds through
/// the evaluator) became `abs(math.atanh(9))` — a `ValueError` in Python,
/// NaN in C and Rust.  Now every back end folds it to the literal, and a
/// constant that is not real (`atanh(9)`, `asin(2)`) is refused.
/// Reference: mpmath 1.3.0 `abs(atanh(9))` = 1.5747537462713396…;
/// `compile()` gives 1.5747537462713397.
#[test]
fn emitted_constants_fold_like_compile_and_complex_ones_are_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.parse("x + abs(atanh(9))").unwrap();
    let vm = e.compile(&["x"]).unwrap();
    assert_eq!(vm(&[0.0]), 1.5747537462713397);
    assert_eq!(e.to_python().unwrap(), "1.5747537462713397 + x");
    assert_eq!(e.to_julia().unwrap(), "1.5747537462713397 + x");
    let c = e.to_c_fn("f", &["x"]).unwrap();
    assert!(c.contains("1.5747537462713397 + x"), "{c}");
    let r = e.to_rust_fn("f", &["x"]).unwrap();
    assert!(r.contains("1.5747537462713397_f64 + x"), "{r}");
    // A constant no back end has a function for is folded too.
    assert_eq!(
        ctx.parse("x + zeta(3)").unwrap().to_python().unwrap(),
        "1.2020569031595942 + x"
    );
    // Formulas that evaluate fine are kept.
    assert_eq!(
        (&(&x * &ctx.int(2).sqrt()) + &(&ctx.pi() / 2))
            .to_python()
            .unwrap(),
        "x*math.sqrt(2) + math.pi/2"
    );
    for bad in ["atanh(9)", "x + asin(2)", "x*log(-1)"] {
        let b = ctx.parse(bad).unwrap();
        for (name, res) in [
            ("python", b.to_python()),
            ("numpy", b.to_numpy()),
            ("julia", b.to_julia()),
            ("rust", b.to_rust_fn("f", &["x"])),
            ("c", b.to_c_fn("f", &["x"])),
            ("python_fn", b.to_python_fn("f", &["x"])),
        ] {
            let err = res.expect_err(&format!("{name}: {bad} is not real"));
            assert!(err.to_string().contains("is not real"), "{name}: {err}");
        }
    }
}

/// `min`/`max` with a NaN operand: `compile()` gives NaN; C `fmin`, Rust
/// `f64::min` and Python's `min` returned the other operand (Python only
/// when the NaN came second).  An even-denominator power of a negative
/// number has no real value: C `pow`, Rust `powf` and `compile()` give NaN,
/// Python's `(-2.0)**(5/4)` was a complex number.  Executed in Python.
#[test]
fn python_min_max_and_even_roots_agree_with_compile() {
    let Some(py) = python() else {
        eprintln!("no Python found; skipping");
        return;
    };
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let cases: Vec<Ex> = vec![
        x.min_with(&y),
        x.max_with(&y),
        y.min_with(&x),
        y.max_with(&x),
        x.pow(&ctx.rational(5, 4)),
        x.pow(&ctx.rational(-3, 2)),
        &x.pow(&ctx.rational(3, 2)) + &y,
    ];
    let points = [
        (-2.0_f64, f64::NAN),
        (f64::NAN, 3.0),
        (2.0, 3.0),
        (-2.0, 1.0),
    ];
    let mut program = String::from("import math\nnan = float('nan')\n");
    let mut expected: Vec<f64> = Vec::new();
    for e in &cases {
        let code = e.to_python().unwrap();
        let f = e.compile(&["x", "y"]).unwrap();
        for &(px, py_) in &points {
            program.push_str(&format!(
                "x = {}\ny = {}\nprint(repr(float({code})))\n",
                if px.is_nan() {
                    "nan".into()
                } else {
                    format!("{px:?}")
                },
                if py_.is_nan() {
                    "nan".into()
                } else {
                    format!("{py_:?}")
                },
            ));
            expected.push(f(&[px, py_]));
        }
    }
    let out = std::process::Command::new(&py)
        .arg("-c")
        .arg(&program)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let got: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().parse::<f64>().unwrap())
        .collect();
    assert_eq!(got.len(), expected.len());
    for (i, (g, w)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-15 * w.abs().max(1.0),
            "case {i}: python {g} vs compile {w}"
        );
    }
    // Spot checks of the semantics themselves.
    let f = x.min_with(&y).compile(&["x", "y"]).unwrap();
    assert!(f(&[-2.0, f64::NAN]).is_nan());
    let f = x.pow(&ctx.rational(5, 4)).compile(&["x", "y"]).unwrap();
    assert!(f(&[-2.0, 0.0]).is_nan());
}

/// The C back end executed: `min`/`max` with a NaN operand are NaN (they
/// were `fmin`/`fmax`, which return the other operand), the folded
/// constant is `abs(atanh(9))`, and `x^(5/4)` of a negative `x` is NaN —
/// each equal to `compile()`.
#[test]
fn c_min_max_nan_and_folded_constants_agree_with_compile() {
    let Some(cc) = ["cc", "clang", "gcc"].into_iter().find(|cc| {
        std::process::Command::new(cc)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    }) else {
        eprintln!("no C compiler; skipping");
        return;
    };
    let ctx = Context::new();
    let e = ctx
        .parse("min(x, y) + 10*max(y, x) + 100*x^(5/4) + abs(atanh(9))")
        .unwrap();
    let code = e.to_c_fn("f", &["x", "y"]).unwrap();
    let f = e.compile(&["x", "y"]).unwrap();
    let points = [
        (-2.0_f64, f64::NAN),
        (f64::NAN, 3.0),
        (2.0, 3.0),
        (3.0, -1.0),
    ];
    let mut main = String::from("#include <stdio.h>\nint main(void) {\n");
    for &(px, py) in &points {
        let lit = |v: f64| {
            if v.is_nan() {
                "NAN".to_string()
            } else {
                format!("{v:?}")
            }
        };
        main.push_str(&format!(
            "    printf(\"%.17g\\n\", f({}, {}));\n",
            lit(px),
            lit(py)
        ));
    }
    main.push_str("    return 0;\n}\n");
    let dir = std::env::temp_dir().join(format!("symplex_v29_leftovers_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("f.c");
    let exe = dir.join("f");
    std::fs::write(&src, format!("{code}\n{main}")).unwrap();
    let status = std::process::Command::new(cc)
        .args(["-std=c99", "-O1", "-o"])
        .arg(&exe)
        .arg(&src)
        .arg("-lm")
        .status()
        .unwrap();
    assert!(status.success(), "C compile failed:\n{code}");
    let out = std::process::Command::new(&exe).output().unwrap();
    let got: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| match l.trim() {
            "nan" | "-nan" => f64::NAN,
            s => s.parse().unwrap(),
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    for (&(px, py), g) in points.iter().zip(&got) {
        let w = f(&[px, py]);
        assert!(
            (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-13 * w.abs().max(1.0),
            "f({px}, {py}): C {g} vs compile {w}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Units: runtime dimension inference
// ═══════════════════════════════════════════════════════════════════════════

/// `infer_dimension` rejected every non-integer power (`ω = √(k/m)` was an
/// error), required dimensionless arguments of `abs`, `min`, `max`, `re`,
/// `sign`, `Heaviside`, `DiracDelta` and `atan2`, called a definite
/// integral dimensionless, and overflowed its `i8` exponents (`x¹²⁷·y`
/// panicked in debug builds and wrapped to a wrong dimension in release).
/// References: SymPy 1.14 `dimsys_SI.get_dimensional_dependencies`:
/// `sqrt(force/length/mass)` -> {time: -1}; `sqrt(area)` -> {length: 1};
/// `area**(3/2)` -> {length: 3}; `1/time` -> {time: -1};
/// `velocity*time` -> {length: 1}.
#[test]
fn dimension_inference_of_roots_piecewise_functions_and_integrals() {
    use symplex::units::{ConstDim, DimMap, infer_dimension};
    let ctx = Context::new();
    let dims = DimMap::new()
        .with("k", ConstDim::FORCE.div(ConstDim::LENGTH))
        .with("m", ConstDim::MASS)
        .with("x", ConstDim::LENGTH)
        .with("y", ConstDim::LENGTH)
        .with("v", ConstDim::VELOCITY)
        .with("t", ConstDim::TIME)
        .with("t0", ConstDim::TIME)
        .with("T", ConstDim::TIME)
        .with("A", ConstDim::LENGTH.mul(ConstDim::LENGTH));
    let inv_time = ConstDim::DIMENSIONLESS.div(ConstDim::TIME);
    for (src, want) in [
        ("sqrt(k/m)", inv_time),
        ("sqrt(A)", ConstDim::LENGTH),
        (
            "A^(3/2)",
            ConstDim::LENGTH.mul(ConstDim::LENGTH).mul(ConstDim::LENGTH),
        ),
        ("abs(x) + y", ConstDim::LENGTH),
        ("min(x, y) + max(y, x)", ConstDim::LENGTH),
        ("re(x)", ConstDim::LENGTH),
        ("x*Heaviside(t - t0)", ConstDim::LENGTH),
        ("DiracDelta(t - t0)", inv_time),
        ("atan2(y, x)", ConstDim::DIMENSIONLESS),
        ("sign(x)", ConstDim::DIMENSIONLESS),
        ("v*T", ConstDim::LENGTH),
        // 0 fits every dimension.
        ("max(x, 0)", ConstDim::LENGTH),
    ] {
        let e = ctx.parse(src).unwrap();
        assert_eq!(infer_dimension(&e, &dims), Ok(want), "{src} [{e}]");
    }
    // The unevaluated definite integral node itself (not just its value).
    let (v, t, big_t) = (ctx.symbol("v"), ctx.symbol("t"), ctx.symbol("T"));
    let f = ctx.apply("u", &[&t]).unwrap();
    let dims2 = dims.clone().with("u(t)", ConstDim::VELOCITY);
    let node = f.integrate_definite(&t, &ctx.int(0), &big_t);
    assert_eq!(
        infer_dimension(&node, &dims2),
        Ok(ConstDim::LENGTH),
        "{node}"
    );
    assert_eq!(
        infer_dimension(&f.diff(&t).diff(&t), &dims2)
            .map(|d| d.eq(ConstDim::ACCELERATION.div(ConstDim::TIME))),
        Ok(true)
    );
    let _ = v;
    // Still errors: √length, x + t, sin(x), exponent overflow (no panic).
    for bad in [
        "sqrt(x)",
        "x + t",
        "sin(x)",
        "x^127*y",
        "x^100*y^100",
        "x^200",
    ] {
        let e = ctx.parse(bad).unwrap();
        assert!(infer_dimension(&e, &dims).is_err(), "{bad}");
    }
}
