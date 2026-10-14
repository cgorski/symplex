//! Tests of the 0.50 refactor (w6b): code generation from the Func table.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

/// The eight functions `compile()` evaluates since 0.48 (Si, Ci, Ei, li,
/// Shi, Chi, erfi, Fresnel S/C) are emitted by `to_rust_fn` and `to_c_fn`
/// through the shared runtime helpers.
///
/// Before (0.49): `NotImplemented("cannot generate Rust code for
/// conjugate/arg, Si/Ci/Ei/li, zeta, polygamma, or KroneckerDelta yet")`,
/// `"… for the special function `erfi`"`, and `"cannot generate C code for
/// `Si`"`.  Oracle for the emitted code: compiled with `cc -O2` (GCC 13,
/// x86-64) and as a release Rust crate on a dense grid (114,254 points per
/// function: ±10⁻²⁰…10²⁰ log-spaced, [−60, 60] step 0.004, every branch
/// switch and zero ±0.02, ±∞, NaN, subnormals), it agrees with `compile()`
/// bit for bit; `compile()` itself is within 9.2 ulps of mpmath at 30
/// digits (0.48).
#[test]
fn the_integrals_erfi_and_fresnel_are_emitted_in_rust_and_c() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases = [
        (x.si(), "sin_integral"),
        (x.ci(), "cos_integral"),
        (x.ei(), "exp_integral_ei"),
        (x.li(), "log_integral"),
        (x.shi(), "sinh_integral"),
        (x.chi(), "cosh_integral"),
        (x.erfi(), "erfi"),
        (x.fresnels(), "fresnel_s"),
        (x.fresnelc(), "fresnel_c"),
    ];
    for (e, helper) in cases {
        let rust = e
            .to_rust_fn("f", &["x"])
            .unwrap_or_else(|err| panic!("{e}: {err}"));
        assert!(
            rust.contains(&format!("symplex_rt::{helper}(x)")),
            "{e}: {rust}"
        );
        assert!(
            rust.contains(&format!("pub fn {helper}(")),
            "{e}: runtime section missing"
        );
        let c = e
            .to_c_fn("f", &["x"])
            .unwrap_or_else(|err| panic!("{e}: {err}"));
        assert!(
            c.contains(&format!("return symplex_{helper}(x);")),
            "{e}: {c}"
        );
        assert!(
            c.contains(&format!("static inline double symplex_{helper}(double x)")),
            "{e}: {c}"
        );
        assert_eq!(c.matches('{').count(), c.matches('}').count(), "{e}");
    }
    // Inside a larger expression, with the dependencies emitted once.
    let e = &x.si() * &x.ci() + x.erfi();
    let c = e.to_c_fn("g", &["x"]).unwrap_or_else(|err| panic!("{err}"));
    assert_eq!(
        c.matches("static inline void symplex_sici_fg(").count(),
        1,
        "{c}"
    );
    assert_eq!(
        c.matches("static inline void symplex_gamma_cf_imag(")
            .count(),
        1,
        "{c}"
    );
}

/// A function a back end refuses is refused with the reason from the
/// function table: the back ends that do evaluate it, and for the Python
/// printers the SciPy routine.
///
/// Before (0.49): `Si(x).to_python()` said only "cannot generate code for
/// `Si`" (SciPy: `scipy.special.sici(x)[0]` is Si, SciPy 1.13 docs), and
/// `zeta(x).to_rust_fn` lumped zeta with "conjugate/arg, Si/Ci/Ei/li".
/// Oracle: which back end evaluates what is the `rt` facet, checked against
/// every emitter by the unit test `the_rt_facet_is_what_every_back_end_does`.
#[test]
fn refusals_name_the_back_ends_and_the_scipy_routine() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let err = x.si().to_python().expect_err("math has no Si");
    let msg = err.to_string();
    assert!(msg.contains("scipy.special.sici(x)[0]"), "{msg}");
    assert!(msg.contains("compile") && msg.contains("to_c_fn"), "{msg}");
    let err = x.zeta().to_rust_fn("f", &["x"]).expect_err("no f64 zeta");
    let msg = err.to_string();
    assert!(matches!(err, SymplexError::NotImplemented(_)), "{msg}");
    assert!(msg.contains("`zeta`") && !msg.contains("Si/Ci"), "{msg}");
    // The library functions keep their SciPy names (pinned since 0.9).
    let err = x.bessel_j(&ctx.int(0)).to_python().expect_err("refused");
    assert!(err.to_string().contains("scipy.special.jv"), "{err}");
}

/// LaTeX of the library functions comes from the `tex` facet, in SymPy's
/// notation.
///
/// Before (0.49): the name was printed and an underscore in it became a
/// subscript: `rising_factorial\left(x, y\right)`,
/// `chebyshev_t\left(n, x\right)`, `euler_number\left(n\right)`.  Oracle:
/// SymPy 1.14 `latex()` of `RisingFactorial(x, y)` = `{x}^{\left(y\right)}`,
/// `RisingFactorial(x + 1, y)` = `{\left(x + 1\right)}^{\left(y\right)}`,
/// `FallingFactorial(x, y)` = `{\left(x\right)}_{y}`, `chebyshevt(n, x)` =
/// `T_{n}\left(x\right)`, `chebyshevu` `U_{n}…`, `euler(n)` = `E_{n}`,
/// `fibonacci(n)` = `F_{n}`, `besselj(nu, x)` = `J_{\nu}\left(x\right)`,
/// `factorial2(n)` = `n!!`, `subfactorial(n + 1)` = `!\left(n + 1\right)`.
/// SymPy has no notation for `harmonic` and `partition`: the standard
/// `H_{n}` (DLMF 25.11.33) and `p\left(n\right)` (DLMF 27.14.1).
#[test]
fn latex_of_library_functions_uses_their_notation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let n = ctx.symbol("n");
    let nu = ctx.symbol("nu");
    let cases = [
        (x.rising_factorial(&y), r"{x}^{\left(y\right)}"),
        (
            (&x + 1).rising_factorial(&y),
            r"{\left(x + 1\right)}^{\left(y\right)}",
        ),
        (x.falling_factorial(&y), r"{\left(x\right)}_{y}"),
        (x.chebyshev_t(&n), r"T_{n}\left(x\right)"),
        (n.euler_number(), r"E_{n}"),
        (n.fibonacci(), r"F_{n}"),
        (n.partition_count(), r"p\left(n\right)"),
        (x.bessel_j(&nu), r"J_{\nu}\left(x\right)"),
    ];
    for (e, want) in cases {
        assert_eq!(e.to_latex(), want, "{e}");
    }
    let parsed = [
        ("chebyshev_u(n, x)", r"U_{n}\left(x\right)"),
        ("factorial2(n)", r"n!!"),
        ("subfactorial(n + 1)", r"!\left(n + 1\right)"),
        ("harmonic(n)", r"H_{n}"),
        ("stirling2(n, 3)", r"\left\{{n \atop 3}\right\}"),
    ];
    for (src, want) in parsed {
        let e = ctx.parse(src).unwrap_or_else(|err| panic!("{src}: {err}"));
        assert_eq!(e.to_latex(), want, "{src}");
        assert!(!e.to_latex().contains('_') || want.contains('_'), "{src}");
    }
}

/// `expand()` expands the elements of a set and the endpoints of an
/// interval.
///
/// Before (0.49): `{3, (x+1)^2}.expand()` returned `{3, (x + 1)^2}` (the
/// rebuild in `expand.rs` ended in `_ => id`).  Oracle: the exact identity
/// (x + 1)² = x² + 2x + 1, and the canonical set built from the expanded
/// elements (`==`).
#[test]
fn expand_looks_inside_sets() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let square = (&x + 1).powi(2);
    let expanded = x.powi(2) + &x * 2 + 1;
    assert_eq!(square.expand(), expanded);
    let three = ctx.int(3);
    let got = ctx.finite_set(&[three.clone(), square.clone()]).expand();
    assert_eq!(got, ctx.finite_set(&[three, expanded.clone()]), "{got}");
    // Two elements equal once expanded merge.
    let got = ctx.finite_set(&[square.clone(), expanded.clone()]).expand();
    assert_eq!(
        got,
        ctx.finite_set(std::slice::from_ref(&expanded)),
        "{got}"
    );
    let closed = symplex::IntervalKind::Closed;
    let got = ctx.interval(&ctx.int(0), &square, closed).expand();
    assert_eq!(got, ctx.interval(&ctx.int(0), &expanded, closed), "{got}");
    let five_six = ctx.interval(&ctx.int(5), &ctx.int(6), closed);
    let got = ctx.finite_set(&[square]).union(&five_six).expand();
    let want = ctx.finite_set(&[expanded]).union(&five_six);
    assert_eq!(got, want, "{got}");
}
