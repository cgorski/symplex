//! Tests of the 0.49 refactor (w6): function facets (conjugation, realness, parity, poles) from the Func table.
//! Every test says what was wrong before and cites its oracle.
//!
//! The facets of every function (`Func::ALL`) are checked against
//! `conjugate`, the assumption system, `canon`/`eval`'s sign extraction and
//! the pole rule next to the table, in `src/base/func.rs`; mpmath checks of
//! every reflection and realness rule are in the step's notes.

#![allow(unused_imports)]
use symplex::base::assumptions::Props;
use symplex::prelude::*;

fn ctx_with_domains() -> Context {
    let ctx = Context::new();
    ctx.symbol("z");
    ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    ctx.symbol_with("y", &[Assumption::Real]).unwrap();
    ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    ctx.symbol_with("n", &[Assumption::Integer]).unwrap();
    ctx.symbol_with("m", &[Assumption::Integer, Assumption::NonPositive])
        .unwrap();
    ctx.symbol_with("w", &[Assumption::Imaginary]).unwrap();
    ctx
}

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap_or_else(|e| panic!("{s}: {e}"))
}

/// `conjugate` knew 14 functions, and no library function: before,
/// `conjugate(erfi(z))` stayed `conjugate(erfi(z))`, and so did `Shi`, the
/// Fresnel and Airy functions and `dirichlet_eta` — all entire and real on
/// the real axis, so `conj f(z) = f(conj z)` (Schwarz reflection; SymPy:
/// `conjugate(erfi(z))` → `erfi(conjugate(z))`).
///
/// Oracle: mpmath, `conj(f(z)) == f(conj(z))` to 30 digits at `z = 1.3 +
/// 0.7i` and `−2 − 1.5i` for each (e.g. `erfi(1.3 − 0.7j)` =
/// `conj(erfi(1.3 + 0.7j))` = `0.82590616547 − 1.84282906706j`).
#[test]
fn conjugate_commutes_with_the_entire_library_functions() {
    let ctx = ctx_with_domains();
    for f in [
        "erfi",
        "Shi",
        "fresnels",
        "fresnelc",
        "airyai",
        "airybi",
        "airyaiprime",
        "airybiprime",
        "dirichlet_eta",
    ] {
        let e = parse(&ctx, &format!("conjugate({f}(z))"));
        assert_eq!(e, parse(&ctx, &format!("{f}(conjugate(z))")), "{f}");
    }
    // Every argument is conjugated (SymPy: `conjugate(beta(a, b))`).
    let e = parse(&ctx, "conjugate(beta(z, 2 + I))");
    assert_eq!(e, parse(&ctx, "beta(conjugate(z), 2 - I)"));
    // `sign z = z/|z|` (SymPy's `sign._eval_conjugate`).
    let e = parse(&ctx, "conjugate(sign(z))");
    assert_eq!(e, parse(&ctx, "sign(conjugate(z))"));
}

/// Before, the assumption system had no handler for a library function:
/// `erfi(x)`, `Shi(x)`, `fresnels(x)`, the Airy functions of a real `x`,
/// `Chi(p)` for a positive `p`, `besselj(n, x)` for an integer order,
/// `besselj(x, p)`, `beta(x, y)` were all `is_real() == None`, and
/// `conjugate(Shi(x))` stayed unevaluated.
///
/// Oracle: DLMF 6.2, 7.2, 9.2, 10.2 (real on the stated domains); SymPy
/// 1.14 `fresnels(x).is_real`, `beta(x, y).is_real` → `True`; mpmath at two
/// points of each domain, imaginary part 0 (`chi(0.3)` = −0.0564…,
/// `besselj(-3, 1.5)` = −0.0610…, `besselj(-1.3, 2.5)` = 0.0349…,
/// `beta(-1.5, 2.3)` = 2.5578…).
#[test]
fn library_functions_are_real_on_their_real_domains() {
    let ctx = ctx_with_domains();
    for s in [
        "erfi(x)",
        "Shi(x)",
        "fresnels(x)",
        "fresnelc(x)",
        "airyai(x)",
        "airybi(x)",
        "airyaiprime(x)",
        "airybiprime(x)",
        "Chi(p)",
        "besselj(n, x)",
        "besselj(x, p)",
        "besseli(n, x)",
        "bessely(x, p)",
        "besselk(x, p)",
        "expint(x, p)",
        "uppergamma(x, p)",
        "beta(x, y)",
        "lambertw(p)",
        "legendre(2, x)",
        "polylog(x, -p)",
        "elliptic_k(-p)",
    ] {
        assert_eq!(parse(&ctx, s).is_real(), Some(true), "{s}");
    }
    let shi = parse(&ctx, "Shi(x)");
    assert_eq!(shi.conjugate(), shi);
}

/// The domains end where the functions stop being real: no claim for
/// `Chi(x)` (`Chi(−2) = 2.4527 + πi`), `bessely(x, x)`, `besselj(x, x)`
/// (`besselj(1/2, −2) = 0.5130i`), `lambertw(x)` (`W(−1/2)` is not real),
/// `elliptic_k(p)` (`K(2)` is not real), and none at a known pole:
/// `beta(m, x)` for an integer `m ≤ 0` (`Γ(m)` is infinite; `beta(−1, 2)`
/// = `zoo`).
///
/// Oracle: mpmath `chi(-2)` = `2.45266692265 + 3.14159265359j`,
/// `besselj(0.5, -2)` = `0.513016136562j`, `lambertw(-0.5)` =
/// `-0.794023632345 + 0.770111750510j`, `ellipk(2)` = `1.31102877714 −
/// 1.31102877714j`.
#[test]
fn no_realness_claim_off_the_domain_or_at_a_pole() {
    let ctx = ctx_with_domains();
    for s in [
        "Chi(x)",
        "bessely(x, x)",
        "besselj(x, x)",
        "lambertw(x)",
        "elliptic_k(p)",
        "beta(m, x)",
        "erfi(z)",
    ] {
        assert_ne!(parse(&ctx, s).is_real(), Some(true), "{s}");
    }
    let pole = parse(&ctx, "beta(m, x)");
    assert_ne!(ctx.query(&pole, Props::FINITE), Some(true));
}

/// Before, `conjugate(polygamma(x, z))` for a real `x` was
/// `polygamma(x, conjugate(z))` everywhere, but for an order that is not a
/// non-negative integer `ψ⁽ˣ⁾` is cut along `z ≤ 0`: at `x = 1/2`, `z =
/// −3/2` the value is not real, so its conjugate is not
/// `polygamma(1/2, −3/2)`.  The reflection now needs an integer order `≥ 0`
/// or a `z` known off the cut.
///
/// Oracle: SymPy 1.14 `N(polygamma(1/2, -3/2))` = `-3.23252738151252 -
/// 1.07469363679048*I` (= the limit from above; from below `+1.0747i`);
/// `N(conjugate(polygamma(1/2 + I/3, 2 + I)))` =
/// `N(polygamma(1/2 - I/3, 2 - I))` = `1.4199112365756 + 0.1794281434286*I`.
#[test]
fn polygamma_reflects_off_its_cut_or_at_an_integer_order() {
    let ctx = ctx_with_domains();
    let e = parse(&ctx, "conjugate(polygamma(x, z))");
    assert_eq!(e.to_string(), "conjugate(polygamma(x, z))");
    let e = parse(&ctx, "conjugate(polygamma(1, z))");
    assert_eq!(e, parse(&ctx, "polygamma(1, conjugate(z))"));
    let e = parse(&ctx, "conjugate(polygamma(x, 2 + I))");
    assert_eq!(e, parse(&ctx, "polygamma(x, 2 - I)"));
}

/// Before, `conjugate` left every function with a branch cut alone:
/// `conjugate(log(2 + I))`, `conjugate(Ci(w))` for an imaginary `w`,
/// `conjugate(besselj(1/2, 2 + I))`, `conjugate(polylog(2, 3 + I))` stayed
/// unevaluated.  Off the cut the principal branch reflects.
///
/// Oracle: mpmath, `conj(f(z)) == f(conj(z))` at points off the cut
/// (`log(2 - 1j)` = `conj(log(2 + 1j))` = `0.804718956217 − 0.463647609001j`;
/// `polylog(2, 3 - 1j)` = `conj(polylog(2, 3 + 1j))`); on the cut it fails
/// (`log(-2)` = `0.693 + 3.142j` is not `conj(log(-2))`), and there the
/// conjugate stays.
#[test]
fn conjugate_reflects_branch_cut_functions_off_the_cut() {
    let ctx = ctx_with_domains();
    for (s, want) in [
        ("conjugate(log(2 + I))", "log(2 - I)"),
        ("conjugate(log(w))", "log(-w)"),
        ("conjugate(Ci(w))", "Ci(-w)"),
        ("conjugate(asin(2 + I))", "asin(2 - I)"),
        ("conjugate(besselj(1/2, 2 + I))", "besselj(1/2, 2 - I)"),
        ("conjugate(polylog(2, 3 + I))", "polylog(2, 3 - I)"),
        ("conjugate(besselj(n, z))", "besselj(n, conjugate(z))"),
    ] {
        assert_eq!(parse(&ctx, s), parse(&ctx, want), "{s}");
    }
    for s in [
        "conjugate(log(z))",
        "conjugate(Chi(x))",
        "conjugate(besselj(x, z))",
    ] {
        assert_eq!(parse(&ctx, s).to_string(), s.replace("log", "ln"), "{s}");
    }
}

/// `evalf` paired the terms of `polylog(2, 3 + i) + polylog(2, 3 − i)` as
/// complex conjugates only through its own list, which lacked `polylog`,
/// `sign` and `dirichlet_eta`: the imaginary part of the (real, positive)
/// sum stayed a rounding residue of undecidable sign, and the logarithm of
/// its negation, on the cut, was refused (before: `PrecisionExhausted`
/// for all three).  The list is now the table's.
///
/// Oracle: mpmath (dps 30) `log(-(polylog(2, 3+1j) + polylog(2, 3-1j)))` =
/// `0.990231565528106273146 + 3.14159265358979323846j`;
/// `log(-((1+1j)/abs(1+1j) + (1-1j)/abs(1-1j)))` = `0.346573590279972654709 +
/// 3.14159265358979323846j`; `log(-(altzeta(2+1j) + altzeta(2-1j)))` =
/// `0.527905919274232832042 + 3.14159265358979323846j`.
#[test]
fn evalf_pairs_conjugate_terms_by_the_table() {
    let ctx = Context::new();
    for (s, want) in [
        (
            "log(-(polylog(2, 3+I) + polylog(2, 3-I)))",
            "0.99023156552810627315 + 3.1415926535897932385*i",
        ),
        (
            "log(-(sign(1+I) + sign(1-I)))",
            "0.34657359027997265471 + 3.1415926535897932385*i",
        ),
        (
            "log(-(dirichlet_eta(2+I) + dirichlet_eta(2-I)))",
            "0.52790591927423283204 + 3.1415926535897932385*i",
        ),
    ] {
        assert_eq!(parse(&ctx, s).eval_decimal(20).unwrap(), want, "{s}");
    }
}

/// The internal functions of the limit engine are reserved.  Before,
/// `ctx.apply("__ei_asym", [x])` built the node the limit engine reads as
/// `e^{−x}·Ei(x)` (its limit at `∞` was 0, its series `1/x + 1/x² + …`),
/// and once a limit computation had left such a node in the context, the
/// parser read `__ei_asym(x)` as it.
///
/// Oracle: the API contract (`Context::apply`: nothing is known about an
/// applied function).
#[test]
fn internal_function_names_are_reserved() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(ctx.apply("__ei_asym", &[&x]).is_err());
    assert!(ctx.apply("__erfcx_asym", &[&x]).is_err());
    // A user function with any other name is still accepted.
    assert!(ctx.apply("__my_f", &[&x]).is_ok());
    let e = parse(&ctx, "exp(-x)*Ei(x)");
    let lim = e.limit(&x, &ctx.infinity());
    assert_eq!(lim.to_string(), "0");
    assert!(ctx.parse("__ei_asym(x)").is_err());
}
