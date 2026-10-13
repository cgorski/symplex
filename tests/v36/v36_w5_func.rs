//! Tests of the 0.49 refactor (w5): one identity for every mathematical function.
//! Every test says what was wrong before and cites its oracle.
//!
//! The crate-internal `Func` registry itself (sort keys, display, parse,
//! LaTeX and arity of every function in `Func::ALL`) is tested next to it,
//! in `src/base/func.rs`: `Func` is not public.

#![allow(unused_imports)]
use symplex::prelude::*;

/// The `b` of S1's reproducer, chosen so that `im Shi(1 + b·i)` and
/// `im erfi(1 + b·i)` agree to 45 digits.
const B: &str =
    "286348331918707048933568752267077911803906783/250000000000000000000000000000000000000000000";

/// evalf's conjugate pairing (`evalf/conjugate.rs`) signed every `Apply`
/// node by its variant alone, so `Shi(1 − b·i)` and `erfi(1 + b·i)` — two
/// *different* conjugate-symmetric library functions at conjugate arguments
/// — "paired" as complex conjugates, and the imaginary part of their sum,
/// +9.31·10⁻⁴⁶, was taken to be exactly 0: the logarithm of the negated
/// sum landed on the wrong side of its cut.  Before:
/// `-0.10189174999241109844 + 3.1415926535897932385*i`.
///
/// Oracle: mpmath (dps 120) `log(-(shi(mpc(1, -b)) + erfi(mpc(1, b))))` =
/// `-0.101891749992411098435258580056 - 3.14159265358979323846264338328j`,
/// and `im(shi(mpc(1, -b)) + erfi(mpc(1, b)))` = `9.3106802716651155753e-46`.
#[test]
fn conjugate_pairing_tells_library_functions_apart() {
    let ctx = Context::new();
    let s = format!("(Shi(1 - {B}*I) + erfi(1 + {B}*I))");
    let log = ctx.parse(&format!("log(-{s})")).unwrap();
    assert_eq!(
        log.eval_decimal(20).unwrap(),
        "-0.10189174999241109844 - 3.1415926535897932385*i"
    );
    let im = ctx.parse(&format!("im({s})")).unwrap();
    assert_eq!(im.eval_decimal(10).unwrap(), "9.310680272e-46");
}

/// The same library function at conjugate arguments still pairs: the sum is
/// exactly real (no imaginary residue), unchanged from 0.48.
///
/// Oracle: mpmath (dps 120) `shi(1-2j) + shi(1+2j)` =
/// `0.91538342257337601134262388028332022 + 0.0j`, `erfi(1-2j) + erfi(1+2j)`
/// = `-0.02251801205763005015280183126329645 + 0.0j`.
#[test]
fn the_same_function_at_conjugate_arguments_still_pairs() {
    let ctx = Context::new();
    for (src, want) in [
        (
            "Shi(1 - 2*I) + Shi(1 + 2*I)",
            "0.915383422573376011342623880283",
        ),
        (
            "erfi(1 - 2*I) + erfi(1 + 2*I)",
            "-0.0225180120576300501528018312633",
        ),
    ] {
        assert_eq!(
            ctx.parse(src).unwrap().eval_decimal(30).unwrap(),
            want,
            "{src}"
        );
    }
}

/// `integrate.rs`'s `single_argument` took the *last* argument of every
/// library function as "the" argument — wrong for `lambertw(x, k)` and
/// `elliptic_f(φ, m)`, whose argument is the first.  Its callers (the
/// by-parts rules) only reach it for one-argument library functions, so no
/// output was wrong (none observable: S1 found none, nor did this step);
/// it now answers only for one-argument functions.  This pins the
/// antiderivatives those rules give, identical to 0.48.
///
/// Oracle: the exact identity `d/dx F = f`, checked at `x = 3/10`
/// (`(d/dx F − f)(3/10)` evaluates to 0 within 1e-12).
#[test]
fn one_argument_library_functions_keep_their_by_parts_antiderivatives() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let at = ctx.rational(3, 10);
    for (src, want) in [
        (
            "Shi(2*x + 1)",
            "-1/2*cosh(2*x + 1) + (x + 1/2)*Shi(2*x + 1)",
        ),
        (
            "Chi(x/2 + 1)",
            "-2*sinh(1/2*x + 1) + (x + 2)*Chi(1/2*x + 1)",
        ),
        ("fresnels(3*x)", "x*fresnels(3*x) + cos(9/2*x^2*pi)/(3*pi)"),
        ("x*Shi(x)", "1/2*x^2*Shi(x) - 1/2*x*cosh(x) + 1/2*sinh(x)"),
    ] {
        let f = ctx.parse(src).unwrap();
        let big_f = f.integrate(&x);
        assert_eq!(big_f.to_string(), want, "{src}");
        let residual = (&big_f.diff(&x) - &f).subs(&x, &at).eval_f64().unwrap();
        assert!(residual.abs() < 1e-12, "{src}: d/dx F − f = {residual}");
    }
}
