//! Tests of session 14 (s14e): non-real roots of real solves, odd-root inversion.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn is_no_solution<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::NoSolution { .. }))
}

fn is_refused<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::ComputationFailed { .. }))
}

fn roots(e: &Ex, v: &Ex) -> Vec<String> {
    e.solve(v)
        .unwrap_or_else(|err| panic!("{e}: {err}"))
        .iter()
        .map(|r| r.to_string())
        .collect()
}

/// S3.  `4^(x + 3/4) + 2·2^(x + 3/4) + 1 = 0` for a real `x` returned two
/// roots, `ln(⅛·4^(1/4)·(±√(−4·4^(3/4) + 8√2) − 2·2^(3/4)))/ln 2`,
/// logarithms of negative numbers: in `t = 2ˣ` the discriminant `8√2 −
/// 4·4^(3/4)` is 0 (`4^(3/4) = 2√2`), which canonical forms do not show, and
/// the double root split into two forms whose sign nothing decided (the
/// unshifted `4ˣ + 2·2ˣ + 1` was right).  Same for the rates `3x` and the
/// shift `1/3`.  Oracle (exact): the left side is `(2^(x + 3/4) + 1)² > 0`
/// for a real `x`; over ℂ its zeros are `2^(x + 3/4) = −1`, the principal
/// one `x = −3/4 + iπ/ln 2` (mpmath at 30 digits: `-0.75 +
/// 4.53236014182719380962768294572j`), once.
#[test]
fn shifted_exponential_square_has_no_real_root() {
    let ctx = Context::new();
    let xr = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    for src in [
        "4^(x + 3/4) + 2*2^(x + 3/4) + 1",
        "4^(3*x + 3/4) + 2*2^(3*x + 3/4) + 1",
        "4^(x + 1/3) + 2*2^(x + 1/3) + 1",
    ] {
        let r = parse(&ctx, src).solve(&xr);
        assert!(is_no_solution(&r), "{src}: {r:?}");
    }
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let rs = parse(&ctx, "4^(x + 3/4) + 2*2^(x + 3/4) + 1")
        .solve(&x)
        .unwrap();
    assert_eq!(rs.len(), 1, "{rs:?}");
    let z = rs[0].eval_complex64().unwrap();
    assert!(
        (z.re + 0.75).abs() < 1e-12 && (z.im - 4.532_360_141_827_194).abs() < 1e-12,
        "{z}"
    );
}

/// S3, the double root itself.  `4^(x + 3/4) − 2·2^(x + 3/4) + 1 = 0` for a
/// real `x` gave the root `−3/4` twice, as `ln(⅛·4^(1/4)·(±√(−4·4^(3/4) +
/// 8√2) + 2·2^(3/4)))/ln 2`.  Oracle (exact): the left side is `(2^(x + 3/4)
/// − 1)²`, zero only at `x = −3/4`, which is not positive.
#[test]
fn hidden_zero_discriminant_gives_one_double_root() {
    let ctx = Context::new();
    let xr = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let e = parse(&ctx, "4^(x + 3/4) - 2*2^(x + 3/4) + 1");
    let rs = e.solve(&xr).unwrap();
    assert_eq!(rs.len(), 1, "{rs:?}");
    let z = rs[0].eval_complex64().unwrap();
    assert!((z.re + 0.75).abs() < 1e-12 && z.im.abs() < 1e-12, "{z}");
    let ctx = Context::new();
    let xp = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let r = parse(&ctx, "4^(x + 3/4) - 2*2^(x + 3/4) + 1").solve(&xp);
    assert!(is_no_solution(&r), "{r:?}");
}

/// S3, the final check: a root returned for a real unknown must be proved
/// real.  `exp(erf x) = 3` for a real `x` gave `[erfinv(ln 3)]` (the range
/// test of `erf` read only numbers), and `erf x = erfinv(ln 3)` gave
/// `[erfinv(erfinv(ln 3))]`, values no evaluation settles.  Oracles: `erf`
/// maps ℝ onto `(−1, 1)` and `ln 3 = 1.0986122886681098 > 1` (mpmath), so
/// `erfinv(ln 3)` is not real and neither equation has a real root.  For a
/// plain `x` the first is refused, as `erf x = 2` is (a value outside
/// `(−1, 1)` is reached only off the real line, where `erfinv` has no
/// principal value; it gave `[erfinv(ln 3)]`).  A real root that
/// only an exact zero test shows to be real stays: `x³ = 1 − √(8√2 −
/// 4·4^(3/4))` (the radicand is 0) has the root `∛(1 − √…)`, which is `1`.
#[test]
fn a_root_of_a_real_unknown_is_proved_real() {
    let ctx = Context::new();
    let xr = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let r = parse(&ctx, "exp(erf(x)) - 3").solve(&xr);
    assert!(is_no_solution(&r), "{r:?}");
    let r = parse(&ctx, "erf(x) - erfinv(ln(3))").solve(&xr);
    assert!(is_refused(&r), "{r:?}");
    let rs = parse(&ctx, "x^3 + sqrt(8*sqrt(2) - 4*4^(3/4)) - 1")
        .solve(&xr)
        .unwrap();
    assert_eq!(rs.len(), 1, "{rs:?}");
    assert!((rs[0].eval_f64().unwrap() - 1.0).abs() < 1e-12, "{rs:?}");
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = parse(&ctx, "exp(erf(x)) - 3").solve(&x);
    assert!(is_refused(&r), "{r:?}");
}

/// S4.  `√(x + ∛x) + (x − y)^(1/5) = 0` solved for `y` gave `y = x + (x +
/// ∛x)^(5/2)`, which is no root (SymPy 1.14 gives the same): `(x − y)^(1/5)
/// = −√w` (`w = x + ∛x`) was inverted to `x − y = (−√w)⁵` without asking
/// whether `−√w` is a principal fifth root.  Oracles: at `x = 2` the
/// residual is `3.26622567141073776 + 1.06126105298289743i` (mpmath, the
/// same at 30 and 60 digits); exactly,
/// `−√w` has its argument in `(π/2, π] ∪ (−π, −π/2]` and a principal fifth
/// root in `(−π/5, π/5]`, so the only solutions are at `w = 0` (`y = x`).
/// The solver now refuses (it cannot say "no root" for every `x`: `w` may be
/// 0).
#[test]
fn odd_root_is_inverted_only_on_its_principal_range() {
    let ctx = Context::new();
    let y = ctx.symbol("y");
    let e = parse(&ctx, "sqrt(x + x^(1/3)) + (x - y)^(1/5)");
    let r = e.solve(&y);
    assert!(is_refused(&r), "{r:?}");
}

/// S4, decided by declared signs.  For a positive `y`: `√y + (y − x)^(1/5)
/// = 0` gave `[y + y^(5/2)]`, `√x + √y = 0` gave `[y]`, `√x + y + 4 = 0`
/// gave `[(y + 4)²]`.  Oracles: `−√y`, `−(y + 4)` are negative, and neither
/// a principal fifth root nor a principal square root is ever negative, so
/// there is no root (at `y = 1`, mpmath: `√2 + (−1)^(1/5) ≠ 0`, `1 + 1 ≠
/// 0`, `5 + 5 ≠ 0`).  The ranges that hold keep their root: `∛x = y`,
/// `(x + y)^(1/3) = √y`, `x^(2/3) = y` (`y^(3/2)` has argument 0) and `√x =
/// y` (substitution: `∛(y³) = y`, `∛(y^(3/2)) = √y`, `(y^(3/2))^(2/3) = y`,
/// `√(y²) = y` for `y > 0`).
#[test]
fn principal_range_decided_by_declared_signs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _y = ctx.symbol_with("y", &[Assumption::Positive]).unwrap();
    for src in [
        "sqrt(y) + (y - x)^(1/5)",
        "sqrt(x) + sqrt(y)",
        "sqrt(x) + y + 4",
    ] {
        let r = parse(&ctx, src).solve(&x);
        assert!(is_no_solution(&r), "{src}: {r:?}");
    }
    for (src, want) in [
        ("x^(1/3) - y", "y^3"),
        ("(x + y)^(1/3) - sqrt(y)", "-y + y^(3/2)"),
        ("x^(2/3) - y", "y^(3/2)"),
        ("sqrt(x) - y", "y^2"),
    ] {
        assert_eq!(roots(&parse(&ctx, src), &x), [want], "{src}");
    }
}

/// S4, a plain (complex) parameter.  `√x = y` gave `[y²]` (SymPy 1.14:
/// `[y**2]`), `∛x = −2y` gave `[−8y³]`, `x^(2/3) = y` gave `[y^(3/2)]`, and
/// `x + y·√x = 0` gave `[y²/2 ∓ y·√(y²)/2]`: each holds only for some `y`.
/// Oracles (exact): `√((−1)²) = 1 ≠ −1` (`y = −1`); `∛(−8) = 1 + i√3 ≠ −2`
/// (`y = 1`); `((−1)^(3/2))^(2/3) = (−i)^(2/3) = e^(−iπ/3) ≠ −1` (`y = −1`);
/// at `y = 2` the root `4` gives `4 + 2·2 = 8`.  The public solvers now
/// refuse such roots (a root is a claim for every value of the parameters).
#[test]
fn conditional_roots_with_a_plain_parameter_are_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "sqrt(x) - y",
        "x^(1/3) + 2*y",
        "x^(2/3) - y",
        "x + y*sqrt(x)",
    ] {
        let r = parse(&ctx, src).solve(&x);
        assert!(is_refused(&r), "{src}: {r:?}");
    }
}

/// A candidate with parameters is checked when the parameters cancel in the
/// substituted equation.  `3(x + y)^(1/6) + √2 + 1 = 0` gave `[−y + 70√2/729
/// + 11/81]` (no root: a principal sixth root is never negative), and
/// `(x − y)^(3/2) + √2/3 = 0` gave the three `y + k`, `k³ = 2/9`, with the
/// real `k`.  Oracles (exact): `((1 + √2)⁶/729)^(1/6) = (1 + √2)/3 > 0`;
/// for the real `k = ∛6/3`, `k^(3/2) = √2/3`, while the two others have
/// `arg k = ±2π/3` and `k^(3/2) = (√2/3)·e^(±iπ) = −√2/3`.
#[test]
fn parametric_candidates_are_checked_when_the_parameters_cancel() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = parse(&ctx, "3*(x + y)^(1/6) + sqrt(2) + 1").solve(&x);
    assert!(is_no_solution(&r), "{r:?}");
    let e = parse(&ctx, "(x - y)^(3/2) + sqrt(2)/3");
    let rs = e.solve(&x).unwrap();
    assert_eq!(rs.len(), 2, "{rs:?}");
    let y = ctx.symbol("y");
    for r in &rs {
        let z = e
            .subs(&x, r)
            .subs(&y, &ctx.int(2))
            .eval_complex64()
            .unwrap();
        assert!(z.norm() < 1e-12, "{r}: {z}");
    }
}
