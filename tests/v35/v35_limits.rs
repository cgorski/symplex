//! Limits of complex-valued functions: finite imaginary parts of infinite
//! limits, directed infinities and the side of a branch cut approached at
//! infinity (the complex-limit hunt after 0.41).  Every reference value
//! cites its oracle: mpmath 1.3 at 60 digits along the path `x = ±10ᵏ`
//! (`k = 10, 20, 40, 80`) or `a ± 10⁻ᵏ`, SymPy 1.14 as a second opinion
//! (it drops finite imaginary parts of infinite limits: `limit(x - I, x,
//! oo)` → `oo`).  An infinite value is written as `oo + I·b` when the real
//! part diverges and the imaginary part tends to `b`, `I·oo + a` the other
//! way round, and `c·oo` for any other direction `c/|c|`, as `subs` writes
//! values at infinities since 0.41.

use symplex::prelude::*;

fn lim(ctx: &Context, src: &str, point: &str, dir: Direction) -> Result<Ex, SymplexError> {
    let f = ctx
        .parse(src)
        .unwrap_or_else(|e| panic!("`{src}` does not parse: {e}"));
    let x = ctx.symbol("x");
    let p = ctx.parse(point).unwrap();
    f.try_limit_dir(&x, &p, dir)
}

fn show(ctx: &Context, src: &str, point: &str, dir: Direction) -> String {
    match lim(ctx, src, point, dir) {
        Ok(v) => v.to_string(),
        Err(e) => panic!("limit of `{src}` at {point} {dir:?} failed: {e}"),
    }
}

fn same(ctx: &Context, src: &str, point: &str, dir: Direction, want: &str) {
    let got = lim(ctx, src, point, dir)
        .unwrap_or_else(|e| panic!("limit of `{src}` at {point} {dir:?} failed: {e}"));
    let want_ex = ctx.parse(want).unwrap().eval();
    assert_eq!(
        got, want_ex,
        "`{src}` at {point} {dir:?}: {got} (want {want})"
    );
}

const B: Direction = Direction::Both;

/// Before: `limit(x − I, x, oo)` was `oo` while `subs` gave `oo − I`: the
/// sum rule dropped the constant once the other part diverged.  An
/// infinite limit keeps the component of the constant perpendicular to its
/// direction (`±oo` still absorbs real constants).
///
/// Oracle: mpmath `10⁴⁰ − i` (imaginary part −1); `subs(x, oo)` → `-I +
/// oo`; `1/x + I` at `0⁺`: mpmath `10⁴⁰ + i`, `ln x + I` at `0⁺`: `−92.1 +
/// i`.  (SymPy: `oo`, `oo`, `-oo`.)
#[test]
fn finite_imaginary_part_of_an_infinite_sum() {
    let ctx = Context::new();
    same(&ctx, "x - I", "oo", B, "oo - I");
    same(&ctx, "-x - I", "oo", B, "-oo - I");
    same(&ctx, "x + 3 - 2*I", "-oo", B, "-oo - 2*I");
    same(&ctx, "log(x) + I", "oo", B, "oo + I");
    same(&ctx, "1/x + I", "0", Direction::Right, "oo + I");
    same(&ctx, "log(x) + I", "0", Direction::Right, "-oo + I");
    // Consistent with substitution:
    let x = ctx.symbol("x");
    let f = ctx.parse("x - I").unwrap();
    assert_eq!(
        f.subs(&x, &ctx.infinity()),
        lim(&ctx, "x - I", "oo", B).unwrap()
    );
    // Real constants are still absorbed.
    same(&ctx, "x + 5", "oo", B, "oo");
}

/// Before: `ln(−x)`, `ln(2ix)`, `ln x` at `−∞` and `asinh(ix)` tended to
/// `oo` (the imaginary part `arg` was dropped), `acosh x` at `−∞` was
/// refused ("oscillates or is undefined"), and `ln(−x) + x` (Gruntz) was
/// `oo`.
///
/// Oracle: mpmath at `x = 10⁴⁰`: `log(−x)` = 92.10 + 3.14159…j,
/// `log(2ix)` = 92.80 + 1.570796…j, `log(−10⁴⁰)` = 92.10 + πj,
/// `asinh(10⁴⁰ i)` = 92.80 + 1.570796…j, `acosh(−10⁴⁰)` = 92.80 + πj.
#[test]
fn logarithmic_growth_keeps_its_argument() {
    let ctx = Context::new();
    same(&ctx, "log(-x)", "oo", B, "oo + I*pi");
    same(&ctx, "log(2*I*x)", "oo", B, "oo + I*pi/2");
    same(&ctx, "log(x)", "-oo", B, "oo + I*pi");
    same(&ctx, "asinh(I*x)", "oo", B, "oo + I*pi/2");
    same(&ctx, "acosh(x)", "-oo", B, "oo + I*pi");
    same(&ctx, "log(-x) + x", "oo", B, "oo + I*pi");
    same(&ctx, "log((1 + I)*x)", "oo", B, "oo + I*pi/4");
    // `ln(x + i) → oo`: its argument tends to 0.
    same(&ctx, "log(x + I)", "oo", B, "oo");
}

/// Before: a logarithm (or a root) of an argument tending to `−∞` (or to
/// 0) along the negative axis took the principal value on the cut from
/// both sides: `atanh(x + i)` → `−iπ/2`, `ln(−x − i) − ln x` → `iπ`,
/// `acosh(−x − i) − ln x` → `ln 2 + iπ`, `√(−x − i)/√x` was refused.
///
/// Oracle: mpmath `atanh(mpc(10⁴⁰, 1))` = 1.0e-40 + 1.5707963…j,
/// `log(−10⁴⁰ − i) − log(10⁴⁰)` = −3.14159…j, `acosh(−10⁴⁰ − i) −
/// log(10⁴⁰)` = 0.693147… − 3.14159…j, `log(−10⁻⁴⁰ − 10⁻⁸⁰ i) +
/// log(10⁴⁰)` = −πj, `sqrt(−10⁴⁰ − i)/10²⁰` = −1.0j.  (SymPy agrees on
/// `atanh`, `I*pi/2`.)
#[test]
fn side_of_the_cut_at_infinity() {
    let ctx = Context::new();
    same(&ctx, "atanh(x + I)", "oo", B, "I*pi/2");
    same(&ctx, "atanh(x - I)", "oo", B, "-I*pi/2");
    same(&ctx, "log(-x - I) - log(x)", "oo", B, "-I*pi");
    same(&ctx, "log(-x + I) - log(x)", "oo", B, "I*pi");
    same(&ctx, "acosh(-x - I) - log(x)", "oo", B, "log(2) - I*pi");
    same(&ctx, "acosh(-x + I) - log(x)", "oo", B, "log(2) + I*pi");
    same(&ctx, "log(-1/x - I/x^2) + log(x)", "oo", B, "-I*pi");
    same(&ctx, "sqrt(-x - I)/sqrt(x)", "oo", B, "-I");
    same(&ctx, "sqrt(-x + I)/sqrt(x)", "oo", B, "I");
    // Unchanged, and right: `asech(2ix) = acosh(−i/(2x)) → −iπ/2`
    // (mpmath `asech(2·10⁴⁰ i)` = 5.0e-41 − 1.5707963…j; SymPy `-I*pi/2`).
    same(&ctx, "asech(2*I*x)", "oo", B, "-I*pi/2");
}

/// Before: `e^{x + 2i}` tended to `oo` and `sign(cosh(ln x + i))` to 1
/// (a real direction claimed for complex growth); `Γ(x − z)` (unassumed
/// `z`, a complex number) and `Γ(x + i)` tended to `oo`, though their
/// argument turns (`arg Γ(x + ib) ≈ b·ln x`); `ix`, `1 + ix`, `√(−x)` and
/// `(1 + i)x` were refused.
///
/// Oracle: mpmath `exp(10³ + 2i)/|·|` = −0.416147 + 0.909297j = `e^{2i}`,
/// `sign(cosh(ln 10⁴⁰ + i))` = 0.540302 + 0.841471j = `e^i`, `arg Γ(10ᵏ +
/// i)` = 0.624, 2.927, −1.053 for `k = 3, 4, 5`, `sqrt(−10⁴⁰)` = 10²⁰j;
/// `subs(x, oo)` gives `exp(2*I)*oo`, `I*oo`, `I*oo + 1`, `(1 + I)*oo`.
#[test]
fn directed_infinities() {
    let ctx = Context::new();
    same(&ctx, "exp(x + 2*I)", "oo", B, "exp(2*I)*oo");
    same(&ctx, "sign(cosh(log(x) + I))", "oo", B, "exp(I)");
    assert!(lim(&ctx, "gamma(x - z)", "oo", B).is_err());
    assert!(lim(&ctx, "gamma(x + I)", "oo", B).is_err());
    same(&ctx, "I*x", "oo", B, "I*oo");
    same(&ctx, "I*x + 1", "oo", B, "1 + I*oo");
    same(&ctx, "sqrt(-x)", "oo", B, "I*oo");
    same(&ctx, "x*(1 + I)", "oo", B, "(1 + I)*oo");
    same(&ctx, "cos(I*x)", "oo", B, "oo");
    same(&ctx, "sin(I*x)", "oo", B, "I*oo");
    assert!(lim(&ctx, "exp(I*x)", "oo", B).is_err());
    // A real `z` keeps `Γ(x − z) → oo`.
    let ctx = Context::new();
    ctx.symbol_with("z", &[Assumption::Real]).unwrap();
    same(&ctx, "gamma(x - z)", "oo", B, "oo");
}

/// An imaginary part that diverges more slowly than the real part leaves
/// a plain `oo` (direction 1); one that converges is kept: `√(x² + ix)`
/// was `oo`.
///
/// Oracle: mpmath `10⁴⁰ + 10²⁰ i` (direction 1, imaginary part diverging),
/// `sqrt(10⁸⁰ + 10⁴⁰ i)` = 10⁴⁰ + 0.5j, `10⁸⁰·exp(i/10⁴⁰)` = 10⁸⁰ + 10⁴⁰ j.
#[test]
fn weak_and_strong_imaginary_parts() {
    let ctx = Context::new();
    assert_eq!(show(&ctx, "x + I*sqrt(x)", "oo", B), "oo");
    assert_eq!(show(&ctx, "x^2*exp(I/x)", "oo", B), "oo");
    same(&ctx, "sqrt(x^2 + I*x)", "oo", B, "oo + I/2");
    same(&ctx, "x*exp(I/x)", "oo", B, "oo + I");
}

/// Reciprocal functions are quotients (`coth x = cosh x/sinh x`); their
/// limits were and are right.  Oracle: mpmath `coth(10⁴⁰)` = 1,
/// `coth(10⁴⁰ + i)` = 1, `acot(10⁴⁰ + i)` = 1.0e-40.
#[test]
fn reciprocal_functions_at_infinity() {
    let ctx = Context::new();
    same(&ctx, "coth(x)", "oo", B, "1");
    same(&ctx, "coth(x + I)", "oo", B, "1");
    same(&ctx, "acot(x + I)", "oo", B, "0");
}
