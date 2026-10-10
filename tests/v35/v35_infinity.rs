//! Infinities in sums and functions, reciprocal functions, `Min`/`Max` of
//! numbers (the infinity hunt after 0.40).  Every reference value cites the
//! oracle that produced it: SymPy 1.14, or mpmath 1.3 at 50 digits along
//! the path `oo → 10ᵏ` (`k = 20, 40, 80`; the limit the symbolic value
//! stands for).

use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s)
        .unwrap_or_else(|e| panic!("`{s}` does not parse: {e}"))
}

fn show(ctx: &Context, s: &str) -> String {
    p(ctx, s).to_string()
}

/// Before: `oo + I` was `oo` (every finite term was absorbed), so
/// `im(oo + I)` was `0`, `oo + I` claimed positive and real, and
/// `atanh(oo + I)` was `atanh(oo) = −iπ/2`.  `±oo` absorbs the real terms
/// only now, as SymPy's `Add.flatten`.
///
/// Oracle: SymPy 1.14 `oo + I` → `oo + I`, `im(oo + I)` → 1, `re(oo + I)`
/// → oo, `oo + x` → `x + oo`, `oo + sqrt(2)` → oo, `2*(oo + I)` → `oo +
/// 2*I`, `oo + I - oo` → nan, `1/(oo + I)` → 0; mpmath `atanh(mpc(1e20,
/// 1))` = 1.0e-20 + 1.5707963…j.
#[test]
fn oo_absorbs_real_terms_only() {
    let ctx = Context::new();
    let s = p(&ctx, "oo + I");
    assert_eq!(s.to_string(), "I + oo");
    assert_eq!(s.im(), ctx.one());
    assert_eq!(s.re(), ctx.infinity());
    assert_ne!(ctx.query(&s, Props::POSITIVE), Some(true));
    assert_ne!(ctx.query(&s, Props::REAL), Some(true));
    assert_ne!(ctx.query(&s, Props::FINITE), Some(true));
    assert_eq!(show(&ctx, "atanh(oo + I)"), "atanh(I + oo)");
    assert_eq!(p(&ctx, "atanh(oo + I)").eval(), p(&ctx, "I*pi/2"));
    assert_eq!(p(&ctx, "atanh(oo - I)").eval(), p(&ctx, "-I*pi/2"));
    // The real-axis convention is unchanged (SymPy `atanh(oo)` → -I*pi/2).
    assert_eq!(p(&ctx, "atanh(oo)").eval(), p(&ctx, "-I*pi/2"));
    assert_eq!(show(&ctx, "oo + x"), "x + oo");
    assert_eq!(show(&ctx, "oo + 1 + I"), "I + oo");
    assert_eq!(p(&ctx, "-oo + sqrt(2) + pi"), ctx.neg_infinity());
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    assert_eq!(&ctx.infinity() + &r, ctx.infinity());
    assert_eq!(show(&ctx, "2*(oo + I)"), "2*I + oo");
    assert_eq!(p(&ctx, "oo + I - oo"), ctx.nan());
    assert_eq!(p(&ctx, "1/(oo + I)"), ctx.zero());
    // Like terms of an infinite sum cannot cancel, nor its powers.
    assert_eq!(p(&ctx, "y*(oo + I) - y*(oo + I)"), ctx.nan());
    assert_eq!(p(&ctx, "(oo + I)*(oo + I)^(-1)"), ctx.nan());
    assert_eq!(p(&ctx, "(oo + I)^0"), ctx.nan());
}

/// Before: `π/2 + i·∞` was `i·∞` (the finite term absorbed by the directed
/// infinity), so `cos(π/2 + i·∞)` was `cos(i·∞) = ∞`, and `−3/2 + i·∞`
/// had real part 0.
///
/// Oracle: mpmath `cos(mpc(pi/2, 1e20))` = 3.0e-… − 1.3e+43429…j (the
/// direction −i; SymPy 1.14 gives nan); SymPy `pi/2 + I*oo` → `pi/2 +
/// oo*I`, `re(-3/2 + oo*I)` → -3/2, `1 + oo*I` stays.
#[test]
fn directed_infinities_keep_finite_terms() {
    let ctx = Context::new();
    assert_eq!(p(&ctx, "cos(pi/2 + I*oo)"), p(&ctx, "-I*oo"));
    assert_eq!(p(&ctx, "re(-3/2 + I*oo)"), ctx.rational(-3, 2));
    assert_eq!(p(&ctx, "re(I*oo + 1)"), ctx.one());
    assert_eq!(show(&ctx, "pi/2 + I*oo"), "1/2*pi + I*oo");
    // A real direction absorbs real terms (`r·∞ + 1 = r·∞`).
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    let ri = &r * &ctx.infinity();
    assert_eq!(&ri + 1, ri);
}

/// Before: `re(oo*x)` and `re(I*oo)` were `nan` — the decomposition
/// multiplied the structural zero imaginary part of `∞` by the other
/// factor's part, `0·∞`.
///
/// Oracle: SymPy 1.14 `re(oo*x)` → `oo*re(x)`, `re(oo*I)` → 0, `im(oo*I)`
/// → oo.
#[test]
fn real_and_imaginary_parts_of_directed_infinities() {
    let ctx = Context::new();
    assert_eq!(show(&ctx, "re(oo*x)"), "re(x)*oo");
    assert_eq!(p(&ctx, "re(I*oo)"), ctx.zero());
    assert_eq!(p(&ctx, "im(I*oo)"), ctx.infinity());
    assert_eq!(p(&ctx, "im(oo*x)"), p(&ctx, "im(x)*oo"));
}

/// Before: `acot(0)` was `atan(zoo)`, `coth(±oo)` and `cot(−I·oo)` were
/// `nan` (quotients of two infinities), `acoth(0)` was `atanh(zoo)`,
/// `asech(0)` was `zoo`.  The reciprocal functions take the values their
/// quotient forms lose first, by parsing and by the `Ex` methods alike.
///
/// Oracle: SymPy 1.14 `acot(0)` → pi/2, `coth(oo)` → 1, `coth(-oo)` → -1,
/// `cot(-I*oo)` → I, `cot(I*oo)` → -I, `acoth(0)` → I*pi/2, `asech(0)` →
/// oo, `asec(0)` → zoo, `acot(oo)` → 0; mpmath `coth(mpc(1e20, 1))` = 1,
/// `cot(mpc(1, 1e20))` = −1.0j.
#[test]
fn reciprocal_functions_at_their_special_points() {
    let ctx = Context::new();
    let half_pi = p(&ctx, "pi/2");
    assert_eq!(p(&ctx, "acot(0)"), half_pi);
    assert_eq!(ctx.zero().acot(), half_pi);
    assert_eq!(p(&ctx, "coth(oo)"), ctx.one());
    assert_eq!(ctx.infinity().coth(), ctx.one());
    assert_eq!(p(&ctx, "coth(-oo)"), ctx.int(-1));
    assert_eq!(p(&ctx, "coth(oo + I)"), ctx.one());
    assert_eq!(p(&ctx, "cot(-I*oo)"), ctx.i_unit());
    assert_eq!(p(&ctx, "-I*oo").cot(), ctx.i_unit());
    assert_eq!(p(&ctx, "cot(1 + I*oo)"), -ctx.i_unit());
    assert_eq!(p(&ctx, "acoth(0)"), p(&ctx, "I*pi/2"));
    assert_eq!(ctx.zero().acoth(), p(&ctx, "I*pi/2"));
    assert_eq!(p(&ctx, "asech(0)"), ctx.infinity());
    assert_eq!(p(&ctx, "asec(0)"), ctx.complex_infinity());
    assert_eq!(p(&ctx, "acot(oo)"), ctx.zero());
    // Elsewhere the quotient forms are unchanged.
    let x = ctx.symbol("x");
    assert_eq!(x.cot(), &x.cos() / &x.sin());
    assert_eq!(x.acot(), (&ctx.one() / &x).atan());
}

/// Before: `Min(0, 10)`, `Max(3, 0)` and `Min(-1, 1)` stayed unevaluated
/// until `eval`, so `ln(x·Min(0, 10))` was not seen to be `ln 0 = zoo`.
/// They fold when built now (rational arguments, `±oo`, `nan`), and `eval`
/// orders real constants.
///
/// Oracle: SymPy 1.14 `Min(0, 10)` → 0, `Max(3, 0)` → 3, `Min(-1, 1)` →
/// -1, `Min(oo, 1)` → 1, `Max(3, 0, x)` → `Max(3, x)`, `Min(pi, 3)` → 3,
/// `Max(sqrt(2), 3/2)` → 3/2.
#[test]
fn min_and_max_of_numbers_fold() {
    let ctx = Context::new();
    assert_eq!(p(&ctx, "Min(0, 10)"), ctx.zero());
    assert_eq!(p(&ctx, "Max(3, 0)"), ctx.int(3));
    assert_eq!(p(&ctx, "Min(-1, 1)"), ctx.int(-1));
    assert_eq!(ctx.int(-1).min_with(&ctx.int(1)), ctx.int(-1));
    assert_eq!(p(&ctx, "ln(x*Min(0, 10))"), ctx.complex_infinity());
    assert_eq!(p(&ctx, "Min(oo, 1)"), ctx.one());
    assert_eq!(p(&ctx, "Min(nan, 1)"), ctx.nan());
    // `zoo` has no order (SymPy raises "not comparable").
    assert_eq!(p(&ctx, "Max(-oo, zoo)"), ctx.nan());
    assert_eq!(p(&ctx, "Max(3, 0, x)"), p(&ctx, "Max(3, x)"));
    assert_eq!(p(&ctx, "Min(pi, 3)").eval(), ctx.int(3));
    assert_eq!(p(&ctx, "Max(sqrt(2), 3/2)").eval(), ctx.rational(3, 2));
}

/// Before: the values at `±oo`, `±I·oo` dropped their finite parts:
/// `asin(oo)` was `−I·oo` (real part π/2 lost), `acos(−oo)` `−I·oo`,
/// `acos(I·oo)` `−I·oo`, `ln(−oo)` and `acosh(−oo)` `oo` (imaginary part π
/// lost), `asinh(I·oo)` and `acosh(I·oo)` `oo`, `erfc(I·oo)` `−I·oo`.
///
/// Oracle: SymPy 1.14 numerically along the cuts (its branch convention):
/// `N(asin(10**20))` = 1.5707… − 46.74…i, `N(acos(-10**20))` = 3.1415… −
/// 46.74…i, `N(acos(10**20*I))` = 1.5707… − 46.74…i, `N(log(-10**20))` =
/// 46.05… + 3.1415…i, `N(acosh(-10**20))` = 46.74… + 3.1415…i,
/// `N(asinh(10**20*I))` = 46.74… + 1.5707…i, `N(acosh(10**20*I))` = 46.74…
/// + 1.5707…i; mpmath `erfc(mpc(0, 1e20))` = 1.0 − 5.2e+…j.
#[test]
fn values_at_infinities_keep_their_finite_parts() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let half_pi = p(&ctx, "pi/2");
    assert_eq!(p(&ctx, "re(asin(oo))"), half_pi);
    assert_eq!(p(&ctx, "im(asin(oo))"), ctx.neg_infinity());
    assert_eq!(p(&ctx, "re(asin(-oo))"), -&half_pi);
    assert_eq!(p(&ctx, "re(acos(-oo))"), pi);
    assert_eq!(p(&ctx, "re(acos(I*oo))"), half_pi);
    assert_eq!(p(&ctx, "im(acos(I*oo))"), ctx.neg_infinity());
    assert_eq!(p(&ctx, "im(ln(-oo))"), pi);
    assert_eq!(p(&ctx, "im(acosh(-oo))"), pi);
    assert_eq!(p(&ctx, "im(asinh(I*oo))"), half_pi);
    assert_eq!(p(&ctx, "im(asinh(-I*oo))"), -&half_pi);
    assert_eq!(p(&ctx, "re(asinh(-I*oo))"), ctx.neg_infinity());
    assert_eq!(p(&ctx, "im(acosh(I*oo))"), half_pi);
    assert_eq!(p(&ctx, "im(ln(-I*oo))"), -&half_pi);
    assert_eq!(p(&ctx, "re(erfc(I*oo))"), ctx.one());
    // Unchanged where nothing was lost.
    assert_eq!(p(&ctx, "acos(oo)"), p(&ctx, "I*oo"));
    assert_eq!(p(&ctx, "ln(oo)"), ctx.infinity());
    assert_eq!(p(&ctx, "asin(I*oo)"), p(&ctx, "I*oo"));
}

/// Functions at a sum of an infinity and finite terms (`oo + I`, `c +
/// I·oo`), which the canonical arithmetic keeps since this release.
///
/// Oracle: SymPy 1.14 `exp(oo + I)` → `oo*exp(I)`, `exp(-oo + I)` → 0,
/// `Abs(oo + I)` → oo, `arg(oo + I)` → 0, `atan2(1, oo)` → 0; mpmath at
/// `t = 1e20`: `tanh(mpc(t, 1))` = 1, `sin(mpc(1, t))` = (i·e^{−i})·∞
/// direction, `log(mpc(-t, 1))` = 46.05… + 3.1415…j, `log(mpc(-t, -1))` =
/// 46.05… − 3.1415…j, `atan(mpc(t, 1))` = 1.5707…, `sinh(mpc(-t, 1))` ∝
/// −e^{−i}.
#[test]
fn functions_at_infinite_sums() {
    let ctx = Context::new();
    assert_eq!(p(&ctx, "exp(oo + I)"), p(&ctx, "exp(I)*oo"));
    assert_eq!(p(&ctx, "exp(-oo + I)"), ctx.zero());
    assert_eq!(p(&ctx, "tanh(oo + I)"), ctx.one());
    assert_eq!(p(&ctx, "abs(oo + I)"), ctx.infinity());
    assert_eq!(p(&ctx, "sign(-oo + I)"), ctx.int(-1));
    assert_eq!(p(&ctx, "arg(oo + I)").eval(), ctx.zero());
    assert_eq!(p(&ctx, "atan(oo + I)").eval(), p(&ctx, "pi/2"));
    assert_eq!(p(&ctx, "sin(1 + I*oo)"), p(&ctx, "I*exp(-I)*oo"));
    assert_eq!(p(&ctx, "sinh(-oo + I)"), p(&ctx, "-exp(-I)*oo"));
    assert_eq!(p(&ctx, "im(ln(-oo + I))"), ctx.pi());
    assert_eq!(p(&ctx, "im(ln(-oo - I))"), -ctx.pi());
    assert_eq!(p(&ctx, "tan(1 - I*oo)"), -ctx.i_unit());
    assert_eq!(p(&ctx, "erf(-oo + I)"), ctx.int(-1));
    assert_eq!(p(&ctx, "atan2(-1, -oo)").eval(), -ctx.pi());
}
