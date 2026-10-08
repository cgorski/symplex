//! Expressions undefined at every point (`0/0`) that still got a value
//! after 0.34 (the round after `tests/v33/v33_undefined.rs`): a constant
//! denominator that is exactly 0 inside a function argument, denominators
//! undefined more than one level deep, a sum distributed over a
//! denominator zero by a trigonometric identity, and a constant zero by
//! `sin²u + cos²u = 1` whose `u` is too costly to evaluate.
//!
//! The oracle is the value at every point: each hidden identity is checked
//! with SymPy 1.14 (`.equals(0)` at the point), replaced by an exact `0`,
//! and the expression evaluated at the point with SymPy's arithmetic
//! (`S(0)/S(0)` → `nan`, `1/S(0)` → `zoo`, `zoo + zoo` → `nan`, `sin(zoo)`
//! → `nan`, `S(3)/zoo` → `0`).  SymPy's own `ratsimp`, `expand` and
//! sequential `subs` give `0` or a sum for several of the symbolic forms
//! (it folds a numerator to 0 before it looks at a denominator):
//! deliberately different, as in the 0.34 tests.

use symplex::prelude::*;

fn show(e: &Ex) -> String {
    format!("{e}")
}

fn parse(src: &str) -> String {
    let ctx = Context::new();
    show(&ctx.parse(src).unwrap())
}

/// `src` with `x`, `y` substituted (in that order) by the rationals `xv`,
/// `yv`, then evaluated (`eval`, which folds `exp(22/7)⁴` into
/// `exp(88/7)`).
fn value_at(src: &str, xv: &str, yv: &str) -> String {
    let ctx = Context::new();
    let e = ctx.parse(src).unwrap();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    show(
        &e.subs(&x, &ctx.parse(xv).unwrap())
            .subs(&y, &ctx.parse(yv).unwrap())
            .eval(),
    )
}

/// The constant `sin²1 + cos²1 − 1` (exactly 0).
const ZC: &str = "(sin(1)^2+cos(1)^2-1)";

/// Before: `ratsimp` gave `0`.  The generator `sin((√2 + e)/(sin²1 + cos²1
/// − 1))` is `sin(zoo) = nan` at every point (the canonical form keeps the
/// quotient: testing every constant denominator numerically is beyond the
/// budget of a constructor), and the factor `(√y + √x)² − x − y − 2√x·√y`
/// is 0: the product is `nan·0`, not the `0` of the polynomial arithmetic.
/// `simplify`, `expand` and `simplify_trig` already gave `nan`; so does
/// the canonical product `0·sin(c/0)` now (it was `0`).
///
/// Oracle: SymPy 1.14 `(sin(1)**2 + cos(1)**2 - 1).equals(0)` → `True`,
/// `((sqrt(y) + sqrt(x))**2 - x - y - 2*sqrt(x)*sqrt(y)).subs({x: 3/7, y:
/// 5/11}).equals(0)` → `True`, `sin((sqrt(2) + E)/S(0))` → `nan`, `nan*0` →
/// `nan`.  (SymPy's `ratsimp` and `expand` give `0`; its `0*sin(…)` is `0`.)
#[test]
fn a_function_of_a_constant_over_zero_is_undefined() {
    let ctx = Context::new();
    let src =
        format!("sin((sqrt(2)+exp(1))/{ZC})*y^2/(1/((sqrt(y)+sqrt(x))^2-x-y-2*sqrt(x)*sqrt(y)))");
    let e = ctx.parse(&src).unwrap();
    assert_eq!(show(&e.ratsimp()), "nan");
    assert_eq!(show(&e.simplify()), "nan");
    assert_eq!(show(&e.expand()), "nan");
    assert_eq!(show(&e.simplify_trig()), "nan");
    assert_eq!(parse(&format!("0*sin((sqrt(2)+exp(1))/{ZC})")), "nan");
    assert_eq!(parse(&format!("zoo*sin(1/{ZC})")), "nan");
    // A function at an argument that is merely a quotient keeps its form.
    assert_eq!(parse("0*sin(1/(x+1))"), "0");
    assert_eq!(
        show(&ctx.parse("sin(x)/(x+1)+cos(1/(x+2))").unwrap().ratsimp()),
        "(x*cos(1/(x + 2)) + sin(x) + cos(1/(x + 2)))/(x + 1)"
    );
}

/// Before: `0` at every substituted point, evaluated (hunter seeds 748,
/// 1534, 4875, 4984, 6140).  The zero numerator met a denominator `c + N/d` with `d`
/// zero whose numerator `N` is itself undefined — two terms over zero
/// denominators (`zoo + zoo`) or a `0/0` — so the denominator is `nan`, not
/// the `zoo` the one-level test of `0·D⁻¹` took it for, and `0/nan` is
/// `nan`.
///
/// Oracle: SymPy 1.14 with each hidden identity verified `.equals(0)` at
/// the point and replaced by `S(0)`, e.g. seed 748 at `x = 44/19, y =
/// 32/17`: `0/(y + (1/(x*0) + ((cos(x) + 2)/0 + x/0)/0)/0)` → `nan`; the
/// others likewise → `nan`.
#[test]
fn denominators_undefined_more_than_one_level_deep() {
    assert_eq!(
        value_at(
            "(exp(x)*(exp(2*x)-exp(x)^2)*(cos(x)+2)*((x+(2))^2-x^2-2*(2)*x-(2)^2))/(y+((1/(x))/(((x+y)^2-x^2-2*x*y-y^2))+(((cos(x)+2))/((exp(x/2)^2-exp(x)))+(x)/((sin(2*x)-2*sin(x)*cos(x))))/(((1+sqrt(2))^2-3-2*sqrt(2))))/(exp(2*x)-exp(x)^2))",
            "44/19",
            "32/17"
        ),
        "nan"
    );
    assert_eq!(
        value_at(
            "((exp(x))*((y/(x+(-4))-y*x/(x^2+(-4)*x))))/((x^2+4)+(((((sqrt(y)+sqrt(x))^2-x-y-2*sqrt(x)*sqrt(y)))/(sin(x)^2+cos(x)^2-1)+(x)/(sin(x)^2+cos(x)^2-1)))/(exp(3*x)-exp(x)^3))",
            "45/11",
            "-53/13"
        ),
        "nan"
    );
    assert_eq!(
        value_at(
            "((sin(((x+(3))^2-x^2-2*(3)*x-(3)^2)))/(1/(2)))/((x^2+9)+(((x)/(1+((y+(-4)))/(exp(2*x)-exp(x)^2)))/((cosh(x)^2-sinh(x)^2-1))+((x*y+(-6))*(x+(-2))^2)/(((1+sqrt(2))^2-3-2*sqrt(2))))/((sqrt(x)+1)^2-x-2*sqrt(x)-1))",
            "-32/17",
            "-2/11"
        ),
        "nan"
    );
    assert_eq!(
        value_at(
            "(((-3)*((sin(x)^2+cos(x)^2-1)))*((x*(y+(-3))-x*y-(-3)*x)))/(x+(((sin(2*x)-2*sin(x)*cos(x)))/((sin(1)^2+cos(1)^2-1))+((sqrt(2)+exp(1)))/(((1+sqrt(2))^2-3-2*sqrt(2)))*(exp(2*x)-exp(x)^2)-sqrt(x+7))/(exp(3*x)-exp(x)^3))",
            "-27/11",
            "12/19"
        ),
        "nan"
    );
    assert_eq!(
        value_at(
            "(((exp(4*x)-exp(x)^4))^2*abs(x-3)*y)/((y+(5))+((((x)/(exp(x)^2*exp(y)-exp(2*x+y))+(exp(x))/(exp(x)^2*exp(y)-exp(2*x+y))))/((y/(x+(-2))-y*x/(x^2+(-2)*x))+sqrt(x+7)))/((1+sqrt(2))^2-3-2*sqrt(2)))",
            "22/7",
            "51/13"
        ),
        "nan"
    );
}

/// Before: `0`, `0`, `oo`.  The canonical constructors decide `0·D⁻¹`,
/// `0·f` and `∞ + t` for a `D`, `f`, `t` undefined below their first level:
/// a sum factor with two terms over constants that are 0, the square of a
/// sum with one.  Correct expressions keep their forms.
///
/// Oracle: SymPy 1.14 `(exp(2) - exp(1)**2).equals(0)` → `True`; `S(0)/(x +
/// y*(zoo + zoo))` at `x = 2/7, y = 5/11` → `nan`, `0*(zoo + 2)**2` →
/// `nan`, `oo + 5*(zoo + zoo)/11` → `nan`.  (SymPy keeps or folds the
/// symbolic forms to `0`: deliberately different.)
#[test]
fn canonical_zero_and_infinity_paths_look_below_the_first_level() {
    let ze = "(exp(2)-exp(1)^2)";
    assert_eq!(parse(&format!("0/(x+y*(1/{ZC}+1/{ze}))")), "nan");
    assert_eq!(parse(&format!("0*(1/{ZC}+2)^2")), "nan");
    assert_eq!(parse(&format!("oo+y*(1/{ZC}+1/{ze})")), "nan");
    assert_eq!(parse("0/(x+y*(1/(x+1)+1/(x+2)))"), "0");
    assert_eq!(parse("0*(1/(x+1)+2)^2"), "0");
    assert_eq!(parse("oo+x*oo"), "x*oo + oo");
    assert_eq!(parse("oo+y*(1/(x+1)+1/(x+2))"), "oo");
}

/// Before: `y²/s + 3y/s` for `s = sin 2x − 2·sin x·cos x`, which is `zoo +
/// zoo = nan` at every point while the value is `y·(y + 3)/0 = zoo`; and
/// the `0` of hunter seed 1831, `−s/((…)·(y·(y + 3)/s + |x − 3|))`, became
/// `nan` by the same distribution.  The residue test now relates `sin`,
/// `cos` and `tan` (and `sinh`, `cosh`, `tanh`) through `e^{iu}` and `eᵘ`,
/// so `s` is suspected without multiplying anything out, and the identity
/// test of `simplify` confirms it.  A sum over a nonzero denominator is
/// still distributed.
///
/// Oracle: SymPy 1.14 `(sin(2*x) - 2*sin(x)*cos(x)).equals(0)` → `True`;
/// `y*(y + 3)/S(0)` at `y = 5/11` → `zoo`; seed 1831 at `x = −1/13, y =
/// 47/7` with `s` → `S(0)`: `0/((…)*(zoo + 10/13))` → `0`.  (SymPy's
/// `expand` gives `y**2/s + 3*y/s`: deliberately different.)
#[test]
fn expand_does_not_distribute_over_an_identity_zero() {
    let ctx = Context::new();
    let s = "(sin(2*x)-2*sin(x)*cos(x))";
    let p = |src: &str| ctx.parse(src).unwrap();
    assert_eq!(show(&p(&format!("y*(y+3)/{s}")).expand()), "zoo");
    assert_eq!(
        show(
            &p(&format!(
                "((({s})/(-1))/(abs(x-3)+((y+(3))*y)/{s}))/(1/((x+(2))^2)-abs(x-3)*-5)"
            ))
            .expand()
        ),
        "0"
    );
    assert_eq!(show(&p("y*(y+3)/(cosh(x)^2-sinh(x)^2-1)").expand()), "zoo");
    assert_eq!(
        show(&p("y*(y+3)/(sin(2*x)+2)").expand()),
        "y^2/(sin(2*x) + 2) + 3*y/(sin(2*x) + 2)"
    );
}

/// Before: `zoo` from `simplify` and `simplify_trig`, the quotient
/// multiplied out from `ratsimp` and `expand`.  `(sin²x + cos²x −
/// 1)²/√(cosh²x − sinh²x − 1)` is `0/0`: the factor beside the pole was
/// tested only beside an integer power of a vanishing base, not beside
/// `s^(−1/2)`, and nothing suspected the numerator in `ratsimp` and
/// `expand` (the residues relate `sin` and `cos` now).
///
/// Oracle: SymPy 1.14 `(sin(x)**2 + cos(x)**2 - 1).equals(0)` and
/// `(cosh(x)**2 - sinh(x)**2 - 1).equals(0)` → `True`;
/// `S(0)**2/sqrt(S(0))` → `nan`.  (SymPy's `simplify` gives `0`, its
/// `ratsimp` the quotient multiplied out: deliberately different.)
#[test]
fn a_zero_over_a_radical_of_an_identity_zero_is_undefined() {
    let ctx = Context::new();
    let e = ctx
        .parse("(sin(x)^2+cos(x)^2-1)^2/sqrt(cosh(x)^2-sinh(x)^2-1)")
        .unwrap();
    assert_eq!(show(&e.ratsimp()), "nan");
    assert_eq!(show(&e.simplify()), "nan");
    assert_eq!(show(&e.simplify_trig()), "nan");
    assert_eq!(show(&e.expand()), "nan");
    // A radical of an identity zero beside a pole (seed 3802 of the
    // variant hunter): `√0/0` makes the denominator `nan`; `ratsimp`,
    // `expand`, `simplify_trig` and `simplify` gave `0`.  Oracle: SymPy
    // 1.14 `(tan(x)*cos(x) - sin(x)).equals(0)` → `True`,
    // `sqrt(S(0))/S(0)` → `nan`, `exp(Rational(2, 7)) + nan` → `nan`.
    let f = ctx
        .parse("((y-6)*-3)/(exp(x)+sqrt(tan(x)*cos(x)-sin(x))/(x*(x-2)-x^2+2*x))")
        .unwrap();
    assert_eq!(show(&f.ratsimp()), "nan");
    assert_eq!(show(&f.expand()), "nan");
    assert_eq!(show(&f.simplify_trig()), "nan");
    assert_eq!(show(&f.simplify()), "nan");
}

/// Before: `zoo`.  `sin²u + cos²u − 1` for `u = besselj(10⁵, 1)` is 0, but
/// the canonical zero test declines to evaluate a special function at a
/// parameter of 10⁵ or more (0.34's cost guard), so `0·zoo` stayed `zoo`.
/// The residue test sees the identity whatever `u` is, and the identity
/// test decides it with `u` replaced by a symbol — `u` is never evaluated.
/// A costly constant that is not 0 is still not taken for 0.
///
/// Oracle: SymPy 1.14 `(sin(x)**2 + cos(x)**2 - 1).equals(0)` → `True`
/// (an identity for every `x`, so at `x = besselj(10**5, 1)`), `0*zoo` →
/// `nan`.  (SymPy 1.14 did not finish `(sin(besselj(10**5, 1))**2 +
/// cos(besselj(10**5, 1))**2 - 1)*zoo` in 60 s.)
#[test]
fn costly_constants_zero_by_an_identity_are_zero() {
    let u = "besselj(10^5,1)";
    let t = std::time::Instant::now();
    assert_eq!(parse(&format!("(sin({u})^2+cos({u})^2-1)*zoo")), "nan");
    assert_eq!(
        parse(&format!(
            "(cosh({u})^2-sinh({u})^2-1)/(cosh({u})^2-sinh({u})^2-1)"
        )),
        "nan"
    );
    assert_eq!(parse(&format!("0/(tan({u})*cos({u})-sin({u}))")), "nan");
    assert_eq!(parse(&format!("(sin({u})+2)*zoo")), "zoo");
    assert_eq!(
        parse("(sin(gamma(10^5))^2+cos(gamma(10^5))^2-2)*zoo"),
        "zoo"
    );
    assert!(t.elapsed().as_secs() < 30, "took {:?}", t.elapsed());
}

/// Before (debug build): `ratsimp` 119 s, `expand` 113 s, `simplify` 107 s
/// on these, `simplify_trig` 30 s with `legendre(99999, 1/3)`: the identity
/// test of `simplify` sampled and then computed `jacobi(999, 1/3, 1/5,
/// 1/7)` exactly, a cost the canonical zero test declines.  The identity test now takes a costly part for a
/// symbol, as the canonical one does, and decides the same values in
/// milliseconds.
///
/// Oracle: SymPy 1.14 `(sin(x)**2 + cos(x)**2 - 1).equals(0)` and
/// `((x + 1)**2 - x**2 - 2*x - 1).equals(0)` → `True`; `S(0)/S(0)` → `nan`,
/// `1/S(0)` → `zoo`.
#[test]
fn identity_tests_never_evaluate_costly_parts() {
    let ctx = Context::new();
    let u = "jacobi(999, 1/3, 1/5, 1/7)";
    let t = std::time::Instant::now();
    let e = ctx
        .parse(&format!("((x+1)^2-x^2-2*x-1)/(sin({u})^2+cos({u})^2-1)"))
        .unwrap();
    assert_eq!(show(&e.ratsimp()), "nan");
    assert_eq!(show(&e.expand()), "nan");
    assert_eq!(show(&e.simplify_trig()), "nan");
    let f = ctx.parse(&format!("1/(sin({u})^2+cos({u})^2-1)")).unwrap();
    assert_eq!(show(&f.simplify()), "zoo");
    let v = "legendre(99999, 1/3)";
    let g = ctx
        .parse(&format!("((x+1)^2-x^2-2*x-1)/(sin({v})^2+cos({v})^2-1)"))
        .unwrap();
    assert_eq!(show(&g.simplify_trig()), "nan");
    assert!(t.elapsed().as_secs() < 30, "took {:?}", t.elapsed());
}

/// What SymPy does too stays as it was: `T/T = 1` and like terms collected
/// over a `T` that is zero only by an identity of its functions (the
/// canonical form tests identities of `exp` and radicals, and constants,
/// not those of `sin` and `cos` in a symbol); `0/T` for such a `T` is `0`.
///
/// Oracle: SymPy 1.14, with `T = sin(x)**2 + cos(x)**2 - 1`: `T/T` → `1`,
/// `2/T + 3/T` → `5/T`, `0/T` → `0`; `ratsimp(1/(x/s + 1/s))` → `s/(x +
/// 1)`.
#[test]
fn what_sympy_also_does_is_unchanged() {
    let t = "(sin(x)^2+cos(x)^2-1)";
    assert_eq!(parse(&format!("{t}/{t}")), "1");
    assert_eq!(
        parse(&format!("2/{t}+3/{t}")),
        "5/(sin(x)^2 + cos(x)^2 - 1)"
    );
    assert_eq!(parse(&format!("0/{t}")), "0");
    let ctx = Context::new();
    let s = "(sin(2*x)-2*sin(x)*cos(x))";
    assert_eq!(
        show(&ctx.parse(&format!("1/(x/{s}+1/{s})")).unwrap().ratsimp()),
        "(-2*sin(x)*cos(x) + sin(2*x))/(x + 1)"
    );
}
