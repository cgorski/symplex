//! Expressions undefined at every point (`0/0`) that evaluation, the
//! canonical constructors, `ratsimp`, `expand` or the trigonometric
//! simplifiers still turned into a value (the 0.34 round after 0.33's
//! `tests/v32/v32_undefined.rs`): a constant denominator that is exactly 0,
//! identities of exponentials and radicals the canonical form keeps
//! (`exp(2x) − exp(x)²`), `zoo + zoo` lost when fractions over a zero
//! denominator are added or collected, and denominators zero by an
//! identity of their functions outside `simplify`.
//!
//! The oracle is the value at every point: a sum that vanishes
//! identically is 0 (checked with SymPy 1.14: `.equals(0)`, `expand`), so
//! `0/0` is `nan`, `P/0` is `zoo`, `zoo + zoo` is `nan`.  SymPy itself
//! gives `0` for several of the symbolic forms (it folds a numerator to 0
//! before it looks at the denominator); symplex's `simplify` has given
//! `nan` there since 0.33, and the other operations now agree with it.

use symplex::prelude::*;

fn show(e: &Ex) -> String {
    format!("{e}")
}

fn parse(src: &str) -> String {
    let ctx = Context::new();
    show(&ctx.parse(src).unwrap())
}

/// `src` with `x`, `y` substituted (in that order) by the rationals `xv`,
/// `yv`.
fn at(src: &str, xv: &str, yv: &str) -> String {
    let ctx = Context::new();
    let e = ctx.parse(src).unwrap();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    show(
        &e.subs(&x, &ctx.parse(xv).unwrap())
            .subs(&y, &ctx.parse(yv).unwrap()),
    )
}

/// Bug 1: `eval` of `(e² − exp(1)²)/(sin²1 + cos²1 − 1)` was `0` —
/// `exp(1)²` evaluates to `exp(2)`, the numerator to 0, and `0·d⁻¹` was `0`
/// for the constant denominator, which is exactly 0 too.  `eval_decimal`
/// and `eval_complex64` (which call `eval`) gave `0` as well; they now
/// refuse.  The canonical constructors decide a constant factor met on
/// their rare paths (`0·d⁻¹`, `zoo·f`, `s·s⁻¹`) numerically, then exactly.
///
/// Oracle: SymPy `(sin(1)**2 + cos(1)**2 - 1).equals(0)` → `True`,
/// `exp(2) - E**2` → `0`: `0/0`.  (SymPy gives `0` for the quotient
/// `(exp(2) - exp(1)**2)/(sin(1)**2 + cos(1)**2 - 1)` and for
/// `S(0)/(sin(1)**2 + cos(1)**2 - 1)`: deliberately different.)
#[test]
fn eval_of_a_constant_zero_over_zero_is_nan() {
    let ctx = Context::new();
    let e = ctx
        .parse("(exp(2)-exp(1)^2)/(sin(1)^2+cos(1)^2-1)")
        .unwrap();
    // The canonical form keeps both (neither is 0 structurally).
    assert_eq!(show(&e), "(-exp(1)^2 + exp(2))/(sin(1)^2 + cos(1)^2 - 1)");
    assert_eq!(show(&e.eval()), "nan");
    assert!(e.eval_decimal(30).is_err());
    assert!(e.eval_complex64().is_err());
    assert_eq!(parse("0/(sin(1)^2+cos(1)^2-1)"), "nan");
    assert_eq!(parse("(sin(1)^2+cos(1)^2-1)/(sin(1)^2+cos(1)^2-1)"), "nan");
    assert_eq!(parse("(exp(2)-exp(1)^2)*zoo"), "nan");
    // Constants that are not 0 are untouched.
    assert_eq!(parse("0/(sqrt(2)-1)"), "0");
    assert_eq!(parse("(sqrt(2)-1)/(sqrt(2)-1)"), "1");
    assert_eq!(show(&ctx.parse("exp(1)^2").unwrap().eval()), "exp(2)");
}

/// Bug 2: the zero test of the canonical constructors took every generator
/// as independent, so `exp(2) − exp(1)²` and `exp(2x) − exp(x)²` (kept by
/// the canonical form: SymPy folds `exp(x)**2` to `exp(2*x)`, symplex only
/// in `eval`) were not zero: `(e² − exp(1)²)/(√2² − 2)` was `zoo` and
/// `(exp(2x) − exp(x)²)/(exp(2x) − exp(x)²)` was `1`.  The residues now
/// treat exponentials multiplicatively and rational powers of a generator,
/// `i` and square roots of integers by their values.
///
/// Oracle: SymPy `(exp(2) - exp(1)**2)/(sqrt(2)**2 - 2)` → `nan`,
/// `(exp(2*x) - exp(x)**2)/(exp(2*x) - exp(x)**2)` → `nan`;
/// `((exp(x+y) - exp(x)*exp(y))/(x - x)).subs({x: 2/7, y: 3/11})` → `nan`;
/// `expand((sqrt(x) + 1)**2 - x - 2*sqrt(x) - 1)` → `0`.
#[test]
fn canonical_zero_test_sees_exponential_and_radical_identities() {
    assert_eq!(parse("(exp(2)-exp(1)^2)/(sqrt(2)^2-2)"), "nan");
    assert_eq!(parse("(exp(2*x)-exp(x)^2)/(exp(2*x)-exp(x)^2)"), "nan");
    assert_eq!(parse("(exp(x/2)^2-exp(x))/(exp(x/2)^2-exp(x))"), "nan");
    assert_eq!(parse("(exp(x+y)-exp(x)*exp(y))/(x-x)"), "nan");
    assert_eq!(parse("((sqrt(x)+1)^2-x-2*sqrt(x)-1)/(x-x)"), "nan");
    assert_eq!(
        parse("((sqrt(y)+sqrt(x))^2-x-y-2*sqrt(x)*sqrt(y))/(x-x)"),
        "nan"
    );
    assert_eq!(
        parse("0/((sqrt(2)+sqrt(-3))^2+1-2*sqrt(2)*sqrt(-3))"),
        "nan"
    );
    // The canonical forms themselves are unchanged.
    assert_eq!(parse("exp(2*x)-exp(x)^2"), "-exp(x)^2 + exp(2*x)");
    assert_eq!(parse("exp(1)^2"), "exp(1)^2");
    assert_eq!(parse("exp(2)"), "exp(2)");
    assert_eq!(parse("(x+1)/(x+1)"), "1");
    assert_eq!(parse("0/(x+1)"), "0");
    assert_eq!(parse("(exp(2*x)+exp(x)^2)/(exp(2*x)+exp(x)^2)"), "1");
}

/// Bug 3: `ratsimp` added two fractions over the denominator 0 as
/// `(p₁ + p₂)/0`, so `1/0 + 0/0` was `zoo` (it is `nan`, as `zoo + zoo`
/// and `nan + …` are) and `x·y/(1/(x(x + 3) − x² − 3x) + ((x + y)² − x² −
/// 2xy − y²)/(x(y + 4) − xy − 4x))` became `x·y/zoo = 0` in `ratsimp` and
/// `simplify`.
///
/// Oracle: SymPy `simplify(...)` → `nan` (its `ratsimp` → `0`:
/// deliberately different); `expand` already gave `nan`.
#[test]
fn ratsimp_adds_two_zero_denominators_to_nan() {
    let ctx = Context::new();
    let e = ctx
        .parse("x*y/(1/(x*(x+3)-x^2-3*x) + ((x+y)^2-x^2-2*x*y-y^2)/(x*(y+4)-x*y-4*x))")
        .unwrap();
    assert_eq!(show(&e.ratsimp()), "nan");
    assert_eq!(show(&e.simplify()), "nan");
    assert_eq!(show(&e.expand()), "nan");
    // A denominator 0 only as a function of its generators is 0 for
    // `ratsimp` too.
    let f = ctx.parse("((x+1)^2-x^2-2*x-1)/(exp(2)-exp(1)^2)").unwrap();
    assert_eq!(show(&f.ratsimp()), "nan");
    let g = ctx
        .parse("(x*y-2)^2/((exp(x/2)^2-exp(x))/(exp(2)-exp(1)^2)+(y+1)/(exp(2)-exp(1)^2))")
        .unwrap();
    assert_eq!(show(&g.ratsimp()), "nan");
    // Ordinary input is unchanged.
    assert_eq!(
        show(&ctx.parse("(x^2-1)/(x-1)").unwrap().ratsimp()),
        "x + 1"
    );
    assert_eq!(
        show(&ctx.parse("(exp(2*x)-1)/(exp(x)-1)").unwrap().ratsimp()),
        "(exp(2*x) - 1)/(exp(x) - 1)"
    );
}

/// Bug 4: like terms over a denominator `d` that vanishes identically were
/// collected, `a/d + b/d → (a + b)/d = zoo`, while each is `zoo` at every
/// point and the sum `nan`; a later division made that `zoo` a `0`.  The
/// 0.33 hunter's reproducer, substituted `x = 3/19` then `y = −16/17`, was
/// `0`.
///
/// Oracle: the exact value at the point: `(3/19)/0 + (3/19 + 3)²/0` (SymPy:
/// `S(3)/19/S(0) + (S(3)/19 + 3)**2/S(0)` → `nan`), and `nan` propagates;
/// the source with the rationals substituted textually parses to `nan`.
#[test]
fn like_terms_over_a_vanishing_denominator_are_nan() {
    let src = "(abs(abs(x-3)))^2*((((x+(-3))^2-x^2-2*(-3)*x-(-3)^2))*(((x+(3))^2-x^2-2*(3)*x-(3)^2)))\
               /(x+((x)/(((x+y)^2-x^2-2*x*y-y^2))+((x+(3))^2)/(((x+y)^2-x^2-2*x*y-y^2)))\
               /(x*(x+(-4))-x^2-(-4)*x))";
    assert_eq!(at(src, "3/19", "-16/17"), "nan");
    let exact = src.replace('x', "(3/19)").replace('y', "(-16/17)");
    assert_eq!(parse(&exact), "nan");
    let d = "((x+y)^2-x^2-2*x*y-y^2)";
    assert_eq!(parse(&format!("2/{d}+3/{d}")), "nan");
    assert_eq!(parse(&format!("1/{d}-1/{d}")), "nan");
    assert_eq!(at(&format!("x/{d}+(x+3)^2/{d}"), "3/19", "-16/17"), "nan");
    // Terms over a constant denominator that is 0 (`y` substituted first:
    // over the symbolic `sin²y + cos²y − 1` the canonical form collects
    // them, as it has no identity test).
    let ctx = Context::new();
    let s = "(sin(y)^2+cos(y)^2-1)";
    let e = ctx.parse(&format!("x/{s}+(x+3)^2/{s}")).unwrap();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let v = e
        .subs(&y, &ctx.parse("-16/17").unwrap())
        .subs(&x, &ctx.parse("3/19").unwrap());
    assert_eq!(show(&v), "nan");
    // Ordinary like terms still collect.
    assert_eq!(parse("x/(x+1)+2*x/(x+1)"), "3*x/(x + 1)");
    assert_eq!(parse("2/(exp(x)+1)+3/(exp(x)+1)"), "5/(exp(x) + 1)");
}

/// Bug 5: outside `simplify`, a denominator zero by an identity of its
/// functions was not seen, and the numerator multiplied out to 0 made the
/// quotient `0`: `expand(((x + 1)² − x² − 2x − 1)/(sin²x + cos²x − 1))` and
/// `simplify_trig(((x + 1)² − x² − 2x − 1)/(tan x·cos x − sin x))` were `0`.
/// `expand`, `ratsimp` and `simplify_powers` test the denominator of such a
/// `0/0` candidate (a numerator the residue test cannot rule out as 0);
/// `simplify_trig` and `fu`, whose identities make a numerator 0, test
/// every such denominator, as `simplify`.
///
/// Oracle: SymPy `(sin(x)**2 + cos(x)**2 - 1).equals(0)` → `True`,
/// `(tan(x)*cos(x) - sin(x)).equals(0)` → `True`: `0/0` at every point.
/// SymPy's `expand`, `ratsimp`, `simplify` and `powsimp` give `0`, its
/// `trigsimp` `zoo*(-x**2 - 2*x + (x + 1)**2 - 1)` (deliberately different:
/// symplex's `simplify` has given `nan` since 0.33).
#[test]
fn identity_zero_denominators_outside_simplify() {
    let ctx = Context::new();
    let n = "((x+1)^2-x^2-2*x-1)";
    let e = ctx.parse(&format!("{n}/(sin(x)^2+cos(x)^2-1)")).unwrap();
    assert_eq!(show(&e.expand()), "nan");
    let f = ctx.parse(&format!("{n}/(tan(x)*cos(x)-sin(x))")).unwrap();
    assert_eq!(show(&f.simplify_trig()), "nan");
    assert_eq!(show(&f.fu()), "nan");
    assert_eq!(show(&f.ratsimp()), "nan");
    assert_eq!(show(&f.simplify()), "nan");
    assert_eq!(
        show(
            &ctx.parse(&format!("x+{n}/(tan(x)*cos(x)-sin(x))"))
                .unwrap()
                .expand()
        ),
        "nan"
    );
    let g = ctx
        .parse("(sin(2*x)-2*sin(x)*cos(x))/(tan(x)*cos(x)-sin(x))")
        .unwrap();
    assert_eq!(show(&g.simplify_trig()), "nan");
    let h = ctx
        .parse("(exp(x)*exp(-x)-1)/(tan(x)*cos(x)-sin(x))")
        .unwrap();
    assert_eq!(show(&h.simplify_powers()), "nan");
    // Ordinary input is unchanged.
    assert_eq!(
        show(&ctx.parse("(x+1)/(sin(x)+2)").unwrap().expand()),
        "x/(sin(x) + 2) + 1/(sin(x) + 2)"
    );
    assert_eq!(
        show(&ctx.parse(&format!("{n}/(sin(x)+2)")).unwrap().expand()),
        "0"
    );
    assert_eq!(
        show(&ctx.parse("sin(x)^2+cos(x)^2").unwrap().simplify_trig()),
        "1"
    );
    assert_eq!(
        show(
            &ctx.parse("(sin(x)^2+cos(x)^2)/(sin(x)+2)")
                .unwrap()
                .simplify_trig()
        ),
        "1/(sin(x) + 2)"
    );
}

/// `expand` replaces a constant denominator that is 0 by `zoo` before it
/// distributes: `expand((x − 5)/(√2 + e + (x − 3)²/(sin²1 + cos²1 − 1)))`
/// was a sum of terms `c/0` (`nan` at every point); the value is `(x −
/// 5)/zoo = 0`.
///
/// Oracle: SymPy `((x - 5)/(sqrt(2) + E + (x - 3)**2/S(0))).subs(x, 2/7)`
/// → `0` (with the unevaluated denominator, SymPy keeps a quotient).
#[test]
fn expand_decides_constant_zero_denominators() {
    let ctx = Context::new();
    let e = ctx
        .parse("(x-5)/(sqrt(2)+exp(1)+(x-3)^2/(sin(1)^2+cos(1)^2-1))")
        .unwrap();
    assert_eq!(show(&e.expand()), "0");
}

/// Substituted points make constants whose zeros need the values of `i`,
/// of square roots and of exponentials with different denominators:
/// `(√y + √x)² − x − y − 2√x·√y` at `x = −38/13`, `y = 41/17` and `exp(x +
/// y) − exp(x)·exp(y)` at `x = −46/11`, `y = −13/7` are 0, and the
/// quotients below `0/0`; both were `0`.
///
/// Oracle: SymPy `((sqrt(y) + sqrt(x))**2 - x - y -
/// 2*sqrt(x)*sqrt(y)).subs({x: -38/13, y: 41/17}).equals(0)` → `True`;
/// `exp(-46/11)*exp(-13/7)` → `exp(-465/77)`.  (SymPy's sequential `subs`
/// gives `0` for both quotients.)
#[test]
fn zero_constants_at_substituted_points() {
    assert_eq!(
        at(
            "((x-4)^2-x^2+8*x-16)/((sqrt(y)+sqrt(x))^2-x-y-2*sqrt(x)*sqrt(y))",
            "-38/13",
            "41/17"
        ),
        "nan"
    );
    assert_eq!(
        at(
            "(x*(y+2)-x*y-2*x)/(exp(x+y)-exp(x)*exp(y))",
            "-46/11",
            "-13/7"
        ),
        "nan"
    );
    assert_eq!(
        at("((x+1)^2-x^2-2*x-1)/(sin(x)^2+cos(x)^2-1)", "2/7", "0"),
        "nan"
    );
}

/// Before: `0` from `simplify`, `ratsimp`, `expand` and `simplify_trig`.
/// The inner denominator `exp(x/2)² − exp(x)` vanishes by the residue
/// test (the numerator over it is `zoo`), the outer `(x − 2)² + (sin²x +
/// cos²x − sin²1 − cos²1)/(e^(x+y) − eˣ·eʸ)` is `0/0` — but its numerator
/// vanishes only by an identity of `sin` and `cos`, which the factor test
/// beside the pole `(e^(x+y) − eˣ·eʸ)⁻¹` did not run, so it was `P/0 = zoo`
/// and the whole `(…)/zoo = 0`.
///
/// Oracle: SymPy 1.14 `(exp(x/2)**2 - exp(x)).equals(0)`, `(exp(x + y) -
/// exp(x)*exp(y)).equals(0)` and `(sin(x)**2 + cos(x)**2 - sin(1)**2 -
/// cos(1)**2).equals(0)` → `True`; at every point `S(0)/S(0)` → `nan`,
/// `(x - 2)**2 + nan` → `nan`.  (SymPy's own `simplify`, `ratsimp` and
/// `expand` give `0`: deliberately different, as in the tests above.)
#[test]
fn an_identity_zero_beside_a_vanishing_denominator_is_nan() {
    let ctx = Context::new();
    let e = ctx
        .parse(
            "((exp(2*x)-exp(x)^2)/((cos(x)+2)+(exp(2*x)+exp(x)^2)/(exp(x/2)^2-exp(x))))\
             /((x-2)^2+((sin(x)^2+cos(x)^2-1)-(sin(1)^2+cos(1)^2-1))/(exp(x+y)-exp(x)*exp(y)))",
        )
        .unwrap();
    assert_eq!(show(&e.simplify()), "nan");
    assert_eq!(show(&e.ratsimp()), "nan");
    assert_eq!(show(&e.expand()), "nan");
    assert_eq!(show(&e.simplify_trig()), "nan");
    let d = "(x-2)^2+((sin(x)^2+cos(x)^2-1)-(sin(1)^2+cos(1)^2-1))/(exp(x+y)-exp(x)*exp(y))";
    let r = ctx.parse(&format!("x/({d})")).unwrap();
    assert_eq!(show(&r.simplify()), "nan");
    assert_eq!(show(&r.ratsimp()), "nan");
    assert_eq!(show(&r.expand()), "nan");
}

/// Before: `0`.  Once something in the expression vanishes — a
/// denominator by the residue test, or a factor — `ratsimp` and `expand`
/// test every denominator that may vanish by an identity of its functions,
/// not only those of a `0/0` candidate product: here `cosh²x − sinh²x − 1`
/// sits in a factor `x − 1 + (x + 2)²/(cosh²x − sinh²x − 1)` that is
/// `zoo`, beside a numerator that is 0: `0·zoo`.  In the third,
/// `tan x·cos x − sin x` was distributed over by `expand`, `xy/0 + 4/0 =
/// nan`, where the value is `0/zoo = 0`.
///
/// Oracle: SymPy 1.14 `(cosh(x)**2 - sinh(x)**2 - 1).equals(0)`, `(exp(3*x)
/// - exp(x)**3).equals(0)`, `(tan(x)*cos(x) - sin(x)).equals(0)` → `True`;
/// `0*zoo` → `nan`, `S(3)/zoo` → `0`.  (SymPy's `ratsimp` gives `0` for the
/// first, its `expand` `0` for the third.)
#[test]
fn identity_denominators_are_tested_once_something_vanishes() {
    let ctx = Context::new();
    let e = ctx
        .parse(
            "((x*(x-3)-x^2+3*x)/(y+(x*y-6)/(exp(3*x)-exp(x)^3)))\
             /((x+5)^2/((x-1)+(x+2)^2/(cosh(x)^2-sinh(x)^2-1)))",
        )
        .unwrap();
    assert_eq!(show(&e.ratsimp()), "nan");
    assert_eq!(show(&e.expand()), "nan");
    let f = ctx
        .parse(
            "(2*x*y-2)/(y+((x+3)^2/(exp(2*x)+exp(x)^2))/(x*(x+4)-x^2-4*x))\
             /((cosh(x)^2-sinh(x)^2-1)/(x+2))",
        )
        .unwrap();
    assert_eq!(show(&f.ratsimp()), "nan");
    assert_eq!(show(&f.expand()), "nan");
    let g = ctx
        .parse(
            "(sqrt(x+7)/((exp(2*x)+exp(x)^2)+(x*y+4)/(tan(x)*cos(x)-sin(x))))\
             /(sqrt(x+7)/(exp(x)^3*exp(y)-exp(3*x+y)))",
        )
        .unwrap();
    assert_eq!(show(&g.expand()), "0");
}

/// Before: `0` from `simplify` and `simplify_trig`.  The denominator
/// `cosh²x − sinh²x − 1` was found to vanish by the identity test, but the
/// factor over it, the constant `sin²1 + cos²1 − 1`, was not tested beside
/// that pole (only beside one the polynomial test had found): `P/0 = zoo`,
/// and `4(y + 2)(x + 5)/(x² + 4 + zoo) = 0`.  It is `0/0`.
///
/// Oracle: SymPy 1.14 `(sin(1)**2 + cos(1)**2 - 1).equals(0)` → `True`,
/// `(cosh(x)**2 - sinh(x)**2 - 1).equals(0)` → `True`; `simplify(…)` →
/// `nan` (its `trigsimp` → `(x + 5)*(4*y + 8)/(x**2 + 4)`).
#[test]
fn a_constant_zero_over_an_identity_pole_is_nan() {
    let ctx = Context::new();
    let e = ctx
        .parse("(y+2)*4*(x+5)/((x^2+4)+(sin(1)^2+cos(1)^2-1)/(cosh(x)^2-sinh(x)^2-1))")
        .unwrap();
    assert_eq!(show(&e.simplify()), "nan");
    assert_eq!(show(&e.simplify_trig()), "nan");
}

/// Before: `cos 0 = 1` and `exp(0) + 0 = 1`.  The argument is `0/0` at
/// every point (`(sin 2x − 2·sin x·cos x)/(e² − exp(1)²)`; `(cosh²x −
/// sinh²x − 1)/(y + 5 + 0/(tan x·cos x − sin x))`), but the test of
/// `simplify` and `simplify_trig` looked only outside the function
/// arguments, and the identities rewrote inside them.
///
/// Oracle: SymPy 1.14 `simplify(cos((y + 4)/(y - 3 + (sin(2*x) -
/// 2*sin(x)*cos(x))/(exp(2) - exp(1)**2))))` → `nan`; `exp(nan)` → `nan`,
/// `(tan(x)*cos(x) - sin(x)).equals(0)` → `True`.
#[test]
fn function_arguments_undefined_everywhere_are_nan() {
    let ctx = Context::new();
    let e = ctx
        .parse("cos((y+4)/((y-3)+(sin(2*x)-2*sin(x)*cos(x))/(exp(2)-exp(1)^2)))")
        .unwrap();
    assert_eq!(show(&e.simplify()), "nan");
    assert_eq!(show(&e.simplify_trig()), "nan");
    let f = ctx
        .parse(
            "exp((cosh(x)^2-sinh(x)^2-1)/((y+5)+(y/(x+1)-y*x/(x^2+x))/(tan(x)*cos(x)-sin(x))))\
             +sqrt((exp(x)^2*exp(y)-exp(2*x+y))^2)",
        )
        .unwrap();
    assert_eq!(show(&f.simplify()), "nan");
    assert_eq!(show(&f.simplify_trig()), "nan");
    // Defined arguments are untouched.
    let p = |s: &str| ctx.parse(s).unwrap();
    assert_eq!(show(&p("cos((sin(x)^2+cos(x)^2-1)/(x+1))").simplify()), "1");
    assert_eq!(
        show(&p("exp(x)*cos(1/x)+sin(1/(x+1))").simplify()),
        "cos(1/x)*exp(x) + sin(1/(x + 1))"
    );
}

/// The forms of correct expressions do not change (the task's list and
/// more): only expressions undefined at every point are affected.
///
/// Oracle: the outputs of symplex 0.33 (`f7ea128`) for the same calls.
#[test]
fn correct_expressions_keep_their_forms() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    assert_eq!(show(&p("(x+1)^2/(x+1)").expand()), "x + 1");
    assert_eq!(
        show(&p("(x^2-1)/(x-1)").expand()),
        "x^2/(x - 1) - 1/(x - 1)"
    );
    assert_eq!(show(&p("1/(x-x)").expand()), "zoo");
    assert_eq!(show(&p("(x^2-1)/(x-1)").simplify()), "x + 1");
    assert_eq!(show(&p("sin(x)^2+cos(x)^2").simplify()), "1");
    assert_eq!(show(&p("1/(sin(x)^2+cos(x)^2-1)").simplify()), "zoo");
    assert_eq!(
        show(&p("1/(ln(x^2)-2*ln(x))").simplify()),
        "1/(-2*ln(x) + ln(x^2))"
    );
    assert_eq!(show(&p("exp(x)^2/exp(2*x)").simplify()), "1");
    assert_eq!(
        show(&p("(exp(1)^2-exp(2)+1)/(sin(1)^2+cos(1)^2)").eval()),
        "1/(sin(1)^2 + cos(1)^2)"
    );
    assert_eq!(
        show(&p("1/(sin(1)^2+cos(1)^2-1)").eval()),
        "1/(sin(1)^2 + cos(1)^2 - 1)"
    );
}
