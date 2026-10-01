//! Undefined expressions (`0/0`) that `expand` or the canonical
//! constructors turned into a value: a numerator and a denominator that
//! vanish identically only once multiplied out or once their fractions
//! are combined, and `zoo` distributed over a sum.
//!
//! The oracle is the value at a point in exact rational arithmetic (the
//! source with the symbols replaced by rationals, parsed), where `0/0` is
//! `nan` and `P/0` is `zoo`; SymPy 1.14 agrees at the point (it gives `0`
//! for some of the symbolic forms: its `expand` multiplies out before it
//! divides, and symplex's `simplify`/`together`/`ratsimp` give `nan` for
//! `0/0` since 0.32).

use symplex::prelude::*;

/// `x·(x + 1) − x² − x`, 0 once multiplied out.
const N: &str = "(x*(x+1)-x^2-x)";
/// `−x/(x + 1) + x·(−x/(x + 1) + 1)`, 0 once its fractions are combined.
const B: &str = "(-x/(x+1)+x*(-x/(x+1)+1))";
/// `(x + y)² − x² − 2xy − y²`.
const D: &str = "((x+y)^2-x^2-2*x*y-y^2)";

fn expand(src: &str) -> String {
    let ctx = Context::new();
    format!("{}", ctx.parse(src).unwrap().expand())
}

fn parse(src: &str) -> String {
    let ctx = Context::new();
    format!("{}", ctx.parse(src).unwrap())
}

/// The bug-hunt reproducers: up to 0.32 `expand((x·(x + 1) − x² −
/// x)/(−x/(x + 1) + x·(−x/(x + 1) + 1)))` and `expand(((x + 3)² − x² − 6x −
/// 9)/(x/(x + 5) + 5/(x + 5) − 1))` were `0` — the numerator was multiplied
/// out to 0 and `0·(…)⁻¹ = 0` before the denominator, 0 once its fractions
/// are combined, was looked at.  Both are `0/0` at every point.
///
/// Oracle: SymPy `(N/B).subs(x, Rational(2, 7))` → `nan`,
/// `together(N/B)` → `nan`, the second `.subs(x, Rational(2, 7))` → `nan`
/// (SymPy's `expand` gives `0` for both: deliberately different, as
/// symplex's `simplify` and `together`, which already gave `nan`).
#[test]
fn expand_of_zero_over_vanishing_denominator_is_nan() {
    assert_eq!(expand(&format!("{N}/{B}")), "nan");
    assert_eq!(expand("((x+3)^2-x^2-6*x-9)/(x/(x+5)+5/(x+5)-1)"), "nan");
    let ctx = Context::new();
    let e = ctx.parse(&format!("{N}/{B}")).unwrap();
    assert_eq!(format!("{}", e.simplify()), "nan");
    assert_eq!(format!("{}", e.together()), "nan");
}

/// The third reproducer: `expand(|x − 3|/(cos x + 2 + (x + 3)/((x + y)² −
/// x² − 2xy − y²)))` was `nan` — the inner `(x + 3)/0 = zoo` was distributed
/// (`x·zoo + 3·zoo = zoo + zoo = nan`).  For `x ≠ −3` the denominator is
/// `zoo` and the quotient `0`.
///
/// Oracle: SymPy `simplify(...)` → `0`, `.subs({x: Rational(2, 7), y:
/// Rational(3, 11)})` → `0` (SymPy's `expand` gives `Abs(x - 3)/(zoo*x +
/// cos(x) + zoo)`).
#[test]
fn expand_does_not_spread_zoo_over_a_sum() {
    assert_eq!(expand(&format!("abs(x-3)/(cos(x)+2+(x+3)/{D})")), "0");
    assert_eq!(expand(&format!("(x+3)/{D}")), "zoo");
    assert_eq!(expand(&format!("(x+1)*(y+2)/{D}")), "zoo");
}

/// A directed infinity is not distributed either: `expand((x + 1)·∞)` was
/// `x·∞ + ∞`, which is `−∞ + ∞ = nan` at `x = −1/2` where `(x + 1)·∞ = ∞`.
///
/// Oracle: SymPy `((x+1)*oo).subs(x, -Rational(1, 2))` → `oo`, while its
/// `expand((x+1)*oo)` → `oo*x + oo` and that at `x = −1/2` → `nan`.
#[test]
fn expand_does_not_distribute_an_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.parse("-1/2").unwrap();
    for src in ["(x+1)*oo", "(x+1)*(-oo)"] {
        let e = ctx.parse(src).unwrap();
        let expanded = e.expand();
        assert_eq!(format!("{expanded}"), format!("{e}"), "{src}");
        assert_eq!(
            format!("{}", expanded.subs(&x, &half)),
            format!("{}", e.subs(&x, &half)),
            "{src}"
        );
    }
    assert_eq!(expand("(x+1)*oo"), "(x + 1)*oo");
    assert_eq!(expand("(x+1)*(-oo)"), "-(x + 1)*oo");
}

/// Bug 5: the canonical `P·0⁻¹ = zoo` took `P = x·(x + 1) − x² − x`, which is
/// 0 for every `x`, as a generic factor: `(x·(x + 1) − x² − x)/0` was `zoo`
/// (now `nan`, `0/0`), and so was `zoo·P`; `P/P` was `1` and `0/((x + y)² −
/// x² − 2xy − y²)` was `0` (both `0/0`).
///
/// Oracle: SymPy `(N/0).subs(x, Rational(2, 7))` → `nan` (SymPy keeps
/// `zoo*(-x**2 + x*(x + 1) - x)` symbolically); `S(0)/0` → `nan`.  SymPy
/// gives `1` for `N/N` and `0` for `0/D` symbolically (its generic
/// cancellation), `nan` at every point.
#[test]
fn canonical_product_with_identically_zero_sum() {
    assert_eq!(parse(&format!("{N}/0")), "nan");
    assert_eq!(parse(&format!("zoo*{N}")), "nan");
    assert_eq!(parse(&format!("{N}^2/0")), "nan");
    assert_eq!(parse("((x+1)^2-x^2-2*x-1)*zoo"), "nan");
    assert_eq!(parse(&format!("{B}/0")), "nan");
    assert_eq!(parse(&format!("abs({N})/0")), "nan");
    assert_eq!(parse(&format!("{N}/{N}")), "nan");
    assert_eq!(parse(&format!("{B}*(x+2)/{B}")), "nan");
    assert_eq!(parse(&format!("0/{D}")), "nan");
    assert_eq!(parse(&format!("zoo + 2/{D}")), "nan");
    // A sum whose own denominator vanishes is `zoo`, not 0: `zoo·(1/D + 2)`
    // stays `zoo`, `0/(1/D + 2) = 0/zoo = 0`.
    assert_eq!(parse(&format!("zoo*(1/{D} + 2)")), "zoo");
    assert_eq!(parse(&format!("0/(1/{D} + 2)")), "0");
}

/// `subs` builds through the same constructors: substituting `x = 11/7` into
/// `(x·(y + 3) − xy − 3x)·((cosh²x − sinh²x − 1)/((x + y)² − x² − 2xy − y²) +
/// cos x + 2)/x` made the first factor `0` while the denominator stayed a sum
/// in `y` that vanishes identically, and the product was `0`; it is `0/0`
/// at every point.
///
/// Oracle: the exact value at `x = 11/7, y = −27/13`: the first factor and
/// the denominator are both 0, `0·(c/0 + …) = 0·zoo = nan` (SymPy: `S(0) *
/// (S(5)/0 + 2)` → `nan`; SymPy's own sequential `subs` gives `0`).
#[test]
fn subs_into_zero_over_vanishing_denominator_is_nan() {
    let ctx = Context::new();
    let e = ctx
        .parse(&format!(
            "(x*(y+3)-x*y-3*x)*((cosh(x)^2 - sinh(x)^2 - 1)/{D} + cos(x) + 2)/x"
        ))
        .unwrap();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let at = e
        .subs(&x, &ctx.parse("11/7").unwrap())
        .subs(&y, &ctx.parse("-27/13").unwrap());
    assert_eq!(format!("{at}"), "nan");
    let exact = ctx
        .parse(
            "((11/7)*((-27/13)+3)-(11/7)*(-27/13)-3*(11/7))*((cosh(11/7)^2 - sinh(11/7)^2 - 1)\
             /(((11/7)+(-27/13))^2-(11/7)^2-2*(11/7)*(-27/13)-(-27/13)^2) + cos(11/7) + 2)/(11/7)",
        )
        .unwrap();
    assert_eq!(format!("{exact}"), "nan");
}

/// Vanishing denominators inside function arguments and under roots, and
/// zeros absorbed by a further division: each expansion now agrees with the
/// value at every point.
///
/// Oracle (SymPy, at `x = 2/7`): `sin(e2)` → `nan`, `N/sqrt(x/(x+5) +
/// 5/(x+5) - 1)` → `nan`, `x/(1 + 1/N)` → `0`, `(x + 1)/N` → `zoo`,
/// `x/N + 1/N` → `nan` (SymPy's `expand` gives `0`, `0`, `0`, `zoo*x +
/// zoo`, `zoo*x + zoo`).
#[test]
fn expand_agrees_with_the_value_at_every_point() {
    let cases = [
        (format!("sin({N}/(x/(x+5)+5/(x+5)-1))"), "nan"),
        (format!("{N}/sqrt(x/(x+5)+5/(x+5)-1)"), "nan"),
        (format!("{N}/sin({B})"), "nan"),
        (format!("{D}/abs(x/(x+5)+5/(x+5)-1)"), "nan"),
        (format!("x/(1+1/{N})"), "0"),
        (format!("(x+1)^2 + y/(1+1/{N})"), "x^2 + 2*x + 1"),
        (format!("(x+1)/{N}"), "zoo"),
        (format!("1/{B}"), "zoo"),
        (format!("x/{N} + 1/{N}"), "nan"),
        // Hunter cases (seeds 9 and 63 of `zz_hunt_undef`): `0` before.
        (
            "abs(x-3)/(y/(x-4)-y*x/(x^2-4*x))*((x-2)^2-x^2+4*x-4)*((x+1)^2-x^2-2*x-1)".to_string(),
            "nan",
        ),
        (format!("1/(exp(x)/{B}+(x^2+16)/(x*(x+2)-x^2-2*x))"), "nan"),
    ];
    for (src, want) in &cases {
        assert_eq!(expand(src), *want, "expand({src})");
    }
}

/// With `deep = false` the expression's own sums, products and powers are
/// decided as with `deep = true`, and function arguments are left alone,
/// as the rest of `expand_with` leaves them (`sin(0/0)` is not expanded,
/// so not turned into a value either).
///
/// Oracle: as `expand_of_zero_over_vanishing_denominator_is_nan`.
#[test]
fn expand_without_deep_decides_its_own_skeleton() {
    use symplex::macros::ExpandOpts;
    let ctx = Context::new();
    let shallow = ExpandOpts::default().deep(false);
    let e = ctx.parse(&format!("{N}/{B}")).unwrap();
    assert_eq!(format!("{}", e.expand_with(&shallow)), "nan");
    let f = ctx.parse(&format!("(x+1)*sin({N}/{B})")).unwrap();
    let g = f.expand_with(&shallow);
    assert_eq!(
        format!("{g}"),
        "x*sin((-x^2 - x + x*(x + 1))/(-x/(x + 1) + x*(-x/(x + 1) + 1))) + \
         sin((-x^2 - x + x*(x + 1))/(-x/(x + 1) + x*(-x/(x + 1) + 1)))"
    );
    assert_eq!(format!("{}", f.expand()), "nan");
}

/// Correct expansions are unchanged (SymPy 1.14 `expand`: `x + 1`, `zoo`,
/// `x**2/(x - 1) - 1/(x - 1)`, `0`; `x/0` and `(x − x)/0` are `zoo` and
/// `nan` when built), and so are cancellations of sums that do not vanish.
#[test]
fn ordinary_expansions_unchanged() {
    assert_eq!(expand("(x+1)^2/(x+1)"), "x + 1");
    assert_eq!(expand("1/(x - x)"), "zoo");
    assert_eq!(expand("(x^2-1)/(x-1)"), "x^2/(x - 1) - 1/(x - 1)");
    assert_eq!(expand("0/x"), "0");
    assert_eq!(expand("x/0"), "zoo");
    assert_eq!(expand("(x-x)/0"), "nan");
    assert_eq!(
        expand("(x+3)^2/(x+1)"),
        "x^2/(x + 1) + 6*x/(x + 1) + 9/(x + 1)"
    );
    assert_eq!(
        expand("(sin(x)^2+cos(x)^2-1)/(x+1)"),
        "sin(x)^2/(x + 1) + cos(x)^2/(x + 1) - 1/(x + 1)"
    );
    assert_eq!(parse("(x+1)/(x+1)"), "1");
    assert_eq!(parse("(x+1)^2/(x+1)"), "x + 1");
    assert_eq!(parse("(x+1)*zoo"), "zoo");
    assert_eq!(parse("(sin(x)^2+cos(x)^2-1)/0"), "zoo");
    assert_eq!(parse("0/(x+1)"), "0");
}
