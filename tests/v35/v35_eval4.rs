//! Exact decisions of relations between constants built from trigonometric
//! values at rational multiples of π, roots of unity and radicals (the
//! eval hunt after 0.44, agent E).  `eval` and Boolean `simplify` decide an
//! equality of such constants exactly, in the cyclotomic field `ℚ(ζ_N)`
//! with the radicals adjoined (`src/transforms/eval/cyclotomic.rs`).
//! Every reference value cites its oracle: SymPy 1.14 for the decisions,
//! mpmath 1.3 at 50 digits for the values.

use symplex::prelude::*;

fn decide(ctx: &Context, s: &str) -> (String, String) {
    let b = ctx
        .parse_bool(s)
        .unwrap_or_else(|e| panic!("`{s}` does not parse: {e}"));
    (b.eval().to_string(), b.simplify().to_string())
}

fn assert_decided(ctx: &Context, s: &str, want: &str) {
    let (e, si) = decide(ctx, s);
    assert_eq!(e, want, "eval of {s}");
    assert_eq!(si, want, "simplify of {s}");
}

/// Before (0.44): `cos(π/16)`, `sin(π/32)` stay unevaluated (SymPy's table
/// stops at the denominators of `cos.eval`), and a relation with their
/// nested radicals stayed a relation under `eval` and `simplify`:
/// `Eq(cos(pi/16), sqrt(2 + sqrt(2 + sqrt(2)))/2)` → itself.
///
/// Oracle: SymPy 1.14 `Eq(cos(pi/16), sqrt(2 + sqrt(2 + sqrt(2)))/2).simplify()`
/// → `True`, likewise `Eq(cos(pi/16)**2, (2 + sqrt(2 + sqrt(2)))/4)`,
/// `Eq(sin(pi/32), sqrt(2 - sqrt(2 + sqrt(2 + sqrt(2))))/2)`,
/// `Eq(cos(pi/32), sqrt(2 + sqrt(2 + sqrt(2 + sqrt(2))))/2)` → `True`;
/// `(cos(pi/16) >= …).simplify()` → `True`, `(cos(pi/16) > …)` → `False`,
/// `Ne(cos(pi/16), …)` → `False`; mpmath `cos(pi/16)` =
/// `0.98078528040323044912618223613423903697393373089334`, equal to
/// `sqrt(2 + sqrt(2 + sqrt(2)))/2` to 50 digits (mpmath at 70 digits; up to
/// 0.46 the value was quoted wrong from its 30th digit on).
#[test]
fn trigonometric_values_against_nested_radicals() {
    let ctx = Context::new();
    for s in [
        "Eq(cos(pi/16), sqrt(2 + sqrt(2 + sqrt(2)))/2)",
        "Eq(cos(pi/16)^2, (2 + sqrt(2 + sqrt(2)))/4)",
        "Eq(sin(pi/32), sqrt(2 - sqrt(2 + sqrt(2 + sqrt(2))))/2)",
        "Eq(cos(pi/32), sqrt(2 + sqrt(2 + sqrt(2 + sqrt(2))))/2)",
        "cos(pi/16) >= sqrt(2 + sqrt(2 + sqrt(2)))/2",
        "cos(pi/16) <= sqrt(2 + sqrt(2 + sqrt(2)))/2",
    ] {
        assert_decided(&ctx, s, "True");
    }
    for s in [
        "cos(pi/16) > sqrt(2 + sqrt(2 + sqrt(2)))/2",
        "Ne(cos(pi/16), sqrt(2 + sqrt(2 + sqrt(2)))/2)",
    ] {
        assert_decided(&ctx, s, "False");
    }
}

/// Before (0.44): identities between values at the heptagonal, nonagonal
/// and 13-gonal angles stayed relations under `eval`
/// (`Eq(cos(pi/7)*cos(2*pi/7)*cos(4*pi/7), -1/8)` → itself); they hold in
/// `ℚ(ζ_N)` coordinate by coordinate.
///
/// Oracle: SymPy 1.14 `.simplify()` of each relation below → `True`, except
/// the last, which SymPy's `simplify` does not finish in 600 s (up to 0.46
/// this claimed `True` for it too); that one is the product-to-sum formula
/// `cos A·cos B = (cos(A − B) + cos(A + B))/2` with `A − B = −43π/187`, `A + B
/// = 485π/187`, and 0 to 50 digits in mpmath;
/// mpmath `cos(pi/7)*cos(2*pi/7)*cos(4*pi/7)` = `-0.125` (to 50 digits),
/// `sin(pi/7)*sin(2*pi/7)*sin(3*pi/7) - sqrt(7)/8` = `0`,
/// `tan(pi/9)*tan(2*pi/9)*tan(4*pi/9) - sqrt(3)` = `0`,
/// `cos(2*pi/13) + cos(6*pi/13) + cos(8*pi/13)` =
/// `0.65138781886599732327980531686762398656282414346131` = `(sqrt(13) - 1)/4`
/// (mpmath at 70 digits; up to 0.46 the value was quoted wrong from its 16th
/// digit on).
#[test]
fn identities_between_values_at_the_same_denominator() {
    let ctx = Context::new();
    for s in [
        "Eq(cos(pi/7)*cos(2*pi/7)*cos(4*pi/7), -1/8)",
        "Eq(cos(pi/7) - cos(2*pi/7) + cos(3*pi/7), 1/2)",
        "Eq(sin(pi/7)*sin(2*pi/7)*sin(3*pi/7), sqrt(7)/8)",
        "Eq(tan(pi/9)*tan(2*pi/9)*tan(4*pi/9), sqrt(3))",
        "Eq(cos(2*pi/13) + cos(6*pi/13) + cos(8*pi/13), (sqrt(13) - 1)/4)",
        "Eq(cos(3*pi/7), 4*cos(pi/7)^3 - 3*cos(pi/7))",
        "Eq(1/cos(pi/7) + 1/cos(3*pi/7) + 1/cos(5*pi/7), 4)",
        "Eq(cos(13*pi/11)*cos(24*pi/17), (cos(43*pi/187) + cos(485*pi/187))/2)",
    ] {
        assert_decided(&ctx, s, "True");
    }
    // Every true relation is decided, so its negation is too.
    assert_decided(&ctx, "Ne(cos(pi/7)*cos(2*pi/7)*cos(4*pi/7), -1/8)", "False");
    assert_decided(&ctx, "cos(pi/7)*cos(2*pi/7)*cos(4*pi/7) < -1/8", "False");
    assert_decided(&ctx, "cos(pi/7)*cos(2*pi/7)*cos(4*pi/7) <= -1/8", "True");
}

/// Before (0.44): a radical of an unevaluated value
/// (`√((1 + cos(π/7))/2) = cos(π/14)`) and Cardano's formula for a root of
/// `x³ + x² − 2x − 1` (`2cos(2π/7)`, complex cube roots) left the
/// relation undecided.  Now the radicals are generators over `ℚ(ζ_N)` and
/// the norm decides.
///
/// Oracle: SymPy 1.14 `Eq(cos(pi/14), sqrt((1 + cos(pi/7))/2)).simplify()`
/// → `True`; `Eq(2*cos(2*pi/7), -1/3 + 7/(9*(7/54 + 7*sqrt(3)*I/18)**(1/3)) +
/// (7/54 + 7*sqrt(3)*I/18)**(1/3)).simplify()` → `True` (the root
/// `roots(x**3 + x**2 - 2*x - 1, cubics=True)` pairs with `2*cos(2*pi/7)`);
/// mpmath `2*cos(2*pi/7)` = `1.2469796037174670610500097680084796212645494617928`
/// (mpmath at 70 digits; up to 0.46 the value was quoted wrong from its 43rd
/// digit on).
#[test]
fn radicals_over_unevaluated_values() {
    let ctx = Context::new();
    for s in [
        "Eq(cos(pi/14), sqrt((1 + cos(pi/7))/2))",
        "Eq(sin(3*pi/22), sqrt((1 - cos(3*pi/11))/2))",
        "Eq(2*cos(2*pi/7), -1/3 + 7/(9*(7/54 + 7*sqrt(3)*I/18)^(1/3)) + (7/54 + 7*sqrt(3)*I/18)^(1/3))",
    ] {
        assert_decided(&ctx, s, "True");
    }
    assert_decided(&ctx, "Eq(cos(pi/14), -sqrt((1 + cos(pi/7))/2))", "False");
}

/// A difference below the precision of the first evaluation is still
/// decided the right way: the exact test says "not zero", the deeper
/// certified evaluation the sign.  (Nothing changed here; this pins that the
/// new test does not take a tiny difference for a zero.)
///
/// Oracle: mpmath `cos(pi/16) - sqrt(2 + sqrt(2 + sqrt(2)))/2 - 10**-25` =
/// `-1.0e-25`; SymPy 1.14 `Eq(cos(pi/16), sqrt(2 + sqrt(2 + sqrt(2)))/2 +
/// 10**-25).simplify()` → `False`.
#[test]
fn near_misses_are_not_identities() {
    let ctx = Context::new();
    assert_decided(
        &ctx,
        "Eq(cos(pi/16), sqrt(2 + sqrt(2 + sqrt(2)))/2 + 1/10^25)",
        "False",
    );
    assert_decided(
        &ctx,
        "cos(pi/16) < sqrt(2 + sqrt(2 + sqrt(2)))/2 + 1/10^25",
        "True",
    );
    assert_decided(
        &ctx,
        "Eq(cos(pi/7)*cos(2*pi/7)*cos(4*pi/7), -1/8 + 1/10^30)",
        "False",
    );
}

/// Before (0.44): `sign` of a constant that is exactly 0 without being
/// written 0 stayed (`sign(cos(π/16) − √(2 + √(2 + √2))/2)`): its value is
/// 0 to any precision, so no evaluation certifies a sign.  The exact test
/// decides the zero now; a nonzero difference keeps its certified sign.
///
/// Oracle: mpmath at 50 digits `cos(pi/16) - sqrt(2 + sqrt(2 + sqrt(2)))/2`
/// = `0.0`, `cos(pi/7)*cos(2*pi/7)*cos(4*pi/7) + 1/8` = `0.0`; SymPy 1.14
/// leaves both `sign(...)` unevaluated (also after `simplify`).
#[test]
fn sign_of_an_exact_zero() {
    let ctx = Context::new();
    for s in [
        "sign(cos(pi/16) - sqrt(2 + sqrt(2 + sqrt(2)))/2)",
        "sign(cos(pi/7)*cos(2*pi/7)*cos(4*pi/7) + 1/8)",
    ] {
        assert_eq!(ctx.parse(s).unwrap().eval().to_string(), "0", "{s}");
    }
    assert_eq!(
        ctx.parse("sign(cos(pi/16) - sqrt(2 + sqrt(2 + sqrt(2)))/2 + 1/10^30)")
            .unwrap()
            .eval()
            .to_string(),
        "1"
    );
}

/// Boolean `simplify` folds a decided relation inside a formula (it shares
/// `eval`'s decision since 0.44).
///
/// Oracle: SymPy 1.14 `(Eq(cos(pi/7)*cos(2*pi/7)*cos(4*pi/7), -S(1)/8) &
/// (x > 0)).simplify()` → `x > 0`, `(Ne(cos(pi/16), sqrt(2 + sqrt(2 +
/// sqrt(2)))/2) | (x > 0)).simplify()` → `x > 0`.
#[test]
fn decided_relations_fold_in_boolean_formulas() {
    let ctx = Context::new();
    ctx.symbol("x");
    for s in [
        "Eq(cos(pi/7)*cos(2*pi/7)*cos(4*pi/7), -1/8) & (x > 0)",
        "Ne(cos(pi/16), sqrt(2 + sqrt(2 + sqrt(2)))/2) | (x > 0)",
    ] {
        let b = ctx.parse_bool(s).unwrap();
        assert_eq!(b.simplify().to_string(), "x > 0", "{s}");
    }
}
