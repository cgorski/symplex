//! The parser after 0.30.

use symplex::prelude::*;

/// `asec`, `acsc`, `acoth`, `asech`, `acsch` (and the `arc…` spellings)
/// parse to SymPy's principal definitions; before 0.31 they were unknown
/// functions (the evalf hunter's generator could not use them).
///
/// SymPy: `asec(x).rewrite(acos)` → `acos(1/x)`, `acsc(x).rewrite(asin)` →
/// `asin(1/x)`, `acoth(x).rewrite(atanh)` → `atanh(1/x)`,
/// `asech(x).rewrite(acosh)` → `acosh(1/x)`, `acsch(x).rewrite(asinh)` →
/// `asinh(1/x)`; `N(asec(3))` → `1.23095941734077`.
#[test]
fn inverse_reciprocal_functions_parse() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inv = &ctx.int(1) / &x;
    for (name, want) in [
        ("asec", inv.acos()),
        ("arcsec", inv.acos()),
        ("acsc", inv.asin()),
        ("acoth", inv.atanh()),
        ("asech", inv.acosh()),
        ("acsch", inv.asinh()),
    ] {
        assert_eq!(ctx.parse(&format!("{name}(x)")).unwrap(), want, "{name}");
    }
    let v = ctx.parse("asec(3)").unwrap().eval_f64().unwrap();
    assert!((v - 1.230_959_417_340_774_7).abs() < 1e-15, "{v}");
}
