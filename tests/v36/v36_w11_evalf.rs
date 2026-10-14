//! Tests of the 0.50 refactor (w11): evalf's ball type and the BigFloat boundary.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn decimal(src: &str, digits: u32) -> Result<String, SymplexError> {
    let ctx = Context::new();
    ctx.parse(src).unwrap().eval_decimal(digits)
}

/// `δ(x)` is a distribution: 0 for every `x ≠ 0`, with no value at 0.  Up
/// to 0.49 `evalf` returned a certified `0` for `DiracDelta(0)`.
///
/// Oracle (the definition; SymPy: `DiracDelta(0).evalf()` stays
/// `DiracDelta(0)`, no number): refused now.
#[test]
fn dirac_delta_at_zero_has_no_value() {
    let err = decimal("DiracDelta(0)", 30).unwrap_err();
    assert!(err.to_string().contains("DiracDelta(0)"), "{err}");
    // Off 0 it is exactly 0 (`δ(x) = 0`, `x ≠ 0`).
    assert_eq!(decimal("DiracDelta(2)", 30).unwrap(), "0");
    assert_eq!(decimal("DiracDelta(-1/10^60)", 30).unwrap(), "0");
    assert_eq!(decimal("1 + DiracDelta(3)", 30).unwrap(), "1");
    // Off the reals `δ` has no value either: this line pinned `0` up to
    // 0.50 (SymPy 1.14 rejects `DiracDelta(I)`; `is_zero` is `None`), see
    // `v36_s14d_assume::dirac_delta_off_the_reals_has_no_value`.
    assert!(decimal("DiracDelta(I)", 30).is_err());
}

/// An argument that is 0 only to the precision reached is no proof that
/// `δ` is 0: up to 0.49 `DiracDelta(sin(1)^2 + cos(1)^2 - 1)` was `0` and
/// `1 + DiracDelta(sin(1)^2 + cos(1)^2 - 1)` was `1`.
///
/// Oracle (exact identity): `sin²1 + cos²1 − 1 = 0`, so both are `δ(0)`,
/// which has no value; refused now (the argument's ball always contains 0).
#[test]
fn dirac_delta_of_a_hidden_zero_is_refused() {
    assert!(decimal("DiracDelta(sin(1)^2 + cos(1)^2 - 1)", 30).is_err());
    assert!(decimal("1 + DiracDelta(sin(1)^2 + cos(1)^2 - 1)", 30).is_err());
}

/// Every decimal result is printed through `base::bigfloat::to_decimal_digits`,
/// which up to 0.49 was astro-float's `convert_to_radix` (its exponent was
/// wrong at 1024 bits, astro-float #33).  It is now an exact conversion
/// from the binary mantissa; beyond a decimal scale of `2¹⁶` it works
/// between two bounds of `5^|s|`.  The digits are unchanged (it agreed with
/// `convert_to_radix` on 10⁶ random values); these values exercise that
/// route.
///
/// Oracle: mpmath, `mp.dps = 60; nstr(exp(10**7), 25)` =
/// `6.592232534618439489560886e+4342944`, `nstr(exp(-10**7), 25)` =
/// `1.516936780898734357589464e-4342945`, `nstr(factorial(10**5), 25)` =
/// `2.824229407960347874293422e+456573`.
#[test]
fn decimal_digits_at_extreme_exponents_match_mpmath() {
    assert_eq!(
        decimal("exp(10^7)", 25).unwrap(),
        "6.592232534618439489560886e4342944"
    );
    assert_eq!(
        decimal("exp(-10^7)", 25).unwrap(),
        "1.516936780898734357589464e-4342945"
    );
    assert_eq!(
        decimal("factorial(10^5)", 25).unwrap(),
        "2.824229407960347874293422e456573"
    );
}

/// 283 digits are printed from a 1024-bit mantissa (`3.4·283 + 64` bits,
/// rounded up to whole words), the precision of astro-float #33; up to
/// 0.49 through `convert_to_radix`, now through the exact conversion.
///
/// Oracle: mpmath, `mp.dps = 320; nstr(pi, 283)` and `nstr(sqrt(2), 283)`:
/// they begin `3.141592653589793238`, `1.414213562373095048` and end
/// `…60348610454326648213`, `…84308714321450839763`.
#[test]
fn decimal_digits_from_a_1024_bit_mantissa_match_mpmath() {
    for (src, head, tail) in [
        ("pi", "3.141592653589793238", "60348610454326648213"),
        ("sqrt(2)", "1.414213562373095048", "84308714321450839763"),
    ] {
        let s = decimal(src, 283).unwrap();
        assert_eq!(s.len(), 284, "{src}: {s}");
        assert!(s.starts_with(head) && s.ends_with(tail), "{src}: {s}");
    }
}
