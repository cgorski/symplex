//! Consistency of the decisions about constants (the hunt after 0.45,
//! agent A): `eval` of `Eq`/`Ne`/orders, Boolean `simplify`, `Ex::equals`,
//! the assumption queries `is_*`, `sign` and `solve` answer alike for the
//! same constant.  Every reference value cites its oracle: SymPy 1.14 for
//! its decisions, mpmath 1.3 at 50 digits for the values.

use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s)
        .unwrap_or_else(|e| panic!("`{s}` does not parse: {e}"))
}

fn rel(ctx: &Context, s: &str) -> String {
    ctx.parse_bool(s)
        .unwrap_or_else(|e| panic!("`{s}` does not parse: {e}"))
        .eval()
        .to_string()
}

/// Before (0.45): `Ex::equals` never asked `eval`'s decision of `Eq`:
/// `(cos(π/16)² − (2 + √(2 + √2))/4).equals(0)` and `cos(π/16).equals(√(2 +
/// √(2 + √2))/2)` were `None` while `Eq(cos(π/16), √(2 + √(2 + √2))/2)`
/// evaluated to `True` (the hunter: 182 equal and 66 unequal pairs
/// undecided by `equals` that `eval` decided).
///
/// Oracle: SymPy 1.14 `(cos(pi/16)**2 - (2 + sqrt(2 + sqrt(2)))/4).equals(0)`
/// → `True`; `Eq(cos(pi/16), sqrt(2 + sqrt(2 + sqrt(2)))/2).simplify()` →
/// `True`; mpmath `atan(1/5)*4 - atan(1/239) - pi/4` = `1.3e-51` at 50
/// digits (Machin's formula, an identity); `cos(pi/16)` −
/// `sqrt(2 + sqrt(2 + sqrt(2)))/2 - 1/10**20` = `1.0e-20`.
#[test]
fn equals_agrees_with_eval_of_eq() {
    let ctx = Context::new();
    let d = p(&ctx, "cos(pi/16)^2 - (2 + sqrt(2 + sqrt(2)))/4");
    assert_eq!(d.equals(&ctx.zero()), Some(true));
    let c = p(&ctx, "cos(pi/16)");
    assert_eq!(
        c.equals(&p(&ctx, "sqrt(2 + sqrt(2 + sqrt(2)))/2")),
        Some(true)
    );
    assert_eq!(
        c.equals(&p(&ctx, "sqrt(2 + sqrt(2 + sqrt(2)))/2 - 1/10^20")),
        Some(false)
    );
    let machin = p(&ctx, "4*atan(1/5) - atan(1/239)");
    assert_eq!(machin.equals(&p(&ctx, "pi/4")), Some(true));
}

/// Before (0.45): the assumption queries had no route through the value of
/// a constant: `cos(π/16).is_positive()`, `(cos(π/7) − 9/10).is_positive()`
/// were `None` (461 "partly undecided" property results in the 0.45
/// hunter), and the zero difference `cos(π/16) − √(2 + √(2 + √2))/2` was not
/// known to be zero although `sign` gave `0`.  A query decides a constant
/// as `eval` decides `x > 0`, `Eq(x, 0)` and realness.
///
/// Oracle: SymPy 1.14 `cos(pi/16).is_positive` → `True`, `(cos(pi/7) -
/// Rational(9, 10)).is_positive` → `True`; mpmath `cos(pi/7) - 0.9` =
/// `9.6886790241912623610231950744505116591916213186e-4`;
/// `cbrt(2 + sqrt(5)) + root(2 - sqrt(5), 3)` (principal roots) =
/// `1.9270509831248422723… + 0.5352331346596348979…i`, not real.
#[test]
fn queries_of_constants_agree_with_eval() {
    let ctx = Context::new();
    let c = p(&ctx, "cos(pi/16)");
    assert_eq!(c.is_positive(), Some(true));
    assert_eq!(c.is_negative(), Some(false));
    let d = p(&ctx, "cos(pi/7) - 9/10");
    assert_eq!(d.is_positive(), Some(true));
    assert_eq!(d.is_nonpositive(), Some(false));
    assert_eq!(d.is_zero(), Some(false));
    assert_eq!(rel(&ctx, "cos(pi/7) > 9/10"), "True");
    let z = p(&ctx, "cos(pi/16) - sqrt(2 + sqrt(2 + sqrt(2)))/2");
    assert_eq!(z.is_zero(), Some(true));
    assert_eq!(z.is_positive(), Some(false));
    assert_eq!(z.is_nonnegative(), Some(true));
    assert_eq!(z.sign().eval().to_string(), "0");
    let w = p(&ctx, "(2 + sqrt(5))^(1/3) + (2 - sqrt(5))^(1/3)");
    assert_eq!(w.is_real(), Some(false));
    assert_eq!(w.is_positive(), Some(false));
    assert_eq!(w.is_zero(), Some(false));
}

/// Before (0.45): `W(w·e^w) = w` was recognised only for a rational `w` in
/// `w·e^w` and for `w = t·ln p` with integers `t`, `p`: `lambertw(√2·ln(2)/2)`
/// (which is `ln(2)/2`), `lambertw(−2·ln(5/2)/5)` (`−ln(5/2)`, on `W₀` since
/// `−ln(5/2) ≥ −1`), `lambertw(√3·e^{√3})` and `W₋₁(−3√2·ln(2)/8)` (`−3·ln(2)/2
/// ≤ −1`) stayed, values no evaluation tells from those numbers.  The
/// branch is the sign of `w + 1`: `W₋₁(−2·ln(5/2)/5)` is not `−ln(5/2)` and
/// stays.
///
/// Oracle: mpmath `lambertw(sqrt(2)*log(2)/2)` =
/// `0.34657359027997265470861606072908828403775006718013` = `log(2)/2`;
/// `lambertw(-2*log(2.5)/5)` = `-0.91629073187415506518352721176801107145010121990826`
/// = `-log(2.5)`; `lambertw(-2*log(2.5)/5, -1)` =
/// `-1.0886585980620697675791452462780964256695223260284`;
/// `lambertw(-3*sqrt(2)/8*log(2), -1)` =
/// `-1.0397207708399179641258481821872648521132502015404` = `-3*log(2)/2`.
/// SymPy 1.14 leaves `LambertW(sqrt(2)*log(2)/2)` unevaluated.
#[test]
fn lambert_w_of_w_exp_w() {
    let ctx = Context::new();
    let cases = [
        ("lambertw(sqrt(2)*log(2)/2)", "1/2*ln(2)"),
        ("lambertw(-2*log(5/2)/5)", "-ln(5/2)"),
        ("lambertw(sqrt(3)*exp(sqrt(3)))", "sqrt(3)"),
        ("lambertw(-3*sqrt(2)*log(2)/8, -1)", "-3/2*ln(2)"),
    ];
    for (s, want) in cases {
        assert_eq!(p(&ctx, s).eval().to_string(), want, "{s}");
    }
    let other = p(&ctx, "lambertw(-2*log(5/2)/5, -1)").eval();
    assert!(other.to_string().starts_with("W("), "{other}");
    let v = other.eval_decimal(30).expect("evaluates");
    assert!(v.starts_with("-1.0886585980620697675791452462"), "{v}");
}

/// Before (0.45): the realness of `W₋₁(a)` for `−1/e ≤ a < 0` was not known,
/// nor the finiteness of the `Apply` node, so `sign(W₋₁(−ln(2)/(5·2^{1/5})) +
/// ln(2)/5)` stayed while `eval` decided `Ne` (the value is −3.17…), and
/// `is_real`/`is_negative` were `None`.
///
/// Oracle: mpmath `lambertw(-log(2)/(5*2**(1/5)), -1) + log(2)/5` =
/// `-3.1735603793189157936117521219400470128482468362755`.
#[test]
fn sign_of_a_lower_branch_lambert_value() {
    let ctx = Context::new();
    let d = p(&ctx, "lambertw(-log(2)/(5*2^(1/5)), -1) + log(2)/5");
    assert_eq!(d.is_real(), Some(true));
    assert_eq!(d.is_negative(), Some(true));
    assert_eq!(d.sign().eval().to_string(), "-1");
    assert_eq!(
        rel(&ctx, "lambertw(-log(2)/(5*2^(1/5)), -1) < -log(2)/5"),
        "True"
    );
}

/// Before (0.45): sums of arctangents of rationals were never decided:
/// `Eq(atan(1/2) + atan(1/3), π/4)`, Machin's `Eq(4·atan(1/5) − atan(1/239),
/// π/4)` stayed relations and `sign(atan(1/2) + atan(1/3) − π/4)` stayed.
/// They are decided exactly: `Σ nᵢ·atan(pᵢ/qᵢ)` is the argument of
/// `∏ (qᵢ + i·pᵢ)^{nᵢ}`.
///
/// Oracle: mpmath at 50 digits: `atan(1/2) + atan(1/3) - pi/4` = 0,
/// `atan(2) + atan(3) - 3*pi/4` = 0, `atan(1/2) + atan(1/3) - pi/5` =
/// `0.15707963267948966192313216916397514420985846996875`; SymPy 1.14
/// `(atan(Rational(1,2)) + atan(Rational(1,3)) - pi/4).equals(0)` → `None`
/// (undecided there).
#[test]
fn arctangent_sums_are_decided() {
    let ctx = Context::new();
    assert_eq!(rel(&ctx, "Eq(atan(1/2) + atan(1/3), pi/4)"), "True");
    assert_eq!(rel(&ctx, "Eq(4*atan(1/5) - atan(1/239), pi/4)"), "True");
    assert_eq!(rel(&ctx, "atan(2) + atan(3) > 3*pi/4"), "False");
    assert_eq!(rel(&ctx, "atan(2) + atan(3) >= 3*pi/4"), "True");
    assert_eq!(rel(&ctx, "Ne(atan(1/2) + atan(1/3), pi/5)"), "True");
    let z = p(&ctx, "atan(1/2) + atan(1/3) - pi/4");
    assert_eq!(z.sign().eval().to_string(), "0");
    assert_eq!(z.is_zero(), Some(true));
    let b = ctx
        .parse_bool("Eq(atan(1/2) + atan(1/3), pi/4)")
        .expect("parses");
    assert_eq!(b.simplify().to_string(), "True");
    // An arctangent part that is 0 by itself leaves the rest to decide.
    assert_eq!(
        rel(
            &ctx,
            "Eq(cos(pi/16) + 4*atan(1/5) - atan(1/239), sqrt(2 + sqrt(2 + sqrt(2)))/2 + pi/4)"
        ),
        "True"
    );
}

/// Before (0.45): with `x` positive, `solve_general` returned the member
/// `atan(1/2) + atan(1/3) − π/4` of `x − atan(1/2) − atan(1/3) + π/4 = 0`
/// (which is 0, not positive), while `solve` refused it since 0.45 ("0 to
/// 20 digits, and not known to be 0").  Both now decide it exactly and
/// return no root; a member that is 0 to every precision without being
/// decided (`sin²1 + cos²1 − 1`) is refused by both.
///
/// Oracle: mpmath `atan(1/2) + atan(1/3) - pi/4` = 0 at 50 digits (the
/// root is 0); SymPy 1.14 `solve(x - atan(1/2) - atan(1/3) + pi/4, x)` with
/// `x` positive returns `[-pi/4 + atan(1/3) + atan(1/2)]` (it keeps the 0).
#[test]
fn solve_general_and_solve_agree_on_zero_members() {
    let ctx = Context::new();
    let x = ctx.symbol("x").assume(Assumption::Positive).unwrap();
    let e = &(&x - &p(&ctx, "atan(1/2) + atan(1/3)")) + &p(&ctx, "pi/4");
    assert!(e.solve(&x).expect("decided").is_empty());
    assert!(e.solve_general(&x).expect("decided").solutions.is_empty());
    let undecided = &x - &p(&ctx, "sin(1)^2 + cos(1)^2 - 1");
    assert!(undecided.solve(&x).is_err());
    assert!(undecided.solve_general(&x).is_err());
    let y = ctx.symbol("y");
    let free = &(&y - &p(&ctx, "atan(1/2) + atan(1/3)")) + &p(&ctx, "pi/4");
    assert_eq!(free.solve(&y).expect("solves").len(), 1);
}

/// Before (0.45): `lowergamma(30, z)` for a small complex `z` folded into
/// `29!·(1 − e^{−z}·Σ zᵏ/k!)`, which cancels 1,200 bits for `|z| ≈ 10⁻¹¹`:
/// `lowergamma(30, (−13/100 + 99/100·i)/10¹¹)` was `PrecisionExhausted` at
/// 43 digits.  It stays unevaluated, and `evalf` takes its series.
///
/// Oracle: mpmath `z**30/30*hyp1f1(30, 31, -z)` for `z = (-0.13 +
/// 0.99j)/10**11` = `2.2755915802351794800217528198152276110984138324669e-332
/// + 2.2304932987107915225504709003087261953234796858339e-332j`.
#[test]
fn lowergamma_of_small_complex_argument_keeps_its_digits() {
    let ctx = Context::new();
    let g = p(&ctx, "lowergamma(30, (-13/100 + 99/100*I)/10^11)").eval();
    assert!(g.to_string().starts_with("lowergamma("), "{g}");
    let v = g.eval_decimal(43).expect("evaluates");
    assert_eq!(
        v,
        "2.275591580235179480021752819815227611098414e-332 + \
         2.23049329871079152255047090030872619532348e-332*i"
    );
}

/// Before (0.45): the cyclotomic zero test depended on which generator came
/// on top of its tower, so `a < b` and `a ≤ b` were decided while `a > b`,
/// `a ≥ b` and `Eq(a, b)` stayed, for `a = tan(13π/32)·√(21 + 4√5)` and `b`
/// its half-angle radical times `1 + 2√5` (the tower took `√(21 + 4√5)`,
/// which is `1 + 2√5`, for a new generator, and two conjugates vanished).
///
/// Oracle: mpmath at 50 digits `tan(13*pi/32)*sqrt(21 + 4*sqrt(5)) -
/// R*(1 + 2*sqrt(5))` = 0 for the half-angle radical `R` of
/// `tan(13*pi/32)` below.
#[test]
fn order_decisions_do_not_depend_on_the_side() {
    let ctx = Context::new();
    let a = "tan(13*pi/32)*sqrt(21 + 4*sqrt(5))";
    let b = "sqrt((1 + sqrt((1 + sqrt((1 - sqrt(2)/2)/2))/2))/2)\
             /sqrt((1 - sqrt((1 + sqrt((1 - sqrt(2)/2)/2))/2))/2)*(1 + 2*sqrt(5))";
    assert_eq!(rel(&ctx, &format!("Eq({a}, {b})")), "True");
    assert_eq!(rel(&ctx, &format!("{a} > {b}")), "False");
    assert_eq!(rel(&ctx, &format!("{a} >= {b}")), "True");
    assert_eq!(rel(&ctx, &format!("{a} < {b}")), "False");
    assert_eq!(rel(&ctx, &format!("{a} <= {b}")), "True");
}
