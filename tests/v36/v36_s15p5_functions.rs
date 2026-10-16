//! Tests of session 15 (s15p5): special functions and integration coverage.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn decimal(ctx: &Context, s: &str, digits: u32) -> String {
    parse(ctx, s)
        .eval_decimal(digits)
        .unwrap_or_else(|err| format!("{s}: {err:?}"))
}

// ── Item 1: the Hermite function at negative integer degrees ──

/// `H_{−m}(x) = (√π/2)·e^{x²}·i^{m−1}erfc(x)` (DLMF 7.18, 12.7.6): `eval`
/// writes the closed form, `H_{−1} = (√π/2)e^{x²}erfc x`, `H_{−2} = 1/2 −
/// (√π/2)·x·e^{x²}erfc x` (the recurrence `H_{ν+1} = 2xH_ν − 2νH_{ν−1}` from
/// `H_0 = 1`).  Up to 0.58 `hermite(−m, x)` stayed and `eval_decimal`
/// refused it ("degree must be a non-negative integer").  mpmath 1.3
/// `hermite(-1, 2)` = `0.22633852499058728968130400071475`, `hermite(-2, 2)` =
/// `0.047322950018825420637391998570509`, `hermite(-3, -0.5)` =
/// `0.77383791263888757253258172617075`, `hermite(-1, 30)` =
/// `0.016657422796805108128450513346014`, `hermite(-1, 1+1j)` =
/// `0.27007252007437786285025675829533 - 0.18452922942453329220852705742729j`,
/// `hermite(-70, 1/3)` = `1.7900566729651096843551108581505e-62` (beyond the
/// closed form's bound of 64: `evalf`'s hypergeometric route).
#[test]
fn hermite_at_negative_integer_degrees_has_values() {
    let ctx = Context::new();
    assert_eq!(
        parse(&ctx, "hermite(-1, x)").eval().to_string(),
        "1/2*sqrt(pi)*exp(x^2)*erfc(x)"
    );
    assert_eq!(
        parse(&ctx, "hermite(-2, x)").eval().to_string(),
        "-1/2*x*sqrt(pi)*exp(x^2)*erfc(x) + 1/2"
    );
    for (s, want) in [
        ("hermite(-1, 2)", "0.226338524990587289681304000715"),
        ("hermite(-2, 2)", "0.0473229500188254206373919985705"),
        ("hermite(-3, -1/2)", "0.773837912638887572532581726171"),
        ("hermite(-1, 30)", "0.016657422796805108128450513346"),
        (
            "hermite(-1, 1 + I)",
            "0.270072520074377862850256758295 - 0.184529229424533292208527057427*i",
        ),
        ("hermite(-70, 1/3)", "1.79005667296510968435511085815e-62"),
    ] {
        assert_eq!(decimal(&ctx, s, 30), want, "{s}");
    }
}

/// The closed form satisfies the derivative rule `H_ν′ = 2ν·H_{ν−1}` at `ν =
/// −1`: `d/dx H_{−1}(x) = −2·H_{−2}(x)` (an identity of the closed forms,
/// checked at `x = 7/10` to 30 digits).
#[test]
fn hermite_function_closed_forms_satisfy_the_derivative_rule() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let h1 = parse(&ctx, "hermite(-1, x)").eval();
    let h2 = parse(&ctx, "-2*hermite(-2, x)").eval();
    let at = |e: &Ex| e.subs(&x, &ctx.rational(7, 10)).eval_decimal(30).unwrap();
    assert_eq!(at(&h1.diff(&x)), at(&h2));
}

// ── Item 2: elliptic_pi with m next to the cut (1, ∞) ──

/// `Π(n | m)` for a non-real `m` with `Re m > 1` whose arguments lie in no
/// half-plane (neither directly nor after the involution): the split at
/// `t = 1/m` continued off the axis.  Up to 0.58 refused ("the arguments
/// 1 - m, 1 and 1 - n lie in no half-plane").  Oracle: a quadrature of
/// `1/((1 − n sin²θ)√(1 − m sin²θ))` over `[0, π/2]` with mpmath at 50
/// digits (K4's hunt truth): `1.3351553735035455048151267175926 +
/// 1.3052548740470521945378137372515i`, `2.7830559393243548669596008074 +
/// 2.0207375189705671048791562300806i`, `−0.1499624497938804218021817178949 +
/// 2.0794924762603318475667229681556i`.
#[test]
fn elliptic_pi_next_to_the_cut_beyond_one_is_evaluated() {
    let ctx = Context::new();
    for (s, want) in [
        (
            "elliptic_pi(1/2 - I/3, 4 + I/10)",
            "1.33515537350354550481512671759 + 1.30525487404705219453781373725*i",
        ),
        (
            "elliptic_pi(2 + I, 2 - I/1000)",
            "2.78305593932435486695960080742 + 2.02073751897056710487915623008*i",
        ),
        (
            "elliptic_pi(4 + I/10, 2 - I/1000)",
            "-0.149962449793880421802181717895 + 2.07949247626033184756672296816*i",
        ),
    ] {
        assert_eq!(decimal(&ctx, s, 30), want, "{s}");
    }
}

// ── Item 3: uppergamma's closed form only while it stays small ──

/// `uppergamma(8, a₁ + … + a₂₅)`: the closed form `7!·e^{−x}·Σ x^k/k!`
/// expanded `x⁷` into `C(32, 7) = 3.4·10⁶` products (24 s and 160 MB of
/// output in 0.58; the G-test on a 5 × 5 table hung on it).  Now `Γ(8, x)`
/// stays; a small argument still gets the closed form (`Γ(3, a + b)`, six
/// terms), and the kept node evaluates numerically: at every `aᵢ = 1/10`,
/// `Γ(8, 5/2) = 7!·e^{−5/2}·Σ_{k<8} (5/2)^k/k!` (mpmath `gammainc(8, 2.5)`
/// = `5018.59665473370368549782541408`).
#[test]
fn uppergamma_of_a_long_sum_keeps_its_node() {
    let ctx = Context::new();
    let names: Vec<String> = (1..=25).map(|i| format!("a{i}")).collect();
    let sum = names.join(" + ");
    let g = parse(&ctx, &format!("uppergamma(8, {sum})")).eval();
    assert!(g.to_string().starts_with("uppergamma(8, "), "{g}");
    let mut at = g;
    for n in &names {
        at = at.subs(&ctx.symbol(n), &ctx.rational(1, 10));
    }
    assert_eq!(at.eval_decimal(15).unwrap(), "5018.5966547337");
    let small = parse(&ctx, "uppergamma(3, a + b)").eval().to_string();
    assert!(!small.contains("uppergamma"), "{small}");
}

// ── Item 4: integration ──

/// A rational function with algebraic coefficients: `∫ dx/(x(x − √2)) =
/// (√2/2)·(ln|x − √2| − ln|x|)` (SymPy 1.14: `−√2·ln(x)/2 + √2·ln(x −
/// √2)/2`); up to 0.58 unevaluated.  Checked by differentiating at `x = 3`:
/// `F′(3) = 1/(3(3 − √2))`.
#[test]
fn rational_function_with_algebraic_coefficients_is_integrated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "1/(x*(x - sqrt(2)))");
    let big_f = f.try_integrate(&x).unwrap();
    assert_eq!(
        big_f.to_string(),
        "-1/2*sqrt(2)*ln(abs(x)) + 1/2*sqrt(2)*ln(abs(x - sqrt(2)))"
    );
    let three = ctx.int(3);
    let lhs = big_f.diff(&x).subs(&x, &three).eval_decimal(25).unwrap();
    let rhs = f.subs(&x, &three).eval_decimal(25).unwrap();
    assert_eq!(lhs, rhs);
}

/// A pole of `R(e^{λx})` (or `R(cˣ)`) with a non-real rate on the
/// interval: `1/(e^{(1+i)x} − 1) ~ 1/((1 + i)x)` at 0 is not integrable, nor
/// is `polylog(0, cˣ) = cˣ/(1 − cˣ) ~ −1/(x·Log c)` (`c = Chi(−3/4)`).  The
/// comparison test with the non-real limit `(x − c)·f → L ≠ 0`.  Up to 0.58
/// refused ("no antiderivative in closed form") instead of `Divergent`.
#[test]
fn integrals_across_a_pole_with_a_complex_rate_diverge() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (s, lo, hi) in [
        ("1/(exp((1+I)*x) - 1)", -1, 1),
        ("1/(exp((1+I)*x) - 1)", 0, 1),
        ("polylog(0, Chi(-3/4)^x)", -1, 1),
    ] {
        let r = parse(&ctx, s).try_integrate_definite(&x, &ctx.int(lo), &ctx.int(hi));
        assert!(
            matches!(r, Err(SymplexError::Divergent { .. })),
            "{s}: {r:?}"
        );
    }
}

// ── Item 5: Max/Min of non-real arguments, 0/f(zoo) ──

/// `Max`/`Min` are defined on the reals: an argument certified non-real
/// leaves no point where the order is defined, so the value is `nan` (SymPy
/// 1.14 raises "The argument 'I' is not comparable"); up to 0.58 `max(1, I)`
/// stayed a node (and `0·max(1, I)` was 0).  `0/f(zoo)` is `0/zoo = 0` where
/// `f` tends to `zoo` in every direction (`W`, `asinh`: `~ ln z`) and has no
/// value where `f` has no limit at `zoo` (`Γ` grows along the positive axis
/// and tends to 0 along the imaginary one; `atan`, `erf` tend to different
/// constants): up to 0.58 `0/gamma(zoo)` and `0/atan(zoo)` were 0.
#[test]
fn max_of_non_real_arguments_and_quotients_at_complex_infinity() {
    let ctx = Context::new();
    for (s, want) in [
        ("max(1, I)", "nan"),
        ("min(1, 2*I)", "nan"),
        ("max(x, zeta(3/7 + I))", "nan"),
        ("0/gamma(zoo)", "nan"),
        ("0/atan(zoo)", "nan"),
        ("0/erf(zoo)", "nan"),
        ("0/LambertW(zoo)", "0"),
        ("0/asinh(zoo)", "0"),
        ("max(1, 2, x)", "max(2, x)"),
    ] {
        assert_eq!(parse(&ctx, s).to_string(), want, "{s}");
    }
}
