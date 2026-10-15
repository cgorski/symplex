//! Tests of session 15 (s15j3): evalf per-part bounds, approximate realness, complex elliptic_pi.
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

// ── Item 1: each part of a power, an exponential, a sine with its own bound ──

/// `(1 + εi)² = 1 − ε² + 2εi`, `(1 + εi)⁻² = (1 − ε² − 2εi)/(1 + ε²)²`
/// (exact identities), `(1 + εi)^(1/3) = 1 + εi/3 + O(ε²)` and `2^(εi) = 1 +
/// εi·ln 2 + O(ε²)` (mpmath `power(1 + 1e-45j, 1/3)` = `1.0 +
/// 3.33333333333333e-46j`, `power(2, 1e-45j)` = `1.0 + 6.93147180559945e-46j`)
/// at `ε = 10⁻⁴⁵`.  Up to 0.54 each printed `1` at 10 digits, `eval_f64`
/// returned `1.0` and `eval_complex64` had `im == 0.0`: the power carried one
/// bound for both parts, the rounding at the magnitude of the value (and, for
/// `exp(x·Log b)`, `|x|·(|ln|b|| + π)·2^(3 − prec)` for every non-real base),
/// which buried the imaginary part.
#[test]
fn power_of_a_value_next_to_the_real_axis_keeps_its_imaginary_part() {
    let ctx = Context::new();
    for (s, want) in [
        ("(1 + I*10^(-45))^2", "1 + 2e-45*i"),
        ("(1 + I/10^45)^(-2)", "1 - 2e-45*i"),
        ("(1 + I/10^45)^(1/3)", "1 + 3.333333333e-46*i"),
        ("2^(I/10^45)", "1 + 6.931471806e-46*i"),
    ] {
        assert_eq!(decimal(&ctx, s, 10), want, "{s}");
        assert!(
            matches!(
                parse(&ctx, s).eval_f64(),
                Err(SymplexError::ComputationFailed { .. })
            ),
            "{s}"
        );
    }
    let z = parse(&ctx, "(1 + I*10^(-45))^2").eval_complex64().unwrap();
    assert_eq!((z.re, z.im), (1.0, 2e-45));
}

/// `cos(a + bi) = cos a·cosh b − i·sin a·sinh b` and `sin(a + bi) = sin a·cosh
/// b + i·cos a·sinh b` (exact identities; `sinh(10⁻⁵⁰) = 10⁻⁵⁰` to 100
/// digits): `cos(1 + 10⁻⁵⁰i) = 0.5403023059 − 8.414709848·10⁻⁵¹i`, and the
/// hyperbolic functions next to the imaginary axis likewise.  Up to 0.54 the
/// imaginary part (the real part of `sinh`) was dropped at 10 and 30 digits
/// and `eval_f64` returned the real part: both parts carried `r·cosh(Im z)`,
/// the error of the larger part.
#[test]
fn trigonometric_functions_next_to_an_axis_keep_the_small_part() {
    let ctx = Context::new();
    for (s, want) in [
        ("cos(1 + I/10^50)", "0.5403023059 - 8.414709848e-51*i"),
        ("sin(1 + I/10^50)", "0.8414709848 + 5.403023059e-51*i"),
        ("cosh(I + 1/10^50)", "0.5403023059 + 8.414709848e-51*i"),
        ("sinh(I + 1/10^50)", "5.403023059e-51 + 0.8414709848*i"),
    ] {
        assert_eq!(decimal(&ctx, s, 10), want, "{s}");
    }
    assert_eq!(
        decimal(&ctx, "cos(1 + I/10^50)", 30),
        "0.540302305868139717400936607443 - 8.4147098480789650665250232163e-51*i"
    );
    assert!(matches!(
        parse(&ctx, "cos(1 + I/10^50)").eval_f64(),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

/// `exp(i(π/2 + ε)) = −sin ε + i·cos ε` (exact identity): `−10⁻⁴⁵ + i` at 10
/// digits.  Up to 0.54 it printed `i`: the real part needs the argument to
/// about 190 bits, and the precision search stopped once the whole value had
/// its digits, dropping a part whose ball contained 0.  It now pursues such a
/// part of a number to the cap.  Likewise `Ci(−69·10³⁰π) = Ci(69·10³⁰π) + iπ`
/// (the value from above on the cut; mpmath `ci(-69e30*pi)` at 150 and 400
/// digits: real part `-2.12814920483801242268177826527e-65`), printed
/// `3.141592654*i`, and
/// `tan(1 + εi) = tan 1 + iε·sec²1 + O(ε²)` (`3.425518821e-50`), printed
/// `1.557407725`.
#[test]
fn precision_search_pursues_a_part_whose_ball_contains_zero() {
    let ctx = Context::new();
    assert_eq!(decimal(&ctx, "exp(I*(pi/2 + 10^(-45)))", 10), "-1e-45 + i");
    let z = parse(&ctx, "exp(I*(pi/2 + 10^(-45)))")
        .eval_complex64()
        .unwrap();
    assert_eq!((z.re, z.im), (-1e-45, 1.0));
    assert_eq!(
        decimal(&ctx, "Ci(-pi*69*10^30)", 10),
        "-2.128149205e-65 + 3.141592654*i"
    );
    assert_eq!(
        decimal(&ctx, "Ci(-pi*69*10^30)", 30),
        "-2.12814920483801242268177826527e-65 + 3.14159265358979323846264338328*i"
    );
    assert_eq!(
        decimal(&ctx, "tan(1 + I/10^50)", 10),
        "1.557407725 + 3.425518821e-50*i"
    );
}

/// A part that is 0 is still dropped (after the pursuit): `Chi(−6) − iπ =
/// Chi(6)` (`Chi(−x) = Chi(x) + iπ` from above; mpmath `chi(6)` =
/// `42.9947010299935`) and `(1 + i)^(1/3)·(1 − i)^(1/3) = 2^(1/3)` print no
/// imaginary part.
#[test]
fn a_part_that_is_zero_is_still_dropped() {
    let ctx = Context::new();
    assert_eq!(decimal(&ctx, "Chi(-6) - I*pi", 10), "42.99470103");
    assert_eq!(decimal(&ctx, "(1+I)^(1/3)*(1-I)^(1/3)", 10), "1.25992105");
    assert_eq!(
        parse(&ctx, "Chi(-6) - I*pi").eval_f64().unwrap(),
        42.99470102999352
    );
}

/// `ln z = ln|z| + i·arg z` with `|z|` from `c_abs`, whose rounding is an
/// absolute error of the real part: `ln(1 + 10⁻¹⁵i) = 4.99999999999999999999
/// 99999999975·10⁻³¹ + 9.999…9667·10⁻¹⁶i` (mpmath `log(1 + 1e-15j)` at 60
/// digits).  Up to 0.54 the bound of the real part was the rounding at the
/// value's magnitude (`10⁻¹⁵`), and the real part printed
/// `5.00000000000000000000000000063e-31` at 30 digits (wrong from the 29th
/// digit) and `4.999999957e-31` at 10; `eval_complex64` gave
/// `4.999999957080667e-31`.
#[test]
fn log_next_to_one_has_the_digits_of_its_real_part() {
    let ctx = Context::new();
    assert_eq!(decimal(&ctx, "log(1 + I/10^15)", 10), "5e-31 + 1e-15*i");
    assert_eq!(decimal(&ctx, "log(1 + I/10^15)", 30), "5e-31 + 1e-15*i");
    assert_eq!(
        decimal(&ctx, "log(1 + I/10^15)", 60),
        "4.9999999999999999999999999999975e-31 + \
         9.99999999999999999999999999999666666666666666666666666666667e-16*i"
    );
    let z = parse(&ctx, "log(1 + I/10^15)").eval_complex64().unwrap();
    assert_eq!((z.re, z.im), (5e-31, 1e-15));
}

// ── Item 2: realness from a certified imaginary part, however small ─────────

/// `Im √(2 + bi) = b/(2·Re √(2 + bi)) > 0` for `b = e⁻⁹⁰` and `Im cos(1 +
/// 10⁻⁵⁰i) = −sin 1·sinh 10⁻⁵⁰ < 0` (exact identities): neither is real, so a
/// real `x` has no solution of `x = c`.  Up to 0.54 `solve` refused
/// ("cannot decide whether the solution … is real") and `solve_as_set`
/// returned `{√(e⁻⁹⁰i + 2)}` for a real unknown: `constant_realness` read a
/// certified imaginary part below the digits (relative to the real part) as
/// no evidence, and a certified nonzero `im(c)` as undecided.
#[test]
fn constant_with_a_tiny_certified_imaginary_part_is_not_real() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    for s in ["x - sqrt(2 + I*exp(-90))", "x - cos(1 + I/10^50)"] {
        let e = parse(&ctx, s);
        assert_eq!(e.solve(&x).unwrap(), Vec::<Ex>::new(), "{s}");
        assert_eq!(e.solve_as_set(&x).to_string(), "EmptySet", "{s}");
    }
}

// ── Item 3: Π(n | m) off its real range ─────────────────────────────────────

/// `Π(n | m)` for `m > 1` or `n > 1` (the limits from below, mpmath's and
/// SymPy's convention) and for non-real arguments.  Oracle: mpmath 1.3
/// `ellippi(1/2, 2)` = `1.53273834902928110285994989876982 −
/// 2.10413339576823759703690437228748j`, `ellippi(2, 1/2)` =
/// `−0.31354468346518404147230722434314 − 1.81379936423421785059407825764216j`
/// (at 35 and 50 digits; the task's values to 4 digits).  Up to 0.54 both
/// were refused ("parameter m must satisfy m ≤ 1", "the characteristic n
/// must satisfy n < 1").
#[test]
fn elliptic_pi_beyond_its_real_range() {
    let ctx = Context::new();
    assert_eq!(
        decimal(&ctx, "elliptic_pi(1/2, 2)", 30),
        "1.53273834902928110285994989877 - 2.10413339576823759703690437229*i"
    );
    assert_eq!(
        decimal(&ctx, "elliptic_pi(2, 1/2)", 30),
        "-0.313544683465184041472307224343 - 1.81379936423421785059407825764*i"
    );
}

/// `Π(n | m)` with both `n > 1` and `m > 1`, and with non-real arguments
/// (Carlson's `R_J` over ℂ, the arguments rotated into the right
/// half-plane).  Oracle: mpmath 1.3 `ellippi` at 50 digits: `(3, 10)` →
/// `−0.11626192268376100748556529690903 − 0.41278962783185593904587431237208j`,
/// `(10, 2)` → `−0.077205475005400174053030015190543 −
/// 0.35917087996424960362905648096976j`, `(1/2 + i, 3)` →
/// `1.7975251485125982286793441884459 − 0.70190793889763885369325753320505j`,
/// `(1/3, 1/2 + i)` → `1.8109025406424515186598940833296 +
/// 0.57821800391523147333065927482702j`, `(5/4, −3)` →
/// `0.43002546026952616151202597978295 − 1.7037678315574018838163037798724j`.
/// Refused up to 0.54.  Arguments `1 − m`, 1, `1 − n` in no half-plane were
/// refused up to 0.55 (mpmath integrates there first); since 0.56 they go
/// through `Π(n | m) = K(m) − Π(m/n | m) + (π/2)/(√(1 − n)√(1 − m/n))`:
/// `(1/2 − i/3, 2 + i)` → `1.995339437 + 0.9025251621i` (mpmath `ellippi`
/// and a quadrature on the real path agree to 40 digits; this test pinned
/// the refusal).
#[test]
fn elliptic_pi_of_complex_arguments() {
    let ctx = Context::new();
    for (s, want) in [
        (
            "elliptic_pi(3, 10)",
            "-0.116261922683761007485565296909 - 0.412789627831855939045874312372*i",
        ),
        (
            "elliptic_pi(10, 2)",
            "-0.0772054750054001740530300151905 - 0.35917087996424960362905648097*i",
        ),
        (
            "elliptic_pi(1/2 + I, 3)",
            "1.79752514851259822867934418845 - 0.701907938897638853693257533205*i",
        ),
        (
            "elliptic_pi(1/3, 1/2 + I)",
            "1.81090254064245151865989408333 + 0.578218003915231473330659274827*i",
        ),
        (
            "elliptic_pi(5/4, -3)",
            "0.430025460269526161512025979783 - 1.70376783155740188381630377987*i",
        ),
    ] {
        assert_eq!(decimal(&ctx, s, 30), want, "{s}");
    }
    assert_eq!(
        decimal(&ctx, "elliptic_pi(1/2 - I/3, 2 + I)", 10),
        "1.995339437 + 0.9025251621*i"
    );
}

// ── Item 4: an order slot at infinity must be known finite ─────────────────

/// `K(1) = ∞` (`besseli(0, 0) = 1`), so `J_{K(I₀(0))}(∞)` and `Γ(K(I₀(0)),
/// ∞)` are not `J_ν(∞) = 0`, `Γ(s, ∞) = 0` for a finite order.  Up to 0.54
/// the constructors folded both to 0 (an order without a literal infinity
/// counted as finite), so `solve` called the equation an identity.  A finite
/// constant order still folds: `besselj(π, ∞) = 0`, `uppergamma(√2, ∞) = 0`
/// (DLMF 10.7.8, 8.11.2).
#[test]
fn order_at_infinity_must_be_known_finite() {
    let ctx = Context::new();
    for s in [
        "besselj(elliptic_k(besseli(0, 0)), oo)",
        "uppergamma(elliptic_k(besseli(0, 0)), oo)",
        "besselk(elliptic_k(besseli(0, 0)), oo)",
    ] {
        assert_ne!(parse(&ctx, s).to_string(), "0", "{s}");
    }
    for s in [
        "besselj(elliptic_k(besseli(0, 0)), oo)",
        "besselk(elliptic_k(besseli(0, 0)), oo)",
    ] {
        assert_eq!(parse(&ctx, s).is_zero(), None, "{s}");
    }
    assert_eq!(parse(&ctx, "besselj(pi, oo)").to_string(), "0");
    assert_eq!(parse(&ctx, "uppergamma(sqrt(2), oo)").to_string(), "0");
}
