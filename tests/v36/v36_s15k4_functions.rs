//! Tests of session 15 (s15k4): special-function coverage and elliptic performance.
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

fn refused(ctx: &Context, s: &str) -> bool {
    parse(ctx, s).eval_decimal(15).is_err()
}

// ── Item 1: values that were refused although they exist ──

/// `L_ν^(−k)(x) = (−x)^k/k!·₁F₁(k − ν; k + 1; x)` (the regularized function;
/// the polynomial identity `L_n^(−k) = (−x)^k(n − k)!/n!·L_{n−k}^(k)` for any
/// degree).  mpmath `laguerre(0.5, -1, 1.5)` = `-2.36147430045963859105772672030`,
/// `laguerre(2.5, -3, 2/3)` = `-0.0539509996784928090209770767058` (at 30 and
/// 60 digits).  Up to 0.55 both were refused: "hypergeometric series with a
/// non-positive integer lower parameter".
#[test]
fn assoc_laguerre_with_a_negative_integer_alpha_is_evaluated() {
    let ctx = Context::new();
    assert_eq!(
        decimal(&ctx, "assoc_laguerre(1/2, -1, 3/2)", 30),
        "-2.3614743004596385910577267203"
    );
    assert_eq!(
        decimal(&ctx, "assoc_laguerre(5/2, -3, 2/3)", 30),
        "-0.0539509996784928090209770767058"
    );
    // The integer degree keeps the recurrence (exact: L_2^(-1)(x) = x²/2 − x).
    assert_eq!(decimal(&ctx, "assoc_laguerre(2, -1, 3/2)", 10), "-0.375");
}

/// mpmath `hermite(0.5, 1+1j)` = `1.58050239538564802416939418665 +
/// 0.602413236899782871355372133225j`, `laguerre(0.5, 0, 1-1j)` =
/// `0.528223962820000637479906192345 + 0.644669002947180008431175215682j`,
/// `hermite(0.5, 2j)` = `1.38896134522779577385153960557 +
/// 1.38704870410248721901661111168j` (a real negative `z²`: Kummer's
/// transformation), `hermite(1j, 0.5)` = `1.1209910338556 + 0.57355221587807j`
/// (a complex degree).  Up to 0.55 every one was refused: "orthogonal
/// polynomial of complex arguments not supported".
#[test]
fn hermite_and_laguerre_of_complex_arguments_are_evaluated() {
    let ctx = Context::new();
    for (s, want) in [
        (
            "hermite(1/2, 1 + I)",
            "1.58050239538564802416939418665 + 0.602413236899782871355372133225*i",
        ),
        (
            "laguerre(1/2, 1 - I)",
            "0.528223962820000637479906192345 + 0.644669002947180008431175215682*i",
        ),
        (
            "hermite(1/2, 2*I)",
            "1.38896134522779577385153960557 + 1.38704870410248721901661111168*i",
        ),
    ] {
        assert_eq!(decimal(&ctx, s, 30), want, "{s}");
    }
    assert_eq!(
        decimal(&ctx, "hermite(I, 1/2)", 14),
        "1.1209910338556 + 0.57355221587807*i"
    );
    // A real negative-integer degree: the Hermite function (0.59; this line
    // pinned the refusal before).  mpmath `hermite(-2, 1+1j)` =
    // `0.045398250501088844941216184277386 - 0.085543290649844570641729700868037j`.
    assert_eq!(
        decimal(&ctx, "hermite(-2, 1 + I)", 30),
        "0.0453982505010888449412161842774 - 0.085543290649844570641729700868*i"
    );
}

/// mpmath `ellipf(1+1j, 0.5)` = `0.804237254059707722231450497249 +
/// 1.16273803720777866019884530807j`, `ellipf(2+1j, 0.5)` =
/// `2.77616428760282857236865038502 + 1.25662824814895094374786644363j`
/// (`Re φ > π/2`: the reduction by `π` adds `2K(m)`).  On mpmath's cut `Re φ =
/// π/2` (`ellipf(pi/2 ∓ 1e-20 + 1j, 0.5)` = `1.2868… / 2.4213… + 1.8541j`) the
/// side is undecidable and the value is refused.  Up to 0.55 every complex
/// amplitude was refused ("elliptic_f of complex argument not yet supported").
#[test]
fn elliptic_f_of_a_complex_amplitude_is_evaluated() {
    let ctx = Context::new();
    assert_eq!(
        decimal(&ctx, "elliptic_f(1 + I, 1/2)", 30),
        "0.804237254059707722231450497249 + 1.16273803720777866019884530807*i"
    );
    assert_eq!(
        decimal(&ctx, "elliptic_f(2 + I, 1/2)", 30),
        "2.77616428760282857236865038502 + 1.25662824814895094374786644363*i"
    );
    assert!(refused(&ctx, "elliptic_f(pi/2 + I, 1/2)"));
}

/// An amplitude on the imaginary axis with a real parameter keeps `1 −
/// m sin²φ = 1 + m sinh²t` exactly real, and where it is negative the side
/// is the convention's (mpmath's principal root), not undecided: mpmath
/// `ellipf(1j, -1.3)` = `0.50478089298225337264383237056548 +
/// 1.1884962063131102970444418792239j`, `ellipf(1j*pi, -2)` =
/// `1.1102300552120122098929219374796 + 1.0010773804561062360796595863838j`
/// (40 and 60 digits).  Refused in the first build of 0.56 (the cut check
/// took the exactly real `y < 0` for a ball across the cut).
#[test]
fn elliptic_f_of_an_imaginary_amplitude_with_a_negative_y() {
    let ctx = Context::new();
    assert_eq!(
        decimal(&ctx, "elliptic_f(I, -13/10)", 30),
        "0.504780892982253372643832370565 + 1.18849620631311029704444187922*i"
    );
    assert_eq!(
        decimal(&ctx, "elliptic_f(I*pi, -2)", 30),
        "1.11023005521201220989292193748 + 1.00107738045610623607965958638*i"
    );
}

/// mpmath `rf(1+1j, 3)` = `10j` (exactly: `(1 + i)(2 + i)(3 + i)`), `rf(2, 1j)`
/// = `0.652965496420166727838646247946 + 0.343065839816545357588735986978j`
/// (`Γ(2 + i)`), `ff(1+1j, 0.5)` = `1.19823068195560986317897904841 +
/// 0.412589662904814163928416175317j`, `rf(-2, 1j)` = 0 (a pole of `Γ(x)`).
/// Up to 0.55 each was refused ("rising_factorial of complex argument not yet
/// supported in evalf").
#[test]
fn rising_and_falling_factorials_of_complex_arguments_are_evaluated() {
    let ctx = Context::new();
    for (s, want) in [
        ("rising_factorial(1 + I, 3)", "10*i"),
        (
            "rising_factorial(2, I)",
            "0.652965496420166727838646247946 + 0.343065839816545357588735986978*i",
        ),
        (
            "falling_factorial(1 + I, 1/2)",
            "1.19823068195560986317897904841 + 0.412589662904814163928416175317*i",
        ),
        ("rising_factorial(-2, I)", "0"),
    ] {
        assert_eq!(decimal(&ctx, s, 30), want, "{s}");
    }
    // `rf(i, 0) = 1` exactly (the empty product, an exact ball), so the outer
    // call has an exactly real argument: `rf(1, −13/10) = Γ(−3/10)` = mpmath
    // `rf(1, -1.3)` = `-4.3268511088251926189372372638427`.  Refused in the
    // first build of 0.56 (the inner value carried a bound with an imaginary
    // part).
    assert_eq!(
        decimal(&ctx, "rising_factorial(rising_factorial(I, 0), -13/10)", 30),
        "-4.32685110882519261893723726384"
    );
}

/// `∫_{1/4}^{1/2} t^{−3/2}(1 − t) dt = 0.757359312880714853594933827371`
/// (mpmath `quad` and `betainc(-0.5, 2, 0.25, 0.5)`), `∫_{1/4}^{1/2} dt/(t(1 − t))
/// = ln 3` (exact), `∫_{1/4}^{1/2} t(1 − t)^{−3/2} dt =
/// 0.201188802791904794841024709115` (mpmath), and the regularized
/// `betainc_regularized(−1/2, 3/2, 1/4, 1/2) = −0.299371351809336089280451276469`
/// (mpmath; `B(−1/2, 3/2) = −π`).  Up to 0.55 every shape `≤ 0` was refused
/// ("betainc requires a, b > 0 for numerical evaluation").  Where the integral
/// diverges (a limit at the singular end) it stays refused (mpmath's
/// `betainc(-0.5, 2, 0, 0.5) = -4.24` is a continuation, not the integral),
/// and so does the regularized value at a pole of `B(a, b)` (mpmath `1e-34`).
#[test]
fn betainc_with_a_shape_at_most_zero_is_the_convergent_integral() {
    let ctx = Context::new();
    for (s, want) in [
        (
            "betainc(-1/2, 2, 1/4, 1/2)",
            "0.757359312880714853594933827371",
        ),
        ("betainc(0, 0, 1/4, 1/2)", "1.09861228866810969139524523692"),
        (
            "betainc(2, -1/2, 1/4, 1/2)",
            "0.201188802791904794841024709115",
        ),
        (
            "betainc_regularized(-1/2, 3/2, 1/4, 1/2)",
            "-0.299371351809336089280451276469",
        ),
    ] {
        assert_eq!(decimal(&ctx, s, 30), want, "{s}");
    }
    assert_eq!(
        decimal(&ctx, "log(3)", 30),
        "1.09861228866810969139524523692"
    );
    for s in [
        "betainc(-1/2, 2, 0, 1/2)",
        "betainc(2, -1/2, 1/2, 1)",
        "betainc_regularized(0, 2, 1/4, 1/2)",
    ] {
        assert!(refused(&ctx, s), "{s}");
    }
}

/// `Π(e − i0 | i) = K(i) − Π(i/e | i) − i(π/2)/(√(e − 1)·√(1 − i/e))` =
/// `0.334581823333586120832631914357 − 1.34837609272325818260446021297i`:
/// mpmath `ellippi(e, 1j)` and a quadrature on a path below the pole agree
/// to 50 digits (the value `−5.932 − 1.285i` quoted for it in 0.55's notes
/// came from a quadrature through the pole).  `Π(1/2 − i/3 | 2 + i)`, whose
/// arguments `1 − m`, 1, `1 − n` lie in no half-plane, =
/// `1.99533943680939413197890634311 + 0.90252516207884737170401677192i`
/// (mpmath, quadrature).  Up to 0.55 both were refused ("not yet supported").
#[test]
fn elliptic_pi_beyond_the_half_plane_is_evaluated() {
    let ctx = Context::new();
    assert_eq!(
        decimal(&ctx, "elliptic_pi(E, I)", 30),
        "0.334581823333586120832631914357 - 1.34837609272325818260446021297*i"
    );
    assert_eq!(
        decimal(&ctx, "elliptic_pi(1/2 - I/3, 2 + I)", 30),
        "1.99533943680939413197890634311 + 0.90252516207884737170401677192*i"
    );
}

// ── Items 2, 3: K and E beyond 1 by the reciprocal-modulus transformation ──

/// `E(m) = √μ·(R_F − R_D/3)(0, 1 − μ, 1) + i((1 − μ)/√μ)·(R_F − R_D/3)(0, μ, 1)`,
/// `μ = 1/m` (DLMF 19.7.3): `E(10⁶⁰⁰) = π/(4·10³⁰⁰)·(1 + O(10⁻⁶⁰⁰)) + 10³⁰⁰i`
/// (mpmath at 1,300 digits: `7.85398163397448309615660845820e-301 + 1.0e+300j`;
/// at 30 digits mpmath's real part is `−1.1e262`).  Up to 0.55 refused ("the
/// value cancels beyond the working precision": the real part is 2,000 bits
/// below the imaginary one).  `K(10⁶⁰⁰)` = `1.57079632679489661923132169164e-300
/// − 6.92161822259333595824231900648e-298i` (mpmath), and `E(2)`, `K(2)` keep
/// their values (mpmath `ellipe(2)` = `0.599070117367796103719961246140 +
/// 0.599070117367796103719961246140j`).
#[test]
fn elliptic_k_and_e_far_beyond_one_use_the_reciprocal_modulus() {
    let ctx = Context::new();
    assert_eq!(
        decimal(&ctx, "elliptic_e(10^600)", 30),
        "7.8539816339744830961566084582e-301 + 1e300*i"
    );
    assert_eq!(
        decimal(&ctx, "elliptic_k(10^600)", 30),
        "1.57079632679489661923132169164e-300 - 6.92161822259333595824231900648e-298*i"
    );
    assert_eq!(
        decimal(&ctx, "elliptic_e(2)", 30),
        "0.59907011736779610371996124614 + 0.59907011736779610371996124614*i"
    );
}
