//! Trigonometric integrands of the Rubi suite (chapters 4.1–4.7) that 0.37
//! left unevaluated, now integrated by general methods:
//!
//! * a rational function of `sin`, `cos`, `tan` of linear arguments with
//!   commensurable rates (`c + d·x`, `2x + 1`, `3a + 3b·x`): `u = c + d·x`,
//!   the multiple-angle and addition formulas, then Bioche's rules
//!   (`t = cos u`, `t = sin u`, `t = tan u`, else `t = tan(u/2)`);
//! * rational functions with `i` in their coefficients, by real and
//!   imaginary parts (`(a + i·a·tan u)ⁿ` through `t = tan u`);
//! * algebraic functions of `tan u` (`√(tan u)`, `√(1 + tan u)`) through
//!   `t = tan u`, and radicals of `sin u` (`cos u`) times an odd power of
//!   `cos u` (`sin u`) through `t = sin u` (`t = cos u`);
//! * radicals of `1 ± sin u`, `1 ± cos u` through the half angle, with a
//!   floor term where the sign of `sin(u/2 + φ)` flips;
//! * a polynomial in `x` times such a function (`(c + d·x)²·cos²(a +
//!   b·x)·sin³(a + b·x)`);
//! * the jumps at the poles of `tan w` of answers with parameters not
//!   declared real, as for real ones (also in the routes of 0.37).
//!
//! Every answer is checked by `F′ = f` at sample points and by `F(b) −
//! F(a)` against mpmath `quad` (`mp.dps = 30`, `[a, b]` split every 0.25)
//! on an interval where the integrand is continuous and that contains a
//! pole of the substitution (`tan(u/2)` or `tan u`), with the parameters at
//! the Rubi harness's values (`a = 6/5`, `b = 3/4`, `c = 5/3`, `d = 2/7`,
//! `e = 11/9` (written `e_`: `e` is Euler's number), `f = 13/6`, `A =
//! 19/10`, `B = 23/12`, `C = 29/14`)
//! substituted after integrating.  The parameters are not declared real:
//! the floor terms that keep the answers continuous are exact for real
//! values of them.  "SymPy" is `integrate` of SymPy 1.14 (25 s limit).

use symplex::prelude::*;

/// The Rubi harness's parameter values.
const PARAMS: [(&str, (i64, i64)); 9] = [
    ("a", (6, 5)),
    ("b", (3, 4)),
    ("c", (5, 3)),
    ("d", (2, 7)),
    ("e_", (11, 9)),
    ("f", (13, 6)),
    ("A", (19, 10)),
    ("B", (23, 12)),
    ("C", (29, 14)),
];

/// `∫ src d(var)`: `F′ − f` at sample points (where `f` is finite) and
/// `F(b) − F(a)` against `want` (`re`, `im`), the parameters substituted
/// after integrating.
fn check(src: &str, var: &str, a: f64, b: f64, want: (f64, f64)) {
    let ctx = Context::new();
    let x = ctx.symbol(var);
    let f = ctx.parse(src).unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "∫ {src} = {big_f}");
    let bind = |e: &Ex| {
        let mut e = e.clone();
        for (name, (p, q)) in PARAMS {
            e = e.subs(&ctx.symbol(name), &ctx.rational(p, q));
        }
        e
    };
    let residual = bind(&(&big_f.diff(&x) - &f));
    let scale = bind(&f);
    for k in 0..4 {
        let point = ctx
            .parse(&format!("{a} + ({b} - {a})*{}/9", 2 * k + 1))
            .unwrap();
        let (Ok(r), Ok(s)) = (
            residual.subs(&x, &point).eval_complex64(),
            scale.subs(&x, &point).eval_complex64(),
        ) else {
            continue;
        };
        assert!(
            r.norm() < 1e-9 * s.norm().max(1.0),
            "∫ {src} = {big_f}: F′ − f = {r} at {point}"
        );
    }
    let bound = bind(&big_f);
    let at = |v: f64| {
        let point = ctx.parse(&format!("{v}")).unwrap();
        bound.subs(&x, &point).eval_complex64().unwrap()
    };
    let got = at(b) - at(a);
    let tol = 1e-9 * want.0.abs().max(want.1.abs()).max(1.0);
    assert!(
        (got.re - want.0).abs() < tol && (got.im - want.1).abs() < tol,
        "∫ {src} = {big_f}: F({b}) − F({a}) = {got}, quadrature {want:?}"
    );
}

/// A linear argument other than `x`, rational in `sin`: `u = 2x + 1`, then
/// `t = tan(u/2)` with the floor term (the poles of `tan(x + 1/2)` at `x =
/// 1.07`, `4.21` are in `[0, 5]`).  Rubi 4.1.1.1 (`1/(5 + 3·sin(c + d·x))`);
/// 0.37: unevaluated; SymPy: a closed form.  mpmath: `quad(lambda t:
/// 1/(5+3*sin(2*t+1)), linspace(0, 5, 21))` = 1.215619898818430602696.
#[test]
fn linear_argument_through_the_half_angle() {
    check(
        "1/(5 + 3*sin(2*x + 1))",
        "x",
        0.0,
        5.0,
        (1.215_619_898_818_430_6, 0.0),
    );
}

/// Parameters in the denominator: the jump `G(+∞) − G(−∞)` of the
/// half-angle answer is written with `sign(im(·))` of its logarithms'
/// constants, exact for real `a`, `b` (at the harness's values `c + d·x`
/// crosses `π` at `x = 5.16`).  Rubi 4.1.1.1; 0.37: unevaluated (with `x`
/// for `c + d·x`: the uncorrected form, which drops by `2π/√(a² − b²)` there);
/// SymPy: a closed form that jumps.  mpmath: `quad(lambda t:
/// 1/(6/5+3/4*sin(5/3+2/7*t)), linspace(-12, 12, 97))` =
/// 27.81303450701655584596.
#[test]
fn parametric_denominator_is_continuous() {
    check(
        "1/(a + b*sin(c + d*x))",
        "x",
        -12.0,
        12.0,
        (27.813_034_507_016_556, 0.0),
    );
}

/// Odd in `cos u`: `t = sin u`, no pole of the substitution, so no
/// correction whatever `a`, `b` are.  Rubi 4.1.1.2; 0.37: unevaluated;
/// SymPy: a closed form.  mpmath: `quad(lambda t:
/// cos(5/3+2/7*t)**3/(6/5+3/4*sin(5/3+2/7*t))**2, linspace(-12, 12, 97))` =
/// 0.07045417472959147738032.
#[test]
fn odd_in_cosine_through_sine() {
    check(
        "cos(c + d*x)^3/(a + b*sin(c + d*x))^2",
        "x",
        -12.0,
        12.0,
        (0.070_454_174_729_591_48, 0.0),
    );
}

/// `tan` in a rational function (0.37: not a rational function of `sin`,
/// `cos` for the half-angle route): `t = tan x`, across `x = π/2`.  0.37:
/// unevaluated; SymPy: a closed form.  mpmath: `quad(lambda t: 1/(2+3*tan(t)),
/// linspace(1, 2.2, 5))` = −0.06009097932038125275602.
#[test]
fn rational_in_tangent() {
    check(
        "1/(2 + 3*tan(x))",
        "x",
        1.0,
        2.2,
        (-0.060_090_979_320_381_25, 0.0),
    );
}

/// `i` in the coefficients: the integrand in `t = tan u` by real and
/// imaginary parts, the parameters `c`, `d` term by term (the jump of each
/// part is decided); the integrand is complex but continuous across `tan
/// u`'s pole at `x = 0.16`.  Rubi 4.3.2.1; 0.37: unevaluated; SymPy: a
/// closed form.  mpmath: `quad(lambda t: (5/3+2/7*tan(11/9+13/6*t))/(6/5 +
/// 6j/5*tan(11/9+13/6*t)), linspace(-1, 1, 9))` = 1.584402758189470665948 −
/// 0.007787744900017497738702i.
#[test]
fn gaussian_coefficients_through_tangent() {
    check(
        "(c + d*tan(e_ + f*x))/(a + I*a*tan(e_ + f*x))",
        "x",
        -1.0,
        1.0,
        (1.584_402_758_189_470_7, -0.007_787_744_900_017_498),
    );
}

/// A rational function with Gaussian coefficients (the rational integrators
/// work over ℚ and ℚ(parameters)).  0.37: unevaluated; SymPy: a closed form.
/// mpmath: `quad(lambda t: (1+1j*t)**5/(1+t**2)**6, linspace(-2, 3, 21))` =
/// 0.0967364111518441123598 − 0.000587333333333333333i.
#[test]
fn gaussian_rational_function() {
    check(
        "(1 + I*t)^5/(1 + t^2)^6",
        "t",
        -2.0,
        3.0,
        (0.096_736_411_151_844_11, -0.000_587_333_333_333_333_3),
    );
}

/// Parameters only in the numerator, term by term: `A·∫ 1/(1 + cos x)² +
/// B·∫ cos x/(1 + cos x)²` (Rubi 4.2.2.1 has it over `a + a·cos(c + d·x)`,
/// unevaluated in 0.37); SymPy: a closed form.  mpmath: `quad(lambda t:
/// (19/10+23/12*cos(t))/(1+cos(t))**2, linspace(-3, 3, 25))` =
/// 38.24226399625058836361.
#[test]
fn numerator_parameters_term_by_term() {
    check(
        "(A + B*cos(x))/(1 + cos(x))^2",
        "x",
        -3.0,
        3.0,
        (38.242_263_996_250_59, 0.0),
    );
}

/// The jump of `ln(tan u + a/b)` at a pole of `tan u` (`+iπ` from `+∞` to
/// `−∞`), with the logarithms written `ln u` as the stage exit writes them
/// for parameters not declared real: before the fix the answer stepped by
/// `−iπ·b/(d·(a² + b²))` at `x = −0.34`.  Rubi 4.3.2.1; 0.37: unevaluated;
/// SymPy: a closed form.  mpmath: `quad(lambda t:
/// 1/tan(5/3+2/7*t)**4/(6/5+3/4*tan(5/3+2/7*t)), linspace(-5.7, 1.5, 29))` =
/// 16892.46376089135445807.
#[test]
fn logarithm_steps_at_tangent_poles() {
    check(
        "cot(c + d*x)^4/(a + b*tan(c + d*x))",
        "x",
        -5.7,
        1.5,
        (16_892.463_760_891_354, 0.0),
    );
}

/// A cubic in `tan u`: a `RootSum` over a polynomial with parameters, its
/// jump by Newton's power sums and `sign(im(ρ))`.  Rubi 4.3.7; 0.37:
/// unevaluated; SymPy: no answer in 25 s.  mpmath: `quad(lambda t:
/// 1/(6/5+3/4*tan(5/3+2/7*t)**3), linspace(-6, 1.5, 31))` =
/// 2.498313835360929587587.
#[test]
fn root_sum_over_a_parametric_polynomial() {
    check(
        "1/(a + b*tan(c + d*x)^3)",
        "x",
        -6.0,
        1.5,
        (2.498_313_835_360_929_6, 0.0),
    );
}

/// Algebraic in `tan u`: `t = tan u`, then a radical substitution; real on
/// one side of the poles of `tan u` only, so no correction there.  Rubi
/// 4.3.0 (`√(b·tan(c + d·x))`) and 4.3.2.1; 0.37: both unevaluated; SymPy:
/// both unevaluated.  mpmath: `quad(lambda t: sqrt(3/4*tan(5/3+2/7*t)),
/// linspace(-5.5, -0.5, 21))` = 5.359989085203508374233;
/// `sqrt(1+tan(t))*tan(t)**5` on `[−0.7, 1.2]` = 15.1927605864581774136.
#[test]
fn algebraic_in_tangent() {
    check(
        "sqrt(b*tan(c + d*x))",
        "x",
        -5.5,
        -0.5,
        (5.359_989_085_203_508, 0.0),
    );
    check(
        "sqrt(1 + tan(x))*tan(x)^5",
        "x",
        -0.7,
        1.2,
        (15.192_760_586_458_177, 0.0),
    );
}

/// A radical of `sin u` times an odd power of `cos u`: `t = sin u`.  Rubi
/// 4.1.1.2; 0.37: unevaluated; SymPy: no answer in 25 s.  mpmath:
/// `quad(lambda t: cos(5/3+2/7*t)**5*sqrt(6/5+3/4*sin(5/3+2/7*t)),
/// linspace(-12, 12, 97))` = 0.001154945690369820430443.
#[test]
fn radical_of_sine_through_sine() {
    check(
        "cos(c + d*x)^5*sqrt(a + b*sin(c + d*x))",
        "x",
        -12.0,
        12.0,
        (0.001_154_945_690_369_820_4, 0.0),
    );
}

/// A polynomial times trigonometric functions of `a + b·x`: `x = (u −
/// a)/b`, multiple angles expanded (`csc u·sin 3u = 3 − 4·sin²u`), by parts
/// on the powers of a polynomial (`(2x + 1)²` was not one).  Rubi 4.7.3;
/// 0.37: all three unevaluated; SymPy: closed forms for the first two, the
/// third unevaluated.  mpmath: `quad(lambda t:
/// (5/3+2/7*t)**2*cos(6/5+3/4*t)**2*sin(6/5+3/4*t)**3, linspace(-6, 6, 49))`
/// = −1.887986910247791276204; `(1+2*t)**2*sin(t)` on `[−4, 4]` =
/// 14.86217590517215525747; `(5/3+2/7*t)*sin(18/5+9/4*t)/sin(6/5+3/4*t)` on
/// `[−1.5, 2.5]` = 6.570444733942239538395.
#[test]
fn polynomial_times_trigonometric() {
    check(
        "(c + d*x)^2*cos(a + b*x)^2*sin(a + b*x)^3",
        "x",
        -6.0,
        6.0,
        (-1.887_986_910_247_791_3, 0.0),
    );
    check(
        "(1 + 2*x)^2*sin(x)",
        "x",
        -4.0,
        4.0,
        (14.862_175_905_172_155, 0.0),
    );
    check(
        "(c + d*x)*csc(a + b*x)*sin(3*a + 3*b*x)",
        "x",
        -1.5,
        2.5,
        (6.570_444_733_942_24, 0.0),
    );
}

/// A radical of `1 ± sin u` (`1 ± cos u`) through the half angle: `√(1 +
/// sin x) = √2·|sin(x/2 + π/4)|`, `∫ = −2·cos x/√(1 + sin x) + 4√2·⌊(x/2 +
/// π/4)/π⌋` (Rubi's answer is the first term and drops by `4√2` at every
/// zero of `1 + sin x`; here `x = −π/2`, `3π/2` are in `[−3, 7]`).  Rubi
/// 4.1.1.1, 4.1.1.2, 4.2.1.1; 0.37: unevaluated; SymPy: unevaluated (no
/// answer in 25 s for the second).  mpmath, `quad` split every 0.25 and at
/// the kinks of the integrands (the zeros of the radicands): `sqrt(1+sin(t))`
/// on `[−3, 7]` = 8.005897203238892729197 (without the split at `−π/2`,
/// `3π/2` `quad` gives 8.0058972970, off by `10⁻⁷`); `(6/5 +
/// 6/5*sin(5/3+2/7*t))**3.5` on `[−12, 12]` = 137.0627445298283831387;
/// `cos(5/3+2/7*t)**2*sqrt(6/5+6/5*sin(5/3+2/7*t))` on `[−12, 12]` =
/// 11.58224275938594607288; `sqrt(6/5-6/5*cos(5/3+2/7*t))` on `[−12, 12]` =
/// 23.77414689002286322969.
#[test]
fn half_angle_radicals_are_continuous() {
    check(
        "sqrt(1 + sin(x))",
        "x",
        -3.0,
        7.0,
        (8.005_897_203_238_893, 0.0),
    );
    check(
        "(a + a*sin(c + d*x))^(7/2)",
        "x",
        -12.0,
        12.0,
        (137.062_744_529_828_4, 0.0),
    );
    check(
        "cos(c + d*x)^2*sqrt(a + a*sin(c + d*x))",
        "x",
        -12.0,
        12.0,
        (11.582_242_759_385_946, 0.0),
    );
    check(
        "sqrt(a - a*cos(c + d*x))",
        "x",
        -12.0,
        12.0,
        (23.774_146_890_022_863, 0.0),
    );
}

/// A pole of the half-angle form that the numerator cancels: `cos⁶u/√(a +
/// a·sin u)` is continuous where `1 + sin u = 0`, so the floor term is
/// needed there too (the first version took it for a pole: `F(12) −
/// F(−12)` had the wrong sign).  Rubi 4.1.1.2; 0.37: unevaluated; SymPy: no
/// answer in 25 s.  mpmath: `quad(lambda t: (1-sin(5/3+2/7*t))**3*(1 +
/// sin(5/3+2/7*t))**2.5/sqrt(6/5), …)` on `[−12, 12]`, split at the zeros of
/// `1 + sin`, = 6.678828082257296212332.
#[test]
fn half_angle_pole_cancelled_by_the_numerator() {
    check(
        "cos(c + d*x)^6/sqrt(a + a*sin(c + d*x))",
        "x",
        -12.0,
        12.0,
        (6.678_828_082_257_296, 0.0),
    );
}

/// A radical of a constant times an even power of the variable: `√(b·x⁴) =
/// C·x²` with `C = √(b·x⁴)·x⁻²` locally constant (`C′ = 0`).  The Möbius
/// substitution `s = x/√(b·x⁴)` gave an answer that stepped by `π·√b` at
/// `x = 0` (3.0924 for 0.3717); through `t = tan u` the Rubi entry
/// `√(b·tan⁴(c + d·x))` (4.3.0) inherited the step at the zeros of `tan`.
/// SymPy: unevaluated for both.  mpmath: `quad(lambda t:
/// sqrt(3/4*t**4)/(1+t**2), [-1, 0, 1])` = 0.3717012843932139055819;
/// `sqrt(3/4*tan(5/3+2/7*t)**4)` on `[−11, −1]` (split at the zero of `tan`,
/// `x = −35/6`) = 39.05716961431088926185.
#[test]
fn radicals_of_even_powers_are_continuous() {
    check(
        "sqrt(b*x^4)/(1 + x^2)",
        "x",
        -1.0,
        1.0,
        (0.371_701_284_393_213_9, 0.0),
    );
    check(
        "sqrt(b*tan(c + d*x)^4)",
        "x",
        -11.0,
        -1.0,
        (39.057_169_614_310_89, 0.0),
    );
}

/// The existing routes with parameters not declared real now take the
/// jump as for real ones too: up to 0.37 `∫ dx/(a + b·cos x)` (`t = tan(x/2)`)
/// dropped by `2π/√(a² − b²)` at `x = π`, and `∫ tan x/(a + tan x) dx` (`u = tan
/// x`) stepped by `iπ·a/(a² + 1)` at `x = π/2` (the continuity hunter's 118
/// jumps).  SymPy: closed forms that jump.  mpmath: `quad(lambda t:
/// 1/(6/5+3/4*cos(t)), linspace(2, 4.5, 11))` = 4.190075357288625563945;
/// `tan(t)/(6/5+tan(t))` on `[1, 2]` = 1.044485974411679514531.
#[test]
fn parametric_jumps_in_the_existing_routes() {
    check(
        "1/(a + b*cos(x))",
        "x",
        2.0,
        4.5,
        (4.190_075_357_288_626, 0.0),
    );
    check(
        "tan(x)/(a + tan(x))",
        "x",
        1.0,
        2.0,
        (1.044_485_974_411_679_5, 0.0),
    );
}

/// `sec`-powers over `a + a·sec`: the content `a` and the numerator
/// parameters split off, so each part's jump is decided.  Rubi 4.5.4.2;
/// 0.37: unevaluated; SymPy: unevaluated.  mpmath: `quad(lambda t:
/// (1/cos(t))**4*(19/10+23/12/cos(t)+29/14/cos(t)**2)/(6/5+6/5/cos(t)),
/// linspace(-1.4, 1.4, 12))` = 1096.527103286874972403.
#[test]
fn secant_family_with_content() {
    check(
        "sec(x)^4*(A + B*sec(x) + C*sec(x)^2)/(a + a*sec(x))",
        "x",
        -1.4,
        1.4,
        (1_096.527_103_286_875, 0.0),
    );
}
