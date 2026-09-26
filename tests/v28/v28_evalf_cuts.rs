//! Branch cuts and honest digits in the arbitrary-precision evaluator
//! (`evalf`, 0.29): each value pinned against an mpmath 1.3 oracle quoted
//! next to it (inputs exact: `h = exp(mpf(10)**-30) - 1 - mpf(10)**-30`,
//! `mp.dps = 100` so that mpmath does not cancel `h` itself).
//!
//! `h ≈ 5·10⁻⁶¹` is exactly positive but cancels to a rounding residue at
//! the working precision of a 20-digit evaluation, so it "hides" on one
//! side of a branch cut or the other.

use symplex::prelude::*;

/// `exp(10⁻³⁰) − 1 − 10⁻³⁰ ≈ +5·10⁻⁶¹`, exactly positive.
const H: &str = "(exp(10^(-30)) - 1 - 10^(-30))";

fn eval(ctx: &Context, s: &str, digits: u32) -> Result<String, SymplexError> {
    ctx.parse(s)
        .unwrap_or_else(|e| panic!("parse {s}: {e:?}"))
        .eval_decimal(digits)
}

fn check(ctx: &Context, s: &str, want: &str) {
    match eval(ctx, s, 20) {
        Ok(got) => assert_eq!(got, want, "{s}"),
        Err(e) => panic!("{s}: {e:?}, want {want}"),
    }
}

// ── The negative real axis: arg, ln, sqrt, z^p, atan2 ───────────────────────

/// Before: every one of these came out on the `+` side of the cut (`arg`
/// `+π`, `√ = 2i`, the cube root `1 + 1.73i`): a value had a single error
/// bound, so an imaginary part `−h` that cancelled to a rounding residue
/// was indistinguishable from an exact 0, and the residue's sign (or the
/// principal-value convention for 0) picked the side.  Now each part has
/// its own bound, the side is taken from the sign of a part outside its
/// error ball, and the evaluation is repeated at a higher precision until
/// it is.
#[test]
fn a_hidden_imaginary_part_decides_the_negative_real_axis_cut() {
    let ctx = Context::new();
    let cases = [
        // mpmath: arg(-1 - 1j*h) = -3.141592653589793238463
        (format!("arg(-1 - I*{H})"), "-3.1415926535897932385"),
        // mpmath: arg(-1 + 1j*h) = 3.141592653589793238463
        (format!("arg(-1 + I*{H})"), "3.1415926535897932385"),
        // mpmath: log(-1 - 1j*h) = (1.25e-121 - 3.141592653589793238463j)
        (format!("log(-1 - I*{H})"), "-3.1415926535897932385*i"),
        // mpmath: sqrt(-4 - 1j*h) = (1.25e-61 - 2.0j)
        (format!("sqrt(-4 - I*{H})"), "-2*i"),
        // mpmath: sqrt(-4 + 1j*h) = (1.25e-61 + 2.0j)
        (format!("sqrt(-4 + I*{H})"), "2*i"),
        // mpmath: power(-8 - 1j*h, mpf(1)/3) = (1.0 - 1.732050807568877293527j)
        (format!("(-8 - I*{H})^(1/3)"), "1 - 1.7320508075688772935*i"),
        // mpmath: power(-8 + 1j*h, mpf(1)/3) = (1.0 + 1.732050807568877293527j)
        (format!("(-8 + I*{H})^(1/3)"), "1 + 1.7320508075688772935*i"),
        // mpmath: power(-8 - 1j*h, mpf(1)/5)
        //         = (1.226240460962557348606 - 0.8909158444501954645929j)
        (
            format!("(-8 - I*{H})^(1/5)"),
            "1.2262404609625573486 - 0.89091584445019546459*i",
        ),
        // mpmath: atan2(-h, -1) = -3.141592653589793238463
        (format!("atan2(-{H}, -1)"), "-3.1415926535897932385"),
    ];
    for (s, want) in &cases {
        check(&ctx, s, want);
    }
}

// ── The other cuts: asin, acos, atanh, acosh (real axis), atan, asinh ──────

/// Before: `asin(2 + i·h)` came out `π/2 − 1.317i` (the conjugate),
/// `atanh(2 + i·h)` `… − iπ/2`, `acosh(−2 − i·h)` `… + iπ`, `acosh(½ −
/// i·h)` `+1.047i`, and across the imaginary-axis cuts `atan(2i − h)` and
/// `asinh(2i − h)` had the wrong sign of their real parts — the part
/// perpendicular to each cut decides its side.
#[test]
fn a_hidden_perpendicular_part_decides_the_inverse_function_cuts() {
    let ctx = Context::new();
    let cases = [
        // mpmath: asin(2 + 1j*h) = (1.570796326794896619231 + 1.316957896924816708625j)
        (
            format!("asin(2 + I*{H})"),
            "1.5707963267948966192 + 1.3169578969248167086*i",
        ),
        // mpmath: asin(2 - 1j*h) = (1.570796326794896619231 - 1.316957896924816708625j)
        (
            format!("asin(2 - I*{H})"),
            "1.5707963267948966192 - 1.3169578969248167086*i",
        ),
        // mpmath: atanh(2 + 1j*h) = (0.5493061443340548456976 + 1.570796326794896619231j)
        (
            format!("atanh(2 + I*{H})"),
            "0.5493061443340548457 + 1.5707963267948966192*i",
        ),
        // mpmath: acosh(-2 - 1j*h) = (1.316957896924816708625 - 3.141592653589793238463j)
        (
            format!("acosh(-2 - I*{H})"),
            "1.3169578969248167086 - 3.1415926535897932385*i",
        ),
        // mpmath: acosh(mpf(1)/2 - 1j*h) = (5.773502691896257645091e-61 - 1.047197551196597746154j)
        (format!("acosh(1/2 - I*{H})"), "-1.0471975511965977462*i"),
        // mpmath: atan(2j - h) = (-1.570796326794896619231 + 0.5493061443340548456976j)
        (
            format!("atan(2*I - {H})"),
            "-1.5707963267948966192 + 0.5493061443340548457*i",
        ),
        // mpmath: atan(2j + h) = (1.570796326794896619231 + 0.5493061443340548456976j)
        (
            format!("atan(2*I + {H})"),
            "1.5707963267948966192 + 0.5493061443340548457*i",
        ),
        // mpmath: asinh(2j - h) = (-1.316957896924816708625 + 1.570796326794896619231j)
        (
            format!("asinh(2*I - {H})"),
            "-1.3169578969248167086 + 1.5707963267948966192*i",
        ),
    ];
    for (s, want) in &cases {
        check(&ctx, s, want);
    }
}

// ── Exactly real arguments: the principal value, as before ─────────────────

/// An argument whose imaginary part is exactly 0 — an exact rational, or a
/// real expression (`cos 2`, `−π`) whose imaginary part is 0 by structure,
/// not by cancellation — lies on the cut, and takes the principal value
/// (continuous counter-clockwise), with no refinement and no refusal.
#[test]
fn exactly_real_arguments_on_a_cut_take_the_principal_value() {
    let ctx = Context::new();
    let cases = [
        // mpmath: log(-2) = (0.6931471805599453094172 + 3.141592653589793238463j)
        (
            "log(-2)",
            "0.69314718055994530942 + 3.1415926535897932385*i",
        ),
        // mpmath: sqrt(mpc(-4)) = (0.0 + 2.0j)
        ("sqrt(-4)", "2*i"),
        // mpmath: arg(-1) = 3.141592653589793238463
        ("arg(-1)", "3.1415926535897932385"),
        // mpmath: power(mpc(-8), mpf(1)/3) = (1.0 + 1.732050807568877293527j)
        ("(-8)^(1/3)", "1 + 1.7320508075688772935*i"),
        // mpmath: acos(mpc(2)) = (0.0 + 1.316957896924816708625j)
        ("acos(2)", "1.3169578969248167086*i"),
        // mpmath: asin(mpc(-2)) = (-1.570796326794896619231 + 1.316957896924816708625j)
        (
            "asin(-2)",
            "-1.5707963267948966192 + 1.3169578969248167086*i",
        ),
        // mpmath: atanh(mpc(2)) = (0.5493061443340548456976 - 1.570796326794896619231j)
        (
            "atanh(2)",
            "0.5493061443340548457 - 1.5707963267948966192*i",
        ),
        // mpmath: acosh(mpc(-2)) = (1.316957896924816708625 + 3.141592653589793238463j)
        (
            "acosh(-2)",
            "1.3169578969248167086 + 3.1415926535897932385*i",
        ),
        // mpmath: atan(2j) = (1.570796326794896619231 + 0.5493061443340548456976j)
        (
            "atan(2*I)",
            "1.5707963267948966192 + 0.5493061443340548457*i",
        ),
        // mpmath: asinh(2j) = (1.316957896924816708625 + 1.570796326794896619231j)
        (
            "asinh(2*I)",
            "1.3169578969248167086 + 1.5707963267948966192*i",
        ),
        // mpmath: log(mpc(cos(2))) = (-0.8767171085319084452359 + 3.141592653589793238463j)
        (
            "log(cos(2))",
            "-0.87671710853190844524 + 3.1415926535897932385*i",
        ),
        // mpmath: power(mpc(-pi), pi) = (-32.91385774189387817826 - 15.68971165343317064139j)
        (
            "(-pi)^pi",
            "-32.913857741893878178 - 15.689711653433170641*i",
        ),
    ];
    for (s, want) in cases {
        check(&ctx, s, want);
    }
}

/// An imaginary part that is exactly 0 only by an identity the numerics
/// cannot see (`sin²1 + cos²1 − 1`) cancels to within its error of 0 at
/// every precision: the side of the cut is undecidable, and the value is
/// refused.  Before: `ln(−1 − i·(sin²1 + cos²1 − 1))` came out `−iπ` (the
/// true argument is exactly −1, whose logarithm is `+iπ`) — the sign of a
/// rounding residue.
#[test]
fn an_undecidable_side_of_a_cut_is_refused_not_guessed() {
    let ctx = Context::new();
    for s in [
        "log(-1 - I*(sin(1)^2 + cos(1)^2 - 1))",
        "sqrt(-4 + I*(sin(1)^2 + cos(1)^2 - 1))",
        "arg(-1 + I*(sin(1)^2 + cos(1)^2 - 1))",
        "asin(2 + I*(sin(1)^2 + cos(1)^2 - 1))",
    ] {
        let r = eval(&ctx, s, 20);
        assert!(
            matches!(r, Err(SymplexError::PrecisionExhausted { .. })),
            "{s}: {r:?}"
        );
    }
}

/// A sum or product of complex-conjugate pairs is exactly real by its
/// structure, though its computed imaginary part is a rounding residue:
/// Cardano's formula in the irreducible case, which the integrator emits
/// for the roots of `8x⁴ − x³ + 8x + 8`.  Its square root then takes the
/// principal value instead of being undecidable (without this, per-part
/// bounds alone made four Rubi rational-function integrals unevaluated).
#[test]
fn conjugate_pairs_are_exactly_real() {
    let ctx = Context::new();
    let w = "cbrt(65/1024 + 21/1024*sqrt(87)*I) + cbrt(65/1024 - 21/1024*sqrt(87)*I)";
    // mpmath: mp.dps=50; w = mpf(65)/1024 + 21*sqrt(87)/1024*1j; s = cbrt(w) + cbrt(conj(w))
    //         s = (1.07221840133771950586 + 0.0j)
    check(&ctx, w, "1.0722184013377195059");
    // mpmath: sqrt(mpc(-2*s.real + mpf(1)/128)) = (0.0 + 1.461719638875882043446j)
    check(
        &ctx,
        &format!("sqrt(-2*({w}) + 1/128)"),
        "1.4617196388758820434*i",
    );
    // mpmath: mp.dps=50; sqrt(mpc(-2*cos(1))) = (0.0 + 1.039521337797488135237j)
    check(&ctx, "sqrt(-exp(I) - exp(-I))", "1.0395213377974881352*i");
    // mpmath: mp.dps=30; log(mpc(-5)) = (1.60943791243410037460075933323 + 3.14159265358979323846264338328j)
    check(
        &ctx,
        "log(-(1 + 2*I)*(1 - 2*I))",
        "1.6094379124341003746 + 3.1415926535897932385*i",
    );
}

// ── Near the branch points ────────────────────────────────────────────────────────

/// Before: `acos(1 + h)`, `acos(−1 − h)` and `asin(1 + h)` were
/// `PrecisionExhausted`: the amplification `1/√(1 − z²)` was estimated from
/// `1 − z²` at 64 bits, which is 0 for `z` within `2⁻⁶⁴` of 1, so the
/// bound was unknown at every precision.  It is now computed from the
/// distances to the branch points at full precision, and within a few error
/// radii of a square-root branch point the Hölder bound applies.
#[test]
fn inverse_functions_just_beyond_their_branch_points_are_evaluated() {
    let ctx = Context::new();
    let cases = [
        // mpmath: acos(mpc(1 + h)) = (0.0 + 1.0e-30j)
        (format!("acos(1 + {H})"), "1e-30*i"),
        // mpmath: acos(1 - h) = 1.0e-30
        (format!("acos(1 - {H})"), "1e-30"),
        // mpmath: asin(mpc(1 + h)) = (1.570796326794896619231 - 1.0e-30j)
        // (the imaginary part is below the 20 printed digits)
        (format!("asin(1 + {H})"), "1.5707963267948966192"),
        // mpmath: acosh(mpc(1 - h)) = (0.0 + 1.0e-30j)
        (format!("acosh(1 - {H})"), "1e-30*i"),
        // mpmath: acosh(mpc(-1 - h)) = (1.0e-30 + 3.141592653589793238463j)
        (format!("acosh(-1 - {H})"), "3.1415926535897932385*i"),
        // mpmath: atanh(mpc(1 + h)) = (69.77069997038131582996 - 1.570796326794896619231j)
        (
            format!("atanh(1 + {H})"),
            "69.77069997038131583 - 1.5707963267948966192*i",
        ),
        // mpmath: atan(1j*(1 + h)) = (1.570796326794896619231 + 69.77069997038131582996j)
        (
            format!("atan(I*(1 + {H}))"),
            "1.5707963267948966192 + 69.77069997038131583*i",
        ),
    ];
    for (s, want) in &cases {
        check(&ctx, s, want);
    }
}

// ── Other functions with cuts ───────────────────────────────────────────────

/// Before: `Unevaluable { "Ci(0) is -∞" }` — the argument cancelled to 0 at
/// the working precision, and a domain error at 0 ended the evaluation
/// instead of asking for a higher precision.
#[test]
fn a_domain_error_at_an_inexact_zero_is_refined() {
    let ctx = Context::new();
    // mpmath: ci(-h) = (-138.2710370953011534899 + 3.141592653589793238463j)
    check(
        &ctx,
        &format!("Ci(-{H})"),
        "-138.27103709530115349 + 3.1415926535897932385*i",
    );
}

// ── RootOf: repeated and clustered roots ─────────────────────────────────────

/// Before: every `RootOf` of a polynomial with a repeated root was
/// `PrecisionExhausted` — the inclusion disks of a multiple root overlap, so
/// none was certified.  The roots are now those of the square-free factors,
/// each counted with its multiplicity in the `(re, im)` order.
#[test]
fn rootof_of_a_polynomial_with_repeated_roots() {
    let ctx = Context::new();
    let cases = [
        // mpmath: polyroots([1, 0, -4, 0, 4]) sorted by (re, im)
        //   = [-1.414213562373095048802, -1.414213562373095048802,
        //      1.414213562373095048802, 1.414213562373095048802]
        ("RootOf((x^2 - 2)^2, x, 1)", "-1.4142135623730950488"),
        ("RootOf((x^2 - 2)^2, x, 2)", "1.4142135623730950488"),
        // mpmath: polyroots([1, -1, -3, 5, -2]) = [-2.0, 1.0, 1.0, 1.0]
        ("RootOf((x - 1)^3*(x + 2), x, 0)", "-2"),
        ("RootOf((x - 1)^3*(x + 2), x, 3)", "1"),
        // mpmath: polyroots([1, -3, 2, -6, 1, -3]) = [-1.0j, -1.0j, 1.0j, 1.0j, 3.0]
        ("RootOf((x^2 + 1)^2*(x - 3), x, 1)", "-i"),
        ("RootOf((x^2 + 1)^2*(x - 3), x, 2)", "i"),
        ("RootOf((x^2 + 1)^2*(x - 3), x, 4)", "3"),
    ];
    for (s, want) in cases {
        check(&ctx, s, want);
    }
}

/// Roots with exactly equal real parts are ordered by their imaginary
/// parts, not by the rounding noise in the computed real parts.  Before,
/// the `(re, im)` sort compared the computed real parts of `±i`, `±2i`
/// (rounding residues, not exact zeros), so which root an index named
/// could depend on that noise; the real parts are now certified equal (the
/// roots lie on the symmetry line `Re z = −aₙ₋₁/(n·aₙ)` of an even or odd
/// shifted polynomial, or form a conjugate pair).
#[test]
fn rootof_orders_equal_real_parts_by_imaginary_part() {
    let ctx = Context::new();
    let cases = [
        // mpmath: polyroots([1, 0, 5, 0, 4]) = [-2.0j, -1.0j, 1.0j, 2.0j]
        ("RootOf(x^4 + 5*x^2 + 4, x, 0)", "-2*i"),
        ("RootOf(x^4 + 5*x^2 + 4, x, 1)", "-i"),
        ("RootOf(x^4 + 5*x^2 + 4, x, 3)", "2*i"),
        // mpmath: polyroots([1, -4, 11, -14, 10])
        //   = [(1.0 - 2.0j), (1.0 - 1.0j), (1.0 + 1.0j), (1.0 + 2.0j)]
        ("RootOf((x^2 - 2*x + 2)*(x^2 - 2*x + 5), x, 1)", "1 - i"),
    ];
    for (s, want) in cases {
        check(&ctx, s, want);
    }
}

/// A cluster closer than the default iteration tolerance (`10⁻³⁰`) is
/// separated by refining at a higher precision.
#[test]
fn rootof_separates_a_tight_cluster() {
    let ctx = Context::new();
    // mpmath: mp.dps=200; e = mpf(10)**-60; b = 2 + e; c = 1 + e
    //         (b + sqrt(b*b - 4*c))/2 = 1.000000000000000000000000000000000000000000000000000000000001
    let e = ctx
        .parse("RootOf((x - 1)*(x - 1 - 10^(-60)), x, 1)")
        .unwrap();
    assert_eq!(
        e.eval_decimal(65).unwrap(),
        "1.000000000000000000000000000000000000000000000000000000000001"
    );
}

// ── Polynomially convergent infinite sums ──────────────────────────────────────

fn infinite_sum(ctx: &Context, body: &str, lo: i64) -> Ex {
    let k = ctx.symbol("k");
    ctx.parse(body)
        .unwrap()
        .summation(&k, &ctx.int(lo), &ctx.infinity())
}

/// Before: `Unevaluable` ("converges only polynomially … not implemented")
/// for every infinite sum of a rational term the summation engine could not
/// close.  Such a sum is now the direct sum up to a cut-off plus the tail
/// `Σ aⱼ·ζ(j, N)` from the Laurent expansion of the term, each Hurwitz zeta
/// by Euler–Maclaurin with its proven remainder; an alternating one is paired
/// first.
#[test]
fn polynomially_convergent_rational_sums_evaluate() {
    let ctx = Context::new();
    let cases = [
        // mpmath: mp.dps=50; (1 + pi*coth(pi))/2 = 2.07667404746858117413405079475
        ("1/(k^2 + 1)", 0, "2.07667404746858117413405079475"),
        // mpmath: mp.dps=50; nsum(lambda k: k/(k**4+1), [0, inf])
        //   = 0.69417302215071523475934719403897
        //   (= fsum(f(k) for k < 1000) + sumem(f, [1000, inf]))
        ("k/(k^4 + 1)", 0, "0.694173022150715234759347194039"),
        // mpmath: mp.dps=50; -(1/mpf(3) + pi/sqrt(3)/sinh(pi*sqrt(3)))/2
        //   = -0.17452676963383902546102623598943
        (
            "(-1)^(k + 1)/(k^2 + 3)",
            0,
            "-0.174526769633839025461026235989",
        ),
        // mpmath: mp.dps=50; (1 + pi*coth(pi))/2 + sum(1/(mpf(k)**2+1) for k in range(1,6))
        //   = 2.973959115341884341554865274388
        ("1/(k^2 + 1)", -5, "2.97395911534188434155486527439"),
        // mpmath: mp.dps=50; (mpf(10)**-6 + pi*coth(1000*pi)/1000)/2
        //   = 0.0015712963267948966192313216916398
        ("1/(k^2 + 10^6)", 0, "0.00157129632679489661923132169164"),
    ];
    for (body, lo, want) in cases {
        let s = infinite_sum(&ctx, body, lo);
        assert!(s.to_string().starts_with("Sum("), "{s}");
        assert_eq!(s.eval_decimal(30).unwrap(), want, "{s}");
    }
    // mpmath: mp.dps=50; (1 + pi*coth(pi))/2 = 2.07667404746858117413405079475
    assert_eq!(
        infinite_sum(&ctx, "1/(k^2 + 1)", 0).eval_f64().unwrap(),
        2.076_674_047_468_581
    );
}

/// Before: a term ratio with an irrational or complex constant (`√2^k/k!`,
/// `i^k/k!`) was "not recognised as hypergeometric" and the sum refused.
/// The constant `C` of the ratio `C·P(k)/Q(k)` is now recovered from two
/// consecutive terms (and checked against a third), with its error carried
/// through the recurrence and the geometric tail bound.
#[test]
fn hypergeometric_sums_with_an_irrational_ratio_constant() {
    let ctx = Context::new();
    let cases = [
        // mpmath: mp.dps=50; nsum(lambda k: sqrt(2)**k/(factorial(k)*(k+1)), [0, inf])
        //   = 2.2014004543689957254268797795001 (= (exp(sqrt(2)) - 1)/sqrt(2))
        (
            "sqrt(2)^k/(factorial(k)*(k + 1))",
            "2.2014004543689957254268797795",
        ),
        // mpmath: mp.dps=50; nsum(lambda k: mpc(0,1)**k/(factorial(k)*(k+1)), [0, inf])
        //   = (0.8414709848078965066525023216303 + 0.45969769413186028259906339255702j)
        (
            "I^k/(factorial(k)*(k + 1))",
            "0.84147098480789650665250232163 + 0.459697694131860282599063392557*i",
        ),
        // mpmath: mp.dps=50; nsum(lambda k: (1/sqrt(3))**k/(k**2+1), [0, inf])
        //   = 1.3853746430845447672881478011818
        ("(1/sqrt(3))^k/(k^2 + 1)", "1.38537464308454476728814780118"),
    ];
    for (body, want) in cases {
        let s = infinite_sum(&ctx, body, 0);
        assert!(s.to_string().starts_with("Sum("), "{s}");
        assert_eq!(s.eval_decimal(30).unwrap(), want, "{s}");
    }
    let s = infinite_sum(&ctx, "(sqrt(2) + 1)^k/(k^2 + 1)", 0);
    let r = s.eval_decimal(20);
    assert!(
        matches!(r, Err(SymplexError::Divergent { .. })),
        "{s}: {r:?}"
    );
}

/// A rational term decaying like `1/k` diverges, alternating or not when
/// it does not decay at all; a pole in the range is no value.
#[test]
fn divergent_rational_sums_and_poles_are_errors() {
    let ctx = Context::new();
    for body in ["k/(k^2 + 1)", "(-1)^k*(k^2 + 1)/(k^2 + 2)"] {
        let s = infinite_sum(&ctx, body, 0);
        let r = s.eval_decimal(20);
        assert!(
            matches!(r, Err(SymplexError::Divergent { .. })),
            "{s}: {r:?}"
        );
    }
    let s = infinite_sum(&ctx, "1/((k - 3)*(k^2 + 1))", 0);
    let r = s.eval_decimal(20);
    assert!(
        s.to_string().starts_with("Sum(") && matches!(r, Err(SymplexError::Unevaluable { .. })),
        "{s}: {r:?}"
    );
}

// ── Signs decided structurally ───────────────────────────────────────────────────

/// Before: `Piecewise((1, exp(−4·10⁹) > 0), (0, True))` was
/// `PrecisionExhausted` — `exp(−4·10⁹)` underflows every precision, so the
/// numerical decision was never certain — although `exp` of a real is
/// positive.  `eval` now decides a comparison of a constant with 0 from the
/// sign the assumption engine knows, before any numerics.
#[test]
fn the_sign_of_exp_of_a_real_is_decided_structurally() {
    let ctx = Context::new();
    let tiny = ctx.parse("exp(-4*10^9)").unwrap();
    let (zero, one, two, t) = (ctx.zero(), ctx.one(), ctx.int(2), ctx.bool_true());
    // exp(x) > 0 for real x (no oracle needed: a theorem).
    let pw = Ex::piecewise(&[(&one, &tiny.gt(&zero)), (&zero, &t)]);
    assert_eq!(pw.eval().to_string(), "1");
    assert_eq!(pw.eval_decimal(20).unwrap(), "1");
    let pw = Ex::piecewise(&[(&one, &zero.ge(&tiny)), (&two, &t)]);
    assert_eq!(pw.eval_decimal(20).unwrap(), "2");
    // A sign the structure does not give is still left to certified numerics.
    let s = ctx.parse("sign(exp(-10^20) - exp(-3*10^20))").unwrap();
    assert!(matches!(
        s.eval_decimal(20),
        Err(SymplexError::PrecisionExhausted { .. })
    ));
}

// ── Definite integrals: only the digits the quadrature has ─────────────────

fn x_to_the_x(ctx: &Context) -> Ex {
    let x = ctx.symbol("x");
    x.pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1))
}

fn damped_cosine(ctx: &Context) -> Ex {
    let x = ctx.symbol("x");
    ((&x * 200).cos() * (-&x).exp()).definite_integral_node(&x, &ctx.int(0), &ctx.int(10))
}

/// Before: an `f64` Gauss–Kronrod result was trusted to the working
/// precision less 16 bits, so `eval_decimal(16)` of `∫₀¹ xˣ dx` printed
/// `0.7834305107121128` — the last three digits wrong — and of
/// `∫₀¹⁰ cos(200x)·e⁻ˣ dx` `2.521090543419162e-5` (four wrong).  The bound
/// is now the quadrature's error estimate plus the rounding of the `f64`
/// integrand values (`∝ ∫|f|`, so the cancelling oscillation counts), and
/// digits beyond it are refused.
#[test]
fn definite_integrals_print_only_certified_digits() {
    let ctx = Context::new();
    // mpmath: mp.dps=50; quad(lambda x: x**x, [0, 1]) = 0.7834305107121344070592644
    assert_eq!(x_to_the_x(&ctx).eval_decimal(10).unwrap(), "0.7834305107");
    // mpmath: mp.dps=50; quad(lambda x: cos(200*x)*exp(-x), linspace(0, 10, 400))
    //         = 2.52109054341912827206351e-5
    assert_eq!(damped_cosine(&ctx).eval_decimal(8).unwrap(), "2.5210905e-5");
    for (name, e) in [("x^x", x_to_the_x(&ctx)), ("cos", damped_cosine(&ctx))] {
        let r = e.eval_decimal(16);
        assert!(
            matches!(r, Err(SymplexError::PrecisionExhausted { .. })),
            "{name} at 16 digits: {r:?}"
        );
    }
    // mpmath: mp.dps=50; quad(lambda x: sin(1/x), linspace(mpf(1)/1000, 1, 200))
    //         = 0.504066497877487051711602
    let x = ctx.symbol("x");
    let s = x
        .powi(-1)
        .sin()
        .definite_integral_node(&x, &ctx.rational(1, 1000), &ctx.int(1));
    assert_eq!(s.eval_decimal(12).unwrap(), "0.504066497877");
}

/// `eval_f64` of an expression with a definite integral is still served:
/// the quadrature certifies the digits of `QUADRATURE_F64_DIGITS`.
#[test]
fn definite_integrals_are_still_served_by_eval_f64() {
    let ctx = Context::new();
    // mpmath: mp.dps=50; quad(lambda x: x**x, [0, 1]) = 0.7834305107121344070592644
    let v = x_to_the_x(&ctx).eval_f64().unwrap();
    assert!((v - 0.783_430_510_712_134_4).abs() < 1e-9, "{v}");
    // mpmath: mp.dps=50; quad(lambda x: cos(200*x)*exp(-x), linspace(0, 10, 400))
    //         = 2.52109054341912827206351e-5
    let v = damped_cosine(&ctx).eval_f64().unwrap();
    assert!((v - 2.521_090_543_419_128e-5).abs() < 1e-13, "{v}");
}
