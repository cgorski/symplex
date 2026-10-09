//! `polylog(s, z)` of a real order at complex, negative and large
//! arguments (before 0.31 every such case was "not yet supported in
//! evalf", for integer orders too), and the pivots of `Matrix` elimination
//! at algebraic numbers that are zero without looking like it.  Reference
//! values cite the mpmath call (exact rational inputs, `mp.dps` 100 and 200
//! agreeing) or the SymPy call.

use symplex::prelude::*;

fn dec(ctx: &Context, src: &str, digits: u32) -> Result<String, SymplexError> {
    ctx.parse(src).unwrap().eval_decimal(digits)
}

/// A non-integer order at a complex argument inside the unit disc was
/// refused ("polylog of complex argument not yet supported in evalf").
///
/// mpmath: `polylog(mpf(3)/2, mpc(mpf(-3)/16, mpf(-5)/18))` →
/// `-0.19639968916150621124023762401751708813128780764387802882054366… −
/// 0.24323109553792615622942695038984248330726714970233503461855511…i`
/// (SymPy: `N(polylog(S(3)/2, -S(3)/16 - 5*I/18), 30)` likewise).
#[test]
fn non_integer_order_at_a_complex_argument() {
    let ctx = Context::new();
    let src = "polylog(3/2, -3/16 - 5/18*I)";
    assert_eq!(
        dec(&ctx, src, 16).unwrap(),
        "-0.1963996891615062 - 0.2432310955379262*i"
    );
    assert_eq!(
        dec(&ctx, src, 60).unwrap(),
        "-0.196399689161506211240237624017517088131287807643878028820544 \
         - 0.243231095537926156229426950389842483307267149702335034618555*i"
    );
    let z = ctx.parse(src).unwrap().eval_complex64().unwrap();
    // Python: `repr(float(v.real)), repr(float(v.imag))`.
    assert_eq!((z.re, z.im), (-0.1963996891615062, -0.24323109553792616));
}

/// On the cut `z > 1` the value is the limit from below (SymPy's and
/// mpmath's convention); an imaginary part that is positive only beyond
/// the first working precision puts the argument above the cut, decided
/// by its certified sign.  Before, all three were refused.
///
/// mpmath: `polylog(mpf(3)/2, 3)` → `0.87749357736198378827292950343711227178…
/// − 3.71558463514058721888265069342972781521…i` (SymPy
/// `N(polylog(S(3)/2, 3), 30)`: `0.877493577361983788272929503437 −
/// 3.71558463514058721888265069343*I`); with `z = 3 + (sin(1 + mpf(10)**-30)
/// − sin(1))·i`: `0.87749357736198378827292950343680771500164412768928474274319…
/// + 3.71558463514058721888265069342942128094637700108396202457244…i`.
#[test]
fn on_the_cut_the_value_is_the_limit_from_below() {
    let ctx = Context::new();
    assert_eq!(
        dec(&ctx, "polylog(3/2, 3)", 16).unwrap(),
        "0.8774935773619838 - 3.715584635140587*i"
    );
    assert_eq!(
        dec(&ctx, "polylog(3/2, 3)", 60).unwrap(),
        "0.877493577361983788272929503437112271784584738951130463205393 \
         - 3.71558463514058721888265069342972781521898709937649975345879*i"
    );
    let above = "polylog(3/2, 3 + (sin(1 + 10^(-30)) - sin(1))*I)";
    assert_eq!(
        dec(&ctx, above, 16).unwrap(),
        "0.8774935773619838 + 3.715584635140587*i"
    );
    assert_eq!(
        dec(&ctx, above, 60).unwrap(),
        "0.8774935773619837882729295034368077150016441276892847427432 \
         + 3.71558463514058721888265069342942128094637700108396202457245*i"
    );
}

/// Left of −1 the value is real, and exactly so (no imaginary part to
/// stand in the way of `eval_f64` or of a branch cut further up).  Before:
/// "polylog outside the unit disc not yet supported in evalf".
///
/// mpmath: `polylog(mpf(3)/2, -3)` → `-1.67908973050482813533749092618276099749…`
/// (`float(re(…))` = `-1.6790897305048282`); `polylog(mpf(5)/2, -1 -
/// mpf(10)**-12)` → `-0.86719988901294928521597250550013381938504348455395183006334…`.
#[test]
fn left_of_minus_one_the_value_is_real() {
    let ctx = Context::new();
    assert_eq!(
        dec(&ctx, "polylog(3/2, -3)", 60).unwrap(),
        "-1.67908973050482813533749092618276099749012139393507251154167"
    );
    let f = ctx.parse("polylog(3/2, -3)").unwrap().eval_f64().unwrap();
    assert_eq!(f, -1.6790897305048282);
    let z = ctx
        .parse("polylog(3/2, -3)")
        .unwrap()
        .eval_complex64()
        .unwrap();
    assert_eq!(z.im, 0.0);
    assert_eq!(
        dec(&ctx, "polylog(5/2, -1 - 10^(-12))", 60).unwrap(),
        "-0.867199889012949285215972505500133819385043484553951830063347"
    );
}

/// Integer orders off the real interval `[−1, 1]` were refused as well.
///
/// mpmath: `polylog(2, 3)` → `2.32018042331309839640619447370310465782660…
/// − 3.45139229522320266143382058381808564515219…i`; `polylog(3, 5j)` →
/// `-1.36164869810397898421403794950321425447897… + 3.77259538354364017968529…i`.
#[test]
fn integer_orders_outside_the_unit_interval() {
    let ctx = Context::new();
    assert_eq!(
        dec(&ctx, "polylog(2, 3)", 60).unwrap(),
        "2.32018042331309839640619447370310465782660471350930766255184 \
         - 3.45139229522320266143382058381808564515219003102569284980437*i"
    );
    assert_eq!(
        dec(&ctx, "polylog(3, 5*I)", 16).unwrap(),
        "-1.361648698103979 + 3.77259538354364*i"
    );
}

/// Large moduli go through the inversion formula with the Hurwitz zeta
/// function.
///
/// mpmath: `polylog(mpf(3)/2, 1000)` → `-12.9421455647893149927145958601439902669…
/// − 9.31694225317435236006770599534810246135…i`; `polylog(mpf(7)/3,
/// mpc(3, 4)*mpf(10)**30)` → `-7432.76340273500128613926181459517013341593077…
/// + 543.592802633775334361819816070069534783794573936352182320780…i`.
#[test]
fn large_moduli() {
    let ctx = Context::new();
    assert_eq!(
        dec(&ctx, "polylog(3/2, 1000)", 16).unwrap(),
        "-12.94214556478931 - 9.316942253174352*i"
    );
    assert_eq!(
        dec(&ctx, "polylog(7/3, 10^30*(3 + 4*I))", 60).unwrap(),
        "-7432.76340273500128613926181459517013341593077122221242881251 \
         + 543.59280263377533436181981607006953478379457393635218232078*i"
    );
}

/// An order next to an integer: the Euler–Maclaurin remainder of the
/// Hurwitz zeta function has the factor `(σ + 1)` with `σ = 1 − s =
/// −1 − 10⁻²⁰`; a first version bounded it in `f64`, where that factor is
/// 0, stopped after one Bernoulli term and was wrong from the 21st digit.
/// Next to `z = 1` the `μ`-expansion takes `ln z` from `|z|² − 1`.
///
/// mpmath: `polylog(2 + mpf(10)**-20, mpc(3, 1))` →
/// `1.34592887082108301278992970053675717210558186055536134788107… +
/// 3.36511040266619428938560044584030089248494981959802185180749…i` (the
/// same at dps 60, 100, 200 and from the inversion formula);
/// `polylog(mpf(1)/2, mpc(1, mpf(10)**-10))` →
/// `125329.953373907930190778010690559076293714555268636421112433846… +
/// 125331.413734683289675545904622289727049125288166052806201851212…i`.
#[test]
fn next_to_an_integer_order_and_next_to_one() {
    let ctx = Context::new();
    assert_eq!(
        dec(&ctx, "polylog(2 + 10^(-20), 3 + I)", 60).unwrap(),
        "1.34592887082108301278992970053675717210558186055536134788107 \
         + 3.36511040266619428938560044584030089248494981959802185180749*i"
    );
    assert_eq!(
        dec(&ctx, "polylog(1/2, 1 + 10^(-10)*I)", 60).unwrap(),
        "125329.953373907930190778010690559076293714555268636421112434 \
         + 125331.413734683289675545904622289727049125288166052806201851*i"
    );
}

/// A complex order was refused here (an error, not a panic or a wrong
/// value) until 0.36.  Since 0.37 a complex order inside the unit disc is the
/// defining series with a bound on its tail (`evalf/polylog.rs`,
/// `polylog_complex_order`): mpmath `mp.dps=30` (and 60)
/// `polylog(mpc(1.5, 0.5), 0.5)` = `(0.612640388900115358817932002617 -
/// 0.0510321042589037241448767526801j)`.  Outside the disc it is still
/// refused.
#[test]
fn complex_order_is_refused() {
    let ctx = Context::new();
    assert_eq!(
        dec(&ctx, "polylog(3/2 + I/2, 1/2)", 16).unwrap(),
        "0.6126403889001154 - 0.05103210425890372*i"
    );
    assert!(dec(&ctx, "polylog(3/2 + I/2, 2)", 16).is_err());
}

// ── Matrix pivots at hidden algebraic zeros ─────────────────────────────────

fn matrix(ctx: &Context, rows: &[&[&str]]) -> Matrix {
    Matrix::new(
        rows.iter()
            .map(|r| r.iter().map(|e| ctx.parse(e).unwrap()).collect())
            .collect(),
    )
    .unwrap()
}

/// `∛(−2) − ∛2·(1/2 + √3·i/2)` is exactly 0 (`(−2)^(1/3) = 2^(1/3)·e^(iπ/3)`),
/// but the pivot test of `rref` only tried the exact test for rational
/// functions over `ℚ(radicals of rationals, i)`, which does not take a
/// cube root of a negative number: `rank` was 2, the null space empty and
/// `lu` returned a factorization with the zero as a pivot (and
/// `[[1, 2], [2, 4 + h]]` below likewise, `h` a nested-radical zero).
///
/// SymPy: `minimal_polynomial(cbrt(-2) - cbrt(2)*(S(1)/2 + sqrt(3)*I/2), x)`
/// → `x`; `Matrix([[that, 0], [0, 1]]).rank()` → `1`.
#[test]
fn pivot_that_is_a_hidden_zero_from_a_cube_root() {
    let ctx = Context::new();
    let h = "cbrt(-2) - cbrt(2)*(1/2 + sqrt(3)*I/2)";
    let m = matrix(&ctx, &[&[h, "0"], &["0", "1"]]);
    assert_eq!(m.rank(), 1);
    let (r, pivots) = m.rref();
    assert_eq!(pivots, vec![1]);
    assert!(r.get(0, 0).is_zero_structural());
    assert_eq!(m.nullspace().len(), 1);
    assert!(m.lu().is_err());
    assert!(m.inv().is_err());
}

/// Hidden zeros from nested radicals: real cube roots of `√5 ± 2` (`φ` and
/// `1/φ`), two different nested square roots (outside the one-nested-root
/// tower of the exact test), and a fourth root of a negative number.
///
/// SymPy: `minimal_polynomial(h, x)` → `x` for `cbrt(sqrt(5) + 2) -
/// cbrt(sqrt(5) - 2) - 1`, `sqrt(3 + 2*sqrt(2)) + sqrt(3 - 2*sqrt(2)) -
/// 2*sqrt(2)` and `(-4)**(S(1)/4) - 1 - I`; `Matrix([[h2, 1], [0, 1]]).rank()`
/// → `1`, `Matrix([[1, 2], [2, 4 + h3]]).rank()` → `1`, `Matrix([[h4, 1, 2],
/// [0, h1, 3], [0, 0, 1]]).rank()` → `2` (null space of dimension 1).
#[test]
fn pivots_that_are_hidden_zeros_from_nested_radicals() {
    let ctx = Context::new();
    let h1 = "cbrt(-2) - cbrt(2)*(1/2 + sqrt(3)*I/2)";
    let h2 = "cbrt(sqrt(5) + 2) - cbrt(sqrt(5) - 2) - 1";
    let h3 = "sqrt(3 + 2*sqrt(2)) + sqrt(3 - 2*sqrt(2)) - 2*sqrt(2)";
    let h4 = "(-4)^(1/4) - 1 - I";
    assert_eq!(matrix(&ctx, &[&[h2, "1"], &["0", "1"]]).rank(), 1);
    let four_plus_h3 = format!("4 + {h3}");
    let m = matrix(&ctx, &[&["1", "2"], &["2", &four_plus_h3]]);
    assert_eq!(m.rank(), 1);
    assert_eq!(m.nullspace().len(), 1);
    assert!(m.lu().is_err());
    assert!(m.inv().is_err());
    let m = matrix(&ctx, &[&[h4, "1", "2"], &["0", h1, "3"], &["0", "0", "1"]]);
    assert_eq!(m.rank(), 2);
    assert_eq!(m.nullspace().len(), 1);
    assert!(m.lu().is_err());
}
