//! After 0.29 — the output hunters: `Display` ⇄ `parse` round trips of
//! API-built expressions, `compile()` against `eval_f64` at random points
//! (with a running error bound), the emitted Python checked the same way,
//! and number theory against SymPy (one disagreement: roots of a
//! polynomial congruence modulo a prime above 2⁶³).
//!
//! Each test says what was wrong before; every reference value cites the
//! oracle call that produced it (mpmath 1.3.0 at `mp.dps = 40`, SymPy 1.14).

// Reference values are quoted at the digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use symplex::prelude::*;

/// `e` displays as `shown` and parses back to the same tree.
fn assert_round_trip(ctx: &Context, e: &Ex, shown: &str) {
    assert_eq!(e.to_string(), shown, "display of {}", e.to_srepr());
    let back = ctx
        .parse(shown)
        .unwrap_or_else(|err| panic!("{shown} does not parse: {err}"));
    assert_eq!(
        back.to_srepr(),
        e.to_srepr(),
        "{shown} parses to a different tree"
    );
}

fn close_rel(got: f64, want: f64, tol: f64) -> bool {
    (got - want).abs() <= tol * want.abs()
}

// ═══════════════════════════════════════════════════════════════════════════
// Display ⇄ parse
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn a_product_chain_parses_as_one_product() {
    // `Mul(-3, x + 1, y + 1)` displays as `-3*(x + 1)*(y + 1)`; the parser
    // folded the chain pairwise, the arena distributes a number over a sum
    // in a two-factor product, and the text came back as
    // `(-3*x - 3)*(y + 1)`.  Likewise `-(a + b)*c` came back as
    // `(-a - b)*c` and `4*(x - 1)*I` as `(4*x - 4)*I`.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = ctx.parse("-3*(x + 1)*(y + 1)").expect("parses");
    assert_eq!(
        e.to_srepr(),
        "Mul(Integer(-3), Add(Integer(1), Symbol('x')), Add(Integer(1), Symbol('y')))"
    );
    assert_round_trip(&ctx, &e, "-3*(x + 1)*(y + 1)");
    let neg = ctx.parse("-(x + 1)*y").expect("parses");
    assert_eq!(
        neg.to_srepr(),
        "Mul(Integer(-1), Symbol('y'), Add(Integer(1), Symbol('x')))"
    );
    assert_round_trip(&ctx, &neg, &neg.to_string());
    // A lone `-(x + 1)` or `2*(x + 1)` still distributes, as in the API.
    assert_eq!(ctx.parse("-(x + 1)").unwrap(), -(&x + 1));
    assert_eq!(ctx.parse("2*(y + 1)").unwrap(), 2 * (&y + 1));
    // The operand of a prefix minus is still tighter than `*`, looser than
    // `^`: `2^-x*y` is `2^(-x)·y` and `-x^2` is `-(x^2)`.
    assert_eq!(ctx.parse("2^-x*y").unwrap(), ctx.int(2).pow(&-&x) * &y);
    assert_eq!(ctx.parse("-x^2").unwrap(), -x.powi(2));
}

#[test]
fn a_parenthesised_divisor_divides_by_each_factor() {
    // `Mul(1/2, x, (y + z)⁻¹)` displays as `x/(2*(y + z))`; the divisor was
    // built first, as `2*y + 2*z`, and the text came back as
    // `x/(2*y + 2*z)`.
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let e = &x * ctx.rational(1, 2) * (&y + &z).powi(-1);
    assert_round_trip(&ctx, &e, "x/(2*(y + z))");
    // 61970997949458179017652487377747171/(182635543817055669182363*(y + assoc_legendre(…)))
    // from the hunter, in miniature:
    let f = ctx.rational(7, 3) * (&y + x.sin()).powi(-1);
    assert_round_trip(&ctx, &f, "7/(3*(y + sin(x)))");
    // The divisor group is not re-canonicalised either: `x*x^I` would merge
    // into `x^(1 + I)` (fuzz_roundtrip, -I/(x*x^I*fresnelc(y))).
    let i = ctx.i_unit();
    let g = -(&i) * x.powi(-1) * x.pow(&i).powi(-1) * y.fresnelc().powi(-1);
    assert_eq!(g.to_string(), "-I/(x*x^I*fresnelc(y))");
    assert_round_trip(&ctx, &g, "-I/(x*x^I*fresnelc(y))");
}

#[test]
fn piecewise_parses_in_its_display_form() {
    // `Piecewise(x if x > 0, -x if True)` is how every Piecewise displays,
    // and the parser had no Piecewise at all ("expected RParen, got Gt"),
    // so no expression containing one round-tripped.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let zero = ctx.int(0);
    let pw = Ex::piecewise(&[
        (&x, &x.gt(&zero).and(&y.lt(&ctx.int(1)))),
        (&(&y + 1), &x.eq_expr(&y).not()),
        (&-&x, &ctx.bool_true()),
    ]);
    let shown = pw.to_string();
    assert_eq!(
        shown,
        "Piecewise(x if x > 0 & 1 > y, y + 1 if !(x == y), -x if True)"
    );
    assert_round_trip(&ctx, &pw, &shown);
    // Nested, and inside a relation (parse_bool).
    let nested = ctx
        .parse("Piecewise(Piecewise(1 if y > 0, 2 if True) if x > 0, 3 if True)")
        .expect("parses");
    assert_round_trip(&ctx, &nested, &nested.to_string());
    let rel = ctx
        .parse_bool("Piecewise(x if x > 0, 0 if True) >= y")
        .expect("parses");
    assert_eq!(ctx.parse_bool(&rel.to_string()).unwrap(), rel);
    // A Boolean value or a missing `if` is an error, not a guess.
    assert!(ctx.parse("Piecewise(x > 0 if True)").is_err());
    assert!(ctx.parse("Piecewise(x, x > 0)").is_err());
}

#[test]
fn heaviside_parses_from_its_display_name() {
    // `Heaviside(x)` displays as `H(x)`, which the parser rejected as an
    // unknown function.  An undefined function the context knows as `H`
    // keeps the name, and `h(x)` is not taken.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let h = (&x - 1).heaviside();
    assert_round_trip(&ctx, &h, "H(x - 1)");
    assert!(ctx.parse("h(x)").is_err());
    let other = Context::new();
    let xo = other.symbol("x");
    let user = other.apply("H", &[&xo]).unwrap();
    assert_eq!(other.parse("H(x)").unwrap(), user);
}

#[test]
fn signed_and_fractional_literals_are_parenthesised() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // factorial(-12/7) displayed as `-12/7!`, which is -12/(7!).
    let f = ctx.rational(-12, 7).factorial();
    assert_round_trip(&ctx, &f, "(-12/7)!");
    let g = ctx.rational(5, 2).factorial() * &x;
    assert_round_trip(&ctx, &g, &g.to_string());
    // (-oo)^(1/3) displayed as `-oo^(1/3)`, which is -(oo^(1/3)) = -oo.
    let r = ctx.parse("373365/377922687763463334998784083").unwrap();
    let p = ctx.neg_infinity().pow(&r);
    assert_round_trip(&ctx, &p, "(-oo)^(373365/377922687763463334998784083)");
}

#[test]
fn a_negative_number_to_the_minus_half_is_not_printed_as_one_over_sqrt() {
    // The arena keeps (-16/7)^(-1/2) as a power but evaluates sqrt(-16/7)
    // to 4/7*sqrt(7)*I, so the display `1/sqrt(-16/7)` re-parsed as
    // -1/4*sqrt(7)*I, a different tree.
    let ctx = Context::new();
    let y = ctx.symbol("y");
    let p = ctx.rational(-16, 7).pow(&ctx.rational(-1, 2));
    assert_round_trip(&ctx, &p, "(-16/7)^(-1/2)");
    let q = &y * ctx.int(-3).pow(&ctx.rational(-1, 2));
    assert_round_trip(&ctx, &q, &q.to_string());
    // Other bases keep the quotient form.
    assert_eq!(y.pow(&ctx.rational(-1, 2)).to_string(), "1/sqrt(y)");
}

// ═══════════════════════════════════════════════════════════════════════════
// compile() against eval_f64
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sign_and_heaviside_of_an_undefined_value_are_undefined() {
    // The VM returned 0 for sign(NaN) and 0.5 for heaviside(NaN): at
    // x = 3, y = 1, sign(sqrt(y - x)) (i·√2 / |…| = i, not real) compiled
    // to 0, and heaviside(asin(x)) to 0.5 at x = 2.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let s = (&y - &x).sqrt().sign().compile(&["x", "y"]).unwrap();
    assert!(s.call(&[3.0, 1.0]).is_nan());
    assert_eq!(s.call(&[1.0, 3.0]), 1.0);
    assert_eq!(s.call(&[1.0, 1.0]), 0.0);
    let h = x.asin().heaviside().compile(&["x"]).unwrap();
    assert!(h.call(&[2.0]).is_nan());
    assert_eq!(h.call(&[0.0]), 0.5);
    // min/max no longer drop an undefined operand (f64::min returns the
    // other one): min(x, sqrt(-1 - x^2)) was x.
    let m = x
        .min_with(&(-1 - x.powi(2)).sqrt())
        .compile(&["x"])
        .unwrap();
    assert!(m.call(&[0.5]).is_nan());
}

#[test]
fn the_emitters_keep_nan_through_sign_and_heaviside() {
    // Python's `math.copysign(1, nan)` is ±1 and the Heaviside chains of
    // Python, Julia, C and Rust ended in their last branch for NaN.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        x.sign().to_python().unwrap(),
        "(0.0 if x == 0 else math.copysign(1, x) if x == x else math.nan)"
    );
    assert_eq!(
        x.heaviside().to_python().unwrap(),
        "(0.0 if x < 0 else (0.5 if x == 0 else (1.0 if x > 0 else math.nan)))"
    );
    let c = x.heaviside().to_c_fn("f", &["x"]).unwrap();
    assert!(c.contains("(x == 0.0 ? 0.5 : x)"), "{c}");
    let r = x.sign().to_rust_fn("f", &["x"]).unwrap();
    assert!(r.contains("else { f64::NAN }"), "{r}");
}

#[test]
fn beta_keeps_its_digits_for_large_arguments() {
    // B(a, b) was exp(lnΓ(a) + lnΓ(b) − lnΓ(a + b)) once a + b ≥ 171: the
    // cancellation of terms of size (a + b)·ln(a + b) cost 1.3e-9 relative
    // at B(5, 666624) and 5e-9 at B(e, 2908160).
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f = x.beta(&y).compile(&["x", "y"]).unwrap();
    // mpmath: beta(5, 666624) = 1.823055964257829699085773595528594998009e-28
    assert!(close_rel(
        f.call(&[5.0, 666_624.0]),
        1.823_055_964_257_829_7e-28,
        1e-14
    ));
    // mpmath: beta(e, 2908160) = 4.219512167453930218088245171913781596233e-18
    assert!(close_rel(
        f.call(&[std::f64::consts::E, 2_908_160.0]),
        4.219_512_167_453_930_2e-18,
        1e-14
    ));
    // mpmath: beta(2863/8, 2863/8) = 6.466533727306182350036515372366560025715e-217
    assert!(close_rel(
        f.call(&[357.875, 357.875]),
        6.466_533_727_306_182_4e-217,
        2e-13
    ));
    // The binomial coefficient goes through B for large n:
    // mpmath: binomial(2001/2, 13/4) = 677360745.3471660739503295433783256980262
    let b = x.binomial(&y).compile(&["x", "y"]).unwrap();
    assert!(close_rel(
        b.call(&[1000.5, 3.25]),
        677_360_745.347_166_07,
        1e-13
    ));
    // A negative n with an integer k > 2000 went through ln Γ (1.7e-11):
    // mpmath: binomial(-3/2, 26656) = 184.2292816997658507616616017231925505473
    assert!(close_rel(
        b.call(&[-1.5, 26_656.0]),
        184.229_281_699_765_85,
        1e-14
    ));
}

#[test]
fn a_falling_factorial_through_zero_is_zero() {
    // falling_factorial(0, 950) = 0·(−1)···(−949): the product ran from the
    // large end, overflowed to inf before reaching the zero factor, and
    // inf·0 was NaN.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f = x.falling_factorial(&y).compile(&["x", "y"]).unwrap();
    assert_eq!(f.call(&[0.0, 950.0]), 0.0);
    let r = x.rising_factorial(&y).compile(&["x", "y"]).unwrap();
    assert_eq!(r.call(&[-949.0, 1900.0]), 0.0);
    assert_eq!(r.call(&[3.0, 4.0]), 360.0);
}

#[test]
fn bessel_functions_of_a_large_argument_keep_their_phase() {
    // The Hankel expansion rounded its phase x − (2n + 1)π/4 in f64, an
    // absolute error of ulp(x): 1e-11 relative at J₂(421888).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let j2 = x.bessel_j(&ctx.int(2)).compile(&["x"]).unwrap();
    // mpmath: besselj(2, 421888) = 0.001129302066648648535998100418303847202091
    assert!(close_rel(
        j2.call(&[421_888.0]),
        0.001_129_302_066_648_648_5,
        1e-14
    ));
    let y1 = x.bessel_y(&ctx.int(1)).compile(&["x"]).unwrap();
    // mpmath: bessely(1, 3922) = -0.0111373981718360467378664897779282181377
    assert!(close_rel(
        y1.call(&[3922.0]),
        -0.011_137_398_171_836_047,
        1e-14
    ));
}

#[test]
fn a_real_constant_compiles_even_through_a_complex_intermediate() {
    // abs(atanh(9)) is real (|0.1257 + iπ/2|) but was lowered as
    // abs(NaN) = NaN, so x + abs(atanh(9))/(x^3 + 1) compiled to NaN
    // everywhere; zeta(3)·x and re(sqrt(-2) + 1)·x did not compile.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let nine = ctx.int(9);
    let f = (&x + nine.atanh().abs() / (x.powi(3) + 1))
        .compile(&["x"])
        .unwrap();
    // mpmath: abs(atanh(9)) = 1.574753746271339719435709904183034024688
    assert!(close_rel(f.call(&[0.0]), 1.574_753_746_271_339_7, 1e-15));
    let z = (ctx.int(3).zeta() * &x).compile(&["x"]).unwrap();
    // mpmath: zeta(3) = 1.202056903159594285399738161511449990765
    assert!(close_rel(z.call(&[1.0]), 1.202_056_903_159_594_3, 1e-15));
    let e = ctx.e();
    let g = (e.abs() - e.asin().sign()).abs().compile(&[]).unwrap();
    // mpmath: abs(e - sign(asin(e))) = 2.156238741505002004309579406323166699423
    assert!(close_rel(g.call(&[]), 2.156_238_741_505_002, 1e-15));
    // A complex constant still has no real value.
    let c = (ctx.int(-2).sqrt() * &x).compile(&["x"]);
    assert!(c.is_err() || c.unwrap().call(&[1.0]).is_nan());
}

#[test]
fn a_huge_rational_constant_is_rounded_not_divided() {
    // numer as f64 / denom as f64 was inf/inf = NaN for 10^400/(10^400 + 1).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let big = ctx.parse("10^400").unwrap();
    let f = (&big / (&big + 1) * &x).compile(&["x"]).unwrap();
    assert_eq!(f.call(&[3.0]), 3.0);
    let g = (&big * &x).compile(&["x"]).unwrap();
    assert_eq!(g.call(&[1.0]), f64::INFINITY);
}

// ═══════════════════════════════════════════════════════════════════════════
// Number theory against SymPy
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn polynomial_congruence_finds_roots_modulo_a_prime_above_2_to_the_63() {
    // Roots modulo a prime p ≥ 2⁶³ need Fₚ[x] beyond machine words, which
    // the general case did not have: they were reported as none.
    use num_bigint::BigInt;
    use symplex::ntheory::polynomial_congruence;
    let big = |s: &str| s.parse::<BigInt>().unwrap();
    let c = |v: &[i64]| -> Vec<BigInt> { v.iter().map(|&t| BigInt::from(t)).collect() };
    // sympy: isprime(697145390090109847836526411) -> True;
    // polynomial_congruence(-2*x**4 + 5*x**3 - 6*x**2 - 15*x - 16,
    //     697145390090109847836526411)
    //   -> [329756319151446431339665577, 400589363968490406928869519]
    assert_eq!(
        polynomial_congruence(
            &c(&[-2, 5, -6, -15, -16]),
            big("697145390090109847836526411")
        ),
        vec![
            big("329756319151446431339665577"),
            big("400589363968490406928869519")
        ]
    );
    // sympy: polynomial_congruence(-10*x**3 - 11*x**2 + 9*x - 20, 6969330228044061141835603)
    //   -> [1566223791104761132620690, 3748562106029372408061526, 6533075490540770400438308]
    assert_eq!(
        polynomial_congruence(&c(&[-10, -11, 9, -20]), big("6969330228044061141835603")),
        vec![
            big("1566223791104761132620690"),
            big("3748562106029372408061526"),
            big("6533075490540770400438308")
        ]
    );
}

#[test]
fn number_theory_agrees_with_sympy_on_the_hunter_edge_cases() {
    // Apart from `polynomial_congruence` above, the hunter (13,931 SymPy
    // cases, exhaustive small ranges) found no disagreement; these are the
    // edge inputs it leaned on.
    use num_bigint::BigInt;
    use symplex::ntheory as nt;
    // sympy: isprime(3317044064679887385961981) -> False (spsp to the first 13 prime bases)
    assert!(!nt::isprime(
        "3317044064679887385961981".parse::<BigInt>().unwrap()
    ));
    // sympy: is_carmichael(512461) -> True
    assert!(nt::is_carmichael(512_461));
    // sympy: factorint(2**64 + 1) -> {274177: 1, 67280421310721: 1}
    let f = nt::factorint(BigInt::from(2u8).pow(64) + 1);
    assert_eq!(
        f,
        vec![
            (BigInt::from(274_177), 1),
            (BigInt::from(67_280_421_310_721_u64), 1)
        ]
    );
    // sympy: sqrt_mod(16, 35, all_roots=True) -> [4, 11, 24, 31]
    let r: Vec<BigInt> = [4, 11, 24, 31].iter().map(|&v| BigInt::from(v)).collect();
    assert_eq!(nt::sqrt_mod_all(16, 35), r);
    // sympy: discrete_log(147878, 67246, 6) raises ValueError ("base is not
    // invertible for the given modulus"), but 6^3955 ≡ 67246 (mod 147878)
    // (python: pow(6, 3955, 147878) -> 67246): the brute-force branch for
    // non-coprime bases finds a solution.
    let x = nt::discrete_log(6, 67_246, 147_878).expect("a solution exists");
    assert_eq!(nt::mod_pow(6, x, 147_878), BigInt::from(67_246));
}
