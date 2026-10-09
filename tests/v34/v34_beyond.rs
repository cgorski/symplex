//! Numerical evaluation beyond the exponent range (0.37): values above it
//! held scaled (`evalf/extended.rs`, `Ext::Above`), and the distance of a
//! function from its limit that a cancellation leaves.  Every value was
//! refused (or printed `0`) before; each cites its oracle (mpmath 1.3 with
//! unbounded exponents, at the precisions given, agreeing at both).

use symplex::prelude::*;

fn dec(ctx: &Context, src: &str, digits: u32) -> Result<String, SymplexError> {
    ctx.parse(src).unwrap().eval_decimal(digits)
}

/// Refused with a reason that contains `what`, at 16 and 30 digits.
fn assert_refused(ctx: &Context, src: &str, what: &str) {
    for digits in [16, 30] {
        match dec(ctx, src, digits) {
            Err(SymplexError::Unevaluable { reason }) if reason.contains(what) => {}
            other => panic!("{src} at {digits} digits: {other:?}"),
        }
    }
}

/// Before: "exp: the value (about 2^1.443e10) overflows the arbitrary-precision
/// exponent range".  A sum, quotient or logarithm of values above the range
/// comes back into it; `e^g` with an exact `g` is a monomial whose
/// reciprocal is `e^(−g)`, so the quotient cancels exactly.
///
/// mpmath `mp.dps=50` (and 100), `g = mpf(10)**10`:
/// `(exp(g)-exp(g-1))/exp(g)` = `0.63212055882855767840447622983853913255418886896823`,
/// `log(exp(g)+exp(g-1))` = `10000000000.313261687518222834048995494967855641915`,
/// `log(cosh(g))` = `9999999999.3068528194400546905827678785418234319245`,
/// `cosh(g+sqrt(2))/sinh(g+sqrt(2))` = `1.0`.
#[test]
fn quotients_and_logarithms_of_values_above_the_range() {
    let ctx = Context::new();
    for (src, digits, want) in [
        (
            "(exp(10^10) - exp(10^10 - 1))/exp(10^10)",
            40,
            "0.6321205588285576784044762298385391325542",
        ),
        (
            "log(exp(10^10) + exp(10^10 - 1))",
            40,
            "10000000000.31326168751822283404899549497",
        ),
        ("cosh(10^10)/sinh(10^10)", 16, "1"),
        ("cosh(10^10+sqrt(2))/sinh(10^10+sqrt(2))", 16, "1"),
        ("cosh(10^10)/exp(10^10)", 16, "0.5"),
        ("log(cosh(10^10))", 30, "9999999999.30685281944005469058"),
        ("exp(10^10)^(1/2)/exp(5*10^9)", 16, "1"),
        ("abs(exp(10^10*(1+I)))/exp(10^10)", 16, "1"),
    ] {
        assert_eq!(dec(&ctx, src, digits).unwrap(), want, "{src}");
    }
}

/// Before: "Gamma: the value (about 2^2.513e9) overflows".  `Γ(x)` above the
/// range is `e^(ln Γ(x))` scaled.
///
/// mpmath `mp.dps=50`: `gamma(mpf(10)**8)/gamma(mpf(10)**8-1)` = `99999999.0`,
/// `log(gamma(mpf(10)**8))` = `1742068066.1038347092762165027891886143045975559156`
/// (= `loggamma(mpf(10)**8)`), `log(factorial(mpf(10)**8))` =
/// `1742068084.5245154532285819749331202517795112167244`,
/// `gamma(mpf(10)**8+sqrt(2))/gamma(mpf(10)**8+sqrt(2)-1)` =
/// `100000000.41421356237309504880168872420969807856967`,
/// `gamma(mpf(10)**8)*exp(-1742068066)` =
/// `1.109417063410153916758811681096523414636698648094`.
#[test]
fn gamma_above_the_range() {
    let ctx = Context::new();
    for (src, digits, want) in [
        ("gamma(10^8)/gamma(10^8 - 1)", 30, "99999999"),
        ("gamma(10^8+1)/gamma(10^8)/10^8", 30, "1"),
        (
            "log(gamma(10^8))",
            40,
            "1742068066.103834709276216502789188614305",
        ),
        (
            "ln(abs(gamma(10^8)))",
            40,
            "1742068066.103834709276216502789188614305",
        ),
        (
            "log(factorial(10^8))",
            30,
            "1742068084.52451545322858197493",
        ),
        (
            "gamma(10^8+sqrt(2))/gamma(10^8+sqrt(2)-1)",
            30,
            "100000000.414213562373095048802",
        ),
        (
            "gamma(10^8)*exp(-1742068066)",
            30,
            "1.1094170634101539167588116811",
        ),
    ] {
        assert_eq!(dec(&ctx, src, digits).unwrap(), want, "{src}");
    }
    // `log(gamma(x)) − loggamma(x)` is 0 (to the precision reached).
    assert_eq!(
        dec(&ctx, "log(gamma(10^8))-loggamma(10^8)", 16).unwrap(),
        "0"
    );
}

/// A value above the range at the root is still refused, with its reason;
/// so is a value below it (`cosh(g) − sinh(g) = e^(−g)`, `log(e^g + 1) − g =
/// ln(1 + e^(−g))`, both nonzero), whose quotients by `e^(−g)` are values.
///
/// mpmath `mp.dps=50`: `(cosh(g)-sinh(g))*exp(g)` = `1.0`,
/// `(log(exp(g)+sqrt(2))-g)*exp(g)` = `1.4142135623730950488016887242096980785696718753769`,
/// `(log(exp(-g)+exp(-2*g))+g)/exp(-g)` = `1.0` (`g = mpf(10)**10`).
#[test]
fn values_outside_the_range_at_the_root_stay_refused() {
    let ctx = Context::new();
    for src in [
        "exp(10^10)",
        "gamma(10^8)",
        "exp(10^10)+sqrt(2)",
        "cosh(10^10)",
    ] {
        assert_refused(
            &ctx,
            src,
            "overflows the arbitrary-precision exponent range",
        );
    }
    for src in [
        "cosh(10^10)-sinh(10^10)",
        "log(exp(10^10)+1)-10^10",
        "(exp(10^10)+sqrt(2))/exp(10^10)-1",
        "cosh(10^10)/exp(10^10)-1/2",
    ] {
        assert_refused(&ctx, src, "not 0 but underflows");
    }
    for (src, want) in [
        ("(cosh(10^10)-sinh(10^10))*exp(10^10)", "1"),
        (
            "(log(exp(10^10)+sqrt(2))-10^10)*exp(10^10)",
            "1.414213562373095",
        ),
        ("(log(exp(-10^10)+exp(-2*10^10))+10^10)/exp(-10^10)", "1"),
    ] {
        assert_eq!(dec(&ctx, src, 16).unwrap(), want, "{src}");
    }
}

/// Before: `erfc(−10⁵) − 2` and `erf(10⁵) − 1` printed `0` (truly
/// `−erfc(10⁵)` = `−5.2348806797540455e-4342944825`, mpmath `mp.dps=20;
/// -erfc(mpf(10)**5)`), and their quotients by `erfc(10⁵)` were refused.
/// `erf(±x)` and `erfc(−x)` keep `erfc(abs(x))` below the range beside their
/// limit.
#[test]
fn erf_next_to_its_limit_below_the_range() {
    let ctx = Context::new();
    for src in ["erfc(-10^5)-2", "erf(10^5)-1", "erf(-10^5)+1"] {
        assert_refused(&ctx, src, "not 0 but underflows");
    }
    for (src, want) in [
        ("(erfc(-10^5)-2)/erfc(10^5)", "-1"),
        ("(erf(10^5)-1)/erfc(10^5)", "-1"),
        ("(erf(-10^5)+1)/erfc(10^5)", "1"),
        ("erfc(-10^5)", "2"),
    ] {
        assert_eq!(dec(&ctx, src, 16).unwrap(), want, "{src}");
    }
}

/// Before: `0` (mpmath at 30 digits prints `0.0` too).  A function at an
/// argument where it is within its rounding of its limit lost the distance
/// to it; a cancellation against the limit left only that distance, and
/// the zero search printed `0`.  The distance is now recorded (a lower bound
/// on it), and the search goes to the precision that shows it.
///
/// mpmath `mp.dps=1300` (and 2600): `tanh(400)-1` = `-7.33574916835537e-348`,
/// `atan(mpf(10)**400)-pi/2` = `-1.0e-400`, `erf(30)-1` =
/// `-2.56465620375611e-393`, `zeta(2000)-1` = `8.70980981621722e-603`,
/// `altzeta(2000)-1` = `-8.70980981621722e-603`,
/// `polylog(2000,mpf(1)/2)-mpf(1)/2` = `2.1774524540543e-603`,
/// `asinh(mpf(10)**400)-log(2*mpf(10)**400)` = `2.5e-801`.
#[test]
fn the_distance_of_a_function_from_its_limit_is_not_lost() {
    let ctx = Context::new();
    for (src, want) in [
        ("tanh(400)-1", "-7.33574916835537e-348"),
        ("1/tanh(400)-1", "7.33574916835537e-348"),
        ("atan(10^400)-pi/2", "-1e-400"),
        ("erf(30)-1", "-2.56465620375611e-393"),
        ("erfc(-30)-2", "-2.56465620375611e-393"),
        ("zeta(2000)-1", "8.70980981621722e-603"),
        ("dirichlet_eta(2000)-1", "-8.70980981621722e-603"),
        ("polylog(2000, 1/2)-1/2", "2.1774524540543e-603"),
        ("asinh(10^400)-log(2*10^400)", "2.5e-801"),
    ] {
        assert_eq!(dec(&ctx, src, 15).unwrap(), want, "{src}");
    }
    // Below the range: refused (the distance, `2·e^(−2·10¹⁰)`, is not 0).
    assert_refused(&ctx, "tanh(10^10)-1", "not known to be 0");
}

/// Before: refused — `PrecisionExhausted` for `J` at the turning point (the
/// ascending series cancels about `0.33·ν` bits, the uniform expansions stop
/// `≈ 25·ν^(1/3)` short of it), "the Neumann series would take n terms" for
/// `Y`, and a negative non-integer order had no large-order method at all.
/// Now the three-term recurrence in the order from where the expansions
/// apply, with a bound from the cross product `J_{ν+1}Y_ν − J_νY_{ν+1} =
/// 2/(πx)`, Landau's bound on `J` and the growth of `J² + Y²` with the order,
/// and `J_{−ν} = cos(νπ)J_ν − sin(νπ)Y_ν`.
///
/// mpmath `mp.dps=45` (and 30) with `maxprec=100000, maxterms=10**6`:
/// `besselj(20000,20000)` = `0.0164789421069740836052225183309627939092364176`,
/// `bessely(20000,20000)` = `-0.0285423663639909349942827546485257845939641093`,
/// `bessely(20000,20100)` = `-0.00738171129289654561109352436822053230411315526`,
/// `besselj(55754,56311)` = `0.00323219823724633064318605614179860884926386238`,
/// `besselj(20000+mpf(1)/3, 20000*(1-mpf(1)/100))` =
/// `4.23783131673096545554694177696690709317257678e-11`,
/// `besselj(-(1000+mpf(1)/3), 2000)` = `0.0190744588065900558048802802019342376476632592`,
/// `bessely(-(1000+mpf(1)/3), 2000)` = `-0.0019386635770522586048217846837956505136662301`;
/// `mp.dps=40`: `bessely(20000,14800)` = `-7.791968993334282070938810513811185148365e+1238`.
#[test]
fn bessel_functions_at_the_turning_point() {
    let ctx = Context::new();
    for (src, digits, want) in [
        (
            "besselj(20000, 20000)",
            30,
            "0.016478942106974083605222518331",
        ),
        (
            "bessely(20000, 20000)",
            30,
            "-0.0285423663639909349942827546485",
        ),
        ("bessely(20000, 20100)", 18, "-0.00738171129289654561"),
        ("besselj(55754, 56311)", 18, "0.00323219823724633064"),
        (
            "bessely(20000, 14800)",
            30,
            "-7.79196899333428207093881051381e1238",
        ),
        (
            "besselj(20000+1/3, 20000*(1-1/100))",
            18,
            "4.23783131673096546e-11",
        ),
        (
            "besselj(-(1000+1/3), 2000)",
            30,
            "0.0190744588065900558048802802019",
        ),
        (
            "bessely(-(1000+1/3), 2000)",
            30,
            "-0.0019386635770522586048217846838",
        ),
        // The cross product (DLMF 10.5.4), from four values.
        (
            "(besselj(20001, 20000)*bessely(20000, 20000) - besselj(20000, 20000)*bessely(20001, 20000))*pi*10000",
            30,
            "1",
        ),
    ] {
        assert_eq!(dec(&ctx, src, digits).unwrap(), want, "{src}");
    }
}

/// Before: `polylog(300, 3)` took 0.36 s (release), `polylog(1000, 19/10)`,
/// `polylog(10000, 3)` and `polylog(1038, −9/10)` were refused, and the
/// imaginary part on the cut of a large order printed `0`: `im(polylog(300,
/// 3))` is `−π·(ln 3)²⁹⁹/Γ(300)` (the expansions had it only to `2^−wp` of the
/// real part).  A large order takes the expansion in `ln z`, stopped by a
/// bound on its rest, and the imaginary part on the cut its closed form.
///
/// mpmath: `mp.dps=50; -pi*log(3)**299/gamma(300)` = `-5.02295632264378e-600`
/// (`polylog(300,3).imag` agrees at `mp.dps=200` — at 20–100 digits mpmath
/// doubles it); `mp.dps=1500; polylog(300, mpf(19)/10).imag` =
/// `-8.16365076190089e-670`; `mp.dps=400` (and 800): `re(polylog(1000,3))-3`
/// = `8.39937256652897e-301`, `polylog(1038, mpf(-9)/10)+mpf(9)/10` =
/// `2.75010654509099e-313`.
#[test]
fn polylog_of_a_large_order() {
    let ctx = Context::new();
    for (src, want) in [
        ("polylog(300, 3)", "3"),
        ("polylog(10000, 3)", "3"),
        ("polylog(1000, 19/10)", "1.9"),
        ("im(polylog(300, 3))", "-5.02295632264378e-600"),
        ("im(polylog(300, 19/10))", "-8.16365076190089e-670"),
        ("polylog(1000, 3) - 3", "8.39937256652897e-301"),
        ("polylog(1038, -9/10) + 9/10", "2.75010654509099e-313"),
    ] {
        assert_eq!(dec(&ctx, src, 15).unwrap(), want, "{src}");
    }
}

/// A cancellation against the limit `z` of a large order: refused before
/// (`PrecisionExhausted`, the inversion's cancellation), and in development
/// of the faster expansion printed `0`, its distance from `z` lost below the
/// error; now a lower bound on `abs(Li_s(z) − z)` (`polylog::tail_lower_bound`,
/// also off the unit disc and for a complex `z`) sends the zero search to it.
///
/// mpmath `mp.dps=500` (and 800): `polylog(1270, -3)+3` =
/// `4.42739240976443e-382`; `mp.dps=1300` (and 1800): `polylog(3598,
/// mpc(3,4)/5) - mpc(3,4)/5` = `(-2.19398489040487e-1084 +
/// 7.52223390995955e-1084j)`.
#[test]
fn polylog_next_to_its_limit() {
    let ctx = Context::new();
    for (src, want) in [
        ("polylog(1270, -3) + 3", "4.42739240976443e-382"),
        (
            "polylog(3598, (3+4*I)/5) - (3+4*I)/5",
            "-2.19398489040487e-1084 + 7.52223390995955e-1084*i",
        ),
    ] {
        assert_eq!(dec(&ctx, src, 15).unwrap(), want, "{src}");
    }
}

/// Before: "polylog of complex order not yet supported in evalf".  Inside
/// the unit disc the defining series, with the bound on its tail of the real
/// part of the order.
///
/// mpmath `mp.dps=30` (and 60): `polylog(mpc(1.5,0.5), 0.5)` =
/// `(0.612640388900115358817932002617 - 0.0510321042589037241448767526801j)`,
/// `polylog(mpc(mpf(5)/3,1), mpf(1)/2)` =
/// `(0.570339795883659347916654308767 - 0.0776119881152776414877100509693j)`.
#[test]
fn polylog_of_a_complex_order() {
    let ctx = Context::new();
    for (src, want) in [
        (
            "polylog(3/2 + I/2, 1/2)",
            "0.612640388900115358817932002617 - 0.0510321042589037241448767526801*i",
        ),
        (
            "polylog(5/3 + I, 1/2)",
            "0.570339795883659347916654308767 - 0.0776119881152776414877100509693*i",
        ),
    ] {
        assert_eq!(dec(&ctx, src, 30).unwrap(), want, "{src}");
    }
    // Outside the disc: still refused.
    assert_refused(&ctx, "polylog(3/2 + I/2, 2)", "not yet supported");
}
