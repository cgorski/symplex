//! Numerical evaluation at extreme parameters: Bessel functions of huge
//! order below the exponent range and what brings them back into it, the
//! uniform (Debye) expansions of large order, `Li_{−n}` beyond `n = 1000`
//! and Jacobi polynomials beyond degree 2000.  Every reference value cites
//! its mpmath 1.3 call (the same digits at both precisions named).

use symplex::prelude::*;

fn dec(ctx: &Context, s: &str) -> Result<String, SymplexError> {
    ctx.parse(s).unwrap().eval_decimal(16)
}

fn check(ctx: &Context, cases: &[(&str, &str)]) {
    for (src, want) in cases {
        match dec(ctx, src) {
            Ok(v) => assert_eq!(v, *want, "{src}"),
            Err(e) => panic!("{src}: {e:?}"),
        }
    }
}

fn refused_nonzero_underflow(ctx: &Context, src: &str) {
    match dec(ctx, src) {
        Err(SymplexError::Unevaluable { reason }) => {
            assert!(reason.contains("not 0 but underflows"), "{src}: {reason}");
        }
        other => panic!("{src}: {other:?}"),
    }
}

/// `J_ν(x)` for `0 < x < ν` has no zero (the first positive zero exceeds
/// `ν`), so a value below the exponent range is a nonzero number, next to
/// which an exact 1 is 1.  Before: `1 ± besselj(10⁹, 1)` was refused ("not
/// 0 but underflows") where the Bessel function arose.  mpmath
/// (`mp.dps = 30` and 50): `log(besselj(10**9, 1))` = `-20416413028.786927917…`,
/// `J₁₀⁹(1) ≈ 10^(−8866735518.66)`.  A sum of two such values of one sign is
/// still refused, as nonzero; an identical difference is 0 by construction.
#[test]
fn besselj_below_the_range_next_to_an_exact_one() {
    let ctx = Context::new();
    check(
        &ctx,
        &[
            ("1 + besselj(10^9, 1)", "1"),
            ("1 - besselj(10^9, 1)", "1"),
            ("2 - besselj(10^9 + 1, -1)", "2"),
            ("1 + besselj(-10^9, 1)", "1"),
            ("besselj(10^9, 1) - besselj(10^9, 1)", "0"),
        ],
    );
    refused_nonzero_underflow(&ctx, "besselj(10^9, 1)");
    // A negative integer order (`J_{−n} = (−1)ⁿ J_n`): not an exact 0 (a
    // development version printed `0`; the extreme-parameter hunter).
    refused_nonzero_underflow(&ctx, "besselj(-3423610157, -281)");
    refused_nonzero_underflow(&ctx, "besselj(-10^20, 1)");
    refused_nonzero_underflow(&ctx, "besselj(10^9, 1) + besselj(10^9, 2)");
    refused_nonzero_underflow(&ctx, "besselj(10^9, 1) - besselj(10^9, 2)");
}

/// A `J_ν` or `I_ν` below the range is held scaled, `e^L·S` with
/// `L = ν·ln(abs(x)/2) − ln Γ(ν + 1)`, so its logarithm and its quotients
/// come back into range.  Before: refused (`log(besseli(…))` as
/// `PrecisionExhausted`).  mpmath (`mp.dps = 30` and 50, the leading term
/// in log space and `hyp0f1` for the series):
/// `nu*log(x/2) - loggamma(nu+1) + log(hyp0f1(nu+1, -x**2/4))` =
/// `-787890140073434963.24…` (`ν = 8345185991999992`, `x = 61/10²⁷`, the
/// series `1 − 1.1·10⁻⁶⁷`), `-20416413028.786927917…` (`J₁₀⁹(1)`),
/// `-20416413028.786927917…` (`I₁₀⁹(1)`, `+x**2/4`),
/// `-20069839438.506955263` (`x = √2`), `-20416413050.203340936`
/// (`−J_{10⁹+1}(−1)`); `besselj(10**9, 1)/besselj(10**9, 2)` =
/// `2.1677979692427824769e-301029996`,
/// `besselj(10**9 + 1, 1)/besselj(10**9, 1)` = `4.9999999950000000062e-10`;
/// a small order at an inexact argument below the `f64` range: `nu*log(x/2)
/// - loggamma(nu+1)` with `nu = 5000`, `x = mpf(10)**-1000000` =
/// `-11512966521.849640097`.
#[test]
fn bessel_values_below_the_range_scaled() {
    let ctx = Context::new();
    check(
        &ctx,
        &[
            (
                "log(besselj(8345185991999992, 61/10^27))",
                "-787890140073435000",
            ),
            ("log(besselj(10^9, 1))", "-20416413028.78693"),
            ("log(besseli(10^9, 1))", "-20416413028.78693"),
            ("log(besselj(10^9, sqrt(2)))", "-20069839438.50696"),
            ("log(-besselj(10^9 + 1, -1))", "-20416413050.20334"),
            ("log(besselj(-10^9, 1))", "-20416413028.78693"),
            ("log(-besselj(-(10^9 + 1), 1))", "-20416413050.20334"),
            (
                "besselj(10^9, 1)/besselj(10^9, 2)",
                "2.167797969242782e-301029996",
            ),
            ("besselj(10^9 + 1, 1)/besselj(10^9, 1)", "4.999999995e-10"),
            ("log(besselj(5000, 10^(-1000000)))", "-11512966521.84964"),
        ],
    );
}

/// A quotient of values below the range: the reciprocal is above it (a
/// power with no value of its own), the product in range.  Before:
/// refused ("its reciprocal overflows it").  mpmath (`mp.dps = 30` and
/// 50): `erfc(mpf(10)**5)/erfc(mpf(10)**5 + 1)` =
/// `2.1413096549419642483e+86859`, `ei(-mpf(10)**10)/ei(-mpf(10)**10 - 1)` =
/// `2.7182818287308734182`.
#[test]
fn quotients_of_values_below_the_range() {
    let ctx = Context::new();
    check(
        &ctx,
        &[
            ("erfc(10^5)/erfc(10^5 + 1)", "2.141309654941964e86859"),
            ("Ei(-10^10)/Ei(-10^10 - 1)", "2.718281828730873"),
        ],
    );
}

/// A factor above the exponent range in a product with one below it: an
/// `exp` that overflows is scaled like one that underflows, and the product
/// comes back.  Before: refused ("exp: the value (about 2^1.443e10)
/// overflows").  mpmath (`mp.dps = 40` and 60):
/// `exp(-mpf(10)**10)/exp(-mpf(10)**10 + 1)` = `0.3678794411714423215955238`,
/// `erfc(mpf(10)**8)*mpf(10)**8*exp(mpf(10)**16)` =
/// `0.564189583547756258738600274173`.
#[test]
fn products_across_the_exponent_range() {
    let ctx = Context::new();
    check(
        &ctx,
        &[
            ("exp(-10^10)/exp(-10^10 + 1)", "0.3678794411714423"),
            ("erfc(10^8)*10^8*exp(10^16)", "0.5641895835477563"),
        ],
    );
}

/// Bessel functions of large order away from the turning point `x = ν`
/// by the uniform expansions of DLMF 10.41 (and 10.19 for `J`, `Y`) with
/// Olver's error bounds, and `Y_n` by its Neumann sum while that sum's
/// terms rise first (`x² > 2(n − 1)`).  Before: refused (`bessely`,
/// `besselk`: "the Neumann series would take n terms"; the others
/// `PrecisionExhausted`, `besseli(10⁶, 10⁶)` "more than 5e4 terms").
/// mpmath (`mp.dps = 30` and 45; `maxprec=400000` for `besselj`, `bessely`):
/// `bessely(20000, 300)` = `-1.3339527347742810646e+33811`, `besselk(20000,
/// 300)` = `2.2082532172661760432e+33810`, `bessely(20000, 30000)` =
/// `0.00056928662735432325425`, `besseli(20000, 30000)` =
/// `1.2335600086142480836e+10226` (dps 45; NoConvergence at 30),
/// `besseli(1000, 120000)` = `3.8867882063036923485e+52110`,
/// `besselj(20000, 19000)` = `1.0287597080905841122e-96`, `besselj(20000,
/// 30000)` = `-0.0053053194445333823534`, `besseli(10**6, 10**6,
/// maxterms=10**7)` = `9.6980872203586648824e+231405`, `besseli(10**6,
/// 5*10**5, maxterms=10**7)` = `4.4445415630123094807e-141411`, `bessely(3909,
/// mpf(3909)/2)` = `-4.633653287822352017e+763` (below the order where the
/// Neumann sum alone served, the full series beyond the cap of its
/// cancellation), `besselj(44907, mpf(2335164)/97)` =
/// `1.9236402772124731081e-7635` (an inexact argument: the bound
/// `abs(J′_ν) ≤ (ν/x)·J_ν` for `x < ν`; the bound by `J_{ν+1}` asked for
/// 5,000 more bits); `besselk(20000, 30000)` (mpmath's `besselk` does not
/// converge there) by `quad` of `∫₀^∞ e^(−x cosh t) cosh(νt) dt` around
/// its peak `sinh t = ν/x` = `1.1241856746565642245e-10231`.
#[test]
fn bessel_functions_of_large_order() {
    let ctx = Context::new();
    check(
        &ctx,
        &[
            ("bessely(20000, 300)", "-1.333952734774281e33811"),
            ("besselk(20000, 300)", "2.208253217266176e33810"),
            ("bessely(20000, 30000)", "0.0005692866273543233"),
            ("besselk(20000, 30000)", "1.124185674656564e-10231"),
            ("besseli(20000, 30000)", "1.233560008614248e10226"),
            ("besseli(1000, 120000)", "3.886788206303692e52110"),
            ("besselj(20000, 19000)", "1.028759708090584e-96"),
            ("besselj(20000, 30000)", "-0.005305319444533382"),
            ("besseli(10^6, 10^6)", "9.698087220358665e231405"),
            ("besseli(10^6, 5*10^5)", "4.444541563012309e-141411"),
            ("bessely(3909, 3909/2)", "-4.633653287822352e763"),
            ("besselj(44907, 2335164/97)", "1.923640277212473e-7635"),
        ],
    );
}

/// At the turning point the expansions do not apply (DLMF 10.20 would):
/// still refused, not guessed.
#[test]
fn bessel_functions_at_the_turning_point_are_refused() {
    let ctx = Context::new();
    for src in ["bessely(20000, 20000)", "besselj(20000, 20000)"] {
        assert!(dec(&ctx, src).is_err(), "{src}");
    }
}

/// `Li_{−n}(z) = n!·Σ_k (2πik − Log z)^(−n−1)` beyond `n = 1000`, where
/// the Stirling sum was refused ("an order below -1000"), and the bound of
/// `∂Li_{−n}/∂z` from the same poles (the series' bound needed 14,000 more
/// bits at `n = 20000`).  References: the exact rational `z·A_n(z)/(1 −
/// z)^(n+1)` with the Eulerian numbers (Python fractions) =
/// `4.0115147284194787039e+4353` (`n = 1500`, `z = 1/2`),
/// `-1.2977319893648465432e+3353` (`z = −1/2`, the two dominant poles
/// conjugate), `1.3683954242887167145e+3367` (`z = −9/10`); mpmath
/// `polylog(-20000, mpf(1)/3)` (`mp.dps = 30` and 50) =
/// `2.1378937870649442049e+76520`.
#[test]
fn polylog_of_large_negative_order() {
    let ctx = Context::new();
    check(
        &ctx,
        &[
            ("polylog(-1500, 1/2)", "4.011514728419479e4353"),
            ("polylog(-1500, -1/2)", "-1.297731989364847e3353"),
            ("polylog(-1500, -9/10)", "1.368395424288717e3367"),
            ("polylog(-20000, 1/3)", "2.137893787064944e76520"),
        ],
    );
}

/// Jacobi polynomials beyond degree 2000 by the explicit sum with its
/// cancellation measured (it was refused: "at most 2000 in evalf"); an
/// odd one with `a = b` is 0 at 0 by symmetry.  mpmath (`mp.dps = 30` and
/// 50): `jacobi(2500, mpf(1)/3, mpf(1)/5, mpf(1)/7)` =
/// `0.017656058952372831225`, `jacobi(5000, …)` = `0.01083267558645746741`.
#[test]
fn jacobi_beyond_degree_2000() {
    let ctx = Context::new();
    check(
        &ctx,
        &[
            ("jacobi(2500, 1/3, 1/5, 1/7)", "0.01765605895237283"),
            ("jacobi(5000, 1/3, 1/5, 1/7)", "0.01083267558645747"),
            ("jacobi(2001, 1/3, 1/3, 0)", "0"),
        ],
    );
}
