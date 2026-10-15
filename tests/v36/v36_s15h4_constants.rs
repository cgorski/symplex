//! Tests of session 15 (s15h4): deciding more constants, summation, codegen binomial.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn d(ctx: &Context, src: &str, digits: u32) -> String {
    ctx.parse(src)
        .unwrap()
        .eval_decimal(digits)
        .unwrap_or_else(|e| panic!("{src}: {e}"))
}

/// `Y_ν(x)` of a non-integer order below the Hankel range was refused ("not
/// yet implemented in small-|x| regime"), so `solve(x·bessely(1/2, 2))` was
/// refused too (D9).  Half-integer orders are `(−1)^{n+1} J_{−ν}`, others
/// `(J_ν cos νπ − J_{−ν})/sin νπ` (DLMF 10.2.3).  Oracle: mpmath
/// `bessely(0.5, 2) = 0.234785710406248469174`, `bessely(-2.5, 2) =
/// 0.223924531468915765845`, `bessely(1/3, 2) = 0.343199966260344342261`.
#[test]
fn bessely_of_non_integer_order_has_a_value() {
    let ctx = Context::new();
    assert_eq!(d(&ctx, "bessely(1/2, 2)", 20), "0.23478571040624846917");
    assert_eq!(d(&ctx, "bessely(-5/2, 2)", 20), "0.22392453146891576584");
    assert_eq!(d(&ctx, "bessely(1/3, 2)", 20), "0.34319996626034434226");
    let x = ctx.symbol("x");
    let c = ctx.parse("bessely(1/2, 2)").unwrap();
    let roots = (&x * &c).solve(&x).unwrap();
    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0].to_string(), "0");
}

/// `K(m)`, `E(m)`, `F(φ | m)` for a real `m > 1` (complex values, DLMF
/// 19.7.3) and for a non-real `m` were refused ("parameter m must satisfy
/// m ≤ 1 for a real value", "of complex argument not yet supported").  On
/// the cut the value is the limit from below, as mpmath and SymPy take it.
/// Oracle: mpmath `ellipk(2) = 1.311028777146059905 − 1.311028777146059905j`,
/// `ellipe(2) = 0.599070117367796104 + 0.599070117367796104j`,
/// `ellipe(log(3)) = 0.901712866818290376 + 0.0747505822497619008j`,
/// `ellipf(1, 2) = 1.311028777146059905 − 0.657163418648656243j`,
/// `ellipf(3, 2) = 2.479505124703953227 − 2.622057554292119810j`,
/// `ellipk(1+2j) = 1.239697448106242801 + 0.564993098757537605j`,
/// `ellipe(1j) = 1.632411781440426111 − 0.369219492375499072j`.
#[test]
fn elliptic_integrals_off_the_real_range() {
    let ctx = Context::new();
    assert_eq!(
        d(&ctx, "elliptic_k(2)", 18),
        "1.31102877714605991 - 1.31102877714605991*i"
    );
    assert_eq!(
        d(&ctx, "elliptic_e(2)", 18),
        "0.599070117367796104 + 0.599070117367796104*i"
    );
    assert_eq!(
        d(&ctx, "elliptic_e(log(3))", 18),
        "0.901712866818290376 + 0.0747505822497619008*i"
    );
    assert_eq!(
        d(&ctx, "elliptic_f(1, 2)", 18),
        "1.31102877714605991 - 0.657163418648656243*i"
    );
    assert_eq!(
        d(&ctx, "elliptic_f(3, 2)", 18),
        "2.47950512470395323 - 2.62205755429211981*i"
    );
    assert_eq!(
        d(&ctx, "elliptic_k(1 + 2*I)", 19),
        "1.239697448106242801 + 0.5649930987575376047*i"
    );
    assert_eq!(
        d(&ctx, "elliptic_e(I)", 18),
        "1.63241178144042611 - 0.369219492375499072*i"
    );
    // A constant factor with a value: `x·E(ln 3) = 0` has the root 0
    // (refused at 0.53: "cannot decide whether the constant factor …").
    let x = ctx.symbol("x");
    let c = ctx.parse("elliptic_e(log(3))").unwrap();
    assert_eq!((&x * &c).solve(&x).unwrap().len(), 1);
    // `Π` off its real range stays refused.
    assert!(
        ctx.parse("elliptic_pi(1/2, 2)")
            .unwrap()
            .eval_decimal(10)
            .is_err()
    );
}

/// The orthogonal polynomials of a non-integer degree were refused ("degree
/// must be a non-negative integer"); they are the hypergeometric functions
/// of DLMF 15.9.1, 18.5.7 (`T_ν(x) = cos(ν acos x)`).  Oracle: mpmath
/// `chebyt(1/2, 1/3) = 0.816496580927726033`, `chebyu(1/2, 1/3) =
/// 1.02062072615965754`, `legendre(1/2, 1/3) = 0.717288174072501923`,
/// `legendre(1/2, 52) = 6.49363326886087940`, `jacobi(1/2, 1, 2, 1/3) =
/// 0.817365326509562823`, `laguerre(1/2, 0, 1/3) = 0.825982014242468867`,
/// `laguerre(1/2, 1, 1/3) = 1.37137683060465698`, `hermite(1/2, 1/3) =
/// 0.999277732557345304`, `chebyt(1/2, 3) = √2`.  (SymPy 1.14 gives 1 for
/// this `jacobi` and `assoc_laguerre`: wrong.)  A real `x ≤ −1`, where the
/// value is not real, stays refused.
#[test]
fn orthogonal_polynomials_of_non_integer_degree() {
    let ctx = Context::new();
    for (src, want) in [
        ("chebyshev_t(1/2, 1/3)", "0.816496580927726033"),
        ("chebyshev_u(1/2, 1/3)", "1.02062072615965754"),
        ("legendre(1/2, 1/3)", "0.717288174072501923"),
        ("jacobi(1/2, 1, 2, 1/3)", "0.817365326509562823"),
        ("laguerre(1/2, 1/3)", "0.825982014242468867"),
        ("assoc_laguerre(1/2, 1, 1/3)", "1.37137683060465698"),
        ("hermite(1/2, 1/3)", "0.999277732557345304"),
        ("chebyshev_t(1/2, 3)", "1.41421356237309505"),
    ] {
        assert_eq!(d(&ctx, src, 18), want, "{src}");
    }
    assert_eq!(d(&ctx, "legendre(1/2, 52)", 19), "6.493633268860879395");
    assert!(
        ctx.parse("legendre(1/2, -3)")
            .unwrap()
            .eval_decimal(10)
            .is_err()
    );
}

/// `factorial2` of a non-integer was refused; its continuation
/// `2^(x/2)·(π/2)^((cos πx − 1)/4)·Γ(x/2 + 1)` takes the integer values `eval`
/// folds.  Oracle: mpmath `fac2(0.5) = 0.962827782446417548`,
/// `fac2(-3.5) = -1.283770376595223397`; `factorial2(-3) = -1` (SymPy).
#[test]
fn factorial2_of_a_non_integer() {
    let ctx = Context::new();
    assert_eq!(d(&ctx, "factorial2(1/2)", 18), "0.962827782446417548");
    assert_eq!(d(&ctx, "factorial2(-7/2)", 19), "-1.283770376595223397");
    assert_eq!(
        ctx.parse("factorial2(-3)").unwrap().eval().to_string(),
        "-1"
    );
    // `compile` agrees (it gave NaN for a non-integer).
    let n = ctx.symbol("n");
    let f = n.factorial2().compile(&["n"]).unwrap();
    let v = f.call(&[0.5]);
    assert!((v - 0.962_827_782_446_417_5).abs() < 1e-14, "{v}");
    assert_eq!(f.call(&[-3.0]), -1.0);
    assert!(f.call(&[-2.0]).is_nan());
}

/// `partition_count(−1)` and the Stirling numbers `S(8, −1)`, `s(8, −1)` stayed
/// unevaluated, so `solve(x·partition_count(−1) + …)` could not decide the
/// constant.  Oracle: SymPy `partition(-1) = 0`; no partition of `n ≥ 0`
/// elements has a negative number of blocks or cycles (Graham–Knuth–
/// Patashnik §6.1).  A non-integer argument stays unevaluated.
#[test]
fn counting_functions_at_negative_arguments() {
    let ctx = Context::new();
    for src in [
        "partition_count(-1)",
        "partition_count(-7)",
        "stirling2(8, -1)",
        "stirling1(8, -3)",
    ] {
        assert_eq!(ctx.parse(src).unwrap().eval().to_string(), "0", "{src}");
    }
    assert_eq!(
        ctx.parse("stirling2(1/2, 1)").unwrap().eval().to_string(),
        "stirling2(1/2, 1)"
    );
}

/// `Σ_{k≥0} C(k + c, c)/2ᵏ` for an integer `c` was `(1/2)^(−c−1)`, which is
/// `1/4` at `c = −3`, where every term `C(k − 3, −3)` is 0 (GKP, D4): the
/// shape `C(k + c, c)` was read as `C(k + c, k)`, a symmetry that needs
/// `c ≥ 0`.  Oracle: exact — the terms at `c = −3` are 0, and
/// `Σ C(k + 2, 2)/2ᵏ = (1 − 1/2)^(−3) = 8`.
#[test]
fn binomial_series_with_an_upper_shift_needs_a_nonnegative_shift() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let c = ctx.symbol_with("c", &[Assumption::Integer]).unwrap();
    let half = ctx.rational(1, 2);
    let body = half.pow(&k) * (&k + &c).binomial(&c);
    let s = body.summation(&k, &ctx.int(0), &ctx.infinity());
    assert_eq!(s.subs_i64(&c, -3).eval().to_string(), "0");
    assert_eq!(s.subs_i64(&c, 2).eval().to_string(), "8");
    // A shift known non-negative, or negative, needs no case split.
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx
        .symbol_with("n", &[Assumption::Integer, Assumption::Positive])
        .unwrap();
    let third = ctx.rational(1, 3);
    let shift = &n - 1;
    let body = third.pow(&k) * (&k + &shift).binomial(&shift);
    let s = body.summation(&k, &ctx.int(0), &ctx.infinity());
    assert_eq!(s.to_string(), "(2/3)^(-n)");
    let m = ctx
        .symbol_with("m", &[Assumption::Integer, Assumption::Negative])
        .unwrap();
    let body = third.pow(&k) * (&k + &m).binomial(&m);
    assert_eq!(
        body.summation(&k, &ctx.int(0), &ctx.infinity()).to_string(),
        "0"
    );
}

/// D2: an infinite sum with a symbolic ratio whose convergence the
/// assumptions do not decide stayed unevaluated; it is the closed form
/// under its exact convergence region.  Oracle: `Σ xᵏ = 1/(1 − x)` for
/// `|x| < 1`, `Σ k xᵏ = x/(1 − x)²` for `|x| < 1`, `Σ_{k≥1} xᵏ/k² = Li₂(x)`
/// for `|x| ≤ 1` (`Li₂(1) = π²/6`), `Σ_{k≥1} xᵏ/k = −ln(1 − x)` for `|x| ≤ 1,
/// x ≠ 1` (SymPy 1.14 gives the same Piecewise conditions).
#[test]
fn symbolic_ratio_sums_carry_their_convergence_region() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let x = ctx.symbol("x");
    let (zero, one, oo) = (ctx.int(0), ctx.int(1), ctx.infinity());
    let geo = x.pow(&k).summation(&k, &zero, &oo);
    assert_eq!(
        geo.to_string(),
        "Piecewise(1/(-x + 1) if 1 > abs(x), Sum(x^k, k=0..oo) if True)"
    );
    assert_eq!(geo.subs(&x, &ctx.rational(1, 2)).eval().to_string(), "2");
    assert_eq!(geo.subs(&x, &ctx.int(2)).eval().to_string(), "oo");
    let li2 = (x.pow(&k) / k.powi(2)).summation(&k, &one, &oo);
    assert_eq!(li2.subs(&x, &one).eval().to_string(), "1/6*pi^2");
    let ln = (x.pow(&k) / &k).summation(&k, &one, &oo);
    assert!(ln.to_string().contains("1 >= abs(x) & x != 1"), "{ln}");
    assert_eq!(
        ln.subs(&x, &ctx.int(-1)).eval().to_string(),
        (-&ctx.int(2).ln()).to_string()
    );
    let arith = (&k * x.pow(&k)).summation(&k, &zero, &oo);
    assert_eq!(
        arith.subs(&x, &ctx.rational(1, 3)).eval().to_string(),
        "3/4"
    );
    // `try_summation` refuses a conditional closed form; `eval` of the
    // formal sum keeps it.
    assert!(x.pow(&k).try_summation(&k, &zero, &oo).is_err());
    let formal = Ex::symbolic_sum(&x.pow(&k), &k, &zero, &oo);
    assert_eq!(formal.eval().to_string(), "Sum(x^k, k=0..oo)");
    // Decided by the assumptions: the closed form, no Piecewise.
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let r = ctx.one() / (&a + 1);
    assert_eq!(r.pow(&k).summation(&k, &zero, &oo).to_string(), "(a + 1)/a");
}

/// An elliptic integral whose parameter is infinite only to the working
/// precision (the reciprocal of a difference that cancels) is retried at a
/// higher precision.  Found by `fuzz_evalf` on the wave-9 merge: the
/// parameter of `elliptic_k(1/(sin(1 + 10⁻⁶⁰) − sin 1))` is `+∞` at
/// 16 digits, the complex duplication ran on `nan` and reported "Carlson
/// R_F did not converge", while 60 digits gave the value.
///
/// Oracle: mpmath at 120 and 240 digits, `ellipk(1/(sin(1 + 10**-60) -
/// sin(1)))` = `1.1546179038880049071e-30 − 5.2020892063355175928e-29i`;
/// `ellipe((sin(1 + 10**-80) - sin(1))**-2)` = `4.24e-81 +
/// 1.8508157176809256179e80i` at 240 digits (mpmath's real part at 120
/// digits, `−2.0e-48`, is not converged).
#[test]
fn elliptic_parameter_infinite_only_to_the_working_precision() {
    let ctx = Context::new();
    for digits in [16, 30, 60] {
        let v = d(&ctx, "elliptic_k(1/(sin(1 + 10^(-60)) - sin(1)))", digits);
        assert!(v.starts_with("1.15461790388"), "{digits}: {v}");
        assert!(v.contains("- 5.2020892063"), "{digits}: {v}");
    }
    let z = ctx
        .parse("elliptic_e((sin(1 + 10^(-80)) - sin(1))^(-2))")
        .unwrap()
        .eval_complex64()
        .unwrap();
    assert!(z.re.abs() < 1e-60 && (z.im / 1.850_815_717_680_925_6e80 - 1.0).abs() < 1e-12);
}

/// The Python printer refused `binomial` and pointed to
/// `scipy.special.binom`, which is NaN at `C(−2, −4)` and `C(−3, 2)`, where
/// `compile` and `eval` give 0 and 6 (GKP, D4).  It now emits a helper with
/// that convention (checked against `compile` on a grid of 900 points with
/// Python 3: equal).  NumPy and base Julia, which have no gamma, still
/// refuse, without the SciPy hint.
#[test]
fn python_binomial_follows_the_gkp_convention() {
    let ctx = Context::new();
    let e = ctx.parse("binomial(n, k)").unwrap();
    let py = e.to_python().unwrap();
    assert!(py.starts_with("(lambda n, k: "), "{py}");
    assert!(py.contains("math.prod") && !py.contains("scipy"), "{py}");
    let np = e.to_numpy().unwrap_err().to_string();
    assert!(!np.contains("scipy.special.binom"), "{np}");
    let f = e.compile(&["n", "k"]).unwrap();
    assert_eq!(f.call(&[-2.0, -4.0]), 0.0);
    assert_eq!(f.call(&[-3.0, 2.0]), 6.0);
}
