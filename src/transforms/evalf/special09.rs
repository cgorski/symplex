//! More special functions (0.9) (part of [`evalf`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// More special functions (0.9)
// ═══════════════════════════════════════════════════════════════════════════
//
// All routines take real arguments and return real results at `prec` bits;
// complex arguments and points outside the real domain of a function are
// refused with `Unevaluable`.  Working precision is `prec + 32` plus
// problem-dependent guard bits wherever an alternating series or a
// difference of nearly equal quantities loses digits.

pub(super) fn unevaluable(reason: impl Into<String>) -> SymplexError {
    SymplexError::Unevaluable {
        reason: reason.into(),
    }
}

/// The refusal of a value that is infinite at the arguments the route
/// received ([`SymplexError::Infinite`]): a pole or a logarithmic
/// singularity.  The claim stands only where those arguments are exact
/// ([`certified_infinity`]); a route says it only where the function has
/// no finite value or limit there (`Γ(a, 0)` for `Re a < 0`, not `Re a =
/// 0`, where it oscillates).
pub(super) fn infinite(reason: impl Into<String>) -> SymplexError {
    SymplexError::Infinite {
        reason: reason.into(),
    }
}

/// The refusal of `f(a, 0)` for a function that behaves like `z^a` at 0
/// (`Γ(a, z)`, `γ(a, z)`, `E_n(z)` with `a = n − 1`) where `Re a ≤ 0`:
/// infinite for `Re a < 0` and `a = 0` (a logarithm), no value at `Re a = 0`,
/// `a ≠ 0`, where `|z^a|` stays bounded and its argument turns without end.
pub(super) fn singular_unless_oscillating(a: &Complex, reason: &str) -> SymplexError {
    if a.0.is_negative() || (a.0.is_zero() && a.1.is_zero()) {
        infinite(reason)
    } else {
        unevaluable(reason)
    }
}

/// Is `e` a refusal at the point the arguments' values give — no value
/// there ([`SymplexError::Unevaluable`]) or an infinite one
/// ([`SymplexError::Infinite`])?  Met at an argument only rounded onto that
/// point, it says nothing of the true argument (see [`eval_ball`]).
pub(super) fn domain_error(e: &SymplexError) -> bool {
    matches!(
        e,
        SymplexError::Unevaluable { .. } | SymplexError::Infinite { .. }
    )
}

/// `e`, the error of node `node`, with an infinity claim that does not rest
/// on exact arguments read as no value ([`SymplexError::Unevaluable`]):
/// the route saw the arguments' values, and an argument within its error
/// of a pole may lie off it.  `Π(n | 1)` and `Π(1 | m)` are infinite
/// whatever the other argument (`elliptic_pi(π, 1) = −∞`): the argument at
/// 1 must be exact.  The claim stands only for the nodes whose routes say
/// it of their own value ([`own_infinity`]): another route may meet a pole
/// of a function it calls where its own value is finite (`B(−3, 1) = −1/3`
/// through `Γ(−3)/Γ(−2)`, `E(1) = 1`), and a definite integral is never
/// infinite here (a value of its integrand or of an antiderivative at a
/// point is not its value).
pub(super) fn certified_infinity(
    node: &ExprNode,
    e: SymplexError,
    cache: Cache<'_>,
) -> SymplexError {
    let SymplexError::Infinite { reason } = e else {
        return e;
    };
    let exact = |c: &ExprId| cache.get(c).is_some() && cache.err(c).is_exact();
    let certified = own_infinity(node)
        && match node {
            ExprNode::Lib(LibFn::EllipticPi, args) => args.iter().any(|c| {
                exact(c)
                    && cache.get(c).is_some_and(|v| {
                        v.1.is_zero() && v.0.cmp(&BigFloat::from_i32(1, 64)) == Some(0)
                    })
            }),
            _ => node.children().iter().all(exact),
        };
    if certified {
        SymplexError::Infinite { reason }
    } else {
        SymplexError::Unevaluable { reason }
    }
}

/// Does every infinity claim ([`infinite`]) the routes of `node` can make
/// concern the value of `node` itself: `∞`, `−∞`, `zoo`, `0` to a negative
/// power, `Γ`, `x!`, `ln Γ`, `ψ`, `ψ⁽ⁿ⁾` and `ζ` at their poles, `Ci`, `Chi`,
/// `Ei` at 0, `li` at 1, `Γ(s, 0)`, `γ(s, ·)` and `E_n(0)` where they
/// diverge, `Li_s(1)` for `s ≤ 1`, `K(1)`, the complete `Π(n | 1)`,
/// `Π(1 | m)`, and the incomplete beta integrals at a divergent end?
fn own_infinity(node: &ExprNode) -> bool {
    match node {
        ExprNode::Infinity
        | ExprNode::NegInfinity
        | ExprNode::ComplexInfinity
        | ExprNode::Pow(..)
        | ExprNode::Gamma(_)
        | ExprNode::Factorial(_)
        | ExprNode::LogGamma(_)
        | ExprNode::Digamma(_)
        | ExprNode::Polygamma(..)
        | ExprNode::Zeta(_)
        | ExprNode::Ci(_)
        | ExprNode::Ei(_)
        | ExprNode::Li(_) => true,
        ExprNode::Lib(LibFn::EllipticPi, args) => args.len() == 2,
        ExprNode::Lib(f, _) => matches!(
            f,
            LibFn::Chi
                | LibFn::UpperGamma
                | LibFn::LowerGamma
                | LibFn::ExpInt
                | LibFn::PolyLog
                | LibFn::EllipticK
                | LibFn::BetaInc
                | LibFn::BetaIncRegularized
        ),
        _ => false,
    }
}

/// A series exhausted its term budget without meeting the negligibility
/// test; the partial sum is *not* returned.
pub(super) fn series_did_not_converge(operation: &'static str, max_terms: usize) -> SymplexError {
    SymplexError::ComputationFailed {
        operation,
        reason: format!("series did not converge in {max_terms} terms"),
    }
}

/// The cached values of `args`, all required to be real.
fn real_args(name: &str, args: &[ExprId], cache: Cache<'_>) -> Result<Vec<BigFloat>, SymplexError> {
    let mut out = Vec::with_capacity(args.len());
    for &a in args {
        let v = get_cached(cache, a)?;
        if !v.1.is_zero() {
            return Err(unevaluable(format!(
                "{name} of complex argument not yet supported in evalf"
            )));
        }
        out.push(v.0.clone());
    }
    Ok(out)
}

/// `|a| < 2^{-bits} |b|` — has the series term `a` become negligible
/// against the accumulated sum `b`?
pub(super) fn negligible(a: &BigFloat, b: &BigFloat, bits: usize) -> bool {
    if a.is_zero() {
        return true;
    }
    match (a.exponent(), b.exponent()) {
        (Some(ae), Some(be)) => (be as i64 - ae as i64) > bits as i64,
        _ => false,
    }
}

/// `a > b`?
pub(super) fn bf_gt(a: &BigFloat, b: &BigFloat) -> bool {
    a.cmp(b).is_some_and(|c| c > 0)
}

/// `a < b`?
pub(super) fn bf_lt(a: &BigFloat, b: &BigFloat) -> bool {
    a.cmp(b).is_some_and(|c| c < 0)
}

/// A `BigFloat` that is an integer exactly, as an `i64` when it fits.
///
/// Before 0.30 a value within `10⁻¹²` of an integer counted (in `f64`),
/// and `polylog(2 + 10⁻²⁰, 999/1000)` was `Li₂(999/1000)`, wrong from its
/// 20th digit at 30 and 60 digits (the degree and order callers already
/// asked for an exact integer as well).
pub(super) fn bf_as_int(x: &BigFloat) -> Result<Option<i64>, SymplexError> {
    if !(x.is_zero() || x.is_int()) {
        return Ok(None);
    }
    let r = f64_of(x)?.round();
    Ok((r.abs() < 9.0e15).then_some(r as i64))
}

/// Numeric value of the library function `f` on `args` (of the arity `f`
/// declares) inside `eval_node`.
///
/// Exhaustive over [`LibFn`].  The integer sequences and combinatorial
/// counts have no arbitrary-precision routine: `eval` folds them exactly
/// for integer arguments before `evalf` runs, and anything left is an
/// error rather than a guess.
pub(super) fn eval_lib_fn(
    f: LibFn,
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    match f {
        // The orthogonal polynomials of a real non-integer degree
        // (`hypergeom.rs`; refused before 0.54).
        f if hypergeom::takes(f, args, cache) => hypergeom::eval(f, args, cache, w),
        LibFn::BesselJ => eval_bessel(arb_bessel_j, "BesselJ", args, cache, w),
        LibFn::BesselY => eval_bessel(arb_bessel_y, "BesselY", args, cache, w),
        LibFn::BesselI => eval_bessel(arb_bessel_i, "BesselI", args, cache, w),
        LibFn::BesselK => eval_bessel(arb_bessel_k, "BesselK", args, cache, w),
        LibFn::Legendre => eval_orthopoly(OrthoPoly::Legendre, f, args, cache, w),
        LibFn::ChebyshevT => eval_orthopoly(OrthoPoly::ChebyshevT, f, args, cache, w),
        LibFn::ChebyshevU => eval_orthopoly(OrthoPoly::ChebyshevU, f, args, cache, w),
        LibFn::Hermite => eval_orthopoly(OrthoPoly::Hermite, f, args, cache, w),
        LibFn::Laguerre => eval_orthopoly(OrthoPoly::Laguerre, f, args, cache, w),
        // A non-real argument: `(1 − 2^(1−s))·ζ(s)` and `ψ(z + 1) + γ`
        // (`zeta.rs`; refused before 0.44).
        LibFn::DirichletEta
            if args.len() == 1 && cache.get(&args[0]).is_some_and(|v| !v.1.is_zero()) =>
        {
            zeta::eta_complex(get_cached(cache, args[0])?, w)
        }
        LibFn::Harmonic
            if args.len() == 1 && cache.get(&args[0]).is_some_and(|v| !v.1.is_zero()) =>
        {
            zeta::harmonic_complex(get_cached(cache, args[0])?, w)
        }
        LibFn::Erfi if args.len() == 1 && cache.get(&args[0]).is_some_and(|v| !v.1.is_zero()) => {
            cerf::erf_family(cerf::Kind::Erfi, get_cached(cache, args[0])?, w)
        }
        // `K`, `E`, `F`, `Π` with a non-real value or parameter (`elliptic.rs`;
        // refused before 0.54: `elliptic_e(2)`, `elliptic_k(1 + 2i)`; `Π`
        // before 0.55: `elliptic_pi(1/2, 2)`).
        LibFn::EllipticK | LibFn::EllipticE | LibFn::EllipticF | LibFn::EllipticPi
            if elliptic::takes(f, args, cache) =>
        {
            elliptic::eval(f, args, cache, w)
        }
        LibFn::Erfi
        | LibFn::ErfInv
        | LibFn::ErfcInv
        | LibFn::ExpInt
        | LibFn::Shi
        | LibFn::Chi
        | LibFn::FresnelS
        | LibFn::FresnelC
        | LibFn::LowerGamma
        | LibFn::UpperGamma
        | LibFn::DirichletEta
        | LibFn::AiryAi
        | LibFn::AiryBi
        | LibFn::AiryAiPrime
        | LibFn::AiryBiPrime
        | LibFn::EllipticK
        | LibFn::EllipticE
        | LibFn::EllipticF
        | LibFn::EllipticPi
        | LibFn::Gegenbauer
        | LibFn::Jacobi
        | LibFn::AssocLegendre
        | LibFn::AssocLaguerre
        | LibFn::BetaInc
        | LibFn::BetaIncRegularized => eval_special_09(f, args, cache, w),
        LibFn::PolyLog => eval_polylog(args, cache, w),
        // A non-real argument (`factorials.rs`; refused before 0.56).
        LibFn::RisingFactorial | LibFn::FallingFactorial
            if factorials::nonreal_args(f, args, cache) =>
        {
            factorials::eval_complex(f, args, cache, w)
        }
        LibFn::RisingFactorial
        | LibFn::FallingFactorial
        | LibFn::Harmonic
        | LibFn::Catalan
        | LibFn::Fibonacci
        | LibFn::Lucas => {
            let v = real_args(f.name(), args, cache)?;
            debug!(prec, name = f.name(), "evalf: Gamma quotient / digamma");
            let r = match f {
                LibFn::RisingFactorial => factorials::rising_factorial(&v[0], &v[1], w)?,
                LibFn::FallingFactorial => factorials::falling_factorial(&v[0], &v[1], w)?,
                LibFn::Catalan => factorials::catalan(&v[0], w)?,
                LibFn::Fibonacci => factorials::fibonacci(&v[0], false, w)?,
                LibFn::Lucas => factorials::fibonacci(&v[0], true, w)?,
                _ => factorials::harmonic(&v[0], w)?,
            };
            Ok((r, BigFloat::new(prec)))
        }
        LibFn::Bernoulli => {
            let v = real_args(f.name(), args, cache)?;
            debug!(prec, "evalf: bernoulli(s) = −s·ζ(1 − s)");
            let r = bernoulli::bernoulli_function(&v[0], w)?;
            Ok((r, BigFloat::new(prec)))
        }
        LibFn::EulerNumber => {
            let v = real_args(f.name(), args, cache)?;
            debug!(prec, "evalf: euler(s) by Hurwitz zeta functions");
            let r = bernoulli::euler_function(&v[0], w)?;
            Ok((r, BigFloat::new(prec)))
        }
        // A real non-integer argument (`factorials.rs`; refused before 0.54).
        LibFn::Factorial2
            if cache
                .get(&args[0])
                .is_some_and(|v| v.1.is_zero() && !v.0.is_int()) =>
        {
            let v = real_args(f.name(), args, cache)?;
            Ok((factorials::factorial2(&v[0], w)?, BigFloat::new(prec)))
        }
        LibFn::Factorial2
        | LibFn::Subfactorial
        | LibFn::Bell
        | LibFn::Stirling1
        | LibFn::Stirling2
        | LibFn::PartitionCount => Err(unevaluable(format!(
            "evalf has no numerical routine for '{f}' (eval folds it exactly at integer arguments)"
        ))),
        // Evaluated from its node, which carries the branch index literal.
        LibFn::LambertW => Err(unevaluable("lambertw is evaluated from its node")),
    }
}

/// `polylog(s, z)` for a real order: `arb_polylog` for a real `z ∈ [−1, 1]`,
/// `polylog::polylog_general` for a non-real `z` or `|z| > 1` (before 0.31
/// these were "not yet supported" for every order: `polylog(3/2, −3/16 −
/// 5/18·i)`, `polylog(2, 3)`, `polylog(1/3, −5)`).
fn eval_polylog(
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let s = get_cached(cache, args[0])?;
    let z = get_cached(cache, args[1])?;
    if !s.1.is_zero() {
        return polylog::polylog_complex_order(s, z, w);
    }
    if polylog::off_unit_interval(z) {
        return polylog::polylog_general(&s.0, z, w);
    }
    debug!(prec, "evalf: polylog on [-1, 1]");
    Ok((arb_polylog(&s.0, &z.0, w)?, BigFloat::new(prec)))
}

/// `W_k(z)` of a Lambert W node (`LambertW(x)`, `lambertw(x, k)`) over ℂ,
/// on every integer branch (`evalf::lambertw`).
pub(super) fn eval_lambertw_node(
    arena: &Arena,
    node: &ExprNode,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let (x, k) = lambertw::lambert_parts(arena, node)
        .unwrap_or_else(|| Err(unevaluable("not a Lambert W node")))?;
    let z = get_cached(cache, x)?;
    debug!(
        prec,
        k, "evalf: LambertW via Halley iteration (mpmath's starting values)"
    );
    lambertw::lambert_w(z, k, w)
}

/// A Bessel function `(order, x)` of real arguments through `arb`.
fn eval_bessel(
    arb: fn(&BigFloat, &BigFloat, &mut Wp<'_>) -> Result<BigFloat, SymplexError>,
    what: &str,
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let order = get_cached(cache, args[0])?;
    let arg = get_cached(cache, args[1])?;
    if !order.1.is_zero() || !arg.1.is_zero() {
        return Err(SymplexError::Unevaluable {
            reason: "Bessel of complex argument not yet supported in evalf".into(),
        });
    }
    tracing::debug!(prec, what, "evalf: Bessel function via series/asymptotic");
    let result = arb(&order.0, &arg.0, w)?;
    Ok((result, BigFloat::new(prec)))
}

/// A classical orthogonal polynomial `(n, x)` of real arguments by
/// recurrence, for a non-negative integer degree.
fn eval_orthopoly(
    kind: OrthoPoly,
    f: LibFn,
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let n_val = get_cached(cache, args[0])?;
    let x_val = get_cached(cache, args[1])?;
    if !n_val.1.is_zero() || !x_val.1.is_zero() {
        return Err(SymplexError::Unevaluable {
            reason: "orthogonal polynomial of complex arguments not supported".into(),
        });
    }
    let n_f = f64_of(&n_val.0)?;
    let n_round = n_f.round();
    // An integer exactly: before 0.29 a degree within 10⁻¹² of one was taken
    // for it, and `legendre(3 + 10⁻¹³, 1/3)` printed P₃(1/3) (wrong from the
    // 14th digit).
    if !n_val.0.is_int() || !(0.0..=1.0e7).contains(&n_round) {
        return Err(SymplexError::Unevaluable {
            reason: format!("{f}: degree must be a non-negative integer"),
        });
    }
    tracing::debug!(
        prec,
        n = n_round,
        "evalf: orthogonal polynomial via recurrence"
    );
    let result = arb_orthopoly(kind, n_round as u64, &x_val.0, w);
    Ok((result, BigFloat::new(prec)))
}

/// Dispatch for the 0.9 special functions inside `eval_node`.
///
/// Exhaustive over [`LibFn`]; the functions outside this family are an
/// error here (they are dispatched by [`eval_lib_fn`] and never arrive).
fn eval_special_09(
    f: LibFn,
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let name = f.name();
    let v = real_args(name, args, cache)?;
    let real = |r: BigFloat| Ok((r, BigFloat::new(prec)));
    debug!(prec, name, "evalf: special function (0.9)");
    match f {
        LibFn::Erfi => real(arb_erfi(&v[0], w)?),
        LibFn::ErfInv => real(arb_erfinv(&v[0], w)?),
        LibFn::ErfcInv => real(arb_erfcinv(&v[0], w)?),
        LibFn::ExpInt => real(arb_expint(&v[0], &v[1], w)?),
        LibFn::Shi => real(arb_shi_chi(&v[0], true, w)?),
        LibFn::Chi => {
            if v[0].is_zero() {
                return Err(infinite("Chi(0) is zoo"));
            }
            let chi = arb_shi_chi(&v[0].abs(), false, w)?;
            if v[0].is_negative() {
                // Chi(−x) = Chi(x) + iπ
                Ok((chi, w.cc.pi(prec, RM).clone()))
            } else {
                real(chi)
            }
        }
        LibFn::FresnelS => real(arb_fresnel(&v[0], true, w)?),
        LibFn::FresnelC => real(arb_fresnel(&v[0], false, w)?),
        LibFn::LowerGamma => real(arb_lowergamma(&v[0], &v[1], w)?),
        LibFn::UpperGamma => real(arb_uppergamma(&v[0], &v[1], w)?),
        LibFn::DirichletEta => real(arb_dirichlet_eta(&v[0], w)?),
        LibFn::AiryAi => real(arb_airy(&v[0], AiryKind::Ai, w)?),
        LibFn::AiryBi => real(arb_airy(&v[0], AiryKind::Bi, w)?),
        LibFn::AiryAiPrime => real(arb_airy(&v[0], AiryKind::AiPrime, w)?),
        LibFn::AiryBiPrime => real(arb_airy(&v[0], AiryKind::BiPrime, w)?),
        LibFn::EllipticK => real(arb_elliptic_k(&v[0], w)?),
        LibFn::EllipticE => real(arb_elliptic_e(&v[0], w)?),
        LibFn::EllipticF => real(arb_elliptic_f(&v[0], &v[1], w)?),
        LibFn::EllipticPi => real(arb_elliptic_pi(&v[0], &v[1], w)?),
        LibFn::Gegenbauer => real(arb_gegenbauer(&v[0], &v[1], &v[2], w)?),
        LibFn::Jacobi => real(arb_jacobi(&v[0], &v[1], &v[2], &v[3], w)?),
        LibFn::AssocLegendre => real(arb_assoc_legendre(&v[0], &v[1], &v[2], w)?),
        LibFn::AssocLaguerre => real(arb_assoc_laguerre(&v[0], &v[1], &v[2], w)?),
        LibFn::BetaInc | LibFn::BetaIncRegularized => real(arb_betainc(
            &v[0],
            &v[1],
            &v[2],
            &v[3],
            f == LibFn::BetaIncRegularized,
            w,
        )?),
        LibFn::Factorial2
        | LibFn::Subfactorial
        | LibFn::RisingFactorial
        | LibFn::FallingFactorial
        | LibFn::Fibonacci
        | LibFn::Lucas
        | LibFn::Bernoulli
        | LibFn::Harmonic
        | LibFn::Catalan
        | LibFn::Bell
        | LibFn::EulerNumber
        | LibFn::Stirling1
        | LibFn::Stirling2
        | LibFn::PartitionCount
        | LibFn::LambertW
        | LibFn::PolyLog
        | LibFn::BesselJ
        | LibFn::BesselY
        | LibFn::BesselI
        | LibFn::BesselK
        | LibFn::Legendre
        | LibFn::ChebyshevT
        | LibFn::ChebyshevU
        | LibFn::Hermite
        | LibFn::Laguerre => Err(unevaluable(format!("cannot evaluate function '{name}'"))),
    }
}
