//! Library functions: exact values (part of [`eval`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Library functions: exact values
// ═══════════════════════════════════════════════════════════════════════════

/// Exact value of the library function `f` on `args` (already evaluated and
/// of the arity `f` declares); `None` leaves the node as is.
///
/// Exhaustive over [`LibFn`]: a function with no exact rules says so here
/// rather than in a wildcard arm.
///
/// A closed form taken at an argument with `±∞` in it
/// ([`value_class::holds_real_infinity`]) that meets `∞ − ∞` gives `nan`
/// where the function may have a value: then the node stays (its values at
/// infinity fold when it is built, `canon::at_infinity`).
/// A `nan` argument makes `nan`, as when the node is built
/// (`canon::lib_function_value`): up to 0.56 an argument that became `nan`
/// when evaluated reached the folds of the other one (`uppergamma(s, ∞) =
/// 0`).  The folds that hold for every finite value of one argument ask it
/// of that argument ([`ValueClass::is_finite_proved`]).  At a `zoo`
/// argument the tabulated values are the only ones ([`at_complex_infinity`],
/// folded when the node is built too): no value is taken from a closed form
/// there (an identity to another library function at `zoo` is), and an
/// untabulated node stays, as it does when built, so that `0·f` and `f − f`
/// read the same before and after `eval`.  Up to 0.57 a closed form gave
/// `nan` or `zoo` (`uppergamma(2, zoo)`, `jacobi(1, zoo, 0, 0)`,
/// `gegenbauer(1, zoo, 2)`) or 0 (`betainc(zoo, 0, 1, 1)`) where the
/// built node stayed: `f − f` was 0 when built and `nan` after `eval`.
pub(super) fn eval_lib_fn(arena: &mut Arena, f: LibFn, args: &[ExprId]) -> Option<ExprId> {
    if args
        .iter()
        .any(|&a| ValueClass::special_atom(arena, a) == Some(ValueClass::NaN))
    {
        return Some(arena.nan);
    }
    if let Some(value) = at_complex_infinity(arena, f, args) {
        return Some(value);
    }
    let at_zoo = args
        .iter()
        .any(|&a| ValueClass::special_atom(arena, a) == Some(ValueClass::ComplexInf));
    let value = eval_lib_fn_closed(arena, f, args)?;
    // At `zoo` an identity to another library function there
    // (`assoc_legendre(ν, 0, z) = legendre(ν, z)`) is kept, a value is not;
    // `lambertw(z, k)` is the call form of `W_k`, rewritten to its node.
    if at_zoo
        && f != LibFn::LambertW
        && !matches!(arena.node(value), ExprNode::Lib(_, a)
            if a.iter().any(|&c| ValueClass::special_atom(arena, c) == Some(ValueClass::ComplexInf)))
    {
        return None;
    }
    if ValueClass::special_atom(arena, value) == Some(ValueClass::NaN)
        && args
            .iter()
            .any(|&a| value_class::holds_real_infinity(arena, a))
    {
        return None;
    }
    Some(value)
}

/// The value of the library function `f` at `args` when an argument is
/// `zoo`: the table of `canon::at_complex_infinity`, which also folds it
/// when the node is built (`zoo`, an integer), and the constant of a
/// polynomial of degree 0 in its variable, taken at 0 (the node of such a
/// polynomial stays when built, and counts as finite).  `None` where the
/// table has no entry.
fn at_complex_infinity(arena: &mut Arena, f: LibFn, args: &[ExprId]) -> Option<ExprId> {
    use crate::base::canon::AtComplexInfinity;
    Some(
        match crate::base::canon::at_complex_infinity(arena, f, args)? {
            AtComplexInfinity::ComplexInfinity => arena.complex_infinity,
            AtComplexInfinity::Integer(k) => arena.int(k),
            AtComplexInfinity::Constant => {
                let mut at_zero = args.to_vec();
                if let Some(x) = at_zero.last_mut() {
                    *x = arena.zero;
                }
                return eval_lib_fn_closed(arena, f, &at_zero);
            }
        },
    )
}

/// The closed forms and exact values of [`eval_lib_fn`].
fn eval_lib_fn_closed(arena: &mut Arena, f: LibFn, args: &[ExprId]) -> Option<ExprId> {
    match f {
        // ── Integer sequences and combinatorial counts ──
        LibFn::Factorial2 => eval_factorial2(arena, args[0]),
        LibFn::Subfactorial => eval_subfactorial(arena, args[0]),
        LibFn::RisingFactorial => eval_rising_factorial(arena, args[0], args[1]),
        LibFn::FallingFactorial => eval_falling_factorial(arena, args[0], args[1]),
        LibFn::Fibonacci => eval_fibonacci(arena, args[0]),
        LibFn::Lucas => eval_lucas(arena, args[0]),
        LibFn::Bernoulli => eval_bernoulli(arena, args[0]),
        LibFn::Harmonic => eval_harmonic(arena, args[0]),
        LibFn::Catalan => eval_catalan(arena, args[0]),
        LibFn::Bell => eval_bell(arena, args[0]),
        LibFn::EulerNumber => eval_euler_number(arena, args[0]),
        LibFn::Stirling1 => eval_stirling1(arena, args[0], args[1]),
        LibFn::Stirling2 => eval_stirling2(arena, args[0], args[1]),
        LibFn::PartitionCount => eval_partition_count(arena, args[0]),
        // `lambertw(x)` and `lambertw(x, 0)` become `ExprNode::LambertW`.
        LibFn::LambertW => eval_lambertw_call(arena, args),

        // ── Bessel functions: J and I at the origin; Y and K diverge there;
        // J, Y and K tend to 0 at infinity ──
        LibFn::BesselJ | LibFn::BesselI => eval_bessel_regular_at_zero(arena, args[0], args[1])
            .or_else(|| eval_bessel_at_infinity(arena, f, args[0], args[1])),
        LibFn::BesselY | LibFn::BesselK => eval_bessel_at_infinity(arena, f, args[0], args[1]),

        // ── Classical orthogonal polynomials of small explicit degree ──
        LibFn::Legendre => eval_orthopoly(arena, args[0], args[1], OrthoFamily::Legendre),
        LibFn::ChebyshevT => eval_orthopoly(arena, args[0], args[1], OrthoFamily::ChebyshevT),
        LibFn::ChebyshevU => eval_orthopoly(arena, args[0], args[1], OrthoFamily::ChebyshevU),
        LibFn::Hermite => eval_orthopoly(arena, args[0], args[1], OrthoFamily::Hermite),
        LibFn::Laguerre => eval_orthopoly(arena, args[0], args[1], OrthoFamily::Laguerre),

        // ── More special functions (0.9) ──
        LibFn::Erfi => eval_erfi(arena, args[0]),
        LibFn::ErfInv => eval_erfinv(arena, args[0]),
        LibFn::ErfcInv => eval_erfcinv(arena, args[0]),
        LibFn::ExpInt => eval_expint(arena, args[0], args[1]),
        LibFn::Shi => eval_shi(arena, args[0]),
        LibFn::Chi => eval_chi(arena, args[0]),
        LibFn::FresnelS | LibFn::FresnelC => eval_fresnel(arena, f, args[0]),
        LibFn::LowerGamma => eval_lowergamma(arena, args[0], args[1]),
        LibFn::UpperGamma => eval_uppergamma(arena, args[0], args[1]),
        LibFn::PolyLog => eval_polylog(arena, args[0], args[1]),
        LibFn::DirichletEta => eval_dirichlet_eta(arena, args[0]),
        LibFn::AiryAi | LibFn::AiryBi | LibFn::AiryAiPrime | LibFn::AiryBiPrime => {
            eval_airy(arena, f, args[0])
        }
        LibFn::EllipticK => eval_elliptic_k(arena, args[0]),
        LibFn::EllipticE => eval_elliptic_e(arena, args[0]),
        LibFn::EllipticF => eval_elliptic_f(arena, args[0], args[1]),
        LibFn::EllipticPi => eval_elliptic_pi(arena, args[0], args[1]),
        LibFn::Gegenbauer => eval_gegenbauer(arena, args[0], args[1], args[2]),
        LibFn::Jacobi => eval_jacobi(arena, args[0], args[1], args[2], args[3]),
        LibFn::AssocLegendre => eval_assoc_legendre(arena, args[0], args[1], args[2]),
        LibFn::AssocLaguerre => eval_assoc_laguerre(arena, args[0], args[1], args[2]),
        LibFn::BetaInc | LibFn::BetaIncRegularized => eval_betainc(
            arena,
            f == LibFn::BetaIncRegularized,
            args[0],
            args[1],
            args[2],
            args[3],
        ),
    }
}

/// `J_0(0) = I_0(0) = 1`; `J_ν(0) = I_ν(0) = 0` for `ν > 0` and for every
/// integer `ν ≠ 0` (`J_{−n} = (−1)ⁿ J_n`, `I_{−n} = I_n`); `zoo` for a
/// negative non-integer `ν` (SymPy: `besselj(-3, 0)` → 0,
/// `besselj(-1/2, 0)` → zoo), when both order and argument are numbers.
/// Before 0.30 a negative order stayed.  (The zeros and poles fold when the
/// application is built; see `canon_function`.)
fn eval_bessel_regular_at_zero(arena: &mut Arena, order: ExprId, x: ExprId) -> Option<ExprId> {
    let order_num = arena.as_num(order)?;
    let arg_num = arena.as_num(x)?;
    if !arg_num.is_zero() {
        return None;
    }
    if order_num.is_zero() {
        Some(arena.one)
    } else if order_num.is_positive() || order_num.is_integer() {
        Some(arena.zero)
    } else {
        Some(arena.complex_infinity)
    }
}

/// `J_ν(±∞) = Y_ν(±∞) = K_ν(+∞) = 0` for an order with no infinity in it,
/// and `I_ν(+∞) = ∞` for a rational order (`J_ν(x), Y_ν(x) ~
/// √(2/(πx))·cos(…)`, `K_ν(x) ~ √(π/(2x))·e^(−x)`, `I_ν(x) ~ e^x/√(2πx)`,
/// DLMF 10.7.8, 10.30.4, 10.25.3; SymPy 1.14: `besselj(nu, oo)`,
/// `bessely(nu, -oo)`, `besselk(nu, oo)` → 0, `besseli(nu, oo)` → oo), as
/// `canon` folds them when they are built; here for an argument that
/// becomes infinite when evaluated.
fn eval_bessel_at_infinity(
    arena: &mut Arena,
    f: LibFn,
    order: ExprId,
    x: ExprId,
) -> Option<ExprId> {
    // An order that is not known finite (`zoo`, an unevaluated pole): no
    // value (up to 0.56 `besselj(elliptic_pi(π, 1), ∞)` was 0).
    if !ValueClass::of(arena, order).is_finite_proved() {
        return None;
    }
    match (f, ValueClass::special_atom(arena, x)) {
        (LibFn::BesselJ | LibFn::BesselY, Some(ValueClass::PosInf | ValueClass::NegInf)) => {
            Some(arena.zero)
        }
        (LibFn::BesselK, Some(ValueClass::PosInf)) => Some(arena.zero),
        (LibFn::BesselI, Some(ValueClass::PosInf)) if arena.as_num(order).is_some() => {
            Some(arena.infinity)
        }
        _ => None,
    }
}

/// How an orthogonal polynomial is evaluated, by whether an argument holds
/// `±∞` ([`expansion`]).
///
/// No closed form is evaluated at an argument holding `∞` or `−∞` (also in
/// `i·∞`, `x·∞`: [`value_class::holds_real_infinity`]): those of `Γ(s, x)`, `γ(s, x)` and
/// `E₀(x)`, sums of `e^{−x}·xᵏ` terms, evaluate to `∞ − ∞` or `∞/∞` (`nan`)
/// where the limit is `0` or `±∞`, `legendre(3, ∞)` gave `∞ − ∞ = nan` up
/// to 0.53, `jacobi(1, a, b, ∞)` gave `(a + 1)·∞ + (b + 1)·∞`, whose
/// value at `a = 0`, `b = −3` is `nan` where `P₁^{(0,−3)}(∞) = −∞`.  The
/// values the functions have there fold when the node is built
/// (`canon::at_infinity`), and the rest stays (up to 0.53 the closed forms
/// of `Γ(s, x)` were taken at `±i·∞`).  At an undirected `zoo`, where the
/// functions have no limit, the closed forms keep giving their `zoo` or
/// `nan` (SymPy 1.14 agrees).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Expansion {
    /// A parameter (`n`, `α`, `β`, `m`) is infinite: no closed form.
    Refused,
    /// The argument is infinite and the degree may be 2 or more, or a
    /// parameter is symbolic: the expansion can meet `∞ − ∞`, so only the
    /// identities that hold at every argument apply (`C_ν^{(1/2)} = P_ν`).
    /// Up to 0.53 the expansions gave `x³·∞ − x·∞` for `chebyshev_t(3,
    /// x·∞)`, `nan` for `hermite(3, ∞ + i)` and `(a + 1)·∞ + (b + 1)·∞` for
    /// `jacobi(1, a, b, ∞)` (`nan` at `a = 0`, `b = −3`, where the value is
    /// `−∞`).
    IdentitiesOnly,
    /// Expanded: a finite argument, or a polynomial of degree 0 or 1 with
    /// numeric parameters (`hermite(1, ∞ + iπ) = ∞ + 2πi`; [`eval_lib_fn`]
    /// refuses a `nan` it gives, `0·(x + ∞)` for `C₁^{(0)}`).
    Expanded,
}

/// The [`Expansion`] of an orthogonal polynomial of degree `n` with the
/// parameters `params` at `x`.
fn expansion(arena: &Arena, params: &[ExprId], n: ExprId, x: ExprId) -> Expansion {
    if params
        .iter()
        .any(|&p| value_class::holds_real_infinity(arena, p))
        || value_class::holds_real_infinity(arena, n)
    {
        return Expansion::Refused;
    }
    let low_degree =
        (n == arena.zero || n == arena.one) && params.iter().all(|&p| arena.as_num(p).is_some());
    if !low_degree && value_class::holds_real_infinity(arena, x) {
        Expansion::IdentitiesOnly
    } else {
        Expansion::Expanded
    }
}

/// A classical orthogonal polynomial `p_n(x)` expanded by `expand` when `n`
/// is a non-negative integer no larger than the configured power limit.
fn eval_orthopoly(arena: &mut Arena, n: ExprId, x: ExprId, family: OrthoFamily) -> Option<ExprId> {
    if expansion(arena, &[], n, x) != Expansion::Expanded {
        return None;
    }
    let n_num = arena.as_num(n)?;
    if !n_num.is_integer() {
        return None;
    }
    if n_num.is_negative() {
        let m = (-n_num.to_integer()).to_u64()?;
        return classical_at_negative_degree(arena, family, m, x);
    }
    let n_int = n_num.to_integer().to_u64()?;
    if n_int > arena.config.max_pow_exponent as u64 {
        return None;
    }
    ortho_value(arena, &family, n_int as usize, x)
}

/// The classical orthogonal polynomials at the degree `−m` (`m ≥ 1`), by
/// the identities of their defining functions (SymPy's conventions; mpmath
/// `legendre`, `chebyt`, `chebyu` agree): `P_{−m}(x) = P_{m−1}(x)` (the
/// Legendre function, DLMF 14.9.5), `T_{−m} = T_m`, `U_{−1} = 0`, `U_{−m} =
/// −U_{m−2}` (`U_n = sin((n + 1)θ)/sin θ`), `L_{−m}(x) = eˣ·L_{m−1}(−x)`
/// (`L_ν = ₁F₁(−ν; 1; ·)` and Kummer's transformation).  `H_{−m}` is the
/// Hermite function, not a polynomial: [`hermite_function_closed`] (SymPy
/// refuses it; up to 0.57 it stayed).  Up to 0.56 every negative degree
/// stayed.
fn classical_at_negative_degree(
    arena: &mut Arena,
    family: OrthoFamily,
    m: u64,
    x: ExprId,
) -> Option<ExprId> {
    let max = arena.config.max_pow_exponent as u64;
    let at = |arena: &mut Arena, family: &OrthoFamily, d: u64, x: ExprId| {
        (d <= max)
            .then(|| ortho_value(arena, family, d as usize, x))
            .flatten()
    };
    match family {
        OrthoFamily::Legendre => at(arena, &family, m - 1, x),
        OrthoFamily::ChebyshevT => at(arena, &family, m, x),
        OrthoFamily::ChebyshevU if m == 1 => Some(arena.zero),
        OrthoFamily::ChebyshevU => {
            let u = at(arena, &family, m - 2, x)?;
            Some(arena.neg(u))
        }
        OrthoFamily::Laguerre => {
            let minus_x = arena.neg(x);
            let l = at(arena, &family, m - 1, minus_x)?;
            let e = arena.exp(x);
            Some(arena.mul(&[e, l]))
        }
        OrthoFamily::Hermite => hermite_function_closed(arena, m, x),
        _ => None,
    }
}

/// The Hermite function of degree `−m` (`1 ≤ m ≤` [`MAX_SPECIAL_EXPANSION`])
/// in closed form, `H_{−m}(x) = A_m(x)·√π·e^{x²}·erfc(x) + B_m(x)` with
/// `A_m`, `B_m ∈ ℚ[x]`: `H_{−m}(x) = (√π/2)·e^{x²}·i^{m−1}erfc(x)`, the
/// repeated integral of `erfc` (DLMF 7.18.2; DLMF 12.7.6 with `H_ν(x) =
/// 2^{ν/2}e^{x²/2}U(−ν − ½, √2x)`), and `A`, `B` from `H_0 = 1`, `H_{−1} =
/// (√π/2)·e^{x²}·erfc x` by the recurrence `H_{ν+1} = 2xH_ν − 2νH_{ν−1}`
/// of the Hermite functions (DLMF 18.9.1 continued; mpmath's `hermite(−1,
/// 1/2) = 0.4244363835`, `hermite(−3, −1/2) = 0.9541049656`): `H_{−k−1} =
/// (H_{−k+1} − 2x·H_{−k})/(2k)`, so `H_{−2} = 1/2 − (√π/2)·x·e^{x²}·erfc x`.
/// `None` beyond the bound and at an argument holding a special atom (no
/// closed form is evaluated there).  Up to 0.57 `hermite(−m, x)` stayed and
/// `evalf` refused it.
fn hermite_function_closed(arena: &mut Arena, m: u64, x: ExprId) -> Option<ExprId> {
    let m = crate::base::cast::bounded_len(m, MAX_SPECIAL_EXPANSION)?;
    if m == 0 || value_class::holds_special_atom(arena, x) {
        return None;
    }
    // (A_k, B_k) for k = 0 and k = 1, coefficients by degree.
    let half = Q::new(BigInt::one(), BigInt::from(2));
    let mut prev: (Vec<Q>, Vec<Q>) = (Vec::new(), vec![Q::one()]);
    let mut cur: (Vec<Q>, Vec<Q>) = (vec![half], Vec::new());
    for k in 1..m {
        // (p − 2x·c)/(2k), coefficientwise.
        let step = |p: &[Q], c: &[Q]| -> Vec<Q> {
            let scale = Q::from_integer(BigInt::from(2 * k));
            let mut out: Vec<Q> = p.to_vec();
            for (i, ci) in c.iter().enumerate() {
                let t = ci * Q::from_integer(BigInt::from(2));
                match out.get_mut(i + 1) {
                    Some(o) => *o -= t,
                    None => {
                        while out.len() < i + 1 {
                            out.push(Q::zero());
                        }
                        out.push(-t);
                    }
                }
            }
            out.iter().map(|o| o / &scale).collect()
        };
        let next = (step(&prev.0, &cur.0), step(&prev.1, &cur.1));
        prev = std::mem::replace(&mut cur, next);
    }
    let poly = |arena: &mut Arena, coeffs: &[Q]| -> ExprId {
        let mut terms: Vec<ExprId> = Vec::with_capacity(coeffs.len());
        for (i, c) in coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let c = arena.num_ratio(c.clone());
            let k = arena.int(i64::try_from(i).unwrap_or(i64::MAX));
            let xk = arena.pow(x, k);
            terms.push(arena.mul(&[c, xk]));
        }
        arena.add(&terms)
    };
    let a = poly(arena, &cur.0);
    let b = poly(arena, &cur.1);
    let two = arena.int(2);
    let x2 = arena.pow(x, two);
    let e = arena.exp(x2);
    let erfc = arena.erfc(x);
    let sqrt_pi = arena.sqrt(arena.pi);
    let head = arena.mul(&[a, sqrt_pi, e, erfc]);
    Some(arena.add(&[head, b]))
}

/// Is `n` a negative integer and every parameter in `params` finite
/// ([`ValueClass::is_finite_proved`])?  Then the
/// parametrised families — Gegenbauer's `C_n^{(λ)}`, Jacobi's
/// `P_n^{(α,β)}`, the generalised Laguerre `L_n^{(α)}` — are the zero
/// polynomial: SymPy's convention for Gegenbauer ("for negative n the
/// polynomials vanish"), the binomial prefactor `C(n + α, n) = 0` of their
/// hypergeometric forms (DLMF 18.5.7, 18.5.12; `C(r, k) = 0` for a negative
/// integer `k`, as `binomial`; mpmath `jacobi(−1, 2, 3, 1j)`, `laguerre(−1,
/// 2, 0.3)` are 0), and what keeps the derivative formulas `Cₙ⁽λ⁾′ =
/// 2λC₍ₙ₋₁₎⁽λ⁺¹⁾`, `Pₙ⁽α,β⁾′ = ½(n + α + β + 1)P₍ₙ₋₁₎⁽α⁺¹,β⁺¹⁾`,
/// `Lₙ⁽α⁾′ = −L₍ₙ₋₁₎⁽α⁺¹⁾` right at `n = 0`.  (`C_{−1}^{(1/2)} = 0` while the
/// Legendre function `P_{−1} = 1`: the identities `C_n^{(1/2)} = P_n`,
/// `P_n^{(0,0)} = P_n`, `L_n^{(0)} = L_n` hold for `n ≥ 0`, as in SymPy.)  Up to
/// 0.56 they stayed, and series ended in `K(jacobi(−1, 2, 3, i) + 1)` =
/// `K(1)` = ∞ or carried coefficients `gegenbauer(−2, 6, 0)`.
fn zero_polynomial_degree(arena: &Arena, n: ExprId, params: &[ExprId]) -> bool {
    arena
        .as_num(n)
        .is_some_and(|q| q.is_integer() && q.is_negative())
        && params
            .iter()
            .all(|&p| ValueClass::of(arena, p).is_finite_proved())
}

/// Intern the library-function node `f(args)` (no folding).
pub(crate) fn apply_named(arena: &mut Arena, f: LibFn, args: &[ExprId]) -> ExprId {
    arena.lib_apply(f, args)
}

/// Largest integer / half-integer parameter for which the incomplete gamma
/// functions, `polylog(−n, z)` and the parametrised orthogonal polynomials
/// are expanded into closed forms.
const MAX_SPECIAL_EXPANSION: i64 = 64;

/// `id` as a small integer in `[lo, hi]`.
fn as_int_in(arena: &Arena, id: ExprId, lo: i64, hi: i64) -> Option<i64> {
    let r = arena.as_num(id)?;
    if !r.is_integer() {
        return None;
    }
    let n: i64 = r.to_integer().try_into().ok()?;
    (lo..=hi).contains(&n).then_some(n)
}

/// `id` as a rational number.
fn as_ratio(arena: &Arena, id: ExprId) -> Option<Q> {
    arena.as_num(id).cloned()
}

/// `π/2`.
pub(super) fn half_pi(arena: &mut Arena) -> ExprId {
    let half = arena.rational(1, 2);
    arena.mul(&[half, arena.pi])
}

/// `erfi(0) = 0`, `erfi(±∞) = ±∞`, odd.
fn eval_erfi(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.zero);
    }
    // (the atom itself: `erfi(±∞) = ±∞`; `nan` folds when built; `zoo`: no
    // limit)
    if ValueClass::special_atom(arena, x).is_some_and(|c| c.real_infinity().is_some()) {
        return Some(x);
    }
    if let Some(y) = as_negated_general(arena, x) {
        let e = apply_named(arena, LibFn::Erfi, &[y]);
        return Some(arena.neg(e));
    }
    None
}

/// `erfinv(0) = 0`, `erfinv(±1) = ±∞`, odd.
fn eval_erfinv(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.zero);
    }
    if x == arena.one {
        return Some(arena.infinity);
    }
    if x == arena.neg_one {
        return Some(arena.neg_infinity);
    }
    if let Some(y) = as_negated_general(arena, x) {
        let e = apply_named(arena, LibFn::ErfInv, &[y]);
        return Some(arena.neg(e));
    }
    None
}

/// `erfcinv(1) = 0`, `erfcinv(0) = ∞`, `erfcinv(2) = −∞`.
fn eval_erfcinv(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.one {
        return Some(arena.zero);
    }
    if x == arena.zero {
        return Some(arena.infinity);
    }
    if as_int_in(arena, x, 2, 2).is_some() {
        return Some(arena.neg_infinity);
    }
    None
}

/// `E_n(∞) = 0`, `E_n(0) = 1/(n−1)` for `n > 1`, `E_0(x) = e^{−x}/x`.
/// `E_0` at another infinite argument is not rewritten (up to 0.51 `e^{−x}/x`
/// there gave `∞/∞`: `expint(0, −∞)` was `nan`, its limit is `−∞`); the
/// values at `−∞` (`E_0(−∞) = −∞`, `E₁(−∞) = −∞ − iπ`) fold when the
/// node is rebuilt (`canon::lib_function_value`).
fn eval_expint(arena: &mut Arena, n: ExprId, x: ExprId) -> Option<ExprId> {
    // For every finite order (up to 0.56 also `expint(zoo, ∞)` and
    // `expint(elliptic_pi(π, 1), ∞)`: no value).
    if ValueClass::special_atom(arena, x) == Some(ValueClass::PosInf) {
        return ValueClass::of(arena, n)
            .is_finite_proved()
            .then_some(arena.zero);
    }
    if n == arena.zero && !value_class::holds_real_infinity(arena, x) {
        let neg_x = arena.neg(x);
        let e = arena.exp(neg_x);
        let v = arena.div(e, x);
        return Some(eval(arena, v));
    }
    if x == arena.zero
        && let Some(r) = as_ratio(arena, n)
        && r > Ratio::one()
    {
        let v = (r - Ratio::one()).recip();
        let nid = arena.intern_num(v);
        return Some(arena.intern(ExprNode::Num(nid)));
    }
    None
}

/// `Shi(0) = 0`, `Shi(±∞) = ±∞`, odd.
fn eval_shi(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.zero);
    }
    // (the atom itself: `Shi(±∞) = ±∞`; `nan` folds when built; `zoo`: no
    // limit)
    if ValueClass::special_atom(arena, x).is_some_and(|c| c.real_infinity().is_some()) {
        return Some(x);
    }
    if let Some(y) = as_negated_general(arena, x) {
        let e = apply_named(arena, LibFn::Shi, &[y]);
        return Some(arena.neg(e));
    }
    None
}

/// `Chi(0) = zoo` (the singularity of `ln z`, as `Ci(0)`; before 0.30
/// `−∞`), `Chi(∞) = ∞`.
fn eval_chi(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.complex_infinity);
    }
    // (`Chi(−∞) = ∞ + iπ` and `nan` fold when built; `zoo`: no limit)
    (ValueClass::special_atom(arena, x) == Some(ValueClass::PosInf)).then_some(arena.infinity)
}

/// Fresnel `S`/`C`: `0 ↦ 0`, `±∞ ↦ ±1/2`, odd.
fn eval_fresnel(arena: &mut Arena, f: LibFn, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.zero);
    }
    match ValueClass::special_atom(arena, x) {
        Some(ValueClass::PosInf) => return Some(arena.rational(1, 2)),
        Some(ValueClass::NegInf) => return Some(arena.rational(-1, 2)),
        // (`nan` folds when built; `zoo`: no limit)
        _ => {}
    }
    if let Some(y) = as_negated_general(arena, x) {
        let e = apply_named(arena, f, &[y]);
        return Some(arena.neg(e));
    }
    None
}

/// `Γ(s, x)` for `s = base + k` (`k ∈ ℤ`) from the base value `Γ(base, x)`
/// via `Γ(s+1, x) = s Γ(s, x) + x^s e^{−x}`, upward or downward.
fn shift_uppergamma(arena: &mut Arena, base: Q, base_val: ExprId, k: i64, x: ExprId) -> ExprId {
    let neg_x = arena.neg(x);
    let e_neg_x = arena.exp(neg_x);
    let mut s = base;
    let mut val = base_val;
    if k >= 0 {
        for _ in 0..k {
            // Γ(s+1) = s Γ(s) + x^s e^{-x}
            let s_id = arena.intern_num(s.clone());
            let s_id = arena.intern(ExprNode::Num(s_id));
            let x_pow = arena.pow(x, s_id);
            let a = arena.mul(&[s_id, val]);
            let b = arena.mul(&[x_pow, e_neg_x]);
            val = arena.add(&[a, b]);
            s += Ratio::one();
        }
    } else {
        for _ in 0..(-k) {
            // Γ(s−1) = (Γ(s) − x^{s−1} e^{-x}) / (s−1)
            s -= Ratio::one();
            let s_id = arena.intern_num(s.clone());
            let s_id = arena.intern(ExprNode::Num(s_id));
            let x_pow = arena.pow(x, s_id);
            let b = arena.mul(&[x_pow, e_neg_x]);
            let numer = arena.sub(val, b);
            val = arena.div(numer, s_id);
        }
    }
    let expanded = crate::transforms::expand::expand(arena, val);
    eval(arena, expanded)
}

/// Most terms the expanded closed form of `Γ(s, x)` (or `γ(s, x)`) may have
/// ([`closed_form_stays_small`]).
const MAX_GAMMA_CLOSED_FORM_TERMS: u64 = 1 << 12;

/// A bound on the number of terms `expand` makes of `x`: a sum adds its
/// terms' counts, a product multiplies them, a power `bᵉ` with a positive
/// integer `e` has at most `C(e + t − 1, t − 1)` terms for a `b` of `t`
/// terms (the monomials of degree `e` in `t` unknowns), anything else is one
/// term; saturating at `cap + 1`.
fn expanded_terms_bound(arena: &Arena, x: ExprId, cap: u64) -> u64 {
    let over = cap.saturating_add(1);
    let mut terms: FxHashMap<ExprId, u64> = FxHashMap::default();
    for id in walk::post_order_ids(arena, x) {
        let of = |c: &ExprId| terms.get(c).copied().unwrap_or(1);
        let t = match arena.node(id) {
            ExprNode::Add(cs) => cs.iter().map(of).fold(0u64, u64::saturating_add),
            ExprNode::Mul(cs) => cs.iter().map(of).fold(1u64, u64::saturating_mul),
            ExprNode::Neg(c) => of(c),
            ExprNode::Pow(b, e) => match arena.as_num(*e) {
                Some(q) if q.is_integer() && q.is_positive() => {
                    let e = q.to_integer().to_u64().unwrap_or(u64::MAX);
                    multisets_bound(of(b), e, over)
                }
                _ => 1,
            },
            _ => 1,
        };
        terms.insert(id, t.min(over));
    }
    terms.get(&x).copied().unwrap_or(1)
}

/// `C(d + t − 1, d)`, the number of monomials of degree `d` in `t` unknowns,
/// saturating at `over`.
fn multisets_bound(t: u64, d: u64, over: u64) -> u64 {
    if t <= 1 || d == 0 {
        return 1;
    }
    let mut c: u128 = 1;
    for k in 1..=u128::from(d) {
        // c·(t − 1 + k)/k stays an integer: C(t − 1 + k, k).
        c = c * (u128::from(t) - 1 + k) / k;
        if c >= u128::from(over) {
            return over;
        }
    }
    u64::try_from(c).unwrap_or(over)
}

/// Does the expanded closed form of `Γ(s, x)` stay below
/// [`MAX_GAMMA_CLOSED_FORM_TERMS`] terms?  Its powers of `x` reach the degree
/// `d = ⌊s − ½⌋` (`s − 1` for an integer, which bounds the half-integer
/// powers `x^{s−1}` of a half-integer `s` too; none for `s ≤ 0`), and
/// for an `x` of at most `t` terms after expansion the powers `x⁰, …, xᵈ`
/// expand into at most `C(d + t, d)` monomials.  Beyond the bound `Γ(s, x)`
/// stays (`evalf` computes it): up to 0.57 `uppergamma(8, a₁ + … + a₂₅)`
/// expanded `x⁷` into `C(32, 7) = 3.4·10⁶` products (24 s, 160 MB of
/// output; the G-test on a 5 × 5 table hung on it).
fn closed_form_stays_small(arena: &Arena, s: &Q, x: ExprId) -> bool {
    let d = (s - Q::new(BigInt::one(), BigInt::from(2))).floor();
    if !d.is_positive() {
        return true;
    }
    let d = d.to_integer().to_u64().unwrap_or(u64::MAX);
    let cap = MAX_GAMMA_CLOSED_FORM_TERMS;
    let t = expanded_terms_bound(arena, x, cap);
    // Σ_{k ≤ d} C(k + t − 1, k) = C(d + t, d), the monomials of degree d in
    // t + 1 unknowns.
    multisets_bound(t.saturating_add(1), d, cap.saturating_add(1)) <= cap
}

/// Closed form of `Γ(s, x)` for integer or half-integer `s` with
/// `|s| ≤ MAX_SPECIAL_EXPANSION` whose expansion stays small
/// ([`closed_form_stays_small`]): bases `Γ(1, x) = e^{−x}`,
/// `Γ(0, x) = E₁(x)`, `Γ(1/2, x) = √π erfc(√x)`.
fn uppergamma_closed(arena: &mut Arena, s: &Q, x: ExprId) -> Option<ExprId> {
    let two = BigInt::from(2);
    let bound = Ratio::from_integer(BigInt::from(MAX_SPECIAL_EXPANSION));
    if s.abs() > bound || !closed_form_stays_small(arena, s, x) {
        return None;
    }
    let neg_x = arena.neg(x);
    let e_neg_x = arena.exp(neg_x);
    if s.is_integer() {
        let n: i64 = s.to_integer().try_into().ok()?;
        if n >= 1 {
            // Γ(n, x) = (n−1)! e^{−x} Σ_{k<n} x^k/k!
            let mut terms: Vec<ExprId> =
                Vec::with_capacity(crate::base::cast::bounded_len(n, MAX_SPECIAL_EXPANSION)?);
            let mut inv_fact = Ratio::<BigInt>::one();
            for k in 0..n {
                if k > 0 {
                    inv_fact /= Ratio::from_integer(BigInt::from(k));
                }
                let c = arena.intern_num(inv_fact.clone());
                let c = arena.intern(ExprNode::Num(c));
                let k_id = arena.int(k);
                let xk = arena.pow(x, k_id);
                terms.push(arena.mul(&[c, xk]));
            }
            let sum = arena.add(&terms);
            let fact = factorial_ratio((n - 1) as usize);
            let f = arena.intern_num(fact);
            let f = arena.intern(ExprNode::Num(f));
            let v = arena.mul(&[f, e_neg_x, sum]);
            let v = crate::transforms::expand::expand(arena, v);
            return Some(eval(arena, v));
        }
        // n ≤ 0: shift down from Γ(0, x) = E₁(x).
        let e1 = apply_named(arena, LibFn::ExpInt, &[arena.one, x]);
        return Some(shift_uppergamma(arena, Ratio::zero(), e1, n, x));
    }
    if s.denom() == &two {
        // s = 1/2 + k
        let half = Ratio::new(BigInt::one(), two);
        let k: i64 = (s - &half).to_integer().try_into().ok()?;
        let sqrt_pi = arena.sqrt(arena.pi);
        let sqrt_x = arena.sqrt(x);
        let erfc = arena.erfc(sqrt_x);
        let base = arena.mul(&[sqrt_pi, erfc]);
        return Some(shift_uppergamma(arena, half, base, k, x));
    }
    None
}

/// `s` is *known* to be non-positive at the arena level: a numeric `s ≤ 0`
/// or `−∞`.  (Symbolic parameters stay `false` — assumptions are not
/// visible here — so folds that hold for `s > 0` are kept for them, which
/// the Gamma-distribution CDF `γ(k, 0)/Γ(k) = 0` relies on.)
fn known_nonpositive(arena: &Arena, s: ExprId) -> bool {
    ValueClass::special_atom(arena, s) == Some(ValueClass::NegInf)
        || arena.as_num(s).is_some_and(|r| !r.is_positive())
}

/// Bits the closed form `γ(s, x) = Γ(s) − Γ(s, x)` may lose to cancellation
/// at a rational `x` before the fold keeps `γ(s, x)` instead.
const LOWERGAMMA_FOLD_MAX_LOSS_BITS: f64 = 64.0;

/// `ln q` for a positive rational, without overflowing on huge numerators
/// or denominators.
fn ln_positive_ratio(q: &Q) -> f64 {
    fn ln_big(n: &BigInt) -> f64 {
        let bits = n.bits();
        if bits <= 1000 {
            return n.to_f64().unwrap_or(f64::INFINITY).ln();
        }
        let shift = bits - 64;
        let top: BigInt = n >> shift;
        top.to_f64().unwrap_or(f64::INFINITY).ln() + shift as f64 * std::f64::consts::LN_2
    }
    ln_big(q.numer()) - ln_big(q.denom())
}

/// Does the closed form `Γ(s) − Γ(s, x)` of `γ(s, x)`, for an integer or
/// half-integer `s > 0` and a rational `x > 0`, subtract two numbers that
/// agree to more than [`LOWERGAMMA_FOLD_MAX_LOSS_BITS`]?  It loses about
/// `−log₂ P(s, x)` bits (`P = γ/Γ`, regularised), which is large only for
/// `x` well below `s`, where `P(s, x) ≤ xˢ e⁻ˣ/Γ(s + 1) · (s + 1)/(s + 1 − x)`
/// (the power series).  There the closed form is an exact expression that
/// evaluates to `0`: `γ(61, 1)/60!` (a Poisson(1) tail, `7.4·10⁻⁸⁵`) is
/// `1 − e⁻¹ Σ_{k ≤ 60} 1/k!`, `γ(5, 10⁻³⁰)` loses 500 bits.
fn lowergamma_closure_cancels(s: &Q, x: &Q) -> bool {
    if !x.is_positive() || x >= s {
        return false;
    }
    let Some(x_f) = x.to_f64() else {
        return false;
    };
    lowergamma_closure_cancels_at(s, x_f, ln_positive_ratio(x))
}

/// [`lowergamma_closure_cancels`] for a real `0 < x < s` known as `x ≈ x_f`,
/// `ln x ≈ ln_x` (`x_f` may underflow to 0).  Before 0.29 only a rational
/// `x` was guarded: `γ(5, √2/10³⁰)` folded into a closed form that
/// evaluated to `0` (true value `1.13·10⁻¹⁵⁰`).
fn lowergamma_closure_cancels_at(s: &Q, x_f: f64, ln_x: f64) -> bool {
    let half_integer = s.denom() == &BigInt::from(2);
    if !(s.is_integer() || half_integer)
        || *s > Q::from_integer(BigInt::from(MAX_SPECIAL_EXPANSION))
    {
        return false;
    }
    let Some(s_f) = s.to_f64() else {
        return false;
    };
    // ln Γ(s + 1) for an integer or half-integer s ≤ MAX_SPECIAL_EXPANSION.
    let mut ln_gamma = if s.is_integer() {
        0.0
    } else {
        0.5 * std::f64::consts::PI.ln()
    };
    let mut k = if s.is_integer() { 2.0 } else { 0.5 };
    while k <= s_f + 0.25 {
        ln_gamma += f64::ln(k);
        k += 1.0;
    }
    let ln_p = s_f * ln_x - x_f - ln_gamma + ((s_f + 1.0) / (s_f + 1.0 - x_f)).ln();
    -ln_p / std::f64::consts::LN_2 > LOWERGAMMA_FOLD_MAX_LOSS_BITS
}

/// [`lowergamma_closure_cancels`] for an `x = re + i·im` off the positive
/// axis (a complex or negative constant): the closed form's terms are
/// at most `Γ(s)·max(1, e^{|x| − re})`, and for `|x| < (s + 1)/2` the power
/// series `γ(s, x) = xˢe⁻ˣ/s · Σ xᵏ/((s + 1)⋯(s + k))` bounds `|γ(s, x)|`
/// below by `|x|ˢe^{−re}/s · (s + 1 − 2|x|)/(s + 1 − |x|)`; the ratio is the
/// loss.  Up to 0.45 only a real positive `x` was guarded: `γ(30, (−13/100 +
/// 99i/100)/10¹¹)` folded into `29!·(1 − e^{−x}·Σ…)`, which lost 1,200 bits
/// and was refused at 43 digits (`PrecisionExhausted`).
fn lowergamma_closure_cancels_complex(s: &Q, re: f64, im: f64) -> bool {
    let half_integer = s.denom() == &BigInt::from(2);
    if !(s.is_integer() || half_integer)
        || *s > Q::from_integer(BigInt::from(MAX_SPECIAL_EXPANSION))
    {
        return false;
    }
    let Some(s_f) = s.to_f64() else {
        return false;
    };
    let rho = re.hypot(im);
    if !rho.is_finite() || rho >= (s_f + 1.0) / 2.0 {
        return false;
    }
    if rho == 0.0 {
        // Below the range of `f64`: the series' first term is all there is.
        return true;
    }
    // ln Γ(s + 1) for an integer or half-integer s ≤ MAX_SPECIAL_EXPANSION.
    let mut ln_gamma = if s.is_integer() {
        0.0
    } else {
        0.5 * std::f64::consts::PI.ln()
    };
    let mut k = if s.is_integer() { 2.0 } else { 0.5 };
    while k <= s_f + 0.25 {
        ln_gamma += f64::ln(k);
        k += 1.0;
    }
    let loss = ln_gamma - s_f * rho.ln() + re + (rho - re).max(0.0)
        - ((s_f + 1.0 - 2.0 * rho) / (s_f + 1.0 - rho)).ln();
    loss / std::f64::consts::LN_2 > LOWERGAMMA_FOLD_MAX_LOSS_BITS
}

/// `γ(s, 0) = 0`, `γ(s, ∞) = Γ(s)`, closed forms for integer / half-integer `s > 0`.
///
/// `γ(s, 0) = 0` holds only for `re s > 0` (`∫₀ˣ t^{s−1} e^{−t} dt` diverges at
/// the lower end otherwise: `γ(s, x) ~ xˢ/s`), so that fold needs `re s > 0`
/// known — a positive number, the assumptions, or the certified value of a
/// constant.  Up to 0.58 it was refused only for a number `≤ 0`:
/// `lowergamma(−√2, 0)`, `lowergamma(i, 0)` and `lowergamma(y, 0)` for a
/// plain `y` were 0, and `limit(γ(y, x), x, 0)` was `Piecewise((0, y ≠ 0))`
/// (`γ(−1/2, x) ~ −2/√x`).
/// The closed form is also refused at a real constant `x` where it would
/// cancel catastrophically ([`lowergamma_closure_cancels`]): `γ(s, x)`
/// stays, and `evalf` computes it by its power series.  An irrational `x`
/// (`√2/10³⁰`, `π/10¹⁰`) is sized by a 16-digit evaluation.
fn eval_lowergamma(arena: &mut Arena, s: ExprId, x: ExprId) -> Option<ExprId> {
    // Both folds hold for a finite `s` only (`lowergamma(zoo, 0)` was 0).
    let at_infinity = ValueClass::special_atom(arena, x) == Some(ValueClass::PosInf);
    if (x == arena.zero || at_infinity) && !ValueClass::of(arena, s).is_finite_proved() {
        return None;
    }
    if x == arena.zero {
        let positive_real_part = match arena.as_num(s) {
            Some(r) => r.is_positive(),
            None => {
                let re = arena.re(s);
                let mut cache = crate::base::assumptions::AssumptionCache::new();
                crate::base::truth::Truth::from(cache.query(
                    arena,
                    re,
                    crate::base::assumptions::Props::POSITIVE,
                ))
                .is_true()
                    || certified_part_signs(arena, s)
                        .is_some_and(|p| p.re == Some(std::cmp::Ordering::Greater))
            }
        };
        return positive_real_part.then_some(arena.zero);
    }
    if at_infinity {
        return Some(arena.gamma(s));
    }
    let r = as_ratio(arena, s)?;
    if !r.is_positive() {
        return None;
    }
    if let Some(xr) = as_ratio(arena, x) {
        if lowergamma_closure_cancels(&r, &xr) {
            return None;
        }
        if !xr.is_positive()
            && let Some(x_f) = xr.to_f64()
            && lowergamma_closure_cancels_complex(&r, x_f, 0.0)
        {
            return None;
        }
    } else if let Some(ln_x) = crate::transforms::evalf::ln_of_positive_constant(arena, x) {
        if r.to_f64().is_some_and(|s_f| ln_x < s_f.ln())
            && lowergamma_closure_cancels_at(&r, ln_x.exp(), ln_x)
        {
            return None;
        }
    } else if walk::free_symbols(arena, x).is_empty()
        && let Ok(z) = crate::transforms::evalf::evalf_complex64(arena, x)
        && lowergamma_closure_cancels_complex(&r, z.re, z.im)
    {
        return None;
    }
    if value_class::holds_real_infinity(arena, x) {
        return None;
    }
    let upper = uppergamma_closed(arena, &r, x)?;
    let gamma_s = eval_gamma(arena, s).unwrap_or_else(|| arena.gamma(s));
    let v = arena.sub(gamma_s, upper);
    let v = crate::transforms::expand::expand(arena, v);
    Some(eval(arena, v))
}

/// `Γ(s, 0) = Γ(s)`, `Γ(s, ∞) = 0`, closed forms for integer / half-integer `s`.
///
/// `Γ(s, 0) = ∫₀^∞ t^{s−1} e^{−t} dt` diverges for `s ≤ 0` (`Γ(s)` there is
/// the analytic continuation, not this integral), so the fold at `x = 0` is
/// refused for `s` known to be non-positive.  The closed forms are not
/// taken at another infinite argument (up to 0.51 `(x + 1)·e^{−x}` at
/// `x = −∞` gave `nan` for `uppergamma(2, −∞)`, whose limit is `−∞`);
/// the values at `−∞` (`Γ(2, −∞) = −∞`, `Γ(1/2, −∞) = √π − i·∞`) fold
/// when the node is rebuilt (`canon::lib_function_value`).
fn eval_uppergamma(arena: &mut Arena, s: ExprId, x: ExprId) -> Option<ExprId> {
    // `Γ(s, ∞) = 0` and `Γ(s, 0) = Γ(s)` for a finite `s` only: up to 0.56
    // `uppergamma(elliptic_k(besseli(0, 0)), ∞)` (the order evaluates to
    // `K(1) = zoo`) was 0 and `is_zero()` claimed it.
    let at_infinity = ValueClass::special_atom(arena, x) == Some(ValueClass::PosInf);
    if (x == arena.zero || at_infinity) && !ValueClass::of(arena, s).is_finite_proved() {
        return None;
    }
    if at_infinity {
        return Some(arena.zero);
    }
    if x == arena.zero {
        if known_nonpositive(arena, s) {
            return None;
        }
        // `Γ(s)` evaluated (`uppergamma(4, 0)` is 6, as SymPy gives it;
        // up to 0.46 it was `Gamma(4)` until a second `eval`).
        return Some(eval_gamma(arena, s).unwrap_or_else(|| arena.gamma(s)));
    }
    let r = as_ratio(arena, s)?;
    // (`Γ(0, x) = E₁(x)` only renames.)
    if !r.is_zero() && value_class::holds_real_infinity(arena, x) {
        return None;
    }
    uppergamma_closed(arena, &r, x)
}

/// Exact polylogarithm values (see [`Ex::polylog`](crate::expr::Ex::polylog)).
fn eval_polylog(arena: &mut Arena, s: ExprId, z: ExprId) -> Option<ExprId> {
    if z == arena.zero {
        return Some(arena.zero);
    }
    if z == arena.one {
        return Some(arena.zeta(s));
    }
    if z == arena.neg_one {
        let eta = eval_dirichlet_eta(arena, s)
            .unwrap_or_else(|| apply_named(arena, LibFn::DirichletEta, &[s]));
        return Some(arena.neg(eta));
    }
    // `−ln(1 − z)` keeps the finite part of an infinite `z` (`Li₁(2 + i·∞) =
    // −∞ + iπ/2`); the rational closed forms of `s ≤ 0` would not.
    if s != arena.one
        && (value_class::holds_real_infinity(arena, s)
            || value_class::holds_real_infinity(arena, z))
    {
        return None;
    }
    if s == arena.one {
        let one_minus_z = arena.sub(arena.one, z);
        let l = eval_ln(arena, one_minus_z).unwrap_or_else(|| arena.ln(one_minus_z));
        return Some(arena.neg(l));
    }
    if s == arena.zero {
        let one_minus_z = arena.sub(arena.one, z);
        return Some(arena.div(z, one_minus_z));
    }
    // Li₂(1/2) = π²/12 − ln²2/2
    if as_int_in(arena, s, 2, 2).is_some()
        && let Some(r) = as_ratio(arena, z)
        && r == Ratio::new(BigInt::one(), BigInt::from(2))
    {
        let two = arena.int(2);
        let pi2 = arena.pow(arena.pi, two);
        let c12 = arena.rational(1, 12);
        let a = arena.mul(&[c12, pi2]);
        let ln2 = arena.ln(two);
        let ln2_sq = arena.pow(ln2, two);
        let half = arena.rational(1, 2);
        let b = arena.mul(&[half, ln2_sq]);
        return Some(arena.sub(a, b));
    }
    if as_int_in(arena, s, 2, 2).is_some()
        && let Some(value) = dilog_special_value(arena, z)
    {
        return Some(value);
    }
    // Li₃(1/2) = 7ζ(3)/8 − π²·ln 2/12 + ln³2/6 (Lewin, *Polylogarithms and
    // Associated Functions*, ch. 6; checked against mpmath at 30 and 50
    // digits).
    if as_int_in(arena, s, 3, 3).is_some()
        && as_ratio(arena, z).is_some_and(|r| r == Ratio::new(BigInt::one(), BigInt::from(2)))
    {
        let two = arena.int(2);
        let three = arena.int(3);
        let zeta3 = arena.zeta(three);
        let c78 = arena.rational(7, 8);
        let a = arena.mul(&[c78, zeta3]);
        let ln2 = arena.ln(two);
        let pi2 = arena.pow(arena.pi, two);
        let c12 = arena.rational(-1, 12);
        let b = arena.mul(&[c12, pi2, ln2]);
        let ln2_cubed = arena.pow(ln2, three);
        let c6 = arena.rational(1, 6);
        let c = arena.mul(&[c6, ln2_cubed]);
        return Some(arena.add(&[a, b, c]));
    }
    // Li_{−n}(z) = [Σ_{k=0}^{n} k! S(n+1, k+1) z^{k+1} (1−z)^{n−k}] / (1−z)^{n+1}
    if let Some(neg_n) = as_int_in(arena, s, -MAX_SPECIAL_EXPANSION, -1) {
        let n = (-neg_n) as usize;
        let one_minus_z = arena.sub(arena.one, z);
        let mut terms: Vec<ExprId> = Vec::with_capacity(n + 1);
        let mut k_fact = BigInt::one();
        for k in 0..=n {
            if k > 0 {
                k_fact *= BigInt::from(k as u64);
            }
            let s2 = crate::domains::combinatorics::stirling2(n as u64 + 1, k as u64 + 1)?;
            let coeff = Ratio::from_integer(&k_fact * s2);
            let c = arena.intern_num(coeff);
            let c = arena.intern(ExprNode::Num(c));
            let e1 = arena.int(k as i64 + 1);
            let zk = arena.pow(z, e1);
            let e2 = arena.int((n - k) as i64);
            let wk = arena.pow(one_minus_z, e2);
            terms.push(arena.mul(&[c, zk, wk]));
        }
        let numer = arena.add(&terms);
        let numer = crate::transforms::expand::expand(arena, numer);
        let numer = eval(arena, numer);
        let e = arena.int(n as i64 + 1);
        let denom = arena.pow(one_minus_z, e);
        return Some(arena.div(numer, denom));
    }
    None
}

/// A dilogarithm value `a·π² + b·ln²2 + c·iπ·ln 2 + d·i·G` (`G` Catalan's
/// constant), each coefficient `[numerator, denominator]`.
struct DilogValue {
    pi_sq: [i64; 2],
    ln2_sq: [i64; 2],
    i_pi_ln2: [i64; 2],
    i_catalan: [i64; 2],
}

/// `Li₂(z)` at the classical arguments other than `1/2` (SymPy's
/// `_dilogtable`; the reflection, inversion and Landen identities of DLMF
/// §25.12(i); Lewin, *Polylogarithms and Associated Functions*, ch. 1): `2`,
/// `±i`, `1 ± i`, `(1 ± i)/2` and the four golden-ratio points `(√5 − 1)/2`,
/// `(3 − √5)/2`, `(1 − √5)/2`, `−(1 + √5)/2`.  On the cut `z = 2` the value
/// is the one the numerical evaluation (and `mpmath.polylog`) gives,
/// `π²/4 − iπ·ln 2`.  There is no such value at `−1/2` (Landen's identity
/// only relates it to `Li₂(1/3)`), so it stays.
fn dilog_special_value(arena: &mut Arena, z: ExprId) -> Option<ExprId> {
    if let Some((re, im)) = gaussian_rational(arena, z) {
        let twice = |q: &Q| -> Option<i64> {
            let t = q * Q::from_integer(BigInt::from(2));
            if t.is_integer() {
                t.to_integer().to_i64()
            } else {
                None
            }
        };
        let v = |pi_sq, ln2_sq, i_pi_ln2, i_catalan| DilogValue {
            pi_sq,
            ln2_sq,
            i_pi_ln2,
            i_catalan,
        };
        let zero = [0, 1];
        let value = match (twice(&re)?, twice(&im)?) {
            (4, 0) => v([1, 4], zero, [-1, 1], zero),
            (0, 2) => v([-1, 48], zero, zero, [1, 1]),
            (0, -2) => v([-1, 48], zero, zero, [-1, 1]),
            (2, -2) => v([1, 16], zero, [-1, 4], [-1, 1]),
            (2, 2) => v([1, 16], zero, [1, 4], [1, 1]),
            (1, -1) => v([5, 96], [-1, 8], [1, 8], [-1, 1]),
            (1, 1) => v([5, 96], [-1, 8], [-1, 8], [1, 1]),
            _ => return None,
        };
        return Some(dilog_combination(arena, &value));
    }
    if !matches!(arena.node(z), ExprNode::Add(cs) if cs.len() == 2) {
        return None;
    }
    let (one, two) = (arena.one, arena.int(2));
    let half = arena.rational(1, 2);
    let sqrt5 = arena.int(5);
    let sqrt5 = arena.pow(sqrt5, half);
    let neg_sqrt5 = arena.neg(sqrt5);
    // g = (√5 − 1)/2 = 1/φ and φ = (√5 + 1)/2.
    let g = arena.add(&[sqrt5, arena.neg_one]);
    let g = arena.div(g, two);
    let phi = arena.add(&[sqrt5, one]);
    let phi = arena.div(phi, two);
    let three = arena.int(3);
    let g_sq = arena.add(&[three, neg_sqrt5]);
    let g_sq = arena.div(g_sq, two);
    let neg_g = arena.neg(g);
    let neg_phi = arena.neg(phi);
    // c·π² + d·ln²(w)
    let (pi_sq, w, ln_coeff) = if z == g {
        ([1, 10], g, [-1, 1])
    } else if z == g_sq {
        ([1, 15], g, [-1, 1])
    } else if z == neg_g {
        ([-1, 15], g, [1, 2])
    } else if z == neg_phi {
        ([-1, 10], phi, [-1, 1])
    } else {
        return None;
    };
    let pi2 = arena.pow(arena.pi, two);
    let c = arena.rational(pi_sq[0], pi_sq[1]);
    let a = arena.mul(&[c, pi2]);
    let ln_w = arena.ln(w);
    let ln_w_sq = arena.pow(ln_w, two);
    let d = arena.rational(ln_coeff[0], ln_coeff[1]);
    let b = arena.mul(&[d, ln_w_sq]);
    Some(arena.add(&[a, b]))
}

/// The sum [`DilogValue`] stands for.
fn dilog_combination(arena: &mut Arena, value: &DilogValue) -> ExprId {
    let two = arena.int(2);
    let pi_sq = arena.pow(arena.pi, two);
    let ln2 = arena.ln(two);
    let ln2_sq = arena.pow(ln2, two);
    let i_pi_ln2 = arena.mul(&[arena.i_unit, arena.pi, ln2]);
    let i_catalan = arena.mul(&[arena.i_unit, arena.catalan]);
    let mut terms: Vec<ExprId> = Vec::with_capacity(4);
    for (c, t) in [
        (value.pi_sq, pi_sq),
        (value.ln2_sq, ln2_sq),
        (value.i_pi_ln2, i_pi_ln2),
        (value.i_catalan, i_catalan),
    ] {
        if c[0] != 0 {
            let c = arena.rational(c[0], c[1]);
            terms.push(arena.mul(&[c, t]));
        }
    }
    arena.add(&terms)
}

/// `η(1) = ln 2`, `η(∞) = 1`, `η(−2k) = 0`, and `η(s) = (1 − 2^{1−s}) ζ(s)`
/// whenever `ζ(s)` itself folds (integer `s`); negative integers below
/// `−MAX_SPECIAL_EXPANSION` are left alone (the factor `2^{1−s}` and the
/// Bernoulli number behind `ζ(s)` both grow with `|s|`).
fn eval_dirichlet_eta(arena: &mut Arena, s: ExprId) -> Option<ExprId> {
    if s == arena.one {
        let two = arena.int(2);
        return Some(arena.ln(two));
    }
    // (`η(−∞)`, `η(zoo)`: no limit; `nan` folds when built)
    if ValueClass::special_atom(arena, s) == Some(ValueClass::PosInf) {
        return Some(arena.one);
    }
    let r = as_ratio(arena, s)?;
    if !r.is_integer() {
        return None;
    }
    let n: i64 = r.to_integer().try_into().ok()?;
    if n < -MAX_SPECIAL_EXPANSION {
        // Trivial zeros of ζ are the only cheap exact values out here.
        return (n % 2 == 0).then_some(arena.zero);
    }
    let zeta = eval_zeta(arena, s)?;
    if matches!(arena.node(zeta), ExprNode::Zeta(_)) {
        return None;
    }
    // 1 − 2^{1−s} as an exact rational.
    let e = 1 - n;
    let two_pow = if e >= 0 {
        Ratio::from_integer(BigInt::one() << (e as usize))
    } else {
        Ratio::new(BigInt::one(), BigInt::one() << ((-e) as usize))
    };
    let coeff = Ratio::one() - two_pow;
    let c = arena.intern_num(coeff);
    let c = arena.intern(ExprNode::Num(c));
    Some(arena.mul(&[c, zeta]))
}

/// Airy functions at `0` and `±∞`.
///
/// At `−∞` only `Ai` and `Bi` tend to `0` (`Ai(−x) ~ x^(−1/4)·sin(⅔x^(3/2) +
/// π/4)/√π`); their derivatives oscillate with the growing amplitude
/// `x^(1/4)/√π` (DLMF 9.7.10, 9.7.12) and have no limit, so `Ai′(−∞)` and
/// `Bi′(−∞)` stay (SymPy 1.14 leaves `airyaiprime(-oo)`; its
/// `airybiprime(-oo)` → 0 is wrong).  Before 0.41 both folded to `0`.
fn eval_airy(arena: &mut Arena, f: LibFn, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        let three = arena.int(3);
        let third = arena.rational(1, 3);
        let two_thirds = arena.rational(2, 3);
        let g13 = arena.gamma(third);
        let g23 = arena.gamma(two_thirds);
        return Some(match f {
            LibFn::AiryAi => {
                // 3^{-2/3} / Γ(2/3)
                let e = arena.rational(-2, 3);
                let p = arena.pow(three, e);
                arena.div(p, g23)
            }
            LibFn::AiryBi => {
                // 3^{-1/6} / Γ(2/3)
                let e = arena.rational(-1, 6);
                let p = arena.pow(three, e);
                arena.div(p, g23)
            }
            LibFn::AiryAiPrime => {
                // -3^{-1/3} / Γ(1/3)
                let e = arena.rational(-1, 3);
                let p = arena.pow(three, e);
                let q = arena.div(p, g13);
                arena.neg(q)
            }
            _ => {
                // 3^{1/6} / Γ(1/3)
                let e = arena.rational(1, 6);
                let p = arena.pow(three, e);
                arena.div(p, g13)
            }
        });
    }
    match ValueClass::special_atom(arena, x) {
        Some(ValueClass::PosInf) => Some(match f {
            LibFn::AiryAi | LibFn::AiryAiPrime => arena.zero,
            _ => arena.infinity,
        }),
        Some(ValueClass::NegInf) if matches!(f, LibFn::AiryAi | LibFn::AiryBi) => Some(arena.zero),
        // (`Ai′(−∞)`, `Bi′(−∞)` oscillate without bound; `zoo`: no limit)
        _ => None,
    }
}

/// `K(0) = π/2`, `K(1) = z∞`, `K(1/2) = Γ(1/4)²/(4√π)`, `K(±∞) = 0`.
fn eval_elliptic_k(arena: &mut Arena, m: ExprId) -> Option<ExprId> {
    if m == arena.zero {
        return Some(half_pi(arena));
    }
    if m == arena.one {
        return Some(arena.complex_infinity);
    }
    if ValueClass::special_atom(arena, m).is_some_and(|c| c.real_infinity().is_some()) {
        return Some(arena.zero);
    }
    if let Some(r) = as_ratio(arena, m)
        && r == Ratio::new(BigInt::one(), BigInt::from(2))
    {
        let quarter = arena.rational(1, 4);
        let g = arena.gamma(quarter);
        let two = arena.int(2);
        let g2 = arena.pow(g, two);
        let sqrt_pi = arena.sqrt(arena.pi);
        let four = arena.int(4);
        let denom = arena.mul(&[four, sqrt_pi]);
        return Some(arena.div(g2, denom));
    }
    None
}

/// `E(0) = π/2`, `E(1) = 1`.
fn eval_elliptic_e(arena: &mut Arena, m: ExprId) -> Option<ExprId> {
    if m == arena.zero {
        return Some(half_pi(arena));
    }
    if m == arena.one {
        return Some(arena.one);
    }
    None
}

/// `F(0 | m) = 0`, `F(φ | 0) = φ`, `F(π/2 | m) = K(m)`, odd in `φ`.
fn eval_elliptic_f(arena: &mut Arena, phi: ExprId, m: ExprId) -> Option<ExprId> {
    if phi == arena.zero {
        return Some(arena.zero);
    }
    if m == arena.zero {
        return Some(phi);
    }
    if let Some(r) = as_pi_multiple(arena, phi)
        && r == Ratio::new(BigInt::one(), BigInt::from(2))
    {
        return Some(
            eval_elliptic_k(arena, m).unwrap_or_else(|| apply_named(arena, LibFn::EllipticK, &[m])),
        );
    }
    if let Some(y) = as_negated_general(arena, phi) {
        let e = eval_elliptic_f(arena, y, m)
            .unwrap_or_else(|| apply_named(arena, LibFn::EllipticF, &[y, m]));
        return Some(arena.neg(e));
    }
    None
}

/// `Π(0 | m) = K(m)`, `Π(n | 0) = π/(2√(1−n))`, `Π(n | n) = E(n)/(1−n)`, `Π(1 | m) = z∞`.
fn eval_elliptic_pi(arena: &mut Arena, n: ExprId, m: ExprId) -> Option<ExprId> {
    if n == arena.zero {
        return Some(
            eval_elliptic_k(arena, m).unwrap_or_else(|| apply_named(arena, LibFn::EllipticK, &[m])),
        );
    }
    // `Π(n | n) = E(n)/(1 − n)` for a finite `n` only (`Π(1 | m) = zoo` for
    // every `m`, so also in the limit `m → ∞`).
    if n == arena.one {
        return Some(arena.complex_infinity);
    }
    if m == arena.zero {
        let one_minus_n = arena.sub(arena.one, n);
        let root = arena.sqrt(one_minus_n);
        let two = arena.int(2);
        let denom = arena.mul(&[two, root]);
        return Some(arena.div(arena.pi, denom));
    }
    if n == m && ValueClass::of(arena, n).is_finite_proved() {
        let e =
            eval_elliptic_e(arena, n).unwrap_or_else(|| apply_named(arena, LibFn::EllipticE, &[n]));
        let one_minus_n = arena.sub(arena.one, n);
        return Some(arena.div(e, one_minus_n));
    }
    None
}

/// Degree parameter of an orthogonal polynomial: a non-negative integer no
/// larger than the configured power-expansion limit.
fn as_poly_degree(arena: &Arena, n: ExprId) -> Option<usize> {
    let r = arena.as_num(n)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let d = r.to_integer().to_u64()?;
    (d <= arena.config.max_pow_exponent as u64).then_some(d as usize)
}

/// Largest degree for which the parametrised orthogonal polynomials are
/// expanded when a parameter (`a`, `b`) is not a plain number: every step
/// then expands a multivariate polynomial in the parameters and `x`, whose
/// size (and the cost of expanding it) grows quickly with the degree.
/// Numeric parameters keep the `max_pow_exponent` bound of
/// [`as_poly_degree`].
const MAX_SYMBOLIC_PARAM_DEGREE: usize = 16;

/// [`as_poly_degree`], further capped at [`MAX_SYMBOLIC_PARAM_DEGREE`]
/// unless every parameter in `params` is numeric.
fn as_param_poly_degree(arena: &Arena, n: ExprId, params: &[ExprId]) -> Option<usize> {
    let deg = as_poly_degree(arena, n)?;
    let numeric = params.iter().all(|&p| arena.as_num(p).is_some());
    (numeric || deg <= MAX_SYMBOLIC_PARAM_DEGREE).then_some(deg)
}

/// The degree for the recurrence path of the parametrised polynomials:
/// symbolic parameters (see [`as_param_poly_degree`]) or numeric ones
/// where the explicit coefficients degenerate (a non-positive integer
/// Gegenbauer `α`, a negative integer Laguerre / Jacobi `α`), capped at
/// [`MAX_DEGENERATE_DEGREE`] for the latter since the recurrence
/// re-expands the polynomial at every step.
fn degenerate_param_degree(arena: &Arena, n: ExprId, params: &[ExprId]) -> Option<usize> {
    let deg = as_param_poly_degree(arena, n, params)?;
    let numeric = params.iter().all(|&p| arena.as_num(p).is_some());
    (!numeric || deg <= MAX_DEGENERATE_DEGREE).then_some(deg)
}

/// Largest degree expanded by the recurrences for degenerate numeric
/// parameters (see [`degenerate_param_degree`]).
const MAX_DEGENERATE_DEGREE: usize = 64;

/// Expand and fold one recurrence step.
pub(super) fn expand_eval(arena: &mut Arena, e: ExprId) -> ExprId {
    let e = crate::transforms::expand::expand(arena, e);
    eval(arena, e)
}

/// `p · q` distributed (both already expanded, so like terms are merged
/// at every step instead of once at the end); not folded.
fn expand_mul2(arena: &mut Arena, p: ExprId, q: ExprId) -> ExprId {
    let m = arena.mul(&[p, q]);
    crate::transforms::expand::expand(arena, m)
}

/// Gegenbauer `C_n^{(a)}(x)`: explicit polynomial for integer `n ≥ 0` via
/// `(k+1) C_{k+1} = 2(k+a) x C_k − (k+2a−1) C_{k−1}`; `C_n^{(1/2)} = P_n`,
/// `C_n^{(1)} = U_n`.
fn eval_gegenbauer(arena: &mut Arena, n: ExprId, a: ExprId, x: ExprId) -> Option<ExprId> {
    if zero_polynomial_degree(arena, n, &[a]) {
        return Some(arena.zero);
    }
    match expansion(arena, &[a], n, x) {
        Expansion::Refused => return None,
        Expansion::IdentitiesOnly => {
            return match as_ratio(arena, a) {
                Some(r) if r == Ratio::new(BigInt::one(), BigInt::from(2)) => {
                    Some(arena.legendre(n, x))
                }
                Some(r) if r.is_one() => Some(arena.chebyshev_u(n, x)),
                _ => None,
            };
        }
        Expansion::Expanded => {}
    }
    if let (Some(deg), Some(alpha)) = (as_poly_degree(arena, n), as_ratio(arena, a))
        && !(alpha.is_integer() && !alpha.is_positive())
    {
        return ortho_value(arena, &OrthoFamily::Gegenbauer(alpha), deg, x);
    }
    if let Some(deg) = degenerate_param_degree(arena, n, &[a]) {
        if deg == 0 {
            return Some(arena.one);
        }
        let two = arena.int(2);
        let c1 = arena.mul(&[two, a, x]);
        if deg == 1 {
            return Some(expand_eval(arena, c1));
        }
        let mut prev = arena.one;
        let mut curr = expand_eval(arena, c1);
        for k in 1..deg {
            let k_id = arena.int(k as i64);
            let k_plus_a = arena.add(&[k_id, a]);
            let t1 = arena.mul(&[two, k_plus_a, x, curr]);
            let two_a = arena.mul(&[two, a]);
            let km1 = arena.int(k as i64 - 1);
            let coeff = arena.add(&[km1, two_a]);
            let t2 = arena.mul(&[coeff, prev]);
            let numer = arena.sub(t1, t2);
            let kp1 = arena.int(k as i64 + 1);
            let next = arena.div(numer, kp1);
            let next = expand_eval(arena, next);
            prev = curr;
            curr = next;
        }
        return Some(curr);
    }
    if let Some(r) = as_ratio(arena, a) {
        if r == Ratio::new(BigInt::one(), BigInt::from(2)) {
            return Some(arena.legendre(n, x));
        }
        if r.is_one() {
            return Some(arena.chebyshev_u(n, x));
        }
    }
    None
}

/// `[1, p, p², …, p^n]` with every power expanded (each from the previous
/// one, so intermediate like terms are merged).
fn expanded_powers(arena: &mut Arena, p: ExprId, n: usize) -> Vec<ExprId> {
    let mut pows: Vec<ExprId> = Vec::with_capacity(n + 1);
    pows.push(arena.one);
    for k in 1..=n {
        let next = expand_mul2(arena, pows[k - 1], p);
        pows.push(next);
    }
    pows
}

/// Jacobi `P_n^{(a,b)}(x)` for integer `n ≥ 0`:
/// `Σ_{s=0}^{n} C(n+a, n−s) C(n+b, s) ((x−1)/2)^s ((x+1)/2)^{n−s}` with the
/// binomials written as polynomials in `a`, `b`; `P_n^{(0,0)} = P_n`.
///
/// Every factor (`∏ (a+j)`, `∏ (b+j)`, the two powers) is expanded
/// incrementally and the summands are multiplied out pairwise, so like
/// terms are merged at each step; distributing the whole `2n`-factor
/// product at once is exponential in `n`.
fn eval_jacobi(arena: &mut Arena, n: ExprId, a: ExprId, b: ExprId, x: ExprId) -> Option<ExprId> {
    if zero_polynomial_degree(arena, n, &[a, b]) {
        return Some(arena.zero);
    }
    match expansion(arena, &[a, b], n, x) {
        Expansion::Refused => return None,
        Expansion::IdentitiesOnly => {
            return (a == arena.zero && b == arena.zero).then(|| arena.legendre(n, x));
        }
        Expansion::Expanded => {}
    }
    if let (Some(deg), Some(ar), Some(br)) = (
        as_poly_degree(arena, n),
        as_ratio(arena, a),
        as_ratio(arena, b),
    ) && !(ar.is_integer() && ar.is_negative())
    {
        return jacobi_value(arena, deg, &ar, &br, x);
    }
    if let Some(deg) = degenerate_param_degree(arena, n, &[a, b]) {
        if deg == 0 {
            return Some(arena.one);
        }
        let half = arena.rational(1, 2);
        let x_minus_1 = arena.sub(x, arena.one);
        let x_plus_1 = arena.add(&[x, arena.one]);
        let lo = arena.mul(&[half, x_minus_1]);
        let lo = expand_eval(arena, lo);
        let hi = arena.mul(&[half, x_plus_1]);
        let hi = expand_eval(arena, hi);
        let lo_pows = expanded_powers(arena, lo, deg);
        let hi_pows = expanded_powers(arena, hi, deg);
        // pa[s] = ∏_{j=s+1}^{n} (a+j)  (pa[n] = 1, pa[s] = pa[s+1]·(a+s+1))
        let mut pa: Vec<ExprId> = vec![arena.one; deg + 1];
        for s in (0..deg).rev() {
            let j = arena.int(s as i64 + 1);
            let factor = arena.add(&[a, j]);
            pa[s] = expand_mul2(arena, pa[s + 1], factor);
        }
        // pb[s] = ∏_{j=n−s+1}^{n} (b+j)  (pb[0] = 1, pb[s] = pb[s−1]·(b+n−s+1))
        let mut pb: Vec<ExprId> = vec![arena.one; deg + 1];
        for s in 1..=deg {
            let j = arena.int((deg - s) as i64 + 1);
            let factor = arena.add(&[b, j]);
            pb[s] = expand_mul2(arena, pb[s - 1], factor);
        }
        let mut terms: Vec<ExprId> = Vec::with_capacity(deg + 1);
        for s in 0..=deg {
            // C(n+a, n−s) C(n+b, s) = pa[s] pb[s] / ((n−s)! s!)
            let denom = factorial_ratio(deg - s) * factorial_ratio(s);
            let c = arena.intern_num(denom.recip());
            let c = arena.intern(ExprNode::Num(c));
            let ab = arena.mul(&[c, pa[s]]);
            let ab = expand_mul2(arena, ab, pb[s]);
            // ((x−1)/2)^s ((x+1)/2)^{n−s}: `n+1` terms once merged.
            let xp = expand_mul2(arena, lo_pows[s], hi_pows[deg - s]);
            terms.push(expand_mul2(arena, ab, xp));
        }
        // Every summand is already distributed; `add` merges like terms.
        let sum = arena.add(&terms);
        return Some(eval(arena, sum));
    }
    if a == arena.zero && b == arena.zero {
        return Some(arena.legendre(n, x));
    }
    None
}

/// Associated Legendre `P_n^m(x)` (Condon–Shortley phase) for integer
/// `n ≥ 0`, integer `m`: zero for `m > n` (the `m`-th derivative of a
/// polynomial of degree `n`; for `m < −n` the Ferrers function is not 0,
/// `P₀⁻¹(x) = √((1 − x)/(1 + x))`, DLMF 14.5.12, and stays: up to 0.56 it was
/// 0), `P_n^0 = P_n`,
/// `P_n^{−m} = (−1)^m (n−m)!/(n+m)! P_n^m`, and for `m > 0`
/// `P_m^m = (−1)^m (2m−1)!! (1−x²)^{m/2}`, `P_{m+1}^m = (2m+1) x P_m^m`,
/// `(k−m+1) P_{k+1}^m = (2k+1) x P_k^m − (k+m) P_{k−1}^m`.
fn eval_assoc_legendre(arena: &mut Arena, n: ExprId, m: ExprId, x: ExprId) -> Option<ExprId> {
    match expansion(arena, &[m], n, x) {
        Expansion::Refused => return None,
        Expansion::IdentitiesOnly => {
            if m == arena.zero {
                return Some(arena.legendre(n, x));
            }
            // `P_n^m = 0` for integers `m > n`.
            let (deg, order) = (as_poly_degree(arena, n)?, as_ratio(arena, m)?);
            return (order.is_integer() && order > q_int(deg as i64)).then_some(arena.zero);
        }
        Expansion::Expanded => {}
    }
    // `P^m_{−ν−1} = P^m_ν` (DLMF 14.9.5): a negative degree `−k` is `k − 1`.
    if let Some(neg) = arena
        .as_num(n)
        .filter(|q| q.is_integer() && q.is_negative())
        .map(|q| q.to_integer())
    {
        let d = arena.num_ratio(Q::from_integer(-neg - 1));
        return Some(
            eval_assoc_legendre(arena, d, m, x)
                .unwrap_or_else(|| apply_named(arena, LibFn::AssocLegendre, &[d, m, x])),
        );
    }
    let deg = match as_poly_degree(arena, n) {
        Some(d) => d,
        None => {
            return (m == arena.zero).then(|| arena.legendre(n, x));
        }
    };
    let order = as_int_in(arena, m, -(deg as i64), deg as i64);
    let order = match order {
        Some(o) => o,
        None => {
            // Integer m > n ↦ 0; m < −n and a non-integer m stay symbolic.
            let r = as_ratio(arena, m)?;
            return (r.is_integer() && r.is_positive()).then_some(arena.zero);
        }
    };
    if order == 0 {
        return ortho_value(arena, &OrthoFamily::Legendre, deg, x);
    }
    let mu = order.unsigned_abs() as usize;
    // (1 − x²)^{μ/2}
    let two = arena.int(2);
    let x2 = arena.pow(x, two);
    let one_minus_x2 = arena.sub(arena.one, x2);
    let half_mu = arena.rational(mu as i64, 2);
    let root_pow = arena.pow(one_minus_x2, half_mu);
    // P_n^μ = (−1)^μ (1 − x²)^{μ/2} d^μP_n/dx^μ (Condon–Shortley, DLMF
    // 14.6.1 with Ferrers' sign): the polynomial part from the explicit
    // Legendre coefficients, c[i]·i!/(i−μ)! at x^{i−μ}.
    let c = ortho_coeffs(arena, &OrthoFamily::Legendre, deg)?;
    let mut r = vec![Q::zero(); deg - mu + 1];
    let mut falling = factorial_ratio(mu); // i!/(i−μ)! at i = μ
    for i in mu..=deg {
        if i > mu {
            falling = falling * q_int(i as i64) / q_int((i - mu) as i64);
        }
        let mut v = &c[i] * &falling;
        if mu % 2 == 1 {
            v = -v;
        }
        if !within_guard_bits(arena, &v) {
            return None;
        }
        r[i - mu] = v;
    }
    let poly = match arena.as_num(x).cloned() {
        Some(xr) => {
            let v = horner(arena, &r, &xr)?;
            guarded_num(arena, v)?
        }
        None => poly_from_coeffs(arena, &r, x)?,
    };
    let prod = arena.mul(&[poly, root_pow]);
    let value = expand_eval(arena, prod);
    if order < 0 {
        // P_n^{−μ} = (−1)^μ (n−μ)!/(n+μ)! P_n^μ
        let mut ratio = factorial_ratio(deg - mu) / factorial_ratio(deg + mu);
        if mu % 2 == 1 {
            ratio = -ratio;
        }
        let c = arena.intern_num(ratio);
        let c = arena.intern(ExprNode::Num(c));
        let v = arena.mul(&[c, value]);
        return Some(expand_eval(arena, v));
    }
    Some(value)
}

/// Generalised Laguerre `L_n^{(a)}(x)` for integer `n ≥ 0` via
/// `(k+1) L_{k+1} = (2k+1+a−x) L_k − (k+a) L_{k−1}`; `L_n^{(0)} = L_n`.
fn eval_assoc_laguerre(arena: &mut Arena, n: ExprId, a: ExprId, x: ExprId) -> Option<ExprId> {
    if zero_polynomial_degree(arena, n, &[a]) {
        return Some(arena.zero);
    }
    match expansion(arena, &[a], n, x) {
        Expansion::Refused => return None,
        Expansion::IdentitiesOnly => return (a == arena.zero).then(|| arena.laguerre(n, x)),
        Expansion::Expanded => {}
    }
    if let (Some(deg), Some(alpha)) = (as_poly_degree(arena, n), as_ratio(arena, a))
        && !(alpha.is_integer() && alpha.is_negative())
    {
        return ortho_value(arena, &OrthoFamily::AssocLaguerre(alpha), deg, x);
    }
    if let Some(deg) = degenerate_param_degree(arena, n, &[a]) {
        if deg == 0 {
            return Some(arena.one);
        }
        let one_plus_a = arena.add(&[arena.one, a]);
        let l1 = arena.sub(one_plus_a, x);
        let l1 = expand_eval(arena, l1);
        if deg == 1 {
            return Some(l1);
        }
        let mut prev = arena.one;
        let mut curr = l1;
        for k in 1..deg {
            let two_k_p1 = arena.int(2 * k as i64 + 1);
            let coeff = arena.add(&[two_k_p1, a]);
            let coeff = arena.sub(coeff, x);
            let t1 = arena.mul(&[coeff, curr]);
            let k_id = arena.int(k as i64);
            let k_plus_a = arena.add(&[k_id, a]);
            let t2 = arena.mul(&[k_plus_a, prev]);
            let numer = arena.sub(t1, t2);
            let kp1 = arena.int(k as i64 + 1);
            let next = arena.div(numer, kp1);
            let next = expand_eval(arena, next);
            prev = curr;
            curr = next;
        }
        return Some(curr);
    }
    if a == arena.zero {
        return Some(arena.laguerre(n, x));
    }
    None
}

/// Largest positive integer parameter `a`, `b` for which
/// `betainc(a, b, x1, x2)` is expanded into an explicit polynomial
/// (degree `a + b − 1`, `b` monomials in each limit).
const MAX_BETAINC_POLY_PARAM: i64 = 20;

/// Exact values of the generalised incomplete beta functions
/// `B_{(x1, x2)}(a, b) = ∫_{x1}^{x2} t^{a−1}(1−t)^{b−1} dt` and
/// `I_{(x1, x2)}(a, b) = B_{(x1, x2)}(a, b) / B(a, b)` (`regularized`):
///
/// * `x1 = x2` → `0`;
/// * `(x1, x2) = (0, 1)` → `B(a, b)` resp. `1` (the integral converges only
///   for `a, b > 0`, so this is refused for parameters known to be `≤ 0`);
/// * positive integers `a, b ≤ MAX_BETAINC_POLY_PARAM` → the integrand is
///   the polynomial `Σ_k C(b−1, k) (−1)^k t^{a−1+k}`, so the value is
///   `F(x2) − F(x1)` with `F(t) = Σ_k C(b−1, k) (−1)^k t^{a+k}/(a+k)`
///   (divided by the exact rational `B(a, b)` when regularised), expanded
///   and folded.  This is what makes the Beta-distribution CDF close for
///   integer shape parameters.
fn eval_betainc(
    arena: &mut Arena,
    regularized: bool,
    a: ExprId,
    b: ExprId,
    x1: ExprId,
    x2: ExprId,
) -> Option<ExprId> {
    if x1 == x2 {
        return Some(arena.zero);
    }
    // At an infinite bound the closed form is taken only at `±∞` itself, and
    // kept where its terms do not meet as `∞ − ∞` (`betainc_regularized(1,
    // 2, −∞, b) = ∞`; up to 0.53 `betainc(3, 2, K(x), ∞)`, whose terms
    // `t³/3 − t⁴/4` cancel so, was `nan` where the integral is `−∞`): see
    // the end.
    let infinite_bound =
        value_class::holds_real_infinity(arena, x1) || value_class::holds_real_infinity(arena, x2);
    // (finite, or `±∞` itself)
    let exact = |t: ExprId| {
        !value_class::holds_real_infinity(arena, t)
            || ValueClass::special_atom(arena, t).is_some_and(|c| c.real_infinity().is_some())
    };
    if value_class::holds_real_infinity(arena, a)
        || value_class::holds_real_infinity(arena, b)
        || !exact(x1)
        || !exact(x2)
    {
        return None;
    }
    if x1 == arena.zero && x2 == arena.one {
        if known_nonpositive(arena, a) || known_nonpositive(arena, b) {
            return None;
        }
        return Some(if regularized {
            arena.one
        } else {
            let beta = arena.beta(a, b);
            eval_beta(arena, a, b).unwrap_or(beta)
        });
    }
    let a_int = as_int_in(arena, a, 1, MAX_BETAINC_POLY_PARAM)?;
    let b_int = as_int_in(arena, b, 1, MAX_BETAINC_POLY_PARAM)?;
    // 1/B(a, b) = (a+b−1)! / ((a−1)! (b−1)!) — exact.
    let scale = if regularized {
        factorial_ratio((a_int + b_int - 1) as usize)
            / (factorial_ratio((a_int - 1) as usize) * factorial_ratio((b_int - 1) as usize))
    } else {
        Ratio::one()
    };
    // Antiderivative coefficients c_k of t^{a+k}, k = 0..b−1.
    let mut coeffs: Vec<Q> = Vec::with_capacity(crate::base::cast::bounded_len(
        b_int,
        MAX_BETAINC_POLY_PARAM,
    )?);
    let mut binom = BigInt::one();
    for k in 0..b_int {
        if k > 0 {
            // C(b−1, k) = C(b−1, k−1) · (b−k) / k
            binom = binom * BigInt::from(b_int - k) / BigInt::from(k);
        }
        let mut c = Ratio::new(binom.clone(), BigInt::from(a_int + k)) * &scale;
        if k % 2 == 1 {
            c = -c;
        }
        coeffs.push(c);
    }
    let antiderivative = |arena: &mut Arena, t: ExprId| -> ExprId {
        if t == arena.zero {
            return arena.zero;
        }
        let mut terms: Vec<ExprId> = Vec::with_capacity(coeffs.len());
        for (k, c) in coeffs.iter().enumerate() {
            let cid = arena.intern_num(c.clone());
            let cid = arena.intern(ExprNode::Num(cid));
            let e = arena.int(a_int + k as i64);
            let p = arena.pow(t, e);
            terms.push(arena.mul(&[cid, p]));
        }
        arena.add(&terms)
    };
    let upper = antiderivative(arena, x2);
    let lower = antiderivative(arena, x1);
    let v = arena.sub(upper, lower);
    let v = expand_eval(arena, v);
    // (a `nan` anywhere in the value: `∞ − ∞` met; an infinity in it is the
    // value's own, so not `value_class::holds_special_atom`)
    (!infinite_bound || !walk::contains(arena, v, arena.nan)).then_some(v)
}
