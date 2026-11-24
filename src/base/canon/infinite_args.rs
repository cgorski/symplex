//! Library functions at infinite arguments (part of [`canon`](super)).

use super::*;

// ── Library functions at infinite arguments ────────────────────────────────────────────

/// An argument of a library function as [`at_infinity`] reads it: finite
/// ([`ValueClass::is_finite_proved`]: no infinity, `zoo` or `nan` in it, and
/// a number, an argument with symbols or a constant known finite or with a
/// certified value), an exact infinity `iᵏ·∞` in an axis direction (`k`
/// quarter turns: `+∞`, `i·∞`, `−∞`, `−i·∞`), or anything else (`x + ∞`,
/// `(1 + i)·∞`, `zoo`, an unevaluated pole `elliptic_pi(π, 1)`).  Up to H1b
/// `elliptic_f(elliptic_pi(π, 1), ∞)` folded to 0 (`Π(π | 1) = −∞`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Slot {
    Finite,
    Infinite(u8),
    Other,
}

/// The [`Slot`] of the argument `a`, from its [`ValueClass`].
pub(super) fn slot(arena: &Arena, a: ExprId) -> Slot {
    match ValueClass::of(arena, a) {
        ValueClass::Finite => Slot::Finite,
        ValueClass::PosInf => Slot::Infinite(0),
        ValueClass::NegInf => Slot::Infinite(2),
        ValueClass::DirectedInf => match classify_arg(arena, a) {
            Arg::ImaginaryInfinity(false) => Slot::Infinite(1),
            Arg::ImaginaryInfinity(true) => Slot::Infinite(3),
            _ => Slot::Other,
        },
        ValueClass::ComplexInf | ValueClass::NaN | ValueClass::Infinite | ValueClass::Unknown => {
            Slot::Other
        }
    }
}

/// `Π(n | iᵗ·∞) = Π(iᵗ·∞ | m) = 0` ([`at_infinity`]) holds off the pole
/// of the other argument at 1.  `at_infinity` reads the arguments as built
/// and so excludes only the literal 1; before a constant argument folds,
/// its difference from 1 must have a certified nonzero value
/// ([`Arena::part_signs`]), so that one equal to 1 (`besseli(0, 0)`) does
/// not.  An argument with symbols is a generic point.  Up to H1b
/// `elliptic_pi(besseli(0, 0), ∞)` was built as 0 (`Π(1 | m) = zoo`: 0.53
/// evaluated it to `zoo`).
pub(super) fn off_the_pole(
    arena: &mut Arena,
    f: crate::base::libfn::LibFn,
    args: &[ExprId],
) -> bool {
    if f != crate::base::libfn::LibFn::EllipticPi {
        return true;
    }
    let constants: SmallVec<[ExprId; 2]> = args
        .iter()
        .copied()
        .filter(|&a| {
            arena.as_num(a).is_none()
                && arena.is_symbol_free(a)
                && !value_class::holds_real_infinity(arena, a)
        })
        .collect();
    constants.into_iter().all(|a| {
        let minus_one = arena.neg_one;
        let d = canon_add(arena, &[a, minus_one]);
        arena.as_num(d).is_some_and(|q| !q.is_zero())
            || arena
                .part_signs(d)
                .is_some_and(|s| s.re.is_some() || s.im.is_some())
    })
}

/// What a library function application is when one of its arguments is an
/// exact infinity in an axis direction ([`at_infinity`]).
pub(super) enum AtInfinity {
    /// Its value there (the limit along the direction), folded when the
    /// node is built.
    Value(Folded),
    /// No finite value is known: it is infinite in a direction the table
    /// does not build, has no limit (`Ai′(−∞)`, `γ(1, i·∞)`), or depends
    /// on a parameter without a value (`besseli(ν, −∞)`, `hermite(n, ∞)`).
    /// `0·f` and `0/f` are `nan` ([`library_value_unknown_at_infinity`]).
    NoFiniteValue,
    /// Not tabulated: no exact infinite argument, or a finite value the
    /// table does not build (`F(i·∞ | m) = i·K(1 − m)`) or may exist in a
    /// slot not analysed (the degree of an orthogonal polynomial, `B_x(a,
    /// b)`): `0·f` keeps folding to 0.
    Untabulated,
}

/// `iᵏ·∞` for `k` quarter turns.
fn directed(turns: u8) -> Folded {
    match turns % 4 {
        0 => Folded::Infinity,
        1 => Folded::ImaginaryInfinity(false),
        2 => Folded::NegInfinity,
        _ => Folded::ImaginaryInfinity(true),
    }
}

/// `n mod 4`, the quarter turns of `iⁿ`.
fn quarter_turns(n: &BigInt) -> u8 {
    use num_integer::Integer;
    n.mod_floor(&BigInt::from(4)).to_u8().unwrap_or(0)
}

/// The value at `iᵗ·∞` of a polynomial of degree `degree` whose leading
/// coefficient is negative (`negative`) or positive: `±i^{t·degree}·∞`,
/// and `±1` for degree 0 (every constant [`at_infinity`] meets is `±1`).
pub(super) fn polynomial_at_infinity(degree: &BigInt, negative: bool, t: u8) -> Folded {
    if degree.is_zero() {
        return Folded::int(if negative { -1 } else { 1 });
    }
    let turns = u32::from(t) * u32::from(quarter_turns(degree)) + if negative { 2 } else { 0 };
    directed((turns % 4) as u8)
}

/// Is `Γ(x) < 0` for a rational `x` that is not a pole?  For `x ∈ (−k, 1 −
/// k)` the sign is `(−1)ᵏ`.
pub(super) fn gamma_is_negative(x: &Q) -> bool {
    use num_integer::Integer;
    x.is_negative() && (-x).ceil().to_integer().is_odd()
}

/// Is the rising factorial `(a)ₙ = a(a + 1)⋯(a + n − 1)` negative, for a
/// rational `a` none of whose factors is 0?  An odd count `min(n, ⌈−a⌉)`
/// of negative factors.
fn rising_is_negative(a: &Q, n: &BigInt) -> bool {
    use num_integer::Integer;
    if !a.is_negative() {
        return false;
    }
    let count = (-a).ceil().to_integer().min(n.clone());
    count.is_odd()
}

/// The values at infinite arguments of the classical orthogonal
/// polynomials `Pν`, `Tν`, `Uν`, `Hν`, `Lν` of a rational degree `nu` (the
/// argument `iᵗ·∞`).  An integer degree with a polynomial: the sign of
/// the leading coefficient (`Pₙ`, `Tₙ`, `Uₙ`, `Hₙ` positive, `Lₙ` of sign
/// `(−1)ⁿ`) and the parity of the degree give the direction, the constant
/// is the value of degree 0; the polynomial identities `P_{−n−1} = Pₙ`,
/// `T_{−n} = Tₙ`, `U₋₁ = 0`, `U_{−n} = −U_{n−2}`.  Otherwise by the
/// asymptotic forms (DLMF 12.7, 14.8, 13.7, 18.5): `Hν(z) ~ (2z)^ν` off
/// the negative axis (`0` for `ν < 0`, `zoo` off the axes) and `Hν(−x) ~
/// √π·e^{x²}x^{−ν−1}/Γ(−ν)`; `Lν = ₁F₁(−ν; 1; ·)` (so `L_{−m}(x) =
/// eˣ L_{m−1}(−x)`) with `Lν(x) ~ eˣx^{−ν−1}/Γ(−ν)` and `Lν(−x) ~
/// x^ν/Γ(1 + ν)`; `P_{−1/2} → 0` and `Uν → 0` for `−2 < ν < 0`, every
/// other degree infinite.  mpmath at `|x| = 10³, 10⁶`: `hermite(−1/2,
/// 10⁶j)` = 7.07·10⁻⁴·e^{−iπ/4}, `hermite(1/2, −10⁶)` = −8.9·10^{434294481893},
/// `laguerre(−1/2, 0, −10⁶)` = 5.6·10⁻⁴.
fn classical_at_infinity(f: crate::base::libfn::LibFn, nu: &Q, t: u8) -> AtInfinity {
    use crate::base::libfn::LibFn;
    use num_integer::Integer;
    if nu.is_integer() {
        let n = nu.to_integer();
        let polynomial = match f {
            LibFn::Legendre if n.is_negative() => (-&n - 1, false),
            LibFn::ChebyshevT => (n.abs(), false),
            LibFn::ChebyshevU if n == BigInt::from(-1) => {
                return AtInfinity::Value(Folded::int(0));
            }
            LibFn::ChebyshevU if n.is_negative() => (-&n - 2, true),
            LibFn::Hermite if n.is_negative() => {
                return AtInfinity::Value(if t == 2 {
                    Folded::Infinity
                } else {
                    Folded::int(0)
                });
            }
            LibFn::Laguerre if n.is_negative() => {
                return match t {
                    0 => AtInfinity::Value(Folded::Infinity),
                    2 => AtInfinity::Value(Folded::int(0)),
                    _ => AtInfinity::NoFiniteValue,
                };
            }
            LibFn::Laguerre => (n.clone(), n.is_odd()),
            _ => (n, false),
        };
        return AtInfinity::Value(polynomial_at_infinity(&polynomial.0, polynomial.1, t));
    }
    let negative = nu.is_negative();
    let minus_half = Q::new(BigInt::from(-1), BigInt::from(2));
    let infinite = |negative_direction: bool| {
        AtInfinity::Value(if negative_direction {
            Folded::NegInfinity
        } else {
            Folded::Infinity
        })
    };
    match f {
        LibFn::Legendre if *nu == minus_half => AtInfinity::Value(Folded::int(0)),
        LibFn::ChebyshevU if negative && *nu > Q::from_integer(BigInt::from(-2)) => {
            AtInfinity::Value(Folded::int(0))
        }
        LibFn::Hermite => match t {
            2 => infinite(gamma_is_negative(&-nu)),
            _ if negative => AtInfinity::Value(Folded::int(0)),
            0 => infinite(false),
            _ => AtInfinity::Value(Folded::ComplexInfinity),
        },
        LibFn::Laguerre => match t {
            0 => infinite(gamma_is_negative(&-nu)),
            2 if negative => AtInfinity::Value(Folded::int(0)),
            2 => infinite(false),
            _ if !negative => AtInfinity::Value(Folded::ComplexInfinity),
            _ if *nu > -Q::one() => AtInfinity::Value(Folded::int(0)),
            _ => AtInfinity::NoFiniteValue,
        },
        _ => AtInfinity::NoFiniteValue,
    }
}

/// The coefficients (the constant first) of the Jacobi polynomial
/// `Pₙ^{(a,b)}(x) = Σ_ℓ (n + a + b + 1)_ℓ (a + ℓ + 1)_{n−ℓ}/(ℓ!(n − ℓ)!)
/// ((x − 1)/2)^ℓ` (DLMF 18.5.7), for a degree small enough to expand.
fn jacobi_coefficients(n: usize, a: &Q, b: &Q) -> Vec<Q> {
    let int = |k: usize| Q::from_integer(BigInt::from(k));
    let c = int(n) + a + b + Q::one();
    let mut coefficients = vec![Q::zero(); n + 1];
    let mut factorials = vec![Q::one(); n + 1];
    for k in 1..=n {
        factorials[k] = &factorials[k - 1] * int(k);
    }
    for l in 0..=n {
        let mut term = Q::one();
        for j in 0..l {
            term *= &c + int(j);
        }
        for j in 0..(n - l) {
            term *= a + int(l + 1 + j);
        }
        term /= &factorials[l] * &factorials[n - l];
        term /= Q::from_integer(BigInt::one() << l);
        // ((x − 1)/2)^ℓ = 2^{−ℓ}·Σⱼ C(ℓ, j)·(−1)^{ℓ−j}·xʲ.
        let mut binomial = Q::one();
        for (j, coefficient) in coefficients.iter_mut().enumerate().take(l + 1) {
            let signed = if (l - j) % 2 == 1 {
                -(&term * &binomial)
            } else {
                &term * &binomial
            };
            *coefficient += signed;
            binomial = binomial * int(l - j) / int(j + 1);
        }
    }
    coefficients
}

/// Largest degree whose Jacobi coefficients [`at_infinity`] expands when
/// the leading one vanishes.
const MAX_JACOBI_EXPANSION: usize = 64;

/// The values at `iᵗ·∞` of the parametrised orthogonal polynomials of a
/// non-negative integer degree `n` with rational parameters: the leading
/// term.  `Cₙ^{(α)}` has the leading coefficient `2ⁿ(α)ₙ/n!`, and for `α =
/// −m` with `0 ≤ m < n` the terms `x^{n−2k}` with `k ≥ n − m` only: 0 when
/// `2(n − m) > n` (`Cₙ^{(0)} = 0`), else the degree `2m − n` with the sign
/// `(−1)ⁿ` (`C₂^{(−1)} = 1`, `C₃^{(−2)} = −4x`).  `Pₙ^{(a,b)}` has the
/// leading coefficient `(n + a + b + 1)ₙ/(2ⁿn!)`; when it vanishes the
/// coefficients are expanded ([`jacobi_coefficients`]).  `Lₙ^{(α)}` has
/// `(−1)ⁿ/n!` for every `α`.  `None` when a vanishing leading
/// coefficient leaves a degree too large to expand.
fn parametrised_at_infinity(
    f: crate::base::libfn::LibFn,
    n: &BigInt,
    params: &[&Q],
    t: u8,
) -> Option<Folded> {
    use crate::base::libfn::LibFn;
    use num_integer::Integer;
    if n.is_zero() {
        return Some(Folded::int(1));
    }
    match (f, params) {
        (LibFn::Gegenbauer, [alpha]) => {
            let m = -*alpha;
            if m.is_integer() && !m.is_negative() && m.to_integer() < *n {
                let m = m.to_integer();
                if (n - &m) * 2 > *n {
                    return Some(Folded::int(0));
                }
                return Some(polynomial_at_infinity(&(&m * 2 - n), n.is_odd(), t));
            }
            Some(polynomial_at_infinity(n, rising_is_negative(alpha, n), t))
        }
        (LibFn::Jacobi, [a, b]) => {
            let c = Q::from_integer(n.clone()) + *a + *b + Q::one();
            let vanishes = c.is_integer() && !c.is_positive() && -c.to_integer() < *n;
            if !vanishes {
                return Some(polynomial_at_infinity(n, rising_is_negative(&c, n), t));
            }
            let degree = n.to_usize().filter(|&d| d <= MAX_JACOBI_EXPANSION)?;
            let coefficients = jacobi_coefficients(degree, a, b);
            Some(match coefficients.iter().rposition(|c| !c.is_zero()) {
                None => Folded::int(0),
                Some(0) => Folded::Rational(coefficients[0].clone()),
                Some(d) => {
                    polynomial_at_infinity(&BigInt::from(d), coefficients[d].is_negative(), t)
                }
            })
        }
        (LibFn::AssocLaguerre, _) => Some(polynomial_at_infinity(n, n.is_odd(), t)),
        _ => None,
    }
}

/// What the library function `f` is at `args` when one of them is an exact
/// infinity in an axis direction: the value the constructors fold, or
/// whether a finite value may exist ([`AtInfinity`]).  Every finite value
/// a tabulated pattern has is folded, so that what stays unevaluated there
/// is not taken for finite (`0·f` is `nan`; up to 0.53 `0·besselj(1,
/// i·∞)` and `0·hermite(1, ∞)` were 0, and `x·hermite(1, ∞)` had the
/// root 0).  Values by DLMF and mpmath at `|x| = 10³, 10⁶` (30 and 60
/// digits), SymPy 1.14 where it has them:
///
/// * orthogonal polynomials at `iᵗ·∞` ([`classical_at_infinity`],
///   [`parametrised_at_infinity`]): `H₁(∞) = ∞`, `L₃(∞) = −∞`, `P₃(i·∞) =
///   −i·∞`, `C₂^{(−1)}(∞) = 1` (the degree slot is not tabulated);
/// * Bessel functions at `±i·∞`: `Jν(±iy) = e^{±iνπ/2}Iν(y)`, `Yν(±iy) ~
///   e^{±i(ν+1)π/2}Iν(y)` (DLMF 10.27.6, 10.27.11: a quarter turn per unit
///   of an integer order, `zoo` for any other), `Iν(±i·∞) = Kν(±i·∞) =
///   0`; at an infinite order `J∞(x) = I∞(x) = 0` (`(x/2)^ν/Γ(ν + 1)`),
///   `K_{±i∞}(x) = 0` for `x > 0`;
/// * `Eν(±i·∞) = 0`, `E_∞(z) = E_{±i∞}(z) = 0` (`e^{−z}/(z + ν)`, DLMF
///   8.20); `Γ(s, ±i·∞) = 0` and `γ(s, ±i·∞) = Γ(s)` for `s < 1` (no
///   limit for `s ≥ 1`: `Γ(1, iy) = e^{−iy}`), `Γ(−∞, x) = 0` for `x ≥
///   1`, `γ(∞, x) = 0` for `|x| ≤ 1`, `Γ(±i∞, x) = γ(±i∞, x) = 0` for
///   `x > 0`;
/// * `Ai`, `Bi`, `Ai′`, `Bi′` at `±i·∞`: `zoo` (`|e^{∓ζ}|` grows while
///   its phase turns); `K(±i·∞) = 0`, `E(±i·∞) = zoo` (`√(∓i)·∞`, off
///   the axes); `Chi(±i·∞) = Shi(±i·∞) = ±iπ/2`, `erfi(±i·∞) = ±i`,
///   `S(±i·∞) = ∓i/2`, `C(±i·∞) = ±i/2` (SymPy agrees);
/// * `W_k(iᵗ·∞) = ∞ + iπ(2k + arg)`, `arg ∈ {0, 1/2, 1, −1/2}` (`W_k(z) ~
///   ln z + 2πik − ln(ln z + 2πik)`: `lambertw(−10⁶, 1)` = 14.2 + 8.8i,
///   tending to `3π`);
/// * `Li_s(iᵗ·∞) = −1` for `s = 0`, `0` for `s < 0`, `−∞` for `0 < s < 1`
///   and at `−∞`, `−ln(1 − z)` for `s = 1` (`−∞ − iπ` at `+∞`), `zoo`
///   otherwise; `Li_∞(z) = z` for `|z| ≤ 1`;
/// * `F(φ | iᵗ·∞) = 0` (SymPy: `elliptic_f(z, oo)` → 0), `Π(n | iᵗ·∞) =
///   Π(iᵗ·∞ | m) = 0` when the other argument is known not to be 1
///   ([`off_the_pole`]; SymPy agrees at `±∞`): up to 0.53
///   `euler_number(5)/elliptic_pi(−∞, π⁴)` was 0;
/// * `(x)ₙ` and `x^{(n)}` at `x = iᵗ·∞`: `i^{tn}·∞` for an integer `n >
///   0`, 1 for `n = 0`, 0 for `n < 0` (SymPy: `rf(oo, −1)` → oo, wrong),
///   `x^ν` otherwise; at an infinite order `(x)_{±i∞} = 0`, `(x)_∞ =
///   Γ(x)⁻¹·∞`, `(−m)_{±∞} = 0`, `x^{(−∞)} = 0`;
/// * the integer sequences at `+∞` (`F_∞ = L_∞ = H_∞ = C_∞ = B_∞ = p(∞)
///   = ∞!! = !∞ = ∞`), `H_{±i∞} = ∞ ± iπ/2`, `C_{±i∞} = 0`, `B_{−∞} =
///   ∞` (`−s·ζ(1 − s)`);
/// * no finite value: `erfinv`, `erfcinv` (defined on `(−1, 1)`), `η` off
///   `+∞`, `F(±∞ | m)`, Bernoulli numbers at `+∞`.
///
/// The values at `±∞` of the functions folded since 0.51–0.53 (Bessel
/// functions, `Eν`, `Γ(s, x)`, `γ(s, x)`, Airy functions, `K`, `E`, `Chi`,
/// `Shi`, `erfi`, Fresnel integrals, `η`) are built by
/// [`lib_function_value`]; what it leaves there is [`AtInfinity::NoFiniteValue`].
pub(super) fn at_infinity(
    arena: &Arena,
    f: crate::base::libfn::LibFn,
    args: &[ExprId],
) -> AtInfinity {
    use crate::base::libfn::LibFn;
    use AtInfinity::{NoFiniteValue, Untabulated, Value};
    // (a cheap filter: `±∞`, or a product with `∞` that may be `±i·∞`)
    let candidate = |a: ExprId| {
        ValueClass::special_atom(arena, a).is_some_and(|c| c.real_infinity().is_some())
            || is_directed_infinity(arena, a)
    };
    if !args.iter().any(|&a| candidate(a)) {
        return Untabulated;
    }
    let slots: SmallVec<[Slot; 4]> = args.iter().map(|&a| slot(arena, a)).collect();
    if !slots.iter().any(|s| matches!(s, Slot::Infinite(_))) || slots.contains(&Slot::Other) {
        return Untabulated;
    }
    let num = |i: usize| args.get(i).and_then(|&a| arena.as_num(a));
    let s = |i: usize| slots.get(i).copied().unwrap_or(Slot::Other);
    let int = |i: i64| Q::from_integer(BigInt::from(i));
    let fin = Slot::Finite;
    let value = Value;
    match f {
        LibFn::Legendre
        | LibFn::ChebyshevT
        | LibFn::ChebyshevU
        | LibFn::Hermite
        | LibFn::Laguerre => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(t)) => match num(0) {
                Some(nu) => classical_at_infinity(f, nu, t),
                None => NoFiniteValue,
            },
            _ => Untabulated,
        },
        LibFn::Gegenbauer | LibFn::Jacobi | LibFn::AssocLaguerre => {
            let last = args.len() - 1;
            let Slot::Infinite(t) = s(last) else {
                return Untabulated;
            };
            if slots[..last].iter().any(|&p| p != fin) {
                return Untabulated;
            }
            let Some(n) = num(0) else {
                return NoFiniteValue;
            };
            if !n.is_integer() || n.is_negative() {
                return Untabulated;
            }
            if n.is_zero() {
                return value(Folded::int(1));
            }
            let params: SmallVec<[&Q; 2]> = match f {
                LibFn::AssocLaguerre => SmallVec::new(),
                _ => match (1..last).map(num).collect::<Option<SmallVec<[&Q; 2]>>>() {
                    Some(p) => p,
                    None => return NoFiniteValue,
                },
            };
            match parametrised_at_infinity(f, &n.to_integer(), &params, t) {
                Some(v) => value(v),
                None => Untabulated,
            }
        }
        LibFn::BesselJ | LibFn::BesselY | LibFn::BesselI | LibFn::BesselK => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(t)) => bessel_at_infinite_argument(f, num(0), t),
            (Slot::Infinite(0), Slot::Finite) => match f {
                LibFn::BesselJ | LibFn::BesselI => value(Folded::int(0)),
                _ => NoFiniteValue,
            },
            (Slot::Infinite(2), Slot::Finite) => NoFiniteValue,
            (Slot::Infinite(_), Slot::Finite) => match num(1) {
                Some(x) if x.is_positive() => match f {
                    LibFn::BesselK => value(Folded::int(0)),
                    _ => NoFiniteValue,
                },
                _ => Untabulated,
            },
            _ => NoFiniteValue,
        },
        LibFn::ExpInt => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(2)) => match num(0) {
                Some(nu) => value(expint_at_minus_infinity(nu)),
                None => NoFiniteValue,
            },
            (Slot::Finite, Slot::Infinite(_)) => value(Folded::int(0)),
            (Slot::Infinite(2), Slot::Finite) => NoFiniteValue,
            (Slot::Infinite(_), Slot::Finite) => value(Folded::int(0)),
            _ => NoFiniteValue,
        },
        LibFn::UpperGamma => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(0)) => value(Folded::int(0)),
            (Slot::Finite, Slot::Infinite(2)) => match num(0) {
                Some(s) => value(upper_gamma_at_minus_infinity(s)),
                None => NoFiniteValue,
            },
            (Slot::Finite, Slot::Infinite(_)) => match num(0) {
                Some(s) if *s < Q::one() => value(Folded::int(0)),
                _ => NoFiniteValue,
            },
            (Slot::Infinite(2), Slot::Finite) => match num(1) {
                Some(x) if *x >= Q::one() => value(Folded::int(0)),
                _ => NoFiniteValue,
            },
            (Slot::Infinite(t), Slot::Finite) if t % 2 == 1 => match num(1) {
                Some(x) if !x.is_negative() => value(Folded::int(0)),
                Some(_) => Untabulated,
                None => NoFiniteValue,
            },
            _ => NoFiniteValue,
        },
        LibFn::LowerGamma => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(0)) => value(Folded::Gamma(args[0])),
            (Slot::Finite, Slot::Infinite(2)) => match num(0) {
                Some(s) if s.is_positive() => value(lower_gamma_at_minus_infinity(s)),
                _ => NoFiniteValue,
            },
            (Slot::Finite, Slot::Infinite(_)) => match num(0) {
                Some(s) if is_nonpositive_int(s) => value(Folded::ComplexInfinity),
                Some(s) if *s < Q::one() => value(Folded::Gamma(args[0])),
                _ => NoFiniteValue,
            },
            (Slot::Infinite(0), Slot::Finite) => match num(1) {
                Some(x) if x.abs() <= Q::one() => value(Folded::int(0)),
                _ => NoFiniteValue,
            },
            (Slot::Infinite(t), Slot::Finite) if t % 2 == 1 => match num(1) {
                Some(x) if x.is_positive() => value(Folded::int(0)),
                Some(x) if x.is_negative() => Untabulated,
                _ => NoFiniteValue,
            },
            _ => NoFiniteValue,
        },
        LibFn::AiryAi | LibFn::AiryBi | LibFn::AiryAiPrime | LibFn::AiryBiPrime => match s(0) {
            Slot::Infinite(0) => match f {
                LibFn::AiryAi | LibFn::AiryAiPrime => value(Folded::int(0)),
                _ => value(Folded::Infinity),
            },
            Slot::Infinite(2) => match f {
                LibFn::AiryAi | LibFn::AiryBi => value(Folded::int(0)),
                _ => NoFiniteValue,
            },
            _ => value(Folded::ComplexInfinity),
        },
        LibFn::EllipticK => value(Folded::int(0)),
        LibFn::EllipticE => value(match s(0) {
            Slot::Infinite(2) => Folded::Infinity,
            _ => Folded::ComplexInfinity,
        }),
        LibFn::Chi | LibFn::Shi | LibFn::Erfi | LibFn::FresnelS | LibFn::FresnelC => {
            let Slot::Infinite(t) = s(0) else {
                return Untabulated;
            };
            let half = Q::new(BigInt::one(), BigInt::from(2));
            let up = t == 1;
            let sign = |q: Q| if up { q } else { -q };
            value(match (f, t) {
                (LibFn::FresnelS | LibFn::FresnelC, 0) => Folded::Rational(half),
                (LibFn::FresnelS | LibFn::FresnelC, 2) => Folded::Rational(-half),
                (LibFn::FresnelS, _) => Folded::Imaginary(sign(-half)),
                (LibFn::FresnelC, _) => Folded::Imaginary(sign(half)),
                (_, 0) => Folded::Infinity,
                (LibFn::Chi, 2) => Folded::InfinitePlus(InfDir::Pos, Q::one(), Q::zero()),
                (_, 2) => Folded::NegInfinity,
                (LibFn::Erfi, _) => Folded::Imaginary(sign(Q::one())),
                _ => Folded::PiImaginary(sign(half)),
            })
        }
        LibFn::DirichletEta => match s(0) {
            Slot::Infinite(0) => value(Folded::int(1)),
            _ => NoFiniteValue,
        },
        LibFn::ErfInv | LibFn::ErfcInv => NoFiniteValue,
        LibFn::LambertW => {
            let Slot::Infinite(t) = s(0) else {
                return NoFiniteValue;
            };
            let k = match args.get(1) {
                None => Q::zero(),
                Some(_) => match num(1) {
                    Some(k) if k.is_integer() && s(1) == fin => k.clone(),
                    _ => return NoFiniteValue,
                },
            };
            let turn = Q::new(BigInt::from(i64::from(t)), BigInt::from(2));
            let arg = if t == 3 { turn - int(2) } else { turn };
            value(Folded::InfinitePlus(
                InfDir::Pos,
                k * int(2) + arg,
                Q::zero(),
            ))
        }
        LibFn::PolyLog => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(t)) => match num(0) {
                Some(s) if s.is_zero() => value(Folded::int(-1)),
                Some(s) if s.is_negative() => value(Folded::int(0)),
                Some(_) if t == 2 => value(Folded::NegInfinity),
                Some(s) if s.is_one() => {
                    let q = match t {
                        0 => -Q::one(),
                        1 => Q::new(BigInt::one(), BigInt::from(2)),
                        _ => Q::new(BigInt::from(-1), BigInt::from(2)),
                    };
                    value(Folded::InfinitePlus(InfDir::Neg, q, Q::zero()))
                }
                Some(s) if *s < Q::one() => value(Folded::NegInfinity),
                Some(_) => value(Folded::ComplexInfinity),
                None => NoFiniteValue,
            },
            (Slot::Infinite(0), Slot::Finite) => match num(1) {
                Some(z) if z.abs() <= Q::one() => value(Folded::Rational(z.clone())),
                Some(_) => NoFiniteValue,
                None => Untabulated,
            },
            _ => NoFiniteValue,
        },
        LibFn::EllipticF => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(_)) => value(Folded::int(0)),
            (Slot::Infinite(t), Slot::Finite) if t % 2 == 1 => Untabulated,
            _ => NoFiniteValue,
        },
        LibFn::EllipticPi => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(_)) if args[0] != arena.one => value(Folded::int(0)),
            (Slot::Infinite(_), Slot::Finite) if args[1] != arena.one => value(Folded::int(0)),
            _ => NoFiniteValue,
        },
        LibFn::RisingFactorial | LibFn::FallingFactorial => {
            let rising = f == LibFn::RisingFactorial;
            match (s(0), s(1)) {
                (Slot::Infinite(t), Slot::Finite) => match num(1) {
                    Some(n) if n.is_integer() && n.is_positive() => {
                        value(polynomial_at_infinity(&n.to_integer(), false, t))
                    }
                    Some(n) if n.is_zero() => value(Folded::int(1)),
                    Some(n) if n.is_integer() => value(Folded::int(0)),
                    Some(n) => match t {
                        2 => NoFiniteValue,
                        _ if n.is_negative() => value(Folded::int(0)),
                        0 => value(Folded::Infinity),
                        _ => value(Folded::ComplexInfinity),
                    },
                    None => NoFiniteValue,
                },
                (Slot::Finite, Slot::Infinite(t)) => factorial_order_at_infinity(rising, num(0), t),
                _ => NoFiniteValue,
            }
        }
        LibFn::Fibonacci
        | LibFn::Lucas
        | LibFn::Harmonic
        | LibFn::Catalan
        | LibFn::Bell
        | LibFn::PartitionCount
        | LibFn::Factorial2
        | LibFn::Subfactorial
        | LibFn::Bernoulli => match (f, s(0)) {
            (LibFn::Bernoulli, Slot::Infinite(0)) => NoFiniteValue,
            (LibFn::Bernoulli, Slot::Infinite(2)) | (_, Slot::Infinite(0)) => {
                value(Folded::Infinity)
            }
            (LibFn::PartitionCount | LibFn::Subfactorial | LibFn::Bernoulli, _) => Untabulated,
            (_, Slot::Infinite(2)) => NoFiniteValue,
            (LibFn::Catalan, _) => value(Folded::int(0)),
            (LibFn::Harmonic, Slot::Infinite(t)) => value(Folded::InfinitePlus(
                InfDir::Pos,
                Q::new(BigInt::from(if t == 1 { 1 } else { -1 }), BigInt::from(2)),
                Q::zero(),
            )),
            _ => Untabulated,
        },
        LibFn::AssocLegendre => match (s(0), s(1), s(2), num(0), num(1)) {
            // (`m < −n`: a Ferrers function, not a polynomial; not tabulated.)
            (Slot::Finite, Slot::Finite, Slot::Infinite(t), Some(n), Some(m))
                if n.is_integer() && !n.is_negative() && m.is_integer() && *m >= -n =>
            {
                value(assoc_legendre_at_infinity(
                    &n.to_integer(),
                    &m.to_integer(),
                    t,
                ))
            }
            _ => Untabulated,
        },
        LibFn::EulerNumber
        | LibFn::Stirling1
        | LibFn::Stirling2
        | LibFn::BetaInc
        | LibFn::BetaIncRegularized => Untabulated,
    }
}

/// `Pₙᵐ(iᵗ·∞)` for integers `n ≥ 0` and `m`, by the definition `eval`
/// expands (Condon–Shortley phase, the principal `(1 − x²)^{m/2}`):
/// `Pₙᵏ(x) = (−1)ᵏ(2n)!/(2ⁿn!(n − k)!)·x^{n−k}(1 − x²)^{k/2} + …` for `k
/// ≥ 0` and `Pₙ^{−k} = (−1)ᵏ(n − k)!/(n + k)!·Pₙᵏ`, where `1 − x² → −∞`
/// on the real axis (`√(−∞) = i·∞`) and `→ +∞` on the imaginary one;
/// 0 for `|m| > n`.  `P₂¹(∞) = −i·∞`, `P₂²(∞) = −∞`, as `eval`'s
/// closed forms gave them before they stopped being taken at infinite
/// arguments.
fn assoc_legendre_at_infinity(n: &BigInt, m: &BigInt, t: u8) -> Folded {
    use num_integer::Integer;
    let k = m.abs();
    if k > *n {
        return Folded::int(0);
    }
    if n.is_zero() {
        return Folded::int(1);
    }
    let negative = k.is_odd() && !m.is_negative();
    let x_turns = u32::from(t) * u32::from(quarter_turns(&(n - &k)));
    let root_turns = if t.is_multiple_of(2) {
        u32::from(quarter_turns(&k))
    } else {
        0
    };
    let turns = x_turns + root_turns + if negative { 2 } else { 0 };
    directed((turns % 4) as u8)
}

/// The Bessel functions at `x = iᵗ·∞` for a finite order `nu` (a rational
/// when `Some`): see [`at_infinity`].  At `±∞` the values are those of
/// [`lib_function_value`] (`Jν(±∞) = Yν(±∞) = Kν(∞) = 0`, `Iν(∞) = ∞`,
/// `Iν(−∞) = e^{iπν}·∞`, `Kν(−∞) = −i·∞` for a rational order).
fn bessel_at_infinite_argument(f: crate::base::libfn::LibFn, nu: Option<&Q>, t: u8) -> AtInfinity {
    use crate::base::libfn::LibFn;
    let value = AtInfinity::Value;
    match (f, t) {
        (LibFn::BesselJ | LibFn::BesselY, 0 | 2) | (LibFn::BesselK, 0) => value(Folded::int(0)),
        (LibFn::BesselI | LibFn::BesselK, 1 | 3) => value(Folded::int(0)),
        (LibFn::BesselI, 0) => match nu {
            Some(_) => value(Folded::Infinity),
            None => AtInfinity::NoFiniteValue,
        },
        (LibFn::BesselI, _) => match nu {
            Some(nu) => value(bessel_i_at_minus_infinity(nu)),
            None => AtInfinity::NoFiniteValue,
        },
        (LibFn::BesselK, _) => match nu {
            Some(_) => value(Folded::ImaginaryInfinity(true)),
            None => AtInfinity::NoFiniteValue,
        },
        // J at ±i·∞: e^{±iνπ/2}·∞; Y: e^{±i(ν+1)π/2}·∞.
        _ => match nu {
            Some(nu) if nu.is_integer() => {
                let shift = if f == LibFn::BesselY { 1 } else { 0 };
                let turns = quarter_turns(&(nu.to_integer() + shift));
                value(directed(if t == 1 { turns } else { (4 - turns) % 4 }))
            }
            Some(_) => value(Folded::ComplexInfinity),
            None => AtInfinity::NoFiniteValue,
        },
    }
}

/// The rising (`rising`) or falling factorial of a finite `x` (a rational
/// when `Some`) at the order `iᵗ·∞`: see [`at_infinity`].  `(x)ν =
/// Γ(x + ν)/Γ(x)` grows as `Γ(x)⁻¹·∞` for `ν → ∞`, is 0 for `x` a pole
/// of `Γ` (`(−m)_ν = 0` off the poles of `Γ(ν − m)`) and tends to 0 along
/// the imaginary axis (`|Γ(x + iy)| ~ e^{−π|y|/2}`); towards `−∞` it meets
/// the poles of `Γ(x + ν)`.  `x^{(ν)} = Γ(x + 1)/Γ(x + 1 − ν)` tends to 0
/// for `ν → −∞` (`x` not a negative integer) and has no limit otherwise
/// (`1/Γ` grows between its zeros; mpmath `ff(2, 10⁶)` = 0 at the integers
/// only).
fn factorial_order_at_infinity(rising: bool, x: Option<&Q>, t: u8) -> AtInfinity {
    let value = AtInfinity::Value;
    let pole = |x: &Q| x.is_integer() && !x.is_positive();
    match (rising, t, x) {
        (true, 1 | 3, _) => value(Folded::int(0)),
        (true, 0 | 2, Some(x)) if pole(x) => value(Folded::int(0)),
        (true, 0, Some(x)) => value(if gamma_is_negative(x) {
            Folded::NegInfinity
        } else {
            Folded::Infinity
        }),
        (false, 2, Some(x)) if !(x.is_integer() && x.to_integer().is_negative()) => {
            value(Folded::int(0))
        }
        _ => AtInfinity::NoFiniteValue,
    }
}
