//! Special value tables (part of [`eval`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Special value tables
// ═══════════════════════════════════════════════════════════════════════════

// ── The digit guard of exact products ────────────────────────────────────
//
// `n!`, `Γ(p/q)` (a rising factorial times `Γ` of the fractional part),
// `C(n, k)`, `(x)ₙ` and `x^(n)` are exact rationals of about `n·log₁₀ n`
// digits.  They follow canon's policy for integer powers: a result beyond
// `max_result_digits` digits is not computed and the node stays symbolic,
// for `evalf` to evaluate numerically.  A lower bound on the digits refuses
// before any multiplication; the exact count of the product decides at the
// limit.  Before 0.29 nothing was bounded and every factor went through a
// `Ratio` multiplication (a gcd per step): `Γ(1000 + 1/3)` took 2.9 s in
// eval (debug), `Γ(3000 + 1/3)`, `3000!` and `ln Γ(3000)` more than a minute,
// so `uppergamma(s, x)/gamma(s)` hung before `evalf` ran.

/// A lower bound on `log₁₀ n!` (`n! ≥ (n/e)ⁿ`), for `n ≥ 0`.
pub(super) fn log10_factorial_lower(n: f64) -> f64 {
    if n < 1.0 {
        0.0
    } else {
        n * (n / std::f64::consts::E).log10().max(0.0)
    }
}

/// Is an exact result of at least `digits` decimal digits (a lower bound)
/// beyond the digit guard?  `NaN` and `∞` are.
pub(super) fn beyond_digit_guard(arena: &Arena, digits: f64) -> bool {
    digits.is_nan() || digits > arena.config.max_result_digits as f64
}

/// `r` as a node when its numerator and denominator together have at most
/// `max_result_digits` digits (canon's count), otherwise `None`.
pub(super) fn guarded_num(arena: &mut Arena, r: Q) -> Option<ExprId> {
    // A b-bit integer has ⌊b·log₁₀2⌋ or one more digits: the bit counts
    // decide unless the total is within two digits of the limit, where the
    // decimal strings are counted (they cost more than the arithmetic for
    // the thousand coefficients of a degree-1000 polynomial).
    let limit = arena.config.max_result_digits as f64;
    let est = (r.numer().bits() + r.denom().bits()) as f64 * std::f64::consts::LOG10_2;
    if est > limit + 2.0 {
        return None;
    }
    if est >= limit - 2.0 {
        let digits = r.numer().to_string().len() + r.denom().to_string().len();
        if digits > arena.config.max_result_digits {
            return None;
        }
    }
    let nid = arena.intern_num(r);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// `∏_{i<n} (a + i·d)` for integers `a`, `d`.
fn progression_product(a: &BigInt, d: &BigInt, n: u64) -> BigInt {
    let mut acc = BigInt::one();
    let mut term = a.clone();
    for _ in 0..n {
        acc *= &term;
        term += d;
    }
    acc
}

/// A lower bound on the decimal digits of `∏_{i<n} |a + i·d|` (`d ≠ 0`, no
/// factor zero): the factors are distinct non-zero integers `|d|` apart, so
/// the `j`-th smallest of them is at least `|d|·⌊j/2⌋`, and when they all
/// have the sign of `a` (`a·d > 0`) the `j`-th is at least `|d|·j` (`j ≥ 1`).
fn progression_digits_lower(a: &BigInt, d: &BigInt, n: u64) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let n = n as f64;
    let log_d = d.abs().to_f64().unwrap_or(f64::INFINITY).log10();
    if a.sign() == d.sign() {
        log10_factorial_lower(n - 1.0) + (n - 1.0) * log_d
    } else {
        2.0 * log10_factorial_lower(((n - 1.0) / 2.0).floor()) + (n - 2.0).max(0.0) * log_d
    }
}

/// `(p/q)ₙ = ∏_{i<n} (p/q + i) = ∏(p + i·q)/qⁿ` (`step = 1`) or the falling
/// `∏_{i<n} (p/q − i)` (`step = −1`) exactly, `None` beyond the digit guard.
/// Numerator and denominator are coprime (`gcd(p + i·q, q) = gcd(p, q) = 1`),
/// so no gcd is taken.
pub(super) fn exact_factorial_power(arena: &mut Arena, x: &Q, n: u64, step: i32) -> Option<ExprId> {
    let (p, q) = (x.numer(), x.denom());
    let d = q * BigInt::from(step);
    // A zero factor: x a non-positive (rising) or non-negative (falling)
    // integer within n steps.
    if x.is_integer() {
        let reach = BigInt::from(n) - BigInt::one();
        let hits_zero = if step > 0 {
            !p.is_positive() && -p <= reach
        } else {
            !p.is_negative() && *p <= reach
        };
        if hits_zero {
            return Some(arena.zero);
        }
    }
    let q_digits = (n as f64) * q.to_f64().unwrap_or(f64::INFINITY).log10();
    if beyond_digit_guard(arena, progression_digits_lower(p, &d, n) + q_digits) {
        return None;
    }
    let numer = progression_product(p, &d, n);
    let denom = num_traits::Pow::pow(q.clone(), n);
    guarded_num(arena, Ratio::new_raw(numer, denom))
}

/// `n!` for a non-negative integer literal `n`, within the digit guard.
pub(super) fn eval_factorial(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
    if beyond_digit_guard(arena, log10_factorial_lower(n as f64)) {
        return None;
    }
    let result = Ratio::from_integer(crate::base::combinatorics::factorial(n));
    guarded_num(arena, result)
}

/// Gamma(n) for positive integer n → (n-1)!
/// Gamma(1/2) → √π
///
/// Every expansion is bounded by the digit guard (see
/// [`beyond_digit_guard`]); beyond it the node stays symbolic.
pub(super) fn eval_gamma(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?.clone();

    // Half-integer: Gamma(p/2) for positive odd p.
    // Uses: Gamma((2k+1)/2) = (2k-1)!! / 2^k · √π
    // where (2k-1)!! = 1·3·5·…·(2k-1) (empty product = 1 when k=0).
    if *r.denom() == BigInt::from(2) && r.is_positive() {
        // p is guaranteed odd because the fraction is in lowest terms with denom 2.
        let p = r.numer().clone();
        // k = (p - 1) / 2
        let k_big = (&p - BigInt::from(1)) / BigInt::from(2);
        let k: u64 = k_big.try_into().ok()?;
        // (2k−1)!! ≥ k!, and the denominator 2^k.
        let kf = k as f64;
        if beyond_digit_guard(
            arena,
            log10_factorial_lower(kf) + kf * std::f64::consts::LOG10_2,
        ) {
            return None;
        }

        // Compute (2k-1)!! = product of odd numbers 1, 3, 5, …, 2k-1.
        let mut double_fact = BigInt::from(1);
        for i in 0..k {
            double_fact *= BigInt::from(2 * i + 1);
        }

        // coefficient = (2k-1)!! / 2^k
        let two_pow_k = BigInt::from(1) << (k as usize);
        let coeff = Ratio::new(double_fact, two_pow_k);

        if coeff.is_one() {
            let pi = arena.pi;
            return Some(arena.sqrt(pi));
        }

        let coeff_node = guarded_num(arena, coeff)?;
        let pi = arena.pi;
        let sqrt_pi = arena.sqrt(pi);
        let result = arena.mul(&[coeff_node, sqrt_pi]);
        return Some(result);
    }

    // Positive integer: Gamma(n) = (n-1)!
    if r.is_integer() && r.is_positive() {
        let n: u64 = r.to_integer().try_into().ok()?;
        if beyond_digit_guard(arena, log10_factorial_lower((n - 1) as f64)) {
            return None;
        }
        let result = Ratio::from_integer(crate::base::combinatorics::factorial(n - 1));
        return guarded_num(arena, result);
    }

    // Gamma recurrence for positive rationals > 1 with denom ∉ {1, 2}:
    // Gamma(frac + n) = rising_factorial(frac, n) * Gamma(frac)
    // where frac ∈ (0,1) and n ≥ 1.
    if r.is_positive() && !r.is_integer() && *r.denom() != BigInt::from(2) {
        let n_big = r.to_integer(); // floor for positive values
        if n_big >= BigInt::from(1) {
            let n: u64 = n_big.clone().try_into().ok()?;
            let frac = &r - Ratio::from_integer(n_big);
            // rising_factorial(frac, n) = frac * (frac+1) * … * (frac+n-1)
            let prod_node = exact_factorial_power(arena, &frac, n, 1)?;
            let frac_id = {
                let nid = arena.intern_num(frac);
                arena.intern(ExprNode::Num(nid))
            };
            // Try to evaluate the inner Gamma (e.g. Gamma(1/2) → √π)
            let gamma_frac = eval_gamma(arena, frac_id).unwrap_or_else(|| arena.gamma(frac_id));
            if prod_node == arena.one {
                return Some(gamma_frac);
            }
            return Some(arena.mul(&[prod_node, gamma_frac]));
        }
    }

    // Gamma recurrence for negative non-integer rationals:
    // Gamma(x) = Gamma(x+m) / (x·(x+1)·…·(x+m-1))
    // where m = ⌈-x⌉ shifts x into (0, 1).
    if r.is_negative() && !r.is_integer() {
        let neg_r = -&r; // positive
        // m = ceil(neg_r); since neg_r is positive and not an integer,
        // ceil = floor + 1 = to_integer + 1.
        let m_big = if neg_r.is_integer() {
            neg_r.to_integer()
        } else {
            neg_r.to_integer() + BigInt::from(1)
        };
        let m: u64 = m_big.clone().try_into().ok()?;
        let frac = &r + Ratio::from_integer(m_big); // frac ∈ (0, 1)
        // Denominator product: x·(x+1)·…·(x+m-1), non-zero as r is not an
        // integer.
        let denom_node = exact_factorial_power(arena, &r, m, 1)?;
        let denom_product = arena.as_num(denom_node)?.clone();
        if denom_product.is_zero() {
            return None;
        }
        let frac_id = {
            let nid = arena.intern_num(frac);
            arena.intern(ExprNode::Num(nid))
        };
        // Try to evaluate the inner Gamma (handles half-integer base, etc.)
        let gamma_frac = eval_gamma(arena, frac_id).unwrap_or_else(|| arena.gamma(frac_id));
        // Gamma(r) = Gamma(frac) / denom_product = (1/denom_product) * Gamma(frac)
        let inv_denom = Ratio::one() / denom_product;
        let coeff_nid = arena.intern_num(inv_denom);
        let coeff_node = arena.intern(ExprNode::Num(coeff_nid));
        return Some(arena.mul(&[coeff_node, gamma_frac]));
    }

    None
}

/// Is the evaluated condition `c` of a `Piecewise` an order relation with a
/// `nan` operand (or a conjunction of such)?  `nan` has no order, the
/// relation stays and can never become true: the `Piecewise` has no value
/// at such a point.  Up to 0.58 it stayed with the dead condition,
/// `Piecewise((0, nan > 0), (nan, True))`: the constructors write "`d` is
/// nonzero" as `|d| > 0` (`canon::push_nonzero_conditions`), which is
/// `nan > 0` exactly where `d` has no value.  (An equality with `nan` is
/// decided: [`relation_truth`].)
pub(super) fn condition_is_undefined(arena: &Arena, c: ExprId) -> bool {
    let nan = arena.nan;
    let order_with_nan = |id: ExprId| matches!(*arena.node(id), ExprNode::Gt(a, b) | ExprNode::Ge(a, b) if a == nan || b == nan);
    match arena.node(c) {
        ExprNode::And(children) => children.iter().all(|&k| order_with_nan(k)),
        _ => order_with_nan(c),
    }
}

/// The four relational node kinds (`a < b` is `Gt(b, a)`, `a ≤ b` is
/// `Ge(b, a)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RelationKind {
    Gt,
    Ge,
    Eq,
    Ne,
}

impl RelationKind {
    /// The kind of a relational node; `Ne` for anything else (callers match
    /// the four relational nodes first).
    pub(super) fn of(node: &ExprNode) -> Self {
        match node {
            ExprNode::Gt(..) => Self::Gt,
            ExprNode::Ge(..) => Self::Ge,
            ExprNode::Eq_(..) => Self::Eq,
            _ => Self::Ne,
        }
    }
}

/// The truth of the relation `kind` between the evaluated operands `a`,
/// `b`, as `eval` decides it: two rationals compare; an order against 0 by
/// the assumptions ([`order_against_zero`]); an order or an equality by the
/// difference `a − b` (a rational, a certified sign, the square-root tower
/// zero test, the minimal polynomial: [`order_by_difference`],
/// [`equal_by_difference`]).  `None` when undecided.  `orig` are the
/// operands before evaluation, whose difference the exact zero tests also
/// try ([`Difference`]).
pub(super) fn relation_truth(
    arena: &mut Arena,
    kind: RelationKind,
    a: ExprId,
    b: ExprId,
    orig: (ExprId, ExprId),
) -> Option<bool> {
    // `nan` equals nothing, itself included (SymPy 1.14: `Eq(nan, -1)` and
    // `Eq(nan, nan)` are `False`), as Boolean `simplify` decides it
    // (`logic::fold_rel`); it has no order, which stays.  Up to 0.46 `eval`
    // left `nan == -1` while `simplify` gave `False`.
    let nan = arena.nan;
    if (a == nan || b == nan) && matches!(kind, RelationKind::Eq | RelationKind::Ne) {
        return Some(kind == RelationKind::Ne);
    }
    let d = Difference { a, b, orig };
    match kind {
        RelationKind::Gt | RelationKind::Ge => {
            let strict = kind == RelationKind::Gt;
            if let (Some(ra), Some(rb)) = (arena.as_num(a), arena.as_num(b)) {
                return Some(if strict { ra > rb } else { ra >= rb });
            }
            order_against_zero(arena, a, b, strict)
                .or_else(|| order_by_difference(arena, d, strict))
        }
        RelationKind::Eq | RelationKind::Ne => {
            let equal = gaussian_equal(arena, a, b).or_else(|| equal_by_difference(arena, d))?;
            Some(equal == (kind == RelationKind::Eq))
        }
    }
}

/// The operands `a`, `b` of a relation after evaluation, and `orig` before:
/// the exact zero tests try `a − b`, then the difference of the original
/// operands, where the trigonometric values are not yet radicals — the
/// cyclotomic test decides `cos(13π/16)·tan(23π/12)·cos(23π/12) =
/// −√(…)·sin(23π/12)` on the original form, not once `tan(23π/12)` is `√3 −
/// 2` and the cosine `(√6 + √2)/4` (radicals it takes as independent
/// generators).
#[derive(Clone, Copy)]
pub(super) struct Difference {
    pub(super) a: ExprId,
    pub(super) b: ExprId,
    pub(super) orig: (ExprId, ExprId),
}

impl Difference {
    /// The difference of the original operands, when they differ from the
    /// evaluated ones.
    fn original(self, arena: &mut Arena) -> Option<ExprId> {
        (self.orig != (self.a, self.b)).then(|| arena.sub(self.orig.0, self.orig.1))
    }
}

/// The truth of the relational node `rel` (`Gt`, `Ge`, `Eq_`, `Ne`) as
/// [`eval`] decides it, after evaluating its two operands; `None` when
/// `eval` leaves it a relation (or `rel` is not one).  For Boolean
/// `simplify`, which must decide every relation `eval` decides
/// (`Eq(cos(π/8), √(2 + √2)/2)`: `eval` gave `True` by the square-root
/// tower, `simplify` kept the relation up to 0.44).
pub(crate) fn decide_relation(arena: &mut Arena, rel: ExprId) -> Option<bool> {
    let (a, b) = match *arena.node(rel) {
        ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
            (a, b)
        }
        _ => return None,
    };
    let kind = RelationKind::of(arena.node(rel));
    let ea = eval(arena, a);
    let eb = eval(arena, b);
    relation_truth(arena, kind, ea, eb, (a, b))
}

/// `a > 0`, `0 > b` (`strict`) or `a ≥ 0`, `0 ≥ b` for a constant (free of
/// symbols) `a` or `b` whose sign the assumption engine knows from its
/// structure — `exp` of a real is positive, a square of a real is
/// non-negative, … — decided before any numerical evaluation.  `None` when
/// neither side is 0 or the sign is not known.
///
/// Before 0.29 only two rationals were compared: `Piecewise((1, exp(−4·10⁹)
/// > 0), (0, True))` was left to `evalf`, whose `exp(−4·10⁹)` underflows, and
/// was refused (`PrecisionExhausted`).
fn order_against_zero(arena: &Arena, a: ExprId, b: ExprId, strict: bool) -> Option<bool> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let is_zero = |id: ExprId| arena.as_num(id).is_some_and(Zero::is_zero);
    // `x > 0` / `x ≥ 0` with the sign of `x`; `0 > x` is `−x > 0`.
    let (x, flipped) = if is_zero(b) {
        (a, false)
    } else if is_zero(a) {
        (b, true)
    } else {
        return None;
    };
    if !walk::free_symbols(arena, x).is_empty() {
        return None;
    }
    let mut signs = AssumptionCache::new();
    let mut known = |p: Props| signs.query(arena, x, p) == Some(true);
    let (holds, fails) = match (strict, flipped) {
        (true, false) => (Props::POSITIVE, Props::NONPOSITIVE),
        (false, false) => (Props::NONNEGATIVE, Props::NEGATIVE),
        (true, true) => (Props::NEGATIVE, Props::NONNEGATIVE),
        (false, true) => (Props::NONPOSITIVE, Props::POSITIVE),
    };
    if known(holds) {
        Some(true)
    } else if known(fails) {
        Some(false)
    } else {
        None
    }
}

/// `a > b` (`strict`) or `a ≥ b` for real `a`, `b` (by their assumptions)
/// whose canonical difference `a − b` is a rational number: `tanh 1 > tanh 1`
/// is false, `tanh 1 ≥ tanh 1` and `π + 1 > π` true.  `None` otherwise
/// (`∞ − ∞` is `nan`, not a number).
///
/// Before 0.41 such a relation stayed for `evalf`, which decided `a > a`
/// true: `Piecewise((π, tanh 1 < tanh 1), (0, True))` evaluated to `π`
/// (SymPy 1.14: `0`).
///
/// A constant difference that is an algebraic number is decided by its
/// minimal polynomial ([`algebraic_rational_value`]) and, when irrational,
/// by its certified sign: `√(5 − 2√6) ≥ √3 − √2` is true (the two are
/// equal; `evalf` refused it before 0.42).
pub(super) fn order_by_difference(
    arena: &mut Arena,
    diff: Difference,
    strict: bool,
) -> Option<bool> {
    let (a, b) = (diff.a, diff.b);
    let d = arena.sub(a, b);
    let rational = arena.as_num(d).cloned();
    if rational.is_none() && !walk::free_symbols(arena, d).is_empty() {
        return None;
    }
    if !known_real(arena, a) || !known_real(arena, b) {
        return None;
    }
    let q = match rational {
        Some(q) => q,
        None => {
            // A certified nonzero value decides at once (the minimal
            // polynomial of a long radical can take seconds).  `d` is real:
            // `a` and `b` are.
            if let Some(s) = certified_real_sign(arena, d) {
                return Some(s == std::cmp::Ordering::Greater);
            }
            // Arctangents of rationals: `atan(1/2) + atan(1/3) ≥ π/4`.
            let d = atan_sum::without_zero_atan_part(arena, d);
            if let Some(zero) = atan_sum::is_zero(arena, d) {
                return zero.then_some(!strict);
            }
            if sqrt_tower_is_zero(arena, d) {
                return Some(!strict);
            }
            match algebraic_rational_value(arena, d) {
                Some(Some(q)) => q,
                // Irrational, so nonzero: the certified sign decides.
                Some(None) => {
                    return Some(certified_real_sign(arena, d)? == std::cmp::Ordering::Greater);
                }
                // Trigonometric values at rational multiples of π (and
                // radicals over them): an exact zero is `a = b`; a nonzero
                // difference whose sign no evaluation certified stays.
                None => {
                    let zero = cyclotomic::is_zero(arena, d).or_else(|| {
                        let original = diff.original(arena)?;
                        exact_zero_test(arena, original)
                    })?;
                    return zero.then_some(!strict);
                }
            }
        }
    };
    Some(if strict {
        q.is_positive()
    } else {
        !q.is_negative()
    })
}

/// Is `x` real: by its assumptions, or, for a constant they leave open
/// (`√(5 − 2√6)`), by a certified evaluation without imaginary part?
///
/// The numerical half evaluates `im(x)` on its own
/// ([`constant_realness_as_declared`](crate::transforms::realness::constant_realness_as_declared)):
/// a certified value of `x` whose imaginary part shows as 0 proves
/// nothing, since an imaginary part below the relative precision of the
/// whole value is not shown — `−1 − i·(e^{10⁻³⁰} − 1 − 10⁻³⁰)` (imaginary
/// part `−5·10⁻⁶¹`) passed for a negative real, and `ln` of it folded to
/// `ln(…) + iπ` on the wrong side of the cut (it is `−iπ + …`).
///
/// `im(x)` must be a certified exact 0 (`Settled::Certified`): one that
/// is only 0 to the precision of the zero search may hide a smaller part
/// (`sin²1 + cos²1 − 1` is exactly 0, but no evaluation proves it, and the
/// side of the cut stays undecided, as `evalf` leaves it).
///
/// An algebraic `x` whose imaginary part is only 0 to the precision
/// reached is decided by its minimal polynomial ([`algebraic_is_real`]):
/// the real root of a cubic written by Cardano's formula, whose complex
/// cube roots cancel only exactly, is real (up to 0.44 it was undecided,
/// and `√λ`, `ln λ` of such a negative eigenvalue were refused by `evalf`
/// as `PrecisionExhausted` on the cut).  Beyond the reach of the minimal
/// polynomial (a root by Ferrari's formula, a radical over trigonometric
/// values) `x − x̄` is tested for zero exactly in `ℚ(ζ)[radicals]`
/// ([`cyclotomic::is_real`]): up to 0.44 the eigenvalues of a 4×4 matrix
/// with a parameter did not evaluate where the quartic has complex roots.
///
/// The zero search stops at the cap of the evaluation (twice the digits
/// plus 256 bits): searching deeper only certifies a smaller nonzero part,
/// and `x` is then not real either — an exact zero shows at any precision,
/// and one that cancels only exactly (the radicands of Ferrari's formula)
/// shows at none.  Up to 0.44 a deep search (1024 bits more) ran in vain on
/// every `eval` of a quartic's roots, and `solve` was ten times slower.
pub(super) fn known_real(arena: &mut Arena, x: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    match AssumptionCache::new().query(arena, x, Props::REAL) {
        Some(real) => real,
        None => {
            if !walk::free_symbols(arena, x).is_empty() {
                return false;
            }
            let fact = crate::base::arena::FactKey::Real;
            let real = crate::base::truth::SignSet::REAL;
            constant_fact(arena, x, fact, real, |arena| {
                // A certified nonzero imaginary part decides at once (the
                // exact tests below are for parts that cancel only exactly;
                // a certified 16-digit value shows a part of any size, here
                // 10⁻¹³ of the modulus).  Without it the cyclotomic walk did
                // exact radical arithmetic on `(((10⁻¹²i +
                // 13/4)^(−3/4))^(−3/2))^(−3/2)` for a second per `eval`
                // (`fuzz_simplify` timed out, 0.47 development).
                if let Ok((z, Settled::Certified)) = evalf_settled(arena, x, 16, ZeroSearch::Cap)
                    && !z.1.is_zero()
                {
                    return false;
                }
                // The exact conjugation test next: a walk over the
                // radicals, which settles a root by Cardano's or Ferrari's
                // formula at once, where the zero search of its imaginary
                // part runs to the cap and the minimal polynomial takes a
                // Zassenhaus factorisation (`solve` of a septic with a
                // quartic factor took 32 ms in 0.46, 1 ms in 0.43).  Any
                // other constant leaves the walk at its first node of
                // another kind.
                if let Some(real) = cyclotomic::is_real(arena, x) {
                    return real;
                }
                let im = arena.im(x);
                let im = eval(arena, im);
                if arena.is_zero_structural(im) {
                    return true;
                }
                if !walk::has_unevaluated(arena, im)
                    && let Ok((z, Settled::Certified)) =
                        evalf_settled(arena, im, 16, ZeroSearch::Cap)
                {
                    return z.0.is_zero() && z.1.is_zero();
                }
                algebraic_is_real(arena, x)
            })
        }
    }
}

/// The answer `prove` gives for the symbol-free constant `x`, memoised in
/// [`Arena::constant_facts`] under `fact` as the signs `proved` when it
/// holds and as `SignSet::UNKNOWN` when it does not: the proofs evaluate,
/// search for zeros and take minimal polynomials, and `eval` asks again on
/// every pass over the same radicals.
pub(super) fn constant_fact(
    arena: &mut Arena,
    x: ExprId,
    fact: crate::base::arena::FactKey,
    proved: crate::base::truth::SignSet,
    prove: impl FnOnce(&mut Arena) -> bool,
) -> bool {
    if let Some(known) = arena.constant_fact(x, fact) {
        return known.is_subset(proved);
    }
    let answer = prove(arena);
    let signs = if answer {
        proved
    } else {
        crate::base::truth::SignSet::UNKNOWN
    };
    arena.remember_constant_fact(x, fact, signs);
    answer
}

/// Is the constant algebraic number `x` real, by its minimal polynomial
/// `m` (small expressions only, see below)?  `m` is irreducible, so `x`
/// is one of its roots; the roots get certified disjoint inclusion disks
/// ([`crate::poly::roots::root_balls`], which also certifies a root real
/// when the mirror image of its disk meets no other disk), and `x` its
/// certified 30-digit value.  When the value's error disk meets exactly one
/// root disk, `x` is that root, and real when the root is.  `false` when
/// anything is not certified.
///
/// `x` is [`small_algebraic`], or a sum of such terms whose minimal
/// polynomials have degrees of product at most [`MAX_RESULTANT_DEGREE`]
/// ([`small_algebraic_sum`]).  The static bound of
/// [`largest_resultant_degree`] counts the radicals of `ω·∛u` (`ω` a cube
/// root of unity, `u` a quadratic complex number) as independent, and a
/// root by Cardano's formula exceeds it, though its two terms have minimal
/// polynomials of degree 6 and the resultant of the sum has degree 36 (70
/// ms in a debug build).  The test runs only for a constant whose
/// imaginary part the zero search to the cap left at 0 to its precision.
fn algebraic_is_real(arena: &mut Arena, x: ExprId) -> bool {
    use crate::base::bigcomplex::{c_abs, c_sub};
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    use astro_float::{BigFloat, RoundingMode};
    if !small_algebraic(arena, x) && !small_algebraic_sum(arena, x) {
        return false;
    }
    let Some(mp) = crate::poly::algebraic::minimal_polynomial(arena, x) else {
        return false;
    };
    match mp.degree() {
        Some(1) => return true,
        Some(d) if (2..=64).contains(&d) => {}
        _ => return false,
    }
    let Ok((z, Settled::Certified)) = evalf_settled(arena, x, 30, ZeroSearch::Cap) else {
        return false;
    };
    const PREC: usize = 256;
    let rm = RoundingMode::Up;
    let roots = crate::poly::roots::rootof_roots(&mp, PREC);
    let balls = crate::poly::roots::root_balls(&mp, &roots, PREC);
    // The value carries 30 digits relative to its modulus: its error is
    // below |z|·10⁻²⁸ (two digits of margin).
    let margin = BigFloat::from_f64(1e-28, 64);
    let eps = c_abs(&z, 64, rm).mul(&margin, 64, rm);
    let mut meets = None;
    for (k, ball) in balls.iter().enumerate() {
        let Some(r) = &ball.radius else {
            return false;
        };
        let d = c_abs(
            &c_sub(&z, &ball.value, PREC, RoundingMode::ToEven),
            64,
            RoundingMode::Down,
        );
        // Reach of the two disks, enlarged by 2⁻³⁰ for the roundings.
        let reach = r.add(&eps, 64, rm);
        let reach = reach.add(&reach.mul(&BigFloat::from_f64(1e-9, 64), 64, rm), 64, rm);
        if d.cmp(&reach).is_none_or(|o| o <= 0) {
            if meets.is_some() {
                return false;
            }
            meets = Some(k);
        }
    }
    meets.is_some_and(|k| balls[k].real)
}

/// The sign of the symbol-free constant `x`, which the caller knows to be
/// real, from a certified 16-digit value (its error ball excludes 0);
/// `None` when it does not settle or is not real to the digits.  The
/// caller's realness proof matters: an imaginary part below the relative
/// precision of the value is not shown (`−1 − i·(e^{10⁻³⁰} − 1 − 10⁻³⁰)`
/// evaluates to `−1`).  [`Arena::sign_of_real_constant`] proves realness
/// by the assumptions.
pub(crate) fn certified_real_sign(arena: &Arena, x: ExprId) -> Option<std::cmp::Ordering> {
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    if !walk::free_symbols(arena, x).is_empty() {
        return None;
    }
    match evalf_settled(arena, x, 16, ZeroSearch::Cap) {
        Ok((z, Settled::Certified)) if z.1.is_zero() && !z.0.is_zero() => {
            Some(if z.0.is_negative() {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            })
        }
        _ => None,
    }
}

/// Does the evaluation of the symbol-free constant `x` meet a value proved
/// infinite (`evalf::meets_infinity`): `x` itself or a part its value
/// needs?  `Arena::meets_infinity` delegates here.
pub(crate) fn meets_infinity(arena: &Arena, x: ExprId) -> bool {
    walk::free_symbols(arena, x).is_empty() && crate::transforms::evalf::meets_infinity(arena, x)
}

/// The signs of the real and imaginary parts of the symbol-free constant
/// `x` from one certified 16-digit value: a part is settled when it is
/// nonzero and within `2⁴⁰` of the larger part, far inside the digits the
/// value carries (a smaller part may be rounding noise next to the larger
/// one, and is left to an evaluation of its own).  `None` when the value
/// does not settle.
pub(crate) fn certified_part_signs(
    arena: &Arena,
    x: ExprId,
) -> Option<crate::base::arena::PartSigns> {
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    use std::cmp::Ordering;
    if !walk::free_symbols(arena, x).is_empty() {
        return None;
    }
    let Ok((z, Settled::Certified)) = evalf_settled(arena, x, 16, ZeroSearch::Cap) else {
        return None;
    };
    let exponent = |part: &astro_float::BigFloat| {
        if part.is_zero() || part.is_nan() || part.is_inf() {
            None
        } else {
            part.exponent()
        }
    };
    let top = exponent(&z.0).max(exponent(&z.1));
    let sign = |part: &astro_float::BigFloat| -> Option<Ordering> {
        let e = exponent(part)?;
        if top.is_some_and(|t| i64::from(e) < i64::from(t) - 40) {
            return None;
        }
        Some(if part.is_negative() {
            Ordering::Less
        } else {
            Ordering::Greater
        })
    };
    Some(crate::base::arena::PartSigns {
        re: sign(&z.0),
        im: sign(&z.1),
    })
}

/// `a = b` decided by the difference `a − b`: 0 or a nonzero Gaussian
/// rational in canonical form (`airyai(18/7) + 3 − 10⁻¹⁵⁰ ≠ 3 +
/// airyai(18/7)`, `x + 1 ≠ x`), or a constant algebraic number that its
/// minimal polynomial decides ([`algebraic_rational_value`]: `√(3 + 2√2) =
/// 1 + √2`).  `None` otherwise.
///
/// Up to 0.41 only two Gaussian rationals were compared, and both relations
/// stayed for `evalf`, which decides an inequality numerically but cannot
/// certify an equality (it refused `√(3 + 2√2) = 1 + √2`; SymPy 1.14:
/// `True`, by `minimal_polynomial` in `Expr.equals`).
fn equal_by_difference(arena: &mut Arena, diff: Difference) -> Option<bool> {
    let (a, b) = (diff.a, diff.b);
    // Numbers only: no difference of sets, truth values or relations.
    let non_scalar = |id: ExprId| {
        let node = arena.node(id);
        node.is_set_node()
            || matches!(
                node,
                ExprNode::BoolTrue
                    | ExprNode::BoolFalse
                    | ExprNode::Gt(..)
                    | ExprNode::Ge(..)
                    | ExprNode::Eq_(..)
                    | ExprNode::Ne(..)
                    | ExprNode::And(_)
                    | ExprNode::Or(_)
                    | ExprNode::Not(_)
            )
    };
    if non_scalar(a) || non_scalar(b) {
        return None;
    }
    let d = arena.sub(a, b);
    if let Some((re, im)) = gaussian_rational(arena, d) {
        return Some(re.is_zero() && im.is_zero());
    }
    if !walk::free_symbols(arena, d).is_empty() {
        return None;
    }
    // A certified nonzero value decides at once, before the minimal
    // polynomial: `tan(79π/40)·cos(79π/40) = sin(79π/40) + 1/448` took 25 s
    // there once the values were radicals.
    if constant_is_nonzero(arena, d) {
        return Some(false);
    }
    exact_zero_test(arena, d).or_else(|| {
        let original = diff.original(arena)?;
        exact_zero_test(arena, original)
    })
}

/// Is the constant `d` zero, as `eval` decides `Eq(d, 0)`?  `Some(false)`
/// when a certified 16-digit value excludes 0, else the exact tests of
/// [`exact_zero_test`] on `eval(d)`, then on `d`; `None` when nothing
/// decides (or `d` has symbols).  The zero test of `eval` of `Eq`/`Ne`
/// ([`equal_by_difference`]), of [`eval_sign`] and of the assumption
/// queries of constants ([`constant_facts`]).
pub(crate) fn constant_zero_decision(arena: &mut Arena, d: ExprId) -> Option<bool> {
    if !walk::free_symbols(arena, d).is_empty() {
        return None;
    }
    if constant_is_nonzero(arena, d) {
        return Some(false);
    }
    let v = eval(arena, d);
    if arena.is_zero_structural(v) {
        return Some(true);
    }
    exact_zero_test(arena, v).or_else(|| (v != d).then(|| exact_zero_test(arena, d)).flatten())
}

/// The exact zero tests of a symbol-free constant `d` whose certified value
/// did not exclude 0: sums of arctangents of rationals and multiples of π
/// ([`atan_sum::is_zero`]), the square-root tower, the minimal polynomial,
/// and trigonometric values at rational multiples of π, roots of unity and
/// radicals over them, exactly in `ℚ(ζ)[radicals]` ([`cyclotomic::is_zero`]).
pub(super) fn exact_zero_test(arena: &mut Arena, d: ExprId) -> Option<bool> {
    // An arctangent part that is 0 by itself leaves the rest to decide.
    let d = atan_sum::without_zero_atan_part(arena, d);
    if let Some(zero) = atan_sum::is_zero(arena, d) {
        return Some(zero);
    }
    if sqrt_tower_is_zero(arena, d) {
        return Some(true);
    }
    match algebraic_rational_value(arena, d) {
        Some(Some(q)) => Some(q.is_zero()),
        Some(None) => Some(false),
        None => cyclotomic::is_zero(arena, d),
    }
}

/// The constant algebraic number `d` as a rational, by its minimal
/// polynomial over ℚ: `Some(Some(q))` when the polynomial is linear,
/// `Some(None)` when `d` is irrational (so not 0).  The polynomial is
/// certified (`minimal_polynomial` returns a factor only when it is the one
/// irreducible factor of the resultant that vanishes at the 320-bit value
/// of `d`; two candidates, a tiny nonzero `d` among them, give `None`).
/// `None` as well when `d` is not small ([`small_algebraic`]).
fn algebraic_rational_value(arena: &mut Arena, d: ExprId) -> Option<Option<Q>> {
    if !small_algebraic(arena, d) {
        return None;
    }
    let mp = crate::poly::algebraic::minimal_polynomial(arena, d)?;
    match mp.degree()? {
        1 => Some(Some(-mp.coeff(0) / mp.coeff(1))),
        _ => Some(None),
    }
}

/// Is `d` built from rationals, `i`, `φ` and rational powers by sums and
/// products, in at most 160 nodes, with the product of the orders of its
/// roots (a bound on the degree of its minimal polynomial) at most 64, and
/// no intermediate resultant above [`MAX_RESULTANT_DEGREE`]
/// ([`largest_resultant_degree`])?  The bounds keep the resultants of
/// [`algebraic_rational_value`] small.
fn small_algebraic(arena: &Arena, d: ExprId) -> bool {
    let order = walk::post_order_ids(arena, d);
    if order.len() > 160 {
        return false;
    }
    let mut degree: u64 = 1;
    for &id in &order {
        let factor = match arena.node(id) {
            ExprNode::Num(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => 1,
            ExprNode::ImaginaryUnit | ExprNode::GoldenRatio => 2,
            ExprNode::Pow(_, e) => match arena.as_num(*e) {
                Some(q) => q.denom().to_u64().unwrap_or(u64::MAX),
                None => return false,
            },
            _ => return false,
        };
        degree = degree.saturating_mul(factor);
        if degree > 64 {
            return false;
        }
    }
    largest_resultant_degree(arena, &order) <= MAX_RESULTANT_DEGREE
}

/// The largest resultant [`algebraic_rational_value`] lets
/// `minimal_polynomial` compute (see [`largest_resultant_degree`]).
const MAX_RESULTANT_DEGREE: u64 = 64;

/// Is `x` a sum of [`small_algebraic`] terms, in at most 160 nodes, whose
/// minimal polynomials have degrees of product at most
/// [`MAX_RESULTANT_DEGREE`]?  `minimal_polynomial` composes a sum term by
/// term, by resultants of degree `deg(partial sum) · deg(term)`, so the
/// product bounds every resultant it forms on `x` (the terms' own work is
/// bounded by [`small_algebraic`]).
fn small_algebraic_sum(arena: &mut Arena, x: ExprId) -> bool {
    let ExprNode::Add(terms) = arena.node(x).clone() else {
        return false;
    };
    if walk::post_order_ids(arena, x).len() > 160 {
        return false;
    }
    let mut degree: usize = 1;
    for t in terms {
        if arena.as_num(t).is_some() {
            continue;
        }
        if !small_algebraic(arena, t) {
            return false;
        }
        let Some(d) = crate::poly::algebraic::minimal_polynomial(arena, t).and_then(|m| m.degree())
        else {
            return false;
        };
        degree = degree.saturating_mul(d);
        if degree as u64 > MAX_RESULTANT_DEGREE {
            return false;
        }
    }
    true
}

/// An upper bound on the degree of the largest resultant
/// `minimal_polynomial` forms on the nodes `order` (a post-order of a
/// [`small_algebraic`] expression): it composes sums and products one
/// operand at a time, by a resultant of degree `deg a · deg b` that is only
/// then factored, and `a^(p/q)` by one of degree `q · deg a`; a minimal
/// polynomial's degree is at most the product of the orders of the distinct
/// radicals in its expression.  The final degree alone does not bound the
/// work: `sin(7π/60)² + cos(7π/60)² − 1` has degree at most 16 but
/// resultants of degree 256 (23 s in a debug build once the values were
/// radicals).
fn largest_resultant_degree(arena: &Arena, order: &[ExprId]) -> u64 {
    // The distinct radicals `(base, order)` of each node.
    type Radicals = smallvec::SmallVec<[(ExprId, u64); 8]>;
    fn field_degree(r: &Radicals) -> u64 {
        r.iter().fold(1u64, |d, &(_, k)| d.saturating_mul(k))
    }
    fn union(a: &mut Radicals, b: &Radicals) {
        for &x in b {
            if !a.contains(&x) {
                a.push(x);
            }
        }
    }
    let mut radicals: FxHashMap<ExprId, Radicals> = FxHashMap::default();
    let mut worst = 1u64;
    for &id in order {
        let mut own = Radicals::new();
        match arena.node(id) {
            ExprNode::ImaginaryUnit | ExprNode::GoldenRatio => own.push((id, 2)),
            ExprNode::Neg(a) => {
                if let Some(r) = radicals.get(a) {
                    own = r.clone();
                }
            }
            ExprNode::Pow(b, e) => {
                let inner = radicals.get(b).cloned().unwrap_or_default();
                let q = arena
                    .as_num(*e)
                    .and_then(|q| q.denom().to_u64())
                    .unwrap_or(u64::MAX);
                own = inner;
                if q > 1 {
                    worst = worst.max(q.saturating_mul(field_degree(&own)));
                    if !own.contains(&(*b, q)) {
                        own.push((*b, q));
                    }
                }
            }
            ExprNode::Add(cs) | ExprNode::Mul(cs) => {
                let mut first = true;
                for c in cs.iter() {
                    let Some(r) = radicals.get(c).filter(|r| !r.is_empty()) else {
                        continue;
                    };
                    if !first {
                        worst = worst.max(field_degree(&own).saturating_mul(field_degree(r)));
                    }
                    first = false;
                    union(&mut own, r);
                }
            }
            _ => {}
        }
        radicals.insert(id, own);
    }
    worst
}
