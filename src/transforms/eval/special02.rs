//! Special functions (0.2): Si, Ci, Ei, li, ζ, ψ, ψ⁽ⁿ⁾, δᵢⱼ (part of [`eval`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Special functions (0.2): Si, Ci, Ei, li, ζ, ψ, ψ⁽ⁿ⁾, δᵢⱼ
// ═══════════════════════════════════════════════════════════════════════════

/// Split `x` into `(-1, y)` when `x = -y` in canonical form (negative
/// numeric coefficient or `Neg`), returning `y`.  Unlike [`as_negated`]
/// this also handles bare negative numbers and `-3*x`.
pub(super) fn as_negated_general(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if let ExprNode::Neg(inner) = arena.node(x) {
        return Some(*inner);
    }
    let (coeff, term) = arena.as_coeff_term(x);
    if coeff.is_negative() {
        Some(arena.make_coeff_term(-coeff, term))
    } else {
        None
    }
}

/// The direction of an imaginary infinity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ImaginaryInfinity {
    /// `i·∞`.
    Up,
    /// `−i·∞`.
    Down,
}

/// Is `x` the canonical `i·∞` or `−i·∞`?
fn imaginary_infinity(arena: &Arena, x: ExprId) -> Option<ImaginaryInfinity> {
    let ExprNode::Mul(children) = arena.node(x) else {
        return None;
    };
    let (i, inf) = (arena.i_unit, arena.infinity);
    match **children {
        [a, b] if a == i && b == inf => Some(ImaginaryInfinity::Up),
        [c, a, b] if c == arena.neg_one && a == i && b == inf => Some(ImaginaryInfinity::Down),
        _ => None,
    }
}

/// `±q·π·i` (`−` for `negative`).
fn signed_pi_i(arena: &mut Arena, negative: bool, q: (i64, i64)) -> ExprId {
    let c = arena.rational(if negative { -q.0 } else { q.0 }, q.1);
    arena.mul(&[c, arena.pi, arena.i_unit])
}

/// Exact values of the sine integral.
///
/// * `Si(0) = 0`, `Si(∞) = π/2`, `Si(−∞) = −π/2`, `Si(±i·∞) = ±i·∞`
///   (`Si(iy) = i·Shi(y)`, DLMF 6.5.5; SymPy: `Si(oo*I)` → `oo*I`; up to
///   0.53 it stayed, and `0·Si(i·∞)` was 0)
/// * odd: `Si(−x) = −Si(x)` (the argument is normalised to a positive
///   leading coefficient)
pub(crate) fn eval_si(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.zero);
    }
    match ValueClass::special_atom(arena, x) {
        Some(ValueClass::PosInf) => {
            let half = arena.rational(1, 2);
            return Some(arena.mul(&[half, arena.pi]));
        }
        Some(ValueClass::NegInf) => {
            let neg_half = arena.rational(-1, 2);
            return Some(arena.mul(&[neg_half, arena.pi]));
        }
        Some(ValueClass::NaN) => return Some(arena.nan),
        // (`zoo`: no limit; `−i·∞` is the odd rule's)
        _ => {}
    }
    if imaginary_infinity(arena, x) == Some(ImaginaryInfinity::Up) {
        return Some(x);
    }
    if let Some(y) = as_negated_general(arena, x) {
        let si_y = arena.si(y);
        return Some(arena.neg(si_y));
    }
    None
}

/// Exact values of the cosine integral: `Ci(∞) = 0`, `Ci(0) = zoo`.
///
/// `Ci(z) = γ + ln z + ∫₀ᶻ (cos t − 1)/t dt` has the singularity of `ln z`
/// at 0 and takes its value, `ln 0 = zoo` (SymPy: `Ci(0)` → zoo; along the
/// negative reals the value is `Ci(|x|) + iπ`).  Before 0.30 it was `−∞`,
/// the limit along the positive reals only.  `Ei(0) = li(1) = −∞` keep
/// theirs: those functions are real on both sides.
///
/// `Ci(−x) = Ci(x) + iπ` (for `x > 0`) is *not* applied automatically
/// because it changes the real/complex character of the expression; its
/// limit `Ci(−∞) = iπ` is folded, and `Ci(±i·∞) = ∞ ± iπ/2` (`Ci(iy) =
/// Chi(y) + iπ/2`, DLMF 6.5.6; mpmath `ci(10**6*1j)` =
/// 1.5·10⁴³⁴²⁸⁸ + 1.5708i; up to 0.53 they stayed).
pub(crate) fn eval_ci(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    match ValueClass::special_atom(arena, x) {
        Some(ValueClass::PosInf) => return Some(arena.zero),
        Some(ValueClass::NegInf) => return Some(signed_pi_i(arena, false, (1, 1))),
        Some(ValueClass::NaN) => return Some(arena.nan),
        // (`zoo`: no limit)
        _ => {}
    }
    if let Some(direction) = imaginary_infinity(arena, x) {
        let finite = signed_pi_i(arena, direction == ImaginaryInfinity::Down, (1, 2));
        return Some(arena.add(&[arena.infinity, finite]));
    }
    if x == arena.zero {
        return Some(arena.complex_infinity);
    }
    None
}

/// Exact values of the exponential integral:
/// `Ei(−∞) = 0`, `Ei(∞) = ∞`, `Ei(0) = −∞`, `Ei(±i·∞) = ±iπ` (`Ei(iy) =
/// Ci(y) + i(Si(y) + π/2)`; mpmath `ei(10**6*1j)` = −3.5·10⁻⁷ +
/// 3.1415917i).
pub(crate) fn eval_ei(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    match ValueClass::special_atom(arena, x) {
        Some(ValueClass::NegInf) => return Some(arena.zero),
        Some(ValueClass::PosInf) => return Some(arena.infinity),
        Some(ValueClass::NaN) => return Some(arena.nan),
        // (`zoo`: no limit)
        _ => {}
    }
    if let Some(direction) = imaginary_infinity(arena, x) {
        return Some(signed_pi_i(
            arena,
            direction == ImaginaryInfinity::Down,
            (1, 1),
        ));
    }
    if x == arena.zero {
        return Some(arena.neg_infinity);
    }
    None
}

/// Exact values of the logarithmic integral:
/// `li(0) = 0`, `li(1) = −∞`, `li(∞) = ∞`, `li(e) = Ei(1)`, `li(eʸ) = Ei(y)`.
///
/// `li z = Ei(Log z)` on all of ℂ (mpmath's and SymPy's definition), so
/// `li(eʸ) = Ei(y)` exactly when `Log eʸ = y`, i.e. `Im y ∈ (−π, π]`
/// ([`log_exp_is_identity`]): a real `y`, or a constant inside the strip.
/// Before 0.41 it was applied to every `y`: `li(e^{4i})` became `Ei(4i)`
/// (mpmath `li(exp(4j))` = `0.352018… − 3.287491…j`, `ei(4j)` =
/// `−0.140981… + 3.328999…j`), and `li(eˣ)` became `Ei(x)` for a complex
/// symbol `x`.
pub(crate) fn eval_li(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.zero);
    }
    if x == arena.one {
        return Some(arena.neg_infinity);
    }
    // `li(−∞)`, `li(±i·∞)`: `Ei(ln x) ~ x/ln x`, whose real and imaginary
    // parts both diverge, at different rates (mpmath `li(-10**6)` =
    // −7.4·10⁴ + 1.8·10⁴i): `zoo`.
    match ValueClass::special_atom(arena, x) {
        Some(ValueClass::PosInf) => return Some(arena.infinity),
        Some(ValueClass::NegInf) => return Some(arena.complex_infinity),
        Some(ValueClass::NaN) => return Some(arena.nan),
        // (`zoo`: no limit)
        _ => {}
    }
    if imaginary_infinity(arena, x).is_some() {
        return Some(arena.complex_infinity);
    }
    if x == arena.e_const {
        return Some(arena.ei(arena.one));
    }
    if let ExprNode::Exp(y) = arena.node(x).clone()
        && log_exp_is_identity(arena, y)
    {
        return Some(arena.ei(y));
    }
    None
}

/// Is `Log(eʸ) = y`, i.e. `Im y ∈ (−π, π]`?  `true` for a `y` real by its
/// assumptions, and for a constant whose exact imaginary part is `0`, `π`,
/// or certified (16 digits) strictly inside `(−π, π)` with a relative
/// margin of `10⁻⁹`; `false` when not known (a complex symbol, an
/// imaginary part too close to `±π` to decide cheaply).
pub(super) fn log_exp_is_identity(arena: &mut Arena, y: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    if AssumptionCache::new().query(arena, y, Props::REAL) == Some(true) {
        return true;
    }
    if !walk::free_symbols(arena, y).is_empty() {
        return false;
    }
    let parts = crate::base::complex::decompose(arena, y);
    if !parts.exact {
        return false;
    }
    let im = eval(arena, parts.im);
    if im == arena.pi || arena.is_zero_structural(im) {
        return true;
    }
    crate::transforms::evalf::evalf_f64(arena, im)
        .is_ok_and(|v| v.is_finite() && v.abs() < std::f64::consts::PI * (1.0 - 1e-9))
}

/// Largest even argument for which `ζ(2k)` is expanded into an exact
/// rational multiple of `π^{2k}`.
const MAX_EXACT_ZETA_EVEN: i64 = 200;

/// Exact values of the Riemann zeta function.
///
/// * `ζ(1) = z∞` (pole), `ζ(∞) = 1`, `ζ(0) = −1/2`
/// * `ζ(−n) = −Bₙ₊₁/(n+1)` for positive integers `n` (so `ζ(−2k) = 0`)
/// * `ζ(2k) = (−1)^{k+1} B₂ₖ (2π)^{2k} / (2·(2k)!)` = `π²/6, π⁴/90, …`
/// * `ζ(2k+1)` for `k ≥ 1` stays symbolic.
pub(crate) fn eval_zeta(arena: &mut Arena, s: ExprId) -> Option<ExprId> {
    match ValueClass::special_atom(arena, s) {
        Some(ValueClass::PosInf) => return Some(arena.one),
        Some(ValueClass::NaN) => return Some(arena.nan),
        // (`ζ(−∞)`, `ζ(zoo)`: no limit)
        _ => {}
    }
    let r = arena.as_num(s)?.clone();
    if !r.is_integer() {
        return None;
    }
    let n: i64 = r.to_integer().try_into().ok()?;
    if n == 1 {
        return Some(arena.complex_infinity);
    }
    if n == 0 {
        return Some(arena.rational(-1, 2));
    }
    if n < 0 {
        // ζ(−n) = −B_{n+1}/(n+1)
        let m = (-n) as usize;
        if m.is_multiple_of(2) {
            return Some(arena.zero);
        }
        // Within the digit guard of `B_{m+1}` (`zeta(−10^5)` ran the
        // quadratic rational recurrence for minutes).
        let b = exact_bernoulli(arena, m as u64 + 1)?;
        let val = -b / Ratio::from_integer(BigInt::from(m as i64 + 1));
        return guarded_num(arena, val);
    }
    if n % 2 == 0 && n <= MAX_EXACT_ZETA_EVEN {
        // ζ(2k) = |B_{2k}| · 2^{2k−1} · π^{2k} / (2k)!
        let two_k = n as usize;
        let b = crate::base::bernoulli::bernoulli(two_k).abs();
        let mut fact = BigInt::from(1);
        for i in 2..=two_k {
            fact *= BigInt::from(i as u64);
        }
        let pow2 = BigInt::from(1) << (two_k - 1);
        let coeff = b * Ratio::from_integer(pow2) / Ratio::from_integer(fact);
        let nid = arena.intern_num(coeff);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let exp_id = arena.int(n);
        let pi_pow = arena.pow(arena.pi, exp_id);
        return Some(arena.mul(&[coeff_id, pi_pow]));
    }
    None
}

/// Largest integer / half-integer shift handled exactly by the polygamma
/// and digamma recurrences.
const MAX_POLYGAMMA_SHIFT: i64 = 64;

/// Most bits the exact shifted form of `ψ⁽ⁿ⁾` may cancel (see
/// [`eval_polygamma`]); `evalf` recovers a few hundred bits of cancellation
/// within its precision search.
const MAX_POLYGAMMA_CANCEL_BITS: f64 = 200.0;

/// `n!` as a rational.
pub(super) fn factorial_ratio(n: usize) -> Q {
    Ratio::from_integer(crate::base::combinatorics::factorial(n as u64))
}

/// Exact values of the digamma function:
/// `ψ(1) = −γ`, `ψ(m) = −γ + H_{m−1}`, `ψ(1/2) = −γ − 2 ln 2`,
/// `ψ(m + 1/2) = ψ(1/2) + Σ_{k<m} 1/(k + 1/2)`, poles at non-positive integers.
pub(crate) fn eval_digamma(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    let r = arena.as_num(x)?.clone();
    let two = BigInt::from(2);
    if r.is_integer() {
        let m: i64 = r.to_integer().try_into().ok()?;
        if m <= 0 {
            return Some(arena.complex_infinity);
        }
        if m > MAX_POLYGAMMA_SHIFT {
            return None;
        }
        // ψ(m) = −γ + H_{m−1}
        let mut h = Ratio::<BigInt>::zero();
        for k in 1..m {
            h += Ratio::new(BigInt::from(1), BigInt::from(k));
        }
        let neg_gamma = arena.neg(arena.euler_gamma);
        let nid = arena.intern_num(h);
        let h_id = arena.intern(ExprNode::Num(nid));
        return Some(arena.add(&[neg_gamma, h_id]));
    }
    if *r.denom() == two {
        // x = m + 1/2 with m = floor(x)
        let m: i64 = r.floor().to_integer().try_into().ok()?;
        if !(0..=MAX_POLYGAMMA_SHIFT).contains(&m) {
            return None;
        }
        // ψ(1/2) = −γ − 2 ln 2
        let neg_gamma = arena.neg(arena.euler_gamma);
        let two_id = arena.int(2);
        let ln2 = arena.ln(two_id);
        let neg_two = arena.int(-2);
        let neg_two_ln2 = arena.mul(&[neg_two, ln2]);
        // Σ_{k=0}^{m−1} 1/(k + 1/2) = Σ 2/(2k+1)
        let mut h = Ratio::<BigInt>::zero();
        for k in 0..m {
            h += Ratio::new(BigInt::from(2), BigInt::from(2 * k + 1));
        }
        let nid = arena.intern_num(h);
        let h_id = arena.intern(ExprNode::Num(nid));
        return Some(arena.add(&[neg_gamma, neg_two_ln2, h_id]));
    }
    None
}

/// Exact values of the polygamma function `ψ⁽ⁿ⁾(x)`.
///
/// * `n = 0` → [`Digamma`](ExprNode::Digamma)
/// * `ψ⁽ⁿ⁾(1) = (−1)^{n+1} n! ζ(n+1)`
/// * `ψ⁽ⁿ⁾(1/2) = (−1)^{n+1} n! (2^{n+1} − 1) ζ(n+1)`
/// * integer / half-integer arguments are shifted to those base points via
///   `ψ⁽ⁿ⁾(x+1) = ψ⁽ⁿ⁾(x) + (−1)ⁿ n!/x^{n+1}`
/// * poles at non-positive integers → `z∞`
pub(crate) fn eval_polygamma(arena: &mut Arena, n: ExprId, x: ExprId) -> Option<ExprId> {
    if [n, x]
        .iter()
        .any(|&a| ValueClass::special_atom(arena, a) == Some(ValueClass::NaN))
    {
        return Some(arena.nan);
    }
    let nr = arena.as_num(n)?.clone();
    if !nr.is_integer() || nr.is_negative() {
        return None;
    }
    if nr.is_zero() {
        return Some(arena.digamma(x));
    }
    // ψ⁽ⁿ⁾(z) ~ (−1)ⁿ⁺¹(n − 1)!/zⁿ → 0 at `∞` and `±i·∞` (SymPy:
    // `polygamma(1, oo)` → 0; mpmath `polygamma(2, 10**6*1j)` ≈ 10⁻¹²).
    if ValueClass::special_atom(arena, x) == Some(ValueClass::PosInf)
        || imaginary_infinity(arena, x).is_some()
    {
        return Some(arena.zero);
    }
    let n_usize: usize = nr.to_integer().try_into().ok()?;
    let xr = arena.as_num(x)?.clone();
    if xr.is_integer() && !xr.is_positive() {
        return Some(arena.complex_infinity);
    }
    // The shift to the base point: m − 1 terms 1/k^{n+1} (integer x = m) or
    // m terms 2^{n+1}/(2k+1)^{n+1} (x = m + 1/2); anything else stays
    // symbolic (and n! is not computed for it).
    let half = *xr.denom() == BigInt::from(2);
    let m: i64 = if xr.is_integer() || half {
        xr.floor().to_integer().try_into().ok()?
    } else {
        return None;
    };
    if !(0..=MAX_POLYGAMMA_SHIFT).contains(&m) {
        return None;
    }
    // n! is within the digit guard, and the shifted form does not cancel:
    // ψ⁽ⁿ⁾(m) = (−1)^{n+1} n! (ζ(n+1) − Σ_{k<m} k^{−n−1}) is ≈ n!·m^{−n−1}
    // while its terms are ≈ n!, and ψ⁽ⁿ⁾(m + 1/2) ≈ n!·(m + 1/2)^{−n−1}
    // while its terms are ≈ n!·2^{n+1}, so (n+1)·log₂ m (resp.
    // (n+1)·log₂(2m+1)) bits cancel.  `polygamma(3000, 3/2)` cancelled
    // 4757 bits, beyond what `evalf` searches, and printed 0
    // (mpmath: −1.47269444412519507002495959e8602); beyond the bound the
    // node stays symbolic for `evalf`.
    if beyond_digit_guard(arena, log10_factorial_lower(n_usize as f64)) {
        return None;
    }
    let shifted_from = if half { 2 * m + 1 } else { m };
    if shifted_from >= 2
        && (n_usize as f64 + 1.0) * (shifted_from as f64).log2() > MAX_POLYGAMMA_CANCEL_BITS
    {
        return None;
    }
    let sign = if n_usize.is_multiple_of(2) { -1 } else { 1 }; // (−1)^{n+1}
    let n_fact = factorial_ratio(n_usize);
    let np1 = arena.int(n_usize as i64 + 1);
    let zeta_np1 = arena.zeta(np1);

    if xr.is_integer() {
        // ψ⁽ⁿ⁾(m) = ψ⁽ⁿ⁾(1) + (−1)ⁿ n! Σ_{k=1}^{m−1} 1/k^{n+1}
        let base_coeff = &n_fact * Ratio::from_integer(BigInt::from(sign));
        let nid = arena.intern_num(base_coeff);
        let base_coeff_id = arena.intern(ExprNode::Num(nid));
        let base = arena.mul(&[base_coeff_id, zeta_np1]);
        let mut shift = Ratio::<BigInt>::zero();
        for k in 1..m {
            let kp = BigInt::from(k).pow((n_usize + 1) as u32);
            shift += Ratio::new(BigInt::from(1), kp);
        }
        let shift = shift * &n_fact * Ratio::from_integer(BigInt::from(-sign));
        let nid = arena.intern_num(shift);
        let shift_id = arena.intern(ExprNode::Num(nid));
        return Some(arena.add(&[base, shift_id]));
    }
    if half {
        // ψ⁽ⁿ⁾(1/2) = (−1)^{n+1} n! (2^{n+1} − 1) ζ(n+1)
        let two_pow = (BigInt::from(1) << (n_usize + 1)) - BigInt::from(1);
        let base_coeff =
            &n_fact * Ratio::from_integer(two_pow) * Ratio::from_integer(BigInt::from(sign));
        let nid = arena.intern_num(base_coeff);
        let base_coeff_id = arena.intern(ExprNode::Num(nid));
        let base = arena.mul(&[base_coeff_id, zeta_np1]);
        // shift: (−1)ⁿ n! Σ_{k=0}^{m−1} 1/(k+1/2)^{n+1} = (−1)ⁿ n! Σ 2^{n+1}/(2k+1)^{n+1}
        let mut shift = Ratio::<BigInt>::zero();
        for k in 0..m {
            let denom = BigInt::from(2 * k + 1).pow((n_usize + 1) as u32);
            let numer = BigInt::from(1) << (n_usize + 1);
            shift += Ratio::new(numer, denom);
        }
        let shift = shift * &n_fact * Ratio::from_integer(BigInt::from(-sign));
        let nid = arena.intern_num(shift);
        let shift_id = arena.intern(ExprNode::Num(nid));
        return Some(arena.add(&[base, shift_id]));
    }
    None
}

/// Exact values of the Kronecker delta: `δᵢᵢ = 1`; `δᵢⱼ = 0` whenever
/// `i − j` canonicalises to a non-zero number; `nan` with a `nan` index (as
/// every function of `nan`; before 0.41 `δ(nan, nan)` was `1`, a value for
/// an undefined comparison — SymPy 1.14 leaves `KroneckerDelta(nan, nan)`).
/// Also `0` when the assumptions show `i ≠ j`: `i − j` known nonzero, or a
/// non-integer number plus terms known to be integers (`δ(1/2, n)` for an
/// integer `n`; SymPy 1.14: 0; up to 0.56 it stayed).  With a plain symbol
/// `δ(1/2, y)` stays: it is 1 at `y = 1/2` (its generic value is 0, which
/// the zero tests of series and limits respect: `may_vanish_on_a_region`).
pub(crate) fn eval_kronecker_delta(arena: &mut Arena, i: ExprId, j: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::base::truth::Truth;
    if [i, j]
        .iter()
        .any(|&a| ValueClass::special_atom(arena, a) == Some(ValueClass::NaN))
    {
        return Some(arena.nan);
    }
    if i == j {
        return Some(arena.one);
    }
    let d = arena.sub(i, j);
    if let Some(r) = arena.as_num(d) {
        return Some(if r.is_zero() { arena.one } else { arena.zero });
    }
    let mut cache = AssumptionCache::new();
    let fraction_plus_integers = match arena.node(d) {
        ExprNode::Add(terms) => {
            let terms = terms.clone();
            terms
                .first()
                .and_then(|&t| arena.as_num(t))
                .is_some_and(|q| !q.is_integer())
                && terms[1..]
                    .iter()
                    .all(|&t| Truth::from(cache.query(arena, t, Props::INTEGER)).is_true())
        }
        _ => false,
    };
    if fraction_plus_integers || Truth::from(cache.query(arena, d, Props::NONZERO)).is_true() {
        return Some(arena.zero);
    }
    None
}
