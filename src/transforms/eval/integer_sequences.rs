//! Combinatorial helper functions (part of [`eval`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Combinatorial helper functions
// ═══════════════════════════════════════════════════════════════════════════

// ── The digit guard of the integer sequences ─────────────────────────────
//
// `n!!`, `!n`, `Fₙ`, `Lₙ`, `Bₙ`, `Hₙ`, `Cₙ`, Bell and Euler numbers,
// Stirling numbers and `p(n)` follow the policy of `n!` above: a lower
// bound on the digits of the result refuses before any arithmetic, the
// exact count decides at the limit ([`guarded_num`]), and beyond it the
// node stays symbolic for `evalf`.  Up to 0.29 none was bounded:
// `fibonacci(10^7)`, `bernoulli(10^5)`, `harmonic(10^6)`, `catalan(10^6)`,
// `bell(10^4)`, `euler_number(10^4)`, `stirling2(10^4, 5000)` and
// `partition_count(10^7)` each ran for more than 20 s (release), and
// `subfactorial(10^5)` built a 456 574-digit integer in 4 s.  The Stirling
// triangle and the pentagonal recurrence of `p(n)` are also bounded in
// work, since their cost outgrows the digits of the result.

/// `log₁₀ φ` of the golden ratio.
const LOG10_PHI: f64 = 0.208_987_640_249_978_73;

/// A non-negative integer argument that fits in a `u64`.
fn nonneg_int_arg(arena: &Arena, id: ExprId) -> Option<u64> {
    let r = arena.as_num(id)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    r.to_integer().try_into().ok()
}

/// An integer as a guarded node (see [`guarded_num`]).
fn guarded_int(arena: &mut Arena, n: BigInt) -> Option<ExprId> {
    guarded_num(arena, Ratio::from_integer(n))
}

/// Double factorial: n!! = n * (n-2) * (n-4) * ... * 1 (or 2).
/// 0!! = 1, 1!! = 1, (-1)!! = 1.  Within the digit guard: `n!! ≥ (2m)!! =
/// 2^m·m!` for `m = ⌊n/2⌋`.
pub(super) fn eval_factorial2(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() {
        return None;
    }
    let n: i64 = r.to_integer().try_into().ok()?;
    // A negative odd `n = −(2m + 1)`: `n!! = (n + 2)!!/(n + 2)`, so
    // `(−1)^m/(2m − 1)!!` (SymPy: `factorial2(−3) = −1`, `factorial2(−17) =
    // 1/2027025`).  A negative even `n` is a pole of the extension: left
    // alone.  Before 0.37 every `n < −1` stayed unevaluated.
    if n < -1 && n % 2 != 0 {
        let m = (-n - 1) / 2;
        if beyond_digit_guard(
            arena,
            (m as f64) * std::f64::consts::LOG10_2 + log10_factorial_lower(m as f64),
        ) {
            return None;
        }
        let mut odd = BigInt::one();
        let mut k = 2 * m - 1;
        while k > 1 {
            odd *= BigInt::from(k);
            k -= 2;
        }
        let sign = if m % 2 == 0 {
            BigInt::one()
        } else {
            -BigInt::one()
        };
        return guarded_num(arena, Ratio::new(sign, odd));
    }
    if n < -1 {
        return None;
    }
    let m = (n.max(0) / 2) as f64;
    if beyond_digit_guard(
        arena,
        m * std::f64::consts::LOG10_2 + log10_factorial_lower(m),
    ) {
        return None;
    }
    let mut result = BigInt::from(1);
    let mut k = n;
    while k > 1 {
        result *= BigInt::from(k);
        k -= 2;
    }
    guarded_int(arena, result)
}

/// Subfactorial (derangement count): !n.
/// Uses recurrence: !0 = 1, !1 = 0, !n = (n-1)(!(n-1) + !(n-2)).
/// Within the digit guard: `!n = round(n!/e) ≥ n!/3` for `n ≥ 4`.
pub(super) fn eval_subfactorial(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = nonneg_int_arg(arena, inner)?;
    if n >= 4 && beyond_digit_guard(arena, log10_factorial_lower(n as f64) - 3f64.log10()) {
        return None;
    }
    let result = if n == 0 {
        BigInt::from(1)
    } else {
        let mut prev = BigInt::from(1); // !0
        let mut curr = BigInt::from(0); // !1
        for i in 2..=n {
            let next = BigInt::from(i - 1) * (&curr + &prev);
            prev = curr;
            curr = next;
        }
        curr
    };
    guarded_int(arena, result)
}

/// Rising factorial (Pochhammer): (x)_n = x * (x+1) * ... * (x+n-1).
/// Works for rational x; n must be a non-negative integer.  Within the
/// digit guard (see [`beyond_digit_guard`]).
pub(super) fn eval_rising_factorial(
    arena: &mut Arena,
    x_id: ExprId,
    n_id: ExprId,
) -> Option<ExprId> {
    let xr = arena.as_num(x_id)?.clone();
    let nr = arena.as_num(n_id)?;
    if !nr.is_integer() {
        return None;
    }
    // A negative order: `(x)₋ₘ = 1/((x − 1)(x − 2)⋯(x − m))`, `zoo` at a zero
    // factor (SymPy's `rf(3, −1) = 1/2`, `rf(3, −3) = zoo`).
    if nr.is_negative() {
        let m: u64 = (-nr.to_integer()).try_into().ok()?;
        let below = xr - Ratio::one();
        return reciprocal_factorial_power(arena, &below, m, -1);
    }
    let n: u64 = nr.to_integer().try_into().ok()?;
    exact_factorial_power(arena, &xr, n, 1)
}

/// `1/((x)(x + s)⋯(x + (m − 1)s))` for `s = step = ±1`: `zoo` when a factor
/// is 0 (a negative-order rising or falling factorial).
fn reciprocal_factorial_power(arena: &mut Arena, x: &Q, m: u64, step: i32) -> Option<ExprId> {
    let p = exact_factorial_power(arena, x, m, step)?;
    let v = arena.as_num(p)?.clone();
    if v.is_zero() {
        return Some(arena.complex_infinity);
    }
    guarded_num(arena, v.recip())
}

/// Falling factorial: x^(n) = x * (x-1) * ... * (x-n+1).
/// Works for rational x; n must be a non-negative integer.  Within the
/// digit guard (see [`beyond_digit_guard`]).
pub(super) fn eval_falling_factorial(
    arena: &mut Arena,
    x_id: ExprId,
    n_id: ExprId,
) -> Option<ExprId> {
    let xr = arena.as_num(x_id)?.clone();
    let nr = arena.as_num(n_id)?;
    if !nr.is_integer() {
        return None;
    }
    // A negative order: `x^(−m) = 1/((x + 1)(x + 2)⋯(x + m))`, `zoo` at a zero
    // factor (SymPy's `ff(3, −1) = 1/4`, `ff(−1, −1) = zoo`).
    if nr.is_negative() {
        let m: u64 = (-nr.to_integer()).try_into().ok()?;
        let above = xr + Ratio::one();
        return reciprocal_factorial_power(arena, &above, m, 1);
    }
    let n: u64 = nr.to_integer().try_into().ok()?;
    exact_factorial_power(arena, &xr, n, -1)
}

/// An integer argument that fits in an `i64`.
fn int_arg(arena: &Arena, id: ExprId) -> Option<i64> {
    let r = arena.as_num(id)?;
    if !r.is_integer() {
        return None;
    }
    r.to_integer().try_into().ok()
}

/// Fibonacci number `F(n)` for any integer `n` (`F(−n) = (−1)^{n+1}·F(n)`,
/// SymPy: `fibonacci(−5) = 5`; before 0.40 a negative index stayed
/// unevaluated), by fast doubling ([`crate::base::combinatorics::fibonacci`]).
/// Within the digit guard: `|Fₙ| ≥ φ^(|n|−2)`.
pub(super) fn eval_fibonacci(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = int_arg(arena, inner)?;
    if beyond_digit_guard(arena, (n.unsigned_abs() as f64 - 2.0) * LOG10_PHI) {
        return None;
    }
    guarded_int(arena, crate::base::combinatorics::fibonacci(i128::from(n)))
}

/// Lucas number `L(n)` for any integer `n` (`L(−n) = (−1)ⁿ·L(n)`, SymPy:
/// `lucas(−3) = −4`; before 0.40 a negative index stayed unevaluated).
/// Within the digit guard: `|Lₙ| ≥ φ^(|n|−1)`.
pub(super) fn eval_lucas(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = int_arg(arena, inner)?;
    if beyond_digit_guard(arena, (n.unsigned_abs() as f64 - 1.0) * LOG10_PHI) {
        return None;
    }
    guarded_int(arena, crate::base::combinatorics::lucas(i128::from(n)))
}

/// The Bernoulli number `Bₙ` exactly, or `None` beyond the digit guard,
/// with `B₁ = +1/2` (`base::bernoulli`; up to 0.49 `−1/2`).
///
/// `|Bₙ| = 2·n!·ζ(n)/(2π)ⁿ ≥ 2·n!/(2π)ⁿ` for even `n ≥ 2` bounds the
/// digits first.  Even indices come from the tangent numbers,
/// `B₂ₘ = (−1)^(m−1)·2m·Tₘ/(2^(2m)·(2^(2m) − 1))`, computed by Brent and
/// Harvey's integer recurrence ("Fast computation of Bernoulli, Tangent
/// and Secant numbers", 2011, Algorithm TangentNumbers): `O(m²)`
/// small-integer multiplications and additions and one final gcd, where
/// the rational recurrence `Bₘ = −Σ C(m+1, k)·Bₖ/(m+1)` took a gcd per
/// term (`bernoulli(2000)` inside the guard would have taken minutes).
/// The numbers are those of the asymptotic series of `evalf`, one cached
/// implementation (`evalf::bernoulli`; before 0.30 this file had its own
/// copy of the recurrence).
pub(crate) fn exact_bernoulli(arena: &Arena, n: u64) -> Option<Q> {
    match n {
        0 => return Some(Ratio::one()),
        1 => return Some(Ratio::new(BigInt::from(1), BigInt::from(2))),
        _ if n % 2 == 1 => return Some(Ratio::zero()),
        _ => {}
    }
    let nf = n as f64;
    let lower = std::f64::consts::LOG10_2 + log10_factorial_lower(nf)
        - nf * (2.0 * std::f64::consts::PI).log10();
    if beyond_digit_guard(arena, lower) {
        return None;
    }
    let m = usize::try_from(n / 2).ok()?;
    let r = crate::transforms::evalf::bernoulli::even(m);
    let digits = r.numer().to_string().len() + r.denom().to_string().len();
    (digits <= arena.config.max_result_digits).then_some(r)
}

/// Bernoulli number B(n), within the digit guard ([`exact_bernoulli`]).
pub(super) fn eval_bernoulli(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = nonneg_int_arg(arena, inner)?;
    let b = exact_bernoulli(arena, n)?;
    guarded_num(arena, b)
}

/// A lower bound on the decimal digits of the reduced denominator of
/// `Hₙ`: every prime `p ∈ (n/2, n]` divides it (`1/p` is the only term
/// with a factor `p`), so it is at least `e^(θ(n) − θ(n/2))`, and Rosser
/// and Schoenfeld's bounds (1962, Theorem 4 and its corollary:
/// `θ(x) > x(1 − 1/ln x)` for `x ≥ 41`, `θ(x) < x(1 + 1/(2 ln x))` for
/// `x > 1`) bound the difference.
fn harmonic_digits_lower(n: u64) -> f64 {
    if n < 82 {
        return 0.0;
    }
    let x = n as f64;
    let h = x / 2.0;
    let theta_diff = x * (1.0 - 1.0 / x.ln()) - h * (1.0 + 1.0 / (2.0 * h.ln()));
    (theta_diff / std::f64::consts::LN_10).max(0.0)
}

/// Harmonic number H(n) = 1 + 1/2 + 1/3 + ... + 1/n.
/// H(0) = 0.  Within the digit guard ([`harmonic_digits_lower`]); the sum
/// is taken over the common denominator `lcm(1, …, n)` and reduced once,
/// where a gcd per term made `Hₙ` quadratic in its digits.  A negative
/// integer is a pole of `H(z) = ψ(z + 1) + γ`: `zoo` (SymPy:
/// `harmonic(-1)` → zoo; before 0.31 it stayed unevaluated).
pub(super) fn eval_harmonic(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if arena
        .as_num(inner)
        .is_some_and(|r| r.is_integer() && r.is_negative())
    {
        return Some(arena.complex_infinity);
    }
    let n = nonneg_int_arg(arena, inner)?;
    if beyond_digit_guard(arena, harmonic_digits_lower(n)) {
        return None;
    }
    let mut lcm = BigInt::one();
    for k in 1..=n {
        let kb = BigInt::from(k);
        let g = num_integer::Integer::gcd(&(&lcm % &kb), &kb);
        lcm *= kb / g;
    }
    let mut numer = BigInt::zero();
    for k in 1..=n {
        numer += &lcm / BigInt::from(k);
    }
    guarded_num(arena, Ratio::new(numer, lcm))
}

/// Catalan number C(n) = (2n)! / ((n+1)! * n!).  Within the digit guard:
/// `Cₙ ≥ 4ⁿ/((2n + 1)(n + 1))`.
///
/// Off the non-negative integers `C(x) = Γ(2x + 1)/(Γ(x + 1)·Γ(x + 2))`,
/// whose limits along the integers SymPy takes: `C(−1) = −1/2`, `C(n) = 0`
/// for `n ≤ −2`, and a pole (`zoo`) at the negative half-integers, where
/// only the numerator meets one.  (Before 0.40 all of these stayed
/// unevaluated.)
pub(super) fn eval_catalan(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if let Some(r) = arena.as_num(inner)
        && r.is_negative()
    {
        if r.is_integer() {
            return Some(if r.to_integer() == BigInt::from(-1) {
                arena.rational(-1, 2)
            } else {
                arena.zero
            });
        }
        if *r.denom() == BigInt::from(2) {
            return Some(arena.complex_infinity);
        }
        return None;
    }
    let n = nonneg_int_arg(arena, inner)?;
    let nf = n as f64;
    if beyond_digit_guard(
        arena,
        nf * 4f64.log10() - ((2.0 * nf + 1.0) * (nf + 1.0)).log10(),
    ) {
        return None;
    }
    // C(n) = C(2n, n) / (n+1) — compute using incremental binomial
    let mut binom = BigInt::from(1); // C(2n, n) built incrementally
    for i in 0..n {
        binom *= BigInt::from(2 * n - i);
        binom /= BigInt::from(i + 1);
    }
    guarded_num(arena, Ratio::new(binom, BigInt::from(n + 1)))
}

/// Largest `n·k·limbs` of the Stirling triangle (`O(n·k)` big-integer
/// operations on numbers of the result's size) computed exactly.
const MAX_STIRLING_WORK: f64 = 2.0e8;

/// `(n, k)` of a Stirling number within the digit guard and the work
/// bound: `S(n, k) ≥ k^(n−k)` (the first `k` elements in distinct blocks,
/// the others anywhere), and `|s(n, k)| ≥ S(n, k)` (every partition into
/// `k` blocks gives a permutation with `k` cycles), `|s(n, 1)| = (n − 1)!`.
/// The closed forms of `combinatorics` for `k ≤ 2` (second kind),
/// `k = 1` (first kind) and `k ≥ n − 1` cost no triangle.
fn stirling_args(
    arena: &Arena,
    n_id: ExprId,
    k_id: ExprId,
    first_kind: bool,
) -> Option<(u64, u64)> {
    let n = nonneg_int_arg(arena, n_id)?;
    let k = nonneg_int_arg(arena, k_id)?;
    if k == 0 || k > n {
        return Some((n, k));
    }
    let (nf, kf) = (n as f64, k as f64);
    let mut lower = (nf - kf) * kf.log10();
    if first_kind && k == 1 {
        lower = log10_factorial_lower(nf - 1.0);
    }
    if beyond_digit_guard(arena, lower) {
        return None;
    }
    let closed_form = k + 1 >= n || k == 1 || (!first_kind && k == 2);
    if !closed_form && nf * kf * (1.0 + lower / 19.0) > MAX_STIRLING_WORK {
        return None;
    }
    Some((n, k))
}

/// Stirling number of the second kind S(n, k), within the digit guard and
/// the work bound ([`stirling_args`]).
pub(super) fn eval_stirling2(arena: &mut Arena, n_id: ExprId, k_id: ExprId) -> Option<ExprId> {
    if let Some(zero) = stirling_negative_k(arena, n_id, k_id) {
        return Some(zero);
    }
    let (n, k) = stirling_args(arena, n_id, k_id, false)?;
    let result = crate::domains::combinatorics::stirling2(n, k)?;
    guarded_int(arena, result)
}

/// Signed Stirling number of the first kind s(n, k), within the digit guard
/// and the work bound ([`stirling_args`]).
pub(super) fn eval_stirling1(arena: &mut Arena, n_id: ExprId, k_id: ExprId) -> Option<ExprId> {
    if let Some(zero) = stirling_negative_k(arena, n_id, k_id) {
        return Some(zero);
    }
    let (n, k) = stirling_args(arena, n_id, k_id, true)?;
    let result = crate::domains::combinatorics::stirling1(n, k)?;
    guarded_int(arena, result)
}

/// Largest `n` whose partition count is computed exactly: the pentagonal
/// recurrence costs `O(n^{3/2})` additions and keeps all `p(k)`, `k ≤ n`
/// (`p(40000)` has 218 digits, far inside the digit guard, and takes
/// about half a second in release).
const MAX_PARTITION_N: u64 = 40_000;

/// Both kinds of Stirling numbers are 0 for an integer `n ≥ 0` and a
/// negative integer `k` (no partition or permutation of `n` elements has a
/// negative number of blocks or cycles; the triangles of Graham–Knuth–
/// Patashnik §6.1 extended by the recurrences).  Up to 0.53
/// `stirling2(8, −1)` stayed unevaluated (SymPy keeps it too).
fn stirling_negative_k(arena: &mut Arena, n_id: ExprId, k_id: ExprId) -> Option<ExprId> {
    nonneg_int_arg(arena, n_id)?;
    let k = arena.as_num(k_id)?;
    (k.is_integer() && k.is_negative()).then_some(arena.zero)
}

/// Number of integer partitions p(n), for `n ≤` [`MAX_PARTITION_N`]; 0 for
/// a negative integer `n` (the coefficients of `Π 1/(1 − xᵏ)`; SymPy:
/// `partition(−1) = 0`; up to 0.53 it stayed unevaluated).
pub(super) fn eval_partition_count(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if arena
        .as_num(inner)
        .is_some_and(|r| r.is_integer() && r.is_negative())
    {
        return Some(arena.zero);
    }
    let n = nonneg_int_arg(arena, inner)?;
    if n > MAX_PARTITION_N {
        return None;
    }
    let result = crate::domains::combinatorics::partition_count(n)?;
    guarded_int(arena, result)
}

/// Bell number B(n) using the Bell triangle.
/// B(0) = 1, B(1) = 1, B(2) = 2, B(3) = 5, B(4) = 15, B(5) = 52.
/// Within the digit guard: `Bₙ ≥ S(n, k) ≥ k^(n−k)` for any `k`, taken at
/// `k ≈ n/ln n`.
pub(super) fn eval_bell(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = nonneg_int_arg(arena, inner)?;
    if n >= 3 {
        let nf = n as f64;
        let k = (nf / nf.ln()).round().clamp(1.0, nf);
        if beyond_digit_guard(arena, (nf - k) * k.log10()) {
            return None;
        }
    }
    if n == 0 {
        let nid = arena.intern_num(Ratio::from_integer(BigInt::from(1)));
        return Some(arena.intern(ExprNode::Num(nid)));
    }
    // Bell triangle: row[0] = B(n-1), then row[j] = row[j-1] + prev_row[j-1]
    let mut row = vec![BigInt::from(1)]; // B(0) = 1, start of row 1
    for _ in 1..n {
        let mut new_row = Vec::with_capacity(row.len() + 1);
        new_row.push(row.last()?.clone()); // first element = last of prev row
        for j in 1..=row.len() {
            let val = &new_row[j - 1] + &row[j - 1];
            new_row.push(val);
        }
        row = new_row;
    }
    // B(n) = last element of the nth row
    let result = row.pop()?;
    guarded_int(arena, result)
}

/// Euler number E(n). Odd indices give 0.
/// E(0) = 1, E(2) = -1, E(4) = 5, E(6) = -61, ...
/// Recurrence for even n >= 2: E(n) = -sum_{k=0,2,4,...,n-2} C(n, k) * E(k).
///
/// Within the digit guard: `|Eₙ| = 2^(n+2)·n!·β(n+1)/π^(n+1)` with the
/// Dirichlet beta `β(s) ≥ 1 − 3^(−s) ≥ 2/3`.
///
/// `E₋₁ = π/2`: SymPy extends `Eₛ = 2^(s+1)·(ζ(−s, 1/2) − 2^(s+1)·ζ(−s, 3/4))`
/// off the non-negative integers, and at `s = −1` the poles of the two
/// Hurwitz zetas cancel to `ψ(3/4) − ψ(1/2) + ln 2 = π/2` (SymPy:
/// `euler(−1) = pi/2`; before 0.40 it stayed unevaluated).
pub(super) fn eval_euler_number(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if arena
        .as_num(inner)
        .is_some_and(|r| r.is_integer() && r.to_integer() == BigInt::from(-1))
    {
        let half = arena.rational(1, 2);
        let pi = arena.pi;
        return Some(arena.mul(&[half, pi]));
    }
    let n = nonneg_int_arg(arena, inner)?;
    // Odd Euler numbers are 0
    if n % 2 == 1 {
        let nid = arena.intern_num(Ratio::from_integer(BigInt::from(0)));
        return Some(arena.intern(ExprNode::Num(nid)));
    }
    let nf = n as f64;
    let lower = log10_factorial_lower(nf) + (nf + 2.0) * std::f64::consts::LOG10_2
        - (nf + 1.0) * std::f64::consts::PI.log10()
        + (2.0f64 / 3.0).log10();
    if n >= 2 && beyond_digit_guard(arena, lower) {
        return None;
    }
    // The secant numbers (Brent–Harvey) behind `ntheory::euler_number`,
    // bounded here by the digit guard alone; before 0.40 a recurrence over
    // binomials, products of big numbers: 0.47 s for `E₁₈₀₀`.
    let result = crate::base::combinatorics::euler_even(usize::try_from(n / 2).ok()?);
    guarded_int(arena, result)
}

fn is_pi(arena: &Arena, id: ExprId) -> bool {
    id == arena.pi
}

/// Check if `id` is the constant `E`.
pub(super) fn is_e(arena: &Arena, id: ExprId) -> bool {
    id == arena.e_const
}

/// Check if `id` is a rational multiple of π: returns the multiplier
/// as `Some(Ratio)`, or `None` if not a multiple of π.
pub(super) fn as_pi_multiple(arena: &Arena, id: ExprId) -> Option<Q> {
    // Exact π.
    if is_pi(arena, id) {
        return Some(Ratio::one());
    }

    // c * π where c is numeric.
    if let ExprNode::Mul(ref children) = arena.node(id).clone()
        && children.len() == 2
        && let ExprNode::Num(nid) = arena.node(children[0])
    {
        let nid = *nid;
        if children[1] == arena.pi {
            return Some(arena.num(nid).clone());
        }
    }

    // Numeric 0 (= 0 * π).
    if let Some(r) = arena.as_num(id)
        && r.is_zero()
    {
        return Some(Ratio::zero());
    }

    None
}

/// Which function [`first_quadrant`] reduces.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TrigKind {
    Sin,
    Cos,
    Tan,
}

/// `sin(qπ)`, `cos(qπ)`, `tan(qπ)` at a rational `q ≥ 0` without a value in
/// radicals, reduced to an angle in `[0, π/2]` by the periods and
/// reflections, as SymPy's `sin.eval` and `cos.eval` reduce them
/// (`sin(104π/105)` → `sin(π/105)`, `cos(106π/105)` → `−cos(π/105)`,
/// `tan(10π/7)` → `tan(3π/7)`); `None` when the angle is already there.
/// Up to 0.46 they stayed, and `sin(104π/105) − sin(π/105)` was not 0.
fn first_quadrant(arena: &mut Arena, q: &Q, kind: TrigKind) -> Option<ExprId> {
    let one = Q::one();
    let half = Ratio::new(BigInt::from(1), BigInt::from(2));
    let period = if kind == TrigKind::Tan {
        one.clone()
    } else {
        Q::from_integer(BigInt::from(2))
    };
    let mut x = mod_positive(q, &period);
    let mut negative = false;
    match kind {
        TrigKind::Sin => {
            if x > one {
                x -= &one;
                negative = true;
            }
            if x > half {
                x = &one - &x;
            }
        }
        TrigKind::Cos => {
            if x > one {
                x -= &one;
                negative = true;
            }
            if x > half {
                x = &one - &x;
                negative = !negative;
            }
        }
        TrigKind::Tan => {
            if x > half {
                x = &one - &x;
                negative = true;
            }
        }
    }
    if x == *q {
        return None;
    }
    let coeff = arena.num_ratio(x);
    let angle = arena.mul(&[coeff, arena.pi]);
    let value = match kind {
        TrigKind::Sin => eval_sin(arena, angle).unwrap_or_else(|| arena.sin(angle)),
        TrigKind::Cos => eval_cos(arena, angle).unwrap_or_else(|| arena.cos(angle)),
        TrigKind::Tan => eval_tan(arena, angle).unwrap_or_else(|| arena.tan(angle)),
    };
    Some(if negative { arena.neg(value) } else { value })
}

/// Evaluate `sin(inner)` for known special values.
pub(super) fn eval_sin(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Odd function: sin(-x) = -sin(x)
    // The special value of the positive argument is taken too: returning
    // `−sin(π)` left `eval(sin(−π))` non-zero to structural tests (Gruntz
    // took it for a leading coefficient, and `lim_{x→0⁺} x/sin(x − π)` was 0).
    if let Some(pos_inner) = as_negated(arena, inner) {
        let sin_pos = eval_sin(arena, pos_inner).unwrap_or_else(|| arena.sin(pos_inner));
        return Some(arena.neg(sin_pos));
    }

    // sin(i*x) = i*sinh(x) — trig-hyperbolic bridge
    if let Some(real_part) = as_pure_imaginary(arena, inner) {
        let sinh_val = arena.sinh(real_part);
        return Some(arena.mul(&[arena.i_unit, sinh_val]));
    }

    let coeff = as_pi_multiple(arena, inner)?;
    // Every value below is at a denominator dividing 4 or 6, which the
    // reductions keep (see `denominator_divides`); the others SymPy
    // evaluates are in radicals ([`sin_pi_radical`]).
    if !is_classical_denominator(&coeff) {
        return sin_pi_radical(arena, &coeff)
            .or_else(|| first_quadrant(arena, &coeff, TrigKind::Sin));
    }

    // Reduce modulo 2 (sin has period 2π).
    let two: Q = Ratio::from_integer(2.into());
    let coeff = mod_positive(&coeff, &two);

    // Known values of sin(k*π).
    // sin(0) = 0
    if coeff.is_zero() {
        return Some(arena.zero);
    }
    // sin(π/2) = 1
    if coeff == Ratio::new(1.into(), 2.into()) {
        return Some(arena.one);
    }
    // sin(π) = 0
    if coeff == Ratio::one() {
        return Some(arena.zero);
    }
    // sin(3π/2) = -1
    if coeff == Ratio::new(3.into(), 2.into()) {
        return Some(arena.neg_one);
    }
    // sin(π/6) = 1/2
    if coeff == Ratio::new(1.into(), 6.into()) {
        let half = arena.rational(1, 2);
        return Some(half);
    }
    // sin(5π/6) = 1/2
    if coeff == Ratio::new(5.into(), 6.into()) {
        let half = arena.rational(1, 2);
        return Some(half);
    }
    // sin(7π/6) = -1/2
    if coeff == Ratio::new(7.into(), 6.into()) {
        let neg_half = arena.rational(-1, 2);
        return Some(neg_half);
    }
    // sin(11π/6) = -1/2
    if coeff == Ratio::new(11.into(), 6.into()) {
        let neg_half = arena.rational(-1, 2);
        return Some(neg_half);
    }
    // sin(π/4) = √2/2
    if coeff == Ratio::new(1.into(), 4.into()) {
        let two = arena.int(2);
        let half_exp = arena.rational(1, 2);
        let sqrt2 = arena.pow(two, half_exp);
        let half = arena.rational(1, 2);
        return Some(arena.mul(&[half, sqrt2]));
    }
    // sin(π/3) = √3/2
    if coeff == Ratio::new(1.into(), 3.into()) {
        let three = arena.int(3);
        let half_exp = arena.rational(1, 2);
        let sqrt3 = arena.pow(three, half_exp);
        let half = arena.rational(1, 2);
        return Some(arena.mul(&[half, sqrt3]));
    }

    // ── Quadrant reductions (covers all remaining standard angles) ──
    // Q2: sin(π - x) = sin(x), for coeff in (1/2, 1)
    let half = Ratio::new(1.into(), 2.into());
    if coeff > half && coeff < Ratio::one() {
        let reflected = Ratio::one() - &coeff;
        let nid = arena.intern_num(reflected);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reflected_id = arena.mul(&[coeff_id, arena.pi]);
        return eval_sin(arena, reflected_id);
    }
    // Q3: sin(π + x) = -sin(x), for coeff in (1, 3/2)
    let three_half = Ratio::new(3.into(), 2.into());
    if coeff > Ratio::one() && coeff < three_half {
        let reduced = &coeff - Ratio::one();
        let nid = arena.intern_num(reduced);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reduced_id = arena.mul(&[coeff_id, arena.pi]);
        if let Some(val) = eval_sin(arena, reduced_id) {
            return Some(arena.neg(val));
        }
    }
    // Q4: sin(2π - x) = -sin(x), for coeff in (3/2, 2)
    let two: Q = Ratio::from_integer(2.into());
    if coeff > three_half && coeff < two {
        let reduced = two - &coeff;
        let nid = arena.intern_num(reduced);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reduced_id = arena.mul(&[coeff_id, arena.pi]);
        if let Some(val) = eval_sin(arena, reduced_id) {
            return Some(arena.neg(val));
        }
    }

    None
}

/// Evaluate `cos(inner)` for known special values.
pub(super) fn eval_cos(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Even function: cos(-x) = cos(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        return Some(eval_cos(arena, pos_inner).unwrap_or_else(|| arena.cos(pos_inner)));
    }

    // cos(i*x) = cosh(x) — trig-hyperbolic bridge
    if let Some(real_part) = as_pure_imaginary(arena, inner) {
        return Some(arena.cosh(real_part));
    }

    let coeff = as_pi_multiple(arena, inner)?;
    if !is_classical_denominator(&coeff) {
        return cos_pi_radical(arena, &coeff)
            .or_else(|| first_quadrant(arena, &coeff, TrigKind::Cos));
    }

    let two: Q = Ratio::from_integer(2.into());
    let coeff = mod_positive(&coeff, &two);

    // cos(0) = 1
    if coeff.is_zero() {
        return Some(arena.one);
    }
    // cos(π/2) = 0
    if coeff == Ratio::new(1.into(), 2.into()) {
        return Some(arena.zero);
    }
    // cos(π) = -1
    if coeff == Ratio::one() {
        return Some(arena.neg_one);
    }
    // cos(3π/2) = 0
    if coeff == Ratio::new(3.into(), 2.into()) {
        return Some(arena.zero);
    }
    // cos(π/3) = 1/2
    if coeff == Ratio::new(1.into(), 3.into()) {
        let half = arena.rational(1, 2);
        return Some(half);
    }
    // cos(2π/3) = -1/2
    if coeff == Ratio::new(2.into(), 3.into()) {
        let neg_half = arena.rational(-1, 2);
        return Some(neg_half);
    }
    // cos(4π/3) = -1/2
    if coeff == Ratio::new(4.into(), 3.into()) {
        let neg_half = arena.rational(-1, 2);
        return Some(neg_half);
    }
    // cos(5π/3) = 1/2
    if coeff == Ratio::new(5.into(), 3.into()) {
        let half = arena.rational(1, 2);
        return Some(half);
    }
    // cos(π/4) = √2/2
    if coeff == Ratio::new(1.into(), 4.into()) {
        let two = arena.int(2);
        let half_exp = arena.rational(1, 2);
        let sqrt2 = arena.pow(two, half_exp);
        let half = arena.rational(1, 2);
        return Some(arena.mul(&[half, sqrt2]));
    }
    // cos(π/6) = √3/2
    if coeff == Ratio::new(1.into(), 6.into()) {
        let three = arena.int(3);
        let half_exp = arena.rational(1, 2);
        let sqrt3 = arena.pow(three, half_exp);
        let half = arena.rational(1, 2);
        return Some(arena.mul(&[half, sqrt3]));
    }

    // ── Quadrant reductions (covers all remaining standard angles) ──
    // Q2: cos(π - x) = -cos(x), for coeff in (1/2, 1)
    let half = Ratio::new(1.into(), 2.into());
    if coeff > half && coeff < Ratio::one() {
        let reflected = Ratio::one() - &coeff;
        let nid = arena.intern_num(reflected);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reflected_id = arena.mul(&[coeff_id, arena.pi]);
        if let Some(val) = eval_cos(arena, reflected_id) {
            return Some(arena.neg(val));
        }
    }
    // Q3: cos(π + x) = -cos(x), for coeff in (1, 3/2)
    let three_half = Ratio::new(3.into(), 2.into());
    if coeff > Ratio::one() && coeff < three_half {
        let reduced = &coeff - Ratio::one();
        let nid = arena.intern_num(reduced);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reduced_id = arena.mul(&[coeff_id, arena.pi]);
        if let Some(val) = eval_cos(arena, reduced_id) {
            return Some(arena.neg(val));
        }
    }
    // Q4: cos(2π - x) = cos(x), for coeff in (3/2, 2)
    let two: Q = Ratio::from_integer(2.into());
    if coeff > three_half && coeff < two {
        let reduced = two - &coeff;
        let nid = arena.intern_num(reduced);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reduced_id = arena.mul(&[coeff_id, arena.pi]);
        return eval_cos(arena, reduced_id);
    }

    None
}

/// Evaluate `tan(inner)` for known special values.
pub(super) fn eval_tan(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Odd function: tan(-x) = -tan(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        let tan_pos = eval_tan(arena, pos_inner).unwrap_or_else(|| arena.tan(pos_inner));
        return Some(arena.neg(tan_pos));
    }
    // tan(i·x) = i·tanh(x), the bridge `sin` and `cos` take (SymPy:
    // `tan(I*pi)` → `I*tanh(pi)`); up to 0.46 `tan(i·x)` stayed.
    if let Some(real_part) = as_pure_imaginary(arena, inner) {
        let t = eval_tanh(arena, real_part).unwrap_or_else(|| arena.tanh(real_part));
        return Some(arena.mul(&[arena.i_unit, t]));
    }

    let coeff = as_pi_multiple(arena, inner)?;
    if !is_classical_denominator(&coeff) {
        return tan_pi_radical(arena, &coeff)
            .or_else(|| first_quadrant(arena, &coeff, TrigKind::Tan));
    }

    let one_ratio: Q = Ratio::one();
    let coeff = mod_positive(&coeff, &one_ratio);

    // tan(0) = 0
    if coeff.is_zero() {
        return Some(arena.zero);
    }
    // tan(π/2) is a pole → ComplexInfinity.
    // The mod_positive reduction ensures 3π/2, 5π/2, etc. all map to coeff = 1/2.
    if coeff == Ratio::new(1.into(), 2.into()) {
        return Some(arena.complex_infinity);
    }
    // tan(π/4) = 1
    if coeff == Ratio::new(1.into(), 4.into()) {
        return Some(arena.one);
    }
    // tan(3π/4) = -1
    if coeff == Ratio::new(3.into(), 4.into()) {
        return Some(arena.neg_one);
    }
    // tan(π/6) = 1/√3 = √3/3
    if coeff == Ratio::new(1.into(), 6.into()) {
        let three = arena.int(3);
        let half_exp = arena.rational(1, 2);
        let sqrt3 = arena.pow(three, half_exp);
        let third = arena.rational(1, 3);
        return Some(arena.mul(&[third, sqrt3]));
    }
    // tan(π/3) = √3
    if coeff == Ratio::new(1.into(), 3.into()) {
        let three = arena.int(3);
        let half_exp = arena.rational(1, 2);
        return Some(arena.pow(three, half_exp));
    }

    // Q2 reduction: tan(π - x) = -tan(x) for coeff in (1/2, 1)
    let half = Ratio::new(BigInt::from(1), BigInt::from(2));
    if coeff > half && coeff < Ratio::one() {
        let reflected = Ratio::one() - &coeff;
        let nid = arena.intern_num(reflected);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reflected_angle = arena.mul(&[coeff_id, arena.pi()]);
        if let Some(val) = eval_tan(arena, reflected_angle) {
            return Some(arena.neg(val));
        }
    }

    None
}
