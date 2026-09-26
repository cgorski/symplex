//! Information theory on finite distributions, exactly: entropies,
//! divergences and distances between probability vectors, and the
//! information shared by the two margins of a joint table.
//!
//! Probability vectors are `&[Q]` (exact rationals summing to `1`), joint
//! distributions are `&[Vec<Q>]` tables (rows = values of the first
//! variable, columns = values of the second).  Every logarithmic quantity
//! is assembled as `Σ c_q · ln q` over the prime factors `q` of the
//! probabilities, so `H(½, ¼, ¼)` is exactly `3/2` bits and `ln 2`
//! never appears twice under different names; quantities that are not
//! rational functions of the probabilities come back as an [`Ex`] to be
//! evaluated when a float is wanted.  The conventions follow Cover &
//! Thomas, *Elements of Information Theory*, ch. 2, and
//! `scipy.stats.entropy` (natural logarithm unless `base` is given;
//! zero-probability terms dropped, `0·ln 0 = 0`).
//!
//! ```
//! use symplex::prelude::*;
//! use symplex::linprog::q;
//! use symplex::stats::information::{Base, entropy, kl_divergence, total_variation};
//!
//! let ctx = Context::new();
//! let p = [q(1, 2), q(1, 4), q(1, 4)];
//! let u = [q(1, 3), q(1, 3), q(1, 3)];
//! // scipy.stats.entropy([.5, .25, .25], base=2) = 1.5
//! assert_eq!(entropy(&ctx, &p, Base::Bits)?, ctx.rational(3, 2));
//! // scipy.stats.entropy(p, u) = 0.05889151782819178 (nats)
//! let kl = kl_divergence(&ctx, &p, &u, Base::Nats)?.eval_f64()?;
//! assert!((kl - 0.05889151782819178).abs() < 1e-12);
//! assert_eq!(total_variation(&p, &u)?, q(1, 6));
//! # Ok::<(), SymplexError>(())
//! ```
//!
//! # Conventions
//!
//! * [`js_divergence`] is the Jensen–Shannon *divergence*
//!   `½ KL(p‖m) + ½ KL(q‖m)`, `m = (p + q)/2`; SciPy's
//!   `scipy.spatial.distance.jensenshannon` returns its **square root**
//!   (the Jensen–Shannon distance).
//! * [`hellinger`] is `√(1 − BC(p, q))` with `BC = Σ √(pᵢ qᵢ)` the
//!   [`bhattacharyya_coefficient`]; `H² = ½ Σ (√pᵢ − √qᵢ)²`.
//! * [`conditional_entropy`] with [`Given::Row`] is `H(column | row)`:
//!   the entropy left in the column variable once the row is known.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
use crate::domains::ntheory::factorint_bounded;

use super::common::invalid;
use super::data::Q;
use super::family::{Distribution, sign_of};

/// The unit of an entropy: natural logarithms (nats, `scipy` default) or
/// base-2 logarithms (bits, shannons).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Base {
    /// Natural logarithm.
    Nats,
    /// Logarithm base 2.
    Bits,
}

/// Which margin of a joint table is known in a conditional entropy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Given {
    /// The row variable is known: `H(column | row)`.
    Row,
    /// The column variable is known: `H(row | column)`.
    Column,
}

/// The normalisation of a normalised mutual information
/// `I(X; Y) / norm(H(X), H(Y))` (`sklearn.metrics.normalized_mutual_info_score`'s
/// `average_method`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Norm {
    /// `(H(X) + H(Y)) / 2`.
    Arithmetic,
    /// `√(H(X) H(Y))`.
    Geometric,
    /// `min(H(X), H(Y))`.
    Min,
    /// `max(H(X), H(Y))`.
    Max,
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact sums of logarithms of rationals
// ═══════════════════════════════════════════════════════════════════════════

/// `Σ cⱼ ln rⱼ` with rational coefficients and positive rational
/// arguments, stored as `Σ c_q ln q` over the primes `q` dividing the
/// numerators and denominators (a large composite that resists factoring
/// is kept whole).  This is the canonical form in which
/// `½ ln 2 + ¼ ln 4 + ¼ ln 4` is `3/2 · ln 2`.
#[derive(Clone, Debug, Default)]
struct LogSum {
    terms: BTreeMap<BigInt, Q>,
}

impl LogSum {
    /// Add `c · ln r` for `r > 0`.
    fn add(&mut self, c: &Q, r: &Q) {
        self.add_int(c, r.numer());
        self.add_int(&(-c), r.denom());
    }

    /// Add `c · ln n` for an integer `n > 0`, split over its prime factors.
    fn add_int(&mut self, c: &Q, n: &BigInt) {
        if n.is_one() || c.is_zero() {
            return;
        }
        let (factors, cofactor) = factorint_bounded(n, 64);
        for (p, e) in factors {
            *self.terms.entry(p).or_insert_with(Q::zero) += c * Q::from_integer(BigInt::from(e));
        }
        if !cofactor.is_one() {
            *self.terms.entry(cofactor).or_insert_with(Q::zero) += c;
        }
    }

    fn scaled(mut self, k: &Q) -> Self {
        for v in self.terms.values_mut() {
            *v *= k;
        }
        self
    }

    fn plus(mut self, other: &LogSum) -> Self {
        for (base, coeff) in &other.terms {
            *self.terms.entry(base.clone()).or_insert_with(Q::zero) += coeff;
        }
        self
    }

    fn negated(self) -> Self {
        self.scaled(&-Q::one())
    }

    /// The non-zero terms.
    fn nonzero(&self) -> impl Iterator<Item = (&BigInt, &Q)> {
        self.terms.iter().filter(|(_, c)| !c.is_zero())
    }

    fn is_zero(&self) -> bool {
        self.nonzero().next().is_none()
    }

    /// `Some((q, c))` when the sum is a single term `c · ln q`.
    fn single(&self) -> Option<(&BigInt, &Q)> {
        let mut it = self.nonzero();
        let first = it.next()?;
        it.next().is_none().then_some(first)
    }

    /// The sum as an expression in nats: `Σ c_q · ln q`.
    fn to_nats(&self, ctx: &Context) -> Ex {
        let mut acc = ctx.zero();
        for (base, coeff) in self.nonzero() {
            acc += ctx.from_ratio(coeff.clone()) * ctx.from_bigint(base.clone()).ln();
        }
        acc
    }

    /// The sum in bits: `Σ c_q · log₂ q`, with `q = 2` folding to the
    /// rational `c₂` and every other prime kept as `c_q · ln q / ln 2`.
    fn to_bits(&self, ctx: &Context) -> Ex {
        let two = BigInt::from(2);
        let ln2 = ctx.int(2).ln();
        let mut acc = ctx.zero();
        for (base, coeff) in self.nonzero() {
            if *base == two {
                acc += ctx.from_ratio(coeff.clone());
            } else {
                acc += ctx.from_ratio(coeff.clone()) * ctx.from_bigint(base.clone()).ln() / &ln2;
            }
        }
        acc
    }

    fn in_base(&self, ctx: &Context, base: Base) -> Ex {
        match base {
            Base::Nats => self.to_nats(ctx),
            Base::Bits => self.to_bits(ctx),
        }
    }
}

/// `a ≤ b` for two constant log-sums: exactly when they coincide, else by
/// the sign of the exact difference `a − b` as [`sign_of`] decides it
/// (`evalf` to the precision the cancellation needs).  A difference that
/// evaluates to `0` without being known to be `0` counts as equal: either
/// side is then the same number to that precision.
///
/// 0.28 compared `Σ c_q ln q` summed in `f64`, whose terms cancel: the
/// entropy `7.0·10⁻²⁹` of `(1 − 10⁻³⁰, 10⁻³⁰)` is a sum of terms near
/// `±69`, so the comparison read rounding noise, and a prime cofactor
/// beyond `f64::MAX` made a term infinite.
fn log_sum_le(ctx: &Context, a: &LogSum, b: &LogSum) -> bool {
    let diff = a.clone().plus(&b.clone().negated());
    diff.is_zero() || sign_of(&diff.to_nats(ctx)) != Some(Ordering::Greater)
}

/// `num / den` for two log-sums: the exact rational `c_n / c_d` when both
/// are multiples of the same `ln q`, else the quotient of expressions.
fn ratio_of(ctx: &Context, num: &LogSum, den: &LogSum) -> Ex {
    if let (Some((qn, cn)), Some((qd, cd))) = (num.single(), den.single())
        && qn == qd
    {
        return ctx.from_ratio(cn / cd);
    }
    (num.to_nats(ctx) / den.to_nats(ctx)).simplify()
}

/// `num / √(a · b)` for log-sums: `c_n / √(c_a c_b)` when all three are
/// multiples of the same `ln q`, else the expression.
fn geometric_ratio(ctx: &Context, num: &LogSum, a: &LogSum, b: &LogSum) -> Ex {
    if let (Some((qn, cn)), Some((qa, ca)), Some((qb, cb))) = (num.single(), a.single(), b.single())
        && qn == qa
        && qa == qb
    {
        return (ctx.from_ratio(cn.clone()) / ctx.from_ratio(ca * cb).sqrt()).simplify();
    }
    (num.to_nats(ctx) / (a.to_nats(ctx) * b.to_nats(ctx)).sqrt()).simplify()
}

// ═══════════════════════════════════════════════════════════════════════════
// Validation
// ═══════════════════════════════════════════════════════════════════════════

fn check_vector(op: &'static str, p: &[Q], name: &str) -> Result<(), SymplexError> {
    if p.is_empty() {
        return Err(invalid(op, format!("{name} is empty")));
    }
    if let Some((i, v)) = p.iter().enumerate().find(|(_, v)| v.is_negative()) {
        return Err(invalid(op, format!("{name}[{i}] = {v} is negative")));
    }
    let total = p.iter().fold(Q::zero(), |acc, v| acc + v);
    if !total.is_one() {
        return Err(invalid(op, format!("{name} sums to {total}, not 1")));
    }
    Ok(())
}

fn check_pair(op: &'static str, p: &[Q], q: &[Q]) -> Result<(), SymplexError> {
    check_vector(op, p, "p")?;
    check_vector(op, q, "q")?;
    if p.len() != q.len() {
        return Err(invalid(
            op,
            format!(
                "p and q have different lengths ({} and {})",
                p.len(),
                q.len()
            ),
        ));
    }
    Ok(())
}

fn check_joint(op: &'static str, joint: &[Vec<Q>]) -> Result<(usize, usize), SymplexError> {
    let r = joint.len();
    if r == 0 {
        return Err(invalid(op, "the joint table is empty"));
    }
    let c = joint[0].len();
    if c == 0 {
        return Err(invalid(op, "the joint table has no columns"));
    }
    if let Some((i, row)) = joint.iter().enumerate().find(|(_, row)| row.len() != c) {
        return Err(invalid(
            op,
            format!("joint row {i} has {} entries, expected {c}", row.len()),
        ));
    }
    let mut total = Q::zero();
    for (i, row) in joint.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            if v.is_negative() {
                return Err(invalid(op, format!("joint[{i}][{j}] = {v} is negative")));
            }
            total += v;
        }
    }
    if !total.is_one() {
        return Err(invalid(
            op,
            format!("the joint table sums to {total}, not 1"),
        ));
    }
    Ok((r, c))
}

/// `H(p) = −Σ_{pᵢ>0} pᵢ ln pᵢ` as a log-sum, for a validated vector.
fn entropy_sum(p: &[Q]) -> LogSum {
    let mut acc = LogSum::default();
    for v in p.iter().filter(|v| v.is_positive()) {
        acc.add(&(-v), v);
    }
    acc
}

/// `D(p‖q) = Σ_{pᵢ>0} pᵢ ln(pᵢ/qᵢ)` as a log-sum, for validated vectors.
fn kl_sum(op: &'static str, p: &[Q], q: &[Q], what: &str) -> Result<LogSum, SymplexError> {
    let mut acc = LogSum::default();
    for (i, (pi, qi)) in p.iter().zip(q).enumerate() {
        if pi.is_zero() {
            continue;
        }
        if qi.is_zero() {
            return Err(invalid(
                op,
                format!("{what} is infinite: q[{i}] = 0 while p[{i}] = {pi} > 0"),
            ));
        }
        acc.add(pi, &(pi / qi));
    }
    Ok(acc)
}

// ═══════════════════════════════════════════════════════════════════════════
// Entropies of one distribution
// ═══════════════════════════════════════════════════════════════════════════

/// The Shannon entropy `H(p) = −Σ pᵢ log pᵢ` (zero terms dropped).
/// `scipy.stats.entropy(p)` / `entropy(p, base=2)`.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::information::{Base, entropy};
///
/// let ctx = Context::new();
/// // A fair coin has one bit of entropy; a certain outcome none.
/// assert_eq!(entropy(&ctx, &[q(1, 2), q(1, 2)], Base::Bits)?, ctx.one());
/// assert_eq!(entropy(&ctx, &[q(1, 1), q(0, 1)], Base::Nats)?, ctx.zero());
/// // H(1/3, 2/3) = ln 3 − (2/3) ln 2 nats
/// let h = entropy(&ctx, &[q(1, 3), q(2, 3)], Base::Nats)?;
/// assert_eq!(h, ctx.int(3).ln() - ctx.rational(2, 3) * ctx.int(2).ln());
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless `p` is non-empty, non-negative
/// and sums to exactly `1`.
pub fn entropy(ctx: &Context, p: &[Q], base: Base) -> Result<Ex, SymplexError> {
    check_vector("entropy", p, "p")?;
    Ok(entropy_sum(p).in_base(ctx, base))
}

/// The perplexity `exp H(p) = Π pᵢ^{−pᵢ}`: the size of a uniform
/// distribution with the same entropy (`4` for a fair four-sided die,
/// `2^{3/2}` for `(½, ¼, ¼)`).
///
/// # Errors
///
/// As [`entropy`].
pub fn perplexity(ctx: &Context, p: &[Q]) -> Result<Ex, SymplexError> {
    check_vector("perplexity", p, "p")?;
    let mut acc = ctx.one();
    for v in p.iter().filter(|v| v.is_positive()) {
        let e = ctx.from_ratio(v.clone());
        acc *= e.pow(&(-&e));
    }
    Ok(acc.simplify())
}

/// The probability vector of a discrete distribution with finitely many
/// values (a [`Finite`](super::Finite) table, a die, a binomial, …), in
/// the order of its support, for use with the functions of this module.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::stats::Distribution;
/// use symplex::stats::information::{Base, entropy, probability_vector};
///
/// let ctx = Context::new();
/// let die = probability_vector(&Distribution::die(ctx.int(6)))?;
/// // ln 6, in the prime-split form every entropy here is returned in
/// assert_eq!(entropy(&ctx, &die, Base::Nats)?, ctx.int(2).ln() + ctx.int(3).ln());
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if the distribution is continuous,
/// has infinitely many values, or a probability is not a rational number.
pub fn probability_vector(dist: &Distribution) -> Result<Vec<Q>, SymplexError> {
    const OP: &str = "probability_vector";
    let support = dist.support();
    if dist.is_continuous() {
        return Err(invalid(
            OP,
            format!(
                "{}: a probability vector needs a discrete distribution",
                dist.name()
            ),
        ));
    }
    let values: Vec<Ex> = if let Some(points) = support.as_points() {
        points
    } else if let Some(iv) = support.as_interval() {
        // `hi − lo` checked: a support spanning more than `i64::MAX` (0.28
        // subtracted and overflowed, a panic) is not a finite table either.
        match (iv.lower.eval().as_i64(), iv.upper.eval().as_i64()) {
            (Some(lo), Some(hi))
                if hi
                    .checked_sub(lo)
                    .is_some_and(|span| (0..1_000_000).contains(&span)) =>
            {
                let ctx = dist.context();
                (lo..=hi).map(|v| ctx.int(v)).collect()
            }
            _ => {
                return Err(invalid(
                    OP,
                    format!(
                        "{}: the support is not a finite range of integers",
                        dist.name()
                    ),
                ));
            }
        }
    } else {
        return Err(invalid(
            OP,
            format!("{}: the support is not a finite set of values", dist.name()),
        ));
    };
    values
        .iter()
        .map(|v| {
            dist.density(v).eval().as_rational().ok_or_else(|| {
                invalid(
                    OP,
                    format!(
                        "{}: the probability of `{v}` is not a rational number",
                        dist.name()
                    ),
                )
            })
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Divergences and distances between two distributions
// ═══════════════════════════════════════════════════════════════════════════

/// The Kullback–Leibler divergence (relative entropy)
/// `D(p‖q) = Σ pᵢ log(pᵢ/qᵢ)` over `pᵢ > 0`.  `scipy.stats.entropy(p, q)`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for invalid or mismatched vectors,
/// or if some `qᵢ = 0 < pᵢ` (the divergence is infinite).
pub fn kl_divergence(ctx: &Context, p: &[Q], q: &[Q], base: Base) -> Result<Ex, SymplexError> {
    const OP: &str = "kl_divergence";
    check_pair(OP, p, q)?;
    Ok(kl_sum(OP, p, q, "KL divergence")?.in_base(ctx, base))
}

/// The cross-entropy `H(p, q) = −Σ pᵢ log qᵢ = H(p) + D(p‖q)`.
///
/// # Errors
///
/// As [`kl_divergence`].
pub fn cross_entropy(ctx: &Context, p: &[Q], q: &[Q], base: Base) -> Result<Ex, SymplexError> {
    const OP: &str = "cross_entropy";
    check_pair(OP, p, q)?;
    let mut acc = LogSum::default();
    for (i, (pi, qi)) in p.iter().zip(q).enumerate() {
        if pi.is_zero() {
            continue;
        }
        if qi.is_zero() {
            return Err(invalid(
                OP,
                format!("cross-entropy is infinite: q[{i}] = 0 while p[{i}] = {pi} > 0"),
            ));
        }
        acc.add(&(-pi), qi);
    }
    Ok(acc.in_base(ctx, base))
}

/// The Jensen–Shannon divergence `½ D(p‖m) + ½ D(q‖m)` with
/// `m = (p + q)/2`: symmetric, finite, and at most `ln 2` (one bit).
/// `scipy.spatial.distance.jensenshannon(p, q)**2` — SciPy returns the
/// square root (the Jensen–Shannon *distance*).
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for invalid or mismatched vectors.
pub fn js_divergence(ctx: &Context, p: &[Q], q: &[Q], base: Base) -> Result<Ex, SymplexError> {
    const OP: &str = "js_divergence";
    check_pair(OP, p, q)?;
    let two = Q::from_integer(2.into());
    let half = Q::one() / &two;
    let m: Vec<Q> = p.iter().zip(q).map(|(a, b)| (a + b) / &two).collect();
    let js = kl_sum(OP, p, &m, "JS divergence")?
        .plus(&kl_sum(OP, q, &m, "JS divergence")?)
        .scaled(&half);
    Ok(js.in_base(ctx, base))
}

/// The total variation distance `½ Σ |pᵢ − qᵢ|`, exactly.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for invalid or mismatched vectors.
pub fn total_variation(p: &[Q], q: &[Q]) -> Result<Q, SymplexError> {
    check_pair("total_variation", p, q)?;
    let sum = p
        .iter()
        .zip(q)
        .fold(Q::zero(), |acc, (a, b)| acc + (a - b).abs());
    Ok(sum / Q::from_integer(2.into()))
}

/// The Bhattacharyya coefficient `BC(p, q) = Σ √(pᵢ qᵢ) ∈ [0, 1]`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] for invalid or mismatched vectors.
pub fn bhattacharyya_coefficient(ctx: &Context, p: &[Q], q: &[Q]) -> Result<Ex, SymplexError> {
    check_pair("bhattacharyya_coefficient", p, q)?;
    let mut acc = ctx.zero();
    for (a, b) in p.iter().zip(q) {
        if a.is_positive() && b.is_positive() {
            acc += ctx.from_ratio(a * b).sqrt();
        }
    }
    Ok(acc.simplify())
}

/// The Bhattacharyya distance `−ln BC(p, q)`.
///
/// # Errors
///
/// As [`bhattacharyya_coefficient`], plus [`SymplexError::InvalidArgument`]
/// if the supports are disjoint (`BC = 0`, infinite distance).
pub fn bhattacharyya_distance(ctx: &Context, p: &[Q], q: &[Q]) -> Result<Ex, SymplexError> {
    let bc = bhattacharyya_coefficient(ctx, p, q)?;
    if bc.is_zero() == Some(true) {
        return Err(invalid(
            "bhattacharyya_distance",
            "Bhattacharyya distance is infinite: p and q have disjoint supports",
        ));
    }
    Ok((-bc.ln()).simplify())
}

/// The Hellinger distance `H(p, q) = √(1 − BC(p, q)) = √(½ Σ (√pᵢ − √qᵢ)²)`,
/// in `[0, 1]`.
///
/// # Errors
///
/// As [`bhattacharyya_coefficient`].
pub fn hellinger(ctx: &Context, p: &[Q], q: &[Q]) -> Result<Ex, SymplexError> {
    let bc = bhattacharyya_coefficient(ctx, p, q)?;
    Ok((ctx.one() - bc).sqrt().simplify())
}

// ═══════════════════════════════════════════════════════════════════════════
// Joint tables
// ═══════════════════════════════════════════════════════════════════════════

/// A joint probability table from a contingency table of counts:
/// `pᵢⱼ = nᵢⱼ / Σ n`, the total summed exactly (0.28 summed in `usize`,
/// which panicked — or, unchecked, wrapped to an "all zeros" error — once
/// the counts passed `usize::MAX`).
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::information::joint_from_counts;
///
/// let j = joint_from_counts(&[vec![usize::MAX, 1]])?;
/// // Fraction(2**64 - 1, 2**64), Fraction(1, 2**64)
/// assert_eq!(j[0][1], "1/18446744073709551616".parse().unwrap());
/// assert_eq!(&j[0][0] + &j[0][1], q(1, 1));
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] if the table is empty or jagged, or
/// every count is zero.
pub fn joint_from_counts(counts: &[Vec<usize>]) -> Result<Vec<Vec<Q>>, SymplexError> {
    const OP: &str = "joint_from_counts";
    if counts.is_empty() || counts[0].is_empty() {
        return Err(invalid(OP, "the count table is empty"));
    }
    let c = counts[0].len();
    if let Some((i, row)) = counts.iter().enumerate().find(|(_, row)| row.len() != c) {
        return Err(invalid(
            OP,
            format!("count row {i} has {} entries, expected {c}", row.len()),
        ));
    }
    let total = counts
        .iter()
        .flatten()
        .fold(BigInt::zero(), |acc, &n| acc + BigInt::from(n));
    if total.is_zero() {
        return Err(invalid(OP, "the count table is all zeros"));
    }
    let total = Q::from_integer(total);
    Ok(counts
        .iter()
        .map(|row| {
            row.iter()
                .map(|&n| Q::from_integer(n.into()) / &total)
                .collect()
        })
        .collect())
}

/// The two marginal distributions of a joint table `pᵢⱼ`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Marginals {
    /// The row sums `(Σⱼ pᵢⱼ)ᵢ`: the distribution of the row variable.
    pub rows: Vec<Q>,
    /// The column sums `(Σᵢ pᵢⱼ)ⱼ`: the distribution of the column
    /// variable.
    pub cols: Vec<Q>,
}

/// The row and column marginals of a joint table.
///
/// ```
/// use symplex::linprog::q;
/// use symplex::stats::information::marginals;
///
/// let joint = vec![vec![q(1, 2), q(1, 4)], vec![q(0, 1), q(1, 4)]];
/// let m = marginals(&joint)?;
/// assert_eq!(m.rows, vec![q(3, 4), q(1, 4)]);
/// assert_eq!(m.cols, vec![q(1, 2), q(1, 2)]);
/// # Ok::<(), symplex::prelude::SymplexError>(())
/// ```
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`] unless the table is rectangular,
/// non-negative and sums to `1`.
pub fn marginals(joint: &[Vec<Q>]) -> Result<Marginals, SymplexError> {
    marginals_of("marginals", joint)
}

/// [`marginals`] labelled with the calling function.
fn marginals_of(op: &'static str, joint: &[Vec<Q>]) -> Result<Marginals, SymplexError> {
    let (_, c) = check_joint(op, joint)?;
    let rows = joint
        .iter()
        .map(|row| row.iter().fold(Q::zero(), |acc, v| acc + v))
        .collect();
    let cols = (0..c)
        .map(|j| joint.iter().fold(Q::zero(), |acc, row| acc + &row[j]))
        .collect();
    Ok(Marginals { rows, cols })
}

fn flatten(joint: &[Vec<Q>]) -> Vec<Q> {
    joint.iter().flatten().cloned().collect()
}

/// The joint entropy `H(X, Y) = −Σᵢⱼ pᵢⱼ log pᵢⱼ`.
///
/// # Errors
///
/// As [`marginals`].
pub fn joint_entropy(ctx: &Context, joint: &[Vec<Q>], base: Base) -> Result<Ex, SymplexError> {
    check_joint("joint_entropy", joint)?;
    Ok(entropy_sum(&flatten(joint)).in_base(ctx, base))
}

/// `I(X; Y) = Σ pᵢⱼ ln(pᵢⱼ / (pᵢ· p·ⱼ))` as a log-sum, with the marginals.
fn mutual_information_sum(
    op: &'static str,
    joint: &[Vec<Q>],
) -> Result<(LogSum, Marginals), SymplexError> {
    let m = marginals_of(op, joint)?;
    let mut acc = LogSum::default();
    for (i, row) in joint.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            if v.is_positive() {
                acc.add(v, &(v / (&m.rows[i] * &m.cols[j])));
            }
        }
    }
    Ok((acc, m))
}

/// The mutual information
/// `I(X; Y) = Σᵢⱼ pᵢⱼ log(pᵢⱼ / (pᵢ· p·ⱼ)) = H(X) + H(Y) − H(X, Y)` of the
/// two margins of a joint table.
///
/// ```
/// use symplex::prelude::*;
/// use symplex::linprog::q;
/// use symplex::stats::information::{Base, mutual_information};
///
/// let ctx = Context::new();
/// // Cover & Thomas, Example 2.2.1: I(X; Y) = 3/8 bit
/// let joint = vec![
///     vec![q(1, 8), q(1, 16), q(1, 32), q(1, 32)],
///     vec![q(1, 16), q(1, 8), q(1, 32), q(1, 32)],
///     vec![q(1, 16), q(1, 16), q(1, 16), q(1, 16)],
///     vec![q(1, 4), q(0, 1), q(0, 1), q(0, 1)],
/// ];
/// assert_eq!(mutual_information(&ctx, &joint, Base::Bits)?, ctx.rational(3, 8));
/// # Ok::<(), SymplexError>(())
/// ```
///
/// # Errors
///
/// As [`marginals`].
pub fn mutual_information(ctx: &Context, joint: &[Vec<Q>], base: Base) -> Result<Ex, SymplexError> {
    let (i, _) = mutual_information_sum("mutual_information", joint)?;
    Ok(i.in_base(ctx, base))
}

/// The information gain of the column variable about the row variable,
/// `H(row) − H(row | column)` — which is the [`mutual_information`]
/// (decision-tree "information gain" with rows = class labels, columns =
/// attribute values).
///
/// # Errors
///
/// As [`marginals`].
pub fn information_gain(ctx: &Context, joint: &[Vec<Q>], base: Base) -> Result<Ex, SymplexError> {
    let (i, _) = mutual_information_sum("information_gain", joint)?;
    Ok(i.in_base(ctx, base))
}

/// The conditional entropy `H(Y | X) = H(X, Y) − H(X)`: with
/// [`Given::Row`] the row variable is known and the entropy of the column
/// variable remains; with [`Given::Column`] the reverse.
///
/// # Errors
///
/// As [`marginals`].
pub fn conditional_entropy(
    ctx: &Context,
    joint: &[Vec<Q>],
    given: Given,
    base: Base,
) -> Result<Ex, SymplexError> {
    let Marginals { rows, cols } = marginals_of("conditional_entropy", joint)?;
    let known = match given {
        Given::Row => rows,
        Given::Column => cols,
    };
    let h = entropy_sum(&flatten(joint)).plus(&entropy_sum(&known).negated());
    Ok(h.in_base(ctx, base))
}

/// The normalised mutual information `I(X; Y) / norm(H(X), H(Y))` in
/// `[0, 1]`, independent of the logarithm base.
/// `sklearn.metrics.normalized_mutual_info_score(average_method=…)`.
///
/// # Errors
///
/// As [`marginals`], plus [`SymplexError::InvalidArgument`] if the
/// normaliser is zero (a margin with a single value under `Min`, or both
/// margins degenerate).
pub fn normalized_mutual_information(
    ctx: &Context,
    joint: &[Vec<Q>],
    norm: Norm,
) -> Result<Ex, SymplexError> {
    const OP: &str = "normalized_mutual_information";
    let (i, m) = mutual_information_sum(OP, joint)?;
    let hx = entropy_sum(&m.rows);
    let hy = entropy_sum(&m.cols);
    let degenerate = match norm {
        Norm::Min => hx.is_zero() || hy.is_zero(),
        Norm::Arithmetic | Norm::Geometric | Norm::Max => hx.is_zero() && hy.is_zero(),
    };
    if degenerate {
        return Err(invalid(
            OP,
            "normalised mutual information is undefined: the normalising entropy is zero",
        ));
    }
    let half = Q::one() / Q::from_integer(2.into());
    Ok(match norm {
        Norm::Arithmetic => ratio_of(ctx, &i, &hx.clone().plus(&hy).scaled(&half)),
        Norm::Geometric => geometric_ratio(ctx, &i, &hx, &hy),
        Norm::Min => {
            if log_sum_le(ctx, &hx, &hy) {
                ratio_of(ctx, &i, &hx)
            } else {
                ratio_of(ctx, &i, &hy)
            }
        }
        Norm::Max => {
            if log_sum_le(ctx, &hy, &hx) {
                ratio_of(ctx, &i, &hx)
            } else {
                ratio_of(ctx, &i, &hy)
            }
        }
    })
}
