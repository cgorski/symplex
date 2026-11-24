//! Exact zero test in a tower of real quadratic extensions (part of [`eval`](super)).

use super::*;

// ── Exact zero test in a tower of real quadratic extensions ─────────────────
//
// The values of `sin`, `cos`, `tan` at the denominators of
// [`COS_PI_DENOMINATORS`] are square-root towers (`√(√5/8 + 5/8)`, `1/(√6/4
// − √2/4)`), and an identity between them (`sin²x + cos²x = 1` at `x =
// 7π/60`) is a zero that `minimal_polynomial` finds only through
// resultants of degree up to 256 (23 s in a debug build).  Here the
// constant is computed exactly in `ℚ(t₁, …, tₖ)`, `tᵢ² = rᵢ` with `rᵢ` in
// the tower below: an element has `2ᵏ` rational coordinates (the
// monomials in the `tᵢ`), a product costs about `4ᵏ` rational products.
// Sending each `tᵢ` to the positive root `√rᵢ` is a ring homomorphism to
// ℝ, so an element computed as 0 has the value 0, whatever relations the
// generators satisfy.  Relations are what make the zero visible
// (`√(1/2 − √2/4) = (√2 − 1)·√(1/2 + √2/4)`), so a new radicand is first
// tried as a square in the tower ([`SqrtTower::sqrt`], every root verified
// by squaring it) and its root taken with the sign of its certified value;
// only otherwise is it a new generator.

/// The most generators a [`SqrtTower`] takes.
const SQRT_TOWER_MAX_GENERATORS: usize = 7;

/// A tower `ℚ(t₁, …, tₖ)` of quadratic extensions (see the section
/// comment).  An element at level `k` is a vector of `2ᵏ` coordinates: the
/// first half the part without `tₖ`, the second half the coefficient of
/// `tₖ`, recursively.
struct SqrtTower {
    /// `rᵢ = tᵢ²`, an element of level `i`.
    radicands: Vec<Vec<Q>>,
    /// The real radical `√b` each generator stands for.
    roots: Vec<ExprId>,
}

fn tower_add(x: &[Q], y: &[Q]) -> Vec<Q> {
    x.iter().zip(y).map(|(a, b)| a + b).collect()
}

fn tower_sub(x: &[Q], y: &[Q]) -> Vec<Q> {
    x.iter().zip(y).map(|(a, b)| a - b).collect()
}

fn tower_is_zero(x: &[Q]) -> bool {
    x.iter().all(Zero::is_zero)
}

/// `x` as an element of level `k ≥` its own (zero coordinates for the new
/// monomials).
fn tower_pad(x: &[Q], k: usize) -> Vec<Q> {
    let mut v = x.to_vec();
    v.resize(1 << k, Q::zero());
    v
}

/// The square root of a non-negative rational, when rational.
fn rational_sqrt(q: &Q) -> Option<Q> {
    if q.is_negative() {
        return None;
    }
    let (n, d) = (q.numer().sqrt(), q.denom().sqrt());
    (&n * &n == *q.numer() && &d * &d == *q.denom()).then(|| Ratio::new(n, d))
}

impl SqrtTower {
    fn level(&self) -> usize {
        self.radicands.len()
    }

    /// `x·y` at level `k`: `(x₀ + x₁t)(y₀ + y₁t) = x₀y₀ + x₁y₁·r +
    /// ((x₀ + x₁)(y₀ + y₁) − x₀y₀ − x₁y₁)·t`.  The recursion is on the
    /// level (at most [`SQRT_TOWER_MAX_GENERATORS`] deep).
    fn mul(&self, x: &[Q], y: &[Q], k: usize) -> Vec<Q> {
        if k == 0 {
            return vec![&x[0] * &y[0]];
        }
        if tower_is_zero(x) || tower_is_zero(y) {
            return vec![Q::zero(); 1 << k];
        }
        let h = 1 << (k - 1);
        let (x0, x1) = x.split_at(h);
        let (y0, y1) = y.split_at(h);
        let p = self.mul(x0, y0, k - 1);
        let q = self.mul(x1, y1, k - 1);
        let s = self.mul(&tower_add(x0, x1), &tower_add(y0, y1), k - 1);
        let qr = self.mul(&q, &self.radicands[k - 1], k - 1);
        let mut out = tower_add(&p, &qr);
        out.extend(tower_sub(&tower_sub(&s, &p), &q));
        out
    }

    /// `1/x` at level `k`, by `1/(x₀ + x₁t) = (x₀ − x₁t)/(x₀² − x₁²r)`;
    /// `None` when a norm on the way is 0.
    fn inverse(&self, x: &[Q], k: usize) -> Option<Vec<Q>> {
        if k == 0 {
            return (!x[0].is_zero()).then(|| vec![x[0].recip()]);
        }
        let h = 1 << (k - 1);
        let (x0, x1) = x.split_at(h);
        let x1_sq = self.mul(x1, x1, k - 1);
        let norm = tower_sub(
            &self.mul(x0, x0, k - 1),
            &self.mul(&x1_sq, &self.radicands[k - 1], k - 1),
        );
        let inv = self.inverse(&norm, k - 1)?;
        let mut out = self.mul(x0, &inv, k - 1);
        out.extend(self.mul(x1, &inv, k - 1).into_iter().map(|c| -c));
        Some(out)
    }

    /// `xⁿ` at level `k` (`|n| ≤ 64`).
    fn pow(&self, x: &[Q], n: i64, k: usize) -> Option<Vec<Q>> {
        if n.unsigned_abs() > 64 {
            return None;
        }
        let mut base = if n < 0 {
            self.inverse(x, k)?
        } else {
            x.to_vec()
        };
        let mut out = tower_pad(&[Q::one()], k);
        let mut e = n.unsigned_abs();
        while e > 0 {
            if e & 1 == 1 {
                out = self.mul(&out, &base, k);
            }
            e >>= 1;
            if e > 0 {
                base = self.mul(&base, &base, k);
            }
        }
        Some(out)
    }

    /// A square root of `x` in the tower of level `k`, when it has one
    /// there; every candidate is verified by squaring.  For `x = a + b·t`:
    /// `√a` or `√(a/r)·t` when `b = 0`, else `c + d·t` with `c² = (a ±
    /// √(a² − b²r))/2`, `d = b/(2c)`.
    fn sqrt(&self, x: &[Q], k: usize) -> Option<Vec<Q>> {
        if tower_is_zero(x) {
            return Some(vec![Q::zero(); 1 << k]);
        }
        if k == 0 {
            return rational_sqrt(&x[0]).map(|r| vec![r]);
        }
        let h = 1 << (k - 1);
        let (a, b) = x.split_at(h);
        let zero = vec![Q::zero(); h];
        let mut candidates: Vec<Vec<Q>> = Vec::new();
        if tower_is_zero(b) {
            if let Some(c) = self.sqrt(a, k - 1) {
                candidates.push([c, zero.clone()].concat());
            } else if let Some(r_inv) = self.inverse(&self.radicands[k - 1], k - 1)
                && let Some(d) = self.sqrt(&self.mul(a, &r_inv, k - 1), k - 1)
            {
                candidates.push([zero.clone(), d].concat());
            }
        } else {
            let b_sq = self.mul(b, b, k - 1);
            let norm = tower_sub(
                &self.mul(a, a, k - 1),
                &self.mul(&b_sq, &self.radicands[k - 1], k - 1),
            );
            let s = self.sqrt(&norm, k - 1)?;
            let half = tower_pad(&[Ratio::new(BigInt::from(1), BigInt::from(2))], k - 1);
            for c_sq in [tower_add(a, &s), tower_sub(a, &s)] {
                let c_sq = self.mul(&c_sq, &half, k - 1);
                let Some(c) = self.sqrt(&c_sq, k - 1) else {
                    continue;
                };
                let two_c = tower_add(&c, &c);
                let Some(inv) = self.inverse(&two_c, k - 1) else {
                    continue;
                };
                let d = self.mul(b, &inv, k - 1);
                candidates.push([c, d].concat());
            }
        }
        candidates.into_iter().find(|c| self.mul(c, c, k) == x)
    }

    /// The arena constant of the element `x` of level `k`.
    fn to_expr(&self, arena: &mut Arena, x: &[Q], k: usize) -> ExprId {
        let mut terms: smallvec::SmallVec<[ExprId; 8]> = smallvec::SmallVec::new();
        for (m, c) in x.iter().enumerate().take(1 << k) {
            if c.is_zero() {
                continue;
            }
            let mut factors: smallvec::SmallVec<[ExprId; 8]> =
                smallvec::smallvec![arena.num_ratio(c.clone())];
            for (i, &root) in self.roots.iter().enumerate() {
                if m & (1 << i) != 0 {
                    factors.push(root);
                }
            }
            terms.push(arena.mul(&factors));
        }
        arena.add(&terms)
    }
}

/// Is the constant `d` (rationals, `φ`, `+ − ×`, integer powers and square
/// roots of positive values) exactly 0 by [`SqrtTower`] arithmetic?  `false`
/// means not shown (anything else in `d`, a negative or undecided radicand,
/// more than [`SQRT_TOWER_MAX_GENERATORS`] independent radicals, more than
/// 400 nodes, a zero divisor).
pub(super) fn sqrt_tower_is_zero(arena: &mut Arena, d: ExprId) -> bool {
    let order = walk::post_order_ids(arena, d);
    if order.len() > 400
        || !order.iter().any(|&id| {
            matches!(arena.node(id), ExprNode::Pow(_, e)
                if arena.as_num(*e).is_some_and(|q| *q.denom() == BigInt::from(2)))
        })
    {
        return false;
    }
    let mut tower = SqrtTower {
        radicands: Vec::new(),
        roots: Vec::new(),
    };
    let mut values: FxHashMap<ExprId, Vec<Q>> = FxHashMap::default();
    // `√b` as an element, by the arena node `b`.
    let mut sqrt_of: FxHashMap<ExprId, Vec<Q>> = FxHashMap::default();
    let half = arena.rational(1, 2);
    for &id in &order {
        let node = arena.node(id).clone();
        let value = match node {
            ExprNode::Num(n) => tower_pad(&[arena.num(n).clone()], tower.level()),
            ExprNode::Neg(a) => {
                let Some(v) = values.get(&a) else {
                    return false;
                };
                tower_pad(v, tower.level())
                    .into_iter()
                    .map(|c| -c)
                    .collect()
            }
            ExprNode::Add(cs) => {
                let k = tower.level();
                let mut acc = tower_pad(&[], k);
                for c in cs.iter() {
                    let Some(v) = values.get(c) else {
                        return false;
                    };
                    acc = tower_add(&acc, &tower_pad(v, k));
                }
                acc
            }
            ExprNode::Mul(cs) => {
                let k = tower.level();
                let mut acc = tower_pad(&[Q::one()], k);
                for c in cs.iter() {
                    let Some(v) = values.get(c) else {
                        return false;
                    };
                    acc = tower.mul(&acc, &tower_pad(v, k), k);
                }
                acc
            }
            ExprNode::GoldenRatio | ExprNode::Pow(..) => {
                // φ = 1/2 + √5/2; b^(p/q) for q ∈ {1, 2}.
                let (b, p, root) = match node {
                    ExprNode::Pow(b, e) => {
                        let Some(q) = arena.as_num(e) else {
                            return false;
                        };
                        let Some(p) = q.numer().to_i64() else {
                            return false;
                        };
                        match q.denom().to_i64() {
                            Some(1) => (b, p, false),
                            Some(2) => (b, p, true),
                            _ => return false,
                        }
                    }
                    _ => (arena.int(5), 1, true),
                };
                let base = if root {
                    if let Some(r) = sqrt_of.get(&b) {
                        r.clone()
                    } else {
                        let k = tower.level();
                        let radicand = match values.get(&b) {
                            Some(v) => tower_pad(v, k),
                            None => match arena.as_num(b) {
                                Some(q) => tower_pad(std::slice::from_ref(q), k),
                                None => return false,
                            },
                        };
                        let r = if tower_is_zero(&radicand) {
                            radicand
                        } else {
                            // Tower elements are real (every radicand so far
                            // was certified positive).
                            if certified_real_sign(arena, b) != Some(std::cmp::Ordering::Greater) {
                                return false;
                            }
                            match tower.sqrt(&radicand, k) {
                                Some(s) => {
                                    let e = tower.to_expr(arena, &s, k);
                                    match certified_real_sign(arena, e) {
                                        Some(std::cmp::Ordering::Greater) => s,
                                        Some(std::cmp::Ordering::Less) => {
                                            s.into_iter().map(|c| -c).collect()
                                        }
                                        _ => return false,
                                    }
                                }
                                None if k < SQRT_TOWER_MAX_GENERATORS => {
                                    tower.radicands.push(radicand);
                                    let root_node = arena.pow(b, half);
                                    tower.roots.push(root_node);
                                    let mut t = tower_pad(&[], k + 1);
                                    t[1 << k] = Q::one();
                                    t
                                }
                                None => return false,
                            }
                        };
                        sqrt_of.insert(b, r.clone());
                        r
                    }
                } else {
                    match values.get(&b) {
                        Some(v) => v.clone(),
                        None => return false,
                    }
                };
                let k = tower.level();
                let base = tower_pad(&base, k);
                let powered = match tower.pow(&base, p, k) {
                    Some(v) => v,
                    None => return false,
                };
                if matches!(node, ExprNode::GoldenRatio) {
                    let h = tower_pad(&[Ratio::new(BigInt::from(1), BigInt::from(2))], k);
                    tower.mul(&tower_add(&powered, &tower_pad(&[Q::one()], k)), &h, k)
                } else {
                    powered
                }
            }
            _ => return false,
        };
        values.insert(id, value);
    }
    values.get(&d).is_some_and(|v| tower_is_zero(v))
}

/// LogGamma(n) for positive integer n → ln((n-1)!)
pub(super) fn eval_log_gamma(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?.clone();
    if !r.is_integer() || !r.is_positive() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
    // (n-1)!, within the digit guard (beyond it `ln Γ(n)` stays symbolic).
    if beyond_digit_guard(arena, log10_factorial_lower((n - 1) as f64)) {
        return None;
    }
    let fact = Ratio::from_integer(crate::base::combinatorics::factorial(n - 1));
    // ln(1) = 0 — handle n=1 and n=2 where (n-1)! = 1
    if fact == Ratio::<BigInt>::one() {
        return Some(arena.zero);
    }
    let fact_id = guarded_num(arena, fact)?;
    Some(arena.ln(fact_id))
}

/// erf(0) → 0
pub(super) fn eval_erf(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // erf(0) = 0
    if let Some(r) = arena.as_num(inner)
        && r.is_zero()
    {
        return Some(arena.zero);
    }

    // erf(∞) = 1
    if matches!(arena.node(inner), ExprNode::Infinity) {
        return Some(arena.one);
    }

    // erf(-∞) = -1
    if matches!(arena.node(inner), ExprNode::NegInfinity) {
        return Some(arena.neg_one);
    }

    // Odd function: erf(-x) = -erf(x)
    if let ExprNode::Neg(pos_inner) = arena.node(inner).clone() {
        let erf_pos = arena.erf(pos_inner);
        return Some(arena.neg(erf_pos));
    }
    // A constant's sign is extracted (SymPy: `erf(-2)` → `-erf(2)`).
    if let Some(pos) = negated_constant(arena, inner) {
        let erf_pos = eval_erf(arena, pos).unwrap_or_else(|| arena.erf(pos));
        return Some(arena.neg(erf_pos));
    }

    None
}

/// `c` for a symbol-free argument `−c` with a negative rational coefficient
/// ([`as_negated`]): the odd functions without a sign rule for symbols
/// (`erf`, `asin`, `atan`, `asinh`, `atanh`) extract it from a constant, as
/// SymPy does (`atanh(-1/2)` → `-atanh(1/2)`).  The identities hold on the
/// principal branches, cuts included.
pub(super) fn negated_constant(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if !walk::free_symbols(arena, inner).is_empty() {
        return None;
    }
    as_negated(arena, inner)
}

/// erfc(0) → 1
pub(super) fn eval_lambertw(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // W(0) = 0
    if let Some(r) = arena.as_num(inner)
        && r.is_zero()
    {
        tracing::debug!("eval: LambertW(0) = 0");
        return Some(arena.zero);
    }

    // W(e) = 1
    if inner == arena.e_const {
        tracing::debug!("eval: LambertW(e) = 1");
        return Some(arena.one);
    }

    // W(∞) = ∞ (W(−∞), W(±i·∞), W(nan) fold when built)
    if ValueClass::special_atom(arena, inner) == Some(ValueClass::PosInf) {
        tracing::debug!("eval: LambertW(∞) = ∞");
        return Some(arena.infinity);
    }

    // W(-1/e) = -1
    {
        let e = arena.e_const;
        let neg1 = arena.neg_one;
        let neg_inv_e = arena.div(neg1, e);
        if inner == neg_inv_e {
            tracing::debug!("eval: LambertW(-1/e) = -1");
            return Some(arena.neg_one);
        }
    }

    // W(-ln(2)/2) = -ln(2)
    {
        let two = arena.int(2);
        let ln2 = arena.ln(two);
        let neg_ln2 = arena.neg(ln2);
        let target = arena.div(neg_ln2, two);
        if inner == target {
            tracing::debug!("eval: LambertW(-ln(2)/2) = -ln(2)");
            return Some(neg_ln2);
        }
    }

    // W(2*ln(2)) = ln(2)
    {
        let two = arena.int(2);
        let ln2 = arena.ln(two);
        let target = arena.mul(&[two, ln2]);
        if inner == target {
            tracing::debug!("eval: LambertW(2*ln(2)) = ln(2)");
            return Some(ln2);
        }
    }

    // W(-π/2) = iπ/2
    {
        let pi = arena.pi;
        let two = arena.int(2);
        let neg_pi = arena.neg(pi);
        let target = arena.div(neg_pi, two);
        if inner == target {
            tracing::debug!("eval: LambertW(-π/2) = iπ/2");
            let i = arena.i_unit;
            let pi_over_2 = arena.div(pi, two);
            return Some(arena.mul(&[i, pi_over_2]));
        }
    }

    // W(e^(1+e)) = e
    {
        let e = arena.e_const;
        let one = arena.one;
        let one_plus_e = arena.add(&[one, e]);
        let target_exp = arena.exp(one_plus_e);
        if inner == target_exp {
            tracing::debug!("eval: LambertW(e^(1+e)) = e");
            return Some(e);
        }
        let target_pow = arena.pow(e, one_plus_e);
        if inner == target_pow {
            tracing::debug!("eval: LambertW(e^(1+e)) = e");
            return Some(e);
        }
    }

    // W(w·e^w) = w for the real w ≥ −1 the principal branch reaches.
    if let Some(w) = lambert_fixed_point(arena, inner, LambertRealBranch::Principal) {
        tracing::debug!("eval: LambertW(w*exp(w)) = w");
        return Some(w);
    }

    None
}

/// `v` (an imaginary part) with every `im(W_k(a))` that is 0 replaced by 0:
/// `W₀(a)` is real for a real constant `a ≥ −1/e`, `W₋₁(a)` for `−1/e ≤ a <
/// 0` (certified signs of `a + 1/e` and `a`).  Up to 0.45 these stayed, and
/// the realness of `W₋₁(−ln(2)/(5·⁵√2)) + ln(2)/5` was not known, so its
/// sign was not decided though its difference from 0 was.
pub(super) fn zero_real_lambert_parts(arena: &mut Arena, v: ExprId) -> ExprId {
    let targets: Vec<ExprId> = walk::post_order_ids(arena, v)
        .into_iter()
        .filter(|&id| matches!(*arena.node(id), ExprNode::Im(w) if lambert_real_argument(arena, w).is_some()))
        .collect();
    let mut out = v;
    for t in targets {
        let ExprNode::Im(w) = *arena.node(t) else {
            continue;
        };
        if lambert_value_is_real(arena, w) {
            let zero = arena.zero;
            out = crate::transforms::subs::subs(arena, out, t, zero);
        }
    }
    out
}

/// `(a, branch)` for `w = W₀(a)` or `W₋₁(a)` with a symbol-free `a`.
fn lambert_real_argument(arena: &Arena, w: ExprId) -> Option<(ExprId, LambertRealBranch)> {
    match arena.node(w) {
        ExprNode::LambertW(a) => Some((*a, LambertRealBranch::Principal)),
        ExprNode::Lib(LibFn::LambertW, args) => match args[..] {
            [a, k] if arena.as_num(k).is_some_and(|q| *q == -Q::one()) => {
                Some((a, LambertRealBranch::MinusOne))
            }
            _ => None,
        },
        _ => None,
    }
    .filter(|&(a, _)| walk::free_symbols(arena, a).is_empty())
}

/// Is `w = W_k(a)` real: `a` real with `a + 1/e ≥ 0` (`W₀`), and `a < 0`
/// as well for `W₋₁`, by certified signs?  (`a = −1/e` exactly is `W =
/// −1`, which `eval` writes as such.)
fn lambert_value_is_real(arena: &mut Arena, w: ExprId) -> bool {
    let Some((a, branch)) = lambert_real_argument(arena, w) else {
        return false;
    };
    if !known_real(arena, a) {
        return false;
    }
    let neg_one = arena.neg_one;
    let inv_e = arena.exp(neg_one);
    let shifted = arena.add(&[a, inv_e]);
    if certified_real_sign(arena, shifted) != Some(std::cmp::Ordering::Greater) {
        return false;
    }
    branch == LambertRealBranch::Principal
        || certified_real_sign(arena, a) == Some(std::cmp::Ordering::Less)
}

/// The two branches of Lambert's W that are real on part of the real axis:
/// `W₀` (values `≥ −1`) and `W₋₁` (values `≤ −1`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum LambertRealBranch {
    Principal,
    MinusOne,
}

/// The exact real `y` with `y·e^y = x` on `branch` (`y ≥ −1` for `W₀`, `y ≤
/// −1` for `W₋₁`, the ranges on which `y ↦ y·e^y` is injective, so that
/// `W_k(x) = y`): the visible forms of [`lambert_visible_fixed_point`], then
/// for a constant `x` a factor `exp(c)` whose cofactor is the real `c`
/// (`W(√2·e^{√2}) = √2`), or a factor `ln m` (`m` a positive rational) with
/// `y = r·ln m`, `e^y = m^r` (`r` rational, read off the value of `W_k(x)`,
/// then verified exactly): `W(√2·ln(2)/2) = ln(2)/2`, `W(−2·ln(5/2)/5) =
/// −ln(5/2)`, `W₋₁(−3√2·ln(2)/8) = −3·ln(2)/2`.  The branch is the
/// certified sign of `y + 1`: `W₋₁(−2·ln(5/2)/5)` is not `−ln(5/2) > −1`
/// and stays.
///
/// Up to 0.45 only rational `c` and integer `t`, `p` in `t·p^t·ln p` were
/// recognised: `W(√2·ln(2)/2)` stayed, a value no evaluation tells from
/// `ln(2)/2`, and the solver recognised `W_k(y·e^y) = y` for its own
/// candidates only.
fn lambert_fixed_point(arena: &mut Arena, x: ExprId, branch: LambertRealBranch) -> Option<ExprId> {
    if let Some(y) = lambert_visible_fixed_point(arena, x, branch) {
        return Some(y);
    }
    if !walk::free_symbols(arena, x).is_empty() {
        return None;
    }
    let ExprNode::Mul(factors) = arena.node(x).clone() else {
        return None;
    };
    // y·e^y with e^y written `exp(y)`.
    for &f in &factors {
        if let ExprNode::Exp(c) = *arena.node(f) {
            let rest = arena.div(x, f);
            if rest == c && known_real(arena, c) {
                return lambert_on_branch(arena, c, branch);
            }
        }
    }
    // y = r·ln m, e^y = m^r.
    for &f in &factors {
        let ExprNode::Ln(m) = *arena.node(f) else {
            continue;
        };
        let Some(ln_m) = arena
            .as_num(m)
            .filter(|q| q.is_positive() && !q.is_one())
            .and_then(|q| Some(q.to_f64()?.ln()))
            .filter(|l| l.is_finite() && *l != 0.0)
        else {
            continue;
        };
        let w = match branch {
            LambertRealBranch::Principal => arena.lambertw(x),
            LambertRealBranch::MinusOne => {
                let k = arena.neg_one;
                arena.lambertw_branch(x, k)
            }
        };
        let Ok(value) = crate::transforms::evalf::evalf_complex64(arena, w) else {
            continue;
        };
        if !value.re.is_finite() || value.im.abs() > 1e-12 * value.re.abs().max(1.0) {
            continue;
        }
        let Some(r) = small_rational_near(value.re / ln_m, LAMBERT_MAX_DENOMINATOR) else {
            continue;
        };
        let r = arena.num_ratio(r);
        let y = arena.mul(&[r, f]);
        let e_y = arena.pow(m, r);
        let product = arena.mul(&[y, e_y]);
        let exact = product == x || {
            let d = arena.sub(product, x);
            constant_zero_decision(arena, d) == Some(true)
        };
        if exact {
            return lambert_on_branch(arena, y, branch);
        }
    }
    None
}

/// Largest denominator of the rational `r` in `W_k(r·ln m·m^r) = r·ln m`
/// that [`lambert_fixed_point`] reads off a value.
const LAMBERT_MAX_DENOMINATOR: i64 = 24;

/// The rational with denominator at most `max_den` within `10⁻¹⁰`
/// (relative) of `x`, the one with the least denominator.
fn small_rational_near(x: f64, max_den: i64) -> Option<Q> {
    if !x.is_finite() || x.abs() > 1e9 {
        return None;
    }
    (1..=max_den).find_map(|q| {
        let p = (x * q as f64).round();
        ((p / q as f64 - x).abs() <= 1e-10 * x.abs().max(1.0))
            .then(|| Q::new(BigInt::from(p as i64), BigInt::from(q)))
    })
}

/// `y` when the real constant `y` lies on `branch` of `W` (`y + 1 ≥ 0` for
/// `W₀`, `≤ 0` for `W₋₁`, by its certified sign); `None` otherwise.
fn lambert_on_branch(arena: &mut Arena, y: ExprId, branch: LambertRealBranch) -> Option<ExprId> {
    let one = arena.one;
    let y_plus_one = arena.add(&[y, one]);
    let sign = if arena.is_zero_structural(y_plus_one) {
        std::cmp::Ordering::Equal
    } else {
        certified_real_sign(arena, y_plus_one)?
    };
    let on_branch = match branch {
        LambertRealBranch::Principal => sign.is_ge(),
        LambertRealBranch::MinusOne => sign.is_le(),
    };
    on_branch.then_some(y)
}

/// The exact `w` with `w·e^w = x` on `branch` (`w ≥ −1` for `W₀`, `w ≤ −1`
/// for `W₋₁`), when `x` is visibly of that form: `c·exp(c)` for a rational
/// `c` (`W₋₁(−2e⁻²) = −2`, `W₀(3e³) = 3`), or `t·p^t·ln p` for an integer
/// `p ≥ 2` and a small integer `t` (`w = t·ln p`: `W₀(−ln(2)/2) = −ln 2`,
/// `W₋₁(−ln(2)/2) = −2·ln 2`, `W₀(3·ln 3) = ln 3`).
fn lambert_visible_fixed_point(
    arena: &mut Arena,
    x: ExprId,
    branch: LambertRealBranch,
) -> Option<ExprId> {
    let on_branch = |w_plus_one: std::cmp::Ordering| match branch {
        LambertRealBranch::Principal => w_plus_one.is_ge(),
        LambertRealBranch::MinusOne => w_plus_one.is_le(),
    };
    let ExprNode::Mul(factors) = arena.node(x).clone() else {
        return None;
    };
    let [a, b] = factors[..] else {
        return None;
    };
    let q = arena.as_num(a)?.clone();
    match arena.node(b).clone() {
        // c·exp(c)
        ExprNode::Exp(c) => {
            let c = arena.as_num(c)?.clone();
            if c != q {
                return None;
            }
            let w_plus_one = (&c + Q::from_integer(BigInt::from(1))).cmp(&Q::zero());
            on_branch(w_plus_one).then(|| arena.num_ratio(c))
        }
        // t·p^t·ln p, w = t·ln p
        ExprNode::Ln(p_id) => {
            let p = arena.as_num(p_id)?.clone();
            if !p.is_integer() || p <= Q::from_integer(BigInt::from(1)) {
                return None;
            }
            let base = p.to_integer();
            let ln_p = base.to_f64()?.ln();
            // `q = t·p^t`: for `t > 0` the numerator of `q` has the bits of
            // `p^t`, for `t < 0` its denominator does; the powers stop once
            // they outgrow them.
            for negative in [false, true] {
                let limit = if negative { q.denom() } else { q.numer() }.bits() + 8;
                let mut power = BigInt::from(1);
                for t_abs in 1i32..=64 {
                    power *= &base;
                    if power.bits() > limit {
                        break;
                    }
                    let t = if negative { -t_abs } else { t_abs };
                    let candidate = if negative {
                        Q::new(BigInt::from(t), power.clone())
                    } else {
                        Q::from_integer(BigInt::from(t) * &power)
                    };
                    if candidate != q {
                        continue;
                    }
                    // `t·ln p + 1` is never 0 (`e^{−1/t}` is not an integer).
                    let w_plus_one = (f64::from(t) * ln_p + 1.0)
                        .partial_cmp(&0.0)
                        .unwrap_or(std::cmp::Ordering::Equal);
                    if on_branch(w_plus_one) {
                        let t_id = arena.int(i64::from(t));
                        let ln_id = arena.ln(p_id);
                        return Some(arena.mul(&[t_id, ln_id]));
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Exact values of `lambertw(x)` and `lambertw(x, k)` (SymPy's
/// `LambertW.eval`): the principal branch, `k = 0` or no `k`, is the
/// `ExprNode::LambertW` node with its rules ([`eval_lambertw`]); every other
/// branch is `−∞` at 0 (logarithmic singularity), and `W₋₁` has
/// `W₋₁(−1/e) = −1`, `W₋₁(−π/2) = −iπ/2` and `W₋₁(w·e^w) = w` for the
/// real `w ≤ −1` of [`lambert_fixed_point`].  A branch index that is not an
/// integer is left alone.
pub(super) fn eval_lambertw_call(arena: &mut Arena, args: &[ExprId]) -> Option<ExprId> {
    let x = args[0];
    let principal =
        |arena: &mut Arena| eval_lambertw(arena, x).unwrap_or_else(|| arena.lambertw(x));
    let Some(&k) = args.get(1) else {
        return Some(principal(arena));
    };
    let kq = arena.as_num(k)?.clone();
    if !kq.is_integer() {
        return None;
    }
    if kq.is_zero() {
        return Some(principal(arena));
    }
    if arena.as_num(x).is_some_and(|q| q.is_zero()) {
        tracing::debug!("eval: LambertW(0, k) = -oo for k != 0");
        return Some(arena.neg_infinity);
    }
    if kq != Q::from_integer(BigInt::from(-1)) {
        return None;
    }
    // W₋₁(−1/e) = −1.
    let e = arena.e_const;
    let neg1 = arena.neg_one;
    let neg_inv_e = arena.div(neg1, e);
    let exp_neg1 = arena.exp(neg1);
    let neg_exp_neg1 = arena.neg(exp_neg1);
    if x == neg_inv_e || x == neg_exp_neg1 {
        tracing::debug!("eval: LambertW(-1/e, -1) = -1");
        return Some(neg1);
    }
    // W₋₁(−π/2) = −iπ/2.
    let pi = arena.pi;
    let neg_half = arena.rational(-1, 2);
    let neg_pi_2 = arena.mul(&[neg_half, pi]);
    if x == neg_pi_2 {
        tracing::debug!("eval: LambertW(-pi/2, -1) = -i*pi/2");
        let i = arena.i_unit;
        return Some(arena.mul(&[neg_half, i, pi]));
    }
    lambert_fixed_point(arena, x, LambertRealBranch::MinusOne)
}

pub(super) fn eval_erfc(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // erfc(0) = 1
    if let Some(r) = arena.as_num(inner)
        && r.is_zero()
    {
        return Some(arena.one);
    }

    // erfc(∞) = 0
    if matches!(arena.node(inner), ExprNode::Infinity) {
        return Some(arena.zero);
    }

    // erfc(-∞) = 2
    if matches!(arena.node(inner), ExprNode::NegInfinity) {
        return Some(arena.int(2));
    }

    None
}

/// `B(a, b)` for positive integers: `(a−1)!(b−1)!/(a+b−1)! =
/// 1/(b·C(a+b−1, b))`, the binomial taken with `j = min(a−1, b)` factors
/// and within the digit guard (`C(n, j) ≥ (n/j)^j`).
///
/// Before 0.30 the three factorials were multiplied out (a `Ratio` product
/// of `a + b` factors): `beta(6, 10^7)`, `beta(10^12, 3)`, `beta(10^8, 17)`
/// and `beta(8, 14!)` never returned, though each is a small rational.
///
/// Other rationals off the poles: `Γ(a)Γ(b)/Γ(a + b)` when the three `Γ`
/// fold (half-integers, a positive integer beside any rational): `B(1,
/// 5/2) = 2/5`, `B(1/2, 1/2) = π` (SymPy folds them; before 0.47 they
/// stayed).
pub(super) fn eval_beta(arena: &mut Arena, a: ExprId, b: ExprId) -> Option<ExprId> {
    let ra = arena.as_num(a)?.clone();
    let rb = arena.as_num(b)?.clone();
    if !ra.is_integer() || !ra.is_positive() || !rb.is_integer() || !rb.is_positive() {
        return beta_by_gamma(arena, &ra, &rb);
    }
    let a_int = ra.to_integer();
    let b_int = rb.to_integer();
    let n = &a_int + &b_int - BigInt::one();
    let j = (&a_int - BigInt::one()).min(b_int.clone());
    let (nf, jf) = (n.to_f64()?, j.to_f64()?);
    if jf >= 1.0 && beyond_digit_guard(arena, jf * (nf / jf).log10()) {
        return None;
    }
    let c = crate::base::combinatorics::binomial(n, j);
    guarded_num(arena, Ratio::new(BigInt::one(), c * b_int))
}

/// `ln(a)/ln(b)` in a product, `a` and `b` positive rationals other than 1
/// with `a = b^k` for a rational `k` (`a^q = b^p`, `q ≤ 64`): `k` — the
/// `log(a, b)` the parser writes as that quotient.  SymPy folds `log(8, 2)`
/// to 3 and `log(27, 3)` to 3; before 0.47 they stayed `ln(8)/ln(2)`.
/// `None` unless the product holds exactly one such pair.
pub(super) fn eval_log_quotient(arena: &mut Arena, factors: &[ExprId]) -> Option<ExprId> {
    let positive_rational = |arena: &Arena, id: ExprId| -> Option<Q> {
        let ExprNode::Ln(a) = *arena.node(id) else {
            return None;
        };
        let r = arena.as_num(a)?;
        (r.is_positive() && !r.is_one() && r.numer().bits() <= 256 && r.denom().bits() <= 256)
            .then(|| r.clone())
    };
    let mut numerators = Vec::new();
    let mut denominators = Vec::new();
    for (i, &f) in factors.iter().enumerate() {
        if let Some(a) = positive_rational(arena, f) {
            numerators.push((i, a));
        } else if let ExprNode::Pow(base, e) = *arena.node(f)
            && arena.as_num(e).is_some_and(|q| *q == -Q::one())
            && let Some(b) = positive_rational(arena, base)
        {
            denominators.push((i, b));
        }
    }
    let ([(i, a)], [(j, b)]) = (numerators.as_slice(), denominators.as_slice()) else {
        return None;
    };
    let k = rational_log(a, b)?;
    let k = arena.num_ratio(k);
    let mut rest: smallvec::SmallVec<[ExprId; 6]> = factors
        .iter()
        .enumerate()
        .filter(|(n, _)| n != i && n != j)
        .map(|(_, &f)| f)
        .collect();
    rest.push(k);
    Some(arena.mul(&rest))
}

/// The rational `k = p/q` (`q ≤ 64`) with `a = b^k` for positive rationals
/// `a`, `b ≠ 1`, checked exactly (`a^q = b^p`).
fn rational_log(a: &Q, b: &Q) -> Option<Q> {
    use num_traits::Pow;
    let lg = |q: &Q| -> Option<f64> {
        let n = q.numer().to_f64()?.ln();
        let d = q.denom().to_f64()?.ln();
        Some(n - d)
    };
    let estimate = lg(a)? / lg(b)?;
    if !estimate.is_finite() || estimate.abs() > 4096.0 {
        return None;
    }
    for q in 1..=64u32 {
        let p = (estimate * f64::from(q)).round();
        if (estimate * f64::from(q) - p).abs() > 1e-6 || p == 0.0 {
            continue;
        }
        let p = p as i64;
        let e = u32::try_from(p.unsigned_abs()).ok()?;
        // a^q = b^p, as `an^q·bd^p = bn^p·ad^q` (`b^−|p|` swapped).
        let (bn, bd) = if p > 0 {
            (b.numer(), b.denom())
        } else {
            (b.denom(), b.numer())
        };
        let lhs: BigInt = Pow::pow(a.numer(), q) * Pow::pow(bd, e);
        let rhs: BigInt = Pow::pow(bn, e) * Pow::pow(a.denom(), q);
        if lhs == rhs {
            return Some(Q::new(BigInt::from(p), BigInt::from(q)));
        }
    }
    None
}

/// Most nodes of a constant whose floor or ceiling is evaluated.
const MAX_INTEGER_PART_NODES: usize = 60;

/// `⌊x⌋` (`ceiling`: `⌈x⌉`) of a real constant `x` that is not a rational:
/// the integer its certified decimal value lies strictly inside, or, when
/// that value is within its last digit of an integer `n`, `n` if `x − n` is
/// exactly 0 ([`exact_zero_test`]).  SymPy's `floor.eval` (`get_integer_part`)
/// folds `floor(E)` to 2 and `ceiling(50!/e)` to an integer; before 0.47
/// they stayed.  `None` when `x` is not known real, its value does not
/// settle, or the integer is beyond 1,000 digits.
///
/// The size is read off the printed exponent before any exact value is
/// built: the 0.47 development code built `digits·10^shift` first, and
/// `floor(Shi(12!))` (`Shi(12!) ≈ 5.3·10²⁰⁸⁰²⁷⁷⁴²`) raised 10 to the
/// 208,027,742nd power (the `fuzz_evalf` timeout of the release campaign).
/// A value below `10⁻⁵` in modulus (a certified nonzero value was printed)
/// is settled by its sign alone, without its `10^−k` either.
///
/// An integer part that took more than the first 25 digits is memoised
/// per arena ([`INTEGER_PART_MEMO`]): every `eval_f64`, `eval_decimal` and
/// `eval_complex64` begins with `eval`, and up to 0.51 each recomputed the
/// 692-digit integer part of `ceiling(Chi(1600))` (1.15 s in a release
/// build; seven of them timed out `fuzz_evalf`).
pub(super) fn integer_part_of_constant(
    arena: &mut Arena,
    x: ExprId,
    ceiling: bool,
) -> Option<ExprId> {
    let key = (arena.serial(), x, ceiling);
    if let Some(n) = INTEGER_PART_MEMO.with(|m| m.borrow().get(&key).cloned()) {
        return Some(arena.num_ratio(n));
    }
    if arena.as_num(x).is_some()
        || !walk::free_symbols(arena, x).is_empty()
        || walk::has_unevaluated(arena, x)
        || walk::post_order_ids(arena, x).len() > MAX_INTEGER_PART_NODES
        || !known_real(arena, x)
    {
        return None;
    }
    let mut digits = 25u32;
    let (value, ulp) = loop {
        // The digits `evalf` prints, without the string (up to 0.57 the
        // printout was split and parsed).
        let cv = crate::transforms::evalf::const_value(arena, x, digits).ok()?;
        if cv.im_sign().is_nonzero().is_true() {
            return None;
        }
        let parts = cv.re_decimal()?;
        let exponent = if parts.is_zero() {
            -1
        } else {
            parts.magnitude()
        };
        if exponent < -5 && !parts.is_zero() {
            // 0 < |x| < 10⁻⁵.
            let n = match (parts.is_negative(), ceiling) {
                (false, false) | (true, true) => 0,
                (false, true) => 1,
                (true, false) => -1,
            };
            return Some(arena.int(n));
        }
        if exponent + 20 < i64::from(digits) {
            break decimal_value_and_ulp(&parts)?;
        }
        if digits > 25 {
            return None;
        }
        digits = u32::try_from(exponent + 25).ok().filter(|&d| d <= 1000)?;
    };
    let round = |q: Q| if ceiling { q.ceil() } else { q.floor() };
    let low = round(&value - &ulp);
    let high = round(&value + &ulp);
    let n = if low == high {
        low
    } else {
        // Within a digit of an integer: that integer if the difference is 0.
        let n = value.round();
        let n_id = arena.num_ratio(n.clone());
        let diff = arena.sub(x, n_id);
        if exact_zero_test(arena, diff) != Some(true) {
            return None;
        }
        n
    };
    if digits > 25 {
        INTEGER_PART_MEMO.with(|m| {
            let mut m = m.borrow_mut();
            if m.len() >= INTEGER_PART_MEMO_ENTRIES {
                m.clear();
            }
            m.insert(key, n.clone());
        });
    }
    Some(arena.num_ratio(n))
}

thread_local! {
    /// The integer parts [`integer_part_of_constant`] computed beyond its
    /// first 25 digits, by (arena serial, constant, ceiling).  A node of an
    /// arena never changes, and the integer part of a constant does not
    /// depend on the configuration or on assumptions (a symbol-free
    /// constant), so an entry stays valid; the map is cleared when full.
    static INTEGER_PART_MEMO: std::cell::RefCell<FxHashMap<(u64, ExprId, bool), Q>> =
        std::cell::RefCell::new(FxHashMap::default());
}

/// The entries [`INTEGER_PART_MEMO`] holds before it is cleared.
const INTEGER_PART_MEMO_ENTRIES: usize = 256;

/// The exact value of the decimal `d` (the digits `evalf` prints) and a
/// unit in its last nonzero digit.  It costs a power of ten of the size of
/// the exponent: callers bound that first.  (Up to 0.57 the printed string
/// was split and parsed.)
fn decimal_value_and_ulp(d: &crate::transforms::evalf::Decimal) -> Option<(Q, Q)> {
    let shift = d.exponent();
    let p = num_traits::pow(
        BigInt::from(10),
        usize::try_from(shift.unsigned_abs()).ok()?,
    );
    let ulp = if shift >= 0 {
        Q::from_integer(p)
    } else {
        Q::new(BigInt::one(), p)
    };
    Some((d.to_ratio(shift.unsigned_abs())?, ulp))
}

/// `B(a, b) = Γ(a)Γ(b)/Γ(a + b)` for rationals `a`, `b` none of which nor
/// `a + b` is a pole of `Γ`, when each `Γ` folds to an expression free of
/// `Γ` (a positive integer one of them, or half-integers: `Γ(n + ½)` is
/// a rational times `√π`); `None` otherwise.
fn beta_by_gamma(arena: &mut Arena, ra: &Q, rb: &Q) -> Option<ExprId> {
    let pole = |q: &Q| q.is_integer() && !q.is_positive();
    let sum = ra + rb;
    if pole(ra) || pole(rb) || pole(&sum) {
        return None;
    }
    // Γ of a rational of a large numerator is a large exact number.
    let small = |q: &Q| q.numer().bits() <= 16 && q.denom().bits() <= 16;
    if !small(ra) || !small(rb) || !small(&sum) {
        return None;
    }
    let mut folded = |q: &Q| -> Option<ExprId> {
        let n = arena.num_ratio(q.clone());
        let g = arena.gamma(n);
        let g = eval(arena, g);
        let free = !walk::post_order_ids(arena, g)
            .iter()
            .any(|&id| matches!(arena.node(id), ExprNode::Gamma(_)));
        free.then_some(g)
    };
    let ga = folded(ra)?;
    let gb = folded(rb)?;
    let gs = folded(&sum)?;
    let num = arena.mul(&[ga, gb]);
    let quotient = arena.div(num, gs);
    Some(eval(arena, quotient))
}

/// `C(n, k)` for non-negative integer literals.  A negative `n` is left
/// unevaluated on purpose: the generalised `C(−3, 2) = 6` of
/// `combinatorics::binomial` is not what the symbolic node promises.
/// The value `p + q·i` of `id` when it is a rational, `i`, or a rational
/// multiple of `i` plus a rational (the canonical forms of an exact
/// Gaussian rational: `1/2`, `I`, `3*I`, `1/2 + 2*I`).
pub(super) fn gaussian_rational(arena: &Arena, id: ExprId) -> Option<(Q, Q)> {
    let imag = |t: ExprId| -> Option<Q> {
        if t == arena.i_unit {
            return Some(Q::one());
        }
        match arena.node(t) {
            ExprNode::Mul(cs) if cs.len() == 2 => {
                let (c, u) = (cs[0], cs[1]);
                match (arena.as_num(c), arena.as_num(u)) {
                    (Some(q), None) if u == arena.i_unit => Some(q.clone()),
                    (None, Some(q)) if c == arena.i_unit => Some(q.clone()),
                    _ => None,
                }
            }
            _ => None,
        }
    };
    if let Some(q) = arena.as_num(id) {
        return Some((q.clone(), Q::zero()));
    }
    if let Some(q) = imag(id) {
        return Some((Q::zero(), q));
    }
    if let ExprNode::Add(cs) = arena.node(id)
        && cs.len() == 2
    {
        for (r, i) in [(cs[0], cs[1]), (cs[1], cs[0])] {
            if let (Some(re), Some(im)) = (arena.as_num(r), imag(i)) {
                return Some((re.clone(), im));
            }
        }
    }
    None
}

/// Are `a` and `b` equal, when both are exact Gaussian rationals
/// ([`gaussian_rational`])?  `Eq`/`Ne` of two numbers fold; before 0.37 only
/// two real rationals did, so `Ne(1/2, I)` — the condition the integrator
/// attaches to `∫ e^(a·x)·cos x` for `a ≠ ±i` — stayed undecided after
/// `a = 1/2` was substituted, and a `Piecewise` on it with it.
pub(super) fn gaussian_equal(arena: &Arena, a: ExprId, b: ExprId) -> Option<bool> {
    let x = gaussian_rational(arena, a)?;
    let y = gaussian_rational(arena, b)?;
    Some(x == y)
}

pub(super) fn eval_binomial(arena: &mut Arena, n: ExprId, k: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::base::truth::Truth;
    // `C(n, k) = 0` for a negative integer `k` and every finite `n`, the
    // convention of Graham, Knuth and Patashnik (*Concrete Mathematics*
    // §5.1) and SymPy: it keeps Pascal's rule `C(r, k) = C(r − 1, k) +
    // C(r − 1, k − 1)` for every integer `k`, which the summation
    // algorithms rely on; the symmetry `C(n, k) = C(n, n − k)` then holds
    // for `n ≥ 0` only.  (At two negative integers the function of two
    // variables has no limit: along `k` fixed it is 0, along `n − k` fixed
    // mpmath's `binomial(−2, −4) = 3`.)  Up to 0.52 a negative integer `n`
    // was left alone here and `evalf` gave mpmath's value, so `eval` kept
    // `binomial(−2, −4)` while `evalf` printed 3 and `compile` 0.  An `n`
    // with symbols is a finite value (the generic convention, `0·x = 0`); a
    // constant `n` must be known finite: one without a value (`min(√2,
    // E₂(−3))`, `nan`) or infinite leaves the coefficient without one.
    let mut cache = AssumptionCache::new();
    let mut known = |id: ExprId, p: Props| Truth::from(cache.query(arena, id, p));
    let negative_integer_k = match arena.as_num(k) {
        Some(q) => q.is_integer() && q.is_negative(),
        None => known(k, Props::INTEGER)
            .and(known(k, Props::NEGATIVE))
            .is_true(),
    };
    let finite_n = if walk::free_symbols(arena, n).is_empty() {
        known(n, Props::FINITE).is_true()
    } else {
        !known(n, Props::FINITE).is_false()
    };
    if negative_integer_k && finite_n {
        return Some(arena.zero);
    }
    let nr = arena.as_num(n)?.clone();
    let kr = arena.as_num(k)?.clone();
    if !kr.is_integer() {
        return None;
    }
    // `n(n − 1)⋯(n − k + 1)/k!` for a rational `n` and an integer `k ≥ 0`
    // (SymPy and mpmath agree: `binomial(−1, 3) = −1`, `binomial(7/2, 2) =
    // 35/8`).  Before 0.37 only `0 ≤ k`, `0 ≤ n` integers folded, and
    // `binomial(−1, 3)` was not even evaluable numerically.
    if !nr.is_integer() || nr.is_negative() {
        let k_u64: u64 = kr.to_integer().try_into().ok()?;
        let falling = exact_factorial_power(arena, &nr, k_u64, -1)?;
        let falling = arena.as_num(falling)?.clone();
        let k_digits = log10_factorial_lower(k_u64 as f64);
        if beyond_digit_guard(arena, k_digits) {
            return None;
        }
        let mut fact = BigInt::one();
        for i in 2..=k_u64 {
            fact *= BigInt::from(i);
        }
        return guarded_num(arena, falling / Ratio::from_integer(fact));
    }
    let (n_big, k_big) = (nr.to_integer(), kr.to_integer());
    if k_big > n_big {
        // 0 ≤ n < k (SymPy: `binomial(1, 2) == 0`).
        return Some(arena.zero);
    }
    // C(n, j) ≥ (n/j)^j for j = min(k, n − k): within the digit guard.
    // (An `n` beyond `u64` with a small `j` folds too: before 0.40
    // `binomial(10²⁰, 2)` stayed unevaluated.)
    let j_big = (&n_big - &k_big).min(k_big.clone());
    let j: u64 = j_big.try_into().ok()?;
    let n_f = n_big.to_f64().unwrap_or(f64::INFINITY);
    if j >= 1 && beyond_digit_guard(arena, j as f64 * (n_f / j as f64).log10()) {
        return None;
    }
    let result = match (u64::try_from(&n_big), u64::try_from(&k_big)) {
        (Ok(n_u64), Ok(k_u64)) => crate::base::combinatorics::binomial(n_u64, k_u64),
        _ => {
            // n(n − 1)⋯(n − j + 1)/j!
            let mut numer = BigInt::one();
            for i in 0..j {
                numer *= &n_big - BigInt::from(i);
            }
            numer / crate::base::combinatorics::factorial(j)
        }
    };
    guarded_num(arena, Ratio::from_integer(result))
}
