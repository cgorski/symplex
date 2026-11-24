//! Sub-tree evaluation helpers (Sum, Product, Piecewise, RootOf, RootSum) (part of [`evalf`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Sub-tree evaluation helpers (Sum, Product, Piecewise, RootOf, RootSum)
// ═══════════════════════════════════════════════════════════════════════════

/// Most terms of a finite `Sum` or `Product` evaluated numerically.
const MAX_RANGE_TERMS: u32 = 10_000;

/// The value and error bound of `expr` (with post-order `post_order`) at
/// the integer `k` for the bound variable `var`: the index is exact when
/// it fits the precision.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_at_integer(
    arena: &Arena,
    expr: ExprId,
    post_order: &[ExprId],
    var: ExprId,
    k: &BigInt,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    let prec = w.bits();
    let value = (
        crate::base::numeric::bigint_to_bigfloat(k, prec),
        BigFloat::new(prec),
    );
    let err = if k.bits() <= prec as u64 {
        accuracy::EXACT
    } else {
        accuracy::rounding(&value, prec)
    };
    evaluate_tree(
        arena,
        expr,
        post_order,
        Some((var, CBall::new(value, accuracy::Bound::real(err)))),
        true,
        w,
    )
}

/// A bound of a `Sum` or `Product` range as an integer: an integer literal
/// exactly, another value when it is real and within `10⁻⁹` of an integer.
fn range_bound(
    arena: &Arena,
    cache: Cache<'_>,
    id: ExprId,
    what: &str,
) -> Result<BigInt, SymplexError> {
    if let Some(q) = arena.as_num(id) {
        if q.is_integer() {
            return Ok(q.to_integer());
        }
        return Err(unevaluable(format!("{what} bounds are not integers")));
    }
    let v = get_cached(cache, id)?;
    if !v.1.is_zero() {
        return Err(unevaluable(format!("{what} bounds must be real")));
    }
    // An integer only when its value is one exactly, with an exact bound
    // (`floor(…)`): up to 0.53 a value within `10⁻⁹` of an integer was
    // taken for it, and `Σ_{k=1}^{3−e⁻¹⁰⁰} k^k` was 32.
    let exact = cache.bound(&id).is_some_and(|b| b.is_exact());
    if !exact || !(v.0.is_zero() || v.0.is_int()) {
        return Err(unevaluable(format!(
            "{what} bounds are not known to be integers"
        )));
    }
    let r = f64_of(&v.0)?;
    if r.abs() > 9.0e15 {
        return Err(unevaluable(format!("{what} bounds are too large")));
    }
    Ok(BigInt::from(r as i64))
}

/// The number of terms of `lo..=hi` (0 when empty), refused beyond
/// [`MAX_RANGE_TERMS`].
fn range_terms(lo: &BigInt, hi: &BigInt, what: &str) -> Result<usize, SymplexError> {
    if hi < lo {
        return Ok(0);
    }
    let n = hi - lo + 1;
    crate::base::cast::bounded_len(&n, MAX_RANGE_TERMS + 1).ok_or_else(|| {
        unevaluable(format!(
            "{what} range too large for numerical evaluation (> {MAX_RANGE_TERMS} terms)"
        ))
    })
}

/// A running sum of values with error bounds: per part, the terms' bounds
/// add, and every partial sum is rounded to the working precision
/// ([`accuracy::PartSum`]); a part that is an exact 0 in every term stays
/// exact (a sum of real terms is real).
pub(super) struct SumAcc {
    pub(super) value: Complex,
    re: accuracy::PartSum,
    im: accuracy::PartSum,
    unknown: bool,
    prec: usize,
}

impl SumAcc {
    pub(super) fn new(prec: usize) -> SumAcc {
        SumAcc {
            value: c_zero(prec),
            re: accuracy::PartSum::new(),
            im: accuracy::PartSum::new(),
            unknown: false,
            prec,
        }
    }

    pub(super) fn add(&mut self, term: &Complex, err: accuracy::Bound) {
        self.value = c_add(&self.value, term, self.prec, RM);
        self.unknown |= err.is_unknown();
        self.re.add(&term.0, err.re, &self.value.0, self.prec);
        self.im.add(&term.1, err.im, &self.value.1, self.prec);
    }

    /// The sum and its error bound.
    pub(super) fn finish(self) -> CBall {
        if self.unknown {
            return CBall::new(self.value, accuracy::Bound::UNKNOWN);
        }
        let bound = accuracy::Bound {
            re: self.re.bound(),
            im: self.im.bound(),
        };
        CBall::new(self.value, bound)
    }
}

/// `Σ_{var = lo}^{hi} body`: a finite range of at most [`MAX_RANGE_TERMS`]
/// terms term by term, `hi = ∞` as a hypergeometric series ([`hypsum`]).
/// The bound is the terms' bounds plus the roundings (before 0.29 it was
/// the working precision whatever the terms lost, and an infinite sum was
/// `Unevaluable`).  A range of more than [`windowsum::MIN_RANGE_TERMS`]
/// terms, or an infinite one whose terms peak more than
/// [`windowsum::MIN_MODE_OFFSET`] above the lower limit, is first tried as
/// a window around the mode ([`windowsum`]: weights moved by their
/// recurrences, both rests bounded by proved geometric series); up to 0.58
/// a finite range beyond `MAX_RANGE_TERMS` was refused (the entropy of
/// `Binomial(10⁵, p)`) and such an infinite one beyond 20,000 terms.
#[allow(clippy::too_many_arguments)]
pub(super) fn eval_sum(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    let prec = w.bits();
    if matches!(arena.node(lo), ExprNode::NegInfinity) {
        return Err(unevaluable("a Sum from −∞ is not evaluated numerically"));
    }
    let lo_i = range_bound(arena, cache, lo, "Sum")?;
    let post_order = walk::post_order_ids(arena, body);
    if matches!(arena.node(hi), ExprNode::Infinity) {
        if let Some(result) = emsum::rational_sum(arena, body, var, &lo_i, prec) {
            debug!("evalf: Sum — infinite sum of a rational term");
            return result;
        }
        if let Some(p) = emsum::power_summand(arena, body, var) {
            debug!("evalf: Sum — a power of a linear term: Hurwitz zeta");
            let mut consts = Vec::with_capacity(p.consts.len());
            for &c in &p.consts {
                let order = walk::post_order_ids(arena, c);
                consts.push(evaluate_at_integer(arena, c, &order, var, &lo_i, w)?);
            }
            return emsum::power_sum(&p, &consts, &lo_i, w);
        }
        match windowsum::windowed_sum(arena, body, var, &lo_i, None, windowsum::MIN_MODE_OFFSET, w)
        {
            windowsum::Windowed::Value(ball) => return Ok(ball),
            // (Refused at once: up to 0.58 after 20,000 terms from the
            // lower limit, 3 to 15 s.)
            windowsum::Windowed::TooWide => {
                return Err(unevaluable(format!(
                    "infinite sum of `{}`: its terms fall too slowly for a window of {} \
                     terms around their peak",
                    arena.display(body),
                    windowsum::MAX_WINDOW_TERMS
                )));
            }
            windowsum::Windowed::Declined => {}
        }
        debug!("evalf: Sum — infinite hypergeometric series");
        let mut orders: FxHashMap<ExprId, Vec<ExprId>> = FxHashMap::default();
        orders.insert(body, post_order);
        let mut part = |id: ExprId, k: &BigInt, p: usize| {
            let order = orders
                .entry(id)
                .or_insert_with(|| walk::post_order_ids(arena, id));
            evaluate_at_integer(arena, id, order, var, k, &mut w.at(p))
        };
        return hypsum::infinite_sum(arena, body, var, &lo_i, prec, &mut part);
    }
    let hi_i = range_bound(arena, cache, hi, "Sum")?;
    // Karr's convention: `Σ_{lo}^{hi} = −Σ_{hi+1}^{lo−1}` for `hi < lo − 1`
    // (it was the empty sum, as `eval` had it up to 0.54).
    let (lo_i, hi_i, negate) = if hi_i < &lo_i - 1 {
        (&hi_i + 1, &lo_i - 1, true)
    } else {
        (lo_i, hi_i, false)
    };
    if &hi_i - &lo_i >= BigInt::from(windowsum::MIN_RANGE_TERMS)
        && let windowsum::Windowed::Value(CBall { mid, err }) =
            windowsum::windowed_sum(arena, body, var, &lo_i, Some(&hi_i), 0, w)
    {
        let mid = if negate {
            (mid.0.neg(), mid.1.neg())
        } else {
            mid
        };
        return Ok(CBall::new(mid, err));
    }
    let terms = range_terms(&lo_i, &hi_i, "Sum")?;
    let mut sum = SumAcc::new(prec);
    let mut k = lo_i.clone();
    for _ in 0..terms {
        let CBall { mid: term, err } = evaluate_at_integer(arena, body, &post_order, var, &k, w)?;
        let term = if negate {
            (term.0.neg(), term.1.neg())
        } else {
            term
        };
        sum.add(&term, err);
        k += 1;
    }
    debug!(lo = %lo_i, hi = %hi_i, "evalf: Sum evaluated");
    Ok(sum.finish())
}

/// `∏_{var = lo}^{hi} body` over a finite range of at most
/// [`MAX_RANGE_TERMS`] terms, bounded as a product of its terms; `hi = ∞`
/// for a rational factor tending to 1 fast enough (`emsum`), or `1 + a(k)`
/// with `a` a hypergeometric term of geometric decay times factors of
/// polynomial growth (`hypsum`).  Up to 0.43 every infinite product was
/// refused ("cannot evaluate infinity": `∏ (1 + 1/k²)`, `∏ (1 − 2^(−k))`).
#[allow(clippy::too_many_arguments)]
pub(super) fn eval_product(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    let prec = w.bits();
    if matches!(arena.node(lo), ExprNode::NegInfinity) {
        return Err(unevaluable(
            "a Product from −∞ is not evaluated numerically",
        ));
    }
    let lo_i = range_bound(arena, cache, lo, "Product")?;
    if matches!(arena.node(hi), ExprNode::Infinity) {
        if let Some(result) = emsum::rational_product(arena, body, var, &lo_i, prec) {
            debug!("evalf: Product — infinite product of a rational factor");
            return result;
        }
        let mut orders: FxHashMap<ExprId, Vec<ExprId>> = FxHashMap::default();
        let mut part = |id: ExprId, k: &BigInt, p: usize| {
            let order = orders
                .entry(id)
                .or_insert_with(|| walk::post_order_ids(arena, id));
            evaluate_at_integer(arena, id, order, var, k, &mut w.at(p))
        };
        if let Some(result) = hypsum::dominated_product(arena, body, var, &lo_i, prec, &mut part) {
            return result;
        }
        return Err(unevaluable(format!(
            "infinite product of `{}`: the factor is neither rational nor 1 plus a \
             term of geometric decay, so no error bound is available",
            arena.display(body)
        )));
    }
    let hi_i = range_bound(arena, cache, hi, "Product")?;
    // Karr's convention: `∏_{lo}^{hi} = 1/∏_{hi+1}^{lo−1}` for `hi < lo − 1`
    // (it was the empty product, as `eval` had it up to 0.55).
    let (lo_i, hi_i, reciprocal) = if hi_i < &lo_i - 1 {
        (&hi_i + 1, &lo_i - 1, true)
    } else {
        (lo_i, hi_i, false)
    };
    let terms = range_terms(&lo_i, &hi_i, "Product")?;
    let post_order = walk::post_order_ids(arena, body);
    let mut factors: Vec<CBall> = Vec::with_capacity(terms);
    let mut k = lo_i.clone();
    for _ in 0..terms {
        factors.push(evaluate_at_integer(arena, body, &post_order, var, &k, w)?);
        k += 1;
    }
    let factors: Vec<&CBall> = factors.iter().collect();
    debug!(lo = %lo_i, hi = %hi_i, "evalf: Product evaluated");
    let product = CBall::product(&factors, prec);
    if reciprocal {
        return reciprocal_ball(&product, prec);
    }
    Ok(product)
}

/// `1/p` with its bound: the rounding of the quotient (`c_div`, a few units
/// of the last place) and the radius `r` of `p = m ± r` carried by `|1/(m +
/// δ) − 1/m| ≤ r/(|m|·(|m| − r)) ≤ 2r/|m|²` for `r ≤ |m|/2`.  A ball that
/// reaches closer to 0 (a factor 0 of Karr's reciprocal product) is refused.
fn reciprocal_ball(p: &CBall, prec: usize) -> Result<CBall, SymplexError> {
    let r = p.err.joint();
    if p.err.is_unknown() || accuracy::contains_zero(&p.mid, accuracy::shift(r, 1.0)) {
        return Err(unevaluable(
            "a Product with reversed limits whose reciprocal range has a factor 0 \
             (Karr's convention: 1/0)",
        ));
    }
    let value = c_div(&c_one(prec), &p.mid, prec, RM);
    let rounding = accuracy::shift(accuracy::rounding(&value, prec), 4.0);
    let carried = accuracy::shift(r, 1.0 - 2.0 * accuracy::lg_abs_low(&p.mid));
    Ok(CBall::new(
        value,
        accuracy::Bound::both(accuracy::lsum(carried, rounding)),
    ))
}

/// An operand of a relation without a value in the cache of `evalf`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Special {
    /// `+∞` (`1`) or `−∞` (`−1`).
    Infinity(i8),
    /// `zoo`.
    ComplexInfinity,
    /// `nan`.
    NaN,
}

/// A condition decided from the values of its operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Decision {
    /// The truth value read off the computed values.
    value: bool,
    /// Does it hold for the true values too?  Every comparison involved
    /// was of exact values, or of a difference outside its error ball.
    certain: bool,
}

/// Decide a boolean condition numerically from the evaluated operands and
/// their error bounds.
///
/// A relation compares the difference of its operands (`Gt`, `Ge` need
/// both real); the decision is certain when both operands are exact (the
/// sign of a rounded difference of exact values is exact) or the same
/// expression, or when the difference is outside its error ball — as
/// `sign` is decided (see [`accuracy`]).  So an equality of distinct
/// inexact expressions is never certain, nor is an order between operands
/// whose imaginary parts are only within their error of 0 (they may not be
/// real).  `And`/`Or`/`Not` combine decisions; a certain `false`
/// operand decides an `And` (a certain `true` an `Or`) whatever the others.
/// An operand `±∞`, `zoo` or `nan` is compared by kind (`−∞ < x < +∞`
/// for a finite real `x`; `nan` equals nothing; orders with `zoo` or `nan`
/// are undecided, as SymPy raises).  `None` when an operand is missing from
/// the cache (free symbol, complex value in an order, unsupported node).
fn decide_condition(
    arena: &Arena,
    cond: ExprId,
    cache: Cache<'_>,
    prec: usize,
) -> Option<Decision> {
    let bound_of = |a: &ExprId| cache.err(a);
    // `a − b`, the exponent of its (joint) error bound, and whether it is
    // exact (both operands exact, or the same expression).
    let difference = |a: &ExprId, b: &ExprId| -> Option<(Complex, accuracy::ErrExp, bool)> {
        let (av, bv) = (cache.get(a)?, cache.get(b)?);
        if a == b {
            return Some((c_zero(prec), accuracy::EXACT, true));
        }
        let (ea, eb) = (bound_of(a), bound_of(b));
        let d = c_sub(av, bv, prec + 64, RM);
        let exact = ea.is_exact() && eb.is_exact();
        let e = if ea.is_unknown() || eb.is_unknown() {
            accuracy::UNKNOWN
        } else {
            // The operands' errors add; the difference rounds at `prec + 64`.
            accuracy::lsum(
                accuracy::lsum(ea.joint(), eb.joint()),
                accuracy::rounding(&d, prec + 64),
            )
        };
        Some((d, e, exact))
    };
    let away_from_zero = |d: &Complex, e: accuracy::ErrExp| {
        !accuracy::is_unknown(e) && !accuracy::contains_zero(d, e)
    };
    // `±∞`, `zoo` and `nan` have no value in the cache: their relations
    // with each other and with a finite (real) value are decided by kind.
    // Up to 0.41 every such relation was undecided: `Piecewise((1, exp(1000)
    // < ∞), (0, True))` was refused (SymPy 1.14: `1`).
    let special = |a: &ExprId| -> Option<Special> {
        match arena.node(*a) {
            ExprNode::Infinity => Some(Special::Infinity(1)),
            ExprNode::NegInfinity => Some(Special::Infinity(-1)),
            ExprNode::ComplexInfinity => Some(Special::ComplexInfinity),
            ExprNode::NaN => Some(Special::NaN),
            _ => None,
        }
    };
    // A finite value of the cache: `Some(true)` when it is real (exactly),
    // `Some(false)` when it may not be.
    let finite_real = |a: &ExprId| -> Option<bool> {
        let v = cache.get(a)?;
        let finite = !(v.0.is_inf() || v.0.is_nan() || v.1.is_inf() || v.1.is_nan());
        finite.then(|| v.1.is_zero() && accuracy::exactly_real(v, bound_of(a)))
    };
    let certainly = |value: bool| Decision {
        value,
        certain: true,
    };
    let order_special = |a: &ExprId, b: &ExprId, strict: bool| -> Option<Decision> {
        // `−∞ < finite real < +∞`; `zoo`, `nan` and complex values have no
        // order (SymPy raises).
        let tier = |x: &ExprId| -> Option<i8> {
            match special(x) {
                Some(Special::Infinity(s)) => Some(s),
                Some(_) => None,
                None => finite_real(x)?.then_some(0),
            }
        };
        let (ta, tb) = (tier(a)?, tier(b)?);
        Some(certainly(match ta.cmp(&tb) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Equal => !strict,
            std::cmp::Ordering::Less => false,
        }))
    };
    let equal_special = |a: &ExprId, b: &ExprId| -> Option<Decision> {
        // `nan` equals nothing (SymPy: `Eq(nan, nan)` is False); an
        // infinity only itself.
        let (sa, sb) = (special(a), special(b));
        if sa == Some(Special::NaN) || sb == Some(Special::NaN) {
            return Some(certainly(false));
        }
        match (sa, sb) {
            (Some(x), Some(y)) => Some(certainly(x == y)),
            (Some(_), None) => finite_real(b).map(|_| certainly(false)),
            (None, Some(_)) => finite_real(a).map(|_| certainly(false)),
            (None, None) => None,
        }
    };
    let order = |a: &ExprId, b: &ExprId, strict: bool| -> Option<Decision> {
        if special(a).is_some() || special(b).is_some() {
            return order_special(a, b, strict);
        }
        let (av, bv) = (cache.get(a)?, cache.get(b)?);
        if !(av.1.is_zero() && bv.1.is_zero()) {
            return None;
        }
        let real =
            accuracy::exactly_real(av, bound_of(a)) && accuracy::exactly_real(bv, bound_of(b));
        let (d, e, exact) = difference(a, b)?;
        // Strictly: astro-float's `is_positive` holds for `+0`, the
        // difference of an operand with itself (up to 0.41 `besselj(1, 2) >
        // besselj(1, 2)` was decided true, and certain).
        let value = bf_strictly_positive(&d.0) || (!strict && d.0.is_zero());
        Some(Decision {
            value,
            certain: exact || (real && away_from_zero(&d, e)),
        })
    };
    let equal = |a: &ExprId, b: &ExprId| -> Option<Decision> {
        if special(a).is_some() || special(b).is_some() {
            return equal_special(a, b);
        }
        let (d, e, exact) = difference(a, b)?;
        let value = d.0.is_zero() && d.1.is_zero();
        Some(Decision {
            value,
            certain: exact || (!value && away_from_zero(&d, e)),
        })
    };

    let post_order = walk::post_order_ids(arena, cond);
    let mut truth: FxHashMap<ExprId, Option<Decision>> = FxHashMap::default();
    for &id in &post_order {
        let kids = |ids: &[ExprId]| -> Vec<Option<Decision>> {
            ids.iter()
                .map(|k| truth.get(k).copied().flatten())
                .collect()
        };
        let v: Option<Decision> = match arena.node(id) {
            ExprNode::BoolTrue => Some(Decision {
                value: true,
                certain: true,
            }),
            ExprNode::BoolFalse => Some(Decision {
                value: false,
                certain: true,
            }),
            ExprNode::Gt(a, b) => order(a, b, true),
            ExprNode::Ge(a, b) => order(a, b, false),
            ExprNode::Eq_(a, b) => equal(a, b),
            ExprNode::Ne(a, b) => equal(a, b).map(|d| Decision {
                value: !d.value,
                ..d
            }),
            ExprNode::Not(inner) => truth.get(inner).copied().flatten().map(|d| Decision {
                value: !d.value,
                ..d
            }),
            ExprNode::And(ks) => combine(&kids(ks), false),
            ExprNode::Or(ks) => combine(&kids(ks), true),
            // Numeric operands and anything else carry no truth value.
            _ => None,
        };
        truth.insert(id, v);
    }
    truth.get(&cond).copied().flatten()
}

/// `Or` (`absorbing = true`) or `And` (`false`) of decisions.  The certain
/// operands combine by Kleene's rules ([`Truth::any`], [`Truth::all`]):
/// a certain absorbing operand decides, and so do certain operands only.
/// Otherwise every operand must be decided, and the result, read off the
/// values, is not certain.
///
/// [`Truth::any`]: crate::base::truth::Truth::any
/// [`Truth::all`]: crate::base::truth::Truth::all
fn combine(operands: &[Option<Decision>], absorbing: bool) -> Option<Decision> {
    use crate::base::truth::Truth;
    let certain = operands.iter().map(|d| match d {
        Some(d) if d.certain => Truth::from_bool(d.value),
        _ => Truth::Unknown,
    });
    let proved = if absorbing {
        Truth::any(certain)
    } else {
        Truth::all(certain)
    };
    if proved.is_known() {
        return Some(Decision {
            value: proved.is_true(),
            certain: true,
        });
    }
    let decided: Option<Vec<Decision>> = operands.iter().copied().collect();
    let decided = decided?;
    Some(Decision {
        value: if absorbing {
            decided.iter().any(|d| d.value)
        } else {
            decided.iter().all(|d| d.value)
        },
        certain: false,
    })
}

/// `Piecewise`: the value of the first branch whose condition is true.
///
/// Conditions are decided in order ([`decide_condition`]).  An undecidable
/// condition is an error — falling through to a later `True` branch would
/// be silently wrong.  A decision that is not certain (a difference within
/// its error of 0) still picks a branch, but the result then has no bound,
/// so the evaluation is repeated at a higher precision and refused when
/// the budget is spent, as `sign` of such a difference is.  Before 0.29 the
/// decision was read off the rounded values: `Piecewise((0, x < √2), (1,
/// True))` at `x = √2 + 10⁻¹⁰⁰` came out `0`.
pub(super) fn eval_piecewise(
    arena: &Arena,
    pairs: &[(ExprId, ExprId)],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    let prec = w.bits();
    debug!(
        branches = pairs.len(),
        "evalf: Piecewise — evaluating conditions"
    );
    let mut certain = true;
    for &(value_id, cond_id) in pairs {
        let Some(decision) = decide_condition(arena, cond_id, cache, prec) else {
            return Err(unevaluable(format!(
                "cannot evaluate piecewise: condition `{}` is undecided",
                arena.display(cond_id)
            )));
        };
        certain &= decision.certain;
        if !decision.value {
            continue;
        }
        let CBall { mid: value, err } = match cache.get(&value_id) {
            Some(v) => CBall::new(v.clone(), cache.err(&value_id)),
            None => eval_ball(arena, value_id, cache, w)?,
        };
        return Ok(CBall {
            mid: value,
            err: if certain {
                err
            } else {
                accuracy::Bound::UNKNOWN
            },
        });
    }
    if !certain {
        // A condition found false only to this precision may be true: no
        // value is known yet.
        return Ok(CBall::new(c_zero(prec), accuracy::Bound::UNKNOWN));
    }
    Err(unevaluable(
        "cannot evaluate piecewise: every condition is false",
    ))
}

/// The error bound of a root ball: its certified radius on both parts of a
/// non-real root, on the real part of a root certified real (whose
/// imaginary part is exactly 0); no bound without a certified radius.
fn radius_error(ball: &crate::poly::roots::RootBall) -> accuracy::Bound {
    let e = match ball.radius.as_ref() {
        None => return accuracy::Bound::UNKNOWN,
        Some(r) if r.is_zero() => accuracy::EXACT,
        Some(r) => accuracy::part_lg(r),
    };
    if ball.real {
        accuracy::Bound::real(e)
    } else {
        accuracy::Bound::both(e)
    }
}

/// `RootOf(poly, var, index)`: the `index`-th entry of
/// [`rootof_roots`](crate::poly::roots::rootof_roots) of `poly` as a
/// polynomial in `var`, bounded by the radius of its certified inclusion
/// disk ([`root_balls`](crate::poly::roots::root_balls)).  A root certified
/// real has an exactly zero imaginary part; without an isolating disk there
/// is no bound.
pub(super) fn eval_rootof(
    arena: &Arena,
    poly_id: ExprId,
    var_id: ExprId,
    idx_id: ExprId,
    prec: usize,
) -> Result<CBall, SymplexError> {
    let idx: usize = match arena.as_num(idx_id) {
        Some(r) if r.is_integer() => {
            let n: i64 = r
                .to_integer()
                .try_into()
                .map_err(|_| unevaluable("RootOf index out of range"))?;
            usize::try_from(n).map_err(|_| unevaluable("RootOf index must be non-negative"))?
        }
        Some(_) => return Err(unevaluable("RootOf index must be an integer")),
        None => return Err(unevaluable("RootOf index must be numeric")),
    };
    let poly = crate::poly::polybridge::expr_to_poly(arena, poly_id, var_id)
        .ok_or_else(|| unevaluable("could not convert RootOf expression to polynomial"))?;
    // The roots with multiplicity in the (re, im) order of `rootof_roots`,
    // the one definition of the order a `RootOf` index refers to
    // (`real_roots` derives its indices from the same call).
    let balls = crate::poly::roots::certified_roots(&poly, prec);
    let found = balls.len();
    let ball = balls.into_iter().nth(idx).ok_or_else(|| {
        unevaluable(format!(
            "RootOf index {idx} exceeds the {found} root(s) found"
        ))
    })?;
    let err = radius_error(&ball);
    Ok(CBall::new(ball.value, err))
}

/// `RootSum(poly, body, sumvar) = Σ_{α: poly(α) = 0} body(α)`: the body at
/// every root found by Aberth's method, each root bounded by its certified
/// inclusion disk.  A body with other free variables fails.
pub(super) fn eval_rootsum(
    arena: &Arena,
    poly_id: ExprId,
    body_id: ExprId,
    sumvar_id: ExprId,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    let prec = w.bits();
    debug!("evalf: RootSum — attempting numerical evaluation via Aberth roots");
    let poly =
        crate::poly::polybridge::expr_to_poly(arena, poly_id, sumvar_id).ok_or_else(|| {
            unevaluable(
                "RootSum: cannot convert polynomial expression to Poly (may contain free symbols)",
            )
        })?;
    let degree = poly.degree().unwrap_or(0);
    let balls = crate::poly::roots::certified_roots(&poly, prec);
    if balls.len() != degree {
        debug!(
            expected = degree,
            found = balls.len(),
            "evalf: RootSum — fewer roots than expected"
        );
        return Err(unevaluable(
            "RootSum: not every root of the polynomial was found",
        ));
    }
    let n_roots = balls.len();
    let post_order = walk::post_order_ids(arena, body_id);
    let mut sum = SumAcc::new(prec);
    for ball in balls {
        let err = radius_error(&ball);
        let seed = Some((sumvar_id, CBall::new(ball.value, err)));
        let CBall { mid: term, err: e } =
            evaluate_tree(arena, body_id, &post_order, seed, true, w)?;
        sum.add(&term, e);
    }
    debug!(n_roots, "evalf: RootSum — evaluated");
    Ok(sum.finish())
}
