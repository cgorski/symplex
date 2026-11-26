//! Definite-integral helpers (part of [`evalf`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Definite-integral helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Does the tree contain a `DefiniteIntegral` node?
pub(super) fn contains_definite_integral(arena: &Arena, root: ExprId) -> bool {
    let mut stack = vec![root];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = arena.node(id);
        if matches!(node, ExprNode::DefiniteIntegral(..)) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// An integration bound as `f64`, with a bound on its distance to the true
/// bound (the rounding to `f64`, and the evaluated value's own error):
/// `±∞` nodes directly (exact), anything else from the evaluated cache
/// (must be real).
fn definite_bound_f64(
    arena: &Arena,
    cache: Cache<'_>,
    id: ExprId,
) -> Result<(f64, f64), SymplexError> {
    match arena.node(id) {
        ExprNode::Infinity => Ok((f64::INFINITY, 0.0)),
        ExprNode::NegInfinity => Ok((f64::NEG_INFINITY, 0.0)),
        _ => {
            let v = get_cached(cache, id)?;
            if !v.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "integration bounds of a definite integral must be real".into(),
                });
            }
            let x = f64_of(&v.0)?;
            let bound = cache.err(&id);
            if bound.is_unknown() || !x.is_finite() {
                return Err(SymplexError::PrecisionExhausted {
                    requested: F64_DIGITS,
                    achieved: 0,
                });
            }
            let exp2 = |e: accuracy::ErrExp| {
                if accuracy::is_exact(e) {
                    0.0
                } else {
                    e.clamp(-1100.0, 1100.0).exp2()
                }
            };
            // |x − v| ≤ ½ ulp(x) ≤ |x|·2⁻⁵³; exact when `v` is an `f64`.
            let exact_in_f64 = bigfloat::from_f64(x, 64) == v.0;
            let conversion = if exact_in_f64 {
                0.0
            } else {
                x.abs() * f64::EPSILON / 2.0
            };
            Ok((x, conversion + exp2(bound.re) + exp2(bound.im)))
        }
    }
}

/// Multiple of `ε·∫|f|` (`ε = 2⁻⁵²`) that bounds the rounding of the `f64`
/// integrand values in a quadrature sum: each value carries a few roundings
/// of its own and the condition number of its evaluation (`sin(1/x)` near
/// `x = 10⁻³` loses three digits to the rounding of `1/x`).
const QUADRATURE_ROUNDING_FACTOR: f64 = 16.0;

/// Decimal digits served by [`evalf_complex64`] (`eval_f64`,
/// `eval_complex64`) for an expression containing a definite integral: the
/// cheap `f64` Gauss–Kronrod estimate is returned when its error bound
/// certifies this many digits (it seldom certifies the 16 of the rest of the
/// `f64` route: its own rounding floor is `50ε` per segment); otherwise the
/// arbitrary-precision quadrature serves it, to every digit of the `f64`.
/// Before 0.29 it was served at a tolerance of `10⁻¹⁰`, accepted up to
/// `10⁻⁷`, with no check at all.
pub(crate) const QUADRATURE_F64_DIGITS: u32 = 8;

/// The highest working precision at which a definite integral is tried by
/// `f64` quadrature first: `evalf`'s minimum, the precision of every
/// request up to 18 digits.
const QUADRATURE_F64_MAX_PREC: usize = 128;

/// A definite integral whose bounds exceed its width by more than `2^this`
/// (or round to one `f64`) is evaluated by its closed form first, when it
/// has one: an `f64` node has fewer than 5 bits inside such a range, and
/// the arbitrary-precision ones need this many bits more than the working
/// precision.  Up to 0.51 the quadrature of `∫ |cos(7x + 1/2)| dx` over
/// `[10¹⁰⁰ + 1/3, 10¹⁰⁰ + 28/3]` spent 17 s before refusing.
const FAR_RANGE_BITS: f64 = 48.0;

/// `∫_lo^hi body d(var)` with an error bound: the quadratures of
/// [`eval_definite_integral_numeric`], and when they refuse (the integrand
/// has no bound at the nodes, the quadrature does not converge) the closed
/// form of the definite integrator evaluated at the working precision, or
/// its proof of divergence (`calculus::definite::closed_form`, sought once
/// per integral; once found it serves the higher precisions directly).
/// Up to 0.51 `eval_f64` of `∫ |sin(x/3 + 1/2)| dx` over
/// `[10¹⁰⁰ + 7/10, 10¹⁰⁰ + 100]` was refused (in 16 s at 0.50): the
/// nodes need 330 bits before the first fractional one of `x/3`.
#[allow(clippy::too_many_arguments)]
pub(super) fn eval_definite_integral(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    // The value of a closed form, its proof of divergence, or `None`.
    let serve = |closed: &quad::ClosedForm, w: &mut Wp<'_>| match &**closed {
        Ok(Some((pa, value))) => Some(closed_form_ball(pa, *value, w)),
        Err(divergent) => Some(Err(divergent.clone())),
        Ok(None) => None,
    };
    let seek = || crate::calculus::definite::closed_form(arena, body, var, lo, hi);
    let key = quad::memo_key(arena, body, var, lo, hi);
    // Bounds far out next to the width: every node needs that many bits
    // more than the working precision, and the closed form comes first.
    let far = match (
        definite_bound_f64(arena, cache, lo),
        definite_bound_f64(arena, cache, hi),
    ) {
        (Ok((a, _)), Ok((b, _))) if a.is_finite() && b.is_finite() => {
            a == b || (a.abs().max(b.abs()) / (b - a).abs()).log2() > FAR_RANGE_BITS
        }
        _ => false,
    };
    let known = if far {
        quad::closed_form(key, seek)
    } else {
        quad::recall_closed_form(&key)
    };
    if let Some(closed) = known
        && let Some(served) = serve(&closed, w)
        && (served.is_ok() || matches!(served, Err(SymplexError::Divergent { .. })))
    {
        return served;
    }
    let refused = match eval_definite_integral_numeric(arena, body, var, lo, hi, cache, w) {
        Err(e) if !matches!(e, SymplexError::Divergent { .. }) => e,
        done => return done,
    };
    let Some(closed) = quad::closed_form(key, seek) else {
        return Err(refused);
    };
    match serve(&closed, w) {
        Some(Ok(ball)) => Ok(ball),
        Some(Err(e @ SymplexError::Divergent { .. })) => Err(e),
        _ => Err(refused),
    }
}

/// The closed form `value` (in the private arena `pa`) of a definite
/// integral as a ball for `w`'s precision: evaluated at that precision, and
/// at twice as many bits, and so on to the configured maximum, while it
/// certifies fewer bits than that (or holds 0).  A closed form may cancel
/// where the quadrature it replaces did not — `∫ sin(10²⁰x) dx` over
/// `[10²⁰, 10²⁰ + 2·10⁻¹⁹]` is `(cos(10⁴⁰) − cos(10⁴⁰ + 20))/10²⁰` — and
/// a ball around 0 within the digits asked is read as an integral equal to
/// 0 ([`evaluate_adaptive`]): in a draft, `eval_f64` gave `0` for
/// `−1.0065408055519122·10⁻²⁰`.
fn closed_form_ball(pa: &Arena, value: ExprId, w: &mut Wp<'_>) -> Result<CBall, SymplexError> {
    let order = walk::post_order_ids(pa, value);
    let target = w.bits();
    let cap = (pa.config.max_evalf_precision as usize).max(target);
    let mut bits = target;
    loop {
        let ball = evaluate_tree(pa, value, &order, None, false, &mut w.at(bits))?;
        let enough = ball.err.is_exact()
            || accuracy::accurate_bits(&ball.mid, ball.err.joint())
                .is_some_and(|a| a >= target as f64);
        if enough || bits >= cap {
            return Ok(ball);
        }
        bits = (2 * bits).min(cap);
    }
}

/// `∫_lo^hi body d(var)` with an error bound by quadrature.
///
/// At the lowest working precision ([`QUADRATURE_F64_MAX_PREC`]) the `f64`
/// adaptive Gauss–Kronrod quadrature is tried first
/// ([`eval_definite_integral_f64`]) and its value served when it resolves
/// the integral to [`QUADRATURE_F64_DIGITS`] digits: `Ex::eval_f64` of a
/// smooth integrand stays a matter of microseconds.  Anything else — more
/// digits, an endpoint singularity the `f64` bisection cannot resolve, an
/// integrand the `f64` compiler does not support — goes to the
/// arbitrary-precision tanh-sinh quadrature (`quad`): the body and the
/// bounds are copied into a private arena (the bounds re-evaluated at every
/// precision a node needs; one that depends on an enclosing bound variable
/// keeps its cached value), interior breakpoints are located there by the
/// definite integrator's scanner, and the pieces are integrated to
/// `2^−(prec − 48)·∫|f|`.  At the lowest precision, on `[0, ∞)`, `(−∞, 0]`
/// and the whole line (where its nodes keep their distance to the finite
/// end), the `f64` quadrature under the endpoint-smoothing substitution
/// (its raw error estimates resolved) is the fallback when the
/// arbitrary-precision one fails or certifies less — an oscillating tail
/// tanh-sinh does not resolve, `∫₀^∞ sin x·e^{−√x}/(2√x) dx` — and the
/// latter then gets a quarter of its work.  The DiracDelta of the variable is refused: 0 at every node,
/// its integral is no sum of values (before 0.43 `∫₋₁¹ δ(x) dx` was `0`).
///
/// An `f64` value over an infinite range is served only when the integrand's
/// mass beyond `|x| = 2¹⁶`, measured by certified evaluations out to
/// `2²⁰⁴⁸`, is within its bound or was seen by the compiled body
/// (`improper::f64_tail_trusted`), and the smoothed one never when the
/// arbitrary-precision quadrature found the terms not decaying: an `f64`
/// body whose intermediate values overflow is 0 beyond the overflow, and
/// before 0.47 `∫_{−∞}^{∞} |x|/((x² + 1)π) dx`, which diverges, was
/// `225.93`.  An integral refused for terms that do not decay is
/// [`SymplexError::Divergent`] when the comparison tests prove it
/// (`improper::divergence`).
///
/// Before 0.43 an integral was refused beyond 16
/// digits, an endpoint singularity was certified to 8 or 9 digits with a
/// bound its error exceeded (`∫₀¹ x⁴e^{x/10}/(√(1 − x)·B(5, ½)) dx`,
/// mpmath `1.0952385618059876`, was `1.0952385480452327` from `eval_f64`
/// with a bound below `10⁻⁹`), `∫₀¹ x^(−9/10) dx` did not converge, and
/// an integral equal to 0 was refused.
#[allow(clippy::too_many_arguments)]
fn eval_definite_integral_numeric(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<CBall, SymplexError> {
    let prec = w.bits();
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return Err(SymplexError::Unevaluable {
            reason: "integration variable of a definite integral must be a symbol".into(),
        });
    }
    // `DiracDelta` is 0 at every node a quadrature can sample: its integral
    // is not a sum of values.
    if walk::post_order_ids(arena, body).iter().any(|&id| {
        matches!(arena.node(id), ExprNode::DiracDelta(inner) if walk::contains(arena, *inner, var))
    }) {
        return Err(SymplexError::Unevaluable {
            reason: "a definite integral of DiracDelta of the integration variable has no \
                     numerical value (integrate it symbolically)"
                .into(),
        });
    }
    if prec > QUADRATURE_F64_MAX_PREC {
        return eval_definite_integral_bigfloat(arena, body, var, lo, hi, cache, w, false)
            .map_err(SymplexError::from);
    }
    if let Some(served) = eval_definite_integral_f64(arena, body, var, lo, hi, cache, w, false)? {
        return Ok(served);
    }
    // An endpoint singularity the plain bisection cannot resolve: the
    // arbitrary-precision quadrature, unless it fails or certifies less than
    // the smoothed `f64` one (an oscillating tail it does not resolve,
    // `∫₀^∞ sin x·e^{−√x}/√x dx`), with less work when the latter exists.
    let fallback = eval_definite_integral_f64(arena, body, var, lo, hi, cache, w, true)?;
    let big =
        eval_definite_integral_bigfloat(arena, body, var, lo, hi, cache, w, fallback.is_some());
    match (big, fallback) {
        (Ok(b), Some(f)) if b.err.joint() > f.err.joint() => Ok(f),
        (Ok(b), _) => Ok(b),
        // Terms that do not decay far beyond the reach of `f64` (out to
        // `x ≈ 10¹¹⁰⁰⁰⁰`, or next to a breakpoint): the `f64` value did not
        // see them.
        (Err(e), Some(_))
            if matches!(
                e,
                quad::QuadError::NotDecaying
                    | quad::QuadError::Failed(SymplexError::Divergent { .. })
            ) =>
        {
            Err(e.into())
        }
        (Err(_), Some(f)) => Ok(f),
        (Err(e), None) => Err(e.into()),
    }
}

/// A bound of a definite integral for `quad`: `None` for `±∞` (with its
/// sign), otherwise a finite end in the private arena `pa`.
fn quadrature_end(
    arena: &Arena,
    id: ExprId,
    cache: Cache<'_>,
    pa: &mut Arena,
    map: &mut FxHashMap<ExprId, ExprId>,
) -> Result<Result<quad::End, bool>, SymplexError> {
    match arena.node(id) {
        ExprNode::Infinity => return Ok(Err(true)),
        ExprNode::NegInfinity => return Ok(Err(false)),
        ExprNode::Num(n) => return Ok(Ok(quad::End::Rational(arena.num(*n).clone()))),
        _ => {}
    }
    if walk::free_symbols(arena, id).is_empty() {
        let p = crate::base::compact::transfer_subtree(arena, pa, id, map);
        return Ok(Ok(quad::End::Expr(p)));
    }
    // A bound that depends on an enclosing bound variable: its cached value.
    let v = get_cached(cache, id)?;
    let b = cache.err(&id);
    if b.is_unknown() || !accuracy::exact_zero(&v.1, b.im) || v.0.is_inf() || v.0.is_nan() {
        return Err(SymplexError::Unevaluable {
            reason: "integration bounds of a definite integral must be real".into(),
        });
    }
    Ok(Ok(quad::End::Value(v.0.clone(), b.re)))
}

/// The arbitrary-precision route of [`eval_definite_integral`].
#[allow(clippy::too_many_arguments)]
fn eval_definite_integral_bigfloat(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
    light: bool,
) -> Result<CBall, quad::QuadError> {
    let prec = w.bits();
    debug!(
        prec,
        light, "evalf: DefiniteIntegral — tanh-sinh quadrature"
    );
    let key = quad::memo_key(arena, body, var, lo, hi);
    if let Some(recalled) = quad::recall(&key, prec) {
        debug!(
            prec,
            "evalf: a quadrature that did not converge at a lower precision"
        );
        let (value, err, real) = recalled?;
        return Ok(quadrature_value(value, err, real, prec));
    }
    let mut pa = Arena::with_config(arena.config.clone());
    let mut map: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let body_p = crate::base::compact::transfer_subtree(arena, &mut pa, body, &mut map);
    let var_p = crate::base::compact::transfer_subtree(arena, &mut pa, var, &mut map);
    let lo_e = quadrature_end(arena, lo, cache, &mut pa, &mut map)?;
    let hi_e = quadrature_end(arena, hi, cache, &mut pa, &mut map)?;
    let exact_zero = || Ok(CBall::exact(c_zero(prec)));
    let probe = prec + 32;
    let mut q = quad::Quad::new(&pa, body_p, var_p, Vec::new(), prec);
    // The range from below to above (`None`: −∞ below, +∞ above), and
    // whether it was reversed.
    let (lo_e, hi_e, negate) = match (lo_e, hi_e) {
        (Err(a), Err(b)) if a == b => return exact_zero(),
        (Err(false), Err(_)) => (None, None, false),
        (Err(_), Err(_)) => (None, None, true),
        (Ok(a), Err(true)) => (Some(a), None, false),
        (Err(false), Ok(b)) => (None, Some(b), false),
        (Err(true), Ok(b)) => (Some(b), None, true),
        (Ok(a), Err(false)) => (None, Some(a), true),
        (Ok(a), Ok(b)) => {
            if quad::same_end(&a, &b) {
                return exact_zero();
            }
            let (av, _) = q.end_value(&a, &mut w.at(probe))?;
            let (bv, _) = q.end_value(&b, &mut w.at(probe))?;
            match av.cmp(&bv) {
                Some(c) if c < 0 => (Some(a), Some(b), false),
                Some(c) if c > 0 => (Some(b), Some(a), true),
                // Equal to the precision: no value yet.
                _ => return Ok(CBall::new(c_zero(prec), accuracy::Bound::UNKNOWN)),
            }
        }
    };
    let mut f64_of = |e: &Option<quad::End>, inf: f64| -> Result<f64, SymplexError> {
        match e {
            Some(e) => f64_of(&q.end_value(e, &mut w.at(probe))?.0),
            None => Ok(inf),
        }
    };
    let lo_f = f64_of(&lo_e, f64::NEG_INFINITY)?;
    let hi_f = f64_of(&hi_e, f64::INFINITY)?;
    drop(q);
    // Interior breakpoints, located in the private arena.
    let range = crate::base::interval::Interval::closed(lo_f, hi_f);
    let points = quadrature_breakpoints(
        &mut pa,
        body_p,
        var_p,
        lo_e.as_ref(),
        hi_e.as_ref(),
        range,
        prec,
    )?;
    debug!(
        lo_f,
        hi_f,
        breakpoints = points.len(),
        "evalf: quadrature pieces"
    );
    // The body shifted to each bound and breakpoint `e` other than 0,
    // `body(e ± x)`: the nodes next to `e` are placed at their distance from
    // it, which an algebraic or logarithmic singularity at `e` then sees
    // exactly (`(x − 1/3)^(−9/10)` becomes `x^(−9/10)`) instead of through a
    // cancellation that needs ten times the precision.
    let mut variants: Vec<(quad::End, bool, ExprId)> = Vec::new();
    let mut shift = |pa: &mut Arena, e: &quad::End, plus: bool| {
        let anchor = match e {
            quad::End::Rational(q) if !q.is_zero() => pa.num_ratio(q.clone()),
            quad::End::Expr(id) => *id,
            _ => return,
        };
        let moved = if plus {
            pa.add(&[anchor, var_p])
        } else {
            pa.sub(anchor, var_p)
        };
        let shifted = crate::transforms::subs::subs(pa, body_p, var_p, moved);
        variants.push((e.clone(), plus, shifted));
    };
    if let Some(a) = &lo_e {
        shift(&mut pa, a, true);
    }
    if let Some(b) = &hi_e {
        shift(&mut pa, b, false);
    }
    for p in &points {
        shift(&mut pa, p, false);
        shift(&mut pa, p, true);
    }
    // Where a refusal is examined for a proof of divergence: the ends and
    // both sides of every breakpoint.
    let mut approaches: Vec<improper::Approach> = Vec::new();
    let anchor = |pa: &mut Arena, e: &quad::End| match e {
        quad::End::Rational(q) => Some(pa.num_ratio(q.clone())),
        quad::End::Expr(id) => Some(*id),
        _ => None,
    };
    match &lo_e {
        None => approaches.push(improper::Approach::NegInf),
        Some(a) => approaches.extend(anchor(&mut pa, a).map(improper::Approach::Above)),
    }
    match &hi_e {
        None => approaches.push(improper::Approach::PosInf),
        Some(b) => approaches.extend(anchor(&mut pa, b).map(improper::Approach::Below)),
    }
    for p in &points {
        if let Some(c) = anchor(&mut pa, p) {
            approaches.push(improper::Approach::Below(c));
            approaches.push(improper::Approach::Above(c));
        }
    }
    let pieces = quad::pieces(lo_e, hi_e, points);
    let mut q = quad::Quad::new(&pa, body_p, var_p, variants, prec);
    if light {
        q.lighten();
    }
    let run = q.run(pieces, w.cc);
    drop(q);
    let out = match run {
        Ok(out) => out,
        Err(e) => {
            // Terms that do not decay, or an integrand without a value at a
            // far node (`eˣ` overflows): a proof of divergence is sought.
            let e = if e.may_diverge() {
                match improper::divergence(&mut pa, body_p, var_p, &approaches) {
                    Some(reason) => quad::QuadError::Failed(SymplexError::Divergent {
                        operation: "evalf",
                        reason: format!("the definite integral diverges: {reason}"),
                    }),
                    None => e,
                }
            } else {
                e
            };
            let divergent = matches!(e, quad::QuadError::Failed(SymplexError::Divergent { .. }));
            if (!light || divergent) && e.is_node_failure() {
                quad::remember(key, prec, Err(e.clone()));
            }
            return Err(e);
        }
    };
    let value = if negate { c_neg(&out.value) } else { out.value };
    if !light && !out.converged && out.discretisation {
        quad::remember(key, prec, Ok((value.clone(), out.err, out.real)));
    }
    debug!(
        err = out.err,
        out.real, out.converged, "evalf: DefiniteIntegral evaluated"
    );
    Ok(quadrature_value(value, out.err, out.real, prec))
}

/// A quadrature's sum `value` with the error bound `2^err` as a node value
/// at `prec` bits: real when every integrand value was.  A sum of exact
/// zeros is no proof of an exact zero: its bound is `2^(−4·prec)`.
fn quadrature_value(mut value: Complex, err: accuracy::ErrExp, real: bool, prec: usize) -> CBall {
    let err = if accuracy::is_exact(err) {
        -(4.0 * prec as f64)
    } else {
        err
    };
    if real {
        value.1 = BigFloat::new(prec);
    }
    let bound = accuracy::Bound {
        re: err,
        im: if real { accuracy::EXACT } else { err },
    };
    CBall::new(value, bound)
}

/// The interior breakpoints of `body` strictly between the ends `lo` and
/// `hi` (`None`: infinite), whose values are `range`, for `quad`: the
/// definite integrator's breakpoint scan between the exact ends (the `f64`
/// ones when an end is not exact), the kinks it cannot place located to
/// `prec + 16` bits.  An error for a family of more than 10,000 poles,
/// kinks or steps, or kinks that cannot be located.
fn quadrature_breakpoints(
    pa: &mut Arena,
    body: ExprId,
    var: ExprId,
    lo: Option<&quad::End>,
    hi: Option<&quad::End>,
    range: crate::base::interval::Interval<f64>,
    prec: usize,
) -> Result<Vec<quad::End>, SymplexError> {
    use crate::base::interval::Interval;
    let lo_x = lo.and_then(|e| e.exact(pa));
    let hi_x = hi.and_then(|e| e.exact(pa));
    let exact = lo_x.zip(hi_x).map(|(l, h)| Interval::closed(l, h));
    let bits = u32::try_from(prec.saturating_add(16)).unwrap_or(u32::MAX);
    let points =
        crate::calculus::definite::quadrature_breakpoints(pa, body, var, range, exact, Some(bits))?;
    Ok(points
        .into_iter()
        .map(|(p, _)| match pa.as_num(p) {
            Some(r) => quad::End::Rational(r.clone()),
            None => quad::End::Expr(p),
        })
        .collect())
}

/// `∫_lo^hi body d(var)` by adaptive `f64` Gauss–Kronrod quadrature
/// (`quadrature_resolved` of the definite integrator), with an honest error
/// bound, when it resolves the integral to [`QUADRATURE_F64_DIGITS`] digits;
/// `None` otherwise (the arbitrary-precision quadrature takes over).
///
/// The body is compiled to a stack-VM function of the integration variable
/// (no arena mutation) and the bounds come from the cache (`±∞` allowed).
/// The quadrature is asked for the accuracy the working precision wants,
/// down to `10⁻¹³` relative (below that it only runs into its own
/// rounding floor).  Only plain bisection is used, and only a run whose
/// every segment was resolved is served: the endpoint-smoothing retry of
/// `quadrature` gives no bound at an algebraic endpoint singularity.  The
/// reported bound is the sum of
///
/// * the Gauss–Kronrod error estimate,
/// * the rounding of the `f64` integrand values,
///   [`QUADRATURE_ROUNDING_FACTOR`]`·ε·∫|f|` — `∫|f|`, not `|∫f|`, so an
///   oscillating integrand whose integral cancels is not credited with
///   digits its values do not have,
/// * the rounding of finite bounds to `f64` (and their own error) times the
///   largest integrand value seen,
///
/// so [`evaluate_adaptive`] refuses the digits the quadrature does not
/// have.  Before 0.29 the result was trusted to the working precision less
/// 16 bits: `eval_decimal(16)` of `∫₀¹ xˣ dx` printed
/// `0.7834305107121128` (mpmath: `0.78343051071213441…`), three wrong
/// digits.
#[allow(clippy::too_many_arguments)]
fn eval_definite_integral_f64(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
    smoothing: bool,
) -> Result<Option<CBall>, SymplexError> {
    let prec = w.bits();
    use crate::calculus::definite::{QuadOpts, QuadResult, quadrature, quadrature_resolved};
    debug!("evalf: DefiniteIntegral — f64 Gauss–Kronrod quadrature");
    let var_name = match arena.node(var) {
        ExprNode::Symbol(sid) => arena.symbol_name(*sid).to_string(),
        _ => return Ok(None),
    };
    let (Ok((a, a_err)), Ok((b, b_err))) = (
        definite_bound_f64(arena, cache, lo),
        definite_bound_f64(arena, cache, hi),
    ) else {
        return Ok(None);
    };
    // Across a kink or a step the estimate is no bound: split at them
    // (all located), or leave the integral to the arbitrary-precision
    // quadrature.
    let kinks = if crate::calculus::definite::has_kink(arena, body, var) {
        match crate::calculus::definite::f64_pieces(arena, body, var, a.min(b), a.max(b)) {
            Some(p) if !smoothing => Some(p),
            _ => return Ok(None),
        }
    } else {
        None
    };
    // The smoothed run is trusted only where its nodes keep their distance
    // to a finite end, which in `f64` they do next to 0 alone (see
    // `quadrature_resolved`): on `[0, ∞)`, `(−∞, 0]` and the whole line.
    // (Float evidence: an end is 0 only with an exact error, `a_err == 0`;
    // every other case refuses.)
    let half_lines = (a == 0.0 && b == f64::INFINITY)
        || (a == f64::NEG_INFINITY && b == 0.0)
        || (a == 0.0 && b == f64::NEG_INFINITY)
        || (a == f64::INFINITY && b == 0.0)
        || (a.is_infinite() && b.is_infinite());
    if smoothing && (!half_lines || a_err != 0.0 || b_err != 0.0) {
        return Ok(None);
    }
    let Ok(func) = crate::output::lambdify::compile_raw(arena, body, &[&var_name]) else {
        return Ok(None);
    };
    let peak = std::cell::Cell::new(0.0f64);
    // The pieces between the cuts: whether a node lay in each, and the
    // largest value its nodes gave (see below).
    let cuts: &[f64] = kinks.as_ref().map_or(&[], |k| k.cuts.as_slice());
    let sights = std::cell::RefCell::new(vec![(false, 0.0f64); cuts.len() + 1]);
    let f = |t: f64| {
        let v = func(&[t]);
        if v.is_finite() {
            peak.set(peak.get().max(v.abs()));
        }
        let i = cuts.partition_point(|&c| c < t);
        if !cuts.is_empty()
            && cuts.get(i) != Some(&t)
            && let Ok(mut sights) = sights.try_borrow_mut()
            && let Some((visited, largest)) = sights.get_mut(i)
        {
            *visited = true;
            *largest = largest.max(v.abs());
        }
        v
    };
    // ∫|f|, to the accuracy a rounding estimate needs: the rounding floor
    // of the quadrature sum, below which its tolerance is pointless.
    let abs_f = |t: f64| func(&[t]).abs();
    let loose = QuadOpts {
        rel_tol: 1e-3,
        abs_tol: 0.0,
        max_subdivisions: 400,
    };
    let l1 = quadrature(&abs_f, a, b, &loose).map_or(f64::INFINITY, |r| r.value + r.error);
    let floor = QUADRATURE_ROUNDING_FACTOR * f64::EPSILON * l1;
    let default = QuadOpts::default();
    let wanted = 2f64.powi(
        -i32::try_from(prec.saturating_sub(64))
            .unwrap_or(i32::MAX)
            .min(1000),
    );
    let opts = QuadOpts {
        rel_tol: wanted.max(1e-13).min(default.rel_tol),
        abs_tol: if floor.is_finite() {
            floor
        } else {
            default.abs_tol
        },
        ..default
    };
    let run = match &kinks {
        Some(k) => crate::calculus::definite::quadrature_resolved_split(
            &f,
            a.min(b),
            a.max(b),
            &k.cuts,
            &opts,
        )
        .map(|r| {
            r.map(|r| QuadResult {
                value: if a > b { -r.value } else { r.value },
                ..r
            })
        }),
        None => quadrature_resolved(&f, a, b, &opts, smoothing),
    };
    let Ok(Some(QuadResult { value, error })) = run else {
        return Ok(None);
    };
    // A piece every node of which gave 0 adds `0 ± 0` to the sum, and the
    // sum's bound, which the other pieces make positive, hides it.  What
    // the integrand holds next to the ends of such a piece is in the bound:
    // nothing where it is an exact 0 there (the 0 branch of a `Piecewise`)
    // or below the exponent range (a far tail), as the arbitrary-precision
    // quadrature counts a piece none of whose nodes saw a value; a first-
    // order bound of the layer otherwise (`improper::held_at_ends`).  Up to
    // 0.60.0 the bound was tested on the sum alone (found by reading, with
    // the defect of `Ex::integrate_numeric_with`, which lost a piece that
    // way: see there; no wrong value served here is on record).
    // (Float evidence: a largest value of 0 proves nothing, it only sends
    // the piece to the certified test; every other case is as before.)
    let gave_value = |largest: f64| largest > 0.0;
    let (low, high) = (a.min(b), a.max(b));
    let unseen: Vec<usize> = sights
        .into_inner()
        .into_iter()
        .enumerate()
        .filter(|(_, (visited, largest))| *visited && !gave_value(*largest))
        .map(|(i, _)| i)
        .collect();
    let mut unseen_held = 0.0f64;
    for i in unseen {
        let from = i
            .checked_sub(1)
            .and_then(|k| cuts.get(k))
            .map_or(low, |c| *c);
        let to = cuts.get(i).map_or(high, |c| *c);
        let held = improper::held_at_ends(arena, body, var, from, to, w.cc);
        if accuracy::is_unknown(held) {
            debug!(
                from,
                to, "evalf: a piece of the f64 quadrature gave 0 at every node, unread at its ends"
            );
            return Ok(None);
        }
        unseen_held += held.exp2();
    }
    let rounding = QUADRATURE_ROUNDING_FACTOR * f64::EPSILON * l1.max(value.abs());
    let endpoints = (a_err + b_err) * peak.get();
    let misplaced = kinks.as_ref().map_or(0.0, |k| k.misplaced) * peak.get();
    let total = error + rounding + endpoints + misplaced + unseen_held;
    debug!(
        value,
        error, rounding, endpoints, "evalf: DefiniteIntegral evaluated in f64"
    );
    // Served only when it certifies the digits of `eval_f64`; a zero bound
    // (a zero integrand at every node) is no proof of an exact zero.  (Both
    // tests only refuse.)
    let needed = f64::from(QUADRATURE_F64_DIGITS) * 3.322 + 4.0;
    if !value.is_finite()
        || !total.is_finite()
        || total <= 0.0
        || total > value.abs() * (-needed).exp2()
    {
        return Ok(None);
    }
    // Beyond the reach of the compiled body (an overflow cuts it off, and
    // no node lies past `f64::MAX`): the integrand's own far tail.
    let compiled = |t: f64| func(&[t]);
    let tail_seen = |upper: bool, cc: &mut Consts| {
        improper::f64_tail_trusted(arena, body, var, upper, &compiled, total, cc)
    };
    if (high == f64::INFINITY && !tail_seen(true, w.cc))
        || (low == f64::NEG_INFINITY && !tail_seen(false, w.cc))
    {
        debug!("evalf: the f64 quadrature cannot have seen the far tail");
        return Ok(None);
    }
    Ok(Some(CBall::new(
        (bigfloat::from_f64(value, prec), BigFloat::new(prec)),
        accuracy::Bound::real(total.log2() + 1e-12),
    )))
}
