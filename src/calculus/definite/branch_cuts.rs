//! Branch cuts of the antiderivative along the real path (part of [`definite`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Branch cuts of the antiderivative along the real path
// ═══════════════════════════════════════════════════════════════════════════

/// Where the principal branch of a function of `u` is discontinuous.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Cut {
    /// The real axis below a bound: `ln`, non-integer powers, `arg`,
    /// `loggamma`, `Ei`, `Ci`, `Chi`, `Eₙ`, `Yν`, `Kν`, incomplete gamma,
    /// `li` (0), `W` (−1/e), `acosh` (1).
    RealBelow(f64),
    /// The real axis above a bound: `polylog(s, u)` (1).
    RealAbove(f64),
    /// The real axis outside `[−1, 1]`: `asin`, `acos`, `atanh`.
    RealOutside,
    /// The imaginary axis outside `[−i, i]`: `atan`, `asinh`.
    ImaginaryOutside,
}

/// The arguments `u` (depending on `x`) of the principal-branch functions
/// in `anti`, with their cuts.
pub(super) fn cut_arguments(arena: &Arena, anti: ExprId, x: ExprId) -> Vec<(ExprId, Cut)> {
    let mut out: Vec<(ExprId, Cut)> = Vec::new();
    for (_, u, cut) in cut_nodes(arena, anti) {
        if walk::contains(arena, u, x) && !out.contains(&(u, cut)) {
            out.push((u, cut));
        }
    }
    out
}

/// The nodes of the principal-branch functions in `anti`: `(node,
/// argument, cut)`.
fn cut_nodes(arena: &Arena, anti: ExprId) -> Vec<(ExprId, ExprId, Cut)> {
    let mut out: Vec<(ExprId, ExprId, Cut)> = Vec::new();
    let mut stack = vec![anti];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = arena.node(id);
        let found = match *node {
            ExprNode::Ln(u)
            | ExprNode::Arg(u)
            | ExprNode::LogGamma(u)
            | ExprNode::Ei(u)
            | ExprNode::Ci(u)
            | ExprNode::Li(u) => Some((u, Cut::RealBelow(0.0))),
            ExprNode::Pow(u, e) if !arena.as_num(e).is_some_and(|r| r.is_integer()) => {
                Some((u, Cut::RealBelow(0.0)))
            }
            ExprNode::LambertW(u) => Some((u, Cut::RealBelow(-(-1.0f64).exp()))),
            ExprNode::Acosh(u) => Some((u, Cut::RealBelow(1.0))),
            ExprNode::Asin(u) | ExprNode::Acos(u) | ExprNode::Atanh(u) => {
                Some((u, Cut::RealOutside))
            }
            ExprNode::Atan(u) | ExprNode::Asinh(u) => Some((u, Cut::ImaginaryOutside)),
            // Before 0.46 the library functions were not looked at: the
            // dilogarithms of `∫ ln(x)·R(x) dx` (cut `[1, ∞)`) and `Chi`, `Eₙ`
            // were taken as continuous whatever their arguments did.
            ExprNode::Lib(f, ref args) => {
                use crate::base::libfn::LibFn;
                match (f, args.as_slice()) {
                    (LibFn::Chi, [u]) => Some((*u, Cut::RealBelow(0.0))),
                    (
                        LibFn::ExpInt
                        | LibFn::BesselY
                        | LibFn::BesselK
                        | LibFn::UpperGamma
                        | LibFn::LowerGamma,
                        [_, u],
                    ) => Some((*u, Cut::RealBelow(0.0))),
                    (LibFn::PolyLog, [_, u]) => Some((*u, Cut::RealAbove(1.0))),
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some((u, cut)) = found {
            out.push((id, u, cut));
        }
        node.for_each_child(|c| stack.push(c));
    }
    out
}

/// [`cut_arguments`] without the arguments whose steps across the cut
/// cancel ([`cancelling_log_steps`]): the arguments where `F` may jump
/// inside the interval.
fn crossing_arguments(arena: &mut Arena, anti: ExprId, x: ExprId) -> Vec<(ExprId, Cut)> {
    let mut out = cut_arguments(arena, anti, x);
    let cancelled = cancelling_log_steps(arena, anti);
    out.retain(|&(u, cut)| !(cut == Cut::RealBelow(0.0) && cancelled.contains(&u)));
    out
}

/// The arguments `u` of `ln`, `Ei`, `Ci`, `Chi` in `anti` whose steps
/// across the negative axis cancel: `u` is the argument of no other
/// function with a cut, of at least two of these nodes, each of them a
/// factor of a term of `anti` multiplied out, and the coefficients of
/// those terms add up to 0 exactly.  Each of these functions is `±iπ`
/// plus the same function off the axis on either side (`ln(−t ± i0) = ln t
/// ± iπ`; `Ei(u) − ln u`, `Ci(u) − ln u`, `Chi(u) − ln u` are entire), so
/// the sum steps by `2πi·Σ cⱼ = 0` across the cut.  The integrator writes
/// `Ei(u) − ln u + ln(x − c)` for an `Ei` whose argument crosses the cut
/// (`integrate::ei_continuous_on_real_line`); up to 0.52 the crossing of
/// `∫₀^∞ e⁻ˣ·sin x/(x² + 1) dx` passed the check only because it fell on a
/// sample point (`x = 1`), where samples were not compared.  (Only for
/// crossings inside an interval: on the axis itself `Ei` has its real
/// value, not the principal `ln`'s `+iπ`.)
fn cancelling_log_steps(arena: &mut Arena, anti: ExprId) -> Vec<ExprId> {
    let nodes = cut_nodes(arena, anti);
    let is_log_step = |arena: &Arena, n: ExprId| match arena.node(n) {
        ExprNode::Ln(_) | ExprNode::Ei(_) | ExprNode::Ci(_) => true,
        ExprNode::Lib(crate::base::libfn::LibFn::Chi, args) => args.len() == 1,
        _ => false,
    };
    let log_steps: Vec<(ExprId, ExprId)> = nodes
        .iter()
        .filter(|&&(n, _, _)| is_log_step(arena, n))
        .map(|&(n, u, _)| (u, n))
        .collect();
    let other: Vec<ExprId> = nodes
        .iter()
        .filter(|&&(n, _, _)| !is_log_step(arena, n))
        .map(|&(_, u, _)| u)
        .collect();
    let mut candidates: Vec<ExprId> = Vec::new();
    for &(u, _) in &log_steps {
        let nodes = log_steps.iter().filter(|&&(v, _)| v == u).count();
        if nodes >= 2 && !other.contains(&u) && !candidates.contains(&u) {
            candidates.push(u);
        }
    }
    if candidates.is_empty() {
        return candidates;
    }
    let expanded = expand::expand(arena, anti);
    let terms: Vec<ExprId> = match arena.node(expanded) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expanded],
    };
    candidates.retain(|&u| {
        let nodes: Vec<ExprId> = log_steps
            .iter()
            .filter(|&&(v, _)| v == u)
            .map(|&(_, n)| n)
            .collect();
        let mut coefficients: Vec<ExprId> = Vec::new();
        for &t in &terms {
            let factors: Vec<ExprId> = match arena.node(t) {
                ExprNode::Mul(fs) => fs.to_vec(),
                _ => vec![t],
            };
            let mut found: Option<usize> = None;
            for (k, &f) in factors.iter().enumerate() {
                if nodes.contains(&f) {
                    if found.is_some() {
                        return false;
                    }
                    found = Some(k);
                } else if nodes.iter().any(|&n| walk::contains(arena, f, n)) {
                    // Nested in another function: no simple step.
                    return false;
                }
            }
            if let Some(k) = found {
                let rest: Vec<ExprId> = factors
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != k)
                    .map(|(_, &f)| f)
                    .collect();
                coefficients.push(arena.mul(&rest));
            }
        }
        if coefficients.len() < 2 {
            return false;
        }
        let total = arena.add(&coefficients);
        let total = safe_eval(arena, total);
        arena.is_zero_structural(total)
    });
    candidates
}

/// Is `e` real at every real `x` where it is defined, under the declared
/// assumptions of the other symbols?  `x`, constants that
/// [`constant_realness_as_declared`](crate::transforms::realness::constant_realness_as_declared)
/// proves real, and sums, products, integer powers, `exp`, `sin`, `cos`,
/// `tan`, `sinh`, `cosh`, `tanh`, `atan`, `|·|` of such.  Conservative.
pub(super) fn real_for_real_x(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    reals: &mut AssumptionCache,
) -> bool {
    real_for_x_in(arena, e, x, reals, false)
}

/// [`real_for_real_x`], and with `nonnegative_x` for `x ≥ 0` only: then
/// also `x^q` for every rational `q` and `ln x` (real where defined).
fn real_for_x_in(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    reals: &mut AssumptionCache,
    nonnegative_x: bool,
) -> bool {
    let mut stack = vec![e];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if id == x {
            continue;
        }
        if !walk::contains(arena, id, x) {
            if !Truth::from(crate::transforms::realness::constant_realness_as_declared(
                arena, id, 30, reals,
            ))
            .is_true()
            {
                return false;
            }
            continue;
        }
        match arena.node(id).clone() {
            ExprNode::Add(children) | ExprNode::Mul(children) => stack.extend(children),
            ExprNode::Pow(b, e) if arena.as_num(e).is_some_and(|r| r.is_integer()) => stack.push(b),
            ExprNode::Pow(b, e) if nonnegative_x && b == x && arena.as_num(e).is_some() => {}
            ExprNode::Ln(b) if nonnegative_x && b == x => {}
            ExprNode::Neg(u)
            | ExprNode::Exp(u)
            | ExprNode::Sin(u)
            | ExprNode::Cos(u)
            | ExprNode::Tan(u)
            | ExprNode::Sinh(u)
            | ExprNode::Cosh(u)
            | ExprNode::Tanh(u)
            | ExprNode::Atan(u)
            | ExprNode::Abs(u) => stack.push(u),
            _ => return false,
        }
    }
    true
}

/// Can the argument `u` of a function with cut `cut` provably not cross
/// that cut as `x` runs along the real axis over the piece `p`, whatever
/// the values of the parameters under their declared assumptions?
///
/// - A real `u` ([`real_for_real_x`]) moves along the real axis: it may run
///   along a cut on it, where the principal value is the limit from one
///   side and continuous, but never across one.  On a piece in `[0, ∞)`
///   that includes `x^q` and `ln x` ([`real_for_x_in`]).
/// - For a cut on the real axis it suffices that `im u` does not depend on
///   `x`: `u = c + g(x)` with `g` real and `c` constant (`ln(x + a + i)`).
/// - A linear `u = α·x + β` runs along a straight segment, which meets the
///   axis of the cut (the real axis, or the imaginary one for `atan`) at
///   most once unless it lies in it.  If it meets it at an endpoint of the
///   piece, off the cut, it does not cross it inside: `∫₀¹ dx/(a·x + 1)` has
///   `u(0) = 1`, and `∫₀¹ dx/(1 + a²x²) = atan(a)/a` has `u(0) = 0`, for
///   every complex `a` (where the integral exists).
/// - For a cut `(−∞, 0]` (`ln`, powers): `u = c·g(x)` with `c` constant and
///   `g` real ([`constant_multiple_of_real`]) stays on the line through 0
///   in the direction of `c`, which meets the real axis only at the branch
///   point 0 (or lies in it).  Passing through 0 is a zero of `u`: a
///   singular point of the integrand, which the breakpoint scan splits at,
///   or a point where `F` is continuous (`u·ln u`, `u^{3/2}`).  So
///   `∫₀¹ ln(a·x + a) dx` and `∫₁² √(a²·x + a²) dx` are differences for
///   every complex `a`.
fn cut_crossing_excluded(
    arena: &mut Arena,
    u: ExprId,
    x: ExprId,
    cut: Cut,
    p: &Piece,
    reals: &mut AssumptionCache,
) -> bool {
    if real_for_real_x(arena, u, x, reals) {
        return true;
    }
    // On a piece in `[0, ∞)` a `u` built from `x^q` and `ln x` is real too:
    // `∫₁^∞ x^(−3/4)/(1 + x²) dx` has the arguments `x^(1/2) ± √(2 − √2)·x^(1/4) + 1`.
    if piece_nonnegative(arena, p, reals) && real_for_x_in(arena, u, x, reals, true) {
        return true;
    }
    if cut != Cut::ImaginaryOutside && imaginary_part_constant_polynomial(arena, u, x, reals) {
        return true;
    }
    if cut != Cut::ImaginaryOutside
        && let ExprNode::Add(children) = arena.node(u).clone()
    {
        let dependent: Vec<ExprId> = children
            .into_iter()
            .filter(|&t| walk::contains(arena, t, x))
            .collect();
        if dependent
            .into_iter()
            .all(|t| real_for_real_x(arena, t, x, reals))
        {
            return true;
        }
    }
    // A line through 0 meets the real axis only at 0: the branch point of
    // a cut `(−∞, 0]`, off a cut `[b, ∞)` with `b > 0`.
    // `b` is the table's branch point of the function, not a computed value.
    let through_origin_ok = match cut {
        Cut::RealBelow(b) => b == 0.0,
        Cut::RealAbove(b) => b > 0.0,
        _ => false,
    };
    if through_origin_ok && constant_multiple_of_real(arena, u, x, reals) {
        return true;
    }
    calculus_util::linear_coeffs_exact(arena, u, x).is_some()
        && [p.lo, p.hi]
            .into_iter()
            .any(|e| meets_axis_off_the_cut(arena, u, x, e, cut, reals))
}

/// Do both ends of the piece `p` lie in `[0, ∞]` (proved)?
fn piece_nonnegative(arena: &Arena, p: &Piece, reals: &mut AssumptionCache) -> bool {
    [p.lo, p.hi].into_iter().all(|e| {
        real_infinity(arena, e) == Some(Sign::Positive)
            || arena.as_num(e).is_some_and(|q| !q.is_negative())
            || (walk::free_symbols(arena, e).is_empty()
                && Truth::from(reals.query(arena, e, Props::NONNEGATIVE)).is_true()
                && Truth::from(reals.query(arena, e, Props::FINITE)).is_true())
    })
}

/// Is `u` a polynomial in `x` whose coefficients other than the constant
/// term are real, so that `im u` does not depend on `x`?  (`(√3/3)·(x + a) +
/// i` is not a sum with a real `x`-dependent part as written.)
pub(super) fn imaginary_part_constant_polynomial(
    arena: &mut Arena,
    u: ExprId,
    x: ExprId,
    reals: &mut AssumptionCache,
) -> bool {
    let Some(coeffs) = calculus_util::poly_coeffs_symbolic(arena, u, x) else {
        return false;
    };
    coeffs.len() >= 2
        && coeffs.into_iter().skip(1).all(|c| {
            Truth::from(crate::transforms::realness::constant_realness_as_declared(
                arena, c, 30, reals,
            ))
            .is_true()
        })
}

/// Is `u = c·g(x)` with `c` constant and `g` real for real `x`?  A product
/// whose `x`-dependent factors are real ([`real_for_real_x`]), or a
/// polynomial in `x` whose coefficients are real multiples of its leading
/// one (`a·x + a`, `a²·x² + 2a²`).
fn constant_multiple_of_real(
    arena: &mut Arena,
    u: ExprId,
    x: ExprId,
    reals: &mut AssumptionCache,
) -> bool {
    if let ExprNode::Mul(factors) = arena.node(u).clone() {
        let dependent: Vec<ExprId> = factors
            .into_iter()
            .filter(|&t| walk::contains(arena, t, x))
            .collect();
        if dependent
            .into_iter()
            .all(|t| real_for_real_x(arena, t, x, reals))
        {
            return true;
        }
    }
    let Some(coeffs) = calculus_util::poly_coeffs_symbolic(arena, u, x) else {
        return false;
    };
    let Some(&lead) = coeffs.last() else {
        return false;
    };
    if coeffs.len() < 2 || arena.is_zero_structural(lead) {
        return false;
    }
    coeffs.into_iter().all(|c| {
        let ratio = arena.div(c, lead);
        let ratio = eval::eval(arena, ratio);
        Truth::from(crate::transforms::realness::constant_realness_as_declared(
            arena, ratio, 30, reals,
        ))
        .is_true()
    })
}

/// Is `u(e)` (at a finite endpoint `e`) a constant on the axis of `cut`
/// but not on the cut itself (a branch point counts as off the cut)?
fn meets_axis_off_the_cut(
    arena: &mut Arena,
    u: ExprId,
    x: ExprId,
    e: ExprId,
    cut: Cut,
    reals: &mut AssumptionCache,
) -> bool {
    if is_infinite(arena, e) {
        return false;
    }
    let at = subs::subs(arena, u, x, e);
    let at = eval::eval(arena, at);
    if !walk::free_symbols(arena, at).is_empty() {
        return false;
    }
    // On the real axis: `u(e)` real; on the imaginary one: `i·u(e)` real.
    let on_axis = match cut {
        Cut::ImaginaryOutside => {
            let i = arena.i_unit();
            let rotated = arena.mul(&[i, at]);
            eval::eval(arena, rotated)
        }
        _ => at,
    };
    if !Truth::from(crate::transforms::realness::constant_realness_as_declared(
        arena, on_axis, 30, reals,
    ))
    .is_true()
    {
        return false;
    }
    let Some(v) = evalf::eval_const_f64(arena, on_axis) else {
        return false;
    };
    match cut {
        Cut::RealBelow(bound) => v >= bound,
        Cut::RealAbove(bound) => v <= bound,
        Cut::RealOutside | Cut::ImaginaryOutside => v.abs() <= 1.0,
    }
}

/// What [`numeric_cut_crossing`] found on a piece.
#[derive(Clone, Copy, Debug, PartialEq)]
enum CutCrossing {
    /// No crossing at the samples.
    None,
    /// A crossing of the cut near `x = t`, or `u(t)` on the cut.
    Near(f64),
    /// The side of the axis at the sample `x = t` is not decided.
    Undecided(f64),
}

/// The first sample point of `[lo, hi]` ([`guard_grid`]) where the
/// parameter-free `u` crosses `cut`: its transversal coordinate (`im u`
/// for a cut on the real axis, `re u` for one on the imaginary axis)
/// changes sign between two samples, at a point (linear interpolation) of
/// the cut, or is 0 at a sample on the cut.
///
/// The sign of the transversal coordinate at a sample is certified
/// ([`imaginary_part_sign`](crate::transforms::realness::imaginary_part_sign)),
/// an exact 0 included; a sign that does not settle is
/// [`CutCrossing::Undecided`].  Up to 0.52 a
/// coordinate below `10⁻¹²·max(1, |u|)` counted as 0 and a sample at 0
/// was skipped, so a crossing at a sample point or by a tiny imaginary
/// part went unseen: `∫₀¹ ln((1 + i)x − 1 − i/2) dx` (the argument crosses
/// the cut at the sample `x = 1/2`; `−0.5810 − 0.2461i`) was `−2.1518 −
/// 1.8169i`.
fn numeric_cut_crossing(
    arena: &mut Arena,
    u: ExprId,
    x: ExprId,
    cut: Cut,
    lo: f64,
    hi: f64,
) -> CutCrossing {
    // The last sample off the axis: (x, transversal value, along value, side).
    let mut prev: Option<(f64, f64, f64, SignSet)> = None;
    for t in guard_grid(lo, hi) {
        let Some(q) = crate::base::numeric::f64_to_ratio_exact(t) else {
            prev = None;
            continue;
        };
        let pt = arena.num_ratio(q);
        let ut = subs::subs(arena, u, x, pt);
        let ut = eval::eval(arena, ut);
        let Ok(z) = evalf::evalf_complex64(arena, ut) else {
            prev = None;
            continue;
        };
        if !z.re.is_finite() || !z.im.is_finite() {
            prev = None;
            continue;
        }
        let (s, c, transversal) = match cut {
            Cut::ImaginaryOutside => {
                // `re u = im(i·u)`.
                let i = arena.i_unit();
                let iu = arena.mul(&[i, ut]);
                (z.re, z.im, eval::eval(arena, iu))
            }
            _ => (z.im, z.re, ut),
        };
        let on_cut = |along: f64| match cut {
            Cut::RealBelow(bound) => along < bound,
            Cut::RealAbove(bound) => along > bound,
            Cut::RealOutside | Cut::ImaginaryOutside => along.abs() > 1.0,
        };
        // `evalf_complex64` gives a part other than `0.0` only when the
        // part is certified nonzero (its ball excludes 0), with that sign;
        // only a `0.0` part needs the exact test (an exact 0 or undecided).
        let side = if s > 0.0 {
            SignSet::POSITIVE
        } else if s < 0.0 {
            SignSet::NEGATIVE
        } else {
            crate::transforms::realness::imaginary_part_sign(arena, transversal)
        };
        if side == SignSet::REAL {
            return CutCrossing::Undecided(t);
        }
        if side == SignSet::ZERO {
            // On the axis: on the cut it may cross there; off it, any
            // crossing at this sample is off the cut.
            if on_cut(c) {
                return CutCrossing::Near(t);
            }
            prev = None;
            continue;
        }
        if let Some((t0, s0, c0, side0)) = prev
            && side0 != side
        {
            // The certified signs differ; the `f64` values only place the
            // crossing (a value that is 0 in `f64` puts it at its sample).
            let w = if s0 == s {
                0.5
            } else {
                (s0 / (s0 - s)).clamp(0.0, 1.0)
            };
            let along = c0 + w * (c - c0);
            if on_cut(along) {
                return CutCrossing::Near(t0 + w * (t - t0));
            }
        }
        prev = Some((t, s, c, side));
    }
    CutCrossing::None
}

/// `F` with every `atan(u)` whose argument may cross the cut of `atan`
/// ([`cut_crossing_excluded`]) but is a line `u = α·x + β` with a real
/// slope `α` written as `(i/2)·(ln(u + i) − ln(u − i))`.  That has the same
/// derivative `u′/(1 + u²)`, and its logarithms have the imaginary parts
/// `im β ± 1`, independent of `x`, so they never cross their cuts: the
/// Newton–Leibniz difference is right for every complex `β`.  (This is
/// SymPy's form: `integrate(1/(x**2 + 2*a*x + a**2 + 1), (x, 0, 1))` →
/// `I*log(a - I)/2 - I*log(a + I)/2 - I*log(a + 1 - I)/2 +
/// I*log(a + 1 + I)/2`.)  For a real `u` nothing changes.
pub(super) fn atan_of_complex_line_to_logs(
    arena: &mut Arena,
    anti: ExprId,
    x: ExprId,
    p: &Piece,
) -> ExprId {
    let mut reals = AssumptionCache::new();
    let mut out = anti;
    for (u, cut) in cut_arguments(arena, anti, x) {
        if cut != Cut::ImaginaryOutside || cut_crossing_excluded(arena, u, x, cut, p, &mut reals) {
            continue;
        }
        let atan_u = arena.atan(u);
        if !walk::contains(arena, out, atan_u) {
            continue;
        }
        let Some(LinearCoeffs { slope, .. }) = calculus_util::linear_coeffs_exact(arena, u, x)
        else {
            continue;
        };
        if !Truth::from(crate::transforms::realness::constant_realness_as_declared(
            arena, slope, 30, &mut reals,
        ))
        .is_true()
        {
            continue;
        }
        let i = arena.i_unit();
        let neg_i = arena.neg(i);
        let up = arena.add(&[u, i]);
        let um = arena.add(&[u, neg_i]);
        let ln_up = arena.ln(up);
        let ln_um = arena.ln(um);
        let diff = arena.sub(ln_up, ln_um);
        let half = arena.rational(1, 2);
        let logs = arena.mul(&[half, i, diff]);
        out = arena.subs_structural(out, atan_u, logs);
    }
    out
}

/// Refuse `F(hi) − F(lo)` when a principal branch in `F` may jump inside
/// the piece because its argument crosses the branch cut.  The
/// antiderivative is then discontinuous where the integrand is smooth, and
/// the Newton–Leibniz difference is off by the jump (a multiple of `2πi`
/// for `ln`, of `π` for `atan`).
///
/// This happens with complex constants: `∫₀¹ dx/(x² + 2a·x + a² + 1)` is
/// `atan(a + 1) − atan(a)` for a real `a`, but for `a = −1/2 + 3i/2` the
/// path `x + a` crosses the cut of `atan` at `x = 1/2` and the true value
/// (mpmath `quad`) differs from that by `π`.  An unassumed parameter is a
/// complex number in the crate's model (decision D4, 0.31), and up to 0.30
/// the difference was returned for every `a`.  (That one is now rewritten
/// first, [`atan_of_complex_line_to_logs`]; `∫₀¹ atan(x + a) dx`, whose
/// antiderivative also has `ln((x + a)² + 1)`, is refused.)
///
/// Each argument is first tried for a proof that no crossing can occur
/// ([`cut_crossing_excluded`]: real, with an imaginary part independent of
/// `x`, as `ln(x + a + i)`, or a line meeting the cut's axis off the cut at
/// an endpoint).  Otherwise an argument with parameters is refused (its
/// crossing depends on their values), and a parameter-free one is sampled
/// on the piece ([`numeric_cut_crossing`]), refused where it crosses and
/// when the piece has symbolic bounds — unless the crossing is located
/// exactly and the antiderivative continued across it
/// ([`located_crossing`]): those crossings are returned, for the
/// Newton–Leibniz difference to be taken on each side of them.
pub(super) fn check_branch_cuts(
    arena: &mut Arena,
    anti: ExprId,
    x: ExprId,
    p: &Piece,
) -> Result<Vec<CrossingAt>, SymplexError> {
    let arguments = crossing_arguments(arena, anti, x);
    let mut crossings = Vec::new();
    if arguments.is_empty() {
        return Ok(crossings);
    }
    let mut reals = AssumptionCache::new();
    let mut range: Option<Option<Interval<f64>>> = None;
    let mut ends_on_a_cut: Option<bool> = None;
    for (u, cut) in arguments {
        if cut_crossing_excluded(arena, u, x, cut, p, &mut reals) {
            continue;
        }
        let shown = arena.display(u).to_string();
        if walk::free_symbols(arena, u).iter().any(|&s| s != x) {
            return Err(failed(format!(
                "the antiderivative's branch cut at {shown} may be crossed along the path for complex values of the parameters"
            )));
        }
        let r = *range.get_or_insert_with(|| numeric_range(arena, p.lo, p.hi));
        let Some(r) = r else {
            return Err(failed(format!(
                "cannot tell whether {shown} crosses the antiderivative's branch cut between symbolic bounds"
            )));
        };
        let sampled = numeric_cut_crossing(arena, u, x, cut, r.lower, r.upper);
        // Where the samples see a crossing, or an end lies on a cut, an
        // exact crossing decides; otherwise the samples refuse where they
        // see one (and the end values where they lie on a cut).
        let at_an_end = sampled == CutCrossing::None
            && *ends_on_a_cut.get_or_insert_with(|| {
                [p.lo, p.hi]
                    .into_iter()
                    .any(|e| !is_infinite(arena, e) && endpoint_on_a_cut(arena, anti, x, e, &[]))
            });
        if sampled != CutCrossing::None || at_an_end {
            match located_crossing(arena, anti, u, x, cut, p) {
                Some(Some(c)) => {
                    crossings.push(c);
                    continue;
                }
                Some(None) => continue,
                None => {}
            }
        }
        // A polynomial or rational argument is located exactly whatever the
        // samples saw: two crossings between neighbouring samples leave no
        // trace in them.
        if let Some(found) = located_polynomial_crossings(arena, anti, u, x, cut, p, r) {
            crossings.extend(found);
            continue;
        }
        match sampled {
            CutCrossing::None => {}
            CutCrossing::Near(t) => {
                return Err(failed(format!(
                    "the antiderivative is discontinuous near x = {t:.6}, where {shown} crosses a branch cut"
                )));
            }
            CutCrossing::Undecided(t) => {
                return Err(failed(format!(
                    "cannot tell on which side of the branch cut {shown} lies at x = {t:.6}"
                )));
            }
        }
    }
    Ok(crossings)
}

/// A crossing of the cut `(−∞, 0]` by an argument of the antiderivative
/// on a piece (inside it or at an end), located exactly
/// ([`located_crossing`]).
#[derive(Clone, Copy, Debug)]
pub(super) struct CrossingAt {
    /// The real point `c` at which the argument lies on the cut.
    pub(super) at: ExprId,
    /// The argument `u`.
    pub(super) arg: ExprId,
    /// The side of the real axis `u` comes from, the sign of `im u` for `x <
    /// c` (`POSITIVE` or `NEGATIVE`); it leaves on the other.
    pub(super) before: SignSet,
}

/// Where a parameter-free linear argument `u = α·x + β` (`im α` certified
/// nonzero) of functions of `anti` with the cut `(−∞, 0]` meets the real
/// axis on the piece `p`, when [`side_value`] continues each of those
/// functions across the cut: `Some(Some(_))` at a point of the cut (`u(c)`
/// proved real and negative) inside the piece or at a finite end of it (`at`
/// is then that end), `Some(None)` when it meets the axis outside the piece
/// or off the cut (no jump), `None` when that is not decided or the
/// argument, its cut or a function of it is of another kind.  The point `c
/// = −im β/im α` is exact: the side of the axis, and so the branch, is known
/// on either side of it.  (Up to 0.53 an end on the cut was refused: `∫₀¹
/// ln((1 + i)x − 1) dx`, `−0.2146 + 2.3562i`.)
fn located_crossing(
    arena: &mut Arena,
    anti: ExprId,
    u: ExprId,
    x: ExprId,
    cut: Cut,
    p: &Piece,
) -> Option<Option<CrossingAt>> {
    if cut != Cut::RealBelow(0.0)
        || !cut_nodes(arena, anti)
            .iter()
            .filter(|&&(_, arg, _)| arg == u)
            .all(|&(n, _, _)| continued_across_cut(arena, n).is_some())
    {
        return None;
    }
    let LinearCoeffs { slope, intercept } = calculus_util::linear_coeffs_exact(arena, u, x)?;
    if !walk::free_symbols(arena, slope).is_empty()
        || !walk::free_symbols(arena, intercept).is_empty()
    {
        return None;
    }
    let slope_side = crate::transforms::realness::imaginary_part_sign(arena, slope);
    if slope_side != SignSet::POSITIVE && slope_side != SignSet::NEGATIVE {
        return None;
    }
    let im_slope = arena.im(slope);
    let im_intercept = arena.im(intercept);
    let ratio = arena.div(im_intercept, im_slope);
    let at = arena.neg(ratio);
    let at = safe_eval(arena, at);
    if walk::has_unevaluated(arena, at)
        || crate::transforms::realness::imaginary_part_sign(arena, at) != SignSet::ZERO
    {
        return None;
    }
    let at = match position_in(arena, at, p.lo, p.hi) {
        Position::Inside => at,
        Position::AtLower => p.lo,
        Position::AtUpper => p.hi,
        Position::Unknown => return None,
        Position::Below | Position::Above => return Some(None),
    };
    let on_axis = subs::subs(arena, u, x, at);
    let on_axis = safe_eval(arena, on_axis);
    if crate::transforms::realness::imaginary_part_sign(arena, on_axis) != SignSet::ZERO {
        return None;
    }
    match sign_of(arena, on_axis)? {
        Ordering::Less => {}
        Ordering::Greater => return Some(None),
        // Through the branch point: a zero of `u` at a real `x`, which the
        // breakpoint scans split at.
        Ordering::Equal => return None,
    }
    // `im u = im α·(x − c)`.
    let before = if slope_side == SignSet::POSITIVE {
        SignSet::NEGATIVE
    } else {
        SignSet::POSITIVE
    };
    Some(Some(CrossingAt { at, arg: u, before }))
}

/// Highest degree of `P·Q̄` for which [`located_polynomial_crossings`] solves
/// for the points where `im u` vanishes.
const MAX_CROSSING_DEGREE: usize = 8;

/// The crossings of the real axis by a parameter-free argument `u = P(x)/Q(x)`
/// (`P`, `Q` polynomials whose product `P·Q̄`, `Q̄` with the conjugate
/// coefficients, has rational real and imaginary parts `M` and `N`) on the
/// piece `p` (`range` its values): `im u = N/|Q|²` for real `x`, so `u` meets
/// the real axis exactly at the real roots of `N`, isolated by Sturm
/// sequences and solved exactly, and lies on the cut there when `re u =
/// M/|Q|² < 0`.  The sides are the signs of `N` between the isolating
/// intervals (rational points); a root where `N` keeps its sign (the axis
/// touched from one side) is no crossing inside the piece.  `Some(vec![])`
/// when `N` has no root in the range (proved, for any cut on the real axis),
/// the [`CrossingAt`]s of [`located_crossing`] for the cut `(−∞, 0]` when
/// every function of `u` continues across it, `None` when the argument is of
/// another form or a root is not decided (on the branch point `u = 0`, a
/// pole, not solved): the caller's samples decide as before.  Up to 0.54
/// such an argument was only sampled: `ln(x² + (1 + i)x − 1)` on `[−2, 1]`
/// was refused for its crossing at 0, and a crossing pair closer than the
/// samples went unseen.
fn located_polynomial_crossings(
    arena: &mut Arena,
    anti: ExprId,
    u: ExprId,
    x: ExprId,
    cut: Cut,
    p: &Piece,
    range: Interval<f64>,
) -> Option<Vec<CrossingAt>> {
    use crate::poly::sturm::SturmChain;
    if cut == Cut::ImaginaryOutside {
        return None;
    }
    let (num, den) = crate::poly::polybridge::as_numer_denom(arena, u);
    let pc = calculus_util::poly_coeffs_symbolic(arena, num, x)?;
    let qc = calculus_util::poly_coeffs_symbolic(arena, den, x)?;
    if pc.len() + qc.len() < 3 || pc.len() + qc.len() - 1 > MAX_CROSSING_DEGREE + 1 {
        return None;
    }
    if pc
        .iter()
        .chain(&qc)
        .any(|&c| !walk::free_symbols(arena, c).is_empty())
    {
        return None;
    }
    // `P·Q̄ = M + i·N` coefficient by coefficient.
    let len = pc.len() + qc.len() - 1;
    let mut re_c: Vec<Ratio<BigInt>> = Vec::with_capacity(len);
    let mut im_c: Vec<Ratio<BigInt>> = Vec::with_capacity(len);
    for k in 0..len {
        let mut terms = Vec::new();
        for (i, &a) in pc.iter().enumerate() {
            if let Some(&b) = k.checked_sub(i).and_then(|j| qc.get(j)) {
                let b = arena.conjugate(b);
                terms.push(arena.mul(&[a, b]));
            }
        }
        let s = arena.add(&terms);
        let s = crate::transforms::expand::expand(arena, s);
        let s = eval::eval(arena, s);
        let re = arena.re(s);
        let re = eval::eval(arena, re);
        let im = arena.im(s);
        let im = eval::eval(arena, im);
        re_c.push(arena.as_num(re)?.clone());
        im_c.push(arena.as_num(im)?.clone());
    }
    let n_poly = crate::poly::Poly::from_coeffs(im_c.clone());
    if n_poly.degree().unwrap_or(0) == 0 {
        // `im u` is a nonzero constant or `u` is real: it never crosses.
        return Some(Vec::new());
    }
    let chain = SturmChain::new(&n_poly);
    // The roots of `N` in (a rational enclosure of) the range.
    let enclose = |v: f64, below: bool| -> Option<Ratio<BigInt>> {
        if !v.is_finite() {
            return None;
        }
        let w = if below {
            v.floor() - 1.0
        } else {
            v.ceil() + 1.0
        };
        crate::base::numeric::f64_to_ratio_exact(w)
    };
    if let (Some(a), Some(b)) = (enclose(range.lower, true), enclose(range.upper, false))
        && chain.count_roots_in_closed(&a, &b) == 0
    {
        return Some(Vec::new());
    }
    if cut != Cut::RealBelow(0.0)
        || !cut_nodes(arena, anti)
            .iter()
            .filter(|&&(_, arg, _)| arg == u)
            .all(|&(n, _, _)| continued_across_cut(arena, n).is_some())
    {
        return None;
    }
    // Isolating intervals separated strictly, then a rational point
    // between each two (where `N` is not 0).
    let mut ivs = chain.isolate_all_real_roots();
    let two = Ratio::from_integer(BigInt::from(2));
    for _ in 0..64 {
        if ivs.windows(2).all(|w| w[0].upper < w[1].lower) {
            break;
        }
        ivs = ivs
            .iter()
            .map(|iv| {
                let half = (iv.upper.clone() - iv.lower.clone()) / two.clone();
                chain.refine_interval(iv, &half)
            })
            .collect();
    }
    if !ivs.windows(2).all(|w| w[0].upper < w[1].lower) {
        return None;
    }
    let one = Ratio::from_integer(BigInt::from(1));
    let mut signs: Vec<SignSet> = Vec::with_capacity(ivs.len() + 1);
    for k in 0..=ivs.len() {
        let t = match (k.checked_sub(1).and_then(|j| ivs.get(j)), ivs.get(k)) {
            (None, Some(next)) => next.lower.clone() - one.clone(),
            (Some(prev), None) => prev.upper.clone() + one.clone(),
            (Some(prev), Some(next)) => (prev.upper.clone() + next.lower.clone()) / two.clone(),
            (None, None) => return None,
        };
        let v = n_poly.eval(&t);
        signs.push(if v.is_positive() {
            SignSet::POSITIVE
        } else if v.is_negative() {
            SignSet::NEGATIVE
        } else {
            return None;
        });
    }
    let mut n_terms = Vec::with_capacity(im_c.len());
    for (k, c) in im_c.iter().enumerate() {
        let c = arena.num_ratio(c.clone());
        let e = arena.int(k as i64);
        let xk = arena.pow(x, e);
        n_terms.push(arena.mul(&[c, xk]));
    }
    let n_expr = arena.add(&n_terms);
    let roots = crate::transforms::solve::known_real_roots(arena, n_expr, x);
    if roots.len() != ivs.len() {
        return None;
    }
    let mut m_terms = Vec::with_capacity(re_c.len());
    for (k, c) in re_c.iter().enumerate() {
        let c = arena.num_ratio(c.clone());
        let e = arena.int(k as i64);
        let xk = arena.pow(x, e);
        m_terms.push(arena.mul(&[c, xk]));
    }
    let m_expr = arena.add(&m_terms);
    let mut found = Vec::new();
    for c in roots {
        if !walk::free_symbols(arena, c).is_empty() {
            return None;
        }
        // The interval holding `c`: `lower < c ≤ upper` (a point: `c` itself).
        let mut slot = None;
        for (k, iv) in ivs.iter().enumerate() {
            let lo = arena.num_ratio(iv.lower.clone());
            let hi = arena.num_ratio(iv.upper.clone());
            let above_lo = iv.lower == iv.upper || cmp_points(arena, c, lo)? == Ordering::Greater;
            if above_lo && cmp_points(arena, c, hi)? != Ordering::Greater {
                slot = Some(k);
                break;
            }
        }
        let k = slot?;
        let (before, after) = (signs[k], signs[k + 1]);
        let at = match position_in(arena, c, p.lo, p.hi) {
            Position::Inside if before == after => continue,
            Position::Inside => c,
            Position::AtLower => p.lo,
            Position::AtUpper => p.hi,
            Position::Below | Position::Above => continue,
            Position::Unknown => return None,
        };
        let re_at = subs::subs(arena, m_expr, x, c);
        let re_at = safe_eval(arena, re_at);
        match sign_of(arena, re_at)? {
            Ordering::Less => {}
            Ordering::Greater => continue,
            Ordering::Equal => return None,
        }
        // `before` is the side for `x < at`; a piece starting at `at` lies on
        // the side `after`, which [`evaluate_antiderivative`] reads as
        // `before.neg()`.
        let before = if at == p.lo { after.neg() } else { before };
        found.push(CrossingAt { at, arg: u, before });
    }
    Some(found)
}

/// How the cut function `node` of an argument `u` on the cut `(−∞, 0]`
/// continues from below the axis to a point `u = −s` of it, from the value
/// there: the principal values of `ln`, of powers, `Ci` and `Chi` are the
/// limits from above, and their continuations from below are `ln u − 2πi`,
/// `u^a·e^{−2πia}`, `Ci u − 2πi`, `Chi u − 2πi`; `Ei(−s)` is real, the mean
/// of `Ei(−s ± i0) = Ei(−s) ± iπ`.  `None` for another function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CutContinuation {
    /// `ln u`, `Ci u`, `Chi u`: above `f(−s)`, below `f(−s) − 2πi`.
    LogStep,
    /// `u^a`: above `(−s)^a`, below `(−s)^a·e^{−2πia}`.
    Power(ExprId),
    /// `Ei u`: `Ei(−s) ± iπ`.
    MeanStep,
}

/// The [`CutContinuation`] of a node of [`cut_nodes`], if known.
fn continued_across_cut(arena: &Arena, node: ExprId) -> Option<CutContinuation> {
    match *arena.node(node) {
        ExprNode::Ln(_) | ExprNode::Ci(_) => Some(CutContinuation::LogStep),
        ExprNode::Lib(crate::base::libfn::LibFn::Chi, ref args) if args.len() == 1 => {
            Some(CutContinuation::LogStep)
        }
        ExprNode::Pow(_, e) if walk::free_symbols(arena, e).is_empty() => {
            Some(CutContinuation::Power(e))
        }
        ExprNode::Ei(_) => Some(CutContinuation::MeanStep),
        _ => None,
    }
}

/// `F(c)` approached from the side of the axis given for each argument
/// crossing at `c` (`sides`: `(u, sign of im u)`), when `F` is continuous
/// there otherwise: each cut function of such a `u` is replaced by its
/// continuation from that side ([`CutContinuation`]) and `x` by `c`.
/// `None` when the value is not finite, a function has no known
/// continuation, or another argument lies on its cut at `c`
/// ([`endpoint_on_a_cut`]).
pub(super) fn side_value(
    arena: &mut Arena,
    anti: ExprId,
    x: ExprId,
    at: ExprId,
    sides: &[(ExprId, SignSet)],
) -> Option<ExprId> {
    let handled: Vec<ExprId> = sides.iter().map(|&(u, _)| u).collect();
    if endpoint_on_a_cut(arena, anti, x, at, &handled) {
        return None;
    }
    let mut f = anti;
    for &(u, side) in sides {
        let nodes: Vec<ExprId> = cut_nodes(arena, f)
            .into_iter()
            .filter(|&(_, arg, _)| arg == u)
            .map(|(n, _, _)| n)
            .collect();
        for n in nodes {
            let two_pi_i = {
                let two = arena.int(2);
                let pi = arena.pi();
                let i = arena.i_unit();
                arena.mul(&[two, pi, i])
            };
            let below = side == SignSet::NEGATIVE;
            let replacement = match continued_across_cut(arena, n)? {
                CutContinuation::LogStep if below => arena.sub(n, two_pi_i),
                CutContinuation::Power(e) if below => {
                    let turn = arena.mul(&[e, two_pi_i]);
                    let turn = arena.neg(turn);
                    let factor = arena.exp(turn);
                    arena.mul(&[n, factor])
                }
                CutContinuation::LogStep | CutContinuation::Power(_) => n,
                CutContinuation::MeanStep => {
                    let half = arena.rational(if below { -1 } else { 1 }, 2);
                    let step = arena.mul(&[half, two_pi_i]);
                    arena.add(&[n, step])
                }
            };
            if replacement != n {
                f = arena.subs_structural(f, n, replacement);
            }
        }
    }
    let v = subs::subs(arena, f, x, at);
    let v = safe_eval(arena, v);
    finite_value(arena, v, x)?;
    Some(v)
}

/// `F(hi⁻) − F(lo⁺)`.  `Ok(None)` when an endpoint value is unknown.  An
/// end at which arguments cross a branch cut (`sides[0]` at `lo`, `sides[1]`
/// at `hi`: each argument with the side of the axis it lies on inside the
/// piece) takes the value from that side ([`side_value`]).
pub(super) fn ftc(
    arena: &mut Arena,
    anti: ExprId,
    x: ExprId,
    p: &Piece,
    sides: &[Vec<(ExprId, SignSet)>; 2],
) -> Result<Option<ExprId>, SymplexError> {
    let sided = |arena: &mut Arena, at: ExprId, sides: &[(ExprId, SignSet)]| {
        side_value(arena, anti, x, at, sides).map(Extended::Finite)
    };
    let hi_val = if sides[1].is_empty() {
        endpoint_value(arena, anti, x, p.hi, Side::FromLeft, p.hi_sing)
    } else {
        sided(arena, p.hi, &sides[1])
    };
    if let Some(inf @ (Extended::PosInf | Extended::NegInf)) = hi_val {
        let shown = arena.display(p.hi).to_string();
        let dir = if inf == Extended::PosInf {
            "+∞"
        } else {
            "−∞"
        };
        return Err(divergent(format!(
            "the antiderivative tends to {dir} as x → {shown}"
        )));
    }
    let lo_val = if sides[0].is_empty() {
        endpoint_value(arena, anti, x, p.lo, Side::FromRight, p.lo_sing)
    } else {
        sided(arena, p.lo, &sides[0])
    };
    if let Some(inf @ (Extended::PosInf | Extended::NegInf)) = lo_val {
        let shown = arena.display(p.lo).to_string();
        let dir = if inf == Extended::PosInf {
            "+∞"
        } else {
            "−∞"
        };
        return Err(divergent(format!(
            "the antiderivative tends to {dir} as x → {shown}"
        )));
    }
    match (hi_val, lo_val) {
        (Some(Extended::Finite(h)), Some(Extended::Finite(l))) => {
            let d = arena.sub(h, l);
            Ok(Some(safe_eval(arena, d)))
        }
        _ => Ok(None),
    }
}

/// Without an antiderivative: try to prove divergence by comparison,
/// otherwise report failure.
pub(super) fn fallback_without_antiderivative(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    p: &Piece,
) -> Result<ExprId, SymplexError> {
    if p.hi_sing && diverges_near(arena, f, x, p.hi, Side::FromLeft) {
        let shown = arena.display(p.hi).to_string();
        return Err(divergent(format!(
            "the integrand is not integrable near x = {shown}"
        )));
    }
    if p.lo_sing && diverges_near(arena, f, x, p.lo, Side::FromRight) {
        let shown = arena.display(p.lo).to_string();
        return Err(divergent(format!(
            "the integrand is not integrable near x = {shown}"
        )));
    }
    Err(failed(
        "no antiderivative in closed form and no matching known integral",
    ))
}

/// Comparison test: near a finite `c`, `∫ f` diverges if
/// `(x − c)·f(x)` has a non-zero (or infinite) one-sided limit; at `±∞`
/// if `x·f(x)` does.
pub(super) fn diverges_near(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    c: ExprId,
    side: Side,
) -> bool {
    let u = fresh_symbol(arena, "defd", 0);
    let h = if real_infinity(arena, c) == Some(Sign::Positive) {
        let xf = arena.mul(&[x, f]);
        subs::subs(arena, xf, x, u)
    } else if real_infinity(arena, c) == Some(Sign::Negative) {
        let xf = arena.mul(&[x, f]);
        let neg_u = arena.neg(u);
        subs::subs(arena, xf, x, neg_u)
    } else {
        // x = c ± 1/u  ⇒  (x − c) f = ± f(c ± 1/u) / u
        let one = arena.one();
        let inv_u = arena.div(one, u);
        let repl = match side {
            Side::FromRight => arena.add(&[c, inv_u]),
            Side::FromLeft => arena.sub(c, inv_u),
        };
        let fu = subs::subs(arena, f, x, repl);
        let scaled = arena.mul(&[fu, inv_u]);
        match side {
            Side::FromRight => scaled,
            Side::FromLeft => arena.neg(scaled),
        }
    };
    let h = safe_eval(arena, h);
    match limit_pos_inf(arena, h, u) {
        Some(Extended::PosInf | Extended::NegInf) => true,
        // A non-real limit `L ≠ 0` diverges too: `f = L/(x − c) + o(1/|x −
        // c|)`, so `|∫ f|` grows like `|L|·ln(1/ε)` (up to 0.57 only a real
        // `L` counted, and `∫₀¹ dx/(e^{(1+i)x} − 1)` was refused).
        Some(Extended::Finite(v)) => {
            sign_of(arena, v).is_some_and(|s| s != Ordering::Equal)
                || arena
                    .part_signs(v)
                    .is_some_and(|s| s.re.is_some() || s.im.is_some())
        }
        None => nonreal_nonzero_limit(arena, h, u),
    }
}

/// Is the limit of `h` as `u → ∞` a constant with a certified nonzero
/// imaginary part (one [`limit_pos_inf`] does not take: its sanity check
/// samples a real function)?  The limit engine's exact value, confirmed by
/// two complex samples (`u = 2²⁰`, `2⁴⁰`): the distance to it shrinks and
/// ends below `2^−10` of its modulus (a guard against an engine slip, as
/// [`numeric_growth_sanity`] is for infinite limits).
fn nonreal_nonzero_limit(arena: &mut Arena, h: ExprId, u: ExprId) -> bool {
    if walk::free_symbols(arena, h).iter().any(|&s| s != u)
        || has_unbounded_oscillation(arena, h, u)
    {
        return false;
    }
    let inf = arena.infinity();
    let Ok(l) = crate::calculus::limit::limit(arena, h, u, inf) else {
        return false;
    };
    if !walk::free_symbols(arena, l).is_empty()
        || crate::base::value_class::holds_special_atom(arena, l)
        || !arena.part_signs(l).is_some_and(|s| s.im.is_some())
    {
        return false;
    }
    let Ok(lv) = evalf::evalf_complex64(arena, l) else {
        return false;
    };
    let mut distances = Vec::with_capacity(2);
    for k in [20, 40] {
        let two = arena.int(2);
        let e = arena.int(k);
        let at = arena.pow(two, e);
        let hk = subs::subs(arena, h, u, at);
        let Ok(hv) = evalf::evalf_complex64(arena, hk) else {
            return false;
        };
        distances.push((hv - lv).norm());
    }
    let scale = lv.norm();
    distances[1] <= distances[0] && distances[1] * 1024.0 <= scale && scale.is_finite()
}

/// `result` of a piece, or `Divergent` where it failed otherwise and the
/// integrand is not integrable at a singular end of the piece
/// ([`diverges_near`]): the antiderivative may have no value the engine can
/// read there (a `Piecewise` in `x`, a complex infinity), while the
/// comparison test decides.  Up to 0.57 `∫₋₁¹ polylog(0, Chi(−3/4)ˣ) dx` and
/// `∫₀¹ dx/(e^{(1+i)x} − 1)` (a pole at 0) were refused.
pub(super) fn divergent_at_singular_end(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    p: &Piece,
    result: Result<ExprId, SymplexError>,
) -> Result<ExprId, SymplexError> {
    let Err(e) = result else {
        return result;
    };
    if matches!(e, SymplexError::Divergent { .. }) {
        return Err(e);
    }
    for (end, sing, side) in [
        (p.lo, p.lo_sing, Side::FromRight),
        (p.hi, p.hi_sing, Side::FromLeft),
    ] {
        if sing && diverges_near(arena, f, x, end, side) {
            let shown = arena.display(end).to_string();
            return Err(divergent(format!(
                "the integrand is not integrable near x = {shown}"
            )));
        }
    }
    Err(e)
}
