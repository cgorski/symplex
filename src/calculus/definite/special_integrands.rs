//! Special integrands: δ, H, |·|, sign, Piecewise, floor (part of [`definite`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Special integrands: δ, H, |·|, sign, Piecewise, floor
// ═══════════════════════════════════════════════════════════════════════════

/// Top-level factor list of `f`.
pub(super) fn factors_of(arena: &Arena, f: ExprId) -> Vec<ExprId> {
    match arena.node(f) {
        ExprNode::Mul(children) => children.to_vec(),
        _ => vec![f],
    }
}

/// Handle integrands containing distributions / piecewise constructs.
pub(super) fn try_special_integrand(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
    depth: usize,
) -> Option<Result<ExprId, SymplexError>> {
    let factors = factors_of(arena, f);

    // ── DiracDelta factor ───────────────────────────────────────────
    for (i, &fac) in factors.iter().enumerate() {
        if let ExprNode::DiracDelta(g) = arena.node(fac).clone()
            && walk::contains(arena, g, x)
        {
            let rest: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &c)| c)
                .collect();
            let rest = if rest.is_empty() {
                arena.one()
            } else {
                arena.mul(&rest)
            };
            return Some(delta_integral(arena, g, rest, x, a, b));
        }
    }

    // ── Heaviside factor ────────────────────────────────────────────
    // Of a linear argument: the interval truncated.  Any other argument is
    // split at its sign changes below, as `|·|` and `sign` are: up to 0.54
    // `H(cos(x/3 + 1/2))` over `[10¹⁰⁰, 10¹⁰⁰ + 200]` was refused here, and
    // `evalf`, which asks for this closed form first on such far bounds,
    // spent 18 s in a quadrature whose nodes next to each step lose the
    // 333 bits of `x/3`.
    for (i, &fac) in factors.iter().enumerate() {
        if let ExprNode::Heaviside(g) = arena.node(fac).clone()
            && walk::contains(arena, g, x)
            && calculus_util::linear_coeffs_exact(arena, g, x).is_some()
        {
            let rest: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &c)| c)
                .collect();
            let rest = if rest.is_empty() {
                arena.one()
            } else {
                arena.mul(&rest)
            };
            return Some(heaviside_integral(arena, g, rest, x, a, b, depth));
        }
    }

    // ── Abs / Sign / Piecewise / Floor anywhere in the tree ─────────
    if contains_piecewise_like(arena, f, x) {
        return Some(split_piecewise_like(arena, f, x, a, b, depth));
    }
    None
}

/// `∫ₐᵇ rest(x)·δ(g(x)) dx` for linear `g`.
fn delta_integral(
    arena: &mut Arena,
    g: ExprId,
    rest: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
) -> Result<ExprId, SymplexError> {
    let Some(LinearCoeffs {
        slope: alpha,
        intercept: beta,
    }) = calculus_util::linear_coeffs_exact(arena, g, x)
    else {
        return Err(failed("DiracDelta with a non-linear argument"));
    };
    if sign_of(arena, alpha) == Some(Ordering::Equal) {
        return Err(failed(
            "DiracDelta argument does not depend on the variable",
        ));
    }
    let neg_beta = arena.neg(beta);
    let c = arena.div(neg_beta, alpha);
    let c = safe_eval(arena, c);
    match position_in(arena, c, a, b) {
        Position::Inside => {
            let rc = subs::subs(arena, rest, x, c);
            let abs_alpha = arena.abs(alpha);
            let v = arena.div(rc, abs_alpha);
            Ok(safe_eval(arena, v))
        }
        Position::Below | Position::Above => Ok(arena.zero()),
        Position::AtLower | Position::AtUpper => Err(failed(
            "DiracDelta located exactly at an integration bound (value is convention-dependent)",
        )),
        Position::Unknown => Err(failed(
            "cannot decide whether the DiracDelta support lies inside the interval",
        )),
    }
}

/// `∫ₐᵇ rest(x)·H(g(x)) dx` for linear `g`: truncate the interval.
fn heaviside_integral(
    arena: &mut Arena,
    g: ExprId,
    rest: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    let Some(LinearCoeffs {
        slope: alpha,
        intercept: beta,
    }) = calculus_util::linear_coeffs_exact(arena, g, x)
    else {
        return Err(failed("Heaviside with a non-linear argument"));
    };
    let alpha_sign = match sign_of(arena, alpha) {
        Some(Ordering::Equal) | None => {
            return Err(failed("Heaviside argument slope has unknown sign"));
        }
        Some(s) => s,
    };
    let neg_beta = arena.neg(beta);
    let c = arena.div(neg_beta, alpha);
    let c = safe_eval(arena, c);
    let pos = position_in(arena, c, a, b);
    let (lo, hi) = match (alpha_sign, pos) {
        // H(x − c): support is x > c.
        (Ordering::Greater, Position::Below | Position::AtLower) => (a, b),
        (Ordering::Greater, Position::Inside) => (c, b),
        (Ordering::Greater, Position::AtUpper | Position::Above) => return Ok(arena.zero()),
        // H(c − x): support is x < c.
        (Ordering::Less, Position::Above | Position::AtUpper) => (a, b),
        (Ordering::Less, Position::Inside) => (a, c),
        (Ordering::Less, Position::Below | Position::AtLower) => return Ok(arena.zero()),
        (_, Position::Unknown) => {
            return Err(failed(
                "cannot decide whether the Heaviside step lies inside the interval",
            ));
        }
        _ => return Err(failed("unexpected Heaviside configuration")),
    };
    integrate_range(arena, rest, x, lo, hi, depth + 1)
}

/// Does `f` contain `Abs`, `Sign`, `Piecewise`, `Floor`, `Ceiling`, or
/// `Heaviside`/`DiracDelta` nodes (not as top-level factors) depending on `x`?
fn contains_piecewise_like(arena: &Arena, f: ExprId, x: ExprId) -> bool {
    let mut stack = vec![f];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Abs(inner)
            | ExprNode::Sign(inner)
            | ExprNode::Floor(inner)
            | ExprNode::Ceiling(inner)
            | ExprNode::Heaviside(inner)
                if walk::contains(arena, *inner, x) =>
            {
                return true;
            }
            ExprNode::Piecewise(_) if walk::contains(arena, id, x) => {
                return true;
            }
            _ => {}
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }
    false
}

/// Split `[a, b]` at the kinks of `f` and rewrite each piece to a smooth
/// expression by resolving `Abs`, `Sign`, `Piecewise`, `Floor`, `Ceiling`
/// and `Heaviside` at a test point of the piece.
fn split_piecewise_like(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    let range = numeric_range(arena, a, b);
    let scan = calculus_util::scan_breakpoints(arena, f, x, scan_range(a, b, range));
    if scan.opaque {
        return Err(failed(
            "the integrand contains a function whose breakpoints cannot be analysed",
        ));
    }

    // The pieces are where every argument of |·|, sign and H keeps its
    // sign: between its zeros (the kinks) and its own poles (`sign(1/x)`).
    // `solve` may miss zeros (it knew no inverse of `asinh`, and returns
    // one branch of `x·eˣ = c`); then each piece was resolved at its test
    // point as if the sign never changed, and `∫_{−1/2}^{3/2} |asinh x| dx`
    // came out as `∫ asinh x dx`.  The arguments are sampled as the
    // singularity path samples denominators, and an unexplained sign
    // change refuses the split.
    let args = kink_arguments(arena, f, x);
    let mut breaks: Vec<Breakpoint> = scan
        .points
        .iter()
        .filter(|bp| bp.kind == BreakKind::Kink)
        .cloned()
        .collect();
    let mut complete = scan.complete;
    for &g in &args {
        let s = calculus_util::scan_breakpoints(arena, g, x, scan_range(a, b, range));
        complete &= s.complete && !s.opaque;
        breaks.extend(
            s.points
                .into_iter()
                .filter(|bp| bp.kind == BreakKind::Singular),
        );
    }
    let known: Vec<f64> = breaks.iter().filter_map(|bp| bp.value).collect();
    let guard = match range {
        Some(r) => sign_changes_explained(arena, &args, x, r.lower, r.upper, &known),
        None => Truth::Unknown,
    };
    match guard {
        Truth::False => {
            return Err(failed(
                "an argument of |·|, sign or Heaviside changes sign inside the interval at a point the solver could not locate",
            ));
        }
        Truth::Unknown if !complete => {
            return Err(failed(
                "could not determine where the arguments of |·|, sign or Heaviside change sign",
            ));
        }
        _ => {}
    }
    // The guard watches the arguments of |·|, sign and H only: the steps of
    // floor/ceiling and the boundaries of Piecewise conditions count only
    // where the scan itself located all of them.  Up to 0.39 an incomplete
    // scan of `floor(x²)` (a non-linear argument) passed with an empty
    // watch list, the one piece was resolved at its midpoint, and
    // `∫₀² a·floor(x²) dx` came out as `2a` (it is `(5 − √2 − √3)·a`).
    if !steps_located(arena, f, x, scan_range(a, b, range)) {
        return Err(failed(
            "could not locate every step of floor, ceiling or a Piecewise condition inside the interval",
        ));
    }

    // Singularities other than sign changes of those arguments are handled
    // downstream.
    let mut points: Vec<(ExprId, Option<f64>)> = Vec::new();
    for bp in &breaks {
        match position_in(arena, bp.point, a, b) {
            Position::Inside if !points.iter().any(|(p, _)| *p == bp.point) => {
                points.push((bp.point, bp.value));
            }
            Position::Unknown => {
                return Err(failed(
                    "cannot decide whether a kink of the integrand lies inside the interval",
                ));
            }
            _ => {}
        }
    }
    if points.iter().any(|(_, v)| v.is_none()) && points.len() > 1 {
        return Err(failed("cannot order symbolic kinks of the integrand"));
    }
    if points.len() > MAX_PIECES {
        return Err(failed("too many kinks inside the interval"));
    }
    let points = sort_points(arena, points)
        .ok_or_else(|| failed("cannot order the kinks of the integrand"))?;

    let mut bounds: Vec<ExprId> = Vec::with_capacity(points.len() + 2);
    bounds.push(a);
    bounds.extend(points.iter().map(|(p, _)| *p));
    bounds.push(b);

    let mut total = Vec::new();
    for w in bounds.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let mid = test_point(arena, lo, hi).ok_or_else(|| {
            failed("cannot choose a test point inside a piece with symbolic bounds")
        })?;
        let smooth = resolve_at(arena, f, x, mid)?;
        if smooth == f {
            return Err(failed(
                "could not resolve piecewise structure of the integrand",
            ));
        }
        let v = integrate_range(arena, smooth, x, lo, hi, depth + 1)?;
        total.push(v);
    }
    let s = arena.add(&total);
    Ok(safe_eval(arena, s))
}

/// Did the breakpoint scan locate every step of the `floor`/`ceiling` nodes
/// and every boundary of the `Piecewise` conditions of `f` depending on
/// `x` (the breakpoints [`sign_changes_explained`] does not check)?
fn steps_located(arena: &mut Arena, f: ExprId, x: ExprId, range: Option<ScanRange>) -> bool {
    let mut probes: Vec<ExprId> = Vec::new();
    for id in walk::post_order_ids(arena, f) {
        match arena.node(id) {
            ExprNode::Floor(g) | ExprNode::Ceiling(g) if walk::contains(arena, *g, x) => {
                probes.push(id);
            }
            ExprNode::Piecewise(pairs) => {
                for &(_, cond) in pairs.iter() {
                    if walk::contains(arena, cond, x) {
                        probes.push(cond);
                    }
                }
            }
            _ => {}
        }
    }
    // A periodic composition's boundaries are one period of a principal
    // solve (`Piecewise((1, cos x > 1/2), (0, True))` on `[0, 10]` has six,
    // not two).
    probes.into_iter().all(|p| {
        let s = calculus_util::scan_breakpoints(arena, p, x, range);
        s.complete && !s.opaque && !s.periodic
    })
}

/// The `x`-dependent arguments of the `|·|`, `sign` and `Heaviside` nodes
/// of `f`, whose sign changes split the interval in
/// [`split_piecewise_like`].
fn kink_arguments(arena: &Arena, f: ExprId, x: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    for id in walk::post_order_ids(arena, f) {
        if let ExprNode::Abs(g) | ExprNode::Sign(g) | ExprNode::Heaviside(g) = *arena.node(id)
            && walk::contains(arena, g, x)
            && !out.contains(&g)
        {
            out.push(g);
        }
    }
    out
}

/// A rational point strictly inside `(lo, hi)` (infinite ends allowed).
pub(super) fn test_point(arena: &mut Arena, lo: ExprId, hi: ExprId) -> Option<ExprId> {
    // A symbolic end next to an infinite one, or two symbolic ends (whose
    // order the caller decided): `h − 1`, `l + 1`, the midpoint.  The
    // Laplace density `e^{−|x − μ|/b}` over ℝ with a real `μ` was refused up
    // to 0.46 for want of a test point in `(−∞, μ)` (SymPy's test_meijerint).
    let (lo_v, hi_v) = match (point_value(arena, lo), point_value(arena, hi)) {
        (Some(l), Some(h)) => (l, h),
        (l, h) => {
            let one = arena.one();
            return Some(match (l, h) {
                (Some(l), None) if l == f64::NEG_INFINITY => arena.sub(hi, one),
                (None, Some(h)) if h == f64::INFINITY => arena.add(&[lo, one]),
                (None, None) => {
                    let half = arena.rational(1, 2);
                    let s = arena.add(&[lo, hi]);
                    let m = arena.mul(&[half, s]);
                    safe_eval(arena, m)
                }
                _ => return None,
            });
        }
    };
    let m = match (lo_v.is_finite(), hi_v.is_finite()) {
        (true, true) => {
            if let (Some(rl), Some(rh)) = (arena.as_num(lo).cloned(), arena.as_num(hi).cloned()) {
                let two = Ratio::from_integer(BigInt::from(2));
                let mid = (rl + rh) / two;
                let nid = arena.intern_num(mid);
                return Some(arena.intern(ExprNode::Num(nid)));
            }
            let m = 0.5 * (lo_v + hi_v);
            // The binary midpoint of the values, when they show it strictly
            // inside; otherwise (ends closer than their `f64` values tell,
            // `[kπ/10²⁰, (k + 1)π/10²⁰]`) the exact midpoint.  Up to 0.49 the
            // binary one was taken anyway, outside such a piece.
            let above = |end: ExprId, v: f64, arena: &Arena| {
                separated(v, m) == Some(Ordering::Less)
                    || (arena.is_zero_structural(end) && m > 0.0)
            };
            let below = |end: ExprId, v: f64, arena: &Arena| {
                separated(m, v) == Some(Ordering::Less)
                    || (arena.is_zero_structural(end) && m < 0.0)
            };
            if !(above(lo, lo_v, arena) && below(hi, hi_v, arena)) {
                let half = arena.rational(1, 2);
                let s = arena.add(&[lo, hi]);
                let mid = arena.mul(&[half, s]);
                return Some(safe_eval(arena, mid));
            }
            m
        }
        (true, false) => lo_v + 1.0,
        (false, true) => hi_v - 1.0,
        (false, false) => 0.0,
    };
    let r = crate::base::numeric::f64_to_ratio_exact(m)?;
    let nid = arena.intern_num(r);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Replace every `Abs`, `Sign`, `Heaviside`, `Floor`, `Ceiling` and
/// `Piecewise` node depending on `x` by its smooth branch as determined at
/// the test point `mid`.
fn resolve_at(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    mid: ExprId,
) -> Result<ExprId, SymplexError> {
    let mut current = f;
    for _ in 0..64 {
        let order = walk::post_order_ids(arena, current);
        let mut target: Option<(ExprId, ExprId)> = None;
        for id in order {
            let node = arena.node(id).clone();
            let replacement = match node {
                ExprNode::Abs(inner) if walk::contains(arena, inner, x) => {
                    match sign_at(arena, inner, x, mid)? {
                        Ordering::Less => Some(arena.neg(inner)),
                        _ => Some(inner),
                    }
                }
                ExprNode::Sign(inner) if walk::contains(arena, inner, x) => {
                    match sign_at(arena, inner, x, mid)? {
                        Ordering::Less => Some(arena.neg_one()),
                        Ordering::Greater => Some(arena.one()),
                        Ordering::Equal => Some(arena.zero()),
                    }
                }
                ExprNode::Heaviside(inner) if walk::contains(arena, inner, x) => {
                    match sign_at(arena, inner, x, mid)? {
                        Ordering::Less => Some(arena.zero()),
                        Ordering::Greater => Some(arena.one()),
                        Ordering::Equal => Some(arena.rational(1, 2)),
                    }
                }
                ExprNode::Floor(inner) | ExprNode::Ceiling(inner)
                    if walk::contains(arena, inner, x) =>
                {
                    Some(integer_part_at(
                        arena,
                        inner,
                        x,
                        mid,
                        matches!(node, ExprNode::Ceiling(_)),
                    )?)
                }
                ExprNode::Piecewise(pairs) if walk::contains(arena, id, x) => {
                    let mut chosen: Option<ExprId> = None;
                    for (val, cond) in pairs.iter() {
                        match condition_at(arena, *cond, x, mid)? {
                            Truth::True => {
                                chosen = Some(*val);
                                break;
                            }
                            Truth::False => {}
                            Truth::Unknown => {
                                return Err(failed(
                                    "cannot decide a Piecewise condition at a test point",
                                ));
                            }
                        }
                    }
                    Some(chosen.unwrap_or_else(|| arena.zero()))
                }
                _ => None,
            };
            if let Some(r) = replacement {
                target = Some((id, r));
                break;
            }
        }
        match target {
            Some((old, new)) => {
                current = subs::subs(arena, current, old, new);
                current = safe_eval(arena, current);
            }
            None => return Ok(current),
        }
    }
    Err(failed("too many piecewise constructs in the integrand"))
}

/// Numeric value of `g` at `x = mid`.
fn value_at(arena: &mut Arena, g: ExprId, x: ExprId, mid: ExprId) -> Result<f64, SymplexError> {
    let gm = subs::subs(arena, g, x, mid);
    let gm = safe_eval(arena, gm);
    if !walk::free_symbols(arena, gm).is_empty() {
        return Err(failed(
            "piecewise structure depends on symbolic parameters; cannot resolve",
        ));
    }
    calculus_util::expr_to_f64(arena, gm)
        .filter(|v| v.is_finite())
        .ok_or_else(|| failed("could not evaluate a piecewise argument at a test point"))
}

/// `⌊g⌋` (`ceiling`: `⌈g⌉`) at `x = mid`, certified (`eval`'s integer part
/// of a constant).  Up to 0.49 it was the `f64` floor of the value, refused
/// beyond `10¹⁵`.
fn integer_part_at(
    arena: &mut Arena,
    g: ExprId,
    x: ExprId,
    mid: ExprId,
    ceiling: bool,
) -> Result<ExprId, SymplexError> {
    let gm = subs::subs(arena, g, x, mid);
    let gm = safe_eval(arena, gm);
    if !walk::free_symbols(arena, gm).is_empty() {
        return Err(failed(
            "piecewise structure depends on symbolic parameters; cannot resolve",
        ));
    }
    let part = if ceiling {
        arena.ceiling(gm)
    } else {
        arena.floor(gm)
    };
    let n = eval::eval(arena, part);
    match arena.as_num(n) {
        Some(r) if r.is_integer() => Ok(n),
        _ => Err(failed(
            "could not decide a floor or ceiling at a test point",
        )),
    }
}

/// Sign of `g` at `x = mid`: from its 16-digit value when that is not 0
/// (a printed digit is certified), otherwise from the certified sign test
/// (up to 0.49 a value 0 to 16 digits was a zero of `g`).
fn sign_at(arena: &mut Arena, g: ExprId, x: ExprId, mid: ExprId) -> Result<Ordering, SymplexError> {
    let gm = subs::subs(arena, g, x, mid);
    let gm = safe_eval(arena, gm);
    if !walk::free_symbols(arena, gm).is_empty() {
        return sign_of(arena, gm).ok_or_else(|| {
            failed("piecewise structure depends on symbolic parameters; cannot resolve")
        });
    }
    match value_at(arena, g, x, mid) {
        Ok(v) if v != 0.0 => return Ok(v.partial_cmp(&0.0).unwrap_or(Ordering::Equal)),
        Ok(_) | Err(_) => {}
    }
    match crate::poly::algebraic::sign_checked(arena, gm) {
        Some(s) => Ok(s.cmp(&0)),
        None => Err(failed(
            "could not decide the sign of a piecewise argument at a test point",
        )),
    }
}

/// Decide a parameter condition (no integration variable involved) with
/// the assumption system: `Unknown` if undecidable.
pub(super) fn decide_condition(arena: &mut Arena, cond: ExprId) -> Truth {
    let order = walk::post_order_ids(arena, cond);
    let mut vals: FxHashMap<ExprId, Truth> = FxHashMap::default();
    for id in order {
        let node = arena.node(id).clone();
        let v = match node {
            ExprNode::BoolTrue => Truth::True,
            ExprNode::BoolFalse => Truth::False,
            ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
                let d = arena.sub(a, b);
                let d = safe_eval(arena, d);
                let s = sign_of(arena, d);
                Truth::from(match node {
                    ExprNode::Gt(..) => s.map(|s| s == Ordering::Greater),
                    ExprNode::Ge(..) => s.map(|s| s != Ordering::Less),
                    ExprNode::Eq_(..) => s.map(|s| s == Ordering::Equal),
                    _ => s.map(|s| s != Ordering::Equal),
                })
            }
            ExprNode::And(children) => Truth::all(children.iter().map(|c| truth_of(&vals, *c))),
            ExprNode::Or(children) => Truth::any(children.iter().map(|c| truth_of(&vals, *c))),
            ExprNode::Not(inner) => truth_of(&vals, inner).not(),
            _ => Truth::Unknown,
        };
        vals.insert(id, v);
    }
    truth_of(&vals, cond)
}

/// The truth value recorded for `id`, `Unknown` when there is none.
fn truth_of(vals: &FxHashMap<ExprId, Truth>, id: ExprId) -> Truth {
    vals.get(&id).copied().unwrap_or(Truth::Unknown)
}

/// Truth value of a boolean condition at `x = mid`, three-valued: the
/// relation of `p` and `q` there is decided by the exact or certified zero
/// and sign tests of `p − q`
/// ([`is_zero_checked`](crate::poly::algebraic::is_zero_checked),
/// [`sign_checked`](crate::poly::algebraic::sign_checked)), and is
/// `Unknown` when they do not decide it (a value 0 to the precision
/// reached, an order of non-real values); so is a connective of an
/// undecided part, by Kleene's rules.  Up to 0.48 the difference was an
/// `f64`, `Eq` held when it was within `10⁻¹²` of 0, and an undecided
/// sub-condition counted as `false`.
fn condition_at(
    arena: &mut Arena,
    cond: ExprId,
    x: ExprId,
    mid: ExprId,
) -> Result<Truth, SymplexError> {
    let order = walk::post_order_ids(arena, cond);
    let mut vals: FxHashMap<ExprId, Truth> = FxHashMap::default();
    for id in order {
        let node = arena.node(id).clone();
        let v = match node {
            ExprNode::BoolTrue => Truth::True,
            ExprNode::BoolFalse => Truth::False,
            ExprNode::Gt(p, q) | ExprNode::Ge(p, q) | ExprNode::Eq_(p, q) | ExprNode::Ne(p, q) => {
                let d = arena.sub(p, q);
                let d = subs::subs(arena, d, x, mid);
                let d = safe_eval(arena, d);
                if !walk::free_symbols(arena, d).is_empty() {
                    return Err(failed(
                        "piecewise structure depends on symbolic parameters; cannot resolve",
                    ));
                }
                match node {
                    ExprNode::Gt(..) | ExprNode::Ge(..) => {
                        match crate::poly::algebraic::sign_checked(arena, d) {
                            Some(s) if matches!(node, ExprNode::Gt(..)) => Truth::from_bool(s > 0),
                            Some(s) => Truth::from_bool(s >= 0),
                            None => Truth::Unknown,
                        }
                    }
                    ExprNode::Eq_(..) => {
                        Truth::from(crate::poly::algebraic::is_zero_checked(arena, d))
                    }
                    _ => Truth::from(crate::poly::algebraic::is_zero_checked(arena, d)).not(),
                }
            }
            ExprNode::And(children) => Truth::all(children.iter().map(|c| truth_of(&vals, *c))),
            ExprNode::Or(children) => Truth::any(children.iter().map(|c| truth_of(&vals, *c))),
            ExprNode::Not(inner) => truth_of(&vals, inner).not(),
            // Relational sub-terms (numbers, symbols) are not booleans.
            _ => {
                if id == cond {
                    return Err(failed("unsupported Piecewise condition"));
                }
                Truth::Unknown
            }
        };
        vals.insert(id, v);
    }
    vals.get(&cond)
        .copied()
        .ok_or_else(|| failed("unsupported Piecewise condition"))
}
