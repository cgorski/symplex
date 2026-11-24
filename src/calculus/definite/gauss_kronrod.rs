//! Numeric quadrature: adaptive Gauss–Kronrod G7/K15 (part of [`definite`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Numeric quadrature: adaptive Gauss–Kronrod G7/K15
// ═══════════════════════════════════════════════════════════════════════════

/// Options for [`quadrature`] and `Ex::integrate_numeric_with`.
///
/// Iteration stops once the global error estimate drops below
/// `max(abs_tol, rel_tol · |value|)` or `max_subdivisions` bisections have
/// been performed.
///
/// ```
/// use symplex::definite::QuadOpts;
///
/// let opts = QuadOpts { rel_tol: 1e-8, ..QuadOpts::default() };
/// assert_eq!(opts.max_subdivisions, QuadOpts::default().max_subdivisions);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadOpts {
    /// Relative tolerance (default `1e-10`).
    pub rel_tol: f64,
    /// Absolute tolerance (default `1e-12`).
    pub abs_tol: f64,
    /// Maximum number of interval bisections (default `2000`).
    pub max_subdivisions: usize,
}

impl Default for QuadOpts {
    fn default() -> Self {
        Self {
            rel_tol: 1e-10,
            abs_tol: 1e-12,
            max_subdivisions: 2000,
        }
    }
}

/// The rule of the arbitrary-precision quadrature behind
/// `Ex::integrate_numeric_decimal` (mpmath's `quad(…, method=…)`).
/// `eval_decimal` of an unevaluated definite integral uses
/// [`QuadRule::TanhSinh`].
///
/// ```
/// use symplex::definite::QuadRule;
///
/// assert_eq!(QuadRule::default(), QuadRule::TanhSinh);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QuadRule {
    /// Tanh-sinh (double exponential): algebraic and logarithmic endpoint
    /// singularities and infinite ranges at full precision.  A finite piece
    /// it does not resolve (an oscillation) is tried by Gauss–Legendre, then
    /// bisected.
    #[default]
    TanhSinh,
    /// Gauss–Legendre of degree `3·2^(m−1)`: fewer nodes for a smooth
    /// integrand, slow at an endpoint singularity.  Infinite ranges are
    /// mapped as for tanh-sinh; a piece it does not resolve is bisected.
    GaussLegendre,
}

/// Result of [`quadrature`] and `Ex::integrate_numeric_with`: the integral
/// estimate and the estimated absolute error of that estimate.
///
/// ```
/// use symplex::definite::{quadrature, QuadOpts, QuadResult};
///
/// let QuadResult { value, error } =
///     quadrature(&|x: f64| x * x, 0.0, 1.0, &QuadOpts::default()).unwrap();
/// assert!((value - 1.0 / 3.0).abs() < 1e-12 && error < 1e-10);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadResult {
    /// The estimate of the integral.
    pub value: f64,
    /// Estimated absolute error of [`value`](Self::value): the sum of
    /// QUADPACK's estimates of the segments, a heuristic (a feature no node
    /// sees is not in it).  A segment the bisection cannot split — next to
    /// a singularity at a nonzero end or interior point, which `f64` nodes
    /// cannot approach closer than its rounding — counts at least its width
    /// times the largest value at its nodes, which covers the mass the nodes
    /// miss for `|x − c|^α` with `α` down to about `−0.99`.
    /// `Ex::integrate_numeric_with` serves an arbitrary-precision value
    /// instead when the estimate exceeds the tolerance.
    pub error: f64,
}

/// Kronrod-15 abscissae (positive half; index 7 is the centre).
const XGK: [f64; 8] = [
    0.9914553711208126,
    0.9491079123427585,
    0.8648644233597691,
    0.7415311855993945,
    0.5860872354676911,
    0.4058451513773972,
    0.20778495500789848,
    0.0,
];

/// Kronrod-15 weights.
const WGK: [f64; 8] = [
    0.022935322010529224,
    0.06309209262997856,
    0.10479001032225019,
    0.14065325971552592,
    0.1690047266392679,
    0.19035057806478542,
    0.20443294007529889,
    0.20948214108472782,
];

/// Gauss-7 weights (for abscissae `XGK[1], XGK[3], XGK[5], XGK[7]`).
const WG: [f64; 4] = [
    0.1294849661688697,
    0.27970539148927664,
    0.3818300505051189,
    0.4179591836734694,
];

/// One G7/K15 rule application on `[a, b]`.  Returns `(k15, err, peak)`;
/// `err` is `NaN` if the integrand was non-finite at a node.  The error
/// estimate is QUADPACK's `resasc·min(1, (200·|K15 − G7|/resasc)^1.5)`, or
/// with `raw` `|K15 − G7|` itself: the reduction (up to about 35-fold where
/// `|K15 − G7|` is small next to `resasc`) assumes the integrand smooth on
/// the segment, the Kronrod error far below the Gauss one.  The estimate is
/// at least `50ε` times the result, and without `raw` times `resabs`, the
/// rule applied to `|f|`, as in QUADPACK: the rounding of a sum that cancels
/// is that of its terms (before 0.44 `∫₋₂⁻¹ sin(2x/3 + 1) dx`, exactly 0,
/// was `3.9·10⁻¹⁷` with an estimate of `3·10⁻²³`).  The caller with
/// `raw` (`evalf`) adds the rounding of the values over the range itself.
/// Without `raw` the estimate also covers an end where the values grow like
/// `|x − end|^α`, `α < −¾`, between the two outermost nodes: twice the mass
/// between the outermost node and the end, `d₀|f₀|/(1 + α)`, extrapolated
/// from them, and is infinite for a faster rise (`α ≤ −1`: a divergence, or
/// an exponential the segment does not resolve, which is bisected until it
/// does) (QUADPACK leaves that to its ε-algorithm; its estimate, the
/// spread of the values, falls short of the error as `α → −1`: before 0.44
/// `∫₀¹ x^(−19/20)·sin(x/4 + 2)/(1 + 5x²) dx` was `3.7·10⁻⁹` off with an
/// estimate of `1.7·10⁻⁹`).  `peak` is `|b −
/// a|·max|f|` over the nodes, the error of a segment the bisection cannot
/// split ([`Floor`]).
fn gk15_with(f: &dyn Fn(f64) -> f64, a: f64, b: f64, raw: bool) -> (f64, f64, f64) {
    let centre = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let fc = f(centre);
    let mut k15 = fc * WGK[7];
    let mut g7 = fc * WG[3];
    let mut fv = [0.0f64; 15];
    fv[7] = fc;
    let mut nonfinite = !fc.is_finite();
    // Nodes that round onto an endpoint (possible for very narrow
    // segments) are nudged one ulp inward so endpoint singularities are
    // never sampled exactly.
    let inward = |x: f64| -> f64 {
        if x <= a {
            a.next_up()
        } else if x >= b {
            b.next_down()
        } else {
            x
        }
    };
    for j in 0..7 {
        let dx = half * XGK[j];
        let f1 = f(inward(centre - dx));
        let f2 = f(inward(centre + dx));
        nonfinite |= !f1.is_finite() || !f2.is_finite();
        fv[j] = f1;
        fv[14 - j] = f2;
        k15 += WGK[j] * (f1 + f2);
        if j % 2 == 1 {
            g7 += WG[j / 2] * (f1 + f2);
        }
    }
    if nonfinite {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let peak = fv.iter().fold(0.0f64, |m, v| m.max(v.abs())) * (b - a).abs();
    let mean = 0.5 * k15;
    let mut resasc = WGK[7] * (fc - mean).abs();
    for j in 0..7 {
        resasc += WGK[j] * ((fv[j] - mean).abs() + (fv[14 - j] - mean).abs());
    }
    let result = k15 * half;
    let resasc = resasc * half.abs();
    let mut err = ((k15 - g7) * half).abs();
    // QUADPACK's error heuristic (an `f64` estimate, judged by the caller).
    if !raw && resasc != 0.0 && err != 0.0 {
        err = resasc * (200.0 * err / resasc).powf(1.5).min(1.0);
    }
    let resabs = if raw {
        result.abs()
    } else {
        let mut s = WGK[7] * fc.abs();
        for j in 0..7 {
            s += WGK[j] * (fv[j].abs() + fv[14 - j].abs());
        }
        s * half.abs()
    };
    let resabs_floor = 50.0 * f64::EPSILON * resabs;
    if resabs_floor > err {
        err = resabs_floor;
    }
    // Nodes that round onto each other (a segment of a few hundred ulps of
    // its position): the rule sees fewer values than it weighs, K15 and G7
    // agree on them, and their difference says nothing.  Up to 0.49 such a
    // segment was "resolved": `∫_{10¹⁶}^{10¹⁶+4} |sin(x/3 + 1/2)| dx` was
    // `3.0537` (truly `2.8325`), its nodes on a grid of spacing 2.
    let nodes_distinct = (0..7).all(|j| {
        let (o, i) = (half * XGK[j], half * XGK[j + 1]);
        inward(centre - o) < inward(centre - i) && inward(centre + i) < inward(centre + o)
    });
    if !nodes_distinct {
        err = err.max(peak);
    }
    if !raw {
        // The outermost nodes lie at `(1 − XGK[0])` and `(1 − XGK[1])`
        // half-widths from the end.  A rise faster than `|x − end|⁻¹` (a
        // divergence, or an exponential the segment does not resolve, the
        // tail of `sech(10⁴x)`) bounds nothing: the segment is bisected.
        let ratio = (1.0 - XGK[1]) / (1.0 - XGK[0]);
        let d0 = half.abs() * (1.0 - XGK[0]);
        let tail = |f0: f64, f1: f64| -> f64 {
            let growth = (f0 / f1).abs();
            if f0 * f1 <= 0.0 || growth <= ratio.powf(0.75) {
                return 0.0;
            }
            let alpha = -growth.ln() / ratio.ln();
            if alpha <= -1.0 {
                return f64::INFINITY;
            }
            2.0 * d0 * f0.abs() / (1.0 + alpha)
        };
        err = err.max(tail(fv[0], fv[1])).max(tail(fv[14], fv[13]));
    }
    (result, err, peak)
}

/// Adaptive Gauss–Kronrod quadrature of `f` over `[a, b]`.
///
/// Infinite bounds are handled by the substitutions `x = a + t/(1−t)` and
/// `x = b − t/(1−t)` (the whole line split at 0); endpoint singularities
/// are absorbed by adaptive bisection (the rule never samples endpoints)
/// and, when that does not meet the tolerance, by a retry under an
/// endpoint-smoothing substitution at the ends that `f64` nodes can
/// approach: a finite end at 0 and an infinite end (see
/// [`QuadResult::error`] for what the estimate covers).
///
/// Returns the estimate and its error estimate as a [`QuadResult`], also
/// when the tolerance was not met.  If the integrand is non-finite on a
/// vanishing interval (a non-integrable interior singularity) or the
/// estimate is not a finite number, `Err(ComputationFailed)` is returned.
///
/// Before 0.44 the smoothing retry was applied to the whole range, so its
/// nodes crowded towards a nonzero finite end (and towards the end of the
/// map of an infinite range), which in `f64` they cannot approach closer
/// than half an ulp: the integrand was cut off at `1 − x ≈ 2⁻⁵³` (at `x ≈
/// 10¹⁶`) and the estimate did not see it.  `∫₀¹ dx/√(1 − x)` was
/// `1.9999999898808887` with an estimate of `3·10⁻¹⁰`, and so was
/// `∫₁^∞ x^(−3/2) dx`; `∫₀¹ log²(1 − x)/√(1 − x) dx` was `15.99998`
/// (truly 16) with an estimate of `3·10⁻⁷`.
///
/// ```
/// use symplex::definite::{quadrature, QuadOpts};
///
/// let r = quadrature(&|x: f64| x.sin(), 0.0, std::f64::consts::PI, &QuadOpts::default()).unwrap();
/// assert!((r.value - 2.0).abs() < 1e-12);
/// assert!(r.error < 1e-8);
///
/// // ∫₀^∞ e^{−x²} dx = √π / 2
/// let r = quadrature(&|x: f64| (-x * x).exp(), 0.0, f64::INFINITY, &QuadOpts::default()).unwrap();
/// assert!((r.value - std::f64::consts::PI.sqrt() / 2.0).abs() < 1e-10);
/// ```
pub fn quadrature(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
) -> Result<QuadResult, SymplexError> {
    quadrature_report(f, a, b, opts).map(|(r, _)| r)
}

/// [`quadrature`], and whether every segment was resolved: none left at the
/// narrowest width the bisection can make, where the estimate of a segment
/// is only `width·max|f|` (see [`QuadResult::error`]).
///
/// The plain bisection runs first, its segments too narrow when below
/// `10⁻¹²` of their own position ([`Floor::Relative`]: next to an end at 0
/// it goes as deep as `f64` allows).  When it is not resolved within the
/// tolerance, the retry of [`smoothed_pieces`] runs, and the result with
/// the smaller error estimate is kept.
///
/// # Errors
///
/// As [`quadrature`].
pub(crate) fn quadrature_report(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
) -> Result<(QuadResult, bool), SymplexError> {
    if a.is_nan() || b.is_nan() {
        return Err(SymplexError::InvalidArgument {
            operation: "integrate_numeric",
            reason: "integration bounds must not be NaN".into(),
        });
    }
    if a == b {
        return Ok((
            QuadResult {
                value: 0.0,
                error: 0.0,
            },
            true,
        ));
    }
    if a > b {
        let (r, resolved) = quadrature_report(f, b, a, opts)?;
        return Ok((
            QuadResult {
                value: -r.value,
                error: r.error,
            },
            resolved,
        ));
    }
    if a.is_infinite() && b.is_infinite() {
        // Split at 0 so that each half must converge on its own: a single
        // symmetric rule on t/(1−t²) would report ∫ x dx = 0 with zero
        // error (a numeric principal value).
        let half = QuadOpts {
            max_subdivisions: opts.max_subdivisions.div_ceil(2),
            ..*opts
        };
        let (left, l_ok) = quadrature_report(f, f64::NEG_INFINITY, 0.0, &half)?;
        let (right, r_ok) = quadrature_report(f, 0.0, f64::INFINITY, &half)?;
        return Ok((
            QuadResult {
                value: left.value + right.value,
                error: left.error + right.error,
            },
            l_ok && r_ok,
        ));
    }
    let plain = plain_report(f, a, b, opts, false, Floor::Relative(None));
    if let Ok((r, true)) = &plain
        && r.error <= opts.abs_tol.max(opts.rel_tol * r.value.abs())
    {
        return plain;
    }
    let Some(retry) = smoothed_pieces(f, a, b, opts, false) else {
        return plain;
    };
    match (plain, retry) {
        (Ok(p), Ok(m)) => Ok(if m.0.error < p.0.error { m } else { p }),
        (Ok(p), Err(_)) => Ok(p),
        (Err(_), Ok(m)) => Ok(m),
        (Err(e), Err(_)) => Err(e),
    }
}

/// The plain bisection of [`quadrature_report`] and [`quadrature_resolved`]
/// on a finite or half-infinite range `a < b`: on `[a, b]` itself, or on
/// `t ∈ [0, 1]` under `x = a + t/(1−t)` (`x = b − t/(1−t)`), the floor of
/// [`Floor::Relative`] judged in `x` too.
fn plain_report(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
    raw: bool,
    floor: Floor<'_>,
) -> Result<(QuadResult, bool), SymplexError> {
    let relative = matches!(floor, Floor::Relative(_));
    match (a.is_finite(), b.is_finite()) {
        (true, true) => quadrature_finite_report(f, a, b, opts, raw, floor),
        (true, false) => {
            // x = a + t/(1−t), t ∈ [0, 1), dx = dt/(1−t)²
            let g = move |t: f64| {
                let om = 1.0 - t;
                f(a + t / om) / (om * om)
            };
            let x = move |t: f64| a + t / (1.0 - t);
            let floor = if relative {
                Floor::Relative(Some(&x))
            } else {
                floor
            };
            quadrature_finite_report(&g, 0.0, 1.0, opts, raw, floor)
        }
        (false, true) => {
            // x = b − t/(1−t)
            let g = move |t: f64| {
                let om = 1.0 - t;
                f(b - t / om) / (om * om)
            };
            let x = move |t: f64| b - t / (1.0 - t);
            let floor = if relative {
                Floor::Relative(Some(&x))
            } else {
                floor
            };
            quadrature_finite_report(&g, 0.0, 1.0, opts, raw, floor)
        }
        (false, false) => Err(SymplexError::ComputationFailed {
            operation: "integrate_numeric",
            reason: "the whole real line is split at 0 before the bisection".into(),
        }),
    }
}

/// The retry of [`quadrature_report`] (and of [`quadrature_resolved`] with
/// `smoothing`): `[a, b]` split so that each end the `f64` nodes can
/// approach — a finite end at 0, an infinite end — is the end at 0 of a
/// piece in a variable of its own, integrated under the endpoint-smoothing
/// substitution of [`smoothed`]; a nonzero finite end is left to plain
/// bisection.  An infinite end becomes `u → 0⁺` under `x = e ± (1/u − 1)`
/// on the piece beyond `e ± 1`, where the nodes keep their relative
/// distance to it (under `x = e ± t/(1 − t)` they cannot come closer to `t =
/// 1` than `2⁻⁵³`, which cuts the integrand off at `x ≈ 10¹⁶`).  `None` when
/// no end is such (both finite and nonzero): the retry would be the plain
/// bisection again.
///
/// The pieces share the tolerance and the subdivisions.
fn smoothed_pieces(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
    raw: bool,
) -> Option<Result<(QuadResult, bool), SymplexError>> {
    let share = QuadOpts {
        rel_tol: 0.5 * opts.rel_tol,
        abs_tol: 0.5 * opts.abs_tol,
        max_subdivisions: opts.max_subdivisions.div_ceil(2),
    };
    // The floor judged in the piece's own variable too: where `s(t)`
    // underflows, the end at 0 is cut off (beyond `x ≈ 10³⁰⁸` for an
    // infinite end) and no estimate sees it.
    let smooth = |g: &dyn Fn(f64) -> f64, h: f64| {
        let s = smoothed(g, 0.0, h);
        let position = move |t: f64| smoothing_map(0.0, h, t);
        quadrature_finite_report(&s, 0.0, 1.0, &share, raw, Floor::Relative(Some(&position)))
    };
    let plain = |g: &dyn Fn(f64) -> f64, lo: f64, hi: f64, x: Option<&dyn Fn(f64) -> f64>| {
        quadrature_finite_report(g, lo, hi, &share, raw, Floor::Relative(x))
    };
    // The bounds pick the smoothing map; the estimate is right either way.
    let (first, second) = match (a.is_finite(), b.is_finite()) {
        (true, true) if a == 0.0 => (smooth(f, 0.5 * b), plain(f, 0.5 * b, b, None)),
        (true, true) if b == 0.0 => {
            let mirrored = move |u: f64| f(-u);
            (plain(f, a, 0.5 * a, None), smooth(&mirrored, -0.5 * a))
        }
        (true, false) | (false, true) => {
            let (e, sign) = if a.is_finite() { (a, 1.0) } else { (b, -1.0) };
            // [e, e ± 1]: x = e ± t/(1 − t), t ∈ [0, ½].
            let near = move |t: f64| {
                let om = 1.0 - t;
                f(e + sign * (t / om)) / (om * om)
            };
            // Beyond e ± 1: x = e ± (1/u − 1), u ∈ (0, ½], dx = du/u².
            let far = move |u: f64| f(e + sign * (1.0 / u - 1.0)) / u / u;
            let x = move |t: f64| e + sign * (t / (1.0 - t));
            let near_run = if e == 0.0 {
                smooth(&near, 0.5)
            } else {
                plain(&near, 0.0, 0.5, Some(&x))
            };
            (near_run, smooth(&far, 0.5))
        }
        _ => return None,
    };
    Some(first.and_then(|(p, p_ok)| {
        second.map(|(q, q_ok)| {
            (
                QuadResult {
                    value: p.value + q.value,
                    error: p.error + q.error,
                },
                p_ok && q_ok,
            )
        })
    }))
}

/// [`quadrature`] for a caller that certifies digits with the result
/// (`evalf`): `None` unless every segment was resolved — none left at the
/// narrowest width the bisection can make, where the estimate of a segment
/// next to a singularity is no bound — and with the raw error estimate
/// `|K15 − G7|` of each segment ([`gk15_with`]).  Plain bisection (its
/// segments too narrow below `10⁻¹²` of the range, [`Floor::Range`]), or
/// with `smoothing` the pieces of [`smoothed_pieces`], the endpoint-smoothing
/// substitution at an end at 0 and an infinite end only (plain bisection
/// when neither end is such).  The caller judges the error estimate: next
/// to a nonzero end `b` no node comes closer to `b` than half an ulp, so the
/// computed integrand is wrong where a singularity at `b` matters and no
/// estimate sees it.  Before 0.44 the smoothed run on `[0, ∞)` crowded its
/// nodes towards the end `t = 1` of `x = t/(1 − t)`, which cut the integrand
/// off at `x ≈ 10¹⁶`.
///
/// Before 0.43 `evalf` took [`quadrature`]'s estimate as a bound, and that
/// is how it went wrong at an endpoint singularity:
/// `∫₀¹ x⁴e^{x/10}/(√(1 − x)·B(5, ½)) dx` was `1.0952385480452327` with an
/// estimate of `4·10⁻¹⁰` (truly `1.0952385618059876`, mpmath; the error,
/// `1.4·10⁻⁸`, is that of the integrand cut off at `1 − x = 2⁻⁵³`), and
/// `∫₀¹ dx/√(1 − x)` `1.9999999898808887`.
///
/// # Errors
///
/// As [`quadrature`].
pub(crate) fn quadrature_resolved(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
    smoothing: bool,
) -> Result<Option<QuadResult>, SymplexError> {
    if a.is_nan() || b.is_nan() {
        return Err(SymplexError::InvalidArgument {
            operation: "integrate_numeric",
            reason: "integration bounds must not be NaN".into(),
        });
    }
    if f64_narrow(a, b) {
        return Ok(None);
    }
    if a == b {
        return Ok(Some(QuadResult {
            value: 0.0,
            error: 0.0,
        }));
    }
    if a > b {
        return Ok(
            quadrature_resolved(f, b, a, opts, smoothing)?.map(|r| QuadResult {
                value: -r.value,
                error: r.error,
            }),
        );
    }
    if a.is_infinite() && b.is_infinite() {
        let half = QuadOpts {
            max_subdivisions: opts.max_subdivisions.div_ceil(2),
            ..*opts
        };
        let left = quadrature_resolved(f, f64::NEG_INFINITY, 0.0, &half, smoothing)?;
        let right = quadrature_resolved(f, 0.0, f64::INFINITY, &half, smoothing)?;
        return Ok(left.zip(right).map(|(l, r)| QuadResult {
            value: l.value + r.value,
            error: l.error + r.error,
        }));
    }
    let smoothed_run = if smoothing {
        smoothed_pieces(f, a, b, opts, true)
    } else {
        None
    };
    let run = match smoothed_run {
        Some(run) => run,
        None => plain_report(f, a, b, opts, true, Floor::Range),
    };
    run.map(|(r, ok)| ok.then_some(r))
}

/// The interior breakpoints of `f` (in `x`) strictly inside `(lo, hi)` for
/// numeric quadrature, sorted: the points the definite integrator's
/// breakpoint scanner locates (poles, branch points of fractional powers
/// and logarithms, `tan` poles, kinks of `|·|`, `sign`, `Heaviside`, steps
/// and `Piecewise` boundaries), and the centres of narrow peaks — the real
/// parts of complex poles next to the range, … ([`peak_points`]) — as a
/// real constant expression and its `f64` value.  Points without a numeric
/// value are skipped; an incomplete scan gives the points it found (the
/// quadrature checks its own convergence).
///
/// `range` holds the bounds' values (`lower ≤ upper`, `±∞` allowed) and
/// `exact`, when known, the finite bounds themselves: the periodic families
/// are enumerated between them and a point is inside when that is
/// certified ([`cmp_values`]); without them the `f64` bounds stand in, as
/// the binary numbers they are.  Up to 0.49 the families were enumerated
/// from the `f64` bounds and a point within `10⁻¹²` of one was dropped: no
/// zero of `sin x` was found on `[10²⁰, 10²⁰ + 4]`.
///
/// # Errors
///
/// [`SymplexError::ComputationFailed`] when a family of poles, kinks or
/// steps has more than 10,000 members in the range (`Γ(9·10¹⁸x)` on `[−1,
/// 1]`): no quadrature resolves them, and up to 0.49 the arbitrary-precision
/// one spent minutes trying.  [`SymplexError::Divergent`] when a pole of a
/// rational factor in the range is proved not integrable
/// ([`pole_divergence_in`]), before any root is solved for.
pub(crate) fn quadrature_breakpoints(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    range: Interval<f64>,
    exact: Option<Interval<ExprId>>,
    locate: Option<u32>,
) -> Result<Vec<(ExprId, f64)>, SymplexError> {
    let (lo, hi) = (range.lower, range.upper);
    if lo < hi && has_rational_pole_factor(arena, f, x) {
        let end = |arena: &mut Arena, e: Option<ExprId>, v: f64, lower: bool| {
            if v.is_infinite() {
                Some(RationalEnd::Unbounded)
            } else {
                match e {
                    Some(e) => rational_end(arena, e, lower),
                    None => inner_rational(v, lower).map(RationalEnd::At),
                }
            }
        };
        let lower = end(arena, exact.as_ref().map(|e| e.lower), lo, true);
        let upper = end(arena, exact.as_ref().map(|e| e.upper), hi, false);
        if let (Some(lower), Some(upper)) = (lower, upper)
            && let Some(reason) = pole_divergence_in(arena, f, x, &lower, &upper)
        {
            return Err(SymplexError::Divergent {
                operation: "evalf",
                reason: format!("the definite integral diverges: {reason}"),
            });
        }
    }
    let ends = match exact {
        Some(e) => Some(e),
        None => ScanRange::from_f64(arena, range).map(|r| r.exact),
    }
    .filter(|_| lo.is_finite() && hi.is_finite());
    let scan_range = ends.and_then(|e| ScanRange::new(e.lower, e.upper, range));
    let scan = calculus_util::scan_breakpoints(arena, f, x, scan_range);
    if scan.crowded {
        return Err(failed(
            "the integrand has more than 10,000 poles, kinks or steps inside the interval",
        ));
    }
    // Kinks and steps the scan could not place (`sign(cos x²)`, `floor(x³ −
    // x)`): located numerically, or the quadrature is refused.  Up to 0.49
    // the quadrature ran across them, and its estimate, which assumes a
    // smooth integrand, missed the jumps: `∫₀³ sign(cos x² − 1/7) dx` was
    // `0.5178` from `eval_f64` (truly `0.5360`).
    // (On an infinite range the kinks are not sought: there may be
    // infinitely many, `|sin x|·e⁻ˣ` on `[0, ∞)`.)
    let mut located: Vec<(ExprId, f64)> = Vec::new();
    if let Some(bits) = locate
        && let Some(e) = ends.filter(|_| lo < hi)
        && (!scan.complete || scan.periodic)
    {
        let unplaced = unlocated_kink_arguments(arena, f, x, scan_range);
        if !unplaced.is_empty() {
            let mut budget = MAX_LOCATE_EVALS;
            for (g, crossing) in unplaced {
                located.extend(locate_crossings(
                    arena,
                    g,
                    crossing,
                    x,
                    e,
                    range,
                    bits,
                    &mut budget,
                )?);
                if located.len() > MAX_PIECES {
                    return Err(failed("too many kinks or steps inside the interval"));
                }
            }
        }
    }
    let mut points: Vec<(ExprId, Option<f64>)> = Vec::new();
    let candidates = scan
        .points
        .into_iter()
        .filter_map(|bp| bp.value.map(|v| (bp.point, v)))
        .chain(located)
        .chain(peak_points(arena, f, x, lo, hi));
    for (p, v) in candidates {
        if !v.is_finite() || !walk::free_symbols(arena, p).is_empty() {
            continue;
        }
        // Strictly inside, certified.
        let inside = match ends {
            Some(e) => {
                let (ev_lo, ev_hi) = (Some(lo), Some(hi));
                cmp_values(arena, (e.lower, ev_lo), (p, Some(v))) == Some(Ordering::Less)
                    && cmp_values(arena, (p, Some(v)), (e.upper, ev_hi)) == Some(Ordering::Less)
            }
            None => lo < v && v < hi,
        };
        if inside && !points.iter().any(|(q, _)| *q == p) {
            points.push((p, Some(v)));
        }
    }
    let points = sort_points(arena, points)
        .ok_or_else(|| failed("cannot order the breakpoints of the integrand"))?;
    Ok(points
        .into_iter()
        .filter_map(|(p, v)| v.map(|v| (p, v)))
        .collect())
}

/// What crossing of an argument `g` makes a kink or a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Crossing {
    /// `g` changes sign: `|g|`, `sign g`, `H(g)`, `min`/`max` (`g` a
    /// difference of two arguments), a condition `p > q` (`g = p − q`).
    Zero,
    /// `g` passes an integer: `floor g`, `ceiling g`.
    Integer,
}

/// The arguments, with their crossings, of the kink and step nodes of `f`
/// (depending on `x`) whose kinks a scan over `range` does not locate.
fn unlocated_kink_arguments(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    range: Option<ScanRange>,
) -> Vec<(ExprId, Crossing)> {
    let mut out: Vec<(ExprId, Crossing)> = Vec::new();
    for id in walk::post_order_ids(arena, f) {
        let args: Vec<(ExprId, Crossing)> = match arena.node(id).clone() {
            ExprNode::Abs(g) | ExprNode::Sign(g) | ExprNode::Heaviside(g) => {
                vec![(g, Crossing::Zero)]
            }
            ExprNode::Floor(g) | ExprNode::Ceiling(g) => vec![(g, Crossing::Integer)],
            ExprNode::Gt(p, q) | ExprNode::Ge(p, q) | ExprNode::Eq_(p, q) | ExprNode::Ne(p, q) => {
                vec![(arena.sub(p, q), Crossing::Zero)]
            }
            ExprNode::Min(args) | ExprNode::Max(args) => {
                let args = args.to_vec();
                let mut d = Vec::new();
                for (i, &p) in args.iter().enumerate() {
                    for &q in &args[i + 1..] {
                        d.push((arena.sub(p, q), Crossing::Zero));
                    }
                }
                d
            }
            _ => continue,
        };
        if !args.iter().any(|&(g, _)| walk::contains(arena, g, x)) {
            continue;
        }
        let s = calculus_util::scan_breakpoints(arena, id, x, range);
        if s.complete && !s.opaque && !s.periodic {
            continue;
        }
        for (g, c) in args {
            let g = safe_eval(arena, g);
            if walk::contains(arena, g, x) && !out.contains(&(g, c)) {
                out.push((g, c));
            }
        }
    }
    out
}

/// Most evaluations of kink arguments one [`quadrature_breakpoints`] call
/// spends locating crossings.
const MAX_LOCATE_EVALS: usize = 20_000;
/// Samples of a kink argument over the range.
const LOCATE_SAMPLES: usize = 256;
/// Samples of the refinement around a minimum of the distance to a level.
const LOCATE_REFINE: usize = 32;

/// The value of `h` at `x = t` for locating its crossings: the 16-digit
/// value, whose sign is certified when it is not 0; for a value that is 0
/// to 16 digits the certified sign test decides (`0.0` for an exact zero,
/// `±f64::MIN_POSITIVE` for a tiny one); `None` when `h` has no real value
/// there or its sign does not settle.
fn crossing_value(arena: &mut Arena, h: ExprId, x: ExprId, t: ExprId) -> Option<f64> {
    let v = subs::subs(arena, h, x, t);
    let v = safe_eval(arena, v);
    if !walk::free_symbols(arena, v).is_empty() {
        return None;
    }
    match calculus_util::expr_to_f64(arena, v) {
        Some(y) if y.is_finite() && y != 0.0 => Some(y),
        Some(0.0) => match crate::poly::algebraic::sign_checked(arena, v)? {
            0 => Some(0.0),
            s => Some(f64::from(s) * f64::MIN_POSITIVE),
        },
        _ => None,
    }
}

/// The crossings of `g` ([`Crossing`]) strictly inside the finite range
/// between the exact ends `ends` (values `range`, `lower < upper`), located
/// numerically: `g` sampled at [`LOCATE_SAMPLES`] points (and
/// [`LOCATE_REFINE`] more around each minimum of its distance to a level
/// that a parabola through three samples puts beyond the level) — by its
/// compiled `f64` form when it has one, a guide only —, each level passed
/// between two samples bracketed by certified signs ([`crossing_value`])
/// and narrowed (secant and bisection steps) to `2⁻ᵇⁱᵗˢ` of the range.  A
/// point returned is a crossing that the signs prove, to that accuracy; a
/// pair of crossings between two samples that no minimum reveals is
/// missed, the limit of every sampling (and of the quadrature's own
/// estimate).
///
/// # Errors
///
/// [`SymplexError::ComputationFailed`] when `g` has no real value at a
/// sample (or at an end), a sign does not settle, the samples cannot be
/// placed in `f64`, or the certified evaluations exceed `budget`.
#[allow(clippy::too_many_arguments)]
fn locate_crossings(
    arena: &mut Arena,
    g: ExprId,
    crossing: Crossing,
    x: ExprId,
    ends: Interval<ExprId>,
    range: Interval<f64>,
    bits: u32,
    budget: &mut usize,
) -> Result<Vec<(ExprId, f64)>, SymplexError> {
    let unlocatable = || failed("cannot locate the kinks or steps of the integrand");
    let (lo_f, hi_f) = (range.lower, range.upper);
    let width = hi_f - lo_f;
    // The samples: the exact ends and binary numbers between them.
    let mut ts: Vec<(ExprId, f64)> = vec![(ends.lower, lo_f)];
    for j in 1..LOCATE_SAMPLES {
        let t = lo_f + width * (j as f64) / (LOCATE_SAMPLES as f64);
        let Some(q) = crate::base::numeric::f64_to_ratio_exact(t) else {
            return Err(unlocatable());
        };
        if t <= ts.last().map_or(lo_f, |p| p.1) || t >= hi_f {
            return Err(unlocatable());
        }
        ts.push((arena.num_ratio(q), t));
    }
    ts.push((ends.upper, hi_f));
    let compiled = match arena.node(x) {
        ExprNode::Symbol(sid) => {
            let name = arena.symbol_name(*sid).to_string();
            crate::output::lambdify::compile(arena, g, &[&name]).ok()
        }
        _ => None,
    };
    // A guide to the value of `g` at the sample `(id, t)`: compiled, or
    // certified when that is not finite (or not available).
    let guide = |arena: &mut Arena, id: ExprId, t: f64, budget: &mut usize| {
        if let Some(func) = &compiled {
            let v = func(&[t]);
            if v.is_finite() {
                return Ok(v);
            }
        }
        spend(budget)?;
        crossing_value(arena, g, x, id).ok_or_else(unlocatable)
    };
    let mut vs: Vec<f64> = Vec::with_capacity(ts.len());
    for &(id, t) in &ts {
        vs.push(guide(arena, id, t, budget)?);
    }
    // The level nearest to a value.
    let nearest = |v: f64| match crossing {
        Crossing::Zero => 0.0,
        Crossing::Integer => v.round(),
    };
    // Refine around minima of the distance to the nearest level that a
    // parabola puts across it.
    let mut fine: Vec<(ExprId, f64, f64)> = Vec::new();
    for j in 1..ts.len() - 1 {
        let level = nearest(vs[j]);
        let d = |v: f64| (v - level).abs();
        if !(d(vs[j]) < d(vs[j - 1]) && d(vs[j]) < d(vs[j + 1])) {
            continue;
        }
        let side = |v: f64| v > level;
        if side(vs[j - 1]) != side(vs[j]) || side(vs[j]) != side(vs[j + 1]) {
            continue;
        }
        let (t0, t1, t2) = (ts[j - 1].1, ts[j].1, ts[j + 1].1);
        let (v0, v1, v2) = (vs[j - 1], vs[j], vs[j + 1]);
        let a = ((v2 - v1) / (t2 - t1) - (v1 - v0) / (t1 - t0)) / (t2 - t0);
        let b = (v1 - v0) / (t1 - t0) - a * (t0 + t1);
        // A flat parabola refines nothing (the guide only).
        if a == 0.0 || !a.is_finite() || !b.is_finite() {
            continue;
        }
        let tv = -b / (2.0 * a);
        let pv = v1 + (v1 - v0) / (t1 - t0) * (tv - t1) + a * (tv - t0) * (tv - t1);
        if !(t0 < tv && tv < t2) || side(pv) == side(v1) {
            continue;
        }
        for k in 1..LOCATE_REFINE {
            let t = t0 + (t2 - t0) * (k as f64) / (LOCATE_REFINE as f64);
            if t == t1 || t <= t0 || t >= t2 {
                continue;
            }
            let Some(q) = crate::base::numeric::f64_to_ratio_exact(t) else {
                continue;
            };
            let id = arena.num_ratio(q);
            let v = guide(arena, id, t, budget)?;
            fine.push((id, t, v));
        }
    }
    let mut all: Vec<(ExprId, f64, f64)> = ts
        .iter()
        .zip(&vs)
        .map(|(&(id, t), &v)| (id, t, v))
        .chain(fine)
        .collect();
    all.sort_by(|p, q| p.1.partial_cmp(&q.1).unwrap_or(Ordering::Equal));
    // Brackets: a level the guide puts between two neighbouring samples, or
    // close to one of them, confirmed by certified signs; or a sample
    // exactly on a level.
    let target = width * f64::powi(2.0, -i32::try_from(bits.min(1000)).unwrap_or(1000));
    let max_steps = 2 * bits.min(1000) + 128;
    let close = |v: f64, level: f64| (v - level).abs() <= 1e-6 * v.abs().max(level.abs()).max(1.0);
    let mut out: Vec<(ExprId, f64)> = Vec::new();
    for (k, w) in all.windows(2).enumerate() {
        let ((l, lt, lv), (r, rt, rv)) = (w[0], w[1]);
        let levels: Vec<f64> = match crossing {
            Crossing::Zero => vec![0.0],
            Crossing::Integer => {
                let (first, last) = (lv.min(rv).floor(), lv.max(rv).ceil());
                if !(first.is_finite() && last.is_finite()) || last - first > 64.0 {
                    return Err(unlocatable());
                }
                let mut ls = Vec::new();
                let mut n = first;
                while n <= last {
                    ls.push(n);
                    n += 1.0;
                }
                ls
            }
        };
        for level in levels {
            let between = (lv - level) * (rv - level) < 0.0;
            if !between && !close(lv, level) && !close(rv, level) {
                continue;
            }
            // The levels are integers, exactly.
            let h = if level == 0.0 {
                g
            } else {
                let Some(q) = crate::base::numeric::f64_to_ratio_exact(level) else {
                    return Err(unlocatable());
                };
                let n = arena.num_ratio(q);
                let h = arena.sub(g, n);
                safe_eval(arena, h)
            };
            spend(budget)?;
            spend(budget)?;
            let sl = crossing_value(arena, h, x, l).ok_or_else(unlocatable)?;
            let sr = crossing_value(arena, h, x, r).ok_or_else(unlocatable)?;
            // `crossing_value` is 0 only for a certified zero.
            if sl == 0.0 {
                // On the level: a crossing unless it is the lower end.
                if k > 0 && !out.iter().any(|&(p, _)| p == l) {
                    out.push((l, lt));
                }
                continue;
            }
            if sr == 0.0 || (sl < 0.0) == (sr < 0.0) {
                continue;
            }
            if let Some(p) = narrow_crossing(
                arena,
                h,
                x,
                (l, lt, sl),
                (r, rt, sr),
                target,
                max_steps,
                budget,
            )? {
                out.push(p);
            }
        }
    }
    Ok(out)
}

/// One evaluation of the budget of [`locate_crossings`].
fn spend(budget: &mut usize) -> Result<(), SymplexError> {
    if *budget == 0 {
        return Err(failed(
            "locating the kinks of the integrand took too many evaluations",
        ));
    }
    *budget -= 1;
    Ok(())
}

/// A crossing of `h` between `l` and `r` (each `(point, f64 value of the
/// point, f64 value of h)`, values of `h` of opposite certified signs),
/// narrowed until the bracket is narrower than `target` (at most
/// `max_steps` evaluations): secant steps (kept a sixteenth of the bracket
/// from its ends) alternating with bisections.  The midpoint of the last
/// bracket, with its value; `None` when the signs do not differ.
#[allow(clippy::too_many_arguments)]
fn narrow_crossing(
    arena: &mut Arena,
    h: ExprId,
    x: ExprId,
    (mut l, mut lt, mut lv): (ExprId, f64, f64),
    (mut r, mut rt, mut rv): (ExprId, f64, f64),
    target: f64,
    max_steps: u32,
    budget: &mut usize,
) -> Result<Option<(ExprId, f64)>, SymplexError> {
    // Values of `crossing_value`: 0 only for a certified zero.
    if (lv < 0.0) == (rv < 0.0) || lv == 0.0 || rv == 0.0 {
        return Ok(None);
    }
    for step in 0..max_steps {
        let w = arena.sub(r, l);
        let w = safe_eval(arena, w);
        let wv = calculus_util::expr_to_f64(arena, w).unwrap_or(f64::INFINITY);
        if wv.is_finite() && wv.abs() <= target {
            break;
        }
        let frac = if step % 2 == 0 {
            (lv / (lv - rv)).clamp(1.0 / 16.0, 15.0 / 16.0)
        } else {
            0.5
        };
        let Some(q) = crate::base::numeric::f64_to_ratio_exact(frac) else {
            return Err(failed("cannot locate the kinks or steps of the integrand"));
        };
        let fq = arena.num_ratio(q);
        let shift = arena.mul(&[fq, w]);
        let m = arena.add(&[l, shift]);
        let m = safe_eval(arena, m);
        spend(budget)?;
        let Some(mv) = crossing_value(arena, h, x, m) else {
            return Err(failed("cannot locate the kinks or steps of the integrand"));
        };
        let mt = lt + (rt - lt) * frac;
        if mv == 0.0 {
            let value = calculus_util::expr_to_f64(arena, m).unwrap_or(mt);
            return Ok(Some((m, value)));
        }
        if (mv < 0.0) == (lv < 0.0) {
            (l, lt, lv) = (m, mt, mv);
        } else {
            (r, rt, rv) = (m, mt, mv);
        }
    }
    let half = arena.rational(1, 2);
    let s = arena.add(&[l, r]);
    let m = arena.mul(&[half, s]);
    let m = safe_eval(arena, m);
    let value = calculus_util::expr_to_f64(arena, m).unwrap_or(0.5 * (lt + rt));
    Ok(Some((m, value)))
}

/// The centres of the narrow peaks of `f` in `(lo, hi)`, features a rule
/// over the whole range may not see: the real parts of the complex poles
/// next to the range — zeros off the real line of a polynomial raised to a
/// negative power (degree 2 to [`MAX_POLE_SPLIT_DEGREE`]) —, the vertices
/// of Gaussians `exp(a₂x² + a₁x + a₀)`, `a₂ < 0`, and the zeros of the
/// linear arguments of `sech` (`cosh` under a negative power), whose half-width
/// `w` (imaginary part, `1/√(2|a₂|)`) is below a tenth of the range (of
/// `max(1, |centre|)` on an infinite one), each with `centre ± 10w` inside
/// the range, as exact binary numbers; and the maximum of a product of
/// powers and an exponential ([`envelope_peak`]).  A split there makes a narrow peak (`1/(1 + 10⁶(x − c)²)`, `exp(−10⁶(x − c)²)`,
/// of width `10⁻³`) the end of two pieces, where the nodes crowd, instead
/// of a feature between the nodes: before 0.44 `integrate_numeric` of
/// `exp(−10⁶(x − 2/3)²)·(x + 1)²cos(4x/3)` on `[0, 2]` was `0` with an
/// error estimate of 0 (truly `0.0031`), and the arbitrary-precision
/// quadrature chased a Lorentzian of width `10⁻³` by bisection until its
/// budget ran out at 54 digits.
fn peak_points(arena: &mut Arena, f: ExprId, x: ExprId, lo: f64, hi: f64) -> Vec<(ExprId, f64)> {
    /// The polynomial a peak is read from.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Source {
        /// A denominator: its complex zeros.
        Pole,
        /// A Gaussian's exponent `a₂x² + a₁x + a₀`.
        Gaussian,
        /// The argument `l₁x + l₀` of `cosh` under a negative power
        /// (`sech(l₁x + l₀)` peaks at its zero, half-width `1/|l₁|`).
        Sech,
    }
    let mut parts: Vec<(ExprId, Source)> = walk::post_order_ids(arena, f)
        .into_iter()
        .filter_map(|id| match *arena.node(id) {
            ExprNode::Pow(base, exp)
                if arena.as_num(exp).is_some_and(Signed::is_negative)
                    && walk::contains(arena, base, x) =>
            {
                match *arena.node(base) {
                    ExprNode::Cosh(arg) => Some((arg, Source::Sech)),
                    _ => Some((base, Source::Pole)),
                }
            }
            ExprNode::Exp(arg) if walk::contains(arena, arg, x) => Some((arg, Source::Gaussian)),
            _ => None,
        })
        .collect();
    parts.sort_unstable_by(|a, b| arena.canonical_cmp(a.0, b.0).then(a.1.cmp(&b.1)));
    parts.dedup();
    let narrow = |centre: f64, half_width: f64| {
        let scale = if lo.is_finite() && hi.is_finite() {
            hi - lo
        } else {
            centre.abs().max(1.0)
        };
        // Where the quadrature splits (a guide).
        half_width != 0.0 && centre > lo && centre < hi && half_width.abs() < 0.1 * scale
    };
    // A peak at `c` of half-width `w` splits at `c` and at `c ± 10w`: the
    // nodes of a piece ending at `c` stay at a distance from it (`0.0043`
    // of its width for Gauss–Kronrod), where a Gaussian's values are 0 and
    // nothing calls for a bisection; the piece `[c − 10w, c]` sees it.
    let mut centres: Vec<f64> = Vec::new();
    let mut push = |c: f64, w: f64| {
        centres.extend(
            [c - 10.0 * w.abs(), c, c + 10.0 * w.abs()]
                .into_iter()
                .filter(|&p| p > lo && p < hi),
        );
    };
    for (part, source) in parts {
        // Expanded only when small: `(x + 1)^5000 + 2` is not.
        let max_degree = match source {
            Source::Pole => MAX_POLE_SPLIT_DEGREE,
            Source::Gaussian => 2,
            Source::Sech => 1,
        };
        if !degree_bound(arena, part, x).is_some_and(|d| d <= max_degree) {
            continue;
        }
        let Some(p) = crate::poly::polybridge::expr_to_poly(arena, part, x) else {
            continue;
        };
        let degree = p.degree().unwrap_or(0);
        let coeff = |k: usize| {
            p.coeffs()
                .get(k)
                .and_then(crate::base::numeric::ratio_to_f64)
        };
        match source {
            Source::Gaussian => {
                let (Some(a1), Some(a2)) = (coeff(1), coeff(2)) else {
                    continue;
                };
                let (c, w) = (-a1 / (2.0 * a2), (-2.0 * a2).sqrt().recip());
                if degree == 2 && a2 < 0.0 && narrow(c, w) {
                    push(c, w);
                }
            }
            Source::Sech => {
                let (Some(l0), Some(l1)) = (coeff(0), coeff(1)) else {
                    continue;
                };
                let (c, w) = (-l0 / l1, l1.recip());
                if degree == 1 && narrow(c, w) {
                    push(c, w);
                }
            }
            Source::Pole if (2..=MAX_POLE_SPLIT_DEGREE).contains(&degree) => {
                for z in crate::poly::roots::nroots_f64(&p, 128) {
                    if narrow(z.re, z.im) {
                        push(z.re, z.im);
                    }
                }
            }
            Source::Pole => {}
        }
    }
    // On an infinite range the nodes lie at a distance from the finite end
    // that grows geometrically: a peak is narrow next to that distance.
    if let Some((c, w)) = envelope_peak(arena, f, x, lo, hi) {
        let scale = match (lo.is_finite(), hi.is_finite()) {
            (true, true) => 0.4 * (hi - lo),
            (true, false) => c - lo,
            (false, true) => hi - c,
            (false, false) => c.abs().max(1.0),
        };
        if c > lo && c < hi && w < 0.25 * scale {
            push(c, w);
        }
    }
    centres
        .into_iter()
        .filter(|c| c.is_finite())
        .filter_map(|c| {
            crate::base::numeric::f64_to_ratio_exact(c).map(|q| (arena.num_ratio(q), c))
        })
        .collect()
}

/// The maximum `c` in `(lo, hi)` of a product `f` of powers `bᵉ` of terms
/// in `x` and exponentials `e^g` (constant, `sin` and `cos` factors aside),
/// with the half-width `w = 1/√(−h″(c))` of its logarithm `h = Σ e·ln|b|
/// + Σ g` there; `None` for another form, a lone Gaussian (whose vertex
/// [`peak_points`] reads exactly) or an `e^{a·x + b}` alone, and when `h`
/// has no interior maximum on the samples.  `h` is sampled in `f64` over
/// the range (at `lo + 2^(k/4)`, `|k| ≤ 400`, towards an infinite end), the
/// best sample refined by golden-section search: a split point, a guide
/// whatever its accuracy.  The densities of the distributions are of this
/// form, and their moment generating and characteristic functions have
/// their mass in a peak next to the end of `[0, ∞)`: up to 0.53
/// `eval_decimal` of the moment generating function of a Weibull
/// distribution of shape `31/2` and scale `6·10⁻⁶` at `t = 50000/3`,
/// `∫₀^∞ e^{tx}·f(x) dx` with a peak `4·10⁻⁷` wide at `6·10⁻⁶`, was refused
/// for want of precision (`1.101523493908031`, mpmath).
fn envelope_peak(arena: &mut Arena, f: ExprId, x: ExprId, lo: f64, hi: f64) -> Option<(f64, f64)> {
    /// Most nodes of an integrand looked at.
    const MAX_NODES: usize = 400;
    /// Samples of a finite range.
    const SAMPLES: i32 = 512;
    /// Quarter powers of 2 sampled towards an infinite end, either way.
    const QUARTERS: i32 = 400;
    if lo.partial_cmp(&hi) != Some(Ordering::Less)
        || walk::post_order_ids(arena, f).len() > MAX_NODES
    {
        return None;
    }
    let x_name = match arena.node(x) {
        ExprNode::Symbol(sid) => arena.symbol_name(*sid).to_string(),
        _ => return None,
    };
    let factors: Vec<ExprId> = match arena.node(f) {
        ExprNode::Mul(cs) => cs.to_vec(),
        _ => vec![f],
    };
    // `h` with `ln|b|` for its values, with `ln b` for its derivatives.
    let (mut values, mut formal) = (Vec::new(), Vec::new());
    let (mut exponentials, mut others) = (0usize, 0usize);
    for c in factors {
        if !walk::contains(arena, c, x) {
            continue;
        }
        match arena.node(c).clone() {
            ExprNode::Exp(g) => {
                // A Gaussian is read exactly by `peak_points`.
                if degree_bound(arena, g, x).is_some_and(|d| d == 2) {
                    return None;
                }
                if !degree_bound(arena, g, x).is_some_and(|d| d <= 1) {
                    others += 1;
                }
                exponentials += 1;
                values.push(g);
                formal.push(g);
            }
            ExprNode::Pow(b, e) if arena.as_num(e).is_some() => {
                let abs = arena.abs(b);
                let ln_abs = arena.ln(abs);
                values.push(arena.mul(&[e, ln_abs]));
                let ln_b = arena.ln(b);
                formal.push(arena.mul(&[e, ln_b]));
                others += 1;
            }
            ExprNode::Sin(_) | ExprNode::Cos(_) => {}
            _ if c == x => {
                let abs = arena.abs(x);
                values.push(arena.ln(abs));
                formal.push(arena.ln(x));
                others += 1;
            }
            _ => return None,
        }
    }
    if exponentials == 0 || others == 0 {
        return None;
    }
    let h = arena.add(&values);
    let hf = crate::output::lambdify::compile(arena, h, &[&x_name]).ok()?;
    let hf = |t: f64| hf(&[t]);
    // Distances from the finite end (or from 0), increasing.
    let steps: Vec<f64> = (-QUARTERS..=QUARTERS)
        .map(|k| (f64::from(k) / 4.0).exp2())
        .collect();
    let xs: Vec<f64> = match (lo.is_finite(), hi.is_finite()) {
        (true, true) => (1..SAMPLES)
            .map(|i| lo + (hi - lo) * f64::from(i) / f64::from(SAMPLES))
            .collect(),
        (true, false) => steps.iter().map(|s| lo + s).collect(),
        (false, true) => steps.iter().rev().map(|s| hi - s).collect(),
        (false, false) => steps
            .iter()
            .rev()
            .map(|s| -s)
            .chain(std::iter::once(0.0))
            .chain(steps.iter().copied())
            .collect(),
    };
    let hs: Vec<f64> = xs.iter().map(|&t| hf(t)).collect();
    let (best, _) = hs
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))?;
    if best == 0 || best + 1 >= xs.len() {
        return None;
    }
    // Golden-section search between the neighbours of the best sample.
    let (mut a, mut b) = (xs[best - 1], xs[best + 1]);
    let ratio = 0.5 * (5f64.sqrt() - 1.0);
    let (mut c1, mut c2) = (b - ratio * (b - a), a + ratio * (b - a));
    let (mut h1, mut h2) = (hf(c1), hf(c2));
    for _ in 0..200 {
        if b - a <= 4.0 * f64::EPSILON * a.abs().max(b.abs()) {
            break;
        }
        if h1 >= h2 {
            (b, c2, h2) = (c2, c1, h1);
            c1 = b - ratio * (b - a);
            h1 = hf(c1);
        } else {
            (a, c1, h1) = (c1, c2, h2);
            c2 = a + ratio * (b - a);
            h2 = hf(c2);
        }
    }
    let c = 0.5 * (a + b);
    let formal = arena.add(&formal);
    let d1 = crate::transforms::diff::diff(arena, formal, x);
    let d2 = crate::transforms::diff::diff(arena, d1, x);
    let curvature = crate::output::lambdify::compile(arena, d2, &[&x_name]).ok()?(&[c]);
    (curvature.is_finite() && curvature < 0.0).then(|| (c, (-curvature).sqrt().recip()))
}

/// The points at which `Ex::integrate_numeric_with` splits its `f64`
/// quadrature of `f` over `(lo, hi)`, sorted: the breakpoints of
/// [`quadrature_breakpoints`] (with the [`peak_points`]).  At a kink or a
/// step at an irrational point (`|sin x|` at `π`) the error estimate of the
/// segment holding it understates its error (QUADPACK's reduction assumes a
/// smooth integrand); a segment across an interior pole may cancel it
/// exactly: up to 0.49, when only the integrands with kinks were scanned,
/// `∫₋₂² tan x dx` and `∫₋₁¹ x/(x² − 1/4) dx`, which diverge, were
/// `Ok(0.0)` (the nodes symmetric about 0, the integrand odd).  A piece
/// that ends at a singularity is left to the caller's arbitrary-precision
/// fallback.
///
/// `exact` holds the finite bounds when known (see
/// [`quadrature_breakpoints`], whose errors this returns).
pub(crate) fn quadrature_split_points(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    lo: f64,
    hi: f64,
    exact: Option<Interval<ExprId>>,
) -> Result<Vec<f64>, SymplexError> {
    let mut points: Vec<f64> =
        quadrature_breakpoints(arena, f, x, Interval::closed(lo, hi), exact, Some(64))?
            .into_iter()
            .map(|(_, v)| v)
            .collect();
    points.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    // Cuts of an `f64` quadrature closer than its resolution are one cut.
    points.dedup_by(|a, b| (*a - *b).abs() <= 1e-14 * a.abs().max(1.0));
    Ok(points)
}

/// [`quadrature_report`] over `[a, b]` split at the sorted interior
/// `points` (`a < b`): the sum of the pieces, each with an equal share of
/// the absolute tolerance and of the subdivisions, and whether every piece
/// was resolved.
///
/// # Errors
///
/// As [`quadrature`].
pub(crate) fn quadrature_split(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    points: &[f64],
    opts: &QuadOpts,
) -> Result<(QuadResult, bool), SymplexError> {
    let inner: Vec<f64> = points.iter().copied().filter(|&p| a < p && p < b).collect();
    if inner.is_empty() {
        return quadrature_report(f, a, b, opts);
    }
    let n = inner.len() + 1;
    let share = QuadOpts {
        abs_tol: opts.abs_tol / n as f64,
        max_subdivisions: opts.max_subdivisions.div_ceil(n).max(1),
        ..*opts
    };
    let ends: Vec<f64> = std::iter::once(a)
        .chain(inner)
        .chain(std::iter::once(b))
        .collect();
    let mut total = QuadResult {
        value: 0.0,
        error: 0.0,
    };
    let mut resolved = true;
    for w in ends.windows(2) {
        let (r, ok) = quadrature_report(f, w[0], w[1], &share)?;
        total.value += r.value;
        total.error += r.error;
        resolved &= ok;
    }
    Ok((total, resolved))
}

/// Does `f` have a kink or a step depending on `x` (`|·|`, `sign`,
/// `Heaviside`, `floor`, `ceiling`, `min`, `max`, `Piecewise`)?
pub(crate) fn has_kink(arena: &Arena, f: ExprId, x: ExprId) -> bool {
    walk::post_order_ids(arena, f).into_iter().any(|id| {
        matches!(
            arena.node(id),
            ExprNode::Abs(_)
                | ExprNode::Sign(_)
                | ExprNode::Heaviside(_)
                | ExprNode::Floor(_)
                | ExprNode::Ceiling(_)
                | ExprNode::Min(_)
                | ExprNode::Max(_)
                | ExprNode::Piecewise(_)
        ) && walk::contains(arena, id, x)
    })
}

/// The pieces of the `f64` quadrature of `evalf` for a body `f` (in `x`,
/// both in `arena`) with a kink or a step on the finite range `[a, b]`
/// (`a < b`, the binary numbers themselves): the interior breakpoints of
/// [`quadrature_breakpoints`] (located to 64 bits where the scan cannot
/// place them) as `f64`s, and a bound on the error of integrating between
/// those instead of the breakpoints, per unit of the integrand's largest
/// value (the jump of a step is at most twice that).  `None` when not every
/// breakpoint is located, or the range is infinite or too narrow
/// ([`f64_narrow`]).
///
/// `evalf` reads the quadrature's error estimate as a bound, which across a
/// kink or a step it is not (QUADPACK's estimate assumes a smooth
/// integrand): up to 0.49 `eval_f64` of `∫_{100.7}^{130.7} frac(x) dx` was
/// `21` (truly 15), and of `∫_{10.7}^{410.7} |sin(x + 1/2)| dx` `254.736821`
/// (truly `254.736834`, mpmath).
pub(crate) fn f64_pieces(arena: &Arena, f: ExprId, x: ExprId, a: f64, b: f64) -> Option<F64Pieces> {
    if a.partial_cmp(&b) != Some(Ordering::Less)
        || !a.is_finite()
        || !b.is_finite()
        || f64_narrow(a, b)
    {
        return None;
    }
    let mut pa = Arena::with_config(arena.config.clone());
    let mut map: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let f_p = crate::base::compact::transfer_subtree(arena, &mut pa, f, &mut map);
    let x_p = crate::base::compact::transfer_subtree(arena, &mut pa, x, &mut map);
    let points =
        quadrature_breakpoints(&mut pa, f_p, x_p, Interval::closed(a, b), None, Some(64)).ok()?;
    let mut cuts = Vec::with_capacity(points.len());
    let mut misplaced = 0.0;
    for (p, v) in points {
        // The nearest `f64` to the point, from its 16-digit value: within
        // two ulps (a located crossing is within `2⁻⁶⁴` of the range of
        // its own value).
        let v = calculus_util::expr_to_f64(&mut pa, p).unwrap_or(v);
        if !(a < v && v < b) {
            return None;
        }
        misplaced += 2.0 * (2.0 * f64::EPSILON * v.abs() + (b - a) * 2f64.powi(-60));
        cuts.push(v);
    }
    cuts.sort_by(|p, q| p.partial_cmp(q).unwrap_or(Ordering::Equal));
    cuts.dedup();
    Some(F64Pieces { cuts, misplaced })
}

/// The split of [`f64_pieces`].
#[derive(Clone, Debug)]
pub(crate) struct F64Pieces {
    /// The interior split points, increasing.
    pub cuts: Vec<f64>,
    /// The error of splitting at `cuts` rather than at the breakpoints, per
    /// unit of the integrand's largest value.
    pub misplaced: f64,
}

/// [`quadrature_resolved`] on `[a, b]` (`a < b`, finite) split at the
/// increasing interior `cuts`: the sum of the pieces, each with an equal
/// share of the absolute tolerance and of the subdivisions; `None` unless
/// every piece is resolved.
///
/// # Errors
///
/// As [`quadrature`].
pub(crate) fn quadrature_resolved_split(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    cuts: &[f64],
    opts: &QuadOpts,
) -> Result<Option<QuadResult>, SymplexError> {
    let n = cuts.len() + 1;
    let share = QuadOpts {
        abs_tol: opts.abs_tol / n as f64,
        max_subdivisions: opts.max_subdivisions.div_ceil(n).max(16),
        ..*opts
    };
    let ends: Vec<f64> = std::iter::once(a)
        .chain(cuts.iter().copied())
        .chain(std::iter::once(b))
        .collect();
    let mut total = QuadResult {
        value: 0.0,
        error: 0.0,
    };
    for w in ends.windows(2) {
        let Some(r) = quadrature_resolved(f, w[0], w[1], &share, false)? else {
            return Ok(None);
        };
        total.value += r.value;
        total.error += r.error;
    }
    Ok(Some(total))
}

/// Is the finite range between `a` and `b` narrower than `2⁻²⁰` of its
/// position, so that `f64` arithmetic cannot resolve it: its nodes are a
/// few thousand ulps apart at most, and an integrand that varies on the
/// scale of the range (`frac(10x)` near `10¹⁵`) is evaluated there with
/// errors of `2²⁰` ulps of its argument.  The `f64` quadratures leave such a
/// range to the arbitrary-precision one; up to 0.49 `∫_{10¹⁵+1/3}^{10¹⁵+1/3+40}
/// frac(10x) dx` was `0` from `integrate_numeric` (truly 20).
pub(crate) fn f64_narrow(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (b - a).abs() < NARROW * a.abs().max(b.abs())
}

/// The relative width below which [`f64_narrow`] holds: `2⁻²⁰`.
const NARROW: f64 = 9.5367431640625e-7;

/// The highest degree of a denominator whose complex zeros
/// [`peak_points`] computes.
const MAX_POLE_SPLIT_DEGREE: usize = 12;

/// An upper bound on the degree of `e` as a polynomial in `x`, from its
/// structure (sums, products, natural powers), without expanding it;
/// `None` when `e` is not visibly a polynomial in `x`.
fn degree_bound(arena: &Arena, e: ExprId, x: ExprId) -> Option<usize> {
    let mut deg: FxHashMap<ExprId, usize> = FxHashMap::default();
    for id in walk::post_order_ids(arena, e) {
        let d = if id == x {
            1
        } else if !walk::contains(arena, id, x) {
            0
        } else {
            match arena.node(id) {
                ExprNode::Add(children) => children
                    .iter()
                    .map(|c| deg.get(c).copied())
                    .try_fold(0usize, |m, d| d.map(|d| m.max(d)))?,
                ExprNode::Mul(children) => children
                    .iter()
                    .map(|c| deg.get(c).copied())
                    .try_fold(0usize, |s, d| d.map(|d| s.saturating_add(d)))?,
                ExprNode::Pow(base, exp) => {
                    let n = arena.as_num(*exp)?;
                    if !n.is_integer() || n.is_negative() {
                        return None;
                    }
                    let n = n.to_integer().to_usize()?;
                    deg.get(base)?.saturating_mul(n)
                }
                _ => return None,
            }
        };
        deg.insert(id, d);
    }
    deg.get(&e).copied()
}

/// `f` on `[a, b]` under the endpoint-smoothing substitution `x = a +
/// (b−a)·s(t)`, `s(t) = t³(10 − 15t + 6t²)`, `t ∈ [0, 1]`: the integrand of
/// the same integral over `[0, 1]`, whose derivative vanishes quadratically
/// at both ends.  The nodes crowd towards both ends, so it is used only on
/// a piece whose singular end is at 0 ([`smoothed_pieces`]).
fn smoothed<'a>(f: &'a dyn Fn(f64) -> f64, a: f64, b: f64) -> impl Fn(f64) -> f64 + 'a {
    let width = b - a;
    move |t: f64| {
        let ds = 30.0 * t * t * (1.0 - t) * (1.0 - t);
        // The map's derivative is 0 at the ends: the weight is 0 there.
        if ds == 0.0 {
            return 0.0;
        }
        // Never exactly on an endpoint.
        let x = smoothing_map(a, b, t);
        let x = if t <= 0.5 {
            if x <= a { a.next_up() } else { x }
        } else if x >= b {
            b.next_down()
        } else {
            x
        };
        f(x) * width * ds
    }
}

/// The map of [`smoothed`], `a + (b−a)·s(t)`, evaluated from the nearer end
/// (`s(1−t) = 1 − s(t)`) so that `x` keeps full relative precision near
/// both ends.
fn smoothing_map(a: f64, b: f64, t: f64) -> f64 {
    let width = b - a;
    if t <= 0.5 {
        a + width * (t * t * t * (10.0 - 15.0 * t + 6.0 * t * t))
    } else {
        let r = 1.0 - t;
        b - width * (r * r * r * (10.0 - 15.0 * r + 6.0 * r * r))
    }
}

/// Work item for the adaptive loop.
#[derive(Clone, Copy, Debug)]
struct Segment {
    a: f64,
    b: f64,
    value: f64,
    err: f64,
    /// `|b − a|·max|f|` over its nodes ([`gk15_with`]).
    peak: f64,
}

impl Segment {
    fn new(a: f64, b: f64, (value, err, peak): (f64, f64, f64)) -> Segment {
        Segment {
            a,
            b,
            value: if value.is_finite() { value } else { 0.0 },
            err: if err.is_finite() { err } else { f64::INFINITY },
            peak: if peak.is_finite() {
                peak
            } else {
                f64::INFINITY
            },
        }
    }
}

/// When a segment of the adaptive loop is too narrow to bisect: its Kronrod
/// nodes would round onto each other or onto its ends.
#[derive(Clone, Copy)]
enum Floor<'a> {
    /// Narrower than `10⁻¹²` of the range (of its largest end, of 1): the
    /// plain bisection of [`quadrature_resolved`] (`evalf`).
    Range,
    /// Narrower than `10⁻¹²` of its own position — in the variable of the
    /// loop, and in `x` through the map when one is given — or than
    /// [`MIN_SEGMENT`]: towards an end at 0 the bisection goes as deep as
    /// `f64` allows, towards a nonzero end (in `x`) no deeper than before.
    Relative(Option<&'a dyn Fn(f64) -> f64>),
}

/// The narrowest segment [`Floor::Relative`] bisects: `10¹⁰` times the
/// smallest normal `f64`, so that its nodes stay normal numbers.
const MIN_SEGMENT: f64 = 1e10 * f64::MIN_POSITIVE;

/// Adaptive loop on a finite interval with a global error budget: the
/// estimate, and whether every segment was resolved (none left at the
/// narrowest width bisection can make, [`Floor`]).  The error of a segment
/// too narrow to bisect is at least its `peak`, `width·max|f|` at its
/// nodes: next to a singularity its estimate misses the mass between its
/// outermost node and its end, which `peak` covers for `|x − c|^α` with `α`
/// down to about `−0.99` (`0.0043^α ≥ 1/(1 + α)`, `0.0043` the relative
/// distance of the outermost Kronrod node to the end).
fn quadrature_finite_report(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    opts: &QuadOpts,
    raw: bool,
    floor: Floor<'_>,
) -> Result<(QuadResult, bool), SymplexError> {
    let nonfinite_err = |x: f64| SymplexError::ComputationFailed {
        operation: "integrate_numeric",
        reason: format!("integrand is not finite near x = {x:e}"),
    };
    // Segments narrower than this cannot be bisected meaningfully: the
    // Kronrod nodes would round onto the endpoints in `f64`.
    let min_width = 1e-12 * (b - a).abs().max(a.abs()).max(b.abs()).max(1.0);
    // The resolution of `f64` nodes: a segment past it is unresolved and the
    // caller refuses or falls back.
    let too_narrow = |lo: f64, hi: f64| -> bool {
        let width = (hi - lo).abs();
        match floor {
            Floor::Range => width < min_width,
            Floor::Relative(map) => {
                let narrow =
                    |w: f64, l: f64, h: f64| w < MIN_SEGMENT || w < 1e-12 * l.abs().max(h.abs());
                narrow(width, lo, hi)
                    || map.is_some_and(|x| {
                        let (xl, xh) = (x(lo), x(hi));
                        xl.is_finite() && xh.is_finite() && narrow((xh - xl).abs(), xl, xh)
                    })
            }
        }
    };
    let mut segments: Vec<Segment> = vec![Segment::new(a, b, gk15_with(f, a, b, raw))];
    let mut total_value = segments[0].value;
    let mut total_err = segments[0].err;
    // A segment that could not be bisected further.
    let mut floor_hit = false;

    for _ in 0..opts.max_subdivisions {
        let tol = opts.abs_tol.max(opts.rel_tol * total_value.abs());
        if total_err.is_finite() && total_err <= tol {
            break;
        }
        // Pick the segment with the largest error.
        let (idx, _) = segments
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.err.partial_cmp(&y.1.err).unwrap_or(Ordering::Equal))
            .ok_or_else(|| SymplexError::ComputationFailed {
                operation: "integrate_numeric",
                reason: "empty work queue".into(),
            })?;
        let worst = segments.swap_remove(idx);
        if too_narrow(worst.a, worst.b) {
            if !worst.peak.is_finite() {
                return Err(nonfinite_err(0.5 * (worst.a + worst.b)));
            }
            // Cannot refine further: its estimate is no bound next to a
            // singularity, its `peak` is (a hundred of them for a rise
            // faster than the outermost nodes can extrapolate).
            let err = if worst.err.is_finite() {
                worst.err.max(worst.peak)
            } else {
                100.0 * worst.peak
            };
            segments.push(Segment { err, ..worst });
            floor_hit = true;
            total_err = segments.iter().map(|s| s.err).sum();
            break;
        }
        let mid = 0.5 * (worst.a + worst.b);
        segments.push(Segment::new(worst.a, mid, gk15_with(f, worst.a, mid, raw)));
        segments.push(Segment::new(mid, worst.b, gk15_with(f, mid, worst.b, raw)));
        total_value = segments.iter().map(|s| s.value).sum();
        total_err = segments.iter().map(|s| s.err).sum();
    }

    if !total_value.is_finite() || !total_err.is_finite() {
        return Err(SymplexError::ComputationFailed {
            operation: "integrate_numeric",
            reason:
                "quadrature did not produce a finite estimate (integrand non-finite or divergent)"
                    .into(),
        });
    }
    let resolved = !floor_hit;
    Ok((
        QuadResult {
            value: total_value,
            error: total_err,
        },
        resolved,
    ))
}
