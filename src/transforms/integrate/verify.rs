//! Nested powers over the reals (part of [`integrate`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Nested powers over the reals
// ═══════════════════════════════════════════════════════════════════════════

/// Flattening of `Pow(Pow(g, m), n)` with rational `m`, `n` that keeps the
/// value — on the principal branch, and on the real line of the
/// integration variable.
///
/// The canon layer merges an integer outer exponent `n` (since 0.25; before,
/// only integer `m` and `n`); here also `−1 < m ≤ 1` and the real-line
/// `|g|` forms.  `(g^m)^n = g^{m·n}` holds
/// for every complex `g` when `n` is an integer or `−1 < m ≤ 1` (the
/// condition of simplify's `pow_pow`); for `g` real on the real line and `m`
/// an even integer, `g^m = |g|^m ≥ 0` and `(g^m)^n = |g|^{m·n}` (`√(x²) =
/// |x|`), which is `g^{m·n}` again when `m·n` is an even integer.  Anything
/// else stays nested (`None`): `(1/x)^{1/2} ≠ x^{−1/2}` for `x < 0` (`i/√|x|`
/// against `−i/√|x|`), and `(g^{2/3})` is not `|g|^{2/3}` for `g < 0` —
/// before 0.23 both were flattened (the second under the real-root
/// convention the evaluator no longer uses), and `∫ √x·√(1/x) dx` came out
/// `x`, whose derivative is wrong for `x < 0` (found by `fuzz_integrate`).
pub(super) fn flatten_nested_pow(
    arena: &mut Arena,
    expr: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let ExprNode::Pow(base, exp) = arena.node(expr).clone() else {
        return None;
    };
    let ExprNode::Pow(inner_base, inner_exp) = arena.node(base).clone() else {
        return None;
    };
    let (m, n) = (arena.as_num(inner_exp)?.clone(), arena.as_num(exp)?.clone());
    let combined = &m * &n;
    let two = num_bigint::BigInt::from(2);
    let even_integer = |q: &crate::base::numeric::Q| q.is_integer() && (q.numer() % &two).is_zero();
    let principal =
        n.is_integer() || (m > crate::base::numeric::qi(-1) && m <= crate::base::numeric::qi(1));
    let abs_form = even_integer(&m) && real_on_reals(arena, inner_base, var_sym);
    let combined_id = arena.num_ratio(combined.clone());
    if principal || (abs_form && even_integer(&combined)) {
        Some(arena.pow(inner_base, combined_id))
    } else if abs_form {
        let abs_base = arena.abs(inner_base);
        Some(arena.pow(abs_base, combined_id))
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric verification of candidate antiderivatives
// ═══════════════════════════════════════════════════════════════════════════

/// Sample points for [`check_antiderivative`]: small rationals of both
/// signs, none of them a root of the denominators met in practice.
const FTC_SAMPLE_POINTS: [(i64, i64); 6] = [(1, 3), (-5, 7), (13, 11), (-17, 5), (7, 5), (-2, 1)];

/// The second round of sample points, tried only when the first found no
/// point where `F′ = f`: nearer to 0 and farther out, for integrands that
/// are real only on part of the line (`√(x − 5)` is complex at every
/// point of the first round, and `F′` may legitimately differ from it
/// there under the real-variable convention).
const FTC_RETRY_POINTS: [(i64, i64); 6] =
    [(1, 19), (-3, 23), (37, 4), (-29, 3), (211, 7), (-307, 9)];

/// Decimal digits requested from `evalf` when checking `F′ − f`: the
/// working precision behind 30 digits is ~50 digits, which separates an
/// exact antiderivative (`F′ − f` at roundoff, ~1e-45) from one built on
/// pseudo-exact decimal coefficients (`nsimplify`'d roots, ~1e-13).
pub(super) const FTC_CHECK_DIGITS: u32 = 30;

/// Tolerance on `|F′(x₀) − f(x₀)| / max(1, |f(x₀)|)` at [`FTC_CHECK_DIGITS`].
const FTC_CHECK_REL_TOL: f64 = 1e-20;

/// How far from the tolerance the distance of the separately evaluated
/// `F′(x₀)` and `f(x₀)` must lie to decide a point of [`ftc_point`] without
/// evaluating `F′ − f` as one expression.  Both values are certified to
/// [`FTC_CHECK_DIGITS`] digits, ten orders of magnitude below the
/// tolerance, so any factor above 1 is safe; this one leaves room for the
/// `f64` rounding of the distance and of the tolerance.
const FTC_DECISIVE_FACTOR: f64 = 16.0;

/// What [`check_antiderivative`] found out about a candidate `F` for `∫ f`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FtcVerdict {
    /// `F′ = f` at one sample point at least, where `F` evaluates too, and
    /// `F′ ≠ f` at none where `f` is real.
    Verified,
    /// `F′ ≠ f` at a sample point where `f` is real, or `F′ − f` is a
    /// non-zero constant.
    Wrong,
    /// `f` evaluates at some sample point, but `F′` agrees with it at none:
    /// `F′` does not evaluate there, it differs from `f` only where `f` is
    /// complex, or it agrees only where `F` itself does not evaluate.
    Undecided,
    /// No numeric test is possible: `F` or `F′` has an unevaluated node, a
    /// parameter has declared assumptions, or `f` itself evaluates at no
    /// sample point.
    Untestable,
}

/// `true` when the candidate `big_f` for `∫ f d(var)` must not be returned.
///
/// A candidate is accepted on **evidence**, not for the lack of it.
/// [`check_antiderivative`] compares `F′` with `f` at sample points, and:
///
/// - [`FtcVerdict::Verified`] — accepted.
/// - [`FtcVerdict::Wrong`] — rejected.
/// - [`FtcVerdict::Undecided`] — **rejected**.  `f` evaluates at some sample
///   point but `F′` agrees with it at none, even after a second round of
///   points ([`FTC_RETRY_POINTS`]).  That is what a quotient by a
///   difference that cancels to 0 looks like: since 0.26 `evalf` refuses
///   such a value with `PrecisionExhausted` instead of returning noise, and
///   up to 0.28.0 such candidates passed unchecked, because "no point
///   evaluates" counted as "not wrong".  An `F′` that is undefined at a
///   generic point where `f` is defined is not the derivative of an
///   antiderivative of `f` there; nor is an `F` that is undefined wherever
///   its `F′` agrees with `f` (the vanishing denominator cancelled in
///   `F′`) an antiderivative.  A difference only where `f` is complex is
///   excused by the real-variable convention but verifies nothing:
///   `ln|x − c|` for `∫ dx/(x − c)` with an imaginary `c` differs from
///   `f` at every point.  A retry at higher precision would
///   add nothing: `evalf` already re-evaluates up to twice the working
///   precision plus 256 bits before it refuses.
/// - [`FtcVerdict::Untestable`] — accepted.  When `f` itself evaluates at no
///   sample point, `F` contains an unevaluated `Integral` (a partial
///   answer, which says so), `F′` a derivative `diff` could not take, or a
///   parameter carries declared assumptions none of the generic values
///   satisfies, no candidate could ever be tested; the answer stands as
///   the route derived it.
///
/// Free parameters are bound to generic positive rationals
/// ([`FTC_PARAMETER_VALUES`]) before sampling; up to 0.28.0 a parameter
/// skipped the check altogether.  A verified candidate with parameters
/// that are not declared real then faces a second round with those
/// parameters at generic **complex** values (decision D4, 0.31; see
/// [`check_antiderivative`]), which rejects the real-parameter forms
/// (`ln|a·x + 1|`, `|a|`, `sign(a)`) that the first round cannot tell from
/// the principal ones.
///
/// Reached through [`candidate_rejected`], by the routes that check their own
/// candidates and by every stage of `integrate_impl`, so that no closed form
/// leaves `integrate` without facing this rule.  The tight tolerance
/// [`FTC_CHECK_REL_TOL`] relies on the certified digits of `evalf`, which
/// refuses a value it cannot certify (the point then says nothing).  On
/// the Rubi suite (0.28) the stage answers came out 8,952 verified and 752
/// untestable (670 of them partial answers), none wrong or undecided.
pub(super) fn antiderivative_rejected(
    arena: &mut Arena,
    f: ExprId,
    big_f: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> bool {
    // A route's accepted candidate is usually its stage's answer too, and
    // is then judged again at the stage's exit: the verdict is remembered
    // for the rest of the top-level call (cleared with `INTEGRATE_MEMO`).
    let key = (arena.serial(), f, big_f, var);
    let verdict = match VERDICT_MEMO.with(|m| m.borrow().get(&key).copied()) {
        Some(verdict) => verdict,
        None => {
            let verdict = stage!(
                arena,
                "verify",
                big_f,
                check_antiderivative(arena, f, big_f, var, var_sym)
            );
            VERDICT_MEMO.with(|m| m.borrow_mut().insert(key, verdict));
            verdict
        }
    };
    match verdict {
        FtcVerdict::Verified => {
            let key = (arena.serial(), f, big_f, var);
            if let Some(jumps) = JUMP_MEMO.with(|m| m.borrow().get(&key).copied()) {
                return jumps;
            }
            let jumps = jumps_at_argument_poles(arena, f, big_f, var, var_sym);
            JUMP_MEMO.with(|m| m.borrow_mut().insert(key, jumps));
            if jumps {
                tracing::debug!("integrate: candidate jumps where the integrand is finite");
            }
            jumps
        }
        FtcVerdict::Untestable => false,
        FtcVerdict::Wrong | FtcVerdict::Undecided => {
            tracing::debug!(?verdict, "integrate: candidate antiderivative rejected");
            true
        }
    }
}

thread_local! {
    /// [`jumps_at_argument_poles`] verdicts during the current top-level
    /// `integrate` (cleared with [`VERDICT_MEMO`]).
    pub(super) static JUMP_MEMO: std::cell::RefCell<FxHashMap<(u64, ExprId, ExprId, ExprId), bool>> =
        std::cell::RefCell::new(FxHashMap::default());
}

/// Does the candidate `big_f` jump at a real pole `p` of the argument of one
/// of its logarithms or dilogarithms (`ln(1/y)`, `Li₂(1 + 1/y)` at `y = 0`)
/// where the integrand `f` stays finite: `|F(p + ε) − F(p − ε)|` well above
/// `2ε·|f|`?  The FTC check samples `F′` away from such points and cannot see
/// it.  Up to 0.46 `∫ (ln(−1/y)/2 − ln(1/y)/2)/y − (ln(1 − 1/y)/2 − ln(1 +
/// 1/y)/2)/y dy` (SymPy's test_risch; its integrand tends to 1 at 0 from
/// both sides) came out with a step of `π²/2` there.  Only for parameter-free
/// candidates with such an argument; numeric poles only.
fn jumps_at_argument_poles(
    arena: &mut Arena,
    f: ExprId,
    big_f: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> bool {
    if crate::base::walk::free_symbols(arena, big_f)
        .iter()
        .any(|&s| s != var)
    {
        return false;
    }
    // A check that can only reject a candidate: its `f64` tolerances below
    // decide nothing else.
    let mut poles: Vec<f64> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, big_f) {
        let arg = match arena.node(id) {
            ExprNode::Ln(a) => *a,
            ExprNode::Lib(crate::base::libfn::LibFn::PolyLog, args) if args.len() == 2 => args[1],
            _ => continue,
        };
        if !contains_var(arena, arg, var_sym) {
            continue;
        }
        let (_, den) = crate::poly::polybridge::as_numer_denom(arena, arg);
        if !contains_var(arena, den, var_sym) {
            continue;
        }
        let Some(roots) = real_roots(arena, den, var) else {
            continue;
        };
        for r in roots {
            if let Some(v) = crate::transforms::evalf::eval_const_f64(arena, r)
                && !poles
                    .iter()
                    .any(|&w| (w - v).abs() <= 1e-12 * v.abs().max(1.0))
            {
                poles.push(v);
            }
        }
        if poles.len() > 4 {
            return false;
        }
    }
    for p in poles {
        let eps = 1e-7 * p.abs().max(1.0);
        let at = |arena: &mut Arena, e: ExprId, t: f64| -> Option<num_complex::Complex64> {
            let q = crate::base::numeric::f64_to_ratio_exact(t)?;
            let tv = arena.num_ratio(q);
            let s = crate::transforms::subs::subs(arena, e, var, tv);
            let s = crate::transforms::eval::eval(arena, s);
            crate::transforms::evalf::evalf_complex64(arena, s)
                .ok()
                .filter(|z| z.re.is_finite() && z.im.is_finite())
        };
        let (Some(f_lo), Some(f_hi)) = (at(arena, f, p - eps), at(arena, f, p + eps)) else {
            continue;
        };
        let scale = f_lo.norm().max(f_hi.norm());
        if scale.is_nan() || scale >= 1e6 || (f_lo - f_hi).norm() > 1e-3 * scale.max(1.0) {
            continue;
        }
        let (Some(big_lo), Some(big_hi)) = (at(arena, big_f, p - eps), at(arena, big_f, p + eps))
        else {
            continue;
        };
        if (big_hi - big_lo).norm() > 1e-4 * big_lo.norm().max(1.0) + 100.0 * eps * scale.max(1.0) {
            return true;
        }
    }
    false
}

thread_local! {
    /// [`check_antiderivative`] verdicts during the current top-level
    /// `integrate`, by (arena, `f`, `F`, variable).
    pub(super) static VERDICT_MEMO: std::cell::RefCell<FxHashMap<(u64, ExprId, ExprId, ExprId), FtcVerdict>> =
        std::cell::RefCell::new(FxHashMap::default());
}

/// Generic values for the free parameters of a checked candidate, handed
/// out in the order of the parameters' names: distinct, positive,
/// non-integer, with pairwise different prime denominators, so that no
/// accidental relation (`a = b`, `m + 1 = n`, `a·b = 1`) holds between
/// them.  None is a value of the Rubi harness's parameter table, so the
/// harness stays an independent check.
pub(super) const FTC_PARAMETER_VALUES: [(i64, i64); 8] = [
    (17, 13),
    (7, 11),
    (23, 17),
    (5, 19),
    (31, 23),
    (11, 29),
    (41, 31),
    (13, 37),
];

/// Imaginary parts of the complex values of the second round of
/// [`check_antiderivative`]: the k-th parameter that is not declared real
/// takes `FTC_PARAMETER_VALUES[k] + i·FTC_PARAMETER_IMAG[k]`.
/// Prime denominators unlike those of the real parts, so the values keep
/// clear of accidental relations; moderate imaginary parts (0.3 to 0.5)
/// and positive real parts, see [`complex_round_differs`].
const FTC_PARAMETER_IMAG: [(i64, i64); 8] = [
    (13, 41),
    (19, 43),
    (17, 47),
    (23, 53),
    (19, 59),
    (29, 61),
    (23, 67),
    (31, 71),
];

/// Which values [`bind_parameters`] gives the parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParameterValues {
    /// The k-th parameter at the k-th value of [`FTC_PARAMETER_VALUES`]
    /// or its negative, whichever satisfies its declaration.
    Real,
    /// A parameter that is not declared real at that value plus
    /// `i·FTC_PARAMETER_IMAG[k]` where that satisfies its declaration (always,
    /// for one without declared assumptions); the others as in `Real`.
    Complex,
}

/// `f` and `big_f` with their parameters bound, and whether any of them
/// may be complex (is not declared real), and so is bound to a complex
/// value in the [`ParameterValues::Complex`] round.
struct BoundCandidate {
    f: ExprId,
    big_f: ExprId,
    may_be_complex: bool,
}

/// `f` and `big_f` with every free symbol other than `var` bound to a
/// value derived from [`FTC_PARAMETER_VALUES`] ([`ParameterValues`]): the
/// k-th parameter in name order takes the k-th value, its negative, or (in
/// the complex round, for a parameter not declared real) that value plus
/// `i·FTC_PARAMETER_IMAG[k]` — the first of these that provably satisfies
/// every declared property (a parameter without assumptions takes any; a
/// `real` or `positive` one is sampled at `17/13`, a `negative` one at
/// `−17/13`).  `None` when no candidate does (an `integer` symbol must not
/// be sampled at `17/13`) or there are more parameters than values.  Up to
/// 0.30 any declared parameter made the candidate untestable.  `subs`
/// replaces free occurrences only, so a `RootOf(a⁵ − a + 1, 0)` or a sum
/// over `a` keeps its bound `a` (0.28.0 renamed such binders here first,
/// when `subs` did not respect them).
fn bind_parameters(
    arena: &mut Arena,
    f: ExprId,
    big_f: ExprId,
    var: ExprId,
    mode: ParameterValues,
) -> Option<BoundCandidate> {
    let mut params: Vec<(String, ExprId)> = Vec::new();
    for root in [f, big_f] {
        for s in crate::base::walk::free_symbols(arena, root) {
            if s == var || params.iter().any(|&(_, p)| p == s) {
                continue;
            }
            let ExprNode::Symbol(sid) = *arena.node(s) else {
                return None;
            };
            params.push((arena.symbol_name(sid).to_owned(), s));
        }
    }
    if params.len() > FTC_PARAMETER_VALUES.len() {
        return None;
    }
    params.sort_by(|a, b| a.0.cmp(&b.0));
    let (mut f, mut big_f) = (f, big_f);
    let mut may_be_complex = false;
    let mut facts = crate::base::assumptions::AssumptionCache::new();
    for (k, &(_, s)) in params.iter().enumerate() {
        let ExprNode::Symbol(sid) = *arena.node(s) else {
            return None;
        };
        let mut declared = arena.symbol_assumptions(sid);
        declared.normalize_declared();
        let complex = !Truth::from(declared.query(crate::base::assumptions::Props::REAL)).is_true();
        may_be_complex |= complex;
        let (p, q) = FTC_PARAMETER_VALUES[k];
        let re = arena.rational(p, q);
        let mut candidates: SmallVec<[ExprId; 3]> = SmallVec::new();
        if mode == ParameterValues::Complex && complex {
            let (r, t) = FTC_PARAMETER_IMAG[k];
            let im = arena.rational(r, t);
            let i = arena.i_unit();
            let im = arena.mul(&[im, i]);
            candidates.push(arena.add(&[re, im]));
        }
        candidates.push(re);
        candidates.push(arena.rational(-p, q));
        // An integer parameter at the numerator (distinct odd primes, so
        // again no accidental relation), its negative, or twice either
        // for an even one.  Up to 0.46 no candidate satisfied `integer`,
        // and every antiderivative with an integer parameter was accepted
        // untested: `∫ sin(kx)·sin(|k|x) dx` as a quotient by `k − |k|`.
        for m in [1, -1, 2, -2] {
            candidates.push(arena.int(m * p));
        }
        let value = candidates
            .into_iter()
            .find(|&v| value_satisfies(arena, v, declared, &mut facts))?;
        f = crate::transforms::subs::subs(arena, f, s, value);
        big_f = crate::transforms::subs::subs(arena, big_f, s, value);
    }
    Some(BoundCandidate {
        f,
        big_f,
        may_be_complex,
    })
}

/// Does the constant `value` provably have every property `declared`
/// asserts and provably lack every one it denies?
fn value_satisfies(
    arena: &mut Arena,
    value: ExprId,
    declared: crate::base::assumptions::Assumptions,
    facts: &mut crate::base::assumptions::AssumptionCache,
) -> bool {
    declared
        .known_true
        .iter()
        .all(|flag| Truth::from(facts.query(arena, value, flag)).is_true())
        && declared
            .known_false
            .iter()
            .all(|flag| Truth::from(facts.query(arena, value, flag)).is_false())
}

/// The numeric FTC check behind [`antiderivative_rejected`]: `F′` against
/// `f` at [`FTC_SAMPLE_POINTS`], then, if no point agreed, at
/// [`FTC_RETRY_POINTS`].
///
/// At each point `f` is evaluated first; a point where it does not
/// evaluate says nothing.  Then `F′` on its own, which decides the point
/// when its distance from `f` is clear of the tolerance (see
/// [`ftc_point`]); otherwise `F′ − f` is evaluated as one expression (its
/// exact cancellation to 0 is what `evalf`'s zero detection is built for).
/// `|F′ − f| ≤ tol·max(1, |f|)` is an agreement, real or complex; a larger
/// difference is
/// [`FtcVerdict::Wrong`] where `f` is real and no evidence where it is
/// complex (the real-variable convention: `ln|x|` for `∫ dx/x` differs
/// from `1/x` for `x < 0` only in a complex continuation, and the
/// continuation of `f` is what is compared there).
///
/// That first round binds every parameter to a real rational
/// ([`ParameterValues::Real`]), where `ln|a·x + 1|/a` and `ln(a·x + 1)/a`
/// both pass.  A verified candidate with parameters that are not declared
/// real — complex numbers in the crate's model — faces a second round with
/// them at complex values ([`complex_round_differs`]); a difference there
/// makes it [`FtcVerdict::Wrong`] (decision D4, 0.31).
pub(super) fn check_antiderivative(
    arena: &mut Arena,
    f: ExprId,
    big_f: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> FtcVerdict {
    if crate::base::walk::has_unevaluated(arena, big_f) {
        return FtcVerdict::Untestable;
    }
    let Some(bound) = bind_parameters(arena, f, big_f, var, ParameterValues::Real) else {
        return FtcVerdict::Untestable;
    };
    let mut real_domain: Vec<ExprId> = Vec::new();
    let verdict =
        check_bound_antiderivative(arena, bound.f, bound.big_f, var, var_sym, &mut real_domain);
    if verdict != FtcVerdict::Verified || !bound.may_be_complex {
        return verdict;
    }
    // The complex round tells apart forms that are not analytic in the
    // parameters; an `f` and `F` built from analytic functions of them
    // alone agree there as they do at the real values (the continuation
    // argument of [`complex_round_differs`]).  It was half of the cost of
    // the check (0.31: 4.5 of 9.7 s on a sample of the Rubi suite's
    // answers, and none of its 72,000 integrands met a candidate the
    // round rejected that is analytic in this sense).
    if analytic_in_parameters(arena, f, var_sym) && analytic_in_parameters(arena, big_f, var_sym) {
        return verdict;
    }
    let Some(complex) = bind_parameters(arena, f, big_f, var, ParameterValues::Complex) else {
        return verdict;
    };
    if real_domain.is_empty() {
        real_domain = real_domain_points(arena, bound.f, var);
    }
    if complex_round_differs(arena, complex.f, complex.big_f, var, var_sym, &real_domain) {
        tracing::debug!(
            "integrate: candidate antiderivative fails the FTC check at complex parameter values"
        );
        return FtcVerdict::Wrong;
    }
    verdict
}

/// The functions [`analytic_in_parameters`] counts as analytic: the
/// elementary functions, `Γ`, `erf`, `erfc`, `Si`, `Ci`, `Ei`, `li`, `ζ`.
pub(super) fn analytic_off_cuts(f: crate::base::func::Func) -> bool {
    use crate::base::func::Func;
    f.elementary().is_some()
        || matches!(
            f,
            Func::Gamma
                | Func::Erf
                | Func::Erfc
                | Func::Si
                | Func::Ci
                | Func::Ei
                | Func::Li
                | Func::Zeta
        )
}

/// Is `e` analytic in its parameters (the free symbols other than the
/// variable): does every node that is not an analytic function of its
/// arguments — `|·|`, `sign`, `re`, `arg`, `Piecewise`, a relation, an
/// unknown function, … — have no parameter below it?  (`ln|x|` is;
/// `ln|a·x + 1|` and `sign(a)` are not.)  Elementary functions, `erf`,
/// `Ei`, `li`, `Si`, `Ci`, `Γ`, `ζ` and `RootSum` (a symmetric function of
/// the roots) count as analytic: their principal branches are analytic off
/// their cuts, and crossing a cut changes `F` by a constant, not `F′`.
pub(super) fn analytic_in_parameters(arena: &Arena, e: ExprId, var_sym: SymbolId) -> bool {
    let mut has_param: FxHashMap<ExprId, bool> = FxHashMap::default();
    for id in crate::base::walk::post_order_ids(arena, e) {
        let node = arena.node(id);
        let mut below = matches!(node, ExprNode::Symbol(s) if *s != var_sym);
        node.for_each_child(|c| below |= has_param.get(&c).copied().unwrap_or(false));
        let analytic = matches!(
            node,
            ExprNode::Num(_)
                | ExprNode::Symbol(_)
                | ExprNode::Pi
                | ExprNode::E
                | ExprNode::ImaginaryUnit
                | ExprNode::EulerGamma
                | ExprNode::Catalan
                | ExprNode::GoldenRatio
                | ExprNode::Add(_)
                | ExprNode::Mul(_)
                | ExprNode::Pow(..)
                | ExprNode::Neg(_)
                | ExprNode::RootSum(..)
        ) || node.func_head().is_some_and(analytic_off_cuts);
        if below && !analytic {
            return false;
        }
        has_param.insert(id, below);
    }
    true
}

/// The points of [`FTC_SAMPLE_POINTS`] and [`FTC_RETRY_POINTS`] where the
/// integrand `f` (parameters bound to real values) evaluates to a real
/// number.
fn real_domain_points(arena: &mut Arena, f: ExprId, var: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    for &(p, q) in FTC_SAMPLE_POINTS.iter().chain(FTC_RETRY_POINTS.iter()) {
        let point = arena.rational(p, q);
        let f_at = crate::transforms::subs::subs(arena, f, var, point);
        // A guide that can only make the complex round stricter: a point
        // whose integrand has a certified but negligible imaginary part
        // (`e⁻⁸⁰·i` next to 1, below the 30 digits) is compared too, where
        // an analytic `F` still has `F′ = f`; a point left out could only
        // let a real-parameter form through.
        if numeric_value(arena, f_at)
            .is_some_and(|v| crate::transforms::evalf::is_real_to_digits(&v, FTC_CHECK_DIGITS))
        {
            out.push(point);
        }
    }
    out
}

/// Points of the complex round that must agree before it gives up
/// looking for a difference.
const FTC_COMPLEX_AGREEMENTS: usize = 3;

/// The second round of [`check_antiderivative`]: `f` and `big_f` with the
/// parameters that are not declared real at complex values
/// ([`ParameterValues::Complex`]), compared at `real_domain` — the real
/// sample points where the integrand is real for the real values of the
/// first round.  `true` when `F′ ≠ f` at one of them.
///
/// Why the comparison is sound although `f` is complex at every point now,
/// where the first round excuses a difference (the real-variable
/// convention): `x` is real and only the parameters moved off the real
/// axis.  At a point where the integrand is real for real parameters,
/// `F′ = f` holds for the parameters on a real interval, and both sides
/// are analytic in the parameters there, so the identity continues to
/// complex parameter values near that interval — unless `F` is not
/// analytic in them, which is exactly what a real-parameter form is:
/// `d/dx ln|a·x + 1| = re(conj(u)·a)/|u|²` differs from `a/u` at every point
/// where `u` is not real; so do `|a|` and `sign(a)`.  The allowance of the
/// first round is kept where it belongs: a point where the integrand is
/// complex for real parameters (`√(x − 5)` at `x = 1/3`) is not in
/// `real_domain` and is not compared, so `ln|x + √(x² − 1)|`-type parts
/// that depend on `x` alone keep their convention whatever their
/// coefficients.  The complex values have positive real parts
/// ([`FTC_PARAMETER_IMAG`]): a principal form such as `√a` has its cut on
/// the negative axis, and values next to the positive reals continue the
/// identity from there without crossing it.  An undecidable point (`F′`
/// or `f` does not evaluate at the complex values) says nothing; the round
/// stops after [`FTC_COMPLEX_AGREEMENTS`] agreeing points.
fn complex_round_differs(
    arena: &mut Arena,
    f: ExprId,
    big_f: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    real_domain: &[ExprId],
) -> bool {
    let d_big_f = crate::transforms::diff::diff(arena, big_f, var);
    if crate::base::walk::has_unevaluated(arena, d_big_f) {
        return false;
    }
    let residual = arena.sub(d_big_f, f);
    if !contains_var(arena, residual, var_sym) {
        return residual_abs(arena, residual).is_some_and(|r| r > FTC_CHECK_REL_TOL);
    }
    let mut agreements = 0;
    for &point in real_domain {
        match ftc_point(arena, f, d_big_f, residual, var, point) {
            FtcPoint::Differs { .. } => {
                tracing::debug!(
                    point = %arena.display(point),
                    "integrate: F′ ≠ f at complex parameter values"
                );
                return true;
            }
            FtcPoint::Agrees { .. } => {
                agreements += 1;
                if agreements >= FTC_COMPLEX_AGREEMENTS {
                    return false;
                }
            }
            FtcPoint::IntegrandUnevaluable | FtcPoint::DerivativeUnevaluable => {}
        }
    }
    false
}

/// The first round of [`check_antiderivative`], on `f` and `big_f` with
/// their parameters bound.  Every sample point where `f` is real and
/// `F′ = f` is appended to `real_domain`.
fn check_bound_antiderivative(
    arena: &mut Arena,
    f: ExprId,
    big_f: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    real_domain: &mut Vec<ExprId>,
) -> FtcVerdict {
    let d_big_f = crate::transforms::diff::diff(arena, big_f, var);
    if crate::base::walk::has_unevaluated(arena, d_big_f) {
        return FtcVerdict::Untestable;
    }
    let residual = arena.sub(d_big_f, f);
    if !contains_var(arena, residual, var_sym)
        && let Some(r) = residual_abs(arena, residual)
    {
        // A constant residual: anything but zero is a wrong constant term
        // in F′.  Zero (`F′ ≡ f`, often structurally) still needs an `F`
        // that is defined somewhere.  One that does not evaluate is
        // sampled below.
        if r > FTC_CHECK_REL_TOL {
            return FtcVerdict::Wrong;
        }
        return if evaluates_at_a_sample_point(arena, big_f, var) {
            FtcVerdict::Verified
        } else if evaluates_at_a_sample_point(arena, f, var) {
            FtcVerdict::Undecided
        } else {
            FtcVerdict::Untestable
        };
    }
    let mut integrand_evaluates = false;
    for points in [&FTC_SAMPLE_POINTS, &FTC_RETRY_POINTS] {
        let mut agrees = false;
        for &(p, q) in points {
            let point = arena.rational(p, q);
            match ftc_point(arena, f, d_big_f, residual, var, point) {
                FtcPoint::IntegrandUnevaluable => {}
                FtcPoint::Agrees { f_real } => {
                    integrand_evaluates = true;
                    if f_real {
                        real_domain.push(point);
                    }
                    // `F′ = f` proves nothing about an `F` that is undefined
                    // (`x·e^{kx}/k − e^{kx}/k²` with a `k` that is 0 without
                    // being so structurally has `F′ = x·e^{kx}` exactly).
                    if !agrees {
                        let big_f_at = crate::transforms::subs::subs(arena, big_f, var, point);
                        agrees = numeric_value(arena, big_f_at).is_some();
                    }
                }
                FtcPoint::DerivativeUnevaluable | FtcPoint::Differs { f_real: false } => {
                    integrand_evaluates = true;
                }
                FtcPoint::Differs { f_real: true } => {
                    tracing::debug!(
                        point = %arena.display(point),
                        "integrate: candidate antiderivative fails the numeric FTC check"
                    );
                    return FtcVerdict::Wrong;
                }
            }
        }
        if agrees {
            return FtcVerdict::Verified;
        }
    }
    if integrand_evaluates {
        FtcVerdict::Undecided
    } else {
        FtcVerdict::Untestable
    }
}

/// The comparison of `F′` with `f` at one sample point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FtcPoint {
    /// `f` does not evaluate here: no information.
    IntegrandUnevaluable,
    /// `f` evaluates, `F′` (and `F′ − f`) does not.
    DerivativeUnevaluable,
    /// `|F′ − f| ≤ tol·max(1, |f|)`, with `f` real or complex here.
    Agrees { f_real: bool },
    /// `|F′ − f| > tol·max(1, |f|)`, with `f` real or complex here.
    Differs { f_real: bool },
}

fn ftc_point(
    arena: &mut Arena,
    f: ExprId,
    d_big_f: ExprId,
    residual: ExprId,
    var: ExprId,
    point: ExprId,
) -> FtcPoint {
    let f_at = crate::transforms::subs::subs(arena, f, var, point);
    let Some(f_val) = numeric_value(arena, f_at) else {
        return FtcPoint::IntegrandUnevaluable;
    };
    let Some(f_abs) = crate::transforms::evalf::abs_to_f64(&f_val).filter(|v| v.is_finite()) else {
        return FtcPoint::IntegrandUnevaluable;
    };
    // `f_real` excuses a difference where `f` is complex (the real-variable
    // convention, `ln|x|` parts) and admits the point to the complex round.
    // Reading an imaginary part negligible next to the real part (below the
    // 30 digits: `e⁻⁸⁰·i` next to 1) as real is the stricter reading, right
    // either way: such a point excuses nothing, and a candidate whose `F′`
    // agrees with `f` to the tolerance there still agrees.
    let f_real = crate::transforms::evalf::is_real_to_digits(&f_val, FTC_CHECK_DIGITS);
    let tol = FTC_CHECK_REL_TOL * f_abs.max(1.0);

    // `F′` on its own first.  Both values are certified to
    // [`FTC_CHECK_DIGITS`] digits, so their distance is `|F′ − f|` to within
    // `10⁻³⁰·(|F′| + |f|)`, far below the tolerance: a distance clear of it
    // by [`FTC_DECISIVE_FACTOR`] either way decides the point as `F′ − f`
    // evaluated as one expression would.  That expression is a true zero
    // for a right answer, which `evalf` can only settle by re-evaluating
    // the whole of `F′` at up to two and a half times the working precision
    // (the zero search to its cap), where `F′` alone settles in one pass:
    // up to 0.38 those re-evaluations were half of the integrator's time on
    // the Rubi suite's trigonometric chapters.  A distance near the
    // tolerance, or an `F′` that does not evaluate on its own, still goes
    // to the residual.
    let d_at = crate::transforms::subs::subs(arena, d_big_f, var, point);
    let d_val = numeric_value(arena, d_at);
    if let Some(d) = &d_val
        && let Some(gap) = crate::transforms::evalf::distance_to_f64(d, &f_val)
        && !gap.is_nan()
    {
        if gap * FTC_DECISIVE_FACTOR <= tol {
            return FtcPoint::Agrees { f_real };
        }
        if gap >= tol * FTC_DECISIVE_FACTOR {
            return FtcPoint::Differs { f_real };
        }
    }

    let r_at = crate::transforms::subs::subs(arena, residual, var, point);
    let gap = match residual_abs(arena, r_at) {
        Some(gap) => gap,
        None => {
            let Some(d_val) = d_val else {
                return FtcPoint::DerivativeUnevaluable;
            };
            match crate::transforms::evalf::distance_to_f64(&d_val, &f_val) {
                Some(gap) if !gap.is_nan() => gap,
                _ => return FtcPoint::DerivativeUnevaluable,
            }
        }
    };
    if gap <= tol {
        FtcPoint::Agrees { f_real }
    } else {
        FtcPoint::Differs { f_real }
    }
}

/// Does `e` (a function of `var` only) evaluate at one of the sample
/// points?
fn evaluates_at_a_sample_point(arena: &mut Arena, e: ExprId, var: ExprId) -> bool {
    FTC_SAMPLE_POINTS
        .iter()
        .chain(FTC_RETRY_POINTS.iter())
        .any(|&(p, q)| {
            let point = arena.rational(p, q);
            let e_at = crate::transforms::subs::subs(arena, e, var, point);
            numeric_value(arena, e_at).is_some()
        })
}

/// `e` with every `RootSum` whose polynomial and body are numeric and whose
/// body is a rational function of the root replaced by its exact value
/// ([`crate::calculus::risch::log_to_real::numeric_rootsum_rational_body`]):
/// the derivative of a `RootSum` answer at a sample point.  `eval` would
/// solve the polynomial in radicals at every point instead.
fn exact_numeric_root_sums(arena: &mut Arena, e: ExprId) -> ExprId {
    let order = crate::base::walk::post_order_ids(arena, e);
    let mut out = e;
    for id in order {
        if let ExprNode::RootSum(poly, body, sumvar) = *arena.node(id)
            && let Some(value) = crate::calculus::risch::log_to_real::numeric_rootsum_rational_body(
                arena, poly, body, sumvar,
            )
        {
            out = arena.subs_structural(out, id, value);
        }
    }
    out
}

/// A variable-free expression evaluated to [`FTC_CHECK_DIGITS`] correct
/// digits (after exact simplification); `None` when it does not evaluate
/// to a finite complex number.
fn numeric_value(arena: &mut Arena, e: ExprId) -> Option<crate::base::bigcomplex::Complex> {
    let e = exact_numeric_root_sums(arena, e);
    let e = crate::transforms::eval::eval(arena, e);
    if crate::base::walk::has_unevaluated(arena, e) {
        return None;
    }
    crate::transforms::evalf::evalf_complex(arena, e, FTC_CHECK_DIGITS).ok()
}

/// `|r|` for a variable-free `F′ − f` sample, evaluated at
/// [`FTC_CHECK_DIGITS`] so that a ~1e-13 discrepancy is not lost; `None`
/// when it does not evaluate to a finite complex number.
fn residual_abs(arena: &mut Arena, residual: ExprId) -> Option<f64> {
    let r = exact_numeric_root_sums(arena, residual);
    let r = crate::transforms::eval::eval(arena, r);
    if crate::base::walk::has_unevaluated(arena, r) {
        return None;
    }
    let v = if let Some(q) = arena.as_num(r) {
        crate::base::numeric::ratio_to_f64(q)?.abs()
    } else {
        let z = crate::transforms::evalf::evalf_complex(arena, r, FTC_CHECK_DIGITS).ok()?;
        crate::transforms::evalf::abs_to_f64(&z)?
    };
    v.is_finite().then_some(v)
}
