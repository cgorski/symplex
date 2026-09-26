//! Judging an antiderivative by differentiation.
//!
//! Every free symbol other than the integration variable is replaced by a
//! fixed generic rational (see [`table_value`]); then `F′` and `f` are
//! evaluated at the real points [`X_POINTS`] with `eval_decimal` (30 digits,
//! complex values allowed) and compared with relative tolerance
//! [`REL_TOL`].  A mismatch is re-evaluated at 80 digits before it counts,
//! so catastrophic cancellation in a large `F′` is not reported as a wrong
//! answer.  A point where either side fails to evaluate is skipped.
//!
//! [`verdict`] then applies symplex's real-variable convention: only points
//! where the integrand is real count against an answer, unless the
//! integrand contains `%i`.  When the five points decide nothing (the
//! integrand of `1/(x·√(ln²x − 3))` is real only for `x > e^√3 ≈ 5.65`
//! and `0 < x < e^−√3 ≈ 0.18`, and none of them lies there), [`check`]
//! scans the wider, log-spaced [`EXTRA_POINTS`] for points where the
//! integrand is real and compares up to [`EXTRA_WANTED`] of them.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use symplex::prelude::{Context, Ex};

use crate::protocol::Status;
use crate::translate::original_name;

/// The sample points for the integration variable, as `(p, q)` = `p/q`.
/// Negative points exercise the other side of `x = 0` (`ln|x|`, `sign x`,
/// even roots of odd powers).
pub const X_POINTS: [(i64, i64); 5] = [(1, 3), (7, 5), (13, 4), (-5, 7), (-13, 4)];
/// Points tried, in this order, when [`X_POINTS`] give no evidence
/// ([`verdict`] is `Undecided`): magnitudes about half a decade apart from
/// 10⁻² to 10² (`4/7, 9/4` next to the base points, then `≈ 0.1, 6.3, 0.03,
/// 10, 0.01, 31, 100`), each followed by its negative, so that an integrand
/// real only for small or only for large `|x|`, on either side of 0, is met
/// early.  Non-integers, distinct from the sample points and from every
/// parameter value (see the test).
pub const EXTRA_POINTS: [(i64, i64); 18] = [
    (4, 7),
    (-4, 7),
    (9, 4),
    (-9, 4),
    (7, 67),
    (-7, 67),
    (19, 3),
    (-19, 3),
    (3, 97),
    (-3, 97),
    (31, 3),
    (-31, 3),
    (1, 97),
    (-1, 97),
    (157, 5),
    (-157, 5),
    (199, 2),
    (-199, 2),
];
/// How many points of [`EXTRA_POINTS`] where the integrand is real and both
/// sides evaluate are compared at most.
pub const EXTRA_WANTED: usize = 3;
/// Relative tolerance on `|F′ − f| / max(|F′|, |f|)`.
pub const REL_TOL: f64 = 1e-8;
/// Digits for the first evaluation and for the re-evaluation of a mismatch.
const DIGITS: u32 = 30;
const RECHECK_DIGITS: u32 = 80;
/// A mismatch that shrinks by at least this factor from 30 to 80 digits is
/// rounding noise around a common value (typically both sides are 0 at the
/// point and one evaluates to 1e-50 at 30 digits, 1e-100 at 80): a real
/// difference does not depend on the working precision.
const NOISE_SHRINK: f64 = 1e-20;

/// Generic values for the parameters that occur in the suite, by Maxima
/// name.  Distinct, positive, non-integer, none equal to a sample point
/// (`a = 7/5` would make `a − x` vanish at `x = 7/5`), with pairwise
/// different denominators for the names that serve as exponents (`j k l m
/// n p q r s`), so that no accidental relation such as `m + 1 = n` holds.
pub fn table_value(name: &str) -> Option<(i64, i64)> {
    Some(match name {
        "a" => (6, 5),
        "b" => (3, 4),
        "c" => (5, 3),
        "d" => (2, 7),
        "e" => (11, 9),
        "f" => (13, 6),
        "g" => (5, 8),
        "h" => (9, 7),
        "i" => (8, 11),
        "j" => (7, 13),
        "k" => (10, 19),
        "l" => (11, 23),
        "m" => (4, 3),
        "n" => (5, 7),
        "p" => (3, 11),
        "q" => (2, 13),
        "r" => (7, 17),
        "s" => (9, 19),
        "t" => (3, 29),
        "u" => (5, 31),
        "v" => (6, 37),
        "w" => (8, 41),
        "y" => (10, 43),
        "z" => (12, 47),
        "A" => (19, 10),
        "B" => (23, 12),
        "C" => (29, 14),
        "D" => (31, 15),
        "E" => (37, 16),
        "F" => (5, 2),
        "G" => (41, 18),
        "H" => (43, 20),
        "I" => (47, 21),
        "K" => (53, 22),
        _ => return None,
    })
}

/// Values for names not in [`table_value`], handed out in order.
const POOL: [(i64, i64); 12] = [
    (59, 24),
    (61, 26),
    (67, 27),
    (71, 28),
    (73, 30),
    (79, 32),
    (83, 33),
    (89, 34),
    (97, 35),
    (101, 36),
    (103, 38),
    (107, 39),
];

/// The substitution for the parameters.
pub struct Env {
    pub pairs: Vec<(Ex, Ex)>,
    /// `a=7/5, b=3/4, …` (Maxima names).
    pub desc: String,
    names: BTreeSet<String>,
    pool_next: usize,
    /// `1` for the table values, `-1` for their negatives
    /// (`--negative-params`).
    sign: i64,
}

impl Env {
    pub fn new() -> Env {
        Env {
            pairs: Vec::new(),
            desc: String::new(),
            names: BTreeSet::new(),
            pool_next: 0,
            sign: 1,
        }
    }

    /// The second parameter set: every value of [`table_value`] and
    /// [`POOL`] negated.  Answers that are right only for positive
    /// parameters (`√a` for `√(a²)`, `ln a` for `ln|a|`, a branch chosen
    /// for `a > 0`) fail here.
    pub fn negated() -> Env {
        Env {
            sign: -1,
            ..Env::new()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Add every free symbol of `exprs` other than `x` that is not bound yet.
    pub fn extend(&mut self, ctx: &Context, x: &Ex, exprs: &[&Ex]) {
        let mut new: Vec<(String, Ex)> = Vec::new();
        for e in exprs {
            for s in e.free_symbols() {
                if &s == x {
                    continue;
                }
                let name = s.to_string();
                if self.names.contains(&name) || new.iter().any(|(n, _)| *n == name) {
                    continue;
                }
                new.push((name, s));
            }
        }
        new.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, s) in new {
            let (p, q) = table_value(original_name(&name)).unwrap_or_else(|| {
                let v = POOL[self.pool_next % POOL.len()];
                self.pool_next += 1;
                v
            });
            let p = self.sign * p;
            if !self.desc.is_empty() {
                self.desc.push_str(", ");
            }
            let _ = write!(self.desc, "{}={p}/{q}", original_name(&name));
            self.pairs.push((s, ctx.rational(p, q)));
            self.names.insert(name);
        }
    }

    pub fn apply(&self, e: &Ex) -> Ex {
        if self.pairs.is_empty() {
            return e.clone();
        }
        let refs: Vec<(&Ex, &Ex)> = self.pairs.iter().map(|(a, b)| (a, b)).collect();
        e.subs_map(&refs)
    }
}

/// A complex value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C {
    pub re: f64,
    pub im: f64,
}

impl C {
    fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    pub fn is_real(self) -> bool {
        self.im.abs() <= 1e-12 * self.norm()
    }
}

impl std::fmt::Display for C {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im == 0.0 {
            write!(f, "{:e}", self.re)
        } else if self.im < 0.0 {
            write!(f, "{:e} - {:e}*I", self.re, -self.im)
        } else {
            write!(f, "{:e} + {:e}*I", self.re, self.im)
        }
    }
}

/// Parse `eval_decimal` output: `1.5`, `-2e-7`, `3*i`, `-i`, `1 + 2*i`,
/// `1 - i`.  Infinities (`oo`) and anything else are `None`.
pub fn parse_complex(s: &str) -> Option<C> {
    let s = s.trim();
    let imag = |t: &str| -> Option<f64> {
        let t = t.trim();
        match t {
            "i" => Some(1.0),
            "-i" => Some(-1.0),
            _ => t
                .strip_suffix("*i")
                .and_then(|m| m.trim().parse::<f64>().ok()),
        }
    };
    let finite = |c: C| (c.re.is_finite() && c.im.is_finite()).then_some(c);
    if let Some(pos) = s.find(" + ") {
        let re = s[..pos].parse::<f64>().ok()?;
        return finite(C {
            re,
            im: imag(&s[pos + 3..])?,
        });
    }
    if let Some(pos) = s.find(" - ") {
        let re = s[..pos].parse::<f64>().ok()?;
        return finite(C {
            re,
            im: -imag(&s[pos + 3..])?,
        });
    }
    if s.ends_with('i') {
        return finite(C {
            re: 0.0,
            im: imag(s)?,
        });
    }
    finite(C {
        re: s.parse::<f64>().ok()?,
        im: 0.0,
    })
}

fn distance(a: C, b: C) -> f64 {
    C {
        re: a.re - b.re,
        im: a.im - b.im,
    }
    .norm()
}

fn close(a: C, b: C) -> bool {
    distance(a, b) <= REL_TOL * a.norm().max(b.norm())
}

fn eval_at(e: &Ex, x: &Ex, xv: &Ex, digits: u32) -> Result<C, String> {
    let v = e.subs(x, xv);
    let s = v.eval_decimal(digits).map_err(|err| err.to_string())?;
    parse_complex(&s).ok_or_else(|| format!("not a finite number: {s}"))
}

/// The comparison at one point.
#[derive(Debug, Clone)]
pub enum Outcome {
    Ok { lhs: C, rhs: C },
    Fail { lhs: C, rhs: C },
    Skip(String),
}

#[derive(Debug, Clone)]
pub struct PointReport {
    pub x: (i64, i64),
    pub outcome: Outcome,
}

impl PointReport {
    pub fn failed(&self) -> bool {
        matches!(self.outcome, Outcome::Fail { .. })
    }

    /// Is the integrand real at this point (only known if it evaluated)?
    pub fn integrand_real(&self) -> Option<bool> {
        match self.outcome {
            Outcome::Ok { rhs, .. } | Outcome::Fail { rhs, .. } => Some(rhs.is_real()),
            Outcome::Skip(_) => None,
        }
    }

    pub fn describe(&self, var: &str) -> String {
        let (p, q) = self.x;
        match &self.outcome {
            Outcome::Ok { lhs, rhs } => format!("{var}={p}/{q}: ok  F'={lhs}  f={rhs}"),
            Outcome::Fail { lhs, rhs } => format!(
                "{var}={p}/{q}: MISMATCH  F'={lhs}  f={rhs}  (integrand {} here)",
                if rhs.is_real() { "real" } else { "complex" }
            ),
            Outcome::Skip(why) => format!("{var}={p}/{q}: skipped ({why})"),
        }
    }
}

/// Compare `lhs` and `rhs` (functions of `x` only) at `x = p/q`.
pub fn compare_at(ctx: &Context, lhs: &Ex, rhs: &Ex, x: &Ex, (p, q): (i64, i64)) -> Outcome {
    let xv = ctx.rational(p, q);
    let a = eval_at(lhs, x, &xv, DIGITS);
    let b = eval_at(rhs, x, &xv, DIGITS);
    match (a, b) {
        (Ok(a), Ok(b)) => {
            if close(a, b) {
                return Outcome::Ok { lhs: a, rhs: b };
            }
            // Re-evaluate at higher precision before calling it a mismatch.
            match (
                eval_at(lhs, x, &xv, RECHECK_DIGITS),
                eval_at(rhs, x, &xv, RECHECK_DIGITS),
            ) {
                (Ok(a2), Ok(b2)) if close(a2, b2) => Outcome::Ok { lhs: a2, rhs: b2 },
                (Ok(a2), Ok(b2)) if distance(a2, b2) <= NOISE_SHRINK * distance(a, b) => {
                    Outcome::Ok { lhs: a2, rhs: b2 }
                }
                (Ok(a2), Ok(b2)) => Outcome::Fail { lhs: a2, rhs: b2 },
                _ => Outcome::Fail { lhs: a, rhs: b },
            }
        }
        (Err(e), _) => Outcome::Skip(format!("F' does not evaluate: {e}")),
        (_, Err(e)) => Outcome::Skip(format!("f does not evaluate: {e}")),
    }
}

/// The classification of a checked answer.
///
/// - `Wrong` if some point mismatches where the integrand is real, or if
///   the integrand contains `%i` (complex by construction, so `ln|u|` & co.
///   are genuine errors) and some point mismatches at all;
/// - `RealVerified` if every mismatch is at a point where the integrand is
///   complex (the real-variable convention) and some point where it is
///   real agrees;
/// - `Verified` if every point where both sides evaluate agrees;
/// - `Undecided` if no point evaluated on both sides, or if the only
///   points that did are mismatches where the integrand is complex.  The
///   real-variable convention excuses such a mismatch, but it is no
///   evidence either: up to 0.28 an answer with no agreeing point at all
///   (`ln|x·√(−a) + a|/√(−a)` for `1/(a + x·√(−a))`, whose derivative is
///   the conjugate of the integrand at every point) was `RealVerified`.
pub fn verdict(reports: &[PointReport], integrand_has_i: bool) -> Status {
    let failed = || reports.iter().filter(|r| r.failed());
    if failed().next().is_some() {
        if integrand_has_i || failed().any(|r| r.integrand_real() == Some(true)) {
            Status::Wrong
        } else if real_agreements(reports) > 0 {
            Status::RealVerified
        } else {
            Status::Undecided
        }
    } else if reports
        .iter()
        .any(|r| matches!(r.outcome, Outcome::Ok { .. }))
    {
        Status::Verified
    } else {
        Status::Undecided
    }
}

/// Points where the integrand is real and both sides agree.
pub fn real_agreements(reports: &[PointReport]) -> usize {
    reports
        .iter()
        .filter(|r| matches!(r.outcome, Outcome::Ok { rhs, .. } if rhs.is_real()))
        .count()
}

/// `F′` against `f` at every sample point of [`X_POINTS`] and, if those
/// give no evidence ([`verdict`] with `integrand_has_i` is `Undecided`), at
/// up to [`EXTRA_WANTED`] points of [`EXTRA_POINTS`], in order, where the
/// integrand is real and both sides evaluate.  Both are functions of `x`
/// and the parameters bound by `env`.  Deterministic: the points depend on
/// the integrand only.
pub fn check(
    ctx: &Context,
    f: &Ex,
    big_f: &Ex,
    x: &Ex,
    env: &Env,
    integrand_has_i: bool,
) -> Vec<PointReport> {
    let f_s = env.apply(f);
    let d = env.apply(big_f).diff(x);
    let mut reports: Vec<PointReport> = X_POINTS
        .iter()
        .map(|&pt| PointReport {
            x: pt,
            outcome: compare_at(ctx, &d, &f_s, x, pt),
        })
        .collect();
    if verdict(&reports, integrand_has_i) != Status::Undecided {
        return reports;
    }
    let mut compared = 0;
    for &(p, q) in &EXTRA_POINTS {
        if compared == EXTRA_WANTED {
            break;
        }
        // The integrand first: it is cheaper than `F′`, and a point where it
        // is complex would be no evidence either.
        if !matches!(eval_at(&f_s, x, &ctx.rational(p, q), DIGITS), Ok(c) if c.is_real()) {
            continue;
        }
        let outcome = compare_at(ctx, &d, &f_s, x, (p, q));
        if matches!(outcome, Outcome::Skip(_)) {
            continue;
        }
        compared += 1;
        reports.push(PointReport { x: (p, q), outcome });
    }
    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_eval_decimal_output() {
        assert_eq!(parse_complex("1.5"), Some(C { re: 1.5, im: 0.0 }));
        assert_eq!(parse_complex("-2e-7"), Some(C { re: -2e-7, im: 0.0 }));
        assert_eq!(parse_complex("i"), Some(C { re: 0.0, im: 1.0 }));
        assert_eq!(parse_complex("-3.5*i"), Some(C { re: 0.0, im: -3.5 }));
        assert_eq!(parse_complex("1 + 2*i"), Some(C { re: 1.0, im: 2.0 }));
        assert_eq!(
            parse_complex("-1.5e-3 - i"),
            Some(C {
                re: -1.5e-3,
                im: -1.0
            })
        );
        assert_eq!(parse_complex("oo"), None);
    }

    #[test]
    fn checks_a_right_and_a_wrong_antiderivative() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let f = ctx.parse("sin(a*x)").unwrap();
        let good = ctx.parse("-cos(a*x)/a").unwrap();
        let bad = ctx.parse("-cos(a*x)").unwrap();
        let mut env = Env::new();
        env.extend(&ctx, &x, &[&f, &good]);
        assert_eq!(env.desc, "a=6/5");
        assert!(
            check(&ctx, &f, &good, &x, &env, false)
                .iter()
                .all(|r| matches!(r.outcome, Outcome::Ok { .. }))
        );
        assert!(
            check(&ctx, &f, &bad, &x, &env, false)
                .iter()
                .all(PointReport::failed)
        );
    }

    #[test]
    fn an_integrand_real_away_from_the_sample_points_gets_real_evidence() {
        // Rubi 3.5 #134: 1/(x·√(ln²x − 3)) is real only for x > e^√3 and
        // 0 < x < e^−√3; at the five sample points it is complex and the
        // answer ln|√(ln²x − 3) + ln x| (right on the real domain) mismatches,
        // so the case was undecided.
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let f = ctx.parse("1/(x*sqrt(ln(x)^2 - 3))").unwrap();
        let good = ctx.parse("ln(abs(sqrt(ln(x)^2 - 3) + ln(x)))").unwrap();
        let env = Env::new();
        let r = check(&ctx, &f, &good, &x, &env, false);
        assert_eq!(r.len(), X_POINTS.len() + EXTRA_WANTED, "{r:?}");
        let extra: Vec<(i64, i64)> = r[X_POINTS.len()..].iter().map(|r| r.x).collect();
        assert_eq!(extra, vec![(7, 67), (19, 3), (3, 97)]);
        assert_eq!(real_agreements(&r), EXTRA_WANTED);
        assert_eq!(verdict(&r, false), Status::RealVerified);
        // A wrong answer is now caught where the integrand is real.
        let bad = ctx.parse("ln(abs(sqrt(ln(x)^2 - 3) - ln(x)))").unwrap();
        let r = check(&ctx, &f, &bad, &x, &env, false);
        assert_eq!(verdict(&r, false), Status::Wrong);
        // Decided by the sample points: nothing more is evaluated.
        let f = ctx.parse("sin(x)").unwrap();
        let r = check(&ctx, &f, &ctx.parse("-cos(x)").unwrap(), &x, &env, false);
        assert_eq!(r.len(), X_POINTS.len());
        // Complex everywhere: the scan finds no point and adds none.
        let f = ctx.parse("sqrt(-1 - x^2)").unwrap();
        let r = check(&ctx, &f, &ctx.parse("x").unwrap(), &x, &env, false);
        assert_eq!(r.len(), X_POINTS.len());
        assert_eq!(verdict(&r, false), Status::Undecided);
    }

    #[test]
    fn extra_points_avoid_the_sample_points_and_the_parameter_values() {
        let same = |(p, q): (i64, i64), (r, s): (i64, i64)| p * s == r * q;
        let params: Vec<(i64, i64)> = (b'a'..=b'z')
            .chain(b'A'..=b'Z')
            .filter_map(|c| table_value(&(c as char).to_string()))
            .chain(POOL)
            .flat_map(|(p, q)| [(p, q), (-p, q)])
            .collect();
        for (i, &pt) in EXTRA_POINTS.iter().enumerate() {
            assert!(pt.0 % pt.1 != 0, "{pt:?} is an integer");
            assert!(!X_POINTS.iter().any(|&b| same(b, pt)), "{pt:?}");
            assert!(!params.iter().any(|&v| same(v, pt)), "{pt:?}");
            assert!(!EXTRA_POINTS[..i].iter().any(|&e| same(e, pt)), "{pt:?}");
        }
    }

    #[test]
    fn the_negated_set_catches_an_answer_right_only_for_positive_parameters() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let f = ctx.parse("sqrt(a^2)").unwrap();
        let big_f = ctx.parse("a*x").unwrap();
        let mut pos = Env::new();
        pos.extend(&ctx, &x, &[&f, &big_f]);
        let mut neg = Env::negated();
        neg.extend(&ctx, &x, &[&f, &big_f]);
        assert_eq!(neg.desc, "a=-6/5");
        let judge = |env: &Env| verdict(&check(&ctx, &f, &big_f, &x, env, false), false);
        assert_eq!(judge(&pos), Status::Verified);
        assert_eq!(judge(&neg), Status::Wrong);
    }

    #[test]
    fn rounding_noise_at_a_common_zero_is_not_a_mismatch() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        // Identically zero, but evaluated term by term: rounding noise
        // (or an exact 0) against an exact 0.
        let lhs = ctx.parse("sin(x)^2 + cos(x)^2 - 1").unwrap();
        let zero = ctx.parse("0*x").unwrap();
        for pt in X_POINTS {
            assert!(matches!(
                compare_at(&ctx, &lhs, &zero, &x, pt),
                Outcome::Ok { .. }
            ));
        }
        let one = ctx.parse("sin(x)^2 + cos(x)^2").unwrap();
        assert!(matches!(
            compare_at(&ctx, &one, &zero, &x, (7, 5)),
            Outcome::Fail { .. }
        ));
    }

    #[test]
    fn verdicts_follow_the_real_variable_convention() {
        let real = C { re: 1.0, im: 0.0 };
        let cplx = C { re: 1.0, im: 1.0 };
        let other = C { re: 2.0, im: 0.0 };
        let pt = |outcome| PointReport { x: (1, 3), outcome };
        let ok_real = || {
            pt(Outcome::Ok {
                lhs: real,
                rhs: real,
            })
        };
        let fail_real = || {
            pt(Outcome::Fail {
                lhs: other,
                rhs: real,
            })
        };
        let fail_cplx = || {
            pt(Outcome::Fail {
                lhs: other,
                rhs: cplx,
            })
        };
        let skip = || pt(Outcome::Skip("pole".into()));

        assert_eq!(verdict(&[ok_real(), skip()], false), Status::Verified);
        assert_eq!(
            verdict(&[ok_real(), fail_cplx()], false),
            Status::RealVerified
        );
        assert_eq!(verdict(&[ok_real(), fail_cplx()], true), Status::Wrong);
        assert_eq!(verdict(&[fail_cplx(), fail_real()], false), Status::Wrong);
        assert_eq!(verdict(&[skip(), skip()], false), Status::Undecided);
        // Mismatches only where the integrand is complex, and no agreement
        // where it is real: excused, but not evidence.
        assert_eq!(
            verdict(&[fail_cplx(), fail_cplx(), skip()], false),
            Status::Undecided
        );
        let ok_cplx = || {
            pt(Outcome::Ok {
                lhs: cplx,
                rhs: cplx,
            })
        };
        assert_eq!(verdict(&[ok_cplx(), fail_cplx()], false), Status::Undecided);
        assert_eq!(verdict(&[fail_cplx()], true), Status::Wrong);
        assert_eq!(real_agreements(&[ok_real(), fail_cplx()]), 1);
    }

    #[test]
    fn complex_values_compare() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        // At x = 13/4 the integrand is imaginary; asin is still an
        // antiderivative there with principal branches.
        let f = ctx.parse("1/sqrt(1 - x^2)").unwrap();
        let big_f = ctx.parse("asin(x)").unwrap();
        let env = Env::new();
        let r = check(&ctx, &f, &big_f, &x, &env, false);
        assert!(r.iter().all(|r| !r.failed()), "{r:?}");
    }
}
