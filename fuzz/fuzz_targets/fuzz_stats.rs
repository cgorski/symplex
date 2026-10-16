//! The symbolic distribution layer (`stats::Distribution`): properties every
//! family must have, for every valid parameter and argument.
//!
//! A family — the 23 built-in ones and finite tables — gets parameters
//! drawn as exact numbers: small fractions, integers, dyadic numbers over
//! `2^±20`, powers of ten up to `10^±300` and dyadics over `2^±900`
//! (extreme shapes and scales), and one draw in eight a symbol-free
//! irrational constant (`√2`, `π`, `e`, `ln 2`, …).  One draw in eight
//! makes a parameter invalid (zero, negative, outside `[0, 1]`, a
//! non-integer count, a broken relation `a < b`, `±∞`, NaN, non-real); one
//! valid draw in three is wrapped once (`affine`, `truncated`, an order
//! statistic, a two-component mixture, a monotone `transformed`).  Claims
//! are decided exactly where they are exact (rational parameters give
//! rational masses, moments and CDF values) and otherwise from certified
//! evaluation — `eval_f64` rounds certified digits once, `eval_decimal`
//! prints certified digits only — so a failure is a wrong answer:
//!
//! * **validation** (every input): the `try_` constructor accepts valid
//!   parameters and rejects invalid ones with an error (never a panic or a
//!   distribution); the unchecked distribution does not panic either;
//! * **density**: real and `≥ 0` at points of the support; integrates to 1
//!   (`integrate_numeric`, a disagreement re-decided by the certified
//!   `integrate_numeric_decimal`), sums to 1 (exactly on a finite support
//!   with rational masses; partial sums plus a ratio bound on the tail for
//!   `Poisson`, `Geometric`, `NegativeBinomial` and a large `Binomial`);
//! * **cdf**: `cdf`, `sf` in `[0, 1]`, monotone, `cdf + sf = 1`, `0`/`1`
//!   beyond and at the support's ends, limits `0`/`1` at its ends;
//! * **derivative**: `d/dx` of the family's closed `cdf`, `cdf_lower` and
//!   `−sf` is the density, and the closed forms agree with each other; on
//!   a lattice `cdf(k) − cdf(k⁻) = sf(k⁻) − sf(k) = pmf(k)`;
//! * **quantile**: a closed quantile inverts the CDF both ways
//!   (`cdf(quantile(p)) = p`, `quantile(cdf(x)) = x`); `quantile_f64` and
//!   `isf_f64` agree with it, or else meet the level (the tail at the
//!   result matches it to `1e-9`, or the tails 16 doubles either side
//!   bracket it); on a lattice `cdf(k) ≥ p > cdf(k⁻)` and
//!   `sf(k) ≤ q < sf(k⁻)`, decided exactly;
//! * **moments**: a moment that exists by the family's own theory (`n < ν`
//!   for a Student t, `n < α` for a Pareto, `n < d₂/2` for an F, none for a
//!   Cauchy, all of them on a bounded support) is finite and matches the
//!   integral — `∫ zⁿ g(z) dz` of the standardized density
//!   `g(z) = σ f(μ + σz)` is `0, 1, skewness, kurtosis` for `n = 1…4` — or
//!   the exact sum; one that does not exist is `±∞` or NaN, never finite;
//! * **mgf / cf**: `mgf(t)` is `+∞` outside the family's domain of
//!   convergence and matches `∫ e^{tx} f` inside; `mgf′(0) = mean`,
//!   `mgf″(0) = E[X²]` for the closed mgf; `|cf(t)| ≤ 1`,
//!   `cf(t) = ∫ e^{itx} f`, and `cf(t) = mgf(it)` where the mgf exists
//!   around 0;
//! * **numdist**: for the eight families of `stats::numdist` the symbolic
//!   `cdf` and `sf` (and the density of the normal and t) agree with the
//!   `f64` kernels to `1e-12` relative where the value is at least
//!   `1e-100`, and to the kernels' documented `≈ 8ε·|ln P|` below (to
//!   `1e-290`); parameters and points are doubles, so both sides see the
//!   same numbers;
//! * **order statistics** of finite tables: the enumerated table has
//!   masses `≥ 0` summing to 1, equal to a brute-force enumeration of
//!   every sample;
//! * **sums**: `sum_distribution` of independent normals, gammas of a
//!   common scale (exponentials, χ²), Poissons, binomials of a common `p`
//!   (Bernoulli) and negative binomials: the result's mean, variance and
//!   third central moment are the sums; mismatched pairs and `X + X` are
//!   not closures (`None`, or a result with the right moments);
//! * **transformed** by a monotone map `g`: `P(Y ≤ g(x))` is `P(X ≤ x)`
//!   (increasing) or `P(X ≥ x)` (decreasing), and the density is
//!   `f(x)/|g′(x)|` (the mass at `x` on a table).
//!
//! The first byte picks the check.  Each input stops starting new work
//! after about five seconds; what refuses (an `Err`, an unevaluated value,
//! a quadrature that does not converge) is skipped.  A failure panics with
//! the case and the values; `FUZZ_SHOW=1` prints the decoded case, the
//! points and the verdict.
#![no_main]

#[path = "common/choose.rs"]
mod choose;

use std::cmp::Ordering;
use std::time::Instant;

use choose::Src;
use libfuzzer_sys::fuzz_target;
use symplex::definite::QuadRule;
use symplex::num_bigint::BigInt;
use symplex::num_traits::{One, ToPrimitive, Zero};
use symplex::prelude::*;
use symplex::stats::numdist;
use symplex::stats::order::order_statistic;
use symplex::stats::{Distribution, Kind, Piece, RandomVariable, Support, sum_distribution};

type Check = Result<String, String>;

macro_rules! fail {
    ($($t:tt)*) => { return Err(format!($($t)*)) };
}

/// `FUZZ_SHOW` is set: print the case as soon as it is drawn and each step.
fn showing() -> bool {
    static SHOW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SHOW.get_or_init(|| std::env::var_os("FUZZ_SHOW").is_some())
}

macro_rules! show {
    ($($t:tt)*) => {
        if showing() {
            eprintln!($($t)*);
        }
    };
}

/// Record the case description (and print it under `FUZZ_SHOW`).
fn describe(desc: &mut String, text: String) {
    show!("  case: {text}");
    *desc = text;
}

// ── Families ────────────────────────────────────────────────────────────────

const NORMAL: usize = 0;
const UNIFORM: usize = 1;
const EXPONENTIAL: usize = 2;
const GAMMA: usize = 3;
const CHI2: usize = 4;
const BETA: usize = 5;
const CAUCHY: usize = 6;
const LAPLACE: usize = 7;
const LOGISTIC: usize = 8;
const LOGNORMAL: usize = 9;
const STUDENT_T: usize = 10;
const F_DIST: usize = 11;
const WEIBULL: usize = 12;
const PARETO: usize = 13;
const TRIANGULAR: usize = 14;
const BERNOULLI: usize = 15;
const BINOMIAL: usize = 16;
const POISSON: usize = 17;
const GEOMETRIC: usize = 18;
const NEG_BINOMIAL: usize = 19;
const HYPERGEOMETRIC: usize = 20;
const DISCRETE_UNIFORM: usize = 21;
const DIE: usize = 22;
const FINITE: usize = 23;
const FAMILIES: u64 = 24;

const NAMES: [&str; 24] = [
    "Normal",
    "Uniform",
    "Exponential",
    "Gamma",
    "ChiSquared",
    "Beta",
    "Cauchy",
    "Laplace",
    "Logistic",
    "LogNormal",
    "StudentT",
    "FDistribution",
    "Weibull",
    "Pareto",
    "Triangular",
    "Bernoulli",
    "Binomial",
    "Poisson",
    "Geometric",
    "NegativeBinomial",
    "Hypergeometric",
    "DiscreteUniform",
    "Die",
    "Finite",
];

/// What a parameter must be.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Role {
    /// A finite real number (a location).
    Real,
    /// A finite positive number (a scale, shape or rate).
    Pos,
    /// A probability in `[0, 1]`.
    Prob,
    /// A probability in `(0, 1]`.
    ProbPos,
    /// A probability in `(0, 1)`.
    ProbOpen,
    /// A non-negative integer.
    Count,
    /// An integer.
    Int,
    /// A positive integer.
    PosInt,
}

fn roles(fam: usize) -> &'static [Role] {
    use Role::*;
    match fam {
        NORMAL | CAUCHY | LAPLACE | LOGISTIC | LOGNORMAL => &[Real, Pos],
        UNIFORM => &[Real, Real],
        EXPONENTIAL | CHI2 | STUDENT_T | POISSON => &[Pos],
        GAMMA | BETA | F_DIST | WEIBULL | PARETO => &[Pos, Pos],
        TRIANGULAR => &[Real, Real, Real],
        BERNOULLI => &[Prob],
        BINOMIAL => &[Count, Prob],
        GEOMETRIC => &[ProbPos],
        NEG_BINOMIAL => &[Pos, ProbOpen],
        HYPERGEOMETRIC => &[Count, Count, Count],
        DISCRETE_UNIFORM => &[Int, Int],
        DIE => &[PosInt],
        _ => &[],
    }
}

/// A drawn family with its parameters (or table) and whether they are valid.
struct Drawn {
    fam: usize,
    params: Vec<Ex>,
    table: Vec<(Ex, Ex)>,
    valid: bool,
    why: String,
}

impl Drawn {
    fn new(fam: usize) -> Self {
        Drawn {
            fam,
            params: Vec::new(),
            table: Vec::new(),
            valid: true,
            why: String::new(),
        }
    }

    fn text(&self) -> String {
        if self.fam == FINITE {
            let cells: Vec<String> = self
                .table
                .iter()
                .map(|(v, p)| format!("{v}: {p}"))
                .collect();
            format!("Finite({{{}}})", cells.join(", "))
        } else {
            let ps: Vec<String> = self.params.iter().map(ToString::to_string).collect();
            format!("{}({})", NAMES[self.fam], ps.join(", "))
        }
    }

    /// The distribution through the `try_` constructor (`checked`) or the
    /// unchecked one.
    fn build(&self, ctx: &Context, checked: bool) -> Result<Distribution, SymplexError> {
        let a = |i: usize| self.params.get(i).cloned().unwrap_or_else(|| ctx.zero());
        macro_rules! mk {
            ($t:ident, $u:ident $(, $i:expr)*) => {
                if checked {
                    Distribution::$t($(a($i)),*)
                } else {
                    Ok(Distribution::$u($(a($i)),*))
                }
            };
        }
        match self.fam {
            NORMAL => mk!(try_normal, normal, 0, 1),
            UNIFORM => mk!(try_uniform, uniform, 0, 1),
            EXPONENTIAL => mk!(try_exponential, exponential, 0),
            GAMMA => mk!(try_gamma, gamma, 0, 1),
            CHI2 => mk!(try_chi_squared, chi_squared, 0),
            BETA => mk!(try_beta, beta, 0, 1),
            CAUCHY => mk!(try_cauchy, cauchy, 0, 1),
            LAPLACE => mk!(try_laplace, laplace, 0, 1),
            LOGISTIC => mk!(try_logistic, logistic, 0, 1),
            LOGNORMAL => mk!(try_log_normal, log_normal, 0, 1),
            STUDENT_T => mk!(try_student_t, student_t, 0),
            F_DIST => mk!(try_f_distribution, f_distribution, 0, 1),
            WEIBULL => mk!(try_weibull, weibull, 0, 1),
            PARETO => mk!(try_pareto, pareto, 0, 1),
            TRIANGULAR => mk!(try_triangular, triangular, 0, 1, 2),
            BERNOULLI => mk!(try_bernoulli, bernoulli, 0),
            BINOMIAL => mk!(try_binomial, binomial, 0, 1),
            POISSON => mk!(try_poisson, poisson, 0),
            GEOMETRIC => mk!(try_geometric, geometric, 0),
            NEG_BINOMIAL => mk!(try_negative_binomial, negative_binomial, 0, 1),
            HYPERGEOMETRIC => mk!(try_hypergeometric, hypergeometric, 0, 1, 2),
            DISCRETE_UNIFORM => mk!(try_discrete_uniform, discrete_uniform, 0, 1),
            DIE => mk!(try_die, die, 0),
            _ => {
                if checked {
                    Distribution::try_finite(ctx, self.table.clone())
                } else {
                    Ok(Distribution::finite(ctx, self.table.clone()))
                }
            }
        }
    }
}

// ── Numbers ─────────────────────────────────────────────────────────────────

fn small_rational(r: &mut Src, ctx: &Context, max_p: i64) -> Ex {
    let p = r.range(1, max_p);
    let q = r.range(1, 16);
    ctx.rational(p, q)
}

/// `m·2^e` with an odd 16-bit mantissa: exact as a rational and as a double
/// (for `|e| ≤ 1000`).
fn dyadic(r: &mut Src, ctx: &Context, lo: i64, hi: i64) -> Ex {
    let m = 1 + 2 * r.below(1 << 15) as i64;
    let e = r.range(lo, hi);
    ctx.int(m) * ctx.int(2).powi(e)
}

/// `m·10^k` or `m·10^−k`, `m ≤ 9`, `1 ≤ k ≤ max_k`.
fn ten_power(r: &mut Src, ctx: &Context, max_k: i64) -> Ex {
    let m = r.range(1, 9);
    let k = r.range(1, max_k.max(1));
    let t = ctx.int(10).powi(k);
    if r.chance(0.5) {
        ctx.int(m) * t
    } else {
        ctx.int(m) / t
    }
}

/// A symbol-free irrational constant times a small rational.
fn irrational(r: &mut Src, ctx: &Context) -> Ex {
    let c = match r.below(7) {
        0 => ctx.int(2).sqrt(),
        1 => ctx.pi(),
        2 => ctx.e(),
        3 => ctx.int(2).ln(),
        4 => ctx.int(3).sqrt() / 2,
        5 => ctx.pi().powi(2) / 6,
        _ => ctx.int(5).sqrt().powi(-1),
    };
    c * small_rational(r, ctx, 16)
}

/// A positive number.
fn positive(r: &mut Src, ctx: &Context) -> Ex {
    match r.below(16) {
        0..=4 => small_rational(r, ctx, 64),
        5..=7 => dyadic(r, ctx, -20, 20),
        8 | 9 => ctx.int(r.range(1, 1000)),
        10 | 11 => ten_power(r, ctx, 300),
        12 => dyadic(r, ctx, -900, 900),
        13 => small_rational(r, ctx, 4),
        _ => irrational(r, ctx),
    }
}

/// A real number (a location).
fn real(r: &mut Src, ctx: &Context) -> Ex {
    match r.below(8) {
        0 => ctx.zero(),
        1..=5 => {
            let v = positive(r, ctx);
            if r.chance(0.5) { -v } else { v }
        }
        _ => ctx.rational(r.range(-64, 64), r.range(1, 16)),
    }
}

/// A probability in `(0, 1)`.
fn probability(r: &mut Src, ctx: &Context) -> Ex {
    match r.below(10) {
        0..=4 => {
            let q = r.range(2, 64);
            ctx.rational(r.range(1, q - 1), q)
        }
        5 => ctx.int(r.range(1, 9)) / ctx.int(10).powi(r.range(1, 300)),
        6 => ctx.one() - ctx.int(r.range(1, 9)) / ctx.int(10).powi(r.range(1, 30)),
        7 => match r.below(4) {
            0 => ctx.int(2).sqrt() / 2,
            1 => ctx.pi() / 4,
            2 => ctx.e().powi(-1),
            _ => ctx.int(2).ln(),
        },
        _ => ctx.rational(1 + r.below(65535) as i64, 65536),
    }
}

/// A non-negative integer: mostly small, sometimes up to `9·10^17`.
fn count(r: &mut Src, ctx: &Context) -> Ex {
    match r.below(8) {
        0..=5 => ctx.int(r.range(0, 40)),
        6 => ctx.int(r.range(41, 2000)),
        _ => ctx.int(r.range(1, 9)) * ctx.int(10).powi(r.range(3, 17)),
    }
}

/// An integer of either sign.
fn integer(r: &mut Src, ctx: &Context) -> Ex {
    match r.below(8) {
        0..=6 => ctx.int(r.range(-50, 50)),
        _ => {
            let v = ctx.int(r.range(1, 9)) * ctx.int(10).powi(r.range(3, 30));
            if r.chance(0.5) { -v } else { v }
        }
    }
}

fn draw_param(role: Role, r: &mut Src, ctx: &Context) -> Ex {
    match role {
        Role::Real => real(r, ctx),
        Role::Pos => positive(r, ctx),
        Role::Prob => match r.below(16) {
            0 => ctx.zero(),
            1 => ctx.one(),
            _ => probability(r, ctx),
        },
        Role::ProbPos => {
            if r.below(16) == 0 {
                ctx.one()
            } else {
                probability(r, ctx)
            }
        }
        Role::ProbOpen => probability(r, ctx),
        Role::Count => count(r, ctx),
        Role::Int => integer(r, ctx),
        Role::PosInt => {
            if r.below(8) == 0 {
                ctx.int(r.range(1, 9)) * ctx.int(10).powi(r.range(3, 12))
            } else {
                ctx.int(r.range(1, 30))
            }
        }
    }
}

/// A value that is invalid for `role` (`valid` is the value it replaces).
fn invalid_value(role: Role, r: &mut Src, ctx: &Context, valid: &Ex) -> Ex {
    match r.below(10) {
        0 => ctx.infinity(),
        1 => ctx.nan(),
        2 => ctx.neg_infinity(),
        3 => valid + ctx.i_unit(),
        _ => match role {
            Role::Real => ctx.complex_infinity(),
            Role::Pos => {
                if r.chance(0.3) {
                    ctx.zero()
                } else {
                    -positive(r, ctx)
                }
            }
            Role::Prob => {
                if r.chance(0.5) {
                    ctx.one() + probability(r, ctx)
                } else {
                    -probability(r, ctx)
                }
            }
            Role::ProbPos => match r.below(3) {
                0 => ctx.zero(),
                1 => ctx.one() + probability(r, ctx),
                _ => -probability(r, ctx),
            },
            Role::ProbOpen => match r.below(4) {
                0 => ctx.zero(),
                1 => ctx.one(),
                2 => ctx.one() + probability(r, ctx),
                _ => -probability(r, ctx),
            },
            Role::Count => match r.below(3) {
                0 => -(count(r, ctx) + 1),
                1 => count(r, ctx) + ctx.rational(1, 2),
                _ => irrational(r, ctx),
            },
            Role::Int => valid + ctx.rational(1, r.range(2, 7)),
            Role::PosInt => match r.below(3) {
                0 => ctx.zero(),
                1 => -ctx.int(r.range(1, 30)),
                _ => ctx.rational(3, 2),
            },
        },
    }
}

/// A family with parameters; one draw in eight invalid when `allow_invalid`.
fn draw(r: &mut Src, ctx: &Context, allow_invalid: bool) -> Drawn {
    let fam = r.below(FAMILIES) as usize;
    let mut dr = Drawn::new(fam);
    match fam {
        FINITE => {
            draw_table(r, ctx, allow_invalid, &mut dr);
            return dr;
        }
        UNIFORM => {
            let lo = real(r, ctx);
            let w = positive(r, ctx);
            dr.params = vec![lo.clone(), lo + w];
        }
        TRIANGULAR => {
            let lo = real(r, ctx);
            let w = positive(r, ctx);
            let u = match r.below(6) {
                0 => ctx.zero(),
                1 => ctx.one(),
                _ => {
                    let q = r.range(2, 32);
                    ctx.rational(r.range(1, q - 1), q)
                }
            };
            dr.params = vec![lo.clone(), &lo + &w, &lo + &(&w * &u)];
        }
        HYPERGEOMETRIC => {
            let total: i64 = match r.below(8) {
                0..=5 => r.range(0, 60),
                6 => r.range(61, 5000),
                _ => r.range(1, 9) * 10_i64.pow(r.range(4, 6) as u32),
            };
            let part = |r: &mut Src| (total * r.range(0, 64)) / 64;
            let (m, n) = (part(r), part(r));
            dr.params = vec![ctx.int(total), ctx.int(m), ctx.int(n)];
        }
        DISCRETE_UNIFORM => {
            let a = integer(r, ctx);
            let w = if r.below(8) == 0 {
                ctx.int(r.range(1, 9)) * ctx.int(10).powi(r.range(3, 20))
            } else {
                ctx.int(r.range(0, 40))
            };
            dr.params = vec![a.clone(), a + w];
        }
        _ => {
            dr.params = roles(fam)
                .iter()
                .map(|&role| draw_param(role, r, ctx))
                .collect();
        }
    }
    if allow_invalid && r.below(8) == 0 {
        invalidate(&mut dr, r, ctx);
    }
    dr
}

/// Break one parameter, or the relation between them.
fn invalidate(dr: &mut Drawn, r: &mut Src, ctx: &Context) {
    let relation = matches!(
        dr.fam,
        UNIFORM | TRIANGULAR | HYPERGEOMETRIC | DISCRETE_UNIFORM
    ) && r.chance(0.5);
    let p = &mut dr.params;
    if relation {
        match dr.fam {
            UNIFORM => {
                p[1] = if r.chance(0.3) {
                    p[0].clone()
                } else {
                    &p[0] - &positive(r, ctx)
                };
                dr.why = "b ≤ a".into();
            }
            TRIANGULAR => {
                match r.below(3) {
                    0 => p[2] = &p[0] - &positive(r, ctx),
                    1 => p[2] = &p[1] + &positive(r, ctx),
                    _ => {
                        p[1] = p[0].clone();
                        p[2] = p[0].clone();
                    }
                }
                dr.why = "mode outside [a, b] or b = a".into();
            }
            HYPERGEOMETRIC => {
                let i = 1 + r.below(2) as usize;
                p[i] = &p[0] + ctx.int(r.range(1, 5));
                dr.why = "successes or draws exceed the population".into();
            }
            _ => {
                p[1] = &p[0] - ctx.int(r.range(1, 5));
                dr.why = "b < a".into();
            }
        }
    } else {
        let rs = roles(dr.fam);
        let i = r.below(rs.len() as u64) as usize;
        let v = invalid_value(rs[i], r, ctx, &p[i]);
        dr.why = format!("parameter {} = {v} ({:?})", i + 1, rs[i]);
        p[i] = v;
    }
    dr.valid = false;
}

/// A finite table of 1–6 distinct values with masses summing to 1.
fn draw_table(r: &mut Src, ctx: &Context, allow_invalid: bool, dr: &mut Drawn) {
    let m = 1 + r.below(6) as usize;
    let mut v = ctx.rational(r.range(-20, 20), r.range(1, 4));
    let mut values = Vec::with_capacity(m);
    for _ in 0..m {
        values.push(v.clone());
        v = &v + &ctx.rational(r.range(1, 8), r.range(1, 4));
    }
    if r.below(8) == 0 {
        let i = r.below(m as u64) as usize;
        values[i] = &values[i] + &(ctx.int(2).sqrt() / 1000);
    }
    if r.chance(0.25) {
        values.reverse();
    }
    let probs: Vec<Ex> = if m == 2 && r.below(4) == 0 {
        let p = ctx.int(2).sqrt() / 2;
        vec![p.clone(), ctx.one() - p]
    } else {
        let mut w: Vec<i64> = (0..m)
            .map(|_| if r.below(16) == 0 { 0 } else { r.range(1, 20) })
            .collect();
        if w.iter().all(|&x| x == 0) {
            w[0] = 1;
        }
        let total: i64 = w.iter().sum();
        w.iter().map(|&x| ctx.rational(x, total)).collect()
    };
    dr.table = values.into_iter().zip(probs).collect();
    if allow_invalid && r.below(8) == 0 {
        dr.valid = false;
        match r.below(4) {
            0 if m >= 2 => {
                dr.table[1].0 = dr.table[0].0.clone();
                dr.why = "a value listed twice".into();
            }
            1 if m >= 2 && dr.table[0].1 != ctx.zero() => {
                let p0 = dr.table[0].1.clone();
                dr.table[0].1 = -&p0;
                dr.table[1].1 = &dr.table[1].1 + &(p0 * 2);
                dr.why = "a negative mass".into();
            }
            3 => {
                dr.table.clear();
                dr.why = "an empty table".into();
            }
            _ => {
                dr.table[0].1 = &dr.table[0].1 + &ctx.rational(1, 7);
                dr.why = "masses not summing to 1".into();
            }
        }
    }
}

// ── Values ──────────────────────────────────────────────────────────────────

/// The value as the `f64` nearest its certified digits; `None` when it does
/// not evaluate to a finite real.
fn val(e: &Ex) -> Option<f64> {
    e.eval_f64().ok().filter(|v| v.is_finite())
}

/// `val`, but an `Err` when the value is certainly not real.
fn real_value(e: &Ex) -> Result<Option<f64>, String> {
    match e.eval_f64() {
        Ok(v) => Ok(v.is_finite().then_some(v)),
        Err(_) => match e.eval_complex64() {
            Ok(z) if z.im != 0.0 && z.im.is_finite() => Err(format!("{e} = {z} is not real")),
            _ => Ok(None),
        },
    }
}

/// `e` to `digits` certified digits, as the nearest `f64`.
fn dec(e: &Ex, digits: u32) -> Option<f64> {
    let s = e.eval_decimal(digits).ok()?;
    s.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// The sign of a constant: exact when it evaluates to a rational, else
/// from certified digits; `None` when undecided (zero to the precision
/// reached, not real, not evaluable).
fn sign(e: &Ex) -> Option<Ordering> {
    let v = e.eval();
    if let Some(q) = v.as_rational() {
        return Some(q.cmp(&Q::zero()));
    }
    let s = v.eval_decimal(20).ok()?;
    if s.contains('i') {
        return None;
    }
    let mantissa = s.split(['e', 'E']).next().unwrap_or("");
    if !mantissa.contains(|c: char| ('1'..='9').contains(&c)) {
        return None;
    }
    Some(if s.starts_with('-') {
        Ordering::Less
    } else {
        Ordering::Greater
    })
}

/// `a < b`, decided.
fn lt(a: &Ex, b: &Ex) -> Option<bool> {
    sign(&(b - a)).map(|o| o == Ordering::Greater)
}

fn exact(ctx: &Context, v: f64) -> Ex {
    ctx.from_f64(v).unwrap_or_else(|_| ctx.nan())
}

fn q_f64(q: &Q) -> f64 {
    q.to_f64().unwrap_or(f64::NAN)
}

fn is_neg_inf(e: &Ex) -> bool {
    *e == e.context().neg_infinity()
}

fn is_pos_inf(e: &Ex) -> bool {
    *e == e.context().infinity()
}

/// `±∞`, `zoo` or NaN.
fn non_finite(e: &Ex, ctx: &Context) -> bool {
    let v = e.eval();
    v == ctx.infinity() || v == ctx.neg_infinity() || v == ctx.complex_infinity() || v == ctx.nan()
}

fn rel_close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs())
}

fn ulp(v: f64) -> f64 {
    (v.abs() * f64::EPSILON).max(f64::from_bits(1))
}

/// Two constants certainly different: exactly when both are rational, else
/// by their certified difference (relative `1e-25`).
fn differ(a: &Ex, b: &Ex) -> Option<String> {
    let (ea, eb) = (a.eval(), b.eval());
    if let (Some(x), Some(y)) = (ea.as_rational(), eb.as_rational()) {
        return (x != y).then(|| format!("{x} ≠ {y}"));
    }
    match (val(&ea), val(&eb)) {
        (Some(x), None) if non_finite(&eb, &eb.context()) => return Some(format!("{x:e} vs {eb}")),
        (None, Some(y)) if non_finite(&ea, &ea.context()) => return Some(format!("{ea} vs {y:e}")),
        _ => {}
    }
    let d = dec(&(&ea - &eb), 20)?;
    let scale = dec(&ea, 20)?.abs().max(dec(&eb, 20)?.abs());
    (d.abs() > 1e-25 * scale).then(|| format!("{ea} − {eb} = {d:e}"))
}

/// Compare two certified values; an `Err` when they differ by more than
/// `rel` (both at least `1e-300`) or one is certainly not real.
fn same(what: &str, a: &Ex, b: &Ex, rel: f64) -> Result<bool, String> {
    let va = real_value(a).map_err(|e| format!("{what}: {e}"))?;
    let vb = real_value(b).map_err(|e| format!("{what}: {e}"))?;
    let (Some(x), Some(y)) = (va, vb) else {
        return Ok(false);
    };
    show!("  {what}: {x:e} vs {y:e}");
    if x.abs().max(y.abs()) < 1e-300 {
        return Ok(false);
    }
    if !rel_close(x, y, rel) {
        fail!("{what}: {x:e} vs {y:e}\n    {a}\n    {b}");
    }
    Ok(true)
}

/// The density is continuous at `x0` as far as `f64` sampling can tell: its
/// values `2⁻⁴⁰` (relative) to either side are within `10⁻⁶` of its value.
/// Only a guard that withholds a derivative claim at a jump (a mixture with
/// a uniform component, at that component's end); it never makes one.
fn density_continuous_at(d: &Distribution, x0: &Ex) -> bool {
    let ctx = x0.context();
    let Some(f0) = val(&d.density(x0)) else {
        return false;
    };
    let step = val(x0).map_or(1.0, |v| v.abs().max(1e-300)) * 2f64.powi(-40);
    let h = exact(&ctx, step);
    [x0 - &h, x0 + &h]
        .iter()
        .all(|x| val(&d.density(x)).is_some_and(|f| (f - f0).abs() <= 1e-6 * f.abs().max(f0.abs())))
}

/// [`same`] for a density against another route: a disagreement at a point
/// where the density jumps is not a finding.
fn same_density(
    what: &str,
    d: &Distribution,
    x0: &Ex,
    a: &Ex,
    b: &Ex,
    rel: f64,
) -> Result<bool, String> {
    match same(what, a, b, rel) {
        Err(msg) if !msg.contains("is not real") && !density_continuous_at(d, x0) => {
            show!("  (the density jumps at {x0}: not compared)");
            Ok(false)
        }
        other => other,
    }
}

struct Budget(Instant);

impl Budget {
    fn left(&self) -> bool {
        self.0.elapsed().as_secs_f64() < 5.0
    }
}

#[derive(Default)]
struct Kahan {
    sum: f64,
    carry: f64,
}

impl Kahan {
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        self.carry += if self.sum.abs() >= x.abs() {
            (self.sum - t) + x
        } else {
            (x - t) + self.sum
        };
        self.sum = t;
    }
    fn value(&self) -> f64 {
        self.sum + self.carry
    }
}

// ── Theory: which moments and generating functions exist ────────────────────

/// What the fuzzer knows independently about a distribution.
#[derive(Clone)]
enum Theory {
    /// A built-in family with its parameters.
    Base(usize, Vec<Ex>),
    /// `aX + b` of a built-in family: the same moments exist.
    Affine(usize, Vec<Ex>),
    /// A mixture of two built-in families.
    Mixture(usize, Vec<Ex>, usize, Vec<Ex>),
    Unknown,
}

/// Both ends of the support finite.
fn bounded(d: &Distribution) -> bool {
    let s = d.support();
    !s.pieces().is_empty()
        && s.pieces().iter().all(|p| match p {
            Piece::Interval(iv) => !is_neg_inf(&iv.lower) && !is_pos_inf(&iv.upper),
            _ => true,
        })
}

fn family_moment_exists(fam: usize, p: &[Ex], n: u32) -> Option<bool> {
    let ctx = p.first()?.context();
    let n_ex = ctx.int(i64::from(n));
    match fam {
        CAUCHY => Some(false),
        STUDENT_T => lt(&n_ex, &p[0]),
        PARETO => lt(&n_ex, &p[1]),
        F_DIST => lt(&(n_ex * 2), &p[1]),
        _ => Some(true),
    }
}

fn moment_exists(d: &Distribution, th: &Theory, n: u32) -> Option<bool> {
    if bounded(d) {
        return Some(true);
    }
    match th {
        Theory::Base(f, p) | Theory::Affine(f, p) => family_moment_exists(*f, p, n),
        Theory::Mixture(f, p, g, q) => {
            match (
                family_moment_exists(*f, p, n)?,
                family_moment_exists(*g, q, n)?,
            ) {
                (true, true) => Some(true),
                _ => Some(false),
            }
        }
        Theory::Unknown => None,
    }
}

/// `E[e^{tX}] < ∞` for the built-in family.
fn family_mgf_finite(fam: usize, p: &[Ex], t: &Ex) -> Option<bool> {
    let ctx = t.context();
    let one = ctx.one();
    let nonpositive = || sign(t).map(|o| o != Ordering::Greater);
    match fam {
        EXPONENTIAL => lt(t, &p[0]),
        GAMMA => lt(&(t * &p[1]), &one),
        CHI2 => lt(t, &ctx.rational(1, 2)),
        CAUCHY | STUDENT_T => sign(t).map(|o| o == Ordering::Equal),
        LAPLACE | LOGISTIC => lt(&(t.abs() * &p[1]), &one),
        LOGNORMAL | PARETO | F_DIST => nonpositive(),
        WEIBULL => match sign(&(&p[1] - &one))? {
            Ordering::Greater => Some(true),
            Ordering::Equal => lt(&(t * &p[0]), &one),
            Ordering::Less => nonpositive(),
        },
        GEOMETRIC | NEG_BINOMIAL => {
            let q = if fam == GEOMETRIC { &p[0] } else { &p[1] };
            if sign(&(&one - q)) == Some(Ordering::Equal) {
                return Some(true);
            }
            lt(t, &-(&one - q).ln())
        }
        _ => Some(true),
    }
}

fn mgf_finite(d: &Distribution, th: &Theory, t: &Ex) -> Option<bool> {
    if bounded(d) {
        return Some(true);
    }
    match th {
        Theory::Base(f, p) => family_mgf_finite(*f, p, t),
        _ => None,
    }
}

/// The mgf is finite on a neighbourhood of 0.
fn mgf_around_zero(d: &Distribution, th: &Theory) -> bool {
    if bounded(d) {
        return true;
    }
    match th {
        Theory::Base(f, p) => match *f {
            CAUCHY | STUDENT_T | LOGNORMAL | PARETO | F_DIST => false,
            WEIBULL => sign(&(&p[1] - &p[1].context().one())) != Some(Ordering::Less),
            _ => true,
        },
        _ => false,
    }
}

/// A bound on `pmf(j + 1)/pmf(j)` for every `j > k` (a built-in family on
/// an unbounded or huge lattice).
fn pmf_ratio_bound(th: &Theory, k: f64) -> Option<f64> {
    let Theory::Base(fam, p) = th else {
        return None;
    };
    let v = |i: usize| p.get(i).and_then(val);
    let ratio = match *fam {
        POISSON => v(0)? / (k + 2.0),
        GEOMETRIC => 1.0 - v(0)?,
        NEG_BINOMIAL => {
            let (r, q) = (v(0)?, v(1)?);
            if r >= 1.0 {
                (k + 1.0 + r) / (k + 2.0) * (1.0 - q)
            } else {
                1.0 - q
            }
        }
        BINOMIAL => {
            let (n, q) = (v(0)?, v(1)?);
            if q >= 1.0 {
                return None;
            }
            ((n - k - 1.0) / (k + 2.0)).max(0.0) * q / (1.0 - q)
        }
        _ => return None,
    };
    Some(ratio * (1.0 + 1e-12))
}

// ── Sample points ───────────────────────────────────────────────────────────

const LEVELS: [f64; 10] = [
    1e-300, 1e-100, 1e-30, 1e-10, 1e-4, 0.01, 0.1, 0.25, 0.4, 0.5,
];

fn level_point(d: &Distribution, r: &mut Src, ctx: &Context) -> Option<Ex> {
    let lv = LEVELS[r.below(LEVELS.len() as u64) as usize];
    show!("  step: a point at the level {lv:e}");
    let v = (if r.chance(0.5) {
        d.quantile_f64(lv)
    } else {
        d.isf_f64(lv)
    })
    .ok()?;
    v.is_finite().then(|| exact(ctx, v))
}

/// A point strictly inside the interval.
fn continuous_point(d: &Distribution, iv: &Interval<Ex>, r: &mut Src, ctx: &Context) -> Option<Ex> {
    let (lo, hi) = (&iv.lower, &iv.upper);
    let (lo_inf, hi_inf) = (is_neg_inf(lo), is_pos_inf(hi));
    let pow2 = |r: &mut Src| ctx.int(2).powi(r.range(-40, 40));
    let x = match r.below(5) {
        0 | 1 => level_point(d, r, ctx)?,
        2 | 3 => match (lo_inf, hi_inf) {
            (false, false) => lo + &((hi - lo) * ctx.rational(r.range(1, 63), 64)),
            (false, true) => lo + &pow2(r),
            (true, false) => hi - &pow2(r),
            (true, true) => {
                if r.chance(0.5) {
                    pow2(r)
                } else {
                    -pow2(r)
                }
            }
        },
        _ => {
            let c = ctx.int(2).sqrt();
            match (lo_inf, hi_inf) {
                (false, false) => lo + &((hi - lo) / &c),
                (false, true) => lo + &(&c * &pow2(r)),
                (true, false) => hi - &(ctx.pi() * pow2(r)),
                (true, true) => {
                    let v = &c * &pow2(r);
                    if r.chance(0.5) { v } else { -v }
                }
            }
        }
    };
    let inside = (lo_inf || lt(lo, &x) == Some(true)) && (hi_inf || lt(&x, hi) == Some(true));
    inside.then_some(x)
}

/// An atom of a lattice interval.
fn lattice_point(d: &Distribution, iv: &Interval<Ex>, r: &mut Src, ctx: &Context) -> Option<Ex> {
    let (lo, hi) = (&iv.lower, &iv.upper);
    let k = match r.below(4) {
        0 | 1 => level_point(d, r, ctx)?,
        2 => {
            if is_neg_inf(lo) {
                hi - ctx.int(r.range(0, 30))
            } else {
                lo + ctx.int(r.range(0, 30))
            }
        }
        _ => {
            if is_pos_inf(hi) {
                lo + ctx.int(2).powi(r.range(0, 20))
            } else {
                hi - ctx.int(r.range(0, 30))
            }
        }
    }
    .eval();
    (d.support().contains(&k) == Some(true)).then_some(k)
}

/// Up to `n` points of the support (inside it for a density), sorted and
/// distinct.
fn points(d: &Distribution, r: &mut Src, ctx: &Context, n: usize) -> Vec<Ex> {
    let support = d.support();
    let pieces = support.pieces();
    if pieces.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<(f64, Ex)> = Vec::new();
    for _ in 0..n {
        let piece = &pieces[r.below(pieces.len() as u64) as usize];
        let p = match (piece, d.kind()) {
            (Piece::Point(v), _) => Some(v.clone()),
            (Piece::Interval(iv), Kind::Continuous) => continuous_point(d, iv, r, ctx),
            (Piece::Interval(iv), _) => lattice_point(d, iv, r, ctx),
            _ => None,
        };
        if let Some(p) = p
            && let Some(v) = val(&p)
        {
            out.push((v, p));
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut kept: Vec<Ex> = Vec::new();
    for (_, p) in out {
        if kept.last().is_none_or(|q| lt(q, &p) == Some(true)) {
            kept.push(p);
        }
    }
    kept
}

/// The atom before `k`: the largest listed value or lattice point below it
/// over every piece of the support (a mixture's support mixes both);
/// `None` at the lowest one.
fn previous_atom(d: &Distribution, k: &Ex) -> Option<Ex> {
    let mut below: Vec<(f64, Ex)> = Vec::new();
    for piece in d.support().pieces() {
        let candidate = match piece {
            Piece::Point(v) => Some(v.clone()),
            Piece::Interval(iv) => {
                // The largest integer below k, inside the piece.
                let f = k.floor().eval();
                let c = if sign(&(k - &f)) == Some(Ordering::Equal) {
                    (&f - 1).eval()
                } else {
                    f
                };
                // `lower ≤ c ≤ upper` (a lattice's ends are atoms).
                let inside = (is_neg_inf(&iv.lower) || lt(&c, &iv.lower) != Some(true))
                    && (is_pos_inf(&iv.upper) || lt(&iv.upper, &c) != Some(true));
                inside.then_some(c)
            }
            _ => None,
        };
        if let Some(c) = candidate
            && lt(&c, k) == Some(true)
            && let Some(f) = val(&c)
        {
            below.push((f, c));
        }
    }
    below.sort_by(|a, b| a.0.total_cmp(&b.0));
    below.pop().map(|(_, v)| v)
}

// ── Sums over atoms ─────────────────────────────────────────────────────────

/// What multiplies the mass in a sum over atoms, for the tail bound.
#[derive(Clone, Copy)]
enum Weight {
    /// `(k − c)ⁿ`.
    Poly(f64, u32),
    /// `e^{tk}`.
    Exp(f64),
    /// `|h| ≤ 1`.
    Bounded,
}

impl Weight {
    fn abs_at(self, k: f64) -> f64 {
        match self {
            Weight::Poly(c, n) => (k - c).abs().powi(n as i32),
            Weight::Exp(t) => (t * k).exp(),
            Weight::Bounded => 1.0,
        }
    }
    /// A bound on `|h(j + 1)/h(j)|` for every `j > k`.
    fn ratio_after(self, k: f64) -> Option<f64> {
        match self {
            Weight::Poly(_, 0) => Some(1.0),
            Weight::Poly(c, n) => {
                (k + 1.0 > c).then(|| ((k + 2.0 - c) / (k + 1.0 - c)).powi(n as i32))
            }
            Weight::Exp(t) => Some(t.exp()),
            Weight::Bounded => Some(1.0),
        }
    }
}

struct Sum {
    value: f64,
    err: f64,
    exact: Option<Q>,
}

/// `Σ h(k)·pmf(k)` over the atoms: every atom of a table or of a lattice
/// of fewer than 3000 points (exact when every term is rational); on an
/// unbounded or huge lattice the partial sum up to where a ratio bound on
/// the rest is below `1e-15` of the sum.  `None` when the atoms cannot be
/// summed that way.
fn atoms_sum(
    d: &Distribution,
    ctx: &Context,
    th: &Theory,
    h: &dyn Fn(&Ex) -> Ex,
    w: Weight,
    b: &Budget,
) -> Option<Sum> {
    struct Acc {
        sum: Kahan,
        abs: f64,
        exact: Option<Q>,
    }
    impl Acc {
        fn add(&mut self, t: &Ex) -> Option<()> {
            match (self.exact.as_mut(), t.as_rational()) {
                (Some(e), Some(q)) => *e += q,
                _ => self.exact = None,
            }
            let v = val(t)?;
            self.sum.add(v);
            self.abs += v.abs();
            Some(())
        }
        fn done(self, tail: Option<f64>) -> Sum {
            Sum {
                value: self.sum.value(),
                err: self.abs * 8.0 * f64::EPSILON + tail.unwrap_or(0.0),
                exact: if tail.is_some() { None } else { self.exact },
            }
        }
    }
    let support = d.support();
    let mut acc = Acc {
        sum: Kahan::default(),
        abs: 0.0,
        exact: Some(Q::zero()),
    };
    let term = |k: &Ex| (h(k) * d.density(k)).eval();
    if let Some(values) = support.as_points() {
        if values.len() > 512 {
            return None;
        }
        for v in &values {
            acc.add(&term(v))?;
        }
        return Some(acc.done(None));
    }
    let iv = support.as_interval()?;
    let lo = iv.lower.eval().as_i64()?;
    let hi = if is_pos_inf(&iv.upper) {
        None
    } else {
        iv.upper.eval().as_i64()
    };
    if let Some(hi) = hi
        && hi - lo < 3000
    {
        for k in lo..=hi {
            if !b.left() {
                return None;
            }
            acc.add(&term(&ctx.int(k)))?;
        }
        return Some(acc.done(None));
    }
    // Only a built-in family has a ratio bound, and only a mass that ends
    // within a few hundred atoms of the start can be summed this way.
    pmf_ratio_bound(th, lo as f64)?;
    let mean = val(&d.mean())?;
    let sd = val(&d.variance())?.max(0.0).sqrt();
    if mean + 12.0 * sd + 40.0 > lo as f64 + 500.0 {
        return None;
    }
    for k in lo..lo + 600 {
        if !b.left() {
            return None;
        }
        acc.add(&term(&ctx.int(k)))?;
        let kf = k as f64;
        let (Some(rho), Some(wr)) = (pmf_ratio_bound(th, kf), w.ratio_after(kf)) else {
            continue;
        };
        let rho = rho * wr;
        if rho >= 0.95 {
            continue;
        }
        let next_mass = val(&d.density(&ctx.int(k + 1)))?;
        let tail = w.abs_at(kf + 1.0) * next_mass / (1.0 - rho);
        if tail <= 1e-15 * acc.abs.max(1e-300) {
            return Some(acc.done(Some(tail)));
        }
    }
    None
}

// ── Quadrature ──────────────────────────────────────────────────────────────

/// The interval pieces of a continuous support.
fn intervals(d: &Distribution) -> Option<Vec<(Ex, Ex)>> {
    d.support()
        .pieces()
        .iter()
        .map(|p| match p {
            Piece::Interval(iv) => Some((iv.lower.clone(), iv.upper.clone())),
            _ => None,
        })
        .collect()
}

/// `Σ ∫ g` over the pieces by `integrate_numeric`.
fn quad(g: &Ex, x: &Ex, pieces: &[(Ex, Ex)]) -> Option<f64> {
    let mut total = 0.0;
    for (lo, hi) in pieces {
        total += g.integrate_numeric(x, lo, hi).ok()?;
    }
    Some(total)
}

/// The same, certified to `digits` digits.
fn quad_certified(g: &Ex, x: &Ex, pieces: &[(Ex, Ex)], digits: u32) -> Option<f64> {
    let mut total = 0.0;
    for (lo, hi) in pieces {
        let s = g
            .integrate_numeric_decimal(x, &[lo, hi], digits, QuadRule::default())
            .ok()?;
        total += s.parse::<f64>().ok()?;
    }
    Some(total)
}

/// Quantiles of `d` at a spread of levels, mapped by `map` into the
/// integration variable: where a quadrature must look (a narrow spike of
/// mass has quantiles on it).
fn quantile_splits(d: &Distribution, ctx: &Context, map: &dyn Fn(&Ex) -> Ex) -> Vec<Ex> {
    let mut out = Vec::new();
    for lv in [1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0 - 1e-6] {
        if let Ok(v) = d.quantile_f64(lv)
            && v.is_finite()
        {
            out.push(map(&exact(ctx, v)).eval());
        }
    }
    out
}

/// The pieces split at those of `splits` that lie strictly inside them.
fn split_pieces(pieces: &[(Ex, Ex)], splits: &[Ex]) -> Vec<(Ex, Ex)> {
    let mut out = Vec::new();
    for (lo, hi) in pieces {
        let mut inside: Vec<(f64, Ex)> = splits
            .iter()
            .filter(|s| {
                (is_neg_inf(lo) || lt(lo, s) == Some(true))
                    && (is_pos_inf(hi) || lt(s, hi) == Some(true))
            })
            .filter_map(|s| val(s).map(|v| (v, s.clone())))
            .collect();
        inside.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut start = lo.clone();
        for (_, s) in inside {
            if lt(&start, &s) == Some(true) || is_neg_inf(&start) {
                out.push((start, s.clone()));
                start = s;
            }
        }
        out.push((start, hi.clone()));
    }
    out
}

/// `∫ g` over the pieces agrees with `expected` (absolute `tol`): the `f64`
/// quadrature, re-decided by the certified one when it disagrees, and that
/// by the certified one over the pieces split at `splits` (the quadratures
/// can step over a spike of mass far narrower than the range: a Cauchy
/// component of scale `10⁻¹⁴⁸` integrates to 0 over `ℝ`, certified).
fn integral_agrees(
    what: &str,
    g: &Ex,
    x: &Ex,
    pieces: &[(Ex, Ex)],
    splits: &dyn Fn() -> Vec<Ex>,
    expected: f64,
    tol: f64,
) -> Check {
    show!("  step: {what}: integrate_numeric");
    let Some(v) = quad(g, x, pieces) else {
        return Ok(format!("skip: {what}: quadrature refused"));
    };
    if (v - expected).abs() <= tol {
        return Ok(format!("{what} ok"));
    }
    show!("  step: {what}: certified quadrature ({v} vs {expected})");
    let c = quad_certified(g, x, pieces, 12);
    if let Some(c) = c
        && (c - expected).abs() <= tol
    {
        return Ok(format!("{what} ok (certified {c}; integrate_numeric {v})"));
    }
    let refined = split_pieces(pieces, &splits());
    if refined.len() == pieces.len() {
        return match c {
            Some(c) => fail!(
                "{what}: the integral is {c} (certified; integrate_numeric {v}), expected {expected}\n  integrand = {g}"
            ),
            None => Ok(format!(
                "skip: {what}: integrate_numeric {v} vs {expected}, certified quadrature refused"
            )),
        };
    }
    show!(
        "  step: {what}: certified quadrature over {} pieces",
        refined.len()
    );
    match quad_certified(g, x, &refined, 12) {
        Some(s) if (s - expected).abs() <= tol => Ok(format!(
            "{what} ok over {} pieces (certified {s}; unsplit {c:?}, integrate_numeric {v})",
            refined.len()
        )),
        Some(s) => fail!(
            "{what}: the integral is {s} (certified, over {} pieces; unsplit {c:?}; integrate_numeric {v}), expected {expected}\n  integrand = {g}",
            refined.len()
        ),
        None => Ok(format!(
            "skip: {what}: integrate_numeric {v} vs {expected}, certified {c:?}, split quadrature refused"
        )),
    }
}

// ── Checks ──────────────────────────────────────────────────────────────────

fn check_validate(dr: &Drawn, ctx: &Context) -> Check {
    match (dr.valid, dr.build(ctx, true)) {
        (true, Ok(_)) => Ok("valid".into()),
        (true, Err(e)) => fail!("validation: valid parameters rejected: {e}"),
        (false, Ok(d)) => fail!("validation: invalid parameters ({}) accepted: {d}", dr.why),
        (false, Err(_)) => {
            // The unchecked distribution answers or refuses; it never panics.
            if let Ok(u) = dr.build(ctx, false) {
                let _ = u.mean();
                let _ = u.quantile_f64(0.3);
            }
            Ok(format!("rejected ({})", dr.why))
        }
    }
}

fn check_density(d: &Distribution, pts: &[Ex], ctx: &Context, th: &Theory, b: &Budget) -> Check {
    for x0 in pts {
        if !b.left() {
            return Ok("budget".into());
        }
        let f = d.density(x0);
        match real_value(&f).map_err(|e| format!("density: at {x0}: {e}"))? {
            Some(v) if v < 0.0 => fail!("density: f({x0}) = {v:e} < 0\n  f = {f}"),
            _ => {}
        }
    }
    if !b.left() {
        return Ok("budget".into());
    }
    if d.kind() == Kind::Continuous {
        let Some(pieces) = intervals(d) else {
            return Ok("skip: support has points".into());
        };
        let x = ctx.symbol("_fz_x");
        let f = d.density(&x);
        let splits = || quantile_splits(d, ctx, &|v: &Ex| v.clone());
        return integral_agrees("density integral", &f, &x, &pieces, &splits, 1.0, 1e-7);
    }
    let one = |_: &Ex| ctx.one();
    let Some(s) = atoms_sum(d, ctx, th, &one, Weight::Poly(0.0, 0), b) else {
        return Ok("skip: atoms not summable".into());
    };
    if let Some(q) = s.exact {
        if !q.is_one() {
            fail!("density: the masses sum to {q} (exactly)");
        }
        return Ok("sum = 1 exactly".into());
    }
    if (s.value - 1.0).abs() > s.err + 1e-13 {
        fail!(
            "density: the masses sum to {} (bound on the error {:e})",
            s.value,
            s.err
        );
    }
    Ok("sum ok".into())
}

/// `e` is certainly not `target` (0 or 1).
fn not_exactly(e: &Ex, target: i64) -> Option<String> {
    let ctx = e.context();
    differ(e, &ctx.int(target))
}

fn check_cdf(d: &Distribution, pts: &[Ex], ctx: &Context, b: &Budget) -> Check {
    let mut prev: Option<(Ex, f64, f64)> = None;
    for x0 in pts {
        if !b.left() {
            return Ok("budget".into());
        }
        show!("  step: cdf and sf at {x0}");
        let (ce, se) = (d.cdf(x0), d.sf(x0));
        let c = real_value(&ce).map_err(|e| format!("cdf: {e}"))?;
        let s = real_value(&se).map_err(|e| format!("sf: {e}"))?;
        for (name, v, e) in [("cdf", c, &ce), ("sf", s, &se)] {
            if let Some(v) = v
                && !(0.0..=1.0).contains(&v)
            {
                fail!("cdf: {name}({x0}) = {v:e} is not a probability\n  {e}");
            }
        }
        let (Some(c), Some(s)) = (c, s) else { continue };
        if (c + s - 1.0).abs() > 4.0 * f64::EPSILON {
            fail!(
                "cdf: cdf + sf = {} at {x0}: cdf = {c:e}, sf = {s:e}\n  cdf = {ce}\n  sf = {se}",
                c + s
            );
        }
        if let Some((px, pc, ps)) = &prev {
            if c < pc - 4.0 * ulp(*pc) {
                fail!("cdf: decreases: cdf({px}) = {pc:e} > cdf({x0}) = {c:e}");
            }
            if s > ps + 4.0 * ulp(*ps) {
                fail!("cdf: sf increases: sf({px}) = {ps:e} < sf({x0}) = {s:e}");
            }
        }
        prev = Some((x0.clone(), c, s));
    }
    // At and beyond the ends of a single interval or a table.
    let support = d.support();
    let ends: Option<(Option<Ex>, Option<Ex>)> = if let Some(values) = support.as_points() {
        let mut vs: Vec<(f64, Ex)> = values
            .into_iter()
            .filter_map(|v| val(&v).map(|f| (f, v)))
            .collect();
        vs.sort_by(|a, b| a.0.total_cmp(&b.0));
        vs.first()
            .zip(vs.last())
            .map(|(a, z)| (Some(a.1.clone()), Some(z.1.clone())))
    } else {
        support.as_interval().map(|iv| {
            (
                (!is_neg_inf(&iv.lower)).then(|| iv.lower.clone()),
                (!is_pos_inf(&iv.upper)).then(|| iv.upper.clone()),
            )
        })
    };
    let continuous = d.kind() == Kind::Continuous;
    if let Some((lo, hi)) = ends {
        let delta = [ctx.one(), ctx.rational(1, 1024), ctx.int(10).powi(40)];
        if let Some(lo) = &lo {
            for dl in &delta {
                let below = lo - dl;
                if let Some(m) = not_exactly(&d.cdf(&below), 0) {
                    fail!("cdf: below the support, cdf({below}) is not 0: {m}");
                }
                if let Some(m) = not_exactly(&d.sf(&below), 1) {
                    fail!("cdf: below the support, sf({below}) is not 1: {m}");
                }
            }
            if continuous && let Some(m) = not_exactly(&d.cdf(lo), 0) {
                fail!("cdf: at the lower end, cdf({lo}) is not 0: {m}");
            }
        }
        if let Some(hi) = &hi {
            for dl in &delta {
                let above = hi + dl;
                if let Some(m) = not_exactly(&d.cdf(&above), 1) {
                    fail!("cdf: above the support, cdf({above}) is not 1: {m}");
                }
                if let Some(m) = not_exactly(&d.sf(&above), 0) {
                    fail!("cdf: above the support, sf({above}) is not 0: {m}");
                }
            }
            if let Some(m) = not_exactly(&d.cdf(hi), 1) {
                fail!("cdf: at the upper end, cdf({hi}) is not 1: {m}");
            }
            if let Some(m) = not_exactly(&d.sf(hi), 0) {
                fail!("cdf: at the upper end, sf({hi}) is not 0: {m}");
            }
        }
    }
    // Limits of the closed forms at the ends of a single interval.
    if continuous
        && let Some(iv) = support.as_interval()
        && b.left()
    {
        let x = ctx.symbol("_fz_x");
        let fam = d.family();
        let forms = [(fam.cdf(&x), 0, 1, "cdf"), (fam.sf(&x), 1, 0, "sf")];
        for (form, at_lo, at_hi, name) in forms {
            let Some(g) = form else { continue };
            for (end, target) in [(&iv.lower, at_lo), (&iv.upper, at_hi)] {
                if !b.left() {
                    return Ok("budget".into());
                }
                show!("  step: limit of {name} at {end}");
                let lim = if is_neg_inf(end) || is_pos_inf(end) {
                    g.limit(&x, end)
                } else if target == at_lo {
                    g.limit_right(&x, end)
                } else {
                    g.limit_left(&x, end)
                };
                if lim.has_unevaluated() || !lim.free_symbols().is_empty() || non_finite(&lim, ctx)
                {
                    continue;
                }
                if let Some(m) = not_exactly(&lim, target) {
                    fail!(
                        "cdf: the limit of {name} at {end} is {lim}, not {target}: {m}\n  {name} = {g}"
                    );
                }
            }
        }
    }
    Ok("ok".into())
}

fn check_derivative(d: &Distribution, pts: &[Ex], ctx: &Context, b: &Budget) -> Check {
    let fam = d.family();
    let mut compared = 0;
    if d.kind() == Kind::Continuous {
        let x = ctx.symbol("_fz_x");
        let forms = [
            ("cdf", fam.cdf(&x), 1_i64),
            ("cdf_lower", fam.cdf_lower(&x), 1_i64),
            ("sf", fam.sf(&x), -1_i64),
        ];
        for (name, form, sgn) in &forms {
            let Some(g) = form else { continue };
            show!("  step: d/dx {name} = d/dx {g}");
            let dg = g.diff(&x) * *sgn;
            for x0 in pts {
                if !b.left() {
                    return Ok(format!("budget ({compared} compared)"));
                }
                let what = format!("derivative: ±d/dx {name} vs the density at {x0}");
                if same_density(&what, d, x0, &dg.subs(&x, x0), &d.density(x0), 1e-11)? {
                    compared += 1;
                }
            }
        }
        // The closed forms agree with each other.
        for x0 in pts {
            if !b.left() {
                break;
            }
            let (c, cl, s) = (fam.cdf(x0), fam.cdf_lower(x0), fam.sf(x0));
            if let (Some(c), Some(cl)) = (&c, &cl) {
                same(
                    &format!("derivative: cdf vs cdf_lower at {x0}"),
                    c,
                    cl,
                    1e-12,
                )?;
            }
            if let (Some(c), Some(s)) = (&c, &s)
                && let (Some(cv), Some(sv)) = (val(c), val(s))
                && (cv + sv - 1.0).abs() > 4.0 * f64::EPSILON
            {
                fail!(
                    "derivative: the closed cdf + sf = {} at {x0}\n  cdf = {c}\n  sf = {s}",
                    cv + sv
                );
            }
        }
        return Ok(format!("{compared} compared"));
    }
    for k in pts {
        if !b.left() {
            break;
        }
        let mass = d.density(k);
        let (lower, upper) = match previous_atom(d, k) {
            Some(prev) => (d.cdf(k) - d.cdf(&prev), d.sf(&prev) - d.sf(k)),
            None => (d.cdf(k), ctx.one() - d.sf(k)),
        };
        for (name, diff) in [("cdf(k) − cdf(k⁻)", lower), ("sf(k⁻) − sf(k)", upper)] {
            let (em, ed) = (mass.eval(), diff.eval());
            if let (Some(qm), Some(qd)) = (em.as_rational(), ed.as_rational()) {
                if qm != qd {
                    fail!("derivative: at {k}, {name} = {qd} but the mass is {qm} (exactly)");
                }
                compared += 1;
                continue;
            }
            if let (Some(vm), Some(vd)) = (dec(&em, 20), dec(&ed, 20))
                && vm.abs().max(vd.abs()) >= 1e-300
            {
                if !rel_close(vm, vd, 1e-12) {
                    fail!(
                        "derivative: at {k}, {name} = {vd:e} but the mass is {vm:e}\n  mass = {em}\n  difference = {ed}"
                    );
                }
                compared += 1;
            }
        }
    }
    Ok(format!("{compared} compared"))
}

/// A probability level: a double, exact as a rational.
fn draw_level(r: &mut Src) -> f64 {
    match r.below(4) {
        0 | 1 => (1 + r.below(65535)) as f64 / 65536.0,
        2 => 2f64.powi(-(r.range(1, 1074) as i32)),
        _ => 1.0 - 2f64.powi(-(r.range(1, 53) as i32)),
    }
}

/// `quantile_f64`/`isf_f64` of a density met the level: the tail at `xf`
/// matches it to `1e-9`, or the tails 16 doubles either side bracket it.
fn continuous_level_met(d: &Distribution, ctx: &Context, xf: f64, upper: bool, lvl: f64) -> Check {
    if !xf.is_finite() {
        return Ok("skip: infinite quantile".into());
    }
    // Compare on the smaller tail: S against 1 − p for a lower level > ½.
    let (use_sf, target) = match (upper, lvl > 0.5) {
        (false, false) => (false, lvl),
        (false, true) => (true, 1.0 - lvl),
        (true, false) => (true, lvl),
        (true, true) => (false, 1.0 - lvl),
    };
    let tail = |v: f64| -> Option<f64> {
        let e = exact(ctx, v);
        val(&if use_sf { d.sf(&e) } else { d.cdf(&e) })
    };
    let Some(at) = tail(xf) else {
        return Ok("skip: tail not evaluable".into());
    };
    if (at - target).abs() <= 1e-9 * target {
        return Ok("level met".into());
    }
    let (mut lo, mut hi) = (xf, xf);
    for _ in 0..16 {
        lo = lo.next_down();
        hi = hi.next_up();
    }
    let (Some(a), Some(c)) = (tail(lo), tail(hi)) else {
        return Ok("skip: neighbours not evaluable".into());
    };
    let (mn, mx) = (a.min(c), a.max(c));
    if mn <= target * (1.0 + 1e-9) && mx >= target * (1.0 - 1e-9) {
        return Ok("level bracketed".into());
    }
    // A quantile below the normal doubles (the level is already met at the
    // smallest of them: a law spread over `e^(±10¹¹⁹)`) is the bottom of
    // the doubles; which subnormal is a convention.
    let met = if use_sf { at <= target } else { at >= target };
    if xf.abs() <= 2.0 * f64::MIN_POSITIVE && met {
        return Ok("level met below the normal doubles".into());
    }
    let name = if upper { "isf_f64" } else { "quantile_f64" };
    let tname = if use_sf { "sf" } else { "cdf" };
    fail!(
        "quantile: {name}({lvl:e}) = {xf:e} but {tname} there is {at:e} (target {target:e}; ±16 ulps: {a:e}, {c:e})"
    )
}

/// The discrete level conditions at `k` (an atom): `F(k) ≥ p > F(k⁻)` for a
/// lower level, `S(k) ≤ q < S(k⁻)` for an upper one, decided exactly.
fn discrete_level_met(d: &Distribution, ctx: &Context, k: &Ex, upper: bool, lvl: f64) -> Check {
    let lvl_ex = exact(ctx, lvl);
    let name = if upper { "isf_f64" } else { "quantile_f64" };
    // Lower levels above ½ are decided on the upper tail, as the library does
    // (exactly equivalent).
    let (tail_k, target, want_le) = if upper {
        (d.sf(k), lvl_ex.clone(), true)
    } else if lvl > 0.5 {
        (d.sf(k), ctx.one() - &lvl_ex, true)
    } else {
        (d.cdf(k), lvl_ex.clone(), false)
    };
    match (sign(&(&tail_k - &target)), want_le) {
        (Some(Ordering::Greater), true) => {
            fail!("quantile: {name}({lvl:e}) = {k}: the tail there {tail_k} exceeds {target}")
        }
        (Some(Ordering::Less), false) => {
            fail!("quantile: {name}({lvl:e}) = {k}: cdf there {tail_k} is below {target}")
        }
        _ => {}
    }
    let Some(prev) = previous_atom(d, k) else {
        return Ok("level met (lowest atom)".into());
    };
    let tail_prev = if want_le { d.sf(&prev) } else { d.cdf(&prev) };
    match (sign(&(&tail_prev - &target)), want_le) {
        (Some(Ordering::Less | Ordering::Equal), true) => fail!(
            "quantile: {name}({lvl:e}) = {k}, but the tail at {prev} is already {tail_prev} ≤ {target}"
        ),
        (Some(Ordering::Greater | Ordering::Equal), false) => fail!(
            "quantile: {name}({lvl:e}) = {k}, but cdf at {prev} is already {tail_prev} ≥ {target}"
        ),
        _ => Ok("level met".into()),
    }
}

/// `xf` (from `quantile_f64`/`isf_f64`) as an atom of the support: the one
/// listed value whose double it is, else the lattice point it is exactly.
fn as_atom(d: &Distribution, ctx: &Context, xf: f64) -> Option<Ex> {
    let support = d.support();
    let mut hits = support.pieces().iter().filter_map(|p| match p {
        Piece::Point(v) if val(v) == Some(xf) => Some(v.clone()),
        _ => None,
    });
    if let Some(first) = hits.next() {
        return hits.next().is_none().then_some(first);
    }
    if support.as_points().is_some() {
        return None;
    }
    Some(exact(ctx, xf))
}

fn check_quantile(d: &Distribution, pts: &[Ex], ctx: &Context, r: &mut Src, b: &Budget) -> Check {
    let p = draw_level(r);
    let q = draw_level(r);
    let p_ex = exact(ctx, p);
    let mut notes = Vec::new();
    if d.kind() == Kind::Continuous {
        // The closed quantile inverts the CDF.
        if let Some(xq) = d.quantile(&p_ex)
            && p >= 1e-290
            && 1.0 - p >= 1e-290
        {
            let (tail, target) = if p <= 0.5 {
                (d.cdf(&xq), p_ex.clone())
            } else {
                (d.sf(&xq), ctx.one() - &p_ex)
            };
            same(
                &format!("quantile: the tail at quantile({p:e}) = {xq}"),
                &tail,
                &target,
                1e-12,
            )?;
            notes.push("closed");
        }
        for x0 in pts {
            if !b.left() {
                return Ok("budget".into());
            }
            let c = d.cdf(x0);
            if let Some(back) = d.quantile(&c) {
                same(&format!("quantile: quantile(cdf({x0}))"), &back, x0, 1e-12)?;
            }
        }
        for (upper, lvl) in [(false, p), (true, q)] {
            if !b.left() {
                return Ok("budget".into());
            }
            let res = if upper {
                d.isf_f64(lvl)
            } else {
                d.quantile_f64(lvl)
            };
            let Ok(xf) = res else { continue };
            let lower_level = exact(ctx, if upper { 1.0 - lvl } else { lvl });
            let closed = if upper {
                // The library evaluates a closed quantile at the exact 1 − q.
                d.quantile(&(ctx.one() - &exact(ctx, lvl)))
            } else {
                d.quantile(&lower_level)
            };
            if let Some(xq) = closed
                && let Some(v) = val(&xq)
                && xf.is_finite()
                && v.abs().max(xf.abs()) >= 1e-300
                && lvl >= 1e-290
            {
                if !rel_close(v, xf, 1e-12) {
                    let name = if upper { "isf_f64" } else { "quantile_f64" };
                    fail!(
                        "quantile: {name}({lvl:e}) = {xf:e} but the closed quantile is {v:e} = {xq}"
                    );
                }
                continue;
            }
            continuous_level_met(d, ctx, xf, upper, lvl)?;
        }
        return Ok(format!("ok {notes:?}"));
    }
    for (upper, lvl) in [(false, p), (true, q)] {
        if !b.left() {
            return Ok("budget".into());
        }
        let res = if upper {
            d.isf_f64(lvl)
        } else {
            d.quantile_f64(lvl)
        };
        let Ok(kf) = res else { continue };
        // Beyond 2⁵³ consecutive atoms are not distinct doubles: the result
        // is the double nearest the quantile, which need not be an atom.
        if !kf.is_finite() || kf.abs() >= 9_007_199_254_740_992.0 {
            continue;
        }
        let Some(k) = as_atom(d, ctx, kf) else {
            continue;
        };
        if d.support().contains(&k) == Some(false) {
            let name = if upper { "isf_f64" } else { "quantile_f64" };
            fail!(
                "quantile: {name}({lvl:e}) = {k} is not in the support {}",
                d.support()
            );
        }
        discrete_level_met(d, ctx, &k, upper, lvl)?;
        // A closed quantile (DiscreteUniform) gives the same atom.
        if !upper
            && let Some(kq) = d.quantile(&p_ex)
            && let Some(m) = differ(&kq, &k)
        {
            fail!("quantile: quantile({p:e}) = {kq} but quantile_f64 = {k}: {m}");
        }
    }
    Ok("ok".into())
}

fn check_moments(d: &Distribution, ctx: &Context, th: &Theory, n: u32, b: &Budget) -> Check {
    let what = ["mean", "variance", "skewness", "kurtosis"][(n - 1) as usize];
    let value = match n {
        1 => d.mean(),
        2 => d.variance(),
        3 => d.skewness(),
        _ => d.kurtosis(),
    };
    show!("  step: the {what} = {value}");
    let v = real_value(&value).map_err(|e| format!("moments: the {what}: {e}"))?;
    let exists = moment_exists(d, th, n);
    let degenerate = || sign(&d.variance()) == Some(Ordering::Equal);
    match exists {
        Some(false) => {
            if let Some(f) = v {
                fail!("moments: E|X|^{n} = ∞, but the {what} is {f:e} = {value}");
            }
            return Ok(format!("divergent: {value}"));
        }
        Some(true) if v.is_none() && non_finite(&value, ctx) => {
            if n >= 3 && degenerate() {
                return Ok("degenerate".into());
            }
            fail!("moments: E|X|^{n} < ∞, but the {what} is {value}");
        }
        _ => {}
    }
    let Some(target) = v else {
        return Ok(format!("skip: the {what} is {value}"));
    };
    if !b.left() {
        return Ok("budget".into());
    }
    let mean = d.mean();
    let Some(mu) = val(&mean) else {
        return Ok("skip: mean not a number".into());
    };
    let var = d.variance();
    let var_v = val(&var);
    if d.kind() == Kind::Continuous {
        let Some(pieces) = intervals(d) else {
            return Ok("skip: support has points".into());
        };
        let scale = match var_v {
            Some(s2) if s2 > 0.0 => var.sqrt(),
            _ if n == 1 => ctx.one(),
            _ => return Ok("skip: no finite variance".into()),
        };
        let (x, z) = (ctx.symbol("_fz_x"), ctx.symbol("_fz_z"));
        let g = d.density(&x).subs(&x, &(&mean + &(&scale * &z))) * &scale;
        let integrand = z.powi(i64::from(n)) * g;
        let zpieces: Vec<(Ex, Ex)> = pieces
            .iter()
            .map(|(lo, hi)| {
                let map = |e: &Ex| {
                    if is_neg_inf(e) || is_pos_inf(e) {
                        e.clone()
                    } else {
                        ((e - &mean) / &scale).eval()
                    }
                };
                (map(lo), map(hi))
            })
            .collect();
        let expected = match n {
            1 => 0.0,
            2 => 1.0,
            _ => target,
        };
        let tol = 1e-7 * expected.abs().max(1.0);
        let to_z = |v: &Ex| (v - &mean) / &scale;
        let splits = || quantile_splits(d, ctx, &to_z);
        return integral_agrees(
            &format!("moments: standardized moment {n} ({what} {value})"),
            &integrand,
            &z,
            &zpieces,
            &splits,
            expected,
            tol,
        );
    }
    // Discrete: Σ (k − μ)ⁿ pmf(k) against 0, Var, skew·Var^{3/2}, kurt·Var².
    let expected = match n {
        1 => ctx.zero(),
        2 => var.clone(),
        3 => &value * &var.sqrt().powi(3),
        _ => &value * &var.powi(2),
    };
    let h = |k: &Ex| (k - &mean).powi(i64::from(n));
    show!("  step: Σ (k − μ)^{n} pmf(k), μ = {mean}");
    let Some(s) = atoms_sum(d, ctx, th, &h, Weight::Poly(mu, n), b) else {
        return Ok("skip: atoms not summable".into());
    };
    let ee = expected.eval();
    if let (Some(qs), Some(qe)) = (&s.exact, ee.as_rational()) {
        if *qs != qe {
            fail!("moments: Σ (k − μ)^{n} pmf(k) = {qs} but the {what} gives {qe} (exactly)");
        }
        return Ok("exact".into());
    }
    let Some(e) = val(&ee) else {
        return Ok("skip: expected value not a number".into());
    };
    let scale = var_v.unwrap_or(1.0).abs().sqrt().powi(n as i32).max(1e-300);
    if (s.value - e).abs() > s.err + 1e-11 * scale.max(e.abs()) {
        fail!(
            "moments: Σ (k − μ)^{n} pmf(k) = {:e} (± {:e}) but the {what} gives {e:e} = {ee}",
            s.value,
            s.err
        );
    }
    Ok("sum ok".into())
}

fn check_mgf(d: &Distribution, ctx: &Context, th: &Theory, r: &mut Src, b: &Budget) -> Check {
    let spread = val(&d.variance())
        .filter(|v| *v > 0.0)
        .map_or(1.0, f64::sqrt);
    let draw_t = |r: &mut Src| {
        let c = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0][r.below(8) as usize];
        let c = if r.chance(0.5) { -c } else { c };
        exact(ctx, c / spread)
    };
    let x = ctx.symbol("_fz_x");
    let continuous = d.kind() == Kind::Continuous;
    let pieces = intervals(d);
    let splits = || quantile_splits(d, ctx, &|v: &Ex| v.clone());
    // mgf at a real t: +∞ outside the domain, the expectation inside.
    let t0 = draw_t(r);
    show!("  step: mgf({t0})");
    let m = d.mgf(&t0);
    match mgf_finite(d, th, &t0) {
        Some(false) => {
            if let Some(v) = val(&m) {
                fail!("mgf: E[e^(tX)] = ∞ at t = {t0}, but mgf = {v:e} = {m}");
            }
        }
        Some(true) if non_finite(&m, ctx) => {
            fail!("mgf: E[e^(tX)] < ∞ at t = {t0}, but mgf = {m}");
        }
        Some(true) => {
            if let Some(mv) = val(&m)
                && b.left()
            {
                if continuous && let Some(pieces) = &pieces {
                    let g = (&t0 * &x).exp() * d.density(&x);
                    integral_agrees(
                        &format!("mgf: E[e^(tX)] at t = {t0} (mgf {m})"),
                        &g,
                        &x,
                        pieces,
                        &splits,
                        mv,
                        1e-7 * mv.abs().max(1.0),
                    )?;
                } else if !continuous && let Some(tv) = val(&t0) {
                    let h = |k: &Ex| (&t0 * k).exp();
                    if let Some(s) = atoms_sum(d, ctx, th, &h, Weight::Exp(tv), b)
                        && (s.value - mv).abs() > s.err + 1e-12 * mv.abs()
                    {
                        fail!(
                            "mgf: Σ e^(tk) pmf(k) = {:e} at t = {t0}, but mgf = {mv:e} = {m}",
                            s.value
                        );
                    }
                }
            }
        }
        None => {}
    }
    // mgf′(0) = E[X], mgf″(0) = E[X²] for the closed mgf.
    let t = ctx
        .symbol_with("_fz_t", &[Assumption::Real])
        .unwrap_or_else(|_| ctx.symbol("_fz_t"));
    if mgf_around_zero(d, th)
        && b.left()
        && let Some(mg) = d.family().mgf(&t)
    {
        show!("  step: mgf derivatives at 0 of {mg}");
        let at_zero = |e: &Ex| -> Ex {
            let v = e.subs(&t, &ctx.zero()).eval();
            if val(&v).is_some() {
                v
            } else {
                e.limit(&t, &ctx.zero())
            }
        };
        let d1 = mg.diff(&t);
        let m1 = at_zero(&d1);
        let mean = d.mean();
        if let Some(m) = differ(&m1, &mean)
            && val(&m1).is_some()
        {
            fail!("mgf: mgf′(0) = {m1} but the mean is {mean}: {m}\n  mgf = {mg}");
        }
        if b.left() {
            let m2 = at_zero(&d1.diff(&t));
            let second = d.variance() + mean.powi(2);
            if let Some(m) = differ(&m2, &second)
                && val(&m2).is_some()
            {
                fail!("mgf: mgf″(0) = {m2} but E[X²] is {second}: {m}\n  mgf = {mg}");
            }
        }
    }
    // The characteristic function at a real s.
    if !b.left() {
        return Ok("budget".into());
    }
    let s0 = draw_t(r);
    show!("  step: cf({s0})");
    let cf = d.characteristic_function(&s0);
    let Ok(z) = cf.eval_complex64() else {
        return Ok("skip: cf not evaluable".into());
    };
    if z.norm() > 1.0 + 1e-12 {
        fail!("mgf: |cf({s0})| = {} > 1: cf = {cf}", z.norm());
    }
    if mgf_around_zero(d, th)
        && let Some(mi) = d.family().mgf(&(ctx.i_unit() * &s0))
        && let Ok(w) = mi.eval_complex64()
        && (w - z).norm() > 1e-12
    {
        fail!("mgf: cf({s0}) = {z} but mgf(i·{s0}) = {w}\n  cf = {cf}\n  mgf(it) = {mi}");
    }
    if continuous && let Some(pieces) = &pieces {
        let (cs, sn) = (
            (&s0 * &x).cos() * d.density(&x),
            (&s0 * &x).sin() * d.density(&x),
        );
        integral_agrees(
            &format!("mgf: Re cf({s0}) = {z}"),
            &cs,
            &x,
            pieces,
            &splits,
            z.re,
            1e-8,
        )?;
        if b.left() {
            integral_agrees(
                &format!("mgf: Im cf({s0}) = {z}"),
                &sn,
                &x,
                pieces,
                &splits,
                z.im,
                1e-8,
            )?;
        }
    } else if !continuous {
        let hc = |k: &Ex| (&s0 * k).cos();
        let hs = |k: &Ex| (&s0 * k).sin();
        if let Some(sc) = atoms_sum(d, ctx, th, &hc, Weight::Bounded, b)
            && let Some(ss) = atoms_sum(d, ctx, th, &hs, Weight::Bounded, b)
            && ((sc.value - z.re).abs() > sc.err + 1e-12
                || (ss.value - z.im).abs() > ss.err + 1e-12)
        {
            fail!(
                "mgf: Σ e^(isk) pmf(k) = {} + {}i at s = {s0}, but cf = {z} = {cf}",
                sc.value,
                ss.value
            );
        }
    }
    Ok("ok".into())
}

// ── numdist ─────────────────────────────────────────────────────────────────

/// The kernels' accuracy at a tail `p` (module docs of `numdist`); `None`
/// below `1e-290`.
fn kernel_tol(p: f64) -> Option<f64> {
    if p.is_nan() || p < 1e-290 {
        return None;
    }
    Some(if p >= 1e-100 {
        1e-12
    } else {
        3e-15 * (-p.ln())
    })
}

fn numdist_case(r: &mut Src, ctx: &Context, desc: &mut String, b: &Budget) -> Check {
    let lr = |r: &mut Src, lo: f64, hi: f64| {
        let u = r.below(65536) as f64 / 65535.0;
        (lo.ln() + u * (hi.ln() - lo.ln())).exp()
    };
    let shape = |r: &mut Src, lo: f64, hi: f64, ext: f64| {
        if r.below(4) == 0 {
            lr(r, 1e-300, ext)
        } else {
            lr(r, lo, hi)
        }
    };
    let fam = [
        NORMAL, STUDENT_T, CHI2, GAMMA, BETA, F_DIST, BINOMIAL, POISSON,
    ][r.below(8) as usize];
    let pv: Vec<f64> = match fam {
        NORMAL => {
            let mu = if r.chance(0.25) {
                0.0
            } else {
                lr(r, 1e-6, 1e6) * if r.chance(0.5) { -1.0 } else { 1.0 }
            };
            vec![mu, lr(r, 1e-6, 1e6)]
        }
        STUDENT_T | CHI2 => vec![shape(r, 1e-2, 1e9, 1e300)],
        GAMMA => vec![shape(r, 1e-3, 1e9, 1e300), lr(r, 1e-3, 1e3)],
        BETA => vec![shape(r, 1e-3, 1e7, 1e15), shape(r, 1e-3, 1e7, 1e15)],
        F_DIST => vec![shape(r, 1e-2, 1e8, 1e15), shape(r, 1e-2, 1e8, 1e15)],
        BINOMIAL => vec![lr(r, 1.0, 1e17).floor(), lr(r, 1e-9, 1.0).min(1.0 - 1e-9)],
        _ => vec![if r.below(4) == 0 {
            lr(r, 1e-300, 1e-6)
        } else {
            lr(r, 1e-6, 1e17)
        }],
    };
    let mut dr = Drawn::new(fam);
    dr.params = pv.iter().map(|&v| exact(ctx, v)).collect();
    describe(desc, dr.text());
    let d = dr
        .build(ctx, true)
        .map_err(|e| format!("validation: valid parameters rejected: {e}"))?;
    let mut compared = 0;
    for _ in 0..3 {
        if !b.left() {
            break;
        }
        let lvl = draw_level(r);
        let via_level = r.chance(0.5);
        let point = |r: &mut Src, lo: f64, hi: f64, centre: f64| {
            if r.chance(0.5) {
                lr(r, lo, hi)
            } else {
                (centre * lr(r, 1e-3, 1e3)).min(f64::MAX)
            }
        };
        // `x` for the kernel; the symbolic side gets the exact same number
        // (for the normal, `μ + σz` with the kernel's standardized `z`).
        let x: f64 = match fam {
            NORMAL => {
                if via_level {
                    numdist::norm::ppf(lvl).unwrap_or(0.0)
                } else {
                    lr(r, 1e-3, 38.0) * if r.chance(0.5) { -1.0 } else { 1.0 }
                }
            }
            STUDENT_T => {
                if via_level {
                    numdist::t::ppf(lvl, pv[0]).unwrap_or(1.0)
                } else {
                    lr(r, 1e-3, 1e200) * if r.chance(0.5) { -1.0 } else { 1.0 }
                }
            }
            CHI2 => {
                if via_level {
                    numdist::chi2::ppf(lvl, pv[0]).unwrap_or(1.0)
                } else {
                    point(r, 1e-6, 1e10, pv[0])
                }
            }
            GAMMA => {
                if via_level {
                    numdist::gamma::ppf(lvl, pv[0], pv[1]).unwrap_or(1.0)
                } else {
                    point(r, 1e-6, 1e12, pv[0] * pv[1])
                }
            }
            BETA => {
                if via_level {
                    numdist::beta::ppf(lvl, pv[0], pv[1]).unwrap_or(0.5)
                } else {
                    r.below(65536) as f64 / 65536.0
                }
            }
            F_DIST => {
                if via_level {
                    numdist::f::ppf(lvl, pv[0], pv[1]).unwrap_or(1.0)
                } else {
                    point(r, 1e-6, 1e6, 1.0)
                }
            }
            BINOMIAL => {
                if via_level {
                    numdist::binom::ppf(lvl, pv[0], pv[1]).unwrap_or(0.0)
                } else {
                    (pv[0] * r.below(65536) as f64 / 65535.0).floor()
                }
            }
            _ => {
                if via_level {
                    numdist::poisson::ppf(lvl, pv[0]).unwrap_or(0.0)
                } else {
                    (pv[0] * lr(r, 1e-3, 10.0)).floor()
                }
            }
        };
        if !x.is_finite() {
            continue;
        }
        show!("  x = {x:e}");
        let (nc, ns, npdf): (f64, f64, Option<f64>) = match fam {
            NORMAL => (
                numdist::norm::cdf(x),
                numdist::norm::sf(x),
                Some(numdist::norm::pdf(x) / pv[1]),
            ),
            STUDENT_T => (
                numdist::t::cdf(x, pv[0]),
                numdist::t::sf(x, pv[0]),
                Some(numdist::t::pdf(x, pv[0])),
            ),
            CHI2 => (
                numdist::chi2::cdf(x, pv[0]),
                numdist::chi2::sf(x, pv[0]),
                None,
            ),
            GAMMA => (
                numdist::gamma::cdf(x, pv[0], pv[1]),
                numdist::gamma::sf(x, pv[0], pv[1]),
                None,
            ),
            BETA => (
                numdist::beta::cdf(x, pv[0], pv[1]),
                numdist::beta::sf(x, pv[0], pv[1]),
                None,
            ),
            F_DIST => (
                numdist::f::cdf(x, pv[0], pv[1]),
                numdist::f::sf(x, pv[0], pv[1]),
                None,
            ),
            BINOMIAL => (
                numdist::binom::cdf(x, pv[0], pv[1]),
                numdist::binom::sf(x, pv[0], pv[1]),
                None,
            ),
            _ => (
                numdist::poisson::cdf(x, pv[0]),
                numdist::poisson::sf(x, pv[0]),
                None,
            ),
        };
        let x_ex = if fam == NORMAL {
            &dr.params[0] + &(&dr.params[1] * &exact(ctx, x))
        } else {
            exact(ctx, x)
        };
        let mut sides = vec![("cdf", d.cdf(&x_ex), nc), ("sf", d.sf(&x_ex), ns)];
        if let Some(np) = npdf {
            sides.push(("pdf", d.density(&x_ex), np));
        }
        for (name, e, kernel) in sides {
            if !b.left() {
                break;
            }
            let Some(sym) = val(&e) else { continue };
            show!("  {name}: symbolic {sym:e}, numdist {kernel:e}");
            let (Some(tol_s), Some(_)) = (kernel_tol(sym), kernel_tol(kernel)) else {
                continue;
            };
            let tol = if name == "pdf" { 1e-12 } else { tol_s };
            if !rel_close(sym, kernel, tol) {
                fail!(
                    "numdist: {name} at x = {x:e}: symbolic {sym:e}, numdist {kernel:e} (relative {:e}, tolerance {tol:e})\n  {e}",
                    (sym - kernel).abs() / sym.abs()
                );
            }
            compared += 1;
        }
    }
    Ok(format!("{compared} compared"))
}

// ── Order statistics of finite tables ───────────────────────────────────────

fn binomial_q(n: i64, k: i64) -> Q {
    if k < 0 || k > n {
        return Q::zero();
    }
    let mut acc = BigInt::one();
    for i in 0..k {
        acc = acc * BigInt::from(n - i) / BigInt::from(i + 1);
    }
    Q::from_integer(acc)
}

fn rat(e: &Ex) -> Option<Q> {
    e.eval().as_rational()
}

fn order_case(r: &mut Src, ctx: &Context, desc: &mut String) -> Check {
    // The parent and, when its values and masses are rational, its exact table.
    let (parent, table): (Distribution, Option<Vec<(Q, Q)>>) = match r.below(6) {
        0..=2 => {
            let mut dr = Drawn::new(FINITE);
            draw_table(r, ctx, false, &mut dr);
            let t: Option<Vec<(Q, Q)>> = dr
                .table
                .iter()
                .map(|(v, p)| Some((rat(v)?, rat(p)?)))
                .collect();
            let d = dr
                .build(ctx, true)
                .map_err(|e| format!("validation: {e}"))?;
            (d, t)
        }
        3 => {
            let p = draw_param(Role::Prob, r, ctx);
            let t = rat(&p).map(|q| vec![(Q::zero(), Q::one() - &q), (Q::one(), q)]);
            (Distribution::bernoulli(p), t)
        }
        4 => {
            let n = r.range(0, 8);
            let p = draw_param(Role::Prob, r, ctx);
            let t = rat(&p).map(|q| {
                (0..=n)
                    .map(|k| {
                        let mass = binomial_q(n, k) * pow_q(&q, k) * pow_q(&(Q::one() - &q), n - k);
                        (Q::from_integer(BigInt::from(k)), mass)
                    })
                    .collect()
            });
            (Distribution::binomial(ctx.int(n), p), t)
        }
        _ => match r.below(3) {
            0 => {
                let a = r.range(-5, 5);
                let w = r.range(0, 6);
                let t = (a..=a + w)
                    .map(|k| {
                        (
                            Q::from_integer(BigInt::from(k)),
                            Q::new(BigInt::one(), BigInt::from(w + 1)),
                        )
                    })
                    .collect();
                (
                    Distribution::discrete_uniform(ctx.int(a), ctx.int(a + w)),
                    Some(t),
                )
            }
            1 => {
                let s = r.range(1, 6);
                let t = (1..=s)
                    .map(|k| {
                        (
                            Q::from_integer(BigInt::from(k)),
                            Q::new(BigInt::one(), BigInt::from(s)),
                        )
                    })
                    .collect();
                (Distribution::die(ctx.int(s)), Some(t))
            }
            _ => {
                let total = r.range(0, 9);
                let m = r.range(0, total);
                let n = r.range(0, total);
                let all = binomial_q(total, n);
                let t = (0..=n)
                    .filter_map(|k| {
                        let mass = binomial_q(m, k) * binomial_q(total - m, n - k) / &all;
                        (!mass.is_zero()).then(|| (Q::from_integer(BigInt::from(k)), mass))
                    })
                    .collect();
                (
                    Distribution::hypergeometric(ctx.int(total), ctx.int(m), ctx.int(n)),
                    Some(t),
                )
            }
        },
    };
    let n = 1 + r.below(5) as usize;
    let k = 1 + r.below(n as u64) as usize;
    describe(
        desc,
        format!("the {k}-th smallest of {n} draws from {parent}"),
    );
    let os = order_statistic(&parent, n, k)
        .map_err(|e| format!("order: refused for a finite parent: {e}"))?;
    let Some(values) = os.support().as_points() else {
        fail!("order: not a table: {os}");
    };
    let mut total = Some(Q::zero());
    let mut total_f = Kahan::default();
    let mut masses: Vec<(Ex, Ex)> = Vec::new();
    for v in &values {
        let m = os.density(v).eval();
        match m.as_rational() {
            Some(q) => {
                if q < Q::zero() {
                    fail!("order: P(X_({k}) = {v}) = {q} < 0");
                }
                total_f.add(q_f64(&q));
                if let Some(t) = total.as_mut() {
                    *t += q;
                }
            }
            None => {
                total = None;
                if sign(&m) == Some(Ordering::Less) {
                    fail!("order: P(X_({k}) = {v}) = {m} < 0");
                }
                match val(&m) {
                    Some(f) => total_f.add(f),
                    None => return Ok("skip: a mass not evaluable".into()),
                }
            }
        }
        masses.push((v.clone(), m));
    }
    match &total {
        Some(t) if !t.is_one() => fail!("order: the masses sum to {t} (exactly)"),
        None if (total_f.value() - 1.0).abs() > 1e-13 => {
            fail!("order: the masses sum to {}", total_f.value())
        }
        _ => {}
    }
    // Brute force over every sample.
    let Some(mut table) = table else {
        return Ok("sum ok".into());
    };
    table.sort_by(|a, b| a.0.cmp(&b.0));
    let m = table.len();
    let Some(count) = m.checked_pow(n as u32).filter(|&c| c <= 4096 && m > 0) else {
        return Ok("sum ok (no brute force)".into());
    };
    let mut want = vec![Q::zero(); m];
    let mut digits = vec![0usize; n];
    for idx in 0..count {
        let mut rest = idx;
        let mut prob = Q::one();
        for slot in digits.iter_mut() {
            *slot = rest % m;
            rest /= m;
            prob *= &table[*slot].1;
        }
        let mut sorted = digits.clone();
        sorted.sort_unstable();
        want[sorted[k - 1]] += prob;
    }
    for (j, (value, w)) in table.iter().enumerate() {
        let got = masses
            .iter()
            .find(|(v, _)| rat(v).as_ref() == Some(value))
            .map(|(_, p)| p.clone());
        match got {
            Some(p) => {
                if let Some(q) = rat(&p) {
                    if q != want[j] {
                        fail!(
                            "order: P(X_({k}) = {value}) = {q}, but enumerating every sample gives {}",
                            want[j]
                        );
                    }
                } else if let Some(mm) = differ(&p, &ctx.from_ratio(want[j].clone())) {
                    fail!(
                        "order: P(X_({k}) = {value}) = {p}, enumerating gives {}: {mm}",
                        want[j]
                    );
                }
            }
            None if !w.is_zero() && !want[j].is_zero() => {
                fail!(
                    "order: {value} is missing from the table; enumerating gives {}",
                    want[j]
                )
            }
            None => {}
        }
    }
    Ok("brute force agrees".into())
}

fn pow_q(q: &Q, n: i64) -> Q {
    let mut acc = Q::one();
    for _ in 0..n {
        acc *= q;
    }
    acc
}

// ── Sums of independent variables ───────────────────────────────────────────

/// A gamma-like distribution of scale `theta`: `Gamma(k, θ)`, `Exponential(1/θ)`
/// or (for `θ = 2`) `ChiSquared(2k)`.
fn gamma_like(r: &mut Src, ctx: &Context, theta: &Ex) -> Distribution {
    match r.below(3) {
        0 => Distribution::exponential((ctx.one() / theta).simplify()),
        1 if *theta == ctx.int(2) => Distribution::chi_squared(positive(r, ctx)),
        _ => Distribution::gamma(positive(r, ctx), theta.clone()),
    }
}

fn sum_case(r: &mut Src, ctx: &Context, desc: &mut String) -> Check {
    let pos = |r: &mut Src| positive(r, ctx);
    let kind = r.below(8);
    let (da, db, closure): (Distribution, Distribution, bool) = match kind {
        0 => (
            Distribution::normal(real(r, ctx), pos(r)),
            Distribution::normal(real(r, ctx), pos(r)),
            true,
        ),
        1 => {
            let theta = if r.chance(0.25) { ctx.int(2) } else { pos(r) };
            (gamma_like(r, ctx, &theta), gamma_like(r, ctx, &theta), true)
        }
        2 => (
            Distribution::poisson(pos(r)),
            Distribution::poisson(pos(r)),
            true,
        ),
        3 => {
            let p = draw_param(Role::Prob, r, ctx);
            let one = |r: &mut Src, p: &Ex| {
                if r.chance(0.3) {
                    Distribution::bernoulli(p.clone())
                } else {
                    Distribution::binomial(count(r, ctx), p.clone())
                }
            };
            (one(r, &p), one(r, &p), true)
        }
        4 => {
            let p = probability(r, ctx);
            (
                Distribution::negative_binomial(pos(r), p.clone()),
                Distribution::negative_binomial(pos(r), p),
                true,
            )
        }
        5 => (
            Distribution::gamma(pos(r), pos(r)),
            Distribution::gamma(pos(r), pos(r)),
            false,
        ),
        6 => (
            Distribution::binomial(count(r, ctx), probability(r, ctx)),
            Distribution::binomial(count(r, ctx), probability(r, ctx)),
            false,
        ),
        _ => {
            let d = Distribution::normal(real(r, ctx), pos(r));
            (d.clone(), d, false)
        }
    };
    let same_var = kind == 7;
    describe(
        desc,
        if same_var {
            format!("X + X, X ~ {da}")
        } else {
            format!("X + Y, X ~ {da}, Y ~ {db}")
        },
    );
    let x = RandomVariable::try_new(ctx, "X", da.clone()).map_err(|e| e.to_string())?;
    let y = RandomVariable::try_new(ctx, "Y", db.clone()).map_err(|e| e.to_string())?;
    let s = if same_var {
        sum_distribution(&x, &x)
    } else {
        sum_distribution(&x, &y)
    };
    let Some(s) = s else {
        if closure {
            fail!("sums: a documented closure was not recognised");
        }
        return Ok("None".into());
    };
    if same_var {
        fail!("sums: X + X treated as a sum of independent variables: {s}");
    }
    for (what, got, want) in [
        ("mean", s.mean(), da.mean() + db.mean()),
        ("variance", s.variance(), da.variance() + db.variance()),
        (
            "third central moment",
            s.central_moment(3),
            da.central_moment(3) + db.central_moment(3),
        ),
    ] {
        if let Some(m) = differ(&got, &want) {
            fail!("sums: the {what} of {s} is {got}, the sum of the {what}s is {want}: {m}");
        }
    }
    Ok(format!("{s}"))
}

// ── Monotone maps ───────────────────────────────────────────────────────────

struct Map {
    g: Ex,
    increasing: bool,
}

/// A map strictly monotone on the support `[lo, hi]` (`strict_positive`: the
/// map must also be finite at a lower end 0).
fn monotone_map(
    r: &mut Src,
    ctx: &Context,
    x: &Ex,
    lo: &Ex,
    hi: &Ex,
    strict_positive: bool,
) -> Option<Map> {
    let ge0 =
        |e: &Ex| !is_neg_inf(e) && matches!(sign(e), Some(Ordering::Greater | Ordering::Equal));
    let gt0 = |e: &Ex| !is_neg_inf(e) && sign(e) == Some(Ordering::Greater);
    let nonneg = if strict_positive { gt0(lo) } else { ge0(lo) };
    let unit = nonneg && !is_pos_inf(hi) && {
        let rest = ctx.one() - hi;
        if strict_positive {
            gt0(&rest)
        } else {
            ge0(&rest)
        }
    };
    let (g, increasing) = match r.below(20) {
        0 | 1 => {
            let a = small_rational(r, ctx, 9);
            let b = ctx.rational(r.range(-9, 9), r.range(1, 4));
            if r.chance(0.5) {
                (&a * x + b, true)
            } else {
                (-(&a * x) + b, false)
            }
        }
        2 => (x.powi(3), true),
        3 => (x.exp(), true),
        4 => ((-x).exp(), false),
        5 => (x.sinh(), true),
        6 => (x.atan(), true),
        7 => (x.powi(3) + x, true),
        8 => (x.tanh(), true),
        9 => (-x.powi(3), false),
        10 => (x.powi(5), true),
        11 if nonneg => (x.ln(), true),
        12 if nonneg => (x.sqrt(), true),
        13 if nonneg => (x.powi(2), true),
        14 if nonneg => (x.powi(-1), false),
        15 if nonneg => (x.cbrt(), true),
        16 if nonneg => (-x.ln(), false),
        17 if nonneg => (x.powi(2) + x, true),
        18 if unit => ((x / (ctx.one() - x)).ln(), true),
        19 if unit => (-(ctx.one() - x).ln(), true),
        _ => return None,
    };
    Some(Map { g, increasing })
}

/// The hull `[lo, hi]` of a support (a single interval, or a table).
fn hull(d: &Distribution) -> Option<(Ex, Ex)> {
    let support = d.support();
    if let Some(iv) = support.as_interval() {
        return Some((iv.lower.clone(), iv.upper.clone()));
    }
    let mut vs: Vec<(f64, Ex)> = support
        .as_points()?
        .into_iter()
        .filter_map(|v| val(&v).map(|f| (f, v)))
        .collect();
    vs.sort_by(|a, b| a.0.total_cmp(&b.0));
    Some((vs.first()?.1.clone(), vs.last()?.1.clone()))
}

fn transform_case(r: &mut Src, ctx: &Context, desc: &mut String, b: &Budget) -> Check {
    let dr = draw(r, ctx, false);
    describe(desc, dr.text());
    let d = dr
        .build(ctx, true)
        .map_err(|e| format!("validation: valid parameters rejected: {e}"))?;
    let discrete = d.kind() != Kind::Continuous;
    if discrete && !bounded(&d) {
        return Ok("skip: infinite lattice".into());
    }
    let Some((lo, hi)) = hull(&d) else {
        return Ok("skip: support shape".into());
    };
    let x = ctx.symbol("_fz_x");
    let Some(map) = monotone_map(r, ctx, &x, &lo, &hi, discrete) else {
        return Ok("skip: no map for this support".into());
    };
    desc.push_str(&format!(" under {}", map.g));
    show!("  map: {}", map.g);
    let y = match d.transformed(&x, &map.g) {
        Ok(y) => y,
        Err(e) => return Ok(format!("skip: refused: {e}")),
    };
    let pts = points(&d, r, ctx, 4);
    let mut compared = 0;
    for x0 in &pts {
        if !b.left() {
            break;
        }
        let y0 = map.g.subs(&x, x0);
        let (fx, sx) = (d.cdf(x0), d.sf(x0));
        let (fy, sy) = (y.cdf(&y0), y.sf(&y0));
        let (want_f, want_s) = match (map.increasing, discrete) {
            (true, _) => (fx, sx),
            (false, false) => (sx, fx),
            (false, true) => {
                let px = d.density(x0);
                (&sx + &px, &fx - &px)
            }
        };
        if same(&format!("transformed: P(Y ≤ g({x0}))"), &fy, &want_f, 1e-12)? {
            compared += 1;
        }
        same(&format!("transformed: P(Y > g({x0}))"), &sy, &want_s, 1e-12)?;
        let want_density = if discrete {
            d.density(x0)
        } else {
            d.density(x0) / map.g.diff(&x).subs(&x, x0).abs()
        };
        let what = format!("transformed: the density of Y at g({x0}) = {y0}");
        if discrete {
            same(&what, &y.density(&y0), &want_density, 1e-11)?;
        } else {
            same_density(&what, &d, x0, &y.density(&y0), &want_density, 1e-11)?;
        }
    }
    Ok(format!("{compared} compared: {y}"))
}

// ── Wrappers for the generic checks ─────────────────────────────────────────

fn wrap(
    base: &Distribution,
    dr: &Drawn,
    r: &mut Src,
    ctx: &Context,
) -> (Distribution, String, Theory) {
    let th = Theory::Base(dr.fam, dr.params.clone());
    let lattice = base.kind() == Kind::Discrete && base.support().as_points().is_none();
    let unchanged = (base.clone(), String::new(), th.clone());
    match r.below(15) {
        0..=9 => unchanged,
        10 => {
            let (a, b) = if lattice {
                (
                    if r.chance(0.5) { ctx.one() } else { -ctx.one() },
                    ctx.int(r.range(-20, 20)),
                )
            } else {
                let a = small_rational(r, ctx, 9);
                (
                    if r.chance(0.5) { a.clone() } else { -a },
                    ctx.rational(r.range(-20, 20), r.range(1, 4)),
                )
            };
            match base.affine(a.clone(), b.clone()) {
                Ok(d) => (
                    d,
                    format!("affine {a}·X + {b}"),
                    Theory::Affine(dr.fam, dr.params.clone()),
                ),
                Err(_) => unchanged,
            }
        }
        11 => {
            let pts = points(base, r, ctx, 3);
            let (Some(a), Some(z)) = (pts.first(), pts.last()) else {
                return unchanged;
            };
            if a == z {
                return unchanged;
            }
            let region = Support::interval(a.clone(), z.clone());
            match base.truncated(&region) {
                Ok(d) => (d, format!("truncated to [{a}, {z}]"), Theory::Unknown),
                Err(_) => unchanged,
            }
        }
        12 => {
            let n = 1 + r.below(5) as usize;
            let k = 1 + r.below(n as u64) as usize;
            match order_statistic(base, n, k) {
                Ok(d) => (d, format!("the {k}-th of {n}"), Theory::Unknown),
                Err(_) => unchanged,
            }
        }
        13 => {
            for _ in 0..6 {
                let other = draw(r, ctx, false);
                let Ok(od) = other.build(ctx, true) else {
                    continue;
                };
                if od.kind() != base.kind() {
                    continue;
                }
                let q = r.range(2, 16);
                let w = ctx.rational(r.range(1, q - 1), q);
                let rest = ctx.one() - &w;
                if let Ok(m) =
                    Distribution::mixture(&[(w.clone(), base.clone()), (rest, od.clone())])
                {
                    return (
                        m,
                        format!("mixture with weight {w} and {od}"),
                        Theory::Mixture(dr.fam, dr.params.clone(), other.fam, other.params.clone()),
                    );
                }
            }
            unchanged
        }
        _ => {
            let Some((lo, hi)) = hull(base) else {
                return unchanged;
            };
            let x = ctx.symbol("_fz_w");
            let discrete = base.kind() != Kind::Continuous;
            if discrete && !bounded(base) {
                return unchanged;
            }
            let Some(map) = monotone_map(r, ctx, &x, &lo, &hi, discrete) else {
                return unchanged;
            };
            match base.transformed(&x, &map.g) {
                Ok(d) => (d, format!("transformed by {}", map.g), Theory::Unknown),
                Err(_) => unchanged,
            }
        }
    }
}

const CHECKS: [&str; 16] = [
    "density",
    "density",
    "cdf",
    "cdf",
    "derivative",
    "derivative",
    "quantile",
    "quantile",
    "moments",
    "moments",
    "mgf",
    "numdist",
    "order",
    "sums",
    "transformed",
    "transformed",
];

/// The generic checks on a drawn (and maybe wrapped) family.
fn generic(r: &mut Src, ctx: &Context, which: usize, desc: &mut String) -> Check {
    let dr = draw(r, ctx, true);
    describe(desc, dr.text());
    let verdict = check_validate(&dr, ctx)?;
    if !dr.valid {
        return Ok(verdict);
    }
    let base = dr
        .build(ctx, true)
        .map_err(|e| format!("validation: valid parameters rejected: {e}"))?;
    let (d, wrapped, th) = wrap(&base, &dr, r, ctx);
    if !wrapped.is_empty() {
        desc.push_str(&format!(", {wrapped}: {d}"));
        show!("  wrapped: {wrapped}: {d}");
    }
    let budget = Budget(Instant::now());
    let pts = points(&d, r, ctx, 4);
    if showing() {
        let ps: Vec<String> = pts.iter().map(ToString::to_string).collect();
        eprintln!("  points: {}", ps.join(", "));
    }
    match CHECKS[which] {
        "density" => check_density(&d, &pts, ctx, &th, &budget),
        "cdf" => check_cdf(&d, &pts, ctx, &budget),
        "derivative" => check_derivative(&d, &pts, ctx, &budget),
        "quantile" => check_quantile(&d, &pts, ctx, r, &budget),
        "moments" => {
            let n = 1 + r.below(4) as u32;
            check_moments(&d, ctx, &th, n, &budget)
        }
        _ => check_mgf(&d, ctx, &th, r, &budget),
    }
}

fuzz_target!(|data: &[u8]| {
    let mut r = Src::new(data);
    let which = r.below(CHECKS.len() as u64) as usize;
    let ctx = Context::new();
    let mut desc = String::new();
    show!("fuzz_stats: check {}", CHECKS[which]);
    let budget = Budget(Instant::now());
    let result = match CHECKS[which] {
        "numdist" => numdist_case(&mut r, &ctx, &mut desc, &budget),
        "order" => order_case(&mut r, &ctx, &mut desc),
        "sums" => sum_case(&mut r, &ctx, &mut desc),
        "transformed" => transform_case(&mut r, &ctx, &mut desc, &budget),
        _ => generic(&mut r, &ctx, which, &mut desc),
    };
    match result {
        Ok(verdict) => show!("  verdict: {verdict}"),
        Err(msg) => panic!("fuzz_stats: {msg}\n  case: {desc}"),
    }
});
