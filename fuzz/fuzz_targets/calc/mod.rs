//! The calculus differential checks of `fuzz_calculus` (`#[path]`-included,
//! not a target itself) — the 0.30 in-house calculus hunter.
//!
//! Every oracle is independent of the routine under test: an adaptive
//! Gauss–Legendre quadrature and a Richardson-extrapolated partial-sum
//! table over this file's own `f64` evaluator of the generated tree, and
//! the crate's certified `eval_decimal` at exact rational points.
//!
//! | routine | claim | oracle |
//! |---|---|---|
//! | [`check_defint`] | `∫ₐᵇ f` (a closed form, or `Divergent`) | quadrature |
//! | [`check_limit`] | `lim f` (a value, `±∞`, or "the sides differ") | `f` at `p ± 10⁻ᵏ`, `k = 10..40` |
//! | [`check_series`] | `f − S = O(hⁿ)` | the residual at `h = ±10⁻²..10⁻⁵` must not grow like a lower power |
//! | [`check_diff`] | `f′(p)` | central differences at `h = 10⁻¹², 10⁻¹⁴` (25 digits) |
//! | [`check_solve`] | every solution satisfies the equation; no real root missed | substitution; sign changes on `[−12, 12]` |
//! | [`check_sum`] | `Σ_{k≥a} t(k)` / `Σ_{k=a}^{n} t(k)` | Richardson on partial sums / the partial sums |
//! | [`check_ode`] | `dsolve` returns a solution | substitution at two points |
//!
//! A check returns [`V::Bad`] only for a wrong answer; everything the
//! oracles cannot decide (non-convergence, domain problems, unevaluated
//! results, refusals) is a [`V::Skip`].
#![allow(dead_code)]

use std::fmt;

use super::choose::Src;
use symplex::prelude::*;

// ───────────────────────────── AST ─────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum U {
    Sqrt,
    Exp,
    Ln,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Asin,
    Acos,
    Atan,
    Asinh,
    Acosh,
    Atanh,
    Abs,
    Sign,
}

impl U {
    fn name(self) -> &'static str {
        match self {
            U::Sqrt => "sqrt",
            U::Exp => "exp",
            U::Ln => "ln",
            U::Sin => "sin",
            U::Cos => "cos",
            U::Tan => "tan",
            U::Sinh => "sinh",
            U::Cosh => "cosh",
            U::Tanh => "tanh",
            U::Asin => "asin",
            U::Acos => "acos",
            U::Atan => "atan",
            U::Asinh => "asinh",
            U::Acosh => "acosh",
            U::Atanh => "atanh",
            U::Abs => "abs",
            U::Sign => "sign",
        }
    }

    /// The real function (`NaN` outside its real domain).
    fn f(self, v: f64) -> f64 {
        match self {
            U::Sqrt if v < 0.0 => f64::NAN,
            U::Sqrt => v.sqrt(),
            U::Exp => v.exp(),
            U::Ln if v <= 0.0 => f64::NAN,
            U::Ln => v.ln(),
            U::Sin => v.sin(),
            U::Cos => v.cos(),
            U::Tan => v.tan(),
            U::Sinh => v.sinh(),
            U::Cosh => v.cosh(),
            U::Tanh => v.tanh(),
            U::Asin | U::Acos if v.abs() > 1.0 => f64::NAN,
            U::Asin => v.asin(),
            U::Acos => v.acos(),
            U::Atan => v.atan(),
            U::Asinh => v.asinh(),
            U::Acosh if v < 1.0 => f64::NAN,
            U::Acosh => v.acosh(),
            U::Atanh if v.abs() >= 1.0 => f64::NAN,
            U::Atanh => v.atanh(),
            U::Abs => v.abs(),
            U::Sign if v.is_nan() => f64::NAN,
            U::Sign if v > 0.0 => 1.0,
            U::Sign if v < 0.0 => -1.0,
            U::Sign => 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub enum E {
    X,
    Int(i64),
    Rat(i64, i64),
    Pi,
    /// An exact constant given as a parse string and its value (`1/10^13`).
    Lit(String, f64),
    Un(U, Box<E>),
    Add(Box<E>, Box<E>),
    Sub(Box<E>, Box<E>),
    Mul(Box<E>, Box<E>),
    Div(Box<E>, Box<E>),
    PowI(Box<E>, i64),
    PowQ(Box<E>, i64, i64),
    PowE(Box<E>, Box<E>),
    /// `factorial(a)` (sums only; the product for integer `a`).
    Fact(Box<E>),
    /// `binomial(a, m)` (sums only).
    Binom(Box<E>, i64),
}

fn fact_f64(v: f64) -> f64 {
    if v < 0.0 || v.fract() != 0.0 || v > 170.0 {
        return f64::NAN;
    }
    (1..=v as i64).fold(1.0, |acc, i| acc * i as f64)
}

fn binom_f64(a: f64, m: i64) -> f64 {
    (0..m).fold(1.0, |acc, i| acc * (a - i as f64) / (i + 1) as f64)
}

impl fmt::Display for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            E::X => write!(f, "x"),
            E::Int(n) if *n < 0 => write!(f, "({n})"),
            E::Int(n) => write!(f, "{n}"),
            E::Rat(p, q) => write!(f, "({p}/{q})"),
            E::Pi => write!(f, "pi"),
            E::Lit(s, _) => write!(f, "({s})"),
            E::Un(u, a) => write!(f, "{}({a})", u.name()),
            E::Add(a, b) => write!(f, "({a} + {b})"),
            E::Sub(a, b) => write!(f, "({a} - {b})"),
            E::Mul(a, b) => write!(f, "({a}*{b})"),
            E::Div(a, b) => write!(f, "({a}/{b})"),
            E::PowI(a, n) => write!(f, "({a})^({n})"),
            E::PowQ(a, p, q) => write!(f, "({a})^({p}/{q})"),
            E::PowE(a, b) => write!(f, "({a})^({b})"),
            E::Fact(a) => write!(f, "factorial({a})"),
            E::Binom(a, m) => write!(f, "binomial({a}, {m})"),
        }
    }
}

impl E {
    /// The real value at `x` (`NaN` outside the real domain).
    pub fn ev(&self, x: f64) -> f64 {
        match self {
            E::X => x,
            E::Int(n) => *n as f64,
            E::Rat(p, q) => *p as f64 / *q as f64,
            E::Pi => std::f64::consts::PI,
            E::Lit(_, v) => *v,
            E::Un(u, a) => u.f(a.ev(x)),
            E::Add(a, b) => a.ev(x) + b.ev(x),
            E::Sub(a, b) => a.ev(x) - b.ev(x),
            E::Mul(a, b) => a.ev(x) * b.ev(x),
            E::Div(a, b) => {
                let d = b.ev(x);
                if d == 0.0 { f64::NAN } else { a.ev(x) / d }
            }
            E::PowI(a, n) => {
                let v = a.ev(x);
                if v == 0.0 && *n < 0 {
                    f64::NAN
                } else {
                    v.powi(*n as i32)
                }
            }
            E::PowQ(a, p, q) => {
                let v = a.ev(x);
                if v < 0.0 || (v == 0.0 && *p < 0) {
                    f64::NAN
                } else {
                    v.powf(*p as f64 / *q as f64)
                }
            }
            E::PowE(a, b) => {
                let v = a.ev(x);
                let e = b.ev(x);
                if v < 0.0 && e.fract() == 0.0 {
                    v.powi(e as i32)
                } else if v <= 0.0 {
                    f64::NAN
                } else {
                    v.powf(e)
                }
            }
            E::Fact(a) => fact_f64(a.ev(x)),
            E::Binom(a, m) => binom_f64(a.ev(x), *m),
        }
    }

    pub fn b(self) -> Box<E> {
        Box::new(self)
    }

    fn size(&self) -> usize {
        match self {
            E::X | E::Int(_) | E::Rat(..) | E::Pi | E::Lit(..) => 1,
            E::Un(_, a) | E::PowI(a, _) | E::PowQ(a, ..) | E::Fact(a) | E::Binom(a, _) => {
                1 + a.size()
            }
            E::Add(a, b) | E::Sub(a, b) | E::Mul(a, b) | E::Div(a, b) | E::PowE(a, b) => {
                1 + a.size() + b.size()
            }
        }
    }

    fn has_x(&self) -> bool {
        match self {
            E::X => true,
            E::Int(_) | E::Rat(..) | E::Pi | E::Lit(..) => false,
            E::Un(_, a) | E::PowI(a, _) | E::PowQ(a, ..) | E::Fact(a) | E::Binom(a, _) => a.has_x(),
            E::Add(a, b) | E::Sub(a, b) | E::Mul(a, b) | E::Div(a, b) | E::PowE(a, b) => {
                a.has_x() || b.has_x()
            }
        }
    }
}

const ALL_U: [U; 17] = [
    U::Sqrt,
    U::Exp,
    U::Ln,
    U::Sin,
    U::Cos,
    U::Tan,
    U::Sinh,
    U::Cosh,
    U::Tanh,
    U::Asin,
    U::Acos,
    U::Atan,
    U::Asinh,
    U::Acosh,
    U::Atanh,
    U::Abs,
    U::Sign,
];

fn gen_const(r: &mut Src) -> E {
    match r.below(10) {
        0..=4 => E::Int(*r.pick(&[1, 2, 3, -1, -2, 1, 2, 5])),
        5..=7 => {
            let (p, q) = *r.pick(&[
                (1, 2),
                (1, 3),
                (2, 3),
                (-1, 2),
                (3, 2),
                (1, 4),
                (-3, 4),
                (5, 3),
            ]);
            E::Rat(p, q)
        }
        8 => E::Pi,
        _ => E::Int(r.range(-4, 6)),
    }
}

fn gen_leaf(r: &mut Src) -> E {
    if r.chance(0.6) { E::X } else { gen_const(r) }
}

/// A random expression in `x` of depth at most `depth`.
pub fn gen_e(r: &mut Src, depth: u32) -> E {
    if depth == 0 || r.chance(0.2) {
        return gen_leaf(r);
    }
    let d = depth - 1;
    match r.below(20) {
        0..=5 => {
            let u = *r.pick(&ALL_U);
            E::Un(u, gen_e(r, d).b())
        }
        6..=8 => E::Add(gen_e(r, d).b(), gen_e(r, d).b()),
        9..=10 => E::Sub(gen_e(r, d).b(), gen_e(r, d).b()),
        11..=13 => E::Mul(gen_e(r, d).b(), gen_e(r, d).b()),
        14..=15 => E::Div(gen_e(r, d).b(), gen_e(r, d).b()),
        16..=17 => {
            let n = *r.pick(&[2, 3, -1, -2, 4, -3]);
            E::PowI(gen_e(r, d).b(), n)
        }
        18 => {
            let (p, q) = *r.pick(&[(1, 2), (1, 3), (3, 2), (-1, 2), (2, 3), (-1, 3), (5, 2)]);
            E::PowQ(gen_e(r, d).b(), p, q)
        }
        _ => E::PowE(gen_leaf(r).b(), gen_e(r, d).b()),
    }
}

/// A random expression that contains `x` (at most 14 nodes).
pub fn gen_x(r: &mut Src, depth: u32) -> E {
    for _ in 0..8 {
        let e = gen_e(r, depth);
        if e.has_x() && e.size() <= 14 {
            return e;
        }
    }
    E::X
}

// ───────────────────────── numeric helpers ─────────────────────────

/// Parse `eval_decimal`'s output (`a`, `b*i`, `a + b*i`, `a - b*i`).
fn parse_dec(s: &str) -> Option<(f64, f64)> {
    let s = s.trim();
    if let Some(body) = s.strip_suffix("*i") {
        if let Some(pos) = body.rfind(" + ").or_else(|| body.rfind(" - ")) {
            let re: f64 = body[..pos].trim().parse().ok()?;
            let sign = if &body[pos..pos + 3] == " - " {
                -1.0
            } else {
                1.0
            };
            let im: f64 = body[pos + 3..].trim().parse().ok()?;
            Some((re, sign * im))
        } else {
            let im: f64 = body.trim().parse().ok()?;
            Some((0.0, im))
        }
    } else if let Some(re) = s.strip_suffix(" + i") {
        Some((re.trim().parse().ok()?, 1.0))
    } else if let Some(re) = s.strip_suffix(" - i") {
        Some((re.trim().parse().ok()?, -1.0))
    } else if s == "i" {
        Some((0.0, 1.0))
    } else if s == "-i" {
        Some((0.0, -1.0))
    } else {
        Some((s.parse().ok()?, 0.0))
    }
}

fn num(e: &Ex, digits: u32) -> Result<(f64, f64), String> {
    match e.eval_decimal(digits) {
        Ok(s) => parse_dec(&s).ok_or_else(|| format!("unparsable decimal {s}")),
        Err(err) => Err(format!("{err}")),
    }
}

fn is_precision_refusal(e: &str) -> bool {
    e.contains("recision")
}

fn cabs(z: (f64, f64)) -> f64 {
    z.0.hypot(z.1)
}

fn cdist(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// Gauss–Legendre nodes/weights on [-1, 1] by Newton's iteration on P_n.
pub fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0f64, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        out.push((z, 2.0 / ((1.0 - z * z) * dp * dp)));
    }
    out
}

pub struct Quad {
    value: f64,
    err: f64,
    ok: bool,
    max_abs: f64,
}

impl Quad {
    fn failed(max_abs: f64) -> Self {
        Quad {
            value: f64::NAN,
            err: f64::INFINITY,
            ok: false,
            max_abs,
        }
    }
}

/// Adaptive Gauss–Legendre (a panel against its two halves) of `g` on
/// [a, b], explicit stack, at most 40,000 subdivisions.
fn quad(g: &dyn Fn(f64) -> f64, a: f64, b: f64, gl: &[(f64, f64)]) -> Quad {
    let panel = |lo: f64, hi: f64, max_abs: &mut f64| -> Option<f64> {
        let (c, h) = (0.5 * (lo + hi), 0.5 * (hi - lo));
        let mut s = 0.0;
        for &(t, w) in gl {
            let v = g(c + h * t);
            if !v.is_finite() {
                return None;
            }
            *max_abs = max_abs.max(v.abs());
            s += w * v;
        }
        Some(s * h)
    };
    let mut max_abs = 0.0f64;
    let Some(whole) = panel(a, b, &mut max_abs) else {
        return Quad::failed(max_abs);
    };
    let mut stack = vec![(a, b, whole, 0u32)];
    let mut total = 0.0;
    let mut err = 0.0;
    let mut evals = 0usize;
    let mut ok = true;
    let tol = 1e-13;
    while let Some((lo, hi, est, depth)) = stack.pop() {
        let mid = 0.5 * (lo + hi);
        evals += 1;
        let (Some(l), Some(r)) = (panel(lo, mid, &mut max_abs), panel(mid, hi, &mut max_abs))
        else {
            return Quad::failed(max_abs);
        };
        let diff = (l + r - est).abs();
        let width = (hi - lo) / (b - a);
        let tiny = hi - lo < 1e-15 * (1.0 + lo.abs());
        if diff <= tol * width.max(1e-3) * (1.0 + (l + r).abs().max(1.0)) || depth > 60 || tiny {
            if (depth > 60 || tiny) && diff > 1e-9 * (1.0 + (l + r).abs()) {
                ok = false;
            }
            total += l + r;
            err += diff;
        } else {
            stack.push((lo, mid, l, depth + 1));
            stack.push((mid, hi, r, depth + 1));
        }
        if evals > 40_000 {
            ok = false;
            break;
        }
    }
    Quad {
        value: total,
        err,
        ok: ok && stack.is_empty(),
        max_abs,
    }
}

// ───────────────────────── verdicts ─────────────────────────

#[derive(Debug, Clone)]
pub enum V {
    Ok,
    Skip(String),
    /// `(class, detail)`: a wrong answer.
    Bad(String, String),
}

fn short(s: &str) -> String {
    s.chars().take(60).collect()
}

fn parse_or_bad(ctx: &Context, s: &str) -> Result<Ex, V> {
    ctx.parse(s)
        .map_err(|e| V::Bad("PARSE".into(), format!("{s}: {e}")))
}

// ───────────────────────── definite integrals ─────────────────────────

#[derive(Clone, Debug)]
pub enum Bd {
    Q(i64, i64),
    PosInf,
    NegInf,
}

impl Bd {
    pub fn s(&self) -> String {
        match self {
            Bd::Q(p, q) => format!("{p}/{q}"),
            Bd::PosInf => "oo".into(),
            Bd::NegInf => "-oo".into(),
        }
    }
    fn f(&self) -> f64 {
        match self {
            Bd::Q(p, q) => *p as f64 / *q as f64,
            Bd::PosInf => f64::INFINITY,
            Bd::NegInf => f64::NEG_INFINITY,
        }
    }
}

pub fn gen_defint(r: &mut Src) -> (E, Bd, Bd) {
    let pts: [(i64, i64); 12] = [
        (-2, 1),
        (-1, 1),
        (-1, 2),
        (0, 1),
        (1, 3),
        (1, 2),
        (1, 1),
        (3, 2),
        (2, 1),
        (3, 1),
        (-3, 2),
        (5, 2),
    ];
    let f = match r.below(10) {
        // abs / sign kinks at a random rational (possibly two close together)
        0 => {
            let (p, q) = *r.pick(&pts);
            let c = E::Rat(p, q);
            let g = gen_e(r, 2);
            let k1 = E::Un(U::Abs, E::Sub(E::X.b(), c.clone().b()).b());
            let k = if r.chance(0.5) {
                let close = E::Add(c.b(), E::Lit("1/10^13".into(), 1e-13).b());
                E::Add(k1.b(), E::Un(U::Abs, E::Sub(E::X.b(), close.b()).b()).b())
            } else if r.chance(0.5) {
                E::Un(U::Sign, E::Sub(E::X.b(), E::Rat(p, q).b()).b())
            } else {
                k1
            };
            E::Mul(k.b(), g.b())
        }
        // x^s * decaying factor, s near a convergence boundary
        1 => {
            let s = *r.pick(&[
                (-1, 2),
                (1, 2),
                (-2, 3),
                (1, 3),
                (3, 2),
                (-3, 4),
                (2, 1),
                (1, 1),
            ]);
            let dec = r
                .pick(&[
                    E::Un(U::Exp, E::Mul(E::Int(-1).b(), E::X.b()).b()),
                    E::PowI(E::Add(E::Int(1).b(), E::PowI(E::X.b(), 2).b()).b(), -1),
                    E::PowI(E::Add(E::Int(1).b(), E::X.b()).b(), -2),
                    E::Un(U::Exp, E::Mul(E::Int(-1).b(), E::PowI(E::X.b(), 2).b()).b()),
                    E::Un(U::Ln, E::Add(E::Int(1).b(), E::X.b()).b()),
                ])
                .clone();
            E::Mul(E::PowQ(E::X.b(), s.0, s.1).b(), dec.b())
        }
        _ => gen_x(r, 3),
    };
    let (a, b) = match r.below(10) {
        0 | 1 => (Bd::Q(*r.pick(&[0, 1, 0, 2]), 1), Bd::PosInf),
        2 => (Bd::NegInf, Bd::PosInf),
        _ => {
            let mut a = *r.pick(&pts);
            let mut b = *r.pick(&pts);
            if (a.0 as f64 / a.1 as f64) > (b.0 as f64 / b.1 as f64) {
                std::mem::swap(&mut a, &mut b);
            }
            if a == b {
                b = (a.0 + a.1, a.1);
            }
            (Bd::Q(a.0, a.1), Bd::Q(b.0, b.1))
        }
    };
    (f, a, b)
}

fn numeric_integral(f: &E, a: &Bd, b: &Bd, gl: &[(f64, f64)]) -> Quad {
    match (a, b) {
        (Bd::Q(..), Bd::Q(..)) => quad(&|x| f.ev(x), a.f(), b.f(), gl),
        (Bd::Q(..), Bd::PosInf) => {
            let a0 = a.f();
            quad(
                &|t| {
                    if t >= 1.0 {
                        return 0.0;
                    }
                    let x = a0 + t / (1.0 - t);
                    let v = f.ev(x);
                    if v == 0.0 {
                        0.0
                    } else {
                        v / ((1.0 - t) * (1.0 - t))
                    }
                },
                0.0,
                1.0,
                gl,
            )
        }
        (Bd::NegInf, Bd::PosInf) => {
            let l = numeric_integral(f, &Bd::Q(0, 1), &Bd::PosInf, gl);
            let neg = subst(f, &E::Mul(E::Int(-1).b(), E::X.b()));
            let r = numeric_integral(&neg, &Bd::Q(0, 1), &Bd::PosInf, gl);
            Quad {
                value: l.value + r.value,
                err: l.err + r.err,
                ok: l.ok && r.ok,
                max_abs: l.max_abs.max(r.max_abs),
            }
        }
        _ => Quad::failed(0.0),
    }
}

pub fn check_defint(f: &E, a: &Bd, b: &Bd) -> V {
    let gl = gauss_legendre(15);
    // Real and finite on a sample grid?
    let (lo, hi) = (a.f(), b.f());
    for i in 1..64 {
        let t = f64::from(i) / 64.0;
        let x = if lo.is_finite() && hi.is_finite() {
            lo + (hi - lo) * t
        } else if lo.is_finite() {
            lo + t / (1.0 - t)
        } else {
            (t - 0.5) * 40.0
        };
        if f.ev(x).is_nan() {
            return V::Skip("domain".into());
        }
    }
    let q = numeric_integral(f, a, b, &gl);
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let fe = match parse_or_bad(&ctx, &f.to_string()) {
        Ok(e) => e,
        Err(v) => return v,
    };
    let (Ok(ae), Ok(be)) = (ctx.parse(&a.s()), ctx.parse(&b.s())) else {
        return V::Skip("parse".into());
    };
    match fe.try_integrate_definite(&x, &ae, &be) {
        Ok(res) => {
            if res.has_unevaluated() {
                return V::Skip("unevaluated".into());
            }
            let v = match num(&res, 20) {
                Ok(v) => v,
                Err(e) => {
                    if q.ok && q.err < 1e-8 {
                        return V::Bad(
                            "defint:closed-form-not-evaluable".into(),
                            format!("result {res}: {e}; quad {}", q.value),
                        );
                    }
                    return V::Skip("closed-unevaluable".into());
                }
            };
            if !q.ok || !q.value.is_finite() {
                return V::Skip(format!("quad-unconverged(result {res})"));
            }
            let scale = q.value.abs().max(1e-3 * q.max_abs.min(1e6)).max(1e-12);
            let d = cdist(v, (q.value, 0.0));
            if d > 1e-8 * scale.max(1.0) + 100.0 * q.err {
                V::Bad(
                    "defint:wrong-value".into(),
                    format!("result {res} = {v:?}; quad {} ± {}", q.value, q.err),
                )
            } else {
                V::Ok
            }
        }
        Err(SymplexError::Divergent { .. }) => {
            if q.ok && q.err < 1e-10 * (1.0 + q.value.abs()) && q.max_abs < 1e8 {
                V::Bad(
                    "defint:false-divergent".into(),
                    format!("quad {} ± {}", q.value, q.err),
                )
            } else {
                V::Ok
            }
        }
        Err(e) => V::Skip(format!("err:{}", short(&e.to_string()))),
    }
}

// ───────────────────────── limits ─────────────────────────

#[derive(Clone, Debug)]
pub enum Pt {
    Q(i64, i64),
    Pi(i64, i64),
    PosInf,
    NegInf,
}

impl Pt {
    pub fn s(&self) -> String {
        match self {
            Pt::Q(p, q) => format!("({p}/{q})"),
            Pt::Pi(p, q) => format!("({p}*pi/{q})"),
            Pt::PosInf => "oo".into(),
            Pt::NegInf => "-oo".into(),
        }
    }
    fn e(&self) -> E {
        match self {
            Pt::Q(p, q) => E::Rat(*p, *q),
            Pt::Pi(p, q) => E::Mul(E::Rat(*p, *q).b(), E::Pi.b()),
            _ => E::X,
        }
    }
}

fn subst(f: &E, v: &E) -> E {
    match f {
        E::X => v.clone(),
        E::Int(_) | E::Rat(..) | E::Pi | E::Lit(..) => f.clone(),
        E::Un(u, a) => E::Un(*u, subst(a, v).b()),
        E::Add(a, b) => E::Add(subst(a, v).b(), subst(b, v).b()),
        E::Sub(a, b) => E::Sub(subst(a, v).b(), subst(b, v).b()),
        E::Mul(a, b) => E::Mul(subst(a, v).b(), subst(b, v).b()),
        E::Div(a, b) => E::Div(subst(a, v).b(), subst(b, v).b()),
        E::PowI(a, n) => E::PowI(subst(a, v).b(), *n),
        E::PowQ(a, p, q) => E::PowQ(subst(a, v).b(), *p, *q),
        E::PowE(a, b) => E::PowE(subst(a, v).b(), subst(b, v).b()),
        E::Fact(a) => E::Fact(subst(a, v).b()),
        E::Binom(a, m) => E::Binom(subst(a, v).b(), *m),
    }
}

pub fn gen_limit(r: &mut Src) -> (E, Pt, Direction) {
    let p = match r.below(10) {
        0..=3 => Pt::Q(0, 1),
        4 => Pt::Q(1, 1),
        5 => Pt::Q(*r.pick(&[-1, 1, 2, -2, 1, 3]), *r.pick(&[1, 2, 3])),
        6 => Pt::Pi(*r.pick(&[1, 1, -1, 2]), *r.pick(&[1, 2, 4, 6])),
        7 | 8 => Pt::PosInf,
        _ => Pt::NegInf,
    };
    let dir = match r.below(3) {
        0 => Direction::Both,
        1 => Direction::Right,
        _ => Direction::Left,
    };
    // t → 0 (or ∞) as the small/large quantity
    let t = match &p {
        Pt::PosInf | Pt::NegInf => E::X,
        _ => E::Sub(E::X.b(), p.e().b()),
    };
    let f = match r.below(6) {
        // (g(x) - g(p)) / (h(x) - h(p)) — 0/0
        0 if !matches!(p, Pt::PosInf | Pt::NegInf) => {
            let g = gen_x(r, 2);
            let h = gen_x(r, 2);
            let pe = p.e();
            E::Div(
                E::Sub(g.clone().b(), subst(&g, &pe).b()).b(),
                E::Sub(h.clone().b(), subst(&h, &pe).b()).b(),
            )
        }
        // g(t) / t^k
        1 => {
            let g = subst(&gen_x(r, 3), &t);
            let k = *r.pick(&[1, 2, 3, 1]);
            E::Div(g.b(), E::PowI(t.b(), k).b())
        }
        // |g(t)|^h(t)
        2 => {
            let g = subst(&gen_x(r, 2), &t);
            let h = subst(&gen_x(r, 2), &t);
            E::PowE(E::Un(U::Abs, g.b()).b(), h.b())
        }
        // g(1/t)
        3 => subst(&gen_x(r, 3), &E::PowI(t.b(), -1)),
        _ => subst(&gen_x(r, 3), &t),
    };
    (f, p, dir)
}

/// f at the exact point p + side·10^-k (or ±10^k), 30 digits.
fn eval_near(
    ctx: &Context,
    fe: &Ex,
    x: &Ex,
    p: &Pt,
    side: i32,
    k: u32,
) -> Result<(f64, f64), String> {
    let s = match p {
        Pt::PosInf => format!("10^{k}"),
        Pt::NegInf => format!("-10^{k}"),
        _ => format!("{} + ({side})/10^{k}", p.s()),
    };
    let pt = ctx.parse(&s).map_err(|e| e.to_string())?;
    num(&fe.subs(x, &pt), 30)
}

#[derive(Debug)]
enum Seq {
    Conv((f64, f64)),
    PosInf,
    NegInf,
    Unclear,
    Undefined,
}

fn numeric_limit(ctx: &Context, fe: &Ex, x: &Ex, p: &Pt, side: i32) -> Seq {
    let ks: &[u32] = &[10, 15, 20, 25, 30, 40];
    let mut ys = Vec::new();
    for &k in ks {
        match eval_near(ctx, fe, x, p, side, k) {
            Ok(v) => ys.push(v),
            Err(_) => return Seq::Undefined,
        }
    }
    let n = ys.len();
    let (a, b, c) = (ys[n - 3], ys[n - 2], ys[n - 1]);
    if c.1 == 0.0
        && c.0.is_infinite()
        && b.1 == 0.0
        && b.0.abs() > 1e100
        && b.0.signum() == c.0.signum()
    {
        return if c.0 > 0.0 { Seq::PosInf } else { Seq::NegInf };
    }
    if !c.0.is_finite() || !c.1.is_finite() {
        return Seq::Unclear;
    }
    let scale = cabs(c).max(1.0);
    if cdist(b, c) <= 1e-9 * scale && cdist(a, b) <= 1e-6 * scale {
        return Seq::Conv(c);
    }
    // monotone growth to ±∞ (real)
    if a.1 == 0.0 && b.1 == 0.0 && c.1 == 0.0 {
        let grows = |s: f64| a.0 * s > 1e3 && b.0 * s > a.0 * s * 1.5 && c.0 * s > b.0 * s * 1.5;
        if grows(1.0) {
            return Seq::PosInf;
        }
        if grows(-1.0) {
            return Seq::NegInf;
        }
    }
    Seq::Unclear
}

pub fn check_limit(f: &E, p: &Pt, dir: Direction) -> V {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let fe = match parse_or_bad(&ctx, &f.to_string()) {
        Ok(e) => e,
        Err(v) => return v,
    };
    let Ok(pe) = ctx.parse(&p.s()) else {
        return V::Skip("parse".into());
    };
    let res = fe.try_limit_dir(&x, &pe, dir);
    let sides: Vec<i32> = match (p, dir) {
        (Pt::PosInf | Pt::NegInf, _) => vec![1],
        (_, Direction::Right) => vec![1],
        (_, Direction::Left) => vec![-1],
        (_, Direction::Both) => vec![1, -1],
    };
    let seqs: Vec<Seq> = sides
        .iter()
        .map(|&s| numeric_limit(&ctx, &fe, &x, p, s))
        .collect();
    let res = match res {
        Ok(r) => r,
        Err(e) => {
            // A two-sided claim of non-existence when both sides converge to the same value?
            if let [Seq::Conv(a), Seq::Conv(b)] = seqs.as_slice()
                && cdist(*a, *b) < 1e-12 * cabs(*a).max(1.0)
                && e.to_string().contains("differ")
            {
                return V::Bad(
                    "limit:false-differ".into(),
                    format!("{e}; numeric {a:?} {b:?}"),
                );
            }
            return V::Skip("err".into());
        }
    };
    if res.has_unevaluated() {
        return V::Skip("unevaluated".into());
    }
    let rs = res.to_string();
    let claim = if rs == "oo" {
        Seq::PosInf
    } else if rs == "-oo" {
        Seq::NegInf
    } else {
        match num(&res, 20) {
            Ok(v) => Seq::Conv(v),
            Err(e) => return V::Skip(format!("result-unevaluable {rs}: {}", short(&e))),
        }
    };
    for s in &seqs {
        match (s, &claim) {
            (Seq::Conv(a), Seq::Conv(b)) => {
                if cdist(*a, *b) > 1e-6 * cabs(*a).max(1.0) {
                    return V::Bad(
                        "limit:wrong-value".into(),
                        format!("claimed {rs}; numeric {seqs:?}"),
                    );
                }
            }
            (Seq::PosInf, Seq::PosInf) | (Seq::NegInf, Seq::NegInf) => {}
            (Seq::Conv(_), _) | (Seq::PosInf | Seq::NegInf, Seq::Conv(_)) => {
                return V::Bad(
                    "limit:wrong-kind".into(),
                    format!("claimed {rs}; numeric {seqs:?}"),
                );
            }
            (Seq::PosInf, Seq::NegInf) | (Seq::NegInf, Seq::PosInf) => {
                return V::Bad(
                    "limit:wrong-sign-inf".into(),
                    format!("claimed {rs}; numeric {seqs:?}"),
                );
            }
            (_, Seq::Unclear | Seq::Undefined) => return V::Skip("claim?".into()),
            (Seq::Unclear, _) => return V::Skip(format!("numeric-unclear(claimed {rs})")),
            (Seq::Undefined, _) => return V::Skip(format!("numeric-undefined(claimed {rs})")),
        }
    }
    V::Ok
}

// ───────────────────────── series ─────────────────────────

pub fn gen_series(r: &mut Src) -> (E, Pt, u32) {
    let p = match r.below(6) {
        0..=2 => Pt::Q(0, 1),
        3 => Pt::Q(1, 1),
        4 => Pt::Q(*r.pick(&[-1, 1, 2, -2, 3]), *r.pick(&[2, 3])),
        _ => Pt::Pi(*r.pick(&[1, -1]), *r.pick(&[2, 4, 6])),
    };
    let n = r.range(1, 6) as u32;
    (gen_x(r, 3), p, n)
}

pub fn check_series(f: &E, p: &Pt, n: u32) -> V {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let fe = match parse_or_bad(&ctx, &f.to_string()) {
        Ok(e) => e,
        Err(v) => return v,
    };
    let Ok(pe) = ctx.parse(&p.s()) else {
        return V::Skip("parse".into());
    };
    let s = fe.series(&x, &pe, n);
    if s.has_unevaluated() {
        return V::Skip("unevaluated".into());
    }
    // residual r(h) = f(p+h) - S(p+h), q(h) = |r|/|h|^n must not blow up.
    let mut report = String::new();
    let mut bad_sides = 0;
    let mut tested = 0;
    for side in [1i32, -1] {
        let mut qs = Vec::new();
        for k in [2u32, 3, 4, 5] {
            let Ok(pt) = ctx.parse(&format!("{} + ({side})/10^{k}", p.s())) else {
                continue;
            };
            let fv = fe.subs(&x, &pt);
            if num(&fv, 10).is_err() {
                break; // f undefined there
            }
            let resid = &fv - &s.subs(&x, &pt);
            match num(&resid, 12) {
                Ok(v) => qs.push(cabs(v) * 10f64.powi((k * n) as i32)),
                Err(e) if is_precision_refusal(&e) => qs.push(0.0), // exactly zero residual
                Err(_) => break,
            }
        }
        if qs.len() < 4 {
            continue;
        }
        tested += 1;
        // growth by ≥ 5x per decade twice in a row, and not tiny
        let g1 = qs[2] > 5.0 * qs[1] && qs[1] > 5.0 * qs[0];
        let g2 = qs[3] > 5.0 * qs[2] && qs[2] > 5.0 * qs[1];
        if (g1 || g2) && qs[3] > 1e-6 {
            bad_sides += 1;
            report += &format!("side {side}: q = {qs:?}; ");
        }
    }
    if tested == 0 {
        return V::Skip("series-unevaluable".into());
    }
    if bad_sides > 0 {
        return V::Bad("series:wrong".into(), format!("S = {s}; {report}"));
    }
    V::Ok
}

// ───────────────────────── diff ─────────────────────────

pub fn gen_diff(r: &mut Src) -> (E, (i64, i64)) {
    let f = gen_x(r, 3);
    let p = *r.pick(&[(1, 3), (1, 2), (2, 1), (-1, 2), (5, 7), (3, 2)]);
    (f, p)
}

pub fn check_diff(f: &E, p: (i64, i64)) -> V {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let fe = match parse_or_bad(&ctx, &f.to_string()) {
        Ok(e) => e,
        Err(v) => return v,
    };
    let d = fe.diff(&x);
    let pt = ctx.rational(p.0, p.1);
    let dv = match num(&d.subs(&x, &pt), 25) {
        Ok(v) => v,
        Err(_) => return V::Skip("d-unevaluable".into()),
    };
    // central differences with h = 1e-12 and 1e-14 at certified precision
    let mut fds = Vec::new();
    for k in [12u32, 14] {
        let h = ctx.rational(1, 10i64.pow(k));
        let hp = &pt + &h;
        let hm = &pt - &h;
        let fd = (&fe.subs(&x, &hp) - &fe.subs(&x, &hm)) / (&h * 2);
        match num(&fd, 25) {
            Ok(v) => fds.push(v),
            Err(_) => return V::Skip("fd-unevaluable".into()),
        }
    }
    let scale = cabs(fds[1]).max(1.0);
    if cdist(fds[0], fds[1]) > 1e-10 * scale {
        return V::Skip("non-smooth".into());
    }
    if cdist(dv, fds[1]) > 1e-9 * scale {
        return V::Bad(
            "diff:wrong".into(),
            format!("d = {d} = {dv:?}; fd = {:?}", fds[1]),
        );
    }
    V::Ok
}

// ───────────────────────── solve ─────────────────────────

pub fn gen_solve(r: &mut Src) -> (E, E) {
    let c = gen_const(r);
    match r.below(12) {
        0 => {
            // polynomial with chosen roots
            let k = r.range(1, 4);
            let mut e = E::Int(1);
            for _ in 0..k {
                let root = match r.below(3) {
                    0 => E::Int(r.range(-3, 3)),
                    1 => E::Rat(*r.pick(&[1, -1, 3, -3]), *r.pick(&[2, 3])),
                    _ => E::PowQ(E::Int(*r.pick(&[2, 3, 5])).b(), 1, 2),
                };
                e = E::Mul(e.b(), E::Sub(E::X.b(), root.b()).b());
            }
            (e, E::Int(0))
        }
        1 => (E::Un(U::Exp, E::X.b()), c),
        2 => (E::Un(U::Sin, E::X.b()), E::Rat(1, 2)),
        3 => (E::Mul(E::X.b(), E::Un(U::Exp, E::X.b()).b()), c),
        4 => (
            E::PowQ(E::X.b(), 1, 2),
            E::Sub(E::X.b(), E::Int(r.range(-2, 4)).b()),
        ),
        5 => (
            E::Un(U::Abs, E::Sub(E::X.b(), E::Int(r.range(-2, 2)).b()).b()),
            E::Int(r.range(-1, 3)),
        ),
        6 => (E::Un(U::Ln, gen_x(r, 1).b()), c),
        7 => {
            // rational equation
            let a = gen_x(r, 1);
            let b = gen_x(r, 1);
            (E::Div(a.b(), b.b()), c)
        }
        8 => (
            E::Add(
                E::PowI(E::X.b(), r.range(2, 5)).b(),
                E::Mul(E::Int(r.range(-3, 3)).b(), E::X.b()).b(),
            ),
            c,
        ),
        _ => (gen_x(r, 2), c),
    }
}

/// Sign changes of `g` on [-12, 12] refined by bisection, where `g` is
/// continuous across the bracket (checked by the midpoint's size).
fn real_roots_by_sampling(g: &dyn Fn(f64) -> f64) -> Vec<f64> {
    let mut roots = Vec::new();
    let n = 4000;
    let (lo, hi) = (-12.0, 12.0);
    let mut prev_x = lo;
    let mut prev = g(lo);
    for i in 1..=n {
        let x = lo + (hi - lo) * f64::from(i) / f64::from(n);
        let v = g(x);
        if prev == 0.0 {
            roots.push(prev_x);
        } else if prev.is_finite() && v.is_finite() && prev * v < 0.0 {
            let (mut a, mut b, mut fa) = (prev_x, x, prev);
            let mut okc = true;
            for _ in 0..200 {
                let m = 0.5 * (a + b);
                let fm = g(m);
                if !fm.is_finite() {
                    okc = false;
                    break;
                }
                if fm == 0.0 {
                    a = m;
                    b = m;
                    break;
                }
                if fa * fm < 0.0 {
                    b = m;
                } else {
                    a = m;
                    fa = fm;
                }
            }
            let m = 0.5 * (a + b);
            // reject poles (value large near the "root")
            if okc && g(m).abs() < 1e-6 {
                roots.push(m);
            }
        }
        prev = v;
        prev_x = x;
    }
    roots.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
    roots
}

pub fn check_solve(lhs: &E, rhs: &E) -> V {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let g = E::Sub(lhs.clone().b(), rhs.clone().b());
    let ge = match parse_or_bad(&ctx, &g.to_string()) {
        Ok(e) => e,
        Err(v) => return v,
    };
    let sols = match ge.solve(&x) {
        Ok(s) => s,
        Err(SymplexError::NoSolution { .. }) => vec![],
        Err(e) => return V::Skip(format!("err:{}", short(&e.to_string()))),
    };
    let shown = || sols.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let mut real_sols = Vec::new();
    for s in &sols {
        if s.has_unevaluated() || !s.free_symbols().is_empty() {
            // parametric (e.g. with an integer n)
            continue;
        }
        let at = ge.subs(&x, s);
        let Ok(sv) = num(s, 20) else {
            continue;
        };
        match num(&at, 15) {
            Ok(v) => {
                let scale = 1.0 + cabs(sv);
                if cabs(v) > 1e-12 * scale {
                    return V::Bad(
                        "solve:spurious".into(),
                        format!("sol {s} = {sv:?}; residual {v:?}; all {:?}", shown()),
                    );
                }
            }
            Err(e) if is_precision_refusal(&e) => {}
            Err(e) => {
                return V::Bad("solve:unevaluable-residual".into(), format!("sol {s}: {e}"));
            }
        }
        if sv.1.abs() < 1e-14 * (1.0 + sv.0.abs()) {
            real_sols.push(sv.0);
        }
    }
    // missing real roots (non-periodic equations only)
    let s = g.to_string();
    if s.contains("sin") || s.contains("cos") || s.contains("tan") || s.contains("sign") {
        return V::Ok;
    }
    for r0 in real_roots_by_sampling(&|t| g.ev(t)) {
        if !real_sols
            .iter()
            .any(|s| (s - r0).abs() < 1e-6 * (1.0 + r0.abs()))
        {
            return V::Bad(
                "solve:missing-root".into(),
                format!("root near {r0}; returned {:?}", shown()),
            );
        }
    }
    V::Ok
}

// ───────────────────────── summation ─────────────────────────

fn poly_x(r: &mut Src) -> E {
    let deg = r.range(0, 3);
    let mut e: Option<E> = None;
    for d in 0..=deg {
        let c = r.range(-3, 4);
        if c == 0 {
            continue;
        }
        let t = E::Mul(E::Int(c).b(), E::PowI(E::X.b(), d).b());
        e = Some(match e {
            None => t,
            Some(acc) => E::Add(acc.b(), t.b()),
        });
    }
    e.unwrap_or(E::Int(1))
}

/// A term in `x` (the summation variable), a lower bound, infinite?
pub fn gen_sum(r: &mut Src) -> (E, i64, bool) {
    let a = r.range(0, 3);
    let infinite = r.chance(0.6);
    let geo = |r: &mut Src| {
        let p = *r.pick(&[1, -1, 2, -2, 1, 3]);
        let q = *r.pick(&[2, 3, 5]);
        E::PowE(E::Rat(p, q).b(), E::X.b())
    };
    let (t, needs_pos) = match r.below(11) {
        0 => (poly_x(r), false),
        1 => (E::Mul(poly_x(r).b(), geo(r).b()), false),
        2 => {
            let (p, q) = (r.range(0, 3), r.range(0, 6));
            (
                E::Div(
                    E::Int(1).b(),
                    E::Mul(
                        E::Add(E::X.b(), E::Int(p).b()).b(),
                        E::Add(E::X.b(), E::Int(q).b()).b(),
                    )
                    .b(),
                ),
                true,
            )
        }
        3 => (E::PowI(E::X.b(), -r.range(2, 5)), true),
        4 => {
            let (c2, c1, c0) = (r.range(1, 3), r.range(0, 4), r.range(1, 5));
            let den = E::Add(
                E::Add(
                    E::Mul(E::Int(c2).b(), E::PowI(E::X.b(), 2).b()).b(),
                    E::Mul(E::Int(c1).b(), E::X.b()).b(),
                )
                .b(),
                E::Int(c0).b(),
            );
            (E::Div(poly_x(r).b(), den.b()), false)
        }
        5 => {
            let p = *r.pick(&[1, -1, 2, 3, -2]);
            let q = *r.pick(&[1, 2, 3]);
            (
                E::Div(
                    E::PowE(E::Rat(p, q).b(), E::X.b()).b(),
                    E::Fact(E::X.b()).b(),
                ),
                false,
            )
        }
        6 => (
            E::Div(
                E::PowE(E::Int(-1).b(), E::X.b()).b(),
                E::Add(E::X.b(), E::Int(r.range(1, 3)).b()).b(),
            ),
            false,
        ),
        7 => (
            E::Div(
                E::Mul(E::X.b(), E::PowE(E::Int(-1).b(), E::X.b()).b()).b(),
                E::Add(E::PowI(E::X.b(), 2).b(), E::Int(r.range(1, 3)).b()).b(),
            ),
            false,
        ),
        8 => {
            let (s, m) = (r.range(0, 3), r.range(0, 3));
            (
                E::Mul(
                    E::Binom(E::Add(E::X.b(), E::Int(s).b()).b(), m).b(),
                    geo(r).b(),
                ),
                false,
            )
        }
        9 => (
            E::Div(
                E::Int(1).b(),
                E::Mul(E::X.b(), E::Add(E::X.b(), E::Int(r.range(1, 3)).b()).b()).b(),
            ),
            true,
        ),
        _ => (gen_x(r, 2), true),
    };
    let lo = if needs_pos { a.max(1) } else { a };
    (t, lo, infinite)
}

/// Σ_{k ≥ lo} t(k) by partial sums at N = 64·2^j and Richardson in 1/N
/// (terms that are rational functions, possibly times (−1)^k at even N,
/// or decay geometrically).  `None` when the table is not self-consistent.
fn numeric_sum(t: &E, lo: i64) -> Option<f64> {
    let mut s = 0.0f64;
    let mut comp = 0.0f64; // Kahan
    let mut k = lo;
    let mut table: Vec<Vec<f64>> = Vec::new();
    for j in 0..13 {
        let n_hi = lo + 64 * (1i64 << j);
        while k < n_hi {
            let v = t.ev(k as f64);
            if !v.is_finite() {
                return None;
            }
            let y = v - comp;
            let tt = s + y;
            comp = (tt - s) - y;
            s = tt;
            k += 1;
        }
        let mut row = vec![s];
        for m in 1..=j.min(6) {
            let f = 2f64.powi(m as i32);
            let prev: &Vec<f64> = &table[j - 1];
            if prev.len() < m {
                break;
            }
            let v = (f * row[m - 1] - prev[m - 1]) / (f - 1.0);
            row.push(v);
        }
        table.push(row);
    }
    let last = table.last()?;
    let prev = &table[table.len() - 2];
    // best column: smallest |last[m] - prev[m]|
    let mut best: Option<(f64, f64)> = None;
    for m in 0..last.len().min(prev.len()) {
        let d = (last[m] - prev[m]).abs();
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, last[m]));
        }
    }
    let (d, v) = best?;
    (d <= 1e-11 * v.abs().max(1.0)).then_some(v)
}

pub fn check_sum(t: &E, lo: i64, infinite: bool) -> V {
    let ctx = Context::new();
    let k = ctx.symbol("x");
    let n = ctx.symbol("n");
    let te = match parse_or_bad(&ctx, &t.to_string()) {
        Ok(e) => e,
        Err(v) => return v,
    };
    if infinite {
        let numeric = numeric_sum(t, lo);
        let s = match te.try_summation(&k, &ctx.int(lo), &ctx.infinity()) {
            Ok(s) => s,
            Err(SymplexError::Divergent { .. }) => {
                if let Some(v) = numeric {
                    // terms must not be summable to a stable value
                    let tail = t.ev(1e6).abs() * 1e6;
                    if tail < 1e-6 {
                        return V::Bad("sum:false-divergent".into(), format!("numeric {v}"));
                    }
                }
                return V::Ok;
            }
            Err(_) => return V::Skip("unevaluated".into()),
        };
        let sv = match num(&s, 20) {
            Ok(v) => v,
            Err(e) => {
                if s.to_string().contains("oo") {
                    if numeric.is_some() && t.ev(1e6).abs() * 1e6 < 1e-6 {
                        return V::Bad(
                            "sum:false-infinite".into(),
                            format!("closed {s}; numeric {numeric:?}"),
                        );
                    }
                    return V::Skip("inf".into());
                }
                return V::Skip(format!("unevaluable {s}: {}", short(&e)));
            }
        };
        let Some(v) = numeric else {
            return V::Skip(format!("oracle-unconverged(closed {s})"));
        };
        if cdist((v, 0.0), sv) > 1e-8 * v.abs().max(1.0) {
            return V::Bad(
                "sum:wrong-infinite".into(),
                format!("closed {s} = {sv:?}; numeric {v:?}"),
            );
        }
        V::Ok
    } else {
        let s = te.summation(&k, &ctx.int(lo), &n);
        if s.has_unevaluated() {
            return V::Skip("unevaluated".into());
        }
        let mut partial = ctx.int(0);
        for m in lo..lo + 12 {
            partial = &partial + &te.subs(&k, &ctx.int(m));
            let at = s.subs(&n, &ctx.int(m));
            let d = &at - &partial;
            match num(&d, 12) {
                Ok(v) => {
                    let scale = num(&partial, 12).map(cabs).unwrap_or(1.0).max(1.0);
                    if cabs(v) > 1e-15 * scale {
                        return V::Bad(
                            "sum:wrong-finite".into(),
                            format!(
                                "S(n) = {s}; at n = {m}: S = {}, partial sum = {}",
                                at.eval(),
                                partial.eval()
                            ),
                        );
                    }
                }
                Err(e) if is_precision_refusal(&e) => {}
                Err(e) => {
                    if num(&partial, 12).is_err() {
                        return V::Skip("partial-unevaluable".into());
                    }
                    return V::Bad(
                        "sum:unevaluable".into(),
                        format!("S(n) = {s} at n = {m}: {e}"),
                    );
                }
            }
        }
        V::Ok
    }
}

// ───────────────────────── dsolve ─────────────────────────

pub fn gen_ode(r: &mut Src) -> String {
    let c = |r: &mut Src| r.range(-3, 3);
    match r.below(6) {
        0 => {
            let (a, b) = (c(r), c(r));
            format!("y'' + ({a})*y' + ({b})*y")
        }
        1 => {
            let a = c(r);
            format!("y' + ({a})*y - ({})", gen_x(r, 1))
        }
        2 => format!("y' - ({})*y^2", gen_x(r, 1)),
        3 => {
            let a = r.range(1, 4);
            format!("y'' + ({a})*y - ({})", gen_x(r, 1))
        }
        4 => {
            let (a, m) = (c(r), r.range(0, 3));
            format!("x*y' + ({a})*y - x^{m}")
        }
        _ => format!("y' - ({})*y", gen_x(r, 2)),
    }
}

pub fn check_ode(ode: &str) -> V {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // the ODE with formal derivatives
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let s = ode.replace("y''", "D2Y").replace("y'", "D1Y");
    let e = match parse_or_bad(&ctx, &s) {
        Ok(e) => e,
        Err(v) => return v,
    };
    let d1 = ctx.symbol("D1Y");
    let d2 = ctx.symbol("D2Y");
    let e2 = e.subs(&d2, &d2y).subs(&d1, &dy);
    let sol = e2.solve_ode(&y, &x);
    if sol.has_unevaluated() {
        return V::Skip("unevaluated".into());
    }
    // substitute the solution: y, y', y''
    let y1 = sol.diff(&x);
    let y2 = y1.diff(&x);
    let resid = e.subs(&d2, &y2).subs(&d1, &y1).subs(&y, &sol);
    let mut consts: Vec<Ex> = resid
        .free_symbols()
        .into_iter()
        .filter(|v| v.to_string() != "x")
        .collect();
    consts.sort_by_key(|v| v.to_string());
    for (pi, pt) in [(1i64, 3i64), (7, 5)].iter().enumerate() {
        let mut rr = resid.subs(&x, &ctx.rational(pt.0, pt.1));
        for (i, cs) in consts.iter().enumerate() {
            rr = rr.subs(cs, &ctx.rational(2 + i as i64 + pi as i64, 3));
        }
        match num(&rr, 12) {
            Ok(v) if cabs(v) > 1e-12 => {
                return V::Bad(
                    "dsolve:not-a-solution".into(),
                    format!("sol {sol}; residual {v:?} at {pt:?}"),
                );
            }
            Ok(_) => {}
            Err(e) if is_precision_refusal(&e) => {}
            Err(_) => return V::Skip("residual-unevaluable".into()),
        }
    }
    V::Ok
}
