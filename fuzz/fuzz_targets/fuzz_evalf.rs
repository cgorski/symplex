//! evalf self-consistency.
//!
//! For a random constant expression (special functions, sums, `RootOf`,
//! complex arguments, catastrophic cancellation, huge and tiny scales —
//! the 0.30 in-house evalf hunter's generator), every digit certified at
//! 16 and 30 significant digits (`eval_decimal`) must agree with the
//! 60-digit value up to one unit in the last certified digit, and
//! `eval_f64` / `eval_complex64` must be within one ulp of it:
//!
//! * a part printed as `0` must be negligible (the zero policy);
//! * `eval_f64` must refuse a value whose imaginary part is not negligible;
//! * an error other than a refusal (`PrecisionExhausted`, `Unevaluable`,
//!   `NotImplemented`, `Divergent`) at one precision while another
//!   precision evaluates is a finding.
//!
//! A failure panics with the expression, the class of the disagreement and
//! every value.  Inputs decode through [`choose::Src`], one decision per
//! byte (or a few for wide ranges); set `FUZZ_SHOW=1` to print the decoded
//! expression before it is evaluated.
#![no_main]

#[path = "common/choose.rs"]
mod choose;

use choose::Src;
use libfuzzer_sys::fuzz_target;
use symplex::num_bigint::BigInt;
use symplex::num_traits::{Signed, Zero};
use symplex::prelude::*;

// ── Expression trees ────────────────────────────────────────────────────────

/// A template with numbered holes `{0}`, `{1}`, … filled by its children
/// (parenthesised unless simple), or a literal.  `PW` is a two-branch
/// `Piecewise` built through the API (the parser does not read it).
#[derive(Clone, Debug)]
enum T {
    L(String),
    N(String, Vec<T>),
}

fn l(s: impl Into<String>) -> T {
    T::L(s.into())
}
fn n(tpl: &str, kids: Vec<T>) -> T {
    T::N(tpl.to_string(), kids)
}

fn simple(s: &str) -> bool {
    s.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

fn render(t: &T) -> String {
    match t {
        T::L(s) => s.clone(),
        T::N(tpl, kids) if tpl == "PW" => format!(
            "Piecewise(({}, {} > {}), ({}, True))",
            render(&kids[0]),
            render(&kids[1]),
            render(&kids[2]),
            render(&kids[3])
        ),
        T::N(tpl, kids) => {
            let mut out = tpl.clone();
            for (i, k) in kids.iter().enumerate() {
                let s = render(k);
                let s = if simple(&s) { s } else { format!("({s})") };
                out = out.replace(&format!("{{{i}}}"), &s);
            }
            out
        }
    }
}

fn build(ctx: &Context, t: &T) -> Result<Ex, SymplexError> {
    if let T::N(tpl, kids) = t
        && tpl == "PW"
    {
        let a = ctx.parse(&render(&kids[0]))?;
        let c = ctx.parse(&render(&kids[1]))?;
        let d = ctx.parse(&render(&kids[2]))?;
        let b = ctx.parse(&render(&kids[3]))?;
        let t = ctx.bool_true();
        return Ok(Ex::piecewise(&[(&a, &c.gt(&d)), (&b, &t)]));
    }
    ctx.parse(&render(t))
}

// ── Generator ───────────────────────────────────────────────────────────────

const KS: &[i64] = &[
    1, 2, 3, 5, 7, 8, 10, 12, 15, 16, 17, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150,
    200, 300,
];

fn k(r: &mut Src) -> i64 {
    *r.pick(KS)
}

fn tiny(r: &mut Src) -> T {
    let kk = k(r);
    match r.below(3) {
        0 => l(format!("10^(-{kk})")),
        1 => n(&format!("{}*10^(-{kk})", r.range(1, 9)), vec![]),
        _ => l(format!("{}/{}*10^(-{kk})", r.range(1, 29), r.range(2, 31))),
    }
}

fn small_int(r: &mut Src) -> i64 {
    let v = r.range(-9, 9);
    if v == 0 { 1 } else { v }
}

fn rational(r: &mut Src) -> String {
    let p = r.range(-40, 40);
    let q = r.range(1, 17);
    if q == 1 {
        format!("{p}")
    } else {
        format!("{p}/{q}")
    }
}

fn pos_rational(r: &mut Src) -> String {
    let p = r.range(1, 60);
    let q = r.range(1, 17);
    if q == 1 {
        format!("{p}")
    } else {
        format!("{p}/{q}")
    }
}

fn leaf(r: &mut Src) -> T {
    match r.below(16) {
        0..=3 => l(rational(r)),
        4 => l(format!("{}", small_int(r))),
        5 => l(format!("10^{}", k(r))),
        6 => l(format!("{}*10^{}", pos_rational(r), k(r))),
        7 => tiny(r),
        8 => l("pi"),
        9 => l("E"),
        10 => l(format!("sqrt({})", r.range(2, 50))),
        11 => l(format!(
            "{}^({}/{})",
            r.range(2, 30),
            r.range(-7, 7),
            r.range(2, 9)
        )),
        12 => l(*r.pick(&[
            "EulerGamma",
            "Catalan",
            "GoldenRatio",
            "pi/2",
            "pi/3",
            "-pi",
        ])),
        13 => l(format!("{}/{}", r.range(1, 99999999), r.range(1, 99999999))),
        14 => l(format!("{}", r.range(-1000, 1000))),
        _ => l(pos_rational(r)),
    }
}

/// A modest positive value (mostly in (0, 10]).
fn pos(r: &mut Src, depth: u32) -> T {
    if depth > 0 && r.chance(0.3) {
        return n("abs({0})", vec![gx(r, depth - 1)]);
    }
    match r.below(6) {
        0 => l(format!("{}/{}", r.range(1, 40), r.range(1, 9))),
        1 => l("pi"),
        2 => l("E"),
        3 => l(format!("sqrt({})", r.range(2, 30))),
        4 => l(format!("{}", r.range(1, 12))),
        _ => l(format!("{}/{}", r.range(1, 9), r.range(10, 99))),
    }
}

/// A value in (-1, 1).
fn unit(r: &mut Src) -> T {
    match r.below(5) {
        0 => l(format!("{}/{}", r.range(-98, 98), 99)),
        1 => l(format!("1 - {}", render(&tiny(r)))),
        2 => l(format!("-1 + {}", render(&tiny(r)))),
        3 => l(format!("{}/{}", r.range(-9, 9), r.range(10, 20))),
        _ => tiny(r),
    }
}

const UNARY: &[&str] = &[
    "sin",
    "cos",
    "tan",
    "exp",
    "ln",
    "sqrt",
    "abs",
    "sign",
    "floor",
    "ceiling",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "gamma",
    "loggamma",
    "digamma",
    "erf",
    "erfc",
    "erfi",
    "airyai",
    "airybi",
    "lambertw",
    "zeta",
    "Ei",
    "li",
    "Si",
    "Ci",
    "Shi",
    "Chi",
    "elliptic_k",
    "elliptic_e",
    "erfinv",
    "erfcinv",
    "fresnels",
    "fresnelc",
    "cot",
    "sec",
    "csc",
    "coth",
    "sech",
    "csch",
    "acot",
    "arg",
    "re",
    "im",
    "conjugate",
    "dirichlet_eta",
    "airyaiprime",
    "airybiprime",
    "cbrt",
];

const ELEM: &[&str] = &[
    "sin", "cos", "tan", "exp", "ln", "sqrt", "atan", "sinh", "cosh", "tanh", "asinh", "erf",
    "gamma", "abs", "cbrt",
];

fn order(r: &mut Src) -> String {
    r.pick(&[
        "0", "1", "2", "3", "1/2", "3/2", "5/2", "1/3", "7", "20", "-1/2", "-3/2", "2/3", "11/4",
        "50", "-5",
    ])
    .to_string()
}

fn special2(r: &mut Src, depth: u32) -> T {
    let x = if r.chance(0.5) {
        gx(r, depth - 1)
    } else {
        pos(r, depth - 1)
    };
    match r.below(15) {
        0 => n(&format!("besselj({}, {{0}})", order(r)), vec![x]),
        1 => n(&format!("bessely({}, {{0}})", order(r)), vec![x]),
        2 => n(&format!("besseli({}, {{0}})", order(r)), vec![x]),
        3 => n(&format!("besselk({}, {{0}})", order(r)), vec![x]),
        4 => {
            let s = *r.pick(&[
                "2", "3", "1", "0", "-1", "-2", "1/2", "5/2", "-3/2", "4", "7/3",
            ]);
            let arg = if r.chance(0.6) { unit(r) } else { x };
            n(&format!("polylog({s}, {{0}})"), vec![arg])
        }
        5 => n(
            &format!(
                "expint({}, {{0}})",
                r.pick(&["0", "1", "2", "3", "1/2", "5/2", "-1", "7"])
            ),
            vec![x],
        ),
        6 => n(&format!("uppergamma({}, {{0}})", order(r)), vec![x]),
        7 => n(&format!("lowergamma({}, {{0}})", order(r)), vec![x]),
        8 => n(
            &format!(
                "polygamma({}, {{0}})",
                r.pick(&["0", "1", "2", "3", "5", "10", "30", "100"])
            ),
            vec![x],
        ),
        9 => n(&format!("binomial({{0}}, {})", r.range(0, 12)), vec![x]),
        10 => n("beta({0}, {1})", vec![x, pos(r, depth - 1)]),
        11 => n("atan2({0}, {1})", vec![x, gx(r, depth - 1)]),
        12 => n("elliptic_f({0}, {1})", vec![x, unit(r)]),
        13 => n("elliptic_pi({0}, {1})", vec![unit(r), unit(r)]),
        _ => {
            let a = pos(r, 0);
            let b = pos(r, 0);
            let (x1, x2) = if r.chance(0.5) {
                (l("0"), unit(r))
            } else {
                (unit(r), l(format!("1 - {}", render(&tiny(r)))))
            };
            n(
                "betainc_regularized({0}, {1}, {2}, {3})",
                vec![a, b, x1, x2],
            )
        }
    }
}

fn sums(r: &mut Src) -> T {
    let x = l(format!("{}/{}", r.range(-9, 9), r.range(2, 12)));
    match r.below(14) {
        0 => l(format!("Sum(1/k^{}, k, 1, oo)", r.range(2, 6))),
        1 => n("Sum({0}^k/factorial(k), k, 0, oo)", vec![x]),
        2 => n(
            "Sum((-1)^k*{0}^(2*k+1)/factorial(2*k+1), k, 0, oo)",
            vec![x],
        ),
        3 => l(format!("Sum(1/(k^2 + {}), k, 1, oo)", pos_rational(r))),
        4 => l("Sum(1/(k*(k+1)), k, 1, oo)".to_string()),
        5 => l(format!(
            "Sum(binomial(2*k, k)*(1/{})^k, k, 0, oo)",
            r.range(5, 30)
        )),
        6 => l(format!("Sum(1/k, k, 1, {})", r.range(1, 200))),
        7 => n(&format!("Sum({{0}}^k, k, 0, {})", r.range(1, 40)), vec![x]),
        8 => l("Sum((-1)^k/(2*k+1), k, 0, oo)".to_string()),
        9 => l(format!("Sum(1/factorial(k), k, 0, {}) - E", r.range(5, 60))),
        10 => l(format!("Product(1 + 1/k^2, k, 1, {})", r.range(1, 30))),
        11 => l(format!("Sum((-1)^k/k^{}, k, 1, oo)", r.range(1, 4))),
        12 => l(format!("Sum(k^{}/2^k, k, 1, oo)", r.range(0, 5))),
        _ => l(format!(
            "Sum(1/(k^2 - {}), k, 2, oo)",
            r.pick(&["1/4", "1/9", "2", "1/2"])
        )),
    }
}

fn rootof(r: &mut Src) -> T {
    match r.below(6) {
        0 => l(format!(
            "RootOf(x^3 - x - {}, x, {})",
            rational(r),
            r.range(0, 2)
        )),
        1 => l(format!(
            "RootOf(x^5 - x + {}, x, {})",
            r.range(1, 5),
            r.range(0, 4)
        )),
        2 => l(format!(
            "RootOf(x^2 - 2*x + 1 - 10^(-{}), x, {})",
            k(r),
            r.range(0, 1)
        )),
        3 => l(format!("RootOf(x^4 - 10*x^2 + 1, x, {})", r.range(0, 3))),
        4 => l(format!(
            "RootOf(x^3 - {}*x + 1, x, {})",
            pos_rational(r),
            r.range(0, 2)
        )),
        _ => l(format!("RootOf(x^2 - 2 - 10^(-{}), x, 1) - sqrt(2)", k(r))),
    }
}

fn complexish(r: &mut Src, depth: u32) -> T {
    let x = gx(r, depth - 1);
    match r.below(10) {
        0 => n("{0} + I*{1}", vec![x, tiny(r)]),
        1 => n("sqrt(-{0} + I*{1})", vec![pos(r, 0), tiny(r)]),
        2 => n("sqrt(-{0} - I*{1})", vec![pos(r, 0), tiny(r)]),
        3 => n("ln(-{0} - I*{1})", vec![pos(r, 0), tiny(r)]),
        4 => {
            let e = *r.pick(&["1/3", "1/2", "5/7", "-2/3", "3", "-1"]);
            n(
                &format!("({{0}} + I*{{1}})^({e})"),
                vec![x, gx(r, depth - 1)],
            )
        }
        5 => n("{0} + I*(exp({1}) - 1 - {1})", vec![x, tiny(r)]),
        6 => n("exp(I*{0})", vec![x]),
        7 => n("{0}*(1 + I*{1})", vec![x, tiny(r)]),
        8 => n("sqrt(-{0})", vec![tiny(r)]),
        _ => {
            let f = *r.pick(UNARY);
            n(&format!("{f}({{0}} + I*{{1}})"), vec![x, gx(r, depth - 1)])
        }
    }
}

fn cancel(r: &mut Src, depth: u32) -> T {
    let t = tiny(r);
    let ni = r.range(0, 6);
    match r.below(40) {
        0 => {
            let f = *r.pick(UNARY);
            let x = if r.chance(0.5) {
                pos(r, depth - 1)
            } else {
                gx(r, depth - 1)
            };
            n(&format!("{f}({{0}} + {{1}}) - {f}({{0}})"), vec![x, t])
        }
        1 => n("exp({0}) - 1 - {0}", vec![t]),
        2 => n(&format!("sin(pi*({ni} + {{0}}))"), vec![t]),
        3 => n(&format!("cos(pi*({ni} + 1/2 + {{0}}))"), vec![t]),
        4 => n("tan(pi/2*(1 - {0}))", vec![t]),
        5 => n(&format!("gamma(-{ni} + {{0}})"), vec![t]),
        6 => n(&format!("digamma(-{ni} + {{0}})"), vec![t]),
        7 => n("acos(1 - {0})", vec![t]),
        8 => n("asin(1 - {0}) - pi/2", vec![t]),
        9 => n("acosh(1 + {0})", vec![t]),
        10 => n("atanh(1 - {0})", vec![t]),
        11 => n("ln(1 + {0})", vec![t]),
        12 => n("lambertw(-exp(-1) + {0})", vec![t]),
        13 => l(format!(
            "exp({}10^{})",
            if r.chance(0.5) { "-" } else { "" },
            r.range(1, 7)
        )),
        14 => n("sqrt({0}^2 + {1}) - {0}", vec![pos(r, depth - 1), t]),
        15 => n("1 - erf({0})", vec![pos(r, depth - 1)]),
        16 => n("erfc({0}) - 1 + erf({0})", vec![pos(r, depth - 1)]),
        17 => l(format!("1/2 - erf({}*sqrt(2))/2", r.range(1, 20))),
        18 => n("cos({0}) - 1", vec![t]),
        19 => n("sinh({0}) - {0}", vec![t]),
        20 => n("sin({0}) - {0}", vec![t]),
        21 => n("{0}^(1 + {1}) - {0}", vec![pos(r, depth - 1), t]),
        22 => n("(1 + {0})^(1/{0}) - E", vec![t]),
        23 => l(format!(
            "besselj(0, {}/10^16)",
            r.pick(&[
                "2404825557695773",
                "24048255576957727686",
                "5520078110286311",
                "86537279129110122170"
            ])
        )),
        24 => n("zeta(1 + {0})", vec![t]),
        25 => n("zeta(-2 + {0})", vec![t]),
        26 => n("polylog(2, 1 - {0})", vec![t]),
        27 => n("elliptic_k(1 - {0})", vec![t]),
        28 => n("atan2({0}, -1)", vec![t]),
        29 => n("atan2(-{0}, -1)", vec![t]),
        30 => {
            let f = *r.pick(&["floor", "ceiling"]);
            n(&format!("{f}({ni} + {{0}})"), vec![n("-{0}", vec![t])])
        }
        31 => n(
            "digamma({0}) - ln({0})",
            vec![l(format!("10^{}", r.range(1, 40)))],
        ),
        32 => n(
            "loggamma({0}) - ({0} - 1/2)*ln({0}) + {0}",
            vec![l(format!("10^{}", r.range(1, 40)))],
        ),
        33 => n(
            &format!("uppergamma({}, {{0}})", order(r)),
            vec![l(format!("{}", r.range(10, 2000)))],
        ),
        34 => n(
            &format!("polygamma({}, {{0}})", r.range(1, 1800)),
            vec![pos(r, 0)],
        ),
        35 => {
            let m = r.range(3, 7);
            let fs: Vec<&str> = (0..m).map(|_| *r.pick(ELEM)).collect();
            let mut tpl = "{0}".to_string();
            for f in fs {
                tpl = format!("{f}({tpl})");
            }
            n(&tpl, vec![gx(r, depth - 1)])
        }
        36 => n("{0} - {0}*(1 + {1})", vec![gx(r, depth - 1), t]),
        37 => l(format!("sign(exp(10^(-{})) - 1 - 10^(-{}))", k(r), k(r))),
        38 => n(
            "airyai(-{0})",
            vec![l(r
                .pick(&[
                    "2338107410459767/10^15",
                    "2338107410459767038489197/10^24",
                    "4087949444130970/10^15",
                ])
                .to_string())],
        ),
        _ => n(
            "Ci({0})",
            vec![l(r
                .pick(&["6165054856207162/10^16", "61650548562071624/10^17"])
                .to_string())],
        ),
    }
}

fn gx(r: &mut Src, depth: u32) -> T {
    if depth == 0 {
        return leaf(r);
    }
    match r.below(100) {
        0..=9 => leaf(r),
        10..=24 => {
            let op = *r.pick(&["{0} + {1}", "{0} - {1}", "{0}*{1}", "{0}/{1}"]);
            let a = gx(r, depth - 1);
            let b = gx(r, depth - 1);
            n(op, vec![a, b])
        }
        25..=29 => {
            let e = *r.pick(&[
                "2", "3", "-1", "-2", "1/2", "1/3", "-1/2", "7", "2/3", "10", "-5/3",
            ]);
            n(&format!("{{0}}^({e})"), vec![gx(r, depth - 1)])
        }
        30..=31 => n("{0}^{1}", vec![pos(r, depth - 1), gx(r, depth - 1)]),
        32..=46 => {
            let f = *r.pick(UNARY);
            n(&format!("{f}({{0}})"), vec![gx(r, depth - 1)])
        }
        47..=49 => n(&format!("factorial({})", r.range(0, 30)), vec![]),
        50..=64 => special2(r, depth),
        65..=86 => cancel(r, depth),
        87..=90 => sums(r),
        91..=92 => rootof(r),
        _ => complexish(r, depth),
    }
}

fn top(r: &mut Src) -> T {
    let depth = 1 + r.below(3) as u32;
    if r.byte() < 8 {
        let d = depth.saturating_sub(1).max(1);
        return n("PW", vec![gx(r, d), gx(r, d), gx(r, d), gx(r, d)]);
    }
    gx(r, depth)
}

// ── Decimal parsing and comparison ─────────────────────────────────────────

/// `m·10^e`, `m ≠ 0`.
#[derive(Clone, Debug)]
struct Dec {
    m: BigInt,
    e: i64,
}

impl Dec {
    /// Position of the leading digit: `10^lead ≤ |v| < 10^(lead+1)`.
    fn lead(&self) -> i64 {
        self.m.abs().to_string().len() as i64 - 1 + self.e
    }
    fn to_f64(&self) -> f64 {
        format!("{}e{}", self.m, self.e)
            .parse::<f64>()
            .unwrap_or(f64::NAN)
    }
}

/// `None` for an unparsable string, `Some(None)` for zero.
fn parse_dec(s: &str) -> Option<Option<Dec>> {
    let s = s.trim();
    if s == "0" || s == "-0" {
        return Some(None);
    }
    let (mant, exp) = match s.find(['e', 'E']) {
        Some(i) => (&s[..i], s[i + 1..].parse::<i64>().ok()?),
        None => (s, 0),
    };
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    let (ip, fp) = mant.split_once('.').unwrap_or((mant, ""));
    let mut m: BigInt = format!("{ip}{fp}").parse().ok()?;
    if neg {
        m = -m;
    }
    if m.is_zero() {
        return Some(None);
    }
    Some(Some(Dec {
        m,
        e: exp - fp.len() as i64,
    }))
}

#[derive(Clone, Debug)]
struct Cx {
    re: Option<Dec>,
    im: Option<Dec>,
}

fn parse_im(s: &str) -> Option<Option<Dec>> {
    let s = s.trim();
    match s {
        "i" => return parse_dec("1"),
        "-i" => return parse_dec("-1"),
        _ => {}
    }
    parse_dec(s.strip_suffix("*i")?)
}

/// What `eval_decimal` prints: `a`, `b*i`, `i`, `-i`, `a ± b*i`, `a ± i`.
fn parse_cx(s: &str) -> Option<Cx> {
    let b = s.as_bytes();
    for i in 1..b.len().saturating_sub(2) {
        if b[i] == b' ' && (b[i + 1] == b'+' || b[i + 1] == b'-') && b[i + 2] == b' ' {
            let re = parse_dec(&s[..i])?;
            let im = parse_im(&s[i + 3..])?;
            let im = im.map(|d| {
                if b[i + 1] == b'-' {
                    Dec { m: -d.m, e: d.e }
                } else {
                    d
                }
            });
            return Some(Cx { re, im });
        }
    }
    if s.ends_with('i') {
        return Some(Cx {
            re: None,
            im: parse_im(s)?,
        });
    }
    Some(Cx {
        re: parse_dec(s)?,
        im: None,
    })
}

fn pow10(k: i64) -> BigInt {
    BigInt::from(10u32).pow(k.max(0) as u32)
}

/// `|a − b| ≤ 10^t`?
fn close(a: Option<&Dec>, b: Option<&Dec>, t: i64) -> bool {
    let la = a.map_or(i64::MIN / 4, |d| d.lead());
    let lb = b.map_or(i64::MIN / 4, |d| d.lead());
    if la < t - 1 && lb < t - 1 {
        return true;
    }
    if (la - lb).abs() > 3 {
        return la.max(lb) < t;
    }
    let ea = a.map_or(i64::MAX / 4, |d| d.e);
    let eb = b.map_or(i64::MAX / 4, |d| d.e);
    let emin = ea.min(eb).min(t);
    if ea.max(eb).max(t).min(i64::MAX / 8) - emin > 20000 && a.is_some() && b.is_some() {
        return false;
    }
    let sc = |d: Option<&Dec>| d.map_or(BigInt::zero(), |d| &d.m * pow10(d.e - emin));
    let diff = (sc(a) - sc(b)).abs();
    diff <= pow10(t - emin)
}

/// Unit of the last of `digits` significant digits of `d`.
fn ulp_exp(d: &Dec, digits: u32) -> i64 {
    d.lead() - i64::from(digits) + 1
}

fn cx_lead(c: &Cx) -> Option<i64> {
    match (c.re.as_ref().map(Dec::lead), c.im.as_ref().map(Dec::lead)) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

/// Compare a certified `digits`-digit value with the reference: `None` when
/// consistent, else the class suffix (`""` wrong digits, `"part"` a part
/// wrong beyond its own digits but within the joint ulp, `"zero"` printed
/// `0` for a nonzero value).
fn compare_dec(got: &Cx, reference: &Cx, digits: u32) -> Option<&'static str> {
    let Some(glead) = cx_lead(got) else {
        return cx_lead(reference).map(|_| "zero");
    };
    let joint_ulp = glead - i64::from(digits) + 1;
    let mut part = false;
    for (g, rf) in [(&got.re, &reference.re), (&got.im, &reference.im)] {
        let t = match g {
            Some(d) => ulp_exp(d, digits),
            None => joint_ulp,
        };
        if !close(g.as_ref(), rf.as_ref(), t) {
            if g.is_some() && close(g.as_ref(), rf.as_ref(), joint_ulp) {
                part = true;
            } else {
                return Some("");
            }
        }
    }
    part.then_some("part")
}

fn ulps(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let key = |x: f64| {
        let i = x.to_bits() as i64;
        if i < 0 { i64::MIN - i } else { i }
    };
    (i128::from(key(a)) - i128::from(key(b)))
        .unsigned_abs()
        .min(u128::from(u64::MAX)) as u64
}

/// An `f64` part `x` against the reference part `rf` (`other_ref` the
/// reference's other part): `None` within one ulp.
fn f64_ok(x: f64, rf: f64, other_ref: f64) -> Option<&'static str> {
    if ulps(x, rf) <= 1 {
        return None;
    }
    if x == 0.0 {
        // an omitted part must be negligible next to the other part
        if rf.abs() <= other_ref.abs() * 2f64.powi(-55) {
            return None;
        }
        return Some("zero");
    }
    if (x - rf).abs() <= other_ref.abs().max(rf.abs()) * 2f64.powi(-50) {
        return Some("part");
    }
    Some("")
}

// ── Checking one expression ─────────────────────────────────────────────────

fn is_refusal(e: &SymplexError) -> bool {
    matches!(
        e,
        SymplexError::PrecisionExhausted { .. }
            | SymplexError::Unevaluable { .. }
            | SymplexError::NotImplemented(_)
            | SymplexError::Divergent { .. }
    )
}

fn tag(class: &str, cls: &str) -> String {
    if cls.is_empty() {
        class.to_string()
    } else {
        format!("{class}-{cls}")
    }
}

/// The findings `(class, detail)` for `ex`.
fn check(ex: &Ex) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let r60 = ex.eval_decimal(60);
    let r16 = ex.eval_decimal(16);
    let r30 = ex.eval_decimal(30);
    let f = ex.eval_f64();
    let c = ex.eval_complex64();
    let values =
        format!("d16 = {r16:?}\n  d30 = {r30:?}\n  d60 = {r60:?}\n  f64 = {f:?}\n  c64 = {c:?}");
    // A non-refusal error at one precision while another evaluates.
    let any_ok = r16.is_ok() || r30.is_ok() || r60.is_ok();
    for (name, r) in [("d16", &r16), ("d30", &r30), ("d60", &r60)] {
        if let Err(e) = r
            && !is_refusal(e)
            && any_ok
        {
            out.push((format!("err-{name}"), values.clone()));
        }
    }
    // Reference: 60 digits, else 30 (for the 16-digit checks only).
    let (ref_str, ref_digits) = match (&r60, &r30) {
        (Ok(s), _) => (s.clone(), 60),
        (_, Ok(s)) => (s.clone(), 30),
        _ => return out,
    };
    let Some(reference) = parse_cx(&ref_str) else {
        out.push(("unparsed-ref".into(), values));
        return out;
    };
    let ref_is_zero = cx_lead(&reference).is_none();
    for (name, digits, r) in [("d16", 16, &r16), ("d30", 30, &r30)] {
        if digits >= ref_digits {
            continue;
        }
        let Ok(s) = r else { continue };
        let Some(g) = parse_cx(s) else {
            out.push((format!("unparsed-{name}"), values.clone()));
            continue;
        };
        if ref_is_zero {
            if cx_lead(&g).is_some() {
                out.push((format!("refzero-{name}"), values.clone()));
            }
            continue;
        }
        if let Some(cls) = compare_dec(&g, &reference, digits) {
            out.push((tag(name, cls), values.clone()));
        }
    }
    let rre = reference.re.as_ref().map_or(0.0, Dec::to_f64);
    let rim = reference.im.as_ref().map_or(0.0, Dec::to_f64);
    if ref_is_zero {
        if let Ok(x) = &f
            && *x != 0.0
        {
            out.push(("refzero-f64".into(), values.clone()));
        }
        return out;
    }
    if let Ok(x) = &f {
        // eval_f64 claims a real value: the imaginary part must be
        // negligible (below the 16th digit of the real part).
        let im_big = reference.im.is_some()
            && reference.re.as_ref().is_none_or(|re| {
                reference
                    .im
                    .as_ref()
                    .is_some_and(|im| im.lead() > re.lead() - 16)
            });
        if im_big {
            out.push(("f64-complex".into(), values.clone()));
        } else if let Some(cls) = f64_ok(*x, rre, 0.0) {
            out.push((tag("f64", cls), values.clone()));
        }
    }
    if let Ok(z) = &c {
        let a = f64_ok(z.re, rre, rim);
        let b = f64_ok(z.im, rim, rre);
        if let Some(cls) = a.or(b) {
            let cls = if a == Some("") || b == Some("") {
                ""
            } else {
                cls
            };
            out.push((tag("c64", cls), values));
        }
    }
    out
}

fuzz_target!(|data: &[u8]| {
    let mut r = Src::new(data);
    let t = top(&mut r);
    let text = render(&t);
    let show = std::env::var_os("FUZZ_SHOW").is_some();
    if show {
        eprintln!("fuzz_evalf: {text}");
    }
    let ctx = Context::new();
    let Ok(ex) = build(&ctx, &t) else {
        if show {
            eprintln!("  verdict: skip (parse)");
        }
        return;
    };
    if let Some((class, detail)) = check(&ex).into_iter().next() {
        panic!("fuzz_evalf: {class} for {text}\n  (parsed as {ex})\n  {detail}");
    }
    if show {
        let compared = ex.eval_decimal(30).is_ok();
        eprintln!(
            "  verdict: {}",
            if compared {
                "ok"
            } else {
                "skip (no reference)"
            }
        );
    }
});
