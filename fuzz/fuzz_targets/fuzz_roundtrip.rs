//! `Display` ⇄ `parse` round trip of API-built expressions.
//!
//! An expression built through the API — the elementary and special
//! functions, `Piecewise` with relational and Boolean conditions,
//! `Derivative`, definite and indefinite `Integral`, `Sum`/`Product`,
//! `Limit`, `Subs` of a derivative of an undefined function, `RootOf`,
//! `min`/`max`, huge and tiny rationals, decimals, `I`, `oo`, `-oo`, `zoo`,
//! `nan` — must display as text that `Context::parse` reads back to the
//! **same tree** (relations: `parse_bool`), and `to_latex`, `pretty`,
//! `pretty_ascii`, `to_srepr` and `to_mathml` must not panic.  The 0.30
//! hunter found the parser's pairwise folding of product chains
//! (`-3*(x + 1)*y` re-read as `(-3*x - 3)*y`), the missing `Piecewise`
//! and `H(x)`, and `(-12/7)!`, `(-oo)^r`, `(-16/7)^(-1/2)` printed without
//! the parentheses (or with a shorthand) their re-parse needs.
//!
//! Inputs decode through [`choose::Src`]; set `FUZZ_SHOW=1` to print each
//! expression.
#![no_main]

#[path = "common/choose.rs"]
mod choose;

use choose::Src;
use libfuzzer_sys::fuzz_target;
use symplex::prelude::*;

struct G<'a, 'b> {
    ctx: &'a Context,
    r: Src<'b>,
    x: Ex,
    y: Ex,
    z: Ex,
    k: Ex,
    /// `FUZZ_TRACE=1`: print every subexpression as it is built.
    trace: bool,
}

fn digits(r: &mut Src, n: usize) -> String {
    let mut s = String::new();
    s.push(char::from(b'1' + r.below(9) as u8));
    for _ in 1..n {
        s.push(char::from(b'0' + r.below(10) as u8));
    }
    s
}

type Unary = fn(&Ex) -> Ex;

const UNARY: &[Unary] = &[
    |a| a.sin(),
    |a| a.cos(),
    |a| a.tan(),
    |a| a.exp(),
    |a| a.ln(),
    |a| a.abs(),
    |a| a.asin(),
    |a| a.acos(),
    |a| a.atan(),
    |a| a.sinh(),
    |a| a.cosh(),
    |a| a.tanh(),
    |a| a.asinh(),
    |a| a.acosh(),
    |a| a.atanh(),
    |a| a.sec(),
    |a| a.csc(),
    |a| a.cot(),
    |a| a.acot(),
    |a| a.coth(),
    |a| a.sech(),
    |a| a.csch(),
    |a| a.sign(),
    |a| a.floor(),
    |a| a.ceiling(),
    |a| a.gamma(),
    |a| a.log_gamma(),
    |a| a.digamma(),
    |a| a.erf(),
    |a| a.erfc(),
    |a| a.erfi(),
    |a| a.erfinv(),
    |a| a.erfcinv(),
    |a| a.lambertw(),
    |a| a.heaviside(),
    |a| a.dirac_delta(),
    |a| a.e1(),
    |a| a.shi(),
    |a| a.chi(),
    |a| a.fresnels(),
    |a| a.fresnelc(),
    |a| a.airyai(),
    |a| a.airybi(),
    |a| a.airyaiprime(),
    |a| a.airybiprime(),
    |a| a.elliptic_k(),
    |a| a.elliptic_e(),
    |a| a.dirichlet_eta(),
    |a| a.asec(),
    |a| a.acsc(),
    |a| a.acoth(),
    |a| a.asech(),
    |a| a.acsch(),
    |a| a.sinc(),
    |a| a.frac(),
    |a| a.re(),
    |a| a.im(),
    |a| a.conjugate(),
    |a| a.arg(),
    |a| a.zeta(),
    |a| a.factorial(),
    |a| a.fibonacci(),
    |a| a.harmonic(),
    |a| a.factorial2(),
    |a| a.subfactorial(),
    |a| a.catalan_number(),
    |a| a.bell(),
];

type Binary = fn(&Ex, &Ex) -> Ex;

const BINARY: &[Binary] = &[
    |a, n| a.bessel_j(n),
    |a, n| a.bessel_y(n),
    |a, n| a.bessel_i(n),
    |a, n| a.bessel_k(n),
    |a, n| a.expint(n),
    |a, n| a.lowergamma(n),
    |a, n| a.uppergamma(n),
    |a, n| a.polylog(n),
    |a, n| a.elliptic_f(n),
    |a, n| a.elliptic_pi(n),
    |a, n| a.legendre(n),
    |a, n| a.chebyshev_t(n),
    |a, n| a.hermite(n),
    |a, n| a.laguerre(n),
    |a, n| a.beta(n),
    |a, n| a.binomial(n),
    |a, n| a.log(n),
    |a, n| a.atan2(n),
    |a, n| a.rising_factorial(n),
    |a, n| a.falling_factorial(n),
    |a, n| a.polygamma(n),
    |a, n| a.kronecker_delta(n),
    |a, n| a.min_with(n),
    |a, n| a.max_with(n),
];

impl G<'_, '_> {
    fn parsed(&self, s: &str, fallback: &Ex) -> Ex {
        self.ctx.parse(s).unwrap_or_else(|_| fallback.clone())
    }

    fn leaf(&mut self) -> Ex {
        match self.r.below(21) {
            0..=4 => {
                let i = self.r.below(3) as usize;
                [&self.x, &self.y, &self.z][i].clone()
            }
            5..=7 => self.ctx.int(self.r.range(-9, 9)),
            8 => {
                let n = self.r.range(15, 60) as usize;
                let neg = if self.r.chance(0.5) { "-" } else { "" };
                let s = format!("{neg}{}", digits(&mut self.r, n));
                self.parsed(&s, &self.x.clone())
            }
            9 | 10 => {
                let q = self.r.range(2, 12);
                self.ctx.rational(self.r.range(-30, 30), q)
            }
            11 => {
                let (a, b) = (self.r.range(1, 40) as usize, self.r.range(1, 40) as usize);
                let neg = if self.r.chance(0.5) { "-" } else { "" };
                let s = format!("{neg}{}/{}", digits(&mut self.r, a), digits(&mut self.r, b));
                self.parsed(&s, &self.x.clone())
            }
            12 => {
                let (c, kk) = (self.r.range(1, 99), self.r.range(10, 80));
                self.parsed(&format!("{c}*10^(-{kk})"), &self.x.clone())
            }
            13 => match self.r.below(6) {
                0 => self.ctx.pi(),
                1 => self.ctx.e(),
                2 => self.ctx.i_unit(),
                3 => self.ctx.euler_gamma(),
                4 => self.ctx.catalan(),
                _ => self.ctx.golden_ratio(),
            },
            14 => match self.r.below(4) {
                0 => self.ctx.infinity(),
                1 => self.ctx.neg_infinity(),
                2 => self.ctx.complex_infinity(),
                _ => self.ctx.nan(),
            },
            15 => {
                let q = self.r.range(1, 9);
                self.ctx.rational(self.r.range(-9, 9), q) * self.ctx.i_unit()
            }
            16 => {
                let ip = self.r.range(-99, 99);
                let nd = self.r.range(1, 20) as usize;
                let s = format!("{ip}.{}", digits(&mut self.r, nd));
                self.parsed(&s, &self.x.clone())
            }
            17 => {
                if self.r.chance(0.5) {
                    self.ctx
                        .apply("f", &[&self.x])
                        .unwrap_or_else(|_| self.x.clone())
                } else {
                    self.ctx
                        .apply("g", &[&self.x, &self.y])
                        .unwrap_or_else(|_| self.y.clone())
                }
            }
            18 => {
                let names = [
                    "x_1", "alpha", "Beta", "theta2", "t", "s", "n", "omega", "xi",
                ];
                let i = self.r.below(names.len() as u64) as usize;
                self.ctx.symbol(names[i])
            }
            19 => {
                let s = match self.r.below(8) {
                    0 => format!(
                        "(-{}/{})^({}/{})",
                        self.r.range(1, 9),
                        self.r.range(2, 9),
                        self.r.range(-9, 9),
                        self.r.range(2, 9)
                    ),
                    1 => format!("{}^(1/{})", -self.r.range(2, 40), self.r.range(2, 7)),
                    2 => "x^(10^30)".into(),
                    3 => "I*x^I".into(),
                    4 => format!("x^({})", self.r.range(-20, 20)),
                    5 => "(x*y)^(-3/2)".into(),
                    6 => "Series(sin(x), x, 0, 5)".into(),
                    _ => "ConditionSet(x, x > 1)".into(),
                };
                self.parsed(&s, &self.x.clone())
            }
            _ => {
                let i = self.r.below(2) as usize;
                [&self.x, &self.y][i].clone()
            }
        }
    }

    fn cond(&mut self, depth: u32) -> BoolEx {
        let a = self.e(depth.saturating_sub(1));
        let b = self.e(depth.saturating_sub(1));
        let base = match self.r.below(6) {
            0 => a.gt(&b),
            1 => a.ge(&b),
            2 => a.lt(&b),
            3 => a.le(&b),
            4 => a.eq_expr(&b),
            _ => a.ne_expr(&b),
        };
        match self.r.below(8) {
            0 => base.and(&self.x.gt(&self.ctx.int(0))),
            1 => base.or(&self.y.lt(&self.ctx.int(1))),
            2 => base.not(),
            _ => base,
        }
    }

    fn e(&mut self, depth: u32) -> Ex {
        let v = self.e_inner(depth);
        if self.trace {
            eprintln!("  built: {v}");
        }
        v
    }

    fn e_inner(&mut self, depth: u32) -> Ex {
        if depth == 0 || self.r.chance(0.15) {
            return self.leaf();
        }
        let d = depth - 1;
        match self.r.below(100) {
            0..=9 => self.e(d) + self.e(d),
            10..=13 => self.e(d) - self.e(d),
            14..=21 => self.e(d) * self.e(d),
            22..=26 => self.e(d) / self.e(d),
            27..=31 => {
                let b = self.e(d);
                match self.r.below(5) {
                    0 => b.powi(self.r.range(-4, 5)),
                    1 => {
                        let q = self.r.range(2, 7);
                        b.pow(&self.ctx.rational(self.r.range(-7, 7), q))
                    }
                    2 => {
                        let ex = self.e(d);
                        b.pow(&ex)
                    }
                    3 => b.sqrt(),
                    _ => b.cbrt(),
                }
            }
            32..=59 => {
                let a = self.e(d);
                let f = UNARY[self.r.below(UNARY.len() as u64) as usize];
                f(&a)
            }
            60..=65 => {
                let a = self.e(d);
                let n = if self.r.chance(0.5) {
                    self.ctx.int(self.r.range(0, 4))
                } else {
                    self.e(d)
                };
                let f = BINARY[self.r.below(BINARY.len() as u64) as usize];
                f(&a, &n)
            }
            66..=71 => {
                let n = self.r.range(1, 3);
                let mut vals = Vec::new();
                let mut conds = Vec::new();
                for _ in 0..n {
                    vals.push(self.e(d));
                    conds.push(self.cond(d));
                }
                vals.push(self.e(d));
                conds.push(if self.r.chance(0.8) {
                    self.ctx.bool_true()
                } else {
                    self.cond(d)
                });
                let pairs: Vec<(&Ex, &BoolEx)> = vals.iter().zip(conds.iter()).collect();
                Ex::piecewise(&pairs)
            }
            72..=74 => {
                let a = self.e(d);
                let v = if self.r.chance(0.5) {
                    self.x.clone()
                } else {
                    self.y.clone()
                };
                let da = a.formal_diff(&v);
                if self.r.chance(0.3) {
                    da.formal_diff(&self.x)
                } else {
                    da
                }
            }
            75..=77 => {
                let a = self.e(d);
                let lo = self.e(d.min(1));
                let hi = self.e(d.min(1));
                let v = if self.r.chance(0.7) {
                    self.x.clone()
                } else {
                    self.y.clone()
                };
                a.definite_integral_node(&v, &lo, &hi)
            }
            78..=80 => {
                let a = self.e(d);
                let s = if self.r.chance(0.5) {
                    format!("Integral({a}, x)")
                } else {
                    let pt = self.e(d.min(1));
                    format!("Limit({a}, x, {pt})")
                };
                self.parsed(&s, &a)
            }
            81..=85 => {
                let ex = self.e(d.min(1));
                let body = self.e(d) * self.k.pow(&ex);
                let lo = self.ctx.int(self.r.range(-2, 3));
                let hi = match self.r.below(3) {
                    0 => self.ctx.infinity(),
                    1 => self.ctx.symbol("n"),
                    _ => self.ctx.int(self.r.range(3, 9)),
                };
                if self.r.chance(0.7) {
                    Ex::symbolic_sum(&body, &self.k, &lo, &hi)
                } else {
                    Ex::symbolic_product(&body, &self.k, &lo, &hi)
                }
            }
            86..=89 => {
                let Ok(f) = self.ctx.apply("f", &[&self.x]) else {
                    return self.leaf();
                };
                let df = if self.r.chance(0.5) {
                    f.diff(&self.x)
                } else {
                    f.diff(&self.x).diff(&self.x)
                };
                let pt = self.e(d.min(2));
                df.subs(&self.x, &pt)
            }
            90..=93 => {
                let deg = self.r.range(2, 5);
                let mut s = format!("x^{deg}");
                for pw in (0..deg).rev() {
                    let c = self.r.range(-5, 5);
                    if c != 0 {
                        s += &format!(" + ({c})*x^{pw}");
                    }
                }
                let idx = self.r.range(0, deg - 1);
                let fallback = self.leaf();
                self.parsed(&format!("RootOf({s}, x, {idx})"), &fallback)
            }
            _ => {
                let a = self.e(d);
                let b = self.e(d);
                let c = self.e(d);
                match self.r.below(4) {
                    0 => a.gegenbauer(&b, &c),
                    1 => a.assoc_legendre(&b, &c),
                    2 => a.assoc_laguerre(&b, &c),
                    _ => a.betainc_regularized(&b, &c, &self.ctx.int(1)),
                }
            }
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let ctx = Context::new();
    let mut g = G {
        ctx: &ctx,
        r: Src::new(data),
        x: ctx.symbol("x"),
        y: ctx.symbol("y"),
        z: ctx.symbol("z"),
        k: ctx.symbol("k"),
        trace: std::env::var_os("FUZZ_TRACE").is_some(),
    };
    let depth = 1 + g.r.below(4) as u32;
    let show = std::env::var_os("FUZZ_SHOW").is_some();
    if g.r.byte() < 26 {
        let b = g.cond(depth);
        let s = b.to_string();
        if show {
            eprintln!("fuzz_roundtrip (relation): {s}\n  tree: {:?}", b.to_tree());
        }
        match ctx.parse_bool(&s) {
            Ok(back) => assert!(
                back == b,
                "fuzz_roundtrip: {s} re-parses as {back}, a different relation"
            ),
            Err(err) => {
                panic!("fuzz_roundtrip: the display of a relation does not parse: {s}\n  {err}")
            }
        }
        let _ = b.to_latex();
        return;
    }
    let e = g.e(depth);
    let s = e.to_string();
    if show {
        eprintln!("fuzz_roundtrip: {s}");
    }
    match ctx.parse(&s) {
        Ok(back) => assert!(
            back == e,
            "fuzz_roundtrip: {s} re-parses to a different tree\n  want {}\n  got  {}",
            e.to_srepr(),
            back.to_srepr()
        ),
        Err(err) => panic!(
            "fuzz_roundtrip: the display does not parse: {s}\n  {err}\n  tree {}",
            e.to_srepr()
        ),
    }
    let _ = e.to_latex();
    let _ = e.pretty();
    let _ = e.pretty_ascii();
    let _ = e.to_mathml();
});
