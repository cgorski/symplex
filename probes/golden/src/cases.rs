//! The deterministic case generator: case id → expression string.
//!
//! Everything here is `std`-only and independent of the symplex tree under
//! test (its own SplitMix64, its own frozen function table), so the same
//! case id yields the same string whichever tree the harness is built
//! against.  The function table is checked against the tree's
//! `LibFn::ALL` by `golden --list-fns`; a function the tree adds is reported
//! there, it does not reshuffle the corpus.
//!
//! Bump [`GEN_VERSION`] on any change that alters the generated strings:
//! TSVs from different generator versions must not be diffed.

use std::collections::BTreeMap;

pub const GEN_VERSION: u32 = 1;

/// SplitMix64 (Steele, Lea & Flood 2014), seeded per case.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform-ish in `0..n` (modulo bias is irrelevant here).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// Inclusive range.
    pub fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }

    pub fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }

    pub fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.below(xs.len() as u64) as usize]
    }
}

fn seed(id: u64) -> u64 {
    id.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ 0x676F_6C64_656E_0000 ^ u64::from(GEN_VERSION)
}

/// Kinds of function argument.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    /// A general subexpression.
    G,
    /// A small integer-like argument (integer sequences, binomial `k`, …).
    S,
    /// An order / index (Bessel, orthogonal polynomials, polygamma, …).
    O,
    /// A Lambert-W branch, −1, 0 or 1.
    K,
}

use Arg::{G, K, O, S};

pub struct Fun {
    /// Spelling used in the generated text (the parser is case-insensitive
    /// for function names).
    pub name: &'static str,
    pub args: &'static [Arg],
    /// For library functions: the `(min, max)` arity `LibFn::arity()` had
    /// at 0.48.0, checked by `--list-fns`.
    pub lib: Option<(u8, u8)>,
}

const fn d(name: &'static str, args: &'static [Arg]) -> Fun {
    Fun { name, args, lib: None }
}

const fn l(name: &'static str, min: u8, max: u8, args: &'static [Arg]) -> Fun {
    Fun { name, args, lib: Some((min, max)) }
}

/// Every function form with a dedicated arm in the 0.48.0 parser.
pub const DEDICATED: &[Fun] = &[
    d("sin", &[G]),
    d("cos", &[G]),
    d("tan", &[G]),
    d("sec", &[G]),
    d("csc", &[G]),
    d("cot", &[G]),
    d("sinh", &[G]),
    d("cosh", &[G]),
    d("tanh", &[G]),
    d("sech", &[G]),
    d("csch", &[G]),
    d("coth", &[G]),
    d("asin", &[G]),
    d("acos", &[G]),
    d("atan", &[G]),
    d("acot", &[G]),
    d("asec", &[G]),
    d("acsc", &[G]),
    d("asinh", &[G]),
    d("acosh", &[G]),
    d("atanh", &[G]),
    d("acoth", &[G]),
    d("asech", &[G]),
    d("acsch", &[G]),
    d("exp", &[G]),
    d("log", &[G]),
    d("ln", &[G]),
    d("log", &[G, S]),
    d("sqrt", &[G]),
    d("cbrt", &[G]),
    d("abs", &[G]),
    d("sign", &[G]),
    d("floor", &[G]),
    d("ceiling", &[G]),
    d("re", &[G]),
    d("im", &[G]),
    d("conjugate", &[G]),
    d("arg", &[G]),
    d("gamma", &[G]),
    d("loggamma", &[G]),
    d("digamma", &[G]),
    d("polygamma", &[O, G]),
    d("erf", &[G]),
    d("erfc", &[G]),
    d("lambertw", &[G]),
    d("beta", &[G, G]),
    d("Si", &[G]),
    d("Ci", &[G]),
    d("Ei", &[G]),
    d("li", &[G]),
    d("E1", &[G]),
    d("zeta", &[G]),
    d("factorial", &[G]),
    d("binomial", &[G, S]),
    d("Heaviside", &[G]),
    d("DiracDelta", &[G]),
    d("KroneckerDelta", &[S, S]),
    d("Min", &[G, G]),
    d("Max", &[G, G]),
    d("Min", &[G, G, G]),
    d("Max", &[G, G, G]),
    d("atan2", &[G, G]),
];

/// `symplex::base::libfn::LibFn::ALL` at 0.48.0, in its order, with the
/// arity `LibFn::arity()` reported.
pub const LIBFNS: &[Fun] = &[
    l("factorial2", 1, 1, &[S]),
    l("subfactorial", 1, 1, &[S]),
    l("rising_factorial", 2, 2, &[G, S]),
    l("falling_factorial", 2, 2, &[G, S]),
    l("fibonacci", 1, 1, &[S]),
    l("lucas", 1, 1, &[S]),
    l("bernoulli", 1, 1, &[S]),
    l("harmonic", 1, 1, &[S]),
    l("catalan", 1, 1, &[S]),
    l("bell", 1, 1, &[S]),
    l("euler_number", 1, 1, &[S]),
    l("stirling1", 2, 2, &[S, S]),
    l("stirling2", 2, 2, &[S, S]),
    l("partition_count", 1, 1, &[S]),
    l("lambertw", 1, 2, &[G, K]),
    l("besselj", 2, 2, &[O, G]),
    l("bessely", 2, 2, &[O, G]),
    l("besseli", 2, 2, &[O, G]),
    l("besselk", 2, 2, &[O, G]),
    l("legendre", 2, 2, &[O, G]),
    l("chebyshev_t", 2, 2, &[O, G]),
    l("chebyshev_u", 2, 2, &[O, G]),
    l("hermite", 2, 2, &[O, G]),
    l("laguerre", 2, 2, &[O, G]),
    l("erfi", 1, 1, &[G]),
    l("erfinv", 1, 1, &[G]),
    l("erfcinv", 1, 1, &[G]),
    l("expint", 2, 2, &[O, G]),
    l("Shi", 1, 1, &[G]),
    l("Chi", 1, 1, &[G]),
    l("fresnels", 1, 1, &[G]),
    l("fresnelc", 1, 1, &[G]),
    l("lowergamma", 2, 2, &[O, G]),
    l("uppergamma", 2, 2, &[O, G]),
    l("polylog", 2, 2, &[O, G]),
    l("dirichlet_eta", 1, 1, &[G]),
    l("airyai", 1, 1, &[G]),
    l("airybi", 1, 1, &[G]),
    l("airyaiprime", 1, 1, &[G]),
    l("airybiprime", 1, 1, &[G]),
    l("elliptic_k", 1, 1, &[G]),
    l("elliptic_e", 1, 1, &[G]),
    l("elliptic_f", 2, 2, &[G, G]),
    l("elliptic_pi", 2, 2, &[G, G]),
    l("gegenbauer", 3, 3, &[O, O, G]),
    l("jacobi", 4, 4, &[O, O, O, G]),
    l("assoc_legendre", 3, 3, &[O, O, G]),
    l("assoc_laguerre", 3, 3, &[O, O, G]),
    l("betainc", 4, 4, &[O, O, G, G]),
    l("betainc_regularized", 4, 4, &[O, O, G, G]),
];

pub fn n_funs() -> usize {
    DEDICATED.len() + LIBFNS.len()
}

pub fn fun(k: usize) -> &'static Fun {
    if k < DEDICATED.len() {
        &DEDICATED[k]
    } else {
        &LIBFNS[k - DEDICATED.len()]
    }
}

/// The expression string of case `id`.
///
/// Families by `id % 20`: 0–5 random tree in x, y (depth ≤ 3); 6–7
/// symbol-free random tree; 8–10 polynomial / rational; 11–12 trigonometric
/// identities; 13–14 equations (`lhs - (rhs)`); 15–19 one function from
/// the table applied to shallow arguments, cycling through the whole table
/// every `20 · n_funs() / 5` cases.
pub fn case(id: u64) -> String {
    let mut g = Gen {
        r: SplitMix64::new(seed(id)),
    };
    match id % 20 {
        0..=5 => g.tree(0, 3, true, true),
        6..=7 => g.tree(0, 3, false, true),
        8..=10 => g.poly_case(),
        11..=12 => g.trig_case(),
        13..=14 => g.equation_case(),
        r => {
            let k = (id / 20) * 5 + (r - 15);
            g.coverage_case((k % n_funs() as u64) as usize)
        }
    }
}

fn num(n: i64) -> String {
    if n < 0 { format!("({n})") } else { n.to_string() }
}

struct Gen {
    r: SplitMix64,
}

impl Gen {
    fn leaf(&mut self, sym: bool) -> String {
        if sym {
            let k = self.r.below(100);
            if k < 40 {
                return "x".into();
            }
            if k < 52 {
                return "y".into();
            }
        }
        match self.r.below(100) {
            0..=39 => num(self.r.int(-3, 5)),
            40..=54 => self.rational(),
            55..=89 => self.constant(),
            _ => "sqrt(2)".into(),
        }
    }

    fn rational(&mut self) -> String {
        let mut p = self.r.int(-5, 4);
        if p >= 0 {
            p += 1;
        }
        let q = self.r.int(2, 7);
        format!("({p}/{q})")
    }

    fn constant(&mut self) -> String {
        const W: &[(u64, &str)] = &[
            (5, "pi"),
            (4, "E"),
            (3, "I"),
            (1, "oo"),
            (2, "EulerGamma"),
            (2, "GoldenRatio"),
            (2, "Catalan"),
        ];
        let total: u64 = W.iter().map(|w| w.0).sum();
        let mut k = self.r.below(total);
        for (w, s) in W {
            if k < *w {
                return (*s).into();
            }
            k -= w;
        }
        "pi".into()
    }

    /// A random tree; `depth` is the current level, leaves at `max`.
    fn tree(&mut self, depth: u32, max: u32, sym: bool, pow: bool) -> String {
        if depth >= max {
            return self.leaf(sym);
        }
        let stop = [8, 25, 45, 60][depth.min(3) as usize];
        if self.r.below(100) < stop {
            return self.leaf(sym);
        }
        if self.r.chance(50) {
            let k = self.r.below(if pow { 12 } else { 10 });
            if k >= 10 {
                let base = self.tree(depth + 1, max, sym, true);
                let e = self.exponent(depth + 1, max, sym);
                return format!("({base}^{e})");
            }
            let a = self.tree(depth + 1, max, sym, pow);
            let b = self.tree(depth + 1, max, sym, pow);
            let op = match k {
                0..=2 => "+",
                3..=4 => "-",
                5..=7 => "*",
                _ => "/",
            };
            format!("({a} {op} {b})")
        } else {
            let f = fun(self.r.below(n_funs() as u64) as usize);
            self.call(f, depth + 1, max, sym)
        }
    }

    fn exponent(&mut self, depth: u32, max: u32, sym: bool) -> String {
        match self.r.below(100) {
            0..=49 => num(self.r.int(-3, 4)),
            50..=74 => self
                .r
                .pick(&["(1/2)", "(1/3)", "(-1/2)", "(3/2)", "(2/3)", "(-1/3)"])
                .into(),
            75..=84 if sym => self.r.pick(&["x", "y"]).into(),
            // No powers inside exponents: keeps towers like 5^4^4 out.
            _ => self.tree(depth, max.min(depth + 2), sym, false),
        }
    }

    fn call(&mut self, f: &Fun, depth: u32, max: u32, sym: bool) -> String {
        let args: Vec<String> = f.args.iter().map(|a| self.arg(*a, depth, max, sym)).collect();
        format!("{}({})", f.name, args.join(", "))
    }

    fn arg(&mut self, a: Arg, depth: u32, max: u32, sym: bool) -> String {
        match a {
            G => self.tree(depth, max, sym, true),
            S => self.small(sym),
            O => self.order(sym),
            K => self.r.pick(&["(-1)", "0", "1"]).into(),
        }
    }

    fn small(&mut self, sym: bool) -> String {
        match self.r.below(100) {
            60..=69 if sym => "x".into(),
            70..=74 if sym => "y".into(),
            75..=84 if sym => "(x + 1)".into(),
            85..=89 => "(1/2)".into(),
            90..=94 => "(-1)".into(),
            95..=99 => "(3/2)".into(),
            _ => self.r.int(0, 8).to_string(),
        }
    }

    fn order(&mut self, sym: bool) -> String {
        match self.r.below(100) {
            70..=84 => self.r.pick(&["(1/2)", "(3/2)", "(-1/2)"]).into(),
            85..=92 => "(-1)".into(),
            93..=99 if sym => "y".into(),
            _ => self.r.int(0, 3).to_string(),
        }
    }

    fn coverage_case(&mut self, k: usize) -> String {
        let f = fun(k);
        let sym = self.r.chance(75);
        let call = self.call(f, 1, 2, sym);
        match self.r.below(100) {
            0..=59 => call,
            60..=69 => format!("({call} + x)"),
            70..=79 => format!("(x*{call})"),
            80..=89 => format!("({call}^2)"),
            _ => format!("({call}/x)"),
        }
    }

    // ── polynomial / rational ─────────────────────────────────────────

    fn factor(&mut self, param: bool) -> P {
        let k = self.r.below(100);
        if k >= 75 && param {
            // x - y, x + y, x + 2y, xy + 1, x² + y, xy - 1
            let terms: Vec<((u32, u32), i64)> = match self.r.below(6) {
                0 => vec![((1, 0), 1), ((0, 1), -1)],
                1 => vec![((1, 0), 1), ((0, 1), 1)],
                2 => vec![((1, 0), 1), ((0, 1), 2)],
                3 => vec![((1, 1), 1), ((0, 0), 1)],
                4 => vec![((2, 0), 1), ((0, 1), 1)],
                _ => vec![((1, 1), 1), ((0, 0), -1)],
            };
            return terms.into_iter().collect();
        }
        if (55..75).contains(&k) {
            let terms: Vec<((u32, u32), i64)> = match self.r.below(5) {
                0 => vec![((2, 0), 1), ((0, 0), self.r.int(1, 5))],
                1 => vec![((2, 0), 1), ((1, 0), 1), ((0, 0), 1)],
                2 => vec![((2, 0), 1), ((0, 0), -2)],
                3 => vec![((2, 0), 1), ((0, 0), -3)],
                _ => vec![((2, 0), 1), ((1, 0), -1), ((0, 0), 2)],
            };
            return terms.into_iter().collect();
        }
        let a = self.r.int(1, 3);
        let b = self.r.int(-4, 4);
        let mut p = P::new();
        p.insert((1, 0), a);
        if b != 0 {
            p.insert((0, 0), b);
        }
        p
    }

    fn factors(&mut self, param: bool, lo: i64, hi: i64) -> Vec<(P, u32)> {
        let n = self.r.int(lo, hi);
        (0..n)
            .map(|_| {
                let f = self.factor(param);
                let m = match self.r.below(100) {
                    0..=69 => 1,
                    70..=94 => 2,
                    _ => 3,
                };
                (f, m)
            })
            .collect()
    }

    fn poly_or_factored(&mut self, fs: &[(P, u32)]) -> String {
        if self.r.chance(50) { p_render(&p_product(fs)) } else { render_factored(fs) }
    }

    fn poly_case(&mut self) -> String {
        let param = self.r.chance(30);
        match self.r.below(100) {
            0..=24 => {
                let fs = self.factors(param, 1, 4);
                p_render(&p_product(&fs))
            }
            25..=39 => {
                let fs = self.factors(param, 2, 4);
                render_factored(&fs)
            }
            40..=74 => {
                let mut num = self.factors(param, 1, 3);
                let mut den = self.factors(param, 1, 3);
                if self.r.chance(50) {
                    let shared = self.factor(param);
                    num.push((shared.clone(), 1));
                    den.push((shared, 1));
                }
                let n = self.poly_or_factored(&num);
                let d = self.poly_or_factored(&den);
                format!("({n})/({d})")
            }
            75..=89 => {
                let k = self.r.int(2, 3);
                let mut terms = Vec::new();
                if self.r.chance(30) {
                    terms.push(format!("x + {}", num(self.r.int(-3, 3))));
                }
                for _ in 0..k {
                    let mut c = self.r.int(-5, 4);
                    if c >= 0 {
                        c += 1;
                    }
                    let f = p_render(&self.factor(param));
                    if self.r.chance(25) {
                        terms.push(format!("{}/({f})^2", num(c)));
                    } else {
                        terms.push(format!("{}/({f})", num(c)));
                    }
                }
                terms.join(" + ")
            }
            _ => {
                const T: &[&str] = &[
                    "(x^2 - {c})/(x - sqrt({c}))",
                    "x^2 - {c}",
                    "(x^2 + {c})/(x - I*sqrt({c}))",
                    "sqrt({c})*x^2 + sqrt({c})*x",
                    "x^2 + 2*I*x - 1",
                    "(x^4 - {cc})/(x^2 - {c})",
                    "sqrt({c})/(x^2 - {c})",
                    "(x^3 - {c}*sqrt({c}))/(x - sqrt({c}))",
                    "x^2 + 2*sqrt({c})*x + {c}",
                    "1/(x^2 + {c})",
                    "(x - sqrt({c}))*(x + sqrt({c}))",
                    "I/(x^2 + 1)",
                ];
                let t = self.r.pick(T);
                let c = self.r.pick(&["2", "3", "5", "7"]);
                let cc = (c.parse::<i64>().unwrap_or(2).pow(2)).to_string();
                t.replace("{cc}", &cc).replace("{c}", c)
            }
        }
    }

    // ── trigonometric identities ──────────────────────────────────────

    fn trig_case(&mut self) -> String {
        const T: &[&str] = &[
            "sin({A})^2 + cos({A})^2",
            "{c}*sin({A})^2 + {c}*cos({A})^2",
            "1 + tan({A})^2 - sec({A})^2",
            "sin(2*{A}) - 2*sin({A})*cos({A})",
            "cos(2*{A}) - cos({A})^2 + sin({A})^2",
            "tan({A}) - sin({A})/cos({A})",
            "cosh({A})^2 - sinh({A})^2",
            "sin({A} + {B}) - sin({A})*cos({B}) - cos({A})*sin({B})",
            "cos({A} - {B}) - cos({A})*cos({B}) - sin({A})*sin({B})",
            "exp(I*{A}) - cos({A}) - I*sin({A})",
            "cot({A})*tan({A})",
            "sin({A})^4 - cos({A})^4",
            "sin({A})^3 + sin({A})*cos({A})^2",
            "(1 - cos(2*{A}))/2 - sin({A})^2",
            "sin({A})/(1 + cos({A})) - tan({A}/2)",
            "csc({A})^2 - cot({A})^2",
            "sin(3*{A}) - 3*sin({A}) + 4*sin({A})^3",
            "tanh({A}) - sinh({A})/cosh({A})",
            "sin({A})*cos({B})",
            "sin({A})^2*cos({A})^3",
            "1/(1 + sin({A}))",
            "sin({A})^2 - {c}",
            "cos({A})^2 + {c}*sin({A})*cos({A})",
            "atan(tan({A}))",
            "asin(sin({A}))",
            "sin(asin({A})) + cos(acos({A}))",
            "exp({A})*sin({A})",
            "sec({A}) - 1/cos({A})",
            "sinh(2*{A}) - 2*sinh({A})*cosh({A})",
            "tan({A} + {B})",
        ];
        const A: &[&str] = &[
            "x", "(2*x)", "(x + y)", "(x/2)", "(x + pi/3)", "(3*x - 1)", "y", "(pi/4)", "(x^2)",
        ];
        const B: &[&str] = &["x", "y", "(2*x)", "(pi/6)"];
        const C: &[&str] = &["2", "3", "5", "(1/2)", "(-3)", "(2/3)"];
        let t = self.r.pick(T);
        let a = self.r.pick(A);
        let b = self.r.pick(B);
        let c = self.r.pick(C);
        t.replace("{A}", a).replace("{B}", b).replace("{c}", c)
    }

    // ── equations, as lhs - (rhs) ─────────────────────────────────────

    fn equation_case(&mut self) -> String {
        if self.r.chance(20) {
            let lhs = self.tree(1, 3, true, true);
            let rhs = self.tree(2, 3, true, true);
            return format!("{lhs} - ({rhs})");
        }
        const T: &[&str] = &[
            "{a}*x + {b} - ({c})",
            "x^2 + {b}*x + {d}",
            "x^3 - {a}*x + {b}",
            "x^4 - {a}*x^2 + {b}",
            "x^2 - {a}*y",
            "exp({a}*x) - {c}",
            "exp(x) - {a}*exp(-x) - {b}",
            "log({a}*x + {b}) - {c}",
            "sin({a}*x) - (1/2)",
            "cos(x) - {r}",
            "tan(x) - {b}",
            "sqrt(x + {a}) - x - ({b})",
            "{a}/(x + {b}) - {c}/(x + {d})",
            "x^{q} - {c}",
            "abs(x - {a}) - {c}",
            "x*exp(x) - {a}",
            "{a}^x - {c}",
            "x^x - {c}",
            "sin(x)^2 - sin(x)",
            "exp(2*x) - {a}*exp(x) + {d}",
            "log(x)^2 - {a}*log(x)",
            "sinh(x) - {a}",
            "x - cos(x)",
            "x^5 - x + {a}",
            "x^2 + y^2 - {c}",
            "(x - {a})*(x - {b})*(x - y)",
            "x^3 - {r}",
        ];
        let t = self.r.pick(T);
        let a = self.r.int(1, 5).to_string();
        let b = {
            let v = self.r.int(-5, 4);
            num(if v >= 0 { v + 1 } else { v })
        };
        let c = self.r.int(1, 6).to_string();
        let dd = {
            let v = self.r.int(-4, 3);
            num(if v >= 0 { v + 1 } else { v })
        };
        let r = self.r.pick(&["0", "(1/2)", "(-1)", "(sqrt(2)/2)", "2", "(1/3)"]);
        let q = self.r.pick(&["(1/2)", "(1/3)", "(3/2)", "(-1)", "(2/3)"]);
        t.replace("{a}", &a)
            .replace("{b}", &b)
            .replace("{c}", &c)
            .replace("{d}", &dd)
            .replace("{r}", r)
            .replace("{q}", q)
    }
}

// ── a tiny bivariate integer polynomial, for expanded forms ────────────

/// `(deg_x, deg_y) → coefficient`.
type P = BTreeMap<(u32, u32), i64>;

fn p_mul(a: &P, b: &P) -> P {
    let mut out = P::new();
    for (&(i, j), &c) in a {
        for (&(k, m), &e) in b {
            *out.entry((i + k, j + m)).or_insert(0) += c * e;
        }
    }
    out.retain(|_, c| *c != 0);
    out
}

fn p_product(fs: &[(P, u32)]) -> P {
    let mut acc: P = [((0, 0), 1)].into_iter().collect();
    for (f, m) in fs {
        for _ in 0..*m {
            acc = p_mul(&acc, f);
        }
    }
    acc
}

fn p_render(p: &P) -> String {
    let mut terms: Vec<(&(u32, u32), &i64)> = p.iter().collect();
    terms.sort_by(|a, b| {
        let (da, db) = (a.0.0 + a.0.1, b.0.0 + b.0.1);
        db.cmp(&da).then(b.0.0.cmp(&a.0.0))
    });
    if terms.is_empty() {
        return "0".into();
    }
    let mut s = String::new();
    for (n, (&(i, j), &c)) in terms.into_iter().enumerate() {
        let mut mono = Vec::new();
        match i {
            0 => {}
            1 => mono.push("x".to_string()),
            _ => mono.push(format!("x^{i}")),
        }
        match j {
            0 => {}
            1 => mono.push("y".to_string()),
            _ => mono.push(format!("y^{j}")),
        }
        let mono = mono.join("*");
        let a = c.abs();
        let body = if mono.is_empty() {
            a.to_string()
        } else if a == 1 {
            mono
        } else {
            format!("{a}*{mono}")
        };
        if n == 0 {
            if c < 0 {
                s.push('-');
            }
        } else {
            s.push_str(if c < 0 { " - " } else { " + " });
        }
        s.push_str(&body);
    }
    s
}

fn render_factored(fs: &[(P, u32)]) -> String {
    let parts: Vec<String> = fs
        .iter()
        .map(|(f, m)| {
            if *m == 1 {
                format!("({})", p_render(f))
            } else {
                format!("({})^{m}", p_render(f))
            }
        })
        .collect();
    parts.join("*")
}
