# Cases of the code rules of the rule set `domains` (review F §6): the two table-driven rewrites.
# The renames of tables/domain_renames.tsv are in cases_renames.py.
from support import case

P = "use symplex::prelude::*;\n"
S = "use symplex::prelude::*;\nuse symplex::stats::Distribution;\n"

CASES = [
    # ---- Distribution constructors -> parameter records
    case("domains.dist-ctor",
         S + "fn t() {\n    let d = Distribution::normal(ctx.int(0), ctx.int(1));\n}\n",
         S + "use symplex::stats::Normal;\nfn t() {\n    let d = Distribution::new(Normal { mean: ctx.int(0), std: ctx.int(1) }).unwrap();\n}\n"),
    case("domains.dist-ctor",  # the try_ form already returns the Result: no unwrap is added
         S + "fn run() -> Result<(), SymplexError> {\n    let g = Distribution::try_gamma(k, theta)?;\n"
             "    let e = Distribution::exponential(rate);\n    Ok(())\n}\n",
         S + "use symplex::stats::{Exponential, Gamma};\nfn run() -> Result<(), SymplexError> {\n"
             "    let g = Distribution::new(Gamma { shape: k, scale: theta })?;\n"
             "    let e = Distribution::new(Exponential { rate })?;\n    Ok(())\n}\n"),
    case("domains.dist-ctor",  # arguments named like the fields use the shorthand; weibull is (scale, shape)
         S + "use symplex::stats::Weibull;\nfn t() {\n    let w = Distribution::weibull(scale, k.clone());\n}\n",
         S + "use symplex::stats::Weibull;\nfn t() {\n    let w = Distribution::new(Weibull { scale, shape: k.clone() }).unwrap();\n}\n"),
    case("domains.dist-ctor",  # spread over lines, with a trailing comma
         S + "use symplex::stats::Hypergeometric;\nfn t() {\n    let h = Distribution::hypergeometric(\n        ctx.int(50),\n"
             "        ctx.int(5),\n        ctx.int(10),\n    );\n}\n",
         S + "use symplex::stats::Hypergeometric;\nfn t() {\n    let h = Distribution::new(\n        Hypergeometric { population: ctx.int(50),\n"
             "        successes: ctx.int(5),\n        draws: ctx.int(10) },\n    ).unwrap();\n}\n"),
    case("domains.dist-ctor",  # a method follows: the Result is unwrapped before it
         S + "use symplex::stats::Poisson;\nfn t() {\n    let m = Distribution::poisson(lam).mean();\n}\n",
         S + "use symplex::stats::Poisson;\nfn t() {\n    let m = Distribution::new(Poisson { rate: lam }).unwrap().mean();\n}\n"),
    case("domains.dist-ctor",
         S + "fn t() { let d = Distribution::die(ctx.int(6)); }\n", manual="more than fill a record"),
    case("domains.dist-ctor",
         S + "fn t() { let d = Distribution::finite(&ctx, table); }\n", manual="private fields"),
    case("domains.dist-ctor",
         S + "fn t() { let ds: Vec<_> = ps.into_iter().map(Distribution::bernoulli).collect(); }\n",
         manual="function value"),
    case("domains.dist-ctor",  # another type's `normal`, and the already migrated form
         S + "fn t() { let a = Other::normal(1, 2); let b = Distribution::new(Normal { mean, std })?; }\n", silent=True),
    # ---- Matrix::get(i, j) -> m[(i, j)]
    case("domains.matrix-get",
         P + "fn t(m: &Matrix) {\n    assert_eq!(m.get(0, 1), &ctx.int(2));\n    let s = m.get(i, j) + m.get(j, i);\n}\n",
         P + "fn t(m: &Matrix) {\n    assert_eq!(&m[(0, 1)], &ctx.int(2));\n    let s = &m[(i, j)] + &m[(j, i)];\n}\n"),
    case("domains.matrix-get",  # a method on the entry: no `&` (it would bind to the whole chain)
         P + "fn t(m: &Matrix) {\n    let e = m.get(0, 0).clone();\n    let f = self.jac.get(r, c + 1).simplify();\n}\n",
         P + "fn t(m: &Matrix) {\n    let e = m[(0, 0)].clone();\n    let f = self.jac[(r, c + 1)].simplify();\n}\n"),
    case("domains.matrix-get",
         P + "fn t(m: &Matrix) { let z = *m.get(0, 0) == zero; let n = -m.get(1, 1); }\n",
         P + "fn t(m: &Matrix) { let z = m[(0, 0)] == zero; let n = -&m[(1, 1)]; }\n"),
    # already the 1.0 `get -> Option`; one-argument `get`; a file without matrices
    case("domains.matrix-get",
         P + "fn t(m: &Matrix) { assert!(m.get(9, 9).is_none()); let a = v.get(0); let b = h.get(&k); }\n", silent=True),
    case("domains.matrix-get",  # no sign of a matrix in the file: listed, not guessed
         "fn t(g: &Grid) { let c = g.get(1, 2); }\n", manual="nothing in this file says"),
    case("domains.matrix-get",  # a matrix method is evidence enough
         "fn t() { let r = quat.to_rotation_matrix(); let v = r.get(i, j).eval(); }\n",
         "fn t() { let r = quat.to_rotation_matrix(); let v = r[(i, j)].eval(); }\n"),
    case("domains.matrix-get",
         P + "struct Grid;\nimpl Grid { fn get(&self, i: usize, j: usize) -> u8 { 0 } }\nfn t(m: &Matrix, g: &Grid) { g.get(1, 2); }\n",
         manual="another type with a two-argument `get`"),
    # ---- the try_ accessors
    case("domains.matrix-try",
         P + "fn t(m: &Matrix) {\n    assert!(m.try_get(5, 5).is_none());\n    let r = m.try_row(0).unwrap();\n}\n",
         P + "fn t(m: &Matrix) {\n    assert!(m.get(5, 5).is_none());\n    let r = m.row(0).unwrap();\n}\n"),
    case("domains.matrix-try",
         P + "fn t(m: &Matrix) { let e = m.try_get(0, 0); keep(e); }\n", manual="not read next to the call"),
]
