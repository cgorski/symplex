//! The operations run on every case, all through symplex's public API.
//!
//! Each `(case, op)` gets a fresh `Context` with the symbols `x`, `y`, `z`
//! created in that order before the expression is parsed, so a result
//! depends only on the expression string and the op — not on which ops or
//! cases ran before (hash-consing ids, symbol ids, assumption caches and
//! per-context budgets all start from the same state).  `main` also runs
//! each op on a fresh thread, so `thread_local!` caches start empty too.

use symplex::prelude::*;

/// Canonical op order (also the output order within a case).  `expr` is the
/// input string itself, printed by `main` without calling symplex.
pub const OPS: &[&str] = &[
    "expr",
    "display",
    "reparse",
    "simplify",
    "expand",
    "factor",
    "factor_all",
    "together",
    "cancel",
    "apart",
    "gcd",
    "diff",
    "integrate",
    "series",
    "limit0",
    "limitoo",
    "solve",
    "solve_set",
    "eval",
    "evald30",
    "evalf64",
    "compile",
    "latex",
    "mathml",
    "json",
    "json_rt",
    "is_positive",
    "is_real",
    "is_zero",
    "free_symbols",
    "trigsimp",
];

fn list(xs: &[Ex]) -> String {
    let parts: Vec<String> = xs.iter().map(|e| e.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

/// `e` at x = 7/10, y = −13/10 (a no-op for symbol-free `e`).
fn at_point(ctx: &Context, e: &Ex, x: &Ex, y: &Ex) -> Ex {
    e.subs(x, &ctx.rational(7, 10)).subs(y, &ctx.rational(-13, 10))
}

/// `Ok(text)` → status OK; `Err(text)` → status ERR (an error symplex
/// returned, or a parse failure prefixed `parse: `).
pub fn run(op: &str, src: &str) -> Result<String, String> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let _z = ctx.symbol("z");
    let e = ctx.parse(src).map_err(|err| format!("parse: {err}"))?;
    let er = |err: SymplexError| err.to_string();
    Ok(match op {
        "display" => e.to_string(),
        "reparse" => {
            let back = ctx
                .parse(&e.to_string())
                .map_err(|err| format!("reparse: {err}"))?;
            if back == e { "eq".into() } else { format!("ne: {back}") }
        }
        "simplify" => e.simplify().to_string(),
        "expand" => e.expand().to_string(),
        "factor" => e.factor(&x).to_string(),
        "factor_all" => e.factor_all().to_string(),
        "together" => e.together().to_string(),
        "cancel" => e.cancel(&x).to_string(),
        "apart" => e.partial_fractions(&x).to_string(),
        "gcd" => {
            let (n, d) = e.as_numer_denom();
            let g = match n.poly_gcd(&d, &x) {
                Some(g) => g.to_string(),
                None => "None".into(),
            };
            format!("gcd={g} num={n} den={d}")
        }
        "diff" => e.diff(&x).to_string(),
        "integrate" => e.integrate(&x).to_string(),
        "series" => e.series(&x, &ctx.int(0), 4).to_string(),
        "limit0" => e.limit(&x, &ctx.int(0)).to_string(),
        "limitoo" => e.limit(&x, &ctx.infinity()).to_string(),
        "solve" => list(&e.solve(&x).map_err(er)?),
        "solve_set" => e.solve_as_set(&x).to_string(),
        "eval" => e.eval().to_string(),
        "evald30" => at_point(&ctx, &e, &x, &y).eval_decimal(30).map_err(er)?,
        "evalf64" => format!("{:?}", at_point(&ctx, &e, &x, &y).eval_f64().map_err(er)?),
        "compile" => {
            let f = e.compile(&["x", "y"]).map_err(er)?;
            format!("{:?}", f.try_call(&[0.7, -1.3]).map_err(er)?)
        }
        "latex" => e.to_latex(),
        "mathml" => e.to_mathml().map_err(er)?,
        "json" => e.to_json().map_err(er)?,
        "json_rt" => {
            let j = e.to_json().map_err(er)?;
            let back = ctx.from_json(&j).map_err(|err| format!("from_json: {err}"))?;
            if back == e { "eq".into() } else { format!("ne: {back}") }
        }
        "is_positive" => format!("{:?}", e.is_positive()),
        "is_real" => format!("{:?}", e.is_real()),
        "is_zero" => format!("{:?}", e.is_zero()),
        "free_symbols" => list(&e.free_symbols()),
        "trigsimp" => e.simplify_trig().to_string(),
        _ => return Err(format!("golden: unknown op {op}")),
    })
}
