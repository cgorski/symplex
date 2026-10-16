//! Ratchet over `src/`: decisions made by comparing a computed float with a
//! small threshold (session 14, C1).  See CONTRIBUTING.md, "Type rules that
//! keep bug classes out", and `src/base/truth.rs` (`Evidence`).
//!
//! Only exact or certified evidence may justify a claim — a returned value,
//! a dropped term, the order of a pole, a branch taken, "converges",
//! "real", "zero".  An `f64` (or a decimal string parsed back into one)
//! carries no error bound: a value below `1e-18` may be `e⁻⁵⁰` (up to 0.50
//! the order of the pole of `1/(z(z + e⁻⁵⁰))` at `0` was decided that way,
//! and its residue `e⁵⁰` came out `0`), and a hidden zero (`√2·√3 − √6`)
//! evaluates to a rounding residue of either sign.  A float comparison may
//! guide a heuristic whose result is right either way — a pivot, the order
//! of methods, a sanity check that can only refuse, the placement of a value
//! whose error bound is far below the margin — but a claim needs exact
//! arithmetic or a certified evaluation (`evalf_verdict(..).proven()`,
//! `calculus_util::hidden_zero`, `cmp_values`, a certified root ball).
//!
//! Two counts per file, both syntactic (a linter sees no types):
//!
//! | column | construct |
//! |---|---|
//! | `tol` | a comparison `<`, `<=`, `>`, `>=` with an operand that contains a small float threshold: a float literal of magnitude at most `1e-3` (`v.abs() < 1e-18`, `z.im.abs() <= 1e-12 * z.norm()`, `re > -1e-9 * scale`), or a name bound to an expression containing one — a `let` of the same function (`let tiny = 1e-12 * s; … v <= tiny`) or a `const` of the same file |
//! | `zero` | `==` or `!=` with the float literal `0.0` as an operand (`z.im == 0.0`): exact equality of a float with zero, which a computed value reaches by luck, underflow or a rounding of a nonzero part |
//!
//! Not counted: comparisons with larger literals (`|x| < 1.0` is a range
//! test), a tolerance passed to a function (`approx_eq(a, b, 1e-12)`), a
//! threshold behind a name from another file, and `#[cfg(test)]` items.  A
//! comparison inside a closure or nested in another comparison counts once,
//! where it stands.  Macro bodies are scanned as expressions when they parse
//! as a comma-separated list.
//!
//! Not every site is wrong: many guard a heuristic or a refusal, or place a
//! value whose certified error is far below the margin.  The table is the
//! review queue.  The counts are exact: a file may not gain a site, and a
//! file that loses one must have its entry tightened (the failure message
//! prints the table to paste).  Files without an entry must have no site.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::punctuated::Punctuated;
use syn::visit::Visit;

/// Per-file counts `[tol, zero]` (the columns of the module doc).  Paths
/// are relative to `src/`; files with no site have no entry.  At 0.50.0:
/// 71 files, 313 `tol`, 179 `zero`; fixed after s14c (the pole order of a
/// residue, the real-part sign of a pole in the Laplace transform, the
/// placement of explicit roots, `ode::residual_verdict`,
/// `integrate::known_zero` made exact or certified): 70 files, 306, 179.
const ALLOWLIST: &[(&str, [usize; 2])] = &[
    // Reviewed in s15h5 (with poly/roots, simplify/nsimplify, eval/atan_sum, domains/linalg,
    // matrix, matrix_decomp, transforms/sets, solve): what is left steers `f64` APIs (Newton,
    // nsimplify), only refuses, or reads certified 16-digit values with a margin far above their
    // error; family index ranges, realness of zeros and roots, `check_solution` and
    // `compare_numeric` are exact or certified since.
    ("api/expr_calculus_util_ext.rs", [0, 1]),
    ("api/expr_funcs.rs", [5, 0]),
    ("api/expr_ops.rs", [2, 0]),
    ("api/expr_solve_ext.rs", [6, 0]),
    ("base/bigfloat.rs", [0, 1]),
    ("base/dense_f64.rs", [2, 5]),
    ("base/numeric.rs", [0, 1]),
    ("calculus/calculus_util.rs", [1, 3]),
    ("calculus/convergence.rs", [2, 0]),
    // Reviewed in s15g4: what is left are the quadrature internals, refusal checks and certified
    // margins; cut crossings, the `ln²` coefficients at infinity and the side of the negative axis
    // are certified since.
    ("calculus/definite.rs", [16, 22]),
    ("calculus/gruntz.rs", [3, 0]),
    ("calculus/limit.rs", [4, 0]),
    // `radical_char_roots` (the Sturm count decides; margins refuse), `is_real_radical` (radicals
    // of rationals), `real_imag_parts`, `CEx::is_zero` (certified margin or `Unknown`),
    // `is_real_constant` (selects a method that is right either way): reviewed in s15g4;
    // `residual_verdict` is certified since s14c.
    ("calculus/ode.rs", [8, 4]),
    // `analytic_at_zero`: branch cuts and poles of Γ/ψ read from an `f64` correctly rounded from a
    // 128-bit value, with margins of 1e-12: a value inside a margin refuses (s14c audit).
    ("calculus/residue.rs", [6, 0]),
    ("calculus/risch/log_to_real.rs", [2, 0]),
    ("calculus/summation.rs", [4, 0]),
    // Reviewed in s15j4 (with stats/*, optimize, robotics, quaternion, output/lambdify, codegen):
    // SOS thresholds steer the search (claims are exact: LDLᵀ + `verify`, rational refutation
    // points); the rest are `f64` APIs' own tests (convergence, exact input values, routes).  The
    // inverse of a transformed distribution, a binomial `n` and a zero Greenwood variance are
    // exact since.
    ("domains/certificates/sos.rs", [18, 4]),
    ("domains/linalg.rs", [1, 0]),
    ("domains/matrix.rs", [2, 0]),
    ("domains/matrix_decomp.rs", [2, 0]),
    ("domains/optimize.rs", [2, 11]),
    ("domains/quaternion.rs", [1, 1]),
    ("domains/robotics.rs", [0, 6]),
    ("domains/stats/anova.rs", [5, 2]),
    ("domains/stats/cox.rs", [2, 1]),
    ("domains/stats/family.rs", [0, 2]),
    ("domains/stats/hypothesis.rs", [11, 1]),
    ("domains/stats/multivariate.rs", [2, 0]),
    ("domains/stats/numdist.rs", [13, 14]),
    ("domains/stats/regression.rs", [3, 5]),
    ("domains/stats/sequential.rs", [1, 5]),
    ("output/codegen.rs", [0, 5]),
    ("output/codegen/numeric_rt.rs", [30, 15]),
    ("output/lambdify.rs", [0, 9]),
    ("plotting/data_export.rs", [2, 1]),
    ("plotting/sampling.rs", [5, 0]),
    ("plotting/svg_plot.rs", [13, 2]),
    ("plotting/textplot.rs", [3, 1]),
    ("plotting/tikz_plot.rs", [1, 1]),
    ("poly/roots.rs", [0, 1]),
    ("simplify/nsimplify.rs", [4, 0]),
    ("transforms/apart.rs", [7, 0]),
    ("transforms/eval.rs", [8, 3]),
    ("transforms/eval/atan_sum.rs", [2, 1]),
    // The evalf files were reviewed in s15h2: polygamma's order, a Sum's range, J_ν(0), the
    // f64-radius exactness of the zeta-family bounds, Hankel's termination and E_n's |n − 1| are
    // exact or certified; what is left guides a route, refuses, or has a certified margin.
    ("transforms/evalf.rs", [5, 7]),
    ("transforms/evalf/accuracy.rs", [1, 1]),
    ("transforms/evalf/asymsum.rs", [1, 0]),
    ("transforms/evalf/ball.rs", [1, 0]),
    ("transforms/evalf/bessel_debye.rs", [3, 0]),
    ("transforms/evalf/bessel_order.rs", [4, 0]),
    ("transforms/evalf/cerf.rs", [1, 0]),
    ("transforms/evalf/cexpint.rs", [0, 4]),
    ("transforms/evalf/cspecial.rs", [1, 0]),
    ("transforms/evalf/hypsum.rs", [7, 1]),
    ("transforms/evalf/lambertw.rs", [1, 2]),
    ("transforms/evalf/loggamma.rs", [0, 1]),
    ("transforms/evalf/polylog.rs", [4, 3]),
    ("transforms/evalf/quad.rs", [1, 0]),
    ("transforms/evalf/sensitivity.rs", [0, 2]),
    ("transforms/evalf/zeta.rs", [2, 2]),
    // 0.59: the `denom == 0.0` test of `eval_node_f64` went with its decimal round trip.
    ("transforms/heurisch.rs", [5, 0]),
    // FTC checks and continuity probes guard refusals (reviewed in s15g4); `known_zero` is exact
    // since s14c; real roots, their order and the side of `Ei`'s cut are certified since s15g4.
    ("transforms/integrate.rs", [11, 3]),
    ("transforms/sets.rs", [6, 1]),
    ("transforms/solve.rs", [4, 2]),
    ("transforms/trig_integ.rs", [4, 1]),
    ("units/inference.rs", [1, 0]),
];

const COLUMNS: [&str; 2] = ["tol", "zero"];

/// Largest magnitude of a literal that counts as a tolerance.
const SMALL: f64 = 1e-3;

fn parse(path: &Path) -> syn::File {
    let text = fs::read_to_string(path).expect("source file is readable");
    syn::parse_file(&text)
        .unwrap_or_else(|e| panic!("{}: does not parse as Rust: {e}", path.display()))
}

fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.path().is_ident("cfg") && a.to_token_stream().to_string().contains("test"))
}

fn float_value(lit: &syn::LitFloat) -> Option<f64> {
    lit.base10_parse::<f64>().ok()
}

fn is_comparison(op: &syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::Lt(_) | syn::BinOp::Le(_) | syn::BinOp::Gt(_) | syn::BinOp::Ge(_)
    )
}

/// `0.0`, `-0.0`, `0f64` (in parentheses or not).
fn is_float_zero(e: &syn::Expr) -> bool {
    match e {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Float(f),
            ..
        }) => float_value(f) == Some(0.0),
        syn::Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => is_float_zero(&u.expr),
        syn::Expr::Paren(p) => is_float_zero(&p.expr),
        _ => false,
    }
}

/// Does `e` contain a small float literal or one of `names` — not looking
/// into closures or nested comparisons (counted where they stand)?
fn has_threshold(e: &syn::Expr, names: &BTreeSet<String>) -> bool {
    struct Find<'a> {
        names: &'a BTreeSet<String>,
        found: bool,
    }
    impl<'ast> Visit<'ast> for Find<'_> {
        fn visit_lit_float(&mut self, f: &'ast syn::LitFloat) {
            if float_value(f).is_some_and(|v| v != 0.0 && v.abs() <= SMALL) {
                self.found = true;
            }
        }
        fn visit_expr_path(&mut self, p: &'ast syn::ExprPath) {
            if let Some(last) = p.path.segments.last()
                && self.names.contains(&last.ident.to_string())
            {
                self.found = true;
            }
            syn::visit::visit_expr_path(self, p);
        }
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
        fn visit_expr_binary(&mut self, b: &'ast syn::ExprBinary) {
            if !is_comparison(&b.op) {
                syn::visit::visit_expr_binary(self, b);
            }
        }
        fn visit_macro(&mut self, mac: &'ast syn::Macro) {
            if let Ok(exprs) =
                mac.parse_body_with(Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated)
            {
                for e in &exprs {
                    self.visit_expr(e);
                }
            }
        }
    }
    let mut f = Find {
        names,
        found: false,
    };
    f.visit_expr(e);
    f.found
}

/// The single name a `let` pattern binds (`x`, `mut x`, `x: f64`).
fn bound_name(p: &syn::Pat) -> Option<String> {
    match p {
        syn::Pat::Ident(i) => Some(i.ident.to_string()),
        syn::Pat::Type(t) => bound_name(&t.pat),
        _ => None,
    }
}

/// Names bound by a `let` in `block` (closures included, nested `fn`s not)
/// to an expression containing a threshold; `known` are the file's
/// threshold constants.  Iterated so that `let a = 1e-9; let b = 2.0 * a;`
/// marks both.
fn threshold_lets(block: &syn::Block, known: &BTreeSet<String>) -> BTreeSet<String> {
    struct Lets<'a> {
        all: Vec<(String, &'a syn::Expr)>,
    }
    impl<'a> Visit<'a> for Lets<'a> {
        fn visit_local(&mut self, l: &'a syn::Local) {
            if let (Some(name), Some(init)) = (bound_name(&l.pat), &l.init) {
                self.all.push((name, &*init.expr));
            }
            syn::visit::visit_local(self, l);
        }
        fn visit_item_fn(&mut self, _: &'a syn::ItemFn) {}
    }
    let mut lets = Lets { all: Vec::new() };
    lets.visit_block(block);
    let mut names = known.clone();
    loop {
        let before = names.len();
        for &(ref name, init) in &lets.all {
            if !names.contains(name) && has_threshold(init, &names) {
                names.insert(name.clone());
            }
        }
        if names.len() == before {
            return names;
        }
    }
}

/// The file's `const` items (outside `#[cfg(test)]`) whose value contains
/// a small float literal.
fn threshold_consts(file: &syn::File) -> BTreeSet<String> {
    struct Consts {
        out: BTreeSet<String>,
    }
    impl<'ast> Visit<'ast> for Consts {
        fn visit_item(&mut self, i: &'ast syn::Item) {
            if !is_cfg_test(item_attrs(i)) {
                syn::visit::visit_item(self, i);
            }
        }
        fn visit_item_const(&mut self, c: &'ast syn::ItemConst) {
            if has_threshold(&c.expr, &BTreeSet::new()) {
                self.out.insert(c.ident.to_string());
            }
        }
        fn visit_impl_item_const(&mut self, c: &'ast syn::ImplItemConst) {
            if has_threshold(&c.expr, &BTreeSet::new()) {
                self.out.insert(c.ident.to_string());
            }
        }
    }
    let mut c = Consts {
        out: BTreeSet::new(),
    };
    c.visit_file(file);
    c.out
}

fn item_attrs(item: &syn::Item) -> &[syn::Attribute] {
    match item {
        syn::Item::Const(i) => &i.attrs,
        syn::Item::Enum(i) => &i.attrs,
        syn::Item::Fn(i) => &i.attrs,
        syn::Item::Impl(i) => &i.attrs,
        syn::Item::Macro(i) => &i.attrs,
        syn::Item::Mod(i) => &i.attrs,
        syn::Item::Static(i) => &i.attrs,
        syn::Item::Struct(i) => &i.attrs,
        syn::Item::Trait(i) => &i.attrs,
        syn::Item::Type(i) => &i.attrs,
        syn::Item::Use(i) => &i.attrs,
        _ => &[],
    }
}

struct Scanner {
    consts: BTreeSet<String>,
    /// Threshold names in scope: one set per enclosing function body.
    names: Vec<BTreeSet<String>>,
    /// The function being scanned (for the site labels).
    fn_names: Vec<String>,
    sites: [Vec<String>; 2],
}

impl Scanner {
    fn in_fn(&mut self, name: String, block: &syn::Block) {
        let names = threshold_lets(block, &self.consts);
        self.names.push(names);
        self.fn_names.push(name);
        self.visit_block(block);
        self.fn_names.pop();
        self.names.pop();
    }

    fn label(&self, e: &syn::ExprBinary) -> String {
        let text = e.to_token_stream().to_string();
        let short: String = text.chars().take(90).collect();
        format!(
            "{}: {short}",
            self.fn_names.last().map_or("(item)", String::as_str)
        )
    }
}

impl<'ast> Visit<'ast> for Scanner {
    fn visit_item(&mut self, i: &'ast syn::Item) {
        if !is_cfg_test(item_attrs(i)) {
            syn::visit::visit_item(self, i);
        }
    }

    fn visit_item_fn(&mut self, f: &'ast syn::ItemFn) {
        self.in_fn(f.sig.ident.to_string(), &f.block);
    }

    fn visit_impl_item_fn(&mut self, m: &'ast syn::ImplItemFn) {
        if !is_cfg_test(&m.attrs) {
            self.in_fn(m.sig.ident.to_string(), &m.block);
        }
    }

    fn visit_trait_item_fn(&mut self, m: &'ast syn::TraitItemFn) {
        if let Some(block) = &m.default
            && !is_cfg_test(&m.attrs)
        {
            self.in_fn(m.sig.ident.to_string(), block);
        }
    }

    fn visit_expr_binary(&mut self, b: &'ast syn::ExprBinary) {
        let empty = BTreeSet::new();
        let names = self.names.last().unwrap_or(&empty);
        let names = if self.names.is_empty() {
            &self.consts
        } else {
            names
        };
        if is_comparison(&b.op) && (has_threshold(&b.left, names) || has_threshold(&b.right, names))
        {
            let l = self.label(b);
            self.sites[0].push(l);
        }
        if matches!(b.op, syn::BinOp::Eq(_) | syn::BinOp::Ne(_))
            && (is_float_zero(&b.left) || is_float_zero(&b.right))
        {
            let l = self.label(b);
            self.sites[1].push(l);
        }
        syn::visit::visit_expr_binary(self, b);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if let Ok(exprs) =
            mac.parse_body_with(Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated)
        {
            for e in &exprs {
                self.visit_expr(e);
            }
        }
    }
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("src/ is readable") {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn float_threshold_decisions_do_not_grow() {
    let src = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let mut files = Vec::new();
    collect_rs_files(src, &mut files);
    files.sort();

    let mut found: BTreeMap<String, [Vec<String>; 2]> = BTreeMap::new();
    for path in &files {
        let rel = path
            .strip_prefix(src)
            .expect("file is under src/")
            .to_string_lossy()
            .replace('\\', "/");
        let file = parse(path);
        let mut scanner = Scanner {
            consts: threshold_consts(&file),
            names: Vec::new(),
            fn_names: Vec::new(),
            sites: [Vec::new(), Vec::new()],
        };
        scanner.visit_file(&file);
        found.insert(rel, scanner.sites);
    }

    let allowed: BTreeMap<&str, [usize; 2]> = ALLOWLIST.iter().copied().collect();
    let mut violations = Vec::new();
    let mut table = Vec::new();
    let mut totals = [0usize; 2];
    for (rel, sites) in &found {
        let have = [sites[0].len(), sites[1].len()];
        totals[0] += have[0];
        totals[1] += have[1];
        if have.iter().any(|&n| n > 0) {
            table.push(format!("    (\"{rel}\", [{}, {}]),", have[0], have[1]));
        }
        let allowance = allowed.get(rel.as_str()).copied().unwrap_or([0; 2]);
        for col in 0..2 {
            let (n, a) = (have[col], allowance[col]);
            if n > a {
                violations.push(format!(
                    "{rel}: {n} `{}` site(s), allowed {a} — decide a claim on exact or certified \
                     evidence (`hidden_zero`, `evalf_verdict(..).proven()`, `cmp_values`), not an \
                     `f64` threshold:\n    {}",
                    COLUMNS[col],
                    sites[col].join("\n    ")
                ));
            } else if n < a {
                violations.push(format!(
                    "{rel}: {n} `{}` site(s), allowed {a} — tighten the allowlist",
                    COLUMNS[col]
                ));
            }
        }
    }
    for (rel, allowance) in &allowed {
        if !found.contains_key(*rel) {
            violations.push(format!(
                "{rel}: allowlisted ({allowance:?}) but not found under src/ — remove the entry"
            ));
        }
    }

    if !violations.is_empty() {
        panic!(
            "float-evidence ratchet: {} violation(s):\n\n{}\n\nCurrent table [tol, zero] \
             (totals {} / {}, {} files):\n{}\n",
            violations.len(),
            violations.join("\n\n"),
            totals[0],
            totals[1],
            table.len(),
            table.join("\n")
        );
    }
}
