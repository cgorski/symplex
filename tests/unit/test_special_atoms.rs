//! Ratchet over `src/`: raw comparisons with the special atoms `∞`, `−∞`,
//! `zoo`, `nan` (`x == arena.infinity`, `args.get(1) == Some(&arena.nan)`,
//! `self.complex_infinity != y` in arena methods) and raw tests for them
//! with `contains` (`walk::contains(arena, a, arena.infinity)`,
//! `args.contains(&arena.nan)`) outside `src/base/value_class.rs` must not
//! grow beyond their allowlisted count per file; an allowlisted file must
//! not lose one without the allowlist being tightened.  See CONTRIBUTING.md,
//! "Values at infinity" (session 15, wave 13, M1).
//!
//! The bug class: a fold written for `±∞` that compares its argument with
//! the atom by hand fires at `zoo`, at an infinity of unknown direction, at
//! an unevaluated pole, or at a constant not known finite, and each area
//! that needed "is this argument finite" wrote its own test, which
//! disagreed with the others (`uppergamma(elliptic_k(besseli(0, 0)), ∞)`
//! was 0 with `is_zero() = Some(true)` up to 0.56).  New code decides with
//! `base::value_class::ValueClass`: `ValueClass::special_atom` for a fold
//! at a literal atom (a `match` that says what happens at each one),
//! `ValueClass::of(..).is_finite_proved()` for "every finite value of this
//! argument", `holds_special_atom`/`holds_real_infinity` for "an infinity
//! inside".  What is left are identity checks that decide nothing (the
//! factor `∞` beside the direction of `c·∞`, a constructor's own atoms) and
//! the calculus files whose comparisons are migrated with their deciders.
//!
//! Two counts, both syntactic (no types are available to a linter):
//!
//! - **Comparisons**: `==`/`!=` with an operand that is, through
//!   parentheses, `&` and `Some(…)`, a field `.infinity`, `.neg_infinity`,
//!   `.complex_infinity` or `.nan`, or an accessor call of one of those
//!   names without arguments (`arena.infinity()`; the arena's atoms: every
//!   such field in `src/` is one).
//! - **Contains**: a call of a function or method named `contains` with
//!   such a field as an argument (through `&`).
//!
//! `#[cfg(test)]` items and modules are skipped.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::visit::Visit;

/// The file that owns the comparisons (`ValueClass::special_atom`).
const OWNER: &str = "base/value_class.rs";

/// Files allowed to compare with the special atoms, with the exact count
/// (0.58: 167, down from 263 — `canon.rs` 50 → 4, `eval.rs` 50 → 0).  Paths
/// are relative to `src/`.
const COMPARE_ALLOWLIST: &[(&str, usize)] = &[
    ("api/expr_calculus_util_ext.rs", 16),
    ("api/expr_integrate_ext.rs", 2),
    // Identity checks that decide nothing (each says so in a comment): the
    // factor `∞` beside the direction of `c·∞`, the spelling `q·i·∞`.
    ("base/canon.rs", 4),
    ("calculus/convergence.rs", 7),
    // The calculus files migrate with their deciders (`NonzeroExpr`, the
    // limit points): frozen.
    ("calculus/definite.rs", 21),
    ("calculus/gruntz.rs", 26),
    ("calculus/laplace.rs", 1),
    ("calculus/limit.rs", 39),
    ("calculus/residue.rs", 1),
    ("calculus/summation.rs", 3),
    // `Ex` comparisons of support ends with `ctx.infinity()`.
    ("domains/stats/continuous.rs", 2),
    ("domains/stats/family.rs", 13),
    ("domains/stats/support.rs", 2),
    ("poly/paramfield.rs", 2),
    ("simplify/ratsimp.rs", 7),
    ("simplify/simplify_engine.rs", 6),
    ("transforms/apart.rs", 2),
    ("transforms/evalf/improper.rs", 2),
    ("transforms/logic.rs", 2),
    ("transforms/sets.rs", 8),
    ("transforms/subs.rs", 1),
];

/// Files allowed to test for the special atoms with `contains`, with the
/// exact count (0.58: 18, down from 32 — `canon.rs` 11 → 1, `eval.rs` 5 → 1).
const CONTAINS_ALLOWLIST: &[(&str, usize)] = &[
    // `is_directed_infinity`: the shape `c·∞` (identity).
    ("base/canon.rs", 1),
    ("calculus/formal_series.rs", 4),
    ("calculus/gruntz.rs", 4),
    ("calculus/limit.rs", 4),
    ("calculus/series.rs", 4),
    // `betainc` at an infinite bound: a `nan` in the antiderivative's value.
    ("transforms/eval.rs", 1),
];

const ATOMS: &[&str] = &["infinity", "neg_infinity", "complex_infinity", "nan"];

fn strip(mut e: &syn::Expr) -> &syn::Expr {
    loop {
        match e {
            syn::Expr::Paren(p) => e = &p.expr,
            syn::Expr::Group(g) => e = &g.expr,
            syn::Expr::Reference(r) => e = &r.expr,
            syn::Expr::Call(c)
                if c.args.len() == 1
                    && matches!(&*c.func, syn::Expr::Path(p) if p.path.is_ident("Some")) =>
            {
                e = &c.args[0];
            }
            _ => return e,
        }
    }
}

/// Is `e` (through parentheses, `&` and `Some(…)`) one of the arena's atoms:
/// a field `x.infinity` or an accessor call `x.infinity()`?
fn is_atom_field(e: &syn::Expr) -> bool {
    match strip(e) {
        syn::Expr::Field(f) => {
            matches!(&f.member, syn::Member::Named(n) if ATOMS.contains(&n.to_string().as_str()))
        }
        syn::Expr::MethodCall(m) => {
            m.args.is_empty() && ATOMS.contains(&m.method.to_string().as_str())
        }
        _ => false,
    }
}

fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.path().is_ident("cfg") && a.to_token_stream().to_string().contains("test"))
}

#[derive(Default)]
struct Finder {
    compare: Vec<String>,
    contains: Vec<String>,
}

fn site(e: &impl ToTokens) -> String {
    let s = e.to_token_stream().to_string();
    if s.chars().count() > 100 {
        format!("{}…", s.chars().take(100).collect::<String>())
    } else {
        s
    }
}

impl<'ast> Visit<'ast> for Finder {
    fn visit_item(&mut self, i: &'ast syn::Item) {
        let attrs = match i {
            syn::Item::Fn(f) => &f.attrs,
            syn::Item::Impl(x) => &x.attrs,
            syn::Item::Mod(x) => &x.attrs,
            syn::Item::Const(x) => &x.attrs,
            syn::Item::Static(x) => &x.attrs,
            syn::Item::Trait(x) => &x.attrs,
            _ => return syn::visit::visit_item(self, i),
        };
        if !is_cfg_test(attrs) {
            syn::visit::visit_item(self, i);
        }
    }

    fn visit_impl_item_fn(&mut self, f: &'ast syn::ImplItemFn) {
        if !is_cfg_test(&f.attrs) {
            syn::visit::visit_impl_item_fn(self, f);
        }
    }

    fn visit_expr_binary(&mut self, b: &'ast syn::ExprBinary) {
        if matches!(b.op, syn::BinOp::Eq(_) | syn::BinOp::Ne(_))
            && (is_atom_field(&b.left) || is_atom_field(&b.right))
        {
            self.compare.push(site(b));
        }
        syn::visit::visit_expr_binary(self, b);
    }

    fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
        if matches!(&*c.func, syn::Expr::Path(p)
            if p.path.segments.last().is_some_and(|s| s.ident == "contains"))
            && c.args.iter().any(is_atom_field)
        {
            self.contains.push(site(c));
        }
        syn::visit::visit_expr_call(self, c);
    }

    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        if m.method == "contains" && m.args.iter().any(is_atom_field) {
            self.contains.push(site(m));
        }
        syn::visit::visit_expr_method_call(self, m);
    }

    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        // `matches!(e, P if guard)`, `assert!(…)`: the expressions inside.
        if m.path.is_ident("matches") {
            if let Ok((e, guard)) = m.parse_body_with(parse_matches) {
                self.visit_expr(&e);
                if let Some(g) = guard {
                    self.visit_expr(&g);
                }
            }
        } else if let Ok(args) = m.parse_body_with(
            syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
        ) {
            for e in &args {
                self.visit_expr(e);
            }
        }
    }
}

/// The scrutinee and the guard of `matches!(e, P if guard)`.
fn parse_matches(input: syn::parse::ParseStream) -> syn::Result<(syn::Expr, Option<syn::Expr>)> {
    let e: syn::Expr = input.parse()?;
    input.parse::<syn::Token![,]>()?;
    syn::Pat::parse_multi_with_leading_vert(input)?;
    let guard = if input.peek(syn::Token![if]) {
        input.parse::<syn::Token![if]>()?;
        Some(input.parse()?)
    } else {
        None
    };
    input.parse::<Option<syn::Token![,]>>()?;
    Ok((e, guard))
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

fn scan_src() -> BTreeMap<String, Finder> {
    let src = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let mut files = Vec::new();
    collect_rs_files(src, &mut files);
    files.sort();
    let mut found = BTreeMap::new();
    for path in &files {
        let rel = path
            .strip_prefix(src)
            .expect("file is under src/")
            .to_string_lossy()
            .replace('\\', "/");
        if rel == OWNER {
            continue;
        }
        let text = fs::read_to_string(path).expect("source file is readable");
        let file = syn::parse_file(&text)
            .unwrap_or_else(|e| panic!("{}: does not parse as Rust: {e}", path.display()));
        let mut f = Finder::default();
        f.visit_file(&file);
        found.insert(rel, f);
    }
    found
}

fn check(
    what: &str,
    advice: &str,
    allowlist: &[(&str, usize)],
    found: &BTreeMap<String, Vec<String>>,
) -> Vec<String> {
    let allowed: BTreeMap<&str, usize> = allowlist.iter().copied().collect();
    let mut violations = Vec::new();
    for (rel, sites) in found {
        let count = sites.len();
        let allowance = allowed.get(rel.as_str()).copied().unwrap_or(0);
        if count > allowance {
            violations.push(format!(
                "{rel}: {count} {what}(s), allowed {allowance} — {advice}:\n    {}",
                sites.join("\n    ")
            ));
        } else if count < allowance {
            violations.push(format!(
                "{rel}: {count} {what}(s), allowed {allowance} — tighten the allowlist"
            ));
        }
    }
    for (rel, &allowance) in &allowed {
        if found.get(*rel).is_none_or(Vec::is_empty) {
            violations.push(format!(
                "{rel}: allowlisted ({allowance}) but none found under src/ — tighten the allowlist"
            ));
        }
    }
    violations
}

const ADVICE: &str = "decide with `base::value_class::ValueClass` (`special_atom` for a fold at a \
                      literal atom, `of(..).is_finite_proved()` for a finite argument, \
                      `holds_special_atom`/`holds_real_infinity` for an infinity inside; \
                      CONTRIBUTING.md, \"Values at infinity\"), or comment an identity check \
                      that decides nothing and raise the count";

#[test]
fn special_atom_comparisons_do_not_grow() {
    let found: BTreeMap<String, Vec<String>> = scan_src()
        .into_iter()
        .filter(|(_, f)| !f.compare.is_empty())
        .map(|(rel, f)| (rel, f.compare))
        .collect();
    let violations = check(
        "raw comparison with a special atom",
        ADVICE,
        COMPARE_ALLOWLIST,
        &found,
    );
    if !violations.is_empty() {
        panic!(
            "special-atom ratchet: {} file(s) out of step:\n\n{}\n",
            violations.len(),
            violations.join("\n\n")
        );
    }
}

#[test]
fn special_atom_contains_tests_do_not_grow() {
    let found: BTreeMap<String, Vec<String>> = scan_src()
        .into_iter()
        .filter(|(_, f)| !f.contains.is_empty())
        .map(|(rel, f)| (rel, f.contains))
        .collect();
    let violations = check(
        "raw `contains` test for a special atom",
        ADVICE,
        CONTAINS_ALLOWLIST,
        &found,
    );
    if !violations.is_empty() {
        panic!(
            "special-atom ratchet: {} file(s) out of step:\n\n{}\n",
            violations.len(),
            violations.join("\n\n")
        );
    }
}
