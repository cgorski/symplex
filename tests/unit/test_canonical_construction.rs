//! Ratchet over `src/`: nodes of variants that have a canonical
//! constructor are built through it, not with a raw `intern` (wave 2, W7).
//!
//! `Arena::intern` stores a node exactly as given.  For most variants that
//! is fine, but some have a canonical form that only their constructor
//! establishes: a `FiniteSet` is sorted and deduplicated, an `Interval`
//! with numeric endpoints in the wrong order is `∅`, a `SetUnion` is
//! flattened and sorted, an `Add`/`Mul`/`Pow` is canonical arithmetic, a
//! `Subs` whose variable is not free is its body.  A raw
//! `intern(ExprNode::FiniteSet([1, 1]))` makes a `{1, 1}` that is not `==`
//! to the `{1}` every other path builds — structural equality, hash-consing
//! and set reasoning then disagree about one set (up to 0.48 substitution
//! and `from_json` built such nodes; the set evaluator's union still does,
//! see [`ALLOWLIST`]).  Outside
//! `src/base/`, build a node value with `Arena::make` (which routes every
//! variant to its constructor) or call the constructor (`arena.add`,
//! `arena.finite_set`, …).
//!
//! The test parses every file under `src/` outside `src/base/` with `syn`
//! and flags `intern(…ExprNode::X(…))` (method or path call; the variant is
//! matched on the last path segment, so `crate::base::node::ExprNode::X`
//! counts) for `X` in [`CANONICAL`], skipping `#[cfg(test)]` and `#[test]`
//! items (tests build raw nodes on purpose).  Justified exceptions go in
//! [`ALLOWLIST`] with an exact count per file.  A node built in a variable
//! and interned later is not seen — use `make` there too.  See
//! CONTRIBUTING.md, "Canonical construction".

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::visit::Visit;

/// Variants with a canonical constructor that `intern` would bypass.
/// (`Min`/`Max` have none; the functions whose constructor folds values
/// — `abs`, `sign`, `re`, … — are built through `make` by the rebuilds and
/// are not listed here because their raw nodes are already canonical
/// whenever `canon_function` in `intern` has nothing to fold.)
const CANONICAL: &[&str] = &[
    "Add",
    "Mul",
    "Pow",
    "FiniteSet",
    "Interval",
    "SetUnion",
    "SetIntersection",
    "Piecewise",
    "DefiniteIntegral",
    "Subs",
    "And",
    "Or",
    "Not",
    "Gt",
    "Ge",
    "Eq_",
    "Ne",
    // A library function's node is built by `Arena::lib_apply` /
    // `Arena::func`, which check its argument count (0.50, W6c).
    "Lib",
    // An internal function's node, by `Arena::func` (0.50, W6d).
    "Internal",
];

/// Files allowed to intern a canonical variant raw, with the exact count.
/// Paths are relative to `src/`.  Empty since 0.52: the set evaluator's
/// union (`rankset_to_expr`), raw up to 0.51 for its ascending order, goes
/// through `canon_set_union`, whose order of real intervals is ascending.
const ALLOWLIST: &[(&str, usize)] = &[];

#[derive(Default)]
struct Scan {
    /// The raw interns found, as `Variant: tokens`.
    raw: Vec<String>,
}

fn is_test_item(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        if a.path().is_ident("test") {
            return true;
        }
        if !a.path().is_ident("cfg") {
            return false;
        }
        let tokens = a.meta.to_token_stream().to_string().replace(' ', "");
        tokens.contains("test") && !tokens.contains("not(test")
    })
}

impl Scan {
    /// Records `node` if it is `…ExprNode::X(…)` (or `X(…)`) for a
    /// canonical `X`.
    fn check_node(&mut self, call: &impl ToTokens, node: &syn::Expr) {
        let syn::Expr::Call(c) = node else { return };
        let syn::Expr::Path(p) = &*c.func else { return };
        let Some(last) = p.path.segments.last() else {
            return;
        };
        let name = last.ident.to_string();
        if CANONICAL.contains(&name.as_str()) {
            let text = call.to_token_stream().to_string().replace(' ', "");
            let short: String = text.chars().take(100).collect();
            self.raw.push(format!("{name}: {short}"));
        }
    }
}

impl<'ast> Visit<'ast> for Scan {
    fn visit_item_mod(&mut self, m: &'ast syn::ItemMod) {
        if !is_test_item(&m.attrs) {
            syn::visit::visit_item_mod(self, m);
        }
    }

    fn visit_item_fn(&mut self, f: &'ast syn::ItemFn) {
        if !is_test_item(&f.attrs) {
            syn::visit::visit_item_fn(self, f);
        }
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        if !is_test_item(&i.attrs) {
            syn::visit::visit_item_impl(self, i);
        }
    }

    fn visit_impl_item_fn(&mut self, f: &'ast syn::ImplItemFn) {
        if !is_test_item(&f.attrs) {
            syn::visit::visit_impl_item_fn(self, f);
        }
    }

    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        if m.method == "intern"
            && m.args.len() == 1
            && let Some(arg) = m.args.first()
        {
            self.check_node(m, arg);
        }
        syn::visit::visit_expr_method_call(self, m);
    }

    fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
        // `Arena::intern(arena, node)`.
        if let syn::Expr::Path(p) = &*c.func
            && p.path.segments.last().is_some_and(|s| s.ident == "intern")
            && c.args.len() == 2
            && let Some(arg) = c.args.last()
        {
            self.check_node(c, arg);
        }
        syn::visit::visit_expr_call(self, c);
    }

    /// Macro bodies (`vec![arena.intern(…)]`, `format!`) are not part of
    /// the syntax tree: parse those that are comma-separated expressions.
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        let parser = syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated;
        if let Ok(args) = m.parse_body_with(parser) {
            for arg in &args {
                self.visit_expr(arg);
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

fn scan_text(text: &str, what: &str) -> Scan {
    let file =
        syn::parse_file(text).unwrap_or_else(|e| panic!("{what}: does not parse as Rust: {e}"));
    let mut scan = Scan::default();
    scan.visit_file(&file);
    scan
}

#[test]
fn canonical_variants_are_not_interned_raw_outside_base() {
    let src = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let mut files = Vec::new();
    collect_rs_files(src, &mut files);
    files.sort();

    let mut found: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in &files {
        let rel = path
            .strip_prefix(src)
            .expect("file is under src/")
            .to_string_lossy()
            .replace('\\', "/");
        if rel.starts_with("base/") {
            continue;
        }
        let text = fs::read_to_string(path).expect("source file is readable");
        let scan = scan_text(&text, &rel);
        if !scan.raw.is_empty() {
            found.insert(rel, scan.raw);
        }
    }

    let allowed: BTreeMap<&str, usize> = ALLOWLIST.iter().copied().collect();
    let mut problems = Vec::new();
    for (file, sites) in &found {
        let a = allowed.get(file.as_str()).copied().unwrap_or(0);
        if sites.len() != a {
            problems.push(format!(
                "{file}: {} raw intern(s) of canonical variants, allowlisted {a} — build with \
                 `Arena::make` or the constructor:\n    {}",
                sites.len(),
                sites.join("\n    ")
            ));
        }
    }
    for (file, &a) in &allowed {
        if !found.contains_key(*file) {
            problems.push(format!(
                "{file}: 0 raw interns, allowlisted {a} — tighten the allowlist"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "canonical-construction ratchet:\n{}",
        problems.join("\n")
    );
}

#[test]
fn raw_intern_detection() {
    let text = r#"
        fn f(arena: &mut Arena, a: ExprId, b: ExprId, p: Pairs) {
            let s = arena.intern(ExprNode::FiniteSet(smallvec![a, b]));
            let u = arena.intern(crate::base::node::ExprNode::Piecewise(p));
            let v = vec![arena.intern(ExprNode::Mul(smallvec![a, b]))];
            let w = Arena::intern(arena, ExprNode::Eq_(a, b));
            // Not canonical variants, or not raw:
            let x = arena.intern(ExprNode::Sin(a));
            let y = arena.make(ExprNode::Add(smallvec![a, b]));
            let z = arena.intern(ExprNode::Min(smallvec![a, b]));
        }
        #[cfg(test)]
        mod tests {
            fn g(arena: &mut Arena) { arena.intern(ExprNode::Add(smallvec![])); }
        }
        #[cfg(test)]
        fn h(arena: &mut Arena) { arena.intern(ExprNode::Pow(a, b)); }
        #[test]
        fn k() { arena.intern(ExprNode::Not(a)); }
    "#;
    let scan = scan_text(text, "sample");
    let names: Vec<&str> = scan
        .raw
        .iter()
        .map(|s| s.split(':').next().unwrap_or_default())
        .collect();
    assert_eq!(
        names,
        ["FiniteSet", "Piecewise", "Mul", "Eq_"],
        "{:?}",
        scan.raw
    );
}
