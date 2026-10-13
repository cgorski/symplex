//! Ratchet over `src/` outside `src/base/`: expression ids are never
//! ordered by their allocation number.
//!
//! `ExprId`, `SymbolId` and `NumId` have no `Ord` (0.49), and their number
//! is private to `base`, so `ids.sort()` and `ids.sort_by_key(|id| id.0)`
//! no longer compile.  Two escape hatches remain, and this test keeps them
//! visible:
//!
//! * `ExprId::index()` — the allocation number, for hashing, memo keys and
//!   side tables.  Its uses are counted per file (exact counts, like the
//!   other ratchets), and a call inside the closure of an ordering method
//!   (`sort_by_key`, `min_by`, `binary_search_by_key`, …) is always an
//!   error: allocation order depends on everything the context built
//!   before, so a result ordered by it changes with the context's history
//!   (`factor_all(x³ − y³)` printed `−(−x + y)(x² + xy + y²)` when `y` had
//!   been created before `x`).
//! * `ArenaOrder(id)` — an explicitly named allocation order, for dedup and
//!   set identity whose order never reaches output.  Its constructions are
//!   counted per file.
//!
//! An order that can reach output is `Arena::canonical_cmp` (sort key, then
//! a structural tie-break).  See CONTRIBUTING.md, "Identifiers have no
//! order".

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::visit::Visit;

/// Files allowed to call `ExprId::index()`, with the exact count.  Paths
/// are relative to `src/`.
const INDEX_ALLOWLIST: &[(&str, usize)] = &[
    // `Ex::equals`' sampling seed, mixed from the two ids (see the
    // found-not-fixed note in the 0.49 report: a structural seed is wave 2).
    ("api/expr_ops.rs", 2),
    // The name of a fresh symbol standing for a geometric base: identity only.
    ("calculus/z_transform.rs", 1),
    // Residue hashing of generators for the modular zero test: identity only.
    ("poly/polybridge.rs", 4),
    // Hash of a symbol picking its test values in the zero test.
    ("simplify/identically_zero.rs", 1),
    // Quadrature failure memo key.
    ("transforms/evalf/quad.rs", 4),
];

/// Files allowed to construct `ArenaOrder`, with the exact count.
const ARENA_ORDER_ALLOWLIST: &[(&str, usize)] = &[
    // `Mono` node lists: a monomial's identity, merged in one fixed order.
    ("transforms/evalf/extended.rs", 2),
    // The substitution scopes: a sorted pair list used as a hash key.
    ("transforms/subs.rs", 2),
];

/// Methods whose closure argument decides an order.
const ORDERING_METHODS: &[&str] = &[
    "sort_by",
    "sort_by_key",
    "sort_by_cached_key",
    "sort_unstable_by",
    "sort_unstable_by_key",
    "min_by",
    "min_by_key",
    "max_by",
    "max_by_key",
    "binary_search_by",
    "binary_search_by_key",
    "select_nth_unstable_by",
    "select_nth_unstable_by_key",
    "is_sorted_by",
    "is_sorted_by_key",
    "partition_point",
];

#[derive(Default)]
struct Scan {
    index_calls: usize,
    arena_order: usize,
    in_ordering_closure: usize,
    violations: Vec<String>,
}

impl<'ast> Visit<'ast> for Scan {
    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        let name = m.method.to_string();
        if name == "index" && m.args.is_empty() {
            self.index_calls += 1;
            if self.in_ordering_closure > 0 {
                self.violations
                    .push(m.to_token_stream().to_string().replace(' ', ""));
            }
        }
        if ORDERING_METHODS.contains(&name.as_str()) {
            self.visit_expr(&m.receiver);
            self.in_ordering_closure += 1;
            for arg in &m.args {
                self.visit_expr(arg);
            }
            self.in_ordering_closure -= 1;
        } else {
            syn::visit::visit_expr_method_call(self, m);
        }
    }

    /// Macro bodies (`format!("{}", id.index())`, `vec![…]`) are not part
    /// of the syntax tree: parse those that are comma-separated expressions.
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        let parser = syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated;
        if let Ok(args) = m.parse_body_with(parser) {
            for arg in &args {
                self.visit_expr(arg);
            }
        }
    }

    fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*c.func
            && p.path
                .segments
                .last()
                .is_some_and(|s| s.ident == "ArenaOrder")
        {
            self.arena_order += 1;
        }
        syn::visit::visit_expr_call(self, c);
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

fn scan_file(path: &Path) -> Scan {
    let text = fs::read_to_string(path).expect("source file is readable");
    let file = syn::parse_file(&text)
        .unwrap_or_else(|e| panic!("{}: does not parse as Rust: {e}", path.display()));
    let mut scan = Scan::default();
    scan.visit_file(&file);
    scan
}

fn check_counts(
    what: &str,
    allowlist: &[(&str, usize)],
    found: &BTreeMap<String, usize>,
    problems: &mut Vec<String>,
) {
    let allowed: BTreeMap<&str, usize> = allowlist.iter().copied().collect();
    for (file, &n) in found {
        let a = allowed.get(file.as_str()).copied().unwrap_or(0);
        if n != a {
            problems.push(format!(
                "{file}: {n} {what}, allowlisted {a} — order ids with `Arena::canonical_cmp`; \
                 update the allowlist only for identity or hashing uses"
            ));
        }
    }
    for (file, &a) in &allowed {
        if !found.contains_key(*file) {
            problems.push(format!(
                "{file}: 0 {what}, allowlisted {a} — tighten the allowlist"
            ));
        }
    }
}

#[test]
fn ids_are_not_ordered_by_allocation_outside_base() {
    let src = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let mut files = Vec::new();
    collect_rs_files(src, &mut files);
    files.sort();

    let mut index_calls = BTreeMap::new();
    let mut arena_order = BTreeMap::new();
    let mut problems = Vec::new();
    for path in &files {
        let rel = path
            .strip_prefix(src)
            .expect("file is under src/")
            .to_string_lossy()
            .replace('\\', "/");
        if rel.starts_with("base/") {
            continue;
        }
        let scan = scan_file(path);
        if scan.index_calls > 0 {
            index_calls.insert(rel.clone(), scan.index_calls);
        }
        if scan.arena_order > 0 {
            arena_order.insert(rel.clone(), scan.arena_order);
        }
        for v in scan.violations {
            problems.push(format!("{rel}: `{v}` orders by allocation number"));
        }
    }
    check_counts(
        "`ExprId::index()` call(s)",
        INDEX_ALLOWLIST,
        &index_calls,
        &mut problems,
    );
    check_counts(
        "`ArenaOrder` construction(s)",
        ARENA_ORDER_ALLOWLIST,
        &arena_order,
        &mut problems,
    );
    assert!(
        problems.is_empty(),
        "id-order ratchet:\n{}",
        problems.join("\n")
    );
}

#[test]
fn ordering_closure_detection() {
    let text = r#"fn f(v: &mut Vec<ExprId>, a: &Arena) {
        v.sort_by_key(|id| id.index());
        v.sort_by(|&x, &y| a.canonical_cmp(x, y));
        let h = v[0].index();
        v.dedup_by_key(|id| ArenaOrder(*id));
        let name = format!("s{}", v[1].index());
    }"#;
    let file = syn::parse_file(text).expect("parses");
    let mut scan = Scan::default();
    scan.visit_file(&file);
    assert_eq!(scan.index_calls, 3);
    assert_eq!(scan.violations.len(), 1);
    assert_eq!(scan.arena_order, 1);
}
