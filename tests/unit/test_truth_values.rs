//! Ratchet over `src/`: three-valued decisions as `Option<bool>` and their
//! implicit readings (wave 5, W8c).  See `src/base/truth.rs`, "Reading a
//! `Truth`", and CONTRIBUTING.md, "Type rules that keep bug classes out".
//!
//! Inside the crate a yes / no / undecided answer is a
//! [`Truth`](symplex::base::truth::Truth), read with a method that names
//! the reading (`is_true`: proved; `!is_true()`: not proved; `is_false`:
//! refuted; `!is_false()`: not refuted), and a sign is a
//! [`SignSet`](symplex::base::truth::SignSet).  `Option<bool>` stays only
//! at the public boundary (CONTRIBUTING.md, "The API Model", Pattern 4),
//! converted there with `Truth::from` / `Truth::to_option`.  This test
//! counts, per file, the constructs that hide a reading or a sign encoding:
//!
//! | column | construct |
//! |---|---|
//! | `cmp` | `x == Some(true)`, `x != Some(true)`, `x == Some(false)`, `x != Some(false)` (either operand order) |
//! | `unwrap` | `x.unwrap_or(true)` / `x.unwrap_or(false)`, except map lookups (`m.get(k)[.copied()].unwrap_or(..)`: absent key → default) |
//! | `opt_fn` | a function returning `Option<bool>` that is not public API |
//! | `sign_fn` | a function named `*sign*` (a `_`-separated word `sign`/`signs`) returning `Option<Ordering>`, `Option<i8>`, `Option<i32>`, `Option<i64>`, `Option<bool>` or `Option<Option<bool>>`, not public API — a sign is a `SignSet` |
//! | `u8_key` | the tuple type `(ExprId, u8)` — a memo keyed by hand-picked codes (the `constant_facts` collision, S2 B1) |
//!
//! "Public API" is approximate, as `syn` sees no name resolution: a plain
//! `pub fn` in a module reachable from the crate root (`pub mod`, a `pub
//! use` of the module in `lib.rs`, or a glob / named re-export from its
//! parent), a plain `pub` method of an inherent impl in such a file or in
//! `src/api/` (the `impl Ex` files), or a method of a public trait.  Trait
//! *implementations* are not signatures of their own and are skipped for
//! the two `fn` columns; `#[cfg(test)]` items are skipped entirely.  The
//! `unwrap` column is a textual heuristic (no types): it cannot tell an
//! `Option<bool>` from `opt.map(pred)`, and both hide a reading.
//! Macro bodies are scanned as expressions when they parse as a
//! comma-separated list, otherwise as token text (so `macro_rules!` bodies
//! count too).
//!
//! The counts are exact: a file may not gain a site, and a file that loses
//! one must have its entry tightened (the failure message prints the table
//! to paste).  Files without an entry must have no site.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::punctuated::Punctuated;
use syn::visit::Visit;

/// Per-file counts `[cmp, unwrap, opt_fn, sign_fn, u8_key]` (the columns of
/// the module doc).  Paths are relative to `src/`; files with no site have
/// no entry.  Fixed at the base of wave 5 (c6bd5ce: 75 files; 559 `cmp`,
/// 8 `unwrap`, 101 `opt_fn`, 32 `sign_fn`, 0 `u8_key`) and tightened as the
/// sites were migrated (W8c: 33 files; 271, 5, 50, 26, 0).
const ALLOWLIST: &[(&str, [usize; 5])] = &[
    // ── Files owned by other steps of wave 5 (current counts) ─────────────
    // `base/{arena,canon}.rs` (W12, byte identity), `base/{assumptions,complex}.rs`,
    // `poly/**` (W10c), `transforms/{eval,evalf,inequalities,sets,solve}` (B3 / evalf).
    ("base/arena.rs", [1, 0, 1, 1, 0]),
    ("base/assumptions.rs", [110, 0, 0, 0, 0]),
    ("base/canon.rs", [47, 2, 1, 3, 0]),
    ("base/complex.rs", [27, 0, 2, 1, 0]),
    ("poly/algebraic.rs", [1, 0, 2, 2, 0]),
    ("poly/dense.rs", [0, 0, 1, 0, 0]),
    ("poly/factor_hensel.rs", [1, 0, 0, 0, 0]),
    ("poly/factor_zassenhaus.rs", [1, 0, 0, 0, 0]),
    ("poly/numfield.rs", [2, 0, 0, 0, 0]),
    ("poly/polysys.rs", [5, 0, 2, 0, 0]),
    ("poly/sturm.rs", [0, 0, 0, 1, 0]),
    ("poly/zerodim.rs", [1, 0, 0, 0, 0]),
    ("transforms/eval.rs", [23, 0, 8, 2, 0]),
    ("transforms/evalf.rs", [0, 1, 0, 0, 0]),
    ("transforms/inequalities.rs", [1, 2, 0, 1, 0]),
    ("transforms/sets.rs", [18, 0, 18, 0, 0]),
    ("transforms/solve.rs", [27, 0, 1, 0, 0]),
    ("transforms/solve/algebraic_roots.rs", [0, 0, 0, 1, 0]),
    ("transforms/solve/context.rs", [4, 0, 0, 0, 0]),
    // `risch/param_rational.rs`: kept for the planned `PFrac` rewrite (the site is in `nonnegative(&PFrac)`).
    ("calculus/risch/param_rational.rs", [1, 0, 0, 0, 0]),
    // ── Kept, with the reason ──────────────────────────────────────────────
    // `is_convergent`: `Arena::is_convergent_expr` (arena.rs) returns it as `Option<bool>`.
    ("calculus/convergence.rs", [0, 0, 1, 0, 0]),
    // `sign_of -> Option<Ordering>` gives `±∞` a sign; `SignSet` is for finite values
    // (truth.rs, "Infinities"): needs a typed extended sign.
    ("calculus/definite.rs", [0, 0, 0, 1, 0]),
    // `param_sign`: `limit::const_sign` of a transform parameter (see limit.rs).
    ("calculus/fourier_transform.rs", [0, 0, 0, 1, 0]),
    ("calculus/laplace.rs", [0, 0, 0, 1, 0]),
    ("calculus/mellin.rs", [0, 0, 0, 1, 0]),
    // `eventual_sign_at_inf -> Option<i32>`: the sign of the leading coefficient (`−1` for
    // `−x + i`), not a `SignSet` atom.
    ("calculus/gruntz.rs", [0, 0, 0, 1, 0]),
    // `const_sign`/`const_sign_depth`/`numeric_sign`/`assumption_sign` sign `±∞` and are
    // multiplied as `i32` at ~50 sites; `re_sign_of`/`im_sign_of`/`eventual_sign_along`
    // build on them and on `eventual_sign_at_inf`.
    ("calculus/limit.rs", [0, 0, 0, 7, 0]),
    // `tower_logarithmic_part_is_non_elementary`: `None` is a zero denominator (a
    // precondition), not "undecided".
    ("calculus/risch/tower_integrate.rs", [0, 0, 1, 0, 0]),
    // The radical tower (`Tower::decide`/`is_zero`/`real_sign`/`real_constant_sign`,
    // `algebraic_function_is_zero`) was planned for W10c (not done).
    ("domains/linalg.rs", [1, 0, 3, 2, 0]),
    // `atan_sum::is_zero`, `cyclotomic::{is_zero, is_real}` and the `Tower` helpers:
    // their caller is eval.rs.
    ("transforms/eval/atan_sum.rs", [0, 0, 1, 0, 0]),
    ("transforms/eval/cyclotomic.rs", [0, 0, 4, 0, 0]),
    // `holds_admissible_integer`: `?`-propagated into `univariate_status`'s `Option`.
    ("transforms/logic.rs", [0, 0, 1, 0, 0]),
    // `constant_realness*`: called from solve.rs and sets.rs.
    ("transforms/realness.rs", [0, 0, 3, 0, 0]),
];

const COLUMNS: [&str; 5] = ["cmp", "unwrap", "opt_fn", "sign_fn", "u8_key"];

/// Which items of a module are reachable from the crate root.
#[derive(Clone, Debug)]
enum Reach {
    /// Every plain-`pub` item (`pub mod`, `pub use m;`, `pub use m::*`).
    All,
    /// Only these names (`pub use m::{A, B}`).
    Named(BTreeSet<String>),
}

fn module_file(src: &Path, module: &str) -> Option<PathBuf> {
    let flat = src.join(format!("{module}.rs"));
    if flat.is_file() {
        return Some(flat);
    }
    let nested = src.join(module).join("mod.rs");
    nested.is_file().then_some(nested)
}

fn parse(path: &Path) -> syn::File {
    let text = fs::read_to_string(path).expect("source file is readable");
    syn::parse_file(&text)
        .unwrap_or_else(|e| panic!("{}: does not parse as Rust: {e}", path.display()))
}

fn is_plain_pub(vis: &syn::Visibility) -> bool {
    matches!(vis, syn::Visibility::Public(_))
}

fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.path().is_ident("cfg") && a.to_token_stream().to_string().contains("test"))
}

/// The re-exports `pub use first::rest…` of one module file: for each
/// direct child module named first, either a glob / module re-export
/// (`None`) or the re-exported names.
fn child_reexports(items: &[syn::Item]) -> BTreeMap<String, Option<BTreeSet<String>>> {
    fn leaf_names(tree: &syn::UseTree, out: &mut BTreeSet<String>, glob: &mut bool) {
        match tree {
            syn::UseTree::Path(p) => leaf_names(&p.tree, out, glob),
            syn::UseTree::Name(n) => {
                out.insert(n.ident.to_string());
            }
            syn::UseTree::Rename(r) => {
                out.insert(r.ident.to_string());
            }
            syn::UseTree::Glob(_) => *glob = true,
            syn::UseTree::Group(g) => {
                for t in &g.items {
                    leaf_names(t, out, glob);
                }
            }
        }
    }
    let mut out: BTreeMap<String, Option<BTreeSet<String>>> = BTreeMap::new();
    for item in items {
        let syn::Item::Use(u) = item else { continue };
        if !is_plain_pub(&u.vis) {
            continue;
        }
        let mut tree = &u.tree;
        if let syn::UseTree::Path(p) = tree
            && p.ident == "self"
        {
            tree = &p.tree;
        }
        let syn::UseTree::Path(p) = tree else {
            continue;
        };
        let child = p.ident.to_string();
        let mut names = BTreeSet::new();
        let mut glob = false;
        leaf_names(&p.tree, &mut names, &mut glob);
        let entry = out.entry(child).or_insert_with(|| Some(BTreeSet::new()));
        if glob {
            *entry = None;
        } else if let Some(set) = entry {
            set.extend(names);
        }
    }
    out
}

/// Module path (`calculus/definite`) → what of it is reachable from the
/// crate root.
fn public_modules(src: &Path) -> BTreeMap<String, Reach> {
    let mut reach: BTreeMap<String, Reach> = BTreeMap::new();
    let lib = parse(&src.join("lib.rs"));
    for item in &lib.items {
        match item {
            syn::Item::Mod(m) if is_plain_pub(&m.vis) && m.content.is_none() => {
                reach.insert(m.ident.to_string(), Reach::All);
            }
            syn::Item::Use(u) if is_plain_pub(&u.vis) => {
                // `pub use a::b;` / `pub use a::b as c;` of a module file.
                let mut segments = Vec::new();
                let mut tree = &u.tree;
                loop {
                    match tree {
                        syn::UseTree::Path(p) => {
                            segments.push(p.ident.to_string());
                            tree = &p.tree;
                        }
                        syn::UseTree::Name(n) => {
                            segments.push(n.ident.to_string());
                            break;
                        }
                        syn::UseTree::Rename(r) => {
                            segments.push(r.ident.to_string());
                            break;
                        }
                        _ => {
                            segments.clear();
                            break;
                        }
                    }
                }
                let module = segments.join("/");
                if !module.is_empty() && module_file(src, &module).is_some() {
                    reach.insert(module, Reach::All);
                }
            }
            _ => {}
        }
    }
    let mut queue: Vec<String> = reach.keys().cloned().collect();
    while let Some(module) = queue.pop() {
        if !matches!(reach.get(&module), Some(Reach::All)) {
            continue;
        }
        let Some(file) = module_file(src, &module) else {
            continue;
        };
        let parsed = parse(&file);
        let reexports = child_reexports(&parsed.items);
        for item in &parsed.items {
            let syn::Item::Mod(m) = item else { continue };
            if m.content.is_some() || is_cfg_test(&m.attrs) {
                continue;
            }
            let child = format!("{module}/{}", m.ident);
            let r = if is_plain_pub(&m.vis) {
                Some(Reach::All)
            } else {
                match reexports.get(&m.ident.to_string()) {
                    Some(None) => Some(Reach::All),
                    Some(Some(names)) => Some(Reach::Named(names.clone())),
                    None => None,
                }
            };
            if let Some(r) = r
                && !reach.contains_key(&child)
            {
                if matches!(r, Reach::All) {
                    queue.push(child.clone());
                }
                reach.insert(child, r);
            }
        }
    }
    reach
}

fn is_some_bool(e: &syn::Expr) -> bool {
    let syn::Expr::Call(call) = e else {
        return false;
    };
    let syn::Expr::Path(p) = &*call.func else {
        return false;
    };
    p.path.is_ident("Some")
        && call.args.len() == 1
        && matches!(
            &call.args[0],
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Bool(_),
                ..
            })
        )
}

/// `m.get(k).unwrap_or(..)`, `m.get(k).copied().unwrap_or(..)`: an absent
/// key read as a default, not a three-valued decision.
fn is_map_lookup(receiver: &syn::Expr) -> bool {
    let mut r = receiver;
    while let syn::Expr::MethodCall(m) = r {
        let name = m.method.to_string();
        if name == "get" || name == "get_mut" {
            return true;
        }
        if name == "copied" || name == "cloned" {
            r = &m.receiver;
            continue;
        }
        return false;
    }
    false
}

fn type_text(t: &syn::Type) -> String {
    t.to_token_stream().to_string().replace(' ', "")
}

fn is_option_bool(t: &syn::Type) -> bool {
    matches!(
        type_text(t).as_str(),
        "Option<bool>" | "std::option::Option<bool>"
    )
}

fn is_sign_encoding(t: &syn::Type) -> bool {
    matches!(
        type_text(t).as_str(),
        "Option<Ordering>"
            | "Option<std::cmp::Ordering>"
            | "Option<cmp::Ordering>"
            | "Option<i8>"
            | "Option<i32>"
            | "Option<i64>"
            | "Option<bool>"
            | "Option<Option<bool>>"
    )
}

fn is_sign_name(name: &str) -> bool {
    name.split('_').any(|w| w == "sign" || w == "signs")
}

#[derive(Default)]
struct Counts {
    sites: [Vec<String>; 5],
}

struct Scanner<'a> {
    file_reach: Option<&'a Reach>,
    under_api: bool,
    /// Depth of function bodies being visited: a nested `fn` is internal.
    fn_depth: usize,
    /// Inside an inline `mod` that is not plain `pub`.
    hidden_mod: usize,
    counts: Counts,
}

impl Scanner<'_> {
    fn reachable(&self, name: &str) -> bool {
        if self.fn_depth > 0 || self.hidden_mod > 0 {
            return false;
        }
        match self.file_reach {
            Some(Reach::All) => true,
            Some(Reach::Named(names)) => names.contains(name),
            None => false,
        }
    }

    fn signature(&mut self, sig: &syn::Signature, api: bool, line_label: &str) {
        let syn::ReturnType::Type(_, ty) = &sig.output else {
            return;
        };
        if api {
            return;
        }
        let name = sig.ident.to_string();
        if is_option_bool(ty) {
            self.counts.sites[2].push(format!("{line_label}fn {name} -> Option<bool>"));
        }
        if is_sign_name(&name) && is_sign_encoding(ty) {
            self.counts.sites[3].push(format!("{line_label}fn {name} -> {}", type_text(ty)));
        }
    }

    /// Token-text scan of a macro body that is not a list of expressions.
    fn scan_tokens(&mut self, text: &str) {
        let s: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        for pat in [
            "==Some(true)",
            "!=Some(true)",
            "==Some(false)",
            "!=Some(false)",
            "Some(true)==",
            "Some(true)!=",
            "Some(false)==",
            "Some(false)!=",
        ] {
            for _ in s.matches(pat) {
                self.counts.sites[0].push(format!("macro tokens: {pat}"));
            }
        }
        for pat in [".unwrap_or(true)", ".unwrap_or(false)"] {
            for (pos, _) in s.match_indices(pat) {
                if !text_is_map_lookup(&s[..pos]) {
                    self.counts.sites[1].push(format!("macro tokens: {pat}"));
                }
            }
        }
    }
}

/// The token-text form of [`is_map_lookup`]: `before` ends with
/// `.get(…)`, optionally followed by `.copied()` / `.cloned()`.
fn text_is_map_lookup(before: &str) -> bool {
    let mut b = before;
    for suffix in [".copied()", ".cloned()"] {
        if let Some(rest) = b.strip_suffix(suffix) {
            b = rest;
        }
    }
    let Some(inner) = b.strip_suffix(')') else {
        return false;
    };
    let mut depth = 1usize;
    for (i, c) in inner.char_indices().rev() {
        match c {
            ')' => depth += 1,
            '(' => {
                depth -= 1;
                if depth == 0 {
                    let head = &inner[..i];
                    return head.ends_with(".get") || head.ends_with(".get_mut");
                }
            }
            _ => {}
        }
    }
    false
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

impl<'ast> Visit<'ast> for Scanner<'_> {
    fn visit_item(&mut self, i: &'ast syn::Item) {
        if is_cfg_test(item_attrs(i)) {
            return;
        }
        syn::visit::visit_item(self, i);
    }

    fn visit_item_mod(&mut self, m: &'ast syn::ItemMod) {
        let hidden = !is_plain_pub(&m.vis);
        if hidden {
            self.hidden_mod += 1;
        }
        syn::visit::visit_item_mod(self, m);
        if hidden {
            self.hidden_mod -= 1;
        }
    }

    fn visit_item_fn(&mut self, f: &'ast syn::ItemFn) {
        let api = is_plain_pub(&f.vis) && self.reachable(&f.sig.ident.to_string());
        self.signature(&f.sig, api, "");
        self.fn_depth += 1;
        syn::visit::visit_item_fn(self, f);
        self.fn_depth -= 1;
    }

    fn visit_item_impl(&mut self, imp: &'ast syn::ItemImpl) {
        let self_name = match &*imp.self_ty {
            syn::Type::Path(p) => p
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default(),
            _ => String::new(),
        };
        let type_public = self.under_api || self.reachable(&self_name);
        for it in &imp.items {
            match it {
                syn::ImplItem::Fn(m) => {
                    if is_cfg_test(&m.attrs) {
                        continue;
                    }
                    if imp.trait_.is_none() {
                        let api = is_plain_pub(&m.vis) && type_public;
                        self.signature(&m.sig, api, &format!("impl {self_name}::"));
                    }
                    self.fn_depth += 1;
                    self.visit_block(&m.block);
                    self.fn_depth -= 1;
                }
                other => syn::visit::visit_impl_item(self, other),
            }
        }
    }

    fn visit_item_trait(&mut self, t: &'ast syn::ItemTrait) {
        let api = is_plain_pub(&t.vis) && (self.under_api || self.reachable(&t.ident.to_string()));
        for it in &t.items {
            if let syn::TraitItem::Fn(m) = it {
                if is_cfg_test(&m.attrs) {
                    continue;
                }
                self.signature(&m.sig, api, &format!("trait {}::", t.ident));
                if let Some(block) = &m.default {
                    self.fn_depth += 1;
                    self.visit_block(block);
                    self.fn_depth -= 1;
                }
            } else {
                syn::visit::visit_trait_item(self, it);
            }
        }
    }

    fn visit_expr_binary(&mut self, b: &'ast syn::ExprBinary) {
        if matches!(b.op, syn::BinOp::Eq(_) | syn::BinOp::Ne(_))
            && (is_some_bool(&b.left) || is_some_bool(&b.right))
        {
            self.counts.sites[0].push(b.to_token_stream().to_string());
        }
        syn::visit::visit_expr_binary(self, b);
    }

    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        if m.method == "unwrap_or"
            && m.args.len() == 1
            && matches!(
                &m.args[0],
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Bool(_),
                    ..
                })
            )
            && !is_map_lookup(&m.receiver)
        {
            self.counts.sites[1].push(m.to_token_stream().to_string());
        }
        syn::visit::visit_expr_method_call(self, m);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        match mac.parse_body_with(Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated) {
            Ok(exprs) => {
                for e in &exprs {
                    self.visit_expr(e);
                }
            }
            Err(_) => self.scan_tokens(&mac.tokens.to_string()),
        }
    }

    fn visit_type_tuple(&mut self, t: &'ast syn::TypeTuple) {
        let elems: Vec<String> = t.elems.iter().map(type_text).collect();
        if elems.len() == 2 && elems[0].ends_with("ExprId") && elems[1] == "u8" {
            self.counts.sites[4].push(t.to_token_stream().to_string().replace(' ', ""));
        }
        syn::visit::visit_type_tuple(self, t);
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

/// The module path of a file relative to `src/`: `a/b.rs` and `a/b/mod.rs`
/// are both `a/b`.
fn module_of(rel: &str) -> String {
    let stem = rel.strip_suffix(".rs").unwrap_or(rel);
    stem.strip_suffix("/mod").unwrap_or(stem).to_string()
}

#[test]
fn truth_values_and_sign_encodings_do_not_grow() {
    let src = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let reach = public_modules(src);
    let mut files = Vec::new();
    collect_rs_files(src, &mut files);
    files.sort();

    let mut found: BTreeMap<String, Counts> = BTreeMap::new();
    for path in &files {
        let rel = path
            .strip_prefix(src)
            .expect("file is under src/")
            .to_string_lossy()
            .replace('\\', "/");
        let module = module_of(&rel);
        let mut scanner = Scanner {
            file_reach: reach.get(&module),
            under_api: rel.starts_with("api/"),
            fn_depth: 0,
            hidden_mod: 0,
            counts: Counts::default(),
        };
        scanner.visit_file(&parse(path));
        found.insert(rel, scanner.counts);
    }

    let allowed: BTreeMap<&str, [usize; 5]> = ALLOWLIST.iter().copied().collect();
    let mut violations = Vec::new();
    let mut table = Vec::new();
    for (rel, counts) in &found {
        let have: Vec<usize> = counts.sites.iter().map(Vec::len).collect();
        if have.iter().any(|&n| n > 0) {
            table.push(format!(
                "    (\"{rel}\", [{}]),",
                have.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let allowance = allowed.get(rel.as_str()).copied().unwrap_or([0; 5]);
        for (col, (&n, &a)) in have.iter().zip(allowance.iter()).enumerate() {
            if n > a {
                violations.push(format!(
                    "{rel}: {n} `{}` site(s), allowed {a} — read a `Truth` with a named method, \
                     return `Truth`/`SignSet` from internal deciders (src/base/truth.rs):\n    {}",
                    COLUMNS[col],
                    counts.sites[col].join("\n    ")
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
            "truth-value ratchet: {} violation(s):\n\n{}\n\nCurrent table [cmp, unwrap, opt_fn, sign_fn, u8_key]:\n{}\n",
            violations.len(),
            violations.join("\n\n"),
            table.join("\n")
        );
    }
}
