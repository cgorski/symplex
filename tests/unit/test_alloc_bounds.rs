//! Ratchet over `src/`: allocations whose length is a cast of a value read
//! from the data must not grow beyond their allowlisted count per file; an
//! allowlisted file must not lose one without the allowlist being
//! tightened.  See CONTRIBUTING.md, "Exact arithmetic has size bounds".
//!
//! A length taken from the data (a degree, an exponent, a valuation, the
//! number of terms of a sum) that sizes an allocation asks the allocator
//! for whatever the input says, and an allocation failure aborts the
//! process — nothing can catch it.  Up to 0.56 the dense coefficient vector
//! of `x^(9·10¹⁰)` (a Weibull shape) asked for 360 GB, the series of
//! `exp(−x^N)` for 137 GB; `inverse_z_transform` sized a list by the pole
//! order of `z/(z − 1)^N`.  New code bounds the length with
//! `base::cast::bounded_len` (refused beyond a cap) or `clamped_len` (a
//! precision the caller lowers, a capacity hint).
//!
//! The count is syntactic (no types are available to a linter): a call
//!
//! - `….with_capacity(len)` / `with_capacity_and_hasher(len, …)` (any path
//!   or method),
//! - `vec![elem; len]`,
//! - `….resize(len, …)` / `resize_with(len, …)`,
//!
//! whose `len` contains an `as usize` cast, a `to_usize()` call, or a
//! `try_into()` / `try_from(…)` conversion outside a call to
//! `bounded_len(…)` / `clamped_len(…)`.  A length computed into a `let`
//! first is not seen: the review of a migrated site checks that the `let`
//! goes through `bounded_len` too.
//!
//! `#[cfg(test)]` items and modules are skipped.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::visit::Visit;

/// Files allowed to contain cast-sized allocations, with the exact count
/// (0.58: 4, down from 37 at 0.57).  Paths are relative to `src/`.  Each
/// entry says where its lengths are bounded.
const ALLOWLIST: &[(&str, usize)] = &[
    // The dense images of Brown's gcd (`image_degree_bounds`, `run_poly`,
    // `lc_main`) are sized by degrees of the inputs, which `brown_gcd`
    // bounds by `cast::MAX_DENSE_LIST` on entry.
    ("poly/brown.rs", 4),
];

/// Functions whose arguments are bounded lengths: casts inside them are
/// the checked conversion itself.
const BOUNDERS: &[&str] = &["bounded_len", "clamped_len"];

fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.path().is_ident("cfg") && a.to_token_stream().to_string().contains("test"))
}

fn site(e: &impl ToTokens) -> String {
    let s = e.to_token_stream().to_string();
    if s.chars().count() > 120 {
        format!("{}…", s.chars().take(120).collect::<String>())
    } else {
        s
    }
}

/// Does the length expression `e` contain an unchecked conversion?
fn has_cast(e: &syn::Expr) -> bool {
    struct Casts(bool);
    impl<'ast> Visit<'ast> for Casts {
        fn visit_expr_cast(&mut self, c: &'ast syn::ExprCast) {
            if c.ty.to_token_stream().to_string() == "usize" {
                self.0 = true;
            }
            syn::visit::visit_expr_cast(self, c);
        }
        fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
            let name = m.method.to_string();
            if BOUNDERS.contains(&name.as_str()) {
                return;
            }
            if matches!(name.as_str(), "to_usize" | "try_into" | "try_from") {
                self.0 = true;
            }
            syn::visit::visit_expr_method_call(self, m);
        }
        fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
            if let syn::Expr::Path(p) = &*c.func
                && let Some(last) = p.path.segments.last()
            {
                let name = last.ident.to_string();
                if BOUNDERS.contains(&name.as_str()) {
                    return;
                }
                if matches!(name.as_str(), "try_from" | "try_into" | "to_usize") {
                    self.0 = true;
                }
            }
            syn::visit::visit_expr_call(self, c);
        }
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
    }
    let mut v = Casts(false);
    v.visit_expr(e);
    v.0
}

#[derive(Default)]
struct Finder {
    sites: Vec<String>,
}

impl Finder {
    fn check(&mut self, what: &str, len: &syn::Expr, whole: &impl ToTokens) {
        if has_cast(len) {
            self.sites.push(format!("{what}: {}", site(whole)));
        }
    }
}

/// `(elem, len)` of `vec![elem; len]`.
fn vec_repeat(mac: &syn::Macro) -> Option<(syn::Expr, syn::Expr)> {
    let parser = |input: syn::parse::ParseStream| -> syn::Result<(syn::Expr, syn::Expr)> {
        let elem: syn::Expr = input.parse()?;
        let _semi: syn::Token![;] = input.parse()?;
        let len: syn::Expr = input.parse()?;
        Ok((elem, len))
    };
    mac.parse_body_with(parser).ok()
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

    fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*c.func
            && let Some(last) = p.path.segments.last()
            && last.ident.to_string().starts_with("with_capacity")
            && let Some(len) = c.args.first()
        {
            self.check("with_capacity", len, c);
        }
        syn::visit::visit_expr_call(self, c);
    }

    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        let name = m.method.to_string();
        if (name.starts_with("with_capacity") || name == "resize" || name == "resize_with")
            && let Some(len) = m.args.first()
        {
            self.check(&name, len, m);
        }
        syn::visit::visit_expr_method_call(self, m);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if mac.path.is_ident("vec")
            && let Some((elem, len)) = vec_repeat(mac)
        {
            self.check("vec![_; n]", &len, mac);
            // The element and the length may hold further allocations.
            self.visit_expr(&elem);
            self.visit_expr(&len);
        } else if let Ok(args) = mac.parse_body_with(
            syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
        ) {
            // `format!`, `debug_assert!`, … : look inside the arguments.
            for a in &args {
                self.visit_expr(a);
            }
        }
        syn::visit::visit_macro(self, mac);
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

fn scan_src() -> BTreeMap<String, Vec<String>> {
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
        let text = fs::read_to_string(path).expect("source file is readable");
        let file = syn::parse_file(&text)
            .unwrap_or_else(|e| panic!("{}: does not parse as Rust: {e}", path.display()));
        let mut f = Finder::default();
        f.visit_file(&file);
        if !f.sites.is_empty() {
            found.insert(rel, f.sites);
        }
    }
    found
}

#[test]
fn cast_sized_allocations_do_not_grow() {
    let found = scan_src();
    let allowed: BTreeMap<&str, usize> = ALLOWLIST.iter().copied().collect();
    let mut violations = Vec::new();
    for (rel, sites) in &found {
        let count = sites.len();
        let allowance = allowed.get(rel.as_str()).copied().unwrap_or(0);
        if count > allowance {
            violations.push(format!(
                "{rel}: {count} cast-sized allocation(s), allowed {allowance} — bound the length \
                 with `base::cast::bounded_len` (refuse beyond a cap) or `clamped_len` \
                 (CONTRIBUTING.md, \"Exact arithmetic has size bounds\"):\n    {}",
                sites.join("\n    ")
            ));
        } else if count < allowance {
            violations.push(format!(
                "{rel}: {count} cast-sized allocation(s), allowed {allowance} — tighten the allowlist"
            ));
        }
    }
    for (rel, &allowance) in &allowed {
        if !found.contains_key(*rel) {
            violations.push(format!(
                "{rel}: allowlisted ({allowance}) but none found under src/ — tighten the allowlist"
            ));
        }
    }
    if !violations.is_empty() {
        panic!(
            "allocation-bound ratchet: {} file(s) out of step:\n\n{}\n",
            violations.len(),
            violations.join("\n\n")
        );
    }
}

/// The detector itself: each form is found, a bounded length is not.
#[test]
fn detector_sees_each_form() {
    let src = r#"
        fn f(n: i64, d: u32, b: &BigInt, v: &mut Vec<u8>) {
            let a = Vec::with_capacity(n as usize);
            let c = vec![0u64; d as usize + 1];
            v.resize(b.to_usize().unwrap(), 0);
            let e = String::with_capacity(usize::try_from(n).unwrap());
            let g: Vec<u8> = Vec::with_capacity(n.try_into().unwrap());
            let ok1 = Vec::with_capacity(cast::bounded_len(n, 10).unwrap_or(0));
            let ok2 = vec![0u8; crate::base::cast::clamped_len(n, 16)];
            let ok3 = Vec::with_capacity(v.len() + 1);
            let h = format!("{:?}", vec![1u8; n as usize]);
        }
        #[cfg(test)]
        mod tests {
            fn t(n: i64) { let _ = Vec::with_capacity(n as usize); }
        }
    "#;
    let file = syn::parse_file(src).expect("parses");
    let mut f = Finder::default();
    f.visit_file(&file);
    assert_eq!(f.sites.len(), 6, "{:#?}", f.sites);
}
