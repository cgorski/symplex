//! Ratchet over `src/`: the public surface — `pub`/`pub(crate)` function
//! signatures, `pub` struct fields, `pub` type aliases, consts and statics —
//! must not gain a *homogeneous tuple type* `(T, T, …)` beyond its
//! allowlisted count per file, and an allowlisted file must not lose one
//! without the allowlist being tightened.  See CONTRIBUTING.md, "Tuples
//! versus structs".
//!
//! A homogeneous tuple is one whose elements all have the same type, so
//! nothing but memory says which position is which: `(lower, upper)`,
//! `(Q, R)`, `(shape, scale)`.  Each remaining site below is either a
//! universal convention that is pattern-matched at every use (`(x, y)`
//! points, `(numer, denom)`, `(var, value)` substitution pairs), an
//! ecosystem convention (`shape() -> (rows, cols)`), or symmetric (the two
//! squares in `n = a² + b²`).  Anything else becomes a struct.
//!
//! Enum tuple variants (`ExprNode::Pow(base, exp)`) are out of scope: they
//! are always destructured by a pattern that names each field at the site.
//! Trait *implementations* are skipped (their signatures belong to the
//! trait, which is checked); `#[cfg(test)]` items and modules are skipped.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::visit::Visit;

/// Files allowed to expose homogeneous tuples, with the exact count and
/// the reason each site is kept.  Paths are relative to `src/`.
const ALLOWLIST: &[(&str, usize)] = &[
    // 0.15 transition: the counts below are the pre-policy inventory.  Each site
    // is being converted to a struct or justified here; see CHANGELOG 0.15.0.
    ("api/expr.rs", 1),
    ("api/expr_complex.rs", 2),
    ("api/expr_funcs.rs", 6),
    ("api/expr_integrate_ext.rs", 1),
    ("api/expr_ops.rs", 3),
    ("api/expr_poly_ext.rs", 6),
    ("api/expr_rules_ext.rs", 1),
    ("api/expr_solve_ext.rs", 1),
    ("api/poly_ex.rs", 2),
    ("base/arena.rs", 6),
    ("base/bigcomplex.rs", 1),
    ("base/canon.rs", 1),
    ("base/complex.rs", 1),
    ("base/interval.rs", 1),
    ("calculus/calculus_util.rs", 4),
    ("calculus/definite.rs", 1),
    ("calculus/gosper.rs", 2),
    ("calculus/mellin.rs", 1),
    ("calculus/summation.rs", 1),
    ("domains/certificates.rs", 6),
    ("domains/certificates/polyhedron.rs", 7),
    ("domains/certificates/sos.rs", 2),
    ("domains/diophantine.rs", 7),
    ("domains/dynamics.rs", 2),
    ("domains/exact_matrix.rs", 9),
    ("domains/linprog.rs", 1),
    ("domains/matrix.rs", 7),
    ("domains/matrix_decomp.rs", 2),
    ("domains/normalforms.rs", 5),
    ("domains/ntheory.rs", 4),
    ("domains/optimize.rs", 8),
    ("domains/polytope.rs", 2),
    ("domains/quaternion.rs", 1),
    ("domains/robotics.rs", 6),
    ("domains/stats/aggregation.rs", 2),
    ("domains/stats/agreement.rs", 1),
    ("domains/stats/data.rs", 1),
    ("domains/stats/discrete.rs", 4),
    ("domains/stats/estimation.rs", 5),
    ("domains/stats/hypothesis.rs", 3),
    ("domains/stats/information.rs", 1),
    ("domains/stats/regression.rs", 4),
    ("domains/stats/reliability.rs", 1),
    ("domains/stats/sequential.rs", 2),
    ("domains/stats/survival.rs", 1),
    ("output/codegen.rs", 1),
    ("output/cse.rs", 2),
    ("output/lean.rs", 1),
    ("plotting/data_export.rs", 1),
    ("plotting/sampling.rs", 3),
    ("plotting/svg_plot.rs", 1),
    ("plotting/textplot.rs", 1),
    ("plotting/tikz_plot.rs", 1),
    ("poly/dense.rs", 2),
    ("poly/generic.rs", 3),
    ("poly/polybridge.rs", 2),
    ("poly/roots.rs", 2),
    ("poly/sturm.rs", 3),
    ("poly/traits.rs", 1),
    ("simplify/factor_terms.rs", 1),
    ("transforms/apart.rs", 1),
    ("transforms/subs.rs", 1),
    ("units/conv_factors.rs", 34),
];

/// Every homogeneous tuple type found in one item's public signature.
struct Finder<'a> {
    item: &'a str,
    sites: Vec<String>,
}

impl<'ast> Visit<'ast> for Finder<'_> {
    fn visit_type_tuple(&mut self, t: &'ast syn::TypeTuple) {
        if t.elems.len() >= 2 {
            let first = t.elems[0].to_token_stream().to_string();
            if t.elems
                .iter()
                .all(|e| e.to_token_stream().to_string() == first)
            {
                self.sites.push(format!(
                    "{}: {}",
                    self.item,
                    t.to_token_stream().to_string().replace(" ", "")
                ));
            }
        }
        syn::visit::visit_type_tuple(self, t);
    }

    // Closure parameter lists `Fn(&[f64], &[f64]) -> f64` are not tuples.
    fn visit_parenthesized_generic_arguments(
        &mut self,
        _: &'ast syn::ParenthesizedGenericArguments,
    ) {
    }
}

fn is_public(vis: &syn::Visibility) -> bool {
    !matches!(vis, syn::Visibility::Inherited)
}

fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.path().is_ident("cfg") && a.to_token_stream().to_string().contains("test"))
}

fn scan_signature(scope: &str, sig: &syn::Signature, out: &mut Vec<String>) {
    let item = format!("{scope}fn {}", sig.ident);
    let mut f = Finder {
        item: &item,
        sites: Vec::new(),
    };
    f.visit_signature(sig);
    out.extend(f.sites);
}

fn scan_type(item: &str, ty: &syn::Type, out: &mut Vec<String>) {
    let mut f = Finder {
        item,
        sites: Vec::new(),
    };
    f.visit_type(ty);
    out.extend(f.sites);
}

fn scan_items(items: &[syn::Item], out: &mut Vec<String>) {
    for item in items {
        match item {
            syn::Item::Fn(f) if is_public(&f.vis) && !is_cfg_test(&f.attrs) => {
                scan_signature("", &f.sig, out);
            }
            syn::Item::Impl(imp) if imp.trait_.is_none() && !is_cfg_test(&imp.attrs) => {
                let scope = format!("impl {} ", imp.self_ty.to_token_stream())
                    .replace(" >", ">")
                    .replace("< ", "<");
                for it in &imp.items {
                    if let syn::ImplItem::Fn(m) = it
                        && is_public(&m.vis)
                        && !is_cfg_test(&m.attrs)
                    {
                        scan_signature(&scope, &m.sig, out);
                    }
                }
            }
            syn::Item::Trait(t) if is_public(&t.vis) && !is_cfg_test(&t.attrs) => {
                let scope = format!("trait {} ", t.ident);
                for it in &t.items {
                    if let syn::TraitItem::Fn(m) = it {
                        scan_signature(&scope, &m.sig, out);
                    }
                }
            }
            syn::Item::Struct(s) if is_public(&s.vis) && !is_cfg_test(&s.attrs) => {
                for (i, field) in s.fields.iter().enumerate() {
                    if is_public(&field.vis) {
                        let name = field
                            .ident
                            .as_ref()
                            .map_or_else(|| i.to_string(), ToString::to_string);
                        scan_type(&format!("struct {}.{name}", s.ident), &field.ty, out);
                    }
                }
            }
            syn::Item::Type(t) if is_public(&t.vis) && !is_cfg_test(&t.attrs) => {
                scan_type(&format!("type {}", t.ident), &t.ty, out);
            }
            syn::Item::Const(c) if is_public(&c.vis) && !is_cfg_test(&c.attrs) => {
                scan_type(&format!("const {}", c.ident), &c.ty, out);
            }
            syn::Item::Static(s) if is_public(&s.vis) && !is_cfg_test(&s.attrs) => {
                scan_type(&format!("static {}", s.ident), &s.ty, out);
            }
            syn::Item::Mod(m) if !is_cfg_test(&m.attrs) => {
                if let Some((_, items)) = &m.content {
                    scan_items(items, out);
                }
            }
            _ => {}
        }
    }
}

fn sites_in_file(path: &Path) -> Vec<String> {
    let text = fs::read_to_string(path).expect("source file is readable");
    let file = syn::parse_file(&text)
        .unwrap_or_else(|e| panic!("{}: does not parse as Rust: {e}", path.display()));
    let mut out = Vec::new();
    scan_items(&file.items, &mut out);
    out
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
fn public_surface_has_no_unallowlisted_homogeneous_tuples() {
    let src = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let mut files = Vec::new();
    collect_rs_files(src, &mut files);
    files.sort();

    let allowed: BTreeMap<&str, usize> = ALLOWLIST.iter().copied().collect();
    let mut found: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in &files {
        let rel = path
            .strip_prefix(src)
            .expect("file is under src/")
            .to_string_lossy()
            .replace('\\', "/");
        found.insert(rel, sites_in_file(path));
    }

    let mut violations = Vec::new();
    for (rel, sites) in &found {
        let count = sites.len();
        let allowance = allowed.get(rel.as_str()).copied().unwrap_or(0);
        if count > allowance {
            violations.push(format!(
                "{rel}: {count} homogeneous tuple(s) in the public surface, allowed {allowance} — \
                 use a struct with named fields, or justify in the allowlist (CONTRIBUTING.md, \"Tuples versus structs\"):\n    {}",
                sites.join("\n    ")
            ));
        } else if count < allowance {
            violations.push(format!(
                "{rel}: {count} homogeneous tuple(s), allowed {allowance} — tighten the allowlist"
            ));
        }
    }
    for (rel, &allowance) in &allowed {
        if !found.contains_key(*rel) {
            violations.push(format!(
                "{rel}: allowlisted ({allowance}) but not found under src/ — tighten the allowlist"
            ));
        }
    }

    if !violations.is_empty() {
        panic!(
            "homogeneous-tuple ratchet: {} file(s) out of step:\n\n{}\n",
            violations.len(),
            violations.join("\n\n")
        );
    }
}
