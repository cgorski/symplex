//! Ratchet over `src/`: the public surface — `pub`/`pub(crate)` function
//! signatures, `pub` struct fields, `pub` type aliases, consts and statics —
//! must not gain a *homogeneous tuple type* `(T, T, …)` beyond its
//! allowlisted count per file, and an allowlisted file must not lose one
//! without the allowlist being tightened.  See CONTRIBUTING.md, "Tuples
//! versus structs".
//!
//! A homogeneous tuple is one in which some element type occurs more than
//! once, so nothing but memory says which of those positions is which:
//! `(lower, upper)`, `(Q, R)`, `(shape, scale)`, `(lo, hi, lo_open, hi_open)`,
//! `(&Angle, &Length, &Length, &Angle)`.  Each remaining site below is either a
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
    // ── Kept: universal conventions, pattern-matched at every use ─────────
    // `subs_map(&[(&var, &value)])` — substitution pairs (SymPy `subs([(x, 1)])`).
    ("api/expr.rs", 1),
    // `as_real_imag -> (re, im)` (symbolic parts; the `f64` form is `Complex64`).
    ("api/expr_complex.rs", 1),
    // `factor_terms -> (coeff, rest)`; `as_numer_denom -> (numer, denom)`;
    // `cse`/`cse_many` `(symbol, definition)` bindings (SymPy `cse`); `plot_data -> (x, y)` points;
    // `eval_f64_with_rational(&[(&var, numer, denom)])` — a rational literal per variable.
    ("api/expr_funcs.rs", 6),
    // `as_ratio_parts`/`as_ratio_i128 -> (numer, denom)`; `eval_at(&[(&var, &value)])`.
    ("api/expr_ops.rs", 3),
    // `poly_div -> (quotient, remainder)` (num_integer `div_rem`); `content_primitive -> (content, primitive)`;
    // `poly_interpolate(&[(x, y)])` points.
    ("api/expr_poly_ext.rs", 3),
    // `separate_vars_dict -> Vec<(var, factor)>` (SymPy `separatevars(dict=True)`).
    ("api/expr_rules_ext.rs", 1),
    // `LinearSolution::pairs -> &[(var, value)]`.
    ("api/expr_solve_ext.rs", 1),
    // Internal mirrors of the public conventions: `as_base_exp -> (base, exp)` (as `Pow(base, exp)`),
    // `subs_map_structural`, `as_numer_denom_expr`, `factor_terms_pair_expr`, `as_real_imag_expr`,
    // `piecewise(&[(expr, cond)])` (SymPy `Piecewise((expr, cond), …)`).
    ("base/arena.rs", 6),
    // `type Complex = (re, im)` for the arbitrary-precision evaluator.
    ("base/bigcomplex.rs", 1),
    // `split_perfect_power(n, k) -> (a, b)` with `n = aᵏ·b`, in the order written; single caller.
    ("base/canon.rs", 1),
    // `as_real_imag -> (re, im)`.
    ("base/complex.rs", 1),
    // `Interval::into_pair -> (lower, upper)` — the documented escape hatch itself.
    ("base/interval.rs", 1),
    // `sign_nodes_of -> Vec<(sign(h), h)>`: a node and its own argument; single caller.
    ("calculus/calculus_util.rs", 1),
    // Gosper's normal form `(p, q, r)` — the algorithm's own names; `hypergeometric_ratio -> (numer, denom)`.
    ("calculus/gosper.rs", 2),
    // `mellin_transform -> (transform, strip)`, typed `(Ex, BoolEx)` at the public boundary.
    ("calculus/mellin.rs", 1),
    // `TermShape.lin_pows: (β, p)` meaning `(k + β)^p`; `facts: (r, a, …)` — documented pattern-matched
    // shape fields of a private analysis struct.
    ("calculus/summation.rs", 2),
    // `pell*/sum_of_two_squares -> (x, y)` (symmetric / `x² − Dy²`), `sum_of_four_squares` (symmetric),
    // `pythagorean_triples -> (a, b, c)`.
    ("domains/diophantine.rs", 6),
    // `shape -> (rows, cols)`.
    ("domains/exact_matrix.rs", 1),
    // `shape -> (rows, cols)`; `subs_map(&[(&var, &value)])`.
    ("domains/matrix.rs", 2),
    // `binomial_coefficients -> Vec<((n, k), C(n, k))>` — `(n, k)` keys.
    ("domains/ntheory.rs", 1),
    // `poly_fit_exact(&[(x, y)])` and `poly_fit_points(&[(x, y)])` — points.
    ("domains/optimize.rs", 2),
    // `to_euler -> (φ, θ, ψ)` in the order the `EulerConvention` names.
    ("domains/quaternion.rs", 1),
    // `fk_position -> (x, y, z)`, `fk_position_typed -> (x, y, z)`, `inverse_kinematics_2dof -> (θ₁, θ₂)`
    // by joint order.
    ("domains/robotics.rs", 3),
    // `RatingTable::paired_ratings(j1, j2) -> (ratings of j1, ratings of j2)` — argument order.
    ("domains/stats/agreement.rs", 1),
    // `Finite.table: Vec<(value, probability)>` — a value → probability map (`HashMap::from([(k, v)])`).
    ("domains/stats/discrete.rs", 4),
    // `param_units: Vec<(parameter, unit)>` — key → value.
    ("output/codegen.rs", 1),
    // `display_sort_key -> (category, …, name bytes, suffix bytes)`: a lexicographic sort key.
    ("output/common.rs", 1),
    // `CseResult`/`CseMultiResult.bindings: Vec<(symbol, definition)>`.
    ("output/cse.rs", 2),
    // `symbol_text: Vec<(symbol, text)>` — key → value.
    ("output/lean.rs", 1),
    // `from_points(&[(x, y)])`.
    ("plotting/data_export.rs", 1),
    // `PlotData.points: Vec<(x, y)>`.
    ("plotting/sampling.rs", 1),
    // `series: &[(&[(x, y)], label)]` ×3.
    ("plotting/svg_plot.rs", 1),
    ("plotting/textplot.rs", 1),
    ("plotting/tikz_plot.rs", 1),
    // `lagrange_interpolate_points(&[(x, y)])`; `kronecker_find_factor -> (factor, cofactor)` (symmetric).
    ("poly/dense.rs", 2),
    // `div_rem`/`try_div_rem -> (quotient, remainder)`.
    ("poly/generic.rs", 2),
    // `as_numer_denom`, `fraction_parts -> (numer, denom)`.
    ("poly/polybridge.rs", 2),
    // `EuclideanDomain::div_rem -> (quotient, remainder)`.
    ("poly/traits.rs", 1),
    // `symbolic_factor_terms_pair -> (coeff, rest)`.
    ("simplify/factor_terms.rs", 1),
    // Hermite reduction step `(A, factor^{n−1}, B, factor)` — the algorithm's own names; single caller.
    ("transforms/apart.rs", 1),
    // `subs_map(&[(var, value)])`.
    ("transforms/subs.rs", 1),
    // Exact conversion factors as `(numer, denom)` rational literals.
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
            let mut seen = std::collections::BTreeSet::new();
            let repeated = t
                .elems
                .iter()
                .any(|e| !seen.insert(e.to_token_stream().to_string()));
            if repeated {
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
