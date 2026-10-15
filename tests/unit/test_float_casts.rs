//! Ratchet over `src/`: `f64` to integer `as` casts after `floor`/`ceil`/
//! `round`/`trunc` (wave 1, W4), and the `(f64 expr) as usize + …`
//! precision pattern, must not grow beyond their allowlisted count per
//! file; an allowlisted file must not lose one without the allowlist being
//! tightened.  See CONTRIBUTING.md, "Checked casts".
//!
//! An `as` cast from `f64` saturates (`1e19 as i64` is `i64::MAX`, `NaN as
//! i64` is `0`) and the `± 1` that typically follows wraps in release: 0.48
//! integrated `floor(10¹⁹x)/10¹⁹` over `[−1, 1]` to `0` and looped forever
//! on `floor(9·10¹⁸x)/10¹⁹` (S4-B1/B2, `calculus_util.rs`).  New code uses
//! `base::cast::{f64_to_i64, integer_span}`, which return `None`/an error.
//!
//! Two counts, both syntactic (no types are available to a linter):
//!
//! - **Rounded casts**: `e as T` with `T` one of `i64 i32 isize u64 u32
//!   usize` and `e` (inside parentheses) a method call `.floor()`,
//!   `.ceil()`, `.round()`, `.round_ties_even()` or `.trunc()`.  Those
//!   methods exist only on floats, so `e` is an `f64` (or `f32`).
//! - **Precision casts**: `e as usize` as a direct operand of `+`, where `e`
//!   looks like a float expression (a float literal, an `as f64`/`as f32`
//!   cast, an `f64::…` path, or a float-only method such as `log2`, `ln`,
//!   `sqrt`, `ceil`).  These are the `(x.log2()) as usize + wp + 40` guard-bit
//!   sites of evalf (review item 13): the cast saturates on `∞` and the sum
//!   then overflows.  A cast that is both is counted in both lists.
//!
//! `#[cfg(test)]` items and modules are skipped.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::visit::Visit;

/// Files allowed to contain rounded casts, with the exact count (0.49: 90,
/// down from 94; 0.50: 87, three evalf guards through `Prec::guarded`).  Paths are relative to `src/`.  Not audited one by one:
/// most are bounded by an earlier guard in the same function (a precision
/// cap, a `digits.clamp`, an `is_finite` test); each is a candidate for
/// `base::cast`, and the list may only shrink.
const ROUNDED_ALLOWLIST: &[(&str, usize)] = &[
    ("api/expr_poly_ext.rs", 1),
    ("base/bernoulli.rs", 2),
    ("base/canon.rs", 1),
    ("base/dense_f64.rs", 1),
    ("domains/ntheory.rs", 2),
    ("domains/stats/anova.rs", 3),
    ("domains/stats/hypothesis.rs", 2),
    ("domains/stats/multivariate.rs", 1),
    ("plotting/textplot.rs", 3),
    ("plotting/tikz_plot.rs", 1),
    ("poly/sturm.rs", 1),
    ("simplify/nsimplify.rs", 4),
    ("transforms/apart.rs", 3),
    // evalf: precisions and term counts from `f64` estimates (wave 3: `Prec`/`Wp`).
    ("transforms/evalf.rs", 22),
    ("transforms/evalf/asymsum.rs", 2),
    ("transforms/evalf/bernoulli.rs", 1),
    ("transforms/evalf/bessel_debye.rs", 2),
    ("transforms/evalf/bessel_order.rs", 2),
    ("transforms/evalf/bessel_recur.rs", 2),
    ("transforms/evalf/cerf.rs", 1),
    ("transforms/evalf/emsum.rs", 1),
    ("transforms/evalf/extended.rs", 1),
    ("transforms/evalf/factorials.rs", 1),
    ("transforms/evalf/hypsum.rs", 1),
    ("transforms/evalf/loggamma.rs", 1),
    ("transforms/evalf/polylog.rs", 6),
    ("transforms/evalf/quad.rs", 1),
    ("transforms/evalf/rszeta.rs", 2),
    ("transforms/evalf/zeta.rs", 5),
    ("transforms/heurisch.rs", 1),
    ("transforms/logic.rs", 1),
    ("transforms/solve.rs", 6),
];

/// Files allowed to contain precision casts, with the exact count (0.49:
/// 44; 0.50: 41).  Guard bits `(x.log2()) as usize + wp + k` (wave 3: `Prec::guarded`
/// replaces them with a checked sum).
const PRECISION_ALLOWLIST: &[(&str, usize)] = &[
    ("api/expr_poly_ext.rs", 1),
    ("base/bernoulli.rs", 2),
    ("transforms/evalf.rs", 20),
    ("transforms/evalf/asymsum.rs", 2),
    ("transforms/evalf/bessel_debye.rs", 2),
    ("transforms/evalf/bessel_order.rs", 1),
    ("transforms/evalf/bessel_recur.rs", 2),
    ("transforms/evalf/cerf.rs", 1),
    ("transforms/evalf/factorials.rs", 2),
    ("transforms/evalf/loggamma.rs", 1),
    ("transforms/evalf/polylog.rs", 3),
    ("transforms/evalf/zeta.rs", 4),
];

const INT_TYPES: &[&str] = &["i64", "i32", "isize", "u64", "u32", "usize"];
const ROUNDING: &[&str] = &["floor", "ceil", "round", "round_ties_even", "trunc"];
/// Methods that exist on `f64` but not on the integer types.
const FLOAT_ONLY: &[&str] = &[
    "floor",
    "ceil",
    "round",
    "round_ties_even",
    "trunc",
    "fract",
    "log2",
    "log10",
    "ln",
    "ln_1p",
    "exp",
    "exp2",
    "exp_m1",
    "sqrt",
    "cbrt",
    "powf",
    "powi",
    "hypot",
    "recip",
    "to_degrees",
    "to_radians",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "is_finite",
    "is_nan",
    "mul_add",
];

fn strip_parens(mut e: &syn::Expr) -> &syn::Expr {
    loop {
        match e {
            syn::Expr::Paren(p) => e = &p.expr,
            syn::Expr::Group(g) => e = &g.expr,
            _ => return e,
        }
    }
}

fn type_ident(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(p) if p.qself.is_none() => p.path.get_ident().map(ToString::to_string),
        _ => None,
    }
}

fn is_int_cast(c: &syn::ExprCast) -> bool {
    type_ident(&c.ty).is_some_and(|t| INT_TYPES.contains(&t.as_str()))
}

fn ends_in_rounding(e: &syn::Expr) -> bool {
    matches!(strip_parens(e), syn::Expr::MethodCall(m)
        if m.args.is_empty() && ROUNDING.contains(&m.method.to_string().as_str()))
}

/// Does `e` look like a float expression?  Does not look inside casts to
/// integer types (`v[x.round() as usize]` is an index, not a float).
fn is_floaty(e: &syn::Expr) -> bool {
    struct Floaty(bool);
    impl<'ast> Visit<'ast> for Floaty {
        fn visit_lit_float(&mut self, _: &'ast syn::LitFloat) {
            self.0 = true;
        }
        fn visit_expr_cast(&mut self, c: &'ast syn::ExprCast) {
            match type_ident(&c.ty).as_deref() {
                Some("f64" | "f32") => self.0 = true,
                _ if is_int_cast(c) => {}
                _ => syn::visit::visit_expr_cast(self, c),
            }
        }
        fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
            if FLOAT_ONLY.contains(&m.method.to_string().as_str()) {
                self.0 = true;
            }
            syn::visit::visit_expr_method_call(self, m);
        }
        fn visit_path(&mut self, p: &'ast syn::Path) {
            if p.segments
                .first()
                .is_some_and(|s| s.ident == "f64" || s.ident == "f32")
            {
                self.0 = true;
            }
            syn::visit::visit_path(self, p);
        }
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
    }
    let mut f = Floaty(false);
    f.visit_expr(e);
    f.0
}

fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.path().is_ident("cfg") && a.to_token_stream().to_string().contains("test"))
}

#[derive(Default)]
struct Finder {
    rounded: Vec<String>,
    precision: Vec<String>,
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

    fn visit_expr_cast(&mut self, c: &'ast syn::ExprCast) {
        if is_int_cast(c) && ends_in_rounding(&c.expr) {
            self.rounded.push(site(c));
        }
        syn::visit::visit_expr_cast(self, c);
    }

    fn visit_expr_binary(&mut self, b: &'ast syn::ExprBinary) {
        if matches!(b.op, syn::BinOp::Add(_)) {
            for side in [&*b.left, &*b.right] {
                if let syn::Expr::Cast(c) = strip_parens(side)
                    && type_ident(&c.ty).as_deref() == Some("usize")
                    && is_floaty(&c.expr)
                {
                    self.precision.push(site(b));
                }
            }
        }
        syn::visit::visit_expr_binary(self, b);
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

#[test]
fn rounded_float_casts_do_not_grow() {
    let found: BTreeMap<String, Vec<String>> = scan_src()
        .into_iter()
        .filter(|(_, f)| !f.rounded.is_empty())
        .map(|(rel, f)| (rel, f.rounded))
        .collect();
    let violations = check(
        "saturating cast of a rounded f64",
        "use `base::cast::f64_to_i64` / `integer_span` (CONTRIBUTING.md, \"Checked casts\")",
        ROUNDED_ALLOWLIST,
        &found,
    );
    if !violations.is_empty() {
        panic!(
            "float-cast ratchet: {} file(s) out of step:\n\n{}\n",
            violations.len(),
            violations.join("\n\n")
        );
    }
}

#[test]
fn precision_float_casts_do_not_grow() {
    let found: BTreeMap<String, Vec<String>> = scan_src()
        .into_iter()
        .filter(|(_, f)| !f.precision.is_empty())
        .map(|(rel, f)| (rel, f.precision))
        .collect();
    let violations = check(
        "`(f64 expr) as usize + …` precision cast",
        "bound the f64 first (a NaN or ∞ saturates and the sum overflows), or convert with \
         `base::cast::f64_to_i64` (CONTRIBUTING.md, \"Checked casts\")",
        PRECISION_ALLOWLIST,
        &found,
    );
    if !violations.is_empty() {
        panic!(
            "precision-cast ratchet: {} file(s) out of step:\n\n{}\n",
            violations.len(),
            violations.join("\n\n")
        );
    }
}
