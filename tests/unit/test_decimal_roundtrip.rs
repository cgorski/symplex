//! Ratchet over `src/`: decimal strings of evaluations read back inside the
//! library (session 15, M5).  See CONTRIBUTING.md, "Type rules that keep
//! bug classes out" ("Numeric values of constants"), and
//! `transforms::evalf::ConstValue`.
//!
//! The value of a constant for a decision is read from its settled parts
//! (`evalf::const_value`: the certified sign of each part, the printed
//! digits as an unscaled `Decimal`, a correctly rounded `f64` view), never
//! by printing a decimal and parsing it back.  Each round trip re-derived
//! from text what the value already held, and each had its bug: the exact
//! rational of a sample value `exp(1000³)` formed `10^434294481` and hung
//! (0.56), an imaginary part the printout omitted was lost (`expr_to_f64`,
//! 0.53), a 16-digit decimal parsed to an `f64` an ulp away from the value.
//!
//! Two counts per file, both syntactic (a linter sees no types):
//!
//! | column | construct |
//! |---|---|
//! | `parse` | text read as a float: `.parse::<f64>()`, `.parse::<f32>()`, `f64::from_str`, `str::parse::<f64>`, a `.parse()` bound by `let v: f64 = …` (or `f32`), and calls of `decimal_to_ratio` |
//! | `string` | a call of a decimal-string evaluator (`evalf`, `evalf_with`, `evalf_quadrature`, `.eval_decimal(…)`, `.evalf_expr(…)`) in a function whose return type does not name `String`: the decimal is read, not passed on |
//!
//! Not counted: `src/output/parse.rs` (the expression parser reads user
//! text), `#[cfg(test)]` items, and functions that return the decimal
//! (`Ex::eval_decimal`, the p-value printers).  A call inside a closure
//! counts for the enclosing function.
//!
//! What is left is user text that is not an evaluation
//! (`plotting/data_export`); the review queue of 0.58 (canon, eval,
//! residue, linalg, matrix, the stats family signs, `poly/roots`, the
//! certified quadrature, heurisch) was migrated at 0.59.  The counts are
//! exact: a file may not gain a site, and a file that loses one must have
//! its entry tightened (the failure message prints the table to paste).
//! Files without an entry must have no site.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::punctuated::Punctuated;
use syn::visit::Visit;

/// Per-file counts `[parse, string]` (the columns of the module doc).
/// Paths are relative to `src/`; files with no site have no entry.  At
/// 0.57.0: 16 files, 12 `parse`, 17 `string`; M5 migrated `inequalities`
/// (0 + 3), `sets` (1 + 2), `solve`, `evalf`, `nsimplify` and
/// `calculus_util` (1 + 1 each) to `evalf::const_value`: 10 files, 7, 8.
/// 0.59 (P3) migrated the other nine (`certified_quadrature`, `canon`'s turn
/// estimate and interval ends, `residue`'s finiteness test, the zero tests
/// of `linalg` and `matrix`, `stats::family::sign_of`, `poly::roots`,
/// `eval::integer_part_of_constant`, `heurisch::eval_node_f64`): 1 file, 1, 0.
const ALLOWLIST: &[(&str, [usize; 2])] = &[
    // A table cell (user text, not an evaluation) exported as JSON `null` when it reads as ±inf/NaN.
    ("plotting/data_export.rs", [1, 0]),
];

const COLUMNS: [&str; 2] = ["parse", "string"];

/// Files exempt from the ratchet: the parser of user text.
const EXEMPT: &[&str] = &["output/parse.rs"];

/// The decimal-string evaluators called as functions (path's last segment).
const STRING_FNS: &[&str] = &["evalf", "evalf_with", "evalf_quadrature"];

/// The decimal-string evaluators called as methods.
const STRING_METHODS: &[&str] = &["eval_decimal", "evalf_expr"];

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

/// `f64` or `f32` (a turbofish argument or a `let` type).
fn is_float_type(ty: &syn::Type) -> bool {
    matches!(ty, syn::Type::Path(p) if p.path.is_ident("f64") || p.path.is_ident("f32"))
}

/// Does `e` contain a `.parse()` method call (any turbofish)?
fn has_parse_call(e: &syn::Expr) -> bool {
    struct Find(bool);
    impl<'ast> Visit<'ast> for Find {
        fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
            if m.method == "parse" {
                self.0 = true;
            }
            syn::visit::visit_expr_method_call(self, m);
        }
    }
    let mut f = Find(false);
    f.visit_expr(e);
    f.0
}

struct Scanner {
    /// For each enclosing function: its name and whether it returns a
    /// `String`.
    fns: Vec<(String, bool)>,
    sites: [Vec<String>; 2],
}

impl Scanner {
    fn in_fn(&mut self, sig: &syn::Signature, block: &syn::Block) {
        let returns_string = match &sig.output {
            syn::ReturnType::Default => false,
            syn::ReturnType::Type(_, ty) => ty.to_token_stream().to_string().contains("String"),
        };
        self.fns.push((sig.ident.to_string(), returns_string));
        self.visit_block(block);
        self.fns.pop();
    }

    fn site(&mut self, col: usize, what: &dyn ToTokens) {
        let text: String = what
            .to_token_stream()
            .to_string()
            .chars()
            .take(90)
            .collect();
        let name = self.fns.last().map_or("(item)", |(n, _)| n.as_str());
        self.sites[col].push(format!("{name}: {text}"));
    }

    fn passes_string_on(&self) -> bool {
        self.fns.last().is_some_and(|&(_, s)| s)
    }
}

impl<'ast> Visit<'ast> for Scanner {
    fn visit_item(&mut self, i: &'ast syn::Item) {
        if !is_cfg_test(item_attrs(i)) {
            syn::visit::visit_item(self, i);
        }
    }

    fn visit_item_fn(&mut self, f: &'ast syn::ItemFn) {
        self.in_fn(&f.sig, &f.block);
    }

    fn visit_impl_item_fn(&mut self, m: &'ast syn::ImplItemFn) {
        if !is_cfg_test(&m.attrs) {
            self.in_fn(&m.sig, &m.block);
        }
    }

    fn visit_trait_item_fn(&mut self, m: &'ast syn::TraitItemFn) {
        if let Some(block) = &m.default
            && !is_cfg_test(&m.attrs)
        {
            self.in_fn(&m.sig, block);
        }
    }

    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        if m.method == "parse"
            && m.turbofish.as_ref().is_some_and(|t| {
                t.args
                    .iter()
                    .any(|a| matches!(a, syn::GenericArgument::Type(ty) if is_float_type(ty)))
            })
        {
            self.site(0, m);
        }
        if STRING_METHODS.iter().any(|s| m.method == s) && !self.passes_string_on() {
            self.site(1, m);
        }
        syn::visit::visit_expr_method_call(self, m);
    }

    fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*c.func {
            let segs: Vec<String> = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            let last = segs.last().map_or("", String::as_str);
            let float_from_str = last == "from_str"
                && segs.len() >= 2
                && matches!(segs[segs.len() - 2].as_str(), "f64" | "f32");
            let str_parse_float = last == "parse"
                && p.path.segments.last().is_some_and(|s| {
                    s.arguments.to_token_stream().to_string().contains("f64")
                        || s.arguments.to_token_stream().to_string().contains("f32")
                });
            if float_from_str || str_parse_float || last == "decimal_to_ratio" {
                self.site(0, c);
            }
            if STRING_FNS.contains(&last) && !self.passes_string_on() {
                self.site(1, c);
            }
        }
        syn::visit::visit_expr_call(self, c);
    }

    fn visit_local(&mut self, l: &'ast syn::Local) {
        if let (syn::Pat::Type(t), Some(init)) = (&l.pat, &l.init)
            && is_float_type(&t.ty)
            && has_parse_call(&init.expr)
        {
            self.site(0, l);
        }
        syn::visit::visit_local(self, l);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if let Ok(exprs) =
            mac.parse_body_with(Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated)
        {
            for e in &exprs {
                self.visit_expr(e);
            }
        } else if let Ok(e) = mac.parse_body_with(|input: syn::parse::ParseStream| {
            // `matches!(expr, pattern)`: the leading expression.
            let e: syn::Expr = input.parse()?;
            input.step(|cursor| {
                let mut rest = *cursor;
                while let Some((_, next)) = rest.token_tree() {
                    rest = next;
                }
                Ok(((), rest))
            })?;
            Ok(e)
        }) {
            self.visit_expr(&e);
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
fn decimal_round_trips_do_not_grow() {
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
        if EXEMPT.contains(&rel.as_str()) {
            continue;
        }
        let mut scanner = Scanner {
            fns: Vec::new(),
            sites: [Vec::new(), Vec::new()],
        };
        scanner.visit_file(&parse(path));
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
                    "{rel}: {n} `{}` site(s), allowed {a} — read the value of a constant with \
                     `evalf::const_value` (certified signs, `Decimal` digits, `to_f64`), not by \
                     parsing its decimal:\n    {}",
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
            "decimal round-trip ratchet: {} violation(s):\n\n{}\n\nCurrent table [parse, string] \
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
