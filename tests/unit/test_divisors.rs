//! Ratchet: divisions by a coefficient in series, limits and Gruntz outside
//! `NonzeroExpr` (session 15, M2).  See CONTRIBUTING.md, "Type rules that
//! keep bug classes out", and `calculus_util::NonzeroExpr`.
//!
//! A series coefficient, a leading term or a derivative at a point can be 0
//! without looking like it (`besselj(−1/2, π/2)`, 0 to every digit) or
//! vanish on a region of a parameter (`H(y)`, `0^y`, `⌊y⌋`).  Each site of
//! the three engines used to choose its own zero test before dividing, and
//! some chose none: L'Hôpital's quotient by `J_{1/2}(π)`, the series
//! reciprocal of `besselj(−1/2, π/2 + x)`, `ln(x + H(y))` and `√(x + H(y))`
//! expanded with `1/H(y)`, `Π(x | m)` with `1/m`.  A divisor is now a
//! `NonzeroExpr`, which only the deciders construct
//! (`ZeroDecision::decide`, `NonzeroExpr::number`, `structural`,
//! `variable`), and the engines divide with `calculus_util::div_by` and
//! `recip`.
//!
//! Two counts per covered file, both syntactic (a linter sees no types):
//!
//! | column | construct |
//! |---|---|
//! | `div` | a method call `.div(n, d)` with two arguments (`arena.div`), or `Arena::div(arena, n, d)` |
//! | `recip` | a method call `.pow(b, e)` whose exponent is `−1`: `….neg_one()`, `….neg_one`, `….int(-1)`, or a name a `let` of the same function binds to one of those (`let m1 = arena.neg_one;`) |
//!
//! Not counted: `div_by`/`recip` (they take a `NonzeroExpr`), powers with
//! other negative exponents (`(x − a)^(−k)` of the variable), and
//! `#[cfg(test)]` items.  Macro bodies are scanned when they parse as a
//! comma-separated list of expressions.
//!
//! What is left are quotients that are not divisions by a value, each with
//! a comment where it stands: a quotient of two functions of the variable
//! whose limit is taken next (`f′/g′`, `ln a/ln b`, `f_exp/g_exp` in
//! Gruntz), identities of functions (`tan u = sin u/cos u`, the asymptotic
//! forms `e^{u²}·E(u²)/u` with `u → ∞`, `1/δ` near a pole of `Γ`), and the
//! dummy of a substitution `x = a ± 1/t`.  The counts are exact: a file may
//! not gain a site, and a file that loses one must have its entry tightened
//! (the failure message prints the table to paste).

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use quote::ToTokens;
use syn::punctuated::Punctuated;
use syn::visit::Visit;

/// The covered files (relative to `src/`) with their exact counts `[div,
/// recip]` (the columns of the module doc).  At 0.57.0, before the
/// migration: series 8 + 5, limit 9 + 2, Gruntz 17 + 6 (47 sites).  0.59:
/// limit + 1, the identity `B(a, b) = Γ(a)Γ(b)/Γ(a + b)` at a pole of `B`
/// (`limit::beta_poles_as_gamma`).
const COVERED: &[(&str, [usize; 2])] = &[
    ("calculus/gruntz.rs", [9, 4]),
    ("calculus/limit.rs", [4, 2]),
    ("calculus/series.rs", [0, 0]),
];

const COLUMNS: [&str; 2] = ["div", "recip"];

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

/// `-1` (in parentheses or not).
fn is_literal_minus_one(e: &syn::Expr) -> bool {
    match e {
        syn::Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => matches!(
            &*u.expr,
            syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(i), .. }) if i.base10_digits() == "1"
        ),
        syn::Expr::Paren(p) => is_literal_minus_one(&p.expr),
        _ => false,
    }
}

/// Is `e` the expression `−1`: `….neg_one()`, `….neg_one`, `….int(-1)`, or
/// one of `names` (bound to such an expression)?
fn is_minus_one(e: &syn::Expr, names: &BTreeSet<String>) -> bool {
    match e {
        syn::Expr::MethodCall(m) => {
            (m.method == "neg_one" && m.args.is_empty())
                || (m.method == "int" && m.args.len() == 1 && is_literal_minus_one(&m.args[0]))
        }
        syn::Expr::Field(f) => matches!(&f.member, syn::Member::Named(n) if n == "neg_one"),
        syn::Expr::Path(p) => p
            .path
            .get_ident()
            .is_some_and(|i| names.contains(&i.to_string())),
        syn::Expr::Paren(p) => is_minus_one(&p.expr, names),
        syn::Expr::Reference(r) => is_minus_one(&r.expr, names),
        _ => false,
    }
}

/// The single name a `let` pattern binds (`x`, `mut x`, `x: ExprId`).
fn bound_name(p: &syn::Pat) -> Option<String> {
    match p {
        syn::Pat::Ident(i) => Some(i.ident.to_string()),
        syn::Pat::Type(t) => bound_name(&t.pat),
        _ => None,
    }
}

/// Names bound by a `let` in `block` (closures and nested blocks included,
/// nested `fn`s not) to `−1`.
fn minus_one_lets(block: &syn::Block) -> BTreeSet<String> {
    struct Lets {
        names: BTreeSet<String>,
    }
    impl<'a> Visit<'a> for Lets {
        fn visit_local(&mut self, l: &'a syn::Local) {
            if let (Some(name), Some(init)) = (bound_name(&l.pat), &l.init)
                && is_minus_one(&init.expr, &BTreeSet::new())
            {
                self.names.insert(name);
            }
            syn::visit::visit_local(self, l);
        }
        fn visit_item_fn(&mut self, _: &'a syn::ItemFn) {}
    }
    let mut lets = Lets {
        names: BTreeSet::new(),
    };
    lets.visit_block(block);
    lets.names
}

struct Scanner {
    /// Names bound to `−1`: one set per enclosing function body.
    names: Vec<BTreeSet<String>>,
    /// The function being scanned (for the site labels).
    fn_names: Vec<String>,
    sites: [Vec<String>; 2],
}

impl Scanner {
    fn in_fn(&mut self, name: String, block: &syn::Block) {
        self.names.push(minus_one_lets(block));
        self.fn_names.push(name);
        self.visit_block(block);
        self.fn_names.pop();
        self.names.pop();
    }

    fn label(&self, e: &dyn ToTokens) -> String {
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

    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        let empty = BTreeSet::new();
        let names = self.names.last().unwrap_or(&empty);
        if m.method == "div" && m.args.len() == 2 {
            let l = self.label(m);
            self.sites[0].push(l);
        }
        if m.method == "pow" && m.args.len() == 2 && is_minus_one(&m.args[1], names) {
            let l = self.label(m);
            self.sites[1].push(l);
        }
        syn::visit::visit_expr_method_call(self, m);
    }

    fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*c.func
            && p.path.segments.last().is_some_and(|s| s.ident == "div")
            && c.args.len() == 3
        {
            let l = self.label(c);
            self.sites[0].push(l);
        }
        syn::visit::visit_expr_call(self, c);
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

/// The sites of `path` (`[div, recip]`).
fn scan(path: &Path) -> [Vec<String>; 2] {
    let file = parse(path);
    let mut scanner = Scanner {
        names: Vec::new(),
        fn_names: Vec::new(),
        sites: [Vec::new(), Vec::new()],
    };
    scanner.visit_file(&file);
    scanner.sites
}

#[test]
fn divisions_by_hand_do_not_grow() {
    let src = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let mut violations = Vec::new();
    let mut table = Vec::new();
    for &(rel, allowance) in COVERED {
        let path = src.join(rel);
        assert!(
            path.is_file(),
            "{rel}: covered but not found under src/ — update COVERED"
        );
        let sites = scan(&path);
        let have = [sites[0].len(), sites[1].len()];
        table.push(format!("    (\"{rel}\", [{}, {}]),", have[0], have[1]));
        for col in 0..2 {
            let (n, a) = (have[col], allowance[col]);
            if n > a {
                violations.push(format!(
                    "{rel}: {n} `{}` site(s), allowed {a} — divide by a `NonzeroExpr` \
                     (`ZeroDecision::decide`, `NonzeroExpr::number`/`structural`/`variable`) \
                     with `div_by`/`recip`, or comment why the quotient is not a division by \
                     a value:\n    {}",
                    COLUMNS[col],
                    sites[col].join("\n    ")
                ));
            } else if n < a {
                violations.push(format!(
                    "{rel}: {n} `{}` site(s), allowed {a} — tighten COVERED",
                    COLUMNS[col]
                ));
            }
        }
    }
    if !violations.is_empty() {
        panic!(
            "divisor ratchet: {} violation(s):\n\n{}\n\nCurrent table [div, recip]:\n{}\n",
            violations.len(),
            violations.join("\n\n"),
            table.join("\n")
        );
    }
}

/// The scanner sees each shape it counts, and nothing else.
#[test]
fn divisor_scanner_counts_its_shapes() {
    let file: syn::File = syn::parse_str(
        r#"
        fn f(arena: &mut Arena, a: ExprId, b: ExprId) {
            let m1 = arena.neg_one;
            let x = arena.div(a, b);
            let y = arena.pow(a, arena.neg_one());
            let z = arena.pow(b, m1);
            let w = arena.pow(b, arena.int(-1));
            let u = Arena::div(arena, a, b);
            let ok = arena.pow(a, arena.int(-2));
            let ok = div_by(arena, a, d);
            let ok = recip(arena, d);
            let v = vec![arena.div(a, b)];
        }
        fn g(arena: &mut Arena, a: ExprId, m1: ExprId) {
            let ok = arena.pow(a, m1);
        }
        #[cfg(test)]
        mod tests {
            fn t(a: &mut Arena, x: ExprId) { let q = a.div(x, x); }
        }
        "#,
    )
    .expect("the sample parses");
    let mut scanner = Scanner {
        names: Vec::new(),
        fn_names: Vec::new(),
        sites: [Vec::new(), Vec::new()],
    };
    scanner.visit_file(&file);
    assert_eq!(scanner.sites[0].len(), 3, "{:?}", scanner.sites[0]);
    assert_eq!(scanner.sites[1].len(), 3, "{:?}", scanner.sites[1]);
}
