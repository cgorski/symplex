//! Ratchet (session 17): the name, the TeX form, the sort class and the parse
//! name of a library function are facets of the `funcs!` table
//! (`src/base/func.rs`); the printers, the parser and the sort key read them
//! from the table instead of repeating them.  This file counts what is left.
//!
//! The bug class: a fact about a function written in several places.  Up to
//! 0.59 a function's name was a string literal in `display.rs`, in
//! `pretty.rs`, in `mathml.rs`, twice in `parse.rs` (the call table and the
//! list of known names) and a byte in `sort_key.rs`, beside the table's
//! entry; adding a function meant finding every one of them, and nothing but
//! a test over `Func::ALL` kept two spellings equal.  The six files now have
//! one arm `func_nodes!() =>` each that reads the facets (`name`, `tex`,
//! `pretty`, `mathml`, `sort`), and the parser looks a call name up with
//! `Func::parsed`.  See CONTRIBUTING.md, "How to Add a New Function".
//!
//! What is counted, syntactically: every string literal of the six files —
//! in expressions, in patterns and inside macro invocations (`matches!`,
//! `format!`) — whose value is a *spelling* of a function of the table: its
//! `name`, its `name` in lower case (what the parser compares), or one of
//! its `parse` names.  The spellings are read from the `funcs!` invocation
//! itself.  Doc comments, attributes and `#[cfg(test)]` items are skipped.
//!
//! The allowlist holds exactly the literals that stay, each with its count
//! and the reason it is not a second copy of a facet; it may only shrink.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use quote::ToTokens;
use syn::parse::{ParseStream, Parser};
use syn::visit::Visit;

/// The table (`src/`-relative); its own literals are the facets.
const TABLE: &str = "base/func.rs";

/// The files that printed, parsed or ranked a function by a literal of
/// their own, `src/`-relative.
const FILES: &[&str] = &[
    "base/sort_key.rs",
    "output/display.rs",
    "output/latex.rs",
    "output/mathml.rs",
    "output/parse.rs",
    "output/pretty.rs",
];

/// One literal that stays: the file, the literal, how often it occurs.
struct Kept {
    file: &'static str,
    literal: &'static str,
    count: usize,
}

const fn kept(file: &'static str, literal: &'static str, count: usize) -> Kept {
    Kept {
        file,
        literal,
        count,
    }
}

/// The literals that equal a spelling of a table function and are not a
/// copy of its facet: 14 literals, none of which names the function it is
/// spelled like.  (This count over the sources of 0.59 was 274:
/// `display.rs` 36, `latex.rs` 6, `mathml.rs` 50, `parse.rs` 152,
/// `pretty.rs` 30.  The LaTeX heads and the sort-key bytes are no spelling
/// of a name, so the arms that repeated them — in `latex.rs` and
/// `sort_key.rs` — went without ever being counted here; the match on
/// `func_nodes!()` keeps them from coming back, since a second arm for a
/// function node is an unreachable pattern.)
const ALLOWLIST: &[Kept] = &[
    // Greek letters of *symbol* names (`GREEK_LETTERS`: `beta` prints as
    // `\beta`): letters, not the functions `B`, `Γ`, `ζ`, `Chi`.
    kept("output/latex.rs", "Chi", 1),
    kept("output/latex.rs", "Gamma", 1),
    kept("output/latex.rs", "beta", 1),
    kept("output/latex.rs", "chi", 1),
    kept("output/latex.rs", "gamma", 1),
    kept("output/latex.rs", "zeta", 1),
    // The same letters in `GREEK` (symbol names → character references);
    // the second `gamma` is Euler's constant `γ` (`ExprNode::EulerGamma`).
    kept("output/mathml.rs", "Chi", 1),
    kept("output/mathml.rs", "Gamma", 1),
    kept("output/mathml.rs", "beta", 1),
    kept("output/mathml.rs", "chi", 1),
    kept("output/mathml.rs", "gamma", 2),
    kept("output/mathml.rs", "zeta", 1),
    // `log(x, b)` with two arguments is the quotient `ln(x)/ln(b)`, not the
    // function `ln` (whose parse name `log` takes one argument): an arm of
    // `call_2`.
    kept("output/parse.rs", "log", 1),
];

fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.path().is_ident("cfg") && a.to_token_stream().to_string().contains("test"))
}

/// Every string literal of a token stream, through the delimiters.
fn literals_of_tokens(input: ParseStream, out: &mut Vec<String>) -> syn::Result<()> {
    while !input.is_empty() {
        if input.peek(syn::LitStr) {
            out.push(input.parse::<syn::LitStr>()?.value());
        } else if input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            literals_of_tokens(&content, out)?;
        } else if input.peek(syn::token::Brace) {
            let content;
            syn::braced!(content in input);
            literals_of_tokens(&content, out)?;
        } else if input.peek(syn::token::Bracket) {
            let content;
            syn::bracketed!(content in input);
            literals_of_tokens(&content, out)?;
        } else {
            input.step(|cursor| match cursor.token_tree() {
                Some((_, next)) => Ok(((), next)),
                None => Err(cursor.error("unexpected end of tokens")),
            })?;
        }
    }
    Ok(())
}

/// The string literals of a file outside tests and attributes.
#[derive(Default)]
struct Literals(Vec<String>);

impl<'ast> Visit<'ast> for Literals {
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

    // Doc comments are `#[doc = "…"]` attributes: not code.
    fn visit_attribute(&mut self, _: &'ast syn::Attribute) {}

    fn visit_lit_str(&mut self, l: &'ast syn::LitStr) {
        self.0.push(l.value());
    }

    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        let mut found = Vec::new();
        let walk = |input: ParseStream| literals_of_tokens(input, &mut found);
        walk.parse2(m.tokens.clone())
            .unwrap_or_else(|e| panic!("macro `{}`: {e}", m.path.to_token_stream()));
        self.0.extend(found);
    }
}

fn parse_src(rel: &str) -> syn::File {
    let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src")).join(rel);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    syn::parse_file(&text)
        .unwrap_or_else(|e| panic!("{}: does not parse as Rust: {e}", path.display()))
}

/// `name: "…"` and `parse: &["…", …]` of every entry of the table.
fn facets_of_tokens(
    input: ParseStream,
    names: &mut Vec<String>,
    parse_names: &mut Vec<String>,
) -> syn::Result<()> {
    while !input.is_empty() {
        if input.peek(syn::Ident) && input.peek2(syn::Token![:]) && !input.peek2(syn::Token![::]) {
            let field: syn::Ident = input.parse()?;
            input.parse::<syn::Token![:]>()?;
            if field == "name" && input.peek(syn::LitStr) {
                names.push(input.parse::<syn::LitStr>()?.value());
            } else if field == "parse" && input.peek(syn::Token![&]) {
                input.parse::<syn::Token![&]>()?;
                let content;
                syn::bracketed!(content in input);
                let list = content
                    .parse_terminated(<syn::LitStr as syn::parse::Parse>::parse, syn::Token![,])?;
                parse_names.extend(list.iter().map(syn::LitStr::value));
            }
        } else if input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            facets_of_tokens(&content, names, parse_names)?;
        } else if input.peek(syn::token::Brace) {
            let content;
            syn::braced!(content in input);
            facets_of_tokens(&content, names, parse_names)?;
        } else if input.peek(syn::token::Bracket) {
            let content;
            syn::bracketed!(content in input);
            facets_of_tokens(&content, names, parse_names)?;
        } else {
            input.step(|cursor| match cursor.token_tree() {
                Some((_, next)) => Ok(((), next)),
                None => Err(cursor.error("unexpected end of tokens")),
            })?;
        }
    }
    Ok(())
}

/// The table's names and parse names, read from the `funcs!` invocation.
fn table_names() -> (Vec<String>, Vec<String>) {
    let file = parse_src(TABLE);
    let mut names = Vec::new();
    let mut parse_names = Vec::new();
    for item in &file.items {
        if let syn::Item::Macro(m) = item
            && m.mac.path.is_ident("funcs")
        {
            let walk = |input: ParseStream| facets_of_tokens(input, &mut names, &mut parse_names);
            walk.parse2(m.mac.tokens.clone())
                .unwrap_or_else(|e| panic!("funcs!: {e}"));
        }
    }
    (names, parse_names)
}

/// Every spelling of a table function: `name`, `name` in lower case, and
/// the `parse` names.
fn spellings() -> BTreeSet<String> {
    let (names, parse_names) = table_names();
    let mut all = BTreeSet::new();
    for n in &names {
        all.insert(n.clone());
        all.insert(n.to_ascii_lowercase());
    }
    all.extend(parse_names);
    all
}

/// The table is found and read: 101 functions (43 with a node, 50 library,
/// 8 internal) and the parse names of 17 of them — a reader that found
/// nothing would pass the ratchet vacuously.
#[test]
fn the_table_is_read() {
    let (names, parse_names) = table_names();
    assert!(names.len() >= 101, "{} names read from funcs!", names.len());
    assert!(
        parse_names.len() >= 17,
        "{} parse names read",
        parse_names.len()
    );
    for expected in ["sin", "Gamma", "H", "besselj", "__ei_asym"] {
        assert!(
            names.iter().any(|n| n == expected),
            "name `{expected}` not read"
        );
    }
    for expected in ["arcsin", "ceil", "heaviside", "w"] {
        assert!(
            parse_names.iter().any(|n| n == expected),
            "parse name `{expected}` not read"
        );
    }
    let all = spellings();
    assert!(all.contains("gamma") && all.contains("kroneckerdelta"));
}

#[test]
fn function_names_are_spelled_in_the_table_only() {
    let spellings = spellings();
    let mut found: BTreeMap<(String, String), usize> = BTreeMap::new();
    for rel in FILES {
        let file = parse_src(rel);
        let mut literals = Literals::default();
        literals.visit_file(&file);
        for l in literals.0 {
            if spellings.contains(&l) {
                *found.entry(((*rel).to_owned(), l)).or_insert(0) += 1;
            }
        }
    }
    let allowed: BTreeMap<(String, String), usize> = ALLOWLIST
        .iter()
        .map(|k| ((k.file.to_owned(), k.literal.to_owned()), k.count))
        .collect();
    assert_eq!(
        allowed.len(),
        ALLOWLIST.len(),
        "an allowlist entry is repeated"
    );
    let mut violations = Vec::new();
    for ((rel, literal), &count) in &found {
        let allowance = allowed
            .get(&(rel.clone(), literal.clone()))
            .copied()
            .unwrap_or(0);
        if count > allowance {
            violations.push(format!(
                "{rel}: the literal {literal:?} occurs {count} time(s), allowed {allowance} — \
                 it spells a function of the `funcs!` table"
            ));
        } else if count < allowance {
            violations.push(format!(
                "{rel}: the literal {literal:?} occurs {count} time(s), allowed {allowance} — \
                 tighten the allowlist"
            ));
        }
    }
    for (rel, literal) in allowed.keys() {
        if !found.contains_key(&(rel.clone(), literal.clone())) {
            violations.push(format!(
                "{rel}: {literal:?} is allowlisted but does not occur — tighten the allowlist"
            ));
        }
    }
    if !violations.is_empty() {
        panic!(
            "function-name ratchet: {} literal(s) out of step:\n\n{}\n\n\
             A literal that spells a function repeats a facet of the table: read the facet \
             instead (`node.func_head()`, then `spec().name`, `.tex`, `.pretty`, `.mathml`, \
             `.sort`; `Func::parsed` in the parser; CONTRIBUTING.md, \"How to Add a New \
             Function\"), adding a facet if the table lacks it.  If the literal does not \
             name that function, allowlist it with the reason.\n",
            violations.len(),
            violations.join("\n")
        );
    }
}
