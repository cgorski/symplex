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
//! `bounded_len(…)` / `clamped_len(…)` — or names a local bound by a `let`
//! (in the same function, earlier) whose value does, directly or through
//! another such local (`let k = d as usize; let n = k + 1; vec![0; n]`).
//! A later `let` of the same name without a cast (`let n =
//! bounded_len(n, CAP)?`) clears it.  Up to 0.58 only the length
//! expression itself was read, and about forty sites hid their cast in a
//! `let` (`combinatorics::stirling*_by_rows`, `BitSieve::new`, the dense
//! coefficient lists of `integrate` and `poly`).
//!
//! `#[cfg(test)]` items and modules are skipped.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::visit::Visit;

/// Files allowed to contain cast-sized allocations, with the exact count
/// (0.58: 4, down from 37 at 0.57).  0.59 sees lengths bound through a
/// `let`: 46 more sites in 20 files, of which 16 (in 9 files) are bounded
/// since and 30 are listed here (34 in 12 files in all).  Paths are
/// relative to `src/`.
/// Each entry says where its lengths are bounded, or that it is a review
/// item.
const ALLOWLIST: &[(&str, usize)] = &[
    // `sym_bits.resize(i + 1)`: `i` is the index of a symbol this arena has
    // interned (the table grows with the symbol table).
    ("base/arena.rs", 1),
    // `Source::Compose` / `Source::Reversion`: `k + 1` and `k` entries, the
    // size of the coefficient prefix `coeffs(0, k)` the same call has just
    // computed (the prefix itself is the review item: a user's `coeff(k)`).
    ("calculus/formal_series.rs", 2),
    // Another agent's write set at 0.59 (series orders and known prefixes):
    // review queue.
    ("calculus/series.rs", 4),
    // The sieves: `sieve_u32` has a `u32` limit, `BitSieve::new` the ECM
    // parameter table's `B₁`/`B₂` and the prime-index bound 10⁷, and
    // `segmented_sieve` a width ≤ 10⁷ (`primerange`) or a segment.
    ("domains/ntheory.rs", 3),
    // Another agent's write set at 0.59: review queue.
    ("domains/stats/anova.rs", 1),
    // The dense images of Brown's gcd (`image_degree_bounds`, `run_poly`,
    // `lc_main`, the evaluation points and images of `gcd_mod_p`) are sized
    // by degrees of the inputs, which `brown_gcd` bounds by
    // `cast::MAX_DENSE_LIST` on entry.
    ("poly/brown.rs", 7),
    // Dense univariate views (`to_uni`, `multipoly_to_uni`) of a factor in
    // one variable, and the coefficient lists of the Hensel error by
    // monomial: the degree is the data's (`u32`); review queue (their
    // callers return `Poly`, not `Option`).
    ("poly/factor_hensel.rs", 2),
    ("poly/factor_zassenhaus.rs", 1),
    // `Li_{−n}(z)`: `n ≤ MAX_SPECIAL_EXPANSION` by `as_int_in` just above.
    ("transforms/eval.rs", 1),
    // Another agent's write set at 0.59 (lengths from the working precision
    // and the Riemann–Siegel term count): review queue.
    ("transforms/evalf.rs", 7),
    ("transforms/evalf/rszeta.rs", 3),
    // Another agent's write set at 0.59 (dense coefficient lists of the
    // Risch-type routes): review queue.
    ("transforms/integrate.rs", 2),
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

/// Methods through which a value keeps being the number it was computed
/// from (`n.max(m)`, `n.checked_add(1)?`, `x.ok()?`): the spine of a
/// length.  Any other call (`v.len()`, `f(n as usize)`) yields a new value;
/// `a.min(b)` is cast-sized only when both are, `clamp` never.
const SPINE_METHODS: &[&str] = &[
    "ok",
    "unwrap",
    "expect",
    "unwrap_or",
    "unwrap_or_default",
    "unwrap_or_else",
    "max",
    "abs",
    "unsigned_abs",
    "div_ceil",
    "next_power_of_two",
    "pow",
    "saturating_add",
    "saturating_sub",
    "saturating_mul",
    "checked_add",
    "checked_sub",
    "checked_mul",
    "checked_div",
    "wrapping_add",
    "wrapping_sub",
];

/// Is the value of `e` computed from an unchecked conversion or from one
/// of the `locals` along its spine: casts, arithmetic, `?`, `Some(…)`,
/// branches, and the [`SPINE_METHODS`]?  `bounded_len`/`clamped_len` end
/// the spine (checked), and so does any other call: its result is not the
/// converted number (`round_decimal_digits(d, n as usize)` is a digit list).
fn spine_is_cast(e: &syn::Expr, locals: &HashSet<String>) -> bool {
    let tail = |b: &syn::Block| match b.stmts.last() {
        Some(syn::Stmt::Expr(x, None)) => spine_is_cast(x, locals),
        _ => false,
    };
    match e {
        syn::Expr::Cast(c) => {
            c.ty.to_token_stream().to_string() == "usize" || spine_is_cast(&c.expr, locals)
        }
        syn::Expr::Paren(x) => spine_is_cast(&x.expr, locals),
        syn::Expr::Group(x) => spine_is_cast(&x.expr, locals),
        syn::Expr::Try(x) => spine_is_cast(&x.expr, locals),
        syn::Expr::Unary(x) => spine_is_cast(&x.expr, locals),
        syn::Expr::Reference(x) => spine_is_cast(&x.expr, locals),
        // `deg − μ + 1`, `n / k`: what is taken away does not size the result.
        syn::Expr::Binary(b) => {
            spine_is_cast(&b.left, locals)
                || (!matches!(
                    b.op,
                    syn::BinOp::Sub(_)
                        | syn::BinOp::Div(_)
                        | syn::BinOp::Rem(_)
                        | syn::BinOp::Shr(_)
                ) && spine_is_cast(&b.right, locals))
        }
        syn::Expr::Tuple(t) => t.elems.iter().any(|x| spine_is_cast(x, locals)),
        syn::Expr::Block(b) => tail(&b.block),
        syn::Expr::If(i) => {
            tail(&i.then_branch)
                || i.else_branch
                    .as_ref()
                    .is_some_and(|(_, x)| spine_is_cast(x, locals))
        }
        syn::Expr::Match(m) => m.arms.iter().any(|a| spine_is_cast(&a.body, locals)),
        syn::Expr::Path(p) => {
            p.qself.is_none()
                && p.path.segments.len() == 1
                && locals.contains(&p.path.segments[0].ident.to_string())
        }
        syn::Expr::MethodCall(m) => {
            let name = m.method.to_string();
            if BOUNDERS.contains(&name.as_str()) || name == "clamp" {
                false
            } else if name == "min" {
                spine_is_cast(&m.receiver, locals)
                    && m.args.iter().all(|a| spine_is_cast(a, locals))
            } else if matches!(name.as_str(), "to_usize" | "try_into") {
                true
            } else if SPINE_METHODS.contains(&name.as_str()) {
                spine_is_cast(&m.receiver, locals)
                    || m.args.iter().any(|a| spine_is_cast(a, locals))
            } else {
                false
            }
        }
        syn::Expr::Call(c) => match &*c.func {
            syn::Expr::Path(p) => {
                let name = p.path.segments.last().map(|s| s.ident.to_string());
                match name.as_deref() {
                    Some(n) if BOUNDERS.contains(&n) => false,
                    Some("try_from" | "try_into" | "to_usize") => true,
                    Some("Some" | "Ok") => c.args.iter().any(|a| spine_is_cast(a, locals)),
                    _ => false,
                }
            }
            _ => false,
        },
        _ => false,
    }
}

/// The names a `let` pattern binds (`n`, `mut n`, `n: usize`, `Some(n)`,
/// `(a, b)`).
fn bound_names(p: &syn::Pat, out: &mut Vec<String>) {
    match p {
        syn::Pat::Ident(i) => out.push(i.ident.to_string()),
        syn::Pat::Type(t) => bound_names(&t.pat, out),
        syn::Pat::Reference(r) => bound_names(&r.pat, out),
        syn::Pat::Paren(x) => bound_names(&x.pat, out),
        syn::Pat::Tuple(t) => t.elems.iter().for_each(|e| bound_names(e, out)),
        syn::Pat::TupleStruct(t) => t.elems.iter().for_each(|e| bound_names(e, out)),
        _ => {}
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
    /// Per enclosing function: the locals whose value holds an unchecked
    /// conversion.
    cast_locals: Vec<HashSet<String>>,
}

impl Finder {
    fn is_cast_sized(&self, len: &syn::Expr) -> bool {
        has_cast(len)
            || self
                .cast_locals
                .last()
                .is_some_and(|l| spine_is_cast(len, l))
    }

    fn check(&mut self, what: &str, len: &syn::Expr, whole: &impl ToTokens) {
        if self.is_cast_sized(len) {
            self.sites.push(format!("{what}: {}", site(whole)));
        }
    }

    fn in_fn(&mut self, visit: impl FnOnce(&mut Self)) {
        self.cast_locals.push(HashSet::new());
        visit(self);
        self.cast_locals.pop();
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
            self.in_fn(|s| syn::visit::visit_impl_item_fn(s, f));
        }
    }

    fn visit_item_fn(&mut self, f: &'ast syn::ItemFn) {
        self.in_fn(|s| syn::visit::visit_item_fn(s, f));
    }

    fn visit_local(&mut self, l: &'ast syn::Local) {
        syn::visit::visit_local(self, l);
        let mut names = Vec::new();
        bound_names(&l.pat, &mut names);
        let empty = HashSet::new();
        let locals = self.cast_locals.last().unwrap_or(&empty);
        let cast = l
            .init
            .as_ref()
            .is_some_and(|i| spine_is_cast(&i.expr, locals));
        if let Some(locals) = self.cast_locals.last_mut() {
            for n in names {
                if cast {
                    locals.insert(n);
                } else {
                    locals.remove(&n);
                }
            }
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
        fn through_lets(d: u32, b: &BigInt) -> Option<()> {
            let k = d as usize;
            let m = k + 1;
            let a = vec![0u8; m];
            let Some(t) = b.to_usize() else { return None };
            let c: Vec<u8> = Vec::with_capacity(t);
            let k = cast::bounded_len(d, 10)?;
            let ok4 = vec![0u8; k + 1];
            let s = v.len();
            let ok5 = Vec::<u8>::with_capacity(s);
            let (digits, carry) = round(&m, d as usize);
            let w = digits.len();
            let ok7 = String::with_capacity(w + 2);
            let q = usize::try_from(d).ok()?.max(100);
            let e = vec![0u8; q];
            let r = (d as usize).min(100);
            let ok8 = vec![0u8; r];
            None
        }
        fn other_fn(k: usize) { let ok6 = vec![0u8; k]; }
        #[cfg(test)]
        mod tests {
            fn t(n: i64) { let _ = Vec::with_capacity(n as usize); }
        }
    "#;
    let file = syn::parse_file(src).expect("parses");
    let mut f = Finder::default();
    f.visit_file(&file);
    assert_eq!(f.sites.len(), 9, "{:#?}", f.sites);
}
