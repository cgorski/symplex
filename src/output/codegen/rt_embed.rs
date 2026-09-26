//! Embedding of the shared numeric runtime into generated Rust code.
//!
//! The source of `numeric_rt` is included verbatim at
//! build time and split on its `// @@begin NAME` / `// @@end NAME` markers.
//! Generated functions reference helpers as `symplex_rt::gamma(x)`; after a
//! function body has been emitted, `used_helpers` scans it and
//! `runtime_module` produces a `mod symplex_rt { … }` block containing
//! exactly the sections that are needed (plus transitive dependencies), with
//! the primitive math layer rewritten for the selected [`MathBackend`].

use super::MathBackend;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

const RUNTIME_SRC: &str = include_str!("numeric_rt.rs");

/// The name of the module emitted into generated code.
pub(crate) const MODULE_NAME: &str = "symplex_rt";

/// A marker-delimited section of the runtime source.
struct Section {
    name: String,
    /// Verbatim text between the markers (marker lines excluded).
    text: String,
    /// Functions defined in this section.
    fns: Vec<String>,
    /// Other sections whose functions are called from this one.
    deps: Vec<String>,
}

struct Runtime {
    sections: Vec<Section>,
    fn_to_section: BTreeMap<String, usize>,
}

fn runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(parse_runtime)
}

fn parse_runtime() -> Runtime {
    let mut sections: Vec<Section> = Vec::new();
    let mut current: Option<(String, String)> = None;
    for line in RUNTIME_SRC.lines() {
        let trimmed = line.trim_start();
        if let Some(name) = trimmed.strip_prefix("// @@begin ") {
            current = Some((name.trim().to_string(), String::new()));
        } else if let Some(name) = trimmed.strip_prefix("// @@end ") {
            if let Some((cur_name, text)) = current.take() {
                debug_assert_eq!(cur_name, name.trim());
                let fns = defined_fns(&text);
                sections.push(Section {
                    name: cur_name,
                    text,
                    fns,
                    deps: Vec::new(),
                });
            }
        } else if let Some((_, text)) = current.as_mut() {
            text.push_str(line);
            text.push('\n');
        }
    }
    let mut fn_to_section = BTreeMap::new();
    for (i, s) in sections.iter().enumerate() {
        for f in &s.fns {
            fn_to_section.insert(f.clone(), i);
        }
    }
    // Dependency analysis: a call `name(` to a function defined elsewhere.
    let all_fns: Vec<(String, usize)> =
        fn_to_section.iter().map(|(f, &i)| (f.clone(), i)).collect();
    for i in 0..sections.len() {
        let mut deps: BTreeSet<String> = BTreeSet::new();
        for (f, owner) in &all_fns {
            if *owner != i && calls(&sections[i].text, f) {
                deps.insert(sections[*owner].name.clone());
            }
        }
        sections[i].deps = deps.into_iter().collect();
    }
    Runtime {
        sections,
        fn_to_section,
    }
}

/// Names of all functions (`fn name(`) defined in `text`.
fn defined_fns(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find("fn ") {
        let before_ok = pos == 0
            || !rest[..pos]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
        let after = &rest[pos + 3..];
        if before_ok {
            let ident: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !ident.is_empty() && after[ident.len()..].starts_with('(') {
                out.push(ident);
            }
        }
        rest = &rest[pos + 3..];
    }
    out
}

/// True when `text` contains a call to `name(` (as a whole identifier).
fn calls(text: &str, name: &str) -> bool {
    let needle = format!("{name}(");
    let mut start = 0;
    while let Some(pos) = text[start..].find(&needle) {
        let abs = start + pos;
        let prev = text[..abs].chars().next_back();
        let is_ident_char = prev.is_some_and(|c| c.is_alphanumeric() || c == '_');
        // Skip definitions (`fn name(`) and identifier suffixes.
        let is_def = text[..abs].ends_with("fn ");
        if !is_ident_char && !is_def {
            return true;
        }
        start = abs + needle.len();
    }
    false
}

/// All public helper functions provided by the runtime, in source order.
pub(crate) fn all_helper_fns() -> Vec<String> {
    let rt = runtime();
    let mut out = Vec::new();
    for s in &rt.sections {
        for f in &s.fns {
            if s.text.contains(&format!("pub fn {f}(")) {
                out.push(f.clone());
            }
        }
    }
    out
}

/// Collect helper names referenced as `symplex_rt::NAME(` in generated code.
pub(crate) fn used_helpers(code: &str) -> BTreeSet<String> {
    let prefix = format!("{MODULE_NAME}::");
    let mut out = BTreeSet::new();
    let mut rest = code;
    while let Some(pos) = rest.find(&prefix) {
        let after = &rest[pos + prefix.len()..];
        let ident: String = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !ident.is_empty() {
            out.insert(ident);
        }
        rest = after;
    }
    out
}

/// Primitive math layer for the given backend (the `prim` section is
/// regenerated rather than copied so that generated code works on `no_std`
/// targets through `libm`).
fn prim_section(backend: MathBackend) -> String {
    // (name, arity, std body, libm body)
    const PRIMS: &[(&str, u8, &str, &str)] = &[
        ("p_exp", 1, "x.exp()", "libm::exp(x)"),
        ("p_ln", 1, "x.ln()", "libm::log(x)"),
        ("p_ln1p", 1, "x.ln_1p()", "libm::log1p(x)"),
        ("p_sin", 1, "x.sin()", "libm::sin(x)"),
        ("p_cos", 1, "x.cos()", "libm::cos(x)"),
        ("p_sqrt", 1, "x.sqrt()", "libm::sqrt(x)"),
        ("p_powf", 2, "x.powf(y)", "libm::pow(x, y)"),
        ("p_abs", 1, "x.abs()", "libm::fabs(x)"),
        ("p_floor", 1, "x.floor()", "libm::floor(x)"),
    ];
    let mut out = String::new();
    for &(name, arity, std_body, libm_body) in PRIMS {
        let params = if arity == 1 {
            "x: f64"
        } else {
            "x: f64, y: f64"
        };
        match backend {
            MathBackend::Std => {
                out.push_str(&format!(
                    "    #[inline(always)]\n    fn {name}({params}) -> f64 {{\n        {std_body}\n    }}\n"
                ));
            }
            MathBackend::Libm => {
                out.push_str(&format!(
                    "    #[inline(always)]\n    fn {name}({params}) -> f64 {{\n        {libm_body}\n    }}\n"
                ));
            }
            MathBackend::CfgGated => {
                out.push_str(&format!(
                    "    #[cfg(feature = \"std\")]\n    #[inline(always)]\n    fn {name}({params}) -> f64 {{\n        {std_body}\n    }}\n"
                ));
                out.push_str(&format!(
                    "    #[cfg(not(feature = \"std\"))]\n    #[inline(always)]\n    fn {name}({params}) -> f64 {{\n        {libm_body}\n    }}\n"
                ));
            }
        }
    }
    out
}

/// Build the `mod symplex_rt { … }` block providing `helper_fns` (and their
/// transitive dependencies).  Returns `None` when no helpers are requested.
pub(crate) fn runtime_module(backend: MathBackend, helper_fns: &[&str]) -> Option<String> {
    let rt = runtime();
    let mut needed: BTreeSet<usize> = BTreeSet::new();
    let mut stack: Vec<usize> = helper_fns
        .iter()
        .filter_map(|f| rt.fn_to_section.get(*f).copied())
        .collect();
    if stack.is_empty() {
        return None;
    }
    while let Some(i) = stack.pop() {
        if needed.insert(i) {
            for d in &rt.sections[i].deps {
                if let Some(j) = rt.sections.iter().position(|s| &s.name == d) {
                    stack.push(j);
                }
            }
        }
    }
    if let Some(consts) = rt.sections.iter().position(|s| s.name == "consts") {
        needed.insert(consts);
    }
    let mut out = String::new();
    out.push_str("#[allow(dead_code, clippy::all)]\n");
    out.push_str(&format!("mod {MODULE_NAME} {{\n"));
    out.push_str("    //! Special-function runtime generated by symplex; do not edit.\n");
    out.push_str(&prim_section(backend));
    for (i, s) in rt.sections.iter().enumerate() {
        if s.name == "prim" || !needed.contains(&i) {
            continue;
        }
        for line in s.text.lines() {
            if line.is_empty() {
                out.push('\n');
            } else {
                out.push_str("    ");
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out.push('}');
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sections_and_deps() {
        let rt = runtime();
        let names: Vec<&str> = rt.sections.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"prim"));
        assert!(names.contains(&"gamma"));
        assert!(names.contains(&"bessel_core"));
        let gamma = &rt.sections[rt.fn_to_section["gamma"]];
        assert!(gamma.fns.contains(&"lanczos_gamma".to_string()));
        assert!(gamma.deps.contains(&"util".to_string()));
        assert!(gamma.deps.contains(&"prim".to_string()));
        let bj = &rt.sections[rt.fn_to_section["bessel_j"]];
        assert!(bj.deps.contains(&"bessel_core".to_string()));
        let beta = &rt.sections[rt.fn_to_section["beta"]];
        assert!(beta.deps.contains(&"gamma".to_string()));
        assert!(beta.deps.contains(&"lgamma".to_string()));
    }

    #[test]
    fn module_contains_only_needed_sections() {
        let m = runtime_module(MathBackend::Std, &["erf"]).unwrap();
        assert!(m.starts_with("#[allow(dead_code, clippy::all)]\nmod symplex_rt {"));
        assert!(m.contains("pub fn erf("));
        assert!(m.contains("fn calerf("));
        assert!(m.contains("fn p_exp("));
        assert!(!m.contains("pub fn gamma("), "gamma not needed for erf");
        assert!(!m.contains("bessel"));
        assert!(m.ends_with('}'));
        // brace balance
        assert_eq!(m.matches('{').count(), m.matches('}').count());
    }

    #[test]
    fn transitive_dependencies_are_included() {
        let m = runtime_module(MathBackend::Std, &["beta"]).unwrap();
        assert!(m.contains("pub fn gamma("));
        assert!(m.contains("pub fn lgamma("));
        assert!(m.contains("fn sin_pi("));
        assert!(m.contains("const EULER_GAMMA"));
        let m = runtime_module(MathBackend::Std, &["bessel_y"]).unwrap();
        assert!(m.contains("fn bessel_miller("));
        assert!(m.contains("fn bessel_hankel("));
    }

    #[test]
    fn backends_change_prims_only() {
        let libm = runtime_module(MathBackend::Libm, &["gamma"]).unwrap();
        assert!(libm.contains("libm::exp(x)"));
        assert!(!libm.contains("x.exp()"));
        let cfg = runtime_module(MathBackend::CfgGated, &["gamma"]).unwrap();
        assert!(cfg.contains("#[cfg(feature = \"std\")]"));
        assert!(cfg.contains("#[cfg(not(feature = \"std\"))]"));
        assert!(cfg.contains("libm::pow(x, y)"));
        assert!(cfg.contains("x.powf(y)"));
        assert!(runtime_module(MathBackend::Std, &["not_a_helper"]).is_none());
        assert!(runtime_module(MathBackend::Std, &[]).is_none());
    }

    #[test]
    fn used_helpers_scans_code() {
        let code =
            "let t0 = symplex_rt::gamma(x);\n symplex_rt::bessel_j(2, t0) + symplex_rt::gamma(y)";
        let used: Vec<String> = used_helpers(code).into_iter().collect();
        assert_eq!(used, vec!["bessel_j".to_string(), "gamma".to_string()]);
        assert!(all_helper_fns().contains(&"lambert_w0".to_string()));
        assert!(!all_helper_fns().contains(&"calerf".to_string()));
    }
}
