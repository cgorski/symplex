//! SymPy oracle, 0.2 surface — `Ex::compile` (native closure) and
//! `Ex::to_c_fn` (C source) for expressions with special functions, checked
//! against SymPy `N(f(x))` at seeded random points.
//!
//! The C source is not compiled here (no toolchain dependency in the test
//! suite); we check that generation succeeds and mentions the argument, and
//! that the compiled closure agrees with the oracle.

mod v02_oracle_common;

use v02_oracle_common::*;

const KNOWN_BUGS: &[KnownBug] = &[];

#[test]
fn codegen_compile_matches_sympy_n() {
    run_with_known_bugs("codegen", "compile", KNOWN_BUGS, |ctx, fx| {
        let e = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let vars = fx.str_list("variables");
        let var_refs: Vec<&str> = vars.iter().map(String::as_str).collect();
        let f = match e.compile(&var_refs) {
            Ok(f) => f,
            Err(err) => return Status::NotImplemented(format!("compile: {err}")),
        };
        let pts = fx.eval_points("eval_points");
        if pts.is_empty() {
            return Status::SkippedOracle("no eval points".into());
        }
        let mut checked = 0;
        for pt in &pts {
            let Some(Num::Finite(re, im)) = pt.value else {
                continue;
            };
            if im.abs() > 1e-12 {
                continue; // real-valued closures only
            }
            let args: Vec<f64> = vars
                .iter()
                .map(|v| pt.subs.get(v).copied().unwrap_or(f64::NAN))
                .collect();
            let got = f.call(&args);
            checked += 1;
            // Compiled f64 special functions: 1e-10 relative.
            if !approx_eq_tol(got, re, 1e-10) {
                return Status::Fail(format!("compiled f({args:?}) = {got}, SymPy {re}"));
            }
        }
        if checked == 0 {
            return Status::SkippedOracle("no real-valued oracle points".into());
        }
        // C source generation must succeed for anything we can compile.
        match e.to_c_fn("f", &var_refs) {
            Ok(src) if src.contains("double f(") && vars.iter().all(|v| src.contains(v)) => {
                Status::Pass
            }
            Ok(src) => Status::Fail(format!("to_c_fn produced unexpected source: {src}")),
            Err(err) => Status::NotImplemented(format!("to_c_fn: {err}")),
        }
    });
}
