//! 0.2 numeric back-ends: common subexpression elimination (`cse`, `cse_many`).

use symplex::prelude::*;

/// Substitute the bindings back and check value preservation numerically.
fn assert_cse_preserves(expr: &Ex, bindings: &[(Ex, Ex)], rewritten: &Ex, vars: &[&str]) {
    let mut restored = rewritten.clone();
    for (name, value) in bindings.iter().rev() {
        restored = restored.subs(name, value);
    }
    let f = expr.compile(vars).unwrap();
    let g = restored.compile(vars).unwrap();
    for i in 0..5 {
        let args: Vec<f64> = (0..vars.len())
            .map(|k| 0.3 + i as f64 * 0.7 + k as f64)
            .collect();
        let (a, b) = (f(&args), g(&args));
        assert!(
            (a - b).abs() <= 1e-12 * (1.0 + a.abs()),
            "CSE changed value at {args:?}: {a} vs {b}"
        );
    }
}

#[test]
fn bindings_are_deterministic_and_ordered_by_first_occurrence() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = x.sin();
    let es = s.exp();
    // exp(sin(x)) + exp(sin(x))^2 + sin(x)*cos(x)
    let e = &(&es + &es.powi(2)) + &(&s * &x.cos());
    let (b1, r1) = e.cse();
    let (b2, r2) = e.cse();
    assert_eq!(b1.len(), b2.len());
    for ((n1, v1), (n2, v2)) in b1.iter().zip(&b2) {
        assert_eq!(format!("{n1}"), format!("{n2}"));
        assert_eq!(format!("{v1}"), format!("{v2}"));
    }
    assert_eq!(format!("{r1}"), format!("{r2}"));
    // sin(x) occurs before exp(sin(x)) in post-order → __cse_0 = sin(x).
    assert_eq!(format!("{}", b1[0].0), "__cse_0");
    assert_eq!(format!("{}", b1[0].1), "sin(x)");
    assert_eq!(format!("{}", b1[1].1), "exp(__cse_0)");
    assert_cse_preserves(&e, &b1, &r1, &["x"]);
}

#[test]
fn cheap_nodes_need_three_uses() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x^2 used twice: not worth a temporary.
    let e = &x.powi(2).sin() + &x.powi(2).cos();
    let (b, _) = e.cse();
    assert!(
        b.iter().all(|(_, v)| format!("{v}") != "x^2"),
        "x^2 used twice should stay inline: {b:?}"
    );
    // x^2 used three times: extracted.
    let e3 = &(&x.powi(2).sin() + &x.powi(2).cos()) + &x.powi(2).exp();
    let (b3, r3) = e3.cse();
    assert!(
        b3.iter().any(|(_, v)| format!("{v}") == "x^2"),
        "x^2 used thrice should be extracted: {b3:?}"
    );
    assert_cse_preserves(&e3, &b3, &r3, &["x"]);
    // Expensive nodes are extracted at two uses.
    let e2 = &(&x * &y).sin() + &(&x * &y).cos();
    let (b2, _) = e2.cse();
    assert!(b2.iter().any(|(_, v)| format!("{v}") == "x*y"), "{b2:?}");
    // Negated atoms are never worth it at two uses.
    let en = &(-&x).exp() + &(-&x).sin();
    let (bn, _) = en.cse();
    assert!(bn.is_empty(), "negations are cheap: {bn:?}");
}

#[test]
fn boolean_nodes_are_never_extracted() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let c = x.gt(&y);
    let pw = Ex::piecewise(&[
        (&x, &c.and(&x.gt(&ctx.zero()))),
        (&y, &c.not()),
        (&ctx.zero(), &c),
    ]);
    let (b, r) = pw.cse();
    assert!(b.is_empty(), "conditions must stay native booleans: {b:?}");
    let code = r.to_rust_fn("pw", &["x", "y"]).unwrap();
    assert!(!code.contains("let t"), "{code}");
    let ccode = pw.to_c_fn("pw", &["x", "y"]).unwrap();
    assert!(!ccode.contains("const double t"), "{ccode}");
}

#[test]
fn cse_many_shares_across_expressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &(&x * &y).sin() * &x.exp();
    let fx = f.diff(&x);
    let fy = f.diff(&y);
    let (bindings, exprs) = Ex::cse_many(&[&fx, &fy]);
    assert_eq!(exprs.len(), 2);
    assert!(
        !bindings.is_empty(),
        "gradient components share cos(x*y), exp(x), x*y"
    );
    let names: Vec<String> = bindings.iter().map(|(n, _)| format!("{n}")).collect();
    for (i, n) in names.iter().enumerate() {
        assert_eq!(n, &format!("__cse_{i}"), "sequential naming");
    }
    // Every binding value only references earlier bindings.
    for (i, (_, v)) in bindings.iter().enumerate() {
        let s = format!("{v}");
        for j in i..bindings.len() {
            assert!(
                !s.contains(&format!("__cse_{j}")),
                "binding {i} refers to later binding {j}: {s}"
            );
        }
    }
    // Value preservation for each output.
    let fxc = fx.compile(&["x", "y"]).unwrap();
    let fyc = fy.compile(&["x", "y"]).unwrap();
    let mut rx = exprs[0].clone();
    let mut ry = exprs[1].clone();
    for (n, v) in bindings.iter().rev() {
        rx = rx.subs(n, v);
        ry = ry.subs(n, v);
    }
    let rxc = rx.compile(&["x", "y"]).unwrap();
    let ryc = ry.compile(&["x", "y"]).unwrap();
    for p in [(0.3, 1.1), (2.0, -0.7), (-1.5, 0.4)] {
        let a = [p.0, p.1];
        assert!((fxc(&a) - rxc(&a)).abs() < 1e-12 * (1.0 + fxc(&a).abs()));
        assert!((fyc(&a) - ryc(&a)).abs() < 1e-12 * (1.0 + fyc(&a).abs()));
    }
    // Empty input.
    let (b0, e0) = Ex::cse_many(&[]);
    assert!(b0.is_empty() && e0.is_empty());
    // Single input agrees with cse().
    let (b1, e1) = Ex::cse_many(&[&fx]);
    let (b1s, e1s) = fx.cse();
    assert_eq!(b1.len(), b1s.len());
    assert_eq!(format!("{}", e1[0]), format!("{e1s}"));
}

#[test]
fn compile_many_uses_shared_cse() {
    // Compare instruction counts: the shared program should be smaller than
    // the sum of the independent programs.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &(&x * &y).sin() * &(&x * &y).exp() + &(&x * &y).cos();
    let fx = f.diff(&x);
    let fy = f.diff(&y);
    let shared = Ex::compile_many(&[&fx, &fy], &["x", "y"]).unwrap();
    let separate = fx.compile(&["x", "y"]).unwrap().instruction_count()
        + fy.compile(&["x", "y"]).unwrap().instruction_count();
    assert!(
        shared.instruction_count() < separate,
        "shared {} vs separate {}",
        shared.instruction_count(),
        separate
    );
}
