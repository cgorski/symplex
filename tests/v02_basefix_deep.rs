//! symplex 0.2 base-layer fixes — deeply nested expressions must not blow
//! the stack during construction, display, evaluation or substitution.
//!
//! Every test runs in a thread with the default 2 MiB test stack so a
//! regression to recursive tree walks shows up as a crash here.

use symplex::prelude::*;

const STACK: usize = 2 * 1024 * 1024;

fn on_small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("deep-expression thread panicked or overflowed its stack")
}

fn nest_sin_plus_one(depth: usize) -> (Context, Ex, Ex) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut e = x.clone();
    for _ in 0..depth {
        e = e.sin() + 1;
    }
    (ctx, x, e)
}

#[test]
fn build_2000_nested_sin_plus_one() {
    on_small_stack(|| {
        let (_ctx, _x, e) = nest_sin_plus_one(2000);
        assert_eq!(e.args().len(), 2);
    });
}

#[test]
fn build_2000_nested_products_and_powers() {
    on_small_stack(|| {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let mut e = x.clone();
        for k in 0..2000 {
            e = if k % 3 == 0 {
                &e * &x + 1
            } else if k % 3 == 1 {
                (&e + 2).powi(2)
            } else {
                -&e + &x
            };
        }
        assert!(!e.is_constant());
    });
}

#[test]
fn display_deep_expression() {
    // `Display` uses an explicit work stack.  (`to_latex`, `pretty` and the
    // `ExprTree` serialiser are still recursive and are *not* exercised at
    // this depth.)
    on_small_stack(|| {
        let (_ctx, _x, e) = nest_sin_plus_one(2000);
        let s = e.to_string();
        assert!(s.starts_with("sin("), "{}", &s[..20]);
        assert_eq!(s.matches("sin(").count(), 2000);
    });
}

#[test]
fn parser_rejects_excessive_nesting_cleanly() {
    // The recursive-descent parser is *bounded* rather than iterative: a
    // nesting depth beyond its limit is a clean ParseError, never a stack
    // overflow.
    on_small_stack(|| {
        let (ctx, _x, e) = nest_sin_plus_one(2000);
        let s = e.to_string();
        let err = ctx.parse(&s).err().expect("depth limit");
        assert!(err.to_string().contains("nesting too deep"), "{err}");
        // Well inside the limit round-trips exactly.
        let (ctx, _x, e) = nest_sin_plus_one(100);
        assert_eq!(ctx.parse(&e.to_string()).unwrap(), e);
    });
}

#[test]
fn eval_subs_free_symbols_on_deep_expression() {
    on_small_stack(|| {
        let (ctx, x, e) = nest_sin_plus_one(2000);
        assert_eq!(e.free_symbols(), vec![x.clone()]);
        let ev = e.eval();
        assert_eq!(ev, e);
        let sub = e.subs(&x, &ctx.int(0));
        assert!(sub.is_constant());
        let v = sub.eval_f64().unwrap();
        assert!(v.is_finite());
    });
}

#[test]
fn diff_of_deep_expression_does_not_overflow() {
    // Depth kept moderate: the chain rule makes the derivative quadratic
    // in size, which is a cost question, not a stack question.
    on_small_stack(|| {
        let (_ctx, x, e) = nest_sin_plus_one(300);
        let d = e.diff(&x);
        assert!(!d.is_constant());
    });
}
