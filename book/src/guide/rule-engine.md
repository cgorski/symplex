# The Rule Engine

New in 0.2: the pattern-matching engine behind `simplify` is public. You can write your own rewrite rules, apply them to a fixpoint or a single pass, trace what fired, and interleave them with the built-in simplifier.

Everything lives in the prelude: `Rule`, `RuleSet`, `Bindings`, `RewriteOpts`, `RewriteStrategy`, `Step`, and the `Ex` methods `rewrite`, `rewrite_once`, `rewrite_traced`, `rewrite_with`, `rewrite_with_traced`, `simplify_with_rules`, `simplify_traced`.

## Wildcards

A rule is built from two ordinary expressions. Any symbol whose name ends in `_` is a **wildcard**; a name ending in `__` is a **sequence wildcard** that absorbs the remaining terms of a sum or product (possibly none, binding to `0` or `1`).

| Pattern symbol | Matches |
|----------------|---------|
| `a_` | any single sub-expression (inside `Add`/`Mul`: one term, or the remaining terms if it is the last plain wildcard) |
| `rest__` | the remaining terms of an `Add`/`Mul` |

`Add` and `Mul` are matched associatively and commutatively with a bounded backtracking search; every other node is matched structurally.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let (a, b) = (ctx.symbol("a_"), ctx.symbol("b_"));

    let sin_sq = Rule::new("sin_sq", &a.sin().powi(2), &(1 - &a.cos().powi(2)));
    let ln_add = Rule::new("ln_add", &(&a.ln() + &b.ln()), &(&a * &b).ln());
    let rules = RuleSet::from_rules(vec![sin_sq.clone(), ln_add]);

    println!("{}", (&x.sin().powi(2) + 3).rewrite(&rules));                 // -cos(x)^2 + 4
    println!("{}", (&x.ln() + &y.ln() + &(&x + 1).ln()).rewrite(&rules));  // ln(x*y*(x + 1))

    // Inspect a match
    let bindings = sin_sq.matches(&(&x * 2).sin().powi(2)).unwrap();
    println!("{}", bindings.get("a_").unwrap());                            // 2*x
}
```

`Rule::try_new` rejects rules whose right-hand side mentions an unbound wildcard, or whose left-hand side is a bare wildcard.

## Guards and closure right-hand sides

`Rule::new_with_guard` takes a predicate on the `Bindings`; `Rule::new_fn` computes the replacement in a closure (returning `None` means "does not apply"). Guards and closures run without the context lock, so any `Ex` method is allowed.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let a = ctx.symbol("a_");
    let rest = ctx.symbol("rest__");

    // exp(3 + x + y) → exp(3)·exp(x + y), but only for a numeric term and a
    // non-empty remainder (otherwise exp(3) alone would match forever).
    let split = Rule::new_with_guard(
        "exp_split",
        &(&a + &rest).exp(),
        &(&a.exp() * &rest.exp()),
        |b| {
            b.get("a_").is_some_and(|v| v.expr_type() == ExprType::Number)
                && b.get("rest__").is_some_and(|r| !r.is_zero_structural())
        },
    );
    println!("{}", (&x + &y + 3).exp().rewrite(&RuleSet::from_rules(vec![split])));
    // exp(3)*exp(x + y)

    // Evaluate small factorials, leave the rest alone.
    let fact = Rule::new_fn("small_factorial", &a.factorial(), |b| {
        let n = b.get("a_")?.as_i64()?;
        (0..=20).contains(&n).then(|| b.get("a_").unwrap().context().from_i128((1..=n as i128).product()))
    });
    let e = &ctx.int(6).factorial() + &x.factorial() + &ctx.int(30).factorial();
    println!("{}", e.rewrite(&RuleSet::from_rules(vec![fact])));            // 30! + x! + 720
}
```

## The `rule!` macro

Rules can also be written in mathematical notation with the `rule!` macro, which works on the arena inside `ctx.with_arena_mut`; wrap the result with `RuleSet::from_macro_rules`.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let raw = ctx.with_arena_mut(|arena| {
        vec![
            rule!(arena, "pyth", sin(w_)^2 + cos(w_)^2 => 1),
            rule!(arena, "double_angle", 2 * sin(w_) * cos(w_) => sin(2 * w_)),
        ]
    });
    let trig = RuleSet::from_macro_rules(&ctx, raw);
    let e = &x.sin().powi(2) + &x.cos().powi(2) + &x.sin() * &x.cos() * 2;
    println!("{}", e.rewrite(&trig));                                       // sin(2*x) + 1
}
```

## Strategies and iteration

`rewrite` iterates bottom-up to a fixpoint (at most `RewriteOpts::default().max_iterations = 50` passes; a tree-size guard stops runaway growth). `rewrite_once` is a single pass. `rewrite_with(&rules, &opts)` selects `RewriteStrategy::{BottomUp, TopDown, Innermost}` and the iteration cap.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let a = ctx.symbol("a_");
    let flatten = RuleSet::from_rules(vec![Rule::new("flatten", &a.exp().exp(), &a.exp())]);
    let nested = x.exp().exp().exp().exp();
    println!("{}", nested.rewrite_once(&flatten));
    println!("{}", nested.rewrite(&flatten));                               // exp(x)
    let opts = RewriteOpts::default().strategy(RewriteStrategy::TopDown).max_iterations(5);
    println!("{}", nested.rewrite_with(&flatten, &opts));                   // exp(x)
}
```

## Tracing

`rewrite_traced` returns every rule application as a `Step { rule_name, before, after }`. `simplify_traced(&SimplifyOpts)` does the same for the built-in simplifier: strategy-level entries are named `strategy:<name>` and rule-level entries carry the rule name. `RuleSet::standard(&ctx)` gives you the built-in rules as a `RuleSet`.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let e = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
    let (result, steps) = e.simplify_traced(&SimplifyOpts::default());
    println!("{e} → {result}");                                             // … → 1
    for s in &steps {
        println!("[{}] {} ⇒ {}", s.rule_name, s.before, s.after);
    }
    println!("{} standard rules", RuleSet::standard(&ctx).len());          // 22
}
```

## Interleaving with the simplifier

`simplify_with_rules(&extra)` alternates `simplify()` with your rules until nothing changes — the way to teach the simplifier an identity it does not know.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let a = ctx.symbol("a_");
    let sinh_def = (&a.exp() - &(-&a).exp()) / 2;
    let extra = RuleSet::from_rules(vec![Rule::new("sinh_def", &a.sinh(), &sinh_def)]);
    let e = &x.sinh() - &(&x.exp() - &(-&x).exp()) / 2;
    println!("{}", e.simplify());                    // unchanged: simplify does not know sinh_def
    println!("{}", e.simplify_with_rules(&extra));   // 0
}
```

## Algebraic substitution

`subs` is structural: `x⁴.subs(x², u)` leaves `x⁴` alone. `subs_algebraic` recognises the old expression inside powers, products and sums; every rewrite is an identity in `old`.

| `self` | `old` | `subs_algebraic` |
|--------|-------|------------------|
| `x^4` | `x^2` | `u^2` |
| `x^3` | `x^2` | `u*x` |
| `1/x^2` | `x^2` | `1/u` |
| `x^6 + 3x^2 + 1` | `x^2` | `u^3 + 3u + 1` |
| `exp(2x)` | `exp(x)` | `u^2` |

See `cargo run --example rule_engine` for the full tour.
