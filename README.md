# Symplex

A fast, correct symbolic mathematics library for Rust.

## Design Principles

1. **Construction is cheap, evaluation is explicit.** Building `x + y` never triggers expansion, simplification, or function evaluation. Call `.eval()`, `.expand()`, or `.simplify()` when *you* choose.
2. **Never silently wrong.** Operations return `Result::Err` instead of silent wrong answers. Structural substitution by default — no algebraic guesses.
3. **One representation per concept.** One assumption system. One polynomial type. One number type. One solve function.
4. **Thread-safe from day one.** `Ex` is `Send + Sync`. Parallel batch simplification with `rayon` just works.
5. **No recursive tree walks.** All traversals use explicit stacks. No stack overflow at any depth.
6. **The compiler is the API contract.** `pub` = stable, semver-protected. Internal types are `pub(crate)` — invisible and free to change.
7. **Extensible without inheritance.** Custom functions via name + registered rules. No subclassing, no metaclasses.

## Quick Start

```rust
use symplex::prelude::*;

let ctx = Context::new();
let x = ctx.symbol("x");

with_ctx(&ctx, || {
    // Build expressions with natural operators
    let expr = (x + 1).pow(3);
    println!("{}", expr);            // (x + 1)**3

    // Expand — explicit, never automatic
    println!("{}", expr.expand());   // x**3 + 3*x**2 + 3*x + 1

    // Differentiate
    println!("{}", expr.diff(x));    // 3*(x + 1)**2

    // Simplify trig identities
    let trig = x.sin().pow(2) + x.cos().pow(2);
    println!("{}", trig.simplify()); // 1

    // Assumptions
    let t = ctx.symbol_with("t", &[Assumption::Positive]);
    assert_eq!((t + 1).is_positive(), Some(true));

    // Expressions are never auto-expanded
    assert_eq!((x + 1).pow(2).to_string(), "(x + 1)**2");
});
```

## Why Symplex?

| | SymPy | Symplex |
|---|---|---|
| `(a+b+c)^20` expand | ~5 seconds | < 50ms |
| Auto-evaluation | Can't turn off | Never automatic |
| Thread safety | GIL-bound | `Send + Sync` day one |
| Wrong answers from `subs` | Silent | Structural subs by default |
| Memory leaks | Known, unfixable | Arena with generational cleanup |
| Timeout support | None | Built-in `CancelToken` |
| Stack overflow on deep expressions | Python recursion limit | Explicit stacks, never recurse |
| `evalf` silently wrong | Returns 0 for nonzero values | `Result::Err` on precision failure |
| Import time | 370+ ms | Zero (compiled Rust) |

## Architecture

Symplex uses an **arena-interned expression tree**:

- Every expression is an `ExprId` — a 4-byte index into an append-only arena.
- Structurally identical expressions share the same `ExprId` (hash-consing).
- Equality is a single `u32` comparison: **O(1)**.
- Hashing is a single `u32` hash: **O(1)**.
- The `Ex` user-facing handle is 8 bytes (`CtxId` + `ExprId`) and implements `Copy`.

Numbers are stored in a side table (`Ratio<BigInt>` from `num-rational`), keeping expression nodes slim (~32 bytes each).

## Feature Flags

| Feature | Default | Description |
|---|---|---|
| `evalf` | ✅ | Numerical evaluation via `astro-float`. Disable for smaller binaries. |

## Minimum Rust Version

Rust 1.93+ with Edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.