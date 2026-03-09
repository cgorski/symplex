# Chapter 3: Common Gotchas

New symplex users run into the same handful of surprises. This chapter covers them all so you can avoid the confusion.

## Gotcha 1: Fraction Literals in `expr!`

The `expr!` macro detects integer-over-integer patterns and constructs exact rationals automatically:

```rust
use symplex::prelude::*;

let half = expr!(1/2);
println!("{half}"); // 1/2 — exact rational, not 0.5
```

This is usually what you want. But be aware: `expr!(1/2)` is *not* integer division — it's `ctx.rational(1, 2)`. If you genuinely want the integer quotient (which would be `0`), compute it outside the macro:

```rust
let ctx = Context::new();
let quotient = ctx.int(1i64 / 2); // 0
```

If you need a rational outside the macro, use `ctx.rational()` directly:

```rust
let ctx = Context::new();
let one_third = ctx.rational(1, 3);
let two_fifths = ctx.rational(2, 5);
```

## Gotcha 2: `Ex` Is Clone, Not Copy

The expression type `Ex` is `Clone + Send + Sync`, but it is **not** `Copy`. This means you can't just use a variable multiple times without borrowing or cloning:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// ❌ This won't compile — x is moved on first use:
// let f = x + x;

// ✅ Use references:
let f = &x + &x;

// ✅ Or use expr! which handles this for you:
let f = expr!(x + x);
```

The `expr!` macro is the easiest workaround — it manages references internally. When building expressions manually, use `&` liberally:

```rust
let ctx = Context::new();
syms!(ctx; x, y);

// Manual construction — note the &'s
let f = &x.powi(2) + &(&x * &y) + &y.powi(2);

// Much cleaner with expr!
let f = expr!(x^2 + x*y + y^2);
```

### Mixed Arithmetic with Integers

Rust's operator overloading lets you mix `&Ex` with `i64` in some positions:

```rust
let ctx = Context::new();
syms!(ctx; x);

let f = &x * 2 + 1;        // OK: &Ex * i64 + i64
let g = &x.powi(2) - &x * 5 + 6; // OK
```

But the order matters — `2 * &x` may not work depending on the trait implementations. When in doubt, use `expr!` or construct constants explicitly:

```rust
let ctx = Context::new();
let two = ctx.int(2);
let f = &two * &x; // always works
```

## Gotcha 3: Structural vs Mathematical Equality

Rust's `==` operator on `Ex` checks **structural** equality — are these the same expression tree? It does *not* check mathematical equivalence:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let a = expr!(x^2 + 2*x + 1);
let b = expr!((x + 1)^2);

// Structural equality: these are different trees
assert_ne!(format!("{a}"), format!("{b}"));
```

Even though `x² + 2x + 1 = (x+1)²` mathematically, the two expressions have different internal representations until you transform them.

To check mathematical equality, use `.equals()`:

```rust
let are_equal = a.equals(&b.expand());
println!("{:?}", are_equal); // Some(true)
```

The `.equals()` method returns `Option<bool>`:
- `Some(true)` — provably equal
- `Some(false)` — provably not equal
- `None` — couldn't determine

It tries several strategies (structural identity, difference-is-zero, expand-and-check). If you need a definitive answer, expand both sides first:

```rust
let equal = a.expand().equals(&b.expand());
```

## Gotcha 4: `^` Is XOR in Rust (Except in `expr!`)

In standard Rust, the `^` operator is bitwise XOR, not exponentiation. This means:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// ❌ WRONG: this is bitwise XOR, not x²
// let f = x ^ 2; // Won't compile — Ex doesn't implement BitXor

// ✅ Use .powi() for integer powers:
let f = x.powi(2);

// ✅ Use .pow() for symbolic powers:
let g = x.pow(&ctx.rational(1, 2)); // x^(1/2) = √x

// ✅ Use expr! where ^ means power:
let f = expr!(x^2);
let g = expr!(x^(1/2));
```

Inside `expr!()`, the `^` operator is reinterpreted as exponentiation. Outside the macro, always use `.powi()` or `.pow()`.

## Gotcha 5: `simplify()` Is Heuristic

The `.simplify()` method applies a single pass of rewrite rules. It's fast, but it doesn't guarantee the "simplest" form:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// simplify() handles this:
let a = expr!(sin(x)^2 + cos(x)^2);
println!("{}", a.simplify()); // 1

// But it may not handle everything in one pass:
let b = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
println!("{}", b.simplify()); // might not be "1" in one pass
println!("{}", b.full_simplify()); // 1 — multiple passes
```

### When to Use What

| Goal | Method |
|------|--------|
| Quick single-pass simplification | `.simplify()` |
| Fixpoint simplification (expand + eval + simplify loop) | `.full_simplify()` |
| Distribute products over sums | `.expand()` |
| Factor a polynomial | `.factor(&x)` |
| Trig identity simplification | `.simplify_trig()` |
| Trig expansion (double-angle, etc.) | `.expand_trig()` |
| Log expansion (`ln(xy) → ln(x)+ln(y)`) | `.expand_log()` |
| Log combination (`ln(x)+ln(y) → ln(xy)`) | `.log_combine()` |
| Power simplification | `.simplify_powers()` |
| Cancel common factors in a rational | `.simplify_rational()` |
| Partial fraction decomposition | `.partial_fractions(&x)` |
| Evaluate known special values (`sin(0)→0`) | `.eval()` |

Use the *specific* function when you know what transformation you need. Reserve `.simplify()` and `.full_simplify()` for "just make it simpler" situations.

## Gotcha 6: Immutable Expressions

All transformation methods return **new** expressions. Nothing is modified in place:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let f = expr!(x^2 + 2*x + 1);
let expanded = f.expand(); // new expression
println!("{f}");            // f is unchanged
println!("{expanded}");     // expanded form
```

This is true for every method: `.diff()`, `.integrate()`, `.subs()`, `.simplify()`, `.expand()`, `.factor()`, etc. They all return a new `Ex`.

## Gotcha 7: `eval()` vs `simplify()` vs `expand()`

These three are often confused:

- **`.eval()`** — Replaces function applications with exact values when arguments are known constants. `sin(0) → 0`, `exp(0) → 1`, `sqrt(4) → 2`. Does *not* expand or rearrange.

- **`.expand()`** — Distributes multiplication over addition and expands integer powers of sums. `(x+1)² → x² + 2x + 1`. Does *not* evaluate functions or apply identities.

- **`.simplify()`** — Applies rewrite rules like `sin²(x) + cos²(x) → 1`. Heuristic, single pass. May not catch everything.

They compose well:

```rust
let ctx = Context::new();
syms!(ctx; x);

let f = expr!((x + 1)^2 - x^2 - 2*x);
println!("{}", f.eval());           // (x + 1)^2 - x^2 - 2*x (no change — nothing to evaluate)
println!("{}", f.expand());         // 1 (expand distributes, canonicalization collects)
println!("{}", f.full_simplify());  // 1 (full_simplify runs all passes)
```

## Gotcha 8: Solve Returns a `Vec`, Not a Single Value

Equations can have multiple solutions. `.solve()` always returns `Result<Vec<Ex>, SymplexError>`:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let roots = expr!(x^2 - 1).solve_or_empty(&x);
// roots = [1, -1] — both solutions

for root in &roots {
    println!("x = {root}");
}
```

Use `.solve_or_empty()` when you don't care about error diagnostics — it returns `Vec::new()` on failure. Use `.solve()` when you want the `Result` to inspect the error.

The solver handles polynomials up to degree 4. For higher degrees or non-polynomial equations, it returns an error. See [Chapter 6: Solving](06-solving.md) for transcendental and numerical alternatives.

## Summary

| Trap | Fix |
|------|-----|
| `x + x` moves `x` twice | Use `&x + &x` or `expr!(x + x)` |
| `x ^ 2` is XOR | Use `x.powi(2)` or `expr!(x^2)` |
| `==` checks structure, not math | Use `.equals()` or compare after `.expand()` |
| `simplify()` misses things | Try `.full_simplify()` or a specific function |
| Expecting mutation | All methods return new expressions |
| `expr!(1/2)` confusion | It's an exact rational — by design |

---

*[← Chapter 2: Getting Started](02-getting-started.md) | [Back to Table of Contents](index.md) | [Chapter 4: Simplification →](04-simplification.md)*