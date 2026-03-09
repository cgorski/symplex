# Chapter 17: Advanced Expression Manipulation

Every expression in symplex is a directed acyclic graph (DAG) of nodes — symbols, numbers, operators, and functions. This chapter covers the tools for inspecting that structure, walking and rewriting trees, extracting polynomial metadata, and serializing expressions for storage or interop.

## Inspecting Expressions

### Free Symbols

`.free_symbols()` returns every symbolic variable that appears in an expression:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y, z);

let f = expr!(x^2 + 2*x*y + z);
let syms = f.free_symbols();
println!("Free symbols: {}", syms.len()); // 3

for s in &syms {
    println!("  {s}");
}
// x, y, z (order is deterministic but unspecified)
```

Constants like `π` and `e` are *not* free symbols — they're built-in atoms:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let f = &x * ctx.pi();
let syms = f.free_symbols();
assert_eq!(syms.len(), 1); // only x, not π
```

### Containment Check

`.contains()` checks whether a subexpression appears anywhere in the tree:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

let f = expr!(sin(x^2) + y);

assert!(f.contains(&x));             // x appears
assert!(f.contains(&y));             // y appears
assert!(f.contains(&x.powi(2)));     // x^2 appears
assert!(f.contains(&x.powi(2).sin())); // sin(x^2) appears
assert!(!f.contains(&x.powi(3)));    // x^3 does not appear
```

This is a structural check — it walks the DAG and compares node identities.

### Expression Classification

`.expr_type()` returns an `ExprType` enum that classifies the top-level node:

```rust
use symplex::prelude::*;
use symplex::expr::ExprType;

let ctx = Context::new();
syms!(ctx; x);

assert_eq!(ctx.int(42).expr_type(), ExprType::Number);
assert_eq!(x.expr_type(), ExprType::Symbol);
assert_eq!(ctx.pi().expr_type(), ExprType::Constant);
assert_eq!((&x + 1).expr_type(), ExprType::Add);
assert_eq!((&x * 2).expr_type(), ExprType::Mul);
assert_eq!(x.powi(2).expr_type(), ExprType::Pow);
assert_eq!(x.sin().expr_type(), ExprType::Function);
```

The full enum:

| `ExprType` | Covers |
|-----------|--------|
| `Number` | Integer and rational literals |
| `Symbol` | Named variables |
| `Constant` | π, e, i, ∞, -∞, NaN |
| `Add` | n-ary sums |
| `Mul` | n-ary products |
| `Pow` | Exponentiation |
| `Neg` | Unary negation |
| `Function` | sin, cos, exp, ln, abs, Σ, Π, etc. |
| `Apply` | User-defined function application |
| `Derivative` | Formal (unevaluated) derivative |
| `Integral` | Formal integral |
| `Set` | Intervals, finite sets, unions, intersections, complements |

### Term and Operation Counts

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// term_count: number of top-level summands
assert_eq!((&x + 1).term_count(), 2);       // x + 1 has 2 terms
assert_eq!(x.powi(2).term_count(), 1);      // x^2 is a single term
assert_eq!(expr!(x^2 + 3*x + 7).term_count(), 3); // 3 terms

// count_ops: number of non-atom nodes (complexity measure)
assert_eq!(x.count_ops(), 0);               // atom — no operations
assert_eq!((&x + 1).count_ops(), 1);        // one Add
assert_eq!(x.sin().powi(2).count_ops(), 2); // Sin + Pow
assert_eq!(expr!(x^2 + 2*x + 1).count_ops(), 5); // several ops
```

`count_ops()` is useful for comparing the "complexity" of equivalent forms — e.g., choosing between an expanded and factored representation.

## Polynomial Inspection

### Degree

`.degree(&var)` returns the polynomial degree with respect to a variable, or `None` if the expression isn't polynomial in that variable:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

assert_eq!(expr!(x^3 + x + 1).degree(&x), Some(3));
assert_eq!(expr!(5*x^2 + 3*x).degree(&x), Some(2));
assert_eq!(ctx.int(42).degree(&x), Some(0)); // constant = degree 0
assert_eq!(x.sin().degree(&x), None);             // not a polynomial
```

### Coefficients

`.coeffs(&var)` returns the coefficient list in ascending degree order:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// x^2 + 3*x + 5 → coefficients [5, 3, 1]
let f = expr!(x^2 + 3*x + 5);
let cs = f.coeffs(&x).unwrap();
let strs: Vec<String> = cs.iter().map(|c| format!("{c}")).collect();
assert_eq!(strs, vec!["5", "3", "1"]);
```

`.coeff(&var, n)` extracts a specific coefficient:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let f = &x.powi(2) * 3 + &x * 5 + 7;
assert_eq!(format!("{}", f.coeff(&x, 2).unwrap()), "3"); // x^2 coefficient
assert_eq!(format!("{}", f.coeff(&x, 1).unwrap()), "5"); // x^1 coefficient
assert_eq!(format!("{}", f.coeff(&x, 0).unwrap()), "7"); // constant term
assert_eq!(format!("{}", f.coeff(&x, 5).unwrap()), "0"); // beyond degree
```

### Numerator and Denominator

`.as_numer_denom()` splits a rational expression:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

let f = &x / &y;
let (n, d) = f.as_numer_denom();
println!("numerator:   {n}"); // x
println!("denominator: {d}"); // y

// Non-rational expressions have denominator 1
let g = expr!(x^2 + 1);
let (n, d) = g.as_numer_denom();
println!("numerator:   {n}"); // x^2 + 1
println!("denominator: {d}"); // 1
```

## Substitution

### Structural Substitution

`.subs(&old, &new)` replaces every occurrence of `old` with `new`:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

let f = expr!(x^2 + 2*x + 1);

// Replace x with a number
let at_3 = f.subs(&x, &ctx.int(3));
println!("{at_3}"); // 16

// Replace x with another expression
let g = f.subs(&x, &expr!(y + 1));
println!("{g}"); // (y + 1)^2 + 2*(y + 1) + 1
```

This is a **structural** operation — only exact node matches are replaced. If `x²` doesn't appear as a node in your expression (e.g., `1/x` stores `x^(-1)`), then `.subs(&x_squared, &something)` won't match.

### Multi-Variable Integer Substitution

`.subs_map_i64()` substitutes multiple variables with integer values at once:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

let f = expr!(x^2 + y^2);
let result = f.subs_map_i64(&[(&x, 3), (&y, 4)]);
println!("{result}"); // 25
```

This is convenient for evaluating multi-variable expressions at specific integer points without manually constructing `Ex` values for each substitution.

## Tree Walking and Rewriting

### `replace` — Custom Tree Rewriting

`.replace()` walks the expression bottom-up and applies a user-defined transformation at every node. Return `Some(replacement)` to rewrite a node, or `None` to keep it:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

let f = x.powi(2);

// Replace every occurrence of x with y
let g = f.replace(|node| {
    if node == &x {
        Some(y.clone())
    } else {
        None
    }
});
println!("{g}"); // y^2
```

The closure receives an `ExprView` — a lightweight, non-locking read-only handle to each sub-expression. You can compare it against `Ex` values using `==`.

### Practical Example: Variable Renaming

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y, theta, phi);

// Rename variables: x → θ, y → φ
let f = expr!(x^2 + 2*x*y + y^2);
let renamed = f.replace(|node| {
    if node == &x {
        Some(theta.clone())
    } else if node == &y {
        Some(phi.clone())
    } else {
        None
    }
});
println!("{renamed}"); // theta^2 + 2*theta*phi + phi^2
```

### Practical Example: Selective Simplification

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// Replace sin(x)^2 with 1 - cos(x)^2 wherever it appears
let sin2 = x.sin().powi(2);
let replacement = &ctx.int(1) - &x.cos().powi(2);

let f = &x.sin().powi(2) + &x * 2;
let g = f.replace(|node| {
    if node == &sin2 {
        Some(replacement.clone())
    } else {
        None
    }
});
println!("{g}"); // 1 - cos(x)^2 + 2*x
```

## Serialization

### JSON Round-Trip

`.to_json()` serializes an expression to a JSON string. `Context::from_json()` deserializes it back:

```rust
use symplex::prelude::*;

syms!(ctx; x);

let f = expr!(x^2 + sin(x));

// Serialize
let json = f.to_json();
println!("{json}"); // {"type":"Add","args":[...]}

// Deserialize
let ctx = Context::new();
let g = ctx.from_json(&json).unwrap();
println!("{g}"); // x^2 + sin(x)
```

For human-readable output, use `.to_json_pretty()`:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let json = x.powi(2).to_json_pretty();
println!("{json}");
// {
//   "type": "Pow",
//   "args": [
//     { "type": "Symbol", "name": "x" },
//     { "type": "Num", "value": "2" }
//   ]
// }
```

JSON serialization is useful for:

- **Caching** derived expressions to disk
- **IPC** between processes (e.g., a Python frontend and a Rust backend)
- **Debugging** — inspect the exact tree structure

### LaTeX Rendering

`.to_latex()` produces publication-quality LaTeX:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let f = expr!(x^2 + 2*x + 1);
println!("{}", f.to_latex()); // x^{2} + 2 x + 1

let g = expr!(sin(x)^2 + cos(x)^2);
println!("{}", g.to_latex()); // \sin^{2}\left(x\right) + \cos^{2}\left(x\right)

let h = &x / &(x.powi(2) + 1);
println!("{}", h.to_latex()); // \frac{x}{x^{2} + 1}
```

Wrappers for inline and display math modes:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let f = expr!(x^2 + 1);
println!("{}", f.to_latex_inline());   // $x^{2} + 1$
println!("{}", f.to_latex_display());  // $$x^{2} + 1$$
```

## Putting It All Together

Here's a workflow that inspects a polynomial, extracts its structure, rewrites it, and exports the result:

```rust
use symplex::prelude::*;
use symplex::expr::ExprType;

let ctx = Context::new();
fn main() {
    syms!(ctx; x);

    let f = expr!(x^4 - 5*x^2 + 4);
    println!("Expression: {f}");
    println!("Type: {:?}", f.expr_type());     // Add
    println!("Terms: {}", f.term_count());      // 3
    println!("Operations: {}", f.count_ops());
    println!("Degree: {:?}", f.degree(&x));     // Some(4)

    // Extract coefficients
    if let Some(coeffs) = f.coeffs(&x) {
        println!("Coefficients (ascending):");
        for (i, c) in coeffs.iter().enumerate() {
            println!("  x^{i}: {c}");
        }
    }

    // Factor
    let factored = f.factor(&x);
    println!("Factored: {factored}");

    // Check what symbols are involved
    let syms = f.free_symbols();
    println!("Variables: {:?}", syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>());

    // Serialize for later
    let json = f.to_json();
    println!("JSON length: {} bytes", json.len());

    // LaTeX for the paper
    println!("LaTeX: {}", f.to_latex());
}
```

## What's Next

The manipulation tools in this chapter give you low-level access to expression structure. There are a few areas planned for expansion:

- **Custom tree walkers** — expose the internal walk infrastructure so users can write arbitrary fold/map operations over expression trees without going through `.replace()`
- **Expression constructors from tree descriptions** — build expressions from a JSON-like tree spec at runtime, for dynamic expression construction
- **Pattern matching API** — a user-facing pattern language for matching and rewriting expressions, beyond the internal `rule!` macro used by the simplifier

For now, the combination of `.replace()`, `.free_symbols()`, `.contains()`, `.subs()`, and the polynomial inspection methods covers most structural manipulation needs.

---

*[← Chapter 12: Plotting](12-plotting.md) | [Back to Table of Contents](index.md)*