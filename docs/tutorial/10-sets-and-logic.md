# Chapter 10: Sets and Logic

Symplex has a type-level distinction between numeric expressions (`Ex`), boolean expressions (`BoolEx`), and set-valued expressions (`SetEx`). This chapter covers how to build and manipulate sets and boolean conditions — and how they connect to equation and inequality solving.

## The Three Expression Sorts

Symplex uses phantom types to separate three kinds of expression at compile time:

| Type | Alias | What it represents |
|------|-------|--------------------|
| `Expr<Numeric>` | `Ex` | Numbers, polynomials, transcendentals |
| `Expr<Boolean>` | `BoolEx` | True/false conditions |
| `Expr<SetValued>` | `SetEx` | Intervals, finite sets, unions |

You can't accidentally add a boolean to a number — the compiler rejects it. This catches entire classes of bugs that dynamically-typed CAS tools silently propagate.

## Building Sets

### Intervals

The most common set type is the interval. You build them from numeric endpoints:

```rust
use symplex::prelude::*;

let ctx = Context::new();

// Closed interval [0, 1]
let unit = ctx.int(0).closed_interval(&ctx.int(1));
println!("{unit}"); // [0, 1]

// Open interval (0, 1)
let open_unit = ctx.int(0).open_interval(&ctx.int(1));
println!("{open_unit}"); // (0, 1)
```

For mixed intervals (half-open), use the `Context::interval` method with explicit flags:

```rust
use symplex::prelude::*;

let ctx = Context::new();

// Half-open [0, 1)
let half_open = ctx.interval(&ctx.int(0), &ctx.int(1), false, true);
println!("{half_open}"); // [0, 1)

// Half-open (0, 1]
let other = ctx.interval(&ctx.int(0), &ctx.int(1), true, false);
println!("{other}"); // (0, 1]
```

The boolean flags are `left_open` and `right_open` — `false` means closed (bracket), `true` means open (parenthesis).

### Special Sets

```rust
use symplex::prelude::*;

let ctx = Context::new();

let empty = ctx.empty_set();
println!("{empty}"); // EmptySet

let universal = ctx.universal_set();
println!("{universal}"); // UniversalSet

// The real line as an interval
let reals = ctx.reals();
println!("{reals}"); // (-oo, oo)
```

### Finite Sets

```rust
use symplex::prelude::*;

let ctx = Context::new();

let s = ctx.finite_set(&[ctx.int(1), ctx.int(2), ctx.int(3)]);
println!("{s}"); // {1, 2, 3}
```

## Set Operations

`SetEx` supports the three fundamental set operations:

### Union

```rust
use symplex::prelude::*;

let ctx = Context::new();
let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);

let u = a.union(&b);
println!("{u}"); // [0, 1] ∪ [2, 3]
```

### Intersection

```rust
use symplex::prelude::*;

let ctx = Context::new();
let a = ctx.interval(&ctx.int(0), &ctx.int(2), false, false);
let empty = ctx.empty_set();

let result = a.intersection(&empty);
println!("{result}"); // EmptySet
```

### Complement (Relative)

The `.complement(other)` method computes `self \ other` — elements in `self` but not in `other`:

```rust
use symplex::prelude::*;

let ctx = Context::new();
let a = ctx.interval(&ctx.int(0), &ctx.int(5), false, false);
let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);

let diff = a.complement(&b);
println!("{diff}"); // [0, 5] \ [2, 3]
```

## Inequality Solving → Sets

This is where sets become practical. The inequality solvers return `SetEx` — the solution is a set, not a list of roots.

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// x² - 4 > 0 → (-∞, -2) ∪ (2, ∞)
let result = expr!(x^2 - 4).solve_gt(&x).unwrap();
println!("x² - 4 > 0: {result}");

// x² - 4 < 0 → (-2, 2)
let result = expr!(x^2 - 4).solve_lt(&x).unwrap();
println!("x² - 4 < 0: {result}");

// x² - 4 >= 0 → (-∞, -2] ∪ [2, ∞)
let result = expr!(x^2 - 4).solve_ge(&x).unwrap();
println!("x² - 4 ≥ 0: {result}");

// x² - 4 <= 0 → [-2, 2]
let result = expr!(x^2 - 4).solve_le(&x).unwrap();
println!("x² - 4 ≤ 0: {result}");
```

The solver uses the sign-chart method: find roots, test each region, and return the union of intervals where the condition holds.

### Equation Solving as a Set

`.solve_as_set()` returns solutions as a `FiniteSet` instead of a `Vec<Ex>`:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let solutions = expr!(x^2 - 5*x + 6).solve_as_set(&x);
println!("{solutions}"); // {2, 3}
```

This is useful when you want to compose the solution with set operations — e.g., intersecting the roots of one equation with the solution region of an inequality.

## Boolean Expressions

### Building Comparisons

Every numeric expression can be compared, producing a `BoolEx`:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let positive = x.gt(&ctx.int(0));   // x > 0
let at_least = x.ge(&ctx.int(0));   // x >= 0
let small = x.lt(&ctx.int(10));     // x < 10
let bounded = x.le(&ctx.int(10));   // x <= 10
let is_one = x.eq_expr(&ctx.int(1)); // x == 1
let not_zero = x.ne_expr(&ctx.int(0)); // x != 0
```

### Boolean Operations

`BoolEx` supports standard logic gates:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let pos = x.gt(&ctx.int(0));
let small = x.lt(&ctx.int(10));

// Conjunction: 0 < x AND x < 10
let in_range = pos.and(&small);

// Disjunction: x > 0 OR x < -5
let neg = x.lt(&ctx.int(-5));
let either = pos.or(&neg);

// Negation: NOT (x > 0)
let non_positive = pos.not();
```

There are also derived operations:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let a = x.gt(&ctx.int(0));
let b = x.lt(&ctx.int(10));

let xor = a.xor(&b);          // exclusive or
let imp = a.implies(&b);      // a → b  (¬a ∨ b)
let iff = a.equivalent(&b);   // a ↔ b
let nand = a.nand(&b);        // ¬(a ∧ b)
let nor = a.nor(&b);          // ¬(a ∨ b)
let ite = a.ite(&b, &a);      // if a then b else a
```

### Escape Hatches

If you need to pass a `BoolEx` or `SetEx` into a context that expects `Ex`, use `.into_ex()` or `.as_ex()`:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let cond = x.gt(&ctx.int(0));
let as_numeric: Ex = cond.as_ex(); // borrow-like, clones the Arc
```

## Piecewise Functions

Piecewise expressions connect boolean conditions to numeric values:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let neg_branch = &x * -1;  // -x when x < 0
let pos_branch = x.clone(); // x when x >= 0

let cond_neg = x.lt(&ctx.int(0));
let cond_pos = x.ge(&ctx.int(0));

let abs_x = Ex::piecewise(&[
    (&neg_branch, &cond_neg),
    (&pos_branch, &cond_pos),
]);
println!("{abs_x}");
```

`Ex::piecewise` takes `&[(&Ex, &BoolEx)]` — pairs of (value, condition). It returns the value of the first pair whose condition is true.

## What's Next

The set and logic infrastructure described above covers construction and basic operations. There are several planned features that will make this system much more powerful:

- **`interval.measure()`** — compute the length (Lebesgue measure) of an interval or union of intervals
- **`interval.contains(point)`** — membership testing: does a point belong to a set?
- **`set.is_subset(other)`** — subset relation between sets
- **`set.inf()` / `set.sup()`** — infimum and supremum (greatest lower bound, least upper bound)
- **`ImageSet`** — the image of a function over a set: `{f(x) : x ∈ S}`
- **`ConditionSet`** — a set defined by a predicate: `{x ∈ S : P(x)}`
- **Periodic solutions** — `solveset` for equations like `sin(x) = 0` returning `{nπ : n ∈ ℤ}` instead of just the principal values
- **Named number sets** — `Naturals`, `Integers`, `Rationals`, `Reals`, `Complexes` as first-class set objects with membership testing

For now, the inequality solvers and interval constructors cover the most common use cases. Head to [Chapter 11: Assumptions](11-assumptions.md) to learn how to attach mathematical properties to symbols and how that affects the rest of the system.

---

*[← Chapter 8: Code Generation](08-code-generation.md) | [Back to Table of Contents](index.md) | [Chapter 11: Assumptions →](11-assumptions.md)*