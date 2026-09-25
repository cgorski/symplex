# Sets and Logic

`SetEx` (set-valued expressions) and `BoolEx` (boolean expressions) are distinct types from `Ex`, checked at compile time. 0.2 gives both a real algebra.

## Building sets

| Constructor | Result |
|-------------|--------|
| `ctx.interval(&lo, &hi, IntervalKind::Closed)` (`Open`, `LeftOpen`, `RightOpen`) | `[lo, hi]`, `(lo, hi)`, `(lo, hi]`, `[lo, hi)` |
| `Interval::closed(lo, hi).to_set()` | the same from an `Interval<Ex>` |
| `ctx.finite_set(&[a, b, c])` | `{a, b, c}` (sorted, deduplicated) |
| `ctx.reals()`, `ctx.empty_set()`, `ctx.universal_set()` | ℝ, ∅, U |
| `x.closed_interval(&hi)`, `x.open_interval(&hi)` | intervals from an `Ex` endpoint |
| `expr.solve_gt(&x)` etc. | solution sets of inequalities |

Set construction is cheap and lazy — `a.union(&b)` is stored as a union — and `simplify()` computes the normal form (disjoint sorted intervals plus a finite set). `difference`, `symmetric_difference` and `absolute_complement` return normalised results directly.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let a = ctx.interval(&ctx.int(0), &ctx.int(5), IntervalKind::Closed);   // [0, 5]
    let b = ctx.interval(&ctx.int(3), &ctx.int(10), IntervalKind::LeftOpen);   // (3, 10]
    let s = ctx.finite_set(&[ctx.int(1), ctx.int(2), ctx.int(7)]);

    println!("{}", a.union(&b));                       // [0, 5] ∪ (3, 10]   (lazy)
    println!("{}", a.union(&b).simplify());            // [0, 10]
    println!("{}", a.intersection(&b).simplify());     // (3, 5]
    println!("{}", a.difference(&b));                  // [0, 3]
    println!("{}", a.symmetric_difference(&b));        // [0, 3] ∪ (5, 10]
    println!("{}", a.absolute_complement());           // (-oo, 0) ∪ (5, oo)
    println!("{}", s.union(&a).simplify());            // [0, 5] ∪ {7}
    println!("{}", s.difference(&ctx.finite_set(&[ctx.int(2)])));   // {1, 7}
}
```

## Queries

All queries are three-valued (`Option<bool>`) and `None` means "cannot decide", typically because a symbol is involved.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let a = ctx.interval(&ctx.int(0), &ctx.int(5), IntervalKind::Closed);
    let b = ctx.interval(&ctx.int(3), &ctx.int(10), IntervalKind::LeftOpen);

    println!("{:?} {:?} {:?}", a.contains(&ctx.int(3)), a.contains(&ctx.int(7)), a.contains(&x));
    // Some(true) Some(false) None
    println!("{:?}", ctx.int(3).is_in(&a));                                          // Some(true)
    println!("{:?}", a.is_subset(&ctx.interval(&ctx.int(0), &ctx.int(10), IntervalKind::Closed)));  // Some(true)
    println!("{:?}", a.is_disjoint(&ctx.interval(&ctx.int(5), &ctx.int(6), IntervalKind::Open)));   // Some(true)
    println!("{:?}", a.intersection(&ctx.interval(&ctx.int(6), &ctx.int(7), IntervalKind::Closed)).is_empty()); // Some(true)
    println!("{:?} {:?}", b.is_open(), a.is_closed());                              // Some(false) Some(true)

    let ab = a.union(&b).simplify();
    println!("{} {} {}", ab.inf().unwrap(), ab.sup().unwrap(), ab.measure().unwrap());   // 0 10 10
    println!("{} {} {}", a.boundary().unwrap(), b.closure().unwrap(), a.interior().unwrap());
    // {0, 5} [3, 10] (0, 5)
}
```

`as_intervals()` returns `Vec<Interval<Ex>>` (each with `lower`, `upper` and `kind`; an isolated point is `Interval::point(p)`) for a set that is a union of intervals, and `Interval<Ex>::to_set()` goes back; `as_finite_set()` returns the elements of a finite set; `to_condition(&x)` converts a set into the `BoolEx` "`x ∈ set`".

## From conditions to sets

`reduce_inequalities(&[BoolEx], &x)` intersects a list of conditions in one variable into a set; `BoolEx::solve_for(&x)` does the same for a single (possibly compound) condition.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let conds = [x.gt(&ctx.int(0)), x.le(&ctx.int(5)), (&x.powi(2) - 4).gt(&ctx.int(0))];
    println!("{}", reduce_inequalities(&conds, &x).unwrap());          // (2, 5]
    println!("{}", x.gt(&ctx.int(0)).and(&x.lt(&ctx.int(3))).solve_for(&x).unwrap());   // (0, 3)
    println!("{}", (&x.powi(2) - 1).ge(&ctx.int(0)).solve_for(&x).unwrap());  // (-oo, -1] ∪ [1, oo)
    let a = ctx.interval(&ctx.int(0), &ctx.int(5), IntervalKind::Closed);
    println!("{}", a.to_condition(&x).unwrap());                       // x >= 0 & 5 >= x
}
```

## Boolean logic

Boolean atoms are relations (`x.gt(&y)`, `x.eq_expr(&y)`, …) combined with `and`, `or`, `not`, `implies`. `BoolEx::simplify` applies boolean algebra (absorption, complementation, constant folding); `to_nnf`/`to_cnf`/`to_dnf` compute normal forms; `is_tautology`, `is_contradiction` and `satisfiable` use DPLL with unit propagation and respect declared assumptions; `atoms()` lists the atoms and `truth_table(&atoms)` enumerates them.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; p, q, r);
    let (pp, qq, rr) = (p.gt(&ctx.int(0)), q.gt(&ctx.int(0)), r.gt(&ctx.int(0)));

    println!("{}", pp.and(&qq).not().to_nnf());              // 0 >= p | 0 >= q
    println!("{}", pp.and(&qq).or(&rr).to_cnf());            // (p > 0 | r > 0) & (q > 0 | r > 0)
    println!("{}", pp.or(&qq).and(&rr).to_dnf());            // p > 0 & r > 0 | q > 0 & r > 0
    println!("{}", pp.and(&qq).or(&pp).simplify());          // p > 0
    println!("{:?}", pp.or(&pp.not()).is_tautology());       // Some(true)
    println!("{:?}", pp.and(&pp.not()).is_contradiction());  // Some(true)
    println!("{:?}", pp.and(&qq).implies(&pp).is_tautology()); // Some(true)
    for (inputs, out) in pp.and(&qq).truth_table(&[pp.clone(), qq.clone()]).unwrap() {
        println!("{inputs:?} → {out}");
    }
}
```

The boolean simplifier does not yet recognise every consensus pattern (`(p ∧ q) ∨ (p ∧ ¬q)` stays as written), but `is_tautology` on the equivalence proves it.

## Relations and assumptions

`BoolEx::eval` folds relations through the assumption system: for a symbol declared `Positive`, `pos > 0` evaluates to `True`; for an unassumed `x`, `x² + 1 > 0` stays open because `x` might be complex.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let pos = ctx.symbol_with("pos", &[Assumption::Positive]).unwrap();
    let t = ctx.symbol_with("t", &[Assumption::Real]).unwrap();
    println!("{}", pos.gt(&ctx.int(0)).eval());           // True
    println!("{}", pos.lt(&ctx.int(0)).eval());           // False
    println!("{}", t.powi(2).ge(&ctx.int(0)).eval());     // True
    println!("{}", (&x.powi(2) + 1).gt(&ctx.int(0)).eval());   // x^2 + 1 > 0
}
```

## Piecewise expressions

`Ex::piecewise(&[(value, condition), …])` builds a `Piecewise` node; `piecewise_simplify()` drops `false` branches, stops at the first `true` branch, removes unreachable repeats, merges adjacent branches with identical values, and collapses a single `true` branch to its value. Piecewise integrands are handled by `integrate_definite`, and `compile`/`to_rust_fn`/`to_c_fn` turn them into `if`/ternary chains.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let never = ctx.int(1).gt(&ctx.int(2));
    let pw = Ex::piecewise(&[(&x, &never), (&x.powi(2), &x.gt(&ctx.int(0))), (&x.powi(2), &x.le(&ctx.int(0)))]);
    println!("{}", pw.piecewise_simplify());              // x^2
}
```

## Assumptions as values

`Assumptions` is a pair of `Props` bitflags (known true / known false) with forward-chaining inference. `implies` compares two assumption sets; `Assumption::negate` gives the `Not*` variant; `Props::EXTENDED_REAL` is new in 0.2 (`is_real` and `is_finite` are both `None` for an `ExtendedReal` symbol, since it may be ±∞).

```rust
use symplex::prelude::*;

fn main() {
    let mut positive = Assumptions::default();
    positive.assert_true(Props::POSITIVE);
    println!("{positive}");   // commutative, complex, real, positive, nonnegative, nonzero, finite, …
    let nonneg = Assumptions { known_true: Props::NONNEGATIVE, ..Assumptions::default() };
    assert!(positive.implies(&nonneg));
    println!("{:?}", Assumption::Positive.negate());     // NotPositive
}
```

See `cargo run --example sets_and_logic` for the full tour.
