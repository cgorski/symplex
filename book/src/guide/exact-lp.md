# Exact Linear Programming

`symplex::linprog` solves linear programs over **ℚ**: a two-phase dense simplex running on `Ratio<BigInt>`, with Bland's rule after the first degenerate step so it cannot cycle. Optima, shadow prices and infeasibility certificates are exact — there are no tolerances and no "numerically infeasible" verdicts. That is what makes it useful as the engine behind certificate searches (Farkas lemmas, Positivstellensatz-style combinations, Carathéodory decompositions), where a floating-point solver can only say "probably".

The module works on plain `Vec<Q>` data (`Q = Ratio<BigInt>`), with two small constructors: `qi(n)` for an integer and `q(n, d)` for `n/d`. `LpSolution::x_ex(&ctx)` converts an optimum back into `Ex` rationals, and [`linprog_matrix`](#matrix-input-linprog_matrix) accepts `Matrix` data directly.

## The builder

`LpProblem::minimize(c)` / `maximize(c)` start a program in `c.len()` variables; `.le(row, rhs)`, `.ge(row, rhs)`, `.eq(row, rhs)` add constraint rows; `.bounds(j, lo, hi)` and `.free(j)` change a variable's bounds from the default `0 ≤ xⱼ < ∞`; `.solve()` returns `Result<LpSolution>`.

```rust
use symplex::prelude::*;
use symplex::linprog::{LpProblem, Q, qi};

/// `Ratio<BigInt>` displays as `3/2`, but `{:?}` on a `Vec<Q>` is verbose — format by hand.
fn show(v: &[Q]) -> String {
    format!("({})", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

fn main() {
    // max 5x + 4y  s.t.  6x + 4y ≤ 24,  x + 2y ≤ 6,  −x + y ≤ 1,  y ≤ 2,  x, y ≥ 0
    let sol = LpProblem::maximize(vec![qi(5), qi(4)])
        .le(vec![qi(6), qi(4)], qi(24))
        .le(vec![qi(1), qi(2)], qi(6))
        .le(vec![qi(-1), qi(1)], qi(1))
        .le(vec![qi(0), qi(1)], qi(2))
        .solve()
        .unwrap();
    println!("{:?}", sol.status);                     // Optimal
    println!("{}", show(&sol.x));                     // (3, 3/2)
    println!("{}", sol.objective.clone().unwrap());   // 21
    println!("{}", show(&sol.duals));                 // (3/4, 1/2, 0, 0)
    println!("{}", sol.is_optimal());                 // true
    let ctx = Context::new();
    println!("{:?}", sol.x_ex(&ctx));                 // [Ex(3), Ex(3/2)]
}
```

The remaining examples on this page reuse the `show` helper.

### Statuses

`solve()` only returns `Err` for *malformed* input — no variables, a row of the wrong length, bounds on a variable that does not exist. The three mathematical outcomes are values of `LpStatus`:

| `status` | Set fields | Meaning |
|----------|-----------|---------|
| `Optimal` | `x`, `objective`, `duals` | finite optimum |
| `Infeasible` | `farkas` (`Some` unless the bounds alone contradict) | no feasible point |
| `Unbounded` | — | the objective improves without limit |

```rust
use symplex::linprog::{LpProblem, qi};

fn main() {
    let sol = LpProblem::maximize(vec![qi(1), qi(1)]).le(vec![qi(1), qi(-1)], qi(1)).solve().unwrap();
    println!("{:?} {:?} {:?}", sol.status, sol.objective, sol.farkas);   // Unbounded None None

    println!("{}", LpProblem::maximize(vec![qi(1), qi(1)]).le(vec![qi(1)], qi(1)).solve().unwrap_err());
    // linprog: invalid argument: constraint 0 has 1 coefficients but there are 2 variables
    println!("{}", LpProblem::maximize(vec![qi(1)]).bounds(3, None, None).solve().unwrap_err());
    // linprog: invalid argument: bounds were set for variable 3 but there are only 1 variables

    // Contradictory bounds: infeasible, but there is no constraint certificate to give.
    let bad = LpProblem::minimize(vec![qi(1)]).bounds(0, Some(qi(3)), Some(qi(1))).solve().unwrap();
    println!("{:?} {:?}", bad.status, bad.farkas);                       // Infeasible None
}
```

### Bounds and free variables

```rust,ignore
// min x − y  s.t.  x + y ≤ 3,  −2 ≤ x,  0 ≤ y ≤ 1
let sol = LpProblem::minimize(vec![qi(1), qi(-1)])
    .le(vec![qi(1), qi(1)], qi(3))
    .bounds(0, Some(qi(-2)), None)
    .bounds(1, Some(qi(0)), Some(qi(1)))
    .solve()
    .unwrap();
println!("{:?} x* = {} objective {} duals {}", sol.status, show(&sol.x), sol.objective.unwrap(), show(&sol.duals));
// Optimal x* = (-2, 1) objective -3 duals (0)      — the constraint is slack, so its price is 0

// A free variable: min x  s.t.  2x ≥ −5
let free = LpProblem::minimize(vec![qi(1)]).free(0).ge(vec![qi(2)], qi(-5)).solve().unwrap();
println!("{:?} x* = {} objective {}", free.status, show(&free.x), free.objective.unwrap());
// Optimal x* = (-5/2) objective -5/2
```

## Duals and complementary slackness

When the status is `Optimal`, `duals` holds one **shadow price** `yᵢ` per constraint, in insertion order: the rate of change of the optimal objective value *of the problem as posed* with respect to `bᵢ`. The sign conventions that follow from that definition are, quoting the module documentation:

> for a minimisation `yᵢ ≤ 0` on `≤` rows and `yᵢ ≥ 0` on `≥` rows (the signs flip for a maximisation), `yᵢ` is free on `=` rows, and with the reduced costs `r = c − Aᵀy`:
>
> * complementary slackness: `yᵢ·(aᵢ·x* − bᵢ) = 0` for every row;
> * `rⱼ = 0` unless `x*ⱼ` sits at a finite bound (for a minimisation `rⱼ ≥ 0` at a lower bound and `rⱼ ≤ 0` at an upper bound; reversed for a maximisation);
> * strong duality: `cᵀx* = yᵀb + Σⱼ rⱼ x*ⱼ`, which reduces to `cᵀx* = yᵀb` under the default bounds `x ≥ 0`.

All of these are identities you can check with exact arithmetic:

```rust
use symplex::linprog::{LpProblem, Q, qi};

fn show(v: &[Q]) -> String {
    format!("({})", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

fn main() {
    let rows = [
        (vec![qi(6), qi(4)], qi(24)),
        (vec![qi(1), qi(2)], qi(6)),
        (vec![qi(-1), qi(1)], qi(1)),
        (vec![qi(0), qi(1)], qi(2)),
    ];
    let mut p = LpProblem::maximize(vec![qi(5), qi(4)]);
    for (row, rhs) in &rows {
        p = p.le(row.clone(), rhs.clone());
    }
    let sol = p.solve().unwrap();

    // Strong duality under x ≥ 0: cᵀx* = yᵀb.
    let ytb: Q = rows.iter().zip(&sol.duals).map(|((_, b), y)| b * y).sum();
    println!("cᵀx* = {}   yᵀb = {ytb}", sol.objective.clone().unwrap());   // cᵀx* = 21   yᵀb = 21

    // Complementary slackness: only binding rows have a non-zero price.
    for (i, (row, b)) in rows.iter().enumerate() {
        let slack: Q = row.iter().zip(&sol.x).map(|(a, x)| a * x).sum::<Q>() - b;
        println!("row {i}: a·x* − b = {slack:>4},  y = {:>3},  y·slack = {}",
            sol.duals[i], &slack * &sol.duals[i]);
    }
    // row 0: a·x* − b =    0,  y = 3/4,  y·slack = 0
    // row 1: a·x* − b =    0,  y = 1/2,  y·slack = 0
    // row 2: a·x* − b = -5/2,  y =   0,  y·slack = 0
    // row 3: a·x* − b = -1/2,  y =   0,  y·slack = 0

    // y₀ = 3/4 is ∂(optimum)/∂b₀: raising b₀ from 24 to 25 adds exactly 3/4.
    let sol2 = LpProblem::maximize(vec![qi(5), qi(4)])
        .le(vec![qi(6), qi(4)], qi(25))
        .le(vec![qi(1), qi(2)], qi(6))
        .le(vec![qi(-1), qi(1)], qi(1))
        .le(vec![qi(0), qi(1)], qi(2))
        .solve()
        .unwrap();
    println!("b₀ = 25 → objective {}", sol2.objective.unwrap());          // 87/4

    // Minimisation with ≥ rows: prices are ≥ 0.
    let m = LpProblem::minimize(vec![qi(1), qi(1)])
        .ge(vec![qi(1), qi(2)], qi(1))
        .ge(vec![qi(3), qi(1)], qi(1))
        .solve()
        .unwrap();
    println!("x* = {}, objective {}, duals {}", show(&m.x), m.objective.unwrap(), show(&m.duals));
    // x* = (1/5, 2/5), objective 3/5, duals (2/5, 1/5)
}
```

## Farkas certificates

When the status is `Infeasible`, `farkas` is a vector `y` with one entry per constraint that *proves* infeasibility. From the module documentation:

> `yᵢ ≥ 0` on `≤` rows, `yᵢ ≤ 0` on `≥` rows, free on `=` rows, such that, with `g = Aᵀy`,
>
> ```text
> Σⱼ  inf { gⱼ·xⱼ : lⱼ ≤ xⱼ ≤ uⱼ }   >   yᵀb
> ```
>
> where every infimum is finite (`gⱼ > 0 ⇒ lⱼ` finite, `gⱼ < 0 ⇒ uⱼ` finite, `gⱼ = 0` contributes `0`). Any feasible `x` would satisfy `(Aᵀy)·x ≤ yᵀb`, so the inequality proves that none exists. With no finite bounds this is the textbook form `Aᵀy = 0, yᵀb < 0`.

Under the default bounds `x ≥ 0` the infima are all `0`, so the certificate reads `Aᵀy ≥ 0` and `yᵀb < 0`:

```rust
use symplex::linprog::{LpProblem, LpStatus, qi};
use num_traits::Signed;

fn main() {
    // x + y ≤ 1  and  x + y ≥ 2  cannot both hold.
    let sol = LpProblem::minimize(vec![qi(1), qi(1)])
        .le(vec![qi(1), qi(1)], qi(1))
        .ge(vec![qi(1), qi(1)], qi(2))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::Infeasible);
    let y = sol.farkas.clone().unwrap();
    println!("y = ({}, {})", y[0], y[1]);                    // y = (1, -1):  y₀ ≥ 0 on the ≤ row, y₁ ≤ 0 on the ≥ row
    let g = &y[0] + &y[1];                                   // both columns of A are (1, 1)
    let ytb = &y[0] + &(&y[1] * qi(2));
    println!("Aᵀy = ({g}, {g}),  yᵀb = {ytb}");              // Aᵀy = (0, 0),  yᵀb = -1
    assert!(!g.is_negative() && ytb.is_negative());
    println!("{} {}", sol.x.len(), sol.duals.is_empty());    // 0 true   (no point, no prices)
}
```

In words: adding the first row to `−1` times the second gives `0 ≤ −1`. The [cookbook](../cookbook/polynomial-certificates.md) shows a Farkas vector being read as a linear functional that separates a polynomial from a cone.

## `feasible_nonneg`: is `b` a non-negative combination?

`feasible_nonneg(&a_eq, &b_eq)` answers "is there an `x ≥ 0` with `A·x = b`?" — `Ok(Some(x))` with a witness, or `Ok(None)`. It is the query behind most certificate searches and is exact even when the data have denominators like `1/3` and `1/7`.

```rust
use symplex::linprog::{Q, feasible_nonneg, q, qi};

fn show(v: &[Q]) -> String {
    format!("({})", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

fn main() {
    // μ ≥ 0 with  μ₁/3 + μ₂/7 + 2μ₃/5 = 1  and  μ₁ + μ₂ + μ₃ = 4
    let a = vec![vec![q(1, 3), q(1, 7), q(2, 5)], vec![qi(1), qi(1), qi(1)]];
    let b = vec![qi(1), qi(4)];
    match feasible_nonneg(&a, &b).unwrap() {
        Some(mu) => {
            println!("μ = {}", show(&mu));                               // μ = (0, 7/3, 5/3)
            for (row, rhs) in a.iter().zip(&b) {
                let lhs: Q = row.iter().zip(&mu).map(|(c, m)| c * m).sum();
                assert_eq!(&lhs, rhs);
            }
            println!("A·μ = b exactly");
        }
        None => println!("no non-negative combination"),
    }

    // x + y = −1 has no non-negative solution.
    println!("{:?}", feasible_nonneg(&[vec![qi(1), qi(1)]], &[qi(-1)]).unwrap());   // None

    // "Is (2, 3, 3) in the cone spanned by (1,0,1), (0,1,1), (1,1,0)?"
    // Columns are the generators, so build the rows by transposing.
    let cols = [[qi(1), qi(0), qi(1)], [qi(0), qi(1), qi(1)], [qi(1), qi(1), qi(0)]];
    let target = [qi(2), qi(3), qi(3)];
    let rows: Vec<Vec<Q>> = (0..3).map(|i| cols.iter().map(|c| c[i].clone()).collect()).collect();
    println!("{:?}", feasible_nonneg(&rows, &target).unwrap().map(|v| show(&v)));   // Some("(1, 2, 1)")
}
```

Note the orientation: `feasible_nonneg` takes **rows** of `A`, so when your generators are naturally columns (vectors, or polynomials laid out by `Poly::coefficient_matrix`), transpose first — or use `Matrix::to_rational_rows()` on a `Matrix` that is already in row form. When the search fails and you want the Farkas certificate too, pose the same equalities through `LpProblem::minimize(vec![Q::zero(); n]).eq(…)` and read `farkas`.

## SciPy-shaped `linprog`

`linprog(c, a_ub, b_ub, a_eq, b_eq, bounds)` minimises `cᵀx` subject to `A_ub·x ≤ b_ub`, `A_eq·x = b_eq` and per-variable bounds (empty `bounds` means `x ≥ 0`). Constraints are numbered `≤` rows first, then `=` rows — that is the order of `duals` and `farkas`.

```rust,ignore
// min −x − y   s.t.  x + 2y ≤ 4,  3x + y ≤ 6,  x, y ≥ 0
let sol = linprog(
    &[qi(-1), qi(-1)],
    &[vec![qi(1), qi(2)], vec![qi(3), qi(1)]],
    &[qi(4), qi(6)],
    &[],
    &[],
    &[],
)
.unwrap();
println!("{:?} x* = {} objective {}", sol.status, show(&sol.x), sol.objective.unwrap());
// Optimal x* = (8/5, 6/5) objective -14/5
```

## Matrix input: `linprog_matrix`

`linprog_matrix(objective, &c, a_ub, b_ub, a_eq, b_eq)` takes `Matrix` data. Entries are constant-folded with `eval()` first, so `1 + 2` or `1/2 + 1/3` are fine; a symbol or `π` is rejected with a clear error rather than approximated. Bounds are the default `x ≥ 0`.

```rust
use symplex::prelude::*;
use symplex::linprog::{Objective, linprog_matrix};

fn main() {
    let ctx = Context::new();
    let c = matrix![ctx, [3], [2]];
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(1)],
        vec![ctx.int(1), &ctx.int(1) + &ctx.int(2)],     // folded to 3
    ])
    .unwrap();
    let b = matrix![ctx, [4], [6]];
    let sol = linprog_matrix(Objective::Maximize, &c, Some(&a), Some(&b), None, None).unwrap();
    println!("x* = {:?}, objective {}", sol.x_ex(&ctx), sol.objective.unwrap());
    // x* = [Ex(4), Ex(0)], objective 12

    let x = ctx.symbol("x");
    let bad = Matrix::new(vec![vec![x, ctx.int(1)]]).unwrap();
    println!("{}", linprog_matrix(Objective::Minimize, &c, Some(&bad), Some(&matrix![ctx, [1]]), None, None).unwrap_err());
    // linprog_matrix: invalid argument: A_ub must contain only numeric literals; found `x`
}
```

## Performance

Every pivot is `O(m·n)` big-rational operations, and the rationals grow with the pivot sequence. In practice a **100-row × 200-variable** program with default bounds solves in seconds in a release build, and the certificate-sized problems in this book (a few dozen rows and columns) are instantaneous. Beyond a few hundred rows, or when the data are floating-point measurements to begin with, an exact solver is the wrong tool — the numerical routines in [Numerical Optimisation](./numerical-optimization.md) or an external LP library are. The pivot count is capped at `10 000 + 50·(m + n)`; exceeding it is reported as `ComputationFailed`, though Bland's rule makes that a theoretical rather than a practical concern.

See `cargo run --example exact_lp` for the complete program.
