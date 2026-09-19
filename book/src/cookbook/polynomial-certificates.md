# Polynomial Inequality Certificates

## Problem

You want to *prove* — not numerically check — that a polynomial is non-negative on a box. Concretely, for

```text
0 ≤ r ≤ 1/2,   0 ≤ f ≤ 1
```

show that

```text
goal(r, f) = 1/4 − (r − f/2)²  ≥  0.
```

The inequality is true (`|r − f/2| ≤ 1/2` on the box) and tight at the corners `(1/2, 0)` and `(0, 1)`, so sampling cannot prove it and any floating-point slack would be suspicious.

## Background: Handelman certificates

The four **hypothesis polynomials**

```text
g₁ = r,   g₂ = 1/2 − r,   g₃ = f,   g₄ = 1 − f
```

are non-negative on the box by definition, and so is every product of them. If we can find non-negative rationals `λᵢ` with

```text
goal = Σ λᵢ · hᵢ,        hᵢ ∈ { 1, gⱼ, gⱼ·gₖ, … }
```

then `goal ≥ 0` follows immediately, and the identity can be checked by expanding both sides — a proof that fits on one line. Handelman's theorem says such a representation always exists for a polynomial that is *strictly* positive on a polytope, if you allow products of high enough degree.

Finding `λ` is a linear feasibility problem: match coefficients monomial by monomial, and require `λ ≥ 0`. That is exactly what `Poly::coefficient_matrix` builds and `linprog::feasible_nonneg` solves — over ℚ, so the certificate is exact.

## Solution

Build the family of products of degree ≤ 2 (fifteen polynomials), lay them out as a coefficient matrix, ask for a non-negative solution, and verify the identity two independent ways.

```rust
use num_rational::Ratio;
use num_traits::{Signed, Zero};
use symplex::linprog::{LpProblem, Q, feasible_nonneg};
use symplex::ntheory::rational_lcm_of_denominators;
use symplex::poly_ex::Poly;
use symplex::prelude::*;

fn show(v: &[Q]) -> String {
    format!("({})", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; r, f);
    let gens: [&Ex; 2] = [&r, &f];

    // ── Hypotheses: each is ≥ 0 on the box 0 ≤ r ≤ 1/2, 0 ≤ f ≤ 1 ─────────────
    let g: Vec<(String, Ex)> = vec![
        ("r".into(), r.clone()),
        ("(1/2 - r)".into(), ctx.rational(1, 2) - &r),
        ("f".into(), f.clone()),
        ("(1 - f)".into(), ctx.int(1) - &f),
    ];
    // Family: 1, gᵢ, gᵢ·gⱼ (i ≤ j) — every product of at most two hypotheses.
    let mut family: Vec<(String, Ex)> = vec![("1".into(), ctx.int(1))];
    family.extend(g.iter().cloned());
    for i in 0..g.len() {
        for j in i..g.len() {
            family.push((format!("{}·{}", g[i].0, g[j].0), &g[i].1 * &g[j].1));
        }
    }
    println!("{} hypothesis polynomials", family.len());
    let hyps: Vec<Poly> = family.iter().map(|(_, e)| e.as_poly(&gens).unwrap()).collect();
    let hyp_refs: Vec<&Poly> = hyps.iter().collect();

    // ── Goal ──────────────────────────────────────────────────────────────────
    let goal_ex = ctx.rational(1, 4) - (&r - &f / 2).powi(2);
    let goal = goal_ex.as_poly(&gens).unwrap();
    println!("goal = {}", goal.to_ex());

    // ── Coefficient matrix: rows = monomials, columns = hypotheses ───────────
    let mut all = hyp_refs.clone();
    all.push(&goal);
    let basis = Poly::monomial_basis(&all).unwrap();
    println!("basis: {basis:?}");
    let m = Poly::coefficient_matrix(&hyp_refs, &basis).unwrap();
    println!("M is {}×{}", m.nrows(), m.ncols());
    let b: Vec<Q> = basis
        .iter()
        .map(|mono| goal.coeff_monomial(mono).unwrap().as_rational().unwrap())
        .collect();
    println!("b = {}", show(&b));

    // ── Solve  M·λ = b,  λ ≥ 0  exactly ──────────────────────────────────────
    let a_rows = m.to_rational_rows().unwrap();
    let lambda = feasible_nonneg(&a_rows, &b).unwrap().expect("certificate exists");
    for ((name, _), l) in family.iter().zip(&lambda) {
        if !l.is_zero() {
            println!("λ = {l:>4}  ·  {name}");
        }
    }

    // ── Verify 1: Poly arithmetic ─────────────────────────────────────────────
    let mut acc = Poly::zero(&ctx, &gens).unwrap();
    for (h, l) in hyps.iter().zip(&lambda) {
        acc = acc.add(&h.scale(&ctx.from_ratio(l.clone())).unwrap()).unwrap();
    }
    println!("Σ λᵢhᵢ == goal (Poly::equals): {}", acc.equals(&goal));

    // ── Verify 2: ratsimp on the unexpanded certificate ──────────────────────
    let cert: Ex = ctx.sum(
        &family
            .iter()
            .zip(&lambda)
            .filter(|(_, l)| !l.is_zero())
            .map(|((_, e), l)| ctx.from_ratio(l.clone()) * e)
            .collect::<Vec<_>>(),
    );
    println!("certificate = {cert}");
    println!("(goal − certificate).ratsimp() = {}", (&goal_ex - &cert).ratsimp());

    // ── An integer certificate ───────────────────────────────────────────────
    let n = rational_lcm_of_denominators(&lambda);
    let ints: Vec<String> = family
        .iter()
        .zip(&lambda)
        .filter(|(_, l)| !l.is_zero())
        .map(|((name, _), l)| format!("{}·{name}", l * Ratio::from_integer(n.clone())))
        .collect();
    println!("{n}·goal = {}", ints.join(" + "));

    // ── Infeasible: a false inequality and its Farkas certificate ────────────
    let bad_ex = ctx.rational(1, 8) - (&r - &f / 2).powi(2);
    let bad = bad_ex.as_poly(&gens).unwrap();
    println!("\nbad goal = {}", bad.to_ex());
    println!("bad(1/2, 0) = {}", bad.eval(&[&ctx.rational(1, 2), &ctx.int(0)]).unwrap());
    let b_bad: Vec<Q> = basis
        .iter()
        .map(|mono| bad.coeff_monomial(mono).unwrap().as_rational().unwrap())
        .collect();
    // Same equalities through LpProblem so that we get the certificate back.
    let mut lp = LpProblem::minimize(vec![Q::zero(); hyps.len()]);
    for (row, rhs) in a_rows.iter().zip(&b_bad) {
        lp = lp.eq(row.clone(), rhs.clone());
    }
    let sol = lp.solve().unwrap();
    println!("status = {:?}", sol.status);
    let y = sol.farkas.clone().unwrap();
    for (mono, yi) in basis.iter().zip(&y) {
        println!("y[r^{} f^{}] = {yi}", mono[0], mono[1]);
    }
    // y defines a linear functional L(p) = Σ_m y_m · coeff_m(p) on polynomials.
    let functional = |p: &Poly| -> Q {
        basis
            .iter()
            .zip(&y)
            .map(|(mono, yi)| p.coeff_monomial(mono).unwrap().as_rational().unwrap() * yi)
            .sum()
    };
    let mins: Vec<Q> = hyps.iter().map(|h| functional(h)).collect();
    println!("min over hypotheses of L(hᵢ) = {}", mins.iter().min().unwrap());
    println!("L(bad goal) = {}", functional(&bad));
    assert!(mins.iter().all(|v| !v.is_negative()));

    // ── Infeasible but true: an interior zero ────────────────────────────────
    let touch_ex = (&r - ctx.rational(1, 4)).powi(2);
    let touch = touch_ex.as_poly(&gens).unwrap();
    let b_touch: Vec<Q> = basis
        .iter()
        .map(|mono| touch.coeff_monomial(mono).unwrap().as_rational().unwrap())
        .collect();
    println!("\n(r − 1/4)² : {:?}", feasible_nonneg(&a_rows, &b_touch).unwrap().map(|v| show(&v)));
    println!("(r − 1/4)² ≥ 0 on [0, 1/2]: {:?}",
        touch_ex.poly_is_nonnegative_on(&r, &ctx.int(0), &ctx.rational(1, 2)));
}
```

## Output

```text
15 hypothesis polynomials
goal = -1/4*f^2 + f*r - r^2 + 1/4
basis: [[2, 0], [1, 1], [1, 0], [0, 2], [0, 1], [0, 0]]
M is 6×15
b = (-1, 1, 0, -1/4, 0, 1/4)
λ =    1  ·  r·(1/2 - r)
λ =  1/2  ·  r·f
λ =  1/2  ·  (1/2 - r)·(1 - f)
λ =  1/4  ·  f·(1 - f)
Σ λᵢhᵢ == goal (Poly::equals): true
certificate = 1/2*f*r + 1/4*f*(-f + 1) + r*(-r + 1/2) + 1/2*(-r + 1/2)*(-f + 1)
(goal − certificate).ratsimp() = 0
4·goal = 4·r·(1/2 - r) + 2·r·f + 2·(1/2 - r)·(1 - f) + 1·f·(1 - f)

bad goal = -1/4*f^2 + f*r - r^2 + 1/8
bad(1/2, 0) = -1/8
status = Infeasible
y[r^2 f^0] = 1
y[r^1 f^1] = 0
y[r^1 f^0] = 2
y[r^0 f^2] = 1
y[r^0 f^1] = 1
y[r^0 f^0] = 5
min over hypotheses of L(hᵢ) = 0
L(bad goal) = -5/8

(r − 1/4)² : None
(r − 1/4)² ≥ 0 on [0, 1/2]: Some(true)
```

## Reading the result

**The certificate.** The solver found

```text
1/4 − (r − f/2)²  =  r(1/2 − r)  +  ½·r·f  +  ½·(1/2 − r)(1 − f)  +  ¼·f(1 − f)
```

Every term on the right is a product of factors that are non-negative on the box, multiplied by a non-negative rational, so the left side is non-negative on the box. That is the whole proof. Multiplying through by the lcm of the denominators (`rational_lcm_of_denominators`) gives the integer form `4·goal = 4·r(1/2 − r) + 2·rf + 2·(1/2 − r)(1 − f) + f(1 − f)`, which a reader can expand by hand. (For this goal the multipliers are in fact unique — the LP polytope is a single point — but in general `feasible_nonneg` returns *some* vertex of the feasible set, and any one of them is a valid certificate.)

**Two verifications.** `Poly::equals` compares the normalised coefficient lists of `Σ λᵢhᵢ` and `goal` — a structural check that uses no simplification. `(goal − certificate).ratsimp()` starts from the *unexpanded* certificate (products of linear factors) and reduces the difference to rational normal form; `0` is the only normal form of the zero function. Agreeing by two different routes is cheap insurance against a bug in either.

**The Farkas certificate.** Lowering the constant to `1/8` makes the inequality false (`bad(1/2, 0) = −1/8`), and the LP is `Infeasible`. Instead of a witness `λ`, `LpProblem` returns a vector `y` with one entry per monomial row. Because all our constraints are equalities with `λ ≥ 0`, the [module's certificate condition](../guide/exact-lp.md#farkas-certificates) reduces to `Mᵀy ≥ 0` and `yᵀb < 0`. Read `y` as a *linear functional* on polynomials, `L(p) = Σₘ yₘ·coeffₘ(p)`: the first condition says `L(hᵢ) ≥ 0` for every hypothesis product (the smallest value is `0`), and the second says `L(bad) = −5/8 < 0`. Any non-negative combination of the `hᵢ` would have `L ≥ 0`, so `bad` is not one — a proof of *non*-existence that is just as checkable as the certificate itself. (A point evaluation `p ↦ p(r₀, f₀)` is one such functional; the LP's `y` is generally not a point, which is why the argument works even when the inequality is true but the family is too small.)

**Infeasible does not mean false.** `(r − 1/4)²` is non-negative on `[0, 1/2]` — `poly_is_nonnegative_on` confirms it exactly via a Sturm sequence — yet no Handelman certificate exists at *any* degree, because the polynomial has a zero in the interior of the box and products of the `gⱼ` vanish only on its boundary. Handelman's theorem needs strict positivity. When a search comes back `None` for an inequality you believe, the options are: raise the degree of the products (for strictly positive goals this eventually works, at the price of `O(dⁿ)` columns), add squares such as `(r − 1/4)²` to the family (moving toward a Positivstellensatz / sum-of-squares certificate), or shrink the box away from the zero.

## Key symplex features used

- `Ex::as_poly` / `Poly` — the polynomial view with exact coefficients ([Polynomials as Data](../guide/polynomials.md))
- `Poly::monomial_basis`, `Poly::coefficient_matrix`, `Poly::coeff_monomial` — from polynomials to a linear system
- `Matrix::to_rational_rows` — `Matrix` → `Vec<Vec<Ratio<BigInt>>>` for the LP
- `linprog::feasible_nonneg`, `LpProblem::eq` + `LpSolution::farkas` — exact feasibility and certificates ([Exact Linear Programming](../guide/exact-lp.md))
- `Poly::scale`, `Poly::add`, `Poly::equals` and `Ex::ratsimp` — two independent verifications
- `ntheory::rational_lcm_of_denominators` — the integer form of the certificate
- `Ex::poly_is_nonnegative_on` — exact sign of a univariate polynomial on an interval

## The one-call version, and a proof Lean can check

Everything above is what `symplex::certificates::prove_nonnegative_on_box` does for you: it enumerates the products of the box inequalities up to a degree, solves the exact LP (preferring few, low-degree products), **re-verifies the identity with exact polynomial arithmetic**, and — when the claim is false — returns an exact counterexample instead. `Certificate::to_lean` then writes the result as a Mathlib theorem whose proof is `nlinarith` over precisely the products of the certificate, so Lean only has to check linear arithmetic.

```rust
use symplex::certificates::{prove_nonnegative_on_box, BoxOutcome};
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let (r, f) = (ctx.symbol("r"), ctx.symbol("f"));
    let goal = ctx.rational(1, 4) - (&r - &f / 2).powi(2);
    let bounds = [
        (r.clone(), ctx.int(0), ctx.rational(1, 2)),
        (f.clone(), ctx.int(0), ctx.int(1)),
    ];
    match prove_nonnegative_on_box(&goal, &bounds, 2).unwrap() {
        BoxOutcome::Proved(cert) => {
            println!("{cert}");
            // -1/4*f^2 + f*r - r^2 + 1/4 = 1/2*f*r + 1/4*f*(-f + 1) + r*(-r + 1/2) + 1/2*(-r + 1/2)*(-f + 1), 0 ≤ r ≤ 1/2, 0 ≤ f ≤ 1
            assert!(cert.verify());
            print!("{}", cert.to_lean("quarter_bound").unwrap());
        }
        BoxOutcome::Refuted { point, value, .. } => println!("false: goal = {value} at {point:?}"),
        BoxOutcome::Unknown(u) => println!("no certificate of degree {}", u.degree),
    }
}
```

The emitted theorem:

```lean
theorem quarter_bound (r f : ℝ) (h_r_lo : (0 : ℝ) ≤ r) (h_r_hi : r ≤ (1 / 2 : ℝ))
    (h_f_lo : (0 : ℝ) ≤ f) (h_f_hi : f ≤ (1 : ℝ)) :
    0 ≤ -(f ^ 2 / 4) + f * r - r ^ 2 + (1 / 4 : ℝ) := by
  nlinarith [mul_nonneg (sub_nonneg.mpr h_r_hi) (sub_nonneg.mpr h_f_hi),
    mul_nonneg (sub_nonneg.mpr h_f_lo) (sub_nonneg.mpr h_f_hi),
    mul_nonneg (sub_nonneg.mpr h_r_lo) (sub_nonneg.mpr h_r_hi),
    mul_nonneg (sub_nonneg.mpr h_r_lo) (sub_nonneg.mpr h_f_lo)]
```

This compiles against Mathlib (Lean 4.30.0) without errors or warnings — including with `linter.style.longLine` on, since every emitter wraps at 100 columns (`lean::wrap_lean`); `cargo run --example certificates_to_lean out.lean` writes a file with several such theorems that you can check with `lake env lean out.lean` inside any Mathlib project. The `(r − ¼)²`-style case from the previous section comes back as `BoxOutcome::Unknown` at every degree — exactly the interior-zero limitation of Handelman's theorem — and a false claim such as `xy − ½ ≥ 0` on the unit square is `Refuted { point: [(x, 0), (y, 0)], value: -1/2, .. }`.

## Half-lines, interior double zeros, and the whole real line

Handelman needs a compact box and a goal that stays strictly positive inside it. Two extensions cover the cases that come up in practice:

* **Half-lines** `x ≥ a` (or `x ≤ a`): with `k = x − a`, if every coefficient of `p(a + k)` is non-negative that is already a proof, and when it is not, a Pólya multiplier `(1 + k)^N` makes it so (guaranteed for a strictly positive goal). `prove_nonnegative_on_halfline` does both, refutes false claims with an exact point, and `prove_nonnegative_on_reals` glues two half-lines into a proof for all of ℝ.
* **Even-multiplicity zeros** inside the domain: `goal = g²·h` is split off by exact factoring and `h` gets the certificate; in Lean every hint becomes `mul_nonneg (sq_nonneg g) (…)`. This works for boxes and half-lines alike.

```rust
use symplex::certificates::{prove_nonnegative_on_halfline, HalfLineOutcome, Ray};
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let j = ctx.symbol("j");
    // (j − 5)²·(j² + 1) ≥ 0 for j ≥ 3: the double zero at 5 is inside the half-line.
    let goal = (&j - 5).powi(2) * (&j.powi(2) + 1);
    match prove_nonnegative_on_halfline(&goal, &j, &ctx.int(3), Ray::AtLeast, 12).unwrap() {
        HalfLineOutcome::Proved(cert) => {
            println!("{cert}");
            // j^4 - 10*j^3 + 26*j^2 - 10*j + 25 = (j - 5)^2*(6*j + (j - 3)^2 - 8), j ≥ 3
            println!("Pólya exponent {}, square {}", cert.polya_power(), cert.square().unwrap());
            // Pólya exponent 0, square Poly(j - 5, j)
            print!("{}", cert.to_lean("square_inside").unwrap());
        }
        HalfLineOutcome::Refuted { point, value, .. } => println!("false at {}: {value}", point[0].1),
        HalfLineOutcome::Unknown(u) => println!("no certificate up to N = {}", u.max_polya_power),
    }
}
```

```lean
theorem square_inside (j : ℝ) (h_j_lo : (3 : ℝ) ≤ j) :
    0 ≤ j ^ 4 - 10 * j ^ 3 + 26 * j ^ 2 - 10 * j + 25 := by
  have hk : 0 ≤ j - (3 : ℝ) := sub_nonneg.mpr h_j_lo
  nlinarith [sq_nonneg (j - 5), mul_nonneg (sq_nonneg (j - 5)) (hk),
    mul_nonneg (sq_nonneg (j - 5)) (pow_nonneg hk 2)]
```

When the shift alone is not enough — `j² − j + 1` on `j ≥ 0` has a negative coefficient — the certificate carries the multiplier: `(j + 1)·(j² − j + 1) = j³ + 1`, and the Lean proof shows `0 ≤ (1 + (j - 0)) ^ 1 * (j ^ 2 - j + 1)` with `nlinarith [pow_nonneg hk 3]` and divides by the positive factor with `nonneg_of_mul_nonneg_right`. A tight minimum costs a larger exponent (`4j² − 6j + 3` needs `N = 12`), which is Pólya's theorem being honest about how close to zero the goal gets.

## Parametric polyhedra: a multiplier on the goal

A decision procedure over a polytope whose facets move with a real parameter `j ≥ j₀` has to show, cell by cell, that a goal holds on the cell for *every* `j` — or that the cell is empty for every `j`. The certificate is again an exact identity with non-negative weights, but with one twist: the Farkas multipliers of `j`-dependent facets are rational functions of `j`, so a polynomial identity only exists after the goal is multiplied by a polynomial `λ(j)`:

```text
λ(j)·g = Σ μ · jᵃ (j − j₀)ᵇ · hₖ + Σ μ · jᵃ (j − j₀)ᵇ + μ₀   (+ Σ μ · hₖ hₗ),   λ(j) = 1 + Σ νₐ jᵃ,  μ, ν ≥ 0.
```

`prove_nonnegative_on_polyhedron` (0.4) runs the search staged — degree-1 multipliers with `λ = 1` first, then higher degrees, pairwise products last — and returns the first certificate, re-verified by polynomial arithmetic. `prove_polyhedron_empty` is the same call with the goal `−1`.

```rust
use symplex::prelude::*;
use symplex::certificates::{PolyhedronOpts, PolyhedronOutcome, prove_nonnegative_on_polyhedron};

fn main() {
    let ctx = Context::new();
    let (j, r, t) = (ctx.symbol("j"), ctx.symbol("r"), ctx.symbol("t"));
    // On { t ≥ r,  t + j·r ≥ j + 1 } the goal t − 1 ≥ 0 holds for every j ≥ 0,
    // but its multipliers are 1/(1 + j) and j/(1 + j): λ(j) = 1 + j is needed.
    let hyps = [&t - &r, &t + &j * &r - &j - 1];
    match prove_nonnegative_on_polyhedron(&(&t - 1), &hyps, Some((&j, &ctx.int(0))), &PolyhedronOpts::default()).unwrap() {
        PolyhedronOutcome::Proved(c) => {
            println!("{c}");
            // (j + 1)*(t - 1) = j*h0 + h1; h0 = -r + t, h1 = j*r - j + t - 1; j ≥ 0
            print!("{}", c.to_lean("needs_lambda").unwrap());
        }
        PolyhedronOutcome::Refuted { point, value, .. } => println!("false: {value} at {point:?}"),
        PolyhedronOutcome::Unknown(u) => println!("no certificate: {u}"),
    }
}
```

```lean
theorem needs_lambda (r t j : ℝ) (hj : (0 : ℝ) ≤ j) (h0 : 0 ≤ -r + t) (h1 : 0 ≤ j * r - j + t - 1) :
    0 ≤ t - 1 := by
  have hJ0 : (0 : ℝ) ≤ j := by linarith
  have h0J := mul_nonneg hJ0 h0
  have hg : (0 : ℝ) ≤ (j + 1) * (t - 1) := by
    linarith only [h0J, h1]
  have hg' := nonneg_of_mul_nonneg_right hg (by linarith only [hJ0])
  linarith only [hg']
```

The proof shape is the one a person writes: one `have … := mul_nonneg …` per product the certificate uses (`h0J` is `j·h₀`; `h0K` would be `(j − j₀)·h₀`, `h0xh1` a pairwise product, `pJJ` the pure power `j²`), then `linarith only […]` over exactly those facts; with `λ ≠ 1`, `0 ≤ λ·g` is shown first and divided out with `nonneg_of_mul_nonneg_right`. An emptiness certificate concludes `False`. When the theorem statement is not yours to write — the goal lives inside a larger lemma — `cert.lean_steps(&PolyhedronLeanNames { hyps: &["e0", "e1"], param_nonneg: "hJ0", shift_nonneg: "hK0" }, &opts)` gives the same `have` lines, hint names and closing block with your hypothesis names, and `to_block("  ")` indents and re-flows them to Mathlib's width. If the proof's parameter is a cast natural, `LeanOpts::default().with_symbol_text("j", "(j : ℝ)")` renders it that way everywhere (0.5). To certify many goals against the same hypotheses, build a `PolyhedronProver` once and call `.prove(&goal)` per facet — or `.prove_poly(&poly)` when the goal is already an exact polynomial (a `MultiPoly` through `Poly::from_multipoly`, in any generator order), which skips the expression round trip (0.6.1). A goal polynomial may carry generators that occur in none of its terms — a tool's ring has the parameter `J` on every row whether or not the row mentions it — and `prove_poly` ignores those even when the prover does not know them; only a generator that actually *occurs* and is not a variable of the hypotheses is an `InvalidArgument` (0.9.1). `cert.used_hyps()` names the hypotheses the identity really uses, so a generated lemma's signature can list exactly those instead of scanning the emitted text for names.

**Budgets (0.9.1).** A deep cell with a couple of dozen `j`-dependent hypotheses can make the degree-3 pairwise stage run for minutes, and nothing in a tree builder should be able to do that unbounded. `PolyhedronOpts::default().with_time_limit(Duration::from_secs(5))` gives every `prove`/`prove_poly`/`prove_empty` call five seconds from the moment it starts (so one `PolyhedronProver` built once gets a fresh allowance per goal; `with_deadline(Instant)` is the absolute form, and the earlier of the two applies), and `with_max_pivots(n)` caps the total simplex pivots across all of a call's LPs — the stage LPs and the refutation's sample LPs share one meter. When the budget runs out the answer is `Unknown` with `u.budget_exhausted == Some(BudgetHit::Deadline | BudgetHit::MaxPivots)` and a `Display` ending `budget exhausted: deadline`; a budget the search fits inside never changes a `Proved` or `Refuted` answer, and the certificate is byte-identical to the unbudgeted one. Underneath, `LpProblem::with_budget(Budget::within(d).with_max_pivots(n))` is the same mechanism on a single exact LP: the simplex checks the budget at every pivot (the count is shared by the `i64 → i128 → BigInt` attempts, so an overflowed attempt does not get its pivots back) and reports `LpStatus::BudgetExhausted` — an answer with empty `x`/`objective`, not an error. Without a budget every pivot sequence is exactly what it was.

### Assembling a whole proof: `lean::Block`

A generator that stitches many certificates into one lemma — a `refine frame_lemma … ?_ ?_` followed by one bullet per facet, inside `rcases` case splits — should not concatenate strings with hand-counted spaces: Lean's tactic blocks are column-sensitive (the tactics of a `by` block must sit strictly right of the tactic that opened it, and a `· ` bullet moves that column by two), and a mis-indented line silently changes which block a tactic belongs to. `symplex::lean::{Block, Tactic, Proof, Decl}` (0.8) is a small structured model of exactly this: `Tactic::have(name, Some(ty), Proof::by(block))`, `Tactic::bullet(block)`, `Tactic::raw("linarith only […]")`, and `Block::render(indent)` places every line from its tactic column and wraps past it. `steps.block()` gives a certificate's closing steps as such a block, so a leaf is

```rust,ignore
use symplex::lean::{Block, Tactic};

let mut leaf = Block::new(vec![Tactic::raw(format!("refine {call}\n  {}", vec!["?_"; facets.len()].join(" ")))]);
for steps in &facet_steps {
    leaf.push(Tactic::bullet(steps.block()));
}
lemma_body.push_str(&leaf.render("  "));
```

and the dispatcher's `rcases le_or_gt (0 : ℝ) (g) with h | h` with its two bullets is `Tactic::raw(…)` followed by two `Tactic::bullet(…)`, each bullet starting with `Tactic::have("e6", Some("(0 : ℝ) ≤ …"), Proof::term("h6"))`. `Decl::new(DeclKind::Lemma, name, statement, body).with_binders(…).with_doc(…)` renders the header in Mathlib's style (binders packed, ` :` at the end of the binder lines, the statement on its own line, ` := by`). `lean::lean_ident` quotes a name with `«…»` when it is not a plain identifier. The renderer's output for the generator's leaf shape is pinned to text that compiled against Mathlib.

Two refinements for generated files (0.9.1). The `refine … ?_ ?_ …` line above is better written as `Tactic::apply("refine leafG346_single_poly", args)` with the arguments — `(20 * (j : ℝ) + 10)`, `ρ`, `hx`, twenty-eight `?_` — as a `Vec<String>`: the renderer packs them greedily onto the head's line and onto continuation lines two columns past the tactic column, and because each argument is an atom it never breaks inside a parenthesised term the way a generic re-flow of a long string might; the result is a fixed point of `wrap_lean`, so a file that mixes both stays stable. And `Decl` has a `preamble: Vec<String>` (`.with_preamble(vec!["set_option maxHeartbeats 400000 in".into(), "-- generated".into()])`): lines emitted verbatim, unwrapped, before the doc comment — the place for `set_option … in`, `open … in` and a provenance comment. Adding the field means a `Decl { … }` struct literal from 0.8 now needs `preamble: vec![]`; `Decl::new` and the builders avoid the question.

Every shape the emitter produces (`λ = 1`, `λ` of degree 1 and 2, `j₀ > 0`, `j₀ = 0`, `j₀ < 0`, mixed `J`/`K` chains, pairwise products, emptiness with and without `λ`, pure parameter powers, no parameter at all) was compiled against Mathlib with the long-line linter on, and the emitted text is pinned to that compiled file in the test suite.

`PolyhedronOutcome::Refuted` carries an exact point of the set where the goal is negative (for the emptiness question: a point *in* the set), found by sampling `j` and minimising the goal over the cell with the exact LP when everything is affine in the free variables. The staged search costs a few milliseconds per facet for a cell with a dozen `j`-dependent hypotheses in a release build, which is what makes it usable inside a tree builder that asks thousands of times.

See `cargo run --example polyhedron_certificates` and, for the exact geometry of the cells themselves (vertices, volume, cuts), [`symplex::polytope`](../guide/exact-lp.md#polytopes-from-half-spaces).

## Sums of squares: interior zeros without a square factor

Everything above multiplies non-negative *hypotheses*. A polynomial that is non-negative on all of ℝⁿ with no hypotheses at all — `(x − 1)² + (y − 1)²`, or `x⁴ + y⁴ + z⁴ + 1 − 4xyz` — needs a different certificate: a **sum of squares** `g = Σ dₖ·pₖ²`. `prove_sos` (0.6) finds one exactly:

```rust
use symplex::prelude::*;
use symplex::certificates::{prove_sos, SosOpts, SosOutcome};

fn main() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let amgm = x.powi(4) + y.powi(4) + z.powi(4) - &x * &y * &z * 4 + 1;
    match prove_sos(&amgm, &[x.clone(), y.clone(), z.clone()], &SosOpts::default()).unwrap() {
        SosOutcome::Proved(c) => {
            println!("{c}");
            // x^4 + y^4 + z^4 - 4*x*y*z + 1 = (-1/3*x^2 - 1/3*y^2 - 1/3*z^2 + 1)^2 + 2/3*(-y*z + x)^2
            //   + 2/3*(-x*z + y)^2 + 2/3*(-x*y + z)^2 + 2/3*(-x^2 + y^2)^2 + 8/9*(-1/2*x^2 - 1/2*y^2 + z^2)^2
            print!("{}", c.to_lean("amgm3").unwrap());
        }
        SosOutcome::Refuted { point, value, .. } => println!("negative: {value} at {point:?}"),
        SosOutcome::Unknown(u) => println!("no decomposition found: {u}"),
    }
}
```

```lean
theorem amgm3 (x y z : ℝ) : 0 ≤ x ^ 4 + y ^ 4 + z ^ 4 - 4 * x * y * z + 1 := by
  have h : x ^ 4 + y ^ 4 + z ^ 4 - 4 * x * y * z + 1 = (-(x ^ 2 / 3) - y ^ 2 / 3 - z ^ 2 / 3 + 1) ^
    2 + (2 / 3 : ℝ) * (-(x * y) + z) ^ 2 + (2 / 3 : ℝ) * (-(x * z) + y) ^ 2 + (2 / 3 : ℝ) *
    (-(y * z) + x) ^ 2 + (8 / 9 : ℝ) * (-(x ^ 2 / 2) - y ^ 2 / 2 + z ^ 2) ^ 2 + (2 / 3 : ℝ) *
    (-x ^ 2 + y ^ 2) ^ 2 := by ring
  rw [h]
  positivity
```

The search is the Peyrl–Parrilo pipeline made exact: write `g = mᵀ Q m` over the monomials of half the degree, solve the semidefinite program for `Q` numerically (a small dense interior-point method is built in — no external solver), round the solution to rationals, project it back onto the coefficient constraints exactly, and test positive semidefiniteness with the rational `L·D·Lᵀ` of `QMatrix::ldl_psd` — whose factorisation is the decomposition. The Lean proof is two deterministic steps: `ring` checks the identity, `positivity` closes a sum of non-negative terms.

A goal with real zeros — every certificate the tree builder cares about touches zero somewhere — has only *singular* Gram matrices, which rounding cannot hit. `prove_sos` then does facial reduction: it reads the kernel off the numerical solution, makes it exact (directly when the kernel is rational, otherwise through its integer relations, found by LLL after Newton-refining the zeros of the goal to double precision), restricts the search to that face and solves again. Sums of two random squares whose common zeros are irrational algebraic points come back as exactly those two squares.

What it cannot do, it says so: `Refuted` carries an exact point where the goal is negative; Motzkin's polynomial `x⁴y² + x²y⁴ − 3x²y² + 1` (non-negative but not a sum of squares) is `Unknown`, never `Proved`. Certificates round-trip through JSON with re-verification like the others, and `lean_hints` gives the `sq_nonneg (pₖ)` terms for an `nlinarith` skeleton of your own.

The SDP has a budget too (0.9.1): `SosOpts::default().with_time_limit(Duration::from_secs(10))` (or `with_deadline(Instant)`) is checked between interior-point iterations and between facial-reduction rounds, and when it passes the answer is `Unknown` with a `reason` starting `budget exhausted: deadline` that names the round it was in. The cheap answers that come before the SDP — an exact counterexample, a constant, odd degree — are unaffected, and a roomy limit gives the same certificate as none.
