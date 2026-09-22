# What's New in 0.5

symplex 0.5.0 is a small minor release driven by the first hours of use of the 0.4 polyhedron certificates. Two mechanical breaking changes: `PolyhedronOutcome::Refuted` gained a `param_value` field, so patterns that destructure it need `..` (or the new field), and `LeanOpts` gained two fields (use `..Default::default()` or the `with_*` builders, as recommended since 0.4). Everything else is additive; the [CHANGELOG](https://github.com/cgorski/symplex/blob/main/CHANGELOG.md) has the list.

## A prover over a fixed hypothesis set

A decision tree certifies dozens of facets per leaf against the *same* hypotheses. `PolyhedronProver::new(&hyps, param, &opts)?` parses them and builds every LP stage's product basis once; `.prove(&goal)` and `.prove_empty()` then run only the goal-dependent part (λ columns, monomial rows, the LP). The one-shot functions are wrappers over it.

```rust
# use symplex::prelude::*;
# use symplex::certificates::{ParamBound, PolyhedronOpts, PolyhedronOutcome, PolyhedronProver};
# let ctx = Context::new();
# let (j, r, t) = (ctx.symbol("j"), ctx.symbol("r"), ctx.symbol("t"));
# let hyps = [&t - &r, &t + &j * &r - &j - 1];
# let param = ParamBound { var: j.clone(), lower: ctx.int(0) };
# let leaf_facets = [&t - 1, &t - &r];
let prover = PolyhedronProver::new(&hyps, Some(&param), &PolyhedronOpts::default())?;
for facet in &leaf_facets {
    match prover.prove(facet)? {
        PolyhedronOutcome::Proved(c) => println!("{c}"),
        PolyhedronOutcome::Refuted { point, value, .. } => println!("false: {value} at {point:?}"),
        PolyhedronOutcome::Unknown(u) => println!("no certificate: {u}"),
    }
}
let empty = prover.prove_empty()?;
# assert!(matches!(empty, PolyhedronOutcome::Unknown(_) | PolyhedronOutcome::Refuted { .. }));
# Ok::<(), SymplexError>(())
```

## Lean rendering hooks

- `LeanOpts::with_symbol_text("J", "(j : ℝ)")` renders a symbol as arbitrary Lean text wherever it occurs — goals, hypotheses, `λ`, the `hg` line — without touching hypothesis *names* like `e1J`. The parameter of a proof is usually a cast natural; now no token-aware post-processing is needed.
- `LeanOpts::with_single_fraction(true)` prints a rational function over one denominator (`(-(8 * j) - 2) / (7 * j + 4)`), undoing the split that `expand` leaves.
- `PolyhedronLeanSteps::to_block(indent)` re-flows to Mathlib's width with the indent included, and `wrap_lean` never starts a continuation line with `:=`, so `have hg : … := by` keeps its `:= by`.

## Parametric polytopes and volume in any dimension

`polytope::ParametricPolytope::new(&hyps, &vars, &j)` holds the family `{x : hₖ(j, x) ≥ 0}`; `.at(&j)` instantiates it exactly and `polytope_at` / `vertices_at` / `volume_at` cache per sample, so a tree builder can drop its own affine-evaluation and vertex code. `Polytope::volume` is no longer limited to three dimensions: an exact facet decomposition around the vertex centroid recurses on each facet's own H-representation (the norms cancel, so everything stays rational), verified on hypercubes and simplices through dimension 5.

## Exact PSD test

`QMatrix::ldl_psd()` returns the rational `L·D·Lᵀ` of a positive-semidefinite matrix (and `None` otherwise), the building block of the sums-of-squares certificates in 0.6.
