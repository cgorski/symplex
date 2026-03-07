# Contributing to symplex

## Architecture

The source code is organized into 9 thematic directories under `src/`. Each directory
represents a layer in the dependency hierarchy — modules may depend on layers below
them but should not reach upward.

```
src/
├── base/         Node types, arena, tree traversal, canonicalization, assumptions
├── poly/         Dense/sparse polynomials, Gröbner bases, Sturm sequences
├── transforms/   Differentiation, integration, evaluation, solving, pattern matching
├── simplify/     Trig, power, log, combinatorial simplification, Fu's algorithm
├── calculus/     Series, limits, Laplace, ODE, Gosper summation, formal power series
├── output/       Display, LaTeX, Rust codegen, CSE, JSON serialization, parser
├── plotting/     Adaptive sampling, textplot, SVG, TikZ, data export, RK4
├── domains/      Matrices, control systems, dynamics, robotics, number theory
├── api/          Public types (Ex, Context), all public methods, operator overloads
└── lib.rs        Module tree, re-exports, prelude, convenience functions
```

**Dependency flow** (each layer may only call downward):

```
base → poly → transforms → simplify → calculus
                                         ↓
                              output / plotting / domains → api
```

`api/` sits at the top and depends on everything — it is the public facade.
`base/arena.rs` is a known exception: it provides convenience methods that
delegate upward into transforms, simplify, and calculus. This is contained
architectural debt, not a pattern to extend.

### When adding a new module

Pick the directory whose **dependency level** matches what your module needs:

| Your module needs… | Put it in |
|-|-|
| Only ExprNode, Arena, Walk | `base/` |
| Polynomial arithmetic | `poly/` |
| Diff, integrate, solve, eval, subs | `transforms/` |
| Simplification strategies or rewrite rules | `simplify/` |
| Composition of multiple transforms (limits, ODE, series) | `calculus/` |
| Rendering expressions to text, LaTeX, code, JSON | `output/` |
| Visualization, sampling, data export | `plotting/` |
| Application-specific math (control, robotics, matrices) | `domains/` |
| A new public method on `Ex` | `api/expr_funcs.rs` |

### Cross-layer references

Accept a cross-layer call when **all** of these hold:

1. One-way — A calls B, B never calls A
2. Narrow — 1–3 call sites, not deep coupling
3. Stable — the interface is unlikely to change
4. Documented — a `// NOTE:` comment explains the violation

Refactor when a cross-reference is circular, wide, or growing.

### Test loop

Use `cargo test --lib` (≈2 s) as the inner development loop. Run the full
`cargo test --all-targets` (≈45 s after a `src/` touch) only before commits.
See `IMPLEMENTATION_PLAN.md §1` for timing details.

## Getting Started

1. Clone the repository:

   ```sh
   git clone https://github.com/cgorski/symplex.git
   cd symplex
   ```

2. Ensure you have Rust **1.93.0** or later installed (this is the MSRV).

3. Build and run the test suite:

   ```sh
   cargo test --all-targets
   cargo test --doc
   ```

## Development Conventions

- **Formatting** — Run `cargo fmt --all` before committing. CI enforces this.
- **Linting** — Run `cargo clippy --all-targets -- -D warnings` and resolve all warnings.
- **Tests** — Every public API change must include tests. Use `proptest` for property-based testing where appropriate.
- **Commits** — Write clear, concise commit messages. Prefer small, focused commits over large ones.
- **Documentation** — All public items must have doc comments. Include examples where they aid understanding.

## How to Add a Feature

1. Open an issue describing the feature and its motivation.
2. Create a branch from `main` with a descriptive name (e.g. `add-trig-simplification`).
3. Implement the feature:
   - Add or update types in the appropriate module.
   - Write unit tests alongside the implementation.
   - Add integration tests under `tests/` if the feature spans multiple modules.
   - Add benchmarks under `benches/` if the feature is performance-sensitive.
4. Run the full CI checks locally:

   ```sh
   cargo fmt --all -- --check
   cargo clippy --all-targets -- -D warnings
   cargo test --all-targets
   cargo test --doc
   ```

5. Open a pull request against `main`. Describe what the PR does and link the related issue.

## License

By contributing, you agree that your contributions will be licensed under the terms of both the MIT license and the Apache License 2.0, at the choice of downstream consumers. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).