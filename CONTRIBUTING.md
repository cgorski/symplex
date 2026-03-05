# Contributing to symplex

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