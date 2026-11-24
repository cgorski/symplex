<!-- Thank you.  CONTRIBUTING.md has the rules; the short version: -->

**What was wrong / what is new** (input, old result, new result)

**The oracle** behind every new expected value (the mpmath/SymPy call and its
output, or the exact identity) — each new test names it in its doc comment.

**Checks run** (`cargo fmt --all`, `cargo clippy --all-targets -- -D warnings`,
`cargo test`, `cargo doc --no-deps`; say what you could not run)

- [ ] No `unwrap`/`expect`/panic in library code; no recursion over expression trees.
- [ ] An answer that is not proved is refused, not returned.
- [ ] A public API change is listed under `### Breaking` or `### Added` in `CHANGELOG.md`.
