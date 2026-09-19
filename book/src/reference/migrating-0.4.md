# Migrating from 0.3 to 0.4

symplex 0.4 is a minor release with two source-level breaking changes, both mechanical. Everything else is additive; results of existing operations are unchanged. The full list is in the [CHANGELOG](https://github.com/cgorski/symplex/blob/main/CHANGELOG.md), and `cargo semver-checks check-release --baseline-version 0.3.5 --release-type minor` reports exactly these two.

## `roots_count_real` → `count_real_roots_in`

The 0.3 alias was kept without a deprecation warning so that `-D warnings` builds were not broken by a patch release; 0.4 removes it as announced.

```rust,ignore
// 0.3
let n = f.roots_count_real(&x, &lo, &hi);
// 0.4
let n = f.count_real_roots_in(&x, &lo, &hi);            // Ex
let n = Poly::new(&f, &[&x]).unwrap().count_real_roots_in(&lo, &hi);   // Poly
```

## `LeanOpts` has a new field

`LeanOpts` gained `prefer_subtraction`, and more fields may follow in minor releases. A struct literal that names every field no longer compiles; use functional update or the builders.

```rust,ignore
// 0.3
let opts = LeanOpts { real_type: "ℚ".into(), ascribe_integers: false };
// 0.4 — either
let opts = LeanOpts { real_type: "ℚ".into(), ..Default::default() };
// or
let opts = LeanOpts::default().with_real_type("ℚ").with_prefer_subtraction(true);
```

`LeanOpts::default()` renders exactly as 0.3 did; `prefer_subtraction` is opt-in.
