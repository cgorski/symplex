# symplex-wasm

WebAssembly bindings for [symplex](https://crates.io/crates/symplex), the exact symbolic-mathematics library for Rust. Exposes the CAS to JavaScript through `wasm-bindgen`, for interactive notebooks and robotics demos in the browser.

Two flavours of API are provided:

- **Stateless functions** — `simplify("x^2 + 2*x + 1")`, `differentiate("sin(x)", "x")`, … Each call parses its string arguments in a fresh `Context` and returns a string (or a JSON array string for lists). Errors surface as rejected `JsValue`s with a human-readable message.
- **`Session`** — a long-lived `Context` in which `define("a", "2*x")` binds names that later `eval`/`simplify`/`latex`/`diff`/`solve` calls can use. Symbols keep their identity across calls.

All arithmetic is exact: a decimal literal such as `0.1` in an input string is parsed as the rational `1/10`.

## Building

Install [`wasm-pack`](https://rustwasm.github.io/wasm-pack/) and the target:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

Build for a bundler (webpack, Vite, …), for plain `<script type="module">`, or for Node:

```sh
cd symplex-wasm
wasm-pack build --release --target bundler      # → pkg/  (default)
wasm-pack build --release --target web          # ES module + init()
wasm-pack build --release --target nodejs
```

The release profile uses `opt-level = "s"` and LTO. The default feature `console_error_panic_hook` forwards Rust panics to `console.error`; drop it with `--no-default-features` for the smallest binary. `getrandom` is configured with the `js` feature for `wasm32-unknown-unknown` (needed by the arbitrary-precision evaluator's dependency chain).

Native tests run with `cargo test`; CI also checks `cargo check --target wasm32-unknown-unknown`.

## Using it from JavaScript

```js
import init, {
  simplify, differentiate, integrate, integrate_definite, solve, limit,
  series, to_latex, eval_decimal, to_rust_fn, to_c_fn, Session,
} from "./pkg/symplex_wasm.js";

await init();                                   // --target web only

simplify("sin(x)^2 + cos(x)^2");                // "1"
differentiate("x^3 - 2*x + 1", "x");            // "3*x^2 - 2"
integrate("x*exp(x)", "x");                     // "x*exp(x) - exp(x)"
integrate_definite("exp(-x^2)", "x", "-oo", "oo"); // "sqrt(pi)"
solve("x^2 - 5*x + 6", "x");                    // '["3","2"]'
limit("sin(x)/x", "x", "0");                    // "1"
series("exp(x)", "x", "0", 4);                  // "1/6*x^3 + 1/2*x^2 + x + 1"
to_latex("(x + 1)^2 / 2");                      // "\\frac{1}{2} \\left(x + 1\\right)^{2}"
eval_decimal("pi", 30);                         // "3.14159265358979323846264338327" (truncated, not rounded)
to_rust_fn("x^2 + sin(x)", "f", "x");           // "#[must_use]\npub fn f(x: f64) -> f64 {\n    x.powi(2) + x.sin()\n}"
to_c_fn("x^2 + sin(x)", "f", "x");              // "… #include <math.h>\ndouble f(double x) {\n    return (x * x) + sin(x);\n}"

const s = new Session();
s.define("r", "2");
s.define("area", "pi * r^2");
s.eval("area");                                 // "4*pi"
s.eval_f64("area");                             // "12.56637061435917"
s.latex("area / 2");                            // "2\\pi"
s.diff("x^2 * r", "x");                         // "4*x"   (definitions are substituted before differentiating)
s.solve("x^2 - r^2", "x");                      // '["2","-2"]'
s.names();                                      // '["area","r"]'
```

Every function throws (rejects) on a parse error, an unsolvable equation, free symbols in a numeric evaluation, and so on; the message is the `SymplexError` text.

## Stateless API

| Function | Returns |
|----------|---------|
| `simplify(expr)` / `simplify_latex(expr)` | simplified expression as text / LaTeX |
| `expand(expr)` | expanded polynomial |
| `factor(expr, var)` | factored over ℤ |
| `differentiate(expr, var)` / `differentiate_latex(expr, var)` | derivative |
| `integrate(expr, var)` | antiderivative, or `Integral(...)` if none is found |
| `integrate_definite(expr, var, lo, hi)` | definite integral; bounds are expression strings (`"0"`, `"pi/2"`, `"oo"`, `"-oo"`); improper integrals and divergence handled as in the library |
| `solve(expr, var)` | JSON array of solution strings for `expr = 0` (errors for identities/contradictions) |
| `limit(expr, var, point)` | limit; `point` may be `"oo"` / `"-oo"` |
| `series(expr, var, point, order)` | Taylor/Laurent series with `order` terms |
| `to_latex(expr)` / `parse_to_latex(expr)` | LaTeX |
| `pretty(expr)` | Unicode pretty-printed text (fractions, superscripts) |
| `eval_f64(expr)` | `f64` as a shortest round-trip decimal string (errors on free symbols or complex results) |
| `eval_decimal(expr, digits)` | arbitrary-precision decimal string |
| `eval_numeric(expr)` | `f64` string if evaluable, otherwise the exactly evaluated symbolic form |
| `to_rust_fn(expr, name, params)` | Rust source; `params` is a comma-separated list |
| `to_c_fn(expr, name, params)` | self-contained C99 source |

### Robotics helpers

| Function | Returns |
|----------|---------|
| `compute_fk(dh_json)` | 4×4 forward-kinematics matrix as a JSON 2-D array of LaTeX strings |
| `compute_jacobian(dh_json)` | 3×N position Jacobian as a JSON 2-D array of LaTeX strings |
| `generate_jacobian_code(dh_json)` | Rust function computing the Jacobian numerically |

`dh_json` is `{"joints": [{"theta": "theta1", "d": 0, "a": 0.3, "alpha": 0}, …]}`; numeric parameters are converted to exact rationals (`0.3` → `3/10`).

## `Session` API

| Method | Effect |
|--------|--------|
| `new Session()` | empty session with its own `Context` |
| `define(name, expr)` | bind `name` (letters, digits, underscores) to the parsed expression with existing definitions substituted; returns the stored value; redefining replaces; self-reference is an error |
| `get(name)` | the stored definition, or an error if undefined |
| `undefine(name)` → `bool` | remove a definition |
| `clear()` | remove all definitions (symbols and assumptions survive) |
| `names()` | JSON array of defined names (sorted) |
| `eval(expr)` | substitute definitions, evaluate exactly |
| `simplify(expr)` | substitute definitions, simplify |
| `latex(expr)` | substitute definitions, render LaTeX |
| `eval_f64(expr)` | substitute definitions, evaluate to an `f64` string |
| `diff(expr, var)` | substitute definitions, differentiate |
| `solve(expr, var)` | substitute definitions, solve `expr = 0` (JSON array) |

## Expression syntax

The parser accepts the same language as `symplex::parse`: `+ - * / ^`, function calls (`sin`, `cos`, `tan`, `exp`, `ln`, `log`, `sqrt`, `abs`, `gamma`, `erf`, `besselj`, `re`, `im`, `conjugate`, `arg`, `zeta`, `polygamma`, `si`, `ci`, `ei`, `li`, …), constants `pi`, `E`, `I`, `oo`, `-oo`, `zoo`, `EulerGamma`, `Catalan`, `GoldenRatio`, and exact decimal literals.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
