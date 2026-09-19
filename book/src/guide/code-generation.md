# Code Generation

Once you have a symbolic result — a Jacobian, a controller, a filter — you want to run it fast. symplex offers three routes:

| Route | Method | When |
|-------|--------|------|
| Compiled closure | `compile(&["x", …]) -> Result<CompiledFn>` | Evaluate now, in this process; no source code |
| Rust source | `to_rust_fn(name, &args) -> Result<String>` | `build.rs` pipelines, `no_std` firmware |
| C99 source | `to_c_fn(name, &args) -> Result<String>` | C/C++ projects, other toolchains |

All three run constant folding and common subexpression elimination first, and all three return `Err(SymplexError::FreeSymbol)` for a symbol not in the parameter list and `Err(NotImplemented)` for a node with no numerical meaning (an unevaluated `Integral`, a set, `I`).

## Compiled closures

`CompiledFn` is `Clone + Send + Sync`, callable directly (`f(&[1.0, 2.0])`), and has `arity()` and `try_call()` (arity-checked). `compile_many` compiles several expressions into one `CompiledFnVec` with a shared CSE pass — the right tool for gradients and Jacobians. Every numerically evaluable node is supported, including Γ, lnΓ, ψ, erf/erfc, Lambert W, Beta, factorials, Bessel J/Y/I/K, orthogonal polynomials, integer sequences, `min`/`max`/`floor`/`sign`/`heaviside`/`atan2`, and piecewise with boolean conditions.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);
    let f = &x.gamma() * &(&x.powi(2) + &y).erf() + &x.lambertw();

    let cf = f.compile(&["x", "y"]).unwrap();
    println!("{} {}", cf.arity(), cf(&[2.0, 1.0]));                 // 2 1.8526…
    assert!(cf.try_call(&[1.0]).is_err());                          // wrong arity
    assert!(matches!((&x + &z).compile(&["x"]), Err(SymplexError::FreeSymbol { .. })));

    let g = &x.sin().powi(2) + &(&x * 2 + &y).exp() * 3;
    let grad = Ex::compile_many(&[&g.diff(&x), &g.diff(&y)], &["x", "y"]).unwrap();
    println!("{:?}", grad.call_vec(&[0.5, 0.25]));                  // [21.97…, 10.59…]
}
```

## Rust source

`to_rust_fn` emits a `pub fn name(args: f64…) -> f64` with `mul_add` for `a*b + c`, `powi` for integer powers, and `let tN = …;` temporaries from CSE. Special functions call into an embedded `mod symplex_rt { … }` runtime that contains only the helpers the expression uses.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let g = &x.sin().powi(2) + &(&x * 2 + &y).exp() * 3;
    println!("{}", g.to_rust_fn("g", &["x", "y"]).unwrap());
    // #[must_use]
    // pub fn g(x: f64, y: f64) -> f64 {
    //     3_f64.mul_add(2_f64.mul_add(x, y).exp(), x.sin().powi(2))
    // }

    let f = &x.gamma() + &x.lambertw();
    let code = f.to_rust_fn("f", &["x"]).unwrap();
    assert!(code.starts_with("#[allow(dead_code, clippy::all)]\nmod symplex_rt {"));
    assert!(code.contains("symplex_rt::gamma(x)"));
}
```

### `CodegenOptions`

`to_rust_fn_with_options(name, &args, &opts)` takes a `CodegenOptions`:

| Field | Default | Effect |
|-------|---------|--------|
| `precision` | `F64` | `F32` emits `f32` and `f`-suffixed literals |
| `math_backend` | `Std` | `Libm` → `libm::sin(x)`; `CfgGated` → `math::sin(x)` with a cfg-gated `mod math` that picks `std` or `libm` (for `no_std`) |
| `inline` / `must_use` | `false` / `true` | attributes on the function |
| `cse` | `true` | common subexpression elimination |
| `use_mul_add` | `true` | fuse `a*b + c` into `mul_add`/`fma` (one rounding instead of two; disable for bit-exact unfused arithmetic) |
| `checked_domain` | `false` | `debug_assert!` domain checks (`ln(x)` needs `x > 0`, `sqrt` needs `x ≥ 0`, `lambertw` needs `x ≥ −1/e`) |
| `emit_runtime` | `true` | emit the `mod symplex_rt` preamble when needed |
| `unit_annotation`, `param_units`, `return_unit` | none | `uom` type annotations at the function boundary |

Presets: `CodegenOptions::no_std()` (cfg-gated + `#[inline]`), `CodegenOptions::embedded_f32()`.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let h = &x.ln() + &x.sqrt();
    let checked = CodegenOptions { checked_domain: true, ..Default::default() };
    println!("{}", h.to_rust_fn_with_options("h", &["x"], &checked).unwrap());
    // { debug_assert!(x >= 0.0_f64, "sqrt: argument {} outside domain", x); x.sqrt() } + …
    println!("{}", h.to_rust_fn_with_options("h_nostd", &["x"], &CodegenOptions::no_std()).unwrap());
    // #[cfg(feature = "std")] mod math { … }  #[cfg(not(feature = "std"))] mod math { … }
    // #[inline] #[must_use] pub fn h_nostd(x: f64) -> f64 { math::sqrt(x) + math::ln(x) }
}
```

### Many functions in one file

Each `to_rust_fn` call embeds its own runtime. When you concatenate many functions into one file, set `emit_runtime: false` and paste `CodegenOptions::runtime_module()` (the complete `mod symplex_rt` for that backend) once at the top. `symplex-build` does this for you.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let shared = CodegenOptions { emit_runtime: false, ..Default::default() };
    let mut file = CodegenOptions::default().runtime_module();
    file.push('\n');
    file.push_str(&x.gamma().to_rust_fn_with_options("g", &["x"], &shared).unwrap());
    file.push('\n');
    file.push_str(&x.lambertw().to_rust_fn_with_options("w", &["x"], &shared).unwrap());
    assert_eq!(file.matches("mod symplex_rt {").count(), 1);
    println!("{} lines", file.lines().count());
}
```

## C99 source

`to_c_fn` emits `#include <math.h>`, the `static inline symplex_*` helpers the expression needs (Lambert W, digamma, Bessel functions, orthogonal polynomials, integer sequences — everything `<math.h>` lacks), then the function with `const double tN = …;` temporaries. `tgamma`, `lgamma`, `erf`, `erfc`, `fma`, `expm1`, `log1p` are used directly; integer powers `|n| ≤ 4` become repeated multiplication; piecewise expressions become ternary chains ending in `NAN`.

`to_c_fn_with_options` honours `precision` (`float` + `sinf`/`expf`/`fmaf`), `cse`, `inline` (`static inline`), `use_mul_add` (`fma`), `checked_domain` (`assert(...)` with `#include <assert.h>`), and `emit_runtime` (pair with `CodegenOptions::c_runtime()` for a shared translation unit).

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let g = &x.sin().powi(2) + &(&x * 2 + &y).exp() * 3 + &x.powi(3) * &y;
    println!("{}", g.to_c_fn("g", &["x", "y"]).unwrap());
    // /* Generated by symplex. */
    // #include <math.h>
    //
    // double g(double x, double y) {
    //     return fma(y, (x * x * x), fma(3.0, exp(fma(2.0, x, y)), pow(sin(x), 2.0)));
    // }

    let opts = CodegenOptions { precision: Precision::F32, inline: true, checked_domain: true, ..Default::default() };
    println!("{}", (&x.ln() + &x.sqrt()).to_c_fn_with_options("h32", &["x"], &opts).unwrap());
    // static inline float h32(float x) {
    //     return (assert(x >= 0.0f), sqrtf(x)) + (assert(x > 0.0f), logf(x));
    // }

    let pw = Ex::piecewise(&[(&x.powi(2), &x.lt(&ctx.int(0))), (&x.sqrt(), &x.ge(&ctx.int(0)))]);
    println!("{}", pw.to_c_fn("pw", &["x"]).unwrap());
    // double pw(double x) { return (0.0 > x) ? (x * x) : ((x >= 0.0) ? sqrt(x) : NAN); }

    assert!(x.lambertw().to_c_fn("w0", &["x"]).unwrap().contains("static inline double symplex_lambert_w0(double x)"));
}
```

`cargo run --example c_codegen` generates C for a special-function expression, compiles it with `cc` if available, and checks that the C program agrees with the compiled Rust closure to ~1e-13.

## Matrices

`Matrix::to_rust_fn(name, &params)` returns a flat row-major `[f64; rows*cols]` with CSE shared across entries; `to_rust_fn_with_options` takes the same options. This is what `symplex-build` uses for Jacobians and forward-kinematics transforms.

## CSE

`cse()` returns `(bindings, rewritten)` for one expression, `Ex::cse_many(&[&a, &b])` shares temporaries across several. Bindings are ordered post-order (deterministic; a binding only refers to earlier bindings); trivially cheap nodes are extracted only when used three or more times, and boolean nodes never are.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let g = &x.sin().powi(2) + &(&x * 2 + &y).exp() * 3;
    let (bindings, exprs) = Ex::cse_many(&[&g.diff(&x), &g.diff(&y)]);
    for (name, value) in &bindings {
        println!("{name} = {value}");        // __cse_0 = exp(2*x + y)
    }
    println!("{:?}", exprs);
}
```

## Build-time generation and `no_std`

The [`symplex-build`](https://github.com/cgorski/symplex/tree/main/symplex-build) crate runs the CAS in `build.rs`: register scalar and matrix functions, choose `no_std`, `f32`, companion tests, and write everything to `$OUT_DIR`. Its cfg-gated `mod math` covers every function the 0.2 Rust backend can emit (`sin` … `atanh`, `powi`/`powf`, `atan2`, `min`/`max`, `expm1`, `log1p`, `log2`, `exp2`, `fma`, `sin_cos`).

## Other outputs

`to_latex()`, `pretty()`/`pretty_ascii()`, `to_json()`/`Context::from_json`, and plotting (`textplot`, `to_svg`, `to_tikz`, `plot_data`, `eval_table` — all `Result` in 0.2). See `cargo run --example latex_output`.

## More targets and interchange (0.9.1)

### Python, NumPy and Julia

`to_python()` prints a Python 3 expression over the `math` module (SymPy: `pycode`); `to_numpy()` the vectorised `numpy.` form (SymPy: `NumPyPrinter`); `to_julia()` base Julia (SymPy: `julia_code`). The `*_fn(name, &args)` twins wrap the expression in a function definition with CSE temporaries `t0`, `t1`, … and report a symbol that is not a parameter as `Err(FreeSymbol)`. Numbers stay exact (`2`, `(1/2)`), integer powers are `x**2`/`x^2`, `x^(1/2)` is `math.sqrt(x)`, relations and connectives print as `x > 0 and 1 > x` / `numpy.logical_and(numpy.greater(x, 0), …)` / `x > 0 && 1 > x`, and piecewise as `(v if c else …)`, `numpy.select([…], […], default=numpy.nan)`, `(c ? v : …)`. Anything the target cannot express — Bessel functions, `digamma`, `LambertW`, unevaluated integrals, sets, `I`; for NumPy also `gamma`/`erf`/`factorial` (SciPy territory); for Julia `gamma`/`erf` (SpecialFunctions.jl) — is `Err(NotImplemented)`, never a silently wrong formula. The expression forms print every symbol by name; the caller supplies `import math` / `import numpy`.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let g = &x.sin().powi(2) + &x.exp();
    assert_eq!(g.to_python().unwrap(), "math.sin(x)**2 + math.exp(x)");
    assert_eq!(g.to_numpy().unwrap(), "numpy.sin(x)**2 + numpy.exp(x)");
    assert_eq!(g.to_julia().unwrap(), "sin(x)^2 + exp(x)");

    let f = &x.sin().powi(2) + &x.sin() * &y;
    println!("{}", f.to_python_fn("f", &["x", "y"]).unwrap());
    // def f(x, y):
    //     t0 = math.sin(x)
    //     return t0**2 + t0*y

    let pw = Ex::piecewise(&[(&x.powi(2), &x.lt(&ctx.int(0))), (&x.sqrt(), &x.ge(&ctx.int(0)))]);
    assert_eq!(pw.to_python().unwrap(), "(x**2 if 0 > x else (math.sqrt(x) if x >= 0 else math.nan))");
    assert!(matches!(x.bessel_j(&ctx.int(0)).to_python(), Err(SymplexError::NotImplemented(_))));
}
```

### Presentation MathML

`to_mathml()` (on `Ex` and `BoolEx`; SymPy: `mathml(expr, printer='presentation')`) returns a `<math xmlns="http://www.w3.org/1998/Math/MathML">…</math>` element that browsers and MathJax render directly. Layout follows `to_latex`: `<mfrac>` for quotients and negative powers, `<msqrt>`/`<mroot>` for roots, `<msup>` for powers (`sin(x)^2` as `<msup><mi>sin</mi><mn>2</mn></msup>`), `<mi>sin</mi><mo>&#x2061;</mo>` (invisible apply) for function application, `<mo>&#x2062;</mo>` (invisible times) between factors, Greek symbol names as character references (`alpha` → `<mi>&#x3B1;</mi>`), `x_1` as `<msub>`, and explicit `<mo>(</mo>…<mo>)</mo>` wherever LaTeX would emit `\left(…\right)` — no `<mfenced>`. Every character reference is numeric, so the output is well-formed XML without a DTD. `Series`, `DSolve`, `RootOf` and `RootSum` have no standard presentation and return `Err(NotImplemented)`.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    assert_eq!(
        (&x.powi(2) + 1).to_mathml().unwrap(),
        "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">\
         <mrow><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mn>1</mn></mrow></math>"
    );
    println!("{}", (&x.sin() / 2 + &x.sqrt()).to_mathml().unwrap());
    println!("{}", x.gt(&ctx.int(0)).and(&x.lt(&ctx.int(1))).to_mathml().unwrap());
}
```

### `srepr` and DOT

`to_srepr()` (SymPy: `srepr`) is the unambiguous constructor form of the exact tree — `Add(Integer(1), Mul(Integer(2), Symbol('x')))`, `Pow(sin(Symbol('x')), Integer(2))`, `StrictGreaterThan(Symbol('x'), Integer(0))`, `Interval(a, b, false, true)` — derived from `to_tree()`, so it is total (every node kind prints) and shows the arena's canonical child order rather than display order. `to_dot()` (SymPy: `dotprint`) is a Graphviz `digraph` with one node per tree position (labelled with the node kind, plus the value for atoms), one edge per child, and ids `n0`, `n1`, … assigned in pre-order, so the output is deterministic; render it with `dot -Tsvg`.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    assert_eq!((2 * &x + 1).to_srepr(), "Add(Integer(1), Mul(Integer(2), Symbol('x')))");
    println!("{}", (2 * &x + 1).to_dot());
    // digraph {
    //     ordering=out;
    //     rankdir=TD;
    //     n0 [label="Add"];
    //     n1 [label="Integer(1)"];
    //     n2 [label="Mul"];
    //     n3 [label="Integer(2)"];
    //     n4 [label="Symbol('x')"];
    //     n0 -> n1;
    //     n0 -> n2;
    //     n2 -> n3;
    //     n2 -> n4;
    // }
}
```

### Parsing relations and implicit application

`Context::parse` already reads juxtaposition as multiplication (`2x`, `2 x`, `x y`, `2(x+1)`, `(x+1)(x-1)`, `2pi`) and is otherwise strict: no relations, and an identifier followed by `(` must be a known function. Two new entry points extend it without changing what `parse` accepts:

- `Context::parse_bool("x > 0 & x < 1") -> Result<BoolEx>` (SymPy: `sympify("x > 0")`) adds `<`, `<=`, `>`, `>=`, `==`, `!=`, the connectives `&`/`&&`/`and`, `|`/`||`/`or`, the prefix negation `~`/`!`/`not`, `True`/`False`, and SymPy's function forms `Eq(a, b)`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`, `And(…)`, `Or(…)`, `Not(a)`. Precedence is mathematical, loosest first: `or` < `and` < comparisons < `+ -` < `* /` < `^`; `not` applies to the following relation; comparisons do not chain (`0 < x < 1` is an error — write `0 < x & x < 1`). Note that Python's `sympify("x > 0 & x < 1")` fails because `&` binds tighter than `>` there. A numeric expression (`x + 1`) or a sort error (`(x > 0) + 1`, `x & y`) is `Err`.
- `Context::parse_implicit("2 sin x") -> Result<Ex>` (SymPy: `parse_expr(s, transformations=implicit_multiplication_application)`) additionally applies textbook function names without parentheses and treats an unknown `f(…)` as a product. The argument of `sin x` is the juxtaposed product that follows, up to the next `+`, `-`, comparison, closing parenthesis or function name: `2 sin x` is `2*sin(x)`, `sin 2x` is `sin(2*x)`, `sin x^2` is `sin(x^2)`, `sin x/2` is `sin(x/2)`, `sin x + 1` is `sin(x) + 1`, and `sin x cos y` is `sin(x)*cos(y)` (SymPy reads `sin(x*cos(y))`). Only the textbook names (trigonometric/hyperbolic and inverses, `exp`, `ln`/`log`, `sqrt`, `cbrt`, `abs`, `floor`, `ceil`, `sign`, `gamma`, `erf`, `erfc`, `factorial`) are applied implicitly; short names that double as variables (`re`, `im`, `arg`, `li`, `zeta`, …) need parentheses, and the one-letter display aliases `C`/`B`/`W` are ordinary symbols (write `binomial`, `beta`, `lambertw`). `x(x+1)` and `f(x)` are `x*(x+1)` and `f*x`.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let p = ctx.parse_bool("x > 0 & x < 1").unwrap();
    assert_eq!(p, x.gt(&ctx.int(0)).and(&x.lt(&ctx.int(1))));
    assert_eq!(p.to_string(), "x > 0 & 1 > x");
    assert_eq!(p.to_lean().unwrap(), "0 < x ∧ x < 1");

    assert_eq!(ctx.parse_implicit("2x + 3(y-1)").unwrap(), 2 * &x + 3 * (&y - 1));
    assert_eq!(ctx.parse_implicit("2 sin x cos y").unwrap(), 2 * &x.sin() * &y.cos());
    assert!(ctx.parse("x > 0").is_err());
}
```
