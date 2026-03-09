# SymPy vs Symplex: An Honest Comparison

Symplex and SymPy are both computer algebra systems, but they occupy very different niches. This document provides an honest, side-by-side comparison — including areas where SymPy is clearly stronger.

## At a Glance

| | **SymPy** | **symplex** |
|---|---|---|
| Language | Python | Rust |
| First release | 2007 | 2024 |
| Maturity | 17+ years, massive community | Young, rapidly evolving |
| Thread safety | GIL-bound | `Send + Sync` from day one |
| Type safety | Dynamic typing | Compile-time checked |
| Code generation | C, Fortran, Julia, Rust (via codegen module) | Native Rust with CSE, `no_std`, `f32` options |
| Build integration | None (runtime library) | `build.rs` pipeline (planned) |
| REPL / Notebook | Jupyter, IPython | Not a goal |
| Licensing | BSD | MIT OR Apache-2.0 |

## Side-by-Side: Common Tasks

### Creating Symbols

**SymPy:**
```python
from sympy import symbols, sqrt, sin, cos, Rational

x, y = symbols('x y')
```

**symplex:**
```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x, y);
```

### Building Expressions

**SymPy:**
```python
f = x**2 + 2*x + 1
g = sin(x)**2 + cos(x)**2
h = Rational(1, 2) * x
```

**symplex:**
```rust
let f = expr!(x^2 + 2*x + 1);
let g = expr!(sin(x)^2 + cos(x)^2);
let h = expr!(1/2 * x);
```

Notes:
- SymPy uses `**` for power (Python convention). symplex uses `^` inside `expr!` and `.powi()` outside.
- `expr!(1/2)` in symplex automatically creates an exact rational. In SymPy, `1/2` is Python integer division (0); you must write `Rational(1, 2)` or `S(1)/2`.

### Differentiation

**SymPy:**
```python
from sympy import diff
f = x**3 - 3*x**2 + 2*x
df = diff(f, x)       # 3*x**2 - 6*x + 2
d2f = diff(f, x, 2)   # 6*x - 6
```

**symplex:**
```rust
let f = expr!(x^3 - 3*x^2 + 2*x);
let df = f.diff(&x);       // 3*x^2 - 6*x + 2
let d2f = f.diff_n(&x, 2); // 6*x - 6
```

Both handle the chain rule, product rule, and all standard functions.

### Integration

**SymPy:**
```python
from sympy import integrate, oo
integrate(x**2, x)              # x**3/3
integrate(x**2, (x, 0, 1))     # 1/3
integrate(exp(-x), (x, 0, oo)) # 1
```

**symplex:**
```rust
let anti = expr!(x^2).integrate(&x);                              // 1/3*x^3
let definite = expr!(x^2).definite_integral(&x, &symplex::int(0), &symplex::int(1)); // 1/3
```

**SymPy advantage:** SymPy implements the full Risch algorithm for symbolic integration. It can handle a vastly larger class of integrands — rational functions of exponentials and logarithms, algebraic functions, and more. Symplex handles polynomials, basic trig, exponentials, and common patterns, but does not implement Risch.

### Simplification

**SymPy:**
```python
from sympy import simplify, trigsimp, expand, factor
simplify(sin(x)**2 + cos(x)**2)  # 1
trigsimp(sin(x)**2 + cos(x)**2)  # 1
expand((x + 1)**2)               # x**2 + 2*x + 1
factor(x**2 - 1)                 # (x - 1)*(x + 1)
```

**symplex:**
```rust
expr!(sin(x)^2 + cos(x)^2).simplify();       // 1
expr!(sin(x)^2 + cos(x)^2).simplify_trig();  // 1
(&x + 1).powi(2).expand();                    // x^2 + 2*x + 1
expr!(x^2 - 1).factor(&x);                    // (x - 1)*(x + 1)
```

Both provide layered simplification. The APIs map closely.

### Equation Solving

**SymPy:**
```python
from sympy import solve, solveset, S
solve(x**2 - 5*x + 6, x)   # [2, 3]
solve(x**4 - 1, x)          # [-1, 1, -I, I]
solveset(x**2 - 1, x, S.Reals)  # {-1, 1}
```

**symplex:**
```rust
expr!(x^2 - 5*x + 6).solve_or_empty(&x);  // [2, 3]
expr!(x^4 - 1).solve_or_empty(&x);          // [1, -1, I, -I]
expr!(x^2 - 1).solve_as_set(&x);            // {-1, 1}
```

Both solve polynomials through degree 4 algebraically. SymPy has broader transcendental equation support.

### Systems of Equations

**SymPy:**
```python
from sympy import solve
solve([x**2 + y**2 - 1, x + y - 1], [x, y])
# [(0, 1), (1, 0)]
```

**symplex:**
```rust
symplex::solve_system(
    &[expr!(x^2 + y^2 - 1), expr!(x + y - 1)],
    &[x.clone(), y.clone()],
).unwrap();
// [[0, 1], [1, 0]]
```

Both use Gröbner bases internally for polynomial system solving.

### Matrix Operations

**SymPy:**
```python
from sympy import Matrix
A = Matrix([[2, 1], [1, 3]])
A.det()          # 5
A.inv()          # Matrix([[3/5, -1/5], [-1/5, 2/5]])
A.eigenvals()    # {5/2 - sqrt(5)/2: 1, 5/2 + sqrt(5)/2: 1}
A.charpoly()     # PurePoly(lambda**2 - 5*lambda + 5, ...)
```

**symplex:**
```rust
let a = matrix![[2, 1], [1, 3]];
a.det();                   // 5
a.inv();                   // Some(Matrix{...})
a.eigenvals(&x);           // exact eigenvalues
a.char_poly(&x);           // x^2 - 5*x + 5
```

Both provide exact symbolic linear algebra for small matrices.

### Limits

**SymPy:**
```python
from sympy import limit, oo
limit(sin(x)/x, x, 0)    # 1
limit(1/x, x, oo)        # 0
```

**symplex:**
```rust
(&x.sin() / &x).limit(&x, &symplex::int(0));  // 1
(1 / &x).limit(&x, &symplex::infinity());      // 0
```

Both use L'Hôpital's rule and series expansion. SymPy additionally implements the Gruntz algorithm for computing limits at infinity, which handles a broader class of expressions.

### Series Expansion

**SymPy:**
```python
from sympy import series
series(sin(x), x, 0, 5)  # x - x**3/6 + x**5/120 + O(x**6)
```

**symplex:**
```rust
x.sin().maclaurin(&x, 5).expand().eval();
// x - 1/6*x^3 + 1/120*x^5
```

Both provide Taylor/Maclaurin expansions. SymPy's `O()` notation for remainder terms is more sophisticated.

### LaTeX Output

**SymPy:**
```python
from sympy import latex
latex(x**2 + 1)  # 'x^{2} + 1'
```

**symplex:**
```rust
expr!(x^2 + 1).to_latex();   // "x^{2} + 1"
```

Both produce clean LaTeX. SymPy has more formatting options (e.g., `mul_symbol`, `mat_delim`).

### ODE Solving

**SymPy:**
```python
from sympy import Function, dsolve, Eq
y = Function('y')
dsolve(y(x).diff(x) + 2*y(x), y(x))
# Eq(y(x), C1*exp(-2*x))
```

**symplex:**
```rust
let dy = y.formal_diff(&x);
let ode = &dy + &(&y * 2);
let (sol, _) = ode.solve_ode(&y, &x).unwrap();
// C1*exp(-2*x)
```

**SymPy advantage:** SymPy supports a much wider range of ODE classes — Bernoulli, Riccati, nth-order linear, systems of ODEs, power series solutions, and more. Symplex handles separable, linear constant/variable coefficient first-order, exact, and second-order constant-coefficient ODEs.

### Laplace Transforms

**SymPy:**
```python
from sympy import laplace_transform, inverse_laplace_transform
laplace_transform(sin(t), t, s)            # (1/(s**2 + 1), 0, True)
inverse_laplace_transform(1/(s - 2), s, t) # exp(2*t)*Heaviside(t)
```

**symplex:**
```rust
t.sin().laplace(&t, &s).unwrap();                      // 1/(s^2 + 1)
(1 / &(&s - 2)).inverse_laplace(&s, &t).unwrap();      // exp(2*t)
```

Both use table lookup and partial fractions. SymPy returns convergence conditions; symplex does not.

### Code Generation

**SymPy:**
```python
from sympy import ccode, fcode, rust_code
from sympy.utilities.codegen import codegen

ccode(sin(x)**2 + cos(x)**2)  # 'pow(sin(x), 2) + pow(cos(x), 2)'
# No CSE by default. CSE available via cse() + manual assembly.
```

**symplex:**
```rust
let f = expr!(sin(x)^2 + cos(x)^2);
let code = f.to_rust_fn("trig_id", &["x"]).unwrap();
// Complete function with CSE, constant folding, clean output
```

**Symplex advantage:** This is symplex's strongest area.

| Feature | SymPy codegen | symplex codegen |
|---------|---------------|-----------------|
| Output languages | C, C++, Fortran, Julia, Rust, JavaScript, etc. | Rust only |
| Complete functions | Requires `codegen()` wrapper | Built-in |
| CSE | Manual (`cse()` function, assemble yourself) | Automatic, cross-entry for matrices |
| Constant folding | Partial | Comprehensive (sin(0)→0, cos(0)→1, etc.) |
| Matrix codegen | Not built-in | Native with cross-entry CSE |
| `no_std` support | N/A | `CodegenOptions::no_std()` |
| f32 precision | Not supported | `CodegenOptions::embedded_f32()` |
| Build script | N/A | `build.rs` pipeline (planned) |
| Output cleanliness | Variable | Clean decimals, proper subtraction |

SymPy's code generation targets more languages but requires more manual assembly. Symplex produces complete, optimized, ready-to-compile Rust functions with a single method call.

## Where SymPy Is Clearly Stronger

### Integration (Risch Algorithm)
SymPy implements substantial portions of the Risch algorithm — the decision procedure for integration in elementary terms. This handles classes of integrals that symplex cannot, including rational functions of exponentials and logarithms, and algebraic functions.

### Geometry Module
SymPy has a full computational geometry module: points, lines, circles, polygons, ellipses, intersections, tangent lines, areas, and more. Symplex has no geometry module.

### Statistics and Probability
SymPy provides symbolic random variables, probability distributions (Normal, Exponential, Poisson, etc.), expected values, variances, and moment-generating functions. Symplex has none of this.

### Combinatorics
SymPy has extensive combinatorics support: permutation groups, partitions, polyhedra, graph theory primitives. Symplex provides factorials and binomial coefficients but not group theory.

### Physics Modules
SymPy includes modules for classical mechanics (Lagrangian/Hamiltonian with automated constraint handling), quantum mechanics (operators, states, Clebsch-Gordan coefficients), optics, and continuum mechanics. Symplex provides Lagrangian dynamics and robotics kinematics but not the broader physics toolkit.

### Assumptions System
SymPy's assumptions engine is more mature and supports richer inference chains (e.g., "if x is positive and integer, then x is a natural number"). Symplex has a working assumptions system with `Positive`, `Real`, `Integer`, etc., but the inference engine is simpler.

### PDE Solving
SymPy solves some classes of partial differential equations. Symplex handles only ODEs.

### Printing and Display
SymPy offers multiple printers: ASCII pretty-print, Unicode pretty-print, LaTeX, MathML, C code, Fortran code, dot graphs, and more. Symplex provides `Display`, `to_latex()`, `to_latex_inline()`, `to_latex_display()`, and code generation.

### Community and Ecosystem
SymPy has 1,000+ contributors, extensive documentation, a textbook, Stack Overflow coverage, and integration with NumPy, SciPy, matplotlib, and Jupyter. Symplex is a young project with a small team.

## Where Symplex Is Clearly Stronger

### Thread Safety
Symplex expressions are `Send + Sync`. You can derive Jacobians for multiple robot configurations in parallel without any locking. SymPy is bound by Python's GIL — true parallelism requires multiprocessing with serialization overhead.

### Type Safety
`Ex` is a concrete Rust type. You can't accidentally pass a matrix where a scalar is expected, or forget to import a module. The Rust compiler catches API misuse at compile time. SymPy uses duck typing — errors surface at runtime.

### Code Generation Quality
Symplex produces complete, optimized Rust functions with:
- Automatic cross-entry CSE for matrices
- Constant folding (`sin(0) → 0` at generation time)
- Clean decimal output (no `0.30000000000000004`)
- Proper subtraction (no `x + (-3.0)`)
- `no_std` and `f32` support for embedded targets
- `#[inline]` and `#[must_use]` annotations

This is not achievable with SymPy's `rust_code()` without significant post-processing.

### Build-Script Integration
Symplex's planned `build.rs` workflow lets you derive math at compile time and emit optimized code — the symbolic library is a build dependency only, not a runtime dependency. SymPy is always a runtime dependency (or requires a separate code-generation script).

### Performance
Symplex is written in Rust with no garbage collector. Expression construction, canonicalization, and simplification are faster than SymPy for common operations. The arena-based memory model avoids allocation churn.

### Embedding
Symplex is a Rust library — you can embed it in any Rust application, CLI tool, or WebAssembly module. SymPy requires a Python runtime.

## Feature Matrix

| Feature | SymPy | symplex |
|---------|:-----:|:-------:|
| Arithmetic (exact rationals) | ✅ | ✅ |
| Polynomial solving (≤ degree 4) | ✅ | ✅ |
| Polynomial solving (> degree 4) | ✅ (some) | ❌ |
| Gröbner bases | ✅ | ✅ |
| Differentiation | ✅ | ✅ |
| Integration (basic) | ✅ | ✅ |
| Integration (Risch) | ✅ | ❌ |
| Limits | ✅ | ✅ |
| Series expansion | ✅ | ✅ |
| ODE solving (basic) | ✅ | ✅ |
| ODE solving (advanced) | ✅ | ❌ |
| PDE solving | ✅ (some) | ❌ |
| Linear algebra (symbolic) | ✅ | ✅ |
| Eigenvalues | ✅ | ✅ |
| LU / Cholesky / RREF | ✅ | ✅ |
| Laplace transforms | ✅ | ✅ |
| Z-transforms | ❌ | ✅ |
| Fourier transforms | ✅ | ✅ |
| Number theory | ✅ | ✅ |
| Combinatorics | ✅ | Partial |
| Geometry | ✅ | ❌ |
| Statistics | ✅ | ❌ |
| Physics modules | ✅ | ❌ |
| Robotics (DH, FK, IK) | ❌ | ✅ |
| Control systems (state-space) | ✅ (control) | ✅ |
| Lagrangian dynamics | ✅ (mechanics) | ✅ |
| Code generation (multi-language) | ✅ | Rust only |
| Code generation (quality) | Basic | Advanced (CSE, folding, clean output) |
| Matrix codegen with CSE | ❌ | ✅ |
| `no_std` / embedded codegen | ❌ | ✅ |
| Build-script pipeline | ❌ | ✅ (planned) |
| Thread safety | ❌ (GIL) | ✅ |
| Type safety | ❌ (dynamic) | ✅ (compile-time) |
| LaTeX output | ✅ | ✅ |
| Assumptions | ✅ (rich) | ✅ (basic) |
| Pattern matching / rewrite rules | ✅ | ✅ |
| Arbitrary precision | ✅ (mpmath) | ✅ (astro-float) |
| WASM target | ❌ | ✅ (via symplex-wasm) |

## When to Use Which

### Use SymPy when:
- You need the Risch algorithm for symbolic integration
- You're working in a Jupyter notebook and want interactive exploration
- You need geometry, statistics, or advanced physics modules
- You need to generate code in C, Fortran, or Julia
- You're prototyping and don't need compiled performance
- You need a feature that symplex doesn't have yet

### Use symplex when:
- You need to generate optimized Rust code from symbolic math
- Thread safety matters (parallel robot kinematics, concurrent solvers)
- You're targeting embedded / `no_std` / `f32` platforms
- You want compile-time math derivation via `build.rs`
- You want type safety — the compiler should catch API misuse
- You're building a Rust application and don't want a Python dependency
- Performance matters — you need fast expression construction and simplification

### Use both when:
- You derive complex integrals in SymPy, simplify the result, then implement the numerical version in symplex for code generation
- You prototype in a Jupyter notebook with SymPy, then translate the working math to symplex for production Rust code

## Conclusion

SymPy is a mature, feature-rich CAS with 17 years of development and a massive community. It is the right choice for exploratory mathematics, teaching, and domains where breadth of coverage matters more than performance or type safety.

Symplex is a young, focused CAS built for a specific workflow: **symbolic derivation → code generation → compiled numerical code**. It trades breadth for depth in code generation quality, thread safety, type safety, and Rust ecosystem integration.

They are complementary tools. Use the one that fits your problem.