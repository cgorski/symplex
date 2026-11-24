# Known Limitations

What is not in the library yet, and what is in it and known to be imperfect, at 0.60. Every example on
this page was run against that code. The `CHANGELOG.md` entry of each release has a *Known issues*
section with more detail; if you meet something that is not listed here or there, it is a bug worth
reporting.

## Not there

- **Hypergeometric and Meijer G functions.** There is no `pFq`, no Meijer G, no Tricomi `U`, no public
  Hurwitz zeta, no inverse incomplete gamma or beta; elliptic `E` and `Π` are the complete integrals
  only. The integrals, sums and transforms whose answers need them stay unevaluated: of the 72,254
  integrands of the Rubi suite, 41,237 are left as an `Integral` (30,768 are answered and verified).
- **Systems of nonlinear equations.** `polysys::solve_system_ex` solves polynomial systems with
  finitely many solutions. A system with infinitely many is `Err(InfiniteSolutions)` (no parametric
  description), and a non-polynomial system has no symbolic solver; `solve_numeric_system` is a
  Newton iteration from a starting point.
- **Geometry, tensor algebra, PDEs, group theory, quantum mechanics.** Not implemented.
- **Series.** No `O()` term; the two-sided `series` refuses fractional powers (`fps` gives Puiseux
  series). **Summation:** no Zeilberger algorithm. **ODEs:** no power-series solutions.
  **Recurrences:** linear with constant coefficients, and first order.
- **Input.** The parser reads symplex's own notation; there is no LaTeX or Mathematica input.
- **Logic.** `satisfiable` answers yes / no / unknown without an assignment, for at most 24 atoms, and
  does not reason about arithmetic: `x > y ∧ x > 0` is `None`.
- **Linear algebra.** Singular values but no full SVD; no sparse matrices; no matrix expressions.
- **Statistics.** 22 distribution families and finite tables (`sympy.stats` has about three times
  as many); joint distributions only as independent variables or the multivariate normal.
- **Code generation.** `expint`, `polylog`, `zeta`, `polygamma`, the Airy functions, elliptic `F` and
  `Π` and `dirichlet_eta` are not compiled by any back end (`Err(NotImplemented)`); the incomplete
  gamma functions and the Gegenbauer, Jacobi and associated polynomials only by `compile`.
- **Lean export.** Elementary expressions and the certificates; special functions are
  `Err(NotImplemented)`.

## Known to be imperfect

**Numerical evaluation**

- `eval_decimal(n)` prints digits inside a tracked error bound or returns an error. The bound is
  proved for the elementary functions and most special-function kernels; for some kernels — among
  them Bessel functions of large order, the orthogonal polynomials and `ζ` left of 0 — its
  derivative part is *estimated* (by re-evaluation at a nudged argument).
- Next to a zero of the function, the bound of a value computed by a three-term recurrence (an
  orthogonal polynomial of a large degree) is an estimate with a margin, not a proof, and refusals are
  not monotone in the digits: `besselj(1000, 200273630423430773792713/10^20)` prints at 16 and 30
  digits and is `PrecisionExhausted` at 50.
- The quadratures (`integrate_numeric`, `integrate_numeric_decimal`, `eval_decimal` of an
  unevaluated integral) accept a value when two rules or two levels agree. A spike much narrower
  than the node spacing, or a breakpoint at a huge offset that rounding moves, can be missed
  without an error.
- A part of a value that is not 0 but lies below the exponent range is refused in most cases and
  still dropped in some: `gamma(pi - I*exp(-3*10^9)) + sqrt(2)` prints a real number.
- Refused although a value exists: `elliptic_pi(1/3, 10^600 + I)` and `ln(-2 + I*exp(-10^10))`
  (`PrecisionExhausted`).

**Symbolic results**

- Two folds are applied too widely: `polylog(s, 1)` becomes `zeta(s)` for every `s` (true for
  `Re s > 1`; `polylog(0, 1)` gives `-1/2` although `Li₀` has a pole at 1), and `uppergamma(y, 0)`
  becomes `Gamma(y)` for a symbol without assumptions (true for `Re y > 0`).
- Powers of 0 and step functions are generic in places: `0*0^y` is `0`, `Heaviside(y)/Heaviside(y)`
  is `1`, `zoo*stirling2(0, 8)` is `zoo` (the second factor is 0).
- A set expression stays unevaluated where a member is real or not real but not proved so: `{p + I} \
  ℝ` for a positive `p` stays `{p + I} \ (-oo, oo)` (it is `{p + I}`), and `reals.contains(p + I)` is
  `None`.
- An inequality with a side that is not real through a constant without a literal `I`, next to the
  variable, still gets a set: `t + acos(2) > 2*t + acos(2)` for a real `t` is `(-oo, 0)` (it holds
  nowhere; with a literal `I` the result is `EmptySet`).
- `eigenvals` is an error where a factor of the characteristic polynomial has no exact roots in the
  library: an irreducible factor of degree 5 or more with symbolic coefficients, or of degree 3 or
  more with `π`, `e` or `ln 2` in it. Only a matrix of rational numbers always has its eigenvalues (as
  `RootOf`).
- Symbolic matrices: a pivot whose value cannot be decided is taken as nonzero (`rref`, `lu`, `pinv`,
  `rank_decomposition` …), so such results hold generically, not for every value of the symbols.
- `series` keeps a function at a pole as a coefficient (`x^2*expint(1, 0)` expands to itself) where
  `limit` refuses; `limit(gamma(floor(y) - 3 + x), x, 0)` is `Gamma(floor(y) - 3)`, a pole for `y < 4`.
- `isprime` beyond `3.3·10²⁴` is the Baillie–PSW test: no counterexample is known, but it is not a
  proof.
- The ASCII pretty printer writes `lgamma(x)` and `psi(x)`, which the parser does not read back.

**Refusals that have an answer**

- `try_` twins, and the Fourier and Mellin transforms, refuse a result that merely carries an
  unevaluated node over from the input (`Sum(sin(k^2), k, 1, m)*exp(-t)` has a `laplace`, no
  `try_laplace`).
- `try_integrate_definite(sin(x), 0, oo)` is `ComputationFailed`, not `Divergent`; the engines have
  no expansion of `lowergamma(s, x)` at 0 (`limit(lowergamma(-1/2, x), x, 0⁺)` is refused).
- Extreme parameters in statistics: the lower tail of a gamma or χ² law with a shape from about
  `4.5·10⁹` (`chi_squared(34000000000).cdf(33999990340.5)`), densities with both shapes huge
  (`F(3e128, 3e128)`, `Beta(1e128, 1e128)`), a windowed sum beyond 65,536 terms (a binomial with
  `n·p·(1 − p)` near `4·10⁶`), the exact Clopper–Pearson interval beyond 120 trials.

**Size and time**

- Some functions that build a list as long as a count they are given still abort or do not return
  for a huge count: `Matrix::identity`/`zeros`, `symbols_indexed`, `equispaced_grid`,
  `finite_diff_weights`, `pell_solutions`, and the sequences `bell`, `partition_count`, `stirling2`,
  `divisor_sigma`, `diff_n`, `Distribution::moment`. Within the caps some calls are slow
  (`coefficients(16385)` of `sin`: more than 20 s).
- The library bounds its work on the common paths, not on every path. For input you do not control,
  see `SECURITY.md`, "Running symplex on untrusted input".

## How it is tested, and how stable it is

- **Platform.** The test suite runs on Linux x86-64 only. `wasm32-unknown-unknown` is type-checked;
  macOS, Windows, 32-bit and big-endian targets are untested.
- **Oracles.** The symbolic engines are tested against SymPy, mpmath, scipy and statsmodels, not
  proved. Two comparisons run continuously: the Rubi suite (0 wrong answers) and the nightly
  statistics oracle. The nightly statistics fuzzer skips three slow cases: lattices of more than
  20,000 atoms are not transformed into tables, a lattice of more than 4,096 atoms whose tails must
  be summed mass by mass is not checked, and without a finite variance the first moment is checked
  only while `|mean| ≤ 10⁸`.
- **Less tested.** The certificate provers, the exact LP and the Lean export are covered by their
  own exact re-verification and by byte-identity tests, but have not been hunted for wrong answers
  the way calculus and statistics have; neither have the code generators for the functions added
  most recently, units, the WebAssembly bindings, the matrix decompositions and plotting. Lean text
  is compared with fixtures compiled at Lean 4.30.0; CI does not run Lean.
- **Speed** has not been measured against other libraries.
- **API stability.** Before 1.0 a minor release may break the API or change a result; the CHANGELOG
  lists such changes first. Only `LeanOpts`, `PolyhedronOpts`, `SosOpts` and `Budget` are
  `#[non_exhaustive]`: adding a field to any other option struct, or a variant to most public
  enums, is a breaking change. There is no promise yet about which printed forms (`Display`,
  LaTeX) stay stable, and types of the `num-*` crates (themselves before 1.0) appear in public
  signatures.
