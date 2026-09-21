# Migrating from SymPy

This page is a reference for developers who know SymPy and want to find the equivalent operations in symplex. It is organized by task, with SymPy on the left and symplex on the right.

## Key Differences

Before the translation table, a few structural differences to be aware of:

| Concept | SymPy | symplex |
|---------|-------|---------|
| State | Global implicit state; symbols are free-standing objects | Every expression belongs to a `Context`; no global state |
| Types | Everything is an `Expr` at runtime | `Ex` (numeric), `BoolEx` (boolean), `SetEx` (set-valued) — distinct at compile time |
| Arithmetic | Python operators on SymPy objects | Rust operators on `&Ex` references (or use `expr!` macro) |
| Evaluation | `simplify()` is the catch-all | `eval()` for exact reduction, `simplify()` for multi-strategy simplification to a fixpoint, `simplify_traced()` to see what fired |
| Failure | Returns unevaluated or raises exception | Returns unevaluated `Ex` (Pattern 1) or `Result` (Patterns 2–6); never raises |
| Realness | Symbols are complex unless `real=True`; `re(z)` stays symbolic | Same in 0.2: `z.re()` is `re(z)` unless `z` is declared `Real` |
| Floats | `Float` type exists alongside exact | No float type in expressions; floats only via `eval_f64()` |
| Printing | `pprint()`, `latex()`, `str()` | `println!("{expr}")`, `expr.to_latex()` |

## Setup

| SymPy | symplex |
|-------|---------|
| `from sympy import *` | `use symplex::prelude::*;` |
| `x, y, z = symbols('x y z')` | `symplex::syms!(ctx; x, y, z);` |
| `x = Symbol('x')` | `let x = ctx.symbol("x");` |
| `x = Symbol('x', positive=True)` | `let x = sym!(ctx; x, Positive);` |
| `x = Symbol('x', integer=True)` | `let x = sym!(ctx; x, Integer);` |

## Building Expressions

| SymPy | symplex |
|-------|---------|
| `x**2 + 2*x + 1` | `&x.powi(2) + &x * 2 + 1` |
| `x**2 + 2*x + 1` | `expr!(ctx, x^2 + 2*x + 1)` |
| `Rational(1, 3)` | `ctx.rational(1, 3)` |
| `r.p`, `r.q` (numerator / denominator of a `Rational`) | `e.as_ratio_i128()` → `Option<(i128, i128)>`, `e.as_ratio_parts()` → `Option<(BigInt, BigInt)>`; the full `Ratio<BigInt>` via `e.as_rational()` (types re-exported as `symplex::num_rational` / `symplex::num_bigint`) |
| `Integer(42)` | `ctx.int(42)` |
| `pi` | `ctx.pi()` |
| `E` | `ctx.e()` |
| `I` | `ctx.i_unit()` |
| `oo` | `ctx.infinity()` |
| `zoo` | `ctx.complex_infinity()` |
| `EulerGamma`, `Catalan`, `GoldenRatio` | `ctx.euler_gamma()`, `ctx.catalan()`, `ctx.golden_ratio()` |
| `Float(0.3)` | no float type — `ctx.from_f64_approx(0.3, 1_000_000)` → `3/10`, `ctx.from_f64(0.3)` → exact dyadic |
| `Rational("22/7")`, `S("0.125")` | `ctx.rational_str("22/7")`, `ctx.decimal_str("0.125")` |
| `sin(x)` | `x.sin()` |
| `cos(x)` | `x.cos()` |
| `exp(x)` | `x.exp()` |
| `log(x)` | `x.ln()` |
| `sqrt(x)` | `x.sqrt()` |
| `Abs(x)` | `x.abs()` |
| `re(z)`, `im(z)`, `conjugate(z)`, `arg(z)` | `z.re()`, `z.im()`, `z.conjugate()`, `z.arg()` |
| `z.as_real_imag()` | `z.as_real_imag()` |
| `expand_complex(z)` | `z.expand_complex()` |
| `x**y` | `x.pow(&y)` |
| `x**5` | `x.powi(5)` |
| `factorial(n)` | `n.factorial()` |
| `binomial(n, k)` | `n.binomial(&k)` |
| `gamma(x)` | `x.gamma()` |
| `erf(x)` | `x.erf()` |
| `Piecewise((a, cond1), (b, cond2))` | `Ex::piecewise(&[(&a, &cond1), (&b, &cond2)])` |
| `Matrix([[1, 2], [3, 4]])` | `matrix![ctx, [1, 2], [3, 4]]` |

## Calculus

| SymPy | symplex |
|-------|---------|
| `diff(f, x)` | `f.diff(&x)` |
| `diff(f, x, 3)` | `f.diff_n(&x, 3)` |
| `diff(f, x, y)` | `f.diff(&x).diff(&y)` |
| `integrate(f, x)` | `f.integrate(&x)` |
| `integrate(f, (x, 0, 1))` | `f.integrate_definite(&x, &ctx.int(0), &ctx.int(1))` (unevaluated node if undecided) or `f.try_integrate_definite(…)?` (`Err(Divergent)` when proven divergent) |
| `integrate(f, (x, 0, oo))` | `f.integrate_definite(&x, &ctx.int(0), &ctx.infinity())` |
| `Integral(f, (x, 0, 1)).evalf()` | `f.integrate_numeric(&x, &ctx.int(0), &ctx.int(1))?` (Gauss–Kronrod) |
| `limit(f, x, 0)` | `f.limit(&x, &ctx.int(0))` |
| `limit(f, x, 0, '+')`, `limit(f, x, 0, '-')` | `f.limit_right(&x, &ctx.int(0))`, `f.limit_left(&x, &ctx.int(0))` |
| `limit(f, x, oo)` | `f.limit(&x, &ctx.infinity())` |
| `series(f, x, 0, 5)` | `f.series(&x, &ctx.int(0), 5)` |
| `f.series(x, 0, 5).removeO()` | `f.maclaurin(&x, 5)` |
| `series(f, x, oo, 4)` | `f.series_at_infinity(&x, 4)` |
| `fps(f, x)` | `f.fps_maclaurin(&x)` → `FormalPowerSeries` (`coefficient(k)`, `general_term(&k)`, `truncate(n)`) |
| `residue(f, z, a)` | `f.residue(&z, &a)` |
| `summation(f, (k, 0, n))` | `f.summation(&k, &ctx.int(0), &n)` (or `try_summation`) |
| `product(f, (k, 1, n))` | `f.product_over(&k, &ctx.int(1), &n)` |
| `Sum(f, (k, 1, oo)).is_convergent()` | `f.is_convergent(&k)` → `Option<bool>` |
| `finite_diff_weights(1, [x-h, x, x+h], x)` | `finite_diff::finite_diff_weights(1, &[…], &x)` |
| `differentiate_finite(f, x, points=[…])` | `f.differentiate_finite(&x, &points, 1)` |

## Simplification

| SymPy | symplex |
|-------|---------|
| `simplify(expr)` | `expr.simplify()` (`simplify_with(&SimplifyOpts::single_pass())` for one pass) |
| `expand(expr)` | `expr.expand()` |
| `factor(expr)` | `expr.factor(&x)`; multivariate `expr.factor_all()` |
| `factor_list(expr)` | `expr.factor_list(&x)` → `(content, Vec<(factor, mult)>)` |
| `sqf_list(expr)` | `expr.sqf_list(&x)` |
| `resultant(f, g, x)`, `discriminant(f, x)` | `f.resultant(&g, &x)`, `f.discriminant(&x)` → `Option<Ex>` |
| `div(f, g, x)`, `gcdex(f, g, x)` | `f.poly_div(&g, &x)`, `f.poly_gcdex(&g, &x)` |
| `decompose(f, x)`, `interpolate(points, x)` | `f.decompose(&x)`, `Ex::poly_interpolate(&points, &x)` |
| `Poly(f).nroots()`, `real_roots(f)`, `count_roots(f)` | `f.nroots(&x, digits)`, `f.real_roots_isolate(&x)`, `f.count_real_roots(&x)` |
| `Poly(f).count_roots(inf, sup)` | `f.count_real_roots_in(&x, &lo, &hi)` / `p.count_real_roots_in(&lo, &hi)` on a `Poly` (endpoints rational or `±∞`) |
| `cancel(expr)` (all variables) / `ratsimp(expr)` | `expr.ratsimp()` — rational normal form, opaque non-rational subexpressions treated as indeterminates |
| `cancel(expr, x)` | `expr.cancel(&x)` |
| `expr.as_numer_denom()`, `fraction(expr)` | `expr.as_numer_denom()` — same semantics: `3/31` → `(3, 31)`, `x/2 + 1/3` → `(3*x + 2, 6)`, sums combined at every depth, nothing cancelled |
| `fraction(together(expr))`, `fraction(cancel(expr))` | `expr.as_numer_denom()` (already deep), `expr.ratsimp().as_numer_denom()` |
| `apart(expr, x)` | `expr.partial_fractions(&x)` |
| `together(expr)` | `expr.together()` |
| `collect(expr, x)` | `expr.collect(&x)` |
| `trigsimp(expr)` | `expr.simplify_trig()` |
| `expand_trig(expr)` | `expr.expand_trig()` |
| `logcombine(expr)` | `expr.log_combine()` |
| `expand_log(expr)` | `expr.expand_log()` |
| `powsimp(expr)` | `expr.simplify_powers()` |
| `combsimp(expr)` | `expr.simplify_combinatorial()` |
| `radsimp(expr)` | `expr.rationalize_denom()` |
| `sqrtdenest(expr)` | `expr.sqrtdenest()` |
| `signsimp(expr)` | `expr.signsimp()` |
| `powdenest(expr, force=True)` | `expr.powdenest(true)` |
| `expand(expr, deep=False)` | `expr.expand_with(&ExpandOpts { deep: false, ..Default::default() })` |
| `expand_power_base(expr, force=True)` | `expr.expand_power_base(true)` |
| `nsimplify(expr)` | `expr.nsimplify(tol)`; `nsimplify(expr, [pi])` → `expr.nsimplify_with_constants(&[&ctx.pi()], tol)` |
| `rcollect(expr, x, y)` | `expr.rcollect(&[&x, &y])` |
| `separatevars(expr)` | `expr.separate_vars(&[&x, &y])`, `separate_vars_dict` |
| `expr.subs(x**2, u)` (algebraic) | `expr.subs_algebraic(&x.powi(2), &u)` |
| `expr.replace(pattern, repl)` with `Wild` | `Rule::new(name, &lhs_with_a_, &rhs)` + `expr.rewrite(&RuleSet::from_rules(vec![rule]))` — see [The Rule Engine](../guide/rule-engine.md) |

## Polynomials (`Poly`)

See [Polynomials as Data](../guide/polynomials.md). Generators are explicit; anything else becomes a (possibly symbolic) coefficient.

| SymPy | symplex |
|-------|---------|
| `Poly(e, x, y)` | `e.as_poly(&[&x, &y])` → `Option<Poly>` (also `Poly::new(&e, &[&x, &y])`) |
| `Poly(e, x, y).as_dict()` / `.terms()` | `e.as_poly(&[&x, &y]).unwrap().terms()` → `Vec<(Vec<u32>, Ex)>`, lex-descending |
| `p.monoms()`, `p.coeffs()` | `p.monoms()`, `p.coeffs()` |
| `p.coeff_monomial(x*y)` | `p.coeff_monomial(&[1, 1])?` |
| `p.LC()`, `p.LM()`, `p.LT()` | `p.leading_coeff()`, `p.leading_monomial()`, `p.leading_term()` |
| `p.degree(x)`, `p.total_degree()`, `p.degree_list()` | `p.degree_in(&x)`, `p.total_degree()`, `p.degree_list()` |
| `p.all_coeffs()` (univariate, highest first) | `p.all_coeffs()` → `Option<Vec<Ex>>` |
| `degree(e, x)`, `Poly(e, x).coeffs()`, `e.coeff(x, 2)`, `LC(e, x)` | `e.degree(&x)`, `e.coeffs(&x)` (ascending), `e.coeff(&x, 2)`, `e.leading_coeff(&x)` — symbolic coefficients allowed since 0.3 |
| `p.eval({x: 1, y: 2})`, `p.eval(x, 2)` | `p.eval(&[&one, &two])?`, `p.eval_gen(&x, &two)?` |
| `p.as_expr()` | `p.to_ex()` |
| `p + q`, `p * q`, `p ** 3`, `p.diff(x)` | `p.add(&q)?`, `p.mul(&q)?`, `p.pow(3)?`, `p.derivative(&x)?` |
| `p.primitive()`, `p.monic()` | `p.content_and_primitive()`, `p.monic()` |
| `Poly(e, x).nroots()` | `e.as_poly(&[&x]).unwrap().nroots(digits)?` (or `e.nroots(&x, digits)?`) |
| `Poly(e, x).count_roots(a, b)`, `real_roots` | `p.count_real_roots_in(&a, &b)`, `p.count_real_roots()`, `p.real_roots_isolate()` |
| `Poly(e, x).shift(a)` | `p.shift(&x, &a)?` — `p(x + a)`; all coefficients `≥ 0` after shifting by `a` certifies `p ≥ 0` on `[a, ∞)` |
| (no equivalent) | `p.is_nonnegative_on(&lo, &hi)`, `p.is_positive_on(&lo, &hi)` → `Option<bool>` (exact, Sturm) |
| `Interval(lo, hi).is_subset(solve_univariate_inequality(e >= 0, x))` | `e.poly_is_nonnegative_on(&x, &lo, &hi)` (and `poly_is_positive_on`) → `Option<bool>`, exact Sturm-based decision |
| `groebner([f, g], x, y)` | `groebner::groebner_basis(&[f.to_multipoly()?, g.to_multipoly()?])`, back with `Poly::from_multipoly` |
| `Matrix` of coefficients by hand | `Poly::monomial_basis(&polys)?`, `Poly::coefficient_matrix(&polys, &basis)?` |

## Solving

| SymPy | symplex |
|-------|---------|
| `solve(f, x)` | `f.solve(&x)` → `Result<Vec<Ex>>`; identities are `Err(InfiniteSolutions)`, contradictions `Err(NoSolution)` |
| `solve(f, x)` (ignoring errors) | `f.solve_or_empty(&x)` → `Vec<Ex>` |
| `solveset(sin(x) - 1/2, x)` (with `ImageSet`) | `f.solve_general(&x)?` → `GeneralSolution { solutions, parameters }` |
| `solveset(f > 0, x)` | `f.solve_gt(&x)` → `SetEx` |
| `solveset(f >= 0, x)` | `f.solve_ge(&x)` → `SetEx` |
| `reduce_inequalities([x > 0, x <= 5], x)` | `reduce_inequalities(&[x.gt(&zero), x.le(&five)], &x)?` → `SetEx` |
| `linsolve([eq1, eq2], [x, y])` | `linsolve(&[eq1, eq2], &[x, y])?` → `LinearSolution::{Unique, Parametric, Inconsistent}` |
| `linsolve((A, b), x1, x2)` | `linsolve_matrix(&a, &b)?` (unknowns are named `x1, x2, …`; singular and inconsistent systems handled) |
| `solve(a*x**2 + b*x + c, x)` (parametric) | `(a x² + b x + c).solve(&x)?` — results are `ratsimp`'d since 0.3 |
| `sympy.solvers.simplex.lpmax(f, constraints)` / `lpmin` | `LpProblem::maximize(c).le(row, rhs)….solve()?` / `LpProblem::minimize` — exact over ℚ, see [Exact Linear Programming](../guide/exact-lp.md) |
| `sympy.solvers.simplex.linprog(c, A, b, A_eq, b_eq, bounds)` | `linprog::linprog(&c, &a_ub, &b_ub, &a_eq, &b_eq, &bounds)?` |
| `scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq)` | `linprog::linprog(…)` (exact) or `linprog_matrix(Objective::Minimize, &c, Some(&a_ub), Some(&b_ub), None, None)?` |
| "is `b` a non-negative combination of these vectors?" | `linprog::feasible_nonneg(&rows, &b)?` → `Option<Vec<Q>>` |
| dual values / infeasibility certificate | `LpSolution::duals`, `LpSolution::farkas` |
| `solve([eq1, eq2], [x, y])` (polynomial) | `symplex::polysys::solve_system_ex(&[eq1, eq2], &[x, y])?` (algebraic solutions) |
| `nsolve(f, x, x0)` | `f.solve_numeric(&x, x0, max_iter, tol)` |
| `nsolve([f1, f2], [x, y], [x0, y0])` | `solve_numeric_system(&[f1, f2], &[x, y], &[x0, y0])?` |
| `dsolve(ode, y(x))` | `ode.solve_ode(&y, &x)` |
| `dsolve(ode, y(x), ics={y(0): 0, y(x).diff(x).subs(x, 0): 1})` | `ode.solve_ode_ivp(&y, &x, &[(0, x0, v0), (1, x0, v1)])?` |
| `rsolve(a(n+2) - a(n+1) - a(n), a(n), {a(0): 0, a(1): 1})` | `rsolve::rsolve_linear(&[c0, c1, c2], forcing, &n, &[a0, a1])?` |

## Linear Algebra

| SymPy | symplex |
|-------|---------|
| `M = Matrix([[1,2],[3,4]])` | `let m = matrix![ctx, [1, 2], [3, 4]];` |
| `M.det()` | `m.det().unwrap()` |
| `M.inv()` | `m.inv().unwrap()` |
| `M.eigenvals()` | `m.eigenvals_with_multiplicity()?` (`m.eigenvals()?` lists with repetition) |
| `M.eigenvects()` | `m.eigenvects()?` → `Vec<(value, multiplicity, Vec<Matrix>)>` |
| `M.charpoly(x)` | `m.char_poly(&x)?` |
| `M.T` | `m.transpose()` |
| `M.H` | `m.adjoint()` |
| `M * N` | `&m * &n` or `m.matmul(&n)?` |
| `2 * M`, `M / 2` | `2 * &m`, `m.clone() / 2` |
| `M[i, j]`, `M[i, j] = v` | `m[(i, j)]`, `m[(i, j)] = v` |
| `M.trace()` | `m.trace()?` |
| `M.rank()` | `m.rank()` → `usize` |
| `M.nullspace()` | `m.nullspace()` → `Vec<Matrix>` (also `rowspace`, `columnspace`, `left_nullspace`) |
| integer kernel (no direct SymPy API) | `m.integer_nullspace()?` — a ℤ-basis, see [Integer Lattices](../guide/integer-lattices.md) |
| `hermite_normal_form(M)` (`sympy.matrices.normalforms`, column style `H = A·V`) | `normalforms::column_hermite_normal_form(&m)?` (leading zero columns kept); row style `H = U·A` is `m.hermite_normal_form()?` / `hermite_normal_form_with_transform` |
| `smith_normal_form(M)` | `m.smith_normal_form()?`, `normalforms::smith_normal_form_with_transforms(&m)?` → `(S, U, V)` |
| `abs(M.det()) == 1` | `normalforms::is_unimodular(&m)?` |
| `M.extract(rows, cols)` | `m.extract(&rows, &cols)?` |
| `M[rows, :]`, `M[:, cols]` | `m.select_rows(&rows)?`, `m.select_cols(&cols)?` |
| `M.row_del(i)`, `M.col_del(j)` | `m.delete_row(i)?`, `m.delete_col(j)?` (returns a new matrix) |
| `M.is_zero` | `m.is_zero()` → `Option<bool>` |
| `all(e.is_integer for e in M)` | `m.is_integer_matrix()` → `Option<bool>` |
| `M.subs({x: y, y: x})` (simultaneous) | `m.subs_map(&[(&x, &y), (&y, &x)])` |
| `Matrix(rows)` from `Rational`/`int`/`float` data | `Matrix::from_ratio(&ctx, &rows)?`, `Matrix::from_bigint`, `Matrix::from_f64_rows` (exact dyadic) |
| `[[e for e in row] for row in M.tolist()]` as numbers | `m.to_rational_rows()`, `m.to_bigint_rows()` → `Option<Vec<Vec<_>>>` |
| `M.diagonalize()` | `m.diagonalize()?` → `(P, D)` |
| `M.is_diagonalizable()` | `m.is_diagonalizable()` → `Option<bool>` |
| `M.jordan_form()` | `m.jordan_form()?` → `(P, J)` |
| `M.exp()` | `m.matrix_exp()?`; `(t*M).exp()` → `m.matrix_exp_t(&t)?` |
| `M**n` (symbolic n) | `m.matrix_pow_symbolic(&n)?` |
| `M.sqrt()` / `M**Rational(1,2)` | `m.matrix_sqrt()?` |
| `M.LUdecomposition()` | `m.lu()?` → `(L, U, permutation)` |
| `M.cholesky()` | `m.cholesky()?` |
| `M.LDLdecomposition()` | `m.ldl()?` |
| `M.QRdecomposition()` | `m.qr()?` |
| `GramSchmidt(vecs, True)` | `matrix_decomp::gram_schmidt(&vecs, true)?` |
| `M.is_symmetric()`, `M.is_positive_definite` | `m.is_symmetric()`, `m.is_positive_definite()` → `Option<bool>` |
| `M.norm()`, `M.norm(1)`, `M.norm(oo)` | `m.norm_frobenius()`, `m.norm_1()`, `m.norm_inf()` |
| `M.solve_least_squares(b)` | `m.solve_least_squares(&b)?` |
| `hessian(f, [x, y])`, `wronskian([f, g], x)` | `matrix_decomp::hessian(&f, &[&x, &y])`, `matrix_decomp::wronskian(&[&f, &g], &x)` |

## Evaluation and Substitution

| SymPy | symplex |
|-------|---------|
| `expr.subs(x, 3)` | `expr.subs(&x, &ctx.int(3))` |
| `expr.subs(x, 3)` (integer shorthand) | `expr.subs_i64(&x, 3)` |
| `expr.subs([(x, 1), (y, 2)])` | `expr.subs_map_i64(&[(&x, 1), (&y, 2)])` or `expr.eval_at(&[(&x, &a), (&y, &b)])` |
| `expr.evalf()` | `expr.eval_f64()` → `Result<f64>` |
| `expr.evalf(50)` | `expr.eval_decimal(50)` → `Result<String>` |
| `expr.is_number` | `expr.is_constant()` / `expr.as_rational().is_some()` |
| `expr.free_symbols` | `expr.free_symbols()` |
| `expr.equals(other)` | `expr.equals(&other)` → `Option<bool>`; `expr.probably_equal(&other, samples)` |
| `expr1 < expr2` (numeric) | `expr1.compare_numeric(&expr2)`, `expr1.is_less_than(&expr2)` → `Option` |

## Code Generation

| SymPy | symplex |
|-------|---------|
| `sstr(expr)` then hand-edit for Lean / Mathlib | `expr.to_lean()?` — Mathlib spacing (`2 * j + 1`, `j ^ 2`), `(3 / 31 : ℝ)`, `(j - 1) / (2 * j)`, `Real.sin x`, `0 < x ∧ x < 1` for a `BoolEx`; `to_lean_with(&LeanOpts::default().with_real_type("ℚ"))` for the carrier type / ascribing every integer |
| (wrap Lean output to ≤ 100 columns by hand) | `lean::wrap_lean(&text, lean::MATHLIB_LINE_WIDTH)`; the certificate emitters already do this |
| `ask(Q.positive(3*u**2 + 2*u + 1))` with `u` declared positive — usually `None` | `e.is_positive()` → `Some(true)` for a rational-coefficient polynomial in one real-assumed symbol (exact Sturm decision); `is_nonnegative`, `is_negative`, `is_nonpositive` likewise |
| `lambdify([x], expr)` | `expr.compile(&["x"])` → `Result<CompiledFn>` (`Clone + Send + Sync`, `arity()`, `try_call()`) |
| `lambdify([x, y], [f1, f2])` | `Ex::compile_many(&[&f1, &f2], &["x", "y"])` → `Result<CompiledFnVec>` (shared CSE) |
| `rust_code(expr)` | `expr.to_rust_fn("name", &["x"])` → `Result<String>` (`to_rust_fn_with_options` for `f32`, `no_std`, `checked_domain`, …) |
| `ccode(expr)` / `codegen(("f", expr), "C99")` | `expr.to_c_fn("f", &["x"])` → `Result<String>` (self-contained C99 with helpers) |
| `cse([expr1, expr2])` | `Ex::cse_many(&[&expr1, &expr2])` → `(Vec<(Ex, Ex)>, Vec<Ex>)`; single: `expr.cse()` |
| `latex(expr)` | `expr.to_latex()` → `String` |
| — | `expr.to_json()` → `Result<String>` |

## Number Theory

| SymPy | symplex |
|-------|---------|
| `isprime(n)` | `symplex::ntheory::isprime(n)` |
| `factorint(n)` | `symplex::ntheory::factorint(n)` |
| `nextprime(n)` | `symplex::ntheory::nextprime(n)` |
| `prevprime(n)` | `symplex::ntheory::prevprime(n)` |
| `divisors(n)` | `symplex::ntheory::divisors(n)` |
| `totient(n)` | `symplex::ntheory::totient(n)` |
| `mobius(n)` | `symplex::ntheory::mobius(n)` |
| `mod_inverse(a, m)` | `symplex::ntheory::mod_inverse(a, m)` |
| `crt([r1,r2], [m1,m2])` | `symplex::ntheory::crt_i64(&[r1,r2], &[m1,m2])` |
| `pow(a, e, m)` (3-arg pow) | `symplex::ntheory::mod_pow(a, e, m)` |
| `igcd(a, b, c)`, `ilcm(a, b, c)` | `symplex::ntheory::igcd(&[a, b, c])`, `ilcm(&[a, b, c])` (any integer type); `gcd_many(&[BigInt])`, `lcm_many` |
| `ilcm(*[r.q for r in rationals])` | `symplex::ntheory::rational_lcm_of_denominators(&rationals)` |
| `primerange(2, 50)` | `symplex::ntheory::primerange(2, 50)` / `primes_up_to(50)` |
| `primepi(n)`, `prime(n)` | `symplex::ntheory::primepi(n)`, `prime(n)` |
| `legendre_symbol(a, p)` | `symplex::ntheory::legendre_symbol(a, p)` |
| `jacobi_symbol(a, n)`, `kronecker_symbol(a, n)` | `symplex::ntheory::jacobi_symbol(a, n)?`, `kronecker_symbol(a, n)` |
| `sqrt_mod(a, p)`, `sqrt_mod(a, p, all_roots=True)` | `symplex::ntheory::sqrt_mod(a, p)`, `sqrt_mod_all(a, p)` |
| `discrete_log(n, a, b)` | `symplex::ntheory::discrete_log(b, a, n)` (base, target, modulus) |
| `primitive_root(p)`, `n_order(a, n)` | `symplex::ntheory::primitive_root(p)`, `n_order(a, n)` |
| `continued_fraction(x)`, `continued_fraction_periodic(0, 1, d)` | `symplex::ntheory::continued_fraction(&ratio)`, `continued_fraction_periodic(d)` |
| `egyptian_fraction(r)` | `symplex::ntheory::egyptian_fraction(&r)` |
| `diophantine(x**2 - 61*y**2 - 1)` | `symplex::diophantine::pell(61)`, `pell_solutions(61, k)` |
| `diophantine(3*x + 5*y - 1)` | `symplex::diophantine::linear_diophantine(3, 5, 1)` |
| `sum_of_squares(n, 2)` | `symplex::diophantine::sum_of_two_squares(n)` |

## Combinatorics

| SymPy | symplex |
|-------|---------|
| `stirling(n, k, kind=2)` | `symplex::combinatorics::stirling2(n, k)` → `Option<BigInt>` |
| `stirling(n, k, kind=1, signed=True)` | `symplex::combinatorics::stirling1(n, k)` → `Option<BigInt>` |
| `npartitions(n)` / `partition(n)` | `symplex::combinatorics::partition_count(n)` → `Option<BigInt>` |
| `multinomial_coefficients(n, k)` | `symplex::combinatorics::multinomial(n, &ks)` → `Option<BigInt>` |
| `bell(n)` | `symplex::combinatorics::bell(n)` or `ctx.int(n).bell().eval()` |
| `catalan(n)` | `symplex::combinatorics::catalan(n)` or `ctx.int(n).catalan_number().eval()` |
| `subfactorial(n)` | `symplex::combinatorics::derangements(n)` or `ctx.int(n).subfactorial().eval()` |
| `fibonacci(n)`, `lucas(n)` | `symplex::ntheory::fibonacci(n)`, `lucas(n)` or `ctx.int(n).fibonacci().eval()` |
| `bernoulli(n)`, `euler(n)`, `harmonic(n)` | `symplex::ntheory::bernoulli(n)`, `euler_number(n)`, `harmonic(n)` |
| `partitions(n)` (iterator) | `symplex::combinatorics::partitions(n)` |

## Special Functions

| SymPy | symplex |
|-------|---------|
| `gamma(x)` | `x.gamma()` |
| `erf(x)` | `x.erf()` |
| `erfc(x)` | `x.erfc()` |
| `beta(a, b)` | `a.beta(&b)` |
| `besselj(n, x)` | `x.bessel_j(&n)` |
| `bessely(n, x)` | `x.bessel_y(&n)` |
| `LambertW(x)` | `x.lambertw()` |
| `DiracDelta(x)` | `x.dirac_delta()` |
| `Heaviside(x)` | `x.heaviside()` |
| `digamma(x)` | `x.digamma()` |
| `polygamma(n, x)` | `x.polygamma(&n)` |
| `loggamma(x)` | `x.log_gamma()` |
| `zeta(s)` | `s.zeta()` |
| `Si(x)`, `Ci(x)`, `Ei(x)`, `li(x)` | `x.si()`, `x.ci()`, `x.ei()`, `x.li()` |
| `KroneckerDelta(i, j)` | `i.kronecker_delta(&j)` |
| `besseli(n, x)`, `besselk(n, x)` | `x.bessel_i(&n)`, `x.bessel_k(&n)` |
| `legendre(n, x)` | `x.legendre(&n)` |
| `chebyshevt(n, x)` | `x.chebyshev_t(&n)` |
| `hermite(n, x)` | `x.hermite(&n)` |
| `laguerre(n, x)` | `x.laguerre(&n)` |

## Sets and Logic

| SymPy | symplex |
|-------|---------|
| `Interval(0, 5)`, `Interval.Lopen(3, 10)` | `ctx.interval(&zero, &five, IntervalKind::Closed)`, `ctx.interval(&three, &ten, IntervalKind::LeftOpen)` |
| `FiniteSet(1, 2, 3)` | `ctx.finite_set(&[one, two, three])` |
| `S.Reals`, `S.EmptySet` | `ctx.reals()`, `ctx.empty_set()` |
| `A.union(B)`, `A.intersect(B)` | `a.union(&b).simplify()`, `a.intersection(&b).simplify()` |
| `A - B`, `A.symmetric_difference(B)`, `A.complement(S.Reals)` | `a.difference(&b)`, `a.symmetric_difference(&b)`, `a.absolute_complement()` |
| `A.contains(x)`, `x in A` | `a.contains(&x)` → `Option<bool>`, `x.is_in(&a)` |
| `A.is_subset(B)`, `A.is_disjoint(B)` | `a.is_subset(&b)`, `a.is_disjoint(&b)` → `Option<bool>` |
| `A.inf`, `A.sup`, `A.measure`, `A.boundary`, `A.closure`, `A.interior` | `a.inf()`, `a.sup()`, `a.measure()`, `a.boundary()`, `a.closure()`, `a.interior()` → `Option` |
| `A.as_relational(x)` | `a.to_condition(&x)?` |
| `And(p, q)`, `Or(p, q)`, `Not(p)`, `Implies(p, q)` | `p.and(&q)`, `p.or(&q)`, `p.not()`, `p.implies(&q)` |
| `to_nnf`, `to_cnf`, `to_dnf` | `b.to_nnf()`, `b.to_cnf()`, `b.to_dnf()` |
| `satisfiable(b)` | `b.satisfiable()` → `Option<bool>`; `b.is_tautology()`, `b.is_contradiction()` |
| `truth_table(b, [p, q])` | `b.truth_table(&[p, q])?` |
| `piecewise_fold` / `Piecewise.simplify()` | `expr.piecewise_simplify()` |

## Transforms

| SymPy | symplex |
|-------|---------|
| `fourier_transform(f, t, w)` (ordinary frequency) | `f.fourier_transform_with(&t, &w, FourierConvention::Ordinary)?` |
| `fourier_transform` with `2π` angular frequency | `f.fourier_transform(&t, &w)?` (non-unitary angular) |
| `inverse_fourier_transform(F, w, t)` | `F.inverse_fourier_transform(&w, &t)?` |
| `mellin_transform(f, x, s)` → `(F, (a, b), cond)` | `f.mellin_transform(&x, &s)?` → `(F, strip: BoolEx)` |
| `inverse_mellin_transform(F, s, x, (a, b))` | `F.inverse_mellin_transform(&s, &x)?` |
| `fourier_series(f, (x, -pi, pi))` | `f.fourier_series_on(&x, &(-ctx.pi()), &ctx.pi(), n)?` |
| `s.truncate(n)`, `s.an`, `s.bn` | `s.truncate(n)`, `s.coefficient_a(k)`, `s.coefficient_b(k)` |
| Z-transform (not in SymPy core) | `f.z_transform(&n, &z)?`, `F.inverse_z_transform(&z, &n)?` |

## Numerical Optimisation (SciPy / NumPy)

SymPy defers to SciPy and NumPy here; symplex ships equivalents in `symplex::optimize` (see [Numerical Optimisation](../guide/numerical-optimization.md)). All are deterministic and return `Result`.

| SciPy / NumPy | symplex |
|---------------|---------|
| `scipy.optimize.brentq(f, a, b)` | `optimize::brent_root(f, a, b, &RootOpts::default())?`; on an `Ex`: `e.find_root_bracket(&x, a, b)?` |
| `scipy.optimize.bisect(f, a, b)` | `optimize::bisect(f, a, b, &opts)?` |
| `scipy.optimize.newton(f, x0, fprime)` | `optimize::newton_root(f, df, x0, &opts)?` (derivative from `e.diff(&x).compile(..)`) |
| `scipy.optimize.minimize(f, x0, method="Nelder-Mead")` | `optimize::nelder_mead(f, &x0, &MinimizeOpts::default())?`; on an `Ex`: `e.minimize_numeric(&[&x, &y], &x0)?` → `MinimizeResult { x, fun, iterations, evaluations, converged }` |
| `scipy.optimize.minimize_scalar(f, bounds=(a, b), method="bounded")` | `optimize::minimize_scalar(f, a, b, &opts)?` (Brent), `golden_section`; on an `Ex`: `e.minimize_scalar_numeric(&x, a, b)?` → `ScalarMinimum { x, value }` |
| `scipy.optimize.differential_evolution(f, bounds, seed=0)` | `optimize::differential_evolution(f, &bounds, &DeOpts { seed, .. })?` with `bounds: &[Interval<f64>]` (closed, `Interval::closed(lo, hi)`); on an `Ex`: `e.minimize_global_numeric(&vars, &bounds, &opts)?` |
| `numpy.polyfit(x, y, deg)` (**highest degree first**) | `optimize::poly_fit(&xs, &ys, deg)?` (**ascending**: `[c₀, c₁, …]`); `eval_poly(&c, x)` evaluates |
| `numpy.polyfit` with exact rationals (no NumPy equivalent) | `optimize::poly_fit_exact(&points, deg)?`, `Ex::poly_fit_points(&ctx, &points, &x, deg)?` → `Ex` |
| `scipy.stats.linregress(x, y)` (slope, intercept) | `optimize::linear_fit(&xs, &ys)?` → `LinearFit { slope, intercept }` |
| `numpy.trapz(y, x)` / `scipy.integrate.trapezoid(y, x)` | `optimize::trapezoid(&ys, &xs)?` |
| `scipy.optimize.fsolve(F, x0)` | `solve_numeric_system(&eqs, &vars, &x0)?` (damped Newton, symbolic Jacobian) |

## Dimensional Analysis

SymPy does not have a built-in compile-time unit system. symplex provides one:

```rust
use symplex::units::*;

let ctx = Context::new();
let m = Mass::symbol(&ctx, "m");
let a = Acceleration::symbol(&ctx, "a");

// Type-safe: the compiler verifies the dimension
let force = dim!(ctx, Force: m * a);

// Typed calculus: d(Length)/d(Time) → Velocity
symplex::syms!(ctx; g, t);
let t_var = Time::symbol(&ctx, "t");
let pos = Length::from_ex(expr!(ctx, 1/2 * g * t^2));
let vel: Velocity = pos.diff_wrt(&t_var);
```

There is no SymPy equivalent for this. The closest is SymPy's `physics.units` module, which performs dimensional analysis at runtime rather than compile time.

## ODE Solving

| SymPy | symplex |
|-------|---------|
| `dsolve(Eq(f(x).diff(x), f(x)), f(x))` | (see below) |
| `classify_ode(ode)` | `ode_expr.classify_ode(&y, &x)` |

ODE solving in symplex uses a different interface from SymPy. Instead of wrapping the ODE in `Eq()` and using `Function('f')`, you build the ODE as an expression involving `y.formal_diff(&x)`:

```rust
let ctx = Context::new();
symplex::syms!(ctx; x);
let y = ctx.symbol("y");
let dy = y.formal_diff(&x);

// y' + 2y = 0
let ode = &dy + &y * 2;
let solution = ode.solve_ode(&y, &x);
println!("{solution}");   // y = C1*exp(-2*x)
```

symplex supports 16 ODE classes: simple separable, full separable, first-order linear (constant and variable coefficient), exact, integrating factor, Bernoulli, Riccati (`solve_riccati` with a particular solution), Euler–Cauchy, homogeneous-coefficient, second-order constant-coefficient (homogeneous and non-homogeneous), nth-order constant-coefficient, reduction of order, variation of parameters, and Clairaut. Initial-value problems use `solve_ode_ivp`; linear systems `x' = Ax` use `ode::solve_ode_system_ivp`.

## Things That Don't Have Direct Equivalents

### In SymPy but not symplex

- `Permutation`, `PermutationGroup`, and abstract algebra
- `geometry` module (Point, Line, Circle, Polygon)
- `stats` module (probability distributions)
- `tensor` module (indexed tensors, Einstein summation)
- `physics.quantum` module
- `pdsolve` (PDE solving)
- General `diophantine()` (symplex has `linear_diophantine`, `pell`, `sum_of_two_squares`, `pythagorean_triples`, not the general classifier)
- `rsolve` for polynomial/rational/hypergeometric coefficients (symplex has constant-coefficient linear and first-order recurrences)
- `hyper`, `meijerg` and the Meijer-G integration engine
- `Eq` as a first-class expression in every API (symplex has `Equation` with `solve`, `solve_for`, arithmetic, and `eq!`, accepted by `linsolve`)
- Pretty-printing with Unicode box drawing (`pprint`) — symplex has `pretty()` / `pretty_ascii()`
- `O()` notation for series remainders

### In symplex but not SymPy

- Compile-time dimensional analysis (`dim!` macro, quantity types)
- Compile-time expression type safety (`Ex` / `BoolEx` / `SetEx`)
- Thread-safe contexts (`Send + Sync`, no GIL)
- Optimized Rust *and* C99 code generation with CSE and an embedded special-function runtime (`to_rust_fn`, `to_c_fn`)
- Compiled closures for fast numerical evaluation (`compile`, `compile_many`)
- Exact `RootOf` eigenvalues for irreducible cubic/quartic characteristic polynomials
- Exact linear programming over ℚ with shadow prices and Farkas certificates (`symplex::linprog`)
- Exact sign of a polynomial on an interval (`poly_is_nonnegative_on`, Sturm-based)
- Row-style Hermite normal form with its unimodular transform, integer kernels, lattice determinants
- Deterministic numerical optimisation (`symplex::optimize`) in the same crate as the CAS
- `Err(Divergent)` / `Err(NoSolution)` / `Err(InfiniteSolutions)` as first-class outcomes
- `EvalConfig` for user-controllable computation limits
- `build.rs` code generation pipeline for embedded targets
- WASM compilation target (`symplex-wasm`)