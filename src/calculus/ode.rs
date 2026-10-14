//! Ordinary Differential Equation (ODE) solver.
//!
//! Solves first-order and second-order ODEs:
//!
//! - **Full separable:** `dy/dx = f(x) * g(y)` → `∫ 1/g(y) dy = ∫ f(x) dx`,
//!   solved for `y` when that has one solution
//! - **Simple separable:** `a(x)·dy/dx = f(x)` (no y dependence)
//! - **First-order linear (variable coefficient):** `a(x)·y' + P(x)*y = Q(x)`
//!   → `y = (1/μ) * [∫ Q(x)*μ dx + C1]` where `μ = exp(∫ P(x) dx)`
//!   (after dividing by `a`)
//! - **First-order linear constant-coefficient:** `y' + a*y = f(x)`
//!   → `y = e^(-ax) * ∫ f(x)*e^(ax) dx`
//! - **Exact** `M + N·y' = 0` (`∂M/∂y = ∂N/∂x`, decided as a rational
//!   function) and with an **integrating factor** `μ(x)` or `μ(y)`
//! - **Bernoulli:** `y' + P·y = Q·yⁿ`
//! - **Second-order linear constant-coefficient:** `y'' + b*y' + c*y = 0`
//!   → characteristic equation `r² + b*r + c = 0`, solution based on roots
//! - **Cauchy–Euler:** `a·x²y'' + b·x·y' + c·y = g(x)` (for `x > 0`; a
//!   forcing term through `x = eᵗ` and the constant-coefficient solvers),
//!   and of any order `Σ aₖ·xᵏ·y⁽ᵏ⁾ = g(x)` the same way
//! - **Variation of parameters** for `y'' + b·y' + c·y = g(x)`
//! - **Homogeneous coefficient:** `y' = f(y/x)` — substitution `v = y/x`
//! - **nth-order reducible:** `F(y, y', y'') = 0` (no `x`) — substitution
//!   `p(y) = y'`; `F(x, y', y'') = 0` (no `y`) — `p(x) = y'`, `y = ∫ p dx`
//! - **nth-order linear constant-coefficient:** `Σ a_k y^(k) = g(x)` for any
//!   order via the characteristic polynomial (repeated roots → `x^k e^{rx}`,
//!   complex pairs → `e^{ax}(cos bx, sin bx)`), with undetermined coefficients
//!   for `poly × exp × {sin, cos}` forcing (resonance handled; irrational
//!   rates and frequencies such as `e^{√2·x}`, `cos(√3·x/2)` in exact
//!   arithmetic, `sinh`/`cosh` as exponentials); other forcing terms by the
//!   Green's function `∫ G(x − t)·g(t) dt` with complex exponentials when
//!   the characteristic polynomial's factors have degree ≤ 2 (`y″ + y =
//!   eˣ/x` → `Ei((1 ∓ i)x)`, `y‴ − y′ = 1/x`); roots of an
//!   irreducible factor of degree ≥ 3 in Cardano's radicals with real and
//!   imaginary parts (`y‴ + y″ + 2y = 0`), or as `RootOf` with
//!   `e^{x·re(RootOf(…))}·cos(x·im(RootOf(…)))` (SymPy's real fundamental
//!   system), and initial conditions fitted by the Lagrange formula for the
//!   inverse Vandermonde matrix (in `Ex::solve_ode_ivp`)
//! - **Riccati with a rational particular solution:** `y′ = b₀ + b₁y + b₂y²`
//!   with rational coefficients, without a given particular solution (SymPy's
//!   `1st_rational_riccati`)
//! - **Linear argument:** `y′ = F(a·x + b·y + c)` — `u = a·x + b·y + c`
//!   (`y′ = (y − x)²`, `y′ = sin(x + y)`); **linear coefficients:**
//!   `y′ = F((a₁x + b₁y + c₁)/(a₂x + b₂y + c₂))` — a shift to the lines'
//!   intersection makes it homogeneous
//! - **Higher-order reducible:** `F(x, y⁽ᵏ⁾, …, y⁽ⁿ⁾) = 0` — `p = y⁽ᵏ⁾`;
//!   an equation of degree 2–4 in its highest derivative is solved for it
//!   (`y″² = 4y′`)
//! - **Linear with symbolic constant coefficients:** `y″ + a·y = 0` →
//!   `C1·e^{x√(−a)} + C2·e^{−x√(−a)}` (roots of the characteristic
//!   polynomial in radicals, principal branches, no case split on the sign
//!   of `a`, as SymPy; the generic solution), any order; a second-order
//!   forcing term by variation of parameters
//! - **Clairaut:** `y = x·y' + f(y')` → `y = C·x + f(C)`
//! - **Riccati:** `y' = q₀ + q₁·y + q₂·y²` given a particular solution
//!   ([`solve_riccati`])
//! - **Constant-coefficient systems:** `ẋ = A·x` → `x(t) = exp(A·t)·c`
//!   via the eigenvectors (real modes for complex pairs) or the exact matrix
//!   exponential of the Jordan form, with initial values via
//!   [`solve_ode_system_ivp`]
//! - **Non-homogeneous systems:** `ẋ = A·x + b(t)` → variation of parameters
//!
//! An equation is also recognised in an equivalent form: multiplied or
//! divided by factors free of `y` (`eˣ·(y″ + y − tan x)`), by factors in
//! `y` alone (`y·(y′ − cos x·(1 + y²))`, whose solution `y = 0` is then
//! not reported), or expanded after such a product, with the coefficient of
//! the highest derivative divided out (see [`dsolve`]).
//!
//! # Implicit solutions
//!
//! A solution that could not be solved for `y` is returned as an expression
//! `G(x, y, C1, …)` whose zero set is the family (`G = 0`); it still
//! contains `y`.  [`dsolve`] prefers an explicit solution found by a later
//! method to an implicit one found by an earlier one.  An implicit
//! relation is solved for `y` with every root a family, and, failing that,
//! with `ln|u|` read as `ln u` and `e^{k·C1}` renamed `C1` (the constant
//! absorbs both; `y′ = y² − 1` gives `(−C1·e^{2x} − 1)/(C1·e^{2x} − 1)`, the
//! family of SymPy's `1/tanh(C1 − x)`); a root without the constant is not
//! a family.
//!
//! # Branches
//!
//! Where the general solution has several families, [`OdeResult::solution`]
//! is the first and [`OdeResult::branches`] holds the others, as SymPy's
//! list of solutions: a Bernoulli equation whose `1/(1−n)` has an even
//! denominator gives `y = ±v^{1/(1−n)}` with `v` the solution of the linear
//! equation (`y' = y³`: `(C1 − 2x)^{−1/2}` and `−(C1 − 2x)^{−1/2}`), a
//! separable equation every explicit root of `∫ dy/g(y) = ∫ f dx + C1`, and
//! `y'' = F(y')` (no `y`) one family per family of `y'` (`y'' = y'³`:
//! `C2 ∓ √(C1 − 2x)`).  Before 0.30 only the first family was returned.
//! [`Ex::solve_ode`](crate::api::expr::Ex::solve_ode) returns the first
//! family, `Ex::solve_ode_all` all of them, and `Ex::solve_ode_ivp` fits the
//! initial conditions on each family in turn (`y' = y³`, `y(0) = −1` is on
//! the negative one).
//!
//! A formula is a solution where it is real on the principal branch:
//! `y' = √y` gives `(x/2 + C1/2)²`, a solution for `x + C1 ≥ 0` only (for
//! `x + C1 < 0` its derivative is negative while `√y ≥ 0`; the global
//! solution is `0` there, the parabola after), and the singular solution
//! `y = 0` is not included; `solve_ode_ivp` keeps only constants for which
//! the ODE holds at the initial point (`y(0) = 1` gives `C1 = 2`, not the
//! other root `C1 = −2`).  Cauchy–Euler solutions are for `x > 0` (`ln x`,
//! `x^r`).

use crate::api::expr::Ex;
use crate::base::arena::Arena;
use crate::base::combinatorics::binomial;
use crate::base::node::{ExprId, ExprNode, SymbolId};
use crate::base::numeric::Q;
use crate::domains::matrix::Matrix;
use num_traits::One;
use num_traits::Signed;

/// Most families [`try_full_separable`] returns (roots of `∫ dy/g(y) =
/// ∫ f dx + C1`); more stay implicit.
const MAX_BRANCHES: usize = 8;

/// An ODE representation: f(x, y, y', y'', ...) = 0
/// For now, we support limited forms detected by pattern matching.
pub struct OdeResult {
    /// The general solution y = ... (may contain constants C1, C2)
    pub solution: ExprId,
    /// Names of the arbitrary constants
    pub constants: Vec<ExprId>,
    /// The other families of the general solution, in the same constants:
    /// with `solution` they form the complete set (`y′ = y³` has
    /// `(C1 − 2x)^(−1/2)` and `−(C1 − 2x)^(−1/2)`).  Empty when `solution`
    /// is the whole family.  See *Branches* in the module docs.
    pub branches: Vec<ExprId>,
}

/// Attempt to solve a first-order or second-order ODE.
///
/// The ODE is given as `expr = 0` where `expr` may contain:
/// - `var` (the independent variable, typically `x`)
/// - `func` (the dependent variable, typically `y`)
/// - `Derivative(func, var)` (first derivative `y'`)
/// - `Derivative(Derivative(func, var), var)` (second derivative `y''`)
///
/// Returns `None` if the ODE type is not recognized.
///
/// The equation is tried as given and then in normal forms (a product
/// reduced to its one factor with a derivative of `func`, the equation
/// expanded with the highest derivative's coefficient divided out): `(x + 3)·(y‴ − 3y″ + 4y′ − 12y)`
/// expanded, `eˣ·(y″ + y − tan x)` or `(y′ − y)/(−1)` were not
/// recognised before (SymPy's `dsolve` solves them).
pub fn dsolve(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId, // y
    var: ExprId,  // x
) -> Option<OdeResult> {
    let ExprNode::Symbol(func_sym) = *arena.node(func) else {
        return None;
    };
    let first = dsolve_as_given(arena, expr, func, var);
    if first
        .as_ref()
        .is_some_and(|r| !contains_sym(arena, r.solution, func_sym))
    {
        return first;
    }
    let mut best = first;
    for form in normalized_forms(arena, expr, func, var, func_sym) {
        let Some(result) = dsolve_as_given(arena, form, func, var) else {
            continue;
        };
        if !contains_sym(arena, result.solution, func_sym) {
            return Some(result);
        }
        if best.is_none() {
            best = Some(result);
        }
    }
    // Every method left the solution implicit: its roots in `func`, each a
    // family (see [`explicit_families`]; the additive constant is the last).
    if let Some(r) = &best
        && let Some(&c) = r.constants.last()
        && let Some((solution, branches)) = explicit_families(arena, expr, r.solution, func, var, c)
    {
        return Some(OdeResult {
            solution,
            constants: r.constants.clone(),
            branches,
        });
    }
    best
}

/// Equivalent forms of the ODE `expr = 0` for [`dsolve`], each different
/// from `expr` and from the previous ones:
///
/// 1. a product with exactly one factor containing a derivative of `func`
///    reduced to that factor (a positive power of it to its base): factors
///    free of `func` are non-zero where the equation is defined, and
///    factors in `func` alone (`y`, `1 + y²`) only add algebraic solutions
///    such as `y = 0` (lost, as when dividing by `y`);
/// 2. that form expanded and divided by the coefficient of its highest
///    derivative, when the equation is linear in that derivative with a
///    coefficient free of `func` that is not a number: each coefficient
///    of a power product of `func` and its derivatives is reduced as a
///    rational function (`(x + 3)·y‴ − (3x + 9)·y″ → y‴ − 3y″`).
fn normalized_forms(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Vec<ExprId> {
    let mut forms: Vec<ExprId> = Vec::new();
    let mut current = expr;
    // Nested products/powers: a few rounds at most.
    for _ in 0..4 {
        match strip_nonderivative_factors(arena, current, func_sym) {
            Some(next) if next != current => current = next,
            _ => break,
        }
    }
    if current != expr {
        forms.push(current);
    }
    if let Some(lead) = divide_leading_coefficient(arena, current, func, var, func_sym)
        && lead != expr
        && !forms.contains(&lead)
    {
        forms.push(lead);
    }
    forms
}

/// Does `e` contain a `Derivative` node (of anything)?
fn has_derivative(arena: &Arena, e: ExprId) -> bool {
    let mut stack = vec![e];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let node = arena.node(n);
        if matches!(node, ExprNode::Derivative(..)) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// Form 1 of [`normalized_forms`] (one step): `None` when `e` is not a
/// product or power of that shape.
fn strip_nonderivative_factors(arena: &mut Arena, e: ExprId, func_sym: SymbolId) -> Option<ExprId> {
    let positive_power_base = |arena: &Arena, f: ExprId| -> ExprId {
        match arena.node(f) {
            ExprNode::Pow(b, p) if arena.as_num(*p).is_some_and(|r| r.is_positive()) => *b,
            _ => f,
        }
    };
    match arena.node(e).clone() {
        ExprNode::Mul(factors) => {
            let carriers: Vec<ExprId> = factors
                .iter()
                .copied()
                .filter(|&f| contains_sym(arena, f, func_sym) && has_derivative(arena, f))
                .collect();
            let [carrier] = carriers.as_slice() else {
                return None;
            };
            // A derivative only in a denominator is not an equation in it.
            let base = positive_power_base(arena, *carrier);
            if base == *carrier && matches!(arena.node(*carrier), ExprNode::Pow(..)) {
                return None;
            }
            Some(base)
        }
        ExprNode::Neg(inner) => Some(inner),
        ExprNode::Pow(..) => {
            let base = positive_power_base(arena, e);
            (base != e && has_derivative(arena, base)).then_some(base)
        }
        _ => None,
    }
}

/// Form 2 of [`normalized_forms`].
fn divide_leading_coefficient(
    arena: &mut Arena,
    e: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Option<ExprId> {
    const MAX_TERMS: usize = 64;
    let chain = derivative_chain(arena, func, var);
    let order = (1..chain.len())
        .rev()
        .find(|&k| expr_contains(arena, e, chain[k]))?;
    let lead = chain[order];
    let expanded = crate::transforms::expand::expand(arena, e);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    let terms: Vec<ExprId> = match arena.node(expanded) {
        ExprNode::Add(c) => c.to_vec(),
        _ => return None,
    };
    if terms.len() > MAX_TERMS {
        return None;
    }
    // (product of the factors in `func`, sum of their coefficients)
    let mut groups: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
    for t in terms {
        let (sign, inner) = match arena.node(t) {
            ExprNode::Neg(i) => (true, *i),
            _ => (false, t),
        };
        let factors: Vec<ExprId> = match arena.node(inner) {
            ExprNode::Mul(fs) => fs.to_vec(),
            _ => vec![inner],
        };
        let (in_func, coeff): (Vec<ExprId>, Vec<ExprId>) = factors
            .into_iter()
            .partition(|&f| contains_sym(arena, f, func_sym));
        let key = match in_func.len() {
            0 => arena.one,
            1 => in_func[0],
            _ => arena.mul(&in_func),
        };
        let mut c = match coeff.len() {
            0 => arena.one,
            1 => coeff[0],
            _ => arena.mul(&coeff),
        };
        if sign {
            c = arena.neg(c);
        }
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some(g) => g.1.push(c),
            None => groups.push((key, vec![c])),
        }
    }
    // Linear in the highest derivative, alone in its group.
    let carriers: Vec<usize> = (0..groups.len())
        .filter(|&i| expr_contains(arena, groups[i].0, lead))
        .collect();
    let [lead_group] = carriers.as_slice() else {
        return None;
    };
    if groups[*lead_group].0 != lead {
        return None;
    }
    let c = {
        let parts = groups[*lead_group].1.clone();
        let s = arena.add(&parts);
        crate::transforms::eval::eval(arena, s)
    };
    if arena.as_num(c).is_some() || contains_sym(arena, c, func_sym) {
        return None;
    }
    let inv = {
        let m1 = arena.int(-1);
        arena.pow(c, m1)
    };
    let mut out = Vec::with_capacity(groups.len());
    for (key, parts) in groups {
        let s = arena.add(&parts);
        let q = arena.mul(&[s, inv]);
        let q = crate::transforms::eval::eval(arena, q);
        let q = crate::simplify::ratsimp::ratsimp(arena, q);
        let q = crate::transforms::eval::eval(arena, q);
        if arena.is_zero_structural(q) {
            continue;
        }
        out.push(arena.mul(&[q, key]));
    }
    let sum = arena.add(&out);
    Some(crate::transforms::eval::eval(arena, sum))
}

/// [`dsolve`] on the equation exactly as given.
fn dsolve_as_given(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId, // y
    var: ExprId,  // x
) -> Option<OdeResult> {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => return None,
    };
    let func_sym = match arena.node(func) {
        ExprNode::Symbol(sid) => *sid,
        _ => return None,
    };

    // Try to detect the ODE type.  An implicit solution (one still in
    // `func`) is kept while the later methods get a chance to find an
    // explicit one: the integrating factor of `y′ = y − y²` gave
    // `x − ln|y| + ln|y − 1| − C1`, Bernoulli gives `y` itself.
    let mut implicit: Option<OdeResult> = None;
    macro_rules! attempt {
        ($result:expr) => {
            if let Some(result) = $result {
                if contains_sym(arena, result.solution, func_sym) {
                    if implicit.is_none() {
                        implicit = Some(result);
                    }
                } else {
                    return Some(result);
                }
            }
        };
    }

    // Type 1: Second-order constant-coefficient: a*y'' + b*y' + c*y = 0
    attempt!(try_second_order_const_coeff(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 1b: Second-order CC nonhomogeneous: a*y'' + b*y' + c*y = f(x)
    attempt!(try_second_order_cc_nonhomogeneous(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 1c: Euler-Cauchy: a·x²·y'' + b·x·y' + c·y = 0
    attempt!(try_euler_cauchy(arena, expr, func, var, func_sym, var_sym));
    // … and of order ≥ 3: Σ aₖ·xᵏ·y⁽ᵏ⁾ = g(x)
    attempt!(try_euler_cauchy_n(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 1e: nth-order linear constant-coefficient (any order ≥ 2),
    // forcing = poly × exp × {sin, cos} via undetermined coefficients.
    attempt!(try_nth_order_linear_const_coeff(
        arena, expr, func, var, func_sym
    ));

    // Type 1e': linear with constant symbolic coefficients (`y″ + a·y = 0`).
    attempt!(try_linear_cc_symbolic(arena, expr, func, var, func_sym));

    // Type 1d: Variation of parameters: y'' + p·y' + q·y = g(x) (fallback)
    attempt!(try_variation_of_parameters(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 1d': any order ≤ 6, roots of degree ≤ 2: Green's function
    // (complex exponentials) for forcing outside undetermined coefficients.
    attempt!(try_linear_cc_green(arena, expr, func, var, func_sym));

    // Type 1f: Clairaut: y = x·y' + f(y')
    attempt!(try_clairaut(arena, expr, func, var, func_sym, var_sym));

    // Type 2: General first-order linear (variable P(x)): y' + P(x)*y = Q(x)
    attempt!(try_first_order_linear_general(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 2b: Exact first-order ODE: M(x,y) + N(x,y)·y' = 0 with ∂M/∂y = ∂N/∂x
    attempt!(try_exact_ode(arena, expr, func, var, func_sym, var_sym));

    // Type 2c: Non-exact ODE with integrating factor μ(x) or μ(y)
    attempt!(try_integrating_factor_ode(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 2d: Bernoulli: y' + P(x)·y = Q(x)·y^n (n ≠ 0, 1)
    attempt!(try_bernoulli(arena, expr, func, var, func_sym, var_sym));

    // Type 2e: Homogeneous coefficient: y' = f(y/x)
    attempt!(try_homogeneous_coefficient(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 2f: nth-order reducible: F(y, y', y'') = 0, no explicit x
    attempt!(try_nth_order_reducible(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 3: Full separable: y' = f(x)*g(y)
    attempt!(try_full_separable(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 4: Simple separable: y' = f(x) (no y dependence) — fallback
    attempt!(try_simple_separable(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 4b: Riccati with rational coefficients and a rational particular
    // solution.
    attempt!(try_rational_riccati(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 5: y' = F(a·x + b·y + c) (substitution u = a·x + b·y + c) and
    // y' = F((a₁x + b₁y + c₁)/(a₂x + b₂y + c₂)) (shift to homogeneous).
    attempt!(try_linear_argument(
        arena, expr, func, var, func_sym, var_sym
    ));
    attempt!(try_linear_coefficients(
        arena, expr, func, var, func_sym, var_sym
    ));

    // Type 6: F(x, y⁽ᵏ⁾, …, y⁽ⁿ⁾) = 0, n ≥ 3 (p = y⁽ᵏ⁾), and an equation
    // polynomial of degree ≥ 2 in its highest derivative (solved for it).
    attempt!(try_reduce_lowest_order(arena, expr, func, var, func_sym));
    attempt!(try_solve_for_highest_derivative(
        arena, expr, func, var, func_sym
    ));

    // Type 7 (last): first-order linear with a non-elementary ∫P dx, in
    // formal integrals.
    if implicit.is_none() {
        attempt!(try_first_order_linear_formal(
            arena, expr, func, var, func_sym, var_sym
        ));
    }

    implicit
}

/// The highest order `n ≥ 1` of a derivative of `func` in `expr`.
fn ode_order(arena: &mut Arena, expr: ExprId, func: ExprId, var: ExprId) -> Option<usize> {
    let chain = derivative_chain(arena, func, var);
    (1..chain.len())
        .rev()
        .find(|&k| expr_contains(arena, expr, chain[k]))
}

/// Solve `F(x, y⁽ᵏ⁾, …, y⁽ⁿ⁾) = 0` with `k ≥ 1` and `n ≥ 3` (orders below `k`,
/// `y` itself included, absent): `p = y⁽ᵏ⁾` satisfies an equation of order
/// `n − k`, and each family of `p` integrated `k` times, plus
/// `Cₘ₊₁ + Cₘ₊₂·x + … + Cₘ₊ₖ·xᵏ⁻¹`, is a family of `y` (SymPy's
/// `nth_order_reducible`).  `y‴ = y″/x` (`C1·x³/6 + C2·x + C3`) and
/// `y‴·y″ = 1` were refused; order 2 is [`try_nth_order_reducible`]'s.
fn try_reduce_lowest_order(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Option<OdeResult> {
    let n = ode_order(arena, expr, func, var)?;
    if n < 3 {
        return None;
    }
    let chain = derivative_chain(arena, func, var);
    // Each derivative replaced by a placeholder, the highest first (a
    // derivative node contains the lower ones): the lowest order present,
    // and `y` itself must be absent.
    let mut probe = expr;
    let mut placeholders = Vec::with_capacity(n);
    for j in (1..=n).rev() {
        let ph = arena.symbol(&format!("__red_d{j}"));
        probe = crate::transforms::subs::subs(arena, probe, chain[j], ph);
        placeholders.push((j, ph));
    }
    if contains_sym(arena, probe, func_sym) {
        return None;
    }
    let k = placeholders
        .iter()
        .rev()
        .find(|&&(_, ph)| crate::base::walk::contains(arena, probe, ph))
        .map(|&(j, _)| j)?;
    if k >= n {
        return None;
    }
    let p = arena.symbol("__p_low");
    let ExprNode::Symbol(p_sym) = *arena.node(p) else {
        return None;
    };
    let pchain = derivative_chain(arena, p, var);
    let mut reduced = expr;
    for j in (k..=n).rev() {
        reduced = crate::transforms::subs::subs(arena, reduced, chain[j], pchain[j - k]);
    }
    let reduced = crate::transforms::eval::eval(arena, reduced);
    if contains_sym(arena, reduced, func_sym) {
        return None;
    }
    let pr = dsolve(arena, reduced, p, var)?;
    if contains_sym(arena, pr.solution, p_sym)
        || crate::base::walk::has_unevaluated(arena, pr.solution)
        || pr.constants.len() != n - k
    {
        return None;
    }
    // Fresh constants after the largest index used.
    let mut next = 1usize;
    for &c in &pr.constants {
        if let ExprNode::Symbol(s) = arena.node(c)
            && let Some(i) = arena
                .symbol_name(*s)
                .strip_prefix('C')
                .and_then(|d| d.parse::<usize>().ok())
        {
            next = next.max(i + 1);
        }
    }
    let fresh: Vec<ExprId> = (0..k)
        .map(|j| arena.symbol(&format!("C{}", next + j)))
        .collect();
    // After the first integration, `ln|u| → ln u` changes an antiderivative
    // by a constant on each interval, which the remaining integrations turn
    // into a polynomial of degree `< k`: absorbed by the added constants.
    let integrate_k = |arena: &mut Arena, fam: ExprId| -> Option<ExprId> {
        let mut f = fam;
        for step in 0..k {
            let mut g = integrate_forms(arena, f, var);
            if integral_failed(arena, g)
                && step > 0
                && let Some(h) = ln_abs_to_ln(arena, f)
            {
                g = integrate_forms(arena, h, var);
            }
            if integral_failed(arena, g) {
                return None;
            }
            f = g;
        }
        let mut terms = vec![f];
        for (j, &c) in fresh.iter().enumerate() {
            let xj = match j {
                0 => arena.one,
                1 => var,
                _ => {
                    let e = arena.int(i64::try_from(j).ok()?);
                    arena.pow(var, e)
                }
            };
            terms.push(arena.mul(&[c, xj]));
        }
        let s = arena.add(&terms);
        Some(crate::transforms::eval::eval(arena, s))
    };
    let solution = integrate_k(arena, pr.solution)?;
    let branches = pr
        .branches
        .iter()
        .filter_map(|&b| integrate_k(arena, b))
        .collect();
    let mut constants = pr.constants;
    constants.extend(fresh);
    Some(OdeResult {
        solution,
        constants,
        branches,
    })
}

/// Most roots in the highest derivative [`try_solve_for_highest_derivative`]
/// follows.
const MAX_DERIVATIVE_ROOTS: usize = 4;

/// Solve an equation polynomial of degree `2..=4` in its highest
/// derivative `y⁽ⁿ⁾` by solving for it: each root `y⁽ⁿ⁾ = Fᵢ` is an equation
/// of the first degree, and its families are families of the original
/// (SymPy's `factorable`).  `y″² = 4y′` (`(x + C1)³/3 + C2`) was refused.
fn try_solve_for_highest_derivative(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Option<OdeResult> {
    let n = ode_order(arena, expr, func, var)?;
    let chain = derivative_chain(arena, func, var);
    let t = arena.symbol("__t_hd");
    let ExprNode::Symbol(t_sym) = *arena.node(t) else {
        return None;
    };
    let g = crate::transforms::subs::subs(arena, expr, chain[n], t);
    let g = crate::transforms::eval::eval(arena, g);
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, g, t)?;
    if !(3..=MAX_DERIVATIVE_ROOTS + 1).contains(&coeffs.len()) {
        return None;
    }
    let roots = known_roots(arena, g, t);
    if roots.is_empty() || roots.iter().any(|r| contains_sym(arena, r.value, t_sym)) {
        return None;
    }
    let mut families: Vec<ExprId> = Vec::new();
    let mut constants: Option<Vec<ExprId>> = None;
    for r in roots {
        let neg = arena.neg(r.value);
        let branch = arena.add(&[chain[n], neg]);
        let branch = crate::transforms::eval::eval(arena, branch);
        let Some(res) = dsolve(arena, branch, func, var) else {
            continue;
        };
        if contains_sym(arena, res.solution, func_sym)
            || crate::base::walk::has_unevaluated(arena, res.solution)
            || res.constants.len() != n
        {
            continue;
        }
        match &constants {
            Some(cs) if *cs != res.constants => continue,
            Some(_) => {}
            None => constants = Some(res.constants.clone()),
        }
        for f in std::iter::once(res.solution).chain(res.branches) {
            if !families.contains(&f) && families.len() < MAX_BRANCHES {
                families.push(f);
            }
        }
    }
    let constants = constants?;
    if families.is_empty() {
        return None;
    }
    let solution = families.remove(0);
    Some(OdeResult {
        solution,
        constants,
        branches: families,
    })
}

/// `y′ = rhs` for a first-order `expr = M + N·y′ = 0` (`N ≠ 0`, `M` and `N`
/// free of `y′`): `rhs = −M/N`.
fn first_order_rhs(arena: &mut Arena, expr: ExprId, func: ExprId, var: ExprId) -> Option<ExprId> {
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));
    if expr_contains(arena, expr, d2y_dx2) {
        return None;
    }
    let (m, n) = extract_m_n(arena, expr, func, var)?;
    if expr_contains(arena, m, dy_dx) || expr_contains(arena, n, dy_dx) {
        return None;
    }
    let n = crate::transforms::eval::eval(arena, n);
    if arena.is_zero_structural(n) {
        return None;
    }
    let neg_m = arena.neg(m);
    let rhs = arena.div(neg_m, n);
    Some(crate::transforms::eval::eval(arena, rhs))
}

/// The affine forms `a·x + b·y + c` (rational `a`, `b`, not both zero;
/// `c` free of `x` and `y`) among the sums in `e`, without repeats.
fn affine_forms(
    arena: &mut Arena,
    e: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Vec<(Q, Q, ExprId)> {
    use num_traits::Zero;
    let mut out: Vec<(Q, Q, ExprId)> = Vec::new();
    let post = crate::base::walk::post_order_ids(arena, e);
    for id in post {
        let ExprNode::Add(terms) = arena.node(id).clone() else {
            continue;
        };
        let (mut a, mut b) = (Q::zero(), Q::zero());
        let mut rest = Vec::new();
        let mut affine = true;
        for &t in terms.iter() {
            let (c, term) = arena.as_coeff_term(t);
            if term == var {
                a += c;
            } else if term == func {
                b += c;
            } else if contains_sym(arena, t, func_sym) || contains_sym(arena, t, var_sym) {
                affine = false;
                break;
            } else {
                rest.push(t);
            }
        }
        if !affine || (a.is_zero() && b.is_zero()) {
            continue;
        }
        let c = match rest.len() {
            0 => arena.zero,
            1 => rest[0],
            _ => arena.add(&rest),
        };
        let c = crate::transforms::eval::eval(arena, c);
        if !out
            .iter()
            .any(|(a2, b2, c2)| *a2 == a && *b2 == b && *c2 == c)
        {
            out.push((a, b, c));
        }
    }
    out
}

/// Solve `y′ = F(a·x + b·y + c)` (`a, b ≠ 0` rational): `u = a·x + b·y + c`
/// gives the autonomous `u′ = a + b·F(u)`, separable; each family of `u`
/// gives `y = (u − a·x − c)/b` (an implicit one `G(x, a·x + b·y + c) = 0`).
/// SymPy's `dsolve` reaches `y′ = (y − x)²` through its rational Riccati
/// solver (`(C1·x + C1 − x·e^{2x} + e^{2x})/(C1 − e^{2x})`, the same
/// family) and `y′ = (y − x + 2)/(y − x + 3)` as `x − 3 ± √(C1 − 2x)`;
/// both were refused.
fn try_linear_argument(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    use num_traits::Zero;
    let rhs = first_order_rhs(arena, expr, func, var)?;
    if !contains_sym(arena, rhs, func_sym) || !contains_sym(arena, rhs, var_sym) {
        return None;
    }
    let u = arena.symbol("__u_lin");
    let ExprNode::Symbol(u_sym) = *arena.node(u) else {
        return None;
    };
    for (a, b, c) in affine_forms(arena, rhs, func, var, func_sym, var_sym) {
        if a.is_zero() || b.is_zero() {
            continue;
        }
        // y = (u − a·x − c)/b
        let a_e = ode_ratio_to_expr(arena, &a);
        let b_e = ode_ratio_to_expr(arena, &b);
        let ax = arena.mul(&[a_e, var]);
        let neg_ax = arena.neg(ax);
        let neg_c = arena.neg(c);
        let num = arena.add(&[u, neg_ax, neg_c]);
        let y_of_u = arena.div(num, b_e);
        let y_of_u = crate::transforms::eval::eval(arena, y_of_u);
        let g = crate::transforms::subs::subs(arena, rhs, func, y_of_u);
        let mut g = crate::transforms::eval::eval(arena, g);
        if contains_sym(arena, g, var_sym) {
            g = crate::simplify::ratsimp::ratsimp(arena, g);
            g = crate::transforms::eval::eval(arena, g);
        }
        if contains_sym(arena, g, var_sym) || !contains_sym(arena, g, u_sym) {
            continue;
        }
        // u′ − (a + b·G(u)) = 0
        let du = arena.intern(ExprNode::Derivative(u, var));
        let bg = arena.mul(&[b_e, g]);
        let rate = arena.add(&[a_e, bg]);
        let neg_rate = arena.neg(rate);
        let u_ode = arena.add(&[du, neg_rate]);
        let u_ode = crate::transforms::eval::eval(arena, u_ode);
        let Some(ur) = dsolve(arena, u_ode, u, var) else {
            continue;
        };
        if ur.constants.len() != 1 || crate::base::walk::has_unevaluated(arena, ur.solution) {
            continue;
        }
        // u = a·x + b·y + c
        let by = arena.mul(&[b_e, func]);
        let u_of_y = arena.add(&[ax, by, c]);
        let back = |arena: &mut Arena, fam: ExprId| -> ExprId {
            let r = if contains_sym(arena, fam, u_sym) {
                crate::transforms::subs::subs(arena, fam, u, u_of_y)
            } else {
                crate::transforms::subs::subs(arena, y_of_u, u, fam)
            };
            crate::transforms::eval::eval(arena, r)
        };
        let solution = back(arena, ur.solution);
        let branches = ur.branches.iter().map(|&f| back(arena, f)).collect();
        return Some(OdeResult {
            solution,
            constants: ur.constants,
            branches,
        });
    }
    None
}

/// Solve `y′ = F(x, y)` whose affine forms `aᵢ·x + bᵢ·y + cᵢ` (rational)
/// all vanish at one point `(h, k) ≠ (0, 0)` (two of them independent):
/// `x = X + h`, `y = Y + k` makes them linear and the equation homogeneous
/// (SymPy's `linear_coefficients`): `y′ = (x + y + 1)/(x − y + 3)` (lines
/// through `(−2, 1)`) was refused.  The families of `Y(X)` come back as
/// `y = Y(x − h) + k`, an implicit one as `G(x − h, y − k) = 0`.
fn try_linear_coefficients(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    use num_traits::Zero;
    let rhs = first_order_rhs(arena, expr, func, var)?;
    if !contains_sym(arena, rhs, func_sym) || !contains_sym(arena, rhs, var_sym) {
        return None;
    }
    let forms = affine_forms(arena, rhs, func, var, func_sym, var_sym);
    let mut rational: Vec<(Q, Q, Q)> = Vec::with_capacity(forms.len());
    for (a, b, c) in &forms {
        rational.push((a.clone(), b.clone(), arena.as_num(*c)?.clone()));
    }
    if rational.iter().all(|(_, _, c)| c.is_zero()) {
        return None;
    }
    // Two independent forms fix (h, k).
    let (i, j) = (0..rational.len())
        .flat_map(|i| (i + 1..rational.len()).map(move |j| (i, j)))
        .find(|&(i, j)| {
            let (a1, b1, _) = &rational[i];
            let (a2, b2, _) = &rational[j];
            !(a1 * b2 - a2 * b1).is_zero()
        })?;
    let (a1, b1, c1) = &rational[i];
    let (a2, b2, c2) = &rational[j];
    let det = a1 * b2 - a2 * b1;
    let h = (b1 * c2 - b2 * c1) / &det;
    let k = (a2 * c1 - a1 * c2) / &det;
    if h.is_zero() && k.is_zero() {
        return None;
    }
    if rational
        .iter()
        .any(|(a, b, c)| !(a * &h + b * &k + c).is_zero())
    {
        return None;
    }
    let h_e = ode_ratio_to_expr(arena, &h);
    let k_e = ode_ratio_to_expr(arena, &k);
    let big_y = arena.symbol("__Y_lin");
    let ExprNode::Symbol(big_y_sym) = *arena.node(big_y) else {
        return None;
    };
    // Y′ = rhs(x + h, Y + k)
    let x_shift = arena.add(&[var, h_e]);
    let y_shift = arena.add(&[big_y, k_e]);
    let shifted = crate::transforms::subs::subs(arena, rhs, func, y_shift);
    let shifted = crate::transforms::subs::subs(arena, shifted, var, x_shift);
    let shifted = crate::transforms::eval::eval(arena, shifted);
    let shifted = crate::transforms::expand::expand(arena, shifted);
    let shifted = crate::transforms::eval::eval(arena, shifted);
    let dy = arena.intern(ExprNode::Derivative(big_y, var));
    let neg = arena.neg(shifted);
    let y_ode = arena.add(&[dy, neg]);
    let y_ode = crate::transforms::eval::eval(arena, y_ode);
    let yr = dsolve(arena, y_ode, big_y, var)?;
    if yr.constants.len() != 1 || crate::base::walk::has_unevaluated(arena, yr.solution) {
        return None;
    }
    let neg_h = arena.neg(h_e);
    let x_back = arena.add(&[var, neg_h]);
    let neg_k = arena.neg(k_e);
    let y_back = arena.add(&[func, neg_k]);
    let back = |arena: &mut Arena, fam: ExprId| -> ExprId {
        let r = if contains_sym(arena, fam, big_y_sym) {
            let r = crate::transforms::subs::subs(arena, fam, big_y, y_back);
            crate::transforms::subs::subs(arena, r, var, x_back)
        } else {
            let r = crate::transforms::subs::subs(arena, fam, var, x_back);
            arena.add(&[r, k_e])
        };
        crate::transforms::eval::eval(arena, r)
    };
    let solution = back(arena, yr.solution);
    let branches = yr.branches.iter().map(|&f| back(arena, f)).collect();
    Some(OdeResult {
        solution,
        constants: yr.constants,
        branches,
    })
}

/// Solve y' = f(x) (simplest separable: no y dependence).
fn try_simple_separable(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<OdeResult> {
    // Pattern: Derivative(y, x) + f(x) = 0, or Derivative(y, x) - f(x) = 0
    // Rearranges to: y' = f(x), then y = ∫ f(x) dx + C

    // Look for Derivative(func, var) in the expression
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));

    // Check if expr is Add containing dy_dx
    if let ExprNode::Add(ref children) = arena.node(expr).clone() {
        // a(x)·y′ + f(x) = 0 with a(x) free of y (before, only a = 1:
        // `2y′ = e⁻ˣ` and `(1 + x²)y′ = x²` were not solved at all).
        let mut deriv_coeffs: Vec<ExprId> = Vec::new();
        let mut other_terms: Vec<ExprId> = Vec::new();

        for &child in children {
            let (coeff, term) = arena.as_coeff_term(child);
            if term == dy_dx {
                deriv_coeffs.push(ode_ratio_to_expr(arena, &coeff));
            } else if !contains_sym(arena, child, func_sym) {
                other_terms.push(child);
            } else if let ExprNode::Mul(factors) = arena.node(term).clone()
                && factors.iter().filter(|&&f| f == dy_dx).count() == 1
                && factors
                    .iter()
                    .all(|&f| f == dy_dx || !contains_sym(arena, f, func_sym))
            {
                let mut rest: Vec<ExprId> =
                    factors.iter().copied().filter(|&f| f != dy_dx).collect();
                rest.push(ode_ratio_to_expr(arena, &coeff));
                deriv_coeffs.push(arena.mul(&rest));
            } else {
                // Term contains y — not simple separable
                return None;
            }
        }

        if !deriv_coeffs.is_empty() {
            let a = arena.add(&deriv_coeffs);
            let a = crate::transforms::eval::eval(arena, a);
            if arena.is_zero_structural(a) {
                return None;
            }
            // a·y' + other_terms = 0 → y' = -other_terms/a → y = -∫ other_terms/a dx + C
            let rhs = if other_terms.is_empty() {
                arena.zero
            } else {
                let sum = arena.add(&other_terms);
                let neg = arena.neg(sum);
                let q = arena.div(neg, a);
                crate::transforms::eval::eval(arena, q)
            };
            let integral = integrate_forms(arena, rhs, var);
            let c1 = arena.symbol("C1");
            let solution = arena.add(&[integral, c1]);
            return Some(OdeResult {
                solution,
                constants: vec![c1],
                branches: Vec::new(),
            });
        }
    }

    // Also handle: Derivative(y, x) = expr pattern (if expr is a single Derivative)
    if expr == dy_dx {
        // y' = 0 → y = C
        let c1 = arena.symbol("C1");
        return Some(OdeResult {
            solution: c1,
            constants: vec![c1],
            branches: Vec::new(),
        });
    }

    None
}

/// Solve y'' + b*y' + c*y = 0 via characteristic equation.
fn try_second_order_const_coeff(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    _func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<OdeResult> {
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    // Pattern: a*y'' + b*y' + c*y = 0
    // Extract coefficients of y'', y', and y
    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut a_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut b_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut c_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == d2y_dx2 {
            a_coeff += coeff;
        } else if term == dy_dx {
            b_coeff += coeff;
        } else if term == func {
            c_coeff += coeff;
        } else {
            return None; // Contains non-homogeneous or non-constant-coefficient terms
        }
    }

    use num_traits::Zero;
    if a_coeff.is_zero() {
        return None; // Not second order
    }

    // Normalize: divide by a
    let b = &b_coeff / &a_coeff;
    let c = &c_coeff / &a_coeff;

    // Check for complex roots: if disc = b² − 4c < 0, use Euler/trig form
    {
        let four_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(4.into());
        let disc = &b * &b - &four_r * &c;
        if disc.is_negative() {
            return build_trig_homogeneous_solution(arena, &b, &disc, var);
        }
    }

    // Characteristic equation: r² + b*r + c = 0
    let r_var = arena.symbol("__r");
    let two = arena.int(2);
    let b_id = {
        let nid = arena.intern_num(b.clone());
        arena.intern(ExprNode::Num(nid))
    };
    let c_id = {
        let nid = arena.intern_num(c.clone());
        arena.intern(ExprNode::Num(nid))
    };

    let r_var_sq = arena.pow(r_var, two);
    let b_r = arena.mul(&[b_id, r_var]);
    let char_eq = arena.add(&[r_var_sq, b_r, c_id]);
    let roots = known_roots(arena, char_eq, r_var);

    let c1 = arena.symbol("C1");
    let c2 = arena.symbol("C2");

    match roots.len() {
        2 => {
            let r1 = roots[0].value;
            let r2 = roots[1].value;
            if r1 == r2 {
                // Repeated root: y = (C1 + C2*x) * e^(r*x)
                let rx = arena.mul(&[r1, var]);
                let exp_rx = arena.exp(rx);
                let c2_x = arena.mul(&[c2, var]);
                let inner = arena.add(&[c1, c2_x]);
                let solution = arena.mul(&[inner, exp_rx]);
                Some(OdeResult {
                    solution,
                    constants: vec![c1, c2],
                    branches: Vec::new(),
                })
            } else {
                // Distinct roots: y = C1*e^(r1*x) + C2*e^(r2*x)
                let r1x = arena.mul(&[r1, var]);
                let r2x = arena.mul(&[r2, var]);
                let exp_r1x = arena.exp(r1x);
                let exp_r2x = arena.exp(r2x);
                let term1 = arena.mul(&[c1, exp_r1x]);
                let term2 = arena.mul(&[c2, exp_r2x]);
                let solution = arena.add(&[term1, term2]);
                Some(OdeResult {
                    solution,
                    constants: vec![c1, c2],
                    branches: Vec::new(),
                })
            }
        }
        1 => {
            // Single root (shouldn't happen for quadratic, but handle)
            let r = roots[0].value;
            let rx = arena.mul(&[r, var]);
            let exp_rx = arena.exp(rx);
            let c2_x = arena.mul(&[c2, var]);
            let inner = arena.add(&[c1, c2_x]);
            let solution = arena.mul(&[inner, exp_rx]);
            Some(OdeResult {
                solution,
                constants: vec![c1, c2],
                branches: Vec::new(),
            })
        }
        _ => None,
    }
}

/// Solve the characteristic equation r² + b·r + c = 0 and construct the
/// homogeneous solution of y'' + b·y' + c·y = 0.
///
/// This is extracted as a helper so that both the homogeneous and
/// nonhomogeneous second-order solvers can reuse it.
fn solve_characteristic_equation(arena: &mut Arena, b: Q, c: Q, var: ExprId) -> Option<OdeResult> {
    // Check for complex roots: if disc = b² − 4c < 0, use Euler/trig form
    {
        let four_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(4.into());
        let disc = &b * &b - &four_r * &c;
        if disc.is_negative() {
            return build_trig_homogeneous_solution(arena, &b, &disc, var);
        }
    }

    let r_var = arena.symbol("__r");
    let two = arena.int(2);
    let b_id = {
        let nid = arena.intern_num(b);
        arena.intern(ExprNode::Num(nid))
    };
    let c_id = {
        let nid = arena.intern_num(c);
        arena.intern(ExprNode::Num(nid))
    };

    let r_var_sq = arena.pow(r_var, two);
    let b_r = arena.mul(&[b_id, r_var]);
    let char_eq = arena.add(&[r_var_sq, b_r, c_id]);
    let roots = known_roots(arena, char_eq, r_var);

    let c1 = arena.symbol("C1");
    let c2 = arena.symbol("C2");

    match roots.len() {
        2 => {
            let r1 = roots[0].value;
            let r2 = roots[1].value;
            if r1 == r2 {
                // Repeated root: y = (C1 + C2*x) * e^(r*x)
                let rx = arena.mul(&[r1, var]);
                let exp_rx = arena.exp(rx);
                let c2_x = arena.mul(&[c2, var]);
                let inner = arena.add(&[c1, c2_x]);
                let solution = arena.mul(&[inner, exp_rx]);
                Some(OdeResult {
                    solution,
                    constants: vec![c1, c2],
                    branches: Vec::new(),
                })
            } else {
                // Distinct roots: y = C1*e^(r1*x) + C2*e^(r2*x)
                let r1x = arena.mul(&[r1, var]);
                let r2x = arena.mul(&[r2, var]);
                let exp_r1x = arena.exp(r1x);
                let exp_r2x = arena.exp(r2x);
                let term1 = arena.mul(&[c1, exp_r1x]);
                let term2 = arena.mul(&[c2, exp_r2x]);
                let solution = arena.add(&[term1, term2]);
                Some(OdeResult {
                    solution,
                    constants: vec![c1, c2],
                    branches: Vec::new(),
                })
            }
        }
        1 => {
            // Single root (treat as repeated)
            let r = roots[0].value;
            let rx = arena.mul(&[r, var]);
            let exp_rx = arena.exp(rx);
            let c2_x = arena.mul(&[c2, var]);
            let inner = arena.add(&[c1, c2_x]);
            let solution = arena.mul(&[inner, exp_rx]);
            Some(OdeResult {
                solution,
                constants: vec![c1, c2],
                branches: Vec::new(),
            })
        }
        _ => None,
    }
}

/// Solve a·y'' + b·y' + c·y = f(x) where f(x) is a polynomial, via
/// the method of undetermined coefficients.
///
/// Returns `None` when:
/// - The expression is not an `Add` node.
/// - The expression is not second-order (no y'' term).
/// - The expression is homogeneous (no forcing terms) — let the
///   dedicated homogeneous solver handle it.
/// - The forcing term is not polynomial in `var`.
/// - The characteristic equation cannot be solved.
fn try_second_order_cc_nonhomogeneous(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<OdeResult> {
    use num_traits::Zero;

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut a_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut b_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut c_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut f_of_x_terms: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == d2y_dx2 {
            a_coeff += coeff;
        } else if term == dy_dx {
            b_coeff += coeff;
        } else if term == func {
            c_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            // Term free of y and its derivatives — forcing f(x)
            f_of_x_terms.push(child);
        } else {
            return None; // nonlinear or variable-coefficient term
        }
    }

    if a_coeff.is_zero() {
        return None; // Not second order
    }

    if f_of_x_terms.is_empty() {
        return None; // Homogeneous — handled by try_second_order_const_coeff
    }

    // Normalise: divide through by a so the leading coefficient is 1.
    let b = &b_coeff / &a_coeff;
    let c = &c_coeff / &a_coeff;

    // Build g(x) = sum of the forcing terms.  The ODE reads
    //   y'' + b·y' + c·y + g(x)/a = 0   ⟹   rhs = −g(x)/a
    let f_sum = if f_of_x_terms.len() == 1 {
        f_of_x_terms[0]
    } else {
        arena.add(&f_of_x_terms)
    };

    // Try polynomial forcing first
    let y_p_poly = if let Some(f_coeffs_expr) = arena.coefficients_of(f_sum, var) {
        if f_coeffs_expr.is_empty() {
            None
        } else {
            let mut rhs_coeffs: Vec<Q> = Vec::new();
            let mut all_numeric = true;
            for &cid in &f_coeffs_expr {
                if let Some(val) = arena.as_num(cid) {
                    rhs_coeffs.push(-val.clone() / &a_coeff);
                } else {
                    all_numeric = false;
                    break;
                }
            }
            if all_numeric {
                find_particular_polynomial(&b, &c, &rhs_coeffs)
                    .map(|pc| build_polynomial_expr(arena, &pc, var))
            } else {
                None
            }
        }
    } else {
        None
    };

    // If polynomial forcing didn't work, try trig/exp forcing
    let y_p = if let Some(yp) = y_p_poly {
        yp
    } else {
        // Compute rhs = -f_sum / a for trig/exp analysis
        let neg_f_sum = arena.neg(f_sum);
        let a_id = ode_ratio_to_expr(arena, &a_coeff);
        let rhs_expr = arena.div(neg_f_sum, a_id);
        let rhs_expr = crate::transforms::eval::eval(arena, rhs_expr);
        try_undetermined_trig_exp(arena, rhs_expr, &b, &c, var)?
    };

    // Homogeneous solution via the characteristic equation
    let homo_result = solve_characteristic_equation(arena, b, c, var)?;

    // General solution = homogeneous + particular
    let solution = arena.add(&[homo_result.solution, y_p]);

    Some(OdeResult {
        solution,
        constants: homo_result.constants,
        branches: Vec::new(),
    })
}

/// Solve the undetermined-coefficients linear system for a polynomial
/// particular solution of  y'' + b·y' + c·y = rhs(x).
///
/// `rhs_coeffs[j]` is the coefficient of x^j on the right-hand side,
/// in ascending degree order.
///
/// Returns ascending-order coefficients of y_p, or `None` on failure.
fn find_particular_polynomial(b: &Q, c: &Q, rhs_coeffs: &[Q]) -> Option<Vec<Q>> {
    use num_bigint::BigInt;
    use num_rational::Ratio;
    use num_traits::Zero;

    let n = rhs_coeffs.len() - 1;

    if !c.is_zero() {
        // ── Case 1: c ≠ 0 ──────────────────────────────────────────────
        // y_p = A_0 + A_1·x + … + A_n·x^n   (same degree as rhs)
        //
        // Matching x^j (j = n … 0):
        //   (j+2)(j+1)·A_{j+2} + b·(j+1)·A_{j+1} + c·A_j = r_j
        //
        // Triangular system, solved top-down.
        let mut a = vec![Ratio::<BigInt>::zero(); n + 1];
        for j in (0..=n).rev() {
            let a_j2 = if j + 2 <= n {
                a[j + 2].clone()
            } else {
                Ratio::zero()
            };
            let a_j1 = if j < n {
                a[j + 1].clone()
            } else {
                Ratio::zero()
            };
            let factor2 = Ratio::from_integer(BigInt::from(((j + 2) * (j + 1)) as i64)) * a_j2;
            let factor1 = Ratio::from_integer(BigInt::from((j + 1) as i64)) * b.clone() * a_j1;
            a[j] = (rhs_coeffs[j].clone() - factor2 - factor1) / c.clone();
        }
        Some(a)
    } else if !b.is_zero() {
        // ── Case 2: c = 0, b ≠ 0 ───────────────────────────────────────
        // Multiply trial by x:  y_p = B_0·x + B_1·x² + … + B_n·x^{n+1}
        //
        // Matching x^j (j = n … 0):
        //   [(j+2)(j+1)·B_{j+1} if j < n] + b·(j+1)·B_j = r_j
        let mut bb = vec![Ratio::<BigInt>::zero(); n + 1];
        for j in (0..=n).rev() {
            let deriv_term = if j < n {
                Ratio::from_integer(BigInt::from(((j + 2) * (j + 1)) as i64)) * bb[j + 1].clone()
            } else {
                Ratio::zero()
            };
            let denom = Ratio::from_integer(BigInt::from((j + 1) as i64)) * b.clone();
            if denom.is_zero() {
                return None;
            }
            bb[j] = (rhs_coeffs[j].clone() - deriv_term) / denom;
        }
        // Shift: x·(B_0 + B_1·x + …) → coefficients [0, B_0, B_1, …]
        let mut result = vec![Ratio::zero()];
        result.extend(bb);
        Some(result)
    } else {
        // ── Case 3: c = 0, b = 0 ───────────────────────────────────────
        // y'' = rhs  ⟹  y_p = ∫∫ rhs dx dx
        // Coefficient of x^{j+2} = r_j / ((j+1)(j+2))
        let mut result = vec![Ratio::<BigInt>::zero(); 2];
        for (j, r_j) in rhs_coeffs.iter().enumerate() {
            let denom = Ratio::from_integer(BigInt::from(((j + 1) * (j + 2)) as i64));
            result.push(r_j.clone() / denom);
        }
        Some(result)
    }
}

/// Build an arena polynomial expression from ascending-order rational
/// coefficients: `coeffs[j]` is the coefficient of `var^j`.
fn build_polynomial_expr(arena: &mut Arena, coeffs: &[Q], var: ExprId) -> ExprId {
    use num_traits::Zero;
    let mut terms = Vec::new();
    for (j, coeff) in coeffs.iter().enumerate() {
        if coeff.is_zero() {
            continue;
        }
        let x_pow_j = if j == 0 {
            arena.one
        } else if j == 1 {
            var
        } else {
            let exp = arena.int(j as i64);
            arena.pow(var, exp)
        };
        let term = arena.make_coeff_term(coeff.clone(), x_pow_j);
        terms.push(term);
    }
    if terms.is_empty() {
        arena.zero
    } else if terms.len() == 1 {
        terms[0]
    } else {
        arena.add(&terms)
    }
}

/// Solve y' + a*y = f(x) (first-order linear with constant coefficient).
fn try_first_order_linear(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<OdeResult> {
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut has_dy = false;
    let mut a_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut f_of_x = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            has_dy = true;
            // coeff should be 1 for standard form
            if !coeff.is_one() {
                return None; // Non-unit coefficient on y'
            }
        } else if term == func {
            a_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            f_of_x.push(child);
        } else {
            return None; // Contains y in a non-linear way
        }
    }

    use num_traits::Zero;
    if !has_dy {
        return None;
    }

    // y' + a*y + f(x) = 0 → y' + a*y = -f(x)
    // Solution: y = e^(-ax) * (∫ -f(x)*e^(ax) dx + C)

    let a_id = {
        let nid = arena.intern_num(a_coeff.clone());
        arena.intern(ExprNode::Num(nid))
    };
    let neg_a = arena.neg(a_id);

    // Build e^(ax) and e^(-ax)
    let ax = arena.mul(&[a_id, var]);
    let neg_ax = arena.mul(&[neg_a, var]);
    let exp_ax = arena.exp(ax);
    let exp_neg_ax = arena.exp(neg_ax);

    if a_coeff.is_zero() {
        // y' = -f(x) → y = -∫ f(x) dx + C
        let f = if f_of_x.is_empty() {
            arena.zero
        } else {
            let sum = arena.add(&f_of_x);
            arena.neg(sum)
        };
        let integral = crate::transforms::integrate::integrate(arena, f, var);
        let c1 = arena.symbol("C1");
        let solution = arena.add(&[integral, c1]);
        return Some(OdeResult {
            solution,
            constants: vec![c1],
            branches: Vec::new(),
        });
    }

    // General case: y = e^(-ax) * (∫ (-f(x))*e^(ax) dx + C1)
    let neg_f = if f_of_x.is_empty() {
        arena.zero
    } else {
        let sum = arena.add(&f_of_x);
        arena.neg(sum)
    };
    // Simplify exp(a)*exp(b) → exp(a+b) before integrating
    let integrand = if let ExprNode::Exp(neg_f_inner) = arena.node(neg_f).clone() {
        let combined_arg = arena.add(&[neg_f_inner, ax]);
        let combined_arg = crate::transforms::eval::eval(arena, combined_arg);
        arena.exp(combined_arg)
    } else {
        arena.mul(&[neg_f, exp_ax])
    };
    let integrand = crate::transforms::eval::eval(arena, integrand);
    let integral = integrate_forms(arena, integrand, var);

    let c1 = arena.symbol("C1");
    let inner = arena.add(&[integral, c1]);
    let solution = arena.mul(&[exp_neg_ax, inner]);

    Some(OdeResult {
        solution,
        constants: vec![c1],
        branches: Vec::new(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Full separable: y' = f(x) * g(y)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve y' = f(x)*g(y) via separation of variables.
///
/// Algorithm:
/// 1. Extract dy/dx from the ODE expression
/// 2. Collect remaining terms as RHS (negated)
/// 3. If RHS factors into x-only × y-only parts, separate them
/// 4. For the common case g(y) = y: solution is y = C1·exp(∫f(x)dx)
/// 5. Otherwise: implicit solution ∫(1/g(y))dy = ∫f(x)dx + C1
fn try_full_separable(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying full separable");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));

    // Extract dy/dx term and collect the rest as RHS
    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut has_dy = false;
    let mut dy_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut other_terms: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            has_dy = true;
            dy_coeff += coeff;
        } else {
            other_terms.push(child);
        }
    }

    use num_traits::Zero;
    if !has_dy || dy_coeff.is_zero() {
        return None;
    }

    // We need coefficient on y' to be 1 (or normalize)
    if !dy_coeff.is_one() {
        // Normalize: divide everything else by dy_coeff
        let inv_coeff = num_rational::Ratio::<num_bigint::BigInt>::one() / &dy_coeff;
        let inv_id = {
            let nid = arena.intern_num(inv_coeff);
            arena.intern(ExprNode::Num(nid))
        };
        let mut scaled = Vec::new();
        for &t in &other_terms {
            scaled.push(arena.mul(&[inv_id, t]));
        }
        other_terms = scaled;
    }

    // RHS = -(other_terms), i.e., y' = -other_terms
    let rhs = if other_terms.is_empty() {
        return None; // y' = 0 is handled by simple separable
    } else {
        let sum = arena.add(&other_terms);
        arena.neg(sum)
    };

    // Now we need: rhs = f(x) * g(y) where f depends only on var and g only on func
    // Check if rhs depends on func at all — if not, this is simple separable
    if !contains_sym(arena, rhs, func_sym) {
        return None; // Let simple separable handle it
    }

    // Try to factor rhs into x-only and y-only parts; `exp(a + b)` is
    // `exp(a)·exp(b)` (`y′ = e^{x+y}` was refused; SymPy gives
    // `−ln(C1 − eˣ)`).
    let mut factors = Vec::new();
    for f in collect_mul_factors(arena, rhs) {
        match arena.node(f).clone() {
            ExprNode::Exp(arg) if matches!(arena.node(arg), ExprNode::Add(_)) => {
                let ExprNode::Add(terms) = arena.node(arg).clone() else {
                    continue;
                };
                for &t in terms.iter() {
                    factors.push(arena.exp(t));
                }
            }
            _ => factors.push(f),
        }
    }

    let mut x_factors: Vec<ExprId> = Vec::new();
    let mut y_factors: Vec<ExprId> = Vec::new();

    for &factor in &factors {
        let has_x = contains_sym(arena, factor, var_sym);
        let has_y = contains_sym(arena, factor, func_sym);

        if has_x && has_y {
            // Factor depends on both x and y — cannot separate
            return None;
        } else if has_y {
            y_factors.push(factor);
        } else {
            // Pure x-factor or constant
            x_factors.push(factor);
        }
    }

    if y_factors.is_empty() {
        return None; // No y dependence — let simple separable handle it
    }

    // f(x) = product of x_factors
    let f_x = if x_factors.is_empty() {
        arena.one
    } else if x_factors.len() == 1 {
        x_factors[0]
    } else {
        arena.mul(&x_factors)
    };

    // g(y) = product of y_factors
    let g_y = if y_factors.len() == 1 {
        y_factors[0]
    } else {
        arena.mul(&y_factors)
    };

    // g(y) = k·y: y = C1·exp(k·∫f(x)dx).  Before, `exp(k·∫f dx + C1)`,
    // which is never zero or negative: the solutions y ≤ 0 were missing.
    {
        let (coeff, base) = arena.as_coeff_term(g_y);
        if base == func {
            let coeff_id = ode_ratio_to_expr(arena, &coeff);
            let scaled_fx = arena.mul(&[coeff_id, f_x]);
            let integral_fx = crate::transforms::integrate::integrate(arena, scaled_fx, var);
            let c1 = arena.symbol("C1");
            let growth = arena.exp(integral_fx);
            let solution = arena.mul(&[c1, growth]);
            return Some(OdeResult {
                solution,
                constants: vec![c1],
                branches: Vec::new(),
            });
        }
    }

    // General case: ∫ dy/g(y) = ∫ f(x) dx + C1, solved for y (every
    // explicit root is a family: `solution` and `branches`), else the
    // implicit `∫ dy/g(y) − ∫ f(x) dx − C1` (= 0).
    // Before, `∫ dy/g(y)` was never attempted (a formal `Integral` was
    // returned even for `y′ = 1 + y²`).
    let neg_one = arena.int(-1);
    let inv_gy = arena.pow(g_y, neg_one);
    let inv_gy = crate::transforms::eval::eval(arena, inv_gy);
    let lhs_attempt = integrate_forms(arena, inv_gy, func);
    let lhs_integral = if integral_failed(arena, lhs_attempt) {
        arena.intern(ExprNode::Integral(inv_gy, func))
    } else {
        lhs_attempt
    };
    let rhs_integral = crate::transforms::integrate::integrate(arena, f_x, var);
    let c1 = arena.symbol("C1");
    let neg_rhs = arena.neg(rhs_integral);
    let neg_c1 = arena.neg(c1);
    let implicit = arena.add(&[lhs_integral, neg_rhs, neg_c1]);
    let implicit = crate::transforms::eval::eval(arena, implicit);
    if !integral_failed(arena, implicit)
        && let Some((solution, branches)) = explicit_families(arena, expr, implicit, func, var, c1)
    {
        return Some(OdeResult {
            solution,
            constants: vec![c1],
            branches,
        });
    }
    Some(OdeResult {
        solution: implicit,
        constants: vec![c1],
        branches: Vec::new(),
    })
}

/// The families `y = …` of the implicit solution `implicit = 0` of the ODE
/// `expr = 0` (one arbitrary constant `c1`): `(first, others)`, or `None`.
///
/// 1. Every root of `implicit` in `func` (at most [`MAX_BRANCHES`], each
///    free of `func`) is a family: `y·y′ = 1` has `y²/2 − x = C1`, so
///    `y = ±√(2C1 + 2x)` (before, an exact or reduced equation whose
///    potential had two roots was left implicit).
/// 2. Failing that, with `ln|u|` replaced by `ln u` and the logarithms'
///    coefficients made integers (`ln|y − 1|/2 − ln|y + 1|/2 = x + C1`
///    as `ln(y − 1) − ln(y + 1) = 2x + 2C1`): on each interval where the
///    sign of `u` is fixed, `ln u − ln|u|` is a constant, absorbed by the
///    arbitrary constant, so each root still solves the ODE (SymPy's
///    `dsolve` integrates without `|·|` likewise).  An exponential
///    `e^{k·C1}` in every occurrence of the constant is then renamed `C1`
///    (its values are the non-zero numbers; `C1 = 0` is the limit family,
///    a solution too where defined): `y′ = y² − 1` gives
///    `(1 + C1·e^{2x})/(1 − C1·e^{2x})` (SymPy: `1/tanh(C1 − x)`; before,
///    the implicit `ln|y − 1|/2 − ln|y + 1|/2 − x − C1`).  Such families
///    are kept only if they verify ([`family_verifies`]).
fn explicit_families(
    arena: &mut Arena,
    expr: ExprId,
    implicit: ExprId,
    func: ExprId,
    var: ExprId,
    c1: ExprId,
) -> Option<(ExprId, Vec<ExprId>)> {
    let ExprNode::Symbol(func_sym) = *arena.node(func) else {
        return None;
    };
    // Every root free of `func` that carries the constant (a root without
    // it is no member of the family: `y′ = 1 + sin y` got `y = ±π`, which
    // are not solutions, from spurious roots of `−2/(tan(y/2) + 1) = x + C1`)
    // and is not refuted numerically.
    let explicit_roots = |arena: &mut Arena, e: ExprId| -> Option<Vec<ExprId>> {
        let sols = known_roots(arena, e, func);
        if sols.is_empty()
            || sols.len() > MAX_BRANCHES
            || sols.iter().any(|s| contains_sym(arena, s.value, func_sym))
        {
            return None;
        }
        let mut fams = Vec::with_capacity(sols.len());
        for s in &sols {
            let f = crate::transforms::eval::eval(arena, s.value);
            if crate::base::walk::contains(arena, f, c1)
                && residual_verdict(arena, expr, f, func, var, c1) != Some(false)
            {
                fams.push(f);
            }
        }
        (!fams.is_empty()).then_some(fams)
    };
    let stripped_families = |arena: &mut Arena| -> Option<Vec<ExprId>> {
        let stripped = strip_log_abs(arena, implicit)?;
        let fams = explicit_roots(arena, stripped)?;
        let mut kept: Vec<ExprId> = Vec::with_capacity(fams.len());
        for f in fams {
            let f = absorb_exp_constant(arena, f, c1);
            if !kept.contains(&f) && family_verifies(arena, expr, f, func, var, c1) {
                kept.push(f);
            }
        }
        (!kept.is_empty()).then_some(kept)
    };
    let mut fams = match explicit_roots(arena, implicit) {
        // Roots through `|x|` (`√(C1·x²·|x| − x²)`, from `ln|x|`) give way to
        // those of the form without `|·|` (`±x·√(C1·x − 1)`, SymPy's).
        Some(direct) if direct.iter().any(|&f| has_abs(arena, f)) => {
            match stripped_families(arena) {
                Some(s) if !s.iter().any(|&f| has_abs(arena, f)) => s,
                _ => direct,
            }
        }
        Some(direct) => direct,
        None => stripped_families(arena)?,
    };
    // A family `e^{k·C1 + r}` misses `C1·e^{r}` for `C1 ≤ 0`.
    for f in &mut fams {
        let renamed = absorb_exp_constant(arena, *f, c1);
        if renamed != *f && family_verifies(arena, expr, renamed, func, var, c1) {
            *f = renamed;
        }
        *f = exp_log_to_power(arena, *f);
        // `W(C1)/x` is the family `C1/x`.
        if let Some(renamed) = rename_constant_function(arena, *f, var, c1)
            && family_verifies(arena, expr, renamed, func, var, c1)
        {
            *f = renamed;
        }
    }
    // Families equal up to `C1 → −C1` are one (`y′ = tan y`:
    // `asin(C1·eˣ)` and `asin(−C1·eˣ)`).
    let neg_c1 = arena.neg(c1);
    let mut unique: Vec<ExprId> = Vec::with_capacity(fams.len());
    for f in fams {
        let mirrored = crate::transforms::subs::subs(arena, f, c1, neg_c1);
        let mirrored = crate::transforms::eval::eval(arena, mirrored);
        if !unique.contains(&f) && !unique.contains(&mirrored) {
            unique.push(f);
        }
    }
    let first = unique.remove(0);
    Some((first, unique))
}

/// `e` with `e^{k·ln u + r} → u^k·e^{r}` for rational `k` (`u^k` is
/// `e^{k·Log u}` by definition of the principal power): the renaming of
/// [`absorb_exp_constant`] leaves `W(C1·x·e^{−ln x})` for `y′ = −(1/x +
/// y)/(x + 1/y)`, whose initial-value fit then failed.
fn exp_log_to_power(arena: &mut Arena, e: ExprId) -> ExprId {
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    let mut changed = false;
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match *arena.node(rebuilt) {
            ExprNode::Exp(arg) => {
                let terms: Vec<ExprId> = match arena.node(arg) {
                    ExprNode::Add(c) => c.to_vec(),
                    _ => vec![arg],
                };
                let mut powers = Vec::new();
                let mut rest = Vec::new();
                for t in terms {
                    let (k, term) = arena.as_coeff_term(t);
                    match *arena.node(term) {
                        ExprNode::Ln(u) => {
                            let ke = ode_ratio_to_expr(arena, &k);
                            powers.push(arena.pow(u, ke));
                        }
                        _ => rest.push(t),
                    }
                }
                if powers.is_empty() {
                    rebuilt
                } else {
                    changed = true;
                    if !rest.is_empty() {
                        let r = arena.add(&rest);
                        powers.push(arena.exp(r));
                    }
                    arena.mul(&powers)
                }
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    if !changed {
        return e;
    }
    let out = cache.get(&e).copied().unwrap_or(e);
    crate::transforms::eval::eval(arena, out)
}

/// `e` with its one subexpression `g(c1)` that contains `c1` but not `var`
/// (every occurrence of `c1` inside it, `g(c1) ≠ c1`) replaced by `c1`;
/// `None` when there is no such single subexpression.  The new constant
/// ranges over the values of `g`, and beyond them by analytic continuation
/// in the constant (the caller verifies the result).
fn rename_constant_function(
    arena: &mut Arena,
    e: ExprId,
    var: ExprId,
    c1: ExprId,
) -> Option<ExprId> {
    let ExprNode::Symbol(c_sym) = *arena.node(c1) else {
        return None;
    };
    let ExprNode::Symbol(v_sym) = *arena.node(var) else {
        return None;
    };
    // Maximal nodes free of `var` containing `c1`: walk from the top.
    let mut found: Option<ExprId> = None;
    let mut stack = vec![e];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) || !contains_sym(arena, n, c_sym) {
            continue;
        }
        if !contains_sym(arena, n, v_sym) {
            match found {
                Some(g) if g != n => return None,
                _ => found = Some(n),
            }
            continue;
        }
        arena.node(n).for_each_child(|c| stack.push(c));
    }
    let g = found?;
    if g == c1 || arena.as_num(g).is_some() {
        return None;
    }
    // A plain multiple or shift of the constant is left as it is.
    if matches!(
        arena.node(g),
        ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_)
    ) {
        return None;
    }
    let out = crate::transforms::subs::subs(arena, e, g, c1);
    Some(crate::transforms::eval::eval(arena, out))
}

/// Does `e` contain an `Abs`?
fn has_abs(arena: &Arena, e: ExprId) -> bool {
    let mut stack = vec![e];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let node = arena.node(n);
        if matches!(node, ExprNode::Abs(_)) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// `e` with every `ln|u|` replaced by `ln u` and multiplied by the least
/// common denominator of the coefficients of its logarithms (as terms of
/// the expanded sum); `None` if `e` has no `ln|u|`.
fn strip_log_abs(arena: &mut Arena, e: ExprId) -> Option<ExprId> {
    let out = ln_abs_to_ln(arena, e)?;
    let terms: Vec<ExprId> = match arena.node(out) {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![out],
    };
    let mut lcm = num_bigint::BigInt::one();
    for t in terms {
        let (c, term) = arena.as_coeff_term(t);
        if matches!(arena.node(term), ExprNode::Ln(_)) {
            lcm = num_integer::Integer::lcm(&lcm, c.denom());
        }
    }
    if lcm.is_one() {
        return Some(out);
    }
    let scale = arena.intern_num(Q::from_integer(lcm));
    let scale = arena.intern(ExprNode::Num(scale));
    let scaled = arena.mul(&[scale, out]);
    let scaled = crate::transforms::expand::expand(arena, scaled);
    Some(crate::transforms::eval::eval(arena, scaled))
}

/// `e` with every `ln|u|` replaced by `ln u`; `None` if it has none.
fn ln_abs_to_ln(arena: &mut Arena, e: ExprId) -> Option<ExprId> {
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    let mut changed = false;
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Ln(inner) => match arena.node(inner).clone() {
                ExprNode::Abs(u) => {
                    changed = true;
                    arena.ln(u)
                }
                _ => rebuilt,
            },
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    if !changed {
        return None;
    }
    let out = cache.get(&e).copied().unwrap_or(e);
    Some(crate::transforms::eval::eval(arena, out))
}

/// `e` with `c1` renamed in `e^{k·c1 + r} → c1^{k/k₀}·e^{r}` when every
/// occurrence of `c1` is in such an exponential with a rational `k`, `k₀`
/// the greatest rational dividing every `k` (the new `c1` is `e^{k₀·c1}`;
/// see [`explicit_families`]); `e` itself otherwise.
fn absorb_exp_constant(arena: &mut Arena, e: ExprId, c1: ExprId) -> ExprId {
    let ExprNode::Symbol(c_sym) = *arena.node(c1) else {
        return e;
    };
    // `e^{k·c1 + r}` (r free of c1) → `(k, r)`.
    let split = |arena: &mut Arena, arg: ExprId| -> Option<(Q, Vec<ExprId>)> {
        let terms: Vec<ExprId> = match arena.node(arg) {
            ExprNode::Add(c) => c.to_vec(),
            _ => vec![arg],
        };
        let mut k: Option<Q> = None;
        let mut rest = Vec::with_capacity(terms.len());
        for t in terms {
            let (c, term) = arena.as_coeff_term(t);
            if term == c1 && k.is_none() {
                k = Some(c);
            } else if contains_sym(arena, t, c_sym) {
                return None;
            } else {
                rest.push(t);
            }
        }
        Some((k?, rest))
    };
    let post = crate::base::walk::post_order_ids(arena, e);
    // Pass 1: the multiples `k`, and every occurrence of `c1` in one.
    let mut erased: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    let mut hits: rustc_hash::FxHashMap<ExprId, (Q, Vec<ExprId>)> =
        rustc_hash::FxHashMap::default();
    let one = arena.one;
    for &id in &post {
        let gone = crate::base::walk::rebuild_with_cache(arena, id, &erased);
        let hit = match *arena.node(id) {
            ExprNode::Exp(arg) => split(arena, arg),
            _ => None,
        };
        let new_gone = match hit {
            Some(h) => {
                hits.insert(id, h);
                one
            }
            None => gone,
        };
        erased.insert(id, new_gone);
    }
    let rest = erased.get(&e).copied().unwrap_or(e);
    if hits.is_empty() || contains_sym(arena, rest, c_sym) {
        return e;
    }
    // k₀ = gcd(numerators)/lcm(denominators), with the sign of the first.
    let mut num_gcd = num_bigint::BigInt::from(0);
    let mut den_lcm = num_bigint::BigInt::one();
    let mut sign_neg = None;
    for &id in &post {
        if let Some((k, _)) = hits.get(&id) {
            num_gcd = num_integer::Integer::gcd(&num_gcd, k.numer());
            den_lcm = num_integer::Integer::lcm(&den_lcm, k.denom());
            sign_neg.get_or_insert(k.is_negative());
        }
    }
    if num_gcd == num_bigint::BigInt::from(0) {
        return e;
    }
    let mut k0 = Q::new(num_gcd, den_lcm);
    if sign_neg == Some(true) {
        k0 = -k0;
    }
    // Pass 2: e^{k·c1 + r} → c1^{k/k₀}·e^{r}.
    let mut renamed: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &renamed);
        let new = match hits.get(&id).cloned() {
            Some((k, rest)) => {
                let r = arena.add(&rest);
                let ex = arena.exp(r);
                let p = ode_ratio_to_expr(arena, &(k / &k0));
                let cp = arena.pow(c1, p);
                arena.mul(&[cp, ex])
            }
            None => rebuilt,
        };
        renamed.insert(id, new);
    }
    let out = renamed.get(&e).copied().unwrap_or(e);
    crate::transforms::eval::eval(arena, out)
}

/// Does `sol` solve `expr = 0`?  Its [`residual_verdict`], or [`checkodesol`]
/// when that is undecided.  Used for the families [`explicit_families`]
/// derives by dropping `|·|` and renaming the constant.
fn family_verifies(
    arena: &mut Arena,
    expr: ExprId,
    sol: ExprId,
    func: ExprId,
    var: ExprId,
    c1: ExprId,
) -> bool {
    match residual_verdict(arena, expr, sol, func, var, c1) {
        Some(v) => v,
        None => checkodesol(arena, expr, sol, func, var),
    }
}

/// The residual of the family `sol` (one constant `c1`; others are set to
/// fixed values) in `expr = 0` at three points and four values of `c1`,
/// evaluated over ℂ on principal branches: `Some(true)` when it is below
/// `1e⁻⁹` of the scale of the derivatives at two or more of them (a
/// formula may be a solution only where its radicals take the right branch:
/// `(x/2 + C1/2)²` for `y′ = √y` needs `x + C1 ≥ 0`), `Some(false)` when it
/// evaluates at two or more and is small at none, `None` otherwise.
fn residual_verdict(
    arena: &mut Arena,
    expr: ExprId,
    sol: ExprId,
    func: ExprId,
    var: ExprId,
    c1: ExprId,
) -> Option<bool> {
    let chain = derivative_chain(arena, func, var);
    let order = (1..chain.len())
        .rev()
        .find(|&k| expr_contains(arena, expr, chain[k]))
        .unwrap_or(0);
    let mut ders = vec![sol];
    for k in 1..=order {
        let d = crate::transforms::diff::diff(arena, ders[k - 1], var);
        ders.push(d);
    }
    let mut resid = expr;
    for k in (0..=order).rev() {
        resid = crate::transforms::subs::subs(arena, resid, chain[k], ders[k]);
    }
    // Other constants (C2, … of a reduced higher-order equation) fixed.
    let others: Vec<ExprId> = crate::base::walk::free_symbols(arena, sol)
        .into_iter()
        .filter(|&s| s != c1 && s != var && is_constant_name(arena, s))
        .collect();
    let (mut evaluated, mut passed) = (0usize, 0usize);
    for (xn, xd) in [(3i64, 7i64), (11, 9), (-5, 4)] {
        for (cn, cd) in [(2i64, 3i64), (-5, 4), (7, 5), (-1, 3)] {
            let xv = arena.rational(xn, xd);
            let cv = arena.rational(cn, cd);
            let ov = arena.rational(3, 5);
            let at = |arena: &mut Arena, e: ExprId| {
                let mut e = crate::transforms::subs::subs(arena, e, var, xv);
                e = crate::transforms::subs::subs(arena, e, c1, cv);
                for &o in &others {
                    e = crate::transforms::subs::subs(arena, e, o, ov);
                }
                let e = crate::transforms::eval::eval(arena, e);
                // Zeros pursued only to the digits asked (true zeros are
                // the rule here; `evalf_complex64` works to certify them).
                let z = crate::transforms::evalf::evalf_complex(arena, e, 12).ok()?;
                crate::transforms::evalf::abs_to_f64(&z).filter(|m| m.is_finite())
            };
            let Some(r) = at(arena, resid) else {
                continue;
            };
            let mut scale = 1.0f64;
            for &d in &ders {
                if let Some(m) = at(arena, d) {
                    scale = scale.max(m);
                }
            }
            evaluated += 1;
            if r <= 1e-9 * scale {
                passed += 1;
                if passed >= 2 {
                    return Some(true);
                }
            }
        }
    }
    if passed >= 2 {
        Some(true)
    } else if evaluated >= 2 && passed == 0 {
        Some(false)
    } else {
        None
    }
}

/// Is the symbol `s` named like an integration constant (`C1`, `C2`, …)?
fn is_constant_name(arena: &Arena, s: ExprId) -> bool {
    match arena.node(s) {
        ExprNode::Symbol(sid) => {
            let name = arena.symbol_name(*sid);
            name.len() > 1 && name.starts_with('C') && name[1..].chars().all(|c| c.is_ascii_digit())
        }
        _ => false,
    }
}

/// Collect multiplicative factors from an expression.
/// If `expr` is `Mul(a, b, c)`, return `[a, b, c]`.
/// If `expr` is `Neg(inner)`, return factors of inner with a `-1` prepended.
/// Otherwise return `[expr]`.
fn collect_mul_factors(arena: &mut Arena, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    let mut expr = expr;
    loop {
        match arena.node(expr).clone() {
            ExprNode::Mul(children) => {
                factors.extend(children.iter().copied());
                return factors;
            }
            ExprNode::Neg(inner) => {
                factors.push(arena.int(-1));
                expr = inner;
            }
            _ => {
                factors.push(expr);
                return factors;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Variable-coefficient first-order linear: y' + P(x)*y = Q(x)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve y' + P(x)*y = Q(x) via integrating factor.
///
/// Algorithm:
/// 1. Extract dy/dx from the ODE expression
/// 2. Among remaining terms, find those containing func (y) → these give P(x)*y
/// 3. Terms not containing func give -Q(x)
/// 4. Extract P(x) by dividing the y-containing terms by func
/// 5. Compute integrating factor: μ = exp(∫P(x)dx)
/// 6. Solution: y = (1/μ)·[∫Q(x)·μ dx + C1]
fn try_first_order_linear_general(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    first_order_linear(arena, expr, func, var, func_sym, var_sym, false)
}

/// [`try_first_order_linear_general`] when `∫P dx` has no closed form:
/// the integrating factor `e^{∫P dx}` with a formal `Integral`, SymPy's
/// `exp(−Integral(P, x))·(C1 + Integral(Q·exp(Integral(P, x)), x))`
/// (`(1 + x²)·y′ + cos(x)·y = eˣ` was refused).  Tried last, after every
/// method that could find a closed form.
fn try_first_order_linear_formal(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    first_order_linear(arena, expr, func, var, func_sym, var_sym, true)
}

/// See [`try_first_order_linear_general`]; `formal`: a non-elementary
/// `∫P dx` stays a formal `Integral` instead of refusing.
fn first_order_linear(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
    formal: bool,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying variable-coefficient first-order linear");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut has_dy = false;
    let mut dy_coeff_rational = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut y_terms: Vec<ExprId> = Vec::new(); // terms that contain func (y)
    let mut free_terms: Vec<ExprId> = Vec::new(); // terms free of func
    // Coefficients a(x) of `y′` that are not plain numbers (`x·y′`).
    let mut dy_coeff_terms: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            has_dy = true;
            dy_coeff_rational += coeff;
        } else if expr_contains(arena, child, dy_dx) {
            // a(x)·y′ with a(x) free of y.
            let ExprNode::Mul(factors) = arena.node(term).clone() else {
                return None;
            };
            let (with_dy, rest): (Vec<ExprId>, Vec<ExprId>) =
                factors.iter().copied().partition(|&f| f == dy_dx);
            if with_dy.len() != 1 || rest.iter().any(|&f| contains_sym(arena, f, func_sym)) {
                return None;
            }
            has_dy = true;
            let c = ode_ratio_to_expr(arena, &coeff);
            let mut parts = vec![c];
            parts.extend(rest);
            dy_coeff_terms.push(arena.mul(&parts));
        } else if contains_sym(arena, child, func_sym) {
            y_terms.push(child);
        } else {
            free_terms.push(child);
        }
    }

    use num_traits::Zero;
    if !has_dy {
        return None;
    }
    // A coefficient a(x) of y′ that is not a number: divide through by it.
    let dy_coeff_expr: Option<ExprId> = if dy_coeff_terms.is_empty() {
        None
    } else {
        let mut parts = dy_coeff_terms.clone();
        if !dy_coeff_rational.is_zero() {
            parts.push(ode_ratio_to_expr(arena, &dy_coeff_rational));
        }
        let a = arena.add(&parts);
        let a = crate::transforms::eval::eval(arena, a);
        match arena.as_num(a) {
            Some(r) => {
                dy_coeff_rational = r.clone();
                None
            }
            None => Some(a),
        }
    };
    if dy_coeff_expr.is_none() && dy_coeff_rational.is_zero() {
        return None;
    }

    // We need at least one y-term for this to be a linear ODE with y dependence.
    // If there are no y-terms, it's simple separable.
    if y_terms.is_empty() {
        return None;
    }

    // Check that the y-terms are LINEAR in y: each y-term = something * y
    // For each y-term, try to extract the coefficient of y.
    let mut p_x_terms: Vec<ExprId> = Vec::new();

    for &yt in &y_terms {
        {
            let px = extract_coeff_of_func(arena, yt, func, func_sym, var_sym)?;
            p_x_terms.push(px);
        }
    }

    // Normalize by dy_coeff: divide P(x) and Q(x) by the coefficient of y'
    if let Some(a) = dy_coeff_expr {
        let neg_one = arena.int(-1);
        let inv_a = arena.pow(a, neg_one);
        p_x_terms = p_x_terms
            .iter()
            .map(|&p| {
                let t = arena.mul(&[inv_a, p]);
                crate::transforms::eval::eval(arena, t)
            })
            .collect();
        free_terms = free_terms
            .iter()
            .map(|&f| {
                let t = arena.mul(&[inv_a, f]);
                crate::transforms::eval::eval(arena, t)
            })
            .collect();
    } else if !dy_coeff_rational.is_one() {
        let inv_coeff = num_rational::Ratio::<num_bigint::BigInt>::one() / &dy_coeff_rational;
        let inv_id = {
            let nid = arena.intern_num(inv_coeff);
            arena.intern(ExprNode::Num(nid))
        };
        let mut scaled_p = Vec::new();
        for &p in &p_x_terms {
            scaled_p.push(arena.mul(&[inv_id, p]));
        }
        p_x_terms = scaled_p;

        let mut scaled_f = Vec::new();
        for &f in &free_terms {
            scaled_f.push(arena.mul(&[inv_id, f]));
        }
        free_terms = scaled_f;
    }

    // P(x) = sum of p_x_terms
    let p_x = if p_x_terms.is_empty() {
        arena.zero
    } else if p_x_terms.len() == 1 {
        p_x_terms[0]
    } else {
        arena.add(&p_x_terms)
    };

    // Check that P(x) doesn't contain y (it shouldn't at this point, but verify)
    if contains_sym(arena, p_x, func_sym) {
        return None;
    }

    // Try the constant-coefficient path first for efficiency — if P(x) is a
    // pure rational number, delegate to the existing constant-coefficient solver
    // which handles the nonhomogeneous case (y' + a*y = Q(x)).
    // The constant-coefficient solver wants `y′ + a·y + f(x)` with a unit
    // coefficient on `y′`: the normalised equation is built for it (before,
    // `2y′ + y = cos x` was handed over as is, refused there, and not
    // solved at all).
    let p_x = crate::transforms::eval::eval(arena, p_x);
    if arena.as_num(p_x).is_some() {
        let normalised = if dy_coeff_expr.is_none() && dy_coeff_rational.is_one() {
            expr
        } else {
            let py = arena.mul(&[p_x, func]);
            let mut parts = vec![dy_dx, py];
            parts.extend(free_terms.iter().copied());
            let e = arena.add(&parts);
            crate::transforms::eval::eval(arena, e)
        };
        if let Some(r) = try_first_order_linear(arena, normalised, func, var, func_sym, var_sym) {
            return Some(r);
        }
    }

    // Check that P(x) actually depends on x — if it's constant, it would have
    // been caught above (as a numeric). But it might be a symbolic constant.
    // We proceed regardless.

    // Q(x) = -(free_terms)  since expr = y' + P(x)*y + free_terms = 0
    //                        means y' + P(x)*y = -free_terms = Q(x)
    let q_x = if free_terms.is_empty() {
        arena.zero
    } else {
        let sum = arena.add(&free_terms);
        arena.neg(sum)
    };

    // Integrating factor: μ = exp(∫P(x)dx)
    let int_px = crate::transforms::integrate::integrate(arena, p_x, var);

    // Check if integration failed (returned an unevaluated Integral node)
    let mu = if integral_failed(arena, int_px) {
        // Integration of P(x) failed: no closed integrating factor.
        if !formal {
            return None;
        }
        let p_x = crate::transforms::eval::eval(arena, p_x);
        let formal_int = arena.intern(ExprNode::Integral(p_x, var));
        arena.exp(formal_int)
    } else if formal {
        return None; // the closed form is the other method's
    } else {
        exp_of_log_sum(arena, int_px)
    };

    // Solution: y = (1/μ) · [∫ Q(x)·μ dx + C1]
    let neg_one = arena.int(-1);
    let inv_mu = arena.pow(mu, neg_one);

    let c1 = arena.symbol("C1");

    if arena.is_zero_structural(q_x) {
        // Homogeneous: y' + P(x)*y = 0 → y = C1 * exp(-∫P(x)dx)
        let solution = arena.mul(&[c1, inv_mu]);
        return Some(OdeResult {
            solution,
            constants: vec![c1],
            branches: Vec::new(),
        });
    }

    // Nonhomogeneous: y = (1/μ) * [∫ Q(x)*μ dx + C1]
    let integrand = arena.mul(&[q_x, mu]);
    // Simplify products of exponentials before integrating
    let integrand = crate::transforms::eval::eval(arena, integrand);
    let integral = integrate_forms(arena, integrand, var);

    let inner = arena.add(&[integral, c1]);
    let solution = arena.mul(&[inv_mu, inner]);

    Some(OdeResult {
        solution,
        constants: vec![c1],
        branches: Vec::new(),
    })
}

/// Try to extract the coefficient of `func` (y) from an expression that should
/// be of the form `P(x) * y`. Returns `Some(P(x))` if the expression is linear
/// in y, `None` otherwise.
fn extract_coeff_of_func(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<ExprId> {
    // Case 1: expr is exactly y
    if expr == func {
        return Some(arena.one);
    }

    // Case 2: expr is Neg(y) → coefficient is -1
    if let ExprNode::Neg(inner) = arena.node(expr).clone() {
        if inner == func {
            let neg_one = arena.int(-1);
            return Some(neg_one);
        }
        // Neg(something) — recurse
        if let Some(inner_coeff) = extract_coeff_of_func(arena, inner, func, func_sym, _var_sym) {
            let result = arena.neg(inner_coeff);
            return Some(result);
        }
        return None;
    }

    // Case 3: expr is Mul([...]) containing func exactly once
    if let ExprNode::Mul(ref children) = arena.node(expr).clone() {
        let mut found_y = false;
        let mut other_factors: Vec<ExprId> = Vec::new();
        let mut y_count = 0;

        for &child in children {
            if child == func {
                y_count += 1;
                if y_count > 1 {
                    return None; // y^2 or higher — nonlinear
                }
                found_y = true;
            } else if contains_sym(arena, child, func_sym) {
                // A factor that contains y but isn't y itself — nonlinear
                return None;
            } else {
                other_factors.push(child);
            }
        }

        if found_y {
            let coeff = if other_factors.is_empty() {
                arena.one
            } else if other_factors.len() == 1 {
                other_factors[0]
            } else {
                arena.mul(&other_factors)
            };
            return Some(coeff);
        }
    }

    // Case 4: expr = coeff_num * something_with_y
    // Use as_coeff_term to peel off a numeric coefficient, then check the term
    {
        let (coeff, term) = arena.as_coeff_term(expr);
        if !coeff.is_one()
            && term != expr
            && let Some(inner_coeff) = extract_coeff_of_func(arena, term, func, func_sym, _var_sym)
        {
            let coeff_id = {
                let nid = arena.intern_num(coeff);
                arena.intern(ExprNode::Num(nid))
            };
            let result = arena.mul(&[coeff_id, inner_coeff]);
            return Some(result);
        }
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact ODE solver: M(x,y)dx + N(x,y)dy = 0
// ═══════════════════════════════════════════════════════════════════════════

/// Extract M(x,y) and N(x,y) from an ODE expression of the form `M + N·y' = 0`.
///
/// Returns `(M, N)` where M is the sum of terms not containing `dy/dx`
/// and N is the total coefficient of `dy/dx`.
///
/// Returns `None` when the expression cannot be decomposed (e.g. `(y')²`).
fn extract_m_n(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId)> {
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));

    // Handle single-term expressions.
    if expr == dy_dx {
        return Some((arena.zero, arena.one));
    }

    let children = match arena.node(expr).clone() {
        ExprNode::Add(c) => c,
        _ => return None,
    };

    let mut m_terms: Vec<ExprId> = Vec::new();
    let mut n_terms: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            // Simple numeric coefficient of dy/dx.
            let coeff_id = {
                let nid = arena.intern_num(coeff);
                arena.intern(ExprNode::Num(nid))
            };
            n_terms.push(coeff_id);
        } else if expr_contains(arena, child, dy_dx) {
            // child contains dy/dx inside a product — try to peel it off.
            if let ExprNode::Mul(ref mul_children) = arena.node(child).clone() {
                let mut found_dy = false;
                let mut other_factors: Vec<ExprId> = Vec::new();
                for &mc in mul_children.iter() {
                    if mc == dy_dx && !found_dy {
                        found_dy = true;
                    } else {
                        other_factors.push(mc);
                    }
                }
                if found_dy {
                    let n_factor = match other_factors.len() {
                        0 => arena.one,
                        1 => other_factors[0],
                        _ => arena.mul(&other_factors),
                    };
                    n_terms.push(n_factor);
                } else {
                    return None; // dy/dx in non-simple position
                }
            } else {
                return None;
            }
        } else {
            m_terms.push(child);
        }
    }

    if n_terms.is_empty() {
        return None; // No dy/dx term
    }

    let m_expr = match m_terms.len() {
        0 => arena.zero,
        1 => m_terms[0],
        _ => arena.add(&m_terms),
    };

    let n_expr = match n_terms.len() {
        1 => n_terms[0],
        _ => arena.add(&n_terms),
    };

    Some((m_expr, n_expr))
}

/// Solve an exact first-order ODE: `M(x,y) + N(x,y)·y' = 0`
/// where `∂M/∂y = ∂N/∂x`.
///
/// The potential function F(x,y) satisfying `∂F/∂x = M` and `∂F/∂y = N`
/// is computed as:
///   1. `F = ∫M dx + g(y)`
///   2. `g'(y) = N − ∂(∫M dx)/∂y`
///   3. `g(y) = ∫ g'(y) dy`
///
/// The implicit solution is `F(x,y) = C1`.  When possible the solver
/// also tries to solve explicitly for `y`.
fn try_exact_ode(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying exact ODE");

    let (m_expr, n_expr) = extract_m_n(arena, expr, func, var)?;

    // At least one of M, N must depend on the dependent variable for this
    // to be a genuinely exact ODE (otherwise the linear / separable solvers
    // are better suited).
    if !contains_sym(arena, m_expr, func_sym) && !contains_sym(arena, n_expr, func_sym) {
        return None;
    }

    // ── Exactness check: ∂M/∂y = ∂N/∂x ───────────────────────────
    let dm_dy = crate::transforms::diff::diff(arena, m_expr, func);
    let dn_dx = crate::transforms::diff::diff(arena, n_expr, var);

    let diff_check = arena.sub(dm_dy, dn_dx);
    let diff_eval = crate::transforms::eval::eval(arena, diff_check);
    let diff_expanded = crate::transforms::expand::expand(arena, diff_eval);
    let diff_simplified = crate::transforms::eval::eval(arena, diff_expanded);

    if diff_simplified != arena.zero && !rational_zero(arena, diff_simplified) {
        return None; // Not exact
    }

    // ── Build potential function F(x,y) ───────────────────────────
    // Step 1: F_partial = ∫ M dx  (treating y as constant).  A case split
    // on `y` (`∫ y·e^{xy} dx = y·Piecewise(e^{xy}/y if y ≠ 0, x)`) is taken
    // on its generic branch: a potential is needed only off a line (`y·e^{xy}
    // + (x·e^{xy} + 1)·y′ = 0` was refused).
    let integral_m = crate::transforms::integrate::integrate(arena, m_expr, var);
    if integral_failed(arena, integral_m) {
        return None; // Integration of M w.r.t. x failed
    }
    let integral_m = generic_branches(arena, integral_m, var).unwrap_or(integral_m);

    // Step 2: g'(y) = N − ∂(∫M dx)/∂y, free of x.  When `∫M dx` has a
    // `ln|u|` whose partner in `N` is `ln u` (`(y/x + 3x²) + (ln x + 1)·y′`:
    // `g′ = ln x + 1 − ln|x|`), the potential is taken with `ln u`, as SymPy
    // does (it differs by a constant on each interval where `u` keeps its
    // sign).
    let g_prime_of = |arena: &mut Arena, im: ExprId| -> ExprId {
        let d_intm_dy = crate::transforms::diff::diff(arena, im, func);
        let g_prime = arena.sub(n_expr, d_intm_dy);
        let g_prime = crate::transforms::eval::eval(arena, g_prime);
        let g_prime = crate::transforms::expand::expand(arena, g_prime);
        crate::transforms::eval::eval(arena, g_prime)
    };
    let mut integral_m = integral_m;
    let mut g_prime = g_prime_of(arena, integral_m);
    if contains_sym(arena, g_prime, var_sym)
        && let Some(stripped) = ln_abs_to_ln(arena, integral_m)
    {
        let g2 = g_prime_of(arena, stripped);
        if !contains_sym(arena, g2, var_sym) {
            integral_m = stripped;
            g_prime = g2;
        }
    }

    // g'(y) must be free of x.
    if contains_sym(arena, g_prime, var_sym) {
        return None;
    }

    // Step 3: g(y) = ∫ g'(y) dy
    let g_y = crate::transforms::integrate::integrate(arena, g_prime, func);
    if integral_failed(arena, g_y) {
        return None;
    }

    // F(x,y) = ∫M dx + g(y)
    let potential = arena.add(&[integral_m, g_y]);
    let potential = crate::transforms::eval::eval(arena, potential);

    let c1 = arena.symbol("C1");

    // Try to solve F(x,y) = C1 for y explicitly.  Several roots stay
    // implicit here: a later method may find nicer families, and
    // [`dsolve`] takes them all in the end (`explicit_families`).
    let f_minus_c1 = arena.sub(potential, c1);
    let solutions = known_roots(arena, f_minus_c1, func);

    if solutions.len() == 1 {
        return Some(OdeResult {
            solution: solutions[0].value,
            constants: vec![c1],
            branches: Vec::new(),
        });
    }

    // The implicit solution `F(x, y) − C1` (= 0), in the form of the other
    // implicit solutions.  Before, `F` alone was returned: the constant was
    // missing from the answer (`2xy + (x² + 3y²)y′ = 0` gave `y³ + x²y`).
    let implicit = crate::transforms::eval::eval(arena, f_minus_c1);
    Some(OdeResult {
        solution: implicit,
        constants: vec![c1],
        branches: Vec::new(),
    })
}

/// Attempt to find an integrating factor for a non-exact first-order ODE.
///
/// Given `M + N·y' = 0` with `∂M/∂y ≠ ∂N/∂x`:
///
/// 1. **μ = μ(x):**  if `(∂M/∂y − ∂N/∂x) / N` depends only on `x`,
///    then `μ = exp(∫ that dx)`.
/// 2. **μ = μ(y):**  if `(∂N/∂x − ∂M/∂y) / M` depends only on `y`,
///    then `μ = exp(∫ that dy)`.
///
/// After multiplying through by μ the ODE becomes exact and is solved
/// via [`try_exact_ode`].
fn try_integrating_factor_ode(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying integrating factor for non-exact ODE");

    let (m_expr, n_expr) = extract_m_n(arena, expr, func, var)?;

    // Need y-dependence for this to be meaningful.
    if !contains_sym(arena, m_expr, func_sym) && !contains_sym(arena, n_expr, func_sym) {
        return None;
    }

    let dm_dy = crate::transforms::diff::diff(arena, m_expr, func);
    let dn_dx = crate::transforms::diff::diff(arena, n_expr, var);

    let diff_mn = arena.sub(dm_dy, dn_dx); // ∂M/∂y − ∂N/∂x
    let diff_eval = crate::transforms::eval::eval(arena, diff_mn);
    let diff_expanded = crate::transforms::expand::expand(arena, diff_eval);
    let diff_simplified = crate::transforms::eval::eval(arena, diff_expanded);

    if diff_simplified == arena.zero || rational_zero(arena, diff_simplified) {
        // Already exact — delegate.
        return try_exact_ode(arena, expr, func, var, func_sym, var_sym);
    }

    // ── Try μ(x): (∂M/∂y − ∂N/∂x) / N free of y ─────────────────
    {
        let ratio = arena.div(diff_simplified, n_expr);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        let ratio = crate::transforms::expand::expand(arena, ratio);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        // Multivariate normal form: the univariate `cancel` in `x` could
        // not cancel `x + 2y` from `(x + 2y)/x² · x/(x + 2y)` (coefficients
        // in `y`), so μ = x of `(2x + y)/x + ((2y + x)/x)·y′` was missed.
        let ratio_cancelled = crate::simplify::ratsimp::ratsimp(arena, ratio);
        let ratio_cancelled = crate::transforms::eval::eval(arena, ratio_cancelled);

        if !contains_sym(arena, ratio_cancelled, func_sym) {
            let int_ratio = crate::transforms::integrate::integrate(arena, ratio_cancelled, var);
            if !integral_failed(arena, int_ratio) {
                let mu = exp_of_log_sum(arena, int_ratio);

                // New M' = μ·M,  N' = μ·N
                let new_m = arena.mul(&[mu, m_expr]);
                let new_n = arena.mul(&[mu, n_expr]);

                let dy_dx = arena.intern(ExprNode::Derivative(func, var));
                let n_dy = arena.mul(&[new_n, dy_dx]);
                let new_expr = arena.add(&[new_m, n_dy]);
                let new_expr = crate::transforms::eval::eval(arena, new_expr);

                if let Some(result) = try_exact_ode(arena, new_expr, func, var, func_sym, var_sym) {
                    return Some(result);
                }
            }
        }
    }

    // ── Try μ(y): (∂N/∂x − ∂M/∂y) / M free of x ─────────────────
    {
        let neg_diff = arena.neg(diff_simplified); // ∂N/∂x − ∂M/∂y
        let ratio = arena.div(neg_diff, m_expr);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        let ratio = crate::transforms::expand::expand(arena, ratio);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        let ratio_cancelled = crate::simplify::ratsimp::ratsimp(arena, ratio);
        let ratio_cancelled = crate::transforms::eval::eval(arena, ratio_cancelled);

        if !contains_sym(arena, ratio_cancelled, var_sym) {
            let int_ratio = crate::transforms::integrate::integrate(arena, ratio_cancelled, func);
            if !integral_failed(arena, int_ratio) {
                let mu = exp_of_log_sum(arena, int_ratio);

                let new_m = arena.mul(&[mu, m_expr]);
                let new_n = arena.mul(&[mu, n_expr]);

                let dy_dx = arena.intern(ExprNode::Derivative(func, var));
                let n_dy = arena.mul(&[new_n, dy_dx]);
                let new_expr = arena.add(&[new_m, n_dy]);
                let new_expr = crate::transforms::eval::eval(arena, new_expr);

                if let Some(result) = try_exact_ode(arena, new_expr, func, var, func_sym, var_sym) {
                    return Some(result);
                }
            }
        }
    }

    None
}

// Check if an expression contains a specific symbol.
// ═══════════════════════════════════════════════════════════════════════════
// Homogeneous coefficient ODE: y' = f(y/x)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a first-order ODE of the form y' = f(y/x).
///
/// Detection: substitute y = v*x in the RHS.  If the result simplifies to a
/// function of v alone (no x), the equation is homogeneous of degree 0.
///
/// Solution via v = y/x:
///   y = v*x  →  y' = v + x*v'
///   v + x*v' = f(v)  →  dv/(f(v) - v) = dx/x
///   ∫ dv/(f(v) - v) = ln|x| + C1
///   Back-substitute v = y/x.
fn try_homogeneous_coefficient(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying homogeneous coefficient");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    // Must be first-order only
    if expr_contains(arena, expr, d2y_dx2) {
        return None;
    }

    // Extract the RHS: expr = dy/dx + ... = 0  →  dy/dx = -...
    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut has_dy = false;
    let mut dy_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut other_terms: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            has_dy = true;
            dy_coeff += coeff;
        } else {
            other_terms.push(child);
        }
    }

    use num_traits::Zero;
    if !has_dy || dy_coeff.is_zero() {
        return None;
    }

    // RHS of y' = RHS (negate the other terms, normalize by dy_coeff)
    let rhs = if other_terms.is_empty() {
        return None;
    } else {
        let sum = arena.add(&other_terms);
        arena.neg(sum)
    };

    let rhs = if !dy_coeff.is_one() {
        let inv_id = ode_ratio_to_expr(
            arena,
            &(num_rational::Ratio::<num_bigint::BigInt>::one() / &dy_coeff),
        );
        let s = arena.mul(&[inv_id, rhs]);
        crate::transforms::eval::eval(arena, s)
    } else {
        rhs
    };

    // The RHS must depend on both x and y for this to be interesting.
    if !contains_sym(arena, rhs, func_sym) || !contains_sym(arena, rhs, var_sym) {
        return None;
    }

    // For a degree-0 homogeneous function f(x,y), f(tx, ty) = f(x,y).
    // In particular f(x, y) = f(1, y/x).  So f(v) = RHS|_{y→v, x→1}.
    //
    // Detection: substitute y→v, x→1.  If the result is free of x, the
    // equation is homogeneous of degree 0.
    //
    // This avoids the need to simplify (v*x)^n / x^n etc.
    let v = arena.symbol("__v");

    // Compute f(v) = RHS(x=1, y=v)
    let rhs_sub = crate::transforms::subs::subs(arena, rhs, func, v);
    let rhs_sub = crate::transforms::subs::subs(arena, rhs_sub, var, arena.one);
    let rhs_sub = crate::transforms::eval::eval(arena, rhs_sub);
    let rhs_sub = crate::transforms::expand::expand(arena, rhs_sub);
    let rhs_sub = crate::transforms::eval::eval(arena, rhs_sub);

    // f(v) must be free of x (it should be, since we set x=1).
    if contains_sym(arena, rhs_sub, var_sym) {
        return None;
    }

    // Verify homogeneity of degree 0: RHS(x, v·x) must reduce to f(v).
    // Without this check any polynomial RHS (e.g. y² + x², degree 2)
    // would be misclassified.  The difference is brought to the
    // multivariate rational normal form (`ratsimp`): the univariate
    // `cancel` in `x` it used could not cancel `x` from `(3vx − x)/(vx + x)`
    // (coefficients in `v`), so every homogeneous RHS that is a quotient of
    // sums (`(3y − x)/(x + y)`, `2xy/(x² − y²)`) was refused.
    {
        let vx = arena.mul(&[v, var]);
        let probe = crate::transforms::subs::subs(arena, rhs, func, vx);
        let probe = crate::transforms::eval::eval(arena, probe);
        let diff = arena.sub(probe, rhs_sub);
        let diff = crate::transforms::eval::eval(arena, diff);
        let diff = crate::simplify::ratsimp::ratsimp(arena, diff);
        let diff = crate::transforms::eval::eval(arena, diff);
        if !arena.is_zero_structural(diff) {
            return None;
        }
    }

    // Now we have:  v + x*v' = f(v)  →  dv/(f(v) - v) = dx/x
    // Integrate:  ∫ dv/(f(v) - v) = ln|x| + C1
    let f_v_minus_v = arena.sub(rhs_sub, v);
    let f_v_minus_v = crate::transforms::eval::eval(arena, f_v_minus_v);
    let f_v_minus_v = crate::simplify::ratsimp::ratsimp(arena, f_v_minus_v);
    let f_v_minus_v = crate::transforms::eval::eval(arena, f_v_minus_v);

    if f_v_minus_v == arena.zero {
        // f(v) = v means y' = y/x → y = C1*x (linear through origin)
        let c1 = arena.symbol("C1");
        let solution = arena.mul(&[c1, var]);
        return Some(OdeResult {
            solution,
            constants: vec![c1],
            branches: Vec::new(),
        });
    }

    let neg_one = arena.int(-1);
    let inv_fv = arena.pow(f_v_minus_v, neg_one);
    let lhs_integral = integrate_forms(arena, inv_fv, v);

    // If integration of 1/(f(v)-v) failed, bail out.
    if integral_failed(arena, lhs_integral) {
        return None;
    }

    // lhs_integral = ln|x| + C1
    let abs_x = arena.abs(var);
    let ln_abs_x = arena.ln(abs_x);
    let c1 = arena.symbol("C1");
    let rhs_eq = arena.add(&[ln_abs_x, c1]);

    // Implicit solution: lhs_integral(v) = ln|x| + C1
    // Back-substitute v = y/x:  lhs(y/x) - ln|x| - C1 = 0
    let y_over_x = arena.div(func, var);
    let lhs_backsub = crate::transforms::subs::subs(arena, lhs_integral, v, y_over_x);
    let lhs_backsub = crate::transforms::eval::eval(arena, lhs_backsub);

    let implicit = arena.sub(lhs_backsub, rhs_eq);
    let implicit = crate::transforms::eval::eval(arena, implicit);

    // Try to solve for y explicitly (several roots: see `try_exact_ode`).
    let solutions = known_roots(arena, implicit, func);
    if solutions.len() == 1 {
        let sol = crate::transforms::eval::eval(arena, solutions[0].value);
        return Some(OdeResult {
            solution: sol,
            constants: vec![c1],
            branches: Vec::new(),
        });
    }

    // Return implicit form.
    Some(OdeResult {
        solution: implicit,
        constants: vec![c1],
        branches: Vec::new(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// nth-order reducible ODE: F(y, y', y'') = 0, no explicit x
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a second-order ODE where the independent variable doesn't appear:
///   F(y, y', y'') = 0
///
/// Substitution: p = y', y'' = p·dp/dy reduces to a first-order ODE in p(y).
fn try_nth_order_reducible(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    _func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying nth-order reducible (missing x)");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    // Must have y'' present.
    if !expr_contains(arena, expr, d2y_dx2) {
        return None;
    }

    // The independent variable x must NOT appear explicitly (only through
    // y and its derivatives).
    // We check: after removing derivative nodes, does x appear?
    // Strategy: substitute y''→__d2, y'→__d1, then check if var_sym remains.
    let d2_placeholder = arena.symbol("__d2");
    let d1_placeholder = arena.symbol("__d1");
    let stripped = crate::transforms::subs::subs(arena, expr, d2y_dx2, d2_placeholder);
    let stripped = crate::transforms::subs::subs(arena, stripped, dy_dx, d1_placeholder);
    let has_x = contains_sym(arena, stripped, var_sym);
    let has_y = contains_sym(arena, stripped, _func_sym);
    if has_x && has_y {
        return None; // neither reduction applies
    }
    let p = arena.symbol("__p");
    let p_sym = match arena.node(p) {
        ExprNode::Symbol(s) => *s,
        _ => return None,
    };
    let c2 = arena.symbol("C2");

    // The reduced first-order solution `p = f(·, C1)` must be explicit and
    // carry its constant.  Before, an implicit one (still in `p`, and
    // without its constant) was integrated as if it were `p`: `y″ + 2y′²
    // = 0` gave `−1/4·ln(−2·__p²·(C2 + x))`.
    let usable = |arena: &Arena, r: &OdeResult| {
        !contains_sym(arena, r.solution, p_sym)
            && r.constants.len() == 1
            && expr_contains(arena, r.solution, r.constants[0])
            && !crate::base::walk::has_unevaluated(arena, r.solution)
    };

    // F(x, y′, y″) = 0 (no y): p(x) = y′ gives F(x, p, p′) = 0, and
    // y = ∫ p dx + C2 is explicit.
    if !has_y {
        let dp_dx = arena.intern(ExprNode::Derivative(p, var));
        let reduced = crate::transforms::subs::subs(arena, expr, d2y_dx2, dp_dx);
        let reduced = crate::transforms::subs::subs(arena, reduced, dy_dx, p);
        let reduced = crate::transforms::eval::eval(arena, reduced);
        if let Some(pr) = dsolve(arena, reduced, p, var)
            && usable(arena, &pr)
        {
            // One family of `y` per family of `p` (`y″ = y′³`: `p = ±(C1 −
            // 2x)^(−1/2)`; before 0.30 only the first).
            let antiderivative = |arena: &mut Arena, p_family: ExprId| {
                let integral = crate::transforms::integrate::integrate(arena, p_family, var);
                (!integral_failed(arena, integral)).then(|| {
                    let s = arena.add(&[integral, c2]);
                    crate::transforms::eval::eval(arena, s)
                })
            };
            if let Some(solution) = antiderivative(arena, pr.solution) {
                let mut branches = Vec::new();
                for &b in &pr.branches {
                    match antiderivative(arena, b) {
                        Some(y_b) => branches.push(y_b),
                        None => tracing::debug!("ode: a family of y' has no closed antiderivative"),
                    }
                }
                return Some(OdeResult {
                    solution,
                    constants: vec![pr.constants[0], c2],
                    branches,
                });
            }
        }
        if has_x {
            return None;
        }
    }

    // Now perform the reduction: let p = dy/dx, then d²y/dx² = p·dp/dy
    let dp_dy = arena.intern(ExprNode::Derivative(p, func));
    let p_dp_dy = arena.mul(&[p, dp_dy]);

    // Substitute: y'' → p·dp/dy,  y' → p
    let reduced = crate::transforms::subs::subs(arena, expr, d2y_dx2, p_dp_dy);
    let reduced = crate::transforms::subs::subs(arena, reduced, dy_dx, p);
    let reduced = crate::transforms::eval::eval(arena, reduced);
    // A factor `p` common to every term is divided out (it only carries
    // the constant solutions `y′ = 0`): `y·y″ − y′²` becomes `y·p′ − p`,
    // linear in `p(y)`.
    let reduced = divide_common_factor(arena, reduced, p).unwrap_or(reduced);

    // Now `reduced` is a first-order ODE in p(y) with independent var = y.
    let p_result = dsolve(arena, reduced, p, func)?;
    if !usable(arena, &p_result) {
        return None;
    }

    // p_result.solution gives p = f(y, C1).
    // Now solve dy/dx = p(y) — this is separable: ∫ dy/p(y) = x + C2.
    let p_sol = p_result.solution;

    // Set up: dy/dx - p_sol = 0  →  ∫ 1/p_sol dy = x + C2
    // We need to integrate 1/p_sol w.r.t. y.
    let neg_one_id = arena.int(-1);
    let inv_p = arena.pow(p_sol, neg_one_id);
    let inv_p = crate::transforms::eval::eval(arena, inv_p);
    let lhs_integral = crate::transforms::integrate::integrate(arena, inv_p, func);

    if integral_failed(arena, lhs_integral) {
        return None; // Can't integrate 1/p(y)
    }

    // Implicit solution: ∫ dy/p(y) = x + C2
    let rhs = arena.add(&[var, c2]);
    let implicit = arena.sub(lhs_integral, rhs);
    let implicit = crate::transforms::eval::eval(arena, implicit);

    // Try to solve explicitly for y (several roots: see `try_exact_ode`).
    let solutions = known_roots(arena, implicit, func);
    if solutions.len() == 1 {
        let sol = crate::transforms::eval::eval(arena, solutions[0].value);
        let mut constants = p_result.constants;
        constants.push(c2);
        return Some(OdeResult {
            solution: sol,
            constants,
            branches: Vec::new(),
        });
    }

    // Return implicit form.
    let mut constants = p_result.constants;
    constants.push(c2);
    Some(OdeResult {
        solution: implicit,
        constants,
        branches: Vec::new(),
    })
}

/// Does `sym` occur free in `expr`?  Bound occurrences (a `Sum` index, a
/// `RootOf` variable; see `walk::binder`) do not count.  Up to 0.28 this
/// was a structural walk with no visited set, exponential on a DAG with
/// shared sub-expressions.
fn contains_sym(arena: &Arena, expr: ExprId, sym: SymbolId) -> bool {
    crate::base::walk::has_free_symbol(arena, expr, sym)
}

/// `∫ e d(var)`, trying equivalent forms of `e` until one integrates: as
/// given, with products of exponentials merged (`powsimp`: variation of
/// parameters builds `x·e⁻ˣ·e⁻ˣ/x²/e⁻²ˣ`, which is `1/x`), as one
/// fraction (`ratsimp`), expanded, and with the common factor of every sum
/// taken out (`factor_terms`: `∫ dv/(v·ln v − v)` is not found, `∫ dv/(v·(ln v
/// − 1)) = ln|ln v − 1|` is; the homogeneous equation `y′ = (y/x)·ln(y/x)`
/// was refused).  The first attempt's result (with its formal `Integral`)
/// when none does.
fn integrate_forms(arena: &mut Arena, e: ExprId, var: ExprId) -> ExprId {
    let first = crate::transforms::integrate::integrate(arena, e, var);
    if !integral_failed(arena, first) {
        return first;
    }
    let merged = crate::simplify::powsimp::powsimp(arena, e);
    let merged = crate::transforms::eval::eval(arena, merged);
    let fraction = crate::simplify::ratsimp::ratsimp(arena, merged);
    let fraction = crate::transforms::eval::eval(arena, fraction);
    let expanded = crate::transforms::expand::expand(arena, merged);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    let factored = {
        let post = crate::base::walk::post_order_ids(arena, merged);
        let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
        for &id in &post {
            let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
            let new = if matches!(arena.node(rebuilt), ExprNode::Add(_)) {
                crate::simplify::factor_terms::factor_terms(arena, rebuilt)
            } else {
                rebuilt
            };
            cache.insert(id, new);
        }
        let f = cache.get(&merged).copied().unwrap_or(merged);
        crate::transforms::eval::eval(arena, f)
    };
    for form in [merged, fraction, expanded, factored] {
        if form == e {
            continue;
        }
        let r = crate::transforms::integrate::integrate(arena, form, var);
        if !integral_failed(arena, r) {
            return r;
        }
    }
    if let Some(r) = integrate_trig_as_exp(arena, merged, var) {
        return r;
    }
    first
}

/// `∫ e d(var)` with `sin`/`cos` of linear arguments written as complex
/// exponentials (and the result's complex exponentials back as
/// `cos + i·sin`): `∫ eˣ·cos x/x dx = (Ei((1 + i)x) + Ei((1 − i)x))/2`,
/// which the integrator does not find in the trigonometric form (the
/// linear equation `y′ + y = cos x/x` was left with a formal integral).
/// `None` when `e` has no such factor, has other trigonometric functions,
/// or the exponential form does not integrate either.
fn integrate_trig_as_exp(arena: &mut Arena, e: ExprId, var: ExprId) -> Option<ExprId> {
    let mut has_circular = false;
    for id in crate::base::walk::post_order_ids(arena, e) {
        match arena.node(id).clone() {
            ExprNode::Sin(a) | ExprNode::Cos(a) => {
                let c = crate::transforms::solve::symbolic_poly_coeffs(arena, a, var)?;
                if c.len() > 2 {
                    return None;
                }
                has_circular = true;
            }
            ExprNode::Tan(_) => {
                return None;
            }
            _ => {}
        }
    }
    if !has_circular {
        return None;
    }
    let x = crate::simplify::rewrite::rewrite_as_exp(arena, e);
    let x = crate::transforms::expand::expand(arena, x);
    let x = crate::transforms::eval::eval(arena, x);
    let x = crate::simplify::powsimp::powsimp(arena, x);
    let x = crate::transforms::eval::eval(arena, x);
    let r = crate::transforms::integrate::integrate(arena, x, var);
    if integral_failed(arena, r) {
        return None;
    }
    let r = complex_exp_as_trig(arena, r, var);
    let r = crate::transforms::expand::expand(arena, r);
    Some(crate::transforms::eval::eval(arena, r))
}

/// `e^{(a + i·b)·x + c + i·d} → e^{a·x + c}·(cos(b·x + d) + i·sin(b·x + d))`
/// for every exponential whose argument is linear in `var` with constant
/// coefficients (real parts `a`, `c` and imaginary parts `b`, `d` from
/// [`real_imag_parts`]); other exponentials, and constant ones, are kept
/// (`rewrite_as_trig` turned the constant `e^{−1 − i}` into
/// `cos(i − 1) + i·sin(i − 1)`).
fn complex_exp_as_trig(arena: &mut Arena, e: ExprId, var: ExprId) -> ExprId {
    let i_unit = arena.i_unit;
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match *arena.node(rebuilt) {
            ExprNode::Exp(arg)
                if contains_sym_id(arena, arg, var)
                    && crate::base::walk::contains(arena, arg, i_unit) =>
            {
                exp_linear_as_trig(arena, arg, var).unwrap_or(rebuilt)
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    let out = cache.get(&e).copied().unwrap_or(e);
    crate::transforms::eval::eval(arena, out)
}

/// The rewrite of [`complex_exp_as_trig`] for one exponential `e^{arg}`.
fn exp_linear_as_trig(arena: &mut Arena, arg: ExprId, var: ExprId) -> Option<ExprId> {
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, arg, var)?;
    if coeffs.len() != 2 {
        return None;
    }
    let mut re = Vec::with_capacity(2);
    let mut im = Vec::with_capacity(2);
    for (k, &c) in coeffs.iter().enumerate() {
        if !crate::base::walk::free_symbols(arena, c).is_empty() {
            return None;
        }
        let (r, i) = real_imag_parts(arena, c);
        if !real_radical_form(arena, r) || !real_radical_form(arena, i) {
            return None;
        }
        let (r, i) = if k == 1 {
            (arena.mul(&[r, var]), arena.mul(&[i, var]))
        } else {
            (r, i)
        };
        re.push(r);
        im.push(i);
    }
    let re = arena.add(&re);
    let re = crate::transforms::eval::eval(arena, re);
    let im = arena.add(&im);
    let im = crate::transforms::eval::eval(arena, im);
    let envelope = arena.exp(re);
    let cos = arena.cos(im);
    let sin = arena.sin(im);
    let i_unit = arena.i_unit;
    let i_sin = arena.mul(&[i_unit, sin]);
    let osc = arena.add(&[cos, i_sin]);
    Some(arena.mul(&[envelope, osc]))
}

/// Is `e` zero as a rational function of its symbols and opaque
/// subexpressions (`ratsimp`)?  Catches `∂M/∂y − ∂N/∂x` of quotients
/// that `expand` leaves as unreduced fractions.
fn rational_zero(arena: &mut Arena, e: ExprId) -> bool {
    let r = crate::simplify::ratsimp::ratsimp(arena, e);
    let r = crate::transforms::eval::eval(arena, r);
    arena.is_zero_structural(r)
}

/// `expr / p` when `p` is a factor of every term of `expand(expr)`
/// (`None` otherwise).
fn divide_common_factor(arena: &mut Arena, expr: ExprId, p: ExprId) -> Option<ExprId> {
    let expanded = crate::transforms::expand::expand(arena, expr);
    let terms: Vec<ExprId> = match arena.node(expanded) {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![expanded],
    };
    let has_factor = |arena: &Arena, t: ExprId| -> bool {
        let is_power_of_p = |f: ExprId| -> bool {
            f == p
                || matches!(arena.node(f), ExprNode::Pow(b, e) if *b == p
                    && arena.as_num(*e).is_some_and(|r| r.is_integer() && r.is_positive()))
        };
        match arena.node(t) {
            ExprNode::Mul(fs) => fs.iter().any(|&f| is_power_of_p(f)),
            ExprNode::Neg(inner) => match arena.node(*inner) {
                ExprNode::Mul(fs) => fs.iter().any(|&f| is_power_of_p(f)),
                _ => is_power_of_p(*inner),
            },
            _ => is_power_of_p(t),
        }
    };
    if terms.is_empty() || !terms.iter().all(|&t| has_factor(arena, t)) {
        return None;
    }
    let quotient = arena.div(expanded, p);
    let quotient = crate::transforms::eval::eval(arena, quotient);
    let quotient = crate::transforms::expand::expand(arena, quotient);
    Some(crate::transforms::eval::eval(arena, quotient))
}

/// Did an antiderivative fail anywhere in `id`?  The integrator returns a
/// formal `Integral` node for the part it cannot do, not necessarily at
/// the top: `∫ 2·sin(y)/cos(y) dy` came back as `2·Integral(…)`, which
/// the top-level checks let through into an integrating factor (and a
/// `Piecewise` of nested integrals was returned as the "solution").
fn integral_failed(arena: &Arena, id: ExprId) -> bool {
    let mut stack = vec![id];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let node = arena.node(n);
        if matches!(node, ExprNode::Integral(..)) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// `exp(Σ cᵢ·ln(fᵢ))` → `Π fᵢ^{cᵢ}` for integrating factors.
///
/// Integrating factors are only needed up to a constant factor, so
/// `ln|f|` is treated as `ln f` (the sign is absorbed into the constant
/// of integration).  Terms that are not logarithms stay inside `exp`.
fn exp_of_log_sum(arena: &mut Arena, integral: ExprId) -> ExprId {
    let terms: Vec<ExprId> = match arena.node(integral).clone() {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![integral],
    };
    let mut factors: Vec<ExprId> = Vec::new();
    let mut leftover: Vec<ExprId> = Vec::new();
    for t in terms {
        let (coeff, term) = arena.as_coeff_term(t);
        let inner = match arena.node(term).clone() {
            ExprNode::Ln(inner) => Some(inner),
            _ => None,
        };
        match inner {
            Some(inner) => {
                let base = match arena.node(inner).clone() {
                    ExprNode::Abs(a) => a,
                    _ => inner,
                };
                if coeff.is_one() {
                    factors.push(base);
                } else {
                    let c_id = ode_ratio_to_expr(arena, &coeff);
                    factors.push(arena.pow(base, c_id));
                }
            }
            None => leftover.push(t),
        }
    }
    if factors.is_empty() {
        return arena.exp(integral);
    }
    if !leftover.is_empty() {
        let rest = arena.add(&leftover);
        factors.push(arena.exp(rest));
    }
    let prod = arena.mul(&factors);
    crate::transforms::eval::eval(arena, prod)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers for complex roots and undetermined coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a `Ratio<BigInt>` to an arena `ExprId`.
fn ode_ratio_to_expr(arena: &mut Arena, r: &Q) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(ExprNode::Num(nid))
}

/// Build the homogeneous solution in trigonometric form when the
/// characteristic equation `r² + b·r + c = 0` has complex roots.
///
/// For discriminant `disc = b² − 4c < 0`:
///   roots = α ± βi  where  α = −b/2,  β = √(−disc)/2
///   solution = exp(α·x)·(C1·cos(β·x) + C2·sin(β·x))
fn build_trig_homogeneous_solution(
    arena: &mut Arena,
    b: &Q,
    disc: &Q,
    var: ExprId,
) -> Option<OdeResult> {
    use num_traits::Zero;

    let c1 = arena.symbol("C1");
    let c2 = arena.symbol("C2");

    // α = −b/2
    let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
    let alpha = -(b.clone()) / &two_r;

    // β = √(−disc) / 2
    let neg_disc = -(disc.clone());
    let neg_disc_id = ode_ratio_to_expr(arena, &neg_disc);
    let half = arena.rational(1, 2);
    let sqrt_neg_disc = arena.pow(neg_disc_id, half);
    let two_id = arena.int(2);
    let beta_expr = arena.div(sqrt_neg_disc, two_id);
    let beta_expr = crate::transforms::eval::eval(arena, beta_expr);

    // β·x
    let beta_x = arena.mul(&[beta_expr, var]);
    let cos_bx = arena.cos(beta_x);
    let sin_bx = arena.sin(beta_x);

    let c1_cos = arena.mul(&[c1, cos_bx]);
    let c2_sin = arena.mul(&[c2, sin_bx]);
    let trig_part = arena.add(&[c1_cos, c2_sin]);

    let solution = if alpha.is_zero() {
        trig_part
    } else {
        let alpha_id = ode_ratio_to_expr(arena, &alpha);
        let alpha_x = arena.mul(&[alpha_id, var]);
        let exp_ax = arena.exp(alpha_x);
        arena.mul(&[exp_ax, trig_part])
    };

    Some(OdeResult {
        solution,
        constants: vec![c1, c2],
        branches: Vec::new(),
    })
}

/// Try to find a particular solution for `y'' + b·y' + c·y = rhs(x)`
/// when rhs is a trigonometric or exponential function.
///
/// Handles:
/// - `rhs = R·sin(ωx)` or `rhs = R·cos(ωx)` → undetermined coefficients
/// - `rhs = R·exp(rx)` → undetermined coefficients (with resonance handling)
fn try_undetermined_trig_exp(
    arena: &mut Arena,
    rhs: ExprId,
    b: &Q,
    c: &Q,
    var: ExprId,
) -> Option<ExprId> {
    use num_traits::Zero;

    let (coeff_r, term) = arena.as_coeff_term(rhs);
    let node = arena.node(term).clone();

    match node {
        ExprNode::Sin(inner) => {
            let (omega, constant) = extract_linear_numeric(arena, inner, var)?;
            if !constant.is_zero() {
                return None;
            }
            let zero_r = num_rational::Ratio::<num_bigint::BigInt>::zero();
            try_trig_particular(arena, &coeff_r, &zero_r, &omega, b, c, var)
        }
        ExprNode::Cos(inner) => {
            let (omega, constant) = extract_linear_numeric(arena, inner, var)?;
            if !constant.is_zero() {
                return None;
            }
            let zero_r = num_rational::Ratio::<num_bigint::BigInt>::zero();
            try_trig_particular(arena, &zero_r, &coeff_r, &omega, b, c, var)
        }
        ExprNode::Exp(inner) => {
            let (r_val, constant) = extract_linear_numeric(arena, inner, var)?;
            if !constant.is_zero() {
                return None;
            }
            try_exp_particular(arena, &coeff_r, &r_val, b, c, var)
        }
        _ => None,
    }
}

/// Extract the numeric coefficient and constant from a linear expression.
/// Returns `Some((a, b))` where `expr = a·var + b`, both rational.
fn extract_linear_numeric(arena: &Arena, expr: ExprId, var: ExprId) -> Option<(Q, Q)> {
    let poly = crate::poly::polybridge::expr_to_poly(arena, expr, var)?;
    if poly.degree()? != 1 {
        return None;
    }
    Some((poly.coeff(1), poly.coeff(0)))
}

/// Compute a particular solution for `y'' + b·y' + c·y = P·sin(ωx) + Q·cos(ωx)`
/// via the method of undetermined coefficients.
fn try_trig_particular(
    arena: &mut Arena,
    p: &Q,
    q: &Q,
    omega: &Q,
    b: &Q,
    c: &Q,
    var: ExprId,
) -> Option<ExprId> {
    use num_traits::Zero;

    let omega_sq = omega * omega;
    let d = c - &omega_sq; // c − ω²
    let bw = b * omega; // b·ω

    let det = &d * &d + &bw * &bw; // (c−ω²)² + (bω)²

    if !det.is_zero() {
        // Non-resonance: y_p = α·sin(ωx) + β·cos(ωx)
        let alpha = (&d * p + &bw * q) / &det;
        let beta = (&d * q - &bw * p) / &det;

        let omega_id = ode_ratio_to_expr(arena, omega);
        let omega_x = arena.mul(&[omega_id, var]);

        let mut terms = Vec::new();
        if !alpha.is_zero() {
            let alpha_id = ode_ratio_to_expr(arena, &alpha);
            let sin_wx = arena.sin(omega_x);
            terms.push(arena.mul(&[alpha_id, sin_wx]));
        }
        if !beta.is_zero() {
            let beta_id = ode_ratio_to_expr(arena, &beta);
            let cos_wx = arena.cos(omega_x);
            terms.push(arena.mul(&[beta_id, cos_wx]));
        }

        match terms.len() {
            0 => Some(arena.zero),
            1 => Some(terms[0]),
            _ => Some(arena.add(&terms)),
        }
    } else {
        // Resonance: d = 0 and bω = 0 ⟹ b = 0 and c = ω²
        if omega.is_zero() {
            return None;
        }
        let two_omega = num_rational::Ratio::from_integer(num_bigint::BigInt::from(2)) * omega;
        let alpha = q / &two_omega;
        let beta = -(p / &two_omega);

        let omega_id = ode_ratio_to_expr(arena, omega);
        let omega_x = arena.mul(&[omega_id, var]);

        let mut inner_terms = Vec::new();
        if !alpha.is_zero() {
            let alpha_id = ode_ratio_to_expr(arena, &alpha);
            let sin_wx = arena.sin(omega_x);
            inner_terms.push(arena.mul(&[alpha_id, sin_wx]));
        }
        if !beta.is_zero() {
            let beta_id = ode_ratio_to_expr(arena, &beta);
            let cos_wx = arena.cos(omega_x);
            inner_terms.push(arena.mul(&[beta_id, cos_wx]));
        }

        if inner_terms.is_empty() {
            Some(arena.zero)
        } else {
            let inner = if inner_terms.len() == 1 {
                inner_terms[0]
            } else {
                arena.add(&inner_terms)
            };
            Some(arena.mul(&[var, inner]))
        }
    }
}

/// Compute a particular solution for `y'' + b·y' + c·y = R·exp(r·x)`
/// via the method of undetermined coefficients.
fn try_exp_particular(
    arena: &mut Arena,
    coeff_r: &Q,
    r: &Q,
    b: &Q,
    c: &Q,
    var: ExprId,
) -> Option<ExprId> {
    use num_traits::Zero;

    // Characteristic value at r: r² + b·r + c
    let char_val = r * r + b * r + c;

    let r_id = ode_ratio_to_expr(arena, r);
    let rx = arena.mul(&[r_id, var]);
    let exp_rx = arena.exp(rx);

    if !char_val.is_zero() {
        // Non-resonance: y_p = A·exp(rx) where A = R / (r²+br+c)
        let a_val = coeff_r / &char_val;
        let a_id = ode_ratio_to_expr(arena, &a_val);
        Some(arena.mul(&[a_id, exp_rx]))
    } else {
        // r is a root of the characteristic equation.
        let deriv_val = num_rational::Ratio::from_integer(num_bigint::BigInt::from(2)) * r + b;
        if !deriv_val.is_zero() {
            // Single root: y_p = A·x·exp(rx) where A = R / (2r + b)
            let a_val = coeff_r / &deriv_val;
            let a_id = ode_ratio_to_expr(arena, &a_val);
            let x_exp = arena.mul(&[var, exp_rx]);
            Some(arena.mul(&[a_id, x_exp]))
        } else {
            // Double root: y_p = A·x²·exp(rx) where A = R / 2
            let two_r_val = num_rational::Ratio::from_integer(num_bigint::BigInt::from(2));
            let a_val = coeff_r / &two_r_val;
            let a_id = ode_ratio_to_expr(arena, &a_val);
            let two_id = arena.int(2);
            let x_sq = arena.pow(var, two_id);
            let x2_exp = arena.mul(&[x_sq, exp_rx]);
            Some(arena.mul(&[a_id, x2_exp]))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bernoulli equations: y' + P(x)·y = Q(x)·y^n  (n ≠ 0, 1)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve Bernoulli equations: y' + P(x)·y = Q(x)·y^n (n ≠ 0, 1).
///
/// Substitution v = y^(1−n) transforms to a first-order linear ODE:
///   v' + (1−n)·P(x)·v = (1−n)·Q(x)
/// Solve for v, then recover y = v^(1/(1−n)).
fn try_bernoulli(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying Bernoulli");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    // Must be first-order (no second derivatives)
    if expr_contains(arena, expr, d2y_dx2) {
        return None;
    }

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    use num_traits::Zero;

    let mut dy_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut p_x_terms: Vec<ExprId> = Vec::new();
    let mut q_x_terms: Vec<(ExprId, Q)> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            dy_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            return None; // Free terms not allowed in standard Bernoulli
        } else if let Some(px) = extract_coeff_of_func(arena, child, func, func_sym, var_sym) {
            p_x_terms.push(px);
        } else if let Some((qx, n)) = extract_bernoulli_term(arena, child, func, func_sym, var_sym)
        {
            q_x_terms.push((qx, n));
        } else {
            return None;
        }
    }

    if dy_coeff.is_zero() || q_x_terms.is_empty() {
        return None;
    }

    // All y^n terms must share the same exponent n ≠ 0, 1
    let n_val = q_x_terms[0].1.clone();
    if n_val.is_zero() || n_val.is_one() {
        return None;
    }
    for (_, n) in &q_x_terms[1..] {
        if *n != n_val {
            return None;
        }
    }

    // Normalize by dy_coeff
    let inv_dy = num_rational::Ratio::<num_bigint::BigInt>::one() / &dy_coeff;
    let inv_dy_id = ode_ratio_to_expr(arena, &inv_dy);

    let p_raw = if p_x_terms.is_empty() {
        arena.zero
    } else if p_x_terms.len() == 1 {
        p_x_terms[0]
    } else {
        arena.add(&p_x_terms)
    };
    let p_x = if dy_coeff.is_one() {
        p_raw
    } else {
        let s = arena.mul(&[inv_dy_id, p_raw]);
        crate::transforms::eval::eval(arena, s)
    };

    // Q_raw·y^n appears on the LHS: y' + P·y + Q_raw·y^n = 0
    // So actual Q in y' + P·y = Q·y^n is −Q_raw
    let q_sum: Vec<ExprId> = q_x_terms.iter().map(|(qx, _)| *qx).collect();
    let q_raw = if q_sum.len() == 1 {
        q_sum[0]
    } else {
        arena.add(&q_sum)
    };
    let neg_q_raw = arena.neg(q_raw);
    let q_x = if dy_coeff.is_one() {
        neg_q_raw
    } else {
        let s = arena.mul(&[inv_dy_id, neg_q_raw]);
        crate::transforms::eval::eval(arena, s)
    };

    if contains_sym(arena, p_x, func_sym) || contains_sym(arena, q_x, func_sym) {
        return None;
    }

    // Substitution: v = y^(1−n)
    // Transformed ODE: v' + (1−n)·P·v = (1−n)·Q
    let one_minus_n = num_rational::Ratio::<num_bigint::BigInt>::one() - &n_val;
    let one_minus_n_id = ode_ratio_to_expr(arena, &one_minus_n);

    let new_p = arena.mul(&[one_minus_n_id, p_x]);
    let new_p = crate::transforms::eval::eval(arena, new_p);
    let new_q = arena.mul(&[one_minus_n_id, q_x]);
    let new_q = crate::transforms::eval::eval(arena, new_q);

    // Build linear ODE for v: v' + new_p·v − new_q = 0
    let v = arena.symbol("__v");
    let dv = arena.intern(ExprNode::Derivative(v, var));
    let pv = arena.mul(&[new_p, v]);
    let neg_new_q = arena.neg(new_q);
    let linear_expr = arena.add(&[dv, pv, neg_new_q]);

    let v_sym = match arena.node(v) {
        ExprNode::Symbol(sid) => *sid,
        _ => return None,
    };

    // Solve the linear ODE for v (fall back to simple separable if P=0)
    let v_result = try_first_order_linear_general(arena, linear_expr, v, var, v_sym, var_sym)
        .or_else(|| try_simple_separable(arena, linear_expr, v, var, v_sym, var_sym))?;

    // Recover y = v^(1/(1−n)); for an even denominator (`y⁻² = v`) the
    // negative root `−v^(1/(1−n))` is a family too (before 0.30 it was
    // missing: `y′ = y³` with `y(0) = −1` had no solution).
    let inv_one_minus_n = num_rational::Ratio::<num_bigint::BigInt>::one() / &one_minus_n;
    let inv_id = ode_ratio_to_expr(arena, &inv_one_minus_n);
    let solution = arena.pow(v_result.solution, inv_id);
    let solution = crate::transforms::eval::eval(arena, solution);
    let two = num_bigint::BigInt::from(2);
    let branches = if inv_one_minus_n.denom() % &two == num_bigint::BigInt::from(0) {
        let negative = arena.neg(solution);
        vec![crate::transforms::eval::eval(arena, negative)]
    } else if !inv_one_minus_n.is_integer() && inv_one_minus_n.numer().abs().is_one() {
        // `1 − n` an odd integer `m` (`y′ = y/x + y⁴/x²`, `y⁻³ = v`): the
        // principal `v^{1/m}` is real only for `v > 0`, and `−(−v)^{1/m}`
        // (`(−(−v)^{1/m})^m = v`) is the real solution where `v < 0`.
        // Before, initial values `y(x₀) < 0` had no real family.
        let neg_v = arena.neg(v_result.solution);
        let root = arena.pow(neg_v, inv_id);
        let negative = arena.neg(root);
        vec![crate::transforms::eval::eval(arena, negative)]
    } else {
        Vec::new()
    };

    Some(OdeResult {
        solution,
        constants: v_result.constants,
        branches,
    })
}

/// Try to extract a Bernoulli term Q(x)·y^n from an expression.
///
/// Returns `Some((Q(x), n))` where Q(x) is free of y and n is a rational
/// constant.
fn extract_bernoulli_term(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<(ExprId, Q)> {
    // Case 1: expr is y^n
    if let ExprNode::Pow(base, exp) = arena.node(expr).clone()
        && base == func
    {
        let n = arena.as_num(exp)?.clone();
        return Some((arena.one, n));
    }

    // Case 2: expr is Mul([..., y^n, ...])
    if let ExprNode::Mul(ref factors) = arena.node(expr).clone() {
        let mut yn_idx = None;
        let mut yn_exp = None;
        for (i, &f) in factors.iter().enumerate() {
            if let ExprNode::Pow(base, exp) = arena.node(f).clone()
                && base == func
                && let Some(n) = arena.as_num(exp)
            {
                yn_idx = Some(i);
                yn_exp = Some(n.clone());
                break;
            }
        }
        if let (Some(idx), Some(n)) = (yn_idx, yn_exp) {
            let mut other: Vec<ExprId> = Vec::new();
            for (i, &f) in factors.iter().enumerate() {
                if i != idx {
                    if contains_sym(arena, f, func_sym) {
                        return None;
                    }
                    other.push(f);
                }
            }
            let qx = match other.len() {
                0 => arena.one,
                1 => other[0],
                _ => arena.mul(&other),
            };
            return Some((qx, n));
        }
    }

    // Case 3: numeric coefficient × something
    {
        let (coeff, term) = arena.as_coeff_term(expr);
        if !coeff.is_one()
            && term != expr
            && let Some((inner_qx, n)) =
                extract_bernoulli_term(arena, term, func, func_sym, _var_sym)
        {
            let coeff_id = ode_ratio_to_expr(arena, &coeff);
            let qx = arena.mul(&[coeff_id, inner_qx]);
            return Some((qx, n));
        }
    }

    // Case 4: Neg(something)
    if let ExprNode::Neg(inner) = arena.node(expr).clone()
        && let Some((inner_qx, n)) = extract_bernoulli_term(arena, inner, func, func_sym, _var_sym)
    {
        let neg_qx = arena.neg(inner_qx);
        return Some((neg_qx, n));
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Euler-Cauchy equations: a·x²·y'' + b·x·y' + c·y = 0
// ═══════════════════════════════════════════════════════════════════════════

/// Solve Euler-Cauchy equations: a·x²·y'' + b·x·y' + c·y = 0.
///
/// The characteristic equation is a·r(r−1) + b·r + c = 0, equivalently
/// a·r² + (b−a)·r + c = 0.
///
/// - Distinct real roots r₁, r₂: y = C1·x^r₁ + C2·x^r₂
/// - Repeated root r: y = (C1 + C2·ln(x))·x^r
/// - Complex roots α ± βi: y = x^α·(C1·cos(β·ln(x)) + C2·sin(β·ln(x)))
///
/// A forcing term `g(x)` is handled by `x = eᵗ` (for `x > 0`, like the
/// homogeneous solutions): `a·Y″ + (b − a)·Y′ + c·Y = g(eᵗ)` has constant
/// coefficients, and its solution with `t = ln x` is the answer.  Before,
/// every forced Cauchy–Euler equation was refused.
fn try_euler_cauchy(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying Euler-Cauchy");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    if !expr_contains(arena, expr, d2y_dx2) {
        return None;
    }

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    use num_traits::Zero;
    let mut a_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut b_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut c_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();

    let two_id = arena.int(2);
    let x_sq = arena.pow(var, two_id);
    let mut forcing: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == func {
            c_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            forcing.push(child);
        } else if let ExprNode::Mul(ref factors) = arena.node(term).clone() {
            let has_d2 = factors.contains(&d2y_dx2);
            let has_d1 = factors.contains(&dy_dx);
            let has_xsq = factors.contains(&x_sq);
            let has_xvar = factors.contains(&var);

            // Collect factors that are not x²/x/y''/y'
            let other: Vec<ExprId> = factors
                .iter()
                .copied()
                .filter(|&f| f != d2y_dx2 && f != dy_dx && f != x_sq && f != var)
                .collect();
            for &of in &other {
                if contains_sym(arena, of, func_sym) || contains_sym(arena, of, var_sym) {
                    return None;
                }
            }
            let extra = if other.is_empty() {
                num_rational::Ratio::<num_bigint::BigInt>::from_integer(1.into())
            } else {
                let e = if other.len() == 1 {
                    other[0]
                } else {
                    arena.mul(&other)
                };
                arena.as_num(e)?.clone()
            };

            if has_d2 && has_xsq && !has_d1 && !has_xvar {
                a_coeff += &coeff * &extra;
            } else if has_d1 && has_xvar && !has_d2 && !has_xsq {
                b_coeff += &coeff * &extra;
            } else {
                return None;
            }
        } else {
            return None;
        }
    }

    if a_coeff.is_zero() {
        return None;
    }

    if !forcing.is_empty() {
        return euler_cauchy_forced(arena, &a_coeff, &b_coeff, &c_coeff, &forcing, var, var_sym);
    }

    // Characteristic equation: a·r² + (b−a)·r + c = 0
    // Normalize: r² + ((b−a)/a)·r + c/a = 0
    let p = (&b_coeff - &a_coeff) / &a_coeff;
    let q = &c_coeff / &a_coeff;

    let four = num_rational::Ratio::<num_bigint::BigInt>::from_integer(4.into());
    let disc = &p * &p - &four * &q;

    let c1 = arena.symbol("C1");
    let c2 = arena.symbol("C2");

    if disc.is_positive() {
        // Distinct real roots — solve r² + p·r + q = 0
        let r_var = arena.symbol("__r");
        let r_sq = arena.pow(r_var, two_id);
        let p_id = ode_ratio_to_expr(arena, &p);
        let q_id = ode_ratio_to_expr(arena, &q);
        let p_r = arena.mul(&[p_id, r_var]);
        let char_eq = arena.add(&[r_sq, p_r, q_id]);
        let roots = known_roots(arena, char_eq, r_var);

        if roots.len() >= 2 {
            let r1 = roots[0].value;
            let r2 = roots[1].value;
            let x_r1 = arena.pow(var, r1);
            let x_r2 = arena.pow(var, r2);
            let t1 = arena.mul(&[c1, x_r1]);
            let t2 = arena.mul(&[c2, x_r2]);
            let solution = arena.add(&[t1, t2]);
            Some(OdeResult {
                solution,
                constants: vec![c1, c2],
                branches: Vec::new(),
            })
        } else {
            None
        }
    } else if disc.is_zero() {
        // Repeated root: r = −p/2
        let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
        let r = -(&p) / &two_r;
        let r_id = ode_ratio_to_expr(arena, &r);
        let x_r = arena.pow(var, r_id);
        let ln_x = arena.ln(var);
        let c2_ln = arena.mul(&[c2, ln_x]);
        let inner = arena.add(&[c1, c2_ln]);
        let solution = arena.mul(&[inner, x_r]);
        Some(OdeResult {
            solution,
            constants: vec![c1, c2],
            branches: Vec::new(),
        })
    } else {
        // Complex roots α ± βi: α = −p/2, β = √(−disc)/2
        let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
        let alpha = -(&p) / &two_r;
        let neg_disc = -disc;
        let neg_disc_id = ode_ratio_to_expr(arena, &neg_disc);
        let half = arena.rational(1, 2);
        let sqrt_neg_disc = arena.pow(neg_disc_id, half);
        let two_expr = arena.int(2);
        let beta = arena.div(sqrt_neg_disc, two_expr);
        let beta = crate::transforms::eval::eval(arena, beta);

        let ln_x = arena.ln(var);
        let beta_ln_x = arena.mul(&[beta, ln_x]);
        let cos_part = arena.cos(beta_ln_x);
        let sin_part = arena.sin(beta_ln_x);
        let c1_cos = arena.mul(&[c1, cos_part]);
        let c2_sin = arena.mul(&[c2, sin_part]);
        let trig_part = arena.add(&[c1_cos, c2_sin]);

        let solution = if alpha.is_zero() {
            trig_part
        } else {
            let alpha_id = ode_ratio_to_expr(arena, &alpha);
            let x_alpha = arena.pow(var, alpha_id);
            arena.mul(&[x_alpha, trig_part])
        };

        Some(OdeResult {
            solution,
            constants: vec![c1, c2],
            branches: Vec::new(),
        })
    }
}

/// `a·x²·y″ + b·x·y′ + c·y + Σ forcing = 0` by `x = eᵗ` (see
/// [`try_euler_cauchy`]).
fn euler_cauchy_forced(
    arena: &mut Arena,
    a: &Q,
    b: &Q,
    c: &Q,
    forcing: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    euler_cauchy_by_exp(arena, &[c.clone(), b - a, a.clone()], forcing, var, var_sym)
}

/// Cauchy–Euler of any order ≥ 3, `Σₖ aₖ·xᵏ·y⁽ᵏ⁾ + g(x) = 0` with rational
/// `aₖ`, by `x = eᵗ`: `xᵏ·y⁽ᵏ⁾ = D(D − 1)⋯(D − k + 1)·Y` with `D = d/dt`,
/// a constant-coefficient equation in `t` (solutions for `x > 0`, as for
/// order 2).  SymPy's `nth_linear_euler_eq_*`; order 3 was refused.
fn try_euler_cauchy_n(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    let euler = euler_operator(arena, expr, func, var, func_sym, 3)?;
    euler_cauchy_by_exp(arena, &euler.coeffs, &euler.forcing, var, var_sym)
}

/// A Cauchy–Euler equation `Σₖ aₖ·xᵏ·y⁽ᵏ⁾ + g(x) = 0` after `x = eᵗ`:
/// `Σⱼ coeffsⱼ·Y⁽ʲ⁾(t) + g = 0`.
struct EulerOperator {
    /// The coefficients of `Dʲ` (`D = d/dt`) of `Σ aₖ·D(D − 1)⋯(D − k + 1)`.
    coeffs: Vec<Q>,
    /// The forcing terms (in `x`).
    forcing: Vec<ExprId>,
}

/// The signed Stirling numbers of the first kind `s(k, j)`, `j ≤ k`: the
/// coefficients of `D(D − 1)⋯(D − k + 1) = Σⱼ s(k, j)·Dʲ`.
fn falling_operator(k: usize) -> Vec<Q> {
    let mut falling = vec![Q::from_integer(1.into())];
    for i in 0..k {
        let shift = Q::from_integer((i as i64).into());
        let mut next = vec![Q::from_integer(0.into()); falling.len() + 1];
        for (j, v) in falling.iter().enumerate() {
            next[j + 1] += v;
            next[j] -= v * &shift;
        }
        falling = next;
    }
    falling
}

/// Recognise a Cauchy–Euler equation of order `≥ min_order` with rational
/// `aₖ` (see [`EulerOperator`]).
fn euler_operator(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    min_order: usize,
) -> Option<EulerOperator> {
    let chain = derivative_chain(arena, func, var);
    let order = (1..chain.len())
        .rev()
        .find(|&k| expr_contains(arena, expr, chain[k]))?;
    if order < min_order {
        return None;
    }
    let ExprNode::Add(children) = arena.node(expr).clone() else {
        return None;
    };
    let mut a = vec![Q::from_integer(0.into()); order + 1];
    let mut forcing = Vec::new();
    for &child in children.iter() {
        if !contains_sym(arena, child, func_sym) {
            forcing.push(child);
            continue;
        }
        let (coeff, term) = arena.as_coeff_term(child);
        let factors: Vec<ExprId> = match arena.node(term) {
            ExprNode::Mul(fs) => fs.to_vec(),
            _ => vec![term],
        };
        let mut k_deriv: Option<usize> = None;
        let mut x_power: i64 = 0;
        for f in factors {
            if let Some(k) = chain.iter().position(|&d| d == f) {
                if k_deriv.replace(k).is_some() {
                    return None;
                }
            } else if f == var {
                x_power += 1;
            } else if let ExprNode::Pow(b, e) = *arena.node(f)
                && b == var
                && let Some(p) = arena.as_num(e).filter(|p| p.is_integer())
            {
                x_power += num_traits::ToPrimitive::to_i64(&p.to_integer())?;
            } else {
                return None;
            }
        }
        let k = k_deriv?;
        if x_power != i64::try_from(k).ok()? {
            return None;
        }
        a[k] += coeff;
    }
    if num_traits::Zero::is_zero(&a[order]) {
        return None;
    }
    // Σₖ aₖ·D(D − 1)⋯(D − k + 1) as coefficients of Dʲ.
    let mut cd = vec![Q::from_integer(0.into()); order + 1];
    for (k, ak) in a.iter().enumerate() {
        for (j, v) in falling_operator(k).iter().enumerate() {
            cd[j] += ak * v;
        }
    }
    Some(EulerOperator {
        coeffs: cd,
        forcing,
    })
}

/// [`linear_cc_ivp_lagrange`] for a Cauchy–Euler equation with all
/// conditions at one point `x₀ > 0`: with `x = x₀·eᵗ` the equation is
/// `Σ cdⱼ·Y⁽ʲ⁾(t) + g(x₀eᵗ) = 0` and the conditions become
/// `Y⁽ʲ⁾(0) = Wⱼ` through `x₀ᵏ·y⁽ᵏ⁾(x₀) = Σⱼ s(k, j)·Wⱼ`; the
/// constant-coefficient fit, mapped back by `t = ln(x/x₀)`.  The generic fit
/// over `RootOf(m⁴ + m + 1, k)` took 5 s (debug build) for
/// `x⁴y⁗ + 6x³y‴ + 7x²y″ + 2xy′ + y = x²`.
fn euler_ivp_lagrange(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    x0: ExprId,
    values: &[ExprId],
) -> Option<ExprId> {
    let ExprNode::Symbol(func_sym) = *arena.node(func) else {
        return None;
    };
    let ExprNode::Symbol(var_sym) = *arena.node(var) else {
        return None;
    };
    if !is_real_constant(arena, x0)
        || !crate::transforms::evalf::evalf_complex64(arena, x0).is_ok_and(|z| z.re > 0.0)
    {
        return None;
    }
    let euler = euler_operator(arena, expr, func, var, func_sym, 2)?;
    let n = euler.coeffs.len() - 1;
    if values.len() != n {
        return None;
    }
    let t = arena.symbol("__t_euler_ivp");
    let big_y = arena.symbol("__Y_euler_ivp");
    // Σ cdⱼ·Y⁽ʲ⁾ + g(x₀·eᵗ)
    let mut parts = Vec::with_capacity(n + 1 + euler.forcing.len());
    let mut d = big_y;
    for (j, c) in euler.coeffs.iter().enumerate() {
        if j > 0 {
            d = arena.intern(ExprNode::Derivative(d, t));
        }
        if !num_traits::Zero::is_zero(c) {
            let c = ode_ratio_to_expr(arena, c);
            parts.push(arena.mul(&[c, d]));
        }
    }
    let ln_x = arena.ln(var);
    let ln_x0 = arena.ln(x0);
    let ln_shifted = arena.add(&[ln_x0, t]);
    for &f in &euler.forcing {
        let g = crate::transforms::subs::subs(arena, f, ln_x, ln_shifted);
        let post = crate::base::walk::post_order_ids(arena, g);
        let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
        for &id in &post {
            let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
            let new = match *arena.node(rebuilt) {
                ExprNode::Pow(b, e) if b == var && !contains_sym(arena, e, var_sym) => {
                    let scale = arena.pow(x0, e);
                    let et = arena.mul(&[e, t]);
                    let growth = arena.exp(et);
                    arena.mul(&[scale, growth])
                }
                _ => rebuilt,
            };
            cache.insert(id, new);
        }
        let g = cache.get(&g).copied().unwrap_or(g);
        let exp_t = arena.exp(t);
        let x_of_t = arena.mul(&[x0, exp_t]);
        let g = crate::transforms::subs::subs(arena, g, var, x_of_t);
        let g = crate::transforms::eval::eval(arena, g);
        let g = crate::transforms::expand::expand(arena, g);
        let g = crate::transforms::eval::eval(arena, g);
        if contains_sym(arena, g, var_sym) {
            return None;
        }
        parts.push(g);
    }
    let ode_t = arena.add(&parts);
    let ode_t = crate::transforms::eval::eval(arena, ode_t);
    // W_k = x₀ᵏ·vₖ − Σ_{j<k} s(k, j)·Wⱼ
    let mut w: Vec<ExprId> = Vec::with_capacity(n);
    for (k, &v) in values.iter().enumerate() {
        let kk = arena.int(i64::try_from(k).ok()?);
        let x0k = arena.pow(x0, kk);
        let mut acc = vec![arena.mul(&[x0k, v])];
        for (j, s) in falling_operator(k).iter().enumerate().take(k) {
            if num_traits::Zero::is_zero(s) {
                continue;
            }
            let c = ode_ratio_to_expr(arena, &-s.clone());
            acc.push(arena.mul(&[c, w[j]]));
        }
        let wk = arena.add(&acc);
        w.push(crate::transforms::eval::eval(arena, wk));
    }
    let zero = arena.zero;
    let sol_t = linear_cc_ivp_lagrange(arena, ode_t, big_y, t, zero, &w)?;
    // t = ln(x/x₀)
    let ratio = arena.div(var, x0);
    let back = arena.ln(ratio);
    let sol = crate::transforms::subs::subs(arena, sol_t, t, back);
    let sol = crate::transforms::eval::eval(arena, sol);
    let sol = exp_log_to_power(arena, sol);
    if contains_sym(arena, sol, func_sym) || crate::base::walk::contains(arena, sol, t) {
        return None;
    }
    Some(sol)
}

/// `Σⱼ cdⱼ·Y⁽ʲ⁾(t) + Σ forcing(x = eᵗ) = 0` solved and mapped back to `x`
/// (`e^{k·t + m} → xᵏ·eᵐ`, `t → ln x`): the Cauchy–Euler equation whose
/// operator in `D = d/dt` has the coefficients `cd`.
fn euler_cauchy_by_exp(
    arena: &mut Arena,
    cd: &[Q],
    forcing: &[ExprId],
    var: ExprId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    let order = cd.len().checked_sub(1)?;
    let t = arena.symbol("__t");
    let big_y = arena.symbol("__Y");
    let (t_sym, y_sym) = match (arena.node(t), arena.node(big_y)) {
        (ExprNode::Symbol(ts), ExprNode::Symbol(ys)) => (*ts, *ys),
        _ => return None,
    };
    let exp_t = arena.exp(t);
    let ln_x = arena.ln(var);
    let mut parts = Vec::with_capacity(cd.len() + forcing.len());
    let mut d = big_y;
    for (j, c) in cd.iter().enumerate() {
        if j > 0 {
            d = arena.intern(ExprNode::Derivative(d, t));
        }
        if num_traits::Zero::is_zero(c) {
            continue;
        }
        let c_id = ode_ratio_to_expr(arena, c);
        parts.push(arena.mul(&[c_id, d]));
    }
    for &f in forcing {
        // ln x = t first (x > 0), then xᵖ = e^{p·t} (`√x` became
        // `(eᵗ)^(1/2)`, which is not `e^{t/2}` for complex `t`, and the
        // forcing was not recognised), then x = eᵗ.
        let g = crate::transforms::subs::subs(arena, f, ln_x, t);
        let post = crate::base::walk::post_order_ids(arena, g);
        let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
        for &id in &post {
            let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
            let new = match *arena.node(rebuilt) {
                ExprNode::Pow(b, e) if b == var && !contains_sym(arena, e, var_sym) => {
                    let et = arena.mul(&[e, t]);
                    arena.exp(et)
                }
                _ => rebuilt,
            };
            cache.insert(id, new);
        }
        let g = cache.get(&g).copied().unwrap_or(g);
        let g = crate::transforms::subs::subs(arena, g, var, exp_t);
        parts.push(crate::transforms::eval::eval(arena, g));
    }
    let ode_t = arena.add(&parts);
    let ode_t = crate::transforms::eval::eval(arena, ode_t);
    if contains_sym(arena, ode_t, var_sym) {
        return None;
    }
    let res = dsolve(arena, ode_t, big_y, t)?;
    if contains_sym(arena, res.solution, y_sym)
        || res.constants.len() != order
        || crate::base::walk::has_unevaluated(arena, res.solution)
    {
        return None;
    }
    // e^{k·t + m} = x^k·e^m, then t = ln x.
    let mut replacements: Vec<(ExprId, ExprId)> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, res.solution) {
        let ExprNode::Exp(arg) = arena.node(id).clone() else {
            continue;
        };
        let arg = crate::transforms::expand::expand(arena, arg);
        let arg = crate::transforms::eval::eval(arena, arg);
        let Some(coeffs) = arena.coefficients_of(arg, t) else {
            continue;
        };
        if coeffs.len() != 2 || contains_sym(arena, coeffs[1], t_sym) {
            continue;
        }
        let x_k = arena.pow(var, coeffs[1]);
        let rest = arena.exp(coeffs[0]);
        let r = arena.mul(&[x_k, rest]);
        let r = crate::transforms::eval::eval(arena, r);
        replacements.push((id, r));
    }
    let solution = crate::transforms::subs::subs_map(arena, res.solution, &replacements);
    let solution = crate::transforms::subs::subs(arena, solution, t, ln_x);
    let solution = crate::transforms::eval::eval(arena, solution);
    if contains_sym(arena, solution, t_sym) {
        return None;
    }
    Some(OdeResult {
        solution,
        constants: res.constants,
        branches: Vec::new(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Variation of parameters: y'' + p·y' + q·y = g(x)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve y'' + p·y' + q·y = g(x) via variation of parameters.
///
/// Used as a fallback when undetermined coefficients fails (e.g., forcing
/// function is tan(x), sec(x), etc.).
///
/// Given homogeneous solutions y₁, y₂:
/// - Wronskian W computed via differentiation (with Abel's identity fallback)
/// - Particular: y_p = −y₁·∫(y₂·g/W)dx + y₂·∫(y₁·g/W)dx
fn try_variation_of_parameters(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying variation of parameters");

    use num_traits::Zero;

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut a_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut b_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut c_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut f_terms: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == d2y_dx2 {
            a_coeff += coeff;
        } else if term == dy_dx {
            b_coeff += coeff;
        } else if term == func {
            c_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            f_terms.push(child);
        } else {
            return None;
        }
    }

    if a_coeff.is_zero() || f_terms.is_empty() {
        return None;
    }

    let b = &b_coeff / &a_coeff;
    let c = &c_coeff / &a_coeff;

    // g(x) = −f_sum / a
    let f_sum = if f_terms.len() == 1 {
        f_terms[0]
    } else {
        arena.add(&f_terms)
    };
    let neg_f = arena.neg(f_sum);
    let a_id = ode_ratio_to_expr(arena, &a_coeff);
    let g_x = arena.div(neg_f, a_id);
    let g_x = crate::transforms::eval::eval(arena, g_x);

    // Solve homogeneous equation
    let homo = solve_characteristic_equation(arena, b.clone(), c.clone(), var)?;
    let (y1, y2) = extract_fundamental_solutions(arena, homo.solution, &homo.constants)?;

    // Wronskian via symbolic differentiation
    let y1_prime = crate::transforms::diff::diff(arena, y1, var);
    let y2_prime = crate::transforms::diff::diff(arena, y2, var);
    let w_term1 = arena.mul(&[y1, y2_prime]);
    let w_term2 = arena.mul(&[y2, y1_prime]);
    let wronskian = arena.sub(w_term1, w_term2);
    let wronskian = crate::transforms::eval::eval(arena, wronskian);
    let wronskian = crate::transforms::expand::expand(arena, wronskian);
    let wronskian = crate::transforms::eval::eval(arena, wronskian);

    // If W still depends on var (e.g. cos²+sin² unsimplified), use Abel's
    // identity: W(x) = W(0)·exp(−b·x).
    let wronskian = if contains_sym(arena, wronskian, var_sym) {
        let w0 = crate::transforms::subs::subs(arena, wronskian, var, arena.zero);
        let w0 = crate::transforms::eval::eval(arena, w0);
        if w0 == arena.zero {
            return None;
        }
        if b.is_zero() {
            w0
        } else {
            let neg_b_id = ode_ratio_to_expr(arena, &(-b.clone()));
            let neg_bx = arena.mul(&[neg_b_id, var]);
            let exp_nbx = arena.exp(neg_bx);
            arena.mul(&[w0, exp_nbx])
        }
    } else {
        wronskian
    };

    if wronskian == arena.zero {
        return None;
    }

    // y_p = −y₁·∫(y₂·g/W)dx + y₂·∫(y₁·g/W)dx
    let y2_g = arena.mul(&[y2, g_x]);
    let integrand1 = arena.div(y2_g, wronskian);
    let integrand1 = crate::transforms::eval::eval(arena, integrand1);
    let integral1 = integrate_forms(arena, integrand1, var);
    if integral_failed(arena, integral1) {
        return None;
    }

    let y1_g = arena.mul(&[y1, g_x]);
    let integrand2 = arena.div(y1_g, wronskian);
    let integrand2 = crate::transforms::eval::eval(arena, integrand2);
    let integral2 = integrate_forms(arena, integrand2, var);
    if integral_failed(arena, integral2) {
        return None;
    }

    let term1 = arena.mul(&[y1, integral1]);
    let neg_term1 = arena.neg(term1);
    let term2 = arena.mul(&[y2, integral2]);
    let y_p = arena.add(&[neg_term1, term2]);
    let y_p = crate::transforms::eval::eval(arena, y_p);

    let solution = arena.add(&[homo.solution, y_p]);
    let solution = crate::transforms::eval::eval(arena, solution);

    Some(OdeResult {
        solution,
        constants: homo.constants,
        branches: Vec::new(),
    })
}

/// Highest order [`try_linear_cc_green`] handles (one antiderivative per
/// root and multiplicity).
const MAX_GREEN_ORDER: usize = 6;

/// A root of a characteristic polynomial for [`try_linear_cc_green`]: its
/// multiplicity `mult` and the cofactor `p(s)/(s − root)^mult` (an
/// expression in `s`).
struct GreenRoot {
    root: ExprId,
    mult: usize,
    cofactor: ExprId,
}

/// The roots of the rational polynomial `p` with their cofactors, when
/// every irreducible factor of `p` over ℚ has degree 1 or 2 (rational
/// roots, real quadratic surds, complex quadratic pairs); `None` otherwise.
fn green_roots(arena: &mut Arena, p: &crate::poly::Poly, s: ExprId) -> Option<Vec<GreenRoot>> {
    let (content, factors) = p.factor_over_z();
    let factor_exprs: Vec<ExprId> = factors
        .iter()
        .map(|(g, _)| crate::poly::polybridge::poly_to_expr(arena, g, s))
        .collect();
    let mut out = Vec::new();
    for (fi, (g, mult)) in factors.iter().enumerate() {
        let mult = *mult as usize;
        let lead = g.leading_coeff()?.clone();
        let roots: Vec<ExprId> = match g.degree()? {
            1 => vec![ode_ratio_to_expr(arena, &(-g.coeff(0) / &lead))],
            2 => {
                let (a, b, c) = (lead.clone(), g.coeff(1), g.coeff(0));
                let disc = &b * &b - Q::from_integer(4.into()) * &a * &c;
                let half = Q::new(1.into(), 2.into());
                let mag = ode_ratio_to_expr(arena, &disc.abs());
                let half_id = ode_ratio_to_expr(arena, &half);
                let sqrt = arena.pow(mag, half_id);
                let sqrt = if disc.is_negative() {
                    let i_unit = arena.i_unit;
                    arena.mul(&[i_unit, sqrt])
                } else {
                    sqrt
                };
                let two_a = ode_ratio_to_expr(arena, &(Q::from_integer(2.into()) * &a));
                let neg_b = ode_ratio_to_expr(arena, &(-b));
                let neg_sqrt = arena.neg(sqrt);
                [sqrt, neg_sqrt]
                    .into_iter()
                    .map(|d| {
                        let num = arena.add(&[neg_b, d]);
                        let r = arena.div(num, two_a);
                        let r = crate::transforms::expand::expand(arena, r);
                        crate::transforms::eval::eval(arena, r)
                    })
                    .collect()
            }
            _ => return None,
        };
        for (ri, &root) in roots.iter().enumerate() {
            // p(s)/(s − r)^m = content · lead^m · (s − r′)^m · Π_{other} h(s)^{m_h}
            let mut parts = vec![ode_ratio_to_expr(arena, &content)];
            let m_id = arena.int(i64::try_from(mult).ok()?);
            let lead_id = ode_ratio_to_expr(arena, &lead);
            parts.push(arena.pow(lead_id, m_id));
            if roots.len() == 2 {
                let other = roots[1 - ri];
                let lin = arena.sub(s, other);
                parts.push(arena.pow(lin, m_id));
            }
            for (hi, (_, mh)) in factors.iter().enumerate() {
                if hi != fi {
                    let e = arena.int(i64::from(*mh));
                    parts.push(arena.pow(factor_exprs[hi], e));
                }
            }
            let cofactor = arena.mul(&parts);
            out.push(GreenRoot {
                root,
                mult,
                cofactor,
            });
        }
    }
    Some(out)
}

/// `Σ aₖ·y⁽ᵏ⁾ = g(x)` with rational `aₖ` (order `2 ≤ n ≤ 6`) and a forcing
/// term outside undetermined coefficients, by the Green's function of
/// `p(D)`: with `1/p(s) = Σ_r Σ_{j≤m_r} A_{r,j}/(s − r)ʲ`,
/// `yₚ(x) = Σ A_{r,j}/(j−1)! · ∫ˣ (x − t)^{j−1}·e^{r(x−t)}·g(t) dt`, the
/// binomial expansion of `(x − t)^{j−1}` leaving the antiderivatives
/// `∫ tⁱ·e^{−rt}·g(t) dt` (trigonometric and hyperbolic functions in `g`
/// written as exponentials).  The roots are those of factors of degree ≤ 2
/// over ℚ, complex ones as complex exponentials: `y″ + y = eˣ/x` gets
/// `Ei((1 ∓ i)x)` (a real function written with `i`), `y‴ − y′ = 1/x` gets
/// `ln x`, `Ei(±x)`.  Before, variation of parameters was only tried for
/// order 2 and with the real basis (`∫ eˣ sin x/x dx` is not found), and
/// these equations were refused.  The homogeneous part is the real basis of
/// [`homogeneous_basis_cc`].
fn try_linear_cc_green(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Option<OdeResult> {
    let ode = extract_linear_cc(arena, expr, func, var, func_sym)?;
    let n = ode.coeffs.len() - 1;
    if !(2..=MAX_GREEN_ORDER).contains(&n) || ode.forcing.is_empty() {
        return None;
    }
    let p = crate::poly::Poly::from_coeffs(ode.coeffs.clone());
    let s = arena.symbol("__s_green");
    let roots = green_roots(arena, &p, s)?;
    let basis = homogeneous_basis_cc(arena, &ode.coeffs, var)?;
    if basis.len() != n {
        return None;
    }
    // g = −(forcing terms), circular and hyperbolic functions as exponentials
    let g = {
        let f = arena.add(&ode.forcing);
        let neg = arena.neg(f);
        let g = crate::simplify::rewrite::rewrite_as_exp(arena, neg);
        crate::transforms::eval::eval(arena, g)
    };
    let mut particular = Vec::new();
    for gr in &roots {
        // F_i = ∫ xⁱ·e^{−rx}·g dx, i < m
        let neg_r = arena.neg(gr.root);
        let neg_rx = arena.mul(&[neg_r, var]);
        let decay = arena.exp(neg_rx);
        let mut antiderivatives = Vec::with_capacity(gr.mult);
        for i in 0..gr.mult {
            let xi = arena.int(i64::try_from(i).ok()?);
            let xi = arena.pow(var, xi);
            let integrand = arena.mul(&[xi, decay, g]);
            let integrand = crate::transforms::expand::expand(arena, integrand);
            let integrand = crate::transforms::eval::eval(arena, integrand);
            let f = integrate_forms(arena, integrand, var);
            if integral_failed(arena, f) {
                return None;
            }
            antiderivatives.push(f);
        }
        let rx = arena.mul(&[gr.root, var]);
        let growth = arena.exp(rx);
        // A_{r,j} = (1/(m−j)!)·[d^{m−j}/ds^{m−j} 1/cofactor](r)
        let mut inv = {
            let m1 = arena.int(-1);
            arena.pow(gr.cofactor, m1)
        };
        let mut derivs = vec![inv];
        for _ in 1..gr.mult {
            inv = crate::transforms::diff::diff(arena, inv, s);
            derivs.push(inv);
        }
        for j in 1..=gr.mult {
            let k = gr.mult - j;
            let at = crate::transforms::subs::subs(arena, derivs[k], s, gr.root);
            let fact = ode_ratio_to_expr(arena, &falling_factorial_rat(k, k));
            let a = arena.div(at, fact);
            let a = crate::transforms::expand::expand(arena, a);
            let a = crate::transforms::eval::eval(arena, a);
            if arena.is_zero_structural(a) {
                continue;
            }
            // Σ_i C(j−1, i)·x^{j−1−i}·(−1)^i·F_i
            let mut inner = Vec::with_capacity(j);
            for (i, &fi) in antiderivatives.iter().enumerate().take(j) {
                let mut c = Q::from_integer(binomial((j - 1) as u64, i as u64));
                if i % 2 == 1 {
                    c = -c;
                }
                let c = ode_ratio_to_expr(arena, &c);
                let xp = arena.int(i64::try_from(j - 1 - i).ok()?);
                let xp = arena.pow(var, xp);
                inner.push(arena.mul(&[c, xp, fi]));
            }
            let inner = arena.add(&inner);
            let jf = ode_ratio_to_expr(arena, &falling_factorial_rat(j - 1, j - 1));
            let term = arena.mul(&[a, growth, inner]);
            particular.push(arena.div(term, jf));
        }
    }
    // Products of exponentials merged, complex exponentials as cos + i·sin
    // (the imaginary parts of the elementary terms then cancel).
    let y_p = arena.add(&particular);
    let y_p = crate::simplify::powsimp::powsimp(arena, y_p);
    let y_p = crate::transforms::eval::eval(arena, y_p);
    let y_p = complex_exp_as_trig(arena, y_p, var);
    let y_p = crate::transforms::expand::expand(arena, y_p);
    let y_p = crate::transforms::eval::eval(arena, y_p);
    if crate::base::walk::has_unevaluated(arena, y_p) {
        return None;
    }
    let mut constants = Vec::with_capacity(n);
    let mut terms = Vec::with_capacity(n + 1);
    for (i, &b) in basis.iter().enumerate() {
        let c = arena.symbol(&format!("C{}", i + 1));
        constants.push(c);
        terms.push(arena.mul(&[c, b]));
    }
    terms.push(y_p);
    let solution = arena.add(&terms);
    let solution = crate::transforms::eval::eval(arena, solution);
    Some(OdeResult {
        solution,
        constants,
        branches: Vec::new(),
    })
}

/// Extract fundamental solutions y₁ and y₂ from a homogeneous solution
/// of the form C1·y₁ + C2·y₂ (or multiplied by a common factor).
///
/// Substitutes C1=1,C2=0 and C1=0,C2=1 to recover the individual solutions.
fn extract_fundamental_solutions(
    arena: &mut Arena,
    homo_solution: ExprId,
    constants: &[ExprId],
) -> Option<(ExprId, ExprId)> {
    if constants.len() != 2 {
        return None;
    }
    let c1 = constants[0];
    let c2 = constants[1];
    let zero = arena.zero;
    let one = arena.one;

    let y1 = crate::transforms::subs::subs(arena, homo_solution, c1, one);
    let y1 = crate::transforms::subs::subs(arena, y1, c2, zero);
    let y1 = crate::transforms::eval::eval(arena, y1);

    let y2 = crate::transforms::subs::subs(arena, homo_solution, c1, zero);
    let y2 = crate::transforms::subs::subs(arena, y2, c2, one);
    let y2 = crate::transforms::eval::eval(arena, y2);

    if y1 == arena.zero || y2 == arena.zero {
        return None;
    }

    Some((y1, y2))
}

// ═══════════════════════════════════════════════════════════════════════════
// nth-order linear constant-coefficient ODEs: Σ a_k y^(k) = g(x)
// ═══════════════════════════════════════════════════════════════════════════

/// Highest derivative order recognised by the nth-order solver.
const MAX_ODE_ORDER: usize = 12;

/// A linear constant-coefficient ODE `Σ a_k·y^(k) + g(x) = 0`.
struct LinearCcOde {
    /// `a_0 … a_n` (ascending derivative order), `a_n ≠ 0`.
    coeffs: Vec<Q>,
    /// The forcing terms `g(x)` as they appear in the zero-form expression.
    forcing: Vec<ExprId>,
}

/// Derivative chain `[y, y', y'', …]` up to [`MAX_ODE_ORDER`].
fn derivative_chain(arena: &mut Arena, func: ExprId, var: ExprId) -> Vec<ExprId> {
    let mut chain = vec![func];
    for k in 1..=MAX_ODE_ORDER {
        let d = arena.intern(ExprNode::Derivative(chain[k - 1], var));
        chain.push(d);
    }
    chain
}

/// Recognise `Σ a_k·y^(k) + g(x) = 0` with rational `a_k`.
fn extract_linear_cc(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Option<LinearCcOde> {
    use num_traits::Zero;
    let chain = derivative_chain(arena, func, var);
    let children: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![expr],
    };
    let mut coeffs = vec![Q::zero(); MAX_ODE_ORDER + 1];
    let mut forcing = Vec::new();
    for child in children {
        let (coeff, term) = arena.as_coeff_term(child);
        if let Some(k) = chain.iter().position(|&d| d == term) {
            coeffs[k] += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            forcing.push(child);
        } else {
            return None;
        }
    }
    while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
        coeffs.pop();
    }
    if coeffs.len() < 2 {
        return None; // no derivative at all
    }
    Some(LinearCcOde { coeffs, forcing })
}

/// Gaussian rational `re + im·i` — exact arithmetic for the complex
/// exponential shift used by undetermined coefficients.
#[derive(Clone, Debug, PartialEq)]
struct CQ {
    re: Q,
    im: Q,
}

impl CQ {
    fn new(re: Q, im: Q) -> Self {
        Self { re, im }
    }
    fn real(re: Q) -> Self {
        Self {
            re,
            im: num_traits::Zero::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        num_traits::Zero::is_zero(&self.re) && num_traits::Zero::is_zero(&self.im)
    }
    fn add(&self, o: &CQ) -> CQ {
        CQ::new(&self.re + &o.re, &self.im + &o.im)
    }
    fn sub(&self, o: &CQ) -> CQ {
        CQ::new(&self.re - &o.re, &self.im - &o.im)
    }
    fn mul(&self, o: &CQ) -> CQ {
        CQ::new(
            &self.re * &o.re - &self.im * &o.im,
            &self.re * &o.im + &self.im * &o.re,
        )
    }
    fn div(&self, o: &CQ) -> CQ {
        let denom = &o.re * &o.re + &o.im * &o.im;
        let num = self.mul(&CQ::new(o.re.clone(), -o.im.clone()));
        CQ::new(num.re / &denom, num.im / &denom)
    }
    fn scale(&self, r: &Q) -> CQ {
        CQ::new(&self.re * r, &self.im * r)
    }
}

/// Evaluate `p^(j)(λ) / j!` for a rational polynomial `p` (ascending
/// coefficients) at a Gaussian rational `λ`.
fn shifted_coefficient(p: &[Q], j: usize, lambda: &CQ) -> CQ {
    // p^(j)(λ)/j! = Σ_{k≥j} C(k, j) a_k λ^{k-j}
    let mut acc = CQ::real(num_traits::Zero::zero());
    let mut lambda_pow = CQ::real(num_traits::One::one());
    for (k, a_k) in p.iter().enumerate().skip(j) {
        let binom = Q::from_integer(binomial(k as u64, j as u64));
        let term = lambda_pow.scale(&(a_k * binom));
        acc = acc.add(&term);
        lambda_pow = lambda_pow.mul(lambda);
    }
    acc
}

/// Trigonometric flavour of a forcing term.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TrigKind {
    None,
    Cos,
    Sin,
}

/// One forcing term `c·x^d·e^{ax}·{1 | cos(bx) | sin(bx)}`.
struct ForcingTerm {
    coeff: Q,
    degree: usize,
    a: Q,
    b: Q,
    kind: TrigKind,
}

/// Parse a single additive forcing term (free of `y`).
fn parse_forcing_term(arena: &mut Arena, term: ExprId, var: ExprId) -> Option<ForcingTerm> {
    use num_traits::{Signed, Zero};
    let (mut coeff, t) = arena.as_coeff_term(term);
    let factors: Vec<ExprId> = if t == arena.one {
        Vec::new()
    } else {
        match arena.node(t).clone() {
            ExprNode::Mul(c) => c.to_vec(),
            _ => vec![t],
        }
    };
    let mut degree = 0usize;
    let mut a = Q::zero();
    let mut b = Q::zero();
    let mut kind = TrigKind::None;
    let mut seen_exp = false;
    for f in factors {
        if f == var {
            degree += 1;
            continue;
        }
        match arena.node(f).clone() {
            ExprNode::Pow(base, exp) if base == var => {
                let n = arena.as_num(exp)?.clone();
                if !n.is_integer() || n.is_negative() {
                    return None;
                }
                degree += usize::try_from(n.to_integer()).ok()?;
            }
            ExprNode::Exp(inner) => {
                if seen_exp {
                    return None;
                }
                let (rate, c) = extract_linear_numeric(arena, inner, var)?;
                if !c.is_zero() {
                    return None;
                }
                seen_exp = true;
                a = rate;
            }
            ExprNode::Sin(inner) => {
                if kind != TrigKind::None {
                    return None;
                }
                let (freq, c) = extract_linear_numeric(arena, inner, var)?;
                if !c.is_zero() || freq.is_zero() {
                    return None;
                }
                kind = TrigKind::Sin;
                if freq.is_negative() {
                    coeff = -coeff;
                    b = -freq;
                } else {
                    b = freq;
                }
            }
            ExprNode::Cos(inner) => {
                if kind != TrigKind::None {
                    return None;
                }
                let (freq, c) = extract_linear_numeric(arena, inner, var)?;
                if !c.is_zero() || freq.is_zero() {
                    return None;
                }
                kind = TrigKind::Cos;
                b = freq.abs();
            }
            _ => {
                let r = arena.as_num(f)?;
                coeff *= r.clone();
            }
        }
    }
    Some(ForcingTerm {
        coeff,
        degree,
        a,
        b,
        kind,
    })
}

/// Build `Σ c_k·x^k` from rational coefficients.
fn rat_poly_expr(arena: &mut Arena, coeffs: &[Q], var: ExprId) -> ExprId {
    build_polynomial_expr(arena, coeffs, var)
}

/// Particular solution of `Σ a_k y^(k) = q(x)·e^{ax}·{1|cos bx|sin bx}` via the
/// complex exponential shift: with `λ = a + bi`, `y_p = Re/Im(w(x)·e^{λx})`
/// where `w` solves the triangular system `Σ_j (p^(j)(λ)/j!)·w^(j) = q`.
/// Resonance (λ a root of multiplicity `s`) is handled by the extra
/// factor `x^s` implicit in the degree of `w`.
fn particular_for_group(
    arena: &mut Arena,
    p: &[Q],
    q: &[Q],
    a: &Q,
    b: &Q,
    kind: TrigKind,
    var: ExprId,
) -> Option<ExprId> {
    use num_traits::Zero;
    let lambda = CQ::new(a.clone(), b.clone());
    let n = p.len() - 1;
    // c_j = p^(j)(λ)/j!
    let c: Vec<CQ> = (0..=n)
        .map(|j| shifted_coefficient(p, j, &lambda))
        .collect();
    let s = c.iter().position(|cj| !cj.is_zero())?;
    let d = q.len().checked_sub(1)?;
    let m = d + s;
    let mut w = vec![CQ::real(Q::zero()); m + 1];
    // Solve from the top degree down: for k = d..0,
    //   Σ_{j≥s} c_j·(k+j)!/k!·w_{k+j} = q_k
    for k in (0..=d).rev() {
        let mut rhs = CQ::real(q[k].clone());
        for j in (s + 1)..=n {
            if k + j > m {
                continue;
            }
            let fact = falling_factorial_rat(k + j, j);
            rhs = rhs.sub(&c[j].mul(&w[k + j]).scale(&fact));
        }
        let lead = c[s].scale(&falling_factorial_rat(k + s, s));
        w[k + s] = rhs.div(&lead);
    }
    let wr: Vec<Q> = w.iter().map(|z| z.re.clone()).collect();
    let wi: Vec<Q> = w.iter().map(|z| z.im.clone()).collect();
    let wr_x = rat_poly_expr(arena, &wr, var);
    let wi_x = rat_poly_expr(arena, &wi, var);
    let exp_ax = if a.is_zero() {
        arena.one
    } else {
        let a_id = ode_ratio_to_expr(arena, a);
        let ax = arena.mul(&[a_id, var]);
        arena.exp(ax)
    };
    let body = match kind {
        TrigKind::None => wr_x,
        TrigKind::Cos | TrigKind::Sin => {
            let b_id = ode_ratio_to_expr(arena, b);
            let bx = arena.mul(&[b_id, var]);
            let cos_bx = arena.cos(bx);
            let sin_bx = arena.sin(bx);
            if kind == TrigKind::Cos {
                // Re(w e^{ibx}) = wr cos - wi sin
                let t1 = arena.mul(&[wr_x, cos_bx]);
                let t2 = arena.mul(&[wi_x, sin_bx]);
                arena.sub(t1, t2)
            } else {
                // Im(w e^{ibx}) = wr sin + wi cos
                let t1 = arena.mul(&[wr_x, sin_bx]);
                let t2 = arena.mul(&[wi_x, cos_bx]);
                arena.add(&[t1, t2])
            }
        }
    };
    let y_p = arena.mul(&[exp_ax, body]);
    let y_p = crate::transforms::expand::expand(arena, y_p);
    Some(crate::transforms::eval::eval(arena, y_p))
}

/// A forcing term `c·x^d·e^{ax}·{1 | cos(bx) | sin(bx)}` whose constants
/// `c`, `a`, `b` are real numbers that are not all rational (`e^{√2·x}`,
/// `e^{x/2}·cos(√3·x/2)`), for [`particular_for_group_sym`].
struct SymForcingTerm {
    coeff: ExprId,
    degree: usize,
    a: ExprId,
    b: ExprId,
    kind: TrigKind,
}

/// Is `e` a real constant: free of symbols and of `i`, with a finite real
/// value?
fn is_real_constant(arena: &Arena, e: ExprId) -> bool {
    crate::base::walk::free_symbols(arena, e).is_empty()
        && real_radical_form(arena, e)
        && crate::transforms::evalf::evalf_complex64(arena, e)
            .is_ok_and(|z| z.re.is_finite() && z.im == 0.0)
}

/// `(rate, offset)` with `e = rate·var + offset`, both real constants.
fn linear_real_constants(arena: &mut Arena, e: ExprId, var: ExprId) -> Option<(ExprId, ExprId)> {
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, e, var)?;
    if coeffs.len() != 2 || !coeffs.iter().all(|&c| is_real_constant(arena, c)) {
        return None;
    }
    Some((coeffs[1], coeffs[0]))
}

/// [`parse_forcing_term`] with real constant (not necessarily rational)
/// coefficient, rates and frequency.  A phase-free `cos`/`sin`; an offset
/// in the exponent goes into the coefficient.
fn parse_forcing_term_sym(arena: &mut Arena, term: ExprId, var: ExprId) -> Option<SymForcingTerm> {
    let factors: Vec<ExprId> = match arena.node(term).clone() {
        ExprNode::Mul(c) => c.to_vec(),
        ExprNode::Neg(inner) => vec![arena.neg_one, inner],
        _ => vec![term],
    };
    let mut coeff = vec![];
    let mut degree = 0usize;
    let (mut a, mut b) = (arena.zero, arena.zero);
    let mut kind = TrigKind::None;
    let mut seen_exp = false;
    for f in factors {
        if f == var {
            degree += 1;
            continue;
        }
        if !contains_sym_id(arena, f, var) {
            if !is_real_constant(arena, f) {
                return None;
            }
            coeff.push(f);
            continue;
        }
        match arena.node(f).clone() {
            ExprNode::Pow(base, exp) if base == var => {
                let n = arena.as_num(exp)?.clone();
                if !n.is_integer() || n.is_negative() {
                    return None;
                }
                degree += usize::try_from(n.to_integer()).ok()?;
            }
            ExprNode::Exp(inner) if !seen_exp => {
                let (rate, offset) = linear_real_constants(arena, inner, var)?;
                seen_exp = true;
                a = rate;
                if !arena.is_zero_structural(offset) {
                    coeff.push(arena.exp(offset));
                }
            }
            // b^(k·x + m) = e^{(k·x + m)·ln b} for a constant b > 0 (`2ˣ`)
            ExprNode::Pow(base, exp)
                if !seen_exp
                    && !contains_sym_id(arena, base, var)
                    && is_real_constant(arena, base)
                    && crate::transforms::evalf::evalf_complex64(arena, base)
                        .is_ok_and(|z| z.re > 0.0) =>
            {
                let (k, m) = linear_real_constants(arena, exp, var)?;
                let ln_b = arena.ln(base);
                let rate = arena.mul(&[k, ln_b]);
                seen_exp = true;
                a = crate::transforms::eval::eval(arena, rate);
                if !arena.is_zero_structural(m) {
                    coeff.push(arena.pow(base, m));
                }
            }
            ExprNode::Sin(inner) | ExprNode::Cos(inner) if kind == TrigKind::None => {
                let (freq, offset) = linear_real_constants(arena, inner, var)?;
                if !arena.is_zero_structural(offset) {
                    return None;
                }
                let negative = crate::transforms::evalf::evalf_complex64(arena, freq)
                    .ok()
                    .filter(|z| z.re != 0.0)?
                    .re
                    < 0.0;
                let is_sin = matches!(arena.node(f), ExprNode::Sin(_));
                kind = if is_sin { TrigKind::Sin } else { TrigKind::Cos };
                b = if negative { arena.neg(freq) } else { freq };
                b = crate::transforms::eval::eval(arena, b);
                if negative && is_sin {
                    coeff.push(arena.neg_one);
                }
            }
            _ => return None,
        }
    }
    let coeff = arena.mul(&coeff);
    Some(SymForcingTerm {
        coeff: crate::transforms::eval::eval(arena, coeff),
        degree,
        a: crate::transforms::eval::eval(arena, a),
        b,
        kind,
    })
}

/// Does `e` contain the expression `var` (a symbol)?
fn contains_sym_id(arena: &Arena, e: ExprId, var: ExprId) -> bool {
    match arena.node(var) {
        ExprNode::Symbol(s) => contains_sym(arena, e, *s),
        _ => true,
    }
}

/// A complex number `re + i·im` with real constant parts, for
/// [`particular_for_group_sym`] (the symbolic counterpart of [`CQ`]).
#[derive(Clone, Copy)]
struct CEx {
    re: ExprId,
    im: ExprId,
}

impl CEx {
    fn norm(arena: &mut Arena, re: ExprId, im: ExprId) -> CEx {
        let mut parts = [re, im];
        for p in &mut parts {
            let e = crate::transforms::expand::expand(arena, *p);
            *p = crate::transforms::eval::eval(arena, e);
        }
        CEx {
            re: parts[0],
            im: parts[1],
        }
    }
    fn add(self, arena: &mut Arena, o: CEx) -> CEx {
        let re = arena.add(&[self.re, o.re]);
        let im = arena.add(&[self.im, o.im]);
        CEx::norm(arena, re, im)
    }
    fn sub(self, arena: &mut Arena, o: CEx) -> CEx {
        let re = arena.sub(self.re, o.re);
        let im = arena.sub(self.im, o.im);
        CEx::norm(arena, re, im)
    }
    fn mul(self, arena: &mut Arena, o: CEx) -> CEx {
        let rr = arena.mul(&[self.re, o.re]);
        let ii = arena.mul(&[self.im, o.im]);
        let ri = arena.mul(&[self.re, o.im]);
        let ir = arena.mul(&[self.im, o.re]);
        let re = arena.sub(rr, ii);
        let im = arena.add(&[ri, ir]);
        CEx::norm(arena, re, im)
    }
    fn scale(self, arena: &mut Arena, r: &Q) -> CEx {
        let r = ode_ratio_to_expr(arena, r);
        let re = arena.mul(&[r, self.re]);
        let im = arena.mul(&[r, self.im]);
        CEx::norm(arena, re, im)
    }
    fn div(self, arena: &mut Arena, o: CEx) -> CEx {
        // (x + iy)/(u + iv) = ((xu + yv) + i(yu − xv))/(u² + v²)
        let two = arena.int(2);
        let u2 = arena.pow(o.re, two);
        let v2 = arena.pow(o.im, two);
        let d = arena.add(&[u2, v2]);
        let d = crate::transforms::expand::expand(arena, d);
        let d = crate::transforms::eval::eval(arena, d);
        let xu = arena.mul(&[self.re, o.re]);
        let yv = arena.mul(&[self.im, o.im]);
        let yu = arena.mul(&[self.im, o.re]);
        let xv = arena.mul(&[self.re, o.im]);
        let re = arena.add(&[xu, yv]);
        let im = arena.sub(yu, xv);
        let mut parts = [re, im];
        for p in &mut parts {
            let e = crate::transforms::expand::expand(arena, *p);
            let e = crate::transforms::eval::eval(arena, e);
            let q = arena.div(e, d);
            *p = crate::simplify::radsimp::rationalize_denom(arena, q);
        }
        CEx::norm(arena, parts[0], parts[1])
    }
    /// `Some(true)` for an exact zero, `Some(false)` for a value certainly
    /// away from zero, `None` when undecided (not structurally zero but
    /// numerically tiny).
    fn is_zero(self, arena: &Arena) -> Option<bool> {
        if arena.is_zero_structural(self.re) && arena.is_zero_structural(self.im) {
            return Some(true);
        }
        let re = crate::transforms::evalf::evalf_complex64(arena, self.re).ok()?;
        let im = crate::transforms::evalf::evalf_complex64(arena, self.im).ok()?;
        (re.norm().max(im.norm()) > 1e-8).then_some(false)
    }
}

/// [`particular_for_group`] for real constant `a`, `b` and polynomial
/// coefficients `q` that are not all rational: the same exponential shift
/// with `λ = a + b·i` in exact arithmetic on real constants ([`CEx`]); the
/// vanishing of `p⁽ʲ⁾(λ)` (resonance: `e^{√2·x}` against the root `√2` of
/// `r⁴ − 2`) must be decided exactly, or the term is refused.
fn particular_for_group_sym(
    arena: &mut Arena,
    p: &[Q],
    q: &[ExprId],
    a: ExprId,
    b: ExprId,
    kind: TrigKind,
    var: ExprId,
) -> Option<ExprId> {
    let lambda = CEx { re: a, im: b };
    let zero = CEx {
        re: arena.zero,
        im: arena.zero,
    };
    let n = p.len() - 1;
    // c_j = Σ_{k≥j} C(k, j)·a_k·λ^{k−j}
    let mut c = Vec::with_capacity(n + 1);
    for j in 0..=n {
        let mut acc = zero;
        let mut lambda_pow = CEx {
            re: arena.one,
            im: arena.zero,
        };
        for (k, a_k) in p.iter().enumerate().skip(j) {
            let coef = a_k * Q::from_integer(binomial(k as u64, j as u64));
            let term = lambda_pow.scale(arena, &coef);
            acc = acc.add(arena, term);
            lambda_pow = lambda_pow.mul(arena, lambda);
        }
        c.push(acc);
    }
    let mut s = None;
    for (j, cj) in c.iter().enumerate() {
        if !cj.is_zero(arena)? {
            s = Some(j);
            break;
        }
    }
    let s = s?;
    let d = q.len().checked_sub(1)?;
    let m = d + s;
    let mut w = vec![zero; m + 1];
    for k in (0..=d).rev() {
        let mut rhs = CEx {
            re: q[k],
            im: arena.zero,
        };
        for j in (s + 1)..=n {
            if k + j > m {
                continue;
            }
            let t = c[j]
                .mul(arena, w[k + j])
                .scale(arena, &falling_factorial_rat(k + j, j));
            rhs = rhs.sub(arena, t);
        }
        let lead = c[s].scale(arena, &falling_factorial_rat(k + s, s));
        w[k + s] = rhs.div(arena, lead);
    }
    let poly = |arena: &mut Arena, parts: &[ExprId]| -> ExprId {
        let mut terms = Vec::with_capacity(parts.len());
        for (k, &ck) in parts.iter().enumerate() {
            let e = arena.int(k as i64);
            let xk = arena.pow(var, e);
            terms.push(arena.mul(&[ck, xk]));
        }
        let sum = arena.add(&terms);
        crate::transforms::eval::eval(arena, sum)
    };
    // each coefficient as one reduced fraction (`(ln²2 + 3)/(ln²2 + 3)²` came
    // out unreduced for the rate `ln 2` of `2ˣ`)
    let reduce = |arena: &mut Arena, e: ExprId| -> ExprId {
        if arena.as_num(e).is_some() {
            return e;
        }
        let r = crate::simplify::ratsimp::ratsimp(arena, e);
        let r = crate::transforms::eval::eval(arena, r);
        let size = |arena: &Arena, x: ExprId| {
            crate::transforms::pattern::tree_size_capped(arena, x, 10_000)
        };
        if size(arena, r) <= size(arena, e) {
            r
        } else {
            e
        }
    };
    let wr: Vec<ExprId> = w.iter().map(|z| reduce(arena, z.re)).collect();
    let wi: Vec<ExprId> = w.iter().map(|z| reduce(arena, z.im)).collect();
    let wr_x = poly(arena, &wr);
    let wi_x = poly(arena, &wi);
    let ax = arena.mul(&[a, var]);
    let exp_ax = arena.exp(ax);
    let body = match kind {
        TrigKind::None => wr_x,
        TrigKind::Cos | TrigKind::Sin => {
            let bx = arena.mul(&[b, var]);
            let cos_bx = arena.cos(bx);
            let sin_bx = arena.sin(bx);
            if kind == TrigKind::Cos {
                let t1 = arena.mul(&[wr_x, cos_bx]);
                let t2 = arena.mul(&[wi_x, sin_bx]);
                arena.sub(t1, t2)
            } else {
                let t1 = arena.mul(&[wr_x, sin_bx]);
                let t2 = arena.mul(&[wi_x, cos_bx]);
                arena.add(&[t1, t2])
            }
        }
    };
    let y_p = arena.mul(&[exp_ax, body]);
    let y_p = crate::transforms::expand::expand(arena, y_p);
    Some(crate::transforms::eval::eval(arena, y_p))
}

/// `sinh u → (eᵘ − e⁻ᵘ)/2`, `cosh u → (eᵘ + e⁻ᵘ)/2`, then expanded:
/// the forcing terms of undetermined coefficients (`y‴ − 2y = sinh x` was
/// refused, its forcing not being of the form `xᵈ·e^{ax}·cos/sin`).
/// `None` when `terms` have no hyperbolic function.
fn hyperbolic_forcing_as_exp(arena: &mut Arena, terms: &[ExprId]) -> Option<Vec<ExprId>> {
    let has_hyperbolic = |arena: &Arena, e: ExprId| {
        crate::base::walk::post_order_ids(arena, e)
            .iter()
            .any(|&id| matches!(arena.node(id), ExprNode::Sinh(_) | ExprNode::Cosh(_)))
    };
    if !terms.iter().any(|&t| has_hyperbolic(arena, t)) {
        return None;
    }
    let sum = arena.add(terms);
    let post = crate::base::walk::post_order_ids(arena, sum);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Sinh(u) | ExprNode::Cosh(u) => {
                let is_sinh = matches!(arena.node(rebuilt), ExprNode::Sinh(_));
                let pos = arena.exp(u);
                let neg_u = arena.neg(u);
                let neg = arena.exp(neg_u);
                let neg = if is_sinh { arena.neg(neg) } else { neg };
                let s = arena.add(&[pos, neg]);
                let half = ode_ratio_to_expr(arena, &Q::new(1.into(), 2.into()));
                arena.mul(&[half, s])
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    let out = cache.get(&sum).copied().unwrap_or(sum);
    let out = crate::transforms::expand::expand(arena, out);
    let out = crate::transforms::eval::eval(arena, out);
    Some(match arena.node(out) {
        ExprNode::Add(c) => c.to_vec(),
        _ if arena.is_zero_structural(out) => Vec::new(),
        _ => vec![out],
    })
}

/// Pairs `c·r·e^{kx} ± c·r·e^{−kx}` (rational `c`, `k > 0`, the same rest
/// `r`) of the sum `e` written back as `2c·r·cosh(kx)`, `2c·r·sinh(kx)`:
/// the particular solution for a hyperbolic forcing term, found after
/// [`hyperbolic_forcing_as_exp`] (`y″ = sinh x` gives `sinh x` again, not
/// `eˣ/2 − e⁻ˣ/2`).
fn exp_pairs_as_hyperbolic(arena: &mut Arena, e: ExprId, var: ExprId) -> ExprId {
    let terms: Vec<ExprId> = match arena.node(e) {
        ExprNode::Add(c) => c.to_vec(),
        _ => return e,
    };
    // (rate k, coefficient c, rest r) of `c·r·e^{kx}`
    let mut parsed: Vec<Option<(Q, Q, ExprId)>> = Vec::with_capacity(terms.len());
    for &t in &terms {
        let (c, body) = arena.as_coeff_term(t);
        let factors: Vec<ExprId> = match arena.node(body) {
            ExprNode::Mul(fs) => fs.to_vec(),
            _ => vec![body],
        };
        let mut rate = None;
        let mut rest = Vec::with_capacity(factors.len());
        for f in factors {
            match *arena.node(f) {
                ExprNode::Exp(arg) if rate.is_none() => {
                    match extract_linear_numeric(arena, arg, var) {
                        Some((k, c0))
                            if num_traits::Zero::is_zero(&c0) && !num_traits::Zero::is_zero(&k) =>
                        {
                            rate = Some(k)
                        }
                        _ => rest.push(f),
                    }
                }
                _ => rest.push(f),
            }
        }
        parsed.push(rate.map(|k| (k, c, arena.mul(&rest))));
    }
    let mut used = vec![false; terms.len()];
    let mut out = Vec::with_capacity(terms.len());
    for i in 0..terms.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let Some((k, c, rest)) = parsed[i].clone() else {
            out.push(terms[i]);
            continue;
        };
        let partner = (0..terms.len()).find(|&j| {
            !used[j]
                && parsed[j].as_ref().is_some_and(|(k2, c2, r2)| {
                    *k2 == -k.clone() && *r2 == rest && (*c2 == c || *c2 == -c.clone())
                })
        });
        let Some(j) = partner else {
            out.push(terms[i]);
            continue;
        };
        used[j] = true;
        let Some((_, c2, _)) = parsed[j].clone() else {
            continue;
        };
        // c·e^{kx} + c2·e^{−kx} with |k| written as the positive rate
        let (pos_c, neg_c) = if k.is_positive() { (c, c2) } else { (c2, c) };
        let rate = ode_ratio_to_expr(arena, &k.abs());
        let arg = arena.mul(&[rate, var]);
        let (func, scale) = if pos_c == neg_c {
            (arena.cosh(arg), pos_c)
        } else {
            (arena.sinh(arg), pos_c)
        };
        let two_c = ode_ratio_to_expr(arena, &(Q::from_integer(2.into()) * scale));
        out.push(arena.mul(&[two_c, rest, func]));
    }
    let sum = arena.add(&out);
    crate::transforms::eval::eval(arena, sum)
}

/// `n·(n-1)·…·(n-j+1)` as a rational.
fn falling_factorial_rat(n: usize, j: usize) -> Q {
    let mut r = Q::from_integer(1.into());
    for i in 0..j {
        r *= Q::from_integer(((n - i) as i64).into());
    }
    r
}

/// Fundamental solutions of `Σ a_k y^(k) = 0` from the square-free
/// factorisation of the characteristic polynomial: `x^j e^{rx}` for a
/// real root of multiplicity `> j`, and `x^j e^{αx} cos(βx)`,
/// `x^j e^{αx} sin(βx)` for a complex pair `α ± βi` (`β > 0`): a real
/// fundamental system, as SymPy's `nth_linear_constant_coeff_homogeneous`.
/// The roots are those of [`characteristic_roots`]: radicals where the
/// solver gives them with real and imaginary parts in real radicals
/// (Cardano's for `r³ + r² + 2`), `RootOf` otherwise.
fn homogeneous_basis_cc(arena: &mut Arena, coeffs: &[Q], var: ExprId) -> Option<Vec<ExprId>> {
    let p = crate::poly::Poly::from_coeffs(coeffs.to_vec());
    let r_sym = arena.symbol("__r_cc");
    let mut basis: Vec<ExprId> = Vec::new();
    for (factor, mult) in p.squarefree_factors() {
        for root in characteristic_roots(arena, &factor, r_sym, var)? {
            for j in 0..mult {
                match root {
                    CharRoot::Real(r) => basis.push(mode_real(arena, r, j, var)),
                    CharRoot::Pair { re, im } => {
                        let (c, s) = mode_complex(arena, re, im, j, var);
                        basis.push(c);
                        basis.push(s);
                    }
                }
            }
        }
    }
    Some(basis)
}

/// A root of a real characteristic polynomial as [`homogeneous_basis_cc`]
/// uses it: a real root, or a non-real pair `re ± i·im` with `im > 0`.
#[derive(Clone, Copy, Debug)]
enum CharRoot {
    /// A real root (an expression free of `i`).
    Real(ExprId),
    /// The pair `re ± i·im`, both parts real expressions, `im > 0`.
    Pair { re: ExprId, im: ExprId },
}

/// The distinct roots of the square-free rational polynomial `f` (one
/// entry per real root or conjugate pair).  First the solver's roots of
/// `f` in radicals ([`radical_char_roots`]); failing that, each factor
/// irreducible over ℚ in radicals or, when the radicals have no real form
/// (degree ≥ 5, a casus-irreducibilis cubic, Ferrari's complex cube
/// roots), as `RootOf` ([`rootof_char_roots`]).  Before, every root
/// containing `i` was taken as non-real and paired by structural equality
/// of the parts: an irreducible cubic (`r³ + r² + 2`, `r³ − 3r + 1`) gave
/// no basis (the ODE was refused), and `RootOf` roots of a quintic were
/// used as complex exponentials `e^{x·RootOf(…)}`.
fn characteristic_roots(
    arena: &mut Arena,
    f: &crate::poly::Poly,
    r_sym: ExprId,
    var: ExprId,
) -> Option<Vec<CharRoot>> {
    if let Some(roots) = radical_char_roots(arena, f, r_sym) {
        return Some(roots);
    }
    let (_, irreducible) = f.factor_over_z();
    let mut out = Vec::new();
    for (g, _) in &irreducible {
        // A single factor is `f` itself, whose radicals just failed.
        let radical = if irreducible.len() > 1 {
            radical_char_roots(arena, g, r_sym)
        } else {
            None
        };
        let roots = match radical {
            Some(r) => r,
            None => rootof_char_roots(arena, g, var)?,
        };
        out.extend(roots);
    }
    Some(out)
}

/// Relative size of the imaginary part below which a root value is taken
/// as real by [`radical_char_roots`] (the exact count of real roots decides
/// how many are; this only orders them), and above [`IM_NONREAL`] non-real.
const IM_REAL: f64 = 1e-9;
/// See [`IM_REAL`].
const IM_NONREAL: f64 = 1e-6;

/// The roots of the square-free rational polynomial `f` from the solver,
/// in radicals: `None` unless there are `deg f` of them, none a `RootOf`,
/// each real root has a real form (free of `i`) and each non-real pair has
/// real and imaginary parts free of `i` and of unevaluated `re`/`im`.  How
/// many roots are real is the exact count of
/// [`SturmChain`](crate::poly::sturm::SturmChain); which ones, their
/// values (the solver's spelling says nothing: a casus-irreducibilis root
/// is real), and every part is checked against the value.
fn radical_char_roots(
    arena: &mut Arena,
    f: &crate::poly::Poly,
    r_sym: ExprId,
) -> Option<Vec<CharRoot>> {
    let deg = f.degree()?;
    let f_expr = crate::poly::polybridge::poly_to_expr(arena, f, r_sym);
    let sols = match crate::transforms::solve::solve_classified_real(arena, f_expr, r_sym)
        .points_if_complete()
    {
        Some(sols) if sols.len() == deg => sols,
        _ => return None,
    };
    let mut values = Vec::with_capacity(deg);
    for &s in &sols {
        if crate::base::walk::contains(arena, s, r_sym) || has_rootof(arena, s) {
            return None;
        }
        let z = crate::transforms::evalf::evalf_complex64(arena, s).ok()?;
        if !(z.re.is_finite() && z.im.is_finite()) {
            return None;
        }
        values.push(z);
    }
    let n_real = crate::poly::sturm::SturmChain::new(f).count_real_roots();
    let rel_im = |z: &num_complex::Complex64| z.im.abs() / z.norm().max(1.0);
    let mut order: Vec<usize> = (0..deg).collect();
    order.sort_by(|&a, &b| rel_im(&values[a]).total_cmp(&rel_im(&values[b])));
    if (n_real > 0 && rel_im(&values[order[n_real - 1]]) > IM_REAL)
        || (n_real < deg && rel_im(&values[order[n_real]]) < IM_NONREAL)
    {
        return None;
    }
    let mut is_real = vec![false; deg];
    for &k in &order[..n_real] {
        is_real[k] = true;
    }
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * (1.0 + b.abs());
    let i_unit = arena.i_unit;
    let mut used = vec![false; deg];
    let mut out = Vec::new();
    for i in 0..deg {
        if used[i] {
            continue;
        }
        used[i] = true;
        if is_real[i] {
            let mut r = sols[i];
            if crate::base::walk::contains(arena, r, i_unit) {
                r = real_imag_parts(arena, r).0;
                if !real_radical_form(arena, r) {
                    return None;
                }
                let z = crate::transforms::evalf::evalf_complex64(arena, r).ok()?;
                if !close(z.re, values[i].re) || z.im.abs() > IM_REAL * (1.0 + z.re.abs()) {
                    return None;
                }
            }
            out.push(CharRoot::Real(r));
            continue;
        }
        // The conjugate partner: the unused non-real root nearest conj(zᵢ).
        let conj = values[i].conj();
        let j = (0..deg)
            .filter(|&k| !used[k] && !is_real[k])
            .min_by(|&a, &b| {
                (values[a] - conj)
                    .norm()
                    .total_cmp(&(values[b] - conj).norm())
            })?;
        if (values[j] - conj).norm() > 1e-9 * (1.0 + conj.norm()) {
            return None;
        }
        used[j] = true;
        let pos = if values[i].im > 0.0 { i } else { j };
        let (re, im) = real_imag_parts(arena, sols[pos]);
        if !real_radical_form(arena, re) || !real_radical_form(arena, im) {
            return None;
        }
        let zr = crate::transforms::evalf::evalf_complex64(arena, re).ok()?;
        let zi = crate::transforms::evalf::evalf_complex64(arena, im).ok()?;
        if !close(zr.re, values[pos].re) || !close(zi.re, values[pos].im) {
            return None;
        }
        out.push(CharRoot::Pair { re, im });
    }
    Some(out)
}

/// `(re, im)` of the constant `z` with real and imaginary parts of real
/// radicals resolved and the signs of `cos`/`sin` arguments normalised:
/// [`Arena::as_real_imag_expr`] leaves `re(cbrt(28/27 − √87/9))` when it
/// cannot tell the radicand is positive (Cardano's complex roots of
/// `r³ + r² + 2`), and gives `cos(−2π/9) + cos(2π/9)` for a
/// casus-irreducibilis root.  `re(u) → u`, `im(u) → 0` when `u` is real
/// by [`is_real_radical`]; `cos(−a) → cos(a)`, `sin(−a) → −sin(a)` for a
/// negative rational multiple `−a` of `π`.
fn real_imag_parts(arena: &mut Arena, z: ExprId) -> (ExprId, ExprId) {
    // √u of a negative real radical u is i·√(−u) (principal branch):
    // `as_real_imag` leaves `re(√(√5/2 − 3/2))`, so the biquadratic
    // `r⁴ + 3r² + 1` fell back to `RootOf`.
    let z = {
        let post = crate::base::walk::post_order_ids(arena, z);
        let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
        for &id in &post {
            let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
            let new = match *arena.node(rebuilt) {
                ExprNode::Pow(b, e)
                    if arena
                        .as_num(e)
                        .is_some_and(|q| *q == Q::new(1.into(), 2.into()))
                        && is_real_radical(arena, b)
                        && crate::transforms::evalf::evalf_complex64(arena, b)
                            .is_ok_and(|v| v.re < -1e-12) =>
                {
                    let neg_b = arena.neg(b);
                    let neg_b = crate::transforms::eval::eval(arena, neg_b);
                    let root = arena.pow(neg_b, e);
                    let i_unit = arena.i_unit;
                    arena.mul(&[i_unit, root])
                }
                _ => rebuilt,
            };
            cache.insert(id, new);
        }
        let z = cache.get(&z).copied().unwrap_or(z);
        crate::transforms::eval::eval(arena, z)
    };
    let (re, im) = arena.as_real_imag_expr(z);
    let mut parts = [re, im];
    for part in &mut parts {
        let post = crate::base::walk::post_order_ids(arena, *part);
        let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
        for &id in &post {
            let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
            let new = match arena.node(rebuilt).clone() {
                ExprNode::Re(u) if is_real_radical(arena, u) => u,
                ExprNode::Im(u) if is_real_radical(arena, u) => arena.zero,
                ExprNode::Cos(a) | ExprNode::Sin(a) => {
                    let (c, t) = arena.as_coeff_term(a);
                    if c.is_negative() && t == arena.pi {
                        let pos = ode_ratio_to_expr(arena, &(-c));
                        let pos = arena.mul(&[pos, t]);
                        if matches!(arena.node(rebuilt), ExprNode::Cos(_)) {
                            arena.cos(pos)
                        } else {
                            let s = arena.sin(pos);
                            arena.neg(s)
                        }
                    } else {
                        rebuilt
                    }
                }
                _ => rebuilt,
            };
            cache.insert(id, new);
        }
        let out = cache.get(part).copied().unwrap_or(*part);
        *part = crate::transforms::eval::eval(arena, out);
    }
    (parts[0], parts[1])
}

/// Is the constant `u` real by construction: free of `i`, with every
/// fractional power and logarithm taken of a positive value (decided
/// numerically, with a margin), and real at its value?
fn is_real_radical(arena: &Arena, u: ExprId) -> bool {
    if !real_radical_form(arena, u) || !crate::base::walk::free_symbols(arena, u).is_empty() {
        return false;
    }
    let positive = |b: ExprId| {
        crate::transforms::evalf::evalf_complex64(arena, b)
            .is_ok_and(|z| z.im == 0.0 && z.re > 1e-12 * (1.0 + z.re.abs()))
    };
    let mut stack = vec![u];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let node = arena.node(n);
        match node {
            ExprNode::Pow(b, e) if !arena.as_num(*e).is_some_and(Q::is_integer) => {
                if !positive(*b) {
                    return false;
                }
            }
            ExprNode::Ln(b) if !positive(*b) => return false,
            _ => {}
        }
        node.for_each_child(|c| stack.push(c));
    }
    crate::transforms::evalf::evalf_complex64(arena, u)
        .is_ok_and(|z| z.im == 0.0 && z.re.is_finite())
}

/// Is `e` free of `i` and of the nodes `re`, `im`, `arg`, `conjugate`
/// (a real part that [`Arena::as_real_imag_expr`] could not take)?
fn real_radical_form(arena: &Arena, e: ExprId) -> bool {
    let i_unit = arena.i_unit;
    let mut stack = vec![e];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if n == i_unit {
            return false;
        }
        let node = arena.node(n);
        if matches!(
            node,
            ExprNode::Re(_) | ExprNode::Im(_) | ExprNode::Arg(_) | ExprNode::Conjugate(_)
        ) {
            return false;
        }
        node.for_each_child(|c| stack.push(c));
    }
    true
}

/// `(re h(z), im h(z))` for the rational polynomial `h` at `z = re + i·im`
/// with real `re`, `im`: Horner's rule on the pair of parts.
fn horner_complex(
    arena: &mut Arena,
    h: &crate::poly::Poly,
    re: ExprId,
    im: ExprId,
) -> (ExprId, ExprId) {
    let (mut a, mut b) = (arena.zero, arena.zero);
    for c in h.coeffs().iter().rev() {
        // (a + i·b)·(re + i·im) + c
        let a_re = arena.mul(&[a, re]);
        let b_im = arena.mul(&[b, im]);
        let neg_b_im = arena.neg(b_im);
        let c = ode_ratio_to_expr(arena, c);
        let a_new = arena.add(&[a_re, neg_b_im, c]);
        let a_im = arena.mul(&[a, im]);
        let b_re = arena.mul(&[b, re]);
        let b_new = arena.add(&[a_im, b_re]);
        a = crate::transforms::eval::eval(arena, a_new);
        b = crate::transforms::eval::eval(arena, b_new);
    }
    (a, b)
}

/// Does `e` contain a `RootOf`?
fn has_rootof(arena: &Arena, e: ExprId) -> bool {
    let mut stack = vec![e];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let node = arena.node(n);
        if matches!(node, ExprNode::RootOf(..)) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// Precision (bits) of the root balls [`rootof_char_roots`] certifies.
const ROOTOF_PREC: usize = 128;

/// The roots of the rational polynomial `g` (square-free, normally
/// irreducible) as `RootOf(g, k)`, SymPy's `CRootOf` form: a real root
/// `RootOf(g, k)`, a non-real pair `re(RootOf(g, k)) ± i·im(RootOf(g, k))`
/// for the index `k` of its member with positive imaginary part.  Which
/// roots are real, and the order the indices refer to, are those of the
/// certified root balls `RootOf` is evaluated with; `None` when a ball is
/// not certified.  The bound variable is named apart from `var`.
fn rootof_char_roots(
    arena: &mut Arena,
    g: &crate::poly::Poly,
    var: ExprId,
) -> Option<Vec<CharRoot>> {
    let deg = g.degree()?;
    let balls = crate::poly::roots::certified_roots(g, ROOTOF_PREC);
    if balls.len() != deg
        || balls.iter().any(|b| b.radius.is_none())
        || balls.iter().filter(|b| b.real).count()
            != crate::poly::sturm::SturmChain::new(g).count_real_roots()
    {
        return None;
    }
    let var_name = match arena.node(var) {
        ExprNode::Symbol(s) => arena.symbol_name(*s).to_string(),
        _ => String::new(),
    };
    let name = ["r", "s", "u", "w"]
        .into_iter()
        .find(|n| *n != var_name)
        .unwrap_or("r");
    let bound = arena.symbol(name);
    let g_expr = crate::poly::polybridge::poly_to_expr(arena, g, bound);
    let mut out = Vec::new();
    let mut pairs = 0usize;
    for (k, ball) in balls.iter().enumerate() {
        let idx = arena.int(i64::try_from(k).ok()?);
        let root = arena.intern(ExprNode::RootOf(g_expr, bound, idx));
        if ball.real {
            out.push(CharRoot::Real(root));
        } else if crate::base::bigfloat::is_sign_positive(&ball.value.1) {
            let re = arena.re(root);
            let im = arena.im(root);
            out.push(CharRoot::Pair { re, im });
            pairs += 1;
        }
    }
    (out.len() + pairs == deg).then_some(out)
}

/// Is `expr = 0` linear in `func` with coefficients free of `var` and
/// `func` (`Σ aₖ·y⁽ᵏ⁾ + g(x)`, each term `aₖ·y⁽ᵏ⁾` or free of `func`)?
/// Its homogeneous solutions are then translation invariant.
pub(crate) fn is_linear_autonomous(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
) -> bool {
    let ExprNode::Symbol(func_sym) = *arena.node(func) else {
        return false;
    };
    let chain = derivative_chain(arena, func, var);
    let children: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![expr],
    };
    let mut has_func = false;
    for child in children {
        let (_, term) = arena.as_coeff_term(child);
        if symbolic_linear_term(arena, term, &chain, var, func_sym).is_some() {
            has_func = true;
        } else if contains_sym(arena, child, func_sym) {
            return false;
        }
    }
    has_func
}

/// Is `expr = 0` a Cauchy–Euler equation `Σ aₖ·xᵏ·y⁽ᵏ⁾ + g(x) = 0`
/// (`aₖ` free of `var` and `func`)?  Its homogeneous solutions are then
/// invariant under `x → λ·x`.
pub(crate) fn is_cauchy_euler(arena: &mut Arena, expr: ExprId, func: ExprId, var: ExprId) -> bool {
    let ExprNode::Symbol(func_sym) = *arena.node(func) else {
        return false;
    };
    let ExprNode::Symbol(var_sym) = *arena.node(var) else {
        return false;
    };
    let chain = derivative_chain(arena, func, var);
    let children: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![expr],
    };
    let mut has_func = false;
    for child in children {
        if !contains_sym(arena, child, func_sym) {
            continue;
        }
        let factors: Vec<ExprId> = match arena.node(child) {
            ExprNode::Mul(fs) => fs.to_vec(),
            ExprNode::Neg(inner) => vec![arena.neg_one, *inner],
            _ => vec![child],
        };
        let mut order = None;
        let mut rest = Vec::with_capacity(factors.len());
        for f in factors {
            match chain.iter().position(|&d| d == f) {
                Some(k) if order.is_none() => order = Some(k),
                Some(_) => return false,
                None if contains_sym(arena, f, func_sym) => return false,
                None => rest.push(f),
            }
        }
        let Some(k) = order else {
            return false;
        };
        // rest / x^k free of x
        let r = arena.mul(&rest);
        let xk = {
            let e = arena.int(i64::try_from(k).unwrap_or(0));
            arena.pow(var, e)
        };
        let q = arena.div(r, xk);
        let q = crate::transforms::eval::eval(arena, q);
        if contains_sym(arena, q, var_sym) {
            return false;
        }
        has_func = true;
    }
    has_func
}

/// The solution of `Σ aₖ·y⁽ᵏ⁾ = g(x)` (rational `aₖ`, order `n ≥ 2`) with
/// `y⁽ᵏ⁾(x₀) = values[k]` for `k < n`, when the characteristic polynomial
/// `p` is square-free and has an irreducible factor of degree `≥ 3`
/// (roots in Cardano's radicals or `RootOf`, see [`characteristic_roots`]);
/// `None` otherwise (the generic fit applies).
///
/// With `y = yₚ + Σ cᵣ·e^{r·(x − x₀)}` the conditions are the Vandermonde
/// system `Σ cᵣ·rᵏ = wₖ = values[k] − yₚ⁽ᵏ⁾(x₀)`, whose inverse is given by
/// the Lagrange polynomials: `cᵣ = Σₖ bₖ(r)·wₖ / p′(r)` with
/// `p(t) = (t − r)·Σ bₖ(r)·tᵏ`.  Reduced modulo the irreducible factor
/// `g` of `r`, `bₖ/p′ = hₖ mod g` has rational coefficients, so a
/// non-real pair contributes `e^{α(x−x₀)}·(A·cos β(x−x₀) + B·sin β(x−x₀))`
/// with `A = 2·Σ wₖ·re hₖ(r)`, `B = −2·Σ wₖ·im hₖ(r)` (`hₖ(re + i·im)` by
/// Horner's rule on the pair of parts).  The generic fit (symbolic
/// elimination) did not finish in a minute for `y⁽⁵⁾ − y′ + y = 0` (over
/// `re`/`im` of `RootOf`s) nor for `y‴ + y″ + 2y = 0` with `y(1/3) = 2/3`,
/// `y′(1/3) = 5/4`, `y″(1/3) = −3/7` (over Cardano's radicals).
pub(crate) fn linear_cc_ivp_lagrange(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    x0: ExprId,
    values: &[ExprId],
) -> Option<ExprId> {
    use crate::poly::Poly;
    let ExprNode::Symbol(func_sym) = *arena.node(func) else {
        return None;
    };
    let Some(ode) = extract_linear_cc(arena, expr, func, var, func_sym) else {
        return euler_ivp_lagrange(arena, expr, func, var, x0, values);
    };
    let n = ode.coeffs.len() - 1;
    if n < 2 || values.len() != n {
        return None;
    }
    let p = Poly::from_coeffs(ode.coeffs.clone()).make_monic();
    if p.squarefree_factors().iter().any(|(_, m)| *m != 1) {
        return None;
    }
    let r_sym = arena.symbol("__r_cc");
    let (_, irreducible) = p.factor_over_z();
    if irreducible.iter().all(|(g, _)| g.degree().unwrap_or(0) < 3) {
        return None;
    }
    let mut groups: Vec<(Poly, Vec<CharRoot>)> = Vec::with_capacity(irreducible.len());
    for (g, _) in irreducible {
        let roots = match radical_char_roots(arena, &g, r_sym) {
            Some(r) => r,
            None => rootof_char_roots(arena, &g, var)?,
        };
        groups.push((g, roots));
    }
    // The particular solution and the conditions left for the modes.
    let general = try_nth_order_linear_const_coeff(arena, expr, func, var, func_sym)?;
    let mut y_p = general.solution;
    let zero = arena.zero;
    for &c in &general.constants {
        y_p = crate::transforms::subs::subs(arena, y_p, c, zero);
    }
    let y_p = crate::transforms::eval::eval(arena, y_p);
    let mut w = Vec::with_capacity(n);
    let mut d = y_p;
    for &v in values {
        let at = crate::transforms::subs::subs(arena, d, var, x0);
        let wk = arena.sub(v, at);
        w.push(crate::transforms::eval::eval(arena, wk));
        d = crate::transforms::diff::diff(arena, d, var);
    }
    // b_{n−1} = 1, b_{k−1} = a_k + t·b_k (p monic).
    let t_poly = Poly::from_coeffs(vec![Q::from_integer(0.into()), Q::from_integer(1.into())]);
    let mut b = vec![Poly::from_coeffs(vec![Q::from_integer(1.into())]); n];
    for k in (1..n).rev() {
        let ak = Poly::from_coeffs(vec![p.coeff(k)]);
        b[k - 1] = ak.add(&t_poly.mul(&b[k]));
    }
    let dp = p.derivative();
    let t_sym = arena.symbol("__t_cc");
    let shift = arena.sub(var, x0);
    let two = arena.int(2);
    let mut terms = vec![y_p];
    for (g, roots) in &groups {
        // x·p′ + y·g = gcd = 1 (monic): x = 1/p′ mod g.
        let e = Poly::extended_gcd(&dp.rem(g), g);
        if e.gcd.degree() != Some(0) {
            return None;
        }
        let inv_dp = e.x;
        let h: Vec<Poly> = b.iter().map(|bk| bk.mul(&inv_dp).rem(g)).collect();
        for root in roots {
            match *root {
                CharRoot::Real(r) => {
                    let mut c = Vec::with_capacity(n);
                    for (hk, wk) in h.iter().zip(&w) {
                        let hk = crate::poly::polybridge::poly_to_expr(arena, hk, t_sym);
                        let at = crate::transforms::subs::subs(arena, hk, t_sym, r);
                        c.push(arena.mul(&[*wk, at]));
                    }
                    let c = arena.add(&c);
                    let rx = arena.mul(&[r, shift]);
                    let e = arena.exp(rx);
                    terms.push(arena.mul(&[c, e]));
                }
                CharRoot::Pair { re, im } => {
                    let (mut a, mut bb) = (Vec::with_capacity(n), Vec::with_capacity(n));
                    for (hk, wk) in h.iter().zip(&w) {
                        let (h_re, h_im) = horner_complex(arena, hk, re, im);
                        a.push(arena.mul(&[*wk, h_re]));
                        bb.push(arena.mul(&[*wk, h_im]));
                    }
                    let a = arena.add(&a);
                    let a = arena.mul(&[two, a]);
                    let bsum = arena.add(&bb);
                    let neg_two = arena.int(-2);
                    let bcoef = arena.mul(&[neg_two, bsum]);
                    let ax = arena.mul(&[re, shift]);
                    let env = arena.exp(ax);
                    let bx = arena.mul(&[im, shift]);
                    let cos_bx = arena.cos(bx);
                    let sin_bx = arena.sin(bx);
                    let ca = arena.mul(&[a, cos_bx]);
                    let sb = arena.mul(&[bcoef, sin_bx]);
                    let osc = arena.add(&[ca, sb]);
                    terms.push(arena.mul(&[env, osc]));
                }
            }
        }
    }
    let sol = arena.add(&terms);
    Some(crate::transforms::eval::eval(arena, sol))
}

/// `x^j·e^{r·x}` (with `e^{0}` folded away).
fn mode_real(arena: &mut Arena, r: ExprId, j: usize, var: ExprId) -> ExprId {
    let x_pow = match j {
        0 => arena.one,
        1 => var,
        _ => {
            let e = arena.int(j as i64);
            arena.pow(var, e)
        }
    };
    if arena.is_zero_structural(r) {
        return x_pow;
    }
    let rx = arena.mul(&[r, var]);
    let exp_rx = arena.exp(rx);
    let m = arena.mul(&[x_pow, exp_rx]);
    crate::transforms::eval::eval(arena, m)
}

/// `(x^j e^{αx} cos βx, x^j e^{αx} sin βx)`.
fn mode_complex(
    arena: &mut Arena,
    alpha: ExprId,
    beta: ExprId,
    j: usize,
    var: ExprId,
) -> (ExprId, ExprId) {
    let envelope = mode_real(arena, alpha, j, var);
    let bx = arena.mul(&[beta, var]);
    let cos_bx = arena.cos(bx);
    let sin_bx = arena.sin(bx);
    let c = arena.mul(&[envelope, cos_bx]);
    let s = arena.mul(&[envelope, sin_bx]);
    (
        crate::transforms::eval::eval(arena, c),
        crate::transforms::eval::eval(arena, s),
    )
}

/// Solve `Σ a_k y^(k) + g(x) = 0` of any order `≥ 2` with rational
/// coefficients, where `g` is a sum of `poly·exp·{sin|cos}` terms (or
/// empty).  Homogeneous part via the characteristic polynomial (repeated
/// and complex roots handled), particular part via undetermined
/// coefficients with resonance handling.
fn try_nth_order_linear_const_coeff(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Option<OdeResult> {
    use num_traits::Zero;
    let ode = extract_linear_cc(arena, expr, func, var, func_sym)?;
    let n = ode.coeffs.len() - 1;
    if n < 2 {
        return None;
    }
    tracing::debug!(order = n, "ode: nth-order linear constant-coefficient");

    // Forcing: L[y] = -g(x).  Parse and group by (a, b, kind); hyperbolic
    // functions as exponentials, terms with irrational constants apart.
    let hyperbolic = hyperbolic_forcing_as_exp(arena, &ode.forcing);
    let had_hyperbolic = hyperbolic.is_some();
    let mut forcing = hyperbolic.unwrap_or(ode.forcing);
    // A product of sums (`(x + ln 2)·eˣ`) is parsed term by term.
    if forcing.iter().any(|&t| {
        parse_forcing_term(arena, t, var).is_none()
            && parse_forcing_term_sym(arena, t, var).is_none()
    }) {
        let sum = arena.add(&forcing);
        let sum = crate::transforms::expand::expand(arena, sum);
        let sum = crate::transforms::eval::eval(arena, sum);
        forcing = match arena.node(sum) {
            ExprNode::Add(c) => c.to_vec(),
            _ => vec![sum],
        };
    }
    let mut groups: Vec<((Q, Q, TrigKind), Vec<Q>)> = Vec::new();
    let mut sym_groups: Vec<((ExprId, ExprId, TrigKind), Vec<ExprId>)> = Vec::new();
    for &term in &forcing {
        let Some(ft) = parse_forcing_term(arena, term, var) else {
            let ft = parse_forcing_term_sym(arena, term, var)?;
            let key = (ft.a, ft.b, ft.kind);
            let index = match sym_groups.iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    sym_groups.push((key, Vec::new()));
                    sym_groups.len() - 1
                }
            };
            let entry = &mut sym_groups[index].1;
            if entry.len() <= ft.degree {
                entry.resize(ft.degree + 1, arena.zero);
            }
            // move to the right-hand side
            let slot = arena.sub(entry[ft.degree], ft.coeff);
            entry[ft.degree] = crate::transforms::eval::eval(arena, slot);
            continue;
        };
        let key = (ft.a.clone(), ft.b.clone(), ft.kind);
        let entry = match groups.iter_mut().find(|(k, _)| *k == key) {
            Some(e) => e,
            None => {
                groups.push((key, Vec::new()));
                groups.last_mut()?
            }
        };
        if entry.1.len() <= ft.degree {
            entry.1.resize(ft.degree + 1, Q::zero());
        }
        entry.1[ft.degree] -= ft.coeff; // move to the right-hand side
    }

    let basis = homogeneous_basis_cc(arena, &ode.coeffs, var)?;
    if basis.len() != n {
        return None;
    }
    let mut constants = Vec::with_capacity(n);
    let mut terms = Vec::with_capacity(n + groups.len());
    for (i, &b) in basis.iter().enumerate() {
        let c = arena.symbol(&format!("C{}", i + 1));
        constants.push(c);
        terms.push(arena.mul(&[c, b]));
    }
    let mut particular = Vec::with_capacity(groups.len() + sym_groups.len());
    for ((a, b, kind), q) in &groups {
        if q.iter().all(Zero::is_zero) {
            continue;
        }
        let y_p = particular_for_group(arena, &ode.coeffs, q, a, b, *kind, var)?;
        particular.push(y_p);
    }
    for ((a, b, kind), q) in &sym_groups {
        if q.iter().all(|&c| arena.is_zero_structural(c)) {
            continue;
        }
        let y_p = particular_for_group_sym(arena, &ode.coeffs, q, *a, *b, *kind, var)?;
        particular.push(y_p);
    }
    if had_hyperbolic && !particular.is_empty() {
        let y_p = arena.add(&particular);
        let y_p = crate::transforms::expand::expand(arena, y_p);
        let y_p = crate::transforms::eval::eval(arena, y_p);
        particular = vec![exp_pairs_as_hyperbolic(arena, y_p, var)];
    }
    terms.extend(particular);
    let solution = arena.add(&terms);
    let solution = crate::transforms::eval::eval(arena, solution);
    Some(OdeResult {
        solution,
        constants,
        branches: Vec::new(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear ODEs with constant symbolic coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// `(k, c)` if `term` is `c·y⁽ᵏ⁾` with `c` free of `x` and `y`.
fn symbolic_linear_term(
    arena: &mut Arena,
    term: ExprId,
    chain: &[ExprId],
    var: ExprId,
    func_sym: SymbolId,
) -> Option<(usize, ExprId)> {
    if let Some(k) = chain.iter().position(|&d| d == term) {
        return Some((k, arena.one));
    }
    let ExprNode::Mul(factors) = arena.node(term).clone() else {
        return None;
    };
    let mut order = None;
    let mut rest = Vec::with_capacity(factors.len());
    for f in factors {
        if let Some(k) = chain.iter().position(|&d| d == f) {
            if order.replace(k).is_some() {
                return None;
            }
        } else if crate::base::walk::contains(arena, f, var) || contains_sym(arena, f, func_sym) {
            return None;
        } else {
            rest.push(f);
        }
    }
    Some((order?, arena.mul(&rest)))
}

/// `e` with every `Piecewise` whose first condition is a parameter
/// inequality (`a ≠ 0`, conjunctions of those; free of `var`) replaced by
/// its first, generic branch; `None` if another `Piecewise` remains.
fn generic_branches(arena: &mut Arena, e: ExprId, var: ExprId) -> Option<ExprId> {
    fn is_generic(arena: &Arena, c: ExprId) -> bool {
        match arena.node(c) {
            ExprNode::Ne(..) => true,
            ExprNode::And(cs) => cs
                .iter()
                .all(|&k| matches!(arena.node(k), ExprNode::Ne(..))),
            _ => false,
        }
    }
    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Piecewise(pairs) => {
                let (v, c) = *pairs.first()?;
                if crate::base::walk::contains(arena, c, var) || !is_generic(arena, c) {
                    return None;
                }
                v
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    let r = cache.get(&e).copied().unwrap_or(e);
    Some(crate::transforms::eval::eval(arena, r))
}

/// Is `e` identically zero after expansion (structurally)?
fn expands_to_zero(arena: &mut Arena, e: ExprId) -> bool {
    let e = crate::transforms::eval::eval(arena, e);
    let e = crate::transforms::expand::expand(arena, e);
    let e = crate::transforms::eval::eval(arena, e);
    arena.is_zero_structural(e)
}

/// Solve `Σ a_k·y⁽ᵏ⁾ = g(x)` whose constant coefficients are not all
/// rational (`y″ + a·y = 0`, `y″ + a·y′ + y = 0`): the roots of the
/// characteristic polynomial in radicals (principal branches), each checked
/// by substitution and given its multiplicity by the vanishing derivatives,
/// so the basis `xʲ·e^{r·x}`; SymPy's form, `C1·e^{−x√(−a)} +
/// C2·e^{x√(−a)}` for `y″ + a·y = 0`, with no case split on the sign of
/// `a` (the exponentials cover both signs: `e^{±x√(−a)}` are the real
/// exponentials for `a < 0` and `e^{±ix√a}` for `a > 0`).  The solution is
/// the generic one: it assumes the leading coefficient non-zero and roots
/// that are distinct as expressions distinct as values (at `a = 0` the
/// basis `e^{±x√(−a)}` degenerates, as in SymPy).  A second-order
/// equation with a forcing term gets its particular solution by variation
/// of parameters; higher orders with forcing are left to other methods.
fn try_linear_cc_symbolic(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
) -> Option<OdeResult> {
    let chain = derivative_chain(arena, func, var);
    let children: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Add(c) => c.to_vec(),
        _ => vec![expr],
    };
    let mut coeffs: Vec<Vec<ExprId>> = vec![Vec::new(); MAX_ODE_ORDER + 1];
    let mut forcing = Vec::new();
    for child in children {
        if let Some((k, c)) = symbolic_linear_term(arena, child, &chain, var, func_sym) {
            coeffs[k].push(c);
        } else if !contains_sym(arena, child, func_sym) {
            forcing.push(child);
        } else {
            return None;
        }
    }
    let mut a: Vec<ExprId> = Vec::with_capacity(coeffs.len());
    for cs in &coeffs {
        let s = match cs.len() {
            0 => arena.zero,
            1 => cs[0],
            _ => arena.add(cs),
        };
        a.push(crate::transforms::eval::eval(arena, s));
    }
    while a.len() > 1 && a.last().is_some_and(|&c| arena.is_zero_structural(c)) {
        a.pop();
    }
    let n = a.len() - 1;
    // Order 1 is the first-order linear solver's; all-rational coefficients
    // are `try_nth_order_linear_const_coeff`'s.
    if n < 2 || a.iter().all(|&c| arena.as_num(c).is_some()) {
        return None;
    }
    if !forcing.is_empty() && n != 2 {
        return None;
    }
    let r = arena.symbol("__r_sym");
    let mut terms = Vec::with_capacity(n + 1);
    for (k, &c) in a.iter().enumerate() {
        let rk = match k {
            0 => arena.one,
            1 => r,
            _ => {
                let ke = arena.int(k as i64);
                arena.pow(r, ke)
            }
        };
        terms.push(arena.mul(&[c, rk]));
    }
    let p = arena.add(&terms);
    let p = crate::transforms::eval::eval(arena, p);
    // Factor over ℤ[parameters] first (`r³ + a·r = r·(r² + a)`, `r² +
    // (a + 1)r + a = (r + 1)(r + a)`: the solver alone finds neither the
    // cubic's roots nor the rational roots of the quadratic, only
    // `√((a − 1)²)`); a factor linear in `r` gives its root directly, the
    // others go to the solver.  If `p` does not factor this way it is
    // solved whole.
    let (_, factors) = crate::simplify::factor::factor_list(arena, p, Some(r));
    let mut basis = Vec::with_capacity(n);
    let mut exps: Vec<(ExprId, usize)> = Vec::new();
    for (f, m) in factors {
        if !crate::base::walk::contains(arena, f, r) {
            continue; // a (generically non-zero) constant factor
        }
        let coeffs = crate::poly::polybridge::poly_coefficients(arena, f, r);
        let roots: Vec<ExprId> = match coeffs.as_deref() {
            Some([c0, c1]) => {
                let q = arena.div(*c0, *c1);
                vec![arena.neg(q)]
            }
            _ => known_roots(arena, f, r)
                .into_iter()
                .map(|s| s.value)
                .collect(),
        };
        if roots.is_empty() {
            return None;
        }
        // Multiplicity of each root in `f`: the number of derivatives
        // f, f′, … that vanish there (the first must, or the root is not
        // verified); in `p` it is that times the factor's multiplicity.
        for r0 in roots {
            let r0 = crate::transforms::eval::eval(arena, r0);
            if crate::base::walk::contains(arena, r0, var) || exps.iter().any(|&(e, _)| e == r0) {
                return None;
            }
            let mut d = f;
            let mut mult = 0usize;
            while mult < n {
                let at = crate::transforms::subs::subs(arena, d, r, r0);
                if !expands_to_zero(arena, at) {
                    break;
                }
                mult += 1;
                d = crate::transforms::diff::diff(arena, d, r);
            }
            if mult == 0 {
                return None;
            }
            let mult = mult * m as usize;
            exps.push((r0, mult));
            for j in 0..mult {
                basis.push(mode_real(arena, r0, j, var));
            }
        }
    }
    if basis.len() != n {
        return None;
    }
    let homogeneous = real_symbolic_basis(arena, &exps, var).unwrap_or_else(|| basis.clone());
    let mut constants = Vec::with_capacity(n);
    let mut sum = Vec::with_capacity(n + 1);
    for (i, &b) in homogeneous.iter().enumerate() {
        let c = arena.symbol(&format!("C{}", i + 1));
        constants.push(c);
        sum.push(arena.mul(&[c, b]));
    }
    if !forcing.is_empty() {
        // a₂y″ + a₁y′ + a₀y = −f:  y_p = −y₁∫y₂g/W + y₂∫y₁g/W, g = −f/a₂.
        let (y1, y2) = (basis[0], basis[1]);
        let f = arena.add(&forcing);
        let nf = arena.neg(f);
        let g = arena.div(nf, a[2]);
        let g = crate::transforms::eval::eval(arena, g);
        // W = y₁y₂′ − y₂y₁′: (r₂ − r₁)e^{(r₁+r₂)x}, or e^{2rx} for a double root.
        let w = if exps.len() == 2 {
            let d = arena.sub(exps[1].0, exps[0].0);
            let s = arena.add(&[exps[0].0, exps[1].0]);
            let sx = arena.mul(&[s, var]);
            let e = arena.exp(sx);
            arena.mul(&[d, e])
        } else {
            let two = arena.int(2);
            let sx = arena.mul(&[two, exps[0].0, var]);
            arena.exp(sx)
        };
        let w = crate::transforms::eval::eval(arena, w);
        let i1 = {
            let num = arena.mul(&[y2, g]);
            let q = arena.div(num, w);
            let q = crate::transforms::eval::eval(arena, q);
            integrate_forms(arena, q, var)
        };
        if integral_failed(arena, i1) {
            return None;
        }
        let i2 = {
            let num = arena.mul(&[y1, g]);
            let q = arena.div(num, w);
            let q = crate::transforms::eval::eval(arena, q);
            integrate_forms(arena, q, var)
        };
        if integral_failed(arena, i2) {
            return None;
        }
        let t1 = arena.mul(&[y1, i1]);
        let t1 = arena.neg(t1);
        let t2 = arena.mul(&[y2, i2]);
        let yp = arena.add(&[t1, t2]);
        // The integrals of `e^{r·x}·g` come back as `Piecewise` on the
        // parameters (`r ≠ 0`); the homogeneous part is already the generic
        // solution, so the generic branch is taken throughout.
        let yp = generic_branches(arena, yp, var)?;
        let yp = expand_combine(arena, yp);
        let yp = crate::poly::polybridge::together(arena, yp);
        let yp = crate::transforms::eval::eval(arena, yp);
        if crate::base::walk::has_unevaluated(arena, yp) {
            return None;
        }
        sum.push(yp);
    }
    let solution = arena.add(&sum);
    let solution = crate::transforms::eval::eval(arena, solution);
    // The particular solution took the generic branch of its integrals:
    // it is returned only if it verifies.
    if !forcing.is_empty() && !checkodesol(arena, expr, solution, func, var) {
        return None;
    }
    Some(OdeResult {
        solution,
        constants,
        branches: Vec::new(),
    })
}

/// The basis `xʲ·e^{r·x}` of [`try_linear_cc_symbolic`] made real where
/// the declared assumptions decide it: a root whose real and imaginary
/// parts are free of `i` and of unevaluated `re`/`im` (`√(−a)` with `a`
/// positive is `i·√a`), with a non-zero imaginary part and its conjugate
/// among the roots at the same multiplicity, gives `xʲ·e^{αx}·cos(βx)` and
/// `xʲ·e^{αx}·sin(βx)` (SymPy's form for a positive `a`: `y″ + a·y = 0`
/// gives `C1·cos(√a·x) + C2·sin(√a·x)`; before, the exponentials
/// `e^{±x√(−a)}`, and the initial-value solution `e^{−i√a·x}/2 + e^{i√a·x}/2`).
/// `None` when no root changes (the generic, assumption-free form stays).
fn real_symbolic_basis(
    arena: &mut Arena,
    exps: &[(ExprId, usize)],
    var: ExprId,
) -> Option<Vec<ExprId>> {
    let mut reals = crate::base::assumptions::AssumptionCache::new();
    let parts: Vec<(ExprId, ExprId)> = exps
        .iter()
        .map(|&(r, _)| {
            // √u with −u declared positive is i·√(−u).
            let post = crate::base::walk::post_order_ids(arena, r);
            let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
            let half = Q::new(1.into(), 2.into());
            for &id in &post {
                let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
                let new = match *arena.node(rebuilt) {
                    ExprNode::Pow(b, e) if arena.as_num(e) == Some(&half) => {
                        let nb = arena.neg(b);
                        let nb = crate::transforms::eval::eval(arena, nb);
                        if reals.query(arena, nb, crate::base::assumptions::Props::POSITIVE)
                            == Some(true)
                        {
                            let s = arena.pow(nb, e);
                            let i_unit = arena.i_unit;
                            arena.mul(&[i_unit, s])
                        } else {
                            rebuilt
                        }
                    }
                    _ => rebuilt,
                };
                cache.insert(id, new);
            }
            let r = cache.get(&r).copied().unwrap_or(r);
            let r = crate::transforms::eval::eval(arena, r);
            let (re, im) = arena.as_real_imag_expr(r);
            (
                crate::transforms::eval::eval(arena, re),
                crate::transforms::eval::eval(arena, im),
            )
        })
        .collect();
    let mut used = vec![false; exps.len()];
    let mut out = Vec::new();
    let mut changed = false;
    for i in 0..exps.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let (r, mult) = exps[i];
        let (re, im) = parts[i];
        let decided = !arena.is_zero_structural(im)
            && real_radical_form(arena, re)
            && real_radical_form(arena, im);
        let partner = decided
            .then(|| {
                (0..exps.len()).find(|&k| {
                    if used[k] || exps[k].1 != mult {
                        return false;
                    }
                    let (re2, im2) = parts[k];
                    let dre = arena.sub(re, re2);
                    let sim = arena.add(&[im, im2]);
                    expands_to_zero(arena, dre) && expands_to_zero(arena, sim)
                })
            })
            .flatten();
        match partner {
            Some(k) => {
                used[k] = true;
                changed = true;
                // β with the sign it has structurally positive when known.
                let beta = if arena.as_num(im).is_some_and(|q| q.is_negative())
                    || matches!(arena.node(im), ExprNode::Neg(_))
                {
                    parts[k].1
                } else {
                    im
                };
                for j in 0..mult {
                    let (c, s) = mode_complex(arena, re, beta, j, var);
                    out.push(c);
                    out.push(s);
                }
            }
            None => {
                for j in 0..mult {
                    out.push(mode_real(arena, r, j, var));
                }
            }
        }
    }
    changed.then_some(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// Clairaut: y = x·y' + f(y')
// ═══════════════════════════════════════════════════════════════════════════

/// Recognise the Clairaut form `y = x·y' + f(y')` and return `f(p)` with
/// `p` standing for `y'`.
fn clairaut_f(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
    p: ExprId,
) -> Option<ExprId> {
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));
    if !expr_contains(arena, expr, dy_dx) || expr_contains(arena, expr, d2y_dx2) {
        return None;
    }
    let g = crate::transforms::subs::subs(arena, expr, dy_dx, p);
    // Linear in y with a constant coefficient.
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, g, func)?;
    if coeffs.len() != 2 {
        return None;
    }
    let c_y = coeffs[1];
    let c_0 = coeffs[0];
    let p_sym = match arena.node(p) {
        ExprNode::Symbol(s) => *s,
        _ => return None,
    };
    if contains_sym(arena, c_y, var_sym) || contains_sym(arena, c_y, p_sym) {
        return None;
    }
    // g / c_y = y + c_0/c_y  must equal  y - x·p - f(p)  ⇒  f = -c_0/c_y - x·p
    let ratio = arena.div(c_0, c_y);
    let neg_ratio = arena.neg(ratio);
    let xp = arena.mul(&[var, p]);
    let f = arena.sub(neg_ratio, xp);
    let f = crate::transforms::eval::eval(arena, f);
    let f = crate::transforms::expand::expand(arena, f);
    let f = crate::transforms::eval::eval(arena, f);
    if contains_sym(arena, f, var_sym) || contains_sym(arena, f, func_sym) {
        return None;
    }
    Some(f)
}

/// Solve a Clairaut equation `y = x·y' + f(y')`: the general solution is
/// the family of lines `y = C·x + f(C)`.  (The singular envelope solution
/// is not returned.)
fn try_clairaut(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    let p = arena.symbol("__clairaut_p");
    let f = clairaut_f(arena, expr, func, var, func_sym, var_sym, p)?;
    tracing::debug!("ode: Clairaut form recognised");
    let c1 = arena.symbol("C1");
    let f_c = crate::transforms::subs::subs(arena, f, p, c1);
    let cx = arena.mul(&[c1, var]);
    let solution = arena.add(&[cx, f_c]);
    let solution = crate::transforms::eval::eval(arena, solution);
    Some(OdeResult {
        solution,
        constants: vec![c1],
        branches: Vec::new(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Riccati: y' = q0(x) + q1(x)·y + q2(x)·y²
// ═══════════════════════════════════════════════════════════════════════════

/// Extract `(q0, q1, q2)` from a Riccati equation in zero form,
/// `c·y' - (q0 + q1·y + q2·y²) = 0` with `c` a nonzero constant.
/// Requires `q2 ≠ 0` and `q0 ≠ 0` (otherwise the equation is Bernoulli /
/// linear) and coefficients free of `y` and `y'`.
fn riccati_coeffs(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));
    if expr_contains(arena, expr, d2y_dx2) {
        return None;
    }
    let (rest, c) = extract_m_n(arena, expr, func, var)?;
    if crate::base::walk::contains(arena, c, func) || crate::base::walk::contains(arena, c, var) {
        return None;
    }
    if crate::base::walk::contains(arena, rest, dy_dx) {
        return None;
    }
    // y' = -rest/c
    let neg_rest = arena.neg(rest);
    let rhs = arena.div(neg_rest, c);
    let rhs = crate::transforms::eval::eval(arena, rhs);
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, rhs, func)?;
    if coeffs.len() != 3 {
        return None;
    }
    let (q0, q1, q2) = (coeffs[0], coeffs[1], coeffs[2]);
    if arena.is_zero_structural(q0) || arena.is_zero_structural(q2) {
        return None;
    }
    Some((q0, q1, q2))
}

/// Solve a Riccati equation `y' = q0(x) + q1(x)·y + q2(x)·y²` given a known
/// particular solution `y_p(x)`.
///
/// The substitution `y = y_p + 1/v` turns the equation into the linear
/// ODE `v' + (q1 + 2·q2·y_p)·v = -q2`, which is solved with [`dsolve`];
/// the result is `y = y_p + 1/v`.
///
/// Returns `None` if `expr` is not of Riccati form, if `particular` does
/// not satisfy it, or if the linear equation for `v` cannot be solved.
pub fn solve_riccati(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    particular: ExprId,
) -> Option<OdeResult> {
    let (q0, q1, q2) = riccati_coeffs(arena, expr, func, var)?;
    // Verify the particular solution.
    if !checkodesol(arena, expr, particular, func, var) {
        // Try the derivative form directly: y_p' - (q0 + q1 y_p + q2 y_p²).
        let yp_prime = crate::transforms::diff::diff(arena, particular, var);
        let yp2 = arena.mul(&[particular, particular]);
        let t1 = arena.mul(&[q1, particular]);
        let t2 = arena.mul(&[q2, yp2]);
        let rhs = arena.add(&[q0, t1, t2]);
        let residual = arena.sub(yp_prime, rhs);
        let residual = crate::transforms::eval::eval(arena, residual);
        let residual = crate::transforms::expand::expand(arena, residual);
        let residual = crate::transforms::eval::eval(arena, residual);
        if !arena.is_zero_structural(residual) {
            return None;
        }
    }
    let _ = q0;
    let v = arena.symbol("__riccati_v");
    let dv = arena.intern(ExprNode::Derivative(v, var));
    // v' + (q1 + 2 q2 y_p) v + q2 = 0
    let two = arena.int(2);
    let two_q2_yp = arena.mul(&[two, q2, particular]);
    let coeff = arena.add(&[q1, two_q2_yp]);
    let coeff = crate::transforms::eval::eval(arena, coeff);
    let coeff_v = arena.mul(&[coeff, v]);
    let lin = arena.add(&[dv, coeff_v, q2]);
    let lin = crate::transforms::eval::eval(arena, lin);
    let v_res = dsolve(arena, lin, v, var)?;
    if crate::base::walk::contains(arena, v_res.solution, v) {
        return None; // implicit
    }
    let neg_one = arena.neg_one;
    let inv_v = arena.pow(v_res.solution, neg_one);
    let solution = arena.add(&[particular, inv_v]);
    let solution = crate::transforms::eval::eval(arena, solution);
    Some(OdeResult {
        solution,
        constants: v_res.constants,
        branches: Vec::new(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational Riccati: rational particular solutions found, general solution
// built from them (SymPy's `1st_rational_riccati`)
// ═══════════════════════════════════════════════════════════════════════════

/// A rational function `num/den` over ℚ in one variable, `den` monic and
/// coprime to `num` (`0/1` for zero).
#[derive(Clone, Debug, PartialEq)]
struct RatFn {
    num: crate::poly::Poly,
    den: crate::poly::Poly,
}

impl RatFn {
    fn new(num: crate::poly::Poly, den: crate::poly::Poly) -> Option<Self> {
        use crate::poly::Poly;
        if den.is_zero() {
            return None;
        }
        if num.is_zero() {
            return Some(Self {
                num: Poly::zero(),
                den: Poly::one(),
            });
        }
        let g = Poly::gcd(&num, &den);
        let (num, den) = if g.degree().unwrap_or(0) > 0 {
            (num.div(&g), den.div(&g))
        } else {
            (num, den)
        };
        let inv = Q::one() / den.leading_coeff()?.clone();
        Some(Self {
            num: num.scale(&inv),
            den: den.scale(&inv),
        })
    }
    fn poly(p: crate::poly::Poly) -> Self {
        Self {
            num: p,
            den: crate::poly::Poly::one(),
        }
    }
    fn constant(c: Q) -> Self {
        Self::poly(crate::poly::Poly::constant(c))
    }
    fn is_zero(&self) -> bool {
        self.num.is_zero()
    }
    fn add(&self, o: &Self) -> Option<Self> {
        Self::new(
            self.num.mul(&o.den).add(&o.num.mul(&self.den)),
            self.den.mul(&o.den),
        )
    }
    fn sub(&self, o: &Self) -> Option<Self> {
        self.add(&o.scale(&-Q::one()))
    }
    fn mul(&self, o: &Self) -> Option<Self> {
        Self::new(self.num.mul(&o.num), self.den.mul(&o.den))
    }
    fn div(&self, o: &Self) -> Option<Self> {
        if o.is_zero() {
            return None;
        }
        Self::new(self.num.mul(&o.den), self.den.mul(&o.num))
    }
    fn scale(&self, c: &Q) -> Self {
        Self::new(self.num.scale(c), self.den.clone()).unwrap_or_else(|| self.clone())
    }
    fn deriv(&self) -> Option<Self> {
        let n = self
            .num
            .derivative()
            .mul(&self.den)
            .sub(&self.num.mul(&self.den.derivative()));
        Self::new(n, self.den.mul(&self.den))
    }
    /// `deg den − deg num` (the valuation at infinity; large for zero).
    fn val_inf(&self) -> i64 {
        let d = |p: &crate::poly::Poly| i64::try_from(p.degree().unwrap_or(0)).unwrap_or(0);
        if self.is_zero() {
            return i64::MAX / 4;
        }
        d(&self.den) - d(&self.num)
    }
    fn to_expr(&self, arena: &mut Arena, var: ExprId) -> ExprId {
        let n = crate::poly::polybridge::poly_to_expr(arena, &self.num, var);
        if self.den.is_constant() {
            return n;
        }
        let d = crate::poly::polybridge::poly_to_expr(arena, &self.den, var);
        arena.div(n, d)
    }
    fn from_expr(arena: &mut Arena, e: ExprId, var: ExprId) -> Option<Self> {
        let t = crate::poly::polybridge::together(arena, e);
        let (n, d) = crate::poly::polybridge::as_numer_denom(arena, t);
        let n = crate::poly::polybridge::expr_to_poly(arena, n, var)?;
        let d = crate::poly::polybridge::expr_to_poly(arena, d, var)?;
        Self::new(n, d)
    }
}

/// The first `count` power-series coefficients of `p/q` at 0 (`q(0) ≠ 0`).
fn power_series(p: &crate::poly::Poly, q: &crate::poly::Poly, count: usize) -> Option<Vec<Q>> {
    use num_traits::Zero;
    let q0 = q.coeff(0);
    if q0.is_zero() {
        return None;
    }
    let qd = q.degree().unwrap_or(0);
    let mut g: Vec<Q> = Vec::with_capacity(count);
    for k in 0..count {
        let mut s = p.coeff(k);
        for j in 1..=k.min(qd) {
            s -= q.coeff(j) * &g[k - j];
        }
        g.push(s / &q0);
    }
    Some(g)
}

/// The Laurent coefficients of `f` at the pole `p` of order `m`: entry `i`
/// is the coefficient of `(x − p)^{i − m}`, `i < count`.
fn laurent_at_pole(f: &RatFn, p: &Q, m: usize, count: usize) -> Option<Vec<Q>> {
    use num_traits::Zero;
    let num = f.num.taylor_shift(p);
    let den = f.den.taylor_shift(p);
    if (0..m).any(|k| !den.coeff(k).is_zero()) {
        return None;
    }
    let den1 = crate::poly::Poly::from_coeffs(den.coeffs().get(m..)?.to_vec());
    power_series(&num, &den1, count)
}

/// The coefficients `rᵢ` of `f(x) = Σ rᵢ·x^{−v−i}` at infinity
/// (`v = val_inf f`), `i < count`.
fn laurent_at_infinity(f: &RatFn, count: usize) -> Option<Vec<Q>> {
    let rev = |p: &crate::poly::Poly| {
        let mut c = p.coeffs().to_vec();
        c.reverse();
        crate::poly::Poly::from_coeffs(c)
    };
    power_series(&rev(&f.num), &rev(&f.den), count)
}

/// `√q` when it is rational.
fn rational_sqrt(q: &Q) -> Option<Q> {
    if q.is_negative() {
        return None;
    }
    let (n, d) = (q.numer(), q.denom());
    let (rn, rd) = (n.sqrt(), d.sqrt());
    (&rn * &rn == *n && &rd * &rd == *d).then(|| Q::new(rn, rd))
}

/// Most finite poles of `a(x)` whose choices [`rational_riccati_particulars`]
/// combines (`2^(poles + 1)` combinations).
const MAX_RICCATI_POLES: usize = 6;
/// Largest degree of the auxiliary polynomial it solves for.
const MAX_RICCATI_AUX_DEGREE: i64 = 24;

/// The rational solutions of the normal Riccati equation `y′ + y² = a(x)`
/// (`a` rational over ℚ with rational poles), by Algorithm 11 of N. T. Vo,
/// *Rational and Algebraic Solutions of First-Order Algebraic ODEs* (PhD
/// thesis, RISC 2016), after SymPy's `riccati.py`: from the Laurent
/// expansions of `a` at its poles and at infinity, every candidate
/// `ȳ = Σ cᵢⱼ/(x − xᵢ)ʲ + Σ dₖ·xᵏ` with a degree bound `m` for the
/// auxiliary polynomial `P` (`P″ + 2ȳP′ + (ȳ′ + ȳ² − a)·P = 0`), and
/// `y = ȳ + P′/P` for each `P` found.  Only rational choices of the
/// square roots are followed (a solution through an irrational one is not
/// found); poles that are not rational make it give up.  Each solution is
/// checked exactly.  Unlike SymPy's, a polynomial `a` (no finite pole) is
/// handled too (`y′ + y² = x² + 1` has `y = x`).
fn rational_riccati_particulars(a: &RatFn) -> Option<Vec<RatFn>> {
    use crate::poly::Poly;
    use num_traits::{ToPrimitive, Zero};
    let one = Q::one();
    let two = Q::from_integer(2.into());
    // Poles and their orders.
    let mut poles: Vec<(Q, usize)> = Vec::new();
    if !a.den.is_constant() {
        let (_, factors) = a.den.factor_over_z();
        for (f, mult) in factors {
            if f.degree() != Some(1) {
                return None;
            }
            poles.push((-(f.coeff(0) / f.coeff(1)), usize::try_from(mult).ok()?));
        }
    }
    if poles.len() > MAX_RICCATI_POLES {
        return None;
    }
    let val_inf = a.val_inf();
    // Necessary conditions.
    if poles.iter().any(|&(_, m)| m != 1 && !m.is_multiple_of(2))
        || !(val_inf >= 2 || (val_inf <= 0 && val_inf % 2 == 0))
    {
        return Some(Vec::new());
    }
    // c-vectors per pole (entry j: coefficient of 1/(x − xᵢ)^{j+1}).
    let mut c_choices: Vec<Vec<Vec<Q>>> = Vec::with_capacity(poles.len());
    for (p, m) in &poles {
        let mut choices = Vec::new();
        if *m == 1 {
            choices.push(vec![one.clone()]);
        } else if *m == 2 {
            let ser = laurent_at_pole(a, p, 2, 1)?;
            let disc = &one + Q::from_integer(4.into()) * &ser[0];
            if disc.is_zero() {
                choices.push(vec![&one / &two]);
            } else if let Some(s) = rational_sqrt(&disc) {
                choices.push(vec![(&one + &s) / &two]);
                choices.push(vec![(&one - &s) / &two]);
            }
        } else {
            let ri = m / 2;
            // ser[key] = coefficient of (x − p)^{−key}.
            let raw = laurent_at_pole(a, p, *m, *m + 1)?;
            let ser = |key: usize| raw.get(*m - key).cloned().unwrap_or_else(Q::zero);
            let Some(lead) = rational_sqrt(&ser(2 * ri)) else {
                c_choices.push(Vec::new());
                continue;
            };
            for sign in [one.clone(), -one.clone()] {
                let mut c = vec![Q::zero(); ri];
                c[ri - 1] = &sign * &lead;
                if c[ri - 1].is_zero() {
                    continue;
                }
                let mut sm = Q::zero();
                let mut s_last = ri;
                for s in (1..ri).rev() {
                    sm = Q::zero();
                    for j in s + 1..ri {
                        sm += &c[j - 1] * &c[ri + s - j - 1];
                    }
                    if s != 1 {
                        c[s - 1] = (ser(ri + s) - &sm) / (&two * &c[ri - 1]);
                    }
                    s_last = s;
                }
                let ri_q = Q::from_integer(ri.into());
                c[0] = (ser(ri + s_last) - &sm - &ri_q * &c[ri - 1]) / (&two * &c[ri - 1]);
                if !choices.contains(&c) {
                    choices.push(c);
                }
            }
        }
        c_choices.push(choices);
    }
    // d-vectors: (d_0 … d_N, d_{−1}).
    let mut d_choices: Vec<(Vec<Q>, Q)> = Vec::new();
    if val_inf < 0 {
        let n = usize::try_from(-val_inf / 2).ok()?;
        let r = laurent_at_infinity(a, n + 2)?;
        // ser(j) = coefficient of x^j = r_{2N − j}.
        let ser = |j: i64| -> Q {
            let i = 2 * i64::try_from(n).unwrap_or(0) - j;
            usize::try_from(i)
                .ok()
                .and_then(|i| r.get(i).cloned())
                .unwrap_or_else(Q::zero)
        };
        let n_i = i64::try_from(n).ok()?;
        if let Some(lead) = rational_sqrt(&ser(2 * n_i)) {
            for sign in [one.clone(), -one.clone()] {
                let mut d = vec![Q::zero(); n + 1];
                d[n] = &sign * &lead;
                if d[n].is_zero() {
                    continue;
                }
                let mut sm = Q::zero();
                for s in (-1..n_i).rev() {
                    sm = Q::zero();
                    for j in (s + 1)..n_i {
                        let (ju, ku) =
                            (usize::try_from(j).ok()?, usize::try_from(n_i + s - j).ok()?);
                        sm += &d[ju] * &d[ku];
                    }
                    if s >= 0 {
                        let su = usize::try_from(s).ok()?;
                        d[su] = (ser(n_i + s) - &sm) / (&two * &d[n]);
                    }
                }
                let n_q = Q::from_integer(n.into());
                let dm1 = (ser(n_i - 1) - &n_q * &d[n] - &sm) / (&two * &d[n]);
                if !d_choices.iter().any(|(v, w)| *v == d && *w == dm1) {
                    d_choices.push((d, dm1));
                }
            }
        }
    } else if val_inf == 0 {
        let r = laurent_at_infinity(a, 2)?;
        if let Some(d0) = rational_sqrt(&r[0])
            && !d0.is_zero()
        {
            for sign in [one.clone(), -one.clone()] {
                let d0s = &sign * &d0;
                let dm1 = &r[1] / (&two * &d0s);
                d_choices.push((vec![d0s], dm1));
            }
        }
    } else {
        let s_inf = if val_inf == 2 {
            laurent_at_infinity(a, 1)?[0].clone()
        } else {
            Q::zero()
        };
        let disc = &one + Q::from_integer(4.into()) * &s_inf;
        if disc.is_zero() {
            d_choices.push((Vec::new(), &one / &two));
        } else if let Some(s) = rational_sqrt(&disc) {
            d_choices.push((Vec::new(), (&one + &s) / &two));
            d_choices.push((Vec::new(), (&one - &s) / &two));
        }
    }
    // Every combination of one c-vector per pole and one d-vector.
    let mut found: Vec<RatFn> = Vec::new();
    let mut idx = vec![0usize; c_choices.len()];
    if c_choices.iter().any(Vec::is_empty) || d_choices.is_empty() {
        return Some(found);
    }
    loop {
        for (d, dm1) in &d_choices {
            let mut m = dm1.clone();
            for (k, ch) in c_choices.iter().enumerate() {
                m -= &ch[idx[k]][0];
            }
            if m.is_integer() && !m.is_negative() {
                let m = m.to_integer().to_i64()?;
                if m <= MAX_RICCATI_AUX_DEGREE {
                    // ȳ
                    let mut ybar = RatFn::poly(Poly::from_coeffs(d.clone()));
                    for (k, (p, _)) in poles.iter().enumerate() {
                        let lin = Poly::from_coeffs(vec![-p.clone(), one.clone()]);
                        let mut pw = Poly::one();
                        for cij in &c_choices[k][idx[k]] {
                            pw = pw.mul(&lin);
                            let term = RatFn::new(Poly::constant(cij.clone()), pw.clone())?;
                            ybar = ybar.add(&term)?;
                        }
                    }
                    if let Some(y) = riccati_aux_solution(a, &ybar, usize::try_from(m).ok()?)
                        && !found.contains(&y)
                    {
                        found.push(y);
                    }
                }
            }
        }
        // Next combination.
        let mut k = 0;
        loop {
            if k == idx.len() {
                return Some(found);
            }
            idx[k] += 1;
            if idx[k] < c_choices[k].len() {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
}

/// `ȳ + P′/P` for a monic polynomial `P` of degree `m` with
/// `P″ + 2ȳ·P′ + (ȳ′ + ȳ² − a)·P = 0` (undetermined coefficients), when
/// it solves `y′ + y² = a` (checked exactly).
fn riccati_aux_solution(a: &RatFn, ybar: &RatFn, m: usize) -> Option<RatFn> {
    use crate::poly::Poly;
    use num_traits::Zero;
    let a0 = ybar.deriv()?.add(&ybar.mul(ybar)?)?.sub(a)?;
    let a1 = ybar.scale(&Q::from_integer(2.into()));
    // Clear denominators: D·P″ + B₁·P′ + B₀·P = 0.
    let lcm = {
        let g = Poly::gcd(&a0.den, &a1.den);
        a0.den.mul(&a1.den).div(&g)
    };
    let b0 = a0.num.mul(&lcm.div(&a0.den));
    let b1 = a1.num.mul(&lcm.div(&a1.den));
    let apply = |p: &Poly| -> Poly {
        let d1 = p.derivative();
        let d2 = d1.derivative();
        lcm.mul(&d2).add(&b1.mul(&d1)).add(&b0.mul(p))
    };
    let mono = |k: usize| Poly::monomial(Q::one(), k);
    let rhs = apply(&mono(m)).neg();
    let cols: Vec<Poly> = (0..m).map(|k| apply(&mono(k))).collect();
    let rows = cols
        .iter()
        .map(|c| c.degree().unwrap_or(0))
        .chain(std::iter::once(rhs.degree().unwrap_or(0)))
        .max()
        .unwrap_or(0)
        + 1;
    // Gaussian elimination over ℚ on [cols | rhs].
    let mut mat: Vec<Vec<Q>> = (0..rows)
        .map(|r| {
            let mut row: Vec<Q> = cols.iter().map(|c| c.coeff(r)).collect();
            row.push(rhs.coeff(r));
            row
        })
        .collect();
    let mut pivots: Vec<(usize, usize)> = Vec::new();
    let mut row = 0;
    for col in 0..m {
        let Some(pr) = (row..rows).find(|&r| !mat[r][col].is_zero()) else {
            continue;
        };
        mat.swap(row, pr);
        let inv = Q::one() / mat[row][col].clone();
        for v in mat[row].iter_mut() {
            *v *= &inv;
        }
        let pivot_row = mat[row].clone();
        for (r, target) in mat.iter_mut().enumerate() {
            if r != row && !target[col].is_zero() {
                let f = target[col].clone();
                for (t, p) in target.iter_mut().zip(&pivot_row) {
                    *t -= &f * p;
                }
            }
        }
        pivots.push((row, col));
        row += 1;
    }
    if (row..rows).any(|r| !mat[r][m].is_zero()) {
        return None;
    }
    let mut coeffs = vec![Q::zero(); m + 1];
    coeffs[m] = Q::one();
    for (r, c) in pivots {
        coeffs[c] = mat[r][m].clone();
    }
    let p = Poly::from_coeffs(coeffs);
    let y = ybar.add(&RatFn::new(p.derivative(), p)?)?;
    // Exact check: y′ + y² − a = 0.
    y.deriv()?.add(&y.mul(&y)?)?.sub(a)?.is_zero().then_some(y)
}

/// Solve the Riccati equation `y′ = b₀ + b₁·y + b₂·y²` with rational
/// coefficients over ℚ (`b₀, b₂ ≠ 0`) from its rational particular
/// solutions ([`rational_riccati_particulars`] on the normal form
/// `v′ + v² = a(x)`, `v = −b₂·y − b₂′/(2b₂) − b₁/2`): with two particular
/// solutions `v₁, v₂` the general solution is `(v₂·u − v₁)/(u − 1)`,
/// `u = C1·exp(∫(v₂ − v₁))`; with one, `v₁ + 1/z`, `z′ = 2v₁·z + 1`; with
/// `a = 0`, `v = 1/(x + C1)`.  SymPy's `1st_rational_riccati` (BSD; see
/// THIRD-PARTY-NOTICES.md).  `y′ = y² − 2/x²` needed a particular solution
/// passed to [`solve_riccati`]; `dsolve` refused it.
fn try_rational_riccati(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<OdeResult> {
    let rhs = first_order_rhs(arena, expr, func, var)?;
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, rhs, func)?;
    if coeffs.len() != 3 || coeffs.iter().any(|&c| contains_sym(arena, c, func_sym)) {
        return None;
    }
    // Rational functions of `var` alone.
    for &c in &coeffs {
        if crate::base::walk::free_symbols(arena, c)
            .iter()
            .any(|&s| s != var)
        {
            return None;
        }
    }
    let b0 = RatFn::from_expr(arena, coeffs[0], var)?;
    let b1 = RatFn::from_expr(arena, coeffs[1], var)?;
    let b2 = RatFn::from_expr(arena, coeffs[2], var)?;
    if b0.is_zero() || b2.is_zero() {
        return None;
    }
    // a = −b₀b₂ + b₁²/4 − b₁′/2 + 3b₂′²/(4b₂²) + b₁b₂′/(2b₂) − b₂″/(2b₂)
    let q = |n: i64, d: i64| Q::new(n.into(), d.into());
    let b1p = b1.deriv()?;
    let b2p = b2.deriv()?;
    let b2pp = b2p.deriv()?;
    let a = b0
        .mul(&b2)?
        .scale(&q(-1, 1))
        .add(&b1.mul(&b1)?.scale(&q(1, 4)))?
        .sub(&b1p.scale(&q(1, 2)))?
        .add(&b2p.mul(&b2p)?.div(&b2.mul(&b2)?)?.scale(&q(3, 4)))?
        .add(&b1.mul(&b2p)?.div(&b2)?.scale(&q(1, 2)))?
        .sub(&b2pp.div(&b2)?.scale(&q(1, 2)))?;
    let c1 = arena.symbol("C1");
    // The general solution v of the normal form.
    let v = if a.is_zero() {
        let s = arena.add(&[var, c1]);
        let m1 = arena.int(-1);
        arena.pow(s, m1)
    } else {
        let parts = rational_riccati_particulars(&a)?;
        let parts: Vec<ExprId> = parts.iter().map(|p| p.to_expr(arena, var)).collect();
        match parts.as_slice() {
            [] => return None,
            [v1] => {
                // v₁ + 1/z, z = e^{F}·(C1 + ∫ e^{−F}), F = ∫ 2v₁.
                let two = arena.int(2);
                let tv = arena.mul(&[two, *v1]);
                let f = integrate_forms(arena, tv, var);
                if integral_failed(arena, f) {
                    return None;
                }
                let ef = exp_of_log_sum(arena, f);
                let m1 = arena.int(-1);
                let emf = arena.pow(ef, m1);
                let emf = crate::transforms::eval::eval(arena, emf);
                let g = integrate_forms(arena, emf, var);
                if integral_failed(arena, g) {
                    return None;
                }
                let inner = arena.add(&[c1, g]);
                let z = arena.mul(&[ef, inner]);
                let inv_z = arena.pow(z, m1);
                arena.add(&[*v1, inv_z])
            }
            [v1, v2, ..] => {
                let neg_v1 = arena.neg(*v1);
                let diff = arena.add(&[*v2, neg_v1]);
                let diff = crate::transforms::eval::eval(arena, diff);
                let f = integrate_forms(arena, diff, var);
                if integral_failed(arena, f) {
                    return None;
                }
                let ef = exp_of_log_sum(arena, f);
                let u = arena.mul(&[c1, ef]);
                let v2u = arena.mul(&[*v2, u]);
                let num = arena.add(&[v2u, neg_v1]);
                let m1 = arena.int(-1);
                let den = arena.add(&[u, m1]);
                arena.div(num, den)
            }
        }
    };
    // y = −v/b₂ + bp, bp = −b₂′/(2b₂²) − b₁/(2b₂).
    let bp = b2p
        .div(&b2.mul(&b2)?)?
        .scale(&q(-1, 2))
        .sub(&b1.div(&b2)?.scale(&q(1, 2)))?;
    let inv_b2 = RatFn::constant(Q::one()).div(&b2)?.scale(&q(-1, 1));
    let inv_b2 = inv_b2.to_expr(arena, var);
    let bp = bp.to_expr(arena, var);
    let vy = arena.mul(&[inv_b2, v]);
    let sol = arena.add(&[vy, bp]);
    let sol = crate::transforms::eval::eval(arena, sol);
    let sol = crate::simplify::ratsimp::ratsimp(arena, sol);
    let sol = crate::transforms::eval::eval(arena, sol);
    if !crate::base::walk::contains(arena, sol, c1)
        || crate::base::walk::has_unevaluated(arena, sol)
        || !family_verifies(arena, expr, sol, func, var, c1)
    {
        return None;
    }
    Some(OdeResult {
        solution: sol,
        constants: vec![c1],
        branches: Vec::new(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// ODE classification
// ═══════════════════════════════════════════════════════════════════════════

/// ODE classification result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OdeType {
    /// y' = f(x) — simple separable, no y dependence
    SimpleSeparable,
    /// y' = f(x)*g(y) — full separable
    FullSeparable,
    /// y' + a*y = f(x) — first-order linear with constant coefficients
    FirstOrderLinearCC,
    /// y' + P(x)*y = Q(x) — first-order linear with variable coefficients
    FirstOrderLinearVC,
    /// M(x,y) + N(x,y)·y' = 0 with ∂M/∂y = ∂N/∂x — exact first-order
    ExactFirstOrder,
    /// a*y'' + b*y' + c*y = 0 — second-order linear constant-coefficient homogeneous
    SecondOrderLinearCCHomogeneous,
    /// a*y'' + b*y' + c*y = f(x) — second-order linear constant-coefficient nonhomogeneous
    SecondOrderLinearCCNonHomogeneous,
    /// y' + P(x)·y = Q(x)·y^n (n ≠ 0, 1) — Bernoulli equation
    Bernoulli,
    /// a·x²·y'' + b·x·y' + c·y = 0 — Euler-Cauchy equation
    EulerCauchy,
    /// y'' + p·y' + q·y = g(x) solved via variation of parameters
    VariationOfParameters,
    /// y' = f(y/x) — homogeneous coefficient (degree-0 homogeneous RHS)
    HomogeneousCoefficient,
    /// F(y, y', y'') = 0 (no explicit x) — reducible via p = y'
    NthOrderReducible,
    /// Σ a_k·y^(k) = g(x) with constant coefficients, order ≥ 3
    /// (orders 1 and 2 use the dedicated variants above)
    NthOrderLinearConstCoeff,
    /// M(x,y) + N(x,y)·y' = 0, non-exact but with an integrating factor
    /// μ(x) or μ(y)
    IntegratingFactor,
    /// y = x·y' + f(y') — Clairaut equation
    Clairaut,
    /// y' = q₀(x) + q₁(x)·y + q₂(x)·y² — Riccati equation (needs a
    /// particular solution; see [`solve_riccati`])
    Riccati,
    /// Unrecognized ODE type
    Unknown,
}

/// Classify an ODE without solving it.
///
/// The ODE is given as `expr = 0` where `expr` may contain derivative nodes.
/// Returns the recognized [`OdeType`].
pub fn classify_ode(arena: &mut Arena, expr: ExprId, func: ExprId, var: ExprId) -> OdeType {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => return OdeType::Unknown,
    };
    let func_sym = match arena.node(func) {
        ExprNode::Symbol(sid) => *sid,
        _ => return OdeType::Unknown,
    };

    // Check for second-order: look for Derivative(Derivative(func, var), var)
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));
    let d3y_dx3 = arena.intern(ExprNode::Derivative(d2y_dx2, var));

    // Order ≥ 3 with constant coefficients.
    if expr_contains(arena, expr, d3y_dx3) {
        if let Some(ode) = extract_linear_cc(arena, expr, func, var, func_sym)
            && ode.coeffs.len() >= 4
        {
            return OdeType::NthOrderLinearConstCoeff;
        }
        return OdeType::Unknown;
    }

    if expr_contains(arena, expr, d2y_dx2) {
        // Try to verify it matches a*y'' + b*y' + c*y [+ f(x)] = 0 pattern
        if let ExprNode::Add(ref children) = arena.node(expr).clone() {
            let mut all_const_coeff = true;
            let mut has_y2 = false;
            let mut forcing: Vec<ExprId> = Vec::new();
            for &child in children {
                let (_coeff, term) = arena.as_coeff_term(child);
                if term == d2y_dx2 {
                    has_y2 = true;
                } else if term == dy_dx {
                    // OK — y' term with constant coefficient
                } else if term == func {
                    // OK — y term with constant coefficient
                } else if !contains_sym(arena, child, func_sym) {
                    // Term free of y — nonhomogeneous forcing term
                    forcing.push(child);
                } else {
                    all_const_coeff = false;
                    break;
                }
            }
            if has_y2 && all_const_coeff {
                if forcing.is_empty() {
                    return OdeType::SecondOrderLinearCCHomogeneous;
                }
                // Undetermined coefficients apply to poly × exp × {sin, cos}
                // forcing; anything else needs variation of parameters.
                let all_uc = forcing
                    .iter()
                    .all(|&t| parse_forcing_term(arena, t, var).is_some());
                if all_uc {
                    return OdeType::SecondOrderLinearCCNonHomogeneous;
                }
                if try_variation_of_parameters(arena, expr, func, var, func_sym, var_sym).is_some()
                {
                    return OdeType::VariationOfParameters;
                }
                return OdeType::Unknown;
            }
        }
        // Check for Euler-Cauchy: a·x²·y'' + b·x·y' + c·y = 0
        if try_euler_cauchy(arena, expr, func, var, func_sym, var_sym).is_some() {
            return OdeType::EulerCauchy;
        }
        // Check for nth-order reducible: F(y, y', y'') = 0 with no explicit x.
        {
            let d2_ph = arena.symbol("__d2_cls");
            let d1_ph = arena.symbol("__d1_cls");
            let stripped = crate::transforms::subs::subs(arena, expr, d2y_dx2, d2_ph);
            let stripped = crate::transforms::subs::subs(arena, stripped, dy_dx, d1_ph);
            if !contains_sym(arena, stripped, var_sym) {
                return OdeType::NthOrderReducible;
            }
        }

        // Even if we can't fully classify, it has a second derivative
        return OdeType::Unknown;
    }

    // Check for first-order
    if expr_contains(arena, expr, dy_dx) {
        // Check if func appears outside derivative terms
        if !contains_sym_outside_deriv(arena, expr, func_sym, dy_dx) {
            return OdeType::SimpleSeparable;
        }

        // Check if it matches a linear pattern: y' + P(x)*y = Q(x)
        if let ExprNode::Add(ref children) = arena.node(expr).clone() {
            let mut is_linear = true;
            let mut has_var_coeff = false;

            for &child in children {
                let (coeff, term) = arena.as_coeff_term(child);
                if term == dy_dx {
                    // dy/dx term — fine
                } else if term == func {
                    // Constant coefficient on y — fine
                } else if contains_sym(arena, child, func_sym) {
                    // Check if it's of the form P(x)*y (linear in y)
                    if let Some(px) = extract_coeff_of_func(arena, child, func, func_sym, var_sym) {
                        let _ = coeff; // suppress warning
                        if contains_sym(arena, px, var_sym) {
                            has_var_coeff = true;
                        }
                    } else {
                        is_linear = false;
                        break;
                    }
                }
                // Otherwise it's f(x) — acceptable
            }
            if is_linear {
                if has_var_coeff {
                    return OdeType::FirstOrderLinearVC;
                }
                return OdeType::FirstOrderLinearCC;
            }
        }

        // Check for exact ODE: M + N·y' = 0 with ∂M/∂y = ∂N/∂x
        if let Some((m_ex, n_ex)) = extract_m_n(arena, expr, func, var)
            && (contains_sym(arena, m_ex, func_sym) || contains_sym(arena, n_ex, func_sym))
        {
            let dm_dy = crate::transforms::diff::diff(arena, m_ex, func);
            let dn_dx = crate::transforms::diff::diff(arena, n_ex, var);
            let check = arena.sub(dm_dy, dn_dx);
            let check = crate::transforms::eval::eval(arena, check);
            let check = crate::transforms::expand::expand(arena, check);
            let check = crate::transforms::eval::eval(arena, check);
            if check == arena.zero {
                return OdeType::ExactFirstOrder;
            }
        }

        // Check for Clairaut: y = x·y' + f(y')  (y' appears nonlinearly)
        {
            let p = arena.symbol("__clairaut_p_cls");
            if clairaut_f(arena, expr, func, var, func_sym, var_sym, p).is_some() {
                return OdeType::Clairaut;
            }
        }

        // Check for Bernoulli: y' + P(x)·y = Q(x)·y^n (n ≠ 0, 1)
        if let ExprNode::Add(ref bn_children) = arena.node(expr).clone() {
            let mut bn_has_dy = false;
            let mut bn_has_yn = false;
            let mut bn_ok = true;
            let mut bn_has_free = false;
            for &child in bn_children {
                let (_, term) = arena.as_coeff_term(child);
                if term == dy_dx {
                    bn_has_dy = true;
                } else if !contains_sym(arena, child, func_sym) {
                    bn_has_free = true;
                } else if extract_coeff_of_func(arena, child, func, func_sym, var_sym).is_some() {
                    // linear in y — OK
                } else if extract_bernoulli_term(arena, child, func, func_sym, var_sym).is_some() {
                    bn_has_yn = true;
                } else {
                    bn_ok = false;
                    break;
                }
            }
            if bn_has_dy && bn_has_yn && bn_ok && !bn_has_free {
                return OdeType::Bernoulli;
            }
        }

        // Check for homogeneous coefficient: y' = f(y/x)
        // Substitute y = v*x in the RHS; if result is free of x → homogeneous
        if let ExprNode::Add(ref hc_children) = arena.node(expr).clone() {
            let mut hc_has_dy = false;
            let mut hc_other: Vec<ExprId> = Vec::new();
            for &child in hc_children {
                let (_, term) = arena.as_coeff_term(child);
                if term == dy_dx {
                    hc_has_dy = true;
                } else {
                    hc_other.push(child);
                }
            }
            if hc_has_dy && !hc_other.is_empty() {
                let hc_rhs = if hc_other.len() == 1 {
                    arena.neg(hc_other[0])
                } else {
                    let s = arena.add(&hc_other);
                    arena.neg(s)
                };
                // RHS must depend on both x and y
                if contains_sym(arena, hc_rhs, func_sym) && contains_sym(arena, hc_rhs, var_sym) {
                    // Degree-0 homogeneity: f(x, v·x) must be free of x after
                    // cancellation.
                    let v_cls = arena.symbol("__v_cls");
                    let vx = arena.mul(&[v_cls, var]);
                    let sub = crate::transforms::subs::subs(arena, hc_rhs, func, vx);
                    let sub = crate::transforms::eval::eval(arena, sub);
                    let sub = crate::transforms::expand::expand(arena, sub);
                    let sub = crate::transforms::eval::eval(arena, sub);
                    let sub = arena.cancel_expr(sub, var);
                    let sub = crate::transforms::eval::eval(arena, sub);
                    if !contains_sym(arena, sub, var_sym) {
                        return OdeType::HomogeneousCoefficient;
                    }
                }
            }
        }

        // Check if it's a full separable: y' = f(x)*g(y)
        // Quick check: if the RHS (after extracting y') factors cleanly
        if let ExprNode::Add(ref children) = arena.node(expr).clone() {
            let mut has_deriv = false;
            let mut other_terms: Vec<ExprId> = Vec::new();

            for &child in children {
                let (_coeff, term) = arena.as_coeff_term(child);
                if term == dy_dx {
                    has_deriv = true;
                } else {
                    other_terms.push(child);
                }
            }

            if has_deriv && !other_terms.is_empty() {
                let rhs = if other_terms.len() == 1 {
                    arena.neg(other_terms[0])
                } else {
                    let sum = arena.add(&other_terms);
                    arena.neg(sum)
                };
                let factors = collect_mul_factors(arena, rhs);
                let mut can_separate = true;
                for &factor in &factors {
                    let has_x = contains_sym(arena, factor, var_sym);
                    let has_y = contains_sym(arena, factor, func_sym);
                    if has_x && has_y {
                        can_separate = false;
                        break;
                    }
                }
                if can_separate {
                    return OdeType::FullSeparable;
                }
            }
        }

        // Non-exact with an integrating factor μ(x) or μ(y).
        if try_integrating_factor_ode(arena, expr, func, var, func_sym, var_sym).is_some() {
            return OdeType::IntegratingFactor;
        }

        // Riccati: y' = q0 + q1·y + q2·y² with q0, q2 ≠ 0 (checked last: a
        // separable or otherwise directly solvable Riccati keeps the more
        // specific class above).
        if riccati_coeffs(arena, expr, func, var).is_some() {
            return OdeType::Riccati;
        }

        return OdeType::Unknown;
    }

    OdeType::Unknown
}

/// Check if `haystack` contains the sub-expression `needle` anywhere
/// (iteratively: [`walk::contains`](crate::base::walk::contains)).
fn expr_contains(arena: &Arena, haystack: ExprId, needle: ExprId) -> bool {
    crate::base::walk::contains(arena, haystack, needle)
}

/// Check if `expr` contains `sym` in a position that is NOT inside `deriv_node`.
///
/// This detects whether `func` (as a bare symbol) appears outside derivative
/// sub-expressions — i.e., `y` appears outside `dy/dx`.  An explicit stack
/// over the shared DAG, each node visited once (0.28 recursed, and a deep
/// ODE expression could overflow the stack).
fn contains_sym_outside_deriv(
    arena: &Arena,
    expr: ExprId,
    sym: SymbolId,
    deriv_node: ExprId,
) -> bool {
    let mut stack = vec![expr];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        // The derivative node is skipped: don't look inside.
        if id == deriv_node || !seen.insert(id) {
            continue;
        }
        let node = arena.node(id);
        if let ExprNode::Symbol(s) = node {
            if *s == sym {
                return true;
            }
            continue;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// ODE solution verification
// ═══════════════════════════════════════════════════════════════════════════

/// Check whether a solution satisfies an ODE.
///
/// Substitutes the solution for `func`, differentiates as needed,
/// and checks if the ODE expression evaluates to zero: by expansion, the
/// full simplifier, and two normal forms (radical bases factored out at
/// their lowest exponent; a rational function of `sin`, `cos` over a common
/// denominator with `sin² = 1 − cos²`).  Every route is a chain of
/// identities, so `true` is a proof; `false` means no route found zero (not
/// necessarily a wrong solution).  There is no numerical fallback.
///
/// The ODE is given as `ode_expr = 0`.
pub fn checkodesol(
    arena: &mut Arena,
    ode_expr: ExprId,
    solution: ExprId,
    func: ExprId,
    var: ExprId,
) -> bool {
    // Derivative nodes y, y', y'', … up to the highest order present.
    let chain = derivative_chain(arena, func, var);
    let max_order = (1..chain.len())
        .rev()
        .find(|&k| expr_contains(arena, ode_expr, chain[k]))
        .unwrap_or(0);

    // Derivatives of the solution.
    let mut sol_derivs = vec![solution];
    for k in 1..=max_order {
        let d = crate::transforms::diff::diff(arena, sol_derivs[k - 1], var);
        sol_derivs.push(d);
    }

    // Substitute the highest derivative first (more specific), down to func.
    let mut result = ode_expr;
    for k in (0..=max_order).rev() {
        result = crate::transforms::subs::subs(arena, result, chain[k], sol_derivs[k]);
    }

    // Evaluate and simplify
    result = crate::transforms::eval::eval(arena, result);
    result = crate::transforms::expand::expand(arena, result);
    result = crate::transforms::eval::eval(arena, result);

    if result == arena.zero {
        return true;
    }

    // Try another round of expand + eval for stubborn expressions
    result = crate::transforms::expand::expand(arena, result);
    result = crate::transforms::eval::eval(arena, result);
    if result == arena.zero {
        return true;
    }

    // The full simplifier.
    let simplified = crate::simplify::simplify_engine::unified_simplify(
        arena,
        result,
        &crate::simplify::simplify_engine::SimplifyOpts::default(),
    );
    if arena.is_zero_structural(simplified.expr) {
        return true;
    }

    // Two normal forms the simplifier misses; each step is an identity or a
    // multiplication by a non-zero factor, so a zero here is a proof.
    radical_normal_form_is_zero(arena, result) || trig_normal_form_is_zero(arena, result)
}

/// Most terms or radical bases [`radical_normal_form_is_zero`] handles.
const MAX_RADICAL_TERMS: usize = 64;

/// A term of an expanded residual: its other factors, and its radicals as
/// (normalised base, exponent).
type TermRadicals = (Vec<ExprId>, Vec<(ExprId, Q)>);

/// `expand`, `eval`, then combine powers of equal bases (`e^{−2x}·e^{2x} =
/// 1`, which canonical products keep apart).
fn expand_combine(arena: &mut Arena, e: ExprId) -> ExprId {
    let e = crate::transforms::expand::expand(arena, e);
    let e = crate::transforms::eval::eval(arena, e);
    let e = crate::simplify::powsimp::powsimp(arena, e);
    crate::transforms::eval::eval(arena, e)
}

/// Is the residual `r` zero once every radical base `B` (a power `B^q`
/// with a non-integer rational `q`) is factored out at its lowest exponent?
///
/// The expanded residual `Σ tᵢ·Π B_j^{q_ij}` is multiplied by
/// `Π B_j^{−min_i q_ij}`: `B^a·B^b = B^{a+b}` holds for principal powers
/// (`B^a = e^{a·Log B}`, `B ≠ 0`), the remaining exponents are non-negative
/// integers, and the product with a non-zero factor is zero exactly when
/// the residual is.  Before, the correct solution
/// `((C1 + e^{−2x})·e^{2x})^{−1/2}` of the Bernoulli equation `y′ + y = y³`
/// was rejected: the residual `−C1e^{2x}B^{−3/2} + B^{−1/2} − B^{−3/2}`
/// is zero only through `B^{−1/2} = B·B^{−3/2}`.
fn radical_normal_form_is_zero(arena: &mut Arena, r: ExprId) -> bool {
    let r = expand_combine(arena, r);
    let terms: Vec<ExprId> = match arena.node(r) {
        ExprNode::Add(cs) => cs.to_vec(),
        _ => vec![r],
    };
    if terms.len() > MAX_RADICAL_TERMS {
        return false;
    }
    // Per term: the other factors and the radicals (normalised base, exponent).
    let mut split: Vec<TermRadicals> = Vec::with_capacity(terms.len());
    let mut bases: Vec<ExprId> = Vec::new();
    for &t in &terms {
        let factors: Vec<ExprId> = match arena.node(t) {
            ExprNode::Mul(fs) => fs.to_vec(),
            _ => vec![t],
        };
        let mut rest = Vec::new();
        let mut rads = Vec::new();
        for f in factors {
            match arena.node(f).clone() {
                ExprNode::Pow(b, e)
                    if arena.as_num(e).is_some_and(|q| !q.is_integer())
                        && !matches!(arena.node(b), ExprNode::Num(_)) =>
                {
                    let q = arena.as_num(e).cloned().unwrap_or_default();
                    let nb = expand_combine(arena, b);
                    if !bases.contains(&nb) {
                        bases.push(nb);
                    }
                    rads.push((nb, q));
                }
                _ => rest.push(f),
            }
        }
        split.push((rest, rads));
    }
    if bases.is_empty() || bases.len() > MAX_RADICAL_TERMS {
        return false;
    }
    let exponent = |rads: &[(ExprId, Q)], b: ExprId| -> Q {
        rads.iter()
            .filter(|(nb, _)| *nb == b)
            .map(|(_, q)| q.clone())
            .sum()
    };
    let mut minima = Vec::with_capacity(bases.len());
    for &b in &bases {
        let mut m: Option<Q> = None;
        for (_, rads) in &split {
            let e = exponent(rads, b);
            m = Some(match m {
                Some(v) if v <= e => v,
                _ => e,
            });
        }
        minima.push(m.unwrap_or_default());
    }
    let mut new_terms = Vec::with_capacity(split.len());
    for (rest, rads) in &split {
        let mut fs = rest.clone();
        for (&b, m) in bases.iter().zip(&minima) {
            let k = exponent(rads, b) - m;
            if !k.is_integer() {
                return false;
            }
            if num_traits::Zero::is_zero(&k) {
                continue;
            }
            let Some(ki) = num_traits::ToPrimitive::to_i64(&k.to_integer()) else {
                return false;
            };
            if ki > 8 {
                return false;
            }
            let ke = arena.int(ki);
            fs.push(arena.pow(b, ke));
        }
        new_terms.push(arena.mul(&fs));
    }
    let s = arena.add(&new_terms);
    let s = expand_combine(arena, s);
    arena.is_zero_structural(s) || trig_normal_form_is_zero(arena, s)
}

/// Is the residual `r` zero as a rational function (of `sin` and `cos`,
/// among other things)?
///
/// Over a common denominator the numerator must expand to zero — which also
/// covers rational functions of the parameters that the simplifier leaves
/// apart (`eˣ·(a + 1)/(a·(1/a + 1)) − eˣ`, the residual of the particular
/// solution of `y″ + a·y = eˣ`).  For trigonometric residuals:
/// `tan u → sin u/cos u`, the sum over a common denominator, and in the
/// expanded numerator `sin²u → 1 − cos²u` until no square of a sine is
/// left: a polynomial in `sin u`, `cos u` vanishes on the circle exactly
/// when this reduction gives zero.  Before, the variation-of-parameters
/// solution `−cos x·ln|sec x + tan x| + 2 sin x` of `y″ + y = tan x` was
/// rejected: its residual is zero only through `sin² + cos² = 1` over the
/// denominator `(sec x + tan x)²`.
fn trig_normal_form_is_zero(arena: &mut Arena, r: ExprId) -> bool {
    let post = crate::base::walk::post_order_ids(arena, r);
    let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Tan(u) => {
                let s = arena.sin(u);
                let c = arena.cos(u);
                arena.div(s, c)
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }
    let t = cache.get(&r).copied().unwrap_or(r);
    let t = crate::transforms::eval::eval(arena, t);
    let t = crate::poly::polybridge::together(arena, t);
    let (num, _) = crate::poly::polybridge::as_numer_denom(arena, t);
    let mut num = expand_combine(arena, num);
    for _ in 0..6 {
        if arena.is_zero_structural(num) {
            return true;
        }
        let post = crate::base::walk::post_order_ids(arena, num);
        let mut cache: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
        let mut changed = false;
        for &id in &post {
            let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
            let new = match arena.node(rebuilt).clone() {
                ExprNode::Pow(b, e) => {
                    let k = arena.as_num(e).and_then(|q| {
                        q.is_integer()
                            .then(|| num_traits::ToPrimitive::to_i64(&q.to_integer()))
                            .flatten()
                    });
                    match (arena.node(b).clone(), k) {
                        (ExprNode::Sin(u), Some(k)) if (2..=64).contains(&k) => {
                            changed = true;
                            let c = arena.cos(u);
                            let two = arena.int(2);
                            let c2 = arena.pow(c, two);
                            let one = arena.one;
                            let one_minus = arena.sub(one, c2);
                            let half = arena.int(k / 2);
                            let p = arena.pow(one_minus, half);
                            if k % 2 == 1 { arena.mul(&[b, p]) } else { p }
                        }
                        _ => rebuilt,
                    }
                }
                _ => rebuilt,
            };
            cache.insert(id, new);
        }
        if !changed {
            return false;
        }
        let n = cache.get(&num).copied().unwrap_or(num);
        num = expand_combine(arena, n);
    }
    arena.is_zero_structural(num)
}

// ═══════════════════════════════════════════════════════════════════════════
// ODE System Solver (constant-coefficient systems: ẋ = Ax)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a system of first-order constant-coefficient ODEs.
///
/// Given `dx/dt = A·x` where `A` is a constant n×n matrix,
/// returns the general solution `x(t)` as a vector of `n` expressions,
/// each containing arbitrary constants `C1, C2, ..., Cn`.
///
/// # Algorithm
///
/// 1. Verifies `A` is square with no entries depending on `t_var`.
/// 2. For diagonal matrices, returns `[C1·exp(a₁₁·t), C2·exp(a₂₂·t), ...]`.
/// 3. For general matrices, computes eigenvalues and eigenvectors:
///    - Real eigenvalue λ with eigenvector v → `Cₖ·exp(λt)·v`
///    - Complex conjugate pair α±βi → trig form with `cos(βt)`, `sin(βt)`
/// 4. Otherwise (repeated or defective eigenvalues, eigenvalues the
///    eigenvector route cannot split into real and imaginary parts) uses
///    the exact matrix exponential `e^{At}·c` from the Jordan form
///    ([`Matrix::matrix_exp_t`]).
///
/// Before 0.30 step 4 was the Taylor polynomial of `e^{At}` to degree 12
/// ([`Matrix::exp_series`]), returned as if it were the solution: every
/// defective matrix (`[[−1, 1], [0, −1]]`) and several diagonalisable ones
/// got a polynomial that does not satisfy the system.
///
/// # Returns
///
/// `Some(vec)` with the solution vector, or `None` if the matrix is not
/// square, empty, contains entries that depend on `t_var`, or its
/// eigenvalues have no closed form.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::matrix::Matrix;
/// let ctx = Context::new();
/// let t = ctx.symbol("t");
/// let a = Matrix::new(vec![
///     vec![ctx.int(0), ctx.int(1)],
///     vec![ctx.int(-2), ctx.int(-3)],
/// ]).unwrap();
/// let sol = symplex::ode::solve_ode_system(&a, &t).unwrap();
/// assert_eq!(sol.len(), 2);
/// ```
pub fn solve_ode_system(a_matrix: &Matrix, t_var: &Ex) -> Option<Vec<Ex>> {
    solve_ode_system_with_form(a_matrix, t_var).map(|(sol, _)| sol)
}

/// Is the general solution of [`solve_ode_system_with_form`] `e^{At}·c`
/// (`c = (C1, …, Cn)`, so `x(0) = c`) or a sum of eigenmodes?
#[derive(Clone, Copy, PartialEq, Eq)]
enum SystemForm {
    /// `e^{At}·c`: diagonal `A` or the exact matrix exponential.
    ExpTimesConstants,
    /// `Σ Cᵢ·(modeᵢ)` from the eigenvectors.
    Modes,
}

/// [`solve_ode_system`] together with the shape of the solution.
fn solve_ode_system_with_form(a_matrix: &Matrix, t_var: &Ex) -> Option<(Vec<Ex>, SystemForm)> {
    let n = a_matrix.nrows();
    if !a_matrix.is_square() || n == 0 {
        return None;
    }

    // Verify constant coefficients: no entry may depend on t_var
    for i in 0..n {
        for j in 0..n {
            if a_matrix.get(i, j).contains(t_var) {
                return None;
            }
        }
    }

    // Special case: diagonal matrix → exact closed-form per component
    if ode_system_is_diagonal(a_matrix, n) {
        return Some((
            solve_ode_system_diagonal(a_matrix, t_var, n),
            SystemForm::ExpTimesConstants,
        ));
    }

    // Try eigenvalue-based exact solution
    if let Some(sol) = solve_ode_system_eigen(a_matrix, t_var, n) {
        return Some((sol, SystemForm::Modes));
    }

    // Otherwise the exact matrix exponential.
    solve_ode_system_expm(a_matrix, t_var, n).map(|sol| (sol, SystemForm::ExpTimesConstants))
}

/// Solve the initial-value problem `dx/dt = A·x`, `x(0) = x0`.
///
/// Computes the general solution with [`solve_ode_system`], evaluates it
/// at `t = 0`, and determines the constants `C1, …, Cn` from the linear
/// system `x(0) = x0` via [`linsolve`](crate::api::expr_solve_ext::linsolve).
/// When the general solution is `e^{At}·c` (a diagonal `A`, or the exact
/// matrix exponential, e.g. for eigenvalues that are `RootOf`s of an
/// irreducible quartic) the solution is `e^{At}·x0` directly: elimination
/// over the `RootOf` entries did not finish (4×4 matrices with irreducible
/// characteristic polynomials hung).
///
/// # Errors
///
/// - [`SymplexError::InvalidArgument`](crate::base::errors::SymplexError::InvalidArgument) if `A` is not square or `x0` has
///   the wrong length.
/// - [`SymplexError::ComputationFailed`](crate::base::errors::SymplexError::ComputationFailed) if the general solution cannot be
///   found or the constants cannot be determined.
/// - [`SymplexError::NoSolution`](crate::base::errors::SymplexError::NoSolution) if the initial data is contradictory.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// let ctx = Context::new();
/// let t = ctx.symbol("t");
/// // x' = y, y' = -x  with x(0) = 1, y(0) = 0  →  x = cos t, y = -sin t
/// let a = matrix![ctx, [0, 1], [-1, 0]];
/// let sol = symplex::ode::solve_ode_system_ivp(&a, &t, &[ctx.int(1), ctx.int(0)]).unwrap();
/// assert_eq!(format!("{}", sol[0].simplify()), "cos(t)");
/// assert_eq!(format!("{}", sol[1].simplify()), "-sin(t)");
/// ```
pub fn solve_ode_system_ivp(
    a_matrix: &Matrix,
    t_var: &Ex,
    x0: &[Ex],
) -> Result<Vec<Ex>, crate::base::errors::SymplexError> {
    use crate::base::errors::SymplexError;
    let n = a_matrix.nrows();
    if !a_matrix.is_square() || n == 0 {
        return Err(SymplexError::InvalidArgument {
            operation: "solve_ode_system_ivp",
            reason: "coefficient matrix must be square and non-empty".into(),
        });
    }
    if x0.len() != n {
        return Err(SymplexError::InvalidArgument {
            operation: "solve_ode_system_ivp",
            reason: format!("expected {n} initial values, got {}", x0.len()),
        });
    }
    let (general, form) = solve_ode_system_with_form(a_matrix, t_var).ok_or_else(|| {
        SymplexError::ComputationFailed {
            operation: "solve_ode_system_ivp",
            reason: "could not solve the homogeneous system".into(),
        }
    })?;
    let ctx = t_var.context();
    let constants: Vec<Ex> = (1..=n).map(|i| ctx.symbol(&format!("C{i}"))).collect();
    if form == SystemForm::ExpTimesConstants {
        let pairs: Vec<(&Ex, &Ex)> = constants.iter().zip(x0).collect();
        return Ok(general
            .iter()
            .map(|g| {
                let e = g.subs_map(&pairs).eval();
                let rootof = {
                    let inner = e.inner.read();
                    has_rootof(&inner.arena, e.raw_id())
                };
                if rootof { e } else { e.simplify() }
            })
            .collect());
    }
    let zero = ctx.int(0);
    let eqs: Vec<Ex> = general
        .iter()
        .zip(x0)
        .map(|(g, v)| (g.subs(t_var, &zero).eval() - v).eval())
        .collect();
    let sol = crate::api::expr_solve_ext::linsolve(&eqs, &constants).map_err(|e| {
        SymplexError::ComputationFailed {
            operation: "solve_ode_system_ivp",
            reason: format!("could not fit initial values: {e}"),
        }
    })?;
    let pairs = match sol {
        crate::api::expr_solve_ext::LinearSolution::Inconsistent => {
            return Err(SymplexError::NoSolution {
                operation: "solve_ode_system_ivp",
                reason: "initial values are inconsistent with the general solution".into(),
            });
        }
        crate::api::expr_solve_ext::LinearSolution::Unique(p) => p,
        crate::api::expr_solve_ext::LinearSolution::Parametric { solution, free } => solution
            .into_iter()
            .filter(|(c, _)| !free.contains(c))
            .collect(),
    };
    Ok(general
        .iter()
        .map(|g| {
            let mut e = g.clone();
            for (c, v) in &pairs {
                e = e.subs(c, v);
            }
            e.eval().simplify()
        })
        .collect())
}

/// Solve a non-homogeneous system `dx/dt = A·x + b(t)`.
///
/// Computes the general solution as:
///
/// `x(t) = x_h(t) + x_p(t)`
///
/// where `x_h` is the homogeneous solution (from [`solve_ode_system`]) and
/// `x_p` is a particular solution obtained via variation of parameters:
///
/// `x_p = exp(At) · ∫ exp(−At) · b(t) dt`
///
/// The matrix exponentials `e^{±At}` are the exact ones from the Jordan
/// form ([`Matrix::matrix_exp_t`]); an antiderivative the integrator cannot
/// find stays a formal `Integral`.  Before 0.30 they were Taylor
/// polynomials of degree 12 ([`Matrix::exp_series`]) whenever the exact
/// exponential of the symbolic `∓A·t` failed — which was always — so the
/// "particular solution" did not satisfy the system even for constant
/// `b(t)`.
///
/// # Returns
///
/// `None` if the homogeneous part cannot be solved, the matrix
/// exponential has no closed form, or dimensions mismatch.
pub fn solve_ode_system_nonhomogeneous(
    a_matrix: &Matrix,
    b_vec: &[Ex],
    t_var: &Ex,
) -> Option<Vec<Ex>> {
    let n = a_matrix.nrows();
    if !a_matrix.is_square() || n == 0 || b_vec.len() != n {
        return None;
    }

    // Homogeneous part (exact when possible)
    let x_h = solve_ode_system(a_matrix, t_var)?;

    // Particular solution via variation of parameters:
    //   x_p = exp(At) · ∫ exp(-At) · b(t) dt
    let neg_a = a_matrix.scale(&t_var.context().int(-1));
    let exp_neg_at = neg_a.matrix_exp_t(t_var).ok()?;

    let b_col = Matrix::col_vector(b_vec.to_vec()).ok()?;
    let integrand_matrix = exp_neg_at.matmul(&b_col).ok()?.eval();

    // Integrate each component w.r.t. t (in the equivalent forms of
    // [`integrate_forms`]: `e^{−2t}·cos 2t·cos t/t` integrates with the
    // cosines written as exponentials, to `Ei` at complex points).
    let mut integrated = Vec::with_capacity(n);
    for i in 0..n {
        let entry = integrand_matrix.get(i, 0);
        let t_id = entry.checked_id(t_var);
        let id = {
            let mut inner = entry.inner.write();
            integrate_forms(&mut inner.arena, entry.raw_id(), t_id)
        };
        integrated.push(entry.wrap(id).eval());
    }
    let integrated_col = Matrix::col_vector(integrated).ok()?;

    // Multiply by exp(At)
    let exp_at = a_matrix.matrix_exp_t(t_var).ok()?;
    let particular = exp_at.matmul(&integrated_col).ok()?.eval();

    // Combine: x = x_h + x_p
    let mut solution = Vec::with_capacity(n);
    for (i, x_h_i) in x_h.iter().enumerate() {
        let xi: Ex = x_h_i + particular.get(i, 0);
        solution.push(xi.eval());
    }
    Some(solution)
}

/// Returns `true` if every entry of `a_matrix` is free of `t_var`,
/// meaning the system `ẋ = A·x` has constant coefficients.
///
/// Also returns `false` for non-square matrices.
pub fn classify_ode_system_is_constant(a_matrix: &Matrix, t_var: &Ex) -> bool {
    if !a_matrix.is_square() {
        return false;
    }
    let n = a_matrix.nrows();
    for i in 0..n {
        for j in 0..n {
            if a_matrix.get(i, j).contains(t_var) {
                return false;
            }
        }
    }
    true
}

// ── ODE system helpers ─────────────────────────────────────────────────

/// Check whether a matrix is diagonal (off-diagonal entries are structurally zero).
fn ode_system_is_diagonal(m: &Matrix, n: usize) -> bool {
    for i in 0..n {
        for j in 0..n {
            if i != j && !m.get(i, j).is_zero_structural() {
                return false;
            }
        }
    }
    true
}

/// Solve a diagonal system: each row decouples to `x_i' = a_{ii} x_i`.
fn solve_ode_system_diagonal(a_matrix: &Matrix, t_var: &Ex, n: usize) -> Vec<Ex> {
    let ctx = t_var.context();
    (0..n)
        .map(|i| {
            let ci = ctx.symbol(&format!("C{}", i + 1));
            let aii = a_matrix.get(i, i);
            if aii.is_zero_structural() {
                ci // x_i' = 0 → x_i = constant
            } else {
                let exp_term = (aii * t_var).exp();
                &ci * &exp_term
            }
        })
        .collect()
}

/// `e^{At}·c` with the exact matrix exponential ([`Matrix::matrix_exp_t`],
/// from the Jordan form), `c = (C1, …, Cn)`.  `None` when the Jordan form
/// is not available in closed form.
fn solve_ode_system_expm(a_matrix: &Matrix, t_var: &Ex, n: usize) -> Option<Vec<Ex>> {
    let ctx = t_var.context();
    let exp_m = a_matrix.matrix_exp_t(t_var).ok()?;
    let constants: Vec<Ex> = (1..=n).map(|i| ctx.symbol(&format!("C{i}"))).collect();
    let c_vec = Matrix::col_vector(constants).ok()?;
    let result = exp_m.matmul(&c_vec).ok()?;
    Some((0..n).map(|i| result.get(i, 0).eval()).collect())
}

/// Eigenvector-based exact solver for diagonalisable constant-coefficient
/// systems, with the eigenvectors of [`Matrix::eigenvects`] (computed in
/// the eigenvalue's number field):
/// - **Real λ** with eigenvector `v`: the mode `C·exp(λt)·v`;
/// - **Complex λ = α + βi** (`β > 0`; the conjugate gives nothing new) with
///   `v = u + i·w`: the real modes `exp(αt)·(cos(βt)·u − sin(βt)·w)` and
///   `exp(αt)·(sin(βt)·u + cos(βt)·w)`, with `u`, `w` the real and
///   imaginary parts of the entries ([`Ex::as_real_imag`]).
///
/// `None` when `A` is not diagonalisable, a part cannot be separated, or
/// the modes do not number `n` (the caller falls back to the matrix
/// exponential).  Before 0.30 the eigenvectors came from `null(A − λI)`,
/// which is empty for every complex or irrational `λ` (its entries are not
/// recognised as zero), and the real part of an entry was taken by
/// substituting `I → 0`, which is wrong for `1/(1 − I)` (real part `1/2`).
fn solve_ode_system_eigen(a_matrix: &Matrix, t_var: &Ex, n: usize) -> Option<Vec<Ex>> {
    let ctx = t_var.context();
    let i_unit = ctx.i_unit();
    let eigen = a_matrix.eigenvects().ok()?;
    if eigen.iter().map(|(_, _, vs)| vs.len()).sum::<usize>() != n
        || eigen.iter().any(|(_, m, vs)| *m != vs.len())
    {
        return None;
    }
    // Real and imaginary part of a constant, both free of `I`.
    let split = |e: &Ex| -> Option<(Ex, Ex)> {
        let (re, im) = e.as_real_imag();
        let (re, im) = (re.eval().simplify(), im.eval().simplify());
        // Free of `I` and of unevaluated `re`/`im` nodes (up to 0.48 found
        // by printing and searching for "re(" and "im(", which a function
        // name ending in `re` or `im` also matched).
        let clean = |p: &Ex| {
            if p.contains(&i_unit) {
                return false;
            }
            let inner = p.inner.read();
            !crate::base::walk::post_order_ids(&inner.arena, p.raw_id())
                .iter()
                .any(|&id| matches!(inner.arena.node(id), ExprNode::Re(_) | ExprNode::Im(_)))
        };
        (clean(&re) && clean(&im)).then_some((re, im))
    };
    let mut solution: Vec<Ex> = (0..n).map(|_| ctx.int(0)).collect();
    let mut const_idx = 1_usize;
    for (ev, _, vectors) in &eigen {
        let (alpha, beta) = split(ev)?;
        if beta.is_zero_structural() {
            let exp_ev_t = if alpha.is_zero_structural() {
                ctx.int(1)
            } else {
                (&alpha * t_var).exp()
            };
            for v in vectors {
                let ci = ctx.symbol(&format!("C{const_idx}"));
                const_idx += 1;
                for (row, sol_row) in solution.iter_mut().enumerate() {
                    let (vr, vi) = split(v.get(row, 0))?;
                    if !vi.is_zero_structural() {
                        return None;
                    }
                    if !vr.is_zero_structural() {
                        *sol_row = &*sol_row + &(&ci * &(&exp_ev_t * &vr));
                    }
                }
            }
            continue;
        }
        // One of each conjugate pair: β > 0.
        let b = beta.eval_f64().ok()?;
        if b < 0.0 {
            continue;
        }
        let exp_alpha_t = if alpha.is_zero_structural() {
            ctx.int(1)
        } else {
            (&alpha * t_var).exp()
        };
        let cos_bt = (&beta * t_var).cos();
        let sin_bt = (&beta * t_var).sin();
        for v in vectors {
            let c_a = ctx.symbol(&format!("C{const_idx}"));
            let c_b = ctx.symbol(&format!("C{}", const_idx + 1));
            const_idx += 2;
            for (row, sol_row) in solution.iter_mut().enumerate() {
                let (u, w) = split(v.get(row, 0))?;
                let m1 = &exp_alpha_t * &(&(&cos_bt * &u) - &(&sin_bt * &w));
                let m2 = &exp_alpha_t * &(&(&sin_bt * &u) + &(&cos_bt * &w));
                *sol_row = &*sol_row + &(&(&c_a * &m1) + &(&c_b * &m2));
            }
        }
    }
    if const_idx != n + 1 {
        return None;
    }
    Some(solution.into_iter().map(|s| s.eval()).collect())
}

/// The roots of `e = 0` in `var` the solver claims
/// ([`known_real_roots`](crate::transforms::solve::known_real_roots): none
/// when it finds none), as `Solution`s.
fn known_roots(
    arena: &mut Arena,
    e: ExprId,
    var: ExprId,
) -> Vec<crate::transforms::solve::Solution> {
    crate::transforms::solve::known_real_roots(arena, e, var)
        .into_iter()
        .map(|value| crate::transforms::solve::Solution { value })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }
    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn solve_dy_dx_eq_x() {
        // y' = x → y = x²/2 + C1
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let dy = a.intern(ExprNode::Derivative(y, x));
        // dy/dx - x = 0
        let expr = a.sub(dy, x);
        let result = dsolve(&mut a, expr, y, x).expect("should solve");
        let s = display(&a, result.solution);
        assert!(s.contains("C1"), "should have constant: {s}");
        assert!(s.contains("x"), "should contain x: {s}");
    }

    #[test]
    fn solve_dy_dx_eq_0() {
        // y' = 0 → y = C1
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let dy = a.intern(ExprNode::Derivative(y, x));
        let result = dsolve(&mut a, dy, y, x).expect("should solve");
        let s = display(&a, result.solution);
        assert!(s.contains("C1"), "should be constant: {s}");
    }

    #[test]
    fn solve_second_order_y_plus_y_eq_0() {
        // y'' + y = 0 → y = C1*cos(x) + C2*sin(x) (via complex roots ±i)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let dy = a.intern(ExprNode::Derivative(y, x));
        let d2y = a.intern(ExprNode::Derivative(dy, x));
        let expr = a.add(&[d2y, y]);
        let r = dsolve(&mut a, expr, y, x).expect("should solve y'' + y = 0");
        let s = display(&a, r.solution);
        assert!(
            s.contains("C1") && s.contains("C2"),
            "should have two constants: {s}"
        );
        assert!(
            s.contains("cos") && s.contains("sin"),
            "should use trig form (cos and sin): {s}"
        );
    }

    #[test]
    fn solve_y_prime_plus_2y_eq_0() {
        // y' + 2y = 0 → y = C1*e^(-2x)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let dy = a.intern(ExprNode::Derivative(y, x));
        let two = a.int(2);
        let two_y = a.mul(&[two, y]);
        let expr = a.add(&[dy, two_y]);
        let result = dsolve(&mut a, expr, y, x).expect("should solve");
        let s = display(&a, result.solution);
        assert!(s.contains("C1"), "should have constant: {s}");
        assert!(s.contains("exp"), "should contain exp: {s}");
    }

    #[test]
    fn solve_full_separable_xy() {
        // y' - x*y = 0 → y' = x*y → y = C1*exp(x²/2)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let dy = a.intern(ExprNode::Derivative(y, x));
        let xy = a.mul(&[x, y]);
        let expr = a.sub(dy, xy); // y' - x*y = 0
        let result = dsolve(&mut a, expr, y, x).expect("should solve y' = xy");
        let s = display(&a, result.solution);
        assert!(s.contains("C1"), "should have constant: {s}");
        assert!(s.contains("exp"), "should contain exp: {s}");
    }

    #[test]
    fn solve_variable_coeff_linear_2xy() {
        // y' + 2*x*y = 0 → y = C1*exp(-x²)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let dy = a.intern(ExprNode::Derivative(y, x));
        let two = a.int(2);
        let two_x_y = a.mul(&[two, x, y]);
        let expr = a.add(&[dy, two_x_y]); // y' + 2*x*y = 0
        let result = dsolve(&mut a, expr, y, x).expect("should solve y' + 2xy = 0");
        let s = display(&a, result.solution);
        assert!(s.contains("C1"), "should have constant: {s}");
        assert!(s.contains("exp"), "should contain exp: {s}");
    }
}
