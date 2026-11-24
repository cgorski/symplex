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
//! A family through the principal Lambert W, with the constant in its
//! argument, is followed by its twin through `W₋₁` (`y′ = 1/(2x + y + 2)`:
//! `−2x − W_k(−2·C1·e^{−4x−1})/2 − 5/2` for `k = 0, −1`; both real for
//! `−1/e ≤ −2·C1·e^{−4x−1} < 0`), since 0.52.
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
use crate::base::truth::Truth;
use crate::domains::matrix::Matrix;
use num_traits::One;
use num_traits::Signed;

mod bernoulli_euler;
mod clairaut_riccati;
mod classify;
mod const_coeff;
mod linear_exact_homog;
mod reducible;
mod separable;
mod systems;
mod variation;
mod verify;
use bernoulli_euler::*;
pub use clairaut_riccati::*;
pub use classify::*;
pub(crate) use const_coeff::*;
use linear_exact_homog::*;
use reducible::*;
use separable::*;
pub use systems::*;
use variation::*;
pub use verify::*;

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
///
/// An explicit family through the principal Lambert W, with a constant in
/// its argument, is followed by its twin through `W₋₁` (see *Branches* in
/// the module docs).
pub fn dsolve(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId, // y
    var: ExprId,  // x
) -> Option<OdeResult> {
    let result = dsolve_families(arena, expr, func, var)?;
    Some(with_lambert_twins(arena, expr, func, var, result))
}

/// [`dsolve`] without the `W₋₁` twins.
fn dsolve_families(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
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

/// `result` with the twin through `W₋₁` ([`lower_lambert_branch`]) of
/// each explicit family appended after the families, when it verifies
/// ([`family_verifies`]) and at most [`MAX_BRANCHES`] families result.
/// An implicit solution is returned as it is.
fn with_lambert_twins(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    mut result: OdeResult,
) -> OdeResult {
    let ExprNode::Symbol(func_sym) = *arena.node(func) else {
        return result;
    };
    if contains_sym(arena, result.solution, func_sym) {
        return result;
    }
    let mut families: Vec<ExprId> = std::iter::once(result.solution)
        .chain(result.branches.iter().copied())
        .collect();
    let mut twins: Vec<(ExprId, ExprId)> = Vec::new();
    for &f in &families {
        if !contains_sym(arena, f, func_sym)
            && let Some(twin) = lower_lambert_branch(arena, f, &result.constants)
        {
            twins.push(twin);
        }
    }
    if twins.is_empty() {
        return result;
    }
    for (twin, constant) in twins {
        if families.len() < MAX_BRANCHES
            && !families.contains(&twin)
            && family_verifies(arena, expr, twin, func, var, constant)
        {
            families.push(twin);
        }
    }
    result.branches = families.split_off(1);
    result
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
