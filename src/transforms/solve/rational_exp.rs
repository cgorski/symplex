//! Rational equations, exponential and half-angle substitutions, log sums (part of [`solve`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Rational equations, exponential and half-angle substitutions, log sums
// ═══════════════════════════════════════════════════════════════════════════
//
// Each strategy reduces the equation to one it solves already and returns
// candidates; every candidate of a non-polynomial equation is then checked
// against the original by `drop_certain_non_roots` (a certified nonzero or
// undefined residual removes it), as SymPy's `checksol`.

/// Walk `expr` top-down through `Add`, `Mul`, `Neg` and integer powers of
/// subexpressions containing `var`, calling `atom` on every other node that
/// contains `var`; `None` as soon as `atom` rejects one.
pub(super) fn walk_var_atoms(
    arena: &Arena,
    expr: ExprId,
    var: ExprId,
    mut atom: impl FnMut(ExprId) -> bool,
) -> Option<()> {
    let mut stack = vec![expr];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || !expr_contains_var(arena, id, var) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Add(ch) | ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
            ExprNode::Neg(b) => stack.push(*b),
            ExprNode::Pow(b, e)
                if !expr_contains_var(arena, *e, var)
                    && arena.as_num(*e).is_some_and(|r| r.is_integer()) =>
            {
                stack.push(*b)
            }
            _ => {
                if !atom(id) {
                    return None;
                }
            }
        }
    }
    Some(())
}

/// Rational equations `P(x)/Q(x) = 0` (the variable only in sums, products
/// and integer powers, at least one of them negative): the numerator of
/// the combined fraction is solved, and roots of the denominator are
/// dropped.  (`x + 1/x = 3` was refused before 0.31; SymPy:
/// `solve(x + 1/x - 3)` → `[3/2 − √5/2, √5/2 + 3/2]`.)
///
/// When `P` and `Q` have rational coefficients, every factor `P` shares
/// with `Q` is divided out of `P` first (`P/gcd(P, Q)` until the gcd is
/// constant): the roots of what is left are exactly the solutions, and none
/// of them is a pole.  Before 0.39 a root of `P` was tested against `Q` by
/// substitution, and a common root in nested radicals or a `RootOf` was
/// not recognised as a zero of `Q`: `(x⁴ − 2x³ − 3x² + 7x − 2)/(x³ − 3x + 1)`
/// (`= x − 2` off the poles) gave `2` and the three poles.  A numerator
/// left constant proves there is no solution (`(x² − 1)/(x − 1) = 2`;
/// SymPy: `[]`).
pub(super) fn try_solve_rational(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let not_found = || Solutions::refused(Reason::NotFound);
    if !is_rational_function_of(arena, expr, var) {
        return not_found();
    }
    let (num, den) = arena.as_numer_denom_expr(expr);
    let num = crate::transforms::expand::expand(arena, num);
    let num = crate::transforms::eval::eval(arena, num);
    if !expr_contains_var(arena, num, var) {
        // A numerator that is a nonzero constant: no root.  Up to 0.51 not
        // found, and the half-angle form `((1 − t²)² + 2(1 − t²)(1 + t²) +
        // (1 + t²)²)/(1 + t²)²` of `cos(−x)² + 2cos(−x) + 1` (numerator 4)
        // left `solve` partial (`[π]` and an unsolved alternative).
        if arena.is_symbol_free(num)
            && crate::base::truth::Truth::from(crate::poly::algebraic::is_zero_checked(arena, num))
                .is_false()
        {
            return Solutions::none(Reason::said(
                "the equation is a fraction with a nonzero constant numerator",
            ));
        }
        return not_found();
    }
    // The numerator is a polynomial: no negative power of `var` is left.
    let still_rational = crate::base::walk::post_order_ids(arena, num)
        .into_iter()
        .any(|id| {
            matches!(arena.node(id), ExprNode::Pow(b, e)
            if expr_contains_var(arena, *b, var)
                && arena.as_num(*e).is_some_and(|r| r.is_negative()))
        });
    if still_rational {
        return not_found();
    }
    if let (Some(p), Some(q)) = (
        polybridge::expr_to_poly(arena, num, var),
        polybridge::expr_to_poly(arena, den, var),
    ) && !q.is_zero()
        && !p.is_zero()
    {
        let p = without_common_factors(p, &q);
        if p.is_constant() {
            return Solutions::none(Reason::said(
                "every root of the numerator is a pole (a root of the denominator)",
            ));
        }
        return Solutions::found(values(solve_rational_poly(arena, var, &p)));
    }
    // Coefficients with parameters: the common factors divided out over
    // ℚ[x, parameters] (`(x² − a²)/(x − a) = 0` is `x + a = 0`; SymPy
    // `[-a]`, before 0.39 `±√(a²)`, the pole `a` among them).
    let num = match without_common_factors_multi(arena, num, den, var) {
        Some(reduced) if !expr_contains_var(arena, reduced, var) => {
            return Solutions::none(Reason::said(
                "every root of the numerator is a pole (a root of the denominator)",
            ));
        }
        Some(reduced) => reduced,
        None => num,
    };
    // (No root of the numerator, or none found: nothing found here; the
    // next strategies may still answer.)
    let (candidates, missing) = match solve_raw(arena, num, var, cx).into_found() {
        Ok(found) => found,
        Err(nested) if nested.is_nested_periods() => return nested,
        // (the poles among conditional roots: `without_guarded_non_roots`)
        Err(conditional) if conditional.is_conditional() => return conditional,
        Err(_) => return not_found(),
    };
    let mut out = Vec::with_capacity(candidates.len());
    for s in candidates {
        let d = crate::transforms::subs::subs(arena, den, var, s);
        let d = crate::transforms::expand::expand(arena, d);
        let d = crate::transforms::eval::eval(arena, d);
        if arena.is_zero_structural(d) {
            continue;
        }
        // A constant denominator value is tested exactly (minimal
        // polynomial of an algebraic number): `(x² − 2x − 1)/(x − 1 − √2)`
        // has the pole `1 + √2` among the roots of its numerator.
        if crate::base::walk::free_symbols(arena, d).is_empty() {
            match crate::poly::algebraic::is_zero_checked(arena, d) {
                Some(true) => continue,
                Some(false) => {}
                None => {
                    return Solutions::refused(Reason::said(format!(
                        "cannot decide whether the root {} of the numerator is a pole",
                        arena.display(s)
                    )));
                }
            }
        }
        out.push(s);
    }
    Solutions::with_missing(out, missing)
}

/// Polynomials in `var` whose coefficients are polynomials in other
/// symbols, of degree ≥ 3 in `var`: factored over ℤ in all the symbols
/// (`a·x³ − (2a² + a)x² + … = a·(x − a)·(x − a − 1)·(x + 1)`), and the
/// answer is the union of the roots of the factors when every factor
/// containing `var` is solved.  SymPy's `solve` factors the same way;
/// before 0.39 the expanded form was refused ("coefficients not all in
/// ℚ(√p…, i)").  `None` when the polynomial does not split or a factor
/// is not solved.
pub(super) fn try_solve_parametric_factors(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let not_found = || Solutions::refused(Reason::NotFound);
    let Some(coeffs) = symbolic_poly_coeffs(arena, expr, var) else {
        return not_found();
    };
    if coeffs.len() < 4 || coeffs.len() > 9 {
        return not_found();
    }
    let factored = arena.factor_expr(expr, var);
    if factored == expr {
        return not_found();
    }
    let parts: Vec<ExprId> = match arena.node(factored) {
        ExprNode::Mul(ch) => ch.to_vec(),
        _ => vec![factored],
    };
    let mut with_var: Vec<ExprId> = Vec::new();
    for p in parts {
        let base = match *arena.node(p) {
            ExprNode::Pow(b, e)
                if arena
                    .as_num(e)
                    .is_some_and(|r| r.is_integer() && r.is_positive()) =>
            {
                b
            }
            _ => p,
        };
        if expr_contains_var(arena, base, var) && !with_var.contains(&base) {
            with_var.push(base);
        }
    }
    // Every factor solved (roots found; a refused branch makes the whole
    // partial), or nothing.
    let mut out: Option<Solutions> = None;
    for f in with_var {
        if f == expr {
            return not_found();
        }
        let factor = solve_raw(arena, f, var, cx);
        if factor.is_nested_periods() {
            return factor;
        }
        if !factor.has_roots() {
            return not_found();
        }
        out = Some(match out {
            Some(roots) => roots.union(factor),
            None => factor,
        });
    }
    out.unwrap_or_else(not_found)
}

/// Is the product `expr` a quotient of polynomials over ℚ in all its
/// symbols (at most four), so that [`without_common_factors_multi`] applies?
pub(super) fn rational_over_q_in_all_symbols(arena: &Arena, expr: ExprId) -> bool {
    let ExprNode::Mul(children) = arena.node(expr) else {
        return false;
    };
    let mut syms = crate::base::walk::free_symbols(arena, expr);
    syms.sort_by(|&a, &b| arena.canonical_cmp(a, b));
    syms.dedup();
    if syms.len() > 4 {
        return false;
    }
    children.iter().all(|&c| {
        let base = match arena.node(c) {
            ExprNode::Pow(b, e) if arena.as_num(*e).is_some_and(|r| r.is_integer()) => *b,
            _ => c,
        };
        polybridge::expr_to_multipoly(arena, base, &syms).is_some()
    })
}

/// The candidates (roots of factors of the product `expr`) that are not
/// poles: each denominator factor (`b^(−k)` containing `var`) at a
/// candidate free of symbols is tested exactly (`is_zero_checked`: certified
/// digits, then the minimal polynomial of an algebraic number); one that is
/// infinite there (`tan(−x) − 2` at `π/2`) leaves `expr` undefined.  Up to
/// 0.51 that was undecided: `solve(cos(3x)/(tan(−x) − 2))` was refused
/// ("cannot decide whether 2*atan(-1) is a pole").  So is each argument `b`
/// of a Lambert W branch `W_k(b)`, `k ≠ 0` (at any candidate: `W_k(0) =
/// −∞`).  `Err` when a test is undecided.
pub(super) fn drop_poles_of_product(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    candidates: Vec<ExprId>,
) -> Result<Vec<ExprId>, String> {
    if !matches!(arena.node(expr), ExprNode::Mul(_)) {
        return Ok(candidates);
    }
    // Every denominator, also inside a factor: up to 0.46 the roots of
    // `d = x⁵ − x + 1` came back from `d·(1 + 1/d) = 0` (`0·∞` there; SymPy
    // returns the roots of `d + 1` only).
    let mut dens: Vec<ExprId> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Pow(b, e) = *arena.node(id)
            && arena.as_num(e).is_some_and(|r| r.is_negative())
            && expr_contains_var(arena, b, var)
            && !dens.contains(&b)
        {
            dens.push(b);
        }
    }
    let lambert_args = lambert_branch_poles(arena, expr, var);
    if dens.is_empty() && lambert_args.is_empty() {
        return Ok(candidates);
    }
    let mut out = Vec::with_capacity(candidates.len());
    'cand: for s in candidates {
        // A Lambert argument with parameters left is nonzero generically, as
        // a denominator of a rational function with parameters (`x + y` at
        // `x = 0`); only an exact zero makes the candidate a pole.
        for &(b, direct) in &lambert_args {
            let d = crate::transforms::subs::subs(arena, b, var, s);
            let d = crate::transforms::expand::expand(arena, d);
            let d = crate::transforms::eval::eval(arena, d);
            let zero = if arena.is_zero_structural(d) {
                Some(true)
            } else if arena.is_symbol_free(d) {
                crate::poly::algebraic::is_zero_checked(arena, d)
            } else {
                Some(false)
            };
            match zero {
                Some(true) if direct => continue 'cand,
                Some(false) => {}
                _ => {
                    return Err(format!(
                        "cannot decide whether {} is a pole",
                        arena.display(s)
                    ));
                }
            }
        }
        if !crate::base::walk::free_symbols(arena, s).is_empty() {
            out.push(s);
            continue;
        }
        for &b in &dens {
            // `RootOf(q, k)` (of an irreducible `q`) is a zero of the
            // polynomial `b` exactly when `q` divides it.
            if let ExprNode::RootOf(q, qv, _) = *arena.node(s)
                && let (Some(qp), Some(bp)) = (
                    polybridge::expr_to_poly(arena, q, qv),
                    polybridge::expr_to_poly(arena, b, var),
                )
            {
                let g = Poly::gcd(&qp, &bp).degree().unwrap_or(0);
                if g > 0 && Some(g) == qp.degree() {
                    continue 'cand;
                }
                if g == 0 {
                    continue;
                }
            }
            let d = crate::transforms::subs::subs(arena, b, var, s);
            let d = crate::transforms::expand::expand(arena, d);
            let d = crate::transforms::eval::eval(arena, d);
            if arena.is_zero_structural(d) || is_undefined_value(arena, d) {
                continue 'cand;
            }
            match crate::poly::algebraic::is_zero_checked(arena, d) {
                Some(true) => continue 'cand,
                Some(false) => {}
                None => {
                    return Err(format!(
                        "cannot decide whether {} is a pole",
                        arena.display(s)
                    ));
                }
            }
        }
        out.push(s);
    }
    Ok(out)
}

/// The arguments `b` (with `var`) of the Lambert W branches `W_k(b)`, `k ≠
/// 0`, in `expr`, each with whether it is reached through sums, products
/// and positive powers only.  Where `b = 0` the branch is `−∞`, which the
/// canonical product `0·W_k(0) = 0` does not show (`canon_mul` folds
/// `W_k(0)` only later, in `eval`): reached so, the term is infinite there
/// and the point is a pole; inside a function or an exponent the infinity
/// may vanish (`e^{W₁(0)} = 0`) or not, undecided.  Since 0.52, when `W_k`
/// is inverted, `x·W₁(x) = 0` would otherwise have the root 0.
fn lambert_branch_poles(arena: &Arena, expr: ExprId, var: ExprId) -> Vec<(ExprId, bool)> {
    let mut out: Vec<(ExprId, bool)> = Vec::new();
    let mut stack = vec![(expr, true)];
    while let Some((id, direct)) = stack.pop() {
        match arena.node(id) {
            ExprNode::Add(c) | ExprNode::Mul(c) => {
                stack.extend(c.iter().map(|&ch| (ch, direct)));
            }
            ExprNode::Neg(inner) => stack.push((*inner, direct)),
            ExprNode::Pow(b, e) => {
                let positive = arena.as_num(*e).is_some_and(|q| q.is_positive());
                stack.push((*b, direct && positive));
                stack.push((*e, false));
            }
            node => {
                if let ExprNode::Lib(crate::base::libfn::LibFn::LambertW, args) = node
                    && let [b, k] = args[..]
                    && !arena.as_num(k).is_some_and(|q| q.is_zero())
                    && expr_contains_var(arena, b, var)
                {
                    match out.iter_mut().find(|(seen, _)| *seen == b) {
                        Some(entry) => entry.1 |= direct,
                        None => out.push((b, direct)),
                    }
                }
                node.for_each_child(|c| stack.push((c, false)));
            }
        }
    }
    out
}

/// Does `var` occur in `expr` only in sums, products and integer powers,
/// at least one of them negative (a rational function of `var` that is
/// not a polynomial)?
pub(super) fn is_rational_function_of(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    let mut negative = false;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if !expr_contains_var(arena, id, var) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Symbol(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => {}
            ExprNode::Pow(_, e) => match arena.as_num(*e) {
                Some(r) if r.is_integer() => negative |= r.is_negative(),
                _ => return false,
            },
            _ => return false,
        }
    }
    negative
}

/// [`without_common_factors`] for polynomials in `var` and up to three
/// parameters with rational coefficients (multivariate gcd over ℤ).  `None`
/// when they are not such polynomials.
fn without_common_factors_multi(
    arena: &mut Arena,
    num: ExprId,
    den: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    let mut syms = crate::base::walk::free_symbols(arena, num);
    syms.extend(crate::base::walk::free_symbols(arena, den));
    syms.sort_by(|&a, &b| arena.canonical_cmp(a, b));
    syms.dedup();
    if !syms.contains(&var) || syms.len() < 2 || syms.len() > 4 {
        return None;
    }
    let mut n = polybridge::expr_to_multipoly(arena, num, &syms)?;
    let d = polybridge::expr_to_multipoly(arena, den, &syms)?;
    if d.is_zero() || n.is_zero() {
        return None;
    }
    let mut changed = false;
    for _ in 0..64 {
        let g = crate::poly::multipoly::MultiPoly::try_gcd(&n, &d)?;
        if g.total_degree().unwrap_or(0) == 0 {
            break;
        }
        n = n.div_exact(&g)?;
        changed = true;
    }
    changed.then(|| polybridge::multipoly_to_expr(arena, &n, &syms))
}

/// `p` with every irreducible factor it shares with `q` removed (all of
/// its multiplicity): `p / gcd(p, q)` repeated until the gcd is constant.
/// The roots of the result are the roots of `p` that are not roots of `q`.
pub(super) fn without_common_factors(mut p: Poly, q: &Poly) -> Poly {
    if q.is_zero() {
        return p;
    }
    loop {
        let g = Poly::gcd(&p, q);
        if g.degree().unwrap_or(0) == 0 || p.is_zero() {
            return p;
        }
        p = p.div(&g);
    }
}

/// `(a, b)` with `arg = a·var + b`, `a ≠ 0` and `b` free of `var`.
pub(super) fn linear_rate(arena: &mut Arena, arg: ExprId, var: ExprId) -> Option<(ExprId, ExprId)> {
    let coeffs = symbolic_poly_coeffs(arena, arg, var)?;
    if coeffs.len() != 2 || arena.is_zero_structural(coeffs[1]) {
        return None;
    }
    Some((coeffs[1], coeffs[0]))
}

/// `c = r^k` with `r ≥ 2` not a perfect power (`c` a rational > 0 that
/// is an integer or the reciprocal of one; `k` negative for `c < 1`).
pub(super) fn perfect_power_base(c: &Q) -> Option<(BigInt, i64)> {
    if !c.is_positive() || c.is_one() {
        return None;
    }
    let (n, sign) = if c.is_integer() {
        (c.to_integer(), 1i64)
    } else if c.numer().is_one() {
        (c.denom().clone(), -1i64)
    } else {
        return None;
    };
    let bits = n.bits();
    for k in (2..=bits.min(64)).rev() {
        let k32 = u32::try_from(k).ok()?;
        let r = n.nth_root(k32);
        if num_traits::Pow::pow(&r, k32) == n {
            return Some((r, sign * i64::try_from(k).ok()?));
        }
    }
    Some((n, sign))
}

/// Equations in exponentials of one rate: every occurrence of the variable
/// is in `exp(a·x + b)` or `c^(a·x + b)` (`c` a rational base), the rates
/// `a` rational multiples of one constant `a₀` (`i` in `e^(ix) + e^(−ix)`).
/// With `t = exp(g·a₀·x)` (or `r^(g·a₀·x)` for the common base `r` of the
/// `c`, e.g. 2 for 4 and 8) and `g` the gcd of the rational ratios, each
/// exponential is a constant times an integer power of `t`; the equation in
/// `t` is solved (a Laurent polynomial by [`try_solve_rational`]), and `t`
/// is peeled back.  SymPy: `solve(exp(x) + exp(-x) - 3)` →
/// `[log(3/2 − √5/2), log(√5/2 + 3/2)]`, `solve(2**(2*x) - 5*2**x + 6)`
/// → `[1, log(3)/log(2)]`; both were refused before 0.31.
///
/// Any other constant base `c ≠ 0` (negative, a rational such as `2/3`,
/// complex) is its own generator when every exponential has it: for the
/// principal power `e^{u·ln c}`, `c^(n·u)` is the `n`-th power of `c^u` for
/// an integer `n`, and `c^(u + b) = c^b·c^u`.  `(−2)^(2x) − 6·(−2)^x + 8` is `t² − 6t + 8`
/// in `t = (−2)^x`, and `(−2)^x = 4`, `(−2)^x = 2` are solved as on their
/// own (for a real `x` on every branch of the logarithm,
/// [`real_roots_on_every_branch`]: `x = 2`); refused up to 0.58.
///
/// `NoSolution` when the equation in `t` is a polynomial over ℚ (its roots
/// all found) and no root has a real preimage (`eˣ + 3e^(x/2) + 2 = 0`:
/// `t = −1, −2`, `exp` of a real argument is positive), as peeling proves
/// `eˣ = −1` has no real solution; before 0.44 such equations were
/// refused.
pub(super) fn try_exponential_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    exponential_substitution(arena, expr, var, cx)
        .unwrap_or_else(|| Solutions::refused(Reason::NotFound))
}

/// [`try_exponential_substitution`]; `None` when the equation does not
/// have its shape.
fn exponential_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Solutions> {
    let mut atoms: Vec<ExprId> = Vec::new();
    walk_var_atoms(arena, expr, var, |id| {
        let ok = match arena.node(id) {
            ExprNode::Exp(_) => true,
            ExprNode::Pow(b, _) => !expr_contains_var(arena, *b, var),
            _ => false,
        };
        if ok && !atoms.contains(&id) {
            atoms.push(id);
        }
        ok
    })?;
    // (atom, rate in units of a₀ and the base's log, constant factor)
    let mut rated: Vec<(ExprId, Q, ExprId)> = Vec::with_capacity(atoms.len());
    let mut unit: Option<ExprId> = None; // a₀
    /// The base of the generator `t`.
    #[derive(Clone, PartialEq)]
    enum Base {
        /// `e`.
        E,
        /// The integer `r ≥ 2`, not a perfect power, the rational bases are
        /// powers of.
        Root(BigInt),
        /// A constant proved nonzero, the base of every exponential.
        Constant(ExprId),
    }
    let mut base: Option<Base> = None;
    for &id in &atoms {
        let (this_base, arg, scale, cbase) = match arena.node(id).clone() {
            ExprNode::Exp(arg) => (Base::E, arg, Q::one(), None),
            ExprNode::Pow(b, arg) => {
                match arena
                    .as_num(b)
                    .cloned()
                    .and_then(|c| perfect_power_base(&c))
                {
                    Some((r, k)) => (
                        Base::Root(r),
                        arg,
                        Q::from_integer(BigInt::from(k)),
                        Some(b),
                    ),
                    None => {
                        let nonzero = match arena.as_num(b) {
                            Some(c) => !c.is_zero(),
                            None => {
                                crate::base::walk::free_symbols(arena, b).is_empty()
                                    && !value_class::holds_special_atom(arena, b)
                                    && crate::base::truth::Truth::from(
                                        crate::poly::algebraic::is_zero_checked(arena, b),
                                    )
                                    .is_false()
                            }
                        };
                        if !nonzero {
                            return None;
                        }
                        (Base::Constant(b), arg, Q::one(), Some(b))
                    }
                }
            }
            _ => return None,
        };
        match &base {
            None => base = Some(this_base.clone()),
            Some(b) if *b == this_base => {}
            Some(_) => return None,
        }
        let (a, b0) = linear_rate(arena, arg, var)?;
        let a0 = *unit.get_or_insert(a);
        let ratio = arena.div(a, a0);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        let ratio = arena.as_num(ratio)?.clone();
        // exp(a·x + b0) = e^b0·exp(a·x); c^(a·x + b0) = c^b0·r^(k·a·x).
        let factor = match cbase {
            None => arena.exp(b0),
            Some(cb) => arena.pow(cb, b0),
        };
        rated.push((id, ratio * scale, factor));
    }
    let unit = unit?;
    let base = base?;
    if rated.is_empty() {
        return None;
    }
    // g = gcd of the rates (gcd of numerators over lcm of denominators).
    let mut gn = BigInt::zero();
    let mut gd = BigInt::one();
    for (_, r, _) in &rated {
        gn = num_integer::Integer::gcd(&gn, r.numer());
        gd = num_integer::Integer::lcm(&gd, r.denom());
    }
    let g = Q::new(gn, gd);
    if g.is_zero() {
        return None;
    }
    let t = arena.symbol("__t_exp");
    let mut sub = expr;
    for (id, r, factor) in &rated {
        let n = (r / &g).to_integer();
        if n.abs() > BigInt::from(MAX_SUBSTITUTION_DEGREE) {
            return None;
        }
        let n_id = arena.big_int(n);
        let tn = arena.pow(t, n_id);
        let repl = arena.mul(&[*factor, tn]);
        sub = arena.subs_structural(sub, *id, repl);
    }
    let sub = crate::transforms::eval::eval(arena, sub);
    if expr_contains_var(arena, sub, var) {
        return None;
    }
    let g_id = arena.num_ratio(g);
    let gx = arena.mul(&[g_id, unit, var]);
    let generator = match base {
        Base::E => arena.exp(gx),
        Base::Root(r) => {
            let r_id = arena.big_int(r);
            arena.pow(r_id, gx)
        }
        Base::Constant(c) => arena.pow(c, gx),
    };
    // A rational equation in `t` with a constant numerator has no root at
    // all: neither has the equation in `x` (`tanh(x − 3)·tanh(x + 3) = 1`;
    // refused before 0.47, SymPy: `[]`).
    {
        let num = full_numerator(arena, sub);
        if polybridge::expr_to_poly(arena, num, t).is_some_and(|p| p.is_constant() && !p.is_zero())
        {
            return Some(Solutions::none(Reason::said(
                "the equation in the exponential has no root",
            )));
        }
    }
    let (t_roots, t_missing) = solutions_unless_identity(arena, sub, t, cx)?
        .without_conditions()
        .into_found()
        .ok()?;
    let back = back_substitute(arena, generator, t_roots, var, cx).lacking(t_missing);
    if back.is_none_proved() {
        let (num, _) = polybridge::fraction_parts(arena, sub);
        let num = crate::transforms::expand::expand(arena, num);
        let num = crate::transforms::eval::eval(arena, num);
        if polybridge::expr_to_poly(arena, num, t).is_some() {
            return Some(Solutions::none(Reason::said(
                "no root of the equation in the exponential has a real preimage (exp of a real argument is positive)",
            )));
        }
        return None;
    }
    (back.has_roots() || back.is_nested_periods()).then_some(back)
}

/// Sums `Σ cᵢ·e^(Sᵢ(x))` of exponentials (`e^L`, `b^L` with constant
/// positive bases `b`, products of them; `S = Σ L·ln b`) whose exponents
/// fall into two groups up to constants: `C₁·e^(S₁) + C₂·e^(S₂) = 0` is
/// `S₁ − S₂ = ln(−C₂/C₁)`, an equation without exponentials (SymPy's
/// `_solve_exponential`).  For a real `S₁ − S₂`: no solution when
/// `−C₂/C₁` is certainly not positive.  `5^(x/2) = 2^(x/3)` gives `x = 0`,
/// `4^(x+1) + 4^(x−1) + 4^(x+2) = 3^(x+2) + 3^(x+3)` one root, `5^(x/2) =
/// 2^(3/x)` the two `±√(6·ln 2/ln 5)`; all refused up to 0.46 (bases
/// with no common power).  Otherwise a general solve takes every branch
/// `ln(−C₂/C₁) + 2nπi` (`2ˣ + 3ˣ = 0` for a plain `x`: `(2n + 1)πi/ln(2/3)`)
/// and a principal solve the principal one.  Up to 0.49 a real unknown was
/// taken to make `S₁ − S₂` real (`eˣ + e^(x + ix) = 0` for a real `x` had
/// "no solution", but `x = π` solves it) and a general solve over ℂ did
/// not split.  Candidates are checked by the caller.
pub(super) fn try_exponential_log_split(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    exponential_log_split(arena, expr, var, cx)
        .unwrap_or_else(|| Solutions::refused(Reason::NotFound))
}

/// [`try_exponential_log_split`]; `None` when the equation does not have
/// its shape.
fn exponential_log_split(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Solutions> {
    let ExprNode::Add(terms) = arena.node(expr).clone() else {
        return None;
    };
    // (coefficient, exponent S) per term
    let mut parts: Vec<(ExprId, ExprId)> = Vec::with_capacity(terms.len());
    for t in terms.iter().copied() {
        let factors: Vec<ExprId> = match arena.node(t) {
            ExprNode::Mul(ch) => ch.to_vec(),
            _ => vec![t],
        };
        let mut coeff: Vec<ExprId> = Vec::new();
        let mut logs: Vec<ExprId> = Vec::new();
        for f in factors {
            if !expr_contains_var(arena, f, var) {
                coeff.push(f);
                continue;
            }
            let node = arena.node(f).clone();
            match node {
                ExprNode::Exp(l) => logs.push(l),
                ExprNode::Pow(b, l)
                    if !expr_contains_var(arena, b, var)
                        && crate::base::walk::free_symbols(arena, b).is_empty()
                        && crate::poly::algebraic::sign_checked(arena, b) == Some(1) =>
                {
                    let ln_b = arena.ln(b);
                    logs.push(arena.mul(&[l, ln_b]));
                }
                _ => return None,
            }
        }
        if logs.is_empty() {
            return None;
        }
        let c = arena.mul(&coeff);
        let s = arena.add(&logs);
        let s = crate::transforms::expand::expand(arena, s);
        let s = crate::transforms::eval::eval(arena, s);
        parts.push((c, s));
    }
    // Group the exponents equal up to a constant: (S, Σ cᵢ·e^(Sᵢ − S)).
    let mut groups: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
    for (c, s) in parts {
        let mut placed = false;
        for (s0, coeffs) in groups.iter_mut() {
            let d = arena.sub(s, *s0);
            let d = crate::transforms::expand::expand(arena, d);
            let d = crate::transforms::eval::eval(arena, d);
            if !expr_contains_var(arena, d, var) {
                let e = arena.exp(d);
                coeffs.push(arena.mul(&[c, e]));
                placed = true;
                break;
            }
        }
        if !placed {
            groups.push((s, vec![c]));
        }
    }
    let [(s1, c1), (s2, c2)] = &groups[..] else {
        return None;
    };
    let c1 = arena.add(c1);
    let c1 = crate::transforms::eval::eval(arena, c1);
    let c2 = arena.add(c2);
    let c2 = crate::transforms::eval::eval(arena, c2);
    if arena.is_zero_structural(c1) || arena.is_zero_structural(c2) {
        return None;
    }
    let ratio = arena.div(c2, c1);
    let ratio = arena.neg(ratio);
    let ratio = crate::transforms::eval::eval(arena, ratio);
    let d = arena.sub(*s1, *s2);
    // `e^(S₁ − S₂) = ratio ≤ 0`: no solution for a real `S₁ − S₂`;
    // otherwise `S₁ − S₂ = ln(ratio)`, all its branches in a general solve
    // (as the arm of `exp` in `solve_by_peeling`).
    let sign = if crate::base::walk::free_symbols(arena, ratio).is_empty() {
        crate::poly::algebraic::sign_checked(arena, ratio)
    } else {
        None
    };
    let real_d = cx.range_restricted(arena, d, var);
    if sign == Some(0) || (sign == Some(-1) && real_d) {
        return Some(Solutions::none(Reason::said(
            "the exponentials of a real argument are positive and the coefficients do not allow it",
        )));
    }
    let ln_ratio = principal_ln(arena, ratio);
    let branch = match cx.period() {
        Some(n) if !real_d => {
            if period_in_use(arena, ratio, cx) {
                return Some(Solutions::refused(Reason::NestedPeriods));
            }
            let p = two_pi_i_n(arena, n);
            arena.add(&[ln_ratio, p])
        }
        _ => ln_ratio,
    };
    let eq = arena.sub(d, branch);
    let eq = crate::transforms::expand::expand(arena, eq);
    let eq = crate::transforms::eval::eval(arena, eq);
    if !expr_contains_var(arena, eq, var) || has_exponential_of(arena, eq, var) {
        return None;
    }
    Some(solve_raw(arena, eq, var, cx))
}

/// Most absolute values split by [`try_abs_cases`] (`2ⁿ` sign patterns).
const MAX_ABS_CASES: usize = 4;

/// What an equation with absolute values says of a complex unknown
/// ([`abs_unknown_shape`]).
enum AbsUnknown {
    /// Every solution is real: the numerator is `A·x + B(|uᵢ|)` with a
    /// nonzero real constant `A` and `B` real for real moduli.
    Real,
    /// The unknown occurs only in the one modulus given (`1 + C(|x|, 4)`).
    ModulusOnly(ExprId),
    /// Neither: `|x| = 1` is a circle.
    Other,
}

/// The shape of `expr` in the absolute values `|uᵢ|` of `var` (each `uᵢ`
/// rational in `var`) for a complex `var` ([`AbsUnknown`]): `|x − π| + x
/// = 0` forces `x = −|x − π|` real, as do `|x|/x + 3 = 0` and `|x| + c −
/// x = 0`; up to 0.48 these were split as for a real `x` because a plain
/// `x` was taken real, and in 0.49 refused.
fn abs_unknown_shape(arena: &mut Arena, expr: ExprId, var: ExprId) -> AbsUnknown {
    use crate::base::assumptions::{AssumptionCache, Assumptions, Props};
    let mut atoms: Vec<ExprId> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Abs(u) = *arena.node(id)
            && expr_contains_var(arena, u, var)
        {
            if !is_rational_in(arena, u, var) {
                return AbsUnknown::Other;
            }
            if !atoms.contains(&id) {
                atoms.push(id);
            }
        }
    }
    if atoms.is_empty() || atoms.len() > MAX_ABS_CASES {
        return AbsUnknown::Other;
    }
    // each modulus a real stand-in `sᵢ`
    let mut replaced = expr;
    for (i, &a) in atoms.iter().enumerate() {
        let s =
            crate::transforms::subs::fresh_symbol(arena, &format!("__abs_modulus_{i}"), &[expr]);
        if let ExprNode::Symbol(sid) = *arena.node(s) {
            let mut real = Assumptions::default();
            real.assert_true(Props::REAL);
            arena.set_symbol_assumptions(sid, real);
        }
        replaced = arena.subs_structural(replaced, a, s);
    }
    if !expr_contains_var(arena, replaced, var) {
        return match atoms[..] {
            [atom] => AbsUnknown::ModulusOnly(atom),
            _ => AbsUnknown::Other,
        };
    }
    let num = full_numerator(arena, replaced);
    let Some(coeffs) = symbolic_poly_coeffs(arena, num, var) else {
        return AbsUnknown::Other;
    };
    let [b, a] = coeffs[..] else {
        return AbsUnknown::Other;
    };
    let a_real_nonzero = crate::base::walk::free_symbols(arena, a).is_empty()
        && matches!(crate::poly::algebraic::sign_checked(arena, a), Some(s) if s != 0);
    if a_real_nonzero
        && !expr_contains_var(arena, b, var)
        && AssumptionCache::new().query(arena, b, Props::REAL) == Some(true)
    {
        AbsUnknown::Real
    } else {
        AbsUnknown::Other
    }
}

/// The roots of `expr = 0` in a complex `var` that occurs only in the
/// modulus `atom = |u|` (real and nonnegative): no root when no real
/// `s ≥ 0` solves `expr(|u| = s) = 0` (`1 + C(|x|, 4)`: `C(s, 4) ≥ −1/24`),
/// the roots of `u = 0` when `s = 0` is the only one; `None` when a
/// positive `s` solves it (`|u| = s` is a curve: `tan |x| = 0` holds on
/// the circles `|x| = nπ`) or nothing is decided.
fn modulus_only_roots(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    atom: ExprId,
    cx: &SolveCx,
) -> Option<Solutions> {
    use crate::base::assumptions::{Assumptions, Props};
    let ExprNode::Abs(u) = *arena.node(atom) else {
        return None;
    };
    let s = crate::transforms::subs::fresh_symbol(arena, "__abs_modulus", &[expr]);
    if let ExprNode::Symbol(sid) = *arena.node(s) {
        let mut real = Assumptions::default();
        real.assert_true(Props::REAL);
        arena.set_symbol_assumptions(sid, real);
    }
    let in_s = arena.subs_structural(expr, atom, s);
    // every real root of the equation in `s` (a general solve: the
    // principal roots of `tan s = 0` are `[0]`, its roots `nπ`)
    let n = branch_parameter(arena, &[expr, var]);
    let points = match solve_impl(arena, in_s, s, &SolveCx::general(Domain::Real, n)) {
        Solutions::Complete(RootSet::Empty(_)) => Vec::new(),
        complete @ Solutions::Complete(RootSet::Points(_) | RootSet::Families { .. }) => {
            complete.claimed_points()
        }
        _ => return None,
    };
    // a family of `s` has positive members (`|x| = π`: a circle)
    if points
        .iter()
        .any(|&p| crate::base::walk::contains(arena, p, n))
    {
        return None;
    }
    let mut zero = false;
    for p in points {
        if crate::transforms::realness::constant_realness_as_declared(
            arena,
            p,
            ROOT_CHECK_DIGITS,
            &mut crate::base::assumptions::AssumptionCache::new(),
        ) == Some(false)
        {
            continue;
        }
        match crate::poly::algebraic::sign_checked(arena, p) {
            Some(s) if s < 0 => {}
            Some(0) => zero = true,
            _ => return None,
        }
    }
    if zero {
        return Some(solve_raw(arena, u, var, cx));
    }
    Some(Solutions::none(Reason::said(
        "no nonnegative real value of the absolute value solves the equation",
    )))
}

/// An equation with absolute values `|uᵢ|` of rational functions `uᵢ` (with
/// rational coefficients) of a variable taken real: for each choice of
/// signs `|uᵢ| = ±uᵢ` the equation without them (a rational one, all its
/// roots found), and the real candidates of all the choices — the caller
/// checks every candidate against the equation, which drops those of a
/// choice the signs at them contradict.  Every real root is found by the
/// choice of the signs of the `uᵢ` at it.  `|x + 3| = 2·|x − 3|` gives `1, 9`,
/// `|x + 1| + |x² − 7| = 9` gives `(√61 − 1)/2, (1 − √69)/2`, refused
/// before 0.47 (SymPy gives the same for a real `x`).  `None` for another shape, a
/// variable that may be non-real, or more than [`MAX_ABS_CASES`] absolute
/// values; `NoSolution` when no choice has a real root.
pub(super) fn try_abs_cases(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    abs_cases(arena, expr, var, cx).unwrap_or_else(|| Solutions::refused(Reason::NotFound))
}

/// [`try_abs_cases`]; `None` when the equation does not have its shape, or
/// a choice of signs is refused or vanishes identically.
fn abs_cases(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Option<Solutions> {
    if !crate::base::walk::free_symbols(arena, expr)
        .iter()
        .all(|&s| s == var)
    {
        return None;
    }
    // (Up to 0.48 a plain `x` was taken real under `Ex::solve`: `|x + 3| =
    // 2·|x − 3|` gave `[9, 1]`, while the Apollonius circle solves it.  A
    // complex unknown is split only when the equation makes it real.)
    let real_cx;
    let cx = if cx.unknown_may_be_nonreal(arena, var) {
        match abs_unknown_shape(arena, expr, var) {
            AbsUnknown::Real => {
                real_cx = cx.with_real_unknown();
                &real_cx
            }
            AbsUnknown::ModulusOnly(atom) => return modulus_only_roots(arena, expr, var, atom, cx),
            AbsUnknown::Other => return None,
        }
    } else {
        cx
    };
    let mut abs_atoms: Vec<ExprId> = Vec::new();
    for id in crate::base::walk::post_order_ids(arena, expr) {
        if let ExprNode::Abs(u) = *arena.node(id)
            && expr_contains_var(arena, u, var)
        {
            // `|u| = ±u` covers every case only for a real `u` (an imaginary
            // variable has `|i·t| = |t|`).
            if !is_rational_in(arena, u, var)
                || !cx.real(arena, u, var)
                || crate::base::walk::post_order_ids(arena, u)
                    .into_iter()
                    .any(|j| matches!(arena.node(j), ExprNode::Abs(_)))
            {
                return None;
            }
            if !abs_atoms.contains(&id) {
                abs_atoms.push(id);
            }
        }
    }
    if abs_atoms.is_empty() || abs_atoms.len() > MAX_ABS_CASES {
        return None;
    }
    let mut out: Vec<ExprId> = Vec::new();
    for mask in 0..(1usize << abs_atoms.len()) {
        let mut branch = expr;
        for (i, &a) in abs_atoms.iter().enumerate() {
            let ExprNode::Abs(u) = *arena.node(a) else {
                return None;
            };
            let signed = if mask & (1 << i) != 0 {
                arena.neg(u)
            } else {
                u
            };
            branch = arena.subs_structural(branch, a, signed);
        }
        let branch = crate::transforms::eval::eval(arena, branch);
        if !is_rational_in(arena, branch, var) {
            return None;
        }
        // Each choice is a rational equation with rational coefficients,
        // whose roots are all found: one refused (or partial) answers
        // nothing; `0 = 0` on a whole choice is an interval of solutions.
        match solve_raw(arena, branch, var, cx).into_found() {
            Ok((roots, None)) => {
                for v in roots {
                    let v = crate::transforms::eval::eval(arena, v);
                    if crate::transforms::realness::constant_realness_as_declared(
                        arena,
                        v,
                        ROOT_CHECK_DIGITS,
                        &mut crate::base::assumptions::AssumptionCache::new(),
                    ) == Some(false)
                    {
                        continue;
                    }
                    if !out.contains(&v) {
                        out.push(v);
                    }
                }
            }
            Err(Solutions::Complete(RootSet::Empty(_))) => {}
            Err(nested) if nested.is_nested_periods() => return Some(nested),
            Ok((_, Some(_))) | Err(_) => return None,
        }
    }
    Some(Solutions::exactly(
        out,
        Reason::said("no choice of the signs of the absolute values has a real root"),
    ))
}

/// Is `expr` a polynomial or a rational function of `var` (integer powers
/// only), with any coefficients free of it?
pub(super) fn is_rational_in(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, expr)
        .into_iter()
        .filter(|&id| expr_contains_var(arena, id, var))
        .all(|id| match arena.node(id) {
            ExprNode::Symbol(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => true,
            ExprNode::Pow(_, e) => arena.as_num(*e).is_some_and(|r| r.is_integer()),
            _ => false,
        })
}

/// Largest `k` of a `C(f, k)` written as a polynomial.
const MAX_BINOMIAL_POLY: i64 = 32;

/// `expr` with each `C(f, k)` (`f` involving `var`, `k` an integer in
/// `0..=MAX_BINOMIAL_POLY`) replaced by `f(f − 1)⋯(f − k + 1)/k!`; `None`
/// when there is none.
pub(super) fn binomials_as_polynomials(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    let mut out = expr;
    let mut any = false;
    for id in crate::base::walk::post_order_ids(arena, expr) {
        let ExprNode::Binomial(f, k) = *arena.node(id) else {
            continue;
        };
        if !expr_contains_var(arena, f, var) {
            continue;
        }
        let Some(kk) = arena
            .as_num(k)
            .filter(|q| q.is_integer())
            .map(|q| q.to_integer())
        else {
            continue;
        };
        let Some(kk) = num_traits::ToPrimitive::to_i64(&kk) else {
            continue;
        };
        let Some(len) = crate::base::cast::bounded_len(kk, MAX_BINOMIAL_POLY) else {
            continue;
        };
        let mut factors: Vec<ExprId> = Vec::with_capacity(len + 1);
        let mut fact = BigInt::one();
        for j in 0..kk {
            let jj = arena.int(j);
            factors.push(arena.sub(f, jj));
            fact *= BigInt::from(j + 1);
        }
        let inv = arena.num_ratio(Q::new(BigInt::one(), fact));
        factors.push(inv);
        let poly = arena.mul(&factors);
        out = arena.subs_structural(out, id, poly);
        any = true;
    }
    if !any {
        return None;
    }
    let out = crate::transforms::eval::eval(arena, out);
    Some(out)
}

/// Does `e` hold an exponential (`e^u`, `b^u`) whose exponent involves `var`?
fn has_exponential_of(arena: &Arena, e: ExprId, var: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .any(|id| match *arena.node(id) {
            ExprNode::Exp(u) => expr_contains_var(arena, u, var),
            ExprNode::Pow(b, u) => {
                !expr_contains_var(arena, b, var) && expr_contains_var(arena, u, var)
            }
            _ => false,
        })
}

/// Equations in `sinh`, `cosh`, `tanh` (and exponentials) of the variable:
/// rewritten in exponentials (`sinh u = (eᵘ − e⁻ᵘ)/2`, …) and solved by
/// [`try_exponential_substitution`]; the candidates are checked against the
/// original equation.  SymPy: `solve(sinh(x) + cosh(x) - 3)` → `[log(3)]`,
/// `solve(sinh(2*x) - sinh(x))` → `[0, -I*pi/3, I*pi/3, I*pi]`; both were
/// refused before 0.44 (the single inversions `sinh x = 2`, `cosh x = 2`
/// were solved).
pub(super) fn try_hyperbolic_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let mut hyperbolic = false;
    let shape = walk_var_atoms(arena, expr, var, |id| match arena.node(id) {
        ExprNode::Sinh(_) | ExprNode::Cosh(_) | ExprNode::Tanh(_) => {
            hyperbolic = true;
            true
        }
        ExprNode::Exp(_) => true,
        ExprNode::Pow(b, _) => !expr_contains_var(arena, *b, var),
        _ => false,
    });
    if shape.is_none() || !hyperbolic {
        return Solutions::refused(Reason::NotFound);
    }
    let rewritten = arena.rewrite_as_exp_expr(expr);
    let rewritten = crate::transforms::eval::eval(arena, rewritten);
    if rewritten == expr {
        return Solutions::refused(Reason::NotFound);
    }
    try_exponential_substitution(arena, rewritten, var, cx)
}

/// The numerator of `expr` over a common denominator when that
/// denominator is a polynomial in `var` and the numerator is not
/// (`ln(x)/x − 1/4` → `4·ln x − x`): every root of `expr` is one of the
/// numerator, and the callers check the numerator's roots against `expr`.
pub(super) fn numerator_over_polynomial(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    if !matches!(arena.node(expr), ExprNode::Add(_) | ExprNode::Mul(_)) {
        return None;
    }
    let combined = polybridge::together(arena, expr);
    let (num, den) = polybridge::as_numer_denom(arena, combined);
    if !expr_contains_var(arena, den, var) || symbolic_poly_coeffs(arena, den, var).is_none() {
        return None;
    }
    let num = crate::transforms::expand::expand(arena, num);
    let num = crate::transforms::eval::eval(arena, num);
    if num == expr
        || !expr_contains_var(arena, num, var)
        || polybridge::expr_to_poly(arena, num, var).is_some()
    {
        return None;
    }
    Some(num)
}

/// The solutions of the substituted equation `sub = 0` in `t`, or `None`
/// when it is an identity: the original equation then holds wherever the
/// substitution is defined, which no finite list (or family) of points
/// describes.  Before 0.39 `sin(x)/cos(x) − tan(x) = 0` (`0 = 0` in
/// `t = tan(x/2)`) gave only `x = π` (the value `t = ∞` checked apart),
/// and `solve_general` the family `π + 2nπ`.  Also `None` when the roots
/// of its numerator cannot all be found ([`numerator_roots`]).
fn solutions_unless_identity(
    arena: &mut Arena,
    sub: ExprId,
    t: ExprId,
    cx: &SolveCx,
) -> Option<Solutions> {
    let combined = polybridge::together(arena, sub);
    let (num, _) = polybridge::as_numer_denom(arena, combined);
    let num = crate::transforms::expand::expand(arena, num);
    let num = crate::transforms::eval::eval(arena, num);
    if arena.is_zero_structural(num) {
        return None;
    }
    // (`together` leaves nested quotients: the whole combined fraction)
    let full_num = full_numerator(arena, sub);
    match numerator_roots(arena, full_num, t, cx) {
        NumeratorRoots::Solver => {}
        NumeratorRoots::Found(roots) => return Some(Solutions::found(roots)),
        NumeratorRoots::Unknown => return None,
    }
    match solve_impl(arena, sub, t, &cx.substituted()) {
        Solutions::Complete(RootSet::All) => None,
        other => Some(other),
    }
}

/// The numerator of `sub` over one denominator (nested quotients
/// combined), expanded.
fn full_numerator(arena: &mut Arena, sub: ExprId) -> ExprId {
    let (full_num, _) = polybridge::fraction_parts(arena, sub);
    let full_num = crate::transforms::expand::expand(arena, full_num);
    crate::transforms::eval::eval(arena, full_num)
}

/// How the roots of the numerator of a substituted equation in `t` are
/// found ([`numerator_roots`]).
enum NumeratorRoots {
    /// The solver finds them all: solve the substituted equation.
    Solver,
    /// Every root of the numerator, found over a number field; candidates
    /// that are poles of the substituted equation are left to the check of
    /// the original equation.
    Found(Vec<ExprId>),
    /// Not all found: the substitution must not answer.
    Unknown,
}

/// Does the solver find every root of the numerator `num` of a substituted
/// equation in `t`?  Over ℚ always (exact factorisation; `RootOf` beyond
/// degree 4), and for degree ≤ 2; otherwise (coefficients with radicals or
/// parameters) only when it finds as many distinct roots as the degree.
/// The solver returns the roots it finds of such a polynomial, which
/// `Ex::solve` checks against the count of distinct roots, but a
/// substitution took them as all its roots: before 0.44 `sin 4x = √3·sin x`
/// (in `t = tan(x/2)` a degree-9 polynomial over ℚ(√3)) gave `[π]`, without
/// the root `0` and the six others.  When the solver does not find them
/// all, a polynomial with algebraic coefficients is solved over the field
/// they generate ([`algebraic_roots::number_field_roots`]): in 0.44 the
/// substitution then refused, `sin 4x = √3·sin x` and `cos³x = √2·cos x`
/// were "not polynomial".  `Solver` when `num` is not a polynomial in `t`
/// (nothing to count).
fn numerator_roots(arena: &mut Arena, num: ExprId, t: ExprId, cx: &SolveCx) -> NumeratorRoots {
    if polybridge::expr_to_poly(arena, num, t).is_some() {
        return NumeratorRoots::Solver;
    }
    let Some(coeffs) = symbolic_poly_coeffs(arena, num, t) else {
        return NumeratorRoots::Solver;
    };
    let degree = coeffs.len().saturating_sub(1);
    if degree <= 2
        || solve_impl(arena, num, t, &cx.substituted())
            .roots()
            .map_or(0, <[ExprId]>::len)
            >= degree
    {
        return NumeratorRoots::Solver;
    }
    match algebraic_roots::number_field_roots(arena, num, t, false) {
        Some(roots) => NumeratorRoots::Found(roots),
        None => NumeratorRoots::Unknown,
    }
}

/// The roots in `t` of the substituted equation `sub = 0` of a change of
/// variable (refused when they cannot all be found, [`numerator_roots`]).
pub(super) fn substitution_roots(
    arena: &mut Arena,
    sub: ExprId,
    t: ExprId,
    cx: &SolveCx,
) -> Solutions {
    let full_num = full_numerator(arena, sub);
    match numerator_roots(arena, full_num, t, cx) {
        NumeratorRoots::Solver => solve_impl(arena, sub, t, &cx.substituted()),
        NumeratorRoots::Found(roots) => Solutions::found(roots),
        NumeratorRoots::Unknown => Solutions::refused(Reason::NotFound),
    }
}

/// Largest power of the new unknown a substitution may create.
pub(super) const MAX_SUBSTITUTION_DEGREE: i64 = 64;

/// Trigonometric equations in `sin(k·x)`, `cos(k·x)`, `tan(k·x)` (integer
/// `k`) of one variable: the multiple angles are expanded, and the
/// half-angle substitution `t = tan(x/2)` (`sin x = 2t/(1 + t²)`,
/// `cos x = (1 − t²)/(1 + t²)`, `tan x = 2t/(1 − t²)`) makes the
/// equation rational in `t`; each root gives `x = 2·atan(t)`.  The value
/// `t = ∞` (`x = π`) is not reached by the substitution and is added when
/// the equation holds there exactly.  SymPy: `solve(sin(x) + cos(x) - 1)`
/// → `[0, pi/2]`, `solve(sin(2*x) - sin(x))` → `[0, -pi/3, pi/3]` (SymPy
/// misses the root `π` there, which this adds); refused before 0.31.
pub(super) fn try_half_angle_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Solutions {
    half_angle_substitution(arena, expr, var, cx)
        .filter(|s| s.has_roots() || s.is_nested_periods())
        .unwrap_or_else(|| Solutions::refused(Reason::NotFound))
}

/// [`try_half_angle_substitution`]; `None` when the equation does not have
/// its shape.
fn half_angle_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    cx: &SolveCx,
) -> Option<Solutions> {
    let mut atoms: Vec<ExprId> = Vec::new();
    let mut multiple = false;
    walk_var_atoms(arena, expr, var, |id| {
        let arg = match arena.node(id) {
            ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) => *a,
            _ => return false,
        };
        let ok = if arg == var {
            true
        } else if let ExprNode::Mul(ch) = arena.node(arg)
            && ch.len() == 2
            && ch.contains(&var)
            && ch
                .iter()
                .any(|&c| arena.as_num(c).is_some_and(|r| r.is_integer()))
        {
            multiple = true;
            true
        } else {
            false
        };
        if ok && !atoms.contains(&id) {
            atoms.push(id);
        }
        ok
    })?;
    if atoms.len() < 2 && !multiple {
        return None;
    }
    let expanded = if multiple {
        // `tan(k·x)` as `sin(k·x)/cos(k·x)`, which `expand_trig` expands
        // (it leaves `tan(2x)` alone): `tan 2x = tan x` was refused before
        // 0.44.  Not as a rational function of `tan x`, which loses the
        // roots at the poles of `tan x` (`tan 2x = 0` at `x = π/2`).
        let mut e = expr;
        for &id in &atoms {
            if let ExprNode::Tan(arg) = *arena.node(id)
                && arg != var
            {
                let s = arena.sin(arg);
                let c = arena.cos(arg);
                let q = arena.div(s, c);
                e = arena.subs_structural(e, id, q);
            }
        }
        let e = arena.expand_trig_expr(e);
        crate::transforms::eval::eval(arena, e)
    } else {
        expr
    };
    // After the expansion only sin x, cos x, tan x may carry the variable.
    let (s, c, tn) = (arena.sin(var), arena.cos(var), arena.tan(var));
    walk_var_atoms(arena, expanded, var, |id| id == s || id == c || id == tn)?;
    let t = arena.symbol("__t_half");
    let one = arena.one;
    let two = arena.int(2);
    let t2 = arena.pow(t, two);
    let one_p = arena.add(&[one, t2]);
    let one_m = arena.sub(one, t2);
    let two_t = arena.mul(&[two, t]);
    let sin_t = arena.div(two_t, one_p);
    let cos_t = arena.div(one_m, one_p);
    let tan_t = arena.div(two_t, one_m);
    let mut sub = arena.subs_structural(expanded, s, sin_t);
    sub = arena.subs_structural(sub, c, cos_t);
    sub = arena.subs_structural(sub, tn, tan_t);
    let sub = crate::transforms::eval::eval(arena, sub);
    if expr_contains_var(arena, sub, var) {
        return None;
    }
    let half = arena.rational(1, 2);
    let half_x = arena.mul(&[half, var]);
    let generator = arena.tan(half_x);
    // (No root in `t`, or none found: `x = π` below may still be one.)
    let (t_roots, t_missing) = match solutions_unless_identity(arena, sub, t, cx)?
        .without_conditions()
        .into_found()
    {
        Ok(found) => found,
        Err(Solutions::Refused(reason)) => (Vec::new(), Some(Missing::Alternative(reason))),
        Err(_) => (Vec::new(), None),
    };
    let mut out = back_substitute(arena, generator, t_roots, var, cx);
    // x = π (t = ∞), when the equation holds there exactly.
    let pi = arena.pi;
    let at_pi = crate::transforms::subs::subs(arena, expr, var, pi);
    let at_pi = crate::transforms::eval::eval(arena, at_pi);
    let zero_at_pi = arena.is_zero_structural(at_pi)
        || (crate::base::walk::free_symbols(arena, at_pi).is_empty()
            && crate::poly::algebraic::is_zero_checked(arena, at_pi) == Some(true));
    if zero_at_pi {
        let value = match cx.period() {
            Some(n) => {
                let two_pi_n = arena.mul(&[two, pi, n]);
                arena.add(&[pi, two_pi_n])
            }
            None => pi,
        };
        out = out.union(Solutions::found(vec![value]));
    }
    Some(out.lacking(t_missing))
}

/// Sums of logarithms `Σ cᵢ·ln fᵢ(x) + g = 0` (rational `cᵢ`, `g` free of
/// `x`): exponentiated, `∏ fᵢ^cᵢ = e^(−g)`, solved, and every candidate
/// checked against the original equation (principal logarithms: the
/// candidate `x = −3` of `ln x + ln(x + 1) = ln 6` has the residual `2πi`
/// and is dropped).  SymPy: `solve(log(x) + log(x + 1) - log(6))` → `[2]`;
/// refused before 0.31.
pub(super) fn try_log_sum(arena: &mut Arena, expr: ExprId, var: ExprId, cx: &SolveCx) -> Solutions {
    match log_sum_equation(arena, expr, var) {
        // (no root of the exponentiated equation is not taken as a proof)
        Some(eq) => match solve_raw(arena, eq, var, cx) {
            found if found.has_roots() || found.is_nested_periods() => found,
            _ => Solutions::refused(Reason::NotFound),
        },
        None => Solutions::refused(Reason::NotFound),
    }
}

/// The exponentiated equation of [`try_log_sum`], or `None` when the
/// equation does not have its shape.
fn log_sum_equation(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    let terms: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => return None,
    };
    let mut factors: Vec<ExprId> = Vec::new();
    let mut logs: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    let mut var_rest: Vec<ExprId> = Vec::new();
    for term in terms {
        if !expr_contains_var(arena, term, var) {
            rest.push(term);
            continue;
        }
        let (c, f) = match arena.node(term).clone() {
            ExprNode::Ln(f) => (arena.one, f),
            ExprNode::Mul(ch) if ch.len() == 2 => {
                let (k, l) = if arena.as_num(ch[0]).is_some() {
                    (ch[0], ch[1])
                } else {
                    (ch[1], ch[0])
                };
                match (arena.as_num(k), arena.node(l)) {
                    (Some(_), ExprNode::Ln(f)) => (k, *f),
                    _ => {
                        var_rest.push(term);
                        continue;
                    }
                }
            }
            _ => {
                var_rest.push(term);
                continue;
            }
        };
        factors.push(arena.pow(f, c));
        logs.push(f);
    }
    if var_rest.is_empty() {
        if factors.len() < 2 {
            return None;
        }
    } else {
        // One logarithm of a non-linear argument beside other terms in `x`
        // (`ln(x²) = 3x` as `x² = e^(3x)`; SymPy `solve(log(x**2) - 3*x)`
        // → `[-2*LambertW(-3/2)/3, -2*LambertW(3/2)/3]`, refused before
        // 0.44).  A linear argument is left to the Lambert forms
        // (`ln x = 2x + 1`), and the other terms must be free of logarithms
        // of `x`.
        let log_free = !var_rest.iter().any(|&r| {
            crate::base::walk::post_order_ids(arena, r).into_iter().any(
                |id| matches!(arena.node(id), ExprNode::Ln(a) if expr_contains_var(arena, *a, var)),
            )
        });
        if factors.len() != 1 || !log_free || linear_rate(arena, logs[0], var).is_some() {
            return None;
        }
        rest.extend(var_rest);
    }
    let lhs = arena.mul(&factors);
    let g = arena.add(&rest);
    let minus_g = arena.neg(g);
    let rhs = arena.exp(minus_g);
    let eq = arena.sub(lhs, rhs);
    let eq = crate::transforms::eval::eval(arena, eq);
    if !expr_contains_var(arena, eq, var) {
        return None;
    }
    Some(eq)
}

/// Collect candidate generator functions: exp(x), sin(x), cos(x), ln(x), x^(k), etc.
pub(super) fn collect_generators(
    arena: &Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Vec<ExprId> {
    let mut generators = Vec::new();
    let mut visited = std::collections::HashSet::new();
    collect_gens_recursive(arena, expr, var, var_sym, &mut generators, &mut visited);
    generators
}

#[allow(clippy::only_used_in_recursion)]
fn collect_gens_recursive(
    arena: &Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
    gens: &mut Vec<ExprId>,
    visited: &mut std::collections::HashSet<ExprId>,
) {
    if !visited.insert(expr) {
        return;
    }
    match arena.node(expr).clone() {
        ExprNode::Exp(inner) if inner == var && !gens.contains(&expr) => {
            gens.push(expr);
        }
        ExprNode::Sin(inner) | ExprNode::Cos(inner) | ExprNode::Tan(inner)
            if inner == var && !gens.contains(&expr) =>
        {
            gens.push(expr);
        }
        ExprNode::Ln(inner) if inner == var && !gens.contains(&expr) => {
            gens.push(expr);
        }
        ExprNode::Pow(base, exp)
            if base == var && !expr_contains_var(arena, exp, var)
            // x^(1/n) or x^k type
            && !gens.contains(&expr) =>
        {
            gens.push(expr);
        }
        ExprNode::Add(ref children) | ExprNode::Mul(ref children) => {
            for &c in children {
                collect_gens_recursive(arena, c, var, var_sym, gens, visited);
            }
        }
        ExprNode::Pow(base, exp) => {
            collect_gens_recursive(arena, base, var, var_sym, gens, visited);
            collect_gens_recursive(arena, exp, var, var_sym, gens, visited);
        }
        ExprNode::Neg(inner)
        | ExprNode::Exp(inner)
        | ExprNode::Ln(inner)
        | ExprNode::Sin(inner)
        | ExprNode::Cos(inner)
        | ExprNode::Tan(inner) => {
            collect_gens_recursive(arena, inner, var, var_sym, gens, visited);
        }
        _ => {}
    }
}

/// Rewrite `exp(k*x)` as `exp(x)^k` for small integer k throughout the expression.
pub(super) fn rewrite_exp_powers(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    gen_exp_x: ExprId,
) -> ExprId {
    let mut result = expr;
    // Check small positive integer multiples: exp(k*x) → exp(x)^k
    for k in 2i64..=6 {
        let k_id = arena.int(k);
        let k_var = arena.mul(&[k_id, var]);
        let exp_k_var = arena.exp(k_var);
        let gen_pow_k = arena.pow(gen_exp_x, k_id);
        result = arena.subs_structural(result, exp_k_var, gen_pow_k);
    }
    // Also handle negative multiples: exp(-k*x) → exp(x)^(-k)
    for k in [-1i64, -2, -3] {
        let k_id = arena.int(k);
        let k_var = arena.mul(&[k_id, var]);
        let exp_k_var = arena.exp(k_var);
        let gen_pow_k = arena.pow(gen_exp_x, k_id);
        result = arena.subs_structural(result, exp_k_var, gen_pow_k);
    }
    result
}
