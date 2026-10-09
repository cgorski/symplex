//! The ODE / recurrence / summation hunt: silent wrong answers and cheap
//! refusals found by substituting solutions back (ODEs), iterating
//! recurrences exactly, and comparing closed-form sums with the explicit
//! finite sum at concrete `n` (and infinite sums with mpmath `nsum`).
//! Every reference value cites the oracle that produced it: the explicit
//! sum, the substitution residual, exact iteration, a SymPy 1.14 call or
//! an mpmath 1.3 `nsum` (`mp.dps = 40`).

use symplex::prelude::*;

/// The ODE `text = 0` with `D1Y … D4Y` standing for `y′ … y⁗` (of `y(x)`).
fn ode(ctx: &Context, text: &str) -> Ex {
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let mut e = ctx.parse(text).unwrap();
    let mut ds = Vec::new();
    let mut d = y.clone();
    for _ in 0..4 {
        d = d.formal_diff(&x);
        ds.push(d.clone());
    }
    for (k, dk) in ds.iter().enumerate().rev() {
        e = e.subs(&ctx.symbol(&format!("D{}Y", k + 1)), dk);
    }
    e
}

/// Number of distinct integration constants `C1, C2, …` in `e`.
fn constant_count(e: &Ex) -> usize {
    e.free_symbols()
        .iter()
        .filter(|v| {
            let s = v.to_string();
            s.len() > 1 && s.starts_with('C') && s[1..].chars().all(|c| c.is_ascii_digit())
        })
        .count()
}

/// `Σ_{k=lo}^{n} term` by adding the terms one by one.
fn explicit_sum(ctx: &Context, term: &Ex, lo: i64, n: i64) -> Ex {
    let k = ctx.symbol("k");
    let nn = ctx.symbol("n");
    let mut acc = ctx.zero();
    for j in lo..=n {
        acc = (&acc + &term.subs_i64(&k, j).subs_i64(&nn, n)).eval();
    }
    acc
}

fn decimal(e: &Ex, digits: u32) -> f64 {
    e.eval_decimal(digits).unwrap().parse::<f64>().unwrap()
}

/// A term with a pole inside the range makes `Σ_{k=lo}^{n}` undefined; the
/// pole check looked at the numerator/denominator split of the term as
/// written, which for a sum of fractions is `(term, 1)`, so
/// `Σ_{k=−1}^{n} (1/k − 1/(k+1))` came out `−1/(n + 1) − 1` (the terms at
/// `k = −1` and `k = 0` are `zoo`) and `Σ_{k=−1}^{n} (1/k² − 1/(k+1)² − 2)`
/// came out `−2n − 1/(n+1)² − 3`.  Oracle: the explicit sum (undefined).
#[test]
fn a_pole_in_the_range_of_a_sum_of_fractions_is_seen() {
    let ctx = Context::new();
    let (k, n) = (ctx.symbol("k"), ctx.symbol("n"));
    for (term, lo) in [
        ("1/k - 1/(k + 1)", -1),
        ("1/k - 1/(k + 1)", 0),
        ("1/k^2 - 1/(k + 1)^2 - 2", -1),
    ] {
        let t = ctx.parse(term).unwrap();
        assert!(
            t.try_summation(&k, &ctx.int(lo), &n).is_err(),
            "Σ_{{k={lo}}}^n {term} has a pole in range"
        );
        assert!(t.try_summation(&k, &ctx.int(lo), &ctx.infinity()).is_err());
    }
    // Without a pole in range the telescoping sum is unchanged.
    let t = ctx.parse("1/k - 1/(k + 1)").unwrap();
    let s = t.try_summation(&k, &ctx.int(1), &n).unwrap();
    for nv in 0..8 {
        assert_eq!(s.subs_i64(&n, nv).eval(), explicit_sum(&ctx, &t, 1, nv));
    }
}

/// Gosper's antidifference divides by the polynomial factors of the term;
/// `(k+1)!/(k−1)!` was kept as factorials, so the antidifference of
/// `(−2)^k·(k+1)!/(k−1)!` had `0·zoo` at `k = 0` and the whole sum
/// `Σ_{k=0}^{n}` came out `nan` (6 of 600 hunter cases).  The ratio is the
/// polynomial `k(k + 1)`.  Oracle: the explicit sum (`1/(−1)! = 0`).
#[test]
fn factorial_ratios_in_gosper_terms_are_polynomials() {
    let ctx = Context::new();
    let (k, n) = (ctx.symbol("k"), ctx.symbol("n"));
    for (term, lo) in [
        ("(-2)^k*factorial(k + 1)/factorial(k - 1)", 0),
        ("2^k*factorial(k + 1)/factorial(k - 1)", -1),
        ("k^3*2^k*factorial(k + 1)/factorial(k - 1)", -1),
    ] {
        let t = ctx.parse(term).unwrap();
        let s = t.try_summation(&k, &ctx.int(lo), &n).unwrap();
        for nv in 0..8 {
            assert_eq!(
                s.subs_i64(&n, nv).eval(),
                explicit_sum(&ctx, &t, lo, nv),
                "Σ {term} at n = {nv}: {s}"
            );
        }
    }
}

/// `Σ_{k=0}^{n} k·C(n,k)·xᵏ = n·x·(1+x)^(n−1)` at `x = −1` was
/// `−n·0^(n−1)`: `nan` at `n = 0`, where the sum is `0`.  It is `−1` at
/// `n = 1` and `0` otherwise (SymPy 1.14 returns `0`, wrong at `n = 1`).
/// Oracle: the explicit sum.
#[test]
fn alternating_k_binomial_sum_is_defined_at_n_zero() {
    let ctx = Context::new();
    let (k, n) = (ctx.symbol("k"), ctx.symbol("n"));
    let t = ctx.parse("k*binomial(n, k)*(-1)^k").unwrap();
    let s = t.try_summation(&k, &ctx.int(0), &n).unwrap();
    for nv in 0..8 {
        assert_eq!(
            s.subs_i64(&n, nv).eval(),
            explicit_sum(&ctx, &t, 0, nv),
            "{s}"
        );
    }
}

/// Binomial sums with a lower limit other than `0` were refused:
/// `Σ_{k=1}^{n} C(n,k)` (SymPy `2**n - 1`), `Σ_{k=1}^{n} C(n,k)²` (SymPy
/// `binomial(2n, n) − 1`), `Σ_{k=1}^{n} (−1)^k·C(n,k)/(k+1)` (SymPy
/// `−n/(n + 1)`); `Σ_{k=0}^{n} C(n,k)·xᵏ/(k+1)` too (SymPy
/// `((x+1)^(n+1) − 1)/(x(n+1))`).  Oracle: the explicit sum, `n = 0..8`.
#[test]
fn binomial_sums_from_other_lower_limits() {
    let ctx = Context::new();
    let (k, n, x) = (ctx.symbol("k"), ctx.symbol("n"), ctx.symbol("x"));
    let s = ctx
        .parse("binomial(n, k)")
        .unwrap()
        .try_summation(&k, &ctx.int(1), &n)
        .unwrap();
    assert_eq!(s.to_string(), "2^n - 1");
    for (term, lo) in [
        ("binomial(n, k)", 2),
        ("binomial(n, k)", -2),
        ("binomial(n, k)^2", 1),
        ("k^2*binomial(n, k)", 3),
        ("(-1)^k*binomial(n, k)/(k + 1)", 1),
        ("binomial(n, k)*(1/2)^k", 2),
        ("binomial(n, k)*(3/7)^k/(k + 1)", 0),
    ] {
        let t = ctx.parse(term).unwrap();
        let s = t.try_summation(&k, &ctx.int(lo), &n).unwrap();
        for nv in 0..8 {
            assert_eq!(
                s.subs_i64(&n, nv).eval(),
                explicit_sum(&ctx, &t, lo, nv),
                "Σ_{{k={lo}}}^n {term} at n = {nv}: {s}"
            );
        }
    }
    // Symbolic x: the closed form at x = 3/7 is the explicit sum.
    let t = ctx.parse("binomial(n, k)*x^k").unwrap();
    let s = t.try_summation(&k, &ctx.int(2), &n).unwrap();
    for nv in 0..6 {
        let want = explicit_sum(&ctx, &t, 2, nv)
            .subs(&x, &ctx.rational(3, 7))
            .eval();
        assert_eq!(
            s.subs_i64(&n, nv).subs(&x, &ctx.rational(3, 7)).eval(),
            want
        );
    }
}

/// `exp(k·x)` is the geometric term `(eˣ)ᵏ`; it was refused (SymPy 1.14:
/// `Piecewise((n + 1, Eq(exp(x), 1)), ((1 - exp(x*(n + 1)))/(1 - exp(x)),
/// True))`).  Oracle: the explicit sum at `x = 3/7`.
#[test]
fn exponential_of_a_multiple_of_k_is_geometric() {
    let ctx = Context::new();
    let (k, n, x) = (ctx.symbol("k"), ctx.symbol("n"), ctx.symbol("x"));
    let t = ctx.parse("exp(k*x)").unwrap();
    let s = t.try_summation(&k, &ctx.int(0), &n).unwrap();
    for nv in 0..6 {
        let xv = ctx.rational(3, 7);
        let got = decimal(&s.subs_i64(&n, nv).subs(&x, &xv), 30);
        let want = decimal(&explicit_sum(&ctx, &t, 0, nv).subs(&x, &xv), 30);
        assert!(
            (got - want).abs() < 1e-12 * want.abs().max(1.0),
            "n = {nv}: {got} vs {want}"
        );
    }
    // A real constant ratio needs no case split.
    let s = ctx
        .parse("exp(2*k)")
        .unwrap()
        .try_summation(&k, &ctx.int(0), &n)
        .unwrap();
    assert!(!s.to_string().contains("Piecewise"), "{s}");
}

/// Even rational functions with non-rational poles, by the residue theorem
/// (SymPy's `eval_sum_residue`); all were refused.  SymPy 1.14:
/// `summation(1/(k**2 + 1), (k, 0, oo)) = 1/2 + pi/(2*tanh(pi))`, from 1
/// `−1/2 + …`, from −3 `13/10 + …`, `(−1)^k/(k² + 1)` from 0
/// `pi/(2*sinh(pi)) + 1/2`, `1/(k² + 2k + 2)` from 0 `−1/2 + pi/(2*tanh(pi))`.
/// mpmath `nsum` (dps 40) for every value.
#[test]
fn even_rational_series_by_residues() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let cases: [(&str, i64, f64); 9] = [
        ("1/(k^2 + 1)", 0, 2.076_674_047_468_581),
        ("1/(k^2 + 1)", 1, 1.076_674_047_468_581_2),
        ("1/(k^2 + 1)", -3, 2.876_674_047_468_581),
        ("(-1)^k/(k^2 + 1)", 0, 0.636_014_527_491_066_6),
        ("1/(k^2 + 2*k + 2)", 0, 1.076_674_047_468_581_2),
        ("1/(k^2 - 2)", 2, 0.943_189_592_299_379_9),
        ("1/(k^4 + 1)", 0, 1.578_477_579_667_136_8),
        ("k^2/(k^4 + 4)", 1, 0.788_337_023_734_290_6),
        ("1/((k^2 + 1)*(k^2 + 4))", 0, 0.388_756_802_049_155_35),
    ];
    for (term, lo, want) in cases {
        let s = ctx
            .parse(term)
            .unwrap()
            .try_summation(&k, &ctx.int(lo), &ctx.infinity())
            .unwrap_or_else(|e| panic!("Σ_{{k≥{lo}}} {term}: {e}"));
        let got = decimal(&s, 30);
        assert!(
            (got - want).abs() < 1e-14,
            "Σ_{{k≥{lo}}} {term} = {s} = {got}, nsum {want}"
        );
    }
    let s = ctx
        .parse("1/(k^2 + 1)")
        .unwrap()
        .try_summation(&k, &ctx.int(0), &ctx.infinity())
        .unwrap();
    assert_eq!(s.to_string(), "cosh(pi)*pi/(2*sinh(pi)) + 1/2");
    // Not even after an integer shift: still refused (as SymPy).
    assert!(
        ctx.parse("1/(k^2 + k + 1)")
            .unwrap()
            .try_summation(&k, &ctx.int(0), &ctx.infinity())
            .is_err()
    );
}

/// Power series with a shifted factorial: `Σ_{k≥0} 1/(k + 2)! = e − 2` and
/// `Σ_{k≥0} 2ᵏ/(k + 1)! = (e² − 1)/2` (SymPy 1.14) were refused.
#[test]
fn power_series_with_shifted_factorials() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let s = ctx
        .parse("1/factorial(k + 2)")
        .unwrap()
        .try_summation(&k, &ctx.int(0), &ctx.infinity())
        .unwrap();
    assert_eq!(s.to_string(), "-2 + E");
    let s = ctx
        .parse("2^k/factorial(k + 1)")
        .unwrap()
        .try_summation(&k, &ctx.int(0), &ctx.infinity())
        .unwrap();
    assert!(
        (decimal(&s, 30) - 3.194_528_049_465_325).abs() < 1e-14,
        "{s}"
    );
    let s = ctx
        .parse("(-1)^k/factorial(2*k + 2)")
        .unwrap()
        .try_summation(&k, &ctx.int(0), &ctx.infinity())
        .unwrap();
    // 1 − cos 1 (mpmath nsum 0.45969769413186028259906339)
    assert!(
        (decimal(&s, 30) - 0.459_697_694_131_860_3).abs() < 1e-14,
        "{s}"
    );
}

/// An ODE multiplied or divided by a factor, or expanded after such a
/// product, was not recognised (229 of 300 hunter cases unsolved); SymPy's
/// `dsolve` solves all of the cases below.  The equation is now also tried
/// with the factors free of `y` dropped and divided by the coefficient of
/// its highest derivative.  Oracle: `check_ode_solution` (substitution) and
/// the number of constants.
#[test]
fn ode_in_a_multiplied_or_expanded_form_is_solved() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for (text, order) in [
        ("(D1Y - y)/(-1)", 1),
        ("exp(x)*(D2Y + y - tan(x))", 2),
        ("(-x^2)*(8*y + 2*D2Y - sin(x)*cos(x))", 2),
        ("(D2Y - 4*D1Y + 3*y - cos(3*x))/(x^2 + 1)", 2),
        ("(4*y + 4*D1Y + D2Y + D3Y - sin(x))/y", 3),
        ("y*(D1Y - cos(x)*(1 + y^2))", 1),
        ("(1/x)*(-x^2*D2Y + 3*x*D1Y - 4*y - 1)", 2),
    ] {
        let e = ode(&ctx, text);
        let sol = e.solve_ode(&y, &x);
        assert!(!sol.has_unevaluated() && !sol.contains(&y), "{text}: {sol}");
        assert_eq!(constant_count(&sol), order, "{text}: {sol}");
        assert!(e.check_ode_solution(&sol, &y, &x), "{text}: {sol}");
    }
    // Expanded product: (x + 3)·(y‴ − 3y″ + 4y′ − 12y) = 0
    // (SymPy: C1·e^{3x} + C2·sin 2x + C3·cos 2x).
    let e = ode(&ctx, "(x + 3)*(D3Y - 3*D2Y + 4*D1Y - 12*y)").expand();
    let sol = e.solve_ode(&y, &x);
    assert_eq!(constant_count(&sol), 3, "{sol}");
    assert!(e.check_ode_solution(&sol, &y, &x), "{sol}");
}

/// `y′ = e^{x+y}` is separable once `exp(x + y)` is `eˣ·eʸ`; it was
/// refused (SymPy 1.14: `y = −log(C1 − exp(x))`).  Oracle:
/// `check_ode_solution`, and the IVP `y(0) = 0` (`−ln(2 − eˣ)`).
#[test]
fn separable_with_an_exponential_of_a_sum() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for text in ["D1Y - exp(x + y)", "D1Y - x*exp(2*x + 3*y)"] {
        let e = ode(&ctx, text);
        let sol = e.solve_ode(&y, &x);
        assert!(!sol.has_unevaluated() && !sol.contains(&y), "{text}: {sol}");
        assert_eq!(constant_count(&sol), 1, "{text}: {sol}");
        assert!(e.check_ode_solution(&sol, &y, &x), "{text}: {sol}");
    }
    let e = ode(&ctx, "D1Y - exp(x + y)");
    let ic = InitialCondition {
        order: 0,
        x: ctx.int(0),
        value: ctx.int(0),
    };
    let sol = e.solve_ode_ivp(&y, &x, &[ic]).unwrap();
    assert_eq!(sol.to_string(), "-ln(-exp(x) + 2)");
}

/// Cauchy–Euler equations of order ≥ 3 were refused (SymPy
/// `nth_linear_euler_eq_homogeneous` / `_nonhomogeneous_*` solve them):
/// `x³y‴ − 3x²y″ + 6xy′ − 6y = ln x` is `C1·x³ + C2·x² + C3·x − ln(x)/6 −
/// 11/36`.  Oracle: `check_ode_solution` and the number of constants.
#[test]
fn cauchy_euler_of_higher_order() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for (text, order) in [
        ("x^3*D3Y - 3*x^2*D2Y + 6*x*D1Y - 6*y - log(x)", 3),
        ("x^3*D3Y - 2*x*D1Y + 2*y", 3),
        ("x^3*D3Y + 3*x^2*D2Y + x*D1Y - x", 3),
        ("x^4*D4Y + 6*x^3*D3Y + 7*x^2*D2Y + x*D1Y - y", 4),
    ] {
        let e = ode(&ctx, text);
        let sol = e.solve_ode(&y, &x);
        assert!(!sol.has_unevaluated(), "{text}: {sol}");
        assert_eq!(constant_count(&sol), order, "{text}: {sol}");
        assert!(e.check_ode_solution(&sol, &y, &x), "{text}: {sol}");
    }
    let e = ode(&ctx, "x^3*D3Y - 3*x^2*D2Y + 6*x*D1Y - 6*y - log(x)");
    assert_eq!(
        e.solve_ode(&y, &x).to_string(),
        "C1*x^3 + C2*x^2 + C3*x - 1/6*ln(x) - 11/36"
    );
}

/// Initial conditions at a point `x₀ ≠ 0` of a fourth-order
/// constant-coefficient equation took over a minute (symbolic elimination
/// over `cos(1/3)`, `e^{−1/6}·sin(√3/6)`, …).  The general solution is now
/// translated to `x₀` first.  Oracle: `check_ode_solution` and the
/// conditions, evaluated.
#[test]
fn initial_conditions_away_from_zero_of_a_fourth_order_equation() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = ode(&ctx, "y + D1Y + 2*D2Y + D3Y + D4Y - exp(2*x)");
    let x0 = ctx.rational(1, 3);
    let values = [
        ctx.rational(2, 3),
        ctx.rational(5, 4),
        ctx.rational(-3, 7),
        ctx.rational(7, 5),
    ];
    let ics: Vec<InitialCondition> = values
        .iter()
        .enumerate()
        .map(|(order, v)| InitialCondition {
            order,
            x: x0.clone(),
            value: v.clone(),
        })
        .collect();
    let sol = e.solve_ode_ivp(&y, &x, &ics).unwrap();
    assert!(e.check_ode_solution(&sol, &y, &x), "{sol}");
    let mut d = sol.clone();
    for v in &values {
        let at = decimal(&(&d.subs(&x, &x0) - v), 30);
        assert!(at.abs() < 1e-20, "{sol}: {at}");
        d = d.diff(&x);
    }
}

/// `a(n+1) = p(n)·a(n) + q(n)` with a rational `p` left the product
/// `Π p(k)` formal; it now has its Gamma/factorial closed form.  Oracle:
/// the recurrence iterated exactly, `n = 0..20`.
#[test]
fn first_order_recurrence_with_a_rational_coefficient() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    for (p, q, a0) in [
        ("(n + 3)/(n + 1)", "1", 2),
        ("(2*n + 1)/(2*n + 2)", "0", 1),
        ("(n + 1)^2", "0", 1),
    ] {
        let (pe, qe) = (ctx.parse(p).unwrap(), ctx.parse(q).unwrap());
        let sol = symplex::rsolve::rsolve_first_order(&pe, &qe, &n, Some(&ctx.int(a0))).unwrap();
        assert!(!sol.has_unevaluated(), "{p}: {sol}");
        let mut a = ctx.int(a0);
        for j in 0..=20 {
            assert_eq!(sol.subs_i64(&n, j).eval(), a, "a({j}) for p = {p}: {sol}");
            a = (&(&pe.subs_i64(&n, j) * &a) + &qe.subs_i64(&n, j)).eval();
        }
    }
}
