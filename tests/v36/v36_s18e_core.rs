//! Tests of session 18. Lane E of the 0.60.1 block: core types and hangs (`args` of a relation, `Poly::is_zero`,
//! `mod_pow`, the compiled `fibonacci`, closures inside `replace`, Gröbner bases under a
//! budget, `Frequency::from_angular_velocity`).
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use std::sync::mpsc;
use std::time::Duration;

use num_bigint::BigInt;
use symplex::macros::Rule;
use symplex::prelude::*;

/// Run `work` on a thread of its own and wait `secs` seconds for its answer: `None` when the
/// call did not return (the thread is then left behind, blocked or computing: under nextest
/// every test is a process of its own).
fn answered_within<T: Send + 'static>(
    secs: u64,
    work: impl FnOnce() -> T + Send + 'static,
) -> Option<T> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(work());
    });
    rx.recv_timeout(Duration::from_secs(secs)).ok()
}

// ── H01: closures inside `replace` ──────────────────────────────────────────

/// H01.  Up to 0.60.0 the closure of `Ex::replace` ran with the context write-locked: a
/// closure that called the context (`ctx.int(2)`) never returned (killed after 20 s).  Now
/// no lock is held while it runs.
///
/// Oracle: substitution by hand, `x² + sin x` at `x := 2` is `4 + sin 2` (SymPy
/// `(x**2 + sin(x)).xreplace({x: Integer(2)})` = `sin(2) + 4`).
#[test]
fn replace_with_a_closure_that_calls_the_context_returns() {
    let shown = answered_within(10, || {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let e = x.powi(2) + x.sin();
        let replaced = e.replace(|v| if v == x { Some(ctx.int(2)) } else { None });
        (replaced == ctx.int(4) + ctx.int(2).sin()).then(|| replaced.to_string())
    });
    assert_eq!(shown.flatten().as_deref(), Some("sin(2) + 4"));
}

/// H01.  A closure that prints a handle (`x.to_string()`, a read of the context) inside
/// `Ex::replace` never returned up to 0.60.0.
///
/// Oracle: the walk replaces nothing, so the result is the expression itself.
#[test]
fn replace_with_a_closure_that_prints_returns() {
    let same = answered_within(10, || {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let e = x.powi(2) + x.sin();
        let names = std::cell::RefCell::new(Vec::new());
        let replaced = e.replace(|v| {
            if v == x {
                names.borrow_mut().push(x.to_string());
            }
            None
        });
        (replaced == e, names.into_inner())
    });
    assert_eq!(same, Some((true, vec!["x".to_string()])));
}

/// H01.  A closure that asks a query (`x.is_positive()`) inside `Ex::replace` never
/// returned up to 0.60.0.
///
/// Oracle: for a positive `p`, `|p|` is `p` and `√(p²)` is `p` (the definition); the
/// replacement `p → 3` asked for by the closure gives `3 + sin(3)`.
#[test]
fn replace_with_a_closure_that_asks_a_query_returns() {
    let shown = answered_within(10, || {
        let ctx = Context::new();
        let p = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
        let three = ctx.int(3);
        let e = &p + &p.sin();
        let replaced = e.replace(|v| {
            if v == p && p.is_positive() == Some(true) {
                Some(three.clone())
            } else {
                None
            }
        });
        replaced.to_string()
    });
    assert_eq!(shown.as_deref(), Some("sin(3) + 3"));
}

/// H01.  `try_replace` held the same lock: a building closure never returned up to 0.60.0.
/// Its two refusals are unchanged: a replacement from another context, and a bound variable
/// replaced by a non-symbol.
///
/// Oracle: substitution by hand, `x² + sin x` at `x + 1`; SymPy
/// `Sum(k*x**2, (x, 0, 3)).xreplace({x: Rational(1, 3)})` raises "Invalid limits".
#[test]
fn try_replace_with_a_closure_that_builds_returns() {
    let outcome = answered_within(10, || {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let e = x.powi(2) + x.sin();
        let built = e
            .try_replace(|v| if v == x { Some(&x + 1) } else { None })
            .map(|r| r.to_string());
        let other = Context::new();
        let foreign = e
            .try_replace(|v| {
                if v == x {
                    Some(other.symbol("y"))
                } else {
                    None
                }
            })
            .is_err();
        let sum = ctx.parse("Sum(k*x^2, x, 0, 3)").unwrap();
        let bound = sum
            .try_replace(|v| {
                if v == x {
                    Some(ctx.rational(1, 3))
                } else {
                    None
                }
            })
            .is_err();
        (built.ok(), foreign, bound)
    });
    assert_eq!(
        outcome,
        Some((Some("(x + 1)^2 + sin(x + 1)".to_string()), true, true))
    );
}

/// H01.  What the closure is shown did not change with the lock: every distinct node of
/// the expression as it was before the call, once, children before parents — also the
/// nodes below one that is replaced — and never a replacement.
///
/// Oracle: the nodes of `sin(x)^2 + cos(x)` counted by hand: `x`, `sin(x)`, `2`,
/// `sin(x)^2`, `cos(x)`, the sum: six.  With `sin(x) → y` the result is `y^2 + cos(x)`.
#[test]
fn replace_shows_the_closure_each_original_node_once() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = x.sin().powi(2) + x.cos();
    let seen = std::cell::RefCell::new(Vec::new());
    let sin_x = x.sin();
    let replaced = e.replace(|v| {
        seen.borrow_mut().push(v.id());
        (v == sin_x).then(|| y.clone())
    });
    assert_eq!(replaced, y.powi(2) + x.cos());
    let seen = seen.into_inner();
    assert_eq!(seen.len(), 6);
    let distinct: std::collections::HashSet<_> = seen.iter().copied().collect();
    assert_eq!(distinct.len(), 6);
    // Children before parents: the root is shown last, and `y` (the replacement) never.
    assert!(seen.last().copied() == Some(e.id()));
    assert!(!seen.contains(&y.id()));
}

/// H01.  A closure may call `replace` itself (it holds no lock either way).  Up to 0.60.0
/// the inner call waited for the outer call's lock for ever.
///
/// Oracle: by hand: in `x*y + y`, `y → (x + 1 with x → 2) = 3` gives `3*x + 3`.
#[test]
fn replace_inside_the_closure_of_replace_returns() {
    let shown = answered_within(10, || {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let e = &x * &y + &y;
        let replaced =
            e.replace(|v| (v == y).then(|| (&x + 1).replace(|w| (w == x).then(|| ctx.int(2)))));
        replaced.to_string()
    });
    assert_eq!(shown.as_deref(), Some("3*x + 3"));
}

/// H01, the other public methods that take a closure: `Equation::apply`,
/// `Ex::apply_until_stable`, `Matrix::map`, `Rule::new_with_guard` and `Rule::new_fn` run
/// their closures with no lock held (they did before; this pins it).
///
/// Oracle: each value by hand — `(x + 1)² − 1` expanded is `x² + 2x`; `sin(a)` with the
/// guard "`a` is positive" fires on `sin(p)` only; the closure rule `f: a² → 2a` on `x²`
/// gives `2x`.
#[test]
fn the_other_closure_taking_methods_hold_no_lock() {
    let done = answered_within(10, || {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let p = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();

        let eq = Equation::new((&x + 1).powi(2), ctx.int(1)).apply(|s| (s - ctx.int(1)).expand());
        assert_eq!(eq.lhs().to_string(), "x^2 + 2*x");
        assert_eq!(*eq.rhs(), ctx.int(0));

        let (stable, _) = (&x + 1)
            .powi(2)
            .apply_until_stable(5, |e| e.expand() + ctx.int(0));
        assert_eq!(stable.to_string(), "x^2 + 2*x + 1");

        let m = Matrix::new(vec![vec![x.clone(), ctx.int(1)]]).unwrap();
        let mapped = m.map(|e| e + ctx.int(1));
        assert_eq!(mapped.try_get(0, 1).unwrap().to_string(), "2");

        let a = ctx.symbol("a_");
        let guarded = Rule::new_with_guard("sin_pos", &a.sin(), &ctx.int(0), |b| {
            b.get("a_").is_some_and(|v| v.is_positive() == Some(true))
        });
        assert_eq!(guarded.apply(&p.sin()), Some(ctx.int(0)));
        assert_eq!(guarded.apply(&x.sin()), None);

        let two = ctx.int(2);
        let doubled = Rule::new_fn("square", &a.powi(2), move |b| b.get("a_").map(|v| v * &two));
        assert_eq!(
            doubled.apply(&x.powi(2)).map(|r| r.to_string()).as_deref(),
            Some("2*x")
        );
        true
    });
    assert_eq!(done, Some(true));
}

// ── H02: Gröbner bases under a work budget ──────────────────────────────────

/// The cyclic system in `n` variables (Björck's `cyclic-n` roots).
fn cyclic(ctx: &Context, n: usize) -> (Vec<Ex>, Vec<Ex>) {
    let vars: Vec<Ex> = (0..n).map(|i| ctx.symbol(&format!("x{i}"))).collect();
    let mut polys = Vec::new();
    for k in 1..n {
        let mut sum = ctx.int(0);
        for i in 0..n {
            let mut prod = ctx.int(1);
            for j in 0..k {
                prod = &prod * &vars[(i + j) % n];
            }
            sum = &sum + &prod;
        }
        polys.push(sum);
    }
    let mut prod = ctx.int(1);
    for v in &vars {
        prod = &prod * v;
    }
    polys.push(&prod - 1);
    (polys, vars)
}

/// H02.  The budget must not change a basis that was computed before: cyclic-5 (0.02 s in a
/// release build, 600,000 of the 100,000,000 work units) has its reduced grevlex basis of 20
/// polynomials and its lex basis of 11, and every generator reduces to 0 modulo the basis.
///
/// Oracle: SymPy 1.14 `len(groebner(cyclic5, *xs, order='grevlex').exprs)` = 20 and
/// `order='lex'` = 11 (the reduced basis is unique); membership of the generators is the
/// definition of a basis of their ideal.
#[test]
fn a_basis_within_the_budget_is_computed_as_before() {
    use symplex::multipoly::MonomialOrder;
    let ctx = Context::new();
    let (polys, vars) = cyclic(&ctx, 5);
    let grevlex = Ex::groebner(&polys, &vars, MonomialOrder::GrevLex).unwrap();
    assert_eq!(grevlex.len(), 20);
    for p in &polys {
        let r = p
            .reduce_modulo(&grevlex, &vars, MonomialOrder::GrevLex)
            .unwrap();
        assert!(
            r.is_zero_structural(),
            "{p} is not in the ideal of the basis"
        );
    }
    let lex = Ex::groebner(&polys, &vars, MonomialOrder::Lex).unwrap();
    assert_eq!(lex.len(), 11);
    // The work is counted, not timed: the same call gives the same basis.
    assert_eq!(
        Ex::groebner(&polys, &vars, MonomialOrder::Lex).unwrap(),
        lex
    );
}

/// A work budget small enough for a debug build: cyclic-6 is refused after 200,000 of the
/// 100,000,000 default units (a fraction of a second), through the same code.
const SMALL_BUDGET: u64 = 200_000;

/// Is `r` the refusal of a Gröbner basis?
fn budget_refusal<T>(r: &Result<T, SymplexError>) -> bool {
    matches!(
        r,
        Err(SymplexError::BudgetExhausted {
            kind: symplex::BudgetKind::GroebnerBasis,
            ..
        })
    )
}

/// H02.  Up to 0.60.0 `Ex::groebner` of cyclic-6 did not return (killed after 20 s, cyclic-7
/// after 300 s) and could not be stopped: Buchberger's loop had no budget.  Now the call is
/// refused with `BudgetExhausted { kind: GroebnerBasis }` when its work budget is spent, in
/// either monomial order; the budget counts work, so the outcome is the same on every run.
/// Run under a small budget (the hidden test hook `with_groebner_work_cap`); the default one
/// is `the_default_budget_refuses_cyclic_6`.
///
/// Oracle: none needed for the refusal (the test asks that the call returns, and what kind
/// of error it is).  For scale: SymPy 1.14 `groebner(cyclic6, order='grevlex')` returns 45
/// polynomials after 43 s (m5).
#[test]
fn a_basis_beyond_the_budget_is_refused() {
    use symplex::__macro_support::with_groebner_work_cap;
    use symplex::multipoly::MonomialOrder;
    let outcome = answered_within(30, || {
        let ctx = Context::new();
        let (polys, vars) = cyclic(&ctx, 6);
        let refused = with_groebner_work_cap(SMALL_BUDGET, || {
            [
                MonomialOrder::GrevLex,
                MonomialOrder::Lex,
                MonomialOrder::GrevLex,
            ]
            .map(|order| budget_refusal(&Ex::groebner(&polys, &vars, order)))
        });
        // The same budget is enough for cyclic-4 (under 15,000 units).
        let (small, vars4) = cyclic(&ctx, 4);
        let within = with_groebner_work_cap(SMALL_BUDGET, || {
            Ex::groebner(&small, &vars4, MonomialOrder::GrevLex).map(|b| b.len())
        });
        (refused, within.ok())
    });
    assert_eq!(outcome, Some(([true, true, true], Some(7))));
}

/// H02, the callers.  `solve_system_ex` (behind `Context::solve_system`) and the plain
/// `solve_polynomial_system` reach the same loop: up to 0.60.0 they did not return on
/// cyclic-6 either.  They now return the budget's error (`Err(String)` for the plain one)
/// — never a solution list from a basis that was not finished.
///
/// Oracle: none needed (a refusal); the system has 156 solutions (Björck and Fröberg 1991),
/// so an `Ok` with fewer would be a wrong answer.
#[test]
fn a_system_whose_basis_is_refused_is_not_solved() {
    use symplex::__macro_support::with_groebner_work_cap;
    use symplex::polysys::{solve_polynomial_system, solve_system_ex};
    let outcome = answered_within(30, || {
        let ctx = Context::new();
        let (polys, vars) = cyclic(&ctx, 6);
        let refs: Vec<&Ex> = vars.iter().collect();
        let exact: Vec<_> = polys
            .iter()
            .map(|p| Poly::new(p, &refs).unwrap().to_multipoly().unwrap())
            .collect();
        with_groebner_work_cap(SMALL_BUDGET, || {
            let symbolic = budget_refusal(&solve_system_ex(&polys, &vars));
            let plain = solve_polynomial_system(&exact)
                .err()
                .is_some_and(|msg| msg.contains("work budget"));
            (symbolic, plain)
        })
    });
    assert_eq!(outcome, Some((true, true)));
}

/// H02, the plain forms.  `groebner::groebner_basis` and `groebner_basis_lex` return a `Vec`
/// and have no error to give: when the budget is spent they return the generators as they
/// were given — the same ideal, not a Gröbner basis, as `is_groebner_basis` says — and
/// `try_groebner_basis` / `try_groebner_basis_lex` (new in 0.60.1) return the error.  Up to
/// 0.60.0 the calls did not return.
///
/// Oracle: the definition (the generators generate their ideal); Buchberger's criterion
/// for "not a basis".
#[test]
fn the_plain_basis_function_returns_the_generators_when_refused() {
    use symplex::__macro_support::with_groebner_work_cap;
    use symplex::groebner::{
        groebner_basis, groebner_basis_lex, is_groebner_basis, try_groebner_basis,
        try_groebner_basis_lex,
    };
    let outcome = answered_within(30, || {
        let ctx = Context::new();
        let (polys, vars) = cyclic(&ctx, 6);
        let refs: Vec<&Ex> = vars.iter().collect();
        let gens: Vec<_> = polys
            .iter()
            .map(|p| Poly::new(p, &refs).unwrap().to_multipoly().unwrap())
            .collect();
        with_groebner_work_cap(SMALL_BUDGET, || {
            let plain = groebner_basis(&gens);
            (
                plain == gens,
                is_groebner_basis(&plain),
                groebner_basis_lex(&gens).len() == gens.len(),
                budget_refusal(&try_groebner_basis(&gens)),
                budget_refusal(&try_groebner_basis_lex(&gens)),
            )
        })
    });
    assert_eq!(outcome, Some((true, false, true, true, true)));
}

/// H02, the default budget (100,000,000 units): cyclic-6 is refused after it, and the call
/// returns.  The one test that spends it; the refusal itself is tested under a small budget
/// by the three tests above and by the unit tests of `src/poly/groebner.rs`.
///
/// Oracle: none needed (a refusal).
#[test]
#[ignore = "about 90 s in a debug build (5.5 s in release): spends the default work budget of a Gröbner basis"]
fn the_default_budget_refuses_cyclic_6() {
    use symplex::multipoly::MonomialOrder;
    let outcome = answered_within(900, || {
        let ctx = Context::new();
        let (polys, vars) = cyclic(&ctx, 6);
        budget_refusal(&Ex::groebner(&polys, &vars, MonomialOrder::GrevLex))
    });
    assert_eq!(outcome, Some(true));
}

// ── W31: `Poly` and coefficients that are zero by an identity ───────────────

/// W31.  Up to 0.60.0 a difference of two polynomials whose coefficients cancel by an
/// identity was not the zero polynomial: `Poly(sin(a)²·x) + Poly(cos(a)²·x) − Poly(x)` had
/// degree 1.  Every arithmetic result is normalised by the certified zero test.
///
/// Oracle: `sin²a + cos²a = 1` (SymPy `simplify(sin(a)**2 + cos(a)**2 - 1)` = 0).
#[test]
fn poly_arithmetic_drops_coefficients_proved_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = |src: &str| Poly::try_new(&ctx.parse(src).unwrap(), &[&x]).unwrap();
    let sum = poly("sin(a)^2*x").add(&poly("cos(a)^2*x")).unwrap();
    let difference = sum.sub(&poly("x")).unwrap();
    assert!(difference.is_zero(), "{difference}");
    assert_eq!(difference.total_degree(), None);
    assert_eq!(difference.leading_coeff(), ctx.int(0));
    // The leading term goes, the rest stays: degree 1 with the coefficient 3.
    let p = poly("(sin(a)^2 + cos(a)^2 - 1)*x^4 + 3*x + (log(6) - log(2) - log(3))");
    assert_eq!(p.total_degree(), Some(1));
    assert_eq!(p.leading_coeff(), ctx.int(3));
    assert_eq!(p.num_terms(), 1);
}

/// W31, the other side: only a proof drops a term.  A coefficient that looks like one of the
/// identities and is not zero is kept (`sin²a + cos²a − 2 = −1`, `π − 22/7`), and so is one
/// the zero test cannot decide — `sin(y·J₋₁⸝₂(π/2))` is 0 because `J₋₁⸝₂(π/2) = 0`, which
/// `Ex::is_zero` does not prove: the polynomial is reported with degree 1, as documented
/// ("the coefficients as proved").
///
/// Oracle: mpmath at 50 digits: `sin(1)**2 + cos(1)**2 - 2` = −1, `pi - mpf(22)/7` =
/// −0.00126…, `besselj(-0.5, pi/2)` = 0 (below 10⁻⁵⁰).
#[test]
fn poly_keeps_coefficients_that_are_not_proved_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["(sin(a)^2 + cos(a)^2 - 2)*x", "(pi - 22/7)*x"] {
        let p = Poly::try_new(&ctx.parse(src).unwrap(), &[&x]).unwrap();
        assert_eq!(p.total_degree(), Some(1), "{src}");
        assert!(!p.is_zero(), "{src}");
    }
    let undecided = ctx.parse("sin(y*besselj(-1/2, pi/2))").unwrap();
    assert_eq!(undecided.is_zero(), None);
    let p = Poly::try_new(&(&undecided * &x), &[&x]).unwrap();
    assert_eq!(p.total_degree(), Some(1));
}

// ── W19: `mod_pow` with a negative exponent ─────────────────────────────────

/// W19.  Up to 0.60.0 every negative exponent gave the stand-in `0`.  Now it is a power of
/// the modular inverse where the inverse exists; where it does not (and for a modulus ≤ 0)
/// the documented `0` stays — for a modulus above 1 it is no unit, hence no such power.
///
/// Oracle: Python 3 `pow(b, e, m)`: `pow(2, -1, 7)` = 4, `pow(3, -2, 7)` = 4,
/// `pow(10, -3, 17)` = 11, `pow(-3, -1, 7)` = 2, `pow(5, -1, 1)` = 0, `pow(1, -5, 2)` = 1,
/// `pow(123456789, -987654321, 1000000007)` = 388173166, `pow(7, -1, 10**18 + 9)` =
/// 285714285714285717, `pow(2, -64, 3**40)` = 3997565229372176830; `pow(2, -1, 4)`,
/// `pow(0, -1, 5)` and `pow(6, -1, 9)` raise "base is not invertible for the given modulus".
#[test]
fn mod_pow_of_a_negative_exponent_is_a_power_of_the_inverse() {
    use symplex::ntheory::mod_pow;
    let big = |s: &str| s.parse::<BigInt>().unwrap();
    let cases: [(BigInt, i64, BigInt, &str); 9] = [
        (2.into(), -1, 7.into(), "4"),
        (3.into(), -2, 7.into(), "4"),
        (10.into(), -3, 17.into(), "11"),
        ((-3).into(), -1, 7.into(), "2"),
        (5.into(), -1, 1.into(), "0"),
        (1.into(), -5, 2.into(), "1"),
        (
            123_456_789.into(),
            -987_654_321,
            1_000_000_007.into(),
            "388173166",
        ),
        (
            7.into(),
            -1,
            big("1000000000000000009"),
            "285714285714285717",
        ),
        (
            2.into(),
            -64,
            big("12157665459056928801"),
            "3997565229372176830",
        ),
    ];
    for (b, e, m, want) in cases {
        assert_eq!(
            mod_pow(b.clone(), e, m.clone()),
            big(want),
            "pow({b}, {e}, {m})"
        );
    }
    // No inverse: the documented stand-in.
    for (b, m) in [(2, 4), (0, 5), (6, 9)] {
        assert_eq!(mod_pow(b, -1, m), BigInt::from(0), "pow({b}, -1, {m})");
    }
    // Non-negative exponents and the non-positive modulus are as before.
    assert_eq!(mod_pow(2, 10, 1000), BigInt::from(24));
    assert_eq!(mod_pow(2, 10, -7), BigInt::from(0));
}

// ── W35: the compiled `fibonacci` and `lucas` off the integers ──────────────

/// `|got − want|` in units of `scale` (at least `|want|`).
fn scaled_error(got: f64, want: f64, scale: f64) -> f64 {
    if got == want {
        return 0.0;
    }
    (got - want).abs() / scale.max(want.abs())
}

/// The scale of the error of `F(x)` and `L(x)` beyond the value itself: on the negative
/// axis, off the half-integers (where the cosine is 0 exactly), the two terms `φˣ` and
/// `cos(πx)·φ⁻ˣ` cancel at the zeros, and the error is in units of the larger one.
/// Elsewhere the error is relative to the value (0).
fn fibonacci_scale(x: f64) -> f64 {
    if x < 0.0 && (2.0 * x).fract() != 0.0 {
        1.618_033_988_749_895_f64.powf(-x)
    } else {
        0.0
    }
}

/// `(x, F(x), L(x))`: `mpmath.fibonacci(mpf(x))` and `fibonacci(x - 1) + fibonacci(x + 1)`
/// (`= φˣ + cos(πx)·φ⁻ˣ`, since `φ + 1/φ = √5`), mpmath 1.3 at 50 and at 100 digits, the
/// same double at both.  Integers, half-integers, negative and large arguments.
const FIBONACCI_POINTS: [(f64, f64, f64); 20] = [
    (0.5, 0.568_864_481_005_783_1, 1.272_019_649_514_069),
    (2.5, 1.489_306_546_265_709_1, 3.330_190_676_785_561_4),
    (-0.5, 0.351_577_584_254_142_9, 0.786_151_377_757_423_3),
    (-7.5, 0.012_108_983_738_178_897, 0.027_076_510_777_007_53),
    (30.5, 1_058_371.229_181_991_7, 2_366_590.013_880_942_5),
    (
        -30.5,
        1.889_696_114_987_731_8e-7,
        4.225_488_969_929_827_6e-7,
    ),
    (0.1, 0.063_917_353_968_524_66, 1.955_672_059_899_691_5),
    (-0.1, -0.020_090_276_188_140_01, 1.950_959_805_113_163_6),
    (3.7, 2.608_887_868_756_674_8, 6.031_800_773_226_733),
    (-3.7, -1.484_128_441_526_466_2, 3.655_725_258_263_917),
    (10.25, 62.032_937_970_940_296, 138.720_061_251_635_06),
    (-10.25, -43.862_299_108_602_8, 98.093_500_511_437_74),
    (100.25, 3.995_084_163_188_156e20, 8.933_279_764_721_579e20),
    (
        -100.25,
        -2.824_951_103_201_328_5e20,
        6.316_782_699_871_194e20,
    ),
    (
        700.75,
        1.254_884_927_530_878_5e146,
        2.806_008_001_898_941_7e146,
    ),
    (
        -700.75,
        8.873_376_418_658_734e145,
        -1.984_147_286_206_456_3e146,
    ),
    (
        -1400.5,
        9.190_197_143_147_062e-294,
        2.054_990_553_870_119_8e-293,
    ),
    // Integers, exact as before (SymPy `fibonacci(n)`, `lucas(n)`).
    (10.0, 55.0, 123.0),
    (-6.0, -8.0, 18.0),
    (70.0, 190_392_490_709_135.0, 425_730_551_631_123.0),
];

/// W35.  Up to 0.60.0 `compile()` gave NaN for `fibonacci(x)` and `lucas(x)` at every
/// non-integer `x`, where `eval_f64` gives `(φˣ ∓ cos(πx)·φ⁻ˣ)` (over `√5` for `F`).  The
/// `f64` kernel now computes that continuation; integers are exact as before.
///
/// Oracle: mpmath (see `FIBONACCI_POINTS`), and `eval_f64` of the same argument.
#[test]
fn compiled_fibonacci_and_lucas_off_the_integers() {
    let ctx = Context::new();
    let fib = ctx.parse("fibonacci(x)").unwrap().compile(&["x"]).unwrap();
    let luc = ctx.parse("lucas(x)").unwrap().compile(&["x"]).unwrap();
    for (x, f, l) in FIBONACCI_POINTS {
        let scale = fibonacci_scale(x);
        assert!(
            scaled_error(fib.call(&[x]), f, scale) < 4e-15,
            "F({x}) = {}",
            fib.call(&[x])
        );
        assert!(
            scaled_error(luc.call(&[x]), l, scale) < 4e-15,
            "L({x}) = {}",
            luc.call(&[x])
        );
        if x.fract() == 0.0 {
            assert_eq!(
                (fib.call(&[x]), luc.call(&[x])),
                (f, l),
                "exact at the integer {x}"
            );
        }
    }
    // Against `eval_f64` at the same (exactly representable) arguments.
    for src in ["1/2", "5/2", "-1/2", "-15/2", "37/10", "-41/4", "401/4"] {
        let x = ctx.parse(src).unwrap();
        let at = x.eval_f64().unwrap();
        for (name, f) in [("fibonacci", &fib), ("lucas", &luc)] {
            let exact = ctx
                .parse(&format!("{name}({src})"))
                .unwrap()
                .eval_f64()
                .unwrap();
            let scale = fibonacci_scale(at);
            assert!(
                scaled_error(f.call(&[at]), exact, scale) < 4e-15,
                "{name}({src})"
            );
        }
    }
    // At the top of the range no intermediate overflows before the value does: mpmath
    // `fibonacci(mpf("1475.5"))` = 1.0274913789757015e308 (`φˣ` alone is beyond the range).
    assert!(scaled_error(fib.call(&[1475.5]), 1.027_491_378_975_701_5e308, 0.0) < 4e-15);
    // Beyond the range: the limit, as for the integers; not a number stays not a number.
    // (The integer `lucas` was NaN there up to 0.60.0: `2·F(n+1) − F(n)` as `∞ − ∞`.)
    assert_eq!(fib.call(&[1600.5]), f64::INFINITY);
    assert_eq!(fib.call(&[-1600.5]), 0.0);
    assert_eq!(luc.call(&[1504.0]), f64::INFINITY);
    assert_eq!(luc.call(&[-1531.0]), f64::NEG_INFINITY);
    assert_eq!(luc.call(&[1600.25]), f64::INFINITY);
    assert!(fib.call(&[f64::NAN]).is_nan() && fib.call(&[f64::NEG_INFINITY]).is_nan());
}

/// Compile and run generated code that defines `f(x)`; `None` when the tool is missing.
fn run_generated(tool: &str, code: &str, xs: &[f64], tag: &str) -> Option<Vec<f64>> {
    use std::process::Command;
    if !Command::new(tool)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        eprintln!("{tool} not available; skipping");
        return None;
    }
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("v36_s18e_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let list: Vec<String> = xs.iter().map(|x| format!("{x:e}")).collect();
    let built = if tool == "cc" {
        let src = format!(
            "{code}\n#include <stdio.h>\nstatic const double XS[] = {{{}}};\nint main(void) {{\n    for (unsigned i = 0; i < {}; i++) printf(\"%.17g\\n\", f(XS[i]));\n    return 0;\n}}\n",
            list.join(", "),
            xs.len()
        );
        std::fs::write(dir.join("f.c"), src).unwrap();
        Command::new("cc")
            .current_dir(&dir)
            .args(["-O1", "f.c", "-lm", "-o", "f"])
            .output()
            .unwrap()
    } else {
        let src = format!(
            "#![allow(dead_code, unused_parens)]\n{code}\nconst XS: [f64; {}] = [{}];\nfn main() {{\n    for x in XS {{\n        println!(\"{{:e}}\", f(x));\n    }}\n}}\n",
            xs.len(),
            list.join(", ")
        );
        std::fs::write(dir.join("f.rs"), src).unwrap();
        Command::new("rustc")
            .current_dir(&dir)
            .args(["--edition", "2021", "-o", "f", "f.rs"])
            .output()
            .unwrap()
    };
    assert!(
        built.status.success(),
        "{}\n{code}",
        String::from_utf8_lossy(&built.stderr)
    );
    let run = Command::new(dir.join("f")).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(
        String::from_utf8_lossy(&run.stdout)
            .lines()
            .map(|l| {
                l.trim()
                    .replace("nan", "NaN")
                    .replace("-NaN", "NaN")
                    .parse()
                    .unwrap_or(f64::NAN)
            })
            .collect(),
    )
}

/// W35, the generated C and Rust.  The kernel exists in two copies — the shared Rust runtime
/// (`compile`, and embedded by `to_rust_fn`) and its C port — and both were NaN off the
/// integers up to 0.60.0.  Each must give the value of `FIBONACCI_POINTS`.  Skipped (with a
/// note) where `cc` or `rustc` is missing.
///
/// Oracle: mpmath (see `FIBONACCI_POINTS`).
#[test]
fn generated_c_and_rust_fibonacci_off_the_integers() {
    let ctx = Context::new();
    let xs: Vec<f64> = FIBONACCI_POINTS.iter().map(|p| p.0).collect();
    for (name, column) in [("fibonacci", 1), ("lucas", 2)] {
        let e = ctx.parse(&format!("{name}(x)")).unwrap();
        let sources = [
            ("cc", e.to_c_fn("f", &["x"]).unwrap()),
            ("rustc", e.to_rust_fn("f", &["x"]).unwrap()),
        ];
        for (tool, code) in sources {
            let Some(got) = run_generated(tool, &code, &xs, &format!("{name}_{tool}")) else {
                continue;
            };
            assert_eq!(got.len(), xs.len(), "{tool} {name}");
            for (g, p) in got.iter().zip(FIBONACCI_POINTS) {
                let want = if column == 1 { p.1 } else { p.2 };
                let scale = fibonacci_scale(p.0);
                assert!(
                    scaled_error(*g, want, scale) < 4e-15,
                    "{tool} {name}({}) = {g}",
                    p.0
                );
            }
        }
    }
}

// ── W27: `args()` and the sorts ─────────────────────────────────────────────

/// W27.  Up to 0.60.0 `args()` wrapped every child in the type of its parent: the sides of
/// `x > 1` were `BoolEx` (`args()[0].not()` built `!x`).  Now a `BoolEx` returns the
/// operands of `And`, `Or` and `Not`; a relation is an atom (`[]`), and its sides are read
/// as numbers from the untyped view.
///
/// Oracle: the type rule; `atoms()` already treated a relation as an atom.
#[test]
fn args_of_a_boolean_expression_are_boolean() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (p, q) = (x.gt(&ctx.int(1)), x.lt(&ctx.int(5)));
    let operands = p.and(&q).args();
    assert_eq!(operands.len(), 2);
    assert!(operands.contains(&p) && operands.contains(&q));
    assert_eq!(p.or(&q).args().len(), 2);
    for atom in [
        &p,
        &q,
        &x.eq_expr(&ctx.int(2)),
        &x.ne_expr(&ctx.int(2)),
        &ctx.bool_true(),
    ] {
        assert!(atom.args().is_empty(), "{atom}");
    }
    assert_eq!(p.as_ex().args(), vec![x.clone(), ctx.int(1)]);
    // Every handle reachable through `args()` is a Boolean expression.
    let nested = p.and(&q).or(&p.not());
    let mut stack = vec![nested];
    while let Some(b) = stack.pop() {
        for child in b.args() {
            let any = ctx.from_json_any(&child.to_json().unwrap()).unwrap();
            assert!(matches!(any, AnyExpr::Bool(_)), "{child}");
            stack.push(child);
        }
    }
}

/// W27, the same hole in the third sort.  Up to 0.60.0 the endpoints of an interval and the
/// elements of a finite set came out of `SetEx::args()` as sets (`args()[0].union(…)` took
/// the number `0` for a set).  A `SetEx` returns the operands of a union, an intersection
/// and a complement; the numbers are read from `as_ex().args()`.  `Ex`, the untyped handle,
/// returns every child as before.
///
/// Oracle: the type rule.
#[test]
fn args_of_a_set_expression_are_sets() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let a = ctx.interval(&x, &ctx.int(1), IntervalKind::Closed);
    let b = ctx.finite_set(&[y.clone(), ctx.int(7)]);
    assert!(a.args().is_empty());
    assert!(b.args().is_empty());
    assert_eq!(a.as_ex().args(), vec![x.clone(), ctx.int(1)]);
    assert_eq!(b.as_ex().args().len(), 2);
    let u = a.union(&b);
    let parts = u.args();
    assert_eq!(parts.len(), 2);
    assert!(parts.contains(&a) && parts.contains(&b));
    // The untyped handle is unchanged.
    assert_eq!((&x + 1).args().len(), 2);
    assert_eq!(x.sin().args(), vec![x.clone()]);
}

// ── W03: `Frequency::from_angular_velocity` ─────────────────────────────────

/// W03.  Up to 0.60.0 `Frequency::from_angular_velocity` re-tagged the value: `2π rad/s`
/// became `2π Hz`.  It divides by `2π` now: every rotation speed gives the same frequency
/// whether it is entered in rpm as a frequency or as an angular velocity, and `ω = 2πf`
/// gives the angular velocity back.
///
/// Oracle: the definition `f = ω/2π`; 1 rpm = 1/60 Hz = 2π/60 rad/s.
#[test]
fn frequency_of_an_angular_velocity_is_omega_over_two_pi() {
    use symplex::units::{AngularVelocity, Frequency};
    let ctx = Context::new();
    for (n, d) in [(60, 1), (1, 1), (3000, 1), (-45, 2), (0, 1), (7, 3)] {
        let r = ctx.rational(n, d);
        let f = Frequency::from_angular_velocity(AngularVelocity::rpm(&r));
        assert_eq!(
            f.inner().eval(),
            Frequency::rpm(&r).inner().eval(),
            "{r} rpm"
        );
    }
    let omega = ctx.symbol("omega");
    let f = Frequency::from_angular_velocity(AngularVelocity::radians_per_second(&omega));
    assert_eq!((f.inner() * &(2 * &ctx.pi())).eval(), omega);
    // 10 rad/s is 5/π Hz (1.5915494309189535).
    let ten = Frequency::from_angular_velocity(AngularVelocity::radians_per_second(&ctx.int(10)));
    assert!((ten.inner().eval_f64().unwrap() - 5.0 / std::f64::consts::PI).abs() < 1e-15);
}

/// W03, the same conversion through `From`.  Up to 0.60.0 `AngularVelocity::from(f)` and
/// `Frequency::from(w)` kept the value: 50 Hz became 50 rad/s and back.  Both follow
/// `ω = 2πf` now, and agree with `Frequency::from_angular_velocity` and with the two `rpm`
/// constructors.
///
/// Oracle: the definition: 50 Hz is 100π rad/s; 60 rpm is 1 Hz and 2π rad/s.
#[test]
fn frequency_and_angular_velocity_convert_by_two_pi() {
    use symplex::units::{AngularVelocity, Frequency};
    let ctx = Context::new();
    let w: AngularVelocity = Frequency::hertz(&ctx.int(50)).into();
    assert_eq!(w.inner().eval(), (ctx.int(100) * ctx.pi()).eval());
    let back: Frequency = w.into();
    assert_eq!(back.inner().eval(), ctx.int(50));
    for (n, d) in [(60, 1), (1, 1), (-45, 2), (0, 1), (7, 3)] {
        let r = ctx.rational(n, d);
        // The same rotation, entered in rpm in either unit.
        let f = Frequency::from(AngularVelocity::rpm(&r));
        assert_eq!(
            f.inner().eval(),
            Frequency::rpm(&r).inner().eval(),
            "{r} rpm"
        );
        let w = AngularVelocity::from(Frequency::rpm(&r));
        assert_eq!(
            w.inner().eval(),
            AngularVelocity::rpm(&r).inner().eval(),
            "{r} rpm"
        );
    }
    // A symbolic frequency: 2πf, and the round trip is the identity.
    let f = ctx.symbol("f");
    let w = AngularVelocity::from(Frequency::hertz(&f));
    assert_eq!(w.inner().eval(), (ctx.int(2) * ctx.pi() * &f).eval());
    assert_eq!(Frequency::from(w).inner().eval(), f);
}
