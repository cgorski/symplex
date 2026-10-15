//! Tests of session 15 (s15h5): float evidence outside calculus: solve, sets, the API layer, roots.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn real_x(ctx: &Context) -> Ex {
    ctx.symbol_with("x", &[Assumption::Real]).unwrap()
}

/// `(x − e⁻³⁰).check_solution(x, 0)` was `Some(true)`: a residual below
/// `10⁻¹⁰` counted as zero.  Oracle: `mpmath.exp(-30) = 9.3576229688401746e-14`,
/// certified nonzero, so `0` is not a solution.
#[test]
fn check_solution_needs_a_proof_of_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("x - exp(-30)").unwrap();
    assert_eq!(f.check_solution(&x, &ctx.int(0)), Some(false));
    let g = ctx.parse("x - sqrt(2)*sqrt(3) + sqrt(6)").unwrap();
    assert_eq!(g.check_solution(&x, &ctx.int(0)), Some(true));
}

/// `(2 + i·e⁻³⁰).compare_numeric(1)` was `Some(Greater)` (and
/// `is_greater_than` `Some(true)`): an imaginary part below `10⁻⁹` of the
/// value was ignored.  Oracle: the imaginary part `e⁻³⁰ ≈ 9.36·10⁻¹⁴` is not 0,
/// and complex numbers are unordered.
#[test]
fn complex_constants_are_not_ordered() {
    let ctx = Context::new();
    let a = ctx.parse("2 + I*exp(-30)").unwrap();
    assert_eq!(a.compare_numeric(&ctx.int(1)), None);
    assert_eq!(a.is_greater_than(&ctx.int(1)), None);
    assert_eq!(
        ctx.pi().compare_numeric(&ctx.int(3)),
        Some(std::cmp::Ordering::Greater)
    );
}

/// `nroots((x − 1)((x − 1)² + 10⁻¹⁴))` was `1, 1, 1`: a computed imaginary
/// part below `10⁻⁶` was dropped when a real root lay within `2⁻³⁰`.  Exact
/// roots: `1` and `1 ± 10⁻⁷i`.  `x² − 2·10⁻³⁵x + 1` had real parts `0`
/// (exact roots `10⁻³⁵ ± i√(1 − 10⁻⁷⁰)`).
#[test]
fn nroots_keep_small_imaginary_and_real_parts() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = ctx.parse("(x - 1)*((x - 1)^2 + 10^(-14))").unwrap();
    let r = p.nroots(&x, 15).unwrap();
    assert_eq!(r.len(), 3);
    assert_eq!(r.iter().filter(|z| z.im == 0.0).count(), 1, "{r:?}");
    assert!(r.iter().all(|z| (z.re - 1.0).abs() < 1e-14), "{r:?}");
    assert!(
        r.iter().any(|z| (z.im - 1e-7).abs() < 1e-20 && z.im > 0.0),
        "{r:?}"
    );
    let q = ctx.parse("x^2 - 2*10^(-35)*x + 1").unwrap();
    let r = q.nroots(&x, 15).unwrap();
    assert!(r.iter().all(|z| (z.re - 1e-35).abs() < 1e-49), "{r:?}");
    // a real part 0 within its certified radius is shown as 0
    let s = ctx.parse("(x^2 + 1)*(x - 10^(-40))").unwrap();
    let r = s.nroots(&x, 15).unwrap();
    assert_eq!(r.iter().filter(|z| z.re == 0.0).count(), 2, "{r:?}");
}

/// `solve_as_set(sin(x)/(x − 2·10¹³π))` was the family `{nπ}` with its pole
/// `2·10¹³π` (member `n = 2·10¹³`, skipped beyond `10¹²`).  Exact: at
/// `x = 2·10¹³π` the denominator is 0.
#[test]
fn far_pole_on_a_family_is_excluded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("sin(x)/(x - 2*10^13*pi)").unwrap();
    let s = f.solve_as_set(&x).to_string();
    assert!(s.contains("\\ {20000000000000*pi}"), "{s}");
}

/// `tan(sin x − 10²⁰) = 1` for a real `x` was "no solution" (`solve_as_set`:
/// `EmptySet`): the member `10²⁰ + π/4 + kπ` of `[−1, 1]` has `k ≈
/// −3.18·10¹⁹`, beyond the `f64` index range.  Oracle (mpmath, 60 digits):
/// `k = −31830988618379067154`, member `0.084046005682102927421`.
#[test]
fn sine_equal_to_a_far_family_member() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let f = ctx
        .parse("tan(sin(x) - 10^20) - 1")
        .unwrap()
        .subs(&ctx.symbol("x"), &x);
    let s = f.solve_as_set(&x);
    assert_eq!(s.is_empty(), Some(false), "{s}");
    let c = ctx
        .parse("-127323954473516268615/4*pi + 100000000000000000000")
        .unwrap();
    assert!(
        s.to_string()
            .contains("asin(-127323954473516268615/4*pi + 100000000000000000000)")
    );
    assert!(
        c.eval_decimal(20)
            .unwrap()
            .starts_with("0.08404600568210292742")
    );
}

/// `solve_as_set(sin(x − 10²⁰)) ∩ [0, 10]` was `EmptySet`, and
/// `reduce_inequalities([sin(x − 10²⁰) = 0, x ≥ 0, x ≤ 10])` too.  Oracle
/// (mpmath, 60 digits): `2.4402404958744478563`, `5.5818331494642410947`,
/// `8.7234258030540343332`.
#[test]
fn members_of_a_far_family_in_a_window() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let f = ctx
        .parse("sin(x - 10^20)")
        .unwrap()
        .subs(&ctx.symbol("x"), &x);
    let window = ctx.interval(&ctx.int(0), &ctx.int(10), IntervalKind::Closed);
    let expected = [
        2.440_240_495_874_448,
        5.581_833_149_464_241,
        8.723_425_803_054_034,
    ];
    let check = |set: &SetEx| {
        let mut v: Vec<f64> = set
            .as_finite_set()
            .unwrap_or_else(|| panic!("{set}"))
            .iter()
            .map(|p| p.eval_f64().unwrap())
            .collect();
        v.sort_by(f64::total_cmp);
        assert_eq!(v.len(), 3, "{set}");
        for (a, b) in v.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12, "{set}");
        }
    };
    check(&f.solve_as_set(&x).intersection(&window).simplify());
    let conds = [
        f.eq_expr(&ctx.int(0)),
        x.ge(&ctx.int(0)),
        x.le(&ctx.int(10)),
    ];
    check(&SetEx::reduce_inequalities(&conds, &x).unwrap());
}

/// `reduce_inequalities(sin(x − 10²⁰) > 0)` was `(−∞, ∞)`: no zero was
/// found in a period.  Exact: `sin(x − 10²⁰) = cos(kπ) = 1` at `x = 10²⁰ −
/// kπ + π/2` and `−1` at `10²⁰ − kπ − π/2`, `k = 31830988618379067152` even.
#[test]
fn periodic_inequality_far_from_the_origin() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let f = ctx
        .parse("sin(x - 10^20)")
        .unwrap()
        .subs(&ctx.symbol("x"), &x);
    let s = SetEx::reduce_inequalities(&[f.gt(&ctx.int(0))], &x).unwrap();
    assert_ne!(s.to_string(), "(-oo, oo)");
    let p_in = ctx.parse("10^20 - 31830988618379067152*pi + pi/2").unwrap();
    let p_out = ctx.parse("10^20 - 31830988618379067152*pi - pi/2").unwrap();
    assert_eq!(s.contains(&p_in), Some(true), "{s}");
    assert_eq!(s.contains(&p_out), Some(false), "{s}");
}

/// `sin x ≥ 1 + e⁻⁷⁰` was refused ("endpoint `asin(e⁻⁷⁰ + 1)` cannot be
/// ordered"): the zero's imaginary part (`≈ 6.4·10⁻¹⁶`) was below the
/// tolerance taken for 0.  Oracle: `|sin x| ≤ 1` on the reals, so `EmptySet`.
#[test]
fn zeros_with_a_tiny_imaginary_part_are_not_real() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let f = ctx
        .parse("sin(x) - 1 - exp(-70)")
        .unwrap()
        .subs(&ctx.symbol("x"), &x);
    let s = SetEx::reduce_inequalities(&[f.ge(&ctx.int(0))], &x).unwrap();
    assert_eq!(s.to_string(), "EmptySet");
}

/// `(x − 1)(x − 1 − e⁻⁷⁰) < 0` was `EmptySet`: the two critical points,
/// `e⁻⁷⁰ ≈ 4·10⁻³¹` apart, were merged.  Exact: the product is negative on
/// `(1, 1 + e⁻⁷⁰)`; refused now (the points are certified distinct, but too
/// close for the sample points of the sign chart).
#[test]
fn close_critical_points_are_not_merged() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let f = ctx
        .parse("(x - 1)*(x - 1 - exp(-70))")
        .unwrap()
        .subs(&ctx.symbol("x"), &x);
    match SetEx::reduce_inequalities(&[f.lt(&ctx.int(0))], &x) {
        Ok(s) => assert_ne!(s.to_string(), "EmptySet"),
        Err(e) => assert!(e.to_string().contains("too close"), "{e}"),
    }
}

/// `solve_as_set((4 − x)·x^(y + i))` was the `ConditionSet` of the equation:
/// a product with a power of the unknown whose exponent is symbolic was
/// solved as a whole.  Exact: at `x = 4` the power `4^(y + i)` is defined for
/// every `y`; at `x = 0`, `0^(y + i) = 0` exactly when `re y > 0` (otherwise
/// undefined).
#[test]
fn product_with_a_symbolic_power_of_the_unknown() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("(4 - x)*x^(y + I)").unwrap();
    assert_eq!(
        f.solve_as_set(&x).to_string(),
        "{4} ∪ (ConditionSet(x, re(y) > 0) ∩ {0})"
    );
    // still not combined where the power is undefined at another root
    let g = ctx.parse("x*(x^y)^(1/3)").unwrap();
    assert!(
        g.solve_as_set(&x)
            .to_string()
            .starts_with("ConditionSet(x, ")
    );
}
