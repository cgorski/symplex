//! Tests of the 0.49 refactor (w3): explicit domains of unknowns in the solvers.
//! Every test says what was wrong before and cites its oracle.
//!
//! Oracles: mpmath 1.3 (`mp.dps = 30`), SymPy 1.14 `solveset`, and exact
//! identities checked by substituting the root back.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn real(ctx: &Context, name: &str) -> Ex {
    ctx.symbol_with(name, &[Assumption::Real]).unwrap()
}

fn is_no_solution<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::NoSolution { .. }))
}

fn is_refusal<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::ComputationFailed { .. }))
}

/// `e` vanishes at `x = r` after exact evaluation.
fn assert_exact_root(e: &Ex, x: &Ex, r: &Ex) {
    let at = e.subs(x, r).eval();
    assert!(at.is_zero_structural(), "{e} at {x} = {r} is {at}");
}

/// `erf(x) = c` and `erfc(x) = c` with `c` outside the real range of the
/// function, for a plain (complex) `x`: up to 0.48 `Err(NoSolution("equation
/// has no real solutions …"))`, the erf arm alone taking a plain `x` real
/// under `Ex::solve`.  Roots exist (mpmath `findroot(lambda z: erf(z) - 2,
/// 1+1j)` = `0.869417565326184121 + 1.340594818530277116j`; `erfc(z) = 3`
/// at its negative, `erfc(z) = −1` at the same point as `erf(z) = 2`), but
/// the inverses have no principal value there: the solver now refuses.
#[test]
fn erf_out_of_range_for_a_complex_unknown_is_refused_not_no_solution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "erf(x) - 2",
        "erfc(x) - 3",
        "erfc(x) + 1",
        "erf(x) + 1",
        "erf(x)**2 - 4",
    ] {
        let r = parse(&ctx, src).solve(&x);
        assert!(is_refusal(&r), "{src}: {r:?}");
    }
    // the root mpmath gives does solve it (symplex's own complex erf)
    let z = parse(&ctx, "erf(0.869417565326184121 + 1.340594818530277116*I)")
        .eval_complex64()
        .unwrap();
    assert!((z.re - 2.0).abs() < 1e-12 && z.im.abs() < 1e-12, "{z}");
}

/// For a real unknown the range restriction still proves it: `erf(t) ∈ (−1,
/// 1)` and `erfc(t) ∈ (0, 2)` for real `t` (exact bounds), unchanged.
#[test]
fn erf_out_of_range_for_a_real_unknown_is_no_solution() {
    let ctx = Context::new();
    let x = real(&ctx, "x");
    for src in [
        "erf(x) - 2",
        "erfc(x) - 3",
        "erfc(x) + 1",
        "erf(2*x + 1) - 3",
    ] {
        let r = parse(&ctx, src).solve(&x);
        assert!(is_no_solution(&r), "{src}: {r:?}");
    }
}

/// `(x − 2)·(erf(x) − 2) = 0` for a plain `x`: up to 0.48 `Ok([2])`, given
/// as all the roots, while `erf(z) = 2` has complex roots (mpmath above).
/// The factor is now refused, and so is the product.
#[test]
fn product_with_an_erf_factor_out_of_range_is_not_complete() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = parse(&ctx, "(x - 2)*(erf(x) - 2)").solve(&x);
    assert!(is_refusal(&r), "{r:?}");
}

/// `erf(i·x) = 2` for a real `x`: up to 0.48 `Ok([−erfinv(2)·i])`, a value
/// with no meaning (mpmath `erfinv(2)` raises "erfinv(x) is defined only for
/// -1 <= x <= 1"); `erf(i·t) = i·erfi(t)` is imaginary for real `t` (exact
/// identity), so there is no real root.  The inversion is now refused (the
/// argument is not real, so the range restriction does not apply).
#[test]
fn erf_of_a_non_real_argument_out_of_range_has_no_bogus_root() {
    let ctx = Context::new();
    let x = real(&ctx, "x");
    let e = parse(&ctx, "erf(I*x) - 2");
    let r = e.solve(&x);
    assert!(is_refusal(&r), "{r:?}");
    let zero = ctx.int(0);
    let red = reduce_inequalities(&[e.eq_expr(&zero)], &x);
    assert!(red.is_err(), "{red:?}");
}

/// `reduce_inequalities` asks for the real zeros with the unknown real
/// (`Domain::Real`): with the erf arm now honouring the declared domain of
/// `Ex::solve`, a plain `x` must still get the real-line answers.  `|erf(t)|
/// < 1` for real `t`, so `erf(x) = 2` has no real root and `erf(x) − 2 < 0`
/// everywhere; `erfc(t) > 0`, so `erfc(x) + 1 > 0` everywhere.
#[test]
fn reduce_inequalities_keeps_the_real_line_for_a_plain_unknown() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let e = parse(&ctx, "erf(x) - 2");
    let eq = reduce_inequalities(&[e.eq_expr(&zero)], &x).unwrap();
    assert_eq!(eq.to_string(), "EmptySet");
    let gt = reduce_inequalities(&[e.gt(&zero)], &x).unwrap();
    assert_eq!(gt.to_string(), "EmptySet");
    let c = parse(&ctx, "erfc(x) + 1");
    let pos = reduce_inequalities(&[c.gt(&zero)], &x).unwrap();
    assert_eq!(pos.to_string(), "(-oo, oo)");
}

/// `|x + i|² = 4` for a real `x`: up to 0.48 `Err(NoSolution)`, and
/// `solve_as_set` / `reduce_inequalities` gave `EmptySet`.  The power peels
/// to `|x + i| = ±2`; the branch `−2` is empty, the branch `2` (a modulus of
/// a non-real argument) was refused, and the merge of the two took the
/// refusal for a proof.  `x = ±√3` solve it (exact: `|±√3 + i|² = 3 + 1`;
/// SymPy `solveset(Abs(x + I)**2 - 4, x, S.Reals)` = `{-sqrt(3), sqrt(3)}`).
/// In 0.49 refused; since 0.50 (w3c) the branch `|x + i| = 2` of a real
/// `x` is solved as `|f|² = (Re f)² + (Im f)² = 4`, `x² + 1 = 4`.
#[test]
fn branch_merge_refused_and_empty_is_refused() {
    let ctx = Context::new();
    let x = real(&ctx, "x");
    let e = parse(&ctx, "Abs(x + I)**2 - 4");
    for r in ["sqrt(3)", "-sqrt(3)"] {
        assert_exact_root(&e, &x, &parse(&ctx, r));
    }
    let mut roots: Vec<String> = e.solve(&x).unwrap().iter().map(|r| r.to_string()).collect();
    roots.sort();
    assert_eq!(roots, ["-sqrt(3)", "sqrt(3)"]);
    assert_ne!(e.solve_as_set(&x).to_string(), "EmptySet");
    let zero = ctx.int(0);
    let red = reduce_inequalities(&[e.eq_expr(&zero)], &x).unwrap();
    for r in ["sqrt(3)", "-sqrt(3)"] {
        assert_eq!(red.contains(&parse(&ctx, r)), Some(true), "{red}");
    }
    // `|i·x|² = 4`: the same merge; `x = ±2` (exact)
    let e2 = parse(&ctx, "Abs(I*x)**2 - 4");
    assert_exact_root(&e2, &x, &ctx.int(2));
    let r2 = e2.solve(&x);
    assert!(!is_no_solution(&r2), "{r2:?}");
}

/// `|x|² = 4` and `|x − 1|² = 9` for a plain (complex) `x`: up to 0.48
/// `solve_as_set` gave `EmptySet` while `solve` returned `[2, −2]` / `[4,
/// −2]`; the solutions are the circles `|x| = 2`, `|x − 1| = 3` (`x = 2`:
/// exact).  Now a `ConditionSet`.
#[test]
fn modulus_circle_is_not_an_empty_set() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (src, root) in [("Abs(x)**2 - 4", "2"), ("Abs(x - 1)**2 - 9", "4")] {
        let e = parse(&ctx, src);
        assert_exact_root(&e, &x, &parse(&ctx, root));
        let set = e.solve_as_set(&x);
        assert_ne!(set.to_string(), "EmptySet", "{src}");
    }
}

/// A merge whose branches are all proved empty is still a proof: for a real
/// `x`, `(eˣ + 2)² = 1` peels to `eˣ = −1` and `eˣ = −3`, both impossible
/// (`eˣ + 2 > 2`, exact), as in 0.48.
#[test]
fn branch_merge_all_empty_is_no_solution() {
    let ctx = Context::new();
    let x = real(&ctx, "x");
    let r = parse(&ctx, "(exp(x) + 2)**2 - 1").solve(&x);
    assert!(is_no_solution(&r), "{r:?}");
}

/// `TransferFunction::poles` of `1/((s⁵ + a·s + 1)(s − 2))`: up to 0.48 `[]`
/// — the same value as for a constant denominator, with even the pole `2`
/// (exact root of the factor `s − 2`) lost.  `try_poles` now says that the
/// poles are not all known, and `poles` stays empty (never a part of them).
#[test]
fn transfer_function_try_poles_tells_refusal_from_no_poles() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    let den = parse(&ctx, "(s**5 + a*s + 1)*(s - 2)");
    let g = TransferFunction::new(ctx.int(1), den.clone(), s.clone());
    assert!(g.try_poles().is_err());
    assert!(g.poles().is_empty());
    assert_exact_root(&den, &s, &ctx.int(2));
    let k = TransferFunction::new(ctx.int(1), ctx.int(2), s.clone());
    assert_eq!(k.try_poles().unwrap(), Vec::<Ex>::new());
    // repeated poles with multiplicity, as before
    let h = TransferFunction::new(parse(&ctx, "s - 3"), parse(&ctx, "(s + 1)**2"), s.clone());
    assert_eq!(h.try_poles().unwrap(), vec![ctx.int(-1), ctx.int(-1)]);
    assert_eq!(h.try_zeros().unwrap(), vec![ctx.int(3)]);
}

/// A transcendental denominator `eˢ + 1` has the poles `s = (2k + 1)πi` for
/// every integer `k` (exact: `e^((2k+1)πi) = −1`); up to 0.48 `poles` gave
/// `[πi]`, the solver's principal root, as if it were all of them.  Now
/// `try_poles` refuses and `poles` is empty.
#[test]
fn transfer_function_poles_of_a_transcendental_denominator_are_not_a_subset() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    let den = parse(&ctx, "exp(s) + 1");
    assert_exact_root(&den, &s, &parse(&ctx, "3*pi*I"));
    let g = TransferFunction::new(ctx.int(1), den, s.clone());
    assert!(g.try_poles().is_err());
    assert!(g.poles().is_empty());
}
