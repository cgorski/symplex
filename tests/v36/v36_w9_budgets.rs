//! Tests of the 0.49 refactor (w9): deterministic budgets, per-arena state and structured errors.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::BudgetKind;
use symplex::prelude::*;

/// A 2×2 matrix whose entries have an enormous *tree* (a tiny DAG):
/// `eₙ₊₁ = sin(eₙ) + cos(eₙ)`, sixteen times — beyond `EXPRESSION_BUDGET`.
fn swollen(ctx: &Context) -> Matrix {
    let mut e = ctx.symbol("x");
    for _ in 0..16 {
        e = &e.sin() + &e.cos();
    }
    Matrix::new(vec![vec![e.clone(), ctx.int(1)], vec![ctx.int(1), e]]).unwrap()
}

/// S5-B4: the series engine's last resort was bounded by 20,000 nodes of
/// arena *growth*; a context that already held the nodes never reached
/// it.  `series(sign(asin(asin(x))), x, 1, 3)` repeated in one context took
/// 0.79, 2.10, 4.46, 7.81, 10.61 s (m4) and the arena grew by 34,010,
/// 22,020, 20,807, 21,943 nodes on the repeats.  It now counts interns
/// attempted, which a warm context makes exactly as a fresh one does: the
/// repeat builds nothing new and gives the same (unevaluated) answer.
/// Oracle: the fresh-context answer, `Series(sign(asin(asin(x))), x, 1,
/// 3)` (`asin(asin(1))` is not real — `asin(π/2)`, `π/2 > 1` — so the
/// expansion at the branch point has no leading term the engine can
/// certify).
#[test]
fn series_last_resort_costs_the_same_in_a_warm_context() {
    let fresh = {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let f = ctx.parse("sign(asin(asin(x)))").unwrap();
        f.series(&x, &ctx.int(1), 3).to_string()
    };
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let f = ctx.parse("sign(asin(asin(x)))").unwrap();
    let first = f.series(&x, &one, 3).to_string();
    let after_first = ctx.node_count();
    let second = f.series(&x, &one, 3).to_string();
    assert_eq!(first, fresh);
    assert_eq!(second, fresh);
    assert_eq!(
        ctx.node_count(),
        after_first,
        "the repeat built new nodes: its budget depends on the context's history"
    );
}

/// S5-B5: every `simplify()` built its rule sets on fresh wild symbols
/// from a process-global counter (`_w4711`, …), so each identical call
/// added 2,346 permanent nodes (2,451 → 4,797 → 7,143 …).  The built-in
/// rules now number their wilds per rule (`_w0`, `_w1`, `_w2`), the same
/// nodes every call.  Oracle: the exact identities `sin x cos y + cos x
/// sin y = sin(x + y)` and `cos²x − sin²x = cos 2x`.
#[test]
fn repeated_simplify_adds_no_nodes() {
    let ctx = Context::new();
    let e = ctx
        .parse("sin(x)*cos(y) + cos(x)*sin(y) + cos(x)^2 - sin(x)^2")
        .unwrap();
    let first = e.simplify();
    let after_first = ctx.node_count();
    for _ in 0..3 {
        assert_eq!(e.simplify(), first);
    }
    assert_eq!(first.to_string(), "sin(x + y) + cos(2*x)");
    assert_eq!(ctx.node_count(), after_first);
}

/// The wild names come from the arena, not the process: every context's
/// `simplify` builds its rules on `_w0`, `_w1`, … whatever ran before it.
/// Before, a second context in the process got `_w2419`, `_w2420`, … (the
/// global counter had moved on), so its nodes differed from a fresh
/// process's.  Oracle: `_w0` is already a node of either context after
/// one `simplify` (creating it adds nothing), and both give the same
/// answer and node count.
#[test]
fn wild_symbols_are_per_arena() {
    let run = || {
        let ctx = Context::new();
        let e = ctx.parse("cos(x)^2 - sin(x)^2 + sin(x)/cos(x)").unwrap();
        let r = e.simplify().to_string();
        let n = ctx.node_count();
        let _w0 = ctx.symbol("_w0");
        assert_eq!(ctx.node_count(), n, "`_w0` was not one of the wilds");
        (n, r)
    };
    let a = run();
    let b = run();
    assert_eq!(a, b);
}

/// `SymplexError` was not `Clone` (`evalf` copied it field by field in a
/// helper and turned unknown variants into `Unevaluable`).  Oracle: the
/// clone prints and matches like the original.
#[test]
fn symplex_error_is_clone() {
    let ctx = Context::new();
    let err = swollen(&ctx).inv().unwrap_err();
    let copy = err.clone();
    assert_eq!(copy.to_string(), err.to_string());
    assert!(matches!(copy, SymplexError::ExpressionSwell { .. }));
}

/// An expression swell was `ComputationFailed` with a reason the matrix
/// code tested with `starts_with("expression swell")` (six sites).  It is
/// now `ExpressionSwell { operation, nodes }` with the same message.
/// Oracle: the 0.48 message text, and `EXPRESSION_BUDGET` = 100,000.
#[test]
fn expression_swell_is_a_variant_with_the_old_message() {
    let ctx = Context::new();
    let err = swollen(&ctx).inv().unwrap_err();
    assert!(
        matches!(
            err,
            SymplexError::ExpressionSwell {
                operation: "inv",
                nodes: 100_000
            }
        ),
        "{err:?}"
    );
    assert_eq!(
        err.to_string(),
        "inv could not be computed: expression swell: intermediate result exceeds the budget of \
         100000 expression nodes; simplify the input or substitute numeric values first"
    );
    // A swell in an inner step of `matrix_exp` is still recognised as one
    // (its fallback through the Sylvester formula is tried, and the inner
    // step's error returned, as in 0.48).
    let err = swollen(&ctx).matrix_exp().unwrap_err();
    assert!(
        matches!(err, SymplexError::ExpressionSwell { .. }),
        "{err:?}"
    );
    assert!(
        err.to_string()
            .contains("could not be computed: expression swell")
    );
}

/// "Not diagonalizable" was `ComputationFailed`, recognised by
/// `starts_with("matrix is not diagonalizable")`; it is now
/// `NotDiagonalizable` with the same message, and the matrix functions
/// that fall back to the Jordan form on it still do.  Oracle: the Jordan
/// block `J = [[1, 1], [0, 1]]` has the single eigenvector `(1, 0)`;
/// `exp(J) = e·[[1, 1], [0, 1]]` and `J^n = [[1, n], [0, 1]]` (exact).
#[test]
fn not_diagonalizable_is_a_variant() {
    let ctx = Context::new();
    let j = Matrix::from_i64(&ctx, &[&[1, 1], &[0, 1]]).unwrap();
    let err = j.diagonalize().unwrap_err();
    assert!(
        matches!(
            err,
            SymplexError::NotDiagonalizable {
                operation: "diagonalize",
                ..
            }
        ),
        "{err:?}"
    );
    assert_eq!(
        err.to_string(),
        "diagonalize could not be computed: matrix is not diagonalizable: geometric multiplicity \
         does not equal algebraic multiplicity for all eigenvalues"
    );
    let e = ctx.e();
    let expected = Matrix::new(vec![vec![e.clone(), e.clone()], vec![ctx.int(0), e]]).unwrap();
    assert_eq!(j.matrix_exp().unwrap(), expected);
    let n = ctx.symbol("n");
    let p = j.matrix_pow_symbolic(&n).unwrap();
    let expected = Matrix::new(vec![
        vec![ctx.int(1), n.clone()],
        vec![ctx.int(0), ctx.int(1)],
    ])
    .unwrap();
    assert_eq!(p.simplify(), expected);
}

/// `BudgetKind` names what a deterministic budget counts; the quadrature's
/// evaluation budget keeps its 0.48 message.  Oracle: the 0.48 text of
/// `quad.rs`'s refusal.
#[test]
fn budget_exhausted_keeps_the_quadrature_message() {
    let err = SymplexError::BudgetExhausted {
        operation: "evalf",
        kind: BudgetKind::QuadratureEvaluations,
    };
    assert_eq!(
        err.to_string(),
        "evalf could not be computed: quadrature of the definite integral did not converge: the \
         evaluation budget was exhausted"
    );
    assert_eq!(BudgetKind::SeriesExpansion.to_string(), "series expansion");
    assert_eq!(BudgetKind::Integration.to_string(), "integration");
}
