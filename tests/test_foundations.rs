//! Tests for Cycle 14 foundational features.
//!
//! This file tests:
//!   • Factorial evaluation via arena-level constructors
//!   • Binomial coefficient evaluation via arena-level constructors
//!   • Display formatting for factorial and binomial nodes
//!   • Equation type (aspirational — `symplex::eq` is not yet publicly
//!     exported in `lib.rs`, so those tests are commented out and marked
//!     with `// ASPIRATIONAL`)
//!
//! Factorial and binomial constructors live on `Arena` (`arena.factorial`,
//! `arena.binomial`) and are accessed through `Context::with_arena_mut`.
//! The `ExprNode::Factorial` and `ExprNode::Binomial` variants exist in
//! `node.rs`, display support is wired in `display.rs`, and evaluation
//! logic is in `eval.rs`.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Factorial evaluation (through arena-level API via Context::with_arena_mut)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorial_of_5() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let five = arena.int(5);
        let expr = arena.factorial(five);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "120");
    });
}

#[test]
fn factorial_of_zero() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let zero = arena.int(0);
        let expr = arena.factorial(zero);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "1");
    });
}

#[test]
fn factorial_of_one() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let one = arena.int(1);
        let expr = arena.factorial(one);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "1");
    });
}

#[test]
fn factorial_of_2() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let two = arena.int(2);
        let expr = arena.factorial(two);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "2");
    });
}

#[test]
fn factorial_of_3() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let three = arena.int(3);
        let expr = arena.factorial(three);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "6");
    });
}

#[test]
fn factorial_of_6() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let six = arena.int(6);
        let expr = arena.factorial(six);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "720");
    });
}

#[test]
fn factorial_of_10() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let ten = arena.int(10);
        let expr = arena.factorial(ten);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "3628800");
    });
}

#[test]
fn factorial_of_20() {
    // 20 is the max the evaluator will compute (guard against huge factorials)
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let twenty = arena.int(20);
        let expr = arena.factorial(twenty);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "2432902008176640000");
    });
}

#[test]
fn factorial_of_symbolic_stays_unevaluated() {
    // factorial(x) should remain as-is when x is symbolic
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let x = arena.symbol("x");
        let expr = arena.factorial(x);
        let result = arena.eval_expr(expr);
        let s = arena.display(result).to_string();
        assert!(
            s.contains("!"),
            "symbolic factorial should display with !: {s}"
        );
        assert!(s.contains("x"), "should still contain x: {s}");
    });
}

#[test]
fn factorial_display_atom() {
    // Factorial of an atom: "5!"
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let five = arena.int(5);
        let expr = arena.factorial(five);
        let s = arena.display(expr).to_string();
        assert_eq!(s, "5!");
    });
}

#[test]
fn factorial_display_symbol() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.symbol("n");
        let expr = arena.factorial(n);
        let s = arena.display(expr).to_string();
        assert_eq!(s, "n!");
    });
}

#[test]
fn factorial_display_compound_gets_parens() {
    // Factorial of a compound expression should get parentheses: (x + 1)!
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let x = arena.symbol("x");
        let one = arena.int(1);
        let sum = arena.add(&[x, one]);
        let expr = arena.factorial(sum);
        let s = arena.display(expr).to_string();
        assert!(
            s.contains("(") && s.contains(")") && s.contains("!"),
            "compound factorial should have parens: {s}"
        );
    });
}

#[test]
fn factorial_21_evaluates() {
    // No artificial limit — BigInt handles arbitrary precision.
    // 21! = 51090942171709440000
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let twenty_one = arena.int(21);
        let expr = arena.factorial(twenty_one);
        let result = arena.eval_expr(expr);
        let s = arena.display(result).to_string();
        assert_eq!(s, "51090942171709440000", "21! should evaluate: {s}");
    });
}

#[test]
fn factorial_negative_stays_unevaluated() {
    // Factorial of negative integer stays unevaluated
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let neg = arena.int(-3);
        let expr = arena.factorial(neg);
        let result = arena.eval_expr(expr);
        let s = arena.display(result).to_string();
        assert!(
            s.contains("!"),
            "factorial(-3) should stay unevaluated: {s}"
        );
    });
}

#[test]
fn factorial_rational_stays_unevaluated() {
    // Factorial of a non-integer rational stays unevaluated
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let half = arena.rational(1, 2);
        let expr = arena.factorial(half);
        let result = arena.eval_expr(expr);
        let s = arena.display(result).to_string();
        assert!(
            s.contains("!"),
            "factorial(1/2) should stay unevaluated: {s}"
        );
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Binomial coefficient evaluation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn binomial_5_choose_2() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let five = arena.int(5);
        let two = arena.int(2);
        let expr = arena.binomial(five, two);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "10");
    });
}

#[test]
fn binomial_10_choose_3() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let ten = arena.int(10);
        let three = arena.int(3);
        let expr = arena.binomial(ten, three);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "120");
    });
}

#[test]
fn binomial_n_choose_0() {
    // C(7, 0) = 1
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let seven = arena.int(7);
        let zero = arena.int(0);
        let expr = arena.binomial(seven, zero);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "1");
    });
}

#[test]
fn binomial_n_choose_n() {
    // C(5, 5) = 1
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let five = arena.int(5);
        let expr = arena.binomial(five, five);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "1");
    });
}

#[test]
fn binomial_n_choose_1() {
    // C(8, 1) = 8
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let eight = arena.int(8);
        let one = arena.int(1);
        let expr = arena.binomial(eight, one);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "8");
    });
}

#[test]
fn binomial_symmetry() {
    // C(7, 2) == C(7, 5)
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let seven = arena.int(7);
        let two = arena.int(2);
        let five = arena.int(5);

        let c72 = arena.binomial(seven, two);
        let c75 = arena.binomial(seven, five);

        let r72 = arena.eval_expr(c72);
        let r75 = arena.eval_expr(c75);

        assert_eq!(
            arena.display(r72).to_string(),
            arena.display(r75).to_string(),
            "C(7,2) should equal C(7,5)"
        );
        assert_eq!(arena.display(r72).to_string(), "21");
    });
}

#[test]
fn binomial_0_choose_0() {
    // C(0, 0) = 1
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let zero = arena.int(0);
        let expr = arena.binomial(zero, zero);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "1");
    });
}

#[test]
fn binomial_20_choose_10() {
    // C(20, 10) = 184756
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let twenty = arena.int(20);
        let ten = arena.int(10);
        let expr = arena.binomial(twenty, ten);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "184756");
    });
}

#[test]
fn binomial_display_symbolic() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.symbol("n");
        let k = arena.symbol("k");
        let expr = arena.binomial(n, k);
        let s = arena.display(expr).to_string();
        assert!(
            s.contains("C(") && s.contains("n") && s.contains("k"),
            "binomial display should be C(n, k): got {s}"
        );
    });
}

#[test]
fn binomial_display_numeric() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let five = arena.int(5);
        let two = arena.int(2);
        let expr = arena.binomial(five, two);
        let s = arena.display(expr).to_string();
        assert!(
            s.contains("C(") && s.contains("5") && s.contains("2"),
            "unevaluated binomial display: got {s}"
        );
    });
}

#[test]
fn binomial_k_greater_than_n_stays_unevaluated() {
    // C(3, 5) — k > n — evaluator returns None, stays symbolic
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let three = arena.int(3);
        let five = arena.int(5);
        let expr = arena.binomial(three, five);
        let result = arena.eval_expr(expr);
        let s = arena.display(result).to_string();
        assert!(s.contains("C("), "C(3,5) should stay unevaluated: {s}");
    });
}

#[test]
fn binomial_negative_stays_unevaluated() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let neg = arena.int(-1);
        let two = arena.int(2);
        let expr = arena.binomial(neg, two);
        let result = arena.eval_expr(expr);
        let s = arena.display(result).to_string();
        assert!(s.contains("C("), "C(-1,2) should stay unevaluated: {s}");
    });
}

#[test]
fn binomial_symbolic_stays_unevaluated() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.symbol("n");
        let two = arena.int(2);
        let expr = arena.binomial(n, two);
        let result = arena.eval_expr(expr);
        let s = arena.display(result).to_string();
        assert!(
            s.contains("C(") && s.contains("n"),
            "C(n,2) should stay unevaluated: {s}"
        );
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-cutting: factorial and binomial in expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorial_in_addition() {
    // 5! + 3 = 123
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let five = arena.int(5);
        let three = arena.int(3);
        let fact = arena.factorial(five);
        let sum = arena.add(&[fact, three]);
        let result = arena.eval_expr(sum);
        assert_eq!(arena.display(result).to_string(), "123");
    });
}

#[test]
fn factorial_in_multiplication() {
    // 5! * 2 = 240
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let five = arena.int(5);
        let two = arena.int(2);
        let fact = arena.factorial(five);
        let product = arena.mul(&[fact, two]);
        let result = arena.eval_expr(product);
        assert_eq!(arena.display(result).to_string(), "240");
    });
}

#[test]
fn binomial_in_expression() {
    // C(5, 2) + 1 = 11
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let five = arena.int(5);
        let two = arena.int(2);
        let one = arena.int(1);
        let binom = arena.binomial(five, two);
        let sum = arena.add(&[binom, one]);
        let result = arena.eval_expr(sum);
        assert_eq!(arena.display(result).to_string(), "11");
    });
}

#[test]
fn factorial_identity_n_choose_k_via_factorials() {
    // Verify C(5,2) = 5! / (2! * 3!) by computing both sides
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        // Left side: C(5, 2)
        let five = arena.int(5);
        let two = arena.int(2);
        let binom = arena.binomial(five, two);
        let binom_val = arena.eval_expr(binom);

        // Right side: 5! / (2! * 3!)
        let three = arena.int(3);
        let fact5 = arena.factorial(five);
        let fact2 = arena.factorial(two);
        let fact3 = arena.factorial(three);
        let denom = arena.mul(&[fact2, fact3]);
        let ratio = arena.div(fact5, denom);
        let ratio_val = arena.eval_expr(ratio);

        assert_eq!(
            arena.display(binom_val).to_string(),
            arena.display(ratio_val).to_string(),
            "C(5,2) should equal 5!/(2!*3!)"
        );
    });
}

#[test]
fn factorial_eval_is_idempotent() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let five = arena.int(5);
        let expr = arena.factorial(five);
        let once = arena.eval_expr(expr);
        let twice = arena.eval_expr(once);
        assert_eq!(
            arena.display(once).to_string(),
            arena.display(twice).to_string(),
            "eval of factorial should be idempotent"
        );
    });
}

#[test]
fn binomial_eval_is_idempotent() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let ten = arena.int(10);
        let three = arena.int(3);
        let expr = arena.binomial(ten, three);
        let once = arena.eval_expr(expr);
        let twice = arena.eval_expr(once);
        assert_eq!(
            arena.display(once).to_string(),
            arena.display(twice).to_string(),
            "eval of binomial should be idempotent"
        );
    });
}

#[test]
fn pascal_triangle_row_4() {
    // Row 4 of Pascal's triangle: C(4,0)=1, C(4,1)=4, C(4,2)=6, C(4,3)=4, C(4,4)=1
    let expected = [1, 4, 6, 4, 1];
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let four = arena.int(4);
        for (k, &exp_val) in expected.iter().enumerate() {
            let k_id = arena.int(k as i64);
            let binom = arena.binomial(four, k_id);
            let result = arena.eval_expr(binom);
            assert_eq!(
                arena.display(result).to_string(),
                exp_val.to_string(),
                "C(4, {k}) should be {exp_val}"
            );
        }
    });
}

#[test]
fn factorial_table_0_through_10() {
    let expected: [(i64, &str); 11] = [
        (0, "1"),
        (1, "1"),
        (2, "2"),
        (3, "6"),
        (4, "24"),
        (5, "120"),
        (6, "720"),
        (7, "5040"),
        (8, "40320"),
        (9, "362880"),
        (10, "3628800"),
    ];
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        for (n, exp_str) in &expected {
            let n_id = arena.int(*n);
            let expr = arena.factorial(n_id);
            let result = arena.eval_expr(expr);
            assert_eq!(
                arena.display(result).to_string(),
                *exp_str,
                "{n}! should be {exp_str}"
            );
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// ExprNode::Factorial and ExprNode::Binomial structural properties
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorial_node_is_not_atom() {
    // Factorial(x) should not be an atom — it has one child
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let x = arena.symbol("x");
        let expr = arena.factorial(x);
        let node = arena.node(expr);
        assert!(!node.is_atom(), "Factorial should not be an atom");
        assert_eq!(
            node.children().len(),
            1,
            "Factorial should have exactly 1 child"
        );
    });
}

#[test]
fn binomial_node_is_not_atom() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.symbol("n");
        let k = arena.symbol("k");
        let expr = arena.binomial(n, k);
        let node = arena.node(expr);
        assert!(!node.is_atom(), "Binomial should not be an atom");
        assert_eq!(
            node.children().len(),
            2,
            "Binomial should have exactly 2 children"
        );
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// ASPIRATIONAL: Equation type
//
// The `Equation` type lives in `symplex::eq` but the module is not yet
// publicly exported in `lib.rs`. Once `pub mod eq;` is added, uncomment
// the tests below. The internal unit tests in `src/eq.rs` cover this
// functionality already.
// ═══════════════════════════════════════════════════════════════════════════

// ASPIRATIONAL: Uncomment when `pub mod eq;` is added to lib.rs
//
// #[test]
// fn equation_basic() {
//     use symplex::eq::Equation;
//     let x = symplex::default_context().symbol("x");
//     let eq = Equation::new(&x + 1, symplex::default_context().int(5));
//     let s = format!("{eq}");
//     assert!(s.contains("="), "equation should display with =: {s}");
// }
//
// #[test]
// fn equation_solve_linear() {
//     use symplex::eq::Equation;
//     let x = symplex::default_context().symbol("x");
//     let eq = Equation::new(&x * 2, symplex::default_context().int(10));
//     let roots = eq.solve_or_empty(&x);
//     assert_eq!(roots.len(), 1);
//     assert_eq!(format!("{}", roots[0]), "5");
// }
//
// #[test]
// fn equation_solve_quadratic() {
//     use symplex::eq::Equation;
//     let x = symplex::default_context().symbol("x");
//     let eq = Equation::new(x.powi(2), symplex::default_context().int(9));
//     let roots = eq.solve_or_empty(&x);
//     assert_eq!(roots.len(), 2, "x²=9 should have 2 roots");
//     let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
//     let joined = strs.join(",");
//     assert!(joined.contains("3"), "should contain root 3: {joined}");
// }
//
// #[test]
// fn equation_subs_check() {
//     use symplex::eq::Equation;
//     let x = symplex::default_context().symbol("x");
//     let eq = Equation::new(&x + 1, symplex::default_context().int(5));
//     let at_4 = eq.subs_i64(&x, 4);
//     assert!(at_4.is_satisfied() == Some(true));
// }
//
// #[test]
// fn equation_subs_wrong() {
//     use symplex::eq::Equation;
//     let x = symplex::default_context().symbol("x");
//     let eq = Equation::new(&x + 1, symplex::default_context().int(5));
//     let at_3 = eq.subs_i64(&x, 3);
//     // 4 ≠ 5
//     assert!(at_3.is_satisfied() != Some(true));
// }
//
// #[test]
// fn equation_to_expr() {
//     use symplex::eq::Equation;
//     let x = symplex::default_context().symbol("x");
//     let eq = Equation::new(x.clone(), symplex::default_context().int(3));
//     let expr = eq.to_expr();
//     // Should be x - 3
//     let s = format!("{expr}");
//     assert!(s.contains("x"), "to_expr should contain x: {s}");
// }
//
// #[test]
// fn equation_simplify() {
//     use symplex::eq::Equation;
//     let x = symplex::default_context().symbol("x");
//     let eq = Equation::new(
//         &x.sin().powi(2) + &x.cos().powi(2),
//         symplex::default_context().int(1),
//     );
//     let simplified = eq.simplify();
//     assert_eq!(format!("{}", simplified.lhs), "1");
// }
//
// #[test]
// fn equation_expand() {
//     use symplex::eq::Equation;
//     let x = symplex::default_context().symbol("x");
//     let eq = Equation::new((&x + 1).powi(2), symplex::default_context().int(4));
//     let expanded = eq.expand();
//     let s = format!("{}", expanded.lhs);
//     assert!(s.contains("x^2") || s.contains("x"), "should expand: {s}");
// }
//
// #[test]
// fn equation_with_complex() {
//     use symplex::eq::Equation;
//     let x = symplex::default_context().symbol("x");
//     let eq = Equation::new(x.powi(2), symplex::default_context().int(-1));
//     let roots = eq.solve_or_empty(&x);
//     assert_eq!(roots.len(), 2, "x²=-1 should have complex roots");
//     let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
//     let joined = strs.join(",");
//     assert!(joined.contains("I"), "should contain I: {joined}");
// }
//
// #[test]
// fn equation_debug_format() {
//     use symplex::eq::Equation;
//     let x = symplex::default_context().symbol("x");
//     let eq = Equation::new(x.powi(2), symplex::default_context().int(4));
//     let s = format!("{eq:?}");
//     assert!(s.contains("Equation"), "debug: {s}");
// }
//
// #[test]
// fn equation_eval() {
//     use symplex::eq::Equation;
//     let x = symplex::default_context().symbol("x");
//     let eq = Equation::new(x.sin().powi(2) + x.cos().powi(2), symplex::default_context().int(2));
//     let evald = eq.eval();
//     let s = format!("{evald}");
//     assert!(s.contains("="), "should still be equation: {s}");
// }
//
// #[test]
// fn equation_multiple_variables() {
//     use symplex::eq::Equation;
//     let ctx = Context::new();
//     let x = ctx.symbol("x");
//     let y = ctx.symbol("y");
//     let eq = Equation::new(&x + &y, symplex::default_context().int(10));
//     let substituted = eq.subs(&y, &symplex::default_context().int(3));
//     let roots = substituted.solve_or_empty(&x);
//     assert!(!roots.is_empty(), "x + 3 = 10 should solve");
//     assert_eq!(format!("{}", roots[0]), "7");
// }

// ═══════════════════════════════════════════════════════════════════════════
// Equation-like workflows using existing public API (lhs - rhs = 0 form)
//
// Until Equation is publicly exported, we can test the same algebraic
// workflows using the existing `Ex::solve` / `Ex::solve_or_empty` API,
// which solves `expr = 0`.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_linear_equation_via_expr() {
    // x + 1 = 5  ⟹  (x + 1) - 5 = 0  ⟹  x - 4 = 0
    let x = symplex::default_context().symbol("x");
    let expr = &x + 1 - 5;
    let roots = expr.solve_or_empty(&x);
    assert_eq!(roots.len(), 1);
    assert_eq!(format!("{}", roots[0]), "4");
}

#[test]
fn solve_linear_2x_eq_10() {
    // 2x = 10  ⟹  2x - 10 = 0
    let x = symplex::default_context().symbol("x");
    let expr = &x * 2 - 10;
    let roots = expr.solve_or_empty(&x);
    assert_eq!(roots.len(), 1);
    assert_eq!(format!("{}", roots[0]), "5");
}

#[test]
fn solve_quadratic_x2_eq_9() {
    // x² = 9  ⟹  x² - 9 = 0
    let x = symplex::default_context().symbol("x");
    let expr = x.powi(2) - 9;
    let roots = expr.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "x²-9=0 should have 2 roots");
    let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    strs.sort();
    assert_eq!(strs, vec!["-3", "3"], "roots of x²-9 should be exactly -3 and 3");
}

#[test]
fn solve_quadratic_x2_eq_neg1_complex() {
    // x² = -1  ⟹  x² + 1 = 0
    let x = symplex::default_context().symbol("x");
    let expr = x.powi(2) + 1;
    let roots = expr.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "x²+1=0 should have 2 complex roots");
    let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    strs.sort();
    assert_eq!(strs, vec!["-I", "I"], "roots of x²+1 should be exactly I and -I");
}

#[test]
fn substitution_verifies_solution() {
    // x + 1 = 5  →  expr = x + 1 - 5  →  solve gives x = 4
    // Verify: subs(x, 4) into expr should give 0
    let x = symplex::default_context().symbol("x");
    let expr = &x + 1 - 5;
    let roots = expr.solve_or_empty(&x);
    assert_eq!(roots.len(), 1);
    let at_root = expr.subs(&x, &roots[0]);
    let simplified = at_root.simplify();
    assert_eq!(
        format!("{simplified}"),
        "0",
        "substituting root should give 0"
    );
}

#[test]
fn pythagorean_identity_simplifies() {
    // sin²(x) + cos²(x) should simplify to 1
    let x = symplex::default_context().symbol("x");
    let expr = x.sin().powi(2) + x.cos().powi(2);
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "1");
}

#[test]
fn expand_square_binomial() {
    // (x + 1)² should expand to x^2 + 2*x + 1
    let x = symplex::default_context().symbol("x");
    let expr = (&x + 1).powi(2);
    let expanded = expr.expand();
    let s = format!("{expanded}");
    assert_eq!(s, "x^2 + 2*x + 1", "expand (x+1)² should be x^2 + 2*x + 1");
}

#[test]
fn to_expr_pattern_lhs_minus_rhs() {
    // Equation "x = 3" → to_expr is "x - 3"
    let x = symplex::default_context().symbol("x");
    let three = symplex::default_context().int(3);
    let to_expr = &x - &three;
    let s = format!("{to_expr}");
    assert_eq!(s, "x - 3", "x - 3 canonical form");
}

#[test]
fn multi_variable_equation_workflow() {
    // x + y = 10, set y = 3, solve for x → x = 7
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let three = ctx.int(3);
    let expr = &x + &y - 10;
    let after_subs = expr.subs(&y, &three);
    let roots = after_subs.solve_or_empty(&x);
    assert!(!roots.is_empty(), "x + 3 - 10 = 0 should solve");
    assert_eq!(format!("{}", roots[0]), "7");
}
