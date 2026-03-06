//! Tests for ExprView — the non-locking expression inspection type.
//!
//! ExprView is only accessible inside `Expr::replace()` closures.
//! These tests verify its inspection methods by writing replace() closures
//! that use them and checking the transformation results.
//!
//! IMPORTANT: `replace()` holds a write lock on the arena, so we must
//! pre-create all replacement expressions *outside* the closure and
//! `.clone()` them inside. Calling `symplex::int()`, `symplex::var()`,
//! etc. inside the closure would deadlock on the global context lock.

use std::cell::Cell;

#[test]
fn expr_view_is_atom_identifies_leaves() {
    // replace() visits every node; replace atoms with 1
    let x = symplex::var("x");
    let expr = x.sin(); // sin(x) — two nodes: sin (not atom) and x (atom)

    // Pre-create the replacement outside the closure
    let one = symplex::int(1);

    let result = expr.replace(|view| {
        if view.is_atom() {
            Some(one.clone())
        } else {
            None
        }
    });
    // sin(x) with x→1 should give sin(1)
    assert_eq!(format!("{result}"), "sin(1)");
}

#[test]
fn expr_view_is_symbol_only_matches_symbols() {
    let x = symplex::var("x");
    let expr = &x + 1; // x + 1 — x is symbol, 1 is Num (atom but not symbol)

    let forty_two = symplex::int(42);

    let result = expr.replace(|view| {
        if view.is_symbol() {
            Some(forty_two.clone())
        } else {
            None
        }
    });
    // x→42, 1 stays: 42 + 1 = 43 (canonicalized at construction)
    assert_eq!(format!("{result}"), "43");
}

#[test]
fn expr_view_is_symbol_false_for_pi() {
    let pi = symplex::pi();
    let expr = &pi + 1;

    let ninety_nine = symplex::int(99);

    let result = expr.replace(|view| {
        if view.is_symbol() {
            Some(ninety_nine.clone()) // should NOT match pi
        } else {
            None
        }
    });
    // pi is not a symbol, so nothing changes
    assert_eq!(format!("{result}"), "1 + pi");
}

#[test]
fn expr_view_partial_eq_with_expr() {
    // Use PartialEq<Expr> to match a specific subexpression
    let x = symplex::var("x");
    let y = symplex::var("y");
    let expr = &x + &y;

    let ten = symplex::int(10);

    let result = expr.replace(|view| if view == x { Some(ten.clone()) } else { None });
    assert_eq!(format!("{result}"), "y + 10");
}

#[test]
fn expr_view_partial_eq_ref_variant() {
    let x = symplex::var("x");
    let expr = x.sin();

    let pi = symplex::pi();

    let result = expr.replace(|view| if view == x { Some(pi.clone()) } else { None });
    // sin(x) → sin(pi); the sin constructor does not auto-evaluate
    let s = format!("{result}");
    assert!(s.contains("pi"), "should have replaced x with pi: {s}");
}

#[test]
fn expr_view_children_count() {
    // Use replace to verify children are accessible.
    // replace() requires Fn (not FnMut), so use Cell for interior mutability.
    let x = symplex::var("x");
    let y = symplex::var("y");
    let expr = &x + &y; // Add node with 2 children

    let add_child_count = Cell::new(0usize);
    let _ = expr.replace(|view| {
        let children = view.children();
        if children.len() == 2 {
            add_child_count.set(children.len());
        }
        None // don't transform anything
    });
    assert_eq!(add_child_count.get(), 2, "Add(x, y) should have 2 children");
}

// Test that replace works for identity (returns None for everything)
#[test]
fn expr_view_replace_identity() {
    let x = symplex::var("x");
    let expr = x.powi(2).sin() + &x;
    let original = format!("{expr}");
    let result = expr.replace(|_| None);
    assert_eq!(format!("{result}"), original);
}

// Test replace with is_atom on a complex nested expression
#[test]
fn expr_view_is_atom_nested() {
    let x = symplex::var("x");
    // Build sin(x^2 + 1) — has atoms: x, 2, 1; non-atoms: Pow, Add, Sin
    let expr = (&x.powi(2) + 1).sin();

    let atom_count = Cell::new(0usize);
    let _ = expr.replace(|view| {
        if view.is_atom() {
            atom_count.set(atom_count.get() + 1);
        }
        None
    });
    // x, 2, 1 are atoms = at least 2 atoms (canonical form may merge constants)
    assert!(
        atom_count.get() >= 2,
        "should find at least 2 atoms, found {}",
        atom_count.get()
    );
}

// Test node() returns correct variant via pattern matching
#[test]
fn expr_view_node_variant() {
    use symplex::__macro_support::ExprNode;
    let x = symplex::var("x");
    let expr = x.sin();

    let found_sin = Cell::new(false);
    let _ = expr.replace(|view| {
        if matches!(view.node(), ExprNode::Sin(_)) {
            found_sin.set(true);
        }
        None
    });
    assert!(found_sin.get(), "should find Sin node in sin(x)");
}

// Verify ExprView cannot outlive the replace() closure (compile-time guarantee).
// This is tested by the fact that ExprView has a lifetime parameter tied to the closure.
// We just need a test that the replace pattern works correctly with multiple symbols.
#[test]
fn expr_view_multiple_replacements() {
    let x = symplex::var("x");
    let y = symplex::var("y");
    let expr = &x.sin() + &y.cos();

    // Pre-create replacement outside the closure
    let zero = symplex::int(0);

    // Replace all symbols with 0
    let result = expr.replace(|view| {
        if view.is_symbol() {
            Some(zero.clone())
        } else {
            None
        }
    });
    // sin(0) + cos(0) should stay symbolic or evaluate; either way x and y are gone
    let s = format!("{result}");
    assert!(
        !s.contains('x') && !s.contains('y'),
        "symbols should be replaced: {s}"
    );
}

// Test that id() returns a meaningful ExprId that can be compared
#[test]
fn expr_view_id_matches_expr_id() {
    let x = symplex::var("x");
    let expr = x.sin();

    let found_matching_id = Cell::new(false);
    let x_id = x.id();
    let _ = expr.replace(|view| {
        if view.id() == x_id {
            found_matching_id.set(true);
        }
        None
    });
    assert!(
        found_matching_id.get(),
        "should find a node whose id matches x.id()"
    );
}

// Test that children() returns empty for atoms
#[test]
fn expr_view_children_empty_for_atom() {
    let x = symplex::var("x");

    let atom_has_no_children = Cell::new(false);
    // sin(x) has two nodes: sin (1 child) and x (0 children)
    let _ = x.sin().replace(|view| {
        if view.is_atom() {
            let children = view.children();
            if children.is_empty() {
                atom_has_no_children.set(true);
            }
        }
        None
    });
    assert!(
        atom_has_no_children.get(),
        "atom nodes should have no children"
    );
}

// Test node() on a symbol returns ExprNode::Symbol
#[test]
fn expr_view_node_symbol_variant() {
    use symplex::__macro_support::ExprNode;
    let x = symplex::var("x");
    let expr = x.sin();

    let found_symbol = Cell::new(false);
    let _ = expr.replace(|view| {
        if matches!(view.node(), ExprNode::Symbol(_)) {
            found_symbol.set(true);
        }
        None
    });
    assert!(found_symbol.get(), "should find Symbol node for x");
}

// Test that is_atom is true for numeric constants but is_symbol is false
#[test]
fn expr_view_is_atom_for_number() {
    let x = symplex::var("x");
    let expr = &x + 1;

    let num_is_atom = Cell::new(false);
    let _ = expr.replace(|view| {
        // The integer 1 is an atom but not a symbol
        if view.is_atom() && !view.is_symbol() {
            num_is_atom.set(true);
        }
        None
    });
    assert!(
        num_is_atom.get(),
        "numeric literal should be an atom but not a symbol"
    );
}

// Test that is_atom is true for pi (a constant, not a symbol)
#[test]
fn expr_view_is_atom_for_pi() {
    use symplex::__macro_support::ExprNode;
    let pi = symplex::pi();
    let expr = pi.sin(); // sin(pi): pi is atom, sin is not

    let pi_is_atom = Cell::new(false);
    let _ = expr.replace(|view| {
        if matches!(view.node(), ExprNode::Pi) && view.is_atom() {
            pi_is_atom.set(true);
        }
        None
    });
    assert!(pi_is_atom.get(), "pi should be an atom");
}

// Test replace identity on a complex expression preserves exact display
#[test]
fn expr_view_replace_identity_complex() {
    let x = symplex::var("x");
    let expr = &x.powi(2) + &x.sin() + 1;
    let original = format!("{expr}");
    let result = expr.replace(|_| None);
    assert_eq!(format!("{result}"), original);
}
