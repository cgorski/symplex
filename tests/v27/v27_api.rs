//! Panics on caller input removed from the public API (0.29): each batch of
//! the `ASSERT_ALLOWLIST` debt converted to `Result` / `Option`, pinned so
//! that the panic cannot come back.  (Batch 1, the dead `plotting::rk4`
//! module, has no API to test: its removal is pinned by the ratchet in
//! `tests/unit/test_no_panics.rs`.)

use symplex::control::{StateSpace, TransferFunction, is_routh_stable, routh_array};
use symplex::linprog::qi;
use symplex::matrix::{cross, dot, jacobian};
use symplex::matrix_decomp::hessian;
use symplex::multipoly::{GrevLex, Lex, MultiPoly, monomial_mul};
use symplex::ntheory::legendre_symbol;
use symplex::prelude::*;
use symplex::sym;
use symplex::vector::{
    CoordinateSystem, curl, directional_derivative, divergence, gradient, gradient_in,
    is_conservative, is_solenoidal, laplacian, line_integral_scalar, line_integral_vector,
};

fn invalid<T: std::fmt::Debug>(r: Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::InvalidArgument { .. }))
}

fn contradictory<T: std::fmt::Debug>(
    r: &Result<T, SymplexError>,
) -> Option<(String, String, String)> {
    match r {
        Err(SymplexError::ContradictoryAssumptions { symbol, a, b }) => {
            Some((symbol.clone(), a.clone(), b.clone()))
        }
        _ => None,
    }
}

// ── Batch 2: StateSpace, routh_array, legendre_symbol, Context::apply, MultiPoly::eval ──

#[test]
fn state_space_new_validates_every_shape() {
    let ctx = Context::new();
    let a = matrix![ctx, [0, 1], [-2, -3]];
    let b = matrix![ctx, [0], [1]];
    let c = matrix![ctx, [1, 0]];
    let d = matrix![ctx, [0]];
    let ss = StateSpace::new(a.clone(), b.clone(), c.clone(), d.clone()).expect("conformant");
    assert_eq!((ss.a(), ss.b(), ss.c(), ss.d()), (&a, &b, &c, &d));
    assert_eq!(
        (ss.num_states(), ss.num_inputs(), ss.num_outputs()),
        (2, 1, 1)
    );

    // A not square; B, C, D not conformant: each panicked before 0.29.
    let a23 = matrix![ctx, [0, 1, 0], [1, 0, 0]];
    assert!(invalid(StateSpace::new(
        a23,
        b.clone(),
        c.clone(),
        d.clone()
    )));
    assert!(invalid(StateSpace::new(
        a.clone(),
        matrix![ctx, [1]],
        c.clone(),
        d.clone()
    )));
    assert!(invalid(StateSpace::new(
        a.clone(),
        b.clone(),
        matrix![ctx, [1, 0, 0]],
        d.clone()
    )));
    assert!(invalid(StateSpace::new(
        a.clone(),
        b.clone(),
        c.clone(),
        matrix![ctx, [0, 0]]
    )));
    assert!(invalid(StateSpace::new(a, b, c, matrix![ctx, [0], [0]])));

    // The only way in is validated, so the derived quantities cannot fail on shape.
    let s = ctx.symbol("s");
    assert_eq!(
        ss.try_char_poly(&s).unwrap().expand(),
        (&s.powi(2) + &s * 3 + 2).expand()
    );
    assert!(ss.controllability_matrix().is_ok() && ss.observability_matrix().is_ok());
    let g = TransferFunction::from_coeffs(&[1], &[2, 3, 1], &s);
    assert_eq!(g.to_state_space().unwrap().a(), ss.a());
}

#[test]
fn routh_array_of_nothing_is_an_error() {
    let ctx = Context::new();
    assert!(invalid(routh_array(&[])));
    assert_eq!(is_routh_stable(&[]), Some(false));
    let table = routh_array(&[ctx.int(1), ctx.int(3), ctx.int(2)]).unwrap();
    assert_eq!(table.len(), 3);
    assert_eq!(routh_array(&[ctx.int(5)]).unwrap(), vec![vec![ctx.int(5)]]);
}

#[test]
fn legendre_symbol_rejects_a_modulus_that_is_not_an_odd_prime() {
    assert_eq!(legendre_symbol(2, 7).unwrap(), 1);
    assert_eq!(legendre_symbol(3, 7).unwrap(), -1);
    assert_eq!(legendre_symbol(14, 7).unwrap(), 0);
    for p in [-7i64, 0, 1, 2, 9, 15, 1001] {
        assert!(invalid(legendre_symbol(3, p)), "p = {p}");
    }
}

#[test]
fn apply_with_an_empty_name_is_an_error() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(invalid(ctx.apply("", &[&x])));
    assert_eq!(ctx.apply("f", &[&x]).unwrap().to_string(), "f(x)");
}

#[test]
fn multipoly_eval_checks_the_number_of_values() {
    let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    let p = x.mul(&y).add(&x);
    assert_eq!(p.eval(&[qi(2), qi(3)]).unwrap(), qi(8));
    assert!(invalid(p.eval(&[qi(2)])));
    assert!(invalid(p.eval(&[qi(2), qi(3), qi(4)])));
    assert_eq!(
        MultiPoly::<GrevLex>::from_int(0, 7).eval(&[]).unwrap(),
        qi(7)
    );
}

// ── Batch 3: matrix and vector-calculus shapes ──

#[test]
fn matrix_calculus_shapes_are_errors() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f = &x.powi(2) * &y;
    assert!(invalid(jacobian(&[], &[&x])));
    assert!(invalid(jacobian(&[&f], &[])));
    assert_eq!(jacobian(&[&f], &[&x, &y]).unwrap().shape(), (1, 2));
    assert!(invalid(hessian(&f, &[])));
    assert_eq!(hessian(&f, &[&x, &y]).unwrap().shape(), (2, 2));

    let (i, j) = (matrix![ctx, [1], [0], [0]], matrix![ctx, [0], [1], [0]]);
    assert_eq!(cross(&i, &j).unwrap(), matrix![ctx, [0], [0], [1]]);
    assert!(invalid(cross(&i, &matrix![ctx, [0], [1]])));
    assert!(invalid(cross(&matrix![ctx, [1, 0, 0]], &j)));
    assert_eq!(dot(&i, &i).unwrap(), ctx.int(1));
    assert!(invalid(dot(&i, &matrix![ctx, [0], [1]])));
    assert!(invalid(dot(
        &matrix![ctx, [1, 0, 0]],
        &matrix![ctx, [1, 0, 0]]
    )));

    let m = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    assert_eq!(
        m.submatrix(1..3, 0..2).unwrap(),
        matrix![ctx, [4, 5], [7, 8]]
    );
    assert!(invalid(m.submatrix(1..1, 0..2)));
    assert!(invalid(m.submatrix(0..2, 2..4)));
    #[allow(clippy::reversed_empty_ranges)]
    let reversed = m.submatrix(2..1, 0..1);
    assert!(invalid(reversed));
}

#[test]
fn vector_calculus_shapes_are_errors() {
    let ctx = Context::new();
    let (x, y, z, t) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("z"),
        ctx.symbol("t"),
    );
    let f = &x * &y;
    // An empty variable list reached `vars[0]` (an index panic the ratchet did not count).
    assert!(invalid(CoordinateSystem::Cartesian.scale_factors(&[])));
    assert!(invalid(gradient(&f, &[])));
    assert!(invalid(laplacian(&f, &[])));
    assert!(invalid(gradient_in(
        &f,
        &[&x, &y],
        CoordinateSystem::Spherical
    )));
    assert!(invalid(CoordinateSystem::Cylindrical.scale_factors(&[&x])));

    let field2 = Matrix::col_vector(vec![y.clone(), x.clone()]).unwrap();
    assert!(invalid(divergence(&field2, &[&x, &y, &z])));
    assert!(invalid(curl(&field2, &[&x, &y])));
    assert!(invalid(directional_derivative(&f, &[&x], &field2)));
    assert_eq!(divergence(&field2, &[&x, &y]).unwrap(), ctx.int(0));
    // The three-valued classifiers keep answering `Some(false)` for a bad shape.
    assert_eq!(is_conservative(&field2, &[&x, &y]), Some(false));
    assert_eq!(is_solenoidal(&field2, &[&x, &y, &z]), Some(false));

    let seg = [&t * 3, &t * 4];
    let (zero, one) = (ctx.int(0), ctx.int(1));
    assert!(invalid(line_integral_scalar(
        &one,
        &[],
        &[],
        &t,
        &zero,
        &one
    )));
    assert!(invalid(line_integral_scalar(
        &one,
        &[&x, &y],
        &seg[..1],
        &t,
        &zero,
        &one
    )));
    assert!(invalid(line_integral_vector(
        &field2,
        &[&x, &y],
        &seg[..1],
        &t,
        &zero,
        &one
    )));
    let len = line_integral_scalar(&one, &[&x, &y], &seg, &t, &zero, &one).unwrap();
    assert_eq!(len.simplify(), ctx.int(5));
}

// ── Batch 4: assumptions ──

#[test]
fn symbol_with_reports_contradictions_by_name() {
    let ctx = Context::new();
    let r = ctx.symbol_with("u", &[Assumption::Positive, Assumption::Negative]);
    assert_eq!(
        contradictory(&r),
        Some(("u".into(), "positive".into(), "negative".into()))
    );
    assert_eq!(
        r.unwrap_err().to_string(),
        "contradictory assumptions for 'u': cannot be both positive and negative"
    );
    let r = ctx.symbol_with(
        "n",
        &[
            Assumption::Real,
            Assumption::Integer,
            Assumption::Irrational,
        ],
    );
    assert_eq!(
        contradictory(&r),
        Some(("n".into(), "integer".into(), "irrational".into()))
    );
    // No single earlier assumption clashes with `NonZero`: the derived fact is named.
    let r = ctx.symbol_with(
        "z",
        &[
            Assumption::NonNegative,
            Assumption::NonPositive,
            Assumption::NonZero,
        ],
    );
    assert_eq!(
        contradictory(&r),
        Some(("z".into(), "zero".into(), "nonzero".into()))
    );
    // Declared finiteness is respected: `[Positive, Infinite]` describes `+oo`.
    let w = ctx
        .symbol_with("w", &[Assumption::Positive, Assumption::Infinite])
        .expect("+oo-like declaration is consistent");
    assert_eq!(w.is_positive(), Some(true));
    // Empty name.
    assert!(invalid(ctx.symbol_with("", &[Assumption::Real])));
}

#[test]
fn a_refused_declaration_changes_nothing() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    assert!(
        ctx.symbol_with("x", &[Assumption::Integer, Assumption::Irrational])
            .is_err()
    );
    assert_eq!(x.is_positive(), Some(true));
    let r = x.clone().assume(Assumption::Negative);
    let (symbol, _, b) = contradictory(&r).expect("contradiction");
    assert_eq!((symbol.as_str(), b.as_str()), ("x", "negative"));
    assert_eq!(x.is_positive(), Some(true));
    // Adding a consistent fact still works.
    let x = x.assume(Assumption::Integer).unwrap();
    assert_eq!(x.is_integer(), Some(true));
}

#[test]
fn refine_with_reports_contradictions_and_leaves_the_context_alone() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol_with("y", &[Assumption::Negative]).unwrap();
    let e = &x.abs() + &y.abs();
    // Between two hypotheses on one symbol, and against a stored assumption.
    let r = e.refine_with(&[(&x, Assumption::Positive), (&x, Assumption::Negative)]);
    assert_eq!(contradictory(&r).map(|c| c.0), Some("x".to_string()));
    let r = e.refine_with(&[(&x, Assumption::Positive), (&y, Assumption::Positive)]);
    assert_eq!(contradictory(&r).map(|c| c.0), Some("y".to_string()));
    assert_eq!(x.is_positive(), None);
    assert_eq!(y.is_negative(), Some(true));
    assert_eq!(
        e.refine_with(&[(&x, Assumption::Positive)]).unwrap(),
        &x - &y
    );
    assert_eq!(x.is_positive(), None);
}

fn declare_with_sym_macro(ctx: &Context) -> Result<Ex, SymplexError> {
    sym!(ctx; t, Positive, Negative);
    Ok(t)
}

#[test]
fn sym_macro_propagates_a_contradiction() {
    let ctx = Context::new();
    let r = declare_with_sym_macro(&ctx);
    assert!(contradictory(&r).is_some());
}

// ── Batch 5: MultiPoly exponent arithmetic ──

#[test]
fn monomial_products_use_checked_arithmetic() {
    assert_eq!(monomial_mul(&[2, 1], &[1, 3]), Some(vec![3, 4]));
    assert_eq!(monomial_mul(&[u32::MAX, 0], &[1, 0]), None);
    assert_eq!(monomial_mul(&[1, 2], &[1]), None);

    let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    let p = x.add(&y);
    assert_eq!(
        p.try_mul_monomial(&qi(3), &[1, 1]),
        Some(p.mul(&x.mul(&y)).scale(&qi(3)))
    );
    assert_eq!(p.try_mul_monomial(&qi(1), &[u32::MAX, 0]), None);
    assert_eq!(p.try_mul_monomial(&qi(1), &[1]), None);
    assert_eq!(
        p.try_mul_monomial(&qi(0), &[1, 0]),
        Some(MultiPoly::zero(2))
    );
    assert_eq!(p.mul_monomial(&qi(2), &[0, 1]), p.mul(&y).scale(&qi(2)));

    // Ring mismatches: `None` from the fallible forms (`div_exact` panicked).
    let z3 = MultiPoly::<GrevLex>::var(3, 0);
    assert_eq!(x.try_add(&z3), None);
    assert_eq!(x.try_sub(&z3), None);
    assert_eq!(x.try_mul(&z3), None);
    assert_eq!(x.div_exact(&z3), None);
    assert_eq!(x.try_add(&y), Some(p.clone()));
}

/// Before 0.29 the exponent wrapped around in release builds: `x · x^MAX`
/// became `x^0`.  It now fails exactly like `mul`, in every build.
#[test]
#[should_panic(expected = "exponent overflow")]
fn mul_monomial_overflow_panics_like_mul() {
    let x = MultiPoly::<GrevLex>::var(1, 0);
    let _ = x.mul_monomial(&qi(1), &[u32::MAX]);
}

/// `reduce` multiplies a divisor by a monomial: in lex order a trailing term
/// of the divisor (`y²`) can overflow where the leading one does not.  In
/// release builds the remainder used to be silently wrong.
#[test]
#[should_panic(expected = "exponent overflow")]
fn reduce_overflow_panics_instead_of_wrapping() {
    let f = MultiPoly::<Lex>::monomial(qi(1), vec![1, u32::MAX]); // x·y^MAX
    let d = MultiPoly::<Lex>::var(2, 0).add(&MultiPoly::monomial(qi(1), vec![0, 2])); // x + y²
    let _ = f.reduce(&[&d]);
}

#[test]
fn div_exact_reports_overflow_as_not_dividing() {
    // Exact quotients never exceed the dividend's exponents, so an overflow
    // on the way means "does not divide".
    let f = MultiPoly::<Lex>::monomial(qi(1), vec![1, u32::MAX]);
    let d = MultiPoly::<Lex>::var(2, 0).add(&MultiPoly::monomial(qi(1), vec![0, 2]));
    assert_eq!(f.div_exact(&d), None);
    let x = MultiPoly::<Lex>::var(2, 0);
    assert_eq!(
        f.div_exact(&x),
        Some(MultiPoly::monomial(qi(1), vec![0, u32::MAX]))
    );
}

// ── Batch 6: Ex::try_replace ──

#[test]
fn try_replace_reports_a_foreign_expression() {
    let (ctx, other) = (Context::new(), Context::new());
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let expr = &x.powi(2) + &x;
    let same = expr
        .try_replace(|e| (e == x).then(|| y.clone()))
        .expect("same context");
    assert_eq!(same, expr.replace(|e| (e == x).then(|| y.clone())));
    let foreign = other.symbol("y");
    assert!(invalid(
        expr.try_replace(|e| (e == x).then(|| foreign.clone()))
    ));
    // `self` is untouched and still usable.
    assert_eq!(expr, &x.powi(2) + &x);
}
