//! 0.18 track: the exact fast path of `Poly`.
//!
//! A `Poly` whose coefficients are all rational literals is stored as an
//! exact `MultiPoly<Lex>` and its arithmetic runs on rationals; a
//! symbolic coefficient switches it to one `Ex` per monomial.  Everything
//! observable — `terms()`, `Display`, `to_ex()`, `to_multipoly()`, the
//! doc examples — must be byte-identical between the two representations
//! and to the pre-0.18 behaviour, because `poly_ex` feeds the certificate
//! provers and the Lean printer.
//!
//! The review's ordering claim is pinned here: `Poly::terms()` is `Lex`
//! descending, which differs from the `GrevLex` order of the `MultiPoly`
//! that `to_multipoly()` returns (`x²z` before `xy²` in `Lex`, after it
//! in `GrevLex`).

use symplex::multipoly::{GrevLex, Lex, MonomialOrd, MultiPoly};
use symplex::num_bigint::BigInt;
use symplex::num_rational::Ratio;
use symplex::poly_ex::Poly;
use symplex::prelude::*;

fn q(p: i64, d: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(p), BigInt::from(d))
}

/// Every observable of two polynomials, side by side.
fn assert_same_view(a: &Poly, b: &Poly) {
    assert_eq!(a.terms(), b.terms(), "terms()\n a = {a}\n b = {b}");
    assert_eq!(a.to_string(), b.to_string(), "Display");
    assert_eq!(a.monoms(), b.monoms(), "monoms()");
    assert_eq!(a.coeffs(), b.coeffs(), "coeffs()");
    assert_eq!(
        a.coeffs_rational(),
        b.coeffs_rational(),
        "coeffs_rational()"
    );
    assert_eq!(a.all_coeffs(), b.all_coeffs(), "all_coeffs()");
    assert_eq!(a.leading_term(), b.leading_term(), "leading_term()");
    assert_eq!(a.leading_coeff(), b.leading_coeff(), "leading_coeff()");
    assert_eq!(
        a.leading_monomial(),
        b.leading_monomial(),
        "leading_monomial()"
    );
    assert_eq!(a.degree_list(), b.degree_list(), "degree_list()");
    assert_eq!(a.total_degree(), b.total_degree(), "total_degree()");
    assert_eq!(a.num_terms(), b.num_terms(), "num_terms()");
    assert_eq!(a.to_ex(), b.to_ex(), "to_ex()");
    assert_eq!(a.to_multipoly(), b.to_multipoly(), "to_multipoly()");
    assert!(a.equals(b), "equals()");
    let a_iter: Vec<(Vec<u32>, Ex)> = a
        .terms_iter()
        .map(|(m, c)| (m.to_vec(), c.clone()))
        .collect();
    assert_eq!(a_iter, a.terms(), "terms_iter() vs terms()");
}

// ═══════════════════════════════════════════════════════════════════════════
// The ordering claim
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn terms_order_is_lex_descending_and_differs_from_grevlex() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    // x²z and xy² have the same total degree; Lex puts x²z first, GrevLex
    // (compare the last variable, reversed) puts xy² first.
    let e = &x * &y.powi(2) + &x.powi(2) * &z + &y.powi(3) + &x * &y * &z + 1;
    let p = Poly::new(&e, &[&x, &y, &z]).unwrap();
    let monoms = p.monoms();
    assert_eq!(
        monoms,
        vec![
            vec![2, 0, 1],
            vec![1, 2, 0],
            vec![1, 1, 1],
            vec![0, 3, 0],
            vec![0, 0, 0]
        ]
    );

    let mut lex = monoms.clone();
    lex.sort_by(|a, b| Lex::cmp_exponents(b, a));
    assert_eq!(monoms, lex, "terms() is Lex descending");

    let mut grevlex = monoms.clone();
    grevlex.sort_by(|a, b| GrevLex::cmp_exponents(b, a));
    assert_ne!(monoms, grevlex, "…and is not GrevLex descending");
    assert_eq!(grevlex[0], vec![1, 2, 0]);
    assert_eq!(grevlex[2], vec![2, 0, 1]);

    // A MultiPoly<Lex> read from the top gives exactly terms().
    let mp: MultiPoly<GrevLex> = p.to_multipoly().unwrap();
    let mpl: MultiPoly<Lex> = mp.convert_order();
    let from_lex: Vec<Vec<u32>> = mpl.terms().rev().map(|(m, _)| m.to_vec()).collect();
    assert_eq!(monoms, from_lex);

    assert_eq!(
        p.to_string(),
        "Poly(x^2*z + x*y^2 + x*y*z + y^3 + 1, x, y, z)"
    );
}

#[test]
fn to_multipoly_still_returns_grevlex_with_the_same_content() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let e = &x * &y.powi(2) * 3 - &x.powi(2) * &z / 2 + &y.powi(3) + 7;
    let p = Poly::new(&e, &[&x, &y, &z]).unwrap();
    let mp: MultiPoly<GrevLex> = p.to_multipoly().unwrap();
    // GrevLex ascending: constant, then degree-3 terms in reverse-lex.
    let order: Vec<Vec<u32>> = mp.terms().map(|(m, _)| m.to_vec()).collect();
    assert_eq!(
        order,
        vec![vec![0, 0, 0], vec![2, 0, 1], vec![0, 3, 0], vec![1, 2, 0]]
    );
    assert_eq!(mp.coeff(&[1, 2, 0]), Some(&q(3, 1)));
    assert_eq!(mp.coeff(&[2, 0, 1]), Some(&q(-1, 2)));
    assert_eq!(mp.coeff(&[0, 0, 0]), Some(&q(7, 1)));
    assert_eq!(mp.num_terms(), 4);

    // Round trip through from_multipoly keeps terms() and Display.
    let back = Poly::from_multipoly(&ctx, &[&x, &y, &z], &mp).unwrap();
    assert_same_view(&p, &back);

    // Symbolic coefficients: still None.
    let a = ctx.symbol("a");
    assert!(
        Poly::new(&(&a * &x + 1), &[&x])
            .unwrap()
            .to_multipoly()
            .is_none()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact arithmetic vs Poly::new of the same polynomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn univariate_arithmetic_matches_construction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Zero coefficients (no x^2 term) and negative rationals.
    let f = Poly::new(&(&x.powi(3) * 2 - &x / 3 + 5), &[&x]).unwrap();
    let g = Poly::new(&(-&x.powi(2) * 7 + &x * ctx.rational(-3, 4) - 1), &[&x]).unwrap();
    assert!(f.has_rational_coeffs() && g.has_rational_coeffs());

    let fe = f.to_ex();
    let ge = g.to_ex();
    let cases: Vec<(Poly, Ex)> = vec![
        (f.add(&g).unwrap(), &fe + &ge),
        (f.sub(&g).unwrap(), &fe - &ge),
        (g.sub(&f).unwrap(), &ge - &fe),
        (f.mul(&g).unwrap(), &fe * &ge),
        (f.neg(), -&fe),
        (
            f.scale(&ctx.rational(-2, 9)).unwrap(),
            &fe * ctx.rational(-2, 9),
        ),
        (f.pow(4).unwrap(), fe.powi(4)),
        (g.pow(3).unwrap(), ge.powi(3)),
        (f.derivative(&x).unwrap(), fe.diff(&x)),
        (f.sub(&f).unwrap(), ctx.int(0)),
        (f.pow(0).unwrap(), ctx.int(1)),
    ];
    for (viaop, ex) in cases {
        let vianew = Poly::new(&ex, &[&x]).unwrap();
        assert_same_view(&viaop, &vianew);
    }
    assert_eq!(
        f.mul(&g).unwrap().to_string(),
        Poly::new(&(&fe * &ge), &[&x]).unwrap().to_string()
    );
    // Concrete expected values, so a wrong sign or a dropped zero would show.
    assert_eq!(
        f.add(&g).unwrap().to_string(),
        "Poly(2*x^3 - 7*x^2 - 13/12*x + 4, x)"
    );
    assert_eq!(
        f.derivative(&x).unwrap().all_coeffs().unwrap(),
        vec![ctx.int(6), ctx.int(0), ctx.rational(-1, 3)]
    );
}

#[test]
fn three_variable_arithmetic_matches_construction() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let gens = [&x, &y, &z];
    let f = Poly::new(
        &(&x * &y.powi(2) - &x.powi(2) * &z * ctx.rational(1, 2) + &y.powi(3) - 4),
        &gens,
    )
    .unwrap();
    let g = Poly::new(&(&x + &y * 3 - &z + ctx.rational(-5, 6)), &gens).unwrap();

    let fe = f.to_ex();
    let ge = g.to_ex();
    let cases: Vec<(Poly, Ex)> = vec![
        (f.add(&g).unwrap(), &fe + &ge),
        (f.sub(&g).unwrap(), &fe - &ge),
        (f.mul(&g).unwrap(), &fe * &ge),
        (g.mul(&g).unwrap(), ge.powi(2)),
        (g.pow(5).unwrap(), ge.powi(5)),
        (f.neg().add(&f).unwrap(), ctx.int(0)),
        (f.scale(&ctx.int(-3)).unwrap(), &fe * -3),
        (f.derivative(&x).unwrap(), fe.diff(&x)),
        (f.derivative(&y).unwrap(), fe.diff(&y)),
        (f.derivative(&z).unwrap(), fe.diff(&z)),
        // Cancellation to zero of a whole monomial: x·y² − x·y².
        (
            f.sub(&Poly::new(&(&x * &y.powi(2)), &gens).unwrap())
                .unwrap(),
            &fe - &x * &y.powi(2),
        ),
    ];
    for (viaop, ex) in cases {
        let vianew = Poly::new(&ex, &gens).unwrap();
        assert_same_view(&viaop, &vianew);
    }
    assert_eq!(f.to_string(), "Poly(-1/2*x^2*z + x*y^2 + y^3 - 4, x, y, z)");
    assert_eq!(
        f.mul(&g).unwrap().leading_term(),
        Some((vec![3, 0, 1], ctx.rational(-1, 2)))
    );
}

#[test]
fn power_of_a_sum_matches_expansion() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let s = Poly::new(&(&x + &y + &z + 1), &[&x, &y, &z]).unwrap();
    let p12 = s.pow(12).unwrap();
    assert_eq!(p12.num_terms(), 455);
    let vianew = Poly::new(&(&x + &y + &z + 1).powi(12), &[&x, &y, &z]).unwrap();
    assert_same_view(&p12, &vianew);
    // Multinomial coefficient 12!/(3!4!5!) = 27720 at x³y⁴z⁵.
    assert_eq!(p12.coeff_monomial(&[3, 4, 5]).unwrap(), ctx.int(27720));
}

#[test]
fn eval_exact_matches_substitution() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let s = Poly::new(&(&x + &y + &z + 1), &[&x, &y, &z]).unwrap();
    let p12 = s.pow(12).unwrap();
    let pt = [ctx.rational(3, 7), ctx.rational(-2, 5), ctx.rational(11, 3)];
    let v = p12.eval(&[&pt[0], &pt[1], &pt[2]]).unwrap();
    // (3/7 − 2/5 + 11/3 + 1)^12 = ((45 − 42 + 385 + 105)/105)^12 = (493/105)^12
    let expected = ctx.rational(493, 105).powi(12).eval();
    assert_eq!(v, expected);
    // Through the expression, the long way round.
    let sub = p12
        .to_ex()
        .subs(&x, &pt[0])
        .subs(&y, &pt[1])
        .subs(&z, &pt[2])
        .eval();
    assert_eq!(v, sub);
    // Integer point, mixed integer/rational point, zero polynomial.
    let f = Poly::new(&(&x.powi(2) * 3 - &y * &z + 1), &[&x, &y, &z]).unwrap();
    assert_eq!(
        f.eval(&[&ctx.int(2), &ctx.int(-3), &ctx.int(5)]).unwrap(),
        ctx.int(28)
    );
    assert_eq!(
        f.eval(&[&ctx.rational(1, 2), &ctx.int(4), &ctx.rational(-1, 8)])
            .unwrap(),
        ctx.rational(9, 4)
    );
    let zero = Poly::zero(&ctx, &[&x, &y, &z]).unwrap();
    assert_eq!(
        zero.eval(&[&ctx.int(1), &ctx.int(2), &ctx.int(3)]).unwrap(),
        ctx.int(0)
    );
    // A symbolic value still works (expression path).
    let a = ctx.symbol("a");
    assert_eq!(
        f.eval(&[&a, &ctx.int(1), &ctx.int(1)]).unwrap(),
        (&a.powi(2) * 3).eval()
    );
}

#[test]
fn eval_gen_exact_matches_symbolic_path() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let p = Poly::new(
        &(&x.powi(2) * &y * ctx.rational(1, 3) + &x - &y.powi(2) + 2),
        &[&x, &y],
    )
    .unwrap();
    let at = p.eval_gen(&x, &ctx.rational(3, 2)).unwrap();
    let expected = Poly::new(
        &(&y * ctx.rational(3, 4) + ctx.rational(3, 2) - &y.powi(2) + 2),
        &[&y],
    )
    .unwrap();
    assert_same_view(&at, &expected);
    assert_eq!(at.to_string(), "Poly(-y^2 + 3/4*y + 7/2, y)");
    // Last generator → ground polynomial with no generators.
    let ground = at.eval_gen(&y, &ctx.int(2)).unwrap();
    assert_eq!(ground.gens().len(), 0);
    assert_eq!(ground.terms(), vec![(vec![], ctx.int(1))]);
    // Polynomial value in the other generator (symbolic path) still works.
    let comp = p.eval_gen(&x, &(&y + 1)).unwrap();
    let comp_expected = Poly::new(
        &((&y + 1).powi(2) * &y / 3 + (&y + 1) - &y.powi(2) + 2),
        &[&y],
    )
    .unwrap();
    assert_same_view(&comp, &comp_expected);
}

// ═══════════════════════════════════════════════════════════════════════════
// Mixed exact × symbolic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mixed_exact_and_symbolic_arithmetic_equals_all_symbolic() {
    let ctx = Context::new();
    let (x, y, a) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a"));
    let gens = [&x, &y];
    let exact = Poly::new(&(&x.powi(2) * 3 - &x * &y / 2 + 1), &gens).unwrap();
    let sym = Poly::new(&(&a * &x - (&a + 1) * &y + 2), &gens).unwrap();
    assert!(exact.has_rational_coeffs());
    assert!(!sym.has_rational_coeffs());

    // The same polynomial as `exact`, but built through a symbolic detour
    // (a − a cancels), so it goes through the symbolic constructor.
    let exact_ex = exact.to_ex();
    let sym_ex = sym.to_ex();

    let check = |mixed: Poly, ex: Ex| {
        let all = Poly::new(&ex, &gens).unwrap();
        assert_same_view(&mixed, &all);
    };
    check(exact.add(&sym).unwrap(), &exact_ex + &sym_ex);
    check(sym.add(&exact).unwrap(), &sym_ex + &exact_ex);
    check(exact.sub(&sym).unwrap(), &exact_ex - &sym_ex);
    check(sym.sub(&exact).unwrap(), &sym_ex - &exact_ex);
    check(exact.mul(&sym).unwrap(), &exact_ex * &sym_ex);
    check(sym.mul(&exact).unwrap(), &sym_ex * &exact_ex);
    check(exact.scale(&(&a + 1)).unwrap(), &exact_ex * (&a + 1));
    check(
        sym.scale(&ctx.rational(2, 3)).unwrap(),
        &sym_ex * ctx.rational(2, 3),
    );
    check(sym.pow(2).unwrap(), sym_ex.powi(2));
    check(sym.derivative(&y).unwrap(), sym_ex.diff(&y));

    // Symbolic coefficients that cancel back to rationals compare equal to
    // the exact polynomial.
    let round_trip = exact.add(&sym).unwrap().sub(&sym).unwrap();
    assert!(round_trip.has_rational_coeffs());
    assert!(round_trip.equals(&exact));
    assert_same_view(&round_trip, &exact);
    assert_eq!(round_trip.to_multipoly(), exact.to_multipoly());

    // And a symbolic polynomial that is secretly rational after scaling by 0.
    let zero_sym = sym.scale(&ctx.int(0)).unwrap();
    assert!(zero_sym.is_zero());
    assert!(zero_sym.equals(&Poly::zero(&ctx, &gens).unwrap()));

    // Display of a mixed result: symbolic sums parenthesised as before.
    assert_eq!(
        exact.add(&sym).unwrap().to_string(),
        "Poly(3*x^2 - 1/2*x*y + a*x + (-a - 1)*y + 3, x, y)"
    );
}

#[test]
fn from_terms_and_constant_constructors_agree_with_new() {
    let ctx = Context::new();
    let (x, y, a) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a"));
    // Duplicates summed, zeros dropped — rational and symbolic.
    let p = Poly::from_terms(
        &ctx,
        &[&x, &y],
        vec![
            (vec![1, 0], ctx.int(2)),
            (vec![1, 0], ctx.int(-2)),
            (vec![0, 2], ctx.rational(-1, 3)),
            (vec![0, 0], ctx.int(0)),
        ],
    )
    .unwrap();
    assert_same_view(&p, &Poly::new(&(-&y.powi(2) / 3), &[&x, &y]).unwrap());
    let s = Poly::from_terms(
        &ctx,
        &[&x, &y],
        vec![
            (vec![1, 0], a.clone()),
            (vec![1, 0], -&a),
            (vec![0, 1], ctx.int(4)),
        ],
    )
    .unwrap();
    assert_same_view(&s, &Poly::new(&(&y * 4), &[&x, &y]).unwrap());
    assert!(s.has_rational_coeffs());

    assert_same_view(
        &Poly::one(&ctx, &[&x]).unwrap(),
        &Poly::new(&ctx.int(1), &[&x]).unwrap(),
    );
    assert_same_view(
        &Poly::constant(&ctx, &[&x], &ctx.rational(-7, 2)).unwrap(),
        &Poly::new(&ctx.rational(-7, 2), &[&x]).unwrap(),
    );
    assert_same_view(
        &Poly::zero(&ctx, &[&x, &y]).unwrap(),
        &Poly::new(&ctx.int(0), &[&x, &y]).unwrap(),
    );
    assert_eq!(Poly::zero(&ctx, &[&x]).unwrap().to_string(), "Poly(0, x)");
    // Errors unchanged.
    assert!(Poly::from_terms(&ctx, &[&x], vec![(vec![1, 1], ctx.int(1))]).is_err());
    assert!(Poly::from_terms(&ctx, &[&x], vec![(vec![1], x.clone())]).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Construction edge cases the fast path must not change
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn new_on_unexpanded_and_non_polynomial_input_is_unchanged() {
    let ctx = Context::new();
    let (x, y, a) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a"));
    // Unexpanded products and powers are read off the tree.
    let e = (&x - &y).powi(3) * (&x + 2) - (&x * &y - 1).powi(2);
    let p = Poly::new(&e, &[&x, &y]).unwrap();
    let q = Poly::new(&e.expand(), &[&x, &y]).unwrap();
    assert_same_view(&p, &q);
    assert_eq!(p.to_ex(), e.expand());
    // Huge symbolic-power coefficients stay expressions (not folded).
    let big = Poly::new(&(&x * ctx.int(2).pow(&ctx.int(100_000))), &[&x]).unwrap();
    assert!(!big.has_rational_coeffs());
    assert_eq!(big.num_terms(), 1);
    // Non-polynomial positions still rejected, with the same reasons.
    assert!(Poly::new(&x.sin(), &[&x]).is_none());
    assert!(Poly::new(&(&x * &y.powi(-1)), &[&x, &y]).is_none());
    assert!(Poly::new(&(&x * &y.powi(-1)), &[&x]).is_some());
    assert!(Poly::new(&x.pow(&a), &[&x]).is_none());
    let err = Poly::try_new(&(&a / (&x + 1)), &[&x])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("negative power (a rational function)"),
        "{err}"
    );
    // Zero from cancellation.
    let z = Poly::new(&(&x * &y - &y * &x), &[&x, &y]).unwrap();
    assert!(z.is_zero());
    assert_eq!(z.to_string(), "Poly(0, x, y)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Content / monic / MultiPoly bridge on the doc examples
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn content_primitive_monic_and_gcd_bridge_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = Poly::new(&(&x.powi(2) * -4 + &x * 6), &[&x]).unwrap();
    let (c, prim) = p.content_and_primitive().unwrap();
    assert_eq!(c, ctx.int(-2));
    assert_eq!(prim.to_ex(), &x.powi(2) * 2 - &x * 3);
    assert_eq!(prim.to_string(), "Poly(2*x^2 - 3*x, x)");
    assert_same_view(&prim, &Poly::new(&prim.to_ex(), &[&x]).unwrap());
    assert!(
        prim.mul(&Poly::constant(&ctx, &[&x], &c).unwrap())
            .unwrap()
            .equals(&p)
    );

    let m = p.monic().unwrap();
    assert_eq!(m.to_string(), "Poly(x^2 - 3/2*x, x)");
    assert_same_view(
        &m,
        &Poly::new(&(&x.powi(2) - &x * ctx.rational(3, 2)), &[&x]).unwrap(),
    );
    assert!(Poly::zero(&ctx, &[&x]).unwrap().monic().is_none());
    let a = ctx.symbol("a");
    assert!(Poly::new(&(&a * &x), &[&x]).unwrap().monic().is_none());
    assert!(
        Poly::new(&(&a * &x), &[&x])
            .unwrap()
            .content_and_primitive()
            .is_none()
    );

    // gcd of two degree-20 polynomials sharing a degree-10 factor, through
    // the MultiPoly bridge (Poly::mul builds the inputs exactly).
    let mk = |seed: i64, deg: i64| {
        let mut e = ctx.int(0);
        for k in 0..=deg {
            let c = (seed * 31 + k * 17) % 19 - 9;
            e += ctx.int(if c == 0 { 1 } else { c }) * x.powi(k);
        }
        Poly::new(&e, &[&x]).unwrap()
    };
    let g = mk(3, 10);
    let f1 = g.mul(&mk(4, 10)).unwrap();
    let f2 = g.mul(&mk(5, 10)).unwrap();
    assert_eq!(f1.total_degree(), Some(20));
    let gg = MultiPoly::gcd(&f1.to_multipoly().unwrap(), &f2.to_multipoly().unwrap());
    let gp = Poly::from_multipoly(&ctx, &[&x], &gg).unwrap();
    assert_eq!(gp.total_degree(), Some(10));
    // gcd is integer-normalised with positive leading coefficient: equal
    // to ±g up to content.
    let (_, g_prim) = g.content_and_primitive().unwrap();
    let (_, gp_prim) = gp.content_and_primitive().unwrap();
    assert_same_view(&g_prim, &gp_prim);

    // MultiPoly::to_ex round trip (the Lean printer's bridge).
    let (j, r) = (ctx.symbol("j"), ctx.symbol("r"));
    let mp: MultiPoly = MultiPoly::var(2, 0)
        .mul(&MultiPoly::var(2, 1))
        .scale(&q(2, 1))
        + 1;
    assert_eq!(
        mp.to_ex(&ctx, &[&j, &r]).unwrap().to_lean().unwrap(),
        "2 * j * r + 1"
    );
}

#[test]
fn coefficient_matrix_and_monomial_basis_unchanged() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let p1 = Poly::new(&(&x + 1), &[&x]).unwrap();
    let p2 = Poly::new(&(&x.powi(2) - 1), &[&x]).unwrap();
    let p3 = Poly::new(&(&a * &x.powi(3) + &x), &[&x]).unwrap();
    let basis = Poly::monomial_basis(&[&p1, &p2, &p3]).unwrap();
    assert_eq!(basis, vec![vec![3], vec![2], vec![1], vec![0]]);
    let m = Poly::coefficient_matrix(&[&p1, &p2, &p3], &basis).unwrap();
    assert_eq!(m.shape(), (4, 3));
    assert_eq!(*m.get(0, 2), a);
    assert_eq!(*m.get(1, 1), ctx.int(1));
    assert_eq!(*m.get(2, 0), ctx.int(1));
    assert_eq!(*m.get(3, 1), ctx.int(-1));
    assert_eq!(*m.get(3, 2), ctx.int(0));
}

#[test]
fn shift_and_root_queries_unchanged() {
    let ctx = Context::new();
    let j = ctx.symbol("j");
    let p = (&j.powi(2) - 4 * &j + 3).as_poly(&[&j]).unwrap();
    let shifted = p.shift(&j, &ctx.int(3)).unwrap();
    assert_eq!(shifted.to_string(), "Poly(j^2 + 2*j, j)");
    assert_eq!(p.count_real_roots(), Some(2));
    assert_eq!(
        p.is_nonnegative_on(&ctx.int(3), &ctx.infinity()),
        Some(true)
    );
    assert_eq!(p.is_positive_on(&ctx.int(3), &ctx.infinity()), Some(false));
    let roots = p.nroots(15).unwrap();
    assert_eq!(roots.len(), 2);
}
