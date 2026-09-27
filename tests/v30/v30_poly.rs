//! Polynomial algebra after 0.30: the parametric discriminant.

use symplex::prelude::*;

/// `discriminant_symbolic` takes `res(f, f′)/lc(f)` from the subresultant
/// PRS in `ℚ[params][x]` (0.31).  Up to 0.30 it took the Berkowitz
/// determinant of the Sylvester matrix over `Ex`, which failed (`None`) on
/// a dense degree-8 polynomial in three parameters and took 1.2 s (release)
/// at degree 6.  The trinomial has a closed form.
///
/// SymPy: `discriminant(x**n + a*x + b, x)` for n = 7, 8, 9 →
/// `-46656*a**7 - 823543*b**6`, `-823543*a**8 + 16777216*b**7`,
/// `16777216*a**9 + 387420489*b**8`.
#[test]
fn discriminant_of_trinomials_matches_the_closed_form() {
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    let cases: [(i64, i64, i64, i64, i64); 3] = [
        (7, -46656, 7, -823_543, 6),
        (8, -823_543, 8, 16_777_216, 7),
        (9, 16_777_216, 9, 387_420_489, 8),
    ];
    for (n, ca, ea, cb, eb) in cases {
        let f = &x.powi(n) + &(&a * &x) + &b;
        let want = &(&a.powi(ea) * ca) + &(&b.powi(eb) * cb);
        assert_eq!(f.discriminant_symbolic(&x).unwrap(), want, "n = {n}");
    }
}

/// A dense polynomial with three parameters: the symbolic discriminant at
/// rational parameter values equals the exact discriminant of the
/// polynomial with those values substituted (`Ex::discriminant`, the
/// rational resultant over ℤ), at several points.
#[test]
fn dense_parametric_discriminant_agrees_with_rational_discriminants() {
    let ctx = Context::new();
    let (x, a, b, c) = (
        ctx.symbol("x"),
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("c"),
    );
    let n = 6i64;
    let mut f = &x.powi(n) * &a;
    for k in 0..n {
        let coeff = &(&(&b * (k + 1)) - &c.powi(k % 3)) + &(&a * k);
        f = &f + &(&x.powi(k) * &coeff);
    }
    let disc = f.discriminant_symbolic(&x).unwrap();
    for (pa, pb, pc) in [
        ((2, 3), (-5, 7), (3, 1)),
        ((-1, 1), (1, 2), (-4, 5)),
        ((7, 2), (0, 1), (1, 9)),
    ] {
        let subs = [
            (&a, ctx.rational(pa.0, pa.1)),
            (&b, ctx.rational(pb.0, pb.1)),
            (&c, ctx.rational(pc.0, pc.1)),
        ];
        let mut at = disc.clone();
        let mut g = f.clone();
        for (s, v) in &subs {
            at = at.subs(s, v);
            g = g.subs(s, v);
        }
        let want = g.expand().discriminant(&x).unwrap();
        assert_eq!(at.expand(), want, "at a={pa:?}, b={pb:?}, c={pc:?}");
    }
}
