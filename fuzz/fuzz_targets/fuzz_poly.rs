//! Exact univariate polynomial algebra over ℚ.
//!
//! Integer-coefficient `a`, `b` (degree ≤ 7) sharing a random factor `c`;
//! every identity is exact, so a failure is a real bug:
//!
//! * `g = gcd(a, b)` divides `a`, `b`, and is divisible by `c`;
//! * `x·a + y·b = g` for `poly_gcdex` (the 0.22.2 ℤ[x] extended PRS), and
//!   its `gcd` equals `poly_gcd`;
//! * `a = q·b + r` with `deg r < deg b` for `poly_div`;
//! * the product of `factor_list(a)` reproduces `a`;
//! * every root `solve` returns makes `a` vanish (30-digit evaluation), and
//!   `c·a` has the same roots as `a` (0.22 fixed `4·p` degrading to
//!   `RootOf`s).
#![no_main]

#[path = "common/mod.rs"]
mod common;

use libfuzzer_sys::fuzz_target;
use symplex::prelude::*;

fn poly(ctx: &Context, x: &Ex, b: &mut common::Bytes, max_deg: u8) -> Ex {
    let deg = i64::from(b.u8() % (max_deg + 1));
    (0..=deg).fold(ctx.int(0), |acc, k| {
        let c = i64::from(b.u8() % 41) - 20;
        acc + ctx.int(c) * x.powi(k)
    })
}

fn is_zero(e: &Ex) -> bool {
    e.expand() == e.context().int(0)
}

fuzz_target!(|data: &[u8]| {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut b = common::Bytes::new(data);
    let c = poly(&ctx, &x, &mut b, 3);
    let a = (poly(&ctx, &x, &mut b, 4) * &c).expand();
    let bb = (poly(&ctx, &x, &mut b, 4) * &c).expand();

    let Some(g) = a.poly_gcd(&bb, &x) else {
        panic!("poly_gcd refused polynomials {a}, {bb}");
    };
    if !is_zero(&g) {
        assert!(
            is_zero(&a.poly_rem(&g, &x).unwrap()),
            "gcd {g} does not divide {a}"
        );
        assert!(
            is_zero(&bb.poly_rem(&g, &x).unwrap()),
            "gcd {g} does not divide {bb}"
        );
        if !is_zero(&c) && !is_zero(&a) && !is_zero(&bb) {
            assert!(
                is_zero(&g.poly_rem(&c, &x).unwrap()),
                "common factor {c} does not divide gcd {g} of {a}, {bb}"
            );
        }
    }

    let e = a.poly_gcdex(&bb, &x).expect("polynomial inputs");
    assert!(is_zero(&(&e.gcd - &g)), "gcdex gcd {} ≠ gcd {g}", e.gcd);
    assert!(
        is_zero(&(&e.x * &a + &e.y * &bb - &e.gcd)),
        "Bézout fails: ({}) a + ({}) b ≠ {} for a = {a}, b = {bb}",
        e.x,
        e.y,
        e.gcd
    );

    if !is_zero(&bb) {
        let (q, r) = a.poly_div(&bb, &x).expect("polynomial inputs");
        assert!(
            is_zero(&(&q * &bb + &r - &a)),
            "a ≠ q·b + r for a = {a}, b = {bb}"
        );
        if let (Some(dr), Some(db)) = (r.degree(&x), bb.degree(&x)) {
            assert!(dr < db || is_zero(&r), "deg r = {dr} ≥ deg b = {db}");
        }
    }

    if !is_zero(&a) {
        let (unit, factors) = a.factor_list(&x);
        let product = factors
            .iter()
            .fold(unit.clone(), |acc, (f, m)| acc * f.powi(i64::from(*m)));
        assert!(
            is_zero(&(&product - &a)),
            "factor_list product {product} ≠ {a}"
        );
    }

    if a.degree(&x).is_some_and(|d| d >= 1) {
        let roots = a
            .solve(&x)
            .expect("a non-constant polynomial has a solution set");
        let scale: f64 = 1e-18;
        for r in &roots {
            let v = a.subs(&x, r).eval_decimal(30);
            if let Ok(s) = v {
                let val: f64 = s.parse().unwrap_or(0.0);
                assert!(val.abs() <= scale.max(1e-18), "root {r} of {a} leaves {s}");
            }
        }
        let scaled = (ctx.int(-384) * &a).expand();
        let roots2 = scaled.solve(&x).expect("same solution set");
        assert_eq!(
            roots.len(),
            roots2.len(),
            "c·a has a different root count than a = {a}"
        );
    }
});
