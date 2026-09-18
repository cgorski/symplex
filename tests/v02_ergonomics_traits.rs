//! v0.2 ergonomics — standard-library traits on `Ex`.
//!
//! Operators with Rust scalars on either side, compound assignment,
//! `Sum` / `Product` (including the empty-iterator contract), `Debug`,
//! and the `ToEx` trait.

use num_bigint::BigInt;
use num_rational::Ratio;
use symplex::eq::ToEx;
use symplex::prelude::*;

fn r(p: i64, q: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(p), BigInt::from(q))
}

// ── f64 operands (exact) ───────────────────────────────────────────────

#[test]
fn f64_rhs_operators_are_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(&x * 0.5, &x * &ctx.rational(1, 2));
    assert_eq!(&x + 0.25, &x + &ctx.rational(1, 4));
    assert_eq!(&x - 1.5, &x - &ctx.rational(3, 2));
    assert_eq!(&x / 4.0, &x / &ctx.int(4));
    assert_eq!(x.clone() * 2.0, &x * 2);
}

#[test]
fn f64_lhs_operators() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(2.5 + &x, &x + &ctx.rational(5, 2));
    assert_eq!(1.0 - &x, &ctx.one() - &x);
    assert_eq!(0.5 * x.clone(), &x * &ctx.rational(1, 2));
    assert_eq!(1.0 / &x, x.powi(-1));
}

#[test]
fn f64_tenth_is_dyadic_in_operators() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x * 0.1;
    assert_ne!(e, &x * &ctx.rational(1, 10));
    assert_eq!(e, &x * &ctx.from_f64(0.1).unwrap());
}

#[test]
fn f64_nan_and_infinity_operands() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(&x + f64::NAN, ctx.nan());
    assert_eq!(&x * f64::NAN, ctx.nan());
    assert_eq!(&x + f64::INFINITY, &x + &ctx.infinity());
    assert_eq!(f64::NEG_INFINITY * &x, &ctx.neg_infinity() * &x);
}

// ── integer operands ───────────────────────────────────────────────────

#[test]
fn unsuffixed_integer_literals_still_infer() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // These must compile without type annotations (i32 fallback on the
    // right, i64 on the left) and produce the same nodes.
    let a = &x + 1;
    let b = 1 + &x;
    let c = x.clone() * 3 - 2;
    let d = 3 * &x - 2;
    assert_eq!(a, b);
    assert_eq!(c, d);
    assert_eq!(format!("{}", a.simplify()), "x + 1");
}

#[test]
fn typed_integer_operands() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(&x + 5i32, &x + 5);
    assert_eq!(&x + 5u32, &x + 5);
    assert_eq!(&x + 5u64, &x + 5);
    assert_eq!(&x + 5i64, &x + 5);
    assert_eq!(&x + 5i128, &x + 5);
    assert_eq!(&x * u64::MAX, &x * &ctx.from_u64(u64::MAX));
    assert_eq!(&x * i128::MIN, &x * &ctx.from_i128(i128::MIN));
}

#[test]
fn bigint_and_ratio_operands_both_sides() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let big = BigInt::from(10).pow(25);
    assert_eq!(&x + big.clone(), &x + &ctx.from_bigint(big.clone()));
    assert_eq!(big.clone() * &x, &ctx.from_bigint(big.clone()) * &x);
    assert_eq!(big.clone() - x.clone(), &ctx.from_bigint(big) - &x);
    assert_eq!(&x * r(1, 3), &x / 3);
    assert_eq!(r(1, 3) / &x, &ctx.rational(1, 3) / &x);
    assert_eq!(r(5, 1) + &x, &x + 5);
}

#[test]
fn integer_division_gives_exact_rational() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!((&x * 3) / 6, &x * &ctx.rational(1, 2));
    assert_eq!(ctx.int(1) / 3, ctx.rational(1, 3));
}

// ── compound assignment ────────────────────────────────────────────────

#[test]
fn compound_assignment_with_scalars() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut e = x.clone();
    e += 2;
    assert_eq!(e, &x + 2);
    e *= 3;
    assert_eq!(e, (&x + 2) * 3);
    e -= 0.5;
    assert_eq!(e, (&x + 2) * 3 - &ctx.rational(1, 2));
    e /= 2i64;
    assert_eq!(e, ((&x + 2) * 3 - &ctx.rational(1, 2)) / 2);
}

#[test]
fn compound_assignment_with_expressions() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let mut e = ctx.zero();
    e += &x;
    e += y.clone();
    assert_eq!(e, &x + &y);
    e *= &x;
    assert_eq!(e, &(&x + &y) * &x);
    e -= x.clone();
    e /= &y;
    assert_eq!(e, (&(&x + &y) * &x - &x) / &y);
}

#[test]
fn accumulate_in_loop_with_add_assign() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut poly = ctx.zero();
    for k in 0..4 {
        poly += x.powi(k) * (k + 1);
    }
    assert_eq!(poly, &x.powi(3) * 4 + &x.powi(2) * 3 + &x * 2 + 1);
}

// ── Sum / Product ──────────────────────────────────────────────────────

#[test]
fn sum_and_product_of_owned_and_borrowed() {
    let ctx = Context::new();
    let v: Vec<Ex> = (1..=5).map(|n| ctx.int(n)).collect();
    assert_eq!(v.iter().sum::<Ex>(), ctx.int(15));
    assert_eq!(v.iter().product::<Ex>(), ctx.int(120));
    assert_eq!(v.clone().into_iter().sum::<Ex>(), ctx.int(15));
    assert_eq!(v.into_iter().product::<Ex>(), ctx.int(120));
}

#[test]
fn sum_of_symbols_canonicalises() {
    let ctx = Context::new();
    let xs = ctx.symbols_indexed("x", 3);
    let s: Ex = xs.iter().sum();
    assert_eq!(s, &(&xs[0] + &xs[1]) + &xs[2]);
    let twice: Ex = xs.iter().chain(xs.iter()).sum();
    assert_eq!(twice, &s * 2);
}

#[test]
#[should_panic(expected = "empty iterator")]
fn sum_of_empty_panics_immediately() {
    let _: Ex = std::iter::empty::<Ex>().sum();
}

#[test]
#[should_panic(expected = "empty iterator")]
fn product_of_empty_panics_immediately() {
    let _: Ex = std::iter::empty::<Ex>().product();
}

#[test]
fn option_sum_product_none_on_empty() {
    let s: Option<Ex> = std::iter::empty::<Ex>().sum();
    let p: Option<Ex> = std::iter::empty::<Ex>().product();
    assert!(s.is_none());
    assert!(p.is_none());
}

#[test]
fn option_sum_product_some_on_values() {
    let ctx = Context::new();
    let v: Vec<Ex> = (1..=3).map(|n| ctx.int(n)).collect();
    let s: Option<Ex> = v.iter().sum();
    let p: Option<Ex> = v.into_iter().product();
    assert_eq!(s, Some(ctx.int(6)));
    assert_eq!(p, Some(ctx.int(6)));
}

#[test]
fn option_sum_result_mixes_with_context_without_panicking() {
    // The old empty-iterator behaviour created a fresh Context, which made
    // the result unusable with the caller's expressions. `None` / the
    // Context::sum fallback avoid that.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s: Option<Ex> = Vec::<Ex>::new().into_iter().sum();
    let total = s.unwrap_or_else(|| ctx.zero()) + &x;
    assert_eq!(total, x);
    assert_eq!(ctx.sum(Vec::<Ex>::new()) + &x, x);
}

#[test]
#[should_panic(expected = "different contexts")]
fn sum_across_contexts_panics() {
    let a = Context::new();
    let b = Context::new();
    let _: Ex = vec![a.int(1), b.int(2)].into_iter().sum();
}

// ── Debug ──────────────────────────────────────────────────────────────

#[test]
fn debug_shows_pretty_form() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(format!("{:?}", x.powi(2) + 1), "Ex(x^2 + 1)");
    assert_eq!(format!("{:?}", ctx.rational(1, 2)), "Ex(1/2)");
    assert_eq!(format!("{:?}", x.gt(&ctx.zero())), "BoolEx(x > 0)");
    assert_eq!(format!("{:?}", ctx.empty_set()), "SetEx(EmptySet)");
}

#[test]
fn debug_in_option_and_vec() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(format!("{:?}", Some(x.clone())), "Some(Ex(x))");
    assert_eq!(
        format!("{:?}", vec![x.clone(), ctx.int(2)]),
        "[Ex(x), Ex(2)]"
    );
}

// ── ToEx ───────────────────────────────────────────────────────────────

#[test]
fn to_ex_for_scalars() {
    let ctx = Context::new();
    assert_eq!(3_i32.to_ex(&ctx), ctx.int(3));
    assert_eq!(3_i64.to_ex(&ctx), ctx.int(3));
    assert_eq!(3_u32.to_ex(&ctx), ctx.int(3));
    assert_eq!(3_u64.to_ex(&ctx), ctx.int(3));
    assert_eq!(3_i128.to_ex(&ctx), ctx.int(3));
    assert_eq!(0.75_f64.to_ex(&ctx), ctx.rational(3, 4));
    assert_eq!(BigInt::from(3).to_ex(&ctx), ctx.int(3));
    assert_eq!(r(3, 4).to_ex(&ctx), ctx.rational(3, 4));
}

#[test]
fn to_ex_for_expressions_is_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.to_ex(&ctx), x);
    let xr: &Ex = &x;
    assert_eq!(xr.to_ex(&ctx), x);
}

#[test]
fn to_ex_f64_special_values() {
    let ctx = Context::new();
    assert_eq!(f64::NAN.to_ex(&ctx), ctx.nan());
    assert_eq!(f64::INFINITY.to_ex(&ctx), ctx.infinity());
    assert_eq!(f64::NEG_INFINITY.to_ex(&ctx), ctx.neg_infinity());
}

#[test]
fn to_ex_is_usable_as_a_bound_in_user_code() {
    fn double<T: ToEx>(ctx: &Context, v: T) -> Ex {
        v.to_ex(ctx) * 2
    }
    let ctx = Context::new();
    assert_eq!(double(&ctx, 4), ctx.int(8));
    assert_eq!(double(&ctx, 0.5), ctx.one());
    assert_eq!(double(&ctx, ctx.symbol("x")), &ctx.symbol("x") * 2);
}

#[test]
fn to_ex_user_implementation() {
    struct Half;
    impl ToEx for Half {
        fn to_ex(&self, ctx: &Context) -> Ex {
            ctx.rational(1, 2)
        }
    }
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.eval_f64_with(&[(&x, Half)]).unwrap(), 0.5);
}

#[test]
#[should_panic(expected = "different contexts")]
fn to_ex_cross_context_panics() {
    let a = Context::new();
    let b = Context::new();
    let _ = a.symbol("x").to_ex(&b);
}

// ── Extraction ─────────────────────────────────────────────────────────

#[test]
fn as_rational_as_bigint_as_i64() {
    let ctx = Context::new();
    assert_eq!(ctx.rational(-4, 6).as_rational(), Some(r(-2, 3)));
    assert_eq!(ctx.rational(-4, 6).as_bigint(), None);
    assert_eq!(ctx.int(9).as_bigint(), Some(BigInt::from(9)));
    assert_eq!(ctx.int(9).as_i64(), Some(9));
    assert_eq!(ctx.from_u64(u64::MAX).as_i64(), None);
    assert_eq!(ctx.symbol("x").as_rational(), None);
    assert_eq!(ctx.pi().as_rational(), None);
    assert_eq!((&ctx.int(6) / 3).as_i64(), Some(2));
}
