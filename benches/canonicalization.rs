//! Benchmarks for core symplex operations.
//!
//! Run with: `cargo bench`

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use symplex::prelude::*;

fn bench_canonicalization(c: &mut Criterion) {
    let ctx = Context::new();

    c.bench_function("canon_add_10_terms", |b| {
        let x = ctx.symbol("x");
        let terms: Vec<Ex> = (1..=10).map(|n| &x * n).collect();
        b.iter(|| {
            let mut sum = terms[0].clone();
            for t in &terms[1..] {
                sum = &sum + t;
            }
            sum
        });
    });

    c.bench_function("canon_add_100_terms", |b| {
        let x = ctx.symbol("x");
        let terms: Vec<Ex> = (1..=100).map(|n| &x.powi(n) * n).collect();
        b.iter(|| Ex::sum_of(&ctx, terms.clone()));
    });

    c.bench_function("canon_polynomial_x2_2x_1", |b| {
        let x = ctx.symbol("x");
        b.iter(|| &x.powi(2) + &x * 2 + 1);
    });
}

fn bench_diff(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    c.bench_function("diff_x3", |b| {
        let expr = x.powi(3);
        b.iter(|| expr.diff(&x));
    });

    c.bench_function("diff_sin_x2", |b| {
        let expr = x.powi(2).sin();
        b.iter(|| expr.diff(&x));
    });

    c.bench_function("diff_polynomial_deg10", |b| {
        let terms: Vec<Ex> = (0..=10).map(|n| &x.powi(n) * (n + 1) as i64).collect();
        let poly = Ex::sum_of(&ctx, terms);
        b.iter(|| poly.diff(&x));
    });
}

fn bench_expand(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    c.bench_function("expand_(x+1)^5", |b| {
        let expr = (&x + 1).powi(5);
        b.iter(|| expr.expand());
    });

    c.bench_function("expand_(x+1)^10", |b| {
        let expr = (&x + 1).powi(10);
        b.iter(|| expr.expand());
    });

    c.bench_function("expand_(x+y)^5", |b| {
        let y = ctx.symbol("y");
        let expr = (&x + &y).powi(5);
        b.iter(|| expr.expand());
    });
}

fn bench_simplify(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    c.bench_function("simplify_sin2_cos2", |b| {
        let expr = &x.sin().powi(2) + &x.cos().powi(2);
        b.iter(|| expr.simplify());
    });

    c.bench_function("simplify_exp_ln", |b| {
        let expr = x.ln().exp();
        b.iter(|| expr.simplify());
    });

    c.bench_function("full_simplify_sin2_cos2_plus_3", |b| {
        let expr = &x.sin().powi(2) + &x.cos().powi(2) + 3;
        b.iter(|| expr.full_simplify());
    });
}

fn bench_solve(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    c.bench_function("solve_linear", |b| {
        let expr = &x * 2 - 6;
        b.iter(|| expr.solve(&x));
    });

    c.bench_function("solve_quadratic", |b| {
        let expr = &x.powi(2) - &x * 5 + 6;
        b.iter(|| expr.solve(&x));
    });

    c.bench_function("solve_cubic", |b| {
        let expr = &x.powi(3) - &x.powi(2) * 6 + &x * 11 - 6;
        b.iter(|| expr.solve(&x));
    });
}

fn bench_integrate(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    c.bench_function("integrate_x3", |b| {
        let expr = x.powi(3);
        b.iter(|| expr.integrate(&x));
    });

    c.bench_function("integrate_sin_x", |b| {
        let expr = x.sin();
        b.iter(|| expr.integrate(&x));
    });

    c.bench_function("integrate_x_sin_x_by_parts", |b| {
        let expr = &x * &x.sin();
        b.iter(|| expr.integrate(&x));
    });
}

fn bench_evalf(c: &mut Criterion) {
    let ctx = Context::new();

    c.bench_function("evalf_pi_50_digits", |b| {
        let pi = ctx.pi();
        b.iter(|| pi.evalf(50));
    });

    c.bench_function("evalf_sqrt2_50_digits", |b| {
        let expr = ctx.int(2).sqrt();
        b.iter(|| expr.evalf(50));
    });

    c.bench_function("evalf_sin_pi_over_7_50_digits", |b| {
        let x = &ctx.pi() / &ctx.int(7);
        let expr = x.sin();
        b.iter(|| expr.evalf(50));
    });
}

fn bench_series(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    c.bench_function("series_sin_order_5", |b| {
        let expr = x.sin();
        b.iter(|| expr.maclaurin(&x, 5));
    });

    c.bench_function("series_exp_order_5", |b| {
        let expr = x.exp();
        b.iter(|| expr.maclaurin(&x, 5));
    });

    c.bench_function("series_exp_order_10", |b| {
        let expr = x.exp();
        b.iter(|| expr.maclaurin(&x, 10));
    });
}

fn bench_parse(c: &mut Criterion) {
    let ctx = Context::new();

    c.bench_function("parse_polynomial", |b| {
        b.iter(|| symplex::parse::parse(&ctx, "x^3 + 2*x^2 + 3*x + 4"));
    });

    c.bench_function("parse_trig_expression", |b| {
        b.iter(|| symplex::parse::parse(&ctx, "sin(x)^2 + cos(x)^2"));
    });
}

fn bench_factor(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    c.bench_function("factor_x2_minus_1", |b| {
        let expr = &x.powi(2) - 1;
        b.iter(|| expr.factor(&x));
    });

    c.bench_function("factor_cubic", |b| {
        let expr = &x.powi(3) - &x.powi(2) * 6 + &x * 11 - 6;
        b.iter(|| expr.factor(&x));
    });
}

fn bench_limit(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    c.bench_function("limit_sin_x_over_x", |b| {
        let expr = &x.sin() / &x;
        let zero = ctx.int(0);
        b.iter(|| expr.limit(&x, &zero));
    });

    c.bench_function("limit_polynomial_direct", |b| {
        let expr = &x.powi(2) + &x + 1;
        let two = ctx.int(2);
        b.iter(|| expr.limit(&x, &two));
    });
}

fn bench_integrate_polynomial(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(5);
    c.bench_function("integrate_x5", |b| {
        b.iter(|| {
            let _ = black_box(&expr).integrate(black_box(&x));
        })
    });
}

fn bench_simplify_sin_cos_ratio(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin() / &x.cos();
    c.bench_function("simplify_sin_div_cos", |b| {
        b.iter(|| {
            let _ = black_box(&expr).simplify();
        })
    });
}

fn bench_eval_unit_circle(c: &mut Criterion) {
    let ctx = Context::new();
    let angles: Vec<_> = (0..12)
        .map(|k| {
            let coeff = ctx.rational(k, 6);
            (&coeff * &ctx.pi()).sin()
        })
        .collect();
    c.bench_function("eval_unit_circle_12", |b| {
        b.iter(|| {
            for angle in &angles {
                let _ = black_box(angle).eval();
            }
        })
    });
}

fn bench_logcombine(c: &mut Criterion) {
    let ctx = Context::new();
    let vars: Vec<_> = ["a", "b", "c", "d", "e"]
        .iter()
        .map(|n| ctx.symbol(n).ln())
        .collect();
    let sum = &(&(&vars[0] + &vars[1]) + &vars[2]) + &(&vars[3] + &vars[4]);
    c.bench_function("logcombine_5_terms", |b| {
        b.iter(|| {
            let _ = black_box(&sum).logcombine();
        })
    });
}

fn bench_full_simplify(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x.sin().powi(2) + &x.cos().powi(2)) + &(&x.exp() * &(-&x).exp());
    c.bench_function("full_simplify_trig_exp", |b| {
        b.iter(|| {
            let _ = black_box(&expr).full_simplify();
        })
    });
}

criterion_group!(
    benches,
    bench_canonicalization,
    bench_diff,
    bench_expand,
    bench_simplify,
    bench_solve,
    bench_integrate,
    bench_evalf,
    bench_series,
    bench_parse,
    bench_factor,
    bench_limit,
    bench_integrate_polynomial,
    bench_simplify_sin_cos_ratio,
    bench_eval_unit_circle,
    bench_logcombine,
    bench_full_simplify,
);
criterion_main!(benches);
