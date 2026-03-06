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
        b.iter(|| pi.eval_decimal(50));
    });

    c.bench_function("evalf_sqrt2_50_digits", |b| {
        let expr = ctx.int(2).sqrt();
        b.iter(|| expr.eval_decimal(50));
    });

    c.bench_function("evalf_sin_pi_over_7_50_digits", |b| {
        let x = &ctx.pi() / &ctx.int(7);
        let expr = x.sin();
        b.iter(|| expr.eval_decimal(50));
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
            let _ = black_box(&sum).log_combine();
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

// ═══════════════════════════════════════════════════════════════════════════
// Large expression scaling
// ═══════════════════════════════════════════════════════════════════════════

fn bench_large_polynomial(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    for &n in &[100, 500, 1000] {
        let terms: Vec<Ex> = (1..=n).map(|i| &x.powi(i) * i).collect();
        let poly = Ex::sum_of(&ctx, terms);

        c.bench_function(&format!("diff_poly_deg{n}"), |b| {
            b.iter(|| black_box(&poly).diff(&x))
        });

        c.bench_function(&format!("subs_poly_deg{n}"), |b| {
            let two = ctx.int(2);
            b.iter(|| black_box(&poly).subs(&x, &two).eval())
        });

        c.bench_function(&format!("display_poly_deg{n}"), |b| {
            b.iter(|| format!("{}", black_box(&poly)))
        });

        c.bench_function(&format!("integrate_poly_deg{n}"), |b| {
            b.iter(|| black_box(&poly).integrate(&x))
        });
    }
}

fn bench_nested_functions(c: &mut Criterion) {
    let x = symplex::var("x");

    for &depth in &[10, 50, 100] {
        let mut expr = x.clone();
        for i in 0..depth {
            expr = match i % 3 {
                0 => expr.sin(),
                1 => expr.cos(),
                _ => expr.exp(),
            };
        }

        c.bench_function(&format!("diff_nested_depth{depth}"), |b| {
            b.iter(|| black_box(&expr).diff(&x))
        });

        c.bench_function(&format!("display_nested_depth{depth}"), |b| {
            b.iter(|| format!("{}", black_box(&expr)))
        });
    }
}

fn bench_expand_binomial(c: &mut Criterion) {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");

    for &n in &[5, 10, 15, 20] {
        let expr = (&a + &b).powi(n);
        c.bench_function(&format!("expand_(a+b)^{n}"), |b| {
            b.iter(|| black_box(&expr).expand())
        });
    }
}

fn bench_expand_multinomial(c: &mut Criterion) {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let cc = ctx.symbol("c");
    let d = ctx.symbol("d");

    // Binomial at higher powers (multinomial path handles these too)
    for &n in &[30, 50, 100] {
        let expr = (&a + &b).powi(n);
        c.bench_function(&format!("expand_binomial_(a+b)^{n}"), |bench| {
            bench.iter(|| black_box(&expr).expand())
        });
    }

    // Trinomial expansions: (a + b + c)^n
    for &n in &[3, 5, 8, 10, 15] {
        let expr = (&a + &b + &cc).powi(n);
        c.bench_function(&format!("expand_trinomial_(a+b+c)^{n}"), |bench| {
            bench.iter(|| black_box(&expr).expand())
        });
    }

    // Quadrinomial expansions: (a + b + c + d)^n
    for &n in &[3, 5, 8] {
        let expr = (&a + &b + &cc + &d).powi(n);
        c.bench_function(&format!("expand_quadrinomial_(a+b+c+d)^{n}"), |bench| {
            bench.iter(|| black_box(&expr).expand())
        });
    }

    // Binomial with symbolic coefficients: (2a + 3b)^n
    for &n in &[5, 10, 20] {
        let expr = (&a * 2 + &b * 3).powi(n);
        c.bench_function(&format!("expand_binomial_(2a+3b)^{n}"), |bench| {
            bench.iter(|| black_box(&expr).expand())
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Matrix benchmarks
// ═══════════════════════════════════════════════════════════════════════════

fn bench_matrix(c: &mut Criterion) {
    let ctx = Context::new();

    for &n in &[3, 5, 8, 10] {
        let data: Vec<Vec<Ex>> = (0..n)
            .map(|i| (0..n).map(|j| ctx.int((i * n + j + 1) as i64)).collect())
            .collect();
        let m = symplex::matrix::Matrix::new(data);

        c.bench_function(&format!("det_integer_{n}x{n}"), |b| {
            b.iter(|| black_box(&m).det())
        });

        c.bench_function(&format!("lu_{n}x{n}"), |b| b.iter(|| black_box(&m).lu()));

        c.bench_function(&format!("rref_{n}x{n}"), |b| {
            b.iter(|| black_box(&m).rref())
        });

        if n <= 8 {
            c.bench_function(&format!("inv_{n}x{n}"), |b| b.iter(|| black_box(&m).inv()));
        }
    }

    // Symbolic 3x3 determinant — much heavier than integer
    let data: Vec<Vec<Ex>> = (0..3)
        .map(|i| {
            (0..3)
                .map(|j| ctx.symbol(&format!("a{}_{}", i, j)))
                .collect()
        })
        .collect();
    let sym_m = symplex::matrix::Matrix::new(data);
    c.bench_function("det_symbolic_3x3", |b| b.iter(|| black_box(&sym_m).det()));
}

// ═══════════════════════════════════════════════════════════════════════════
// Laplace transform benchmarks
// ═══════════════════════════════════════════════════════════════════════════

fn bench_laplace(c: &mut Criterion) {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    c.bench_function("laplace_sin_3t", |b| {
        let expr = (&t * 3).sin();
        b.iter(|| black_box(&expr).laplace(&t, &s))
    });

    c.bench_function("laplace_exp_2t", |b| {
        let expr = (&t * 2).exp();
        b.iter(|| black_box(&expr).laplace(&t, &s))
    });

    c.bench_function("laplace_t2", |b| {
        let expr = t.powi(2);
        b.iter(|| black_box(&expr).laplace(&t, &s))
    });

    c.bench_function("laplace_sum_5_exp", |b| {
        let mut expr = ctx.int(0);
        for i in 1..=5 {
            expr = &expr + &(&t * i).exp();
        }
        b.iter(|| black_box(&expr).laplace(&t, &s))
    });

    c.bench_function("inverse_laplace_1/(s-2)", |b| {
        let expr = &ctx.int(1) / &(&s - 2);
        b.iter(|| black_box(&expr).inverse_laplace(&s, &t))
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Special function benchmarks
// ═══════════════════════════════════════════════════════════════════════════

fn bench_special_functions(c: &mut Criterion) {
    c.bench_function("evalf_gamma_3.5", |b| {
        let val = symplex::rational(7, 2);
        b.iter(|| black_box(&val).gamma().eval_f64())
    });

    c.bench_function("evalf_erf_1", |b| {
        let one = symplex::int(1);
        b.iter(|| black_box(&one).erf().eval_f64())
    });

    c.bench_function("eval_gamma_half_integers", |b| {
        b.iter(|| {
            for n in [1, 3, 5, 7, 9, 11] {
                let _ = symplex::rational(n, 2).gamma().eval();
            }
        })
    });

    c.bench_function("diff_erf_x", |b| {
        let x = symplex::var("x");
        let expr = x.erf();
        b.iter(|| black_box(&expr).diff(&x))
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Solve + inequality benchmarks
// ═══════════════════════════════════════════════════════════════════════════

fn bench_solve_quartic(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    c.bench_function("solve_quartic_x4-5x2+4", |b| {
        let expr = &x.powi(4) - &x.powi(2) * 5 + 4;
        b.iter(|| black_box(&expr).solve(&x))
    });

    c.bench_function("solve_cubic_x3-6x2+11x-6", |b| {
        let expr = &x.powi(3) - &x.powi(2) * 6 + &x * 11 - 6;
        b.iter(|| black_box(&expr).solve(&x))
    });
}

fn bench_inequality(c: &mut Criterion) {
    let x = symplex::var("x");

    c.bench_function("inequality_x2-4>0", |b| {
        let expr = &x.powi(2) - 4;
        b.iter(|| black_box(&expr).solve_gt(&x))
    });

    c.bench_function("inequality_x3-x>0", |b| {
        let expr = &x.powi(3) - &x;
        b.iter(|| black_box(&expr).solve_gt(&x))
    });

    c.bench_function("solveset_x2-5x+6", |b| {
        let expr = &x.powi(2) - &x * 5 + 6;
        b.iter(|| black_box(&expr).solve_as_set(&x))
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Codegen benchmarks
// ═══════════════════════════════════════════════════════════════════════════

fn bench_codegen(c: &mut Criterion) {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    c.bench_function("codegen_poly_deg10", |b| {
        let terms: Vec<Ex> = (0..=10).map(|n| &x.powi(n) * (n + 1) as i64).collect();
        let poly = Ex::sum_of(&ctx, terms);
        b.iter(|| black_box(&poly).to_rust_fn("f", &["x"]))
    });

    c.bench_function("codegen_sin2+cos2", |b| {
        let expr = &x.sin().powi(2) + &x.cos().powi(2);
        b.iter(|| black_box(&expr).to_rust_fn("f", &["x"]))
    });

    c.bench_function("lambdify_poly_deg10", |b| {
        let terms: Vec<Ex> = (0..=10).map(|n| &x.powi(n) * (n + 1) as i64).collect();
        let poly = Ex::sum_of(&ctx, terms);
        b.iter(|| black_box(&poly).compile(&["x"]))
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Simplification benchmarks (new strategies)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_trigsimp(c: &mut Criterion) {
    let x = symplex::var("x");

    c.bench_function("trigsimp_sin2+cos2", |b| {
        let expr = &x.sin().powi(2) + &x.cos().powi(2);
        b.iter(|| black_box(&expr).simplify_trig())
    });

    c.bench_function("trigsimp_sin2+cos2+x", |b| {
        let expr = &x.sin().powi(2) + &x.cos().powi(2) + &x;
        b.iter(|| black_box(&expr).simplify_trig())
    });

    c.bench_function("powsimp_x^a*x^b", |b| {
        let x = symplex::var("x");
        let a = symplex::var("a");
        let bv = symplex::var("b");
        let expr = &x.pow(&a) * &x.pow(&bv);
        b.iter(|| black_box(&expr).simplify_powers())
    });

    c.bench_function("smart_simplify_trig", |b| {
        let expr = &x.sin().powi(2) + &x.cos().powi(2) + &x * 3;
        b.iter(|| black_box(&expr).smart_simplify())
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
    bench_large_polynomial,
    bench_nested_functions,
    bench_expand_binomial,
    bench_expand_multinomial,
    bench_matrix,
    bench_laplace,
    bench_special_functions,
    bench_solve_quartic,
    bench_inequality,
    bench_codegen,
    bench_trigsimp,
);
criterion_main!(benches);
