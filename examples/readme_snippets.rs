//! Every code block in `README.md`, compiled and executed.
//!
//! This example exists so that the README can never drift from the API: each
//! function below mirrors one README section verbatim (plus `println!`s and
//! assertions for the values quoted in the README comments).
//!
//! Run with: `cargo run --example readme_snippets`

use symplex::prelude::*;
use symplex::syms;

fn quick_example() {
    println!("--- Quick Example ---");
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let f = expr!(ctx, x ^ 3 - 2 * x + 1);
    let df = f.diff(&x);
    println!("f'(x) = {df}");
    assert_eq!(df.to_string(), "3*x^2 - 2");

    let roots = expr!(ctx, x ^ 2 - 5 * x + 6).solve(&x).unwrap();
    println!("roots: {roots:?}");
    assert_eq!(roots.len(), 2);

    let gauss =
        expr!(ctx, exp(-x ^ 2)).integrate_definite(&x, &ctx.neg_infinity(), &ctx.infinity());
    println!("∫ e^(-x²) dx = {gauss}");
    assert_eq!(gauss.to_string(), "sqrt(pi)");

    let one = expr!(ctx, sin(x) ^ 2 + cos(x) ^ 2).simplify();
    println!("{one}");
    assert_eq!(one.to_string(), "1");

    let code = df.to_rust_fn("gradient", &["x"]).unwrap();
    println!("{code}");
    assert!(code.contains("pub fn gradient(x: f64) -> f64"));

    let grad = df.compile(&["x"]).unwrap();
    println!("f'(2) = {}", grad(&[2.0]));
    assert_eq!(grad(&[2.0]), 10.0);
}

fn calculus() {
    println!("\n--- Calculus ---");
    let ctx = Context::new();
    syms!(ctx; x);

    let d = expr!(ctx, sin(x ^ 2)).diff(&x);
    println!("{d}");
    assert_eq!(d.to_string(), "2*x*cos(x^2)");
    let i = expr!(ctx, x * exp(x)).integrate(&x);
    println!("{i}");
    assert_eq!(i.to_string(), "x*exp(x) - exp(x)");
    let l = expr!(ctx, sin(x) / x).limit(&x, &ctx.int(0));
    println!("{l}");
    assert_eq!(l.to_string(), "1");
    let s = expr!(ctx, exp(x)).series(&x, &ctx.int(0), 5);
    println!("{s}");
    assert!(s.to_string().contains("1/24*x^4"));

    let r = (1 / &x).limit_right(&x, &ctx.int(0));
    let left = (1 / &x).limit_left(&x, &ctx.int(0));
    println!("{r}  {left}");
    assert_eq!(r.to_string(), "oo");
    assert_eq!(left.to_string(), "-oo");
}

fn definite_integration() {
    println!("\n--- Definite, Improper and Numeric Integration ---");
    let ctx = Context::new();
    syms!(ctx; x);
    let (zero, one, inf) = (ctx.int(0), ctx.int(1), ctx.infinity());

    let a = x.powi(2).integrate_definite(&x, &zero, &one);
    let b = (-&x).exp().integrate_definite(&x, &zero, &inf);
    let c = (&x.sin() / &x).integrate_definite(&x, &zero, &inf);
    let d = x.ln().integrate_definite(&x, &zero, &one);
    let e = x.abs().integrate_definite(&x, &ctx.int(-2), &ctx.int(3));
    println!("{a}  {b}  {c}  {d}  {e}");
    assert_eq!(a.to_string(), "1/3");
    assert_eq!(b.to_string(), "1");
    assert_eq!(c.to_string(), "1/2*pi");
    assert_eq!(d.to_string(), "-1");
    assert_eq!(e.to_string(), "13/2");

    let r = x.powi(-2).try_integrate_definite(&x, &ctx.int(-1), &one);
    assert!(matches!(r, Err(SymplexError::Divergent { .. })));
    println!("∫₋₁¹ dx/x² → Divergent");

    let v = x.powi(2).exp().integrate_numeric(&x, &zero, &one).unwrap();
    println!("∫₀¹ e^(x²) dx ≈ {v}");
    assert!((v - 1.4626517459071816).abs() < 1e-9);

    let z = ctx.symbol("z");
    let res = (&z.exp() / &z.powi(3)).residue(&z, &zero);
    let res_inf = (1 / (&z.powi(2) + 1)).residue_at_infinity(&z);
    println!("{res}  {res_inf}");
    assert_eq!(res.to_string(), "1/2");
    assert_eq!(res_inf.to_string(), "0");
}

fn summation() {
    println!("\n--- Summation, Products and Series ---");
    let ctx = Context::new();
    syms!(ctx; k, x);
    let n = ctx.symbol_with("n", &[Assumption::Integer, Assumption::Positive]);
    let (zero, one, inf) = (ctx.int(0), ctx.int(1), ctx.infinity());

    let faul = k.powi(5).summation(&k, &one, &n);
    let gosper = (&k * &ctx.int(2).pow(&k)).summation(&k, &zero, &n);
    let binom = n.binomial(&k).summation(&k, &zero, &n);
    let basel = k.powi(-2).summation(&k, &one, &inf);
    let z3 = k.powi(-3).summation(&k, &one, &inf);
    let expx = (&x.pow(&k) / &k.factorial()).summation(&k, &zero, &inf);
    let prod = (1 - k.powi(-2)).product_over(&k, &ctx.int(2), &inf);
    println!("{faul}\n{gosper}\n{binom}\n{basel}\n{z3}\n{expx}\n{prod}");
    assert_eq!(faul.to_string(), "1/6*n^6 + 1/2*n^5 + 5/12*n^4 - 1/12*n^2");
    assert_eq!(gosper.to_string(), "2^(n + 1)*(n - 1) + 2");
    assert_eq!(binom.to_string(), "2^n");
    assert_eq!(basel.to_string(), "1/6*pi^2");
    assert_eq!(z3.to_string(), "zeta(3)");
    assert_eq!(expx.to_string(), "exp(x)");
    assert_eq!(prod.to_string(), "1/2");

    assert_eq!((1 / &k).is_convergent(&k), Some(false));

    let s = x.sin().fps_maclaurin(&x);
    let c51 = s.coefficient(51);
    let gt = s.general_term(&k).unwrap();
    let asin: Vec<String> = s
        .reversion()
        .unwrap()
        .coefficients(6)
        .iter()
        .map(|e| e.to_string())
        .collect();
    println!("a_51 = {c51}\ngeneral term {gt}\nasin {asin:?}");
    assert_eq!(
        c51.to_string(),
        "-1/1551118753287382280224243016469303211063259720016986112000000000000"
    );
    assert_eq!(gt.to_string(), "sin(1/2*k*pi)/k!");
    assert_eq!(asin, ["0", "1", "0", "1/6", "0", "3/40"]);
}

fn complex_analysis() {
    println!("\n--- Complex Analysis and Special Functions ---");
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let y = ctx.symbol_with("y", &[Assumption::Real]);
    let i = ctx.i_unit();

    let w = &x + &i * &y;
    assert_eq!(w.conjugate().to_string(), "x - y*I");
    assert_eq!(w.abs_squared().to_string(), "x^2 + y^2");
    let (re, im) = w.exp().as_real_imag();
    assert_eq!(re.to_string(), "cos(y)*exp(x)");
    assert_eq!(im.to_string(), "sin(y)*exp(x)");
    assert_eq!(z.re().to_string(), "re(z)");
    assert_eq!(z.exp().re().to_string(), "cos(im(z))*exp(re(z))");
    assert_eq!((1 / &ctx.int(0)).eval().to_string(), "zoo");
    println!(
        "conj(w) = {}, |w|² = {}, re(z) = {}, 1/0 = {}",
        w.conjugate(),
        w.abs_squared(),
        z.re(),
        (1 / &ctx.int(0)).eval()
    );

    assert_eq!(ctx.int(1).digamma().eval().to_string(), "-EulerGamma");
    assert_eq!(ctx.int(4).zeta().eval().to_string(), "1/90*pi^4");
    assert_eq!(
        ctx.int(1).polygamma(&ctx.int(1)).eval().to_string(),
        "1/6*pi^2"
    );
    assert_eq!(ctx.infinity().si().eval().to_string(), "1/2*pi");
    let g = ctx.catalan().eval_decimal(30).unwrap();
    println!(
        "ψ(1) = {}, ζ(4) = {}, G = {g}",
        ctx.int(1).digamma().eval(),
        ctx.int(4).zeta().eval()
    );
    assert_eq!(g, "0.915965594177219015054603514932");
}

fn algebra() {
    println!("\n--- Algebra and Factoring ---");
    let ctx = Context::new();
    syms!(ctx; x, y);

    let f12 = expr!(ctx, x ^ 12 - 1).factor(&x);
    let mv = expr!(ctx, x ^ 3 - x * y ^ 2 + x ^ 2 - y ^ 2).factor_all();
    let ex = expr!(ctx, (x + 1) ^ 3).expand();
    let ca = expr!(ctx, (x ^ 2 - 1) / (x - 1)).cancel(&x);
    println!("{f12}\n{mv}\n{ex}\n{ca}");
    assert_eq!(
        f12.to_string(),
        "(x - 1)*(x + 1)*(x^2 + x + 1)*(x^2 + 1)*(x^2 - x + 1)*(x^4 - x^2 + 1)"
    );
    assert_eq!(mv.to_string(), "(x + 1)*(x + y)*(x - y)");
    assert_eq!(ex.to_string(), "x^3 + 3*x^2 + 3*x + 1");
    assert_eq!(ca.to_string(), "x + 1");

    assert_eq!(
        expr!(ctx, x ^ 3 - x).discriminant(&x).unwrap().to_string(),
        "4"
    );
    assert_eq!(expr!(ctx, x ^ 5 - x - 1).count_real_roots(&x), Some(1));
    assert_eq!(expr!(ctx, x ^ 4 + 1).is_irreducible(&x), Some(true));
}

fn rule_engine() {
    println!("\n--- Simplification and the Rule Engine ---");
    let ctx = Context::new();
    syms!(ctx; x, y);
    let (a, b) = (ctx.symbol("a_"), ctx.symbol("b_"));

    let rules = RuleSet::from_rules(vec![
        Rule::new("sin_sq", &a.sin().powi(2), &(1 - &a.cos().powi(2))),
        Rule::new("ln_add", &(&a.ln() + &b.ln()), &(&a * &b).ln()),
    ]);
    let r1 = (&x.sin().powi(2) + 3).rewrite(&rules);
    let r2 = (&x.ln() + &y.ln()).rewrite(&rules);
    println!("{r1}\n{r2}");
    assert_eq!(r1.to_string(), "-cos(x)^2 + 4");
    assert_eq!(r2.to_string(), "ln(x*y)");

    let (result, steps) =
        (&x.sin().powi(2) + &x.cos().powi(2)).simplify_traced(&SimplifyOpts::default());
    println!("{result} via {} steps", steps.len());
    assert_eq!(result.to_string(), "1");
    assert!(!steps.is_empty());

    let sa = x.powi(4).subs_algebraic(&x.powi(2), &y);
    let dn = (ctx.int(5) + ctx.int(24).sqrt()).sqrt().sqrtdenest();
    println!("{sa}\n{dn}");
    assert_eq!(sa.to_string(), "y^2");
    assert_eq!(dn.to_string(), "sqrt(2) + sqrt(3)");
}

// `x - x` is deliberate: it demonstrates the identity outcome of `solve`.
#[allow(clippy::eq_op)]
fn solving() {
    println!("\n--- Equation Solving ---");
    let ctx = Context::new();
    syms!(ctx; x, y, z);

    let r = expr!(ctx, x ^ 2 - 5 * x + 6).solve(&x).unwrap();
    assert_eq!(
        r.iter().map(|e| e.to_string()).collect::<Vec<_>>(),
        ["3", "2"]
    );
    let r = (&x.sin() - &ctx.rational(1, 2)).solve(&x).unwrap();
    assert_eq!(
        r.iter().map(|e| e.to_string()).collect::<Vec<_>>(),
        ["1/6*pi", "5/6*pi"]
    );
    assert!(matches!(
        (&x.sin() - 2).solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
    assert!(matches!(
        (&x - &x).solve(&x),
        Err(SymplexError::InfiniteSolutions { .. })
    ));
    println!("solve semantics OK");

    let fam = (&x.sin() - &ctx.rational(1, 2)).solve_general(&x).unwrap();
    println!("{:?} with {:?}", fam.solutions, fam.parameters);
    assert_eq!(fam.solutions.len(), 2);
    assert_eq!(fam.parameters.len(), 1);

    let sol = linsolve(
        &[&x + &y + &z - 6, &x - &y - 2],
        &[x.clone(), y.clone(), z.clone()],
    )
    .unwrap();
    println!("{sol:?}");
    assert!(matches!(sol, LinearSolution::Parametric { .. }));
    assert_eq!(sol.get(&x).unwrap().to_string(), "-1/2*z + 4");

    let sols = symplex::polysys::solve_system_ex(
        &[&x.powi(2) + &y.powi(2) - 1, &x - &y],
        &[x.clone(), y.clone()],
    )
    .unwrap();
    println!("{sols:?}");
    assert_eq!(sols.len(), 2);
    assert_eq!(sols[0][0].to_string(), "1/2*sqrt(2)");

    let gt = expr!(ctx, x ^ 2 - 4).solve_gt(&x);
    let lt = (&(&x - 1).abs() - 2).solve_lt(&x);
    println!("{gt}\n{lt}");
    assert_eq!(gt.to_string(), "(-oo, -2) ∪ (2, oo)");
    assert_eq!(lt.to_string(), "(-1, 3)");
}

fn odes() {
    println!("\n--- Differential Equations and Recurrences ---");
    let ctx = Context::new();
    syms!(ctx; x, n);
    let y = ctx.symbol("y");
    let (d1, d2) = (y.formal_diff(&x), y.formal_diff(&x).formal_diff(&x));

    let general = (&d2 + &y).solve_ode(&y, &x);
    println!("{general}");
    assert!(general.contains(&ctx.symbol("C1")) && general.contains(&ctx.symbol("C2")));
    let ivp = (&d2 + &y)
        .solve_ode_ivp(
            &y,
            &x,
            &[(0, ctx.int(0), ctx.int(0)), (1, ctx.int(0), ctx.int(1))],
        )
        .unwrap();
    println!("{}", ivp.simplify());
    assert_eq!(ivp.simplify().to_string(), "sin(x)");
    let kind = (&d1 * &x - &y - &d1.powi(2)).classify_ode(&y, &x);
    println!("{kind:?}");
    assert_eq!(kind, symplex::ode::OdeType::Clairaut);

    let fib = symplex::rsolve::rsolve_linear(
        &[ctx.int(-1), ctx.int(-1), ctx.int(1)],
        None,
        &n,
        &[ctx.int(0), ctx.int(1)],
    )
    .unwrap();
    println!("{fib}");
    assert_eq!(fib.subs_i64(&n, 10).eval().simplify().to_string(), "55");
}

fn sets_and_logic() {
    println!("\n--- Sets and Logic ---");
    let ctx = Context::new();
    syms!(ctx; x, p, q);

    let a = ctx.interval(&ctx.int(0), &ctx.int(5), false, false);
    let b = ctx.interval(&ctx.int(3), &ctx.int(10), true, false);
    assert_eq!(a.intersection(&b).simplify().to_string(), "(3, 5]");
    assert_eq!(a.symmetric_difference(&b).to_string(), "[0, 3] ∪ (5, 10]");
    assert_eq!(a.contains(&ctx.int(7)), Some(false));
    assert_eq!(a.contains(&x), None);
    assert_eq!(a.union(&b).measure().unwrap().to_string(), "10");
    println!(
        "A ∩ B = {}, A Δ B = {}",
        a.intersection(&b).simplify(),
        a.symmetric_difference(&b)
    );

    let conds = [
        x.gt(&ctx.int(0)),
        x.le(&ctx.int(5)),
        (&x.powi(2) - 4).gt(&ctx.int(0)),
    ];
    let red = reduce_inequalities(&conds, &x).unwrap();
    println!("{red}");
    assert_eq!(red.to_string(), "(2, 5]");

    let (pp, qq) = (p.gt(&ctx.int(0)), q.gt(&ctx.int(0)));
    assert_eq!(pp.and(&qq).or(&pp).simplify().to_string(), "p > 0");
    assert_eq!(pp.and(&qq).not().to_nnf().to_string(), "0 >= p | 0 >= q");
    assert_eq!(pp.or(&pp.not()).is_tautology(), Some(true));
    println!("logic OK");
}

fn linear_algebra() {
    println!("\n--- Linear Algebra ---");
    let ctx = Context::new();
    syms!(ctx; t, n);
    let m = matrix![ctx, [2, 1], [1, 2]];

    assert_eq!(m.det().unwrap().to_string(), "3");
    let ev: Vec<String> = m
        .eigenvals()
        .unwrap()
        .iter()
        .map(|e| e.to_string())
        .collect();
    assert_eq!(ev, ["3", "1"]);
    let lam = ctx.symbol("λ");
    println!("char poly: {}", m.char_poly(&lam).unwrap());
    let (_p, d) = m.diagonalize().unwrap();
    assert_eq!(d.get(0, 0).to_string(), "3");
    let et = m.matrix_exp_t(&t).unwrap();
    println!("exp(tM) = {et}");
    let pn = m.matrix_pow_symbolic(&n).unwrap();
    assert_eq!(pn.get(0, 0).to_string(), "1/2*3^n + 1/2");
    let _sq = m.matrix_sqrt().unwrap();

    let spd = matrix![ctx, [4, 12, -16], [12, 37, -43], [-16, -43, 98]];
    let l = spd.cholesky().unwrap();
    assert_eq!(l.get(2, 0).to_string(), "-8");
    assert_eq!(spd.is_positive_definite(), Some(true));
    let (q, r) = matrix![ctx, [1, 1, 0], [1, 0, 1], [0, 1, 1]].qr().unwrap();
    assert_eq!(q.is_orthogonal(), Some(true));
    println!("R = {r}");

    let cubic = matrix![ctx, [0, 1, 0], [0, 0, 1], [1, 1, 0]]
        .eigenvals()
        .unwrap();
    println!("{}", cubic[0]);
    assert!(cubic[0].to_string().starts_with("RootOf("));
}

fn transforms() {
    println!("\n--- Transforms ---");
    let ctx = Context::new();
    syms!(ctx; t, w, s, x);
    let a = ctx.symbol_with("a", &[Assumption::Positive]);

    assert_eq!(
        (-&a * t.abs())
            .exp()
            .fourier_transform(&t, &w)
            .unwrap()
            .to_string(),
        "2*a/(a^2 + w^2)"
    );
    assert_eq!(
        (-t.powi(2))
            .exp()
            .fourier_transform(&t, &w)
            .unwrap()
            .to_string(),
        "sqrt(pi)*exp(-1/4*w^2)"
    );
    let (mf, strip) = (1 / (1 + &x)).mellin_transform(&x, &s).unwrap();
    assert_eq!(mf.to_string(), "pi/sin(s*pi)");
    assert_eq!(strip.to_string(), "re(s) > 0 & 1 > re(s)");
    assert_eq!(t.sin().laplace(&t, &s).to_string(), "1/(s^2 + 1)");
    assert_eq!(
        ((&s * -2).exp() / &s).inverse_laplace(&s, &t).to_string(),
        "H(t - 2)"
    );
    let sq = x
        .sign()
        .fourier_series_on(&x, &(-ctx.pi()), &ctx.pi(), 5)
        .unwrap()
        .truncate(5);
    println!("{mf} on {strip}\n{sq}");
    assert_eq!(
        sq.to_string(),
        "4*sin(x)/pi + 4/3*1/pi*sin(3*x) + 4/5*1/pi*sin(5*x)"
    );
}

fn number_theory() {
    println!("\n--- Number Theory and Combinatorics ---");
    use num_bigint::BigInt;
    use symplex::combinatorics::*;
    use symplex::diophantine;
    use symplex::ntheory::*;

    assert!(!isprime(561));
    let f = factorint(1_099_532_599_387u64);
    println!("{f:?}");
    assert_eq!(
        f,
        vec![(BigInt::from(1_048_583), 1), (BigInt::from(1_048_589), 1)]
    );
    assert_eq!(sqrt_mod(2, 7), Some(BigInt::from(3)));
    assert_eq!(discrete_log(3, 13, 17), Some(BigInt::from(4)));
    assert_eq!(primepi(1_000_000), Some(78498));
    let (head, period) = continued_fraction_periodic(23).unwrap();
    assert_eq!(head, vec![BigInt::from(4)]);
    assert_eq!(period.len(), 4);
    assert_eq!(
        diophantine::pell(61),
        Some((BigInt::from(1766319049u64), BigInt::from(226153980u64)))
    );
    assert_eq!(
        diophantine::sum_of_two_squares(65),
        Some((BigInt::from(4), BigInt::from(7)))
    );
    assert_eq!(stirling2(10, 4), Some(BigInt::from(34105)));
    assert_eq!(partition_count(100), Some(BigInt::from(190569292)));
    assert_eq!(crt_i64(&[2, 3, 2], &[3, 5, 7]), Some(23));
    println!("number theory OK");
}

fn codegen() {
    println!("\n--- Code Generation ---");
    let ctx = Context::new();
    syms!(ctx; x, y);
    let f = &x.sin().powi(2) + &(&x * 2 + &y).exp() * 3;

    let rust = f.to_rust_fn("f", &["x", "y"]).unwrap();
    println!("{rust}");
    assert!(rust.contains("3_f64.mul_add(2_f64.mul_add(x, y).exp(), x.sin().powi(2))"));
    let c = f.to_c_fn("f", &["x", "y"]).unwrap();
    println!("{c}");
    assert!(c.contains("#include <math.h>"));
    assert!(c.contains("fma(3.0, exp(fma(2.0, x, y)), pow(sin(x), 2.0))"));
    assert!(
        x.lambertw()
            .to_c_fn("w0", &["x"])
            .unwrap()
            .contains("symplex_lambert_w0")
    );

    let opts = CodegenOptions {
        precision: Precision::F32,
        checked_domain: true,
        ..Default::default()
    };
    let g = x.ln().to_c_fn_with_options("g", &["x"], &opts).unwrap();
    println!("{g}");
    assert!(g.contains("assert(x > 0.0f), logf(x)"));

    let cf = f.compile(&["x", "y"]).unwrap();
    let v = cf(&[0.5, 0.25]);
    let grad = Ex::compile_many(&[&f.diff(&x), &f.diff(&y)], &["x", "y"]).unwrap();
    let gv = grad.call_vec(&[0.5, 0.25]);
    println!("f = {v}, ∇f = {gv:?}");
    assert!((v - (0.5f64.sin().powi(2) + 3.0 * (1.25f64).exp())).abs() < 1e-12);

    let tex = f.to_latex();
    println!("{tex}");
    assert_eq!(tex, r"\sin^{2}\left(x\right) + 3\exp\left(2x + y\right)");
}

fn units() {
    println!("\n--- Compile-Time Dimensional Analysis ---");
    use symplex::units::*;

    let ctx = Context::new();
    let m = Mass::symbol(&ctx, "m");
    let a = Acceleration::symbol(&ctx, "a");
    let force = dim!(ctx, Force: m * a);
    println!("F = {force}");

    syms!(ctx; g, t);
    let t_var = Time::symbol(&ctx, "t");
    let position = Length::from_ex(expr!(ctx, 1 / 2 * g * t ^ 2));
    let velocity: Velocity = position.diff_wrt(&t_var);
    println!("v = {velocity}");
    assert_eq!(velocity.inner().to_string(), "g*t");
}

fn api_model() {
    println!("\n--- The API Model ---");
    let ctx = Context::new();
    syms!(ctx; x);
    let hard_expr = x.powi(2).exp();
    let anti = hard_expr.integrate(&x);
    if anti.has_unevaluated() {
        println!("integration produced formal result: {anti}");
    }
    assert!(anti.has_unevaluated());
    // RootOf is not "unevaluated".
    let quintic_root = &(&x.powi(5) - &x - 1).solve(&x).unwrap()[0];
    assert!(!quintic_root.has_unevaluated());
    println!("{quintic_root} is a complete answer");
}

fn main() {
    println!("=== README snippets ===\n");
    quick_example();
    calculus();
    definite_integration();
    summation();
    complex_analysis();
    algebra();
    rule_engine();
    solving();
    odes();
    sets_and_logic();
    linear_algebra();
    transforms();
    number_theory();
    codegen();
    units();
    api_model();
    println!("\n✓ Every README snippet ran.");
}
