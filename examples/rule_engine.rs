//! The public rewrite-rule engine (symplex 0.2).
//!
//! Demonstrates:
//! - building `Rule`s from expressions whose `name_` symbols are wildcards
//!   (and `rest__` sequence wildcards for the remaining terms of a sum or
//!   product);
//! - guarded rules (`Rule::new_with_guard`) and closure-computed right-hand
//!   sides (`Rule::new_fn`);
//! - the `rule!` macro via `RuleSet::from_macro_rules`;
//! - `rewrite`, `rewrite_once`, `rewrite_with` + `RewriteStrategy`,
//!   `rewrite_traced` and `simplify_traced` for step-by-step traces;
//! - `simplify_with_rules` to interleave user rules with the built-in
//!   simplifier;
//! - `subs_algebraic` (substitution inside powers, products and sums);
//! - the new targeted simplifiers `sqrtdenest`, `signsimp`,
//!   `powdenest(force)`, `expand_with(ExpandOpts)`, `nsimplify`, `rcollect`.
//!
//! Run with: `cargo run --example rule_engine`

use symplex::prelude::*;

fn main() {
    println!("=== Rewrite-Rule Engine ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; x, y, t);
    // Wildcards: any symbol whose name ends in `_`.
    let a = ctx.symbol("a_");
    let b = ctx.symbol("b_");

    // ── 1. Template rules ───────────────────────────────────────────────
    println!("--- Template rules ---");
    // sin(a_)² → 1 − cos(a_)²
    let sin_sq = Rule::new("sin_sq", &a.sin().powi(2), &(1 - &a.cos().powi(2)));
    // ln(a_) + ln(b_) → ln(a_·b_)   (matched associatively/commutatively)
    let ln_add = Rule::new("ln_add", &(&a.ln() + &b.ln()), &(&a * &b).ln());
    let rules = RuleSet::from_rules(vec![sin_sq.clone(), ln_add.clone()]);

    let e1 = &x.sin().powi(2) + 3;
    println!("{e1}  →  {}", e1.rewrite(&rules));
    let e2 = &x.ln() + &y.ln() + &(&x + 1).ln();
    println!("{e2}  →  {}", e2.rewrite(&rules));

    // Matching returns the wildcard bindings.
    let bindings = sin_sq.matches(&(&x * 2).sin().powi(2)).unwrap();
    println!(
        "bindings of sin_sq against sin(2x)²: a_ = {}",
        bindings.get("a_").unwrap()
    );
    println!("wildcards of ln_add: {:?}", ln_add.wildcards());

    // ── 2. Sequence wildcards ───────────────────────────────────────────
    println!("\n--- Sequence wildcards (rest__) ---");
    let rest = ctx.symbol("rest__");
    // Pull the numeric term out of exp(a_ + rest__): exp(3 + x + y) → exp(3)·exp(x + y).
    // `rest__` may bind to the empty remainder (`0` for sums, `1` for
    // products), so exp(3) alone would match with rest__ = 0 and the rule
    // would fire forever; the guard requires a genuine remainder.
    let exp_split = Rule::new_with_guard(
        "exp_split",
        &(&a + &rest).exp(),
        &(&a.exp() * &rest.exp()),
        |bind| {
            bind.get("a_")
                .is_some_and(|v| v.expr_type() == ExprType::Number)
                && bind.get("rest__").is_some_and(|r| !r.is_zero_structural())
        },
    );
    let rules = RuleSet::from_rules(vec![exp_split]);
    let e3 = (&x + &y + 3).exp();
    println!("{e3}  →  {}", e3.rewrite(&rules));
    let e4 = (&x + &y).exp();
    println!(
        "{e4}  →  {}   (guard: no numeric term, so no rewrite)",
        e4.rewrite(&rules)
    );

    // ── 3. Closure right-hand sides ─────────────────────────────────────
    println!("\n--- Closure rules (Rule::new_fn) ---");
    // Replace a_! by its value when a_ is a small integer, otherwise leave it.
    let fact_eval = Rule::new_fn("small_factorial", &a.factorial(), |bind| {
        let n = bind.get("a_")?.as_i64()?;
        (0..=20).contains(&n).then(|| {
            let ctx = bind.get("a_").unwrap().context();
            ctx.from_i128((1..=n as i128).product())
        })
    });
    let rules = RuleSet::from_rules(vec![fact_eval]);
    let e5 = &ctx.int(6).factorial() + &x.factorial() + &ctx.int(30).factorial();
    println!("{e5}  →  {}", e5.rewrite(&rules));

    // ── 4. The rule! macro ──────────────────────────────────────────────
    println!("\n--- rule! macro ---");
    let raw = ctx.with_arena_mut(|arena| {
        vec![
            rule!(arena, "pyth", sin(w_)^2 + cos(w_)^2 => 1),
            rule!(arena, "double_angle", 2 * sin(w_) * cos(w_) => sin(2 * w_)),
        ]
    });
    let trig = RuleSet::from_macro_rules(&ctx, raw);
    let e6 = &x.sin().powi(2) + &x.cos().powi(2) + &x.sin() * &x.cos() * 2;
    println!("{e6}  →  {}", e6.rewrite(&trig));

    // ── 5. Strategies and single passes ─────────────────────────────────
    println!("\n--- Strategies ---");
    let flatten = RuleSet::from_rules(vec![Rule::new("flatten", &a.exp().exp(), &a.exp())]);
    let nested = x.exp().exp().exp().exp();
    println!(
        "rewrite_once:      {nested}  →  {}",
        nested.rewrite_once(&flatten)
    );
    println!(
        "rewrite (fixpoint): {nested}  →  {}",
        nested.rewrite(&flatten)
    );
    let opts = RewriteOpts::default().strategy(RewriteStrategy::TopDown);
    println!(
        "TopDown:           {nested}  →  {}",
        nested.rewrite_with(&flatten, &opts)
    );

    // ── 6. Tracing ──────────────────────────────────────────────────────
    println!("\n--- Tracing ---");
    let e7 = &x.sin().powi(2) + &x.cos().powi(2) + &t.ln() + &y.ln();
    let all = RuleSet::from_rules(vec![sin_sq, ln_add]);
    let (result, steps) = e7.rewrite_traced(&all);
    println!("{e7}  →  {result}");
    for s in &steps {
        println!("  [{}]  {}  ⇒  {}", s.rule_name, s.before, s.after);
    }

    // The built-in simplifier can be traced too: strategy-level entries are
    // named `strategy:<name>`; rule-level entries carry the rule name.
    let e8 = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
    let (result, steps) = e8.simplify_traced(&SimplifyOpts::default());
    println!("\nsimplify_traced: {e8}  →  {result}");
    for s in steps.iter().take(6) {
        println!("  [{}]  {}  ⇒  {}", s.rule_name, s.before, s.after);
    }
    if steps.len() > 6 {
        println!("  … {} more steps", steps.len() - 6);
    }

    // The standard rule set is available as a RuleSet as well.
    let standard = RuleSet::standard(&ctx);
    println!("\nRuleSet::standard has {} rules", standard.len());

    // ── 7. User rules inside simplify ───────────────────────────────────
    println!("\n--- simplify_with_rules ---");
    // Teach the simplifier your own identity: sinh(a_) → (eᵃ − e⁻ᵃ)/2
    let sinh_def = (&a.exp() - &(-&a).exp()) / 2;
    let extra = RuleSet::from_rules(vec![Rule::new("sinh_def", &a.sinh(), &sinh_def)]);
    let e9 = &x.sinh() - &(&x.exp() - &(-&x).exp()) / 2;
    println!("{e9}");
    println!("  simplify()            → {}", e9.simplify());
    println!(
        "  simplify_with_rules() → {}",
        e9.simplify_with_rules(&extra)
    );

    // ── 8. Algebraic substitution ───────────────────────────────────────
    println!("\n--- subs_algebraic ---");
    let u = ctx.symbol("u");
    let x2 = x.powi(2);
    for e in [
        x.powi(4).clone(),
        x.powi(3).clone(),
        x.powi(-2).clone(),
        &x.powi(6) + &x.powi(2) * 3 + 1,
    ] {
        println!(
            "{e:<20} with x² → u :  subs = {:<14} subs_algebraic = {}",
            e.subs(&x2, &u).to_string(),
            e.subs_algebraic(&x2, &u)
        );
    }
    let e10 = (&x * 2).exp();
    println!(
        "{e10:<20} with eˣ → u :  subs_algebraic = {}",
        e10.subs_algebraic(&x.exp(), &u)
    );

    // ── 9. Targeted simplifiers new in 0.2 ──────────────────────────────
    println!("\n--- Targeted simplifiers ---");
    let nested_sqrt = (ctx.int(5) + &ctx.int(24).sqrt()).sqrt();
    println!("sqrtdenest: {nested_sqrt}  →  {}", nested_sqrt.sqrtdenest());
    let signs = (-&x - &y) * (-&t);
    println!("signsimp:   {signs}  →  {}", signs.signsimp());
    let pd = x.powi(2).pow(&ctx.rational(1, 2));
    println!(
        "powdenest:  {pd}  →  {} (force=false)  /  {} (force=true, assumes x ≥ 0)",
        pd.powdenest(false),
        pd.powdenest(true)
    );
    let tri = (&x + &y).powi(3);
    println!("expand_multinomial: {tri}  →  {}", tri.expand_multinomial());
    let opts = ExpandOpts {
        deep: false,
        ..ExpandOpts::default()
    };
    let e11 = (&x + 1) * &((&y + 1).powi(2)).sin();
    println!(
        "expand_with(deep=false): {e11}  →  {}",
        e11.expand_with(&opts)
    );
    let approx = ctx.from_f64(0.333333333333).unwrap();
    println!("nsimplify: {approx}  →  {}", approx.nsimplify(1e-9));
    let pi_ish = ctx.from_f64(std::f64::consts::PI / 2.0).unwrap();
    println!(
        "nsimplify_with_constants: {pi_ish}  →  {}",
        pi_ish.nsimplify_with_constants(&[&ctx.pi()], 1e-12)
    );
    let poly = &x * &y + &x * &t + &y * &t + &x;
    println!(
        "rcollect by [x, y]: {poly}  →  {}",
        poly.rcollect(&[&x, &y])
    );

    println!("\n✓ Done!");
}
