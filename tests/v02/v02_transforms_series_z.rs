//! symplex 0.2 — Fourier series on arbitrary intervals (`FourierSeries`)
//! and the Z-transform table / inverse.
//!
//! Fourier partial sums are checked numerically against the function at
//! interior points; Z-transforms are checked against the defining sum
//! `Σ x[n] z₀⁻ⁿ` at a point of convergence, and inverses by evaluating the
//! sequence at `n = 0..8`.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Fourier series
// ═══════════════════════════════════════════════════════════════════════════

/// The N-term partial sum must be closer to `f` than the M-term one
/// (M < N) at interior points, and reasonably close for the larger N.
fn check_partial_sums(f: &Ex, x: &Ex, lower: f64, upper: f64, small: u32, large: u32) {
    let ctx = f.context();
    let lo = ctx.from_f64(lower).unwrap();
    let hi = ctx.from_f64(upper).unwrap();
    let fs = f
        .fourier_series_on(x, &lo, &hi, large)
        .unwrap_or_else(|e| panic!("Fourier series of {f}: {e}"));
    let s_small = fs.truncate(small).compile(&["x"]).unwrap();
    let s_large = fs.truncate(large).compile(&["x"]).unwrap();
    let fc = f.compile(&["x"]).unwrap();
    let mut err_small = 0.0f64;
    let mut err_large = 0.0f64;
    // Interior points away from jumps (avoid 0 and the end points).
    for k in 1..=7 {
        let p = lower + (upper - lower) * (k as f64 + 0.37) / 9.0;
        let exact = fc.call(&[p]);
        err_small += (s_small.call(&[p]) - exact).abs();
        err_large += (s_large.call(&[p]) - exact).abs();
    }
    assert!(
        err_large < err_small,
        "{f}: {large}-term error {err_large} not below {small}-term error {err_small}"
    );
    // Sum over 7 points; discontinuous waves converge only like 1/N.
    assert!(
        err_large < 1.5,
        "{f}: {large}-term partial sum too far off ({err_large})"
    );
}

#[test]
fn sawtooth_square_and_triangle_waves() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let neg_pi = -&pi;

    // Sawtooth x: b_k = 2(−1)^{k+1}/k
    let saw = x.fourier_series_on(&x, &neg_pi, &pi, 4).unwrap();
    assert_eq!(format!("{}", saw.a0), "0");
    assert!(saw.an.iter().all(|a| a.is_zero_structural()));
    assert_eq!(
        saw.bn.iter().map(ToString::to_string).collect::<Vec<_>>(),
        ["2", "-1", "2/3", "-1/2"]
    );
    assert_eq!(format!("{}", saw.coefficient_b(5)), "2/5");
    assert_eq!(format!("{}", saw.period), "2*pi");
    assert_eq!(format!("{}", saw.omega0()), "1");
    assert_eq!(saw.n_terms(), 4);
    check_partial_sums(&x, &x, -3.0, 3.0, 2, 8);

    // Square wave sign(x): b_k = 4/(kπ) for odd k
    let sq = x.sign().fourier_series_on(&x, &neg_pi, &pi, 5).unwrap();
    assert_eq!(format!("{}", sq.a0), "0");
    assert_eq!(sq.coefficient_b(1), 4 / &pi);
    assert_eq!(format!("{}", sq.coefficient_b(2)), "0");
    assert_eq!(sq.coefficient_b(3), 4 / (3 * &pi));
    assert_eq!(sq.coefficient_b(5), 4 / (5 * &pi));
    check_partial_sums(&x.sign(), &x, -3.0, 3.0, 1, 9);

    // Triangle wave |x|: a₀ = π, a_k = −4/(k²π) for odd k
    let tri = x.abs().fourier_series_on(&x, &neg_pi, &pi, 3).unwrap();
    assert_eq!(tri.a0, pi.clone());
    assert_eq!(tri.coefficient_a(1), -4 / &pi);
    assert_eq!(format!("{}", tri.coefficient_a(2)), "0");
    assert_eq!(tri.coefficient_a(3), -4 / (9 * &pi));
    assert!(tri.bn.iter().all(|b| b.is_zero_structural()));
    check_partial_sums(&x.abs(), &x, -3.0, 3.0, 1, 5);
}

#[test]
fn general_intervals_piecewise_and_complex_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let otherwise = ctx.int(1).gt(&ctx.int(0));

    // x on [0, 1]: period 1, ω₀ = 2π, a₀ = 1, a_k = 0, b_k = −1/(kπ)
    let fs = x
        .fourier_series_on(&x, &ctx.int(0), &ctx.int(1), 3)
        .unwrap();
    assert_eq!(format!("{}", fs.period), "1");
    assert_eq!(fs.omega0(), 2 * &pi);
    assert_eq!(format!("{}", fs.a0), "1");
    assert_eq!(fs.coefficient_b(1), -1 / &pi);
    assert_eq!(fs.coefficient_b(3), -1 / (3 * &pi));
    // c₀ = a₀/2, c_k = (a_k − i b_k)/2, c_{−k} = conjugate
    assert_eq!(format!("{}", fs.coefficient_c(0)), "1/2");
    let i = ctx.i_unit();
    assert_eq!(fs.coefficient_c(1), &i / (2 * &pi));
    assert_eq!(fs.coefficient_c(-1), -&i / (2 * &pi));
    check_partial_sums(&x, &x, 0.0, 1.0, 2, 8);

    // Step H(x) on [−1, 1]: a₀ = 1, b_k = 2/(kπ) for odd k
    let st = x
        .heaviside()
        .fourier_series_on(&x, &ctx.int(-1), &ctx.int(1), 3)
        .unwrap();
    assert_eq!(format!("{}", st.a0), "1");
    assert_eq!(st.coefficient_b(1), 2 / &pi);
    assert_eq!(format!("{}", st.coefficient_b(2)), "0");
    assert_eq!(
        st.truncate(1),
        ctx.rational(1, 2) + 2 * (&pi * &x).sin() / &pi
    );

    // Piecewise square pulse (1 for x < 0, 0 otherwise) on [−π, π]
    let pw = Ex::piecewise(&[(&ctx.int(1), &x.lt(&ctx.int(0))), (&ctx.int(0), &otherwise)]);
    let ps = pw.fourier_series_on(&x, &(-&pi), &pi, 3).unwrap();
    assert_eq!(format!("{}", ps.a0), "1");
    assert_eq!(ps.coefficient_b(1), -2 / &pi);
    assert_eq!(ps.coefficient_b(3), -2 / (3 * &pi));

    // x² on [−π, π]: a₀ = 2π²/3, a_k = 4(−1)^k/k²
    let sq = x.powi(2).fourier_series_on(&x, &(-&pi), &pi, 3).unwrap();
    assert_eq!(sq.a0, 2 * pi.powi(2) / 3);
    assert_eq!(
        sq.an.iter().map(ToString::to_string).collect::<Vec<_>>(),
        ["-4", "1", "-4/9"]
    );
    check_partial_sums(&x.powi(2), &x, -3.0, 3.0, 1, 4);
}

#[test]
fn fourier_series_errors() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    assert!(matches!(
        x.fourier_series_on(&ctx.int(1), &ctx.int(0), &ctx.int(1), 2),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        x.fourier_series_on(&x, &ctx.int(1), &ctx.int(1), 2),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        x.fourier_series_on(&x, &ctx.int(1), &ctx.int(0), 2),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // No closed-form coefficient integral → Err rather than an Integral node.
    let hard = (x.powi(2).sin() * &y).exp();
    assert!(
        hard.fourier_series_on(&x, &ctx.int(0), &ctx.int(1), 1)
            .is_err()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Z-transform
// ═══════════════════════════════════════════════════════════════════════════

/// Check `X(z₀)` against `Σ_{n=0}^{N} x[n] z₀⁻ⁿ` at `z₀ = 4`.
fn verify_z(x: &Ex, _n: &Ex, z: &Ex, big_x: &Ex) {
    let ctx = x.context();
    let z0 = 4.0f64;
    let seq = x
        .compile(&["n"])
        .unwrap_or_else(|e| panic!("cannot compile {x}: {e}"));
    let mut sum = 0.0;
    for k in 0..80 {
        sum += seq.call(&[k as f64]) * z0.powi(-k);
    }
    let sym = big_x
        .subs(z, &ctx.from_f64(z0).unwrap())
        .eval_f64()
        .unwrap_or_else(|e| panic!("{big_x} at z={z0}: {e}"));
    assert!(
        (sum - sym).abs() < 1e-8 * sym.abs().max(1.0),
        "Z{{{x}}} = {big_x}: sum {sum} vs {sym}"
    );
}

/// Two sequences agree at `n = 0..8`. (Inverse results write discrete
/// steps as `H(n − k + 1/2)`, which is exactly `0`/`1` at integers; the
/// reference sequences below use the same form.)
fn assert_seq_eq(a: &Ex, b: &Ex, n: &Ex, label: &str) {
    let ctx = a.context();
    // `KroneckerDelta` is not compilable, so evaluate by exact substitution.
    let at = |e: &Ex, k: i64| -> f64 {
        e.subs(n, &ctx.int(k))
            .eval()
            .eval_f64()
            .unwrap_or_else(|err| panic!("{label}: {e} at n={k}: {err}"))
    };
    for k in 0..8 {
        let va = at(a, k);
        let vb = at(b, k);
        assert!(
            (va - vb).abs() < 1e-9,
            "{label}: at n={k}: {va} vs {vb} ({a} vs {b})"
        );
    }
}

fn zt(x: &Ex, n: &Ex, z: &Ex) -> Ex {
    x.z_transform(n, z)
        .unwrap_or_else(|e| panic!("Z{{{x}}} failed: {e}"))
}

fn izt(big_x: &Ex, z: &Ex, n: &Ex) -> Ex {
    big_x
        .inverse_z_transform(z, n)
        .unwrap_or_else(|e| panic!("Z⁻¹{{{big_x}}} failed: {e}"))
}

#[test]
fn z_transform_powers_of_n() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();

    let x = n.powi(2);
    let big_x = zt(&x, &n, &z);
    assert!(
        (&big_x - &z * (&z + 1) / (&z - 1).powi(3))
            .simplify()
            .is_zero_structural(),
        "{big_x}"
    );
    verify_z(&x, &n, &z, &big_x);

    let x = n.powi(3);
    let big_x = zt(&x, &n, &z);
    verify_z(&x, &n, &z, &big_x);

    let x = n.powi(2) * ctx.int(2).pow(&n);
    let big_x = zt(&x, &n, &z);
    assert!(
        (&big_x - 2 * &z * (&z + 2) / (&z - 2).powi(3))
            .simplify()
            .is_zero_structural(),
        "{big_x}"
    );
    verify_z(&x, &n, &z, &big_x);

    let big_x = zt(&(n.powi(2) * a.pow(&n)), &n, &z);
    assert!(
        (&big_x - &a * &z * (&z + &a) / (&z - &a).powi(3))
            .simplify()
            .is_zero_structural(),
        "{big_x}"
    );
}

#[test]
fn z_transform_trig_steps_deltas_binomials() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();

    let x = ctx.rational(1, 2).pow(&n) * (2 * &n).cos();
    let big_x = zt(&x, &n, &z);
    verify_z(&x, &n, &z, &big_x);
    let x = ctx.rational(1, 2).pow(&n) * (2 * &n).sin();
    let big_x = zt(&x, &n, &z);
    verify_z(&x, &n, &z, &big_x);
    let big_x = zt(&(a.pow(&n) * (2 * &n).cos()), &n, &z);
    assert_eq!(
        big_x,
        &z * (&z - &a * ctx.int(2).cos())
            / (z.powi(2) - 2 * &a * &z * ctx.int(2).cos() + a.powi(2))
    );

    assert_eq!(zt(&(&n - 2).heaviside(), &n, &z), 1 / (&z * (&z - 1)));
    assert_eq!(zt(&(&n - 3).dirac_delta(), &n, &z), z.powi(-3));
    assert_eq!(format!("{}", zt(&n.dirac_delta(), &n, &z)), "1");
    assert_eq!(zt(&n.binomial(&ctx.int(2)), &n, &z), &z / (&z - 1).powi(3));
    assert_eq!(zt(&(1 / n.factorial()), &n, &z), (1 / &z).exp());
    verify_z(&(1 / n.factorial()), &n, &z, &(1 / &z).exp());

    // Delay rule: x[n−k]H(n−k) → z^{−k} X(z)
    assert_eq!(
        zt(&(ctx.int(2).pow(&(&n - 2)) * (&n - 2).heaviside()), &n, &z),
        1 / (&z * (&z - 2))
    );
    assert_eq!(
        zt(&((&n - 1) * (&n - 1).heaviside()), &n, &z),
        1 / (&z - 1).powi(2)
    );

    // Unknown sequence → Err.
    assert!(n.ln().z_transform(&n, &z).is_err());
}

#[test]
fn inverse_z_transform_rational_and_delays() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();

    assert_eq!(format!("{}", izt(&(&z / (&z - 3)), &z, &n)), "3^n");
    assert_eq!(izt(&(&z / (&z - &a)), &z, &n), a.pow(&n));
    let step1 = (&n - ctx.rational(1, 2)).heaviside(); // u[n − 1]
    assert_seq_eq(
        &izt(&(1 / (&z - 3)), &z, &n),
        &(ctx.int(3).pow(&(&n - 1)) * &step1),
        &n,
        "1/(z−3)",
    );
    assert_seq_eq(
        &izt(&(&z / (&z - 3).powi(2)), &z, &n),
        &(&n * ctx.int(3).pow(&(&n - 1))),
        &n,
        "z/(z−3)²",
    );
    assert_seq_eq(
        &izt(&(&z / (&z - 3).powi(3)), &z, &n),
        &(&n * (&n - 1) / 2 * ctx.int(3).pow(&(&n - 2))),
        &n,
        "z/(z−3)³",
    );
    assert_eq!(
        format!("{}", izt(&(1 / z.powi(2)), &z, &n)),
        "KroneckerDelta(2, n)"
    );
    assert_eq!(
        format!("{}", izt(&ctx.int(1), &z, &n)),
        "KroneckerDelta(0, n)"
    );
    // Both delta spellings are accepted on input.
    assert_eq!(zt(&izt(&(1 / z.powi(2)), &z, &n), &n, &z), z.powi(-2));
    assert_eq!(izt(&(1 / &z).exp(), &z, &n), 1 / n.factorial());
    // Trigonometric with irrational/symbolic parameters.
    let w = ctx.symbol("w");
    let cos_x = &z * (&z - w.cos()) / (z.powi(2) - 2 * &z * w.cos() + 1);
    assert_eq!(izt(&cos_x, &z, &n), (&w * &n).cos());
    let sin_x = &z * ctx.int(1).sin() / (z.powi(2) - 2 * &z * ctx.int(1).cos() + 1);
    assert_eq!(izt(&sin_x, &z, &n), n.sin());

    // Partial fractions with repeated and distinct roots, improper fractions.
    let step2 = (&n - ctx.rational(3, 2)).heaviside(); // u[n − 2]
    for (x, reference, label) in [
        (n.powi(2), n.powi(2), "n²"),
        (n.powi(3), n.powi(3), "n³"),
        (
            n.powi(2) * ctx.int(2).pow(&n),
            n.powi(2) * ctx.int(2).pow(&n),
            "n²2ⁿ",
        ),
        (
            &n * ctx.int(2).pow(&n) + ctx.int(3).pow(&n),
            &n * ctx.int(2).pow(&n) + ctx.int(3).pow(&n),
            "n2ⁿ + 3ⁿ",
        ),
        (
            (&n - 2).heaviside() * ctx.int(2).pow(&(&n - 2)),
            &step2 * ctx.int(2).pow(&(&n - 2)),
            "delayed 2ⁿ",
        ),
    ] {
        let big_x = zt(&x, &n, &z);
        let back = izt(&big_x, &z, &n);
        assert_seq_eq(
            &back,
            &reference,
            &n,
            &format!("round trip {label} via {big_x}"),
        );
    }
    let improper = z.powi(2) / ((&z - 1) * (&z - 2));
    let back = izt(&improper, &z, &n);
    // x[n] = 2^{n+1} − 1 for n ≥ 0 (δ[n] + 4·2^{n−1}H(n−1) − H(n−1))
    assert_seq_eq(&back, &(2 * ctx.int(2).pow(&n) - 1), &n, "improper");
    assert!(z.ln().inverse_z_transform(&z, &n).is_err());
}
