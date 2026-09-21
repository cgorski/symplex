//! Complex analysis and the new special functions/constants (symplex 0.2).
//!
//! Demonstrates:
//! - `re`, `im`, `conjugate`, `arg`, `abs_squared`, `as_real_imag`,
//!   `expand_complex`, `polar`, `is_real_valued`;
//! - honest handling of symbols with *unknown* realness: `re(z)` stays
//!   `re(z)` unless `z` is assumed real (0.1 silently assumed real);
//! - the constants `EulerGamma`, `Catalan`, `GoldenRatio` and
//!   `complex_infinity` (`zoo`);
//! - `si`, `ci`, `ei`, `li`, `zeta`, `polygamma`, `kronecker_delta` with
//!   exact values, derivatives and arbitrary-precision evaluation;
//! - residues at simple and higher-order poles, roots of unity.
//!
//! Run with: `cargo run --example complex_analysis`

use symplex::prelude::*;

fn main() {
    println!("=== Complex Analysis ===\n");

    let ctx = Context::new();
    let i = ctx.i_unit();
    let z = ctx.symbol("z"); // nothing assumed: may be complex
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let y = ctx.symbol_with("y", &[Assumption::Real]);

    // ── 1. Real/imaginary parts with assumptions ────────────────────────
    println!("--- re / im / conjugate / arg (x, y real; z unknown) ---");
    let w = &x + &i * &y;
    println!("w = {w}");
    println!("re(w)         = {}", w.re());
    println!("im(w)         = {}", w.im());
    println!("conjugate(w)  = {}", w.conjugate());
    println!("|w|²          = {}", w.abs_squared());
    println!("arg(w)        = {}", w.arg());
    let (re2, im2) = w.powi(2).as_real_imag();
    println!("w²            = ({re2}) + ({im2})·i");
    let (re_e, im_e) = w.exp().as_real_imag();
    println!("exp(w)        = ({re_e}) + ({im_e})·i");
    let (re_s, im_s) = w.sin().as_real_imag();
    println!("sin(w)        = ({re_s}) + ({im_s})·i");
    let (re_r, im_r) = (1 / &w).as_real_imag();
    println!("1/w           = ({re_r}) + ({im_r})·i");
    println!("expand_complex(exp(w)) = {}", w.exp().expand_complex());

    println!();
    println!("Without a realness assumption nothing is guessed:");
    println!("re(z)         = {}", z.re());
    println!("conjugate(z)  = {}", z.conjugate());
    println!(
        "conjugate(z²+1) = {}   (conjugation distributes)",
        (&z.powi(2) + 1).conjugate()
    );
    println!("conjugate(exp(z)) = {}", z.exp().conjugate());
    println!("re(i·z)       = {}", (&i * &z).re());
    println!("re(exp(z))    = {}", z.exp().re());
    println!("im(conj(z))   = {}", z.conjugate().im());
    println!(
        "is_real_valued: x²+1 → {:?},  x+iy → {:?},  z → {:?}",
        (&x.powi(2) + 1).is_real_valued(),
        w.is_real_valued(),
        z.is_real_valued()
    );

    // ── 2. Concrete complex arithmetic ──────────────────────────────────
    println!("\n--- Concrete complex numbers ---");
    let a = ctx.complex(&ctx.int(3), &ctx.int(4)); // 3 + 4i
    let b = ctx.int(1) - &i * 2; // 1 − 2i
    println!("a = {a},  b = {b}");
    println!("a·b           = {}", (&a * &b).expand());
    let (qr, qi) = (&a / &b).as_real_imag();
    println!("a/b           = {qr} + {qi}·i");
    println!("|a|²          = {}", a.abs_squared());
    println!("arg(1+i)      = {}", (ctx.int(1) + &i).arg().eval());
    println!("arg(−1)       = {}", ctx.int(-1).arg().eval());
    println!("(1+i)⁸        = {}", (ctx.int(1) + &i).powi(8).expand());
    println!("exp(iπ)       = {}", (&i * &ctx.pi()).exp().eval());
    println!("exp(iπ/3)     = {}", (&i * &ctx.pi() / 3).exp().eval());
    println!("ln(−1)        = {}", ctx.int(-1).ln().eval());
    println!("√(−4)         = {}", ctx.int(-4).sqrt());
    println!("cos(i·x)      = {}", (&i * &x).cos().expand_complex());
    println!("sin(i·x)      = {}", (&i * &x).sin().expand_complex());
    println!(
        "i^i           ≈ {}   (exact form stays i^i)",
        i.pow(&i).eval_decimal(20).unwrap()
    );
    let exp_i = i.exp().eval_complex64().unwrap();
    println!("exp(i) as f64 = ({:.6}, {:.6})", exp_i.re, exp_i.im);

    // ── 3. Roots of unity ───────────────────────────────────────────────
    println!("\n--- Roots of unity ---");
    for k in [3, 4, 6] {
        let roots = (&z.powi(k) - 1).solve(&z).unwrap();
        let shown: Vec<String> = roots.iter().map(|r| r.to_string()).collect();
        println!("z^{k} = 1 → {}", shown.join(",  "));
    }

    // ── 4. Complex infinity ─────────────────────────────────────────────
    println!("\n--- Complex infinity ---");
    println!("1/0           = {}", (1 / &ctx.int(0)).eval());
    println!(
        "parse(\"zoo\") = {}",
        symplex::parse::parse(&ctx, "zoo").unwrap()
    );
    println!(
        "zoo == complex_infinity(): {}",
        ctx.complex_infinity() == (1 / &ctx.int(0)).eval()
    );

    // ── 5. New constants ────────────────────────────────────────────────
    println!("\n--- Constants ---");
    let gamma = ctx.euler_gamma();
    println!("γ  = {gamma} ≈ {}", gamma.eval_decimal(30).unwrap());
    println!(
        "G  = {} ≈ {}",
        ctx.catalan(),
        ctx.catalan().eval_decimal(30).unwrap()
    );
    let phi = ctx.golden_ratio();
    println!("φ  = {phi} ≈ {}", phi.eval_decimal(30).unwrap());
    // φ² − φ − 1 = 0: the simplifier does not yet reduce this symbolically,
    // but the numeric recognizer and f64 evaluation both see zero.
    let phi_poly = &phi.powi(2) - &phi - 1;
    println!(
        "φ² − φ − 1: simplify → {},  nsimplify → {},  f64 → {}",
        phi_poly.simplify(),
        phi_poly.nsimplify(1e-12),
        phi_poly.eval_f64().unwrap()
    );

    // ── 6. Special functions ────────────────────────────────────────────
    println!("\n--- Special functions ---");
    println!("ψ(1)          = {}", ctx.int(1).digamma().eval());
    println!("ψ(5)          = {}", ctx.int(5).digamma().eval());
    println!("ψ(1/2)        = {}", ctx.rational(1, 2).digamma().eval());
    println!(
        "ψ'(1) = polygamma(1, 1) = {}",
        ctx.int(1).polygamma(&ctx.int(1)).eval()
    );
    println!(
        "polygamma(2, 1) = {}",
        ctx.int(1).polygamma(&ctx.int(2)).eval()
    );
    println!("d/dx ψ(x)     = {}", x.digamma().diff(&x));
    println!(
        "ζ(2) = {},  ζ(4) = {},  ζ(0) = {},  ζ(−1) = {}",
        ctx.int(2).zeta().eval(),
        ctx.int(4).zeta().eval(),
        ctx.int(0).zeta().eval(),
        ctx.int(-1).zeta().eval()
    );
    println!(
        "ζ(3)          = {} ≈ {}   (no closed form; stays symbolic)",
        ctx.int(3).zeta().eval(),
        ctx.int(3).zeta().eval_decimal(30).unwrap()
    );
    println!(
        "Si(0) = {},  Si(∞) = {}",
        ctx.int(0).si().eval(),
        ctx.infinity().si().eval()
    );
    println!(
        "d/dx Si(x) = {},  d/dx Ci(x) = {}",
        x.si().diff(&x),
        x.ci().diff(&x)
    );
    println!(
        "d/dx Ei(x) = {},  d/dx li(x) = {}",
        x.ei().diff(&x),
        x.li().diff(&x)
    );
    println!("Si(1) ≈ {}", ctx.int(1).si().eval_decimal(25).unwrap());
    println!("Ei(1) ≈ {}", ctx.int(1).ei().eval_decimal(25).unwrap());
    println!("li(2) ≈ {}", ctx.int(2).li().eval_decimal(25).unwrap());
    println!(
        "δ(z, z) = {},  δ(1, 2) = {},  δ(x, z) = {}",
        z.kronecker_delta(&z),
        ctx.int(1).kronecker_delta(&ctx.int(2)),
        x.kronecker_delta(&z)
    );
    println!("d/dx J₀(x)    = {}", x.bessel_j(&ctx.int(0)).diff(&x));
    println!(
        "I₁(2) ≈ {}",
        ctx.int(2).bessel_i(&ctx.int(1)).eval_decimal(20).unwrap()
    );
    println!(
        "K₀(2) ≈ {}",
        ctx.int(2).bessel_k(&ctx.int(0)).eval_decimal(20).unwrap()
    );
    println!(
        "P₁₀(1/3)      = {}",
        ctx.rational(1, 3).legendre(&ctx.int(10)).eval()
    );

    // ── 7. Residues ─────────────────────────────────────────────────────
    println!("\n--- Residues ---");
    let f = 1 / (&z.powi(2) + 1);
    println!("Res_{{z=i}}  1/(z²+1) = {}", f.residue(&z, &i));
    println!("Res_{{z=−i}} 1/(z²+1) = {}", f.residue(&z, &(-&i)));
    let g = &z.exp() / &z.powi(3);
    println!(
        "Res_{{z=0}}  eᶻ/z³    = {}   (third-order pole)",
        g.residue(&z, &ctx.int(0))
    );
    let h = &z.cos() / &z.powi(2);
    println!("Res_{{z=0}}  cos z/z² = {}", h.residue(&z, &ctx.int(0)));
    println!("Res_{{z=∞}}  1/(z²+1) = {}", f.residue_at_infinity(&z));

    println!("\n✓ Done!");
}
