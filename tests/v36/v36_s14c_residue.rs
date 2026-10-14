//! Tests of session 14 (s14c): residues and the evidence ratchet.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn residue_at_0(src: &str) -> String {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let f = ctx.parse(src).unwrap();
    f.residue(&z, &ctx.int(0)).to_string()
}

/// Up to 0.50 the order of a pole was decided by a 20-digit decimal: a
/// coefficient of the shifted denominator below `10⁻¹⁸` counted as 0.  The
/// simple pole of `1/(z(z + e⁻⁵⁰))` at `0` was taken for a double pole and
/// the residue came out `0`; for `e^z/(z² + e⁻⁵⁰z)` it was `1` (the residue
/// of `e^z/z²`).  The coefficients are now decided by
/// `calculus_util::hidden_zero` (exact or certified evidence).
///
/// Oracle (exact): `1/(z(z + δ)) = (1/z − 1/(z + δ))/δ`, so the residue at
/// `0` is `1/δ = e⁵⁰`, and `e^z/(z(z + δ))` has `e⁰/δ = e⁵⁰`; mpmath
/// `(1/2πi)∮_{|z| = δ/2}` gives `5.18470552858707e21` for both.
#[test]
fn a_tiny_coefficient_does_not_raise_the_order_of_a_pole() {
    assert_eq!(residue_at_0("1/(z*(z+exp(-50)))"), "exp(50)");
    assert_eq!(residue_at_0("exp(z)/(z^2+exp(-50)*z)"), "exp(50)");
}

/// A coefficient that is an exact zero hidden in radicals plus a tiny
/// constant, `√2·√3 − √6 + e⁻⁴⁵ = e⁻⁴⁵`: up to 0.50 its decimal was below
/// `10⁻¹⁸` and the residue of `1/(z² + c·z)` at `0` came out `0`.  Without
/// the tiny part the pole is double and the residue is `0`; the zero is
/// proved (not read off a decimal).
///
/// Oracle (exact): `√2·√3 = √6`, so `c = e⁻⁴⁵` and the residue is `1/c =
/// e⁴⁵` (mpmath, contour of radius `c/2`: `3.49342710574851e19`); with
/// `c = 0` it is the residue of `1/z²`, `0`, and of `e^z/z²`, `1`.
#[test]
fn a_hidden_zero_plus_a_tiny_constant_is_not_zero() {
    assert_eq!(
        residue_at_0("1/(z^2+(sqrt(2)*sqrt(3)-sqrt(6)+exp(-45))*z)"),
        "exp(45)"
    );
    assert_eq!(residue_at_0("1/(z^2+(sqrt(2)*sqrt(3)-sqrt(6))*z)"), "0");
    assert_eq!(
        residue_at_0("exp(z)/(z^2+(sqrt(2)*sqrt(3)-sqrt(6))*z)"),
        "1"
    );
}

/// The hunt of s14c (rational and exponential functions with pole
/// locations `0` and `−c`, `c` tiny, an exact zero in radicals or
/// logarithms, or both; at `0`, `1` and `i`) found 1,604 wrong residues
/// out of 2,970 at 0.50; none now.  Two of the families, at a point
/// other than `0`: `(z − 1)⁻²(z − 1 + c)⁻¹` and `(z − i)⁻¹(z − i + c)⁻²`
/// with `c = (√2 − 1)⁶⁰ ≈ 1.1·10⁻²³` (0.50: `0` and `0`).
///
/// Oracle (exact partial fractions): `Res_{w=0} 1/(w²(w + c)) = −1/c²`
/// and `Res_{w=0} 1/(w(w + c)²) = 1/c²`; mpmath (contour of radius
/// `c/2`, 30 and 50 digits): `∓7.4·10⁴⁵`.
#[test]
fn tiny_pole_offsets_at_other_points() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let c2 = ctx.parse("(sqrt(2)-1)^120").unwrap();
    let f = ctx.parse("1/((z-1)^2*(z-1+(sqrt(2)-1)^60))").unwrap();
    let r = f.residue(&z, &ctx.int(1));
    let v = (&r * &c2).eval_f64().unwrap();
    assert!((v + 1.0).abs() < 1e-12, "{r}");
    let g = ctx.parse("1/((z-I)*(z-I+(sqrt(2)-1)^60)^2)").unwrap();
    let r = g.residue(&z, &ctx.parse("I").unwrap());
    let v = (&r * &c2).eval_f64().unwrap();
    assert!((v - 1.0).abs() < 1e-12, "{r}");
}

/// Parameters keep the generic reading: `1/(z(z + a))` has a simple pole
/// at `0` with residue `1/a` (the coefficient `a` is certified nonzero at
/// a sample point, so nonzero for generic `a`), and `1/(z² + (a − a)z)`
/// a double one.
///
/// Oracle (exact): partial fractions, `1/(z(z + a)) = (1/z − 1/(z + a))/a`.
#[test]
fn parametric_residues_keep_the_generic_reading() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let a = ctx.symbol("a");
    let f = ctx.parse("1/(z*(z+a))").unwrap();
    let r = f.residue(&z, &ctx.int(0));
    assert!((&r - &a.powi(-1)).simplify().is_zero_structural(), "{r}");
    let g = ctx
        .parse("1/(z^2+(a*(sqrt(2)*sqrt(3)-sqrt(6)))*z)")
        .unwrap();
    assert_eq!(g.residue(&z, &ctx.int(0)).to_string(), "0");
}

/// `L{1/((t − 10⁻¹⁰)² + 1)}(s)` is a sum over the poles `ρ = 10⁻¹⁰ ± i` of
/// `Res·L{1/(t − ρ)}`, and `L{1/(t − ρ)}(s)` is `e^{−ρs}E₁(−ρs)` for `Re ρ ≤
/// 0` but `e^{−ρs}(±iπ − Ei(ρs))` for `Re ρ > 0`.  Up to 0.50 the form was
/// chosen by `Re ρ ≤ 10⁻⁹|ρ|` in floating point, so these poles took the
/// `E₁` form, wrong by `2πi·e^{−ρs}·Res` between the cut and the imaginary
/// axis: at `s = 10⁻¹² − i` the transform was `−0.5778636749 +
/// 0.6467611227i`.  The sign of `Re ρ` is now exact (here `−b/2a`).
///
/// Oracle: mpmath, the integral `∫₀^∞ f(t)e^{−st} dt` along the ray `arg t
/// = π/4` (no pole between it and the real axis), 30 and 45 digits:
/// `0.577863674930835 + 0.646761122836339i`; at `s = 1/2` (real, both
/// forms agree) `0.860526765783132`.
#[test]
fn laplace_of_a_pole_just_right_of_the_imaginary_axis() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let f = ctx.parse("1/((t-10^(-10))^2+1)").unwrap();
    let l = f.try_laplace(&t, &s).unwrap();
    let at = |v: &str| l.subs(&s, &ctx.parse(v).unwrap()).eval_complex64().unwrap();
    let w = at("10^(-12)-I");
    assert!(
        (w.re - 0.577863674930835).abs() < 1e-9 && (w.im - 0.646761122836339).abs() < 1e-9,
        "{w}"
    );
    let w = at("1/2");
    assert!(
        (w.re - 0.860526765783132).abs() < 1e-12 && w.im.abs() < 1e-12,
        "{w}"
    );
}

/// A real pole at `−10⁻¹³` is off `[0, ∞)`, so `L{1/((t + 10⁻¹³)(t + 2))}`
/// converges for `s > 0`.  Up to 0.50 a real pole with `Re ρ > −10⁻¹²|ρ|`
/// counted as one on `[0, ∞)` and the transform was refused.
///
/// Oracle: `(e^{as}E₁(as) − e^{2s}E₁(2s))/(2 − a)` with `a = 10⁻¹³`
/// (partial fractions), mpmath at `s = 1/2`: `14.7265951811304`; the same
/// by quadrature at 30 and 45 digits.
#[test]
fn laplace_of_a_pole_just_left_of_zero() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let f = ctx.parse("1/((t+10^(-13))*(t+2))").unwrap();
    let l = f.try_laplace(&t, &s).unwrap();
    let v = l.subs(&s, &ctx.parse("1/2").unwrap()).eval_f64().unwrap();
    assert!((v - 14.7265951811304).abs() < 1e-10, "{v}");
}
