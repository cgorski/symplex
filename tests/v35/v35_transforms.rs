//! Integral transforms after the 0.39 hunt.  Every test names what was
//! wrong before and the oracle of its reference value (mpmath 1.3.0 at
//! `mp.dps = 30`, or SymPy 1.14).

use symplex::prelude::*;

fn c64(e: &Ex) -> Complex64 {
    e.eval_complex64()
        .unwrap_or_else(|err| panic!("{e}: {err}"))
}

/// `L{f(t)/t} = ∫_s^∞ F(u) du` was taken with the real-variable
/// antiderivative: `L{(e^{−2t} − e^{−3t})/t}` came out as `ln|s + 3| −
/// ln|s + 2|`, which is not analytic in `s` (at `s = 1 + i` its imaginary
/// part is 0).  The transform is `ln(s + 3) − ln(s + 2)`.  mpmath:
/// quad((exp(-2t) - exp(-3t))/t·exp(-(1+i)t), [0, 1, inf]) =
/// 0.265314125531085198 − 0.076771891269778039i.
#[test]
fn laplace_division_by_t_is_analytic_in_s() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let f = ((&t * -2).exp() - (&t * -3).exp()) / &t;
    let big_f = f.try_laplace(&t, &s).unwrap();
    let shown = big_f.to_string();
    assert!(!shown.contains("abs"), "{shown}");
    let z = c64(&big_f.subs(&s, &(1 + ctx.i_unit())));
    assert!(
        (z - Complex64::new(0.265_314_125_531_085_2, -0.076_771_891_269_778_04)).norm() < 1e-14,
        "{shown}: {z}"
    );

    // A parameter declared positive: (e^{−bt} − e^{−3t})/t → ln((s + 3)/(s + b)).
    // mpmath at b = 1, s = 2 + i: log((5+i)/(3+i)) = quad((exp(-t) -
    // exp(-3t))/t·exp(-(2+i)t), [0, 1, inf]) = 0.47775572251371818 − 0.12435499454676144i.
    let b = ctx.symbol_with("b", &[Assumption::Positive]).unwrap();
    let g = ((-(&b * &t)).exp() - (&t * -3).exp()) / &t;
    let big_g = g.try_laplace(&t, &s).unwrap();
    let z = c64(&big_g.subs(&b, &ctx.int(1)).subs(&s, &(2 + ctx.i_unit())));
    assert!(
        (z - Complex64::new(0.477_755_722_513_718_2, -0.124_354_994_546_761_44)).norm() < 1e-14,
        "{big_g}: {z}"
    );
}
