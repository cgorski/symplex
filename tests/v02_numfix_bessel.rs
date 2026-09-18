//! Regression tests for the large-argument Bessel bug (`J0(20)` evaluated
//! to `0.333` instead of `0.167`): the Hankel `P/Q` expansion double-counted
//! `a_0`, had the sign of `Q` flipped, and was switched on far too early
//! (`x ≈ 11.5` where its optimal-truncation error is only `~e^{-2x}`).
//!
//! Reference values are mpmath at 40 digits.

use symplex::prelude::*;

/// Reference values `(x, J0, J1, J5, Y0, Y1, I0, K0)` from mpmath (`mp.dps = 40`).
const REF: &[(f64, [&str; 7])] = &[
    (
        0.5,
        [
            "0.93846980724081290422840467359971263",
            "0.24226845767487388638395457614153164",
            "0.0000080536272413574740859781853303090647",
            "-0.44451873350670655714839847506833191",
            "-1.4714723926702430691885846353232975",
            "1.0634833707413235192631844154453565",
            "0.92441907122766586178192416753021699",
        ],
    ),
    (
        5.0,
        [
            "-0.17759677131433830434739701307475871",
            "-0.32757913759146522203773432191016913",
            "0.26114054612017009005480553851291853",
            "-0.30851762524903378007364898421204661",
            "0.14786314339122684480105067548803721",
            "27.239871823604446894544232075884419",
            "0.0036910983340425942747352610074569951",
        ],
    ),
    (
        12.0,
        [
            "0.047689310796833536623811689141429138",
            "-0.22344710449062761236769771636429716",
            "-0.073470963101658581265788425544661515",
            "-0.22523731263436143368768510978745849",
            "-0.057099218260896521050415271335470805",
            "18948.925349296308861208108229753966",
            "0.0000022008253973114914005155997745454167",
        ],
    ),
    (
        20.0,
        [
            "0.16702466434058315472732054470138404",
            "0.06683312417585004557899297419364672",
            "0.15116976798239497460710045572485227",
            "0.062640596809383831161729014105358174",
            "-0.16551161436252129586397602324369624",
            "43558282.559553533272106660089217692",
            "0.00000000057412378153365242927167020616229738",
        ],
    ),
    (
        50.0,
        [
            "0.055812327669251815004750478529433968",
            "-0.097511828125175137661458953873701614",
            "-0.081400247696569639643974037928222066",
            "-0.098064995470077079029211453440370432",
            "-0.05679566856201476794181954923776336",
            "293255378384933632665.46750794568539",
            "3.4101677497894955139206755123529522e-23",
        ],
    ),
    (
        100.0,
        [
            "0.019985850304223122424228390950848991",
            "-0.07714535201411215803268549492723447",
            "-0.074195736964513920834135049813019587",
            "-0.077244313365083152254228221367198771",
            "-0.02037231200275979330470393266641456",
            "1.0737517071310738235197208576034947e+42",
            "4.6566282291759020189390052894838864e-45",
        ],
    ),
];

const NAMES: [&str; 7] = ["J0", "J1", "J5", "Y0", "Y1", "I0", "K0"];

fn point(ctx: &Context, x: f64) -> Ex {
    if x == 0.5 {
        ctx.rational(1, 2)
    } else {
        ctx.int(x as i64)
    }
}

fn functions(ctx: &Context, x: &Ex) -> [Ex; 7] {
    [
        x.bessel_j(&ctx.int(0)),
        x.bessel_j(&ctx.int(1)),
        x.bessel_j(&ctx.int(5)),
        x.bessel_y(&ctx.int(0)),
        x.bessel_y(&ctx.int(1)),
        x.bessel_i(&ctx.int(0)),
        x.bessel_k(&ctx.int(0)),
    ]
}

/// Significant digits (leading-zero stripped mantissa) as a string.
fn sig_digits(s: &str) -> String {
    let mant = s.split(['e', 'E']).next().unwrap_or(s);
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.trim_start_matches('0').to_string()
}

/// Number of leading significant digits on which `got` and `want` agree.
fn agreeing_digits(got: &str, want: &str) -> usize {
    let g = sig_digits(got);
    let w = sig_digits(want);
    g.chars().zip(w.chars()).take_while(|(a, b)| a == b).count()
}

fn assert_digits(got: &str, want: &str, min: usize, what: &str) {
    // Magnitude / sign must agree (guards the digit-string comparison).
    let gf: f64 = got
        .parse()
        .unwrap_or_else(|_| panic!("{what}: unparsable {got}"));
    let wf: f64 = want.parse().unwrap();
    assert!(
        ((gf - wf) / wf).abs() < 1e-13,
        "{what}: magnitude mismatch {got} vs {want}"
    );
    let n = agreeing_digits(got, want);
    assert!(n >= min, "{what}: only {n} digits agree: {got} vs {want}");
}

fn check_30_digits(idx: usize) {
    let ctx = Context::new();
    for (x, refs) in REF {
        let e = &functions(&ctx, &point(&ctx, *x))[idx];
        let got = e
            .eval_decimal(30)
            .unwrap_or_else(|err| panic!("{}({x}): {err}", NAMES[idx]));
        assert_digits(&got, refs[idx], 25, &format!("{}({x})", NAMES[idx]));
    }
}

#[test]
fn evalf_bessel_j0_30_digits() {
    check_30_digits(0);
}

#[test]
fn evalf_bessel_j1_30_digits() {
    check_30_digits(1);
}

#[test]
fn evalf_bessel_j5_30_digits() {
    check_30_digits(2);
}

#[test]
fn evalf_bessel_y0_30_digits() {
    check_30_digits(3);
}

#[test]
fn evalf_bessel_y1_30_digits() {
    check_30_digits(4);
}

#[test]
fn evalf_bessel_i0_30_digits() {
    check_30_digits(5);
}

#[test]
fn evalf_bessel_k0_30_digits() {
    check_30_digits(6);
}

/// The original reproducer: `J0(20)` and `Y0(30)` through `eval_f64`.
#[test]
fn evalf_bessel_f64_reproducer() {
    let ctx = Context::new();
    let v = ctx.int(20).bessel_j(&ctx.int(0)).eval_f64().unwrap();
    assert!((v - 0.16702466434058315).abs() < 1e-15, "J0(20) = {v}");
    let v = ctx.int(20).bessel_j(&ctx.int(5)).eval_f64().unwrap();
    assert!((v - 0.15116976798239497).abs() < 1e-15, "J5(20) = {v}");
    let y = ctx.int(30).bessel_y(&ctx.int(0)).eval_f64().unwrap();
    assert!((y + 0.11729573168666402).abs() < 1e-15, "Y0(30) = {y}");
}

/// `eval_f64` for every reference point (f64 path uses a lower working
/// precision than `eval_decimal(30)`, so exercise it separately).
#[test]
fn evalf_bessel_f64_all_points() {
    let ctx = Context::new();
    for (x, refs) in REF {
        for (i, e) in functions(&ctx, &point(&ctx, *x)).iter().enumerate() {
            let got = e
                .eval_f64()
                .unwrap_or_else(|err| panic!("{}({x}): {err}", NAMES[i]));
            let want: f64 = refs[i].parse().unwrap();
            let rel = ((got - want) / want).abs();
            assert!(
                rel < 1e-14,
                "{}({x}) = {got}, want {want}, rel {rel:e}",
                NAMES[i]
            );
        }
    }
}

/// Non-integer order and negative argument take different code paths.
#[test]
fn evalf_bessel_half_order_and_negative_argument() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    // Hankel regime (x = 50) and series regime (x = 3) for ν = 1/2.
    let got = ctx.int(50).bessel_j(&half).eval_decimal(30).unwrap();
    assert_digits(
        &got,
        "-0.029605831888924612568029521626679575",
        25,
        "J_{1/2}(50)",
    );
    let got = ctx.int(3).bessel_j(&half).eval_decimal(30).unwrap();
    assert_digits(
        &got,
        "0.065008182877375778114004696404628946",
        25,
        "J_{1/2}(3)",
    );
    // J_n(−x) = (−1)^n J_n(x) in both regimes.
    for x in [20i64, 35] {
        let j0 = ctx.int(-x).bessel_j(&ctx.int(0)).eval_decimal(30).unwrap();
        let j0p = ctx.int(x).bessel_j(&ctx.int(0)).eval_decimal(30).unwrap();
        assert_eq!(j0, j0p, "J0(−{x})");
        let j1 = ctx.int(-x).bessel_j(&ctx.int(1)).eval_f64().unwrap();
        let j1p = ctx.int(x).bessel_j(&ctx.int(1)).eval_f64().unwrap();
        assert_eq!(j1, -j1p, "J1(−{x})");
    }
}

/// Very large argument: the phase `x − νπ/2 − π/4` needs extra bits.
#[test]
fn evalf_bessel_huge_argument() {
    let ctx = Context::new();
    let x = ctx.int(1_000_000);
    let got = x.bessel_j(&ctx.int(0)).eval_decimal(30).unwrap();
    assert_digits(
        &got,
        "0.00033104301373987374098796304221962544",
        25,
        "J0(1e6)",
    );
    let got = x.bessel_y(&ctx.int(0)).eval_decimal(30).unwrap();
    assert_digits(
        &got,
        "-0.00072596852233517916568272174336768651",
        25,
        "Y0(1e6)",
    );
}

/// Higher orders at moderate x (the Hankel heuristic depends on ν).
#[test]
fn evalf_bessel_order_ten() {
    let ctx = Context::new();
    let x = ctx.int(20);
    let got = x.bessel_j(&ctx.int(10)).eval_decimal(30).unwrap();
    assert_digits(&got, "0.18648255802394508321410826451219876", 25, "J10(20)");
    let got = x.bessel_y(&ctx.int(10)).eval_decimal(30).unwrap();
    assert_digits(
        &got,
        "-0.043894653515658394899365436176372007",
        25,
        "Y10(20)",
    );
}

/// Other special functions at large arguments (all were already right;
/// pinned here against mpmath so they stay that way).
#[test]
fn evalf_trig_exp_integrals_large_argument() {
    let ctx = Context::new();
    let cases: &[(&str, Ex, &str)] = &[
        (
            "Si(50)",
            ctx.int(50).si(),
            "1.5516170724859358947279855948593775",
        ),
        (
            "Ci(100)",
            ctx.int(100).ci(),
            "-0.0051488251426104921444435539053444979",
        ),
        (
            "Ei(40)",
            ctx.int(40).ei(),
            "6039718263611241.5783592314185106913",
        ),
        (
            "li(40)",
            ctx.int(40).li(),
            "15.839544272256263910693699094336842",
        ),
        (
            "I0(30)",
            ctx.int(30).bessel_i(&ctx.int(0)),
            "781672297823.97748971738981670529501",
        ),
        (
            "K0(30)",
            ctx.int(30).bessel_k(&ctx.int(0)),
            "2.1324774964630563711668960629653764e-14",
        ),
    ];
    for (what, e, want) in cases {
        let got = e
            .eval_decimal(30)
            .unwrap_or_else(|err| panic!("{what}: {err}"));
        assert_digits(&got, want, 25, what);
    }
}

/// The `compile()` runtime must be within 1e-12 (relative) at the same points.
#[test]
fn compiled_runtime_bessel_1e12() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let fns = functions(&ctx, &x).map(|e| e.compile(&["x"]).expect("compiles"));
    for (p, refs) in REF {
        for (i, f) in fns.iter().enumerate() {
            let got = f(&[*p]);
            let want: f64 = refs[i].parse().unwrap();
            let rel = ((got - want) / want).abs();
            assert!(
                rel < 1e-12,
                "{}({p}) = {got}, want {want}, rel {rel:e}",
                NAMES[i]
            );
        }
    }
}
