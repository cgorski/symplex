//! `evalf` of the zeta family at complex arguments, of slowly converging
//! sums of smooth terms, and the bound of long products (0.44).  Reference
//! values: mpmath 1.3 (`python3 -c "from mpmath import *; mp.dps = 45;
//! print(…)"`), each cited at its test.

use std::time::Duration;

use symplex::prelude::*;

use crate::perf_assert::CpuTimer;

/// `eval_decimal(digits)` of `src`, which must start with `want`.
fn starts(src: &str, digits: u32, want: &str) {
    let ctx = Context::new();
    let e = ctx.parse(src).unwrap();
    let got = e.eval_decimal(digits).unwrap();
    assert!(
        got.starts_with(want),
        "{src} at {digits}: {got} (want {want}…)"
    );
}

/// The real and imaginary parts of `eval_decimal(digits)` (printed `a + b*i`
/// or `a - b*i`) start with `re` and `im`.
fn starts_c(src: &str, digits: u32, re: &str, im: &str) {
    let ctx = Context::new();
    let e = ctx.parse(src).unwrap();
    let got = e.eval_decimal(digits).unwrap();
    let (a, b) = match got.rsplit_once(" + ") {
        Some((a, b)) => (a.to_string(), b.trim_end_matches("*i").to_string()),
        None => {
            let (a, b) = got.rsplit_once(" - ").unwrap();
            (a.to_string(), format!("-{}", b.trim_end_matches("*i")))
        }
    };
    assert!(
        a.starts_with(re) && b.starts_with(im),
        "{src} at {digits}: {got} (want {re}, {im})"
    );
}

/// `zeta(1/2 + 14i)` was refused ("zeta of complex argument not yet
/// supported in evalf"), in `eval_complex64` too.  mpmath `zeta(0.5+14j)` =
/// `0.022241142609993589246213199204 − 0.103258123266450057902363095553i`.
#[test]
fn zeta_on_the_critical_line() {
    starts_c(
        "zeta(1/2 + 14*I)",
        30,
        "0.022241142609993589246213199",
        "-0.1032581232664500579023630955",
    );
    let ctx = Context::new();
    let z = ctx
        .parse("zeta(1/2 + 14*I)")
        .unwrap()
        .eval_complex64()
        .unwrap();
    assert!((z.re - 0.022_241_142_609_993_59).abs() < 1e-16);
    assert!((z.im + 0.103_258_123_266_450_06).abs() < 1e-16);
}

/// Left of 0 by the functional equation, and an inexact argument (refused
/// before).  mpmath `zeta(-2.5+3j)` = `0.0687636790336464816284776659999 +
/// 0.13398028393783442697393013743i`; `zeta(sqrt(2)/2 + 14j)` =
/// `0.165610234313853526832256561200183700496 − 0.0666219780430703008662791186511304794834i`.
#[test]
fn zeta_left_of_zero_and_inexact() {
    starts_c(
        "zeta(-5/2 + 3*I)",
        30,
        "0.06876367903364648162847766599",
        "0.1339802839378344269739301374",
    );
    starts_c(
        "zeta(sqrt(2)/2 + 14*I)",
        35,
        "0.16561023431385352683225656120018",
        "-0.066621978043070300866279118651130",
    );
}

/// Next to a zero (`1/2 + 14.134725i`) the value is 6 orders below its
/// terms: the loss is made up.  mpmath `zeta(mpf(1)/2 +
/// mpf(14134725)/1000000*1j)` = `1.7674298356433245354517983702553163385e-8
/// − 1.11020288948576643564804419097304159836e-7i`.
#[test]
fn zeta_next_to_a_zero() {
    starts_c(
        "zeta(1/2 + 14134725/1000000*I)",
        30,
        "1.767429835643324535451798370",
        "-1.110202889485766435648044190",
    );
}

/// `polygamma(5, 5 + i)` was refused ("polygamma of complex arguments not
/// yet supported").  mpmath `polygamma(5, 5+1j)` =
/// `0.00525426966237870789174361197423 − 0.00960700076289637727242013103488i`;
/// `polygamma(2, mpf(-7)/2 + 1j/mpf(3))` =
/// `−0.0603313747787315211469800156138516535416 + 18.8950022568613423177303825181466164245i`.
#[test]
fn polygamma_of_complex_arguments() {
    starts_c(
        "polygamma(5, 5 + I)",
        30,
        "0.005254269662378707891743611974",
        "-0.009607000762896377272420131034",
    );
    starts_c(
        "polygamma(2, -7/2 + I/3)",
        30,
        "-0.06033137477873152114698001561",
        "18.89500225686134231773038251",
    );
}

/// `digamma` and `harmonic` of complex arguments (refused before).  mpmath
/// `digamma(1+1j)` = `0.0946503206224769772718784827219 +
/// 1.07667404746858117413405079475i`; `digamma(-100+1j)` =
/// `4.61021135278774915067856142163844638969 + 3.14339825662099015669340430314321606265i`;
/// `harmonic(0.5+1j)` = `0.925454013907120317813909105370308212743 +
/// 0.764940517815879282638124504955773948001i`.
#[test]
fn digamma_and_harmonic_of_complex_arguments() {
    starts_c(
        "digamma(1 + I)",
        30,
        "0.09465032062247697727187848272",
        "1.076674047468581174134050794",
    );
    starts_c(
        "digamma(-100 + I)",
        30,
        "4.6102113527877491506785614216",
        "3.1433982566209901566934043031",
    );
    starts_c(
        "harmonic(1/2 + I)",
        30,
        "0.92545401390712031781390910537",
        "0.76494051781587928263812450495",
    );
}

/// `gamma`, `factorial` and `beta` of complex arguments (refused before),
/// next to a pole too.  mpmath `gamma(1+1j)` =
/// `0.498015668118356042713691117462 − 0.154949828301810685124955130484i`;
/// `gamma(-3 + mpf(10)**-20*1j)` = `−0.209352944738633412121136873875155150382
/// + 16666666666666666666.6666666666666666667i`; `beta(1+1j, 2-0.5j)` =
/// `0.1397038132885542769375016324132352859629 − 0.2109257713486453960603958630941289707822i`
/// (and `beta(1+1j, 1-1j)` = `0.272029054982133162950236583672037555841`, real).
#[test]
fn gamma_factorial_beta_of_complex_arguments() {
    starts_c(
        "gamma(1 + I)",
        30,
        "0.4980156681183560427136911174",
        "-0.1549498283018106851249551304",
    );
    starts_c(
        "factorial(I)",
        30,
        "0.4980156681183560427136911174",
        "-0.1549498283018106851249551304",
    );
    starts_c(
        "gamma(-3 + 10^(-20)*I)",
        40,
        "-0.209352944738633412121136873875155150",
        "16666666666666666666.66666666666666666",
    );
    starts_c(
        "beta(1 + I, 2 - I/2)",
        30,
        "0.13970381328855427693750163241",
        "-0.21092577134864539606039586309",
    );
    starts("beta(1 + I, 1 - I)", 30, "0.2720290549821331629502365836");
}

/// `Γ(z̄) = conj Γ(z)` and `ζ(z̄) = conj ζ(z)`: a product or sum of a conjugate
/// pair is exactly real, so the square root of `Γ(1 + i)Γ(1 − i) − 1 = π/sinh π −
/// 1 < 0` is on its cut and takes the principal side.  mpmath `sqrt(1 -
/// pi/sinh(pi))` = `0.8532121336560250987543062084727557612683`; `2*re(zeta(0.5+14j))`
/// = `0.04448228521998717849242639840793725277357`.
#[test]
fn conjugate_pairs_are_exactly_real() {
    let ctx = Context::new();
    let r = ctx
        .parse("sqrt(gamma(1 + I)*gamma(1 - I) - 1)")
        .unwrap()
        .eval_decimal(30)
        .unwrap();
    assert!(
        r.starts_with("0.85321213365602509875430620847") && r.ends_with("*i"),
        "{r}"
    );
    let e = ctx.parse("zeta(1/2 + 14*I) + zeta(1/2 - 14*I)").unwrap();
    assert!((e.eval_f64().unwrap() - 0.044_482_285_219_987_18).abs() < 1e-17);
}

/// `dirichlet_eta(1/2 + i)` was refused; next to `s = 1` the factor `1 −
/// 2^(1−s)` is formed with as many guard bits as it is small.  mpmath
/// `altzeta(0.5+1j)` = `0.639861913936747431136431313776 +
/// 0.193514466045493766081002966304i`; `altzeta(1 + mpf(10)**-30*1j)` =
/// `0.693147180559945309417232121458176568075500134 + 1.59868903742430971756947870324916570496e-31i`.
#[test]
fn dirichlet_eta_of_complex_arguments() {
    starts_c(
        "dirichlet_eta(1/2 + I)",
        30,
        "0.6398619139367474311364313137",
        "0.1935144660454937660810029663",
    );
    starts_c(
        "dirichlet_eta(1 + 10^(-30)*I)",
        40,
        "0.6931471805599453094172321214581765680755",
        "1.5986890374243097175694787032491657",
    );
}

/// `erf`, `erfc`, `erfi` of complex arguments were refused ("erf of complex
/// argument not yet supported in evalf").  mpmath `erf(1+1j)` =
/// `1.31615128169794764488027108024367036903 + 0.190453469237834686284108861969162442438i`;
/// `erfc(mpc(10, 0.5))` = `−2.17257910063724902448223597972854640957e-45 +
/// 1.56647414170157992315912835567489803255e-45i` (45 digits of `erf` cancel);
/// `erf(mpc(sqrt(2), 3))` = `190.416992010059146921933482820405323448 −
/// 28.7102368429617112366284065385128439996i`; `erfi(1+1j)` swaps the parts of
/// `erf(1+1j)`.
#[test]
fn erf_family_of_complex_arguments() {
    starts_c(
        "erf(1 + I)",
        30,
        "1.3161512816979476448802710802",
        "0.19045346923783468628410886196",
    );
    starts_c(
        "erfc(10 + I/2)",
        30,
        "-2.1725791006372490244822359797",
        "1.5664741417015799231591283556",
    );
    starts_c(
        "erf(sqrt(2) + 3*I)",
        30,
        "190.41699201005914692193348282",
        "-28.710236842961711236628406538",
    );
    starts_c(
        "erfi(1 + I)",
        30,
        "0.19045346923783468628410886196",
        "1.3161512816979476448802710802",
    );
}

/// `polylog` of a complex order at `abs(z) ≥ 1` was refused ("polylog of
/// complex order at abs(z) >= 1 … not yet supported").  mpmath
/// `polylog(2+1j, 2)` = `1.56420703927663429923965391065356491727 −
/// 1.17962877557018853027932220518000197172i` (inversion);
/// `polylog(mpc(1.5,0.5), mpf(101)/100)` = `1.58552727615147428770501674782698128704
/// − 0.952854073700415068505732427897254732556i` (the expansion in `ln z`);
/// `polylog(mpc(-3.5,3), mpf(11)/10)` = `11.666759046750575123384092054748781310 −
/// 5.7479114807878146516229931166028986284i` (on the cut, below it);
/// `polylog(3+1j, 1)` = `zeta(3+1j)` = `1.10721440843140919562510020577522339768 −
/// 0.148290867178175348490764125668624680794i`.
#[test]
fn polylog_of_complex_order_off_the_disc() {
    starts_c(
        "polylog(2 + I, 2)",
        30,
        "1.5642070392766342992396539106",
        "-1.1796287755701885302793222051",
    );
    starts_c(
        "polylog(3/2 + I/2, 101/100)",
        30,
        "1.5855272761514742877050167478",
        "-0.95285407370041506850573242789",
    );
    starts_c(
        "polylog(-7/2 + 3*I, 11/10)",
        30,
        "11.666759046750575123384092054",
        "-5.7479114807878146516229931166",
    );
    starts_c(
        "polylog(3 + I, 1)",
        30,
        "1.1072144084314091956251002057",
        "-0.14829086717817534849076412566",
    );
}

/// The four sums of the report were refused ("neither hypergeometric … nor
/// a hypergeometric term times factors of polynomial growth").  mpmath
/// `-zeta(2, 1, 1)` = `0.93754825431584375370257409456786497789786`;
/// `nsum(lambda k: 1/(k**2+sqrt(2)), [0, inf], method='e')` (200 terms
/// directly) = `1.67593383514085910022745294629700942201615`;
/// `−(euler*log(2) − log(2)**2/2)` = `−0.1598689037424309717569478703249165704962`;
/// `pi**4/72` = `1.3529040421389227393950046206764598784684`.
#[test]
fn slowly_converging_sums_of_smooth_terms() {
    starts(
        "Sum(log(k)/k^2, k, 1, oo)",
        35,
        "0.93754825431584375370257409456786",
    );
    starts(
        "Sum(1/(k^2 + sqrt(2)), k, 0, oo)",
        35,
        "1.6759338351408591002274529462970",
    );
    starts(
        "Sum((-1)^k*log(k + 1)/(k + 1), k, 0, oo)",
        35,
        "-0.159868903742430971756947870324916",
    );
    starts(
        "Sum(harmonic(k)/k^3, k, 1, oo)",
        35,
        "1.3529040421389227393950046206764",
    );
    let ctx = Context::new();
    let v = ctx
        .parse("Sum(log(k)/k^2, k, 1, oo)")
        .unwrap()
        .eval_f64()
        .unwrap();
    assert!((v - 0.937_548_254_315_843_8).abs() < 1e-15);
}

/// More smooth terms, checked against closed forms: mpmath
/// `log(sinh(pi)/pi)` = `1.30184639860371267777043366300789533013`
/// (`Σ ln(1 + 1/k²)`), `zeta(mpf(3)/2, mpf(5)/2)` =
/// `1.40377976885682579581829820433759681831` (`Σ_{k≥1} (k + 3/2)^(−3/2)`),
/// `pi**2/12 − log(2)**2/2` = `0.5822405264650125059026563201596801087442`
/// (`−Σ (−1)^k H_k/k`), and `1/sqrt(mpf(5)/2) + nsum(lambda j:
/// binomial(-0.5, j)*(mpf(3)/2)**j*zeta(mpf(3)/2+j, 2), [0, inf])` =
/// `2.05932226217072023321174214628989581158` (`Σ 1/(k√(k + 3/2))`; mpmath's
/// `nsum` on the terms themselves is wrong from the 30th digit).  `Σ ln k/k`
/// diverges.
#[test]
fn smooth_sums_against_closed_forms() {
    starts(
        "Sum(log(1 + 1/k^2), k, 1, oo)",
        35,
        "1.3018463986037126777704336630078",
    );
    starts(
        "Sum((k + 3/2)^(-3/2), k, 1, oo)",
        35,
        "1.4037797688568257958182982043375",
    );
    starts(
        "Sum((-1)^k*harmonic(k)/k, k, 1, oo)",
        35,
        "-0.58224052646501250590265632015968",
    );
    starts(
        "Sum(1/(k*sqrt(k + 3/2)), k, 1, oo)",
        35,
        "2.0593222621707202332117421462898",
    );
    let ctx = Context::new();
    let r = ctx
        .parse("Sum(log(k)/k, k, 1, oo)")
        .unwrap()
        .eval_decimal(20);
    assert!(matches!(r, Err(SymplexError::Divergent { .. })), "{r:?}");
}

/// Logarithms in the denominator (negative powers of `ln k` in the
/// expansion), refused before.  `Σ_{k≥1} (−1)^k/ln(k + 1)` is minus
/// `0.92429989722293885595957018135959005377` (OEIS A099769; mpmath
/// `nsum(lambda k: (-1)**k/log(k+1), [1, inf])`); mpmath at 30 digits,
/// `fsum(f(k) for k < 8000) + 1/log(8001) + quad(lambda x:
/// 1/(x*(x+1)*log(x+1)**2), [8000, inf]) + f(8000)/2 − f′(8000)/12 +
/// f‴(8000)/720` for `f = 1/(k ln²(k + 1))` = `3.3877355319520023163695371791`
/// (`nsum` on the terms is wrong from the 3rd digit); `Σ 1/(k² ln(k + 1))` =
/// `1.88203974588235075456329696355129338746` (`nsum`, 200 terms first).
/// `Σ 1/(k ln(k + 1))` diverges.
#[test]
fn sums_with_logarithms_in_the_denominator() {
    starts(
        "Sum((-1)^k/log(k + 1), k, 1, oo)",
        20,
        "-0.924299897222938855",
    );
    starts(
        "Sum(1/(k*log(k + 1)^2), k, 1, oo)",
        20,
        "3.387735531952002316",
    );
    starts(
        "Sum(1/(k^2*log(k + 1)), k, 1, oo)",
        20,
        "1.882039745882350754",
    );
    let ctx = Context::new();
    let r = ctx
        .parse("Sum(1/(k*log(k + 1)), k, 1, oo)")
        .unwrap()
        .eval_decimal(20);
    assert!(matches!(r, Err(SymplexError::Divergent { .. })), "{r:?}");
}

/// A product of 2000 factors in `[1, 2)`: the bound of every step took
/// `abs(x) < 2^m` for `2^(m−1) ≤ abs(x)`, twice the factor, and grew like `2^2000`
/// (`PrecisionExhausted` at 20 digits).  mpmath `fprod(1 + mpf(1)/k**2 for k
/// in range(1, 2001))` = `3.67424079013296631133328412126821688497702`.
#[test]
fn long_products_keep_their_bound() {
    starts(
        "Product(1 + 1/k^2, k, 1, 2000)",
        20,
        "3.6742407901329663113",
    );
    starts(
        "Product(1 + 1/k^2, k, 1, 2000)",
        40,
        "3.674240790132966311333284121268216884977",
    );
}

/// `Γ(z)` for a complex `z` whose `ln Γ` lies beyond the exponent range
/// (about `1.49·10⁹`) is refused before `exp` is asked for it.
/// `fuzz_parser` found the text below: parsing builds a product with `0`,
/// whose finiteness test evaluates `((5288046 + 9926/70^i)!)!`, and
/// `exp(ln Γ)` ran at a working precision holding every bit of `Im ln Γ`
/// (over a minute; milliseconds now).  `Γ(10⁶ + 3i)` is still in range.
/// mpmath `gamma(10**6 + 3j)` = `-6.7931874052667e+5565702 -
/// 4.70580037779834e+5565702j`.
#[test]
fn gamma_beyond_the_exponent_range_is_refused_quickly() {
    let ctx = Context::new();
    let t = CpuTimer::start();
    let src = "3!e*bi2(vi/(05407/5*04890**01+9926/70**i)!!5014!0)6!iv!!0)6!iv";
    assert!(symplex::parse::parse(&ctx, src).is_err());
    for src in [
        "factorial(factorial(5288046 + 9926/70^I))",
        "gamma(10^8 + I)",
    ] {
        let r = ctx.parse(src).unwrap().eval_decimal(15);
        assert!(
            matches!(r, Err(SymplexError::Unevaluable { .. })),
            "{src}: {r:?}"
        );
    }
    t.assert_within(Duration::from_secs(20), "the parse error and the refusals");
    starts_c(
        "gamma(1000000 + 3*I)",
        15,
        "-6.7931874052667e5565702",
        "-4.70580037779834e5565702",
    );
}
