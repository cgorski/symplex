//! Tests of session 17, wave 17 (s17n1_zeros): accuracy of real special-function routes next to their zeros.
//! Every test says what was wrong before and cites its oracle.
//!
//! The arguments are dyadic rationals `N/2^b` whose numerator fits the working
//! precision of the digits asked for (`max(128, 3.4·digits + 64)` bits): exactly
//! representable, so that the argument has no error and the bound of the value
//! is the route's own.  Each is a zero of the function rounded to that many
//! bits (mpmath `besseljzero`, `besselyzero`, `airyaizero`, `airybizero`,
//! `findroot`), so the value is about `2^(−precision)`.
//!
//! Oracle of every value: mpmath 1.3.0 at `mp.dps = 150` and `230` (the two
//! agree to 70 digits), `target/scratch/n1/test_oracle.py` and `oracle.py` of
//! the wave; the call is given with each case.  The same cases are rows of
//! the wave's hunts (`h1`–`h5` and `deep`, 28,539 cases next to zeros: 0
//! wrong digits at 16, 30 and 50 digits and in `eval_f64` after the fix;
//! 0.59 had 56, 554, 550 and 56).

#![allow(unused_imports)]
use symplex::errors::SymplexError;
use symplex::prelude::*;

/// `src` to `digits` digits.
fn dec(ctx: &Context, src: &str, digits: u32) -> String {
    ctx.parse(src).unwrap().eval_decimal(digits).unwrap()
}

/// `src` as an `f64`.
fn f64_of(ctx: &Context, src: &str) -> f64 {
    ctx.parse(src).unwrap().eval_f64().unwrap()
}

/// `J_ν` next to a zero.  Before (0.59): the ascending series was summed at
/// `prec + 64 + 1.44·x` bits and its sum, whatever it cancelled to, was
/// certified to two ulps of itself:
///
/// * `besselj(0, x)`, `x` the first zero to 166 bits, at 30 digits printed
///   `6.49874092411216031186725338065e-51` (wrong from the 27th digit), at 50
///   digits `…33832529111062354923020504e-51` (wrong from the 46th);
/// * `besselj(1, x)`, `x` the first zero to 234 bits, at 50 digits printed
///   `4.7811906103442857233102598346096929705668270299756e-71` (wrong from
///   the 28th digit);
/// * the Hankel expansion likewise (`cos ω·P − sin ω·Q` at `prec + 32` bits):
///   `besselj(0, x)`, `x` the 20th zero (62.05) to 128 bits, at 16 digits
///   printed `1.072899793541053e-38` and `eval_f64` gave
///   `1.0728997935410535e-38` (wrong from the 14th digit);
/// * a half-integer order and a larger one: `besselj(5/2, x)`, `x` its first
///   zero (5.763) to 166 bits, at 30 digits printed
///   `1.26664704585075219261649995844e-51` (wrong from the 27th digit);
///   `besselj(20, x)`, `x` its first zero (25.42) to 234 bits, at 50 digits
///   printed `-4.5069081024409898691435315923753003285267380374171e-71`
///   (wrong from the 41st).
///
/// Oracle: `besselj(0, mpf(28117251920540589153062978256639537926577956413888)/2**163)`
/// = `6.4987409241121603118672533832529111062354923019868e-51`;
/// `besselj(1, mpf(13222731340927645485092774184164750541236406026445616742523129851096200)/2**231)`
/// = `4.7811906103442857233102598175343252915624153128567e-71`;
/// `besselj(0, mpf(164953124686495525949401102535850027511)/2**121)`
/// = `1.07289979354102700311050658184e-38`;
/// `besselj(mpf(5)/2, mpf(33693220211804941214875327335188460911422012814671)/2**162)`
/// = `1.26664704585075219261649995295e-51`;
/// `besselj(20, mpf(10963916172461648506748651249639677960129090743953021109455242824267350)/2**228)`
/// = `-4.5069081024409898691435315923753003285270854771452e-71`.
#[test]
fn bessel_j_next_to_a_zero_has_its_digits() {
    let ctx = Context::new();
    let first = "besselj(0, 28117251920540589153062978256639537926577956413888/2^163)";
    assert_eq!(dec(&ctx, first, 30), "6.49874092411216031186725338325e-51");
    assert_eq!(
        dec(&ctx, first, 50),
        "6.4987409241121603118672533832529111062354923019868e-51"
    );
    assert_eq!(
        dec(
            &ctx,
            "besselj(1, 13222731340927645485092774184164750541236406026445616742523129851096200/2^231)",
            50
        ),
        "4.7811906103442857233102598175343252915624153128567e-71"
    );
    let far = "besselj(0, 164953124686495525949401102535850027511/2^121)";
    assert_eq!(dec(&ctx, far, 16), "1.072899793541027e-38");
    assert_eq!(f64_of(&ctx, far), 1.072899793541027e-38);
    assert_eq!(
        dec(
            &ctx,
            "besselj(5/2, 33693220211804941214875327335188460911422012814671/2^162)",
            30
        ),
        "1.26664704585075219261649995295e-51"
    );
    assert_eq!(
        dec(
            &ctx,
            "besselj(20, 10963916172461648506748651249639677960129090743953021109455242824267350/2^228)",
            50
        ),
        "-4.5069081024409898691435315923753003285270854771452e-71"
    );
}

/// `Y_ν` next to a zero.  Before (0.59):
///
/// * the Neumann series (`finite part + (2/π) ln(x/2) J_n + series`, three
///   parts that cancel): `bessely(0, x)`, `x` the first zero (0.8936) to 166
///   bits, at 30 digits printed `-2.31612143674128605044602699707e-51` (wrong
///   from the 21st digit);
/// * the Hankel expansion: `bessely(0, x)`, `x` the 20th zero (60.48) to 128
///   bits, at 16 digits printed `-1.390420564918671e-39` and `eval_f64` gave
///   `-1.390420564918671e-39` (wrong from the 14th digit);
/// * a half-integer order, `Y_(1/2) = −J_(−1/2)` through `(J_ν cos νπ −
///   J_(−ν))/sin νπ`, whose loss test saw no cancellation (`cos νπ = 0`):
///   `bessely(1/2, x)`, `x = 3π/2` to 166 bits, at 30 digits printed
///   `-8.18595986147356494139703130537e-51` (wrong from the 28th digit).
///
/// Oracle: `bessely(0, mpf(83581708755083798884794722849114268496462373033861)/2**166)`
/// = `-2.31612143674128605044115419236e-51`;
/// `bessely(0, mpf(122146675524051569168989165350989348631)/2**120)`
/// = `-1.39042056491869551583648652304e-39`;
/// `bessely(mpf(1)/2, mpf(36731542456385751371078441597946489027633240195345)/2**164)`
/// = `-8.18595986147356494139703130886e-51`.
#[test]
fn bessel_y_next_to_a_zero_has_its_digits() {
    let ctx = Context::new();
    assert_eq!(
        dec(
            &ctx,
            "bessely(0, 83581708755083798884794722849114268496462373033861/2^166)",
            30
        ),
        "-2.31612143674128605044115419236e-51"
    );
    let far = "bessely(0, 122146675524051569168989165350989348631/2^120)";
    assert_eq!(dec(&ctx, far, 16), "-1.390420564918696e-39");
    assert_eq!(f64_of(&ctx, far), -1.3904205649186955e-39);
    assert_eq!(
        dec(
            &ctx,
            "bessely(1/2, 36731542456385751371078441597946489027633240195345/2^164)",
            30
        ),
        "-8.18595986147356494139703130886e-51"
    );
}

/// `Ci` and `Chi` next to a zero (`γ + ln x + Σ`, and `f sin x − g cos x`
/// beyond the switch to the asymptotic expansion), also on the cut, where
/// `Ci(−x) = Ci(x) + iπ` takes its real part from the same routine.  Before
/// (0.59):
///
/// * `Ci(x)`, `x` its first zero (0.6165) to 166 bits, at 30 digits printed
///   `4.37912927047584328995165659427e-51` (wrong from the 25th digit), and
///   `Ci(−x)` `4.37912927047584328995165657376e-51 + 3.14…*i` (from the 27th);
/// * `Ci(x)`, `x` the zero near `36π` (113.9) to 128 bits, at 16 digits
///   printed `-8.880722468532212e-40` and `eval_f64` gave
///   `-8.880722468532211e-40` (wrong from the 11th digit);
/// * `Chi(x)`, `x` its zero (0.5238) to 128 bits, at 16 digits printed
///   `-1.464288802275449e-39` and `eval_f64` gave `-1.4642888022754486e-39`
///   (wrong from the 15th digit).
///
/// Oracle: `ci(mpf(57665521706122252869595292235705244663818123672649)/2**166)`
/// = `4.37912927047584328995165656747e-51` (`ci(-x)`: the same `+ πi`);
/// `ci(mpf(150343892697503437547800758175296216755)/2**120)`
/// = `-8.88072246844355074894443829441e-40`;
/// `chi(mpf(178247584439155322669533330017089363757)/2**128)`
/// = `-1.46428880227544275492677790121e-39`.
#[test]
fn ci_and_chi_next_to_a_zero_have_their_digits() {
    let ctx = Context::new();
    assert_eq!(
        dec(
            &ctx,
            "Ci(57665521706122252869595292235705244663818123672649/2^166)",
            30
        ),
        "4.37912927047584328995165656747e-51"
    );
    assert_eq!(
        dec(
            &ctx,
            "Ci(-57665521706122252869595292235705244663818123672649/2^166)",
            30
        ),
        "4.37912927047584328995165656747e-51 + 3.14159265358979323846264338328*i"
    );
    let far = "Ci(150343892697503437547800758175296216755/2^120)";
    assert_eq!(dec(&ctx, far, 16), "-8.880722468443551e-40");
    assert_eq!(f64_of(&ctx, far), -8.88072246844355e-40);
    let chi = "Chi(178247584439155322669533330017089363757/2^128)";
    assert_eq!(dec(&ctx, chi, 16), "-1.464288802275443e-39");
    assert_eq!(f64_of(&ctx, chi), -1.4642888022754428e-39);
    assert_eq!(
        dec(
            &ctx,
            "Chi(-178247584439155322669533330017089363757/2^128)",
            16
        ),
        "-1.464288802275443e-39 + 3.141592653589793*i"
    );
}

/// The Airy functions and their derivatives next to a zero (`c₁ f − c₂ g` of
/// two series that each cancel for `x < 0`, and `sin θ·P − cos θ·Q` beyond
/// the switch to the asymptotic expansion).  Before (0.59):
///
/// * `airyai(x)`, `x` the first zero (−2.338) to 166 bits, at 30 digits
///   printed `-1.98879268556479007370830765028e-50` (wrong from the 22nd
///   digit);
/// * `airyai(x)`, `x` the 30th zero (−26.99) to 128 bits, at 16 digits
///   printed `9.915398416568843e-38` and `eval_f64` gave
///   `9.915398416568842e-38` (wrong from the 14th digit);
/// * `airybi(x)`, `x` the first zero (−1.174) to 166 bits, at 30 digits
///   printed `-7.63885344343547489585186184087e-51` (from the 21st);
///   `airybi(x)`, `x` the 80th zero (−51.75) to 128 bits, at 16 digits
///   `1.269913975064016e-37` (from the 14th);
/// * `airyaiprime(x)`, `x` its first zero (−1.019) to 166 bits, at 30 digits
///   `-3.22105014611449982632809667149e-51` (from the 21st);
///   `airybiprime(x)`, `x` its first zero (−2.294) to 166 bits,
///   `-1.1946523452108772976634238996e-50` (from the 23rd).
///
/// Oracle: `airyai(-mpf(27337182469139736116683211541402392898793557384887)/2**163)`
/// = `-1.98879268556479007370803243570e-50`;
/// `airyai(-mpf(143487424528711585299495522731728443827)/2**122)`
/// = `9.91539841656918761733240796184e-38`;
/// `airybi(-mpf(27446140747941138682266650978856175028455263694147)/2**164)`
/// = `-7.63885344343547489584955884974e-51`;
/// `airybi(-mpf(137865860165465727328921580036660048953)/2**121)`
/// = `1.26991397506395442227925619128e-37`;
/// `airyai(-mpf(23823481538623922413960720303654505055735727889496)/2**164, derivative=1)`
/// = `-3.22105014611449982632461014662e-51`;
/// `airybi(-mpf(26826618823180306807022623087426057302407867769205)/2**163, derivative=1)`
/// = `-1.19465234521087729766330751839e-50`.
#[test]
fn airy_functions_next_to_a_zero_have_their_digits() {
    let ctx = Context::new();
    assert_eq!(
        dec(
            &ctx,
            "airyai(-27337182469139736116683211541402392898793557384887/2^163)",
            30
        ),
        "-1.9887926855647900737080324357e-50"
    );
    let far = "airyai(-143487424528711585299495522731728443827/2^122)";
    assert_eq!(dec(&ctx, far, 16), "9.915398416569188e-38");
    assert_eq!(f64_of(&ctx, far), 9.915398416569188e-38);
    assert_eq!(
        dec(
            &ctx,
            "airybi(-27446140747941138682266650978856175028455263694147/2^164)",
            30
        ),
        "-7.63885344343547489584955884974e-51"
    );
    let far = "airybi(-137865860165465727328921580036660048953/2^121)";
    assert_eq!(dec(&ctx, far, 16), "1.269913975063954e-37");
    assert_eq!(f64_of(&ctx, far), 1.2699139750639543e-37);
    assert_eq!(
        dec(
            &ctx,
            "airyaiprime(-23823481538623922413960720303654505055735727889496/2^164)",
            30
        ),
        "-3.22105014611449982632461014662e-51"
    );
    assert_eq!(
        dec(
            &ctx,
            "airybiprime(-26826618823180306807022623087426057302407867769205/2^163)",
            30
        ),
        "-1.19465234521087729766330751839e-50"
    );
}

/// `li` next to its zero `μ = 1.4513…` and `I_ν` of a negative order that
/// is not an integer next to its zero.  Before (0.59):
///
/// * `li(x) = Ei(ln x)` with `ln x` rounded at `prec + 32` bits, an absolute
///   error of `x·2^(−prec−32)` in a value that was certified to two ulps of
///   itself: `li(x)`, `x = μ` to 166 bits, at 30 digits printed
///   `-3.55124931147984672946490625093e-50` (wrong from the 28th digit);
/// * the ascending series of `I_ν`, documented as "all terms positive": for
///   `ν = −3/2` the first term is negative (`1/Γ(−1/2)`) and cancels the
///   rest at 1.1997: `besseli(-3/2, x)`, `x` that zero to 166 bits, at 30
///   digits printed `2.49916944919829326354115559942e-50` (wrong in the 30th
///   digit; at 50 digits, for the zero to 234 bits, from the 27th).
///
/// Oracle: `li(mpf(33938856210460179588780149376788169124828439945919)/2**164)`
/// = `-3.55124931147984672946490625023e-50`;
/// `besseli(-mpf(3)/2, mpf(28053316752121428210351574332014715515080983425197)/2**164)`
/// = `2.49916944919829326354115559945e-50`.
#[test]
fn li_and_bessel_i_next_to_a_zero_have_their_digits() {
    let ctx = Context::new();
    assert_eq!(
        dec(
            &ctx,
            "li(33938856210460179588780149376788169124828439945919/2^164)",
            30
        ),
        "-3.55124931147984672946490625023e-50"
    );
    assert_eq!(
        dec(
            &ctx,
            "besseli(-3/2, 28053316752121428210351574332014715515080983425197/2^164)",
            30
        ),
        "2.49916944919829326354115559945e-50"
    );
}

/// Away from the zeros the routes charge an error far below the rounding of
/// their value, and nothing changes: these are the digits 0.59 printed
/// (the same strings before and after, `target/scratch/n1/t_base.out` and
/// `t_new.out` of the wave).
///
/// Oracle (mpmath, 30 digits): `besselj(0, 1)` =
/// `0.765197686557966551449717526103`; `besselj(mpf(5)/2, 7)` =
/// `-0.283436651201699198215614814451`; `besselj(3, 100)` =
/// `0.0762842017203319434092902108933`; `bessely(0, 1)` =
/// `0.0882569642156769579829267660235`; `bessely(2, 90)` =
/// `-0.0803584144906059458130830225590`; `bessely(mpf(1)/3, 2)` =
/// `0.343199966260344342261499177313`; `ci(1)` =
/// `0.337403922900968134662646203889`; `ci(200)` =
/// `-0.00437844609302782567916569771749`; `chi(1)` =
/// `0.837866940980208240894678579436`; `airyai(1)` =
/// `0.135292416312881415524147423515`; `airyai(-30)` =
/// `-0.0879681884568421628326238583239`; `airybi(-1)` =
/// `0.103997389496944611888689990979`; `airybi(-25, derivative=1)` =
/// `0.815719715754605857877618928648`; `li(2)` =
/// `1.04516378011749278484458888919`; `besseli(-mpf(3)/2, 2)` =
/// `0.984941053000236439697123464155`; `besseli(2, 3)` =
/// `2.24521244092995115462547838563`.
#[test]
fn values_away_from_the_zeros_are_unchanged() {
    let ctx = Context::new();
    for (src, value) in [
        ("besselj(0, 1)", "0.765197686557966551449717526103"),
        ("besselj(5/2, 7)", "-0.283436651201699198215614814451"),
        ("besselj(3, 100)", "0.0762842017203319434092902108933"),
        ("bessely(0, 1)", "0.0882569642156769579829267660235"),
        ("bessely(2, 90)", "-0.080358414490605945813083022559"),
        ("bessely(1/3, 2)", "0.343199966260344342261499177313"),
        ("Ci(1)", "0.337403922900968134662646203889"),
        ("Ci(200)", "-0.00437844609302782567916569771749"),
        ("Chi(1)", "0.837866940980208240894678579436"),
        ("airyai(1)", "0.135292416312881415524147423515"),
        ("airyai(-30)", "-0.0879681884568421628326238583239"),
        ("airybi(-1)", "0.103997389496944611888689990979"),
        ("airybiprime(-25)", "0.815719715754605857877618928648"),
        ("li(2)", "1.04516378011749278484458888919"),
        ("besseli(-3/2, 2)", "0.984941053000236439697123464155"),
        ("besseli(2, 3)", "2.24521244092995115462547838563"),
    ] {
        assert_eq!(dec(&ctx, src, 30), value, "{src}");
    }
    assert_eq!(f64_of(&ctx, "besselj(0, 1)"), 0.7651976865579666);
    assert_eq!(f64_of(&ctx, "Ci(200)"), -0.004378446093027825);
    assert_eq!(f64_of(&ctx, "airyai(-30)"), -0.08796818845684216);
}

/// An argument that is not exactly representable carries its own rounding,
/// which the bound always had: the value next to a zero was right before,
/// and is the same now (0.59 and this tree print the same strings).
/// `besselj(0, x)` for `x = 24048255576957727686216318793264546431243/10^40`,
/// within `10⁻⁴⁰` of the first zero.
///
/// Oracle: `besselj(0, mpf(24048255576957727686216318793264546431243)/mpf(10)**40)`
/// = `-2.86002789947008260248906919344e-41` (30 digits),
/// `-2.860027899470083e-41` (16).
#[test]
fn a_rounded_argument_next_to_a_zero_keeps_its_value() {
    let ctx = Context::new();
    let src = "besselj(0, 24048255576957727686216318793264546431243/10^40)";
    assert_eq!(dec(&ctx, src, 16), "-2.860027899470083e-41");
    assert_eq!(dec(&ctx, src, 30), "-2.86002789947008260248906919344e-41");
}

/// The orthogonal polynomials of an integer degree next to a zero, where
/// `eval` does not fold them exactly (a degree of 90 and more at an
/// argument of 166 and 234 bits): the three-term recurrences of
/// `evalf/orthopoly.rs`, run at `prec + 32 + log₂ n` bits, end in a
/// difference of two terms of the size of the neighbouring members.  Before
/// (0.59), at 50 digits (wrong from the 24th to the 27th digit):
///
/// * `legendre(100, x)`, `x` a zero (0.3147) to 234 bits:
///   `4.6124217044228027324163396646113422908665058928319e-71`;
/// * `chebyshev_t(120, x)`, `x = cos(59π/240)` to 234 bits:
///   `-2.7123315692771218957104426591727479148284369395346e-69`;
/// * `hermite(90, x)`, `x` its smallest positive zero (0.1162) to 234 bits:
///   `-208638522505.33203577706409703740320082943071611226` (a value of
///   `2·10¹¹` in an oscillation of amplitude `10⁸¹`);
/// * `laguerre(100, x)`, `x` its smallest zero (0.01439) to 234 bits:
///   `4.863554611178517821943062812363663001703857953095e-72`;
/// * `gegenbauer(100, 1/4, x)`, `x` a zero (0.3153) to 234 bits:
///   `-6.4663166442414725515137863425017701810269930275369e-74`;
/// * `assoc_laguerre(100, 1/2, x)`, `x` its smallest zero (0.0407) to 234
///   bits: `1.1526516611102033441648654763007672502751994168126e-70`;
/// * `assoc_legendre(90, 40, x)`, `x` a zero (0.3153) to 234 bits:
///   `829361.28747524770054994814119653234740781044820324`;
/// * and at 30 digits `chebyshev_u(101, x)`, `x = cos(20π/102)` to 166 bits:
///   `-1.21389392065794085783503731208e-48` (wrong in the 28th digit).
///
/// Oracle (the values asserted, to 50 digits):
/// `legendre(100, mpf(17798794606925014998078840683233776870609402119942521049963006112341786)/2**235)`;
/// `chebyt(120, mpf(19774937284920354262927492770794805142833630289462856964284625478230526)/2**234)`;
/// `hermite(90, mpf(25786458436813163921038000390349258236370664416257912796956603739153838)/2**237)`;
/// `laguerre(100, 0, mpf(25418121592327615825405765376226944881149488070466736619404296425155539)/2**240)`;
/// `gegenbauer(100, mpf(1)/4, mpf(17841623527871184512936513183296942161669391912501434046792687347541492)/2**235)`;
/// `laguerre(100, mpf(1)/2, mpf(21635708595801713776091472835780412424382145561002578009501139747343475)/2**239)`;
/// `legenp(90, 40, mpf(17915949684832078620889018402846731297869203821505613902351613656599811)/2**235)`;
/// `chebyu(101, mpf(76343879922754909982184408933397300111871905606809)/2**166)`
/// = `-1.21389392065794085783503731220e-48`.
#[test]
fn orthogonal_polynomials_next_to_a_zero_have_their_digits() {
    let ctx = Context::new();
    for (src, value) in [
        (
            "legendre(100, 17798794606925014998078840683233776870609402119942521049963006112341786/2^235)",
            "4.6124217044228027324163396714093573344335052870778e-71",
        ),
        (
            "chebyshev_t(120, 19774937284920354262927492770794805142833630289462856964284625478230526/2^234)",
            "-2.7123315692771218957104426563909870289036957793758e-69",
        ),
        (
            "hermite(90, 25786458436813163921038000390349258236370664416257912796956603739153838/2^237)",
            "-208638522505.33203577706409576576801604354179650236",
        ),
        (
            "laguerre(100, 25418121592327615825405765376226944881149488070466736619404296425155539/2^240)",
            "4.8635546111785178219430530045566754469256644371143e-72",
        ),
        (
            "gegenbauer(100, 1/4, 17841623527871184512936513183296942161669391912501434046792687347541492/2^235)",
            "-6.4663166442414725515137846241827444544676680922507e-74",
        ),
        (
            "assoc_laguerre(100, 1/2, 21635708595801713776091472835780412424382145561002578009501139747343475/2^239)",
            "1.1526516611102033441648604006208297694006059428573e-70",
        ),
        (
            "assoc_legendre(90, 40, 17915949684832078620889018402846731297869203821505613902351613656599811/2^235)",
            "829361.28747524770054994817795347816660022995196928",
        ),
    ] {
        assert_eq!(dec(&ctx, src, 50), value, "{src}");
    }
    assert_eq!(
        dec(
            &ctx,
            "chebyshev_u(101, 76343879922754909982184408933397300111871905606809/2^166)",
            30
        ),
        "-1.2138939206579408578350373122e-48"
    );
}

/// A zero that the recurrence gives exactly keeps a bound of the value's
/// size: an odd member of a symmetric family next to `x = 0` (every step is
/// a difference of terms with the factor `x`), `U_101` next to its rational
/// zero `1/2`.  These were right before and are the same now, at the same
/// precision (0.59 and this tree print the same strings).
///
/// Oracle: `legendre(101, mpf(1)/2**126)` =
/// `9.44922659242314585758141159533e-38`; `hermite(91, mpf(1)/2**164)` =
/// `-96666507477287110463219903841506657.024725540703224`;
/// `chebyu(101, mpf(3450873173395281893717377931138512726225554486085193277581262111899649)/2**232)`
/// = `-1.9705157675526926111699843576971834914765187887151e-68`.
#[test]
fn a_polynomial_next_to_a_rational_zero_keeps_its_value() {
    let ctx = Context::new();
    assert_eq!(
        dec(&ctx, "legendre(101, 1/2^126)", 16),
        "9.449226592423146e-38"
    );
    assert_eq!(
        dec(&ctx, "legendre(101, 1/2^126)", 30),
        "9.44922659242314585758141159533e-38"
    );
    assert_eq!(
        dec(&ctx, "hermite(91, 1/2^164)", 50),
        "-96666507477287110463219903841506657.024725540703224"
    );
    assert_eq!(
        dec(
            &ctx,
            "chebyshev_u(101, 3450873173395281893717377931138512726225554486085193277581262111899649/2^232)",
            50
        ),
        "-1.9705157675526926111699843576971834914765187887151e-68"
    );
    assert_eq!(dec(&ctx, "legendre(101, 0)", 30), "0");
}

/// An argument too close to a zero for the budget is refused, as before: a
/// refusal is correct, a digit that is not certified is not.  `besselj(0, x)`
/// for `x` the first zero to 1140 bits (`gen_deep.py` of the wave): the
/// argument is exact only from 1140 bits on, where the search for a value
/// that may be 0 ends — `PrecisionExhausted` in 0.59 and now.  (A larger
/// budget may print the value instead: then its digits.)
///
/// Oracle (mpmath at `mp.dps = 700` and `900`): `besselj(0, mpf(N)/2**1137)`
/// = `-1.385634667410608384e-343` for the `N` below.
#[test]
fn an_argument_beyond_the_budget_is_refused_not_guessed() {
    let ctx = Context::new();
    let src = "besselj(0, 4489403582108593964283000847693736692240823654903711765585654449706254705217424396227558008181486358693282474570416289485043878315310926911493270804374644116113883700276414154366119006039064270024664399813094953526635938215166993310208523461410182099644636472042300959576411647116201755326076982952021081614094486553878943815145161196526043879/2^1137)";
    match ctx.parse(src).unwrap().eval_decimal(16) {
        Err(SymplexError::PrecisionExhausted { .. }) => {}
        other => assert_eq!(other.unwrap(), "-1.385634667410608e-343"),
    }
}
