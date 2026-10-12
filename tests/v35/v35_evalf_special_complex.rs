//! `evalf` of the exponential integrals, the incomplete gamma functions,
//! the Bessel, Airy and Fresnel functions and `binomial` at complex
//! arguments (and at negative real ones where the value is complex), the
//! complex error function at large arguments, and `erfinv`/`erfcinv` next
//! to the ends of their domains (0.45).  Before, every one of them was
//! refused.  Reference values: mpmath 1.3 at 50 and 70 digits (`python3 -c
//! "from mpmath import *; mp.dps = 50; print(…)"`, the two agreeing), each
//! cited at its test, or a closed form.

use symplex::prelude::*;

/// `eval_decimal(digits)` of `src`.
fn value(src: &str, digits: u32) -> String {
    let ctx = Context::new();
    ctx.parse(src)
        .unwrap()
        .eval_decimal(digits)
        .unwrap_or_else(|e| panic!("{src} at {digits}: {e:?}"))
}

/// The real and imaginary parts of a printed value (`a + b*i`, `a - b*i`,
/// `b*i` or `a`), `"0"` for an absent one.
fn split(got: &str) -> (String, String) {
    if let Some((a, b)) = got.rsplit_once(" + ") {
        return (a.to_string(), b.trim_end_matches("*i").to_string());
    }
    if let Some((a, b)) = got.rsplit_once(" - ") {
        return (a.to_string(), format!("-{}", b.trim_end_matches("*i")));
    }
    if let Some(b) = got.strip_suffix("*i") {
        return ("0".to_string(), b.to_string());
    }
    (got.to_string(), "0".to_string())
}

/// A printed part `m` or `me±k` starts with the mantissa prefix of `want`
/// (same form) and has its exponent.
fn part_matches(got: &str, want: &str) -> bool {
    let (gm, ge) = got.split_once('e').unwrap_or((got, ""));
    let (wm, we) = want.split_once('e').unwrap_or((want, ""));
    gm.starts_with(wm) && ge == we
}

/// `eval_decimal(digits)` of `src` has the parts `re` and `im` (prefixes of
/// the reference, two digits short of `digits`).
fn assert_c(src: &str, digits: u32, re: &str, im: &str) {
    let got = value(src, digits);
    let (a, b) = split(&got);
    assert!(
        part_matches(&a, re) && part_matches(&b, im),
        "{src} at {digits}: {got} (want {re}…, {im}…)"
    );
}

/// `Ei(1 + i)` was refused ("Ei of complex argument not yet supported in
/// evalf").  mpmath `ei(mpc(1,1))` =
/// `1.76462598556385406842673816135123796600830441 +
/// 2.38776985151052241926279208910379606440733385i`.
#[test]
fn ei_of_a_complex_argument() {
    assert_c(
        "Ei(1 + I)",
        40,
        "1.764625985563854068426738161351237966",
        "2.387769851510522419262792089103796064",
    );
}

/// `expint(2, 3 − 4i)` and `E1(1 + i)` were refused ("expint of complex
/// argument not yet supported in evalf").  mpmath `expint(2, mpc(3,-4))` =
/// `9.9719916962280385400153163443504514084e-6 −
/// 0.0078645366073551881990464736610514516341i`; `e1(mpc(1,1))` =
/// `0.00028162445198141832550992803865893083224 −
/// 0.17932453503935894014528414940346100536i`.
#[test]
fn expint_of_a_complex_argument() {
    assert_c(
        "expint(2, 3 - 4*I)",
        40,
        "9.9719916962280385400153163443504514e-6",
        "-0.0078645366073551881990464736610514516",
    );
    assert_c(
        "E1(1 + I)",
        40,
        "0.00028162445198141832550992803865893083",
        "-0.17932453503935894014528414940346100",
    );
}

/// On the cut the value is taken from above, as mpmath and SymPy:
/// `expint(1, −1)` was refused ("expint of negative argument is complex;
/// not yet supported in evalf").  mpmath `expint(1, mpc(-1,0))` =
/// `−1.8951178163559367554665209343316342690 − πi` (`−Ei(1) − iπ`);
/// `expint(2, mpc(-1,0))` = `0.82316401210310847989376653702102822874 − πi`.
#[test]
fn expint_on_the_cut_from_above() {
    assert_c(
        "expint(1, -1)",
        40,
        "-1.8951178163559367554665209343316342",
        "-3.1415926535897932384626433832795028",
    );
    assert_c(
        "expint(2, -1)",
        40,
        "0.82316401210310847989376653702102822",
        "-3.1415926535897932384626433832795028",
    );
}

/// `li(2 + i)` and `li(−2)` were refused ("li of complex or negative
/// argument not yet supported in evalf").  mpmath `li(mpc(2,1))` =
/// `1.41125904201780100568439320706248438589 +
/// 1.22470693841030271349717531822422764236i`; `li(mpc(-2,0))` =
/// `0.0355322759135604677869013496215783195040 +
/// 3.73515048255125463977638213891342169791i`.
#[test]
fn li_of_complex_and_negative_arguments() {
    assert_c(
        "li(2 + I)",
        40,
        "1.4112590420178010056843932070624843",
        "1.2247069384103027134971753182242276",
    );
    assert_c(
        "li(-2)",
        40,
        "0.035532275913560467786901349621578319",
        "3.7351504825512546397763821389134216",
    );
}

/// `Si`, `Ci`, `Shi`, `Chi` of complex arguments were refused ("Si of
/// complex argument not yet supported in evalf").  mpmath `si(mpc(1,3))` =
/// `3.04276782129088392563601637591701434114 +
/// 3.91346236603293744067883540780746320599i`; `ci(mpc(-2,1))` =
/// `0.584475996878247678745015223696810601257 +
/// 3.43908783122792725941396157994924956020i`; `shi(mpc(1,1))` =
/// `0.882453805007917743376124044694948448420 +
/// 1.10422265823558173955875396985016752952i`; `chi(mpc(2,-1))` =
/// `2.03029639329172163778518545049148022039 −
/// 1.72270348197075350361042140107047628150i`.
#[test]
fn sine_and_cosine_integrals_of_complex_arguments() {
    assert_c(
        "Si(1 + 3*I)",
        40,
        "3.0427678212908839256360163759170143",
        "3.9134623660329374406788354078074632",
    );
    assert_c(
        "Ci(-2 + I)",
        40,
        "0.58447599687824767874501522369681060",
        "3.4390878312279272594139615799492495",
    );
    assert_c(
        "Shi(1 + I)",
        40,
        "0.88245380500791774337612404469494844",
        "1.1042226582355817395587539698501675",
    );
    assert_c(
        "Chi(2 - I)",
        40,
        "2.0302963932917216377851854504914802",
        "-1.7227034819707535036104214010704762",
    );
}

/// Large arguments go through `E1(±z)` and `E1(±iz)` with the remainder
/// bound of the asymptotic expansion.  mpmath `ci(mpc(30,40))` =
/// `−1149478706763855.11132733651557538938987 +
/// 2097913261506351.89672415317959923960572i`; `si(mpc(0,200))` =
/// `1.81561761657967842619835502192321252323e84·i`; `chi(mpc(-300,-1))` =
/// `1.76415418614251317654620696790179277551e127 +
/// 2.72740045201863315832187414111825281782e127i`.
#[test]
fn sine_and_cosine_integrals_of_large_arguments() {
    assert_c(
        "Ci(30 + 40*I)",
        40,
        "-1149478706763855.1113273365155753893",
        "2097913261506351.8967241531795992396",
    );
    assert_c(
        "Si(200*I)",
        40,
        "0",
        "1.8156176165796784261983550219232125e84",
    );
    assert_c(
        "Chi(-300 - I)",
        40,
        "1.7641541861425131765462069679017927e127",
        "2.7274004520186331583218741411825281e127",
    );
}

/// Bessel functions of complex arguments and orders were refused ("Bessel
/// of complex argument not yet supported in evalf").  mpmath
/// `besselj(mpf(1)/3, mpc(2,3))` = `1.69009520279896263993160291066213662496
/// − 3.94540478670958386270748871549307873146i`; `besselk(2, mpc(-1,1))` =
/// `−1.13217605442457556045136235090230111505 +
/// 0.972192305970505525615774401358009191840i`; `bessely(0, mpc(1,1))` =
/// `0.445474488936032514031953505326782769445 +
/// 0.710158582003734521179561702630437186460i`; `besseli(mpc(1,1),
/// mpc(2,-1))` = `0.542523028791547170762926839473848237896 −
/// 2.74036134407027545628573574181561951666i`.
#[test]
fn bessel_functions_of_complex_arguments_and_orders() {
    assert_c(
        "besselj(1/3, 2 + 3*I)",
        40,
        "1.6900952027989626399316029106621366",
        "-3.9454047867095838627074887154930787",
    );
    assert_c(
        "besselk(2, -1 + I)",
        40,
        "-1.1321760544245755604513623509023011",
        "0.97219230597050552561577440135800919",
    );
    assert_c(
        "bessely(0, 1 + I)",
        40,
        "0.44547448893603251403195350532678276",
        "0.71015858200373452117956170263043718",
    );
    assert_c(
        "besseli(1 + I, 2 - I)",
        40,
        "0.54252302879154717076292683947384823",
        "-2.7403613440702754562857357418156195",
    );
}

/// A negative real argument, from above: `besselj(1/2, −2)` was refused
/// ("BesselJ of negative argument with non-integer order is complex"),
/// `besselk(0, −2)` too ("BesselK undefined at x ≤ 0").  mpmath
/// `besselj(mpf(1)/2, mpc(-2,0))` = `0.513016136561827751665691848627284422355i`;
/// `besselk(0, mpc(-2,0))` = `0.113893872749533435652719574932481832998 −
/// 7.16152843905025666211114074470707393946i`.
#[test]
fn bessel_functions_on_the_cut() {
    assert_c(
        "besselj(1/2, -2)",
        40,
        "0",
        "0.51301613656182775166569184862728442",
    );
    assert_c(
        "besselk(0, -2)",
        40,
        "0.11389387274953343565271957493248183",
        "-7.1615284390502566621111407447070739",
    );
}

/// Large arguments by Hankel's expansion of `K_ν` with Olver's remainder
/// bound.  mpmath `besselj(0, mpc(300,10))` =
/// `−360.738106555598582740434710161028273661 +
/// 356.538996516423064320812667542081692479i`; `besselk(mpf(1)/3,
/// mpc(0,1000))` = `−0.00740990987638207574784341348551709439974 −
/// 0.0389344250418044407396559793979330920618i`; `bessely(3, mpc(-200,1))` =
/// `−0.0112108735945051598404215750808217418627 −
/// 0.104365927449519448921438867754480461013i`.
#[test]
fn bessel_functions_of_large_arguments() {
    assert_c(
        "besselj(0, 300 + 10*I)",
        40,
        "-360.73810655559858274043471016102827",
        "356.53899651642306432081266754208169",
    );
    assert_c(
        "besselk(1/3, 1000*I)",
        40,
        "-0.0074099098763820757478434134855170943",
        "-0.038934425041804440739655979397933092",
    );
    assert_c(
        "bessely(3, -200 + I)",
        40,
        "-0.011210873594505159840421575080821741",
        "-0.10436592744951944892143886775448046",
    );
}

/// Airy functions of complex arguments were refused ("airyai of complex
/// argument not yet supported in evalf").  mpmath `airyai(mpc(1,1))` =
/// `0.0604583083718381491965329781166461468682 −
/// 0.151889565877181402354947912592231564711i`; `airybi(mpc(-5,2))` =
/// `−0.497328950063074436355642520494785543189 +
/// 16.7491663513661334894003417196110735536i`; `airyai(mpc(2,-1),1)` =
/// `−0.0151102792832269579304082359551461996336 −
/// 0.0624589547136001381551945916693061338642i`; `airybi(mpc(-1,3),1)` =
/// `−7.98798044435166100704465932135140518299 −
/// 10.3625904067448540428036185933661565783i`.
#[test]
fn airy_functions_of_complex_arguments() {
    assert_c(
        "airyai(1 + I)",
        40,
        "0.060458308371838149196532978116646146",
        "-0.15188956587718140235494791259223156",
    );
    assert_c(
        "airybi(-5 + 2*I)",
        40,
        "-0.49732895006307443635564252049478554",
        "16.749166351366133489400341719611073",
    );
    assert_c(
        "airyaiprime(2 - I)",
        40,
        "-0.015110279283226957930408235955146199",
        "-0.062458954713600138155194591669306133",
    );
    assert_c(
        "airybiprime(-1 + 3*I)",
        40,
        "-7.9879804443516610070446593213514051",
        "-10.362590406744854042803618593366156",
    );
}

/// Large arguments by Hankel's expansion of `K_{1/3}` and the connection
/// formulas.  mpmath `airyai(mpc(100,50))` =
/// `−1.09859616888553881937769443963742992971e-264 −
/// 9.17785686792102594858242359602872347424e-265i`; `airybi(mpc(-200,1))` =
/// `14446.4678547397671235485996660313198190 +
/// 102982.278517136210912563280319716754357i`.
#[test]
fn airy_functions_of_large_arguments() {
    assert_c(
        "airyai(100 + 50*I)",
        40,
        "-1.0985961688855388193776944396374299e-264",
        "-9.1778568679210259485824235960287234e-265",
    );
    assert_c(
        "airybi(-200 + I)",
        40,
        "14446.467854739767123548599666031319",
        "102982.27851713621091256328031971675",
    );
}

/// `binomial` of complex arguments was refused ("Binomial of complex
/// arguments not yet supported in evalf").  mpmath
/// `binomial(mpc(0.5,1), mpc(3,-1))` = `5.15837038347751668745146658647601223917
/// − 1.25839762617940431993286713482152065721i`.  An integer lower index is
/// the falling factorial: `binomial(1 + 2i, 3) = (1+2i)(2i)(−1+2i)/6 =
/// −5i/3` exactly (mpmath prints a spurious real part `1.9e-56`), and a
/// negative one gives 0 (mpmath `binomial(mpc(1,2), -2)` = 0).
#[test]
fn binomial_of_complex_arguments() {
    assert_c(
        "binomial(1/2 + I, 3 - I)",
        40,
        "5.1583703834775166874514665864760122",
        "-1.2583976261794043199328671348215206",
    );
    assert_eq!(
        value("binomial(1 + 2*I, 3)", 30),
        "-1.66666666666666666666666666667*i"
    );
    assert_eq!(value("binomial(1 + 2*I, -2)", 30), "0");
}

/// The incomplete gamma functions of complex arguments were refused
/// ("uppergamma of complex argument not yet supported in evalf"), and of
/// negative real ones ("uppergamma of negative argument not yet supported
/// in evalf").  mpmath `gammainc(mpc(1,1), 2)` =
/// `0.0646320441729587523784177704583199294634 +
/// 0.112840294049425907076119254568373073506i`; `gammainc(-mpf(5)/2, 0,
/// mpc(-3,3))` = `−0.863328124965292194648179919041785536414 −
/// 0.120887207066166820830269515465091136325i`; `gammainc(mpf(1)/3,
/// mpc(-pi,0))` = `−5.20726142636788216533943305188588515197 −
/// 13.6592990112306932714557127311763894267i`; `gammainc(mpc(3,2),
/// mpc(0,500))` (the asymptotic expansion) =
/// `8777.77328208313994733134206090204664585 −
/// 6372.22317984181161529858025002420071705i`.
#[test]
fn incomplete_gamma_of_complex_arguments() {
    assert_c(
        "uppergamma(1 + I, 2)",
        40,
        "0.064632044172958752378417770458319929",
        "0.11284029404942590707611925456837307",
    );
    assert_c(
        "lowergamma(-5/2, -3 + 3*I)",
        40,
        "-0.86332812496529219464817991904178553",
        "-0.12088720706616682083026951546509113",
    );
    assert_c(
        "uppergamma(1/3, -pi)",
        40,
        "-5.2072614263678821653394330518858851",
        "-13.659299011230693271455712731176389",
    );
    assert_c(
        "uppergamma(3 + 2*I, 500*I)",
        40,
        "8777.7732820831399473313420609020466",
        "-6372.2231798418116152985802500242007",
    );
}

/// The Fresnel integrals of complex arguments were refused ("fresnels of
/// complex argument not yet supported in evalf").  mpmath
/// `fresnels(mpc(1,1))` = `−2.06188821919484046808071653668570860082 +
/// 2.06188821919484046808071653668570860082i`; `fresnelc(59*pi*1j)` (by the
/// asymptotic expansion of `erfc`; mpmath at 30 digits is off in its last
/// two, at 50 and 70 they agree) = `0.500249751277831407374129365856204533645i`.
#[test]
fn fresnel_integrals_of_complex_arguments() {
    assert_c(
        "fresnels(1 + I)",
        40,
        "-2.0618882191948404680807165366857086",
        "2.0618882191948404680807165366857086",
    );
    assert_c(
        "fresnelc(59*pi*I)",
        40,
        "0",
        "0.50024975127783140737412936585620453",
    );
}

/// `erf(30 + 31i)` exhausted the precision (`PrecisionExhausted`: its power
/// series cancels about 2,700 bits); the asymptotic expansion of `erfc`
/// with a proven remainder takes over.  mpmath `erf(mpc(30,31))` =
/// `−2265955196266120550219768.10402525780675261740 +
/// 3368755649936101859963903.05694870605967628496i`.
#[test]
fn erf_of_a_large_complex_argument() {
    assert_c(
        "erf(30 + 31*I)",
        40,
        "-2265955196266120550219768.104025257806",
        "3368755649936101859963903.056948706059",
    );
}

/// `erfinv(1 − 10⁻¹⁰⁰⁰)` was refused at every precision the configuration
/// allows (`PrecisionExhausted`: the argument rounds onto 1 below 3,322
/// bits); it is `erfcinv(10⁻¹⁰⁰⁰)` from the exact complement.  Oracle:
/// mpmath `findroot(lambda x: log(erfc(x)) - log(u), sqrt(-log(u)))` at 60
/// digits for `u = 10⁻¹⁰⁰⁰` = `47.938946404264510689915788168577822684327`,
/// `u = 3·10⁻⁴⁰⁰` = `30.264707706463698322872175741250980106675`, `u =
/// 10⁻⁵⁰⁰` = `33.870299787058693395485848917682853793788`.
#[test]
fn inverse_error_functions_next_to_the_ends() {
    assert_c(
        "erfinv(1 - 10^(-1000))",
        40,
        "47.938946404264510689915788168577822",
        "0",
    );
    assert_c(
        "erfinv(-1 + 3*10^(-400))",
        40,
        "-30.264707706463698322872175741250980",
        "0",
    );
    assert_c(
        "erfcinv(2 - 10^(-500))",
        40,
        "-33.870299787058693395485848917682853",
        "0",
    );
}

/// A sum of conjugate values is exactly real (`conjugate.rs`), so a square
/// root of it that is negative takes its principal branch rather than
/// being refused for an undecidable side: `Ei(1 + i) + Ei(1 − i) =
/// 3.52925197112770813685347632270247593202` (twice mpmath's real part of
/// `ei(mpc(1,1))`), and `sqrt(Ei(1 + i) + Ei(1 − i) − 4) =
/// 0.686110799268086045731116345088876719217i` (mpmath
/// `sqrt(2*re(ei(mpc(1,1))) - 4)`).
#[test]
fn conjugate_pairs_of_the_new_functions_are_real() {
    assert_eq!(
        value("Ei(1 + I) + Ei(1 - I)", 30),
        "3.5292519711277081368534763227"
    );
    let (re, im) = split(&value("sqrt(Ei(1 + I) + Ei(1 - I) - 4)", 30));
    assert_eq!(re, "0");
    assert!(im.starts_with("0.6861107992680860457311163"), "{im}");
    let (_, im) = split(&value("besselj(1/3, 2 + 3*I) + besselj(1/3, 2 - 3*I)", 30));
    assert_eq!(im, "0");
}

/// An argument whose imaginary part is 0 only to the precision reached
/// leaves the side of the cut undecided: refused, not guessed
/// (`cos(π/7) + cos(3π/7) + cos(5π/7) = 1/2`).  An entire function of it is
/// evaluated: `Si(−2 + 0i) = −Si(2)` (mpmath `si(-2)` =
/// `−1.6054129768026948485767201481985889`).
#[test]
fn an_undecided_side_of_the_cut_is_refused() {
    let ctx = Context::new();
    let e = ctx
        .parse("Ei(-5 + (cos(pi/7) + cos(3*pi/7) + cos(5*pi/7) - 1/2)*I)")
        .unwrap();
    assert!(e.eval_decimal(30).is_err());
    assert_c(
        "Si(-2 + (cos(pi/7) + cos(3*pi/7) + cos(5*pi/7) - 1/2)*I)",
        30,
        "-1.6054129768026948485767201",
        "0",
    );
}

/// `ζ(s)` far up the critical strip by the Riemann–Siegel formula: `ζ(1/2 +
/// 10⁶i)` was refused ("Hurwitz zeta: an order or argument this large is
/// not supported in evalf"), as was every `abs(t)` beyond about `5·10⁵`.
/// mpmath `zeta(mpc(0.5,10**6))` =
/// `0.0760890697382271000055645583799273223107986047 +
/// 2.80510210101929895539383671656494023646250615277i`; `zeta(mpc(2,10**6))`
/// = `0.964812264725801206930167217061876635282630224 +
/// 0.278708918755311467154040663921640725159341748i`;
/// `zeta(mpc(-0.5,10**7))` = `344000.352916333495718061464649621833081107683
/// − 2813240.20612312896803844705052023805545640474i`.
#[test]
fn zeta_far_up_the_critical_strip() {
    assert_c(
        "zeta(1/2 + 10^6*I)",
        40,
        "0.076089069738227100005564558379927322",
        "2.8051021010192989553938367165649402",
    );
    assert_c(
        "zeta(2 + 10^6*I)",
        40,
        "0.96481226472580120693016721706187663",
        "0.27870891875531146715404066392164072",
    );
    assert_c(
        "zeta(-1/2 + 10^7*I)",
        40,
        "344000.35291633349571806146464962183",
        "-2813240.2061231289680384470505202380",
    );
}

/// At 100 digits the Riemann–Siegel parameters reach the edge of the `f64`
/// range (a working precision above 1,000 bits made one of them overflow
/// during development, and the phase factor was formed at 16 bits).  mpmath
/// `zeta(mpc(0.5,10**6))` at 110 digits =
/// `0.0760890697382271000055645583799273223107986047447222826579782173850625
/// 0498238818630801746583609816946808751573476 + …i`.
#[test]
fn zeta_far_up_the_critical_strip_at_100_digits() {
    let got = value("zeta(1/2 + 10^6*I)", 100);
    let (re, _) = split(&got);
    assert!(
        re.starts_with(
            "0.07608906973822710000556455837992732231079860474472228265797821738506249823881863080174658360981"
        ),
        "{got}"
    );
}

/// An inexact argument off the real axis: the bound of `ζ′` over its ball
/// needed `abs(s)` terms summed one by one, refused beyond `10⁵` — every
/// precision up to the cap was tried (20 s), and the radius, below the
/// `f64` range at 1,162 bits, then counted as 0 (an exact argument).  The
/// partial sums are now bounded in closed form and the radius kept in
/// `log₂`.  mpmath `zeta(mpc(sqrt(2)/2, 200000))` =
/// `1.64536831084872459975862783042650090045397501 +
/// 0.537204099630915349894895469494298099629603651i`.
#[test]
fn zeta_of_an_inexact_argument_far_up() {
    assert_c(
        "zeta(sqrt(2)/2 + 200000*I)",
        40,
        "1.6453683108487245997586278304265009",
        "0.53720409963091534989489546949429809",
    );
}

/// `eval_complex64` takes the same routines (refused before).  mpmath
/// `airyai(mpc(1,1))` = `0.060458308371838149 − 0.15188956587718140i`.
#[test]
fn complex64_of_the_new_functions() {
    let ctx = Context::new();
    let z = ctx
        .parse("airyai(1 + I)")
        .unwrap()
        .eval_complex64()
        .unwrap();
    assert!((z.re - 0.060_458_308_371_838_15).abs() < 1e-16, "{z}");
    assert!((z.im + 0.151_889_565_877_181_4).abs() < 1e-16, "{z}");
}
