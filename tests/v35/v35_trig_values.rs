//! Trigonometric functions at rational multiples of `π` in radicals, and
//! the `sign`, `|·|`, `arg` of directions (0.43).  Before, `eval` gave the
//! values of `sin`, `cos`, `tan` only at the denominators 1, 2, 3, 4, 6, so
//! `cos(π/5)` stayed and `Eq(sin(π/12), (√6 − √2)/4)` was undecided.  The
//! denominators are SymPy 1.14's (`cos.eval`, `tan.eval`, `cot.eval` in
//! `sympy/functions/elementary/trigonometric.py`): `sin`, `cos`, `sec`,
//! `csc` at `n ∈ {5, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120}`, `tan`,
//! `cot` also at `n ∈ {16, 48, 80, 240}`; found by a sweep of `n ≤ 500`
//! (`not f(Rational(p, n)*pi).has(f)`).  Every value is checked against
//! mpmath 1.3.0 (`mp.cos(k*mp.pi/n)` at 80 digits, printed to 60).

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// The significant digits of a decimal (`"-0.0123"` → `('-', "123")`).
fn digits(s: &str) -> (bool, String) {
    let negative = s.starts_with('-');
    let d: String = s
        .chars()
        .filter(char::is_ascii_digit)
        .skip_while(|&c| c == '0')
        .collect();
    (negative, d)
}

/// Do the decimals `got` and `want` agree to 56 significant digits (both
/// have 60; the last ones may round differently)?
fn agrees(got: &str, want: &str) -> bool {
    let ((sg, dg), (sw, dw)) = (digits(got), digits(want));
    sg == sw && dg.len() >= 56 && dw.len() >= 56 && dg[..56] == dw[..56]
}

/// `(k, n, value)` of a table entry `"k/n value"`.
fn entry(e: &str) -> (i64, i64, &str) {
    let (angle, value) = e.split_once(' ').unwrap();
    let (k, n) = angle.split_once('/').unwrap();
    (k.parse().unwrap(), n.parse().unwrap(), value)
}

/// `f(k·π/n)` evaluated: in radicals (no `f` left) and equal to `want` at
/// 60 digits.
fn check_value(ctx: &Context, f: &str, k: i64, n: i64, want: &str) {
    let input = format!("{f}({k}*pi/{n})");
    let v = parse(ctx, &input).eval();
    let s = v.to_string();
    assert!(
        !s.contains("sin") && !s.contains("cos") && !s.contains("tan"),
        "{input} not evaluated: {s}"
    );
    let got = v.eval_decimal(60).unwrap();
    assert!(agrees(&got, want), "{input} = {s} = {got}, mpmath {want}");
}

/// `"k/n cos(kπ/n)"` for `0 < k < n/2`, `gcd(k, n) = 1`, `n ∈ {5, 8, 10,
/// 12, 15, 20, 24, 30, 40, 60, 120}`: mpmath 1.3.0, `mp.dps = 80`,
/// `mp.nstr(mp.cos(k*mp.pi/n), 60, strip_zeros=False)`.
const COS_MPMATH: &[&str] = &[
    "1/5 0.809016994374947424102293417182819058860154589902881431067724",
    "2/5 0.309016994374947424102293417182819058860154589902881431067724",
    "1/8 0.923879532511286756128183189396788286822416625863642486115098",
    "3/8 0.382683432365089771728459984030398866761344562485627041433801",
    "1/10 0.951056516295153572116439333379382143405698634125750222447306",
    "3/10 0.587785252292473129168705954639072768597652437643145991072272",
    "1/12 0.965925826289068286749743199728897367633904839008404550402343",
    "5/12 0.258819045102520762348898837624048328349068901319930513814003",
    "1/15 0.978147600733805637928566747869599532459737808862677107885178",
    "2/15 0.913545457642600895502127571985317177940810459377474545061000",
    "4/15 0.669130606358858213826273330686780473599583218959795676817453",
    "7/15 0.104528463267653471399834154802498119080655869474593113993275",
    "1/20 0.987688340595137726190040247693437260758406861589880434923905",
    "3/20 0.891006524188367862359709571413626312770518519036088745405522",
    "7/20 0.453990499739546791560408366357871198983047703049021556985316",
    "9/20 0.156434465040230869010105319467166892313899892085660790084641",
    "1/24 0.991444861373810411144557526928562871277738274448102271458775",
    "5/24 0.793353340291235164579776961501299276628675921051910234224481",
    "7/24 0.608761429008720639416097542898164004516393711962475230024974",
    "11/24 0.130526192220051591548406227895489010193740704811732251890617",
    "1/30 0.994521895368273336922691944980570381520792088709319427366559",
    "7/30 0.743144825477394235014697048974256977189113873498026386040124",
    "11/30 0.406736643075800207753985990341497612923139651066173436294286",
    "13/30 0.207911690817759337101742284405125166216584760627723836407182",
    "1/40 0.996917333733127976197773408742044420158926881594843892194446",
    "3/40 0.972369920397676601833645834118797644002576369406261190704212",
    "7/40 0.852640164354092221519383458130412135817266281929000395124755",
    "9/40 0.760405965600030938174594364844901999888747066038990138867734",
    "11/40 0.649448048330183655726320770893762879277502812171913975187558",
    "13/40 0.522498564715948864987897880178293823415387191523263709251821",
    "17/40 0.233445363855905411767744430202870848785741903887519600364243",
    "19/40 0.0784590957278449450329602459934596986819558771060315613466223",
    "1/60 0.998629534754573873784492058439436580590952290767785532441441",
    "7/60 0.933580426497201748990043063139570741405965268537466806881650",
    "11/60 0.838670567945424029637590941804547894039500265095923824922013",
    "13/60 0.777145961456970879979937743672403849092065376451806016797008",
    "17/60 0.629320391049837452705902458279970426566862412129866639460328",
    "19/60 0.544639035015027082224083692081565381607904587718763975456125",
    "23/60 0.358367949545300273484137789413466834191544449460013795463577",
    "29/60 0.0523359562429438327221186296090784187310182539401649204835094",
    "1/120 0.999657324975557280036760888367679875949875971241078461264606",
    "7/120 0.983254907563954584554632056430508987574623654665667482500766",
    "11/120 0.958819734868193049761028541392598291049180918849058982894275",
    "13/120 0.942641491092178394777167736282311882844866690666503885763317",
    "17/120 0.902585284349860606762645149095771756816383827099208518821524",
    "19/120 0.878817112661965374129995143684524799610657309609109033343541",
    "23/120 0.824126188622015661729684903102312058134427949993176957474902",
    "29/120 0.725374371012287637993284111189727442263439014961539382530032",
    "31/120 0.688354575693753984389256143419612293486469312096005967582644",
    "37/120 0.566406236924832831821625052233764932518688456099811494675230",
    "41/120 0.477158760259608415048863008189386052534488779717814752012785",
    "43/120 0.430511096808295144376148356508279487640238425495032397519216",
    "47/120 0.333806859233770928828311285536746108297120842493753507313209",
    "49/120 0.284015344703922617444389690699185350516107057310255223121567",
    "53/120 0.182235525492147456602573371437409882956119624627513746895582",
    "59/120 0.0261769483078731526106116855541126637933910276801086382187863",
];

/// `"k/n tan(kπ/n)"` for `0 < k < n/2`, `gcd(k, n) = 1`, `n ∈ {16, 48,
/// 80, 240}`: mpmath 1.3.0, `mp.dps = 80`, `mp.nstr(mp.tan(k*mp.pi/n), 60,
/// strip_zeros=False)`.
const TAN_MPMATH: &[&str] = &[
    "1/16 0.198912367379658006911597622644676228597850501321590981921117",
    "3/16 0.668178637919298919997757686523080761552472251379082816185914",
    "5/16 1.49660576266548901760113513494247691869181600213297896253927",
    "7/16 5.02733949212584810451497507106407238573719425207548712827448",
    "1/48 0.0655434628152382285645437248799427949221949914029802006303101",
    "5/48 0.339454258863375815537118900893280693015855082795791343962515",
    "7/48 0.493145426031304123925300893992431448885010796514566341048675",
    "11/48 0.876976462992756867248688567129922322677601328472341584619169",
    "13/48 1.14028145816754857419174148194935652986322953291881673172630",
    "17/48 2.02779940198922480977138427799656280956317251469477170767231",
    "19/48 2.94590500454578732727341691887392258592279578573447419836293",
    "23/48 15.2570516882655391096173820057228664698790442110909778599607",
    "1/80 0.0392901070076696401722471542213737395549928139666187018950771",
    "3/80 0.118357799640767835379529804324443591672604217302687324121835",
    "7/80 0.282029168302153105135058706075076803972572012183464480794421",
    "9/80 0.368919477109827082280552506484068950575762992267467055517017",
    "11/80 0.461006314427318166715648360347198827169720067788636105129297",
    "13/80 0.560026908474076952912440772929244403218293907595889373304129",
    "17/80 0.788336434585092506589265690088615113434441908863622320180510",
    "19/80 0.924390491658207054768094471090420960287050207869918349327395",
    "21/80 1.08179390530744395089359744970185165783524485536778430265819",
    "23/80 1.26849395274532458510589717876628231908850876853265991816817",
    "27/80 1.78562848475394094619859403513663520335780936551450603694699",
    "29/80 2.16916768535425144176137014093428872671619351275678934749001",
    "31/80 2.71061860933490553268180665784862449585790130153055348680469",
    "33/80 3.54573254255973234047603961328509754741019133668735019052060",
    "37/80 8.44895733982160224311472299784969073755351408727866167019335",
    "39/80 25.4516995793570789322156071119055486748924694740617572604661",
    "1/240 0.0130907170848350856026966017331329035390337599041347009268211",
    "7/240 0.0918870916679029348772803913879974259884681042455387121434823",
    "11/240 0.144993099491635311553606490543597769683695541869333241507298",
    "13/240 0.171831426830125180174502920980581327849173817338816184638418",
    "17/240 0.226276898255000784215192353398683812865995533283396782015973",
    "19/240 0.253967646474943659621704885890056008798121172086027016602200",
    "23/240 0.310508253462831067623164599802517864458848871375539305774378",
    "29/240 0.398959545973719412524162322633398524545599440301800977224992",
    "31/240 0.429633905966836026668197116411391546920244900695470381183269",
    "37/240 0.526125451492034005495925559049925836976347059572170263647856",
    "41/240 0.594937481538893353274038630996176003859823913054292531463544",
    "43/240 0.630953011384306208367363677406402299208274271797942513533886",
    "47/240 0.706730127054298973690263365743718325177893149212485571558637",
    "49/240 0.746735417783721671737500140715208796900055083382471590962549",
    "53/240 0.831691220879731105627183151911822644175063480614120320691102",
    "59/240 0.974156870921680978014733431314843186571102089869145276893676",
    "61/240 1.02652871406005483579346108590389067534716188516362215914062",
    "67/240 1.20236931074279985411898726004147261405219339608170155153072",
    "71/240 1.33916240770788216022591955001857329342023441968340101568405",
    "73/240 1.41496727211569506123347441409705574078694558045193461725023",
    "77/240 1.58490407678062657446682723465138363856978041028453445905397",
    "79/240 1.68084888081576711128454917420564688823186468351878463218858",
    "83/240 1.90068736869526044043316949340334067403260024822597772952440",
    "89/240 2.32756303939659557168804726514633655347491917918311485811326",
    "91/240 2.50651979653564377369923576926428103091397343002923499204877",
    "97/240 3.22052631080772068644620216656901318527257504867525448118374",
    "101/240 3.93750941854185972556046707434532135610426464378903370407381",
    "103/240 4.41936409643134974246262140758675005309372764209235879805753",
    "107/240 5.81965719803172686194241275083765930367272858489017137601504",
    "109/240 6.89687994467412741934921379025808276903742300683218938542554",
    "113/240 10.8829214403061778284027338452518121402387470914859000379911",
    "119/240 76.3900093111360531970084191937835776253189384276436749395577",
];

/// `cos(kπ/n)` and `sin((n − 2k)π/(2n))` (the same number) for every
/// `0 < k < n/2` prime to `n` at the denominators SymPy evaluates and
/// symplex did not.  Before, all of them stayed unevaluated.  Oracle:
/// `COS_MPMATH`.
#[test]
fn cos_and_sin_at_the_new_denominators_agree_with_mpmath() {
    let ctx = Context::new();
    for (k, n, want) in COS_MPMATH.iter().copied().map(entry) {
        check_value(&ctx, "cos", k, n, want);
        check_value(&ctx, "sin", n - 2 * k, 2 * n, want);
    }
}

/// `tan(kπ/n)` and `cot((n/2 − k)π/n)` for `n ∈ {16, 48, 80, 240}`, where
/// SymPy evaluates `tan` and `cot` (by `tan(x/2) = (1 − cos x)/sin x`) but
/// not `sin`, `cos`; and `tan`, `cot` at `n = 5, 10, 15, 24, 40, 120`.
/// Before, all stayed (`cot` as `cos/sin`).  Oracle: `TAN_MPMATH`, and
/// mpmath `tan`/`cot` at 80 digits for the second list.
#[test]
fn tan_and_cot_agree_with_mpmath() {
    let ctx = Context::new();
    for (k, n, want) in TAN_MPMATH.iter().copied().map(entry) {
        check_value(&ctx, "tan", k, n, want);
        if k % 3 == 1 {
            check_value(&ctx, "cot", n / 2 - k, n, want);
        }
    }
    for (f, k, n, want) in [
        (
            "tan",
            1,
            5,
            "0.726542528005360885895466757480618749616092392965208462750066",
        ),
        (
            "tan",
            3,
            10,
            "1.37638192047117353820720958191088767952589933600815866336568",
        ),
        (
            "tan",
            2,
            15,
            "0.445228685308536163922367030645665754987430678427926516556270",
        ),
        (
            "tan",
            7,
            24,
            "1.30322537284120575586814900899032094645347035146934142720021",
        ),
        (
            "tan",
            9,
            40,
            "0.854080685463466637522860890293544949773236722484076621180354",
        ),
        (
            "tan",
            17,
            120,
            "0.476975532698160183049731778622490669680753069243295972760040",
        ),
        (
            "cot",
            1,
            5,
            "1.37638192047117353820720958191088767952589933600815866336568",
        ),
        (
            "cot",
            7,
            15,
            "0.105104235265676462511502380139881915250246649545686882878662",
        ),
        (
            "cot",
            5,
            24,
            "1.30322537284120575586814900899032094645347035146934142720021",
        ),
        (
            "cot",
            11,
            40,
            "0.854080685463466637522860890293544949773236722484076621180354",
        ),
    ] {
        check_value(&ctx, f, k, n, want);
    }
}

/// Which applications evaluate: `sin`, `cos`, `sec`, `csc` exactly at the
/// denominators of SymPy's `cos.eval`, `tan`, `cot` also at twice them, at
/// every numerator in `(−n, 2n]` prime to `n` (the reductions of the
/// period and the quadrants), with the value of the unevaluated application
/// (symplex's numerical `sin`/`cos`, an independent code path) at 30
/// digits.  `cos(π/7)`, `cos(π/16)`, `tan(π/7)` stay, as in SymPy.  Before,
/// only the denominators 1, 2, 3, 4, 6 evaluated.  Oracle: SymPy 1.14 (the
/// sweep in the module documentation).
#[test]
fn every_function_evaluates_exactly_at_sympys_denominators() {
    const COS_SET: [i64; 16] = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120];
    const TAN_EXTRA: [i64; 4] = [16, 48, 80, 240];
    let ctx = Context::new();
    let gcd = |mut a: i64, mut b: i64| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a.abs()
    };
    let others = [7, 9, 11, 14, 18, 32, 36, 45];
    for n in COS_SET.iter().chain(&TAN_EXTRA).chain(&others).copied() {
        // A sample of the numerators at the large denominators.
        let step = if n >= 80 { 13 } else { 1 };
        for p in ((-n + 1)..=(2 * n)).step_by(step) {
            if gcd(p, n) != 1 {
                continue;
            }
            for f in ["sin", "cos", "tan", "cot", "sec", "csc"] {
                let input = format!("{f}({p}*pi/{n})");
                let e = parse(&ctx, &input);
                let v = e.eval();
                let s = v.to_string();
                let folded = !s.contains("sin(") && !s.contains("cos(") && !s.contains("tan(");
                let expected =
                    COS_SET.contains(&n) || (matches!(f, "tan" | "cot") && TAN_EXTRA.contains(&n));
                assert_eq!(folded, expected, "{input} -> {s}");
                if folded {
                    match (v.eval_decimal(30), e.eval_decimal(30)) {
                        (Ok(a), Ok(b)) => assert_eq!(a, b, "{input} -> {s}"),
                        // Poles: both refuse (`tan(π/2)` is `zoo`).
                        (Err(_), Err(_)) => assert_eq!(s, "zoo", "{input}"),
                        (a, b) => panic!("{input} -> {s}: {a:?} vs {b:?}"),
                    }
                }
            }
        }
    }
}

/// The values come out in SymPy's normal forms (its formulas through the
/// canonical arithmetic), except that a square root of a difference
/// `√(a − b√c)` is written `√(a² − b²c)/√(a + b√c)` (`sin(π/5)` is
/// `√5/(4·√(√5/8 + 5/8))`): with a radicand that is a sum of positive terms
/// the assumptions know the value is real, and `re`, `im`, `|·|` of it
/// decompose (with SymPy's form, `∫ 1/(x⁸ + 1)` grew from 566 to 2,986
/// characters of `re(…)`, `im(…)`).  Before, every input stayed
/// unevaluated.
/// SymPy 1.14: `cos(pi/5)` → `1/4 + sqrt(5)/4`, `sin(pi/12)` →
/// `-sqrt(2)/4 + sqrt(6)/4`, `tan(pi/8)` → `-1 + sqrt(2)`, `cos(pi/15)` →
/// `-1/8 + sqrt(5)/8 + sqrt(3)*sqrt(sqrt(5)/8 + 5/8)/2`, `cos(pi/24)` →
/// `sqrt(1/2 - sqrt(2)/4)/2 + sqrt(3)*sqrt(sqrt(2)/4 + 1/2)/2`,
/// `sin(pi/5)` → `sqrt(5/8 - sqrt(5)/8)`, `tan(pi/10)` → `sqrt(1 -
/// 2*sqrt(5)/5)`, `cot(pi/8)` → `1 + sqrt(2)`, `cot(pi/12)` → `sqrt(3) +
/// 2`, `cot(pi/5)` → `sqrt(2*sqrt(5)/5 + 1)`, `sec(pi/5)` → `1/(1/4 +
/// sqrt(5)/4)`, `csc(pi/12)` → `1/(-sqrt(2)/4 + sqrt(6)/4)`, `cos(pi/20)`
/// → `sqrt(2)*sqrt(5/8 - sqrt(5)/8)/2 + sqrt(2)*(1/4 + sqrt(5)/4)/2`;
/// `cos(pi/7)`, `cos(pi/16)`, `tan(pi/9)` stay.
#[test]
fn values_come_out_in_sympys_normal_forms() {
    let ctx = Context::new();
    for (input, want) in [
        ("cos(pi/5)", "1/4*sqrt(5) + 1/4"),
        ("cos(2*pi/5)", "1/4*sqrt(5) - 1/4"),
        ("sin(pi/12)", "-1/4*sqrt(2) + 1/4*sqrt(6)"),
        ("cos(pi/12)", "1/4*sqrt(2) + 1/4*sqrt(6)"),
        ("tan(pi/8)", "sqrt(2) - 1"),
        ("tan(pi/12)", "-sqrt(3) + 2"),
        (
            "cos(pi/15)",
            "1/2*sqrt(3)*sqrt(1/8*sqrt(5) + 5/8) + 1/8*sqrt(5) - 1/8",
        ),
        (
            "cos(pi/24)",
            "1/2*sqrt(3)*sqrt(1/4*sqrt(2) + 1/2) + sqrt(2)/(8*sqrt(1/4*sqrt(2) + 1/2))",
        ),
        ("sin(pi/5)", "sqrt(5)/(4*sqrt(1/8*sqrt(5) + 5/8))"),
        ("cos(pi/8)", "sqrt(1/4*sqrt(2) + 1/2)"),
        ("cos(3*pi/8)", "sqrt(2)/(4*sqrt(1/4*sqrt(2) + 1/2))"),
        ("tan(pi/10)", "sqrt(5)/(5*sqrt(2/5*sqrt(5) + 1))"),
        ("tan(pi/5)", "sqrt(5)/sqrt(2*sqrt(5) + 5)"),
        (
            "tan(pi/16)",
            "2*sqrt(2)*sqrt(1/4*sqrt(2) + 1/2) - 2*sqrt(2)*(1/4*sqrt(2) + 1/2)",
        ),
        ("tan(7*pi/10)", "-sqrt(2/5*sqrt(5) + 1)"),
        ("cot(pi/8)", "sqrt(2) + 1"),
        ("cot(pi/12)", "sqrt(3) + 2"),
        ("cot(pi/5)", "sqrt(2/5*sqrt(5) + 1)"),
        ("sec(pi/5)", "1/(1/4*sqrt(5) + 1/4)"),
        ("csc(pi/12)", "1/(-1/4*sqrt(2) + 1/4*sqrt(6))"),
        (
            "cos(pi/20)",
            "sqrt(10)/(8*sqrt(1/8*sqrt(5) + 5/8)) + 1/2*sqrt(2)*(1/4*sqrt(5) + 1/4)",
        ),
        // Known real: `re` is the value itself.
        (
            "re(sin(pi/8) + I*cos(pi/8))",
            "sqrt(2)/(4*sqrt(1/4*sqrt(2) + 1/2))",
        ),
        ("cos(pi/7)", "cos(1/7*pi)"),
        ("cos(pi/16)", "cos(1/16*pi)"),
        ("tan(pi/9)", "tan(1/9*pi)"),
        // Unchanged at the classical denominators.
        ("cos(pi/4)", "1/2*sqrt(2)"),
        ("cot(pi/3)", "1/3*sqrt(3)"),
        ("tan(5*pi/6)", "-1/3*sqrt(3)"),
    ] {
        assert_eq!(parse(&ctx, input).eval().to_string(), want, "{input}");
    }
}

/// `sinh`, `cosh`, `tanh` (and `coth = cosh/sinh`) at `i·kπ/n` are `i·sin`,
/// `cos`, `i·tan`, `−i·cot` there.  Before, only the rational values
/// folded (`cosh(iπ/3) = 1/2`) and `cosh(iπ/4)` stayed.  SymPy 1.14:
/// `cosh(I*pi/5)` → `1/4 + sqrt(5)/4`, `sinh(I*pi/8)` → `I*sqrt(1/2 -
/// sqrt(2)/4)`, `tanh(I*pi/12)` → `I*(2 - sqrt(3))`, `sinh(I*pi/4)` →
/// `sqrt(2)*I/2`, `coth(I*pi/8)` → `-I*(1 + sqrt(2))`; `cosh(I*pi/7)`
/// → `cos(pi/7)` (here `cosh(iπ/7)` stays).
#[test]
fn hyperbolic_functions_at_imaginary_multiples_of_pi() {
    let ctx = Context::new();
    for (input, want) in [
        ("cosh(I*pi/5)", "1/4*sqrt(5) + 1/4"),
        ("sinh(I*pi/8)", "sqrt(2)*I/(4*sqrt(1/4*sqrt(2) + 1/2))"),
        ("tanh(I*pi/12)", "(-sqrt(3) + 2)*I"),
        ("sinh(I*pi/4)", "1/2*sqrt(2)*I"),
        ("cosh(I*pi/4)", "1/2*sqrt(2)"),
        ("sinh(-I*pi/4)", "-1/2*sqrt(2)*I"),
        ("coth(I*pi/8)", "-(sqrt(2) + 1)*I"),
        ("cosh(I*pi/7)", "cosh(1/7*pi*I)"),
    ] {
        assert_eq!(parse(&ctx, input).eval().to_string(), want, "{input}");
    }
}

/// The inverse functions read the new forms back where SymPy's tables do
/// (the denominators 3, 4, 5, 6, 8, 10, 12), and keep the others.  SymPy
/// 1.14: `acos(1/4 + sqrt(5)/4)` → `pi/5`, `acos(cos(pi/12))` → `pi/12`,
/// `asin(sin(pi/5))` → `pi/5`, `atan(tan(pi/8))` → `pi/8`,
/// `acos(sqrt(sqrt(5)/8 + 5/8))` → `pi/10`; `acos(cos(pi/15))` and
/// `acos(cos(pi/24))` stay `acos(<radical>)`.
#[test]
fn inverse_functions_read_the_values_back() {
    let ctx = Context::new();
    for (input, want) in [
        ("acos(1/4 + sqrt(5)/4)", "1/5*pi"),
        ("acos(cos(pi/12))", "1/12*pi"),
        ("asin(sin(pi/5))", "1/5*pi"),
        ("atan(tan(pi/8))", "1/8*pi"),
        ("atan(tan(3*pi/10))", "3/10*pi"),
        ("acos(cos(7*pi/12))", "7/12*pi"),
        ("acos(sqrt(sqrt(5)/8 + 5/8))", "1/10*pi"),
    ] {
        assert_eq!(parse(&ctx, input).eval().to_string(), want, "{input}");
    }
    for input in ["acos(cos(pi/15))", "acos(cos(pi/24))"] {
        let s = parse(&ctx, input).eval().to_string();
        assert!(
            s.starts_with("acos(") && !s.contains("pi"),
            "{input} -> {s}"
        );
    }
}

/// `asin`, `acos`, `atan` of `sin`, `cos`, `tan` at a rational multiple of
/// `π` that stays unevaluated are its principal angle.  Before, they
/// stayed.  SymPy 1.14: `asin(sin(7*pi/102))` → `7*pi/102`,
/// `acos(cos(14*pi/9))` → `4*pi/9`, `asin(cos(49*pi/85))` → `-13*pi/170`,
/// `acos(sin(-17*pi/11))` → `pi/22`, `atan(tan(-34*pi/43))` → `9*pi/43`,
/// `atan(tan(pi/9))` → `pi/9`, `asin(sin(-15*pi/11))` → `4*pi/11`,
/// `atan(cot(-41*pi/23))` → `13*pi/46`, `acot(tan(162*pi/114))` →
/// `3*pi/38`, `acot(tan(-13*pi/9))` → `-pi/18` (`cot` is `cos/sin` and
/// `acot(y)` is `atan(1/y)` here).
#[test]
fn inverse_functions_of_unevaluated_values_are_principal_angles() {
    let ctx = Context::new();
    for (input, want) in [
        ("asin(sin(7*pi/102))", "7/102*pi"),
        ("acos(cos(14*pi/9))", "4/9*pi"),
        ("asin(cos(49*pi/85))", "-13/170*pi"),
        ("acos(sin(-17*pi/11))", "1/22*pi"),
        ("atan(tan(-34*pi/43))", "9/43*pi"),
        ("atan(tan(pi/9))", "1/9*pi"),
        ("asin(sin(-15*pi/11))", "4/11*pi"),
        ("acos(cos(-207*pi/104))", "1/104*pi"),
        ("atan(cot(-41*pi/23))", "13/46*pi"),
        ("acot(tan(162*pi/114))", "3/38*pi"),
        ("acot(tan(-13*pi/9))", "-1/18*pi"),
    ] {
        assert_eq!(parse(&ctx, input).eval().to_string(), want, "{input}");
    }
}

/// Equalities between the values are decided, also those that need an
/// exact zero test of a long radical (`sin²x + cos²x = 1` at `x = 7π/60`,
/// whose minimal polynomial goes through resultants of degree 256: the
/// first version took 23 s here, a clearly false one 25 s).  Before,
/// `eval` left `Eq(sin(π/12), (√6 − √2)/4)` undecided.  SymPy 1.14:
/// `Eq(sin(pi/12), (sqrt(6) - sqrt(2))/4)` → `True`, `Eq(cos(pi/5) -
/// cos(2*pi/5), S(1)/2)` → `True`, `Eq(sin(102*pi/80)**2 +
/// cos(102*pi/80)**2, 1)` → `True`, `Eq(tan(79*pi/40)*cos(79*pi/40),
/// sin(79*pi/40) + 1/448)` → `False`; `Eq(tan(79*pi/40)*cos(79*pi/40),
/// sin(79*pi/40))` did not finish in 25 s (here `True`).
#[test]
fn equalities_between_values_are_decided() {
    let ctx = Context::new();
    for (input, want) in [
        ("Eq(sin(pi/12), (sqrt(6) - sqrt(2))/4)", "True"),
        ("Eq(cos(pi/5) - cos(2*pi/5), 1/2)", "True"),
        ("Eq(sin(7*pi/60)^2 + cos(7*pi/60)^2, 1)", "True"),
        ("Eq(sin(102*pi/80)^2 + cos(102*pi/80)^2, 1)", "True"),
        ("Eq(tan(79*pi/40)*cos(79*pi/40), sin(79*pi/40))", "True"),
        ("Eq(tan(pi/24)*cos(pi/24), sin(pi/24))", "True"),
        ("Eq(cos(2*94*pi/120), 2*cos(94*pi/120)^2 - 1)", "True"),
        (
            "Eq(tan(79*pi/40)*cos(79*pi/40), sin(79*pi/40) + 1/448)",
            "False",
        ),
        ("Ne(cos(pi/5), (1 + sqrt(5))/4)", "False"),
        ("Eq(sqrt(3 + 2*sqrt(2)), 1 + sqrt(2))", "True"),
    ] {
        let b = ctx.parse_bool(input).unwrap();
        assert_eq!(b.eval().to_string(), want, "{input}");
    }
    assert_eq!(
        parse(&ctx, "cos(pi/5) - cos(2*pi/5)").eval().to_string(),
        "1/2"
    );
}

/// `exp(iπk/n)` keeps its form off the classical denominators, as in
/// SymPy (`exp(I*pi/5)` stays); `e^{iπ/4} = (1 + i)·√2/2` as before.
#[test]
fn exp_at_imaginary_multiples_of_pi_keeps_its_form() {
    let ctx = Context::new();
    for (input, want) in [
        ("exp(I*pi/5)", "exp(1/5*pi*I)"),
        ("exp(I*pi/12)", "exp(1/12*pi*I)"),
        ("exp(I*pi/4)", "1/2*sqrt(2) + 1/2*sqrt(2)*I"),
    ] {
        assert_eq!(parse(&ctx, input).eval().to_string(), want, "{input}");
    }
}

/// `r·sign(r) = |r|` for a real `r` (at `r = 0` too), so `|m| − m·sign(m)`
/// is 0 and the quotient below is `0/0 = nan`.  Before, the product stayed
/// and `ratsimp` gave 0 (true value `nan`).  For a non-real `z`,
/// `z·sign(z) = z²/|z|` is not `|z|` and stays.  SymPy 1.14 keeps `Abs(m)
/// - m*sign(m)` (`simplify` gives 0) and `m*sign(m)`.
#[test]
fn a_real_times_its_sign_is_its_modulus() {
    let ctx = Context::new();
    ctx.symbol_with("m", &[Assumption::Real]).unwrap();
    for (input, want) in [
        ("m*sign(m)", "abs(m)"),
        ("abs(m) - m*sign(m)", "0"),
        ("-3*m*sign(m)", "-3*abs(m)"),
        ("(m - 1)*sign(m - 1)", "abs(m - 1)"),
        ("z*sign(z)", "z*sign(z)"),
        ("m^2*sign(m)", "m^2*sign(m)"),
    ] {
        assert_eq!(parse(&ctx, input).to_string(), want, "{input}");
    }
    let q = parse(&ctx, "((x + 1)^2 - x^2 - 2*x - 1)/(abs(m) - m*sign(m))");
    assert_eq!(q.ratsimp().to_string(), "nan");
    let z = parse(&ctx, "abs(z) - z*sign(z)");
    assert_ne!(z.to_string(), "0");
    let at = z.subs(&parse(&ctx, "z"), &parse(&ctx, "1 + 2*I"));
    assert_ne!(at.eval_decimal(20).unwrap(), "0");
}

/// `sign`, `|·|`, `arg` of a directed infinity and of `e^z` depend only on
/// a direction.  Before, all of these stayed.  SymPy 1.14:
/// `Abs(exp(I)*oo)` → `oo`, `Abs((1 + I)*oo)` → `oo`, `sign(exp(I)*oo)` →
/// `sign(exp(I))` (= `exp(I)`, its value since `|e^i| = 1`),
/// `sign((1 + I)*oo)` → `sign(1 + I)`, `arg(exp(2*I))` → `2`,
/// `arg(exp(4*I))` → `4 - 2*pi`, `arg(exp(-10*I))` → `-10 + 4*pi`,
/// `arg(exp(sqrt(2)*I))` → `sqrt(2)`; mpmath at 40 digits:
/// `arg(exp(1000j))` = 0.97353615844575016887… = `1000 − 318π`,
/// `arg(exp(4j))` = −2.28318530717958647692… = `4 − 2π`.  `|x·∞|` (`x` may
/// be 0) and `arg(e^{ix})` for a real symbol `x` stay.
#[test]
fn sign_abs_and_arg_of_directions() {
    let ctx = Context::new();
    ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    for (input, want) in [
        ("sign(exp(I)*oo)", "exp(I)"),
        ("sign(-exp(I)*oo)", "-exp(I)"),
        ("abs(exp(I)*oo)", "oo"),
        ("abs((1 + I)*oo)", "oo"),
        ("abs((2 - 2*I)*oo)", "oo"),
        ("abs(sqrt(2)*exp(14*I/3))", "sqrt(2)"),
        ("abs(pi*exp(10*I))", "pi"),
        ("sign((1 + I)*oo)", "sign(1 + I)"),
        ("sign(exp(I))", "exp(I)"),
        ("sign(exp(2 + 3*I))", "exp(3*I)"),
        ("sign(exp(r*I))", "exp(r*I)"),
        ("sign(exp(r))", "1"),
        ("arg(exp(I))", "1"),
        ("arg(exp(2*I))", "2"),
        ("arg(exp(4*I))", "-2*pi + 4"),
        ("arg(3*exp(4*I))", "-2*pi + 4"),
        ("arg(exp(-10*I))", "4*pi - 10"),
        ("arg(exp(1000*I))", "-318*pi + 1000"),
        ("arg(exp(sqrt(2)*I))", "sqrt(2)"),
        ("arg(exp(2 + 3*I))", "3"),
        ("abs(x*oo)", "abs(x*oo)"),
        ("arg(exp(r*I))", "arg(exp(r*I))"),
    ] {
        assert_eq!(parse(&ctx, input).to_string(), want, "{input}");
    }
    let v = parse(&ctx, "arg(exp(1000*I))").eval_decimal(30).unwrap();
    assert!(v.starts_with("0.97353615844575016887940411711"), "{v}");
}
