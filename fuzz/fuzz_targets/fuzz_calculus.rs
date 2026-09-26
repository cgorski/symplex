//! The calculus routines against independent numerical oracles.
//!
//! The first input byte picks a routine, the rest decode its case (see
//! `calc/mod.rs` for the generators and oracles): definite integrals
//! against adaptive quadrature, limits against the function at
//! `p ± 10⁻ᵏ`, series against the order of their residual, derivatives
//! against central differences, `solve` by substitution (and no real root
//! missed), infinite sums against Richardson-extrapolated partial sums and
//! finite sums against the partial sums, `dsolve` by substitution.  A
//! wrong answer panics with the case and both values; a case an oracle
//! cannot decide is skipped.  Set `FUZZ_SHOW=1` to print each decoded case.
#![no_main]

#[path = "common/choose.rs"]
mod choose;

#[path = "calc/mod.rs"]
mod calc;

use calc::V;
use choose::Src;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut r = Src::new(data);
    let routine = r.below(7);
    let (desc, run): (String, Box<dyn FnOnce() -> V>) = match routine {
        0 => {
            let (f, a, b) = calc::gen_defint(&mut r);
            (
                format!("integrate({f}, (x, {}, {}))", a.s(), b.s()),
                Box::new(move || calc::check_defint(&f, &a, &b)),
            )
        }
        1 => {
            let (f, p, d) = calc::gen_limit(&mut r);
            (
                format!("limit({f}, x, {}, {d:?})", p.s()),
                Box::new(move || calc::check_limit(&f, &p, d)),
            )
        }
        2 => {
            let (f, p, n) = calc::gen_series(&mut r);
            (
                format!("series({f}, x, {}, {n})", p.s()),
                Box::new(move || calc::check_series(&f, &p, n)),
            )
        }
        3 => {
            let (f, p) = calc::gen_diff(&mut r);
            (
                format!("diff({f}, x) at {}/{}", p.0, p.1),
                Box::new(move || calc::check_diff(&f, p)),
            )
        }
        4 => {
            let (l, rr) = calc::gen_solve(&mut r);
            (
                format!("solve({l} = {rr}, x)"),
                Box::new(move || calc::check_solve(&l, &rr)),
            )
        }
        5 => {
            let (t, a, inf) = calc::gen_sum(&mut r);
            let up = if inf { "oo" } else { "n" };
            (
                format!("summation({t}, (x, {a}, {up}))"),
                Box::new(move || calc::check_sum(&t, a, inf)),
            )
        }
        _ => {
            let o = calc::gen_ode(&mut r);
            (
                format!("dsolve({o})"),
                Box::new(move || calc::check_ode(&o)),
            )
        }
    };
    let show = std::env::var_os("FUZZ_SHOW").is_some();
    if show {
        eprintln!("fuzz_calculus: {desc}");
    }
    match run() {
        V::Bad(class, detail) => panic!("fuzz_calculus: {class} for {desc}\n  {detail}"),
        V::Ok if show => eprintln!("  verdict: ok"),
        V::Skip(why) if show => eprintln!("  verdict: skip ({why})"),
        _ => {}
    }
});
