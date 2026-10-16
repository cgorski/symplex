//! Tests of session 15 (s15m1): ValueClass, one decider for special values.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, src: &str) -> Ex {
    ctx.parse(src).unwrap()
}

/// The powers of the special atoms are one table over their classes
/// (`canon::special_atom_power`), which says what happens at each one.  Up
/// to 0.57 `∞^zoo` and `(−∞)^zoo` were left as powers (`zoo` has no
/// direction: `|b^e| = e^{Re(e·ln b)}` has no limit, as for `zoo^∞` and
/// `0^zoo`, which were already `nan`), and `(−∞)^n` for an odd `n ≥ 2⁶³`
/// was `∞` (the parity was read from an `i64` that saturated to 0).
///
/// Oracle: SymPy 1.14 `oo**zoo` → `nan`, `(-oo)**zoo` → `nan`,
/// `(-oo)**(2**63 + 1)` → `-oo`; `(−1)^(2⁶³ + 1) = −1` exactly.  The rest
/// of the table is unchanged (`∞^∞ = ∞`, `(−∞)^∞ = nan`, `0^zoo = nan`, and
/// the exact power `1^∞ = 1`).
#[test]
fn powers_of_the_special_atoms_are_one_table() {
    let ctx = Context::new();
    for (src, value) in [
        ("oo**zoo", "nan"),
        ("(-oo)**zoo", "nan"),
        ("(-oo)**9223372036854775809", "-oo"),
        ("(-oo)**9223372036854775808", "oo"),
        ("(-oo)**3", "-oo"),
        ("oo**oo", "oo"),
        ("oo**(-oo)", "0"),
        ("(-oo)**oo", "nan"),
        ("zoo**zoo", "nan"),
        ("zoo**(-2)", "0"),
        ("0**zoo", "nan"),
        ("0**(-oo)", "zoo"),
        ("oo**0", "nan"),
        ("1**oo", "1"),
    ] {
        assert_eq!(parse(&ctx, src).to_string(), value, "{src}");
    }
    assert_eq!(parse(&ctx, "sqrt(-oo)").to_string(), "I*oo");
}

/// The folds at a special argument ask the other argument's class: a
/// finite order keeps folding, an order that is `zoo`, an infinity of
/// unknown direction or an unevaluated pole does not (these held since
/// 0.56/0.57 through `canon::finite_slot`; `ValueClass` replaces it with
/// the same decisions).
///
/// Oracle: DLMF 8.11.2 (`Γ(s, x) ~ x^{s−1}e^{−x}` → 0), 10.7.8 (`J_ν(x) ~
/// √(2/(πx))·cos(…)` → 0); DLMF 19.6.1 (`K(1) = ∞`): no finite order.
#[test]
fn folds_at_infinity_ask_the_class_of_the_other_argument() {
    let ctx = Context::new();
    for (src, value) in [
        ("uppergamma(3/2, oo)", "0"),
        ("uppergamma(s, oo)", "0"),
        ("besselj(pi, oo)", "0"),
        ("expint(log(2), oo)", "0"),
        ("lowergamma(3, oo)", "2"),
    ] {
        assert_eq!(parse(&ctx, src).eval().to_string(), value, "{src}");
    }
    for src in [
        "uppergamma(elliptic_k(besseli(0, 0)), oo)",
        "uppergamma(zoo, oo)",
        "uppergamma(y*oo, oo)",
        "besselj(elliptic_pi(pi, 1), oo)",
        "expint(I*oo, oo)",
    ] {
        let e = parse(&ctx, src);
        assert_ne!(e.eval().to_string(), "0", "{src}");
        assert_ne!(e.is_zero(), Some(true), "{src}");
    }
}
