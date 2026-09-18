//! SymPy oracle, 0.2 surface — number theory, combinatorics and diophantine
//! equations.  Exact integer answers are compared as strings; multi-valued
//! results (`sqrt_mod`, `primitive_root`, `discrete_log`) are verified by
//! membership / congruence rather than by insisting on SymPy's choice.

use super::v02_oracle_common;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, Zero};
use symplex::combinatorics;
use symplex::diophantine;
use symplex::ntheory;
use v02_oracle_common::*;

const KNOWN_BUGS: &[KnownBug] = &[];

fn want_str(fx: &Fixture, key: &str) -> Option<String> {
    fx.str(key).map(str::to_string)
}

fn exact(got: impl ToString, want: &str) -> Status {
    let g = got.to_string();
    if g == want {
        Status::Pass
    } else {
        Status::Fail(format!("symplex={g} sympy={want}"))
    }
}

#[test]
fn ntheory_factorint() {
    run_with_known_bugs("ntheory", "factorint", KNOWN_BUGS, |_ctx, fx| {
        let n = bigint(fx.str("input").unwrap_or("0"));
        let want: Vec<(String, u32)> = fx
            .field("factors")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|p| {
                        let arr = p.as_array()?;
                        Some((arr[0].as_str()?.to_string(), arr[1].as_u64()? as u32))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut got: Vec<(String, u32)> = ntheory::factorint(n.clone())
            .into_iter()
            .map(|(p, e)| (p.to_string(), e))
            .collect();
        got.sort_by_key(|(p, _)| bigint(p));
        // Every reported factor must be prime and the product must be n.
        let mut prod = BigInt::one();
        for (p, e) in &got {
            let pb = bigint(p);
            if !ntheory::isprime(pb.clone()) {
                return Status::Fail(format!("factor {p} is not prime (symplex={got:?})"));
            }
            prod *= pb.pow(*e);
        }
        if prod != n {
            return Status::Fail(format!("product of factors {got:?} != {n}"));
        }
        if got != want {
            return Status::Fail(format!("symplex={got:?} sympy={want:?}"));
        }
        Status::Pass
    });
}

#[test]
fn ntheory_isprime_pseudoprimes_and_large() {
    run_with_known_bugs("ntheory", "isprime", KNOWN_BUGS, |_ctx, fx| {
        let n = bigint(fx.str("input").unwrap_or("0"));
        let Some(want) = fx.bool("value_bool") else {
            return Status::SkippedOracle("no verdict".into());
        };
        let got = ntheory::isprime(n.clone());
        if got == want {
            Status::Pass
        } else {
            Status::Fail(format!("isprime({n}) = {got}, sympy={want}"))
        }
    });
}

#[test]
fn ntheory_sqrt_mod() {
    run_with_known_bugs("ntheory", "sqrt_mod", KNOWN_BUGS, |_ctx, fx| {
        let a = bigint(fx.str("a").unwrap_or("0"));
        let m = bigint(fx.str("modulus").unwrap_or("1"));
        let want: Vec<BigInt> = fx.str_list("roots").iter().map(|s| bigint(s)).collect();
        match ntheory::sqrt_mod(a.clone(), m.clone()) {
            Some(r) => {
                if (&r * &r - &a).mod_floor(&m) != BigInt::zero() {
                    return Status::Fail(format!("{r}^2 != {a} (mod {m})"));
                }
                if want.is_empty() {
                    return Status::Fail(format!(
                        "symplex found root {r} but SymPy says none exist"
                    ));
                }
                if !want.contains(&r) {
                    return Status::Fail(format!("root {r} not in SymPy's list {want:?}"));
                }
                Status::Pass
            }
            None => {
                if want.is_empty() {
                    Status::Pass
                } else {
                    Status::Fail(format!("symplex found no root, SymPy: {want:?}"))
                }
            }
        }
    });
}

#[test]
fn ntheory_discrete_log() {
    run_with_known_bugs("ntheory", "discrete_log", KNOWN_BUGS, |_ctx, fx| {
        let base = bigint(fx.str("base").unwrap_or("1"));
        let target = bigint(fx.str("target").unwrap_or("1"));
        let m = bigint(fx.str("modulus").unwrap_or("1"));
        let solvable = fx.bool("solvable").unwrap_or(false);
        match ntheory::discrete_log(base.clone(), target.clone(), m.clone()) {
            Some(xv) => {
                if !solvable {
                    return Status::Fail(format!("symplex found x={xv} but SymPy says unsolvable"));
                }
                let check = ntheory::mod_pow(base.clone(), xv.clone(), m.clone());
                if check != target.mod_floor(&m) {
                    return Status::Fail(format!("{base}^{xv} = {check} != {target} (mod {m})"));
                }
                Status::Pass
            }
            None => {
                if solvable {
                    Status::Fail(format!(
                        "symplex found no x, SymPy: x={}",
                        fx.str("value").unwrap_or("?")
                    ))
                } else {
                    Status::Pass
                }
            }
        }
    });
}

#[test]
fn ntheory_primitive_root() {
    run_with_known_bugs("ntheory", "primitive_root", KNOWN_BUGS, |_ctx, fx| {
        let n = bigint(fx.str("input").unwrap_or("1"));
        let want = want_str(fx, "value");
        let all: Vec<String> = fx.str_list("all_roots");
        match (ntheory::primitive_root(n.clone()), want) {
            (None, None) => Status::Pass,
            (Some(g), None) => Status::Fail(format!(
                "symplex says {g} is a primitive root mod {n}; SymPy says none exist"
            )),
            (None, Some(w)) => Status::Fail(format!("symplex found none, SymPy: {w}")),
            (Some(g), Some(w)) => {
                let gs = g.to_string();
                if gs == w || all.contains(&gs) {
                    Status::Pass
                } else if all.is_empty() {
                    // Large modulus: both should return the smallest root.
                    Status::Fail(format!("symplex={gs} sympy(smallest)={w}"))
                } else {
                    Status::Fail(format!(
                        "{gs} is not a primitive root mod {n} (valid: {all:?})"
                    ))
                }
            }
        }
    });
}

#[test]
fn ntheory_jacobi_symbol() {
    run_with_known_bugs("ntheory", "jacobi_symbol", KNOWN_BUGS, |_ctx, fx| {
        let a = bigint(fx.str("a").unwrap_or("0"));
        let n = bigint(fx.str("n").unwrap_or("1"));
        let want = fx.i64("value_int").unwrap_or(0);
        match ntheory::jacobi_symbol(a, n) {
            Ok(v) => exact(v, &want.to_string()),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

#[test]
fn ntheory_continued_fraction() {
    run_with_known_bugs("ntheory", "continued_fraction", KNOWN_BUGS, |_ctx, fx| {
        let p = bigint(fx.str("numerator").unwrap_or("0"));
        let q = bigint(fx.str("denominator").unwrap_or("1"));
        let want: Vec<i64> = fx
            .field("value_list")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
            .unwrap_or_default();
        let r = num_rational::Ratio::new(p, q);
        let got: Vec<String> = ntheory::continued_fraction(&r)
            .iter()
            .map(ToString::to_string)
            .collect();
        let want: Vec<String> = want.iter().map(ToString::to_string).collect();
        if got == want {
            Status::Pass
        } else {
            Status::Fail(format!("symplex={got:?} sympy={want:?}"))
        }
    });
}

fn unary_bigint(fx: &Fixture, f: impl Fn(BigInt) -> Option<String>) -> Status {
    let n = bigint(fx.str("input").unwrap_or("0"));
    let Some(want) = want_str(fx, "value") else {
        return Status::SkippedOracle("no value".into());
    };
    match f(n) {
        Some(g) => exact(g, &want),
        None => Status::NotImplemented("returned None".into()),
    }
}

#[test]
fn ntheory_primepi() {
    run_with_known_bugs("ntheory", "primepi", KNOWN_BUGS, |_ctx, fx| {
        unary_bigint(fx, |n| ntheory::primepi(n).map(|v| v.to_string()))
    });
}

#[test]
fn ntheory_totient() {
    run_with_known_bugs("ntheory", "totient", KNOWN_BUGS, |_ctx, fx| {
        unary_bigint(fx, |n| Some(ntheory::totient(n).to_string()))
    });
}

#[test]
fn ntheory_mobius() {
    run_with_known_bugs("ntheory", "mobius", KNOWN_BUGS, |_ctx, fx| {
        unary_bigint(fx, |n| Some(ntheory::mobius(n).to_string()))
    });
}

#[test]
fn ntheory_partition_numbers() {
    run_with_known_bugs("ntheory", "partition", KNOWN_BUGS, |_ctx, fx| {
        unary_bigint(fx, |n| {
            combinatorics::partition_count(n).map(|v| v.to_string())
        })
    });
}

#[test]
fn ntheory_bell_numbers() {
    run_with_known_bugs("ntheory", "bell", KNOWN_BUGS, |_ctx, fx| {
        unary_bigint(fx, |n| combinatorics::bell(n).map(|v| v.to_string()))
    });
}

#[test]
fn ntheory_catalan_numbers() {
    run_with_known_bugs("ntheory", "catalan", KNOWN_BUGS, |_ctx, fx| {
        unary_bigint(fx, |n| combinatorics::catalan(n).map(|v| v.to_string()))
    });
}

#[test]
fn ntheory_fibonacci() {
    run_with_known_bugs("ntheory", "fibonacci", KNOWN_BUGS, |_ctx, fx| {
        unary_bigint(fx, |n| Some(ntheory::fibonacci(n).to_string()))
    });
}

#[test]
fn ntheory_lucas() {
    run_with_known_bugs("ntheory", "lucas", KNOWN_BUGS, |_ctx, fx| {
        unary_bigint(fx, |n| Some(ntheory::lucas(n).to_string()))
    });
}

#[test]
fn ntheory_bernoulli_numbers() {
    run_with_known_bugs("ntheory", "bernoulli", KNOWN_BUGS, |_ctx, fx| {
        // B₁ is +1/2 in SymPy ≥ 1.12 and −1/2 in the classical convention
        // used by symplex; both are accepted for n = 1 only.
        if fx.str("input") == Some("1") {
            return match ntheory::bernoulli(BigInt::one()) {
                Some(r) if r.denom() == &BigInt::from(2) && r.numer().abs().is_one() => {
                    Status::Pass
                }
                Some(r) => Status::Fail(format!("B1 = {r}, expected ±1/2")),
                None => Status::NotImplemented("returned None".into()),
            };
        }
        unary_bigint(fx, |n| {
            ntheory::bernoulli(n).map(|r| {
                if r.denom().is_one() {
                    r.numer().to_string()
                } else {
                    format!("{}/{}", r.numer(), r.denom())
                }
            })
        })
    });
}

#[test]
fn ntheory_divisor_sigma() {
    run_with_known_bugs("ntheory", "divisor_sigma", KNOWN_BUGS, |_ctx, fx| {
        let n = bigint(fx.str("input").unwrap_or("0"));
        let k = fx.u64("k").unwrap_or(1) as u32;
        let Some(want) = want_str(fx, "value") else {
            return Status::SkippedOracle("no value".into());
        };
        exact(ntheory::divisor_sigma(n, k), &want)
    });
}

fn stirling(fx: &Fixture, kind: u8) -> Status {
    let n = bigint(fx.str("n").unwrap_or("0"));
    let k = bigint(fx.str("k").unwrap_or("0"));
    let Some(want) = want_str(fx, "value") else {
        return Status::SkippedOracle("no value".into());
    };
    let got = if kind == 1 {
        combinatorics::stirling1(n, k)
    } else {
        combinatorics::stirling2(n, k)
    };
    match got {
        Some(v) => exact(v, &want),
        None => Status::NotImplemented("returned None".into()),
    }
}

#[test]
fn ntheory_stirling_first_kind_signed() {
    run_with_known_bugs("ntheory", "stirling1", KNOWN_BUGS, |_ctx, fx| {
        stirling(fx, 1)
    });
}

#[test]
fn ntheory_stirling_second_kind() {
    run_with_known_bugs("ntheory", "stirling2", KNOWN_BUGS, |_ctx, fx| {
        stirling(fx, 2)
    });
}

// ── diophantine ────────────────────────────────────────────────────────

#[test]
fn diophantine_linear() {
    run_with_known_bugs("diophantine", "linear", KNOWN_BUGS, |_ctx, fx| {
        let a = bigint(fx.str("a").unwrap_or("0"));
        let b = bigint(fx.str("b").unwrap_or("0"));
        let c = bigint(fx.str("c").unwrap_or("0"));
        let solvable = fx.bool("solvable").unwrap_or(false);
        let g = fx.i64("gcd").map(BigInt::from).unwrap_or_else(|| a.gcd(&b));
        match diophantine::linear_diophantine(a.clone(), b.clone(), c.clone()) {
            Some((x0, y0, dx, dy)) => {
                if !solvable {
                    return Status::Fail(format!(
                        "symplex solved {a}x+{b}y={c} (SymPy: no solution)"
                    ));
                }
                if &a * &x0 + &b * &y0 != c {
                    return Status::Fail(format!("{a}·{x0} + {b}·{y0} != {c}"));
                }
                // The step must be the primitive direction (±b/g, ∓a/g).
                if a.is_zero() && b.is_zero() {
                    return Status::Pass;
                }
                let (sb, sa) = (&b / &g, &a / &g);
                let step_ok = (dx == sb && dy == -&sa) || (dx == -&sb && dy == sa);
                if !step_ok {
                    return Status::Fail(format!("step ({dx}, {dy}) is not ±({sb}, -{sa})"));
                }
                Status::Pass
            }
            None => {
                if solvable {
                    Status::Fail(format!(
                        "symplex found no solution of {a}x+{b}y={c}, SymPy: {}",
                        fx.str("sympy_result").unwrap_or("?")
                    ))
                } else {
                    Status::Pass
                }
            }
        }
    });
}

#[test]
fn diophantine_pell_fundamental_solution() {
    run_with_known_bugs("diophantine", "pell", KNOWN_BUGS, |_ctx, fx| {
        let d = bigint(fx.str("d").unwrap_or("2"));
        let (Some(wx), Some(wy)) = (want_str(fx, "x"), want_str(fx, "y")) else {
            return Status::SkippedOracle("no value".into());
        };
        match diophantine::pell(d.clone()) {
            Some((x, y)) => {
                if &x * &x - &d * &y * &y != BigInt::one() {
                    return Status::Fail(format!("({x}, {y}) does not satisfy x² − {d}y² = 1"));
                }
                if x.to_string() == wx && y.to_string() == wy {
                    Status::Pass
                } else {
                    Status::Fail(format!(
                        "fundamental solution: symplex=({x}, {y}) sympy=({wx}, {wy})"
                    ))
                }
            }
            None => Status::NotImplemented("pell returned None".into()),
        }
    });
}
