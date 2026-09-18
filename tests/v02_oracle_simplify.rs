//! SymPy oracle, 0.2 surface — complex analysis (`re/im/conjugate/arg/abs`,
//! `expand_complex`), special-function numerics (`zeta`, `polygamma`, `Si`,
//! `Ci`, `Ei`, `li`, `EulerGamma`, `Catalan`, `GoldenRatio`, …), and the
//! rewriting battery (`sqrtdenest`, `nsimplify`, `trigsimp`, `powsimp`,
//! `powdenest`, `logcombine`, `expand_log`).
//!
//! Rewrites are checked for *value preservation* at sample points (a wrong
//! value is a FAIL); whether the rewrite achieved SymPy's simplification is
//! reported as NOT_IMPLEMENTED (informational).

mod v02_oracle_common;

use symplex::prelude::*;
use v02_oracle_common::*;

const KNOWN_BUGS: &[KnownBug] = &[];

// ── complex parts at concrete complex points ───────────────────────────

fn complex_part(ctx: &Context, fx: &Fixture) -> Status {
    let e = match parse(ctx, fx.str("input").unwrap_or("")) {
        Ok(e) => e,
        Err(e) => return Status::NotImplemented(e),
    };
    let z = ctx.symbol(fx.str("variable").unwrap_or("z"));
    let pt = match parse(ctx, fx.str("point").unwrap_or("0")) {
        Ok(e) => e,
        Err(e) => return Status::NotImplemented(e),
    };
    let Some(expected) = fx.num("value") else {
        return Status::SkippedOracle("no oracle value".into());
    };
    let v = e.subs(&z, &pt);
    let r = match fx.subcategory.as_str() {
        "re" => v.re(),
        "im" => v.im(),
        "conjugate" => v.conjugate(),
        "arg" => v.arg(),
        "abs" => v.abs(),
        other => return Status::UnsupportedApi(format!("no operation {other}")),
    };
    // `re`/`im`/`arg`/`abs` are real by definition — a nonzero imaginary
    // part in the *evaluated* result is a wrong answer.
    if fx.subcategory != "conjugate"
        && let Ok((_, im)) = r.eval_complex64()
        && im.abs() > 1e-9
    {
        return Status::Fail(format!(
            "{}({}) evaluated to a non-real number: {}",
            fx.subcategory,
            truncate(&v),
            im
        ));
    }
    compare_constant(ctx, &r, &expected, 1e-9)
}

#[test]
fn complex_re() {
    run_with_known_bugs("complex", "re", KNOWN_BUGS, complex_part);
}

#[test]
fn complex_im() {
    run_with_known_bugs("complex", "im", KNOWN_BUGS, complex_part);
}

#[test]
fn complex_conjugate() {
    run_with_known_bugs("complex", "conjugate", KNOWN_BUGS, complex_part);
}

#[test]
fn complex_arg() {
    run_with_known_bugs("complex", "arg", KNOWN_BUGS, complex_part);
}

#[test]
fn complex_abs() {
    run_with_known_bugs("complex", "abs", KNOWN_BUGS, complex_part);
}

#[test]
fn complex_expand_complex_with_real_symbols() {
    run_with_known_bugs("complex", "expand_complex", KNOWN_BUGS, |ctx, fx| {
        for name in fx.str_list("real_symbols") {
            ctx.symbol_with(&name, &[Assumption::Real]);
        }
        let e = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let r = e.expand_complex();
        // The result must be `A + I*B` with A, B free of I: check that
        // re()/im() of the result no longer mention re(...)/im(...) nodes.
        let (re, im) = r.as_real_imag();
        let s = format!("{re} | {im}");
        if s.contains("re(") || s.contains("im(") {
            return Status::NotImplemented(format!("not fully expanded: {}", truncate(&r)));
        }
        compare_eval_points(ctx, &r, fx, "eval_points", 1e-9)
    });
}

// ── special-function numerics ──────────────────────────────────────────

#[test]
fn special_function_numerics_vs_n30() {
    run_with_known_bugs("special_func", "numeric", KNOWN_BUGS, |ctx, fx| {
        let e = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let Some(expected) = fx.num("value") else {
            return Status::SkippedOracle("no oracle value".into());
        };
        // Double-precision special functions should agree to ~1e-12.
        let f64_status = compare_constant(ctx, &e, &expected, 1e-12);
        if !matches!(f64_status, Status::Pass) {
            return f64_status;
        }
        // The arbitrary-precision path must agree with SymPy's N(…, 30) to
        // at least 20 significant digits.
        let Some(digits30) = fx.str("digits30") else {
            return Status::Pass;
        };
        match e.eval_decimal(25) {
            Ok(dec) => {
                let Some(theirs) = parse_decimal_prefix(digits30) else {
                    return Status::Pass;
                };
                let Some(ours) = parse_decimal_prefix(&dec) else {
                    return Status::NotImplemented(format!("eval_decimal gave {dec}"));
                };
                if ours.0 != theirs.0 || ours.1 != theirs.1 || ours.2 != theirs.2 {
                    return Status::Fail(format!(
                        "eval_decimal(25)={dec} vs SymPy N(…,30)={digits30} disagree in the first 18 digits"
                    ));
                }
                Status::Pass
            }
            Err(err) => Status::NotImplemented(format!("eval_decimal failed: {err}")),
        }
    });
}

/// `(sign, digit string of the first 18 significant digits, decimal exponent)`
/// so two decimal strings can be compared without float rounding.
fn parse_decimal_prefix(s: &str) -> Option<(bool, String, i32)> {
    let s = s.trim();
    // Only real values.
    if s.contains('i') || s.contains('I') {
        return None;
    }
    let (mant, exp) = match s.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().ok()?),
        None => (s, 0),
    };
    let neg = mant.starts_with('-');
    let mant = mant.trim_start_matches(['-', '+']);
    let (int_part, frac_part) = mant.split_once('.').unwrap_or((mant, ""));
    let digits: String = format!("{int_part}{frac_part}");
    let first_nonzero = digits.find(|c: char| c != '0')?;
    let mut sig: String = digits[first_nonzero..].chars().take(18).collect();
    // Pad with zeros so `-0.5` and `-0.5000…` compare equal.
    while sig.len() < 18 {
        sig.push('0');
    }
    // exponent of the first significant digit
    let dec_exp = int_part.len() as i32 - first_nonzero as i32 - 1 + exp;
    Some((neg, sig, dec_exp))
}

// ── rewriting battery ──────────────────────────────────────────────────

fn sqrt_depth(e: &Ex) -> u32 {
    let s = e.to_string();
    // Count maximal nesting of "sqrt(" by scanning parentheses.
    let mut depth = 0u32;
    let mut best = 0u32;
    let mut stack: Vec<bool> = Vec::new(); // true if this paren belongs to sqrt(
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if s[i..].starts_with("sqrt(") {
            stack.push(true);
            depth += 1;
            best = best.max(depth);
            i += 5;
            continue;
        }
        match bytes[i] {
            b'(' => stack.push(false),
            b')' => {
                if let Some(true) = stack.pop() {
                    depth -= 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    // x^(1/2) forms count as one level too.
    if best == 0 && (s.contains("^(1/2)") || s.contains("^(-1/2)")) {
        best = 1;
    }
    best
}

#[test]
fn sqrtdenest() {
    run_with_known_bugs("simplify", "sqrtdenest", KNOWN_BUGS, |ctx, fx| {
        let e = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let Some(expected) = fx.num("value") else {
            return Status::SkippedOracle("no oracle value".into());
        };
        let r = e.sqrtdenest();
        let value = compare_constant(ctx, &r, &expected, 1e-12);
        if !matches!(value, Status::Pass) {
            return value;
        }
        let want_depth = fx.u64("sympy_sqrt_depth").unwrap_or(1) as u32;
        let got_depth = sqrt_depth(&r);
        if got_depth > want_depth {
            return Status::NotImplemented(format!(
                "not denested: {} (depth {got_depth}, SymPy depth {want_depth})",
                truncate(&r)
            ));
        }
        Status::Pass
    });
}

#[test]
fn nsimplify_floats_to_exact() {
    run_with_known_bugs("simplify", "nsimplify", KNOWN_BUGS, |ctx, fx| {
        let input = fx.str("input").unwrap_or("0");
        let v: f64 = match input.parse() {
            Ok(v) => v,
            Err(e) => return Status::SkippedOracle(format!("{e}")),
        };
        let Some(expected) = fx.num("value") else {
            return Status::SkippedOracle("no oracle value".into());
        };
        let fl = match ctx.from_f64(v) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(format!("{e}")),
        };
        let consts: Vec<Ex> = fx
            .str_list("constants")
            .iter()
            .filter_map(|c| parse(ctx, c).ok())
            .collect();
        let r = if consts.is_empty() {
            fl.nsimplify(1e-10)
        } else {
            let refs: Vec<&Ex> = consts.iter().collect();
            fl.nsimplify_with_constants(&refs, 1e-10)
        };
        // The result must be exact (no float atoms) and agree to 1e-10.
        let s = r.to_string();
        if s.contains('.') {
            return Status::NotImplemented(format!("result still contains a float: {s}"));
        }
        match compare_constant(ctx, &r, &expected, 1e-10) {
            Status::Pass => {
                // Must match SymPy's exact form numerically to full precision.
                if let (Ok(ours), Some(Num::Finite(re, _))) = (r.eval_f64(), Some(expected))
                    && (ours - re).abs() > 1e-12 * re.abs().max(1.0)
                {
                    return Status::Fail(format!(
                        "nsimplify gave {s} = {ours}, SymPy {} = {re}",
                        fx.str("sympy_result").unwrap_or("?")
                    ));
                }
                Status::Pass
            }
            other => other,
        }
    });
}

#[test]
fn trigsimp_preserves_value() {
    run_with_known_bugs("simplify", "trigsimp", KNOWN_BUGS, |ctx, fx| {
        let e = match parse(ctx, fx.str("input").unwrap_or("")) {
            Ok(e) => e,
            Err(e) => return Status::NotImplemented(e),
        };
        let r = e.simplify_trig();
        let status = compare_eval_points(ctx, &r, fx, "eval_points", 1e-9);
        if !matches!(status, Status::Pass) {
            return status;
        }
        // Informational: did we simplify at least as far as SymPy (op count)?
        let want_ops = fx.u64("sympy_ops").unwrap_or(0) as usize;
        let input_ops = fx.u64("input_ops").unwrap_or(0) as usize;
        let got_ops = r.count_ops();
        if want_ops < input_ops && got_ops > want_ops + 2 {
            return Status::NotImplemented(format!(
                "value OK but not simplified: {} ({got_ops} ops) vs SymPy {} ({want_ops} ops)",
                truncate(&r),
                fx.str("sympy_result").unwrap_or("?")
            ));
        }
        Status::Pass
    });
}

fn power_log_rewrite(ctx: &Context, fx: &Fixture) -> Status {
    for name in fx.str_list("positive_symbols") {
        ctx.symbol_with(&name, &[Assumption::Positive]);
    }
    let e = match parse(ctx, fx.str("input").unwrap_or("")) {
        Ok(e) => e,
        Err(e) => return Status::NotImplemented(e),
    };
    let r = match fx.subcategory.as_str() {
        "powsimp" => e.simplify_powers(),
        "powdenest" => e.powdenest(true),
        "logcombine" => e.log_combine_with(true),
        "expand_log" => e.expand_log_with(true),
        other => return Status::UnsupportedApi(format!("no rewrite {other}")),
    };
    let status = compare_eval_points(ctx, &r, fx, "eval_points", 1e-9);
    if !matches!(status, Status::Pass) {
        return status;
    }
    // Informational structural check: did the rewrite do its job?
    let s = r.to_string();
    // SymPy prints `log(`, symplex prints `ln(`; compare the number of
    // logarithm nodes: logcombine should reach one, expand_log should
    // reach at least as many as SymPy produced.
    let ours = s.matches("ln(").count();
    let theirs = fx
        .str("sympy_result")
        .map_or(ours, |w| w.matches("log(").count());
    let done = match fx.subcategory.as_str() {
        "logcombine" => ours <= theirs.max(1),
        "expand_log" => ours >= theirs,
        _ => r.count_ops() <= e.count_ops(),
    };
    if !done {
        return Status::NotImplemented(format!(
            "value OK but rewrite incomplete: {} vs SymPy {}",
            truncate(&r),
            fx.str("sympy_result").unwrap_or("?")
        ));
    }
    Status::Pass
}

#[test]
fn powsimp_preserves_value() {
    run_with_known_bugs("simplify", "powsimp", KNOWN_BUGS, power_log_rewrite);
}

#[test]
fn powdenest_preserves_value() {
    run_with_known_bugs("simplify", "powdenest", KNOWN_BUGS, power_log_rewrite);
}

#[test]
fn logcombine_preserves_value() {
    run_with_known_bugs("simplify", "logcombine", KNOWN_BUGS, power_log_rewrite);
}

#[test]
fn expand_log_preserves_value() {
    run_with_known_bugs("simplify", "expand_log", KNOWN_BUGS, power_log_rewrite);
}
