//! SymPy oracle, 0.2 surface — set algebra on intervals (membership at
//! sample points after `simplify()`), `reduce_inequalities` (membership of
//! the solution set at sample points vs. direct evaluation of the
//! conditions), and boolean logic (`simplify`, `to_cnf`, `satisfiable`
//! compared against brute-force truth tables over 4 variables).

mod v02_oracle_common;

use serde_json::Value;
use symplex::prelude::*;
use v02_oracle_common::*;

/// Library bugs surfaced by this file (strict xfail — see common module).
const KNOWN_BUGS: &[KnownBug] = &[];

/// Regression: `reduce_inequalities([1/x > 2], x)` used to return
/// `EmptySet` (the pole of `1/x` was not a sign-change point).
#[test]
fn bug_reduce_inequalities_reciprocal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cond = (1 / &x).gt(&ctx.int(2));
    let sol = reduce_inequalities(&[cond], &x).unwrap();
    assert_eq!(sol.contains(&ctx.rational(1, 4)), Some(true), "got {sol}");
    assert_eq!(sol.contains(&ctx.int(1)), Some(false), "got {sol}");
}

/// Regression: rational inequality used to lose the branch where both
/// numerator and denominator are negative.
#[test]
fn bug_rational_inequality_drops_negative_branch() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sol = ((&x - 1) / (&x + 1)).solve_ge(&x);
    assert_eq!(sol.contains(&ctx.int(-5)), Some(true), "got {sol}");
    assert_eq!(sol.contains(&ctx.int(0)), Some(false), "got {sol}");
}

/// Regression: `x/(x-2) < 1` used to be reported as true for all reals.
#[test]
fn bug_rational_inequality_returns_all_reals() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sol = (&x / (&x - 2) - 1).solve_lt(&x);
    assert_eq!(sol.contains(&ctx.int(3)), Some(false), "got {sol}"); // 3/(3-2) = 3 > 1
    assert_eq!(sol.contains(&ctx.int(0)), Some(true), "got {sol}");
}

/// Regression: `sqrt(x) < 2` must respect the domain of sqrt.
#[test]
fn bug_sqrt_inequality_ignores_domain() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sol = (x.sqrt() - 2).solve_lt(&x);
    assert_eq!(sol.contains(&ctx.int(-1)), Some(false), "got {sol}");
    assert_eq!(sol.contains(&ctx.int(1)), Some(true), "got {sol}");
}

// ── sets ───────────────────────────────────────────────────────────────

fn build_set(ctx: &Context, tree: &Value) -> Result<SetEx, String> {
    let obj = tree.as_object().ok_or("set node is not an object")?;
    if let Some(iv) = obj.get("interval").and_then(Value::as_array) {
        let lo = parse(ctx, iv[0].as_str().ok_or("bad lo")?)?;
        let hi = parse(ctx, iv[1].as_str().ok_or("bad hi")?)?;
        return Ok(ctx.interval(
            &lo,
            &hi,
            iv[2].as_bool().unwrap_or(false),
            iv[3].as_bool().unwrap_or(false),
        ));
    }
    if let Some(els) = obj.get("finite").and_then(Value::as_array) {
        let elems: Result<Vec<Ex>, String> = els
            .iter()
            .map(|e| parse(ctx, e.as_str().unwrap_or("0")))
            .collect();
        return Ok(ctx.finite_set(&elems?));
    }
    if obj.contains_key("reals") {
        return Ok(ctx.reals());
    }
    if obj.contains_key("empty") {
        return Ok(ctx.empty_set());
    }
    let args: Result<Vec<SetEx>, String> = obj
        .get("args")
        .and_then(Value::as_array)
        .ok_or("no args")?
        .iter()
        .map(|a| build_set(ctx, a))
        .collect();
    let args = args?;
    let op = obj.get("op").and_then(Value::as_str).ok_or("no op")?;
    match op {
        "union" => Ok(args[1..]
            .iter()
            .fold(args[0].clone(), |acc, s| acc.union(s))),
        "intersection" => Ok(args[1..]
            .iter()
            .fold(args[0].clone(), |acc, s| acc.intersection(s))),
        "complement" => Ok(args[0].complement(&args[1])),
        "symmetric_difference" => Ok(args[0].symmetric_difference(&args[1])),
        other => Err(format!("unknown set op {other}")),
    }
}

fn membership_check(ctx: &Context, set: &SetEx, fx: &Fixture) -> Status {
    let pts = fx.str_list("sample_points");
    let want: Vec<bool> = fx
        .field("membership")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_bool).collect())
        .unwrap_or_default();
    if pts.len() != want.len() || pts.is_empty() {
        return Status::SkippedOracle("no membership data".into());
    }
    let mut mismatches = Vec::new();
    let mut undecided = Vec::new();
    for (p, w) in pts.iter().zip(&want) {
        let pe = match parse(ctx, p) {
            Ok(e) => e,
            Err(e) => return Status::SkippedOracle(e),
        };
        match set.contains(&pe) {
            Some(g) if g == *w => {}
            Some(g) => mismatches.push(format!("{p}: symplex={g} sympy={w}")),
            None => undecided.push(p.clone()),
        }
    }
    if !mismatches.is_empty() {
        return Status::Fail(format!(
            "set {}: {}",
            truncate(&set.as_ex()),
            mismatches.join(", ")
        ));
    }
    if !undecided.is_empty() {
        return Status::NotImplemented(format!(
            "membership undecided at {:?} for {}",
            undecided,
            truncate(&set.as_ex())
        ));
    }
    Status::Pass
}

#[test]
fn sets_interval_operations_membership() {
    run_with_known_bugs("sets", "interval_ops", KNOWN_BUGS, |ctx, fx| {
        let Some(tree) = fx.field("set") else {
            return Status::SkippedOracle("no set".into());
        };
        let set = match build_set(ctx, tree) {
            Ok(s) => s,
            Err(e) => return Status::NotImplemented(e),
        };
        let simplified = set.simplify();
        membership_check(ctx, &simplified, fx)
    });
}

// ── inequalities ───────────────────────────────────────────────────────

#[test]
fn inequalities_reduce_membership() {
    run_with_known_bugs("inequalities", "reduce", KNOWN_BUGS, |ctx, fx| {
        let x = ctx.symbol(fx.str("variable").unwrap_or("x"));
        let mut conds: Vec<BoolEx> = Vec::new();
        for c in fx
            .field("conditions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let (Some(lhs), Some(op), Some(rhs)) = (
                c.get("lhs").and_then(Value::as_str),
                c.get("op").and_then(Value::as_str),
                c.get("rhs").and_then(Value::as_str),
            ) else {
                return Status::SkippedOracle("bad condition".into());
            };
            let (l, r) = match (parse(ctx, lhs), parse(ctx, rhs)) {
                (Ok(l), Ok(r)) => (l, r),
                (Err(e), _) | (_, Err(e)) => return Status::NotImplemented(e),
            };
            conds.push(match op {
                ">" => l.gt(&r),
                ">=" => l.ge(&r),
                "<" => l.lt(&r),
                "<=" => l.le(&r),
                "=" => l.eq_expr(&r),
                "!=" => l.ne_expr(&r),
                other => return Status::UnsupportedApi(format!("relation {other}")),
            });
        }
        match reduce_inequalities(&conds, &x) {
            Ok(set) => membership_check(ctx, &set, fx),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

// ── boolean logic ──────────────────────────────────────────────────────

fn build_bool(tree: &Value, atoms: &[(String, BoolEx)]) -> Result<BoolEx, String> {
    let obj = tree.as_object().ok_or("bool node is not an object")?;
    if let Some(v) = obj.get("var").and_then(Value::as_str) {
        return atoms
            .iter()
            .find(|(n, _)| n == v)
            .map(|(_, a)| a.clone())
            .ok_or_else(|| format!("unknown variable {v}"));
    }
    let args: Result<Vec<BoolEx>, String> = obj
        .get("args")
        .and_then(Value::as_array)
        .ok_or("no args")?
        .iter()
        .map(|a| build_bool(a, atoms))
        .collect();
    let args = args?;
    match obj.get("op").and_then(Value::as_str).ok_or("no op")? {
        "not" => Ok(args[0].not()),
        "and" => Ok(args[1..].iter().fold(args[0].clone(), |acc, b| acc.and(b))),
        "or" => Ok(args[1..].iter().fold(args[0].clone(), |acc, b| acc.or(b))),
        "xor" => Ok(args[0].xor(&args[1])),
        "implies" => Ok(args[0].implies(&args[1])),
        "equivalent" => Ok(args[0].equivalent(&args[1])),
        other => Err(format!("unknown bool op {other}")),
    }
}

/// Truth table as a 16-entry vector (p most significant), via symplex.
fn truth_vector(f: &BoolEx, atoms: &[(String, BoolEx)]) -> Result<Vec<bool>, String> {
    let vars: Vec<BoolEx> = atoms.iter().map(|(_, a)| a.clone()).collect();
    let rows = f.truth_table(&vars).map_err(|e| e.to_string())?;
    if rows.len() != 1 << vars.len() {
        return Err(format!("truth_table returned {} rows", rows.len()));
    }
    let mut out = vec![false; rows.len()];
    for (bits, val) in rows {
        let mut idx = 0usize;
        for b in &bits {
            idx = (idx << 1) | usize::from(*b);
        }
        out[idx] = val;
    }
    Ok(out)
}

#[test]
fn logic_boolean_normal_forms_and_satisfiability() {
    run_with_known_bugs("logic", "boolean", KNOWN_BUGS, |ctx, fx| {
        let names = fx.str_list("variables");
        let atoms: Vec<(String, BoolEx)> = names
            .iter()
            .map(|n| (n.clone(), ctx.symbol(n).gt(&ctx.int(0))))
            .collect();
        let Some(tree) = fx.field("formula") else {
            return Status::SkippedOracle("no formula".into());
        };
        let f = match build_bool(tree, &atoms) {
            Ok(f) => f,
            Err(e) => return Status::NotImplemented(e),
        };
        let want: Vec<bool> = fx
            .field("truth_table")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_bool).collect())
            .unwrap_or_default();
        if want.len() != 16 {
            return Status::SkippedOracle("bad truth table".into());
        }
        // 1. The formula itself.
        match truth_vector(&f, &atoms) {
            Ok(got) if got == want => {}
            Ok(got) => {
                return Status::Fail(format!(
                    "truth table of the formula differs: symplex={got:?} sympy={want:?}"
                ));
            }
            Err(e) => return Status::NotImplemented(format!("truth_table: {e}")),
        }
        // 2. simplify() and to_cnf() must be equivalent to the formula.
        for (name, g) in [("simplify", f.simplify()), ("to_cnf", f.to_cnf())] {
            match truth_vector(&g, &atoms) {
                Ok(got) if got == want => {}
                Ok(got) => {
                    return Status::Fail(format!(
                        "{name} changed the truth table: {} → {got:?} vs {want:?}",
                        truncate(&g.as_ex())
                    ));
                }
                Err(e) => {
                    // Constant results (True/False) have no variables: check directly.
                    let s = g.to_string();
                    let constant = if s == "True" {
                        Some(true)
                    } else if s == "False" {
                        Some(false)
                    } else {
                        None
                    };
                    match constant {
                        Some(c) if want.iter().all(|w| *w == c) => {}
                        Some(c) => {
                            return Status::Fail(format!(
                                "{name} gave constant {c} but the table is {want:?}"
                            ));
                        }
                        None => {
                            return Status::NotImplemented(format!(
                                "{name}: truth_table failed on {}: {e}",
                                truncate(&g.as_ex())
                            ));
                        }
                    }
                }
            }
        }
        // 3. CNF shape: conjunction of disjunctions of literals.
        let cnf = f.to_cnf();
        let cnf_s = cnf.to_string();
        if cnf_s.contains(" & ") && cnf_s.contains(" | ") {
            // every '|' group must be parenthesised inside '&'; a '&' inside
            // parentheses would violate CNF.  Check no '(' … '&' … ')' nesting.
            let mut depth = 0;
            for ch in cnf_s.chars() {
                match ch {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    '&' if depth > 0 => {
                        return Status::Fail(format!("to_cnf result is not in CNF: {cnf_s}"));
                    }
                    _ => {}
                }
            }
        }
        // 4. satisfiable / tautology.
        let want_sat = fx.bool("satisfiable").unwrap_or(true);
        match f.satisfiable() {
            Some(s) if s == want_sat => {}
            Some(s) => return Status::Fail(format!("satisfiable={s}, expected {want_sat}")),
            None => return Status::NotImplemented("satisfiable undecided".into()),
        }
        let want_taut = fx.bool("tautology").unwrap_or(false);
        match f.is_tautology() {
            Some(t) if t == want_taut => Status::Pass,
            Some(t) => Status::Fail(format!("is_tautology={t}, expected {want_taut}")),
            None => Status::NotImplemented("is_tautology undecided".into()),
        }
    });
}
