//! The value dump of the nightly statistics oracle
//! (`scripts/stats_oracle/README.md`, `.github/workflows/stats-oracle.yml`).
//!
//! `stats_oracle_dump` reads the case rows of `scripts/stats_oracle/cases/*.tsv`
//! (`id, kind, target, params, method, arg`), computes symplex's values and
//! writes one line per value: `id <TAB> output <TAB> value <TAB> ms`.  It runs
//! only when `STATS_ORACLE_CASES` (a case file or a directory of them) and
//! `STATS_ORACLE_OUT` are set — `scripts/stats_oracle/run_dump.py` sets them,
//! shards the rows over processes (`STATS_ORACLE_SHARD=i/n`) and resumes
//! after a row that hangs (`STATS_ORACLE_START`); without them it runs a few
//! built-in rows and checks their values, so every `cargo test` keeps the
//! dumper working.
//!
//! Values: `D:<digits>` (`eval_decimal(30)`, every digit certified; a complex
//! value as `a+b*i`), `F:<f64>` (an `f64` route), `C:<re>,<im>`
//! (`eval_complex64`), `Q:<n>/<d>` (an exact rational), `INF`, `-INF`, `NAN`
//! (the exact value is that symbol), `NONE` (the method has no answer by
//! design, e.g. no closed-form quantile), `OFF` (a density asked off the
//! support, where it is 0 by definition), `ERR:<variant>:<message>`,
//! `PANIC:<message>`, `HANG`.

use std::io::Write;
use std::str::FromStr;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use symplex::linprog::Q;
use symplex::prelude::*;
use symplex::stats::Distribution;
use symplex::stats::anova::anova_one_way;
use symplex::stats::hypothesis::{self as hyp, Alternative, RankMethod, TestResult};
use symplex::stats::numdist;
use symplex::stats::regression::ols;

/// Digits of every certified decimal in the dump.
const DIGITS: u32 = 30;

/// One case row.
#[derive(Clone, Debug)]
struct Row {
    id: String,
    kind: String,
    target: String,
    params: String,
    method: String,
    arg: String,
}

impl Row {
    fn parse(line: &str) -> Option<Row> {
        if line.starts_with('#') || line.trim().is_empty() {
            return None;
        }
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() != 6 {
            return Some(Row {
                id: c.first().copied().unwrap_or("?").to_string(),
                kind: "malformed".into(),
                target: String::new(),
                params: String::new(),
                method: String::new(),
                arg: String::new(),
            });
        }
        Some(Row {
            id: c[0].into(),
            kind: c[1].into(),
            target: c[2].into(),
            params: c[3].into(),
            method: c[4].into(),
            arg: c[5].into(),
        })
    }
}

/// `(output name, value)` pairs of one row.
type Outputs = Vec<(String, String)>;

fn err_value(e: &SymplexError) -> String {
    let debug = format!("{e:?}");
    let variant: String = debug.chars().take_while(|c| c.is_alphanumeric()).collect();
    let msg: String = e
        .to_string()
        .replace(['\n', '\t'], " ")
        .chars()
        .take(160)
        .collect();
    format!("ERR:{variant}:{msg}")
}

fn special(e: &Ex) -> Option<&'static str> {
    let ctx = e.context();
    if *e == ctx.infinity() {
        Some("INF")
    } else if *e == ctx.neg_infinity() {
        Some("-INF")
    } else if *e == ctx.nan() {
        Some("NAN")
    } else {
        None
    }
}

fn decimal(e: &Ex) -> String {
    if let Some(s) = special(e) {
        return s.into();
    }
    match e.eval_decimal(DIGITS) {
        Ok(s) => format!("D:{}", s.replace(' ', "")),
        Err(err) => err_value(&err),
    }
}

fn float(e: &Ex) -> String {
    if let Some(s) = special(e) {
        return s.into();
    }
    match e.eval_f64() {
        Ok(v) => format!("F:{v:e}"),
        Err(err) => err_value(&err),
    }
}

fn complex(e: &Ex) -> String {
    if let Some(s) = special(e) {
        return s.into();
    }
    match e.eval_complex64() {
        Ok(z) => format!("C:{:e},{:e}", z.re, z.im),
        Err(err) => err_value(&err),
    }
}

fn f64_result(r: Result<f64, SymplexError>) -> String {
    match r {
        Ok(v) => format!("F:{v:e}"),
        Err(e) => err_value(&e),
    }
}

fn exact(v: &Q) -> String {
    format!("Q:{}/{}", v.numer(), v.denom())
}

fn rational(s: &str) -> Result<Q, String> {
    Q::from_str(s.trim()).map_err(|e| format!("bad rational {s:?}: {e}"))
}

fn rationals(s: &str) -> Result<Vec<Q>, String> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',').map(rational).collect()
}

fn ex_of(ctx: &Context, s: &str) -> Result<Ex, String> {
    ctx.parse(s.trim())
        .map_err(|e| format!("bad number {s:?}: {e}"))
}

fn distribution(ctx: &Context, family: &str, params: &str) -> Result<Distribution, String> {
    let built = if family == "finite" {
        let mut table = Vec::new();
        for item in params.split(',') {
            let (v, p) = item
                .split_once(':')
                .ok_or_else(|| format!("bad finite atom {item:?}"))?;
            table.push((ex_of(ctx, v)?, ex_of(ctx, p)?));
        }
        Distribution::try_finite(ctx, table)
    } else {
        let p: Vec<Ex> = params
            .split(',')
            .map(|s| ex_of(ctx, s))
            .collect::<Result<_, _>>()?;
        let arity = |n: usize| {
            if p.len() == n {
                Ok(())
            } else {
                Err(format!("{family} takes {n} parameters, got {}", p.len()))
            }
        };
        let c = |i: usize| p[i].clone();
        match family {
            "normal" => arity(2).map(|()| Distribution::try_normal(c(0), c(1)))?,
            "uniform" => arity(2).map(|()| Distribution::try_uniform(c(0), c(1)))?,
            "exponential" => arity(1).map(|()| Distribution::try_exponential(c(0)))?,
            "gamma" => arity(2).map(|()| Distribution::try_gamma(c(0), c(1)))?,
            "chi_squared" => arity(1).map(|()| Distribution::try_chi_squared(c(0)))?,
            "beta" => arity(2).map(|()| Distribution::try_beta(c(0), c(1)))?,
            "cauchy" => arity(2).map(|()| Distribution::try_cauchy(c(0), c(1)))?,
            "laplace" => arity(2).map(|()| Distribution::try_laplace(c(0), c(1)))?,
            "logistic" => arity(2).map(|()| Distribution::try_logistic(c(0), c(1)))?,
            "log_normal" => arity(2).map(|()| Distribution::try_log_normal(c(0), c(1)))?,
            "student_t" => arity(1).map(|()| Distribution::try_student_t(c(0)))?,
            "f" => arity(2).map(|()| Distribution::try_f_distribution(c(0), c(1)))?,
            "weibull" => arity(2).map(|()| Distribution::try_weibull(c(0), c(1)))?,
            "pareto" => arity(2).map(|()| Distribution::try_pareto(c(0), c(1)))?,
            "triangular" => arity(3).map(|()| Distribution::try_triangular(c(0), c(1), c(2)))?,
            "bernoulli" => arity(1).map(|()| Distribution::try_bernoulli(c(0)))?,
            "binomial" => arity(2).map(|()| Distribution::try_binomial(c(0), c(1)))?,
            "poisson" => arity(1).map(|()| Distribution::try_poisson(c(0)))?,
            "geometric" => arity(1).map(|()| Distribution::try_geometric(c(0)))?,
            "negative_binomial" => {
                arity(2).map(|()| Distribution::try_negative_binomial(c(0), c(1)))?
            }
            "hypergeometric" => {
                arity(3).map(|()| Distribution::try_hypergeometric(c(0), c(1), c(2)))?
            }
            "discrete_uniform" => {
                arity(2).map(|()| Distribution::try_discrete_uniform(c(0), c(1)))?
            }
            _ => return Err(format!("unknown family {family:?}")),
        }
    };
    built.map_err(|e| err_value(&e))
}

/// An exact value read two ways: certified digits and the `f64` route.
fn both(e: &Ex) -> Outputs {
    vec![("d30".into(), decimal(e)), ("f64".into(), float(e))]
}

fn dist_row(r: &Row) -> Result<Outputs, String> {
    let ctx = Context::new();
    let d = match distribution(&ctx, &r.target, &r.params) {
        Ok(d) => d,
        // An invalid parameter must be refused cleanly; the refusal is the value.
        Err(e) if e.starts_with("ERR:") => return Ok(vec![("d30".into(), e)]),
        Err(e) => return Err(e),
    };
    let arg = || ex_of(&ctx, &r.arg);
    let level = || {
        r.arg
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("bad level {:?}: {e}", r.arg))
    };
    Ok(match r.method.as_str() {
        // `density` is the formula on the support (its documented contract);
        // off the support the density is 0 by definition, not a claim.
        "pdf" => {
            let x = arg()?;
            match d.support().contains(&x) {
                Some(true) => both(&d.density(&x)),
                Some(false) => vec![("d30".into(), "OFF".into())],
                None => vec![("d30".into(), "NONE".into())],
            }
        }
        "cdf" => both(&d.cdf(&arg()?)),
        "sf" => both(&d.sf(&arg()?)),
        "quantile" => match d.quantile(&arg()?) {
            Some(e) => both(&e),
            None => vec![("d30".into(), "NONE".into())],
        },
        "median" => match d.median() {
            Some(e) => both(&e),
            None => vec![("d30".into(), "NONE".into())],
        },
        "quantile_f64" => vec![("f64".into(), f64_result(d.quantile_f64(level()?)))],
        "isf_f64" => vec![("f64".into(), f64_result(d.isf_f64(level()?)))],
        "mean" => both(&d.mean()),
        "variance" => both(&d.variance()),
        "skewness" => both(&d.skewness()),
        "kurtosis" => both(&d.kurtosis()),
        "entropy" => both(&d.entropy()),
        "mgf" => both(&d.mgf(&arg()?)),
        "cf" => {
            let e = d.characteristic_function(&arg()?);
            vec![("d30".into(), decimal(&e)), ("c64".into(), complex(&e))]
        }
        m => return Err(format!("unknown method {m:?}")),
    })
}

fn kernel_row(r: &Row) -> Result<Outputs, String> {
    let a: Vec<f64> = r
        .params
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|e| format!("bad f64 {s:?}: {e}"))
        })
        .collect::<Result<_, _>>()?;
    let need = |n: usize| {
        if a.len() == n {
            Ok(())
        } else {
            Err(format!("{} takes {n} arguments, got {}", r.target, a.len()))
        }
    };
    use numdist::{beta, binom, chi2, f, gamma, norm, poisson, t};
    let v: Result<f64, SymplexError> = match r.target.as_str() {
        "betainc_regularized" => {
            need(3).map(|()| Ok(numdist::betainc_regularized_f64(a[0], a[1], a[2])))?
        }
        "gammainc_lower_regularized" => {
            need(2).map(|()| Ok(numdist::gammainc_lower_regularized_f64(a[0], a[1])))?
        }
        "gammainc_upper_regularized" => {
            need(2).map(|()| Ok(numdist::gammainc_upper_regularized_f64(a[0], a[1])))?
        }
        "norm.cdf" => need(1).map(|()| Ok(norm::cdf(a[0])))?,
        "norm.sf" => need(1).map(|()| Ok(norm::sf(a[0])))?,
        "norm.pdf" => need(1).map(|()| Ok(norm::pdf(a[0])))?,
        "norm.ppf" => need(1).map(|()| norm::ppf(a[0]))?,
        "norm.isf" => need(1).map(|()| norm::isf(a[0]))?,
        "t.cdf" => need(2).map(|()| Ok(t::cdf(a[0], a[1])))?,
        "t.sf" => need(2).map(|()| Ok(t::sf(a[0], a[1])))?,
        "t.pdf" => need(2).map(|()| Ok(t::pdf(a[0], a[1])))?,
        "t.ppf" => need(2).map(|()| t::ppf(a[0], a[1]))?,
        "t.isf" => need(2).map(|()| t::isf(a[0], a[1]))?,
        "gamma.cdf" => need(3).map(|()| Ok(gamma::cdf(a[0], a[1], a[2])))?,
        "gamma.sf" => need(3).map(|()| Ok(gamma::sf(a[0], a[1], a[2])))?,
        "gamma.ppf" => need(3).map(|()| gamma::ppf(a[0], a[1], a[2]))?,
        "gamma.isf" => need(3).map(|()| gamma::isf(a[0], a[1], a[2]))?,
        "chi2.cdf" => need(2).map(|()| Ok(chi2::cdf(a[0], a[1])))?,
        "chi2.sf" => need(2).map(|()| Ok(chi2::sf(a[0], a[1])))?,
        "chi2.ppf" => need(2).map(|()| chi2::ppf(a[0], a[1]))?,
        "chi2.isf" => need(2).map(|()| chi2::isf(a[0], a[1]))?,
        "beta.cdf" => need(3).map(|()| Ok(beta::cdf(a[0], a[1], a[2])))?,
        "beta.sf" => need(3).map(|()| Ok(beta::sf(a[0], a[1], a[2])))?,
        "beta.ppf" => need(3).map(|()| beta::ppf(a[0], a[1], a[2]))?,
        "beta.isf" => need(3).map(|()| beta::isf(a[0], a[1], a[2]))?,
        "f.cdf" => need(3).map(|()| Ok(f::cdf(a[0], a[1], a[2])))?,
        "f.sf" => need(3).map(|()| Ok(f::sf(a[0], a[1], a[2])))?,
        "f.ppf" => need(3).map(|()| f::ppf(a[0], a[1], a[2]))?,
        "f.isf" => need(3).map(|()| f::isf(a[0], a[1], a[2]))?,
        "binom.cdf" => need(3).map(|()| Ok(binom::cdf(a[0], a[1], a[2])))?,
        "binom.sf" => need(3).map(|()| Ok(binom::sf(a[0], a[1], a[2])))?,
        "binom.ppf" => need(3).map(|()| binom::ppf(a[0], a[1], a[2]))?,
        "binom.isf" => need(3).map(|()| binom::isf(a[0], a[1], a[2]))?,
        "poisson.cdf" => need(2).map(|()| Ok(poisson::cdf(a[0], a[1])))?,
        "poisson.sf" => need(2).map(|()| Ok(poisson::sf(a[0], a[1])))?,
        "poisson.ppf" => need(2).map(|()| poisson::ppf(a[0], a[1]))?,
        "poisson.isf" => need(2).map(|()| poisson::isf(a[0], a[1]))?,
        k => return Err(format!("unknown kernel {k:?}")),
    };
    Ok(vec![("f64".into(), f64_result(v))])
}

/// `key=value;…` of a test or model row.
fn fields(params: &str) -> Vec<(&str, &str)> {
    params
        .split(';')
        .filter(|s| !s.is_empty())
        .filter_map(|kv| kv.split_once('='))
        .collect()
}

fn field<'a>(f: &[(&str, &'a str)], key: &str) -> Result<&'a str, String> {
    f.iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| *v)
        .ok_or_else(|| format!("missing field {key:?}"))
}

fn alternative(s: &str) -> Result<Alternative, String> {
    match s {
        "two-sided" => Ok(Alternative::TwoSided),
        "less" => Ok(Alternative::Less),
        "greater" => Ok(Alternative::Greater),
        _ => Err(format!("bad alternative {s:?}")),
    }
}

fn flag(s: &str) -> Result<bool, String> {
    match s {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("bad flag {s:?}")),
    }
}

fn count(s: &str) -> Result<usize, String> {
    s.trim()
        .parse::<usize>()
        .map_err(|e| format!("bad count {s:?}: {e}"))
}

/// Rows separated by `:`, entries by `,`.
fn table(s: &str) -> Result<Vec<Vec<Q>>, String> {
    s.split(':').map(rationals).collect()
}

fn test_outputs(r: Result<TestResult, SymplexError>) -> Outputs {
    match r {
        Ok(t) => {
            let mut out = vec![
                ("statistic".into(), decimal(&t.statistic)),
                ("pvalue".into(), decimal(&t.p_value)),
                ("pvalue_f64".into(), f64_result(t.p_value_f64())),
            ];
            if let Some(df) = &t.df {
                out.push(("df".into(), decimal(df)));
            }
            out
        }
        Err(e) => vec![("statistic".into(), err_value(&e))],
    }
}

fn test_row(r: &Row) -> Result<Outputs, String> {
    let ctx = Context::new();
    let f = fields(&r.params);
    let alt = || alternative(field(&f, "alt")?);
    let xs = |k: &str| rationals(field(&f, k)?);
    Ok(match r.target.as_str() {
        "t_test_one_sample" => test_outputs(hyp::t_test_one_sample(
            &ctx,
            &xs("x")?,
            &rational(field(&f, "mu0")?)?,
            alt()?,
        )),
        "t_test_two_sample" => test_outputs(hyp::t_test_two_sample(
            &ctx,
            &xs("x")?,
            &xs("y")?,
            flag(field(&f, "equal_var")?)?,
            alt()?,
        )),
        "t_test_paired" => test_outputs(hyp::t_test_paired(&ctx, &xs("x")?, &xs("y")?, alt()?)),
        "binomial_test" => test_outputs(hyp::binomial_test(
            &ctx,
            count(field(&f, "k")?)?,
            count(field(&f, "n")?)?,
            &rational(field(&f, "p0")?)?,
            alt()?,
        )),
        "fisher_exact" => {
            let t = table(field(&f, "table")?)?;
            let c = |i: usize, j: usize| -> Result<usize, String> {
                t.get(i).and_then(|row| row.get(j)).map_or_else(
                    || Err("fisher_exact needs a 2x2 table".to_string()),
                    |v| count(&v.to_string()),
                )
            };
            test_outputs(hyp::fisher_exact(
                &ctx,
                [[c(0, 0)?, c(0, 1)?], [c(1, 0)?, c(1, 1)?]],
                alt()?,
            ))
        }
        "chi_square_independence" | "chi_square_goodness_of_fit" => {
            let res = if r.target == "chi_square_independence" {
                hyp::chi_square_independence(
                    &ctx,
                    &table(field(&f, "table")?)?,
                    flag(field(&f, "correction")?)?,
                )
            } else {
                hyp::chi_square_goodness_of_fit(&ctx, &xs("observed")?, None, 0)
            };
            match res {
                Ok(c) => vec![
                    ("statistic".into(), exact(&c.statistic)),
                    ("pvalue".into(), decimal(&c.p_value)),
                    ("pvalue_f64".into(), f64_result(c.p_value_f64())),
                    ("df".into(), format!("Q:{}/1", c.df)),
                ],
                Err(e) => vec![("statistic".into(), err_value(&e))],
            }
        }
        "z_test_proportion" => test_outputs(hyp::z_test_proportion(
            &ctx,
            count(field(&f, "k")?)?,
            count(field(&f, "n")?)?,
            &rational(field(&f, "p0")?)?,
            alt()?,
        )),
        "pearson_test" => test_outputs(hyp::pearson_test(&ctx, &xs("x")?, &xs("y")?, alt()?)),
        "spearman_test" => test_outputs(hyp::spearman_test(&ctx, &xs("x")?, &xs("y")?, alt()?)),
        "mann_whitney_u" => {
            let method = match field(&f, "method")? {
                "exact" => RankMethod::Exact,
                "asymptotic" => RankMethod::Asymptotic { continuity: false },
                "asymptotic_cc" => RankMethod::Asymptotic { continuity: true },
                m => return Err(format!("bad method {m:?}")),
            };
            test_outputs(hyp::mann_whitney_u(
                &ctx,
                &xs("x")?,
                &xs("y")?,
                alt()?,
                method,
            ))
        }
        "wilcoxon_signed_rank" => {
            let method = match field(&f, "method")? {
                "exact" => RankMethod::Exact,
                "asymptotic" => RankMethod::Asymptotic { continuity: false },
                "asymptotic_cc" => RankMethod::Asymptotic { continuity: true },
                m => return Err(format!("bad method {m:?}")),
            };
            test_outputs(hyp::wilcoxon_signed_rank(
                &ctx,
                &xs("x")?,
                None,
                alt()?,
                method,
            ))
        }
        "kruskal_wallis" => test_outputs(hyp::kruskal_wallis(&ctx, &table(field(&f, "groups")?)?)),
        "anova_one_way" => match anova_one_way(&ctx, &table(field(&f, "groups")?)?) {
            Ok(a) => vec![
                ("statistic".into(), exact(&a.f)),
                ("pvalue".into(), decimal(&a.p_value)),
                ("pvalue_f64".into(), f64_result(a.p_value_f64())),
            ],
            Err(e) => vec![("statistic".into(), err_value(&e))],
        },
        t => return Err(format!("unknown test {t:?}")),
    })
}

fn model_row(r: &Row) -> Result<Outputs, String> {
    if r.target != "ols" {
        return Err(format!("unknown model {:?}", r.target));
    }
    let ctx = Context::new();
    let f = fields(&r.params);
    let y = rationals(field(&f, "y")?)?;
    let mut cols = Vec::new();
    for k in 1.. {
        match f.iter().find(|(key, _)| *key == format!("x{k}")) {
            Some((_, v)) => cols.push(rationals(v)?),
            None => break,
        }
    }
    if cols.iter().any(|c| c.len() != y.len()) {
        return Err("every column needs one entry per observation".into());
    }
    let x: Vec<Vec<Q>> = (0..y.len())
        .map(|i| cols.iter().map(|c| c[i].clone()).collect())
        .collect();
    let fit = match ols(&y, &x, flag(field(&f, "intercept")?)?) {
        Ok(fit) => fit,
        Err(e) => return Ok(vec![("coef.0".into(), err_value(&e))]),
    };
    let mut out: Outputs = Vec::new();
    for (i, b) in fit.coefficients.iter().enumerate() {
        out.push((format!("coef.{i}"), exact(b)));
    }
    for (i, se) in fit.standard_errors(&ctx).iter().enumerate() {
        out.push((format!("se.{i}"), decimal(se)));
    }
    match fit.t_statistics(&ctx) {
        Ok(ts) => out.extend(
            ts.iter()
                .enumerate()
                .map(|(i, t)| (format!("t.{i}"), decimal(t))),
        ),
        Err(e) => out.push(("t.0".into(), err_value(&e))),
    }
    match fit.p_values(&ctx) {
        Ok(ps) => out.extend(
            ps.iter()
                .enumerate()
                .map(|(i, p)| (format!("p.{i}"), decimal(p))),
        ),
        Err(e) => out.push(("p.0".into(), err_value(&e))),
    }
    out.push(("rsquared".into(), exact(&fit.r_squared)));
    out.push(("rsquared_adj".into(), exact(&fit.adjusted_r_squared)));
    match fit.f_statistic() {
        Ok(v) => out.push(("fvalue".into(), exact(&v))),
        Err(e) => out.push(("fvalue".into(), err_value(&e))),
    }
    match fit.f_test(&ctx) {
        Ok(t) => out.push(("f_pvalue".into(), decimal(&t.p_value))),
        Err(e) => out.push(("f_pvalue".into(), err_value(&e))),
    }
    for (name, v) in [
        ("llf", fit.log_likelihood(&ctx)),
        ("aic", fit.aic(&ctx)),
        ("bic", fit.bic(&ctx)),
    ] {
        out.push((
            name.into(),
            v.map_or_else(|e| err_value(&e), |e| decimal(&e)),
        ));
    }
    Ok(out)
}

fn run_row(r: &Row) -> Outputs {
    let res = match r.kind.as_str() {
        "dist" => dist_row(r),
        "kernel" => kernel_row(r),
        "test" => test_row(r),
        "model" => model_row(r),
        k => Err(format!("unknown kind {k:?}")),
    };
    // A row the dumper cannot read is the case file's error, not symplex's.
    res.unwrap_or_else(|e| {
        vec![(
            "*".into(),
            format!("BADROW:{}", e.replace(['\n', '\t'], " ")),
        )]
    })
}

/// Runs one row on a thread with a generous stack, `None` after `limit`.
fn run_bounded(r: &Row, limit: Duration) -> Option<Outputs> {
    let (tx, rx) = mpsc::channel();
    let row = r.clone();
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(move || {
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_row(&row)))
                .unwrap_or_else(|p| {
                    let msg = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_string()))
                        .unwrap_or_default();
                    vec![(
                        "*".into(),
                        format!("PANIC:{}", msg.replace(['\n', '\t'], " ")),
                    )]
                });
            let _ = tx.send(out);
        })
        .expect("spawn a worker thread");
    rx.recv_timeout(limit).ok()
}

fn read_rows(path: &std::path::Path) -> Vec<Row> {
    let mut files: Vec<std::path::PathBuf> = if path.is_dir() {
        std::fs::read_dir(path)
            .expect("read the case directory")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "tsv"))
            .collect()
    } else {
        vec![path.to_path_buf()]
    };
    files.sort();
    files
        .iter()
        .flat_map(|f| {
            std::fs::read_to_string(f)
                .expect("read a case file")
                .lines()
                .filter_map(Row::parse)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn env_usize_pair(name: &str) -> (usize, usize) {
    std::env::var(name)
        .ok()
        .and_then(|s| {
            let (a, b) = s.split_once('/')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((0, 1))
}

/// Built-in rows for a plain `cargo test`, with the values they must give.
const SMOKE: &[(&str, &str, &str)] = &[
    ("smoke.cdf\tdist\tnormal\t0,1\tcdf\t0", "d30", "D:0.5"),
    (
        "smoke.pdf\tdist\tbinomial\t5,1/3\tpdf\t2",
        "d30",
        "D:0.329218106995884773662551440329",
    ),
    ("smoke.mean\tdist\tcauchy\t0,1\tmean\t-", "d30", "NAN"),
    (
        "smoke.q\tdist\tlogistic\t0,1\tquantile_f64\t0.5",
        "f64",
        "F:0e0",
    ),
    (
        "smoke.bad\tdist\tnormal\t0,-1\tcdf\t0",
        "d30",
        "ERR:InvalidArgument",
    ),
    ("smoke.kernel\tkernel\tnorm.cdf\t0.0\t-\t-", "f64", "F:5e-1"),
    (
        "smoke.test\ttest\tbinomial_test\tk=3;n=10;p0=1/2;alt=greater\t-\t-",
        "pvalue",
        "D:0.9453125",
    ),
    (
        "smoke.model\tmodel\tols\ty=1,2,4;x1=0,1,2;intercept=true\t-\t-",
        "coef.1",
        "Q:3/2",
    ),
];

#[test]
fn stats_oracle_dump() {
    let (Ok(cases), Ok(out_path)) = (
        std::env::var("STATS_ORACLE_CASES"),
        std::env::var("STATS_ORACLE_OUT"),
    ) else {
        for (line, output, want) in SMOKE {
            let row = Row::parse(line).expect("a smoke row");
            let got = run_row(&row);
            let value = got
                .iter()
                .find(|(name, _)| name == output)
                .map(|(_, v)| v.as_str())
                .unwrap_or_else(|| panic!("{}: no output {output} in {got:?}", row.id));
            assert!(
                value.starts_with(want),
                "{}: {output} = {value}, expected {want}",
                row.id
            );
        }
        return;
    };
    let rows = read_rows(std::path::Path::new(&cases));
    let (shard, shards) = env_usize_pair("STATS_ORACLE_SHARD");
    let start: usize = std::env::var("STATS_ORACLE_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let limit = Duration::from_secs(
        std::env::var("STATS_ORACLE_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60),
    );
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .expect("open STATS_ORACLE_OUT");
    for (idx, row) in rows.iter().enumerate() {
        if idx < start || idx % shards.max(1) != shard {
            continue;
        }
        writeln!(out, "#ROW\t{idx}\t{}", row.id).expect("write");
        out.flush().expect("flush");
        let t0 = Instant::now();
        match run_bounded(row, limit) {
            Some(outputs) => {
                let ms = t0.elapsed().as_millis();
                for (name, value) in outputs {
                    writeln!(out, "{}\t{name}\t{value}\t{ms}", row.id).expect("write");
                }
                out.flush().expect("flush");
            }
            None => {
                // The worker thread cannot be stopped: record the hang and end
                // the process; run_dump.py resumes after this row.
                writeln!(out, "{}\t*\tHANG\t{}", row.id, t0.elapsed().as_millis()).expect("write");
                out.flush().expect("flush");
                std::process::exit(3);
            }
        }
    }
}
