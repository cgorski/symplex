//! Arbitrary-precision numerical evaluation with complex number support.
//!
//! This module implements [`evalf`], which evaluates a symbolic expression
//! to a decimal string with a specified number of significant digits,
//! using the [`astro_float`] crate for arbitrary-precision arithmetic.
//!
//! # Design
//!
//! Evaluation is performed bottom-up using an explicit post-order
//! traversal (no recursion).  Each sub-expression is evaluated to a
//! [`Complex`] (a pair of [`BigFloat`] values for the real and imaginary
//! parts) and cached, so shared sub-expressions (common in a hash-consed
//! DAG) are only evaluated once.
//!
//! The working precision is set higher than the requested output
//! precision to absorb rounding errors from intermediate computations.
//! If a sub-expression cannot be evaluated (e.g., it contains free
//! symbols), the function returns an error.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Zero;
use rustc_hash::FxHashMap;

use astro_float::{BigFloat, Consts, Radix, RoundingMode, Sign};

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use tracing::debug;

// ═══════════════════════════════════════════════════════════════════════════
// Complex type
// ═══════════════════════════════════════════════════════════════════════════

/// A complex number represented as (real_part, imaginary_part).
type Complex = (BigFloat, BigFloat);

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate `expr` numerically to `digits` decimal digits of precision.
///
/// Returns the decimal string representation of the result, or an error
/// if the expression contains free symbols, infinities, or NaN.
pub(crate) fn evalf(arena: &Arena, expr: ExprId, digits: u32) -> Result<String, SymplexError> {
    // Convert decimal digits to binary precision with guard bits.
    // log2(10) ≈ 3.3219, so we use digits * 3.4 + 64 extra bits.
    let binary_prec = (digits as usize) * 34 / 10 + 64;
    // Ensure a reasonable minimum.
    let prec = binary_prec.max(128);

    debug!(
        digits = digits,
        binary_precision = prec,
        "evalf: starting numerical evaluation"
    );

    // Enforce the configured maximum precision.
    let max_prec = arena.config.max_evalf_precision as usize;
    if prec > max_prec {
        return Err(SymplexError::PrecisionExhausted {
            requested: digits,
            achieved: (max_prec * 10 / 34).saturating_sub(6) as u32,
        });
    }

    // Free symbols make the whole expression unevaluable; report the
    // offending symbol by name up front instead of surfacing an internal
    // cache miss from some parent node later.  Bound variables (a `Sum`
    // index, a `RootOf` polynomial variable, …) are not free and are
    // handled by the sub-tree evaluators below.
    let free = walk::free_symbols(arena, expr);
    if !free.is_empty() {
        let mut names: Vec<&str> = free
            .iter()
            .filter_map(|&id| match arena.node(id) {
                ExprNode::Symbol(sid) => Some(arena.symbol_name(*sid)),
                _ => None,
            })
            .collect();
        names.sort_unstable();
        if let Some(name) = names.first() {
            debug!(symbol = name, "evalf: expression has free symbols");
            return Err(SymplexError::FreeSymbol {
                name: (*name).to_owned(),
            });
        }
    }

    let rm = RoundingMode::ToEven;
    let mut cc = Consts::new().map_err(|e| {
        SymplexError::NotImplemented(format!("astro-float constants init failed: {e:?}"))
    })?;

    // Post-order traversal — evaluate children before parents.
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, Complex> = FxHashMap::default();

    for &id in &post_order {
        match eval_node(arena, id, &cache, prec, rm, &mut cc) {
            Ok(value) => {
                cache.insert(id, value);
            }
            Err(e) => {
                if id == expr {
                    return Err(e);
                }
                // Non-root node failed — skip it.  Parent nodes
                // (Sum, Product, Piecewise) will handle their own
                // sub-tree evaluation with substitution.
            }
        }
    }

    let result = cache.get(&expr).ok_or_else(|| {
        SymplexError::NotImplemented("evalf: expression not found in cache".into())
    })?;

    if result.0.is_nan() || result.1.is_nan() {
        return Err(SymplexError::PrecisionExhausted {
            requested: digits,
            achieved: 0,
        });
    }

    format_complex(result, digits, prec, rm, &mut cc)
}

/// Evaluate a constant (variable-free) arena expression to `f64`.
///
/// Returns `None` if the expression contains free symbols, produces
/// a complex result, or evaluation fails for any reason.
///
/// The expression is first simplified via [`eval`](crate::transforms::eval::eval)
/// to reduce exact values (e.g., `sin(π) → 0`, `Γ(5) → 24`) before
/// numerical evaluation.
pub(crate) fn eval_const_f64(arena: &mut Arena, expr: ExprId) -> Option<f64> {
    // Simplify exact values first.
    let evaled = crate::transforms::eval::eval(arena, expr);

    // Fast path: if the result is already a rational number, convert directly
    // without invoking the expensive arbitrary-precision machinery.
    if let Some(r) = arena.as_num(evaled) {
        let n: f64 = r.numer().to_string().parse().ok()?;
        let d: f64 = r.denom().to_string().parse().ok()?;
        if d == 0.0 {
            tracing::debug!("eval_const_f64: rational with zero denominator");
            return None;
        }
        let result = n / d;
        tracing::trace!(result, "eval_const_f64: rational fast path");
        return Some(result);
    }

    // Fall back to arbitrary-precision evaluation to 16 decimal digits,
    // then parse the resulting string.  The `evalf` call takes `&Arena`
    // (immutable), which Rust allows via automatic reborrowing of our
    // `&mut Arena`.
    match evalf(arena, evaled, 16) {
        Ok(s) => {
            let result = s.parse::<f64>().ok();
            tracing::trace!(?result, decimal_str = %s, "eval_const_f64: evalf path");
            result
        }
        Err(e) => {
            tracing::debug!(?e, "eval_const_f64: evalf failed");
            None
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Node evaluation
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate a single node given its children's values in the cache.
fn eval_node(
    arena: &Arena,
    id: ExprId,
    cache: &FxHashMap<ExprId, Complex>,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Complex, SymplexError> {
    let node = arena.node(id);

    match node {
        // ── Atoms ──────────────────────────────────────────────────
        ExprNode::Num(nid) => {
            let r = arena.num(*nid);
            Ok((ratio_to_bigfloat(r, prec, rm), BigFloat::new(prec)))
        }

        ExprNode::Pi => Ok((cc.pi(prec, rm).clone(), BigFloat::new(prec))),

        ExprNode::E => Ok((cc.e(prec, rm).clone(), BigFloat::new(prec))),

        ExprNode::ImaginaryUnit => Ok((BigFloat::new(prec), BigFloat::from_i32(1, prec))),

        // ── Named constants (arbitrary precision) ───────────────────────────
        ExprNode::EulerGamma => {
            debug!(prec, "evalf: EulerGamma via Brent–McMillan");
            let g = arb_euler_gamma(prec, rm, cc)?;
            Ok((g, BigFloat::new(prec)))
        }
        ExprNode::Catalan => {
            debug!(prec, "evalf: Catalan via Ramanujan series");
            let g = arb_catalan(prec, rm, cc)?;
            Ok((g, BigFloat::new(prec)))
        }
        ExprNode::GoldenRatio => {
            // φ = (1 + √5)/2, exactly via a single square root.
            let wp = prec + 16;
            let five = BigFloat::from_i32(5, wp);
            let sqrt5 = five.sqrt(wp, rm);
            let one = BigFloat::from_i32(1, wp);
            let two = BigFloat::from_i32(2, wp);
            let phi = one.add(&sqrt5, wp, rm).div(&two, prec, rm);
            Ok((phi, BigFloat::new(prec)))
        }

        // ── Complex analysis ────────────────────────────────────────────
        ExprNode::Re(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok((val.0.clone(), BigFloat::new(prec)))
        }
        ExprNode::Im(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok((val.1.clone(), BigFloat::new(prec)))
        }
        ExprNode::Conjugate(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok((val.0.clone(), val.1.neg()))
        }
        ExprNode::Arg(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.0.is_zero() && val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "arg(0) is undefined".into(),
                });
            }
            Ok((atan2_bf(&val.1, &val.0, prec, rm, cc), BigFloat::new(prec)))
        }

        // ── Special functions (0.2) ──────────────────────────────────────
        ExprNode::Si(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Si of complex argument not yet supported in evalf".into(),
                });
            }
            debug!(prec, "evalf: Si via series/asymptotic");
            let (si, _ci) = arb_si_ci(&val.0, true, prec, rm, cc)?;
            Ok((si, BigFloat::new(prec)))
        }
        ExprNode::Ci(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Ci of complex argument not yet supported in evalf".into(),
                });
            }
            if val.0.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Ci(0) is -∞".into(),
                });
            }
            debug!(prec, "evalf: Ci via series/asymptotic");
            let x_abs = val.0.abs();
            let (_si, ci) = arb_si_ci(&x_abs, false, prec, rm, cc)?;
            if val.0.is_negative() {
                // Ci(-x) = Ci(x) + iπ (principal branch of the logarithm).
                Ok((ci, cc.pi(prec, rm).clone()))
            } else {
                Ok((ci, BigFloat::new(prec)))
            }
        }
        ExprNode::Ei(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Ei of complex argument not yet supported in evalf".into(),
                });
            }
            if val.0.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Ei(0) is -∞".into(),
                });
            }
            debug!(prec, "evalf: Ei via series/asymptotic");
            let r = arb_ei(&val.0, prec, rm, cc)?;
            Ok((r, BigFloat::new(prec)))
        }
        ExprNode::Li(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() || val.0.is_negative() {
                return Err(SymplexError::Unevaluable {
                    reason: "li of complex or negative argument not yet supported in evalf".into(),
                });
            }
            if val.0.is_zero() {
                return Ok(c_zero(prec));
            }
            let wp = prec + 32;
            let ln_x = val.0.ln(wp, rm, cc);
            if ln_x.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "li(1) is -∞".into(),
                });
            }
            debug!(prec, "evalf: li via Ei(ln x)");
            let r = arb_ei(&ln_x, prec, rm, cc)?;
            Ok((r, BigFloat::new(prec)))
        }
        ExprNode::Zeta(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "zeta of complex argument not yet supported in evalf".into(),
                });
            }
            debug!(prec, "evalf: zeta via Borwein / functional equation");
            let r = arb_zeta(&val.0, prec, rm, cc)?;
            Ok((r, BigFloat::new(prec)))
        }
        ExprNode::Polygamma(n_id, x_id) => {
            let n_val = get_cached(cache, *n_id)?;
            let x_val = get_cached(cache, *x_id)?;
            if !n_val.1.is_zero() || !x_val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "polygamma of complex arguments not yet supported in evalf".into(),
                });
            }
            let n_f = bigfloat_to_f64(&n_val.0, rm, cc)?;
            let n_round = n_f.round();
            if (n_f - n_round).abs() > 1e-12 || !(0.0..=10_000.0).contains(&n_round) {
                return Err(SymplexError::Unevaluable {
                    reason: "polygamma order must be a non-negative integer".into(),
                });
            }
            let n = n_round as u32;
            if n == 0 {
                let r = arb_digamma(&x_val.0, prec, rm, cc)?;
                return Ok((r, BigFloat::new(prec)));
            }
            debug!(
                prec,
                n, "evalf: polygamma via recurrence + asymptotic series"
            );
            let r = arb_polygamma(n, &x_val.0, prec, rm, cc)?;
            Ok((r, BigFloat::new(prec)))
        }
        ExprNode::KroneckerDelta(i_id, j_id) => {
            let iv = get_cached(cache, *i_id)?;
            let jv = get_cached(cache, *j_id)?;
            let d_re = iv.0.sub(&jv.0, prec, rm);
            let d_im = iv.1.sub(&jv.1, prec, rm);
            if d_re.is_zero() && d_im.is_zero() {
                Ok(c_one(prec))
            } else {
                Ok(c_zero(prec))
            }
        }

        ExprNode::PhysicalConstant(_, value_id) => {
            // Recursively evaluate the stored exact value to a float.
            eval_node_or_subtree(arena, *value_id, cache, prec, rm, cc)
        }

        ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity => {
            Err(SymplexError::Unevaluable {
                reason: "cannot evaluate infinity to finite precision".into(),
            })
        }

        ExprNode::Factorial(_) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate symbolic factorial; call eval() first to reduce"
                .into(),
        }),

        ExprNode::Binomial(n_id, k_id) => {
            debug!(prec, "evalf: Binomial via arbitrary-precision Gamma");
            let n_val = get_cached(cache, *n_id)?;
            let k_val = get_cached(cache, *k_id)?;
            if !n_val.1.is_zero() || !k_val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Binomial of complex arguments not yet supported in evalf".into(),
                });
            }
            // C(n, k) = Gamma(n+1) / (Gamma(k+1) * Gamma(n-k+1))
            let one_bf = BigFloat::from_i32(1, prec);
            let n_plus_1 = n_val.0.add(&one_bf, prec, rm);
            let k_plus_1 = k_val.0.add(&one_bf, prec, rm);
            let n_minus_k = n_val.0.sub(&k_val.0, prec, rm);
            let n_minus_k_plus_1 = n_minus_k.add(&one_bf, prec, rm);

            let g_numer = arb_gamma_real(&n_plus_1, prec, rm, cc)?;
            let g_k = arb_gamma_real(&k_plus_1, prec, rm, cc)?;
            let g_nmk = arb_gamma_real(&n_minus_k_plus_1, prec, rm, cc)?;
            let g_denom = g_k.mul(&g_nmk, prec, rm);
            if g_denom.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Binomial denominator is zero (pole in Gamma)".into(),
                });
            }
            let result = g_numer.div(&g_denom, prec, rm);
            Ok((result, BigFloat::new(prec)))
        }

        ExprNode::Gamma(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Gamma of complex argument not yet supported in evalf".into(),
                });
            }
            debug!(prec, "evalf: Gamma via Stirling series");
            let result = arb_gamma_real(&val.0, prec, rm, cc)?;
            Ok((result, BigFloat::new(prec)))
        }

        ExprNode::LogGamma(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "LogGamma of complex argument not yet supported in evalf".into(),
                });
            }
            // Use arbitrary-precision Stirling series for log Γ(x).
            // For positive x, compute directly; for negative x, use
            // reflection: log Γ(x) = log(π) − log(sin(πx)) − log Γ(1−x).
            let x_approx = bigfloat_to_f64(&val.0, rm, cc)?;
            if x_approx <= 0.0 {
                let rounded = x_approx.round();
                if (rounded - x_approx).abs() < 1e-12 && rounded <= 0.0 {
                    return Err(SymplexError::Unevaluable {
                        reason: "LogGamma at non-positive integer pole".into(),
                    });
                }
                // Reflection: ln Γ(x) = ln π − ln|sin(πx)| − ln Γ(1−x)
                let one = BigFloat::from_i32(1, prec);
                let one_minus_x = one.sub(&val.0, prec, rm);
                let log_gamma_1mx = stirling_log_gamma(&one_minus_x, prec, rm, cc)?;
                let pi_val = cc.pi(prec, rm).clone();
                let pi_x = pi_val.mul(&val.0, prec, rm);
                let sin_pi_x = pi_x.sin(prec, rm, cc);
                let abs_sin = sin_pi_x.abs();
                if abs_sin.is_zero() {
                    return Err(SymplexError::Unevaluable {
                        reason: "LogGamma at non-positive integer pole".into(),
                    });
                }
                let ln_pi = cc.pi(prec, rm).clone().ln(prec, rm, cc);
                let ln_abs_sin = abs_sin.ln(prec, rm, cc);
                let result = ln_pi
                    .sub(&ln_abs_sin, prec, rm)
                    .sub(&log_gamma_1mx, prec, rm);
                Ok((result, BigFloat::new(prec)))
            } else {
                debug!(prec, "evalf: LogGamma via Stirling series");
                let result = stirling_log_gamma(&val.0, prec, rm, cc)?;
                Ok((result, BigFloat::new(prec)))
            }
        }

        ExprNode::Digamma(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Digamma of complex argument not yet supported in evalf".into(),
                });
            }
            let result = arb_digamma(&val.0, prec, rm, cc)?;
            Ok((result, BigFloat::new(prec)))
        }

        ExprNode::Erf(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "erf of complex argument not yet supported in evalf".into(),
                });
            }
            debug!(prec, "evalf: erf via arbitrary-precision series");
            let result = arb_erf(&val.0, prec, rm, cc)?;
            Ok((result, BigFloat::new(prec)))
        }

        ExprNode::Erfc(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "erfc of complex argument not yet supported in evalf".into(),
                });
            }
            debug!(prec, "evalf: erfc via arbitrary-precision series");
            let erf_val = arb_erf(&val.0, prec, rm, cc)?;
            let one = BigFloat::from_i32(1, prec);
            let result = one.sub(&erf_val, prec, rm);
            Ok((result, BigFloat::new(prec)))
        }

        ExprNode::LambertW(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "LambertW of complex argument not yet supported in evalf".into(),
                });
            }
            debug!(prec, "evalf: LambertW via Halley iteration");
            let result = arb_lambert_w(&val.0, prec, rm, cc)?;
            Ok((result, BigFloat::new(prec)))
        }

        ExprNode::Beta(a_id, b_id) => {
            let a_val = get_cached(cache, *a_id)?;
            let b_val = get_cached(cache, *b_id)?;
            if !a_val.1.is_zero() || !b_val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Beta of complex arguments not yet supported in evalf".into(),
                });
            }
            // Beta(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b) at arbitrary precision.
            debug!(prec, "evalf: Beta via arbitrary-precision Gamma");
            let ga = arb_gamma_real(&a_val.0, prec, rm, cc)?;
            let gb = arb_gamma_real(&b_val.0, prec, rm, cc)?;
            let a_plus_b = a_val.0.add(&b_val.0, prec, rm);
            let gab = arb_gamma_real(&a_plus_b, prec, rm, cc)?;
            if gab.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Beta function: Gamma(a+b) is zero".into(),
                });
            }
            let numer = ga.mul(&gb, prec, rm);
            let result = numer.div(&gab, prec, rm);
            Ok((result, BigFloat::new(prec)))
        }

        ExprNode::NaN => Err(SymplexError::Unevaluable {
            reason: "cannot evaluate NaN".into(),
        }),

        ExprNode::Symbol(sid) => {
            let name = arena.symbol_name(*sid);
            Err(SymplexError::FreeSymbol {
                name: name.to_owned(),
            })
        }

        // ── Add ────────────────────────────────────────────────────
        ExprNode::Add(children) => {
            let mut sum = c_zero(prec);
            for &child in children.iter() {
                let val = get_cached(cache, child)?;
                sum = c_add(&sum, val, prec, rm);
            }
            Ok(sum)
        }

        // ── Mul ────────────────────────────────────────────────────
        ExprNode::Mul(children) => {
            let mut product = c_one(prec);
            for &child in children.iter() {
                let val = get_cached(cache, child)?;
                product = c_mul(&product, val, prec, rm);
            }
            Ok(product)
        }

        // ── Pow ────────────────────────────────────────────────────
        ExprNode::Pow(base, exp) => {
            let b = get_cached(cache, *base)?;
            let e = get_cached(cache, *exp)?;

            let b_is_real = b.1.is_zero();
            let e_is_real = e.1.is_zero();

            // Optimization: use sqrt for x^(1/2) with non-negative real base.
            if b_is_real
                && !b.0.is_negative()
                && let ExprNode::Num(nid) = arena.node(*exp)
            {
                let r = arena.num(*nid);
                if *r == Ratio::new(BigInt::from(1), BigInt::from(2)) {
                    return Ok((b.0.sqrt(prec, rm), BigFloat::new(prec)));
                }
            }

            // Optimization: small integer exponents.
            if let Some(n) = try_as_small_int(arena, *exp) {
                if b_is_real {
                    // Real base with integer exponent: use real powi.
                    if n >= 0 {
                        return Ok((b.0.powi(n as usize, prec, rm), BigFloat::new(prec)));
                    } else {
                        let pow_pos = b.0.powi((-n) as usize, prec, rm);
                        let one_bf = BigFloat::from_i32(1, prec);
                        return Ok((one_bf.div(&pow_pos, prec, rm), BigFloat::new(prec)));
                    }
                } else {
                    // Complex base with integer exponent: use c_powi.
                    if n >= 0 {
                        return Ok(c_powi(b, n as usize, prec, rm));
                    } else {
                        let pow_pos = c_powi(b, (-n) as usize, prec, rm);
                        let one_c = c_one(prec);
                        return Ok(c_div(&one_c, &pow_pos, prec, rm));
                    }
                }
            }

            // Real positive base with real exponent: exp(e·ln b).
            // (`BigFloat::pow` is avoided — it hangs on exactly
            // representable results such as 4^(1/2); see `bf_pow`.)
            if b_is_real && e_is_real && b.0.is_positive() {
                return Ok((bf_pow(&b.0, &e.0, prec, rm, cc), BigFloat::new(prec)));
            }

            // General complex power: b^e = exp(e * ln(b)).
            Ok(c_pow(b, e, prec, rm, cc))
        }

        // ── Neg ────────────────────────────────────────────────────
        ExprNode::Neg(inner) => {
            let val = get_cached(cache, *inner)?;
            Ok(c_neg(val, prec, rm))
        }

        // ── Trig ───────────────────────────────────────────────────
        ExprNode::Sin(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.sin(prec, rm, cc), BigFloat::new(prec)))
            } else {
                Ok(c_sin(val, prec, rm, cc))
            }
        }

        ExprNode::Cos(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.cos(prec, rm, cc), BigFloat::new(prec)))
            } else {
                Ok(c_cos(val, prec, rm, cc))
            }
        }

        ExprNode::Tan(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.tan(prec, rm, cc), BigFloat::new(prec)))
            } else {
                let s = c_sin(val, prec, rm, cc);
                let c = c_cos(val, prec, rm, cc);
                Ok(c_div(&s, &c, prec, rm))
            }
        }

        // ── Exp / Ln ───────────────────────────────────────────────
        ExprNode::Exp(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.exp(prec, rm, cc), BigFloat::new(prec)))
            } else {
                Ok(c_exp(val, prec, rm, cc))
            }
        }

        ExprNode::Ln(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() && val.0.is_positive() {
                Ok((val.0.ln(prec, rm, cc), BigFloat::new(prec)))
            } else {
                Ok(c_ln(val, prec, rm, cc))
            }
        }

        // ── Abs ────────────────────────────────────────────────────
        ExprNode::Abs(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.abs(), BigFloat::new(prec)))
            } else {
                Ok((c_abs(val, prec, rm), BigFloat::new(prec)))
            }
        }

        // ── Inverse trig ──────────────────────────────────────────
        ExprNode::Asin(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.asin(prec, rm, cc), BigFloat::new(prec)))
            } else {
                Ok(c_asin(val, prec, rm, cc))
            }
        }

        ExprNode::Acos(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.acos(prec, rm, cc), BigFloat::new(prec)))
            } else {
                Ok(c_acos(val, prec, rm, cc))
            }
        }

        ExprNode::Atan(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.atan(prec, rm, cc), BigFloat::new(prec)))
            } else {
                Ok(c_atan(val, prec, rm, cc))
            }
        }

        // ── Atan2 ─────────────────────────────────────────────────
        ExprNode::Atan2(y_id, x_id) => {
            let yv = get_cached(cache, *y_id)?;
            let xv = get_cached(cache, *x_id)?;
            // Both args must be real for a real atan2 result.
            if yv.1.is_zero() && xv.1.is_zero() {
                Ok((atan2_bf(&yv.0, &xv.0, prec, rm, cc), BigFloat::new(prec)))
            } else {
                // Complex fallback: atan2 is not standard for complex args;
                // compute as -i * ln((x + iy) / sqrt(x² + y²))
                // but for now just use the real parts as a best-effort.
                Err(SymplexError::Unevaluable {
                    reason: "atan2 with complex arguments is not supported".into(),
                })
            }
        }

        // ── Hyperbolic ────────────────────────────────────────────
        ExprNode::Sinh(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.sinh(prec, rm, cc), BigFloat::new(prec)))
            } else {
                Ok(c_sinh(val, prec, rm, cc))
            }
        }

        ExprNode::Cosh(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.cosh(prec, rm, cc), BigFloat::new(prec)))
            } else {
                Ok(c_cosh(val, prec, rm, cc))
            }
        }

        ExprNode::Tanh(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.tanh(prec, rm, cc), BigFloat::new(prec)))
            } else {
                let s = c_sinh(val, prec, rm, cc);
                let c = c_cosh(val, prec, rm, cc);
                Ok(c_div(&s, &c, prec, rm))
            }
        }

        // ── Inverse hyperbolic ────────────────────────────────────
        ExprNode::Asinh(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.asinh(prec, rm, cc), BigFloat::new(prec)))
            } else {
                Ok(c_asinh(val, prec, rm, cc))
            }
        }

        ExprNode::Acosh(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.acosh(prec, rm, cc), BigFloat::new(prec)))
            } else {
                Ok(c_acosh(val, prec, rm, cc))
            }
        }

        ExprNode::Atanh(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                Ok((val.0.atanh(prec, rm, cc), BigFloat::new(prec)))
            } else {
                Ok(c_atanh(val, prec, rm, cc))
            }
        }

        // ── Sign ───────────────────────────────────────────────────
        ExprNode::Sign(inner) => {
            let val = get_cached(cache, *inner)?;
            if val.1.is_zero() {
                // Real case: sign returns -1, 0, or 1
                if val.0.is_zero() {
                    Ok(c_zero(prec))
                } else if val.0.is_negative() {
                    Ok((BigFloat::from_i32(-1, prec), BigFloat::new(prec)))
                } else {
                    Ok((BigFloat::from_i32(1, prec), BigFloat::new(prec)))
                }
            } else {
                // Complex case: sign(z) = z / |z|
                let modulus = c_abs(val, prec, rm);
                if modulus.is_zero() {
                    Ok(c_zero(prec))
                } else {
                    let denom = (modulus, BigFloat::new(prec));
                    Ok(c_div(val, &denom, prec, rm))
                }
            }
        }

        // ── Floor ──────────────────────────────────────────────────
        ExprNode::Floor(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "floor of complex number is not defined".into(),
                });
            }
            // BigFloat::int() returns the integer part (truncation toward zero).
            // floor(x) = x if x is integer, else truncate toward -infinity.
            let truncated = val.0.int();
            let result = if val.0.is_negative() && truncated != val.0 {
                // For negative non-integers, floor = trunc - 1
                truncated.sub(&BigFloat::from_i32(1, prec), prec, rm)
            } else {
                truncated
            };
            Ok((result, BigFloat::new(prec)))
        }

        // ── Ceiling ────────────────────────────────────────────────
        ExprNode::Ceiling(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "ceiling of complex number is not defined".into(),
                });
            }
            let truncated = val.0.int();
            let result = if val.0.is_positive() && truncated != val.0 {
                // For positive non-integers, ceil = trunc + 1
                truncated.add(&BigFloat::from_i32(1, prec), prec, rm)
            } else {
                truncated
            };
            Ok((result, BigFloat::new(prec)))
        }

        // ── Min ────────────────────────────────────────────────────
        ExprNode::Min(children) => {
            if children.is_empty() {
                return Err(SymplexError::Unevaluable {
                    reason: "min of empty set".into(),
                });
            }
            let mut best = get_cached(cache, children[0])?.clone();
            if !best.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "min requires real arguments".into(),
                });
            }
            for &child in &children[1..] {
                let val = get_cached(cache, child)?;
                if !val.1.is_zero() {
                    return Err(SymplexError::Unevaluable {
                        reason: "min requires real arguments".into(),
                    });
                }
                if val.0.sub(&best.0, prec, rm).is_negative() {
                    best = val.clone();
                }
            }
            Ok(best)
        }

        // ── Max ────────────────────────────────────────────────────
        ExprNode::Max(children) => {
            if children.is_empty() {
                return Err(SymplexError::Unevaluable {
                    reason: "max of empty set".into(),
                });
            }
            let mut best = get_cached(cache, children[0])?.clone();
            if !best.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "max requires real arguments".into(),
                });
            }
            for &child in &children[1..] {
                let val = get_cached(cache, child)?;
                if !val.1.is_zero() {
                    return Err(SymplexError::Unevaluable {
                        reason: "max requires real arguments".into(),
                    });
                }
                if val.0.sub(&best.0, prec, rm).is_positive() {
                    best = val.clone();
                }
            }
            Ok(best)
        }

        // ── Heaviside ──────────────────────────────────────────────
        ExprNode::Heaviside(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Heaviside of complex argument not supported in evalf".into(),
                });
            }
            if val.0.is_zero() {
                // H(0) = 0.5
                Ok((BigFloat::from_f64(0.5, prec), BigFloat::new(prec)))
            } else if val.0.is_negative() {
                Ok(c_zero(prec))
            } else {
                Ok((BigFloat::from_i32(1, prec), BigFloat::new(prec)))
            }
        }

        // ── DiracDelta ─────────────────────────────────────────────
        // Distributional: zero everywhere except at a single point of measure zero.
        ExprNode::DiracDelta(inner) => {
            let _val = get_cached(cache, *inner)?;
            // Numerically, δ(x) = 0 for all representable floats
            Ok(c_zero(prec))
        }

        // ── Apply-based special functions ───────────────────────────
        ExprNode::Apply(sid, args) => {
            let name = arena.symbol_name(*sid);
            match name {
                "besselj" if args.len() == 2 => {
                    let order = get_cached(cache, args[0])?;
                    let arg = get_cached(cache, args[1])?;
                    if !order.1.is_zero() || !arg.1.is_zero() {
                        return Err(SymplexError::Unevaluable {
                            reason: "Bessel of complex argument not yet supported in evalf".into(),
                        });
                    }
                    tracing::debug!(prec, "evalf: BesselJ via series/asymptotic");
                    let result = arb_bessel_j(&order.0, &arg.0, prec, rm, cc)?;
                    Ok((result, BigFloat::new(prec)))
                }
                "bessely" if args.len() == 2 => {
                    let order = get_cached(cache, args[0])?;
                    let arg = get_cached(cache, args[1])?;
                    if !order.1.is_zero() || !arg.1.is_zero() {
                        return Err(SymplexError::Unevaluable {
                            reason: "Bessel of complex argument not yet supported in evalf".into(),
                        });
                    }
                    tracing::debug!(prec, "evalf: BesselY via series/asymptotic");
                    let result = arb_bessel_y(&order.0, &arg.0, prec, rm, cc)?;
                    Ok((result, BigFloat::new(prec)))
                }
                "besseli" if args.len() == 2 => {
                    let order = get_cached(cache, args[0])?;
                    let arg = get_cached(cache, args[1])?;
                    if !order.1.is_zero() || !arg.1.is_zero() {
                        return Err(SymplexError::Unevaluable {
                            reason: "Bessel of complex argument not yet supported in evalf".into(),
                        });
                    }
                    tracing::debug!(prec, "evalf: BesselI via ascending series");
                    let result = arb_bessel_i(&order.0, &arg.0, prec, rm, cc)?;
                    Ok((result, BigFloat::new(prec)))
                }
                "besselk" if args.len() == 2 => {
                    let order = get_cached(cache, args[0])?;
                    let arg = get_cached(cache, args[1])?;
                    if !order.1.is_zero() || !arg.1.is_zero() {
                        return Err(SymplexError::Unevaluable {
                            reason: "Bessel of complex argument not yet supported in evalf".into(),
                        });
                    }
                    tracing::debug!(prec, "evalf: BesselK via series/asymptotic");
                    let result = arb_bessel_k(&order.0, &arg.0, prec, rm, cc)?;
                    Ok((result, BigFloat::new(prec)))
                }
                "legendre" | "chebyshev_t" | "chebyshev_u" | "hermite" | "laguerre"
                    if args.len() == 2 =>
                {
                    let n_val = get_cached(cache, args[0])?;
                    let x_val = get_cached(cache, args[1])?;
                    if !n_val.1.is_zero() || !x_val.1.is_zero() {
                        return Err(SymplexError::Unevaluable {
                            reason: "orthogonal polynomial of complex arguments not supported"
                                .into(),
                        });
                    }
                    let n_f = bigfloat_to_f64(&n_val.0, rm, cc)?;
                    let n_round = n_f.round();
                    if (n_f - n_round).abs() > 1e-12 || !(0.0..=1.0e7).contains(&n_round) {
                        return Err(SymplexError::Unevaluable {
                            reason: format!("{name}: degree must be a non-negative integer"),
                        });
                    }
                    tracing::debug!(
                        prec,
                        n = n_round,
                        "evalf: orthogonal polynomial via recurrence"
                    );
                    let kind = match name {
                        "legendre" => OrthoPoly::Legendre,
                        "chebyshev_t" => OrthoPoly::ChebyshevT,
                        "chebyshev_u" => OrthoPoly::ChebyshevU,
                        "hermite" => OrthoPoly::Hermite,
                        _ => OrthoPoly::Laguerre,
                    };
                    let result = arb_orthopoly(kind, n_round as u64, &x_val.0, prec, rm);
                    Ok((result, BigFloat::new(prec)))
                }
                _ => Err(SymplexError::Unevaluable {
                    reason: format!("cannot evaluate function '{name}'"),
                }),
            }
        }

        ExprNode::Derivative(_, _) => Err(SymplexError::Unevaluable {
            reason: "cannot evaluate unevaluated derivative".into(),
        }),

        ExprNode::Integral(_, _) => Err(SymplexError::Unevaluable {
            reason: "cannot evaluate unevaluated integral".into(),
        }),

        ExprNode::Sum(body_id, var_id, lo_id, hi_id) => {
            debug!("evalf: Sum — attempting finite evaluation");
            let lo_val = get_cached(cache, *lo_id)?;
            let hi_val = get_cached(cache, *hi_id)?;
            if !lo_val.1.is_zero() || !hi_val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Sum bounds must be real".into(),
                });
            }
            let lo_f = bigfloat_to_f64(&lo_val.0, rm, cc)?;
            let hi_f = bigfloat_to_f64(&hi_val.0, rm, cc)?;
            let lo_i = lo_f.round() as i64;
            let hi_i = hi_f.round() as i64;
            if (lo_f - lo_i as f64).abs() > 1e-9 || (hi_f - hi_i as f64).abs() > 1e-9 {
                return Err(SymplexError::Unevaluable {
                    reason: "Sum bounds are not integers".into(),
                });
            }
            if hi_i - lo_i > 10_000 {
                return Err(SymplexError::Unevaluable {
                    reason: "Sum range too large for numerical evaluation (> 10000 terms)".into(),
                });
            }
            let mut acc = c_zero(prec);
            for i in lo_i..=hi_i {
                let sub_val = (BigFloat::from_f64(i as f64, prec), BigFloat::new(prec));
                let term =
                    evalf_subtree_with_sub(arena, *body_id, *var_id, &sub_val, prec, rm, cc)?;
                acc = c_add(&acc, &term, prec, rm);
            }
            debug!(lo = lo_i, hi = hi_i, "evalf: Sum evaluated");
            Ok(acc)
        }

        ExprNode::Product_(body_id, var_id, lo_id, hi_id) => {
            debug!("evalf: Product — attempting finite evaluation");
            let lo_val = get_cached(cache, *lo_id)?;
            let hi_val = get_cached(cache, *hi_id)?;
            if !lo_val.1.is_zero() || !hi_val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Product bounds must be real".into(),
                });
            }
            let lo_f = bigfloat_to_f64(&lo_val.0, rm, cc)?;
            let hi_f = bigfloat_to_f64(&hi_val.0, rm, cc)?;
            let lo_i = lo_f.round() as i64;
            let hi_i = hi_f.round() as i64;
            if (lo_f - lo_i as f64).abs() > 1e-9 || (hi_f - hi_i as f64).abs() > 1e-9 {
                return Err(SymplexError::Unevaluable {
                    reason: "Product bounds are not integers".into(),
                });
            }
            if hi_i - lo_i > 10_000 {
                return Err(SymplexError::Unevaluable {
                    reason: "Product range too large for numerical evaluation (> 10000 terms)"
                        .into(),
                });
            }
            let mut acc = c_one(prec);
            for i in lo_i..=hi_i {
                let sub_val = (BigFloat::from_f64(i as f64, prec), BigFloat::new(prec));
                let term =
                    evalf_subtree_with_sub(arena, *body_id, *var_id, &sub_val, prec, rm, cc)?;
                acc = c_mul(&acc, &term, prec, rm);
            }
            debug!(lo = lo_i, hi = hi_i, "evalf: Product evaluated");
            Ok(acc)
        }

        ExprNode::Piecewise(pairs) => {
            debug!(
                branches = pairs.len(),
                "evalf: Piecewise — evaluating conditions"
            );
            // Branches are examined in order.  The first condition that is
            // decidedly true selects its value; a decidedly false condition
            // is skipped.  An *undecided* condition is an error — falling
            // through to a later `True` branch would be silently wrong.
            for &(value_id, cond_id) in pairs.iter() {
                match decide_condition(arena, cond_id, cache, prec, rm) {
                    Some(true) => {
                        return eval_node_or_subtree(arena, value_id, cache, prec, rm, cc);
                    }
                    Some(false) => continue,
                    None => {
                        return Err(SymplexError::Unevaluable {
                            reason: format!(
                                "cannot evaluate piecewise: condition `{}` is undecided",
                                arena.display(cond_id)
                            ),
                        });
                    }
                }
            }
            Err(SymplexError::Unevaluable {
                reason: "cannot evaluate piecewise: every condition is false".into(),
            })
        }

        ExprNode::BoolTrue
        | ExprNode::BoolFalse
        | ExprNode::Gt(_, _)
        | ExprNode::Ge(_, _)
        | ExprNode::Eq_(_, _)
        | ExprNode::Ne(_, _)
        | ExprNode::And(_)
        | ExprNode::Or(_)
        | ExprNode::Not(_) => Err(SymplexError::Unevaluable {
            reason: "boolean expression".into(),
        }),

        ExprNode::EmptySet
        | ExprNode::UniversalSet
        | ExprNode::Interval(_, _, _)
        | ExprNode::FiniteSet(_)
        | ExprNode::SetUnion(_)
        | ExprNode::SetIntersection(_)
        | ExprNode::SetComplement(_, _) => Err(SymplexError::Unevaluable {
            reason: "set-valued expressions cannot be numerically evaluated".into(),
        }),

        ExprNode::Limit(_, _, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate unevaluated Limit".into(),
        }),

        ExprNode::Series(_, _, _, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate unevaluated Series".into(),
        }),

        ExprNode::LaplaceTransform(_, _, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate unevaluated LaplaceTransform".into(),
        }),

        ExprNode::InverseLaplaceTransform(_, _, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate unevaluated InverseLaplaceTransform".into(),
        }),

        ExprNode::Residue(_, _, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate unevaluated Residue".into(),
        }),

        ExprNode::RootOf(poly_id, idx_id) => {
            // ── Extract the index as a non-negative integer ────────
            let idx: usize = match arena.node(*idx_id) {
                ExprNode::Num(nid) => {
                    let r = arena.num(*nid);
                    if r.is_integer() {
                        let n = r.to_integer();
                        // Convert BigInt → i64 → usize safely.
                        let n_i64: i64 = n.try_into().map_err(|_| SymplexError::Unevaluable {
                            reason: "RootOf index out of range".into(),
                        })?;
                        if n_i64 < 0 {
                            return Err(SymplexError::Unevaluable {
                                reason: "RootOf index must be non-negative".into(),
                            });
                        }
                        n_i64 as usize
                    } else {
                        return Err(SymplexError::Unevaluable {
                            reason: "RootOf index must be an integer".into(),
                        });
                    }
                }
                _ => {
                    return Err(SymplexError::Unevaluable {
                        reason: "RootOf index must be numeric".into(),
                    });
                }
            };

            // ── Identify the variable in the polynomial ───────────
            let syms = walk::free_symbols(arena, *poly_id);
            if syms.is_empty() {
                return Err(SymplexError::Unevaluable {
                    reason: "RootOf polynomial has no variables".into(),
                });
            }
            let var_id = syms[0];

            // ── Convert expression → dense Poly over ℚ ───────────
            let poly = match crate::poly::polybridge::expr_to_poly(arena, *poly_id, var_id) {
                Some(p) => p,
                None => {
                    return Err(SymplexError::Unevaluable {
                        reason: "could not convert RootOf expression to polynomial".into(),
                    });
                }
            };

            // ── Find all roots via Aberth's method ────────────────
            // Aberth handles both real and complex roots simultaneously
            // with cubic convergence.
            let roots = crate::poly::roots::aberth_roots(&poly, prec + 64, 200);

            if idx >= roots.len() {
                return Err(SymplexError::Unevaluable {
                    reason: format!(
                        "RootOf index {} exceeds the {} root(s) found",
                        idx,
                        roots.len()
                    ),
                });
            }

            let (re, im) = &roots[idx];
            Ok((re.clone(), im.clone()))
        }

        // ── RootSum: numerical evaluation via root-finding + summation ──
        //
        // RootSum(poly, body, sumvar) = Σ_{α: poly(α)=0} body(α, x).
        // We find the roots of poly numerically, substitute each into body,
        // evaluate, and sum.  If the body contains free variables other than
        // sumvar, this will fail (those variables must be substituted first).
        ExprNode::RootSum(poly, body, sumvar) => {
            // RootSum(poly, body, sumvar) = Σ_{α: poly(α)=0} body(α, x).
            //
            // Strategy:
            // 1. Convert the polynomial expression to a Poly via expr_to_poly
            //    (which takes &Arena — no mutation needed).
            // 2. Find ALL roots numerically via Aberth's method.
            // 3. For each root α_k, evaluate body with sumvar = α_k using
            //    evalf_subtree_with_sub (same mechanism as Sum evaluation).
            // 4. Sum all contributions.
            //
            // The imaginary parts should cancel for real-valued integrals.

            let poly_id = *poly;
            let body_id = *body;
            let sumvar_id = *sumvar;

            tracing::debug!("evalf: RootSum — attempting numerical evaluation via Aberth roots");

            // Step 1: Convert polynomial expression to Poly.
            let poly_obj = crate::poly::polybridge::expr_to_poly(arena, poly_id, sumvar_id)
                .ok_or_else(|| SymplexError::Unevaluable {
                    reason: "RootSum: cannot convert polynomial expression to Poly \
                             (may contain free symbols)"
                        .into(),
                })?;

            let poly_deg = poly_obj.degree().unwrap_or(0);
            tracing::debug!(
                degree = poly_deg,
                "evalf: RootSum — polynomial extracted, finding roots"
            );

            // Step 2: Find all roots via Aberth's method.
            // Use the same working precision as the rest of the evalf computation.
            let roots = crate::poly::roots::aberth_roots(&poly_obj, prec, 200);

            if roots.len() != poly_deg {
                tracing::debug!(
                    expected = poly_deg,
                    found = roots.len(),
                    "evalf: RootSum — Aberth returned fewer roots than expected"
                );
            }

            // Step 3+4: Evaluate body at each root and sum.
            let mut sum = c_zero(prec);
            for (k, root) in roots.iter().enumerate() {
                let term = evalf_subtree_with_sub(arena, body_id, sumvar_id, root, prec, rm, cc)?;
                tracing::trace!(root_idx = k, "evalf: RootSum — evaluated body at root");
                sum = c_add(&sum, &term, prec, rm);
            }

            tracing::debug!(
                n_roots = roots.len(),
                "evalf: RootSum — numerical evaluation complete"
            );
            Ok(sum)
        }

        ExprNode::DSolve(_, _, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate unevaluated DSolve".into(),
        }),

        ExprNode::ConditionSet(_, _) => Err(SymplexError::Unevaluable {
            reason: "cannot numerically evaluate ConditionSet".into(),
        }),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sub-tree evaluation helpers (Sum, Product, Piecewise)
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate a sub-expression with one variable replaced by a given numeric
/// value.  Used by `Sum` and `Product_` to iterate over finite ranges.
fn evalf_subtree_with_sub(
    arena: &Arena,
    expr: ExprId,
    var_id: ExprId,
    var_value: &Complex,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Complex, SymplexError> {
    let post_order = walk::post_order_ids(arena, expr);
    let mut sub_cache: FxHashMap<ExprId, Complex> = FxHashMap::default();
    // Pre-load the substitution so `var_id` resolves to the numeric value.
    sub_cache.insert(var_id, var_value.clone());

    for &id in &post_order {
        if sub_cache.contains_key(&id) {
            continue; // already present (e.g. the substituted variable)
        }
        let value = eval_node(arena, id, &sub_cache, prec, rm, cc)?;
        sub_cache.insert(id, value);
    }

    sub_cache
        .get(&expr)
        .cloned()
        .ok_or_else(|| SymplexError::Unevaluable {
            reason: "subtree evaluation with substitution failed".into(),
        })
}

/// Decide a boolean condition numerically using already-evaluated operands.
///
/// Relational nodes compare the cached real values of their operands;
/// `And`/`Or`/`Not` are combined with three-valued logic.  Returns `None`
/// when any needed operand is missing from the cache (free symbol, complex
/// value, unsupported node), so the caller can refuse rather than guess.
fn decide_condition(
    arena: &Arena,
    cond: ExprId,
    cache: &FxHashMap<ExprId, Complex>,
    prec: usize,
    rm: RoundingMode,
) -> Option<bool> {
    // Real-valued difference `a - b` when both operands are cached reals.
    let real_diff = |a: &ExprId, b: &ExprId| -> Option<BigFloat> {
        let (av, bv) = (cache.get(a)?, cache.get(b)?);
        if av.1.is_zero() && bv.1.is_zero() {
            Some(av.0.sub(&bv.0, prec, rm))
        } else {
            None
        }
    };

    let order = walk::post_order_ids(arena, cond);
    let mut truth: FxHashMap<ExprId, Option<bool>> = FxHashMap::default();
    for &id in &order {
        let v: Option<bool> = match arena.node(id) {
            ExprNode::BoolTrue => Some(true),
            ExprNode::BoolFalse => Some(false),
            ExprNode::Gt(a, b) => real_diff(a, b).map(|d| d.is_positive()),
            ExprNode::Ge(a, b) => real_diff(a, b).map(|d| d.is_positive() || d.is_zero()),
            ExprNode::Eq_(a, b) => real_diff(a, b).map(|d| d.is_zero()),
            ExprNode::Ne(a, b) => real_diff(a, b).map(|d| !d.is_zero()),
            ExprNode::Not(inner) => truth.get(inner).copied().flatten().map(|b| !b),
            ExprNode::And(kids) => {
                let vals: Vec<Option<bool>> = kids
                    .iter()
                    .map(|k| truth.get(k).copied().flatten())
                    .collect();
                if vals.contains(&Some(false)) {
                    Some(false)
                } else if vals.iter().all(|v| *v == Some(true)) {
                    Some(true)
                } else {
                    None
                }
            }
            ExprNode::Or(kids) => {
                let vals: Vec<Option<bool>> = kids
                    .iter()
                    .map(|k| truth.get(k).copied().flatten())
                    .collect();
                if vals.contains(&Some(true)) {
                    Some(true)
                } else if vals.iter().all(|v| *v == Some(false)) {
                    Some(false)
                } else {
                    None
                }
            }
            // Numeric operands and anything else carry no truth value.
            _ => None,
        };
        truth.insert(id, v);
    }
    truth.get(&cond).copied().flatten()
}

/// Return the cached value for `id`, or fall back to calling `eval_node`.
fn eval_node_or_subtree(
    arena: &Arena,
    id: ExprId,
    cache: &FxHashMap<ExprId, Complex>,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Complex, SymplexError> {
    if let Some(val) = cache.get(&id) {
        return Ok(val.clone());
    }
    eval_node(arena, id, cache, prec, rm, cc)
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex arithmetic helpers
// ═══════════════════════════════════════════════════════════════════════════

pub(crate) fn c_zero(prec: usize) -> Complex {
    (BigFloat::new(prec), BigFloat::new(prec))
}

pub(crate) fn c_one(prec: usize) -> Complex {
    (BigFloat::from_i32(1, prec), BigFloat::new(prec))
}

fn c_i(prec: usize) -> Complex {
    (BigFloat::new(prec), BigFloat::from_i32(1, prec))
}

pub(crate) fn c_from_real(r: BigFloat, prec: usize) -> Complex {
    (r, BigFloat::new(prec))
}

pub(crate) fn c_add(a: &Complex, b: &Complex, prec: usize, rm: RoundingMode) -> Complex {
    (a.0.add(&b.0, prec, rm), a.1.add(&b.1, prec, rm))
}

pub(crate) fn c_sub(a: &Complex, b: &Complex, prec: usize, rm: RoundingMode) -> Complex {
    (a.0.sub(&b.0, prec, rm), a.1.sub(&b.1, prec, rm))
}

pub(crate) fn c_mul(a: &Complex, b: &Complex, prec: usize, rm: RoundingMode) -> Complex {
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    let ac = a.0.mul(&b.0, prec, rm);
    let bd = a.1.mul(&b.1, prec, rm);
    let ad = a.0.mul(&b.1, prec, rm);
    let bc = a.1.mul(&b.0, prec, rm);
    (ac.sub(&bd, prec, rm), ad.add(&bc, prec, rm))
}

pub(crate) fn c_div(a: &Complex, b: &Complex, prec: usize, rm: RoundingMode) -> Complex {
    // (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
    let ac = a.0.mul(&b.0, prec, rm);
    let bd = a.1.mul(&b.1, prec, rm);
    let bc = a.1.mul(&b.0, prec, rm);
    let ad = a.0.mul(&b.1, prec, rm);
    let denom =
        b.0.mul(&b.0, prec, rm)
            .add(&b.1.mul(&b.1, prec, rm), prec, rm);
    let re = ac.add(&bd, prec, rm).div(&denom, prec, rm);
    let im = bc.sub(&ad, prec, rm).div(&denom, prec, rm);
    (re, im)
}

#[allow(unused_variables)]
pub(crate) fn c_neg(a: &Complex, prec: usize, rm: RoundingMode) -> Complex {
    (a.0.neg(), a.1.neg())
}

pub(crate) fn c_abs(a: &Complex, prec: usize, rm: RoundingMode) -> BigFloat {
    // |z| = sqrt(re² + im²)
    let re2 = a.0.mul(&a.0, prec, rm);
    let im2 = a.1.mul(&a.1, prec, rm);
    let sum = re2.add(&im2, prec, rm);
    sum.sqrt(prec, rm)
}

fn c_exp(z: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // exp(a+bi) = exp(a)(cos(b) + i·sin(b))
    let exp_a = z.0.exp(prec, rm, cc);
    let cos_b = z.1.cos(prec, rm, cc);
    let sin_b = z.1.sin(prec, rm, cc);
    (exp_a.mul(&cos_b, prec, rm), exp_a.mul(&sin_b, prec, rm))
}

fn c_ln(z: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // ln(z) = ln|z| + i·arg(z)
    let r = c_abs(z, prec, rm);
    let theta = atan2_bf(&z.1, &z.0, prec, rm, cc);
    (r.ln(prec, rm, cc), theta)
}

fn c_sin(z: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // sin(a+bi) = sin(a)cosh(b) + i·cos(a)sinh(b)
    let sin_a = z.0.sin(prec, rm, cc);
    let cos_a = z.0.cos(prec, rm, cc);
    let cosh_b = z.1.cosh(prec, rm, cc);
    let sinh_b = z.1.sinh(prec, rm, cc);
    (sin_a.mul(&cosh_b, prec, rm), cos_a.mul(&sinh_b, prec, rm))
}

fn c_cos(z: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // cos(a+bi) = cos(a)cosh(b) - i·sin(a)sinh(b)
    let cos_a = z.0.cos(prec, rm, cc);
    let sin_a = z.0.sin(prec, rm, cc);
    let cosh_b = z.1.cosh(prec, rm, cc);
    let sinh_b = z.1.sinh(prec, rm, cc);
    (
        cos_a.mul(&cosh_b, prec, rm),
        sin_a.mul(&sinh_b, prec, rm).neg(),
    )
}

fn c_sinh(z: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // sinh(a+bi) = sinh(a)cos(b) + i·cosh(a)sin(b)
    let sinh_a = z.0.sinh(prec, rm, cc);
    let cosh_a = z.0.cosh(prec, rm, cc);
    let cos_b = z.1.cos(prec, rm, cc);
    let sin_b = z.1.sin(prec, rm, cc);
    (sinh_a.mul(&cos_b, prec, rm), cosh_a.mul(&sin_b, prec, rm))
}

fn c_cosh(z: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // cosh(a+bi) = cosh(a)cos(b) + i·sinh(a)sin(b)
    let cosh_a = z.0.cosh(prec, rm, cc);
    let sinh_a = z.0.sinh(prec, rm, cc);
    let cos_b = z.1.cos(prec, rm, cc);
    let sin_b = z.1.sin(prec, rm, cc);
    (cosh_a.mul(&cos_b, prec, rm), sinh_a.mul(&sin_b, prec, rm))
}

fn c_pow(base: &Complex, exp: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // base^exp = exp(exp * ln(base))
    // Special case: base is zero.
    if base.0.is_zero() && base.1.is_zero() {
        // 0^0 = 1 by convention; 0^(positive) = 0.
        if exp.0.is_zero() && exp.1.is_zero() {
            return c_one(prec);
        }
        return c_zero(prec);
    }
    let ln_base = c_ln(base, prec, rm, cc);
    let product = c_mul(exp, &ln_base, prec, rm);
    c_exp(&product, prec, rm, cc)
}

fn c_powi(base: &Complex, n: usize, prec: usize, rm: RoundingMode) -> Complex {
    if n == 0 {
        return c_one(prec);
    }
    let mut result = c_one(prec);
    let mut b = base.clone();
    let mut exp = n;
    while exp > 0 {
        if exp & 1 == 1 {
            result = c_mul(&result, &b, prec, rm);
        }
        b = c_mul(&b, &b, prec, rm);
        exp >>= 1;
    }
    result
}

fn c_sqrt(z: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    if z.0.is_zero() && z.1.is_zero() {
        return c_zero(prec);
    }
    let half = c_from_real(
        BigFloat::from_i32(1, prec).div(&BigFloat::from_i32(2, prec), prec, rm),
        prec,
    );
    c_pow(z, &half, prec, rm, cc)
}

// ── Inverse trig (complex) ─────────────────────────────────────────

fn c_asin(z: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // asin(z) = -i * ln(iz + sqrt(1 - z²))
    let i_unit = c_i(prec);
    let one = c_one(prec);
    let z_sq = c_mul(z, z, prec, rm);
    let one_minus_z_sq = c_sub(&one, &z_sq, prec, rm);
    let sqrt_term = c_sqrt(&one_minus_z_sq, prec, rm, cc);
    let iz = c_mul(&i_unit, z, prec, rm);
    let sum = c_add(&iz, &sqrt_term, prec, rm);
    let ln_sum = c_ln(&sum, prec, rm, cc);
    let neg_i = c_neg(&i_unit, prec, rm);
    c_mul(&neg_i, &ln_sum, prec, rm)
}

fn c_acos(z: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // acos(z) = pi/2 - asin(z)
    let half_pi_val = cc.pi(prec, rm).div(&BigFloat::from_i32(2, prec), prec, rm);
    let half_pi = c_from_real(half_pi_val, prec);
    let asin_z = c_asin(z, prec, rm, cc);
    c_sub(&half_pi, &asin_z, prec, rm)
}

fn c_atan(z: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // atan(z) = (i/2) * ln((i+z)/(i-z))
    let i_unit = c_i(prec);
    let two = c_from_real(BigFloat::from_i32(2, prec), prec);
    let i_over_2 = c_div(&i_unit, &two, prec, rm);
    let i_plus_z = c_add(&i_unit, z, prec, rm);
    let i_minus_z = c_sub(&i_unit, z, prec, rm);
    let ratio = c_div(&i_plus_z, &i_minus_z, prec, rm);
    let ln_ratio = c_ln(&ratio, prec, rm, cc);
    c_mul(&i_over_2, &ln_ratio, prec, rm)
}

// ── Inverse hyperbolic (complex) ──────────────────────────────────

fn c_asinh(z: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // asinh(z) = ln(z + sqrt(z² + 1))
    let one = c_one(prec);
    let z_sq = c_mul(z, z, prec, rm);
    let z_sq_plus_one = c_add(&z_sq, &one, prec, rm);
    let sqrt_term = c_sqrt(&z_sq_plus_one, prec, rm, cc);
    let sum = c_add(z, &sqrt_term, prec, rm);
    c_ln(&sum, prec, rm, cc)
}

fn c_acosh(z: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // acosh(z) = ln(z + sqrt(z² - 1))
    let one = c_one(prec);
    let z_sq = c_mul(z, z, prec, rm);
    let z_sq_minus_one = c_sub(&z_sq, &one, prec, rm);
    let sqrt_term = c_sqrt(&z_sq_minus_one, prec, rm, cc);
    let sum = c_add(z, &sqrt_term, prec, rm);
    c_ln(&sum, prec, rm, cc)
}

fn c_atanh(z: &Complex, prec: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // atanh(z) = (1/2) * ln((1+z)/(1-z))
    let one = c_one(prec);
    let two = c_from_real(BigFloat::from_i32(2, prec), prec);
    let half = c_div(&one, &two, prec, rm);
    let one_plus_z = c_add(&one, z, prec, rm);
    let one_minus_z = c_sub(&one, z, prec, rm);
    let ratio = c_div(&one_plus_z, &one_minus_z, prec, rm);
    let ln_ratio = c_ln(&ratio, prec, rm, cc);
    c_mul(&half, &ln_ratio, prec, rm)
}

// ═══════════════════════════════════════════════════════════════════════════
// Real-valued helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute atan2(y, x) using BigFloat arithmetic.
fn atan2_bf(
    y: &BigFloat,
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> BigFloat {
    if x.is_zero() && y.is_zero() {
        return BigFloat::new(prec);
    }

    let pi_val = cc.pi(prec, rm);
    let two = BigFloat::from_i32(2, prec);
    let half_pi = pi_val.div(&two, prec, rm);

    if x.is_zero() {
        return if y.is_positive() {
            half_pi
        } else {
            half_pi.neg()
        };
    }

    let ratio = y.div(x, prec, rm);
    let atan_val = ratio.atan(prec, rm, cc);

    if x.is_positive() {
        atan_val
    } else {
        let pi_val2 = cc.pi(prec, rm);
        if y.is_negative() {
            atan_val.sub(&pi_val2, prec, rm)
        } else {
            atan_val.add(&pi_val2, prec, rm)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Special function helpers (f64-based)
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a `BigFloat` to `f64` via decimal radix conversion.
fn bigfloat_to_f64(bf: &BigFloat, rm: RoundingMode, cc: &mut Consts) -> Result<f64, SymplexError> {
    if bf.is_zero() {
        return Ok(0.0);
    }
    if bf.is_inf_pos() {
        return Ok(f64::INFINITY);
    }
    if bf.is_inf_neg() {
        return Ok(f64::NEG_INFINITY);
    }
    if bf.is_nan() {
        return Err(SymplexError::Unevaluable {
            reason: "NaN in BigFloat to f64 conversion".into(),
        });
    }

    let (sign, mantissa, exponent) =
        bf.convert_to_radix(Radix::Dec, rm, cc)
            .map_err(|e| SymplexError::Unevaluable {
                reason: format!("BigFloat to f64 conversion failed: {e:?}"),
            })?;

    // mantissa is [d1, d2, ...] representing 0.d1d2d3... × 10^exponent
    let mut s = String::with_capacity(mantissa.len() + 8);
    if sign == Sign::Neg {
        s.push('-');
    }
    s.push_str("0.");
    for &d in &mantissa {
        s.push((b'0' + d) as char);
    }
    s.push('e');
    s.push_str(&exponent.to_string());

    s.parse::<f64>().map_err(|_| SymplexError::Unevaluable {
        reason: "failed to parse BigFloat decimal representation as f64".into(),
    })
}

/// Lanczos approximation for the Gamma function (g=7, 9 coefficients).
#[allow(dead_code)]
#[allow(clippy::excessive_precision)]
fn lanczos_gamma_f64(x: f64) -> Result<f64, SymplexError> {
    if x.is_nan() || x.is_infinite() {
        return Err(SymplexError::Unevaluable {
            reason: "Gamma of special float value".into(),
        });
    }

    // Reflection formula for x < 0.5:  Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
    if x < 0.5 {
        let reflected = lanczos_gamma_f64(1.0 - x)?;
        let sin_pi_x = (std::f64::consts::PI * x).sin();
        if sin_pi_x.abs() < 1e-300 {
            return Err(SymplexError::Unevaluable {
                reason: "Gamma at non-positive integer pole".into(),
            });
        }
        let result = std::f64::consts::PI / (sin_pi_x * reflected);
        if result.is_finite() {
            return Ok(result);
        }
        return Err(SymplexError::Unevaluable {
            reason: "Gamma at pole".into(),
        });
    }

    // Lanczos approximation with g=7
    const COEFFICIENTS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let x = x - 1.0; // Lanczos uses Gamma(x+1) = x!
    let t = x + 7.0 + 0.5; // g = 7

    let mut sum = COEFFICIENTS[0];
    for (i, &c) in COEFFICIENTS[1..].iter().enumerate() {
        sum += c / (x + i as f64 + 1.0);
    }

    let result = (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum;
    Ok(result)
}

/// Error function via Taylor series (small |x|) or asymptotic expansion (large |x|).
#[allow(dead_code)]
fn erf_f64(x: f64) -> f64 {
    if x.abs() < 4.0 {
        // Taylor series: erf(x) = 2/sqrt(pi) * sum_{n=0}^{inf} (-1)^n * x^(2n+1) / (n! * (2n+1))
        let mut sum = 0.0;
        let mut term = x; // first term: x
        sum += term;
        for n in 1..50 {
            term *= -x * x / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // For large |x|, use complementary: erf(x) = 1 - erfc(x)
        // erfc(x) ~ exp(-x^2)/(x*sqrt(pi)) * sum_{n=0} (-1)^n * (2n-1)!! / (2x^2)^n
        let sign = x.signum();
        let ax = x.abs();
        let mut sum = 1.0;
        let mut term = 1.0;
        for n in 1..20 {
            term *= -(2 * n - 1) as f64 / (2.0 * ax * ax);
            if term.abs() < 1e-16 {
                break;
            }
            sum += term;
        }
        let erfc = (-ax * ax).exp() / (ax * std::f64::consts::PI.sqrt()) * sum;
        sign * (1.0 - erfc)
    }
}

/// Arbitrary-precision digamma (psi) function via recurrence + asymptotic series.
///
/// Algorithm:
/// 1. For x < 0, use reflection: ψ(x) = ψ(1−x) − π·cot(πx)
/// 2. Use recurrence ψ(x+1) = ψ(x) + 1/x to shift x to a large value
/// 3. Asymptotic expansion: ψ(x) ~ ln(x) − 1/(2x) − Σ B_{2k}/(2k · x^{2k})
fn arb_digamma(
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    let guard = 20;
    let wp = prec + guard;

    if x.is_nan() || x.is_inf_pos() || x.is_inf_neg() {
        return Err(SymplexError::Unevaluable {
            reason: "Digamma of special float value".into(),
        });
    }

    let one = BigFloat::from_i32(1, wp);

    // Handle negative x via reflection: ψ(x) = ψ(1−x) − π·cot(πx)
    if x.is_negative() {
        let pi_val = cc.pi(wp, rm).clone();
        let pi_x = pi_val.mul(x, wp, rm);
        let sin_val = pi_x.sin(wp, rm, cc);

        // Check for pole at non-positive integers
        if sin_val.is_zero() {
            return Err(SymplexError::Unevaluable {
                reason: "Digamma at non-positive integer pole".into(),
            });
        }
        if let Some(s_exp) = sin_val.exponent()
            && (s_exp as i64) < -(wp as i64 / 2)
        {
            return Err(SymplexError::Unevaluable {
                reason: "Digamma at non-positive integer pole".into(),
            });
        }

        let cos_val = pi_x.cos(wp, rm, cc);
        let cot_val = cos_val.div(&sin_val, wp, rm);
        let one_minus_x = one.sub(x, wp, rm);
        let psi_1mx = arb_digamma(&one_minus_x, prec, rm, cc)?;
        let pi_cot = cc.pi(wp, rm).clone().mul(&cot_val, wp, rm);
        return Ok(psi_1mx.sub(&pi_cot, wp, rm));
    }

    // Use recurrence ψ(x+1) = ψ(x) + 1/x to shift x >= threshold
    let threshold_val = (wp as i32 / 3).max(10);
    let threshold = BigFloat::from_i32(threshold_val, wp);
    let mut result = BigFloat::new(wp); // 0
    let mut x = x.clone();

    while x.sub(&threshold, wp, rm).is_negative() {
        // Check for pole at zero
        if x.is_zero() {
            return Err(SymplexError::Unevaluable {
                reason: "Digamma at non-positive integer pole".into(),
            });
        }
        if let Some(x_exp) = x.exponent()
            && (x_exp as i64) < -(wp as i64 / 2)
        {
            return Err(SymplexError::Unevaluable {
                reason: "Digamma at non-positive integer pole".into(),
            });
        }
        let inv_x = one.div(&x, wp, rm);
        result = result.sub(&inv_x, wp, rm);
        x = x.add(&one, wp, rm);
    }

    // Asymptotic expansion: ψ(x) ~ ln(x) − 1/(2x) − Σ_{k=1}^{N} B_{2k}/(2k · x^{2k})
    let ln_x = x.ln(wp, rm, cc);
    result = result.add(&ln_x, wp, rm);

    let two = BigFloat::from_i32(2, wp);
    let half_inv_x = one.div(&x.mul(&two, wp, rm), wp, rm);
    result = result.sub(&half_inv_x, wp, rm);

    let x2 = x.mul(&x, wp, rm);
    let mut x_pow = x2.clone(); // x^2

    // Bernoulli numbers B_{2k} for k=1..12 as (numerator, denominator):
    // B2=1/6, B4=−1/30, B6=1/42, B8=−1/30, B10=5/66, B12=−691/2730
    // B14=7/6, B16=−3617/510, B18=43867/798, B20=−174611/330
    // B22=854513/138, B24=−236364091/2730
    let bernoulli_nums: &[(i128, i128)] = &[
        (1, 6),
        (-1, 30),
        (1, 42),
        (-1, 30),
        (5, 66),
        (-691, 2730),
        (7, 6),
        (-3617, 510),
        (43867, 798),
        (-174611, 330),
        (854513, 138),
        (-236364091, 2730),
    ];

    let n_terms = (prec / 6 + 2).min(bernoulli_nums.len());

    for (k_idx, &(bn, bd)) in bernoulli_nums[..n_terms].iter().enumerate() {
        let k = (k_idx + 1) as i128;
        let two_k = 2 * k;
        // Term = B_{2k} / (2k · x^{2k})
        let coeff_n = BigFloat::from_i128(bn, wp);
        let coeff_d = BigFloat::from_i128(bd * two_k, wp);
        let coeff = coeff_n.div(&coeff_d, wp, rm);
        let inv_xpow = one.div(&x_pow, wp, rm);
        let term = coeff.mul(&inv_xpow, wp, rm);
        result = result.sub(&term, wp, rm);

        // Convergence check via binary exponents
        if let (Some(t_exp), Some(r_exp)) = (term.exponent(), result.exponent())
            && (r_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }

        x_pow = x_pow.mul(&x2, wp, rm);
    }

    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Arbitrary-precision Gamma via Stirling series
// ═══════════════════════════════════════════════════════════════════════════

/// Compute log Γ(z) for real z > 0 using the Stirling asymptotic series.
///
/// Algorithm:
/// 1. Argument reduction — shift z by integer r until z + r ≥ 0.2 · p.
/// 2. Stirling series:
///    log Γ(z) = (z − ½) ln(z) − z + ½ ln(2π) + Σ B₂ₖ / [2k(2k−1) z^{2k−1}].
/// 3. Undo shift:
///    log Γ(z) = log Γ(z + r) − Σ_{i=0}^{r−1} ln(z + i).
fn stirling_log_gamma(
    z: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    tracing::debug!(prec, "stirling_log_gamma: entry");

    let guard = 20; // extra guard bits
    let wp = prec + guard;

    // Determine how far to shift z for convergence.
    let z_approx = bigfloat_to_f64(z, rm, cc)?;
    let threshold = ((wp as f64) * 0.2).ceil() as i64;
    let shift = if z_approx < threshold as f64 {
        (threshold - z_approx.floor() as i64).max(0) as usize
    } else {
        0
    };

    tracing::trace!(
        z_approx,
        threshold,
        shift,
        "stirling_log_gamma: argument reduction"
    );

    // z_shifted = z + shift
    let shift_bf = BigFloat::from_i128(shift as i128, wp);
    let z_shifted = z.add(&shift_bf, wp, rm);

    // Main Stirling formula:
    //   log Γ(z) = (z − 1/2)·ln(z) − z + (1/2)·ln(2π) + Σ ...
    let half = BigFloat::from_f64(0.5, wp);
    let z_minus_half = z_shifted.sub(&half, wp, rm);
    let log_z = z_shifted.ln(wp, rm, cc);

    let mut result = z_minus_half.mul(&log_z, wp, rm);
    result = result.sub(&z_shifted, wp, rm);

    // (1/2) · ln(2π) at working precision.
    let two = BigFloat::from_i32(2, wp);
    let pi_val = cc.pi(wp, rm).clone();
    let two_pi = two.mul(&pi_val, wp, rm);
    let half_log_2pi = two_pi.ln(wp, rm, cc).mul(&half, wp, rm);
    result = result.add(&half_log_2pi, wp, rm);

    // Series: Σ_{k=1}^{N} B_{2k} / [2k(2k−1) · z^{2k−1}]
    let z_sq = z_shifted.mul(&z_shifted, wp, rm);
    let mut z_pow = z_shifted.clone(); // z^1 → z^3 → z^5 → …

    // Upper bound on useful terms: safely below the divergence point πz.
    let max_terms = ((wp as f64) * 0.35) as usize + 10;
    let mut prev_term_exp: Option<i32> = None;

    for k in 1..=max_terms {
        let b2k = crate::base::bernoulli::bernoulli(2 * k);
        if b2k.is_zero() {
            continue;
        }

        // Advance z_pow: z^1, z^3, z^5, …
        if k > 1 {
            z_pow = z_pow.mul(&z_sq, wp, rm);
        }

        // term = B_{2k} / (2k · (2k−1) · z^{2k−1})
        let b2k_bf = ratio_to_bigfloat(&b2k, wp, rm);
        let denom_int = (2 * k * (2 * k - 1)) as i128;
        let denom = BigFloat::from_i128(denom_int, wp);
        let denom_full = denom.mul(&z_pow, wp, rm);
        let term = b2k_bf.div(&denom_full, wp, rm);

        // Divergence / convergence detection via binary exponents.
        if let Some(t_exp) = term.exponent() {
            if let Some(prev) = prev_term_exp
                && t_exp > prev + 10
            {
                tracing::trace!(k, "stirling series: diverging, stopping");
                break;
            }
            prev_term_exp = Some(t_exp);

            // Term is negligible relative to accumulated result.
            if let Some(r_exp) = result.exponent()
                && (r_exp as i64 - t_exp as i64) > wp as i64
            {
                tracing::trace!(k, "stirling series: converged");
                break;
            }
        }

        result = result.add(&term, wp, rm);
    }

    // Undo argument reduction:
    //   log Γ(z) = log Γ(z+shift) − Σ_{i=0}^{shift−1} ln(z + i)
    for i in 0..shift {
        let z_plus_i = z.add(&BigFloat::from_i128(i as i128, wp), wp, rm);
        let log_zi = z_plus_i.ln(wp, rm, cc);
        result = result.sub(&log_zi, wp, rm);
    }

    Ok(result)
}

/// Compute Γ(x) at arbitrary precision for real x.
///
/// For positive x, evaluates exp(stirling_log_gamma(x)).
/// For negative non-integer x, applies the reflection formula
/// Γ(z) = π / (sin(πz) · Γ(1 − z)).
fn arb_gamma_real(
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    // Pole at zero.
    if x.is_zero() {
        return Err(SymplexError::Unevaluable {
            reason: "Gamma at non-positive integer pole".into(),
        });
    }

    let x_approx = bigfloat_to_f64(x, rm, cc)?;

    // Detect non-positive integer poles via f64 approximation.
    if x_approx <= 0.0 {
        let rounded = x_approx.round();
        if (rounded - x_approx).abs() < 1e-12 && rounded <= 0.0 {
            return Err(SymplexError::Unevaluable {
                reason: "Gamma at non-positive integer pole".into(),
            });
        }
    }

    if x_approx > 0.0 {
        // Direct Stirling path.
        tracing::debug!(x_approx, "arb_gamma_real: positive argument");
        let log_gamma = stirling_log_gamma(x, prec, rm, cc)?;
        Ok(log_gamma.exp(prec, rm, cc))
    } else {
        // Reflection: Γ(z) = π / (sin(πz) · Γ(1 − z))
        tracing::debug!(x_approx, "arb_gamma_real: reflection formula");
        let one = BigFloat::from_i32(1, prec);
        let one_minus_x = one.sub(x, prec, rm);

        let log_gamma_1mx = stirling_log_gamma(&one_minus_x, prec, rm, cc)?;
        let gamma_1mx = log_gamma_1mx.exp(prec, rm, cc);

        let pi_val = cc.pi(prec, rm).clone();
        let pi_x = pi_val.mul(x, prec, rm);
        let sin_pi_x = pi_x.sin(prec, rm, cc);

        // Guard against the pole (sin(πz) ≈ 0 for integer z).
        if sin_pi_x.is_zero() {
            return Err(SymplexError::Unevaluable {
                reason: "Gamma at non-positive integer pole".into(),
            });
        }
        if let Some(s_exp) = sin_pi_x.exponent()
            && (s_exp as i64) < -(prec as i64 / 2)
        {
            return Err(SymplexError::Unevaluable {
                reason: "Gamma at non-positive integer pole".into(),
            });
        }

        let denom = sin_pi_x.mul(&gamma_1mx, prec, rm);
        let pi_val2 = cc.pi(prec, rm).clone();
        Ok(pi_val2.div(&denom, prec, rm))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Arbitrary-precision erf via Taylor series / asymptotic expansion
// ═══════════════════════════════════════════════════════════════════════════

/// Compute erf(x) at arbitrary precision for real x.
///
/// For small-to-moderate |x|, uses the Taylor series:
///   erf(x) = (2/√π) Σ_{n=0}^{N} (-1)^n x^{2n+1} / (n! (2n+1))
///
/// For large |x|, uses the asymptotic expansion of erfc:
///   erfc(x) = exp(-x²) / (x√π) · Σ_{n=0}^{N} (-1)^n (2n-1)!! / (2x²)^n
///   erf(x) = sign(x) · (1 − erfc(|x|))
fn arb_erf(
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    if x.is_zero() {
        return Ok(BigFloat::new(prec));
    }

    let guard = 32;
    let wp = prec + guard;

    let x_approx = bigfloat_to_f64(x, rm, cc)?;
    // Threshold: for |x| beyond this, the asymptotic expansion converges
    // faster than the Taylor series. Roughly √(wp * ln(2) / 2).
    let threshold = ((wp as f64) * 0.35).sqrt() + 2.0;

    if x_approx.abs() < threshold {
        // ── Taylor series ──────────────────────────────────────────
        // erf(x) = (2/√π) · Σ_{n=0}^{N} prod_n / (2n+1)
        // where prod_0 = x, prod_{n+1} = prod_n · (-x²) / (n+1)
        let neg_x_sq = x.mul(x, wp, rm).neg();
        let mut prod = x.clone(); // (-x²)^n · x / n!
        let mut sum = x.clone(); // accumulator (first term = x)

        let max_terms = (wp as f64 * 0.6) as usize + 60;
        for n in 1..=max_terms {
            // prod *= -x² / n
            prod = prod.mul(&neg_x_sq, wp, rm);
            prod = prod.div(&BigFloat::from_i32(n as i32, wp), wp, rm);

            // term = prod / (2n+1)
            let divisor = BigFloat::from_i32((2 * n + 1) as i32, wp);
            let term = prod.div(&divisor, wp, rm);

            // Convergence check
            if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && (s_exp as i64 - t_exp as i64) > wp as i64
            {
                break;
            }

            sum = sum.add(&term, wp, rm);
        }

        // Multiply by 2/√π
        let two = BigFloat::from_i32(2, wp);
        let pi_val = cc.pi(wp, rm).clone();
        let sqrt_pi = pi_val.sqrt(wp, rm);
        let two_over_sqrt_pi = two.div(&sqrt_pi, wp, rm);

        Ok(sum.mul(&two_over_sqrt_pi, wp, rm))
    } else {
        // ── Asymptotic expansion for large |x| ────────────────────
        // erfc(x) = exp(-x²)/(x√π) · Σ_{n=0}^{N} (-1)^n (2n-1)!! / (2x²)^n
        let is_neg = x.is_negative();
        let ax = x.abs();
        let x_sq = ax.mul(&ax, wp, rm);
        let two_x_sq = x_sq.mul(&BigFloat::from_i32(2, wp), wp, rm);

        let mut sum = BigFloat::from_i32(1, wp);
        let mut term = BigFloat::from_i32(1, wp);
        let max_terms = wp / 2 + 30;

        for n in 1..=max_terms {
            // term *= -(2n-1) / (2x²)
            let factor = BigFloat::from_i32(2 * n as i32 - 1, wp);
            term = term.mul(&factor, wp, rm);
            term = term.div(&two_x_sq, wp, rm);
            term = term.neg();

            // Divergence check: if |term| starts growing, stop
            if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && t_exp > s_exp
            {
                break;
            }

            sum = sum.add(&term, wp, rm);
        }

        let exp_neg_x_sq = x_sq.neg().exp(wp, rm, cc);
        let sqrt_pi = cc.pi(wp, rm).clone().sqrt(wp, rm);
        let x_sqrt_pi = ax.mul(&sqrt_pi, wp, rm);
        let erfc_val = exp_neg_x_sq.mul(&sum, wp, rm).div(&x_sqrt_pi, wp, rm);

        let one = BigFloat::from_i32(1, wp);
        let erf_val = one.sub(&erfc_val, wp, rm);

        if is_neg {
            Ok(erf_val.neg())
        } else {
            Ok(erf_val)
        }
    }
}

/// Arbitrary-precision Lambert W function (principal branch) via Halley iteration.
///
/// Solves w·exp(w) = x for w, using cubic-convergent Halley steps.
fn arb_lambert_w(
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    // Work with extra guard bits for intermediate rounding.
    let wp = prec + 32;

    // Handle x = 0 exactly.
    if x.is_zero() {
        return Ok(BigFloat::new(prec));
    }

    // Handle negative x near -1/e boundary.
    // The principal branch is defined for x >= -1/e.
    let neg_inv_e = {
        let e_val = cc.e(wp, rm).clone();
        let one = BigFloat::from_i32(1, wp);
        one.div(&e_val, wp, rm).neg()
    };

    let diff = x.sub(&neg_inv_e, wp, rm);
    if diff.is_negative() {
        return Err(SymplexError::Unevaluable {
            reason: "LambertW: argument < -1/e, outside principal branch domain".into(),
        });
    }

    // Initial guess.
    let mut w = if x.is_negative() {
        // Near -1/e: use series w ≈ -1 + sqrt(2(1 + ex))
        let e_val = cc.e(wp, rm).clone();
        let ex = e_val.mul(x, wp, rm);
        let one = BigFloat::from_i32(1, wp);
        let two = BigFloat::from_i32(2, wp);
        let inner = one.add(&ex, wp, rm).mul(&two, wp, rm);
        if inner.is_negative() || inner.is_zero() {
            BigFloat::from_i32(-1, wp)
        } else {
            let sq = inner.sqrt(wp, rm);
            BigFloat::from_i32(-1, wp).add(&sq, wp, rm)
        }
    } else {
        // For small positive x, w ≈ x is a good start.
        // For large x, w ≈ ln(x) - ln(ln(x)).
        let threshold = BigFloat::from_f64(2.5, wp);
        if x.sub(&threshold, wp, rm).is_negative() {
            x.clone()
        } else {
            let ln_x = x.ln(wp, rm, cc);
            let ln_ln_x = ln_x.ln(wp, rm, cc);
            ln_x.sub(&ln_ln_x, wp, rm)
        }
    };

    // Halley iteration: cubically convergent.
    //
    // Given f(w) = w·e^w − x, f'(w) = (w+1)·e^w, f''(w) = (w+2)·e^w,
    // the Halley step is:
    //   δ = f / (f' − f·f'' / (2·f'))
    //     = (w·e^w − x) / ((w+1)·e^w − (w+2)·(w·e^w − x) / (2·(w+1)))
    let max_iter = 100;
    for _ in 0..max_iter {
        let ew = w.exp(wp, rm, cc);
        let wew = w.mul(&ew, wp, rm);
        let residual = wew.sub(x, wp, rm); // w·e^w − x

        let w_plus_1 = w.add(&BigFloat::from_i32(1, wp), wp, rm);
        let denom_base = w_plus_1.mul(&ew, wp, rm); // (w+1)·e^w

        // Halley denominator: (w+1)·e^w − (w+2)·residual / (2·(w+1))
        let w_plus_2 = w.add(&BigFloat::from_i32(2, wp), wp, rm);
        let two_w_plus_1 = w_plus_1.mul(&BigFloat::from_i32(2, wp), wp, rm);

        let correction_numer = w_plus_2.mul(&residual, wp, rm);
        let correction = if two_w_plus_1.is_zero() {
            BigFloat::new(wp)
        } else {
            correction_numer.div(&two_w_plus_1, wp, rm)
        };
        let denom = denom_base.sub(&correction, wp, rm);

        if denom.is_zero() {
            break;
        }

        let delta = residual.div(&denom, wp, rm);
        w = w.sub(&delta, wp, rm);

        // Check convergence: |delta| exponent is far below working precision.
        if delta.is_zero() {
            break;
        }
        if let (Some(d_exp), Some(w_e)) = (delta.exponent(), w.exponent())
            && (d_exp as i64) < (w_e as i64) - (wp as i64)
        {
            break;
        }
    }

    Ok(w)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Euler-Mascheroni constant at arbitrary precision (Brent-McMillan B1)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Euler-Mascheroni constant γ at `prec` bits of precision
/// using the Brent-McMillan B1 algorithm.
///
/// This is the fastest known algorithm for computing γ.  It uses the identity
/// involving modified Bessel functions, simplified to a fixed-point summation
/// with convergence rate O(e^{-4n}) where n = 2^p.
///
/// Reference: Brent & McMillan (1980), mpmath `euler_fixed`.
fn arb_euler_gamma(
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    // We need ln(2) at working precision.
    let extra = 30;
    let wp = prec + extra;

    // Choose p such that e^{-4·2^p} < 2^{-wp}, i.e. 4·2^p > wp·ln(2),
    // i.e. p > log2(wp·ln(2)/4).
    let p = ((wp as f64 / 4.0) * std::f64::consts::LN_2).log2().ceil() as u64 + 1;
    let n: i128 = 1i128 << p;
    let n_sq = n * n;

    // ln(2) at working precision.
    let two_bf = BigFloat::from_i32(2, wp);
    let ln2 = two_bf.ln(wp, rm, cc);

    // A = U = -p · ln(2)   (= -ln(n) since n = 2^p)
    let p_bf = BigFloat::from_i128(p as i128, wp);
    let neg_p_ln2 = p_bf.mul(&ln2, wp, rm).neg();

    let mut a = neg_p_ln2.clone();
    let mut u = neg_p_ln2;
    // B = V = 1
    let one = BigFloat::from_i32(1, wp);
    let mut b = one.clone();
    let mut v = one.clone();

    let n_sq_bf = BigFloat::from_i128(n_sq, wp);

    let mut k: i128 = 1;
    loop {
        let k_bf = BigFloat::from_i128(k, wp);
        let k_sq_bf = k_bf.mul(&k_bf, wp, rm);

        // B = B · n² / k²
        b = b.mul(&n_sq_bf, wp, rm).div(&k_sq_bf, wp, rm);

        // A = (A · n² / k + B) / k
        a = a.mul(&n_sq_bf, wp, rm).div(&k_bf, wp, rm);
        a = a.add(&b, wp, rm).div(&k_bf, wp, rm);

        u = u.add(&a, wp, rm);
        v = v.add(&b, wp, rm);

        // Convergence: both A and B are negligibly small.
        let a_small = a.exponent().is_none_or(|e| e < -(wp as i32) + 10);
        let b_small = b.exponent().is_none_or(|e| e < -(wp as i32) + 10);
        if a_small && b_small {
            tracing::trace!(k, "arb_euler_gamma: converged");
            break;
        }

        k += 1;
        if k > 10 * (wp as i128) {
            return Err(SymplexError::ComputationFailed {
                operation: "euler_gamma",
                reason: "Brent-McMillan did not converge".into(),
            });
        }
    }

    // γ = U / V
    Ok(u.div(&v, prec, rm))
}

// ═══════════════════════════════════════════════════════════════════════════
// Hankel P/Q series for Bessel asymptotic expansion
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Hankel auxiliary functions P_ν(z) and Q_ν(z) for the
/// Bessel asymptotic expansion.
///
/// P_ν(z) = Σ_{k=0}^{N} (-1)^k · a_{2k}(ν) / z^{2k}
/// Q_ν(z) = Σ_{k=0}^{N} (-1)^k · a_{2k+1}(ν) / z^{2k+1}
///
/// where a_k(ν) = [(1/2-ν)_k · (1/2+ν)_k] / [(-2)^k · k!]
///
/// Uses optimal truncation: stops when terms start increasing (divergent
/// series).  The error is bounded by the first omitted term for real
/// positive z (Stieltjes bound, DLMF §10.17).
fn hankel_pq(
    order: &BigFloat,
    z: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    _cc: &mut Consts,
) -> (BigFloat, BigFloat) {
    // Compute coefficients a_k incrementally:
    // a_0 = 1
    // a_{k+1} = a_k · -(4ν² - (2k+1)²) / (8(k+1))
    let one = BigFloat::from_i32(1, prec);
    let four = BigFloat::from_i32(4, prec);
    let eight = BigFloat::from_i32(8, prec);

    let four_nu_sq = four.mul(&order.mul(order, prec, rm), prec, rm);

    let z_inv = one.div(z, prec, rm);

    // P and Q accumulators.
    let mut p_sum = one.clone(); // a_0 = 1 contributes to P (even index)
    let mut q_sum = BigFloat::new(prec); // Q starts at 0

    let mut a_k = one.clone(); // a_0 = 1
    let mut z_power = one.clone(); // z^0 = 1

    // Maximum terms: approximately |z| terms before divergence.
    let z_f64 = z.exponent().unwrap_or(0) as f64 * 0.693; // rough |z|
    let max_terms = (2.0 * z_f64.exp() + 20.0).min(10000.0) as usize;

    let mut prev_a_abs: Option<BigFloat> = None;

    for k in 0..max_terms {
        if k > 0 {
            // a_k = a_{k-1} · -(4ν² - (2k-1)²) / (8k)
            let two_km1 = BigFloat::from_i32((2 * k as i32) - 1, prec);
            let two_km1_sq = two_km1.mul(&two_km1, prec, rm);
            let numer = four_nu_sq.sub(&two_km1_sq, prec, rm).neg();
            let k_bf = BigFloat::from_i32(k as i32, prec);
            let denom = eight.mul(&k_bf, prec, rm);
            a_k = a_k.mul(&numer, prec, rm).div(&denom, prec, rm);

            // Update z_power: multiply by 1/z each step.
            z_power = z_power.mul(&z_inv, prec, rm);
        }

        // term = a_k / z^k
        let term = a_k.mul(&z_power, prec, rm);

        // Check for divergence: if |term| > |prev_term|, stop.
        let term_abs = term.abs();
        if let Some(ref prev) = prev_a_abs
            && term_abs.partial_cmp(prev) == Some(std::cmp::Ordering::Greater)
            && k > 2
        {
            tracing::trace!(k, "hankel_pq: optimal truncation (terms diverging)");
            break;
        }

        // Convergence: term negligible relative to accumulated sums.
        if let Some(t_exp) = term.exponent() {
            let p_exp = p_sum.exponent().unwrap_or(0);
            let q_exp = q_sum.exponent().unwrap_or(0);
            let ref_exp = p_exp.max(q_exp);
            if (ref_exp as i64 - t_exp as i64) > prec as i64 {
                tracing::trace!(k, "hankel_pq: converged (term negligible)");
                break;
            }
        }

        prev_a_abs = Some(term_abs);

        // Even k → contributes to P with sign (-1)^(k/2)
        // Odd k → contributes to Q with sign (-1)^((k-1)/2)
        if k % 2 == 0 {
            // P term: (-1)^(k/2) · a_k / z^k
            if (k / 2) % 2 == 0 {
                p_sum = p_sum.add(&term, prec, rm);
            } else {
                p_sum = p_sum.sub(&term, prec, rm);
            }
        } else {
            // Q term: (-1)^((k-1)/2) · a_k / z^k
            if ((k - 1) / 2) % 2 == 0 {
                q_sum = q_sum.add(&term, prec, rm);
            } else {
                q_sum = q_sum.sub(&term, prec, rm);
            }
        }
    }

    (p_sum, q_sum)
}

// ═══════════════════════════════════════════════════════════════════════════
// Bessel function evaluation (arbitrary precision)
// ═══════════════════════════════════════════════════════════════════════════

/// Arbitrary-precision Bessel function of the first kind J_ν(x).
///
/// Uses ascending power series for small |x|:
///   J_ν(x) = Σ_{k=0}^{N} (-1)^k · (x/2)^(ν+2k) / (k! · Γ(ν+k+1))
///
/// For large |x|, uses Hankel asymptotic leading term:
///   J_ν(x) ≈ √(2/(πx)) · cos(x - νπ/2 - π/4)
fn arb_bessel_j(
    order: &BigFloat,
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    let guard = 32;
    let wp = prec + guard;

    // Special case: x = 0.
    if x.is_zero() {
        let order_f64 = bigfloat_to_f64(order, rm, cc)?;
        return if order_f64.abs() < 1e-15 {
            // J_0(0) = 1
            Ok(BigFloat::from_i32(1, prec))
        } else if order_f64 > 0.0 {
            // J_n(0) = 0 for n > 0
            Ok(BigFloat::new(prec))
        } else {
            Err(SymplexError::Unevaluable {
                reason: "BesselJ at x=0 with negative order".into(),
            })
        };
    }

    let x_f64 = bigfloat_to_f64(x, rm, cc)?;
    let order_f64 = bigfloat_to_f64(order, rm, cc)?;

    // Check for integer order (most common case).
    let order_int = order_f64.round() as i64;
    let is_int_order = (order_f64 - order_int as f64).abs() < 1e-12;

    // Negative integer order: J_{-n}(x) = (-1)^n J_n(x).
    if is_int_order && order_int < 0 {
        let pos_order = BigFloat::from_i128((-order_int) as i128, wp);
        let j = arb_bessel_j(&pos_order, x, prec, rm, cc)?;
        return Ok(if order_int % 2 == 0 { j } else { j.neg() });
    }

    // Threshold: use series for |x| < sqrt(wp), asymptotic for larger.
    let threshold = ((wp as f64) * 0.5).sqrt() + 5.0;

    if x_f64.abs() < threshold {
        // ── Ascending series ──────────────────────────────────────
        // J_ν(x) = (x/2)^ν · Σ_{k=0}^N (-1)^k · (x/2)^{2k} / (k! · Γ(ν+k+1))
        let two = BigFloat::from_i32(2, wp);
        let x_half = x.div(&two, wp, rm);
        let x_half_sq = x_half.mul(&x_half, wp, rm);
        let neg_x_half_sq = x_half_sq.neg();

        // Compute (x/2)^ν.  For integer order, use repeated multiplication.
        let prefix = if is_int_order && order_int >= 0 {
            let mut p = BigFloat::from_i32(1, wp);
            for _ in 0..order_int {
                p = p.mul(&x_half, wp, rm);
            }
            p
        } else {
            // General: (x/2)^ν = exp(ν · ln(x/2))
            let ln_xh = x_half.abs().ln(wp, rm, cc);
            let nu_ln = order.mul(&ln_xh, wp, rm);
            nu_ln.exp(wp, rm, cc)
        };

        // Series: sum = Σ (-1)^k · (x/2)^{2k} / (k! · Γ(ν+k+1))
        // We track term = (-x²/4)^k / (k! · Γ(ν+k+1)) incrementally.
        // term_{k+1} = term_k · (-x²/4) / ((k+1) · (ν+k+1))
        let mut sum = BigFloat::new(wp); // will add 1/Γ(ν+1) as first term

        // First term (k=0): 1 / Γ(ν+1)
        let gamma_nu1 = arb_gamma_real(&order.add(&BigFloat::from_i32(1, wp), wp, rm), wp, rm, cc)?;
        let mut term = BigFloat::from_i32(1, wp).div(&gamma_nu1, wp, rm);
        sum = sum.add(&term, wp, rm);

        let max_terms = (wp as f64 * 0.8) as usize + 40;
        for k in 1..=max_terms {
            // term *= (-x²/4) / (k · (ν + k))
            let k_bf = BigFloat::from_i32(k as i32, wp);
            let nu_plus_k = order.add(&k_bf, wp, rm);
            let denom = k_bf.mul(&nu_plus_k, wp, rm);
            term = term.mul(&neg_x_half_sq, wp, rm);
            term = term.div(&denom, wp, rm);

            // Convergence check.
            if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && (s_exp as i64 - t_exp as i64) > wp as i64
            {
                tracing::trace!(k, "arb_bessel_j: series converged");
                break;
            }

            sum = sum.add(&term, wp, rm);
        }

        Ok(prefix.mul(&sum, wp, rm))
    } else {
        // ── Hankel asymptotic with full P/Q series ────────────────
        // J_ν(x) = √(2/(πx)) · [cos(ω)·P_ν(x) - sin(ω)·Q_ν(x)]
        // where ω = x - νπ/2 - π/4
        tracing::trace!("arb_bessel_j: using full Hankel P/Q expansion");
        let (p_val, q_val) = hankel_pq(order, x, wp, rm, cc);
        let pi = cc.pi(wp, rm).clone();
        let two = BigFloat::from_i32(2, wp);
        let four = BigFloat::from_i32(4, wp);

        let two_over_pi_x = two.div(&pi.mul(x, wp, rm), wp, rm);
        let amplitude = two_over_pi_x.sqrt(wp, rm);

        let nu_pi_half = order.mul(&pi, wp, rm).div(&two, wp, rm);
        let pi_quarter = pi.div(&four, wp, rm);
        let phase = x.sub(&nu_pi_half, wp, rm).sub(&pi_quarter, wp, rm);

        let cos_phase = phase.cos(wp, rm, cc);
        let sin_phase = phase.sin(wp, rm, cc);

        // J = amplitude * (cos(ω)·P - sin(ω)·Q)
        let term1 = cos_phase.mul(&p_val, wp, rm);
        let term2 = sin_phase.mul(&q_val, wp, rm);
        Ok(amplitude.mul(&term1.sub(&term2, wp, rm), wp, rm))
    }
}

/// Arbitrary-precision Bessel function of the second kind Y_ν(x).
///
/// For integer order ν = n, uses the Neumann series:
///   Y_n(x) = (2/π)·J_n(x)·[ln(x/2) + γ] - (1/π)·Σ_{k=0}^{n-1} (n-k-1)!/k! · (x/2)^{2k-n}
///             - (1/π)·Σ_{k=0}^∞ [ψ(k+1)+ψ(n+k+1)]·(-1)^k·(x/2)^{n+2k}/(k!·(n+k)!)
///
/// For v1 simplicity: uses asymptotic leading term for large |x| and
/// the relation via J for moderate |x| with the logarithmic series for Y_0.
fn arb_bessel_y(
    order: &BigFloat,
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    let guard = 32;
    let wp = prec + guard;

    if x.is_zero() || x.is_negative() {
        return Err(SymplexError::Unevaluable {
            reason: "BesselY undefined at x ≤ 0".into(),
        });
    }

    let x_f64 = bigfloat_to_f64(x, rm, cc)?;
    let order_f64 = bigfloat_to_f64(order, rm, cc)?;
    let order_int = order_f64.round() as i64;
    let is_int_order = (order_f64 - order_int as f64).abs() < 1e-12;

    // Negative integer order: Y_{-n}(x) = (-1)^n Y_n(x).
    if is_int_order && order_int < 0 {
        let pos_order = BigFloat::from_i128((-order_int) as i128, wp);
        let y = arb_bessel_y(&pos_order, x, prec, rm, cc)?;
        return Ok(if order_int % 2 == 0 { y } else { y.neg() });
    }

    let threshold = ((wp as f64) * 0.5).sqrt() + 5.0;

    if x_f64 >= threshold {
        // ── Hankel asymptotic with full P/Q series ────────────────
        // Y_ν(x) = √(2/(πx)) · [sin(ω)·P_ν(x) + cos(ω)·Q_ν(x)]
        tracing::trace!("arb_bessel_y: using full Hankel P/Q expansion");
        let (p_val, q_val) = hankel_pq(order, x, wp, rm, cc);
        let pi = cc.pi(wp, rm).clone();
        let two = BigFloat::from_i32(2, wp);
        let four = BigFloat::from_i32(4, wp);

        let two_over_pi_x = two.div(&pi.mul(x, wp, rm), wp, rm);
        let amplitude = two_over_pi_x.sqrt(wp, rm);

        let nu_pi_half = order.mul(&pi, wp, rm).div(&two, wp, rm);
        let pi_quarter = pi.div(&four, wp, rm);
        let phase = x.sub(&nu_pi_half, wp, rm).sub(&pi_quarter, wp, rm);

        let sin_phase = phase.sin(wp, rm, cc);
        let cos_phase = phase.cos(wp, rm, cc);

        // Y = amplitude * (sin(ω)·P + cos(ω)·Q)
        let term1 = sin_phase.mul(&p_val, wp, rm);
        let term2 = cos_phase.mul(&q_val, wp, rm);
        return Ok(amplitude.mul(&term1.add(&term2, wp, rm), wp, rm));
    }

    // ── Small |x|, integer order n ≥ 0: Neumann series (A&S 9.1.11) ─────
    //
    //   Y_n(x) = −(1/π)(x/2)^{−n} Σ_{k=0}^{n−1} (n−k−1)!/k! (x²/4)^k
    //            + (2/π) ln(x/2) J_n(x)
    //            − (1/π)(x/2)^n Σ_{k≥0} [ψ(k+1) + ψ(n+k+1)] (−x²/4)^k / (k!(n+k)!)
    //
    // with ψ(m+1) = −γ + H_m.  Computed directly for every n (no forward
    // recurrence or Wronskian division, which is unstable near zeros of J).
    // The log term and the series cancel to O(e^{x}) relative size, so a
    // few extra guard bits are added.

    if !is_int_order {
        return Err(SymplexError::Unevaluable {
            reason: format!(
                "BesselY for non-integer order {order_f64} not yet implemented in small-|x| regime"
            ),
        });
    }
    let n = order_int as usize;
    let wp = wp + cancellation_guard_bits(x_f64);
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, rm);

    let pi = cc.pi(wp, rm).clone();
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let inv_pi = one.div(&pi, wp, rm);
    let x_half = xw.div(&two, wp, rm);
    let x_half_sq = x_half.mul(&x_half, wp, rm);
    let ln_x_half = x_half.ln(wp, rm, cc);
    let euler_gamma = arb_euler_gamma(wp, rm, cc)?;
    let n_bf = BigFloat::from_i128(n as i128, wp);
    let j_n = arb_bessel_j(&n_bf, &xw, wp, rm, cc)?;

    // Finite sum: −(1/π)(x/2)^{−n} Σ_{k<n} (n−k−1)!/k! (x²/4)^k
    let mut finite = BigFloat::new(wp);
    if n > 0 {
        let mut f = Ratio::<BigInt>::from_integer(BigInt::from(1)); // (n−1)!
        for i in 2..n {
            f *= Ratio::from_integer(BigInt::from(i as u64));
        }
        let mut pow = one.clone();
        for k in 0..n {
            if k > 0 {
                f /= Ratio::from_integer(BigInt::from(((n - k) * k) as u64));
                pow = pow.mul(&x_half_sq, wp, rm);
            }
            let coeff = ratio_to_bigfloat(&f, wp, rm);
            finite = finite.add(&coeff.mul(&pow, wp, rm), wp, rm);
        }
        let x_half_pow_neg_n = one.div(&x_half.powi(n, wp, rm), wp, rm);
        finite = inv_pi
            .mul(&x_half_pow_neg_n, wp, rm)
            .mul(&finite, wp, rm)
            .neg();
    }

    // Log term: (2/π) ln(x/2) J_n(x)
    let log_term = two
        .mul(&inv_pi, wp, rm)
        .mul(&ln_x_half, wp, rm)
        .mul(&j_n, wp, rm);

    // Series: −(1/π)(x/2)^n Σ_k [ψ(k+1) + ψ(n+k+1)] (−x²/4)^k / (k!(n+k)!)
    let mut h_k = BigFloat::new(wp);
    let mut h_nk = BigFloat::new(wp);
    for m in 1..=n {
        h_nk = h_nk.add(
            &one.div(&BigFloat::from_i128(m as i128, wp), wp, rm),
            wp,
            rm,
        );
    }
    let mut n_fact = Ratio::<BigInt>::from_integer(BigInt::from(1));
    for i in 2..=n {
        n_fact *= Ratio::from_integer(BigInt::from(i as u64));
    }
    let mut inv_fact = one.div(&ratio_to_bigfloat(&n_fact, wp, rm), wp, rm);
    let neg_x_half_sq = x_half_sq.neg();
    let mut pow = one.clone();
    let mut series = BigFloat::new(wp);
    let two_gamma = euler_gamma.mul(&two, wp, rm);
    let max_terms = (x_f64.abs() * 2.0) as usize + wp + 40;
    for k in 0..=max_terms {
        if k > 0 {
            let k_bf = BigFloat::from_i128(k as i128, wp);
            let nk_bf = BigFloat::from_i128((n + k) as i128, wp);
            h_k = h_k.add(&one.div(&k_bf, wp, rm), wp, rm);
            h_nk = h_nk.add(&one.div(&nk_bf, wp, rm), wp, rm);
            inv_fact = inv_fact.div(&k_bf.mul(&nk_bf, wp, rm), wp, rm);
            pow = pow.mul(&neg_x_half_sq, wp, rm);
        }
        let psi_sum = h_k.add(&h_nk, wp, rm).sub(&two_gamma, wp, rm);
        let term = psi_sum.mul(&pow, wp, rm).mul(&inv_fact, wp, rm);
        series = series.add(&term, wp, rm);
        if k > 2
            && let (Some(t_exp), Some(s_exp)) = (term.exponent(), series.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            tracing::trace!(k, "arb_bessel_y: Neumann series converged");
            break;
        }
    }
    let series_term = inv_pi
        .mul(&x_half.powi(n, wp, rm), wp, rm)
        .mul(&series, wp, rm)
        .neg();

    let y = finite.add(&log_term, wp, rm).add(&series_term, wp, rm);
    Ok(round_to(y, prec, rm))
}

// ═══════════════════════════════════════════════════════════════════════════
// Catalan's constant
// ═══════════════════════════════════════════════════════════════════════════

/// Catalan's constant `G` at `prec` bits via Ramanujan's formula
///
/// ```text
/// G = (π/8)·ln(2 + √3) + (3/8)·Σ_{n≥0} 1 / ((2n+1)² · C(2n, n))
/// ```
///
/// The series gains ~2 bits per term (`1/C(2n,n) ~ 4^{-n}`), so about
/// `prec/2` terms are needed; every term is positive (no cancellation).
fn arb_catalan(prec: usize, rm: RoundingMode, cc: &mut Consts) -> Result<BigFloat, SymplexError> {
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let three = BigFloat::from_i32(3, wp);
    let eight = BigFloat::from_i32(8, wp);

    // (π/8)·ln(2 + √3)
    let sqrt3 = three.sqrt(wp, rm);
    let ln_term = two.add(&sqrt3, wp, rm).ln(wp, rm, cc);
    let pi = cc.pi(wp, rm).clone();
    let first = pi.div(&eight, wp, rm).mul(&ln_term, wp, rm);

    // Σ b_n / (2n+1)² with b_n = 1/C(2n,n), b_{n+1} = b_n·(n+1)/(2(2n+1)).
    let mut b = one.clone();
    let mut sum = BigFloat::new(wp);
    let max_terms = wp / 2 + 64;
    for n in 0..max_terms {
        let two_n_plus_1 = BigFloat::from_i128(2 * n as i128 + 1, wp);
        let denom = two_n_plus_1.mul(&two_n_plus_1, wp, rm);
        let term = b.div(&denom, wp, rm);
        sum = sum.add(&term, wp, rm);
        if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
        let n_plus_1 = BigFloat::from_i128(n as i128 + 1, wp);
        let two_2n_plus_1 = two.mul(&two_n_plus_1, wp, rm);
        b = b.mul(&n_plus_1, wp, rm).div(&two_2n_plus_1, wp, rm);
    }
    let second = three.div(&eight, wp, rm).mul(&sum, wp, rm);
    let mut g = first.add(&second, wp, rm);
    g.set_precision(prec, rm)
        .map_err(|e| SymplexError::ComputationFailed {
            operation: "catalan",
            reason: format!("precision adjustment failed: {e:?}"),
        })?;
    Ok(g)
}

// ═══════════════════════════════════════════════════════════════════════════
// Trigonometric / exponential / logarithmic integrals
// ═══════════════════════════════════════════════════════════════════════════

/// Number of guard bits needed to absorb the cancellation in an
/// alternating power series whose largest term is `~e^{|x|}`.
fn cancellation_guard_bits(x_abs: f64) -> usize {
    (x_abs * std::f64::consts::LOG2_E).ceil() as usize + 32
}

/// Auxiliary functions `f(x)`, `g(x)` of the asymptotic expansions
///
/// ```text
/// Si(x) = π/2 − f(x) cos x − g(x) sin x,   Ci(x) = f(x) sin x − g(x) cos x
/// f(x) ~ (1/x) Σ (−1)^k (2k)!/x^{2k},   g(x) ~ (1/x²) Σ (−1)^k (2k+1)!/x^{2k}
/// ```
///
/// for `x > 0`.  Uses optimal truncation (stop when terms grow); the
/// caller guarantees `x` is large enough that the smallest term is below
/// the working precision.
fn si_ci_asymptotic_fg(x: &BigFloat, wp: usize, rm: RoundingMode) -> (BigFloat, BigFloat) {
    let one = BigFloat::from_i32(1, wp);
    let inv_x = one.div(x, wp, rm);
    let inv_x2 = inv_x.mul(&inv_x, wp, rm);

    let mut f_sum = BigFloat::new(wp);
    let mut g_sum = BigFloat::new(wp);
    // t_f = (2k)!/x^{2k}, t_g = (2k+1)!/x^{2k}
    let mut t_f = one.clone();
    let mut t_g = one.clone();
    let mut prev_f: Option<BigFloat> = None;
    let max_terms = wp * 2 + 100;
    for k in 0..max_terms {
        if k > 0 {
            // (2k)! / (2k-2)! = (2k-1)(2k);  (2k+1)!/(2k-1)! = (2k)(2k+1)
            let a = BigFloat::from_i128((2 * k as i128 - 1) * (2 * k as i128), wp);
            let b = BigFloat::from_i128((2 * k as i128) * (2 * k as i128 + 1), wp);
            t_f = t_f.mul(&a, wp, rm).mul(&inv_x2, wp, rm);
            t_g = t_g.mul(&b, wp, rm).mul(&inv_x2, wp, rm);
        }
        // Optimal truncation: stop once terms start growing.
        if let Some(ref p) = prev_f
            && t_f.abs().cmp(p).is_some_and(|c| c > 0)
        {
            break;
        }
        if k % 2 == 0 {
            f_sum = f_sum.add(&t_f, wp, rm);
            g_sum = g_sum.add(&t_g, wp, rm);
        } else {
            f_sum = f_sum.sub(&t_f, wp, rm);
            g_sum = g_sum.sub(&t_g, wp, rm);
        }
        if let (Some(t_exp), Some(s_exp)) = (t_f.exponent(), f_sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
        prev_f = Some(t_f.abs());
    }
    (f_sum.mul(&inv_x, wp, rm), g_sum.mul(&inv_x2, wp, rm))
}

/// Sine and cosine integrals for real `x`.
///
/// * `want_si == true`: returns `(Si(x), 0)` — valid for all real `x`
///   (`Si` is odd).
/// * `want_si == false`: returns `(0, Ci(x))` — requires `x > 0`.
///
/// Power series for moderate `|x|` (with guard bits for the `e^{|x|}`
/// cancellation) and the Hankel-type asymptotic expansion when
/// `|x| > wp·ln 2` (where its optimal-truncation error `~e^{−x}` is below
/// the working precision).
fn arb_si_ci(
    x: &BigFloat,
    want_si: bool,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<(BigFloat, BigFloat), SymplexError> {
    if x.is_nan() || x.is_inf_pos() || x.is_inf_neg() {
        return Err(SymplexError::Unevaluable {
            reason: "Si/Ci of special float value".into(),
        });
    }
    let x_f64 = bigfloat_to_f64(x, rm, cc)?;
    let x_abs = x_f64.abs();
    let base_wp = prec + 32;

    if x.is_zero() {
        return if want_si {
            Ok((BigFloat::new(prec), BigFloat::new(prec)))
        } else {
            Err(SymplexError::Unevaluable {
                reason: "Ci(0) is -∞".into(),
            })
        };
    }
    if !want_si && x.is_negative() {
        return Err(SymplexError::Unevaluable {
            reason: "Ci requires a positive argument here".into(),
        });
    }

    let switch = (base_wp as f64) * std::f64::consts::LN_2;
    if x_abs > switch {
        // ── Asymptotic expansion ──
        let wp = base_wp;
        let ax = x.abs();
        let (f, g) = si_ci_asymptotic_fg(&ax, wp, rm);
        let sin_x = ax.sin(wp, rm, cc);
        let cos_x = ax.cos(wp, rm, cc);
        let pi = cc.pi(wp, rm).clone();
        let two = BigFloat::from_i32(2, wp);
        let half_pi = pi.div(&two, wp, rm);
        let mut si =
            half_pi
                .sub(&f.mul(&cos_x, wp, rm), wp, rm)
                .sub(&g.mul(&sin_x, wp, rm), wp, rm);
        if x.is_negative() {
            si = si.neg();
        }
        let ci = f.mul(&sin_x, wp, rm).sub(&g.mul(&cos_x, wp, rm), wp, rm);
        return Ok((round_to(si, prec, rm), round_to(ci, prec, rm)));
    }

    // ── Power series ──
    let wp = base_wp + cancellation_guard_bits(x_abs);
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, rm);
    let x2 = xw.mul(&xw, wp, rm);
    let max_terms = (x_abs * 1.5) as usize + wp + 40;

    let si = if want_si {
        // Si(x) = Σ (−1)^k t_k/(2k+1),  t_k = x^{2k+1}/(2k+1)!
        let mut t = xw.clone();
        let mut sum = BigFloat::new(wp);
        for k in 0..max_terms {
            if k > 0 {
                let d = BigFloat::from_i128((2 * k as i128) * (2 * k as i128 + 1), wp);
                t = t.mul(&x2, wp, rm).div(&d, wp, rm);
            }
            let term = t.div(&BigFloat::from_i128(2 * k as i128 + 1, wp), wp, rm);
            if k % 2 == 0 {
                sum = sum.add(&term, wp, rm);
            } else {
                sum = sum.sub(&term, wp, rm);
            }
            if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && (s_exp as i64 - t_exp as i64) > wp as i64
            {
                break;
            }
        }
        round_to(sum, prec, rm)
    } else {
        BigFloat::new(prec)
    };

    let ci = if want_si {
        BigFloat::new(prec)
    } else {
        // Ci(x) = γ + ln x + Σ_{k≥1} (−1)^k u_k/(2k),  u_k = x^{2k}/(2k)!
        let gamma = arb_euler_gamma(wp, rm, cc)?;
        let ln_x = xw.ln(wp, rm, cc);
        let mut sum = BigFloat::new(wp);
        let mut u = BigFloat::from_i32(1, wp);
        for k in 1..max_terms {
            let d = BigFloat::from_i128((2 * k as i128 - 1) * (2 * k as i128), wp);
            u = u.mul(&x2, wp, rm).div(&d, wp, rm);
            let term = u.div(&BigFloat::from_i128(2 * k as i128, wp), wp, rm);
            if k % 2 == 0 {
                sum = sum.add(&term, wp, rm);
            } else {
                sum = sum.sub(&term, wp, rm);
            }
            if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && (s_exp as i64 - t_exp as i64) > wp as i64
            {
                break;
            }
        }
        let ci = gamma.add(&ln_x, wp, rm).add(&sum, wp, rm);
        round_to(ci, prec, rm)
    };

    Ok((si, ci))
}

/// Exponential integral `Ei(x)` for real `x ≠ 0`.
///
/// * `|x| ≤ wp·ln 2`: `Ei(x) = γ + ln|x| + Σ_{k≥1} xᵏ/(k·k!)`, with guard
///   bits for the alternating cancellation when `x < 0`;
/// * otherwise the asymptotic expansion `Ei(x) ~ (eˣ/x) Σ k!/xᵏ` with
///   optimal truncation (relative error `~e^{−|x|}`).
fn arb_ei(
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    if x.is_nan() || x.is_inf_pos() || x.is_inf_neg() || x.is_zero() {
        return Err(SymplexError::Unevaluable {
            reason: "Ei of zero or special float value".into(),
        });
    }
    let x_f64 = bigfloat_to_f64(x, rm, cc)?;
    let x_abs = x_f64.abs();
    let base_wp = prec + 32;
    let switch = (base_wp as f64) * std::f64::consts::LN_2;

    if x_abs > switch {
        // ── Asymptotic: Ei(x) ~ (e^x / x) Σ_{k≥0} k! / x^k ──
        let wp = base_wp;
        let one = BigFloat::from_i32(1, wp);
        let inv_x = one.div(x, wp, rm);
        let mut term = one.clone();
        let mut sum = BigFloat::new(wp);
        let mut prev: Option<BigFloat> = None;
        let max_terms = wp * 2 + 100;
        for k in 0..max_terms {
            if k > 0 {
                term = term
                    .mul(&BigFloat::from_i128(k as i128, wp), wp, rm)
                    .mul(&inv_x, wp, rm);
            }
            if let Some(ref p) = prev
                && term.abs().cmp(p).is_some_and(|c| c > 0)
            {
                break;
            }
            sum = sum.add(&term, wp, rm);
            if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && (s_exp as i64 - t_exp as i64) > wp as i64
            {
                break;
            }
            prev = Some(term.abs());
        }
        let ex = x.exp(wp, rm, cc);
        let r = ex.mul(&inv_x, wp, rm).mul(&sum, wp, rm);
        return Ok(round_to(r, prec, rm));
    }

    // ── Power series ──
    let wp = if x.is_negative() {
        base_wp + cancellation_guard_bits(x_abs)
    } else {
        base_wp
    };
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, rm);
    let gamma = arb_euler_gamma(wp, rm, cc)?;
    let ln_abs_x = xw.abs().ln(wp, rm, cc);
    let mut v = BigFloat::from_i32(1, wp); // x^k/k!
    let mut sum = BigFloat::new(wp);
    let max_terms = (x_abs * 3.0) as usize + wp + 40;
    for k in 1..max_terms {
        let k_bf = BigFloat::from_i128(k as i128, wp);
        v = v.mul(&xw, wp, rm).div(&k_bf, wp, rm);
        let term = v.div(&k_bf, wp, rm);
        sum = sum.add(&term, wp, rm);
        if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
            && (k as f64) > x_abs
        {
            break;
        }
    }
    let r = gamma.add(&ln_abs_x, wp, rm).add(&sum, wp, rm);
    Ok(round_to(r, prec, rm))
}

/// Round a `BigFloat` down to `prec` bits (no-op on failure).
fn round_to(mut v: BigFloat, prec: usize, rm: RoundingMode) -> BigFloat {
    let _ = v.set_precision(prec, rm);
    v
}

/// `b^e` for real `b > 0` (or integer `e`), computed as `powi` for integer
/// exponents and `exp(e·ln b)` otherwise.
///
/// This deliberately avoids [`BigFloat::pow`], whose correct-rounding loop
/// never terminates when the exact result is representable in binary
/// (e.g. `4^{1/2} = 2`), which is exactly what happens for the integer
/// bases used in series like Borwein's `ζ` algorithm.
fn bf_pow(b: &BigFloat, e: &BigFloat, wp: usize, rm: RoundingMode, cc: &mut Consts) -> BigFloat {
    if e.is_zero() {
        return BigFloat::from_i32(1, wp);
    }
    if e.is_int()
        && let Ok(ef) = bigfloat_to_f64(e, rm, cc)
        && ef.abs() < 1.0e9
    {
        let n = ef.abs() as usize;
        let p = b.powi(n, wp, rm);
        return if ef < 0.0 {
            BigFloat::from_i32(1, wp).div(&p, wp, rm)
        } else {
            p
        };
    }
    let ln_b = b.ln(wp + 32, rm, cc);
    ln_b.mul(e, wp + 32, rm).exp(wp, rm, cc)
}

// ═══════════════════════════════════════════════════════════════════════════
// Riemann zeta function
// ═══════════════════════════════════════════════════════════════════════════

/// `ζ(s)` for real `s ≠ 1`.
///
/// * `s > 0`: Borwein's algorithm (P. Borwein, *An efficient algorithm for
///   the Riemann zeta function*, 1991, Algorithm 2) — an accelerated
///   alternating series with error `≤ 3/(3+√8)ⁿ`, so `n ≈ 0.39·wp` terms.
/// * `s < 0`: functional equation
///   `ζ(s) = 2ˢ π^{s−1} sin(πs/2) Γ(1−s) ζ(1−s)`; trivial zeros at the
///   negative even integers are returned exactly.
/// * `s = 0`: `−1/2`.
fn arb_zeta(
    s: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    if s.is_nan() || s.is_inf_neg() {
        return Err(SymplexError::Unevaluable {
            reason: "zeta of special float value".into(),
        });
    }
    if s.is_inf_pos() {
        return Ok(BigFloat::from_i32(1, prec));
    }
    if s.is_zero() {
        return Ok(BigFloat::from_f64(-0.5, prec));
    }
    let s_f64 = bigfloat_to_f64(s, rm, cc)?;
    if (s_f64 - 1.0).abs() < 1e-300 {
        return Err(SymplexError::Unevaluable {
            reason: "zeta(1) is a pole".into(),
        });
    }
    let wp = prec + 32;

    if s.is_negative() {
        // Trivial zeros.
        let s_round = s_f64.round();
        if (s_f64 - s_round).abs() < 1e-14 && (s_round as i64) % 2 == 0 {
            let mut sw = s.clone();
            let _ = sw.set_precision(wp, rm);
            let diff = sw.sub(&BigFloat::from_f64(s_round, wp), wp, rm);
            if diff.is_zero() {
                return Ok(BigFloat::new(prec));
            }
        }
        // ζ(s) = 2^s π^{s−1} sin(π s/2) Γ(1−s) ζ(1−s)
        let mut sw = s.clone();
        let _ = sw.set_precision(wp, rm);
        let one = BigFloat::from_i32(1, wp);
        let two = BigFloat::from_i32(2, wp);
        let pi = cc.pi(wp, rm).clone();
        let one_minus_s = one.sub(&sw, wp, rm);
        let two_pow_s = bf_pow(&two, &sw, wp, rm, cc);
        let s_minus_1 = sw.sub(&one, wp, rm);
        let pi_pow = bf_pow(&pi, &s_minus_1, wp, rm, cc);
        let half_pi_s = pi.mul(&sw, wp, rm).div(&two, wp, rm);
        let sin_term = half_pi_s.sin(wp, rm, cc);
        let gamma_term = arb_gamma_real(&one_minus_s, wp, rm, cc)?;
        let zeta_term = arb_zeta_borwein(&one_minus_s, wp, rm, cc)?;
        let r = two_pow_s
            .mul(&pi_pow, wp, rm)
            .mul(&sin_term, wp, rm)
            .mul(&gamma_term, wp, rm)
            .mul(&zeta_term, wp, rm);
        return Ok(round_to(r, prec, rm));
    }

    let r = arb_zeta_borwein(s, wp, rm, cc)?;
    Ok(round_to(r, prec, rm))
}

/// Borwein's Algorithm 2 for `ζ(s)`, real `s > 0`, `s ≠ 1`.
///
/// ```text
/// d_k = n Σ_{i=0}^{k} (n+i−1)! 4ⁱ / ((n−i)! (2i)!)
/// ζ(s) = − 1/(d_n (1 − 2^{1−s})) · Σ_{k=0}^{n−1} (−1)ᵏ (d_k − d_n)/(k+1)ˢ
/// ```
fn arb_zeta_borwein(
    s: &BigFloat,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    // n such that (3+√8)^{-n} < 2^{-wp}:  n > wp·ln2/ln(3+√8) ≈ 0.393·wp
    let n = ((wp as f64) * std::f64::consts::LN_2 / (3.0 + 8f64.sqrt()).ln()).ceil() as usize + 4;

    // Exact integer coefficients d_k via rational arithmetic.
    let mut d: Vec<Ratio<BigInt>> = Vec::with_capacity(n + 1);
    let mut acc = Ratio::<BigInt>::zero();
    // term_i = n · (n+i−1)! · 4^i / ((n−i)! (2i)!)
    // term_0 = n · (n−1)!/n! = 1
    let mut term = Ratio::from_integer(BigInt::from(1));
    for i in 0..=n {
        if i > 0 {
            // term_i / term_{i−1} = (n+i−1)·4·(n−i+1) / ((2i−1)(2i))
            let numer = BigInt::from((n + i - 1) as u64)
                * BigInt::from(4u64)
                * BigInt::from((n - i + 1) as u64);
            let denom = BigInt::from((2 * i - 1) as u64) * BigInt::from((2 * i) as u64);
            term *= Ratio::new(numer, denom);
        }
        acc += &term;
        d.push(acc.clone());
    }
    let d_n = ratio_to_bigfloat(&d[n], wp, rm);

    // S = Σ_{k=0}^{n−1} (−1)^k (d_k − d_n) / (k+1)^s
    let mut sum = BigFloat::new(wp);
    let s_is_int = s.is_int();
    let s_int = if s_is_int {
        bigfloat_to_f64(s, rm, cc)?.round() as usize
    } else {
        0
    };
    for k in 0..n {
        let coeff = ratio_to_bigfloat(&(&d[k] - &d[n]), wp, rm);
        let kp1 = BigFloat::from_i128(k as i128 + 1, wp);
        let kp1_pow_s = if s_is_int {
            kp1.powi(s_int, wp, rm)
        } else {
            bf_pow(&kp1, s, wp, rm, cc)
        };
        let term = coeff.div(&kp1_pow_s, wp, rm);
        if k % 2 == 0 {
            sum = sum.add(&term, wp, rm);
        } else {
            sum = sum.sub(&term, wp, rm);
        }
    }

    // 1 − 2^{1−s}
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let one_minus_s = one.sub(s, wp, rm);
    let two_pow = bf_pow(&two, &one_minus_s, wp, rm, cc);
    let denom_factor = one.sub(&two_pow, wp, rm);
    if denom_factor.is_zero() {
        return Err(SymplexError::Unevaluable {
            reason: "zeta(1) is a pole".into(),
        });
    }
    let denom = d_n.mul(&denom_factor, wp, rm);
    Ok(sum.div(&denom, wp, rm).neg())
}

// ═══════════════════════════════════════════════════════════════════════════
// Polygamma
// ═══════════════════════════════════════════════════════════════════════════

/// `ψ⁽ⁿ⁾(x)` for integer `n ≥ 1` and real `x` (not a non-positive integer).
///
/// 1. Recurrence `ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(x+N) − (−1)ⁿ n! Σ_{k<N} 1/(x+k)^{n+1}`
///    shifts the argument to `x + N ≥ max(wp/3, n) + 10`.
/// 2. Asymptotic series (DLMF 5.15.8):
///    `ψ⁽ⁿ⁾(x) ~ (−1)^{n+1} [ (n−1)!/xⁿ + n!/(2x^{n+1}) + Σ_{k≥1} B₂ₖ (2k+n−1)!/((2k)! x^{2k+n}) ]`
///    with exact Bernoulli numbers, truncated when the terms fall below
///    the working precision (or start to diverge).
fn arb_polygamma(
    n: u32,
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    _cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    if x.is_nan() || x.is_inf_pos() || x.is_inf_neg() {
        return Err(SymplexError::Unevaluable {
            reason: "polygamma of special float value".into(),
        });
    }
    let guard = 32;
    let wp = prec + guard;
    let n_us = n as usize;
    let one = BigFloat::from_i32(1, wp);

    // n! and (n−1)! as BigFloats.
    let mut n_fact = Ratio::<BigInt>::from_integer(BigInt::from(1));
    for i in 2..=n_us {
        n_fact *= Ratio::from_integer(BigInt::from(i as u64));
    }
    let nm1_fact = &n_fact / Ratio::from_integer(BigInt::from(n_us as u64));
    let n_fact_bf = ratio_to_bigfloat(&n_fact, wp, rm);
    let nm1_fact_bf = ratio_to_bigfloat(&nm1_fact, wp, rm);
    // (−1)^{n+1}
    let outer_sign_neg = n_us.is_multiple_of(2);

    // ── Step 1: shift x upward ──
    let threshold_val = ((wp / 3).max(n_us) + 10) as i128;
    let threshold = BigFloat::from_i128(threshold_val, wp);
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, rm);
    let mut shift_sum = BigFloat::new(wp); // Σ 1/(x+k)^{n+1}
    let mut steps: usize = 0;
    while xw.sub(&threshold, wp, rm).is_negative() {
        if xw.is_zero() {
            return Err(SymplexError::Unevaluable {
                reason: "polygamma at non-positive integer pole".into(),
            });
        }
        if let Some(x_exp) = xw.exponent()
            && (x_exp as i64) < -(wp as i64 / 2)
        {
            return Err(SymplexError::Unevaluable {
                reason: "polygamma at non-positive integer pole".into(),
            });
        }
        let p = xw.powi(n_us + 1, wp, rm);
        let inv = one.div(&p, wp, rm);
        shift_sum = shift_sum.add(&inv, wp, rm);
        xw = xw.add(&one, wp, rm);
        steps += 1;
        if steps > 10 * wp + 1_000_000 {
            return Err(SymplexError::ComputationFailed {
                operation: "polygamma",
                reason: "argument shift did not terminate".into(),
            });
        }
    }
    // ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(x+N) + (−1)^{n+1} n! Σ 1/(x+k)^{n+1}
    let shift_term = n_fact_bf.mul(&shift_sum, wp, rm);

    // ── Step 2: asymptotic series at xw ──
    // bracket = (n−1)!/x^n + n!/(2 x^{n+1}) + Σ B_{2k} (2k+n−1)!/((2k)! x^{2k+n})
    let x_pow_n = xw.powi(n_us, wp, rm);
    let x_pow_np1 = x_pow_n.mul(&xw, wp, rm);
    let two = BigFloat::from_i32(2, wp);
    let mut bracket = nm1_fact_bf.div(&x_pow_n, wp, rm);
    bracket = bracket.add(&n_fact_bf.div(&two.mul(&x_pow_np1, wp, rm), wp, rm), wp, rm);

    let x2 = xw.mul(&xw, wp, rm);
    let mut x_pow = x_pow_n.mul(&x2, wp, rm); // x^{n+2} for k = 1
    // (2k+n−1)!/(2k)! as a running rational: start k=1 → (n+1)!/2!
    let mut ratio_fact = &n_fact * Ratio::from_integer(BigInt::from(n_us as u64 + 1))
        / Ratio::from_integer(BigInt::from(2));
    let mut prev_term_exp: Option<i32> = None;
    let max_terms = wp / 2 + 20;
    for k in 1..=max_terms {
        if k > 1 {
            // multiply by (2k+n−2)(2k+n−1) / ((2k−1)(2k))
            let a =
                BigInt::from((2 * k + n_us - 2) as u64) * BigInt::from((2 * k + n_us - 1) as u64);
            let b = BigInt::from((2 * k - 1) as u64) * BigInt::from((2 * k) as u64);
            ratio_fact *= Ratio::new(a, b);
            x_pow = x_pow.mul(&x2, wp, rm);
        }
        let b2k = crate::base::bernoulli::bernoulli(2 * k);
        let coeff = ratio_to_bigfloat(&(&b2k * &ratio_fact), wp, rm);
        let term = coeff.div(&x_pow, wp, rm);
        if let Some(t_exp) = term.exponent() {
            if let Some(prev) = prev_term_exp
                && t_exp > prev + 10
            {
                break; // diverging
            }
            prev_term_exp = Some(t_exp);
            if let Some(b_exp) = bracket.exponent()
                && (b_exp as i64 - t_exp as i64) > wp as i64
            {
                bracket = bracket.add(&term, wp, rm);
                break;
            }
        }
        bracket = bracket.add(&term, wp, rm);
    }

    let total = bracket.add(&shift_term, wp, rm);
    let result = if outer_sign_neg { total.neg() } else { total };
    Ok(round_to(result, prec, rm))
}

// ═══════════════════════════════════════════════════════════════════════════
// Modified Bessel functions
// ═══════════════════════════════════════════════════════════════════════════

/// Ascending series for `I_ν(x)` at working precision `wp`:
/// `I_ν(x) = (x/2)^ν Σ_{k≥0} (x²/4)^k / (k! Γ(ν+k+1))`.
///
/// All terms are positive for `x > 0`, so no cancellation occurs; the
/// number of terms is `O(|x| + wp)`.
fn bessel_i_series(
    order: &BigFloat,
    x: &BigFloat,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    let two = BigFloat::from_i32(2, wp);
    let x_half = x.div(&two, wp, rm);
    let x_half_sq = x_half.mul(&x_half, wp, rm);
    let order_f64 = bigfloat_to_f64(order, rm, cc)?;
    let order_int = order_f64.round() as i64;
    let is_int_order = (order_f64 - order_int as f64).abs() < 1e-12;

    if x.is_zero() {
        return if is_int_order && order_int == 0 {
            Ok(BigFloat::from_i32(1, wp))
        } else if order_f64 > 0.0 {
            Ok(BigFloat::new(wp))
        } else {
            Err(SymplexError::Unevaluable {
                reason: "BesselI at x=0 with negative order".into(),
            })
        };
    }

    // (x/2)^ν
    let prefix = if is_int_order && order_int >= 0 {
        x_half.powi(order_int as usize, wp, rm)
    } else if is_int_order {
        // Negative integer order: I_{-n} = I_n.
        x_half.powi((-order_int) as usize, wp, rm)
    } else {
        if x.is_negative() {
            return Err(SymplexError::Unevaluable {
                reason: "BesselI of negative argument with non-integer order is complex".into(),
            });
        }
        let ln_xh = x_half.ln(wp, rm, cc);
        order.mul(&ln_xh, wp, rm).exp(wp, rm, cc)
    };
    let eff_order = if is_int_order && order_int < 0 {
        BigFloat::from_i128((-order_int) as i128, wp)
    } else {
        order.clone()
    };

    // term_0 = 1/Γ(ν+1); term_{k+1} = term_k · (x²/4) / ((k+1)(ν+k+1))
    let one = BigFloat::from_i32(1, wp);
    let gamma_nu1 = arb_gamma_real(&eff_order.add(&one, wp, rm), wp, rm, cc)?;
    let mut term = one.div(&gamma_nu1, wp, rm);
    let mut sum = term.clone();
    let x_f64 = bigfloat_to_f64(x, rm, cc)?.abs();
    let max_terms = (x_f64 * 2.0) as usize + wp + 40;
    for k in 1..=max_terms {
        let k_bf = BigFloat::from_i128(k as i128, wp);
        let nu_plus_k = eff_order.add(&k_bf, wp, rm);
        let denom = k_bf.mul(&nu_plus_k, wp, rm);
        term = term.mul(&x_half_sq, wp, rm).div(&denom, wp, rm);
        sum = sum.add(&term, wp, rm);
        if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
    }
    Ok(prefix.mul(&sum, wp, rm))
}

/// Modified Bessel function of the first kind `I_ν(x)` (real `ν`, `x`).
fn arb_bessel_i(
    order: &BigFloat,
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    let wp = prec + 32;
    let r = bessel_i_series(order, x, wp, rm, cc)?;
    Ok(round_to(r, prec, rm))
}

/// Asymptotic expansion for `K_ν(x)`, large `x > 0` (DLMF 10.40.2):
/// `K_ν(x) ~ √(π/(2x)) e^{−x} Σ_k a_k(ν)/x^k`,
/// `a_k(ν) = ∏_{j=1}^{k} (4ν² − (2j−1)²) / (k! 8^k)`.
fn bessel_k_asymptotic(
    order: &BigFloat,
    x: &BigFloat,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> BigFloat {
    let one = BigFloat::from_i32(1, wp);
    let four = BigFloat::from_i32(4, wp);
    let eight = BigFloat::from_i32(8, wp);
    let four_nu_sq = four.mul(&order.mul(order, wp, rm), wp, rm);
    let inv_x = one.div(x, wp, rm);
    let mut a_k = one.clone();
    let mut sum = one.clone();
    let mut prev: Option<BigFloat> = None;
    let max_terms = wp * 2 + 100;
    for k in 1..max_terms {
        let two_km1 = BigFloat::from_i32((2 * k as i32) - 1, wp);
        let numer = four_nu_sq.sub(&two_km1.mul(&two_km1, wp, rm), wp, rm);
        let denom = eight.mul(&BigFloat::from_i32(k as i32, wp), wp, rm);
        a_k = a_k
            .mul(&numer, wp, rm)
            .div(&denom, wp, rm)
            .mul(&inv_x, wp, rm);
        if let Some(ref p) = prev
            && a_k.abs().cmp(p).is_some_and(|c| c > 0)
        {
            break;
        }
        sum = sum.add(&a_k, wp, rm);
        if let (Some(t_exp), Some(s_exp)) = (a_k.exponent(), sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
        prev = Some(a_k.abs());
    }
    let pi = cc.pi(wp, rm).clone();
    let two = BigFloat::from_i32(2, wp);
    let amp = pi.div(&two.mul(x, wp, rm), wp, rm).sqrt(wp, rm);
    let e_neg_x = x.neg().exp(wp, rm, cc);
    amp.mul(&e_neg_x, wp, rm).mul(&sum, wp, rm)
}

/// Modified Bessel function of the second kind `K_ν(x)`, `x > 0`.
///
/// * Large `x` (`2x > wp·ln 2`): asymptotic expansion.
/// * Integer order `n` (A&S 9.6.11):
///   `K_n(x) = ½(x/2)^{−n} Σ_{k<n} (n−k−1)!/k! (−x²/4)^k + (−1)^{n+1} ln(x/2) I_n(x)
///           + (−1)^n ½ (x/2)^n Σ_{k≥0} [ψ(k+1) + ψ(n+k+1)] (x²/4)^k/(k!(n+k)!)`
///   with `ψ(m+1) = −γ + H_m`.
/// * Non-integer order: `K_ν = (π/2)(I_{−ν} − I_ν)/sin(νπ)`.
///
/// Both series formulas suffer `e^{x}`-scale cancellation (the result is
/// `~e^{−x}`), which is absorbed by extra guard bits.
fn arb_bessel_k(
    order: &BigFloat,
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    if x.is_zero() || x.is_negative() || x.is_nan() {
        return Err(SymplexError::Unevaluable {
            reason: "BesselK undefined at x ≤ 0".into(),
        });
    }
    let base_wp = prec + 32;
    let x_f64 = bigfloat_to_f64(x, rm, cc)?;
    let order_f64 = bigfloat_to_f64(order, rm, cc)?;
    let order_int = order_f64.round() as i64;
    let is_int_order = (order_f64 - order_int as f64).abs() < 1e-12;
    let order_abs = if is_int_order {
        BigFloat::from_i128(order_int.unsigned_abs() as i128, base_wp)
    } else {
        order.abs()
    };

    // Large x: asymptotic (relative truncation error ~ e^{-2x}).
    if 2.0 * x_f64 > (base_wp as f64) * std::f64::consts::LN_2 {
        let r = bessel_k_asymptotic(&order_abs, x, base_wp, rm, cc);
        return Ok(round_to(r, prec, rm));
    }

    // Series: guard against e^{x} cancellation (two exponentially large
    // terms of opposite sign nearly cancel).
    let wp = base_wp + 2 * cancellation_guard_bits(x_f64);
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, rm);

    if !is_int_order {
        // K_ν = (π/2)(I_{−ν} − I_ν)/sin(νπ)
        let neg_order = order.neg();
        let i_neg = bessel_i_series(&neg_order, &xw, wp, rm, cc)?;
        let i_pos = bessel_i_series(order, &xw, wp, rm, cc)?;
        let pi = cc.pi(wp, rm).clone();
        let two = BigFloat::from_i32(2, wp);
        let sin_nu_pi = order.mul(&pi, wp, rm).sin(wp, rm, cc);
        let r = pi
            .div(&two, wp, rm)
            .mul(&i_neg.sub(&i_pos, wp, rm), wp, rm)
            .div(&sin_nu_pi, wp, rm);
        return Ok(round_to(r, prec, rm));
    }

    let n = order_int.unsigned_abs() as usize;
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let half = one.div(&two, wp, rm);
    let x_half = xw.div(&two, wp, rm);
    let x_half_sq = x_half.mul(&x_half, wp, rm);
    let ln_x_half = x_half.ln(wp, rm, cc);
    let gamma = arb_euler_gamma(wp, rm, cc)?;
    let n_bf = BigFloat::from_i128(n as i128, wp);
    let i_n = bessel_i_series(&n_bf, &xw, wp, rm, cc)?;

    // Finite sum: ½ (x/2)^{-n} Σ_{k=0}^{n-1} (n−k−1)!/k! (−x²/4)^k
    let mut finite = BigFloat::new(wp);
    if n > 0 {
        // f_k = (n−k−1)!/k! ; f_0 = (n−1)!
        let mut f = Ratio::<BigInt>::from_integer(BigInt::from(1));
        for i in 2..n {
            f *= Ratio::from_integer(BigInt::from(i as u64));
        }
        let mut pow = one.clone(); // (−x²/4)^k
        let neg_x_half_sq = x_half_sq.neg();
        for k in 0..n {
            if k > 0 {
                // f_k = f_{k−1} / ((n−k) · k)
                f /= Ratio::from_integer(BigInt::from(((n - k) * k) as u64));
                pow = pow.mul(&neg_x_half_sq, wp, rm);
            }
            let coeff = ratio_to_bigfloat(&f, wp, rm);
            finite = finite.add(&coeff.mul(&pow, wp, rm), wp, rm);
        }
        let x_half_pow_neg_n = one.div(&x_half.powi(n, wp, rm), wp, rm);
        finite = half.mul(&x_half_pow_neg_n, wp, rm).mul(&finite, wp, rm);
    }

    // Log term: (−1)^{n+1} ln(x/2) I_n(x)
    let log_term = ln_x_half.mul(&i_n, wp, rm);
    let log_term = if n.is_multiple_of(2) {
        log_term.neg()
    } else {
        log_term
    };

    // Infinite sum: (−1)^n ½ (x/2)^n Σ_k [ψ(k+1) + ψ(n+k+1)] (x²/4)^k/(k!(n+k)!)
    // ψ(m+1) = −γ + H_m.
    let mut h_k = BigFloat::new(wp); // H_0 = 0
    let mut h_nk = BigFloat::new(wp); // H_n
    for m in 1..=n {
        h_nk = h_nk.add(
            &one.div(&BigFloat::from_i128(m as i128, wp), wp, rm),
            wp,
            rm,
        );
    }
    // 1/(k!(n+k)!) start: 1/n!
    let mut n_fact = Ratio::<BigInt>::from_integer(BigInt::from(1));
    for i in 2..=n {
        n_fact *= Ratio::from_integer(BigInt::from(i as u64));
    }
    let mut inv_fact = one.div(&ratio_to_bigfloat(&n_fact, wp, rm), wp, rm);
    let mut pow = one.clone();
    let mut series = BigFloat::new(wp);
    let two_gamma = gamma.mul(&two, wp, rm);
    let max_terms = (x_f64 * 2.0) as usize + wp + 40;
    for k in 0..=max_terms {
        if k > 0 {
            let k_bf = BigFloat::from_i128(k as i128, wp);
            let nk_bf = BigFloat::from_i128((n + k) as i128, wp);
            h_k = h_k.add(&one.div(&k_bf, wp, rm), wp, rm);
            h_nk = h_nk.add(&one.div(&nk_bf, wp, rm), wp, rm);
            inv_fact = inv_fact.div(&k_bf.mul(&nk_bf, wp, rm), wp, rm);
            pow = pow.mul(&x_half_sq, wp, rm);
        }
        let psi_sum = h_k.add(&h_nk, wp, rm).sub(&two_gamma, wp, rm);
        let term = psi_sum.mul(&pow, wp, rm).mul(&inv_fact, wp, rm);
        series = series.add(&term, wp, rm);
        if k > 2
            && let (Some(t_exp), Some(s_exp)) = (term.exponent(), series.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
    }
    let series_term = half
        .mul(&x_half.powi(n, wp, rm), wp, rm)
        .mul(&series, wp, rm);
    let series_term = if n.is_multiple_of(2) {
        series_term
    } else {
        series_term.neg()
    };

    let r = finite.add(&log_term, wp, rm).add(&series_term, wp, rm);
    Ok(round_to(r, prec, rm))
}

// ═══════════════════════════════════════════════════════════════════════════
// Orthogonal polynomials
// ═══════════════════════════════════════════════════════════════════════════

/// Families of classical orthogonal polynomials evaluated numerically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrthoPoly {
    Legendre,
    ChebyshevT,
    ChebyshevU,
    Hermite,
    Laguerre,
}

/// Evaluate `P_n(x)` (or `T_n`, `U_n`, `H_n`, `L_n`) for any non-negative
/// integer `n` via the three-term recurrence:
///
/// * Legendre: `(k+1) P_{k+1} = (2k+1) x P_k − k P_{k−1}`
/// * Chebyshev T/U: `T_{k+1} = 2x T_k − T_{k−1}` (`T_1 = x`, `U_1 = 2x`)
/// * Hermite (physicists'): `H_{k+1} = 2x H_k − 2k H_{k−1}`
/// * Laguerre: `(k+1) L_{k+1} = (2k+1−x) L_k − k L_{k−1}`
fn arb_orthopoly(kind: OrthoPoly, n: u64, x: &BigFloat, prec: usize, rm: RoundingMode) -> BigFloat {
    let wp = prec + 32 + (n as f64).log2().ceil().max(0.0) as usize;
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, rm);
    if n == 0 {
        return BigFloat::from_i32(1, prec);
    }
    let mut p_prev = one.clone();
    let mut p_curr = match kind {
        OrthoPoly::Legendre | OrthoPoly::ChebyshevT => xw.clone(),
        OrthoPoly::ChebyshevU | OrthoPoly::Hermite => two.mul(&xw, wp, rm),
        OrthoPoly::Laguerre => one.sub(&xw, wp, rm),
    };
    for k in 1..n {
        let k_bf = BigFloat::from_i128(k as i128, wp);
        let kp1 = BigFloat::from_i128(k as i128 + 1, wp);
        let next = match kind {
            OrthoPoly::Legendre => {
                let two_k_plus_1 = BigFloat::from_i128(2 * k as i128 + 1, wp);
                let a = two_k_plus_1.mul(&xw, wp, rm).mul(&p_curr, wp, rm);
                let b = k_bf.mul(&p_prev, wp, rm);
                a.sub(&b, wp, rm).div(&kp1, wp, rm)
            }
            OrthoPoly::ChebyshevT | OrthoPoly::ChebyshevU => two
                .mul(&xw, wp, rm)
                .mul(&p_curr, wp, rm)
                .sub(&p_prev, wp, rm),
            OrthoPoly::Hermite => {
                let a = two.mul(&xw, wp, rm).mul(&p_curr, wp, rm);
                let b = two.mul(&k_bf, wp, rm).mul(&p_prev, wp, rm);
                a.sub(&b, wp, rm)
            }
            OrthoPoly::Laguerre => {
                let two_k_plus_1 = BigFloat::from_i128(2 * k as i128 + 1, wp);
                let a = two_k_plus_1.sub(&xw, wp, rm).mul(&p_curr, wp, rm);
                let b = k_bf.mul(&p_prev, wp, rm);
                a.sub(&b, wp, rm).div(&kp1, wp, rm)
            }
        };
        p_prev = p_curr;
        p_curr = next;
    }
    round_to(p_curr, prec, rm)
}

fn get_cached(cache: &FxHashMap<ExprId, Complex>, id: ExprId) -> Result<&Complex, SymplexError> {
    cache.get(&id).ok_or_else(|| SymplexError::Unevaluable {
        reason: format!("sub-expression {id:?} not in cache (likely contains free symbols)"),
    })
}

/// Convert a `Ratio<BigInt>` to a `BigFloat` with the given precision in bits.
fn ratio_to_bigfloat(r: &Ratio<BigInt>, prec: usize, rm: RoundingMode) -> BigFloat {
    if r.is_zero() {
        return BigFloat::from_i32(0, prec);
    }

    let numer = r.numer();
    let denom = r.denom();

    let n_bf = bigint_to_bigfloat(numer, prec);

    if denom == &BigInt::from(1) {
        return n_bf;
    }

    let d_bf = bigint_to_bigfloat(denom, prec);
    n_bf.div(&d_bf, prec, rm)
}

/// Convert a `BigInt` to a `BigFloat`, exactly up to the final rounding
/// to `prec` bits.
///
/// Values that fit in `i128` are converted directly.  Larger integers are
/// accumulated limb-by-limb (`acc = acc·2⁶⁴ + limb`) at a working
/// precision wide enough to hold every bit, then rounded once.
fn bigint_to_bigfloat(n: &BigInt, prec: usize) -> BigFloat {
    if let Ok(v) = <BigInt as TryInto<i128>>::try_into(n.clone()) {
        return BigFloat::from_i128(v, prec);
    }
    let (sign, limbs) = n.to_u64_digits();
    let wp = (limbs.len() * 64 + 64).max(prec);
    let rm = RoundingMode::ToEven;
    let base = BigFloat::from_u64(1u64 << 32, wp).powi(2, wp, rm); // 2^64
    let mut acc = BigFloat::new(wp);
    for &limb in limbs.iter().rev() {
        acc = acc
            .mul(&base, wp, rm)
            .add(&BigFloat::from_u64(limb, wp), wp, rm);
    }
    if sign == num_bigint::Sign::Minus {
        acc = acc.neg();
    }
    let _ = acc.set_precision(prec, rm);
    acc
}

/// Try to extract a small integer exponent (fits in i32) from an ExprId.
fn try_as_small_int(arena: &Arena, id: ExprId) -> Option<i32> {
    let r = arena.as_num(id)?;
    if !r.is_integer() {
        return None;
    }
    let n = r.to_integer();
    let val: i64 = n.try_into().ok()?;
    if val >= i32::MIN as i64 && val <= i32::MAX as i64 {
        Some(val as i32)
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Decimal / complex formatting
// ═══════════════════════════════════════════════════════════════════════════

/// Determine whether `part` is negligibly small compared to `other`.
///
/// Returns `true` if `part` is zero, or if its binary exponent is more
/// than `digits * log2(10)` bits below `other`'s exponent (meaning it
/// is below the requested output precision).
fn is_negligible_part(part: &BigFloat, other: &BigFloat, digits: u32) -> bool {
    if part.is_zero() {
        return true;
    }
    if other.is_zero() {
        return false;
    }
    match (part.exponent(), other.exponent()) {
        (Some(p_exp), Some(o_exp)) => {
            let bit_threshold = (digits as i64) * 34 / 10 + 4;
            (o_exp as i64 - p_exp as i64) > bit_threshold
        }
        _ => false,
    }
}

/// Format a complex result as a string.
///
/// If the imaginary part is negligible, formats as a real number.
/// If the real part is negligible, formats as a pure imaginary number.
/// Otherwise formats as `a + b*i` or `a - b*i`.
fn format_complex(
    z: &Complex,
    digits: u32,
    _prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<String, SymplexError> {
    let re = &z.0;
    let im = &z.1;

    let re_negligible = is_negligible_part(re, im, digits);
    let im_negligible = is_negligible_part(im, re, digits);

    if re_negligible && im_negligible {
        return Ok("0".to_string());
    }

    if im_negligible {
        return format_decimal(re, digits, rm, cc);
    }

    if re_negligible {
        // Pure imaginary.
        let im_str = format_decimal(im, digits, rm, cc)?;
        return if im_str == "1" {
            Ok("i".to_string())
        } else if im_str == "-1" {
            Ok("-i".to_string())
        } else {
            Ok(format!("{im_str}*i"))
        };
    }

    // Both parts present.
    let re_str = format_decimal(re, digits, rm, cc)?;
    if im.is_negative() {
        let im_abs_str = format_decimal(&im.abs(), digits, rm, cc)?;
        if im_abs_str == "1" {
            Ok(format!("{re_str} - i"))
        } else {
            Ok(format!("{re_str} - {im_abs_str}*i"))
        }
    } else {
        let im_str = format_decimal(im, digits, rm, cc)?;
        if im_str == "1" {
            Ok(format!("{re_str} + i"))
        } else {
            Ok(format!("{re_str} + {im_str}*i"))
        }
    }
}

/// Round a decimal digit string `0.d1d2d3…` to `digits` significant digits.
///
/// Rounds to nearest, ties to even (the digits beyond the requested
/// count are exact decimal digits of the binary value, so an exact tie is
/// a genuine `…5000…`).  Carries propagate leftwards; a carry out of the
/// leading digit (`9.9996 → 10.00`) is reported as an exponent increment
/// so the caller can re-place the decimal point.
///
/// Returns `(rounded_digits, exponent_increment)`; when `mantissa` has no
/// more than `digits` digits it is returned unchanged.
fn round_decimal_digits(mantissa: &[u8], digits: usize) -> (Vec<u8>, i64) {
    if digits == 0 || mantissa.len() <= digits {
        return (mantissa.to_vec(), 0);
    }
    let mut kept: Vec<u8> = mantissa[..digits].to_vec();
    let next = mantissa[digits];
    let rest_nonzero = mantissa[digits + 1..].iter().any(|&d| d != 0);
    let round_up = match next.cmp(&5) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        // Exactly half: ties to even.
        std::cmp::Ordering::Equal => rest_nonzero || kept[digits - 1] % 2 == 1,
    };
    if !round_up {
        return (kept, 0);
    }
    // Propagate the carry from the last kept digit leftwards.
    for d in kept.iter_mut().rev() {
        if *d == 9 {
            *d = 0;
        } else {
            *d += 1;
            return (kept, 0);
        }
    }
    // Carried out of the leading digit: 0.999… → 1.000… × 10.
    kept[0] = 1;
    (kept, 1)
}

/// Format a `BigFloat` as a decimal string with `digits` significant digits.
///
/// Uses astro-float's `convert_to_radix` for reliable base-10 conversion,
/// rounds to `digits` significant digits, then formats the result with
/// proper decimal point placement.
fn format_decimal(
    val: &BigFloat,
    digits: u32,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<String, SymplexError> {
    if val.is_zero() {
        return Ok("0".to_string());
    }

    if val.is_inf_pos() {
        return Ok("oo".to_string());
    }
    if val.is_inf_neg() {
        return Ok("-oo".to_string());
    }

    // Use the library's decimal conversion.
    let (sign, mantissa, exponent) = val
        .convert_to_radix(Radix::Dec, rm, cc)
        .map_err(|e| SymplexError::NotImplemented(format!("decimal conversion failed: {e:?}")))?;

    let sign_str = if sign == Sign::Neg { "-" } else { "" };

    // mantissa is Vec<u8> of decimal digits [d1, d2, d3, ...]
    // representing the value 0.d1d2d3... × 10^exponent.
    // So the actual value is d1.d2d3... × 10^(exponent-1).

    // Round (not truncate) to the requested digit count.
    let (rounded, exp_carry) = round_decimal_digits(&mantissa, digits as usize);
    let n = rounded.len();
    if n == 0 {
        return Ok("0".to_string());
    }

    let digit_chars: Vec<char> = rounded.iter().map(|&d| (b'0' + d) as char).collect();

    // The "adjusted exponent" is exponent-1 (shifting from 0.ddd to d.ddd notation).
    let adj_exp = exponent as i64 - 1 + exp_carry;

    // Decide between plain decimal and scientific notation.
    if adj_exp >= 0 && (adj_exp as usize) < n {
        // The decimal point falls within the digit string.
        // E.g., digits=[3,1,4,1,5], adj_exp=0 → "3.1415"
        let dot_pos = adj_exp as usize + 1;
        let mut s = String::with_capacity(n + 2);
        s.push_str(sign_str);
        for (i, &ch) in digit_chars.iter().enumerate() {
            if i == dot_pos && dot_pos < n {
                s.push('.');
            }
            s.push(ch);
        }
        // Trim trailing zeros after the decimal point.
        if s.contains('.') {
            let trimmed = s.trim_end_matches('0').trim_end_matches('.');
            Ok(trimmed.to_string())
        } else {
            Ok(s)
        }
    } else if adj_exp >= 0 && (adj_exp as usize) < n + 6 {
        // Small integer — pad with trailing zeros.
        let mut s = String::with_capacity(adj_exp as usize + 2);
        s.push_str(sign_str);
        for &ch in &digit_chars {
            s.push(ch);
        }
        let needed_zeros = (adj_exp as usize + 1).saturating_sub(digit_chars.len());
        for _ in 0..needed_zeros {
            s.push('0');
        }
        Ok(s)
    } else if (-4..0).contains(&adj_exp) {
        // Small fraction like 0.00123.
        let mut s = String::with_capacity(n + (-adj_exp) as usize + 3);
        s.push_str(sign_str);
        s.push_str("0.");
        let leading_zeros = (-adj_exp - 1) as usize;
        for _ in 0..leading_zeros {
            s.push('0');
        }
        for &ch in &digit_chars {
            s.push(ch);
        }
        // Trim trailing zeros.
        let trimmed = s.trim_end_matches('0');
        Ok(trimmed.to_string())
    } else {
        // Scientific notation: d1.d2d3...dn × 10^adj_exp.
        let mut s = String::with_capacity(n + 10);
        s.push_str(sign_str);
        s.push(digit_chars[0]);
        if n > 1 {
            s.push('.');
            for &ch in &digit_chars[1..] {
                s.push(ch);
            }
            // Trim trailing zeros in the fractional part.
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        s.push_str(&format!("e{adj_exp}"));
        Ok(s)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    /// Helper: evaluate and assert the result starts with expected prefix.
    fn assert_evalf_starts_with(arena: &Arena, expr: ExprId, digits: u32, prefix: &str) {
        let result = evalf(arena, expr, digits).unwrap();
        assert!(
            result.starts_with(prefix),
            "expected to start with '{prefix}', got: '{result}'"
        );
    }

    /// Helper: evaluate and assert exact string match.
    fn assert_evalf_eq(arena: &Arena, expr: ExprId, digits: u32, expected: &str) {
        let result = evalf(arena, expr, digits).unwrap();
        assert_eq!(result, expected, "evalf mismatch");
    }

    // ── Integers ────────────────────────────────────────────────────

    #[test]
    fn integer_zero() {
        let a = Arena::new();
        assert_evalf_eq(&a, a.zero, 10, "0");
    }

    #[test]
    fn integer_one() {
        let a = Arena::new();
        assert_evalf_eq(&a, a.one, 10, "1");
    }

    #[test]
    fn integer_negative_one() {
        let a = Arena::new();
        assert_evalf_eq(&a, a.neg_one, 10, "-1");
    }

    #[test]
    fn integer_five() {
        let mut a = Arena::new();
        let five = a.int(5);
        assert_evalf_eq(&a, five, 10, "5");
    }

    #[test]
    fn integer_large() {
        let mut a = Arena::new();
        let n = a.int(123456);
        assert_evalf_eq(&a, n, 10, "123456");
    }

    // ── Rationals ───────────────────────────────────────────────────

    #[test]
    fn rational_half() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        assert_evalf_eq(&a, half, 10, "0.5");
    }

    #[test]
    fn rational_one_third() {
        let mut a = Arena::new();
        let third = a.rational(1, 3);
        assert_evalf_starts_with(&a, third, 10, "0.333333333");
    }

    #[test]
    fn rational_negative() {
        let mut a = Arena::new();
        let r = a.rational(-3, 4);
        assert_evalf_starts_with(&a, r, 10, "-0.75");
    }

    // ── Constants ───────────────────────────────────────────────────

    #[test]
    fn constant_pi_15_digits() {
        let a = Arena::new();
        assert_evalf_starts_with(&a, a.pi, 15, "3.14159265358979");
    }

    #[test]
    fn constant_pi_50_digits() {
        let a = Arena::new();
        let result = evalf(&a, a.pi, 50).unwrap();
        assert!(
            result.starts_with("3.1415926535897932384626433832795"),
            "pi to 50 digits: {result}"
        );
    }

    #[test]
    fn constant_e_15_digits() {
        let a = Arena::new();
        assert_evalf_starts_with(&a, a.e_const, 15, "2.71828182845905");
    }

    // ── Arithmetic ──────────────────────────────────────────────────

    #[test]
    fn add_integers() {
        let mut a = Arena::new();
        let two = a.int(2);
        let three = a.int(3);
        let expr = a.add(&[two, three]);
        assert_evalf_eq(&a, expr, 10, "5");
    }

    #[test]
    fn mul_integers() {
        let mut a = Arena::new();
        let two = a.int(2);
        let three = a.int(3);
        let expr = a.mul(&[two, three]);
        assert_evalf_eq(&a, expr, 10, "6");
    }

    #[test]
    fn pow_integer() {
        let mut a = Arena::new();
        let two = a.int(2);
        let ten = a.int(10);
        let expr = a.pow(two, ten);
        assert_evalf_eq(&a, expr, 10, "1024");
    }

    #[test]
    fn pow_negative_exponent() {
        let mut a = Arena::new();
        let two = a.int(2);
        let neg_one = a.int(-1);
        let expr = a.pow(two, neg_one);
        assert_evalf_eq(&a, expr, 10, "0.5");
    }

    #[test]
    fn pi_plus_one() {
        let mut a = Arena::new();
        let pi = a.pi;
        let one = a.one;
        let expr = a.add(&[pi, one]);
        assert_evalf_starts_with(&a, expr, 15, "4.14159265358979");
    }

    // ── Transcendental functions ─────────────────────────────────────

    #[test]
    fn sin_zero() {
        let mut a = Arena::new();
        let zero = a.zero;
        let expr = a.sin(zero);
        assert_evalf_eq(&a, expr, 10, "0");
    }

    #[test]
    fn cos_zero() {
        let mut a = Arena::new();
        let zero = a.zero;
        let expr = a.cos(zero);
        assert_evalf_eq(&a, expr, 10, "1");
    }

    #[test]
    fn exp_zero() {
        let mut a = Arena::new();
        let zero = a.zero;
        let expr = a.exp(zero);
        assert_evalf_eq(&a, expr, 10, "1");
    }

    #[test]
    fn exp_one() {
        let mut a = Arena::new();
        let one = a.one;
        let expr = a.exp(one);
        assert_evalf_starts_with(&a, expr, 15, "2.71828182845905");
    }

    #[test]
    fn ln_one() {
        let mut a = Arena::new();
        let one = a.one;
        let expr = a.ln(one);
        assert_evalf_eq(&a, expr, 10, "0");
    }

    #[test]
    fn ln_e() {
        let mut a = Arena::new();
        let e = a.e_const;
        let expr = a.ln(e);
        // ln(e) should be 1, or very close.
        let result = evalf(&a, expr, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(999.0);
        assert!(
            (val - 1.0).abs() < 1e-10,
            "ln(e) should be ~1, got: {result}"
        );
    }

    #[test]
    fn sqrt_four() {
        let mut a = Arena::new();
        let four = a.int(4);
        let expr = a.sqrt(four);
        assert_evalf_eq(&a, expr, 10, "2");
    }

    #[test]
    fn sqrt_two_30_digits() {
        let mut a = Arena::new();
        let two = a.int(2);
        let expr = a.sqrt(two);
        assert_evalf_starts_with(&a, expr, 30, "1.4142135623730950488");
    }

    #[test]
    fn sin_pi_is_near_zero() {
        let mut a = Arena::new();
        let pi = a.pi;
        let expr = a.sin(pi);
        let result = evalf(&a, expr, 20).unwrap();
        // sin(pi) is mathematically 0, numerically very small.
        let val: f64 = result.parse().unwrap_or(999.0);
        assert!(val.abs() < 1e-15, "sin(pi) should be ~0, got: {result}");
    }

    // ── Neg and Abs ─────────────────────────────────────────────────

    #[test]
    fn neg_pi() {
        let mut a = Arena::new();
        let pi = a.pi;
        let expr = a.neg(pi);
        assert_evalf_starts_with(&a, expr, 10, "-3.14159265");
    }

    #[test]
    fn abs_neg_pi() {
        let mut a = Arena::new();
        let pi = a.pi;
        let neg_pi = a.neg(pi);
        let expr = a.abs(neg_pi);
        assert_evalf_starts_with(&a, expr, 10, "3.14159265");
    }

    #[test]
    fn abs_negative_integer() {
        let mut a = Arena::new();
        let neg_five = a.int(-5);
        let expr = a.abs(neg_five);
        assert_evalf_eq(&a, expr, 10, "5");
    }

    // ── Error cases ─────────────────────────────────────────────────

    #[test]
    fn free_symbol_errors() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = evalf(&a, x, 10);
        assert!(result.is_err(), "evalf of free symbol should error");
    }

    #[test]
    fn infinity_errors() {
        let a = Arena::new();
        let result = evalf(&a, a.infinity, 10);
        assert!(result.is_err(), "evalf of infinity should error");
    }

    #[test]
    fn nan_errors() {
        let a = Arena::new();
        let result = evalf(&a, a.nan, 10);
        assert!(result.is_err(), "evalf of NaN should error");
    }

    #[test]
    fn imaginary_unit_works() {
        let a = Arena::new();
        let result = evalf(&a, a.i_unit, 10);
        assert!(result.is_ok(), "evalf of i should work now");
        let s = result.unwrap();
        assert!(
            s.contains("i") || s.contains("I"),
            "should display as imaginary: {s}"
        );
    }

    // ── Complex number tests ────────────────────────────────────────

    #[test]
    fn evalf_one_plus_i() {
        let mut a = Arena::new();
        let one = a.one;
        let i = a.i_unit;
        let expr = a.add(&[one, i]);
        let result = evalf(&a, expr, 10);
        assert!(result.is_ok(), "evalf(1+i) should work: {:?}", result.err());
        let s = result.unwrap();
        assert!(
            s.contains("i") || s.contains("I"),
            "1+i should contain 'i': {s}"
        );
    }

    #[test]
    fn evalf_exp_i_pi() {
        // e^(i*pi) ≈ -1
        let mut a = Arena::new();
        let i = a.i_unit;
        let pi = a.pi;
        let i_times_pi = a.mul(&[i, pi]);
        let expr = a.exp(i_times_pi);
        let result = evalf(&a, expr, 15);
        assert!(
            result.is_ok(),
            "evalf(exp(i*pi)) should work: {:?}",
            result.err()
        );
        let s = result.unwrap();
        let val: f64 = s.parse().unwrap_or(999.0);
        assert!(
            (val - (-1.0)).abs() < 1e-10,
            "exp(i*pi) should be ~-1, got: {s}"
        );
    }

    #[test]
    fn evalf_i_squared() {
        // i^2 = -1 (may be canonicalized by the arena)
        let mut a = Arena::new();
        let i = a.i_unit;
        let two = a.int(2);
        let expr = a.pow(i, two);
        let result = evalf(&a, expr, 10);
        assert!(result.is_ok(), "evalf(i^2) should work: {:?}", result.err());
        let s = result.unwrap();
        let val: f64 = s.parse().unwrap_or(999.0);
        assert!((val - (-1.0)).abs() < 1e-10, "i^2 should be -1, got: {s}");
    }

    #[test]
    fn evalf_sqrt_neg_one() {
        // sqrt(-1) = (-1)^(1/2) should give i
        let mut a = Arena::new();
        let neg_one = a.neg_one;
        let expr = a.sqrt(neg_one);
        let result = evalf(&a, expr, 10);
        assert!(
            result.is_ok(),
            "evalf(sqrt(-1)) should work: {:?}",
            result.err()
        );
        let s = result.unwrap();
        assert!(
            s.contains("i") || s.contains("I"),
            "sqrt(-1) should be imaginary: {s}"
        );
    }

    #[test]
    fn evalf_abs_3_plus_4i() {
        // |3 + 4i| = 5
        let mut a = Arena::new();
        let three = a.int(3);
        let four = a.int(4);
        let i = a.i_unit;
        let four_i = a.mul(&[four, i]);
        let sum = a.add(&[three, four_i]);
        let expr = a.abs(sum);
        let result = evalf(&a, expr, 10);
        assert!(
            result.is_ok(),
            "evalf(|3+4i|) should work: {:?}",
            result.err()
        );
        let s = result.unwrap();
        let val: f64 = s.parse().unwrap_or(999.0);
        assert!((val - 5.0).abs() < 1e-10, "|3+4i| should be 5, got: {s}");
    }

    // ── Composite expressions ───────────────────────────────────────

    #[test]
    fn polynomial_at_value() {
        // Evaluate x^2 + 2*x + 1 at x=3 → 16.
        // We substitute first, then evalf.
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let expr = a.add(&[x_sq, two_x, a.one]);

        // Substitute x → 3.
        let three = a.int(3);
        let evaluated = crate::transforms::subs::subs(&mut a, expr, x, three);
        assert_evalf_eq(&a, evaluated, 10, "16");
    }

    #[test]
    fn sin_cos_identity() {
        // sin^2(1) + cos^2(1) should be very close to 1.
        let mut a = Arena::new();
        let one = a.one;
        let two = a.int(2);
        let sin_1 = a.sin(one);
        let cos_1 = a.cos(one);
        let sin_sq = a.pow(sin_1, two);
        let cos_sq = a.pow(cos_1, two);
        let expr = a.add(&[sin_sq, cos_sq]);
        let result = evalf(&a, expr, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(0.0);
        assert!(
            (val - 1.0).abs() < 1e-12,
            "sin^2(1) + cos^2(1) should be ~1, got: {result}"
        );
    }

    // ── Idempotence ─────────────────────────────────────────────────

    #[test]
    fn same_precision_same_result() {
        let a = Arena::new();
        let r1 = evalf(&a, a.pi, 20).unwrap();
        let r2 = evalf(&a, a.pi, 20).unwrap();
        assert_eq!(r1, r2, "same precision should give same result");
    }

    // ── Bessel function evaluation ─────────────────────────────────

    #[test]
    fn bessel_j0_at_zero() {
        // J_0(0) = 1
        let mut a = Arena::new();
        let zero = a.zero;
        let j0_0 = a.besselj(zero, zero);
        let result = evalf(&a, j0_0, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(f64::NAN);
        assert!(
            (val - 1.0).abs() < 1e-10,
            "J_0(0) should be 1.0, got {result}"
        );
    }

    #[test]
    fn bessel_j0_at_one() {
        // J_0(1) ≈ 0.7651976865579666
        let mut a = Arena::new();
        let zero = a.zero;
        let one = a.one;
        let j0_1 = a.besselj(zero, one);
        let result = evalf(&a, j0_1, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(f64::NAN);
        assert!(
            (val - 0.7651976865579666).abs() < 1e-8,
            "J_0(1) should be ~0.7652, got {result}"
        );
    }

    #[test]
    fn bessel_j1_at_zero() {
        // J_1(0) = 0
        let mut a = Arena::new();
        let one = a.one;
        let zero = a.zero;
        let j1_0 = a.besselj(one, zero);
        let result = evalf(&a, j1_0, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(f64::NAN);
        assert!(val.abs() < 1e-10, "J_1(0) should be 0, got {result}");
    }

    #[test]
    fn bessel_j1_at_one() {
        // J_1(1) ≈ 0.44005058574493355
        let mut a = Arena::new();
        let one = a.one;
        let j1_1 = a.besselj(one, one);
        let result = evalf(&a, j1_1, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(f64::NAN);
        assert!(
            (val - 0.44005058574493355).abs() < 1e-8,
            "J_1(1) should be ~0.4401, got {result}"
        );
    }

    #[test]
    fn bessel_y0_at_one() {
        // Y_0(1) ≈ 0.08825696421567696
        let mut a = Arena::new();
        let zero = a.zero;
        let one = a.one;
        let y0_1 = a.bessely(zero, one);
        let result = evalf(&a, y0_1, 15).unwrap();
        let val: f64 = result.parse().unwrap_or(f64::NAN);
        assert!(
            (val - 0.08825696421567696).abs() < 1e-6,
            "Y_0(1) should be ~0.0883, got {result}"
        );
    }

    // ── 0.2: named constants ───────────────────────────────────────────────

    #[test]
    fn euler_gamma_and_catalan_high_precision() {
        let a = Arena::new();
        assert_evalf_starts_with(
            &a,
            a.euler_gamma,
            60,
            "0.57721566490153286060651209008240243104215933593992359880576",
        );
        assert_evalf_starts_with(
            &a,
            a.catalan,
            60,
            "0.91596559417721901505460351493238411077414937428167213426649",
        );
        assert_evalf_starts_with(
            &a,
            a.golden_ratio,
            40,
            "1.61803398874989484820458683436563811772",
        );
    }

    #[test]
    fn bigint_to_bigfloat_is_exact_for_huge_integers() {
        // 2^200 + 1 is far beyond i128; its BigFloat must keep all bits.
        let n = (BigInt::from(1) << 200) + BigInt::from(1);
        let bf = bigint_to_bigfloat(&n, 256);
        let two200 = BigFloat::from_i32(2, 256).powi(200, 256, RoundingMode::ToEven);
        let diff = bf.sub(&two200, 256, RoundingMode::ToEven);
        assert_eq!(diff, BigFloat::from_i32(1, 256));
    }

    #[test]
    fn bf_pow_handles_exactly_representable_results() {
        // BigFloat::pow would hang on 4^(1/2) = 2; bf_pow must not.
        let mut cc = Consts::new().unwrap();
        let rm = RoundingMode::ToEven;
        let four = BigFloat::from_i32(4, 160);
        let half = BigFloat::from_f64(0.5, 160);
        let r = bf_pow(&four, &half, 160, rm, &mut cc);
        let diff = r.sub(&BigFloat::from_i32(2, 160), 160, rm);
        assert!(
            diff.is_zero() || diff.exponent().is_none_or(|e| e < -140),
            "4^0.5 = {r}"
        );
        // integer exponent path
        let three = BigFloat::from_i32(3, 160);
        let r = bf_pow(&three, &BigFloat::from_i32(5, 160), 160, rm, &mut cc);
        assert_eq!(r, BigFloat::from_i32(243, 160));
        let r = bf_pow(&three, &BigFloat::from_i32(-1, 160), 160, rm, &mut cc);
        let back = r.mul(&three, 160, rm);
        let d = back.sub(&BigFloat::from_i32(1, 160), 160, rm);
        assert!(d.is_zero() || d.exponent().is_none_or(|e| e < -140));
    }

    // ── 0.2: special functions ──────────────────────────────────────────────

    #[test]
    fn si_ci_ei_li_at_one_and_two() {
        let mut a = Arena::new();
        let one = a.one;
        let two = a.int(2);
        let si = a.si(one);
        assert_evalf_starts_with(&a, si, 20, "0.94608307036718301494");
        let ci = a.ci(one);
        assert_evalf_starts_with(&a, ci, 20, "0.33740392290096813466");
        let ei = a.ei(one);
        assert_evalf_starts_with(&a, ei, 20, "1.8951178163559367555");
        let li = a.li(two);
        assert_evalf_starts_with(&a, li, 20, "1.0451637801174927848");
    }

    #[test]
    fn si_asymptotic_and_series_regimes_agree() {
        // Evaluate Si(40) at two precisions: at low precision the asymptotic
        // expansion is used (40 > 160·ln2 ≈ 111 is false …), so force the
        // regimes by comparing against the same value at higher precision
        // where the series is used.
        let mut a = Arena::new();
        let x = a.int(200);
        let si = a.si(x);
        let lo = evalf(&a, si, 12).unwrap(); // asymptotic (200 > 128·ln2)
        let hi = evalf(&a, si, 120).unwrap(); // series (200 < 472·ln2)
        assert!(hi.starts_with(&lo[..12]), "{lo} vs {hi}");
    }

    #[test]
    fn zeta_borwein_and_functional_equation() {
        let mut a = Arena::new();
        let three = a.int(3);
        let z3 = a.zeta(three);
        assert_evalf_starts_with(&a, z3, 30, "1.20205690315959428539973816151");
        let half = a.rational(1, 2);
        let zh = a.zeta(half);
        assert_evalf_starts_with(&a, zh, 20, "-1.4603545088095868129");
        // ζ(−5/2) > 0 (ζ is positive on (−4, −2); ζ(−3) = 1/120).
        let neg = a.rational(-5, 2);
        let zn = a.zeta(neg);
        assert_evalf_starts_with(&a, zn, 15, "0.00851692877785");
        // ζ(−1/2) < 0
        let neg_half = a.rational(-1, 2);
        let znh = a.zeta(neg_half);
        assert_evalf_starts_with(&a, znh, 15, "-0.20788622497735");
    }

    #[test]
    fn polygamma_matches_trigamma_closed_forms() {
        let mut a = Arena::new();
        // Build the node structurally so the exact folding does not kick in.
        let one = a.one;
        let x = a.rational(7, 3);
        let pg = a.intern(ExprNode::Polygamma(one, x));
        assert_evalf_starts_with(&a, pg, 20, "0.53309712542709408179");
        // ψ'(1/2) = π²/2 through the numerical path
        let half = a.rational(1, 2);
        let pg_half = a.intern(ExprNode::Polygamma(one, half));
        assert_evalf_starts_with(&a, pg_half, 20, "4.9348022005446793094");
    }

    #[test]
    fn bessel_i_k_reference() {
        let mut a = Arena::new();
        let zero = a.zero;
        let one = a.one;
        let i0 = a.besseli(zero, one);
        assert_evalf_starts_with(&a, i0, 20, "1.2660658777520083356");
        let k0 = a.besselk(zero, one);
        assert_evalf_starts_with(&a, k0, 20, "0.42102443824070833334");
        let k1 = a.besselk(one, one);
        assert_evalf_starts_with(&a, k1, 20, "0.60190723019723457474");
    }

    #[test]
    fn orthogonal_polynomials_large_degree() {
        let mut a = Arena::new();
        let n = a.int(100);
        let third = a.rational(1, 3);
        let t = a.chebyshev_t(n, third);
        let s = evalf(&a, t, 15).unwrap();
        let v: f64 = s.parse().unwrap();
        let expected = (100.0 * (1.0f64 / 3.0).acos()).cos();
        assert!((v - expected).abs() < 1e-12, "{v} vs {expected}");
        let p = a.legendre(n, a.one);
        assert_evalf_starts_with(&a, p, 10, "1");
    }

    #[test]
    fn complex_nodes_evaluate_numerically() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let re_z = a.intern(ExprNode::Re(z));
        let arg_z = a.intern(ExprNode::Arg(z));
        let conj_z = a.intern(ExprNode::Conjugate(z));
        let three = a.int(3);
        let four = a.int(4);
        let four_i = a.mul(&[four, a.i_unit]);
        let w = a.add(&[three, four_i]);
        // Substitute structurally *without* refolding: intern the nodes
        // directly on w.
        let _ = re_z;
        let re_w = a.intern(ExprNode::Re(w));
        let arg_w = a.intern(ExprNode::Arg(w));
        let conj_w = a.intern(ExprNode::Conjugate(w));
        let _ = (arg_z, conj_z);
        assert_eq!(evalf(&a, re_w, 10).unwrap(), "3");
        assert_evalf_starts_with(&a, arg_w, 15, "0.92729521800161");
        let s = evalf(&a, conj_w, 10).unwrap();
        assert!(s.contains('3') && s.contains('4') && s.contains('-'), "{s}");
    }
}
