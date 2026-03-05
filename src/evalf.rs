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

use crate::arena::Arena;
use crate::errors::SymplexError;
use crate::node::{ExprId, ExprNode};
use crate::walk;
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
            debug!("evalf: Binomial via Gamma");
            let n_val = get_cached(cache, *n_id)?;
            let k_val = get_cached(cache, *k_id)?;
            if !n_val.1.is_zero() || !k_val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Binomial of complex arguments not yet supported in evalf".into(),
                });
            }
            let n_f = bigfloat_to_f64(&n_val.0, rm, cc)?;
            let k_f = bigfloat_to_f64(&k_val.0, rm, cc)?;
            let numer = lanczos_gamma_f64(n_f + 1.0)?;
            let denom = lanczos_gamma_f64(k_f + 1.0)? * lanczos_gamma_f64(n_f - k_f + 1.0)?;
            if denom.abs() < 1e-300 {
                return Err(SymplexError::Unevaluable {
                    reason: "Binomial denominator is zero (pole in Gamma)".into(),
                });
            }
            let result = numer / denom;
            if !result.is_finite() {
                return Err(SymplexError::Unevaluable {
                    reason: "Binomial result is not finite".into(),
                });
            }
            Ok((f64_to_bigfloat(result, prec), BigFloat::new(prec)))
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
            let x = bigfloat_to_f64(&val.0, rm, cc)?;
            let gamma = lanczos_gamma_f64(x)?;
            if gamma <= 0.0 {
                return Err(SymplexError::Unevaluable {
                    reason: "LogGamma undefined for non-positive Gamma value".into(),
                });
            }
            Ok((f64_to_bigfloat(gamma.ln(), prec), BigFloat::new(prec)))
        }

        ExprNode::Digamma(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Digamma of complex argument not yet supported in evalf".into(),
                });
            }
            let x = bigfloat_to_f64(&val.0, rm, cc)?;
            let result = digamma_f64(x)?;
            Ok((f64_to_bigfloat(result, prec), BigFloat::new(prec)))
        }

        ExprNode::Erf(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "erf of complex argument not yet supported in evalf".into(),
                });
            }
            let x_val = bigfloat_to_f64(&val.0, rm, cc)?;
            debug!(x = ?x_val, "evalf: erf via series");
            let result = erf_f64(x_val);
            Ok((f64_to_bigfloat(result, prec), BigFloat::new(prec)))
        }

        ExprNode::Erfc(inner) => {
            let val = get_cached(cache, *inner)?;
            if !val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "erfc of complex argument not yet supported in evalf".into(),
                });
            }
            let x = bigfloat_to_f64(&val.0, rm, cc)?;
            let result = 1.0 - erf_f64(x);
            Ok((f64_to_bigfloat(result, prec), BigFloat::new(prec)))
        }

        ExprNode::Beta(a_id, b_id) => {
            let a_val = get_cached(cache, *a_id)?;
            let b_val = get_cached(cache, *b_id)?;
            if !a_val.1.is_zero() || !b_val.1.is_zero() {
                return Err(SymplexError::Unevaluable {
                    reason: "Beta of complex arguments not yet supported in evalf".into(),
                });
            }
            let a = bigfloat_to_f64(&a_val.0, rm, cc)?;
            let b = bigfloat_to_f64(&b_val.0, rm, cc)?;
            let ga = lanczos_gamma_f64(a)?;
            let gb = lanczos_gamma_f64(b)?;
            let gab = lanczos_gamma_f64(a + b)?;
            let result = ga * gb / gab;
            if !result.is_finite() {
                return Err(SymplexError::Unevaluable {
                    reason: "Beta function result is not finite".into(),
                });
            }
            Ok((f64_to_bigfloat(result, prec), BigFloat::new(prec)))
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

            // Real non-negative base with real exponent: use real pow.
            if b_is_real && e_is_real && b.0.is_positive() {
                return Ok((b.0.pow(&e.0, prec, rm, cc), BigFloat::new(prec)));
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

        // ── Unevaluable ────────────────────────────────────────────
        ExprNode::Apply(sid, _) => {
            let name = arena.symbol_name(*sid);
            Err(SymplexError::Unevaluable {
                reason: format!("cannot evaluate user function '{name}'"),
            })
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
            for &(value_id, cond_id) in pairs.iter() {
                // Check if condition is literally BoolTrue
                if cond_id == arena.bool_true {
                    return eval_node_or_subtree(arena, value_id, cache, prec, rm, cc);
                }
                // Check if condition is literally BoolFalse — skip
                if cond_id == arena.bool_false {
                    continue;
                }
                // Try to evaluate relational conditions numerically
                match arena.node(cond_id) {
                    ExprNode::Gt(a, b) => {
                        if let (Some(av), Some(bv)) = (cache.get(a), cache.get(b))
                            && av.1.is_zero()
                            && bv.1.is_zero()
                        {
                            let diff = av.0.sub(&bv.0, prec, rm);
                            if diff.is_positive() {
                                return eval_node_or_subtree(arena, value_id, cache, prec, rm, cc);
                            }
                            continue; // condition is false
                        }
                    }
                    ExprNode::Ge(a, b) => {
                        if let (Some(av), Some(bv)) = (cache.get(a), cache.get(b))
                            && av.1.is_zero()
                            && bv.1.is_zero()
                        {
                            let diff = av.0.sub(&bv.0, prec, rm);
                            if diff.is_positive() || diff.is_zero() {
                                return eval_node_or_subtree(arena, value_id, cache, prec, rm, cc);
                            }
                            continue;
                        }
                    }
                    ExprNode::Eq_(a, b) => {
                        if let (Some(av), Some(bv)) = (cache.get(a), cache.get(b))
                            && av.1.is_zero()
                            && bv.1.is_zero()
                        {
                            let diff = av.0.sub(&bv.0, prec, rm);
                            if diff.is_zero() {
                                return eval_node_or_subtree(arena, value_id, cache, prec, rm, cc);
                            }
                            continue;
                        }
                    }
                    ExprNode::Not(inner) => {
                        if *inner == arena.bool_true {
                            continue; // Not(True) = False
                        }
                        if *inner == arena.bool_false {
                            return eval_node_or_subtree(arena, value_id, cache, prec, rm, cc);
                        }
                    }
                    _ => {}
                }
            }
            // No condition was definitively true — try the last branch (often the "else")
            if let Some(&(value_id, _)) = pairs.last() {
                debug!("evalf: Piecewise — falling back to last branch");
                return eval_node_or_subtree(arena, value_id, cache, prec, rm, cc);
            }
            Err(SymplexError::Unevaluable {
                reason: "cannot evaluate piecewise: no condition is definitively true".into(),
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

fn c_zero(prec: usize) -> Complex {
    (BigFloat::new(prec), BigFloat::new(prec))
}

fn c_one(prec: usize) -> Complex {
    (BigFloat::from_i32(1, prec), BigFloat::new(prec))
}

fn c_i(prec: usize) -> Complex {
    (BigFloat::new(prec), BigFloat::from_i32(1, prec))
}

fn c_from_real(r: BigFloat, prec: usize) -> Complex {
    (r, BigFloat::new(prec))
}

fn c_add(a: &Complex, b: &Complex, prec: usize, rm: RoundingMode) -> Complex {
    (a.0.add(&b.0, prec, rm), a.1.add(&b.1, prec, rm))
}

fn c_sub(a: &Complex, b: &Complex, prec: usize, rm: RoundingMode) -> Complex {
    (a.0.sub(&b.0, prec, rm), a.1.sub(&b.1, prec, rm))
}

fn c_mul(a: &Complex, b: &Complex, prec: usize, rm: RoundingMode) -> Complex {
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    let ac = a.0.mul(&b.0, prec, rm);
    let bd = a.1.mul(&b.1, prec, rm);
    let ad = a.0.mul(&b.1, prec, rm);
    let bc = a.1.mul(&b.0, prec, rm);
    (ac.sub(&bd, prec, rm), ad.add(&bc, prec, rm))
}

fn c_div(a: &Complex, b: &Complex, prec: usize, rm: RoundingMode) -> Complex {
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
fn c_neg(a: &Complex, prec: usize, rm: RoundingMode) -> Complex {
    (a.0.neg(), a.1.neg())
}

fn c_abs(a: &Complex, prec: usize, rm: RoundingMode) -> BigFloat {
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

/// Convert an `f64` to a `BigFloat` with the given precision.
fn f64_to_bigfloat(f: f64, prec: usize) -> BigFloat {
    BigFloat::from_f64(f, prec)
}

/// Lanczos approximation for the Gamma function (g=7, 9 coefficients).
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

/// Digamma function via recurrence + asymptotic series.
///
/// Uses psi(x+1) = psi(x) + 1/x to shift x to a large value,
/// then the asymptotic expansion:
///   psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
fn digamma_f64(x: f64) -> Result<f64, SymplexError> {
    if x.is_nan() || x.is_infinite() {
        return Err(SymplexError::Unevaluable {
            reason: "Digamma of special float value".into(),
        });
    }

    // Handle negative x via reflection: psi(1-x) - psi(x) = pi*cot(pi*x)
    if x < 0.0 {
        let sin_val = (std::f64::consts::PI * x).sin();
        if sin_val.abs() < 1e-300 {
            return Err(SymplexError::Unevaluable {
                reason: "Digamma at non-positive integer pole".into(),
            });
        }
        let cos_val = (std::f64::consts::PI * x).cos();
        let psi_1mx = digamma_f64(1.0 - x)?;
        return Ok(psi_1mx - std::f64::consts::PI * cos_val / sin_val);
    }

    // Use recurrence to shift x >= 8 for good convergence of asymptotic series
    let mut result = 0.0;
    let mut x = x;
    while x < 8.0 {
        if x.abs() < 1e-300 {
            return Err(SymplexError::Unevaluable {
                reason: "Digamma at non-positive integer pole".into(),
            });
        }
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion: psi(x) ~ ln(x) - 1/(2x) - sum B_{2k}/(2k * x^{2k})
    // Bernoulli numbers: B2=1/6, B4=-1/30, B6=1/42, B8=-1/30, B10=5/66, B12=-691/2730
    result += x.ln() - 0.5 / x;
    let x2 = x * x;
    let mut x_pow = x2; // x^2
    // B2/(2*x^2) = 1/(12*x^2)
    result -= 1.0 / (12.0 * x_pow);
    x_pow *= x2; // x^4
    result += 1.0 / (120.0 * x_pow);
    x_pow *= x2; // x^6
    result -= 1.0 / (252.0 * x_pow);
    x_pow *= x2; // x^8
    result += 1.0 / (240.0 * x_pow);
    x_pow *= x2; // x^10
    result -= 5.0 / (660.0 * x_pow);
    x_pow *= x2; // x^12
    result += 691.0 / (32760.0 * x_pow);

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
        let b2k = crate::bernoulli::bernoulli(2 * k);
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
                && t_exp > prev + 10 {
                    tracing::trace!(k, "stirling series: diverging, stopping");
                    break;
                }
            prev_term_exp = Some(t_exp);

            // Term is negligible relative to accumulated result.
            if let Some(r_exp) = result.exponent()
                && (r_exp as i64 - t_exp as i64) > wp as i64 {
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
            && (s_exp as i64) < -(prec as i64 / 2) {
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
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Look up a cached value, returning an error if not found.
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

/// Convert a `BigInt` to a `BigFloat`.
///
/// Uses i128 for values that fit, f64 for larger values (with some
/// precision loss for very large integers — acceptable for MVP).
fn bigint_to_bigfloat(n: &BigInt, prec: usize) -> BigFloat {
    // Try i128 first (covers most practical integers).
    if let Ok(v) = <BigInt as TryInto<i128>>::try_into(n.clone()) {
        return BigFloat::from_i128(v, prec);
    }
    // Fallback: parse through f64 (loses precision for huge integers).
    let s = n.to_string();
    if let Ok(f) = s.parse::<f64>()
        && f.is_finite()
    {
        return BigFloat::from_f64(f, prec);
    }
    // Last resort.
    BigFloat::from_i32(0, prec)
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

/// Format a `BigFloat` as a decimal string with `digits` significant digits.
///
/// Uses astro-float's `convert_to_radix` for reliable base-10 conversion,
/// then formats the result with proper decimal point placement.
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

    // Truncate to requested digit count.
    let n = (digits as usize).min(mantissa.len());
    if n == 0 {
        return Ok("0".to_string());
    }

    let digit_chars: Vec<char> = mantissa[..n].iter().map(|&d| (b'0' + d) as char).collect();

    // The "adjusted exponent" is exponent-1 (shifting from 0.ddd to d.ddd notation).
    let adj_exp = exponent as i64 - 1;

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
    use crate::arena::Arena;

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
        assert_evalf_starts_with(&a, a.e_const, 15, "2.71828182845904");
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
        assert_evalf_starts_with(&a, expr, 15, "2.71828182845904");
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
        let evaluated = crate::subs::subs(&mut a, expr, x, three);
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
}
