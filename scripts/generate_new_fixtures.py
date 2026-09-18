#!/usr/bin/env python3
"""Generate cross-validation fixtures for symplex correctness audit.

This script produces ground-truth values from SymPy for:
  - Definite integrals (antiderivative evaluated at bounds)
  - FTC verification points (integrand value at test points)
  - ODE solution verification residuals
  - Simplification identity checks
  - Gosper sum closed forms
  - Series expansion coefficients

Usage:
    /Users/chris.gorski/repos/math/symplex/.venv/bin/python scripts/generate_new_fixtures.py
        # writes tests/fixtures/new_capabilities.json (deterministic)
    ... --check   # exit 1 if the committed file would change

Deterministic (no timestamps, sorted keys) and bounded: every fixture is
built under ``PER_FIXTURE_TIMEOUT`` seconds; a fixture that times out or
whose SymPy computation raises is *kept* with ``sympy_timeout`` /
``sympy_error`` so the consumer can report it as SKIPPED instead of the case
silently vanishing.

Requires: sympy >= 1.12
"""

import json
import os
import signal
import sys

try:
    import sympy as sp
except ImportError:
    print("ERROR: sympy is required. Install with: pip install sympy", file=sys.stderr)
    sys.exit(1)

x, y, a, n, k = sp.symbols("x y a n k")

fixtures = []
fixture_id = 0


def next_id():
    global fixture_id
    fixture_id += 1
    return fixture_id


PER_FIXTURE_TIMEOUT = 20.0  # seconds of wall clock per fixture


class _FixtureTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _FixtureTimeout()


def guarded(category, subcategory, label, make_fn, *args):
    """Build one fixture under a wall-clock limit.  Never drops a case: a
    timeout yields ``{"sympy_timeout": true}``, a failure ``{"sympy_error"}``."""
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, PER_FIXTURE_TIMEOUT)
    try:
        fx = make_fn(*args)
    except _FixtureTimeout:
        print(f"  TIMEOUT {label}", file=sys.stderr)
        fx = {"sympy_timeout": True}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if fx is None:
        fx = {"sympy_error": "generator could not build this fixture (see log)"}
    if isinstance(fx, dict) and "__error__" in fx:
        fx = {"sympy_error": fx["__error__"]}
    if "category" not in fx:
        fx.update({"id": next_id(), "category": category, "subcategory": subcategory, "label": label})
    fixtures.append(fx)
    return fx


def safe_float(val):
    """Convert a sympy expression to float, returning None on failure."""
    try:
        f = complex(val.evalf(20))
        return {"re": float(f.real), "im": float(f.imag)}
    except Exception:
        return None


def make_definite_integral_fixture(label, integrand, var, lo, hi, category_tag="integration"):
    """Create a fixture for a definite integral with FTC verification."""
    try:
        antideriv = sp.integrate(integrand, var)
        antideriv_str = str(antideriv)

        # Definite integral value
        F_hi = antideriv.subs(var, hi)
        F_lo = antideriv.subs(var, lo)
        definite_val = safe_float(F_hi - F_lo)
        if definite_val is None:
            return None

        # FTC check points: evaluate integrand at several points in [lo, hi]
        ftc_points = []
        test_xs = [
            lo + (hi - lo) * sp.Rational(1, 4),
            lo + (hi - lo) * sp.Rational(1, 2),
            lo + (hi - lo) * sp.Rational(3, 4),
        ]
        for tx in test_xs:
            f_val = safe_float(integrand.subs(var, tx))
            Fp_val = safe_float(sp.diff(antideriv, var).subs(var, tx))
            if f_val is not None and Fp_val is not None:
                ftc_points.append({
                    "x": float(tx.evalf()),
                    "integrand_value": f_val,
                    "antideriv_deriv_value": Fp_val,
                })

        return {
            "id": next_id(),
            "category": "definite_integral",
            "subcategory": category_tag,
            "label": label,
            "input": str(integrand),
            "variable": str(var),
            "lo": float(sp.N(lo)),
            "hi": float(sp.N(hi)),
            "antiderivative": antideriv_str,
            "definite_value": definite_val,
            "ftc_points": ftc_points,
        }
    except Exception as e:
        print(f"  SKIP {label}: {e}", file=sys.stderr)
        return {"__error__": f"{type(e).__name__}: {e}"}


def make_ftc_fixture(label, integrand, var, test_points, category_tag="ftc_check"):
    """Create a fixture that stores integrand values at points for FTC verification."""
    try:
        antideriv = sp.integrate(integrand, var)
        antideriv_str = str(antideriv)

        # Check it's not unevaluated
        if antideriv.has(sp.Integral):
            return None

        points = []
        for pt in test_points:
            f_val = safe_float(integrand.subs(var, pt))
            ad_val = safe_float(antideriv.subs(var, pt))
            if f_val is not None and ad_val is not None:
                points.append({
                    "x": float(sp.N(pt)),
                    "integrand_value": f_val,
                    "antideriv_value": ad_val,
                })

        if len(points) < 2:
            return None

        return {
            "id": next_id(),
            "category": "ftc_check",
            "subcategory": category_tag,
            "label": label,
            "input": str(integrand),
            "variable": str(var),
            "antiderivative": antideriv_str,
            "eval_points": points,
        }
    except Exception as e:
        print(f"  SKIP ftc {label}: {e}", file=sys.stderr)
        return {"__error__": f"{type(e).__name__}: {e}"}


def make_ode_fixture(label, ode_expr, func, var, category_tag="ode"):
    """Create a fixture for ODE solution verification."""
    try:
        f = sp.Function("y")
        # Convert string-based ODE to sympy ODE
        ode_sym = ode_expr
        sol = sp.dsolve(ode_sym, f(var))

        # Get particular solution by setting constants to 1
        particular = sol.rhs
        for c in particular.free_symbols:
            if str(c).startswith("C"):
                particular = particular.subs(c, 1)

        # Verify: substitute back into ODE
        test_points_vals = [sp.Rational(1, 2), sp.Rational(1, 1), sp.Rational(3, 2)]
        residuals = []
        for pt in test_points_vals:
            try:
                # Substitute y = particular, compute ODE residual
                res = ode_sym.subs(f(var), particular).doit()
                res = res.subs(var, pt)
                res_val = safe_float(res)
                if res_val is not None:
                    residuals.append({
                        "x": float(sp.N(pt)),
                        "residual": res_val,
                    })
            except Exception:
                pass

        return {
            "id": next_id(),
            "category": "ode_verify",
            "subcategory": category_tag,
            "label": label,
            "ode": str(ode_expr),
            "solution": str(sol.rhs),
            "particular": str(particular),
            "residuals": residuals,
        }
    except Exception as e:
        print(f"  SKIP ode {label}: {e}", file=sys.stderr)
        return {"__error__": f"{type(e).__name__}: {e}"}


def make_simplify_fixture(label, original, simplified_expected, test_points):
    """Create fixture verifying numerical equivalence of simplified form."""
    points = []
    for pt in test_points:
        orig_val = safe_float(original.subs(x, pt))
        simp_val = safe_float(simplified_expected.subs(x, pt))
        if orig_val is not None and simp_val is not None:
            points.append({
                "x": float(sp.N(pt)),
                "original_value": orig_val,
                "simplified_value": simp_val,
            })

    if len(points) < 2:
        return None

    return {
        "id": next_id(),
        "category": "simplify_verify",
        "subcategory": "trig_identity",
        "label": label,
        "input": str(original),
        "expected": str(simplified_expected),
        "eval_points": points,
    }


def make_gosper_fixture(label, summand_expr, var, lo, hi):
    """Create fixture for Gosper summation verification."""
    try:
        closed = sp.summation(summand_expr, (var, lo, hi))
        closed_val = safe_float(closed)

        # Brute force for small ranges
        brute = sum(summand_expr.subs(var, i) for i in range(int(lo), int(hi) + 1))
        brute_val = safe_float(brute)

        if closed_val is None or brute_val is None:
            return None

        return {
            "id": next_id(),
            "category": "gosper_sum",
            "label": label,
            "input": str(summand_expr),
            "variable": str(var),
            "lo": int(lo),
            "hi": int(hi),
            "closed_form": str(closed),
            "closed_value": closed_val,
            "brute_force_value": brute_val,
        }
    except Exception as e:
        print(f"  SKIP gosper {label}: {e}", file=sys.stderr)
        return {"__error__": f"{type(e).__name__}: {e}"}


def make_series_fixture(label, expr_sym, var, point, order):
    """Create fixture for series expansion verification."""
    try:
        series = sp.series(expr_sym, var, point, order)
        poly_part = sp.Poly(series.removeO(), var) if series.removeO().is_polynomial(var) else None

        # Evaluate truncated series at a few points near expansion point
        trunc = series.removeO()
        test_pts = [
            point + sp.Rational(1, 10),
            point + sp.Rational(1, 5),
            point + sp.Rational(3, 10),
        ]

        points = []
        for pt in test_pts:
            exact_val = safe_float(expr_sym.subs(var, pt))
            approx_val = safe_float(trunc.subs(var, pt))
            if exact_val is not None and approx_val is not None:
                points.append({
                    "x": float(sp.N(pt)),
                    "exact_value": exact_val,
                    "series_value": approx_val,
                })

        return {
            "id": next_id(),
            "category": "series_verify",
            "label": label,
            "input": str(expr_sym),
            "variable": str(var),
            "point": float(sp.N(point)),
            "order": order,
            "series_str": str(trunc),
            "eval_points": points,
        }
    except Exception as e:
        print(f"  SKIP series {label}: {e}", file=sys.stderr)
        return {"__error__": f"{type(e).__name__}: {e}"}


# ══════════════════════════════════════════════════════════════════════
# Generate fixtures
# ══════════════════════════════════════════════════════════════════════

print("Generating new capability fixtures...", file=sys.stderr)

# ── 1. Definite integrals with FTC verification ──────────────────────
print("  Definite integrals...", file=sys.stderr)

definite_cases = [
    # (label, integrand, var, lo, hi, tag)
    ("x_squared", x**2, x, sp.Rational(1, 2), 1, "basic"),
    ("x_cubed", x**3, x, sp.Rational(1, 2), 1, "basic"),
    ("sin_x", sp.sin(x), x, sp.Rational(1, 2), 1, "basic"),
    ("cos_x", sp.cos(x), x, sp.Rational(1, 2), 1, "basic"),
    ("exp_x", sp.exp(x), x, sp.Rational(1, 2), 1, "basic"),
    ("one_over_x", 1/x, x, sp.Rational(1, 2), 2, "basic"),
    ("tan_x", sp.tan(x), x, sp.Rational(1, 4), 1, "basic"),
    ("ln_x", sp.log(x), x, sp.Rational(1, 2), 2, "basic"),
    # By-parts
    ("x_exp_x", x * sp.exp(x), x, sp.Rational(1, 2), 1, "by_parts"),
    ("x_sin_x", x * sp.sin(x), x, sp.Rational(1, 2), 1, "by_parts"),
    ("x_cos_x", x * sp.cos(x), x, sp.Rational(1, 2), 1, "by_parts"),
    ("x2_exp_x", x**2 * sp.exp(x), x, sp.Rational(1, 2), 1, "by_parts"),
    ("x_ln_x", x * sp.log(x), x, sp.Rational(1, 2), 2, "by_parts"),
    ("ln_x_sq", sp.log(x)**2, x, sp.Rational(1, 2), 2, "by_parts"),
    ("x2_sin_x", x**2 * sp.sin(x), x, sp.Rational(1, 2), 1, "by_parts"),
    ("x3_exp_x", x**3 * sp.exp(x), x, sp.Rational(1, 2), 1, "by_parts"),
    # Trig powers
    ("sin2_x", sp.sin(x)**2, x, sp.Rational(1, 2), 1, "trig_power"),
    ("cos2_x", sp.cos(x)**2, x, sp.Rational(1, 2), 1, "trig_power"),
    ("sin3_x", sp.sin(x)**3, x, sp.Rational(1, 2), 1, "trig_power"),
    ("sin4_x", sp.sin(x)**4, x, sp.Rational(1, 2), 1, "trig_power"),
    ("tan2_x", sp.tan(x)**2, x, sp.Rational(1, 4), 1, "trig_power"),
    ("sin_x_cos_x", sp.sin(x)*sp.cos(x), x, sp.Rational(1, 2), 1, "trig_power"),
    ("sin2_cos2", sp.sin(x)**2 * sp.cos(x)**2, x, sp.Rational(1, 2), 1, "trig_power"),
    # Rational functions
    ("inv_x2p1", 1/(x**2 + 1), x, sp.Rational(1, 2), 1, "rational"),
    ("inv_x2m1", 1/(x**2 - 1), x, 2, 3, "rational"),
    ("x_over_x2p1_sq", x/(x**2 + 1)**2, x, sp.Rational(1, 2), 1, "rational"),
    # Sqrt forms
    ("inv_sqrt_1mx2", 1/sp.sqrt(1 - x**2), x, sp.Rational(1, 4), sp.Rational(1, 2), "sqrt"),
    ("inv_sqrt_x2p1", 1/sp.sqrt(x**2 + 1), x, sp.Rational(1, 2), 1, "sqrt"),
    ("x_over_sqrt_x2p1", x/sp.sqrt(x**2 + 1), x, sp.Rational(1, 2), 1, "sqrt"),
    # Cyclic IBP
    ("exp_sin", sp.exp(x)*sp.sin(x), x, sp.Rational(1, 2), 1, "cyclic"),
    ("exp_cos", sp.exp(x)*sp.cos(x), x, sp.Rational(1, 2), 1, "cyclic"),
    # Inverse trig
    ("asin_x", sp.asin(x), x, sp.Rational(1, 4), sp.Rational(1, 2), "inv_trig"),
    ("acos_x", sp.acos(x), x, sp.Rational(1, 4), sp.Rational(1, 2), "inv_trig"),
    ("atan_x", sp.atan(x), x, sp.Rational(1, 2), 1, "inv_trig"),
    # Parametric (with specific a values)
    ("sin_2x", sp.sin(2*x), x, sp.Rational(1, 2), 1, "parametric"),
    ("exp_2x", sp.exp(2*x), x, sp.Rational(1, 2), 1, "parametric"),
    ("sin_3x", sp.sin(3*x), x, sp.Rational(1, 2), 1, "parametric"),
    ("x_exp_2x", x * sp.exp(2*x), x, sp.Rational(1, 2), 1, "parametric"),
    # Hyperbolic
    ("sinh_x", sp.sinh(x), x, sp.Rational(1, 2), 1, "hyperbolic"),
    ("cosh_x", sp.cosh(x), x, sp.Rational(1, 2), 1, "hyperbolic"),
    ("tanh_x", sp.tanh(x), x, sp.Rational(1, 2), 1, "hyperbolic"),
    ("sinh2_x", sp.sinh(x)**2, x, sp.Rational(1, 2), 1, "hyperbolic"),
    # Linear substitution
    ("2x_plus_1_pow5", (2*x + 1)**5, x, sp.Rational(1, 2), 1, "linear_sub"),
    ("inv_3x_plus_2", 1/(3*x + 2), x, sp.Rational(1, 2), 1, "linear_sub"),
    ("sqrt_2x_plus_1", sp.sqrt(2*x + 1), x, sp.Rational(1, 2), 1, "linear_sub"),
    # U-substitution
    ("2x_exp_x2", 2*x*sp.exp(x**2), x, sp.Rational(1, 2), 1, "u_sub"),
    ("cos_exp_sin", sp.cos(x)*sp.exp(sp.sin(x)), x, sp.Rational(1, 2), 1, "u_sub"),
    ("x_over_x2p1", x/(x**2 + 1), x, sp.Rational(1, 2), 1, "u_sub"),
    # Completing the square
    ("inv_x2_2x_5", 1/(x**2 + 2*x + 5), x, sp.Rational(1, 2), 1, "complete_sq"),
]

for case in definite_cases:
    label, integrand, var, lo, hi, tag = case
    guarded("definite_integral", tag, label, make_definite_integral_fixture, label, integrand, var, lo, hi, tag)

# ── 2. FTC verification (integrand vs derivative of antiderivative) ──
print("  FTC verification points...", file=sys.stderr)

ftc_test_points = [sp.Rational(1, 2), sp.Rational(3, 4), sp.Rational(1, 1), sp.Rational(3, 2), sp.Rational(2, 1)]

ftc_cases = [
    ("x^2", x**2, x, "basic"),
    ("sin(x)", sp.sin(x), x, "basic"),
    ("cos(x)", sp.cos(x), x, "basic"),
    ("exp(x)", sp.exp(x), x, "basic"),
    ("x*exp(x)", x*sp.exp(x), x, "by_parts"),
    ("x*sin(x)", x*sp.sin(x), x, "by_parts"),
    ("ln(x)^2", sp.log(x)**2, x, "by_parts"),
    ("sin^2(x)", sp.sin(x)**2, x, "trig_power"),
    ("cos^2(x)", sp.cos(x)**2, x, "trig_power"),
    ("tan^2(x)", sp.tan(x)**2, x, "trig_power"),
    ("sinh^2(x)", sp.sinh(x)**2, x, "hyperbolic"),
    ("exp(x)*sin(x)", sp.exp(x)*sp.sin(x), x, "cyclic"),
    ("exp(x)*cos(x)", sp.exp(x)*sp.cos(x), x, "cyclic"),
    ("1/(x^2+1)", 1/(x**2+1), x, "rational"),
    ("asin(x)", sp.asin(x), x, "inv_trig"),
    ("atan(x)", sp.atan(x), x, "inv_trig"),
    ("sin(2*x)", sp.sin(2*x), x, "parametric"),
    ("exp(2*x)", sp.exp(2*x), x, "parametric"),
    ("sinh(x)", sp.sinh(x), x, "hyperbolic"),
    ("cosh(x)", sp.cosh(x), x, "hyperbolic"),
    ("tanh(x)", sp.tanh(x), x, "hyperbolic"),
    ("(2x+1)^5", (2*x+1)**5, x, "linear_sub"),
    ("sqrt(2x+1)", sp.sqrt(2*x+1), x, "linear_sub"),
    ("2x*exp(x^2)", 2*x*sp.exp(x**2), x, "u_sub"),
    ("x/(x^2+1)", x/(x**2+1), x, "u_sub"),
]

for label, integrand, var, tag in ftc_cases:
    pts = ftc_test_points
    # Restrict domain for asin
    if "asin" in label:
        pts = [sp.Rational(1, 4), sp.Rational(1, 3), sp.Rational(1, 2)]
    guarded("ftc_check", tag, label, make_ftc_fixture, label, integrand, var, pts, tag)

# ── 3. Simplification identity verification ──────────────────────────
print("  Simplification identities...", file=sys.stderr)

simp_points = [sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(1, 1),
               sp.Rational(3, 2), sp.Rational(2, 1)]

simp_cases = [
    ("sin2+cos2=1", sp.sin(x)**2 + sp.cos(x)**2, sp.Integer(1)),
    ("tan2+1=sec2", sp.tan(x)**2 + 1, 1/sp.cos(x)**2),
    ("1-sin2=cos2", 1 - sp.sin(x)**2, sp.cos(x)**2),
    ("cosh2-sinh2=1", sp.cosh(x)**2 - sp.sinh(x)**2, sp.Integer(1)),
    ("2sin_cos=sin2x", 2*sp.sin(x)*sp.cos(x), sp.sin(2*x)),
    ("cos2x_identity", sp.cos(x)**2 - sp.sin(x)**2, sp.cos(2*x)),
    ("exp_ln_x", sp.exp(sp.log(x)), x),
    ("ln_exp_x", sp.log(sp.exp(x)), x),
]

for label, orig, expected in simp_cases:
    guarded("simplify_verify", "trig_identity", label, make_simplify_fixture, label, orig, expected, simp_points)

# ── 4. Gosper summation verification ─────────────────────────────────
print("  Gosper sums...", file=sys.stderr)

gosper_cases = [
    ("sum_k", k, k, 1, 100),
    ("sum_k2", k**2, k, 1, 50),
    ("sum_k3", k**3, k, 1, 30),
    ("sum_2k", 2**k, k, 0, 15),
    ("sum_1_over_k_kp1", 1/(k*(k+1)), k, 1, 100),
    ("sum_1_over_k_kp1_kp2", 1/(k*(k+1)*(k+2)), k, 1, 50),
    ("sum_fib_like", k*(k+1)/2, k, 1, 20),
    ("sum_alternating_k", (-1)**k * k, k, 1, 99),
]

for label, summand, var, lo, hi in gosper_cases:
    guarded("gosper_sum", "closed_form", label, make_gosper_fixture, label, summand, var, lo, hi)

# ── 5. Series expansion verification ─────────────────────────────────
print("  Series expansions...", file=sys.stderr)

series_cases = [
    ("sin_x", sp.sin(x), x, 0, 8),
    ("cos_x", sp.cos(x), x, 0, 8),
    ("exp_x", sp.exp(x), x, 0, 8),
    ("ln_1px", sp.log(1+x), x, 0, 8),
    ("inv_1mx", 1/(1-x), x, 0, 8),
    ("atan_x", sp.atan(x), x, 0, 8),
    ("sinh_x", sp.sinh(x), x, 0, 8),
    ("cosh_x", sp.cosh(x), x, 0, 8),
    ("tanh_x", sp.tanh(x), x, 0, 8),
    ("sqrt_1px", sp.sqrt(1+x), x, 0, 6),
    ("asin_x", sp.asin(x), x, 0, 8),
]

for label, expr_sym, var, point, order in series_cases:
    guarded("series_verify", "taylor", label, make_series_fixture, label, expr_sym, var, point, order)

# ── 6. Parametric integration with specific parameter values ─────────
print("  Parametric integrals at specific parameter values...", file=sys.stderr)

param_cases = [
    # (label, integrand_with_a, var, a_val, lo, hi)
    ("sin_ax_a2", sp.sin(a*x), x, 2, sp.Rational(1, 2), 1),
    ("exp_ax_a2", sp.exp(a*x), x, 2, sp.Rational(1, 2), 1),
    ("sin_ax_a3", sp.sin(a*x), x, 3, sp.Rational(1, 2), 1),
    ("cos_ax_a2", sp.cos(a*x), x, 2, sp.Rational(1, 2), 1),
    ("x_exp_ax_a2", x*sp.exp(a*x), x, 2, sp.Rational(1, 2), 1),
    ("inv_x2pa2_a3", 1/(x**2 + a**2), x, 3, sp.Rational(1, 2), 1),
    ("ax2_a5", a*x**2, x, 5, sp.Rational(1, 2), 1),
]

for label, integrand, var, a_val, lo, hi in param_cases:
    concrete = integrand.subs(a, a_val)
    guarded("definite_integral", "parametric_concrete", label, make_definite_integral_fixture, label, concrete, var, lo, hi, "parametric_concrete")


# ══════════════════════════════════════════════════════════════════════
# Write output
# ══════════════════════════════════════════════════════════════════════

output = {
    "generated_by": f"SymPy {sp.__version__}",
    "generator": "scripts/generate_new_fixtures.py",
    "fixture_count": len(fixtures),
    "description": "Correctness audit fixtures for symplex new capabilities",
    "categories": {
        "definite_integral": "Definite integral values via F(hi)-F(lo)",
        "ftc_check": "FTC verification: f(x) values and F(x) values at test points",
        "simplify_verify": "Numerical equivalence of original and simplified expressions",
        "gosper_sum": "Closed-form sums vs brute-force computation",
        "series_verify": "Truncated series vs exact function at nearby points",
        "ode_verify": "ODE solution residuals (should be zero)",
    },
    "fixtures": fixtures,
}

# Ensure output directory exists
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "tests", "fixtures")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "new_capabilities.json")
text = json.dumps(output, indent=2, sort_keys=True) + "\n"
if "--check" in sys.argv:
    with open(out_path) as f:
        if f.read() != text:
            print("MISMATCH: fixture file differs from generator output", file=sys.stderr)
            sys.exit(1)
    print("OK: fixture file is up to date", file=sys.stderr)
else:
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nGenerated {len(fixtures)} fixtures -> {out_path}", file=sys.stderr)
gaps = [f for f in fixtures if f.get("sympy_timeout") or f.get("sympy_error")]
print(f"  oracle gaps kept (timeout/error): {len(gaps)}", file=sys.stderr)

# Print summary by category
from collections import Counter
cats = Counter(f["category"] for f in fixtures)
for cat, count in sorted(cats.items()):
    print(f"  {cat}: {count}", file=sys.stderr)
