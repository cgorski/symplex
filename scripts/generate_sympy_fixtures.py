#!/usr/bin/env python3
"""Generate ALL 263 SymPy cross-validation fixtures for symplex.

Covers every category:
  diff             (59):  basic(15), trig(8), exp_log(8), hyperbolic(5),
                          chain_rule(7), higher_order(8), partial(8)
  integrate        (34):  basic(10), trig(8), exp_log(6), by_parts(4), rational(6)
  definite_integral(15)
  simplify         (21):  trig(8), algebraic(8), exp_log(5)
  expand           (14):  algebraic(8), trig(6)
  solve            (23):  linear(5), quadratic(10), cubic(5), other(3)
  eval             (15)
  series           (10)
  limit            (15)
  matrix           (22):  det(8), inverse(5), eigenvalue(4), trace(3), multiply(2)
  algebra          (28):  factor(8), collect(5), together(5), cancel(5), apart(5)
  evalf             (4)
  special_func      (3):  factorial(3)

Usage:
    cd math/symplex
    source .venv/bin/activate
    python3 scripts/generate_sympy_fixtures.py > tests/fixtures/sympy_cross_validation.json 2>fixture_generation.log
"""

import datetime
import json
import math
import sys
import traceback

import sympy as sp
from sympy import (
    E,
    I,
    Matrix,
    Rational,
    atan,
    cos,
    cosh,
    exp,
    factorial,
    log,
    oo,
    pi,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)

x, y = sp.symbols("x y")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

STD_PTS_X = [
    {x: Rational(3, 2)},  # 1.5
    {x: Rational(7, 10)},  # 0.7
    {x: Rational(2, 1)},  # 2.0
    {x: Rational(-333, 1000)},  # -0.333
]

STD_PTS_XY = [
    {x: Rational(3, 2), y: Rational(1, 2)},  # x=1.5, y=0.5
    {x: Rational(7, 10), y: Rational(-3, 10)},  # x=0.7, y=-0.3
]


def eval_at(expr, subs_dict):
    """Evaluate expression numerically at given substitution point.

    Returns {"re": float, "im": float} or None on failure.
    """
    try:
        val = complex(expr.subs(subs_dict).evalf())
        re_part = val.real
        im_part = val.imag
        # Clean up tiny imaginary noise from floating point
        if abs(im_part) < 1e-15:
            im_part = 0.0
        if not (math.isfinite(re_part) and math.isfinite(im_part)):
            return None
        return {"re": re_part, "im": im_part}
    except Exception:
        return None


def subs_display(subs_dict):
    """Convert sympy subs dict to JSON-friendly {"x": 1.5, ...}."""
    return {str(k): float(v) for k, v in subs_dict.items()}


def make_eval_points(expr, pts=None):
    """Generate eval_points list, auto-detecting 1-var vs 2-var."""
    if pts is None:
        pts = STD_PTS_XY if y in expr.free_symbols else STD_PTS_X
    result = []
    for p in pts:
        val = eval_at(expr, p)
        if val is not None:
            result.append({"subs": subs_display(p), "value": val})
    return result


def make_value(expr):
    """Evaluate a constant expression to {"re": ..., "im": ...} or None."""
    try:
        val = complex(expr.evalf())
        re_part = val.real
        im_part = val.imag
        if abs(im_part) < 1e-15:
            im_part = 0.0
        if not (math.isfinite(re_part) and math.isfinite(im_part)):
            return None
        return {"re": re_part, "im": im_part}
    except Exception:
        return None


def matrix_to_list(m):
    """Convert sympy Matrix to list of lists of Python numbers."""
    result = []
    for i in range(m.rows):
        row = []
        for j in range(m.cols):
            v = m[i, j]
            if v.is_Integer:
                row.append(int(v))
            elif v.is_Rational:
                row.append(float(v))
            else:
                row.append(float(v.evalf()))
        result.append(row)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Category: diff  (59 fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_diff():
    fixtures = []

    # ── diff/basic (15) ──────────────────────────────────────────────────────
    basic = [
        "x**3 + 2*x + 1",
        "x**5 - 3*x**3 + 2*x",
        "x**4 + x**2",
        "5*x**3 - 2*x**2 + x - 7",
        "x**6",
        "1/x",
        "sqrt(x)",
        "x**(3/2)",
        "x**(-2)",
        "x**10",
        "(x + 1)*(x - 1)",
        "x**3/3 - x**2/2 + x",
        "3*x**4 - 8*x + 1",
        "x**(1/3)",
        "x**2 + 3*x + 5",
    ]
    for s in basic:
        expr = sp.sympify(s)
        result = sp.diff(expr, x)
        fixtures.append(
            {
                "category": "diff",
                "subcategory": "basic",
                "input": s,
                "operation": "diff",
                "variable": "x",
                "params": {},
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── diff/trig (8) ────────────────────────────────────────────────────────
    trig = [
        "sin(x)",
        "cos(x)",
        "tan(x)",
        "1/cos(x)",
        "sin(x)*cos(x)",
        "sin(x)**2",
        "cos(x)**2",
        "tan(x)**2",
    ]
    for s in trig:
        expr = sp.sympify(s)
        result = sp.diff(expr, x)
        fixtures.append(
            {
                "category": "diff",
                "subcategory": "trig",
                "input": s,
                "operation": "diff",
                "variable": "x",
                "params": {},
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── diff/exp_log (8) ─────────────────────────────────────────────────────
    exp_log = [
        "exp(x)",
        "exp(2*x)",
        "log(x)",
        "log(x**2 + 1)",
        "x*exp(x)",
        "x**2*exp(x)",
        "exp(x)*sin(x)",
        "exp(-x)",
    ]
    for s in exp_log:
        expr = sp.sympify(s)
        result = sp.diff(expr, x)
        fixtures.append(
            {
                "category": "diff",
                "subcategory": "exp_log",
                "input": s,
                "operation": "diff",
                "variable": "x",
                "params": {},
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── diff/hyperbolic (5) ──────────────────────────────────────────────────
    hyp = [
        "sinh(x)",
        "cosh(x)",
        "tanh(x)",
        "sinh(x)**2",
        "cosh(x)*sinh(x)",
    ]
    for s in hyp:
        expr = sp.sympify(s)
        result = sp.diff(expr, x)
        fixtures.append(
            {
                "category": "diff",
                "subcategory": "hyperbolic",
                "input": s,
                "operation": "diff",
                "variable": "x",
                "params": {},
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── diff/chain_rule (7) ──────────────────────────────────────────────────
    chain = [
        "sin(x**2)",
        "exp(sin(x))",
        "log(sin(x))",
        "cos(exp(x))",
        "sqrt(x**2 + 1)",
        "sin(3*x + 1)",
        "exp(x**2)",
    ]
    for s in chain:
        expr = sp.sympify(s)
        result = sp.diff(expr, x)
        fixtures.append(
            {
                "category": "diff",
                "subcategory": "chain_rule",
                "input": s,
                "operation": "diff",
                "variable": "x",
                "params": {},
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── diff/higher_order (8) ────────────────────────────────────────────────
    higher = [
        ("x**4", 2),
        ("x**5", 2),
        ("x**6", 3),
        ("sin(x)", 2),
        ("cos(x)", 2),
        ("exp(x)", 3),
        ("x**3 + x**2", 2),
        ("sin(x)", 4),
    ]
    for s, order in higher:
        expr = sp.sympify(s)
        result = sp.diff(expr, x, order)
        fixtures.append(
            {
                "category": "diff",
                "subcategory": "higher_order",
                "input": s,
                "operation": "diff",
                "variable": "x",
                "order": order,
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── diff/partial (8) ─────────────────────────────────────────────────────
    partial = [
        ("x**2*y + x*y**2", "y"),
        ("x**2*y + x*y**2", "x"),
        ("x*y", "x"),
        ("x*y", "y"),
        ("x**2 + y**2", "x"),
        ("x**2 + y**2", "y"),
        ("exp(x + y)", "x"),
        ("sin(x)*cos(y)", "x"),
    ]
    for s, dv in partial:
        expr = sp.sympify(s)
        diff_sym = x if dv == "x" else y
        result = sp.diff(expr, diff_sym)
        fixtures.append(
            {
                "category": "diff",
                "subcategory": "partial",
                "input": s,
                "diff_var": dv,
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    return fixtures  # 59


# ═══════════════════════════════════════════════════════════════════════════════
# Category: integrate  (34 fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_integrate():
    fixtures = []

    # ── integrate/basic (10) ─────────────────────────────────────────────────
    basic = [
        "x**2",
        "x**3",
        "x**4",
        "x**5",
        "1/x",
        "sqrt(x)",
        "x**(3/2)",
        "x + 1",
        "3*x**2 + 2*x + 1",
        "x**(-2)",
    ]
    for s in basic:
        expr = sp.sympify(s)
        result = sp.integrate(expr, x)
        fixtures.append(
            {
                "category": "integrate",
                "subcategory": "basic",
                "input": s,
                "operation": "integrate",
                "variable": "x",
                "params": {},
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── integrate/trig (8) ───────────────────────────────────────────────────
    trig = [
        "sin(x)",
        "cos(x)",
        "sin(x)**2",
        "cos(x)**2",
        "1/cos(x)**2",
        "sin(x)*cos(x)",
        "sin(2*x)",
        "cos(2*x)",
    ]
    for s in trig:
        expr = sp.sympify(s)
        result = sp.integrate(expr, x)
        fixtures.append(
            {
                "category": "integrate",
                "subcategory": "trig",
                "input": s,
                "operation": "integrate",
                "variable": "x",
                "params": {},
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── integrate/exp_log (6) ────────────────────────────────────────────────
    exp_log = [
        "exp(x)",
        "exp(2*x)",
        "exp(-x)",
        "x*exp(x)",
        "x**2*exp(x)",
        "exp(x)*sin(x)",
    ]
    for s in exp_log:
        expr = sp.sympify(s)
        result = sp.integrate(expr, x)
        fixtures.append(
            {
                "category": "integrate",
                "subcategory": "exp_log",
                "input": s,
                "operation": "integrate",
                "variable": "x",
                "params": {},
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── integrate/by_parts (4) ───────────────────────────────────────────────
    by_parts = [
        "x*sin(x)",
        "x*cos(x)",
        "x*log(x)",
        "x**2*sin(x)",
    ]
    for s in by_parts:
        expr = sp.sympify(s)
        result = sp.integrate(expr, x)
        fixtures.append(
            {
                "category": "integrate",
                "subcategory": "by_parts",
                "input": s,
                "operation": "integrate",
                "variable": "x",
                "params": {},
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── integrate/rational (6) ───────────────────────────────────────────────
    rational = [
        "1/(x**2 + 1)",
        "1/(x + 1)",
        "x/(x**2 + 1)",
        "1/(x**2 + 4)",
        "x/(x**2 + 4)",
        "1/(x**2 + x + 1)",
    ]
    for s in rational:
        expr = sp.sympify(s)
        result = sp.integrate(expr, x)
        fixtures.append(
            {
                "category": "integrate",
                "subcategory": "rational",
                "input": s,
                "operation": "integrate",
                "variable": "x",
                "params": {},
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    return fixtures  # 34


# ═══════════════════════════════════════════════════════════════════════════════
# Category: definite_integral  (15 fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_definite_integral():
    fixtures = []

    cases = [
        ("x**2", "0", "1"),
        ("x**3", "0", "1"),
        ("sin(x)", "0", "pi"),
        ("cos(x)", "0", "pi/2"),
        ("exp(x)", "0", "1"),
        ("x", "0", "1"),
        ("x**2", "-1", "1"),
        ("1/(x**2 + 1)", "0", "1"),
        ("x**3", "1", "2"),
        ("sqrt(x)", "0", "4"),
        ("1/x", "1", "E"),
        ("sin(x)**2", "0", "pi"),
        ("x*exp(x)", "0", "1"),
        ("cos(x)**2", "0", "pi"),
        ("1/(1 + x**2)", "-1", "1"),
    ]

    for expr_str, lower_str, upper_str in cases:
        expr = sp.sympify(expr_str)
        lower = sp.sympify(lower_str)
        upper = sp.sympify(upper_str)
        result = sp.integrate(expr, (x, lower, upper))
        value = make_value(result)

        fixture = {
            "category": "definite_integral",
            "input": expr_str,
            "variable": "x",
            "lower": str(lower),
            "upper": str(upper),
            "sympy_result": str(result),
        }
        if value is not None:
            fixture["value"] = value
        fixtures.append(fixture)

    return fixtures  # 15


# ═══════════════════════════════════════════════════════════════════════════════
# Category: simplify  (21 fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_simplify():
    fixtures = []

    # ── simplify/trig (8) ────────────────────────────────────────────────────
    trig_cases = [
        "sin(x)**2 + cos(x)**2",
        "1 - sin(x)**2",
        "1 - cos(x)**2",
        "sin(2*x)/(2*cos(x))",
        "tan(x)*cos(x)",
        "sin(x)/cos(x)",
        "(1 - cos(2*x))/2",
        "(1 + cos(2*x))/2",
    ]
    for s in trig_cases:
        expr = sp.sympify(s)
        result = sp.simplify(expr)
        fixtures.append(
            {
                "category": "simplify",
                "subcategory": "trig",
                "input": s,
                "operation": "simplify",
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── simplify/algebraic (8) ───────────────────────────────────────────────
    alg_cases = [
        "(x**2 - 1)/(x - 1)",
        "(x**3 - x)/(x**2 - 1)",
        "(x**2 + 2*x + 1)/(x + 1)",
        "x*(x + 1) - x**2",
        "(x + 1)**2 - x**2 - 2*x",
        "x/x",
        "(x**2 - 4)/(x - 2)",
        "x**2/x",
    ]
    for s in alg_cases:
        expr = sp.sympify(s)
        result = sp.simplify(expr)
        fixtures.append(
            {
                "category": "simplify",
                "subcategory": "algebraic",
                "input": s,
                "operation": "simplify",
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── simplify/exp_log (5) ─────────────────────────────────────────────────
    exp_log_cases = [
        "exp(log(x))",
        "log(exp(x))",
        "exp(2*log(x))",
        "log(x**2)",
        "exp(log(x) + log(y))",
    ]
    for s in exp_log_cases:
        expr = sp.sympify(s)
        result = sp.simplify(expr)
        fixtures.append(
            {
                "category": "simplify",
                "subcategory": "exp_log",
                "input": s,
                "operation": "simplify",
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    return fixtures  # 21


# ═══════════════════════════════════════════════════════════════════════════════
# Category: expand  (14 fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_expand():
    fixtures = []

    # ── expand/algebraic (8) ─────────────────────────────────────────────────
    alg = [
        "(x + 1)**2",
        "(x + 1)**3",
        "(x + 1)**4",
        "(x + y)**2",
        "(x - 1)*(x + 1)",
        "(2*x + 3)**2",
        "(x + 1)*(x + 2)*(x + 3)",
        "(x**2 + 1)*(x - 1)",
    ]
    for s in alg:
        expr = sp.sympify(s)
        result = sp.expand(expr)
        fixtures.append(
            {
                "category": "expand",
                "subcategory": "algebraic",
                "input": s,
                "operation": "expand",
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── expand/trig (6) ──────────────────────────────────────────────────────
    trig = [
        "sin(2*x)",
        "cos(2*x)",
        "sin(3*x)",
        "cos(3*x)",
        "sin(x + y)",
        "cos(x + y)",
    ]
    for s in trig:
        expr = sp.sympify(s)
        result = sp.expand_trig(expr)
        fixtures.append(
            {
                "category": "expand",
                "subcategory": "trig",
                "input": s,
                "operation": "expand_trig",
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    return fixtures  # 14


# ═══════════════════════════════════════════════════════════════════════════════
# Category: solve  (23 fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_solve():
    fixtures = []

    def _make_solve(expr_str, subcat):
        expr = sp.sympify(expr_str)
        roots = sp.solve(expr, x)
        root_data = []
        for r in roots:
            try:
                val = complex(r.evalf())
                re_part = val.real
                im_part = val.imag
                if abs(im_part) < 1e-15:
                    im_part = 0.0
                if abs(re_part) < 1e-15 and abs(im_part) > 1e-10:
                    re_part = 0.0
                root_data.append(
                    {
                        "symbolic": str(r),
                        "re": re_part,
                        "im": im_part,
                    }
                )
            except Exception:
                root_data.append(
                    {
                        "symbolic": str(r),
                        "re": None,
                        "im": None,
                    }
                )
        fixtures.append(
            {
                "category": "solve",
                "subcategory": subcat,
                "input": expr_str,
                "operation": "solve",
                "variable": "x",
                "sympy_roots": root_data,
            }
        )

    # ── solve/linear (5) ─────────────────────────────────────────────────────
    for s in ["x - 3", "2*x - 6", "3*x + 9", "x/2 - 1", "5*x - 15"]:
        _make_solve(s, "linear")

    # ── solve/quadratic (10) ─────────────────────────────────────────────────
    for s in [
        "x**2 - 4",
        "x**2 - 5*x + 6",
        "x**2 + 1",
        "x**2 - 2",
        "x**2 - x - 6",
        "x**2 + x - 2",
        "x**2 - 9",
        "x**2 + 4*x + 4",
        "x**2 - 6*x + 9",
        "2*x**2 - 8",
    ]:
        _make_solve(s, "quadratic")

    # ── solve/cubic (5) ──────────────────────────────────────────────────────
    for s in [
        "x**3 - 1",
        "x**3 - 8",
        "x**3 - x",
        "x**3 + x**2 - 2*x",
        "x**3 - 6*x**2 + 11*x - 6",
    ]:
        _make_solve(s, "cubic")

    # ── solve/other (3) ──────────────────────────────────────────────────────
    for s in ["x**4 - 1", "x**4 - 16", "x**2 + x"]:
        _make_solve(s, "other")

    return fixtures  # 23


# ═══════════════════════════════════════════════════════════════════════════════
# Category: eval  (15 fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_eval():
    fixtures = []

    cases = [
        "sin(pi/6)",
        "sin(pi/4)",
        "sin(pi/3)",
        "sin(pi/2)",
        "cos(pi/6)",
        "cos(pi/3)",
        "cos(pi/4)",
        "tan(pi/4)",
        "tan(pi/6)",
        "tan(2*pi/3)",
        "exp(0)",
        "exp(1)",
        "log(1)",
        "log(E)",
        "sqrt(2)",
    ]

    for s in cases:
        expr = sp.sympify(s)
        value = make_value(expr)
        fixture = {
            "category": "eval",
            "input": s,
            "sympy_result": str(expr),
        }
        if value is not None:
            fixture["value"] = value
        fixtures.append(fixture)

    return fixtures  # 15


# ═══════════════════════════════════════════════════════════════════════════════
# Category: series  (10 fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_series():
    fixtures = []

    cases = [
        "sin(x)",
        "cos(x)",
        "exp(x)",
        "log(1 + x)",
        "1/(1 - x)",
        "1/(1 + x)",
        "tan(x)",
        "sinh(x)",
        "cosh(x)",
        "atan(x)",
    ]

    series_pt = [{x: Rational(1, 4)}]  # x = 0.25, close to 0 for convergence

    for s in cases:
        expr = sp.sympify(s)
        result = sp.series(expr, x, 0, 6).removeO()
        fixtures.append(
            {
                "category": "series",
                "input": s,
                "variable": "x",
                "point": "0",
                "order": 6,
                "sympy_result": str(result),
                "eval_points": make_eval_points(result, pts=series_pt),
            }
        )

    return fixtures  # 10


# ═══════════════════════════════════════════════════════════════════════════════
# Category: limit  (15 fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_limit():
    fixtures = []

    cases = [
        ("sin(x)/x", x, 0),
        ("(exp(x) - 1)/x", x, 0),
        ("(1 - cos(x))/x**2", x, 0),
        ("(x**2 - 1)/(x - 1)", x, 1),
        ("x/(x + 1)", x, sp.oo),
        ("1/x", x, sp.oo),
        ("exp(-x)", x, sp.oo),
        ("(2*x + 1)/(x + 1)", x, sp.oo),
        ("log(x)/x", x, sp.oo),
        ("tan(x)/x", x, 0),
        ("(exp(x) - 1 - x)/x**2", x, 0),
        ("x/(x**2 + 1)", x, sp.oo),
        ("(3*x**2 + 1)/(x**2 + 1)", x, sp.oo),
        ("x*exp(-x)", x, sp.oo),
        ("(x**3 - 1)/(x - 1)", x, 1),
    ]

    for expr_str, var, point in cases:
        expr = sp.sympify(expr_str)
        result = sp.limit(expr, var, point)
        value = make_value(result)

        fixture = {
            "category": "limit",
            "input": expr_str,
            "variable": str(var),
            "point": str(point),
            "sympy_result": str(result),
        }
        if value is not None:
            fixture["value"] = value
        fixtures.append(fixture)

    return fixtures  # 15


# ═══════════════════════════════════════════════════════════════════════════════
# Category: matrix  (22 fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_matrix():
    fixtures = []

    # ── matrix/det (8) ───────────────────────────────────────────────────────
    det_cases = [
        [[1, 2], [3, 4]],
        [[2, 0], [0, 2]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[2, 1], [1, 2]],
        [[3, -1], [2, 4]],
        [[1, 2, 3], [0, 1, 2], [0, 0, 1]],
        [[5]],
    ]
    for mat_data in det_cases:
        m = Matrix(mat_data)
        det_val = m.det()
        value = make_value(det_val)
        fixture = {
            "category": "matrix",
            "subcategory": "det",
            "matrix": mat_data,
            "operation": "det",
            "sympy_result": str(det_val),
        }
        if value is not None:
            fixture["value"] = value
        fixtures.append(fixture)

    # ── matrix/inverse (5) ───────────────────────────────────────────────────
    inv_cases = [
        [[1, 2], [3, 4]],
        [[2, 0], [0, 2]],
        [[1, 0], [0, 1]],
        [[2, 1], [1, 2]],
        [[3, -1], [2, 4]],
    ]
    for mat_data in inv_cases:
        m = Matrix(mat_data)
        try:
            inv = m.inv()
            fixtures.append(
                {
                    "category": "matrix",
                    "subcategory": "inverse",
                    "matrix": mat_data,
                    "operation": "inverse",
                    "sympy_result": str(inv),
                    "result_matrix": matrix_to_list(inv),
                }
            )
        except Exception as e:
            print(f"  WARN: matrix inverse failed for {mat_data}: {e}", file=sys.stderr)

    # ── matrix/eigenvalue (4) ────────────────────────────────────────────────
    eig_cases = [
        [[2, 0], [0, 3]],
        [[1, 2], [2, 1]],
        [[4, 1], [2, 3]],
        [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
    ]
    for mat_data in eig_cases:
        m = Matrix(mat_data)
        eigenvals = m.eigenvals()  # dict: eigenvalue -> multiplicity
        eig_list = []
        for eigval, mult in sorted(
            eigenvals.items(),
            key=lambda item: (
                complex(item[0].evalf()).real,
                complex(item[0].evalf()).imag,
            ),
        ):
            val = make_value(eigval)
            eig_list.append(
                {
                    "symbolic": str(eigval),
                    "re": val["re"] if val else None,
                    "im": val["im"] if val else None,
                    "multiplicity": int(mult),
                }
            )
        fixtures.append(
            {
                "category": "matrix",
                "subcategory": "eigenvalue",
                "matrix": mat_data,
                "operation": "eigenvalues",
                "eigenvalues": eig_list,
            }
        )

    # ── matrix/trace (3) ─────────────────────────────────────────────────────
    trace_cases = [
        [[1, 2], [3, 4]],
        [[5, 0], [0, 5]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ]
    for mat_data in trace_cases:
        m = Matrix(mat_data)
        tr = m.trace()
        value = make_value(tr)
        fixture = {
            "category": "matrix",
            "subcategory": "trace",
            "matrix": mat_data,
            "operation": "trace",
            "sympy_result": str(tr),
        }
        if value is not None:
            fixture["value"] = value
        fixtures.append(fixture)

    # ── matrix/multiply (2) ──────────────────────────────────────────────────
    mul_cases = [
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        ([[1, 0], [0, 1]], [[3, 4], [5, 6]]),
    ]
    for a_data, b_data in mul_cases:
        a = Matrix(a_data)
        b = Matrix(b_data)
        c = a * b
        fixtures.append(
            {
                "category": "matrix",
                "subcategory": "multiply",
                "matrix_a": a_data,
                "matrix_b": b_data,
                "operation": "multiply",
                "result_matrix": matrix_to_list(c),
            }
        )

    return fixtures  # 22


# ═══════════════════════════════════════════════════════════════════════════════
# Category: algebra  (28 fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_algebra():
    fixtures = []

    # ── algebra/factor (8) ───────────────────────────────────────────────────
    factor_cases = [
        "x**2 - 4",
        "x**2 + 2*x + 1",
        "x**3 - x",
        "x**2 - 1",
        "x**3 - 8",
        "x**4 - 1",
        "x**2 - 5*x + 6",
        "x**3 - 6*x**2 + 11*x - 6",
    ]
    for s in factor_cases:
        expr = sp.sympify(s)
        result = sp.factor(expr)
        fixtures.append(
            {
                "category": "algebra",
                "subcategory": "factor",
                "input": s,
                "operation": "factor",
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── algebra/collect (5) ──────────────────────────────────────────────────
    collect_cases = [
        ("x**2 + 2*x*y + y**2", "x"),
        ("x*y + x*y**2 + x*y**3", "x"),
        ("x**2*y + x*y + y", "x"),
        ("x*y + x + y + 1", "x"),
        ("x**2 + x*y + x + y", "x"),
    ]
    for s, var_str in collect_cases:
        expr = sp.sympify(s)
        var = x if var_str == "x" else y
        result = sp.collect(expr, var)
        fixtures.append(
            {
                "category": "algebra",
                "subcategory": "collect",
                "input": s,
                "variable": var_str,
                "operation": "collect",
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── algebra/together (5) ─────────────────────────────────────────────────
    together_cases = [
        "1/x + 1/y",
        "1/x + 1/(x + 1)",
        "x/2 + x/3",
        "1/(x - 1) + 1/(x + 1)",
        "1/x + 1/x**2",
    ]
    for s in together_cases:
        expr = sp.sympify(s)
        result = sp.together(expr)
        fixtures.append(
            {
                "category": "algebra",
                "subcategory": "together",
                "input": s,
                "operation": "together",
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── algebra/cancel (5) ───────────────────────────────────────────────────
    cancel_cases = [
        "(x**2 - 1)/(x - 1)",
        "(x**2 - 4)/(x - 2)",
        "(x**2 + 2*x + 1)/(x + 1)",
        "x**2/x",
        "(x**3 - x)/(x**2 - 1)",
    ]
    for s in cancel_cases:
        expr = sp.sympify(s)
        result = sp.cancel(expr)
        fixtures.append(
            {
                "category": "algebra",
                "subcategory": "cancel",
                "input": s,
                "operation": "cancel",
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    # ── algebra/apart (5) ────────────────────────────────────────────────────
    apart_cases = [
        "1/(x**2 - 1)",
        "1/(x*(x + 1))",
        "(x + 1)/(x*(x - 1))",
        "1/(x**2 - 4)",
        "x/(x**2 - 1)",
    ]
    for s in apart_cases:
        expr = sp.sympify(s)
        result = sp.apart(expr, x)
        fixtures.append(
            {
                "category": "algebra",
                "subcategory": "apart",
                "input": s,
                "variable": "x",
                "operation": "apart",
                "sympy_result": str(result),
                "eval_points": make_eval_points(result),
            }
        )

    return fixtures  # 28


# ═══════════════════════════════════════════════════════════════════════════════
# Category: evalf  (4 fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_evalf():
    fixtures = []

    cases = [
        ("pi", sp.pi),
        ("E", sp.E),
        ("sqrt(2)", sp.sqrt(2)),
        ("log(2)", sp.log(2)),
    ]

    for input_str, expr in cases:
        result = expr.evalf(50)
        fixtures.append(
            {
                "category": "evalf",
                "input": input_str,
                "digits": 50,
                "sympy_result": str(result),
            }
        )

    return fixtures  # 4


# ═══════════════════════════════════════════════════════════════════════════════
# Category: special_func  (3 fixtures)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_special_func():
    fixtures = []

    factorial_cases = [5, 10, 0]

    for n in factorial_cases:
        result = sp.factorial(n)
        value = make_value(result)
        fixtures.append(
            {
                "category": "special_func",
                "subcategory": "factorial",
                "input": str(n),
                "operation": "factorial",
                "sympy_result": str(result),
                "value": value,
            }
        )

    return fixtures  # 3


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    all_fixtures = []

    generators = [
        ("diff", gen_diff, 59),
        ("integrate", gen_integrate, 34),
        ("definite_integral", gen_definite_integral, 15),
        ("simplify", gen_simplify, 21),
        ("expand", gen_expand, 14),
        ("solve", gen_solve, 23),
        ("eval", gen_eval, 15),
        ("series", gen_series, 10),
        ("limit", gen_limit, 15),
        ("matrix", gen_matrix, 22),
        ("algebra", gen_algebra, 28),
        ("evalf", gen_evalf, 4),
        ("special_func", gen_special_func, 3),
    ]

    total_expected = sum(exp for _, _, exp in generators)
    print(
        f"Generating {total_expected} fixtures with SymPy {sp.__version__}...",
        file=sys.stderr,
    )

    errors = []
    for name, gen_fn, expected_count in generators:
        try:
            fixtures = gen_fn()
            actual = len(fixtures)
            status = (
                "OK"
                if actual == expected_count
                else f"MISMATCH (expected {expected_count})"
            )
            print(f"  {name:20s}: {actual:3d}  {status}", file=sys.stderr)
            all_fixtures.extend(fixtures)
        except Exception as e:
            print(f"  {name:20s}: ERROR - {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            errors.append(f"{name}: {e}")

    # Assign sequential IDs
    for i, f in enumerate(all_fixtures, 1):
        f["id"] = i

    output = {
        "generated_by": f"SymPy {sp.__version__}",
        "generated_at": datetime.datetime.now().isoformat(),
        "fixture_count": len(all_fixtures),
        "fixtures": all_fixtures,
    }

    json.dump(output, sys.stdout, indent=2)
    print("", file=sys.stdout)  # trailing newline

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Total fixtures generated: {len(all_fixtures)}", file=sys.stderr)
    if len(all_fixtures) == total_expected:
        print(f"Count matches expected: {total_expected}", file=sys.stderr)
    else:
        print(
            f"COUNT MISMATCH: got {len(all_fixtures)}, expected {total_expected}",
            file=sys.stderr,
        )
    if errors:
        print(f"Errors encountered: {len(errors)}", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
    else:
        print("No errors.", file=sys.stderr)


if __name__ == "__main__":
    main()
