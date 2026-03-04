#!/usr/bin/env python3
"""Generate SymPy cross-validation fixtures for symplex.

Usage:
    python3 scripts/generate_sympy_fixtures.py > tests/fixtures/sympy_cross_validation.json

Requires: pip install sympy
"""

import datetime
import json

import sympy as sp

x, y = sp.symbols("x y")

fixtures = []


def eval_at(expr, point_dict):
    """Evaluate numerically, return (re, im) or None."""
    try:
        val = complex(expr.subs(point_dict).evalf())
        return {"re": val.real, "im": val.imag}
    except:
        return None


# Standard evaluation points
pts_x = [
    {x: sp.Rational(3, 2)},
    {x: sp.Rational(7, 10)},
    {x: sp.Rational(2, 1)},
]
pts_xy = [
    {x: sp.Rational(3, 2), y: sp.Rational(1, 2)},
]

# === DIFF ===
diff_cases = [
    "x**3 + 2*x + 1",
    "sin(x)",
    "cos(x)**2",
    "exp(x)*sin(x)",
    "log(x**2 + 1)",
    "atan(x)",
    "sinh(x)",
    "x**2*exp(x)",
]
for expr_str in diff_cases:
    expr = sp.sympify(expr_str)
    result = sp.diff(expr, x)
    evals = [{"x": float(p[x]), "value": eval_at(result, p)} for p in pts_x]
    fixtures.append(
        {
            "category": "diff",
            "input": expr_str,
            "sympy_result": str(result),
            "eval_points": evals,
        }
    )

# === INTEGRATE ===
int_cases = [
    "x**2",
    "sin(x)",
    "cos(x)",
    "exp(x)",
    "x*exp(x)",
    "x*sin(x)",
    "x*log(x)",
    "1/(x**2 + 1)",
]
for expr_str in int_cases:
    expr = sp.sympify(expr_str)
    result = sp.integrate(expr, x)
    evals = [{"x": float(p[x]), "value": eval_at(result, p)} for p in pts_x]
    fixtures.append(
        {
            "category": "integrate",
            "input": expr_str,
            "sympy_result": str(result),
            "eval_points": evals,
        }
    )

# === SIMPLIFY ===
simp_cases = [
    ("sin(x)**2 + cos(x)**2", sp.simplify(sp.sin(x) ** 2 + sp.cos(x) ** 2)),
    ("exp(log(x))", sp.simplify(sp.exp(sp.log(x)))),
    ("(x**2 - 1)/(x - 1)", sp.simplify((x**2 - 1) / (x - 1))),
    ("sin(x)/cos(x)", sp.simplify(sp.sin(x) / sp.cos(x))),
]
for expr_str, result in simp_cases:
    evals = [{"x": float(p[x]), "value": eval_at(result, p)} for p in pts_x]
    fixtures.append(
        {
            "category": "simplify",
            "input": expr_str,
            "sympy_result": str(result),
            "eval_points": evals,
        }
    )

# === EXPAND ===
expand_cases = [
    ("(x + 1)**2", sp.expand((x + 1) ** 2)),
    ("(x + 1)**3", sp.expand((x + 1) ** 3)),
    ("sin(2*x)", sp.expand_trig(sp.sin(2 * x))),
    ("cos(2*x)", sp.expand_trig(sp.cos(2 * x))),
]
for expr_str, result in expand_cases:
    evals = [{"x": float(p[x]), "value": eval_at(result, p)} for p in pts_x]
    fixtures.append(
        {
            "category": "expand",
            "input": expr_str,
            "sympy_result": str(result),
            "eval_points": evals,
        }
    )

# === SOLVE ===
solve_cases = [
    "x**2 - 4",
    "x**2 - 5*x + 6",
    "x**2 + 1",
]
for expr_str in solve_cases:
    expr = sp.sympify(expr_str)
    roots = sp.solve(expr, x)
    fixtures.append(
        {
            "category": "solve",
            "input": expr_str,
            "sympy_roots": [
                {
                    "symbolic": str(r),
                    "re": float(complex(r.evalf()).real),
                    "im": float(complex(r.evalf()).imag),
                }
                for r in roots
            ],
        }
    )

# === EVAL (special values) ===
eval_cases = [
    ("sin(pi/6)", sp.sin(sp.pi / 6)),
    ("sin(pi/4)", sp.sin(sp.pi / 4)),
    ("cos(pi/3)", sp.cos(sp.pi / 3)),
    ("tan(pi/4)", sp.tan(sp.pi / 4)),
    ("tan(2*pi/3)", sp.tan(2 * sp.pi / 3)),
    ("exp(0)", sp.exp(0)),
    ("log(1)", sp.log(1)),
]
for expr_str, result in eval_cases:
    val = complex(result.evalf())
    fixtures.append(
        {
            "category": "eval",
            "input": expr_str,
            "sympy_result": str(result),
            "value": {"re": val.real, "im": val.imag},
        }
    )

# === SERIES ===
series_cases = [
    ("sin(x)", sp.series(sp.sin(x), x, 0, 6).removeO()),
    ("cos(x)", sp.series(sp.cos(x), x, 0, 6).removeO()),
    ("exp(x)", sp.series(sp.exp(x), x, 0, 6).removeO()),
]
for expr_str, result in series_cases:
    evals = [{"x": 0.25, "value": eval_at(result, {x: sp.Rational(1, 4)})}]
    fixtures.append(
        {
            "category": "series",
            "input": expr_str,
            "sympy_result": str(result),
            "eval_points": evals,
        }
    )

output = {
    "generated_by": f"SymPy {sp.__version__}",
    "generated_at": datetime.datetime.now().isoformat(),
    "description": "Cross-validation fixtures: symplex results should match these numerical evaluations",
    "fixtures": fixtures,
}

print(json.dumps(output, indent=2))
