#!/usr/bin/env python3
"""Review A's prelude count (417 imports in 247 test files), re-derived with A's own scan
(retired_paths.py) and matched against the tool's `paths.prelude-names` sites in tests/.

    python3 tools/compare_A_prelude.py out/paths.sites.tsv
"""
import collections
import csv
import os
import re
import sys

REPO = os.environ.get("SYMPLEX_REPO", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
NEW = "Context Expr Ex BoolEx SetEx Symbol Equation Truth SymplexError Assumption Matrix expr! eq! matrix! syms!".split()
OLD = """Context Equation AnyExpr BoolEx Boolean Ex Expr ExprType Numeric SetEx SetValued SimplifyOpts Sort
Scalar ToEx Bindings RewriteOpts RewriteStrategy Rule RuleSet Step reduce_inequalities InitialCondition
GeneralSolution LinearSolution NewtonOpts ZeroForm linsolve linsolve_matrix solve_numeric_system
solve_numeric_system_with Direction FourierConvention FourierSeries ExprView Poly Assumption Assumptions Props
EvalConfig SymplexError Extended Bounds Interval IntervalKind QuadOpts QuadResult FormalPowerSeries StateSpace
TransferFunction Diagonalization HermiteNormalForm Hessenberg JordanForm Ldl LllReduction Lu Qr RankDecomposition
SmithNormalForm Feasibility LpProblem LpSolution LpStatus Q Matrix QMatrix ZMatrix MinimizeOpts MinimizeResult
RootOpts Quaternion EulerConvention CoordinateSystem CodegenOptions MathBackend Precision CompiledFn CompiledFnVec
MonomialOrder MultiPoly ExpandOpts Complex64 dim! eq! expr! matrix! rule!""".split()
dropped = [n for n in OLD if n not in NEW]

mine = set()
for r in csv.DictReader(open(sys.argv[1]), delimiter="\t", quoting=csv.QUOTE_NONE):
    if r["rule"] == "paths.prelude-names" and r["corpus"] == "tests":
        mine.add((r["file"], r["name"]))
a = set()
cat = collections.Counter()
files = set()
other = []
for dp, _dn, fn in os.walk(os.path.join(REPO, "tests")):
    for f in fn:
        if not f.endswith(".rs"):
            continue
        p = os.path.join(dp, f)
        t = open(p, errors="replace").read()
        if not re.search(r"symplex::prelude::\*", t):
            continue
        rel = os.path.relpath(p, REPO)
        uses = "\n".join(ln for ln in t.splitlines() if ln.lstrip().startswith("use "))
        body = "\n".join(ln for ln in t.splitlines() if not ln.lstrip().startswith("use ") and not ln.lstrip().startswith("//"))
        for n in dropped:
            bare = n.rstrip("!")
            rx = r"(?<![\w:])%s\s*[\(\[\{]" % re.escape(n) if n.endswith("!") else r"(?<![\w:.])%s\b(?!\s*:[^:])" % re.escape(n)
            if not re.search(rx, body):
                continue
            a.add((rel, bare))
            files.add(rel)
            if (rel, bare) in mine:
                cat["the tool adds the import too"] += 1
            elif n == "Q":
                cat["`Q`: renamed by rule set numbers, no import"] += 1
            elif re.search(r"\b%s\b" % re.escape(bare), uses):
                cat["the file already imports the name in a `use` line"] += 1
            else:
                cat["other (strings, a variable of that name, an item the final table removes)"] += 1
                other.append((rel, bare))
print("A's scan today: %d (file, name) pairs in %d test files" % (len(a), len(files)))
for k, v in cat.most_common():
    print("   %4d  %s" % (v, k))
print("the tool: %d imports in %d test files; %d of them A's scan does not have" % (
    len(mine), len(set(f for f, _n in mine)), len(mine - a)))
print("names of the tool-only pairs:", dict(collections.Counter(n for _f, n in mine - a).most_common(12)))
print("names of A's `other`:", dict(collections.Counter(n for _f, n in other).most_common(14)))
