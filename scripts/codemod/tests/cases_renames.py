# Cases of the table-driven renames: the rule sets `core-renames`, `subs`, and the rename rows of
# `domains`.  One case (at least) per row of tables/core_renames.tsv and tables/domain_renames.tsv.
#
# The expected output of a row is NOT taken from the rule: each shape has a template below that
# says, in Rust text, what the shape means; the table only supplies the names.  Hand-written
# cases for the shapes' edges follow the generated ones.
import csv
import os

from support import case

TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
P = "use symplex::prelude::*;\n"


def rows(name):
    with open(os.path.join(TABLES, name + ".tsv"), newline="") as f:
        lines = [ln for ln in f if not ln.startswith("#") and ln.strip()]
    return list(csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE))


def f(body, head=P):
    return head + "fn t() {\n" + body + "}\n"


def generated(row):
    rid, shape, old, new, extra = row["id"], row["shape"], row["old"], row["new"], row["extra"]
    out = []
    if shape == "method" or shape.startswith("method/"):
        n = int(shape.split("/")[1]) if "/" in shape else 2
        args = ", ".join(["&a", "b.clone()", "3"][:n])
        out.append(case(rid, f("    let y = e.%s(%s);\n    let z = w.inner().%s(%s)?;\n" % (old, args, old, args)),
                        f("    let y = e.%s(%s);\n    let z = w.inner().%s(%s)?;\n" % (new, args, new, args))))
        # not a method call of that name: a string, a comment, a field, a free function
        out.append(case(rid, f("    let s = \".%s(\"; // e.%s(x)\n    let v = cfg.%s;\n" % (old, old, old)), silent=True))
        if "/" in shape:
            other = ", ".join(["&a", "b", "c"][:n + 1])
            out.append(case(rid, f("    let y = e.%s(%s);\n" % (old, other)), silent=True))
        if row["units"] == "keep":
            out.append(case(rid, "use symplex::units::Length;\n" + f("    let y = len.%s(%s);\n" % (old, args), ""),
                            manual="unit quantities"))
    elif shape.startswith("path:"):
        ty = shape[5:]
        out.append(case(rid, f("    let r = %s::%s(\"n\", &a, b);\n" % (ty, old)),
                        f("    let r = %s::%s(\"n\", &a, b);\n" % (ty, new))))
        out.append(case(rid, f("    let r = Other::%s(1);\n    let q = x.%s(2);\n" % (old, old)), silent=True))
    elif shape.startswith("in:"):
        seg = shape[3:].split("|")[0]
        out.append(case(rid, f("    let q = %s::%s(0.975)?;\n    let r = numdist::%s::%s(0.5);\n" % (seg, old, seg, old)),
                        f("    let q = %s::%s(0.975)?;\n    let r = numdist::%s::%s(0.5);\n" % (seg, new, seg, new))))
        out.append(case(rid, f("    let q = scipy::%s(0.975);\n" % old), silent=True))
    elif shape == "flag":
        out.append(case(rid, f("    let a = e.%s(true);\n    let b = e.%s(false).simplify();\n" % (old, old)),
                        f("    let a = e.%s();\n    let b = e.%s().simplify();\n" % (extra, new))))
        out.append(case(rid, f("    let a = e.%s(force);\n" % old), manual="not a literal"))
    elif shape in ("recv-first", "ctx-first"):
        out.append(case(rid, f("    let s = %s::%s(&body, &k, &lo, &hi);\n    let t = %s::%s(&(a + b), &k);\n"
                               % (extra, old, extra, old)),
                        f("    let s = body.%s(&k, &lo, &hi);\n    let t = (a + b).%s(&k);\n" % (new, new))))
        out.append(case(rid, f("    let s = Other::%s(&body, &k);\n" % old), silent=True))
    elif shape == "append-arg":
        text, _, imp = extra.partition("|")
        head = P + "use %s;\n" % imp if imp else P
        out.append(case(rid, f("    let y = e.%s(&rules);\n" % old),
                        f("    let y = e.%s(&rules, %s);\n" % (new, text), head)))
    elif shape == "ineq":
        out.append(case(rid, f("    let s = (&x - 1).%s(&x);\n" % old),
                        f("    let s = (&x - 1).%s(0).%s(&x);\n" % (extra, new))))
    elif shape == "to-field":
        out.append(case(rid, f("    let p = t_test(&a, &b).unwrap().%s();\n    let q = res.%s.%s();\n" % (old, extra, old)),
                        f("    let p = t_test(&a, &b).unwrap().%s.%s();\n    let q = res.%s.%s();\n"
                          % (extra, new, extra, new))))
    elif shape == "free":
        out.append(case(rid, "use symplex::ntheory::{%s, is_prime};\n" % old + f("    let n = %s(7);\n" % old, ""),
                        "use symplex::ntheory::{%s, is_prime};\n" % new + f("    let n = %s(7);\n" % new, "")))
        out.append(case(rid, "fn %s(n: u64) -> u64 { n }\n" % old + f("    let n = %s(7);\n" % old, ""),
                        manual="defines a function named"))
    elif shape == "type":
        out.append(case(rid, f("    let o: %s = %s::default();\n    let v: Vec<%s> = vec![];\n" % (old, old, old)),
                        f("    let o: %s = %s::default();\n    let v: Vec<%s> = vec![];\n" % (new, new, new))))
        out.append(case(rid, "struct %s;\n" % old + f("    let o = %s;\n" % old, ""), manual="defines its own"))
    elif shape == "manual":
        out.append(case(rid, f("    let y = e.%s(&x);\n" % old), manual=""))
    elif shape == "manual-free":
        out.append(case(rid, f("    let y = %s(&a, &b);\n" % old), manual=""))
    elif shape == "manual-type":
        out.append(case(rid, f("    let y: %s = make();\n" % old), manual=""))
    else:
        raise AssertionError("no template for shape %r (row %s)" % (shape, rid))
    return out


CASES = []
for _row in rows("core_renames") + rows("domain_renames"):
    CASES.extend(generated(_row))

CASES += [
    # ---- edges, written by hand
    # the receiver expression of `Ex::symbolic_sum` keeps its own calls, which other rules rewrite in a later pass
    case("core.symbolic-sum",
         f("    let s = Ex::symbolic_sum(\n        &term.piecewise_simplify(),\n        &k,\n        &lo,\n        &hi,\n    );\n"),
         f("    let s = term.simplify_piecewise().formal_summation(\n        &k,\n        &lo,\n        &hi,\n    );\n")),
    case("core.sum-of",
         f("    let s = Ex::sum_of(&ctx, terms.iter());\n    let p = Ex::product_of(&ctx, [a, b]);\n"),
         f("    let s = ctx.sum(terms.iter());\n    let p = ctx.product([a, b]);\n")),
    # `factor(var)` and `factor_all()` swap names in ONE run; a second run must not chain them
    case("domains.factor-in",
         f("    let a = e.factor(&x);\n    let b = e.factor_all();\n"),
         f("    let a = e.factor_in(&x);\n    let b = e.factor();\n")),
    case("domains.factor-all", f("    let b = e.factor();\n"), silent=True),
    # expand_power_base keeps its name: the flag-less call is the migrated form
    case("core.expand-power-base", f("    let a = e.expand_power_base();\n"), silent=True),
    # the receiver of a p-value method that IS a p-value: listed, because `.p_value.` would be wrong
    case("domains.p-value-f64",
         f("    for p in &fit.p_values {\n        let v = p.p_value_f64().unwrap();\n    }\n"), manual="is bound from"),
    case("domains.p-value-f64",
         f("    let v = fit.p_values[0].p_value_f64().unwrap();\n"), manual="may be a `p_value` value itself"),
    # density on a user's own Family implementation keeps its name
    case("domains.density",
         P + "impl Family for Mine {\n    fn density(&self, x: &Ex) -> Ex { x.clone() }\n}\n" + f("    let d = m.density(&x);\n", ""),
         manual="implements `Family`"),
    # a definition is not a call
    case("subs.i64", "trait T {\n    fn subs_i64(&self, x: &Ex, n: i64) -> Ex;\n}\n", silent=True),
    # the README's one-liner
    case("subs.i64",
         "```rust\nlet y = f.subs_i64(&x, 3);            // 28\n```\n",
         "```rust\nlet y = f.subs(&x, 3);            // 28\n```\n", kind="md", corpus="readme"),
    # a doctest in the library's doc comments
    case("subs.i64",
         "/// ```\n/// # use symplex::prelude::*;\n/// let y = f.subs_i64(&x, 3);\n/// ```\npub fn g() { let internal = h.subs_i64(&x, 1); }\n",
         "/// ```\n/// # use symplex::prelude::*;\n/// let y = f.subs(&x, 3);\n/// ```\npub fn g() { let internal = h.subs_i64(&x, 1); }\n",
         corpus="src"),
]

CASES += [
    # a loop over results is not a loop over p-values
    case("domains.p-value-f64",
         f("    for r in &results {\n        assert!(r.p_value_f64().unwrap() < 0.05);\n    }\n"),
         f("    for r in &results {\n        assert!(r.p_value.to_f64().unwrap() < 0.05);\n    }\n")),
    case("domains.p-value-f64",
         f("    let ps: Vec<f64> = fit.p_values.iter().map(|p| p.p_value_f64().unwrap()).collect();\n"),
         manual="is bound from"),
    case("domains.p-value-f64",
         f("    let p = &res.p_value;\n    let v = p.p_value_f64().unwrap();\n"), manual="is bound from"),
]
