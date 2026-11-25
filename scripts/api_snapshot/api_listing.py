#!/usr/bin/env python3
"""Write the flat, tab-separated listing of the public API that the 1.0 planning
tables are keyed on (`public_api_<version>.tsv`, `api_fate.tsv`).

It is the listing of the earlier survey (`api_walk.py`), reproduced row for row
so that two releases can be compared with `diff`: one row per module, struct,
enum, trait, function, method (inherent, trait declaration, and trait
implementation), constant, type alias and macro, in the order rustdoc lists
them, with the columns

    kind  path  container  trait  signature  file  has_docs  flags

Differences from the snapshot in `api/`, which is the file to review:
  - types are written as in the source (`crate::api::expr::Ex`), not resolved;
  - fields, variants, second paths and `impl` headers have no rows; the methods
    of a trait implementation have one row each instead;
  - `#[doc(hidden)]` items are left out;
  - the signature column repeats every parameter name (`(n: n: i64) -> Q`).  The
    0.59 listing did, the reviewers' tables were joined against it, and a
    corrected column would make every row differ.

Usage: api_listing.py [--json target/doc/symplex.json] --out FILE
Python 3, standard library only.
"""

import argparse
import collections
import json
import os
import sys

sys.dont_write_bytecode = True   # no __pycache__ beside the scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_snapshot import Surface, is_doc_hidden, kind_of  # noqa: E402

COLUMNS = ["kind", "path", "container", "trait", "signature", "file", "has_docs", "flags"]


# The survey's rendering of types: paths as written in the source.
def render_args(args):
    if not args:
        return ""
    if "angle_bracketed" in args:
        ab = args["angle_bracketed"]
        parts = []
        for a in ab["args"]:
            if "type" in a:
                parts.append(render(a["type"]))
            elif "lifetime" in a:
                parts.append(a["lifetime"])
            elif "const" in a:
                parts.append(str(a["const"].get("expr", "_")))
            else:
                parts.append("_")
        for c in ab.get("constraints", []):
            b = c.get("binding", {})
            if "equality" in b:
                eq = b["equality"]
                parts.append(c["name"] + "=" + (render(eq["type"]) if "type" in eq else "_"))
            else:
                parts.append(c["name"] + ": _")
        return "<" + ", ".join(parts) + ">" if parts else ""
    if "parenthesized" in args:
        pa = args["parenthesized"]
        s = "(" + ", ".join(render(t) for t in pa["inputs"]) + ")"
        if pa.get("output"):
            s += " -> " + render(pa["output"])
        return s
    return ""


def render_bound(b):
    if "trait_bound" in b:
        t = b["trait_bound"]["trait"]
        return t["path"] + render_args(t.get("args"))
    if "outlives" in b:
        return b["outlives"]
    return "?"


def render(t):
    if t is None:
        return "()"
    if "resolved_path" in t:
        rp = t["resolved_path"]
        return rp["path"] + render_args(rp.get("args"))
    if "borrowed_ref" in t:
        br = t["borrowed_ref"]
        return "&" + ("mut " if br["is_mutable"] else "") + render(br["type"])
    if "generic" in t:
        return t["generic"]
    if "primitive" in t:
        return t["primitive"]
    if "tuple" in t:
        return "(" + ", ".join(render(x) for x in t["tuple"]) + ")"
    if "slice" in t:
        return "[" + render(t["slice"]) + "]"
    if "array" in t:
        return "[" + render(t["array"]["type"]) + "; " + str(t["array"]["len"]) + "]"
    if "impl_trait" in t:
        return "impl " + " + ".join(render_bound(b) for b in t["impl_trait"])
    if "dyn_trait" in t:
        return "dyn " + " + ".join(tr["trait"]["path"] + render_args(tr["trait"].get("args")) for tr in t["dyn_trait"]["traits"])
    if "qualified_path" in t:
        qp = t["qualified_path"]
        s = render(qp["self_type"])
        if qp.get("trait"):
            return "<" + s + " as " + qp["trait"]["path"] + ">::" + qp["name"]
        return s + "::" + qp["name"]
    if "raw_pointer" in t:
        rp = t["raw_pointer"]
        return "*" + ("mut " if rp["is_mutable"] else "const ") + render(rp["type"])
    if "function_pointer" in t:
        fp = t["function_pointer"]["sig"]
        s = "fn(" + ", ".join(render(x[1]) for x in fp["inputs"]) + ")"
        if fp.get("output"):
            s += " -> " + render(fp["output"])
        return s
    if "infer" in t:
        return "_"
    if "pat" in t:
        return render(t["pat"]["type"])
    return "?" + next(iter(t.keys()))


def best_path(paths):
    return "::".join(sorted(paths, key=lambda p: ("prelude" in p, "__macro_support" in p, len(p), p))[0])


def listing(surface):
    """The rows of the listing, in the survey's order."""
    public = collections.OrderedDict()
    for key, path, hidden in surface.sightings:
        if isinstance(key, tuple) or hidden or is_doc_hidden(surface.item(key)):
            continue
        public.setdefault(key, []).append(list(path))

    rows = []

    def file_of(item):
        return (item.get("span") or {}).get("filename", "")

    def function_row(item, container, container_path, trait):
        sig = item["inner"]["function"]["sig"]
        # `name: name: Type`: see the note at the top of this file.
        inputs = ", ".join("%s: %s: %s" % (name, name, render(ty)) for name, ty in sig["inputs"])
        output = render(sig["output"]) if sig["output"] else "()"
        docs = item.get("docs") or ""
        flags = [flag for flag, on in (("doc:panics", "# Panics" in docs), ("doc:refusal", "# Refusal" in docs)) if on]
        rows.append(["method" if container else "fn",
                     container_path + "::" + item["name"] if container else container_path,
                     container or "", trait or "", "(%s) -> %s" % (inputs, output),
                     file_of(item), str(bool(docs)), ",".join(flags)])

    def plain_row(kind, item, path, flags=""):
        rows.append([kind, path, "", "", "", file_of(item), "", flags])

    for key, paths in public.items():
        item = surface.item(key)
        kind = kind_of(item)
        path = best_path(paths)
        if kind in ("struct", "enum", "union"):
            for impl_id in item["inner"][kind]["impls"]:
                impl_item = surface.item(impl_id)
                if impl_item is None or is_doc_hidden(impl_item):
                    continue
                impl = impl_item["inner"]["impl"]
                if impl.get("is_synthetic") or impl.get("blanket_impl"):
                    continue
                trait = impl.get("trait")
                trait_name = trait["path"] + render_args(trait.get("args")) if trait is not None else None
                for member_id in impl["items"]:
                    member = surface.item(member_id)
                    if member is None or kind_of(member) != "function" or is_doc_hidden(member):
                        continue
                    if trait is not None or member["visibility"] == "public":
                        function_row(member, item["name"], path, trait_name)
            plain_row(kind, item, path, "non_exhaustive" if "non_exhaustive" in item.get("attrs", []) else "")
        elif kind == "function":
            function_row(item, None, path, None)
        elif kind == "trait":
            for member_id in item["inner"]["trait"]["items"]:
                member = surface.item(member_id)
                if member is not None and kind_of(member) == "function" and not is_doc_hidden(member):
                    function_row(member, item["name"], path, "(decl)")
            plain_row(kind, item, path)
        elif kind in ("constant", "type_alias", "macro", "module", "static"):
            plain_row(kind, item, path)
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    parser = argparse.ArgumentParser(description="Write the flat listing of the public API (tab-separated).")
    parser.add_argument("--json", default=os.path.join(root, "target", "doc", "symplex.json"),
                        help="rustdoc JSON of the library (default: target/doc/symplex.json)")
    parser.add_argument("--out", required=True, help="file to write")
    args = parser.parse_args()
    with open(args.json, encoding="utf-8") as handle:
        surface = Surface(json.load(handle))
    rows = listing(surface)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        for row in [COLUMNS] + rows:
            handle.write("\t".join(cell.replace("\t", " ").replace("\n", " ") for cell in row) + "\n")
    counts = collections.Counter(row[0] for row in rows)
    print("%d rows (%s)" % (len(rows), ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
