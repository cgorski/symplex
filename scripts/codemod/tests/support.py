# Shared helpers of the unit tests.  Run the whole suite from the tool's directory:
#
#     python3 -m unittest discover -s tests -v
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cm import engine  # noqa: E402
from cm.rulesets import RULESETS  # noqa: E402

# One case of a rule.
#   rule     the rule id the case exercises
#   src      the input text
#   want     the expected output; None = the tool must leave the text exactly as it is
#   manual   None, or a substring of the reason with which the site must be LISTED (not rewritten)
#   kind     "rs" (a Rust file) or "md" (a Markdown file)
#   corpus   which root the file pretends to live in ("tests", "examples", "src", "book", ..)
#   opts     rule options
#   src_code True = also rewrite library code (the --src-code switch)
#   silent   True = the rule must not even list a site (the shape is none of its business)
#   info     None, or a substring of the reason of a site that is counted but needs no edit
Case = collections.namedtuple("Case", "rule src want manual kind corpus opts src_code silent info")


def case(rule, src, want=None, manual=None, kind="rs", corpus="tests", opts=None, src_code=False, silent=False,
         info=None):
    return Case(rule, src, want, manual, kind, corpus, opts or {}, src_code, silent, info)


def ruleset_of(rule_id):
    for rs in RULESETS.values():
        for r in rs.rules:
            if r.id == rule_id:
                return rs
    raise KeyError(rule_id)


def run(ruleset, src, kind="rs", corpus="tests", opts=None, src_code=False, rules=None):
    """Transform `src` with a rule set (a name or a RuleSet).  Returns (text, rewritten, manual, notes)."""
    rs = RULESETS[ruleset] if isinstance(ruleset, str) else ruleset
    if rules:
        rs = engine.RuleSet(rs.name, rs.description, [r for r in rs.rules if r.id in rules], rs.needs, rs.package)
    name = "case.%s" % kind
    rel = {"src": "src/case.rs", "book": "book/src/case.md"}.get(corpus, "%s/%s" % (corpus, name))
    cx = engine.Cx(dict(opts or {}))
    return engine.transform(src, "/nowhere/" + rel, rel, corpus, rs, cx, src_code=src_code)


def all_rule_ids():
    return [r.id for rs in RULESETS.values() for r in rs.rules]
