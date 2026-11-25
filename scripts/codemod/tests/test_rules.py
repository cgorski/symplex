"""Every rule of every rule set against its cases (tests/cases_*.py).

A case gives an input and either the expected output, or the reason with which the site
must be listed instead of rewritten, or says that the rule must not see a site at all.
Every case is also run a second time on its own output: the tool must then change nothing.
"""
import glob
import importlib
import os
import unittest

import support
from support import run, ruleset_of, all_rule_ids
from cm.engine import REWRITE, MANUAL, INFO


def load_cases():
    cases = []
    here = os.path.dirname(os.path.abspath(__file__))
    for path in sorted(glob.glob(os.path.join(here, "cases_*.py"))):
        mod = importlib.import_module(os.path.basename(path)[:-3])
        cases.extend(mod.CASES)
    return cases


CASES = load_cases()


class RuleCases(unittest.TestCase):
    maxDiff = None

    def test_cases(self):
        for n, c in enumerate(CASES):
            with self.subTest(rule=c.rule, n=n, src=c.src[-90:]):
                rs = ruleset_of(c.rule)
                text, rewritten, manual, notes = run(rs, c.src, c.kind, c.corpus, c.opts, c.src_code)
                mine_rw = [s for s in rewritten if s.rule == c.rule]
                mine_man = [s for s in manual if s.rule == c.rule and s.status == MANUAL]
                mine_info = [s for s in manual if s.rule == c.rule and s.status == INFO]
                if c.want is None:
                    self.assertEqual(text, c.src, "the tool changed a text it must leave alone")
                else:
                    self.assertEqual(text, c.want)
                    if c.manual is None and c.info is None:
                        self.assertTrue(mine_rw, "the output is right but no site of %s was counted" % c.rule)
                if c.manual is not None:
                    self.assertTrue(any(c.manual in s.reason for s in mine_man),
                                    "no site listed with %r; listed: %s" % (c.manual, [s.reason for s in mine_man]))
                if c.info is not None:
                    self.assertTrue(any(c.info in s.reason for s in mine_info),
                                    "no unchanged-site with %r; got: %s" % (c.info, [s.reason for s in mine_info]))
                if c.silent:
                    self.assertEqual([(s.status, s.reason) for s in mine_rw + mine_man + mine_info], [])
                # idempotence: the output is a fixpoint
                again, rw2, _man2, _n2 = run(rs, text, c.kind, c.corpus, c.opts, c.src_code)
                self.assertEqual(again, text, "a second run changed the output of the first")
                self.assertEqual(rw2, [])

    def test_every_rule_has_a_case(self):
        covered = set(c.rule for c in CASES if c.want is not None or c.manual is not None or c.info is not None)
        missing = [r for r in all_rule_ids() if r not in covered]
        self.assertEqual(missing, [], "rules without a unit test")

    def test_every_rule_is_described(self):
        for rs in support.RULESETS.values():
            self.assertTrue(rs.needs and rs.package, rs.name)
            for r in rs.rules:
                self.assertTrue(r.id.count(".") >= 1 and len(r.description) > 20, r.id)
        ids = all_rule_ids()
        self.assertEqual(len(ids), len(set(ids)), "rule ids are unique")


if __name__ == "__main__":
    unittest.main()
