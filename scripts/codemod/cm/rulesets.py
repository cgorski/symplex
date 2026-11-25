# The rule sets, one per future work package.
import collections

from . import rules_truth, rules_errors, rules_numbers, rules_renames, rules_paths, rules_domains

RULESETS = collections.OrderedDict()
for _rs in (rules_truth.RULESET, rules_errors.RULESET, rules_numbers.RULESET, rules_renames.CORE, rules_renames.SUBS,
            rules_paths.RULESET, rules_domains.RULESET):
    RULESETS[_rs.name] = _rs
