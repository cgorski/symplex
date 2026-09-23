# Rubi integration test suite (Maxima syntax), vendored

This directory is a verbatim copy of the Rubi (Rule-based Integration) test
suite in Maxima syntax:

- Source: <https://github.com/RuleBasedIntegration/MaximaSyntaxTestSuite>
- Commit: `60295e21c571ca210ecfbb695f4af99947454adf`
- Licence: MIT, "Copyright (c) 2018 Rule-based Integration" — see
  [`LICENSE`](LICENSE) in this directory, copied unchanged from upstream.

The 215 `.mac` files (in the chapter folders `0 Independent test suites` …
`8 Special functions`) and `LICENSE` are **unmodified**.  The only file not
from upstream is this `README.md`, which replaces the upstream one (a single
heading line, `# MaximaSyntaxTestSuite`).

Each file holds a Maxima list `lst: '[ ... ]` of entries
`[integrand, variable, rubi_step_count, optimal_antiderivative]` (a few
entries carry a fifth field, an alternative antiderivative).  The harness in
the parent directory (`rubi-harness/`) reads these files; see its
`README.md`.
