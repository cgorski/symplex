//! A small structured model of a Lean 4 tactic proof — `have`s, bullets,
//! `by` blocks and raw tactics — that renders itself with correct,
//! indentation-sensitive layout.
//!
//! Lean's tactic blocks are whitespace-sensitive in two ways that string
//! concatenation gets wrong sooner or later: the tactics of a block must
//! share a column, and a `by` block's tactics must sit strictly right of
//! the tactic that opened it (a `· ` bullet moves that column by two).  A
//! [`Block`] knows its structure, so [`Block::render`] places every line
//! from the *tactic column* and wraps long lines past it; the text it
//! produces is what a person indenting by hand would write, and the
//! certificate emitters use the same renderer for their proof steps.
//!
//! ```
//! use symplex::lean::{Block, Proof, Tactic};
//!
//! let proof = Block::new(vec![
//!     Tactic::raw("rcases le_or_gt (0 : ℝ) x with hx | hx"),
//!     Tactic::bullet(Block::new(vec![
//!         Tactic::have("e", Some("(0 : ℝ) ≤ x"), Proof::term("hx")),
//!         Tactic::raw("linarith only [e]"),
//!     ])),
//!     Tactic::bullet(Block::new(vec![
//!         Tactic::have("e", Some("(0 : ℝ) ≤ -x"), Proof::by(Block::new(vec![
//!             Tactic::raw("linarith only [hx]"),
//!         ]))),
//!         Tactic::raw("nlinarith [e, sq_nonneg x]"),
//!     ])),
//! ]);
//! assert_eq!(
//!     proof.render("  "),
//!     "  rcases le_or_gt (0 : ℝ) x with hx | hx\n\
//!      \x20 · have e : (0 : ℝ) ≤ x := hx\n\
//!      \x20   linarith only [e]\n\
//!      \x20 · have e : (0 : ℝ) ≤ -x := by\n\
//!      \x20     linarith only [hx]\n\
//!      \x20   nlinarith [e, sq_nonneg x]\n"
//! );
//! ```
//!
//! An application with many arguments — `refine lemma a b ?_ ?_ …` — is a
//! [`Tactic::apply`]: its arguments are atoms that are packed greedily onto
//! the head's line and continuation lines two columns past the tactic
//! column, never split inside an argument.
//!
//! ```
//! use symplex::lean::{Block, Tactic};
//!
//! let args: Vec<String> = ["(20 * (j : ℝ) + 10)", "ρ", "hx"]
//!     .into_iter()
//!     .map(String::from)
//!     .chain(std::iter::repeat_n("?_".to_string(), 6))
//!     .collect();
//! let leaf = Block::new(vec![Tactic::apply("refine leaf_lemma", args)]);
//! assert_eq!(
//!     leaf.render_width("  ", 42),
//!     "  refine leaf_lemma (20 * (j : ℝ) + 10) ρ\n\
//!      \x20   hx ?_ ?_ ?_ ?_ ?_ ?_\n"
//! );
//! ```

use std::fmt;

use super::lean::{MATHLIB_LINE_WIDTH, lean_ident, wrap_lean};

/// A sequence of tactics at one column.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Block {
    /// The tactics, in order.
    pub tactics: Vec<Tactic>,
}

/// One tactic of a [`Block`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tactic {
    /// `have name : ty := proof` (or `have name := proof` without a type).
    Have {
        /// The hypothesis name (rendered through [`lean_ident`]).
        name: String,
        /// The stated type, if any.
        ty: Option<String>,
        /// The proof: a term on the same line, or a `by` block below.
        proof: Proof,
    },
    /// `· block` — a focused goal; the block's tactics sit two columns
    /// right of the bullet.
    Bullet(Block),
    /// `head arg₁ arg₂ …` — an application whose arguments are atoms
    /// (`?_`, a name, a `(…)`-wrapped term).  The renderer packs the
    /// arguments greedily onto the head's line and onto continuation lines
    /// indented two columns past the tactic column, and never breaks
    /// inside an argument.
    Apply {
        /// The applied tactic and function, e.g. `refine leaf_lemma`.
        head: String,
        /// The arguments, each already parenthesised where needed.
        args: Vec<String>,
    },
    /// Any other tactic, verbatim (`linarith only [a, b]`, `rcases … with
    /// h | h`, `refine f ?_ ?_`, `exact h`, `positivity`).  May span
    /// several lines: the first is placed at the tactic column, each later
    /// line two columns past it *plus* the line's own leading whitespace (so
    /// a continuation indented by two in the text lands four columns past
    /// the tactic column).  An empty string renders as `skip`.
    Raw(String),
}

/// The right-hand side of a `have`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Proof {
    /// A term, rendered after `:=` on the `have` line.
    Term(String),
    /// A tactic block, rendered on the lines below `:= by`, two columns in
    /// (`skip` when the block is empty, so the `by` still parses).
    By(Block),
}

/// One rendered line (indentation included, no trailing newline).
struct Line {
    text: String,
    /// Laid out to the width already ([`Tactic::Apply`] packs its atoms
    /// itself); [`wrap_lean`] must not re-split it inside an argument.
    verbatim: bool,
}

impl Line {
    fn wrap(text: String) -> Self {
        Line {
            text,
            verbatim: false,
        }
    }

    fn verbatim(text: String) -> Self {
        Line {
            text,
            verbatim: true,
        }
    }
}

/// The `skip` placeholder for an empty block, so `by` / `·` still parse.
fn skip_line(indent: &str) -> Line {
    Line::wrap(format!("{indent}skip"))
}

impl Proof {
    /// A term proof.
    pub fn term(text: impl Into<String>) -> Self {
        Proof::Term(text.into())
    }

    /// A `by` block.
    pub fn by(block: Block) -> Self {
        Proof::By(block)
    }
}

impl Tactic {
    /// `have name : ty := proof`.
    pub fn have(name: impl Into<String>, ty: Option<&str>, proof: Proof) -> Self {
        Tactic::Have {
            name: name.into(),
            ty: ty.map(str::to_string),
            proof,
        }
    }

    /// `· block`.
    pub fn bullet(block: Block) -> Self {
        Tactic::Bullet(block)
    }

    /// `head arg₁ arg₂ …` with the arguments laid out by the renderer
    /// (see [`Tactic::Apply`]).
    pub fn apply(head: impl Into<String>, args: Vec<String>) -> Self {
        Tactic::Apply {
            head: head.into(),
            args,
        }
    }

    /// A verbatim tactic.
    pub fn raw(text: impl Into<String>) -> Self {
        Tactic::Raw(text.into())
    }

    /// Hypothesis names this tactic introduces (`have` names, recursively
    /// through bullets and `by` blocks).
    pub fn introduced_names(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_names(&mut out);
        out
    }

    fn collect_names(&self, out: &mut Vec<String>) {
        match self {
            Tactic::Have { name, proof, .. } => {
                out.push(name.clone());
                if let Proof::By(b) = proof {
                    for t in &b.tactics {
                        t.collect_names(out);
                    }
                }
            }
            Tactic::Bullet(b) => {
                for t in &b.tactics {
                    t.collect_names(out);
                }
            }
            Tactic::Apply { .. } | Tactic::Raw(_) => {}
        }
    }
}

impl Block {
    /// A block from its tactics.
    pub fn new(tactics: Vec<Tactic>) -> Self {
        Block { tactics }
    }

    /// Append a tactic.
    pub fn push(&mut self, tactic: Tactic) -> &mut Self {
        self.tactics.push(tactic);
        self
    }

    /// `true` if the block has no tactics.
    pub fn is_empty(&self) -> bool {
        self.tactics.is_empty()
    }

    /// The proof text with every tactic of this block at `indent`, nested
    /// blocks two columns further in, and long lines wrapped to Mathlib's
    /// width past their tactic column.  Ends with a newline (empty for an
    /// empty block).
    pub fn render(&self, indent: &str) -> String {
        self.render_width(indent, MATHLIB_LINE_WIDTH)
    }

    /// [`render`](Self::render) with an explicit line width.
    pub fn render_width(&self, indent: &str, width: usize) -> String {
        let mut lines = Vec::new();
        self.write_lines(indent, width, &mut lines);
        let mut out = String::new();
        for line in lines {
            if line.verbatim {
                out.push_str(&line.text);
            } else {
                out.push_str(&wrap_lean(&line.text, width));
            }
            out.push('\n');
        }
        out
    }

    /// The logical lines, each prefixed by its indentation.  Only
    /// [`Tactic::Apply`] is laid out to `width` here (its arguments are
    /// atoms the generic wrapper must not split, so its lines are marked
    /// verbatim); everything else is wrapped afterwards by [`wrap_lean`].
    fn write_lines(&self, indent: &str, width: usize, out: &mut Vec<Line>) {
        for t in &self.tactics {
            t.write_lines(indent, width, out);
        }
    }
}

impl Tactic {
    fn write_lines(&self, indent: &str, width: usize, out: &mut Vec<Line>) {
        match self {
            Tactic::Have { name, ty, proof } => {
                let head = match ty {
                    Some(t) => format!("have {} : {t}", lean_ident(name)),
                    None => format!("have {}", lean_ident(name)),
                };
                match proof {
                    Proof::Term(term) => {
                        out.push(Line::wrap(format!("{indent}{head} := {term}")));
                    }
                    Proof::By(block) => {
                        out.push(Line::wrap(format!("{indent}{head} := by")));
                        let inner = format!("{indent}  ");
                        let before = out.len();
                        block.write_lines(&inner, width, out);
                        if out.len() == before {
                            // `have h : T := by` with nothing below is a parse
                            // error; `skip` keeps the block well-formed.
                            out.push(skip_line(&inner));
                        }
                    }
                }
            }
            Tactic::Bullet(block) => {
                // The first tactic shares the bullet's line; the rest of the
                // block is at the bullet's column + 2.
                let inner = format!("{indent}  ");
                let mut body = Vec::new();
                block.write_lines(&inner, width, &mut body);
                let Some(first) = body.first_mut() else {
                    out.push(Line::wrap(format!("{indent}· skip")));
                    return;
                };
                // Replace the first line's leading `inner` with `indent· `
                // (every line written at `inner` starts with it).
                if let Some(rest) = first.text.strip_prefix(inner.as_str()) {
                    first.text = format!("{indent}· {rest}");
                }
                out.append(&mut body);
            }
            Tactic::Apply { head, args } => {
                // Greedy packing: an argument goes on the current line if it
                // fits, otherwise it opens a continuation line two columns
                // past the tactic column (the indentation given here).  A
                // continuation line is opened *with* its first argument, so
                // an atom wider than the line is emitted whole, never split
                // — which is why these lines bypass `wrap_lean`.
                let cont = format!("{indent}  ");
                let cont_len = cont.chars().count();
                let mut line = format!("{indent}{head}");
                let mut used = line.chars().count();
                for arg in args {
                    let len = arg.chars().count();
                    if used + 1 + len > width {
                        out.push(Line::verbatim(line));
                        line = format!("{cont}{arg}");
                        used = cont_len + len;
                    } else {
                        line.push(' ');
                        line.push_str(arg);
                        used += 1 + len;
                    }
                }
                out.push(Line::verbatim(line));
            }
            Tactic::Raw(text) => {
                for (i, line) in text.lines().enumerate() {
                    // Continuation lines of a multi-line raw tactic keep
                    // their relative indentation, two columns in.
                    let cont = if i > 0 { "  " } else { "" };
                    out.push(Line::wrap(format!("{indent}{cont}{line}")));
                }
                if text.is_empty() {
                    out.push(skip_line(indent));
                }
            }
        }
    }
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render(""))
    }
}

/// A `theorem` / `lemma` / `example` declaration with a tactic proof.
///
/// Build it with [`Decl::new`] and the `with_*` builders, or as a struct
/// literal; [`render`](Self::render) emits the `preamble` lines verbatim,
/// then the doc comment, the header and the body.
///
/// ```
/// use symplex::lean::{Block, Decl, DeclKind, Tactic};
///
/// let d = Decl::new(
///     DeclKind::Lemma,
///     "leaf_0",
///     "0 ≤ r + (j : ℝ)",
///     Block::new(vec![Tactic::raw("have hJ : (0 : ℝ) ≤ (j : ℝ) := by positivity"),
///                     Tactic::raw("linarith only [e0, hJ]")]),
/// )
/// .with_binders(vec!["(j : ℕ)".into(), "(hj : 1 ≤ j)".into(), "(r t : ℝ)".into(),
///                    "(e0 : (0 : ℝ) ≤ r)".into()])
/// .with_doc("Leaf 0: an example.")
/// .with_preamble(vec!["-- generated".into(), "set_option maxHeartbeats 400000 in".into()]);
/// assert_eq!(
///     d.render(),
///     "-- generated\n\
///      set_option maxHeartbeats 400000 in\n\
///      /-- Leaf 0: an example. -/\n\
///      lemma leaf_0 (j : ℕ) (hj : 1 ≤ j) (r t : ℝ) (e0 : (0 : ℝ) ≤ r) :\n\
///      \x20   0 ≤ r + (j : ℝ) := by\n\
///      \x20 have hJ : (0 : ℝ) ≤ (j : ℝ) := by positivity\n\
///      \x20 linarith only [e0, hJ]\n"
/// );
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decl {
    /// `theorem`, `lemma` or `example`.
    pub kind: DeclKind,
    /// The declaration name (ignored for `example`).
    pub name: String,
    /// Binders such as `(x : ℝ)` or `(h : 0 ≤ x)`, each already
    /// parenthesised.
    pub binders: Vec<String>,
    /// The proposition proved.
    pub statement: String,
    /// The proof (rendered as `skip` when empty, so the `by` still parses).
    pub body: Block,
    /// An optional doc comment (`/-- … -/`), without the delimiters.  A
    /// `-/` or `/-` inside the text is escaped as `-\/` / `/\-` when
    /// rendered (Lean's comment lexer would otherwise close the comment or
    /// open a nested one; Markdown drops the backslash).
    pub doc: Option<String>,
    /// Lines emitted verbatim (one per entry, unwrapped) before the doc
    /// comment: `set_option maxHeartbeats 400000 in`, an `open … in`, a
    /// `-- comment`.
    pub preamble: Vec<String>,
}

/// The kind of a [`Decl`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclKind {
    /// `theorem`.
    Theorem,
    /// `lemma` (Mathlib).
    Lemma,
    /// `example` (anonymous).
    Example,
}

impl Decl {
    /// A declaration with no binders, doc comment or preamble.
    pub fn new(
        kind: DeclKind,
        name: impl Into<String>,
        statement: impl Into<String>,
        body: Block,
    ) -> Self {
        Decl {
            kind,
            name: name.into(),
            binders: Vec::new(),
            statement: statement.into(),
            body,
            doc: None,
            preamble: Vec::new(),
        }
    }

    /// Set [`binders`](Self::binders).
    #[must_use]
    pub fn with_binders(mut self, binders: Vec<String>) -> Self {
        self.binders = binders;
        self
    }

    /// Set [`doc`](Self::doc).
    #[must_use]
    pub fn with_doc(mut self, doc: impl Into<String>) -> Self {
        self.doc = Some(doc.into());
        self
    }

    /// Set [`preamble`](Self::preamble).
    #[must_use]
    pub fn with_preamble(mut self, preamble: Vec<String>) -> Self {
        self.preamble = preamble;
        self
    }

    /// The declaration as Lean source, wrapped to Mathlib's width: the
    /// preamble lines verbatim, then the doc comment, then the header,
    /// which packs binders greedily onto the first line and continuation
    /// lines indented four columns, ends the binder list with ` :`, puts the
    /// statement on its own line and closes with ` := by`; the body follows
    /// at two columns.
    pub fn render(&self) -> String {
        self.render_width(MATHLIB_LINE_WIDTH)
    }

    /// [`render`](Self::render) with an explicit width.
    pub fn render_width(&self, width: usize) -> String {
        let mut out = String::new();
        for line in &self.preamble {
            out.push_str(line);
            out.push('\n');
        }
        if let Some(doc) = &self.doc {
            out.push_str("/-- ");
            out.push_str(&escape_doc_comment(doc));
            out.push_str(" -/\n");
        }
        let keyword = match self.kind {
            DeclKind::Theorem => "theorem",
            DeclKind::Lemma => "lemma",
            DeclKind::Example => "example",
        };
        let mut header = match self.kind {
            DeclKind::Example => keyword.to_string(),
            _ => format!("{keyword} {}", lean_ident(&self.name)),
        };
        for b in &self.binders {
            header.push(' ');
            header.push_str(b);
        }
        header.push_str(" :\n    ");
        header.push_str(&self.statement);
        header.push_str(" := by\n");
        out.push_str(&wrap_lean(&header, width));
        if self.body.is_empty() {
            out.push_str("  skip\n");
        } else {
            out.push_str(&self.body.render_width("  ", width));
        }
        out
    }
}

/// Neutralise the comment delimiters inside a `/-- … -/` body: `-/` would
/// close it early and `/-` would open a nested comment that swallows the
/// rest of the file.  A backslash breaks each pair (`-\/`, `/\-`) and is a
/// Markdown escape, so rendered documentation shows the original text.
fn escape_doc_comment(doc: &str) -> String {
    doc.replace("-/", "-\\/").replace("/-", "/\\-")
}

impl fmt::Display for Decl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_by_blocks_and_bullets_indent_structurally() {
        let b = Block::new(vec![
            Tactic::have(
                "hg",
                Some("(0 : ℝ) ≤ x"),
                Proof::by(Block::new(vec![Tactic::raw("nlinarith [hx]")])),
            ),
            Tactic::bullet(Block::new(vec![
                Tactic::have(
                    "inner",
                    None,
                    Proof::by(Block::new(vec![Tactic::raw("simp"), Tactic::raw("ring")])),
                ),
                Tactic::bullet(Block::new(vec![Tactic::raw("exact inner")])),
            ])),
            Tactic::raw("linarith only [hg]"),
        ]);
        assert_eq!(
            b.render("    "),
            "    have hg : (0 : ℝ) ≤ x := by\n      nlinarith [hx]\n    · have inner := by\n        simp\n        ring\n      · exact inner\n    linarith only [hg]\n"
        );
    }

    #[test]
    fn long_bullet_lines_wrap_past_the_tactic_column() {
        let long = format!(
            "linarith only [{}]",
            (0..30)
                .map(|i| format!("h{i}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let b = Block::new(vec![Tactic::bullet(Block::new(vec![
            Tactic::raw(long),
            Tactic::raw("exact h"),
        ]))]);
        let text = b.render_width("  ", 60);
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("  · linarith only [h0,"));
        // Continuation lines sit right of the bullet's tactic column (4).
        assert!(lines[1].starts_with("      h"), "{text}");
        assert!(lines.iter().all(|l| l.chars().count() <= 60), "{text}");
        assert_eq!(*lines.last().unwrap(), "    exact h");
    }

    /// The leaf shape of a downstream generator (a `refine … ?_` call
    /// followed by one bullet per facet, each bullet a polyhedron
    /// certificate's closing block), rendered from the structure.  The
    /// expected text is taken verbatim from a file that compiled against
    /// Mathlib: the wrapped `have hg` type continues at the bullet's tactic
    /// column + 2, and the `by` block's `linarith` sits at the same column
    /// — legal because it follows `:= by`, not a completed tactic.
    #[test]
    fn generator_leaf_shape_matches_mathlib_compiled_text() {
        let hg_ty = "(0 : ℝ) ≤ (150 * (j : ℝ) / 431 + 1) * (-(60 * (j : ℝ) * t) + 80 * (j : ℝ) * r + 42 * (j : ℝ) + 40 * r - 30 * t + 21)";
        let facet = Block::new(vec![
            Tactic::have(
                "hg",
                Some(hg_ty),
                Proof::by(Block::new(vec![Tactic::raw(
                    "linarith only [e0, e0J, e0JJ, e2JK, e3, e3J, e5, e5J, e5JJ, e6, hK0]",
                )])),
            ),
            Tactic::raw("have hg' := nonneg_of_mul_nonneg_right hg (by linarith only [hJ0])"),
            Tactic::raw("linarith only [hg']"),
        ]);
        let leaf = Block::new(vec![
            Tactic::raw(
                "refine leafG346_single_poly (20 * (j : ℝ) + 10) ρ r t u T₅ hD hρ hx\n  ?_ ?_",
            ),
            Tactic::bullet(facet),
            Tactic::bullet(Block::new(vec![Tactic::raw(
                "linarith only [e6, e9, e11, e15]",
            )])),
        ]);
        assert_eq!(
            leaf.render("  "),
            "  refine leafG346_single_poly (20 * (j : ℝ) + 10) ρ r t u T₅ hD hρ hx\n\
             \x20     ?_ ?_\n\
             \x20 · have hg : (0 : ℝ) ≤ (150 * (j : ℝ) / 431 + 1) *\n\
             \x20     (-(60 * (j : ℝ) * t) + 80 * (j : ℝ) * r + 42 * (j : ℝ) + 40 * r - 30 * t + 21) := by\n\
             \x20     linarith only [e0, e0J, e0JJ, e2JK, e3, e3J, e5, e5J, e5JJ, e6, hK0]\n\
             \x20   have hg' := nonneg_of_mul_nonneg_right hg (by linarith only [hJ0])\n\
             \x20   linarith only [hg']\n\
             \x20 · linarith only [e6, e9, e11, e15]\n"
        );
    }

    /// A `refine` with many `?_` placeholders: the head and as many
    /// arguments as fit on the first line, the rest packed onto
    /// continuation lines two columns past the tactic column, no argument
    /// ever split, and the result a fixed point of `wrap_lean`.
    #[test]
    fn apply_packs_atoms_greedily_and_is_stable_under_wrapping() {
        let fixed = [
            "(20 * (j : ℝ) + 10)",
            "ρ",
            "r",
            "t",
            "u",
            "T₅",
            "hD",
            "hρ",
            "hx",
        ];
        let args: Vec<String> = fixed
            .iter()
            .map(|s| s.to_string())
            .chain(std::iter::repeat_n("?_".to_string(), 28))
            .collect();
        let tactic = Tactic::apply("refine leafG346_single_poly", args.clone());
        assert!(tactic.introduced_names().is_empty());
        let text = Block::new(vec![tactic.clone()]).render("  ");
        let lines: Vec<&str> = text.lines().collect();
        assert!(
            lines[0].starts_with(
                "  refine leafG346_single_poly (20 * (j : ℝ) + 10) ρ r t u T₅ hD hρ hx ?_"
            ),
            "{text}"
        );
        assert!(lines.len() >= 2, "{text}");
        for l in &lines[1..] {
            assert!(l.starts_with("    ?_") && !l.starts_with("     "), "{l:?}");
        }
        assert!(lines.iter().all(|l| l.chars().count() <= 100), "{text}");
        // Greedy: every line but the last would overflow with one more `?_`.
        for l in &lines[..lines.len() - 1] {
            assert!(l.chars().count() + 3 > 100, "not greedy: {l:?}");
        }
        // Nothing lost, nothing split.
        let tokens: Vec<&str> = text.split_whitespace().collect();
        assert_eq!(tokens.iter().filter(|t| **t == "?_").count(), 28);
        let mut expected = vec!["refine", "leafG346_single_poly"];
        expected.extend(args.iter().flat_map(|a| a.split_whitespace()));
        assert_eq!(tokens, expected);
        assert_eq!(wrap_lean(&text, MATHLIB_LINE_WIDTH), text, "idempotent");
        // Inside a bullet the tactic column moves by two and so does the
        // continuation indent.
        let bulleted = Block::new(vec![Tactic::bullet(Block::new(vec![tactic]))]).render("  ");
        let bl: Vec<&str> = bulleted.lines().collect();
        assert!(
            bl[0].starts_with("  · refine leafG346_single_poly"),
            "{bulleted}"
        );
        assert!(
            bl[1].starts_with("      ?_") && !bl[1].starts_with("       "),
            "{bulleted}"
        );
        assert!(bl.iter().all(|l| l.chars().count() <= 100), "{bulleted}");
        assert_eq!(wrap_lean(&bulleted, MATHLIB_LINE_WIDTH), bulleted);
        // A head that fills the line pushes the first argument down; a
        // continuation line always carries its argument whole.
        let narrow = Block::new(vec![Tactic::apply(
            "refine long_lemma_name",
            vec!["(a + b)".into(), "?_".into()],
        )])
        .render_width("", 24);
        assert_eq!(narrow, "refine long_lemma_name\n  (a + b) ?_\n");
        assert_eq!(
            Block::new(vec![Tactic::apply("exact h", vec![])]).render("  "),
            "  exact h\n"
        );
    }

    #[test]
    fn decl_builders_and_preamble() {
        let d = Decl::new(
            DeclKind::Theorem,
            "t",
            "0 ≤ x",
            Block::new(vec![Tactic::raw("exact hx")]),
        );
        assert_eq!(d.render(), "theorem t :\n    0 ≤ x := by\n  exact hx\n");
        let d = d
            .with_binders(vec!["(x : ℝ)".into(), "(hx : 0 ≤ x)".into()])
            .with_doc("Trivial.")
            .with_preamble(vec![
                "set_option maxHeartbeats 400000 in".into(),
                "-- a comment longer than any width would allow if it were wrapped, which it is not".into(),
            ]);
        let text = d.render_width(40);
        assert!(text.starts_with(
            "set_option maxHeartbeats 400000 in\n-- a comment longer than any width would allow if it were wrapped, which it is not\n/-- Trivial. -/\ntheorem t (x : ℝ) (hx : 0 ≤ x) :\n"
        ), "{text}");
        assert!(text.ends_with("    0 ≤ x := by\n  exact hx\n"), "{text}");
        // The struct literal still works with the new field.
        let lit = Decl {
            kind: DeclKind::Example,
            name: String::new(),
            binders: vec![],
            statement: "True".into(),
            body: Block::new(vec![Tactic::raw("trivial")]),
            doc: None,
            preamble: vec![],
        };
        assert_eq!(lit.render(), "example :\n    True := by\n  trivial\n");
    }

    #[test]
    fn names_and_empty_blocks() {
        let b = Block::new(vec![
            Tactic::have("a", None, Proof::term("rfl")),
            Tactic::bullet(Block::new(vec![Tactic::have(
                "b",
                None,
                Proof::by(Block::new(vec![Tactic::have(
                    "c",
                    None,
                    Proof::term("rfl"),
                )])),
            )])),
        ]);
        let names: Vec<String> = b
            .tactics
            .iter()
            .flat_map(Tactic::introduced_names)
            .collect();
        assert_eq!(names, ["a", "b", "c"]);
        assert_eq!(Block::default().render("  "), "");
        assert_eq!(
            Tactic::bullet(Block::default()).introduced_names(),
            Vec::<String>::new()
        );
        assert_eq!(
            Block::new(vec![Tactic::bullet(Block::default())]).render(""),
            "· skip\n"
        );
    }
}
