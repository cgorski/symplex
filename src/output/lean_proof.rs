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
    /// Any other tactic, verbatim (`linarith only [a, b]`, `rcases … with
    /// h | h`, `refine f ?_ ?_`, `exact h`, `positivity`).  May span
    /// several lines; each line is placed at the tactic column, later lines
    /// indented as given relative to the first.
    Raw(String),
}

/// The right-hand side of a `have`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Proof {
    /// A term, rendered after `:=` on the `have` line.
    Term(String),
    /// A tactic block, rendered on the lines below `:= by`, two columns in.
    By(Block),
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
            Tactic::Raw(_) => {}
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
        let mut out = String::new();
        self.write_lines(indent, &mut out);
        wrap_lean(&out, width)
    }

    /// The logical lines (unwrapped), each prefixed by its indentation.
    fn write_lines(&self, indent: &str, out: &mut String) {
        for t in &self.tactics {
            t.write_lines(indent, out);
        }
    }
}

impl Tactic {
    fn write_lines(&self, indent: &str, out: &mut String) {
        match self {
            Tactic::Have { name, ty, proof } => {
                let head = match ty {
                    Some(t) => format!("have {} : {t}", lean_ident(name)),
                    None => format!("have {}", lean_ident(name)),
                };
                match proof {
                    Proof::Term(term) => {
                        out.push_str(indent);
                        out.push_str(&head);
                        out.push_str(" := ");
                        out.push_str(term);
                        out.push('\n');
                    }
                    Proof::By(block) => {
                        out.push_str(indent);
                        out.push_str(&head);
                        out.push_str(" := by\n");
                        let inner = format!("{indent}  ");
                        block.write_lines(&inner, out);
                    }
                }
            }
            Tactic::Bullet(block) => {
                // The first tactic shares the bullet's line; the rest of the
                // block is at the bullet's column + 2.
                let inner = format!("{indent}  ");
                let mut body = String::new();
                block.write_lines(&inner, &mut body);
                if body.is_empty() {
                    out.push_str(indent);
                    out.push_str("· skip\n");
                    return;
                }
                // Replace the first line's leading `inner` with `indent· `.
                let first_rest = &body[inner.len()..];
                out.push_str(indent);
                out.push_str("· ");
                out.push_str(first_rest);
            }
            Tactic::Raw(text) => {
                for (i, line) in text.lines().enumerate() {
                    out.push_str(indent);
                    if i > 0 {
                        // Continuation lines of a multi-line raw tactic keep
                        // their relative indentation, two columns in.
                        out.push_str("  ");
                    }
                    out.push_str(line);
                    out.push('\n');
                }
                if text.is_empty() {
                    out.push_str(indent);
                    out.push_str("skip\n");
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
/// ```
/// use symplex::lean::{Block, Decl, DeclKind, Tactic};
///
/// let d = Decl {
///     kind: DeclKind::Lemma,
///     name: "leaf_0".into(),
///     binders: vec!["(j : ℕ)".into(), "(hj : 1 ≤ j)".into(), "(r t : ℝ)".into(),
///                   "(e0 : (0 : ℝ) ≤ r)".into()],
///     statement: "0 ≤ r + (j : ℝ)".into(),
///     body: Block::new(vec![Tactic::raw("have hJ : (0 : ℝ) ≤ (j : ℝ) := by positivity"),
///                           Tactic::raw("linarith only [e0, hJ]")]),
///     doc: Some("Leaf 0: an example.".into()),
/// };
/// assert_eq!(
///     d.render(),
///     "/-- Leaf 0: an example. -/\n\
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
    /// The proof.
    pub body: Block,
    /// An optional doc comment (`/-- … -/`), without the delimiters.
    pub doc: Option<String>,
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
    /// The declaration as Lean source, wrapped to Mathlib's width: the
    /// header packs binders greedily onto the first line and continuation
    /// lines indented four columns, ends the binder list with ` :`, puts the
    /// statement on its own line and closes with ` := by`; the body follows
    /// at two columns.
    pub fn render(&self) -> String {
        self.render_width(MATHLIB_LINE_WIDTH)
    }

    /// [`render`](Self::render) with an explicit width.
    pub fn render_width(&self, width: usize) -> String {
        let mut out = String::new();
        if let Some(doc) = &self.doc {
            out.push_str("/-- ");
            out.push_str(doc);
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
        out.push_str(&self.body.render_width("  ", width));
        out
    }
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
