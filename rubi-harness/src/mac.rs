//! Reader for the Rubi `.mac` test files.
//!
//! A file is a Maxima assignment `lst: '[ entry, entry, ... ]$` whose
//! entries are themselves lists `[integrand, variable, steps, optimal, ...]`.
//! Comments (`/* ... */`, nestable in Maxima) are blanked out first, keeping
//! newlines so that line numbers stay meaningful; the top-level list is then
//! split into entries by bracket nesting, and each entry into fields at its
//! depth-0 commas.

use std::path::Path;

/// One test case.
#[derive(Debug, Clone)]
pub struct Entry {
    /// 1-based line of the entry's opening `[` in the file.
    pub line: usize,
    /// The comma-separated fields, trimmed.  Normally four:
    /// integrand, variable, Rubi's step count, optimal antiderivative.
    pub fields: Vec<String>,
}

impl Entry {
    pub fn integrand(&self) -> &str {
        self.fields.first().map_or("", String::as_str)
    }

    pub fn variable(&self) -> &str {
        self.fields.get(1).map_or("", String::as_str)
    }

    /// Rubi's optimal antiderivative.  `None` if missing, or if the step
    /// count is negative: the suite marks integrands Rubi cannot do that
    /// way, with a placeholder (usually `0`) as the "antiderivative".
    pub fn optimal(&self) -> Option<&str> {
        if self
            .fields
            .get(2)
            .is_some_and(|s| s.trim_start().starts_with('-'))
        {
            return None;
        }
        self.fields.get(3).map(String::as_str)
    }
}

/// Read and split a `.mac` file.
pub fn read_file(path: &Path) -> Result<Vec<Entry>, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_source(&src).map_err(|e| format!("{}: {e}", path.display()))
}

/// Split the text of a `.mac` file into entries.
pub fn parse_source(src: &str) -> Result<Vec<Entry>, String> {
    let text = strip_comments(src)?;
    let bytes = text.as_bytes();
    let start = text.find("lst:").ok_or("no `lst:` assignment found")?;
    let open = text[start..]
        .find('[')
        .map(|p| start + p)
        .ok_or("no list after `lst:`")?;

    let mut entries = Vec::new();
    let mut line = 1 + bytes[..open].iter().filter(|&&b| b == b'\n').count();
    // Nesting depth of `(`/`[` counted from the outer list's `[` (depth 1).
    let mut depth = 1usize;
    let mut entry_start: Option<(usize, usize)> = None;
    let mut i = open + 1;
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b'\n' => line += 1,
            b'[' | b'(' => {
                if depth == 1 {
                    if b == b'(' {
                        return Err(format!("line {line}: `(` at list level"));
                    }
                    entry_start = Some((i + 1, line));
                }
                depth += 1;
            }
            b']' | b')' => {
                depth -= 1;
                if depth == 1 {
                    let (s, l) = entry_start
                        .take()
                        .ok_or_else(|| format!("line {line}: unbalanced `{}`", b as char))?;
                    entries.push(Entry {
                        line: l,
                        fields: split_top_level(&text[s..i]),
                    });
                } else if depth == 0 {
                    return Ok(entries);
                }
            }
            b',' | b' ' | b'\t' | b'\r' => {}
            _ if depth == 1 => {
                return Err(format!(
                    "line {line}: unexpected `{}` between entries",
                    b as char
                ));
            }
            _ => {}
        }
        i += 1;
    }
    Err("unterminated top-level list".into())
}

/// Replace every comment by spaces (newlines are kept).  Maxima comments
/// nest.
fn strip_comments(src: &str) -> Result<String, String> {
    let bytes = src.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut depth = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
            depth += 1;
            out.extend_from_slice(b"  ");
            i += 2;
        } else if depth > 0 && bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
            depth -= 1;
            out.extend_from_slice(b"  ");
            i += 2;
        } else {
            let b = bytes[i];
            out.push(if depth > 0 && b != b'\n' { b' ' } else { b });
            i += 1;
        }
    }
    if depth > 0 {
        return Err("unterminated comment".into());
    }
    // Only ASCII bytes inside comments were replaced by ASCII spaces, but a
    // multi-byte character inside a comment is blanked byte by byte, which
    // keeps the output valid UTF-8 (every byte becomes a space).
    String::from_utf8(out).map_err(|e| e.to_string())
}

/// Split at commas that are not nested in `()` or `[]`; fields are trimmed.
pub fn split_top_level(s: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                fields.push(s[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    fields.push(s[start..].trim().to_string());
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_entries_and_fields() {
        let src = "/* header \"a\\b\" */\n\nlst: '[\n/* c /* nested */ still */\n[sin(a+b*x),x,1,-cos(a+b*x)/b],\n[f(x,[1,2]),x,2,g(x),h],\n[1,x,1,x]]$\n";
        let es = parse_source(src).unwrap();
        assert_eq!(es.len(), 3);
        assert_eq!(es[0].line, 5);
        assert_eq!(es[0].fields, vec!["sin(a+b*x)", "x", "1", "-cos(a+b*x)/b"]);
        assert_eq!(es[1].fields, vec!["f(x,[1,2])", "x", "2", "g(x)", "h"]);
        assert_eq!(es[2].line, 7);
        assert_eq!(es[2].optimal(), Some("x"));
    }
}
