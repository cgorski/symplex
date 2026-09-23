//! The line protocol between the driver and its worker processes, and the
//! classification shared by both.
//!
//! A worker writes one record per line on stdout, tab-separated:
//!
//! ```text
//! BEGIN <idx>                 an entry starts (integration deadline starts)
//! PHASE <idx> check           integration finished, verification starts
//! KV    <idx> <key> <value>   a detail, sent as soon as it is known
//! END   <idx> <status>        the entry's final classification
//! ```
//!
//! Values are escaped with [`escape`] so that they fit on one line.
//! Details are streamed before `END` so that a worker killed for a timeout
//! (or dying from a stack overflow) still leaves what it had found.

use std::fmt;

/// Classification of one entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Status {
    /// The integrand could not be translated or parsed, or uses functions
    /// symplex lacks.
    Unsupported,
    /// `F′ = f` numerically at every point where both sides evaluate.
    Verified,
    /// `F′ = f` at every point where the integrand is real; `F′ ≠ f` only at
    /// points where it is complex, and the integrand has no `%i`.  This is
    /// symplex's real-variable convention (`ln|u|` for `∫u′/u`), not a bug.
    RealVerified,
    /// The result contains an unevaluated `Integral` (or other formal node).
    Unevaluated,
    /// `F′ ≠ f` at some point where the integrand is real, or anywhere if
    /// the integrand contains `%i` (then it is complex by construction).
    Wrong,
    /// No point evaluated on both sides (or the check itself timed out).
    Undecided,
    /// The integration exceeded its wall-clock limit (or memory limit).
    Timeout,
    /// symplex panicked (or the worker process died) during parsing,
    /// integration or the check.
    Panic,
}

impl Status {
    pub const ALL: [Status; 8] = [
        Status::Unsupported,
        Status::Verified,
        Status::RealVerified,
        Status::Unevaluated,
        Status::Wrong,
        Status::Undecided,
        Status::Timeout,
        Status::Panic,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Status::Unsupported => "unsupported",
            Status::Verified => "verified",
            Status::RealVerified => "real_verified",
            Status::Unevaluated => "unevaluated",
            Status::Wrong => "wrong",
            Status::Undecided => "undecided",
            Status::Timeout => "timeout",
            Status::Panic => "panic",
        }
    }

    pub fn parse(s: &str) -> Option<Status> {
        Status::ALL.into_iter().find(|st| st.as_str() == s)
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Make `s` safe for one tab-separated field.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
}

/// Inverse of [`escape`].
pub fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some(o) => out.push(o),
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_round_trips() {
        let s = "a\tb\\n\nc\r";
        assert_eq!(unescape(&escape(s)), s);
        assert!(!escape(s).contains('\t'));
        assert!(!escape(s).contains('\n'));
    }

    #[test]
    fn status_names_round_trip() {
        for st in Status::ALL {
            assert_eq!(Status::parse(st.as_str()), Some(st));
        }
    }
}
