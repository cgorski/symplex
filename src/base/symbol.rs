use rustc_hash::FxHashMap;

use crate::base::node::SymbolId;

/// A table that interns symbol names and maps them to unique [`SymbolId`]s.
///
/// Each distinct name is stored exactly once. Subsequent calls to [`intern`](SymbolTable::intern)
/// with the same name return the previously assigned [`SymbolId`], guaranteeing
/// identity-based equality for symbols throughout the system.
pub struct SymbolTable {
    /// Symbol names indexed by [`SymbolId`].
    names: Vec<String>,
    /// Per-symbol mathematical assumptions, parallel to `names`.
    assumptions: Vec<crate::base::assumptions::Assumptions>,
    /// Reverse lookup from name to [`SymbolId`] for deduplication.
    lookup: FxHashMap<String, SymbolId>,
    /// The library's own dummy symbols, by name: a second name space,
    /// which [`intern`](Self::intern) and [`get`](Self::get) do not read.
    /// See [`intern_reserved`](Self::intern_reserved).
    reserved: FxHashMap<String, SymbolId>,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolTable {
    /// Creates an empty symbol table.
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            assumptions: Vec::new(),
            lookup: FxHashMap::default(),
            reserved: FxHashMap::default(),
        }
    }

    /// Interns a symbol name, returning its [`SymbolId`].
    ///
    /// If `name` has already been interned, the existing [`SymbolId`] is returned.
    /// Otherwise a new [`SymbolId`] is allocated and associated with `name`.
    pub fn intern(&mut self, name: &str) -> SymbolId {
        if let Some(&id) = self.lookup.get(name) {
            return id;
        }

        let id = self.push_name(name);
        self.lookup.insert(name.to_owned(), id);
        id
    }

    /// A new entry named `name`, in neither name space yet.
    fn push_name(&mut self, name: &str) -> SymbolId {
        let id = SymbolId(
            u32::try_from(self.names.len())
                .expect("symbol table overflow: more than 4 billion symbols"),
        );
        self.names.push(name.to_owned());
        self.assumptions
            .push(crate::base::assumptions::Assumptions {
                known_true: crate::base::assumptions::Props::empty(),
                known_false: crate::base::assumptions::Props::empty(),
            });
        id
    }

    /// Returns the name associated with the given [`SymbolId`].
    ///
    /// # Panics
    ///
    /// Panics if `id` was not produced by this table.
    pub fn name(&self, id: SymbolId) -> &str {
        &self.names[id.0 as usize]
    }

    /// The [`SymbolId`] of `name` if it has been interned, without
    /// interning it.
    pub fn get(&self, name: &str) -> Option<SymbolId> {
        self.lookup.get(name).copied()
    }

    /// Interns `name` in the name space of the library's own dummy symbols
    /// (the variable of a substitution inside the limit engine, a symbol
    /// declared positive for one computation).  No name given to
    /// [`intern`](Self::intern) or [`get`](Self::get) — the entry points of
    /// `Context::symbol`, of the parser and of the JSON reader — ever
    /// returns such a symbol, so declaring an assumption on it cannot touch
    /// a symbol of the user's, and a user's symbol cannot be captured by
    /// it.  The symbol prints as `name`.
    ///
    /// Two symbols with one name have one sort key, so the caller chooses a
    /// name that [`get`](Self::get) does not know at that moment
    /// (`Arena::internal_symbol` does).
    pub(crate) fn intern_reserved(&mut self, name: &str) -> SymbolId {
        if let Some(&id) = self.reserved.get(name) {
            return id;
        }
        let id = self.push_name(name);
        self.reserved.insert(name.to_owned(), id);
        id
    }

    /// The symbol of the library's own name space named `name`, if there is
    /// one ([`intern_reserved`](Self::intern_reserved)), without making it.
    pub(crate) fn get_reserved(&self, name: &str) -> Option<SymbolId> {
        self.reserved.get(name).copied()
    }

    /// Is `id` a symbol of the library's own name space
    /// ([`intern_reserved`](Self::intern_reserved))?
    pub(crate) fn is_reserved(&self, id: SymbolId) -> bool {
        self.names
            .get(id.0 as usize)
            .is_some_and(|name| self.reserved.get(name) == Some(&id))
    }

    /// Returns the number of interned symbols.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Set mathematical assumptions for a symbol.
    pub fn set_assumptions(&mut self, id: SymbolId, a: crate::base::assumptions::Assumptions) {
        self.assumptions[id.0 as usize] = a;
    }

    /// Get the stored assumptions for a symbol.
    pub fn get_assumptions(&self, id: SymbolId) -> crate::base::assumptions::Assumptions {
        self.assumptions[id.0 as usize]
    }
}
