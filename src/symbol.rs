use rustc_hash::FxHashMap;

use crate::node::SymbolId;

/// A table that interns symbol names and maps them to unique [`SymbolId`]s.
///
/// Each distinct name is stored exactly once. Subsequent calls to [`intern`](SymbolTable::intern)
/// with the same name return the previously assigned [`SymbolId`], guaranteeing
/// identity-based equality for symbols throughout the system.
pub struct SymbolTable {
    /// Symbol names indexed by [`SymbolId`].
    names: Vec<String>,
    /// Per-symbol mathematical assumptions, parallel to `names`.
    assumptions: Vec<crate::assumptions::Assumptions>,
    /// Reverse lookup from name to [`SymbolId`] for deduplication.
    lookup: FxHashMap<String, SymbolId>,
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

        let id = SymbolId(self.names.len() as u32);
        self.names.push(name.to_owned());
        self.assumptions.push(crate::assumptions::Assumptions {
            known_true: crate::assumptions::Props::empty(),
            known_false: crate::assumptions::Props::empty(),
        });
        self.lookup.insert(name.to_owned(), id);
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

    /// Returns the number of interned symbols.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Returns `true` if no symbols have been interned.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Set mathematical assumptions for a symbol.
    pub fn set_assumptions(&mut self, id: SymbolId, a: crate::assumptions::Assumptions) {
        self.assumptions[id.0 as usize] = a;
    }

    /// Get the stored assumptions for a symbol.
    pub fn get_assumptions(&self, id: SymbolId) -> crate::assumptions::Assumptions {
        self.assumptions[id.0 as usize]
    }
}
