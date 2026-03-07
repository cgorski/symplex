//! A non-locking, read-only view of an expression node.
//!
//! [`ExprView`] is passed to the closure in [`Expr::replace()`].
//! It provides structural inspection of the current node without
//! acquiring any locks.  You **cannot** call `.sin()`, `.expand()`,
//! `format!()`, or any mutating/locking method on this type.

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use smallvec::SmallVec;

/// A non-locking, read-only view into an expression node within the arena.
///
/// This type intentionally does NOT implement `Display`, `Clone`, or any
/// method that would require locking the context.  This makes it
/// impossible to deadlock when used inside `Expr::replace()`.
pub struct ExprView<'a> {
    pub(crate) id: ExprId,
    pub(crate) arena: &'a Arena,
}

impl<'a> ExprView<'a> {
    /// Returns the [`ExprId`] of the viewed expression.
    #[inline]
    pub fn id(&self) -> ExprId {
        self.id
    }

    /// Returns the [`ExprNode`] variant of this expression.
    #[inline]
    pub fn node(&self) -> &ExprNode {
        self.arena.node(self.id)
    }

    /// Returns `true` if this node is an atom (leaf node with no children).
    #[inline]
    pub fn is_atom(&self) -> bool {
        self.arena.node(self.id).is_atom()
    }

    /// Returns `true` if this node is a symbol.
    #[inline]
    pub fn is_symbol(&self) -> bool {
        matches!(self.arena.node(self.id), ExprNode::Symbol(_))
    }

    /// Returns the children of this node.
    pub fn children(&self) -> SmallVec<[ExprId; 6]> {
        self.arena.node(self.id).children()
    }
}

// PartialEq with Expr — compare by ExprId only (no locking needed)
impl<S: crate::api::expr::Sort> PartialEq<crate::api::expr::Expr<S>> for ExprView<'_> {
    fn eq(&self, other: &crate::api::expr::Expr<S>) -> bool {
        self.id == other.id
    }
}

impl<S: crate::api::expr::Sort> PartialEq<&crate::api::expr::Expr<S>> for ExprView<'_> {
    fn eq(&self, other: &&crate::api::expr::Expr<S>) -> bool {
        self.id == other.id
    }
}
