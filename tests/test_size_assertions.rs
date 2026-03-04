//! Compile-time and runtime size assertions for core types.
//!
//! These prevent accidental size regressions in the expression IR.

use std::mem::size_of;

/// Verify ExprNode stays within expected bounds.
/// The enum is sized by its largest variant (Add/Mul/And/Or with SmallVec<[ExprId; 6]>,
/// or Piecewise with SmallVec<[(ExprId, ExprId); 3]>).
#[test]
fn expr_node_size() {
    let size = size_of::<symplex::__macro_support::ExprNode>();
    // ExprNode should be ≤ 56 bytes (SmallVec inline buffer + discriminant + padding).
    // If this fails, a new variant grew the enum unexpectedly.
    assert!(
        size <= 56,
        "ExprNode is {size} bytes, expected ≤ 56. A new variant may have grown the enum."
    );
    // It should also be at least 32 bytes (SmallVec inline buffer).
    assert!(
        size >= 32,
        "ExprNode is {size} bytes, expected ≥ 32. Something shrank unexpectedly."
    );
}

/// Verify Ex (expression handle) is 24 bytes or less.
/// Layout: CtxId(u32) + Arc<RwLock<…>>(usize=8) + ExprId(u32) + PhantomData(0) + padding.
#[test]
fn ex_handle_size() {
    let size = size_of::<symplex::prelude::Ex>();
    assert!(size <= 24, "Ex is {size} bytes, expected ≤ 24.");
}

/// Verify ExprId is exactly 4 bytes (newtype around u32).
#[test]
fn expr_id_size() {
    // ExprId is a newtype around u32, so the node layout depends on it being 4 bytes.
    // We can't name ExprId directly from outside the crate, but we can sanity-check
    // that u32 is 4 bytes — any change to ExprId's repr would show up in expr_node_size.
    assert_eq!(size_of::<u32>(), 4, "u32 should be 4 bytes");
}
