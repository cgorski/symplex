//! Property-based tests (`tests/proptests/*.rs`: the former `proptest_*.rs`,
//! `test_proptest_new.rs`, `test_quality_props.rs` and `test_units_proptest.rs`).
//! Each module's `<stem>.proptest-regressions` file lives next to its source.

mod common;

#[path = "proptests/proptest_algebraic.rs"]
mod proptest_algebraic;
#[path = "proptests/proptest_bounded.rs"]
mod proptest_bounded;
#[path = "proptests/proptest_canon.rs"]
mod proptest_canon;
#[path = "proptests/proptest_roundtrips.rs"]
mod proptest_roundtrips;
#[path = "proptests/proptest_stress.rs"]
mod proptest_stress;
#[path = "proptests/proptest_transforms.rs"]
mod proptest_transforms;
#[path = "proptests/proptest_verification.rs"]
mod proptest_verification;
#[path = "proptests/test_proptest_new.rs"]
mod test_proptest_new;
#[path = "proptests/test_quality_props.rs"]
mod test_quality_props;
#[path = "proptests/test_units_proptest.rs"]
mod test_units_proptest;
