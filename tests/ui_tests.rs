#[test]
fn units_compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
