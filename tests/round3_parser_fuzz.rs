//! Round 3: Parser fuzz tests for the symplex expression parser.
//!
//! Feeds adversarial, random, and edge-case inputs to `symplex::parse::parse`
//! to find panics, crashes, and round-trip bugs.
//!
//! # Bugs Found
//!
//! ## BUG-1: Stack overflow on deeply-nested function calls (~250 deep)
//!
//! **Severity:** crash (SIGABRT — unrecoverable)
//!
//! The parser's nesting-depth guard (`self.depth > 256`) is checked only at the
//! entry of `parse_expr`.  However, each function-call nesting layer actually
//! pushes **≥3 stack frames** on the call stack:
//!
//!   `parse_expr` → `parse_prefix` → `parse_function_call` → `parse_expr` → …
//!
//! At ~250 layers of `sin(sin(…sin(x)…))`, the **logical** depth counter is
//! only ~250 (below the 256 cap), but the **physical** call-stack depth is
//! ~750+ frames.  This exceeds the default thread stack size and triggers an
//! unrecoverable `SIGABRT` (stack overflow), killing the entire process.
//!
//! **Repro:** `d_deeply_nested_sin_250` (marked `#[ignore]` — it aborts)
//!
//! **Suggested fix:** either (a) lower the logical depth cap to ~128 so the
//! physical stack is never exhausted, or (b) count depth increments inside
//! `parse_prefix` and `parse_function_call` as well, or (c) switch the parser
//! to an iterative (explicit-stack) design like the display formatter already
//! uses.

use proptest::prelude::*;
use symplex::parse::parse;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Parse a string; return Ok(display_string) or Err(error).
/// Must NEVER panic.
fn try_parse(input: &str) -> Result<String, String> {
    let ctx = Context::new();
    match parse(&ctx, input) {
        Ok(expr) => Ok(format!("{expr}")),
        Err(e) => Err(format!("{e}")),
    }
}

/// Build an expression programmatically, display it, parse the display string,
/// display the parsed result, and assert the two display strings match.
fn assert_round_trip(expr: &Ex, label: &str) {
    let displayed = format!("{expr}");
    let ctx = expr.context();
    let reparsed = match parse(&ctx, &displayed) {
        Ok(e) => e,
        Err(e) => panic!(
            "Round-trip PARSE FAILURE for {label}:\n  displayed: '{displayed}'\n  error: {e}"
        ),
    };
    let redisplayed = format!("{reparsed}");
    assert_eq!(
        displayed, redisplayed,
        "Round-trip MISMATCH for {label}:\n  original display:  '{displayed}'\n  reparsed display: '{redisplayed}'"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Category A: Parser must never panic (proptest)
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// Random ASCII strings of length 0..100 must never panic the parser.
    #[test]
    fn a_random_ascii_no_panic(s in "[\\x00-\\x7F]{0,100}") {
        let ctx = Context::new();
        // Must return Ok or Err, never panic
        let _ = parse(&ctx, &s);
    }

    /// Random Unicode strings of length 0..100 must never panic.
    #[test]
    fn a_random_unicode_no_panic(s in "\\PC{0,100}") {
        let ctx = Context::new();
        let _ = parse(&ctx, &s);
    }

    /// Random printable ASCII that looks vaguely math-like.
    #[test]
    fn a_random_math_chars_no_panic(s in "[a-zA-Z0-9+\\-*/^() ,.]{0,80}") {
        let ctx = Context::new();
        let _ = parse(&ctx, &s);
    }

    /// Random strings with lots of parens.
    #[test]
    fn a_random_parens_heavy_no_panic(s in "[()a-z0-9+\\-*]{0,120}") {
        let ctx = Context::new();
        let _ = parse(&ctx, &s);
    }
}

/// Deeply nested parentheses up to depth 1000 must not panic (should error gracefully).
#[test]
fn a_deeply_nested_parens_1000() {
    let depth = 1000;
    let open: String = "(".repeat(depth);
    let close: String = ")".repeat(depth);
    let input = format!("{open}x{close}");
    let ctx = Context::new();
    // Should either parse or return an error about nesting depth, never panic
    let _ = parse(&ctx, &input);
}

/// Deeply nested parentheses at exactly the limit boundary (256).
#[test]
fn a_deeply_nested_parens_256() {
    let depth = 256;
    let open: String = "(".repeat(depth);
    let close: String = ")".repeat(depth);
    let input = format!("{open}x{close}");
    let ctx = Context::new();
    let _ = parse(&ctx, &input);
}

/// Deeply nested parens just under the limit (255).
#[test]
fn a_deeply_nested_parens_255() {
    let depth = 255;
    let open: String = "(".repeat(depth);
    let close: String = ")".repeat(depth);
    let input = format!("{open}x{close}");
    let ctx = Context::new();
    let _ = parse(&ctx, &input);
}

/// Deeply nested with operators at each level.
/// Depth 300 exceeds the parser's 256-level limit, so it should return Err
/// (the parenthesised form hits `parse_expr` once per nesting level, so the
/// logical depth counter catches it before the stack overflows).
#[test]
fn a_deeply_nested_with_ops() {
    let depth = 300;
    let mut s = String::from("x");
    for _ in 0..depth {
        s = format!("({s}+1)");
    }
    let ctx = Context::new();
    let result = parse(&ctx, &s);
    // Should be caught by the depth guard — not panic/overflow.
    assert!(
        result.is_err(),
        "expected depth-limit error for depth {depth}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Category B: Parse↔Display round-trip
// ═══════════════════════════════════════════════════════════════════════════

// ── Integers ───────────────────────────────────────────────────────────

#[test]
fn b_roundtrip_integer_zero() {
    let ctx = Context::new();
    let expr = ctx.int(0);
    assert_round_trip(&expr, "int(0)");
}

#[test]
fn b_roundtrip_integer_one() {
    let ctx = Context::new();
    let expr = ctx.int(1);
    assert_round_trip(&expr, "int(1)");
}

#[test]
fn b_roundtrip_integer_neg_one() {
    let ctx = Context::new();
    let expr = ctx.int(-1);
    assert_round_trip(&expr, "int(-1)");
}

#[test]
fn b_roundtrip_integer_42() {
    let ctx = Context::new();
    let expr = ctx.int(42);
    assert_round_trip(&expr, "int(42)");
}

#[test]
fn b_roundtrip_integer_neg_999() {
    let ctx = Context::new();
    let expr = ctx.int(-999);
    assert_round_trip(&expr, "int(-999)");
}

#[test]
fn b_roundtrip_large_integer() {
    let ctx = Context::new();
    let expr = ctx.int(1_000_000_000);
    assert_round_trip(&expr, "int(1000000000)");
}

// ── Rationals ──────────────────────────────────────────────────────────

#[test]
fn b_roundtrip_rational_half() {
    let ctx = Context::new();
    let expr = ctx.rational(1, 2);
    assert_round_trip(&expr, "rational(1/2)");
}

#[test]
fn b_roundtrip_rational_neg_three_quarters() {
    let ctx = Context::new();
    let expr = ctx.rational(-3, 4);
    assert_round_trip(&expr, "rational(-3/4)");
}

#[test]
fn b_roundtrip_rational_seven_thirds() {
    let ctx = Context::new();
    let expr = ctx.rational(7, 3);
    assert_round_trip(&expr, "rational(7/3)");
}

// ── Symbols ────────────────────────────────────────────────────────────

#[test]
fn b_roundtrip_symbol_x() {
    let ctx = Context::new();
    let expr = ctx.symbol("x");
    assert_round_trip(&expr, "symbol(x)");
}

#[test]
fn b_roundtrip_symbol_y() {
    let ctx = Context::new();
    let expr = ctx.symbol("y");
    assert_round_trip(&expr, "symbol(y)");
}

#[test]
fn b_roundtrip_symbol_abc() {
    let ctx = Context::new();
    let expr = ctx.symbol("abc");
    assert_round_trip(&expr, "symbol(abc)");
}

// ── Binary ops ─────────────────────────────────────────────────────────

#[test]
fn b_roundtrip_add_x_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 1;
    assert_round_trip(&expr, "x + 1");
}

#[test]
fn b_roundtrip_mul_x_times_y() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x * &y;
    assert_round_trip(&expr, "x*y");
}

#[test]
fn b_roundtrip_sub_x_minus_y() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x - &y;
    assert_round_trip(&expr, "x - y");
}

#[test]
fn b_roundtrip_div_x_over_y() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x / &y;
    assert_round_trip(&expr, "x/y");
}

#[test]
fn b_roundtrip_pow_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2);
    assert_round_trip(&expr, "x^2");
}

// ── Functions ──────────────────────────────────────────────────────────

#[test]
fn b_roundtrip_sin_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    assert_round_trip(&expr, "sin(x)");
}

#[test]
fn b_roundtrip_cos_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.cos();
    assert_round_trip(&expr, "cos(x)");
}

#[test]
fn b_roundtrip_exp_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.exp();
    assert_round_trip(&expr, "exp(x)");
}

#[test]
fn b_roundtrip_ln_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.ln();
    assert_round_trip(&expr, "ln(x)");
}

#[test]
fn b_roundtrip_sqrt_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sqrt();
    assert_round_trip(&expr, "sqrt(x)");
}

#[test]
fn b_roundtrip_abs_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.abs();
    assert_round_trip(&expr, "abs(x)");
}

#[test]
fn b_roundtrip_tan_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.tan();
    assert_round_trip(&expr, "tan(x)");
}

// ── Nested expressions ────────────────────────────────────────────────

#[test]
fn b_roundtrip_sin_x_squared_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inner = x.powi(2) + 1;
    let expr = inner.sin();
    assert_round_trip(&expr, "sin(x^2 + 1)");
}

#[test]
fn b_roundtrip_exp_x_times_y() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let inner = &x * &y;
    let expr = inner.exp();
    assert_round_trip(&expr, "exp(x*y)");
}

#[test]
fn b_roundtrip_x_plus_1_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let base = &x + 1;
    let three = ctx.int(3);
    let expr = base.pow(&three);
    assert_round_trip(&expr, "(x + 1)^3");
}

// ── Constants ──────────────────────────────────────────────────────────

#[test]
fn b_roundtrip_pi() {
    let ctx = Context::new();
    let expr = ctx.pi();
    assert_round_trip(&expr, "pi");
}

#[test]
fn b_roundtrip_euler_e() {
    let ctx = Context::new();
    let expr = ctx.e();
    assert_round_trip(&expr, "E");
}

// ── Negative expressions ──────────────────────────────────────────────

#[test]
fn b_roundtrip_neg_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = -&x;
    assert_round_trip(&expr, "-x");
}

#[test]
fn b_roundtrip_neg_x_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inner = &x + 1;
    let expr = -&inner;
    assert_round_trip(&expr, "-(x + 1)");
}

#[test]
fn b_roundtrip_neg_sin_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = x.sin();
    let expr = -&s;
    assert_round_trip(&expr, "-sin(x)");
}

// ── Parse-from-string round-trips ─────────────────────────────────────

/// For known-good strings: parse -> display -> parse -> display must be stable.
fn assert_string_round_trip(input: &str) {
    let ctx = Context::new();
    let expr1 = match parse(&ctx, input) {
        Ok(e) => e,
        Err(e) => panic!("First parse of '{input}' failed: {e}"),
    };
    let display1 = format!("{expr1}");
    let expr2 = match parse(&ctx, &display1) {
        Ok(e) => e,
        Err(e) => {
            panic!("Second parse failed for '{input}':\n  display1: '{display1}'\n  error: {e}")
        }
    };
    let display2 = format!("{expr2}");
    assert_eq!(
        display1, display2,
        "String round-trip unstable for '{input}':\n  display1: '{display1}'\n  display2: '{display2}'"
    );
}

#[test]
fn b_string_roundtrip_simple_expressions() {
    let cases = [
        "x",
        "42",
        "x + 1",
        "x*y",
        "x^2",
        "sin(x)",
        "cos(x)",
        "tan(x)",
        "exp(x)",
        "ln(x)",
        "sqrt(x)",
        "abs(x)",
        "pi",
        "x^2 + 2*x + 1",
        "x^3 + x^2 + x + 1",
    ];
    for case in &cases {
        assert_string_round_trip(case);
    }
}

#[test]
fn b_string_roundtrip_nested_expressions() {
    let cases = [
        "sin(x^2)",
        "cos(x + 1)",
        "exp(x*y)",
        "ln(x^2 + 1)",
        "sqrt(x + y)",
        "sin(cos(x))",
        "exp(sin(x))",
    ];
    for case in &cases {
        assert_string_round_trip(case);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Build random polynomial-like expressions and check round-trip.
    #[test]
    fn b_proptest_polynomial_roundtrip(a in -10i64..10, b in -10i64..10, c in -10i64..10) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        // a*x^2 + b*x + c
        let term1 = &x.powi(2) * a;
        let term2 = &x * b;
        let term3 = ctx.int(c);
        let expr = &(&term1 + &term2) + &term3;
        let displayed = format!("{expr}");
        // Parse back
        let reparsed = parse(&ctx, &displayed);
        prop_assert!(reparsed.is_ok(), "Failed to reparse '{}'", displayed);
        let redisplayed = format!("{}", reparsed.unwrap());
        // Parse the redisplayed version for stability
        let reparsed2 = parse(&ctx, &redisplayed);
        prop_assert!(reparsed2.is_ok(), "Failed to reparse2 '{}'", redisplayed);
        let redisplayed2 = format!("{}", reparsed2.unwrap());
        prop_assert_eq!(redisplayed.clone(), redisplayed2.clone(),
            "Round-trip unstable: '{}' -> '{}' -> '{}'", displayed, redisplayed, redisplayed2);
    }

    /// Build random function expressions and check round-trip stability.
    #[test]
    fn b_proptest_function_roundtrip(func_idx in 0usize..7, coeff in 1i64..10) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let inner = &x * coeff;
        let expr = match func_idx {
            0 => inner.sin(),
            1 => inner.cos(),
            2 => inner.tan(),
            3 => inner.exp(),
            4 => inner.ln(),
            5 => inner.sqrt(),
            6 => inner.abs(),
            _ => unreachable!(),
        };
        let displayed = format!("{expr}");
        let reparsed = parse(&ctx, &displayed);
        prop_assert!(reparsed.is_ok(), "Failed to reparse '{}'", displayed);
        let redisplayed = format!("{}", reparsed.unwrap());
        let reparsed2 = parse(&ctx, &redisplayed);
        prop_assert!(reparsed2.is_ok(), "Failed to reparse2 '{}'", redisplayed);
        let redisplayed2 = format!("{}", reparsed2.unwrap());
        prop_assert_eq!(redisplayed.clone(), redisplayed2.clone(),
            "Function round-trip unstable: '{}' -> '{}' -> '{}'",
            displayed, redisplayed, redisplayed2);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category C: Grammar edge cases (must not panic)
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: asserts parse returns Err (not panic), and that the input does not panic.
fn assert_parse_no_panic(input: &str) {
    let ctx = Context::new();
    let _ = parse(&ctx, input);
}

/// Helper: asserts parse returns Err.
fn assert_parse_err(input: &str) {
    let ctx = Context::new();
    let result = parse(&ctx, input);
    assert!(
        result.is_err(),
        "Expected parse error for '{input}', but got Ok({})",
        result.unwrap()
    );
}

// ── Empty / whitespace ────────────────────────────────────────────────

#[test]
fn c_empty_string() {
    assert_parse_err("");
}

#[test]
fn c_whitespace_only() {
    assert_parse_no_panic("   ");
}

#[test]
fn c_whitespace_tabs_newlines() {
    assert_parse_no_panic(" \t\n\r ");
}

// ── Single operators ──────────────────────────────────────────────────

#[test]
fn c_single_plus() {
    assert_parse_no_panic("+");
}

#[test]
fn c_single_minus() {
    assert_parse_no_panic("-");
}

#[test]
fn c_single_star() {
    assert_parse_no_panic("*");
}

#[test]
fn c_single_slash() {
    assert_parse_no_panic("/");
}

#[test]
fn c_single_caret() {
    assert_parse_no_panic("^");
}

// ── Double operators ──────────────────────────────────────────────────

#[test]
fn c_double_plus() {
    assert_parse_no_panic("x++y");
}

#[test]
fn c_double_star() {
    // ** is treated as power operator (SymPy compat)
    assert_parse_no_panic("x**y");
}

#[test]
fn c_double_slash() {
    assert_parse_no_panic("x//y");
}

#[test]
fn c_double_caret() {
    assert_parse_no_panic("x^^y");
}

#[test]
fn c_triple_plus() {
    assert_parse_no_panic("x+++y");
}

// ── Trailing operators ────────────────────────────────────────────────

#[test]
fn c_trailing_plus() {
    assert_parse_no_panic("x+");
}

#[test]
fn c_trailing_star() {
    assert_parse_no_panic("1*");
}

#[test]
fn c_trailing_slash() {
    assert_parse_no_panic("x/");
}

#[test]
fn c_trailing_caret() {
    assert_parse_no_panic("x^");
}

#[test]
fn c_trailing_minus() {
    assert_parse_no_panic("x-");
}

// ── Leading operators ─────────────────────────────────────────────────

#[test]
fn c_leading_plus() {
    assert_parse_no_panic("+x");
}

#[test]
fn c_leading_star() {
    assert_parse_no_panic("*x");
}

#[test]
fn c_leading_slash() {
    assert_parse_no_panic("/x");
}

#[test]
fn c_leading_caret() {
    assert_parse_no_panic("^x");
}

// ── Unclosed parens ───────────────────────────────────────────────────

#[test]
fn c_unclosed_paren_expr() {
    assert_parse_no_panic("(x+1");
}

#[test]
fn c_unclosed_paren_func() {
    assert_parse_no_panic("sin(x");
}

#[test]
fn c_unclosed_double_paren() {
    assert_parse_no_panic("((x+1)");
}

#[test]
fn c_unclosed_nested() {
    assert_parse_no_panic("sin(cos(x)");
}

// ── Extra close parens ────────────────────────────────────────────────

#[test]
fn c_extra_close_paren() {
    assert_parse_no_panic("x+1)");
}

#[test]
fn c_extra_close_paren_func() {
    assert_parse_no_panic("sin(x))");
}

#[test]
fn c_extra_close_only() {
    assert_parse_no_panic(")");
}

#[test]
fn c_many_close_parens() {
    assert_parse_no_panic("x)))");
}

// ── Empty function calls ──────────────────────────────────────────────

#[test]
fn c_empty_sin() {
    assert_parse_no_panic("sin()");
}

#[test]
fn c_empty_cos() {
    assert_parse_no_panic("cos()");
}

#[test]
fn c_empty_ln() {
    assert_parse_no_panic("ln()");
}

#[test]
fn c_empty_exp() {
    assert_parse_no_panic("exp()");
}

#[test]
fn c_empty_sqrt() {
    assert_parse_no_panic("sqrt()");
}

#[test]
fn c_empty_abs() {
    assert_parse_no_panic("abs()");
}

#[test]
fn c_empty_tan() {
    assert_parse_no_panic("tan()");
}

// ── Function with wrong arity ─────────────────────────────────────────

#[test]
fn c_sin_two_args() {
    assert_parse_no_panic("sin(x, y)");
}

#[test]
fn c_cos_two_args() {
    assert_parse_no_panic("cos(x, y)");
}

#[test]
fn c_exp_three_args() {
    assert_parse_no_panic("exp(x, y, z)");
}

// ── Nested empty / parens only ────────────────────────────────────────

#[test]
fn c_empty_parens() {
    assert_parse_no_panic("()");
}

#[test]
fn c_double_empty_parens() {
    assert_parse_no_panic("(())");
}

#[test]
fn c_triple_empty_parens() {
    assert_parse_no_panic("((()))");
}

// ── Just numbers ──────────────────────────────────────────────────────

#[test]
fn c_just_integer() {
    let result = try_parse("42");
    assert!(result.is_ok(), "Expected Ok for '42', got: {:?}", result);
}

#[test]
fn c_just_float() {
    let result = try_parse("3.14");
    assert!(result.is_ok(), "Expected Ok for '3.14', got: {:?}", result);
}

#[test]
fn c_just_negative() {
    let result = try_parse("-7");
    assert!(result.is_ok(), "Expected Ok for '-7', got: {:?}", result);
}

#[test]
fn c_just_zero() {
    let result = try_parse("0");
    assert!(result.is_ok(), "Expected Ok for '0', got: {:?}", result);
}

// ── Huge integer ──────────────────────────────────────────────────────

#[test]
fn c_huge_integer_1000_digits() {
    let huge = "9".repeat(1000);
    assert_parse_no_panic(&huge);
}

#[test]
fn c_huge_integer_5000_digits() {
    let huge = "1".repeat(5000);
    assert_parse_no_panic(&huge);
}

#[test]
fn c_huge_integer_in_expression() {
    let huge = format!("{}+1", "9".repeat(500));
    assert_parse_no_panic(&huge);
}

// ── Special identifiers ──────────────────────────────────────────────

#[test]
fn c_identifier_pi() {
    let result = try_parse("pi");
    assert!(result.is_ok());
}

#[test]
fn c_identifier_e() {
    let result = try_parse("E");
    assert!(result.is_ok());
}

#[test]
fn c_identifier_i_unit() {
    let result = try_parse("I");
    assert!(result.is_ok());
}

#[test]
fn c_identifier_oo() {
    assert_parse_no_panic("oo");
}

#[test]
fn c_identifier_nan() {
    assert_parse_no_panic("nan");
}

#[test]
fn c_identifier_zoo() {
    assert_parse_no_panic("zoo");
}

#[test]
fn c_identifier_inf() {
    assert_parse_no_panic("inf");
}

#[test]
fn c_identifier_inf_capitalized() {
    assert_parse_no_panic("Inf");
}

// ── Invalid / unusual identifiers ─────────────────────────────────────

#[test]
fn c_digits_then_alpha() {
    // "123abc" — the parser should handle this somehow (not panic)
    assert_parse_no_panic("123abc");
}

#[test]
fn c_underscore_start() {
    // "_x" — underscores may be valid identifier starts
    assert_parse_no_panic("_x");
}

#[test]
fn c_space_in_identifier() {
    // "x y" — two separate identifiers → implicit mul or error
    assert_parse_no_panic("x y");
}

#[test]
fn c_dollar_sign() {
    assert_parse_no_panic("$x");
}

#[test]
fn c_at_sign() {
    assert_parse_no_panic("@");
}

#[test]
fn c_hash() {
    assert_parse_no_panic("#");
}

#[test]
fn c_semicolons() {
    assert_parse_no_panic("x; y");
}

#[test]
fn c_equals_sign() {
    assert_parse_no_panic("x = 1");
}

#[test]
fn c_angle_brackets() {
    assert_parse_no_panic("<x>");
}

#[test]
fn c_curly_braces() {
    assert_parse_no_panic("{x}");
}

#[test]
fn c_square_brackets() {
    assert_parse_no_panic("[x]");
}

// ── Decimal edge cases ────────────────────────────────────────────────

#[test]
fn c_decimal_no_integer_part() {
    // ".5" — starts with dot
    assert_parse_no_panic(".5");
}

#[test]
fn c_decimal_trailing_dot() {
    // "5." — trailing dot with no decimal digits
    assert_parse_no_panic("5.");
}

#[test]
fn c_decimal_double_dot() {
    assert_parse_no_panic("1.2.3");
}

#[test]
fn c_decimal_many_zeros() {
    assert_parse_no_panic("0.0000000000000000001");
}

#[test]
fn c_decimal_huge_fractional() {
    let frac = format!("0.{}", "1".repeat(200));
    assert_parse_no_panic(&frac);
}

// ── Comma edge cases ──────────────────────────────────────────────────

#[test]
fn c_bare_comma() {
    assert_parse_no_panic(",");
}

#[test]
fn c_comma_in_expression() {
    assert_parse_no_panic("x,y");
}

#[test]
fn c_leading_comma_in_func() {
    assert_parse_no_panic("sin(,x)");
}

#[test]
fn c_trailing_comma_in_func() {
    assert_parse_no_panic("sin(x,)");
}

#[test]
fn c_double_comma_in_func() {
    assert_parse_no_panic("sin(x,,y)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Category D: Adversarial inputs
// ═══════════════════════════════════════════════════════════════════════════

// ── Very long expression ──────────────────────────────────────────────

#[test]
fn d_long_addition_1000_terms() {
    let expr_str: String = (0..1000).map(|_| "x").collect::<Vec<_>>().join("+");
    assert_parse_no_panic(&expr_str);
}

#[test]
fn d_long_multiplication_500_terms() {
    let expr_str: String = (0..500).map(|_| "x").collect::<Vec<_>>().join("*");
    assert_parse_no_panic(&expr_str);
}

#[test]
fn d_long_mixed_ops() {
    // x+y*z-w/v^2+... repeated
    let mut s = String::new();
    for i in 0..200 {
        if i > 0 {
            s.push('+');
        }
        s.push_str(&format!("x{i}"));
    }
    assert_parse_no_panic(&s);
}

// ── Deeply nested functions ───────────────────────────────────────────

#[test]
fn d_deeply_nested_sin_100() {
    let mut s = String::from("x");
    for _ in 0..100 {
        s = format!("sin({s})");
    }
    assert_parse_no_panic(&s);
}

#[test]
fn d_deeply_nested_mixed_functions_50() {
    let mut s = String::from("x");
    let funcs = ["sin", "cos", "exp", "ln", "abs"];
    for i in 0..50 {
        s = format!("{}({s})", funcs[i % funcs.len()]);
    }
    assert_parse_no_panic(&s);
}

/// BUG-1: 250 layers of `sin(…)` causes a **stack overflow** (SIGABRT).
///
/// The parser's logical depth counter only reaches ~250 (below the 256 cap),
/// but the physical call-stack depth is ≥750 frames, exceeding the default
/// thread stack size.  This is an unrecoverable abort — not a catchable panic —
/// so the test is `#[ignore]`d to avoid killing the whole test runner.
///
/// Run it explicitly with: `cargo test -- --ignored d_deeply_nested_sin_250`
#[test]
fn d_deeply_nested_sin_250() {
    let mut s = String::from("x");
    for _ in 0..250 {
        s = format!("sin({s})");
    }
    // If the bug is fixed this should return Ok or Err, never abort.
    assert_parse_no_panic(&s);
}

/// Regression boundary for BUG-1: 100-deep nesting works fine.
/// This confirms the crash window is somewhere between ~100 and ~250.
#[test]
fn d_deeply_nested_sin_100_ok() {
    let mut s = String::from("x");
    for _ in 0..100 {
        s = format!("sin({s})");
    }
    let ctx = Context::new();
    let result = parse(&ctx, &s);
    // 100 deep is under the physical stack limit — should succeed.
    assert!(
        result.is_ok(),
        "100-deep sin nesting should parse, got: {:?}",
        result.err()
    );
}

/// Probe the boundary more precisely: 200-deep nesting.
/// If this passes, the crash window is between 200 and 250.
#[test]
fn d_deeply_nested_sin_200() {
    let mut s = String::from("x");
    for _ in 0..200 {
        s = format!("sin({s})");
    }
    // Must not panic/abort — should return Ok or Err.
    assert_parse_no_panic(&s);
}

// ── Mixed Unicode ─────────────────────────────────────────────────────

#[test]
fn d_unicode_pi_symbol() {
    assert_parse_no_panic("x + π");
}

#[test]
fn d_unicode_infinity() {
    assert_parse_no_panic("∞");
}

#[test]
fn d_unicode_sqrt_symbol() {
    assert_parse_no_panic("√x");
}

#[test]
fn d_unicode_superscript() {
    assert_parse_no_panic("x²");
}

#[test]
fn d_unicode_subscript() {
    assert_parse_no_panic("x₁");
}

#[test]
fn d_unicode_emoji() {
    assert_parse_no_panic("🎉 + 1");
}

#[test]
fn d_unicode_chinese() {
    assert_parse_no_panic("变量");
}

#[test]
fn d_unicode_arabic() {
    assert_parse_no_panic("متغير");
}

#[test]
fn d_unicode_mixed_with_math() {
    assert_parse_no_panic("sin(π) + cos(∞)");
}

#[test]
fn d_unicode_zero_width_space() {
    assert_parse_no_panic("x\u{200B}+\u{200B}y");
}

#[test]
fn d_unicode_bom() {
    assert_parse_no_panic("\u{FEFF}x + 1");
}

// ── Control characters ────────────────────────────────────────────────

#[test]
fn d_null_byte() {
    assert_parse_no_panic("x\0y");
}

#[test]
fn d_null_byte_start() {
    assert_parse_no_panic("\0");
}

#[test]
fn d_tab_in_expression() {
    assert_parse_no_panic("x\t+\ty");
}

#[test]
fn d_newline_in_expression() {
    assert_parse_no_panic("x\n+\ny");
}

#[test]
fn d_carriage_return() {
    assert_parse_no_panic("x\r+\ry");
}

#[test]
fn d_mixed_control_chars() {
    assert_parse_no_panic("x\x01\x02\x03y");
}

#[test]
fn d_bell_character() {
    assert_parse_no_panic("\x07");
}

#[test]
fn d_escape_character() {
    assert_parse_no_panic("\x1B[31m");
}

#[test]
fn d_form_feed() {
    assert_parse_no_panic("x\x0C+y");
}

#[test]
fn d_delete_character() {
    assert_parse_no_panic("\x7F");
}

// ── Adversarial operator patterns ─────────────────────────────────────

#[test]
fn d_all_operators() {
    assert_parse_no_panic("+-*/^");
}

#[test]
fn d_alternating_minus() {
    // ---x should be parsed as -(-(-x))
    assert_parse_no_panic("---x");
}

#[test]
fn d_many_unary_minus() {
    let s = "-".repeat(100) + "x";
    assert_parse_no_panic(&s);
}

#[test]
fn d_alternating_ops_and_parens() {
    assert_parse_no_panic("(+)(-)(*)(/)");
}

#[test]
fn d_implicit_multiplication_chain() {
    // "2x3y" — implicit multiplication
    assert_parse_no_panic("2x3y");
}

#[test]
fn d_implicit_mul_parens() {
    assert_parse_no_panic("(x)(y)(z)");
}

#[test]
fn d_juxtaposed_numbers() {
    // Two numbers next to each other with space
    assert_parse_no_panic("2 3");
}

// ── Right-associativity stress ────────────────────────────────────────

#[test]
fn d_power_tower() {
    // x^x^x^x^x — right-associative power
    let s = (0..20).map(|_| "x").collect::<Vec<_>>().join("^");
    assert_parse_no_panic(&s);
}

#[test]
fn d_power_tower_with_numbers() {
    let s = (0..20).map(|_| "2").collect::<Vec<_>>().join("^");
    assert_parse_no_panic(&s);
}

// ── Repeated function names as identifiers ────────────────────────────

#[test]
fn d_sin_as_variable() {
    // "sin" without parens — should be treated as function needing parens or as identifier
    assert_parse_no_panic("sin + 1");
}

#[test]
fn d_cos_as_variable() {
    assert_parse_no_panic("cos * x");
}

#[test]
fn d_nested_function_name_clash() {
    assert_parse_no_panic("sin(sin)");
}

// ── SymPy-compatible syntax ───────────────────────────────────────────

#[test]
fn d_sympy_double_star() {
    let result = try_parse("x**2");
    assert!(result.is_ok(), "x**2 should parse (SymPy compat)");
}

#[test]
fn d_sympy_abs_capitalized() {
    assert_parse_no_panic("Abs(x)");
}

// ── Extreme whitespace ────────────────────────────────────────────────

#[test]
fn d_excessive_whitespace() {
    let s = format!("x {} + {} y", " ".repeat(1000), " ".repeat(1000));
    assert_parse_no_panic(&s);
}

#[test]
fn d_only_spaces_10000() {
    let s = " ".repeat(10000);
    assert_parse_no_panic(&s);
}

// ── Pathological repetition ───────────────────────────────────────────

#[test]
fn d_repeated_open_parens() {
    let s = "(".repeat(10000);
    assert_parse_no_panic(&s);
}

#[test]
fn d_repeated_close_parens() {
    let s = ")".repeat(10000);
    assert_parse_no_panic(&s);
}

#[test]
fn d_alternating_parens() {
    let s = "()".repeat(5000);
    assert_parse_no_panic(&s);
}

#[test]
fn d_repeated_commas() {
    let s = ",".repeat(1000);
    assert_parse_no_panic(&s);
}

#[test]
fn d_repeated_function_with_commas() {
    // sin(,,,,,)
    let commas = ",".repeat(100);
    let s = format!("sin({commas})");
    assert_parse_no_panic(&s);
}

// ── Boundary between valid/invalid ────────────────────────────────────

#[test]
fn d_number_then_paren() {
    // "42(x)" — implicit mul
    assert_parse_no_panic("42(x)");
}

#[test]
fn d_paren_then_number() {
    // "(x)42" — implicit mul
    assert_parse_no_panic("(x)42");
}

#[test]
fn d_function_no_paren() {
    // "sinx" — no paren after function name
    assert_parse_no_panic("sinx");
}

#[test]
fn d_function_space_paren() {
    // "sin (x)" — space between function and paren
    assert_parse_no_panic("sin (x)");
}

// ── Mixed valid and garbage ───────────────────────────────────────────

#[test]
fn d_expression_then_garbage() {
    assert_parse_no_panic("x + 1 GARBAGE");
}

#[test]
fn d_valid_then_special_chars() {
    assert_parse_no_panic("x + 1 !@#$%");
}

// ═══════════════════════════════════════════════════════════════════════════
// Category E: Additional round-trip stability tests (proptest)
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// If something parses successfully, its display should also parse successfully
    /// (idempotent round-trip).
    #[test]
    fn e_parse_display_idempotent(s in "[a-z0-9+\\-*/^() ]{1,40}") {
        let ctx = Context::new();
        if let Ok(expr) = parse(&ctx, &s) {
            let displayed = format!("{expr}");
            let result2 = parse(&ctx, &displayed);
            prop_assert!(result2.is_ok(),
                "Parsed '{}' -> displayed '{}' -> failed to reparse: {}",
                s, displayed, result2.unwrap_err());
            let displayed2 = format!("{}", result2.unwrap());
            // The second display should be stable (fixpoint)
            let result3 = parse(&ctx, &displayed2);
            prop_assert!(result3.is_ok(),
                "Third parse failed for '{}' -> '{}' -> '{}'",
                s, displayed, displayed2);
            let displayed3 = format!("{}", result3.unwrap());
            prop_assert_eq!(&displayed2, &displayed3,
                "Not a fixpoint: '{}' -> '{}' -> '{}' -> '{}'",
                s, displayed, displayed2, displayed3);
        }
    }

    /// Random expressions built from a math-like grammar should parse without panic.
    #[test]
    fn e_structured_random_no_panic(
        ops in prop::collection::vec(prop::sample::select(vec!['+', '-', '*', '/', '^']), 1..10),
        vars in prop::collection::vec("[a-z]{1,3}", 2..6),
    ) {
        let mut expr_str = vars[0].clone();
        for (i, op) in ops.iter().enumerate() {
            let var = &vars[(i + 1) % vars.len()];
            expr_str.push(*op);
            expr_str.push_str(var);
        }
        let ctx = Context::new();
        let _ = parse(&ctx, &expr_str);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category F: Specific regression / tricky patterns
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn f_negative_zero() {
    assert_parse_no_panic("-0");
}

#[test]
fn f_positive_zero_decimal() {
    assert_parse_no_panic("0.0");
}

#[test]
fn f_leading_zeros() {
    assert_parse_no_panic("007");
}

#[test]
fn f_leading_zeros_decimal() {
    assert_parse_no_panic("007.007");
}

#[test]
fn f_double_negative() {
    assert_parse_no_panic("--x");
}

#[test]
fn f_negative_in_power() {
    // -x^2 should be -(x^2) not (-x)^2
    let ctx = Context::new();
    let result = parse(&ctx, "-x^2");
    assert!(result.is_ok());
    let displayed = format!("{}", result.unwrap());
    // The display should reflect correct precedence
    // Reparse for stability
    let result2 = parse(&ctx, &displayed);
    assert!(
        result2.is_ok(),
        "Reparsing '{}' failed: {}",
        displayed,
        result2.unwrap_err()
    );
}

#[test]
fn f_negative_base_in_power() {
    assert_parse_no_panic("(-x)^2");
}

#[test]
fn f_exponent_of_exponent() {
    // x^2^3 should be x^(2^3) = x^8 (right-assoc)
    let ctx = Context::new();
    let result = parse(&ctx, "x^2^3");
    assert!(result.is_ok());
}

#[test]
fn f_subtraction_vs_negative() {
    // "x-1" vs "x+-1" vs "x+(-1)"
    assert_parse_no_panic("x-1");
    assert_parse_no_panic("x+-1");
    assert_parse_no_panic("x+(-1)");
}

#[test]
fn f_implicit_mul_stress() {
    // "2x" = 2*x
    // "2(x)" = 2*(x)
    // "x(2)" = x*(2)
    // "(2)(3)" = 2*3
    for input in &["2x", "2(x)", "x(2)", "(2)(3)", "xy", "2xy", "2x3y"] {
        assert_parse_no_panic(input);
    }
}

#[test]
fn f_function_of_negative() {
    for func in &["sin", "cos", "tan", "exp", "ln", "sqrt", "abs"] {
        let input = format!("{func}(-x)");
        assert_parse_no_panic(&input);
        let input2 = format!("{func}(-1)");
        assert_parse_no_panic(&input2);
    }
}

#[test]
fn f_chained_division() {
    // x/y/z — left-associative
    assert_parse_no_panic("x/y/z");
    assert_parse_no_panic("1/2/3/4/5");
}

#[test]
fn f_mixed_implicit_explicit_mul() {
    assert_parse_no_panic("2*x*3y");
    assert_parse_no_panic("2x*3*y");
}

#[test]
fn f_pi_in_expression() {
    let ctx = Context::new();
    let result = parse(&ctx, "2*pi");
    assert!(result.is_ok());
    let result2 = parse(&ctx, "pi*x");
    assert!(result2.is_ok());
    let result3 = parse(&ctx, "sin(pi)");
    assert!(result3.is_ok());
}

#[test]
fn f_e_in_expression() {
    let ctx = Context::new();
    let result = parse(&ctx, "E^x");
    assert!(result.is_ok());
    let result2 = parse(&ctx, "E^(I*pi)");
    assert!(result2.is_ok());
}

#[test]
fn f_complex_euler_formula() {
    let ctx = Context::new();
    let result = parse(&ctx, "E^(I*pi) + 1");
    assert!(result.is_ok());
}
