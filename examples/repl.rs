//! Simple REPL for symplex — demonstrates runtime expression parsing.
//!
//! Run with: `cargo run --example repl`
//!
//! Type mathematical expressions and see them evaluated.
//!
//! Example session:
//! ```text
//! > x^2 + 2*x + 1
//!   = x^2 + 2*x + 1
//! > :diff x^3 x
//!   = 3*x^2
//! > :solve x^2 - 4 x
//!   x = 2
//!   x = -2
//! > :latex sin(x)^2
//!   \sin^{2}\left(x\right)
//! > :factor x^2 - 1
//!   = (x - 1)*(x + 1)
//! > :quit
//! ```

use std::io::{self, BufRead, Write};

fn main() {
    let ctx = symplex::prelude::Context::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    println!("symplex REPL — type expressions, :help for commands, :quit to exit");
    println!();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break; // EOF
        }
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        if line == ":quit" || line == ":q" {
            break;
        }

        if line == ":help" || line == ":h" {
            println!("Commands:");
            println!("  <expr>              — parse and display an expression");
            println!("  :expand <expr>      — expand the expression");
            println!("  :eval <expr>        — evaluate special values");
            println!("  :simplify <expr>    — full simplify (fixpoint)");
            println!("  :diff <expr> [var]  — differentiate (default var: x)");
            println!("  :integrate <expr>   — integrate w.r.t. x");
            println!("  :solve <expr> <var> — solve an equation for <var>");
            println!("  :latex <expr>       — render as LaTeX");
            println!("  :factor <expr>      — factor a polynomial");
            println!("  :json <expr>        — show JSON serialization");
            println!("  :quit / :q          — exit");
            println!();
            continue;
        }

        let (command, input) = if line.starts_with(':') {
            let mut parts = line.splitn(2, ' ');
            let cmd = parts.next().unwrap_or("");
            let rest = parts.next().unwrap_or("").trim();
            (cmd, rest)
        } else {
            ("", line)
        };

        match command {
            ":solve" => {
                // Split input into "<expr> <var>" — the last whitespace-delimited
                // token is the variable name, everything before it is the expression.
                let (expr_str, var_name) = match rsplit_last_token(input) {
                    Some(pair) => pair,
                    None => {
                        println!("  usage: :solve <expr> <var>");
                        continue;
                    }
                };
                let expr = match symplex::parse::parse(&ctx, expr_str) {
                    Ok(e) => e,
                    Err(e) => {
                        println!("  error parsing expression: {e}");
                        continue;
                    }
                };
                let var = ctx.symbol(var_name);
                let roots = expr.solve_or_empty(&var);
                if roots.is_empty() {
                    println!("  no solutions found");
                } else {
                    for root in &roots {
                        println!("  {var_name} = {root}");
                    }
                }
            }
            ":latex" => {
                let expr = match symplex::parse::parse(&ctx, input) {
                    Ok(e) => e,
                    Err(e) => {
                        println!("  error: {e}");
                        continue;
                    }
                };
                println!("  {}", expr.to_latex());
            }
            ":factor" => {
                let expr = match symplex::parse::parse(&ctx, input) {
                    Ok(e) => e,
                    Err(e) => {
                        println!("  error: {e}");
                        continue;
                    }
                };
                // Use the first free symbol as the factoring variable,
                // or fall back to "x".
                let symbols = expr.free_symbols();
                let var = if symbols.is_empty() {
                    ctx.symbol("x")
                } else {
                    symbols[0].clone()
                };
                println!("  = {}", expr.factor(&var));
            }
            ":diff" => {
                // Accept ":diff <expr>" (defaults to x) or ":diff <expr> <var>"
                // Heuristic: try parsing the whole input as one expression first.
                // If that works and the input has no trailing bare identifier after
                // the main expression, use x. Otherwise split the last token as var.
                let (expr, var) = match parse_expr_and_optional_var(&ctx, input) {
                    Ok(pair) => pair,
                    Err(e) => {
                        println!("  error: {e}");
                        continue;
                    }
                };
                println!("  = {}", expr.diff(&var));
            }
            _ => {
                // Commands that take the whole input as one expression
                let expr = match symplex::parse::parse(&ctx, input) {
                    Ok(e) => e,
                    Err(e) => {
                        println!("  error: {e}");
                        continue;
                    }
                };

                match command {
                    "" => {
                        println!("  = {expr}");
                    }
                    ":expand" => {
                        println!("  = {}", expr.expand());
                    }
                    ":eval" => {
                        println!("  = {}", expr.eval());
                    }
                    ":simplify" => {
                        println!("  = {}", expr.full_simplify());
                    }
                    ":integrate" => {
                        let x = ctx.symbol("x");
                        println!("  = {}", expr.integrate(&x));
                    }
                    ":json" => {
                        println!("  {}", expr.to_json_pretty().unwrap());
                    }
                    _ => {
                        println!("  unknown command: {command}. Type :help for help.");
                    }
                }
            }
        }
    }

    println!("goodbye");
}

/// Split `input` into `(everything_before_last_token, last_token)`.
///
/// Returns `None` if the input has fewer than two tokens.
fn rsplit_last_token(input: &str) -> Option<(&str, &str)> {
    let trimmed = input.trim_end();
    let last_space = trimmed.rfind(|c: char| c.is_whitespace())?;
    let expr_part = trimmed[..last_space].trim_end();
    let var_part = trimmed[last_space..].trim();
    if expr_part.is_empty() || var_part.is_empty() {
        return None;
    }
    Some((expr_part, var_part))
}

/// Parse an expression and an optional trailing variable name.
///
/// If the last whitespace-separated token is a single identifier (and
/// stripping it still leaves a parseable expression), use it as the
/// variable. Otherwise parse the whole string as the expression and
/// default the variable to `x`.
fn parse_expr_and_optional_var(
    ctx: &symplex::prelude::Context,
    input: &str,
) -> Result<(symplex::prelude::Ex, symplex::prelude::Ex), String> {
    // Try splitting off the last token as a variable name
    if let Some((expr_str, var_candidate)) = rsplit_last_token(input) {
        // A variable name should be a simple identifier (no operators, parens, etc.)
        let looks_like_var = !var_candidate.is_empty()
            && var_candidate
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_');

        if looks_like_var && let Ok(expr) = symplex::parse::parse(ctx, expr_str) {
            let var = ctx.symbol(var_candidate);
            return Ok((expr, var));
        }
    }

    // Fallback: parse the whole input, default variable to x
    match symplex::parse::parse(ctx, input) {
        Ok(expr) => {
            let x = ctx.symbol("x");
            Ok((expr, x))
        }
        Err(e) => Err(format!("{e}")),
    }
}
