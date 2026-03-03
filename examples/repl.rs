//! Simple REPL for symplex — demonstrates runtime expression parsing.
//!
//! Run with: `cargo run --example repl`
//!
//! Type mathematical expressions and see them evaluated:
//!   > x^2 + 2*x + 1
//!   = 1 + x^2 + 2*x
//!   > diff(x^3, x)
//!   (note: diff is not a parsed function — use the API instead)
//!
//! Special commands:
//!   :quit or :q    — exit the REPL
//!   :json <expr>   — show JSON serialization
//!   :expand <expr> — expand the expression
//!   :eval <expr>   — evaluate special values
//!   :simplify <expr> — full simplify
//!   :diff <expr>   — differentiate w.r.t. x

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
            println!("  <expr>           — parse and display an expression");
            println!("  :expand <expr>   — expand the expression");
            println!("  :eval <expr>     — evaluate special values");
            println!("  :simplify <expr> — full simplify (fixpoint)");
            println!("  :diff <expr>     — differentiate w.r.t. x");
            println!("  :integrate <expr> — integrate w.r.t. x");
            println!("  :json <expr>     — show JSON serialization");
            println!("  :quit / :q       — exit");
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
            ":diff" => {
                let x = ctx.symbol("x");
                println!("  = {}", expr.diff(&x));
            }
            ":integrate" => {
                let x = ctx.symbol("x");
                println!("  = {}", expr.integrate(&x));
            }
            ":json" => {
                println!("  {}", expr.to_json_pretty());
            }
            _ => {
                println!("  unknown command: {command}. Type :help for help.");
            }
        }
    }

    println!("goodbye");
}
