//! Compile symbolic expressions to callable numerical functions.
//!
//! [`compile`] lowers one or more expressions into a compact stack-VM
//! program ([`Program`]) after a shared common-subexpression-elimination
//! pass.  The program is wrapped in [`CompiledFn`] (scalar) or
//! [`CompiledFnVec`] (vector-valued) handles that are `Clone + Send + Sync`
//! and evaluate without touching the arena or taking any locks.
//!
//! Every numerically evaluable [`ExprNode`] is supported, including the
//! special functions (Γ, ln Γ, ψ, erf/erfc, Lambert W, Beta, factorials,
//! binomials), Bessel functions and orthogonal polynomials with integer
//! order, integer sequences, piecewise expressions and boolean/relational
//! nodes (represented as `0.0`/`1.0`).  The special-function algorithms live
//! in [`numeric_rt`](crate::output::codegen::numeric_rt) and are shared with
//! the Rust code generator, so `compile()` and `to_rust_fn()` agree bit for
//! bit on those routines.
//!
//! The compiler never recurses over the expression tree — it uses an
//! explicit work stack — so arbitrarily deep expressions are safe.

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode, SymbolId};
use crate::output::codegen::numeric_rt as rt;
use num_traits::{One, ToPrimitive, Zero};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

/// Boxed, thread-safe closure type behind [`CompiledFn`]'s `Deref`.
type SharedFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

// ═══════════════════════════════════════════════════════════════════════════
// Public handle types
// ═══════════════════════════════════════════════════════════════════════════

/// A compiled scalar function `f(x₁, …, xₙ) → f64`.
///
/// Produced by [`Ex::compile`](crate::api::expr::Expr::compile).  The handle
/// is cheap to clone (it shares the underlying program), is `Send + Sync`,
/// and can be called directly like a closure thanks to a `Deref` to
/// `dyn Fn(&[f64]) -> f64`:
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let f = (x.powi(2) + 1).compile(&["x"]).unwrap();
/// assert_eq!(f.arity(), 1);
/// assert_eq!(f(&[3.0]), 10.0);        // closure-style call
/// assert_eq!(f.call(&[3.0]), 10.0);   // explicit call
/// assert!(f.call(&[1.0, 2.0]).is_nan()); // wrong arity → NaN
/// assert!(f.try_call(&[1.0, 2.0]).is_err());
/// ```
///
/// Calling with the wrong number of arguments returns `NaN` from
/// [`call`](Self::call) (and the closure form); use
/// [`try_call`](Self::try_call) to get an error instead.
#[derive(Clone)]
pub struct CompiledFn {
    program: Arc<Program>,
    func: SharedFn,
}

impl CompiledFn {
    fn from_program(program: Program) -> Self {
        let program = Arc::new(program);
        let p = Arc::clone(&program);
        let func: SharedFn = Arc::new(move |args: &[f64]| p.run_scalar(args));
        Self { program, func }
    }

    /// Number of arguments the function expects.
    #[must_use]
    pub fn arity(&self) -> usize {
        self.program.arity
    }

    /// Evaluate the function.  Returns `NaN` if `args.len() != self.arity()`.
    #[must_use]
    pub fn call(&self, args: &[f64]) -> f64 {
        self.program.run_scalar(args)
    }

    /// Evaluate the function, returning an error on arity mismatch.
    pub fn try_call(&self, args: &[f64]) -> Result<f64, SymplexError> {
        self.program.check_arity("CompiledFn::try_call", args)?;
        Ok(self.program.run_scalar(args))
    }

    /// Number of VM instructions in the compiled program (a rough cost measure).
    #[must_use]
    pub fn instruction_count(&self) -> usize {
        self.program.code.len()
    }
}

impl Deref for CompiledFn {
    type Target = dyn Fn(&[f64]) -> f64 + Send + Sync;
    fn deref(&self) -> &Self::Target {
        &*self.func
    }
}

impl fmt::Debug for CompiledFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledFn")
            .field("arity", &self.program.arity)
            .field("instructions", &self.program.code.len())
            .field("locals", &self.program.n_locals)
            .finish()
    }
}

/// A compiled vector-valued function `F(x₁, …, xₙ) → (f₁, …, fₖ)`.
///
/// Produced by [`Ex::compile_many`](crate::api::expr::Expr::compile_many);
/// all outputs share one common-subexpression-elimination pass, which makes
/// this the right tool for gradients, Jacobians and other expression
/// families with heavy overlap.
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let y = ctx.symbol("y");
/// let f = &x.sin() * &y;
/// let grad = Ex::compile_many(&[&f.diff(&x), &f.diff(&y)], &["x", "y"]).unwrap();
/// assert_eq!(grad.len(), 2);
/// let g = grad.call_vec(&[0.0, 2.0]);
/// assert_eq!(g, vec![2.0, 0.0]);
/// ```
#[derive(Clone)]
pub struct CompiledFnVec {
    program: Arc<Program>,
}

impl CompiledFnVec {
    /// Number of arguments the function expects.
    #[must_use]
    pub fn arity(&self) -> usize {
        self.program.arity
    }

    /// Number of output values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.program.n_outputs
    }

    /// True when the function produces no outputs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.program.n_outputs == 0
    }

    /// Evaluate into `out`.
    ///
    /// On arity mismatch, or if `out.len() != self.len()`, every element of
    /// `out` is set to `NaN` and no other work is done.
    pub fn call(&self, args: &[f64], out: &mut [f64]) {
        if args.len() != self.program.arity || out.len() != self.program.n_outputs {
            out.fill(f64::NAN);
            return;
        }
        self.program.run(args, out);
    }

    /// Evaluate into `out`, returning an error on arity or length mismatch.
    pub fn try_call(&self, args: &[f64], out: &mut [f64]) -> Result<(), SymplexError> {
        self.program.check_arity("CompiledFnVec::try_call", args)?;
        if out.len() != self.program.n_outputs {
            return Err(SymplexError::InvalidArgument {
                operation: "CompiledFnVec::try_call",
                reason: format!(
                    "output slice has length {}, expected {}",
                    out.len(),
                    self.program.n_outputs
                ),
            });
        }
        self.program.run(args, out);
        Ok(())
    }

    /// Evaluate and return the outputs as a freshly allocated vector
    /// (all `NaN` on arity mismatch).
    #[must_use]
    pub fn call_vec(&self, args: &[f64]) -> Vec<f64> {
        let mut out = vec![f64::NAN; self.program.n_outputs];
        self.call(args, &mut out);
        out
    }

    /// Number of VM instructions in the compiled program.
    #[must_use]
    pub fn instruction_count(&self) -> usize {
        self.program.code.len()
    }
}

impl fmt::Debug for CompiledFnVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledFnVec")
            .field("arity", &self.program.arity)
            .field("outputs", &self.program.n_outputs)
            .field("instructions", &self.program.code.len())
            .field("locals", &self.program.n_locals)
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry points (crate-internal)
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a single expression into a [`CompiledFn`].
///
/// Runs `eval()` (cheap constant folding) and CSE before lowering.
pub(crate) fn compile(
    arena: &mut Arena,
    expr: ExprId,
    var_names: &[&str],
) -> Result<CompiledFn, SymplexError> {
    let program = compile_program(arena, &[expr], var_names)?;
    Ok(CompiledFn::from_program(program))
}

/// Compile several expressions into one [`CompiledFnVec`] with a shared CSE pass.
pub(crate) fn compile_many(
    arena: &mut Arena,
    exprs: &[ExprId],
    var_names: &[&str],
) -> Result<CompiledFnVec, SymplexError> {
    let program = compile_program(arena, exprs, var_names)?;
    Ok(CompiledFnVec {
        program: Arc::new(program),
    })
}

/// A [`CompiledFnVec`] with no outputs (no arena required).
pub(crate) fn compile_many_empty(var_names: &[&str]) -> Result<CompiledFnVec, SymplexError> {
    check_params(var_names)?;
    Ok(CompiledFnVec {
        program: Arc::new(Program {
            code: Vec::new(),
            arity: var_names.len(),
            n_locals: 0,
            n_outputs: 0,
        }),
    })
}

fn check_params(var_names: &[&str]) -> Result<(), SymplexError> {
    for (i, a) in var_names.iter().enumerate() {
        if var_names[..i].contains(a) {
            return Err(SymplexError::InvalidArgument {
                operation: "compile",
                reason: format!("duplicate parameter name '{a}'"),
            });
        }
    }
    Ok(())
}

/// Shared lowering pipeline: eval → CSE → bytecode.
fn compile_program(
    arena: &mut Arena,
    exprs: &[ExprId],
    var_names: &[&str],
) -> Result<Program, SymplexError> {
    check_params(var_names)?;

    // Pre-pass: fold exact constants (sin(0)→0, cos(π)→−1, 4^(1/2)→2, …).
    //
    // Two families are lowered as-is instead:
    // * `Piecewise` — `eval()` currently mis-simplifies piecewise nodes whose
    //   earlier conditions are symbolic (it selects the first literal-`True`
    //   branch);
    // * orthogonal polynomials — `eval()` expands them into monomials, which
    //   is numerically much worse than the three-term recurrence used by the
    //   runtime (1e-13 vs 1e-16 absolute error for P₁₁ on [−1, 1]).
    let evaled: Vec<ExprId> = exprs
        .iter()
        .map(|&e| {
            if needs_raw_lowering(arena, e) {
                e
            } else {
                crate::transforms::eval::eval(arena, e)
            }
        })
        .collect();

    // Shared CSE across all outputs.
    let cse = crate::output::cse::cse_multi(arena, &evaled);

    let mut locals: FxHashMap<SymbolId, usize> = FxHashMap::default();
    for (i, (name_id, _)) in cse.bindings.iter().enumerate() {
        if let ExprNode::Symbol(sid) = arena.node(*name_id) {
            locals.insert(*sid, i);
        }
    }

    let mut em = Emitter::new(arena, var_names, locals);
    for (i, (_, value)) in cse.bindings.iter().enumerate() {
        em.lower(*value)?;
        em.emit(Instruction::StoreLocal(i));
    }
    for (i, &e) in cse.exprs.iter().enumerate() {
        em.lower(e)?;
        em.emit(Instruction::StoreOut(i));
    }
    let code = em.finish()?;

    Ok(Program {
        code,
        arity: var_names.len(),
        n_locals: cse.bindings.len(),
        n_outputs: exprs.len(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Program / VM
// ═══════════════════════════════════════════════════════════════════════════

/// A lowered stack-machine program.
struct Program {
    code: Vec<Instruction>,
    arity: usize,
    n_locals: usize,
    n_outputs: usize,
}

impl Program {
    fn check_arity(&self, operation: &'static str, args: &[f64]) -> Result<(), SymplexError> {
        if args.len() != self.arity {
            return Err(SymplexError::InvalidArgument {
                operation,
                reason: format!("expected {} argument(s), got {}", self.arity, args.len()),
            });
        }
        Ok(())
    }

    fn run_scalar(&self, args: &[f64]) -> f64 {
        if args.len() != self.arity {
            return f64::NAN;
        }
        let mut out = [f64::NAN];
        self.run(args, &mut out);
        out[0]
    }

    /// Execute the program.  Callers must have validated arities.
    fn run(&self, args: &[f64], out: &mut [f64]) {
        let mut stack: SmallVec<[f64; 32]> = SmallVec::new();
        let mut locals: SmallVec<[f64; 16]> = SmallVec::new();
        locals.resize(self.n_locals, 0.0);
        let code = &self.code;
        let mut pc = 0usize;

        macro_rules! pop {
            () => {
                stack.pop().unwrap_or(f64::NAN)
            };
        }
        macro_rules! un {
            ($f:expr) => {{
                let a = pop!();
                stack.push($f(a));
            }};
        }
        macro_rules! bin {
            ($f:expr) => {{
                let b = pop!();
                let a = pop!();
                stack.push($f(a, b));
            }};
        }

        while pc < code.len() {
            match code[pc] {
                Instruction::PushConst(v) => stack.push(v),
                Instruction::PushVar(i) => stack.push(args.get(i).copied().unwrap_or(f64::NAN)),
                Instruction::LoadLocal(i) => stack.push(locals.get(i).copied().unwrap_or(f64::NAN)),
                Instruction::StoreLocal(i) => {
                    let v = pop!();
                    if let Some(slot) = locals.get_mut(i) {
                        *slot = v;
                    }
                }
                Instruction::StoreOut(i) => {
                    let v = pop!();
                    if let Some(slot) = out.get_mut(i) {
                        *slot = v;
                    }
                }
                Instruction::Add => bin!(|a: f64, b: f64| a + b),
                Instruction::Mul => bin!(|a: f64, b: f64| a * b),
                Instruction::Div => bin!(|a: f64, b: f64| a / b),
                Instruction::Neg => un!(|a: f64| -a),
                Instruction::Pow => bin!(|a: f64, b: f64| a.powf(b)),
                Instruction::Powi(n) => un!(|a: f64| a.powi(n)),
                Instruction::Sqrt => un!(|a: f64| a.sqrt()),
                Instruction::Cbrt => un!(|a: f64| a.cbrt()),
                Instruction::ExpM1 => un!(|a: f64| a.exp_m1()),
                Instruction::Ln1p => un!(|a: f64| a.ln_1p()),
                Instruction::Sin => un!(|a: f64| a.sin()),
                Instruction::Cos => un!(|a: f64| a.cos()),
                Instruction::Tan => un!(|a: f64| a.tan()),
                Instruction::Exp => un!(|a: f64| a.exp()),
                Instruction::Ln => un!(|a: f64| a.ln()),
                Instruction::Abs => un!(|a: f64| a.abs()),
                Instruction::Asin => un!(|a: f64| a.asin()),
                Instruction::Acos => un!(|a: f64| a.acos()),
                Instruction::Atan => un!(|a: f64| a.atan()),
                Instruction::Sinh => un!(|a: f64| a.sinh()),
                Instruction::Cosh => un!(|a: f64| a.cosh()),
                Instruction::Tanh => un!(|a: f64| a.tanh()),
                Instruction::Asinh => un!(|a: f64| a.asinh()),
                Instruction::Acosh => un!(|a: f64| a.acosh()),
                Instruction::Atanh => un!(|a: f64| a.atanh()),
                Instruction::Sign => un!(|a: f64| if a > 0.0 {
                    1.0
                } else if a < 0.0 {
                    -1.0
                } else {
                    0.0
                }),
                Instruction::Heaviside => un!(|a: f64| if a > 0.0 {
                    1.0
                } else if a < 0.0 {
                    0.0
                } else {
                    0.5
                }),
                // Distributional: pointwise evaluation is 0 everywhere (see docs).
                Instruction::DiracDelta => un!(|a: f64| if a.is_nan() { f64::NAN } else { 0.0 }),
                Instruction::Atan2 => bin!(|y: f64, x: f64| y.atan2(x)),
                Instruction::Floor => un!(|a: f64| a.floor()),
                Instruction::Ceiling => un!(|a: f64| a.ceil()),
                Instruction::Min2 => bin!(|a: f64, b: f64| a.min(b)),
                Instruction::Max2 => bin!(|a: f64, b: f64| a.max(b)),
                Instruction::Gamma => un!(rt::gamma),
                Instruction::LogGamma => un!(rt::lgamma),
                Instruction::Digamma => un!(rt::digamma),
                Instruction::Erf => un!(rt::erf),
                Instruction::Erfc => un!(rt::erfc),
                Instruction::LambertW => un!(rt::lambert_w0),
                Instruction::Factorial => un!(rt::factorial),
                Instruction::Binomial => bin!(rt::binomial),
                Instruction::Beta => bin!(rt::beta),
                Instruction::BesselJ(n) => un!(|a: f64| rt::bessel_j(n, a)),
                Instruction::BesselY(n) => un!(|a: f64| rt::bessel_y(n, a)),
                Instruction::BesselI(n) => un!(|a: f64| rt::bessel_i(n, a)),
                Instruction::BesselK(n) => un!(|a: f64| rt::bessel_k(n, a)),
                Instruction::LegendreP(n) => un!(|a: f64| rt::legendre_p(n, a)),
                Instruction::ChebyshevT(n) => un!(|a: f64| rt::chebyshev_t(n, a)),
                Instruction::ChebyshevU(n) => un!(|a: f64| rt::chebyshev_u(n, a)),
                Instruction::HermiteH(n) => un!(|a: f64| rt::hermite_h(n, a)),
                Instruction::LaguerreL(n) => un!(|a: f64| rt::laguerre_l(n, a)),
                Instruction::Fibonacci => un!(rt::fibonacci),
                Instruction::Lucas => un!(rt::lucas),
                Instruction::Harmonic => un!(rt::harmonic),
                Instruction::Factorial2 => un!(rt::factorial2),
                Instruction::RisingFactorial => bin!(rt::rising_factorial),
                Instruction::FallingFactorial => bin!(rt::falling_factorial),
                Instruction::Gt => bin!(|a: f64, b: f64| bool_f64(a > b)),
                Instruction::Ge => bin!(|a: f64, b: f64| bool_f64(a >= b)),
                Instruction::Eq => bin!(|a: f64, b: f64| bool_f64(a == b)),
                Instruction::Ne => bin!(|a: f64, b: f64| bool_f64(a != b)),
                Instruction::And => bin!(|a: f64, b: f64| bool_f64(a != 0.0 && b != 0.0)),
                Instruction::Or => bin!(|a: f64, b: f64| bool_f64(a != 0.0 || b != 0.0)),
                Instruction::Not => un!(|a: f64| bool_f64(a == 0.0)),
                Instruction::JumpIfZero(target) => {
                    let c = pop!();
                    if c == 0.0 || c.is_nan() {
                        pc = target;
                        continue;
                    }
                }
                Instruction::Jump(target) => {
                    pc = target;
                    continue;
                }
            }
            pc += 1;
        }
    }
}

#[inline(always)]
fn bool_f64(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

/// A single stack-machine instruction.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Instruction {
    PushConst(f64),
    /// Push argument `i`.
    PushVar(usize),
    /// Push CSE temporary `i`.
    LoadLocal(usize),
    /// Pop into CSE temporary `i`.
    StoreLocal(usize),
    /// Pop into output slot `i`.
    StoreOut(usize),
    Add,
    Mul,
    /// `second / top` (produced by a peephole on `Powi(-1); Mul`).
    Div,
    Neg,
    /// `second ^ top`.
    Pow,
    Powi(i32),
    Sqrt,
    Cbrt,
    ExpM1,
    Ln1p,
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Abs,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Sign,
    Heaviside,
    DiracDelta,
    /// `atan2(second, top)`.
    Atan2,
    Floor,
    Ceiling,
    Min2,
    Max2,
    Gamma,
    LogGamma,
    Digamma,
    Erf,
    Erfc,
    LambertW,
    Factorial,
    /// `binomial(second, top)`.
    Binomial,
    /// `beta(second, top)`.
    Beta,
    BesselJ(i32),
    BesselY(i32),
    BesselI(i32),
    BesselK(i32),
    LegendreP(i32),
    ChebyshevT(i32),
    ChebyshevU(i32),
    HermiteH(i32),
    LaguerreL(i32),
    Fibonacci,
    Lucas,
    Harmonic,
    Factorial2,
    RisingFactorial,
    FallingFactorial,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
    Not,
    /// Pop a boolean (0/1); jump to the absolute index if it is zero (or NaN).
    JumpIfZero(usize),
    Jump(usize),
}

// ═══════════════════════════════════════════════════════════════════════════
// Emitter (iterative lowering with an explicit work stack)
// ═══════════════════════════════════════════════════════════════════════════

/// Work item for the explicit-stack lowering loop.
#[derive(Clone, Copy)]
enum Task {
    /// Lower this expression (its value ends up on the VM stack).
    Node(ExprId),
    /// Emit one instruction.
    Emit(Instruction),
    /// Bind label `l` to the current code position.
    Label(usize),
    /// Emit a `JumpIfZero` to label `l` (patched at the end).
    JumpIfZero(usize),
    /// Emit a `Jump` to label `l` (patched at the end).
    Jump(usize),
    /// Lower a boolean-valued expression (pushes 0.0/1.0).
    Bool(ExprId),
}

struct Emitter<'a> {
    arena: &'a Arena,
    vars: FxHashMap<&'a str, usize>,
    locals: FxHashMap<SymbolId, usize>,
    code: Vec<Instruction>,
    labels: Vec<Option<usize>>,
    fixups: Vec<(usize, usize)>,
    /// Peephole optimisations must not reach back across this position
    /// (a label may point here).
    barrier: usize,
    work: Vec<Task>,
}

impl<'a> Emitter<'a> {
    fn new(arena: &'a Arena, var_names: &[&'a str], locals: FxHashMap<SymbolId, usize>) -> Self {
        let vars = var_names.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        Self {
            arena,
            vars,
            locals,
            code: Vec::new(),
            labels: Vec::new(),
            fixups: Vec::new(),
            barrier: 0,
            work: Vec::new(),
        }
    }

    fn new_label(&mut self) -> usize {
        self.labels.push(None);
        self.labels.len() - 1
    }

    fn emit(&mut self, inst: Instruction) {
        // Peephole: `x; y; Powi(-1); Mul` → `x; y; Div` (one rounding instead of two).
        if inst == Instruction::Mul
            && self.code.len() > self.barrier
            && self.code.last() == Some(&Instruction::Powi(-1))
        {
            self.code.pop();
            self.code.push(Instruction::Div);
            return;
        }
        self.code.push(inst);
    }

    /// Push tasks so that they execute in the given (forward) order.
    fn push_seq(&mut self, tasks: &[Task]) {
        for t in tasks.iter().rev() {
            self.work.push(*t);
        }
    }

    /// Lower `root`, appending code that leaves its value on the VM stack.
    fn lower(&mut self, root: ExprId) -> Result<(), SymplexError> {
        self.work.push(Task::Node(root));
        while let Some(task) = self.work.pop() {
            match task {
                Task::Emit(inst) => self.emit(inst),
                Task::Label(l) => {
                    self.labels[l] = Some(self.code.len());
                    self.barrier = self.code.len();
                }
                Task::JumpIfZero(l) => {
                    self.fixups.push((self.code.len(), l));
                    self.code.push(Instruction::JumpIfZero(usize::MAX));
                }
                Task::Jump(l) => {
                    self.fixups.push((self.code.len(), l));
                    self.code.push(Instruction::Jump(usize::MAX));
                }
                Task::Node(id) => self.lower_node(id)?,
                Task::Bool(id) => self.lower_bool(id)?,
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<Instruction>, SymplexError> {
        for (pos, label) in self.fixups.drain(..) {
            let target = self.labels[label].ok_or_else(|| SymplexError::ComputationFailed {
                operation: "compile",
                reason: "internal error: unbound jump label".to_string(),
            })?;
            match &mut self.code[pos] {
                Instruction::JumpIfZero(t) | Instruction::Jump(t) => *t = target,
                _ => {}
            }
        }
        Ok(self.code)
    }

    fn unsupported(&self, what: &str) -> SymplexError {
        SymplexError::NotImplemented(format!("cannot compile `{what}` to a numerical function"))
    }

    /// Resolve an expression to a compile-time integer (for orders of Bessel
    /// functions and degrees of orthogonal polynomials).
    fn const_i32(&self, id: ExprId, what: &str) -> Result<i32, SymplexError> {
        if let Some(r) = self.arena.as_num(id)
            && r.is_integer()
            && let Some(n) = r.to_integer().to_i32()
        {
            return Ok(n);
        }
        if let ExprNode::Neg(inner) = self.arena.node(id)
            && let Some(r) = self.arena.as_num(*inner)
            && r.is_integer()
            && let Some(n) = r.to_integer().to_i32()
        {
            return Ok(-n);
        }
        Err(SymplexError::NotImplemented(format!(
            "{what} requires a constant integer order/degree, got `{}`",
            self.arena.display(id)
        )))
    }

    fn unary(&mut self, child: ExprId, inst: Instruction) {
        self.push_seq(&[Task::Node(child), Task::Emit(inst)]);
    }

    fn binary(&mut self, a: ExprId, b: ExprId, inst: Instruction) {
        self.push_seq(&[Task::Node(a), Task::Node(b), Task::Emit(inst)]);
    }

    /// Left fold: `c₀ op c₁ op c₂ …` with `empty` for zero children.
    fn nary(&mut self, children: &[ExprId], inst: Instruction, empty: f64) {
        if children.is_empty() {
            self.emit(Instruction::PushConst(empty));
            return;
        }
        let mut tasks: Vec<Task> = Vec::with_capacity(children.len() * 2);
        tasks.push(Task::Node(children[0]));
        for &c in &children[1..] {
            tasks.push(Task::Node(c));
            tasks.push(Task::Emit(inst));
        }
        self.push_seq(&tasks);
    }

    fn lower_node(&mut self, id: ExprId) -> Result<(), SymplexError> {
        let arena = self.arena;
        match arena.node(id).clone() {
            ExprNode::Num(nid) => {
                let r = arena.num(nid);
                let n = r.numer().to_f64().unwrap_or(f64::NAN);
                let d = r.denom().to_f64().unwrap_or(f64::NAN);
                self.emit(Instruction::PushConst(n / d));
            }
            ExprNode::Symbol(sid) => {
                if let Some(&slot) = self.locals.get(&sid) {
                    self.emit(Instruction::LoadLocal(slot));
                } else {
                    let name = arena.symbol_name(sid);
                    match self.vars.get(name) {
                        Some(&idx) => self.emit(Instruction::PushVar(idx)),
                        None => {
                            return Err(SymplexError::FreeSymbol {
                                name: name.to_string(),
                            });
                        }
                    }
                }
            }
            ExprNode::Pi => self.emit(Instruction::PushConst(std::f64::consts::PI)),
            ExprNode::E => self.emit(Instruction::PushConst(std::f64::consts::E)),
            ExprNode::PhysicalConstant(_, value_id) => self.work.push(Task::Node(value_id)),
            ExprNode::Infinity => self.emit(Instruction::PushConst(f64::INFINITY)),
            ExprNode::NegInfinity => self.emit(Instruction::PushConst(f64::NEG_INFINITY)),
            ExprNode::NaN | ExprNode::ComplexInfinity => {
                self.emit(Instruction::PushConst(f64::NAN))
            }
            ExprNode::ImaginaryUnit => return Err(self.unsupported("ImaginaryUnit")),

            ExprNode::Add(ref children) => {
                // exp(x) + (-1) [+ rest] → exp_m1(x) [+ rest]
                if children.len() >= 2 {
                    let exp_idx = children
                        .iter()
                        .position(|&c| matches!(arena.node(c), ExprNode::Exp(_)));
                    let neg_one_idx = children.iter().position(|&c| is_neg_one(arena, c));
                    if let (Some(ei), Some(ni)) = (exp_idx, neg_one_idx)
                        && ei != ni
                        && let ExprNode::Exp(inner) = arena.node(children[ei]).clone()
                    {
                        let mut tasks = vec![Task::Node(inner), Task::Emit(Instruction::ExpM1)];
                        for (i, &c) in children.iter().enumerate() {
                            if i != ei && i != ni {
                                tasks.push(Task::Node(c));
                                tasks.push(Task::Emit(Instruction::Add));
                            }
                        }
                        self.push_seq(&tasks);
                        return Ok(());
                    }
                }
                self.nary(children, Instruction::Add, 0.0);
            }
            ExprNode::Mul(ref children) => self.nary(children, Instruction::Mul, 1.0),
            ExprNode::Min(ref children) => self.nary(children, Instruction::Min2, f64::INFINITY),
            ExprNode::Max(ref children) => {
                self.nary(children, Instruction::Max2, f64::NEG_INFINITY)
            }
            ExprNode::Pow(base, exp) => {
                if let Some(r) = arena.as_num(exp) {
                    if r.is_integer()
                        && let Some(n) = r.to_integer().to_i32()
                    {
                        self.unary(base, Instruction::Powi(n));
                        return Ok(());
                    }
                    let one: num_bigint::BigInt = One::one();
                    if *r.numer() == one {
                        if *r.denom() == num_bigint::BigInt::from(2) {
                            self.unary(base, Instruction::Sqrt);
                            return Ok(());
                        }
                        if *r.denom() == num_bigint::BigInt::from(3) {
                            self.unary(base, Instruction::Cbrt);
                            return Ok(());
                        }
                    }
                    // Odd denominator q: the real root b^(p/q) = (sign(b)|b|^(1/q))^p,
                    // i.e. |b|^e with the sign of b restored only for odd p.
                    let two = num_bigint::BigInt::from(2);
                    if (r.denom() % &two) != Zero::zero() {
                        let odd_numer = (r.numer() % &two) != Zero::zero();
                        if odd_numer {
                            self.push_seq(&[
                                Task::Node(base),
                                Task::Emit(Instruction::Sign),
                                Task::Node(base),
                                Task::Emit(Instruction::Abs),
                                Task::Node(exp),
                                Task::Emit(Instruction::Pow),
                                Task::Emit(Instruction::Mul),
                            ]);
                        } else {
                            self.push_seq(&[
                                Task::Node(base),
                                Task::Emit(Instruction::Abs),
                                Task::Node(exp),
                                Task::Emit(Instruction::Pow),
                            ]);
                        }
                        return Ok(());
                    }
                }
                self.binary(base, exp, Instruction::Pow);
            }
            ExprNode::Neg(x) => self.unary(x, Instruction::Neg),
            ExprNode::Floor(x) => self.unary(x, Instruction::Floor),
            ExprNode::Ceiling(x) => self.unary(x, Instruction::Ceiling),
            ExprNode::Sin(x) => self.unary(x, Instruction::Sin),
            ExprNode::Cos(x) => self.unary(x, Instruction::Cos),
            ExprNode::Tan(x) => self.unary(x, Instruction::Tan),
            ExprNode::Exp(x) => self.unary(x, Instruction::Exp),
            ExprNode::Ln(x) => {
                // ln(1 + y) → ln_1p(y)
                if let ExprNode::Add(ref ch) = arena.node(x).clone()
                    && ch.len() == 2
                {
                    if is_one(arena, ch[0]) {
                        self.unary(ch[1], Instruction::Ln1p);
                        return Ok(());
                    }
                    if is_one(arena, ch[1]) {
                        self.unary(ch[0], Instruction::Ln1p);
                        return Ok(());
                    }
                }
                self.unary(x, Instruction::Ln);
            }
            ExprNode::Abs(x) => self.unary(x, Instruction::Abs),
            ExprNode::Asin(x) => self.unary(x, Instruction::Asin),
            ExprNode::Acos(x) => self.unary(x, Instruction::Acos),
            ExprNode::Atan(x) => self.unary(x, Instruction::Atan),
            ExprNode::Atan2(y, x) => self.binary(y, x, Instruction::Atan2),
            ExprNode::Sinh(x) => self.unary(x, Instruction::Sinh),
            ExprNode::Cosh(x) => self.unary(x, Instruction::Cosh),
            ExprNode::Tanh(x) => self.unary(x, Instruction::Tanh),
            ExprNode::Asinh(x) => self.unary(x, Instruction::Asinh),
            ExprNode::Acosh(x) => self.unary(x, Instruction::Acosh),
            ExprNode::Atanh(x) => self.unary(x, Instruction::Atanh),
            ExprNode::Sign(x) => self.unary(x, Instruction::Sign),
            ExprNode::Heaviside(x) => self.unary(x, Instruction::Heaviside),
            ExprNode::DiracDelta(x) => self.unary(x, Instruction::DiracDelta),

            ExprNode::Gamma(x) => self.unary(x, Instruction::Gamma),
            ExprNode::LogGamma(x) => self.unary(x, Instruction::LogGamma),
            ExprNode::Digamma(x) => self.unary(x, Instruction::Digamma),
            ExprNode::Erf(x) => self.unary(x, Instruction::Erf),
            ExprNode::Erfc(x) => self.unary(x, Instruction::Erfc),
            ExprNode::LambertW(x) => self.unary(x, Instruction::LambertW),
            ExprNode::Beta(a, b) => self.binary(a, b, Instruction::Beta),
            ExprNode::Factorial(x) => self.unary(x, Instruction::Factorial),
            ExprNode::Binomial(n, k) => self.binary(n, k, Instruction::Binomial),

            ExprNode::Piecewise(ref branches) => {
                let end = self.new_label();
                let mut tasks: Vec<Task> = Vec::with_capacity(branches.len() * 5 + 2);
                for &(value, cond) in branches.iter() {
                    let next = self.new_label();
                    tasks.push(Task::Bool(cond));
                    tasks.push(Task::JumpIfZero(next));
                    tasks.push(Task::Node(value));
                    tasks.push(Task::Jump(end));
                    tasks.push(Task::Label(next));
                }
                tasks.push(Task::Emit(Instruction::PushConst(f64::NAN)));
                tasks.push(Task::Label(end));
                self.push_seq(&tasks);
            }

            ExprNode::BoolTrue
            | ExprNode::BoolFalse
            | ExprNode::Gt(_, _)
            | ExprNode::Ge(_, _)
            | ExprNode::Eq_(_, _)
            | ExprNode::Ne(_, _)
            | ExprNode::And(_)
            | ExprNode::Or(_)
            | ExprNode::Not(_) => self.work.push(Task::Bool(id)),

            ExprNode::Apply(sid, ref args) => {
                let name = arena.symbol_name(sid).to_string();
                self.lower_apply(&name, args)?;
            }

            ExprNode::Derivative(_, _) => return Err(self.unsupported("Derivative")),
            ExprNode::Integral(_, _) => return Err(self.unsupported("Integral")),
            ExprNode::Sum(_, _, _, _) => return Err(self.unsupported("Sum")),
            ExprNode::Product_(_, _, _, _) => return Err(self.unsupported("Product")),
            ExprNode::Limit(_, _, _) => return Err(self.unsupported("Limit")),
            ExprNode::Series(_, _, _, _) => return Err(self.unsupported("Series")),
            ExprNode::LaplaceTransform(_, _, _) => {
                return Err(self.unsupported("LaplaceTransform"));
            }
            ExprNode::InverseLaplaceTransform(_, _, _) => {
                return Err(self.unsupported("InverseLaplaceTransform"));
            }
            ExprNode::Residue(_, _, _) => return Err(self.unsupported("Residue")),
            ExprNode::RootOf(_, _) => return Err(self.unsupported("RootOf")),
            ExprNode::RootSum(_, _, _) => return Err(self.unsupported("RootSum")),
            ExprNode::DSolve(_, _, _) => return Err(self.unsupported("DSolve")),
            ExprNode::ConditionSet(_, _) => return Err(self.unsupported("ConditionSet")),
            ExprNode::EmptySet => return Err(self.unsupported("EmptySet")),
            ExprNode::UniversalSet => return Err(self.unsupported("UniversalSet")),
            ExprNode::Interval(_, _, _) => return Err(self.unsupported("Interval")),
            ExprNode::FiniteSet(_) => return Err(self.unsupported("FiniteSet")),
            ExprNode::SetUnion(_) => return Err(self.unsupported("SetUnion")),
            ExprNode::SetIntersection(_) => return Err(self.unsupported("SetIntersection")),
            ExprNode::SetComplement(_, _) => return Err(self.unsupported("SetComplement")),
        }
        Ok(())
    }

    /// Lower a boolean-valued node to 0.0/1.0.
    fn lower_bool(&mut self, id: ExprId) -> Result<(), SymplexError> {
        let arena = self.arena;
        match arena.node(id).clone() {
            ExprNode::BoolTrue => self.emit(Instruction::PushConst(1.0)),
            ExprNode::BoolFalse => self.emit(Instruction::PushConst(0.0)),
            ExprNode::Gt(a, b) => self.binary(a, b, Instruction::Gt),
            ExprNode::Ge(a, b) => self.binary(a, b, Instruction::Ge),
            ExprNode::Eq_(a, b) => self.binary(a, b, Instruction::Eq),
            ExprNode::Ne(a, b) => self.binary(a, b, Instruction::Ne),
            ExprNode::And(ref ch) => self.nary_bool(ch, Instruction::And, 1.0),
            ExprNode::Or(ref ch) => self.nary_bool(ch, Instruction::Or, 0.0),
            ExprNode::Not(x) => self.push_seq(&[Task::Bool(x), Task::Emit(Instruction::Not)]),
            // A numeric expression in boolean position: nonzero ⇒ true.
            _ => self.work.push(Task::Node(id)),
        }
        Ok(())
    }

    fn nary_bool(&mut self, children: &[ExprId], inst: Instruction, empty: f64) {
        if children.is_empty() {
            self.emit(Instruction::PushConst(empty));
            return;
        }
        let mut tasks: Vec<Task> = Vec::with_capacity(children.len() * 2);
        tasks.push(Task::Bool(children[0]));
        for &c in &children[1..] {
            tasks.push(Task::Bool(c));
            tasks.push(Task::Emit(inst));
        }
        self.push_seq(&tasks);
    }

    /// Lower library `Apply` nodes (Bessel functions, orthogonal polynomials,
    /// integer sequences, factorial variants).
    fn lower_apply(&mut self, name: &str, args: &[ExprId]) -> Result<(), SymplexError> {
        use crate::base::arena as names;
        let arity_err = |n: usize| {
            SymplexError::NotImplemented(format!(
                "cannot compile `{name}` with {} argument(s) (expected {n})",
                args.len()
            ))
        };
        // (order, x) families with compile-time integer order.
        let ordered: Option<fn(i32) -> Instruction> = match name {
            n if n == names::FN_BESSELJ => Some(Instruction::BesselJ),
            n if n == names::FN_BESSELY => Some(Instruction::BesselY),
            n if n == names::FN_BESSELI => Some(Instruction::BesselI),
            n if n == names::FN_BESSELK => Some(Instruction::BesselK),
            n if n == names::FN_LEGENDRE => Some(Instruction::LegendreP),
            n if n == names::FN_CHEBYSHEV_T => Some(Instruction::ChebyshevT),
            n if n == names::FN_CHEBYSHEV_U => Some(Instruction::ChebyshevU),
            n if n == names::FN_HERMITE => Some(Instruction::HermiteH),
            n if n == names::FN_LAGUERRE => Some(Instruction::LaguerreL),
            _ => None,
        };
        if let Some(make) = ordered {
            if args.len() != 2 {
                return Err(arity_err(2));
            }
            let order = self.const_i32(args[0], name)?;
            self.unary(args[1], make(order));
            return Ok(());
        }
        let unary: Option<Instruction> = match name {
            n if n == names::FN_FIBONACCI => Some(Instruction::Fibonacci),
            n if n == names::FN_LUCAS => Some(Instruction::Lucas),
            n if n == names::FN_HARMONIC => Some(Instruction::Harmonic),
            n if n == names::FN_FACTORIAL2 => Some(Instruction::Factorial2),
            _ => None,
        };
        if let Some(inst) = unary {
            if args.len() != 1 {
                return Err(arity_err(1));
            }
            self.unary(args[0], inst);
            return Ok(());
        }
        let binary: Option<Instruction> = match name {
            n if n == names::FN_RISING_FACTORIAL => Some(Instruction::RisingFactorial),
            n if n == names::FN_FALLING_FACTORIAL => Some(Instruction::FallingFactorial),
            _ => None,
        };
        if let Some(inst) = binary {
            if args.len() != 2 {
                return Err(arity_err(2));
            }
            self.binary(args[0], args[1], inst);
            return Ok(());
        }
        Err(SymplexError::NotImplemented(format!(
            "cannot compile `Apply({name}, …)` to a numerical function"
        )))
    }
}

/// Iterative check for nodes that must bypass the `eval()` pre-pass
/// (`Piecewise` and orthogonal-polynomial `Apply` nodes).
fn needs_raw_lowering(arena: &Arena, root: ExprId) -> bool {
    use crate::base::arena as names;
    let mut stack = vec![root];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = arena.node(id);
        match node {
            ExprNode::Piecewise(_) => return true,
            ExprNode::Apply(sid, _) => {
                let name = arena.symbol_name(*sid);
                if name == names::FN_LEGENDRE
                    || name == names::FN_CHEBYSHEV_T
                    || name == names::FN_CHEBYSHEV_U
                    || name == names::FN_HERMITE
                    || name == names::FN_LAGUERRE
                {
                    return true;
                }
            }
            _ => {}
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// Check if a node is the constant −1 (either `Num(-1)` or `Neg(Num(1))`).
fn is_neg_one(arena: &Arena, id: ExprId) -> bool {
    if let Some(r) = arena.as_num(id) {
        return *r == num_rational::Ratio::from_integer((-1).into());
    }
    if let ExprNode::Neg(inner) = arena.node(id)
        && let Some(r) = arena.as_num(*inner)
    {
        return r.is_one();
    }
    false
}

/// Check if a node is the constant 1.
fn is_one(arena: &Arena, id: ExprId) -> bool {
    arena.as_num(id).is_some_and(|r| r.is_one())
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn lambdify_polynomial() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let three = a.int(3);
        let x2 = a.pow(x, two);
        let three_x = a.mul(&[three, x]);
        let one = a.int(1);
        let expr = a.add(&[x2, three_x, one]);
        // f(x) = x^2 + 3x + 1
        let f = compile(&mut a, expr, &["x"]).unwrap();
        assert!((f(&[2.0]) - 11.0).abs() < 1e-10); // 4 + 6 + 1 = 11
        assert!((f(&[0.0]) - 1.0).abs() < 1e-10);
        assert!((f(&[-1.0]) - (-1.0)).abs() < 1e-10); // 1 - 3 + 1 = -1
        assert_eq!(f.arity(), 1);
    }

    #[test]
    fn lambdify_trig() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let expr = a.sin(x);
        let f = compile(&mut a, expr, &["x"]).unwrap();
        assert!((f(&[0.0]) - 0.0).abs() < 1e-10);
        assert!((f(&[std::f64::consts::FRAC_PI_2]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn lambdify_two_vars() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let xy = a.mul(&[x, y]);
        let one = a.int(1);
        let expr = a.add(&[xy, one]);
        let f = compile(&mut a, expr, &["x", "y"]).unwrap();
        assert!((f(&[3.0, 4.0]) - 13.0).abs() < 1e-10);
    }

    #[test]
    fn lambdify_exp_sin() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let sin_x = a.sin(x);
        let expr = a.exp(sin_x);
        let f = compile(&mut a, expr, &["x"]).unwrap();
        let expected = 0.5f64.sin().exp();
        assert!((f(&[0.5]) - expected).abs() < 1e-14);
    }

    #[test]
    fn lambdify_pi_e() {
        let mut a = Arena::new();
        let pi = a.pi;
        let f = compile(&mut a, pi, &[]).unwrap();
        assert!((f(&[]) - std::f64::consts::PI).abs() < 1e-10);
        let e = a.e_const;
        let f2 = compile(&mut a, e, &[]).unwrap();
        assert!((f2(&[]) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn lambdify_complex_fails() {
        let mut a = Arena::new();
        let i = a.i_unit;
        let result = compile(&mut a, i, &[]);
        assert!(matches!(result, Err(SymplexError::NotImplemented(_))));
    }

    #[test]
    fn lambdify_free_symbol_is_error() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let expr = a.add(&[x, y]);
        match compile(&mut a, expr, &["x"]) {
            Err(SymplexError::FreeSymbol { name }) => assert_eq!(name, "y"),
            other => panic!("expected FreeSymbol, got {other:?}"),
        }
    }

    #[test]
    fn arity_mismatch_is_nan_or_error() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let f = compile(&mut a, x, &["x"]).unwrap();
        assert!(f.call(&[]).is_nan());
        assert!(f(&[1.0, 2.0]).is_nan());
        assert!(f.try_call(&[]).is_err());
        assert_eq!(f.try_call(&[7.0]).unwrap(), 7.0);
    }

    #[test]
    fn duplicate_param_is_error() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        assert!(matches!(
            compile(&mut a, x, &["x", "x"]),
            Err(SymplexError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn division_peephole() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let m1 = a.int(-1);
        let inv_y = a.pow(y, m1);
        let expr = a.mul(&[x, inv_y]);
        let f = compile(&mut a, expr, &["x", "y"]).unwrap();
        assert_eq!(f(&[1.0, 3.0]), 1.0 / 3.0);
    }

    #[test]
    fn piecewise_and_booleans() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let zero = a.int(0);
        let one = a.int(1);
        let two = a.int(2);
        let gt = a.gt(x, zero);
        let neg_x = a.neg(x);
        let eq1 = a.eq_(x, one);
        let t = a.bool_true;
        // piecewise: x==1 → 2 ; x>0 → x ; else → -x
        let pw = a.piecewise(&[(two, eq1), (x, gt), (neg_x, t)]);
        let f = compile(&mut a, pw, &["x"]).unwrap();
        assert_eq!(f(&[1.0]), 2.0);
        assert_eq!(f(&[3.0]), 3.0);
        assert_eq!(f(&[-4.0]), 4.0);
        // no matching branch → NaN
        let pw2 = a.piecewise(&[(x, gt)]);
        let g = compile(&mut a, pw2, &["x"]).unwrap();
        assert_eq!(g(&[2.0]), 2.0);
        assert!(g(&[-2.0]).is_nan());
        // nested boolean logic
        let five = a.int(5);
        let lt5 = a.gt(five, x);
        let both = a.and(&[gt, lt5]);
        let not_both = a.not(both);
        let pw3 = a.piecewise(&[(one, not_both), (zero, t)]);
        let h = compile(&mut a, pw3, &["x"]).unwrap();
        assert_eq!(h(&[3.0]), 0.0);
        assert_eq!(h(&[7.0]), 1.0);
        assert_eq!(h(&[-1.0]), 1.0);
    }

    #[test]
    fn special_functions_compile() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let g = a.gamma(x);
        let f = compile(&mut a, g, &["x"]).unwrap();
        assert!(close(f(&[0.5]), std::f64::consts::PI.sqrt(), 1e-15));
        let e = a.erf(x);
        let f = compile(&mut a, e, &["x"]).unwrap();
        assert!(close(f(&[1.0]), 0.84270079294971486934, 1e-15));
        let w = a.lambertw(x);
        let f = compile(&mut a, w, &["x"]).unwrap();
        assert!(close(f(&[1.0]), 0.56714329040978387300, 1e-15));
        let two = a.int(2);
        let j2 = a.besselj(two, x);
        let f = compile(&mut a, j2, &["x"]).unwrap();
        assert!(close(f(&[10.0]), 0.25463031368512062253, 1e-14));
        let sym_order = a.symbol("n");
        let jn = a.besselj(sym_order, x);
        assert!(matches!(
            compile(&mut a, jn, &["x", "n"]),
            Err(SymplexError::NotImplemented(_))
        ));
        let fac = a.factorial(x);
        let f = compile(&mut a, fac, &["x"]).unwrap();
        assert_eq!(f(&[5.0]), 120.0);
        let fib = a.fibonacci(x);
        let f = compile(&mut a, fib, &["x"]).unwrap();
        assert_eq!(f(&[20.0]), 6765.0);
    }

    #[test]
    fn compile_many_shares_cse() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let s = a.sin(x);
        let c = a.cos(x);
        let two = a.int(2);
        let s2 = a.pow(s, two);
        let e1 = a.add(&[s2, c]);
        let e2 = a.mul(&[s, c]);
        let f = compile_many(&mut a, &[e1, e2], &["x"]).unwrap();
        assert_eq!(f.len(), 2);
        assert_eq!(f.arity(), 1);
        let out = f.call_vec(&[0.3]);
        let (s, c) = (0.3f64.sin(), 0.3f64.cos());
        assert!(close(out[0], s * s + c, 1e-15));
        assert!(close(out[1], s * c, 1e-15));
        let mut buf = [0.0; 1];
        f.call(&[0.3], &mut buf);
        assert!(buf[0].is_nan());
        assert!(f.try_call(&[0.3], &mut buf).is_err());
        let empty = compile_many(&mut a, &[], &["x"]).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.call_vec(&[1.0]).len(), 0);
        let empty2 = compile_many_empty(&["x", "y"]).unwrap();
        assert_eq!(empty2.arity(), 2);
        assert!(empty2.is_empty());
    }

    #[test]
    fn deep_expression_does_not_overflow_stack() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let mut e = x;
        for _ in 0..20_000 {
            e = a.sin(e);
        }
        let f = compile(&mut a, e, &["x"]).unwrap();
        assert!(f(&[1.0]).is_finite());
    }
}
