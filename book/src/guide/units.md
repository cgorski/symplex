# Dimensional Analysis

`symplex::units` provides **compile-time** dimensional analysis: a quantity's dimension is part of its Rust type, so adding a `Mass` to a `Length` does not compile, and differentiating a `Length` with respect to a `Time` yields a `Velocity`. Underneath, every quantity is an `Ex` in SI base units, so all the symbolic machinery still applies.

## Quantity types

`Qty<D>` is generic over a type-level dimension `Dim<L, M, T, I, Θ, N, J>` (powers of length, mass, time, current, temperature, amount, luminous intensity, as `typenum` integers). Thirty named aliases cover the common cases: `Length`, `Mass`, `Time`, `Current`, `Temperature`, `Area`, `Volume`, `Velocity`, `Acceleration`, `AngularVelocity`, `Frequency`, `Force`, `Energy`, `Torque`, `Power`, `Momentum`, `AngularMomentum`, `MomentOfInertia`, `Pressure`, `Stiffness`, `Damping`, `Voltage`, `Resistance`, `Inductance`, `Capacitance`, `Charge`, `MagneticFlux`, `Dimensionless`, `Angle`, ….

```rust
use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let ctx = Context::new();
    let m = Mass::symbol(&ctx, "m");
    let a = Acceleration::symbol(&ctx, "a");

    // The `: Force` annotation is checked by the compiler.
    let f = dim!(ctx, Force: m * a);
    println!("F = {f}");                          // a*m [N]

    let d = Length::symbol(&ctx, "d");
    let w = dim!(ctx, Energy: f * d);
    println!("W = {w}");                          // a*d*m [J]

    // Substitute and evaluate like any Ex
    let f_num = f.subs(&m, &ctx.int(10)).subs(&a, &ctx.rational(981, 100)).eval();
    println!("{f_num}");                          // 981/10 [N]
    println!("{}", f_num.eval_f64().unwrap());    // 98.1
}
```

`Mass + Length` is a type error; `Mass * Acceleration` is a `Force`; `Energy / Time` is a `Power`. Multiplication and division of quantities compute the resulting dimension at the type level.

## Typed calculus

`diff_wrt(&Time)` divides the dimension by time; `integrate_wrt` multiplies. Build the formula with `expr!` on raw symbols and wrap it with `from_ex`:

```rust
use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; g, t);                    // raw symbols for expr!
    let t_var = Time::symbol(&ctx, "t");

    let x = Length::from_ex(expr!(ctx, 1 / 2 * g * t ^ 2));
    let v: Velocity = x.diff_wrt(&t_var);         // d(Length)/d(Time)
    let acc: Acceleration = v.diff_wrt(&t_var);
    println!("{x}\n{v}\n{acc}");                  // 1/2*g*t^2 [m]  g*t [m/s]  g [m/s²]
    println!("{}", acc.inner());                  // the underlying Ex: g
}
```

## Exact conversions

Named constructors convert from other units with **exact rational** factors: `Length::inches(&x)`, `Length::miles(&x)`, `Force::pound_force(&x)`, `Pressure::psi(&x)`, `Energy::btu(&x)`, `Volume::us_gallons(&x)`, … (about 100 in total). The result is in SI.

```rust
use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let ctx = Context::new();
    let one = ctx.int(1);
    println!("{}", Length::inches(&one).eval());          // 127/5000 [m]
    println!("{}", Force::pound_force(&one).eval());      // exact rational newtons
    println!("{}", Pressure::psi(&one).eval());
    println!("{}", Energy::btu(&one).eval());
    println!("{:.6}", Volume::us_gallons(&one).eval_f64().unwrap() * 1000.0);   // 3.785412 L
}
```

## Physical constants

`symplex::units::constants` provides typed constants (`speed_of_light`, `standard_gravity`, `planck_constant`, `boltzmann_constant`, `gravitational_constant`, …) as exact `PhysicalConstant` nodes that display by name and evaluate on demand. `cargo run --example physics_constants` shows them.

## Compile-time assertions

`symplex::const_assert_dim!` asserts a dimensional identity at compile time (for example that `Energy` equals `Force × Length`); `symplex::units::inference` infers dimensions of an untyped expression from a `DimMap` of symbol dimensions. See `examples/units_physics.rs`.

## Code generation with `uom`

`CodegenOptions::with_uom()` (plus `param_units` / `return_unit`) annotates generated Rust functions with [`uom`](https://crates.io/crates/uom) quantity types at the boundary (raw `f64` inside). `cargo run --example units_engineering` shows a motor-design workflow ending in typed generated code.

## Examples

```sh
cargo run --example units_physics           # Newton, Ohm, pendulum, compile-time assertions
cargo run --example units_electrical        # Circuit analysis with units
cargo run --example units_engineering       # Motor design, imperial conversions, uom codegen
cargo run --example units_kinematics        # Kinematics with typed calculus
cargo run --example units_lagrangian        # Lagrangian mechanics with units
```
