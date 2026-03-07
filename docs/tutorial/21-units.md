# Chapter 21: Compile-Time Dimensional Analysis

Symplex's `units` module catches physics bugs at compile time. You annotate symbolic expressions with physical dimensions — `Force`, `Voltage`, `Length` — and the Rust type system prevents you from ever adding meters to kilograms or confusing energy with momentum. No runtime cost, no overhead in release builds. If it compiles, the dimensions are correct.

## 1. Why Dimensional Analysis?

On September 23, 1999, NASA's Mars Climate Orbiter disintegrated in the Martian atmosphere. The root cause: one software module output thrust impulse in pound-force·seconds, while another expected newton·seconds. A factor-of-4.45 error, uncaught by any test, because both values were just `f64`.

This class of bug is shockingly common. Consider a motor controller:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(v, i, r);

// Intent: compute power dissipation P = I²R
// Bug: accidentally wrote V * I * R instead of I * I * R
let p_wrong = &v * &i * &r;  // compiles fine — it's just Ex × Ex × Ex
let p_right = &i * &i * &r;  // also compiles — both are just `Ex`
```

Both lines compile. Both produce an `Ex`. Nothing in the type system distinguishes voltage from current from resistance. The bug hides until it produces wrong numbers in production.

Now with dimensional analysis:

```rust
use symplex::prelude::*;
use symplex::units::*;

let v = Voltage::symbol("V");
let i = Current::symbol("I");
let r = Resistance::symbol("R");

// P = I * I * R → Current × Current × Resistance
// The mul table knows: Current × Resistance = Voltage,
// then Voltage × Current = Power. But I*I is not in the named table,
// so this goes through Qty<D> and you'd use the generic path.

// The correct way with named types:
let p: Power = &v * &i;       // Voltage × Current = Power ✓
// let wrong: Power = &v * &r; // ERROR: Voltage × Resistance is not Power
```

The compiler rejects the mistake before your code ever runs.

## 2. Getting Started

Import everything from the `units` module:

```rust
use symplex::prelude::*;
use symplex::units::*;
```

### Three Ways to Build Typed Quantities

Symplex offers three complementary patterns. Use whichever fits your situation:

**a) `expr!` + `from_ex()` — Most Ergonomic (recommended for complex formulas)**

```rust
use symplex::prelude::*;
use symplex::units::*;

symplex::vars!(m, v, g, h);

let ke = Energy::from_ex(expr!(1/2 * m * v^2));
let pe = Energy::from_ex(expr!(m * g * h));
let total: Energy = &ke + &pe;
```

The `expr!` macro gives you natural math syntax (`^` for powers, implicit multiplication with `*`, fractions with `/`). Wrap the result with `from_ex()` to attach a dimension. The `IntoEx` trait means `.clone()` is never needed when passing values to `from_ex()`.

**b) Named Type Arithmetic — Compile-Time Checked (best for simple products/quotients)**

```rust
use symplex::prelude::*;
use symplex::units::*;

let m = Mass::symbol("m");
let a = Acceleration::symbol("a");
let f: Force = &m * &a;  // Mass × Acceleration = Force (compiler-verified)
```

The multiplication/division table (~46 rules) catches dimension errors at compile time. If the result type is wrong, the compiler tells you immediately.

**c) `diff_wrt` / `integrate_wrt` — Typed Calculus (unique feature)**

```rust
use symplex::prelude::*;
use symplex::units::*;

symplex::vars!(a, t);
let position = Length::from_ex(expr!(1/2 * a * t^2));
let t_var = Time::symbol("t");
let velocity: Velocity = position.diff_wrt(&t_var);
let acceleration: Acceleration = velocity.diff_wrt(&t_var);
```

The `DiffWrt` and `IntWrt` traits encode physical laws like d(Length)/d(Time) = Velocity directly in the type system. See [Section 9](#9-differentiation-and-integration) for the full table.

### Creating Symbolic Variables

Every named quantity type has a `symbol()` constructor that creates a symbolic variable with that dimension:

```rust
use symplex::prelude::*;
use symplex::units::*;

let m = Mass::symbol("m");       // a symbolic mass
let a = Acceleration::symbol("a"); // a symbolic acceleration
let t = Time::symbol("t");       // a symbolic time
```

### Constants and Rationals

Use `constant()` for integer values and `rational()` for exact fractions:

```rust
use symplex::prelude::*;
use symplex::units::*;

// Gravitational acceleration: 9.81 m/s² as exact rational 981/100
let g = Acceleration::rational(981, 100);

// A 5 kg mass
let m = Mass::constant(5);

// A force of 100 N
let f = Force::constant(100);
```

### The Zero Value

Every type has a `zero()` constructor:

```rust
use symplex::prelude::*;
use symplex::units::*;

let no_force = Force::zero();
let no_velocity = Velocity::zero();
```

### Wrapping Raw Expressions

Use `expr!` + `from_ex()` to build complex formulas ergonomically:

```rust
use symplex::prelude::*;
use symplex::units::*;

symplex::vars!(m, a, t);

// expr! gives natural math syntax; from_ex() attaches the dimension
let f = Force::from_ex(expr!(m * a));
let position = Length::from_ex(expr!(1/2 * a * t^2));

// Also works with a plain Ex variable
let raw_expr = symplex::var("F_applied");
let f2 = Force::from_ex(raw_expr);
```

The `IntoEx` trait means `from_ex()` accepts both `Ex` and `&Ex` — no `.clone()` needed.

## 3. Named Types and Arithmetic

### Multiplication: F = ma

The named multiplication table knows that `Mass × Acceleration = Force`:

```rust
use symplex::prelude::*;
use symplex::units::*;

let m = Mass::symbol("m");
let a = Acceleration::symbol("a");

let f: Force = m * a;
println!("{f}"); // m*a [N]
```

The return type is `Force` — not `Ex`, not some generic wrapper. If you annotate the wrong type, the compiler tells you exactly what went wrong:

```rust
// let v: Velocity = m * a;
// ERROR: expected `Velocity`, found `Force`
```

### Addition and Subtraction: Same Type Only

You can only add or subtract quantities of the same dimension:

```rust
use symplex::prelude::*;
use symplex::units::*;

let f1 = Force::symbol("F1");
let f2 = Force::symbol("F2");
let f_total: Force = f1 + f2;
println!("{f_total}"); // F1 + F2 [N]
```

Trying to add different types is a compile error:

```rust
// let m = Mass::symbol("m");
// let l = Length::symbol("l");
// let nonsense = m + l;
// ERROR: expected `Mass`, found `Length`
```

The error message is exactly that clear — named newtypes produce errors like "expected `Mass`, found `Length`", not pages of typenum gibberish.

### Negation

```rust
use symplex::prelude::*;
use symplex::units::*;

let f = Force::symbol("F");
let reaction: Force = -&f;
println!("{reaction}"); // -F [N]
```

### Scalar Multiplication

Multiply by `i64` or `&Ex` to scale a quantity without changing its dimension:

```rust
use symplex::prelude::*;
use symplex::units::*;

let f = Force::symbol("F");

// Integer scaling
let double_f: Force = &f * 2;
let also_double: Force = 2 * &f;

// Symbolic scaling
let half = symplex::rational(1, 2);
let half_f: Force = &f * &half;
let also_half: Force = &half * f;

println!("{half_f}"); // 1/2*F [N]
```

### Division by Integer

```rust
use symplex::prelude::*;
use symplex::units::*;

let e = Energy::symbol("E");
let half_energy: Energy = e / 2;
println!("{half_energy}"); // 1/2*E [J]
```

## 4. The Multiplication/Division Table

Symplex encodes ~46 multiplication rules and ~23 division rules between named types. These are the physical relationships the type system enforces.

### Key Multiplication Rules

| Left | × | Right | = | Result |
|------|---|-------|---|--------|
| `Mass` | × | `Acceleration` | → | `Force` |
| `Force` | × | `Length` | → | `Energy` |
| `Force` | × | `Velocity` | → | `Power` |
| `Mass` | × | `Velocity` | → | `Momentum` |
| `Current` | × | `Resistance` | → | `Voltage` |
| `Voltage` | × | `Current` | → | `Power` |
| `Power` | × | `Time` | → | `Energy` |
| `MomentOfInertia` | × | `AngularAcceleration` | → | `Torque` |
| `Inductance` | × | `Current` | → | `MagneticFlux` |
| `Velocity` | × | `Time` | → | `Length` |
| `Stiffness` | × | `Length` | → | `Force` |
| `Damping` | × | `Velocity` | → | `Force` |

All rules work in both argument orders: `Mass * Acceleration` and `Acceleration * Mass` both return `Force`.

### Key Division Rules

```rust
use symplex::prelude::*;
use symplex::units::*;

let f = Force::symbol("F");
let m = Mass::symbol("m");

// Force / Mass = Acceleration
let a: Acceleration = f / m;
println!("{a}"); // F/m [m/s²]
```

| Numerator | ÷ | Denominator | = | Result |
|-----------|---|-------------|---|--------|
| `Force` | ÷ | `Mass` | → | `Acceleration` |
| `Force` | ÷ | `Acceleration` | → | `Mass` |
| `Energy` | ÷ | `Time` | → | `Power` |
| `Energy` | ÷ | `Length` | → | `Force` |
| `Length` | ÷ | `Time` | → | `Velocity` |
| `Velocity` | ÷ | `Time` | → | `Acceleration` |
| `Voltage` | ÷ | `Current` | → | `Resistance` |
| `Voltage` | ÷ | `Resistance` | → | `Current` |
| `Power` | ÷ | `Velocity` | → | `Force` |
| `MagneticFlux` | ÷ | `Current` | → | `Inductance` |
| `Charge` | ÷ | `Time` | → | `Current` |

### Dimensionless and Angle Scaling

Both `Dimensionless` and `Angle` act as scalars when multiplied with any named type:

```rust
use symplex::prelude::*;
use symplex::units::*;

let f = Force::symbol("F");
let ratio = Dimensionless::symbol("eta");
let scaled: Force = ratio * f;  // Dimensionless × Force = Force
```

This also works with `Angle`:

```rust
use symplex::prelude::*;
use symplex::units::*;

let theta = Angle::symbol("theta");
let m = Mass::symbol("m");
let scaled: Mass = theta * m;  // Angle acts as dimensionless for multiplication
```

## 5. The Generic Qty&lt;D&gt; Fallback

Not every product of dimensions has a named type. When you compute something the named table doesn't cover, you use the generic `Qty<D>` type.

### When Named Types Aren't Enough

The named table knows `Mass × Acceleration → Force`, but it doesn't have a named type for every conceivable combination. For instance, there's no `VelocitySquared` type. When you need an intermediate computation that passes through an unnamed dimension, convert to `Qty<D>` with `.as_qty()`:

```rust
use symplex::prelude::*;
use symplex::units::*;

let m = Mass::symbol("m");
let v = Velocity::symbol("v");

// Convert to Qty for generic arithmetic
let m_qty = m.as_qty();
let v_qty = v.as_qty();

// Qty<VelocityDim> × Qty<VelocityDim> → Qty<SomeDim> (velocity²)
// Then Qty × Qty<MassDim> → Qty<EnergyDim>
let ke_qty = &m_qty * &(&v_qty * &v_qty);

// Convert back to a named type
let ke: Energy = ke_qty.into();
println!("{ke}"); // m*v*v [J]

// Scale by 1/2 for actual kinetic energy
let half = symplex::rational(1, 2);
let ke_half: Energy = ke * &half;
println!("{ke_half}"); // 1/2*m*v*v [J]
```

### Blanket Mul/Div on Qty

The generic `Qty<D>` has blanket `Mul` and `Div` implementations that compute the output dimension via typenum arithmetic. Any `Qty<D1> * Qty<D2>` adds dimension exponents; any `Qty<D1> / Qty<D2>` subtracts them:

```rust
use symplex::prelude::*;
use symplex::units::*;

let l = Length::symbol("L");
let t = Time::symbol("t");

// Work entirely in Qty-land
let l_qty = l.as_qty();
let t_qty = t.as_qty();

// Length / Time → Qty<VelocityDim>
let v_qty = &l_qty / &t_qty;

// Convert to named type
let v: Velocity = v_qty.into();
println!("{v}"); // L/t [m/s]
```

### From/Into Conversions

Named types convert **to** `Qty<D>` via `From`/`Into`:

```rust
use symplex::prelude::*;
use symplex::units::*;

let f = Force::symbol("F");

// Named → Qty (always works via .as_qty() or Into)
let f_qty: Qty<ForceDim> = f.into();
```

And `Qty<D>` converts **back** to the primary named type for that dimension via `From`/`Into`:

```rust
use symplex::prelude::*;
use symplex::units::*;

let f_qty: Qty<ForceDim> = Qty::from_ex(symplex::var("F"));
let f: Force = f_qty.into();  // Qty<ForceDim> → Force
```

### The map() Method

Transform the inner expression while preserving the dimension:

```rust
use symplex::prelude::*;
use symplex::units::*;

let f_qty: Qty<ForceDim> = Qty::from_ex(symplex::var("x") * symplex::var("x"));
let simplified = f_qty.map(|ex| ex.simplify());
```

## 6. Trig Functions and Dimensionless Quantities

### Trig Only on Angle

The `sin()`, `cos()`, and `tan()` methods are defined **only** on the `Angle` type. Each returns a `Dimensionless`:

```rust
use symplex::prelude::*;
use symplex::units::*;

let theta = Angle::symbol("theta");

let s: Dimensionless = theta.sin();   // sin(θ) → dimensionless
let c: Dimensionless = theta.cos();   // cos(θ) → dimensionless
let t: Dimensionless = theta.tan();   // tan(θ) → dimensionless

println!("{s}"); // sin(theta) [1]
println!("{c}"); // cos(theta) [1]
```

Calling `.sin()` on any other type is a compile error:

```rust
// let l = Length::symbol("L");
// let nope = l.sin();  // ERROR: no method named `sin` found for struct `Length`
```

This is exactly the protection you want. Taking the sine of a length is physically meaningless.

### Dimensionless vs. Angle

`Dimensionless` and `Angle` have the same SI dimension vector (all zeros), but they are distinct Rust types. This is intentional: angles are dimensionless in SI, but you don't want to accidentally pass a strain ratio to a trig function.

Convert between them explicitly:

```rust
use symplex::prelude::*;
use symplex::units::*;

let ratio = Dimensionless::symbol("eta");

// Dimensionless → Angle (named constructor)
let theta = Angle::from_dimensionless(ratio);

// Now you can take the sine
let s = theta.sin();
```

### Angular Velocity × Time → Angle

The multiplication table gives `AngularVelocity × Time → Angle`:

```rust
use symplex::prelude::*;
use symplex::units::*;

let omega = AngularVelocity::symbol("omega");
let t = Time::symbol("t");

let theta: Angle = omega * t;
let displacement: Dimensionless = theta.cos();
println!("{displacement}"); // cos(omega*t) [1]
```

## 7. Unit Conversions

All quantities are stored internally in SI base units. Conversion constructors accept a value in the specified unit and normalize to SI:

```rust
use symplex::prelude::*;
use symplex::units::*;

let five = symplex::int(5);

// 5 kilometers → 5000 meters internally
let d = Length::kilometers(&five);
println!("{d}"); // 5000 [m]

// 1 horsepower → 745.7 watts internally
let p = Power::horsepower(&symplex::int(1));

// 100°C → 373.15 K internally
let temp = Temperature::from_celsius(&symplex::int(100));
```

Conversion constructors work with symbolic expressions too:

```rust
use symplex::prelude::*;
use symplex::units::*;

let x = symplex::var("x");
let d = Length::kilometers(&x);
println!("{d}"); // 1000*x [m]
```

### Available Conversion Constructors

**Length:** `meters`, `kilometers`, `centimeters`, `millimeters`, `micrometers`, `inches`, `feet`, `yards`, `miles`

**Mass:** `kilograms`, `grams`, `milligrams`, `tonnes`, `pounds`

**Time:** `seconds`, `milliseconds`, `microseconds`, `nanoseconds`, `minutes`, `hours`, `days`

**Angle:** `radians`, `degrees`, `revolutions`

**Velocity:** `meters_per_second`, `kilometers_per_hour`

**Force:** `newtons`, `kilonewtons`

**Energy:** `joules`, `kilojoules`, `kilowatt_hours`

**Power:** `watts`, `kilowatts`, `megawatts`, `horsepower`

**Voltage:** `volts`, `millivolts`, `kilovolts`

**Current:** `amperes`, `milliamperes`, `microamperes`

**Resistance:** `ohms`, `milliohms`, `kilohms`, `megohms`

**Pressure:** `pascals`, `kilopascals`, `megapascals`, `bars`, `atmospheres`

**Frequency:** `hertz`, `kilohertz`, `megahertz`, `gigahertz`, `rpm`

**Temperature:** `kelvins`, `from_celsius`, `from_fahrenheit`

**Torque:** `newton_meters`

**Acceleration:** `meters_per_second_squared`, `standard_gravity`

**Area:** `square_meters`, `square_kilometers`, `square_centimeters`, `hectares`

**Volume:** `cubic_meters`, `liters`, `milliliters`

**Momentum:** `kilogram_meters_per_second`

**AngularVelocity:** `radians_per_second`, `rpm`, `degrees_per_second`

**Charge:** `coulombs`, `milliampere_hours`, `ampere_hours`

**Capacitance:** `farads`, `microfarads`, `nanofarads`, `picofarads`

**Inductance:** `henrys`, `millihenrys`, `microhenrys`

**MagneticFlux:** `webers`

**Stiffness:** `newtons_per_meter`, `kilonewtons_per_meter`

**Damping:** `newton_seconds_per_meter`

**Dimensionless:** `percent`, `per_mille`

### Temperature Is Special

Temperature conversions are affine (they involve an offset, not just a scale factor):

```rust
use symplex::prelude::*;
use symplex::units::*;

// 0°C = 273.15 K
let freezing = Temperature::from_celsius(&symplex::int(0));

// 212°F = 373.15 K (boiling point of water)
let boiling = Temperature::from_fahrenheit(&symplex::int(212));
```

## 8. Compile-Time Assertions

### assert_dim! — Checkpoint Assertions

Use `assert_dim!` to verify that an expression has the expected dimension at a specific point in your code:

```rust
use symplex::prelude::*;
use symplex::units::*;

let m = Mass::symbol("m");
let a = Acceleration::symbol("a");

// Checkpoint: verify this is a Force
let f = symplex::assert_dim!(m * a, Force);

// You can keep using f — the macro returns the value
println!("{f}"); // m*a [N]
```

If the dimension is wrong, the compiler shows a clear mismatch:

```rust
// let m = Mass::symbol("m");
// let a = Acceleration::symbol("a");
// let wrong = symplex::assert_dim!(m * a, Velocity);
// ERROR: expected `Velocity`, found `Force`
```

### const_assert_dim! — Formula Verification

For compile-time verification of dimensional formulas without any runtime values, use `const_assert_dim!` with `ConstDim`:

```rust
use symplex::units::ConstDim;

// Verify at compile time: F = ma → Mass × Acceleration = Force
symplex::const_assert_dim!(
    ConstDim::MASS.mul(ConstDim::ACCELERATION),
    ConstDim::FORCE,
    "F = ma: Mass × Acceleration must equal Force"
);

// Verify: E = Fd → Force × Length = Energy
symplex::const_assert_dim!(
    ConstDim::FORCE.mul(ConstDim::LENGTH),
    ConstDim::ENERGY,
    "W = Fd: Force × Length must equal Energy"
);

// Verify: P = E/t → Energy / Time = Power
symplex::const_assert_dim!(
    ConstDim::ENERGY.div(ConstDim::TIME),
    ConstDim::POWER,
    "P = E/t: Energy / Time must equal Power"
);

// Verify Ohm's law: V = IR → Current × Resistance = Voltage
symplex::const_assert_dim!(
    ConstDim::CURRENT.mul(ConstDim::RESISTANCE),
    ConstDim::VOLTAGE,
    "V = IR: Current × Resistance must equal Voltage"
);
```

If a formula is wrong, the compiler emits your custom message as a compile-time panic:

```rust
// This would fail at compile time with the message "this formula is wrong":
// symplex::const_assert_dim!(
//     ConstDim::MASS.mul(ConstDim::VELOCITY),
//     ConstDim::FORCE,
//     "this formula is wrong"
// );
```

`ConstDim` supports `.mul()`, `.div()`, and `.eq()` — all as `const fn`. The full list of named constants: `DIMENSIONLESS`, `ANGLE`, `LENGTH`, `MASS`, `TIME`, `CURRENT`, `TEMPERATURE`, `AREA`, `VOLUME`, `VELOCITY`, `ACCELERATION`, `ANGULAR_VELOCITY`, `ANGULAR_ACCELERATION`, `FREQUENCY`, `FORCE`, `ENERGY`, `TORQUE`, `POWER`, `MOMENTUM`, `ANGULAR_MOMENTUM`, `MOMENT_OF_INERTIA`, `PRESSURE`, `STIFFNESS`, `DAMPING`, `VOLTAGE`, `RESISTANCE`, `INDUCTANCE`, `CAPACITANCE`, `CHARGE`, `MAGNETIC_FLUX`.

## 9. Differentiation and Integration

Symplex provides two levels of typed calculus:

1. **`DiffWrt` / `IntWrt` traits** — the recommended approach. Call `.diff_wrt()` / `.integrate_wrt()` directly on named types. The compiler verifies the physical law (e.g., d(Length)/d(Time) = Velocity) and returns the correct named type. No manual wrapping needed.

2. **`diff_qty` / `integrate_qty` functions** — the generic fallback. Work on `Qty<D>` and compute output dimensions via typenum arithmetic. Useful when no `DiffWrt`/`IntWrt` impl exists for your pair.

### The DiffWrt and IntWrt Traits (Recommended)

The `DiffWrt<Var>` trait encodes "differentiating `Self` with respect to `Var` produces `Output`." The `IntWrt<Var>` trait is the integration counterpart. Both operate directly on named types:

```rust
use symplex::prelude::*;
use symplex::units::*;

symplex::vars!(a, t);

// Typed differentiation: the compiler verifies d(Length)/d(Time) = Velocity
let position = Length::from_ex(expr!(1/2 * a * t^2));
let t_var = Time::symbol("t");
let velocity: Velocity = position.diff_wrt(&t_var);
let acceleration: Acceleration = velocity.diff_wrt(&t_var);

// Typed integration: ∫ Acceleration dt → Velocity
let v_back: Velocity = acceleration.integrate_wrt(&t_var);
```

No `.as_qty()`, no `.into()`, no manual dimension arithmetic. The compiler does all the work.

### Available DiffWrt Pairs

Every row below is a compile-time-checked differentiation rule:

| Expression type | d/d(Variable) | → | Result type | Physical law |
|----------------|---------------|---|-------------|--------------|
| `Length` | `Time` | → | `Velocity` | v = dx/dt |
| `Velocity` | `Time` | → | `Acceleration` | a = dv/dt |
| `Angle` | `Time` | → | `AngularVelocity` | ω = dθ/dt |
| `AngularVelocity` | `Time` | → | `AngularAcceleration` | α = dω/dt |
| `Energy` | `Time` | → | `Power` | P = dE/dt |
| `Momentum` | `Time` | → | `Force` | F = dp/dt (Newton's 2nd law) |
| `AngularMomentum` | `Time` | → | `Torque` | τ = dL/dt |
| `Charge` | `Time` | → | `Current` | I = dQ/dt |
| `MagneticFlux` | `Time` | → | `Voltage` | V = -dΦ/dt (Faraday's law) |
| `Energy` | `Length` | → | `Force` | F = -dU/dx |
| `Energy` | `Angle` | → | `Torque` | τ = -dU/dθ |
| `Momentum` | `Length` | → | `Stiffness` | dp/dx (wave context) |
| `Force` | `Length` | → | `Stiffness` | k = dF/dx |
| `Energy` | `Velocity` | → | `Momentum` | p = ∂T/∂v (Lagrangian) |
| `Energy` | `AngularVelocity` | → | `AngularMomentum` | L = ∂T/∂ω̇ (Lagrangian) |
| `Power` | `Current` | → | `Voltage` | V = dP/dI |
| `Power` | `Voltage` | → | `Current` | I = dP/dV |

### Available IntWrt Pairs

Every row is a compile-time-checked integration rule (reverse of the above):

| Integrand type | ∫ d(Variable) | → | Result type | Physical law |
|---------------|---------------|---|-------------|--------------|
| `Velocity` | `Time` | → | `Length` | x = ∫v dt |
| `Acceleration` | `Time` | → | `Velocity` | v = ∫a dt |
| `AngularVelocity` | `Time` | → | `Angle` | θ = ∫ω dt |
| `AngularAcceleration` | `Time` | → | `AngularVelocity` | ω = ∫α dt |
| `Power` | `Time` | → | `Energy` | E = ∫P dt |
| `Force` | `Time` | → | `Momentum` | p = ∫F dt (impulse) |
| `Torque` | `Time` | → | `AngularMomentum` | L = ∫τ dt |
| `Current` | `Time` | → | `Charge` | Q = ∫I dt |
| `Voltage` | `Time` | → | `MagneticFlux` | Φ = ∫V dt |
| `Force` | `Length` | → | `Energy` | W = ∫F dx (work) |
| `Stiffness` | `Length` | → | `Force` | F = ∫k dx |
| `Momentum` | `Velocity` | → | `Energy` | KE = ∫p dv |
| `AngularMomentum` | `AngularVelocity` | → | `Energy` | KE = ∫L dω |

### Lagrangian Mechanics Example

The `DiffWrt` pairs for `Energy` w.r.t. generalized coordinates and velocities map directly to the Euler-Lagrange equations:

```rust
use symplex::prelude::*;
use symplex::units::*;

symplex::vars!(m, l, g, theta, theta_dot);

let theta_var = Angle::symbol("theta");
let theta_dot_var = AngularVelocity::symbol("theta_dot");

// Build Lagrangian with expr!
let ke = Energy::from_ex(expr!(1/2 * m * l^2 * theta_dot^2));
let pe = Energy::from_ex(expr!(m * g * l * (1 - cos(theta))));
let lagrangian: Energy = &ke - &pe;

// ∂L/∂θ̇ → AngularMomentum (generalized momentum)
let p_theta: AngularMomentum = lagrangian.diff_wrt(&theta_dot_var);

// ∂L/∂θ → Torque (generalized force)
let tau: Torque = lagrangian.diff_wrt(&theta_var);
```

### Generic diff_qty / integrate_qty (Fallback)

When no `DiffWrt`/`IntWrt` impl exists for a particular pair, use the generic functions. These work on `Qty<D>` and compute dimensions via typenum:

```rust
use symplex::prelude::*;
use symplex::units::*;

let t_sym = symplex::var("t");

// Position: x(t) = ½ a t²
let x: Qty<LengthDim> = Qty::from_ex(expr!(1/2 * a * t^2));
let t: Qty<TimeDim> = Qty::from_ex(t_sym);

// Differentiate: v = dx/dt
let v: Velocity = diff_qty(&x, &t).into();
```

Integration works the same way:

```rust
use symplex::prelude::*;
use symplex::units::*;

let f: Qty<ForceDim> = Qty::from_ex(symplex::var("F"));
let x: Qty<LengthDim> = Qty::from_ex(symplex::var("x"));

// ∫ F dx → Force × Length = Energy
let w: Energy = integrate_qty(&f, &x).into();
```

### Fundamental Theorem of Calculus Roundtrip

Differentiation and integration are inverses — and the type system proves it:

```rust
use symplex::prelude::*;
use symplex::units::*;

symplex::vars!(a, t);
let t_var = Time::symbol("t");

let accel = Acceleration::from_ex(expr!(a));

// ∫ Acceleration dt → Velocity
let velocity: Velocity = accel.integrate_wrt(&t_var);

// d(Velocity)/dt → Acceleration
let accel_back: Acceleration = velocity.diff_wrt(&t_var);

// The type system proves: Acceleration →(×Time)→ Velocity →(÷Time)→ Acceleration
```

## 10. Escape Hatches

Sometimes you need to break out of the dimensional type system. Symplex provides three controlled escape hatches.

### into_inner() — Drop Dimensions

Every named type and `Qty<D>` has `into_inner()`, which returns the raw `Ex`:

```rust
use symplex::prelude::*;
use symplex::units::*;

let f = Force::symbol("F");
let raw: Ex = f.into_inner();  // now just a plain Ex, no dimension tracking

// Also works on Qty
let q: Qty<ForceDim> = Qty::from_ex(symplex::var("F"));
let raw2: Ex = q.into_inner();
```

Use `inner()` to borrow without consuming:

```rust
use symplex::prelude::*;
use symplex::units::*;

let f = Force::symbol("F");
let inner_ref: &Ex = f.inner();
println!("{inner_ref}"); // F
```

### assume_dimension::<D>(ex) — Re-Enter with a Specific Dimension

When you have a raw `Ex` and know its dimension, use `assume_dimension`:

```rust
use symplex::prelude::*;
use symplex::units::*;

let raw_expr: Ex = symplex::var("F_external");

// "I promise this expression has dimension Force"
let f: Qty<ForceDim> = assume_dimension::<ForceDim>(raw_expr);

// Convert to named type
let force: Force = f.into();
```

This is **unchecked** — the compiler trusts you. Use it at module boundaries where dimensional information is lost (e.g., reading from a file or calling into non-dimensioned code).

### Qty::from_ex(ex) — Generic Wrapping

Similar to `assume_dimension`, but the dimension is inferred from context:

```rust
use symplex::prelude::*;
use symplex::units::*;

// The type annotation tells Rust which dimension to use
let f: Qty<ForceDim> = Qty::from_ex(symplex::var("F"));
```

### When to Use Each

| Situation | Escape hatch |
|-----------|-------------|
| Pass dimensioned value to non-dimension-aware function | `into_inner()` |
| Peek at the expression without consuming | `inner()` |
| Import raw `Ex` from external source with known dimension | `assume_dimension::<D>(ex)` |
| Build a `Qty` where the dimension is clear from context | `Qty::from_ex(ex)` |
| Convert named type to named type via `Ex` round-trip | `NamedType::from_ex(qty.into_inner())` |

## 11. Real-World Examples

### Example A: Ohm's Law and Power Dissipation

```rust
use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let i = Current::symbol("I");
    let r = Resistance::symbol("R");

    // Ohm's law: V = IR (named Mul: Current × Resistance → Voltage)
    let v: Voltage = &i * &r;
    println!("V = IR: {v}"); // I*R [V]

    // Power: P = IV (named Mul: Current × Voltage → Power)
    let p: Power = &i * &v;
    println!("P = IV: {p}"); // I*V [W]

    // Typed calculus: dP/dI → Voltage
    let dp_di: Voltage = p.diff_wrt(&i);
    println!("dP/dI: {dp_di}"); // 2*I*R [V]

    // Compile-time formula check
    symplex::const_assert_dim!(
        ConstDim::CURRENT.mul(ConstDim::RESISTANCE),
        ConstDim::VOLTAGE,
        "Ohm's law: I × R must equal V"
    );

    symplex::const_assert_dim!(
        ConstDim::VOLTAGE.mul(ConstDim::CURRENT),
        ConstDim::POWER,
        "P = IV: Voltage × Current must equal Power"
    );
}
```

### Example B: Simple Pendulum Lagrangian

```rust
use symplex::prelude::*;
use symplex::units::*;

fn main() {
    symplex::vars!(m, l, g, theta, theta_dot);

    let theta_var = Angle::symbol("theta");
    let theta_dot_var = AngularVelocity::symbol("theta_dot");

    // Build energies with expr! — natural math syntax
    let ke = Energy::from_ex(expr!(1/2 * m * l^2 * theta_dot^2));
    let pe = Energy::from_ex(expr!(m * g * l * (1 - cos(theta))));
    println!("T = {ke}");
    println!("V = {pe}");

    // Lagrangian: Energy - Energy = Energy (dimension checked!)
    let lagrangian: Energy = &ke - &pe;
    println!("L = T - V = {lagrangian}");

    // Typed calculus via DiffWrt:
    // ∂L/∂θ̇ → AngularMomentum (generalized momentum)
    let p_theta: AngularMomentum = lagrangian.diff_wrt(&theta_dot_var);
    println!("∂L/∂θ̇ = {p_theta}");

    // ∂L/∂θ → Torque (generalized force)
    let tau: Torque = lagrangian.diff_wrt(&theta_var);
    println!("∂L/∂θ = {tau}");
}
```

### Example C: Motor Back-EMF Circuit

```rust
use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let r = Resistance::symbol("R");
    let l_ind = Inductance::symbol("L");
    let i = Current::symbol("I");
    let omega = AngularVelocity::symbol("omega");
    let ke = MagneticFlux::symbol("Ke"); // back-EMF constant [Wb = V·s/rad]

    // Resistive voltage drop: V_R = IR
    let v_r: Voltage = &i * &r;
    println!("V_R = IR: {v_r}"); // I*R [V]

    // Back-EMF voltage: V_emf = Ke × ω
    // MagneticFlux × AngularVelocity = Voltage (in the named table!)
    let v_emf: Voltage = &ke * &omega;
    println!("V_emf = Ke·ω: {v_emf}"); // Ke*omega [V]

    // Inductive voltage: V_L = L × dI/dt
    // Use diff_qty for the dI/dt part
    let i_qty = i.as_qty();
    let t_qty = Time::symbol("t").as_qty();
    let di_dt = diff_qty(&i_qty, &t_qty);
    let l_qty = l_ind.as_qty();
    let v_l_qty = &l_qty * &di_dt;
    let v_l: Voltage = v_l_qty.into();
    println!("V_L = L·dI/dt: {v_l}"); // L*Derivative(I, t) [V]

    // Kirchhoff's voltage law: V_supply = V_R + V_emf + V_L
    let v_total: Voltage = &v_r + &(&v_emf + &v_l);
    println!("V = IR + Ke·ω + L·dI/dt: {v_total}");

    // Faraday's law via DiffWrt: d(MagneticFlux)/d(Time) → Voltage
    let flux = MagneticFlux::symbol("Phi");
    let t_var = Time::symbol("t");
    let induced_emf: Voltage = flux.diff_wrt(&t_var);
    println!("V_induced = dΦ/dt: {induced_emf}");
}
```

## 12. Dimension Collision Handling

Some physically distinct quantities share the same SI dimension vector. Symplex handles these with distinct named types and explicit conversion methods.

### Energy vs. Torque

Energy (J = kg·m²/s²) and Torque (N·m = kg·m²/s²) have identical SI dimensions. They are separate Rust types:

```rust
use symplex::prelude::*;
use symplex::units::*;

let e = Energy::symbol("E");
let tau = Torque::symbol("tau");

// Cannot add Energy + Torque directly:
// let nonsense = e + tau;  // ERROR: expected `Energy`, found `Torque`

// Convert explicitly:
let tau_from_e = Torque::from_energy(e.clone());
let e_from_tau: Energy = tau.into();
```

The `From` trait works bidirectionally:

```rust
use symplex::prelude::*;
use symplex::units::*;

let e = Energy::symbol("E");
let tau: Torque = e.into();       // Energy → Torque via From

let tau2 = Torque::symbol("tau");
let e2: Energy = tau2.into();     // Torque → Energy via From
```

When converting from `Qty<EnergyDim>`, the primary type `Energy` is returned (not `Torque`). To get `Torque`, use the named constructor:

```rust
use symplex::prelude::*;
use symplex::units::*;

let q: Qty<EnergyDim> = Qty::from_ex(symplex::var("x"));
let e: Energy = q.clone().into();              // Qty → Energy (primary)
let tau = Torque::from_energy(q.into());       // Qty → Energy → Torque
```

### Frequency vs. AngularVelocity

Both have dimension T⁻¹. Same pattern:

```rust
use symplex::prelude::*;
use symplex::units::*;

let f = AngularVelocity::symbol("omega");
let freq = Frequency::from_angular_velocity(f);

let f2 = Frequency::symbol("f");
let omega: AngularVelocity = f2.into();
```

`AngularVelocity` is the primary type (it gets `From<Qty<…>>`). `Frequency` uses `Frequency::from_angular_velocity()`.

### Dimensionless vs. Angle

Both have all-zero dimension vectors:

```rust
use symplex::prelude::*;
use symplex::units::*;

let d = Dimensionless::symbol("ratio");
let theta = Angle::from_dimensionless(d);  // explicit conversion

// Dimensionless is the primary type
let q: Qty<DimensionlessDim> = Qty::from_ex(symplex::var("x"));
let d2: Dimensionless = q.into();  // Qty → Dimensionless (primary)
```

### Summary of Collision Groups

| Primary type | Secondary type | Shared dimension | Secondary constructor |
|-------------|---------------|-----------------|----------------------|
| `Dimensionless` | `Angle` | all zeros | `Angle::from_dimensionless(d)` |
| `Energy` | `Torque` | L²·M·T⁻² | `Torque::from_energy(e)` |
| `AngularVelocity` | `Frequency` | T⁻¹ | `Frequency::from_angular_velocity(w)` |

## 13. Error Message Guide

Here are the five most common dimension errors you'll encounter, with the actual compiler output.

### Error 1: Adding Different Named Types

```rust
// let m = Mass::symbol("m");
// let l = Length::symbol("l");
// let bad = m + l;
```

```text
error[E0308]: mismatched types
  --> src/main.rs:5:19
   |
5  |     let bad = m + l;
   |                   ^ expected `Mass`, found `Length`
```

**Fix:** You can only add quantities of the same type. Check your formula.

### Error 2: Wrong Type Annotation on Multiplication

```rust
// let m = Mass::symbol("m");
// let a = Acceleration::symbol("a");
// let v: Velocity = m * a;
```

```text
error[E0308]: mismatched types
  --> src/main.rs:5:23
   |
5  |     let v: Velocity = m * a;
   |            --------   ^^^^^ expected `Velocity`, found `Force`
```

**Fix:** `Mass × Acceleration = Force`, not `Velocity`. Correct your type annotation or your formula.

### Error 3: Trig on Non-Angle Type

```rust
// let l = Length::symbol("L");
// let s = l.sin();
```

```text
error[E0599]: no method named `sin` found for struct `Length` in the current scope
  --> src/main.rs:4:16
   |
4  |     let s = l.sin();
   |                ^^^ method not found in `Length`
```

**Fix:** `sin()`, `cos()`, and `tan()` are only available on `Angle`. Convert your value to an angle first, or check your formula.

### Error 4: Adding Qty with Different Dimensions

```rust
// let a: Qty<LengthDim> = Qty::from_ex(symplex::var("a"));
// let b: Qty<MassDim> = Qty::from_ex(symplex::var("b"));
// let c = a + b;
```

```text
error[E0277]: cannot add or subtract quantities with different physical dimensions
  --> src/main.rs:5:17
   |
5  |     let c = a + b;
   |                 ^ incompatible physical dimension
   |
   = note: addition and subtraction require both sides to have the same dimension
   = note: use .into_inner() to drop dimension tracking
```

**Fix:** The `#[diagnostic::on_unimplemented]` attribute on `SameDim` produces this clear message. Either fix the formula or use `into_inner()` if you're intentionally mixing.

### Error 5: Wrong Dimension in assert_dim!

```rust
// let f = Force::symbol("F");
// let checked = symplex::assert_dim!(f, Voltage);
```

```text
error[E0308]: mismatched types
  --> src/main.rs:4:42
   |
4  |     let checked = symplex::assert_dim!(f, Voltage);
   |                                        ^ expected `Voltage`, found `Force`
```

**Fix:** The expression has a different dimension than you expected. The assertion caught a bug in your reasoning — check the formula.

## 14. Available Methods on Named Types

Every named quantity type (`Force`, `Velocity`, `Energy`, etc.) and `Qty<D>` provides these methods. All preserve the dimension type:

| Method | Description |
|--------|-------------|
| `diff(&self, var: &Ex) → Ex` | Symbolic differentiation (returns raw `Ex`) |
| `integrate(&self, var: &Ex) → Ex` | Symbolic integration (returns raw `Ex`) |
| `diff_wrt(&self, var: &T) → Output` | Typed differentiation (returns named type, requires `DiffWrt` impl) |
| `integrate_wrt(&self, var: &T) → Output` | Typed integration (returns named type, requires `IntWrt` impl) |
| `simplify(self)` | Basic simplification |
| `simplify_full(self)` | Multi-pass aggressive simplification |
| `simplify_trig(self)` | Trigonometric identity simplification |
| `simplify_powers(self)` | Power/exponent simplification |
| `expand(self)` | Algebraic expansion |
| `expand_trig(self)` | Trig expansion (e.g., sin(a+b) → sin(a)cos(b)+cos(a)sin(b)) |
| `factor(&self, var: &Ex)` | Factor a polynomial w.r.t. a variable |
| `collect(&self, var: &Ex)` | Collect terms by variable |
| `cancel(&self, var: &Ex)` | Cancel common polynomial factors |
| `together(self)` | Combine fractions over a common denominator |
| `eval(self)` | Evaluate constants symbolically |
| `eval_f64(&self)` | Evaluate to `f64` (all symbols must be eliminated) |
| `eval_f64_with(&self, subs: &[(&Ex, i64)])` | Substitute integer values and evaluate to `f64` |
| `subs(&self, var: &Ex, val: &Ex)` | Substitute a variable with a value |
| `to_latex(&self) → String` | Render as LaTeX |
| `inner(&self) → &Ex` | Borrow the inner expression |
| `into_inner(self) → Ex` | Consume and return the inner expression |

## 15. Limitations and Future Work

### Current Limitations

**No mixed-dimension matrices.** Symplex's `Matrix` type holds plain `Ex` values. You cannot (yet) have a matrix where row 1 is forces and row 2 is torques with compile-time dimension tracking per element. For now, build dimensioned values individually and use `into_inner()` when constructing matrices.

**No symbolic powi on Qty.** Raising a `Qty<D>` to a power requires knowing the exponent at the type level. You can't write `length.powi(2)` and get `Qty<AreaDim>` — the exponent must be in the type, not a runtime value. Workaround: multiply explicitly (`&l * &l`) or drop to `Ex` with `into_inner()`.

**No runtime dimension tracking.** All dimension checking happens at compile time via the type system. In release builds, `Qty<D>` is zero-cost — the `PhantomData<D>` field is erased entirely. This means you cannot inspect dimensions at runtime or dynamically dispatch on them.

**Amount of substance and luminous intensity.** The dimension vector carries slots for amount (N, mol) and luminous intensity (J, cd), but no named types or conversions use them yet.

### Planned Features

**Codegen with uom types.** The code generation system (Chapter 8) will gain an option to emit functions that accept and return `uom` crate types, bridging symplex's symbolic world with `uom`'s runtime units.

**Matrix dimension tracking.** A `DimMatrix<R, C, Dims>` type that tracks per-element dimensions, enabling type-safe state-space models where the state vector mixes positions, velocities, and angles.

**Symbolic powi.** A `qty.pow::<N>()` method using typenum to compute the output dimension at compile time: `Length::symbol("x").pow::<P2>()` → `Area`.

**More named types.** Density, specific heat, viscosity, electric field, and other commonly-needed quantities.

**More DiffWrt/IntWrt pairs.** Additional typed calculus rules as use cases arise (e.g., `Torque` w.r.t. `Angle` → `Energy`).

---

*[← Chapter 20: Finite Differences](20-finite-differences.md) | [Back to Table of Contents](index.md)*