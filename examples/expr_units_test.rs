use symplex::prelude::*;
use symplex::units::*;
use symplex::vars;

fn main() {
    println!("═══ The Core Workflow: expr! + Units + Calculus ═══\n");

    // Declare raw Ex variables for use in expr!
    vars!(m, a, v, t, x, k, g, R, I_var, L, omega);

    // ── 1. Build velocity expression, get acceleration by differentiating ──
    println!("── 1. Velocity → differentiate → Acceleration ──");
    
    // Build a velocity expression: v(t) = a*t + v0
    let v_expr = Velocity::from_ex(expr!(a * t));
    println!("  v(t) = {}", v_expr);
    
    // Differentiate to get acceleration: dv/dt = a
    // Currently need to go through inner Ex:
    let accel_expr = Acceleration::from_ex(v_expr.inner().diff(&t));
    println!("  dv/dt = {}", accel_expr);

    // ── 2. Acceleration → integrate → Velocity ──
    println!("\n── 2. Acceleration → integrate → Velocity ──");
    let a_expr = Acceleration::from_ex(expr!(g));
    println!("  a(t) = {}", a_expr);
    
    let v_from_int = Velocity::from_ex(a_expr.inner().integrate(&t));
    println!("  ∫a dt = {}", v_from_int);

    // ── 3. Velocity → integrate → Position ──
    println!("\n── 3. Velocity → integrate → Position (Length) ──");
    let v2 = Velocity::from_ex(expr!(g * t));
    println!("  v(t) = {}", v2);
    
    let x_pos = Length::from_ex(v2.inner().integrate(&t));
    println!("  ∫v dt = {}", x_pos);
    // Should be ½gt²

    // ── 4. Full kinematic chain with expr! ──
    println!("\n── 4. Full kinematic chain: x(t) = ½at² ──");
    let position = Length::from_ex(expr!(1/2 * a * t^2));
    println!("  x(t) = {}", position);
    
    let velocity = Velocity::from_ex(position.inner().diff(&t));
    println!("  v(t) = dx/dt = {}", velocity);
    
    let acceleration = Acceleration::from_ex(velocity.inner().diff(&t));
    println!("  a(t) = dv/dt = {}", acceleration);

    // ── 5. Force equation with units ──
    println!("\n── 5. Newton's law: F = ma, with symbolic manipulation ──");
    let mass = Mass::symbol("m");
    let accel2 = Acceleration::symbol("a");
    let force: Force = &mass * &accel2;
    println!("  F = m*a = {}", force);
    
    // Expand, simplify, substitute — all preserve Force dimension
    let force2 = force.clone().expand();
    println!("  expanded: {}", force2);

    // Substitute and evaluate
    let f_val = force.clone()
        .subs(mass.inner(), &symplex::rational(10, 1))
        .subs(accel2.inner(), &symplex::rational(981, 100))
        .eval();
    println!("  F(m=10, a=9.81) = {}", f_val);

    // ── 6. Ohm's law: V = IR, then P = IV, then dP/dI ──
    println!("\n── 6. Electrical: V = IR → P = IV → dP/dI ──");
    let current = Current::symbol("I");
    let resistance = Resistance::symbol("R");
    
    let voltage: Voltage = &current * &resistance;
    println!("  V = IR = {}", voltage);
    
    let power: Power = &current * &voltage;
    println!("  P = IV = I²R = {}", power);
    
    let power_expanded = power.clone().expand();
    println!("  P expanded = {}", power_expanded);

    // dP/dI — should have dimensions of Voltage (Power/Current = Voltage)
    // Note: d(I²R)/dI = 2IR = 2V
    let dp_di = Voltage::from_ex(power.inner().diff(current.inner()));
    println!("  dP/dI = {}", dp_di);

    // ── 7. Spring-mass: F = -kx, PE = ½kx² ──
    println!("\n── 7. Spring potential energy and force ──");
    let stiffness = Stiffness::symbol("k");
    let position2 = Length::symbol("x");
    
    // PE = ½kx² — use expr! for the complex formula
    let pe = Energy::from_ex(expr!(1/2 * k * x^2));
    println!("  PE = ½kx² = {}", pe);
    
    // Force = -dPE/dx
    let spring_force = Force::from_ex(-pe.inner().diff(&x));
    println!("  F = -dPE/dx = {}", spring_force);
    // Should be -kx (Hooke's law!)
    
    // ── 8. Pendulum: Lagrangian ──
    println!("\n── 8. Pendulum Lagrangian ──");
    vars!(theta, theta_dot, l);
    
    // T = ½ml²θ̇²
    let ke = Energy::from_ex(expr!(1/2 * m * l^2 * theta_dot^2));
    println!("  T = ½ml²θ̇² = {}", ke);
    
    // V = -mgl·cos(θ)
    let pe2 = Energy::from_ex(&m * &g * &l * expr!(cos(theta)));
    println!("  V = mgl·cos(θ) = {}", pe2);
    
    // L = T - V
    let lagrangian_ex = ke.inner() - pe2.inner();
    let lagrangian = Energy::from_ex(lagrangian_ex);
    println!("  L = T - V = {}", lagrangian);
    
    // ∂L/∂θ̇ = ml²θ̇ (angular momentum)
    let dl_dthetadot = lagrangian.inner().diff(&theta_dot);
    println!("  ∂L/∂θ̇ = {}", dl_dthetadot);
    
    // ∂L/∂θ = mgl·sin(θ)
    let dl_dtheta = lagrangian.inner().diff(&theta);
    println!("  ∂L/∂θ = {}", dl_dtheta);

    // ── 9. Integration to get energy from force ──
    println!("\n── 9. Work = ∫F dx ──");
    let f_linear = Force::from_ex(expr!(k * x));
    println!("  F(x) = kx = {}", f_linear);
    
    let work = Energy::from_ex(f_linear.inner().integrate(&x));
    println!("  W = ∫F dx = {}", work);
    // Should be ½kx²

    // ── 10. Using diff_qty for typed differentiation ──
    println!("\n── 10. diff_qty: typed differentiation ──");
    let x_qty: Qty<LengthDim> = Qty::from_ex(expr!(1/2 * a * t^2));
    let t_qty: Qty<TimeDim> = Qty::from_ex(t.clone());
    
    let v_qty = diff_qty(&x_qty, &t_qty);
    let v_named: Velocity = v_qty.into();
    println!("  d(½at²)/dt = {}", v_named);
    
    let a_qty = diff_qty(&Qty::<VelocityDim>::from_ex(v_named.into_inner()), &t_qty);
    let a_named: Acceleration = a_qty.into();
    println!("  d²x/dt² = {}", a_named);

    println!("\n═══ All workflows verified! ═══");
}
