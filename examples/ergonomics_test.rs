use symplex::prelude::*;
use symplex::units::*;
use symplex::vars;

fn main() {
    vars!(t);

    // THE DREAM: write velocity, diff to get acceleration, integrate to get position
    // Pattern A: Named types all the way
    let v = Velocity::from_ex(symplex::var("v_of_t"));
    let t_var = Time::symbol("t");

    // What I want: v.diff_wrt(&t_var) → Acceleration
    // What I have: Acceleration::from_ex(v.inner().diff(t_var.inner()))

    // Pattern B: expr! for building, from_ex for typing
    let v2 = Velocity::from_ex(expr!(a * t).clone());
    // Works! (with .clone() because expr! returns &Ex)

    // Pattern C: Build kinetic energy with expr!, differentiate to get momentum
    let ke = Energy::from_ex(expr!(1/2 * m * v^2).clone());
    let p = Momentum::from_ex(ke.inner().diff(&symplex::var("v")));
    println!("dKE/dv = {}", p);
    // Works!

    // Pattern D: Position → Velocity → Acceleration chain
    let x = Length::from_ex(expr!(1/2 * a * t^2).clone());
    let dx_dt = x.inner().diff(&t);  // Returns Ex
    let v3 = Velocity::from_ex(dx_dt.clone());
    let dv_dt = v3.inner().diff(&t);  // Returns Ex
    let a3 = Acceleration::from_ex(dv_dt);
    println!("x = {}", x);
    println!("dx/dt = {}", v3);
    println!("d²x/dt² = {}", a3);

    // Pattern E: Hooke's law from potential energy
    let pe = Energy::from_ex(expr!(1/2 * k * x^2).clone());
    let f = Force::from_ex(-pe.inner().diff(&symplex::var("x")));
    println!("F = -dPE/dx = {}", f);

    // Pattern F: Ohm's law with named types (simple product — perfect)
    let i_cur = Current::symbol("I");
    let r = Resistance::symbol("R");
    let v_ohm: Voltage = &i_cur * &r;
    println!("V = IR = {}", v_ohm);

    // Pattern G: Power expanded and differentiated
    let p_elec: Power = &i_cur * &v_ohm;
    let dp_di = p_elec.inner().diff(i_cur.inner());
    let dp_di_voltage = Voltage::from_ex(dp_di);
    println!("dP/dI = {}", dp_di_voltage);

    println!("\n✓ All patterns work!");
}
