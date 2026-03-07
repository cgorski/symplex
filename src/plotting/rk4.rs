//! Fourth-order Runge–Kutta integrator for systems of ODEs.
//!
//! Provides [`rk4_integrate`] for numerically solving initial-value problems
//! of the form **dy/dt = f(t, y)** where **y** is a state vector.

#![allow(dead_code)]

/// Fourth-order Runge–Kutta integrator for systems of ODEs.
///
/// Integrates `dy/dt = f(t, y)` from `t_start` to `t_end` with fixed step
/// size `dt`. The state vector `y` can have any dimension.
///
/// # Arguments
///
/// * `f` — right-hand side function `f(t, y) -> dy/dt`
/// * `y0` — initial state vector at `t = t_start`
/// * `t_start` — start time
/// * `t_end` — end time (must be > `t_start`)
/// * `dt` — integration step size (must be > 0)
///
/// # Returns
///
/// A vector of `(t, y)` pairs at each time step, including the initial state.
///
/// # Panics
///
/// Panics if `dt <= 0` or `t_end <= t_start` or `y0` is empty.
pub(crate) fn rk4_integrate(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    y0: &[f64],
    t_start: f64,
    t_end: f64,
    dt: f64,
) -> Vec<(f64, Vec<f64>)> {
    assert!(dt > 0.0, "step size dt must be positive");
    assert!(t_end > t_start, "t_end must be greater than t_start");
    assert!(!y0.is_empty(), "initial state y0 must be non-empty");

    let n = y0.len();
    let num_steps = ((t_end - t_start) / dt).ceil() as usize;

    let mut result = Vec::with_capacity(num_steps + 1);
    let mut t = t_start;
    let mut y = y0.to_vec();

    result.push((t, y.clone()));

    for _ in 0..num_steps {
        // Clamp the last step so we don't overshoot t_end
        let h = dt.min(t_end - t);
        if h <= 0.0 {
            break;
        }

        let k1 = f(t, &y);

        // y + k1 * h/2
        let y_tmp: Vec<f64> = (0..n).map(|i| y[i] + k1[i] * h * 0.5).collect();
        let k2 = f(t + h * 0.5, &y_tmp);

        // y + k2 * h/2
        let y_tmp: Vec<f64> = (0..n).map(|i| y[i] + k2[i] * h * 0.5).collect();
        let k3 = f(t + h * 0.5, &y_tmp);

        // y + k3 * h
        let y_tmp: Vec<f64> = (0..n).map(|i| y[i] + k3[i] * h).collect();
        let k4 = f(t + h, &y_tmp);

        // y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        for i in 0..n {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;

        result.push((t, y.clone()));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_decay() {
        // dy/dt = -y, y(0) = 1  =>  y(t) = e^{-t}
        let result = rk4_integrate(&|_t, y| vec![-y[0]], &[1.0], 0.0, 1.0, 0.001);

        let last = result.last().unwrap();
        let expected = (-1.0_f64).exp(); // e^{-1} ≈ 0.36788
        assert!(
            (last.1[0] - expected).abs() < 1e-8,
            "expected y(1) ≈ {expected}, got {}",
            last.1[0]
        );
    }

    #[test]
    fn linear_growth() {
        // dy/dt = 1, y(0) = 0  =>  y(t) = t
        let result = rk4_integrate(&|_t, _y| vec![1.0], &[0.0], 0.0, 5.0, 0.1);

        let last = result.last().unwrap();
        assert!(
            (last.0 - 5.0).abs() < 1e-10,
            "expected t = 5.0, got {}",
            last.0
        );
        assert!(
            (last.1[0] - 5.0).abs() < 1e-8,
            "expected y(5) = 5.0, got {}",
            last.1[0]
        );
    }

    #[test]
    fn harmonic_oscillator() {
        // x'' = -x  =>  system: y0 = x, y1 = x'
        // dy0/dt = y1, dy1/dt = -y0
        // x(0) = 1, x'(0) = 0  =>  x(t) = cos(t)
        let result = rk4_integrate(
            &|_t, y| vec![y[1], -y[0]],
            &[1.0, 0.0],
            0.0,
            2.0 * std::f64::consts::PI,
            0.001,
        );

        let last = result.last().unwrap();
        // After one full period, x should be back to ~1, x' back to ~0
        assert!(
            (last.1[0] - 1.0).abs() < 1e-6,
            "expected x(2π) ≈ 1.0, got {}",
            last.1[0]
        );
        assert!(
            last.1[1].abs() < 1e-5,
            "expected x'(2π) ≈ 0.0, got {}",
            last.1[1]
        );
    }

    #[test]
    fn includes_initial_state() {
        let result = rk4_integrate(&|_t, y| vec![-y[0]], &[1.0], 0.0, 0.1, 0.1);
        assert_eq!(result[0].0, 0.0);
        assert_eq!(result[0].1, vec![1.0]);
        assert!(result.len() >= 2);
    }

    #[test]
    #[should_panic(expected = "step size dt must be positive")]
    fn zero_dt_panics() {
        rk4_integrate(&|_t, y| vec![-y[0]], &[1.0], 0.0, 1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "t_end must be greater than t_start")]
    fn bad_range_panics() {
        rk4_integrate(&|_t, y| vec![-y[0]], &[1.0], 1.0, 0.0, 0.1);
    }
}
