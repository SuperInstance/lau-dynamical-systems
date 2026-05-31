//! Continuous dynamical systems (ODE integration).

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// A continuous dynamical system defined by an ODE right-hand side.
pub trait ContinuousSystem: Send + Sync {
    /// Dimension of the state vector.
    fn dim(&self) -> usize;

    /// Evaluate the vector field f(x, t).
    fn f(&self, x: &DVector<f64>, t: f64) -> DVector<f64>;

    /// Jacobian df/dx. Default: finite-difference approximation.
    fn jacobian(&self, x: &DVector<f64>, t: f64) -> DMatrix<f64> {
        let n = self.dim();
        let eps = 1e-7;
        let fx = self.f(x, t);
        let mut jac = DMatrix::zeros(n, n);
        for j in 0..n {
            let mut xp = x.clone();
            xp[j] += eps;
            let fxp = self.f(&xp, t);
            for i in 0..n {
                jac[(i, j)] = (fxp[i] - fx[i]) / eps;
            }
        }
        jac
    }
}

/// 4th-order Runge-Kutta integrator result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationResult {
    pub times: Vec<f64>,
    pub states: Vec<DVector<f64>>,
}

/// Integrate a continuous system using classical RK4.
pub fn rk4<S: ContinuousSystem>(
    system: &S,
    x0: &DVector<f64>,
    t0: f64,
    t1: f64,
    dt: f64,
) -> IntegrationResult {
    let mut times = Vec::new();
    let mut states = Vec::new();
    let mut x = x0.clone();
    let mut t = t0;

    times.push(t);
    states.push(x.clone());

    while t < t1 - 1e-12 {
        let step = dt.min(t1 - t);
        let k1 = system.f(&x, t);
        let k2 = system.f(&(&x + &k1 * (step / 2.0)), t + step / 2.0);
        let k3 = system.f(&(&x + &k2 * (step / 2.0)), t + step / 2.0);
        let k4 = system.f(&(&x + &k3 * step), t + step);
        x += &(&k1 * step / 6.0 + &k2 * step / 3.0 + &k3 * step / 3.0 + &k4 * step / 6.0);
        t += step;
        times.push(t);
        states.push(x.clone());
    }

    IntegrationResult { times, states }
}

// ---------------------------------------------------------------------------
// Lorenz system
// ---------------------------------------------------------------------------

/// Lorenz system parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LorenzParams {
    pub sigma: f64,
    pub rho: f64,
    pub beta: f64,
}

impl Default for LorenzParams {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        }
    }
}

/// The classic Lorenz system.
pub struct LorenzSystem {
    pub params: LorenzParams,
}

impl LorenzSystem {
    pub fn new(params: LorenzParams) -> Self {
        Self { params }
    }

    /// Canonical chaotic parameters.
    pub fn chaotic() -> Self {
        Self::new(LorenzParams::default())
    }
}

impl ContinuousSystem for LorenzSystem {
    fn dim(&self) -> usize { 3 }

    fn f(&self, x: &DVector<f64>, _t: f64) -> DVector<f64> {
        let (s, r, b) = (self.params.sigma, self.params.rho, self.params.beta);
        DVector::from_vec(vec![
            s * (x[1] - x[0]),
            x[0] * (r - x[2]) - x[1],
            x[0] * x[1] - b * x[2],
        ])
    }

    fn jacobian(&self, x: &DVector<f64>, _t: f64) -> DMatrix<f64> {
        let (s, r, b) = (self.params.sigma, self.params.rho, self.params.beta);
        DMatrix::from_row_slice(3, 3, &[
            -s, s, 0.0,
            r - x[2], -1.0, -x[0],
            x[1], x[0], -b,
        ])
    }
}

// ---------------------------------------------------------------------------
// Lotka-Volterra system
// ---------------------------------------------------------------------------

/// Lotka-Volterra predator-prey parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LotkaVolterraParams {
    pub alpha: f64, // prey growth
    pub beta: f64,  // prey predation
    pub gamma: f64, // predator death
    pub delta: f64, // predator growth from prey
}

impl Default for LotkaVolterraParams {
    fn default() -> Self {
        Self {
            alpha: 1.1,
            beta: 0.4,
            gamma: 0.4,
            delta: 0.1,
        }
    }
}

/// Lotka-Volterra predator-prey system.
pub struct LotkaVolterraSystem {
    pub params: LotkaVolterraParams,
}

impl Default for LotkaVolterraSystem {
    fn default() -> Self {
        Self::new(LotkaVolterraParams::default())
    }
}

impl LotkaVolterraSystem {
    pub fn new(params: LotkaVolterraParams) -> Self {
        Self { params }
    }
}

impl ContinuousSystem for LotkaVolterraSystem {
    fn dim(&self) -> usize { 2 }

    fn f(&self, x: &DVector<f64>, _t: f64) -> DVector<f64> {
        let (a, b, g, d) = (self.params.alpha, self.params.beta, self.params.gamma, self.params.delta);
        DVector::from_vec(vec![
            a * x[0] - b * x[0] * x[1],
            d * x[0] * x[1] - g * x[1],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_lorenz_origin_is_fixed_point() {
        let sys = LorenzSystem::chaotic();
        let x0 = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let f = sys.f(&x0, 0.0);
        assert_abs_diff_eq!(f.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lorenz_nonzero_fixed_points() {
        let sys = LorenzSystem::chaotic();
        let r = sys.params.rho;
        let b = sys.params.beta;
        let c = (b * (r - 1.0)).sqrt();
        let x = DVector::from_vec(vec![c, c, r - 1.0]);
        let f = sys.f(&x, 0.0);
        assert_abs_diff_eq!(f.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rk4_lorenz_trajectory_bounded() {
        let sys = LorenzSystem::chaotic();
        let x0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let result = rk4(&sys, &x0, 0.0, 10.0, 0.01);
        // Chaotic trajectory should stay bounded (z < 100 for canonical params)
        for s in &result.states {
            assert!(s[2].abs() < 100.0, "z exceeded 100: {}", s[2]);
        }
    }

    #[test]
    fn test_lotka_volterra_preserves_equilibrium() {
        let sys = LotkaVolterraSystem::default();
        let (a, b, g, d) = (sys.params.alpha, sys.params.beta, sys.params.gamma, sys.params.delta);
        let x_eq = DVector::from_vec(vec![g / d, a / b]);
        let f = sys.f(&x_eq, 0.0);
        assert_abs_diff_eq!(f.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lotka_volterra_cycles() {
        let sys = LotkaVolterraSystem::default();
        let x0 = DVector::from_vec(vec![10.0, 5.0]);
        let result = rk4(&sys, &x0, 0.0, 20.0, 0.001);
        // Populations should remain positive
        for s in &result.states {
            assert!(s[0] > 0.0, "prey went negative");
            assert!(s[1] > 0.0, "predator went negative");
        }
    }

    #[test]
    fn test_rk4_conserves_length_trivial() {
        // A trivial system dx/dt = 0
        struct ZeroSystem;
        impl ContinuousSystem for ZeroSystem {
            fn dim(&self) -> usize { 2 }
            fn f(&self, _x: &DVector<f64>, _t: f64) -> DVector<f64> {
                DVector::zeros(2)
            }
        }
        let x0 = DVector::from_vec(vec![3.0, 4.0]);
        let result = rk4(&ZeroSystem, &x0, 0.0, 10.0, 0.1);
        assert_abs_diff_eq!(result.states.last().unwrap().norm(), 5.0, epsilon = 1e-10);
    }
}
