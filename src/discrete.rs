//! Discrete dynamical systems (iterated maps).

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// A discrete dynamical system (iterated map).
pub trait DiscreteSystem: Send + Sync {
    fn dim(&self) -> usize;
    fn step(&self, x: &DVector<f64>) -> DVector<f64>;
    fn jacobian(&self, x: &DVector<f64>) -> DMatrix<f64> {
        let n = self.dim();
        let eps = 1e-8;
        let fx = self.step(x);
        let mut jac = DMatrix::zeros(n, n);
        for j in 0..n {
            let mut xp = x.clone();
            xp[j] += eps;
            let fxp = self.step(&xp);
            for i in 0..n {
                jac[(i, j)] = (fxp[i] - fx[i]) / eps;
            }
        }
        jac
    }
}

/// Iterate a discrete system for n steps.
pub fn iterate<S: DiscreteSystem>(system: &S, x0: &DVector<f64>, n: usize) -> Vec<DVector<f64>> {
    let mut traj = Vec::with_capacity(n + 1);
    traj.push(x0.clone());
    let mut x = x0.clone();
    for _ in 0..n {
        x = system.step(&x);
        traj.push(x.clone());
    }
    traj
}

// ---------------------------------------------------------------------------
// Logistic map
// ---------------------------------------------------------------------------

/// Logistic map x_{n+1} = r * x_n * (1 - x_n).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticMap {
    pub r: f64,
}

impl LogisticMap {
    pub fn new(r: f64) -> Self {
        Self { r }
    }

    pub fn step_scalar(&self, x: f64) -> f64 {
        self.r * x * (1.0 - x)
    }
}

impl DiscreteSystem for LogisticMap {
    fn dim(&self) -> usize { 1 }
    fn step(&self, x: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![self.r * x[0] * (1.0 - x[0])])
    }
    fn jacobian(&self, x: &DVector<f64>) -> DMatrix<f64> {
        DMatrix::from_row_slice(1, 1, &[self.r * (1.0 - 2.0 * x[0])])
    }
}

// ---------------------------------------------------------------------------
// Hénon map
// ---------------------------------------------------------------------------

/// Hénon map parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HenonParams {
    pub a: f64,
    pub b: f64,
}

impl Default for HenonParams {
    fn default() -> Self {
        Self { a: 1.4, b: 0.3 }
    }
}

/// Hénon map.
pub struct HenonMap {
    pub params: HenonParams,
}

impl HenonMap {
    pub fn new(params: HenonParams) -> Self {
        Self { params }
    }

    pub fn chaotic() -> Self {
        Self::new(HenonParams::default())
    }
}

impl DiscreteSystem for HenonMap {
    fn dim(&self) -> usize { 2 }
    fn step(&self, x: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![
            1.0 - self.params.a * x[0] * x[0] + x[1],
            self.params.b * x[0],
        ])
    }
    fn jacobian(&self, x: &DVector<f64>) -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[
            -2.0 * self.params.a * x[0], 1.0,
            self.params.b, 0.0,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_logistic_map_fixed_point_r3() {
        // For r=3, the nonzero fixed point is x* = 1 - 1/r = 2/3
        let map = LogisticMap::new(3.0);
        let x_star = 2.0 / 3.0;
        let next = map.step_scalar(x_star);
        assert_abs_diff_eq!(next, x_star, epsilon = 1e-10);
    }

    #[test]
    fn test_logistic_map_period_doubling_r3_2() {
        // At r=3.2, the fixed point is unstable — orbit is period-2
        let map = LogisticMap::new(3.2);
        let x0 = 0.5;
        let mut x = x0;
        for _ in 0..1000 { x = map.step_scalar(x); }
        let x1 = map.step_scalar(x);
        let x2 = map.step_scalar(x1);
        // Should return to same value (period-2)
        assert_abs_diff_eq!(x, x2, epsilon = 1e-8);
        // And x != x1 (actually period-2)
        assert!((x - x1).abs() > 0.01);
    }

    #[test]
    fn test_logistic_map_known_values() {
        let map = LogisticMap::new(4.0);
        assert_abs_diff_eq!(map.step_scalar(0.5), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(map.step_scalar(0.25), 0.75, epsilon = 1e-12);
        assert_abs_diff_eq!(map.step_scalar(0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_logistic_map_chaos_r4() {
        // r=4 is fully chaotic on [0,1]; check sensitive dependence
        let map = LogisticMap::new(4.0);
        let x0 = 0.1;
        let x1 = 0.100001;
        let mut a = x0;
        let mut b = x1;
        for _ in 0..50 {
            a = map.step_scalar(a);
            b = map.step_scalar(b);
        }
        // Trajectories should have diverged significantly
        assert!((a - b).abs() > 0.01, "Should diverge: a={a}, b={b}");
    }

    #[test]
    fn test_henon_fixed_points() {
        // Hénon map fixed points satisfy x = 1 - a*x^2 + b*x, y = b*x
        let map = HenonMap::chaotic();
        // The fixed point: x = (-(b-1) - sqrt((b-1)^2 + 4a)) / (2a)
        let a = map.params.a;
        let b = map.params.b;
        let disc = (b - 1.0).powi(2) + 4.0 * a;
        let x_fp = ((b - 1.0) - disc.sqrt()) / (2.0 * a);
        let y_fp = b * x_fp;
        let x = DVector::from_vec(vec![x_fp, y_fp]);
        let next = map.step(&x);
        assert_abs_diff_eq!(next[0], x[0], epsilon = 1e-6);
        assert_abs_diff_eq!(next[1], x[1], epsilon = 1e-6);
    }

    #[test]
    fn test_henon_bounded_orbit() {
        let map = HenonMap::chaotic();
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let traj = iterate(&map, &x0, 5000);
        for s in &traj[100..] {
            assert!(s[0].abs() < 5.0);
            assert!(s[1].abs() < 5.0);
        }
    }

    #[test]
    fn test_iterate_count() {
        let map = LogisticMap::new(2.0);
        let x0 = DVector::from_vec(vec![0.5]);
        let traj = iterate(&map, &x0, 100);
        assert_eq!(traj.len(), 101);
    }
}
