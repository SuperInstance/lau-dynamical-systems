//! Strange attractors: Lyapunov spectrum and fractal dimension estimation.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::continuous::{ContinuousSystem, rk4};
use crate::discrete::DiscreteSystem;

/// Result of Lyapunov exponent computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyapunovSpectrum {
    /// Lyapunov exponents sorted largest first.
    pub exponents: Vec<f64>,
    /// Kaplan-Yorke (Lyapunov) dimension.
    pub ky_dimension: f64,
}

/// Compute the full Lyapunov spectrum for a continuous system using the
/// Benettin algorithm (variational equations with Gram-Schmidt reorthonormalization).
pub fn lyapunov_spectrum_continuous<S: ContinuousSystem>(
    system: &S,
    x0: &DVector<f64>,
    t0: f64,
    t1: f64,
    dt: f64,
    renorm_steps: usize,
) -> LyapunovSpectrum {
    let n = system.dim();
    let mut x = x0.clone();
    let mut q = DMatrix::identity(n, n); // orthonormal perturbation vectors
    let mut sums = vec![0.0_f64; n];
    let mut t = t0;
    let mut count = 0;

    while t < t1 - 1e-12 {
        let step = dt.min(t1 - t);

        // RK4 for the state
        let k1 = system.f(&x, t);
        let k2 = system.f(&(&x + &k1 * (step / 2.0)), t + step / 2.0);
        let k3 = system.f(&(&x + &k2 * (step / 2.0)), t + step / 2.0);
        let k4 = system.f(&(&x + &k3 * step), t + step);
        x += &(&k1 * (step / 6.0) + &k2 * (step / 3.0) + &k3 * (step / 3.0) + &k4 * (step / 6.0));

        // Evolve perturbation vectors: dQ/dt = J(x) * Q
        let jac = system.jacobian(&x, t);
        let k1q = &jac * &q;
        let k2q = &jac * &(&q + &k1q * (step / 2.0));
        let k3q = &jac * &(&q + &k2q * (step / 2.0));
        let k4q = &jac * &(&q + &k3q * step);
        q += &(&k1q * (step / 6.0) + &k2q * (step / 3.0) + &k3q * (step / 3.0) + &k4q * (step / 6.0));

        t += step;
        count += 1;

        if count % renorm_steps == 0 {
            // Gram-Schmidt orthonormalization
            gram_schmidt_inplace(&mut q, &mut sums, count * renorm_steps);
        }
    }

    // Final renormalization
    gram_schmidt_inplace(&mut q, &mut sums, count);

    let total_time = t1 - t0;
    let mut exponents: Vec<f64> = sums.iter().map(|s| s / total_time).collect();
    exponents.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let ky_dim = kaplan_yorke_dimension(&exponents);

    LyapunovSpectrum {
        exponents,
        ky_dimension: ky_dim,
    }
}

/// Compute the full Lyapunov spectrum for a discrete system.
pub fn lyapunov_spectrum_discrete<S: DiscreteSystem>(
    system: &S,
    x0: &DVector<f64>,
    n_steps: usize,
) -> LyapunovSpectrum {
    let n = system.dim();
    let mut x = x0.clone();
    let mut q = DMatrix::identity(n, n);
    let mut sums = vec![0.0_f64; n];

    for step in 0..n_steps {
        // Advance state
        x = system.step(&x);

        // Advance perturbation vectors
        let jac = system.jacobian(&x);
        q = jac * q;

        // Gram-Schmidt every step for discrete maps
        gram_schmidt_inplace(&mut q, &mut sums, step + 1);
    }

    let mut exponents: Vec<f64> = sums.iter().map(|s| s / n_steps as f64).collect();
    exponents.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let ky_dim = kaplan_yorke_dimension(&exponents);

    LyapunovSpectrum {
        exponents,
        ky_dimension: ky_dim,
    }
}

/// Gram-Schmidt orthonormalization, accumulating log stretching factors.
fn gram_schmidt_inplace(q: &mut DMatrix<f64>, sums: &mut [f64], _n: usize) {
    let nvecs = q.ncols();
    for i in 0..nvecs {
        // Subtract projections
        for j in 0..i {
            let dot = q.column(i).dot(&q.column(j));
            for k in 0..q.nrows() {
                q[(k, i)] -= dot * q[(k, j)];
            }
        }
        // Normalize
        let norm = q.column(i).norm();
        if norm > 1e-30 {
            sums[i] += norm.ln();
            for k in 0..q.nrows() {
                q[(k, i)] /= norm;
            }
        }
    }
}

/// Compute the Kaplan-Yorke (Lyapunov) dimension from Lyapunov exponents.
pub fn kaplan_yorke_dimension(exponents: &[f64]) -> f64 {
    if exponents.is_empty() {
        return 0.0;
    }
    // Find j such that sum(lambda_1..lambda_j) >= 0 but sum(lambda_1..lambda_{j+1}) < 0
    let mut cumsum = 0.0;
    for (j, &lambda) in exponents.iter().enumerate() {
        let next_sum = cumsum + lambda;
        if next_sum < 0.0 {
            // D_KY = j + cumsum / |lambda_{j+1}|
            if lambda.abs() < 1e-15 {
                return j as f64;
            }
            return j as f64 + cumsum / lambda.abs();
        }
        cumsum = next_sum;
    }
    // All positive: dimension = number of exponents
    exponents.len() as f64
}

/// Estimate the correlation dimension using the Grassberger-Procaccia algorithm.
pub fn correlation_dimension(points: &[DVector<f64>], max_r: f64, n_bins: usize) -> f64 {
    let n = points.len();
    if n < 10 {
        return 0.0;
    }

    // Subsample for performance
    let max_pairs = 10000;
    let step = ((n * (n - 1)) / 2).div_ceil(max_pairs).max(1);
    
    let mut distances: Vec<f64> = Vec::new();
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            count += 1;
            if count % step == 0 {
                distances.push((&points[i] - &points[j]).norm());
            }
        }
    }

    if distances.is_empty() {
        return 0.0;
    }

    // Compute C(r) at logarithmic bins
    let dr = max_r / n_bins as f64;
    let mut log_r: Vec<f64> = Vec::new();
    let mut log_c: Vec<f64> = Vec::new();
    let total = distances.len() as f64;

    for i in 1..=n_bins {
        let r = dr * i as f64;
        let c = distances.iter().filter(|&&d| d < r).count() as f64 / total;
        if c > 0.0 && r > 0.0 {
            log_r.push(r.ln());
            log_c.push(c.ln());
        }
    }

    if log_r.len() < 3 {
        return 0.0;
    }

    // Linear regression in log-log space
    let n_pts = log_r.len() as f64;
    let sum_x: f64 = log_r.iter().sum();
    let sum_y: f64 = log_c.iter().sum();
    let sum_xy: f64 = log_r.iter().zip(log_c.iter()).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = log_r.iter().map(|x| x * x).sum();

    let slope = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x * sum_x);
    slope.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous::LorenzSystem;
    use crate::discrete::{LogisticMap, HenonMap};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_lorenz_lyapunov_positive() {
        let sys = LorenzSystem::chaotic();
        let x0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let spectrum = lyapunov_spectrum_continuous(&sys, &x0, 0.0, 50.0, 0.01, 10);
        // Largest Lyapunov exponent should be positive for chaotic Lorenz (~0.9)
        assert!(spectrum.exponents[0] > 0.5, "Largest LE should be > 0.5, got {}", spectrum.exponents[0]);
        // Sum should be approximately -(sigma + beta + 1) = -(10 + 8/3 + 1) ≈ -13.67
        let sum: f64 = spectrum.exponents.iter().sum();
        assert_abs_diff_eq!(sum, -(10.0 + 8.0 / 3.0 + 1.0), epsilon = 3.0);
    }

    #[test]
    fn test_lorenz_lyapunov_dimension() {
        let sys = LorenzSystem::chaotic();
        let x0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let spectrum = lyapunov_spectrum_continuous(&sys, &x0, 0.0, 50.0, 0.01, 10);
        // KY dimension should be between 2 and 3 for Lorenz
        assert!(spectrum.ky_dimension > 2.0, "KY dim > 2, got {}", spectrum.ky_dimension);
        assert!(spectrum.ky_dimension < 3.0, "KY dim < 3, got {}", spectrum.ky_dimension);
    }

    #[test]
    fn test_logistic_lyapunov_positive_at_r4() {
        let map = LogisticMap::new(4.0);
        let x0 = DVector::from_vec(vec![0.1]);
        let spectrum = lyapunov_spectrum_discrete(&map, &x0, 10000);
        // LE for r=4 should be ln(2) ≈ 0.693
        assert!(spectrum.exponents[0] > 0.3, "LE should be positive, got {}", spectrum.exponents[0]);
    }

    #[test]
    fn test_logistic_lyapunov_negative_at_r2() {
        let map = LogisticMap::new(2.0);
        let x0 = DVector::from_vec(vec![0.1]);
        let spectrum = lyapunov_spectrum_discrete(&map, &x0, 10000);
        // LE should be negative (stable fixed point)
        assert!(spectrum.exponents[0] < 0.0, "LE should be negative, got {}", spectrum.exponents[0]);
    }

    #[test]
    fn test_henon_lyapunov_positive() {
        let map = HenonMap::chaotic();
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let spectrum = lyapunov_spectrum_discrete(&map, &x0, 10000);
        assert!(spectrum.exponents[0] > 0.1, "Largest LE positive, got {}", spectrum.exponents[0]);
    }

    #[test]
    fn test_kaplan_yorke_dimension() {
        // Lorenz-like: +0.9, 0, -14.5 → D_KY = 2 + 0.9/14.5 ≈ 2.062
        let exps = vec![0.9, 0.0, -14.5];
        let dim = kaplan_yorke_dimension(&exps);
        assert_abs_diff_eq!(dim, 2.0 + 0.9 / 14.5, epsilon = 0.01);
    }

    #[test]
    fn test_correlation_dimension_basic() {
        // Points on a line should have dimension ~1
        let points: Vec<DVector<f64>> = (0..100)
            .map(|i| DVector::from_vec(vec![i as f64 / 100.0, 0.0]))
            .collect();
        let dim = correlation_dimension(&points, 2.0, 20);
        assert!(dim > 0.5 && dim < 2.0, "Line dimension should be ~1, got {dim}");
    }
}
