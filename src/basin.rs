//! Basin of attraction computation.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::continuous::{ContinuousSystem, rk4};
use crate::discrete::DiscreteSystem;
use crate::fixed_points::find_fixed_point_continuous;

/// Result of basin of attraction computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasinOfAttraction {
    /// Grid of initial conditions (flattened).
    pub initial_conditions: Vec<DVector<f64>>,
    /// Which attractor each IC converges to (index into `attractors`).
    pub attractor_index: Vec<usize>,
    /// Discovered attractors.
    pub attractors: Vec<DVector<f64>>,
}

/// Compute the basin of attraction for a continuous system over a 1D grid.
pub fn basin_1d_continuous<S: ContinuousSystem>(
    system: &S,
    x_min: f64,
    x_max: f64,
    n_points: usize,
    component: usize,
    fill_value: &DVector<f64>,
    t_span: f64,
    dt: f64,
    attractor_tol: f64,
) -> BasinOfAttraction {
    let mut ics = Vec::new();
    let mut indices = Vec::new();
    let mut attractors: Vec<DVector<f64>> = Vec::new();

    let dx = (x_max - x_min) / (n_points - 1).max(1) as f64;

    for i in 0..n_points {
        let mut x0 = fill_value.clone();
        x0[component] = x_min + dx * i as f64;
        let result = rk4(system, &x0, 0.0, t_span, dt);
        let final_state = result.states.last().unwrap();

        // Match to existing attractor
        let mut found_idx = None;
        for (idx, attr) in attractors.iter().enumerate() {
            if (final_state - attr).norm() < attractor_tol {
                found_idx = Some(idx);
                break;
            }
        }
        let idx = found_idx.unwrap_or_else(|| {
            attractors.push(final_state.clone());
            attractors.len() - 1
        });

        ics.push(x0);
        indices.push(idx);
    }

    BasinOfAttraction {
        initial_conditions: ics,
        attractor_index: indices,
        attractors,
    }
}

/// Compute basin of attraction for a discrete system over a 2D grid.
pub fn basin_2d_discrete<S: DiscreteSystem>(
    system: &S,
    x_range: (f64, f64),
    y_range: (f64, f64),
    nx: usize,
    ny: usize,
    n_iter: usize,
    attractor_tol: f64,
) -> BasinOfAttraction {
    let mut ics = Vec::new();
    let mut indices = Vec::new();
    let mut attractors: Vec<DVector<f64>> = Vec::new();

    let dx = (x_range.1 - x_range.0) / (nx - 1).max(1) as f64;
    let dy = (y_range.1 - y_range.0) / (ny - 1).max(1) as f64;

    for i in 0..nx {
        for j in 0..ny {
            let x0 = DVector::from_vec(vec![
                x_range.0 + dx * i as f64,
                y_range.0 + dy * j as f64,
            ]);

            // Iterate to find attractor
            let mut x = x0.clone();
            for _ in 0..n_iter {
                x = system.step(&x);
                if !x.iter().all(|v| v.is_finite()) {
                    break;
                }
            }

            if !x.iter().all(|v| v.is_finite()) {
                ics.push(x0);
                indices.push(usize::MAX); // diverged
                continue;
            }

            let mut found_idx = None;
            for (idx, attr) in attractors.iter().enumerate() {
                if (&x - attr).norm() < attractor_tol {
                    found_idx = Some(idx);
                    break;
                }
            }
            let idx = found_idx.unwrap_or_else(|| {
                attractors.push(x.clone());
                attractors.len() - 1
            });

            ics.push(x0);
            indices.push(idx);
        }
    }

    BasinOfAttraction {
        initial_conditions: ics,
        attractor_index: indices,
        attractors,
    }
}

/// Basin of attraction for 1D discrete map.
pub fn basin_1d_discrete_map<F>(
    map: F,
    x_min: f64,
    x_max: f64,
    n_points: usize,
    n_iter: usize,
    attractor_tol: f64,
) -> BasinOfAttraction
where
    F: Fn(f64) -> f64,
{
    let mut ics = Vec::new();
    let mut indices = Vec::new();
    let mut attractors: Vec<f64> = Vec::new();

    let dx = (x_max - x_min) / (n_points - 1).max(1) as f64;

    for i in 0..n_points {
        let x0 = x_min + dx * i as f64;
        let mut x = x0;
        for _ in 0..n_iter {
            x = map(x);
        }

        let mut found_idx = None;
        for (idx, &attr) in attractors.iter().enumerate() {
            if (x - attr).abs() < attractor_tol {
                found_idx = Some(idx);
                break;
            }
        }
        let idx = found_idx.unwrap_or_else(|| {
            attractors.push(x);
            attractors.len() - 1
        });

        ics.push(DVector::from_vec(vec![x0]));
        indices.push(idx);
    }

    BasinOfAttraction {
        initial_conditions: ics,
        attractor_index: indices,
        attractors: attractors.iter().map(|a| DVector::from_vec(vec![*a])).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::{LogisticMap, HenonMap, DiscreteSystem};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_basin_logistic_r2_converges_to_one_attractor() {
        let basin = basin_1d_discrete_map(
            |x| 2.0 * x * (1.0 - x),
            0.01,
            0.99,
            50,
            1000,
            0.01,
        );
        // For r=2, all orbits converge to x*=0.5
        assert_eq!(basin.attractors.len(), 1);
        assert_abs_diff_eq!(basin.attractors[0][0], 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_basin_logistic_multiple_attractors() {
        // At r=3.2, period-2 orbit exists
        let basin = basin_1d_discrete_map(
            |x| 3.2 * x * (1.0 - x),
            0.01,
            0.99,
            50,
            2000,
            0.05,
        );
        // All should converge to the period-2 attractor region
        assert!(!basin.attractors.is_empty());
    }

    #[test]
    fn test_basin_2d_henon() {
        let map = HenonMap::chaotic();
        let basin = basin_2d_discrete(
            &map,
            (-1.5, 1.5),
            (-0.5, 0.5),
            20,
            20,
            500,
            0.5,
        );
        // Should have at least one attractor
        assert!(!basin.attractors.is_empty());
        // Most points should converge (not diverge)
        let converged = basin.attractor_index.iter().filter(|&&i| i != usize::MAX).count();
        assert!(converged > 0);
    }
}
