//! Poincaré maps: section crossing detection and return map construction.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::continuous::{ContinuousSystem, rk4};
use crate::limit_cycles::{SectionCrossing, poincare_section_crossings};

/// A Poincaré map: the return map on a section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoincareMap {
    /// The section crossings (states projected onto section).
    pub crossings: Vec<SectionCrossing>,
    /// Component index defining the section hyperplane.
    pub section_component: usize,
    /// Value of the section hyperplane.
    pub section_value: f64,
}

impl PoincareMap {
    /// Build a Poincaré map from a trajectory.
    pub fn from_trajectory(
        states: &[DVector<f64>],
        times: &[f64],
        section_component: usize,
        section_value: f64,
    ) -> Self {
        let crossings = poincare_section_crossings(states, times, section_component, section_value);
        PoincareMap {
            crossings,
            section_component,
            section_value,
        }
    }

    /// Number of return map points.
    pub fn len(&self) -> usize {
        self.crossings.len()
    }

    /// Whether there are any crossings.
    pub fn is_empty(&self) -> bool {
        self.crossings.is_empty()
    }

    /// Get the return map as pairs (x_n, x_{n+1}) for a specific component.
    pub fn return_map_pairs(&self, component: usize) -> Vec<(f64, f64)> {
        self.crossings
            .windows(2)
            .map(|w| (w[0].state[component], w[1].state[component]))
            .collect()
    }

    /// Extract the projected section coordinates (all components except the section one).
    pub fn section_points(&self) -> Vec<DVector<f64>> {
        let n = self.crossings.first().map(|c| c.state.nrows()).unwrap_or(0);
        self.crossings
            .iter()
            .map(|c| {
                let mut coords = Vec::new();
                for i in 0..n {
                    if i != self.section_component {
                        coords.push(c.state[i]);
                    }
                }
                DVector::from_vec(coords)
            })
            .collect()
    }
}

/// Compute a Poincaré map for a continuous system.
pub fn compute_poincare_map<S: ContinuousSystem>(
    system: &S,
    x0: &DVector<f64>,
    t0: f64,
    t1: f64,
    dt: f64,
    section_component: usize,
    section_value: f64,
) -> PoincareMap {
    let result = rk4(system, x0, t0, t1, dt);
    PoincareMap::from_trajectory(
        &result.states,
        &result.times,
        section_component,
        section_value,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous::{LorenzSystem, LotkaVolterraSystem};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_lorenz_poincare_map() {
        let sys = LorenzSystem::chaotic();
        let x0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let pmap = compute_poincare_map(&sys, &x0, 0.0, 50.0, 0.01, 2, 25.0);
        // Should have many crossings for chaotic Lorenz
        assert!(pmap.len() > 10, "Expected many crossings, got {}", pmap.len());
    }

    #[test]
    fn test_poincare_return_map_pairs() {
        let sys = LorenzSystem::chaotic();
        let x0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let pmap = compute_poincare_map(&sys, &x0, 0.0, 50.0, 0.01, 2, 25.0);
        let pairs = pmap.return_map_pairs(0);
        if !pairs.is_empty() {
            // Each pair should have finite values
            for (x, y) in &pairs {
                assert!(x.is_finite());
                assert!(y.is_finite());
            }
        }
    }

    #[test]
    fn test_poincare_section_points_dimension() {
        let sys = LorenzSystem::chaotic();
        let x0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let pmap = compute_poincare_map(&sys, &x0, 0.0, 30.0, 0.01, 2, 25.0);
        let pts = pmap.section_points();
        // 3D system with 1 section component → 2D section points
        for p in &pts {
            assert_eq!(p.nrows(), 2);
        }
    }

    #[test]
    fn test_lotka_volterra_poincare_regular() {
        let sys = LotkaVolterraSystem::default();
        let x0 = DVector::from_vec(vec![10.0, 5.0]);
        let pmap = compute_poincare_map(&sys, &x0, 0.0, 30.0, 0.001, 1, 5.0);
        // Periodic orbit should produce roughly evenly spaced crossings
        if pmap.len() >= 3 {
            let times: Vec<f64> = pmap.crossings.iter().map(|c| c.time).collect();
            let intervals: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
            let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
            for &dt in &intervals {
                assert_abs_diff_eq!(dt, mean, epsilon = mean * 0.1);
            }
        }
    }
}
