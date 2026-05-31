//! Agent behavioral dynamics: attractors as stable behavioral patterns.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::continuous::{ContinuousSystem, rk4};
use crate::fixed_points::{FixedPointAnalysis, Stability, find_fixed_point_continuous, analyze_continuous_fixed_point};
use crate::strange_attractors::LyapunovSpectrum;

/// A behavioral attractor represents a stable pattern of agent behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralAttractor {
    /// Name/label for this behavioral pattern.
    pub label: String,
    /// The attractor state (behavioral configuration).
    pub state: DVector<f64>,
    /// Stability classification.
    pub stability: Stability,
    /// Optional Lyapunov spectrum at this attractor.
    pub lyapunov_spectrum: Option<Vec<f64>>,
}

impl BehavioralAttractor {
    /// Whether this is a stable behavioral pattern.
    pub fn is_stable(&self) -> bool {
        matches!(self.stability, Stability::Stable)
    }

    /// Whether this behavioral pattern is chaotic (positive Lyapunov exponent).
    pub fn is_chaotic(&self) -> bool {
        self.lyapunov_spectrum
            .as_ref()
            .map(|s| s.first().copied().unwrap_or(0.0) > 0.0)
            .unwrap_or(false)
    }
}

/// A behavioral dynamics model for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralDynamics {
    /// Dimension labels (what each state variable represents).
    pub dimension_labels: Vec<String>,
    /// Known behavioral attractors.
    pub attractors: Vec<BehavioralAttractor>,
}

impl BehavioralDynamics {
    /// Create a new behavioral dynamics model.
    pub fn new(dimension_labels: Vec<String>) -> Self {
        Self {
            dimension_labels,
            attractors: Vec::new(),
        }
    }

    /// Discover attractors from a continuous system model of behavior.
    pub fn discover_attractors<S: ContinuousSystem>(
        &mut self,
        system: &S,
        search_points: &[DVector<f64>],
        t: f64,
    ) {
        for x0 in search_points {
            if let Some(fp) = find_fixed_point_continuous(system, x0, 200, 1e-10) {
                let analysis = analyze_continuous_fixed_point(system, &fp, t);
                // Check if we already found this attractor
                let already_known = self.attractors.iter().any(|a| (&a.state - &fp).norm() < 1e-4);
                if !already_known {
                    self.attractors.push(BehavioralAttractor {
                        label: format!("Attractor_{}", self.attractors.len()),
                        state: fp,
                        stability: analysis.stability,
                        lyapunov_spectrum: None,
                    });
                }
            }
        }
    }

    /// Predict which attractor an agent starting at `x0` will converge to.
    pub fn predict_behavior<S: ContinuousSystem>(
        &self,
        system: &S,
        x0: &DVector<f64>,
        t_span: f64,
        dt: f64,
        tol: f64,
    ) -> Option<&BehavioralAttractor> {
        let result = rk4(system, x0, 0.0, t_span, dt);
        let final_state = result.states.last()?;

        self.attractors
            .iter()
            .find(|a| (&a.state - final_state).norm() < tol)
    }

    /// Number of stable behavioral patterns.
    pub fn stable_pattern_count(&self) -> usize {
        self.attractors.iter().filter(|a| a.is_stable()).count()
    }

    /// Number of chaotic behavioral patterns.
    pub fn chaotic_pattern_count(&self) -> usize {
        self.attractors.iter().filter(|a| a.is_chaotic()).count()
    }
}

/// Simple behavioral dynamics system: 2D system modeling approach-avoidance.
///
/// State: (approach, avoidance) where both are behavioral tendencies.
/// Parameters: (drive, conflict, damping)
pub struct ApproachAvoidanceSystem {
    /// Drive strength toward goal.
    pub drive: f64,
    /// Conflict level (creates interesting dynamics).
    pub conflict: f64,
    /// Damping coefficient.
    pub damping: f64,
}

impl ApproachAvoidanceSystem {
    pub fn new(drive: f64, conflict: f64, damping: f64) -> Self {
        Self { drive, conflict, damping }
    }

    /// Parameters chosen to guarantee a real fixed point.
    pub fn with_fixed_point() -> Self {
        Self { drive: 1.0, conflict: 0.2, damping: 0.5 }
    }
}

impl ContinuousSystem for ApproachAvoidanceSystem {
    fn dim(&self) -> usize { 2 }

    fn f(&self, x: &DVector<f64>, _t: f64) -> DVector<f64> {
        let (a, v) = (x[0], x[1]); // approach, avoidance
        DVector::from_vec(vec![
            self.drive - self.conflict * v - self.damping * a,
            -self.damping * v + self.conflict * a * (1.0 - a),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous::LorenzSystem;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_behavioral_dynamics_create() {
        let bd = BehavioralDynamics::new(vec!["approach".into(), "avoidance".into()]);
        assert_eq!(bd.dimension_labels.len(), 2);
        assert!(bd.attractors.is_empty());
    }

    #[test]
    fn test_approach_avoidance_has_fixed_point() {
        let sys = ApproachAvoidanceSystem::with_fixed_point();
        // Try multiple starting points
        let guesses = vec![
            DVector::from_vec(vec![0.5, 0.5]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![3.0, 1.0]),
        ];
        let mut found = false;
        for x0 in guesses {
            if find_fixed_point_continuous(&sys, &x0, 500, 1e-8).is_some() {
                found = true;
                break;
            }
        }
        assert!(found, "Should find a fixed point for approach-avoidance system");
    }

    #[test]
    fn test_discover_attractors() {
        let sys = ApproachAvoidanceSystem::with_fixed_point();
        let mut bd = BehavioralDynamics::new(vec!["approach".into(), "avoidance".into()]);
        let search = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 1.0]),
            DVector::from_vec(vec![2.0, 2.0]),
        ];
        bd.discover_attractors(&sys, &search, 0.0);
        assert!(!bd.attractors.is_empty());
    }

    #[test]
    fn test_predict_behavior() {
        let sys = ApproachAvoidanceSystem::with_fixed_point();
        let mut bd = BehavioralDynamics::new(vec!["approach".into(), "avoidance".into()]);
        let search = vec![
            DVector::from_vec(vec![0.5, 0.5]),
        ];
        bd.discover_attractors(&sys, &search, 0.0);
        
        if !bd.attractors.is_empty() {
            let x0 = DVector::from_vec(vec![0.1, 0.1]);
            let prediction = bd.predict_behavior(&sys, &x0, 50.0, 0.01, 0.5);
            assert!(prediction.is_some());
        }
    }

    #[test]
    fn test_stable_pattern_count() {
        let mut bd = BehavioralDynamics::new(vec!["x".into()]);
        bd.attractors.push(BehavioralAttractor {
            label: "stable".into(),
            state: DVector::from_vec(vec![1.0]),
            stability: Stability::Stable,
            lyapunov_spectrum: Some(vec![-1.0]),
        });
        bd.attractors.push(BehavioralAttractor {
            label: "unstable".into(),
            state: DVector::from_vec(vec![0.0]),
            stability: Stability::Unstable,
            lyapunov_spectrum: Some(vec![1.0]),
        });
        assert_eq!(bd.stable_pattern_count(), 1);
        assert_eq!(bd.chaotic_pattern_count(), 1);
    }

    #[test]
    fn test_behavioral_attractor_is_stable() {
        let a = BehavioralAttractor {
            label: "test".into(),
            state: DVector::from_vec(vec![0.0]),
            stability: Stability::Stable,
            lyapunov_spectrum: None,
        };
        assert!(a.is_stable());
        assert!(!a.is_chaotic());
    }
}
