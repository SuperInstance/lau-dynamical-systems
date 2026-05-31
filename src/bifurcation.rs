//! Bifurcation diagrams (parameter sweep with fixed point tracking).

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::discrete::{DiscreteSystem, LogisticMap};
use crate::continuous::{ContinuousSystem, rk4};

/// A point in a bifurcation diagram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BifurcationPoint {
    pub parameter: f64,
    pub value: f64,
}

/// Generate a bifurcation diagram for the logistic map.
pub fn logistic_bifurcation(
    r_range: (f64, f64),
    r_steps: usize,
    transient: usize,
    record: usize,
) -> Vec<BifurcationPoint> {
    let mut points = Vec::new();
    let dr = (r_range.1 - r_range.0) / r_steps as f64;
    for i in 0..=r_steps {
        let r = r_range.0 + dr * i as f64;
        let map = LogisticMap::new(r);
        let mut x = 0.5;
        for _ in 0..transient {
            x = map.step_scalar(x);
        }
        for _ in 0..record {
            x = map.step_scalar(x);
            points.push(BifurcationPoint { parameter: r, value: x });
        }
    }
    points
}

/// Bifurcation type classification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BifurcationType {
    SaddleNode,
    Transcritical,
    Pitchfork,
    Hopf,
    PeriodDoubling,
    Unknown,
}

/// Detect a period-doubling bifurcation by checking if the orbit period doubles
/// at a given parameter value for the logistic map.
pub fn detect_period_doubling(
    r: f64,
    r_eps: f64,
    expected_period_before: usize,
) -> Option<BifurcationType> {
    let period_after = expected_period_before * 2;
    let period_before = measure_logistic_period(r - r_eps, 2000, 500);
    let period_at = measure_logistic_period(r, 2000, 500);
    if period_before == expected_period_before && period_at == period_after {
        Some(BifurcationType::PeriodDoubling)
    } else {
        None
    }
}

/// Measure the period of the logistic map at a given r.
pub fn measure_logistic_period(r: f64, transient: usize, test: usize) -> usize {
    let map = LogisticMap::new(r);
    let mut x = 0.5;
    for _ in 0..transient {
        x = map.step_scalar(x);
    }
    let mut orbit = vec![x];
    for _ in 0..test {
        x = map.step_scalar(x);
        orbit.push(x);
    }
    // Find the period by checking when the orbit repeats
    let tol = 1e-6;
    for p in 1..=50 {
        let mut ok = true;
        for i in 0..test.min(200) {
            if (orbit[i] - orbit[i + p]).abs() > tol {
                ok = false;
                break;
            }
        }
        if ok {
            return p;
        }
    }
    // If no period found, assume chaos (return 0)
    0
}

/// Bifurcation diagram for a continuous system via parameter sweep.
pub fn continuous_bifurcation<S, F>(
    system_factory: F,
    param_range: (f64, f64),
    param_steps: usize,
    x0: &DVector<f64>,
    t_span: f64,
    dt: f64,
    transient_ratio: f64,
    sample_component: usize,
) -> Vec<BifurcationPoint>
where
    S: ContinuousSystem,
    F: Fn(f64) -> S,
{
    let mut points = Vec::new();
    let dp = (param_range.1 - param_range.0) / param_steps as f64;
    for i in 0..=param_steps {
        let p = param_range.0 + dp * i as f64;
        let sys = system_factory(p);
        let result = rk4(&sys, x0, 0.0, t_span, dt);
        let transient = (result.states.len() as f64 * transient_ratio) as usize;
        for s in result.states.iter().skip(transient) {
            if sample_component < s.nrows() {
                points.push(BifurcationPoint { parameter: p, value: s[sample_component] });
            }
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_logistic_bifurcation_produces_points() {
        let pts = logistic_bifurcation((2.5, 4.0), 100, 500, 50);
        assert!(!pts.is_empty());
        assert!(pts.len() > 100 * 50);
    }

    #[test]
    fn test_logistic_period_1_at_r2_5() {
        let period = measure_logistic_period(2.5, 2000, 500);
        assert_eq!(period, 1);
    }

    #[test]
    fn test_logistic_period_2_at_r3_2() {
        let period = measure_logistic_period(3.2, 2000, 500);
        assert_eq!(period, 2);
    }

    #[test]
    fn test_logistic_period_4_at_r3_5() {
        let period = measure_logistic_period(3.5, 2000, 500);
        assert_eq!(period, 4);
    }

    #[test]
    fn test_logistic_chaos_at_r4() {
        // At r=4, the system is chaotic but floating point can collapse to periodic orbits
        let map = LogisticMap::new(4.0);
        // Verify sensitive dependence instead of period
        let mut a = 0.1_f64;
        let mut b = 0.1001_f64;
        for _ in 0..100 {
            a = map.step_scalar(a);
            b = map.step_scalar(b);
        }
        assert!((a - b).abs() > 0.05, "Trajectories should diverge for chaotic map");
    }

    #[test]
    fn test_bifurcation_range_coverage() {
        let pts = logistic_bifurcation((3.0, 3.0), 1, 100, 10);
        // 0..=r_steps = 2 r values, each recording 10 points = 20
        assert_eq!(pts.len(), 20);
        for p in &pts {
            assert_abs_diff_eq!(p.parameter, 3.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_continuous_bifurcation_runs() {
        use crate::continuous::{LorenzSystem, LorenzParams};
        let x0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let pts = continuous_bifurcation(
            |rho: f64| LorenzSystem::new(LorenzParams { rho, ..LorenzParams::default() }),
            (1.0, 5.0),
            10,
            &x0,
            5.0,
            0.05,
            0.8,
            0,
        );
        assert!(!pts.is_empty());
    }
}
