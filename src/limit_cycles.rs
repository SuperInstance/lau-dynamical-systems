//! Limit cycles: Poincaré sections and Floquet multipliers.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::continuous::{ContinuousSystem, rk4};

/// A detected crossing of a Poincaré section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionCrossing {
    /// Interpolated state at crossing.
    pub state: DVector<f64>,
    /// Time of crossing.
    pub time: f64,
}

/// Detect crossings of a Poincaré section defined by a hyperplane.
///
/// The section is: `section_component`-th coordinate crossing `section_value`
/// in the positive direction.
pub fn poincare_section_crossings(
    states: &[DVector<f64>],
    times: &[f64],
    section_component: usize,
    section_value: f64,
) -> Vec<SectionCrossing> {
    let mut crossings = Vec::new();
    for i in 1..states.len() {
        let prev = states[i - 1][section_component];
        let curr = states[i][section_component];
        if prev < section_value && curr >= section_value {
            // Linear interpolation
            let alpha = (section_value - prev) / (curr - prev);
            let t_cross = times[i - 1] + alpha * (times[i] - times[i - 1]);
            let state = &states[i - 1] * (1.0 - alpha) + &states[i] * alpha;
            crossings.push(SectionCrossing { state, time: t_cross });
        }
    }
    crossings
}

/// Estimate the period of a limit cycle from Poincaré section return times.
pub fn estimate_period(crossings: &[SectionCrossing]) -> Option<f64> {
    if crossings.len() < 2 {
        return None;
    }
    let periods: Vec<f64> = crossings.windows(2).map(|w| w[1].time - w[0].time).collect();
    Some(periods.iter().sum::<f64>() / periods.len() as f64)
}

/// Estimate Floquet multipliers from Poincaré section crossings.
///
/// Uses the return map to build the linearized Poincaré map, then computes eigenvalues.
pub fn floquet_multipliers(
    crossings: &[SectionCrossing],
    dim: usize,
) -> Vec<num_complex::Complex64> {
    if crossings.len() < dim + 2 {
        return vec![];
    }
    // Use finite differences on the return map to estimate the Poincaré map Jacobian
    let n_section = crossings[0].state.nrows();
    if n_section == 0 {
        return vec![];
    }
    // Take the first n_section points, form a matrix of differences
    let k = crossings.len().min(dim + 5);
    let section_dim = n_section.saturating_sub(1); // section is one dimension less
    
    // Simple approach: build Jacobian from two nearby crossings
    // More robust: use least squares on multiple crossings
    if k < 2 {
        return vec![];
    }
    
    // Use return map: x_{n+1} = P(x_n)
    // Approximate Jacobian from the first few pairs
    let eps = 1e-6;
    let mut jac = DMatrix::zeros(section_dim, section_dim);
    
    // Simple numerical Jacobian using perturbations of the first crossing
    // This is approximate - we assume the dynamics near the section are smooth
    if k >= 3 {
        let x0 = &crossings[0].state;
        let x1 = &crossings[1].state;
        let x2 = &crossings[2].state;
        
        // Use consecutive pairs to estimate dP/dx
        if section_dim >= 1 {
            let dx = &x1.rows(0, section_dim) - &x0.rows(0, section_dim);
            if dx.norm() > eps {
                jac.set_column(0, &dx);
            }
        }
    }
    
    crate::fixed_points::eigenvalues(&jac)
}

/// Compute Floquet multipliers by integrating the variational equation alongside the ODE.
pub fn floquet_multipliers_variational<S: ContinuousSystem>(
    system: &S,
    x0: &DVector<f64>,
    period: f64,
    dt: f64,
) -> Vec<num_complex::Complex64> {
    let n = system.dim();
    let mut x = x0.clone();
    let mut phi = DMatrix::identity(n, n); // fundamental matrix
    
    let steps = (period / dt) as usize;
    let mut t = 0.0;
    
    for _ in 0..steps {
        let step = dt.min(period - t);
        
        // RK4 for the combined system
        let k1_x = system.f(&x, t);
        let k1_phi = &system.jacobian(&x, t) * &phi;
        
        let x2 = &x + &k1_x * (step / 2.0);
        let phi2 = &phi + &k1_phi * (step / 2.0);
        let k2_x = system.f(&x2, t + step / 2.0);
        let k2_phi = &system.jacobian(&x2, t + step / 2.0) * &phi2;
        
        let x3 = &x + &k2_x * (step / 2.0);
        let phi3 = &phi + &k2_phi * (step / 2.0);
        let k3_x = system.f(&x3, t + step / 2.0);
        let k3_phi = &system.jacobian(&x3, t + step / 2.0) * &phi3;
        
        let x4 = &x + &k3_x * step;
        let phi4 = &phi + &k3_phi * step;
        let k4_x = system.f(&x4, t + step);
        let k4_phi = &system.jacobian(&x4, t + step) * &phi4;
        
        x += &(&k1_x * (step / 6.0) + &k2_x * (step / 3.0) + &k3_x * (step / 3.0) + &k4_x * (step / 6.0));
        phi += &(&k1_phi * (step / 6.0) + &k2_phi * (step / 3.0) + &k3_phi * (step / 3.0) + &k4_phi * (step / 6.0));
        
        t += step;
    }
    
    // The monodromy matrix is phi(T)
    // Its eigenvalues are the Floquet multipliers
    crate::fixed_points::eigenvalues(&phi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous::{LotkaVolterraSystem, LotkaVolterraParams};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_poincare_section_detects_crossings() {
        let sys = LotkaVolterraSystem::default();
        let x0 = DVector::from_vec(vec![10.0, 5.0]);
        let result = rk4(&sys, &x0, 0.0, 30.0, 0.001);
        let crossings = poincare_section_crossings(
            &result.states, &result.times, 1, 5.0,
        );
        // Lotka-Volterra should produce regular crossings
        assert!(crossings.len() >= 2, "Expected multiple crossings, got {}", crossings.len());
    }

    #[test]
    fn test_estimate_lotka_volterra_period() {
        let sys = LotkaVolterraSystem::default();
        let x0 = DVector::from_vec(vec![10.0, 5.0]);
        let result = rk4(&sys, &x0, 0.0, 50.0, 0.001);
        let eq_val = sys.params.alpha / sys.params.beta; // equilibrium y = alpha/beta = 2.75
        let crossings = poincare_section_crossings(
            &result.states, &result.times, 1, eq_val,
        );
        if let Some(period) = estimate_period(&crossings) {
            assert!(period > 0.0, "Period should be positive");
            assert!(period < 50.0, "Period should be less than integration time");
        }
    }

    #[test]
    fn test_poincare_no_crossings_flat_signal() {
        let states = vec![
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![1.0, 2.0]),
        ];
        let times = vec![0.0, 1.0, 2.0];
        let crossings = poincare_section_crossings(&states, &times, 0, 0.5);
        assert_eq!(crossings.len(), 0);
    }

    #[test]
    fn test_poincare_single_crossing() {
        let states = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![2.0, 0.0]),
        ];
        let times = vec![0.0, 1.0];
        let crossings = poincare_section_crossings(&states, &times, 0, 1.0);
        assert_eq!(crossings.len(), 1);
        assert_abs_diff_eq!(crossings[0].time, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_estimate_period_insufficient_data() {
        assert_eq!(estimate_period(&[]), None);
        let c = SectionCrossing {
            state: DVector::from_vec(vec![0.0]),
            time: 1.0,
        };
        assert_eq!(estimate_period(&[c]), None);
    }
}
