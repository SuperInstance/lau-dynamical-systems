//! Fixed point finding and stability analysis.

use nalgebra::{DVector, DMatrix, ComplexField};
use serde::{Serialize, Deserialize};
use crate::continuous::ContinuousSystem;
use crate::discrete::DiscreteSystem;

/// Stability classification of a fixed point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Stability {
    /// All eigenvalues have negative real part (continuous) or |λ| < 1 (discrete).
    Stable,
    /// At least one eigenvalue has positive real part (continuous) or |λ| > 1 (discrete).
    Unstable,
    /// On the boundary — max |Re(λ)| < eps or max ||λ| - 1| < eps.
    Marginal,
    /// Has both stable and unstable directions.
    Saddle,
}

impl std::fmt::Display for Stability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stability::Stable => write!(f, "Stable"),
            Stability::Unstable => write!(f, "Unstable"),
            Stability::Marginal => write!(f, "Marginal"),
            Stability::Saddle => write!(f, "Saddle"),
        }
    }
}

/// Result of fixed point analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPointAnalysis {
    pub point: DVector<f64>,
    pub eigenvalues: Vec<num_complex::Complex64>,
    pub stability: Stability,
}

/// Classify stability from continuous-system eigenvalues.
pub fn classify_continuous(eigenvalues: &[num_complex::Complex64], eps: f64) -> Stability {
    let re: Vec<f64> = eigenvalues.iter().map(|e| e.re).collect();
    let has_positive = re.iter().any(|&r| r > eps);
    let has_negative = re.iter().any(|&r| r < -eps);
    if has_positive && has_negative {
        Stability::Saddle
    } else if has_positive {
        Stability::Unstable
    } else if has_negative {
        Stability::Stable
    } else {
        Stability::Marginal
    }
}

/// Classify stability from discrete-system eigenvalues (|λ| vs 1).
pub fn classify_discrete(eigenvalues: &[num_complex::Complex64], eps: f64) -> Stability {
    let mags: Vec<f64> = eigenvalues.iter().map(|e| e.norm()).collect();
    let has_outside = mags.iter().any(|&m| m > 1.0 + eps);
    let has_inside = mags.iter().any(|&m| m < 1.0 - eps);
    if has_outside && has_inside {
        Stability::Saddle
    } else if has_outside {
        Stability::Unstable
    } else if has_inside {
        Stability::Stable
    } else {
        Stability::Marginal
    }
}

/// Compute eigenvalues of a real matrix.
pub fn eigenvalues(mat: &DMatrix<f64>) -> Vec<num_complex::Complex64> {
    let n = mat.nrows();
    if n == 0 {
        return vec![];
    }
    // Use nalgebra's eigen decomposition for symmetric; fallback to characteristic polynomial
    // For general matrices, we use a simple QR-based approach via nalgebra's SymmetricEigen for symmetric
    // or approximate for non-symmetric.
    let sym = (mat - &mat.transpose()).norm() < 1e-10;
    if sym {
        let eig = mat.symmetric_eigenvalues();
        eig.iter().map(|&r| num_complex::Complex64::new(r, 0.0)).collect()
    } else {
        // Use nalgebra's Schur decomposition for non-symmetric
        // Fallback: compute via the balance-reduce method
        eigenvalues_qr(mat)
    }
}

/// Simple QR-iteration eigenvalue estimation for small matrices.
fn eigenvalues_qr(mat: &DMatrix<f64>) -> Vec<num_complex::Complex64> {
    use nalgebra::RealField;
    let n = mat.nrows();
    let mut a = mat.clone();
    for _ in 0..200 {
        let qr = a.qr();
        let q = qr.q();
        let r = qr.r();
        a = &r * q;
    }
    // Extract eigenvalues: 1x1 and 2x2 blocks
    let mut eigs = Vec::new();
    let mut i = 0;
    while i < n {
        if i + 1 < n && a[(i + 1, i)].abs() > 1e-10 {
            // 2x2 block
            let a11 = a[(i, i)];
            let a12 = a[(i, i + 1)];
            let a21 = a[(i + 1, i)];
            let a22 = a[(i + 1, i + 1)];
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = tr * tr - 4.0 * det;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                eigs.push(num_complex::Complex64::new((tr + sq) / 2.0, 0.0));
                eigs.push(num_complex::Complex64::new((tr - sq) / 2.0, 0.0));
            } else {
                let sq = (-disc).sqrt();
                eigs.push(num_complex::Complex64::new(tr / 2.0, sq / 2.0));
                eigs.push(num_complex::Complex64::new(tr / 2.0, -sq / 2.0));
            }
            i += 2;
        } else {
            eigs.push(num_complex::Complex64::new(a[(i, i)], 0.0));
            i += 1;
        }
    }
    eigs
}

/// Analyze a continuous system's fixed point.
pub fn analyze_continuous_fixed_point<S: ContinuousSystem>(
    system: &S,
    point: &DVector<f64>,
    t: f64,
) -> FixedPointAnalysis {
    let jac = system.jacobian(point, t);
    let eigs = eigenvalues(&jac);
    let stability = classify_continuous(&eigs, 1e-8);
    FixedPointAnalysis {
        point: point.clone(),
        eigenvalues: eigs,
        stability,
    }
}

/// Analyze a discrete system's fixed point.
pub fn analyze_discrete_fixed_point<S: DiscreteSystem>(
    system: &S,
    point: &DVector<f64>,
) -> FixedPointAnalysis {
    let jac = system.jacobian(point);
    let eigs = eigenvalues(&jac);
    let stability = classify_discrete(&eigs, 1e-6);
    FixedPointAnalysis {
        point: point.clone(),
        eigenvalues: eigs,
        stability,
    }
}

/// Find a fixed point of a continuous system by Newton's method.
pub fn find_fixed_point_continuous<S: ContinuousSystem>(
    system: &S,
    x0: &DVector<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<DVector<f64>> {
    let n = system.dim();
    let mut x = x0.clone();
    for _ in 0..max_iter {
        let f = system.f(&x, 0.0);
        if f.norm() < tol {
            return Some(x);
        }
        let jac = system.jacobian(&x, 0.0);
        if let Some(delta) = jac.clone().lu().solve(&f) {
            x -= &delta;
        } else {
            return None;
        }
    }
    if system.f(&x, 0.0).norm() < tol * 10.0 {
        Some(x)
    } else {
        None
    }
}

/// Find a fixed point of a discrete system by Newton's method.
pub fn find_fixed_point_discrete<S: DiscreteSystem>(
    system: &S,
    x0: &DVector<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<DVector<f64>> {
    let n = system.dim();
    let mut x = x0.clone();
    for _ in 0..max_iter {
        let g = &system.step(&x) - &x;
        if g.norm() < tol {
            return Some(x);
        }
        let jac = system.jacobian(&x);
        // dg/dx = Jac - I
        let mut dg = jac.clone();
        for i in 0..n {
            dg[(i, i)] -= 1.0;
        }
        if let Some(delta) = dg.lu().solve(&g) {
            x -= &delta;
        } else {
            return None;
        }
    }
    let g = &system.step(&x) - &x;
    if g.norm() < tol * 10.0 {
        Some(x)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous::{LorenzSystem, LorenzParams};
    use crate::discrete::{LogisticMap, HenonMap};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_lorenz_origin_stability() {
        let sys = LorenzSystem::chaotic();
        let x0 = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let analysis = analyze_continuous_fixed_point(&sys, &x0, 0.0);
        assert_eq!(analysis.stability, Stability::Saddle);
    }

    #[test]
    fn test_lorenz_nontrivial_fixed_points_are_stable_or_saddle() {
        let sys = LorenzSystem::chaotic();
        let r = sys.params.rho;
        let b = sys.params.beta;
        let c = (b * (r - 1.0)).sqrt();
        let fp = DVector::from_vec(vec![c, c, r - 1.0]);
        let analysis = analyze_continuous_fixed_point(&sys, &fp, 0.0);
        // For canonical Lorenz, these are saddle points
        assert!(matches!(analysis.stability, Stability::Unstable | Stability::Saddle));
    }

    #[test]
    fn test_find_lorenz_fixed_point() {
        let sys = LorenzSystem::chaotic();
        let x0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let fp = find_fixed_point_continuous(&sys, &x0, 100, 1e-12);
        assert!(fp.is_some());
        let fp = fp.unwrap();
        // Should be close to origin or one of the nontrivial fixed points
        let f = sys.f(&fp, 0.0);
        assert_abs_diff_eq!(f.norm(), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_logistic_stability_at_r2() {
        // At r=2, fixed point x*=0.5 is stable
        let map = LogisticMap::new(2.0);
        let fp = DVector::from_vec(vec![0.5]);
        let analysis = analyze_discrete_fixed_point(&map, &fp);
        assert_eq!(analysis.stability, Stability::Stable);
    }

    #[test]
    fn test_logistic_instability_at_r4() {
        // At r=4, fixed point x*=0.75 is unstable
        let map = LogisticMap::new(4.0);
        let fp = DVector::from_vec(vec![0.75]);
        let analysis = analyze_discrete_fixed_point(&map, &fp);
        assert_eq!(analysis.stability, Stability::Unstable);
    }

    #[test]
    fn test_find_logistic_fixed_point() {
        let map = LogisticMap::new(3.0);
        let x0 = DVector::from_vec(vec![0.5]);
        let fp = find_fixed_point_discrete(&map, &x0, 50, 1e-12);
        assert!(fp.is_some());
        let fp = fp.unwrap();
        assert_abs_diff_eq!(fp[0], 2.0 / 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_henon_fixed_point_analysis() {
        let map = HenonMap::chaotic();
        let a = map.params.a;
        let b = map.params.b;
        let disc = (b - 1.0).powi(2) + 4.0 * a;
        let x_fp = (-(b - 1.0) - disc.sqrt()) / (2.0 * a);
        let y_fp = b * x_fp;
        let fp = DVector::from_vec(vec![x_fp, y_fp]);
        let analysis = analyze_discrete_fixed_point(&map, &fp);
        assert!(matches!(analysis.stability, Stability::Unstable | Stability::Saddle));
    }

    #[test]
    fn test_classify_continuous_all_negative() {
        let eigs = vec![
            num_complex::Complex64::new(-1.0, 0.0),
            num_complex::Complex64::new(-2.0, 1.0),
        ];
        assert_eq!(classify_continuous(&eigs, 1e-8), Stability::Stable);
    }

    #[test]
    fn test_classify_continuous_saddle() {
        let eigs = vec![
            num_complex::Complex64::new(-1.0, 0.0),
            num_complex::Complex64::new(2.0, 0.0),
        ];
        assert_eq!(classify_continuous(&eigs, 1e-8), Stability::Saddle);
    }

    #[test]
    fn test_classify_discrete_stable() {
        let eigs = vec![
            num_complex::Complex64::new(0.5, 0.0),
            num_complex::Complex64::new(-0.3, 0.0),
        ];
        assert_eq!(classify_discrete(&eigs, 1e-6), Stability::Stable);
    }

    #[test]
    fn test_eigenvalues_symmetric() {
        let mat = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let eigs = eigenvalues(&mat);
        let mut reals: Vec<f64> = eigs.iter().map(|e| e.re).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_abs_diff_eq!(reals[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(reals[1], 3.0, epsilon = 1e-6);
    }
}
