# lau-dynamical-systems

A Rust library for **dynamical systems theory**: continuous ODE systems, discrete iterated maps, fixed-point stability analysis, bifurcation diagrams, limit cycles, strange attractors, Lyapunov exponents, Poincaré maps, basins of attraction, and agent behavioral dynamics.

---

## What This Does

| Module | What you get |
|---|---|
| `continuous` | Trait for ODE systems, RK4 integrator, Lorenz system, Lotka-Volterra predator-prey |
| `discrete` | Trait for iterated maps, logistic map, Hénon map |
| `fixed_points` | Find fixed points via Newton's method; classify stability (Stable/Unstable/Saddle/Marginal) via eigenvalue analysis |
| `bifurcation` | Bifurcation diagrams for logistic map & continuous systems; period measurement; period-doubling detection |
| `limit_cycles` | Poincaré section crossings, period estimation, Floquet multipliers (return-map & variational equation approaches) |
| `strange_attractors` | Full Lyapunov spectrum (Benettin algorithm with Gram-Schmidt), Kaplan-Yorke dimension, Grassberger-Procaccia correlation dimension |
| `poincare` | Poincaré map construction, return map pairs, section point projection |
| `basin` | Basin of attraction computation (1D continuous, 1D discrete, 2D discrete grids) |
| `agent_dynamics` | Behavioral attractor discovery, approach-avoidance behavioral model, behavior prediction |

---

## Key Idea

A dynamical system evolves a state over time. This library lets you **define** such systems (continuous or discrete), **analyze** them (fixed points, stability, bifurcations, chaos), and **apply** the theory to agent behavioral modeling. Every algorithm runs in a single process—no external solvers needed.

---

## Install

```toml
[dependencies]
lau-dynamical-systems = { git = "https://github.com/SuperInstance/lau-dynamical-systems" }
```

Requires Rust 2021 edition. Dependencies: `nalgebra` 0.33, `num-complex` 0.4, `serde` 1.

---

## Quick Start

### Integrate the Lorenz system

```rust
use lau_dynamical_systems::continuous::{LorenzSystem, rk4};
use nalgebra::DVector;

let sys = LorenzSystem::chaotic(); // sigma=10, rho=28, beta=8/3
let x0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
let result = rk4(&sys, &x0, 0.0, 50.0, 0.01);

println!("Final state: {:?}", result.states.last().unwrap());
println!("Time steps: {}", result.times.len());
```

### Iterate the logistic map

```rust
use lau_dynamical_systems::discrete::{LogisticMap, iterate};
use nalgebra::DVector;

let map = LogisticMap::new(3.2); // period-2 regime
let x0 = DVector::from_vec(vec![0.5]);
let traj = iterate(&map, &x0, 1000);

// Or use the scalar API directly
let mut x = 0.5;
for _ in 0..1000 { x = map.step_scalar(x); }
```

### Fixed-point stability analysis

```rust
use lau_dynamical_systems::fixed_points::{
    analyze_continuous_fixed_point, find_fixed_point_continuous, Stability
};
use lau_dynamical_systems::continuous::LorenzSystem;
use nalgebra::DVector;

let sys = LorenzSystem::chaotic();

// Analyze the origin
let analysis = analyze_continuous_fixed_point(
    &sys, &DVector::from_vec(vec![0.0, 0.0, 0.0]), 0.0
);
assert_eq!(analysis.stability, Stability::Saddle);

// Find a fixed point via Newton's method
let fp = find_fixed_point_continuous(
    &sys, &DVector::from_vec(vec![1.0, 1.0, 1.0]), 100, 1e-12
).unwrap();
```

### Lyapunov exponents and chaos detection

```rust
use lau_dynamical_systems::strange_attractors::lyapunov_spectrum_continuous;
use lau_dynamical_systems::continuous::LorenzSystem;
use nalgebra::DVector;

let sys = LorenzSystem::chaotic();
let x0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
let spectrum = lyapunov_spectrum_continuous(&sys, &x0, 0.0, 50.0, 0.01, 10);

println!("Lyapunov exponents: {:?}", spectrum.exponents);
// Largest ≈ 0.9 (chaos confirmed)
println!("Kaplan-Yorke dimension: {:.3}", spectrum.ky_dimension);
// ≈ 2.06 (strange attractor, fractal dimension between 2 and 3)
```

### Bifurcation diagram

```rust
use lau_dynamical_systems::bifurcation::logistic_bifurcation;

let points = logistic_bifurcation((2.5, 4.0), 200, 500, 50);
for pt in &points {
    println!("r={:.3}, x={:.6}", pt.parameter, pt.value);
}
```

### Basin of attraction

```rust
use lau_dynamical_systems::basin::basin_1d_discrete_map;

let basin = basin_1d_discrete_map(
    |x| 2.0 * x * (1.0 - x), // r=2 logistic map
    0.01, 0.99, 50, 1000, 0.01
);
println!("Found {} attractor(s)", basin.attractors.len());
// All converge to x*=0.5
```

### Agent behavioral dynamics

```rust
use lau_dynamical_systems::agent_dynamics::{
    BehavioralDynamics, ApproachAvoidanceSystem
};
use nalgebra::DVector;

let sys = ApproachAvoidanceSystem::with_fixed_point();
let mut bd = BehavioralDynamics::new(vec!["approach".into(), "avoidance".into()]);

bd.discover_attractors(&sys, &[
    DVector::from_vec(vec![0.5, 0.5]),
    DVector::from_vec(vec![1.0, 1.0]),
], 0.0);

// Predict where a new agent ends up
let prediction = bd.predict_behavior(&sys, &DVector::from_vec(vec![0.1, 0.1]), 50.0, 0.01, 0.5);
```

---

## API Reference

### `continuous`

| Type / Function | Description |
|---|---|
| `ContinuousSystem` (trait) | Define an ODE: `dim()`, `f(x, t)`, optional `jacobian(x, t)` (defaults to finite differences). |
| `rk4(system, x0, t0, t1, dt)` | Classical 4th-order Runge-Kutta integration. Returns `IntegrationResult { times, states }`. |
| `LorenzSystem` | 3D chaotic system with `sigma`, `rho`, `beta`. `chaotic()` gives canonical parameters. |
| `LotkaVolterraSystem` | 2D predator-prey with `alpha`, `beta`, `gamma`, `delta`. |

### `discrete`

| Type / Function | Description |
|---|---|
| `DiscreteSystem` (trait) | Define a map: `dim()`, `step(x)`, optional `jacobian(x)`. |
| `iterate(system, x0, n)` | Run n iterations. Returns trajectory of length n+1. |
| `LogisticMap` | 1D: x → r·x·(1−x). Also has `step_scalar()`. |
| `HenonMap` | 2D: canonical chaotic map with a=1.4, b=0.3. |

### `fixed_points`

| Type / Function | Description |
|---|---|
| `Stability` | Enum: `Stable`, `Unstable`, `Marginal`, `Saddle`. |
| `FixedPointAnalysis` | Point + eigenvalues + stability classification. |
| `classify_continuous(eigenvalues, eps)` | From real parts: all negative → Stable, mixed → Saddle, etc. |
| `classify_discrete(eigenvalues, eps)` | From magnitudes: all |λ|<1 → Stable, etc. |
| `eigenvalues(matrix)` | QR-iteration eigenvalue computation for general real matrices. |
| `analyze_continuous_fixed_point(system, point, t)` | Jacobian → eigenvalues → stability. |
| `analyze_discrete_fixed_point(system, point)` | Same for discrete systems. |
| `find_fixed_point_continuous(system, x0, max_iter, tol)` | Newton's method on f(x)=0. |
| `find_fixed_point_discrete(system, x0, max_iter, tol)` | Newton's method on g(x)−x=0. |

### `bifurcation`

| Type / Function | Description |
|---|---|
| `BifurcationPoint` | `(parameter, value)` pair. |
| `logistic_bifurcation(r_range, r_steps, transient, record)` | Generate bifurcation diagram for logistic map. |
| `BifurcationType` | Enum: `SaddleNode`, `Transcritical`, `Pitchfork`, `Hopf`, `PeriodDoubling`. |
| `measure_logistic_period(r, transient, test)` | Measure orbit period at given r. |
| `detect_period_doubling(r, eps, expected_period)` | Detect period-doubling bifurcation. |
| `continuous_bifurcation(factory, param_range, ...)` | Bifurcation diagram for arbitrary continuous systems. |

### `limit_cycles`

| Type / Function | Description |
|---|---|
| `SectionCrossing` | Interpolated state + time at a Poincaré section crossing. |
| `poincare_section_crossings(states, times, component, value)` | Detect upward crossings of a hyperplane. |
| `estimate_period(crossings)` | Average time between consecutive crossings. |
| `floquet_multipliers(crossings, dim)` | Estimate Floquet multipliers from return map. |
| `floquet_multipliers_variational(system, x0, period, dt)` | Exact Floquet multipliers via variational equation integration. |

### `strange_attractors`

| Type / Function | Description |
|---|---|
| `LyapunovSpectrum` | Exponents (sorted largest-first) + Kaplan-Yorke dimension. |
| `lyapunov_spectrum_continuous(system, x0, t0, t1, dt, renorm_steps)` | Benettin algorithm with Gram-Schmidt reorthonormalization. |
| `lyapunov_spectrum_discrete(system, x0, n_steps)` | Same for discrete maps. |
| `kaplan_yorke_dimension(exponents)` | Compute D_KY from Lyapunov exponents. |
| `correlation_dimension(points, max_r, n_bins)` | Grassberger-Procaccia algorithm for fractal dimension estimation. |

### `poincare`

| Type / Function | Description |
|---|---|
| `PoincareMap` | Stores crossings, section definition. |
| `PoincareMap::from_trajectory(...)` | Build from integrated states. |
| `return_map_pairs(component)` | Get (x_n, x_{n+1}) pairs for plotting. |
| `section_points()` | Project crossings onto section coordinates. |
| `compute_poincare_map(system, x0, t0, t1, dt, component, value)` | Integrate + build map. |

### `basin`

| Type / Function | Description |
|---|---|
| `BasinOfAttraction` | Initial conditions + attractor index + discovered attractors. |
| `basin_1d_continuous(system, x_min, x_max, n, component, fill, t_span, dt, tol)` | 1D grid sweep for ODE systems. |
| `basin_2d_discrete(system, x_range, y_range, nx, ny, n_iter, tol)` | 2D grid sweep for iterated maps. |
| `basin_1d_discrete_map(map, x_min, x_max, n, n_iter, tol)` | 1D sweep for scalar maps. |

### `agent_dynamics`

| Type / Function | Description |
|---|---|
| `BehavioralAttractor` | Label + state + stability + optional Lyapunov spectrum. `is_stable()`, `is_chaotic()`. |
| `BehavioralDynamics` | Collection of attractors with `discover_attractors()`, `predict_behavior()`, `stable_pattern_count()`, `chaotic_pattern_count()`. |
| `ApproachAvoidanceSystem` | 2D behavioral model: drive, conflict, damping. Implements `ContinuousSystem`. |

---

## How It Works

The library builds from primitives upward:

1. **Continuous systems** implement `ContinuousSystem` with a vector field f(x,t) and optional analytic Jacobian. The RK4 integrator steps forward with classical 4th-order accuracy.

2. **Discrete systems** implement `DiscreteSystem` with a step function. The `iterate` function accumulates trajectories.

3. **Fixed-point analysis** uses the Jacobian at a point to compute eigenvalues. For continuous systems, stability depends on the sign of the real parts (all negative → stable). For discrete systems, on the magnitude relative to 1 (all inside unit circle → stable). Newton's method finds fixed points by solving f(x)=0 (continuous) or g(x)−x=0 (discrete).

4. **Bifurcation diagrams** sweep a parameter, discard transients, and record the orbit. Period measurement detects periodic windows in the logistic map.

5. **Limit cycles** are analyzed via Poincaré sections: the trajectory is integrated, and crossings of a hyperplane are detected by sign changes with linear interpolation. Return times give the period. Floquet multipliers come from integrating the variational equation (fundamental matrix Φ(T)) alongside the ODE; the eigenvalues of Φ(T) are the multipliers.

6. **Lyapunov exponents** use the Benettin algorithm: perturbation vectors are evolved alongside the trajectory by the Jacobian, and periodically reorthonormalized via Gram-Schmidt. The logarithm of the stretching factors, divided by total time, gives the exponents. The Kaplan-Yorke dimension interpolates between exponents to estimate the attractor's fractal dimension.

7. **Correlation dimension** uses the Grassberger-Procaccia algorithm: compute pairwise distances, build C(r) (fraction of pairs closer than r), and fit a line in log-log space.

8. **Poincaré maps** reduce continuous dynamics to a discrete return map. Each crossing of the section hyperplane becomes a point in the map; the return map pairs (x_n, x_{n+1}) reveal the attractor's structure.

9. **Basins of attraction** sweep a grid of initial conditions, integrate or iterate each, and cluster final states into attractors by proximity.

10. **Agent dynamics** applies all of the above to behavioral modeling: an agent's state evolves according to a dynamical system, and stable fixed points correspond to persistent behavioral patterns.

---

## The Math

### RK4 Integration
The classical 4th-order Runge-Kutta method for dx/dt = f(x, t):
- k₁ = f(x, t)
- k₂ = f(x + h·k₁/2, t + h/2)
- k₃ = f(x + h·k₂/2, t + h/2)
- k₄ = f(x + h·k₃, t + h)
- x(t+h) = x(t) + h·(k₁ + 2k₂ + 2k₃ + k₄)/6

Local truncation error is O(h⁵).

### Linear Stability
At a fixed point x* where f(x*) = 0, the Jacobian J = ∂f/∂x determines stability:
- **Continuous:** Stable if all eigenvalues λ have Re(λ) < 0. Saddle if some Re(λ) > 0 and some Re(λ) < 0.
- **Discrete:** Stable if all eigenvalues |λ| < 1. Saddle if some inside and some outside the unit circle.

### Lyapunov Exponents (Benettin Algorithm)
For an n-dimensional system, evolve n orthonormal tangent vectors w₁, ..., wₙ alongside the trajectory. Each vector is stretched by the Jacobian. Periodically apply Gram-Schmidt to reorthonormalize. The i-th Lyapunov exponent is:

λᵢ = lim(T→∞) (1/T) Σₖ ln ‖wᵢ⁽ᵏ⁾‖ / ‖ŵᵢ⁽ᵏ⁾‖

where ŵᵢ is the vector before normalization at step k.

### Kaplan-Yorke Dimension
Given Lyapunov exponents λ₁ ≥ λ₂ ≥ ... ≥ λₙ, find the largest j such that Σᵢ₌₁ʲ λᵢ ≥ 0. Then:

D_KY = j + (Σᵢ₌₁ʲ λᵢ) / |λ_{j+1}|

This estimates the information dimension of the attractor.

### Logistic Map Bifurcations
The logistic map x → r·x·(1−x) undergoes period-doubling cascade:
- r < 1: fixed point at 0
- 1 < r < 3: stable fixed point x* = 1 − 1/r
- r ≈ 3: period-2 cycle appears
- r ≈ 3.45: period-4
- r ≈ 3.57: onset of chaos (accumulation point)
- r = 4: fully chaotic, Lyapunov exponent = ln(2) ≈ 0.693

### Floquet Theory
For a periodic orbit with period T, the monodromy matrix M = Φ(T) maps perturbations after one full revolution. Its eigenvalues μᵢ (Floquet multipliers) determine stability: the orbit is stable iff |μᵢ| < 1 for all i ≠ 1 (one multiplier is always 1, corresponding to perturbations along the orbit).

### Grassberger-Procaccia Algorithm
The correlation integral C(r) = (2/N(N−1)) · #{(i,j) : ‖xᵢ − xⱼ‖ < r}. For a fractal attractor, C(r) ∝ r^D₂ where D₂ is the correlation dimension, estimated by the slope of ln C(r) vs ln r.

---

## License

MIT
