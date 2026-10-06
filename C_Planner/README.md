# Frenet Optimal Planner - C++ Implementation

## Project Structure

The C++ implementation follows the same modular structure as the Python version:

```
C_Planner/
├── CMakeLists.txt                    # Build configuration
├── Frenet_Planner.h                  # Main planner class header
├── Frenet_Planner.cpp                # Main planner implementation
├── Fiss_Plus_Planner.h/cpp           # FISS+ search on top of the same backend
├── Iterative_Sampling_Planner.h/cpp  # FOP with an iteratively refined Gaussian sampler
├── CEM_Planner.h/cpp                 # cross-entropy update of that sampler
├── MPPI_Planner.h/cpp                # MPPI-style update of that sampler
└── common/
    ├── scenario/
    │   ├── frenet.h                  # FrenetState, FrenetTrajectory structs
    │   ├── obstacles.h               # ObstacleView over the Python obstacle arrays
    │   ├── road_profile.h/cpp        # Lane width / road extents along s
    │   └── vehicle_params.h          # VehicleParams
    ├── geometry/
    │   ├── polynomial.h/cpp          # QuarticPolynomial, QuinticPolynomial
    │   └── cubic_spline.h/cpp        # CubicSpline1D, CubicSpline2D
    ├── cost/
    │   └── cost_function.h/cpp       # FISS+ Eq. (4) planning cost, Eq. (7)+(8) scenario cost
    ├── sampling/
    │   └── gaussian_proposal.h/cpp   # Diagonal Gaussian over the unit search space
    ├── evaluator/
    │   ├── plan_stats.h              # PlanStats / TimingStats counters
    │   └── trajectory_evaluator.h/cpp # Shared generate -> transform -> check -> cost backend
    ├── collision/
    │   └── collision_checker.h/cpp   # Polygon collision detection
    └── utils/
        └── array_utils.h/cpp        # Array utilities for data conversion
```

## Key Classes and Functions

### 1. **SettingParameters** (Frenet_Planner.h)
Configuration struct with sampling parameters:
- `tick_t`: Time resolution (0.1s)
- `max_road_width`: Road width sampling (3.5m)
- `num_width`, `num_speed`, `num_t`: Sampling counts
- `highest_speed`, `lowest_speed`: Speed bounds
- `min_t`, `max_t`: Time horizon bounds
- Obstacle and boundary checking flags

### 2. **VehicleParams** (common/scenario/vehicle_params.h)
Vehicle specifications:
- Dimensions: length, width
- Kinematic parameters: wheelbase, track widths
- Constraints: max speed, acceleration, steering angle

### 3. **FrenetTrajectory** (common/scenario/frenet.h)
Stores trajectory data in both Frenet and world frames:
- **Frenet coordinates**: s, s_d, s_dd, s_ddd, d, d_d, d_dd, d_ddd
- **World coordinates**: x, y, yaw, ds, curvature (c, c_d, c_dd)
- **Cost and status**: cost_final, collision_passed, constraint_passed

### 4. **Frenet_Planner** (Main Class)

#### Methods:
- `search_space(current_s)`: Sampling bounds of (d, s_d, t), shared by every planner
- `get_samples()`: Generates all combinations of (d, s_d, t) parameters
- `make_context()`: Builds the `PlanningContext` of one planning cycle
- `plan()`: Main entry point orchestrating the full pipeline
- `best_traj_generation()`: Same pipeline for externally provided samples
- `evaluate_samples()`: Per-sample `EvaluationResult` (feasibility, violation, cost)
- `generate_frenet_frame()`: Initializes centerline spline

### TrajectoryEvaluator (common/evaluator/)

Every planner (FOP, FISS+, and later Random / CEM / learned samplers) evaluates its
candidates through one `TrajectoryEvaluator`, so generator, constraints, collision
check and objective are identical:

- `generate()`: quintic lateral / quartic longitudinal polynomial for z = (d, s_d, t)
- `frenet_cost()`: Eq. (4) without J_D (FISS+ ranks with this before any check)
- `check_feasibility()`: Cartesian transform, dynamic limits, road boundary, polygon collision
- `evaluate()`: `check_feasibility()` + planning objective; FOP adds J_D

`EvaluationResult` holds the feasibility flag, the first failed check, the
constraint violation V and the cost terms. Per check, V is the share of the
horizon from the first violating sample on, (N - k_first) / N, so V is 0 exactly
when every hard check passes and a trajectory that fails later counts as closer
to feasible; `ranks_before()` orders trajectories lexicographically by
(feasible, V, J). `EvalMode::kEarlyExit` (used by all planners) stops at the first
failed check and the collision check at the first colliding step;
`EvalMode::kFullViolation` runs every check and computes J for all candidates
(teacher data). All counters and stage timings are incremented inside the
evaluator (`PlanStats`).

### Lateral motion model and Frenet -> Cartesian transform

Below `low_speed_threshold` (start speed, default 4 m/s as in the CommonRoad
reactive planner) the lateral motion is a quintic in the travelled arc length,
d(s(t)) over sigma = s - s0 in [0, max(s(T) - s0, low_speed_min_lateral_length)]
(Werling's thesis Sec. 3.5.1), so the vehicle cannot move sideways while standing.
Above it the lateral motion is the usual quintic d(t). Every trajectory stores
d' = dd/ds and d'' (`d_s`, `d_ss`; high speed via Werling (A.7)/(A.8), held at
standstill), and `to_global()` computes heading and curvature from them with
Werling's Appendix A.1, (A.3)/(A.5), instead of finite differences of x, y. This
keeps heading and curvature defined at v = 0 (waiting is a feasible candidate:
v_end = 0 is part of every sampling space). Trajectories with s_d < -0.01 m/s
(reversing) are rejected by the dynamic constraint. `FrenetState` carries d', d''
(`d_s`, `d_ss`, NaN if unknown) so they stay continuous across cycles at standstill.

### Safe following distance (clearance)

All planners share a check between the constraint and the polygon collision check
(`check_clearance` and the `clearance_*` fields of `SettingParameters`, `SAFETY`
section of `cfgs/demo_config.yaml`). Python projects every obstacle polygon onto
the reference line once per frame (`planners/common/scenario/obstacle_frenet.py`)
and passes (s_min, s_max, l_min, l_max) per time step via
`set_obstacle_frenet_bounds()`. An obstacle is ahead in the path of a candidate at
step k if its lateral extent overlaps the ego's (plus `clearance_lateral_margin`)
and its rear is beyond the ego center; from `clearance_grace_time` on, the gap
from the ego front to its rear must reach `clearance_min_gap + clearance_time_gap * v_k`.
If the gap the ego would keep at its current speed is smaller (recorded traffic
often follows at 1.5-2 s), the requirement starts at that gap and grows to the
target within `clearance_recovery_time`, quadratically in time. If no candidate
satisfies it, the cycle is re-planned without it (`clearance_fallback`,
counted in `num_clearance_fallbacks`).

### Iterative sampling planners (CEM, MPPI)

`Iterative_Sampling_Planner` replaces FOP's grid by a Gaussian proposal over
(d, s_d, t) in unit coordinates of `search_space()`: `num_iterations` rounds of
`population` samples, each evaluated by `evaluate_batch()` (the FOP backend,
multi-threaded within a round), warm-started at the previous cycle's solution.
It returns the best feasible trajectory of all rounds. `CEM_Planner` refits the
proposal to the elite set ranked by (feasible, V, J); `MPPI_Planner` moves it
towards the exp(-score / gamma)-weighted mean and covariance with learning rate
eta. Settings: `CEMSettings`, `MPPISettings` (Python: `CEM` / `MPPI` sections of
`cfgs/demo_config.yaml`, planners `CEM_CPP` / `MPPI_CPP`).

### FOP_CPP road-width profiles

The Python FOP wrapper passes CommonRoad geometry once per reference frame using
`set_road_profile(s, lane_width, left_extent, right_extent)`. The four arrays have
matching lengths; `s` is strictly increasing and spans the spline's full chord-length
coordinate range. Extents are positive distances from the reference line to the
left and right edges of the same-direction roadway. They need not be symmetric.

Each `plan()` call samples lateral endpoints inside the current lane using the
linearly interpolated width at the initial state's `s`, minus vehicle width.
`get_samples(current_s=0.0)` exposes the same sampling behavior. A lane narrower
than the vehicle produces no samples. Sampling at the current position can be
conservative when the road widens farther ahead.

When `check_boundary` is enabled and a profile is present, every candidate point
is checked against the left/right extents at its own `s`. The lateral vehicle
envelope includes width, length, and heading relative to the reference tangent.
Bounds are never relaxed to the starting offset. Out-of-range or incomplete
horizons are rejected, and boundary rejections populate `num_rejected_offroad`.
This is a discrete local Frenet envelope check, not exact vehicle-polygon
containment on curved roads or continuous collision checking between time steps.

The wrapper accepts existing centerline columns `[x, y, yaw, lane_width, left, right]`.
Four-column input uses half the lane width for each road edge. XY-only callers
retain fixed `max_road_width` sampling and have no map-boundary check. Generating
a new frame clears the old profile. Ego-frame ML collection preserves the width
and extent columns when rotating/translating the reference. The FISS+ wrapper
supplies the same profile and FISS+ samples inside the same `search_space()`.

Regression tests (after rebuilding both extensions):

```bash
poetry run python -m unittest discover -s tests -p 'test_cpp_*.py' -v
```

### 5. **Polynomial Classes** (common/geometry/)
- **QuarticPolynomial**: 4th-order polynomial for longitudinal motion
  - Takes: initial state (x, vx, ax), final velocity (vx_e), final acceleration, time
  - Methods: calc_point(), calc_first/second/third_derivative()
  
- **QuinticPolynomial**: 5th-order polynomial for lateral motion
  - Takes: initial state (x, vx, ax), final state (x_e, vx_e, ax_e), time
  - Methods: same as QuarticPolynomial

### 6. **CubicSpline Classes** (common/geometry/)
- **CubicSpline1D**: 1D spline interpolation
  - Solves tridiagonal matrix system for smooth curves
  - Methods: calc_position(), calc_first/second_derivative()

- **CubicSpline2D**: 2D spline (Frenet reference path)
  - Parametric curves for centerline
  - Methods: calc_position(), calc_yaw(), calc_curvature()

### 7. **CostFunction** (common/cost/)
Cost of FISS+ (Sun et al., IROS 2023), no normalization. Weights
w_T=0.1, w_V=0.01, w_A=0.1, w_J=0.1, w_D=1, w_dist=0.1, w_LC=10 (CommonRoad WX1
with w_T, w_V, w_D and w_dist re-weighted, see `CostWeights`).
Integrals are sums over the samples times dt:
- `frenet_terms()`: Eq. (4), J_T = w_T (t_max - t_f), J_V, J_A, J_J, J_LC
- `obstacle_term()`: J_D of Eq. (7), max over obstacles in front of exp(-w_dist d_i)
- `scenario_cost()`: Eq. (7) + (8) on the executed trajectory (`frenet_planner_cpp.scenario_cost`);
  the terminal term is J_ter = w_T * t_f (time taken), Eq. (8) as printed has the sign of
  Eq. (4) and would reward slower arrival

FOP plans with Eq. (4) + J_D, FISS+ with Eq. (4) only (it ranks before the
Cartesian transform). Every executed scenario is scored with `scenario_cost()`.

### 8. **Collision Detection** (common/collision/)
Polygon-based collision checking:
- `point_in_polygon()`: Ray casting algorithm
- `segments_intersect()`: Line segment intersection
- `aabb_collision()`: Axis-aligned bounding box pre-check
- `polygon_collision()`: Complete polygon intersection test
- `compute_vehicle_polygon()`: Vehicle footprint at given pose
- `polygon_penetration_depth()`: SAT overlap depth (collision violation)
- `check_trajectory_collision()`: Footprint vs. obstacles over a trajectory

### 9. **Array Utilities** (common/utils/)
Helper functions for NumPy/Python array compatibility:
- `prepare_trajectory_array()`: Converts vector of trajectories to flat array
- `get_trajectory_point()`: Safe indexed access to trajectory data
- `get_obstacle_vertex()`: Safe indexed access to obstacle data

## Data Flow

```
Python Input (numpy arrays)
    ↓
generate_frenet_frame(centerline_pts)
    ↓ Creates CubicSpline2D
plan(frenet_state, obstacles_array, num_vertices_array)
    ↓
get_samples() → Generate (d, s_d, t) combinations
    ↓
TrajectoryEvaluator::generate() → Polynomial trajectory generation
    ↓
TrajectoryEvaluator::evaluate() → Cartesian transform, constraints,
                                  polygon collision, cost
    ↓
Select minimum cost trajectory
    ↓
Return best_traj (FrenetTrajectory)
```

## Compilation

### Requirements
- C++17 compiler
- Eigen3 library (for matrix operations)

### Build
```bash
cd C_Planner
mkdir build
cd build
cmake ..
make
```

This generates:
- `libFrenetOptimalPlanner.so` (shared library)
- `libFrenetOptimalPlanner_static.a` (static library)

## Array Format for Collision Detection

**Obstacles Array Layout:**
- Shape: `[num_time_steps][num_obstacles][max_vertices][2]`
- Each value is a float representing (x, y) coordinates
- Stored in C-contiguous flat array

**Vertex Count Array Layout:**
- Shape: `[num_time_steps][num_obstacles]`
- Each value is an int representing actual vertex count for that polygon

## Multithreading Features

The collision checker supports:
- **Current**: Serial implementation (single-threaded collision checking)
- **Future Enhancement**: Parallelization using:
  - Thread pool for trajectory checking
  - OpenMP pragmas for SIMD vectorization
  - Lock-free data structures for trajectory results

## Key Design Decisions

1. **Float vs Double**: Using `float` for memory efficiency and NumPy compatibility
2. **STL Vectors**: Used for dynamic-sized data (trajectories, samples)
3. **Eigen3 Matrices**: Used only for linear system solving in polynomials/splines
4. **Raw Pointers in Arrays**: Flat arrays from Python remain as raw pointers for zero-copy efficiency
5. **Struct-based Design**: Emphasis on data locality and cache efficiency

## Integration with Python

The C++ library can be called from Python via:
- ctypes (direct C library binding)
- pybind11 (modern Python-C++ bindings)
- CFFI (C Foreign Function Interface)

Typical usage:
```python
# Python calls C++ planner
planner.generate_frenet_frame(centerline_pts)
best_trajectory = planner.plan(
    frenet_state, 
    target_speed,
    obstacles_array,
    num_vertices_array
)
```

## Performance Considerations

- **Trajectory Generation**: O(num_samples * num_time_steps)
- **Global Path Conversion**: O(num_trajs * num_time_steps)
- **Collision Detection**: O(num_trajs * num_obstacles * num_vertices) per time step
- **Memory**: All trajectory data stored in vectors for cache locality

## Future Enhancements

1. ✓ Multithreaded collision detection with OpenMP
2. ✓ SIMD optimizations for distance calculations
3. ✓ GPU acceleration for large-scale planning
4. ✓ Incremental spline updates
5. ✓ Cost function customization via function pointers
6. ✓ C++20 concepts for type safety

## Testing

Example test structure (to be implemented):
```cpp
TEST(FrenetPlannerTest, SamplingTest) {
    SettingParameters settings;
    VehicleParams vehicle;
    Frenet_Planner planner(settings, vehicle);
    
    auto samples = planner.get_samples();
    EXPECT_EQ(samples.size(), 5*5*5);  // 125 samples
}
```
