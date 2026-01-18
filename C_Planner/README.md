# Frenet Optimal Planner - C++ Implementation

## Project Structure

The C++ implementation follows the same modular structure as the Python version:

```
C_Planner/
├── CMakeLists.txt                    # Build configuration
├── Frenet_Planner.h                  # Main planner class header
├── Frenet_Planner.cpp                # Main planner implementation
├── Stats.cpp                         # Statistics tracking
└── common/
    ├── scenario/
    │   └── frenet.h                  # FrenetState, FrenetTrajectory structs
    ├── geometry/
    │   ├── polynomial.h/cpp          # QuarticPolynomial, QuinticPolynomial
    │   └── cubic_spline.h/cpp        # CubicSpline1D, CubicSpline2D
    ├── cost/
    │   └── cost_function.h/cpp       # CostFunction for trajectory evaluation
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

### 2. **VehicleParams** (Frenet_Planner.h)
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
- `get_samples()`: Generates all combinations of (d, s_d, t) parameters
- `calc_frenet_paths()`: Creates trajectories using polynomial curves
  - Uses **QuinticPolynomial** for lateral motion
  - Uses **QuarticPolynomial** for longitudinal motion
- `calc_global_paths()`: Converts Frenet to Cartesian coordinates
  - Uses **CubicSpline2D** for centerline interpolation
  - Calculates yaw and curvature
- `check_constraints()`: Validates speed and acceleration limits
- `check_collision_multithread()`: Array-based collision detection
  - Takes pre-processed obstacle arrays from Python
  - Returns collision-free trajectories
- `plan()`: Main entry point orchestrating the full pipeline
- `generate_frenet_frame()`: Initializes centerline spline

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
Evaluates trajectory quality:
- `cost_velocity_offset()`: Speed deviation from target
- `cost_acceleration()`: Smoothness of motion
- `cost_jerk()`: Comfort (rate of acceleration change)
- `cost_lane_center_offset()`: Staying near lane center
- `cost_total()`: Weighted sum of all costs

### 8. **Collision Detection** (common/collision/)
Polygon-based collision checking:
- `point_in_polygon()`: Ray casting algorithm
- `segments_intersect()`: Line segment intersection
- `aabb_collision()`: Axis-aligned bounding box pre-check
- `polygon_collision()`: Complete polygon intersection test
- `compute_vehicle_polygon()`: Vehicle footprint at given pose

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
calc_frenet_paths() → Polynomial trajectory generation
    ↓
calc_global_paths() → Frenet → Cartesian conversion
    ↓
check_constraints() → Validate speed/acceleration
    ↓
check_collision_multithread() → Array-based collision checking
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

