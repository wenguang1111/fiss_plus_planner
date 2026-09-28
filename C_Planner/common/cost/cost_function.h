#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include "../scenario/frenet.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

class CostFunction {
public:
    // Importance weights. Each cost term below is first non-dimensionalized to a
    // comparable ~[0,1]-per-timestep scale using the reference constants further
    // down, so these weights express *relative priority* between terms rather than
    // having to also absorb unit conversions between e.g. m/s^2 and m.
    double w_T;   // weight for terminal time preference
    double w_V;   // weight for velocity tracking
    double w_A;   // weight for acceleration comfort
    double w_J;   // weight for jerk comfort
    double w_D;   // weight for obstacle-distance safety
    double w_LC;  // weight for lane center offset

    // Reference scales used only to non-dimensionalize the raw physical quantities
    // above. These are cost-shaping references, not hard dynamic limits (those are
    // enforced separately, e.g. in the planner's check_constraints/collision check).
    double max_speed;          // velocity normalization reference [m/s]
    double max_accel_ref;      // acceleration normalization reference [m/s^2]
    double max_jerk_ref;       // jerk normalization reference [m/s^3]
    double max_lat_offset_ref; // lane-center-offset normalization reference [m]
    double time_horizon_ref;   // terminal-time normalization reference [s]
    double d_safe;             // safety-distance threshold for the obstacle risk ramp [m]

    CostFunction(const std::string& cost_type = "WX1");
    
    double cost_terminal_time(double terminal_time);
    double cost_velocity_offset(const std::vector<double>& vels, double v_target);
    double cost_acceleration(const std::vector<double>& accels);
    double cost_jerk(const std::vector<double>& jerks);
    double cost_lane_center_offset(const std::vector<double>& offsets);
    double cost_dist_obstacle(const double* obstacles_array,
                              const int* num_vertices_array,
                              int num_time_steps,
                              int num_obstacles,
                              int max_vertices,
                              const FrenetTrajectory& traj,
                              int time_step_now = 0);
    double cost_singleTrajectory(const FrenetTrajectory& traj,
                                 double target_speed,
                                 const double* obstacles_array,
                                 const int* num_vertices_array,
                                 int num_time_steps,
                                 int num_obstacles,
                                 int max_vertices,
                                 int time_step_now = 0);
    void calc_cost(std::vector<FrenetTrajectory>& fplist,
                   double target_speed,
                   const double* obstacles_array,
                   const int* num_vertices_array,
                   int num_time_steps,
                   int num_obstacles,
                   int max_vertices,
                   int time_step_now = 0);
    double cost_total(const FrenetTrajectory& traj, double target_speed);
};

#endif // COST_FUNCTION_H
