#ifndef FRENET_PLANNER_H
#define FRENET_PLANNER_H

#include <vector>
#include <memory>
#include <tuple>
#include <thread>
#include <mutex>
#include "common/scenario/frenet.h"
#include "common/geometry/cubic_spline.h"
#include "common/geometry/polynomial.h"
#include "common/cost/cost_function.h"

struct SettingParameters {
    // time resolution between two planned waypoints
    double tick_t;  // time tick [s]
    
    // sampling parameters
    double max_road_width;       // maximum road width [m]
    int num_width;              // road width sampling number

    double highest_speed;        // highest sampling speed [m/s]
    double lowest_speed;         // lowest sampling speed [m/s]
    int num_speed;              // speed sampling number
    
    double min_t;                // min prediction time [s]
    double max_t;                // max prediction time [s]
    int num_t;                  // time sampling number

    bool check_obstacle;        // True if check collision with obstacles
    bool check_boundary;        // True if check collision with road boundaries

    // Constructor with default values matching Python implementation
    SettingParameters(int num_width_param = 5, int num_speed_param = 5, int num_t_param = 5) 
        : tick_t(0.1),
          max_road_width(3.5),
          num_width(num_width_param),
          highest_speed(14.0),
          lowest_speed(0.0),
          num_speed(num_speed_param),
          min_t(3.0),
          max_t(5.0),
          num_t(num_t_param),
          check_obstacle(true),
          check_boundary(true) {}
};

// Vehicle parameters struct
struct VehicleParams {
    double l;          // length [m]
    double w;          // width [m]
    double a;          // distance from base_link to CoG [m]
    double b;          // distance from CoG to front_link [m]
    double T_f;        // front track width [m]
    double T_r;        // rear track width [m]
    double max_speed;  // maximum speed [m/s]
    double max_accel;  // maximum acceleration [m/ss]
    double max_steering_angle;  // maximum steering angle [rad]
    double max_steering_rate;   // maximum steering rate [rad/s]
};

// Result structure for plan_multithread
struct PlanResult {
    std::vector<FrenetTrajectory> frenet_paths;       // All generated frenet paths
    std::vector<FrenetTrajectory> collision_free_paths; // Paths that passed collision check
};

// Statistics structure for planning
struct PlanStats {
    int num_trajs_generated;    // Number of frenet paths generated
    int num_trajs_validated;    // Number of trajectories that passed constraint check
    int num_collision_checks;   // Number of collision checks performed
    int num_FOP_intervention; // Number of times FOP was used for intervention (if applicable)
    
    PlanStats() : num_trajs_generated(0), num_trajs_validated(0), num_collision_checks(0), num_FOP_intervention(0) {}
    
    // Accumulate stats from another PlanStats
    PlanStats& operator+=(const PlanStats& other) {
        num_trajs_generated += other.num_trajs_generated;
        num_trajs_validated += other.num_trajs_validated;
        num_collision_checks += other.num_collision_checks;
        num_FOP_intervention += other.num_FOP_intervention;
        return *this;
    }
};

class Frenet_Planner {
public:
    SettingParameters settings;
    VehicleParams vehicle_params;
    CostFunction cost_function;
    CubicSpline2D* cubic_spline;
    FrenetTrajectory best_traj;
    std::vector<std::vector<FrenetTrajectory>> all_trajs;
    std::vector<FrenetTrajectory> last_fplist;
    
    // Statistics for the last planning cycle
    PlanStats last_stats;
    
    // Obstacle data members
    const double* obstacles_array;
    const int* num_vertices_array;
    int num_time_steps;
    int num_obstacles;
    int max_vertices;
    
    Frenet_Planner(const SettingParameters& settings_param, 
                   const VehicleParams& vehicle_param,
                   const double* obs_array,
                   const int* num_verts,
                   int n_time_steps,
                   int n_obstacles,
                   int max_verts);
    ~Frenet_Planner();
    void recordObstacleArray();
    void recordTrajectory(const FrenetTrajectory& traj);
    
    // Generate sampling parameters (d, s_d, t)
    std::vector<std::tuple<double, double, double>> get_samples();
    
    // Calculate Frenet frame trajectories
    std::vector<FrenetTrajectory> calc_frenet_paths(const FrenetState& frenet_state,
                                                     const std::vector<std::tuple<double, double, double>>& samples);
    
    // Convert Frenet paths to global (x, y) coordinates using cubic spline
    std::vector<FrenetTrajectory> calc_global_paths(const std::vector<FrenetTrajectory>& fplist);
    
    // Check trajectory constraints (speed, acceleration, etc.)
    std::vector<FrenetTrajectory> check_constraints(const std::vector<FrenetTrajectory>& trajs);
    
    // Multi-threaded collision detection using pre-processed obstacle data
    std::vector<FrenetTrajectory> check_collision_multithread(const std::vector<FrenetTrajectory>& trajs,
                                                               int time_step_now = 0);
    
    // Main planning function - simplified interface
    FrenetTrajectory plan(const FrenetState& frenet_state,
                         double max_target_speed,
                         int time_step_now = 0,
                         int num_threads=1);

    // Main planning function using externally provided sampling parameters (d, s_d, t)
    FrenetTrajectory best_traj_generation(
        const FrenetState& frenet_state,
        const std::vector<std::tuple<double, double, double>>& samples,
        double max_target_speed,
        int time_step_now = 0,
        int num_threads = 1);

    // Multithreaded planning function
    PlanResult plan_multithread(
        const std::vector<std::vector<std::tuple<double, double, double>>>& samples_per_thread_vec,
        const FrenetState& frenet_state,
        int time_step_now);

    std::vector<FrenetTrajectory> getAllSuccessfulTrajectories() const {
        return last_fplist;
    }
    
    // Get statistics from the last planning cycle
    PlanStats get_stats() const {
        return last_stats;
    }
    
    // Generate Frenet frame from centerline points
    void generate_frenet_frame(const double* centerline_pts, int num_points, int pts_dim);
};

#endif // FRENET_PLANNER_H
