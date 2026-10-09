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
#include "common/evaluator/plan_stats.h"
#include "common/evaluator/trajectory_evaluator.h"
#include "common/scenario/obstacle_frenet.h"
#include "common/scenario/obstacles.h"
#include "common/scenario/road_profile.h"
#include "common/scenario/search_space.h"
#include "common/scenario/vehicle_params.h"

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

    // Low-speed lateral model (Werling's thesis Sec. 3.5.1, CommonRoad reactive planner):
    // below low_speed_threshold the lateral motion is a quintic in the travelled arc length,
    // d(s(t)), so the vehicle cannot move sideways without moving forward.
    double low_speed_threshold;            // [m/s] start speed below which d(s) is used
    double low_speed_min_lateral_length;   // [m] lower bound of the lateral arc length

    bool check_obstacle;        // True if check collision with obstacles
    bool check_boundary;        // True if check collision with road boundaries

    // Safe following distance to obstacles ahead in the ego path (needs
    // set_obstacle_frenet_bounds): from clearance_grace_time on, every trajectory step
    // keeps gap >= clearance_min_gap + clearance_time_gap * v, gap measured from the ego
    // front to the obstacle's rear along the reference line.
    bool check_clearance;
    double clearance_time_gap;        // [s]
    double clearance_min_gap;         // [m]
    double clearance_lateral_margin;  // [m] added on each ego side for the in-path test
    double clearance_grace_time;      // [s] start of the horizon without the requirement
    double clearance_recovery_time;   // [s] a gap below the target is restored within this time
                                      // (requirement grows from the gap kept at the current
                                      // speed to the target, quadratically in time)
    bool clearance_fallback;          // if no candidate keeps the gap, plan without it

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
          low_speed_threshold(4.0),
          low_speed_min_lateral_length(5.0),
          check_obstacle(true),
          check_boundary(true),
          check_clearance(true),
          clearance_time_gap(2.0),
          clearance_min_gap(3.0),
          clearance_lateral_margin(0.2),
          clearance_grace_time(1.0),
          clearance_recovery_time(3.0),
          clearance_fallback(true) {}
};

// Outcome of evaluating a batch of samples z = (d, s_d, t)
struct BatchResult {
    std::vector<EvaluationResult> results;   // one per sample, in sample order
    std::vector<FrenetTrajectory> feasible;  // feasible trajectories with cost, in sample order
};

class Frenet_Planner {
public:
    SettingParameters settings;
    VehicleParams vehicle_params;
    CostFunction cost_function;
    CubicSpline2D* cubic_spline;
    RoadProfile road_profile;
    ObstacleView obstacles;
    ObstacleFrenetBounds obstacle_frenet;
    FrenetTrajectory best_traj;
    std::vector<std::vector<FrenetTrajectory>> all_trajs;
    std::vector<FrenetTrajectory> last_fplist;
    
    // Statistics for the last planning cycle
    PlanStats last_stats;
    
    Frenet_Planner(const SettingParameters& settings_param, 
                   const VehicleParams& vehicle_param,
                   const double* obs_array,
                   const int* num_verts,
                   int n_time_steps,
                   int n_obstacles,
                   int max_verts);
    virtual ~Frenet_Planner();  // base of FISS+, CEM and MPPI
    void recordObstacleArray();
    void recordTrajectory(const FrenetTrajectory& traj);
    
    // Sampling region at the current position: lateral bounds are the lane width at
    // current_s minus the vehicle width. False if the vehicle does not fit.
    bool search_space(double current_s, SearchSpace& space) const;

    // Generate the sampling grid (d, s_d, t) inside search_space(current_s)
    std::vector<std::tuple<double, double, double>> get_samples(double current_s = 0.0);
    
    // The planning problem of one cycle starting at `start`. Every planner evaluates its
    // candidates with a TrajectoryEvaluator built from this context.
    PlanningContext make_context(const FrenetState& start, double v_des, int time_step_now,
                                 bool use_obstacle_cost) const;

    // Main planning function - simplified interface.
    // max_target_speed: upper bound of the sampled terminal speed. desired_speed: v_des of
    // J_V; a negative value means v_des = max_target_speed.
    FrenetTrajectory plan(const FrenetState& frenet_state,
                         double max_target_speed,
                         int time_step_now = 0,
                         int num_threads=1,
                         double desired_speed = -1.0);

    // Main planning function using externally provided sampling parameters (d, s_d, t)
    FrenetTrajectory best_traj_generation(
        const FrenetState& frenet_state,
        const std::vector<std::tuple<double, double, double>>& samples,
        double max_target_speed,
        int time_step_now = 0,
        int num_threads = 1,
        double desired_speed = -1.0);

    // Trajectory of sample z = (d, s_d, t) from frenet_state with the shared generator and
    // Frenet -> Cartesian transform, without any check (also for infeasible samples), e.g. to
    // rebuild stored candidates exactly.
    FrenetTrajectory generate_trajectory(const FrenetState& frenet_state, double d, double s_d, double t,
                                         int time_step_now = 0) const;

    // Evaluates every sample with the shared evaluator (single thread) and returns
    // one result per sample, e.g. for offline teacher data or violation analysis.
    std::vector<EvaluationResult> evaluate_samples(
        const FrenetState& frenet_state,
        const std::vector<std::tuple<double, double, double>>& samples,
        double max_target_speed,
        int time_step_now = 0,
        bool full_violation = true,
        double desired_speed = -1.0);

    std::vector<FrenetTrajectory> getAllSuccessfulTrajectories() const {
        return last_fplist;
    }
    
    // Get statistics from the last planning cycle
    PlanStats get_stats() const {
        return last_stats;
    }
    
    // Generate Frenet frame from centerline points
    void generate_frenet_frame(const double* centerline_pts, int num_points, int pts_dim);

    // Distances in the same reference frame as trajectory s. Positive extents
    // measure from the reference line to the left/right same-direction road edge.
    void set_road_profile(const std::vector<double>& s,
                          const std::vector<double>& lane_width,
                          const std::vector<double>& left_extent,
                          const std::vector<double>& right_extent);

    // Obstacle extents in the current Frenet frame, (num_time_steps x num_obstacles x 4)
    // values (s_min, s_max, l_min, l_max), NaN where absent; cleared with a new frame.
    void set_obstacle_frenet_bounds(const double* bounds, int num_time_steps, int num_obstacles);

protected:
    // Baseline of the safe-distance requirement: for every obstacle, the first step at which
    // it is ahead in the path of the ego vehicle driving on at its current speed and lateral
    // offset, and the gap at that step. Independent of the candidate trajectories.
    std::vector<ClearanceBaseline> clearance_baseline(const FrenetState& start, int time_step_now) const;

    // Evaluates the samples with the shared evaluator on num_threads threads (contiguous
    // chunks; num_threads <= 0 uses all cores). Counters and timings are added to last_stats.
    BatchResult evaluate_batch(const FrenetState& start,
                               const std::vector<SamplingParam>& samples,
                               const PlanningContext& context,
                               int num_threads,
                               EvalMode mode = EvalMode::kEarlyExit);

    // v_des of J_V: desired_speed when given (>= 0), otherwise max_target_speed
    static double resolve_desired_speed(double max_target_speed, double desired_speed) {
        return desired_speed >= 0.0 ? desired_speed : max_target_speed;
    }

    // True if a cycle without any feasible trajectory may be re-planned without the
    // clearance requirement; switches it off in context and counts the fallback.
    bool fall_back_without_clearance(PlanningContext& context);
};

#endif // FRENET_PLANNER_H
