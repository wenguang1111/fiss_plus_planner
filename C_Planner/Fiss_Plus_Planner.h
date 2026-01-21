#ifndef FISS_PLUS_PLANNER_H
#define FISS_PLUS_PLANNER_H

#include "Frenet_Planner.h"
#include <queue>
#include <array>
#include <functional>

// Settings for FISS+ planner, extends base SettingParameters
struct FissPlusPlannerSettings : public SettingParameters {
    bool refine_trajectory;
    int max_refine_iters;
    bool has_time_limit;
    double time_limit;
    double decaying_factor;
    double w_heuristic;
    bool vis_all_candidates;

    FissPlusPlannerSettings(int num_width_param = 5, int num_speed_param = 5, int num_t_param = 5, int refine_iters = 3)
        : SettingParameters(num_width_param, num_speed_param, num_t_param),
          refine_trajectory(true),
          max_refine_iters(refine_iters),
          has_time_limit(false),
          time_limit(0.5),
          decaying_factor(0.5),
          w_heuristic(10.0),
          vis_all_candidates(false) {}
};

// Priority queue entry for trajectory candidates
struct TrajCandidate {
    double cost;
    std::array<int, 3> idx;
    
    bool operator>(const TrajCandidate& other) const {
        return cost > other.cost;
    }
};

// 3D trajectory array type
using Trajs3D = std::vector<std::vector<std::vector<FrenetTrajectory>>>;

class Fiss_Plus_Planner : public Frenet_Planner {
public:
    FissPlusPlannerSettings fiss_settings;
    
    // Sampling parameters
    std::array<double, 3> sampling_res;
    std::array<double, 3> sampling_min;
    std::array<double, 3> sampling_max;
    
    // 3D trajectory candidate array
    Trajs3D trajs_3d;
    std::array<int, 3> sizes;
    
    // Start state for current planning cycle
    FrenetState start_state;
    
    // Previous best trajectory index (for heuristic)
    std::array<int, 3> prev_best_idx;
    bool has_prev_best_idx;
    
    // Priority queues
    std::priority_queue<TrajCandidate, std::vector<TrajCandidate>, std::greater<TrajCandidate>> candidate_trajs;
    std::priority_queue<TrajCandidate, std::vector<TrajCandidate>, std::greater<TrajCandidate>> frontier_idxs;
    std::priority_queue<FrenetTrajectory, std::vector<FrenetTrajectory>, 
                        std::function<bool(const FrenetTrajectory&, const FrenetTrajectory&)>> refined_trajs;
    
    // Trajectories generated per timestep (for visualization)
    std::vector<FrenetTrajectory> trajs_per_timestep;
    
    // Constructor
    Fiss_Plus_Planner(const FissPlusPlannerSettings& settings_param,
                      const VehicleParams& vehicle_param,
                      const double* obs_array,
                      const int* num_verts,
                      int n_time_steps,
                      int n_obstacles,
                      int max_verts);
    
    // Sample end Frenet states and create 3D trajectory array
    Trajs3D sample_end_frenet_states();
    
    // Generate trajectory at given index
    // Returns (is_new, cost)
    std::pair<bool, double> generate_trajectory(const std::array<int, 3>& idx);
    
    // Generate trajectory by end state (for refinement)
    double generate_trajectory_by_end_state(const FrenetState& end_state);
    
    // Find initial guess based on estimated cost
    std::array<int, 3> find_initial_guess(bool& found);
    
    // Explore neighbors of given index
    // Returns (is_local_minimum, best_idx)
    std::pair<bool, std::array<int, 3>> explore_neighbors(const std::array<int, 3>& idx);
    
    // Gradient descent for refinement
    // Returns (valid, J_new, x_new, resolutions_new)
    std::tuple<bool, double, std::array<double, 3>, std::array<double, 3>> 
    gradient_descent(double J, const std::array<double, 3>& x, 
                     std::array<double, 3> resolutions, double decaying_factor);
    
    // Refine solution using gradient descent
    FrenetTrajectory* refine_solution(const FrenetTrajectory& traj, double time_limit, int time_step_now);
    
    // Main planning function - overrides base class
    FrenetTrajectory plan(const FrenetState& frenet_state,
                          double max_target_speed,
                          int time_step_now = 0);
    
private:
    // Helper to clear priority queues
    void clear_queues();
    
    // Helper to clip value to range
    double clip(double value, double min_val, double max_val);
    
    // Helper to clip array to range
    std::array<double, 3> clip_array(const std::array<double, 3>& arr, 
                                      const std::array<double, 3>& min_arr,
                                      const std::array<double, 3>& max_arr);
};

#endif // FISS_PLUS_PLANNER_H
