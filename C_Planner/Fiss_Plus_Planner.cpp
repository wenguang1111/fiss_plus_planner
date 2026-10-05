#include "Fiss_Plus_Planner.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <iostream>

Fiss_Plus_Planner::Fiss_Plus_Planner(const FissPlusPlannerSettings& settings_param,
                                      const VehicleParams& vehicle_param,
                                      const double* obs_array,
                                      const int* num_verts,
                                      int n_time_steps,
                                      int n_obstacles,
                                      int max_verts)
    : Frenet_Planner(settings_param, vehicle_param, obs_array, num_verts, n_time_steps, n_obstacles, max_verts),
      fiss_settings(settings_param),
      has_prev_best_idx(false),
      refined_trajs([](const FrenetTrajectory& a, const FrenetTrajectory& b) {
          return a.cost_final > b.cost_final;  // min-heap based on cost
      })
{
    sampling_res.fill(0.0);
    sampling_min.fill(0.0);
    sampling_max.fill(0.0);
    sizes.fill(0);
    prev_best_idx.fill(0);
}

void Fiss_Plus_Planner::clear_queues() {
    while (!candidate_trajs.empty()) candidate_trajs.pop();
    while (!frontier_idxs.empty()) frontier_idxs.pop();
    while (!refined_trajs.empty()) refined_trajs.pop();
}

double Fiss_Plus_Planner::clip(double value, double min_val, double max_val) {
    return std::max(min_val, std::min(value, max_val));
}

std::array<double, 3> Fiss_Plus_Planner::clip_array(const std::array<double, 3>& arr,
                                                     const std::array<double, 3>& min_arr,
                                                     const std::array<double, 3>& max_arr) {
    std::array<double, 3> result;
    for (int i = 0; i < 3; ++i) {
        result[i] = clip(arr[i], min_arr[i], max_arr[i]);
    }
    return result;
}

Trajs3D Fiss_Plus_Planner::sample_end_frenet_states(const SearchSpace& space) {
    trajs_3d.clear();
    
    // Heuristic parameters
    double max_sqr_dist = std::pow(fiss_settings.num_width, 2) + 
                          std::pow(fiss_settings.num_speed, 2) + 
                          std::pow(fiss_settings.num_t, 2);
    
    // Set sampling bounds (same search space as every other planner)
    sampling_min = {space.d_min, space.v_min, space.t_min};
    sampling_max = {space.d_max, space.v_max, space.t_max};
    const std::array<int, 3> nums = {fiss_settings.num_width, fiss_settings.num_speed, fiss_settings.num_t};
    
    // Calculate sampling resolutions; a single sample sits in the middle of the range
    for (int dim = 0; dim < 3; ++dim) {
        sampling_res[dim] = nums[dim] > 1 ? (sampling_max[dim] - sampling_min[dim]) / (nums[dim] - 1) : 0.0;
    }
    auto grid_value = [&](int dim, int i) {
        return nums[dim] > 1 ? sampling_min[dim] + i * sampling_res[dim]
                             : 0.5 * (sampling_min[dim] + sampling_max[dim]);
    };
    
    // Estimate lateral cost normalization
    double lat_norm = std::max({std::pow(space.d_min, 2), std::pow(space.d_max, 2), 1e-12});
    double speed_range = std::max(space.v_max - space.v_min, 1e-12);
    double time_range = std::max(space.t_max - space.t_min, 1e-12);
    
    // Sample lateral positions
    for (int i = 0; i < fiss_settings.num_width; ++i) {
        double d = grid_value(0, i);
        double cost_est_lat = std::pow(d, 2) / lat_norm;
        
        std::vector<std::vector<FrenetTrajectory>> trajs_2d;
        
        // Sample velocities
        for (int j = 0; j < fiss_settings.num_speed; ++j) {
            double v = grid_value(1, j);
            double cost_est_speed = std::pow(space.v_max - v, 2) / std::pow(speed_range, 2);
            
            std::vector<FrenetTrajectory> trajs_1d;
            
            // Sample time horizons
            for (int k = 0; k < fiss_settings.num_t; ++k) {
                double t = grid_value(2, k);
                
                // Planning horizon cost (encourage longer planning horizon)
                double cost_est_time = 1.0 - (t - space.t_min) / time_range;
                
                // Fixed cost terms
                double cost_est = cost_est_lat + cost_est_time + cost_est_speed;
                
                // Estimated heuristic cost terms
                double cost_heu = 0.0;
                if (has_prev_best_idx) {
                    double heu_sqr_dist = std::pow(i - prev_best_idx[0], 2) + 
                                          std::pow(j - prev_best_idx[1], 2) + 
                                          std::pow(k - prev_best_idx[2], 2);
                    cost_heu = fiss_settings.w_heuristic * heu_sqr_dist / max_sqr_dist;
                }
                
                // Create trajectory placeholder
                FrenetTrajectory traj;
                traj.idx[0] = i;
                traj.idx[1] = j;
                traj.idx[2] = k;
                // Set end state
                traj.end_state.t = t;
                traj.end_state.s_d = v;
                traj.end_state.d = d;
                traj.cost_heu = cost_heu;
                traj.cost_est = cost_est + cost_heu;
                traj.is_generated = false;
                
                trajs_1d.push_back(traj);
            }
            trajs_2d.push_back(trajs_1d);
        }
        trajs_3d.push_back(trajs_2d);
    }
    
    return trajs_3d;
}

double Fiss_Plus_Planner::generate_from_end_state(FrenetTrajectory& traj) {
    const TrajectoryEvaluator ev = evaluator();
    FrenetTrajectory generated = ev.generate(
        start_state, SamplingParam(traj.end_state.d, traj.end_state.s_d, traj.end_state.t), last_stats);
    std::copy(std::begin(traj.idx), std::end(traj.idx), std::begin(generated.idx));
    generated.cost_heu = traj.cost_heu;
    generated.cost_est = traj.cost_est;
    traj = std::move(generated);
    traj.cost_final = ev.frenet_cost(traj, last_stats).total();
    return traj.cost_final;
}

std::pair<bool, double> Fiss_Plus_Planner::generate_trajectory(const std::array<int, 3>& idx) {
    FrenetTrajectory& traj = trajs_3d[idx[0]][idx[1]][idx[2]];
    
    if (traj.is_generated) {
        return {false, traj.cost_final};
    }
    
    traj.idx[0] = idx[0];
    traj.idx[1] = idx[1];
    traj.idx[2] = idx[2];
    const double cost = generate_from_end_state(traj);
    
    // Add to candidate queue
    trajs_per_timestep.push_back(traj);
    candidate_trajs.push({cost, idx});
    
    return {true, cost};
}

double Fiss_Plus_Planner::generate_trajectory_by_end_state(const FrenetState& end_state) {
    FrenetTrajectory traj;
    traj.end_state.t = end_state.t;
    traj.end_state.s_d = end_state.s_d;
    traj.end_state.d = end_state.d;
    const double cost = generate_from_end_state(traj);
    
    // Add to refined queue
    refined_trajs.push(traj);
    
    if (fiss_settings.vis_all_candidates) {
        all_trajs.push_back({traj});
    }
    
    return cost;
}

std::array<int, 3> Fiss_Plus_Planner::find_initial_guess(bool& found) {
    std::array<int, 3> best_idx = {0, 0, 0};
    double min_cost = std::numeric_limits<double>::infinity();
    found = false;
    
    for (int i = 0; i < sizes[0]; ++i) {
        for (int j = 0; j < sizes[1]; ++j) {
            for (int k = 0; k < sizes[2]; ++k) {
                const FrenetTrajectory& traj = trajs_3d[i][j][k];
                if (!traj.is_generated && traj.cost_est <= min_cost) {
                    min_cost = traj.cost_est;
                    best_idx = {i, j, k};
                    found = true;
                }
            }
        }
    }
    
    return best_idx;
}

std::pair<bool, std::array<int, 3>> Fiss_Plus_Planner::explore_neighbors(const std::array<int, 3>& idx) {
    auto [_, cost_center] = generate_trajectory(idx);
    double min_cost = cost_center;
    std::array<int, 3> best_idx = idx;
    bool is_local_minimum = true;
    
    // Explore all six neighbors on three dimensions
    for (int dim = 0; dim < 3; ++dim) {
        // Left neighbor
        if (idx[dim] >= 1) {
            std::array<int, 3> prev_idx = idx;
            prev_idx[dim] -= 1;
            auto [is_new, cost] = generate_trajectory(prev_idx);
            if (is_new && cost <= cost_center) {
                frontier_idxs.push({cost, prev_idx});
            }
            if (cost <= min_cost) {
                min_cost = cost;
                best_idx = prev_idx;
                is_local_minimum = false;
            }
        }
        
        // Right neighbor
        if (idx[dim] < sizes[dim] - 1) {
            std::array<int, 3> next_idx = idx;
            next_idx[dim] += 1;
            auto [is_new, cost] = generate_trajectory(next_idx);
            if (is_new && cost <= cost_center) {
                frontier_idxs.push({cost, next_idx});
            }
            if (cost <= min_cost) {
                min_cost = cost;
                best_idx = next_idx;
                is_local_minimum = false;
            }
        }
    }
    
    return {is_local_minimum, best_idx};
}

std::tuple<bool, double, std::array<double, 3>, std::array<double, 3>>
Fiss_Plus_Planner::gradient_descent(double J, const std::array<double, 3>& x,
                                     std::array<double, 3> resolutions, double decaying_factor) {
    std::array<double, 3> d_J, d_x;
    
    for (int dim = 0; dim < 3; ++dim) {
        // Left neighbor
        std::array<double, 3> x_l = x;
        x_l[dim] -= resolutions[dim];
        x_l = clip_array(x_l, sampling_min, sampling_max);
        FrenetState end_state_l(x_l[2], 0.0, x_l[1], 0.0, 0.0, x_l[0], 0.0, 0.0, 0.0);
        double J_l = generate_trajectory_by_end_state(end_state_l);
        
        // Right neighbor
        std::array<double, 3> x_r = x;
        x_r[dim] += resolutions[dim];
        x_r = clip_array(x_r, sampling_min, sampling_max);
        FrenetState end_state_r(x_r[2], 0.0, x_r[1], 0.0, 0.0, x_r[0], 0.0, 0.0, 0.0);
        double J_r = generate_trajectory_by_end_state(end_state_r);
        
        // Gradient
        d_J[dim] = J_r - J_l;
        d_x[dim] = x_r[dim] - x_l[dim];
    }
    
    // Compute gradient and new location
    std::array<double, 3> grad;
    double grad_norm = 0.0;
    for (int dim = 0; dim < 3; ++dim) {
        if (std::abs(d_x[dim]) > 1e-10) {
            grad[dim] = d_J[dim] / d_x[dim];
        } else {
            grad[dim] = 0.0;
        }
        grad_norm += grad[dim] * grad[dim];
    }
    grad_norm = std::sqrt(grad_norm);
    
    // Update resolutions
    for (int dim = 0; dim < 3; ++dim) {
        resolutions[dim] *= decaying_factor;
    }
    
    // Compute new position
    std::array<double, 3> x_new;
    if (grad_norm > 1e-10) {
        for (int dim = 0; dim < 3; ++dim) {
            x_new[dim] = x[dim] - resolutions[dim] * grad[dim] / grad_norm;
        }
    } else {
        x_new = x;
    }
    
    // Clip to sampling region
    std::array<double, 3> x_new_clipped = clip_array(x_new, sampling_min, sampling_max);
    
    // Generate candidate at new location
    FrenetState end_state_new(x_new_clipped[2], 0.0, x_new_clipped[1], 0.0, 0.0, x_new_clipped[0], 0.0, 0.0, 0.0);
    double J_new = generate_trajectory_by_end_state(end_state_new);
    
    return {true, J_new, x_new_clipped, resolutions};
}

FrenetTrajectory* Fiss_Plus_Planner::refine_solution(const FrenetTrajectory& traj, double time_limit) {
    auto t_start = std::chrono::high_resolution_clock::now();
    std::array<double, 3> resolutions = sampling_res;
    double alpha = fiss_settings.decaying_factor;
    
    double J_new = traj.cost_final;
    std::array<double, 3> x = {traj.end_state.d, traj.end_state.s_d, traj.end_state.t};
    
    for (int i = 0; i < fiss_settings.max_refine_iters; ++i) {
        auto [valid, J_updated, x_updated, res_updated] = gradient_descent(J_new, x, resolutions, alpha);
        
        if (!valid) {
            break;
        }
        
        J_new = J_updated;
        x = x_updated;
        resolutions = res_updated;
        
        auto t_now = std::chrono::high_resolution_clock::now();
        double t_spent = std::chrono::duration<double>(t_now - t_start).count();
        if (fiss_settings.has_time_limit && t_spent >= time_limit) {
            break;
        }
    }
    
    // Check refined trajectories
    while (!refined_trajs.empty()) {
        FrenetTrajectory candidate = refined_trajs.top();
        refined_trajs.pop();
        
        if (candidate.cost_final > traj.cost_final) {
            break;
        }
        
        if (evaluator().check_feasibility(candidate, last_stats).feasible) {
            best_traj = candidate;
            return &best_traj;
        }
    }
    
    return nullptr;
}

FrenetTrajectory Fiss_Plus_Planner::plan(const FrenetState& frenet_state,
                                          double max_target_speed,
                                          int time_step_now,
                                          double desired_speed) {
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Reset values for each planning cycle
    last_stats = PlanStats();
    fiss_settings.highest_speed = max_target_speed;
    settings.highest_speed = max_target_speed;  // Also update base class settings
    start_state = frenet_state;
    clear_queues();
    best_traj = FrenetTrajectory();
    trajs_per_timestep.clear();
    context = make_context(desired_speed >= 0.0 ? desired_speed : max_target_speed, time_step_now, false);
    auto finish = [&]() {
        last_stats.timing.total_ms =
            std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();
        return best_traj;
    };
    
    // Sample all end states in 3D, inside the search space shared with FOP
    SearchSpace space;
    if (!search_space(frenet_state.s, space)) {
        return finish();
    }
    trajs_3d = sample_end_frenet_states(space);
    sizes = {static_cast<int>(trajs_3d.size()),
             static_cast<int>(trajs_3d[0].size()),
             static_cast<int>(trajs_3d[0][0].size())};
    
    std::array<int, 3> best_idx = {0, 0, 0};
    bool best_traj_found = false;
    
    while (!best_traj_found) {
        last_stats.num_search_iterations++;
        
        // ===================== Initial Guess =====================
        if (candidate_trajs.empty()) {
            bool found;
            best_idx = find_initial_guess(found);
            if (!found) {
                // All samples searched, no feasible candidate found
                break;
            }
        } else {
            best_idx = candidate_trajs.top().idx;
        }
        
        // ===================== Search Process =====================
        bool converged = false;
        while (!converged) {
            auto [is_minimum, new_best_idx] = explore_neighbors(best_idx);
            
            if (frontier_idxs.empty()) {
                converged = true;
            } else {
                best_idx = frontier_idxs.top().idx;
                frontier_idxs.pop();
            }
        }
        
        // ===================== Validation Process =====================
        if (!candidate_trajs.empty()) {
            TrajCandidate candidate_entry = candidate_trajs.top();
            candidate_trajs.pop();
            
            FrenetTrajectory candidate = trajs_3d[candidate_entry.idx[0]]
                                                 [candidate_entry.idx[1]]
                                                 [candidate_entry.idx[2]];
            
            // Cartesian transform, constraints and collision in the shared evaluator
            if (evaluator().check_feasibility(candidate, last_stats).feasible) {
                best_traj = candidate;
                best_traj_found = true;
                prev_best_idx = {best_traj.idx[0], best_traj.idx[1], best_traj.idx[2]};
                has_prev_best_idx = true;
                break;
            }
        } else {
            break;
        }
    }
    
    // ===================== Refinement =====================
    if (best_traj_found && fiss_settings.refine_trajectory) {
        auto t_now = std::chrono::high_resolution_clock::now();
        double time_spent = std::chrono::duration<double>(t_now - t_start).count();
        double time_left = fiss_settings.time_limit - time_spent;
        
        if (!fiss_settings.has_time_limit || time_left > 0.0) {
            FrenetTrajectory* refined_traj = refine_solution(best_traj, time_left);
            if (refined_traj != nullptr) {
                best_traj = *refined_traj;
            }
        }
    }
    
    // Store trajectories for visualization
    if (fiss_settings.vis_all_candidates) {
        std::vector<FrenetTrajectory> global_trajs;
        const TrajectoryEvaluator ev = evaluator();
        for (FrenetTrajectory traj : trajs_per_timestep) {
            if (ev.to_global(traj)) global_trajs.push_back(std::move(traj));
        }
        all_trajs.push_back(global_trajs);
    } else {
        all_trajs.push_back(trajs_per_timestep);
    }
    trajs_per_timestep.clear();
    
    return finish();
}
