#include "Fiss_Plus_Planner.h"
#include "common/collision/collision_checker.h"
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

Trajs3D Fiss_Plus_Planner::sample_end_frenet_states() {
    trajs_3d.clear();
    
    // Heuristic parameters
    double max_sqr_dist = std::pow(fiss_settings.num_width, 2) + 
                          std::pow(fiss_settings.num_speed, 2) + 
                          std::pow(fiss_settings.num_t, 2);
    
    // Define the lateral sampling positions
    double sampling_width = fiss_settings.max_road_width - vehicle_params.w + 0.3;
    double left_bound = -sampling_width / 2.0;
    double right_bound = sampling_width / 2.0;
    
    // Set sampling bounds
    sampling_min[0] = left_bound;
    sampling_max[0] = right_bound;
    sampling_min[1] = fiss_settings.lowest_speed;
    sampling_max[1] = fiss_settings.highest_speed;
    sampling_min[2] = fiss_settings.min_t;
    sampling_max[2] = fiss_settings.max_t;
    
    // Calculate sampling resolutions
    sampling_res[0] = (right_bound - left_bound) / (fiss_settings.num_width - 1);
    sampling_res[1] = (fiss_settings.highest_speed - fiss_settings.lowest_speed) / (fiss_settings.num_speed - 1);
    sampling_res[2] = (fiss_settings.max_t - fiss_settings.min_t) / (fiss_settings.num_t - 1);
    
    // Estimate lateral cost normalization
    double lat_norm = std::max(std::pow(left_bound, 2), std::pow(right_bound, 2));
    
    // Sample lateral positions
    for (int i = 0; i < fiss_settings.num_width; ++i) {
        double d = left_bound + i * sampling_res[0];
        double cost_est_lat = std::pow(d, 2) / lat_norm;
        
        std::vector<std::vector<FrenetTrajectory>> trajs_2d;
        
        // Sample velocities
        for (int j = 0; j < fiss_settings.num_speed; ++j) {
            double v = fiss_settings.lowest_speed + j * sampling_res[1];
            double cost_est_speed = std::pow(fiss_settings.highest_speed - v, 2) / 
                                    std::pow(fiss_settings.highest_speed - fiss_settings.lowest_speed, 2);
            
            std::vector<FrenetTrajectory> trajs_1d;
            
            // Sample time horizons
            for (int k = 0; k < fiss_settings.num_t; ++k) {
                double t = fiss_settings.min_t + k * sampling_res[2];
                
                // Planning horizon cost (encourage longer planning horizon)
                double cost_est_time = 1.0 - (t - fiss_settings.min_t) / (fiss_settings.max_t - fiss_settings.min_t);
                
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
                traj.end_state.s = 0.0;
                traj.end_state.s_d = v;
                traj.end_state.s_dd = 0.0;
                traj.end_state.d = d;
                traj.end_state.d_d = 0.0;
                traj.end_state.d_dd = 0.0;
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

std::pair<bool, double> Fiss_Plus_Planner::generate_trajectory(const std::array<int, 3>& idx) {
    FrenetTrajectory& traj = trajs_3d[idx[0]][idx[1]][idx[2]];
    
    if (traj.is_generated) {
        return {false, traj.cost_final};
    }
    
    last_stats.num_trajs_generated++;
    traj.is_generated = true;
    traj.idx[0] = idx[0];
    traj.idx[1] = idx[1];
    traj.idx[2] = idx[2];
    
    // Generate time steps
    const auto& end_state = traj.end_state;
    traj.t.clear();
    for (double t = 0.0; t < end_state.t; t += fiss_settings.tick_t) {
        traj.t.push_back(t);
    }
    
    // Generate lateral quintic polynomial
    QuinticPolynomial lat_qp(start_state.d, start_state.d_d, start_state.d_dd,
                             end_state.d, end_state.d_d, end_state.d_dd, end_state.t);
    traj.d.clear();
    traj.d_d.clear();
    traj.d_dd.clear();
    traj.d_ddd.clear();
    for (double t : traj.t) {
        traj.d.push_back(lat_qp.calc_point(t));
        traj.d_d.push_back(lat_qp.calc_first_derivative(t));
        traj.d_dd.push_back(lat_qp.calc_second_derivative(t));
        traj.d_ddd.push_back(lat_qp.calc_third_derivative(t));
    }
    
    // Generate longitudinal quartic polynomial
    QuarticPolynomial lon_qp(start_state.s, start_state.s_d, start_state.s_dd,
                             end_state.s_d, end_state.s_dd, end_state.t);
    traj.s.clear();
    traj.s_d.clear();
    traj.s_dd.clear();
    traj.s_ddd.clear();
    for (double t : traj.t) {
        traj.s.push_back(lon_qp.calc_point(t));
        traj.s_d.push_back(lon_qp.calc_first_derivative(t));
        traj.s_dd.push_back(lon_qp.calc_second_derivative(t));
        traj.s_ddd.push_back(lon_qp.calc_third_derivative(t));
    }
    
    // Compute the final cost
    traj.cost_final = cost_function.cost_total(traj, fiss_settings.highest_speed);
    
    // Add to candidate queue
    trajs_per_timestep.push_back(traj);
    candidate_trajs.push({traj.cost_final, idx});
    
    return {true, traj.cost_final};
}

double Fiss_Plus_Planner::generate_trajectory_by_end_state(const FrenetState& end_state) {
    FrenetTrajectory traj;
    // Copy end state values
    traj.end_state.t = end_state.t;
    traj.end_state.s = end_state.s;
    traj.end_state.s_d = end_state.s_d;
    traj.end_state.s_dd = end_state.s_dd;
    traj.end_state.d = end_state.d;
    traj.end_state.d_d = end_state.d_d;
    traj.end_state.d_dd = end_state.d_dd;
    
    last_stats.num_trajs_generated++;
    traj.is_generated = true;
    
    // Generate time steps
    traj.t.clear();
    for (double t = 0.0; t < end_state.t; t += fiss_settings.tick_t) {
        traj.t.push_back(t);
    }
    
    // Generate lateral quintic polynomial
    QuinticPolynomial lat_qp(start_state.d, start_state.d_d, start_state.d_dd,
                             end_state.d, end_state.d_d, end_state.d_dd, end_state.t);
    for (double t : traj.t) {
        traj.d.push_back(lat_qp.calc_point(t));
        traj.d_d.push_back(lat_qp.calc_first_derivative(t));
        traj.d_dd.push_back(lat_qp.calc_second_derivative(t));
        traj.d_ddd.push_back(lat_qp.calc_third_derivative(t));
    }
    
    // Generate longitudinal quartic polynomial
    QuarticPolynomial lon_qp(start_state.s, start_state.s_d, start_state.s_dd,
                             end_state.s_d, end_state.s_dd, end_state.t);
    for (double t : traj.t) {
        traj.s.push_back(lon_qp.calc_point(t));
        traj.s_d.push_back(lon_qp.calc_first_derivative(t));
        traj.s_dd.push_back(lon_qp.calc_second_derivative(t));
        traj.s_ddd.push_back(lon_qp.calc_third_derivative(t));
    }
    
    // Compute the final cost
    traj.cost_final = cost_function.cost_total(traj, fiss_settings.highest_speed);
    
    // Add to refined queue
    refined_trajs.push(traj);
    
    if (fiss_settings.vis_all_candidates) {
        all_trajs.push_back({traj});
    }
    
    return traj.cost_final;
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

FrenetTrajectory* Fiss_Plus_Planner::refine_solution(const FrenetTrajectory& traj, double time_limit, int time_step_now) {
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
        
        last_stats.num_trajs_validated++;
        
        // Convert to global frame
        std::vector<FrenetTrajectory> candidates = calc_global_paths({candidate});
        if (candidates.empty()) continue;
        
        // Check constraints
        std::vector<FrenetTrajectory> passed_candidates = check_constraints(candidates);
        if (passed_candidates.empty()) continue;
        
        // Check collisions
        last_stats.num_collision_checks++;
        std::vector<FrenetTrajectory> safe_candidates = check_collision(
            passed_candidates,
            obstacles_array,
            num_vertices_array,
            num_time_steps,
            num_obstacles,
            max_vertices,
            vehicle_params.l,
            vehicle_params.w,
            time_step_now,
            1  // check_resolution
        );
        
        if (!safe_candidates.empty()) {
            // Return the first safe candidate - store in member variable
            best_traj = safe_candidates[0];
            return &best_traj;
        }
    }
    
    return nullptr;
}

FrenetTrajectory Fiss_Plus_Planner::plan(const FrenetState& frenet_state,
                                          double max_target_speed,
                                          int time_step_now) {
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Reset values for each planning cycle
    last_stats = PlanStats();
    fiss_settings.highest_speed = max_target_speed;
    settings.highest_speed = max_target_speed;  // Also update base class settings
    start_state = frenet_state;
    clear_queues();
    best_traj = FrenetTrajectory();
    trajs_per_timestep.clear();
    
    // Sample all end states in 3D
    trajs_3d = sample_end_frenet_states();
    sizes = {static_cast<int>(trajs_3d.size()),
             static_cast<int>(trajs_3d[0].size()),
             static_cast<int>(trajs_3d[0][0].size())};
    
    std::array<int, 3> best_idx = {0, 0, 0};
    bool best_traj_found = false;
    
    while (!best_traj_found) {
        last_stats.num_trajs_validated++;  // Using as iteration counter
        
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
            
            FrenetTrajectory& candidate = trajs_3d[candidate_entry.idx[0]]
                                                  [candidate_entry.idx[1]]
                                                  [candidate_entry.idx[2]];
            
            last_stats.num_trajs_validated++;
            
            // Convert to global frame
            std::vector<FrenetTrajectory> candidates = calc_global_paths({candidate});
            if (candidates.empty()) continue;
            
            // Check constraints
            std::vector<FrenetTrajectory> passed_candidates = check_constraints(candidates);
            if (passed_candidates.empty()) continue;
            
            // Check collisions
            last_stats.num_collision_checks++;
            std::vector<FrenetTrajectory> safe_candidates = check_collision(
                passed_candidates,
                obstacles_array,
                num_vertices_array,
                num_time_steps,
                num_obstacles,
                max_vertices,
                vehicle_params.l,
                vehicle_params.w,
                time_step_now,
                1  // check_resolution
            );
            
            if (!safe_candidates.empty()) {
                best_traj_found = true;
                best_traj = safe_candidates[0];
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
            FrenetTrajectory* refined_traj = refine_solution(best_traj, time_left, time_step_now);
            if (refined_traj != nullptr) {
                best_traj = *refined_traj;
            }
        }
    }
    
    // Store trajectories for visualization
    if (fiss_settings.vis_all_candidates) {
        std::vector<FrenetTrajectory> global_trajs = calc_global_paths(trajs_per_timestep);
        all_trajs.push_back(global_trajs);
    } else {
        all_trajs.push_back(trajs_per_timestep);
    }
    trajs_per_timestep.clear();
    
    return best_traj;
}
