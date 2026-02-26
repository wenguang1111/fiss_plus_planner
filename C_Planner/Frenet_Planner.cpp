#include "Frenet_Planner.h"
#include "common/collision/collision_checker.h"
#include "Recorder4Cpp/recorder.h"
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>
#include <limits>
#include <iostream>

Frenet_Planner::Frenet_Planner(const SettingParameters& settings_param, 
                               const VehicleParams& vehicle_param,
                               const double* obs_array,
                               const int* num_verts,
                               int n_time_steps,
                               int n_obstacles,
                               int max_verts)
    : settings(settings_param), 
      vehicle_params(vehicle_param), 
      cost_function("WX1"), 
      cubic_spline(nullptr),
      obstacles_array(obs_array),
      num_vertices_array(num_verts),
      num_time_steps(n_time_steps),
      num_obstacles(n_obstacles),
      max_vertices(max_verts) 
{
    // recordObstacleArray();
}

Frenet_Planner::~Frenet_Planner() {
    if (cubic_spline != nullptr) {
        delete cubic_spline;
    }
}

void Frenet_Planner::recordObstacleArray()
{
    #ifdef USE_RECORDER
        for (int t = 0; t < num_time_steps; ++t) {
            for (int obs = 0; obs < num_obstacles; ++obs) {
                int num_verts = num_vertices_array[t * num_obstacles + obs];
                for (int v = 0; v < max_vertices; ++v) {
                    int idx = t * num_obstacles * max_vertices * 2
                            + obs * max_vertices * 2
                            + v * 2;
                    Recorder::getInstance()->saveData<double>("obstacles.t", t);
                    Recorder::getInstance()->saveData<double>("obstacles.obs", obs);
                    Recorder::getInstance()->saveData<double>("obstacles.v", v);
                    Recorder::getInstance()->saveData<double>("obstacles.num_vertices", num_verts);
                    Recorder::getInstance()->saveData<double>("obstacles.x", obstacles_array[idx]);
                    Recorder::getInstance()->saveData<double>("obstacles.y", obstacles_array[idx + 1]);
                }
            }
        }
    #endif
}

std::vector<std::tuple<double, double, double>> Frenet_Planner::get_samples() {
    // TODO: Generate sampling parameters (d, s_d, t)
    // Calculate sampling range for lateral position
    double sampling_width = settings.max_road_width - vehicle_params.w;
    
    std::vector<double> d_samples;
    for (int i = 0; i < settings.num_width; i++) {
        double d = -sampling_width / 2.0 + i * sampling_width / (settings.num_width-1);
        d_samples.push_back(d);
    }
    
    std::vector<double> s_d_samples;
    for (int i = 0; i < settings.num_speed; i++) {
        double s_d = settings.lowest_speed + i * (settings.highest_speed - settings.lowest_speed) / (settings.num_speed-1);
        s_d_samples.push_back(s_d);
    }
    
    std::vector<double> t_samples;
    for (int i = 0; i < settings.num_t; i++) {
        double t = settings.min_t + i * (settings.max_t - settings.min_t) / (settings.num_t-1);
        t_samples.push_back(t);
    }
    
    // Generate all combinations
    std::vector<std::tuple<double, double, double>> samples;
    for (double d : d_samples) {
        for (double s_d : s_d_samples) {
            for (double t : t_samples) {
                samples.push_back(std::make_tuple(d, s_d, t));
            }
        }
    }
    
    return samples;
}

std::vector<FrenetTrajectory> Frenet_Planner::calc_frenet_paths(const FrenetState& frenet_state,
                                                                  const std::vector<std::tuple<double, double, double>>& samples) {
    // TODO: Calculate Frenet frame trajectories using quintic and quartic polynomials
    std::vector<FrenetTrajectory> frenet_paths;
    
    for (const auto& sample : samples) {
        double di = std::get<0>(sample);      // target lateral position
        double tv = std::get<1>(sample);      // target velocity
        double Ti = std::get<2>(sample);      // time horizon
        
        // Ensure Ti is at least tick_t
        Ti = std::max(Ti, settings.tick_t);
        
        FrenetTrajectory fp;
        
        // Lateral trajectory using quintic polynomial
        QuinticPolynomial lat_qp(frenet_state.d, frenet_state.d_d, frenet_state.d_dd,
                                 di, 0.0f, 0.0f, Ti);
        
        // Generate time steps
        for (double t = 0.0; t < Ti; t += settings.tick_t) {
            fp.t.push_back(t);
            fp.d.push_back(lat_qp.calc_point(t));
            fp.d_d.push_back(lat_qp.calc_first_derivative(t));
            fp.d_dd.push_back(lat_qp.calc_second_derivative(t));
            fp.d_ddd.push_back(lat_qp.calc_third_derivative(t));
        }
        
        // Longitudinal trajectory using quartic polynomial
        QuarticPolynomial lon_qp(frenet_state.s, frenet_state.s_d, frenet_state.s_dd,
                                 tv, 0.0f, Ti);
        
        for (size_t i = 0; i < fp.t.size(); i++) {
            double t = fp.t[i];
            fp.s.push_back(lon_qp.calc_point(t));
            fp.s_d.push_back(lon_qp.calc_first_derivative(t));
            fp.s_dd.push_back(lon_qp.calc_second_derivative(t));
            fp.s_ddd.push_back(lon_qp.calc_third_derivative(t));
        }
        
        // Compute final cost
        // fp.cost_final = cost_function.cost_total(fp, settings.highest_speed);
        fp.is_generated = true;
        fp.sampling_param = SamplingParam(di, tv, Ti);
        frenet_paths.push_back(fp);
    }
    
    return frenet_paths;
}

std::vector<FrenetTrajectory> Frenet_Planner::calc_global_paths(const std::vector<FrenetTrajectory>& fplist) {
    // TODO: Convert Frenet paths to global (x, y) coordinates using cubic spline
    std::vector<FrenetTrajectory> passed_fplist;
    
    if (cubic_spline == nullptr) {
        std::cerr << "Cubic spline is not initialized!" << std::endl;
        return passed_fplist;
    }
    
    for (auto fp : fplist) {
        size_t n = std::min(fp.s.size(), fp.d.size());
        if (n == 0) {
            continue;
        }

        // Calculate global positions
        for (size_t i = 0; i < n; i++) {
            auto [ix, iy] = cubic_spline->calc_position(fp.s[i]);

            // Stop adding points if position is invalid
            if (std::isnan(ix) || std::isnan(iy)) {
                break;
            }

            double i_yaw = cubic_spline->calc_yaw(fp.s[i]);
            double di = fp.d[i];

            // Convert from Frenet to Cartesian coordinates
            double fx = ix + di * std::cos(i_yaw + M_PI / 2.0);
            double fy = iy + di * std::sin(i_yaw + M_PI / 2.0);

            fp.x.push_back(fx);
            fp.y.push_back(fy);
        }

        if (fp.x.size() < 2) {
            continue;
        }

        // Calculate yaw and ds
        for (size_t i = 0; i + 1 < fp.x.size(); i++) {
            double dx = fp.x[i + 1] - fp.x[i];
            double dy = fp.y[i + 1] - fp.y[i];
            fp.yaw.push_back(std::atan2(dy, dx));
            fp.ds.push_back(std::sqrt(dx * dx + dy * dy));
        }

        fp.yaw.push_back(fp.yaw.back());

        // Calculate curvature
        double dt = settings.tick_t;
        for (size_t i = 0; i + 1 < fp.yaw.size(); i++) {
            fp.c.push_back((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i]);
        }

        for (size_t i = 0; i + 1 < fp.c.size(); i++) {
            fp.c_d.push_back((fp.c[i + 1] - fp.c[i]) / dt);
        }

        for (size_t i = 0; i + 1 < fp.c_d.size(); i++) {
            fp.c_dd.push_back((fp.c_d[i + 1] - fp.c_d[i]) / dt);
        }

        passed_fplist.push_back(fp);
    }
    
    return passed_fplist;
}

std::vector<FrenetTrajectory> Frenet_Planner::check_constraints(const std::vector<FrenetTrajectory>& trajs) {
    // Check trajectory constraints (speed, acceleration, etc.)
    std::vector<FrenetTrajectory> passed;
    
    for (const auto& traj : trajs) {
        bool valid = true;
        
        // Check max speed
        for (double v : traj.s_d) {
            if (v > vehicle_params.max_speed) {
                valid = false;
                break;
            }
        }
        
        if (!valid) continue;
        
        // Check max acceleration
        for (double a : traj.s_dd) {
            if (std::abs(a) > vehicle_params.max_accel) {
                valid = false;
                break;
            }
        }
        
        if (valid) {
            passed.push_back(traj);
            passed.back().constraint_passed = true;
        }
    }
    
    return passed;
}

//FIXME: may need delete this if not needed in python
std::vector<FrenetTrajectory> Frenet_Planner::check_collision_multithread(const std::vector<FrenetTrajectory>& trajs,
    int time_step_now) {
// Multi-threaded collision detection using pre-processed obstacle data
// If no obstacles data available, return all trajectories as valid
if (trajs.empty() || obstacles_array == nullptr || num_vertices_array == nullptr ||
num_time_steps <= 0 || num_obstacles <= 0) {
return trajs;
}

std::vector<FrenetTrajectory> passed;
std::vector<bool> collision_flags(trajs.size(), false);
std::mutex collision_mutex;

// Process each trajectory in parallel
std::vector<std::thread> threads;
int num_threads = std::thread::hardware_concurrency();
if (num_threads == 0) num_threads = 4;  // fallback default

int trajs_per_thread = (trajs.size() + num_threads - 1) / num_threads;

for (int t = 0; t < num_threads && t * trajs_per_thread < (int)trajs.size(); t++) {
threads.emplace_back([this, &trajs, &collision_flags, &collision_mutex, time_step_now, t, trajs_per_thread]() {
int start_idx = t * trajs_per_thread;
int end_idx = std::min(start_idx + trajs_per_thread, (int)trajs.size());

for (int i = start_idx; i < end_idx; i++) {
const auto& traj = trajs[i];
bool has_collision = false;

// Check collision for this trajectory
int t_step_max = std::min((int)traj.x.size(), num_time_steps - time_step_now);

for (int t_check = 0; t_check < t_step_max && !has_collision; t_check++) {
int t_step = t_check + time_step_now;
if (t_step >= num_time_steps) break;

// Check ego vehicle position against all obstacles at this time step
double ego_x = traj.x[t_check];
double ego_y = traj.y[t_check];

// Simple AABB collision check as a placeholder
// In a real implementation, this would use polygon intersection
for (int obs_idx = 0; obs_idx < num_obstacles; obs_idx++) {
int num_verts = num_vertices_array[t_step * num_obstacles + obs_idx];
if (num_verts <= 0) continue;

// Get obstacle polygon vertices
double min_x = 1e6, max_x = -1e6;
double min_y = 1e6, max_y = -1e6;

for (int v = 0; v < num_verts; v++) {
int idx = t_step * num_obstacles * max_vertices * 2 + obs_idx * max_vertices * 2 + v * 2;
double vx = obstacles_array[idx];
double vy = obstacles_array[idx + 1];
min_x = std::min(min_x, vx);
max_x = std::max(max_x, vx);
min_y = std::min(min_y, vy);
max_y = std::max(max_y, vy);
}

// AABB collision check with vehicle bounding box
double ego_min_x = ego_x - vehicle_params.w / 2.0;
double ego_max_x = ego_x + vehicle_params.w / 2.0;
double ego_min_y = ego_y - vehicle_params.l / 2.0;
double ego_max_y = ego_y + vehicle_params.l / 2.0;

if (!(ego_max_x < min_x || ego_min_x > max_x ||
ego_max_y < min_y || ego_min_y > max_y)) {
has_collision = true;
break;
}
}
}

{
std::lock_guard<std::mutex> lock(collision_mutex);
collision_flags[i] = has_collision;
}
}
});
}

// Wait for all threads to complete
for (auto& thread : threads) {
thread.join();
}

// Collect non-colliding trajectories
for (size_t i = 0; i < trajs.size(); i++) {
if (!collision_flags[i]) {
passed.push_back(trajs[i]);
}
}

return passed;
}

void Frenet_Planner::recordTrajectory(const FrenetTrajectory& traj)
{
    #ifdef USE_RECORDER
    Recorder::getInstance()->saveData<double>("traj.cost_fix", traj.cost_fix);
    Recorder::getInstance()->saveData<double>("traj.cost_dyn", traj.cost_dyn);
    Recorder::getInstance()->saveData<double>("traj.cost_heu", traj.cost_heu);
    Recorder::getInstance()->saveData<double>("traj.cost_est", traj.cost_est);
    Recorder::getInstance()->saveData<double>("traj.cost_final", traj.cost_final);

    Recorder::getInstance()->saveData<int>("traj.idx0", traj.idx[0]);
    Recorder::getInstance()->saveData<int>("traj.idx1", traj.idx[1]);
    Recorder::getInstance()->saveData<int>("traj.idx2", traj.idx[2]);
    Recorder::getInstance()->saveData<int>("traj.lane_id", traj.lane_id);
    Recorder::getInstance()->saveData<int>("traj.is_generated", traj.is_generated ? 1 : 0);
    Recorder::getInstance()->saveData<int>("traj.is_searched", traj.is_searched ? 1 : 0);
    Recorder::getInstance()->saveData<int>("traj.constraint_passed", traj.constraint_passed ? 1 : 0);
    Recorder::getInstance()->saveData<int>("traj.collision_passed", traj.collision_passed ? 1 : 0);

    Recorder::getInstance()->saveData<int>("traj.t.size", static_cast<int>(traj.t.size()));
    for (size_t i = 0; i < traj.t.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.t.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.t", traj.t[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.s.size", static_cast<int>(traj.s.size()));
    for (size_t i = 0; i < traj.s.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.s.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.s", traj.s[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.s_d.size", static_cast<int>(traj.s_d.size()));
    for (size_t i = 0; i < traj.s_d.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.s_d.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.s_d", traj.s_d[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.s_dd.size", static_cast<int>(traj.s_dd.size()));
    for (size_t i = 0; i < traj.s_dd.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.s_dd.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.s_dd", traj.s_dd[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.s_ddd.size", static_cast<int>(traj.s_ddd.size()));
    for (size_t i = 0; i < traj.s_ddd.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.s_ddd.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.s_ddd", traj.s_ddd[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.d.size", static_cast<int>(traj.d.size()));
    for (size_t i = 0; i < traj.d.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.d.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.d", traj.d[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.d_d.size", static_cast<int>(traj.d_d.size()));
    for (size_t i = 0; i < traj.d_d.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.d_d.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.d_d", traj.d_d[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.d_dd.size", static_cast<int>(traj.d_dd.size()));
    for (size_t i = 0; i < traj.d_dd.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.d_dd.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.d_dd", traj.d_dd[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.d_ddd.size", static_cast<int>(traj.d_ddd.size()));
    for (size_t i = 0; i < traj.d_ddd.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.d_ddd.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.d_ddd", traj.d_ddd[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.x.size", static_cast<int>(traj.x.size()));
    for (size_t i = 0; i < traj.x.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.x.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.x", traj.x[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.y.size", static_cast<int>(traj.y.size()));
    for (size_t i = 0; i < traj.y.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.y.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.y", traj.y[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.yaw.size", static_cast<int>(traj.yaw.size()));
    for (size_t i = 0; i < traj.yaw.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.yaw.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.yaw", traj.yaw[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.ds.size", static_cast<int>(traj.ds.size()));
    for (size_t i = 0; i < traj.ds.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.ds.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.ds", traj.ds[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.c.size", static_cast<int>(traj.c.size()));
    for (size_t i = 0; i < traj.c.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.c.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.c", traj.c[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.c_d.size", static_cast<int>(traj.c_d.size()));
    for (size_t i = 0; i < traj.c_d.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.c_d.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.c_d", traj.c_d[i]);
    }

    Recorder::getInstance()->saveData<int>("traj.c_dd.size", static_cast<int>(traj.c_dd.size()));
    for (size_t i = 0; i < traj.c_dd.size(); ++i) {
        Recorder::getInstance()->saveData<int>("traj.c_dd.step", static_cast<int>(i));
        Recorder::getInstance()->saveData<double>("traj.c_dd", traj.c_dd[i]);
    }
    #endif
}

FrenetTrajectory Frenet_Planner::plan(const FrenetState& frenet_state,
                                      double max_target_speed,
                                      int time_step_now,
                                      int num_threads) {
    std::vector<std::tuple<double, double, double>> samples = get_samples();
    return best_traj_generation(frenet_state, samples, max_target_speed, time_step_now, num_threads);
}

FrenetTrajectory Frenet_Planner::best_traj_generation(
    const FrenetState& frenet_state,
    const std::vector<std::tuple<double, double, double>>& samples,
    double max_target_speed,
    int time_step_now,
    int num_threads) {
    settings.highest_speed = max_target_speed;

    // Ensure num_threads is at least 1
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = 1;
    }

    // Handle edge case: no samples
    if (samples.empty()) {
        std::cerr << "empty samples" << std::endl;
        return FrenetTrajectory();
    }

    int samples_per_thread = (samples.size() + num_threads - 1) / num_threads;

    std::vector<std::vector<std::tuple<double, double, double>>> samples_per_thread_vec;
    for (int t = 0; t < num_threads && t * samples_per_thread < (int)samples.size(); t++) {
        int start_idx = t * samples_per_thread;
        int end_idx = std::min(start_idx + samples_per_thread, (int)samples.size());

        std::vector<std::tuple<double, double, double>> thread_samples(
            samples.begin() + start_idx,
            samples.begin() + end_idx
        );
        samples_per_thread_vec.push_back(thread_samples);
    }

    // Get collision-free paths from multithreaded planning
    PlanResult plan_result = plan_multithread(samples_per_thread_vec, frenet_state, time_step_now);
    last_fplist = plan_result.collision_free_paths;

    // Find minimum cost path
    best_traj = FrenetTrajectory();
    best_traj.cost_final = std::numeric_limits<double>::infinity();
    for (const auto& fp : plan_result.collision_free_paths) {
        if (fp.cost_final < best_traj.cost_final) {
            best_traj = fp;
        }
    }

    #ifdef USE_RECORDER
        Recorder::getInstance()->writeDataToCSV();
    #endif

    return best_traj;
}

PlanResult Frenet_Planner::plan_multithread(
                                                   const std::vector<std::vector<std::tuple<double, double, double>>>& samples_per_thread_vec,
                                                   const FrenetState& frenet_state,
                                                   int time_step_now) {
    PlanResult result;
    
    if (samples_per_thread_vec.empty()) {
        return result;
    }

    int num_threads = samples_per_thread_vec.size();

    std::vector<std::thread> threads;
    std::vector<std::vector<FrenetTrajectory>> thread_frenet_paths(num_threads);
    std::vector<std::vector<FrenetTrajectory>> thread_collision_free_paths(num_threads);
    std::vector<PlanStats> thread_stats(num_threads);  // Statistics for each thread

    // Process samples in parallel
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([this, &samples_per_thread_vec, &thread_frenet_paths, &thread_collision_free_paths, &thread_stats, &frenet_state, t, time_step_now]() {
            const auto& thread_samples = samples_per_thread_vec[t];
            PlanStats local_stats;
            
            // Step 1: Generate Frenet paths
            std::vector<FrenetTrajectory> frenet_paths = calc_frenet_paths(frenet_state, thread_samples);

            // Store frenet_paths for this thread
            thread_frenet_paths[t] = frenet_paths;
            local_stats.num_trajs_generated = frenet_paths.size();

            // Step 2: Convert to global coordinates
            std::vector<FrenetTrajectory> global_paths = calc_global_paths(frenet_paths);

            // Step 3: Check constraints
            std::vector<FrenetTrajectory> constrained_paths = check_constraints(global_paths);
            local_stats.num_trajs_validated = constrained_paths.size();

            // Step 4: Check collisions (within each thread for better cache locality)
            local_stats.num_collision_checks = constrained_paths.size();
            std::vector<FrenetTrajectory> collision_free_paths = check_collision(
                constrained_paths,
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

            // Recompute final cost with obstacle-distance term, matching Python calc_cost flow.
            cost_function.calc_cost(
                collision_free_paths,
                settings.highest_speed,
                obstacles_array,
                num_vertices_array,
                num_time_steps,
                num_obstacles,
                max_vertices,
                time_step_now
            );

            // Store collision_free_paths for this thread
            thread_collision_free_paths[t] = collision_free_paths;
            
            // Store statistics for this thread
            thread_stats[t] = local_stats;
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Merge all thread results
    for (const auto& paths : thread_frenet_paths) {
        result.frenet_paths.insert(result.frenet_paths.end(), paths.begin(), paths.end());
    }
    for (const auto& paths : thread_collision_free_paths) {
        result.collision_free_paths.insert(result.collision_free_paths.end(), paths.begin(), paths.end());
    }
    
    // Aggregate statistics from all threads
    last_stats = PlanStats();  // Reset stats
    for (const auto& stats : thread_stats) {
        last_stats += stats;
    }

    return result;
}

void Frenet_Planner::generate_frenet_frame(const double* centerline_pts, int num_points, int pts_dim) {
    // TODO: Generate Frenet frame from centerline points
    // Expected input: centerline_pts is a flat array of shape [num_points, pts_dim]
    // pts_dim should be 2 (x, y coordinates)
    
    if (centerline_pts == nullptr || num_points < 2 || pts_dim != 2) {
        return;
    }
    
    std::vector<double> x_coords, y_coords;
    for (int i = 0; i < num_points; i++) {
        x_coords.push_back(centerline_pts[i * pts_dim]);
        y_coords.push_back(centerline_pts[i * pts_dim + 1]);
        // #ifdef USE_RECORDER
        //     Recorder::getInstance()->saveData<double>("centerline_pts.x", centerline_pts[i * pts_dim]);
        //     Recorder::getInstance()->saveData<double>("centerline_pts.y", centerline_pts[i * pts_dim + 1]);
        // #endif
    }
    
    if (cubic_spline != nullptr) {
        delete cubic_spline;
    }
    
    cubic_spline = new CubicSpline2D(x_coords, y_coords);

}
