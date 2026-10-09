#include "Frenet_Planner.h"
#include "Recorder4Cpp/recorder.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <limits>
#include <iostream>
#include <stdexcept>

namespace {

// n evenly spaced values in [lo, hi]; the midpoint when n == 1.
std::vector<double> linspace(double lo, double hi, int n) {
    std::vector<double> values;
    if (n == 1) {
        values.push_back(0.5 * (lo + hi));
        return values;
    }
    for (int i = 0; i < n; i++) {
        values.push_back(lo + i * (hi - lo) / (n - 1));
    }
    return values;
}

}  // namespace

Frenet_Planner::Frenet_Planner(const SettingParameters& settings_param, 
                               const VehicleParams& vehicle_param,
                               const double* obs_array,
                               const int* num_verts,
                               int n_time_steps,
                               int n_obstacles,
                               int max_verts)
    : settings(settings_param), 
      vehicle_params(vehicle_param), 
      cubic_spline(nullptr)
{
    obstacles.vertices = obs_array;
    obstacles.num_vertices = num_verts;
    obstacles.num_time_steps = n_time_steps;
    obstacles.num_obstacles = n_obstacles;
    obstacles.max_vertices = max_verts;
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
        for (int t = 0; t < obstacles.num_time_steps; ++t) {
            for (int obs = 0; obs < obstacles.num_obstacles; ++obs) {
                int num_verts = obstacles.num_vertices[t * obstacles.num_obstacles + obs];
                const double* poly = obstacles.polygon(t, obs);
                for (int v = 0; v < obstacles.max_vertices; ++v) {
                    Recorder::getInstance()->saveData<double>("obstacles.t", t);
                    Recorder::getInstance()->saveData<double>("obstacles.obs", obs);
                    Recorder::getInstance()->saveData<double>("obstacles.v", v);
                    Recorder::getInstance()->saveData<double>("obstacles.num_vertices", num_verts);
                    Recorder::getInstance()->saveData<double>("obstacles.x", poly[v * 2]);
                    Recorder::getInstance()->saveData<double>("obstacles.y", poly[v * 2 + 1]);
                }
            }
        }
    #endif
}

bool Frenet_Planner::search_space(double current_s, SearchSpace& space) const {
    double lane_width = settings.max_road_width;
    double left, right;
    if (!road_profile.empty() && !road_profile.at(current_s, lane_width, left, right)) {
        return false;
    }
    const double sampling_width = lane_width - vehicle_params.w;
    if (sampling_width < 0.0) {
        return false;
    }
    space.d_min = -sampling_width / 2.0;
    space.d_max = sampling_width / 2.0;
    space.v_min = settings.lowest_speed;
    space.v_max = settings.highest_speed;
    space.t_min = settings.min_t;
    space.t_max = settings.max_t;
    return true;
}

std::vector<std::tuple<double, double, double>> Frenet_Planner::get_samples(double current_s) {
    SearchSpace space;
    if (!search_space(current_s, space)) {
        return {};
    }

    // Generate all combinations
    std::vector<std::tuple<double, double, double>> samples;
    for (double d : linspace(space.d_min, space.d_max, settings.num_width)) {
        for (double s_d : linspace(space.v_min, space.v_max, settings.num_speed)) {
            for (double t : linspace(space.t_min, space.t_max, settings.num_t)) {
                samples.push_back(std::make_tuple(d, s_d, t));
            }
        }
    }
    return samples;
}

PlanningContext Frenet_Planner::make_context(const FrenetState& start, double v_des, int time_step_now,
                                             bool use_obstacle_cost) const {
    PlanningContext ctx;
    ctx.spline = cubic_spline;
    ctx.road = &road_profile;
    ctx.obstacles = obstacles;
    ctx.vehicle = vehicle_params;
    ctx.dt = settings.tick_t;
    ctx.v_des = v_des;
    ctx.t_max = settings.max_t;
    ctx.time_step_now = time_step_now;
    ctx.low_speed_mode = start.s_d < settings.low_speed_threshold;
    ctx.low_speed_min_lateral_length = settings.low_speed_min_lateral_length;
    ctx.check_boundary = settings.check_boundary;
    ctx.check_obstacle = settings.check_obstacle;
    ctx.use_obstacle_cost = use_obstacle_cost;
    ctx.obstacle_frenet = &obstacle_frenet;
    ctx.check_clearance = settings.check_clearance && !obstacle_frenet.empty();
    ctx.clearance_time_gap = settings.clearance_time_gap;
    ctx.clearance_min_gap = settings.clearance_min_gap;
    ctx.clearance_lateral_margin = settings.clearance_lateral_margin;
    ctx.clearance_grace_time = settings.clearance_grace_time;
    ctx.clearance_recovery_time = settings.clearance_recovery_time;
    if (ctx.check_clearance) {
        ctx.clearance_baseline = clearance_baseline(start, time_step_now);
    }
    return ctx;
}

std::vector<ClearanceBaseline> Frenet_Planner::clearance_baseline(const FrenetState& start,
                                                                  int time_step_now) const {
    const ObstacleFrenetBounds& bounds = obstacle_frenet;
    std::vector<ClearanceBaseline> baseline(bounds.num_obstacles);
    const double dt = settings.tick_t;
    const int horizon = static_cast<int>(std::ceil(settings.max_t / dt));
    const double half_width = 0.5 * vehicle_params.w + settings.clearance_lateral_margin;
    const double speed = std::max(start.s_d, 0.0);

    for (int j = 0; j < bounds.num_obstacles; ++j) {
        ClearanceBaseline& base = baseline[j];
        base.gap.assign(horizon + 1, std::numeric_limits<double>::quiet_NaN());
        for (int k = 0; k <= horizon && time_step_now + k < bounds.num_time_steps; ++k) {
            const double* b = bounds.at(time_step_now + k, j);  // s_min, s_max, l_min, l_max
            const double ego_s = start.s + speed * k * dt;
            if (std::isnan(b[0]) || b[1] <= ego_s ||
                b[3] < start.d - half_width || b[2] > start.d + half_width) {
                continue;
            }
            if (base.k0 < 0) base.k0 = k;
            base.gap[k] = std::max(0.0, b[0] - (ego_s + 0.5 * vehicle_params.l));
        }
    }
    return baseline;
}

bool Frenet_Planner::fall_back_without_clearance(PlanningContext& context) {
    if (!context.check_clearance || !settings.clearance_fallback) {
        return false;
    }
    context.check_clearance = false;
    ++last_stats.num_clearance_fallbacks;
    return true;
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
                                      int num_threads,
                                      double desired_speed) {
    // The speed range of the grid must already be the one of this cycle
    settings.highest_speed = max_target_speed;
    std::vector<std::tuple<double, double, double>> samples = get_samples(frenet_state.s);
    return best_traj_generation(frenet_state, samples, max_target_speed, time_step_now, num_threads,
                                desired_speed);
}

FrenetTrajectory Frenet_Planner::best_traj_generation(
    const FrenetState& frenet_state,
    const std::vector<std::tuple<double, double, double>>& samples,
    double max_target_speed,
    int time_step_now,
    int num_threads,
    double desired_speed) {
    const auto t_start = std::chrono::steady_clock::now();
    settings.highest_speed = max_target_speed;
    last_stats = PlanStats();
    last_fplist.clear();
    best_traj = FrenetTrajectory();

    // Handle edge case: no samples
    if (samples.empty()) {
        std::cerr << "empty samples" << std::endl;
        return FrenetTrajectory();
    }

    std::vector<SamplingParam> params;
    params.reserve(samples.size());
    for (const auto& sample : samples) {
        params.emplace_back(std::get<0>(sample), std::get<1>(sample), std::get<2>(sample));
    }

    // FOP evaluates complete trajectories, so J_D is part of its objective
    PlanningContext context = make_context(
        frenet_state, resolve_desired_speed(max_target_speed, desired_speed), time_step_now, true);
    last_fplist = evaluate_batch(frenet_state, params, context, num_threads).feasible;
    if (last_fplist.empty() && fall_back_without_clearance(context)) {
        last_fplist = evaluate_batch(frenet_state, params, context, num_threads).feasible;
    }

    // Find minimum cost path
    best_traj.cost_final = std::numeric_limits<double>::infinity();
    for (const auto& fp : last_fplist) {
        if (fp.cost_final < best_traj.cost_final) {
            best_traj = fp;
        }
    }

    #ifdef USE_RECORDER
        Recorder::getInstance()->writeDataToCSV();
    #endif

    last_stats.timing.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
    return best_traj;
}

std::vector<EvaluationResult> Frenet_Planner::evaluate_samples(
    const FrenetState& frenet_state,
    const std::vector<std::tuple<double, double, double>>& samples,
    double max_target_speed,
    int time_step_now,
    bool full_violation,
    double desired_speed) {
    last_stats = PlanStats();
    std::vector<SamplingParam> params;
    params.reserve(samples.size());
    for (const auto& sample : samples) {
        params.emplace_back(std::get<0>(sample), std::get<1>(sample), std::get<2>(sample));
    }
    const PlanningContext context = make_context(
        frenet_state, resolve_desired_speed(max_target_speed, desired_speed), time_step_now, true);
    return evaluate_batch(frenet_state, params, context, 1,
                          full_violation ? EvalMode::kFullViolation : EvalMode::kEarlyExit).results;
}

FrenetTrajectory Frenet_Planner::generate_trajectory(const FrenetState& frenet_state, double d, double s_d,
                                                     double t, int time_step_now) const {
    const PlanningContext context = make_context(frenet_state, settings.highest_speed, time_step_now, false);
    const TrajectoryEvaluator evaluator(context, cost_function);
    PlanStats stats;
    FrenetTrajectory traj = evaluator.generate(frenet_state, SamplingParam(d, s_d, t), stats);
    evaluator.to_global(traj);
    return traj;
}

BatchResult Frenet_Planner::evaluate_batch(const FrenetState& start,
                                           const std::vector<SamplingParam>& samples,
                                           const PlanningContext& context,
                                           int num_threads,
                                           EvalMode mode) {
    if (num_threads <= 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    }
    const int n = static_cast<int>(samples.size());
    num_threads = std::max(1, std::min(num_threads, n));
    const int chunk = (n + num_threads - 1) / num_threads;

    const TrajectoryEvaluator evaluator(context, cost_function);
    BatchResult batch;
    batch.results.resize(n);
    std::vector<std::vector<FrenetTrajectory>> thread_feasible(num_threads);
    std::vector<PlanStats> thread_stats(num_threads);

    auto worker = [&](int t) {
        const int end = std::min(n, (t + 1) * chunk);
        for (int i = t * chunk; i < end; ++i) {
            FrenetTrajectory traj = evaluator.generate(start, samples[i], thread_stats[t]);
            batch.results[i] = evaluator.evaluate(traj, thread_stats[t], mode);
            if (batch.results[i].feasible) {
                thread_feasible[t].push_back(std::move(traj));
            }
        }
    };

    if (num_threads == 1) {
        worker(0);
    } else {
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back(worker, t);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }

    // Merge thread results, keeping the sample order
    for (int t = 0; t < num_threads; t++) {
        batch.feasible.insert(batch.feasible.end(),
                              std::make_move_iterator(thread_feasible[t].begin()),
                              std::make_move_iterator(thread_feasible[t].end()));
        last_stats += thread_stats[t];
    }
    return batch;
}

void Frenet_Planner::generate_frenet_frame(const double* centerline_pts, int num_points, int pts_dim) {
    // Expected input: centerline_pts is a flat array of shape [num_points, pts_dim]
    // pts_dim should be 2 (x, y coordinates)
    
    if (centerline_pts == nullptr || num_points < 2 || pts_dim != 2) {
        return;
    }
    
    std::vector<double> x_coords, y_coords;
    for (int i = 0; i < num_points; i++) {
        x_coords.push_back(centerline_pts[i * pts_dim]);
        y_coords.push_back(centerline_pts[i * pts_dim + 1]);
    }
    
    if (cubic_spline != nullptr) {
        delete cubic_spline;
    }
    
    cubic_spline = new CubicSpline2D(x_coords, y_coords);
    // A profile belongs to one reference frame; never reuse it for a new route.
    road_profile.clear();
    obstacle_frenet = ObstacleFrenetBounds();
}

void Frenet_Planner::set_obstacle_frenet_bounds(const double* bounds, int num_time_steps, int num_obstacles) {
    if (bounds == nullptr || num_time_steps <= 0 || num_obstacles <= 0) {
        obstacle_frenet = ObstacleFrenetBounds();
        return;
    }
    obstacle_frenet.num_time_steps = num_time_steps;
    obstacle_frenet.num_obstacles = num_obstacles;
    obstacle_frenet.data.assign(bounds, bounds + 4L * num_time_steps * num_obstacles);
}

void Frenet_Planner::set_road_profile(const std::vector<double>& s,
                                      const std::vector<double>& lane_width,
                                      const std::vector<double>& left_extent,
                                      const std::vector<double>& right_extent) {
    if (cubic_spline == nullptr) {
        throw std::invalid_argument("Road profile requires a reference frame and equally sized arrays of at least two points");
    }
    road_profile.set(s, lane_width, left_extent, right_extent, cubic_spline->s.back());
}
