#include "trajectory_evaluator.h"
#include "../collision/collision_checker.h"
#include "../geometry/polynomial.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Mean of max(0, (|x| - limit) / limit)^2; > 0 <=> some |x| exceeds the limit.
// use_abs = false only penalizes positive excess, matching the speed check.
double hinge_violation(const std::vector<double>& values, double limit, bool use_abs,
                       bool stop_at_first) {
    if (values.empty()) return 0.0;
    const double scale = std::max(std::abs(limit), 1e-6);
    double sum = 0.0;
    for (double x : values) {
        const double excess = (use_abs ? std::abs(x) : x) - limit;
        if (excess > 0.0) {
            const double normalized = excess / scale;
            sum += normalized * normalized;
            if (stop_at_first) break;
        }
    }
    return sum / static_cast<double>(values.size());
}

}  // namespace

bool ranks_before(const EvaluationResult& a, const EvaluationResult& b) {
    if (a.feasible != b.feasible) return a.feasible;
    if (!a.feasible) return a.violation.total() < b.violation.total();
    return a.cost.total() < b.cost.total();
}

FrenetTrajectory TrajectoryEvaluator::generate(const FrenetState& start, const SamplingParam& z,
                                               PlanStats& stats) const {
    const auto t0 = Clock::now();
    const double Ti = std::max(z.t, ctx_.dt);

    FrenetTrajectory fp;
    QuinticPolynomial lat_qp(start.d, start.d_d, start.d_dd, z.d, 0.0, 0.0, Ti);
    QuarticPolynomial lon_qp(start.s, start.s_d, start.s_dd, z.s_d, 0.0, Ti);

    for (double t = 0.0; t < Ti; t += ctx_.dt) {
        fp.t.push_back(t);
        fp.d.push_back(lat_qp.calc_point(t));
        fp.d_d.push_back(lat_qp.calc_first_derivative(t));
        fp.d_dd.push_back(lat_qp.calc_second_derivative(t));
        fp.d_ddd.push_back(lat_qp.calc_third_derivative(t));
        fp.s.push_back(lon_qp.calc_point(t));
        fp.s_d.push_back(lon_qp.calc_first_derivative(t));
        fp.s_dd.push_back(lon_qp.calc_second_derivative(t));
        fp.s_ddd.push_back(lon_qp.calc_third_derivative(t));
    }

    fp.is_generated = true;
    fp.sampling_param = SamplingParam(z.d, z.s_d, Ti);
    fp.end_state.t = Ti;
    fp.end_state.s_d = z.s_d;
    fp.end_state.d = z.d;

    ++stats.num_trajs_generated;
    stats.timing.generation_ms += ms_since(t0);
    return fp;
}

CostBreakdown TrajectoryEvaluator::frenet_cost(const FrenetTrajectory& traj, PlanStats& stats) const {
    const auto t0 = Clock::now();
    CostBreakdown cost = cost_.frenet_terms(traj, ctx_.v_des, ctx_.t_max,
                                            traj.sampling_param.t, ctx_.dt);
    ++stats.num_cost_evaluations;
    stats.timing.cost_ms += ms_since(t0);
    return cost;
}

bool TrajectoryEvaluator::to_global(FrenetTrajectory& fp) const {
    fp.x.clear(); fp.y.clear(); fp.yaw.clear(); fp.ds.clear();
    fp.c.clear(); fp.c_d.clear(); fp.c_dd.clear();
    if (ctx_.spline == nullptr) {
        return false;
    }

    const size_t n = std::min(fp.s.size(), fp.d.size());
    for (size_t i = 0; i < n; i++) {
        auto [ix, iy] = ctx_.spline->calc_position(fp.s[i]);
        // Stop adding points if position is invalid (beyond the reference line)
        if (std::isnan(ix) || std::isnan(iy)) {
            break;
        }
        const double i_yaw = ctx_.spline->calc_yaw(fp.s[i]);
        const double di = fp.d[i];
        fp.x.push_back(ix + di * std::cos(i_yaw + M_PI / 2.0));
        fp.y.push_back(iy + di * std::sin(i_yaw + M_PI / 2.0));
    }
    if (fp.x.size() < 2) {
        return false;
    }

    for (size_t i = 0; i + 1 < fp.x.size(); i++) {
        const double dx = fp.x[i + 1] - fp.x[i];
        const double dy = fp.y[i + 1] - fp.y[i];
        fp.yaw.push_back(std::atan2(dy, dx));
        fp.ds.push_back(std::sqrt(dx * dx + dy * dy));
    }
    fp.yaw.push_back(fp.yaw.back());

    const double dt = ctx_.dt;
    for (size_t i = 0; i + 1 < fp.yaw.size(); i++) {
        fp.c.push_back((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i]);
    }
    for (size_t i = 0; i + 1 < fp.c.size(); i++) {
        fp.c_d.push_back((fp.c[i + 1] - fp.c[i]) / dt);
    }
    for (size_t i = 0; i + 1 < fp.c_d.size(); i++) {
        fp.c_dd.push_back((fp.c_d[i + 1] - fp.c_d[i]) / dt);
    }
    return true;
}

void TrajectoryEvaluator::check_dynamics(const FrenetTrajectory& traj, bool stop_at_first,
                                         ConstraintViolation& v) const {
    v.speed = hinge_violation(traj.s_d, ctx_.vehicle.max_speed, false, stop_at_first);
    if (stop_at_first && v.speed > 0.0) return;
    v.acceleration = hinge_violation(traj.s_dd, ctx_.vehicle.max_accel, true, stop_at_first);
}

void TrajectoryEvaluator::check_road(const FrenetTrajectory& traj, bool stop_at_first,
                                     ConstraintViolation& v) const {
    // Projects the rectangular footprint onto the local reference normal. This is a
    // local Frenet envelope, not an exact polygon containment test on curved roads or
    // between discrete time steps. Incomplete horizons are rejected rather than
    // checking only their prefix: every sample without geometry counts 1.
    constexpr double kTol = 1e-6;
    const size_t n = traj.s.size();
    if (n == 0 || traj.d.size() != n) {
        v.road = 1.0;
        return;
    }
    const double w = std::max(ctx_.vehicle.w, 1e-6);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double width, left, right;
        if (i >= traj.yaw.size() || !ctx_.road->at(traj.s[i], width, left, right) ||
            !std::isfinite(traj.d[i]) || !std::isfinite(traj.yaw[i])) {
            sum += 1.0;
        } else {
            const double delta_yaw = traj.yaw[i] - ctx_.spline->calc_yaw(traj.s[i]);
            const double half_extent = 0.5 * (ctx_.vehicle.w * std::abs(std::cos(delta_yaw)) +
                                              ctx_.vehicle.l * std::abs(std::sin(delta_yaw)));
            if (!std::isfinite(half_extent)) {
                sum += 1.0;
            } else {
                const double e_left = std::max(0.0, traj.d[i] + half_extent - left - kTol) / w;
                const double e_right = std::max(0.0, -right - (traj.d[i] - half_extent) - kTol) / w;
                sum += e_left * e_left + e_right * e_right;
            }
        }
        if (stop_at_first && sum > 0.0) break;
    }
    v.road = sum / static_cast<double>(n);
}

EvaluationResult TrajectoryEvaluator::check_feasibility(FrenetTrajectory& traj, PlanStats& stats,
                                                        EvalMode mode) const {
    const bool early_exit = (mode == EvalMode::kEarlyExit);
    EvaluationResult result;
    auto reject = [&result](Rejection r) {
        if (result.rejection == Rejection::kNone) result.rejection = r;
    };

    // Step 1: Frenet -> Cartesian
    auto t0 = Clock::now();
    const bool transformed = to_global(traj);
    ++stats.num_global_transforms;
    stats.timing.transform_ms += ms_since(t0);
    if (!transformed) {
        ++stats.num_rejected_transform;
        result.violation.transform = 1.0;
        reject(Rejection::kTransform);
        return result;  // nothing below can be checked without Cartesian points
    }

    // Step 2: dynamic and road-boundary constraints
    t0 = Clock::now();
    ++stats.num_constraint_checks;
    check_dynamics(traj, early_exit, result.violation);
    const bool dynamic_ok = result.violation.speed == 0.0 && result.violation.acceleration == 0.0;
    bool road_ok = true;
    if ((dynamic_ok || !early_exit) && ctx_.check_boundary && ctx_.road != nullptr && !ctx_.road->empty()) {
        check_road(traj, early_exit, result.violation);
        road_ok = result.violation.road == 0.0;
    }
    stats.timing.constraint_ms += ms_since(t0);
    if (!dynamic_ok) {
        ++stats.num_rejected_dynamic;
        reject(Rejection::kDynamic);
    } else if (!road_ok) {
        ++stats.num_rejected_offroad;
        reject(Rejection::kOffroad);
    } else {
        ++stats.num_constraint_passed;
        traj.constraint_passed = true;
    }
    if (early_exit && result.rejection != Rejection::kNone) {
        return result;
    }

    // Step 3: polygon collision check
    if (ctx_.check_obstacle) {
        t0 = Clock::now();
        ++stats.num_collision_checks;
        const TrajectoryCollision collision = check_trajectory_collision(
            traj, ctx_.obstacles, ctx_.vehicle.l, ctx_.vehicle.w, ctx_.time_step_now, early_exit);
        stats.timing.collision_ms += ms_since(t0);
        result.violation.collision = collision.violation;
        if (collision.collided) {
            reject(Rejection::kCollision);
        } else {
            ++stats.num_collision_free;
        }
    }

    result.feasible = (result.rejection == Rejection::kNone);
    traj.collision_passed = result.feasible;
    return result;
}

EvaluationResult TrajectoryEvaluator::evaluate(FrenetTrajectory& traj, PlanStats& stats,
                                               EvalMode mode) const {
    EvaluationResult result = check_feasibility(traj, stats, mode);
    if (!result.feasible && mode == EvalMode::kEarlyExit) {
        return result;
    }

    result.cost = frenet_cost(traj, stats);
    if (ctx_.use_obstacle_cost) {
        const auto t0 = Clock::now();
        result.cost.obstacle = cost_.obstacle_term(traj, ctx_.obstacles, ctx_.time_step_now, ctx_.dt);
        stats.timing.cost_ms += ms_since(t0);
    }
    result.has_cost = true;
    traj.cost_final = result.cost.total();
    return result;
}
