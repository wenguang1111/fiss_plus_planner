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

// Longitudinal speed below which the vehicle counts as standing still [m/s]
constexpr double kStandstillSpeed = 1e-3;
// Reversing tolerance of the no-reversing constraint [m/s]
constexpr double kReverseTolerance = 1e-2;
// Arc-length step of the numerical reference curvature derivative k_r' [m]
constexpr double kCurvatureStep = 0.1;

// d' and d'' at the start: given directly, otherwise from the time derivatives (Werling
// (A.7), (A.8)); 0 at standstill without that information (aligned with the reference).
void start_arc_derivatives(const FrenetState& start, double& d_s, double& d_ss) {
    if (std::isfinite(start.d_s)) {
        d_s = start.d_s;
        d_ss = std::isfinite(start.d_ss) ? start.d_ss : 0.0;
    } else if (start.s_d > kStandstillSpeed) {
        d_s = start.d_d / start.s_d;
        d_ss = (start.d_dd - d_s * start.s_dd) / (start.s_d * start.s_d);
    } else {
        d_s = 0.0;
        d_ss = 0.0;
    }
}

// Share of the horizon from the first violating sample on, 0 if there is none.
double violation_from(long first_violation, size_t n) {
    if (first_violation < 0 || n == 0) return 0.0;
    return static_cast<double>(n - static_cast<size_t>(first_violation)) / static_cast<double>(n);
}

// First sample whose value (or |value|) exceeds limit, -1 if none.
long first_exceeding(const std::vector<double>& values, double limit, bool use_abs) {
    for (size_t k = 0; k < values.size(); ++k) {
        if ((use_abs ? std::abs(values[k]) : values[k]) > limit) return static_cast<long>(k);
    }
    return -1;
}

}  // namespace

bool ranks_before(const EvaluationResult& a, const EvaluationResult& b) {
    if (a.feasible != b.feasible) return a.feasible;
    if (!a.feasible) return a.violation.total() < b.violation.total();
    return a.cost.total() < b.cost.total();
}

EvaluationResult early_exit_view(EvaluationResult full) {
    ConstraintViolation& v = full.violation;
    bool failed = false;
    for (double* component : {&v.transform, &v.speed, &v.acceleration, &v.road, &v.clearance, &v.collision}) {
        if (failed) {
            *component = 0.0;
        } else if (*component > 0.0) {
            failed = true;
        }
    }
    if (!full.feasible) {
        full.has_cost = false;
        full.cost = CostBreakdown();
    }
    return full;
}

FrenetTrajectory TrajectoryEvaluator::generate(const FrenetState& start, const SamplingParam& z,
                                               PlanStats& stats) const {
    const auto t0 = Clock::now();
    const double Ti = std::max(z.t, ctx_.dt);

    FrenetTrajectory fp;
    QuarticPolynomial lon_qp(start.s, start.s_d, start.s_dd, z.s_d, 0.0, Ti);
    for (double t = 0.0; t < Ti; t += ctx_.dt) {
        fp.t.push_back(t);
        fp.s.push_back(lon_qp.calc_point(t));
        fp.s_d.push_back(lon_qp.calc_first_derivative(t));
        fp.s_dd.push_back(lon_qp.calc_second_derivative(t));
        fp.s_ddd.push_back(lon_qp.calc_third_derivative(t));
    }

    double d_s, d_ss;
    start_arc_derivatives(start, d_s, d_ss);
    if (!ctx_.low_speed_mode) {
        // High speed: lateral quintic over time, d' and d'' from Werling (A.7), (A.8);
        // at standstill they keep their last value.
        QuinticPolynomial lat_qp(start.d, start.d_d, start.d_dd, z.d, 0.0, 0.0, Ti);
        for (size_t k = 0; k < fp.t.size(); ++k) {
            const double t = fp.t[k];
            const double sd = fp.s_d[k];
            fp.d.push_back(lat_qp.calc_point(t));
            fp.d_d.push_back(lat_qp.calc_first_derivative(t));
            fp.d_dd.push_back(lat_qp.calc_second_derivative(t));
            fp.d_ddd.push_back(lat_qp.calc_third_derivative(t));
            if (sd > kStandstillSpeed) {
                d_s = fp.d_d.back() / sd;
                d_ss = (fp.d_dd.back() - d_s * fp.s_dd[k]) / (sd * sd);
            }
            fp.d_s.push_back(d_s);
            fp.d_ss.push_back(d_ss);
        }
    } else {
        // Low speed (Werling's thesis Sec. 3.5.1): lateral quintic over the travelled arc
        // length sigma = s - s0, ending where the longitudinal trajectory ends, so lateral
        // motion only happens with longitudinal motion. Time derivatives by the chain rule.
        const double length = std::max(lon_qp.calc_point(Ti) - start.s, ctx_.low_speed_min_lateral_length);
        QuinticPolynomial lat_qp(start.d, d_s, d_ss, z.d, 0.0, 0.0, length);
        for (size_t k = 0; k < fp.t.size(); ++k) {
            const double sigma = std::clamp(fp.s[k] - start.s, 0.0, length);
            const double p1 = lat_qp.calc_first_derivative(sigma);
            const double p2 = lat_qp.calc_second_derivative(sigma);
            const double p3 = lat_qp.calc_third_derivative(sigma);
            const double sd = fp.s_d[k], sdd = fp.s_dd[k], sddd = fp.s_ddd[k];
            fp.d.push_back(lat_qp.calc_point(sigma));
            fp.d_s.push_back(p1);
            fp.d_ss.push_back(p2);
            fp.d_d.push_back(p1 * sd);
            fp.d_dd.push_back(p2 * sd * sd + p1 * sdd);
            fp.d_ddd.push_back(p3 * sd * sd * sd + 3.0 * p2 * sd * sdd + p1 * sddd);
        }
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

    // Heading and curvature from the Frenet state (Werling's thesis, Appendix A.1):
    //   d' = (1 - k_r d) tan(dtheta)                                                  (A.3)
    //   d'' = -(k_r' d + k_r d') tan(dtheta)
    //         + (1 - k_r d) / cos^2(dtheta) * (k (1 - k_r d) / cos(dtheta) - k_r)     (A.5)
    // solved for dtheta = theta - theta_r and k; well defined at standstill.
    const double s_end = ctx_.spline->s.back();
    const size_t n = std::min({fp.s.size(), fp.d.size(), fp.d_s.size(), fp.d_ss.size()});
    for (size_t i = 0; i < n; i++) {
        auto [ix, iy] = ctx_.spline->calc_position(fp.s[i]);
        // Stop adding points if position is invalid (beyond the reference line)
        if (std::isnan(ix) || std::isnan(iy)) {
            break;
        }
        const double d = fp.d[i];
        const double ref_yaw = ctx_.spline->calc_yaw(fp.s[i]);
        const double k_r = ctx_.spline->calc_curvature(fp.s[i]);
        const double s_lo = std::max(fp.s[i] - kCurvatureStep, 0.0);
        const double s_hi = std::min(fp.s[i] + kCurvatureStep, s_end);
        const double k_r_d = s_hi > s_lo
            ? (ctx_.spline->calc_curvature(s_hi) - ctx_.spline->calc_curvature(s_lo)) / (s_hi - s_lo)
            : 0.0;
        const double one_kr_d = 1.0 - k_r * d;
        if (!(one_kr_d > 0.0)) {
            break;  // beyond the reference line's center of curvature
        }
        const double dtheta = std::atan(fp.d_s[i] / one_kr_d);
        const double cos_t = std::cos(dtheta);
        fp.x.push_back(ix + d * std::cos(ref_yaw + M_PI / 2.0));
        fp.y.push_back(iy + d * std::sin(ref_yaw + M_PI / 2.0));
        fp.yaw.push_back(ref_yaw + dtheta);
        fp.c.push_back(((fp.d_ss[i] + (k_r_d * d + k_r * fp.d_s[i]) * std::tan(dtheta)) * cos_t * cos_t / one_kr_d + k_r)
                       * cos_t / one_kr_d);
    }
    if (fp.x.size() < 2) {
        return false;
    }

    for (size_t i = 0; i + 1 < fp.x.size(); i++) {
        fp.ds.push_back(std::hypot(fp.x[i + 1] - fp.x[i], fp.y[i + 1] - fp.y[i]));
    }
    const double dt = ctx_.dt;
    for (size_t i = 0; i + 1 < fp.c.size(); i++) {
        fp.c_d.push_back((fp.c[i + 1] - fp.c[i]) / dt);
    }
    for (size_t i = 0; i + 1 < fp.c_d.size(); i++) {
        fp.c_dd.push_back((fp.c_d[i + 1] - fp.c_d[i]) / dt);
    }
    return true;
}

void TrajectoryEvaluator::check_dynamics(const FrenetTrajectory& traj, bool early_exit,
                                         ConstraintViolation& v) const {
    // Speed: above the vehicle maximum or reversing (s_d < 0)
    long first_speed = first_exceeding(traj.s_d, ctx_.vehicle.max_speed, false);
    for (size_t k = 0; k < traj.s_d.size(); ++k) {
        if (first_speed >= 0 && static_cast<long>(k) >= first_speed) break;
        if (traj.s_d[k] < -kReverseTolerance) { first_speed = static_cast<long>(k); break; }
    }
    v.speed = violation_from(first_speed, traj.s.size());
    if (early_exit && v.speed > 0.0) return;
    v.acceleration = violation_from(first_exceeding(traj.s_dd, ctx_.vehicle.max_accel, true), traj.s.size());
}

double TrajectoryEvaluator::road_violation(const FrenetTrajectory& traj) const {
    // Projects the rectangular footprint onto the local reference normal. This is a
    // local Frenet envelope, not an exact polygon containment test on curved roads or
    // between discrete time steps. Samples without road geometry (incomplete horizons,
    // beyond the profile) count as violations.
    constexpr double kTol = 1e-6;
    const size_t n = traj.s.size();
    if (n == 0 || traj.d.size() != n) {
        return 1.0;
    }
    for (size_t i = 0; i < n; ++i) {
        double width, left, right;
        if (i >= traj.yaw.size() || !ctx_.road->at(traj.s[i], width, left, right) ||
            !std::isfinite(traj.d[i]) || !std::isfinite(traj.yaw[i])) {
            return violation_from(static_cast<long>(i), n);
        }
        const double delta_yaw = traj.yaw[i] - ctx_.spline->calc_yaw(traj.s[i]);
        const double half_extent = 0.5 * (ctx_.vehicle.w * std::abs(std::cos(delta_yaw)) +
                                          ctx_.vehicle.l * std::abs(std::sin(delta_yaw)));
        if (!std::isfinite(half_extent) || traj.d[i] - half_extent < -right - kTol ||
            traj.d[i] + half_extent > left + kTol) {
            return violation_from(static_cast<long>(i), n);
        }
    }
    return 0.0;
}

double TrajectoryEvaluator::clearance_violation(const FrenetTrajectory& traj) const {
    // Obstacles ahead in the ego path: lateral extent overlapping the ego's lateral
    // interval (plus margin) at the same time step, rear beyond the ego center. The gap
    // from the ego front to the obstacle rear must reach the target min_gap + time_gap * v.
    // Where the baseline gap (ego driving on at its current speed) is smaller, the
    // requirement starts at the baseline gap and reaches the target within
    // clearance_recovery_time after the obstacle is first met (step k0). It grows with the
    // square of the elapsed share of that time, like the distance gained by braking.
    const ObstacleFrenetBounds& bounds = *ctx_.obstacle_frenet;
    const size_t n = traj.s.size();
    const int first_step = static_cast<int>(std::ceil(ctx_.clearance_grace_time / ctx_.dt - 1e-9));
    const double half_width = 0.5 * ctx_.vehicle.w + ctx_.clearance_lateral_margin;
    const double half_length = 0.5 * ctx_.vehicle.l;

    for (size_t k = static_cast<size_t>(std::max(first_step, 0)); k < n; ++k) {
        const int t_idx = ctx_.time_step_now + static_cast<int>(k);
        if (t_idx >= bounds.num_time_steps) break;
        const double s = traj.s[k];
        const double d = traj.d[k];
        const double target = ctx_.clearance_min_gap + ctx_.clearance_time_gap * std::max(traj.s_d[k], 0.0);
        for (int j = 0; j < bounds.num_obstacles; ++j) {
            const double* b = bounds.at(t_idx, j);  // s_min, s_max, l_min, l_max
            if (std::isnan(b[0]) || b[1] <= s) continue;                       // absent or behind
            if (b[3] < d - half_width || b[2] > d + half_width) continue;      // not in the path
            double required = target;
            if (j < static_cast<int>(ctx_.clearance_baseline.size())) {
                const ClearanceBaseline& base = ctx_.clearance_baseline[j];
                const double coast_gap = k < base.gap.size() ? base.gap[k] : std::nan("");
                if (base.k0 >= 0 && !std::isnan(coast_gap) && coast_gap < target) {
                    const double progress = ctx_.clearance_recovery_time > 0.0
                        ? std::min(1.0, (static_cast<int>(k) - base.k0) * ctx_.dt / ctx_.clearance_recovery_time)
                        : 1.0;
                    required = coast_gap + (target - coast_gap) * progress * progress;
                }
            }
            if (b[0] - (s + half_length) < required) {
                return violation_from(static_cast<long>(k), n);
            }
        }
    }
    return 0.0;
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
        result.violation.road = road_violation(traj);
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

    // Step 3: safe following distance (cheap interval test before the polygon check)
    if (ctx_.check_clearance && ctx_.obstacle_frenet != nullptr && !ctx_.obstacle_frenet->empty()) {
        t0 = Clock::now();
        ++stats.num_clearance_checks;
        result.violation.clearance = clearance_violation(traj);
        stats.timing.collision_ms += ms_since(t0);
        if (result.violation.clearance > 0.0) {
            ++stats.num_rejected_clearance;
            reject(Rejection::kClearance);
            if (early_exit) return result;
        }
    }

    // Step 4: polygon collision check
    if (ctx_.check_obstacle) {
        t0 = Clock::now();
        ++stats.num_collision_checks;
        const int first_hit = first_collision_step(
            traj, ctx_.obstacles, ctx_.vehicle.l, ctx_.vehicle.w, ctx_.time_step_now);
        stats.timing.collision_ms += ms_since(t0);
        result.violation.collision = violation_from(first_hit, traj.s.size());
        if (first_hit >= 0) {
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
