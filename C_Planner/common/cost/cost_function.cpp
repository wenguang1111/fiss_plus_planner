#include "cost_function.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

double sum_sq(const std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) sum += v * v;
    return sum;
}

double sum_sq_offset(const std::vector<double>& values, double ref) {
    double sum = 0.0;
    for (double v : values) {
        const double diff = ref - v;
        sum += diff * diff;
    }
    return sum;
}

}  // namespace

CostBreakdown CostFunction::frenet_terms(const FrenetTrajectory& traj, double v_des,
                                         double t_max, double t_f, double dt) const {
    CostBreakdown cost;
    cost.time = weights.w_T * (t_max - t_f);
    cost.velocity = weights.w_V * sum_sq_offset(traj.s_d, v_des) * dt;
    cost.acceleration = weights.w_A * (sum_sq(traj.s_dd) + sum_sq(traj.d_dd)) * dt;
    cost.jerk = weights.w_J * (sum_sq(traj.s_ddd) + sum_sq(traj.d_ddd)) * dt;
    cost.lane_center = weights.w_LC * sum_sq(traj.d) * dt;
    return cost;
}

double CostFunction::obstacle_term(const FrenetTrajectory& traj, const ObstacleView& obstacles,
                                   int time_step_now, double dt) const {
    if (obstacles.empty()) {
        return 0.0;
    }

    const size_t num_points = std::min({traj.x.size(), traj.y.size(), traj.yaw.size()});
    double xi_sum = 0.0;
    for (size_t k = 0; k < num_points; ++k) {
        const int t_idx = time_step_now + static_cast<int>(k);
        if (t_idx < 0 || t_idx >= obstacles.num_time_steps) {
            break;
        }
        const double heading_x = std::cos(traj.yaw[k]);
        const double heading_y = std::sin(traj.yaw[k]);

        double max_xi = 0.0;
        for (int j = 0; j < obstacles.num_obstacles; ++j) {
            const int n = obstacles.vertex_count(t_idx, j);
            if (n <= 0) {
                continue;
            }
            const double* poly = obstacles.polygon(t_idx, j);
            double cx = 0.0, cy = 0.0;
            for (int v = 0; v < n; ++v) {
                cx += poly[2 * v];
                cy += poly[2 * v + 1];
            }
            const double dx = cx / n - traj.x[k];
            const double dy = cy / n - traj.y[k];
            if (dx * heading_x + dy * heading_y <= 0.0) {
                continue;  // only obstacles in front of the ego vehicle
            }
            max_xi = std::max(max_xi, std::exp(-weights.w_dist * std::hypot(dx, dy)));
        }
        xi_sum += max_xi;
    }
    return weights.w_D * xi_sum * dt;
}

CostBreakdown CostFunction::scenario_cost(const FrenetTrajectory& executed,
                                          const std::vector<double>& v_des,
                                          double dt, const ObstacleView& obstacles) const {
    const size_t n = executed.s_d.size();
    if (v_des.size() != n) {
        throw std::invalid_argument("scenario_cost needs one v_des per executed sample");
    }
    const double t_f = n > 0 ? static_cast<double>(n - 1) * dt : 0.0;
    CostBreakdown cost = frenet_terms(executed, 0.0, 0.0, t_f, dt);
    double velocity_sq = 0.0;
    for (size_t k = 0; k < n; ++k) {
        const double diff = v_des[k] - executed.s_d[k];
        velocity_sq += diff * diff;
    }
    cost.velocity = weights.w_V * velocity_sq * dt;
    cost.time = weights.w_T * t_f;
    cost.obstacle = obstacle_term(executed, obstacles, 0, dt);
    return cost;
}
