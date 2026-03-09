#include "cost_function.h"
#include <limits>

CostFunction::CostFunction(const std::string& cost_type) {
    w_T = 1.0;
    w_V = 0.1;
    w_A = 10.0;
    w_J = 10.0;
    w_D = 100.0;
    w_LC = 10.0;
    w_dist = 0.1;
    max_speed = 14.0;
}

double CostFunction::cost_terminal_time(double terminal_time) {
    return w_T * terminal_time;
}

double CostFunction::cost_velocity_offset(const std::vector<double>& vels, double v_target) {
    double cost = 0.0;
    for (double vel : vels) {
        double diff = vel - v_target;
        cost += diff * diff;
    }
    return w_V * cost;
}

double CostFunction::cost_acceleration(const std::vector<double>& accels) {
    double cost = 0.0;
    for (double accel : accels) {
        cost += accel * accel;
    }
    return w_A * cost;
}

double CostFunction::cost_jerk(const std::vector<double>& jerks) {
    double cost = 0.0;
    for (double jerk : jerks) {
        cost += jerk * jerk;
    }
    return w_J * cost;
}

double CostFunction::cost_lane_center_offset(const std::vector<double>& offsets) {
    double cost = 0.0;
    for (double offset : offsets) {
        cost += offset * offset;
    }
    return w_LC * cost;
}

double CostFunction::cost_dist_obstacle(const double* obstacles_array,
                                        const int* num_vertices_array,
                                        int num_time_steps,
                                        int num_obstacles,
                                        int max_vertices,
                                        const FrenetTrajectory& traj,
                                        int time_step_now) {
    if (obstacles_array == nullptr || num_vertices_array == nullptr ||
        num_time_steps <= 0 || num_obstacles <= 0 || max_vertices <= 0 ||
        traj.x.empty() || traj.y.empty()) {
        return 0.0;
    }

    const size_t num_traj_points = std::min(traj.x.size(), traj.y.size());
    double exp_sum = 0.0;

    for (size_t i = 0; i < num_traj_points; ++i) {
        const int t_idx = time_step_now + i;
        if (t_idx < 0 || t_idx >= num_time_steps) {
            break;
        }

        const double traj_x = traj.x[i];
        const double traj_y = traj.y[i];
        double min_dist = std::numeric_limits<double>::infinity();

        for (int obs_idx = 0; obs_idx < num_obstacles; ++obs_idx) {
            const int num_verts = num_vertices_array[t_idx * num_obstacles + obs_idx];
            const int valid_num_verts = std::min(num_verts, max_vertices);
            if (valid_num_verts <= 0) {
                continue;
            }

            double center_x = 0.0;
            double center_y = 0.0;
            for (int v = 0; v < valid_num_verts; ++v) {
                const int array_idx = (t_idx * num_obstacles + obs_idx) * max_vertices * 2 + v * 2;
                center_x += obstacles_array[array_idx];
                center_y += obstacles_array[array_idx + 1];
            }
            center_x /= static_cast<double>(valid_num_verts);
            center_y /= static_cast<double>(valid_num_verts);

            const double dist = std::hypot(traj_x - center_x, traj_y - center_y);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }

        if (std::isfinite(min_dist)) {
            exp_sum += std::exp(-min_dist * w_dist);
        }
    }

    return w_D * exp_sum;
}

double CostFunction::cost_singleTrajectory(const FrenetTrajectory& traj,
                                           double target_speed,
                                           const double* obstacles_array,
                                           const int* num_vertices_array,
                                           int num_time_steps,
                                           int num_obstacles,
                                           int max_vertices,
                                           int time_step_now) {
    if (traj.t.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    double cost_time = cost_terminal_time(15.0 - 0.1 * static_cast<double>(traj.t.size()));
    double cost_obstacle = cost_dist_obstacle(obstacles_array, num_vertices_array, num_time_steps, num_obstacles, max_vertices, traj, time_step_now);
    double cost_speed = cost_velocity_offset(traj.s_d, max_speed);
    double cost_accel = cost_acceleration(traj.s_dd) + cost_acceleration(traj.d_dd);
    double cost_jerk_val = cost_jerk(traj.s_ddd) + cost_jerk(traj.d_ddd);
    double cost_offset = cost_lane_center_offset(traj.d);

    return (cost_time + cost_obstacle + cost_speed + cost_accel + cost_jerk_val + cost_offset) /
           static_cast<double>(traj.t.size());
}

void CostFunction::calc_cost(std::vector<FrenetTrajectory>& fplist,
                             double target_speed,
                             const double* obstacles_array,
                             const int* num_vertices_array,
                             int num_time_steps,
                             int num_obstacles,
                             int max_vertices,
                             int time_step_now) {
    for (auto& traj : fplist) {
        traj.cost_final = cost_singleTrajectory(
            traj,
            target_speed,
            obstacles_array,
            num_vertices_array,
            num_time_steps,
            num_obstacles,
            max_vertices,
            time_step_now
        );
    }
}

double CostFunction::cost_total(const FrenetTrajectory& traj, double target_speed) {
    (void)target_speed;

    if (traj.t.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    double cost_time = cost_terminal_time(15.0 - 0.1 * static_cast<double>(traj.t.size()));
    double cost_obstacle = 0.0;
    double cost_speed = cost_velocity_offset(traj.s_d, max_speed);

    double cost_accel = cost_acceleration(traj.s_dd) + cost_acceleration(traj.d_dd);
    double cost_jerk_val = cost_jerk(traj.s_ddd) + cost_jerk(traj.d_ddd);
    double cost_offset = cost_lane_center_offset(traj.d);

    double total_cost = (cost_time + cost_obstacle + cost_speed + cost_accel + cost_jerk_val + cost_offset) /
                        static_cast<double>(traj.t.size());
    
    return total_cost;
}
