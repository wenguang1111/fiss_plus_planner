#include "cost_function.h"

CostFunction::CostFunction(const std::string& cost_type) {
    if (cost_type == "WX1") {
        w_T = 10.0;
        w_V = 1.0;
        w_A = 0.1;
        w_J = 0.1;
        w_D = 0.1;
        w_LC = 10.0;
    } else {
        // Default values
        w_T = 10.0;
        w_V = 1.0;
        w_A = 0.1;
        w_J = 0.1;
        w_D = 0.1;
        w_LC = 10.0;
    }
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

double CostFunction::cost_total(const FrenetTrajectory& traj, double target_speed) {
    // Cost for time: prefer shorter trajectories
    double cost_time = 10.0 - traj.t.back();
    
    // Cost for speed deviation
    double cost_speed = cost_velocity_offset(traj.s_d, target_speed);
    
    // Cost for accelerations
    double cost_accel = cost_acceleration(traj.s_dd) + cost_acceleration(traj.d_dd);
    
    // Cost for jerks
    double cost_jerk_val = cost_jerk(traj.s_ddd) + cost_jerk(traj.d_ddd);
    
    // Cost for lane offset
    double cost_offset = cost_lane_center_offset(traj.d);
    
    // Normalize by trajectory length
    double total_cost = (cost_time + cost_speed + cost_accel + cost_jerk_val + cost_offset) / traj.t.size();
    
    return total_cost;
}
