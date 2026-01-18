#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include "../scenario/frenet.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

class CostFunction {
public:
    double w_T;   // weight for terminal time
    double w_V;   // weight for velocity offset
    double w_A;   // weight for acceleration
    double w_J;   // weight for jerk
    double w_D;   // weight for distance to obstacle
    double w_LC;  // weight for lane center offset
    
    CostFunction(const std::string& cost_type = "WX1");
    
    double cost_velocity_offset(const std::vector<double>& vels, double v_target);
    double cost_acceleration(const std::vector<double>& accels);
    double cost_jerk(const std::vector<double>& jerks);
    double cost_lane_center_offset(const std::vector<double>& offsets);
    double cost_total(const FrenetTrajectory& traj, double target_speed);
};

#endif // COST_FUNCTION_H
