#ifndef FRENET_H
#define FRENET_H

#include <vector>
#include <cmath>

struct FrenetTrajectory {
    // Trajectory indices and metadata
    int idx[3] = {-1, -1, -1};          // trajectory id
    int lane_id = -1;                   // lane id
    
    // Status flags
    bool is_generated = false;
    bool is_searched = false;
    bool constraint_passed = false;
    bool collision_passed = false;
    
    // Cost values
    double cost_fix = 0.0;
    double cost_dyn = 0.0;
    double cost_heu = 0.0;
    double cost_est = 0.0;
    double cost_final = 0.0;
    
    // Frenet frame coordinates
    std::vector<double> t;      // time [s]
    std::vector<double> s;      // longitudinal position [m]
    std::vector<double> s_d;    // longitudinal velocity [m/s]
    std::vector<double> s_dd;   // longitudinal acceleration [m/ss]
    std::vector<double> s_ddd;  // longitudinal jerk [m/sss]
    std::vector<double> d;      // lateral position [m]
    std::vector<double> d_d;    // lateral velocity [m/s]
    std::vector<double> d_dd;   // lateral acceleration [m/ss]
    std::vector<double> d_ddd;  // lateral jerk [m/sss]
    
    // World frame coordinates
    std::vector<double> x;      // x position [m]
    std::vector<double> y;      // y position [m]
    std::vector<double> yaw;    // yaw angle [rad]
    std::vector<double> ds;     // distance increment [m]
    std::vector<double> c;      // curvature [1/m]
    std::vector<double> c_d;    // curvature derivative [1/m/s]
    std::vector<double> c_dd;   // curvature second derivative [1/m/ss]
    
    // Comparison operators for cost-based sorting
    bool operator<(const FrenetTrajectory& other) const {
        return cost_final < other.cost_final;
    }
    
    bool operator>(const FrenetTrajectory& other) const {
        return cost_final > other.cost_final;
    }
    
    bool operator==(const FrenetTrajectory& other) const {
        return cost_final == other.cost_final;
    }
};

struct FrenetState {
    double t = 0.0;         // time [s]
    double s = 0.0;         // longitudinal position [m]
    double s_d = 0.0;       // longitudinal velocity [m/s]
    double s_dd = 0.0;      // longitudinal acceleration [m/ss]
    double s_ddd = 0.0;     // longitudinal jerk [m/sss]
    double d = 0.0;         // lateral position [m]
    double d_d = 0.0;       // lateral velocity [m/s]
    double d_dd = 0.0;      // lateral acceleration [m/ss]
    double d_ddd = 0.0;     // lateral jerk [m/sss]
};

#endif // FRENET_H
