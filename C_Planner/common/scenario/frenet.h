#ifndef FRENET_H
#define FRENET_H

#include <vector>
#include <cmath>

struct SamplingParam {
    double d;     // lateral offset [m]
    double s_d;   // longitudinal speed [m/s]
    double t;     // time [s]
    
    // Default constructor
    SamplingParam() : d(0.0), s_d(0.0), t(0.0) {}
    
    SamplingParam(double d_, double s_d_, double t_)
        : d(d_), s_d(s_d_), t(t_) {}
    //add = operator for assignment
    SamplingParam& operator=(const SamplingParam& other) {
        d = other.d;
        s_d = other.s_d;
        t = other.t;
        return *this;
    }
};

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
    
    // End state for FISS+ planner
    struct EndState {
        double t = 0.0;
        double s = 0.0;
        double s_d = 0.0;
        double s_dd = 0.0;
        double d = 0.0;
        double d_d = 0.0;
        double d_dd = 0.0;
    } end_state;

    struct SamplingParam sampling_param; // sampling parameters (d, s_d, t)
    
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
    
    // Default constructor
    FrenetState() = default;
    
    // Constructor with all parameters
    FrenetState(double t_, double s_, double s_d_, double s_dd_, double s_ddd_,
                double d_, double d_d_, double d_dd_, double d_ddd_)
        : t(t_), s(s_), s_d(s_d_), s_dd(s_dd_), s_ddd(s_ddd_),
          d(d_), d_d(d_d_), d_dd(d_dd_), d_ddd(d_ddd_) {}
};

#endif // FRENET_H
