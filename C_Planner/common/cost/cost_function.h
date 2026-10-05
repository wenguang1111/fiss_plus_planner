#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include "../scenario/frenet.h"
#include "../scenario/obstacles.h"
#include <vector>

// Cost definitions follow FISS+ (Sun et al., IROS 2023, doi:10.1109/IROS55552.2023.10341498):
//   planning cost  J      = J_T + J_V + J_A + J_J + J_LC (+ J_D)   Eq. (4), J_D from Eq. (7)
//   scenario cost  J_total = J_run + J_ter                          Eq. (7), (8)
// Eq. (8) as printed, J_ter = w_T (t_max - t_f), would reward slower arrival; the
// scenario cost uses J_ter = w_T * t_f, the time taken.
// Integrals are evaluated as Riemann sums over the trajectory samples, sum_k f_k * dt.
// No normalization is applied. The weights are the CommonRoad WX1 values (Table I:
// w_T=10, w_V=1, w_A=0.1, w_J=0.1, w_D=0.1, w_dist=1, w_LC=10) except for w_T, w_V, w_D and
// w_dist: on UsedForSelfEvaluation, WX1 made J_V and J_ter ~98% of J_total while J_D
// was ~0 (center distances are >= ~4.5 m, so exp(-1 * d) <= 0.011).
struct CostWeights {
    double w_T = 0.1;     // terminal time
    double w_V = 0.01;    // velocity tracking
    double w_A = 0.1;     // acceleration
    double w_J = 0.1;     // jerk
    double w_D = 1.0;     // distance to obstacles
    double w_dist = 0.1;  // decay of xi_i = exp(-w_dist * d_i)
    double w_LC = 10.0;   // lane-center offset
};

// Weighted cost terms. `obstacle` stays 0 when J_D is not part of the objective.
struct CostBreakdown {
    double time = 0.0;
    double velocity = 0.0;
    double acceleration = 0.0;
    double jerk = 0.0;
    double lane_center = 0.0;
    double obstacle = 0.0;

    double running() const { return velocity + acceleration + jerk + lane_center + obstacle; }
    double total() const { return time + running(); }
};

class CostFunction {
public:
    CostWeights weights;

    explicit CostFunction(const CostWeights& w = CostWeights()) : weights(w) {}

    // Eq. (4): every term that only needs the Frenet profile. J_T = w_T (t_max - t_f),
    // t_f being the trajectory's terminal time. Usable before the global transform.
    CostBreakdown frenet_terms(const FrenetTrajectory& traj, double v_des,
                               double t_max, double t_f, double dt) const;

    // J_D of Eq. (7): w_D * integral of max_i exp(-w_dist * d_i) over the obstacles in
    // front of the ego vehicle, d_i measured from the ego position to the obstacle
    // polygon center. Needs x / y / yaw. Trajectory sample k is matched with obstacle
    // time step time_step_now + k.
    double obstacle_term(const FrenetTrajectory& traj, const ObstacleView& obstacles,
                         int time_step_now, double dt) const;

    // Eq. (7) + (8) on the executed scenario trajectory, J_ter = w_T * t_f with t_f the
    // executed duration. v_des holds v_des(x(t)) for every executed sample (same length
    // as executed.s_d). The trajectory starts at obstacle time step 0.
    CostBreakdown scenario_cost(const FrenetTrajectory& executed, const std::vector<double>& v_des,
                                double dt, const ObstacleView& obstacles) const;
};

#endif // COST_FUNCTION_H
