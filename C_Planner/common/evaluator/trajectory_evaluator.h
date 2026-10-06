#ifndef TRAJECTORY_EVALUATOR_H
#define TRAJECTORY_EVALUATOR_H

#include "plan_stats.h"
#include "../cost/cost_function.h"
#include "../geometry/cubic_spline.h"
#include "../scenario/frenet.h"
#include "../scenario/obstacle_frenet.h"
#include "../scenario/obstacles.h"
#include "../scenario/road_profile.h"
#include "../scenario/vehicle_params.h"

// Gap to an obstacle ahead in the ego path if the ego vehicle drove on at its current
// speed and lateral offset (see Frenet_Planner::clearance_baseline).
struct ClearanceBaseline {
    int k0 = -1;              // first step the obstacle is ahead in that path; -1: never
    std::vector<double> gap;  // gap from the ego front to the obstacle rear per step [m],
                              // clamped at 0; NaN where the obstacle is not in that path
};

// Everything that defines one planning problem. All planners evaluating the same
// cycle build the same context, so they share generator, checks and objective.
struct PlanningContext {
    CubicSpline2D* spline = nullptr;        // reference line (Frenet frame), read-only use
    const RoadProfile* road = nullptr;      // optional; empty -> no boundary check
    ObstacleView obstacles;
    VehicleParams vehicle{};
    double dt = 0.1;              // trajectory time resolution [s]
    double v_des = 0.0;           // desired velocity of J_V [m/s]
    double t_max = 5.0;           // t_max of J_T, the largest sampled horizon [s]
    int time_step_now = 0;        // obstacle time step of trajectory sample 0
    bool check_boundary = true;
    bool check_obstacle = true;
    bool use_obstacle_cost = true;  // add J_D to the planning objective

    // Safe following distance to obstacles ahead in the ego path, see SettingParameters
    const ObstacleFrenetBounds* obstacle_frenet = nullptr;
    bool check_clearance = false;
    double clearance_time_gap = 2.0;
    double clearance_min_gap = 3.0;
    double clearance_lateral_margin = 0.2;
    double clearance_grace_time = 1.0;
    // Where the baseline gap is below the target, the requirement grows from the baseline
    // gap to the target within clearance_recovery_time after step k0, quadratically in
    // the elapsed time (the distance gained by braking grows with t^2).
    double clearance_recovery_time = 3.0;
    std::vector<ClearanceBaseline> clearance_baseline;  // one per obstacle
};

// Constraint violation V(tau): for each hard check, the share of the horizon from the
// first violating sample on, (N - k_first) / N, and 0 if the check passes. A trajectory
// that violates later is closer to feasible; V == 0 <=> feasible. Every check stops at
// its first violation, so V is exact even with EvalMode::kEarlyExit.
struct ConstraintViolation {
    double speed = 0.0;         // s_d above vehicle max speed
    double acceleration = 0.0;  // |s_dd| above vehicle max acceleration
    double road = 0.0;          // footprint beyond the road edge or outside the road profile
    double clearance = 0.0;     // closer than the safe following distance to an obstacle ahead
    double collision = 0.0;     // ego polygon overlaps an obstacle polygon
    double transform = 0.0;     // 1 when the trajectory has no Cartesian representation

    double total() const { return speed + acceleration + road + clearance + collision + transform; }
};

// First check a trajectory failed, in pipeline order.
enum class Rejection { kNone, kTransform, kDynamic, kOffroad, kClearance, kCollision };

struct EvaluationResult {
    bool feasible = false;
    Rejection rejection = Rejection::kNone;
    ConstraintViolation violation;
    bool has_cost = false;   // cost holds the planning objective J
    CostBreakdown cost;
};

// Lexicographic ranking (F, V, J): any feasible trajectory ranks before any infeasible
// one; feasible trajectories are ordered by J, infeasible ones by V.
bool ranks_before(const EvaluationResult& a, const EvaluationResult& b);

enum class EvalMode {
    // Stop at the first failed check (later checks are skipped); J only for feasible
    // trajectories. Used by every planner.
    kEarlyExit,
    // Run every check and compute J for every trajectory, e.g. for offline teacher data.
    // Costs extra checks, which are counted.
    kFullViolation,
};

// The single trajectory backend used by every planner: generation, Cartesian
// transform, dynamic/road constraints, polygon collision check and cost.
class TrajectoryEvaluator {
public:
    TrajectoryEvaluator(const PlanningContext& context, const CostFunction& cost_function)
        : ctx_(context), cost_(cost_function) {}

    // Quintic lateral / quartic longitudinal polynomial from start to z = (d, s_d, t).
    FrenetTrajectory generate(const FrenetState& start, const SamplingParam& z,
                              PlanStats& stats) const;

    // Eq. (4) without J_D. Needs only the Frenet profile, so search-based planners
    // (FISS+) can rank candidates before the expensive checks.
    CostBreakdown frenet_cost(const FrenetTrajectory& traj, PlanStats& stats) const;

    // Cartesian transform + constraints + collision. Fills x/y/yaw/... in traj.
    EvaluationResult check_feasibility(FrenetTrajectory& traj, PlanStats& stats,
                                       EvalMode mode = EvalMode::kEarlyExit) const;

    // check_feasibility + planning objective (J_D included if use_obstacle_cost).
    // Sets traj.cost_final when the cost is computed.
    EvaluationResult evaluate(FrenetTrajectory& traj, PlanStats& stats,
                              EvalMode mode = EvalMode::kEarlyExit) const;

    // Frenet -> Cartesian (x, y, yaw, ds, curvature). False if fewer than two points
    // lie on the reference line. Not counted in the stats (visualization use).
    bool to_global(FrenetTrajectory& traj) const;

    const PlanningContext& context() const { return ctx_; }

private:
    void check_dynamics(const FrenetTrajectory& traj, bool early_exit,
                        ConstraintViolation& v) const;
    double road_violation(const FrenetTrajectory& traj) const;
    double clearance_violation(const FrenetTrajectory& traj) const;

    const PlanningContext& ctx_;
    const CostFunction& cost_;
};

#endif // TRAJECTORY_EVALUATOR_H
