#ifndef PLAN_STATS_H
#define PLAN_STATS_H

// Per-stage processing time [ms]. Stage times are summed over worker threads (CPU
// time); total_ms is the wall-clock time of one plan() call.
struct TimingStats {
    double generation_ms = 0.0;
    double transform_ms = 0.0;
    double constraint_ms = 0.0;
    double collision_ms = 0.0;
    double cost_ms = 0.0;
    double total_ms = 0.0;

    TimingStats& operator+=(const TimingStats& other) {
        generation_ms += other.generation_ms;
        transform_ms += other.transform_ms;
        constraint_ms += other.constraint_ms;
        collision_ms += other.collision_ms;
        cost_ms += other.cost_ms;
        total_ms += other.total_ms;
        return *this;
    }
};

// Counters of one planning cycle. Every counter is incremented inside
// TrajectoryEvaluator, so all planners are counted in exactly the same way.
struct PlanStats {
    int num_search_iterations = 0;   // planner-specific outer iterations (FISS+, CEM, ...)
    int num_trajs_generated = 0;     // Frenet polynomials generated
    int num_global_transforms = 0;   // Frenet -> Cartesian conversions
    int num_rejected_transform = 0;  // fewer than two valid Cartesian points
    int num_constraint_checks = 0;
    int num_constraint_passed = 0;
    int num_rejected_dynamic = 0;    // over max speed or max acceleration
    int num_rejected_offroad = 0;    // left the drivable roadway
    int num_collision_checks = 0;
    int num_collision_free = 0;
    int num_cost_evaluations = 0;
    int num_FOP_intervention = 0;    // Number of times FOP was used for intervention (if applicable)
    TimingStats timing;

    int num_rejected_collision() const { return num_collision_checks - num_collision_free; }

    // Accumulate stats from another PlanStats
    PlanStats& operator+=(const PlanStats& other) {
        num_search_iterations += other.num_search_iterations;
        num_trajs_generated += other.num_trajs_generated;
        num_global_transforms += other.num_global_transforms;
        num_rejected_transform += other.num_rejected_transform;
        num_constraint_checks += other.num_constraint_checks;
        num_constraint_passed += other.num_constraint_passed;
        num_rejected_dynamic += other.num_rejected_dynamic;
        num_rejected_offroad += other.num_rejected_offroad;
        num_collision_checks += other.num_collision_checks;
        num_collision_free += other.num_collision_free;
        num_cost_evaluations += other.num_cost_evaluations;
        num_FOP_intervention += other.num_FOP_intervention;
        timing += other.timing;
        return *this;
    }
};

#endif // PLAN_STATS_H
