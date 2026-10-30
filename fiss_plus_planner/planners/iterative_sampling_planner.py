import numpy as np
from numba import njit, prange
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.state import InitialState

from fiss_plus_planner.planners.common.scenario.frenet import FrenetState, FrenetTrajectory
from fiss_plus_planner.planners.common.utils import prepare_trajectory_array, compute_vehicle_polygon, polygon_collision
from fiss_plus_planner.planners.common.vehicle.vehicle import Vehicle
from fiss_plus_planner.planners.frenet_optimal_planner import FrenetOptimalPlanner, FrenetOptimalPlannerSettings, Stats


class IterativeSamplingSettings(FrenetOptimalPlannerSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5, num_iterations: int = 4):
        super().__init__(num_width, num_speed, num_t)
        # Same budget as the FOP grid: num_iterations * population = num_width * num_speed * num_t
        self.num_iterations = num_iterations
        self.population = max(1, (num_width * num_speed * num_t) // num_iterations)
        self.init_std = 0.3     # initial standard deviation in unit coordinates
        self.seed = 0           # the generator of a cycle is seeded with seed + time_step_now


@njit(parallel=True)
def first_collision_steps(trajectories, traj_lengths, obstacles_array, num_vertices,
                          vehicle_length, vehicle_width, time_step_now):
    """First colliding state of each trajectory (traj_length if none), same test as
    check_trajectories_collision_parallel_static."""
    num_trajs = trajectories.shape[0]
    num_time_steps = obstacles_array.shape[0]
    num_obstacles = obstacles_array.shape[1]
    first = traj_lengths.copy()
    for traj_idx in prange(num_trajs):
        ego_poly = np.zeros((4, 2), dtype=np.float64)
        max_steps = min(traj_lengths[traj_idx], num_time_steps - time_step_now)
        for state_idx in range(max_steps):
            compute_vehicle_polygon(trajectories[traj_idx, state_idx, 0], trajectories[traj_idx, state_idx, 1],
                                    trajectories[traj_idx, state_idx, 2], vehicle_length, vehicle_width, ego_poly)
            hit = False
            for obs_idx in range(num_obstacles):
                n = num_vertices[time_step_now + state_idx, obs_idx]
                if n > 0 and polygon_collision(ego_poly, obstacles_array[time_step_now + state_idx, obs_idx, :n, :]):
                    hit = True
                    break
            if hit:
                first[traj_idx] = state_idx
                break
    return first


class IterativeSamplingPlanner(FrenetOptimalPlanner):
    """FOP with an adaptive sampler: z = (d, s_d, t) is drawn from a diagonal Gaussian over the
    unit cube of the FOP search space and refined over a few iterations. Every sample goes through
    the FOP functions (calc_frenet_paths, calc_global_paths, check_constraints,
    check_collision_multithread, calc_cost). The proposal starts at the previous cycle's solution
    (warm start). Subclasses define the proposal update.

    Samples are ranked lexicographically: feasible before infeasible, feasible ones by cost,
    infeasible ones by the violation V = share of the horizon from the first violated state on,
    (N - k) / N, of the first failed check (off-road, then collision).
    """

    def __init__(self, planner_settings: IterativeSamplingSettings, ego_vehicle: Vehicle,
                 obstacles_array=None, obstacles_num_vertices=None):
        super().__init__(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices)
        self.prev_best = None  # SamplingParam of the previous cycle's plan

    def search_space(self) -> tuple:
        """Lower / upper bounds of (d, s_d, t), the region FrenetOptimalPlanner.get_samples covers."""
        sampling_width = self.road_width - self.vehicle.w
        lo = np.array([-sampling_width / 2, self.settings.lowest_speed, self.settings.min_t])
        hi = np.array([sampling_width / 2, self.settings.highest_speed, self.settings.max_t])
        return lo, hi

    def update_proposal(self, mean: np.ndarray, std: np.ndarray, units: np.ndarray,
                        feasible: np.ndarray, violation: np.ndarray, cost: np.ndarray) -> tuple:
        raise NotImplementedError

    def offroad_violation(self, traj: FrenetTrajectory) -> float:
        # Same condition as check_constraints, for a trajectory it rejected
        lat = np.asarray(traj.d, dtype=float)
        d_min, d_max = self.lateral_bounds_at(traj.s)
        bad = (lat < np.minimum(d_min, lat[0])) | (lat > np.maximum(d_max, lat[0]))
        return (len(lat) - np.argmax(bad)) / len(lat)

    def collision_violation(self, trajs: list, time_step_now: int) -> np.ndarray:
        trajectories, traj_lengths = prepare_trajectory_array(trajs, return_lengths=True)
        first = first_collision_steps(trajectories, traj_lengths, self.obstacles_array, self.obstacles_num_vertices,
                                      self.vehicle.l, self.vehicle.w, time_step_now)
        return (traj_lengths - first) / traj_lengths

    def evaluate(self, frenet_state: FrenetState, samples: np.ndarray, max_target_speed: float,
                 time_step_now: int) -> tuple:
        """Feasibility, violation V and cost of every sample, with the FOP pipeline."""
        n = len(samples)
        feasible = np.zeros(n, dtype=bool)
        violation = np.ones(n)  # samples dropped by calc_global_paths keep V = 1
        cost = np.full(n, np.inf)

        fplist = self.calc_frenet_paths(frenet_state, [tuple(z) for z in samples])
        index = {id(fp): i for i, fp in enumerate(fplist)}
        self.stats.num_trajs_generated += len(fplist)
        fplist = self.calc_global_paths(fplist)
        self.stats.num_trajs_validated += len(fplist)

        passed = self.check_constraints(fplist)
        passed_ids = {id(fp) for fp in passed}
        for fp in fplist:
            if id(fp) not in passed_ids:
                violation[index[id(fp)]] = self.offroad_violation(fp)
        self.stats.num_collison_checks += len(passed)

        safe = self.check_collision_multithread(passed, time_step_now)
        safe_ids = {id(fp) for fp in safe}
        collided = [fp for fp in passed if id(fp) not in safe_ids]
        if collided:
            for fp, v in zip(collided, self.collision_violation(collided, time_step_now)):
                violation[index[id(fp)]] = v

        safe = self.cost_function.calc_cost(safe, max_target_speed, self.obstacles_array,
                                            self.obstacles_num_vertices, time_step_now)
        for fp in safe:
            i = index[id(fp)]
            feasible[i] = True
            violation[i] = 0.0
            cost[i] = fp.cost_final
            if self.best_traj is None or fp.cost_final < self.best_traj.cost_final:
                self.best_traj = fp
        return feasible, violation, cost

    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list[Obstacle],
             time_step_now: int = 0, initial_state: InitialState = None) -> FrenetTrajectory:
        self.stats = Stats()
        self.settings.highest_speed = max_target_speed
        self.best_traj = None
        rng = np.random.default_rng(self.settings.seed + time_step_now)
        lo, hi = self.search_space()
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)

        mean = np.full(3, 0.5)
        if self.prev_best is not None:
            z = np.array([self.prev_best.d, self.prev_best.s_d, self.prev_best.t])
            mean = np.where(hi - lo > 1e-12, np.clip((z - lo) / span, 0.0, 1.0), 0.5)
        std = np.full(3, self.settings.init_std)

        num_lists = len(self.all_trajs)
        for it in range(self.settings.num_iterations):
            self.stats.num_iter += 1
            units = np.clip(mean + std * rng.standard_normal((self.settings.population, 3)), 0.0, 1.0)
            if it == 0:
                units[0] = mean  # the first sample is the mean itself
            feasible, violation, cost = self.evaluate(frenet_state, lo + units * (hi - lo),
                                                      max_target_speed, time_step_now)
            if it + 1 < self.settings.num_iterations:
                mean, std = self.update_proposal(mean, std, units, feasible, violation, cost)

        # One entry of all_trajs per planning cycle, as for FOP
        cycle_trajs = [fp for trajs in self.all_trajs[num_lists:] for fp in trajs]
        self.all_trajs[num_lists:] = [cycle_trajs]

        self.prev_best = None if self.best_traj is None else self.best_traj.sampling_param
        return self.best_traj

    @staticmethod
    def fit(units: np.ndarray, weights: np.ndarray) -> tuple:
        """Weighted mean and per-dimension standard deviation."""
        mean = np.average(units, axis=0, weights=weights)
        std = np.sqrt(np.average((units - mean) ** 2, axis=0, weights=weights))
        return mean, std
