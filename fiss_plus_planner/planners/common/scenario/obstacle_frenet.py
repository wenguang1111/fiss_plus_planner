import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from fiss_plus_planner.planners.common.geometry.cubic_spline import CubicSpline2D


def obstacle_frenet_bounds(cubic_spline: CubicSpline2D, obstacles_array: np.ndarray,
                           obstacles_num_vertices: np.ndarray, ds: float = 0.25) -> np.ndarray:
    """Extent of every obstacle polygon in the Frenet frame of `cubic_spline`.

    Each vertex is projected onto the nearest point of the reference line sampled every `ds`
    meters (s along the line, l positive to the left). Returns an array of shape
    (num_time_steps, num_obstacles, 4) with (s_min, s_max, l_min, l_max), NaN where the obstacle
    is absent. Computed once per reference frame; the C++ planners use it to find obstacles
    ahead in the ego path (safe following distance).
    """
    num_time_steps, num_obstacles = obstacles_num_vertices.shape
    bounds = np.full((num_time_steps, num_obstacles, 4), np.nan)

    s_ref = np.arange(0.0, cubic_spline.s[-1], ds)
    ref_xy = np.array([cubic_spline.calc_position(s) for s in s_ref], dtype=float)
    ref_yaw = np.array([cubic_spline.calc_yaw(s) for s in s_ref], dtype=float)
    valid_ref = np.isfinite(ref_xy).all(axis=1) & np.isfinite(ref_yaw)
    s_ref, ref_xy, ref_yaw = s_ref[valid_ref], ref_xy[valid_ref], ref_yaw[valid_ref]
    if len(s_ref) == 0:
        return bounds

    # Vertices of all present obstacles, with their (time step, obstacle) index
    max_vertices = obstacles_array.shape[2]
    vertex_mask = np.arange(max_vertices)[None, None, :] < obstacles_num_vertices[:, :, None]
    t_idx, o_idx, _ = np.nonzero(vertex_mask)
    points = obstacles_array[vertex_mask]
    if len(points) == 0:
        return bounds

    _, nearest = cKDTree(ref_xy).query(points)
    rel = points - ref_xy[nearest]
    cos_yaw, sin_yaw = np.cos(ref_yaw[nearest]), np.sin(ref_yaw[nearest])
    s = s_ref[nearest] + rel[:, 0] * cos_yaw + rel[:, 1] * sin_yaw
    l = -rel[:, 0] * sin_yaw + rel[:, 1] * cos_yaw

    flat = t_idx * num_obstacles + o_idx
    out = bounds.reshape(-1, 4)
    for col, values, reduce in ((0, s, np.fmin), (1, s, np.fmax), (2, l, np.fmin), (3, l, np.fmax)):
        column = np.full(num_time_steps * num_obstacles, np.nan)
        reduce.at(column, flat, values)
        out[:, col] = column
    return bounds


def clearance_baseline(bounds: np.ndarray, s0: float, d0: float, v0: float, time_step_now: int, horizon: int,
                       dt: float, half_width: float, half_length: float) -> tuple:
    """Gap to every obstacle ahead in the ego path if the ego vehicle drove on at its current speed and lateral
    offset (Frenet_Planner::clearance_baseline of the C++ planners on the Experiment branch).

    Returns k0 (num_obstacles,), the first step the obstacle is ahead in that path (-1: never), and gap
    (num_obstacles, horizon + 1), the gap from the ego front to the obstacle rear, clamped at 0, NaN where the
    obstacle is not in that path.
    """
    num_time_steps, num_obstacles = bounds.shape[:2]
    k0 = np.full(num_obstacles, -1, dtype=np.int64)
    gap = np.full((num_obstacles, horizon + 1), np.nan)
    num_steps = max(0, min(horizon + 1, num_time_steps - time_step_now))
    if num_steps == 0 or num_obstacles == 0:
        return k0, gap
    k = np.arange(num_steps)
    b = bounds[time_step_now:time_step_now + num_steps]            # (k, obstacle, 4)
    ego_s = s0 + max(v0, 0.0) * k * dt
    with np.errstate(invalid="ignore"):
        ahead = (~np.isnan(b[..., 0]) & (b[..., 1] > ego_s[:, None])
                 & ~(b[..., 3] < d0 - half_width) & ~(b[..., 2] > d0 + half_width))
    g = np.maximum(0.0, b[..., 0] - (ego_s[:, None] + half_length))
    gap[:, :num_steps] = np.where(ahead, g, np.nan).T
    met = ahead.any(axis=0)
    k0[met] = ahead.argmax(axis=0)[met]
    return k0, gap


@njit(parallel=True, cache=True)
def clearance_first_violation(s, d, s_d, lengths, bounds, k0, gap, time_step_now, first_step, half_width,
                              half_length, min_gap, time_gap, dt, recovery_time):
    """First step of every trajectory closer than the safe following distance to an obstacle ahead in its path
    (lengths[i] if none); TrajectoryEvaluator::clearance_violation of the C++ planners on the Experiment branch.

    s, d, s_d: (num_trajs, max_len) Frenet states. From first_step on, the gap from the ego front to the rear of an
    obstacle ahead (rear beyond the ego center, lateral extent overlapping the ego +- half_width) must reach
    min_gap + time_gap * v. Where the baseline gap (gap, ego driving on at its current speed) is below that, the
    requirement grows from the baseline gap to the target within recovery_time after step k0, quadratically.
    """
    num_trajs = s.shape[0]
    num_time_steps, num_obstacles = bounds.shape[0], bounds.shape[1]
    first = lengths.copy()
    for i in prange(num_trajs):
        n = lengths[i]
        done = False
        for k in range(max(first_step, 0), n):
            t_idx = time_step_now + k
            if t_idx >= num_time_steps or done:
                break
            target = min_gap + time_gap * max(s_d[i, k], 0.0)
            for j in range(num_obstacles):
                s_min, s_max, l_min, l_max = bounds[t_idx, j, 0], bounds[t_idx, j, 1], bounds[t_idx, j, 2], bounds[t_idx, j, 3]
                if np.isnan(s_min) or s_max <= s[i, k]:
                    continue  # absent or behind
                if l_max < d[i, k] - half_width or l_min > d[i, k] + half_width:
                    continue  # not in the path
                required = target
                coast_gap = gap[j, k] if k < gap.shape[1] else np.nan
                if k0[j] >= 0 and not np.isnan(coast_gap) and coast_gap < target:
                    progress = min(1.0, (k - k0[j]) * dt / recovery_time) if recovery_time > 0.0 else 1.0
                    required = coast_gap + (target - coast_gap) * progress * progress
                if s_min - (s[i, k] + half_length) < required:
                    first[i] = k
                    done = True
                    break
    return first
