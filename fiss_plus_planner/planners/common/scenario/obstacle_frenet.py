import numpy as np
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
