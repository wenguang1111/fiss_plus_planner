from numba import njit, prange
import numpy as np
from typing import Tuple, List
import numba

def configure_numba_threads(n: int) -> None:
    if n and n > 0:
        numba.set_num_threads(n)

@njit
def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = point[0], point[1]
    n = polygon.shape[0]
    inside = False
    
    p1x, p1y = polygon[0, 0], polygon[0, 1]
    for i in range(1, n + 1):
        p2x = polygon[i % n, 0]
        p2y = polygon[i % n, 1]
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside


@njit
def segments_intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray
) -> bool:
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


@njit
def aabb_collision(
    poly1: np.ndarray,
    poly2: np.ndarray
) -> bool:
    min_x1 = poly1[:, 0].min()
    max_x1 = poly1[:, 0].max()
    min_y1 = poly1[:, 1].min()
    max_y1 = poly1[:, 1].max()
    
    min_x2 = poly2[:, 0].min()
    max_x2 = poly2[:, 0].max()
    min_y2 = poly2[:, 1].min()
    max_y2 = poly2[:, 1].max()
    
    if max_x1 < min_x2 or max_x2 < min_x1:
        return False
    if max_y1 < min_y2 or max_y2 < min_y1:
        return False
    
    return True


@njit
def polygon_collision(
    poly1: np.ndarray,
    poly2: np.ndarray
) -> bool:
    if not aabb_collision(poly1, poly2):
        return False
    
    for i in range(poly1.shape[0]):
        if point_in_polygon(poly1[i], poly2):
            return True
    
    for i in range(poly2.shape[0]):
        if point_in_polygon(poly2[i], poly1):
            return True

    for i in range(poly1.shape[0]):
        for j in range(poly2.shape[0]):
            p1 = poly1[i]
            p2 = poly1[(i + 1) % poly1.shape[0]]
            p3 = poly2[j]
            p4 = poly2[(j + 1) % poly2.shape[0]]
            
            if segments_intersect(p1, p2, p3, p4):
                return True
    
    return False

@njit
def compute_vehicle_polygon(
    x: float,
    y: float,
    yaw: float,
    vehicle_length: float,
    vehicle_width: float,
    poly: np.ndarray
) -> np.ndarray:
    
    corner_offsets = np.array([
        [vehicle_length / 2, vehicle_width / 2],
        [vehicle_length / 2, -vehicle_width / 2],
        [-vehicle_length / 2, -vehicle_width / 2],
        [-vehicle_length / 2, vehicle_width / 2]
    ], dtype=np.float64)
    
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    for i in range(4):
        x_offset = corner_offsets[i, 0]
        y_offset = corner_offsets[i, 1]
        
        rotated_x = x_offset * cos_yaw - y_offset * sin_yaw
        rotated_y = x_offset * sin_yaw + y_offset * cos_yaw
        
        poly[i, 0] = rotated_x + x
        poly[i, 1] = rotated_y + y
    
    return poly

@njit(parallel=True)
def check_trajectories_collision_parallel_static(
    trajectories: np.ndarray,           # shape (num_trajs, num_states, 3)
    traj_lengths: np.ndarray,           # shape (num_trajs,)
    obstacles_array: np.ndarray,        # shape (num_time_steps, num_obstacles, max_vertices, 2)
    num_vertices: np.ndarray,           # shape (num_time_steps, num_obstacles)
    vehicle_length: float,
    vehicle_width: float,
    time_step_now: int = 0,
    check_resolution: int = 1
) -> np.ndarray:
    num_trajs = trajectories.shape[0]
    num_states = trajectories.shape[1]
    num_time_steps = obstacles_array.shape[0]
    num_obstacles = obstacles_array.shape[1]
    
    collision_results = np.zeros(num_trajs, dtype=np.bool_)
    checks = np.zeros(num_trajs, dtype=np.int64)
    
    for traj_idx in prange(num_trajs):
        trajectory = trajectories[traj_idx]  # shape (num_states, 3)
        local_checks = 0
        
        traj_len = traj_lengths[traj_idx]
        max_steps = min(traj_len, num_time_steps-time_step_now)
        ego_poly =  np.zeros((4, 2), dtype=np.float64)
        for state_idx in range(0, max_steps, check_resolution):
            x = trajectory[state_idx, 0]
            y = trajectory[state_idx, 1]
            yaw = trajectory[state_idx, 2]
    
            compute_vehicle_polygon(
                x, y, yaw, vehicle_length, vehicle_width, ego_poly
            )

            for obs_idx in range(num_obstacles):
                time_step = time_step_now + state_idx
                actual_verts = num_vertices[time_step, obs_idx]
                if actual_verts > 0:
                    obstacle_poly = obstacles_array[time_step, obs_idx, :actual_verts, :]
                    local_checks += 1
                    
                    if polygon_collision(ego_poly, obstacle_poly):
                        collision_results[traj_idx] = True
                        break
            
            if collision_results[traj_idx]:
                break
        
        checks[traj_idx] = local_checks
    
    return collision_results

def prepare_trajectory_array(fplist: list, return_lengths: bool = False):
    # return array as [num_trajs, num_states, 3]
    num_trajs = len(fplist)
    if num_trajs == 0:
        empty_trajs = np.array([], dtype=np.float64).reshape(0, 0, 3)
        if return_lengths:
            return empty_trajs, np.array([], dtype=np.int32)
        return empty_trajs
    
    traj_lengths = np.array([len(traj.x) for traj in fplist], dtype=np.int32)
    max_length = int(traj_lengths.max()) if traj_lengths.size > 0 else 0
    
    trajectories = np.zeros((num_trajs, max_length, 3), dtype=np.float64)
    
    for i, traj in enumerate(fplist):
        traj_len = len(traj.x)
        trajectories[i, :traj_len, 0] = traj.x
        trajectories[i, :traj_len, 1] = traj.y
        trajectories[i, :traj_len, 2] = traj.yaw 
    
    if return_lengths:
        return trajectories, traj_lengths
    return trajectories

def prepare_obstacles_polygons_time_series(
    obstacles: list,
    num_time_steps: int,
    time_step_now: int = 0,
    max_vertices: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    num_obstacles = len(obstacles)
    if num_obstacles == 0 or num_time_steps <= 0:
        return (
            np.array([], dtype=np.float64).reshape(0, 0, max_vertices, 2),
            np.array([], dtype=np.int32).reshape(0, 0)
        )
    
    obstacles_array = np.zeros(
        (num_time_steps, num_obstacles, max_vertices, 2),
        dtype=np.float64
    )
    num_vertices = np.zeros((num_time_steps, num_obstacles), dtype=np.int32)
    
    for obs_idx, obstacle in enumerate(obstacles):
        try:
            shapely_poly = obstacle.obstacle_shape.shapely_object
            coords = np.array(shapely_poly.exterior.coords[:-1], dtype=np.float64)
            num_verts = min(len(coords), max_vertices)
        except Exception as e:
            print(f"Error processing obstacle {obs_idx}: {e}")
            continue
        
        # for static obstacles, use initial state if no prediction
        default_state = None
        if getattr(obstacle, "prediction", None) is None:
            default_state = getattr(obstacle, "initial_state", None)
        
        for t in range(num_time_steps):
            state = obstacle.state_at_time(time_step_now + t)
            if state is None and default_state is not None:
                state = default_state
            if state is None:
                continue
            
            num_vertices[t, obs_idx] = num_verts
            obs_x = state.position[0]
            obs_y = state.position[1]
            obs_yaw = state.orientation if state.orientation is not None else 0.0
            
            cos_yaw = np.cos(obs_yaw)
            sin_yaw = np.sin(obs_yaw)
            
            for i in range(num_verts):
                dx = coords[i, 0]
                dy = coords[i, 1]
                obstacles_array[t, obs_idx, i, 0] = dx * cos_yaw - dy * sin_yaw + obs_x
                obstacles_array[t, obs_idx, i, 1] = dx * sin_yaw + dy * cos_yaw + obs_y
    
    return obstacles_array, num_vertices


def check_trajectories_collision(
    trajectories: np.ndarray,           # shape (num_trajs, num_states, 3) - preprocessed
    traj_lengths: np.ndarray,           # shape (num_trajs,) - trajectory lengths
    obstacles_array: np.ndarray,        # shape (num_time_steps, num_obstacles, max_vertices, 2) - preprocessed
    num_vertices: np.ndarray,           # shape (num_time_steps, num_obstacles) - actual vertices count
    vehicle_length: float,
    vehicle_width: float,
    time_step_now: int = 0,
    check_resolution: int = 1
) -> np.ndarray:
    # if trajectories.size == 0 or obstacles_array.shape[0] == 0:
    #     return np.array([], dtype=np.bool_), 0
    # if traj_lengths.size == 0 or traj_lengths.max() == 0:
    #     return np.zeros(trajectories.shape[0], dtype=np.bool_), 0

    collision_results = check_trajectories_collision_parallel_static(
        trajectories,
        traj_lengths,
        obstacles_array,
        num_vertices,
        vehicle_length,
        vehicle_width,
        time_step_now,
        check_resolution
    )
    return collision_results
