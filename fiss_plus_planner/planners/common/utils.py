"""
Collision detection utilities for trajectory planning with Numba JIT compilation
支持任意多边形（不限于四边形）

轨迹数据格式: [num_trajs, num_states, 3] 其中最后一维为 [x, y, yaw]
障碍物数据格式: list of Polygon objects (Shapely Polygon)
"""

from numba import njit, prange
import numpy as np
from typing import Tuple, List
import numba

def configure_numba_threads(n: int) -> None:
    if n and n > 0:
        numba.set_num_threads(n)


# ============================================================================
# 基础碰撞检测函数（JIT 编译）
# ============================================================================

@njit
def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    使用射线投射法检测点是否在多边形内
    
    Args:
        point: 点坐标，形状 (2,)
        polygon: 多边形顶点，形状 (n, 2) 其中 n 可以是任意数
    
    Returns:
        True 如果点在多边形内，否则 False
    """
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
    """
    检测两条线段是否相交（使用 CCW 方法）
    
    Args:
        p1, p2: 第一条线段的端点
        p3, p4: 第二条线段的端点
    
    Returns:
        True 如果相交，否则 False
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


@njit
def polygon_collision(
    poly1: np.ndarray,
    poly2: np.ndarray
) -> bool:
    """
    检测两个任意多边形是否碰撞
    
    Args:
        poly1: 多边形1顶点，形状 (n, 2)
        poly2: 多边形2顶点，形状 (m, 2)
    
    Returns:
        True 如果碰撞，否则 False
    """
    # 检查 poly1 的任何顶点是否在 poly2 内
    for i in range(poly1.shape[0]):
        if point_in_polygon(poly1[i], poly2):
            return True
    
    # 检查 poly2 的任何顶点是否在 poly1 内
    for i in range(poly2.shape[0]):
        if point_in_polygon(poly2[i], poly1):
            return True
    
    # 检查边是否相交
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
    vehicle_width: float
) -> np.ndarray:
    """
    计算车辆的多边形顶点（矩形，4个顶点）
    
    Args:
        x, y: 车辆中心位置
        yaw: 车辆偏航角
        vehicle_length: 车辆长度
        vehicle_width: 车辆宽度
    
    Returns:
        多边形顶点，形状 (4, 2)
    """
    poly = np.zeros((4, 2), dtype=np.float32)
    
    # 定义矩形角点相对于中心的偏移
    corner_offsets = np.array([
        [vehicle_length / 2, vehicle_width / 2],
        [vehicle_length / 2, -vehicle_width / 2],
        [-vehicle_length / 2, -vehicle_width / 2],
        [-vehicle_length / 2, vehicle_width / 2]
    ], dtype=np.float32)
    
    # 旋转矩阵
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # 计算旋转后的顶点
    for i in range(4):
        x_offset = corner_offsets[i, 0]
        y_offset = corner_offsets[i, 1]
        
        rotated_x = x_offset * cos_yaw - y_offset * sin_yaw
        rotated_y = x_offset * sin_yaw + y_offset * cos_yaw
        
        poly[i, 0] = rotated_x + x
        poly[i, 1] = rotated_y + y
    
    return poly


# ============================================================================
# 高性能多线程碰撞检测函数
# ============================================================================

@njit(parallel=True)
def check_trajectories_collision_parallel(
    trajectories: np.ndarray,           # shape (num_trajs, num_states, 3) - x, y, yaw
    obstacles_polygons: List,           # List of np.ndarray, 每个形状 (n, 2)
    vehicle_length: float,
    vehicle_width: float,
    check_resolution: int = 1
) -> Tuple[np.ndarray, int]:
    """
    并行检测多条轨迹与任意多边形障碍物的碰撞（多线程版本）
    
    直接在 frenet_optimal_planner 中使用
    
    Args:
        trajectories: 轨迹集合，形状 (num_trajs, num_states, 3)
                     每个轨迹包含 num_states 个状态点，每个点有 (x, y, yaw)
        obstacles_polygons: 障碍物多边形列表，每个元素形状 (n, 2)
        vehicle_length: 车辆长度
        vehicle_width: 车辆宽度
        check_resolution: 检查间隔（每隔 n 个状态检查一次）
    
    Returns:
        Tuple[np.ndarray, int]: 
        - collision_results: 碰撞结果数组，形状 (num_trajs,)，True 表示碰撞
        - total_checks: 执行的碰撞检查总数
    """
    num_trajs = trajectories.shape[0]
    num_states = trajectories.shape[1]
    num_obstacles = len(obstacles_polygons)
    
    collision_results = np.zeros(num_trajs, dtype=np.bool_)
    total_checks = 0
    
    # 并行处理每条轨迹
    for traj_idx in prange(num_trajs):
        trajectory = trajectories[traj_idx]  # shape (num_states, 3)
        
        # 按间隔检查轨迹上的每个状态点
        for state_idx in range(0, num_states, check_resolution):
            x = trajectory[state_idx, 0]
            y = trajectory[state_idx, 1]
            yaw = trajectory[state_idx, 2]
            
            # 计算车辆在此状态的多边形
            ego_poly = compute_vehicle_polygon(
                x, y, yaw, vehicle_length, vehicle_width
            )
            
            # 检查与每个障碍物的碰撞
            for obs_idx in range(num_obstacles):
                obstacle_poly = obstacles_polygons[obs_idx]
                total_checks += 1
                
                # 检测碰撞
                if polygon_collision(ego_poly, obstacle_poly):
                    collision_results[traj_idx] = True
                    break
    
    return collision_results, total_checks


@njit(parallel=True)
def check_trajectories_collision_parallel_static(
    trajectories: np.ndarray,           # shape (num_trajs, num_states, 3)
    traj_lengths: np.ndarray,           # shape (num_trajs,)
    obstacles_array: np.ndarray,        # shape (num_time_steps, num_obstacles, max_vertices, 2)
    num_vertices: np.ndarray,           # shape (num_time_steps, num_obstacles)
    vehicle_length: float,
    vehicle_width: float,
    check_resolution: int = 1
) -> Tuple[np.ndarray, int]:
    """
    并行检测多条轨迹与任意多边形障碍物的碰撞（时间序列数组版本）
    
    使用预分配的时间序列数组，避免动态内存分配
    
    Args:
        trajectories: 轨迹集合，形状 (num_trajs, num_states, 3)
        traj_lengths: 每条轨迹的实际长度，形状 (num_trajs,)
        obstacles_array: 预分配的障碍物数组，形状 (num_time_steps, num_obstacles, max_vertices, 2)
        num_vertices: 每个障碍物的实际顶点数，形状 (num_time_steps, num_obstacles)
        vehicle_length: 车辆长度
        vehicle_width: 车辆宽度
        check_resolution: 检查间隔
    
    Returns:
        Tuple[collision_results, total_checks]
    """
    num_trajs = trajectories.shape[0]
    num_states = trajectories.shape[1]
    num_time_steps = obstacles_array.shape[0]
    num_obstacles = obstacles_array.shape[1]
    
    collision_results = np.zeros(num_trajs, dtype=np.bool_)
    checks = np.zeros(num_trajs, dtype=np.int64)
    
    # 并行处理每条轨迹
    for traj_idx in prange(num_trajs):
        trajectory = trajectories[traj_idx]  # shape (num_states, 3)
        local_checks = 0
        
        traj_len = traj_lengths[traj_idx]
        max_steps = traj_len if traj_len < num_time_steps else num_time_steps
        
        # 按间隔检查轨迹上的每个状态点
        for state_idx in range(0, max_steps, check_resolution):
            x = trajectory[state_idx, 0]
            y = trajectory[state_idx, 1]
            yaw = trajectory[state_idx, 2]
            
            # 计算车辆在此状态的多边形
            ego_poly = compute_vehicle_polygon(
                x, y, yaw, vehicle_length, vehicle_width
            )
            
            # 检查与每个障碍物的碰撞
            for obs_idx in range(num_obstacles):
                # 只检查实际顶点数的部分
                actual_verts = num_vertices[state_idx, obs_idx]
                if actual_verts > 0:
                    obstacle_poly = obstacles_array[state_idx, obs_idx, :actual_verts, :]
                    local_checks += 1
                    
                    # 检测碰撞
                    if polygon_collision(ego_poly, obstacle_poly):
                        collision_results[traj_idx] = True
                        break
            
            if collision_results[traj_idx]:
                break
        
        checks[traj_idx] = local_checks
    
    total_checks = checks.sum()
    return collision_results, total_checks


# ============================================================================
# 用于 FrenetOptimalPlanner 的包装函数
# ============================================================================

def prepare_trajectory_array(fplist: list, return_lengths: bool = False):
    """
    将轨迹列表转换为 [num_trajs, num_states, 3] 格式的数组
    
    Args:
        fplist: FrenetTrajectory 对象列表
    
    Returns:
        trajectories: 形状 (num_trajs, num_states, 3) 的 NumPy 数组
                     其中最后一维为 [x, y, yaw]
        traj_lengths (optional): 形状 (num_trajs,) 的数组，记录每条轨迹的实际长度
    """
    num_trajs = len(fplist)
    if num_trajs == 0:
        empty_trajs = np.array([], dtype=np.float32).reshape(0, 0, 3)
        if return_lengths:
            return empty_trajs, np.array([], dtype=np.int32)
        return empty_trajs
    
    # 获取最大轨迹长度
    traj_lengths = np.array([len(traj.x) for traj in fplist], dtype=np.int32)
    max_length = int(traj_lengths.max()) if traj_lengths.size > 0 else 0
    
    # 创建输出数组
    trajectories = np.zeros((num_trajs, max_length, 3), dtype=np.float32)
    
    # 填充数据
    for i, traj in enumerate(fplist):
        traj_len = len(traj.x)
        trajectories[i, :traj_len, 0] = traj.x  # x 坐标
        trajectories[i, :traj_len, 1] = traj.y  # y 坐标
        trajectories[i, :traj_len, 2] = traj.yaw  # 偏航角
    
    if return_lengths:
        return trajectories, traj_lengths
    return trajectories


def prepare_obstacles_polygons_static(
    obstacles: list,
    time_step_now: int = 0,
    max_vertices: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为障碍物准备多边形顶点的静态数组版本（高性能）
    
    使用预分配的固定大小数组，避免动态 append
    
    Args:
        obstacles: CommonRoad 障碍物列表
        time_step_now: 当前时间步
        max_vertices: 每个多边形最多的顶点数（预分配大小）
    
    Returns:
        Tuple[obstacles_array, num_vertices]:
        - obstacles_array: 形状 (num_obstacles, max_vertices, 2) 的静态数组
        - num_vertices: 形状 (num_obstacles,) 的数组，记录每个障碍物的实际顶点数
    """
    num_obstacles = len(obstacles)
    
    if num_obstacles == 0:
        return np.array([], dtype=np.float32).reshape(0, max_vertices, 2), np.array([], dtype=np.int32)
    
    # 预分配静态数组
    obstacles_array = np.zeros((num_obstacles, max_vertices, 2), dtype=np.float32)
    num_vertices = np.zeros(num_obstacles, dtype=np.int32)
    
    # 填充数据
    for obs_idx, obstacle in enumerate(obstacles):
        state = obstacle.state_at_time(time_step_now)
        if state is not None:
            try:
                # 获取 obstacle 的 Shapely Polygon
                shapely_poly = obstacle.obstacle_shape.shapely_object
                
                # 提取顶点坐标
                coords = np.array(shapely_poly.exterior.coords[:-1], dtype=np.float32)
                
                # 限制顶点数不超过 max_vertices
                num_verts = min(len(coords), max_vertices)
                num_vertices[obs_idx] = num_verts
                
                # 平移和旋转到当前位置和方向
                obs_x = state.position[0]
                obs_y = state.position[1]
                obs_yaw = state.orientation
                
                # 旋转坐标
                cos_yaw = np.cos(obs_yaw)
                sin_yaw = np.sin(obs_yaw)
                
                # 直接写入到预分配的数组
                for i in range(num_verts):
                    dx = coords[i, 0]
                    dy = coords[i, 1]
                    obstacles_array[obs_idx, i, 0] = dx * cos_yaw - dy * sin_yaw + obs_x
                    obstacles_array[obs_idx, i, 1] = dx * sin_yaw + dy * cos_yaw + obs_y
            
            except Exception as e:
                print(f"Error processing obstacle {obs_idx}: {e}")
                num_vertices[obs_idx] = 0
                continue
    
    return obstacles_array, num_vertices


def prepare_obstacles_polygons_time_series(
    obstacles: list,
    num_time_steps: int,
    time_step_now: int = 0,
    max_vertices: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为障碍物准备多边形顶点的时间序列数组（支持动态障碍物）
    
    Args:
        obstacles: CommonRoad 障碍物列表
        num_time_steps: 需要覆盖的时间步长度
        time_step_now: 当前时间步
        max_vertices: 每个多边形最多的顶点数（预分配大小）
    
    Returns:
        Tuple[obstacles_array, num_vertices]:
        - obstacles_array: 形状 (num_time_steps, num_obstacles, max_vertices, 2)
        - num_vertices: 形状 (num_time_steps, num_obstacles)
    """
    num_obstacles = len(obstacles)
    if num_obstacles == 0 or num_time_steps <= 0:
        return (
            np.array([], dtype=np.float32).reshape(0, 0, max_vertices, 2),
            np.array([], dtype=np.int32).reshape(0, 0)
        )
    
    obstacles_array = np.zeros(
        (num_time_steps, num_obstacles, max_vertices, 2),
        dtype=np.float32
    )
    num_vertices = np.zeros((num_time_steps, num_obstacles), dtype=np.int32)
    
    for obs_idx, obstacle in enumerate(obstacles):
        try:
            shapely_poly = obstacle.obstacle_shape.shapely_object
            coords = np.array(shapely_poly.exterior.coords[:-1], dtype=np.float32)
            num_verts = min(len(coords), max_vertices)
        except Exception as e:
            print(f"Error processing obstacle {obs_idx}: {e}")
            continue
        
        # 对静态障碍物，使用初始状态填充所有时间步
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
    fplist: list,
    obstacles: list,
    vehicle_length: float,
    vehicle_width: float,
    time_step_now: int = 0,
    check_resolution: int = 1,
    max_vertices: int = None
) -> Tuple[np.ndarray, int]:
    """
    检测轨迹碰撞的主函数 - 直接在 frenet_optimal_planner.py 中调用
    
    使用多线程 Numba JIT 编译实现高性能碰撞检测
    支持任意多边形障碍物，使用时间序列数组处理动态障碍物
    
    Args:
        fplist: FrenetTrajectory 对象列表
        obstacles: CommonRoad 障碍物列表
        vehicle_length: 车辆长度
        vehicle_width: 车辆宽度
        time_step_now: 当前时间步
        check_resolution: 检查间隔（默认1表示检查所有点）
        max_vertices: 每个多边形预分配的最大顶点数（性能权衡参数）
                     如果为 None，将从 obstacles 中动态计算
    
    Returns:
        Tuple[collision_mask, num_checks]:
        - collision_mask: 布尔数组，形状 (num_trajs,)，True 表示碰撞
        - num_checks: 执行的碰撞检查总数
    
    Example:
        >>> fplist = [traj1, traj2, ...]
        >>> obstacles = [obs1, obs2, ...]
        >>> collision_mask, checks = check_trajectories_collision(
        ...     fplist, obstacles, 4.5, 1.8, time_step_now=0
        ... )
        >>> safe_trajs = [fplist[i] for i, collide in enumerate(collision_mask) if not collide]
    """
    # 将轨迹转换为 [num_trajs, num_states, 3] 格式
    trajectories, traj_lengths = prepare_trajectory_array(fplist, return_lengths=True)
    
    if trajectories.size == 0 or len(obstacles) == 0:
        return np.array([], dtype=np.bool_), 0
    if traj_lengths.size == 0 or traj_lengths.max() == 0:
        return np.zeros(len(fplist), dtype=np.bool_), 0
    
    # 如果 max_vertices 为 None，从 obstacles 中动态计算
    if max_vertices is None:
        max_vertices = 10  # 默认最小值
        try:
            for obstacle in obstacles:
                # 尝试获取 obstacle 的顶点数
                try:
                    shapely_poly = obstacle.obstacle_shape.shapely_object
                    coords = np.array(shapely_poly.exterior.coords[:-1], dtype=np.float32)
                    num_verts = len(coords)
                    if num_verts > max_vertices:
                        max_vertices = num_verts
                except Exception:
                    # 如果某个 obstacle 读取失败，继续处理下一个
                    continue
        except Exception as e:
            print(f"Warning: Failed to calculate max_vertices from obstacles: {e}")
            max_vertices = 10
    
    num_time_steps = int(traj_lengths.max())
    
    # 准备障碍物多边形时间序列（支持动态障碍物）
    obstacles_array, num_vertices = prepare_obstacles_polygons_time_series(
        obstacles,
        num_time_steps=num_time_steps,
        time_step_now=time_step_now,
        max_vertices=max_vertices
    )
    
    if obstacles_array.shape[0] == 0 or obstacles_array.shape[1] == 0:
        return np.zeros(len(fplist), dtype=np.bool_), 0
    
    # 调用多线程碰撞检测
    return check_trajectories_collision_parallel_static(
        trajectories,
        traj_lengths,
        obstacles_array,
        num_vertices,
        vehicle_length,
        vehicle_width,
        check_resolution
    )
