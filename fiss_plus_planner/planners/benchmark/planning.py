import os
import signal
import time
import csv
import fcntl
import json
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from PIL import Image
from omegaconf import DictConfig
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
import pandas as pd
from matplotlib import font_manager
from shapely import affinity
from shapely.geometry import Polygon as ShapelyPolygon

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.solution import VehicleType
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.scenario.state import InitialState
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping

from fiss_plus_planner.planners.common.scenario.frenet import FrenetState, State, FrenetTrajectory
from fiss_plus_planner.planners.common.vehicle.vehicle import Vehicle
from fiss_plus_planner.planners.commonroad_interface.global_planner import GlobalPlanner, extend_centerline
from fiss_plus_planner.planners.fiss_planner import FissPlanner, FissPlannerSettings
from fiss_plus_planner.planners.fiss_plus_planner import FissPlusPlanner, FissPlusPlannerSettings
from fiss_plus_planner.planners.fop_plus_planner import FopPlusPlanner
from fiss_plus_planner.planners.frenet_optimal_planner import FrenetOptimalPlanner, FrenetOptimalPlannerSettings, Stats
from fiss_plus_planner.planners.sparse_planner import SparsePlannerSettings, SparsePlanner
from fiss_plus_planner.planners.sparse_planner_cpp import SparsePlannerSettings_CPP, SparsePlanner_CPP
from fiss_plus_planner.planners.FOP_cpp_wrapper import FOP_CPP_Wrapper
from fiss_plus_planner.planners import FOP_cpp_wrapper as fop_cpp
from fiss_plus_planner.planners.fiss_plus_cpp_wrapper import FissPlusCppWrapper
from fiss_plus_planner.planners.CEM_cpp_wrapper import CEM_CPP_Wrapper
from fiss_plus_planner.planners.MPPI_cpp_wrapper import MPPI_CPP_Wrapper
from fiss_plus_planner.SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from fiss_plus_planner.SMP.motion_planner.motion_planner import MotionPlanner, MotionPlannerType
from fiss_plus_planner.SMP.motion_planner.utility import create_trajectory_from_list_states
from fiss_plus_planner.planners.common.utils import configure_numba_threads, transform_points_to_ego
from fiss_plus_planner.planners.sparse_planning.scenario_drawer import ScenarioDrawer
from fiss_plus_planner.planners.sparse_planner_optimized import SparsePlannerOptimizedSettings, SparsePlannerOptimized
from fiss_plus_planner.planners.sparse_planner_fop import SparsePlannerFOPSettings, SparsePlannerFOP
from fiss_plus_planner.planners.sparse_planner_world_model import SparsePlannerWorldModelSettings, SparsePlannerWorldModel

# === IEEE-like font family and sizes (10pt doc) ===
S = {
    "normalsize": 18,      # body text
    "small": 16,            # axis labels / lane labels
    "footnotesize": 14,     # tick labels / legend
    "large": 22,           # figure title
}
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",                   # Times-like math
    "axes.titlesize": S["large"],
    "axes.labelsize": S["large"],
    "xtick.labelsize": S["large"],
    "ytick.labelsize": S["large"],
    "legend.fontsize": S["large"],
    "svg.fonttype": "none",                       # keep text as text in SVG
})

# Pick an available Times-like font so figures stay consistent on systems
# without proprietary Times faces.
_FONT_CANDIDATES = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "DejaVu Serif",
    "STIXGeneral",
]


def _select_font_property():
    for family in _FONT_CANDIDATES:
        prop = font_manager.FontProperties(family=family)
        try:
            font_manager.findfont(prop, fallback_to_default=False)
        except ValueError:
            continue
        return prop
    return font_manager.FontProperties(family="serif")


_BASE_FONT = _select_font_property()


def font_prop(size_key: str) -> font_manager.FontProperties:
    prop = _BASE_FONT.copy()
    prop.set_size(S[size_key])
    return prop
##-------------------------------------------------------------------------------------------------

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


# [m/s] v_des when neither a goal velocity nor a speed-limit sign on the route applies
DEFAULT_SPEED_LIMIT = 14.0


def apply_settings(planner_settings, section_cfg: dict = None):
    """Overrides planner settings with a config section (SAFETY, MOTION_MODEL)."""
    for name, value in (section_cfg or {}).items():
        if not hasattr(planner_settings, name):
            raise KeyError(f"unknown planner setting: {name}")
        setattr(planner_settings, name, value)


def route_lanelet_index(position, global_plan, lanelet_network):
    """Index of the furthest route lanelet containing `position`, None when off the route."""
    ids = lanelet_network.find_lanelet_by_position([np.asarray(position, dtype=float)])[0]
    route_ids = [lanelet.lanelet_id for lanelet in global_plan.lanelets]
    indices = [route_ids.index(i) for i in ids if i in route_ids]
    return max(indices) if indices else None


def rear_end_threat(obstacles_array: np.ndarray, obstacles_num_vertices: np.ndarray, time_step: int,
                    state: InitialState, vehicle: Vehicle, horizon_steps: int) -> bool:
    """True when a vehicle now behind the ego in its lane drives into the ego footprint within
    `horizon_steps`, even if the ego stands still.

    The recorded traffic does not react to the ego, so such a follower makes every candidate
    infeasible; the ego cannot avoid it by braking (nuPlan's at-fault collision metric likewise does
    not count rear collisions into a stopped or slower ego against the ego). Used to label planning
    failures caused by a follower.
    """
    heading = np.array([np.cos(state.orientation), np.sin(state.orientation)])
    normal = np.array([-heading[1], heading[0]])
    corners = [(vehicle.l / 2, vehicle.w / 2), (vehicle.l / 2, -vehicle.w / 2),
               (-vehicle.l / 2, -vehicle.w / 2), (-vehicle.l / 2, vehicle.w / 2)]
    ego = ShapelyPolygon([state.position + a * heading + b * normal for a, b in corners])
    last_step = min(time_step + horizon_steps, obstacles_num_vertices.shape[0] - 1)
    for j in range(obstacles_num_vertices.shape[1]):
        n = obstacles_num_vertices[time_step, j]
        if n == 0:
            continue
        rel = obstacles_array[time_step, j, :n].mean(axis=0) - state.position
        if rel @ heading > -vehicle.l / 2 or abs(rel @ normal) > vehicle.w:
            continue  # not behind the ego in its lane
        for k in range(time_step + 1, last_step + 1):
            n = obstacles_num_vertices[k, j]
            if n > 0 and ShapelyPolygon(obstacles_array[k, j, :n]).intersects(ego):
                return True
    return False


def evaluate_scenario_cost(executed: FrenetTrajectory, v_des: list, dt: float,
                           obstacles_array: np.ndarray, obstacles_num_vertices: np.ndarray) -> Tuple[float, dict]:
    """Scenario-level cost J_total = J_run + J_ter (FISS+ Eq. 7, 8) of the executed trajectory.

    Every planner is scored by the same C++ implementation, independent of the objective
    it used while planning.
    """
    if not fop_cpp.CPP_MODULE_AVAILABLE:
        print("Warning: frenet_planner_cpp not available, scenario cost not computed")
        return float('nan'), {}
    traj = fop_cpp.frenet_planner_cpp.FrenetTrajectory()
    for name in ('t', 's', 's_d', 's_dd', 's_ddd', 'd', 'd_d', 'd_dd', 'd_ddd', 'x', 'y', 'yaw'):
        setattr(traj, name, [float(v) for v in getattr(executed, name)])
    cost = fop_cpp.frenet_planner_cpp.scenario_cost(
        traj, [float(v) for v in v_des], dt,
        np.ascontiguousarray(obstacles_array, dtype=np.float64),
        np.ascontiguousarray(obstacles_num_vertices, dtype=np.int32))
    terms = {name: getattr(cost, name)
             for name in ('time', 'velocity', 'acceleration', 'jerk', 'lane_center', 'obstacle')}
    terms['running'] = cost.running()
    return cost.total(), terms


def frenet_optimal_planning(scenario: Scenario, planning_problem: PlanningProblem, vehicle_params: DictConfig, method: str, num_samples: tuple, 
                            input_dir: str, file: str, output_dir: str, number_threads: int, runtime_measurement: bool, collect_data_for_ml: bool,
                            sampler_cfg: dict = None, safety_cfg: dict = None, motion_model_cfg: dict = None
                            ) -> Tuple[bool, Trajectory, float, list, Stats, list, list]:
    """sampler_cfg: settings of the iterative sampling planners, {'CEM': {...}, 'MPPI': {...}}.
    safety_cfg / motion_model_cfg: SAFETY / MOTION_MODEL config sections of the C++ planners."""
    sampler_cfg = sampler_cfg or {}
    # Plan a global route
    global_planner = GlobalPlanner()
    try:
        global_plan = global_planner.plan_global_route(scenario, planning_problem)
    except ValueError as e:
        print(f"    Failed to plan global route: {e}")
        return None, None, None, None, None, None, None
    
    ego_lane_pts = global_plan.concat_centerline

    # Goal
    goal_region = planning_problem.goal
    
    # Check if goal state list is available
    has_goal_state = goal_region.state_list is not None and len(goal_region.state_list) > 0
    
    # Check if goal position info is available
    goal_position_available = (
        goal_region.lanelets_of_goal_position is not None and 
        len(goal_region.lanelets_of_goal_position) > 0
    ) or (has_goal_state and goal_region.state_list[0].has_value("position"))

    # v_des of J_V: the goal velocity bound if given, otherwise the speed limit of the route
    # lanelet the ego is on (updated every cycle below), otherwise DEFAULT_SPEED_LIMIT
    if has_goal_state and goal_region.state_list[0].has_value("velocity"):
        goal_speed = goal_region.state_list[0].velocity.end
    else:
        goal_speed = None
    desired_speed = goal_speed if goal_speed is not None else DEFAULT_SPEED_LIMIT
    desired_speed_list = []  # v_des(x(t)) of every executed state, for the scenario cost
    
    # Get goal lanelet and center position
    if goal_region.lanelets_of_goal_position is not None and len(goal_region.lanelets_of_goal_position) > 0:
        goal_lanelet_idx = goal_region.lanelets_of_goal_position[0][0]
        goal_lanelet = scenario.lanelet_network.find_lanelet_by_id(goal_lanelet_idx)
        center_vertices = goal_lanelet.center_vertices
        mid_idx = int((center_vertices.shape[0] - 1) / 2)
        goal_center = center_vertices[mid_idx]
    else:
        # Fallback: use goal position from goal state if available
        if has_goal_state and goal_region.state_list[0].has_value("position"):
            goal_center = goal_region.state_list[0].position.center
        else:
            # Use the end of the reference path as goal
            goal_center = ego_lane_pts[-1]

    stats = Stats()
    # Obstacle lists
    obstacles_static = scenario.static_obstacles
    obstacles_dynamic = scenario.dynamic_obstacles
    obstacles_all = obstacles_static + obstacles_dynamic

    obstacle_positions = []
    obstacles_final_time_step = []
    # obstacles_final_time_step = [obs.prediction.final_time_step for obs in scenario.dynamic_obstacles]
    for obs in scenario.dynamic_obstacles:
        if obs.prediction is not None:
            obstacles_final_time_step.append(obs.prediction.final_time_step)
        else:
            stats.success = False
            goal_reached = False
            return goal_reached, None, None, None, stats, None, None
    if len(obstacles_final_time_step) == 0:
        stats.success = False
        goal_reached = False
        return goal_reached, None, None, None, stats, None, None
    final_time_step = max(obstacles_final_time_step)

    for t_step in range(final_time_step):
        frame_positions = []
        for obstacle in obstacles_all:
            if obstacle.state_at_time(t_step) is not None:
                frame_positions.append(obstacle.state_at_time(t_step).position)
        obstacle_positions.append(frame_positions)

    # Initialize local planner
    vehicle = Vehicle(vehicle_params)
    num_width, num_speed, num_t = num_samples

    # Prepare obstacles data once before creating planner
    max_vertices = 10  # default minimal value
    try:
        for obstacle in obstacles_all:
            try:
                shapely_poly = obstacle.obstacle_shape.shapely_object
                coords = np.array(shapely_poly.exterior.coords[:-1], dtype=np.float64)
                num_verts = len(coords)
                if num_verts > max_vertices:
                    max_vertices = num_verts
            except Exception:
                continue
    except Exception as e:
        print(f"Warning: Failed to calculate max_vertices from obstacles: {e}")
        max_vertices = 10
    
    obstacles_array, obstacles_num_vertices = prepare_obstacles_polygons_time_series(
        obstacles_all,
        num_time_steps=final_time_step,
        time_step_now=0,
        max_vertices=max_vertices
    )

    if method == 'FOP':
        planner_settings = FrenetOptimalPlannerSettings(
            num_width, num_speed, num_t)
        planner = FrenetOptimalPlanner(planner_settings, vehicle, obstacles_array, obstacles_num_vertices)
        use_cpp_planner = False  # Python FOP planner
    elif method == 'FOP+':
        planner_settings = FrenetOptimalPlannerSettings(
            num_width, num_speed, num_t)
        planner = FopPlusPlanner(planner_settings, vehicle, obstacles_array, obstacles_num_vertices)
        use_cpp_planner = False
    elif method == 'FISS':
        planner_settings = FissPlannerSettings(num_width, num_speed, num_t)
        planner = FissPlanner(planner_settings, vehicle, obstacles_array, obstacles_num_vertices)
        use_cpp_planner = False
    elif method == 'FISS+':
        planner_settings = FissPlusPlannerSettings(num_width, num_speed, num_t)
        planner = FissPlusPlanner(planner_settings, vehicle, obstacles_array, obstacles_num_vertices)
        use_cpp_planner = False
    elif method == 'Sparse':
        planner_settings = SparsePlannerSettings(num_width, num_speed, num_t, input_dir, file)
        planner = SparsePlanner(planner_settings, vehicle, obstacles_array, obstacles_num_vertices)
        use_cpp_planner = False
    elif method == 'Sparse_Optimized':
        planner_settings = SparsePlannerOptimizedSettings(num_width, num_speed, num_t, input_dir, file)
        planner = SparsePlannerOptimized(planner_settings, vehicle, obstacles_array, obstacles_num_vertices)
        use_cpp_planner = False
    elif method == 'Sparse_FOP':
        planner_settings = SparsePlannerFOPSettings(num_width, num_speed, num_t, input_dir, file)
        planner = SparsePlannerFOP(planner_settings, vehicle, obstacles_array, obstacles_num_vertices)
        use_cpp_planner = False
    elif method == 'Sparse_WorldModel':
        planner_settings = SparsePlannerWorldModelSettings(num_width, num_speed, num_t, input_dir, file)
        planner = SparsePlannerWorldModel(planner_settings, vehicle, obstacles_array, obstacles_num_vertices)
        use_cpp_planner = False
    elif method == 'FOP_CPP':
        # Use C++ Frenet Optimal Planner with pybind11
        planner_settings = FrenetOptimalPlannerSettings(num_width, num_speed, num_t)
        apply_settings(planner_settings, safety_cfg)
        apply_settings(planner_settings, motion_model_cfg)
        planner = FOP_CPP_Wrapper(planner_settings, vehicle, obstacles_array, obstacles_num_vertices, number_threads, runtime_measurement)
        use_cpp_planner = True  # Check if C++ planner was successfully initialized
        # planner.recordObstaclesForDebug("python_obstacle.csv")
    elif method == 'FISS+_CPP':
        # Use C++ FISS+ Planner with pybind11
        planner_settings = FissPlusPlannerSettings(num_width, num_speed, num_t)
        apply_settings(planner_settings, safety_cfg)
        apply_settings(planner_settings, motion_model_cfg)
        planner = FissPlusCppWrapper(planner_settings, vehicle, obstacles_array, obstacles_num_vertices, number_threads, runtime_measurement)
        use_cpp_planner = True
    elif method == 'CEM_CPP':
        planner_settings = FrenetOptimalPlannerSettings(num_width, num_speed, num_t)
        apply_settings(planner_settings, safety_cfg)
        apply_settings(planner_settings, motion_model_cfg)
        cem_cfg = dict(sampler_cfg.get('CEM') or {})
        if collect_data_for_ml:  # every candidate with full constraint labels, see save_cem_data
            cem_cfg['record_candidates'] = True
        planner = CEM_CPP_Wrapper(planner_settings, vehicle, obstacles_array, obstacles_num_vertices, number_threads,
                                  runtime_measurement, cem_cfg=cem_cfg)
        use_cpp_planner = True
    elif method == 'MPPI_CPP':
        planner_settings = FrenetOptimalPlannerSettings(num_width, num_speed, num_t)
        apply_settings(planner_settings, safety_cfg)
        apply_settings(planner_settings, motion_model_cfg)
        planner = MPPI_CPP_Wrapper(planner_settings, vehicle, obstacles_array, obstacles_num_vertices, number_threads,
                                   runtime_measurement, mppi_cfg=sampler_cfg.get('MPPI'))
        use_cpp_planner = True
    elif method == 'Sparse_CPP':
        # Use C++ Sparse Planner with pybind11
        planner_settings = SparsePlannerSettings_CPP(num_width, num_speed, num_t, input_dir, file)
        planner = SparsePlanner_CPP(planner_settings, vehicle, obstacles_array, obstacles_num_vertices)
        use_cpp_planner = True
    else:
        print("ERROR: Planning method entered is not recognized!")
        raise ValueError

    # The map ends with the scenario: continue the reference straight beyond the route end, so the
    # trajectories near the end do not run off the reference line; the run stops at the route end.
    num_route_pts = len(ego_lane_pts)
    ego_lane_pts = extend_centerline(ego_lane_pts, vehicle.max_speed * planner_settings.max_t)
    csp_ego, ref_ego_lane_pts = planner.generate_frenet_frame(ego_lane_pts)
    route_end_s = csp_ego.s[num_route_pts - 1]

    # Initial state
    initial_state = planning_problem.initial_state
    start_state = State(t=0.0, x=initial_state.position[0], y=initial_state.position[1],
                        yaw=initial_state.orientation, v=initial_state.velocity, a=initial_state.acceleration)
    current_frenet_state = FrenetState()
    current_frenet_state.from_state(start_state, ref_ego_lane_pts)

    # Start planning in simulation (matplotlib)
    show_animation = False
    area = 20.0  # animation area length [m]

    processing_time = 0
    num_cycles = 0
    state_list = []
    frenet_state_list = []
    global_state_list = []
    global_coordination_state_list = []

    time_list = []

    goal_reached = False
    next_state = initial_state
    best_trajs_all_time_steps = []

    # ML data (Collect_Data_For_ML): the planner works in the global frame (CEM keeps its warm
    # start); planned and reference paths are stored in the ego frame of each cycle (ego at the
    # origin, yaw 0), a rigid transform of the global result.
    collect_ml_data = collect_data_for_ml and method in ('FOP_CPP', 'CEM_CPP')
    cem_cycles = []  # CEM_CPP: context and candidate records of every planned cycle
    all_trajs_accumulated = []
    reference_path_lookahead_m = ScenarioDrawer.VIEW_SIZE_DEFAULT / 2.0
    optimal_path_x_local_list = []
    optimal_path_y_local_list = []
    ref_path_x_local_list = []
    ref_path_y_local_list = []

    for i in range(final_time_step):
        num_cycles += 1

        inital_state = InitialState(
            time_step=i,
            position=next_state.position,
            orientation=next_state.orientation,
            velocity=next_state.velocity,
            acceleration=next_state.acceleration,
            yaw_rate=next_state.yaw_rate
        )
        # global_coordination_state_list.append(inital_state)
        global_state_list.append(inital_state)
        frenet_state_list.append(current_frenet_state)

        if goal_speed is None:
            route_idx = route_lanelet_index(next_state.position, global_plan, scenario.lanelet_network)
            if route_idx is not None:  # off the route: keep the previous cycle's value
                limit = global_plan.speed_limits[route_idx]
                desired_speed = limit if limit is not None else DEFAULT_SPEED_LIMIT
        desired_speed_list.append(desired_speed)
        # Sampling bound: never below the current speed, so a vehicle that starts above
        # v_des is not forced to brake hard; J_V pulls it back to v_des.
        max_speed = max(desired_speed, current_frenet_state.s_d)

        start_time = time.time()
        if method in ('FOP_CPP', 'FISS+_CPP', 'CEM_CPP', 'MPPI_CPP'):
            best_traj_ego = planner.plan(current_frenet_state, max_speed, obstacles_all, i, next_state,
                                         desired_speed=desired_speed)
        else:
            best_traj_ego = planner.plan(current_frenet_state, max_speed, obstacles_all, i, next_state)
        end_time = time.time()

        if collect_ml_data:
            ego_pos, ego_yaw = np.asarray(inital_state.position, dtype=float), float(inital_state.orientation)
            global_coordination_state_list.append(inital_state)
            ref_local = reference_ahead_local(ref_ego_lane_pts[:, :2], ego_pos, ego_yaw, reference_path_lookahead_m)
            if method == 'CEM_CPP':
                cem_cycles.append(cem_cycle_record(planner, i, current_frenet_state, inital_state,
                                                   desired_speed, max_speed, best_traj_ego, ref_local))
            if best_traj_ego is not None and len(best_traj_ego.x) >= 2:
                path_local = transform_points_to_ego(np.column_stack([best_traj_ego.x, best_traj_ego.y]),
                                                     ego_pos, ego_yaw)
                optimal_path_x_local_list.append(path_local[:, 0].tolist())
                optimal_path_y_local_list.append(path_local[:, 1].tolist())
                ref_path_x_local_list.append(ref_local[:, 0].tolist())
                ref_path_y_local_list.append(ref_local[:, 1].tolist())

        if planner.all_trajs:
            all_trajs_accumulated.append(planner.all_trajs[-1])

        # planner.stats is reset at the start of every Python plan() call. Copy the values before
        # the early-failure break below so the cycle which actually stopped planning is retained.
        cycle_stats = planner.get_stats() if use_cpp_planner else planner.stats
        stats.last_cycle_num_rejected_dynamic = getattr(cycle_stats, 'num_rejected_dynamic', 0)
        stats.last_cycle_num_rejected_offroad = getattr(cycle_stats, 'num_rejected_offroad', 0)
        stats.last_cycle_num_rejected_collision = getattr(cycle_stats, 'num_rejected_collision', 0)

        best_trajs_all_time_steps.append(best_traj_ego)

        if best_traj_ego is None or len(best_traj_ego.x) < 2:
            print(f"Planning failed at time step {i}")
            stats.time_step_have_to_break = i
            stats.rear_end_failure = rear_end_threat(
                obstacles_array, obstacles_num_vertices, i, inital_state, vehicle,
                int(round(planner.settings.max_t / planner.settings.tick_t)))
            break
        processing_time = (end_time - start_time)
        stats.runtime_history.append(processing_time)
        if method == 'Sparse_FOP' or method == 'Sparse' or method == 'Sparse_CPP' or method == 'Sparse_WorldModel':
            stats.average_runtime += processing_time - planner.time_image_generation
            stats.num_FOP_intervention += planner.num_FOP_intervention
        else:
            stats.average_runtime += processing_time
        stats.best_traj_costs.append(best_traj_ego.cost_final)
        if not use_cpp_planner:
            stats += planner.stats
        else:
            stats += planner.get_stats()

        # Update and record the vehicle's trajectory
        next_step_idx = 1
        current_state = best_traj_ego.state_at_time_step(next_step_idx)
        current_frenet_state = best_traj_ego.frenet_state_at_time_step(
            next_step_idx)
        
        #TODO: update initial_state for low speed scenarios
        dt = planner.settings.tick_t
        yaw = best_traj_ego.yaw
        buf_yaw_rate = np.diff(yaw, prepend=yaw[0]) / dt

        next_state = InitialState(
            time_step=i,
            position=np.array([current_state.x, current_state.y]),
            orientation=current_state.yaw,
            velocity=current_state.v,
            acceleration=current_state.a,
            yaw_rate=buf_yaw_rate[next_step_idx]
        )
        
        
        state_list.append(next_state)
        time_list.append(end_time - start_time)

        # break when goal is reached
        if goal_position_available:
            if goal_region.is_reached(next_state):
                print("Goal Reached")
                goal_reached = True
                stats.success = True
                break
            # if goal_polygon.contains_properly()
            elif np.hypot(next_state.position[0] - goal_center[0], next_state.position[1] - goal_center[1]) <= vehicle.l/2:
                print("Goal Reached")
                stats.success = True
                goal_reached = True
                break
            elif current_frenet_state.s >= route_end_s - 3.0:
                print("Reaching End of the Map, Stopping, Goal Not Reached")
                goal_reached = True
                stats.success = True
                break
        
        # Standstill is allowed (the C++ planners switch to the low-speed lateral model d(s)
        # and can wait or start again); only the time spent standing is recorded.
        if abs(next_state.velocity) < 0.01:
            stats.standstill_time += dt

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(ref_ego_lane_pts[:, 0], ref_ego_lane_pts[:, 1])
            if len(obstacle_positions) > i:
                obstacle_markers = np.array(obstacle_positions[i])
                plt.plot(obstacle_markers[:, 0], obstacle_markers[:, 1], "X")
                plt.plot(best_traj_ego.x[next_step_idx:],
                         best_traj_ego.y[next_step_idx:], "-or")
                plt.plot(best_traj_ego.x[next_step_idx],
                         best_traj_ego.y[next_step_idx], "vc")
                plt.xlim(best_traj_ego.x[next_step_idx] -
                         area, best_traj_ego.x[next_step_idx] + area)
                plt.ylim(best_traj_ego.y[next_step_idx] -
                         area, best_traj_ego.y[next_step_idx] + area)

                plt.title(
                    "v[km/h]:" + str(best_traj_ego.s_d[next_step_idx] * 3.6)[0:4])
                plt.grid(True)
                plt.pause(0.0001)

        if i == final_time_step-1:
            stats.success = True
            goal_reached = True

    if collect_ml_data and global_coordination_state_list:
        scenario_name = os.path.splitext(file)[0]
        if method == 'CEM_CPP':
            save_cem_data(os.path.join(output_dir, "cem_data"), scenario_name, cem_cycles, stats, planner)
        drawer = ScenarioDrawer(
            scenario_name=scenario_name,
            scenario_dir=input_dir,
            save_dir=output_dir,
            ref_ego_lane_pts=ref_ego_lane_pts,
            vehicle_params=vehicle_params,
            obstacles_array=obstacles_array,
            obstacles_num_vertices=obstacles_num_vertices,
        )
        collect_data(
            drawer,
            scenario_name,
            optimal_path_x_local_list,
            optimal_path_y_local_list,
            ref_path_x_local_list,
            ref_path_y_local_list,
            global_coordination_state_list,
            output_dir,
            planner.settings.highest_speed,
            save_paths=(method == 'FOP_CPP'),
        )
    # construct the final frenet trajectory and calculate the final cost
    final_trajectory = FrenetTrajectory.from_frenet_states_list(frenet_state_list, global_state_list)
    dt = planner.settings.tick_t
    final_trajectory.cost_final, stats.final_cost_terms = evaluate_scenario_cost(
        final_trajectory, desired_speed_list, dt, obstacles_array, obstacles_num_vertices)
    stats.final_traj_cost = final_trajectory.cost_final
    # print(f"Final trajectory cost: {final_trajectory.cost_final}")
    avg_processing_time = processing_time / num_cycles
    stats.step_number = num_cycles
    stats.average(num_cycles)
    # print("average inferecence time:", planner.time_inference / num_cycles)
    # print("average image generation time:", planner.time_image_generation / num_cycles)
    
    # create the planned trajectory starting at time step 0
    if state_list:
        ego_vehicle_traj = Trajectory(
            initial_time_step=0, state_list=state_list)
    else:
        ego_vehicle_traj = None


    return goal_reached, ego_vehicle_traj, avg_processing_time, time_list, stats, all_trajs_accumulated, best_trajs_all_time_steps


def timeout_handler(signum, frame):
    raise BaseException("Program exceeded 10 seconds")


def informed_planning(scenario: Scenario, planning_problem: PlanningProblem, vehicle_params: DictConfig):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)

    # load the xml with stores the 524 motion primitives
    name_file_motion_primitives = 'V_0.0_20.0_Vstep_2.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml'
    # generate automaton
    automaton = ManeuverAutomaton.generate_automaton(
        name_file_motion_primitives)
    # plot motion primitives
    # plot_primitives(automaton.list_primitives)

    # load the xml with stores the 167 motion primitives
    name_file_motion_primitives = 'V_0.0_20.0_Vstep_4.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml'
    # generate automaton
    automaton = ManeuverAutomaton.generate_automaton(
        name_file_motion_primitives)
    # plot motion primitives
    # plot_primitives(automaton.list_primitives)

    # construct motion planner
    type_motion_planner = MotionPlannerType.GBFS  # UCS, ASTAR, STUDENT_EXAMPLE
    motion_planner = MotionPlanner.create(scenario=scenario,
                                          planning_problem=planning_problem,
                                          automaton=automaton,
                                          motion_planner_type=type_motion_planner)

    # solve for solution
    start_time = time.time()
    list_paths_primitives, _, _ = motion_planner.execute_search()
    end_time = time.time()
    processing_time = end_time - start_time

    ego_vehicle_trajectory = create_trajectory_from_list_states(
        list_paths_primitives, vehicle_params.b)
    return True, ego_vehicle_trajectory, processing_time, None


def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def planning(cfg: dict, output_dir: str, input_dir: str, file: str) -> Stats:
    # Global benchmark settings
    method = cfg['PLANNER']  # 'informed', 'FOP', 'FOP+', 'FISS', 'FISS+'
    num_samples = (cfg['N_W_SAMPLE'], cfg['N_S_SAMPLE'], cfg['N_T_SAMPLE'])
    save_gif = cfg['SAVE_GIF']
    show_sampled_trajs = cfg['SHOW_SAMPLED_TRAJECTORIES']
    #set number of threads for numba parallel collision checker
    number_threads = cfg['Num_Threads_For_CollisionChecker']
    runtime_measurement = cfg.get('Runtime_Measurement')
    collect_data_for_ml = cfg.get('Collect_Data_For_ML')
    if collect_data_for_ml and method == 'CEM_CPP' and not getattr(fop_cpp.frenet_planner_cpp, "DATA_COLLECTION", False):
        # Checked here, outside the "not feasible" handler below, so a wrong build cannot pass silently
        raise ImportError("Collect_Data_For_ML with CEM_CPP needs C_Planner/build compiled with "
                          "-DENABLE_DATA_COLLECTION=ON")
    configure_numba_threads(number_threads)

    vehicle_type = VehicleType.VW_VANAGON  # FORD_ESCORT, BMW_320i, VW_VANAGON
    vehicle_params = VehicleParameterMapping[vehicle_type.name].value

    ##################################################### Planning #########################################################
    # Read the Commonroad scenario
    file_path = os.path.join(input_dir, file)
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    planning_problem = list(
        planning_problem_set.planning_problem_dict.values())[0]
    initial_state = planning_problem.initial_state

    try:
        # Plan!
        if method == 'informed':
            _, ego_vehicle_trajectory, _, time_list = informed_planning(
                scenario, planning_problem, vehicle_params)
        else:
            _, ego_vehicle_trajectory, _, time_list, measurment, fplist, best_trajs = frenet_optimal_planning(
                scenario, planning_problem, vehicle_params, method, num_samples, input_dir, file, output_dir, 
                number_threads, runtime_measurement, collect_data_for_ml,
                sampler_cfg={'CEM': cfg.get('CEM'), 'MPPI': cfg.get('MPPI')}, safety_cfg=cfg.get('SAFETY'),
                motion_model_cfg=cfg.get('MOTION_MODEL'))

        if ego_vehicle_trajectory is None:
            print("No ego vehicle trajectory found")
            # A failure in the very first planning cycle still contains useful rejection
            # diagnostics. Return them instead of dropping the measurement entirely.
            if method != 'informed':
                return measurment
            raise RuntimeError

        # The ego vehicle can be visualized by converting it into a DynamicObstacle
        ego_vehicle_shape = Rectangle(
            length=vehicle_params.l, width=vehicle_params.w)
        ego_vehicle_prediction = TrajectoryPrediction(
            trajectory=ego_vehicle_trajectory, shape=ego_vehicle_shape)
        ego_vehicle_type = ObstacleType.CAR
        ego_vehicle = DynamicObstacle(obstacle_id=100, obstacle_type=ego_vehicle_type,
                                      obstacle_shape=ego_vehicle_shape, initial_state=initial_state,
                                      prediction=ego_vehicle_prediction)
        
        # record_sampling_parameters_to_csv(fplist)

    except RuntimeError:
        print("   ", f"{file} not feasible!")
        return

    ##################################################### Visualization #########################################################
    if save_gif and fplist:
        best_traj_lines = None
        images = []
        scenario_id = os.path.splitext(file)[0]
        # For each
        for i in range(len(fplist)):
            plt.figure(figsize=(25, 10))
            mpl.rcParams['font.size'] = 20
            rnd = MPRenderer()
            rnd.draw_params.time_begin = i
            # Disable drawing of dynamic obstacle trajectories (the black dots)
            rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = False
            rnd.draw_params.dynamic_obstacle.occupancy.draw_occupancies = False
            rnd.draw_params.lanelet_network.traffic_light.draw_traffic_lights = False
            rnd.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = False
            # Disable drawing of initial state arrow (green direction marker)
            rnd.draw_params.planning_problem.initial_state.state.draw_arrow = False
            scenario.draw(rnd, rnd.draw_params)
            # ...existing code...
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "g"
            ego_vehicle.draw(rnd)
            # planning_problem_set.draw(rnd)
            v_min, v_max = 0, 200
            norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
            rnd.render()
            if show_sampled_trajs:
                costs = []
                xs = []
                ys = []
                for fp in fplist[i]:
                    costs.append(fp.cost_final)
                    xs.append(fp.x[1:])
                    ys.append(fp.y[1:])
                lc = multiline(xs, ys, costs, ax=rnd.ax,
                               cmap='RdYlGn_r', lw=2, zorder=20)
                plt.colorbar(lc)
            else:
                if i < len(best_trajs):
                    best_fp = best_trajs[i]
                    if best_fp is not None and len(best_fp.x) > 1 and len(best_fp.y) > 1:
                        costs = [best_fp.cost_final]
                        xs = [best_fp.x[1:]]
                        ys = [best_fp.y[1:]]
                        lc = multiline(xs, ys, costs, ax=rnd.ax,norm=norm,
                                       cmap='RdYlGn_r', lw=2, zorder=20)
                        plt.colorbar(lc)

            x_coords = [state.position[0]
                        for state in ego_vehicle_trajectory.state_list]
            y_coords = [state.position[1]
                        for state in ego_vehicle_trajectory.state_list]
            x_coords_p = [state.position[0]
                          for state in ego_vehicle_trajectory.state_list[0:i]]
            y_coords_p = [state.position[1]
                          for state in ego_vehicle_trajectory.state_list[0:i]]
            x_coords_f = [state.position[0]
                          for state in ego_vehicle_trajectory.state_list[i:]]
            y_coords_f = [state.position[1]
                          for state in ego_vehicle_trajectory.state_list[i:]]
            dx_ego_f = np.diff(x_coords_f)
            dy_ego_f = np.diff(y_coords_f)
            # rnd.ax.plot(x_coords_p, y_coords_p, color='#9400D3',
            #             alpha=1,  zorder=25, lw=1)
            # rnd.ax.plot(x_coords_f, y_coords_f, color='#AFEEEE',
            #             alpha=1,  zorder=25, lw=1)
            # rnd.ax.quiver(x_coords_f[:-1:5], y_coords_f[:-1:5], dx_ego_f[::5], dy_ego_f[::5],
            #               scale_units='xy', angles='xy', scale=1, width=0.009, color='#AFEEEE', zorder=26)

            x_min = min(x_coords)-30
            x_max = max(x_coords)+30
            y_min = min(y_coords)-30
            y_max = max(y_coords)+30
            l = max(x_max-x_min, y_max-y_min)

            if l == x_max - x_min:
                plt.xlim(x_min, x_max)
                plt.ylim(y_min - (l-(y_max-y_min))/2,
                         y_max + (l-(y_max-y_min))/2)
            else:
                plt.xlim(x_min - (l-(x_max-x_min))/2,
                         x_max + (l-(x_max-x_min))/2)
                plt.ylim(y_min, y_max)

            for obs in scenario.dynamic_obstacles:
                t = 0
                obs_traj_x = []
                obs_traj_y = []
                while obs.state_at_time(t) is not None:
                    obs_traj_x.append(obs.state_at_time(t).position[0])
                    obs_traj_y.append(obs.state_at_time(t).position[1])
                    t += 1
                dx = np.diff(obs_traj_x)
                dy = np.diff(obs_traj_y)
                obs_traj_x = obs_traj_x[:-1]
                obs_traj_y = obs_traj_y[:-1]
                # rnd.ax.quiver(obs_traj_x[:i:5], obs_traj_y[:i:5], dx[:i:5], dy[:i:5],
                #               scale_units='xy', angles='xy', scale=1, width=0.006, color='#BA55D3', zorder=25)
                # rnd.ax.quiver(obs_traj_x[i::5], obs_traj_y[i::5], dx[i::5], dy[i::5],
                #               scale_units='xy', angles='xy', scale=1, width=0.006, color='#1d7eea', zorder=25)
                # rnd.ax.plot(obs_traj_x[0:i], obs_traj_y[0:i],
                #             color='#BA55D3', alpha=0.8,  zorder=25, lw=0.6)
                # rnd.ax.plot(obs_traj_x[i:], obs_traj_y[i:],
                #             color='#1d7eea', alpha=0.8,  zorder=25, lw=0.6)
            time_list.append(0)

            plt.title("{method}: {time}s".format(
                method=method, time=round(time_list[i], 3)))
            plt.suptitle(f'Scenario ID: {scenario_id}',
                         fontsize=20, x=0.59, y=0.06)

            # Write the figure into a jpg file
            result_path = os.path.join(
                output_dir, 'gif_cache', method, scenario_id)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
                print("Target directory: {} Created".format(result_path))
            fig_path = os.path.join(
                result_path, "{time_step}.jpg".format(time_step=i))
            plt.savefig(fig_path, dpi=200, bbox_inches='tight')
            print("Fig saved to:", fig_path)

            plt.close()

            images.append(Image.open(fig_path))

        # Genereate a gif file from the previously saved jpg files
        gif_dirpath = os.path.join(output_dir, 'gif/', method)
        if not os.path.exists(gif_dirpath):
            os.makedirs(gif_dirpath)
            print("Target directory: {} Created".format(gif_dirpath))
        gif_filepath = os.path.join(gif_dirpath, f"{scenario_id}.gif")
        images[0].save(gif_filepath, save_all=True,
                       append_images=images[1:], optimize=True, duration=100, loop=0)
        print("Gif saved to:", gif_filepath)

    return measurment

def save_data(scenario_name: str, optimal_path_x_list: list, optimal_path_y_list: list,
              ref_path_x_list: list, ref_path_y_list: list, output_dir: str):
    """Persist the ego-centered optimal path (sampled_vars.parquet) and ego-centered
    reference path (conditions.parquet) for one scenario. All coordinates here are
    already in the ego frame (ego at origin, yaw=0) at the time step they were planned."""
    os.makedirs(output_dir, exist_ok=True)
    # The parquet files are shared by every scenario: parallel processes (demo_cr.py --num_shards)
    # take turns on the read-append-write below.
    with open(os.path.join(output_dir, 'paths.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        samples_path = os.path.join(output_dir, 'sampled_vars.parquet')
        conditions_path = os.path.join(output_dir, 'conditions.parquet')

        # Check if scenario already exists in the parquet files
        if os.path.exists(samples_path):
            df_samples_existing = pd.read_parquet(samples_path)
            if scenario_name in df_samples_existing['scenario'].values:
                print(f"Scenario {scenario_name} already exists in data, skipping...")
                return
        else:
            df_samples_existing = None

        if os.path.exists(conditions_path):
            df_conditions_existing = pd.read_parquet(conditions_path)
        else:
            df_conditions_existing = None

        # Build sampled_vars data: the ego-centered optimal path returned by the planner.
        sampled_vars = {
            "scenario": [],
            "time_step": [],
            "x": [],
            "y": [],
            "path_length": [],
        }

        # Build conditions data: the ego-centered reference path ahead of the vehicle.
        conditions = {
            "scenario": [],
            "time_step": [],
            "ref_x": [],
            "ref_y": [],
            "ref_path_length": [],
        }

        for time_step, (path_x, path_y, ref_x, ref_y) in enumerate(
                zip(optimal_path_x_list, optimal_path_y_list, ref_path_x_list, ref_path_y_list)):
            sampled_vars["scenario"].append(scenario_name)
            sampled_vars["time_step"].append(time_step)
            sampled_vars["x"].append(path_x)
            sampled_vars["y"].append(path_y)
            sampled_vars["path_length"].append(len(path_x))

            conditions["scenario"].append(scenario_name)
            conditions["time_step"].append(time_step)
            conditions["ref_x"].append(ref_x)
            conditions["ref_y"].append(ref_y)
            conditions["ref_path_length"].append(len(ref_x))

        df_samples_new = pd.DataFrame(sampled_vars)
        df_conditions_new = pd.DataFrame(conditions)

        # Append to existing data if available
        if df_samples_existing is not None:
            df_samples = pd.concat([df_samples_existing, df_samples_new], ignore_index=True)
        else:
            df_samples = df_samples_new

        if df_conditions_existing is not None:
            df_conditions = pd.concat([df_conditions_existing, df_conditions_new], ignore_index=True)
        else:
            df_conditions = df_conditions_new

        df_samples.to_parquet(samples_path, index=False)
        df_conditions.to_parquet(conditions_path, index=False)

        print(f"Saved {len(optimal_path_x_list)} time steps for scenario {scenario_name}")


def reference_ahead_local(ref_xy: np.ndarray, ego_pos: np.ndarray, ego_yaw: float, lookahead: float) -> np.ndarray:
    """Reference path in the ego frame from the point nearest to the ego onward, up to `lookahead` m."""
    ref_local = transform_points_to_ego(ref_xy, ego_pos, ego_yaw)
    ahead = ref_local[int(np.argmin(np.linalg.norm(ref_local, axis=1))):]
    if len(ahead) > 1:
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ahead, axis=0), axis=1))])
        ahead = ahead[dist <= lookahead]
    return ahead


def cem_cycle_record(planner, time_step: int, frenet_state: FrenetState, state: InitialState,
                     v_des: float, max_speed: float, best: FrenetTrajectory, ref_local: np.ndarray) -> dict:
    """Context of one CEM cycle with its candidates and proposals (CVAE training data); ref_local is
    the reference path ahead in the ego frame (reference_ahead_local)."""
    rec = planner.cpp_planner.get_cycle_record()
    found = best is not None and len(best.x) >= 2
    context = {
        "time_step": time_step,
        **{k: getattr(frenet_state, k) for k in ("s", "s_d", "s_dd", "d", "d_d", "d_dd")},
        "d_s": np.nan if frenet_state.d_s is None else frenet_state.d_s,
        "x": state.position[0], "y": state.position[1], "yaw": state.orientation,
        "v": state.velocity, "a": state.acceleration,
        "v_des": v_des, "v_max_sample": max_speed, **rec["space"],
        "best_d": best.sampling_param.d if found else np.nan,
        "best_v": best.sampling_param.s_d if found else np.nan,
        "best_T": best.sampling_param.t if found else np.nan,
        "best_J": best.cost_final if found else np.nan,
        "clearance_fallback": int(np.any(rec["candidates"]["pass"] == 1)),
    }
    tables = {name: pd.DataFrame(rec[name]) for name in ("candidates", "proposals")}
    for table in tables.values():
        table.insert(0, "time_step", time_step)
    reference = {"time_step": time_step, "ref_x": ref_local[:, 0].astype(np.float32),
                 "ref_y": ref_local[:, 1].astype(np.float32)}
    return {"context": context, "reference": reference, **tables}


# Integer columns of the CEM tables; every other column is stored as float32
CEM_INT_COLUMNS = {"time_step": "int16", "pass": "int8", "iteration": "int8", "index": "int16",
                   "rank": "int16", "feasible": "int8", "rejection": "int8"}


def save_cem_data(root: str, scenario_name: str, cycles: list, stats: Stats, planner) -> None:
    """CEM training data of one scenario in root/<scenario>/: contexts.parquet (one row per
    planning cycle), conditions.parquet (reference path ahead in the ego frame per cycle),
    candidates.parquet (every candidate of every iteration: all violation
    components and J terms), proposals.parquet (Gaussian proposal of every iteration) and
    scenario.json. root/dataset_info.json documents the columns and the planner setup."""
    out = os.path.join(root, scenario_name)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame([c["context"] for c in cycles]).to_parquet(os.path.join(out, "contexts.parquet"), index=False)
    # reference path ahead in the ego frame, the format of the FOP path data (save_data)
    pd.DataFrame([c["reference"] for c in cycles]).to_parquet(os.path.join(out, "conditions.parquet"), index=False,
                                                              compression="zstd")
    for name in ("candidates", "proposals"):
        table = pd.concat([c[name] for c in cycles], ignore_index=True)
        table = table.astype({col: CEM_INT_COLUMNS.get(col, "float32") for col in table.columns})
        table.to_parquet(os.path.join(out, f"{name}.parquet"), index=False, compression="zstd")
    with open(os.path.join(out, "scenario.json"), "w") as fh:
        json.dump({"success": bool(stats.success), "rear_end_failure": bool(stats.rear_end_failure),
                   "cycles": len(cycles), "failed_at_time_step": stats.time_step_have_to_break if not stats.success else None},
                  fh, indent=1)
    try:
        with open(os.path.join(root, "dataset_info.json"), "x") as fh:  # first scenario writes it
            json.dump(cem_dataset_info(planner), fh, indent=1)
    except FileExistsError:
        pass


def cem_dataset_info(planner) -> dict:
    """Planner setup and column documentation of the CEM training data."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    git = lambda *args: subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()
    weights = planner.cpp_planner.cost_weights
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "planner_settings": {k: v for k, v in vars(planner.settings).items() if isinstance(v, (bool, int, float, str))},
        "cem_settings": planner.cem_cfg,
        "cost_weights": {k: getattr(weights, k) for k in dir(weights) if k.startswith("w_")},
        "vehicle": {k: getattr(planner.vehicle, k) for k in ("l", "w", "max_speed", "max_accel")},
        "rejection_codes": {0: "none", 1: "transform", 2: "dynamic", 3: "offroad", 4: "clearance", 5: "collision"},
        "notes": [
            "candidates: one row per sample; d, v, T = terminal lateral offset, terminal speed, horizon.",
            "pass 1 = re-plan without the clearance requirement (only when pass 0 found nothing feasible).",
            "rank: rank in its iteration by (feasible, V, J) of the early-exit view, as used by CEM; "
            "elites are rank < round(elite_fraction * population).",
            "V_*: share of the horizon from the first violation on, (N - k) / N, every check run "
            "(kFullViolation); feasible / rejection as in a normal run.",
            "J_*: weighted cost terms, NaN without a Cartesian trajectory.",
            "proposals: mean / std of the diagonal Gaussian each iteration was drawn from, in unit "
            "coordinates of the search space (contexts d_min..T_max); samples are clipped to [0, 1]; "
            "index 0 of iteration 0 is the proposal mean itself.",
            "conditions: per cycle the route's reference path in the ego frame of that cycle (ego at the "
            "origin, x forward, y left, as the BEV image), from its point nearest to the ego up to "
            "ScenarioDrawer.VIEW_SIZE_DEFAULT / 2 = 43.75 m ahead, points every 0.1 m (ref_x, ref_y); the "
            "format of the FOP path data and of the world-model CVAE (CVAE_trajectory_planning).",
        ],
    }


def collect_data(drawer: ScenarioDrawer, scenario_name: str,
                 optimal_path_x_list: list, optimal_path_y_list: list,
                 ref_path_x_list: list, ref_path_y_list: list,
                 global_coordination_state_list: list, output_dir: str,
                 highest_speed: float, save_paths: bool = True):
    """Images of every cycle, the planned paths (save_paths) and the completion marker."""
    if save_paths:
        save_data(scenario_name, optimal_path_x_list, optimal_path_y_list,
                  ref_path_x_list, ref_path_y_list, str(output_dir))

    # Save images for all time steps
    if drawer.save_dir is not None:
        drawer.save_scenario_imgs(
            ego_state_list=global_coordination_state_list,
            highest_speed=highest_speed,
        )
        print(f"Saved images for scenario {scenario_name}")

    done_dir = os.path.join(output_dir, "completed")
    os.makedirs(done_dir, exist_ok=True)

    marker_path = os.path.join(done_dir, f"{scenario_name}.done")
    with open(marker_path, "w") as marker:
        marker.write(str(len(optimal_path_x_list)))

def record_sampling_parameters_to_csv(fplist: list, output_path: str = "samplingParameterWithCost.csv"):
        """
        Record sampling parameters (d, s_d, t) and their costs to a CSV file.
        
        Args:
            fplist: List of lists containing FrenetTrajectory objects for each time step
            output_path: Path to the output CSV file
        """
        print(f"DEBUG: fplist length = {len(fplist) if fplist else 0}")
        
        total_trajs = 0
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(['time_step', 'd', 's_d', 't', 'cost'])
            # Write data - fplist is a list of lists (one per time step)
            for step_idx, step_trajs in enumerate(fplist):
                if step_trajs is None:
                    print(f"DEBUG: step {step_idx} is None")
                    continue
                print(f"DEBUG: step {step_idx} has {len(step_trajs)} trajectories")
                for fp in step_trajs:
                    if hasattr(fp, 'sampling_param') and hasattr(fp, 'cost_final'):
                        total_trajs += 1
                        writer.writerow([
                            step_idx,
                            fp.sampling_param.d,
                            fp.sampling_param.s_d,
                            fp.sampling_param.t,
                            fp.cost_final
                        ])
        
        print(f"Saved {total_trajs} sampling parameters to {output_path}")
