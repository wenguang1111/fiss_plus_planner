import os
import signal
import time
import csv

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
from fiss_plus_planner.planners.commonroad_interface.global_planner import GlobalPlanner
from fiss_plus_planner.planners.fiss_planner import FissPlanner, FissPlannerSettings
from fiss_plus_planner.planners.fiss_plus_planner import FissPlusPlanner, FissPlusPlannerSettings
from fiss_plus_planner.planners.fop_plus_planner import FopPlusPlanner
from fiss_plus_planner.planners.frenet_optimal_planner import FrenetOptimalPlanner, FrenetOptimalPlannerSettings, Stats
from fiss_plus_planner.planners.sparse_planner import SparsePlannerSettings, SparsePlanner
from fiss_plus_planner.planners.sparse_planner_cpp import SparsePlannerSettings_CPP, SparsePlanner_CPP
from fiss_plus_planner.planners.FOP_cpp_wrapper import FOP_CPP_Wrapper
from fiss_plus_planner.planners.fiss_plus_cpp_wrapper import FissPlusCppWrapper
from fiss_plus_planner.SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from fiss_plus_planner.SMP.motion_planner.motion_planner import MotionPlanner, MotionPlannerType
from fiss_plus_planner.SMP.motion_planner.utility import create_trajectory_from_list_states
from fiss_plus_planner.planners.common.utils import configure_numba_threads
from fiss_plus_planner.planners.sparse_planning.scenario_drawer import ScenarioDrawer
from fiss_plus_planner.planners.sparse_planner_optimized import SparsePlannerOptimizedSettings, SparsePlannerOptimized
from fiss_plus_planner.planners.sparse_planner_fop import SparsePlannerFOPSettings, SparsePlannerFOP

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
    "pdf.fonttype": 42, "ps.fonttype": 42,        # keep text searchable
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


def frenet_optimal_planning(scenario: Scenario, planning_problem: PlanningProblem, vehicle_params: DictConfig, method: str, num_samples: tuple, 
                            input_dir: str, file: str, output_dir: str, number_threads: int, runtime_measurement: bool, collect_data_for_ml: bool
                            ) -> Tuple[bool, Trajectory, float, list, Stats, list, list]:
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

    if has_goal_state and goal_region.state_list[0].has_value("velocity"):
        speed_interval = goal_region.state_list[0].velocity
        min_speed = speed_interval.start
        max_speed = speed_interval.end
    else:
        min_speed = 0.0
        max_speed = 14
    
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
    elif method == 'FOP_CPP':
        # Use C++ Frenet Optimal Planner with pybind11
        planner_settings = FrenetOptimalPlannerSettings(num_width, num_speed, num_t)
        planner = FOP_CPP_Wrapper(planner_settings, vehicle, obstacles_array, obstacles_num_vertices, number_threads, runtime_measurement)
        use_cpp_planner = True  # Check if C++ planner was successfully initialized
        # planner.recordObstaclesForDebug("python_obstacle.csv")
    elif method == 'FISS+_CPP':
        # Use C++ FISS+ Planner with pybind11
        planner_settings = FissPlusPlannerSettings(num_width, num_speed, num_t)
        planner = FissPlusCppWrapper(planner_settings, vehicle, obstacles_array, obstacles_num_vertices, number_threads, runtime_measurement)
        use_cpp_planner = True
    elif method == 'Sparse_CPP':
        # Use C++ Sparse Planner with pybind11
        planner_settings = SparsePlannerSettings_CPP(num_width, num_speed, num_t, input_dir, file)
        planner = SparsePlanner_CPP(planner_settings, vehicle, obstacles_array, obstacles_num_vertices)
        use_cpp_planner = True
    else:
        print("ERROR: Planning method entered is not recognized!")
        raise ValueError

    csp_ego, ref_ego_lane_pts = planner.generate_frenet_frame(ego_lane_pts)

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
    
    sampling_params_cross_all_scenarios = []
    goal_reached = False
    next_state = initial_state
    best_trajs_all_time_steps = []
    
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
        global_coordination_state_list.append(inital_state)
        global_state_list.append(inital_state)
        frenet_state_list.append(current_frenet_state)

        start_time = time.time()
        best_traj_ego = planner.plan(current_frenet_state, max_speed, obstacles_all, i, next_state)
        end_time = time.time()

        best_trajs_all_time_steps.append(best_traj_ego)

        if best_traj_ego is None or len(best_traj_ego.x) < 2:
            print(f"Planning failed at time step {i}")
            stats.time_step_have_to_break = i
            break
        processing_time = (end_time - start_time)
        stats.runtime_history.append(processing_time)
        if method == 'Sparse_FOP' or method == 'Sparse':
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
        sampling_params_cross_all_scenarios.append(best_traj_ego.sampling_param)

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
            elif np.hypot(next_state.position[0] - ref_ego_lane_pts[-1, 0], next_state.position[1] - ref_ego_lane_pts[-1, 1]) <= 3.0:
                print("Reaching End of the Map, Stopping, Goal Not Reached")
                goal_reached = True
                stats.success = True
                break
        
        #break when the speed is close to zero, this is a simple model with out standstill feature.
        if abs(next_state.velocity) < 0.01:
            goal_reached = True
            stats.success = False
            break

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
            
    # construct the final frenet trajectory and calculate the final cost
    final_trajectory = FrenetTrajectory.from_frenet_states_list(frenet_state_list, global_state_list)
    final_trajectory.cost_final = planner.cost_function.final_trajectory_cost(
        traj=final_trajectory,
        target_speed=max_speed,
        obstacles_array=obstacles_array,
        obstacles_num_vertices=obstacles_num_vertices,
    )
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

    if collect_data_for_ml and method == 'FOP_CPP' and goal_reached:
        scenario_name = os.path.splitext(file)[0]
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
            sampling_params_cross_all_scenarios,
            frenet_state_list,
            global_coordination_state_list,
            output_dir,
            planner.settings.highest_speed,
        )

    return goal_reached, ego_vehicle_traj, avg_processing_time, time_list, stats, planner.all_trajs, best_trajs_all_time_steps


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
                number_threads, runtime_measurement, collect_data_for_ml)

        if ego_vehicle_trajectory is None:
            print("No ego vehicle trajectory found")
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
        # For each
        for i in range(len(fplist)):
            plt.figure(figsize=(25, 10))
            mpl.rcParams['font.size'] = 20
            rnd = MPRenderer()
            rnd.draw_params.time_begin = i
            # Disable drawing of dynamic obstacle trajectories (the black dots)
            rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = False
            rnd.draw_params.dynamic_obstacle.occupancy.draw_occupancies = False
            # Disable drawing of initial state arrow (green direction marker)
            rnd.draw_params.planning_problem.initial_state.state.draw_arrow = False
            scenario.draw(rnd)
            # ...existing code...
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "g"
            ego_vehicle.draw(rnd)
            planning_problem_set.draw(rnd)
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
                        lc = multiline(xs, ys, costs, ax=rnd.ax,
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
            scenario_id = os.path.splitext(file)[0]
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

            # plt.show()
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

def save_data(scenario_name: str, frenet_state_list: list, global_coordination_state_list: list, sampling_params: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Build sampled_vars data from sampling_params
    sampled_vars = {
        "scenario": [],
        "time_step": [],
        "t": [],
        "d": [],
        "v": []
    }
    
    # Build conditions data from state_list
    conditions = {
        "scenario": [],
        "time_step": [],
        "x": [],
        "y": [],
        "theta": [],
        "velocity": [],
        "acceleration": [],
        "yaw_rate": [],
        "s": [],
        "s_d": [],
        "s_dd": [],
        "s_ddd": [],
        "d": [],
        "d_d": [],
        "d_dd": [],
        "d_ddd": []
    }
    
    for time_step, (frenet_state, global_state, sampling_param) in enumerate(zip(frenet_state_list, global_coordination_state_list, sampling_params)):
        # Add sampling params
        sampled_vars["scenario"].append(scenario_name)
        sampled_vars["time_step"].append(time_step)
        sampled_vars["t"].append(sampling_param.t)
        sampled_vars["d"].append(sampling_param.d)
        sampled_vars["v"].append(sampling_param.s_d)
        
        # Add conditions from state
        conditions["scenario"].append(scenario_name)
        conditions["time_step"].append(time_step)
        conditions["x"].append(global_state.position[0])
        conditions["y"].append(global_state.position[1])
        conditions["theta"].append(global_state.orientation)
        conditions["velocity"].append(global_state.velocity)
        conditions["acceleration"].append(global_state.acceleration)
        conditions["yaw_rate"].append(global_state.yaw_rate)
        conditions["s"].append(frenet_state.s)
        conditions["s_d"].append(frenet_state.s_d)
        conditions["s_dd"].append(frenet_state.s_dd)
        conditions["s_ddd"].append(frenet_state.s_ddd)
        conditions["d"].append(frenet_state.d)
        conditions["d_d"].append(frenet_state.d_d)
        conditions["d_dd"].append(frenet_state.d_dd)
        conditions["d_ddd"].append(frenet_state.d_ddd)
    
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
    
    print(f"Saved {len(global_coordination_state_list)} time steps for scenario {scenario_name}")


def collect_data(drawer: ScenarioDrawer, scenario_name: str, sampling_params_cross_all_scenarios: list,
                 frenet_state_list: list, global_coordination_state_list: list, output_dir: str,
                 highest_speed: float):
    save_data(scenario_name, frenet_state_list, global_coordination_state_list, sampling_params_cross_all_scenarios, str(output_dir))
    
    # Save images for all time steps
    if drawer.save_dir is not None:
        drawer.save_scenario_imgs(
            ego_state_list=global_coordination_state_list,
            highest_speed=highest_speed,
        )
        print(f"Saved images for scenario {scenario_name}")

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
