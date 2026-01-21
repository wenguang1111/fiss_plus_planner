import copy
import csv
import os
import math
import time
from itertools import product
import numpy as np
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState
from shapely import Polygon, affinity

from fiss_plus_planner.planners.common.cost.cost_function import CostFunction
from fiss_plus_planner.planners.common.geometry.cubic_spline import CubicSpline2D
from fiss_plus_planner.planners.common.geometry.polynomial import QuarticPolynomial, QuinticPolynomial
from fiss_plus_planner.planners.common.scenario.frenet import FrenetState, FrenetTrajectory
from fiss_plus_planner.planners.common.vehicle.vehicle import Vehicle
from fiss_plus_planner.planners.common.utils import prepare_trajectory_array, check_trajectories_collision
from fiss_plus_planner.planners.common.utils import check_trajectories_collision_parallel_static
from fiss_plus_planner.planners.frenet_optimal_planner import FrenetOptimalPlannerSettings
from typing import Tuple
import sys
from pathlib import Path

# Try to import C++ pybind11 module
try:
    # Get the project root directory
    # frenet_optimal_planner.py is at: fiss_plus_planner/planners/frenet_optimal_planner.py
    # Project root: /home/wenguang/workplace/fiss_plus_planner/
    # So: __file__.parent.parent.parent = project root
    project_root = Path(__file__).parent.parent.parent
    cpp_planner_build = project_root / 'C_Planner' / 'build'
    
    if cpp_planner_build.exists():
        # Add the build directory directly to sys.path so Python can find the .so file
        sys.path.insert(0, str(cpp_planner_build))
    
    import frenet_planner_cpp
    CPP_MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    import traceback
    traceback.print_exc()
    CPP_MODULE_AVAILABLE = False
except Exception as e:
    import traceback
    traceback.print_exc()
    CPP_MODULE_AVAILABLE = False

class Stats(object):
    def __init__(self):
        self.num_iter = 0
        self.num_trajs_generated = 0
        self.num_trajs_validated = 0
        self.num_collison_checks = 0
        self.average_runtime = 0.0
        self.step_number = 0
        self.best_traj_costs = [] # float("inf")
        self.average_cost = 0.0
        self.runtime_history = []
        self.time_step_have_to_break = 0 # for the code to break early when no feasible traj found in planning.py
        self.success = False
        
    def __add__(self, other):
        self.num_iter += other.num_iter
        self.num_trajs_generated += other.num_trajs_generated
        self.num_trajs_validated += other.num_trajs_validated
        self.num_collison_checks += other.num_collison_checks
        return self
    
    def average(self, value: int):
        self.num_iter /= value
        self.num_trajs_generated /= value
        self.num_trajs_validated /= value
        self.num_collison_checks /= value
        self.average_runtime /= value
        if len(self.best_traj_costs) > 0:
            self.average_cost = np.mean(self.best_traj_costs)
        return self

class FOP_CPP_Wrapper(object):
    def __init__(self, planner_settings: FrenetOptimalPlannerSettings, ego_vehicle: Vehicle, 
                 obstacles_array: np.ndarray = None, obstacles_num_vertices: np.ndarray = None, number_threads: int = 1 ,runtime_measurement: bool = False):
        self.settings = planner_settings
        self.vehicle = ego_vehicle
        self.cost_function = CostFunction("WX1")
        self.cubic_spline = None
        self.best_traj = None
        self.all_trajs = []
        self.doing_runtime_measurement = runtime_measurement
        self.number_threads = number_threads
        
        # Pre-processed obstacles data (optional)
        self.obstacles_array = obstacles_array
        self.obstacles_num_vertices = obstacles_num_vertices
        
        # Statistics
        self.stats = Stats()
        
        # C++ planner instance (optional)
        self.cpp_planner = None
        CPP_MODULE_AVAILABLE
        
        if CPP_MODULE_AVAILABLE:
            self._init_cpp_planner()
    
    def _init_cpp_planner(self):
        """Initialize C++ planner using pybind11 bindings"""
        try:
            # Create C++ SettingParameters
            cpp_settings = frenet_planner_cpp.SettingParameters(
                self.settings.num_width,
                self.settings.num_speed,
                self.settings.num_t
            )
            cpp_settings.tick_t = self.settings.tick_t
            cpp_settings.max_road_width = self.settings.max_road_width
            cpp_settings.highest_speed = self.settings.highest_speed
            cpp_settings.lowest_speed = self.settings.lowest_speed
            cpp_settings.min_t = self.settings.min_t
            cpp_settings.max_t = self.settings.max_t
            cpp_settings.check_obstacle = self.settings.check_obstacle
            cpp_settings.check_boundary = self.settings.check_boundary
            
            # Create C++ VehicleParams
            cpp_vehicle = frenet_planner_cpp.VehicleParams()
            cpp_vehicle.l = self.vehicle.l
            cpp_vehicle.w = self.vehicle.w
            cpp_vehicle.a = self.vehicle.a
            cpp_vehicle.b = self.vehicle.b
            cpp_vehicle.T_f = self.vehicle.T_f
            cpp_vehicle.T_r = self.vehicle.T_r
            cpp_vehicle.max_speed = self.vehicle.max_speed
            cpp_vehicle.max_accel = self.vehicle.max_accel
            cpp_vehicle.max_steering_angle = self.vehicle.max_steering_angle
            cpp_vehicle.max_steering_rate = self.vehicle.max_steering_rate
            
            # Prepare obstacle arrays (keep references to prevent GC)
            if self.obstacles_array is not None and self.obstacles_num_vertices is not None:
                self._cpp_obs_array = np.ascontiguousarray(self.obstacles_array, dtype=np.float64)
                self._cpp_num_verts_array = np.ascontiguousarray(self.obstacles_num_vertices, dtype=np.int32)
            else:
                # Create empty obstacle arrays
                self._cpp_obs_array = np.zeros((1, 1, 10, 2), dtype=np.float64)
                self._cpp_num_verts_array = np.zeros((1, 1), dtype=np.int32)

            obs_array = self._cpp_obs_array
            num_verts_array = self._cpp_num_verts_array
            
            num_time_steps = obs_array.shape[0]
            num_obstacles = obs_array.shape[1]
            max_vertices = obs_array.shape[2]
            
            # Create C++ planner
            self.cpp_planner = frenet_planner_cpp.FrenetPlanner(
                cpp_settings,
                cpp_vehicle,
                obs_array,
                num_verts_array,
                num_time_steps,
                num_obstacles,
                max_vertices
            )
        except Exception as e:
            print(f"Warning: Failed to initialize C++ planner: {e}")
            self.cpp_planner = None

    def get_stats(self) -> Stats:
        """Get statistics from C++ planner and convert to Python Stats object"""
        stats = Stats()
        if self.cpp_planner is not None:
            try:
                cpp_stats = self.cpp_planner.get_stats()
                stats.num_trajs_generated = cpp_stats.num_trajs_generated
                stats.num_trajs_validated = cpp_stats.num_trajs_validated
                stats.num_collison_checks = cpp_stats.num_collision_checks
            except Exception as e:
                print(f"Warning: Failed to get stats from C++ planner: {e}")
        return stats

    def get_samples(self):
        """ Get sampling parameters d, s_d, t """
        
        sampling_width = self.settings.max_road_width - self.vehicle.w
        
        d_samples = np.linspace(-sampling_width/2, sampling_width/2, self.settings.num_width)
        s__samples = np.linspace(self.settings.lowest_speed, self.settings.highest_speed, self.settings.num_speed)
        t_samples = np.linspace(self.settings.min_t, self.settings.max_t, self.settings.num_t)
        
        samples = list(product(d_samples, s__samples, t_samples))
        
        return samples

    def calc_frenet_paths(self, frenet_state: FrenetState, samples = None) -> list:
        """Calculate Frenet paths - can use C++ or Python implementation"""
        # if self.use_cpp and self.cpp_planner is not None:
        #     # Use C++ implementation
        #     try:
        #         cpp_state = frenet_planner_cpp.FrenetState()
        #         cpp_state.t = frenet_state.t
        #         cpp_state.s = frenet_state.s
        #         cpp_state.s_d = frenet_state.s_d
        #         cpp_state.s_dd = frenet_state.s_dd
        #         cpp_state.s_ddd = frenet_state.s_ddd
        #         cpp_state.d = frenet_state.d
        #         cpp_state.d_d = frenet_state.d_d
        #         cpp_state.d_dd = frenet_state.d_dd
        #         cpp_state.d_ddd = frenet_state.d_ddd
                
        #         cpp_trajs = self.cpp_planner.calc_frenet_paths(cpp_state)
                
        #         # Convert C++ trajectories back to Python
        #         frenet_paths = []
        #         for cpp_traj in cpp_trajs:
        #             fp = FrenetTrajectory()
        #             fp.t = list(cpp_traj.t)
        #             fp.s = list(cpp_traj.s)
        #             fp.s_d = list(cpp_traj.s_d)
        #             fp.s_dd = list(cpp_traj.s_dd)
        #             fp.s_ddd = list(cpp_traj.s_ddd)
        #             fp.d = list(cpp_traj.d)
        #             fp.d_d = list(cpp_traj.d_d)
        #             fp.d_dd = list(cpp_traj.d_dd)
        #             fp.d_ddd = list(cpp_traj.d_ddd)
        #             fp.cost_final = cpp_traj.cost_final
        #             frenet_paths.append(fp)
                
        #         self.all_trajs.append(frenet_paths)
        #         return frenet_paths
        #     except Exception as e:
        #         print(f"Warning: C++ calc_frenet_paths failed: {e}, falling back to Python")
        
        # -----------Python implementation (fallback or primary)-----------------
        frenet_paths = []

        if samples is None:
            samples = self.get_samples()
        
        traj_per_timestep = []
        for s in samples:
            
            di, tv, Ti = s
            
            Ti = max(Ti, self.settings.tick_t)
            
            fp = FrenetTrajectory()
            
            # lateral trajectory
            lat_qp = QuinticPolynomial(frenet_state.d, frenet_state.d_d, frenet_state.d_dd, di, 0.0, 0.0, Ti)
            fp.t = [t for t in np.arange(0.0, Ti, self.settings.tick_t)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            tfp = copy.deepcopy(fp)
            
            # longitudinal trajectory
            lon_qp = QuarticPolynomial(frenet_state.s, frenet_state.s_d, frenet_state.s_dd, tv, 0.0, Ti)
            tfp.s = [lon_qp.calc_point(t) for t in fp.t]
            tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
            tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
            tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
            
            # Compute the final cost
            tfp.cost_final = self.cost_function.cost_total(tfp, self.settings.highest_speed)
            frenet_paths.append(tfp)
            traj_per_timestep.append(tfp)
            
        self.all_trajs.append(traj_per_timestep)
        
        # print(f"Generated {len(frenet_paths)} frenet paths.")

        return frenet_paths

    def calc_global_paths(self, fplist: list) -> list:
        """Convert Frenet paths to global coordinates - can use C++ or Python implementation"""
        # if self.use_cpp and self.cpp_planner is not None:
        #     # Use C++ implementation
        #     try:
        #         cpp_trajs = []
        #         for fp in fplist:
        #             cpp_traj = frenet_planner_cpp.FrenetTrajectory()
        #             cpp_traj.t = list(fp.t)
        #             cpp_traj.s = list(fp.s)
        #             cpp_traj.s_d = list(fp.s_d)
        #             cpp_traj.s_dd = list(fp.s_dd)
        #             cpp_traj.s_ddd = list(fp.s_ddd)
        #             cpp_traj.d = list(fp.d)
        #             cpp_traj.d_d = list(fp.d_d)
        #             cpp_traj.d_dd = list(fp.d_dd)
        #             cpp_traj.d_ddd = list(fp.d_ddd)
        #             cpp_traj.cost_final = fp.cost_final
        #             cpp_trajs.append(cpp_traj)
                
        #         cpp_result = self.cpp_planner.calc_global_paths(cpp_trajs)
                
        #         # Convert back to Python
        #         passed_fplist = []
        #         for cpp_traj in cpp_result:
        #             fp = FrenetTrajectory()
        #             fp.t = list(cpp_traj.t)
        #             fp.s = list(cpp_traj.s)
        #             fp.s_d = list(cpp_traj.s_d)
        #             fp.s_dd = list(cpp_traj.s_dd)
        #             fp.s_ddd = list(cpp_traj.s_ddd)
        #             fp.d = list(cpp_traj.d)
        #             fp.d_d = list(cpp_traj.d_d)
        #             fp.d_dd = list(cpp_traj.d_dd)
        #             fp.d_ddd = list(cpp_traj.d_ddd)
        #             fp.x = list(cpp_traj.x)
        #             fp.y = list(cpp_traj.y)
        #             fp.yaw = list(cpp_traj.yaw)
        #             fp.ds = list(cpp_traj.ds)
        #             fp.c = list(cpp_traj.c)
        #             fp.c_d = list(cpp_traj.c_d)
        #             fp.c_dd = list(cpp_traj.c_dd)
        #             fp.cost_final = cpp_traj.cost_final
        #             passed_fplist.append(fp)
                
        #         return passed_fplist
        #     except Exception as e:
        #         print(f"Warning: C++ calc_global_paths failed: {e}, falling back to Python")
        
        # -----------------Python implementation (fallback or primary)-------------------
        passed_fplist = []
        for fp in fplist:
            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = self.cubic_spline.calc_position(fp.s[i])
                if ix is None:
                    break
                i_yaw = self.cubic_spline.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)
            
            if len(fp.x) >= 2:
                # calc yaw and ds
                fp.x = np.array(fp.x)
                fp.y = np.array(fp.y)
                x_d = np.diff(fp.x)
                y_d = np.diff(fp.y)
                fp.yaw = np.arctan2(y_d, x_d)
                fp.ds = np.hypot(x_d, y_d)
                fp.yaw = np.append(fp.yaw, fp.yaw[-1])
                # calc curvature
                dt = self.settings.tick_t
                fp.c = np.divide(np.diff(fp.yaw), fp.ds)
                fp.c_d = np.divide(np.diff(fp.c), dt)
                fp.c_dd = np.divide(np.diff(fp.c_d), dt)
                
                passed_fplist.append(fp)

        return passed_fplist
    
    def check_constraints(self, trajs: list) -> list:
        """Check trajectory constraints - can use C++ or Python implementation"""
        # if self.use_cpp and self.cpp_planner is not None:
        #     # Use C++ implementation
        #     try:
        #         cpp_trajs = []
        #         for traj in trajs:
        #             cpp_traj = frenet_planner_cpp.FrenetTrajectory()
        #             cpp_traj.s_d = list(traj.s_d)
        #             cpp_traj.s_dd = list(traj.s_dd)
        #             cpp_trajs.append(cpp_traj)
                
        #         cpp_result = self.cpp_planner.check_constraints(cpp_trajs)
                
        #         # Map results back to original trajectories
        #         result_indices = set()
        #         for cpp_traj in cpp_result:
        #             for i, traj in enumerate(trajs):
        #                 if list(cpp_traj.s_d) == traj.s_d and list(cpp_traj.s_dd) == traj.s_dd:
        #                     result_indices.add(i)
        #                     break
                
        #         return [trajs[i] for i in sorted(result_indices)]
        #     except Exception as e:
        #         print(f"Warning: C++ check_constraints failed: {e}, falling back to Python")
        
        # ---------------------Python implementation (fallback or primary)----------------------
        passed = []

        for i, traj in enumerate(trajs):
            # Max curvature check
            # if any([abs(c) > self.vehicle.max_curvature for c in traj.c]):
            #     continue
            # if any([abs(c_d) > self.vehicle.max_kappa_d for c_d in traj.c_d]):
            #     continue
            # if any([abs(c_dd) > self.vehicle.max_kappa_dd for c_dd in traj.c_dd]):
            #     continue
            # Max speed check
            if any([v > self.vehicle.max_speed for v in traj.s_d]):
                continue
            # Max accel check
            if any([abs(a) > self.vehicle.max_accel for a in traj.s_dd]):
                continue

            passed.append(i)
            
        return [trajs[i] for i in passed]
    
    def construct_polygon(self, polygon: Polygon, x: float, y: float, yaw: float) -> Polygon:
        polygon_translated = affinity.translate(polygon, xoff=x, yoff=y)
        polygon_rotated = affinity.rotate(polygon_translated, yaw, use_radians=True)
        
        return polygon_rotated
    
    def has_collision(self, traj: FrenetTrajectory, obstacles: list, time_step_now: int = 0, check_res: int = 1) -> tuple:
        num_polys = 0
        if len(obstacles) <= 0:
            return False, 0
        
        final_time_step = obstacles[0].prediction.final_time_step
        t_step_max = min(len(traj.x), final_time_step - time_step_now)
        for i in range(t_step_max):
            if i%check_res == 0:
                # construct a polygon for the ego vehicle at time step i
                try:
                    ego_polygon = self.construct_polygon(self.vehicle.polygon, traj.x[i], traj.y[i], traj.yaw[i])
                except:
                    print(f"Failed to create Polygon for t={i} x={traj.x[i]}, y={traj.y[i]}, yaw={traj.y[i]}")
                    return True, num_polys
                else:
                    # construct a polygon for the obstacle at time step i
                    t_step = i + time_step_now
                    for obstacle in obstacles:
                        state = obstacle.state_at_time(t_step)
                        if state is not None:
                            obstacle_polygon = self.construct_polygon(obstacle.obstacle_shape.shapely_object, state.position[0], state.position[1], state.orientation)
                            num_polys += 1
                            if ego_polygon.intersects(obstacle_polygon):
                                # plot_collision(ego_polygon, obstacle_polygon, t_step)
                                return True, num_polys

        return False, num_polys

    def check_collisions(self, trajs: list, obstacles: list, time_step_now: int = 0) -> list:
        """Sequential collision detection (base version for all planners)."""
        passed = []
        for i, traj in enumerate(trajs):
            collision, num_polys = self.has_collision(traj, obstacles, time_step_now, 1)
            if collision:
                continue
            passed.append(i)
        return [trajs[i] for i in passed]

    def check_collision_multithread(self, trajs: list, time_step_now: int = 0) -> list:
        """Multi-threaded collision detection using pre-processed obstacle data."""
        # ----------------------Try C++ implementation first-----------------
        # if self.use_cpp and self.cpp_planner is not None:
        #     try:
        #         cpp_trajs = []
        #         for traj in trajs:
        #             cpp_traj = frenet_planner_cpp.FrenetTrajectory()
        #             cpp_traj.t = list(traj.t)
        #             cpp_traj.x = list(traj.x)
        #             cpp_traj.y = list(traj.y)
        #             cpp_traj.yaw = list(traj.yaw)
        #             cpp_traj.s = list(traj.s)
        #             cpp_traj.s_d = list(traj.s_d)
        #             cpp_traj.s_dd = list(traj.s_dd)
        #             cpp_traj.d = list(traj.d)
        #             cpp_traj.d_d = list(traj.d_d)
        #             cpp_traj.d_dd = list(traj.d_dd)
        #             cpp_traj.cost_final = traj.cost_final
        #             cpp_trajs.append(cpp_traj)
                
        #         cpp_result = self.cpp_planner.check_collision_multithread(cpp_trajs, time_step_now)
                
        #         # Map results back - use cost_final as unique identifier
        #         result_costs = set(cpp_traj.cost_final for cpp_traj in cpp_result)
        #         return [traj for traj in trajs if traj.cost_final in result_costs]
        #     except Exception as e:
        #         print(f"Warning: C++ check_collision_multithread failed: {e}, falling back to Python")
        
        # ----------------Python implementation (fallback or primary)-----------------
        if len(trajs) == 0 or self.obstacles_array is None or self.obstacles_num_vertices is None:
            return trajs
        
        # Prepare trajectory data
        trajectories, traj_lengths = prepare_trajectory_array(trajs, return_lengths=True)
        
        # Use multi-threaded collision detection
        if len(trajs) == 0 or trajectories.shape[0] == 0:
            return trajs
        
        collision_mask = check_trajectories_collision(
            trajectories,
            traj_lengths,
            self.obstacles_array,
            self.obstacles_num_vertices,
            vehicle_length=self.vehicle.l,
            vehicle_width=self.vehicle.w,
            time_step_now=time_step_now,
            check_resolution=1
        )

        passed_indices = np.where(~collision_mask)[0]
        return [trajs[i] for i in passed_indices]

    def recordObstaclesForDebug(self, output_path: str = "python_obstacle.csv"):
        """Record obstacle array data to CSV file in the same format as C++ recordObstacleArray"""
        if self.obstacles_array is None or self.obstacles_num_vertices is None:
            return
        
        obstacles_array = self.obstacles_array
        num_vertices_array = self.obstacles_num_vertices
        
        # Get dimensions
        num_time_steps = obstacles_array.shape[0]
        num_obstacles = obstacles_array.shape[1]
        max_vertices = obstacles_array.shape[2]
        
        data = {}
        
        def save(key, value):
            data.setdefault(key, []).append(value)
        
        # Iterate through all time steps, obstacles, and vertices
        for t in range(num_time_steps):
            for obs in range(num_obstacles):
                num_verts = int(num_vertices_array[t, obs])
                for v in range(max_vertices):
                    # Save the metadata
                    save("obstacles.t", t)
                    save("obstacles.obs", obs)
                    save("obstacles.v", v)
                    save("obstacles.num_vertices", num_verts)
                    
                    # Save the vertex coordinates
                    x_coord = obstacles_array[t, obs, v, 0]
                    y_coord = obstacles_array[t, obs, v, 1]
                    save("obstacles.x", x_coord)
                    save("obstacles.y", y_coord)
        
        # Write to CSV file
        keys = sorted(data.keys())
        max_len = max(len(values) for values in data.values()) if data else 0
        
        def format_value(value):
            if value is None:
                return ""
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            return f"{float(value):.15f}"
        
        file_exists = os.path.exists(output_path)
        with open(output_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or os.path.getsize(output_path) == 0:
                writer.writerow(keys)
            for i in range(max_len):
                row = []
                for key in keys:
                    values = data[key]
                    if i < len(values):
                        row.append(format_value(values[i]))
                    else:
                        row.append("")
                writer.writerow(row)

    def recordPathForDebug(self, path: FrenetTrajectory, output_path: str = "py_traj_debug.csv"):
        if path is None:
            return

        data = {}

        def save(key, value):
            data.setdefault(key, []).append(value)

        def save_seq(name, seq):
            save(f"{name}.size", len(seq))
            for i, val in enumerate(seq):
                save(f"{name}.step", i)
                save(name, val)

        save("traj.cost_fix", path.cost_fix)
        save("traj.cost_dyn", path.cost_dyn)
        save("traj.cost_heu", path.cost_heu)
        save("traj.cost_est", path.cost_est)
        save("traj.cost_final", path.cost_final)

        save("traj.idx0", int(path.idx[0]))
        save("traj.idx1", int(path.idx[1]))
        save("traj.idx2", int(path.idx[2]))
        save("traj.lane_id", int(path.lane_id))
        save("traj.is_generated", 1 if path.is_generated else 0)
        save("traj.is_searched", 1 if path.is_searched else 0)
        save("traj.constraint_passed", 1 if path.constraint_passed else 0)
        save("traj.collision_passed", 1 if path.collision_passed else 0)

        save_seq("traj.t", path.t)
        save_seq("traj.s", path.s)
        save_seq("traj.s_d", path.s_d)
        save_seq("traj.s_dd", path.s_dd)
        save_seq("traj.s_ddd", path.s_ddd)
        save_seq("traj.d", path.d)
        save_seq("traj.d_d", path.d_d)
        save_seq("traj.d_dd", path.d_dd)
        save_seq("traj.d_ddd", path.d_ddd)
        save_seq("traj.x", path.x)
        save_seq("traj.y", path.y)
        save_seq("traj.yaw", path.yaw)
        save_seq("traj.ds", path.ds)
        save_seq("traj.c", path.c)
        save_seq("traj.c_d", path.c_d)
        save_seq("traj.c_dd", path.c_dd)

        keys = sorted(data.keys())
        max_len = max(len(values) for values in data.values()) if data else 0

        def format_value(value):
            if value is None:
                return ""
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            return f"{float(value):.15f}"

        file_exists = os.path.exists(output_path)
        with open(output_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or os.path.getsize(output_path) == 0:
                writer.writerow(keys)
            for i in range(max_len):
                row = []
                for key in keys:
                    values = data[key]
                    if i < len(values):
                        row.append(format_value(values[i]))
                    else:
                        row.append("")
                writer.writerow(row)

    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, time_step_now: int = 0, initial_state: InitialState = None) -> FrenetTrajectory:
        if self.cpp_planner is not None:
            try:
                cpp_state = frenet_planner_cpp.FrenetState()
                cpp_state.t = frenet_state.t
                cpp_state.s = frenet_state.s
                cpp_state.s_d = frenet_state.s_d
                cpp_state.s_dd = frenet_state.s_dd
                cpp_state.s_ddd = frenet_state.s_ddd
                cpp_state.d = frenet_state.d
                cpp_state.d_d = frenet_state.d_d
                cpp_state.d_dd = frenet_state.d_dd
                cpp_state.d_ddd = frenet_state.d_ddd
                
                cpp_traj = self.cpp_planner.plan(cpp_state, max_target_speed, time_step_now, self.number_threads)
                
                if cpp_traj.is_generated:
                    py_traj = FrenetTrajectory()
                    py_traj.t = list(cpp_traj.t)
                    py_traj.s = list(cpp_traj.s)
                    py_traj.s_d = list(cpp_traj.s_d)
                    py_traj.s_dd = list(cpp_traj.s_dd)
                    py_traj.s_ddd = list(cpp_traj.s_ddd)
                    py_traj.d = list(cpp_traj.d)
                    py_traj.d_d = list(cpp_traj.d_d)
                    py_traj.d_dd = list(cpp_traj.d_dd)
                    py_traj.d_ddd = list(cpp_traj.d_ddd)
                    py_traj.x = list(cpp_traj.x)
                    py_traj.y = list(cpp_traj.y)
                    py_traj.yaw = list(cpp_traj.yaw)
                    py_traj.ds = list(cpp_traj.ds)
                    py_traj.c = list(cpp_traj.c)
                    py_traj.c_d = list(cpp_traj.c_d)
                    py_traj.c_dd = list(cpp_traj.c_dd)
                    py_traj.cost_final = cpp_traj.cost_final
                    self.best_traj = py_traj

                    if self.doing_runtime_measurement:
                        py_fplist = []
                        cpp_fplist = self.cpp_planner.getAllSuccessfulTrajectories()
                        for cpp_fp in cpp_fplist:
                            fp = FrenetTrajectory()
                            fp.t = list(cpp_fp.t)
                            fp.s = list(cpp_fp.s)
                            fp.s_d = list(cpp_fp.s_d)
                            fp.s_dd = list(cpp_fp.s_dd)
                            fp.s_ddd = list(cpp_fp.s_ddd)
                            fp.d = list(cpp_fp.d)
                            fp.d_d = list(cpp_fp.d_d)
                            fp.d_dd = list(cpp_fp.d_dd)
                            fp.d_ddd = list(cpp_fp.d_ddd)
                            fp.x = list(cpp_fp.x)
                            fp.y = list(cpp_fp.y)
                            fp.yaw = list(cpp_fp.yaw)
                            fp.ds = list(cpp_fp.ds)
                            fp.c = list(cpp_fp.c)
                            fp.c_d = list(cpp_fp.c_d)
                            fp.c_dd = list(cpp_fp.c_dd)
                            fp.cost_final = cpp_fp.cost_final
                            py_fplist.append(fp)
                        self.all_trajs.append(py_fplist)
                        
                    return py_traj
                else:
                    return None
            except Exception as e:
                print(f"Warning: C++ plan failed: {e}, falling back to Python")

    def generate_frenet_frame(self, centerline_pts: np.ndarray):
        # Python implementation
        self.cubic_spline = CubicSpline2D(centerline_pts[:, 0], centerline_pts[:, 1])
        s = np.arange(0, self.cubic_spline.s[-1], 0.1)
        ref_xy = [self.cubic_spline.calc_position(i_s) for i_s in s]
        ref_yaw = [self.cubic_spline.calc_yaw(i_s) for i_s in s]
        ref_rk = [self.cubic_spline.calc_curvature(i_s) for i_s in s]
        #-----------CPP start-------------------------------------------
        # C++ implementation: pass centerline directly to C++ planner
        # C++ planner will internally create and store the cubic spline
        if self.cpp_planner is not None:
            try:
                centerline_pts_cpp = np.asarray(centerline_pts, dtype=np.float64)
                centerline_pts_xy = np.column_stack(
                    (centerline_pts_cpp[:, 0], centerline_pts_cpp[:, 1])
                )
                self.cpp_planner.generate_frenet_frame(centerline_pts_xy)
            except Exception as e:
                print(f"Warning: Failed to set C++ planner frenet frame: {e}")
        #-----------CPP end-------------------------------------------
        return self.cubic_spline, np.column_stack((ref_xy, ref_yaw, ref_rk))
