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
from fiss_plus_planner.planners.fiss_plus_planner import FissPlusPlannerSettings
from typing import Tuple
from fiss_plus_planner.planners.FOP_cpp_wrapper import Stats
import sys
from pathlib import Path

# Try to import C++ pybind11 module
CPP_MODULE_AVAILABLE = False
fiss_plus_planner_cpp = None

try:
    project_root = Path(__file__).parent.parent.parent
    cpp_planner_build = project_root / 'C_Planner' / 'build'
    
    if cpp_planner_build.exists():
        # Add the build directory directly to sys.path so Python can find the .so file
        if str(cpp_planner_build) not in sys.path:
            sys.path.insert(0, str(cpp_planner_build))
    
    import fiss_plus_planner_cpp
    CPP_MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Failed to import C++ module: {e}")
    import traceback
    traceback.print_exc()
    CPP_MODULE_AVAILABLE = False
except Exception as e:
    print(f"Warning: Exception during C++ module import: {e}")
    import traceback
    traceback.print_exc()
    CPP_MODULE_AVAILABLE = False


class FissPlusCppWrapper(object):
    def __init__(self, planner_settings: FissPlusPlannerSettings, ego_vehicle: Vehicle, 
                 obstacles_array: np.ndarray = None, obstacles_num_vertices: np.ndarray = None, 
                 number_threads: int = 1, runtime_measurement: bool = False):
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
        
        if CPP_MODULE_AVAILABLE:
            self._init_cpp_planner()
    
    def _init_cpp_planner(self):
        """Initialize C++ planner using pybind11 bindings"""
        try:
            # Create C++ FissPlusPlannerSettings
            cpp_settings = fiss_plus_planner_cpp.FissPlusPlannerSettings(
                self.settings.num_width,
                self.settings.num_speed,
                self.settings.num_t,
                self.settings.max_refine_iters
            )
            cpp_settings.tick_t = self.settings.tick_t
            cpp_settings.max_road_width = self.settings.max_road_width
            cpp_settings.highest_speed = self.settings.highest_speed
            cpp_settings.lowest_speed = self.settings.lowest_speed
            cpp_settings.min_t = self.settings.min_t
            cpp_settings.max_t = self.settings.max_t
            cpp_settings.check_obstacle = self.settings.check_obstacle
            cpp_settings.check_boundary = self.settings.check_boundary
            
            # FISS+ specific settings
            cpp_settings.refine_trajectory = self.settings.refine_trajectory
            cpp_settings.max_refine_iters = self.settings.max_refine_iters
            cpp_settings.has_time_limit = self.settings.has_time_limit
            cpp_settings.time_limit = self.settings.time_limit
            cpp_settings.decaying_factor = self.settings.decaying_factor
            cpp_settings.w_heuristic = self.settings.w_heuristic
            cpp_settings.vis_all_candidates = self.settings.vis_all_candidates
            
            if self.obstacles_array is not None and self.obstacles_num_vertices is not None:
                self._cpp_obs_array = np.ascontiguousarray(self.obstacles_array, dtype=np.float64)
                self._cpp_num_verts_array = np.ascontiguousarray(self.obstacles_num_vertices, dtype=np.int32)
            else:
                self._cpp_obs_array = np.zeros((1, 1, 10, 2), dtype=np.float64)
                self._cpp_num_verts_array = np.zeros((1, 1), dtype=np.int32)

            obs_array = self._cpp_obs_array
            num_verts_array = self._cpp_num_verts_array
            
            num_time_steps = obs_array.shape[0]
            num_obstacles = obs_array.shape[1]
            max_vertices = obs_array.shape[2]
            
            # Create C++ FISS+ planner
            # Pass self.vehicle directly - C++ will extract attributes
            self.cpp_planner = fiss_plus_planner_cpp.FissPlusPlanner(
                cpp_settings,
                self.vehicle,
                obs_array,
                num_verts_array,
                num_time_steps,
                num_obstacles,
                max_vertices
            )
        except Exception as e:
            print(f"Warning: Failed to initialize C++ FISS+ planner: {e}")
            import traceback
            traceback.print_exc()
            self.cpp_planner = None

    def get_stats(self) -> Stats:
        """Get statistics from C++ planner and convert to Python Stats object"""
        stats = Stats()
        if self.cpp_planner is not None:
            try:
                cpp_stats = self.cpp_planner.get_stats()  # Returns dict
                stats.num_trajs_generated = cpp_stats["num_trajs_generated"]
                stats.num_trajs_validated = cpp_stats["num_trajs_validated"]
                stats.num_collison_checks = cpp_stats["num_collision_checks"]
            except Exception as e:
                print(f"Warning: Failed to get stats from C++ planner: {e}")
        return stats

    def _dict_to_frenet_trajectory(self, d: dict) -> FrenetTrajectory:
        """Convert a dict returned from C++ to FrenetTrajectory"""
        fp = FrenetTrajectory()
        fp.t = list(d["t"])
        fp.s = list(d["s"])
        fp.s_d = list(d["s_d"])
        fp.s_dd = list(d["s_dd"])
        fp.s_ddd = list(d["s_ddd"])
        fp.d = list(d["d"])
        fp.d_d = list(d["d_d"])
        fp.d_dd = list(d["d_dd"])
        fp.d_ddd = list(d["d_ddd"])
        fp.x = list(d["x"])
        fp.y = list(d["y"])
        fp.yaw = list(d["yaw"])
        fp.ds = list(d["ds"])
        fp.c = list(d["c"])
        fp.c_d = list(d["c_d"])
        fp.c_dd = list(d["c_dd"])
        fp.cost_final = d["cost_final"]
        fp.idx = np.array(d["idx"])
        return fp

    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, 
             time_step_now: int = 0, initial_state: InitialState = None) -> FrenetTrajectory:
        if self.cpp_planner is not None:
            try:
                # Pass Python FrenetState directly - C++ will extract attributes
                cpp_traj_dict = self.cpp_planner.plan(frenet_state, max_target_speed, time_step_now)
                
                if cpp_traj_dict["is_generated"]:
                    py_traj = self._dict_to_frenet_trajectory(cpp_traj_dict)
                    self.best_traj = py_traj

                    if self.doing_runtime_measurement:
                        py_fplist = []
                        cpp_fplist = self.cpp_planner.getAllSuccessfulTrajectories()  # Returns list of dicts
                        for cpp_fp_dict in cpp_fplist:
                            fp = self._dict_to_frenet_trajectory(cpp_fp_dict)
                            py_fplist.append(fp)
                        self.all_trajs.append(py_fplist)
                        
                    return py_traj
                else:
                    return None
            except Exception as e:
                print(f"Warning: C++ FISS+ plan failed: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print("Warning: C++ FISS+ planner not available")
            return None

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
