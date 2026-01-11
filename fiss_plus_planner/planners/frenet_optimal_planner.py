import copy
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
from fiss_plus_planner.planners.common.utils import check_trajectories_collision

from fiss_plus_planner.planners.common.utils import configure_numba_threads



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
    
class FrenetOptimalPlannerSettings(object):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5):
        # time resolution between two planned waypoints
        self.tick_t = 0.1  # time tick [s]
        
        # sampling parameters
        self.max_road_width = 3.5           # maximum road width [m]
        self.num_width = num_width          # road width sampling number

        self.highest_speed = 13.4112        # highest sampling speed [m/s]
        self.lowest_speed = 0.0             # lowest sampling speed [m/s]
        self.num_speed = num_speed          # speed sampling number
        
        self.min_t = 8.0                    # min prediction time [m]
        self.max_t = 10.0                   # max prediction time [m]
        self.num_t = num_t                  # time sampling number

        self.check_obstacle = True          # True if check collison with obstacles
        self.check_boundary = True          # True if check collison with road boundaries

class FrenetOptimalPlanner(object):
    def __init__(self, planner_settings: FrenetOptimalPlannerSettings, ego_vehicle: Vehicle):
        self.settings = planner_settings
        self.vehicle = ego_vehicle
        self.cost_function = CostFunction("WX1")
        self.cubic_spline = None
        self.best_traj = None
        self.all_trajs = []
        
        # Statistics
        self.stats = Stats()
        
    def get_samples(self):
        """ Get sampling parameters d, s_d, t """
        
        sampling_width = self.settings.max_road_width - self.vehicle.w
        
        d_samples = np.linspace(-sampling_width/2, sampling_width/2, self.settings.num_width)
        s__samples = np.linspace(self.settings.lowest_speed, self.settings.highest_speed, self.settings.num_speed)
        t_samples = np.linspace(self.settings.min_t, self.settings.max_t, self.settings.num_t)
        
        samples = list(product(d_samples, s__samples, t_samples))
        
        return samples

    def calc_frenet_paths(self, frenet_state: FrenetState, samples = None) -> list:
        frenet_paths = []

        if samples is None:
            samples = self.get_samples()
        
        traj_per_timestep = []
        for s in samples:
            
            di, tv, Ti = s
            
            # FIXME: ensure Ti is at least tick_t (temporary, these are not cvae dataset scenarios)
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

        return fplist
    
    def check_constraints(self, trajs: list) -> list:
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

    def check_collisions_parallel(
        self,
        trajs: list,
        obstacles: list,
        time_step_now: int = 0,
        check_resolution: int = 1
    ) -> list:
        """
        使用多线程并行碰撞检测过滤轨迹
        
        Args:
            trajs: 轨迹列表
            obstacles: 障碍物列表
            time_step_now: 当前时间步
            check_resolution: 检查间隔（每隔 n 个点检查一次）
        
        Returns:
            通过碰撞检测的轨迹列表
        """
        if len(trajs) == 0 or len(obstacles) == 0:
            return trajs
        
        try:
            # 调用多线程碰撞检测函数
            collision_mask, num_checks = check_trajectories_collision(
                trajs,
                obstacles,
                vehicle_length=self.vehicle.l,
                vehicle_width=self.vehicle.w,
                time_step_now=time_step_now,
                check_resolution=check_resolution
            )
            
            # 更新统计信息
            self.stats.num_collison_checks = num_checks
            
            # 返回未碰撞的轨迹
            passed_indices = np.where(~collision_mask)[0]
            return [trajs[i] for i in passed_indices]
        
        except Exception as e:
            print(f"Error in parallel collision detection: {e}")
            print("Falling back to sequential collision detection...")
            # 如果并行检测失败，回退到顺序检测
            return self.check_collisions_sequential(trajs, obstacles, time_step_now)

    def check_collisions_sequential(
        self,
        trajs: list,
        obstacles: list,
        time_step_now: int = 0
    ) -> list:
        """
        顺序碰撞检测（备用方案，兼容原有逻辑）
        """
        passed = []
        for i, traj in enumerate(trajs):
            collision, num_polys = self.has_collision(traj, obstacles, time_step_now, 2)
            if collision:
                continue
            passed.append(i)
        return [trajs[i] for i in passed]

    def check_collisions(self, trajs: list, obstacles: list, time_step_now: int = 0) -> list:
        """
        碰撞检测主函数 - 使用多线程并行版本
        
        Args:
            trajs: 轨迹列表
            obstacles: 障碍物列表
            time_step_now: 当前时间步
        
        Returns:
            通过碰撞检测的轨迹列表
        """
        return self.check_collisions_parallel(trajs, obstacles, time_step_now, check_resolution=1)
    
    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, time_step_now: int = 0, initial_state: InitialState = None) -> FrenetTrajectory:
        # reset stats
        self.stats = Stats()
        self.settings.highest_speed = max_target_speed
        
        fplist = self.calc_frenet_paths(frenet_state)
        fplist = self.calc_global_paths(fplist)
        self.stats.num_trajs_generated = len(fplist)
        self.stats.num_trajs_validated = len(fplist)
        self.stats.num_collison_checks = len(fplist)
        fplist = self.check_constraints(fplist)
        # print(len(fplist), "trajectories passed constraint check")
        fplist = self.check_collisions(fplist, obstacles, time_step_now)
        # print(len(fplist), "trajectories passed collision check")

        # find minimum cost path
        min_cost = float("inf")
        for fp in fplist:
            if min_cost >= fp.cost_final:
                min_cost = fp.cost_final
                self.best_traj = fp
                        
        return self.best_traj

    def generate_frenet_frame(self, centerline_pts: np.ndarray):
        self.cubic_spline = CubicSpline2D(centerline_pts[:, 0], centerline_pts[:, 1])
        s = np.arange(0, self.cubic_spline.s[-1], 0.1)
        ref_xy = [self.cubic_spline.calc_position(i_s) for i_s in s]
        ref_yaw = [self.cubic_spline.calc_yaw(i_s) for i_s in s]
        ref_rk = [self.cubic_spline.calc_curvature(i_s) for i_s in s]
        return self.cubic_spline, np.column_stack((ref_xy, ref_yaw, ref_rk))
