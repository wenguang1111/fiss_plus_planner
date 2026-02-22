import numpy as np
from fiss_plus_planner.planners.common.scenario.frenet import FrenetTrajectory

class CostFunction:
    def __init__(self, cost_type: str):
        if cost_type is "WX1":
            self.w_T = 10
            self.w_V = 0.1
            self.w_A = 0.1
            self.w_J = 0.1
            self.w_D = 10
            self.w_LC = 1
    
    def cost_time(self) -> float:
        pass
    
    def cost_terminal_time(self, terminal_time: float) -> float:
        return self.w_T * terminal_time
    
    def _compute_polygon_center(self, vertices: np.ndarray, num_verts: int) -> np.ndarray:
        if num_verts <= 0:
            return np.array([np.inf, np.inf])
        return vertices[:num_verts].mean(axis=0)
    
    def cost_dist_obstacle(
        self, 
        obstacles_array: np.ndarray, 
        obstacles_num_vertices: np.ndarray,
        traj: FrenetTrajectory, 
        time_step_now: int = 0
    ) -> float:
        num_traj_points = len(traj.x)
        num_time_steps = obstacles_array.shape[0]
        num_obstacles = obstacles_array.shape[1]
        
        if num_obstacles == 0 or num_traj_points == 0:
            return 0.0
        
        min_dists = []
        
        for i in range(num_traj_points):
            t_idx = time_step_now + i
            # t_idx = time_step_now
            
            if t_idx >= num_time_steps:
                break
                
            traj_x = traj.x[i]
            traj_y = traj.y[i]
            
            min_dist = np.inf
            
            for obs_idx in range(num_obstacles):
                num_verts = obstacles_num_vertices[t_idx, obs_idx]

                if num_verts <= 0:
                    continue
                
                vertices = obstacles_array[t_idx, obs_idx]
                center = self._compute_polygon_center(vertices, num_verts)
                
                dist = np.sqrt((traj_x - center[0])**2 + (traj_y - center[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
            
            if min_dist < np.inf:
                min_dists.append(min_dist)
        
        if len(min_dists) == 0:
            return 0.0
        
        Xis = np.exp(-np.array(min_dists))
        return self.w_D * np.sum(Xis)
    
    def cost_velocity_offset(self, vels: list, v_target: float) -> float:
        return self.w_V * sum(np.power(np.subtract(vels, v_target), 2))
    
    def cost_acceleration(self, accels: list) -> float:
        return self.w_A * sum(np.power(accels, 2))
            
    def cost_jerk(self, jerks: list) -> float:
        return self.w_J * sum(np.power(jerks, 2))
    
    def cost_lane_center_offset(self, offsets: list) -> float:
        return self.w_LC * sum(np.power(offsets, 2))
    
    def cost_total(self, traj: FrenetTrajectory, target_speed: float) -> float:
        cost_time = self.cost_terminal_time(10.0 - traj.t[-1])  # self.cost_time()
        cost_obstacle = 0.0 # self.cost_dist_obstacle()
        cost_speed = self.cost_velocity_offset(traj.s_d, target_speed)
        cost_accel = self.cost_acceleration(traj.s_dd) + self.cost_acceleration(traj.d_dd)
        cost_jerk = self.cost_jerk(traj.s_ddd) + self.cost_jerk(traj.d_ddd)
        cost_offset = self.cost_lane_center_offset(traj.d)
        # return cost_speed + cost_accel + cost_jerk + cost_offset
        cost_total = (cost_time + cost_obstacle + cost_speed + cost_accel + cost_jerk + cost_offset)/len(traj.t)
        return cost_total
    
    def cost_singleTrajectory(
        self, 
        traj: FrenetTrajectory, 
        target_speed: float, 
        obstacles_array: np.ndarray,
        obstacles_num_vertices: np.ndarray,
        time_step_now: int
    ) -> float:
        cost_time = self.cost_terminal_time(10.0 - traj.t[-1]) 
        cost_obstacle = self.cost_dist_obstacle(obstacles_array, obstacles_num_vertices, traj, time_step_now)
        cost_speed = self.cost_velocity_offset(traj.s_d, target_speed)
        cost_accel = self.cost_acceleration(traj.s_dd) + self.cost_acceleration(traj.d_dd)
        cost_jerk = self.cost_jerk(traj.s_ddd) + self.cost_jerk(traj.d_ddd)
        cost_offset = self.cost_lane_center_offset(traj.d)
        cost_total = (cost_time + cost_obstacle + cost_speed + cost_accel + cost_jerk + cost_offset)/len(traj.t)
        return cost_total
    
    def final_trajectory_cost(self,
                              traj: FrenetTrajectory,
                              target_speed: float,
                              obstacles_array: np.ndarray,
                              obstacles_num_vertices: np.ndarray) -> float:
        """
        calculate the cost of the final executed trajectory
        """
        
        cost_obstacle = 0.0
        
        cost_time = self.cost_terminal_time(10.0 - traj.t[-1])
        for i in range(len(traj.t)):
            cost_obstacle += self.cost_dist_obstacle(obstacles_array, obstacles_num_vertices, traj, i)
        cost_speed = self.cost_velocity_offset(traj.s_d, target_speed)
        cost_accel = self.cost_acceleration(traj.s_dd) + self.cost_acceleration(traj.d_dd)
        cost_jerk = self.cost_jerk(traj.s_ddd) + self.cost_jerk(traj.d_ddd)
        cost_offset = self.cost_lane_center_offset(traj.d)
        cost_total = (cost_time + cost_obstacle + cost_speed + cost_accel + cost_jerk + cost_offset)/len(traj.t)
        # print("Cost Distance To Obstacles:", cost_obstacle)
        # print("Cost Time:", cost_time)
        # print("Cost Speed:", cost_speed)
        # print("Cost Acceleration:", cost_accel)
        # print("Cost Jerk:", cost_jerk)
        # print("Cost Lane Center Offset:", cost_offset)
        # print("len(traj.t):", len(traj.t))
        # print("Total Cost:", cost_total)
        return cost_total
    
    def calc_cost(self, fplist:list, target_speed: float, obstacles_array: np.ndarray,
        obstacles_num_vertices: np.ndarray, time_step_now: int) -> list:
        for traj in fplist:
            traj.cost_final = self.cost_singleTrajectory(traj, target_speed, obstacles_array, obstacles_num_vertices, time_step_now)
        return fplist

