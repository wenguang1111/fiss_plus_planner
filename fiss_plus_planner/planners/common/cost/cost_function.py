import numpy as np
from fiss_plus_planner.planners.common.scenario.frenet import FrenetTrajectory

class CostFunction:
    def __init__(self, cost_type: str):
        if cost_type is "WX1":
            self.w_T = 10.0
            self.w_V = 0.1
            self.w_A = 1.0
            self.w_J = 1.0
            self.w_D = 100.0
            self.w_LC = 10.0
            self.w_dist = 1.0
            self.max_speed = 14.0  # for normalization, can be adjusted based on

    
    def cost_time(self) -> float:
        pass
    
    def cost_terminal_time(self, terminal_time: float) -> float:
        return self.w_T * terminal_time
    
    def _compute_polygon_center(self, vertices: np.ndarray, num_verts: int) -> np.ndarray:
        if num_verts <= 0:
            return np.array([np.inf, np.inf])
        return vertices[:num_verts].mean(axis=0)

    def is_obstacle_front(self, ego_pose: np.ndarray, obs_center: np.ndarray) -> bool:
        # returns whether the obstacle is in front of the ego vehicle
        # based on the dot product between the relative vector between ego and obstacle and the ego direction vector
        # if > 0, then the obstacle is in front, otherwise it's behind
        
        ego_direction = np.array([np.cos(ego_pose[2]), np.sin(ego_pose[2])])
        relative_vector = obs_center - ego_pose[:2]
        dot_product = np.dot(relative_vector, ego_direction)
        # print(dot_product)
        return dot_product > 0
    
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
        
        Xis = np.exp(-np.array(min_dists)*self.w_dist)
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
        cost_time = self.cost_terminal_time(15.0 - 0.1*len(traj.t))
        cost_obstacle = 0.0 # self.cost_dist_obstacle()
        cost_speed = self.cost_velocity_offset(np.abs(traj.s_d), self.max_speed)
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
        cost_time = self.cost_terminal_time(15.0 - 0.1*len(traj.t)) 
        cost_obstacle = 0.0 #self.cost_dist_obstacle(obstacles_array, obstacles_num_vertices, traj, time_step_now)
        cost_speed = self.cost_velocity_offset(np.abs(traj.s_d), self.max_speed)
        cost_accel = self.cost_acceleration(traj.s_dd) + self.cost_acceleration(traj.d_dd)
        cost_jerk = self.cost_jerk(traj.s_ddd) + self.cost_jerk(traj.d_ddd)
        cost_offset = self.cost_lane_center_offset(traj.d)
        cost_total = (cost_time + cost_obstacle + cost_speed + cost_accel + cost_jerk + cost_offset)/len(traj.t)
        return cost_total
    
    def final_trajectory_cost(self,
                              traj: FrenetTrajectory,
                              obstacles_array: np.ndarray,
                              obstacles_num_vertices: np.ndarray) -> float:
        """
        calculate the cost of the final executed trajectory
        """
        
        cost_time = self.cost_terminal_time(0.1*len(traj.t))  # assuming each time step is 0.1s, and the total time is 10s
        cost_obstacle = self.cost_dist_obstacle(obstacles_array, obstacles_num_vertices, traj)
        cost_speed = self.cost_velocity_offset(np.abs(traj.v), self.max_speed)
        cost_accel = self.cost_acceleration(traj.s_dd) + self.cost_acceleration(traj.d_dd)
        cost_jerk = self.cost_jerk(traj.s_ddd) + self.cost_jerk(traj.d_ddd)
        cost_offset = self.cost_lane_center_offset(traj.d)
        cost_total = ( cost_obstacle + cost_speed + cost_accel + cost_jerk + cost_offset)/len(traj.t) + cost_time
        # print("velocity:", traj.v)
        print("Cost Distance To Obstacles:", cost_obstacle)
        print("Cost Time:", cost_time)
        print("Cost Speed:", cost_speed)
        print("Cost Acceleration:", cost_accel)
        print("Cost Jerk:", cost_jerk)
        print("Cost Lane Center Offset:", cost_offset)
        print("len(traj.t):", len(traj.t))
        print("Total Cost:", cost_total)
        return cost_total
    
    def calc_cost(self, fplist:list, target_speed: float, obstacles_array: np.ndarray,
        obstacles_num_vertices: np.ndarray, time_step_now: int) -> list:
        for traj in fplist:
            traj.cost_final = self.cost_singleTrajectory(traj, target_speed, obstacles_array, obstacles_num_vertices, time_step_now)
        return fplist

