import numpy as np
from typing import Optional
from fiss_plus_planner.planners.common.scenario.frenet import FrenetTrajectory

class CostFunction:
    def __init__(self, cost_type: str):
        # FIXME: Here "WX1" is redundant, could be removed for interface to stay the same 
        if cost_type is "WX1":
            self.w_T = 10
            self.w_V = 1
            self.w_A = 0.1
            self.w_J = 0.1
            self.w_D = 0.1
            self.w_LC = 10
    
    def cost_time(self) -> float:
        pass
    
    def cost_terminal_time(self, terminal_time: float) -> float:
        pass
        # return self.w_T * terminal_time
    
    def cost_dist_obstacle(self, obstacles: list, time_step_now: int = 0) -> float:
        pass
        # dists = []
        # for obstacle in obstacles:
        #     obstacle
        # Xis = np.exp(-dists)
        # return self.w_D * sum(Xis)
    
    def cost_velocity_offset(self, vels: list, v_target: float) -> float:
        return self.w_V * sum(np.power(np.subtract(vels, v_target), 2))
    
    def cost_acceleration(self, accels: list) -> float:
        return self.w_A * sum(np.power(accels, 2))
            
    def cost_jerk(self, jerks: list) -> float:
        return self.w_J * sum(np.power(jerks, 2))
    
    def cost_lane_center_offset(self, offsets: list) -> float:
        return self.w_LC * sum(np.power(offsets, 2))
    
    def cost_total(self, traj: FrenetTrajectory, target_speed: float) -> float:
        cost_time = 10.0 - traj.t[-1] # self.cost_time()
        cost_obstacle = 0.0 # self.cost_dist_obstacle()
        cost_speed = self.cost_velocity_offset(traj.s_d, target_speed)
        cost_accel = self.cost_acceleration(traj.s_dd) + self.cost_acceleration(traj.d_dd)
        cost_jerk = self.cost_jerk(traj.s_ddd) + self.cost_jerk(traj.d_ddd)
        cost_offset = self.cost_lane_center_offset(traj.d)
        # return cost_speed + cost_accel + cost_jerk + cost_offset
        cost_total = (cost_time + cost_obstacle + cost_speed + cost_accel + cost_jerk + cost_offset)/len(traj.t)
        return cost_total
    
    
class DefaultCostFunction:
    """
    Default cost function for comfort driving
    """
    # FIXME:
    # for the interface to stay the same as before,
    # desired_d should be deleted (redundant) as it will always be 0 (we want to penalize any deviation from the reference)
    # I need you to confirm that the desired_s should also be deleted as we do not consider a target s position
    # (we are in velocity keeping)
    # one difference between reactive planner and fiss is that fiss sets desired speed as the highest speed,
    # while reactive planner sets it as the speed of the inital state (we need to stick to a convention)
    def __init__(self, desired_speed: Optional[float] = None, desired_d: float = 0.0,
                 desired_s: Optional[float] = None):
        super(DefaultCostFunction, self).__init__()
        # target states
        self.desired_speed = desired_speed
        self.desired_d = desired_d
        self.desired_s = desired_s

        # weights
        self.w_a = 5    # acceleration weight

    # FIXME: 
    # Here we are missing velocity and acceleration in cartesian coordinates
    # and the orientation in the frenet frame (c_yaw) 
    def cost_total(self, trajectory: FrenetTrajectory) -> float:
        costs = 0.0
        # acceleration costs
        costs += np.sum((self.w_a * trajectory.a) ** 2)
        # velocity costs
        if self.desired_speed is not None:
            costs += np.sum((5 * (trajectory.v - self.desired_speed)) ** 2) + \
                     (50 * (trajectory.v[-1] - self.desired_speed) ** 2) + \
                     (100 * (trajectory.v[int(len(trajectory.v)/2)] - self.desired_speed) ** 2)
        # if we do not consider stopping then this part is not needed
        if self.desired_s is not None:
            costs += np.sum((0.25 * (self.desired_s - trajectory.s)) ** 2) + \
                 (20 * (self.desired_s - trajectory.s[-1])) ** 2

        # distance costs
        costs += np.sum((0.25 * (self.desired_d - trajectory.d)) ** 2) + \
                 (20 * (self.desired_d - trajectory.d[-1])) ** 2
        # orientation costs
        costs += np.sum((0.25 * np.abs(trajectory.c_yaw)) ** 2) + (
                5 * (np.abs(trajectory.c_yaw[-1]))) ** 2

        return costs