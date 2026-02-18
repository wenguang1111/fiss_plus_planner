import numpy as np

from fiss_plus_planner.planners.frenet_optimal_planner import FrenetOptimalPlanner, FrenetOptimalPlannerSettings
from fiss_plus_planner.planners.common.vehicle.vehicle import Vehicle

# backup planner inherited from FrenetOptimalPlanner, used in planning.py in case CVAE fails
class SP_FOP_Planner(FrenetOptimalPlanner):
    def __init__(self, planner_settings: FrenetOptimalPlannerSettings, ego_vehicle: Vehicle, 
                 obstacles_array: np.ndarray = None, obstacles_num_vertices: np.ndarray = None):
        
            super().__init__(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices)
