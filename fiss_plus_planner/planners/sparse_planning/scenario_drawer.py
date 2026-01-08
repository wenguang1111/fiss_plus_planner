import os
from pathlib import Path
from typing import Iterable, List
from PIL import Image
import io

import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_params import DynamicObstacleParams
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.state import State
from commonroad.prediction.prediction import TrajectoryPrediction

from torch import Tensor
import torchvision.transforms as transforms


class ScenarioDrawer:
    """Render CommonRoad scenarios to image files."""

    def __init__(self, scenario_name: str, scenario_dir: str | Path, save_dir: str | Path = None):
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.scenario_dir = Path(scenario_dir)
        self.scenario_name = scenario_name
        self.ego_params = DynamicObstacleParams()
        self.ego_params.draw_icon = True
        self.ego_params.vehicle_shape.occupancy.shape.facecolor = "#ff0000"
        # self.ego_params.trajectory.facecolor = "#00B427"
        # self.ego_params.trajectory.line_width = 0.25
        self.ego_params.draw_icon = True
        self.ego_id = None
        self.ego_type = ObstacleType.CAR
        self.shape = Rectangle(width=1.8, length=4.3)
        print(os.path.join(self.scenario_dir, self.scenario_name + ".xml"))
        print(self.scenario_dir, self.scenario_name)
        self.scenario, _ = CommonRoadFileReader(os.path.join(self.scenario_dir, self.scenario_name + ".xml")).open()
        if save_dir is not None:
            os.makedirs(self.save_dir / self.scenario_name, exist_ok=True)
        
        self._transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
        ])

    def generate_image_at_time_step(self, time_step: int, ego_state: State, ego_trajectory: TrajectoryPrediction = None) -> Tensor:
        """Render image at a specific time step and return as torch tensor."""
        
        if self.ego_id == None:
            self.ego_id = self.scenario.generate_object_id()

        ego_vehicle = DynamicObstacle(
                    self.ego_id,
                    self.ego_type,
                    self.shape,
                    ego_state,
                    ego_trajectory,
                )
        # if self.scenario._is_object_id_used(self.ego_id) is False:
        #     self.scenario.add_objects(ego_vehicle)

        self.ego_params.time_begin = time_step

        renderer = MPRenderer()        
        renderer.focus_obstacle_id = self.ego_id
        renderer.draw_params.axis_visible = False
        renderer.draw_params.time_begin = time_step
        renderer.draw_params.dynamic_obstacle.draw_shape = True
        renderer.draw_params.dynamic_obstacle.draw_icon = True
        renderer.draw_params.dynamic_obstacle.trajectory.line_width = 0.25

        self.scenario.draw(renderer)
        ego_vehicle.draw(renderer, draw_params=self.ego_params)

        plt.gca().set_aspect("equal")
        renderer.render(show=True)
        # buf = io.BytesIO()
        # fig = plt.gcf()
        plt.savefig(self.save_dir / self.scenario_name / f"{time_step}.png", 
                    format="png", 
                    bbox_inches="tight", 
                    pad_inches=0, 
                    dpi=300)
        plt.close()
        # img = Image.open(buf).convert("RGB")
        # img = self._transform(img)
        # img = img.unsqueeze(0)
        # buf.close()
        # return img
    
    def save_scenario_imgs(
        self,
        trajectory
    ) -> None:
        """
        save an image of the scenarios at a specific timestep.
        
        Args:
                Scenarios path.
                Img save directory.
                Scenario name.
                Ego vehicle id.
                Time step.
        """

        os.makedirs(os.path.join(self.save_dir, self.scenario_name), exist_ok=True)

        ego_params = DynamicObstacleParams()
        ego_params.vehicle_shape.occupancy.shape.facecolor = "#ff0000"
        ego_params.trajectory.facecolor = "#00B427"
        ego_params.trajectory.line_width = 0.25
        ego_params.draw_icon = True
        
        ego_id = self.scenario.generate_object_id()
        
        for timestep, ego_state in enumerate(trajectory):
            # plt.figure(figsize=(6, 6))
            ego_vehicle = DynamicObstacle(
                ego_id,
                ObstacleType.CAR,
                Rectangle(width=1.8, length=4.3),
                ego_state,
            )

            # loop through all time steps and save images
            # for ts in range(0, no_time_step + 1):
            renderer = MPRenderer()
            # focus on ego vehicle
            renderer.focus_obstacle_id = ego_id
            renderer.draw_params.axis_visible = False
            renderer.draw_params.time_begin = timestep
            renderer.draw_params.dynamic_obstacle.draw_shape = True
            renderer.draw_params.dynamic_obstacle.draw_icon = True
            renderer.draw_params.dynamic_obstacle.trajectory.line_width = 0.25
            
            ego_params.time_begin = timestep
            
            self.scenario.draw(renderer)
            # pp.draw(renderer)
            
            ego_vehicle.draw(renderer, draw_params=ego_params)
            
            plt.gca().set_aspect("equal")
            renderer.render()
            
            # save img without padding
            plt.savefig(
                os.path.join(self.save_dir, self.scenario_name, f"{timestep}.png"),
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
            )
            plt.clf()

    def save_single_time_step(self, time_step: int, ego_state: State, ego_trajectory: TrajectoryPrediction = None) -> Path:
        """Render one time step and save as PNG."""
        # genearate only figure around ego vehicle
        png_bytes = self.generate_image_at_time_step(time_step, ego_state, ego_trajectory)

        if  self.save_dir is None:
            raise ValueError("save_dir is None, cannot save PNG to disk. Check the EnviromentDrawer initialization.")
        else:
            output_dir = self.save_dir / self.scenario_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"time_step_{time_step}.png"
            output_path.write_bytes(png_bytes)
            return output_path


#----------------------------- Example Usage -----------------------#
from commonroad.scenario.state import InitialState
from commonroad.scenario.trajectory import Trajectory
import numpy as np

def main():
    scenario_dir = "/home/wenguang/workplace/test/wenguang/fiss_plus_planner/data/demo"
    save_dir = "/home/wenguang/workplace/test/wenguang/fiss_plus_planner/data/output/"
    scenario_name = "ARG_Carcarana-1_2_T-1"
    drawer = ScenarioDrawer(scenario_name, scenario_dir, save_dir)
    scenario, planning_problem_set = CommonRoadFileReader(os.path.join(scenario_dir, scenario_name) + ".xml").open()

    planning_problem = next(iter(planning_problem_set.planning_problem_dict.values()))
    for time_step in range(0, 9):
        # Move ego vehicle forward by 1 meter each time step
        new_position = np.array(planning_problem.initial_state.position) + np.array([time_step * 1.0, 0.0])
        new_state = InitialState(
            time_step=time_step,
            position=new_position,
            orientation=planning_problem.initial_state.orientation,
            velocity=planning_problem.initial_state.velocity,
        )
        traj = Trajectory(initial_time_step=time_step, state_list=[new_state])
        trajectory_prediction = TrajectoryPrediction(trajectory=traj, shape=Rectangle(length=1.0, width=1.0))
        # drawer.save_single_time_step(new_state, trajectory_prediction, time_step)
        drawer.save_single_time_step(time_step, new_state, trajectory_prediction)


if __name__ == "__main__":
    main()
