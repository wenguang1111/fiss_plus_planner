from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from commonroad.scenario.state import CustomState
from torchvision import transforms

from fiss_plus_planner.planners.common.scenario.frenet import FrenetTrajectory
from fiss_plus_planner.planners.common.utils import transform_points_from_ego
from fiss_plus_planner.planners.constant_velocity_prediction import ConstantVelocityObstaclePredictor
from fiss_plus_planner.planners.fop_plus_wm_selection import FopPlusWMSelectionPlanner


class FopPlusCVSelectionPlanner(FopPlusWMSelectionPlanner):
    """FOP+_WM with the world model replaced by constant-velocity obstacle prediction.

    Each cycle the surrounding obstacles are extrapolated once with lane-relative constant velocity.
    For every candidate, the future BEV frames are rendered by the same ``ScenarioDrawer`` that draws
    the observed BEV, centred on the candidate's own future ego poses. Candidates are then ranked with
    exactly the FOP+_WM settings and costs (risk ROI, step weights, lateral and progress terms).
    """

    def _init_bev_predictor(self) -> None:
        scenario = self.scenario_drawer.scenario
        self.obstacle_predictor = ConstantVelocityObstaclePredictor(
            scenario.lanelet_network,
            scenario.obstacles,
            max_vertices=self.obstacles_array.shape[2],
            dt=scenario.dt,
        )
        self._last_candidate_bevs: List[List[Image.Image]] = []

    def _score_candidates(self, candidates: List[FrenetTrajectory], images: List[Image.Image],
                          ego_pos: np.ndarray, ego_yaw: float, time_step_now: int
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return each candidate's weighted rollout collision risk and its ego-frame action deltas."""
        rollout = self.settings.world_model_rollout
        actions = self._candidate_actions(candidates, ego_pos, ego_yaw)
        ego_poses = self._rollout_ego_poses(actions, ego_pos, ego_yaw, rollout)
        obstacle_polygons, obstacle_num_vertices = self.obstacle_predictor.predict_footprints(
            time_step_now, rollout
        )
        initial_collision_risk = self.collision_risk(self._bev_tensor(images[-1]))

        self._last_candidate_bevs = []
        collision_scores = []
        for candidate_poses in ego_poses:
            bevs = [
                self.scenario_drawer.create_scenario_img_with_obstacles(
                    CustomState(time_step=time_step_now + step, position=pose[:2], orientation=pose[2]),
                    obstacle_polygons[step - 1],
                    obstacle_num_vertices[step - 1],
                ).convert("L")
                for step, pose in enumerate(candidate_poses, start=1)
            ]
            rollout_collision_risks = torch.stack([
                self.collision_risk(self._bev_tensor(bev)) for bev in bevs
            ]) - initial_collision_risk
            collision_scores.append((rollout_collision_risks * self._world_model_risk_step_weights).sum())
            self._last_candidate_bevs.append(bevs)

        return torch.stack(collision_scores), actions

    @staticmethod
    def _rollout_ego_poses(actions: torch.Tensor, ego_pos: np.ndarray, ego_yaw: float,
                           num_steps: int) -> np.ndarray:
        """Global ego poses (x, y, yaw) of rollout frames 1..num_steps, shape (candidates, num_steps, 3).

        Frame k shows the scene 0.1 * k s ahead, centred on candidate point k and heading towards
        point k + 1: the frame convention of the world-model rollout (``correct_reference_frame``).
        The poses come from the same padded action deltas the world model receives, so a candidate
        cut short where the reference path ends is extended by repeating its last displacement.
        """
        steps = np.arange(1, num_steps + 1)
        positions = torch.cumsum(actions, dim=1).cpu().numpy().astype(float)  # current ego frame
        direction = positions[:, steps + 1] - positions[:, steps]
        local_heading = np.arctan2(direction[..., 1], direction[..., 0])
        global_xy = transform_points_from_ego(
            positions[:, steps].reshape(-1, 2), ego_pos, ego_yaw
        ).reshape(len(positions), len(steps), 2)
        return np.concatenate((global_xy, (local_heading + ego_yaw)[..., None]), axis=-1)

    def _bev_tensor(self, image: Image.Image) -> torch.Tensor:
        return transforms.ToTensor()(image)[0].to(self.settings.device)

    def save_failure_rollout_plot(self, output_path: str | Path) -> bool:
        """Save the rendered BEV frames of the latest selected candidate."""
        if not self._last_candidate_total_costs or not self._last_candidate_bevs:
            return False
        selected_index = int(np.argmin(self._last_candidate_total_costs))
        frames = [np.asarray(bev, dtype=np.float32) / 255.0 for bev in self._last_candidate_bevs[selected_index]]
        return self._save_bev_rollout_plot(frames, output_path, "CV")
