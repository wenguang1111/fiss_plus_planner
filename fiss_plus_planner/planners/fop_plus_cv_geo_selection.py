from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from commonroad.scenario.state import InitialState
from PIL import Image

from fiss_plus_planner.planners.common.scenario.frenet import FrenetTrajectory
from fiss_plus_planner.planners.common.utils import compute_vehicle_polygon, polygon_collision
from fiss_plus_planner.planners.fop_plus_cv_selection import FopPlusCVSelectionPlanner, FopPlusCVSelectionSettings


class FopPlusCVGeoSelectionSettings(FopPlusCVSelectionSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5,
                 scenario_dir: str = "", scenario_file: str = "", horizon_steps: int = 5,
                 motion_model: str = "cv"):
        super().__init__(num_width, num_speed, num_t, scenario_dir, scenario_file, motion_model)
        # Collision-checked steps of 0.1 s: 5 matches the world-model rollout, 10 covers the whole
        # 1.0 s candidate. The lateral and progress costs keep the FOP+_WM action sequence either way.
        self.collision_horizon_steps = horizon_steps
        # All cost weights are FOP+_WM's. Per-step risk weights: FOP+_WM's own over its rollout
        # horizon; over the longer 1.0 s horizon they are uniform and likewise sum to 1.
        self.collision_step_weights = (
            self.world_model_risk_step_weights if horizon_steps == self.world_model_rollout
            else (1.0 / horizon_steps,) * horizon_steps
        )


class FopPlusCVGeoSelectionPlanner(FopPlusCVSelectionPlanner):
    """Classical baseline for FOP+_WM: object-level prediction (constant velocity, constant acceleration or
    IDM) and geometric collision checking, without any BEV.

    Candidates, cost weights, lateral and progress costs, and the post-selection checks are those of
    FOP+_WM. Only the risk term differs: at step k of the collision horizon the ego footprint at
    candidate pose k is intersected with the obstacle footprints predicted for k steps ahead, with the
    polygon test of the ground-truth collision check; the risk is the step-weighted sum of collisions.
    """

    def _init_bev_predictor(self) -> None:
        super()._init_bev_predictor()
        self._last_rollout: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _update_image_history(self, time_step_now: int, current_state: InitialState) -> List[Image.Image]:
        return []  # scoring is purely geometric, so no BEV is drawn

    def _score_candidates(self, candidates: List[FrenetTrajectory], images: List[Image.Image],
                          ego_pos: np.ndarray, ego_yaw: float, time_step_now: int
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return each candidate's step-weighted predicted collisions and its action deltas."""
        horizon = self.settings.collision_horizon_steps
        # Pose k needs candidate points k and k + 1, i.e. horizon + 2 points including the start.
        pose_actions = self._candidate_actions(candidates, ego_pos, ego_yaw, num_steps=horizon + 2)
        ego_poses = self._rollout_ego_poses(pose_actions, ego_pos, ego_yaw, horizon)
        obstacle_polygons, obstacle_num_vertices = self.obstacle_predictor.predict_footprints(
            time_step_now, horizon
        )
        self._last_rollout = (ego_poses, obstacle_polygons, obstacle_num_vertices)

        collisions = torch.tensor([
            [self._collides(pose, obstacle_polygons[step], obstacle_num_vertices[step])
             for step, pose in enumerate(candidate_poses)]
            for candidate_poses in ego_poses
        ], dtype=torch.float32, device=self.settings.device)
        step_weights = torch.tensor(self.settings.collision_step_weights, device=self.settings.device)
        return (collisions * step_weights).sum(dim=1), self._candidate_actions(candidates, ego_pos, ego_yaw)

    def _ego_footprint(self, pose: np.ndarray) -> np.ndarray:
        x, y, yaw = pose
        return compute_vehicle_polygon(x, y, yaw, self.vehicle.l, self.vehicle.w, np.zeros((4, 2)))

    def _collides(self, pose: np.ndarray, polygons: np.ndarray, num_vertices: np.ndarray) -> bool:
        """Whether the ego footprint at ``pose`` intersects any of the obstacle footprints."""
        ego = self._ego_footprint(pose)
        return any(polygon_collision(ego, polygon[:count])
                   for polygon, count in zip(polygons, num_vertices) if count > 0)

    def save_failure_rollout_plot(self, output_path: str | Path) -> bool:
        """Plot the latest selected candidate's ego footprints against the predicted obstacles."""
        if not self._last_candidate_total_costs or self._last_rollout is None:
            return False
        import matplotlib.pyplot as plt

        ego_poses, obstacle_polygons, obstacle_num_vertices = self._last_rollout
        candidate_poses = ego_poses[int(np.argmin(self._last_candidate_total_costs))]
        view_radius = self.scenario_drawer.VIEW_SIZE_DEFAULT / 4.0
        model = self.settings.obstacle_motion_model.upper()

        figure, axes = plt.subplots(1, len(candidate_poses), figsize=(4 * len(candidate_poses), 4), squeeze=False)
        for step, (axis, pose) in enumerate(zip(axes[0], candidate_poses)):
            for polygon, count in zip(obstacle_polygons[step], obstacle_num_vertices[step]):
                if count > 0:
                    axis.add_patch(plt.Polygon(polygon[:count], color="gray"))
            collides = self._collides(pose, obstacle_polygons[step], obstacle_num_vertices[step])
            axis.add_patch(plt.Polygon(self._ego_footprint(pose), color="red" if collides else "tab:blue"))
            axis.set_xlim(pose[0] - view_radius, pose[0] + view_radius)
            axis.set_ylim(pose[1] - view_radius, pose[1] + view_radius)
            axis.set_aspect("equal")
            axis.set_title(f"{model}-Geo prediction {step + 1}/{len(candidate_poses)}")

        figure.suptitle(f"{model}-Geo rollout before episode failure (red: predicted collision)")
        figure.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return True
