import importlib.util
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.state import InitialState

from fiss_plus_planner.planners.common.scenario.frenet import FrenetState, FrenetTrajectory
from fiss_plus_planner.planners.common.utils import transform_points_to_ego
from fiss_plus_planner.planners.common.vehicle.vehicle import Vehicle
from fiss_plus_planner.planners.fop_plus_planner import FopPlusPlanner
from fiss_plus_planner.planners.frenet_optimal_planner import FrenetOptimalPlannerSettings, Stats
from fiss_plus_planner.planners.sparse_planning.scenario_drawer import ScenarioDrawer


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FopPlusWMSelectionSettings(FrenetOptimalPlannerSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5,
                 scenario_dir: str = "", scenario_file: str = ""):
        super().__init__(num_width, num_speed, num_t)
        self.scenario_dir = scenario_dir
        self.scenario_file = scenario_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.world_model_history = 3
        self.world_model_rollout = 5
        self.world_model_action_horizon = 5
        self.world_model_batch_size = 16
        self.world_model_risk_step_weights = (0.2, 0.2, 0.2, 0.2, 0.2)
        self.world_model_starting_point = 8
        self.world_model_front = 20
        self.world_model_width = 5
        # Combined score includes world-model risks, path quality, and Frenet dynamics costs.
        self.world_model_collision_weight = 1.0
        self.world_model_lateral_weight = 1.0
        self.world_model_progress_weight = 1e-3
        self.reference_resample_ds = 2.0
        self.reference_path_lookahead_m = ScenarioDrawer.VIEW_SIZE_DEFAULT / 2.0
        self.reference_path_length = 10
        self.img_size = 256

        project_root = Path(__file__).parent.parent.parent
        self.world_model_weights_path = (
            project_root / "trajectory_planner_world_model" / "World_Model" / "best_world_model.pt"
        )


class FopPlusWMSelectionPlanner(FopPlusPlanner):
    """One-second FOP+ planner ranked by world-model and Frenet trajectory costs."""

    def __init__(self, planner_settings: FopPlusWMSelectionSettings, ego_vehicle: Vehicle,
                 obstacles_array=None, obstacles_num_vertices=None):
        super().__init__(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices)
        # The base FOP+ planner defaults to 0.5 seconds; the world-model rollout needs
        # the longer candidate sequence to supply its full sequence of planned actions.
        self.settings.min_t = 1.0
        self.settings.max_t = 1.0
        self.settings.tick_t = 0.1
        risk_weights = torch.as_tensor(
            self.settings.world_model_risk_step_weights,
            dtype=torch.float32,
            device=self.settings.device,
        )
        if risk_weights.numel() != self.settings.world_model_rollout:
            raise ValueError(
                "world_model_risk_step_weights must contain one value per rollout step"
            )
        if torch.any(risk_weights < 0) or not torch.isclose(
            risk_weights.sum(), torch.tensor(1.0, device=self.settings.device)
        ):
            raise ValueError("World-model risk weights must be nonnegative and sum to 1")
        self._world_model_risk_step_weights = risk_weights

        self.scenario_drawer = ScenarioDrawer(
            self.settings.scenario_file,
            self.settings.scenario_dir,
            obstacles_array=self.obstacles_array,
            obstacles_num_vertices=self.obstacles_num_vertices,
        )

        project_root = Path(__file__).parent.parent.parent
        world_model_dir = project_root / "trajectory_planner_world_model"
        world_model_module = _load_module_from_path(
            "fiss_plus_planner._fop_plus_world_model",
            world_model_dir / "models" / "world_model.py",
        )
        self.world_model = world_model_module.WorldModel(
            num_attention_blocks=1,
            dim=256,
            in_channels_state=self.settings.world_model_history,
            input_dim_action=2,
            trajectory_len=self.settings.world_model_action_horizon,
            out_channels=1,
            pretrained=True,
            img_size=(self.settings.img_size, self.settings.img_size),
        ).to(self.settings.device)
        self.world_model.load_state_dict(torch.load(
            str(self.settings.world_model_weights_path), map_location=self.settings.device
        ))
        self.world_model.eval()

        reward_module = _load_module_from_path(
            "fiss_plus_planner._fop_plus_world_model_reward",
            world_model_dir / "world_model_evaluation" / "reward_function.py",
        )
        self.collision_risk = reward_module.CollisionRisk(
            starting_point_x=self.settings.world_model_starting_point,
            front=self.settings.world_model_front,
            width=self.settings.world_model_width,
        )

        self.lateral_deviation_per_sample = reward_module.lateral_deviation_per_sample
        self.progress_loss_per_sample = reward_module.progress_loss_per_sample

        if str(world_model_dir) not in sys.path:
            sys.path.append(str(world_model_dir))
        self.inference_module = _load_module_from_path(
            "fiss_plus_planner._fop_plus_cvae_inference",
            world_model_dir / "inference.py",
        )
        self.image_history: List[Tuple[int, Image.Image]] = []
        self.ref_lane_pts_global: np.ndarray | None = None
        self._failure_rollout_actions: torch.Tensor | None = None
        self._failure_rollout_images: List[Image.Image] = []
        self._last_candidate_collision_risks: List[float] = []
        self._last_candidate_lateral_costs: List[float] = []
        self._last_candidate_progress_costs: List[float] = []
        self._last_candidate_total_costs: List[float] = []
        self._last_scoring_time_step: int | None = None

    def generate_frenet_frame(self, centerline_pts: np.ndarray):
        cubic_spline, ref_lane_pts = super().generate_frenet_frame(centerline_pts)
        self.ref_lane_pts_global = ref_lane_pts
        return cubic_spline, ref_lane_pts

    def _update_image_history(self, time_step_now: int, current_state: InitialState) -> List[Image.Image]:
        image = self.scenario_drawer.create_scenario_img_at_time_step(
            time_step_now, current_state
        ).convert("L")
        self.image_history.append((time_step_now, image))

        if time_step_now >= 2:
            return [self.image_history[-3][1], self.image_history[-2][1], self.image_history[-1][1]]
        if time_step_now == 1:
            return [self.image_history[-2][1], self.image_history[-1][1], self.image_history[-1][1]]
        return [self.image_history[-1][1]] * self.settings.world_model_history

    def _candidate_actions(self, candidates: List[FrenetTrajectory], ego_pos: np.ndarray,
                           ego_yaw: float) -> torch.Tensor:
        """Convert each global trajectory to the ego-frame action deltas expected by the model."""
        required_steps = self.settings.world_model_action_horizon + self.settings.world_model_rollout - 1
        action_batches = []

        for candidate in candidates:
            global_xy = np.column_stack((candidate.x, candidate.y))
            local_xy = transform_points_to_ego(global_xy, ego_pos, ego_yaw)
            previous = np.vstack((np.zeros((1, 2), dtype=float), local_xy[:-1]))
            actions = local_xy - previous

            if len(actions) == 0:
                actions = np.zeros((1, 2), dtype=float)
            if len(actions) < required_steps:
                actions = np.vstack((
                    actions,
                    np.repeat(actions[-1:], required_steps - len(actions), axis=0),
                ))
            else:
                actions = actions[:required_steps]
            action_batches.append(actions)

        return torch.as_tensor(np.stack(action_batches), dtype=torch.float32, device=self.settings.device)

    def _reference_path_deltas(self, ego_pos: np.ndarray, ego_yaw: float) -> torch.Tensor:
        """Prepare reference points: ego frame, path ahead,
        2m arc-length resampling, then point-to-delta conversion and length-10 truncation."""
        ref_local = transform_points_to_ego(
            self.ref_lane_pts_global[:, :2], ego_pos, ego_yaw
        )
        start_idx = int(np.argmin(np.linalg.norm(ref_local, axis=1)))
        ref_ahead = ref_local[start_idx:]
        if len(ref_ahead) < 2:
            raise ValueError("The reference path has fewer than two points ahead of the ego")

        distances = np.linalg.norm(np.diff(ref_ahead, axis=0), axis=1)
        cumulative_distances = np.concatenate(([0.0], np.cumsum(distances)))
        ref_ahead = ref_ahead[
            cumulative_distances <= self.settings.reference_path_lookahead_m
        ]
        if len(ref_ahead) < 2:
            raise ValueError("The reference path is too short to resample")

        required_length = (
            self.settings.reference_path_length + 1
        ) * self.settings.reference_resample_ds
        path_length = float(np.linalg.norm(np.diff(ref_ahead, axis=0), axis=1).sum())
        if path_length < required_length:
            direction = ref_ahead[-1] - ref_ahead[-2]
            norm = np.linalg.norm(direction)
            if norm > 1e-9:
                direction /= norm
                extra_count = int(np.ceil(
                    (required_length - path_length) / self.settings.reference_resample_ds
                ))
                extra = ref_ahead[-1] + np.outer(
                    np.arange(1, extra_count + 1) * self.settings.reference_resample_ds,
                    direction,
                )
                ref_ahead = np.vstack((ref_ahead, extra))

        ref_points = torch.as_tensor(
            ref_ahead, dtype=torch.float32, device=self.settings.device
        )
        ref_points = self.inference_module.resample_path(
            ref_points, ds=self.settings.reference_resample_ds
        )
        previous_points = torch.cat((
            torch.zeros((1, 2), dtype=ref_points.dtype, device=ref_points.device),
            ref_points[:-1],
        ), dim=0)
        return (ref_points - previous_points)[:self.settings.reference_path_length].unsqueeze(0)
    

    def correct_reference_frame(self, actions, step, length):
        """Return ``length`` XY-delta actions in the ego frame at trajectory point ``step``.

        ``actions`` are consecutive displacements in the initial ego frame. Integrate
        them to positions, move the selected future path so its current point is the
        origin, rotate it into the current ego frame, then difference the positions
        again. The first returned action is therefore zero, as in the training data's
        point-to-delta conversion from an ego-origin starting point.
        """
        if actions.ndim not in (2, 3) or actions.shape[-1] != 2:
            raise ValueError("actions must have shape (N, 2) or (B, N, 2)")
        batched = actions.ndim == 3
        if not batched:
            actions = actions.unsqueeze(0)

        batch_size, num_actions, _ = actions.shape
        if length < 1 or step < 0 or step + length > num_actions:
            raise ValueError(
                f"Need {length} points from step {step}, "
                f"but trajectory has only {num_actions} actions."
            )

        # The dataset stores XY displacements, not absolute points. Recover the
        # trajectory points before translating/rotating the path into the new frame.
        positions = torch.cumsum(actions, dim=1)
        selected_positions = positions[:, step:step + length]
        relative_positions = selected_positions - positions[:, step:step + 1]

        if step == 0:
            # Input images at rollout step zero already use the initial ego frame.
            heading = actions.new_zeros(batch_size)
        else:
            # Heading at the current point is the direction of its next path segment.
            # At the end of the path, fall back to the preceding segment.
            if step + 1 < num_actions:
                direction = positions[:, step + 1] - positions[:, step]
            else:
                direction = positions[:, step] - positions[:, step - 1]
            heading = torch.atan2(direction[:, 1], direction[:, 0])

        c = torch.cos(heading)
        s = torch.sin(heading)
        rotation = torch.stack((
            torch.stack((c, s), dim=-1),
            torch.stack((-s, c), dim=-1),
        ), dim=1)
        local_positions = torch.bmm(relative_positions, rotation.transpose(1, 2))

        previous_positions = torch.cat((
            torch.zeros_like(local_positions[:, :1]),
            local_positions[:, :-1],
        ), dim=1)
        local_actions = local_positions - previous_positions
        return local_actions if batched else local_actions.squeeze(0)
    

    def _score_candidates(self, candidates: List[FrenetTrajectory], images: List[Image.Image],
                          ego_pos: np.ndarray, ego_yaw: float
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self._candidate_actions(candidates, ego_pos, ego_yaw)
        image_tensor = torch.cat(
            [transforms.ToTensor()(image) for image in images], dim=0
        ).unsqueeze(0).to(self.settings.device)
        collision_scores = []
        action_horizon = self.settings.world_model_action_horizon

        with torch.inference_mode():
            for start in range(0, len(candidates), self.settings.world_model_batch_size):
                end = min(start + self.settings.world_model_batch_size, len(candidates))
                batch_actions = actions[start:end]
                batch_size = end - start
                state = image_tensor.expand(batch_size, -1, -1, -1).clone()
                initial_collision_risk = self.collision_risk(state[:, -1:])
                rollout_collision_risks = []

                for step in range(self.settings.world_model_rollout):
                    action = self.correct_reference_frame(
                        batch_actions, step, action_horizon
                    )
                    prediction, _ = self.world_model(state, action)
                    collision_risk = torch.stack([
                        self.collision_risk(prediction[i:i + 1])
                        for i in range(batch_size)
                    ]) - initial_collision_risk
                    rollout_collision_risks.append(collision_risk)
                    state = torch.cat((state[:, 1:], prediction), dim=1)

                per_step_collision_risks = torch.stack(rollout_collision_risks, dim=1)
                collision_scores.append(
                    (per_step_collision_risks * self._world_model_risk_step_weights).sum(dim=1)
                )

            return torch.cat(collision_scores), actions

    def _truncate_trajectory(self, trajectory: FrenetTrajectory, point_count: int) -> FrenetTrajectory:
        """Keep only the output prefix while leaving the full candidate available for scoring."""
        if point_count <= 0 or point_count > len(trajectory.t):
            raise ValueError(
                f"Cannot keep {point_count} points from a trajectory with {len(trajectory.t)} points"
            )

        for attribute in (
            "t", "s", "s_d", "s_dd", "s_ddd", "d", "d_d", "d_dd", "d_ddd",
            "x", "y", "yaw",
        ):
            setattr(trajectory, attribute, getattr(trajectory, attribute)[:point_count])

        interval_count = point_count - 1
        for attribute in ("ds", "c"):
            setattr(trajectory, attribute, getattr(trajectory, attribute)[:interval_count])
        trajectory.c_d = trajectory.c_d[:max(interval_count - 1, 0)]
        trajectory.c_dd = trajectory.c_dd[:max(interval_count - 2, 0)]

        trajectory.end_state = trajectory.frenet_state_at_time_step(point_count - 1)
        if trajectory.sampling_param is not None:
            trajectory.sampling_param.d = trajectory.d[-1]
            trajectory.sampling_param.s_d = trajectory.s_d[-1]
            trajectory.sampling_param.t = trajectory.t[-1]
        return trajectory

    def save_failure_rollout_plot(self, output_path: str | Path) -> bool:
        """Save all predicted BEV frames from the latest selected candidate's rollout."""
        if self._failure_rollout_actions is None or len(self._failure_rollout_images) != self.settings.world_model_history:
            return False

        image_tensor = torch.cat(
            [transforms.ToTensor()(image) for image in self._failure_rollout_images], dim=0
        ).unsqueeze(0).to(self.settings.device)
        actions = self._failure_rollout_actions.unsqueeze(0)
        state = image_tensor
        predictions = []

        with torch.inference_mode():
            for step in range(self.settings.world_model_rollout):
                action = self.correct_reference_frame(
                    actions, step, self.settings.world_model_action_horizon
                )
                prediction, _ = self.world_model(state, action)
                predictions.append(prediction[0, 0].detach().cpu().numpy())
                state = torch.cat((state[:, 1:], prediction), dim=1)

        if len(predictions) != self.settings.world_model_rollout:
            return False

        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(
            1, len(predictions), figsize=(4 * len(predictions), 4), squeeze=False
        )
        for step, (axis, predicted_bev) in enumerate(zip(axes[0], predictions), start=1):
            axis.imshow(predicted_bev, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(f"World-model prediction {step}/{self.settings.world_model_rollout}")
            axis.axis("off")

        figure.suptitle("World-model rollout before episode failure")
        figure.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return True

    def print_failure_candidate_costs(self) -> None:
        """Print each failed-cycle candidate's risk, path costs, and combined objective."""
        if not self._last_candidate_collision_risks:
            print("No world-model candidate costs are available for the failed cycle")
            return

        print(
            "World-model candidate costs "
            f"(planning time step {self._last_scoring_time_step}):"
        )
        print(
            "  trajectory   collision risk    lateral dev."
            "   lateral x wt.   neg. progress    progress x wt."
            "   total"
        )
        for candidate_index, (risk, lateral, progress, total) in enumerate(zip(
            self._last_candidate_collision_risks,
            self._last_candidate_lateral_costs,
            self._last_candidate_progress_costs,
            self._last_candidate_total_costs,
        )):
            weighted_lateral = self.settings.world_model_lateral_weight * lateral
            weighted_progress = self.settings.world_model_progress_weight * progress
            print(
                f"  {candidate_index:10d} {risk:16.8f} {lateral:15.8f}"
                f" {weighted_lateral:15.8f} {progress:15.8f}"
                f" {weighted_progress:16.8f} {total:12.8f}"
            )

    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list[Obstacle],
             time_step_now: int = 0, initial_state: InitialState = None) -> FrenetTrajectory:
        self.stats = Stats()
        self.settings.highest_speed = max_target_speed
        self._last_candidate_collision_risks = []
        self._last_candidate_lateral_costs = []
        self._last_candidate_progress_costs = []
        self._last_candidate_total_costs = []
        self._last_scoring_time_step = None
        self._failure_rollout_actions = None
        self._failure_rollout_images = []

        image_start = time.time()
        images = self._update_image_history(time_step_now, initial_state)
        self.time_image_generation = time.time() - image_start

        candidates = self.calc_frenet_paths(frenet_state)
        candidates = self.calc_global_paths(candidates)
        self.stats.num_trajs_generated = len(candidates)
        if not candidates:
            return None

        ego_pos = np.asarray(initial_state.position, dtype=float)
        ego_yaw = float(initial_state.orientation)
        collision_risk_scores, candidate_actions = self._score_candidates(
            candidates, images, ego_pos, ego_yaw
        )
        self._last_candidate_collision_risks = collision_risk_scores.detach().cpu().tolist()
        self._last_scoring_time_step = time_step_now
        reference_deltas = self._reference_path_deltas(ego_pos, ego_yaw)
        lateral_scores = self.lateral_deviation_per_sample(
            candidate_actions,
            reference_deltas.expand(candidate_actions.shape[0], -1, -1),
        )
        progress_scores = self.progress_loss_per_sample(
            candidate_actions,
            reference_deltas.expand(candidate_actions.shape[0], -1, -1),
        )
        combined_scores = (
            self.settings.world_model_collision_weight * collision_risk_scores
            + self.settings.world_model_lateral_weight * lateral_scores
            + self.settings.world_model_progress_weight * progress_scores
        )
        self._last_candidate_lateral_costs = lateral_scores.detach().cpu().tolist()
        self._last_candidate_progress_costs = progress_scores.detach().cpu().tolist()
        self._last_candidate_total_costs = combined_scores.detach().cpu().tolist()
        selected_index = int(combined_scores.argmin().item())
        selected = candidates[selected_index]
        self._failure_rollout_actions = candidate_actions[selected_index].detach().clone()
        self._failure_rollout_images = [image.copy() for image in images]
        selected = self._truncate_trajectory(
            selected, self.settings.world_model_action_horizon
        )

        passed = self.check_constraints([selected])
        self.stats.num_trajs_validated += 1
        if not passed:
            return None

        safe = self.check_collision_multithread(passed, time_step_now)
        self.stats.num_collison_checks += 1
        if not safe:
            return None

        self.cost_function.calc_cost(
            safe, max_target_speed, self.obstacles_array,
            self.obstacles_num_vertices, time_step_now,
        )
        self.best_traj = safe[0]
        return self.best_traj
