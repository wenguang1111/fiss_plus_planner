import io
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_params import DynamicObstacleParams
from commonroad.scenario.obstacle import ObstacleType
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.state import State

from torch import Tensor
import torchvision.transforms as transforms

from fiss_plus_planner.planners.common.geometry.cubic_spline import CubicSpline2D
from fiss_plus_planner.planners.commonroad_interface.global_planner import GlobalPlanner


class ScenarioDrawer:
    """Render CommonRoad scenarios to image files."""

    ARROW_LENGTH_CONSTANT = 0.2  # meters
    LINE_WIDTH = 1.0
    LANE_DASH_LENGTH = 2.0
    LANE_DASH_GAP = 2.0
    ARROW_WIDTH = 0.006
    VIEW_SIZE_DEFAULT = 140.0 # highest_speed 13.4 x 5s < 70; 70*2=140: left and right
    COLOR_BLACK = "#000000"
    COLOR_GRAY = "#808080"

    def __init__(
        self,
        scenario_name: str,
        scenario_dir: str | Path,
        save_dir: str | Path = None,
        ref_ego_lane_pts: Optional[np.ndarray] = None,
        vehicle_params: Optional[object] = None,
        obstacles_array: np.ndarray = None,
        obstacles_num_vertices: np.ndarray = None,
    ):
        self.save_dir = Path(save_dir) / "imgs" if save_dir is not None else None
        self.scenario_dir = Path(scenario_dir)
        self.scenario_name = scenario_name
        self.ego_params = DynamicObstacleParams()
        self.ego_params.draw_icon = True
        self.ego_params.vehicle_shape.occupancy.shape.facecolor = "#ff0000"
        self.ego_params.draw_icon = True
        self.ego_id = None
        self.ego_type = ObstacleType.CAR
        self.shape = Rectangle(width=1.8, length=4.3)

        scenario_file = self.scenario_name
        if not scenario_file.endswith(".xml"):
            scenario_file = scenario_file + ".xml"

        self.scenario, self.planning_problem_set = CommonRoadFileReader(
            os.path.join(self.scenario_dir, scenario_file)
        ).open()
        if save_dir is not None:
            os.makedirs(self.save_dir / self.scenario_name, exist_ok=True)

        self._transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        self.ref_ego_lane_pts = ref_ego_lane_pts
        if self.ref_ego_lane_pts is None:
            self.ref_ego_lane_pts = self._compute_ref_ego_lane_pts()

        if vehicle_params is not None:
            self.vehicle_length = float(vehicle_params.l)
            self.vehicle_width = float(vehicle_params.w)
        else:
            self.vehicle_length = float(self.shape.length)
            self.vehicle_width = float(self.shape.width)

        self.obstacles_array = obstacles_array
        self.obstacles_num_vertices = obstacles_num_vertices

    def save_scenario_imgs(
        self,
        ego_state_list: Iterable[State],
        highest_speed: float,
        image_format: str = "png",
    ):
        if self.save_dir is None or ego_state_list is None:
            return

        output_dir = self.save_dir / self.scenario_name
        os.makedirs(output_dir, exist_ok=True)

        for time_step, ego_state in enumerate(ego_state_list):
            img = self._render_frame(
                ego_state=ego_state,
                time_step=time_step,
                highest_speed=highest_speed,
            )
            img.save(output_dir / f"{time_step}.{image_format}")

    def generate_image_at_time_step(
        self,
        time_step: int,
        ego_state: State,
        highest_speed: Optional[float] = None,
    ) -> Tensor:
        img = self._render_frame(
            ego_state=ego_state,
            time_step=time_step,
            highest_speed=highest_speed,
        )
        return self._transform(img).unsqueeze(0)

    def _compute_ref_ego_lane_pts(self) -> Optional[np.ndarray]:
        if self.planning_problem_set is None:
            return None
        try:
            planning_problem = next(iter(self.planning_problem_set.planning_problem_dict.values()))
            global_plan = GlobalPlanner().plan_global_route(self.scenario, planning_problem)
            centerline_pts = global_plan.concat_centerline[:, :2]
            return self._generate_ref_lane_pts(centerline_pts)
        except Exception as exc:
            print(f"Warning: failed to compute ref_ego_lane_pts: {exc}")
            return None

    def _generate_ref_lane_pts(self, centerline_pts: np.ndarray) -> Optional[np.ndarray]:
        if centerline_pts is None or len(centerline_pts) < 2:
            return None
        spline = CubicSpline2D(centerline_pts[:, 0], centerline_pts[:, 1])
        s = np.arange(0, spline.s[-1], 0.1)
        ref_xy = [spline.calc_position(i_s) for i_s in s]
        ref_yaw = [spline.calc_yaw(i_s) for i_s in s]
        ref_rk = [spline.calc_curvature(i_s) for i_s in s]
        return np.column_stack((ref_xy, ref_yaw, ref_rk))

    def _render_frame(
        self,
        ego_state: State,
        time_step: int,
        highest_speed: Optional[float],
    ) -> Image.Image:
        view_size = self.VIEW_SIZE_DEFAULT
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300, facecolor="white")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_facecolor("white")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        ego_x, ego_y = ego_state.position
        ax.set_xlim(ego_x - view_size / 2.0, ego_x + view_size / 2.0)
        ax.set_ylim(ego_y - view_size / 2.0, ego_y + view_size / 2.0)

        self._draw_lanelet_boundaries(ax)
        self._draw_lane_ahead(ax, ego_state)
        self._draw_obstacles(ax, time_step)
        self._draw_ego(ax, ego_state)
        self._draw_speed_arrow(ax, ego_state, highest_speed)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches=None, pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _draw_lane_ahead(self, ax, ego_state: State):
        if self.ref_ego_lane_pts is None:
            return
        lane_xy = self.ref_ego_lane_pts[:, :2]
        if lane_xy.size == 0:
            return
        ego_pos = np.array(ego_state.position, dtype=float)
        distances = np.linalg.norm(lane_xy - ego_pos, axis=1)
        start_idx = int(np.argmin(distances))
        lane_ahead = lane_xy[start_idx:]
        if len(lane_ahead) < 2:
            return
        ax.plot(
            lane_ahead[:, 0],
            lane_ahead[:, 1],
            color=self.COLOR_GRAY,
            linewidth=self.LINE_WIDTH,
            linestyle=(0, (self.LANE_DASH_LENGTH, self.LANE_DASH_GAP)),
            zorder=10,
        )

    def _draw_lanelet_boundaries(self, ax):
        lanelet_network = getattr(self.scenario, "lanelet_network", None)
        if lanelet_network is None:
            return
        for lanelet in lanelet_network.lanelets:
            left = np.asarray(lanelet.left_vertices, dtype=float)
            right = np.asarray(lanelet.right_vertices, dtype=float)
            if left.shape[0] >= 2:
                ax.plot(
                    left[:, 0],
                    left[:, 1],
                    color=self.COLOR_GRAY,
                    linewidth=self.LINE_WIDTH,
                    zorder=5,
                )
            if right.shape[0] >= 2:
                ax.plot(
                    right[:, 0],
                    right[:, 1],
                    color=self.COLOR_GRAY,
                    linewidth=self.LINE_WIDTH,
                    zorder=5,
                )

    def _draw_obstacles(self, ax, time_step: int):
        num_vertices_row = self.obstacles_num_vertices[time_step]
        for obs_idx, num_vertices in enumerate(num_vertices_row):
            if num_vertices < 3:
                continue
            coords = self.obstacles_array[time_step, obs_idx, :num_vertices, :]
            patch = Polygon(
                coords,
                closed=True,
                fill=False,
                edgecolor=self.COLOR_GRAY,
                linewidth=self.LINE_WIDTH,
                zorder=20,
            )
            ax.add_patch(patch)

    def _draw_ego(self, ax, ego_state: State):
        ego_x, ego_y = ego_state.position
        yaw = 0.0 if ego_state.orientation is None else float(ego_state.orientation)
        half_l = self.vehicle_length / 2.0
        half_w = self.vehicle_width / 2.0
        corners = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners = corners @ rot.T + np.array([ego_x, ego_y])
        patch = Polygon(
            corners,
            closed=True,
            facecolor=self.COLOR_BLACK,
            edgecolor=self.COLOR_BLACK,
            linewidth=self.LINE_WIDTH,
            zorder=30,
        )
        ax.add_patch(patch)

    def _draw_speed_arrow(self, ax, ego_state: State, highest_speed: Optional[float]):
        if highest_speed is None or highest_speed <= 0:
            return
        speed = 0.0 if ego_state.velocity is None else float(ego_state.velocity)

        #size definition
        length = self.VIEW_SIZE_DEFAULT * self.ARROW_LENGTH_CONSTANT * (speed / highest_speed)

        if length <= 0.0:
            return
        yaw = 0.0 if ego_state.orientation is None else float(ego_state.orientation)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        dx = length * cos_yaw
        dy = length * sin_yaw
        start_x = ego_state.position[0] + cos_yaw * (self.vehicle_length / 2.0)
        start_y = ego_state.position[1] + sin_yaw * (self.vehicle_length / 2.0)
        ax.quiver(
            start_x,
            start_y,
            dx,
            dy,
            scale_units="xy",
            angles="xy",
            scale=1,
            width=self.ARROW_WIDTH,
            color=self.COLOR_GRAY,
            zorder=35,
        )
