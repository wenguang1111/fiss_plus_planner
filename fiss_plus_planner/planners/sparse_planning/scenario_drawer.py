import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.collections import LineCollection
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
    VIEW_SIZE_DEFAULT = 105.0 # highest_speed 13.4 x 5s < 70; 70*2=140: left and right
    COLOR_BLACK = "#000000"
    COLOR_GRAY = "#808080"
    COLOR_LightGray = "#D3D3D3"
    GENERATE_GIF = True
    GIF_DURATION_MS = 100
    GIF_LOOP = 0
    img_dim = 256

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
        self.ego_params.vehicle_shape.occupancy.shape.facecolor = self.COLOR_BLACK
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
            transforms.Resize((self.img_dim, self.img_dim)),
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
        self._fig = None
        self._ax = None
        self._canvas = None

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

        images = [] if self.GENERATE_GIF else None
        for time_step, ego_state in enumerate(ego_state_list):
            img = self._render_frame(
                ego_state=ego_state,
                time_step=time_step,
                # highest_speed=highest_speed,
            )
            img.save(output_dir / f"{time_step}.{image_format}")
            if images is not None:
                images.append(img)

        if images:
            gif_path = output_dir / f"{self.scenario_name}.gif"
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                optimize=True,
                duration=self.GIF_DURATION_MS,
                loop=self.GIF_LOOP,
            )

    def create_scenario_img_at_time_step(
        self,
        time_step: int,
        ego_state: State,
    ) -> Image.Image:

        img = self._render_frame(
                ego_state=ego_state,
                time_step=time_step,
                # highest_speed=highest_speed,
        )
        return img

    def generate_image_at_time_step(
        self,
        time_step: int,
        ego_state: State,
        highest_speed: Optional[float] = None,
    ) -> Tensor:
        img = self._render_frame(
            ego_state=ego_state,
            time_step=time_step,
            # highest_speed=highest_speed,
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
        # highest_speed: Optional[float],
    ) -> Image.Image:
        view_size = self.VIEW_SIZE_DEFAULT
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(4, 4), dpi=64, facecolor="white")
            self._fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            self._canvas = FigureCanvas(self._fig)
        ax = self._ax
        ax.clear()
        ax.set_facecolor("white")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        ego_x, ego_y = ego_state.position
        yaw = 0.0 if ego_state.orientation is None else float(ego_state.orientation)
        transform = self._build_ego_transform(np.array([ego_x, ego_y]), yaw)

        ax.set_xlim(-view_size / 2.0, view_size / 2.0)
        ax.set_ylim(-view_size / 2.0, view_size / 2.0)

        self._draw_lanelet_boundaries(ax, transform)
        self._draw_lane_ahead(ax, ego_state, transform)
        self._draw_obstacles(ax, time_step, transform)
        self._draw_ego(ax)
        # self._draw_speed_arrow(ax, ego_state, highest_speed)

        self._canvas.draw()
        width, height = self._canvas.get_width_height()
        buf = np.frombuffer(self._canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        return Image.fromarray(buf)

    def _draw_lane_ahead(self, ax, ego_state: State, transform: np.ndarray):
        if self.ref_ego_lane_pts is None:
            return
        lane_xy = self.ref_ego_lane_pts[:, :2]
        if lane_xy.size == 0:
            return
        ego_pos = np.array(ego_state.position, dtype=float)
        distances = np.linalg.norm(lane_xy - ego_pos, axis=1)
        start_idx = int(np.argmin(distances))
        lane_ahead = self._apply_transform(lane_xy[start_idx:], transform)
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

    def _draw_lanelet_boundaries(self, ax, transform: np.ndarray):
        lanelet_network = getattr(self.scenario, "lanelet_network", None)
        if lanelet_network is None:
            return
        segments = []
        for lanelet in lanelet_network.lanelets:
            left = self._apply_transform(np.asarray(lanelet.left_vertices, dtype=float), transform)
            right = self._apply_transform(np.asarray(lanelet.right_vertices, dtype=float), transform)
            if left.shape[0] >= 2:
                segments.append(left)
            if right.shape[0] >= 2:
                segments.append(right)
        if segments:
            collection = LineCollection(
                segments,
                colors=self.COLOR_LightGray,
                linewidths=self.LINE_WIDTH,
                zorder=5,
            )
            ax.add_collection(collection)

    def _draw_obstacles(self, ax, time_step: int, transform: np.ndarray):
        num_vertices_row = self.obstacles_num_vertices[time_step]
        for obs_idx, num_vertices in enumerate(num_vertices_row):
            if num_vertices < 3:
                continue
            coords = self._apply_transform(
                self.obstacles_array[time_step, obs_idx, :num_vertices, :],
                transform,
            )
            patch = Polygon(
                coords,
                closed=True,
                facecolor=self.COLOR_GRAY,
                edgecolor=self.COLOR_GRAY,
                linewidth=self.LINE_WIDTH,
                zorder=30,
            )
            ax.add_patch(patch)

    def _draw_ego(self, ax):
        ego_x, ego_y = 0.0, 0.0
        yaw = 0.0
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
        dx = length
        dy = 0.0
        start_x = self.vehicle_length / 2.0
        start_y = 0.0
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

    def _build_ego_transform(self, ego_pos: np.ndarray, ego_yaw: float) -> np.ndarray:
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        transform = np.eye(3, dtype=float)
        transform[:2, :2] = rot
        transform[:2, 2] = -rot @ ego_pos
        return transform

    def _apply_transform(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        pts = np.asarray(points, dtype=float)
        ones = np.ones((pts.shape[0], 1), dtype=float)
        hom = np.hstack((pts, ones))
        transformed = hom @ transform.T
        return transformed[:, :2]
