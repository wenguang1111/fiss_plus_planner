import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
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
    ARROW_WIDTH = 0.006
    VIEW_SIZE_DEFAULT = 105.0/1.2 # highest_speed 13.4 x 5s < 70; 70*2=140: left and right
    COLOR_BLACK = "#000000"
    COLOR_WHITE = "#FFFFFF"
    COLOR_GRAY = "#808080"
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
            img = self._render_frame(ego_state, *self._scenario_obstacles_at(time_step))
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

        return self._render_frame(ego_state, *self._scenario_obstacles_at(time_step))

    def create_scenario_img_with_obstacles(
        self,
        ego_state: State,
        obstacle_polygons: np.ndarray,
        obstacle_num_vertices: np.ndarray,
    ) -> Image.Image:
        """Render the BEV at ``ego_state`` with caller-supplied obstacles, e.g. predicted ones.

        ``obstacle_polygons`` has shape (num_obstacles, max_vertices, 2) in global coordinates and
        ``obstacle_num_vertices`` gives each obstacle's vertex count, as in one time step of
        ``obstacles_array`` / ``obstacles_num_vertices``.
        """
        return self._render_frame(ego_state, obstacle_polygons, obstacle_num_vertices)

    def generate_image_at_time_step(
        self,
        time_step: int,
        ego_state: State,
        highest_speed: Optional[float] = None,
    ) -> Tensor:
        img = self._render_frame(ego_state, *self._scenario_obstacles_at(time_step))
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

    def _scenario_obstacles_at(self, time_step: int):
        return self.obstacles_array[time_step], self.obstacles_num_vertices[time_step]

    def _create_figure(self):
        view_size = self.VIEW_SIZE_DEFAULT
        self._fig, self._ax = plt.subplots(figsize=(4, 4), dpi=64, facecolor="black")
        self._fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self._canvas = FigureCanvas(self._fig)
        self._ax.set_facecolor("black")
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.axis("off")
        self._ax.set_xlim(-view_size / 2.0, view_size / 2.0)
        self._ax.set_ylim(-view_size / 2.0, view_size / 2.0)

    # Geometry farther than this from the ego cannot reach the square view: half its diagonal plus a
    # margin far larger than the outline stroke (~0.3 m at this scale).
    CULL_RADIUS = VIEW_SIZE_DEFAULT / 2.0 * np.sqrt(2.0) + 2.0

    def _render_frame(
        self,
        ego_state: State,
        obstacle_polygons: np.ndarray,
        obstacle_num_vertices: np.ndarray,
    ) -> Image.Image:
        if self._fig is None:
            self._create_figure()
            # Patches are created once and then only moved (set_xy) or hidden between frames;
            # creating and removing matplotlib artists every frame dominated the render time.
            self._lanelet_patches, self._obstacle_patches, self._ego_patch = [], [], None
            self._lanelet_geometry = self._prepare_lanelet_geometry()
        ax = self._ax
        # The ego is re-added last every frame so that it is still drawn after all obstacles.
        if self._ego_patch is not None:
            self._ego_patch.remove()

        ego_xy = np.array(ego_state.position, dtype=float)
        yaw = 0.0 if ego_state.orientation is None else float(ego_state.orientation)
        transform = self._build_ego_transform(np.array([ego_xy[0], ego_xy[1]]), yaw)

        self._draw_lanelet_boundaries(ax, transform, ego_xy)
        self._draw_obstacles(ax, obstacle_polygons, obstacle_num_vertices, transform, ego_xy)
        self._ego_patch = self._draw_ego(ax)
        # self._draw_speed_arrow(ax, ego_state, highest_speed)

        self._canvas.draw()
        width, height = self._canvas.get_width_height()
        buf = np.frombuffer(self._canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        return Image.fromarray(buf)

    def _prepare_lanelet_geometry(self) -> list:
        """(left, right, (xmin, xmax, ymin, ymax)) in world coordinates for every drawable lanelet."""
        lanelet_network = getattr(self.scenario, "lanelet_network", None)
        if lanelet_network is None:
            return []
        geometry = []
        for lanelet in lanelet_network.lanelets:
            left = np.asarray(lanelet.left_vertices, dtype=float)
            right = np.asarray(lanelet.right_vertices, dtype=float)
            if left.shape[0] < 2 or right.shape[0] < 2:
                continue
            both = np.vstack([left, right])
            bounds = (both[:, 0].min(), both[:, 0].max(), both[:, 1].min(), both[:, 1].max())
            geometry.append((left, right, bounds))
        return geometry

    def _near_ego(self, bounds, ego_xy: np.ndarray) -> bool:
        """Whether a box (xmin, xmax, ymin, ymax) comes within CULL_RADIUS of the ego."""
        xmin, xmax, ymin, ymax = bounds
        dx = max(xmin - ego_xy[0], 0.0, ego_xy[0] - xmax)
        dy = max(ymin - ego_xy[1], 0.0, ego_xy[1] - ymax)
        return dx * dx + dy * dy <= self.CULL_RADIUS * self.CULL_RADIUS

    def _update_patches(self, ax, patches: list, polygons: list, color: str, zorder: int):
        """Show ``polygons`` (ego-frame vertex arrays) on the pooled ``patches``, in order; hide the rest."""
        for i, coords in enumerate(polygons):
            if i == len(patches):
                patch = Polygon(coords, closed=True, facecolor=color, edgecolor=color,
                                linewidth=self.LINE_WIDTH, zorder=zorder)
                # Not ax.add_patch: its data-limit update is useless with fixed view limits.
                ax.add_artist(patch)
                patches.append(patch)
            else:
                patches[i].set_xy(coords)
                patches[i].set_visible(True)
        for patch in patches[len(polygons):]:
            patch.set_visible(False)

    def _draw_lanelet_boundaries(self, ax, transform: np.ndarray, ego_xy: np.ndarray):
        polygons = []
        for left, right, bounds in self._lanelet_geometry:
            if not self._near_ego(bounds, ego_xy):
                continue
            left_local = self._apply_transform(left, transform)
            right_local = self._apply_transform(right, transform)
            polygons.append(np.vstack([left_local, right_local[::-1]]))
        self._update_patches(ax, self._lanelet_patches, polygons, self.COLOR_GRAY, zorder=5)

    def _draw_obstacles(self, ax, obstacle_polygons: np.ndarray, obstacle_num_vertices: np.ndarray,
                        transform: np.ndarray, ego_xy: np.ndarray):
        polygons = []
        for polygon, num_vertices in zip(obstacle_polygons, obstacle_num_vertices):
            if num_vertices < 3:
                continue
            vertices = polygon[:num_vertices, :]
            bounds = (vertices[:, 0].min(), vertices[:, 0].max(), vertices[:, 1].min(), vertices[:, 1].max())
            if not self._near_ego(bounds, ego_xy):
                continue
            polygons.append(self._apply_transform(vertices, transform))
        self._update_patches(ax, self._obstacle_patches, polygons, self.COLOR_WHITE, zorder=30)

    def _draw_ego(self, ax) -> Polygon:
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
            facecolor=self.COLOR_WHITE,
            edgecolor=self.COLOR_WHITE,
            linewidth=self.LINE_WIDTH,
            zorder=30,
        )
        ax.add_artist(patch)
        return patch

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
