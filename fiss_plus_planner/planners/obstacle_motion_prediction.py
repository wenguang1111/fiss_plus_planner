"""Lane-relative constant-velocity / constant-acceleration prediction of obstacle footprints."""
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.scenario.obstacle import Obstacle, StaticObstacle
from commonroad.scenario.state import State
from commonroad_dc.costs.route_matcher import create_cosy_from_lanelet, get_orientation_at_position
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

# Length [m] up to which a lanelet is merged with its successors to build the centerline coordinate system.
MERGED_LANELET_LENGTH = 300.0


def constant_velocity_speeds(velocity: float, acceleration: float, dt: float, num_steps: int) -> np.ndarray:
    """Mean speed over each of the ``num_steps`` steps of a constant-velocity motion (``acceleration`` unused)."""
    return np.full(num_steps, velocity)


def constant_acceleration_speeds(velocity: float, acceleration: float, dt: float, num_steps: int) -> np.ndarray:
    """Mean speed over each of the ``num_steps`` steps of a constant-acceleration motion.

    A decelerating obstacle stops when its speed reaches zero and does not reverse.
    """
    t = dt * np.arange(num_steps + 1)
    if acceleration < 0.0:
        t = np.minimum(t, max(velocity, 0.0) / -acceleration)
    distance = velocity * t + 0.5 * acceleration * t ** 2
    return np.diff(distance) / dt


MOTION_MODELS: Dict[str, Callable[[float, float, float, int], np.ndarray]] = {
    "cv": constant_velocity_speeds,
    "ca": constant_acceleration_speeds,
}


class ObstacleMotionPredictor:
    """Predict obstacle footprints by extrapolating each obstacle's current state with a motion model.

    The motion model ("cv" or "ca") gives the obstacle's speed along its heading over each future step.
    The obstacle moves along that heading, kept constant relative to the centerline of its lanelet
    merged with its successors: the step displacement is integrated in the curvilinear (s, d) frame and
    the predicted yaw is the centerline orientation plus the initial relative heading. Obstacles off
    the lanelet network, outside the centerline's projection domain, or on a lanelet whose centerline
    yields no coordinate system move in a straight line along their current heading instead; static
    obstacles stay where they are. For "cv" this reproduces crpred's ``ConstantVelocityLinearPredictor``.
    """

    def __init__(self, lanelet_network: LaneletNetwork, obstacles: List[Obstacle],
                 max_vertices: int, dt: float, motion_model: str = "cv"):
        self.lanelet_network = lanelet_network
        self.obstacles = obstacles
        self.max_vertices = max_vertices
        self.dt = dt
        self.motion_model = motion_model
        self._step_speeds = MOTION_MODELS[motion_model]
        self._footprints = [self._footprint(obstacle) for obstacle in obstacles]
        self._cosy_by_lanelet: Dict[int, Optional[CurvilinearCoordinateSystem]] = {}

    def predict_footprints(self, time_step: int, num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Predict footprints at ``time_step + 1 .. time_step + num_steps`` from the states at ``time_step``.

        Returns polygons of shape (num_steps, num_obstacles, max_vertices, 2) in global coordinates
        and vertex counts of shape (num_steps, num_obstacles), the layout of ``obstacles_array``.
        Obstacles without a state at ``time_step`` get zero vertices.
        """
        polygons = np.zeros((num_steps, len(self.obstacles), self.max_vertices, 2))
        num_vertices = np.zeros((num_steps, len(self.obstacles)), dtype=np.int32)
        for obs_idx, poses in self._predict_obstacle_poses(time_step, num_steps).items():
            footprint = self._footprints[obs_idx]
            for step, (x, y, yaw) in enumerate(poses):
                c, s = np.cos(yaw), np.sin(yaw)
                polygons[step, obs_idx, :len(footprint)] = footprint @ np.array([[c, s], [-s, c]]) + (x, y)
                num_vertices[step, obs_idx] = len(footprint)
        return polygons, num_vertices

    def _present_obstacles(self, time_step: int) -> Iterator[Tuple[int, Obstacle, State]]:
        """(index, obstacle, state) of the obstacles that have a state and a footprint at ``time_step``."""
        for obs_idx, (obstacle, footprint) in enumerate(zip(self.obstacles, self._footprints)):
            state = obstacle.state_at_time(time_step)
            if state is not None and footprint is not None:
                yield obs_idx, obstacle, state

    def _predict_obstacle_poses(self, time_step: int, num_steps: int) -> Dict[int, np.ndarray]:
        """Predicted (x, y, yaw) rows of shape (num_steps, 3) by obstacle index, for the present obstacles."""
        return {obs_idx: self._predict_poses(obstacle, state, time_step, num_steps)
                for obs_idx, obstacle, state in self._present_obstacles(time_step)}

    def _footprint(self, obstacle: Obstacle) -> Optional[np.ndarray]:
        """Obstacle shape vertices in its own frame, truncated to ``max_vertices``."""
        try:
            coords = np.asarray(obstacle.obstacle_shape.shapely_object.exterior.coords[:-1], dtype=float)
        except AttributeError:
            return None
        return coords[:self.max_vertices]

    def _predict_poses(self, obstacle: Obstacle, state: State, time_step: int,
                       num_steps: int) -> np.ndarray:
        """Predicted (x, y, yaw) rows for the ``num_steps`` steps after ``time_step``."""
        if isinstance(obstacle, StaticObstacle):
            return np.tile((*state.position, state.orientation), (num_steps, 1))
        speeds = self._step_speeds(state.velocity, self._acceleration(obstacle, state, time_step),
                                   self.dt, num_steps)
        poses = self._lane_relative_poses(state, speeds)
        return poses if poses is not None else self._straight_line_poses(state, speeds)

    def _acceleration(self, obstacle: Obstacle, state: State, time_step: int) -> float:
        """The state's acceleration, else the backward difference of the velocity, else zero."""
        acceleration = getattr(state, "acceleration", None)
        if acceleration is not None:
            return acceleration
        previous = obstacle.state_at_time(time_step - 1)
        if previous is None or getattr(previous, "velocity", None) is None:
            return 0.0
        return (state.velocity - previous.velocity) / self.dt

    def _lane_relative_poses(self, state: State, speeds: np.ndarray) -> Optional[np.ndarray]:
        cosy = self._curvilinear_cosy(state.position)
        if cosy is None:
            return None
        lane_yaw = get_orientation_at_position(cosy, state.position)
        if lane_yaw is None:
            return None
        relative_yaw = state.orientation - lane_yaw
        p_lon, p_lat = cosy.convert_to_curvilinear_coords(state.position[0], state.position[1])
        p_lon = self._integrate(p_lon, speeds * np.cos(relative_yaw) * self.dt)
        p_lat = self._integrate(p_lat, speeds * np.sin(relative_yaw) * self.dt)

        poses = np.empty((len(speeds), 3))
        for step, (lon, lat) in enumerate(zip(p_lon, p_lat)):
            try:  # the prediction may leave the centerline's projection domain
                position = cosy.convert_to_cartesian_coords(lon, lat)
            except ValueError:
                return None
            yaw = get_orientation_at_position(cosy, position)
            if yaw is None:
                return None
            poses[step] = (*position, yaw + relative_yaw)
        return poses

    def _straight_line_poses(self, state: State, speeds: np.ndarray) -> np.ndarray:
        yaw = state.orientation
        return np.column_stack((
            self._integrate(state.position[0], speeds * np.cos(yaw) * self.dt),
            self._integrate(state.position[1], speeds * np.sin(yaw) * self.dt),
            np.full(len(speeds), yaw),
        ))

    @staticmethod
    def _integrate(start: float, increments: np.ndarray) -> np.ndarray:
        """``start`` plus the running sum of ``increments``, added step by step."""
        return np.cumsum(np.concatenate(([start], increments)))[1:]

    def _curvilinear_cosy(self, position: np.ndarray) -> Optional[CurvilinearCoordinateSystem]:
        """Centerline coordinate system of the lanelet at ``position`` merged with its successors."""
        lanelet_ids = self.lanelet_network.find_lanelet_by_position([position])[0]
        if not lanelet_ids:
            return None
        lanelet_id = lanelet_ids[0]
        if lanelet_id not in self._cosy_by_lanelet:
            merged_lanelets, _ = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                self.lanelet_network.find_lanelet_by_id(lanelet_id), self.lanelet_network,
                MERGED_LANELET_LENGTH,
            )
            try:
                self._cosy_by_lanelet[lanelet_id] = create_cosy_from_lanelet(merged_lanelets[0])
            except RuntimeError:  # the C++ library cannot build a projection domain for this centerline
                self._cosy_by_lanelet[lanelet_id] = None
        return self._cosy_by_lanelet[lanelet_id]
