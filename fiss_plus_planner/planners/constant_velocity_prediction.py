"""Lane-relative constant-velocity prediction of obstacle footprints.

Wraps ``ConstantVelocityLinearPredictor`` from the ``commonroad-prediction`` submodule (crpred).
"""
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import Obstacle, StaticObstacle
from commonroad.scenario.state import State
from commonroad_dc.costs.route_matcher import create_cosy_from_lanelet, get_orientation_at_position
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

CRPRED_DIR = Path(__file__).parent.parent.parent / "commonroad-prediction"

# crpred imports commonroad_clcs only for the CurvilinearCoordinateSystem type hint; the coordinate
# systems it actually uses come from commonroad_dc. commonroad-clcs is not installed here (crpred
# targets commonroad-drivability-checker >= 2025.1), so register commonroad_dc's class under that name.
try:
    import commonroad_clcs.clcs  # noqa: F401
except ModuleNotFoundError:
    _clcs = types.ModuleType("commonroad_clcs.clcs")
    _clcs.CurvilinearCoordinateSystem = CurvilinearCoordinateSystem
    _clcs_package = types.ModuleType("commonroad_clcs")
    _clcs_package.clcs = _clcs
    sys.modules.update({"commonroad_clcs": _clcs_package, "commonroad_clcs.clcs": _clcs})

if str(CRPRED_DIR) not in sys.path:
    sys.path.append(str(CRPRED_DIR))

from crpred.basic_models.constant_velocity_predictor import ConstantVelocityLinearPredictor  # noqa: E402
from crpred.basic_models.motion_model_predictor import InitialStateValues  # noqa: E402
from crpred.utility.common import get_merged_laneletes_from_position  # noqa: E402


class ConstantVelocityObstaclePredictor:
    """Predict obstacle footprints with crpred's ``ConstantVelocityLinearPredictor``.

    Each obstacle keeps the speed and lane-relative heading it has at the current time step: it moves
    by a constant (s, d) increment along the centerline of its lanelet merged with its successors.
    crpred's ``predict`` always starts from ``initial_state``, so this class builds the predictor's
    initial values from the current state and calls its per-obstacle step directly. Obstacles off
    the lanelet network or outside the centerline's projection domain move in a straight line along
    their heading instead, as are those whose lanelet centerline yields no coordinate system; static
    obstacles stay where they are.
    """

    def __init__(self, lanelet_network: LaneletNetwork, obstacles: List[Obstacle],
                 max_vertices: int, dt: float):
        self.lanelet_network = lanelet_network
        self.obstacles = obstacles
        self.max_vertices = max_vertices
        self.dt = dt
        self._predictor = ConstantVelocityLinearPredictor()
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
        for obs_idx, (obstacle, footprint) in enumerate(zip(self.obstacles, self._footprints)):
            state = obstacle.state_at_time(time_step)
            if state is None or footprint is None:
                continue
            for step, (x, y, yaw) in enumerate(self._predict_poses(obstacle, state, time_step, num_steps)):
                c, s = np.cos(yaw), np.sin(yaw)
                polygons[step, obs_idx, :len(footprint)] = footprint @ np.array([[c, s], [-s, c]]) + (x, y)
                num_vertices[step, obs_idx] = len(footprint)
        return polygons, num_vertices

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
        poses = self._lane_relative_poses(state, time_step, num_steps)
        return poses if poses is not None else self._straight_line_poses(state, num_steps)

    def _lane_relative_poses(self, state: State, time_step: int,
                             num_steps: int) -> Optional[np.ndarray]:
        cosy = self._curvilinear_cosy(state.position)
        if cosy is None:
            return None
        lane_yaw = get_orientation_at_position(cosy, state.position)
        if lane_yaw is None:
            return None
        p_lon, p_lat = cosy.convert_to_curvilinear_coords(state.position[0], state.position[1])
        initial_values = InitialStateValues(
            p_lon=p_lon,
            p_lat=p_lat,
            v=state.velocity,
            orientation_in_ccosy=state.orientation - lane_yaw,
            acceleration=0.0,  # unused by the constant-velocity model
            yaw_rate=0.0,  # unused by the linear model
        )
        try:
            predicted = self._predictor._predict_states(
                initial_values, self.dt, cosy, range(time_step + 1, time_step + 1 + num_steps)
            )
        # The prediction leaves the centerline's projection domain: the coordinate conversion raises
        # ValueError, or get_orientation_at_position swallows it and crpred adds None to the heading.
        except (ValueError, TypeError):
            return None
        return np.array([(*s.position, s.orientation) for s in predicted])

    def _straight_line_poses(self, state: State, num_steps: int) -> np.ndarray:
        distance = state.velocity * self.dt * np.arange(1, num_steps + 1)
        yaw = state.orientation
        return np.column_stack((
            state.position[0] + distance * np.cos(yaw),
            state.position[1] + distance * np.sin(yaw),
            np.full(num_steps, yaw),
        ))

    def _curvilinear_cosy(self, position: np.ndarray) -> Optional[CurvilinearCoordinateSystem]:
        """Centerline coordinate system of the lanelet at ``position``, as crpred builds it."""
        lanelet_ids = self.lanelet_network.find_lanelet_by_position([position])[0]
        if not lanelet_ids:
            return None
        lanelet_id = lanelet_ids[0]
        if lanelet_id not in self._cosy_by_lanelet:
            merged_lanelets, _ = get_merged_laneletes_from_position(self.lanelet_network, position)
            try:
                self._cosy_by_lanelet[lanelet_id] = create_cosy_from_lanelet(merged_lanelets[0])
            except RuntimeError:  # the C++ library cannot build a projection domain for this centerline
                self._cosy_by_lanelet[lanelet_id] = None
        return self._cosy_by_lanelet[lanelet_id]
