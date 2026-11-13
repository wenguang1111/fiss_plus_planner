"""Intelligent Driver Model (IDM) prediction of obstacle footprints.

Runs ``IDMAgent`` from the ``commonroad-prediction`` submodule (crpred) on all obstacles together, re-started from
their current states every planning cycle.
"""
import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle, Obstacle
from commonroad.scenario.state import CustomState, InitialState, State
from commonroad.scenario.trajectory import Trajectory

from fiss_plus_planner.planners.obstacle_motion_prediction import ObstacleMotionPredictor

CRPRED_DIR = Path(__file__).resolve().parents[2] / "commonroad-prediction"
# Smallest net gap [m] to the leader used in the IDM interaction term.
MIN_LEADER_GAP = 0.1
if str(CRPRED_DIR) not in sys.path:
    sys.path.append(str(CRPRED_DIR))

from commonroad_clcs.pycrccosy import (  # noqa: E402
    CartesianProjectionDomainError, CurvilinearProjectionDomainLateralError,
    CurvilinearProjectionDomainLongitudinalError,
)
from commonroad_clcs.util import compute_orientation_from_polyline, compute_pathlength_from_polyline  # noqa: E402
from crpred.advanced_models.idm_agent import IDMAgent  # noqa: E402
from crpred.utility.config import IDMParams  # noqa: E402


class _CycleScenario:
    """What crpred's agents read from a ``Scenario``: the lanelet network, which records the obstacles on each
    lanelet per time step, the time step size, and the obstacles by id."""

    def __init__(self, lanelet_network: LaneletNetwork, dt: float, obstacles: Dict[int, DynamicObstacle]):
        self.lanelet_network = lanelet_network
        self.dt = dt
        self._obstacles = obstacles

    def obstacle_by_id(self, obstacle_id: int) -> Optional[DynamicObstacle]:
        return self._obstacles.get(obstacle_id)


_PROJECTION_DOMAIN_ERRORS = (CartesianProjectionDomainError, CurvilinearProjectionDomainLateralError,
                             CurvilinearProjectionDomainLongitudinalError)


class _ValueErrorClcs:
    """A commonroad_clcs coordinate system whose projection-domain errors are ``ValueError``s, with memoized
    projections onto the curvilinear frame.

    crpred was written against coordinate systems that raise ``ValueError`` outside their projection domain and
    handles that case with ``except ValueError`` (e.g. it skips a leader candidate it cannot project); the
    commonroad_clcs 2025 errors derive from ``Exception`` only and would escape those handlers.

    crpred's leader search projects every obstacle on the agent's lane, for every agent and step; agents on the
    same lane share this coordinate system, so the projections of a position are computed once.

    Whether commonroad_clcs accepts a longitudinal coordinate within a few millimetres of either end of the
    reference path varies between processes (seen in 2025.1.0 to 2025.2.0), which made the prediction differ
    from run to run; coordinates within ``END_MARGIN`` of the ends are therefore always outside the domain.
    """

    MAX_MEMOIZED = 100_000
    END_MARGIN = 0.01  # [m]

    def __init__(self, clcs):
        self._clcs = clcs
        self._length = clcs.length()
        self._curvilinear: Dict[Tuple[float, float], Optional[Tuple[float, float]]] = {}

    def __getattr__(self, name):
        return getattr(self._clcs, name)

    def convert_to_curvilinear_coords(self, x: float, y: float):
        key = (float(x), float(y))
        if key not in self._curvilinear:
            if len(self._curvilinear) >= self.MAX_MEMOIZED:
                self._curvilinear.clear()
            try:
                self._curvilinear[key] = self._clcs.convert_to_curvilinear_coords(x, y)
            except _PROJECTION_DOMAIN_ERRORS:
                self._curvilinear[key] = None
        result = self._curvilinear[key]
        if result is None:
            raise ValueError(f"({x}, {y}) is outside the projection domain")
        return result.copy()

    def convert_to_cartesian_coords(self, p_lon: float, p_lat: float):
        if not self.END_MARGIN < p_lon < self._length - self.END_MARGIN:
            raise ValueError(f"longitudinal coordinate {p_lon} is at or beyond an end of the reference path")
        try:
            return self._clcs.convert_to_cartesian_coords(p_lon, p_lat)
        except _PROJECTION_DOMAIN_ERRORS as exc:
            raise ValueError(str(exc)) from exc


class _IDMAgent(IDMAgent):
    """crpred's ``IDMAgent``, with
    - centerline coordinate systems that report projection-domain errors as ``ValueError`` (``_ValueErrorClcs``),
    - the leader searched only on the merged lanelets that contain the agent,
    - the distance to the leader taken bumper to bumper, as in IDM, instead of centre to centre,
    - the coordinate systems, and their reference paths' arc length and orientation, shared between agents and
      planning cycles instead of recomputed (same results)."""

    def __init__(self, obstacle: DynamicObstacle, sc: _CycleScenario, config: IDMParams, clcs_cache: Dict,
                 reference_path_cache: Dict):
        self._clcs_cache = clcs_cache
        self._reference_path_cache = reference_path_cache
        super().__init__(obstacle, sc, config)

    def _update_leader_at_time_step(self, time_step: int):
        """crpred's leader search over only the merged lanelets whose coordinate system contains the agent, with
        the distance to the leader taken as the IDM net gap.

        crpred projects the agent onto every merged lanelet it started on and drops the agent when one of them,
        e.g. a branch it did not take, cannot hold its position (a leader candidate it skips in that case).
        crpred measures the leader distance between the two vehicles' centres; IDM's gap runs from the agent's
        front to the leader's rear, i.e. that distance minus half of each vehicle's length.
        """
        position = self.state_at_step(time_step, self._obstacle, self._traj_state_list).position
        merged_lanelets = self._list_lanelets_merged
        self._list_lanelets_merged = [lanelet for lanelet in merged_lanelets
                                      if self._contains(self._dict_clcs[lanelet.lanelet_id], position)]
        try:
            super()._update_leader_at_time_step(time_step)
        finally:
            self._list_lanelets_merged = merged_lanelets
        if self._id_agent_leader:
            leader = self._current_scenario.obstacle_by_id(self._id_agent_leader)
            half_lengths = 0.5 * (self._obstacle.obstacle_shape.length + leader.obstacle_shape.length)
            # Overlapping footprints (gap <= 0) brake at the deceleration limit instead of dividing by zero.
            self._dis_to_leader = max(self._dis_to_leader - half_lengths, MIN_LEADER_GAP)

    @staticmethod
    def _contains(clcs: _ValueErrorClcs, position: np.ndarray) -> bool:
        try:
            clcs.convert_to_curvilinear_coords(position[0], position[1])
        except ValueError:
            return False
        return True

    def _convert_to_curvilinear_state(self, state: State, clcs):
        """crpred's ``Agent._convert_to_curvilinear_state``, which spends most of the prediction time recomputing
        the reference path's arc length and orientation, with both cached per coordinate system."""
        if clcs not in self._reference_path_cache:
            reference_path = np.array(clcs.reference_path())
            self._reference_path_cache[clcs] = (compute_pathlength_from_polyline(reference_path),
                                                compute_orientation_from_polyline(reference_path))
        length_ref, orientation_ref = self._reference_path_cache[clcs]
        p_lon, p_lat = clcs.convert_to_curvilinear_coords(state.position[0], state.position[1])
        o_ref = np.interp(p_lon, length_ref, orientation_ref)
        v_lon = state.velocity * np.cos(state.orientation - o_ref)
        v_lat = state.velocity * np.sin(state.orientation - o_ref)
        return p_lon, p_lat, v_lon, v_lat, o_ref

    def _merge_lanelet_and_create_clcs(self, lanelet: Lanelet, merge_predecessors: bool = True):
        key = (lanelet.lanelet_id, merge_predecessors)
        if key not in self._clcs_cache:
            try:
                merged_lanelets, clcs_by_lanelet, merge_ids = super()._merge_lanelet_and_create_clcs(
                    lanelet, merge_predecessors)
                clcs_by_lanelet = {i: _ValueErrorClcs(clcs) for i, clcs in clcs_by_lanelet.items()}
                self._clcs_cache[key] = (merged_lanelets, clcs_by_lanelet, merge_ids)
            # commonroad_clcs rejects some centerlines, e.g. with a turn of more than 90 degrees (AssertionError)
            except Exception:  # noqa: BLE001
                self._clcs_cache[key] = None
        if self._clcs_cache[key] is None:
            raise ValueError(f"no curvilinear coordinate system for lanelet {lanelet.lanelet_id}")
        return self._clcs_cache[key]


class IDMObstaclePredictor(ObstacleMotionPredictor):
    """Predict obstacle footprints with crpred's IDM agents, which follow their lane and keep their distance to
    the obstacle ahead on it.

    Every cycle all dynamic obstacles present at the current time step are simulated together from their current
    states, so a follower reacts to its leader's predicted motion. The ego vehicle is not part of the simulation:
    obstacles do not react to it. Each agent's desired speed is the speed limit of its lanelet if the lanelet has
    one, and its current speed otherwise. An obstacle that cannot be an IDM agent (e.g. off the lanelet network,
    or no coordinate system for its lanelet) is predicted with lane-relative constant velocity instead; an agent
    that leaves IDM's domain mid-horizon continues in a straight line at its last speed. Static obstacles stay in
    place and, as in crpred, are not leaders.

    crpred's ``IDMPredictor.predict`` starts from the scenario's initial states and stops with an exception on
    our scenarios, so this class drives the agents itself. It works around these crpred defects: agents look up
    their leader's state in ``obstacle.prediction``, which ``IDMPredictor`` never fills (it is written here after
    every step); projection-domain errors of commonroad_clcs 2025 are not ``ValueError``s and escape crpred's
    handlers; and an agent is dropped when one of its merged lanelets cannot project it. Unlike crpred, the
    distance to the leader is IDM's net gap, not the distance between vehicle centres (see ``_IDMAgent``).
    """

    def __init__(self, lanelet_network: LaneletNetwork, obstacles: List[Obstacle], max_vertices: int, dt: float):
        super().__init__(lanelet_network, obstacles, max_vertices, dt, motion_model="cv")
        # crpred records the obstacles on each lanelet per time step in the lanelet objects themselves: keep that
        # in a private copy of the network, cleared every cycle.
        self._agent_lanelet_network = copy.deepcopy(lanelet_network)
        self._clcs_cache: Dict = {}
        self._reference_path_cache: Dict = {}
        self._speed_limits = self._lanelet_speed_limits(lanelet_network)

    @staticmethod
    def _lanelet_speed_limits(lanelet_network: LaneletNetwork) -> Dict[int, float]:
        """Lowest MAX_SPEED sign [m/s] of each lanelet that has one. Matched by sign name, since the sign ids of
        commonroad's ``TrafficSignInterpreter`` depend on the country and miss e.g. the Belgian scenarios."""
        limits = {}
        for lanelet in lanelet_network.lanelets:
            values = [float(element.additional_values[0])
                      for sign_id in lanelet.traffic_signs
                      for element in lanelet_network.find_traffic_sign_by_id(sign_id).traffic_sign_elements
                      if element.traffic_sign_element_id.name == "MAX_SPEED"]
            if values:
                limits[lanelet.lanelet_id] = min(values)
        return limits

    def _desired_speed(self, state: State) -> float:
        lanelet_ids = self.lanelet_network.find_lanelet_by_position([state.position])[0]
        limits = [self._speed_limits[i] for i in lanelet_ids if i in self._speed_limits]
        return min(limits) if limits else max(state.velocity, 0.0)

    def _predict_obstacle_poses(self, time_step: int, num_steps: int) -> Dict[int, np.ndarray]:
        for lanelet in self._agent_lanelet_network.lanelets:
            lanelet.dynamic_obstacles_on_lanelet = {}

        poses: Dict[int, np.ndarray] = {}
        proxies: Dict[int, Tuple[int, DynamicObstacle, State]] = {}  # obstacle id -> (index, proxy, state)
        for obs_idx, obstacle, state in self._present_obstacles(time_step):
            if isinstance(obstacle, DynamicObstacle):
                proxies[obstacle.obstacle_id] = (obs_idx, self._proxy(obstacle, state, time_step), state)
            else:
                poses[obs_idx] = self._predict_poses(obstacle, state, time_step, num_steps)

        scenario = _CycleScenario(self._agent_lanelet_network, self.dt,
                                  {obstacle_id: proxy for obstacle_id, (_, proxy, _) in proxies.items()})
        for obstacle_id, (_, proxy, _) in proxies.items():
            self._assign_to_lanelets(obstacle_id, proxy.occupancy_at_time(time_step).shape, time_step)

        agents: Dict[int, _IDMAgent] = {}
        scripted: Dict[int, List[CustomState]] = {}  # obstacles moved without IDM, all their future states
        for obstacle_id, (obs_idx, proxy, state) in proxies.items():
            try:
                agents[obstacle_id] = _IDMAgent(proxy, scenario, IDMParams(v_desired=self._desired_speed(state)),
                                                self._clcs_cache, self._reference_path_cache)
            except Exception:  # noqa: BLE001 - any crpred failure falls back to constant velocity
                cv_poses = self._predict_poses(self.obstacles[obs_idx], state, time_step, num_steps)
                scripted[obstacle_id] = self._states_from_poses(cv_poses, state.velocity, time_step)
                proxy.prediction = self._prediction(proxy, scripted[obstacle_id])

        for step in range(1, num_steps + 1):
            step_time = time_step + step
            for obstacle_id, agent in list(agents.items()):
                try:
                    agent.step_forward(step_time)
                except Exception:  # noqa: BLE001 - any other crpred failure ends the agent as crpred's own do
                    agent._valid = False
                if len(agent._traj_state_list) < step:  # the agent left IDM's domain
                    del agents[obstacle_id]
                    last = agent._traj_state_list[-1] if agent._traj_state_list else proxies[obstacle_id][2]
                    scripted[obstacle_id] = agent._traj_state_list + self._straight_line_states(
                        last, step_time, num_steps - step + 1)
            # Leaders are looked up at the previous step: publish this step's states before the next one.
            for obstacle_id, agent in agents.items():
                proxy = proxies[obstacle_id][1]
                proxy.prediction = self._prediction(proxy, agent._traj_state_list)
                self._assign_to_lanelets(obstacle_id, agent.set_ids_lanelets_current, step_time)
            for obstacle_id, states in scripted.items():
                proxy = proxies[obstacle_id][1]
                if proxy.prediction is None or proxy.prediction.final_time_step < time_step + num_steps:
                    proxy.prediction = self._prediction(proxy, states)
                self._assign_to_lanelets(obstacle_id, proxy.occupancy_at_time(step_time).shape, step_time)

        for obstacle_id, (obs_idx, proxy, _) in proxies.items():
            states = proxy.prediction.trajectory.state_list
            poses[obs_idx] = np.array([(*s.position, s.orientation) for s in states])
        return poses

    @staticmethod
    def _proxy(obstacle: DynamicObstacle, state: State, time_step: int) -> DynamicObstacle:
        """The obstacle re-started from its state at ``time_step``, without a future trajectory."""
        initial_state = InitialState(
            time_step=time_step, position=np.array(state.position), orientation=state.orientation,
            velocity=state.velocity, acceleration=getattr(state, "acceleration", None) or 0.0,
            yaw_rate=0.0, slip_angle=0.0,
        )
        return DynamicObstacle(obstacle.obstacle_id, obstacle.obstacle_type, obstacle.obstacle_shape, initial_state)

    def _assign_to_lanelets(self, obstacle_id: int, shape_or_lanelet_ids, time_step: int) -> None:
        """Record the obstacle on the lanelets it occupies (given by id, or by its shape) at ``time_step``."""
        if isinstance(shape_or_lanelet_ids, (set, frozenset, list)):
            lanelet_ids = shape_or_lanelet_ids
        else:
            lanelet_ids = self._agent_lanelet_network.find_lanelet_by_shape(shape_or_lanelet_ids)
        for lanelet_id in lanelet_ids:
            lanelet = self._agent_lanelet_network.find_lanelet_by_id(lanelet_id)
            lanelet.dynamic_obstacles_on_lanelet.setdefault(time_step, set()).add(obstacle_id)

    @staticmethod
    def _prediction(proxy: DynamicObstacle, states: List[State]) -> TrajectoryPrediction:
        return TrajectoryPrediction(Trajectory(states[0].time_step, list(states)), proxy.obstacle_shape)

    # The states carry the attributes of crpred's IDM states: a trajectory needs the same attributes in all states.
    @staticmethod
    def _states_from_poses(poses: np.ndarray, velocity: float, time_step: int) -> List[CustomState]:
        return [CustomState(time_step=time_step + step, position=np.array(pose[:2]), orientation=pose[2],
                            velocity=velocity, acceleration=0.0)
                for step, pose in enumerate(poses, start=1)]

    def _straight_line_states(self, last: State, first_time_step: int, num_steps: int) -> List[CustomState]:
        """``num_steps`` states from ``first_time_step`` on, continuing ``last`` in a straight line at its speed."""
        direction = np.array([np.cos(last.orientation), np.sin(last.orientation)])
        speed = max(last.velocity, 0.0)
        return [CustomState(time_step=first_time_step + k,
                            position=np.asarray(last.position) + speed * self.dt * (k + 1) * direction,
                            orientation=last.orientation, velocity=speed, acceleration=0.0)
                for k in range(num_steps)]
