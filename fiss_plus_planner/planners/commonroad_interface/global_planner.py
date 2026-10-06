import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.utility.visualization import visualize_route
from commonroad.scenario.scenario import Scenario

def _outermost_boundaries(lanelet, llnet):
    """Left and right boundary polylines of the whole same-direction roadway holding `lanelet`.

    Walks the adjacency chain outward while the neighbour runs the same way, so the result spans every
    lane the ego may legally occupy and stops at oncoming lanes. The visited set guards against maps
    whose adjacency links form a cycle.
    """
    left = lanelet
    seen = {left.lanelet_id}
    while left.adj_left is not None and left.adj_left_same_direction and left.adj_left not in seen:
        seen.add(left.adj_left)
        left = llnet.find_lanelet_by_id(left.adj_left)

    right = lanelet
    seen = {right.lanelet_id}
    while right.adj_right is not None and right.adj_right_same_direction and right.adj_right not in seen:
        seen.add(right.adj_right)
        right = llnet.find_lanelet_by_id(right.adj_right)

    return left.left_vertices, right.right_vertices


def _distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest point on `polyline`, clamped to its segments."""
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    denom = np.einsum('ij,ij->i', ab, ab)
    denom = np.where(denom > 0.0, denom, 1.0)
    ap = points[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum('nmj,mj->nm', ap, ab) / denom, 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(points[:, None, :] - proj, axis=2).min(axis=1)


def _extent_inside(area, point: np.ndarray, direction: np.ndarray, max_length: float = 20.0) -> float:
    """Length of the ray from `point` along `direction` until it first leaves `area`."""
    ray = LineString([point, point + max_length * direction]).intersection(area)
    for part in getattr(ray, 'geoms', [ray]):
        if part.length > 0.0 and part.distance(Point(point)) < 1e-6:
            return part.length
    return 0.0


def _merge_extents(lanelet, llnet, left: np.ndarray, right: np.ndarray):
    """Left / right extents of `lanelet` widened by the lanes merging with it.

    Lanelets sharing a successor with `lanelet` overlap it at the merge, and a vehicle coming from one
    of them occupies both (BEL_Brussels-51 starts there). Each extent is the distance along the normal
    of the centerline point that stays inside the union of the merging lanelets, kept where it exceeds
    the roadway extent.
    """
    ids = {p for s in lanelet.successor for p in llnet.find_lanelet_by_id(s).predecessor} | {lanelet.lanelet_id}
    if len(ids) == 1:
        return left, right
    area = unary_union([llnet.find_lanelet_by_id(i).polygon.shapely_object.buffer(1e-3) for i in ids])
    center = lanelet.center_vertices
    tangent = np.gradient(center, axis=0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / np.linalg.norm(tangent, axis=1)[:, None]
    left = np.maximum(left, [_extent_inside(area, p, n) for p, n in zip(center, normal)])
    right = np.maximum(right, [_extent_inside(area, p, -n) for p, n in zip(center, normal)])
    return left, right


def extend_centerline(centerline: np.ndarray, length: float, step: float = 1.0) -> np.ndarray:
    """Route centerline (x, y, yaw, width, left, right) continued straight past its end by `length` m.

    The CommonRoad map ends where the scenario ends, so close to the route end every sampled trajectory
    would run beyond the reference line and count as off-road. The extension keeps the last heading and
    road profile, as if the road went on.
    """
    last = centerline[-1]
    offsets = step * np.arange(1, int(np.ceil(length / step)) + 1)
    extension = np.repeat(last[None, :], len(offsets), axis=0)
    extension[:, 0] += offsets * np.cos(last[2])
    extension[:, 1] += offsets * np.sin(last[2])
    return np.vstack((centerline, extension))


# Traffic-sign elements whose first additional value is a maximum speed [m/s]
_SPEED_LIMIT_SIGNS = ('MAX_SPEED', 'MAX_SPEED_ZONE_START')


def _route_speed_limits(lanelets, llnet):
    """Speed limit [m/s] valid on each route lanelet, None before the first limit sign.

    A limit sign applies from its lanelet onward along the route until the next limit sign; if a
    lanelet carries several limits, the lowest one is used.
    """
    limits, current = [], None
    for lanelet in lanelets:
        values = [float(el.additional_values[0])
                  for sign_id in lanelet.traffic_signs
                  for el in llnet.find_traffic_sign_by_id(sign_id).traffic_sign_elements
                  if el.traffic_sign_element_id.name in _SPEED_LIMIT_SIGNS and el.additional_values]
        if values:
            current = min(values)
        limits.append(current)
    return limits


class GlobalPlan(object):
    def __init__(self):
        self.lanelets = None
        self.lanelet_centerlines = None
        self.concat_centerline = None
        self.speed_limits = None
        self.required_speeds = None
        # self.widths = None

class GlobalPlanner(object):
    
    #  NETWORKX: uses built-in functions from the networkx package, tends to change lane earlier
    #  PRIORITY_QUEUE: uses A-star search to find routes, lane change maneuver depends on the heuristic cost
    def plan_global_route(self, scenario: Scenario, planning_problem, view_route: bool = False):
        ''' Plan the global route for a given scenario and problem
        
        Parameters
        ----------
        
        `scenerio` (`commonroad.scenario.Scenerio`): the CommonRoad scenerio
        `planning_problem` (`commonroad.planning_problem.PlanningProblemSet`): the CommonRoad planning problem

        Returns
        -------
        (`commonroad_interface.global_planner.GlobalPlan`): planned global route information
        '''
        
        # initialize the route planner
        route_planner = RoutePlanner(scenario, planning_problem)

        # plan routes, retrieve the first route
        route = route_planner.plan_routes().retrieve_first_route()
        
        # Assemble the GlobalPlan
        global_plan = GlobalPlan()
        
        # generate the lanelet network
        llnet = scenario.lanelet_network
        laneletlist = route.lanelet_ids
        
        if len(llnet.find_lanelet_by_id(laneletlist[len(laneletlist)-1]).successor) != 0:
            lastlanelet = llnet.find_lanelet_by_id(laneletlist[len(laneletlist)-1]).successor[0]
        else:
            lastlanelet = None
        global_plan.lanelets = [llnet.find_lanelet_by_id(laneletlist[i]) for i in range(len(laneletlist))]
        if lastlanelet is not None:
            lastlanelet = llnet.find_lanelet_by_id(lastlanelet)
            global_plan.lanelets.append(lastlanelet)
        global_plan.lanelet_centerlines = np.array([lanelet.center_vertices for lanelet in global_plan.lanelets],dtype=object)
        global_plan.speed_limits = _route_speed_limits(global_plan.lanelets, llnet)
        
        # Concatenate the centerlines into one np.ndarray, and remove duplicates
        concat_centerline = np.concatenate(global_plan.lanelet_centerlines)
        concat_centerline = np.array(concat_centerline,dtype=float)
        _, unqiue_indices = np.unique(concat_centerline, return_index=True, axis=0)
        concat_centerline = concat_centerline[np.sort(unqiue_indices)]
        
        # Calculate the orientation of each centerline point
        diff_y = np.diff(concat_centerline[:, 1].flatten())
        diff_x = np.diff(concat_centerline[:, 0].flatten())
        yaws = np.arctan2(diff_y, diff_x)
        yaws = np.append(yaws, [yaws[-1]], axis=0)
        
        # Calcuate the width of each centerline point, and remove duplicates
        widths = np.concatenate(
            [np.array([np.linalg.norm(lanelet.left_vertices[i] - lanelet.right_vertices[i]) for i in range(len(lanelet.left_vertices))])
            for lanelet in global_plan.lanelets]
            )[np.sort(unqiue_indices)]

        # Distance from the route centerline out to each edge of the whole same-direction roadway, not
        # just the ego lane. Kept as two separate columns because a road is routinely asymmetric about
        # the lane the route follows (ESP_Barcelona has two lanes to the left and none to the right),
        # so a single symmetric half-width cannot describe where the vehicle may go.
        # Lanes merging into a route lanelet widen it where they overlap (_merge_extents).
        extents = [_merge_extents(lanelet, llnet,
                                  _distance_to_polyline(lanelet.center_vertices, _outermost_boundaries(lanelet, llnet)[0]),
                                  _distance_to_polyline(lanelet.center_vertices, _outermost_boundaries(lanelet, llnet)[1]))
                   for lanelet in global_plan.lanelets]
        left_extents = np.concatenate([left for left, _ in extents])[np.sort(unqiue_indices)]
        right_extents = np.concatenate([right for _, right in extents])[np.sort(unqiue_indices)]

        global_plan.concat_centerline = np.hstack((
            concat_centerline, yaws[:, np.newaxis], widths[:, np.newaxis],
            left_extents[:, np.newaxis], right_extents[:, np.newaxis]))
        
        # Visualization
        if view_route:
            print('Global Planning Results:')
            print('Passing through:', global_plan.lanelet_centerlines.shape[0], 'lanelets')
            print('Contains:', global_plan.concat_centerline.shape, 'lane points')
            print(global_plan.concat_centerline)
            visualize_route(route, draw_route_lanelets=True, draw_reference_path=True, size_x=6)
        
        return global_plan
