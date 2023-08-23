from collections import deque
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.polyline_map_object import \
    NuPlanPolylineMapObject
from nuplan.common.maps.nuplan_map.utils import (
    extract_roadblock_objects, get_distance_between_map_object_and_point)
from nuplan.planning.training.preprocessing.features.raster_utils import \
    _cartesian_to_projective_coords


def get_lane_obj_from_id(
    map_api: AbstractMap,
    lane_id: str,
) -> MapObject:
    """
    Get lane or lane connector object from id.
    """
    lane = map_api.get_map_object(lane_id, SemanticMapLayer.LANE)
    if not lane:
        lane = map_api.get_map_object(lane_id, SemanticMapLayer.LANE_CONNECTOR)
    return lane


def get_roadblock_obj_from_id(
    map_api: AbstractMap,
    road_id: str,
) -> MapObject:
    """
    Get road or road connector object from id.
    """
    road = map_api.get_map_object(road_id, SemanticMapLayer.ROADBLOCK)
    if not road:
        road = map_api.get_map_object(road_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
    return road


def get_current_center_and_side_lane_from_roadblock(state: StateSE2, roadblock_obj: MapObject, need_center_lane = True) -> List[MapObject]:
    """
    Get the current lane that contains the point, leftmost lane and rightmost lane in the given roadblock.
    :param state: reference state for the current lane.
    :param roadblock_obj: current roadblock to get lanes.
    :param need_center_lane: extract center lane if True.
    :return: [(current lane id), leftmost lane id, rightmost lane id].
    """
    lanes = roadblock_obj.interior_edges
    point = state.point
    # if we need to get center lane, a sort on all lanes based on distance is performed
    if need_center_lane:
        lanes.sort(key=lambda map_obj: float(get_distance_between_map_object_and_point(point, map_obj)))

    global_transform = np.linalg.inv(state.as_matrix())
    # By default the map is right-oriented, this makes it top-oriented.
    map_align_transform = R.from_euler(
        'z', 90, degrees=True).as_matrix().astype(
        np.float32)
    transform = map_align_transform @ global_transform
    left_points = []
    right_points = []
    for lane in lanes:
        left_polyline: NuPlanPolylineMapObject = lane.left_boundary
        right_polyline: NuPlanPolylineMapObject = lane.right_boundary
        left_dist = left_polyline.get_nearest_arc_length_from_position(point)
        right_dist = right_polyline.get_nearest_arc_length_from_position(point)
        left_point = left_polyline.linestring.interpolate(left_dist)
        right_point = right_polyline.linestring.interpolate(right_dist)
        left_points.append([left_point.x, left_point.y])
        right_points.append([right_point.x, right_point.y])
    left_coords = (transform @ _cartesian_to_projective_coords(
        np.array(left_points)).T).T[:, :2]
    right_coords = (transform @ _cartesian_to_projective_coords(
        np.array(right_points)).T).T[:, :2]
    left_lane_ind = np.argmin(left_coords[:, 0])
    right_lane_ind = np.argmax(right_coords[:, 0])
    
    if need_center_lane:
        return [lanes[0], lanes[left_lane_ind], lanes[right_lane_ind]]
    else:
        return [lanes[left_lane_ind], lanes[right_lane_ind]]

def _find_center_lanes(
    roads: List[RoadBlockGraphEdgeMapObject],
    center_lanes: List[LaneGraphEdgeMapObject],
    step: int,
) -> List[str]:
    """
    Helper function to find all center lane ids using backtracking.
    """
    if step == len(roads):
        return center_lanes
    
    return_lanes = center_lanes

    all_lane_id_in_road = [lane.id for lane in roads[step].interior_edges]
    # find the next center lane candidates
    center_lane_candidates = center_lanes[-1].outgoing_edges
    center_lane_candidates = [lane for lane in center_lane_candidates if lane.id in all_lane_id_in_road]
    # search every case
    for lane in center_lane_candidates:
        new_center_lanes = _find_center_lanes(roads, center_lanes + [lane], step + 1)
        # we find all lanes
        if len(new_center_lanes) == len(roads):
            return new_center_lanes
        # otherwise return lanes with maximum path that could follow route roadblocks.
        if len(new_center_lanes) > len(return_lanes):
            return_lanes = new_center_lanes
    return return_lanes 

def _search_roadpath_to_target_road(
    initial_road_objs: List[RoadBlockGraphEdgeMapObject],
    target_road_id: str,
    max_depth: int = 15,
) -> Tuple[List[RoadBlockGraphEdgeMapObject], bool]:
    """
    Helper function search a path from initial road objs to target road obj using DFS.
    """
    visited = set()
    # Each element in the queue is a tuple (node, depth, path_so_far)
    queue = deque([(node, 0, [node]) for node in initial_road_objs])
    max_depth_path = [initial_road_objs[0]]
    while queue:
        current_node, depth, path = queue.popleft()
        # Check if current node's id matches the target id
        if current_node.id == target_road_id:
            return path, True
        if depth < max_depth:
            for neighbor in current_node.outgoing_edges:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    if depth + 1 > len(max_depth_path):
                        max_depth_path = new_path
                    queue.append((neighbor, depth + 1, new_path))

    return max_depth_path, False

def _get_future_road_ids(
    initial_road_objs: List[MapObject],
    all_road_ids: List[str],
):
    """
    Get future roadblocks that follow a predefined roadblock lists starting from initial roadblocks.
    """
    cur_road_ids = [road.id for road in initial_road_objs]
    future_road_ids = []

    # normal case
    for i, road_id in enumerate(all_road_ids):
        if road_id in cur_road_ids:
            future_road_ids = all_road_ids[i:]
            return future_road_ids

    # special case, scenario route block ids is wrong
    future_roads, reached = _search_roadpath_to_target_road(initial_road_objs, all_road_ids[0])
    future_road_ids = [road.id for road in future_roads]
    if reached:
        future_road_ids += all_road_ids[1:]
    return future_road_ids

def _remove_consecutive_duplicates(lst: List) -> List:
    """
    Helper function to remove consecutive duplicates in a List.
    """
    if not lst:
        return []
    result = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] != lst[i-1]:  # Compare the current element with the previous one
            result.append(lst[i])
    return result

def get_roadblock_and_lane_ids_from_state(
    map_api: AbstractMap,
    ego_state: EgoState,
    all_road_ids: List[str],
) -> Tuple[List[str], List[List[str]]]:
    """
    Extract ids of roadblocks and lane ids start from reference ego state with the guidance of
        scenario's recorded roadblocks that follow the route.
    Lane includes current lane that contains the trajectory, leftmost lane in the roadblock and
        rightmost lane in the roadblock.
    :param map_api: map to perform extraction on.
    :param ego_states: sequence of agent states representing trajectory.
    :return : List of ids of roadblocks and lanes(center, leftmost and rightmost)
        that follow ego's route.
    """
    # get initial roadblock from all roadblock list
    point = ego_state.rear_axle.point
    initial_road_objs = extract_roadblock_objects(map_api, point)
    # special case, the initial ego state is no on any roadblocks/connectors
    if not initial_road_objs:
        road_id, _ = map_api.get_distance_to_nearest_map_object(point, SemanticMapLayer.ROADBLOCK)
        roadconnector_id, _ = map_api.get_distance_to_nearest_map_object(point, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        initial_road_objs = [
            get_roadblock_obj_from_id(map_api, road_id),
            get_roadblock_obj_from_id(map_api, roadconnector_id),
        ]

    future_road_ids = _get_future_road_ids(initial_road_objs, all_road_ids)

    future_road_ids = _remove_consecutive_duplicates(future_road_ids)
    
    future_roads = [get_roadblock_obj_from_id(map_api, road_id) for road_id in future_road_ids]
    # first step
    center_lane, left_lane, right_lane = get_current_center_and_side_lane_from_roadblock(ego_state.rear_axle, future_roads[0], True)
    center_lanes: List[LaneGraphEdgeMapObject] = [center_lane]
    left_lanes: List[LaneGraphEdgeMapObject] = [left_lane]
    right_lanes: List[LaneGraphEdgeMapObject] = [right_lane]

    while True:
        center_lanes = _find_center_lanes(future_roads, center_lanes, len(center_lanes))
        if len(center_lanes) == len(future_roads):
            break
        step = len(center_lanes)
        last_state = center_lanes[-1].baseline_path.discrete_path[-1]
        all_lanes_in_next_road = future_roads[step].interior_edges
        # find the closest lane to last fineded center lane
        next_lane = min(all_lanes_in_next_road, key=lambda lane: last_state.distance_to(lane.baseline_path.discrete_path[0]))
        center_lanes.append(next_lane)

    assert len(center_lanes) == len(future_roads)

    for road, center_lane in zip(future_roads[1:], center_lanes[1:]):
        reference_state = center_lane.baseline_path.discrete_path[0]
        left_lane, right_lane = get_current_center_and_side_lane_from_roadblock(reference_state, road, False)
        left_lanes.append(left_lane)
        right_lanes.append(right_lane)

    center_lane_ids: List[str] = [lane.id for lane in center_lanes]
    left_lane_ids: List[str] = [lane.id for lane in left_lanes]
    right_lane_ids: List[str] = [lane.id for lane in right_lanes]

    return future_road_ids, [center_lane_ids, left_lane_ids, right_lane_ids]