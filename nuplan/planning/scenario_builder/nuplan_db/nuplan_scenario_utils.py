from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Set, Tuple, Union, cast

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.geometry.interpolate_state import interpolate_future_waypoints
from nuplan.database.common.blob_store.creator import BlobStoreCreator
from nuplan.database.common.blob_store.local_store import LocalStore
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_future_waypoints_for_agents_from_db,
    get_lidarpc_token_timestamp_from_db,
    get_sampled_lidarpc_tokens_in_time_window_from_db,
    get_tracked_objects_for_lidarpc_token_from_db,
    get_tracked_objects_within_time_interval_from_db,
)
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
import numpy as np
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.polyline_map_object import NuPlanPolylineMapObject
from nuplan.common.maps.nuplan_map.utils import (
    extract_roadblock_objects,
)
from nuplan.planning.training.preprocessing.features.raster_utils import (
    _cartesian_to_projective_coords,
)
from nuplan.common.maps.nuplan_map.e2e_utils import (
    get_lane_obj_from_id,
    get_current_center_and_side_lane_from_roadblock,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    MapObjectPolylines,
)
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

LIDAR_PC_CACHE = 16 * 2**10  # 16K

DEFAULT_SCENARIO_NAME = 'unknown'  # name of scenario (e.g. ego overtaking)
DEFAULT_SCENARIO_DURATION = 20.0  # [s] duration of the scenario (e.g. extract 20s from when the event occurred)
DEFAULT_EXTRACTION_OFFSET = 0.0  # [s] offset of the scenario (e.g. start at -5s from when the event occurred)
DEFAULT_SUBSAMPLE_RATIO = 1.0  # ratio used sample the scenario (e.g. a 0.1 ratio means sample from 20Hz to 2Hz)


@dataclass(frozen=True)
class ScenarioExtractionInfo:
    """
    Structure containing information used to extract a scenario (lidarpc sequence).
    """

    scenario_name: str = DEFAULT_SCENARIO_NAME  # name of the scenario
    scenario_duration: float = DEFAULT_SCENARIO_DURATION  # [s] duration of the scenario
    extraction_offset: float = DEFAULT_EXTRACTION_OFFSET  # [s] offset of the scenario
    subsample_ratio: float = DEFAULT_SUBSAMPLE_RATIO  # ratio to sample the scenario

    def __post_init__(self) -> None:
        """Sanitize class attributes."""
        assert 0.0 < self.scenario_duration, f'Scenario duration has to be greater than 0, got {self.scenario_duration}'
        assert (
            0.0 < self.subsample_ratio <= 1.0
        ), f'Subsample ratio has to be between 0 and 1, got {self.subsample_ratio}'


class ScenarioMapping:
    """
    Structure that maps each scenario type to instructions used in extracting it.
    """

    def __init__(
        self,
        scenario_map: Dict[str, Union[Tuple[float, float, float], Tuple[float, float]]],
        subsample_ratio_override: Optional[float],
    ) -> None:
        """
        Initializes the scenario mapping class.
        :param scenario_map: Dictionary with scenario name/type as keys and
                             tuples of (scenario duration, extraction offset, subsample ratio) as values.
        :subsample_ratio_override: The override for the subsample ratio if not provided.
        """
        self.mapping: Dict[str, ScenarioExtractionInfo] = {}
        self.subsample_ratio_override = (
            subsample_ratio_override if subsample_ratio_override is not None else DEFAULT_SUBSAMPLE_RATIO
        )

        for name in scenario_map:
            this_ratio: float = scenario_map[name][2] if len(scenario_map[name]) == 3 else self.subsample_ratio_override  # type: ignore

            self.mapping[name] = ScenarioExtractionInfo(
                scenario_name=name,
                scenario_duration=scenario_map[name][0],
                extraction_offset=scenario_map[name][1],
                subsample_ratio=this_ratio,
            )

    def get_extraction_info(self, scenario_type: str) -> Optional[ScenarioExtractionInfo]:
        """
        Accesses the scenario mapping using a query scenario type.
        If the scenario type is not found, a default extraction info object is returned.
        :param scenario_type: Scenario type to query for.
        :return: Scenario extraction information for the queried scenario type.
        """
        return (
            self.mapping[scenario_type]
            if scenario_type in self.mapping
            else ScenarioExtractionInfo(subsample_ratio=self.subsample_ratio_override)
        )


def download_file_if_necessary(data_root: str, potentially_remote_path: str, verbose: bool = False) -> str:
    """
    Downloads the db file if necessary.
    :param potentially_remote_path: The path from which to download the file.
    :param verbose: Verbosity level.
    :return: The local path for the file.
    """
    # If the file path is a local directory and exists, then return that.
    # e.g. /data/sets/nuplan/nuplan-v1.1/file.db
    if os.path.exists(potentially_remote_path):
        return potentially_remote_path

    log_name = absolute_path_to_log_name(potentially_remote_path)
    download_name = log_name + ".db"

    # TODO: CacheStore seems to be buggy.
    # Behavior seems to be different on our cluster vs locally regarding downloaded file paths.
    #
    # Use the underlying stores manually.
    os.makedirs(data_root, exist_ok=True)
    local_store = LocalStore(data_root)

    if not local_store.exists(download_name):
        blob_store = BlobStoreCreator.create_nuplandb(data_root, verbose=verbose)

        # If we have no matches, download the file.
        logger.info("DB path not found. Downloading to %s..." % download_name)
        start_time = time.time()
        content = blob_store.get(potentially_remote_path)
        local_store.put(download_name, content)
        logger.info("Downloading db file took %.2f seconds." % (time.time() - start_time))

    return os.path.join(data_root, download_name)


def _process_future_trajectories_for_windowed_agents(
    log_file: str,
    tracked_objects: List[TrackedObject],
    agent_indexes: Dict[int, Dict[str, int]],
    future_trajectory_sampling: TrajectorySampling,
) -> List[TrackedObject]:
    """
    A helper method to interpolate and parse the future trajectories for windowed agents.
    :param log_file: The log file to query.
    :param tracked_objects: The tracked objects to parse.
    :param agent_indexes: A mapping of [timestamp, [track_token, tracked_object_idx]]
    :param future_trajectory_sampling: The future trajectory sampling to use for future waypoints.
    :return: The tracked objects with predicted trajectories included.
    """
    agent_future_trajectories: Dict[int, Dict[str, List[Waypoint]]] = {}
    for timestamp in agent_indexes:
        agent_future_trajectories[timestamp] = {}

        for token in agent_indexes[timestamp]:
            agent_future_trajectories[timestamp][token] = []

    for timestamp_time in agent_future_trajectories:
        end_time = timestamp_time + int(
            1e6 * (future_trajectory_sampling.time_horizon + future_trajectory_sampling.interval_length)
        )

        # TODO: This is somewhat inefficient because the resampling should happen in SQL layer

        for track_token, waypoint in get_future_waypoints_for_agents_from_db(
            log_file, list(agent_indexes[timestamp_time].keys()), timestamp_time, end_time
        ):
            agent_future_trajectories[timestamp_time][track_token].append(waypoint)

    for timestamp in agent_future_trajectories:
        for key in agent_future_trajectories[timestamp]:
            # We can only interpolate waypoints if there is more than one in the future.
            if len(agent_future_trajectories[timestamp][key]) == 1:
                tracked_objects[agent_indexes[timestamp][key]]._predictions = [
                    PredictedTrajectory(1.0, agent_future_trajectories[timestamp][key])
                ]
            elif len(agent_future_trajectories[timestamp][key]) > 1:
                tracked_objects[agent_indexes[timestamp][key]]._predictions = [
                    PredictedTrajectory(
                        1.0,
                        interpolate_future_waypoints(
                            agent_future_trajectories[timestamp][key],
                            future_trajectory_sampling.time_horizon,
                            future_trajectory_sampling.interval_length,
                        ),
                    )
                ]

    return tracked_objects


def extract_tracked_objects_within_time_window(
    token: str,
    log_file: str,
    past_time_horizon: float,
    future_time_horizon: float,
    filter_track_tokens: Optional[Set[str]] = None,
    future_trajectory_sampling: Optional[TrajectorySampling] = None,
) -> TrackedObjects:
    """
    Extracts the tracked objects in a time window centered on a token.
    :param token: The token on which to center the time window.
    :param past_time_horizon: The time in the past for which to search.
    :param future_time_horizon: The time in the future for which to search.
    :param filter_track_tokens: If provided, objects with track_tokens missing from the set will be excluded.
    :param future_trajectory_sampling: If provided, the future trajectory sampling to use for future waypoints.
    :return: The retrieved TrackedObjects.
    """
    tracked_objects: List[TrackedObject] = []
    agent_indexes: Dict[int, Dict[str, int]] = {}

    token_timestamp = get_lidarpc_token_timestamp_from_db(log_file, token)
    start_time = token_timestamp - (1e6 * past_time_horizon)
    end_time = token_timestamp + (1e6 * future_time_horizon)

    for idx, tracked_object in enumerate(
        get_tracked_objects_within_time_interval_from_db(log_file, start_time, end_time, filter_track_tokens)
    ):
        if future_trajectory_sampling and isinstance(tracked_object, Agent):
            if tracked_object.metadata.timestamp_us not in agent_indexes:
                agent_indexes[tracked_object.metadata.timestamp_us] = {}

            agent_indexes[tracked_object.metadata.timestamp_us][tracked_object.metadata.track_token] = idx
        tracked_objects.append(tracked_object)

    if future_trajectory_sampling:
        _process_future_trajectories_for_windowed_agents(
            log_file, tracked_objects, agent_indexes, future_trajectory_sampling
        )

    return TrackedObjects(tracked_objects=tracked_objects)


def extract_tracked_objects(
    token: str,
    log_file: str,
    future_trajectory_sampling: Optional[TrajectorySampling] = None,
) -> TrackedObjects:
    """
    Extracts all boxes from a lidarpc.
    :param lidar_pc: Input lidarpc.
    :param future_trajectory_sampling: If provided, the future trajectory sampling to use for future waypoints.
    :return: Tracked objects contained in the lidarpc.
    """
    tracked_objects: List[TrackedObject] = []
    agent_indexes: Dict[str, int] = {}
    agent_future_trajectories: Dict[str, List[Waypoint]] = {}

    for idx, tracked_object in enumerate(get_tracked_objects_for_lidarpc_token_from_db(log_file, token)):
        if future_trajectory_sampling and isinstance(tracked_object, Agent):
            agent_indexes[tracked_object.metadata.track_token] = idx
            agent_future_trajectories[tracked_object.metadata.track_token] = []
        tracked_objects.append(tracked_object)

    if future_trajectory_sampling and len(tracked_objects) > 0:
        timestamp_time = get_lidarpc_token_timestamp_from_db(log_file, token)
        end_time = timestamp_time + int(
            1e6 * (future_trajectory_sampling.time_horizon + future_trajectory_sampling.interval_length)
        )

        # TODO: This is somewhat inefficient because the resampling should happen in SQL layer
        for track_token, waypoint in get_future_waypoints_for_agents_from_db(
            log_file, list(agent_indexes.keys()), timestamp_time, end_time
        ):
            agent_future_trajectories[track_token].append(waypoint)

        for key in agent_future_trajectories:
            # We can only interpolate waypoints if there is more than one in the future.
            if len(agent_future_trajectories[key]) == 1:
                tracked_objects[agent_indexes[key]]._predictions = [
                    PredictedTrajectory(1.0, agent_future_trajectories[key])
                ]
            elif len(agent_future_trajectories[key]) > 1:
                tracked_objects[agent_indexes[key]]._predictions = [
                    PredictedTrajectory(
                        1.0,
                        interpolate_future_waypoints(
                            agent_future_trajectories[key],
                            future_trajectory_sampling.time_horizon,
                            future_trajectory_sampling.interval_length,
                        ),
                    )
                ]

    return TrackedObjects(tracked_objects=tracked_objects)


def extract_lidarpc_tokens_as_scenario(
    log_file: str, anchor_timestamp: float, scenario_extraction_info: ScenarioExtractionInfo
) -> Generator[str, None, None]:
    """
    Extract a list of lidarpc tokens that form a scenario around an anchor timestamp.
    :param log_file: The log file to access
    :param anchor_timestamp: Timestamp of Lidarpc representing the start of the scenario.
    :param scenario_extraction_info: Structure containing information used to extract the scenario.
    :return: List of extracted lidarpc tokens representing the scenario.
    """
    start_timestamp = int(anchor_timestamp + scenario_extraction_info.extraction_offset * 1e6)
    end_timestamp = int(start_timestamp + scenario_extraction_info.scenario_duration * 1e6)
    subsample_step = int(1.0 / scenario_extraction_info.subsample_ratio)

    return cast(
        Generator[str, None, None],
        get_sampled_lidarpc_tokens_in_time_window_from_db(log_file, start_timestamp, end_timestamp, subsample_step),
    )


def absolute_path_to_log_name(absolute_path: str) -> str:
    """
    Gets the log name from the absolute path to a log file.
    E.g.
        input: data/sets/nuplan/nuplan-v1.1/mini/2021.10.11.02.57.41_veh-50_01522_02088.db
        output: 2021.10.11.02.57.41_veh-50_01522_02088

        input: /tmp/abcdef
        output: abcdef
    :param absolute_path: The absolute path to a log file.
    :return: The log name.
    """
    filename = os.path.basename(absolute_path)

    # Files generated during caching do not end with ".db"
    # They have no extension.
    if filename.endswith(".db"):
        filename = os.path.splitext(filename)[0]
    return filename


def check_point_on_intersection(map_api: AbstractMap, point: Point2D) -> bool:
    """
    Check whether the point is in an intersection.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :return List of roadblocks/roadblock connectors containing point if they exist.
    """
    roadblock = map_api.get_one_map_object(point, SemanticMapLayer.ROADBLOCK)
    if roadblock:
        return False
    else:
        return True

def get_future_pathway_from_lane_ids(
    ego_state: EgoState,
    map_api: AbstractMap,
    lane_ids: List[str],
    left_ids: List[str],
    right_ids: List[str],
    interval: float = 4,
    initial_interval: float = 6,
    num_steps: int = 12,
) -> List[Union[npt.NDArray, List[bool]]]:
    """
    Get following future pathway landmarks [N, 2] in equal interval for certain steps:
        1. Route lane center line;
        2. Route lane left line;
        3. Route lane right line;
        4. Route roadblock left edge;
        5. Route roadblock right edge;
        6. Whether route center line point is in intersection.
    Total distance: initial_interval + (num_steps-1) * interval
    All points transformed to ego rear axle system.

    :param interval: interval(m) between each point that follows the forward direction.
    :param initial_interval: initial interval(m) between ego rear axle and first point.
    :param num_steps: how many points to find.
    """
    global_transform = np.linalg.inv(ego_state.rear_axle.as_matrix())
    # By default the map is right-oriented, this makes it top-oriented.
    map_align_transform = R.from_euler(
        'z', 90, degrees=True).as_matrix().astype(
        np.float32)
    #TODO: do we need map_align_transform??
    # transform = map_align_transform @ global_transform
    transform = global_transform

    center_lane = get_lane_obj_from_id(map_api, lane_ids[0])
    left_lane = get_lane_obj_from_id(map_api, left_ids[0])
    right_lane = get_lane_obj_from_id(map_api, right_ids[0])

    total_dist = 0
    dist_along_current_lane = -100
    future_route: List[Point2D] = []
    left_lms: List[Point2D] = []
    right_lms: List[Point2D] = []
    left_edges: List[Point2D] = []
    right_edges: List[Point2D] = []
    on_intersection: List[bool] = []
    for i, lane_id in enumerate(lane_ids):
        center_lane = get_lane_obj_from_id(map_api, lane_id)
        left_lane = get_lane_obj_from_id(map_api, left_ids[i])
        right_lane = get_lane_obj_from_id(map_api, right_ids[i])
        center_polyline: NuPlanPolylineMapObject  = center_lane.baseline_path
        left_polyline: NuPlanPolylineMapObject  = center_lane.left_boundary
        right_polyline: NuPlanPolylineMapObject  = center_lane.right_boundary
        left_edge_polyline: NuPlanPolylineMapObject  = left_lane.left_boundary
        right_edge_polyline: NuPlanPolylineMapObject  = right_lane.right_boundary

        if dist_along_current_lane == -100:
            dist_along_current_lane = center_polyline.get_nearest_arc_length_from_position(ego_state.center.point)
        cur_interval = initial_interval if len(future_route) == 0 else interval
        while dist_along_current_lane + cur_interval < center_polyline.length:
            point = center_polyline.linestring.interpolate(dist_along_current_lane + cur_interval)
            left_point = left_polyline.linestring.interpolate(left_polyline.linestring.project(point))
            right_point = right_polyline.linestring.interpolate(right_polyline.linestring.project(point))

            future_route.append(Point2D(point.x, point.y))
            left_lms.append(Point2D(left_point.x, left_point.y))
            right_lms.append(Point2D(right_point.x, right_point.y))
            on_intersection.append(check_point_on_intersection(map_api, future_route[-1]))
            
            left_edge = left_edge_polyline.linestring.interpolate(dist_along_current_lane + cur_interval)
            right_edge = right_edge_polyline.linestring.interpolate(dist_along_current_lane + cur_interval)
            left_edges.append(Point2D(left_edge.x, left_edge.y))
            right_edges.append(Point2D(right_edge.x, right_edge.y))

            total_dist += cur_interval
            dist_along_current_lane += cur_interval
            cur_interval = interval
            if len(future_route) == num_steps:
                all_points = MapObjectPolylines([future_route, left_lms, right_lms, left_edges, right_edges]).to_vector()
                all_points = [(transform @ _cartesian_to_projective_coords(
                    np.array(polylines)).T).T[:, :2] for polylines in all_points]
                return [*all_points, on_intersection]
        # switch to next lane
        dist_along_current_lane = dist_along_current_lane - center_polyline.length
    
    # if future route lane is not enough to cover total steps, extend reasonably forward
    while len(future_route) < num_steps:
        next_lanes = center_lane.outgoing_edges
        if not next_lanes:
            if len(future_route) > 0:
                all_points = MapObjectPolylines([future_route, left_lms, right_lms, left_edges, right_edges]).to_vector()
                all_points = [(transform @ _cartesian_to_projective_coords(
                    np.array(polylines)).T).T[:, :2] for polylines in all_points]
                return [*all_points, on_intersection]
            else:
                return [future_route, left_lms, right_lms, left_edges, right_edges, on_intersection] 
        center_lane = next_lanes[0]
        reference_state = center_lane.baseline_path.discrete_path[0]
        road = extract_roadblock_objects(map_api, reference_state.point)[0]
        left_lane, right_lane = get_current_center_and_side_lane_from_roadblock(reference_state, road, False)
        center_polyline: NuPlanPolylineMapObject  = center_lane.baseline_path
        left_polyline: NuPlanPolylineMapObject  = center_lane.left_boundary
        right_polyline: NuPlanPolylineMapObject  = center_lane.right_boundary
        left_edge_polyline: NuPlanPolylineMapObject  = left_lane.left_boundary
        right_edge_polyline: NuPlanPolylineMapObject  = right_lane.right_boundary

        cur_interval = initial_interval if len(future_route) == 0 else interval
        while dist_along_current_lane + cur_interval < center_polyline.length:
            point = center_polyline.linestring.interpolate(dist_along_current_lane + cur_interval)
            left_point = left_polyline.linestring.interpolate(left_polyline.linestring.project(point))
            right_point = right_polyline.linestring.interpolate(right_polyline.linestring.project(point))

            future_route.append(Point2D(point.x, point.y))
            left_lms.append(Point2D(left_point.x, left_point.y))
            right_lms.append(Point2D(right_point.x, right_point.y))
            on_intersection.append(check_point_on_intersection(map_api, future_route[-1]))
            
            left_edge = left_edge_polyline.linestring.interpolate(dist_along_current_lane + cur_interval)
            right_edge = right_edge_polyline.linestring.interpolate(dist_along_current_lane + cur_interval)
            left_edges.append(Point2D(left_edge.x, left_edge.y))
            right_edges.append(Point2D(right_edge.x, right_edge.y))

            total_dist += cur_interval
            dist_along_current_lane += cur_interval
            cur_interval = interval
            if len(future_route) == num_steps:
                all_points = MapObjectPolylines([future_route, left_lms, right_lms, left_edges, right_edges]).to_vector()
                all_points = [(transform @ _cartesian_to_projective_coords(
                    np.array(polylines)).T).T[:, :2] for polylines in all_points]
                return [*all_points, on_intersection]
        # switch to next lane
        dist_along_current_lane = dist_along_current_lane - center_polyline.length