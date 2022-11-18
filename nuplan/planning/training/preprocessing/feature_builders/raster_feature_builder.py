from __future__ import annotations

from typing import Dict, List, Type

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.raster_utils import (
    get_baseline_paths_raster,
    get_ego_raster,
    get_roadmap_raster,
    get_route_raster,
    get_past_current_agents_raster,
    get_speed_raster,
)

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses


class RasterFeatureBuilder(AbstractFeatureBuilder):
    """
    Raster builder responsible for constructing model input features.
    """

    def __init__(
        self,
        map_features: Dict[str, int],
        num_input_channels: int,
        target_width: int,
        target_height: int,
        target_pixel_size: float,
        ego_width: float,
        ego_front_length: float,
        ego_rear_length: float,
        ego_longitudinal_offset: float,
        baseline_path_thickness: int,
    ) -> None:
        """
        Initializes the builder.
        :param map_features: name of map features to be drawn and their color for encoding.
        :param num_input_channels: number of input channel of the raster model.
        :param target_width: [pixels] target width of the raster
        :param target_height: [pixels] target height of the raster
        :param target_pixel_size: [m] target pixel size in meters
        :param ego_width: [m] width of the ego vehicle
        :param ego_front_length: [m] distance between the rear axle and the front bumper
        :param ego_rear_length: [m] distance between the rear axle and the rear bumper
        :param ego_longitudinal_offset: [%] offset percentage to place the ego vehicle in the raster.
                                        0.0 means place the ego at 1/2 from the bottom of the raster image.
                                        0.25 means place the ego at 1/4 from the bottom of the raster image.
        :param baseline_path_thickness: [pixels] the thickness of baseline paths in the baseline_paths_raster.
        """
        self.map_features = map_features
        self.num_input_channels = num_input_channels
        self.target_width = target_width
        self.target_height = target_height
        self.target_pixel_size = target_pixel_size

        self.ego_longitudinal_offset = ego_longitudinal_offset
        self.baseline_path_thickness = baseline_path_thickness
        self.raster_shape = (self.target_width, self.target_height)

        x_size = self.target_width * self.target_pixel_size / 2.0
        y_size = self.target_height * self.target_pixel_size / 2.0
        x_offset = 2.0 * self.ego_longitudinal_offset * x_size
        self.x_range = (-x_size + x_offset, x_size + x_offset)
        self.y_range = (-y_size, y_size)

        self.ego_width_pixels = int(ego_width / self.target_pixel_size)
        self.ego_front_length_pixels = int(ego_front_length / self.target_pixel_size)
        self.ego_rear_length_pixels = int(ego_rear_length / self.target_pixel_size)

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "raster"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Raster  # type: ignore

    def get_features_from_scenario(self, scenario: AbstractScenario, iteration: int) -> Raster:
        """Inherited, see superclass."""

        ego_state = scenario.get_ego_state_at_iteration(iteration)

        detections = scenario.get_tracked_objects_at_iteration(iteration)

        map_api = scenario.map_api

        route_roadblock_ids = scenario.get_route_roadblock_ids()

        past_ego_states = list(scenario.get_ego_past_trajectory(
            iteration=iteration, time_horizon=2, num_samples=4)) # 2 second, 4 step, 0.5s interval
        past_detections = list(scenario.get_past_tracked_objects(
            iteration=iteration, time_horizon=2, num_samples=4))
        trajectory_past_relative_poses = convert_absolute_to_relative_poses(
            ego_state.rear_axle, [state.rear_axle for state in past_ego_states]
        )
        result = self._compute_feature(ego_state, detections, map_api, route_roadblock_ids, trajectory_past_relative_poses, past_detections)

        return result

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> Raster:
        """Inherited, see superclass."""
        history = current_input.history
        ego_state = history.ego_states[-1]
        observation = history.observations[-1]
        route_roadblock_ids = initialization.route_roadblock_ids
        past_trajectory = convert_absolute_to_relative_poses(
            ego_state.rear_axle, [state.rear_axle for state in history.ego_states[-11:-10]]
        )
        past_detection = history.observations[::-10][::-1][:-1]
        feature = self._compute_feature(ego_state, observation, initialization.map_api, route_roadblock_ids, past_trajectory, past_detection)

        if isinstance(observation, DetectionsTracks):
            return feature
        else:
            raise TypeError(f"Observation was type {observation.detection_type()}. Expected DetectionsTracks")

    def _compute_feature(
        self,
        ego_state: EgoState,
        detections: DetectionsTracks,
        map_api: AbstractMap,
        route_roadblock_ids: List[str],
        past_ego_trajectory,
        past_detections: List[DetectionsTracks],
    ) -> Raster:
        # Add task A for 1s.
        # profiler.start("roadmap")
        # Construct map, agents and ego layers
        len_steps = len(past_detections) if past_detections is not None else 0
        if len_steps == 0:
            past_detections = []
        roadmap_raster = get_roadmap_raster(
            ego_state.agent,
            map_api,
            self.map_features,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
        )

        agents_raster = np.zeros(self.raster_shape, dtype=np.float32)

        # Agent historical data
        for past_step, past_detections in enumerate(
                past_detections + [detections], start=1):
            agents_raster = get_past_current_agents_raster(
                agents_raster,
                ego_state,
                past_detections,
                self.x_range,
                self.y_range,
                self.raster_shape,
                color_value=past_step / (len_steps + 1),
            )
        agents_raster = np.asarray(agents_raster)
        agents_raster = np.flip(agents_raster, axis=0)
        agents_raster = np.ascontiguousarray(agents_raster, dtype=np.float32)

        # ego_raster current
        ego_raster = get_ego_raster(
            self.raster_shape,
            self.ego_longitudinal_offset,
            self.ego_width_pixels,
            self.ego_front_length_pixels,
            self.ego_rear_length_pixels,
        )

        baseline_paths_raster = get_baseline_paths_raster(
            ego_state.agent,
            map_api,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
            self.baseline_path_thickness,
        )

        route_raster = get_route_raster(
            ego_state.agent,
            route_roadblock_ids,
            map_api,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
        )
        route_raster /= 255.0
        ego_speed_raster = get_speed_raster(
            past_ego_trajectory,
            self.raster_shape,
        )
        # hardcode
        ego_speed_raster /= 8.0
        collated_layers: npt.NDArray[np.float32] = np.dstack(
            [
                ego_raster,
                agents_raster,
                roadmap_raster,
                baseline_paths_raster,
                route_raster,
                ego_speed_raster,
            ]
        ).astype(np.float32)

        # Ensures channel is the last dimension.
        if collated_layers.shape[-1] != self.num_input_channels:
            raise RuntimeError(
                f'Invalid raster numpy array. '
                f'Expected {self.num_input_channels} channels, got {collated_layers.shape[-1]} '
                f'Shape is {collated_layers.shape}'
            )

        result = Raster(data=collated_layers)
        return result
