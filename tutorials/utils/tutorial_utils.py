import itertools
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from bokeh.document.document import Document
from bokeh.io import show
from bokeh.layouts import column

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_lidarpc_token_map_name_from_db,
    get_lidarpc_token_timestamp_from_db,
    get_lidarpc_tokens_with_scenario_tag_from_db,
)
from nuplan.planning.nuboard.base.data_class import NuBoardFile, SimulationScenarioKey
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.base.simulation_tile import SimulationTile
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import discover_log_dbs
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioExtractionInfo,
)
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

logger = logging.getLogger(__name__)


@dataclass
class HydraConfigPaths:
    """
    Stores relative hydra paths to declutter tutorial.
    """

    common_dir: str
    config_name: str
    config_path: str
    experiment_dir: str


def construct_nuboard_hydra_paths(base_config_path: str) -> HydraConfigPaths:
    """
    Specifies relative paths to nuBoard configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return Hydra config path.
    """
    common_dir = "file://" + join(base_config_path, 'config', 'common')
    config_name = 'default_nuboard'
    config_path = join(base_config_path, 'config/nuboard')
    experiment_dir = "file://" + join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)


def construct_simulation_hydra_paths(base_config_path: str) -> HydraConfigPaths:
    """
    Specifies relative paths to simulation configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return Hydra config path.
    """
    common_dir = "file://" + join(base_config_path, 'config', 'common')
    config_name = 'default_simulation'
    config_path = join(base_config_path, 'config', 'simulation')
    experiment_dir = "file://" + join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)


def save_scenes_to_dir(
    scenario: AbstractScenario, save_dir: str, simulation_history: SimulationHistory
) -> SimulationScenarioKey:
    """
    Save scenes to a directory.
    :param scenario: Scenario.
    :param save_dir: Save path.
    :param simulation_history: Simulation history.
    :return Scenario key of simulation.
    """
    planner_name = "tutorial_planner"
    scenario_type = scenario.scenario_type
    scenario_name = scenario.scenario_name
    log_name = scenario.log_name

    save_path = Path(save_dir)
    file = save_path / planner_name / scenario_type / log_name / scenario_name / (scenario_name + ".msgpack.xz")
    file.parent.mkdir(exist_ok=True, parents=True)

    # Create a dummy planner
    dummy_planner = _create_dummy_simple_planner(acceleration=[5.0, 5.0])
    simulation_log = SimulationLog(
        planner=dummy_planner, scenario=scenario, simulation_history=simulation_history, file_path=file
    )
    simulation_log.save_to_file()

    return SimulationScenarioKey(
        planner_name=planner_name,
        scenario_name=scenario_name,
        scenario_type=scenario_type,
        nuboard_file_index=0,
        log_name=log_name,
        files=[file],
    )


def _create_dummy_simple_planner(
    acceleration: List[float], horizon_seconds: float = 10.0, sampling_time: float = 20.0
) -> SimplePlanner:
    """
    Create a dummy simple planner.
    :param acceleration: [m/s^2] constant ego acceleration, till limited by max_velocity.
    :param horizon_seconds: [s] time horizon being run.
    :param sampling_time: [s] sampling timestep.
    """
    acceleration_np: npt.NDArray[np.float32] = np.asarray(acceleration)
    return SimplePlanner(
        horizon_seconds=horizon_seconds,
        sampling_time=sampling_time,
        acceleration=acceleration_np,
    )


def _create_dummy_simulation_history_buffer(
    scenario: AbstractScenario, iteration: int = 0, time_horizon: int = 2, num_samples: int = 2, buffer_size: int = 2
) -> SimulationHistoryBuffer:
    """
    Create dummy SimulationHistoryBuffer.
    :param scenario: Scenario.
    :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
    :param num_samples: number of entries in the future.
    :param time_horizon: the desired horizon to the future.
    :param buffer_size: size of buffer.
    :return: SimulationHistoryBuffer.
    """
    past_observation = list(
        scenario.get_past_tracked_objects(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples)
    )

    past_ego_states = list(
        scenario.get_ego_past_trajectory(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples)
    )

    # Dummy history buffer
    history_buffer = SimulationHistoryBuffer.initialize_from_list(
        buffer_size=buffer_size,
        ego_states=past_ego_states,
        observations=past_observation,
        sample_interval=scenario.database_interval,
    )

    return history_buffer


def serialize_scenario(
    scenario: AbstractScenario, num_poses: int = 12, future_time_horizon: float = 6.0
) -> SimulationHistory:
    """
    Serialize a scenario to a list of scene dicts.
    :param scenario: Scenario.
    :param num_poses: Number of poses in trajectory.
    :param future_time_horizon: Future time horizon in trajectory.
    :return A list of scene dicts.
    """
    simulation_history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
    ego_controller = PerfectTrackingController(scenario)
    simulation_time_controller = StepSimulationTimeController(scenario)
    observations = TracksObservation(scenario)

    # Dummy history buffer
    history_buffer = _create_dummy_simulation_history_buffer(scenario=scenario)

    # Get all states
    for _ in range(simulation_time_controller.number_of_iterations()):
        iteration = simulation_time_controller.get_iteration()
        ego_state = ego_controller.get_state()
        observation = observations.get_observation()
        traffic_light_status = list(scenario.get_traffic_light_status_at_iteration(iteration.index))

        # Log play back trajectory
        current_state = scenario.get_ego_state_at_iteration(iteration.index)
        states = scenario.get_ego_future_trajectory(iteration.index, future_time_horizon, num_poses)
        trajectory = InterpolatedTrajectory(list(itertools.chain([current_state], states)))

        simulation_history.add_sample(
            SimulationHistorySample(iteration, ego_state, trajectory, observation, traffic_light_status)
        )
        next_iteration = simulation_time_controller.next_iteration()

        if next_iteration:
            ego_controller.update_state(iteration, next_iteration, ego_state, trajectory)
            observations.update_observation(iteration, next_iteration, history_buffer)

    return simulation_history


def visualize_scenario(scenario: NuPlanScenario, save_dir: str = '/tmp/scenario_visualization/') -> None:
    """
    Visualize a scenario in Bokeh.
    :param scenario: Scenario object to be visualized.
    :param save_dir: Dir to save serialization and visualization artifacts.
    """
    map_factory = NuPlanMapFactory(get_maps_db(map_root=scenario.map_root, map_version=scenario.map_version))

    simulation_history = serialize_scenario(scenario)
    simulation_scenario_key = save_scenes_to_dir(
        scenario=scenario, save_dir=save_dir, simulation_history=simulation_history
    )
    visualize_scenarios([simulation_scenario_key], map_factory, Path(save_dir))


def visualize_scenarios(
    simulation_scenario_keys: List[SimulationScenarioKey], map_factory: NuPlanMapFactory, save_path: Path
) -> None:
    """
    Visualize scenarios in Bokeh.
    :param simulation_scenario_keys: A list of simulation scenario keys.
    :param map_factory: Map factory object to use for rendering.
    :param save_path: Path where to save the scene dict.
    """

    def complete_message() -> None:
        logger.info("Done rendering!")

    def bokeh_app(doc: Document) -> None:
        """Run bokeh app in jupyter notebook."""
        # Change simulation_main_path to a folder where you want to save rendered videos.
        nuboard_file = NuBoardFile(
            simulation_main_path=save_path.name,
            simulation_folder='',
            metric_main_path='',
            metric_folder='',
            aggregator_metric_folder='',
        )

        experiment_file_data = ExperimentFileData(file_paths=[nuboard_file])
        # Create a simulation tile
        simulation_tile = SimulationTile(
            doc=doc,
            map_factory=map_factory,
            experiment_file_data=experiment_file_data,
            vehicle_parameters=get_pacifica_parameters(),
        )

        # Render a simulation tile
        simulation_scenario_data = simulation_tile.render_simulation_tiles(simulation_scenario_keys)

        # Create layouts
        simulation_figures = [data.plot for data in simulation_scenario_data]
        simulation_layouts = column(simulation_figures)

        # Add the layouts to the bokeh document
        doc.add_root(simulation_layouts)
        doc.add_next_tick_callback(complete_message)

    show(bokeh_app)


def get_default_scenario_extraction(
    scenario_duration: float = 15.0,
    extraction_offset: float = -2.0,
    subsample_ratio: float = 0.5,
) -> ScenarioExtractionInfo:
    """
    Get default scenario extraction instructions used in visualization.
    :param scenario_duration: [s] Duration of scenario.
    :param extraction_offset: [s] Offset of scenario (e.g. -2 means start scenario 2s before it starts).
    :param subsample_ratio: Scenario resolution.
    :return: Scenario extraction info object.
    """
    return ScenarioExtractionInfo(DEFAULT_SCENARIO_NAME, scenario_duration, extraction_offset, subsample_ratio)


def get_default_scenario_from_token(
    data_root: str, log_file_full_path: str, token: str, map_root: str, map_version: str
) -> NuPlanScenario:
    """
    Build a scenario with default parameters for visualization.
    :param log_db: Log database object that the token belongs to.
    :param log_file_full_path: The full path to the log db file to use.
    :param token: Lidar pc token to be used as anchor for the scenario.
    :param map_root: The root directory to use for looking for maps.
    :param map_version: The map version to use.
    :return: Instantiated scenario object.
    """
    timestamp = get_lidarpc_token_timestamp_from_db(log_file_full_path, token)
    map_name = get_lidarpc_token_map_name_from_db(log_file_full_path, token)
    return NuPlanScenario(
        data_root=data_root,
        log_file_load_path=log_file_full_path,
        initial_lidar_token=token,
        initial_lidar_timestamp=timestamp,
        scenario_type=DEFAULT_SCENARIO_NAME,
        map_root=map_root,
        map_version=map_version,
        map_name=map_name,
        scenario_extraction_info=get_default_scenario_extraction(),
        ego_vehicle_parameters=get_pacifica_parameters(),
        ground_truth_predictions=None,
    )


def get_scenario_type_token_map(db_files: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Get a map from scenario types to lists of all instances for a given scenario type in the database."""
    available_scenario_types = defaultdict(list)
    for db_file in db_files:
        for tag, token in get_lidarpc_tokens_with_scenario_tag_from_db(db_file):
            available_scenario_types[tag].append((db_file, token))

    return available_scenario_types


def visualize_nuplan_scenarios(data_root: str, db_files: str, map_root: str, map_version: str) -> None:
    """Create a dropdown box populated with unique scenario types to visualize from a database."""
    from IPython.display import clear_output, display
    from ipywidgets import Dropdown, Output

    log_db_files = discover_log_dbs(db_files)

    scenario_type_token_map = get_scenario_type_token_map(log_db_files)

    out = Output()
    drop_down = Dropdown(description='Scenario', options=sorted(scenario_type_token_map.keys()))

    def scenario_dropdown_handler(change: Any) -> None:
        """Dropdown handler that randomly chooses a scenario from the selected scenario type and renders it."""
        with out:
            clear_output()
            logger.info("Randomly rendering a scenario...")
            scenario_type = str(change.new)
            log_db_file, token = random.choice(scenario_type_token_map[scenario_type])
            scenario = get_default_scenario_from_token(data_root, log_db_file, token, map_root, map_version)

            visualize_scenario(scenario)

    display(drop_down)
    display(out)
    drop_down.observe(scenario_dropdown_handler, names='value')
