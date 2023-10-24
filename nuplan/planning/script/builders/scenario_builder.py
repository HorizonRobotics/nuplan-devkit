import logging
from collections import defaultdict
from pathlib import Path
import pickle
from typing import Dict, List, Set, cast, Optional

from multiprocessing import Pool
from omegaconf import DictConfig
import pandas
from tqdm import tqdm

from nuplan.common.utils.s3_utils import check_s3_path_exists, expand_s3_dir, get_cache_metadata_paths, split_s3_path
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.cache.cached_scenario import CachedScenario
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.training.experiments.cache_metadata_entry import (
    extract_field_from_cache_metadata_entries,
    read_cache_metadata,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map

logger = logging.getLogger(__name__)


def get_s3_scenario_cache(
    cache_path: str,
    feature_names: Set[str],
    worker: WorkerPool,
) -> List[Path]:
    """
    Get a list of cached scenario paths from a remote (S3) cache.
    :param cache_path: Root path of the remote cache dir.
    :param feature_names: Set of required feature names to check when loading scenario paths from the cache.
    :return: List of discovered cached scenario paths.
    """
    # Retrieve all filenames contained in the remote location.
    assert check_s3_path_exists(cache_path), 'Remote cache {cache_path} does not exist!'

    # Get metadata files from s3 cache path provided
    s3_bucket, s3_key = split_s3_path(cache_path)
    metadata_files = get_cache_metadata_paths(s3_key, s3_bucket)
    if len(metadata_files) > 0:
        logger.info("Reading s3 directory from metadata.")
        cache_metadata_entries = read_cache_metadata(Path(cache_path), metadata_files, worker)
        s3_filenames = extract_field_from_cache_metadata_entries(cache_metadata_entries, 'file_name')
    else:  # If cache does not have any metadata csv files, fetch files directly from s3
        logger.warning("Not using metadata! This will be slow...")
        s3_filenames = expand_s3_dir(cache_path)
    assert len(s3_filenames) > 0, f'No files found in the remote cache {cache_path}!'

    # Create a 3-level hash with log names, scenario types and scenario tokens as keys and the set of contained features as values.
    cache_map: Dict[str, Dict[str, Dict[str, Set[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for s3_filename in s3_filenames:
        path = Path(s3_filename)
        cache_map[path.parent.parent.parent.name][path.parent.parent.name][path.parent.name].add(path.stem)

    # Keep only dir paths that contain all required feature names
    scenario_cache_paths = [
        Path(f'{log_name}/{scenario_type}/{scenario_token}')
        for log_name, scenario_types in cache_map.items()
        for scenario_type, scenarios in scenario_types.items()
        for scenario_token, features in scenarios.items()
        if not (feature_names - features)
    ]

    return scenario_cache_paths


def valid_check(path_and_feature_names):
    path, feature_names_set = path_and_feature_names
    return path if feature_names_set <= {feature_name.stem for feature_name in path.iterdir()} else None


def valid_check_sequential(path_and_feature_names):
    scene_path, feature_names_set = path_and_feature_names
    valid = [feature_names_set <= {feature_name.stem for feature_name in path.iterdir()} for path in scene_path.iterdir()]
    if all(valid):
        return scene_path
    else:
        return None


def get_local_scenario_cache(cache_path: str, feature_names: Optional[Set[str]], cache_metadata_path: Optional[str]=None, is_sequential: bool=False) -> List[Path]:
    """
    Get a list of cached scenario paths from a local cache.
    :param cache_path: Root path of the local cache dir.
    :param feature_names: Set of required feature names to check when loading scenario paths from the cache.
    :return: List of discovered cached scenario paths.
    """
    cache_dir = Path(cache_path)
    assert cache_dir.exists(), f'Local cache {cache_dir} does not exist!'
    assert any(cache_dir.iterdir()), f'No files found in the local cache {cache_dir}!'

    if cache_metadata_path is not None:
        cache_metadata_file = Path(cache_metadata_path)
        assert cache_metadata_file.suffix == '.csv', f"{cache_metadata_file} is not a CSV file."
        logger.info(f"Loading from {cache_metadata_file}...")
        csv = pandas.read_csv(cache_metadata_file)
        if is_sequential:
            candidate_scenario_dirs = list(set([Path(i).parent.parent for i in csv['file_name'].tolist()]))
        else:
            candidate_scenario_dirs = list(set([Path(i).parent for i in csv['file_name'].tolist()]))
    else:
        candidate_scenario_dirs = [path for log_dir in cache_dir.iterdir() for type_dir in log_dir.iterdir() for path in type_dir.iterdir()]

    # Keep only dir paths that contains all required feature names
    if feature_names is not None:
        logger.info("Validate candidate scenarios...")
        logger.info(f"feautre_names : {feature_names}")
        check_func = valid_check_sequential if is_sequential else valid_check
        with Pool(48) as p:
            scenario_cache_dirs = [path for path in tqdm(p.imap(check_func, [(path, feature_names) for path in candidate_scenario_dirs]), total=len(candidate_scenario_dirs)) if path is not None]
        logger.info(f"Found {len(scenario_cache_dirs)} scenarios in cache.")

    return candidate_scenario_dirs


def extract_scenarios_from_cache(
    cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper
) -> List[AbstractScenario]:
    """
    Build the scenario objects that comprise the training dataset from cache.
    :param cfg: Omegaconf dictionary.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :param model: NN model used for training.
    :return: List of extracted scenarios.
    """
    logger.info("Extracting scenarios from cache...")
    cache_path = str(cfg.cache.cache_path)
    cache_metadata_path = str(cfg.cache.cache_metadata_path) if cfg.cache.cache_metadata_path is not None else None

    # Find all required feature/target names to load from cache
    cache_valid_check = cfg.cache.get("valid_check", True)
    if cache_valid_check:
        feature_builders = model.get_list_of_required_feature()
        target_builders = model.get_list_of_computed_target()
        feature_names = {builder.get_feature_unique_name() for builder in feature_builders + target_builders}
    else:
        feature_names = None

    is_sequential = cfg.data_loader.params.sequential_train

    # Get cached scenario paths locally or remotely
    scenario_cache_paths = (
        get_s3_scenario_cache(cache_path, feature_names, worker)
        if cache_path.startswith('s3://')
        else get_local_scenario_cache(cache_path, feature_names, cache_metadata_path, is_sequential)
    )

    def filter_scenario_cache_paths_by_scenario_type(paths: List[Path]) -> List[Path]:
        """
        Filter the scenario cache paths by scenario type.
        :param paths: Scenario cache paths
        :return: Scenario cache paths filtered by desired scenario types
        """
        scenario_types_to_include = cfg.scenario_filter.scenario_types

        filtered_scenario_cache_paths = [path for path in paths if path.parent.name in scenario_types_to_include]
        return filtered_scenario_cache_paths

    # If user inputs desired scenario types and scenario_type is in cache path.
    if cfg.scenario_filter.scenario_types:
        validate_scenario_type_in_cache_path(scenario_cache_paths)
        logger.info('Filtering by desired scenario types')
        scenario_cache_paths = worker_map(
            worker,
            filter_scenario_cache_paths_by_scenario_type,
            scenario_cache_paths,
        )
        assert (
            len(scenario_cache_paths) > 0
        ), f"Zero scenario cache paths after filtering by desired scenario types: {cfg.scenario_filter.scenario_types}. Please check if the cache contains the desired scenario type."

    if cfg.data_loader.params.sequential_train:
        scenarios = worker_map(worker, create_closed_loop_scenario_from_paths, scenario_cache_paths)
    else:
        scenarios = worker_map(worker, create_scenario_from_paths, scenario_cache_paths)
    return cast(List[AbstractScenario], scenarios)


def extract_scenarios_from_cache_records(
    cached_scenario_records: List[Dict], worker: WorkerPool, 
) -> List[AbstractScenario]:
    logger.info("Loading cached scenario in versatile manner...")
    scenarios = worker_map(worker, create_scenario_from_records, cached_scenario_records)
    return cast(List[AbstractScenario], scenarios)


def extract_scenarios_from_dataset(cfg: DictConfig, worker: WorkerPool) -> List[AbstractScenario]:
    """
    Extract and filter scenarios by loading a dataset using the scenario builder.
    :param cfg: Omegaconf dictionary.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :return: List of extracted scenarios.
    """
    logger.info("Extracting scenarios from dataset...")
    scenario_builder = build_scenario_builder(cfg)
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    scenarios: List[AbstractScenario] = scenario_builder.get_scenarios(scenario_filter, worker)

    return scenarios


def build_scenarios(cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper) -> List[AbstractScenario]:
    """
    Build the scenario objects that comprise the training dataset.
    Add versatile caching function, will use cached scenarios directly unless:
    1. force recompute features are provided;
    2. no cache record file available;
    3. there are required features not in cache record file.
    :param cfg: Omegaconf dictionary.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :param model: NN model used for training.
    :return: List of extracted scenarios.
    """
    if cfg.cache.get('versatile_caching', False):

        feature_builders = model.get_list_of_required_feature()
        target_builders = model.get_list_of_computed_target()

        force_recompute_features = cfg.cache.force_recompute_features
        if force_recompute_features:
            actual_recompute_features = []
            for builder in feature_builders + target_builders:
                if builder.get_feature_unique_name() in force_recompute_features:
                    builder.force_recompute = True
                    actual_recompute_features.append(builder.get_feature_unique_name())
            logger.info(f'Versatile caching: recompute features {actual_recompute_features}')
            scenarios = extract_scenarios_from_dataset(cfg, worker)
        else:
            versatile_cache_pickle_file = Path(cfg.cache.versatile_cache_pickle_file)
            if versatile_cache_pickle_file.exists():
                with open(versatile_cache_pickle_file, 'rb') as f:
                    cached_scenario_records = pickle.load(f)
                required_features = {builder.get_feature_unique_name() for builder in feature_builders + target_builders}
                existing_features = set(cached_scenario_records[-1])
                if required_features - existing_features:
                    logger.info(f'Versatile caching: there are missing features, need to compute!')
                    scenarios = extract_scenarios_from_dataset(cfg, worker)
                else:
                    logger.info(f'Versatile caching: all features exist, use cache directly')
                    subsample = cfg.cache.versatile_cache_subsample
                    scenarios = extract_scenarios_from_cache_records(cached_scenario_records[0:-1:subsample], worker)
            else:
                logger.info(f'Versatile caching: no cached_scenario file, will create normal scenario!')
                scenarios = extract_scenarios_from_dataset(cfg, worker)
    else:
        if cfg.cache.use_cache_without_dataset:
            scenarios = extract_scenarios_from_cache(cfg, worker, model)
        else:
            scenarios = extract_scenarios_from_dataset(cfg, worker)

    logger.info(f'Extracted {len(scenarios)} scenarios for training')
    assert len(scenarios) > 0, 'No scenarios were retrieved for training, check the scenario_filter parameters!'

    return scenarios


def validate_scenario_type_in_cache_path(paths: List[Path]) -> None:
    """
    Checks if scenario_type is in cache path.
    :param path: Scenario cache path
    :return: Whether scenario type is in cache path
    """
    sample_cache_path = paths[0]
    assert all(
        not char.isdigit() for char in sample_cache_path.parent.name
    ), "Unable to filter cache by scenario types as it was generated without scenario type information. Please regenerate a new cache if scenario type filtering is required."


def create_closed_loop_scenario_from_paths(paths: List[Path]) -> List[AbstractScenario]:
    """
    Create scenario objects from a list of cache paths in the format of ".../log_name/scenario_token".
    :param paths: List of paths to load scenarios from.
    :return: List of created scenarios.
    """
    scenarios = [
        CachedScenario(
            log_name=path.parent.parent.name,
            token=path.name,
            scenario_type=path.parent.name,
            closed_loop_scenario_path=path
        )
        for path in paths
    ]

    return scenarios

def create_scenario_from_paths(paths: List[Path]) -> List[AbstractScenario]:
    """
    Create scenario objects from a list of cache paths in the format of ".../log_name/scenario_token".
    :param paths: List of paths to load scenarios from.
    :return: List of created scenarios.
    """
    scenarios = [
        CachedScenario(
            log_name=path.parent.parent.name,
            token=path.name,
            scenario_type=path.parent.name,
        )
        for path in paths
    ]

    return scenarios

def create_scenario_from_records(records: List[Dict]) -> List[AbstractScenario]:
    """
    Create scenario objects from a list of cached scenario records.
    :param paths: List of paths to load scenarios from.
    :return: List of created scenarios.
    """
    scenarios = [
        CachedScenario(
            log_name=record["log_name"],
            token=record["token"],
            scenario_type=record["scenario_type"],
            lidarpc_tokens=record["lidarpc_tokens"],
        )
        for record in records
    ]

    return scenarios
