import gc
import itertools
import logging
import os
import uuid
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig

from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder, RepartitionStrategy
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.training.experiments.cache_metadata_entry import (
    CacheMetadataEntry,
    CacheResult,
    save_cache_metadata,
)
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import chunk_list, worker_map

logger = logging.getLogger(__name__)


def cache_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
    """
    Performs the caching of scenario DB files in parallel.
    :param args: A list of dicts containing the following items:
        "scenario": the scenario as built by scenario_builder
        "cfg": the DictConfig to use to process the file.
    :return: A dict with the statistics of the job. Contains the following keys:
        "successes": The number of successfully processed scenarios.
        "failures": The number of scenarios that couldn't be processed.
    """
    # Define a wrapper method to help with memory garbage collection.
    # This way, everything will go out of scope, allowing the python GC to clean up after the function.
    #
    # This is necessary to save memory when running on large datasets.
    def cache_scenarios_internal(args: List[Dict[str, Union[List[AbstractScenario], DictConfig]]]) -> List[CacheResult]:
        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        scenarios: List[AbstractScenario] = [a["scenario"] for a in args]
        cfg: DictConfig = args[0]["cfg"]

        model = build_torch_module_wrapper(cfg.model)
        feature_builders = model.get_list_of_required_feature()
        target_builders = model.get_list_of_computed_target()
        if cfg.cache.force_recompute_features is not None:
            for builder in feature_builders + target_builders:
                if builder.get_feature_unique_name() in cfg.cache.force_recompute_features:
                    builder.force_recompute = True

        # Now that we have the feature and target builders, we do not need the model any more.
        # Delete it so it gets gc'd and we can save a few system resources.
        del model

        # Create feature preprocessor
        assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"
        preprocessor = FeaturePreprocessor(
            cache_path=cfg.cache.cache_path,
            force_feature_computation=cfg.cache.force_feature_computation,
            feature_builders=feature_builders,
            target_builders=target_builders,
            versatile_cache=cfg.cache.versatile_caching,
        )

        logger.info("Extracted %s scenarios for thread_id=%s, node_id=%s.", str(len(scenarios)), thread_id, node_id)
        num_failures = 0
        num_successes = 0
        all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []
        failed_scenarios: List[str] = []
        for idx, scenario in enumerate(scenarios):
            logger.info(
                "Processing scenario %s / %s in thread_id=%s, node_id=%s",
                idx + 1,
                len(scenarios),
                thread_id,
                node_id,
            )

            file_cache_metadata_cur_scenario: List[Optional[CacheMetadataEntry]] = []
            num_failures_cur_scenario = 0
            num_successes_cur_scenario = 0

            for iteration in range(scenario.get_number_of_iterations()):
                features, targets, file_cache_metadata = preprocessor.compute_features(scenario, iteration)
                cur_scenario_fail = any([feature is None for feature in itertools.chain(features.values(), targets.values())])
                if cur_scenario_fail:
                    break

                scenario_num_failures = sum(
                    0 if feature.is_valid else 1 for feature in itertools.chain(features.values(), targets.values())
                )
                scenario_num_successes = len(features.values()) + len(targets.values()) - scenario_num_failures
                num_failures_cur_scenario += scenario_num_failures
                num_successes_cur_scenario += scenario_num_successes
                file_cache_metadata_cur_scenario += file_cache_metadata
            
            if not cur_scenario_fail:
                num_failures += num_failures_cur_scenario
                num_successes += num_successes_cur_scenario
                all_file_cache_metadata += file_cache_metadata_cur_scenario
            else:
                failed_scenarios.append(scenario.token)

        logger.info("Finished processing scenarios for thread_id=%s, node_id=%s", thread_id, node_id)
        return [CacheResult(failures=num_failures, successes=num_successes, cache_metadata=all_file_cache_metadata, failed_scenarios=failed_scenarios)]

    result = cache_scenarios_internal(args)

    # Force a garbage collection to clean up any unused resources
    gc.collect()

    return result


def build_scenarios_from_config(
    cfg: DictConfig, scenario_builder: AbstractScenarioBuilder, worker: WorkerPool
) -> List[AbstractScenario]:
    """
    Build scenarios from config file.
    :param cfg: Omegaconf dictionary
    :param scenario_builder: Scenario builder.
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: A list of scenarios
    """
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    return scenario_builder.get_scenarios(scenario_filter, worker)  # type: ignore


def cache_data(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the lightning datamodule and cache all samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """
    assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"

    scenario_builder = build_scenario_builder(cfg)
    if int(os.environ.get("NUM_NODES", 1)) > 1 and cfg.distribute_by_scenario:
        # Partition differently based on how the scenario builder loads the data
        repartition_strategy = scenario_builder.repartition_strategy
        if repartition_strategy == RepartitionStrategy.REPARTITION_FILE_DISK:
            scenario_filter = DistributedScenarioFilter(
                cfg=cfg,
                worker=worker,
                node_rank=int(os.environ.get("NODE_RANK", 0)),
                num_nodes=int(os.environ.get("NUM_NODES", 1)),
                synchronization_path=cfg.cache.cache_path,
                timeout_seconds=cfg.get("distributed_timeout_seconds", 3600),
                distributed_mode=cfg.get("distributed_mode", DistributedMode.LOG_FILE_BASED),
            )
            scenarios = scenario_filter.get_scenarios()
        elif repartition_strategy == RepartitionStrategy.INLINE:
            scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)
            num_nodes = int(os.environ.get("NUM_NODES", 1))
            node_id = int(os.environ.get("NODE_RANK", 0))
            scenarios = chunk_list(scenarios, num_nodes)[node_id]
        else:
            expected_repartition_strategies = [e.value for e in RepartitionStrategy]
            raise ValueError(
                f"Expected repartition strategy to be in {expected_repartition_strategies}, got {repartition_strategy}."
            )
    else:
        logger.debug(
            "Building scenarios without distribution, if you're running on a multi-node system, make sure you aren't"
            "accidentally caching each scenario multiple times!"
        )
        scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)

    data_points = [{"scenario": scenario, "cfg": cfg} for scenario in scenarios]
    logger.info("Starting dataset caching of %s files...", str(len(data_points)))

    cache_results = worker_map(worker, cache_scenarios, data_points)

    num_success = sum(result.successes for result in cache_results)
    num_fail = sum(result.failures for result in cache_results)
    num_total = num_success + num_fail
    logger.info("Completed dataset caching! Failed features and targets: %s out of %s", str(num_fail), str(num_total))

    if cfg.cache.get('versatile_caching', False):
        logger.info(f"Saving versatile cache pkl to {cfg.cache.versatile_cache_pickle_file}.")
        versatile_cache_pickle_file = Path(cfg.cache.versatile_cache_pickle_file)
        all_failed_scenarios = [token for cache_result in cache_results for token in cache_result.failed_scenarios]
        scenario_metadata = [
            {
                "log_name": scenario.log_name,
                "token": scenario.token,
                "scenario_type": scenario.scenario_type,
                "lidarpc_tokens": scenario._lidarpc_tokens,
            } for scenario in scenarios if scenario.token not in all_failed_scenarios
        ]
        cache_metadata = cache_results[0].cache_metadata
        all_features = [cache_meta.file_name.stem for cache_meta in cache_metadata]
        all_features = list(set(all_features))
        scenario_metadata.append(all_features)
        with open(versatile_cache_pickle_file, 'wb') as f:
            pickle.dump(scenario_metadata, f)
    else:
        cached_metadata = [
            cache_metadata_entry
            for cache_result in cache_results
            for cache_metadata_entry in cache_result.cache_metadata
            if cache_metadata_entry is not None
        ]

        node_id = int(os.environ.get("NODE_RANK", 0))
        logger.info(f"Node {node_id}: Storing metadata csv file containing cache paths for valid features and targets...")
        save_cache_metadata(cached_metadata, Path(cfg.cache.cache_path), node_id)
        logger.info("Done storing metadata csv file.")
