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

            # for iteration in range(scenario.get_number_of_iterations()):
            features, targets, file_cache_metadata = preprocessor.compute_features(scenario)

            scenario_num_failures = sum(
                0 if feature.is_valid else 1 for feature in itertools.chain(features.values(), targets.values())
            )
            scenario_num_successes = len(features.values()) + len(targets.values()) - scenario_num_failures
            num_failures += scenario_num_failures
            num_successes += scenario_num_successes
            all_file_cache_metadata += file_cache_metadata

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

    # nuplan_e2e_scenarios = []
    # from nuplan_extent.planning.scenario_builder.nuplan_db.nuplan_e2e_scenario import NuPlanE2EScenario
    # for s in scenarios:
    #     e2e_s = NuPlanE2EScenario(
    #         data_root=s._data_root,
    #         log_file_load_path=s._log_file_load_path,
    #         initial_lidar_token=s._initial_lidar_token,
    #         initial_lidar_timestamp=s._initial_lidar_timestamp,
    #         scenario_type=s._scenario_type,
    #         map_root=s._map_root,
    #         map_version=s._map_version,
    #         map_name=s._map_name,
    #         scenario_extraction_info=s._scenario_extraction_info,
    #         ego_vehicle_parameters=s._ego_vehicle_parameters,
    #         sensor_root=s._sensor_root,
    #     )
    #     if e2e_s._log_name == '2021.10.05.04.38.41_veh-50_00996_01109' and e2e_s.token == 'cc2f8b7da7685a63':
    #         continue
    #     nuplan_e2e_scenarios.append(e2e_s)
    # scenarios = nuplan_e2e_scenarios
    # '''
    # Failed to compute features for scenario token cc2f8b7da7685a63 in log 2021.10.05.04.38.41_veh-50_00996_01109
    # '''



    # import pickle
    # from pathlib import Path
    # from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
    # with open('/home/users/siqi01.chai/hoplan-alf/scenario_all_test_no_first.pkl', 'rb') as f:
    #     # to fix a bug while reading csv, we use scenario_all_test_no_first instead of scenario_all_test
    #     scenarios_all = pickle.load(f)
    # scenarios = []

    # for sid, s in enumerate(scenarios_all):
    #     # scenario_model_path = '/mnt/nas20/siqi01.chai/train-scenes-300/train-scene-{}/model-new/checkpoints/PointTexture_stage_0_epoch_23.pth'
    #     scenario_model_path = '/mnt/nas20/siqi01.chai/test-scenes/test-scene-{}/model-new/checkpoints/PointTexture_stage_0_epoch_23.pth'
    #     scenario_model_path = Path(scenario_model_path.format(sid))
    #     new_extraction_info = ScenarioExtractionInfo(
    #         scenario_name=s._scenario_extraction_info.scenario_name,
    #         scenario_duration=s._scenario_extraction_info.scenario_duration,
    #         extraction_offset=s._scenario_extraction_info.extraction_offset,
    #         subsample_ratio=1,
    #         )
    #     s._scenario_extraction_info = new_extraction_info
    #     if scenario_model_path.is_file():
    #         scenarios.append(s)


    # Nuscenes only
    from nuplan_extent.common.maps.nusc_map.nusc_map import NuscMap
    from nuplan_extent.planning.scenario_builder.nuscenes_db.nuscenes_scenario import NuscScenario
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.map_expansion.map_api import NuScenesMap


    # split = 'v1.0-mini'
    # load_dir = '/home/vad/nuscenes-converted-data'
    # nusc =  NuScenes(version=split, dataroot=load_dir, verbose=True)
    
    # scenarios = []
    # nusc_maps = {}
    # from tqdm import tqdm
    # scene_ids = [0,1,2,3,4,5,6,7,8,9]
    # # scene_ids = [0]
    # for id in tqdm(scene_ids):
    #     nusc_scenario = nusc.scene[id]
    #     scene_data = nusc.get('scene', nusc_scenario['token'])
    #     scene_log = nusc.get('log', scene_data['log_token'])
    #     nusc_can_bus = NuScenesCanBus(dataroot=load_dir)
    #     map_name = scene_log['location']
    #     if not map_name in nusc_maps:
    #         map = NuScenesMap(dataroot=load_dir, map_name=map_name)
    #         map_api = NuscMap(nuscenes_map_api=map, map_name=map_name)
    #         nusc_maps[map_name] = map_api
    #     s = NuscScenario(nusc=nusc, nusc_can_bus=nusc_can_bus, 
    #                      nusc_map=nusc_maps[map_name],
    #                      scene_id=id, renderer_path=os.path.join('mini', str(id).zfill(3)), interpolate_N=4)
    #     scenarios.append(s)

    split = 'trainval'
    load_dir = '/mnt/nas26/siqi01.chai/nuscenes-dataset/'
    nusc =  NuScenes(version='v1.0-{}'.format(split), dataroot=load_dir, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=load_dir)
    from nuscenes.utils import splits
    # caching_split = splits.val
    caching_split = splits.train

    import json
    with open('/home/hoplan/nusc_renderer_mapping.json', 'r') as file:
        nusc_renderer_mapping = json.load(file)
    scene_ids = []
    for id, scene in enumerate(nusc.scene):
        if not scene['name'] in caching_split:
            continue
        if not str(id) in nusc_renderer_mapping:
            continue
        renderer_name = nusc_renderer_mapping[str(id)]
        render_base_path = '/mnt/nas26/siqi01.chai/models-drivestudio-nuscenes'
        render = os.path.join(split, renderer_name)
        if not os.path.exists(os.path.join(render_base_path, render, 'checkpoint_final.pth')):
            continue
        try:
            _ = nusc_can_bus.get_messages(scene['name'], 'pose')
        except:
            continue
        scene_ids.append((id, render))
    print(scene_ids)
    print('preparing to cache {} scenes'.format(len(scene_ids)))
    
    scenarios = []
    nusc_maps = {}
    from tqdm import tqdm
    for (id, renderer_path) in tqdm(scene_ids):
        nusc_scenario = nusc.scene[id]
        scene_data = nusc.get('scene', nusc_scenario['token'])
        scene_log = nusc.get('log', scene_data['log_token'])
        nusc_can_bus = NuScenesCanBus(dataroot=load_dir)
        map_name = scene_log['location']
        if not map_name in nusc_maps:
            map = NuScenesMap(dataroot=load_dir, map_name=map_name)
            map_api = NuscMap(nuscenes_map_api=map, map_name=map_name)
            nusc_maps[map_name] = map_api
        s = NuscScenario(nusc=nusc, nusc_can_bus=nusc_can_bus, 
                         nusc_map=nusc_maps[map_name],
                         scene_id=id, renderer_path=renderer_path, interpolate_N=4)
        scenarios.append(s)

    data_points = [{"scenario": scenario, "cfg": cfg} for scenario in scenarios]
    logger.info("Starting dataset caching of %s files...", str(len(data_points)))

    # cache_results = cache_scenarios(data_points)
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
