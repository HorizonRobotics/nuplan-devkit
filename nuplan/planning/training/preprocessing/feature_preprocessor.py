from __future__ import annotations

import logging
import pathlib
import traceback
from functools import wraps
from typing import List, Optional, Tuple, Type, Union

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle, FeatureCacheS3
from nuplan.planning.training.preprocessing.utils.utils_cache import compute_or_load_feature

logger = logging.getLogger(__name__)


def handle_errors_if_needed(func):
    """
    Decorator that performs exception check only when versatile_cache is False (Default behavior).
    """
    @wraps(func)
    def wrapper(instance, *args, **kwargs):
        if instance._versatile_cache:
            return func(instance, *args, **kwargs)
        else:
            try:
                return func(instance, *args, **kwargs)
            except Exception as error:
                msg = (
                    f"Failed to compute features for scenario token {args[0].token} in log {args[0].log_name}\n"
                    f"Error: {error}"
                )
                logger.error(msg)
                traceback.print_exc()
                raise RuntimeError(msg)
    return wrapper


class FeaturePreprocessor:
    """
    Compute features and targets for a scenario. This class also manages cache. If a feature/target
    is not present in a cache, it is computed, otherwise it is loaded
    """

    def __init__(
        self,
        cache_path: Optional[str],
        force_feature_computation: bool,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        versatile_cache: bool = False,
    ):
        """
        Initialize class.
        :param cache_path: Whether to cache features.
        :param force_feature_computation: If true, even if cache exists, it will be overwritten.
        :param feature_builders: List of feature builders.
        :param target_builders: List of target builders.
        """
        self._cache_path = pathlib.Path(cache_path) if cache_path else None
        self._force_feature_computation = force_feature_computation
        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._storing_mechanism = (
            FeatureCacheS3(cache_path) if str(cache_path).startswith('s3://') else FeatureCachePickle()
        )
        self._versatile_cache = versatile_cache

        assert len(feature_builders) != 0, "Number of feature builders has to be grater than 0!"

    @property
    def feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: all feature builders
        """
        return self._feature_builders

    @property
    def target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: all target builders
        """
        return self._target_builders

    def get_list_of_feature_types(self) -> List[Type[AbstractModelFeature]]:
        """
        :return all features that are computed by the builders
        """
        return [builder.get_feature_type() for builder in self._feature_builders]

    def get_list_of_target_types(self) -> List[Type[AbstractModelFeature]]:
        """
        :return all targets that are computed by the builders
        """
        return [builder.get_feature_type() for builder in self._target_builders]

    @handle_errors_if_needed
    def compute_features(
        self, scenario: AbstractScenario, iteration: int=0
    ) -> Tuple[FeaturesType, TargetsType, List[CacheMetadataEntry]]:
        """
        Compute features for a scenario, in case cache_path is set, features will be stored in cache,
        otherwise just recomputed
        * perform try/except when not versatile_cache
        :param scenario for which features and targets should be computed
        :return: model features and targets and cache metadata
        """
        all_features: FeaturesType
        all_feature_cache_metadata: List[CacheMetadataEntry]
        all_targets: TargetsType
        all_targets_cache_metadata: List[CacheMetadataEntry]

        all_features, all_feature_cache_metadata = self._compute_all_features(scenario, self._feature_builders, iteration)
        all_targets, all_targets_cache_metadata = self._compute_all_features(scenario, self._target_builders, iteration)

        all_cache_metadata = all_feature_cache_metadata + all_targets_cache_metadata
        return all_features, all_targets, all_cache_metadata

    def _compute_all_features(
        self, scenario: AbstractScenario, builders: List[Union[AbstractFeatureBuilder, AbstractTargetBuilder]], iteration: int
    ) -> Tuple[Union[FeaturesType, TargetsType], List[Optional[CacheMetadataEntry]]]:
        """
        Compute all features/targets from builders for scenario
        * perform try/except when versatile_cache
        :param scenario: for which features should be computed
        :param builders: to use for feature computation
        :return: computed features/targets and the metadata entries for the computed features/targets
        """
        # Features to be computed
        all_features: FeaturesType = {}
        all_features_metadata_entries: List[CacheMetadataEntry] = []

        for builder in builders:
            if self._versatile_cache:
                try:
                    feature, feature_metadata_entry = compute_or_load_feature(
                        scenario, self._cache_path, builder, self._storing_mechanism, self._force_feature_computation, iteration, self._versatile_cache,           )
                    all_features[builder.get_feature_unique_name()] = feature
                    all_features_metadata_entries.append(feature_metadata_entry)
                except Exception as error:
                    msg = (
                        f"Failed to compute {builder.get_feature_unique_name()} for scenario token {scenario.token} in log {scenario.log_name}\n"
                        f"Error: {error}"
                    )
                    logger.error(msg)
                    all_features[builder.get_feature_unique_name()] = None
                    all_features_metadata_entries.append(None)
            else:
                feature, feature_metadata_entry = compute_or_load_feature(
                    scenario, self._cache_path, builder, self._storing_mechanism, self._force_feature_computation, iteration, self._versatile_cache,           )
                all_features[builder.get_feature_unique_name()] = feature
                all_features_metadata_entries.append(feature_metadata_entry)

        return all_features, all_features_metadata_entries
