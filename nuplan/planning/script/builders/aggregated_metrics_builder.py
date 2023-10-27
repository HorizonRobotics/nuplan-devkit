import logging
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics import Metric

from nuplan.planning.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def build_aggregated_metrics(cfg: DictConfig) -> List[Metric]:
    """
    Build metrics based on config
    :param cfg: config
    :return list of metrics.
    """
    instantiated_metrics = []
    for metric_name, cfg_metric in cfg.aggregated_metric.items():
        new_metric: Metric = instantiate(cfg_metric)
        validate_type(new_metric, Metric)
        instantiated_metrics.append(new_metric)
    return instantiated_metrics
