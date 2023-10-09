import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
import torch.nn as nn
import torch.nn.init as init

logger = logging.getLogger(__name__)


def kaiming_normal_init(m):
    # For Convolutional layers
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    
    # For Linear layers
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)


def build_torch_module_wrapper(cfg: DictConfig) -> TorchModuleWrapper:
    """
    Builds the NN module.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of TorchModuleWrapper.
    """
    logger.info('Building TorchModuleWrapper...')
    model = instantiate(cfg)
    # model.apply(kaiming_normal_init)
    validate_type(model, TorchModuleWrapper)
    logger.info('Building TorchModuleWrapper...DONE!')

    return model
