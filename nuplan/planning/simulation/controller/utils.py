from typing import List

import torch

from nuplan.common.actor_state.state_representation import TimePoint


def forward_integrate(init: float, delta: float, sampling_time: TimePoint) -> float:
    """
    Performs a simple euler integration.
    :param init: Initial state
    :param delta: The rate of chance of the state.
    :param sampling_time: The time duration to propagate for.
    :return: The result of integration
    """
    return float(init + delta * sampling_time.time_s)


def forward_integrate_tensor(
    init: List[float],
    delta: List[float],
    sampling_time: List[float],
    device: torch.device = None,
) -> torch.Tensor:
    init_tensor = torch.tensor(init, device=device)
    delta_tensor = torch.tensor(delta, device=device)
    time_tensor = torch.tensor([i.time_s for i in sampling_time], device=device)
    return init_tensor + delta_tensor * time_tensor
