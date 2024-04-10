import os
from typing import List, Type

import alf
import torch
from alf.algorithms.data_transformer import create_data_transformer
from alf.data_structures import (AlgStep, Experience, LossInfo, StepType,
                                 TimeStep)
from alf.trainers import policy_trainer
from alf.trainers.policy_trainer import Trainer
from alf.utils import common
from alf.utils.checkpoint_utils import Checkpointer
from nuplan_extent.planning.simulation.logsim_environment import LogSimObservation
from torch.utils.data import default_collate

from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks, Observation)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput, PlannerReport)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import \
    transform_predictions_to_states
from nuplan.planning.simulation.trajectory.abstract_trajectory import \
    AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import \
    InterpolatedTrajectory
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import \
    AbstractFeatureBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

class AlfPlanner(AbstractPlanner):
    def __init__(
            self, 
            root_dir: str, 
            additional_conf_params: List[str], 
            feature_builders: List[AbstractFeatureBuilder],
            future_trajectory_sampling: TrajectorySampling
        ) -> None:
        if torch.cuda.is_available():
            alf.set_default_device('cuda')
        conf_file = os.path.join(root_dir, "config_files/hybrid_conf.py")
        ckpt_dir = os.path.join(root_dir, "train/algorithm")
        assert os.path.isfile(conf_file), conf_file
        assert os.path.isdir(ckpt_dir), ckpt_dir
        self.conf_file = conf_file
        self.ckpt_dir = ckpt_dir
        self.conf_params = additional_conf_params
        self.feature_builders = feature_builders

        alf.parse_config(self.conf_file, self.conf_params)
        config = policy_trainer.TrainerConfig(root_dir="")
        env = alf.get_env()
        env.reset()
        data_transformer = create_data_transformer(config.data_transformer_ctor, env.observation_spec())
        config.data_transformer = data_transformer
        common.set_global_env(env)
        observation_spec = data_transformer.transformed_observation_spec
        common.set_transformed_observation_spec(observation_spec)

        algorithm_ctor = config.algorithm_ctor
        algorithm = algorithm_ctor(observation_spec=observation_spec, action_spec=env.action_spec(), reward_spec=env.reward_spec(), config=config)
        algorithm.set_path('')

        checkpointer = Checkpointer(ckpt_dir=self.ckpt_dir, algorithm=algorithm, trainer_progress=Trainer._trainer_progress)
        recovered_global_step = checkpointer.load("latest", ignored_parameter_prefixes=[], including_optimizer=False, including_replay_buffer=False, including_data_transformers=True, strict=True)
        algorithm.eval()
        self.algorithm = algorithm

        self._future_horizon = future_trajectory_sampling.time_horizon
        self._step_interval = future_trajectory_sampling.step_time

    def initialize(self, initialization: PlannerInitialization) -> None:
        torch.set_grad_enabled(False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialization = initialization
        self.rl_state = self.algorithm.get_initial_predict_state(1)
        self._initialized = True

    def name(self) -> str:
        return self.__class__.__name__

    def observation_type(self) -> type[Observation]:
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        history = current_input.history

        features = {}
        for builder in self.feature_builders:
            features[builder.get_feature_unique_name()] = builder.get_features_from_simulation(current_input, self._initialization)
        features = {name: feature.to_feature_tensor() for name, feature in features.items()}
        features = {name: feature.to_device(self.device) if hasattr(feature, 'to_device') else feature.to(self.device) for name, feature in features.items()}
        features = {name: feature.collate([feature]) if hasattr(feature, 'collate') else default_collate([feature]) for name, feature in features.items()}

        observation = LogSimObservation(features=features, targets=None, sim_type=torch.tensor(1), valid_target=False, scenario_id=9999)

        time_step = TimeStep(
            step_type=StepType.FIRST,
            reward=torch.tensor(0.),
            discount=0.99,
            observation=observation,
            prev_action=None,
            env_id=0,
            untransformed=None,
            env_info=None,
        )


        alg_step = self.algorithm.predict_step(time_step, self.rl_state)
        self.rl_state = alg_step.state
        
        alg_trajectory = torch.concat([alg_step.output.x, alg_step.output.y, alg_step.output.heading], dim=0).T.cpu().numpy()
        assert alg_trajectory.shape == (16, 3)
        # Convert relative poses to absolute states and wrap in a trajectory object.
        states = transform_predictions_to_states(
            alg_trajectory, history.ego_states, self._future_horizon, self._step_interval
        )
        trajectory = InterpolatedTrajectory(states)

        return trajectory