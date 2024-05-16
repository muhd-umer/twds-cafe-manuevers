"""
RLlib custom agent implementation for SMARTS;

Uses PyTorch unlike the reference implementation which uses TensorFlow; for ease
of familiarity with the PyTorch API

Action Space:
- Discrete: lane_change -> right lane, keep lane, left lane
- Continuous: target_speed

Observation Space:
- speed
- steering
- ego_ttc
- ego_lane_dist
"""

from pathlib import Path

import gymnasium as gym
import numpy as np

# Ray RLlib
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict

# SMARTS
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.zoo.agent_spec import AgentSpec

from utils.custom_obs import observation_adapter

torch, nn = try_import_torch()


def observation_adapter(agent_observation, /):
    return gym.spaces.utils.flatten(
        OBSERVATION_SPACE, observation_adapter.transform(agent_observation)
    )


def action_adapter(agent_action, /):
    throttle, brake, steering = agent_action
    return np.array([throttle, brake, steering * np.pi * 0.25], dtype=np.float32)


# ACTION_SPACE
ACTION_SPACE = gym.spaces.Dict(
    {
        "target_speed": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
        "lane_change": gym.spaces.Discrete(3),
    }
)

# OBSERVATION_SPACE
OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
    }
)

# FLATTENED_OBSERVATION_SPACE
FLATTENED_OBSERVATION_SPACE = gym.spaces.utils.flatten_space(OBSERVATION_SPACE)


# Custom Model
class TrainingModel(FullyConnectedNetwork):
    NAME = "FullyConnectedNetwork"

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return super().forward(input_dict, state, seq_lens)


# Register the custom model
ModelCatalog.register_custom_model(TrainingModel.NAME, TrainingModel)


class RLlibTorchSavedModelAgent(Agent):
    def __init__(self, path_to_model, observation_space, policy_name="default_policy"):
        path_to_model = str(path_to_model)  # might be a str or a Path, normalize to str
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._model = torch.load(path_to_model)
        self._model.eval()

    def act(self, obs):
        obs = self._prep.transform(obs)
        obs = torch.tensor(obs).float()
        action = self._model(obs)
        return action.detach().numpy()


rllib_agent = {
    "agent_spec": AgentSpec(
        interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=500),
        agent_params={
            "path_to_model": Path(__file__).resolve().parent / "model",
            "observation_space": FLATTENED_OBSERVATION_SPACE,
        },
        agent_builder=RLlibTorchSavedModelAgent,
        observation_adapter=observation_adapter,
        action_adapter=action_adapter,
    ),
    "observation_space": FLATTENED_OBSERVATION_SPACE,
    "action_space": ACTION_SPACE,
}
