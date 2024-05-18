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

from hiway_env.custom_obs import observation_adapter

torch, nn = try_import_torch()


def obs_adapter(agent_observation, /):
    return gym.spaces.utils.flatten(
        OBSERVATION_SPACE, observation_adapter.transform(agent_observation)
    )


# ACTION_SPACE
ACTION_SPACE = gym.spaces.Tuple(
    (
        gym.spaces.Box(low=0, high=60, shape=(), dtype=float),
        gym.spaces.Box(low=0, high=2, shape=(), dtype=int),
    )
)

# OBSERVATION_SPACE
OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "distance_from_center": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
        "speed": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        "steering": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        "ego_lane_dist": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "ego_ttc": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
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
        path_to_model = Path(path_to_model)  # Ensure path is a Path object
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._model = torch.load(path_to_model)
        self._model.eval()

    def act(self, obs):
        obs = self._prep.transform(obs)
        obs_tensor = torch.tensor([obs], dtype=torch.float32)
        with torch.no_grad():
            action_tensor = self._model(obs_tensor)
        action = action_tensor.numpy()[0]
        return action


rllib_agent = {
    "agent_spec": AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.LanerWithSpeed,
            neighborhood_vehicle_states=True,
            max_episode_steps=500,
        ),
        agent_params={
            "path_to_model": Path(__file__).resolve().parent / "model.pt",
            "observation_space": FLATTENED_OBSERVATION_SPACE,
        },
        agent_builder=RLlibTorchSavedModelAgent,
        observation_adapter=obs_adapter,
        # action_adapter=action_adapter,
    ),
    "observation_space": FLATTENED_OBSERVATION_SPACE,
    "action_space": ACTION_SPACE,
}
