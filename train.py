"""
Main training script for the project.
"""

import argparse
import warnings
from typing import Final

from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1
from smarts.env.utils.action_conversion import ActionOptions
from smarts.env.utils.observation_conversion import ObservationOptions
from smarts.sstudio.scenario_construction import build_scenarios

warnings.filterwarnings("ignore")

N_AGENTS = 3
AGENT_IDS: Final[list] = ["Agent %i" % i for i in range(N_AGENTS)]


class RandomLanerAgent(Agent):
    def __init__(self, action_space) -> None:
        self._action_space = action_space

    def act(self, obs, **kwargs):
        return self._action_space.sample()


class KeepLaneAgent(Agent):
    def __init__(self, action_space) -> None:
        self._action_space = action_space

    def act(self, obs, **kwargs):
        return self._action_space.sample()


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    # This interface must match the action returned by the agent
    agent_interfaces = {
        agent_id: AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        )
        for agent_id in AGENT_IDS
    }
    defaults = dict(
        agent_interfaces=agent_interfaces,
        scenarios=scenarios,
        headless=headless,
    )

    env = HiWayEnvV1(
        observation_options=ObservationOptions.full,
        action_options=ActionOptions.full,
        **defaults,
    )

    for episode in episodes(n=num_episodes):
        agents = {
            agent_id: RandomLanerAgent(env.action_space[agent_id])
            for agent_id in agent_interfaces.keys()
        }
        observations, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        terminateds = {"__all__": False}
        while not terminateds["__all__"]:
            actions = {
                agent_id: agent.act(observations) for agent_id, agent in agents.items()
            }
            observations, rewards, terminateds, truncateds, infos = env.step(actions)
            episode.record_step(observations, rewards, terminateds, truncateds, infos)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("robotics_proj")
    parser.add_argument(
        "--episodes",
        help="The number of episodes to run the simulation for.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--headless", help="Run the simulation in headless mode.", action="store_true"
    )
    parser.add_argument(
        "--max_episode_steps",
        help="Maximum number of steps to run each episode for.",
        type=int,
        default=100,
    )

    args = parser.parse_args()
    scens = ["./sumo"]

    build_scenarios(scenarios=scens)

    main(
        scenarios=scens,
        headless=args.headless,
        num_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )
