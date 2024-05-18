"""
Train multiple agents using the PPO algorithm on a set of scenarios
"""

import warnings
from pathlib import Path
from pprint import pprint
from typing import Dict, Literal, Optional, Union

import gymnasium as gym
import numpy as np
import smarts
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import PolicyID
from ray.tune.logger import CSVLogger
from smarts.sstudio.scenario_construction import build_scenarios
from termcolor import colored

from agent import TrainingModel, rllib_agent
from config import default_parser
from hiway_env import HiWayEnv

gym.logger.set_level(50)
warnings.filterwarnings("ignore")


# Custom metrics to be added to tensorboard
class Callbacks(DefaultCallbacks):
    @staticmethod
    def on_episode_start(
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs,
    ):
        episode.user_data["fei"] = []
        episode.user_data["speed"] = []
        episode.user_data["acceleration"] = []
        episode.user_data["jerk"] = []
        episode.user_data["collisions"] = []

    @staticmethod
    def on_episode_step(
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs,
    ):
        single_agent_id = list(episode.get_agents())[0]
        infos = episode._last_infos.get(single_agent_id)

        if infos is not None:
            episode.user_data["fei"].append(infos["reward"])
            episode.user_data["speed"].append(infos["speed"])
            episode.user_data["acceleration"].append(infos["acceleration"])
            episode.user_data["jerk"].append(infos["jerk"])
            episode.user_data["collisions"].append(infos["collisions"])

    @staticmethod
    def on_episode_end(
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs,
    ):

        fei = np.mean(episode.user_data["fei"])
        speed = np.mean(episode.user_data["speed"])
        acceleration = np.mean(episode.user_data["acceleration"])
        jerk = np.mean(episode.user_data["jerk"])
        collisions = np.mean(episode.user_data["collisions"])

        print(
            colored(
                f"ep. {episode.episode_id:<12} ended; fei={fei:.2f}; speed={speed:.2f}; acceleration={acceleration:.2f}; jerk={jerk:.2f}; collisions={collisions:.2f}",
                "cyan",
            )
        )
        episode.custom_metrics["mean_fei"] = fei
        episode.custom_metrics["mean_speed"] = speed
        episode.custom_metrics["mean_acceleration"] = acceleration
        episode.custom_metrics["mean_jerk"] = jerk
        episode.custom_metrics["mean_collisions"] = collisions

        # Log fei to CSV file
        with open(f"{result_dir}/logs.csv", "a") as f:
            f.write(
                f"{episode.episode_id},{fei},{speed},{acceleration},{jerk},{collisions}\n"
            )


def main(
    scenarios,
    envision,
    time_total,
    rollout_length,
    batch_size,
    seed,
    num_agents,
    num_workers,
    resume,
    result_dir,
    checkpoint_freq: int,
    checkpoint_num: Optional[int],
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"],
):
    rllib_policies = {
        f"AGENT-{i}": (
            None,
            rllib_agent["observation_space"],
            rllib_agent["action_space"],
            {"model": {"custom_model": TrainingModel.NAME}},
        )
        for i in range(num_agents)
    }
    agent_specs = {f"AGENT-{i}": rllib_agent["agent_spec"] for i in range(num_agents)}

    smarts.core.seed(seed)
    assert len(set(rllib_policies.keys()).difference(agent_specs)) == 0
    algo_config: AlgorithmConfig = (
        PPOConfig()
        .environment(
            env=HiWayEnv,
            env_config={
                "seed": seed,
                "scenarios": [
                    str(Path(scenario).expanduser().resolve().absolute())
                    for scenario in scenarios
                ],
                "headless": not envision,
                "agent_specs": agent_specs,
                "observation_options": "multi_agent",
            },
        )
        .framework(framework="torch")  # Use PyTorch framework
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=1,
            rollout_fragment_length=rollout_length,
        )
        .training(
            lr_schedule=[[0, 1e-3], [1e3, 5e-4], [1e5, 1e-4], [1e7, 5e-5], [1e8, 1e-5]],
            train_batch_size=batch_size,
        )
        .multi_agent(
            policies=rllib_policies,
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: f"{agent_id}",
        )
        .callbacks(callbacks_class=Callbacks)
        .debugging(
            logger_config={
                "type": CSVLogger,
                "logdir": result_dir,
            },
            log_level=log_level,
        )
    )

    def get_checkpoint_dir(num):
        checkpoint_dir = Path(result_dir) / f"checkpoint_{num}" / f"checkpoint-{num}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    if resume:
        checkpoint = str(get_checkpoint_dir("latest"))
    if checkpoint_num:
        checkpoint = str(get_checkpoint_dir(checkpoint_num))
    else:
        checkpoint = None

    print(colored(f"\nCheckpointing at {str(result_dir)}", "green"))

    algo = algo_config.build()
    if checkpoint is not None:
        algo.load_checkpoint(checkpoint=checkpoint)
    result = {}
    current_iteration = 0
    checkpoint_iteration = checkpoint_num or 0

    try:
        # Create fei.csv file to log fei data
        with open(f"{result_dir}/logs.csv", "w") as f:
            f.write("episode_id,fei\n")

        while result.get("time_total_s", 0) < time_total:
            result = algo.train()
            print(colored(f"\nIteration {result['training_iteration']}", "blue"))
            pprint(result, depth=1)

            if current_iteration % checkpoint_freq == 0:
                checkpoint_dir = get_checkpoint_dir(checkpoint_iteration)
                print(colored(f"\nSaving checkpoint {checkpoint_iteration}", "yellow"))
                algo.save_checkpoint(checkpoint_dir)
                checkpoint_iteration += 1
            current_iteration += 1
        algo.save_checkpoint(get_checkpoint_dir(checkpoint_iteration))
    finally:
        algo.save_checkpoint(get_checkpoint_dir("latest"))
        algo.stop()


if __name__ == "__main__":
    parser = default_parser("proj")
    parser.add_argument(
        "--checkpoint-num",
        type=int,
        default=None,
        help="Checkpoint number to restart from.",
    )
    parser.add_argument(
        "--rollout-length",
        type=str,
        default="auto",
        help="Episodes are divided into fragments of this many steps for each rollout.",
    )
    args = parser.parse_args()

    if not args.scenarios:
        raise ValueError("Please provide scenarios to train on.")

    result_dir = str(
        Path(__file__).resolve().parent / f"results/{args.scenarios[0].split('/')[-1]}"
    )
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    build_scenarios(scenarios=args.scenarios, clean=False, seed=args.seed)

    main(
        scenarios=args.scenarios,
        envision=args.envision,
        time_total=args.time_total,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        seed=args.seed,
        num_agents=args.num_agents,
        num_workers=args.num_workers,
        resume=args.resume,
        result_dir=result_dir,
        checkpoint_freq=max(args.checkpoint_freq, 1),
        checkpoint_num=args.checkpoint_num,
        log_level=args.log_level,
    )
