# Towards Collaborative and Fuel-Efficient Maneuvers for Autonomous Vehicles

The increasing prevalence of autonomous vehicles (AVs) necessitates the development of advanced decision-making algorithms that prioritize safety, fuel efficiency, and passenger comfort. This paper presents a novel approach that leverages multi-agent coordination and deep reinforcement learning (DRL) to achieve these objectives. Building upon existing game-theoretic frameworks for lane-changing scenarios, we propose a system that incorporates interactions with multiple surrounding vehicles. This allows AVs to anticipate and respond strategically to the actions of others, promoting smoother traffic flow and reducing unnecessary braking or acceleration. Furthermore, our framework introduces a fuel optimization variable, enabling AVs to dynamically adjust their speed and trajectory to minimize fuel consumption while maintaining safety and compliance with traffic regulations. The result is an intelligent system that enhances fuel efficiency, passenger comfort, and vehicle safety, contributing to a more efficient and enjoyable driving experience.

## Block Diagram

<p align="center">
<img src="resources/block.png" width="700px"/>
</p>

The environment, simulated using SMARTS, represents a multi-lane highway populated with AVs and human-driven vehicles. The AVs are controlled by trained agents, which receive observations from the environment and generate actions. The agents employ a hybrid proximal policy optimization (H-PPO) algorithm to learn optimal strategies, balancing fuel efficiency, passenger comfort, and safety. Each agent has two separate actor networks, one for discrete lane change decisions and another for continuous speed control. A single critic network estimates the value function, guiding the policy optimization process.

## Usage

To run the project, follow these steps:

1. Clone the repository:
   ```shell
   git clone https://github.com/muhd-umer/twds-cafe-manuevers.git
   ```

2. Install dependencies:
   ```shell
   pip install -r requirements.txt
   ```

3. Train the agents:
   ```shell
   python train.py --scenarios resources/scenarios/lane_changing --time-total 2000 --envision
   ```

4. Run the trained model:
   ```shell
   python train.py --scenarios resources/scenarios/lane_changing --time-total 0 --envision --resume
   ```

*Note: It is recommended to create a new virtual environment so that updates/downgrades of packages do not break other projects.*

The `train.py` script takes several optional arguments:

- `--scenarios`: Path to the scenarios folder containing the SMARTS scenario files.
- `--time-total`: Total training time in seconds.
- `--envision`: Enable visualization of the simulation.
- `--resume`: Resume training from the latest checkpoint.
- `--checkpoint-num`: Restart training from a specific checkpoint.
- `--rollout-length`: Episodes are divided into fragments of this many steps for each rollout.
- `--batch-size`: Number of steps in a training batch.
- `--seed`: Random seed for the simulation.
- `--num-agents`: Number of AVs in the simulation.
- `--num-workers`: Number of parallel workers for training.
- `--checkpoint-freq`: Frequency of checkpoint saving during training.
- `--log-level`: Logging level for debugging.

For detailed instructions on how to customize the simulation setup, modify the agent configuration, or analyze the training results, please refer to the code comments.

## BibTeX
If you find this work useful for your research, please cite our paper:

```
@inproceedings{}
```
