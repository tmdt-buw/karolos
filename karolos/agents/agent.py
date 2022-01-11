import os
import pathlib
import sys
import warnings

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
from replay_buffers import get_replay_buffer

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils import unwind_space_shapes, unwind_dict_values

from gym import spaces


class Agent:

    def __init__(self, config, observation_space, action_space,
                 reward_function=None,
                 experiment_dir=None):

        self.observation_space = observation_space
        self.action_space = action_space

        if reward_function is None:
            warnings.warn(message="""Reward function not specified. Using a constant reward of 0.""",
                          category=UserWarning,
                          )

            reward_function = lambda **kwargs: 0

        self.reward_function = reward_function

        observation_shapes = unwind_space_shapes(observation_space)

        self.state_dim = (sum(map(np.product, observation_shapes)),)

        if type(self.action_space) is spaces.Box:
            self.action_dim = self.action_space.shape
        elif type(self.action_space) is spaces.Discrete:
            self.action_dim = (self.action_space.n,)

        assert len(self.state_dim) == 1
        assert len(self.action_dim) == 1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.batch_size = config.get('batch_size', 64)
        self.reward_discount = config.get('reward_discount', .99)
        self.reward_scale = config.get('reward_scale', 1.)
        self.her_ratio = config.get('her_ratio', 0.)

        buffer_config = config.get('replay_buffer', {"name": "fifo", "buffer_size": int(1e6)})
        self.memory = get_replay_buffer(buffer_config)

        if experiment_dir:
            self.writer = SummaryWriter(os.path.join(experiment_dir, "agent"), "agent")
        else:
            self.writer = None

        self.sample_training_ratio = config.get("sample_training_ratio", 0)
        self.learning_step = 0

    def train(self, total_samples):
        if self.sample_training_ratio:
            # train for every batch of newly collected samples (specified by `sample_training_ratio`)
            learning_steps = (total_samples - self.learning_step) // self.sample_training_ratio

            for _ in range(learning_steps):
                self.learn()
        else:
            # train once, regardless of collected samples. In this case total_samples should be displayed on x-axis of performance plot
            self.learning_step = total_samples
            self.learn()

    def learn(self):
        raise NotImplementedError()

    def add_experience_trajectory(self, trajectory):
        assert len(trajectory) % 2

        trajectory_length = len(trajectory) // 2

        rewards = []

        for trajectory_step in range(trajectory_length):
            state, _ = trajectory[trajectory_step * 2]
            action = trajectory[trajectory_step * 2 + 1]
            next_state, goal_info = trajectory[trajectory_step * 2 + 2]
            done = trajectory_step == len(trajectory) // 2 - 1
            reward = self.reward_function(goal_info=goal_info, done=done)

            state = unwind_dict_values(state)
            next_state = unwind_dict_values(next_state)

            experience = {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            }

            self.memory.add(experience)
            rewards.append(reward)

        for _ in range(int(self.her_ratio * len(trajectory) // 2)):
            # sample random step
            trajectory_step = np.random.randint(len(trajectory) // 2)

            state, _ = trajectory[trajectory_step * 2]
            action = trajectory[trajectory_step * 2 + 1]
            next_state, goal_info = trajectory[trajectory_step * 2 + 2]
            done = trajectory_step == len(trajectory) // 2 - 1

            # sample future goal_info
            goal_step = np.random.randint(trajectory_step, len(trajectory) // 2)
            _, future_goal_info = trajectory[goal_step * 2]

            goal_info = goal_info.copy()
            goal_info["desired"] = future_goal_info["achieved"]

            reward = self.reward_function(goal_info=goal_info, done=done)

            state = unwind_dict_values(state)
            next_state = unwind_dict_values(next_state)

            experience = {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            }

            self.memory.add(experience)

        return rewards

    def update_priorities(self, indices, predicted_values, target_values):
        if self.memory.uses_priority:
            errors = (predicted_values - target_values).abs().flatten().detach().cpu().numpy()

            for idx, error in zip(indices, errors):
                self.memory.update(idx, error)

    @staticmethod
    def update_target(network, target_network, tau):
        for network_parameters, target_network_parameters in zip(network.parameters(), target_network.parameters()):
            target_network_parameters.data.copy_(
                target_network_parameters.data * (1. - tau) + network_parameters.data * tau)


class OnPolAgent(Agent):
    def __init__(self, config, observation_space, action_space,
                 reward_function=None,
                 experiment_dir=None):
        super(OnPolAgent, self).__init__(config, observation_space,
                                         action_space, reward_function, experiment_dir)
        self.is_on_policy = True
        buffer_config = config.get('replay_buffer',
                                   {"name": "OnPolBuffer",
                                    "size": config.get("batch_size", 96000),
                                    "number_envs": config.get("number_threads", 1) * config.get("number_processes", 1),
                                    "state_shape": self.state_dim[0],
                                    "action_shape": self.action_dim[0],
                                    "device": self.device})

        assert buffer_config['name'] == "OnPolBuffer"
        self.memory = get_replay_buffer(buffer_config)

    def add_experiences(self, experiences, env_id):
        if (not self.memory.is_full()) and (env_id not in self.memory.full_ids()):
            for exp in experiences:
                self.memory.add(exp, env_id)

    def add_experience_trajectory(self, trajectory):
        assert len(trajectory) % 2

        trajectory_length = len(trajectory) // 2

        rewards = []

        for trajectory_step in range(trajectory_length):
            next_state, goal_info = trajectory[trajectory_step * 2 + 2]
            done = trajectory_step == len(trajectory) // 2 - 1
            reward = self.reward_function(goal_info=goal_info, done=done)
            rewards.append(reward)
        return rewards