import os

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

# from agents.nnfactory.ddpg import Policy, Critic
from agents.utils.replay_buffer import ReplayBuffer


class Agent():

    def __init__(self, config, observation_space, action_space,
                 reward_function, experiment_dir="."):

        self.observation_space = observation_space
        self.action_space = action_space

        self.state_dim = (
            sum(map(np.product, [observation_space["robot"].shape,
                                 observation_space["task"].shape,
                                 observation_space["goal"].shape])),)
        self.action_dim = self.action_space.shape

        assert len(self.state_dim) == 1
        assert len(self.action_dim) == 1

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.memory_size = config['memory_size']
        self.memory = ReplayBuffer(self.memory_size, reward_function)

        self.writer = SummaryWriter(os.path.join(experiment_dir, "debug"),
                                    "debug")

    def add_experience(self, experiences):
        rewards = []

        for experience in experiences:
            state, action, next_state, done, goal = experience
            reward = self.memory.add(state, action, next_state, done, goal)
            rewards.append(reward)

        return rewards

    def add_trajectory(self, trajectory):
        assert len(trajectory) % 2

        trajectory_reward = 0

        for trajectory_step in range(len(trajectory) // 2):
            observation = trajectory[trajectory_step * 2]
            action = trajectory[trajectory_step * 2 + 1]
            next_observation = trajectory[trajectory_step * 2 + 2]

            state, _ = observation
            next_state, goal = next_observation

            done = trajectory_step == len(trajectory) // 2 - 1

            reward = self.memory.add(state, action, next_state, done, goal)
            trajectory_reward += reward

        return trajectory_reward

    @staticmethod
    def update_target(network, target_network, tau):
        for network_parameters, target_network_parameters in \
                zip(network.parameters(), target_network.parameters()):
            target_network_parameters.data.copy_(
                target_network_parameters.data * (
                        1. - tau) + network_parameters.data * tau)
