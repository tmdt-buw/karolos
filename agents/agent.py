import os

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from agents.utils.replay_buffer import ReplayBuffer
from utils import unwind_space_shapes


class Agent:

    def __init__(self, config, observation_space, action_space,
                 experiment_dir=None):

        self.observation_space = observation_space
        self.action_space = action_space

        observation_shapes = unwind_space_shapes(observation_space)

        self.state_dim = (sum(map(np.product, observation_shapes)),)
        self.action_dim = self.action_space.shape

        assert len(self.state_dim) == 1
        assert len(self.action_dim) == 1

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.memory_size = config['memory_size']
        self.memory = ReplayBuffer(self.memory_size)

        if experiment_dir:
            self.writer = SummaryWriter(os.path.join(experiment_dir, "debug"),
                                        "debug")
        else:
            self.writer = None

        self.sample_training_ratio = config["sample_training_ratio"]
        self.learning_step = 0

    def train(self, total_samples):
        if self.sample_training_ratio:
            # train for every batch of newly collected samples (specified by `sample_training_ratio`)
            learning_steps = (
                                         total_samples - self.learning_step) // self.sample_training_ratio

            for _ in range(learning_steps):
                self.learn()
        else:
            # train once, regardless of collected samples. In this case total_samples should be displayed on x-axis of performance plot
            self.learning_step = total_samples
            self.learn()

    def learn(self):
        raise NotImplementedError()

    def add_experiences(self, experiences):
        for experience in experiences:
            self.memory.add(experience)

    @staticmethod
    def update_target(network, target_network, tau):
        for network_parameters, target_network_parameters in \
                zip(network.parameters(), target_network.parameters()):
            target_network_parameters.data.copy_(
                target_network_parameters.data * (
                        1. - tau) + network_parameters.data * tau)
