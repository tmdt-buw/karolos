import os

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from agents.utils.replay_buffer import ReplayBuffer


class Agent:

    def __init__(self, config, observation_space, action_space,
                 experiment_dir=None):

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
        self.memory = ReplayBuffer(self.memory_size)

        if experiment_dir:
            self.writer = SummaryWriter(os.path.join(experiment_dir, "debug"),
                                        "debug")
        else:
            self.writer = None

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
