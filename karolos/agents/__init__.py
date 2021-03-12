import os

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from .replay_buffers import get_replay_buffer
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

        self.batch_size = config.get('batch_size', 64)
        self.reward_discount = config.get('reward_discount', .99)
        self.reward_scale = config.get('reward_scale', 1.)

        buffer_config = config.get('replay_buffer', {"name": "fifo", "buffer_size": int(1e6)})
        self.memory = get_replay_buffer(buffer_config)

        if experiment_dir:
            self.writer = SummaryWriter(os.path.join(experiment_dir, "agent"),
                                        "agent")
        else:
            self.writer = None

        self.sample_training_ratio = config.get("sample_training_ratio", 0)
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

    def update_priorities(self, indices, predicted_values, target_values):
        if self.memory.uses_priority:
            errors = (predicted_values - target_values).abs().flatten().detach().cpu().numpy()

            for idx, error in zip(indices, errors):
                self.memory.update(idx, error)

    @staticmethod
    def update_target(network, target_network, tau):
        for network_parameters, target_network_parameters in \
                zip(network.parameters(), target_network.parameters()):
            target_network_parameters.data.copy_(
                target_network_parameters.data * (
                        1. - tau) + network_parameters.data * tau)


def get_agent(agent_config, observation_space, action_space,
              experiment_dir):

    algorithm = agent_config.pop("algorithm")

    if algorithm == "sac":
        from agents.sac import AgentSAC
        agent = AgentSAC(agent_config, observation_space, action_space,
                         experiment_dir)
    elif algorithm == "ddpg":
        from agents.ddpg import AgentDDPG
        # todo refactor ddpg to match sac
        agent = AgentDDPG(agent_config, observation_space, action_space,
                          experiment_dir)
    else:
        raise NotImplementedError(f"Unknown algorithm {algorithm}")

    return agent
