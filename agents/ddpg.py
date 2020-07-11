"""
https://spinningup.openai.com/en/latest/algorithms/sac.html

"""

import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard.writer import SummaryWriter

from agents.utils.nn import NeuralNetwork
# from agents.nnfactory.ddpg import Policy, Critic
from agents.utils.replay_buffer import ReplayBuffer


class Policy(NeuralNetwork):
    def __init__(self, state_dims, action_dim, network_structure):
        in_dim = int(
            np.sum([np.product(state_dim) for state_dim in state_dims]))

        out_dim = int(np.product(action_dim))

        super(Policy, self).__init__(in_dim, network_structure)

        dummy = super(Policy, self).forward(torch.zeros((1, in_dim)))

        self.operators.append(nn.Linear(dummy.shape[1], out_dim))

    def forward(self, *state_args, deterministic=True):
        action = super(Policy, self).forward(*state_args)

        if not deterministic:
            normal = Normal(action, torch.ones_like(action))
            action = normal.rsample()

        return action


class Critic(NeuralNetwork):
    def __init__(self, state_dims, action_dim, network_structure):
        in_dim = int(
            np.sum([np.product(arg) for arg in state_dims]) + np.product(
                action_dim))

        super(Critic, self).__init__(in_dim, network_structure)

        dummy = super(Critic, self).forward(torch.zeros((1, in_dim)))

        self.operators.append(nn.Linear(dummy.shape[1], 1))

    def forward(self, *args):
        return super(Critic, self).forward(*args)


class AgentDDPG:
    def __init__(self, config, observation_space, action_space,
                 reward_function, experiment_dir="."):

        self.observation_space = observation_space
        self.action_space = action_space

        state_dim = (sum(map(np.product, [observation_space["robot"].shape,
                                          observation_space["task"].shape,
                                          observation_space["goal"].shape])),)
        action_dim = self.action_space.shape

        assert len(state_dim) == 1
        assert len(action_dim) == 1

        self.learning_rate_critic = config["learning_rate_critic"]
        self.learning_rate_policy = config["learning_rate_policy"]
        self.weight_decay = config["weight_decay"]
        self.batch_size = config['batch_size']
        self.reward_discount = config['reward_discount']
        self.reward_scale = config.get('reward_scale')
        self.memory_size = config['memory_size']
        self.tau = config['tau']

        self.policy_structure = config['policy_structure']
        self.critic_structure = config['critic_structure']

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # generate networks
        self.critic = Critic(state_dim, action_dim,
                             self.critic_structure).to(self.device)
        self.target_critic = Critic(state_dim, action_dim,
                                    self.critic_structure).to(self.device)
        self.policy = Policy(state_dim, action_dim,
                             self.policy_structure).to(self.device)
        self.target_policy = Policy(state_dim, action_dim,
                                    self.policy_structure).to(self.device)

        self.optimizer_critic = torch.optim.AdamW(self.critic.parameters(),
                                                  lr=self.learning_rate_critic,
                                                  weight_decay=self.weight_decay)
        self.optimizer_policy = torch.optim.AdamW(self.policy.parameters(),
                                                  lr=self.learning_rate_policy,
                                                  weight_decay=self.weight_decay)

        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_policy.parameters(),
                                       self.policy.parameters()):
            target_param.data.copy_(param.data)

        self.criterion_critic = nn.MSELoss()

        self.memory = ReplayBuffer(self.memory_size, reward_function)

        self.writer = SummaryWriter(os.path.join(experiment_dir, "debug"),
                                    "debug")

    def learn(self, step):

        self.policy.train()
        self.critic.train()
        self.target_critic.train()
        self.target_policy.train()

        experiences = self.memory.sample(self.batch_size)

        states, goals, actions, rewards, next_states, dones = experiences

        states = torch.FloatTensor(states).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(
            self.device)

        rewards *= self.reward_scale

        predicted_value = self.critic(states, goals, actions)

        predicted_next_action = self.policy(next_states, goals)

        # Train critic
        target_value = rewards + (1 - dones) * self.reward_discount * \
                       self.target_critic(next_states, goals,
                                          predicted_next_action)

        critic_loss = self.criterion_critic(predicted_value, target_value)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Training policy
        predicted_action = self.policy(states, goals)
        loss_policy = -self.critic(states, goals, predicted_action).mean()

        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        self.optimizer_policy.step()

        # Update target
        for target_critic_layer_parameters, critic_layer_parameters in \
                zip(self.target_critic.parameters(),
                    self.critic.parameters()):
            target_critic_layer_parameters.data.copy_(
                target_critic_layer_parameters.data * (
                        1.0 - self.tau) +
                critic_layer_parameters.data * self.tau
            )

    def add_experience(self, experiences):
        rewards = []

        for experience in experiences:
            state, action, next_state, done, goal = experience
            reward = self.memory.add(state, action, next_state, done, goal)
            rewards.append(reward)

        return rewards

    def save(self, path):

        if not osp.exists(path):
            os.makedirs(path)

        torch.save(self.policy.state_dict(), osp.join(path, "policy.pt"))
        torch.save(self.critic.state_dict(), osp.join(path, "critic_1.pt"))
        torch.save(self.target_critic.state_dict(),
                   osp.join(path, "target_critic_1.pt"))

        torch.save(self.optimizer_policy.state_dict(),
                   osp.join(path, "optimizer_policy.pt"))
        torch.save(self.optimizer_critic.state_dict(),
                   osp.join(path, "optimizer_critic_1.pt"))

    def load(self, path, train_mode=True):
        self.policy.load_state_dict(
            torch.load(osp.join(path, "policy.pt")))
        self.critic.load_state_dict(
            torch.load(osp.join(path, "critic_1.pt")))
        self.target_critic.load_state_dict(
            torch.load(osp.join(path, "target_critic_1.pt")))

        self.optimizer_policy.load_state_dict(
            torch.load(osp.join(path, "optimizer_policy.pt")))
        self.optimizer_critic.load_state_dict(
            torch.load(osp.join(path, "optimizer_critic_1.pt")))

    def predict(self, states, goals, deterministic=True):

        self.policy.eval()

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        goals = torch.tensor(goals, dtype=torch.float).to(self.device)

        action = self.policy(states, goals, deterministic=deterministic)

        action = action.detach().cpu().numpy()

        action = action.clip(self.action_space.low, self.action_space.high)

        return action

    def set_target_entropy(self, target_entropy):
        self.target_entropy = target_entropy
