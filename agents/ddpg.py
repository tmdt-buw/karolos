"""
https://spinningup.openai.com/en/latest/algorithms/sac.html

"""

import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from agents import Agent
from agents.utils.nn import NeuralNetwork, init_xavier_uniform


class Policy(NeuralNetwork):
    def __init__(self, state_dims, action_dim, network_structure):
        in_dim = int(
            np.sum([np.product(state_dim) for state_dim in state_dims]))

        out_dim = int(np.product(action_dim))

        super(Policy, self).__init__(in_dim, network_structure)

        dummy = super(Policy, self).forward(torch.zeros((1, in_dim)))

        self.operators.append(nn.Linear(dummy.shape[1], out_dim))

        self.operators.apply(init_xavier_uniform)

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

        self.operators.apply(init_xavier_uniform)

    def forward(self, *args):
        return super(Critic, self).forward(*args)


class AgentDDPG(Agent):
    def __init__(self, config, observation_space, action_space,
                 experiment_dir="."):

        super(AgentDDPG, self).__init__(config, observation_space,
                                        action_space, experiment_dir)

        self.learning_rate_critic = config.get("learning_rate_critic", 5e-4)
        self.learning_rate_policy = config.get("learning_rate_policy", 5e-4)
        self.weight_decay = config.get("weight_decay", 1e-4)
        self.tau = config.get('tau', 2.5e-3)

        self.policy_structure = config['policy_structure']
        self.critic_structure = config['critic_structure']

        # generate networks
        self.critic = Critic(self.state_dim, self.action_dim,
                             self.critic_structure).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim,
                                    self.critic_structure).to(self.device)
        self.policy = Policy(self.state_dim, self.action_dim,
                             self.policy_structure).to(self.device)
        self.target_policy = Policy(self.state_dim, self.action_dim,
                                    self.policy_structure).to(self.device)

        self.optimizer_critic = torch.optim.AdamW(self.critic.parameters(),
                                                  lr=self.learning_rate_critic,
                                                  weight_decay=self.weight_decay)
        self.optimizer_policy = torch.optim.AdamW(self.policy.parameters(),
                                                  lr=self.learning_rate_policy,
                                                  weight_decay=self.weight_decay)

        self.update_target(self.policy, self.target_policy, 1.)
        self.update_target(self.critic, self.target_critic, 1.)

        self.criterion_critic = nn.MSELoss()

    def learn(self):
        self.policy.train()
        self.critic.train()
        self.target_critic.train()
        self.target_policy.train()

        experiences, indices = self.memory.sample(self.batch_size)

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

        self.update_priorities(indices, predicted_value, target_value)

        # Update target
        self.update_target(self.critic, self.target_critic, self.tau)

        if self.writer:
            self.writer.add_histogram('rewards', rewards, self.learning_step)
            self.writer.add_histogram('predicted_value', predicted_value, self.learning_step)

        self.learning_step += self.sample_training_ratio

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

    def load(self, path):
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
