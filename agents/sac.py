"""
https://spinningup.openai.com/en/latest/algorithms/sac.html

"""

import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from agents.agent import Agent
from agents.utils.nn import NeuralNetwork, Clamp, init_xavier_uniform


class Policy(NeuralNetwork):
    def __init__(self, state_dims, action_dim, network_structure,
                 log_std_min=-20, log_std_max=2):
        in_dim = int(
            np.sum([np.product(state_dim) for state_dim in state_dims]))

        out_dim = int(np.product(action_dim)) * 2

        super(Policy, self).__init__(in_dim, network_structure)

        dummy = super(Policy, self).forward(torch.zeros((1, in_dim)))

        self.operators.append(nn.Linear(dummy.shape[1], out_dim))

        self.operators.apply(init_xavier_uniform)

        self.std_clamp = Clamp(log_std_min, log_std_max)

    def forward(self, *state_args, deterministic=True):
        x = super(Policy, self).forward(*state_args)

        mean, log_std = torch.split(x, x.shape[1] // 2, dim=1)

        log_std = self.std_clamp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(log_std.shape[0]).unsqueeze_(-1)
        else:
            std = log_std.exp()

            normal = MultivariateNormal(mean, torch.diag_embed(std.pow(2)))
            action_base = normal.rsample()

            log_prob = normal.log_prob(action_base)
            log_prob.unsqueeze_(-1)

            action = torch.tanh(action_base)

            action_bound_compensation = torch.log(
                1. - action.pow(2) + np.finfo(float).eps).sum(dim=1,
                                                              keepdim=True)

            log_prob.sub_(action_bound_compensation)

        return action, log_prob


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


class AgentSAC(Agent):
    def __init__(self, config, observation_space, action_space, experiment_dir=None):

        super(AgentSAC, self).__init__(config, observation_space, action_space, experiment_dir)

        self.learning_rate_critic = config.get("learning_rate_critic", 5e-4)
        self.learning_rate_policy = config.get("learning_rate_policy", 5e-4)
        self.learning_rate_entropy_regularization = config.get(
            "learning_rate_entropy_regularization", 5e-5)
        self.weight_decay = config.get("weight_decay", 1e-4)
        self.tau = config.get('tau', 2.5e-3)
        self.entropy_regularization = config.get("entropy_regularization", 1)
        self.automatic_entropy_regularization = config.get(
            'automatic_entropy_regularization', True)


        self.policy_structure = config.get('policy_structure', [])
        self.critic_structure = config.get('critic_structure', [])

        self.target_entropy = -1 * self.action_dim[0]

        # generate networks
        self.critic_1 = Critic(self.state_dim, self.action_dim,
                               self.critic_structure).to(self.device)
        self.critic_2 = Critic(self.state_dim, self.action_dim,
                               self.critic_structure).to(self.device)
        self.target_critic_1 = Critic(self.state_dim, self.action_dim,
                                      self.critic_structure).to(self.device)
        self.target_critic_2 = Critic(self.state_dim, self.action_dim,
                                      self.critic_structure).to(self.device)
        self.policy = Policy(self.state_dim, self.action_dim,
                             self.policy_structure).to(self.device)

        self.log_entropy_regularization = torch.tensor(
            [np.log(self.entropy_regularization)], dtype=torch.float,
            requires_grad=True, device=self.device)

        self.optimizer_critic_1 = torch.optim.AdamW(self.critic_1.parameters(),
                                                    lr=self.learning_rate_critic,
                                                    weight_decay=self.weight_decay)
        self.optimizer_critic_2 = torch.optim.AdamW(self.critic_2.parameters(),
                                                    lr=self.learning_rate_critic,
                                                    weight_decay=self.weight_decay)
        self.optimizer_policy = torch.optim.AdamW(self.policy.parameters(),
                                                  lr=self.learning_rate_policy,
                                                  weight_decay=self.weight_decay)
        self.optimizer_entropy_regularization = torch.optim.AdamW(
            [self.log_entropy_regularization],
            lr=self.learning_rate_entropy_regularization,
            weight_decay=self.weight_decay)

        self.update_target(self.critic_1, self.target_critic_1, 1.)
        self.update_target(self.critic_2, self.target_critic_2, 1.)

        self.criterion_critic_1 = nn.MSELoss()
        self.criterion_critic_2 = nn.MSELoss()

    def learn(self):

        self.policy.train()
        self.critic_1.train()
        self.critic_2.train()
        self.target_critic_1.train()
        self.target_critic_2.train()

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

        predicted_value_1 = self.critic_1(states, goals, actions)
        predicted_value_2 = self.critic_2(states, goals, actions)

        predicted_action, log_prob = self.policy(states, goals,
                                                 deterministic=False)
        predicted_next_action, next_log_prob = self.policy(next_states, goals,
                                                           deterministic=False)

        if self.automatic_entropy_regularization is True:
            entropy_regularization_loss = -(self.log_entropy_regularization * (
                    log_prob + self.target_entropy).detach()).mean()

            self.optimizer_entropy_regularization.zero_grad()
            entropy_regularization_loss.backward()
            self.optimizer_entropy_regularization.step()
            self.entropy_regularization = self.log_entropy_regularization.exp()
        else:
            self.entropy_regularization = 1.

        # Train critic
        target_critic_min = torch.min(
            self.target_critic_1(next_states, goals, predicted_next_action),
            self.target_critic_2(next_states, goals, predicted_next_action))

        target_critic_min.sub_(self.entropy_regularization * next_log_prob)

        target_q_value = rewards + (
                1 - dones) * self.reward_discount * target_critic_min

        q_val_loss_1 = self.criterion_critic_1(predicted_value_1,
                                               target_q_value.detach())

        q_val_loss_2 = self.criterion_critic_2(predicted_value_2,
                                               target_q_value.detach())

        self.optimizer_critic_1.zero_grad()
        self.optimizer_critic_2.zero_grad()

        q_val_loss_1.backward()
        q_val_loss_2.backward()

        self.optimizer_critic_1.step()
        self.optimizer_critic_2.step()

        # Training policy
        predicted_new_q_val = torch.min(
            self.critic_1(states, goals, predicted_action),
            self.critic_2(states, goals, predicted_action))
        loss_policy = (
                self.entropy_regularization * log_prob - predicted_new_q_val).mean()

        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        self.optimizer_policy.step()

        # Update target
        self.update_target(self.critic_1, self.target_critic_1, self.tau)
        self.update_target(self.critic_2, self.target_critic_2, self.tau)

        if self.writer:
            self.writer.add_scalar('entropy_regularization', self.entropy_regularization)
            self.writer.add_histogram('predicted_value_1', predicted_value_1, self.learning_step)
            self.writer.add_histogram('predicted_value_2', predicted_value_2, self.learning_step)
            self.writer.add_histogram('rewards', rewards, self.learning_step)
            try:
                self.writer.add_histogram('target_critic_min_1', target_critic_min,
                                          self.learning_step)
            except:
                raise
            self.writer.add_histogram('target_critic_min_2', target_critic_min,
                                      self.learning_step)
            self.writer.add_histogram('target_q_value', target_q_value, self.learning_step)
            self.writer.add_scalar('q_val_loss1', q_val_loss_1.item(), self.learning_step)
            self.writer.add_scalar('q_val_loss2', q_val_loss_2.item(), self.learning_step)

        self.learning_step += self.sample_training_ratio

    def save(self, path):

        if not osp.exists(path):
            os.makedirs(path)

        torch.save(self.policy.state_dict(), osp.join(path, "policy.pt"))
        torch.save(self.critic_1.state_dict(), osp.join(path, "critic_1.pt"))
        torch.save(self.critic_2.state_dict(), osp.join(path, "critic_2.pt"))
        torch.save(self.target_critic_1.state_dict(),
                   osp.join(path, "target_critic_1.pt"))
        torch.save(self.target_critic_2.state_dict(),
                   osp.join(path, "target_critic_2.pt"))

        torch.save(self.optimizer_policy.state_dict(),
                   osp.join(path, "optimizer_policy.pt"))
        torch.save(self.optimizer_critic_1.state_dict(),
                   osp.join(path, "optimizer_critic_1.pt"))
        torch.save(self.optimizer_critic_2.state_dict(),
                   osp.join(path, "optimizer_critic_2.pt"))

    def load(self, path):
        self.policy.load_state_dict(
            torch.load(osp.join(path, "policy.pt")))
        self.critic_1.load_state_dict(
            torch.load(osp.join(path, "critic_1.pt")))
        self.critic_2.load_state_dict(
            torch.load(osp.join(path, "critic_2.pt")))
        self.target_critic_1.load_state_dict(
            torch.load(osp.join(path, "target_critic_1.pt")))
        self.target_critic_2.load_state_dict(
            torch.load(osp.join(path, "target_critic_2.pt")))

        self.optimizer_policy.load_state_dict(
            torch.load(osp.join(path, "optimizer_policy.pt")))
        self.optimizer_critic_1.load_state_dict(
            torch.load(osp.join(path, "optimizer_critic_1.pt")))
        self.optimizer_critic_2.load_state_dict(
            torch.load(osp.join(path, "optimizer_critic_2.pt")))

    def predict(self, states, goals, deterministic=True):

        self.policy.eval()

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        goals = torch.tensor(goals, dtype=torch.float).to(self.device)

        action, _ = self.policy(states, goals, deterministic=deterministic)

        action = action.detach().cpu().numpy()

        action = action.clip(self.action_space.low, self.action_space.high)

        return action

    def set_target_entropy(self, target_entropy):
        self.target_entropy = target_entropy
