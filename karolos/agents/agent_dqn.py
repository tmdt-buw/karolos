"""
Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013.

"""

import os
import os.path as osp
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent))

from agent import Agent
from nn import NeuralNetwork, init_xavier_uniform


class Policy(NeuralNetwork):
    def __init__(self, state_dims, action_dim, network_structure):
        in_dim = int(np.sum([np.product(state_dim) for state_dim in state_dims]))

        out_dim = int(np.product(action_dim))

        super(Policy, self).__init__(in_dim, network_structure)

        dummy = super(Policy, self).forward(torch.zeros((1, in_dim)))

        self.operators.append(nn.Linear(dummy.shape[1], out_dim))

        self.operators.apply(init_xavier_uniform)

    def forward(self, *state_args):
        action = super(Policy, self).forward(*state_args)

        return action


class AgentDQN(Agent):
    def __init__(self, config, state_space, goal_space, action_space, reward_function, experiment_dir=None):

        super(AgentDQN, self).__init__(config, state_space, goal_space, action_space, reward_function, experiment_dir)

        learning_rate_policy = config.get("learning_rate", 5e-4)
        weight_decay = config.get("weight_decay", 1e-4)

        exploration_probability_config = config.get("exploration_probability", {})

        exploration_probability_start = exploration_probability_config.get("start", 1)
        exploration_probability_end = exploration_probability_config.get("end", 0)
        exploration_probability_steps = exploration_probability_config.get("steps", np.inf)

        self.tau = config.get('tau', 2.5e-3)
        self.exploration_probability = lambda step: max(1 - step / exploration_probability_steps, 0) * (
                exploration_probability_start - exploration_probability_end) + exploration_probability_end

        policy_structure = config.get('policy_structure', [])

        # generate networks
        self.policy = Policy((self.state_dim, self.goal_dim), self.action_dim, policy_structure).to(self.device)
        self.target_policy = Policy((self.state_dim, self.goal_dim), self.action_dim, policy_structure).to(self.device)

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate_policy, weight_decay=weight_decay)

        self.update_target(self.policy, self.target_policy, 1.)

        self.loss_function_q = nn.MSELoss()

    def learn(self):

        self.policy.train()

        experiences, indices = self.replay_buffer.sample(self.batch_size)

        states, goals, actions, rewards, next_states, next_goals, dones = experiences

        states = torch.FloatTensor(states).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_goals = torch.FloatTensor(next_goals).to(self.device)
        dones = torch.FloatTensor(np.float32(dones)).to(self.device)

        rewards *= self.reward_scale

        action_values = self.policy(states, goals)
        action_values = action_values[torch.arange(len(action_values)), actions]

        next_action_values = self.target_policy(next_states, next_goals)
        next_action_values_max, _ = next_action_values.max(-1)

        target_q_values = rewards + (1 - dones) * self.reward_discount * next_action_values_max

        loss = self.loss_function_q(action_values, target_q_values)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.update_priorities(indices, action_values, target_q_values)

        # Update target
        self.update_target(self.policy, self.target_policy, self.tau)

        if self.writer:
            self.writer.add_histogram('predicted_action_values', action_values, self.learning_step)
            self.writer.add_histogram('rewards', rewards, self.learning_step)
            self.writer.add_histogram('target_q_value', target_q_values, self.learning_step)
            self.writer.add_scalar('loss', loss.item(), self.learning_step)
            self.writer.add_scalar('exploration probability', self.exploration_probability(self.learning_step),
                                   self.learning_step)

        self.learning_step += self.sample_training_ratio

    def save(self, path):

        if not osp.exists(path):
            os.makedirs(path)

        torch.save(self.policy.state_dict(), osp.join(path, "q_network.pt"))
        torch.save(self.target_policy.state_dict(), osp.join(path, "target_q_network.pt"))
        torch.save(self.optimizer.state_dict(), osp.join(path, "optimizer.pt"))

    def load(self, path):
        self.policy.load_state_dict(torch.load(osp.join(path, "q_network.pt")))
        self.target_policy.load_state_dict(torch.load(osp.join(path, "target_q_network.pt")))
        self.optimizer.load_state_dict(torch.load(osp.join(path, "optimizer.pt")))

    def predict(self, states, goals, deterministic=True):

        if not deterministic:
            mask_deterministic = np.random.random(len(states)) > self.exploration_probability(self.learning_step)
        else:
            mask_deterministic = np.ones(len(states))

        actions = np.random.randint(0, self.action_space.n, len(states))

        if mask_deterministic.any():
            # at least one action to be determined deterministically
            indices_deterministic = np.argwhere(mask_deterministic).flatten()

            states_deterministic = [states[idx] for idx in indices_deterministic]
            goals_deterministic = [goals[idx] for idx in indices_deterministic]

            self.policy.eval()

            states_deterministic = torch.FloatTensor(states_deterministic).to(self.device)
            goals_deterministic = torch.FloatTensor(goals_deterministic).to(self.device)

            action_values_deterministic = self.policy(states_deterministic, goals_deterministic)

            _, actions_deterministic = action_values_deterministic.max(-1)

            actions_deterministic = actions_deterministic.detach().cpu().numpy()

            actions[indices_deterministic] = actions_deterministic

        return actions,
