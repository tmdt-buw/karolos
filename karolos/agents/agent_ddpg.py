"""
https://spinningup.openai.com/en/latest/algorithms/ddpg.html

"""

import os
import os.path as osp
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.append(str(Path(__file__).resolve().parent))

from agent import Agent
from nn import NeuralNetwork, init_xavier_uniform, Critic


class Actor(NeuralNetwork):
    def __init__(self, state_dims, action_dim, network_structure):
        in_dim = int(np.sum([np.product(state_dim) for state_dim in state_dims]))

        out_dim = int(np.product(action_dim))

        super(Actor, self).__init__(in_dim, network_structure)

        dummy = super(Actor, self).forward(torch.zeros((1, in_dim)))

        self.operators.append(nn.Linear(dummy.shape[1], out_dim))

        self.operators.apply(init_xavier_uniform)

    def forward(self, *state_args, deterministic=True):
        action = super(Actor, self).forward(*state_args)

        if not deterministic:
            normal = Normal(action, torch.ones_like(action))
            action = normal.rsample()

        return action


class AgentDDPG(Agent):
    def __init__(self, config, state_space, goal_space, action_space, reward_function, experiment_dir="."):

        super(AgentDDPG, self).__init__(config, state_space, goal_space, action_space, reward_function, experiment_dir)

        self.replay_buffer.experience_keys += ["expert_action"]

        learning_rate_critic = config.get("learning_rate_critic", 5e-4)
        learning_rate_actor = config.get("learning_rate_actor", 5e-4)
        weight_decay = config.get("weight_decay", 1e-4)

        self.tau = config.get('tau', 2.5e-3)

        # generate networks
        actor_structure = config.get('actor_structure', [])
        critic_structure = config.get('critic_structure', [])

        self.actor = Actor((self.state_dim, self.goal_dim), self.action_dim, actor_structure).to(self.device)
        self.actor_target = Actor((self.state_dim, self.goal_dim), self.action_dim, actor_structure).to(self.device)
        self.critic = Critic((self.state_dim, self.goal_dim), self.action_dim, critic_structure).to(self.device)
        self.critic_target = Critic((self.state_dim, self.goal_dim), self.action_dim, critic_structure).to(self.device)

        self.optimizer_actor = torch.optim.AdamW(self.actor.parameters(), lr=learning_rate_actor,
                                                 weight_decay=weight_decay)
        self.optimizer_critic = torch.optim.AdamW(self.critic.parameters(), lr=learning_rate_critic,
                                                  weight_decay=weight_decay)

        self.update_target(self.actor, self.actor_target, 1.)
        self.update_target(self.critic, self.critic_target, 1.)

        self.loss_critic = nn.MSELoss()
        self.loss_expert_imitation = nn.MSELoss()

    def learn(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

        experiences, indices = self.replay_buffer.sample(self.batch_size)

        states, goals, actions, rewards, next_states, next_goals, dones, expert_actions = experiences

        states = torch.FloatTensor(states).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_goals = torch.FloatTensor(next_goals).to(self.device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(self.device)
        expert_actions = torch.FloatTensor(expert_actions).to(self.device)

        rewards *= self.reward_scale

        predicted_value = self.critic(states, goals, actions)

        predicted_next_action = self.actor(next_states, next_goals)

        # Train critic
        target_value = rewards + (1 - dones) * self.reward_discount * self.critic_target(next_states,
                                                                                         predicted_next_action)

        critic_loss = self.loss_critic(predicted_value, target_value)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Training actor
        predicted_actions = self.actor(states)
        loss_actor = -self.critic(states, predicted_actions).mean()

        # imitate expert

        # replace missing expert actions
        expert_actions_mask = expert_actions.isnan().any(1)
        expert_actions[expert_actions_mask] = predicted_actions[expert_actions_mask]

        loss_imitation = self.imitation_weight(self.learning_step) * self.loss_expert_imitation(predicted_actions,
                                                                                                expert_actions)
        self.optimizer_actor.zero_grad()
        loss_actor.backward(retain_graph=True)
        loss_imitation.backward()
        self.optimizer_actor.step()

        self.update_priorities(indices, predicted_value, target_value)

        # Update target
        self.update_target(self.critic, self.critic_target, self.tau)

        if self.writer:
            self.writer.add_histogram('rewards', rewards, self.learning_step)
            self.writer.add_histogram('predicted_value', predicted_value, self.learning_step)

        self.learning_step += self.sample_training_ratio

    def save(self, path):
        if not osp.exists(path):
            os.makedirs(path)

        torch.save(self.actor.state_dict(), osp.join(path, "actor.pt"))
        torch.save(self.critic.state_dict(), osp.join(path, "critic_1.pt"))
        torch.save(self.critic_target.state_dict(), osp.join(path, "target_critic_1.pt"))

        torch.save(self.optimizer_actor.state_dict(), osp.join(path, "optimizer_actor.pt"))
        torch.save(self.optimizer_critic.state_dict(), osp.join(path, "optimizer_critic_1.pt"))

    def load(self, path):
        self.actor.load_state_dict(torch.load(osp.join(path, "actor.pt")))
        self.critic.load_state_dict(torch.load(osp.join(path, "critic_1.pt")))
        self.critic_target.load_state_dict(torch.load(osp.join(path, "target_critic_1.pt")))

        self.optimizer_actor.load_state_dict(torch.load(osp.join(path, "optimizer_actor.pt")))
        self.optimizer_critic.load_state_dict(torch.load(osp.join(path, "optimizer_critic_1.pt")))

    def predict(self, states, goals, deterministic=True):
        self.actor.eval()

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        goals = torch.tensor(goals, dtype=torch.float).to(self.device)

        action = self.actor(states, goals, deterministic=deterministic)

        action = action.detach().cpu().numpy()

        action = action.clip(self.action_space.low, self.action_space.high)

        return action
