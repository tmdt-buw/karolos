"""
https://spinningup.openai.com/en/latest/algorithms/sac.html

"""

import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

sys.path.append(str(Path(__file__).resolve().parent))

from agent import Agent
from nn import NeuralNetwork, Clamp, init_xavier_uniform, Critic


class Actor(NeuralNetwork):
    def __init__(self, state_dims, action_dim, network_structure, log_std_min=-20, log_std_max=2):
        in_dim = int(np.sum([np.product(state_dim) for state_dim in state_dims]))

        out_dim = int(np.product(action_dim)) * 2

        super(Actor, self).__init__(in_dim, network_structure)

        dummy = super(Actor, self).forward(torch.zeros((1, in_dim)))

        self.operators.append(nn.Linear(dummy.shape[1], out_dim))

        self.operators.apply(init_xavier_uniform)

        self.std_clamp = Clamp(log_std_min, log_std_max)

    def forward(self, *state_args, deterministic=True):
        x = super(Actor, self).forward(*state_args)

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

            action_bound_compensation = torch.log(1. - action.pow(2) + np.finfo(float).eps).sum(dim=1, keepdim=True)

            log_prob.sub_(action_bound_compensation)

        return action, log_prob


class AgentSAC(Agent):
    def __init__(self, config, observation_space, action_space, reward_function, experiment_dir=None):

        super(AgentSAC, self).__init__(config, observation_space, action_space, reward_function, experiment_dir)

        self.replay_buffer.experience_keys += ["expert_action"]

        learning_rate_critic = config.get("learning_rate_critic", 5e-4)
        learning_rate_actor = config.get("learning_rate_actor", 5e-4)
        learning_rate_entropy_regularization = config.get("learning_rate_entropy_regularization", 5e-5)
        weight_decay = config.get("weight_decay", 1e-4)
        entropy_regularization = config.get("entropy_regularization", 1)

        self.automatic_entropy_regularization = config.get('automatic_entropy_regularization', True)
        self.log_entropy_regularization = torch.tensor([np.log(entropy_regularization)], dtype=torch.float,
                                                       requires_grad=True, device=self.device)
        self.target_entropy = -1 * self.action_dim[0]
        self.tau = config.get('tau', 2.5e-3)

        # generate networks
        actor_structure = config.get('actor_structure', [])
        critic_structure = config.get('critic_structure', [])

        self.actor = Actor(self.state_dim, self.action_dim, actor_structure).to(self.device)
        self.critic_1 = Critic(self.state_dim, self.action_dim, critic_structure).to(self.device)
        self.critic_2 = Critic(self.state_dim, self.action_dim, critic_structure).to(self.device)
        self.critic_target_1 = Critic(self.state_dim, self.action_dim, critic_structure).to(self.device)
        self.critic_target_2 = Critic(self.state_dim, self.action_dim, critic_structure).to(self.device)

        self.optimizer_actor = torch.optim.AdamW(self.actor.parameters(), lr=learning_rate_actor,
                                                 weight_decay=weight_decay)
        self.optimizer_critic_1 = torch.optim.AdamW(self.critic_1.parameters(), lr=learning_rate_critic,
                                                    weight_decay=weight_decay)
        self.optimizer_critic_2 = torch.optim.AdamW(self.critic_2.parameters(), lr=learning_rate_critic,
                                                    weight_decay=weight_decay)
        self.optimizer_entropy_regularization = torch.optim.AdamW([self.log_entropy_regularization],
                                                                  lr=learning_rate_entropy_regularization,
                                                                  weight_decay=weight_decay)

        self.update_target(self.critic_1, self.critic_target_1, 1.)
        self.update_target(self.critic_2, self.critic_target_2, 1.)

        self.loss_function_critic_1 = nn.MSELoss()
        self.loss_function_critic_2 = nn.MSELoss()

        self.loss_function_expert_imitation = lambda x, y: 1 - nn.functional.cosine_similarity(x, y).mean()

    def learn(self):

        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.critic_target_1.train()
        self.critic_target_2.train()

        experiences, indices = self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, dones, expert_actions = experiences

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(self.device)
        expert_actions = torch.FloatTensor(expert_actions).to(self.device)

        rewards *= self.reward_scale

        predicted_value_1 = self.critic_1(states, actions)
        predicted_value_2 = self.critic_2(states, actions)

        predicted_actions, log_prob = self.actor(states, deterministic=False)
        predicted_next_actions, next_log_prob = self.actor(next_states, deterministic=False)

        if self.automatic_entropy_regularization is True:
            entropy_regularization_loss = -(
                    self.log_entropy_regularization * (log_prob + self.target_entropy).detach()).mean()

            self.optimizer_entropy_regularization.zero_grad()
            entropy_regularization_loss.backward()
            self.optimizer_entropy_regularization.step()
            entropy_regularization = self.log_entropy_regularization.exp()
        else:
            entropy_regularization = 1.

        # Train critic
        target_critic_min = torch.min(self.critic_target_1(next_states, predicted_next_actions),
                                      self.critic_target_2(next_states, predicted_next_actions))

        target_critic_min.sub_(entropy_regularization * next_log_prob)

        target_q_value = rewards + (1 - dones) * self.reward_discount * target_critic_min

        q_val_loss_1 = self.loss_function_critic_1(predicted_value_1, target_q_value.detach())

        q_val_loss_2 = self.loss_function_critic_2(predicted_value_2, target_q_value.detach())

        self.optimizer_critic_1.zero_grad()
        self.optimizer_critic_2.zero_grad()

        q_val_loss_1.backward()
        q_val_loss_2.backward()

        self.optimizer_critic_1.step()
        self.optimizer_critic_2.step()

        # Training actor
        predicted_new_q_val = torch.min(self.critic_1(states, predicted_actions),
                                        self.critic_2(states, predicted_actions))
        loss_actor = (entropy_regularization * log_prob - predicted_new_q_val).mean()

        # imitate expert

        # replace missing expert actions
        expert_actions_mask = expert_actions.isnan().any(1)
        expert_actions[expert_actions_mask] = predicted_actions[expert_actions_mask]

        loss_imitation = self.loss_function_expert_imitation(predicted_actions, expert_actions)
        loss_imitation_weighted = self.imitation_weight(self.learning_step) * loss_imitation

        self.optimizer_actor.zero_grad()
        loss_actor.backward(retain_graph=True)
        loss_imitation_weighted.backward()
        self.optimizer_actor.step()

        predicted_value_avg = (predicted_value_1 + predicted_value_2) / 2

        self.update_priorities(indices, predicted_value_avg, target_q_value)

        # Update target
        self.update_target(self.critic_1, self.critic_target_1, self.tau)
        self.update_target(self.critic_2, self.critic_target_2, self.tau)

        if self.writer:
            self.writer.add_scalar('entropy_regularization', entropy_regularization, self.learning_step)
            self.writer.add_histogram('predicted_value_1', predicted_value_1, self.learning_step)
            self.writer.add_histogram('predicted_value_2', predicted_value_2, self.learning_step)
            self.writer.add_histogram('rewards', rewards, self.learning_step)
            try:
                self.writer.add_histogram('target_critic_min_1', target_critic_min, self.learning_step)
            except:
                raise
            self.writer.add_histogram('target_critic_min_2', target_critic_min, self.learning_step)
            self.writer.add_histogram('target_q_value', target_q_value, self.learning_step)
            self.writer.add_scalar('q_val_loss1', q_val_loss_1.item(), self.learning_step)
            self.writer.add_scalar('q_val_loss2', q_val_loss_2.item(), self.learning_step)
            self.writer.add_scalar('imitation_weight', self.imitation_weight(self.learning_step), self.learning_step)
            self.writer.add_scalar('loss_imitation', loss_imitation.item(), self.learning_step)

        self.learning_step += self.sample_training_ratio

    def save(self, path):

        if not osp.exists(path):
            os.makedirs(path)

        torch.save(self.actor.state_dict(), osp.join(path, "actor.pt"))
        torch.save(self.critic_1.state_dict(), osp.join(path, "critic_1.pt"))
        torch.save(self.critic_2.state_dict(), osp.join(path, "critic_2.pt"))
        torch.save(self.critic_target_1.state_dict(), osp.join(path, "target_critic_1.pt"))
        torch.save(self.critic_target_2.state_dict(), osp.join(path, "target_critic_2.pt"))

        torch.save(self.optimizer_actor.state_dict(), osp.join(path, "optimizer_actor.pt"))
        torch.save(self.optimizer_critic_1.state_dict(), osp.join(path, "optimizer_critic_1.pt"))
        torch.save(self.optimizer_critic_2.state_dict(), osp.join(path, "optimizer_critic_2.pt"))

    def load(self, path):
        self.actor.load_state_dict(torch.load(osp.join(path, "actor.pt")))
        self.critic_1.load_state_dict(torch.load(osp.join(path, "critic_1.pt")))
        self.critic_2.load_state_dict(torch.load(osp.join(path, "critic_2.pt")))
        self.critic_target_1.load_state_dict(torch.load(osp.join(path, "target_critic_1.pt")))
        self.critic_target_2.load_state_dict(torch.load(osp.join(path, "target_critic_2.pt")))

        self.optimizer_actor.load_state_dict(torch.load(osp.join(path, "optimizer_actor.pt")))
        self.optimizer_critic_1.load_state_dict(torch.load(osp.join(path, "optimizer_critic_1.pt")))
        self.optimizer_critic_2.load_state_dict(torch.load(osp.join(path, "optimizer_critic_2.pt")))

    def predict(self, states, deterministic=True):

        self.actor.eval()

        states = torch.tensor(states, dtype=torch.float).to(self.device)

        with torch.no_grad():
            actions, _ = self.actor(states, deterministic=deterministic)

        actions = actions.detach().cpu().numpy()

        actions = actions.clip(self.action_space.low, self.action_space.high)

        return actions

    def set_target_entropy(self, target_entropy):
        # todo: do we still need this function? Where is it used?
        self.target_entropy = target_entropy
