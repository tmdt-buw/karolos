"""
https://spinningup.openai.com/en/latest/algorithms/sac.html

"""

import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard.writer import SummaryWriter

from agents.utils.nn import NeuralNetwork, Clamp, init_xavier_uniform
from agents.utils.replay_buffer import ReplayBuffer


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


class AgentSAC:
    def __init__(self, config, observation_space, action_space,
                 reward_function,
                 experiment_dir="."):

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
        self.learning_rate_entropy_regularization = config[
            "learning_rate_entropy_regularization"]
        self.entropy_regularization = config["entropy_regularization"]
        self.weight_decay = config["weight_decay"]
        self.batch_size = config['batch_size']
        self.reward_discount = config.get('reward_discount', .99)
        self.reward_scale = config.get('reward_scale', 1.)
        self.memory_size = config['memory_size']
        self.tau = config['tau']
        self.automatic_entropy_regularization = config[
            'automatic_entropy_regularization']

        self.policy_structure = config['policy_structure']
        self.critic_structure = config['critic_structure']

        self.target_entropy = -1 * action_dim[0]

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # generate networks
        self.critic_1 = Critic(state_dim, action_dim,
                               self.critic_structure).to(
            self.device)
        self.critic_2 = Critic(state_dim, action_dim,
                               self.critic_structure).to(
            self.device)
        self.target_critic_1 = Critic(state_dim, action_dim,
                                      self.critic_structure).to(self.device)
        self.target_critic_2 = Critic(state_dim, action_dim,
                                      self.critic_structure).to(self.device)
        self.policy = Policy(state_dim, action_dim, self.policy_structure).to(
            self.device)

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
        self.entropy_regularization_optim = torch.optim.AdamW(
            [self.log_entropy_regularization],
            lr=self.learning_rate_entropy_regularization,
            weight_decay=self.weight_decay)

        for target_param, param in zip(self.target_critic_1.parameters(),
                                       self.critic_1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic_2.parameters(),
                                       self.critic_2.parameters()):
            target_param.data.copy_(param.data)

        self.criterion_critic_1 = nn.MSELoss()
        self.criterion_critic_2 = nn.MSELoss()

        self.memory = ReplayBuffer(self.memory_size, reward_function)

        self.writer = SummaryWriter(os.path.join(experiment_dir, "debug"),
                                    "debug")

    def learn(self, step):

        self.policy.train()
        self.critic_1.train()
        self.critic_2.train()
        self.target_critic_1.train()
        self.target_critic_2.train()

        experiences = self.memory.sample(self.batch_size)

        states, goals, actions, rewards, next_states, dones = experiences

        rewards *= self.reward_scale

        states = torch.FloatTensor(states).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(
            self.device)

        predicted_q1 = self.critic_1(states, goals, actions)
        predicted_q2 = self.critic_2(states, goals, actions)
        self.writer.add_histogram('predicted_q1', predicted_q1, step)
        self.writer.add_histogram('predicted_q2', predicted_q2, step)

        predicted_action, log_prob = self.policy(states, goals,
                                                 deterministic=False)
        predicted_next_action, next_log_prob = self.policy(next_states, goals,
                                                           deterministic=False)

        self.writer.add_histogram('rewards', rewards, step)

        if self.automatic_entropy_regularization is True:
            entropy_regularization_loss = -(self.log_entropy_regularization * (
                    log_prob + self.target_entropy).detach()).mean()
            self.writer.add_scalar('entropy_regularization_loss',
                                   entropy_regularization_loss, step)

            self.entropy_regularization_optim.zero_grad()
            entropy_regularization_loss.backward()
            self.entropy_regularization_optim.step()
            self.entropy_regularization = self.log_entropy_regularization.exp()
            self.writer.add_scalar('entropy_regularization',
                                   self.entropy_regularization, step)
        else:
            self.entropy_regularization = 1.

        # Train critic
        target_critic_min = torch.min(
            self.target_critic_1(next_states, goals, predicted_next_action),
            self.target_critic_2(next_states, goals, predicted_next_action))
        self.writer.add_histogram('target_critic_min_1', target_critic_min,
                                  step)

        target_critic_min.sub_(self.entropy_regularization * next_log_prob)
        self.writer.add_histogram('target_critic_min_2', target_critic_min,
                                  step)

        target_q_value = rewards + (
                1 - dones) * self.reward_discount * target_critic_min
        self.writer.add_histogram('target_q_value', target_q_value, step)

        q_val_loss1 = self.criterion_critic_1(predicted_q1,
                                              target_q_value.detach())
        self.writer.add_scalar('q_val_loss1', q_val_loss1.item(), step)

        q_val_loss2 = self.criterion_critic_2(predicted_q2,
                                              target_q_value.detach())
        self.writer.add_scalar('q_val_loss2', q_val_loss2.item(), step)

        self.optimizer_critic_1.zero_grad()
        self.optimizer_critic_2.zero_grad()

        q_val_loss1.backward()
        q_val_loss2.backward()

        self.optimizer_critic_1.step()
        self.optimizer_critic_2.step()

        # errors = torch.max(nn.MSELoss(reduction="none")(predicted_q1,
        #                                                 target_q_value.detach()),
        #                    nn.MSELoss(reduction="none")(predicted_q2,
        #                                                 target_q_value.detach())).flatten()
        #
        # self.update_experience(indices, errors)

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
        for target_critic_layer_parameters, critic_layer_parameters in \
                zip(self.target_critic_1.parameters(),
                    self.critic_1.parameters()):
            target_critic_layer_parameters.data.copy_(
                target_critic_layer_parameters.data * (
                        1.0 - self.tau) +
                critic_layer_parameters.data * self.tau
            )

        for target_critic_layer_parameters, critic_layer_parameters in \
                zip(self.target_critic_2.parameters(),
                    self.critic_2.parameters()):
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

    # def update_experience(self, indices, errors):
    #     for index, error in zip(indices, errors):
    #         self.memory.update(index, error)

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


if __name__ == '__main__':
    import gym
    from gym import spaces
    import matplotlib.pyplot as plt
    import pprint
    from tqdm import tqdm

    agent_config = {
        "learning_rate_critic": 3e-4,
        "learning_rate_policy": 3e-4,
        "entropy_regularization": 1,
        "learning_rate_entropy_regularization": 3e-4,
        "batch_size": 256,
        "weight_decay": 0,
        "reward_discount": 0.99,
        "reward_scale": 1,
        "memory_size": 1_000_000,
        "tau": 5e-3,
        "policy_structure": [('linear', 256), ('relu', None)] * 2,
        "critic_structure": [('linear', 256), ('relu', None)] * 2,
        "automatic_entropy_regularization": True,
    }

    pprint.pprint(agent_config)


    # agent.save(".")
    # agent.load(".")

    class NormalizedEnv(gym.ActionWrapper):
        def action(self, action):
            low = self.action_space.low
            high = self.action_space.high

            action = low + (action + 1.0) * 0.5 * (high - low)
            action = np.clip(action, low, high)

            return action

        def reverse_action(self, action):
            low = self.action_space.low
            high = self.action_space.high

            action = 2 * (action - low) / (high - low) - 1
            action = np.clip(action, low, high)

            return action


    max_steps = 200
    episodes = 50

    rewards = []

    env = NormalizedEnv(gym.make("Pendulum-v0"))
    # env = NormalizedActions(gym.make("LunarLanderContinuous-v2"))

    observation_space = spaces.Dict({
        'robot': env.observation_space,
        'task': spaces.Box(-1, 1, shape=(0,)),
        'goal': spaces.Box(-1, 1, shape=(1,))
    })


    def reward_function(goal, **kwargs):
        reward = goal["achieved"]

        return reward


    agent = AgentSAC(agent_config, observation_space, env.action_space,
                     reward_function)

    task = np.array([]).reshape((0,))
    goal = np.zeros(1)

    for eps in tqdm(range(episodes)):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):

            action = \
                agent.predict(np.expand_dims(state, 0),
                              np.expand_dims(goal, 0),
                              deterministic=False)[0]

            next_state, reward, _, _ = env.step(action)

            goal_reached = next_state[0] > .98 and next_state[2] < 1e-5
            done = goal_reached

            experience = [
                {
                    "robot": state,
                    "task": task
                },
                action,
                {
                    "robot": next_state,
                    "task": task
                },
                done,
                {
                    "desired": goal,
                    "achieved": np.array([reward]),
                    "reached": goal_reached
                }
            ]

            reward = agent.add_experience([experience])[0]

            episode_reward += reward
            if eps * max_steps + step > agent_config["batch_size"]:
                agent.learn(eps * max_steps + step)

            state = next_state

            if done:
                break

        # agent.learn(eps)

        rewards.append(episode_reward)

        if eps % min(25, episodes // 5) == 0 and eps > 0:
            plt.plot(rewards)
            plt.show()

    plt.plot(rewards)
    plt.show()

    while True:
        state = env.reset()

        env.render()

        for step in range(max_steps):

            action = \
                agent.predict(np.expand_dims(state, 0),
                              np.expand_dims(goal, 0))[0]

            next_state, reward, _, _ = env.step(action)

            goal_reached = next_state[0] > .98 and next_state[2] < 1e-5
            done = goal_reached

            env.render()

            state = next_state

            if done:
                break
