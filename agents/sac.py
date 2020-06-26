"""
https://spinningup.openai.com/en/latest/algorithms/sac.html

"""

import os
import os.path as osp

import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

from agents.nnfactory.sac import Policy, Critic
from agents.utils.replay_buffer import ReplayBuffer

# todo make device parameter of Agent
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('\n DEVICE', device, '\n')


class AgentSAC:
    def __init__(self, config, observation_space, action_space, experiment_dir):

        self.observation_space = observation_space
        self.action_space = action_space

        state_dim = self.observation_space.shape
        action_dim = self.action_space.shape

        assert len(state_dim) == 1
        assert len(action_dim) == 1

        self.learning_rate_critic = config["learning_rate_critic"]
        self.learning_rate_policy = config["learning_rate_policy"]
        self.learning_rate_alpha = config["learning_rate_alpha"]
        self.alpha = config["alpha"]
        self.weight_decay = config["weight_decay"]
        self.batch_size = config['batch_size']
        self.reward_discount = config['reward_discount']
        self.memory_size = config['memory_size']
        self.tau = config['tau']
        self.auto_entropy = config['auto_entropy']
        self.grad_clipping = config["gradient_clipping"]
        self.clip_val = 10

        self.policy_structure = config['policy_structure']
        self.critic_structure = config['critic_structure']

        self.target_entropy = -1 * action_dim[0]

        # generate networks
        self.critic_1 = Critic(state_dim, action_dim,
                               self.critic_structure).to(
            device)
        self.critic_2 = Critic(state_dim, action_dim,
                               self.critic_structure).to(
            device)
        self.target_critic_1 = Critic(state_dim, action_dim,
                                      self.critic_structure).to(device)
        self.target_critic_2 = Critic(state_dim, action_dim,
                                      self.critic_structure).to(device)
        self.policy = Policy(in_dim=state_dim, action_dim=action_dim,
                             network_structure=self.policy_structure).to(
            device)

        self.log_alpha = torch.zeros(1, dtype=torch.float32,
                                     requires_grad=True, device=device)

        # Adam and AdamW adapt their learning rates, no need for manual lr decay/cycling
        self.optimizer_critic_1 = torch.optim.AdamW(self.critic_1.parameters(),
                                                    lr=self.learning_rate_critic,
                                                    weight_decay=self.weight_decay)
        self.optimizer_critic_2 = torch.optim.AdamW(self.critic_2.parameters(),
                                                    lr=self.learning_rate_critic,
                                                    weight_decay=self.weight_decay)
        self.optimizer_policy = torch.optim.AdamW(self.policy.parameters(),
                                                  lr=self.learning_rate_policy,
                                                  weight_decay=self.weight_decay)
        self.alpha_optim = torch.optim.AdamW([self.log_alpha],
                                             lr=self.learning_rate_alpha)
                                             #weight_decay=self.weight_decay)

        for target_param, param in zip(self.target_critic_1.parameters(),
                                       self.critic_1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic_2.parameters(),
                                       self.critic_2.parameters()):
            target_param.data.copy_(param.data)

        self.criterion_critic_1 = nn.MSELoss()
        self.criterion_critic_2 = nn.MSELoss()

        self.memory = ReplayBuffer(buffer_size=self.memory_size)

        self.writer = SummaryWriter(osp.join(experiment_dir, "debug"))

    def learn(self, step):

        self.policy.train()
        self.critic_1.train()
        self.critic_2.train()
        self.target_critic_1.train()
        self.target_critic_2.train()

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(device)

        predicted_q1 = self.critic_1(states, actions)
        predicted_q2 = self.critic_2(states, actions)

        predicted_action, log_prob = self.policy(states, deterministic=False)
        predicted_next_action, next_log_prob = self.policy(next_states,
                                                           deterministic=False)

        if self.auto_entropy is True:
            alpha_loss = -(self.log_alpha * (
                    log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.

        # Train critic
        target_critic_min = torch.min(
            self.target_critic_1(next_states, predicted_next_action),
            self.target_critic_2(next_states, predicted_next_action))
        target_critic_min.sub_(self.alpha * next_log_prob)
        target_q_value = rewards + (
                1 - dones) * self.reward_discount * target_critic_min
        q_val_loss1 = self.criterion_critic_1(predicted_q1,
                                              target_q_value.detach())

        q_val_loss2 = self.criterion_critic_2(predicted_q2,
                                              target_q_value.detach())
        self.optimizer_critic_1.zero_grad()
        self.optimizer_critic_2.zero_grad()

        q_val_loss1.backward()
        q_val_loss2.backward()

        if self.grad_clipping:
            torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.clip_val)
            torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.clip_val)

        self.optimizer_critic_1.step()
        self.optimizer_critic_2.step()

        self.writer.add_scalar('q_val_loss1', q_val_loss1.item(), step)
        self.writer.add_scalar('q_val_loss2', q_val_loss2.item(), step)
        self.writer.add_scalar('alpha', self.alpha, step)
        self.writer.add_histogram('rewards', rewards, step)
        self.writer.add_histogram('target_q_value', target_q_value, step)

        # Training policy
        predicted_new_q_val = torch.min(
            self.critic_1(states, predicted_action),
            self.critic_2(states, predicted_action))
        loss_policy = (self.alpha * log_prob - predicted_new_q_val).mean()

        self.optimizer_policy.zero_grad()
        loss_policy.backward()

        if self.grad_clipping:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_val)

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
        for experience in experiences:
            self.memory.add(*experience)

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

    def load(self, path, train_mode=True):
        try:
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
        except FileNotFoundError:
            print('###################')
            print('Could not load agent, missing files in', path)
            print('###################')
            raise

        if train_mode:
            self.policy.train()
            self.critic_1.train()
            self.critic_2.train()
            self.target_critic_1.train()
            self.target_critic_2.train()
        else:
            self.policy.eval()
            self.critic_1.eval()
            self.critic_2.eval()
            self.target_critic_1.eval()
            self.target_critic_2.eval()

    def predict(self, states, deterministic=True):

        self.policy.eval()

        states = torch.FloatTensor(states).to(device)

        action, _ = self.policy(states, deterministic)

        action = action.detach().cpu().numpy()

        action = action.clip(self.action_space.low, self.action_space.high)

        return action

    def set_target_entropy(self, target_entropy):
        self.target_entropy = target_entropy


if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt
    import pprint
    from tqdm import tqdm

    agent_config = {
        "learning_rate_critic": 0.01,
        "learning_rate_policy": 0.01,
        "alpha": 1,
        "learning_rate_alpha": 0.0003,
        "batch_size": 128,
        "weight_decay": 0,
        "reward_discount": 0.99,
        "memory_size": 100_000,
        "tau": 0.01,
        "policy_structure": [('linear', 256), ('relu', None)] * 2,
        "critic_structure": [('linear', 256), ('relu', None)] * 2,
        "auto_entropy": True,
    }

    pprint.pprint(agent_config)


    # agent.save(".")
    # agent.load(".")

    """
    Use this function to test your agents on the pendulum v0 open ai gym. Continuous space
    """

    import gym
    import numpy as np

    import matplotlib.pyplot as plt


    class NormalizedActions(gym.ActionWrapper):
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


    max_steps = 500
    episodes = 50

    rewards = []

    env = NormalizedActions(gym.make("Pendulum-v0"))
    # env = NormalizedActions(gym.make("LunarLanderContinuous-v2"))

    agent = AgentSAC(agent_config, env.observation_space, env.action_space)

    for eps in tqdm(range(episodes)):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):

            action = agent.predict(np.expand_dims(state, 0),
                                   deterministic=False)[0]

            next_state, reward, done, _ = env.step(action)

            experience = [state, action, reward, next_state, done]

            agent.add_experience([experience])
            agent.learn(step)

            state = next_state
            episode_reward += reward

            if done:
                break

        if eps % 10 == 0 and eps > 0:
            plt.plot(rewards)
            plt.show()

    plt.plot(rewards)
    plt.show()

    while True:
        state = env.reset()

        env.render()

        for step in range(max_steps):

            action = agent.predict(np.expand_dims(state, 0))[0]

            next_state, reward, done, _ = env.step(action)

            env.render()

            state = next_state

            if done:
                break
