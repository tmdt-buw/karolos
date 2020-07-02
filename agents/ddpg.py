"""
https://spinningup.openai.com/en/latest/algorithms/sac.html

"""

import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

from agents.nnfactory.ddpg import Policy, Critic
from agents.utils.replay_buffer import ReplayBuffer

# todo make device parameter of Agent
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('\n DEVICE', device, '\n')


class AgentDDPG:
    def __init__(self, config, observation_space, action_space,
                 experiment_folder="."):

        self.writer = SummaryWriter("debug")

        self.observation_space = observation_space
        self.action_space = action_space

        state_dim = self.observation_space.shape
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

        self.target_entropy = -1 * action_dim[0]

        # generate networks
        self.critic = Critic(state_dim, action_dim,
                             self.critic_structure).to(
            device)
        self.target_critic = Critic(state_dim, action_dim,
                                    self.critic_structure).to(device)
        self.policy = Policy(in_dim=state_dim, action_dim=action_dim,
                             network_structure=self.policy_structure).to(
            device)
        self.target_policy = Policy(in_dim=state_dim, action_dim=action_dim,
                                    network_structure=self.policy_structure).to(
            device)

        # Adam and AdamW adapt their learning rates, no need for manual lr decay/cycling
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

        self.memory = ReplayBuffer(buffer_size=self.memory_size)

    def learn(self, step):

        self.policy.train()
        self.critic.train()
        self.target_critic.train()
        self.target_policy.train()

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(device)

        rewards *= self.reward_scale

        predicted_value = self.critic(states, actions)

        predicted_next_action = self.policy(next_states)

        value_target = self.target_critic(next_states,
                                          self.policy(next_states))
        value_target.mul_(1 - dones)
        value_target.add_(rewards)

        value_predicted = self.target_critic(states, actions)

        critic_loss = self.criterion_critic(value_predicted, value_target)

        # Train critic

        target_q_value = rewards + (1 - dones) * self.reward_discount * \
                         self.target_critic(next_states, predicted_next_action)

        q_val_loss1 = self.criterion_critic(predicted_value,
                                            target_q_value.detach())

        self.optimizer_critic.zero_grad()
        q_val_loss1.backward()
        self.optimizer_critic.step()

        # Training policy
        predicted_action = self.policy(states)
        loss_policy = -self.critic(states, predicted_action).mean()

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
        for experience in experiences:
            self.memory.add(experience)

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
        try:
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
        except FileNotFoundError:
            print('###################')
            print('Could not load agent, missing files in', path)
            print('###################')
            raise

        if train_mode:
            self.policy.train()
            self.critic.train()
            self.target_critic.train()
        else:
            self.policy.eval()
            self.critic.eval()
            self.target_critic.eval()

    def predict(self, states, deterministic=True):

        self.policy.eval()

        states = torch.FloatTensor(states).to(device)

        action = self.policy(states, deterministic)

        action = action.detach().cpu().numpy()

        action = action.clip(self.action_space.low, self.action_space.high)

        return action

    def set_target_entropy(self, target_e):
        self.target_entropy = target_e
        return


if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt
    import pprint
    from tqdm import tqdm

    agent_config = {
        "learning_rate_critic": 0.001,
        "learning_rate_policy": 0.001,
        "batch_size": 128,
        "weight_decay": 0,
        "reward_discount": 0.99,
        "memory_size": 100_000,
        "tau": 0.005,
        "policy_structure": [('linear', 256), ('relu', None)] * 2,
        "critic_structure": [('linear', 256), ('relu', None)] * 2,
    }

    pprint.pprint(agent_config)


    # agent.save(".")
    # agent.load(".")

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

    agent = AgentDDPG(agent_config, env.observation_space, env.action_space)

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

        rewards.append(episode_reward)

        if eps % (episodes // 5) == 0 and eps > 0:
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
