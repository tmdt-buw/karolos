"""
https://spinningup.openai.com/en/latest/algorithms/sac.html

"""

import numpy as np
import torch
import torch.nn as nn
import os.path as osp

from agents.nnfactory.sac import PolicyNet, SoftQNetwork
from agents.utils.replay_buffer import ReplayBuffer


# todo make device parameter of Agent
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# print("Using device: -{}- torch.backends.cudnn.enabled = {} torch.cuda.is_available() = {}".format(device, torch.backends.cudnn.enabled, torch.cuda.is_available()))

class AgentSAC:
    def __init__(self, config, state_dim, action_dim):

        #state_dim = np.product(state_dim)
        #action_dim = np.product(action_dim)

        assert len(state_dim) == 1
        assert len(action_dim) == 1

        self.h_dim = config["hidden_dim"]
        self.action_dim = action_dim
        self.soft_q_lr = config["soft_q_lr"]
        self.pol_lr = config["policy_lr"]
        self.alpha_lr = config["alpha_lr"]
        self.alpha = config["alpha"] # tradeoff between exploration and exploitation
        self.weight_decay = config["weight_decay"]
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.memory_size = config['memory_size']
        self.tau = config['tau']
        # self.root_path = config['root_path']
        # self.run_name = config["run_name"]
        self.backup_interval = config["backup_interval"]
        self.auto_entropy = config['auto_entropy']
        self.tb_histogram_interval = config["tensorboard_histogram_interval"]

        self.reward_scale = 10.
        self.target_entropy = -1 * action_dim

        torch.manual_seed(config['seed'])

        # run_no = int(self.run_name.split('_')[1])
        self.no_step = 0
        self.random = True

        self.critic_1 = SoftQNetwork(state_dim, action_dim, self.h_dim).to(device)
        self.critic_2 = SoftQNetwork(state_dim, action_dim, self.h_dim).to(device)
        self.target_critic_1 = SoftQNetwork(state_dim, action_dim, self.h_dim).to(device)
        self.target_critic_2 = SoftQNetwork(state_dim, action_dim, self.h_dim).to(device)
        self.policy = PolicyNet(in_dim=state_dim, action_dim=action_dim, hidden_dim=self.h_dim, device=device).to(device)

        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print("SAC Trainer target entropy:", self.target_entropy)
        print('Soft Q Network (1,2): ', self.critic_1)
        print('Policy Network: ', self.policy)

        # Adam and AdamW adapt their learning rates, no need for manual lr decay/cycling
        self.optimizer_critic_1 = torch.optim.AdamW(self.critic_1.parameters(), lr=self.soft_q_lr, weight_decay=self.weight_decay)
        self.optimizer_critic_2 = torch.optim.AdamW(self.critic_2.parameters(), lr=self.soft_q_lr, weight_decay=self.weight_decay)
        self.optimizer_policy = torch.optim.AdamW(self.policy.parameters(), lr=self.pol_lr, weight_decay=self.weight_decay)
        self.alpha_optim = torch.optim.AdamW([self.log_alpha], lr=self.alpha_lr, weight_decay=self.weight_decay)

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_1_criterion = nn.MSELoss()
        self.soft_q_2_criterion = nn.MSELoss()

        self.memory = ReplayBuffer(buffer_size=self.memory_size, batch_size=self.batch_size, seed=config["seed"])

        # Tensorboard
        # self.tb = SummaryWriter('{}/{}'.format(self.root_path, self.run_name))
        # self.tb.flush()

    def learn(self):
        if not self.random:
            state, actions, reward, next_state, done = self.memory.sample()

            state = torch.FloatTensor(state).to(device)
            actions = torch.FloatTensor(actions).to(device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

            predicted_q1 = self.critic_1(state, actions)
            predicted_q2 = self.critic_2(state, actions)

            new_action, log_prob, _, _, log_std = self.policy.evaluate(state)
            new_next_action, new_log_prob, _, _, _ = self.policy.evaluate(next_state)

            # normalize with batch mean std
            reward = self.reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)

            if self.auto_entropy is True:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.poll()
                self.alpha = self.log_alpha.exp()
            else:
                self.alpha = 1.
                alpha_loss = 0

            # Train Q function
            target_q_min = torch.min(self.target_critic_1(next_state, new_next_action),
                                     self.target_critic_2(next_state, new_next_action)) - self.alpha * new_log_prob
            target_q_value = reward + (1-done) * self.gamma * target_q_min
            q_val_loss1 = self.soft_q_1_criterion(predicted_q1, target_q_value.detach())
            q_val_loss2 = self.soft_q_2_criterion(predicted_q2, target_q_value.detach())
            self.optimizer_critic_1.zero_grad()
            q_val_loss1.backward()
            self.optimizer_critic_1.poll()
            self.optimizer_critic_2.zero_grad()
            q_val_loss2.backward()
            self.optimizer_critic_2.poll()

            # Training Policy Function
            predicted_new_q_val = torch.min(self.critic_1(state, new_action), self.critic_2(state, new_action))
            pol_loss = (self.alpha * log_prob - predicted_new_q_val).mean()

            self.optimizer_policy.zero_grad()
            pol_loss.backward()
            self.optimizer_policy.poll()

            # Target update
            for target_p, p in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_p.data.copy_(
                    target_p.data * (1.0 - self.tau) + p.data * self.tau
                )

            for target_p, p in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_p.data.copy_(
                    target_p.data * (1.0 - self.tau) + p.data * self.tau
                )

            return predicted_new_q_val.mean()

    def add_experience(self, experience):

        for e in experience:
            self.memory.add(*e)
            print("add", e)

    def save(self, path):
        torch.save(self.policy.state_dict(), osp.join(path, "policy.pt"))
        torch.save(self.critic_1.state_dict(), osp.join(path, "critic_1.pt"))
        torch.save(self.critic_2.state_dict(), osp.join(path, "critic_2.pt"))
        torch.save(self.target_critic_1.state_dict(), osp.join(path, "target_critic_1.pt"))
        torch.save(self.target_critic_2.state_dict(), osp.join(path, "target_critic_2.pt"))
        
        torch.save(self.optimizer_policy.state_dict(), osp.join(path, "optimizer_policy.pt"))
        torch.save(self.optimizer_critic_1.state_dict(), osp.join(path, "optimizer_critic_1.pt"))
        torch.save(self.optimizer_critic_2.state_dict(), osp.join(path, "optimizer_critic_2.pt"))

    def load(self, path, train_mode=True):
                
        self.policy.load_state_dict(torch.load(osp.join(path, "policy.pt")))
        self.critic_1.load_state_dict(torch.load(osp.join(path, "critic_1.pt")))
        self.critic_2.load_state_dict(torch.load(osp.join(path, "critic_2.pt")))
        self.target_critic_1.load_state_dict(torch.load(osp.join(path, "target_critic_1.pt")))
        self.target_critic_2.load_state_dict(torch.load(osp.join(path, "target_critic_2.pt")))

        self.optimizer_policy.load_state_dict(torch.load(osp.join(path, "optimizer_policy.pt")))
        self.optimizer_critic_1.load_state_dict(torch.load(osp.join(path, "optimizer_critic_1.pt")))
        self.optimizer_critic_2.load_state_dict(torch.load(osp.join(path, "optimizer_critic_2.pt")))
        
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

        states = torch.FloatTensor(states).to(device)

        mean, log_std = self.policy(states)

        mean = mean.detach().cpu().numpy()

        if deterministic:
            action = mean
        else:
            std = log_std.exp()
            std = std.detach().cpu().numpy()

            action = []

            for mean_, std_ in zip(mean, std):
                action_ = np.random.multivariate_normal(mean_, np.diag(std_))
                action_ = np.tanh(action_)
                action.append(action_)

            action = np.stack(action)

        return action

    def random_action(self, states):

        action_dim = (len(states),) + tuple(self.action_dim)

        print(action_dim)

        action = np.random.rand(*action_dim) * 2 - 1

        return action

    def set_target_entropy(self, target_e):
        self.target_entropy = target_e
        return

if __name__ == '__main__':
    import os
    import pprint
    from pathlib import Path
    config = {
          "soft_q_lr": 0.0003,
          "policy_lr": 0.0003,
          "alpha": 1,
          "alpha_lr": 0.0003,
          "batch_size": 128,
          "weight_decay": 1e-3,
          "gamma": 0.99,
          "memory_size": 100000,
          "tau": 0.01,
          "backup_interval": 100,
          "hidden_dim": 512,
          "state_dim": 3,
          "action_dim": 1,
          "seed": 192,
          "tensorboard_histogram_interval": 100,
        "auto_entropy": True,
          "root_path": Path(__file__).parent.resolve().as_posix().replace(' ', '\ ') + '/test_0/',
          "run_name": 'test_0'
    }

    os.system('mkdir -p {}'.format(config['root_path']))

    pprint.pprint(config)

    agent = AgentSAC(config, state_dim=(5,), action_dim=(5,))

    agent.save(".")
    agent.load(".")

    exit()

    """
    Use this function to test your agents on the pendulum v0 open ai gym. Continuous space
    """
    def test_agent():
        import math
        import random

        import gym
        import numpy as np

        from IPython.display import clear_output
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from IPython.display import display

        import argparse
        import time

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


        max_steps = 150
        max_episodes = 1000
        frame_idx = 0

        explore_steps = 200
        rewards = []

        def plot(rewards):
            clear_output(True)
            plt.figure(figsize=(20, 5))
            plt.plot(rewards)
            plt.savefig('sac_v2.png')
            # plt.show()

        env = NormalizedActions(gym.make("Pendulum-v0"))
        #env = gym.make("Pendulum-v0")
        action_space = env.action_space
        state_space = env.observation_space
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_range = 1.
        print('env action space: ', env.action_space, "sample:",env.action_space.sample(),  "shape:", env.action_space.shape)
        print('env state space: ', env.observation_space, "sample:",env.observation_space.sample(), "shape:", env.observation_space.shape)

        ag = AgentSAC(config, state_dim=state_dim, action_dim=action_dim)
        ag.batch_size = 256
        target_ent = -1*env.action_space.shape[0]
        print('target_entropy:', target_ent)
        ag.set_target_entropy(target_ent)


        for eps in range(max_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                if frame_idx > explore_steps:
                    ag.random = False
                    action = ag.act(state)
                else:
                    action = ag.sample_action()

                next_state, reward, done, _ = env.step(action)
                #next_state = [item for sublist in next_state for item in sublist]
                env.render()
                #print('a:', action, 's:', state)
                ag.step(state, action, reward, done, next_state, tensorboard=None)

                state = next_state
                episode_reward += reward
                frame_idx += 1

                if done:
                    break

            if eps % 20 == 0 and eps>0:
                plot(rewards)
            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
            rewards.append(episode_reward)

    test_agent()

