"""
https://spinningup.openai.com/en/latest/algorithms/sac.html

"""

import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn

try:
    from . import Agent
    from .utils.nn import NeuralNetwork, init_xavier_uniform
except:
    from karolos.agents import Agent
    from karolos.agents.utils.nn import NeuralNetwork, init_xavier_uniform


class Policy(NeuralNetwork):
    def __init__(self, state_dims, action_dim, network_structure):

        in_dim = int(
            np.sum([np.product(state_dim) for state_dim in state_dims]))

        out_dim = int(np.product(action_dim))

        super(Policy, self).__init__(in_dim, network_structure)

        dummy = super(Policy, self).forward(torch.zeros((1, in_dim)))

        self.operators.append(nn.Linear(dummy.shape[1], out_dim))

        self.operators.apply(init_xavier_uniform)

    def forward(self, *state_args):
        action = super(Policy, self).forward(*state_args)

        return action


class AgentDQN(Agent):
    def __init__(self, config, observation_space, action_space,
                 reward_function, experiment_dir=None):

        super(AgentDQN, self).__init__(config, observation_space, action_space,
                                       reward_function, experiment_dir)

        self.learning_rate_policy = config.get("learning_rate", 5e-4)
        self.weight_decay = config.get("weight_decay", 1e-4)
        self.tau = config.get('tau', 2.5e-3)

        exploration_probability_config = config.get("exploration_probability", {})

        exploration_probability_start = exploration_probability_config.get("start", 1)
        exploration_probability_end = exploration_probability_config.get("end", 0)
        exploration_probability_steps = exploration_probability_config.get("steps", np.inf)

        self.exploration_probability = lambda step: max(1 - step / exploration_probability_steps, 0) * (
                exploration_probability_start - exploration_probability_end) + exploration_probability_end

        self.policy_structure = config.get('policy_structure', [])

        # generate networks
        self.q_network = self.policy = Policy(self.state_dim, self.action_dim, self.policy_structure).to(self.device)
        self.target_q_network = self.policy = Policy(self.state_dim, self.action_dim, self.policy_structure).to(self.device)

        self.optimizer = torch.optim.AdamW(self.q_network.parameters(),
                                           lr=self.learning_rate_policy,
                                           weight_decay=self.weight_decay)

        self.update_target(self.q_network, self.target_q_network, 1.)

        self.criterion = nn.MSELoss()

    def learn(self):

        self.q_network.train()

        experiences, indices = self.memory.sample(self.batch_size)

        states, actions, rewards, next_states, dones = experiences

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(np.float32(dones)).to(self.device)

        rewards *= self.reward_scale

        action_values = self.q_network(states)
        action_values = action_values[torch.arange(len(action_values)), actions]

        next_action_values = self.target_q_network(next_states)
        next_action_values_max, _ = next_action_values.max(-1)

        target_q_values = rewards + (1 - dones) * self.reward_discount * next_action_values_max

        loss = self.criterion(action_values, target_q_values)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.update_priorities(indices, action_values, target_q_values)

        # Update target
        self.update_target(self.q_network, self.target_q_network, self.tau)

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

        torch.save(self.q_network.state_dict(), osp.join(path, "q_network.pt"))
        torch.save(self.target_q_network.state_dict(), osp.join(path, "target_q_network.pt"))
        torch.save(self.optimizer.state_dict(), osp.join(path, "optimizer.pt"))

    def load(self, path):
        self.q_network.load_state_dict(torch.load(osp.join(path, "q_network.pt")))
        self.target_q_network.load_state_dict(torch.load(osp.join(path, "target_q_network.pt")))
        self.optimizer.load_state_dict(torch.load(osp.join(path, "optimizer.pt")))

    def predict(self, states, deterministic=True):

        if not deterministic:
            mask_deterministic = np.random.random(len(states)) > self.exploration_probability(self.learning_step)
        else:
            mask_deterministic = np.ones(len(states))

        actions = np.random.randint(0, self.action_space.n, len(states))

        if mask_deterministic.any():
            # at least one action to be determined deterministically
            indices_deterministic = np.argwhere(mask_deterministic).flatten()

            states_deterministic = [states[idx] for idx in indices_deterministic]

            self.q_network.eval()

            states_deterministic = torch.FloatTensor([state for state in states_deterministic]).to(self.device)

            action_values_deterministic = self.q_network(states_deterministic)

            _, actions_deterministic = action_values_deterministic.max(-1)

            actions_deterministic = actions_deterministic.detach().cpu().numpy()

            actions[indices_deterministic] = actions_deterministic

        return actions

    def set_target_entropy(self, target_entropy):
        self.target_entropy = target_entropy


if __name__ == '__main__':
    from gym import spaces
    from .utils import unwind_space_shapes

    config = {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "reward_scale": 100,
        "batch_size": 16,
        "tau": 0.0025,

        "buffer": {"name": "priority", "buffer_size": 1e6},

        "policy_structure": [('linear', 64), ('relu', None)] * 3,

        "exploration_probability": {"start": .6,
                                    }

    }

    action_space = spaces.Discrete(6)

    observation_space = spaces.Dict({
        'state': spaces.Box(-1, 1, shape=(24,)),
    })


    def reward_function(**kwargs):
        return 0


    dqn = AgentDQN(config, observation_space, action_space, reward_function)

    state_spaces = unwind_space_shapes(observation_space)


    def dummy_state():
        state = {}
        goal_info = {}

        for space_name, space in observation_space.spaces.items():
            state[space_name] = space.sample()

        return state, goal_info


    def dummy_action():
        return action_space.sample()


    trajectory = []

    for _ in range(50):
        trajectory.append(dummy_state())
        trajectory.append(dummy_action())

    trajectory.append(dummy_state())

    dqn.add_experience_trajectory(trajectory)

    dqn.learn()

    states = [dummy_state()[0] for _ in range(10)]

    actions = dqn.predict(states, deterministic=False)

    print(len(actions))
    print(actions)
