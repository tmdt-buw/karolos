import gym
import numpy as np
import torch
from gym import spaces

from agents.sac import AgentSAC, Policy, Critic


def test_policy():
    policy_structure = [('linear', 64), ('relu', None), ('dropout', 0.2),
                        ('linear', 32)]

    policy = Policy([[100]], [10], policy_structure)
    dummy_mean, dummy_std = policy(torch.zeros((1, 100)))
    assert dummy_mean.shape == (1, 10)
    assert dummy_std.shape == (1, 1)

    dummy_mean, dummy_std = policy(torch.zeros((100, 100)))
    assert dummy_mean.shape == (100, 10)
    assert dummy_std.shape == (100, 1)

    # test multiple state components
    policy = Policy([[100], [50]], [10], policy_structure)
    dummy_mean, dummy_std = policy(torch.zeros((1, 100)), torch.zeros((1, 50)))
    assert dummy_mean.shape == (1, 10)
    assert dummy_std.shape == (1, 1)

    dummy_mean, dummy_std = policy(torch.zeros((100, 100)),
                                   torch.zeros((100, 50)))
    assert dummy_mean.shape == (100, 10)
    assert dummy_std.shape == (100, 1)


def test_critic():
    critic_structure = [('linear', 64), ('relu', None), ('dropout', 0.2),
                        ('linear', 32)]

    critic = Critic([[100]], [10], critic_structure)
    dummy = critic(torch.zeros((1, 100)), torch.zeros((1, 10)))
    assert dummy.shape == (1, 1)

    dummy = critic(torch.zeros((100, 100)), torch.zeros((100, 10)))
    assert dummy.shape == (100, 1)

    # test multiple state components
    critic = Critic([[100], [50]], [10], critic_structure)
    dummy = critic(torch.zeros((1, 100)), torch.zeros((1, 50)),
                   torch.zeros((1, 10)))
    assert dummy.shape == (1, 1)

    dummy = critic(torch.zeros((100, 100)), torch.zeros((100, 50)),
                   torch.zeros((100, 10)))
    assert dummy.shape == (100, 1)


def test_sac_pendulum():
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

    env = NormalizedEnv(gym.make("Pendulum-v0"))

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

    results_episodes = []
    rewards_episodes = []

    for eps in range(episodes):
        state = env.reset()
        episode_reward = 0

        goal_reached = False

        for step in range(max_steps):

            action = agent.predict(np.expand_dims(state, 0),
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
            agent.learn(eps * max_steps + step)

            state = next_state

            if done:
                break

        results_episodes.append(goal_reached)
        rewards_episodes.append(episode_reward)

        if len(results_episodes) > 5 and np.all(results_episodes[-5:]):
            return

    for eps in range(episodes):
        state = env.reset()
        episode_reward = 0

        goal_reached = False

        for step in range(max_steps):

            action = agent.predict(np.expand_dims(state, 0),
                                   np.expand_dims(goal, 0),
                                   deterministic=True)[0]

            next_state, reward, _, _ = env.step(action)

            goal_reached = next_state[0] > .98 and next_state[2] < 1e-5
            done = goal_reached

            episode_reward += reward

            state = next_state

            if done:
                break

        results_episodes.append(goal_reached)
        rewards_episodes.append(episode_reward)

        if np.all(results_episodes[-5:]):
            return

    raise AssertionError(f"Agent was not successful. {results_episodes}")
