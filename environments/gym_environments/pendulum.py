import gym
from gym import spaces
import numpy as np
from utils import unwind_dict_values

class NormalizedEnv(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space
        self.action_space_low = self.action_space.low
        self.action_space_high = self.action_space.high

        self.action_space.low = np.array([-1])
        self.action_space.high = np.array([1])

    def action(self, action):
        action = self.action_space_low + (action + 1.0) * 0.5 * (self.action_space_high - self.action_space_low)
        action = np.clip(action, self.action_space_low, self.action_space_high)

        return action

class Pendulum:

    def __init__(self, max_steps=200, render=False, **kwargs):

        self.env = NormalizedEnv(gym.make("Pendulum-v0"))
        # self.env = gym.make("Pendulum-v0")
        self.step_counter = 0
        self.max_steps = max_steps

        self.render = render

        self.goal = np.array([0, 0])

        self.action_space = self.env.action_space

        self.observation_space = spaces.Dict({
            'pendulum': self.env.observation_space,
            'goal': spaces.Box(-1, 1, shape=(2,))
        })

    def reward_function(self, done, goal, action, **kwargs):
        if self.success_criterion(goal):
            reward = 1.
        elif done:
            reward = -1.
        else:
            theta = goal["achieved"][0] - goal["desired"][0]
            theta_dt = goal["achieved"][1] - goal["desired"][1]

            reward = -(theta ** 2 + 0.1 * theta_dt ** 2 + 0.001 * action[0] ** 2)
            # reward /= 16.2736044

        return reward

    @staticmethod
    def success_criterion(goal):
        goal_achieved = unwind_dict_values(goal["achieved"])
        goal_desired = unwind_dict_values(goal["desired"])

        goal_distance = np.abs(goal_achieved - goal_desired)

        return np.all(goal_distance < [0.088, 0.002])

    def step(self, action):
        state, reward, done, _ = self.env.step(action)

        self.step_counter += 1

        done |= self.step_counter >= self.max_steps

        theta = np.arctan2(state[1], state[0])
        theta_dt = state[2]

        goal = {
            'achieved': np.array([theta, theta_dt]),
            'desired': self.goal
        }

        if self.render:
            self.env.render()

        return state, goal, done

    def reset(self, desired_state=None):

        state = self.env.reset()

        self.step_counter = 0
        self.episode_reward = 0

        theta = np.arctan2(state[1], state[0])
        theta_dt = state[2]

        goal = {
            'achieved': np.array([theta, theta_dt]),
            'desired': self.goal
        }

        return state, goal

    def render(self, mode='human'):
        ...

if __name__ == "__main__":

    env = Pendulum(max_steps=1000, render=True)

    while True:
        state = env.reset()
        done = False

        while not done:
            action = np.ones_like(env.action_space.sample()) * .0
            next_state, goal, done = env.step(action)

            done |= env.success_criterion(goal)

            print(env.reward_function(state=state, action=action,
                                             next_state=next_state, done=done,
                                             goal=goal))

            state = next_state


