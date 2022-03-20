import sys
from pathlib import Path

import gym
import numpy as np
from gym import spaces

sys.path.append(str(Path(__file__).resolve().parent))
from environment import Environment


class GymWrapper(Environment):
    class ActionWrapper(gym.ActionWrapper):
        def __init__(self, env):
            self.action_space = env.action_space
            super().__init__(env)

        def action(self, act):
            # modify act
            act = (act + 1) / 2 # [-1,1] -> [0,1]
            act *= self.action_space.high - self.action_space.low # [-1,1] -> [0,h-l]
            act += self.action_space.low # [0,h-l] -> [l,h]

            return act

    def __init__(self, name, max_steps=200, reward_success_threshold=np.inf, step_success_threshold=np.inf,
                 render=False, **kwargs):
        self.env = self.ActionWrapper(gym.make(name))
        self.step_counter = 0
        self.max_steps = max_steps

        self.render = render

        self.reward_success_threshold = reward_success_threshold
        self.step_success_threshold = step_success_threshold

        self.state_space = spaces.Dict({
            'state': self.env.observation_space,
        })

        self.goal_space = spaces.Dict({})
        self.action_space = self.env.action_space

    def reward_function(self, info, **kwargs):
        reward = info["achieved"]["reward"]

        # if self.success_criterion(info):
        #     reward = 1
        # elif done:
        #     reward = -1
        # else:
        #     reward = info["achieved"]["reward"]
        return reward

    @staticmethod
    def success_criterion(info, **kwargs):
        reward_achieved = info["achieved"]["reward"] >= info["desired"]["reward"]
        steps_achieved = info["achieved"]["step"] >= info["desired"]["step"]

        return reward_achieved or steps_achieved

    def reset(self, desired_state=None):
        self.step_counter = 0
        self.episode_reward = 0

        state = self.env.reset()

        state, goal, done, info = self.get_status(state)

        return state, goal, info

    def step(self, action):
        state, reward, done, _ = self.env.step(action)

        self.step_counter += 1

        self.__render()

        return self.get_status(state, reward, done)

    def get_status(self, state, reward=None, done=False):
        state = {
            "state": state
        }

        goal = {}

        done |= self.step_counter >= self.max_steps

        info = {
            "achieved": {
                "reward": reward,
                "step": self.step_counter
            },
            "desired": {
                "reward": self.reward_success_threshold,
                "step": self.step_success_threshold
            }
        }

        return state, goal, done, info

    def __render(self):
        if self.render:
            self.env.render()
