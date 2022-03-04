import sys
from pathlib import Path

import gym
import numpy as np
from gym import spaces

sys.path.append(str(Path(__file__).resolve().parent))
from environment import Environment


class GymWrapper(Environment):

    def __init__(self, name, max_steps=200, reward_success_threshold=np.inf, step_success_threshold=np.inf,
                 render=False, **kwargs):
        self.env = gym.make(name)
        self.step_counter = 0
        self.max_steps = max_steps

        self.render = render

        self.goal_desired = {
            'reward': reward_success_threshold,
            'step': step_success_threshold,
        }

        self.action_space = self.env.action_space

        self.state_space = spaces.Dict({
            'state': self.env.observation_space,
        })

    def reward_function(self, goal_info, **kwargs):
        reward = goal_info["achieved"]["reward"]

        return reward

    @staticmethod
    def success_criterion(goal_info):
        reward_achieved = goal_info["achieved"]["reward"] >= goal_info["desired"]["reward"]
        steps_achieved = goal_info["achieved"]["step"] >= goal_info["desired"]["step"]

        return reward_achieved or steps_achieved

    def step(self, action):
        state, reward, done, _ = self.env.step(action)

        self.step_counter += 1

        done |= self.step_counter >= self.max_steps

        self.__render()

        return *self.get_status(state, reward), done

    def get_status(self, state, reward=None):
        state = {
            "state": state
        }

        goal_info = {
            'achieved': {
                'reward': reward,
                'step': self.step_counter
            },
            'desired': self.goal_desired
        }

        return state, goal_info

    def reset(self, desired_state=None):
        self.step_counter = 0
        self.episode_reward = 0

        state = self.env.reset()

        return self.get_status(state)

    def __render(self):
        if self.render:
            self.env.render()
