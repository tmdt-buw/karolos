import gym
import numpy as np
from gym import spaces

class GymWrapper:

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

        self.observation_space = spaces.Dict({
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


if __name__ == "__main__":

    env = GymWrapper(name="Pendulum-v0", max_steps=1000, render=True)

    while True:
        state, goal_info = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()  # * .0
            next_state, goal_info, done = env.step(action)

            done |= env.success_criterion(goal_info)

            # if done:
            print(env.reward_function(goal_info))

            state = next_state
