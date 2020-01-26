from robots.panda import Panda
from tasks.reach import Reach
import pybullet as p
import pybullet_data as pd
import gym
from gym import spaces
import numpy as np


class Environment(gym.Env):

    def __init__(self, bullet_client, task, robot, kwargs_task=None, kwargs_robot=None,
                 render=False):

        if kwargs_task is None:
            kwargs_task = {}

        if kwargs_robot is None:
            kwargs_robot = {}

        if render:
            bullet_client.connect(p.GUI)
        else:
            bullet_client.connect(p.DIRECT)

        bullet_client.setAdditionalSearchPath(pd.getDataPath())

        time_step = 1. / 60.
        bullet_client.setTimeStep(time_step)
        bullet_client.setRealTimeSimulation(0)

        self.task = task(p, **kwargs_task)
        self.robot = robot(p, **kwargs_robot)

        self.action_space = self.robot.action_space

        self.observation_space_task = self.task.observation_space
        self.observation_space_robot = self.robot.observation_space

        shape_observation_space = tuple(
            np.array(self.observation_space_task.shape) + np.array(
                self.observation_space_robot.shape))

        self.observation_space = spaces.Box(-1, 1,
                                            shape=shape_observation_space)

        self.reset()

    # def get_observation(self):
    #
    #     observation_task = self.task.get_observation()
    #     observation_robot = self.robot.get_observation()
    #
    #     # todo scale with respective observation space
    #     if observation_task is None or observation_robot is None:
    #         return None
    #
    #     observation = np.concatenate((observation_task, observation_robot))
    #
    #     assert self.observation_space.contains(observation)
    #
    #     return observation

    def reset(self):
        """Reset the environment and return new state
        """

        observation_robot = self.robot.reset()
        observation_task = self.task.reset()

        observation = np.concatenate((observation_robot, observation_task))

        assert self.observation_space.contains(observation)

        return observation

    def render(self, mode='human'):
        ...

    def step(self, action):
        success_robot, observation_robot = self.robot.step(action)
        success_task, observation_task = self.task.step()

        success = success_robot and success_task

        done, reward = self.task.calculate_reward(self.robot, success)

        observation = np.concatenate((observation_robot, observation_task))

        info = {}

        return observation, reward, done, info

    def test_compatability(self):
        # todo check if task can be completed with robot (dimensionalities)
        ...


if __name__ == "__main__":

    task = Reach
    robot = Panda

    env_kwargs = {
        "task": task,
        "robot": robot,
        "render": True,
        "kwargs_task": {"dof": 2},
        "kwargs_robot": {"dof": 3}
    }

    env = Environment(p, **env_kwargs)

    while True:

        done = False

        obs = env.reset()

        step = 0

        while not done:
            action = env.action_space.sample()

            obs, reward, done, info = env.step(action)

            print(step, reward, done)

            step += 1