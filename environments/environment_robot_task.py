from robots.panda import Panda
from tasks.reach import Reach
import pybullet as p
import pybullet_data as pd
import gym
from gym import spaces
import numpy as np
import pybullet_utils.bullet_client as bc
import time

class Environment(gym.Env):

    def __init__(self, task_cls, robot_cls, kwargs_task=None, kwargs_robot=None,
                 render=False, bullet_client=None):

        self.render = render
        self.task_cls = task_cls
        self.robot_cls = robot_cls

        if kwargs_task is None:
            kwargs_task = {}

        if kwargs_robot is None:
            kwargs_robot = {}

        if bullet_client is None:
            connection_mode = p.GUI if render else p.DIRECT

            bullet_client = bc.BulletClient(connection_mode)

            bullet_client.setAdditionalSearchPath(pd.getDataPath())

            time_step = 1. / 60.
            bullet_client.setTimeStep(time_step)
            bullet_client.setRealTimeSimulation(0)

        self.bullet_client = bullet_client

        planeId = self.bullet_client.loadURDF("plane.urdf")

        self.task = self.task_cls(self.bullet_client, **kwargs_task)
        self.robot = self.robot_cls(self.bullet_client, **kwargs_robot)

        self.action_space = self.robot.action_space

        self.observation_space_task = self.task.observation_space
        self.observation_space_robot = self.robot.observation_space

        shape_observation_space = tuple(
            np.array(self.observation_space_task.shape) + np.array(
                self.observation_space_robot.shape))

        self.observation_space = spaces.Box(-1, 1,
                                            shape=shape_observation_space)


    def reset(self):
        """Reset the environment and return new state
        """

        # print("env reset", self.bullet_client.getConnectionInfo())
        #
        # if not self.bullet_client.getConnectionInfo()["isConnected"]:
        #     self.reset_pybullet()
        #
        #     print("env reset pybullet", self.bullet_client.getConnectionInfo())



        # if not self.bullet_client.getConnectionInfo()["isConnected"]:
        #     self.bullet_client = bc.BulletClient(p.DIRECT)

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

    import multiprocessing

    task = Reach
    robot = Panda

    env_kwargs1 = {
        "task_cls": task,
        "robot_cls": robot,
        "render": True,
        "kwargs_task": {"dof": 1},
        "kwargs_robot": {"dof": 3}
    }

    env_kwargs2 = {
        "task_cls": task,
        "robot_cls": robot,
        "render": False,
        "kwargs_task": {"dof": 2},
        "kwargs_robot": {"dof": 3}
    }



    env1 = Environment(**env_kwargs1)
    env2 = Environment(**env_kwargs2)

    while True:

        done1 = False
        done2 = False

        obs1 = env1.reset()
        obs2 = env2.reset()

        step = 0

        while not done1 or not done2:
            action1 = env1.action_space.sample()
            obs1, reward1, done1, info1 = env1.step(action1)

            action2 = env2.action_space.sample()
            obs2, reward2, done2, info2 = env2.step(action2)

            step += 1