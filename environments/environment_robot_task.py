from environments.robots.panda import Panda
from environments.tasks.reach import Reach
import pybullet as p
import pybullet_data as pd
import gym
from gym import spaces
import numpy as np
import pybullet_utils.bullet_client as bc


class Environment(gym.Env):

    def __init__(self, task_config, robot_config, render=False,
                 bullet_client=None):

        self.render = render

        self.task_config = task_config
        self.robot_config = robot_config

        if bullet_client is None:
            connection_mode = p.GUI if render else p.DIRECT

            bullet_client = bc.BulletClient(connection_mode)

            bullet_client.setAdditionalSearchPath(pd.getDataPath())

            time_step = 1. / 60.
            bullet_client.setTimeStep(time_step)
            bullet_client.setRealTimeSimulation(0)

        self.bullet_client = bullet_client

        # planeId = self.bullet_client.loadURDF("plane.urdf")

        self.task = self.make_task(task_config, self.bullet_client)

        self.robot = self.make_robot(robot_config, self.bullet_client)

        self.action_space = self.robot.action_space

        self.observation_space_task = self.task.observation_space
        self.observation_space_robot = self.robot.observation_space

        shape_observation_space = tuple(
            np.array(self.observation_space_task.shape) + np.array(
                self.observation_space_robot.shape))

        self.observation_space = spaces.Box(-1, 1,
                                            shape=shape_observation_space)

    def make_task(self, task_config, bullet_client):

        task_name = task_config.pop("name")

        print(task_config)

        if task_name == 'reach':
            task = Reach(bullet_client, **task_config)
        else:
            raise ValueError()

        return task

    def make_robot(self, robot_config, bullet_client):
        robot_name = robot_config.pop("name")

        if robot_name == 'pandas':
            robot = Panda(bullet_client, **robot_config)
        else:
            raise ValueError()

        return robot

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

        done, reward, goal_reached = self.task.calculate_reward(self.robot,
                                                                success)

        observation = np.concatenate((observation_robot, observation_task))

        info = {
            "goal_reached": goal_reached
        }

        return observation, reward, done, info

    def test_compatability(self):
        # todo check if task can be completed with robot (dimensionalities)
        ...


if __name__ == "__main__":

    task = Reach
    robot = Panda

    env_kwargs1 = {
        "render": False,
        "task_config": {"name": "reach",
                        "dof": 3,
                        "only_positive": False
                        },
        "robot_config": {
            "name": "pandas",
            "dof": 3
        }
    }

    env_kwargs2 = {
            "render": True,
            "task_config": {"name": "reach",
                            "dof": 3,
                            "only_positive": False
                            },
            "robot_config": {
                "name": "pandas",
                "dof": 3
            }
        }

    env1 = Environment(**env_kwargs1)
    # env2 = Environment(**env_kwargs2)

    while True:

        done = False

        obs = env1.reset()

        step = 0

        while not done:
            action1 = env1.action_space.sample()
            obs, reward, done, info = env1.step(action1)

            step += 1

        print(info)
