from environments.robots import get_robot
from environments.tasks import get_task
import pybullet as p
import pybullet_data as pd
import gym
from gym import spaces
import numpy as np
import pybullet_utils.bullet_client as bc


class Environment(gym.Env):

    def __init__(self, task_config, robot_config, render=False,
                 bullet_client=None, **kwargs):
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

        self.task = get_task(task_config, self.bullet_client)

        self.robot = get_robot(robot_config, self.bullet_client)

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

        info = {
            "success": {
                "robot": success_robot,
                "task": success_task
            }
        }

        success = np.all(info["success"].values())

        achieved_goal, desired_goal = self.task.get_goals(self.robot, success)

        info["achieved_goal"] = achieved_goal
        info["desired_goal"] = desired_goal

        reward = self.task.compute_reward(achieved_goal, desired_goal)
        done, goal_reached = self.task.compute_done(achieved_goal,
                                                    desired_goal)

        info["goal_reached"] = goal_reached

        if success:
            # todo make generic by merging task observation and achived goal
            her_goal = achieved_goal
            her_done, _ = self.task.compute_done(achieved_goal,
                                                          her_goal)

            info["her"] = {"observation": np.concatenate((observation_robot,
                                                          her_goal)),
                           "reward": self.task.compute_reward(achieved_goal,
                                                              her_goal),
                           "done": her_goal
                           }

        observation = np.concatenate((observation_robot, observation_task))

        return observation, reward, done, info


if __name__ == "__main__":

    env_kwargs1 = {
        "render": False,
        "task_config": {"name": "push",
                        "dof": 3,
                        "only_positive": False
                        },
        "robot_config": {
            "name": "pandas",
            "dof": 3
        }
    }

    env1 = Environment(**env_kwargs1)

    while True:

        done = False

        obs = env1.reset()

        step = 0

        while not done:
            action1 = env1.action_space.sample()
            obs, reward, done, info = env1.step(action1)

            step += 1

        print(info)
