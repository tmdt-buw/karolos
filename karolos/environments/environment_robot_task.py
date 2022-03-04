import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
from gym import spaces

sys.path.append(str(Path(__file__).resolve().parent))

from environment import Environment
from environments_robot_task.robots import get_robot
from environments_robot_task.tasks import get_task


class EnvironmentRobotTask(Environment):

    def __init__(self, task_config, robot_config, render=False,
                 bullet_client=None, **kwargs):
        self.render = render

        self.task_config = task_config
        self.robot_config = robot_config

        if bullet_client is None:
            connection_mode = p.GUI if render else p.DIRECT

            bullet_client = bc.BulletClient(connection_mode)

            bullet_client.setAdditionalSearchPath(pd.getDataPath())

            time_step = 1. / 300.
            bullet_client.setTimeStep(time_step)
            bullet_client.setRealTimeSimulation(0)

        bullet_client.loadURDF("plane.urdf")

        self.bullet_client = bullet_client

        self.task = get_task(task_config, self.bullet_client)

        self.robot = get_robot(robot_config, self.bullet_client)

        self.action_space = self.robot.action_space

        self.state_space = spaces.Dict({
            'robot': self.robot.state_space,
            'task': self.task.state_space,
        })

        self.reward_function = self.task.reward_function
        self.success_criterion = self.task.success_criterion

    def __exit__(self):
        del self.robot
        del self.task

    def reset(self, desired_state=None):
        """
        Reset the environment and return new state
        """

        try:
            if desired_state is not None:
                state_robot = self.robot.reset(desired_state.get("robot"))
                state_task, goal_info = self.task.reset(self.robot, state_robot, desired_state.get("task"))
            else:
                state_robot = self.robot.reset()
                state_task, goal_info = self.task.reset(self.robot, state_robot)

        except AssertionError as e:
            return e

        state = {
            'robot': state_robot,
            'task': state_task
        }

        return state, goal_info

    def step(self, action):
        state_robot = self.robot.step(action)
        state_task, goal_info, done = self.task.step(state_robot, self.robot)

        state = {
            'robot': state_robot,
            'task': state_task
        }

        return state, goal_info, done
